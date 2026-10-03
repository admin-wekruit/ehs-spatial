"""Photo-linked object catalog in the current reconstruction's native frame.

CPU-only: consumes saved geometry, SAM masks and meshes; never runs a provider.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehs_spatial.measurements import measure_observed_points
from ehs_spatial.platform.reconstruction import _mesh
from ehs_spatial.platform.spatial import primitive_mesh

EXTRA_WORDS = ('light curtain', 'work platform', 'control cabinet', 'signal light',
               'stack light', 'warning sign', 'workcell sign', 'folding safety barrier',
               'cable tray', 'instruction poster', 'transparent safety panel', 'floor marking')


def button_basis(geometry):
    anchor = geometry['anchor']
    normal = np.asarray(anchor['normal'], float); normal /= np.linalg.norm(normal)
    up = np.asarray(geometry['floor']['normal'], float); up /= np.linalg.norm(up)
    upright = up-normal*np.dot(up, normal); upright /= np.linalg.norm(upright)
    return np.column_stack((np.cross(upright, normal), upright, normal))


def button_meshes(geometry):
    """Shared component construction; known part sizes never alter observed bounds."""
    anchor = geometry['anchor']
    fit = anchor.get('referenceFit', {})
    if fit.get('status') == 'available':
        if fit.get('fittedNuisanceParameters') is None:
            raise ValueError('Accepted reference fit is missing its 3D geometry')
        if not fit.get('camerasFixed'):
            raise ValueError('Cannot place a refitted-camera button into the original scene')
        from scripts.workcell_metrology_models import reference_meshes
        names = {'gray-housing': 'gray-base', 'yellow-body': 'yellow-body', 'red-actuator': 'red-cap'}
        return {'emergency-button-'+names[name]: mesh for name, mesh in reference_meshes(fit['fittedNuisanceParameters']).items()}
    # Rejected calibration candidates belong in their diagnostic export. Keep
    # the existing image-supported display envelope independent of that fit.
    center = np.asarray(anchor['centerNative'], float)
    basis = button_basis(geometry)
    width, height = float(anchor['nativeWidth']), float(anchor['nativeHeight'])
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError('Invalid button envelope')
    meshes = {}
    for name, color, fraction, y, radius in [('gray-base', [105,110,115,255], .32, -.34, None),
                                            ('yellow-body', [245,196,23,255], .43, .035, None),
                                            ('red-cap', [207,32,33,255], .25, .375, .43)]:
        # ponytail: retain the existing three-part height partition; measured part heights would replace these fractions.
        if radius is not None:
            radius *= width
        spec = {'kind':'box', 'dimensions':[width*(.8 if name=='gray-base' else 1), height*fraction, width*.45]} if radius is None else {
            'kind':'cylinder', 'radius':radius, 'height':height*fraction, 'segments':32}
        primitive = primitive_mesh(spec)
        mesh = trimesh.Trimesh(primitive.vertices, primitive.faces, process=False)
        if radius is not None:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0]))
        mesh.vertices = (mesh.vertices + [0, height*y, 0]) @ basis.T + center
        mesh.visual.vertex_colors = color
        meshes['emergency-button-'+name] = mesh
    return meshes


def _project(points, frame):
    local = (np.asarray(points) - frame['pose'][:3, 3]) @ frame['pose'][:3, :3]
    uv = local @ frame['K'].T
    return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-12), local[:, 2]


def _observation(mask, photo, source):
    yy, xx = np.where(mask)
    if not len(xx):
        return None
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for contour in contours:
        polygon = cv2.approxPolyDP(contour, 0, True).reshape(-1, 2)
        if len(polygon) >= 3 and cv2.contourArea(polygon) > 0:
            polygons.append(polygon.tolist())
        else:
            # Thin fragments retain their pixel footprints; a bounding box could
            # fill empty space between diagonal pixels or disconnected pieces.
            for x, y in np.unique(contour.reshape(-1, 2), axis=0).tolist():
                polygons.append([[x, y], [x+1, y], [x+1, y+1], [x, y+1]])
    box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    return {'photo': photo, 'box': box, 'polygons': polygons, 'source': source,
            'supportedPixels': int(mask.sum())}


SCALE_WARNING = 'Absolute scale is conditional on the provisional button dimension hypothesis.'


def decode_mask(frame, encoded):
    """One SAM RLE mask on the frame's canonical grid by exact nearest sampling.

    An OpenCV nearest warp rounds through fixed point and differs between OpenCV
    builds by a few boundary pixels (the 7.7x pixel-centre map hits half-pixel ties
    every seventh column); visible extents are min/max over the support, so one
    boundary pixel on background depth can move them far. floor(x + 0.5) of the
    inverse map in elementwise IEEE float64 (no LAPACK solve) is deterministic in every
    environment; at an exact tie float rounding picks one of the two equally near pixels.
    """
    item = json.loads(encoded)
    counts = np.asarray(item['counts'], np.int64)
    if counts.sum() != np.prod(item['size']):
        raise ValueError('SAM 3 mask RLE length mismatch')
    raw = np.repeat(np.arange(len(counts), dtype=np.uint8) % 2, counts).reshape(item['size'], order='F').astype(bool)
    shape = tuple(frame['image']['shape'][:2])
    if raw.shape == shape:
        return raw
    affine = np.asarray(frame['input_mask_transform']['input_to_canonical_pixel_centres'], float)
    if not np.array_equal(affine[2], [0, 0, 1]):
        raise ValueError('Raw-to-canonical map must be affine')
    (a, b, c), (d, e, f) = affine[:2]
    yy, xx = np.indices(shape)
    x, y, det = xx - c, yy - f, a * e - b * d
    column = np.floor((e * x - b * y) / det + .5).astype(np.int64)
    row = np.floor((a * y - d * x) / det + .5).astype(np.int64)
    inside = (column >= 0) & (column < raw.shape[1]) & (row >= 0) & (row < raw.shape[0])
    mask = np.zeros(shape, bool)
    mask[inside] = raw[row[inside], column[inside]]
    return mask


def measure_support(frame, mask, floor, photo, source):
    """Visible extents of one support mask: its valid frame points against ``floor``."""
    supported = mask & frame['valid']
    measured = measure_observed_points(
        frame['points'][supported],
        {'floor_plane': [*floor['normal'], floor['offset']], 'warnings': [SCALE_WARNING]},
        mask_pixels=int(mask.sum()), source={'photo': photo, 'evidence': source})
    return measured, int(supported.sum())


def support_mask(observation, frame, geometry, segmentation, inputs, plane=None):
    """Rebuild one saved observation's support with the rule build() used, on ``frame``.

    The observation's source names the rule: a SAM instance mask (guard-board
    instances also inside guard-input.npz), the union of robot instances, the
    selected cart mask, the safety-fence union inside its fitted plane, the
    floor band, or the anchor envelope. Saved outline polygons are never used:
    they fill see-through holes with background points. ``plane`` is the fence
    item's geometryPlaneIndex: a refreshed catalog keeps the source string when
    planes are renumbered, so the item, not the string, names the plane.
    """
    import re
    from scripts.workcell_photo_oneshot import GUARD_WORD
    photo, source = observation['photo'], observation['source']
    words = [prompt['text'] for prompt in segmentation['prompts']]
    def rles(word):
        return segmentation['results'][photo - 1][words.index(word)].get('rle', [])
    def union(word):
        mask = np.zeros(frame['valid'].shape, bool)
        for encoded in rles(word):
            mask |= decode_mask(frame['raw'], encoded)
        return mask
    if match := re.fullmatch(r'SAM: (.+); instance (\d+)', source):
        mask = decode_mask(frame['raw'], rles(match[1])[int(match[2])])
        return mask & (inputs('guard-input.npz')[f'v{photo}_mask'] > 0) if match[1] == GUARD_WORD else mask
    if source == 'SAM industrial robot arm':
        return union('industrial robot arm')
    if source == 'Selected cart mask; cart-mask-selection.json':
        return inputs('cart-input.npz')[f'v{photo}_mask'] > 0
    floor = geometry['floor']
    if match := re.fullmatch(r'SAM safety fence intersected with fitted plane (\d+)', source):
        fitted = geometry['fence']['planes'][int(match[1]) if plane is None else plane]
        return union('safety fence') & (np.abs(frame['points'] @ np.asarray(fitted['normal']) + fitted['offset'])
                                         < max(floor['residualP95Native'] * 5, 1e-6))
    if source == 'Pointmap support near inferred floor plane':
        return frame['valid'] & (np.abs(frame['points'] @ np.asarray(floor['normal'], float) + float(floor['offset']))
                                 < floor['residualP95Native'])
    if source == 'geometry.anchor component envelope transformed from original photo':
        view = next(v for v in geometry['anchor']['views'] if v['photo'] == photo)
        x0, y0, x1, y1 = view['boxRaw']
        corners = np.asarray([[x0, y0, 1], [x1, y1, 1]]) @ np.asarray(frame['raw']['input_mask_transform']['input_to_canonical_pixel_centres']).T
        lo, hi = np.floor(corners[0, :2]).astype(int), np.ceil(corners[1, :2]).astype(int)
        mask = np.zeros(frame['valid'].shape, bool)
        lo, hi = np.maximum(lo, 0), np.minimum(hi, mask.shape[::-1])
        mask[lo[1]:hi[1], lo[0]:hi[0]] = True
        return mask
    raise ValueError('No objects-stage support rule for observation source: ' + source)


PANEL_CELLS = 100  # workcell_photo_geometry builds an observed fence panel only from >= 100 supported two-pixel cells
FENCE_REASON = ('Plane-filtered safety-fence support forms fewer than 100 supported two-pixel cells '
                '(the observed-panel minimum of workcell_photo_geometry); stray coplanar pixels are not evidence of the section')


def fence_panel_supported(supported):
    """True when a fence section's plane-filtered support could form an observed panel (geometry-stage rule)."""
    keep = supported[::2, ::2]
    return int((keep[:-1, :-1] & keep[1:, :-1] & keep[:-1, 1:] & keep[1:, 1:]).sum()) >= PANEL_CELLS


def _unsupported(ident, photo, source, supported):
    ys, xs = np.nonzero(supported)
    return {'objectId': ident, 'photo': photo, 'source': source, 'supportedPixels': int(supported.sum()),
            'box': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None,
            'reason': FENCE_REASON}


def review_fence_observations(root):
    """Apply build()'s fence-panel rule to a catalog written before it existed; rewrites objects.json.

    Only observations whose rebuilt support fails the rule are removed and listed in
    coverage.unsupportedObservations; every other observation stays byte-identical.
    Everything bound to the catalog (report, semantics, candidate fits) must be rebuilt after.
    """
    from scripts.workcell_photo_metrology import _fence_plane_index  # lazy: metrology imports this module
    from scripts.workcell_photo_oneshot import _array, _frame
    root = Path(root)
    text = (root / 'objects.json').read_text()
    catalog = json.loads(text)
    geometry = json.loads((root / 'geometry.json').read_text())
    segmentation = json.loads((root / 'sam3.json').read_text())
    dropped = []
    for item in catalog['objects']:
        if item['kind'] != 'safety fence':
            continue
        kept = []
        for observation in item['observations']:
            raw = _frame(root, observation['photo'])
            points = _array(raw['pts3d'])
            frame = {'raw': raw, 'points': points,
                     'valid': _array(raw['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2)}
            supported = support_mask(observation, frame, geometry, segmentation, None, _fence_plane_index(item)) & frame['valid']
            if fence_panel_supported(supported):
                kept.append(observation)
            else:
                dropped.append(_unsupported(item['id'], observation['photo'], observation['source'], supported))
        item['observations'] = kept
        if len(kept) <= 1 and item['measurements'].get('groundClearance', {}).get('status') == 'conditional-model-estimate':
            item['measurements']['groundClearance'] = _unknown('Visible geometry does not establish complete physical dimensions.')['groundClearance']
    if not dropped:
        return dropped  # nothing to remove: the catalog bytes, and every hash bound to them, stay as they are
    coverage = catalog['coverage']
    coverage['observations'] = sum(len(item['observations']) for item in catalog['objects'])
    coverage['unsupportedObservations'] = coverage.get('unsupportedObservations', []) + dropped
    ending = '\n' if text.endswith('\n') else ''
    (root / 'objects.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + ending)
    return dropped


def support_inputs_missing(root, objects):
    """Run files the support rules of these observations need but the directory lacks."""
    from scripts.workcell_photo_oneshot import GUARD_WORD
    needed = {'sam3.json'}
    for item in objects:
        for observation in item['observations']:
            needed.add(f"frame_{observation['photo']:04d}.json.gz")
            if observation['source'] == 'Selected cart mask; cart-mask-selection.json':
                needed.add('cart-input.npz')
            elif observation['source'].startswith(f'SAM: {GUARD_WORD};'):
                needed.add('guard-input.npz')
    return sorted(name for name in needed if not (Path(root) / name).is_file())


def remeasure(root, objects, geometry):
    """Re-measure every saved observation on this run's frames and floor, with its objects-stage support.

    Returns ({objectId: [(observedMeasurements, supportedPixels, maskPixels)]}, []) aligned with each
    object's observations, or (None, missing files) when the directory cannot rebuild the supports
    (synthetic runs, experiment uploads); the report then labels its extents as objects-stage values.
    """
    from scripts.workcell_photo_metrology import _fence_plane_index  # lazy: metrology imports this module
    from scripts.workcell_photo_oneshot import _array, _frame
    root = Path(root)
    missing = support_inputs_missing(root, objects)
    if missing:
        return None, missing
    segmentation = json.loads((root / 'sam3.json').read_text())
    frames, loaded = {}, {}
    def inputs(name):
        if name not in loaded:
            loaded[name] = np.load(root / name)
        return loaded[name]
    result = {}
    for item in objects:
        rows = []
        for observation in item['observations']:
            photo = observation['photo']
            if photo not in frames:
                raw = _frame(root, photo)
                points = _array(raw['pts3d'])
                frames[photo] = {'raw': raw, 'points': points,
                                 'valid': _array(raw['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2)}
            plane = _fence_plane_index(item) if item.get('kind') == 'safety fence' else None
            mask = support_mask(observation, frames[photo], geometry, segmentation, inputs, plane)
            measured, supported = measure_support(frames[photo], mask, geometry['floor'], photo, observation['source'])
            rows.append((measured, supported, int(mask.sum())))
        result[item['id']] = rows
    return result, []


def _unknown(reason):
    return {key: {'valueNative': None, 'status': 'unknown', 'source': reason}
            for key in ('height', 'width', 'depth', 'groundClearance')}


def _nodes(root, file):
    scene = trimesh.load(root / file, force='scene')
    return scene, list(scene.graph.nodes_geometry)


def _points(scene, nodes):
    return np.concatenate([trimesh.transform_points(scene.geometry[scene.graph[n][1]].vertices,
                                                  scene.graph[n][0]) for n in nodes])


def _inside(points, candidate, frames):
    """Fraction of one instance's 3D support reprojecting into another mask."""
    frame = frames[candidate['photo']]
    uv, depth = _project(points[::max(1, len(points)//400)], frame)
    xy = np.rint(uv).astype(int)
    h, w = candidate['mask'].shape
    valid = (depth > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    hit = np.zeros(len(xy), bool)
    hit[valid] = candidate['mask'][xy[valid, 1], xy[valid, 0]]
    return float(hit.mean()) if len(hit) else 0.


def _same(a, b, frames):
    # ponytail: pairwise matching is sufficient for this four-photo catalog;
    # use a spatial index if captures grow beyond a few hundred detections.
    if a['photo'] == b['photo']:
        intersection = (a['mask'] & b['mask']).sum()
        return intersection / max(1, (a['mask'] | b['mask']).sum()) > .7
    if not len(a['points']) or not len(b['points']):
        return False
    ca, cb = np.median(a['points'], axis=0), np.median(b['points'], axis=0)
    extent = min(np.linalg.norm(np.ptp(a['points'], axis=0)), np.linalg.norm(np.ptp(b['points'], axis=0)))
    return np.linalg.norm(ca-cb) < max(.03, extent*.6) and min(_inside(a['points'], b, frames), _inside(b['points'], a, frames)) >= .4


def _guard_parts(root, detections, frames, masks):
    """Exhaustive source-face partition by the three observed board instances."""
    from workcell_photo_oneshot import GUARD_WORD
    from scipy.optimize import linear_sum_assignment
    candidates = {}
    for photo in frames:
        rows = []
        for row in detections[GUARD_WORD]:
            if row['photo'] != photo or row['score'] < .6:
                continue
            mask = row['mask'] & (masks[f'v{photo}_mask'] > 0)
            if mask.any():
                rows.append({**row, 'mask': mask, 'points': frames[photo]['points'][mask]})
        largest = max((r['mask'].sum() for r in rows), default=0)
        candidates[photo] = [r for r in rows if r['mask'].sum() >= .2 * largest]
    complete = [p for p, rows in candidates.items() if len(rows) == 3]
    if not complete:
        raise ValueError('Three independent guard-board masks are required for a three-part report')
    anchor = max(complete, key=lambda p: sum(r['mask'].sum() for r in candidates[p]))
    rows = sorted(candidates[anchor], key=lambda r: np.where(r['mask'])[1].mean())
    mesh = trimesh.load(root / 'guard-multi.glb', force='mesh')
    uv, depth = _project(mesh.triangles_center, frames[anchor])
    if np.any(depth <= 0):
        raise ValueError('Guard mesh crosses its partition camera')
    xy = np.rint(uv).astype(int); h, w = rows[0]['mask'].shape
    xy[:, 0] = xy[:, 0].clip(0, w-1); xy[:, 1] = xy[:, 1].clip(0, h-1)
    distances = np.stack([cv2.distanceTransform((~r['mask']).astype(np.uint8), cv2.DIST_L2, 5)[xy[:, 1], xy[:, 0]] for r in rows])
    owners = distances.argmin(0)
    parts = []
    for index, side in enumerate(('left', 'center', 'right')):
        indices = np.flatnonzero(owners == index)
        part = mesh.submesh([indices], append=True, repair=False)
        file = f'guard-{side}.glb'; part.export(root / file)
        parts.append({'side': side, 'file': file, 'mesh': part, 'observations': [rows[index]],
                      'sourceFaceIndices': indices.tolist()})
    for photo, other in candidates.items():
        if photo == anchor or not other:
            continue
        support = np.array([[_inside(part['mesh'].vertices, row, frames) for row in other] for part in parts])
        a, b = linear_sum_assignment(-support)
        for i, j in zip(a, b):
            if support[i, j] >= .2:
                parts[i]['observations'].append(other[j])
    (root / 'guard-partition.json').write_text(json.dumps({'sourceFile': 'guard-multi.glb',
        'anchorPhoto': anchor, 'method': 'source-mask triangle ownership; no coordinate cuts or mesh deformation',
        'parts': [{k: p[k] for k in ('side', 'file', 'sourceFaceIndices')} for p in parts]}, indent=2))
    return parts


# Files build() writes into its root; everything else there is an input.
BUILD_OUTPUTS = ('guard-left.glb', 'guard-center.glb', 'guard-right.glb', 'guard-partition.json', 'object-extras.glb',
                 'object-proxies.glb', 'objects.json')


def build(root, sources, proxy_textures=True):
    from workcell_photo_oneshot import _array, _frame
    root = Path(root)
    sources = [Path(p) for p in sources]
    if len(sources) != 4 or len(set(p.resolve() for p in sources)) != 4 or not all(p.is_file() for p in sources):
        raise ValueError('Four distinct source photos are required')
    geometry = json.loads((root / 'geometry.json').read_text())
    segmentation = json.loads((root / 'sam3.json').read_text())
    if len(segmentation['results']) != 4:
        raise ValueError('SAM results must contain four photos')
    frames = {}
    for i in range(1, 5):
        raw = _frame(root, i)
        points, rgb = _array(raw['pts3d']), _array(raw['image'])
        valid = _array(raw['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2)
        frame = {'raw': raw, 'points': points, 'rgb': rgb, 'valid': valid,
                 'pose': _array(raw['camera_poses']), 'K': _array(raw['intrinsics'])}
        if points.shape != rgb.shape or valid.shape != rgb.shape[:2] or not valid.any():
            raise ValueError(f'Photo {i}: invalid point/image shape or support')
        uv, depth = _project(points[valid], frame)
        yy, xx = np.where(valid)
        residual = np.linalg.norm(uv-np.column_stack((xx, yy)), axis=1)
        if not np.isfinite(residual).all() or np.median(residual) >= 3 or np.any(depth <= 0):
            raise ValueError(f'Photo {i}: native point/camera projection mismatch')
        with Image.open(sources[i-1]) as image:
            image = ImageOps.exif_transpose(image)
            affine = np.asarray(raw['input_mask_transform']['input_to_canonical_pixel_centres'])
            if not np.isfinite(affine).all() or affine.shape != (3, 3) or abs(np.linalg.det(affine)) < 1e-12:
                raise ValueError('Invalid raw-to-canonical transform')
            frame['sourceSize'] = image.size
        frame['projectionMedianPixels'] = float(np.median(residual))
        frames[i] = frame
    words = [x['text'] for x in segmentation['prompts']]
    detections = {}
    for word in words:
        j = words.index(word)
        detections[word] = []
        for i, frame in frames.items():
            response = segmentation['results'][i-1][j]
            if len(response.get('rle', [])) != len(response.get('scores', [])):
                raise ValueError('SAM mask/score counts disagree')
            for k, (encoded, score) in enumerate(zip(response.get('rle', []), response.get('scores', []))):
                mask = decode_mask(frame['raw'], encoded)
                if mask.any():
                    detections[word].append({'photo': i, 'instance': k, 'word': word, 'score': float(score),
                                            'mask': mask, 'points': frame['points'][mask & frame['valid']]})
    catalog, scene, rejected, represented_posts, unsupported = [], trimesh.Scene(), [], [], []
    def observe(mask, photo, source):
        observation = _observation(mask, photo, source)
        if observation:
            # The report re-applies this support (support_mask) on its own frame and floor.
            observation['maskPixels'] = int(mask.sum())
            observation['observedMeasurements'], observation['supportedPixels'] = measure_support(
                frames[photo], mask, geometry['floor'], photo, source)
        return observation
    def add(ident, label, kind, file, nodes, observations, representation, notes=()):
        item = {'id': ident, 'label': label, 'kind': kind, 'model': {'file': file, 'nodes': nodes} if nodes else None,
                'observations': observations, 'measurements': _unknown('Visible geometry does not establish complete physical dimensions.'),
                'representation': representation, 'notes': list(notes)}
        catalog.append(item)
        return item
    def observations(rows):
        return [observe(r['mask'], r['photo'], f"SAM: {r['word']}; instance {r['instance']}") for r in rows]
    robot_models = {}
    for i in frames:
        _, nodes = _nodes(root, f'robot-v{i}.glb')
        robot_models[str(i)] = {'file': f'robot-v{i}.glb', 'nodes': nodes}
    robot_obs = []
    for i in frames:
        rows = [r for r in detections.get('industrial robot arm', []) if r['photo'] == i]
        if rows:
            robot_obs.append(observe(np.logical_or.reduce([r['mask'] for r in rows]), i, 'SAM industrial robot arm'))
    robot = add('robot', 'Industrial robot arm', 'robot', 'robot-v1.glb', robot_models['1']['nodes'], robot_obs,
                'generated pose per source photo', ['Robot configurations differ across photos; no static cross-view shape claim.'])
    robot['modelsByPhoto'] = robot_models
    cart_masks = np.load(root / 'cart-input.npz')
    _, nodes = _nodes(root, 'cart-single.glb')
    add('cart', 'Work cart', 'cart', 'cart-single.glb', nodes,
        [observe(cart_masks[f'v{i}_mask'] > 0, i, 'Selected cart mask; cart-mask-selection.json') for i in frames],
        'generated from RecGen photo 1; linked silhouette observations from four photos')
    guard_masks = np.load(root / 'guard-input.npz')
    for part in _guard_parts(root, detections, frames, guard_masks):
        _, nodes = _nodes(root, part['file'])
        add('v-guard-'+part['side'], {'left':'左侧折弯护板', 'center':'中间折弯护板', 'right':'右侧折弯护板'}[part['side']],
            'folded guard board', part['file'], nodes, observations(part['observations']),
            'source-mask partition of the generated guard; unchanged source triangles',
            ['Each board contains its own two adjoining sheet faces; bend means their interior angle (flat = 180 degrees).',
             'Triangle ownership is recorded in guard-partition.json; model-derived angles are not surveyed physical measurements.'])
    posts, nodes = _nodes(root, 'posts.glb')
    for node in sorted(nodes):
        word = 'yellow safety post' if node.startswith('box') else 'black bollard'
        points = _points(posts, [node]); rows = []
        for i in frames:
            candidates = [r for r in detections.get(word, []) if r['photo'] == i]
            if candidates:
                best = max(candidates, key=lambda r: _inside(points, r, frames))
                if _inside(points, best, frames) > .2:
                    rows.append(best)
        item = add('post-'+node, word.title()+' '+node.split('-')[-1], word, 'posts.glb', [node], observations(rows),
                   'image-supported primitive', ['Cross-view links require projected mesh support inside the instance mask. Primitive thickness is assumed.',
                   'Axis is constrained perpendicular to the inferred ground for display; it is not a measured physical axis.'])
        item['modelDimensionsNative'] = np.ptp(points, axis=0).tolist()
        represented_posts.append((item, rows))
    fence, fence_nodes = _nodes(root, 'fence-fitted.glb')
    up = np.asarray(geometry['floor']['normal'], float)
    floor_offset = float(geometry['floor']['offset'])
    for pi, plane in enumerate(geometry['fence']['planes']):
        nodes = [r['id'] for r in geometry['fence']['beams'] + geometry['fence']['continuations'] if r['plane'] == pi and r['id'] in fence_nodes]
        nodes = list(dict.fromkeys(nodes))
        rows = []
        for i, frame in frames.items():
            masks = [r['mask'] for r in detections.get('safety fence', []) if r['photo'] == i]
            if masks:
                mask = np.logical_or.reduce(masks) & (np.abs(frame['points'] @ np.asarray(plane['normal']) + plane['offset']) < max(geometry['floor']['residualP95Native'] * 5, 1e-6))
                source = f'SAM safety fence intersected with fitted plane {pi}'
                if not fence_panel_supported(mask & frame['valid']):
                    if mask.any():
                        unsupported.append(_unsupported(f'fence-{pi}', i, source, mask & frame['valid']))
                    continue
                obs = observe(mask, i, source)
                if obs: rows.append(obs)
        item = add(f'fence-{pi}', f'Safety fence section {pi+1}', 'safety fence', 'fence-fitted.glb', nodes, rows,
                   'fitted structural members; inferred continuations', geometry['fence']['assumptions'])
        item['geometryPlaneIndex'] = pi
        clearance = next((c for c in geometry['clearances'] if c['id'] == f'fence-plane-{pi}-lower-rail'), None)
        if clearance and len(rows) > 1:
            item['measurements']['groundClearance'] = {'valueNative': clearance.get('heightNative'), 'status': 'conditional-model-estimate',
                'source': 'geometry.json lower observed rail to modeled floor; provisional scale; not globally lowest rail'}
    _, nodes = _nodes(root, 'floor-fitted.glb')
    floor_obs = []
    for i, frame in frames.items():
        mask = frame['valid'] & (np.abs(frame['points'] @ up + floor_offset) < geometry['floor']['residualP95Native'])
        obs = observe(mask, i, 'Pointmap support near inferred floor plane')
        if obs: floor_obs.append(obs)
    add('floor', 'Floor', 'floor', 'floor-fitted.glb', nodes, floor_obs, 'fitted local plane', [geometry['floor']['status']])
    anchor = geometry['anchor']
    button_nodes = []
    for node, mesh in button_meshes(geometry).items():
        scene.add_geometry(mesh, node_name=node, geom_name=node); button_nodes.append(node)
    button_obs = []
    for view in anchor['views']:
        i = view['photo']; frame = frames[i]
        x0,y0,x1,y1 = view['boxRaw']
        corners = np.asarray([[x0,y0,1],[x1,y1,1]]) @ np.asarray(frame['raw']['input_mask_transform']['input_to_canonical_pixel_centres']).T
        lo = np.floor(corners[0,:2]).astype(int); hi = np.ceil(corners[1,:2]).astype(int)
        mask = np.zeros(frame['valid'].shape, bool)
        lo = np.maximum(lo,0); hi = np.minimum(hi, mask.shape[::-1])
        mask[lo[1]:hi[1],lo[0]:hi[0]] = True
        button_obs.append(observe(mask,i,'geometry.anchor component envelope transformed from original photo'))
    button = add('emergency-button', 'Emergency stop button', 'emergency stop button', 'object-extras.glb', button_nodes,
                 button_obs, 'parametric component; image-supported position/envelope',
                 anchor['assumptions'] + ['Red/yellow/gray part proportions and unseen thickness are rendering assumptions. Supplied dimensions remain an input hypothesis.'])
    for key, field in [('height','nativeHeight'),('width','nativeWidth')]:
        button['measurements'][key] = {'valueNative': anchor[field], 'status':'input-hypothesis', 'source':'geometry.anchor provisional whole-component envelope; uniform scale applies'}
    # Compatible aliases may identify the same instance, but class alone never joins observations.
    families = {'stack light':'signal light', 'workcell sign':'sign', 'warning sign':'sign', 'instruction poster':'sign'}
    groups, aliases = [], []
    for word in EXTRA_WORDS:
        for detection in detections.get(word, []):
            if word == 'work platform':
                i = detection['photo']
                assembly = (cart_masks[f'v{i}_mask'] > 0) | (guard_masks[f'v{i}_mask'] > 0)
                if (detection['mask'] & assembly).sum() / max(1, detection['mask'].sum()) > .5:
                    aliases.append({'category':word, 'photo':i, 'instance':detection['instance'],
                                    'objectId':'cart', 'reason':'Assembly covered by independent cart and V-guard masks'})
                    continue
            existing = next((item for item, rows in represented_posts
                             if word == 'light curtain' and any(r['photo']==detection['photo'] and _same(detection,r,frames) for r in rows)), None)
            if existing:
                existing.setdefault('semanticAliases', [])
                if word not in existing['semanticAliases']: existing['semanticAliases'].append(word)
                aliases.append({'category':word, 'photo':detection['photo'], 'instance':detection['instance'], 'objectId':existing['id']})
                continue
            family = families.get(word, word)
            group = next((g for g in groups if g['family']==family and all(_same(detection,r,frames) for r in g['rows'])), None)
            if group is None:
                group = {'family':family, 'rows':[]}; groups.append(group)
            group['rows'].append(detection)
    for number, group in enumerate(groups,1):
        rows = group['rows']; ident = group['family'].replace(' ','-')+f'-{number}'
        nodes, retained = [], []
        for row in rows:
            duplicate = next((r for r in retained if r['photo']==row['photo'] and _same(row,r,frames)), None)
            if duplicate is not None:
                aliases.append({'category':row['word'], 'photo':row['photo'], 'instance':row['instance'], 'objectId':ident})
                continue
            retained.append(row)
            frame = frames[row['photo']]
            native = _mesh(frame['points'], frame['valid'], frame['rgb'], row['mask'])
            if native is None:
                rejected.append({'category':row['word'], 'photo':row['photo'], 'instance':row['instance'], 'objectId':ident, 'reason':('No valid finite depth support' if not len(row['points']) else 'No supported adjacent triangle; photo observation retained')})
                continue
            node = f"{ident}-photo-{row['photo']}-{len(nodes)}"
            mesh = trimesh.Trimesh(native.vertices, native.faces, vertex_colors=np.rint(native.colors*255).astype(np.uint8), process=False)
            scene.add_geometry(mesh,node_name=node,geom_name=node)
            nodes.append(node)
        if retained:
            item = add(ident, rows[0]['word'].title()+' '+str(number), group['family'], 'object-extras.glb', nodes,
                       observations(retained), ('observed surface; original pointmap topology and colors' if nodes else 'unknown geometry; detected photo observation'),
                       ['No back faces, filled holes, or inferred cuboid. Folds follow observed pointmap geometry.',
                        'Cross-view identity requires reciprocal mask reprojection and compatible 3D support; correspondence is inferred.',
                        'Physical thickness, hidden extent and complete object dimensions are unknown.'])
            if nodes:
                item['modelDimensionsNative'] = np.ptp(np.concatenate([r['points'] for r in retained]),axis=0).tolist()
    result = {'schemaVersion':1, 'coordinateSystem':geometry['coordinateSystem'], 'objects':catalog,
              'coverage':{'objects':len(catalog), 'observations':sum(len(o['observations']) for o in catalog),
                          'requestedExtraCategories':list(EXTRA_WORDS),
                          'categoriesNotSegmented':[w for w in EXTRA_WORDS if w not in words],
                          'detectedExtraInstances':sum(len(detections.get(w,[])) for w in EXTRA_WORDS),
                          'unmeshedDetections':rejected, 'detectionsLinkedToExistingObjects':aliases,
                          'unsupportedObservations':unsupported,
                          'projectionMedianPixels':{str(i):f['projectionMedianPixels'] for i,f in frames.items()},
                          'status':'detected evidence coverage; semantic completeness requires visual review'}}
    (root/'object-extras.glb').write_bytes(scene.export(file_type='glb'))
    (root/'objects.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    # One placed model per extra object: merge repeats, alias floor/robot/cart/guard hits, replace raw fragments.
    from scripts.workcell_extra_models import consolidate
    consolidate(root, sources, textures=proxy_textures)
    return json.loads((root/'objects.json').read_text())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--sources',type=Path,nargs=4)
    parser.add_argument('--review-fence-support',action='store_true',help='Apply the fence-panel support rule to an existing catalog')
    args=parser.parse_args()
    if args.review_fence_support:
        print(json.dumps(review_fence_observations(args.root),indent=2))
    elif args.sources:
        print(json.dumps(build(args.root,args.sources)['coverage'],indent=2))
    else:
        parser.error('--sources is required to build a catalog')
