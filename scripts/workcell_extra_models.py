"""One placed model per extra object: merge repeated detections, alias floor hits, replace raw depth fragments.

The objects stage gave each extra detection (signs, lamps, cable tray, floor tape, a 'work platform')
its raw single-photo pointmap surface. Those fragments carry flying-pixel edges, repeat one physical
object once per photo (and a hazard tape once per stripe), and change with the photo they came from.
This catalog pass keeps every observation and its support rule, and then:

* aliases an extra whose cleaned support lies on the floor plane to the floor (unless it is a floor marking);
* merges floor markings whose footprints on the floor plane touch (a tape and its stripes, across photos);
* merges other extras of one family whose robust 3D centres are within MERGE_NATIVE (the same lamp or sign
  seen in several photos or split into two instances);
* gives each remaining extra exactly one proxy model in object-proxies.glb: a quad on the floor plane
  (floor markings), a quad in the fitted plane (thin supports), or a floor-aligned box (everything else),
  textured from the original photo when the originals are supplied.

Proxies are placement and visible-extent models from cleaned pointmap support (eroded instance mask, camera
depth within 3 MAD of its median); thickness and hidden extent remain unknown, and objects mounted on
unmodelled structure (gantry, ceiling) still appear without that structure.

python scripts/workcell_extra_models.py --root RUN --sources PHOTO_1.jpg ... PHOTO_N.jpg
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROXY_FILE = 'object-proxies.glb'
MERGE_NATIVE = .3      # duplicates of one lamp/sign sit within 0.03-0.2 native; distinct ones >= 0.9 apart in this capture
FLOOR_GAP_NATIVE = .06  # footprint gap still treated as touching (black tape bands between yellow stripes)
GRID_NATIVE = .02
OUTSIDE_MARGIN = .25    # native; a proxy is placed only above the fitted workcell floor (gantry lamps and signs included)
MERGEABLE = ('observed surface', 'unknown geometry', 'proxy:')


def _clean(observation, frame, geometry, segmentation, inputs):
    """Objects-stage support, eroded by one pixel, with camera depth within 3 MAD of its median."""
    from scripts.workcell_photo_objects import support_mask
    mask = support_mask(observation, frame, geometry, segmentation, inputs)
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    support = (eroded if (eroded & frame['valid']).sum() >= 20 else mask) & frame['valid']
    points, colors = frame['points'][support].astype(float), frame['rgb'][support]
    pixels = np.argwhere(support)[:, ::-1].astype(float)
    if not len(points):
        return points, colors, mask, pixels
    depth = (points - frame['pose'][:3, 3]) @ frame['pose'][:3, 2]
    median = np.median(depth)
    keep = np.abs(depth - median) <= max(3 * 1.4826 * np.median(np.abs(depth - median)), .02 * median)
    return points[keep], colors[keep], mask, pixels[keep]


def _floor_basis(geometry):
    up = np.asarray(geometry['floor']['normal'], float)
    length = np.linalg.norm(up)
    up, offset = up / length, float(geometry['floor']['offset']) / length
    a = np.cross(up, [1., 0, 0] if abs(up[0]) < .9 else [0, 1., 0]); a /= np.linalg.norm(a)
    return up, offset, a, np.cross(up, a)


def _footprint(points, e1, e2):
    cells = np.unique(np.floor(np.c_[points @ e1, points @ e2] / GRID_NATIVE).astype(np.int64), axis=0)
    return {tuple(cell) for cell in cells}


def _touch(a, b):
    reach = int(np.ceil(FLOOR_GAP_NATIVE / GRID_NATIVE))
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return any((x + dx, y + dy) in large for x, y in small for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1))


class _Union:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _quad(corners, color):
    mesh = trimesh.Trimesh(np.asarray(corners, float), [[0, 1, 2], [0, 2, 3]], process=False)
    mesh.visual.vertex_colors = np.tile(np.r_[color, 255], (4, 1)).astype(np.uint8)
    return mesh


def _rectangle(xy):
    """Robust oriented rectangle of 2D points: minimum-area orientation, 1-99 percentile extents."""
    (_, _), (_, _), angle = cv2.minAreaRect(xy.astype(np.float32))
    theta = np.radians(angle)
    axes = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    local = xy @ axes.T
    low, high = np.percentile(local, [1, 99], axis=0)
    return np.array([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]]) @ axes


def _billboard(pixels, points, frame):
    """A sign too small or noisy for a plane fit: its image rectangle at the median depth, facing that camera."""
    pose, K = np.asarray(frame['pose'], float), np.asarray(frame['K'], float)
    depth = float(np.median((points - pose[:3, 3]) @ pose[:3, 2]))
    corners = np.c_[_rectangle(pixels), np.ones(4)] @ np.linalg.inv(K).T * depth
    return corners @ pose[:3, :3].T + pose[:3, 3]


def _proxy(points, colors, kind, geometry, pixels=None, frame=None):
    """One model for one object's cleaned support (native coordinates)."""
    up, offset, e1, e2 = _floor_basis(geometry)
    color = np.median(colors, axis=0).astype(np.uint8)
    if kind == 'floor':
        xy = np.c_[points @ e1, points @ e2]
        corners = _rectangle(xy) @ np.stack([e1, e2]) + (-offset + .003) * up
        return _quad(corners, color), {'kind': 'floor quad', 'note': 'flat on the fitted floor plane'}
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    core = points[radii <= np.quantile(radii, .95)] if len(points) >= 20 else points
    _, singular, vectors = np.linalg.svd(core - core.mean(0), full_matrices=False)
    if len(core) >= 20 and singular[2] <= .15 * singular[1]:
        u, v = vectors[0], vectors[1]
        xy = np.c_[(points - center) @ u, (points - center) @ v]
        corners = center + _rectangle(xy) @ np.stack([u, v])
        return _quad(corners, color), {'kind': 'plane quad', 'note': 'thin support: quad in its fitted plane'}
    if kind == 'sign' and frame is not None:
        return _quad(_billboard(pixels, points, frame), color), {'kind': 'plane quad', 'note': 'sign: image rectangle at the median depth, facing its best photo'}
    horizontal = np.c_[points @ e1, points @ e2]
    _, _, rotation = np.linalg.svd(horizontal - horizontal.mean(0), full_matrices=False)
    a = rotation[0, 0] * e1 + rotation[0, 1] * e2
    b = np.cross(up, a)
    axes = np.stack([a, b, up])
    local = points @ axes.T
    low, high = np.percentile(local, [2, 98], axis=0)
    box = trimesh.creation.box(extents=np.maximum(high - low, 1e-3))
    box.apply_translation((low + high) / 2)
    box.vertices = box.vertices @ axes
    box.visual.vertex_colors = np.tile(np.r_[color, 255], (len(box.vertices), 1)).astype(np.uint8)
    return box, {'kind': 'box', 'note': 'floor-aligned box over the 2-98 percentile support'}


def consolidate(root, sources=None, textures=True):
    """Rewrite objects.json and object-proxies.glb; returns the merge/alias record. Deterministic and idempotent."""
    from scripts.workcell_photo_oneshot import _array, _frame, scene_photos
    root = Path(root)
    text = (root / 'objects.json').read_text()
    catalog = json.loads(text)
    geometry = json.loads((root / 'geometry.json').read_text())
    segmentation = json.loads((root / 'sam3.json').read_text())
    up, offset, e1, e2 = _floor_basis(geometry)
    residual = geometry['floor']['residualP95Native'] / np.linalg.norm(geometry['floor']['normal'])
    frames, loaded = {}, {}
    def frame(photo):
        if photo not in frames:
            raw = _frame(root, photo)
            points = _array(raw['pts3d'])
            frames[photo] = {'raw': raw, 'points': points, 'rgb': _array(raw['image']),
                             'valid': _array(raw['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2),
                             'pose': _array(raw['camera_poses']), 'K': _array(raw['intrinsics'])}
        return frames[photo]
    def inputs(name):
        if name not in loaded:
            loaded[name] = np.load(root / name)
        return loaded[name]
    extras = [item for item in catalog['objects'] if item.get('representation', '').startswith(MERGEABLE)
              and item['kind'] not in ('emergency stop button', 'floor')]
    support = {}
    for item in extras:
        rows = [_clean(o, frame(o['photo']), geometry, segmentation, inputs) for o in item['observations']]
        points = np.concatenate([r[0] for r in rows]) if rows else np.empty((0, 3))
        heights = points @ up + offset
        support[item['id']] = {'rows': rows, 'points': points,
                               'floor': bool(len(points) and abs(np.median(heights)) <= 2 * residual and np.percentile(heights, 90) <= 4 * residual)}
    record = catalog['coverage'].get('extraConsolidation') or {
        'rule': {'mergeNative': MERGE_NATIVE, 'floorGapNative': FLOOR_GAP_NATIVE,
                 'floorLying': '|median height| <= 2 x floor residual and 90th percentile <= 4 x residual',
                 'insideModelled': 'more than half of every observation support inside one robot/cart/guard support of the same photo',
                 'cleaning': 'objects-stage support eroded 1 px; camera depth within 3 MAD of its median'},
        'aliased': [], 'merged': []}
    hosts = [item for item in catalog['objects'] if item['kind'] in ('robot', 'cart', 'folded guard board')]
    host_masks = {}
    def host_of(item):
        """The robot/cart/guard whose same-photo support contains more than half of every observation of item."""
        from scripts.workcell_photo_objects import support_mask
        votes = None
        for observation, (_, _, mask, _) in zip(item['observations'], support[item['id']]['rows']):
            inside = set()
            for host in hosts:
                for other in host['observations']:
                    if other['photo'] != observation['photo']:
                        continue
                    key = (host['id'], other['photo'], other['source'])
                    if key not in host_masks:
                        host_masks[key] = support_mask(other, frame(other['photo']), geometry, segmentation, inputs)
                    if (mask & host_masks[key]).sum() > .5 * max(1, mask.sum()):
                        inside.add(host['id'])
            votes = inside if votes is None else votes & inside
        return sorted(votes)[0] if votes else None
    named = {item['id']: item for item in catalog['objects']}
    kept = []
    for item in extras:
        target = 'floor' if support[item['id']]['floor'] and item['kind'] != 'floor marking' and 'floor' in named else host_of(item)
        if target is None:
            kept.append(item)
            continue
        host = named[target]
        host['semanticAliases'] = sorted({*host.get('semanticAliases', []), item['kind']})
        record['aliased'].append({'objectId': item['id'], 'category': item['kind'], 'to': target,
                                  'reason': 'support lies on the floor plane' if target == 'floor' else 'support lies inside this object',
                                  'observations': [[o['photo'], o['source']] for o in item['observations']]})
    union = _Union([item['id'] for item in kept])
    centres = {item['id']: np.median(support[item['id']]['points'], axis=0) for item in kept if len(support[item['id']]['points'])}
    prints = {item['id']: _footprint(support[item['id']]['points'], e1, e2) for item in kept
              if item['kind'] == 'floor marking' and support[item['id']]['floor']}
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            if a['kind'] != b['kind'] or a['id'] not in centres or b['id'] not in centres:
                continue
            if a['id'] in prints and b['id'] in prints:
                if _touch(prints[a['id']], prints[b['id']]):
                    union.join(a['id'], b['id'])
            elif a['id'] not in prints and b['id'] not in prints and np.linalg.norm(centres[a['id']] - centres[b['id']]) <= MERGE_NATIVE:
                union.join(a['id'], b['id'])
    groups = {}
    for item in kept:
        groups.setdefault(union.find(item['id']), []).append(item)
    order = {item['id']: n for n, item in enumerate(catalog['objects'])}
    texture_frames = None
    if textures:
        if sources is None:
            raise ValueError('Textured proxies need every original photo of the scene')
        from scripts.workcell_photo_texture import source_texture_frames
        texture_frames = source_texture_frames(root, {p: frame(p) for p in range(1, scene_photos(root)[0] + 1)}, [Path(p) for p in sources])
    floor_hull = None
    if (root / 'floor-fitted.glb').is_file():
        vertices = np.asarray(trimesh.load(root / 'floor-fitted.glb', force='mesh').vertices)
        floor_hull = cv2.convexHull(np.c_[vertices @ e1, vertices @ e2].astype(np.float32))
    def _outside(centre):
        """Horizontal distance beyond the fitted floor footprint, or None when within OUTSIDE_MARGIN of it."""
        if floor_hull is None:
            return None
        distance = -cv2.pointPolygonTest(floor_hull, (float(centre @ e1), float(centre @ e2)), True)
        return distance if distance > OUTSIDE_MARGIN else None
    scene, survivors = trimesh.Scene(), []
    for members in sorted(groups.values(), key=lambda rows: order[rows[0]['id']]):
        head = max(members, key=lambda item: (len(support[item['id']]['points']), -order[item['id']]))
        observations = [(o, r) for _, _, _, o, r in sorted(
            (o['photo'], order[item['id']], n, o, r) for item in members
            for n, (o, r) in enumerate(zip(item['observations'], support[item['id']]['rows'])))]
        if len(members) > 1:
            record['merged'].append({'objectId': head['id'], 'members': sorted((item['id'] for item in members), key=order.get),
                                     'rule': 'touching floor footprints' if head['id'] in prints else f'robust centres within {MERGE_NATIVE} native'})
            head['semanticAliases'] = sorted({*head.get('semanticAliases', []), *(item['kind'] for item in members)} - {head['kind']})
        head['observations'] = [o for o, _ in observations]
        points = np.concatenate([r[0] for _, r in observations])
        colors = np.concatenate([r[1] for _, r in observations])
        if not len(points):
            head.update(model=None, representation='unknown geometry; detected photo observation')
            survivors.append(head)
            continue
        best = max(observations, key=lambda pair: len(pair[1][0]))
        use_all = support[head['id']]['floor'] or head['id'] in prints
        source_points, source_colors = (points, colors) if use_all else best[1][:2]
        outside = _outside(np.median(source_points, axis=0))
        if outside is not None:
            head.update(model=None, representation=f'photo observation only: outside the modelled workcell ({outside:.2f} native beyond the fitted floor)')
            head['notes'] = ['Detected and linked to its photos; no 3D model is placed outside the fitted workcell floor.']
            head.pop('proxyModel', None); head.pop('modelDimensionsNative', None)
            survivors.append(head)
            continue
        mesh, proxy = _proxy(source_points, source_colors, 'floor' if use_all else head['kind'], geometry,
                             pixels=best[1][3], frame=frame(best[0]['photo']))
        if texture_frames is not None and proxy['kind'] != 'box':
            from scripts.workcell_photo_texture import texture_planar_mesh
            masks = {}
            for o, r in observations:
                masks[o['photo']] = masks.get(o['photo'], np.zeros(r[2].shape, bool)) | (r[2] & frame(o['photo'])['valid'])
            mesh, appearance = texture_planar_mesh(mesh, texture_frames, masks, max_size=512)
            proxy['textureSourcePhotos'] = appearance['sourcePhotos']
        scene.add_geometry(mesh, node_name=head['id'], geom_name=head['id'])
        head['model'] = {'file': PROXY_FILE, 'nodes': [head['id']]}
        head['representation'] = f"proxy: {proxy['kind']} from cleaned pointmap support ({proxy['note']})"
        head['proxyModel'] = {**proxy, 'sourcePhotos': sorted({o['photo'] for o, _ in observations}) if use_all else [best[0]['photo']],
                              'supportPoints': int(len(source_points)), 'physicalValidation': 'none'}
        head['modelDimensionsNative'] = np.ptp(np.asarray(mesh.vertices), axis=0).tolist()
        head['notes'] = ['One proxy per object: placement and visible extent from cleaned pointmap support; thickness and hidden extent unknown.',
                         'Repeated detections of the same object are merged; supporting structure (gantry, ceiling) is not modelled.']
        survivors.append(head)
    removed = {item['id'] for item in extras} - {item['id'] for item in survivors}
    catalog['objects'] = [item for item in catalog['objects'] if item['id'] not in removed]
    coverage = catalog['coverage']
    coverage['objects'] = len(catalog['objects'])
    coverage['observations'] = sum(len(item['observations']) for item in catalog['objects'])
    coverage['extraConsolidation'] = record
    if len(scene.geometry):
        (root / PROXY_FILE).write_bytes(scene.export(file_type='glb'))
    ending = '\n' if text.endswith('\n') else ''
    updated = json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + ending
    if updated != text:
        (root / 'objects.json').write_text(updated)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs='+', help='Every photo of the scene, in photo order')
    parser.add_argument('--no-textures', action='store_true')
    args = parser.parse_args()
    print(json.dumps(consolidate(args.root, args.sources, textures=not args.no_textures), ensure_ascii=False, indent=2))
