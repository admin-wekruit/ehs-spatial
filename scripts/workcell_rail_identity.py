"""Are the left and right "fence lower rail beside the light curtain" endpoints the same physical member class?

Evidence audit only: saved cameras, planes, floor and models are read, never
adjusted, refit or written back. In every photo, each side's endpoint rail
member, the endpoint and its floor foot, the plane's other lower/frame members,
its detected beam lines and its unpaired horizontal line evidence are projected
through the SAVED camera onto the ORIGINAL full-resolution photo:

* photo<P>-<side>-<fence>.jpg        crop around the curtain base, with legend;
* photo<P>-<side>-<fence>-plane.jpg  the same photo resampled on the SAME fitted
  fence plane (rows = plane height above the saved floor, columns = along the
  rail from the endpoint; dark red = hidden behind a nearer pointmap surface).

Image evidence: luminance on that plane grid is averaged along the rail beside
the curtain (NEAR native from the endpoint; samples behind a nearer pointmap
surface dropped; a view is usable only if >= VISIBLE_GATE of the span is visible
at the member line). Reported per view: the strongest opposite-sign gradient
pairs within +-WINDOW of the member, and the underside edge (lowest strong
dark-below/bright-above step = lower edge of the lowest bright horizontal
member). Offsets are raw pixels at the visible middle of the span; that pixel is
converted back to a height on the SAME fitted plane by ray-plane intersection
with the saved camera, with plane-depth sensitivities. Diagnostic only; no
physical validation; centimetres use the run's conditional, unvalidated scale.

python scripts/workcell_rail_identity.py --root RUN --sources a.jpg b.jpg c.jpg d.jpg --out NEW_DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageOps

from workcell_button_bundle import _pixels, _project_raw
from workcell_endpoint_estimate import _bottom_face, _centerline
from workcell_photo_geometry import _intersect, _unit
from workcell_photo_metrology import _fence_plane_index
from workcell_photo_oneshot import _array, _frame

STEP = .002                        # plane grid spacing (native), rows and columns
PLANE_ALONG, PLANE_HEIGHT = (-.2, 2.), (0., .6)  # rectified view around the endpoint
NEAR = 1.5                         # profiled rail length from the endpoint, away from the curtain
WINDOW = .2                        # profile half-window around the member bottom, as plane height
PAIR_HEIGHT = (.008, .2)           # face height of one horizontal member: thin tube up to broad bottom profile
MIN_VISIBLE = .1                   # visible rail length (native) needed for a profile row
OCCLUSION = .03                    # a pointmap surface nearer by > 3 % of the range hides the sample
VISIBLE_GATE = .5                  # profile is usable only if >= 50 % of the span is visible at the member line
UNDERSIDE_FRACTION = .5            # underside = lowest bright-above step >= 50 % of the strongest in the window
PLANE_SHIFT = .05                  # sensitivity of the implied height to a plane-depth error (native)
CONTEXT_HEIGHT = (-.05, 1.)        # crop region on the plane (heights above floor), along = PLANE_ALONG
MAX_SIDE, LEGEND_H = 1100, 170
COLOURS = {'rail member': (255, 0, 255), 'endpoint': (0, 0, 255), 'floor foot': (255, 90, 0),
           'other lower/frame': (0, 165, 255), 'beam H': (0, 230, 0), 'beam V': (0, 110, 0),
           'H line evidence': (255, 255, 255), 'curtain terminal': (0, 230, 255),
           'underside edge': (255, 255, 0), 'strongest pair': (200, 170, 255), 'profiled span': (170, 170, 170)}  # BGR


def _camera(root, photo):
    frame = _frame(root, photo)
    transform = frame['input_mask_transform']
    camera = {'pose': _array(frame['camera_poses']).astype(float), 'K': _array(frame['intrinsics']).astype(float),
              'A': np.asarray(transform['input_to_canonical_pixel_centres'], float),
              'resizedHW': transform['resized_shape_hw'], 'crop': transform['crop_xyxy']}
    points = _array(frame['pts3d']).astype(float)
    valid = _array(frame['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2)
    camera['depth'] = np.where(valid, (points - camera['pose'][:3, 3]) @ camera['pose'][:3, 2], np.nan)
    return camera, points, valid, _array(frame['image'])


def _alignment(camera, points, valid, canonical, rgb, beams, planes):
    """Runtime proof that overlays sit on the frame's own pixels; raises on any convention error."""
    A, (top, left) = camera['A'], camera['crop'][1::-1]
    # 1. The original photo, EXIF-transposed, resamples onto the frame's canonical image
    #    best at zero shift: orientation and the raw<->canonical affine are the frame's own.
    small = cv2.resize(rgb, tuple(camera['resizedHW'][::-1]), interpolation=cv2.INTER_AREA).astype(float)
    h, w = valid.shape
    mae = {(dx, dy): float(np.mean(abs(small[top + dy + 1:top + dy + h - 1, left + dx + 1:left + dx + w - 1] - canonical[1:-1, 1:-1])))
           for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    assert min(mae, key=mae.get) == (0, 0) and mae[0, 0] < 8, f'Photo does not match frame image: {mae}'
    # 2. Saved pts3d of every valid canonical pixel reproject to that pixel's centre (mapped
    #    back to raw) through the saved camera with no systematic offset. A half-pixel
    #    convention slip shows as ~0.5 px bias, a swapped axis/pose inversion as >> 1 px.
    #    Single pixels scatter (~0.6 px) because the pinhole is a fit to the network rays.
    ys, xs = np.nonzero(valid)
    grid = _pixels(np.c_[xs, ys], np.linalg.inv(A))
    raw, _ = _project_raw(points[ys, xs], camera)
    error = (raw - grid) * np.diag(A)[:2]                # canonical pixels
    size = np.linalg.norm(error, axis=1)
    bias = error.mean(0)
    assert np.all(abs(bias) < .05) and np.median(size) < 1., f'Camera does not reproduce its pixel grid: {bias}, {np.median(size)}'
    # 3. Members were built by ray-plane intersection of raw image edges with this camera;
    #    projecting those 3D points returns the detected raw pixels.
    trips = []
    for beam in beams:
        edge = np.asarray(beam['rawEdges'], float).reshape(-1, 2)
        plane = planes[beam['plane']]
        back, _ = _project_raw(_intersect(_pixels(edge, A), camera['K'], camera['pose'], plane['normal'], plane['offset']), camera)
        trips.append(np.max(abs(back - edge)))
    # Float32 saved rotations are orthonormal to ~1e-7, hence a 1e-3 raw-px (not 1e-9) bound.
    assert not trips or max(trips) < 1e-3, 'Projection is not the inverse of the member ray model'
    return {'photoResampleMAE': mae[0, 0], 'photoResampleMAEBestOtherShift': min(v for k, v in mae.items() if k != (0, 0)),
            'pts3dReprojectionMeanSignedCanonicalPx': bias.tolist(), 'pts3dPixels': int(len(xs)),
            'pts3dReprojectionMedianCanonicalPx': float(np.median(size)),
            'pts3dReprojectionP90CanonicalPx': float(np.percentile(size, 90)),
            'pts3dFractionWithinHalfCanonicalPx': float(np.mean(size < .5)),
            'memberRayRoundTripMaxRawPx': float(max(trips)) if trips else None, 'memberRayRoundTripEdges': len(trips),
            'scope': ('Photo/affine: zero shift is the best of 9 one-pixel shifts. Camera: pts3d reprojection of all valid pixels has no '
                      'systematic offset (|mean signed| < 0.05 canonical px); single pixels scatter because the saved pinhole is a fit '
                      'to the network ray field. Members: inverse of the ray-plane construction to < 1e-3 raw px.')}


def _member(scene, node):
    transform, name = scene.graph[node]
    mesh = scene.geometry[name].copy()
    mesh.apply_transform(transform)
    return mesh


def _canvas(uv, origin, scale):
    return [tuple(int(v) for v in np.rint((p - origin) * scale)) for p in np.atleast_2d(uv)]


def _segment(image, points, camera, origin, scale, colour, thickness):
    uv, z = _project_raw(np.asarray(points), camera)
    if (z > 1e-3).all():
        cv2.line(image, *_canvas(uv, origin, scale), colour, thickness, cv2.LINE_AA)


def _draw_mesh(image, mesh, camera, origin, scale, colour, thickness):
    for a, b in mesh.face_adjacency_edges[mesh.face_adjacency_angles > .1]:
        _segment(image, mesh.vertices[[a, b]], camera, origin, scale, colour, thickness)


def _plane_grid(rgb, camera, plane, ground, point, axis):
    """The photo resampled on the fitted plane: rows = heights (top = high), columns = along the rail from the endpoint."""
    pn, po = plane
    gn, go = ground
    up = _unit(gn - (gn @ pn) * pn)                      # in-plane upward direction
    along = np.arange(PLANE_ALONG[0], PLANE_ALONG[1] - 1e-9, STEP)   # end-exclusive: 1100 columns
    heights = np.arange(PLANE_HEIGHT[1], PLANE_HEIGHT[0] - 1e-9, -STEP)
    base = point + along[:, None] * axis
    xyz = base[None] + ((heights[:, None] - (base @ gn + go)[None]) / (up @ gn))[..., None] * up
    uv, z = _project_raw(xyz.reshape(-1, 3), camera)
    uv, z = uv.reshape(*xyz.shape[:2], 2), z.reshape(xyz.shape[:2])
    rows, cols = rgb.shape[:2]
    inside = (z > 0) & (uv[..., 0] >= 0) & (uv[..., 0] <= cols - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= rows - 1)
    safe = np.where(inside[..., None], uv, -1).astype(np.float32)
    colour = cv2.remap(rgb, safe[..., 0], safe[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    index = np.rint(_pixels(np.where(inside[..., None], uv, 0).reshape(-1, 2), camera['A'])).astype(int)
    depth_rows, depth_cols = camera['depth'].shape
    known = inside.ravel() & (index[:, 0] >= 0) & (index[:, 0] < depth_cols) & (index[:, 1] >= 0) & (index[:, 1] < depth_rows)
    depth = np.full(len(index), np.nan)
    depth[known] = camera['depth'][index[known, 1], index[known, 0]]
    hidden = inside & (depth.reshape(z.shape) < z * (1 - OCCLUSION))
    gray = cv2.cvtColor(colour, cv2.COLOR_RGB2GRAY).astype(float)
    gray[~inside | hidden] = np.nan
    return {'along': along, 'heights': heights, 'uv': uv, 'inside': inside, 'hidden': hidden, 'colour': colour, 'gray': gray}


def _smooth(values, sigma=2.):
    """Normalised Gaussian smoothing that ignores missing samples."""
    finite = np.isfinite(values)
    kernel = np.exp(-.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
    weight = np.convolve(finite.astype(float), kernel, 'same')
    out = np.convolve(np.where(finite, values, 0), kernel, 'same') / np.maximum(weight, 1e-9)
    return np.where(weight > .5 * kernel.sum(), out, np.nan)


def _row_mean(block, minimum):
    finite = np.isfinite(block)
    count = finite.sum(1)
    return np.where(count >= minimum, np.where(finite, block, 0).sum(1) / np.maximum(count, 1), np.nan)


def _edge_pairs(grid, model_h):
    """Strongest opposite-sign gradient pairs of the along-rail luminance profile (ascending height)."""
    rows = np.flatnonzero(abs(grid['heights'] - model_h) <= WINDOW + 1e-9)[::-1]   # ascending height
    columns = np.flatnonzero((grid['along'] >= 0) & (grid['along'] <= NEAR + 1e-9))
    block = grid['gray'][np.ix_(rows, columns)]
    h = grid['heights'][rows]
    profile = _row_mean(block, MIN_VISIBLE / STEP)
    gradient = np.gradient(_smooth(profile))           # gray levels per STEP of plane height
    finite = np.isfinite(gradient)
    peaks = [i for i in range(1, len(gradient) - 1) if finite[i - 1:i + 2].all() and gradient[i] != 0
             and abs(gradient[i]) >= abs(gradient[i - 1]) and abs(gradient[i]) >= abs(gradient[i + 1])]
    noise = float(np.median(abs(gradient[finite]))) if finite.any() else None
    pairs = sorted(((min(abs(gradient[a]), abs(gradient[b])), a, b) for a in peaks for b in peaks
                    if b > a and np.sign(gradient[a]) != np.sign(gradient[b]) and PAIR_HEIGHT[0] <= h[b] - h[a] <= PAIR_HEIGHT[1]),
                   reverse=True)
    visible = np.flatnonzero(np.isfinite(block).any(0))

    def agreement(a, b):
        # A coherent horizontal edge keeps both signs at the same heights in every quarter of the visible rail.
        count = 0
        for part in np.array_split(visible, 4):
            g = np.gradient(_smooth(_row_mean(block[:, part], 3)))
            ok = True
            for i in (a, b):
                local = g[max(0, i - 3):i + 4]
                local = local[np.isfinite(local)]
                ok &= len(local) > 0 and np.max(np.sign(gradient[i]) * local) >= abs(gradient[i]) / 3
            count += bool(ok)
        return count

    found = [{'lowerRow': int(rows[a]), 'upperRow': int(rows[b]), 'lowerHeightNative': float(h[a]), 'upperHeightNative': float(h[b]),
              'faceHeightNative': float(h[b] - h[a]), 'lowerGradientGrayPerStep': float(gradient[a]),
              'upperGradientGrayPerStep': float(gradient[b]), 'polarity': 'bright band' if gradient[a] > 0 > gradient[b] else 'dark band',
              'pairStrengthGrayPerStep': float(score), 'contrastToMedianGradient': float(score / noise) if noise else None,
              'quartersAgreeing': agreement(a, b)} for score, a, b in pairs[:3]]
    strongest = int(np.nanargmax(abs(gradient))) if finite.any() else None
    # ponytail: post-hoc rule (chosen after viewing these profiles, not pre-registered): the lower edge of the
    # lowest bright horizontal member is the lowest dark-below/bright-above step that is at least half as strong
    # as the strongest such step in the window. The visual labels remain the primary evidence.
    rising = [i for i in peaks if gradient[i] > 0]
    underside = None
    if rising:
        top = max(gradient[i] for i in rising)
        i = min(i for i in rising if gradient[i] >= UNDERSIDE_FRACTION * top)
        underside = {'row': int(rows[i]), 'heightNative': float(h[i]), 'gradientGrayPerStep': float(gradient[i]),
                     'strongestRisingGrayPerStep': float(top), 'strongestRisingHeightNative': float(h[[j for j in rising if gradient[j] == top][0]]),
                     'contrastToMedianGradient': float(gradient[i] / noise) if noise else None}
    return {'pairs': found, 'underside': underside, 'heights': h, 'profile': profile, 'gradient': gradient, 'noise': noise,
            'strongest': None if strongest is None else (int(rows[strongest]), float(h[strongest]), float(gradient[strongest])),
            'visibleColumns': visible, 'columns': columns, 'rows': rows}


def _fmt(value, digits=2):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def _plane_view(grid, side, plane_beams, evidence, best, underside, title):
    """Rectified evidence picture with height grid, member rectangles and the found pair."""
    image = cv2.cvtColor(grid['colour'], cv2.COLOR_RGB2BGR)
    image[~grid['inside']] = 0
    image[grid['hidden']] = (.45 * image[grid['hidden']] + .55 * np.array([30, 0, 140])).astype(np.uint8)
    col = lambda a: int(round((a - PLANE_ALONG[0]) / STEP))
    row = lambda hgt: int(round((PLANE_HEIGHT[1] - hgt) / STEP))
    for hgt in np.arange(PLANE_HEIGHT[0], PLANE_HEIGHT[1] + 1e-9, .05):
        major = abs(hgt * 10 - round(hgt * 10)) < 1e-6
        cv2.line(image, (0, row(hgt)), (image.shape[1], row(hgt)), (60, 60, 230) if major else (150, 150, 230), 1)
        if major:
            cv2.putText(image, f'{hgt:.1f}', (2, row(hgt) - 3), cv2.FONT_HERSHEY_SIMPLEX, .38, (60, 60, 255), 1, cv2.LINE_AA)

    def box(points, colour, thickness=1):
        a, hgt = (points - side['point']) @ side['axis'], points @ side['ground'][0] + side['ground'][1]
        cv2.rectangle(image, (col(a.min()), row(hgt.max())), (col(a.max()), row(hgt.min())), colour, thickness)

    for beam in plane_beams:
        ends = np.asarray(beam['endsNative'])
        across = _unit(np.cross(side['planeTuple'][0], ends[1] - ends[0])) * beam['widthNative'] / 2
        box(np.vstack([ends + across, ends - across]), COLOURS['beam H' if beam['horizontal'] else 'beam V'])
    for item in evidence:
        if item['kind'] == 'unpaired horizontal line':
            a, hgt = (item['xyz'] - side['point']) @ side['axis'], item['xyz'] @ side['ground'][0] + side['ground'][1]
            cv2.line(image, (col(a[0]), row(hgt[0])), (col(a[1]), row(hgt[1])), COLOURS['H line evidence'], 1)
    for mesh in side['lowerMeshes']:
        box(mesh.vertices, COLOURS['other lower/frame'])
    box(side['mesh'].vertices, COLOURS['rail member'], 2)
    for a in (0, NEAR):
        cv2.line(image, (col(a), 0), (col(a), image.shape[0]), COLOURS['profiled span'], 1)
    if best:
        for key in ('lowerHeightNative', 'upperHeightNative'):
            cv2.line(image, (col(0), row(best[key])), (col(NEAR), row(best[key])), COLOURS['strongest pair'], 1)
    if underside:
        cv2.line(image, (col(0), row(underside['heightNative'])), (col(NEAR), row(underside['heightNative'])), COLOURS['underside edge'], 2)
    header = np.full((16 + 17 * len(title), image.shape[1], 3), 24, np.uint8)
    for i, text in enumerate(title):
        cv2.putText(header, text, (6, 16 + 17 * i), cv2.FONT_HERSHEY_SIMPLEX, .38, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([header, image])


def run(root, sources, out):
    root, out = Path(root), Path(out)
    if out.exists():
        raise FileExistsError(f'Refusing existing output directory: {out}')
    estimate = json.loads((root / 'model-endpoint-estimate.json').read_text())
    report = json.loads((root / 'scene-report.json').read_text())
    geometry = json.loads((root / 'geometry.json').read_text())
    catalog = {item['id']: item for item in json.loads((root / 'objects.json').read_text())['objects']}
    glb = (root / 'fence-fitted.glb').read_bytes()
    scene = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type='glb', force='scene', process=False)
    gn = np.asarray(estimate['ground']['normal'], float)
    gn, go = gn / np.linalg.norm(gn), float(estimate['ground']['offset']) / np.linalg.norm(gn)
    planes = [{'normal': np.asarray(p['normal'], float) / np.linalg.norm(p['normal']),
               'offset': float(p['offset']) / np.linalg.norm(p['normal'])} for p in geometry['fence']['planes']]
    scale = report['modelMeasurementScale']['nativeToMeters']
    cm = lambda native: None if native is None or scale is None else native * scale * 100
    published = {row['id']: row for row in report['endpointEstimation']['endpoints']}
    objects = {row['id']: row for row in estimate['objects']}
    roles = {row['id']: row['role'] for row in geometry['fence']['continuations']}
    roles |= {row['meshNode']: 'measured multiview lower rail (clearance mesh node)' for row in geometry['clearances'] if row.get('meshNode')}
    cameras = {photo: _camera(root, photo) for photo in range(1, 5)}
    sides = []
    for row in estimate['objects']:
        if row.get('measurementScope') != 'model_lower_rail_near_curtain':
            continue
        if row['modelSha256'] != hashlib.sha256(glb).hexdigest() or published[row['id']]['heightNative'] != row['heightNative']:
            raise ValueError('Endpoint is not bound to this fence model revision: ' + row['id'])
        pi = _fence_plane_index(catalog[row['objectId']])
        plane = (planes[pi]['normal'], planes[pi]['offset'])
        ends = np.asarray(row['bottomCenterlineEndsNative'])
        # Profile runs from the endpoint along the member, away from the curtain-side end.
        axis = _unit(ends[1] - ends[0]) * (1 if row['closestAlongRailFraction'] <= .5 else -1)
        point = np.asarray(row['pointNative'])
        evidence = []
        for kind, items in (('paired beam', [b for b in geometry['fence']['beams'] if b['plane'] == pi and b['horizontal']]),
                            ('unpaired horizontal line', [l for l in geometry['fence']['horizontalLineEvidence'] if l['plane'] == pi])):
            for item in items:
                if kind == 'paired beam':
                    xyz = np.asarray(item['endsNative']) - item['widthNative'] / 2 * gn
                else:   # re-measured on the saved plane and current floor through its own source camera
                    c = cameras[item['sourcePhoto']][0]
                    xyz = _intersect(_pixels(np.asarray(item['rawEnds']), c['A']), c['K'], c['pose'], *plane)
                offsets = sorted(float((p - point) @ axis) for p in xyz)
                if np.isfinite(xyz).all() and offsets[1] >= PLANE_ALONG[0] and offsets[0] <= PLANE_ALONG[1]:
                    evidence.append({'kind': kind, 'id': item.get('id'), 'sourcePhoto': item['sourcePhoto'],
                                     'bottomHeightNative': float(np.mean(xyz @ gn + go)), 'bottomHeightCm': cm(float(np.mean(xyz @ gn + go))),
                                     'alongFromEndpointNative': offsets, 'xyz': xyz})
        lower = sorted(node for node, role in roles.items() if node != row['node']
                       and node in catalog[row['objectId']]['model']['nodes'] and ('lower' in role or 'end-frame' in role))
        sides.append({'endpoint': row, 'side': published[row['id']]['side'], 'plane': pi, 'planeTuple': plane, 'ground': (gn, go),
                      'lower': lower, 'lowerMeshes': [_member(scene, node) for node in lower],
                      'role': roles.get(row['node']), 'curtain': objects[row['pairedObjectId'] + ':terminal'], 'axis': axis, 'point': point,
                      'topHeightNative': float(np.mean(_centerline(_bottom_face(scene, row['node'], -gn)) @ gn + go)),
                      'mesh': _member(scene, row['node']), 'evidence': evidence})
    out.mkdir(parents=True)
    metrics = {'schemaVersion': 1, 'scope': 'Evidence audit of the saved revision; no refit, no camera change, no physical validation.',
               'root': str(root), 'nativeToMetersConditional': scale, 'scaleStatus': report['modelMeasurementScale']['status'],
               'parameters': {'stepNative': STEP, 'planeAlongNative': PLANE_ALONG, 'planeHeightNative': PLANE_HEIGHT, 'nearNative': NEAR,
                              'windowNative': WINDOW, 'pairHeightNative': PAIR_HEIGHT, 'minVisibleNative': MIN_VISIBLE,
                              'occlusionFraction': OCCLUSION, 'contextHeightNative': CONTEXT_HEIGHT, 'visibleGate': VISIBLE_GATE,
                              'undersideFraction': UNDERSIDE_FRACTION, 'planeShiftNative': PLANE_SHIFT},
               'inputSha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                               ('model-endpoint-estimate.json', 'geometry.json', 'objects.json', 'fence-fitted.glb', 'scene-report.json')},
               'sourceSha256': {}, 'ground': {'normal': gn.tolist(), 'offset': go}, 'alignment': {},
               'sides': [{'side': s['side'], 'endpointId': s['endpoint']['id'], 'objectId': s['endpoint']['objectId'], 'plane': s['plane'],
                          'node': s['endpoint']['node'], 'nodeRole': s['role'], 'modelBottomHeightNative': s['endpoint']['heightNative'],
                          'modelBottomHeightCm': cm(s['endpoint']['heightNative']), 'modelTopHeightNative': s['topHeightNative'],
                          'otherLowerOrFrameMembers': s['lower'],
                          'horizontalEvidenceNearCurtain': [{k: v for k, v in e.items() if k != 'xyz'} for e in s['evidence']]}
                         for s in sides], 'views': []}
    for photo, source in enumerate(sources, 1):
        source = Path(source)
        metrics['sourceSha256'][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        with Image.open(source) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
        camera, points, valid, canonical = cameras[photo]
        metrics['alignment'][photo] = _alignment(camera, points, valid, canonical, rgb,
                                                 [b for b in geometry['fence']['beams'] if b['sourcePhoto'] == photo], planes)
        for side in sides:
            e, pi, plane = side['endpoint'], side['plane'], side['planeTuple']
            view = {'photo': photo, 'side': side['side'], 'endpointId': e['id'], 'objectId': e['objectId'], 'plane': pi, 'node': e['node']}
            grid = _plane_grid(rgb, camera, plane, (gn, go), side['point'], side['axis'])
            model_row = int(np.argmin(abs(grid['heights'] - e['heightNative'])))
            near = np.flatnonzero((grid['along'] >= 0) & (grid['along'] <= NEAR + 1e-9))
            in_frame = near[grid['inside'][model_row, near]]
            if not len(in_frame):
                view['status'] = 'near-curtain rail segment not in this frame'
                metrics['views'].append(view)
                continue
            found = _edge_pairs(grid, e['heightNative'])
            seen = near[np.isfinite(grid['gray'][model_row, near])]
            ref = int(np.median(seen if len(seen) else in_frame))    # visible middle of the profiled span
            column = grid['uv'][:, ref]
            direction = _unit(column[0] - column[-1])               # image direction of increasing plane height
            offset = lambda r: float((column[r] - column[model_row]) @ direction)

            def ray_height(r):
                xyz = _intersect(_pixels(column[r][None], camera['A']), camera['K'], camera['pose'], *plane)[0]
                return float(xyz @ gn + go)

            for pair in found['pairs']:
                pair.update(lowerOffsetRawPx=offset(pair['lowerRow']), upperOffsetRawPx=offset(pair['upperRow']),
                            lowerRayPlaneHeightNative=ray_height(pair['lowerRow']), lowerHeightCm=cm(pair['lowerHeightNative']),
                            faceHeightCm=cm(pair['faceHeightNative']))
            best = found['pairs'][0] if found['pairs'] else None
            usable = len(seen) >= VISIBLE_GATE * len(near)
            under = found['underside']
            if under:
                pixel = column[under['row']]

                def shifted(delta):
                    xyz = _intersect(_pixels(pixel[None], camera['A']), camera['K'], camera['pose'], plane[0], plane[1] + delta)[0]
                    return float(xyz @ gn + go)

                height = ray_height(under['row'])
                # Where does the saved pointmap put the visible face just above this edge, relative to the plane?
                # A face in front of the plane makes the same-plane height read low (diagnostic, not a refit).
                side_sign = np.sign(camera['pose'][:3, 3] @ plane[0] + plane[1])
                band = np.ix_((grid['heights'] >= under['heightNative'] + .01) & (grid['heights'] <= under['heightNative'] + .08), near)
                cells = np.unique(np.rint(_pixels(grid['uv'][band][grid['inside'][band]], camera['A'])).astype(int), axis=0)
                cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < valid.shape[1]) & (cells[:, 1] >= 0) & (cells[:, 1] < valid.shape[0])]
                cells = cells[valid[cells[:, 1], cells[:, 0]]]
                toward = (points[cells[:, 1], cells[:, 0]] @ plane[0] + plane[1]) * side_sign
                face = float(np.median(toward)) if len(toward) else None
                under.update(offsetRawPx=offset(under['row']), rawPx=pixel.tolist(), rayPlaneHeightNative=height, heightCm=cm(height),
                             minusModelNative=height - e['heightNative'], minusModelCm=cm(height - e['heightNative']),
                             heightIfPlaneShiftedNative={f'{-PLANE_SHIFT:+}': shifted(-PLANE_SHIFT), f'{PLANE_SHIFT:+}': shifted(PLANE_SHIFT)},
                             faceDepthCheck=None if face is None else {
                                 'pointmapFaceOffsetTowardCameraNative': face, 'quartilesNative': np.percentile(toward, [25, 75]).tolist(),
                                 'pointmapPixels': int(len(toward)), 'heightIfEdgeAtFaceDepthNative': shifted(-face * side_sign),
                                 'heightIfEdgeAtFaceDepthCm': cm(shifted(-face * side_sign)),
                                 'scope': 'Median saved-pointmap distance of the face band 0.01-0.08 native above the edge; a parallel '
                                          'plane at that depth is a sensitivity reading, not a refit or a validated position'})
            top_row = int(np.argmin(abs(grid['heights'] - side['topHeightNative'])))
            marks, z = _project_raw(np.array([e['pointNative'], e['footNative'], side['curtain']['pointNative'],
                                              side['curtain']['footNative']]), camera)
            strongest = found['strongest']
            view.update(status='profiled', modelBottomHeightNative=e['heightNative'], modelBottomHeightCm=cm(e['heightNative']),
                        profiledSpanNative=[0, NEAR], nearSamplesInFrameAtModelLine=int(len(in_frame)),
                        nearSamplesVisibleAtModelLine=int(len(seen)), nearSamples=int(len(near)),
                        referenceAlongNative=float(grid['along'][ref]), referenceRawPx=column[model_row].tolist(),
                        rawPxPerNativeHeightAtReference=float(np.linalg.norm(column[0] - column[-1]) / np.ptp(PLANE_HEIGHT)),
                        modelTopOffsetRawPx=offset(top_row), endpointRawPx=marks[0].tolist(), footRawPx=marks[1].tolist(),
                        curtainTerminalRawPx=marks[2].tolist(), medianAbsGradientGrayPerStep=found['noise'],
                        strongestSingleEdge=None if strongest is None else {
                            'heightNative': strongest[1], 'offsetRawPx': offset(strongest[0]), 'gradientGrayPerStep': strongest[2]},
                        edgePairs=found['pairs'], undersideEdge=under,
                        profileUsable=bool(usable),
                        imageImpliedRailBottom=None if not (usable and under) else {
                            k: under[k] for k in ('offsetRawPx', 'rayPlaneHeightNative', 'heightCm', 'minusModelNative', 'minusModelCm',
                                                  'heightIfPlaneShiftedNative', 'faceDepthCheck')} | {
                            'method': 'underside edge (lowest strong dark-below/bright-above step); its raw pixel at the reference column '
                                      'intersected with the fitted fence plane through the saved camera'},
                        profile={'heightsNative': np.round(found['heights'], 4).tolist(),
                                 'offsetsRawPxAtReference': [round(offset(r), 2) for r in found['rows']],
                                 'luminance': [None if not np.isfinite(v) else round(float(v), 2) for v in found['profile']],
                                 'visibleColumns': np.isfinite(grid['gray'][np.ix_(found['rows'], found['columns'])]).sum(1).tolist()})
            stem = f"photo{photo}-{side['side']}-{e['objectId']}"
            under_text = ('underside edge: ' + ('none' if not under else
                          f"h={under['rayPlaneHeightNative']:.3f} ({_fmt(under['heightCm'])} cm cond.), {under['offsetRawPx']:+.0f} raw px vs member")
                          + ('' if usable else f'  [NOT USABLE: {len(seen)}/{len(near)} visible < {VISIBLE_GATE:.0%}]'))
            pair_text = 'strongest pair: ' + ('none' if not best else
                         f"{best['lowerHeightNative']:.3f}..{best['upperHeightNative']:.3f} {best['polarity']}, quarters {best['quartersAgreeing']}/4")
            title = [f"Photo {photo} | {side['side']} side | {e['objectId']} plane {pi} | photo resampled ON the fitted plane: rows = height above saved floor",
                     f"(red grid every 0.05 native), columns = along rail from endpoint {PLANE_ALONG[0]}..{PLANE_ALONG[1]} native; dark red = hidden; gray = profiled span",
                     f"magenta = {e['node']} ({side['role']}), bottom {e['heightNative']:.3f}; orange = other lower/frame; green = beams; white = H lines",
                     f'cyan = {under_text}; pink = {pair_text}']
            cv2.imwrite(str(out / f'{stem}-plane.jpg'), _plane_view(grid, side, [b for b in geometry['fence']['beams'] if b['plane'] == pi],
                                                                    side['evidence'], best, under, title), [cv2.IMWRITE_JPEG_QUALITY, 85])
            # Crop: the plane region around the curtain base plus endpoint, foot and curtain terminal.
            corners = [side['point'] + a * side['axis'] + (hgt - e['heightNative']) * gn for a in PLANE_ALONG for hgt in CONTEXT_HEIGHT]
            uv, zz = _project_raw(np.array(corners), camera)
            box = np.vstack([uv[zz > 0], marks[z > 0]])
            lo, hi = box.min(0), box.max(0)
            lo, hi = lo - .05 * (hi - lo), hi + .05 * (hi - lo)
            lo = np.clip(np.floor(lo), 0, [rgb.shape[1] - 2, rgb.shape[0] - 2]).astype(int)
            hi = np.clip(np.ceil(hi), lo + 2, [rgb.shape[1], rgb.shape[0]]).astype(int)
            factor = min(2., MAX_SIDE / (hi[0] - lo[0]), (MAX_SIDE - LEGEND_H) / (hi[1] - lo[1]))
            image = cv2.cvtColor(cv2.resize(rgb[lo[1]:hi[1], lo[0]:hi[0]], None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_RGB2BGR)
            thin = max(1, round(1.5 * factor))
            for item in side['evidence']:
                if item['kind'] == 'unpaired horizontal line':
                    _segment(image, item['xyz'], camera, lo, factor, COLOURS['H line evidence'], 1)
            for beam in geometry['fence']['beams']:
                if beam['plane'] == pi:
                    ends = np.asarray(beam['endsNative'])
                    across = _unit(np.cross(plane[0], ends[1] - ends[0])) * beam['widthNative'] / 2
                    for sign in (-1, 1):
                        _segment(image, ends + sign * across, camera, lo, factor, COLOURS['beam H' if beam['horizontal'] else 'beam V'],
                                 thin if beam['horizontal'] else 1)
            for mesh in side['lowerMeshes']:
                _draw_mesh(image, mesh, camera, lo, factor, COLOURS['other lower/frame'], thin)
            _draw_mesh(image, side['mesh'], camera, lo, factor, COLOURS['rail member'], thin)
            span = side['point'] + np.outer([0, NEAR], side['axis'])
            for a in span:
                _segment(image, [a, a + .5 * gn], camera, lo, factor, COLOURS['profiled span'], 1)
            if best:
                for key in ('lowerHeightNative', 'upperHeightNative'):
                    _segment(image, span + (best[key] - e['heightNative']) * gn, camera, lo, factor, COLOURS['strongest pair'], 1)
            if under:
                _segment(image, span + (under['heightNative'] - e['heightNative']) * gn, camera, lo, factor, COLOURS['underside edge'], thin)
            if z[0] > 0 and z[1] > 0:
                p, f = _canvas(marks[:2], lo, factor)
                cv2.line(image, p, f, COLOURS['endpoint'], 1, cv2.LINE_AA)
                cv2.circle(image, p, 5 * thin, COLOURS['endpoint'], thin, cv2.LINE_AA)
                cv2.circle(image, f, 5 * thin, COLOURS['floor foot'], thin, cv2.LINE_AA)
            if z[2] > 0:
                cv2.drawMarker(image, _canvas(marks[2], lo, factor)[0], COLOURS['curtain terminal'], cv2.MARKER_CROSS, 14 * thin, thin)
            # Legend band; narrow crops are padded so the text fits (long side stays <= MAX_SIDE).
            canvas = np.full((image.shape[0] + LEGEND_H, max(image.shape[1], 640), 3), 24, np.uint8)
            canvas[:image.shape[0], :image.shape[1]] = image
            text = [f"Photo {photo} | {side['side']} side | {e['objectId']} plane {pi} | node {e['node']}",
                    f"role: {side['role']}",
                    f"member bottom h={e['heightNative']:.4f} native = {_fmt(cm(e['heightNative']))} cm (conditional scale); top h={side['topHeightNative']:.4f}",
                    under_text, pair_text,
                    f"near-curtain samples visible at member line: {len(seen)}/{len(near)} ({len(in_frame)} in frame; span 0..{NEAR} native)"]
            y = image.shape[0] + 16
            for line in text:
                cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .38, (240, 240, 240), 1, cv2.LINE_AA)
                y += 16
            for i, (label, colour) in enumerate(COLOURS.items()):
                x, yy = 8 + (i % 4) * 155, y + 2 + (i // 4) * 16
                cv2.rectangle(canvas, (x, yy - 9), (x + 12, yy + 1), colour, -1)
                cv2.putText(canvas, label, (x + 16, yy), cv2.FONT_HERSHEY_SIMPLEX, .36, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.imwrite(str(out / f'{stem}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
            view.update(crop=f'{stem}.jpg', planeView=f'{stem}-plane.jpg', cropRawXYXY=[*lo.tolist(), *hi.tolist()], cropScale=factor)
            metrics['views'].append(view)
        del rgb
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=1, allow_nan=False) + '\n')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs=4, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.sources, args.out)
    for view in result['views']:
        implied = view.get('imageImpliedRailBottom')
        print(view['photo'], view['side'], view['node'], view['status'], view.get('nearSamplesVisibleAtModelLine'),
              '' if not implied else f"underside {implied['offsetRawPx']:+.0f}px h={implied['rayPlaneHeightNative']:.4f} ({implied['heightCm']:.2f} cm)")
