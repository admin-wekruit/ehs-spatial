"""Which physical face of each light-curtain housing does the saved multiview fit use?

P1-1 evidence audit: deterministic, CPU only, no fitting, no optimisation and no
camera change. It reads a frozen run (frames, sam3.json, objects.json), the four
original photos and a saved post-shells fit (post-shells.json plus its GLBs).
For every curtain x photo it writes one JPEG (full housing crop, terminal zooms
and an unrolled strip between the two associated side lines) and metrics.json:
the side-line ids the fit associated, the saved rectangle's residuals against
exactly those lines, the photo's own pointmap plane (normal, width) and where a
dark band inside the yellow housing (lens-strip candidate) lies relative to the
bounded face. Face labels from looking at the crops belong in a separate file.

PYTHONPATH=.:scripts:modal_apps python scripts/workcell_curtain_face_identity.py --root RUN \
    --post-shells G/post-shells --sources image_01.jpg image_02.jpg image_03.jpg image_04.jpg --out NEW_DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageOps
import trimesh

from workcell_photo_geometry import _intersect, _raw_mask
from workcell_photo_metrology import _horizontal, _pixels
from workcell_photo_oneshot import _array, _frame, _response
from workcell_post_faces import _face_errors, _face_projection, _raw_tolerance

# boundariesRaw order written by complete_face_observations: bottom, side group 1, top, side group 0.
NAMES = ('lower_terminal', 'side_1', 'upper_terminal', 'side_0')
BGR = {'fit': (255, 255, 0), 'side_0': (255, 0, 255), 'side_1': (0, 165, 255),
       'lower_terminal': (0, 200, 0), 'upper_terminal': (255, 70, 0), 'other': (150, 150, 150), 'dark': (0, 0, 255)}
T = np.linspace(-1., 2., 241)  # unrolled strip: t=0 image-left side line, t=1 image-right side line
SOURCES = ('workcell_curtain_face_identity.py', 'workcell_post_faces.py', 'workcell_photo_metrology.py',
           'workcell_photo_objects.py', 'workcell_photo_geometry.py', 'workcell_photo_oneshot.py')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unit(vector):
    vector = np.asarray(vector, float)
    return vector / np.linalg.norm(vector)


def _fit_line(points):
    """Total-least-squares image line through raw endpoints: (point, unit direction, image-up)."""
    points = np.asarray(points, float).reshape(-1, 2)
    center = points.mean(0)
    direction = np.linalg.svd(points - center, full_matrices=False)[2][0]
    return center, -direction if direction[1] > 0 else direction


def _meet(line, other):
    (c0, d0), (c1, d1) = line, other
    return c0 + np.linalg.solve(np.column_stack([d0, -d1]), c1 - c0)[0] * d0


def _across(line, frame2d, s):
    """Image-right coordinate u where a line crosses the across-axis row at axial height s."""
    origin, axis, perp = frame2d
    rhs = (np.asarray(line[0]) - origin)[None] - np.outer(np.atleast_1d(s), axis)
    return np.linalg.solve(np.column_stack([perp, -line[1]]), rhs.T)[0]


def _point(frame2d, s, u):
    origin, axis, perp = frame2d
    return origin + np.outer(s, axis) + np.outer(u, perp)


def _runs(flags, join=0.):
    runs, start = [], None
    for index, flag in enumerate(np.r_[flags, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if runs and T[start] - T[runs[-1][1]] <= join:
                runs[-1][1] = index - 1
            else:
                runs.append([start, index - 1])
            start = None
    return runs


def _plane(cloud, camera, up=None):
    """Free plane (SVD, as fit_face) or upright plane (horizontal normal, as _source_initializations)."""
    if up is None:
        center = cloud.mean(0)
        singular, vectors = np.linalg.svd(cloud - center, full_matrices=False)[1:]
        normal = vectors[-1]
    else:
        center, basis = np.median(cloud, axis=0), _horizontal(up)
        singular, vectors = np.linalg.svd((cloud - center) @ basis, full_matrices=False)[1:]
        normal = basis @ vectors[-1]
    normal = normal if normal @ (camera - center) >= 0 else -normal
    return normal, -float(normal @ center), singular


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(_unit(a), _unit(b))), 0, 1))))


def _width(frame, lines, frame2d, s_values, normal, offset):
    """3D distance between the two observed side lines after back-projection onto one plane."""
    ends = []
    for line in lines:
        raw = _point(frame2d, s_values, _across(line, frame2d, s_values))
        ends.append(_intersect(_pixels(raw, frame['A']), frame['K'], frame['pose'], normal, offset))
    a, b = ends
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    direction = _unit(a[-1] - a[0] + b[-1] - b[0])
    delta = b - a
    widths = np.linalg.norm(delta - np.outer(delta @ direction, direction), axis=1)
    return {'medianNative': float(np.median(widths)), 'minNative': float(widths.min()), 'maxNative': float(widths.max()),
            'sideLineAngleDeg': _angle(a[-1] - a[0], b[-1] - b[0])}


def _dark_bands(strip):
    """Yellow housing silhouette and dark vertical bands in face coordinates t (median over unoccluded rows)."""
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV).astype(float)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = (hue >= 15) & (hue <= 40) & (sat >= 90) & (val >= 80)
    rows = yellow.mean(1) >= .1  # a row whose housing is entirely occluded says nothing about the strip
    yellow_value = float(np.median(val[yellow]))
    dark = val < .5 * yellow_value
    yellow_fraction, dark_fraction = yellow[rows].mean(0), dark[rows].mean(0)
    # ponytail: fixed HSV/fraction thresholds tuned on these four photos; a new site needs a look first.
    yellow_runs = _runs(yellow_fraction >= .5, join=.2)
    if not yellow_runs:
        return {'yellowSilhouetteT': None, 'lensStripCandidateT': None, 'lensStripCandidateRelativeToFace': 'not_detected',
                'reason': 'no yellow column between the side lines'}, None
    housing = max(yellow_runs, key=lambda run: (run[0] <= 120 <= run[1], run[1] - run[0]))  # T[120] = 0.5
    interior = [run for run in _runs((dark_fraction >= .2) & (yellow_fraction < .8))
                if run[0] > housing[0] and run[1] < housing[1] and run[1] > run[0]]  # >= 2 columns (~2.5 raw px)
    left = max(0, housing[0] - 30)
    edges = [run for run in _runs(dark_fraction >= .35)
             if (left <= run[1] < housing[0]) or (housing[1] < run[0] <= housing[1] + 30)]
    band = max(interior, key=lambda run: (run[1] - run[0], -run[0]), default=None)
    # Strongest brightness step inside the silhouette (plate junction / shade change), 0.1 t away from its borders.
    value_median, sat_median = np.median(val[rows], 0), np.median(sat[rows], 0)
    smooth = np.convolve(value_median, np.ones(3) / 3, mode='same')
    step = np.abs(np.diff(smooth))
    inside = np.arange(len(step))
    inside = (inside >= housing[0] + 8) & (inside < housing[1] - 8)
    step_index = int(np.argmax(np.where(inside, step, -1))) if inside.any() else None
    if band is None:
        position = 'not_detected'
    else:
        low, high = T[band[0]], T[band[1]]
        position = ('outside' if high < -.08 or low > 1.08 else
                    'inside' if low > .08 and high < .92 else 'on_edge')
    return {'yellowSilhouetteT': [float(T[housing[0]]), float(T[housing[1]])],
            'interiorDarkBandsT': [[float(T[a]), float(T[b])] for a, b in interior],
            'edgeDarkBandsT': [[float(T[a]), float(T[b])] for a, b in edges],
            'lensStripCandidateT': None if band is None else [float(T[band[0]]), float(T[band[1]])],
            'lensStripCandidateRelativeToFace': position,
            'strongestInteriorShadeStepT': None if step_index is None else float((T[step_index] + T[step_index + 1]) / 2),
            'strongestInteriorShadeStepValue': None if step_index is None else float(step[step_index]),
            'rowsUsed': int(rows.sum()), 'rowsSampled': int(len(rows)), 'housingYellowValueMedian': yellow_value,
            'profileT': T[::4].round(3).tolist(), 'yellowFraction': yellow_fraction[::4].round(3).tolist(),
            'darkFraction': dark_fraction[::4].round(3).tolist(), 'valueMedian': value_median[::4].round(1).tolist(),
            'saturationMedian': sat_median[::4].round(1).tolist(),
            'rule': 'yellow: OpenCV H 15-40, S>=90, V>=80; dark: V < 0.5 x housing yellow median; rows with >=10% yellow; '
                    'interior band: dark>=0.2 and yellow<0.8 strictly inside the yellow silhouette; t=0/1 are the image-left/right associated side lines'}, band


def _dashed(canvas, a, b, color, thickness=1, dash=7, gap=5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    for start in np.arange(0., length, dash + gap) if length > 0 else []:
        p, q = a + (b - a) * start / length, a + (b - a) * min(start + dash, length) / length
        cv2.line(canvas, tuple(np.rint(p).astype(int)), tuple(np.rint(q).astype(int)), color, thickness, cv2.LINE_AA)


def _draw(canvas, to_display, layers, dashed):
    for segments, color, thickness in layers:
        for segment in segments:
            a, b = to_display(np.asarray(segment, float).reshape(2, 2))
            if dashed:
                _dashed(canvas, a, b, color, max(1, thickness - 1))
            else:
                cv2.line(canvas, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), color, thickness, cv2.LINE_AA)


def _figure(image, layers, projected, windows, strip, strip_marks, title, legend, bounds):
    height, gap = 1000, 6
    zoom = (height - 2 * gap) // 3  # three 300x300 raw windows, magnified zoom/300
    points = np.concatenate([bounds, projected])
    width = float(np.ptp(projected[:, 0]))
    lo = np.maximum(np.floor(points.min(0) - [max(220, 2 * width), 90]), 0).astype(int)
    hi = np.minimum(np.ceil(points.max(0) + [max(220, 2 * width), 90]), image.shape[1::-1]).astype(int)
    factor = height / (hi[1] - lo[1])
    full = cv2.resize(image[lo[1]:hi[1], lo[0]:hi[0]], None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    _draw(full, lambda p: (p - lo) * factor, layers, False)
    zooms = []
    for name, center in windows:
        w0 = np.rint(center - 150).astype(int)
        crop = np.zeros((300, 300, 3), np.uint8)
        y0, x0 = max(w0[1], 0), max(w0[0], 0)
        part = image[y0:w0[1] + 300, x0:w0[0] + 300]
        crop[y0 - w0[1]:y0 - w0[1] + part.shape[0], x0 - w0[0]:x0 - w0[0] + part.shape[1]] = part
        crop = cv2.resize(crop, (zoom, zoom), interpolation=cv2.INTER_CUBIC)
        _draw(crop, lambda p, w0=w0: (p - w0) * zoom / 300, layers, True)
        cv2.putText(crop, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(crop, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA)
        zooms.append(crop)
    zoom_column = np.concatenate([np.pad(z, ((0, gap if i < 2 else 0), (0, 0), (0, 0))) for i, z in enumerate(zooms)])
    unrolled = cv2.resize(strip, (180, height), interpolation=cv2.INTER_AREA)
    sx = lambda t: (t - T[0]) / (T[-1] - T[0]) * 179
    for t, color in strip_marks['sides']:
        _dashed(unrolled, (sx(t), 0), (sx(t), height - 1), color, 1)
    model = strip_marks['model']
    for column in model:
        pts = np.c_[sx(column), np.linspace(0, height - 1, len(column))][::17]
        for a, b in zip(pts[:-1:2], pts[1::2]):
            cv2.line(unrolled, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), BGR['fit'], 1, cv2.LINE_AA)
    if strip_marks['band'] is not None:
        a, b = (int(round(sx(T[i]))) for i in strip_marks['band'])
        for y in (2, height - 12):
            cv2.rectangle(unrolled, (a, y), (max(b, a + 1), y + 9), BGR['dark'], -1)
    body = np.zeros((height, full.shape[1] + zoom + 180 + 2 * gap, 3), np.uint8)
    body[:, :full.shape[1]] = full
    body[:zoom_column.shape[0], full.shape[1] + gap:full.shape[1] + gap + zoom] = zoom_column
    body[:, -180:] = unrolled
    header = np.full((100, body.shape[1], 3), 24, np.uint8)
    for row, text in enumerate(title):
        cv2.putText(header, text, (6, 17 + 17 * row), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
    x, y = 6, 17 + 17 * len(title) + 4
    for label, color in legend:
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)[0][0]
        if x + 22 + size > header.shape[1] - 4:
            x, y = 6, y + 16
        cv2.line(header, (x, y - 4), (x + 16, y - 4), color, 3)
        cv2.putText(header, label, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (235, 235, 235), 1, cv2.LINE_AA)
        x += 30 + size
    return np.concatenate([header, body])


def audit(root, post_shells, sources, out):
    root, post_shells, out = Path(root), Path(post_shells), Path(out)
    if out.exists():
        raise SystemExit(f'Refusing existing output directory: {out}')
    if len(sources) != 4:
        raise SystemExit('Provide the four original photos in order')
    fit = json.loads((post_shells / 'post-shells.json').read_text())
    saved = {row['photo']: row for row in fit['sourceInputs']}
    up = _unit(fit['ground']['normal'])
    segmentation = json.loads((root / 'sam3.json').read_text())
    catalog = {row['id']: row for row in json.loads((root / 'objects.json').read_text())['objects']}
    report = root / 'scene-report.json'
    sides = {row['objectId']: row.get('side') for row in json.loads(report.read_text())['endpointEstimation']['endpoints']} if report.exists() else {}
    frames, images, camera_rows = {}, {}, []
    for photo, source in enumerate(sources, 1):
        raw = _frame(root, photo)
        K, pose = _array(raw['intrinsics']).astype(float), _array(raw['camera_poses']).astype(float)
        A = np.asarray(raw['input_mask_transform']['input_to_canonical_pixel_centres'], float)
        pairs = ((K, saved[photo]['K']), (pose, saved[photo]['pose']), (A, saved[photo]['inputToCanonicalPixelCentres']))
        camera_rows.append({'photo': photo, 'bitIdentical': all(np.array_equal(a, np.asarray(b, float)) for a, b in pairs),
                            'maxAbsDiffK': float(np.abs(K - saved[photo]['K']).max()),
                            'maxAbsDiffPose': float(np.abs(pose - saved[photo]['pose']).max()),
                            'maxAbsDiffRawToCanonical': float(np.abs(A - saved[photo]['inputToCanonicalPixelCentres']).max())})
        points = _array(raw['pts3d']).astype(float)
        valid = _array(raw['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2)
        frames[photo] = {'photo': photo, 'K': K, 'pose': pose, 'A': A, 'points': points, 'valid': valid, 'shape': valid.shape}
        if _sha(source) != saved[photo]['sha256']:
            raise SystemExit(f'Photo {photo} is not the fit source photo (sha256 differs)')
        image = cv2.cvtColor(np.asarray(ImageOps.exif_transpose(Image.open(source)).convert('RGB')), cv2.COLOR_RGB2BGR)
        if list(image.shape[:2]) != [raw['original_image']['height'], raw['original_image']['width']]:
            raise SystemExit(f'Photo {photo} shape {image.shape[:2]} differs from the recorded original image')
        images[photo] = image
    identical = all(row['bitIdentical'] for row in camera_rows)
    if not identical:
        print('WARNING: saved fit cameras differ from the run cameras; every projection below uses the run cameras', flush=True)
    out.mkdir(parents=True)
    metrics = {'schemaVersion': 1, 'task': 'P1-1 light-curtain face identity evidence audit',
               'scope': 'Image evidence and evaluation of the saved fit only. No fitting, no optimisation, cameras unchanged; '
                        'not physical validation. Units native (mPerNative not applied).',
               'inputs': {'root': str(root), 'postShells': str(post_shells), 'postShellsSha256': _sha(post_shells / 'post-shells.json'),
                          'photos': [{'photo': p, 'path': str(s), 'sha256': saved[p]['sha256']} for p, s in enumerate(sources, 1)],
                          'runFiles': {name: _sha(root / name) for name in ('sam3.json', 'objects.json', 'geometry.json')},
                          'fitSourceFiles': fit.get('sourceFiles'),
                          'code': {name: _sha(Path(__file__).parent / name) for name in SOURCES}},
               'cameraCheck': {'bitIdentical': identical, 'maxAbsDiff': max(max(r['maxAbsDiffK'], r['maxAbsDiffPose'], r['maxAbsDiffRawToCanonical']) for r in camera_rows),
                               'byPhoto': camera_rows, 'projectionCameras': 'current run frames (equal to the saved fit cameras when bitIdentical)'},
               'groundUsed': {'source': 'post-shells.json ground (the fit\'s own)', 'up': up.tolist()},
               'items': []}
    for item in fit['items']:
        ident = item['id']
        corners = np.asarray(item['cornersNative'], float)
        scene = trimesh.load(post_shells / item['model']['file'], force='scene', process=False)
        transform, name = scene.graph[item['model']['nodes'][0]]
        glb = trimesh.transform_points(scene.geometry[name].vertices, transform)
        assert glb.shape == (4, 3) and np.allclose(glb, corners, atol=1e-6, rtol=0), f'{ident}: GLB vertices differ from cornersNative'
        normal = _unit(np.cross(corners[1] - corners[0], corners[3] - corners[0]))
        offset = -float(normal @ corners.mean(0))
        candidates = item['sourceFaceCandidates']
        # Recorded (3D corners, raw corner pixels) pairs: pointmap initialisations of complete observations.
        init_error = 0.
        for initial in item.get('initializations', []):
            row = next(r for r in candidates[str(initial['photo'])] if r['observationId'] == initial['observationId'])
            projected = _face_projection(np.asarray(initial['cornersNative']), frames[initial['photo']])
            init_error = max(init_error, float(np.abs(projected - np.asarray(row['rawCorners'])).max()))
        # Tolerance 1e-3 raw px: the float32 saved rotation is orthonormal only to ~1e-7, and _project uses R^T
        # where back-projection used R, so an exact round trip is ~1e-4 raw px (four orders below the 7.7 px gate).
        if identical:
            assert init_error < 1e-3, f'{ident}: projected initialisation corners miss recorded raw corners by {init_error} px'
        recorded = {row['photo']: row for row in item['fitGate']['exportedModelReprojectionByPhoto']}
        side = sides.get(ident)
        result = {'id': ident, 'reviewerSideInPhoto4': side,
                  'multiviewFace': {'cornersNative': corners.tolist(), 'normal': normal.tolist(),
                                    'widthNative': float(np.mean(np.linalg.norm(corners[[1, 2]] - corners[[0, 3]], axis=1))),
                                    'heightNative': float(np.mean(np.linalg.norm(corners[[3, 2]] - corners[[0, 1]], axis=1))),
                                    'normalAngleFromHorizontalDeg': float(np.degrees(np.arcsin(abs(normal @ up)))),
                                    'orientationMode': item['fitGate']['orientationMode'], 'gateAccepted': item['fitGate']['accepted'],
                                    'modelFile': item['model']['file'], 'glbVerticesEqualCornersNative': True},
                  'consistency': {'initializationCornersMaxRawPx': init_error, 'initializationsChecked': len(item.get('initializations', [])),
                                  'note': 'post-shells.json records no raw pixels for the final fitted rectangle; its projection is '
                                          'proven aligned by reproducing the recorded per-photo residuals (below) from the same '
                                          'projected array that every overlay draws.'},
                  'photos': []}
        for observation in sorted(item['sourceSurfaceObservations'], key=lambda row: row['photo']):
            photo = observation['photo']
            frame, image = frames[photo], images[photo]
            flipped = bool(observation['imageSideOrderFlipped'])
            boundaries = [observation['bottomRawSegments'], observation['fullSideSegmentsRaw'][1],
                          observation['topRawSegments'], observation['fullSideSegmentsRaw'][0]]
            matches = [row for row in candidates[str(photo)] if row['boundariesRaw'] == boundaries]
            assert matches, f'{ident} photo {photo}: associated boundaries not among the saved face candidates'
            row = matches[0]
            projected = _face_projection(corners, frame)
            errors = _face_errors(corners, row, frame, flipped, True, projected=projected)
            lengths = np.linalg.norm(np.concatenate(errors), axis=1)
            rms, worst = float(np.sqrt(np.mean(lengths ** 2))), float(lengths.max())
            if identical:
                assert abs(rms - recorded[photo]['rmsRawPx']) < 1e-3 and abs(worst - recorded[photo]['maxRawPx']) < 1e-3, \
                    f'{ident} photo {photo}: residual {rms:.4f}/{worst:.4f} does not reproduce the recorded fit'
            order = (0, 3, 2, 1) if flipped else (0, 1, 2, 3)
            centroid = projected.mean(0)
            per_boundary = {}
            for edge, source in enumerate(order):
                vectors = errors[edge]
                if not len(vectors):
                    per_boundary[NAMES[source]] = None
                    continue
                a, b = projected[edge], projected[(edge + 1) % 4]
                outward = _unit([a[1] - b[1], b[0] - a[0]])
                outward = outward if outward @ ((a + b) / 2 - centroid) >= 0 else -outward
                size = np.linalg.norm(vectors, axis=1)
                per_boundary[NAMES[source]] = {'modelEdgeCorners': [edge, (edge + 1) % 4], 'points': int(len(vectors)),
                                               'maxRawPx': float(size.max()), 'rmsRawPx': float(np.sqrt(np.mean(size ** 2))),
                                               'meanSignedOutwardRawPx': float(np.mean(vectors @ outward))}
            groups = {'side_0': row['sideGroups'][0], 'side_1': row['sideGroups'][1]}
            lines = {key: _fit_line(group['rawSegments']) for key, group in groups.items()}
            axis = _unit(lines['side_0'][1] + lines['side_1'][1])
            evidence = np.concatenate([np.asarray(b, float).reshape(-1, 2) for b in boundaries if b])
            frame2d = (evidence.mean(0), axis, np.array([-axis[1], axis[0]]))
            s_all = (evidence - frame2d[0]) @ axis
            s_low, s_high = float(s_all.min()), float(s_all.max())
            # Model sides matched to observed sides exactly as _face_errors pairs them.
            model_side = {NAMES[order[1]]: (projected[1], _unit(projected[2] - projected[1])),
                          NAMES[order[3]]: (projected[3], _unit(projected[0] - projected[3]))}
            mid_u = {key: float(_across(lines[key], frame2d, (s_low + s_high) / 2)[0]) for key in lines}
            left_key, right_key = sorted(lines, key=mid_u.get)
            heights = []
            for fraction in (.2, .5, .8):
                s = s_low + fraction * (s_high - s_low)
                u_obs = {key: float(_across(lines[key], frame2d, s)[0]) for key in lines}
                u_mod = {key: float(_across(model_side[key], frame2d, s)[0]) for key in lines}
                center = np.mean(list(u_mod.values()))
                heights.append({'axialFraction': fraction, 'observedWidthRawPx': abs(u_obs['side_1'] - u_obs['side_0']),
                                'modelWidthRawPx': abs(u_mod['side_1'] - u_mod['side_0']),
                                'outwardOffsetRawPx': {key: (u_obs[key] - u_mod[key]) * np.sign(u_mod[key] - center) for key in lines}})
            # Single-view face polygon: observed side lines closed by the observed terminals, or by the observed extent.
            terminal = {'top': _fit_line(boundaries[2]) if boundaries[2] else (_point(frame2d, [s_high], [0])[0], frame2d[2]),
                        'bottom': _fit_line(boundaries[0]) if boundaries[0] else (_point(frame2d, [s_low], [0])[0], frame2d[2])}
            polygon = np.array([_meet(terminal['bottom'], lines[left_key]), _meet(terminal['bottom'], lines[right_key]),
                                _meet(terminal['top'], lines[right_key]), _meet(terminal['top'], lines[left_key])])
            instance = int(re.search(r'instance (\d+)', observation['source'])[1])
            sam_raw = _raw_mask({'rle': [_response(segmentation, photo, 'yellow safety post')['rle'][instance]]}, image.shape[:2])
            canonical = np.zeros(frame['shape'], np.uint8)
            cv2.fillConvexPoly(canonical, np.rint(_pixels(polygon, frame['A'])).astype(np.int32), 1)
            sam = cv2.warpPerspective(sam_raw, frame['A'], frame['shape'][::-1], flags=cv2.INTER_NEAREST)
            support = cv2.erode(canonical & sam, np.ones((3, 3), np.uint8)).astype(bool) & frame['valid']
            cloud = frame['points'][support]
            camera = frame['pose'][:3, 3]
            s_poly = (polygon - frame2d[0]) @ axis
            s_values = np.linspace(s_poly.min(), s_poly.max(), 11)[1:-1]
            view = _unit(corners.mean(0) - camera)
            single = {'supportPoints': int(len(cloud)), 'support': 'one-pixel-eroded observed face polygon x SAM instance x valid pointmap (as fit_face)'}
            if len(cloud) >= 6:
                for mode, plane_up in (('free', None), ('upright', up)):
                    n, d, singular = _plane(cloud, camera, plane_up)
                    ratio = float(singular[-1] / max(singular[-2], 1e-15))
                    centre_ray = _point(frame2d, [(s_poly.min() + s_poly.max()) / 2], [(mid_u[left_key] + mid_u[right_key]) / 2])
                    depth = [_intersect(_pixels(centre_ray, frame['A']), frame['K'], frame['pose'], nn, dd)[0] for nn, dd in ((n, d), (normal, offset))]
                    supported = bool(ratio <= .35 and (mode == 'upright' or singular[1] >= singular[0] * 1e-4))
                    # Grazing rule: every associated face is 70-122 raw px wide and the multiview face is seen at <= 44 deg,
                    # so a plane > 60 deg from its own camera ray is depth-edge bleeding, not a face normal.
                    edge_on = _angle(n, view) > 60
                    single[mode] = {'normal': n.tolist(), 'singularValues': singular.tolist(), 'minorToNextSingularRatio': ratio,
                                    # fit_face rule; for upright it is the horizontal-footprint minor/major ratio.
                                    'planeSupported': supported, 'edgeOnToCamera': bool(edge_on),
                                    'usableForFaceIdentity': bool(supported and not edge_on),
                                    'residualP95Native': float(np.percentile(abs(cloud @ n + d), 95)),
                                    'angleToMultiviewNormalDeg': _angle(n, normal), 'angleToCameraRayDeg': _angle(n, view),
                                    'widthOnOwnPlane': _width(frame, [lines[left_key], lines[right_key]], frame2d, s_values, n, d),
                                    'faceCentreRayDistanceOwnMinusMultiviewNative': _finite(np.linalg.norm(depth[0] - camera) - np.linalg.norm(depth[1] - camera))}
            usable = [mode for mode in ('free', 'upright') if single.get(mode, {}).get('usableForFaceIdentity')]
            single['summary'] = {'usableModes': usable,
                                 'normalAngleToMultiviewDeg': single[usable[0]]['angleToMultiviewNormalDeg'] if usable else None,
                                 'widthOnOwnPlaneNative': (single[usable[0]]['widthOnOwnPlane'] or {}).get('medianNative') if usable else None,
                                 'reason': ('first usable mode' if usable else
                                            'pointmap does not support a plane on this strip (minor/next singular ratio > 0.35 or normal > 60 deg from the camera ray); '
                                            'single-view normal and width are not evidence here')}
            single['widthOnMultiviewPlane'] = _width(frame, [lines[left_key], lines[right_key]], frame2d, s_values, normal, offset)
            single['observedFacePolygonRaw'] = polygon.tolist()
            # Where each associated side line lands across the multiview face (u=0 corner 0, u=1 corner 1) when
            # back-projected onto its plane, also with the plane moved +-0.05 native along its normal. One physical
            # edge should land at one u in every photo; photos 1 and 3 are nearly co-located, so their difference is
            # almost independent of the plane depth.
            axis_u = corners[1] - corners[0] + corners[2] - corners[3]
            width_u = np.linalg.norm(axis_u) / 2
            on_plane = {}
            for key, group in groups.items():
                pts = _pixels(np.asarray(group['rawSegments'], float).reshape(-1, 2), frame['A'])
                values = {}
                for shift in (-.05, 0., .05):
                    xyz = _intersect(pts, frame['K'], frame['pose'], normal, offset - shift)
                    values[f'{shift:+.2f}'] = _finite(np.median((xyz - corners[0]) @ _unit(axis_u)) / width_u)
                on_plane[key] = {'id': group['id'], 'medianUFractionOfFitWidthByPlaneShiftNative': values}
            # Unrolled strip between the two observed side lines over the observed extent (+60 px).
            s_rows = np.linspace(s_high + 60, s_low - 60, 1000)
            u_left, u_right = _across(lines[left_key], frame2d, s_rows), _across(lines[right_key], frame2d, s_rows)
            grid = _point(frame2d, np.repeat(s_rows, len(T)), (u_left[:, None] + T[None] * (u_right - u_left)[:, None]).ravel()).reshape(len(s_rows), len(T), 2)
            strip = cv2.remap(image, grid[..., 0].astype(np.float32), grid[..., 1].astype(np.float32), cv2.INTER_LINEAR)
            lens, band = _dark_bands(strip[(s_rows <= s_high) & (s_rows >= s_low)])
            model_t = [(_across(model_side[key], frame2d, s_rows) - u_left) / (u_right - u_left) for key in (left_key, right_key)]
            others = [g for g in next(v for v in next(x for x in fit['sourceEdgeDiagnostics'] if x['id'] == ident)['views']
                                     if v['photo'] == photo)['bottom'].get('sideSupportGroups', [])
                      if g['id'] not in (groups['side_0']['id'], groups['side_1']['id'])]
            layers = [([s for g in others for s in g['rawSegments']], BGR['other'], 1),
                      ([projected[[i, (i + 1) % 4]] for i in range(4)], BGR['fit'], 1)] + \
                     [(boundaries[i], BGR[NAMES[i]], 2) for i in range(4) if boundaries[i]]
            top_center = np.asarray(boundaries[2], float).reshape(-1, 2).mean(0) if boundaries[2] else projected[[2, 3]].mean(0)
            bottom_center = np.asarray(boundaries[0], float).reshape(-1, 2).mean(0) if boundaries[0] else projected[[0, 1]].mean(0)
            windows = [('upper terminal', top_center), ('middle', (top_center + bottom_center) / 2),
                       ('lower terminal' if boundaries[0] else 'lower: fit only (not observed)', bottom_center)]
            figure_name = f'{ident}-photo-{photo}.jpg'
            ids = f"#{groups['side_0']['id']}/#{groups['side_1']['id']}"
            title = [f"{ident} ({side or '?'}) photo {photo}: {'complete' if row['complete'] else 'partial'} {row['observationId']}, "
                     f"sides {ids}{', flipped' if flipped else ''}",
                     f"saved fit: max {worst:.1f} rms {rms:.1f} raw px (gate {_raw_tolerance(frame):.1f}); "
                     f"width obs/fit {heights[1]['observedWidthRawPx']:.0f}/{heights[1]['modelWidthRawPx']:.0f} px",
                     'full crop | dashed zooms 1.1x | unrolled strip t=-1..2 (red=dark band)']
            legend = [('fit (saved cam)', BGR['fit']), (f"side_0 #{groups['side_0']['id']}", BGR['side_0']),
                      (f"side_1 #{groups['side_1']['id']}", BGR['side_1']), ('lower terminal', BGR['lower_terminal']),
                      ('upper terminal', BGR['upper_terminal']), ('other RGB side groups', BGR['other'])]
            strip_marks = {'sides': [(0., BGR[left_key]), (1., BGR[right_key])], 'model': model_t, 'band': band}
            canvas = _figure(image, layers, projected, windows, strip, strip_marks, title, legend, evidence)
            cv2.imwrite(str(out / figure_name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
            result['photos'].append({
                'photo': photo, 'figure': figure_name, 'samInstance': observation['source'],
                'samInstanceMatchesRunCatalog': any(r['photo'] == photo and r['source'] == observation['source'] for r in catalog[ident]['observations']),
                'association': {'observationIds': [r['observationId'] for r in matches], 'complete': bool(row['complete']),
                                'sideGroupIds': {key: group['id'] for key, group in groups.items()},
                                'sideGroupScope': {key: group.get('supportScope', 'full-height RGB side group') for key, group in groups.items()},
                                'imageLeftSide': left_key, 'imageSideOrderFlipped': flipped,
                                'observedBoundaries': [NAMES[i] for i in range(4) if boundaries[i]],
                                'terminalLocalSideIds': {'top': (row['top'] or {}).get('faceSideIds'), 'bottom': (row['bottom'] or {}).get('faceSideIds')}},
                'residualsOfSavedFit': {'rmsRawPx': rms, 'maxRawPx': worst, 'recordedRmsRawPx': recorded[photo]['rmsRawPx'],
                                        'recordedMaxRawPx': recorded[photo]['maxRawPx'], 'gateRawPx': _raw_tolerance(frame),
                                        'byBoundary': per_boundary,
                                        'meaning': 'distance of every observed boundary endpoint to the saved rectangle edge it was paired with; signed outward > 0 means the observed line lies outside the projected rectangle'},
                'projectedFitCornersRaw': projected.tolist(),
                'pixelWidths': heights,
                'viewAngleToMultiviewNormalDeg': _angle(view, normal),
                'singleViewFace': single,
                'sideLinesOnMultiviewPlane': on_plane,
                'lensStrip': lens})
            print(f"{ident} photo {photo}: max {worst:.2f} rms {rms:.2f} px; width obs/model {heights[1]['observedWidthRawPx']:.0f}/{heights[1]['modelWidthRawPx']:.0f}; "
                  f"single-view free angle {single.get('free', {}).get('angleToMultiviewNormalDeg', float('nan')):.1f}, "
                  f"upright {single.get('upright', {}).get('angleToMultiviewNormalDeg', float('nan')):.1f}; lens {lens['lensStripCandidateRelativeToFace']} {lens['lensStripCandidateT']}", flush=True)
        metrics['items'].append(result)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--post-shells', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs=4, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.post_shells, args.sources, args.out)
