"""Joint source-track / known-button bundle adjustment, in the saved native gauge.

The source model is a hypothesis: coaxial finite circular red/yellow sections
above a gray housing with a fitted elliptic bottom rim. Section heights, housing dimensions, axis and
yaw are fitted, never copied from display meshes. Exact total axial height and
the two circular diameters enter projection inside the same camera/point fit.
An unsupported fit is saved as a diagnostic candidate, never a metric result.
"""
import hashlib
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from workcell_photo_metrology import FEATURES, _reference


ANGLES = np.arange(16) * (2 * np.pi / 16)
DIRECTIONS = np.c_[np.cos(ANGLES), np.sin(ANGLES)]
ASSUMPTIONS = [
    'The red color silhouette is a complete finite circular cylinder; doming, occlusion and reflections can invalidate this model.',
    'The yellow body is a coaxial frustum with fitted smaller upper radius; its observed lower rim is hypothesized to expose the supplied maximum diameter.',
    'The curved gray housing bottom is hypothesized to be an elliptic rim perpendicular to the common axis; its width, depth and yaw are unmeasured nuisance parameters, not calibrated dimensions.',
    'The detected lower luminance edge is the physical gray housing bottom, excluding its bracket; this identity remains image-derived.',
    'The supplied 10 cm dimension is total axial extent. No silhouette pixel is treated as a projected axis endpoint.',
    'Raw JPEG pixels are square; each photo has a free focal length and fixed source principal point, with zero skew and no fitted distortion.',
    'The first camera pose and one other camera-center coordinate fix only similarity gauge. Source tracks constrain all other camera variables.',
    'All internal section heights and the button axis are fitted. Display mesh part-height fractions never enter the estimator.',
    'Whole-button leave-one-photo-out checks keep only non-button scene tracks from the withheld photo; pixel fit does not certify physical accuracy.',
    'Directions affected by locally rejected contour extrema are excluded in both fit and held-out errors; an inward convex-hull chord is not a measured boundary.',
]


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError('Degenerate direction')
    return v / n


def _basis(axis):
    u = _unit(np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))]))
    return np.c_[u, np.cross(axis, u)]


def _pixels(uv, transform):
    q = np.c_[np.asarray(uv), np.ones(len(uv))] @ transform.T
    return q[:, :2] / q[:, 2:3]


def _project_raw(points, frame):
    q = (np.asarray(points) - frame['pose'][:3, 3]) @ frame['pose'][:3, :3]
    pixels = q @ (np.linalg.inv(frame['A']) @ frame['K']).T
    return pixels[:, :2] / np.maximum(pixels[:, 2:3], 1e-9), q[:, 2]


def _disk_support(center, u, v, radius, frame, directions):
    """Exact perspective support of a circular rim; no sampled raster extrema."""
    R, camera = frame['pose'][:3, :3], frame['pose'][:3, 3]
    q = np.asarray([center - camera, radius * u, radius * v]) @ R
    image = q @ (np.linalg.inv(frame['A']) @ frame['K']).T
    a = image[:, :2] @ directions.T
    b = image[:, 2]
    denominator = max(b[0] ** 2 - b[1] ** 2 - b[2] ** 2, 1e-12)
    middle = a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
    discriminant = np.maximum(middle ** 2 - denominator * (a[0] ** 2 - a[1] ** 2 - a[2] ** 2), 0.)
    return (middle + np.sqrt(discriminant)) / denominator, float(b[0] - np.hypot(b[1], b[2]))


def _supports(frame, shape):
    all_support, depths = [], []
    for color, low, high in (
            ('red', shape['grayHeight'] + shape['yellowHeight'], shape['height']),
            ('yellow', shape['grayHeight'], shape['grayHeight'] + shape['yellowHeight'])):
        radii = [shape[color + 'Radius']] * 2
        if color == 'yellow':
            radii[1] *= shape['yellowTopRadiusFraction']
        rims = [_disk_support(shape['base'] + h * shape['axis'], shape['u'], shape['v'],
                              radius, frame, DIRECTIONS) for h, radius in zip((low, high), radii)]
        all_support.extend(np.maximum(rims[0][0], rims[1][0]))
        depths.extend(r[1] for r in rims)
    gray_support, gray_depth = _disk_support(shape['base'], shape['grayWidth'] / 2 * shape['u'],
                                             shape['grayDepth'] / 2 * shape['v'], 1., frame,
                                             np.array([[0., 1.], [1., 0.], [-1., 0.]]))
    all_support.append(float(gray_support[0]))
    depths.append(gray_depth)
    return np.asarray(all_support), float(min(depths)), np.array([-gray_support[2], gray_support[1]])


def _observed_support(row):
    return np.r_[np.max(np.asarray(row['redHullRaw']) @ DIRECTIONS.T, axis=0),
                 np.max(np.asarray(row['yellowHullRaw']) @ DIRECTIONS.T, axis=0),
                 row['housingBottomSupportRaw']]



def _support_validity(row):
    """Exclude support directions whose extrema were removed as locally obscured.

    The convex hull of remaining vertices creates an inward chord. That chord
    is not an observed object boundary and must not constrain scale or camera.
    The pre-removal hull is used only to locate affected directions, never as
    replacement ground truth for the obscured edge.
    """
    masks = []
    for color in ('red', 'yellow'):
        hull = np.asarray(row[color + 'HullRaw'], float)
        boundary = row.get(color + 'Boundary', {})
        rejected = boundary.get('unsupportedBoundaryVerticesRaw', [])
        valid = np.ones(len(DIRECTIONS), bool)
        if rejected:
            original = np.asarray(boundary.get('graphCutHullRaw'), float)
            points = np.asarray([vertex['pixel'] for vertex in rejected], float)
            if (original.ndim != 2 or original.shape[1] != 2 or len(original) < 3 or
                    points.ndim != 2 or points.shape[1] != 2 or
                    not np.isfinite(original).all() or not np.isfinite(points).all()):
                raise ValueError('Rejected boundary vertices require a finite pre-removal RGB hull')
            before = np.max(original @ DIRECTIONS.T, axis=0)
            after = np.max(hull @ DIRECTIONS.T, axis=0)
            obscured = np.max(points @ DIRECTIONS.T, axis=0)
            valid = ~((before > after + 1e-6) & (obscured >= before - 1e-6))
        kept_angles = ANGLES[valid]
        # Eight distributed directions are the minimum contour coverage here;
        # camera/scale Jacobian rank and held-out projection remain separate gates.
        if len(kept_angles) < 8 or np.max(np.diff(np.r_[kept_angles, kept_angles[0] + 2 * np.pi])) >= np.pi:
            raise ValueError(f'Insufficient supported {color} silhouette directions in photo {row["photo"]}')
        masks.extend(valid)
    return np.r_[np.asarray(masks, bool), True]


def _track_coverage(tracks, photos):
    pairs = {pair: 0 for pair in itertools.combinations(sorted(photos), 2)}
    counts = {p: 0 for p in photos}
    lengths = {str(n): 0 for n in range(2, len(photos) + 1)}
    for track in tracks:
        observed = sorted(o['photo'] for o in track['observations'])
        lengths[str(len(observed))] += 1
        for photo in observed:
            counts[photo] += 1
        for pair in itertools.combinations(observed, 2):
            pairs[pair] += 1
    # Four cameras permit exhaustive graph cuts. Count distinct tracks crossing
    # each cut, not pairwise edges that overcount a three/four-view track.
    cuts = []
    photos = sorted(photos)
    for size in range(len(photos)):
        for extra in itertools.combinations(photos[1:], size):
            side = {photos[0], *extra}
            if len(side) == len(photos):
                continue
            crossing = sum(bool(side & {o['photo'] for o in t['observations']}) and
                           bool(set(photos) - side & {o['photo'] for o in t['observations']}) for t in tracks)
            cuts.append({'photos': sorted(side), 'otherPhotos': sorted(set(photos) - side), 'crossingTracks': crossing})
    weakest = min(cuts, key=lambda c: c['crossingTracks'])
    return {'tracksByPhoto': counts, 'tracksByPair': {f'{a}-{b}': n for (a, b), n in pairs.items()},
            'trackLengthCounts': lengths, 'connected': weakest['crossingTracks'] > 0, 'weakestCut': weakest}


def _select_tracks(tracks, frames, maximum, mode):
    """Keep the historical selector as an explicit controlled experiment arm."""
    if mode not in ('spatial_round_robin', 'connectivity_balanced'):
        raise ValueError('Unknown track selection experiment arm')
    ordered = sorted(tracks, key=lambda t: (-len(t['observations']), t['initialMaxErrorRawPx'], str(t['id'])))

    def cells(track):
        return [(o['photo'], *np.floor(np.asarray(o['uvRaw']) / np.asarray(frames[o['photo']].get('rawShape', [1200, 1600]))[::-1] * 6).astype(int))
                for o in sorted(track['observations'], key=lambda o: o['photo'])]

    kept = []
    # Balancing allocates a limited track budget. With no truncation, preserve
    # the original parameter/residual order and its numerical solver behavior.
    if mode == 'spatial_round_robin' or len(tracks) <= maximum:
        bins = {}
        for track in ordered:
            bins.setdefault(cells(track)[0], []).append(track)
        while bins and len(kept) < maximum:
            for cell in sorted(bins):
                kept.append(bins[cell].pop(0))
                if not bins[cell]:
                    del bins[cell]
                if len(kept) == maximum:
                    break
        return kept
    memberships = [tuple(itertools.combinations(sorted(o['photo'] for o in t['observations']), 2)) for t in ordered]
    locations = [cells(t) for t in ordered]
    available = {pair: {i for i, pairs in enumerate(memberships) if pair in pairs}
                 for pair in itertools.combinations(sorted(frames), 2)}
    capacity = {pair: len(indices) for pair, indices in available.items()}
    pair_counts = dict.fromkeys(available, 0)
    cell_counts = {}
    # ponytail: four photos and a 180-track cap make this bounded greedy pass
    # small. It cannot create missing cross-view tracks; retain capacity evidence.
    while any(available.values()) and len(kept) < maximum:
        pair = min((p for p, ids in available.items() if ids), key=lambda p: (pair_counts[p], capacity[p], p))
        index = min(available[pair], key=lambda i: (-len(ordered[i]['observations']),
                    sum(pair_counts[p] for p in memberships[i]) / len(memberships[i]),
                    sum(cell_counts.get(cell, 0) for cell in locations[i]),
                    ordered[i]['initialMaxErrorRawPx'], str(ordered[i]['id'])))
        kept.append(ordered[index])
        for p in memberships[index]:
            pair_counts[p] += 1
            available[p].remove(index)
        for cell in locations[index]:
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
    return kept


def _prepare(frames, tracks, observations, max_tracks, *, refine_cameras=True, track_selection='spatial_round_robin'):
    if isinstance(max_tracks, bool) or not isinstance(max_tracks, int) or max_tracks <= 0:
        raise ValueError('Track cap must be a positive integer')
    if len(frames) < 2 or set(frames) != set(range(1, len(frames) + 1)):
        raise ValueError('Every source camera of the scene (photos 1..N, N >= 2) is required')
    for frame in frames.values():
        K, A, pose = (np.asarray(frame[k], float) for k in ('K', 'A', 'pose'))
        if K.shape != (3, 3) or A.shape != (3, 3) or pose.shape != (4, 4) or not all(np.isfinite(a).all() for a in (K, A, pose)):
            raise ValueError('Invalid camera or raw/canonical transform')
        if abs(np.linalg.det(A)) < 1e-12 or not np.allclose(A[2], [0, 0, 1]) or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError('Invalid affine pixel transform or rigid pose')
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(pose[:3, :3]) < .99999:
            raise ValueError('Camera rotations must be proper rigid rotations')
        rawK = np.linalg.inv(A) @ K
        if min(rawK[0, 0], rawK[1, 1]) <= 0 or not np.allclose(rawK[[0, 1], [1, 0]], 0, atol=1e-5) or (refine_cameras and not np.isclose(rawK[0, 0], rawK[1, 1], rtol=1e-4)):
            raise ValueError('Joint model requires the raw-square-pixel camera control')
    rows = {int(r['photo']): r for r in observations}
    if len(rows) < 3 or not set(rows).issubset(frames) or len(rows) != len(observations):
        raise ValueError('At least three distinct complete button observations are required')
    for row in observations:
        for key in ('redHullRaw', 'yellowHullRaw'):
            hull = np.asarray(row[key], float)
            if hull.ndim != 2 or hull.shape[1] != 2 or len(hull) < 5 or not np.isfinite(hull).all() or np.linalg.matrix_rank(hull - hull.mean(0)) < 2:
                raise ValueError('Incomplete or degenerate colored silhouette')
        if not np.isfinite(row['housingBottomSupportRaw']) or row.get('occluded', False):
            raise ValueError('Button endpoint is missing or marked occluded')
        if row.get('grayHousingEdgeContrast', 1.) <= 0:
            raise ValueError('Gray housing lower edge has no observed luminance support')
        _support_validity(row)
    if not refine_cameras:
        return [], {'inputTracks': 0, 'selectedTracks': 0, 'tracksByPhoto': {p: 0 for p in frames},
                    'selection': 'Provided cameras fixed; metric support remains conditional on these cameras'}
    selected, rejected = [], {'buttonRoi': 0, 'invalidOrWeakTrack': 0}
    for track in tracks:
        xyz = np.asarray(track.get('xyz'), float)
        obs = track.get('observations', [])
        if xyz.shape != (3,) or not np.isfinite(xyz).all() or len(obs) < 2 or len({o['photo'] for o in obs}) != len(obs) or any(o['photo'] not in frames for o in obs):
            rejected['invalidOrWeakTrack'] += 1
            continue
        clean, errors, rays, in_button = [], [], [], False
        for o in obs:
            frame, row = frames[o['photo']], rows.get(o['photo'])
            uv = np.asarray(o['uv'], float)
            if uv.shape != (2,) or not np.isfinite(uv).all():
                break
            raw = _pixels([uv], np.linalg.inv(frame['A']))[0]
            if row is not None:
                box = np.asarray(row['boxRaw'], float)
                in_button |= bool(np.all(raw >= box[:2] - 3) and np.all(raw <= box[2:] + 3))
            pixels, depth = _project_raw([xyz], frame)
            if depth[0] <= 0:
                break
            errors.append(float(np.linalg.norm(pixels[0] - raw)))
            rays.append(_unit(xyz - frame['pose'][:3, 3]))
            clean.append({'photo': o['photo'], 'uv': uv.tolist(), 'uvRaw': raw.tolist()})
        if in_button:
            rejected['buttonRoi'] += 1
            continue
        if len(clean) != len(obs) or max(errors, default=np.inf) > 8 or max((np.linalg.norm(a - b) for a in rays for b in rays), default=0) < .0087:
            rejected['invalidOrWeakTrack'] += 1
            continue
        selected.append({'id': track.get('id', len(selected)), 'xyz': xyz.tolist(), 'observations': clean,
                         'initialMaxErrorRawPx': max(errors)})
    kept = _select_tracks(selected, frames, max_tracks, track_selection)
    coverage = _track_coverage(kept, frames)
    counts = coverage['tracksByPhoto']
    if min(counts.values()) < 12 or len(kept) < 24:
        raise ValueError(f'Insufficient non-button scene tracks: {counts}')
    return kept, {'inputTracks': len(tracks), 'selectedTracks': len(kept), 'rejected': rejected,
                  'tracksByPhoto': counts, 'maxTracks': max_tracks, 'method': track_selection,
                  'eligibleCoverage': _track_coverage(selected, frames), 'selectedCoverage': coverage,
                  'selectedTrackIds': [t['id'] for t in kept],
                  'selection': 'unchanged button-ROI, positive-depth, parallax and source-reprojection filters; '+track_selection,
                  'scope': 'Track graph support only; connectivity does not validate correspondences, cameras, shape or physical scale.'}


def _button_seed(frames, observations, reference, axis):
    """Training-view-only rough triangulation; never read a saved all-photo button anchor."""
    matrix, rhs, rows = [], [], []
    for row in observations:
        frame = frames[row['photo']]
        hull = np.asarray(row['yellowHullRaw'], float)
        uv = (hull.min(0) + hull.max(0)) / 2
        rawK = np.linalg.inv(frame['A']) @ frame['K']
        ray = _unit(frame['pose'][:3, :3] @ np.linalg.solve(rawK, np.r_[uv, 1.]))
        projector = np.eye(3) - np.outer(ray, ray)
        matrix.append(projector); rhs.append(projector @ frame['pose'][:3, 3])
        rows.append((frame, hull, rawK))
    matrix, rhs = np.concatenate(matrix), np.concatenate(rhs)
    if np.linalg.matrix_rank(matrix) != 3:
        raise ValueError('Training button views do not triangulate a seed')
    center = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    widths = []
    for frame, hull, rawK in rows:
        _, depth = _project_raw([center], frame)
        if depth[0] <= 0:
            raise ValueError('Training button seed is behind a camera')
        widths.append(np.ptp(hull[:, 0]) / rawK[0, 0] * depth[0])
    scale = reference['features'][FEATURES[1]] / float(np.median(widths))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Training button seed has invalid scale')
    return {'baseNative': center - .5 * reference['features'][FEATURES[0]] / scale * axis, 'mPerNative': scale}


def _observability(reduced, core_columns, scale_column):
    """Marginalize all shape nuisances before testing camera/scale identifiability."""
    norms = np.linalg.norm(reduced, axis=0)
    normalized = reduced / np.maximum(norms, 1e-12)
    singular = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-7)) if len(singular) and singular[0] > 0 else 0
    nuisance = [i for i in range(reduced.shape[1]) if i not in core_columns]
    core = normalized[:, core_columns].copy()
    if nuisance:
        nuisance_u, nuisance_s, _ = np.linalg.svd(normalized[:, nuisance], full_matrices=False)
        basis = nuisance_u[:, nuisance_s > max(nuisance_s[0] * 1e-7, 1e-12)]
        core -= basis @ (basis.T @ core)
    _, core_s, core_v = np.linalg.svd(core, full_matrices=False)
    # Absolute threshold matters: a wholly absorbed core must not be considered full rank.
    identifiable = len(core_s) == len(core_columns) and bool(core_s[-1] > max(core_s[0] * 1e-7, 1e-7))
    scale_sigma = None
    if identifiable:
        j = core_columns.index(scale_column)
        scale_sigma = float(np.linalg.norm(core_v[:, j] / core_s) / max(norms[scale_column], 1e-12))
    return {'allParametersIdentifiable': rank == reduced.shape[1],
            'cameraAndScaleIdentifiable': identifiable, 'reducedJacobianRank': rank,
            'reducedJacobianCondition': float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 1e-15 else None,
            'reducedJacobianSingularValues': singular.tolist(), 'cameraScaleSingularValues': core_s.tolist(),
            'localLogScaleSigmaAtTwoRawPixelNoise': scale_sigma,
            'sigmaScope': 'Local linearized independent-residual perturbation only; correlated silhouette supports and shape bias are not modeled.'}


def solve(frames, tracks, observations, reference, initial, *, max_nfev=100, max_tracks=180, holdouts=True, workers=1,
          refine_cameras=True, track_selection='spatial_round_robin'):
    """Small synthetic-friendly core. Observations and tracks are in raw/canonical pixels respectively."""
    reference = _reference(reference)
    known = reference['features']
    scene, selection = _prepare(frames, tracks, observations, max_tracks, refine_cameras=refine_cameras,
                                track_selection=track_selection)
    photos = sorted(frames)
    button_photos = sorted(row['photo'] for row in observations)
    origin = np.asarray(frames[1]['pose'][:3, 3])
    centers = {p: np.asarray(frames[p]['pose'][:3, 3]) for p in photos}
    fixed_photo = max(photos[1:], key=lambda p: np.linalg.norm(centers[p] - origin))
    fixed_axis = int(np.argmax(abs(centers[fixed_photo] - origin)))
    baseline = float(np.linalg.norm(centers[fixed_photo] - origin))
    if baseline < 1e-6:
        raise ValueError('Camera baseline does not fix the scale gauge')
    up = _unit(initial['axisNative'])
    horizontal = _basis(up)
    training_seed = _button_seed(frames, observations, reference, up)
    m0 = float(training_seed['mPerNative'])
    if not np.isfinite(m0) or m0 <= 0:
        raise ValueError('Invalid initial metric scale')
    x0, lower, upper, camera_columns = [], [], [], {}
    for p in photos:
        if not refine_cameras:
            camera_columns[p] = []
            continue
        start = len(x0)
        if p != 1:
            x0.extend([0., 0., 0.]); lower.extend([-np.pi] * 3); upper.extend([np.pi] * 3)
            for axis in range(3):
                if (p, axis) != (fixed_photo, fixed_axis):
                    x0.append((centers[p][axis] - origin[axis]) / baseline)
                    lower.append(-20.); upper.append(20.)
        x0.append(0.); lower.append(-np.log(4)); upper.append(np.log(4))
        camera_columns[p] = list(range(start, len(x0)))
    shape_start = len(x0)
    fractions = np.ones(3)
    width = known['mainBodyDiameterM'] * .8
    depth = known['mainBodyDiameterM'] * .7
    x0.extend([*((np.asarray(training_seed['baseNative']) - origin) / baseline), 0., 0., np.log(m0),
               *np.log(fractions[:2] / fractions[2]), np.log(width), np.log(depth), 0., .7])
    # Ellipse width/depth exchange can represent a quarter-turn only after a
    # discontinuous axis swap. Restricting yaw to +/-45 degrees trapped valid
    # fits at an artificial boundary; allow a continuous physical half-turn.
    lower.extend([-20.] * 3 + [-2.] * 2 + [np.log(m0 / 10)] + [-6.] * 2 + [np.log(known[FEATURES[0]] / 1000)] * 2 + [-np.pi, .02])
    upper.extend([20.] * 3 + [2.] * 2 + [np.log(m0 * 10)] + [6.] * 2 + [np.log(known[FEATURES[0]] * 10)] * 2 + [np.pi, 1.000001])
    global_count = len(x0)
    x0.extend(((np.asarray([t['xyz'] for t in scene]).reshape(-1, 3) - origin) / baseline).ravel())
    lower.extend([-1000.] * (3 * len(scene))); upper.extend([1000.] * (3 * len(scene)))
    x0, lower, upper = map(np.asarray, (x0, lower, upper))
    if not np.isfinite(x0).all() or np.any(x0 <= lower) or np.any(x0 >= upper):
        raise ValueError('Initialization is nonfinite or outside the declared numerical bounds')

    def unpack(x):
        cameras = {}
        for p in photos:
            original = frames[p]
            pose = np.asarray(original['pose']).copy()
            if not refine_cameras:
                cameras[p] = {'photo': p, 'pose': pose, 'K': np.asarray(original['K']).copy(), 'A': original['A']}
                continue
            columns = camera_columns[p]
            j = columns[0]
            if p != 1:
                pose[:3, :3] = original['pose'][:3, :3] @ Rotation.from_rotvec(x[j:j + 3]).as_matrix()
                j += 3
                for axis in range(3):
                    if (p, axis) != (fixed_photo, fixed_axis):
                        pose[axis, 3] = origin[axis] + baseline * x[j]; j += 1
            rawK = np.linalg.inv(original['A']) @ original['K']
            rawK = rawK.copy(); rawK[0, 0] *= np.exp(x[j]); rawK[1, 1] = rawK[0, 0]
            cameras[p] = {'photo': p, 'pose': pose, 'K': original['A'] @ rawK, 'A': original['A']}
        q = x[shape_start:global_count]
        axis = _unit(up + horizontal @ q[3:5])
        # Transport a fixed seed basis continuously; _basis(axis) would jump
        # when the least-aligned coordinate changes during finite differences.
        u = _unit(horizontal[:, 0] - axis * (horizontal[:, 0] @ axis))
        v = np.cross(axis, u)
        u, v = np.cos(q[10]) * u + np.sin(q[10]) * v, -np.sin(q[10]) * u + np.cos(q[10]) * v
        m = np.exp(q[5]); height = known[FEATURES[0]] / m
        weights = np.exp(np.r_[q[6:8], 0.]); heights = height * weights / weights.sum()
        shape = {'base': origin + baseline * q[:3], 'axis': axis, 'u': u, 'v': v,
                 'height': height, 'grayHeight': heights[0], 'yellowHeight': heights[1], 'redHeight': heights[2],
                 'yellowRadius': known[FEATURES[1]] / (2 * m), 'redRadius': known[FEATURES[2]] / (2 * m),
                 'grayWidth': np.exp(q[8]) / m, 'grayDepth': np.exp(q[9]) / m,
                 'yaw': q[10], 'yellowTopRadiusFraction': q[11], 'mPerNative': m}
        return cameras, shape, origin + baseline * x[global_count:].reshape(-1, 3)

    point_rows, track_observations = [], []
    row_count = 0
    for index, track in enumerate(scene):
        point_rows.append(list(range(row_count, row_count + 3 * len(track['observations']))))
        for obs in track['observations']:
            track_observations.append((index, obs))
        row_count += 3 * len(track['observations'])
    scene_groups = {p: [(j, index, o['uvRaw']) for j, (index, o) in enumerate(track_observations) if o['photo'] == p] for p in photos}
    observed = {row['photo']: _observed_support(row) for row in observations}
    support_valid = {row['photo']: _support_validity(row) for row in observations}
    boxes = {row['photo']: np.asarray(row['boxRaw']) for row in observations}

    def residual(x, fit_photos):
        cameras, shape, points = unpack(x)
        scene_values = np.empty((len(track_observations), 3))
        for p, group in scene_groups.items():
            if not group:
                continue
            rows, indices, uv_raw = zip(*group)
            uv, depth = _project_raw(points[list(indices)], cameras[p])
            scene_values[list(rows), :2] = (uv - np.asarray(uv_raw)) / 2.
            scene_values[list(rows), 2] = np.maximum(0., 1e-5 - depth / baseline) * 1000.
        values = scene_values.ravel().tolist()
        for p in fit_photos:
            support, depth, gray_bounds = _supports(cameras[p], shape)
            values.extend((support - observed[p]) * support_valid[p] / 2.)
            values.extend([max(0., boxes[p][0] - gray_bounds[0]) / 2.,
                           max(0., gray_bounds[1] - boxes[p][2]) / 2.,
                           max(0., 1e-5 - depth / baseline) * 1000.])
        return np.asarray(values)

    def fit(fit_photos):
        seed = x0.copy()
        trained = _button_seed(frames, [row for row in observations if row['photo'] in fit_photos], reference, up)
        seed[shape_start:shape_start + 3] = (trained['baseNative'] - origin) / baseline
        seed[shape_start + 5] = np.log(trained['mPerNative'])
        fit_lower, fit_upper = lower.copy(), upper.copy()
        fit_lower[shape_start + 5] = seed[shape_start + 5] - np.log(10)
        fit_upper[shape_start + 5] = seed[shape_start + 5] + np.log(10)
        pattern = lil_matrix((row_count + 36 * len(fit_photos), len(x0)), dtype=int)
        offset = 0
        for index, o in track_observations:
            pattern[offset:offset + 3, camera_columns[o['photo']]] = 1
            pattern[offset:offset + 3, global_count + 3 * index:global_count + 3 * index + 3] = 1
            offset += 3
        for p in fit_photos:
            pattern[offset:offset + 36, camera_columns[p]] = 1
            pattern[offset:offset + 36, shape_start:global_count] = 1
            offset += 36
        result = least_squares(residual, seed, args=(fit_photos,), bounds=(fit_lower, fit_upper),
                               jac_sparsity=pattern.tocsr() if refine_cameras else None,
                               x_scale='jac', max_nfev=max_nfev, ftol=1e-7, xtol=1e-8, gtol=1e-7,
                               tr_options={'atol': 1e-8, 'btol': 1e-8})
        cameras, shape, points = unpack(result.x)
        # Eliminate independently fitted scene XYZ before testing identifiability
        # of camera, scale, axis and unmeasured shape parameters.
        J = result.jac.toarray() if refine_cameras else result.jac
        reduced = J[:, :global_count].copy()
        for index, rows in enumerate(point_rows):
            local = J[np.ix_(rows, range(global_count + 3 * index, global_count + 3 * index + 3))]
            reduced[rows] -= local @ np.linalg.lstsq(local, reduced[rows], rcond=1e-10)[0]
        observability = _observability(reduced, list(range(shape_start)) + [shape_start + 5], shape_start + 5)
        scene_errors, depths = [], []
        for index, o in track_observations:
            uv, depth = _project_raw([points[index]], cameras[o['photo']])
            scene_errors.append(float(np.linalg.norm(uv[0] - o['uvRaw']))); depths.extend(depth)
        feature_errors = {}
        for p in button_photos:
            prediction, depth, _ = _supports(cameras[p], shape)
            error = prediction - observed[p]
            valid = support_valid[p]
            feature_errors[p] = {'redMaxRawPx': float(abs(error[:16][valid[:16]]).max()),
                                 'mainMaxRawPx': float(abs(error[16:32][valid[16:32]]).max()),
                                 'heightEndpointRawPx': float(abs(error[32])),
                                 'maxRawPx': float(abs(error[valid]).max()),
                                 'supportValidity': valid.tolist(),
                                 'validSupportCount': int(valid.sum()),
                                 'excludedSupportIndices': np.flatnonzero(~valid).tolist(),
                                 'maxExcludedErrorRawPx': float(abs(error[~valid]).max()) if np.any(~valid) else None}
            depths.append(depth)
        scale_sigma = observability['localLogScaleSigmaAtTwoRawPixelNoise']
        active_bound = bool(np.any(np.minimum(result.x - fit_lower, fit_upper - result.x) < 1e-5))
        checks = {'finitePositiveDepth': bool(np.isfinite(result.x).all() and min(depths) > 0),
                  'optimizerConverged': bool(result.success), 'cameraAndScaleIdentifiable': observability['cameraAndScaleIdentifiable'],
                  'scaleLocallySupported': scale_sigma is not None and scale_sigma < .25,
                  'numericalBoundsInactive': not active_bound,
                  'sceneReprojectionSupported': not refine_cameras or float(np.percentile(scene_errors, 95)) <= 4.,
                  'fittedButtonSupported': max(feature_errors[p]['maxRawPx'] for p in fit_photos) <= 4.}
        diagnostics = {'checks': checks, 'fitPhotos': list(fit_photos), 'nfev': int(result.nfev), 'message': str(result.message),
                       'cost': float(result.cost), 'initialCost': float(.5 * np.sum(residual(seed, fit_photos) ** 2)),
                       'sceneReprojectionRmsRawPx': float(np.sqrt(np.mean(np.square(scene_errors)))) if scene_errors else None,
                       'sceneReprojectionP95RawPx': float(np.percentile(scene_errors, 95)) if scene_errors else None,
                       'globalParameterCount': global_count, **observability, 'featureErrorsByPhoto': feature_errors,
                       'seedButtonPhotos': list(fit_photos), 'seedMPerNative': trained['mPerNative'],
                       'seedBaseNative': trained['baseNative'].tolist()}
        return {'x': result.x, 'cameras': cameras, 'shape': shape, 'points': points, 'diagnostics': diagnostics,
                'accepted': all(checks.values())}

    print(f'button bundle: full fit with {len(scene)} non-button tracks', flush=True)
    full = fit(button_photos)
    heldout = []
    if holdouts:
        def fold(photo):
            print(f'button bundle: withhold all button observations from photo {photo}', flush=True)
            # Recompute the button seed and scale bounds only from training views.
            # The all-photo anchor and full-fit solution must not initialize a fold.
            result = fit([p for p in button_photos if p != photo])
            error = result['diagnostics']['featureErrorsByPhoto'][photo]
            widths = [np.ptp(np.asarray(row[key])[:, 0]) for row in observations if row['photo'] == photo for key in ('redHullRaw', 'yellowHullRaw')]
            tolerance = min(4., .1 * min(widths))
            accepted = result['accepted'] and error['maxRawPx'] <= tolerance
            return {'photo': photo, 'fitButtonPhotos': [p for p in button_photos if p != photo],
                    'withheldButtonResidualCount': 0, 'cameraRetainedSceneTracks': selection['tracksByPhoto'][photo],
                    'status': 'available' if accepted else 'unsupported', 'candidateMPerNative': float(result['shape']['mPerNative']),
                    'maxErrorRawPx': error['maxRawPx'], 'toleranceRawPx': tolerance, 'featureErrors': error,
                    'diagnostics': result['diagnostics']}
        with ThreadPoolExecutor(max_workers=min(max(1, workers), 2)) as pool:
            heldout = list(pool.map(fold, button_photos))
    valid = full['accepted'] and len(heldout) == len(button_photos) and all(row['status'] == 'available' for row in heldout)
    shape = full['shape']
    scales = [row['candidateMPerNative'] for row in heldout]
    spread = float(np.ptp(scales) / shape['mPerNative']) if scales else None
    valid &= spread is not None and spread <= .15
    reason = None if valid else 'Joint source/shape fit, identifiability, or independent whole-button holdout failed; see diagnostics'
    output = {'schemaVersion': 1, 'status': 'available' if valid else 'unsupported', 'reason': reason,
              'reference': reference, 'knownDimensions': known,
              'mPerNative': float(shape['mPerNative']) if valid else None,
              'candidateMPerNative': float(shape['mPerNative']), 'rangeMPerNative': [min(scales), max(scales)] if valid else None,
              'observations': observations, 'heldOutPhotos': heldout, 'heldOutRelativeScaleRange': spread,
              'referenceGeometryIdentifiable': full['diagnostics']['allParametersIdentifiable'],
              'supportScope': 'Conditional on the stated contour/shape model and supplied cameras; this is not certified physical calibration.',
              'cameraMode': 'joint source-track refinement' if refine_cameras else 'fixed supplied cameras',
              'fittedNuisanceParameters': {k: v.tolist() if isinstance(v, np.ndarray) else float(v) for k, v in shape.items()},
              'exactMetricDimensions': {'wholeComponentHeightM': float(shape['height'] * shape['mPerNative']),
                                        'mainBodyDiameterM': float(2 * shape['yellowRadius'] * shape['mPerNative']),
                                        'redActuatorDiameterM': float(2 * shape['redRadius'] * shape['mPerNative'])},
              'assumptions': ASSUMPTIONS, 'diagnostics': {**full['diagnostics'], 'trackSelection': selection},
              'gauge': {'firstCameraPhoto': 1, 'fixedCenterPhoto': fixed_photo, 'fixedCenterAxis': fixed_axis,
                        'fixedCenterNative': float(centers[fixed_photo][fixed_axis]), 'baselineNative': baseline,
                        'worldFrame': 'MapAnything native'},
              'cameraFile': 'cameras.json', 'tracksFile': 'tracks.json'}
    camera_output = {'worldFrame': 'MapAnything native', 'bundleCandidateStatus': output['status'],
                     'frames': [{'photo': p, 'K': full['cameras'][p]['K'].tolist(), 'pose': full['cameras'][p]['pose'].tolist()} for p in photos]}
    track_output = {'basis': 'same genuine source tracks jointly fitted with known-button dimensions',
                    'tracks': [{**t, 'xyz': point.tolist()} for t, point in zip(scene, full['points'])]}
    return output, camera_output, track_output


def fit_reference_shape(frames, observations, reference, axis, *, max_nfev=100, holdouts=True):
    """Fit the existing exact 3D reference model without changing native cameras.

    Axial height and both circular diameters enter the same perspective model;
    a projected silhouette envelope is never substituted for axial height.
    """
    result, _, _ = solve(frames, [], observations, reference, {'axisNative': axis},
                         max_nfev=max_nfev, holdouts=holdouts, refine_cameras=False)
    camera_inputs = [{'photo': p, **{key: np.asarray(frames[p][key]).tolist() for key in ('K', 'pose', 'A')}}
                     for p in sorted(frames)]
    result.update(camerasFixed=True, cameraFile=None, tracksFile=None,
                  cameraProvenance={'worldFrame': 'unchanged supplied native world',
                                    'framesSha256': hashlib.sha256(json.dumps(camera_inputs, sort_keys=True).encode()).hexdigest(),
                                    'frames': camera_inputs},
                  sourceContourSha256=hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest())
    result['assumptions'] = [line for line in ASSUMPTIONS if not line.startswith(('Raw JPEG', 'The first camera', 'Whole-button'))] + [
        'Provided camera intrinsics and poses remain fixed, including their original pixel transform; their physical accuracy is unverified.',
        'Whole-button leave-one-photo-out fits exclude all button evidence in that photo while retaining the provided cameras.']
    return result


def observe_reference(frame, candidate):
    """Reuse the accepted raw-image contour extraction for either fit route."""
    from workcell_photo_metrology import _color_observation

    red_boundary = _color_observation(frame['rgb'], candidate['boxRaw'], 'red')
    yellow_boundary = _color_observation(frame['rgb'], candidate['boxRaw'], 'yellow')
    red, yellow = (np.asarray(boundary['hullRaw']) for boundary in (red_boundary, yellow_boundary))
    h, w = frame['rgb'].shape[:2]
    if any(np.any(hull[:, 0] <= 1) or np.any(hull[:, 0] >= w - 2) or np.any(hull[:, 1] <= 1) or np.any(hull[:, 1] >= h - 2) for hull in (red, yellow)):
        raise ValueError(f'Button silhouette reaches source-image border in photo {candidate["photo"]}')
    return {'photo': candidate['photo'], 'redHullRaw': red.tolist(), 'yellowHullRaw': yellow.tolist(),
            'redBoundary': red_boundary, 'yellowBoundary': yellow_boundary,
            'boxRaw': candidate['boxRaw'], 'housingBottomSupportRaw': candidate['boxRaw'][3],
            'grayHousingEdgeContrast': candidate['grayHousingEdgeContrast'],
            'endpointMethod': 'automatic strongest downward luminance edge below associated yellow region',
            'occlusionStatus': 'unconfirmed; complete-color and bottom-face hypothesis tested by withheld projection'}


def build(root, out, sources, reference, cameras, tracks, *, max_nfev=100, max_tracks=180, workers=2,
          track_selection='spatial_round_robin'):
    from workcell_photo_metrology import _load

    start = time.monotonic()
    root, out = Path(root), Path(out)
    if root.resolve() == out.resolve():
        raise ValueError('Joint output must not replace the source experiment')
    out.mkdir(parents=True, exist_ok=False)
    reference = _reference(reference)
    evidence = []
    try:
        geometry, _, _, frames, anchors, source_inputs = _load(root, list(map(Path, sources)), cameras)
        anchor = geometry['anchor']
        center = np.asarray(anchor['centerNative'])
        for photo in sorted(frames):
            matches = [a for a in anchors if a['photo'] == photo and np.linalg.norm(a['centerNative'] - center) < 4 * anchor['nativeHeight']]
            if not matches:
                continue
            candidate = min(matches, key=lambda a: np.linalg.norm(a['centerNative'] - center))
            evidence.append(observe_reference(frames[photo], candidate))
        # Retain only the small camera and source-evidence arrays during fits.
        compact = {p: {**{k: f[k] for k in ('K', 'pose', 'A')}, 'rawShape': list(f['rgb'].shape[:2])} for p, f in frames.items()}
        del frames, anchors
        up = _unit(geometry['floor']['normal'])
        initial = {'axisNative': up}
        track_data = json.loads(Path(tracks).read_text()) if isinstance(tracks, (str, Path)) else tracks
        result, camera_data, track_data = solve(compact, track_data['tracks'], evidence, reference, initial,
                                               max_nfev=max_nfev, max_tracks=max_tracks, workers=workers,
                                               track_selection=track_selection)
        source_hash = hashlib.sha256(json.dumps(source_inputs, sort_keys=True).encode()).hexdigest()
        camera_data['sourceHash'] = source_hash
        camera_bytes = (json.dumps(camera_data, indent=2, allow_nan=False) + '\n').encode()
        (out / 'cameras.json').write_bytes(camera_bytes)
        (out / 'tracks.json').write_text(json.dumps(track_data, indent=2, allow_nan=False) + '\n')
        result.update(cameraSha256=hashlib.sha256(camera_bytes).hexdigest(), sourceHash=source_hash,
                      sourceInputs=source_inputs)
    except (ValueError, KeyError, np.linalg.LinAlgError, FloatingPointError) as error:
        result = {'schemaVersion': 1, 'status': 'unsupported', 'reason': f'{type(error).__name__}: {error}',
                  'reference': reference, 'knownDimensions': reference['features'], 'mPerNative': None,
                  'candidateMPerNative': None, 'observations': evidence, 'heldOutPhotos': [],
                  'assumptions': ASSUMPTIONS, 'cameraFile': None, 'tracksFile': None,
                  'diagnostics': {'stage': 'source evidence or joint fit failed before export'}}
    result['wallSeconds'] = time.monotonic() - start
    (out / 'joint-reference.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
