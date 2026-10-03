"""Isolate single-view button-shape residuals without calibrating the scene.

Each photo independently fits the existing 3D reference family with unchanged
intrinsics and source contours. A passing fit is not a metric scene result:
independent object poses deliberately remove cross-view camera constraints.
"""
import argparse
from collections import defaultdict
from itertools import combinations
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from workcell_button_bundle import DIRECTIONS, _disk_support, _observed_support, _supports, _support_validity
from workcell_photo_metrology import FEATURES, _reference


def scene_track_residuals(frames, tracks):
    """Square root Sampson distance in raw pixels; no fitted 3D XYZ is used."""
    errors = defaultdict(list)
    for track in tracks:
        for a, b in combinations(track['observations'], 2):
            i, j = a['photo'], b['photo']
            fi, fj = frames[i], frames[j]
            Ki, Kj = [np.linalg.inv(f['A']) @ f['K'] for f in (fi, fj)]
            relative = np.linalg.inv(fj['pose']) @ fi['pose']
            tx, ty, tz = relative[:3, 3]
            skew = np.array([[0., -tz, ty], [tz, 0., -tx], [-ty, tx, 0.]])
            F = np.linalg.inv(Kj).T @ skew @ relative[:3, :3] @ np.linalg.inv(Ki)
            raw = []
            for obs, frame in ((a, fi), (b, fj)):
                q = np.linalg.inv(frame['A']) @ np.r_[obs['uv'], 1.]
                q /= q[2]
                if 'uvRaw' in obs and not np.allclose(q[:2], obs['uvRaw'], atol=1e-6):
                    raise ValueError('Saved raw and canonical track pixels disagree')
                raw.append(q)
            x, y = raw
            l, m = F @ x, F.T @ y
            denominator = np.sqrt(l[:2] @ l[:2] + m[:2] @ m[:2])
            if not np.isfinite(denominator) or denominator <= 1e-12:
                raise ValueError('Degenerate epipolar geometry')
            errors[tuple(sorted((i, j)))].append(float(abs(y @ l) / denominator))

    def stats(values):
        return {'pairs': len(values), 'medianRawPx': float(np.median(values)),
                'p95RawPx': float(np.percentile(values, 95)), 'maxRawPx': max(values)}
    if not errors:
        raise ValueError('No observed scene track pairs')
    return {'metric': 'sqrt Sampson distance in raw pixels; not 3D distance or an independent camera calibration',
            'all': stats(sum(errors.values(), [])),
            'byPhotoPair': {f'{i}-{j}': stats(v) for (i, j), v in sorted(errors.items())}}


def _shape_supports(frame, shape):
    support, depth, gray_bounds = _supports(frame, shape)
    if 'yellowShoulderHeightFraction' in shape:
        height = shape['grayHeight'] + shape['yellowHeight'] * shape['yellowShoulderHeightFraction']
        radius = shape['yellowRadius'] * shape['yellowShoulderRadiusFraction']
        middle, z = _disk_support(shape['base'] + height * shape['axis'], shape['u'], shape['v'], radius, frame, DIRECTIONS)
        support[16:32] = np.maximum(support[16:32], middle)
        depth = min(depth, z)
    return support, depth, gray_bounds


def fit_photo(frame, observation, reference, seed_shape, *, max_nfev=150,
              shoulder_height=None, independent_seed=None):
    known = _reference(reference)['features']
    K, A, pose = (np.asarray(frame[k], float) for k in ('K', 'A', 'pose'))
    if K.shape != (3, 3) or A.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError('Invalid camera shapes')
    if not all(np.isfinite(v).all() for v in (K, A, pose)) or abs(np.linalg.det(A)) < 1e-12:
        raise ValueError('Invalid camera values')
    local = {'K': K.copy(), 'A': A.copy(), 'pose': np.eye(4)}
    if independent_seed is not None:
        seed_shape = independent_seed
        pose = np.eye(4)
    scale = float(seed_shape['mPerNative'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid seed scale')
    base = (np.asarray(seed_shape['base']) - pose[:3, 3]) @ pose[:3, :3] * scale
    basis = np.c_[seed_shape['u'], seed_shape['v'], seed_shape['axis']]
    basis = pose[:3, :3].T @ basis
    heights = np.array([seed_shape[k] for k in ('grayHeight', 'yellowHeight', 'redHeight')])
    observed, valid = _observed_support(observation), _support_validity(observation)
    box = np.asarray(observation['boxRaw'], float)
    if base[2] <= 0 or not np.isfinite(observed).all() or np.any(heights <= 0):
        raise ValueError('Invalid source observation or shape seed')
    seed = np.r_[base, [0., 0., 0.], np.log(heights[:2] / heights[2]),
                 np.log([seed_shape['grayWidth'] * scale, seed_shape['grayDepth'] * scale]),
                 seed_shape['yellowTopRadiusFraction']]
    # ponytail: this local diagnostic starts from the saved joint candidate;
    # failure is not proof of impossibility. A successful fit is constructive.
    lower = np.r_[[-100., -100., 1e-5], [-np.pi] * 3, [-6.] * 2,
                  [np.log(known[FEATURES[0]] / 1000)] * 2, .02]
    upper = np.r_[[100.] * 3, [np.pi] * 3, [6.] * 2,
                  [np.log(known[FEATURES[0]] * 10)] * 2, 1.000001]
    if shoulder_height is not None:
        # Existing parameters start at the two-ring solution. A near-outer
        # shoulder activates the new ring: placing it inside the old convex
        # hull leaves its support Jacobian zero and cannot test this hypothesis.
        seed = np.r_[seed, shoulder_height, .95]
        lower = np.r_[lower, .02, .000001]
        upper = np.r_[upper, .98, .999999]
    if np.any(seed <= lower) or np.any(seed >= upper):
        raise ValueError('Shape seed is outside the existing family bounds')

    def unpack(x):
        axes = Rotation.from_rotvec(x[3:6]).as_matrix() @ basis
        fractions = np.exp(np.r_[x[6:8], 0.]); fractions /= fractions.sum()
        heights = known[FEATURES[0]] * fractions
        shape = {'base': x[:3], 'u': axes[:, 0], 'v': axes[:, 1], 'axis': axes[:, 2],
                'height': known[FEATURES[0]], 'grayHeight': heights[0],
                'yellowHeight': heights[1], 'redHeight': heights[2],
                'grayWidth': np.exp(x[8]), 'grayDepth': np.exp(x[9]),
                'yellowTopRadiusFraction': x[10], 'redRadius': known[FEATURES[2]] / 2,
                'yellowRadius': known[FEATURES[1]] / 2, 'mPerNative': 1.}
        if shoulder_height is not None:
            shape['yellowShoulderHeightFraction'] = x[11]
            shape['yellowShoulderRadiusFraction'] = x[10] + (1. - x[10]) * x[12]
        return shape

    def residual(x):
        support, depth, gray_bounds = _shape_supports(local, unpack(x))
        return np.r_[(support - observed) * valid / 2.,
                     max(0., box[0] - gray_bounds[0]) / 2.,
                     max(0., gray_bounds[1] - box[2]) / 2.,
                     max(0., 1e-5 - depth) * 1000.]

    start = time.monotonic()
    result = least_squares(residual, seed, bounds=(lower, upper), x_scale='jac',
                           max_nfev=max_nfev, ftol=1e-7, xtol=1e-8, gtol=1e-7)
    shape = unpack(result.x)
    support, depth, _ = _shape_supports(local, shape)
    error = abs(support - observed)
    checks = {'optimizerConverged': bool(result.success),
              'finitePositiveDepth': bool(np.isfinite(result.x).all() and depth > 0),
              'numericalBoundsInactive': bool(np.min(np.minimum(result.x - lower, upper - result.x)) >= 1e-5),
              'sameFourRawPixelFitThreshold': bool(error[valid].max() <= 4.)}
    return {'photo': observation['photo'], 'shapeCanExplainThisPhoto': all(checks.values()),
            'checks': checks, 'maxRawPx': float(error[valid].max()),
            'rmsRawPx': float(np.sqrt(np.mean(error[valid] ** 2))),
            'redMaxRawPx': float(error[:16][valid[:16]].max()),
            'yellowMaxRawPx': float(error[16:32][valid[16:32]].max()),
            'heightEndpointRawPx': float(error[32]), 'nfev': result.nfev,
            'cost': float(result.cost), 'initialCost': float(.5 * np.sum(residual(seed) ** 2)),
            'shapeFamily': 'three-ring monotonic yellow body' if shoulder_height is not None else 'original two-ring yellow frustum',
            'shoulderHeightSeed': shoulder_height,
            'seconds': time.monotonic() - start, 'message': result.message,
            'shapeInIndependentCameraMetres': {k: v.tolist() if isinstance(v, np.ndarray) else float(v) for k, v in shape.items()}}


def diagnose(geometry, *, max_nfev=150, shoulder_ablation=False):
    fit = geometry['anchor']['referenceFit']
    frames = {r['photo']: r for r in fit['cameraProvenance']['frames']}
    results = [fit_photo(frames[r['photo']], r, fit['reference'], fit['fittedNuisanceParameters'], max_nfev=max_nfev)
               for r in fit['observations']]
    if shoulder_ablation:
        for result, row in zip(results, fit['observations']):
            attempts = [fit_photo(frames[row['photo']], row, fit['reference'], fit['fittedNuisanceParameters'],
                                  max_nfev=max_nfev, shoulder_height=h,
                                  independent_seed=result['shapeInIndependentCameraMetres']) for h in (.25, .5, .75)]
            result['shoulderAblation'] = {'selection': 'minimum source residual cost among three deterministic shoulder-height starts; existing parameters unchanged from baseline, new radius progress starts at 0.95; no evaluation truth',
                                         'attempts': attempts, 'best': min(attempts, key=lambda r: r['cost'])}
    return {'schemaVersion': 1, 'status': 'diagnostic_only', 'mPerNative': None,
            'sourceContourSha256': fit['sourceContourSha256'],
            'scope': 'Independent object pose per image, fixed original intrinsics, same colored contours and three known dimensions. Original two-ring yellow family plus optional intermediate-ring ablation. No shared world, no scene scale, no evaluation truth.',
            'interpretation': 'Passing views constructively show this shape family can explain individual images. This does not prove the same physical shape or correct relative cameras. Failure of this local fit is not proof that no solution exists.',
            'jointFeatureErrorsByPhoto': fit['diagnostics']['featureErrorsByPhoto'], 'photos': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-nfev', type=int, default=150)
    parser.add_argument('--tracks', type=Path, help='Optional saved genuine scene tracks for independent epipolar audit')
    parser.add_argument('--shoulder-ablation', action='store_true')
    args = parser.parse_args()
    if args.max_nfev < 1:
        parser.error('--max-nfev must be positive')
    geometry = json.loads(args.geometry.read_text())
    value = diagnose(geometry, max_nfev=args.max_nfev, shoulder_ablation=args.shoulder_ablation)
    if args.tracks:
        frames = {r['photo']: {k: np.asarray(r[k]) for k in ('K', 'pose', 'A')}
                  for r in geometry['anchor']['referenceFit']['cameraProvenance']['frames']}
        value['sourceTrackFile'] = str(args.tracks)
        value['sceneTrackResiduals'] = scene_track_residuals(frames, json.loads(args.tracks.read_text())['tracks'])
    with args.out.open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps([{k: r[k] for k in ('photo', 'shapeCanExplainThisPhoto', 'maxRawPx', 'nfev', 'seconds')} for r in value['photos']]))
