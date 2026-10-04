"""Explicit single-photo button-scale experiment, separate from metric acceptance.

Only the named physical reference and frozen source observations are consumed.
No ground-clearance evaluation value enters this calculation.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from workcell_photo_geometry import _unit
from workcell_photo_metrology import _horizontal, _pixels, _reference, _tangencies
from workcell_photo_oneshot import _array, _frame, scene_photos


def model_measurement_scale(root, geometry):
    """One explicit scale for model tools; never promote a conditional fit.

    The conditional scale reads the button only in this scene's reference photo; a scene whose
    reference photo has no associated button observation keeps native units (no scale is borrowed)."""
    from scripts.workcell_photo_calibration import accepted_scale
    accepted = accepted_scale(geometry)
    if accepted is not None:
        return {'status': 'accepted_3d_reference', 'nativeToMeters': accepted,
                'rangeNativeToMeters': [accepted, accepted],
                'source': '已通过当前固定相机三维参考拟合的按钮标尺。'}
    photo = scene_photos(root)[1]
    observations = geometry['anchor'].get('referenceFit', {}).get('observations', [])
    if not (Path(root) / 'reference-input.json').is_file() or not any(o.get('photo') == photo for o in observations):
        return {'status': 'uncalibrated', 'nativeToMeters': None, 'rangeNativeToMeters': None, 'referencePhoto': photo,
                'source': f'参考照片 {photo} 中没有可用于本报告的按钮轮廓标尺；保留原生单位，不借用其他场景的比例。'}
    evidence = conditional_scale(root, photo=photo)
    primary, red = evidence['primary'], evidence['redCrosscheck']
    return {'status': evidence['status'], 'nativeToMeters': evidence['conditionalMPerNative'],
            'rangeNativeToMeters': evidence['rangeMPerNative'], 'referencePhoto': photo,
            'source': f"照片 {photo} 主体直径 {primary['knownDimensionM'] * 100:g} cm 的条件比例；红帽 {red['knownDimensionM'] * 100:g} cm 交叉检查。三尺寸联合标定仍未通过。",
            'evidence': evidence}


def raw_support(hull, frame):
    """Native inferred pointmap pixels under the accepted raw-image silhouette."""
    hull = np.asarray(hull, float)
    if hull.ndim != 2 or hull.shape[1] != 2 or len(hull) < 3 or not np.isfinite(hull).all():
        raise ValueError('A finite accepted raw-image contour is required')
    mask = np.zeros(frame['valid'].shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(_pixels(hull, frame['A'])).astype(np.int32), 1)
    points = frame['points'][(mask > 0) & frame['valid']]
    if len(points) < 3:
        raise ValueError('Insufficient valid inferred depth under the reference contour')
    return points


def depth_anchored_diameter(tangents, points, frame, up, quantile):
    """Circle tangency plus visible surface depth used as the unknown axis depth.

    Surface-to-axis offset remains a systematic error, outside percentile bounds.
    """
    H = _horizontal(up)
    forward = frame['pose'][:3, 2] - np.dot(frame['pose'][:3, 2], up) * up
    if np.linalg.norm(forward) < 1e-8:
        raise ValueError('Upright circle is unobservable along its axis')
    forward = _unit(forward)
    matrix = np.asarray([[*(r['normal'] @ H), -1.] for r in tangents] + [[*(forward @ H), 0.]])
    rhs = [r['normal'] @ r['camera'] for r in tangents] + [np.percentile(points @ forward, quantile)]
    result = np.linalg.solve(matrix, rhs)
    if result[2] <= 0 or not np.isfinite(result).all():
        raise ValueError('No finite positive conditional reference diameter')
    return float(2 * result[2])


def conditional_scale(root, photo=4):
    root = Path(root)
    geometry = json.loads((root / 'geometry.json').read_text())
    reference = _reference(json.loads((root / 'reference-input.json').read_text()))
    anchor = geometry['anchor']
    observations = anchor['referenceFit']['observations']
    contour_sha = hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest()
    if contour_sha != anchor['referenceFit']['sourceContourSha256']:
        raise ValueError('Frozen reference contour bytes changed')
    observation = next(o for o in observations if o['photo'] == photo)
    raw = _frame(root, photo)
    points = _array(raw['pts3d'])
    frame = {'photo': photo, 'K': _array(raw['intrinsics']), 'pose': _array(raw['camera_poses']),
             'A': np.asarray(raw['input_mask_transform']['input_to_canonical_pixel_centres']),
             'points': points, 'valid': _array(raw['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2)}
    source = next(r for r in json.loads((root / 'geometry-timing.json').read_text())['frameSummaries'] if r['photo'] == photo)
    for actual, expected in [('A', 'inputToCanonicalPixelCentres'), ('K', 'K'), ('pose', 'cameraPose')]:
        if not np.allclose(frame[actual], source[expected]):
            raise ValueError('Native depth camera or pixel transform differs from source provenance')
    normal = np.asarray(geometry['floor']['normal'], float)
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8:
        raise ValueError('Finite inferred ground-normal axis is required')
    up = _unit(normal)
    features = []
    for name, color in [('mainBodyDiameterM', 'yellow'), ('redActuatorDiameterM', 'red')]:
        hull = np.asarray(observation[f'{color}HullRaw'], float)
        support = raw_support(hull, frame)
        tangents = _tangencies(hull, frame, up)
        diameters = [depth_anchored_diameter(tangents, support, frame, up, q) for q in [5, 50, 95]]
        scales = [reference['features'][name] / value for value in diameters]
        features.append({'photo': photo, 'feature': name, 'knownDimensionM': reference['features'][name],
                         'nativeDiameter': diameters[1], 'conditionalMPerNative': scales[1],
                         'rangeMPerNative': [min(scales), max(scales)], 'supportPixelCount': len(support),
                         'rawHullBounds': [hull.min(0).tolist(), hull.max(0).tolist()],
                         'tangentPixelsRaw': [row['uvRaw'].tolist() for row in tangents]})
    primary, red = features
    return {'status': 'conditional_unvalidated', 'acceptedMPerNative': None,
            'conditionalMPerNative': primary['conditionalMPerNative'], 'rangeMPerNative': primary['rangeMPerNative'],
            'method': 'Same-photo upright circular silhouette tangencies; inferred visible surface depth substitutes for unobserved axis depth.',
            'selection': 'Use the larger 8.5 cm body as the reference; independently cross-check the 4 cm actuator. No evaluation clearance is used.',
            'primary': primary, 'redCrosscheck': red,
            'crosscheckDifferenceOverMean': abs(primary['conditionalMPerNative']-red['conditionalMPerNative']) / np.mean([primary['conditionalMPerNative'], red['conditionalMPerNative']]),
            'legacyHeightEnvelope': {'conditionalMPerNative': reference['features']['wholeComponentHeightM'] / anchor['nativeHeight'],
                                     'nativeHeight': anchor['nativeHeight'], 'knownDimensionM': reference['features']['wholeComponentHeightM'],
                                     'scope': 'Historical planar association envelope, not measured axial height; comparison only.'},
            'source': {'photo': photo, 'frameSha256': hashlib.sha256((root / f'frame_{photo:04}.json.gz').read_bytes()).hexdigest(),
                       'recordedPhotoSha256': source['sourceSha256'], 'sourceContourSha256': contour_sha,
                       'referenceInputSha256': hashlib.sha256((root / 'reference-input.json').read_bytes()).hexdigest()},
            'axisNative': up.tolist(), 'rangeScope': '5th/95th percentile surface-depth sensitivity only, not a confidence interval or accuracy bound.',
            'assumptions': ['Circle axis follows the inferred ground normal; this orientation is unverified.',
                            'Full maximum-radius circular silhouette is visible and the named component dimensions correspond to it.',
                            'Raw pointmap depth and native camera geometry are correct locally.',
                            'Visible surface depth is substituted for circle-axis depth; the missing surface-to-axis offset is a primary systematic error.',
                            'The failed three-dimension joint calibration remains unsupported; this result does not validate centimetre measurements.']}


if __name__ == '__main__':
    # Analytic circle with translated/rotated cameras, not a same-frame round trip.
    up, camera, radius, distance = np.array([0., 1., 0.]), np.array([1.2, -.7, 2.]), .2, 4.
    sine = radius / distance
    for angle in [0., .57]:
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        pose = np.eye(4); pose[:3, :3] = R; pose[:3, 3] = camera
        tangents = [{'normal': R @ np.array([sign*np.sqrt(1-sine*sine), 0., sine]), 'camera': camera} for sign in [-1, 1]]
        points = np.array([camera + R @ [0., 0., distance]])
        assert abs(depth_anchored_diameter(tangents, points, {'pose': pose}, up, 50) - 2*radius) < 1e-10
    print('conditional scale synthetic check: PASS')
