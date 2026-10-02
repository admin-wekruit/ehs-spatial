"""Four-photo metrology experiment; source features, never evaluation distances.

All geometry uses MapAnything's native world and canonical camera intrinsics.
Original JPEG pixel centres are mapped with the saved crop/resize affine. SAM
and pointmaps associate source features; RGB edges determine object boundaries.
Replacement cameras always retriangulate edges, circular tangencies and floor
tracks. Their pointmaps are used for pixel association only.
"""
import argparse
import hashlib
import itertools
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError

from workcell_guard_joint import _features, _unit
from workcell_photo_geometry import _anchor_candidates, _consensus, _edge_overlap, _intersect, _raw_mask, _rays
from workcell_photo_objects import _project
from workcell_photo_oneshot import _array, _frame, _mask, _response


TARGETS = ('fence-0', 'post-box-1', 'post-box-2')
FEATURES = ('wholeComponentHeightM', 'mainBodyDiameterM', 'redActuatorDiameterM')


def _reference(value):
    """A deliberate whitelist: no evaluation field can reach the estimator."""
    value = value.get('reference', value)
    dims = {name: value.get('features', {}).get(name) for name in FEATURES}
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for v in dims.values()):
        raise ValueError('Three finite positive named reference dimensions are required')
    return {'objectId': 'emergency-button', 'features': dims,
            'scope': value.get('scope', 'whole component; gray lower housing scope unconfirmed'),
            'scopeStatus': value.get('scopeStatus', 'pending_confirmation')}


def _joint_reference(value, reference, cameras):
    """Bind a joint-fit scale to its exact camera output and named input sizes."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if not isinstance(value, dict) or _reference(value.get('reference', {})) != reference:
        raise ValueError('Joint fit must use the same three reference dimensions and scope')
    if not isinstance(cameras, (str, Path)) or not Path(cameras).is_file():
        raise ValueError('Joint fit requires its exported camera file')
    if value.get('cameraSha256') != hashlib.sha256(Path(cameras).read_bytes()).hexdigest():
        raise ValueError('Joint fit scale and camera bytes disagree')
    scale = value.get('mPerNative')
    if value.get('status') == 'available':
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
            raise ValueError('Supported joint fit requires a positive finite scale')
    elif value.get('status') != 'unsupported' or scale is not None:
        raise ValueError('Unsupported joint fit cannot supply a metric scale')
    return value


def _pixels(uv, transform):
    uv = np.asarray(uv, float).reshape(-1, 2)
    result = np.c_[uv, np.ones(len(uv))] @ np.asarray(transform).T
    return result[:, :2] / result[:, 2:3]


def _horizontal(up):
    axis = np.eye(3)[np.argmin(abs(up))]
    u = _unit(np.cross(up, axis))
    return np.c_[u, np.cross(up, u)]


def _unavailable(reason, **extra):
    return {'status': 'unsupported', 'reason': reason, 'pointNative': None,
            'footNative': None, 'heightNative': None, 'rangeNative': None,
            'sourcePhotos': [], **extra}


def _load(root, sources, cameras):
    geometry = json.loads((root / 'geometry.json').read_text())
    catalog = {o['id']: o for o in json.loads((root / 'objects.json').read_text())['objects']}
    segmentation = json.loads((root / 'sam3.json').read_text())
    if isinstance(cameras, (str, Path)):
        cameras = json.loads(Path(cameras).read_text())
    replacements = {int(row['photo']): row for row in (cameras or {}).get('frames', [])}
    if cameras is not None and (set(replacements) != {1, 2, 3, 4} or cameras.get('worldFrame') != 'MapAnything native'):
        raise ValueError('Replacement cameras must contain four cameras in MapAnything native gauge')
    frames, candidates, metadata = {}, [], []
    for photo, source in enumerate(sources, 1):
        raw = _frame(root, photo)
        image = cv2.imread(str(source))
        if image is None or list(image.shape[:2]) != [raw['original_image']['height'], raw['original_image']['width']]:
            raise ValueError('Original JPEG dimensions must match the recorded transform')
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        A = np.asarray(raw['input_mask_transform']['input_to_canonical_pixel_centres'], float)
        K, pose = _array(raw['intrinsics']), _array(raw['camera_poses'])
        replacement = replacements.get(photo, {})
        new_K, new_pose = np.asarray(replacement.get('K', K), float), np.asarray(replacement.get('pose', pose), float)
        if (A.shape != (3, 3) or not np.isfinite(A).all() or abs(np.linalg.det(A)) < 1e-12 or
                new_K.shape != (3, 3) or new_pose.shape != (4, 4) or
                not np.isfinite(new_K).all() or not np.isfinite(new_pose).all() or
                min(new_K[0, 0], new_K[1, 1]) <= 0 or
                not np.allclose(new_pose[3], [0, 0, 0, 1]) or
                not np.allclose(new_pose[:3, :3].T @ new_pose[:3, :3], np.eye(3), atol=1e-4) or
                np.linalg.det(new_pose[:3, :3]) < .9999):
            raise ValueError('Invalid rigid camera or raw-to-canonical pixel transform')
        factor = min(1., 1800 / max(rgb.shape[:2]))
        small = cv2.resize(rgb, None, fx=factor, fy=factor)
        resize = np.array([[factor, 0, (factor - 1) / 2], [0, factor, (factor - 1) / 2], [0, 0, 1]])
        points = _array(raw['pts3d'])
        valid = _array(raw['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2)
        excluded = np.zeros(valid.shape, bool)
        semantic_masks = {}
        for prompt in segmentation['prompts']:
            if prompt['text'] != 'floor marking':
                semantic_masks[prompt['text']] = _mask(raw, _response(segmentation, photo, prompt['text']))
                excluded |= semantic_masks[prompt['text']]
        frames[photo] = {'photo': photo, 'K': new_K, 'pose': new_pose, 'initialK': K,
                         'initialPose': pose, 'A': A, 'C': resize @ np.linalg.inv(A),
                         'rgb': rgb, 'analysisRgb': small, 'points': points, 'valid': valid,
                         'excluded': excluded, 'semanticMasks': semantic_masks, 'shape': valid.shape}
        candidates.extend(_anchor_candidates(rgb, raw, photo))
        metadata.append({'photo': photo, 'name': source.name,
                         'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'rawShape': list(rgb.shape[:2]), 'canonicalShape': list(valid.shape),
                         'inputToCanonicalPixelCentres': A.tolist(),
                         'K': new_K.tolist(), 'pose': new_pose.tolist()})
    return geometry, catalog, segmentation, frames, candidates, metadata


def _legacy(item, geometry):
    floor = geometry['floor']
    norm = np.linalg.norm(floor['normal'])
    up, offset = np.asarray(floor['normal']) / norm, floor['offset'] / norm
    if item['id'] == 'fence-0':
        plane_index = item.get('geometryPlaneIndex', 0)
        feature = next((row for row in geometry.get('clearances', []) if row['id'] == f'fence-plane-{plane_index}-lower-rail'), None)
        if feature:
            return {'status': 'available', 'reason': None,
                    **{key: feature[key] for key in ('pointNative', 'footNative', 'heightNative', 'sourcePhotos')},
                    'rangeNative': feature['observedViewHeightRangeNative'], 'localFloor': {'normal': up.tolist(), 'offset': offset},
                    'method': 'saved source rail feature and saved floor'}
    rows = []
    for observation in item['observations']:
        corners = np.asarray((observation.get('observedMeasurements', {}).get('basis') or {}).get('corners_native'), float)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            continue
        heights = corners @ up + offset
        height = float(heights.min())
        point = corners[np.isclose(heights, height, atol=1e-6, rtol=0)].mean(0)
        rows.append({'photo': observation['photo'], 'heightNative': height, 'pointNative': point.tolist(),
                     'footNative': (point - height * up).tolist()})
    positive = [row for row in rows if row['heightNative'] >= 0]
    if len(positive) < 2:
        return _unavailable('Saved mask extrema have fewer than two nonnegative views', byPhoto=rows)
    median = np.median([row['heightNative'] for row in positive])
    center = np.median([row['pointNative'] for row in positive], axis=0)
    foot = center - (center @ up + offset) * up
    return {'status': 'available', 'reason': None,
            'pointNative': (foot + median * up).tolist(), 'footNative': foot.tolist(), 'heightNative': float(median),
            'rangeNative': [min(row['heightNative'] for row in positive), max(row['heightNative'] for row in positive)],
            'sourcePhotos': sorted(row['photo'] for row in positive), 'byPhoto': rows,
            'localFloor': {'normal': up.tolist(), 'offset': offset},
            'method': 'median of saved all-valid-mask-point extrema; virtual display endpoint, not a physical edge'}


def _supported_color_hull(rgb, hull):
    """Do not let a locally obscured boundary vertex define an outer tangent."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    kept, rejected = [], []
    for index, point in enumerate(hull):
        before, after = point - hull[index - 1], hull[(index + 1) % len(hull)] - point
        tangent = _unit(_unit(before) + _unit(after))
        normal = np.array([tangent[1], -tangent[0]])
        # Compare a boundary's RGB contrast with adjacent portions of the same
        # edge. A crossing wire can erase that boundary without erasing its
        # neighbors. Exclude the unsupported vertex; do not shift all edges.
        pixels = (point + np.array([-8, -6, -4, 0, 4, 6, 8])[:, None, None] * tangent +
                  np.array([-2, 2])[None, :, None] * normal)
        samples = cv2.remap(lab, pixels[:, :, 0].astype(np.float32), pixels[:, :, 1].astype(np.float32),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        contrast = np.linalg.norm(samples[:, 1] - samples[:, 0], axis=1)
        adjacent = float(np.median(np.r_[contrast[:3], contrast[4:]]))
        if adjacent > 20 and contrast[3] < .5 * adjacent:
            rejected.append({'pixel': point.tolist(), 'edgeContrastLab': float(contrast[3]),
                             'adjacentEdgeContrastLab': adjacent})
        else:
            kept.append(point)
    if len(kept) < 3:
        raise ValueError('Insufficient locally supported RGB boundary vertices')
    return cv2.convexHull(np.asarray(kept, np.float32)).reshape(-1, 2), rejected


def _color_observation(rgb, box, color):
    """Color associates a component; local RGB contrast refines its silhouette.

    The HSV hull alone incorporates connected reflections and background wedges.
    GrabCut reclassifies only a four-raw-pixel boundary band, with an eroded color
    core as foreground. This remains an image-supported silhouette hypothesis,
    not confirmation that the full physical part is visible.
    """
    box = np.asarray(box, float)
    if (color not in ('red', 'yellow') or box.shape != (4,) or not np.isfinite(box).all() or
            np.any(box != np.rint(box))):
        raise ValueError('A named color and finite integer pixel box are required')
    x0, y0, x1, y1 = box.astype(int)
    if not (0 <= x0 < x1 <= rgb.shape[1] and 0 <= y0 < y1 <= rgb.shape[0]):
        raise ValueError('Color box must lie within the original image')
    hsv = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
    mask = ((cv2.inRange(hsv, (0, 100, 90), (12, 255, 255)) |
             cv2.inRange(hsv, (170, 100, 90), (179, 255, 255))) if color == 'red' else
            cv2.inRange(hsv, (16, 85, 80), (42, 255, 255)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise ValueError('No connected color component')
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    yy, xx = np.where(labels == label)
    if len(xx) < 30:
        raise ValueError('Too few component color pixels')
    original_hull = cv2.convexHull(np.c_[xx + x0, yy + y0].astype(np.float32)).reshape(-1, 2)
    # ponytail: a fixed four-pixel raw-image band limits correction to local RGB
    # evidence; larger segmentation errors require new instance observations.
    band = 4
    low = np.maximum([x0 - 2 * band, y0 - 2 * band], 0)
    high = np.minimum([x1 + 2 * band, y1 + 2 * band], rgb.shape[1::-1])
    crop = rgb[low[1]:high[1], low[0]:high[0]]
    component = np.zeros(crop.shape[:2], np.uint8)
    component[yy + y0 - low[1], xx + x0 - low[0]] = 1
    core = cv2.distanceTransform(component, cv2.DIST_L2, 5) > band
    if core.sum() < 10:
        raise ValueError('Color component has insufficient interior for RGB contour refinement')
    labels = np.full(component.shape, cv2.GC_BGD, np.uint8)
    labels[cv2.dilate(component, np.ones((2 * band + 1, 2 * band + 1), np.uint8)) > 0] = cv2.GC_PR_BGD
    labels[core] = cv2.GC_FGD
    # Extraction is sequential per process; fix OpenCV's GMM initialization so
    # a previous ROI or an unrelated OpenCV call cannot change the silhouette.
    cv2.setRNGSeed(0)
    cv2.grabCut(crop, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    refined = (labels & 1).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(refined)
    label = max(range(1, count), key=lambda value: np.count_nonzero(core & (labels == value)))
    yy, xx = np.where(labels == label)
    initial_hull = cv2.convexHull(np.c_[xx, yy].astype(np.float32)).reshape(-1, 2)
    hull, rejected = _supported_color_hull(crop, initial_hull)
    hull += low
    for row in rejected:
        row['pixel'] = (np.asarray(row['pixel']) + low).tolist()
    return {'hullRaw': hull.tolist(), 'colorHullRaw': original_hull.tolist(),
            'graphCutHullRaw': (initial_hull + low).tolist(), 'unsupportedBoundaryVerticesRaw': rejected,
            'method': 'HSV instance association; seeded local RGB GrabCut; convex silhouette support',
            'boundaryBandRawPx': band, 'seedCorePixels': int(core.sum()),
            'colorPixels': int(component.sum()), 'refinedPixels': len(xx),
            'scope': 'visible RGB silhouette hypothesis; occlusion and physical part identity remain unverified'}


def _largest_color(rgb, box, color):
    return np.asarray(_color_observation(rgb, box, color)['hullRaw'], np.float32)


def _tangencies(hull, frame, up):
    """Silhouette support planes tangent to the maximum upright circular section."""
    H = _horizontal(up)
    rays = _rays(_pixels(hull, frame['A']), frame['K'], frame['pose'])
    azimuth = np.arctan2(rays @ H[:, 1], rays @ H[:, 0])
    # Unwrap around the observed cone. A stale depth anchor can lie outside
    # both silhouettes after camera refinement and cannot orient these planes.
    center_angle = np.arctan2(np.sin(azimuth).mean(), np.cos(azimuth).mean())
    relative = np.angle(np.exp(1j * (azimuth - center_angle)))
    if np.ptp(relative) >= np.pi / 2:
        raise ValueError('Circular silhouette crosses the camera azimuth singularity')
    rows = []
    for sign, index in ((1., int(np.argmin(relative))), (-1., int(np.argmax(relative)))):
        normal = sign * _unit(np.cross(up, rays[index]))
        rows.append({'photo': frame['photo'], 'normal': normal, 'camera': frame['pose'][:3, 3],
                     'uvRaw': hull[index], 'uv': _pixels([hull[index]], frame['A'])[0]})
    return rows


def _fit_circle(rows, frames, up):
    H = _horizontal(up)
    matrix = np.asarray([[*(row['normal'] @ H), -1.] for row in rows])
    rhs = np.asarray([row['normal'] @ row['camera'] for row in rows])
    if len({row['photo'] for row in rows}) < 2 or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError('Insufficient distinct camera support for a circle radius')
    fit = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    if fit[2] <= 0 or not np.isfinite(fit).all():
        raise ValueError('Circular tangent fit has no positive radius')
    center, radius = H @ fit[:2], float(fit[2])
    residuals = []
    for row in rows:
        frame = frames[row['photo']]
        # Height is a free nuisance parameter. No internal component fractions.
        horizontal_tangent = center - radius * row['normal']
        ray = _rays([row['uv']], frame['K'], frame['pose'])[0]
        parameter = np.linalg.lstsq(np.c_[ray, -up], horizontal_tangent - row['camera'], rcond=None)[0]
        xyz = horizontal_tangent + parameter[1] * up
        uv, depth = _project([xyz], frame)
        if depth[0] <= 0:
            raise ValueError('Circular support lies behind a camera')
        residuals.append(float(np.linalg.norm(_pixels(uv, np.linalg.inv(frame['A']))[0] - row['uvRaw'])))
    return center, radius, residuals, float(np.linalg.cond(matrix))


def _circle_errors(rows, frames, up, center, radius):
    errors = []
    for row in rows:
        frame = frames[row['photo']]
        # The distance from observed tangent plane to the fitted circle converts
        # to an angular image error in this view, independently of part height.
        ray = _rays([row['uv']], frame['K'], frame['pose'])[0]
        along = np.linalg.lstsq(np.c_[ray, -up], center - radius * row['normal'] - row['camera'], rcond=None)[0]
        xyz = center - radius * row['normal'] + along[1] * up
        uv, depth = _project([xyz], frame)
        errors.append(float(np.linalg.norm(_pixels(uv, np.linalg.inv(frame['A']))[0] - row['uvRaw'])) if depth[0] > 0 else 1e6)
    return errors


def _calibrate(candidates, frames, geometry, reference, up, drawings):
    initial = np.asarray(geometry['anchor']['centerNative'])
    per_photo = {}
    for candidate in candidates:
        if np.linalg.norm(candidate['centerNative'] - initial) > geometry['anchor']['nativeHeight'] * 4:
            continue
        old = per_photo.get(candidate['photo'])
        if old is None or np.linalg.norm(candidate['centerNative'] - initial) < np.linalg.norm(old['centerNative'] - initial):
            per_photo[candidate['photo']] = candidate
    output = []
    for name, color in (('redActuatorDiameterM', 'red'), ('mainBodyDiameterM', 'yellow')):
        rows, views = [], []
        for photo, candidate in per_photo.items():
            try:
                observation = _color_observation(frames[photo]['rgb'], candidate['boxRaw'], color)
                hull = np.asarray(observation['hullRaw'])
                tangents = _tangencies(hull, frames[photo], up)
                rows.extend(tangents)
                views.append({'photo': photo, **observation, 'tangentPixelsRaw': [row['uvRaw'].tolist() for row in tangents]})
                drawings[photo]['standard'].append({'name': color, 'hullRaw': hull.tolist(), 'tangentPixelsRaw': views[-1]['tangentPixelsRaw']})
            except ValueError as error:
                views.append({'photo': photo, 'reason': str(error)})
        result = {'feature': name, 'knownDiameterM': reference['features'][name], 'views': views,
                  'method': 'multiview tangent planes to maximum upright circular cross-section',
                  'assumption': 'Circular section and common upright axis; full maximum-radius silhouette must be visible.',
                  'status': 'unsupported', 'mPerNative': None, 'reason': None}
        try:
            center, radius, errors, condition = _fit_circle(rows, frames, up)
            photos = sorted({row['photo'] for row in rows})
            heldout, scales = [], []
            for photo in photos:
                retained = [row for row in rows if row['photo'] != photo]
                try:
                    c, r, _, _ = _fit_circle(retained, frames, up)
                    e = _circle_errors([row for row in rows if row['photo'] == photo], frames, up, c, r)
                    heldout.append({'photo': photo, 'maxErrorRawPx': max(e)})
                    scales.append(reference['features'][name] / (2 * r))
                except ValueError as error:
                    heldout.append({'photo': photo, 'reason': str(error), 'maxErrorRawPx': None})
            widths = [np.linalg.norm(np.asarray(view['tangentPixelsRaw'])[1] - view['tangentPixelsRaw'][0]) for view in views if 'tangentPixelsRaw' in view]
            normalized = max((v['maxErrorRawPx'] for v in heldout if v['maxErrorRawPx'] is not None), default=1e6) / max(np.median(widths), 1.)
            result.update(nativeDiameter=2 * radius, candidateMPerNative=reference['features'][name] / (2 * radius),
                          centerAxisNative=center.tolist(), axisNative=up.tolist(), condition=condition,
                          maxFitErrorRawPx=max(errors), heldOutPhotos=heldout,
                          heldOutRelativeDiameterError=float(normalized), sourcePhotos=photos,
                          leaveOnePhotoScaleRange=[min(scales), max(scales)] if scales else None)
            if len(photos) < 3 or len(scales) != len(photos):
                result['reason'] = 'At least three photos with independent held-out radius fits are required'
            elif normalized > .12 or condition > 2000:
                result['reason'] = 'Withheld-image tangencies do not support a stable circular diameter'
            else:
                result.update(status='available', mPerNative=result['candidateMPerNative'])
        except (ValueError, np.linalg.LinAlgError) as error:
            result['reason'] = str(error)
        output.append(result)
    supported = [row for row in output if row['status'] == 'available']
    new = {'status': 'unsupported', 'mPerNative': None, 'rangeMPerNative': None,
           'reason': 'No named circular feature passed withheld-image support'}
    if supported:
        values = [row['mPerNative'] for row in supported]
        disagreement = np.ptp(values) / np.mean(values)
        if len(values) == 2 and disagreement > .15:
            new['reason'] = 'The two independent named diameters disagree; neither is selected'
        else:
            selected = min(supported, key=lambda row: row['heldOutRelativeDiameterError'])
            bounds = [value for row in supported for value in row['leaveOnePhotoScaleRange']]
            new.update(status='available', mPerNative=selected['mPerNative'],
                       rangeMPerNative=[min(bounds), max(bounds)], selectedFeature=selected['feature'], reason=None,
                       relativeFeatureDisagreement=float(disagreement),
                       selection='Smallest normalized held-out tangent error among source-supported named circles')
    old = reference['features']['wholeComponentHeightM'] / geometry['anchor']['nativeHeight']
    return {'baseline': {'status': 'conditional', 'mPerNative': old,
                         'nativeHeight': geometry['anchor']['nativeHeight'], 'scopeStatus': reference['scopeStatus'],
                         'method': 'provided whole-component height / saved whole-component native envelope'},
            'new': new, 'candidates': output,
            'selectionReason': new.get('selection', new['reason']),
            'heightScope': 'The 10 cm gray-base extent remains explicit and is not imposed on circle fits.'}


def _sample(mask, uv):
    xy = np.rint(uv).astype(int)
    valid = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
    values = np.zeros(len(xy), bool)
    values[valid] = mask[xy[valid, 1], xy[valid, 0]] > 0
    return values


def _combine_fragments(fragments, A):
    """Same RGB face, same line; missing spans stay missing observations."""
    groups = []
    for fragment in sorted(fragments, key=lambda row: -row['visibleLengthRawPx']):
        chosen = None
        for group in groups:
            if fragment['faceSideIds'] != group[0]['faceSideIds']:
                continue
            points = np.asarray([row['rawEnds'] for row in group + [fragment]]).reshape(-1, 2)
            vx, vy, x, y = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, .01, .01).ravel()
            normal = np.array([-vy, vx])
            if np.max(abs((points - [x, y]) @ normal)) <= 1.5:
                chosen = group
                break
        if chosen is None:
            groups.append([fragment])
        else:
            chosen.append(fragment)
    combined = []
    for group in groups:
        segments = np.asarray([row['rawEnds'] for row in group])
        points = segments.reshape(-1, 2)
        vx, vy, x, y = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, .01, .01).ravel()
        axis, origin = np.asarray([vx, vy], float), np.asarray([x, y], float)
        intervals = _interval_union(sorted(segment @ axis - origin @ axis) for segment in segments)
        visible = sum(b - a for a, b in intervals)
        envelope = origin + np.asarray([intervals[0][0], intervals[-1][1]])[:, None] * axis
        width = max(row['faceWidthRawPx'] for row in group)
        combined.append({**group[0], 'rawEnds': envelope.tolist(), 'uv': _pixels(envelope, A).tolist(),
                         'rawSegments': segments.tolist(), 'uvSegments': [_pixels(segment, A).tolist() for segment in segments],
                         'sourceSegments': [{'segmentIndex': row['sourceSegmentIndex'], 'rawEnds': row['rawEnds']} for row in group],
                         'fragmentCount': len(group), 'lineOriginRaw': origin.tolist(), 'lineDirectionRaw': axis.tolist(),
                         'visibleIntervalsRawPx': intervals,
                         'gapIntervalsRawPx': [[a[1], b[0]] for a, b in zip(intervals, intervals[1:])],
                         'visibleLengthRawPx': float(visible), 'faceWidthRawPx': width,
                         'spanAccepted': bool(visible >= .18 * width),
                         'score': float(np.mean([row['aboveColorFraction'] for row in group]) * min(1., visible / width)),
                         'supportScope': 'Only rawSegments are observed; rawEnds is the collinear envelope and may span gaps.'})
    return combined


def _terminal_edges(rgb, mask, down, photo, A, occluders=None):
    """RGB line at the end of a long colored body, not a depth/order statistic."""
    yy, xx = np.where(mask)
    if len(xx) < 20:
        return [], {'reason': 'No source instance mask'}
    margin = max(12, int((xx.max() - xx.min()) * .35))
    origin = np.maximum([xx.min() - margin, yy.min() - margin], 0)
    high = np.minimum([xx.max() + margin + 1, yy.max() + margin + 1], rgb.shape[1::-1])
    crop = rgb[origin[1]:high[1], origin[0]:high[0]]
    support = mask[origin[1]:high[1], origin[0]:high[0]]
    blocked = (np.zeros_like(support) if occluders is None else
               occluders[origin[1]:high[1], origin[0]:high[0]])
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    yellow = (cv2.inRange(hsv, (16, 65, 65), (45, 255, 255)) > 0) & cv2.dilate(support.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(yellow.astype(np.uint8))
    if count < 2:
        return [], {'reason': 'No yellow housing component'}
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    y, x = np.where(labels == largest)
    pts = np.c_[x, y]
    predicted_down = _unit(np.asarray(down))
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    lines = [] if detected is None else detected.reshape(-1, 2, 2)
    rough_width = max(3., min(cv2.minAreaRect(pts.astype(np.float32))[1]))
    rough_across = np.array([predicted_down[1], -predicted_down[0]])
    side_fragments = []
    for index, line in enumerate(lines):
        vector = line[1] - line[0]
        length = np.linalg.norm(vector)
        if length < 4:
            continue
        direction = vector / length
        if abs(direction @ predicted_down) < .94:
            continue
        samples = line[0] + np.linspace(.1, .9, 11)[:, None] * vector
        adjacent = (_sample(yellow, samples + rough_across * max(2., rough_width * .06)) |
                    _sample(yellow, samples - rough_across * max(2., rough_width * .06)))
        if adjacent.mean() < .5:
            continue
        side_fragments.append((index, line, length))
    # Labels and reflections split LSD side edges. Sum the observed collinear
    # intervals, never the envelope across missing pixels, for the same 4-width
    # axis support requirement. Face width below still needs a local fragment.
    groups = []
    for index, line, length in sorted(side_fragments, key=lambda row: -row[2]):
        matches = []
        for group_index, group in enumerate(groups):
            points = np.concatenate([row[1] for row in group] + [line])
            vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, .01, .01).ravel()
            residual = np.max(abs((points - [x, y]) @ np.array([-vy, vx])))
            if residual <= 1.5:
                matches.append((residual, group_index))
        if matches:
            groups[min(matches)[1]].append((index, line, length))
        else:
            groups.append([(index, line, length)])
    axis_lines, axis_ids, directions, lengths, side_support, observed_lengths = [], [], [], [], [], []
    for group in groups:
        points = np.concatenate([row[1] for row in group])
        vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, .01, .01).ravel()
        axis = np.array([vx, vy])
        intervals = _interval_union(sorted(row[1] @ axis) for row in group)
        visible = sum(b - a for a, b in intervals)
        observed_lengths.append(visible)
        if visible < rough_width * 4:
            continue
        ident = min(row[0] for row in group)
        side_support.append({'id': ident, 'sourceSegmentIndices': [row[0] for row in group],
                             'rawSegments': [(row[1] + origin).tolist() for row in group],
                             'visibleLengthRawPx': visible,
                             'envelopeLengthRawPx': intervals[-1][1] - intervals[0][0]})
        directions.append(axis); lengths.append(visible)
        for _, line, _ in group:
            axis_lines.append((line + origin).tolist()); axis_ids.append(ident)
    if not axis_lines:
        return [], {'reason': 'No long RGB side edge establishes the housing image axis',
                    'detectedRgbLines': len(lines), 'predictedDownRaw': predicted_down.tolist(),
                    'sideFragmentCount': len(side_fragments),
                    'requiredObservedSideLengthRawPx': float(rough_width * 4),
                    'largestObservedSideLengthRawPx': max(observed_lengths, default=0.)}
    covariance = sum(length * np.outer(direction, direction) for length, direction in zip(lengths, directions))
    down = np.linalg.eigh(covariance)[1][:, -1]
    down *= 1 if down @ predicted_down >= 0 else -1
    across = np.array([down[1], -down[0]])
    low, high_side = np.min(pts @ across), np.max(pts @ across)
    width = max(3., high_side - low)
    # ponytail: connected-component filtering removes isolated color specks.
    # Occluded body pieces remain only when collinear and larger than a cap pixel.
    retained = np.zeros_like(yellow)
    for label in range(1, count):
        y, x = np.where(labels == label)
        coordinates = np.c_[x, y]
        if len(x) >= max(20, width * width * .12) and low - width * .2 <= np.median(coordinates @ across) <= high_side + width * .2:
            retained |= labels == label
    y, x = np.where(retained)
    body = np.c_[x, y]
    if len(body) == 0 or np.ptp(body @ down) < width * 4:
        return [], {'reason': 'Color support is not a long housing'}
    termination = np.max(body @ down)
    # Short endcap sides can continue an otherwise long housing edge. Their
    # endpoint contact distinguishes the outer end from an internal color seam.
    side_ends = []
    for line in lines:
        vector = line[1] - line[0]
        length = np.linalg.norm(vector)
        if length < 4 or abs((vector / length) @ down) < .94:
            continue
        if not low - width * .3 <= line.mean(0) @ across <= high_side + width * .3:
            continue
        samples = line[0] + np.linspace(.1, .9, 7)[:, None] * vector
        adjacent = np.logical_or.reduce([_sample(retained, samples + across * side * max(2., width * .06) - down * height)
                                         for side in (-1, 1) for height in (0., width * .5)])
        if adjacent.mean() >= .4:
            side_ends.append(line[np.argmax(line @ down)])
    candidates, rejected, terminal_witnesses = [], [], []
    counts = {key: 0 for key in ('tooShort', 'outsideBodyAcross', 'notTransverse', 'span',
                                 'notNearColorEnd', 'insufficientColorAbove', 'colorContinuesBelow',
                                 'occluded', 'noSideEndContact', 'lowerColorContinuation',
                                 'innerEndcapSeam', 'samMaskContinuesBelow', 'faceWidthUnsupported',
                                 'insufficientVisibleSpan', 'combinedCollinearFragments', 'accepted')}
    def reject(reason, line, near=False):
        counts[reason] += 1
        if near:
            rejected.append({'rawEnds': (line + origin).tolist(), 'reason': reason})
    for segment_index, line in enumerate(lines):
        vector = line[1] - line[0]
        length = np.linalg.norm(vector)
        if length < 4:
            reject('tooShort', line)
            continue
        direction, center = vector / length, line.mean(0)
        side = float(center @ across)
        if not low - .3 * width <= side <= high_side + .3 * width:
            reject('outsideBodyAcross', line)
            continue
        near_end = abs(center @ down - termination) <= width * .65
        if abs(direction @ down) > .5:
            reject('notTransverse', line, near_end)
            continue
        if length > 2.2 * width:
            reject('span', line, near_end)
            continue
        if not near_end:
            reject('notNearColorEnd', line)
            continue
        # Width belongs to this RGB face, not the connected yellow union of
        # front and rear flanges. Locally observed side fragments must bound it;
        # a long group's unobserved gap cannot supply a virtual local side.
        side_positions = []
        axial = float(center @ down)
        for ident, raw_side in zip(axis_ids, axis_lines):
            edge = np.asarray(raw_side) - origin
            extent = sorted(edge @ down)
            if axial < extent[0] - width * .65 or axial > extent[1] + width * .65:
                continue
            delta = edge[1] - edge[0]
            intersection = edge[0] + delta * ((axial - edge[0] @ down) / (delta @ down))
            side_positions.append((float(intersection @ across), ident, raw_side))
        span = sorted(line @ across)
        left = max((row for row in side_positions if row[0] <= span[0] + 1.5), default=None, key=lambda row: row[0])
        right = min((row for row in side_positions if row[0] >= span[1] - 1.5), default=None, key=lambda row: row[0])
        if left is None or right is None or right[0] - left[0] < 4:
            reject('faceWidthUnsupported', line, True)
            continue
        face_width = right[0] - left[0]
        face_ids = [left[1], right[1]]
        samples = line[0] + np.linspace(.12, .88, 9)[:, None] * vector
        above = max(float(_sample(retained, samples - down * width * distance).mean()) for distance in (.12, .3, .6))
        below = float(_sample(retained, samples + down * width * .35).mean())
        outside = float(_sample(support, samples + down * max(3., width * .08)).mean())
        if outside > .33:
            counts['samMaskContinuesBelow'] += 1
        if above < .44:
            reject('insufficientColorAbove', line, True)
            continue
        if below > .22:
            reject('colorContinuesBelow', line, True)
            continue
        obstruction = float((_sample(blocked, samples) |
                             _sample(blocked, samples + down * max(3., width * .08))).mean())
        contacts = []
        for endpoint in line:
            for side_end in side_ends:
                delta = endpoint - side_end
                gap = float(delta @ down)
                if -4. <= gap <= width * .65:
                    contacts.append((abs(float(delta @ across)), gap))
        side_contact, side_gap = min(contacts, default=(1e6, 1e6))
        if side_contact > max(4., width * .15):
            reject('noSideEndContact', line, True)
            continue
        # A same-instance yellow continuation below an occluding paper edge
        # disqualifies that edge as a body termination.
        continuation = body[(body @ down > center @ down + width) &
                            (abs(body @ across - side) < width * .65)]
        if len(continuation) > width * width * .12:
            reject('lowerColorContinuation', line, True)
            continue
        raw = line + origin
        terminal_witnesses.append({'rawEnds': raw.tolist(), 'position': float(center @ down), 'faceSideIds': face_ids})
        if obstruction > .22:
            reject('occluded', line, True)
            continue
        candidates.append({'photo': photo, 'rawEnds': raw.tolist(), 'uv': _pixels(raw, A).tolist(),
                           'sourceSegmentIndex': segment_index, 'faceSideIds': face_ids,
                           'faceWidthRawPx': float(face_width), 'faceSideEdgesRaw': [left[2], right[2]],
                           'faceWidthBasis': 'locally observed fragments of supported collinear RGB sides at this terminal segment',
                           'aboveColorFraction': above, 'belowColorFraction': below,
                           'belowInstanceFraction': outside,
                           'obstructionFraction': obstruction,
                           'sideEndContactRawPx': side_contact,
                           'sideEndAxialGapRawPx': side_gap,
                           'visibleLengthRawPx': float(length), 'bodyWidthRawPx': float(width),
                           'score': above * min(1., length / width), 'position': float(center @ down)})
    outer = []
    for candidate in candidates:
        edge = np.asarray(candidate['rawEnds']) - origin
        span = sorted(edge @ across)
        inner_seam = False
        # An occluded outer end still proves an upper color seam is internal;
        # it must not make that seam become an accepted shorter housing.
        for other in terminal_witnesses:
            if other['faceSideIds'] != candidate['faceSideIds']:
                continue
            separation = other['position'] - candidate['position']
            if not 2 < separation < width * .65:
                continue
            other_edge = np.asarray(other['rawEnds']) - origin
            other_span = sorted(other_edge @ across)
            overlap = _edge_overlap(edge, other_edge, across)[0]
            if overlap > .7 * min(np.ptp(span), np.ptp(other_span)):
                inner_seam = True
                break
        if inner_seam:
            reject('innerEndcapSeam', edge, True)
        else:
            outer.append(candidate)
    candidates = []
    for row in _combine_fragments(outer, A):
        counts['combinedCollinearFragments'] += row['fragmentCount'] - 1
        if row['spanAccepted']:
            candidates.append(row)
        else:
            for segment in row['rawSegments']:
                reject('insufficientVisibleSpan', np.asarray(segment) - origin, True)
    candidates.sort(key=lambda row: (-row['score'], -row['position']))
    counts['accepted'] = len(candidates)
    return candidates[:6], {'candidateCount': len(candidates), 'longSideEdgesRaw': axis_lines,
                            'sideSupportGroups': side_support,
                            'sideFragmentCount': len(side_fragments),
                            'requiredObservedSideLengthRawPx': float(rough_width * 4),
                            'detectedRgbLines': len(lines), 'rejectionCounts': counts,
                            'nearEndRejectedEdges': rejected,
                            'obstructedTerminationsRejected': counts['occluded'],
                            'predictedDownRaw': predicted_down.tolist(), 'fittedDownRaw': down.tolist(),
                            'axisDeviationDeg': float(np.degrees(np.arccos(np.clip(down @ predicted_down, -1, 1)))),
                            'axisSource': 'collinear visible RGB side fragments adjacent to the same yellow housing; gaps excluded from support length',
                            'widthRawPx': float(width), 'sourceColorPixels': len(body),
                            'scope': 'visible yellow housing end edge; occluded width remains unknown'}


def _interval_union(intervals):
    result = []
    for low, high in sorted(intervals):
        if high <= low:
            continue
        if result and low <= result[-1][1]:
            result[-1][1] = max(result[-1][1], float(high))
        else:
            result.append([float(low), float(high)])
    return result


def _line_fit(rows, frames, up):
    normals, offsets = [], []
    for row in rows:
        frame = frames[row['photo']]
        rays = _rays(row['uv'], frame['K'], frame['pose'])
        normal = _unit(np.cross(*rays))
        normals.append(normal)
        offsets.append(float(normal @ frame['pose'][:3, 3]))
    M, rhs = np.asarray(normals), np.asarray(offsets)
    H = _horizontal(up)
    _, _, vt = np.linalg.svd(M @ H, full_matrices=True)
    initial_axis = H @ vt[-1]
    across = _unit(np.cross(initial_axis, up))
    design = M @ np.c_[across, up]
    if np.linalg.matrix_rank(design) < 2 or np.linalg.cond(design) > 300:
        raise ValueError('Viewing planes do not constrain a stable horizontal edge')
    transverse = np.linalg.lstsq(design, rhs, rcond=None)[0]
    raw_cameras, observed = [], []
    for row in rows:
        frame = frames[row['photo']]
        raw_cameras.append((np.linalg.inv(frame['A']) @ frame['K'], frame['pose']))
        observed.append(np.c_[np.asarray(row.get('rawSegments', [row['rawEnds']])).reshape(-1, 2),
                              np.ones(2 * len(row.get('rawSegments', [row['rawEnds']])))])
    def unpack(parameters):
        axis = H @ np.array([np.cos(parameters[0]), np.sin(parameters[0])])
        # The origin-nearest point fixes the line's free along-axis coordinate.
        # An old object/depth anchor never enters the noisy geometric solve.
        point = np.cross(axis, up) * parameters[1] + up * parameters[2]
        return axis, point
    def residual(parameters):
        axis, point = unpack(parameters)
        errors = []
        for (K, pose), pixels in zip(raw_cameras, observed):
            p = K @ ((point - pose[:3, 3]) @ pose[:3, :3])
            direction = K @ (axis @ pose[:3, :3])
            line = np.cross(p, direction)
            errors.extend(pixels @ line / max(np.linalg.norm(line[:2]), 1e-12))
        return np.asarray(errors)
    parameters = np.r_[np.arctan2(vt[-1, 1], vt[-1, 0]), transverse]
    fit = least_squares(residual, parameters, loss='soft_l1', f_scale=1.5, max_nfev=150)
    if not fit.success or np.linalg.matrix_rank(fit.jac) < 3:
        raise ValueError('Horizontal RGB line fit is unconstrained or did not converge')
    axis, point = unpack(fit.x)
    design = M @ np.c_[np.cross(axis, up), up]
    if np.linalg.matrix_rank(design) < 2 or np.linalg.cond(design) > 300:
        raise ValueError('Optimized viewing planes do not constrain a stable edge')
    intervals, errors = [], []
    for row in rows:
        frame = frames[row['photo']]
        visible, error = [], []
        for segment in row.get('uvSegments', [row['uv']]):
            spans = []
            for uv, ray in zip(segment, _rays(segment, frame['K'], frame['pose'])):
                values = np.linalg.lstsq(np.c_[ray, -axis], point - frame['pose'][:3, 3], rcond=None)[0]
                if values[0] <= 0:
                    raise ValueError('Triangulated edge is behind a camera')
                projected, _ = _project([point + values[1] * axis], frame)
                error.append(float(np.linalg.norm(_pixels(projected, np.linalg.inv(frame['A']))[0] - _pixels([uv], np.linalg.inv(frame['A']))[0])))
                spans.append(float(values[1]))
            visible.append(sorted(spans))
        intervals.append(_interval_union(visible))
        errors.append(max(error))
    overlaps = [(i, j, [max(a[0], b[0]), min(a[1], b[1])])
                for i, j in itertools.combinations(range(len(intervals)), 2)
                for a in intervals[i] for b in intervals[j]
                if max(a[0], b[0]) < min(a[1], b[1])]
    shared_intervals = _interval_union(span for _, _, span in overlaps)
    if not shared_intervals:
        raise ValueError('No common visible segment between two source views')
    # Infinite-line agreement cannot make a distant visible fragment support
    # this edge. Every counted view must join the same finite-overlap group.
    connected = {0}
    for _ in intervals:
        reached = connected | {j for i, j, _ in overlaps if i in connected} | {i for i, j, _ in overlaps if j in connected}
        if reached == connected:
            break
        connected = reached
    if len(connected) != len(intervals):
        raise ValueError('No common visible segment connects every source view')
    shared = max(shared_intervals, key=lambda span: span[1] - span[0])
    all_visible = _interval_union(span for view in intervals for span in view)
    full = [all_visible[0][0], all_visible[-1][1]]
    visible_native = {str(row['photo']): [[(point + s * axis).tolist() for s in span] for span in view]
                      for row, view in zip(rows, intervals)}
    shared_native = [[(point + s * axis).tolist() for s in span] for span in shared_intervals]
    gaps_native = [[(point + s * axis).tolist() for s in (a[1], b[0])] for a, b in zip(all_visible, all_visible[1:])]
    point += np.mean(shared) * axis
    return {'pointNative': point.tolist(), 'endsNative': [(point + (value - np.mean(shared)) * axis).tolist() for value in full],
            'sharedEndsNative': [(point + (value - np.mean(shared)) * axis).tolist() for value in shared],
            'sharedSegmentsNative': shared_native, 'visibleSegmentsNativeByPhoto': visible_native,
            'unobservedGapSegmentsNative': gaps_native,
            'envelopeScope': 'endsNative spans the fitted line envelope; only listed visible/shared segments are observed',
            'method': 'three-parameter horizontal line fit to raw visible-endpoint projection-line residuals; origin-nearest gauge',
            'axisNative': axis.tolist(), 'reprojectionErrorsRawPx': errors,
            'maxReprojectionErrorRawPx': max(errors), 'condition': float(np.linalg.cond(design)),
            'sourcePhotos': sorted({row['photo'] for row in rows}), 'observations': rows}


def _match_edges(candidates, frames, up, lowest=False):
    fits = []
    for a, b in itertools.combinations(candidates, 2):
        if a['photo'] == b['photo']:
            continue
        try:
            fitted = _line_fit([a, b], frames, up)
            if fitted['maxReprojectionErrorRawPx'] > 3.:
                continue
            selected = [a, b]
            for photo in sorted(set(row['photo'] for row in candidates) - {a['photo'], b['photo']}):
                additions = []
                for candidate in candidates:
                    if candidate['photo'] != photo:
                        continue
                    try:
                        trial = _line_fit(selected + [candidate], frames, up)
                        if trial['maxReprojectionErrorRawPx'] <= 3.:
                            additions.append(trial)
                    except (ValueError, np.linalg.LinAlgError):
                        pass
                if additions:
                    fitted = min(additions, key=lambda row: row['maxReprojectionErrorRawPx'])
                    selected = fitted['observations']
            fits.append(fitted)
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not fits:
        raise ValueError('No common RGB end edge has stable multiview reprojection support')
    if lowest:
        fitted = min(fits, key=lambda row: (np.asarray(row['pointNative']) @ up, -len(row['sourcePhotos']), row['maxReprojectionErrorRawPx']))
    else:
        fitted = min(fits, key=lambda row: (-len(row['sourcePhotos']), row['maxReprojectionErrorRawPx'], -sum(o.get('score', 1.) for o in row['observations'])))
    # A pixel sensitivity envelope is not a confidence interval or camera bound.
    sensitivity = np.asarray(fitted['sharedSegmentsNative']).reshape(-1, 3).tolist()
    for i, observation in enumerate(fitted['observations']):
        raw = np.asarray(observation['rawEnds'])
        transverse = _unit(np.array([-(raw[1] - raw[0])[1], (raw[1] - raw[0])[0]]))
        for direction in (-1, 1):
            rows = [dict(row) for row in fitted['observations']]
            moved = raw + direction * 1.5 * transverse
            rows[i] = {**rows[i], 'rawEnds': moved.tolist(), 'uv': _pixels(moved, frames[observation['photo']]['A']).tolist()}
            if 'rawSegments' in observation:
                segments = np.asarray(observation['rawSegments']) + direction * 1.5 * transverse
                rows[i].update(rawSegments=segments.tolist(), uvSegments=[_pixels(segment, frames[observation['photo']]['A']).tolist() for segment in segments])
            try:
                sensitivity.extend(np.asarray(_line_fit(rows, frames, up)['sharedSegmentsNative']).reshape(-1, 3).tolist())
            except (ValueError, np.linalg.LinAlgError):
                pass
    fitted['pixelSensitivityPointsNative'] = sensitivity
    return fitted


def _object_edges(catalog, segmentation, geometry, frames, up, legacy, drawings):
    objects, diagnostics = [], []
    initial_frames = {photo: {**frame, 'K': frame['initialK'], 'pose': frame['initialPose']} for photo, frame in frames.items()}
    for ident in TARGETS:
        item = catalog[ident]
        initial = np.asarray(legacy[ident]['pointNative'], float) if legacy[ident]['pointNative'] else None
        bottom, top, views = [], [], []
        try:
            if ident == 'fence-0':
                plane_index = item.get('geometryPlaneIndex', 0)
                plane = geometry['fence']['planes'][plane_index]
                beams = sorted((row for row in geometry['fence']['beams'] if row['plane'] == plane_index and row['horizontal']), key=lambda row: row['heightNative'])
                for photo in frames:
                    for row in [row for row in beams if row['sourcePhoto'] == photo][:10]:
                        edges = []
                        for edge in row['rawEdges']:
                            uv = _pixels(edge, frames[photo]['A'])
                            xyz = _intersect(uv, initial_frames[photo]['K'], initial_frames[photo]['pose'], np.asarray(plane['normal']), plane['offset'])
                            edges.append((float(np.mean(xyz @ up)), edge, uv))
                        _, edge, uv = min(edges, key=lambda e: e[0])
                        bottom.append({'photo': photo, 'rawEnds': edge, 'uv': uv.tolist(), 'score': 1., 'sourceBeam': row['id']})
            else:
                centers = []
                for observation in item['observations']:
                    photo = observation['photo']
                    match = re.search(r'instance (\d+)', observation['source'])
                    if match is None:
                        continue
                    response = _response(segmentation, photo, 'yellow safety post')
                    rawmask = _raw_mask({'rle': [response['rle'][int(match[1])]]}, frames[photo]['rgb'].shape[:2])
                    frame = frames[photo]
                    occluders = np.zeros_like(rawmask)
                    obstruction_words = {'black bollard', 'cart', 'industrial robot arm', 'warning sign',
                                         'instruction poster', 'workcell sign', 'control cabinet', 'work platform'}
                    for prompt in segmentation['prompts']:
                        if prompt['text'] in obstruction_words:
                            occluders |= _raw_mask(_response(segmentation, photo, prompt['text']), rawmask.shape)
                    if initial is None:
                        xy = np.mean(np.asarray(observation['box']).reshape(2, 2), axis=0).astype(int)
                        centers.append(frame['points'][xy[1], xy[0]])
                        center = centers[-1]
                    else:
                        center = initial
                    projection, _ = _project([center, center + up], initial_frames[photo])
                    raw_uv = _pixels(projection, np.linalg.inv(frame['A']))
                    down = _unit(raw_uv[0] - raw_uv[1])
                    found, detail = _terminal_edges(frame['rgb'], rawmask, down, photo, frame['A'], occluders)
                    upper, upper_detail = _terminal_edges(frame['rgb'], rawmask, -down, photo, frame['A'], occluders)
                    bottom.extend(found); top.extend(upper)
                    views.append({'photo': photo, 'bottom': detail, 'top': upper_detail})
                    drawings[photo]['candidates'].extend({**row, 'objectId': ident} for row in found)
                if initial is None and centers:
                    initial = np.median(centers, axis=0)
            if initial is None or not np.isfinite(initial).all():
                raise ValueError('No source-only association point for this object')
            fitted = _match_edges(bottom, frames, up, lowest=ident == 'fence-0')
            result = {'id': ident, 'status': 'edge_supported', 'reason': None, 'bottomEdge': fitted,
                      'axisNative': up.tolist(), 'topEdge': None,
                      'identity': 'observed lower fence rail' if ident == 'fence-0' else 'yellow housing; light-curtain semantic identity unconfirmed',
                      'scope': 'visible rigid bottom edge; occluded portions and hardware are not measured'}
            if top:
                try:
                    result['topEdge'] = _match_edges(top, frames, up)
                except ValueError as error:
                    result['topEdgeReason'] = str(error)
            for row in fitted['observations']:
                drawings[row['photo']]['selected'].append({**row, 'objectId': ident})
        except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
            result = _unavailable(str(error), id=ident, bottomEdge=None, topEdge=None)
        objects.append(result)
        diagnostics.append({'id': ident, 'bottomCandidates': bottom, 'topCandidates': top, 'views': views})
    return objects, diagnostics


def _floor_evidence(points, observations, frames, drawings, normal=None, offset=None, inlier=None):
    """Overlay the samples actually fitted, including retriangulated track pixels."""
    records = {photo: [] for photo in frames}
    for index, (point, seen) in enumerate(zip(points, observations)):
        for observation in seen:
            photo = observation['photo']
            raw_uv = _pixels([observation['uv']], np.linalg.inv(frames[photo]['A']))[0]
            records[photo].append({'uvRaw': raw_uv.tolist(), 'pointNative': point.tolist(),
                                   'planeInlier': bool(inlier[index]) if inlier is not None else None,
                                   'signedResidualNative': float(point @ normal + offset) if normal is not None else None})
    output = []
    for photo, rows in records.items():
        saved = rows[::max(1, len(rows) // 350)]
        drawings[photo]['floorPixelsRaw'] = [row['uvRaw'] for row in saved if row['planeInlier'] is True]
        drawings[photo]['floorRejectedPixelsRaw'] = [row['uvRaw'] for row in saved if row['planeInlier'] is False]
        output.append({'photo': photo, 'observedSampleCount': len(rows),
                       'planeInlierCount': sum(row['planeInlier'] is True for row in rows),
                       'samples': saved, 'sampling': 'deterministic bounded display sample; counts include every fitted observation'})
    return output


def _fit_ground(geometry, frames, legacy, objects, refined, drawings, catalog):
    up = _unit(np.asarray(geometry['floor']['normal'], float))
    offset = geometry['floor']['offset'] / np.linalg.norm(geometry['floor']['normal'])
    # The old source-derived body span only bounds pixel association; it never
    # supplies a new floor point, object end or replacement-camera depth.
    spans = [abs(float((np.asarray(obj['topEdge']['pointNative']) - obj['bottomEdge']['pointNative']) @ up))
             for obj in objects if obj['id'].startswith('post-') and obj.get('topEdge') and obj.get('bottomEdge')]
    radius_basis = 'triangulated visible top-to-bottom housing span'
    if not spans:
        spans = [max(catalog[ident]['modelDimensionsNative']) for ident in TARGETS if ident.startswith('post-')]
        radius_basis = 'saved source-pointmap housing span; association only'
    radius = max(float(np.median(spans)) * .3, geometry['floor']['residualP95Native'] * 6)
    locations = {ident: np.asarray(row['footNative']) for ident, row in legacy.items() if row['footNative'] is not None}
    masks, source_points, source_photos, point_observations, pixel_diagnostics = {}, [], [], [], []
    for photo, frame in frames.items():
        pts = frame['points']
        near = np.zeros(frame['shape'], bool)
        for foot in locations.values():
            delta = pts - foot
            distance = np.linalg.norm(delta - (delta @ up)[..., None] * up, axis=2)
            near |= distance <= radius
        lower_roi = near & (np.indices(frame['shape'])[0] > frame['shape'][0] * .5)
        valid_roi = lower_roi & frame['valid']
        geometric = valid_roi & (abs(pts @ up + offset) < max(geometry['floor']['residualP95Native'] * 4, radius * .12))
        selected = geometric & ~frame['excluded']
        semantic = []
        for word, mask in frame['semanticMasks'].items():
            yy, xx = np.where(mask & geometric)
            if len(xx):
                sample = np.arange(len(xx))[::max(1, len(xx) // 60)]
                semantic.append({'prompt': word, 'removedCandidatePixels': len(xx),
                                 'rawSamples': _pixels(np.c_[xx[sample], yy[sample]], np.linalg.inv(frame['A'])).tolist()})
        pixel_diagnostics.append({'photo': photo, 'localLowerRoiPixels': int(lower_roi.sum()),
                                  'validRoiPixels': int(valid_roi.sum()), 'withinSavedPlaneBandPixels': int(geometric.sum()),
                                  'unsegmentedCandidatePixels': int(selected.sum()),
                                  'semanticRemovedPixels': int((geometric & frame['excluded']).sum()),
                                  'semanticMaskIntersections': semantic,
                                  'floorIdentity': 'Unsegmented horizontal-surface hypothesis; no concrete-material classifier or floor annotation is available.'})
        masks[photo] = selected
        y, x = np.where(selected)
        sample = np.arange(len(x))[::max(1, len(x) // 2500)]
        source_points.extend(pts[y[sample], x[sample]])
        source_photos.extend([photo] * len(sample))
        point_observations.extend([{'photo': photo, 'uv': [float(px), float(py)]}] for px, py in zip(x[sample], y[sample]))
    tracks = None
    if refined:
        found, tracks_diagnostics = _features(frames, {'left': {'masks': masks}})
        tracks = found['left']
        points = np.asarray([row['xyz'] for row in tracks])
        point_observations = [row['observations'] for row in tracks]
        photos_per_point = [sorted(o['photo'] for o in row['observations']) for row in tracks]
        method = 'RGB floor tracks independently retriangulated with replacement cameras'
    else:
        points = np.asarray(source_points)
        photos_per_point = [[photo] for photo in source_photos]
        tracks_diagnostics = None
        method = 'local source pointmap support with original cameras'
    if len(points) < 20:
        return {'status': 'unsupported', 'reason': 'Fewer than twenty local floor samples/tracks',
                'normal': None, 'offset': None, 'patches': [], 'method': method,
                'trackDiagnostics': tracks_diagnostics, 'supportPoints': len(points),
                'pixelSelectionDiagnostics': pixel_diagnostics,
                'sourceSupport': _floor_evidence(points, point_observations, frames, drawings)}
    threshold = max(geometry['floor']['residualP95Native'], radius * .012)
    try:
        normal, d, inlier = _consensus(points, up, threshold)
        if inlier.sum() < 20:
            raise ValueError('Too few common floor plane inliers')
        support = points[inlier]
        H = _horizontal(normal)
        if np.linalg.svd((support - support.mean(0)) @ H, compute_uv=False)[-1] < threshold * 2:
            raise ValueError('Floor support is a narrow line, not a two-dimensional patch')
        patches = []
        for ident, location in locations.items():
            obj = next(row for row in objects if row['id'] == ident)
            if obj.get('bottomEdge'):
                ends = np.asarray(obj['bottomEdge']['sharedSegmentsNative']).reshape(-1, 3)
                current = ends[np.argmin(ends @ normal)]
                location = current - (current @ normal + d) * normal
            nearby = np.linalg.norm((support - location) @ H, axis=1) < radius
            patch = support[nearby]
            if len(patch) < 12:
                patches.append({'objectId': ident, 'status': 'unsupported', 'reason': 'Insufficient nearby floor support', 'supportPoints': len(patch)})
                continue
            residual = abs(patch @ normal + d)
            nearest = float(np.linalg.norm((patch - location) @ H, axis=1).min())
            hull = ConvexHull(patch @ H)
            inside = bool(np.all(hull.equations[:, :2] @ (location @ H) + hull.equations[:, 2] <= 1e-8))
            patches.append({'objectId': ident, 'status': 'available' if nearest < radius * .55 else 'unsupported',
                            'reason': None if nearest < radius * .55 else 'Floor foot would extrapolate beyond local support',
                            'supportPoints': len(patch), 'nearestSupportNative': nearest,
                            'footInsideSupportHull': inside, 'radiusNative': radius,
                            'associationRadiusBasis': radius_basis, 'footNative': location.tolist(),
                            'residualP95Native': float(np.percentile(residual, 95)),
                            'supportHullNative': patch[hull.vertices].tolist()})
        return {'status': 'available', 'reason': None, 'normal': normal.tolist(), 'offset': float(d),
                'supportPoints': int(inlier.sum()), 'inputPoints': len(points), 'method': method,
                'residualP95Native': float(np.percentile(abs(support @ normal + d), 95)),
                'sourcePhotos': sorted({photo for keep, photos in zip(inlier, photos_per_point) if keep for photo in photos}),
                'patches': patches, 'trackDiagnostics': tracks_diagnostics,
                'pixelSelectionDiagnostics': pixel_diagnostics,
                'sourceSupport': _floor_evidence(points, point_observations, frames, drawings, normal, d, inlier),
                'semanticStatus': 'conditional; concrete floor identity unverified',
                'assumption': 'Dominant unsegmented local horizontal surface is concrete floor; mounting plates and rails must be inspected in source overlays.'}
    except (ValueError, np.linalg.LinAlgError, QhullError) as error:
        return {'status': 'unsupported', 'reason': str(error), 'normal': None, 'offset': None, 'patches': [],
                'supportPoints': len(points), 'method': method, 'trackDiagnostics': tracks_diagnostics,
                'pixelSelectionDiagnostics': pixel_diagnostics,
                'sourceSupport': _floor_evidence(points, point_observations, frames, drawings)}


def _measure(objects, ground, frames, up):
    measured = []
    for row in objects:
        if row['status'] != 'edge_supported':
            measured.append(row); continue
        patch = next((patch for patch in ground['patches'] if patch['objectId'] == row['id']), None)
        if ground['status'] != 'available' or patch is None or patch['status'] != 'available':
            measured.append({**row, **_unavailable('No sufficiently supported local floor', bottomEdge=row['bottomEdge'], topEdge=row['topEdge'])}); continue
        normal = np.asarray(ground['normal'])
        shared = np.asarray(row['bottomEdge']['sharedSegmentsNative']).reshape(-1, 3)
        point = shared[np.argmin(shared @ normal)]
        height = float(point @ normal + ground['offset'])
        if height < 0:
            measured.append({**row, **_unavailable('Source edge falls below the fitted local floor')}); continue
        error = patch['residualP95Native']
        sensitivity = np.asarray(row['bottomEdge']['pixelSensitivityPointsNative']) @ normal + ground['offset']
        uncertainty = {'edgePixelPerturbationRawPx': 1.5, 'floorResidualP95Native': error,
                       'cameraSystematicErrorBounded': False, 'semanticOcclusionErrorBounded': False,
                       'meaning': 'Image-edge sensitivity plus source floor scatter; not a statistical confidence interval.'}
        measured.append({**row, 'status': 'conditional',
                         'reason': 'Visible source edge and inferred local floor; camera and semantic systematic errors remain unbounded',
                         'pointNative': point.tolist(),
                         'footNative': (point - height * normal).tolist(), 'heightNative': height,
                         'rangeNative': [max(0., float(sensitivity.min() - error)), float(sensitivity.max() + error)],
                         'sourcePhotos': row['bottomEdge']['sourcePhotos'],
                         'localFloor': {'normal': ground['normal'], 'offset': ground['offset'], 'patch': patch},
                         'uncertainty': uncertainty})
    return measured


def source_physical_clearances(geometry, catalog, segmentation, frames, *, cameras_refined=False, drawings=None):
    """Native source-edge distances; mask extrema serve association only.

    Frames use the existing _load contract (raw RGB, raw-to-canonical A,
    K/C2W pose, pointmaps and masks). No metric reference or evaluation target
    enters this function. Only conditional objects have physical endpoints.
    The returned unit ground plane is the exact plane used by every endpoint.
    """
    start = time.monotonic()
    up = _unit(np.asarray(geometry['floor']['normal'], float))
    if drawings is None:
        drawings = {photo: {'standard': [], 'selected': [], 'candidates': [], 'floorPixelsRaw': []} for photo in frames}
    legacy = {ident: _legacy(catalog[ident], geometry) for ident in TARGETS}
    objects, edge_diagnostics = _object_edges(catalog, segmentation, geometry, frames, up, legacy, drawings)
    stage_errors = []
    try:
        ground = _fit_ground(geometry, frames, legacy, objects, cameras_refined, drawings, catalog)
    except Exception as error:
        stage_errors.append({'stage': 'localFloor', 'error': f'{type(error).__name__}: {error}'})
        ground = {'status': 'unsupported', 'reason': stage_errors[-1]['error'], 'normal': None, 'offset': None, 'patches': []}
    objects = _measure(objects, ground, frames, up)
    return {'schemaVersion': 1, 'coordinateSystem': 'MapAnything native', 'units': 'native',
            'ground': ground, 'objects': objects,
            'associationFloor': geometry['floor'],
            'observedEnvelopeBaselines': [{'id': ident, **legacy[ident]} for ident in TARGETS],
            'diagnostics': {'objectEdges': edge_diagnostics, 'stageErrors': stage_errors},
            'scope': 'Visible rigid source edges in two or more photos to one inferred local floor. Mask-envelope baselines are not physical bottoms. Occluded object portions and hardware remain unknown.',
            'timingScope': 'Source edge and local-ground analysis after loading; excludes depth, segmentation and model generation.',
            'wallSeconds': time.monotonic() - start}


def apply_source_clearances(root, sources):
    """Cloud/main-flow bridge after objects.json exists; no scale is required."""
    import trimesh
    root = Path(root)
    sources = [Path(source) for source in sources]
    if len(sources) != 4:
        raise ValueError('Exactly four original JPEG sources are required in recorded order')
    geometry, catalog, segmentation, frames, _, inputs = _load(root, sources, None)
    drawings = {photo: {'standard': [], 'selected': [], 'candidates': [], 'floorPixelsRaw': []} for photo in frames}
    result = source_physical_clearances(geometry, catalog, segmentation, frames, drawings=drawings)
    result['sourceInputs'] = inputs
    result['evidenceImages'] = _overlays(root, frames, drawings, result['objects'])
    ground = result['ground']
    floor_data = None
    if ground['status'] == 'available':
        normal, offset = np.asarray(ground['normal'], float), float(ground['offset'])
        if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(offset) or not np.isclose(np.linalg.norm(normal), 1.):
            raise ValueError('Physical clearances require a finite unit ground plane')
        floor = trimesh.load(root / 'floor-fitted.glb', force='scene', process=False)
        if len(floor.graph.nodes_geometry) != 1:
            raise ValueError('Expected the existing single fitted floor surface')
        node = floor.graph.nodes_geometry[0]
        transform, name = floor.graph[node]
        native = trimesh.transform_points(floor.geometry[name].vertices, transform)
        projected = native - (native @ normal + offset)[:, None] * normal
        floor.geometry[name].vertices = trimesh.transform_points(projected, np.linalg.inv(transform))
        floor_data = floor.export(file_type='glb')
        # Preserve the existing node identity and footprint. The viewer and
        # measurement now consume this same plane in this same native world.
        geometry['floor'] = {**geometry['floor'], **{key: ground[key] for key in
                             ('normal', 'offset', 'supportPoints', 'residualP95Native', 'sourcePhotos', 'method')},
                             'status': 'conditional local floor fit; concrete-floor semantic identity unverified',
                             'displayExtent': 'existing fitted footprint projected onto the physical-clearance ground plane'}
    geometry['physicalClearances'] = result
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    geometry_serialized = json.dumps(geometry, indent=2, allow_nan=False) + '\n'
    if floor_data is not None:
        (root / 'floor-fitted.glb').write_bytes(floor_data)
        (root / 'floor-reference.json').write_text(json.dumps(geometry['floor'], indent=2, allow_nan=False) + '\n')
    (root / 'physical-clearances.json').write_text(serialized)
    (root / 'geometry.json').write_text(geometry_serialized)
    return result


def _route(label, objects, scale):
    rows = []
    for obj in objects:
        row = dict(obj)
        if scale is None:
            row.update(status='unsupported', reason='No source-supported metric scale', heightM=None, rangeM=None)
        elif obj.get('heightNative') is None:
            row.update(heightM=None, rangeM=None)
        else:
            row.update(heightM=obj['heightNative'] * scale,
                       rangeM=[v * scale for v in obj['rangeNative']])
        rows.append(row)
    return {'label': label, 'scaleMPerNative': scale, 'objects': rows}


def _overlays(out, frames, drawings, objects):
    files = []
    colors = {'fence-0': (0, 255, 255), 'post-box-1': (0, 230, 0), 'post-box-2': (255, 120, 0)}
    for photo, frame in frames.items():
        image = cv2.cvtColor(frame['rgb'], cv2.COLOR_RGB2BGR)
        drawing = drawings[photo]
        for uv in drawing['floorPixelsRaw']:
            cv2.circle(image, tuple(np.rint(uv).astype(int)), 3, (210, 140, 40), -1)
        for uv in drawing.get('floorRejectedPixelsRaw', []):
            cv2.circle(image, tuple(np.rint(uv).astype(int)), 3, (40, 70, 200), 1)
        for standard in drawing['standard']:
            cv2.polylines(image, [np.rint(standard['hullRaw']).astype(np.int32)], True, (220, 0, 220), 2)
            for uv in standard['tangentPixelsRaw']:
                cv2.circle(image, tuple(np.rint(uv).astype(int)), 6, (220, 0, 220), 2)
        for candidate in drawing['candidates']:
            for segment in candidate.get('rawSegments', [candidate['rawEnds']]):
                cv2.line(image, *[tuple(np.rint(uv).astype(int)) for uv in segment], (130, 130, 130), 2)
        for selected in drawing['selected']:
            for segment in selected.get('rawSegments', [selected['rawEnds']]):
                endpoints = [tuple(np.rint(uv).astype(int)) for uv in segment]
                cv2.line(image, *endpoints, colors[selected['objectId']], 5)
            cv2.putText(image, selected['objectId'], endpoints[0], cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[selected['objectId']], 3)
        for obj in objects:
            if obj.get('heightNative') is None or photo not in obj['sourcePhotos']:
                continue
            uv, depth = _project([obj['pointNative'], obj['footNative']], frame)
            if np.all(depth > 0):
                ends = _pixels(uv, np.linalg.inv(frame['A']))
                cv2.line(image, *[tuple(np.rint(p).astype(int)) for p in ends], colors[obj['id']], 4)
                for end in ends:
                    cv2.circle(image, tuple(np.rint(end).astype(int)), 7, colors[obj['id']], 2)
        name = f'raw-image-features-{photo}.jpg'
        factor = min(1., 1600 / max(image.shape[:2]))
        display = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1 else image
        if not cv2.imwrite(str(out / name), display):
            raise OSError('Could not save raw-image feature overlay')
        files.append({'photo': photo, 'file': name, 'rawShape': list(image.shape[:2]),
                      'displayShape': list(display.shape[:2]), 'displayScaleFromRaw': factor,
                      'legend': 'magenta: named circular silhouettes/tangencies; gray: end-edge candidates; colored: selected visible edge and floor foot; blue filled: actual floor-fit inliers; red hollow: rejected floor-fit samples'})
    return files


def _validate_results(result):
    if set(result['routes']) not in ({'A', 'B', 'C', 'D'}, {'A', 'B', 'C', 'D', 'J'}):
        raise ValueError('All four comparison routes and only the optional joint route are allowed')
    if 'J' in result['routes']:
        joint = result.get('jointReference', {})
        if not joint or result['routes']['J']['scaleMPerNative'] != joint.get('mPerNative'):
            raise ValueError('Joint route must use its own validated metric scale')
    for route in result['routes'].values():
        if {row['id'] for row in route['objects']} != set(TARGETS):
            raise ValueError('Every route must preserve all target identities')
        for row in route['objects']:
            if row.get('heightNative') is None:
                continue
            p, f = np.asarray(row['pointNative']), np.asarray(row['footNative'])
            plane = row['localFloor']
            n = np.asarray(plane['normal'])
            if (not np.isclose(np.linalg.norm(n), 1., atol=1e-7) or
                    not np.allclose(p - f, row['heightNative'] * n, atol=1e-7) or
                    not np.isclose(f @ n + plane['offset'], 0., atol=1e-7)):
                raise ValueError('Feature, foot and ground plane disagree')
            if row.get('heightM') is not None and not np.isclose(row['heightM'], row['heightNative'] * route['scaleMPerNative']):
                raise ValueError('Metric/native route mismatch')
    json.dumps(result, allow_nan=False)


def build(root, out, sources, reference, cameras=None, joint_reference=None):
    start = time.monotonic()
    root, out = Path(root), Path(out)
    sources = [Path(source) for source in sources]
    reference = _reference(reference)
    joint_reference = _joint_reference(joint_reference, reference, cameras)
    if len(sources) != 4:
        raise ValueError('Exactly four original JPEG sources are required in recorded order')
    if root.resolve() == out.resolve():
        raise ValueError('Output must not replace the saved source experiment')
    out.mkdir(parents=True, exist_ok=False)
    geometry, catalog, segmentation, frames, anchors, inputs = _load(root, sources, cameras)
    up = _unit(np.asarray(geometry['floor']['normal'], float))
    drawings = {photo: {'standard': [], 'selected': [], 'candidates': [], 'floorPixelsRaw': []} for photo in frames}
    physical = source_physical_clearances(geometry, catalog, segmentation, frames,
                                         cameras_refined=cameras is not None, drawings=drawings)
    objects, ground = physical['objects'], physical['ground']
    stage_errors = physical['diagnostics']['stageErrors']
    # B is a scale-only ablation: a newly inferred floor must not silently
    # change the circular-standard upright prior, especially for sparse tracks.
    calibration_up = up
    try:
        calibration = _calibrate(anchors, frames, geometry, reference, calibration_up, drawings)
    except Exception as error:
        stage_errors.append({'stage': 'namedCircularStandard', 'error': f'{type(error).__name__}: {error}'})
        calibration = {'baseline': {'status': 'conditional', 'mPerNative': reference['features']['wholeComponentHeightM'] / geometry['anchor']['nativeHeight']},
                       'new': {'status': 'unsupported', 'mPerNative': None, 'reason': stage_errors[-1]['error']},
                       'candidates': [], 'selectionReason': 'Source fit failed; see diagnostic error'}
    calibration['uprightAxisSource'] = 'saved floor normal; fixed unrefined upright-axis assumption for scale-only comparison'
    calibration['uprightAxisNative'] = up.tolist()
    calibration['newFloorUpDifferenceDeg'] = (float(np.degrees(np.arccos(np.clip(np.asarray(ground['normal']) @ up, -1., 1.))))
                                              if ground['status'] == 'available' else None)
    original = physical['observedEnvelopeBaselines']
    old, new = calibration['baseline']['mPerNative'], calibration['new']['mPerNative']
    result = {'schemaVersion': 1, 'coordinateSystem': 'MapAnything native',
              'cameraRoute': 'replacement cameras; independently retriangulated source geometry' if cameras is not None else 'original MapAnything cameras',
              'reference': reference, 'sourceInputs': inputs, 'calibration': calibration,
              'ground': ground, 'objects': objects,
              'routes': {'A': _route('Saved geometry / saved whole-height scale', original, old),
                         'B': _route('Saved geometry / new named-circle scale', original, new),
                         'C': _route('Source end edges and local floor / saved whole-height scale', objects, old),
                         'D': _route('Source end edges and local floor / new named-circle scale', objects, new)},
              'diagnostics': physical['diagnostics'],
              'limitations': ['Only four supplied photos; no measured camera calibration or camera height.',
                              'Circular standard sections and housing uprightness are image-tested hypotheses.',
                              'A/B intentionally retain old points and are labelled baselines; they are not recalibrated geometry.',
                              'SAM and old pointmaps associate image regions; replacement-camera floor points and edges are newly triangulated.',
                              'Occluded whole-object minima, mounting hardware and physical accuracy remain unverified.',
                              'Pixel sensitivity and floor scatter do not bound camera or semantic systematic error.',
                              'Ground-truth evaluation values are excluded from estimator inputs and method selection.'],
              'wallSeconds': time.monotonic() - start}
    if joint_reference is not None:
        result['jointReference'] = joint_reference
        result['cameraRoute'] = 'Joint three-dimension button and camera fit; independently retriangulated source geometry'
        result['routes']['J'] = _route('Joint button height, both diameters, cameras and scene tracks', objects,
                                      joint_reference['mPerNative'])
    result['evidenceImages'] = _overlays(out, frames, drawings, objects)
    _validate_results(result)
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs=4, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--cameras', type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text()) if args.reference else {
        'features': dict(zip(FEATURES, (.10, .085, .04))), 'scopeStatus': 'pending_confirmation'}
    result = build(args.root, args.out, args.sources, reference, args.cameras)
    print(json.dumps({'output': str(args.out), 'cameraRoute': result['cameraRoute'],
                      'objects': [{'id': row['id'], 'status': row['status']} for row in result['objects']],
                      'newScaleStatus': result['calibration']['new']['status']}))


if __name__ == '__main__':
    main()
