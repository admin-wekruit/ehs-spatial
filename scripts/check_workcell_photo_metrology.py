"""Run: PYTHONPATH=.:scripts python scripts/check_workcell_photo_metrology.py."""
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import trimesh

from ehs_spatial.measurements import measure_observed_points
import workcell_photo_metrology as metrology
from workcell_photo_metrology import (TARGETS, _color_observation, _fit_circle, _floor_evidence, _legacy,
                                     _joint_reference, _line_fit, _measure, _pixels, _reference, _route,
                                     _supported_color_hull, _tangencies, _terminal_edges, _validate_results)
from workcell_photo_objects import _project


def main():
    up = np.array([0., 0., 1.])
    A = np.array([[.25, 0., -.375], [0., .25, -2.375], [0., 0., 1.]])
    Kraw = np.array([[800., 0., 640.], [0., 820., 480.], [0., 0., 1.]])
    frames = {}
    for photo, position in enumerate(([0., -4., 1.4], [1.8, -4., 1.6], [-1.8, -3.8, 1.5]), 1):
        position = np.array(position)
        forward = np.array([0., 0., 1.]) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up); right /= np.linalg.norm(right)
        pose = np.eye(4)
        pose[:3, :3] = np.c_[right, np.cross(forward, right), forward]
        pose[:3, 3] = position
        frames[photo] = {'photo': photo, 'K': A @ Kraw, 'pose': pose, 'A': A}
    endpoints = np.array([[-.15, 0., .45], [.15, 0., .45]])
    observations = []
    for photo, frame in frames.items():
        uv, _ = _project(endpoints, frame)
        raw = _pixels(uv, np.linalg.inv(A))
        independently_raw, _ = _project(endpoints, {**frame, 'K': Kraw})
        assert np.allclose(raw, independently_raw), 'Pixel-centre crop transform or C2W convention changed'
        assert np.allclose(_pixels(raw, A), uv)
        observations.append({'photo': photo, 'uv': uv.tolist(), 'rawEnds': raw.tolist()})
    edge = _line_fit(observations, frames, up)
    assert abs(edge['pointNative'][2] - .45) < 1e-8, 'Initial pointmap outlier moved the true RGB edge'
    assert edge['maxReprojectionErrorRawPx'] < 1e-6
    assert np.allclose(np.asarray(edge['sharedEndsNative'])[:, 2], .45)
    noisy_observations = copy.deepcopy(observations)
    noisy_observations[0]['rawEnds'][0][1] += 1.5
    noisy_observations[0]['uv'] = _pixels(noisy_observations[0]['rawEnds'], A).tolist()
    noisy_line = _line_fit(noisy_observations, frames, up)
    translated = copy.deepcopy(frames)
    for frame in translated.values():
        frame['pose'][0, 3] += 1.
    translated_line = _line_fit(noisy_observations, translated, up)
    assert np.allclose(np.asarray(translated_line['pointNative']) - [1., 0., 0.], noisy_line['pointNative'], atol=1e-6), 'Line height depends on the arbitrary along-axis origin'

    def segmented_observation(photo, spans):
        segments = [np.array([[a, 0., .45], [b, 0., .45]]) for a, b in spans]
        raw_segments = [_pixels(_project(segment, frames[photo])[0], np.linalg.inv(A)).tolist() for segment in segments]
        envelope = np.array([[min(a for a, _ in spans), 0., .45], [max(b for _, b in spans), 0., .45]])
        uv = _project(envelope, frames[photo])[0]
        return {'photo': photo, 'uv': uv.tolist(), 'rawEnds': _pixels(uv, np.linalg.inv(A)).tolist(),
                'rawSegments': raw_segments, 'uvSegments': [_pixels(segment, A).tolist() for segment in raw_segments]}
    separated = segmented_observation(1, [(-.15, -.05), (.05, .15)])
    only_gap = segmented_observation(2, [(-.025, .025)])
    try:
        _line_fit([separated, only_gap], frames, up)
    except ValueError as error:
        assert 'common visible segment' in str(error), str(error)
    else:
        raise AssertionError('An unseen gap became a cross-view observation')
    same_segments = _line_fit([separated, segmented_observation(2, [(-.15, -.05), (.05, .15)])], frames, up)
    assert len(same_segments['sharedSegmentsNative']) == 2
    assert len(same_segments['unobservedGapSegmentsNative']) == 1
    replacement = copy.deepcopy(frames)
    for frame in replacement.values():
        frame['pose'][:3, 3] *= 1.2
    refitted = _line_fit(observations, replacement, up)
    assert abs(refitted['pointNative'][2] - .54) < 1e-8, 'Replacement cameras reused old edge points'

    # Sensitivity and the local foot use the final tilted floor, including
    # lateral pixel movement that the original-up scalar would miss.
    normal = np.array([.1, 0., 1.]); normal /= np.linalg.norm(normal)
    perturbations = [[-.17, 0., .45], [.17, 0., .45]]
    tested_edge = {**edge, 'pixelSensitivityPointsNative': perturbations}
    grounded = _measure([{'id': 'post-box-1', 'status': 'edge_supported', 'bottomEdge': tested_edge,
                          'topEdge': None}],
                        {'status': 'available', 'normal': normal.tolist(), 'offset': 0.,
                         'patches': [{'objectId': 'post-box-1', 'status': 'available', 'residualP95Native': .002}]},
                        frames, up)[0]
    assert np.allclose(grounded['rangeNative'], [min(np.asarray(perturbations) @ normal) - .002,
                                                max(np.asarray(perturbations) @ normal) + .002])
    assert abs(np.asarray(grounded['footNative']) @ normal) < 1e-9
    floor_points = np.array([[0., 0., .45], [0., .1, .50]])
    floor_observations = [[{'photo': photo, 'uv': _project([point], frames[photo])[0][0].tolist()}]
                          for photo, point in zip((1, 3), floor_points)]
    drawings = {photo: {} for photo in frames}
    evidence = _floor_evidence(floor_points, floor_observations, frames, drawings, up, -.45, [True, False])
    assert len(drawings[1]['floorPixelsRaw']) == 1 and drawings[2]['floorPixelsRaw'] == []
    assert len(drawings[3]['floorRejectedPixelsRaw']) == 1 and drawings[3]['floorPixelsRaw'] == []
    assert evidence[1]['observedSampleCount'] == 0, 'A floor overlay displayed association pixels as fitted samples'

    # A mask-supported foreground outlier changes the historical extrema but
    # cannot enter the source-edge triangulation above.
    rng = np.random.default_rng(17)
    cloud = np.c_[rng.uniform(-.15, .15, 120), rng.uniform(-.015, .015, 120), rng.uniform(.45, 1.8, 120)]
    cloud = np.vstack([cloud, [0., 0., .01]])
    measured = measure_observed_points(cloud, {'floor_plane': [0, 0, 1, 0]}, mask_pixels=len(cloud))
    item = {'id': 'post-box-1', 'observations': [{'photo': p, 'observedMeasurements': measured} for p in (1, 2)]}
    old = _legacy(item, {'floor': {'normal': [0, 0, 1], 'offset': 0}})
    assert abs(old['heightNative'] - .01) < 1e-9 and abs(edge['pointNative'][2] - .45) < 1e-8
    even = {'id': 'post-box-1', 'observations': []}
    for photo, height in enumerate((.2, .4, .6, .8), 1):
        sample = copy.deepcopy(measured)
        corners = np.asarray(sample['basis']['corners_native'])
        corners[:, 2] += height - corners[:, 2].min()
        sample['basis']['corners_native'] = corners.tolist()
        even['observations'].append({'photo': photo, 'observedMeasurements': sample})
    median = _legacy(even, {'floor': {'normal': [0, 0, 1], 'offset': 0}})
    assert median['heightNative'] == .5 and median['pointNative'][2] == .5, median

    selected_plane = {'floor': {'normal': [0, 0, 1], 'offset': 0}, 'clearances': [
        {'id': f'fence-plane-{index}-lower-rail', 'pointNative': [0, 0, height],
         'footNative': [0, 0, 0], 'heightNative': height, 'sourcePhotos': [1, 2],
         'observedViewHeightRangeNative': [height, height]}
        for index, height in enumerate((.2, .7))]}
    assert _legacy({'id': 'fence-0', 'geometryPlaneIndex': 1}, selected_plane)['heightNative'] == .7

    # A connected yellow-colored background wedge was entering the convex hull.
    # Its local RGB appearance differs from the body; no physical dimensions or
    # desired clearance is supplied to the image-only contour refinement.
    color_rgb = np.full((150, 190, 3), [115, 110, 100], np.uint8)
    color_rgb[35:115, 70:135] = [230, 205, 20]
    color_rgb[90:114, 53:70] = [105, 85, 35]
    color_rgb[90:110, 54:70] = [115, 110, 100]
    contour = _color_observation(color_rgb, [40, 20, 150, 130], 'yellow')
    assert min(p[0] for p in contour['colorHullRaw']) == 53
    assert min(p[0] for p in contour['hullRaw']) >= 69, 'Connected background still expands silhouette'
    assert max(p[0] for p in contour['hullRaw']) == 134, 'Body boundary changed without image evidence'
    # A crossing bright wire erases just a short stretch of the body edge.
    # Those unsupported outward vertices cannot define the physical diameter;
    # the independently visible body edges must stay at their original pixels.
    wire_rgb = np.full((150, 190, 3), 30, np.uint8)
    wire_rgb[35:115, 70:135] = [230, 205, 20]
    clean_hull = np.array([[70, 35], [134, 35], [134, 114], [70, 114]], np.float32)
    clean, _ = _supported_color_hull(wire_rgb, clean_hull)
    assert set(map(tuple, clean)) == set(map(tuple, clean_hull)), 'Supported contours were globally eroded'
    wire_rgb[75:80, 133:150] = [240, 235, 220]
    bulge = np.array([[70, 35], [134, 35], [136, 76], [136, 78], [134, 114], [70, 114]], np.float32)
    supported, rejected = _supported_color_hull(wire_rgb, bulge)
    assert max(supported[:, 0]) == 134 and len(rejected) == 2, 'A locally obscured contour expanded the diameter'
    cv2.setRNGSeed(12345)
    assert _color_observation(color_rgb, [40, 20, 150, 130], 'yellow') == contour, 'RGB refinement depends on unrelated RNG history'
    for absent in (np.full_like(color_rgb, 110), np.pad(color_rgb[35:37, 70:95], ((20, 128), (30, 135), (0, 0)))):
        try:
            _color_observation(absent, [0, 0, absent.shape[1], absent.shape[0]], 'yellow')
        except ValueError:
            pass
        else:
            raise AssertionError('Missing component or insufficient color interior became a contour')
    for bad_box in ([-1, 0, 10, 10], [1, 1, 1, 5], [40, 20, 150.5, 130]):
        try:
            _color_observation(color_rgb, bad_box, 'yellow')
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid source-image region was accepted')

    # The visible physical endcap must beat an interior color transition and a
    # thin mask tail. No percentile or desired ground distance is supplied.
    rgb = np.full((290, 140, 3), 135, np.uint8)
    rgb[25:220, 50:80] = [235, 205, 15]
    rgb[220:227, 50:80] = [75, 75, 75]
    rgb[270, 63] = [235, 205, 15]
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[25:227, 50:80] = 1; mask[227:275, 63] = 1
    ends, detail = _terminal_edges(rgb, mask, [0., 1.], 1, np.eye(3))
    assert ends, detail
    assert all(abs(np.mean(row['rawEnds'], axis=0)[1] - 227) < 3 for row in ends), ends
    noisy_mask = mask.copy()
    noisy_mask[227:254, 48:83] = 1
    noisy_ends, noisy_detail = _terminal_edges(rgb, noisy_mask, [.15, 1.], 1, np.eye(3))
    assert noisy_ends, noisy_detail
    assert all(abs(np.mean(row['rawEnds'], axis=0)[1] - 227) < 3 for row in noisy_ends), noisy_ends
    assert abs(noisy_detail['fittedDownRaw'][0]) < .005, noisy_detail
    assert noisy_detail['rejectionCounts']['samMaskContinuesBelow'] > 0
    obstruction = np.zeros_like(mask)
    obstruction[227:] = 1
    occluded, _ = _terminal_edges(rgb, mask, [0., 1.], 1, np.eye(3), obstruction)
    assert not occluded, 'An occlusion termination was accepted as a physical housing end'

    # Labels break each visible side into short collinear pieces. Their actual
    # observed lengths jointly establish the axis; unseen gaps add no support.
    fragmented_rgb = np.full((560, 180, 3), 135, np.uint8)
    fragmented_rgb[25:500, 55:125] = [235, 205, 15]
    fragmented_rgb[500:507, 55:125] = [75, 75, 75]
    for start in (125, 245, 365):
        fragmented_rgb[start:start+15, 45:135] = 135
    fragmented_mask = np.zeros(fragmented_rgb.shape[:2], np.uint8)
    fragmented_mask[25:507, 55:125] = 1
    fragmented, fragmented_detail = _terminal_edges(fragmented_rgb, fragmented_mask, [0., 1.], 1, np.eye(3))
    assert fragmented, fragmented_detail
    assert all(abs(np.mean(row['rawEnds'], axis=0)[1] - 507) < 3 for row in fragmented), fragmented
    assert any(len(row['sourceSegmentIndices']) > 1 for row in fragmented_detail['sideSupportGroups'])
    assert all(row['visibleLengthRawPx'] <= row['envelopeLengthRawPx'] for row in fragmented_detail['sideSupportGroups'])
    sparse_rgb = fragmented_rgb.copy(); sparse_rgb[110:450] = 135
    sparse, _ = _terminal_edges(sparse_rgb, fragmented_mask, [0., 1.], 1, np.eye(3))
    assert not sparse, 'Long unseen gaps supplied the required observed side support'

    split_rgb = np.full((640, 200, 3), 135, np.uint8)
    split_rgb[25:560, 50:150] = [235, 205, 15]
    split_rgb[560:567, 50:150] = [75, 75, 75]
    split_rgb[520:590, 64:136] = [95, 95, 95]
    split_mask = np.zeros(split_rgb.shape[:2], np.uint8); split_mask[25:567, 50:150] = 1
    terminal = np.zeros_like(split_mask); terminal[520:590, 64:136] = 1
    split, detail = _terminal_edges(split_rgb, split_mask, [0., 1.], 1, np.eye(3), terminal)
    assert len(split) == 1 and split[0]['fragmentCount'] == 2, detail
    assert all(np.linalg.norm(np.diff(segment, axis=0)) < .18 * split[0]['faceWidthRawPx'] for segment in split[0]['rawSegments'])
    assert split[0]['visibleLengthRawPx'] >= .18 * split[0]['faceWidthRawPx']
    assert len(split[0]['gapIntervalsRawPx']) == 1 and split[0]['visibleLengthRawPx'] < 30
    assert all(abs(np.mean(segment, axis=0)[1] - 567) < 2 for segment in split[0]['rawSegments'])

    # Connected front/rear yellow faces have different ends. A single exposed
    # front-face fragment uses that face's side width and needs no invented mate.
    faces_rgb = np.full((840, 220, 3), 135, np.uint8)
    faces_rgb[25:730, 30:80] = [220, 190, 10]
    faces_rgb[25:760, 82:170] = [235, 205, 15]
    faces_rgb[25:60, 30:170] = [235, 205, 15]
    faces_rgb[760:767, 82:170] = [75, 75, 75]
    faces_rgb[700:790, 107:185] = [95, 95, 95]
    faces_mask = np.zeros(faces_rgb.shape[:2], np.uint8)
    faces_mask[25:730, 30:80] = 1; faces_mask[25:767, 82:170] = 1; faces_mask[25:60, 30:170] = 1
    faces_block = np.zeros_like(faces_mask); faces_block[700:790, 107:185] = 1
    faces, _ = _terminal_edges(faces_rgb, faces_mask, [0., 1.], 1, np.eye(3), faces_block)
    front = [row for row in faces if abs(np.mean(row['rawEnds'], axis=0)[1] - 767) < 2]
    assert len(front) == 1 and front[0]['fragmentCount'] == 1
    assert .18 * front[0]['faceWidthRawPx'] <= front[0]['visibleLengthRawPx'] < .18 * front[0]['bodyWidthRawPx']
    assert front[0]['gapIntervalsRawPx'] == []

    angle = np.arange(2048) * 2 * np.pi / 2048
    circle = np.c_[.06 * np.cos(angle), .06 * np.sin(angle), np.full(len(angle), 1.4)]
    tangencies = []
    for frame in frames.values():
        uv, _ = _project(circle, frame)
        raw = _pixels(uv, np.linalg.inv(A))
        tangencies.extend(_tangencies(raw, frame, up))
    center, radius, residual, _ = _fit_circle(tangencies, frames, up)
    assert np.linalg.norm(center) < 1e-5 and abs(radius - .06) < 1e-6
    assert max(residual) < .001, residual
    stale_anchor = np.array([.8, 0., 1.4])
    changed_tangencies = []
    for frame in replacement.values():
        raw = _pixels(_project(circle, frame)[0], np.linalg.inv(A))
        stale_pixel = _pixels(_project([stale_anchor], frame)[0], np.linalg.inv(A))[0]
        assert stale_pixel[0] < raw[:, 0].min() or stale_pixel[0] > raw[:, 0].max()
        rows = _tangencies(raw, frame, up)
        assert any(row['normal'] @ (stale_anchor - frame['pose'][:3, 3]) < 0 for row in rows), 'Synthetic anchor must expose the historical sign bug'
        changed_tangencies.extend(rows)
    changed_center, changed_radius, _, _ = _fit_circle(changed_tangencies, replacement, up)
    assert np.linalg.norm(changed_center) < 1e-5 and abs(changed_radius - .06) < 1e-6

    known = {'features': {'wholeComponentHeightM': .1, 'mainBodyDiameterM': .085, 'redActuatorDiameterM': .04},
             'scopeStatus': 'pending_confirmation'}
    wrapped = {'reference': known, 'evaluation': {'synthetic_private_distance': 9876}}
    changed = {'reference': known, 'evaluation': {'synthetic_private_distance': -1234}}
    assert json.dumps(_reference(wrapped), sort_keys=True) == json.dumps(_reference(changed), sort_keys=True)
    assert 'evaluation' not in json.dumps(_reference(wrapped))
    with tempfile.TemporaryDirectory(prefix='joint-reference-check-') as directory:
        camera_path = Path(directory) / 'cameras.json'
        camera_path.write_text('{}')
        joint = {'reference': _reference(wrapped), 'status': 'available', 'mPerNative': .6,
                 'cameraSha256': hashlib.sha256(camera_path.read_bytes()).hexdigest()}
        assert _joint_reference(joint, _reference(wrapped), camera_path)['mPerNative'] == .6
        for invalid in ('diameter', 'camera', 'status', 'scale'):
            bad = copy.deepcopy(joint)
            if invalid == 'diameter':
                bad['reference']['features']['redActuatorDiameterM'] = .041
            elif invalid == 'camera':
                bad['cameraSha256'] = '0' * 64
            elif invalid == 'status':
                bad['status'] = 'unsupported'
            else:
                bad['mPerNative'] = float('nan')
            try:
                _joint_reference(bad, _reference(wrapped), camera_path)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Joint reference {invalid} mismatch was accepted')
        joint.update(status='unsupported', mPerNative=None)
        assert _joint_reference(joint, _reference(wrapped), camera_path)['mPerNative'] is None

    obj = {'status': 'available', 'reason': None, 'pointNative': [0., 0., .45], 'footNative': [0., 0., 0.],
           'heightNative': .45, 'rangeNative': [.44, .46], 'sourcePhotos': [1, 2, 3],
           'localFloor': {'normal': [0., 0., 1.], 'offset': 0.}}
    objects = [{'id': ident, **obj} for ident in TARGETS]
    result = {'schemaVersion': 1, 'reference': _reference(wrapped),
              'routes': {key: _route(key, objects, .6) for key in 'ABCD'}}
    _validate_results(result)
    joint_result = copy.deepcopy(result)
    joint_result['jointReference'] = {'status': 'available', 'mPerNative': .7}
    joint_result['routes']['J'] = _route('Joint sizes and cameras', objects, .7)
    _validate_results(joint_result)
    joint_result['routes']['J']['scaleMPerNative'] = .6
    try:
        _validate_results(joint_result)
    except ValueError:
        pass
    else:
        raise AssertionError('Joint route accepted a scale from another camera fit')
    joint_result['jointReference'] = {'status': 'unsupported', 'mPerNative': None}
    joint_result['routes']['J'] = _route('Rejected joint sizes and cameras', objects, None)
    _validate_results(joint_result)
    assert all(row['heightM'] is None for row in joint_result['routes']['J']['objects'])
    with tempfile.TemporaryDirectory(prefix='workcell-metrology-check-') as temporary:
        path = Path(temporary) / 'results.json'
        path.write_text(json.dumps(result, allow_nan=False))
        _validate_results(json.loads(path.read_text()))
    broken = copy.deepcopy(result)
    broken['routes']['D']['objects'][0]['heightM'] += .01
    try:
        _validate_results(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('Metric/native geometry disagreement was accepted')
    broken = copy.deepcopy(result)
    broken['routes']['C']['objects'][0]['footNative'][2] = .02
    try:
        _validate_results(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('Foot outside the saved plane was accepted')

    # Integration persists one native plane for inference and the viewer,
    # preserving the floor model node even when it has a nonidentity transform.
    # An absent metric scale never prevents native physical-edge geometry.
    with tempfile.TemporaryDirectory(prefix='physical-clearance-bridge-') as directory:
        root = Path(directory)
        floor = trimesh.Scene()
        vertices = np.array([[-1., -1., .2], [1., -1., .2], [1., 1., .2], [-1., 1., .2]])
        transform = np.eye(4); transform[:3, 3] = [.3, .4, .1]
        floor.add_geometry(trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False),
                           node_name='saved-floor-node', geom_name='saved-floor-geometry', transform=transform)
        floor_path = root / 'floor-fitted.glb'
        floor_path.write_bytes(floor.export(file_type='glb'))
        old_floor = {'normal': [0., 0., 1.], 'offset': -.3}
        ground = {'status': 'available', 'normal': normal.tolist(), 'offset': 0.,
                  'supportPoints': 24, 'residualP95Native': .002, 'sourcePhotos': [1, 2], 'method': 'synthetic source support'}
        physical = {'ground': ground, 'objects': [grounded, {'id': 'post-box-2', **metrology._unavailable('one source view')}],
                    'coordinateSystem': 'MapAnything native'}
        synthetic_geometry = {'floor': old_floor, 'anchor': {'mPerNative': None}}
        with patch.object(metrology, '_load', return_value=(synthetic_geometry, {}, {}, {}, [], [])), \
             patch.object(metrology, 'source_physical_clearances', return_value=physical), \
             patch.object(metrology, '_overlays', return_value=[]):
            metrology.apply_source_clearances(root, [root / f'source-{i}.jpg' for i in range(4)])
        saved = json.loads((root / 'geometry.json').read_text())
        assert saved['floor']['normal'] == ground['normal'] and saved['floor']['offset'] == ground['offset']
        assert saved['physicalClearances'] == json.loads((root / 'physical-clearances.json').read_text())
        assert saved['physicalClearances']['objects'][1]['heightNative'] is None, 'Single-view evidence became a physical distance'
        loaded = trimesh.load(floor_path, force='scene', process=False)
        assert loaded.graph.nodes_geometry == ['saved-floor-node'], 'Floor node identity changed after objects.json was built'
        matrix, name = loaded.graph['saved-floor-node']
        actual = trimesh.transform_points(loaded.geometry[name].vertices, matrix)
        native = trimesh.transform_points(vertices, transform)
        expected = native - (native @ normal)[:, None] * normal
        assert np.allclose(actual, expected, atol=1e-6) and np.max(abs(actual @ normal)) < 1e-6
        before = floor_path.read_bytes()
        unsupported = {'ground': {'status': 'unsupported', 'normal': None, 'offset': None},
                       'objects': [{'id': 'post-box-1', **metrology._unavailable('no source edge')}]}
        with patch.object(metrology, '_load', return_value=(saved, {}, {}, {}, [], [])), \
             patch.object(metrology, 'source_physical_clearances', return_value=unsupported), \
             patch.object(metrology, '_overlays', return_value=[]):
            metrology.apply_source_clearances(root, [root / f'source-{i}.jpg' for i in range(4)])
        assert floor_path.read_bytes() == before, 'Unsupported ground changed the display plane'
    print('PASS: crop/C2W, RGB contour refinement, fence plane identity, stale-anchor circle signs, noisy-line gauge, fragmented sides, split endcap/gaps, occlusion, floor sensitivity, source-ground bridge, truth exclusion, JSON/metric contract')


if __name__ == '__main__':
    main()
