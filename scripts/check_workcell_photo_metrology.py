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
    assert _legacy({'id': 'fence-1'}, selected_plane)['heightNative'] == .7
    for invalid in (True, -1, .5, '1'):
        try:
            _legacy({'id': 'fence-1', 'geometryPlaneIndex': invalid}, selected_plane)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid fence plane identity was accepted')

    # Both physical rail lower faces follow their own plane through the same
    # raw-pixel triangulation and common-ground measurement path. The old
    # association clearances deliberately disagree with the true source edges.
    fence_geometry = {**selected_plane, 'fence': {'planes': [], 'beams': []}}
    for plane, depth in enumerate((0., .3)):
        fence_geometry['fence']['planes'].append({'normal': [0., 1., 0.], 'offset': -depth})
        for photo, frame in frames.items():
            raw_edges = []
            for height in (.50, .45):
                line = np.array([[-.15, depth, height], [.15, depth, height]])
                raw_edges.append(_pixels(_project(line, frame)[0], np.linalg.inv(A)).tolist())
            fence_geometry['fence']['beams'].append({'id': f'plane-{plane}-photo-{photo}',
                'plane': plane, 'horizontal': True, 'sourcePhoto': photo,
                'heightNative': .45, 'rawEdges': raw_edges})
    fence_catalog = {ident: {'id': ident, 'observations': []} for ident in TARGETS}
    association = {ident: {'pointNative': [0., 0., .2]} for ident in TARGETS}
    fence_frames = {photo: {**frame, 'initialK': frame['K'], 'initialPose': frame['pose']}
                    for photo, frame in frames.items()}
    fence_drawings = {photo: {'selected': [], 'candidates': []} for photo in frames}
    def detected_bottoms(item, geometry, frames, up):
        # RGB/topology extraction has its real-image algorithm check in
        # check_workcell_fence_bottom.py; this check isolates its shared caller.
        plane = metrology._fence_plane_index(item)
        rows = []
        for beam in geometry['fence']['beams']:
            if beam['plane'] == plane:
                raw = beam['rawEdges'][1]
                rows.append({'photo': beam['sourcePhoto'], 'rawEnds': raw,
                             'uv': _pixels(raw, A).tolist(), 'sourceBeam': beam['id'],
                             'planeIndex': plane,
                             'identityEvidence': {'independentlySupported': True}})
        return rows, []
    with patch('workcell_fence_bottom.fence_bottom_candidates', side_effect=detected_bottoms):
        source_edges, _ = metrology._object_edges(fence_catalog, {}, fence_geometry,
                                                  fence_frames, up, association, fence_drawings)
    # A whole-face solver requests only RGB post candidates. It must neither
    # access a fence catalog/geometry entry nor run unused 3D line matching.
    post_ids = ('post-box-1', 'post-box-2')
    post_catalog = {ident: {'id': ident, 'observations': [{'photo': 1, 'source': 'instance 0'}]}
                    for ident in post_ids}
    post_frames = {1: {**fence_frames[1], 'rgb': np.zeros((40, 40, 3), np.uint8)}}
    post_legacy = {ident: association[ident] for ident in post_ids}
    terminal = {'photo': 1, 'rawEnds': [[5., 5.], [10., 5.]], 'uv': [[5., 5.], [10., 5.]]}
    fresh_drawings = lambda: {1: {'selected': [], 'candidates': []}}
    with patch.object(metrology, '_response', return_value={'rle': [None]}), \
            patch.object(metrology, '_raw_mask', return_value=np.ones((40, 40), np.uint8)), \
            patch.object(metrology, '_terminal_edges', return_value=([terminal], {'source': 'same RGB lines'})) as extract, \
            patch('workcell_fence_bottom.fence_bottom_candidates', side_effect=AssertionError('Unrequested fence work')), \
            patch.object(metrology, '_match_edges', side_effect=AssertionError('Unused 3D matching')):
        only_edges, only_diagnostics = metrology._object_edges(post_catalog, {'prompts': []}, {}, post_frames,
            up, post_legacy, fresh_drawings(), targets=post_ids, match_edges=False)
        assert extract.call_count == 4  # bottom and top, once for each post
        assert all(row['status'] == 'candidates_only' and row['bottomEdge'] is None for row in only_edges)
    with patch.object(metrology, '_response', return_value={'rle': [None]}), \
            patch.object(metrology, '_raw_mask', return_value=np.ones((40, 40), np.uint8)), \
            patch.object(metrology, '_terminal_edges', return_value=([terminal], {'source': 'same RGB lines'})), \
            patch.object(metrology, '_match_edges', return_value={'observations': [terminal]}) as match:
        matched_edges, matched_diagnostics = metrology._object_edges(post_catalog, {'prompts': []}, {}, post_frames,
            up, post_legacy, fresh_drawings(), targets=post_ids)
        assert match.call_count == 4 and all(row['status'] == 'edge_supported' for row in matched_edges)
    assert only_diagnostics == matched_diagnostics, 'Skipping unused line fits changed the RGB source candidates'
    for invalid_targets in (('post-box-1', 'post-box-1'), ('unknown',)):
        try:
            metrology._object_edges({}, {}, {}, {}, up, {}, {}, targets=invalid_targets, match_edges=False)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid requested target set was silently accepted')
    fences = [row for row in source_edges if row['id'].startswith('fence-')]
    assert {row['id'] for row in fences} == {'fence-0', 'fence-1'}
    for plane, row in enumerate(fences):
        assert row['status'] == 'unsupported', row
        assert row['bottomEdge']['physicalPromotionAllowed'] is False
        assert np.allclose(row['bottomEdge']['pointNative'][1:], [plane * .3, .45])
        assert all(obs['sourceBeam'].startswith(f'plane-{plane}-') for obs in row['bottomEdge']['observations'])
    shared_floor = {'status': 'available', 'normal': [0., 0., 1.], 'offset': 0.,
                    'patches': [{'objectId': row['id'], 'status': 'available', 'residualP95Native': 0.}
                                for row in fences]}
    rail_distances = _measure(fences, shared_floor, frames, up)
    assert all(row['heightNative'] is None for row in rail_distances)
    assert all(np.isclose(row['bottomEdge']['pointNative'][2], .45) for row in rail_distances)
    # Isolate measurement arithmetic with the known synthetic identities;
    # the actual RGB caller above must not promote an unverified face.
    rail_distances = _measure([{**row, 'status': 'edge_supported'} for row in fences], shared_floor, frames, up)
    assert all(np.isclose(row['heightNative'], .45) for row in rail_distances)
    assert all(row['localFloor']['normal'] == shared_floor['normal'] for row in rail_distances)

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
    assert all(len(set(row['faceSideIds'])) == 2 for row in ends)
    # A real RGB edge can have a three-pixel desaturated yellow blur outside it.
    # Test the detected edge location, not a newly chosen threshold contour.
    blurred_rgb = np.full((280, 140, 3), 135, np.uint8)
    blurred_rgb[25:220, 50:80] = [235, 205, 15]
    blurred_rgb[220:223, 50:80] = cv2.cvtColor(np.uint8([[[25, 75, 200]]]), cv2.COLOR_HSV2RGB)[0, 0]
    blurred_mask = np.zeros(blurred_rgb.shape[:2], np.uint8); blurred_mask[25:223, 50:80] = 1
    blurred_lines = np.array([[[50., 25.], [50., 222.]], [[80., 25.], [80., 222.]], [[50., 219.5], [80., 219.5]]])
    with patch.object(cv2, 'createLineSegmentDetector') as detector:
        detector.return_value.detect.return_value = [(blurred_lines - [38., 13.]).astype(np.float32).reshape(-1, 1, 4)]
        blurred, _ = _terminal_edges(blurred_rgb, blurred_mask, [0., 1.], 1, np.eye(3))
    assert blurred and abs(np.mean(blurred[0]['rawEnds'], axis=0)[1] - 219.5) < .01, blurred
    # One long side establishes orientation; the observed short opposite side
    # still bounds the terminal face. Do not fabricate missing side pixels.
    local_rgb = np.full((560, 180, 3), 135, np.uint8); local_rgb[25:500, 50:110] = [235, 205, 15]
    local_mask = np.zeros(local_rgb.shape[:2], np.uint8); local_mask[25:500, 50:110] = 1
    local_lines = np.array([[[50., 25.], [50., 500.]], [[110., 430.], [110., 500.]], [[50., 500.], [110., 500.]]])
    with patch.object(cv2, 'createLineSegmentDetector') as detector:
        detector.return_value.detect.return_value = [(local_lines - [30., 5.]).astype(np.float32).reshape(-1, 1, 4)]
        local_end, _ = _terminal_edges(local_rgb, local_mask, [0., 1.], 1, np.eye(3))
    assert local_end and abs(np.mean(local_end[0]['rawEnds'], axis=0)[1] - 500) < .01, local_end
    # Interior seams must not permanently split two real bottom fragments
    # into different narrow faces. Keep the outer observed-side hypothesis;
    # its unobserved middle span remains a gap, never manufactured evidence.
    seam_rgb = np.full((640, 200, 3), 135, np.uint8)
    seam_rgb[20:560, 30:150] = [235, 205, 15]
    seam_rgb[560:567, 30:150] = [75, 75, 75]
    seam_rgb[60:560, 69:72] = [95, 95, 95]
    seam_rgb[60:560, 109:112] = [95, 95, 95]
    seam_mask = np.zeros(seam_rgb.shape[:2], np.uint8); seam_mask[20:567, 30:150] = 1
    seam_lines = np.array([[[30., 20.], [30., 567.]], [[150., 20.], [150., 567.]],
                           [[70., 60.], [70., 560.]], [[110., 60.], [110., 560.]],
                           [[30., 567.], [65., 567.]], [[115., 567.], [150., 567.]]])
    with patch.object(cv2, 'createLineSegmentDetector') as detector:
        detector.return_value.detect.return_value = [seam_lines.astype(np.float32).reshape(-1, 1, 4)]
        seam_ends, seam_detail = _terminal_edges(seam_rgb, seam_mask, [0., 1.], 1, np.eye(3))
    broad = [row for row in seam_ends if abs(row['faceWidthRawPx'] - 120.) < .01]
    assert len(broad) == 1 and broad[0]['fragmentCount'] == 2, (seam_ends, seam_detail)
    assert broad[0]['gapIntervalsRawPx'] and abs(broad[0]['visibleLengthRawPx'] - 70.) < .01
    assert len({tuple(row['faceSideIds']) for row in seam_ends}) > 1, 'Hypothesis became an asserted physical face'

    # The final output cap must not undo broad-face retention. Twelve genuine
    # internal reflection lines create >12 plausible pairs whose short widths
    # score higher than the outer face's two disjoint 20px end fragments.
    many_rgb = np.full((980, 260, 3), 135, np.uint8)
    many_rgb[20:920, 30:210] = [235, 205, 15]
    many_rgb[920:927, 30:210] = [75, 75, 75]
    many_mask = np.zeros(many_rgb.shape[:2], np.uint8); many_mask[20:927, 30:210] = 1
    interiors = np.linspace(55., 185., 12)
    for x in interiors:
        many_rgb[60:920, round(x):round(x)+1] = [210, 180, 15]
    many_lines = np.array([[[30., 20.], [30., 927.]], [[210., 20.], [210., 927.]],
                           *[[[x, 60.], [x, 920.]] for x in interiors],
                           [[30., 927.], [50., 927.]], [[190., 927.], [210., 927.]]])
    with patch.object(cv2, 'createLineSegmentDetector') as detector:
        detector.return_value.detect.return_value = [many_lines.astype(np.float32).reshape(-1, 1, 4)]
        many_ends, many_detail = _terminal_edges(many_rgb, many_mask, [0., 1.], 1, np.eye(3))
    assert many_detail['candidateCount'] > 12 and len(many_ends) == 12, many_detail
    outer = [row for row in many_ends if row['faceSideIds'] == [0, 1]]
    assert len(outer) == 1 and outer[0]['fragmentCount'] == 2, [row['faceWidthRawPx'] for row in many_ends]
    assert abs(outer[0]['visibleLengthRawPx'] - 40.) < .01 and len(outer[0]['gapIntervalsRawPx']) == 1
    assert len({tuple(row['faceSideIds']) for row in many_ends}) == 12
    assert many_detail['candidateLimitDiscarded'] == len(many_detail['omittedCandidateFaces'])
    assert all(row['partIdentity']['frontOrSide'] == 'unresolved' for row in many_ends)

    # A much longer adjacent wing used to remove this broad terminal through
    # both the global near-end and global color-continuation conditions.
    step_rgb = np.full((880, 210, 3), 135, np.uint8)
    step_rgb[20:600, 30:130] = [235, 205, 15]
    step_rgb[20:800, 136:170] = [235, 205, 15]
    step_rgb[20:50, 30:170] = [235, 205, 15]
    step_mask = np.zeros(step_rgb.shape[:2], np.uint8)
    step_mask[20:600, 30:130] = 1; step_mask[20:800, 136:170] = 1; step_mask[20:50, 30:170] = 1
    step_lines = np.array([[[30., 20.], [30., 600.]], [[130., 20.], [130., 600.]],
                           [[136., 20.], [136., 800.]], [[170., 20.], [170., 800.]],
                           [[30., 600.], [130., 600.]], [[136., 800.], [170., 800.]]])
    with patch.object(cv2, 'createLineSegmentDetector') as detector:
        detector.return_value.detect.return_value = [step_lines.astype(np.float32).reshape(-1, 1, 4)]
        stepped, step_detail = _terminal_edges(step_rgb, step_mask, [0., 1.], 1, np.eye(3))
    assert any(abs(np.mean(row['rawEnds'], axis=0)[1] - 600.) < .01 for row in stepped), step_detail
    assert any(abs(np.mean(row['rawEnds'], axis=0)[1] - 800.) < .01 for row in stepped), step_detail
    assert all(not row['partIdentity']['commonBottomPlaneSupported'] for row in stepped)
    assert any(row['reason'] == 'notNearColorEnd' and row.get('faceSideIds') == [0, 3]
               and row['rawEnds'] == step_lines[4].tolist() for row in step_detail['nearEndRejectedEdges']), step_detail
    assert step_detail['rejectionCounts']['topologyChecks'] <= step_detail['rejectionCounts']['facePairHypotheses']

    # Two short fragments of ONE long side diverge when extrapolated. Their
    # apparent 6.2px span formerly passed as two sides of a fictitious face.
    degenerate_rgb = np.full((240, 100, 3), 135, np.uint8)
    degenerate_rgb[20:206, 42:59] = [235, 205, 15]
    degenerate_mask = np.zeros(degenerate_rgb.shape[:2], np.uint8)
    degenerate_mask[20:206, 42:59] = 1
    raw_lines = np.array([[[50., 20.], [50., 193.]], [[49.65, 195.], [48.55, 199.]],
                          [[50.35, 195.], [51.45, 199.]], [[46.9, 205.], [53.1, 205.]]])
    with patch.object(cv2, 'createLineSegmentDetector') as detector:
        detector.return_value.detect.return_value = [(raw_lines - [30., 8.]).astype(np.float32).reshape(-1, 1, 4)]
        degenerate, diagnostic = _terminal_edges(degenerate_rgb, degenerate_mask, [0., 1.], 1, np.eye(3))
    assert not degenerate and diagnostic['rejectionCounts']['sameSideGroup'] == 1, diagnostic
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
    assert front[0]['partIdentity']['frontOrSide'] == 'unresolved'
    assert not front[0]['partIdentity']['commonBottomPlaneSupported'], 'Stepped faces became one physical plane'
    assert any(row['partIdentity']['otherFaceColorFurtherAlongImageAxis'] for row in faces), 'An image-higher face was silently discarded instead of tagged'

    # A narrow face beside a wide flange must not borrow the union width to
    # label a detached mounting bracket as its physical terminal edge.
    bracket_rgb = np.full((620, 200, 3), 135, np.uint8)
    bracket_rgb[25:480, 30:110] = [220, 180, 10]
    bracket_rgb[25:520, 112:145] = [235, 205, 15]
    bracket_rgb[25:60, 30:145] = [235, 205, 15]
    bracket_rgb[555:563, 112:145] = [65, 65, 65]
    bracket_mask = np.zeros(bracket_rgb.shape[:2], np.uint8)
    bracket_mask[25:563, 30:145] = 1
    bracket_ends, _ = _terminal_edges(bracket_rgb, bracket_mask, [0., 1.], 1, np.eye(3))
    assert any(abs(np.mean(row['rawEnds'], axis=0)[1] - 520) < 2 for row in bracket_ends), bracket_ends
    assert all(min(abs(np.mean(row['rawEnds'], axis=0)[1] - y) for y in (480, 520)) < 2 for row in bracket_ends), bracket_ends

    # Green/yellow wire and a nearby neutral bracket both lie inside the broad
    # SAM envelope. Neither is continuously connected housing/endcap support.
    contact_rgb = np.full((620, 180, 3), 135, np.uint8)
    contact_rgb[25:520, 50:130] = [235, 205, 15]
    contact_rgb[520:528, 50:130] = [75, 75, 75]
    contact_mask = np.zeros(contact_rgb.shape[:2], np.uint8); contact_mask[25:580, 50:130] = 1
    cable_rgb = contact_rgb.copy(); cable_rgb[528:555, 82:101] = [120, 150, 70]
    cable_ends, cable_detail = _terminal_edges(cable_rgb, contact_mask, [0., 1.], 1, np.eye(3))
    assert cable_ends and all(abs(np.mean(row['rawEnds'], axis=0)[1] - 528) < 2 for row in cable_ends), cable_ends
    assert all(row['directHousingContactFraction'] >= .6 and row['housingContactDepthRawPx'] > 3 for row in cable_ends)
    nearby_rgb = contact_rgb.copy(); nearby_rgb[538:544, 50:130] = [65, 65, 65]
    nearby_ends, nearby_detail = _terminal_edges(nearby_rgb, contact_mask, [0., 1.], 1, np.eye(3))
    assert nearby_ends and all(abs(np.mean(row['rawEnds'], axis=0)[1] - 528) < 2 for row in nearby_ends), nearby_ends
    assert nearby_detail['rejectionCounts']['disconnectedFaceEnd'] > 0, nearby_detail
    json.dumps({'cable': cable_ends, 'nearby': nearby_ends, 'faces': faces}, allow_nan=False)

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

    # Publishing a tilted, translated floor refreshes all derived clearances
    # without moving source points or meshes to force equal object heights.
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
        ground = {'status': 'available', 'normal': normal.tolist(), 'offset': -.12,
                  'supportPoints': 24, 'residualP95Native': .002, 'sourcePhotos': [1, 2], 'method': 'synthetic source support',
                  'patches': [{'objectId': 'post-box-1', 'status': 'available', 'residualP95Native': .002}]}
        physical = {'ground': ground, 'objects': _measure([{'id': 'post-box-1', 'status': 'edge_supported',
                    'bottomEdge': tested_edge, 'topEdge': None}], ground, frames, up) +
                    [{'id': 'post-box-2', **metrology._unavailable('one source view')}],
                    'coordinateSystem': 'MapAnything native'}
        clearances = [{'id': f'fence-plane-{i}-lower-rail', 'pointNative': [x, .2, .6],
                       'footNative': [x, .2, .3], 'heightNative': .3, 'sourcePhotos': [1, 2],
                       'observedViewHeightRangeNative': [.29, .31]}
                      for i, x in enumerate((-.8, .8))]
        clearances[0]['observedViewPointsNative'] = [[-.9, .2, .59], [-.7, .2, .61]]
        beams = [{'id': f'beam-{i}', 'pointNative': row['pointNative'], 'footNative': row['footNative'],
                  'heightNative': row['heightNative'], 'endsNative': [[x, .1, .65], [x, .3, .65]]}
                 for i, (row, x) in enumerate(zip(clearances, (-.8, .8)))]
        synthetic_geometry = {'floor': old_floor, 'anchor': {'mPerNative': None},
                              'clearances': copy.deepcopy(clearances), 'fence': {'beams': copy.deepcopy(beams)}}
        source_path = root / 'fence-fitted.glb'
        source_bytes = trimesh.creation.box(extents=[.2, .3, 1.]).export(file_type='glb')
        source_path.write_bytes(source_bytes)
        with patch.object(metrology, '_load', return_value=(synthetic_geometry, {}, {}, {}, [], [])), \
             patch.object(metrology, 'source_physical_clearances', return_value=physical), \
             patch.object(metrology, '_overlays', return_value=[]):
            metrology.apply_source_clearances(root, [root / f'source-{i}.jpg' for i in range(4)])
        saved = json.loads((root / 'geometry.json').read_text())
        assert saved['floor']['normal'] == ground['normal'] and saved['floor']['offset'] == ground['offset']
        assert saved['physicalClearances'] == json.loads((root / 'physical-clearances.json').read_text())
        assert saved['physicalClearances']['objects'][1]['heightNative'] is None, 'Single-view evidence became a physical distance'
        for original, refreshed in zip(clearances + beams, saved['clearances'] + saved['fence']['beams']):
            point, foot = np.asarray(refreshed['pointNative']), np.asarray(refreshed['footNative'])
            assert refreshed['pointNative'] == original['pointNative'], 'Ground update moved an observed point'
            assert refreshed.get('endsNative') == original.get('endsNative'), 'Ground update moved source beam ends'
            assert np.isclose(refreshed['heightNative'], point @ normal + ground['offset'])
            assert np.isclose(foot @ normal + ground['offset'], 0., atol=1e-9), 'Foot still uses the old ground'
            assert np.allclose(point - foot, refreshed['heightNative'] * normal)
        assert saved['clearances'][0]['heightNative'] != saved['clearances'][1]['heightNative'], 'Ground update forced equal heights'
        heights = np.asarray(clearances[0]['observedViewPointsNative']) @ normal + ground['offset']
        assert np.allclose(saved['clearances'][0]['observedViewHeightRangeNative'], [heights.min(), heights.max()])
        assert saved['clearances'][1]['observedViewHeightRangeNative'] is None, 'Old scalar range survived a changed ground'
        legacy = _legacy({'id': 'fence-0', 'geometryPlaneIndex': 1}, saved)
        assert legacy['rangeNative'] is None
        assert _route('saved feature with unknown view range', [legacy], .6)['objects'][0]['rangeM'] is None
        assert source_path.read_bytes() == source_bytes, 'Ground publication changed the native source GLB'
        loaded = trimesh.load(floor_path, force='scene', process=False)
        assert loaded.graph.nodes_geometry == ['saved-floor-node'], 'Floor node identity changed after objects.json was built'
        matrix, name = loaded.graph['saved-floor-node']
        actual = trimesh.transform_points(loaded.geometry[name].vertices, matrix)
        native = trimesh.transform_points(vertices, transform)
        expected = native - (native @ normal + ground['offset'])[:, None] * normal
        assert np.allclose(actual, expected, atol=1e-6) and np.max(abs(actual @ normal + ground['offset'])) < 1e-6
        with patch.object(metrology, '_load', return_value=(copy.deepcopy(saved), {}, {}, {}, [], [])), \
             patch.object(metrology, 'source_physical_clearances', return_value=copy.deepcopy(physical)), \
             patch.object(metrology, '_overlays', return_value=[]):
            metrology.apply_source_clearances(root, [root / f'source-{i}.jpg' for i in range(4)])
        repeated = json.loads((root / 'geometry.json').read_text())
        assert repeated == saved, 'Repeated publication changed saved ground-derived data'
        reloaded = trimesh.load(floor_path, force='scene', process=False)
        repeated_transform, repeated_name = reloaded.graph['saved-floor-node']
        assert np.allclose(trimesh.transform_points(reloaded.geometry[repeated_name].vertices, repeated_transform), actual, atol=1e-6)
        assert source_path.read_bytes() == source_bytes
        before = floor_path.read_bytes()
        unsupported = {'ground': {'status': 'unsupported', 'normal': None, 'offset': None},
                       'objects': [{'id': 'post-box-1', **metrology._unavailable('no source edge')}]}
        with patch.object(metrology, '_load', return_value=(saved, {}, {}, {}, [], [])), \
             patch.object(metrology, 'source_physical_clearances', return_value=unsupported), \
             patch.object(metrology, '_overlays', return_value=[]):
            metrology.apply_source_clearances(root, [root / f'source-{i}.jpg' for i in range(4)])
        assert floor_path.read_bytes() == before, 'Unsupported ground changed the display plane'
    print('PASS: crop/C2W, RGB contour refinement, both fence lower edges and common ground, distinct terminal sides, stale-anchor circle signs, noisy-line gauge, fragmented sides, split endcap/gaps, connected housing contact, cable/bracket rejection, stepped-face ambiguity, occlusion, floor sensitivity, source-ground bridge, truth exclusion, JSON/metric contract')


if __name__ == '__main__':
    main()
