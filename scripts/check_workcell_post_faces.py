"""Run: PYTHONPATH=.:scripts python scripts/check_workcell_post_faces.py."""
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import trimesh


def check():
    assert importlib.util.find_spec('workcell_post_faces'), 'Source-face builder is missing'
    from workcell_post_faces import fit_face, _comparison
    from workcell_photo_geometry import _intersect
    from workcell_photo_metrology import _interval_union, _pixels, _terminal_edges
    from workcell_photo_objects import _project

    K = np.array([[80., 0, 60], [0, 80, 80], [0, 0, 1]])
    A = np.array([[.5, 0, -.25], [0, .5, -.25], [0, 0, 1]])
    pose = trimesh.transformations.rotation_matrix(.2, [0., 1., 0.])
    pose[:3, 3] = [1.2, -.3, .7]
    pn = np.array([-.2, 0, 1.]); pn /= np.linalg.norm(pn)
    normal = pose[:3, :3] @ pn
    offset = -3 / np.sqrt(1.04) - normal @ pose[:3, 3]
    y, x = np.indices((160, 120))
    points = _intersect(np.c_[x.ravel(), y.ravel()], K, pose, normal, offset).reshape(160, 120, 3)
    frame = {'photo': 1, 'K': K, 'pose': pose, 'A': A, 'points': points,
             'valid': np.ones((160, 120), bool), 'rgb': np.full((320, 240, 3), [245, 195, 30], np.uint8)}
    mask = np.zeros((320, 240), np.uint8); mask[70:271, 90:131] = 1
    sides = [[[90., 70.], [90., 270.]], [[130., 70.], [130., 270.]]]
    bottom = {'photo': 1, 'rawEnds': [[90., 270.], [130., 270.]], 'faceSideIds': [1, 2],
              'faceSideEdgesRaw': sides, 'rawSegments': [[[90., 270.], [130., 270.]]], 'gapIntervalsRawPx': []}
    top = {**bottom, 'rawEnds': [[90., 70.], [130., 70.]],
           'rawSegments': [[[90., 70.], [130., 70.]]]}
    up = pose[:3, :3] @ np.array([0., -1., 0.])
    ground = {'normal': (2 * up).tolist(), 'offset': float(2 * (3. - up @ pose[:3, 3]))}
    mesh, record = fit_face(frame, mask, bottom, top, ground)
    uv, depth = _project(mesh.vertices, frame)
    assert (depth > 0).all() and np.isfinite(mesh.vertices).all()
    assert not mesh.is_watertight and record['thicknessNative'] is None
    assert mesh.visual.kind == 'texture' and record['appearance']['geometryUnchanged']
    assert record['appearance']['photoTexelFraction'] > .1
    assert record['mPerNative'] is None and record['measurementStatus'] == 'conditional_single_photo'
    raw = _pixels(uv, np.linalg.inv(A))
    assert np.allclose(raw.min(0), [90, 70], atol=1e-5)
    assert np.allclose(raw.max(0), [130, 270], atol=1e-5)
    bottom_native = np.asarray(record['bottomSegmentsNative'])[0]
    expected = _intersect(_pixels(bottom['rawEnds'], A), K, pose, normal, offset)
    assert np.allclose(bottom_native, expected, atol=1e-7)
    assert all(np.linalg.norm(mesh.vertices - point, axis=1).min() < 1e-7 for point in expected)
    measured = record['endpointEstimate']
    point, foot = np.asarray(measured['pointNative']), np.asarray(measured['footNative'])
    assert min(np.linalg.norm(point - endpoint) for endpoint in bottom_native) < 1e-7
    unit_offset = ground['offset'] / 2
    assert abs(foot @ up + unit_offset) < 1e-7
    assert np.allclose(point - foot, measured['heightNative'] * up, atol=1e-7)
    assert record['sourceReprojectionMaxRawPx'] < 1e-7
    diagnostic = {'bottomCandidates': [bottom], 'topCandidates': [top]}
    row, _ = _comparison(record, frame, diagnostic, None)
    assert row['status'] == 'by_construction' and row['bestSameSidePair']['maxSymmetricP95RawPx'] < 1e-7
    second = {**frame, 'photo': 2, 'pose': pose.copy()}
    second['pose'][:3, 3] += .2 * pose[:3, 0]
    other = {key: [{**edge, 'photo': 2} for edge in edges] for key, edges in diagnostic.items()}
    row, _ = _comparison(record, second, other, None)
    assert row['status'] == 'not_supported' and row['bestSameSidePair']['maxSymmetricP95RawPx'] > 3
    other['bottomCandidates'][0]['gapIntervalsRawPx'] = [[-5, 5]]
    row, _ = _comparison(record, second, other, None)
    assert row['bestSameSidePair'] is None
    # OpenCV LSD supplies float32 projections. Overlapping side fragments must
    # retain numeric JSON fields throughout the real detector/build schema.
    intervals = _interval_union([list(pair) for pair in np.array([[1, 3], [2, 4]], np.float32)])
    group = {'visibleLengthRawPx': sum(b - a for a, b in intervals),
             'envelopeLengthRawPx': intervals[-1][1] - intervals[0][0]}
    rgb = np.full_like(frame['rgb'], 20); rgb[70:271, 90:131] = [245, 195, 30]
    rgb[75:260, 110:111] = 20
    detected, detail = _terminal_edges(rgb, mask, np.array([0., 1.]), 1, A)
    payload = {'candidates': [{**record, 'views': [row]}], 'rejections': [],
               'sourceEdgeDiagnostics': [{'id': 'post-box-1', 'bottomCandidates': detected,
                   'views': [{'photo': 1, 'bottom': detail, 'top': {'sideSupportGroups': [group]}}]}]}
    saved = json.loads(json.dumps(payload, allow_nan=False))
    assert saved['sourceEdgeDiagnostics'][0]['views'][0]['top']['sideSupportGroups'][0]['visibleLengthRawPx'] == 3.
    for invalid in ('gap', 'different face', 'no plane', 'nonplanar'):
        changed = copy.deepcopy(bottom); bad_frame = dict(frame)
        if invalid == 'gap': changed['gapIntervalsRawPx'] = [[-5, 5]]
        if invalid == 'different face': changed['faceSideIds'] = [1, 3]
        if invalid == 'no plane': bad_frame['valid'] = np.zeros_like(frame['valid'])
        if invalid == 'nonplanar':
            bad_frame['points'] = points + np.random.default_rng(1).normal(0, 2., points.shape)
        try:
            fit_face(bad_frame, mask, changed, top, ground)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted unsupported source face: ' + invalid)
    print('PASS: source affine and perspective; textured open face; mesh and measurement share bottom; normalized ground; gaps and unsupported planes rejected; no metric promotion')


def multiview_check():
    from check_workcell_button_bundle import fixture
    from workcell_post_faces import fit_multiview_face, complete_face_observations, _raw_tolerance
    from workcell_button_bundle import _project_raw
    frames, *_ = fixture()
    frames = {photo: frame for photo, frame in frames.items() if photo <= 3}
    truth = np.array([[-.08, 0., .25], [.08, 0., .25], [.08, 0., 1.15], [-.08, 0., 1.15]])
    observations = {}
    diagnostics = {'bottomCandidates': [], 'topCandidates': [], 'views': []}
    for photo, frame in frames.items():
        raw = _project_raw(truth, frame)[0]
        sides = [raw[[0, 3]].tolist(), raw[[1, 2]].tolist()]
        # Distinct local terminal IDs are not enough to accept or reject. Both
        # termini must agree with actual long-side RGB coordinates.
        bottom = {'photo': photo, 'rawEnds': raw[:2].tolist(), 'rawSegments': [raw[:2].tolist()],
                  'faceSideIds': [11, 12], 'faceSideEdgesRaw': sides}
        top = {'photo': photo, 'rawEnds': raw[[3, 2]].tolist(), 'rawSegments': [raw[[3, 2]].tolist()],
               'faceSideIds': [21, 22], 'faceSideEdgesRaw': sides}
        groups = [{'id': 31 + i, 'rawSegments': [side], 'visibleLengthRawPx': float(np.linalg.norm(np.diff(side, axis=0)))} for i, side in enumerate(sides)]
        diagnostics['bottomCandidates'].append(bottom); diagnostics['topCandidates'].append(top)
        diagnostics['views'].append({'photo': photo, 'bottom': {'sideSupportGroups': groups}, 'top': {}})
        observations[photo] = [{'photo': photo, 'observationId': f'face-{photo}', 'rawCorners': raw.tolist(),
                               'boundariesRaw': [[raw[[i, (i + 1) % 4]].tolist()] for i in range(4)]}]
    extracted, _ = complete_face_observations(diagnostics, frames)
    assert set(extracted) == set(frames), extracted
    xyz, gate, matches = fit_multiview_face(observations, frames, {'normal': [0, 0, 2.], 'offset': 0.}, max_starts=2)
    assert gate['accepted'] and gate['jacobianRank'] == gate['parameterCount'] == 6, gate
    assert gate['rankSource'] == 'source_reprojection_without_priors'
    assert all(row['jacobianRank'] == 6 and row['maxRawPx'] < 1e-5 for row in gate['leaveOnePhotoOut'])
    assert np.max(np.min(np.linalg.norm(xyz[:, None] - truth[None], axis=2), axis=1)) < 1e-7
    assert abs(np.min(xyz[:, 2]) - .25) < 1e-7
    assert all(row['thresholdRawPx'] == _raw_tolerance(frames[row['photo']]) for row in gate['reprojectionByPhoto'])
    # Contradictory complete boundaries cannot earn acceptance merely by moving
    # an old box bottom or selecting a different rectangle edge in one view.
    wrong = copy.deepcopy(observations)
    wrong[3][0]['boundariesRaw'][0] = (np.asarray(wrong[3][0]['boundariesRaw'][0]) + [0., 45.]).tolist()
    _, rejected, _ = fit_multiview_face(wrong, frames, {'normal': [0, 0, 1.], 'offset': 0.}, max_starts=2)
    assert not rejected['accepted'], rejected
    # Duplicated source cameras do not become stereo evidence.
    same_frames = {p: frames[1] for p in frames}
    same = {p: copy.deepcopy(observations[1]) for p in frames}
    try:
        fit_multiview_face(same, same_frames, {'normal': [0, 0, 1.], 'offset': 0.}, max_starts=2)
    except ValueError as error:
        assert 'triangulate' in str(error)
    else:
        raise AssertionError('Identical source cameras produced an accepted observable depth')
    bad_sides = copy.deepcopy(diagnostics)
    for row in bad_sides['views']:
        for group in row['bottom']['sideSupportGroups']:
            group['rawSegments'] = (np.asarray(group['rawSegments']) + [50., 0.]).tolist()
    unmatched, _ = complete_face_observations(bad_sides, frames)
    assert not any(row.get('rawCorners') is not None for rows in unmatched.values() for row in rows), 'Local side IDs established a complete face without full-length RGB side agreement'
    partial = copy.deepcopy(observations)
    partial[1][0].update(rawCorners=None)
    partial[1][0]['boundariesRaw'][2] = []
    partial[3][0].update(rawCorners=None)
    partial[3][0]['boundariesRaw'][0] = []
    initial = [{'photo': 2, 'observationId': 'face-2', 'cornersNative': (truth + [.05, .12, -.03]).tolist()}]
    partial_xyz, partial_gate, _ = fit_multiview_face(partial, frames, {'normal': [0, 0, 1.], 'offset': 0.}, initial_corners=initial, max_starts=2)
    assert partial_gate['accepted'] and partial_gate['jacobianRank'] == 6, partial_gate
    assert np.max(np.min(np.linalg.norm(partial_xyz[:, None] - truth[None], axis=2), axis=1)) < 1e-5
    assert partial[1][0]['boundariesRaw'][2] == [] and partial[3][0]['boundariesRaw'][0] == []
    # Two real faces may both explain the photos. Preserve them and do not call
    # an arbitrarily selected wing terminal the unique whole-housing minimum.
    ambiguous = copy.deepcopy(observations)
    wing = truth + [0., .04, .2]
    for photo, frame in frames.items():
        raw = _project_raw(wing, frame)[0]
        ambiguous[photo].append({'photo': photo, 'observationId': f'wing-{photo}', 'rawCorners': raw.tolist(),
                                 'boundariesRaw': [[raw[[i, (i + 1) % 4]].tolist()] for i in range(4)]})
    _, ambiguous_gate, _ = fit_multiview_face(ambiguous, frames, {'normal': [0, 0, 1.], 'offset': 0.}, max_starts=6)
    assert ambiguous_gate['terminalPartAmbiguity']['resolved'] is False
    heights = [row['bottomHeightNative'] for row in ambiguous_gate['surfaceAlternatives']]
    assert heights and np.ptp(heights) > .15, heights
    json.dumps(gate, allow_nan=False)
    print('PASS: free full-face stereo geometry and physical terminal; observed full sides; data-only rank; fixed same-boundary holdouts; contradictory and zero-baseline inputs rejected')
    print('PASS: one complete anchor plus complementary partial views recovers geometry; missing boundaries remain absent; distinct supported terminal parts retained')


def orientation_check():
    from scipy.spatial.transform import Rotation
    from check_workcell_button_bundle import fixture
    from workcell_post_faces import fit_multiview_face, _remove_dominated_partials
    from workcell_button_bundle import _project_raw
    frames, *_ = fixture()
    frames = {p: frame for p, frame in frames.items() if p <= 3}
    base = np.array([0., 0., .3])
    rectangle = np.array([[-.08, 0., 0.], [.08, 0., 0.], [.08, 0., .9], [-.08, 0., .9]])
    truth = rectangle @ Rotation.from_euler('x', 12, degrees=True).as_matrix().T + base
    observations = {}
    for p, frame in frames.items():
        raw = _project_raw(truth, frame)[0]
        observations[p] = [{'photo': p, 'observationId': f'tilted-{p}', 'rawCorners': raw.tolist(),
                            'boundariesRaw': [[raw[[i, (i + 1) % 4]].tolist()] for i in range(4)]}]
    ground = {'normal': [0, 0, 1.], 'offset': 0.}
    _, rigid_gate, rigid_matches = fit_multiview_face(observations, frames, ground, max_starts=2)
    free_xyz, free_gate, free_matches = fit_multiview_face(observations, frames, ground, max_starts=2,
                                                        orientation='free', fixed_matches=rigid_matches)
    assert not rigid_gate['accepted'] and free_gate['accepted'], (rigid_gate, free_gate)
    assert free_gate['parameterCount'] == free_gate['jacobianRank'] == 8
    assert abs(free_gate['angleFromGroundNormalDeg'] - 12) < 1e-5
    assert np.max(np.min(np.linalg.norm(free_xyz[:, None] - truth[None], axis=2), axis=1)) < 1e-7
    assert [(p, r['observationId'], f) for p, r, f in rigid_matches] == [(p, r['observationId'], f) for p, r, f in free_matches]
    changed_ground = {'normal': [0, .2, 1.], 'offset': .1}
    changed_xyz, _, _ = fit_multiview_face(observations, frames, changed_ground, max_starts=2,
                                          orientation='free', fixed_matches=rigid_matches)
    assert np.allclose(free_xyz, changed_xyz, atol=1e-7), 'Free RGB geometry moved when only the measuring ground changed'
    # A real but inconsistent bottom cannot be omitted as if it were occluded.
    conflicting = copy.deepcopy(observations)
    conflicting[3][0]['boundariesRaw'][0] = (np.asarray(conflicting[3][0]['boundariesRaw'][0]) + [0., 45.]).tolist()
    partial = copy.deepcopy(conflicting[3][0]); partial.update(rawCorners=None, observationId='top-only')
    partial['boundariesRaw'][0] = []
    conflicting[3].append(partial)
    assert len(_remove_dominated_partials(conflicting[3])) == 1
    _, rejected, matches = fit_multiview_face(conflicting, frames, ground, max_starts=2, orientation='free')
    assert not rejected['accepted']
    assert all(row['observationId'] != 'top-only' for _, row, _ in matches)
    # The same observed boundary may be split differently by LSD. Adjacent
    # finite fragments cover it; a genuine gap between them does not.
    complete = {'rawCorners': [[0, 0], [1, 0], [1, 10], [0, 10]],
                'boundariesRaw': [[[[0, 0], [1, 0]]], [[[1, 0], [1, 5]], [[1, 5], [1, 10]]],
                                  [[[1, 10], [0, 10]]], [[[0, 10], [0, 0]]]]}
    partial = copy.deepcopy(complete); partial['rawCorners'] = None
    partial['boundariesRaw'][0] = []
    partial['boundariesRaw'][1] = [[[1, 10], [1, 0]]]
    assert len(_remove_dominated_partials([complete, partial])) == 1
    complete['boundariesRaw'][1] = [[[1, 0], [1, 4]], [[1, 6], [1, 10]]]
    assert len(_remove_dominated_partials([complete, partial])) == 2
    print('PASS: same-source 6-vs-8 orientation control; tilted face recovered; free RGB geometry independent of measuring ground; known terminal cannot be omitted by a partial subset')
    print('PASS: dominated partial uses the finite observed segment union; adjacent fragments cover; true gaps remain unsupported')


def batch_check():
    from check_workcell_button_bundle import fixture
    from workcell_button_bundle import _project_raw
    import workcell_post_faces as faces
    frames, *_ = fixture(); frames = {p: frame for p, frame in frames.items() if p <= 3}
    truth = np.array([[-.08, 0., .25], [.08, 0., .25], [.08, 0., 1.15], [-.08, 0., 1.15]])
    observations = {}
    for photo, frame in frames.items():
        raw = _project_raw(truth, frame)[0]
        boundaries = [[raw[[i, (i + 1) % 4]].tolist()] for i in range(4)]
        if photo != 2:
            boundaries[2 if photo == 1 else 0] = []
        if photo == 3:
            boundaries = [boundaries[i] for i in (0, 3, 2, 1)]
        row = {'photo': photo, 'observationId': f'correct-{photo}', 'rawCorners': raw.tolist() if photo == 2 else None,
               'boundariesRaw': boundaries}
        wrong = copy.deepcopy(row); wrong['observationId'] = f'wrong-{photo}'
        wrong['boundariesRaw'] = [(np.asarray(b) + [35., 8.]).tolist() if b else [] for b in boundaries]
        if wrong['rawCorners'] is not None:
            wrong['rawCorners'] = (raw + [35., 8.]).tolist()
        observations[photo] = [wrong, row]
    initial = [{'photo': 2, 'observationId': 'correct-2', 'cornersNative': truth.tolist()}]
    ground = {'normal': [0, 0, 1.], 'offset': 0.}
    optimize = faces.least_squares
    probes = []
    def audited_optimizer(use_reference):
        def solve(function, x0, *args, **kwargs):
            call = inspect.getclosurevars(function).nonlocals
            evaluate = call['evaluate']; env = inspect.getclosurevars(evaluate).nonlocals
            selected_photos = call.get('training', inspect.signature(evaluate).parameters['chosen_photos'].default)
            fixed = call.get('anchor', call.get('fixed'))
            def reference(parameters):
                xyz = env['corners'](parameters); residual, matches = [], []
                for photo in selected_photos:
                    locked = env['locked'].get(photo, fixed.get(photo) if fixed is not None else None)
                    options = [locked] if locked is not None else [(row, flip) for row in observations[photo] for flip in (False, True)]
                    trials = []
                    for row, flip in options:
                        try:
                            vector = np.concatenate(faces._face_errors(xyz, row, frames[photo], flip)).ravel()
                            trials.append((float(vector @ vector) / sum(bool(b) for b in row['boundariesRaw']), vector, row, flip))
                        except ValueError:
                            pass
                    if not trials:
                        return np.full(len(selected_photos) * 72, 1e6), []
                    _, vector, row, flip = min(trials, key=lambda trial: trial[0])
                    residual.extend(vector); matches.append((photo, row['observationId'], flip))
                return np.asarray(residual), matches
            for parameters in (x0, x0 + np.arange(len(x0)) * 1e-5):
                vector, matches = evaluate(parameters, selected_photos, fixed)
                expected, identities = reference(parameters)
                assert np.array_equal(vector, expected), np.max(abs(vector - expected))
                assert [(p, row['observationId'], flip) for p, row, flip in matches] == identities
                probes.append(identities)
            return optimize((lambda p: reference(p)[0]) if use_reference else function, x0, *args, **kwargs)
        return solve
    results = []
    for use_reference in (False, True):
        with patch.object(faces, 'least_squares', side_effect=audited_optimizer(use_reference)):
            upright = faces.fit_multiview_face(observations, frames, ground, initial_corners=initial, max_starts=2)
            free = faces.fit_multiview_face(observations, frames, ground, initial_corners=initial, max_starts=2,
                                           orientation='free', fixed_matches=upright[2])
            results.append((upright, free))
    for batched, reference in zip(*results):
        assert np.array_equal(batched[0], reference[0]) and batched[1] == reference[1]
    assert any(flip for identities in probes for _, _, flip in identities)
    assert results[0][0][1]['accepted'] and results[0][1][1]['accepted']
    print('PASS: batched versus original per-candidate residuals, selection, optimized vertices and gates exactly match; multiple candidates, partial edges, flipped sides, fixed assignments and anchors exercised')


def side_identity_check():
    from workcell_post_faces import complete_face_observations, _line
    frame = {'A': np.diag([1 / 8., 1 / 8., 1.]), 'rawShape': [1200, 1200]}
    groups = [{'id': i, 'rawSegments': [[[x, 300], [x, 480]], [[x, 520], [x, 700]]]} for i, x in enumerate((100, 140))]
    bottom = {'photo': 1, 'rawEnds': [[100, 900], [140, 900]], 'rawSegments': [[[100, 900], [140, 900]]],
              'faceSideIds': [10, 11], 'faceSideEdgesRaw': [[[100, 850], [100, 900]], [[140, 850], [140, 900]]]}
    top = {'photo': 1, 'rawEnds': [[110, 100], [150, 100]], 'rawSegments': [[[110, 100], [150, 100]]],
           'faceSideIds': [20, 21], 'faceSideEdgesRaw': [[[110, 100], [110, 150]], [[150, 100], [150, 150]]]}
    diagnostic = {'bottomCandidates': [bottom], 'topCandidates': [top],
                  'views': [{'photo': 1, 'bottom': {'sideSupportGroups': groups}}]}
    # The old joint fit admits a 10px different local edge by rotating a long
    # reference group whose actual segments lie between the two termini.
    for i, group in enumerate(groups):
        local = np.asarray([bottom['faceSideEdgesRaw'][i], top['faceSideEdgesRaw'][i]]).reshape(-1, 2)
        points = np.asarray(group['rawSegments']).reshape(-1, 2)
        assert np.max(abs(np.c_[local, np.ones(len(local))] @ _line(points, 8.))) > 8.
        _line(np.r_[local, points], 8.)
    observed, _ = complete_face_observations(diagnostic, {1: frame})
    assert not any(row['complete'] for row in observed[1])
    top_partial = next(row for row in observed[1] if row['top'] is not None)
    assert all(group.get('supportScope') == 'local observed side only' for group in top_partial['sideGroups'])
    valid = copy.deepcopy(diagnostic)
    for key in ('rawEnds', 'rawSegments', 'faceSideEdgesRaw'):
        valid['topCandidates'][0][key] = (np.asarray(valid['topCandidates'][0][key]) - [10, 0]).tolist()
    observed, _ = complete_face_observations(valid, {1: frame})
    assert any(row['complete'] for row in observed[1]), 'Valid fragmented long sides lost their complete face'
    print('PASS: fixed long-side identity rejects circular terminal refitting in complete and partial branches; valid multiple source fragments remain supported')


def volume_check():
    from scipy.spatial.transform import Rotation
    from workcell_post_faces import _extruded_face, _volume_mask_metrics
    from workcell_guard_silhouette import _mesh_mask
    from check_workcell_button_bundle import fixture
    frames, *_ = fixture()
    corners = np.array([[-.08, 0., .3], [.08, 0., .3], [.08, 0., 1.2], [-.08, 0., 1.2]])
    corners = corners @ Rotation.from_euler('x', 12, degrees=True).as_matrix().T
    observed_height = corners[:2, 2].min(); minimums = []
    with tempfile.TemporaryDirectory() as directory:
        for sign in (-1, 1):
            mesh = _extruded_face(corners, .07, sign)
            assert mesh.vertices.shape == (8, 3) and mesh.faces.shape == (12, 3)
            assert mesh.is_watertight and mesh.volume > 0 and np.array_equal(mesh.vertices[:4], corners)
            path = Path(directory) / f'volume-{sign}.glb'
            scene = trimesh.Scene(); scene.add_geometry(mesh, node_name='housing', geom_name='housing'); scene.export(path)
            saved = trimesh.load(path, force='scene', process=False)
            transform, name = saved.graph['housing']; restored = saved.geometry[name].copy(); restored.apply_transform(transform)
            assert restored.vertices.shape == (8, 3) and restored.faces.shape == (12, 3) and restored.is_watertight
            assert np.allclose(restored.vertices, mesh.vertices, atol=1e-7, rtol=0)
            assert abs(restored.vertices[:2, 2].min() - observed_height) < 1e-7
            assert abs(restored.vertices[:, 2].min() - mesh.vertices[:, 2].min()) < 1e-7
            minimums.append(restored.vertices[:, 2].min())
            frame = frames[1]; shape = frame.get('shape', (512, 512))
            mask = _mesh_mask(mesh, frame, shape)
            metrics = _volume_mask_metrics(mesh, [{'photo': 1, 'frame': frame, 'mask': mask}], boundaries=True)
            assert mask.any() and metrics['meanIoU'] == 1.
            assert metrics['views'][0]['symmetricBoundaryMeanPx'] == 0.
    assert min(minimums) < observed_height - .01, 'Hypothetical rear minimum must remain distinct from the unchanged observed lower edge'
    print('PASS: closed 8-vertex/12-face extrusion exports both signs; GLB preserves visible vertex identities and native minimum; hypothetical rear bottom is distinct; full-image mask metric reuses renderer')


if __name__ == '__main__':
    check()
    multiview_check()
    orientation_check()
    batch_check()
    side_identity_check()
    volume_check()
