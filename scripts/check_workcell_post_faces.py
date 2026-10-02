"""Run: PYTHONPATH=.:scripts python scripts/check_workcell_post_faces.py."""
import copy
import importlib.util
import json

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


if __name__ == '__main__':
    check()
