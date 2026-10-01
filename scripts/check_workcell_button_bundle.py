"""Small synthetic check: PYTHONPATH=.:scripts python scripts/check_workcell_button_bundle.py."""
import copy
import json

import numpy as np


def legacy_missing_coupling():
    # A measured change in either diameter leaves the old height-only scale
    # unchanged. Three independently computed scales cannot enforce one object.
    native = np.array([.25, .2125, .1])
    supplied = np.array([.1, .085, .04])
    for index in (1, 2):
        changed = supplied.copy()
        changed[index] *= 1.15
        separate = changed / native
        assert separate[0] == .4 and np.ptp(separate) > .05


def fixture():
    from workcell_button_bundle import _project_raw
    dims = {'wholeComponentHeightM': .1, 'mainBodyDiameterM': .085, 'redActuatorDiameterM': .04}
    reference = {'features': dims, 'scopeStatus': 'confirmed'}
    axis = np.array([0., 0., 1.])
    yaw = .31
    u, v = np.array([np.cos(yaw), np.sin(yaw), 0.]), np.array([-np.sin(yaw), np.cos(yaw), 0.])
    shape = {'base': np.array([0., 0., .48]), 'axis': axis, 'u': u, 'v': v, 'height': .2,
             'grayHeight': .07, 'yellowHeight': .08, 'redHeight': .05, 'yellowRadius': .085,
             'redRadius': .04, 'grayWidth': .12, 'grayDepth': .09, 'mPerNative': .5, 'yaw': yaw,
             'yellowTopRadiusFraction': .7}
    frames = {}
    for photo, center in enumerate(([0., -2.5, 1.2], [1.7, -2.1, 1.55], [-1.3, -2.3, .9], [2., -.7, 1.1]), 1):
        center = np.asarray(center)
        forward = np.array([0., 0., .6]) - center; forward /= np.linalg.norm(forward)
        right = np.cross(forward, axis); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        pose = np.eye(4); pose[:3, :3] = np.c_[right, down, forward]; pose[:3, 3] = center
        A = np.array([[.225, 0., -.3875], [0., .333, -12.8335], [0., 0., 1.]])
        Kraw = np.array([[1800., 0., 1000.], [0., 1800., 750.], [0., 0., 1.]])
        frames[photo] = {'K': A @ Kraw, 'pose': pose, 'A': A, 'rawShape': [1500, 2000]}
    angle = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    circle = np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v
    observations = []
    for photo, frame in frames.items():
        row = {'photo': photo, 'grayHousingEdgeContrast': 40.}
        for color, radius, heights in (('red', .04, (.15, .2)), ('yellow', .085, (.07, .15))):
            radii = (radius, radius * shape['yellowTopRadiusFraction']) if color == 'yellow' else (radius, radius)
            points = np.concatenate([shape['base'] + h * axis + r * circle for h, r in zip(heights, radii)])
            row[color + 'HullRaw'] = _project_raw(points, frame)[0].tolist()
        bottom = shape['base'] + .06 * np.cos(angle)[:, None] * u + .045 * np.sin(angle)[:, None] * v
        pixels = _project_raw(bottom, frame)[0]
        row['housingBottomSupportRaw'] = float(pixels[:, 1].max())
        all_pixels = np.concatenate([row['redHullRaw'], row['yellowHullRaw'], pixels])
        row['boxRaw'] = [*all_pixels.min(0), *all_pixels.max(0)]
        observations.append(row)
    rng = np.random.default_rng(402)
    tracks = []
    for xyz in rng.uniform([-.8, -.55, .15], [.8, .6, 1.65], (36, 3)):
        obs = []
        overlaps = False
        for photo, frame in frames.items():
            uv, _ = _project_raw([xyz], frame)
            box = np.asarray(observations[photo - 1]['boxRaw'])
            overlaps |= bool(np.all(uv[0] >= box[:2] - 4) and np.all(uv[0] <= box[2:] + 4))
            canonical = frame['A'] @ np.r_[uv[0], 1.]
            obs.append({'photo': photo, 'uv': canonical[:2].tolist()})
        if not overlaps:
            tracks.append({'id': len(tracks), 'xyz': xyz.tolist(), 'observations': obs})
    initial = {'baseNative': shape['base'] + [.002, -.001, .001], 'axisNative': axis,
               'mPerNative': .49, 'sectionFractions': [.35, .4, .25], 'grayWidthM': .061,
               'grayDepthM': .044, 'yaw': .30}
    return frames, tracks, observations, reference, initial, shape


def observability_and_seed():
    import workcell_button_bundle as bundle
    assert hasattr(bundle, '_observability'), 'Need nuisance-marginalized camera/scale observability'
    # Unknown gray geometry alone must not invalidate fully observed cameras/scale.
    J = np.diag([2., 3., 0.])
    obs = bundle._observability(J, [0, 1], 1)
    assert obs['cameraAndScaleIdentifiable'] and not obs['allParametersIdentifiable']
    assert abs(obs['localLogScaleSigmaAtTwoRawPixelNoise'] - 1 / 3) < 1e-12
    # A camera-scale ambiguity must remain rejected even with perfect residuals.
    obs = bundle._observability(np.array([[1., 1., 0.], [0., 0., 1.]]), [0, 1], 1)
    assert not obs['cameraAndScaleIdentifiable'] and obs['localLogScaleSigmaAtTwoRawPixelNoise'] is None
    frames, tracks, rows, reference, initial, shape = fixture()
    a = bundle._button_seed(frames, rows[1:], reference, initial['axisNative'])
    changed = copy.deepcopy(rows)
    changed[0]['redHullRaw'] = (np.asarray(changed[0]['redHullRaw']) + 300).tolist()
    b = bundle._button_seed(frames, changed[1:], reference, initial['axisNative'])
    assert np.array_equal(a['baseNative'], b['baseNative']) and a['mPerNative'] == b['mPerNative']
    assert np.linalg.norm(a['baseNative'] - shape['base']) < .05
    assert abs(a['mPerNative'] - shape['mPerNative']) < .1



def occluded_support_directions():
    import workcell_button_bundle as bundle
    assert hasattr(bundle, '_support_validity'), 'Removed contour extrema still become exact inward-chord constraints'
    frames, tracks, rows, reference, initial, shape = fixture()
    row = rows[0]
    original = np.asarray(row['yellowHullRaw'])
    removed = original[:, 0] > original[:, 0].max() - 12.
    row['yellowHullRaw'] = original[~removed].tolist()
    row['yellowBoundary'] = {'graphCutHullRaw': original.tolist(),
                            'unsupportedBoundaryVerticesRaw': [{'pixel': p.tolist()} for p in original[removed]]}
    valid = bundle._support_validity(row)
    assert not valid[16] and valid[16 + 4] and valid[16 + 8]
    result, _, _ = bundle.solve(frames, tracks, rows, reference, initial, max_nfev=120, workers=2)
    assert result['status'] == 'available', result['diagnostics']
    assert abs(result['mPerNative'] - .5) < .001
    diagnostics = result['diagnostics']['featureErrorsByPhoto'][1]
    assert 16 in diagnostics['excludedSupportIndices']
    assert diagnostics['maxExcludedErrorRawPx'] > 10.
    assert diagnostics['maxRawPx'] < .01
    heldout = next(r for r in result['heldOutPhotos'] if r['photo'] == 1)
    assert heldout['status'] == 'available' and 16 in heldout['featureErrors']['excludedSupportIndices']
    sparse = copy.deepcopy(row)
    sparse['yellowHullRaw'] = original[original[:, 0] < original[:, 0].min() + 1.].tolist()
    sparse['yellowBoundary']['unsupportedBoundaryVerticesRaw'] = [{'pixel': p.tolist()} for p in original]
    try:
        bundle._support_validity(sparse)
    except ValueError as error:
        assert 'Insufficient supported' in str(error)
    else:
        raise AssertionError('Unobserved contour directions were accepted')


def main():
    occluded_support_directions()
    observability_and_seed()
    legacy_missing_coupling()
    from workcell_button_bundle import _project_raw, _reference, _supports, solve
    frames, tracks, observations, reference, initial, shape = fixture()
    from workcell_button_bundle import _observed_support
    assert max(abs(_supports(frames[r['photo']], shape)[0] - _observed_support(r)).max() for r in observations) < .001
    result, cameras, fitted_tracks = solve(frames, tracks, observations, reference, initial, max_nfev=120, workers=2)
    print(json.dumps({'status': result['status'], 'diagnostics': result['diagnostics'], 'heldout': [{k: r[k] for k in ('photo', 'status', 'maxErrorRawPx')} for r in result['heldOutPhotos']]}, indent=2))
    assert result['status'] == 'available', result['reason']
    assert abs(result['mPerNative'] - .5) < .01
    for name, dimension in reference['features'].items():
        assert abs(result['exactMetricDimensions'][name] - dimension) < 1e-12
    assert np.allclose(cameras['frames'][0]['pose'], frames[1]['pose'], atol=1e-12)
    gauge = result['gauge']
    p = next(r for r in cameras['frames'] if r['photo'] == gauge['fixedCenterPhoto'])
    assert p['pose'][gauge['fixedCenterAxis']][3] == gauge['fixedCenterNative']
    assert result['diagnostics']['sceneReprojectionP95RawPx'] < .05
    for row in result['heldOutPhotos']:
        assert row['photo'] not in row['fitButtonPhotos'] and row['withheldButtonResidualCount'] == 0
        assert row['cameraRetainedSceneTracks'] >= 12
        assert row['diagnostics']['seedButtonPhotos'] == row['fitButtonPhotos']
        from workcell_button_bundle import _button_seed
        trained = _button_seed(frames, [o for o in observations if o['photo'] != row['photo']], reference, initial['axisNative'])
        assert row['diagnostics']['seedMPerNative'] == trained['mPerNative']
        assert np.array_equal(row['diagnostics']['seedBaseNative'], trained['baseNative'])
    assert _reference({'reference': reference, 'evaluation': {'targets': [{'groundTruthM': 999.}]}}) == _reference(reference)
    for name in reference['features']:
        changed = copy.deepcopy(reference)
        changed['features'][name] *= 1.15
        perturbed, altered, _ = solve(frames, tracks, observations, changed, initial, max_nfev=100, holdouts=False)
        assert perturbed['diagnostics']['initialCost'] > result['diagnostics']['initialCost'] + 1.
        assert abs(perturbed['exactMetricDimensions'][name] - changed['features'][name]) < 1e-12
        if name != 'wholeComponentHeightM':
            camera_change = max(np.linalg.norm(np.asarray(a['K']) - b['K']) + np.linalg.norm(np.asarray(a['pose']) - b['pose']) for a, b in zip(altered['frames'], cameras['frames']))
            assert camera_change > .001, (name, camera_change)
    missing = observations[:2]
    try:
        solve(frames, tracks, missing, reference, initial, holdouts=False)
    except ValueError as error:
        assert 'At least three distinct complete button observations' in str(error)
    else:
        raise AssertionError('Missing endpoint view was accepted')
    occluded = copy.deepcopy(observations); occluded[0]['occluded'] = True
    try:
        solve(frames, tracks, occluded, reference, initial, holdouts=False)
    except ValueError as error:
        assert 'occluded' in str(error)
    else:
        raise AssertionError('Explicit occlusion was accepted')
    no_scene = tracks[:5]
    try:
        solve(frames, no_scene, observations, reference, initial, holdouts=False)
    except ValueError as error:
        assert 'Insufficient non-button' in str(error)
    else:
        raise AssertionError('Unsupported scene cameras were accepted')
    bad = copy.deepcopy(observations)
    bad[0]['redHullRaw'] = (np.asarray(bad[0]['redHullRaw']) + [30., 0.]).tolist()
    bad[0]['boxRaw'][2] += 30.
    rejected, _, _ = solve(frames, tracks, bad, reference, initial, max_nfev=60, holdouts=False)
    assert rejected['status'] == 'unsupported' and rejected['mPerNative'] is None
    assert not rejected['diagnostics']['checks']['fittedButtonSupported']
    print('PASS: exact joint dimensions, both diameters alter cameras, raw/canonical pixels, gauge, whole-button holdouts, no evaluation leakage, unsupported evidence')


if __name__ == '__main__':
    main()
