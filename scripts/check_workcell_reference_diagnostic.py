"""Run with PYTHONPATH=.:scripts python scripts/check_workcell_reference_diagnostic.py."""
import copy

import numpy as np

from check_workcell_button_bundle import fixture
from workcell_reference_diagnostic import _shape_supports, fit_photo, scene_track_residuals
from workcell_button_bundle import _supports


def main():
    frames, tracks, observations, reference, _, shape = fixture()
    before = copy.deepcopy((frames, observations))
    seed = copy.deepcopy(shape)
    seed['base'] = seed['base'] + [.001, -.001, .001]
    result = fit_photo(frames[2], observations[1], reference, seed)
    assert result['shapeCanExplainThisPhoto'], result
    assert result['maxRawPx'] < .001, result
    fitted = result['shapeInIndependentCameraMetres']
    assert abs(fitted['height'] - .1) < 1e-12
    assert abs(2 * fitted['yellowRadius'] - .085) < 1e-12
    assert abs(2 * fitted['redRadius'] - .04) < 1e-12
    for photo in frames:
        for key in ('K', 'A', 'pose'):
            assert np.array_equal(frames[photo][key], before[0][photo][key])
    assert observations == before[1]
    changed_truth = {**reference, 'evaluation': {'fenceBottomM': 999}}
    repeated = fit_photo(frames[2], observations[1], changed_truth, seed)
    assert repeated['maxRawPx'] == result['maxRawPx']
    assert repeated['shapeInIndependentCameraMetres'] == fitted
    exact = scene_track_residuals(frames, tracks)
    assert exact['all']['maxRawPx'] < 1e-8, exact
    wrong = copy.deepcopy(frames)
    wrong[2]['pose'][1, 3] += .02
    assert scene_track_residuals(wrong, tracks)['all']['p95RawPx'] > 1.
    original = _supports(frames[2], shape)[0]
    shoulder = {**shape, 'yellowShoulderHeightFraction': .75, 'yellowShoulderRadiusFraction': .95}
    expanded = _shape_supports(frames[2], shoulder)[0]
    assert np.array_equal(original[:16], expanded[:16])
    assert original[32] == expanded[32]
    assert np.all(expanded[16:32] >= original[16:32])
    assert max(expanded[16:32] - original[16:32]) > 1.
    trial = fit_photo(frames[2], observations[1], reference, seed, shoulder_height=.75)
    assert trial['shapeInIndependentCameraMetres']['yellowRadius'] == .0425
    fitted = trial['shapeInIndependentCameraMetres']
    assert fitted['yellowTopRadiusFraction'] <= fitted['yellowShoulderRadiusFraction'] <= 1.
    print('PASS: synthetic shape, fixed cameras/contours, three dimensions, truth exclusion, independent epipolar check')


if __name__ == '__main__':
    main()
