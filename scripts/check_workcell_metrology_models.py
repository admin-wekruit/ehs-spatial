"""Tiny GLB contract check: PYTHONPATH=.:scripts python scripts/check_workcell_metrology_models.py."""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import trimesh


def main():
    assert importlib.util.find_spec('workcell_metrology_models'), 'Reference candidate exporter is missing'
    from workcell_metrology_models import export_reference_candidate
    axis = np.array([0., .6, .8]); u = np.array([1., 0., 0.]); v = np.cross(axis, u)
    base = np.array([2., -3., .7])
    shape = {'base': base.tolist(), 'axis': axis.tolist(), 'u': u.tolist(), 'v': v.tolist(),
             'height': .2, 'grayHeight': .07, 'yellowHeight': .08, 'redHeight': .05,
             'yellowRadius': .085, 'redRadius': .04, 'grayWidth': .12, 'grayDepth': .09,
             'yellowTopRadiusFraction': .7, 'mPerNative': .5}
    dims = {'wholeComponentHeightM': .10, 'mainBodyDiameterM': .085, 'redActuatorDiameterM': .04}
    source = {'status': 'unsupported', 'mPerNative': None, 'candidateMPerNative': .5,
              'knownDimensions': dims, 'fittedNuisanceParameters': shape, 'reason': 'held-out contour failed'}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        out = root / 'unsupported'
        manifest = export_reference_candidate(source, out)
        assert manifest['metricScaleMPerNative'] is None and manifest['candidateScaleForPreviewOnly'] == .5
        assert manifest['previewOnly'] and manifest['measurementStatus'] == 'unsupported'
        assert json.loads((out / 'reference-candidate.json').read_text()) == manifest
        scene = trimesh.load(out / manifest['modelFile'], force='scene')
        assert set(scene.geometry) == {'red-actuator', 'yellow-body', 'gray-housing'}
        basis = np.c_[u, v, axis]
        local = {name: (mesh.vertices - base) @ basis for name, mesh in scene.geometry.items()}
        all_vertices = np.concatenate(list(local.values()))
        assert np.isclose(all_vertices[:, 2].min(), 0., atol=3e-7)
        assert np.isclose(np.ptp(all_vertices[:, 2]) * .5, .10, atol=3e-7)
        for name, diameter in [('red-actuator', .04), ('yellow-body', .085)]:
            assert np.isclose(np.max(np.linalg.norm(local[name][:, :2], axis=1)) * 2 * .5, diameter, atol=3e-7)
        gray = local['gray-housing']
        assert np.max((gray[:, 0] / .06) ** 2 + (gray[:, 1] / .045) ** 2) < 1.00002
        assert all(mesh.is_watertight for mesh in scene.geometry.values())
        repeat = export_reference_candidate(source, root / 'repeat')
        assert repeat['modelSha256'] == manifest['modelSha256']
        good = copy.deepcopy(source); good.update(status='available', mPerNative=.5)
        accepted = export_reference_candidate(good, root / 'accepted')
        assert accepted['metricScaleMPerNative'] == .5 and not accepted['previewOnly']
        assert accepted['measurementStatus'] == 'conditional_fit'
        no_shape = export_reference_candidate({'status': 'unsupported', 'reason': 'missing contours'}, root / 'absent')
        assert no_shape['modelFile'] is None and no_shape['metricScaleMPerNative'] is None
        bad = copy.deepcopy(source); bad['fittedNuisanceParameters']['base'][0] = float('nan')
        try:
            export_reference_candidate(bad, root / 'bad')
        except ValueError:
            assert not (root / 'bad').exists()
        else:
            raise AssertionError('Nonfinite candidate was exported')
        bad_scale = copy.deepcopy(source); bad_scale['candidateMPerNative'] = .6
        try:
            export_reference_candidate(bad_scale, root / 'bad-scale')
        except ValueError:
            pass
        else:
            raise AssertionError('Conflicting metric scale was accepted')
    print('PASS: native-world three-part GLB, exact input dimensions, elliptic gray housing, conditional scale, missing geometry and invalid-input rejection')


if __name__ == '__main__':
    main()
