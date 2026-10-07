"""Run: PYTHONPATH=.:scripts python scripts/check_workcell_photo_calibration.py."""
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.workcell_photo_calibration import accepted_scale, apply_measurements, load_measurements, measurement_evaluation, validate_measurements
from scripts.workcell_photo_objects import button_meshes
from scripts.workcell_photo_report import _ground_distance

measurements = load_measurements(Path(__file__).resolve().parents[1]/'docs/workcell-photo/measurements-2026-10-01.json')


def packed(value):
    a = np.asarray(value, dtype=float)
    return {'data': base64.b64encode(a.tobytes()).decode(), 'shape': list(a.shape), 'dtype': str(a.dtype)}


def fixture(root, supported):
    shape = {'base': [0, .7, 3], 'axis': [0, 1, 0], 'u': [1, 0, 0], 'v': [0, 0, -1],
             'height': .2, 'grayHeight': .07, 'yellowHeight': .08, 'redHeight': .05,
             'yellowRadius': .085, 'redRadius': .04, 'grayWidth': .12, 'grayDepth': .06,
             'yellowTopRadiusFraction': .8, 'mPerNative': .5}
    geometry = {'anchor': {'nativeHeight': .3, 'nativeWidth': .24, 'centerNative': [0, .8, 3],
        'normal': [0, 0, 1], 'mPerNative': .5 if supported else None,
        'assumedHeightM': .1, 'assumedWidthM': .085,
        'views': [{'photo': p, 'boxRaw': [40, 55, 60, 70]} for p in (1, 2)],
        'referenceFit': {'status': 'available' if supported else 'unsupported', 'camerasFixed': True,
                         'knownDimensions': measurements['reference']['features'],
                         'mPerNative': .5 if supported else None, 'candidateMPerNative': .5,
                         'fittedNuisanceParameters': shape}},
        'floor': {'normal': [0, 1, 0], 'offset': 0}, 'physicalClearances': {'objects': [
            {'id': 'fence-0', 'status': 'conditional', 'heightNative': .4, 'sourcePhotos': [1, 2],
             'rangeNative': [.38, .42], 'pointNative': [0, .4, 3], 'footNative': [0, 0, 3]}]}}
    scene = trimesh.Scene()
    for node, mesh in button_meshes(geometry).items():
        scene.add_geometry(mesh, node_name=node, geom_name=node)
    scene.add_geometry(trimesh.creation.box(), node_name='unchanged', geom_name='unchanged')
    scene.export(root/'object-extras.glb')
    objects = [{'id': 'emergency-button', 'label': 'button', 'measurements': {}, 'observations': [],
                'model': {'file': 'object-extras.glb', 'nodes': list(button_meshes(geometry))}},
               {'id': 'unchanged', 'label': 'unchanged', 'model': {'file': 'object-extras.glb', 'nodes': ['unchanged']}}]
    for ident in ('fence-0', 'post-box-1', 'post-box-2'):
        # Contaminated minimum must never become a physical bottom estimate.
        objects.append({'id': ident, 'label': ident, 'observations': [{'photo': p, 'observedMeasurements': {
            'status': 'available', 'basis': {'corners_native': [[x,y,z] for x in (0,1) for y in (.001,1) for z in (2,3)]}}} for p in (1,2)],
            'measurements': {}, 'model': None})
    camera_inputs = [{'photo': p, 'K': [[50.,0.,50.],[0.,50.,50.],[0.,0.,1.]], 'pose': np.eye(4).tolist(), 'A': np.eye(3).tolist()} for p in range(1,5)]
    geometry['anchor']['referenceFit']['cameraProvenance'] = {'framesSha256': hashlib.sha256(json.dumps(camera_inputs, sort_keys=True).encode()).hexdigest()}
    for name, data in [('geometry.json', geometry), ('objects.json', {'objects': objects})]:
        (root/name).write_text(json.dumps(data))
    for p in range(1,5):
        frame = {'camera_poses': packed(np.eye(4)), 'intrinsics': packed([[50,0,50],[0,50,50],[0,0,1]]),
                 'input_mask_transform': {'input_to_canonical_pixel_centres': [[1.,0,0],[0,1.,0],[0,0,1]]}}
        with gzip.open(root/f'frame_{p:04}.json.gz', 'wt') as f: json.dump(frame, f)


def check(root, inputs, supported):
    before = trimesh.load(root/'object-extras.glb', force='scene').geometry['unchanged'].vertices.copy()
    geometry = apply_measurements(root, inputs)
    scale = geometry['anchor']['mPerNative']
    assert scale == (.5 if supported else None), 'Old envelope must not override accepted 3D scale'
    assert geometry['calibration']['diagnostics']['envelopeUsedForCalibration'] is False
    invalid = copy.deepcopy(geometry); invalid['anchor']['mPerNative'] = .8
    try: accepted_scale(invalid)
    except ValueError: pass
    else: raise AssertionError('Stale/mismatched scene scale accepted')
    assert np.array_equal(before, trimesh.load(root/'object-extras.glb', force='scene').geometry['unchanged'].vertices)
    transform = trimesh.geometry.align_vectors([0,1,0], [0,0,1])
    objects = json.loads((root/'objects.json').read_text())['objects']
    for item in objects:
        item['visibleHeightByPhoto'] = {}
        item['groundDistance'] = _ground_distance(item, geometry, transform)
    fence = next(o for o in objects if o['id'] == 'fence-0')['groundDistance']['feature']
    assert np.isclose(np.linalg.norm(np.subtract(fence['pointNative'], fence['footNative'])), .4)
    assert np.isclose(fence['footNative'][2], 0)
    evaluation = measurement_evaluation(objects, geometry, inputs)
    assert evaluation['comparisons'][0]['estimateM'] == (.2 if supported else None)
    assert all(r['estimateM'] is None for r in evaluation['comparisons'][1:]), 'No point minimum fallback'
    # Check values bind to the photo set they were supplied for; another capture never inherits them by (ordinal) object id.
    capture = {'sources': [{'photo': 1, 'sha256': 'a' * 64}, {'photo': 2, 'sha256': 'b' * 64}]}
    unbound = measurement_evaluation(objects, geometry, inputs, capture)
    assert unbound['comparisons'] == [] and [r['objectId'] for r in unbound['absentTargets']] == [t['objectId'] for t in inputs['evaluation']['targets']]
    bound = copy.deepcopy(inputs); bound['evaluation']['capture'] = {'photoSha256': ['b' * 64, 'a' * 64]}
    assert measurement_evaluation(objects, geometry, bound, capture)['comparisons'] == evaluation['comparisons']
    other = copy.deepcopy(bound); other['evaluation']['capture'] = {'photoSha256': ['a' * 64]}
    assert measurement_evaluation(objects, geometry, other, capture)['comparisons'] == []
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.glb')}
    return geometry, evaluation, hashes


for name in measurements['reference']['features']:
    for value in (None, True, 0, -1, float('nan'), float('inf')):
        bad = copy.deepcopy(measurements); bad['reference']['features'][name] = value
        try: validate_measurements(bad)
        except ValueError: pass
        else: raise AssertionError('Invalid input accepted')
for binding in ({}, {'photoSha256': []}, {'photoSha256': ['not-a-hash']}):
    bad = copy.deepcopy(measurements); bad['evaluation']['capture'] = binding
    try: validate_measurements(bad)
    except ValueError: pass
    else: raise AssertionError('Malformed capture binding accepted')
for supported in (True, False):
    with tempfile.TemporaryDirectory() as tmp:
        roots = [Path(tmp)/name for name in ('a','b')]
        for root in roots: root.mkdir(); fixture(root, supported)
        altered = copy.deepcopy(measurements)
        for i, target in enumerate(altered['evaluation']['targets']): target['groundTruthM'] = 10+i
        a, b = check(roots[0], measurements, supported), check(roots[1], altered, supported)
        assert a[0] == b[0] and a[2] == b[2], 'Evaluation truth changed calibration or models'
        assert [r['estimateM'] for r in a[1]['comparisons']] == [r['estimateM'] for r in b[1]['comparisons']]
print('PASS: accepted 3D scale survives metadata; physical endpoints share viewer floor; no bbox minima fallback; unsupported stays native; GT cannot change models or predictions')

# A late report failure must return completed model evidence, not lose paid compute.
from modal_apps.workcell_photo_all import _archive_result
import io
import tarfile
import time
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root/'finished.glb').write_bytes(b'completed model')
    (root/'source-1.jpg').write_bytes(b'raw input excluded')
    result = _archive_result(root, time.monotonic(), {'type': 'ValueError', 'message': 'test'})
    with tarfile.open(fileobj=io.BytesIO(result['archive']), mode='r:gz') as archive:
        assert set(archive.getnames()) == {'finished.glb', 'failure.json'}
        assert archive.extractfile('finished.glb').read() == b'completed model'
    assert result['error']['type'] == 'ValueError'
print('PASS: late failure retains completed model artifacts')
