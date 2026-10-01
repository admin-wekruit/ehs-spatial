"""Run: PYTHONPATH=. python scripts/check_workcell_photo_calibration.py [SAVED_REPORT_DIR]."""
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.workcell_photo_calibration import apply_measurements, load_measurements, measurement_evaluation, resolve_dimensions, validate_measurements
from scripts.workcell_photo_objects import button_basis, button_meshes
from scripts.workcell_photo_oneshot import _export_metric_scene
from scripts.workcell_photo_report import _ground_distance

REPO = Path(__file__).resolve().parents[1]
measurements = load_measurements(REPO/'docs/workcell-photo/measurements-2026-10-01.json')


def reject(data):
    try:
        validate_measurements(data)
    except ValueError:
        return
    raise AssertionError('Invalid measurement accepted')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packed(array):
    array = np.array(array, dtype=float)
    return {'data': base64.b64encode(array.tobytes()).decode(), 'shape': list(array.shape), 'dtype': str(array.dtype)}


def fixture(root):
    geometry = {'anchor': {'nativeHeight': .16759952545629753, 'nativeWidth': .20976605899025014,
        'centerNative': [0, 1, 3], 'normal': [0, 0, 1], 'mPerNative': 1.193320801210449,
        'assumedHeightM': .2, 'assumedWidthM': .2, 'views': [{'photo': p, 'boxRaw': [45, 62, 56, 72]} for p in (1, 2)]},
        'floor': {'normal': [0, 1, 0], 'offset': 0},
        'clearances': [{'id': 'fence-plane-0-lower-rail', 'heightNative': .287,
            'pointNative': [0, .287, 3], 'footNative': [0, 0, 3], 'sourcePhotos': [1, 2],
            'observedViewHeightRangeNative': [.278, .287]}]}
    scene = trimesh.Scene()
    for node, mesh in button_meshes(geometry).items():
        scene.add_geometry(mesh, node_name=node, geom_name=node)
    scene.add_geometry(trimesh.creation.box([1, 2, 3]), node_name='unrelated-post', geom_name='unrelated-post',
                       transform=trimesh.transformations.translation_matrix([2, 1, 4]))
    (root/'object-extras.glb').write_bytes(scene.export(file_type='glb'))
    objects = [{'id': 'emergency-button', 'label': 'Reference', 'measurements': {}, 'observations': [],
                'model': {'file': 'object-extras.glb', 'nodes': list(button_meshes(geometry))}},
               {'id': 'unrelated-post', 'model': {'file': 'object-extras.glb', 'nodes': ['unrelated-post']}}]
    for ident, bottoms in (('fence-0', [.3, 1.8]), ('post-box-1', [.23, .36, .25, .16]), ('post-box-2', [.27, .26, .266, .217])):
        obs = [{'photo': i, 'observedMeasurements': {'status': 'available', 'basis': {
            'corners_native': [[x, y, z] for x in (-1, 1) for y in (bottom, bottom+2) for z in (2, 4)]}}} for i, bottom in enumerate(bottoms, 1)]
        objects.append({'id': ident, 'label': ident, 'observations': obs, 'measurements': {}, 'model': None})
    objects.append({'id': 'single-view', 'label': 'single-view', 'observations': objects[-1]['observations'][:1], 'model': None})
    objects.append({'id': 'robot', 'label': 'robot', 'observations': objects[-2]['observations'], 'model': None, 'modelsByPhoto': {'1': None}})
    (root/'objects.json').write_text(json.dumps({'objects': objects}))
    (root/'geometry.json').write_text(json.dumps(geometry))
    for p in (1, 2):
        frame = {'camera_poses': packed(np.eye(4)), 'intrinsics': packed([[50, 0, 50], [0, 50, 50], [0, 0, 1]]),
                 'input_mask_transform': {'input_to_canonical_pixel_centres': np.eye(3).tolist()}}
        with gzip.open(root/f'frame_{p:04}.json.gz', 'wt') as stream:
            json.dump(frame, stream)


def enriched(root, geometry):
    objects = json.loads((root/'objects.json').read_text())['objects']
    for item in objects:
        item.setdefault('observations', []); item.setdefault('label', item['id'])
        item['visibleHeightByPhoto'] = {str(o['photo']): o['observedMeasurements'].get('dimensions_native', {}).get('height', 2)
                                      for o in item['observations'] if o.get('observedMeasurements', {}).get('status') == 'available'}
        item['groundDistance'] = _ground_distance(item, geometry, np.eye(4))
    return objects


def check(root, inputs):
    original = json.loads((root/'geometry.json').read_text())
    catalog = json.loads((root/'objects.json').read_text())
    frames = {p.name: sha(p) for p in root.glob('frame_*')}
    source = trimesh.load(root/'object-extras.glb', force='scene')
    geometry = apply_measurements(root, inputs)
    assert geometry['anchor']['nativeWidth'] == original['anchor']['nativeWidth']
    assert geometry['anchor']['nativeHeight'] == original['anchor']['nativeHeight']
    assert geometry['anchor']['centerNative'] == original['anchor']['centerNative']
    assert geometry['anchor']['normal'] == original['anchor']['normal']
    assert {k:v for k,v in geometry.items() if k not in ('anchor','calibration')} == {k:v for k,v in original.items() if k not in ('anchor','calibration')}
    assert {p.name: sha(p) for p in root.glob('frame_*')} == frames
    updated = json.loads((root/'objects.json').read_text())
    assert [o.get('observations') for o in updated['objects']] == [o.get('observations') for o in catalog['objects']]
    saved = trimesh.load(root/'object-extras.glb', force='scene')
    assert set(saved.graph.nodes_geometry) == set(source.graph.nodes_geometry)
    for node in source.graph.nodes_geometry:
        if node.startswith('emergency-button-'): continue
        before, after = source.geometry[source.graph[node][1]], saved.geometry[saved.graph[node][1]]
        assert np.array_equal(before.vertices, after.vertices) and np.array_equal(before.faces, after.faces)
        assert np.array_equal(source.graph[node][0], saved.graph[node][0])
    scale = geometry['anchor']['mPerNative']
    assert np.isclose(scale, .10/original['anchor']['nativeHeight'])
    basis = button_basis(geometry); local = []
    for node in saved.graph.nodes_geometry:
        if not node.startswith('emergency-button-'): continue
        matrix, name = saved.graph[node]; mesh = saved.geometry[name]
        points = (trimesh.transform_points(mesh.vertices, matrix)-geometry['anchor']['centerNative'])@basis*scale
        local.append(points)
        if node.endswith(('yellow-body','red-cap')):
            diameter = .085 if node.endswith('yellow-body') else .04
            assert np.allclose(np.ptp(points, axis=0)[[0,2]], diameter, atol=2e-6)
    assert np.isclose(np.ptp(np.concatenate(local)[:,1]), .10, atol=2e-6)
    _export_metric_scene(root, geometry)
    metric = trimesh.load(root/'workcell-metric.glb', force='scene')
    assert metric.metadata['calibration']['reference']['features'] == inputs['reference']['features']
    assert 'button_width_m' not in metric.metadata, 'Measured metadata must distinguish main diameter from envelope width'
    floor_rotation = trimesh.geometry.align_vectors(geometry['floor']['normal'], [0,1,0])[:3,:3]
    local = []
    for node in metric.graph.nodes_geometry:
        if not node.startswith('emergency-button:'): continue
        matrix, name = metric.graph[node]
        points = trimesh.transform_points(metric.geometry[name].vertices, matrix)@floor_rotation@basis
        local.append(points)
        if node.endswith(('yellow-body','red-cap')):
            assert np.allclose(np.ptp(points,axis=0)[[0,2]], .085 if node.endswith('yellow-body') else .04, atol=2e-6)
    assert np.isclose(np.ptp(np.concatenate(local)[:,1]), .10, atol=2e-6)
    result = measurement_evaluation(enriched(root, geometry), geometry, inputs)
    for row in result['comparisons']:
        assert np.isclose(row['signedErrorM'], row['estimateM']-row['groundTruthM'])
        if row['objectId'].startswith('post-box-'):
            assert np.isclose(row['estimateNative'], np.median([v['valueNative'] for v in row['byPhoto'].values()]))
    return geometry, result, {p.name:sha(p) for p in root.glob('*.glb')}


for value in (None, True, 0, -.1, float('nan'), float('inf')):
    for name in measurements['reference']['features']:
        bad = copy.deepcopy(measurements); bad['reference']['features'][name] = value; reject(bad)
bad = copy.deepcopy(measurements); bad['reference']['scope'] = 'red cap only'; reject(bad)
bad = copy.deepcopy(measurements); bad['reference']['features']['wholeEnvelopeWidthM'] = bad['reference']['features'].pop('mainBodyDiameterM'); reject(bad)
for overrides in ((.085, None), (None, .2)):
    try: resolve_dimensions(measurements, *overrides)
    except ValueError: pass
    else: raise AssertionError('Conflicting dimensions accepted')
assert resolve_dimensions(None) == (.2, .2)
assert resolve_dimensions(measurements, None, .1) == (.2, .1)
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)/'original'; root.mkdir()
    if len(sys.argv)>1:
        baseline = Path(sys.argv[1])
        for path in baseline.iterdir():
            if path.is_file() and (path.suffix == '.glb' or path.name in ('geometry.json','objects.json') or path.name.startswith('frame_')):
                shutil.copy2(path, root/path.name)
    else:
        fixture(root)
    changed_root = Path(tmp)/'changed'; shutil.copytree(root, changed_root)
    geometry, result, hashes = check(root, measurements)
    altered = copy.deepcopy(measurements)
    for i, target in enumerate(altered['evaluation']['targets'], 1): target['groundTruthM'] = i*100.
    changed_geometry, changed_result, changed_hashes = check(changed_root, altered)
    invalid_geometry = copy.deepcopy(geometry)
    next(c for c in invalid_geometry['clearances'] if c['id'] == 'fence-plane-0-lower-rail')['heightNative'] = -.1
    invalid_objects = enriched(root, invalid_geometry)
    invalid_fence = next(o for o in invalid_objects if o['id'] == 'fence-0')
    assert invalid_fence['groundDistance']['feature']['valueNative'] is None
    assert any(v['valueNative'] is not None for v in invalid_fence['groundDistance']['byPhoto'].values())
    invalid_result = measurement_evaluation(invalid_objects, invalid_geometry, measurements)
    invalid_comparison = next(c for c in invalid_result['comparisons'] if c['objectId'] == 'fence-0')
    assert invalid_comparison['method'] == 'recognized lower rail feature'
    assert all(invalid_comparison[k] is None for k in ('estimateNative', 'estimateM', 'rangeNative', 'rangeM', 'signedErrorM', 'absoluteErrorM', 'relativeError')), 'Invalid recognized rail must stay unknown despite available whole-mask estimates'
    assert geometry == changed_geometry and hashes == changed_hashes, 'GT changed calibration or mesh bytes'
    assert result['objectEstimates'] == changed_result['objectEstimates']
    error_fields = {'groundTruthM','signedErrorM','absoluteErrorM','relativeError','target'}
    for before, after in zip(result['comparisons'], changed_result['comparisons']):
        assert {k:v for k,v in before.items() if k not in error_fields} == {k:v for k,v in after.items() if k not in error_fields}
    for item in result['objectEstimates']:
        if item['objectId'] in ('single-view','robot'):
            assert item['visibleHeight']['medianNative'] is None and item['lowerEdge']['medianNative'] is None
    print(json.dumps({'scale':geometry['anchor']['mPerNative'],'comparisons':[{k:r[k] for k in ('objectId','estimateM','groundTruthM','signedErrorM')} for r in result['comparisons']]},indent=2))
print('PASS: input validation; reference feature/scope mapping; native evidence unchanged; reference and metric GLB dimensions; per-view medians; no ground-truth leakage into geometry, estimates or model hashes')
