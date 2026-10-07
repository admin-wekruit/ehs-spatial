"""Small synthetic check: physical source edges share the displayed ground."""
import copy
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.workcell_photo_report import _ground_distance


# Saved mask bounds cannot supply a physical endpoint, even in multiple views.
item = {'id': 'post-box-1', 'observations': [
    {'photo': photo, 'observedMeasurements': {'status': 'available', 'basis': {
        'corners_native': [[x, y, z] for x in (-1, 1) for y in (-2, 2) for z in (2.1, 3.1)]}}}
    for photo in (1, 2)]}
geometry = {'floor': {'normal': [0, 0, 2], 'offset': -4}, 'clearances': []}
transform = np.eye(4); transform[2, 3] = -2
assert _ground_distance(item, geometry, transform)['feature'] is None
row = {'id': item['id'], 'status': 'conditional', 'heightNative': .3,
       'pointNative': [.2, -.4, 2.3], 'footNative': [.2, -.4, 2.],
       'sourcePhotos': [2, 1, 2], 'rangeNative': [.28, .32]}
geometry['physicalClearances'] = {'objects': [row]}
result = _ground_distance(item, geometry, transform)
feature = result['feature']
assert result['byPhoto'] == {}, 'A common physical edge became per-photo mask extrema'
assert result['sourcePhotos'] == [1, 2] and result['rangeNative'] == [.28, .32]
assert np.allclose(feature['pointNative'], [.2, -.4, .3])
assert np.allclose(feature['footNative'], [.2, -.4, 0.])
assert np.isclose(np.linalg.norm(np.array(feature['pointNative']) - feature['footNative']), feature['valueNative'])

for changes in ({'status': 'unsupported'}, {'sourcePhotos': [1, 1]}, {'heightNative': -.1},
                {'heightNative': float('nan')}, {'pointNative': []}, {'footNative': [0, 0, float('inf')]}):
    invalid = {**geometry, 'physicalClearances': {'objects': [{**row, **changes}]}}
    assert _ground_distance(item, invalid, transform)['feature'] is None, changes
unknown_range = {**geometry, 'physicalClearances': {'objects': [{**row, 'rangeNative': None}]}}
assert _ground_distance(item, unknown_range, transform)['feature']['rangeNative'] is None

# Floor normal and offset must be normalized together. Rotating the whole
# source world leaves the report's upright endpoint and height unchanged.
rotation = trimesh.transformations.rotation_matrix(.6, [1, 0, 0])
rotated_row = copy.deepcopy(row)
for key in ('pointNative', 'footNative'):
    rotated_row[key] = trimesh.transform_points([row[key]], rotation)[0].tolist()
tilted = {'floor': {'normal': (rotation[:3, :3] @ [0, 0, 2]).tolist(), 'offset': -4},
          'physicalClearances': {'objects': [rotated_row]}}
rotated_result = _ground_distance(item, tilted, transform @ np.linalg.inv(rotation))
assert np.allclose(rotated_result['feature']['pointNative'], feature['pointNative'])
assert np.allclose(rotated_result['feature']['footNative'], feature['footNative'])
assert rotated_result['feature']['valueNative'] == feature['valueNative']

for floor in ({'normal': [0, 0, 0], 'offset': 0},
              {'normal': [0, 0, float('nan')], 'offset': -4},
              {'normal': [0, 0], 'offset': -4},
              {'normal': [0, 0, 2], 'offset': float('inf')},
              {'normal': [0, 0, 2], 'offset': -4.1}):
    try:
        _ground_distance(item, {**geometry, 'floor': floor}, transform)
    except ValueError:
        pass
    else:
        raise AssertionError(f'Invalid or mismatched displayed plane was accepted: {floor}')

broken = {**row, 'footNative': [.2, -.4, 2.02]}
try:
    _ground_distance(item, {**geometry, 'physicalClearances': {'objects': [broken]}}, transform)
except ValueError:
    pass
else:
    raise AssertionError('Physical endpoint outside the displayed plane was accepted')
print('PASS: physical-edge evidence, normalized/tilted floor, rigid endpoint transform, unknown evidence and mismatched ground rejection')
