"""Runnable end-to-end export check: raw points, one floor transform, photo binding."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.workcell_photo_report import build


def encoded(value):
    value = np.ascontiguousarray(value)
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'data': base64.b64encode(value.tobytes()).decode()}


def read_points(path):
    raw = path.read_bytes(); size = struct.unpack_from('<I', raw, 12)[0]
    header = json.loads(raw[20:20+size]); blob = raw[28+size:]
    assert header['meshes'][0]['primitives'][0]['mode'] == 0
    count = header['accessors'][0]['count']
    xyz = np.frombuffer(blob, '<f4', count=3*count).reshape(-1, 3)
    rgba = np.frombuffer(blob, np.uint8, offset=12*count, count=4*count).reshape(-1, 4)
    return xyz, rgba[:, :3]


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    geometry = {'floor': {'normal': [0, -1, 0], 'offset': 2},
                'anchor': {'assumedHeightM': .1, 'assumedWidthM': .085, 'mPerNative': None}}
    (root / 'geometry.json').write_text(json.dumps(geometry))
    # The generated mesh is deliberately unrelated to the source cloud.
    scene = trimesh.Scene(); scene.add_geometry(trimesh.creation.box(), node_name='far-model')
    scene.apply_translation([500, 500, 500]); scene.export(root / 'model.glb')
    observation = {'photo': 1, 'box': [1, 1, 4, 4], 'polygons': [[[1,1],[3,1],[3,3],[1,3]]], 'source': 'saved source mask'}
    catalog = {'objects': [{'id': 'button-test', 'kind': 'emergency_button', 'label': 'test',
        'observations': [observation], 'representation': 'generated',
        'model': {'file': 'model.glb', 'nodes': ['far-model']}}], 'coverage': {}}
    (root / 'objects.json').write_text(json.dumps(catalog))
    yy, xx = np.indices((8, 8))
    points = np.stack([xx / 10, yy / 10, np.full_like(xx, 4)], -1).astype(np.float32)
    points[0, 0] = np.nan
    valid = np.ones((8, 8), bool); valid[7, 7] = False
    rgb = np.stack([xx * 20, yy * 20, np.full_like(xx, 80)], -1).astype(np.uint8)
    for i in range(1, 5):
        frame = {'pts3d': encoded(points), 'image': encoded(rgb), 'non_ambiguous_mask': encoded(valid),
                 'camera_poses': encoded(np.eye(4)), 'intrinsics': encoded(np.eye(3))}
        with gzip.open(root / f'frame_{i:04d}.json.gz', 'wt') as stream:
            json.dump(frame, stream)
        Image.fromarray(rgb).save(root / f'photo-{i}.png')
    result = build(root)
    doc = result['revision']['document']; transform = np.asarray(result['sceneTransformNative'])
    assert result['nativeToMetersDefault'] is None
    assert doc['coordinateFrames'][0]['scale']['nativeToMeters'] is None
    context = next(e for e in doc['entities'] if e.get('sourceContext'))
    assert len(context['representations']) == len(context['observationRefs']) == 4
    good = valid & np.isfinite(points).all(-1)
    for i, rep in enumerate(context['representations'], 1):
        assert rep['sourceRefs'][0]['imageId'] == f'photo-{i}'
        assert rep['sourceRefs'][0]['observationId'] in context['observationRefs']
        asset = next(a for a in doc['assets'] if a['id'] == rep['assetId'])
        path = root / result['assetURLs'][rep['assetId']]
        assert path.name.startswith('entity-')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset['sha256']
        xyz, colors = read_points(path)
        assert len(xyz) == 62 and np.allclose(xyz, trimesh.transform_points(points[good], transform))
        assert np.array_equal(colors, rgb[good])
        # Restore source camera coordinates: verifies that floor transform was applied once.
        camera = np.asarray(doc['cameras'][i-1]['cameraToWorld'])
        assert np.allclose(trimesh.transform_points(xyz, np.linalg.inv(camera)), points[good])
    obj = next(e for e in doc['entities'] if e['id'] == 'button-test')
    rep = next(r for r in obj['representations'] if r['kind'] == 'point_cloud')
    xyz, colors = read_points(root / result['assetURLs'][rep['assetId']])
    assert len(xyz) == 9 and np.max(abs(xyz)) < 10  # no points sampled from the far generated mesh
    assert np.array_equal(colors, rgb[1:4, 1:4].reshape(-1, 3))
    assert rep['sourceRefs'][0]['observationId'] == obj['observationRefs'][0]
print('PASS: full source points/RGB, exactly one floor transform, photo/object binding, no metric or mesh fabrication')
