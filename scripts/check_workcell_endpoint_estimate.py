"""Minimal model-height regression: local rail end, ground pose, units and feet."""
import json
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from workcell_endpoint_estimate import estimate, measure_edge


with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    for angle in (0., .43):
        rotation = trimesh.transformations.rotation_matrix(angle, [1, 2, 0])
        rotation[:3, 3] = [4, -2, 3]
        scenes = {'posts.glb': trimesh.Scene(), 'fence-fitted.glb': trimesh.Scene()}
        for file, node, extents, center in (
            ('posts.glb', 'box-1', [.2, .2, 1], [0, 0, 2.5]),
            ('fence-fitted.glb', 'section-0-continued-3', [3, .02, .2], [2.5, 0, 1.1]),
            # A farther lower rail of the same plane: pairing must take the nearest member.
            ('fence-fitted.glb', 'section-0-continued-4', [3, .02, .2], [9., 0, .7]),
        ):
            mesh = trimesh.creation.box(extents=extents)
            transform = np.eye(4); transform[:3, 3] = center
            scenes[file].add_geometry(mesh, node_name=node, transform=rotation @ transform)
        for file, scene in scenes.items():
            scene.export(root / file)
        normal = rotation[:3, 2]
        offset = -float(normal @ rotation[:3, 3])
        (root / 'physical-clearances.json').write_text(json.dumps({'ground': {'normal': (normal * 3).tolist(), 'offset': offset * 3}}))
        (root / 'objects.json').write_text(json.dumps({'objects': [
            {'id': 'post-box-1', 'kind': 'yellow safety post', 'model': {'file': 'posts.glb', 'nodes': ['box-1']}},
            {'id': 'fence-0', 'kind': 'safety fence', 'model': {'file': 'fence-fitted.glb', 'nodes': ['section-0-continued-3', 'section-0-continued-4']}}]}))
        (root / 'geometry.json').write_text(json.dumps({'fence': {'continuations': [
            {'id': f'section-0-continued-{i}', 'plane': 0, 'role': 'lower-rail continuation'} for i in (3, 4)]}}))
        result = estimate(root)
        light, fence = result['objects']
        assert light['objectId'] == 'post-box-1' and fence['objectId'] == 'fence-0' and fence['node'] == 'section-0-continued-3'
        assert np.isclose(light['heightNative'], 2., atol=1e-6)
        assert np.isclose(fence['heightNative'], 1., atol=1e-6)
        assert result['curtainMinusRail'] == [{'minuendId': light['id'], 'subtrahendId': fence['id'], 'valueNative': light['heightNative'] - fence['heightNative']}]
        assert np.isclose(result['curtainMinusRail'][0]['valueNative'], 1., atol=1e-6)
        expected = trimesh.transform_points([[1, 0, 1]], rotation)[0]
        assert np.allclose(fence['pointNative'], expected, atol=1e-6), fence
        assert result['mPerNative'] is None
        for item in result['objects']:
            point, foot = np.array(item['pointNative']), np.array(item['footNative'])
            assert abs(foot @ normal + offset) < 1e-6
            assert np.allclose(point - foot, item['heightNative'] * normal, atol=1e-6)
            assert abs(item['footReportNative'][2]) < 1e-6
            assert np.isclose(item['pointReportNative'][2], item['heightNative'], atol=1e-6)
yy, xx = np.indices((24, 24))
K = np.array([[100., 0., 12.], [0., 100., 12.], [0., 0., 1.]])
points = np.stack([(xx - 12.) * .05, (yy - 12.) * .05, np.full(xx.shape, 5.)], -1)
frame = {'points': points, 'conf': np.ones(xx.shape), 'valid': np.ones(xx.shape, bool),
         'A': np.eye(3), 'K': K, 'pose': np.eye(4)}
result = measure_edge(frame, yy <= 14, [[8., 14.], [16., 14.]], {'normal': [0., -1., 0.], 'offset': 3.})
assert np.isclose(result['directPointmap']['heightNative'], 2.9)
assert np.isclose(result['localSurface']['heightNative'], 2.9)
assert np.allclose(result['localSurface']['pixelPerturbationRangeNative'], [2.825, 2.975])
assert result['localSurface']['planeResidualP95Native'] < 1e-10
print('PASS: catalog-driven curtain and nearest lower rail, arbitrary floor/units/feet, named RGB edge and local plane intersection')
