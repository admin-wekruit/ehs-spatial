"""Run: PYTHONPATH=.:scripts python scripts/check_workcell_photo_report_endpoints.py."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import trimesh

from check_workcell_post_geometry import packed
from scripts.workcell_photo_report import build, _ground_distance


def check():
    normal = np.array([.2, -.7, 1.]); normal /= np.linalg.norm(normal)
    offset = -.4
    transform = trimesh.geometry.align_vectors(normal, [0, 0, 1])
    transform[2, 3] = offset
    geometry = {'floor': {'normal': (normal * 3).tolist(), 'offset': offset * 3},
                'anchor': {'assumedHeightM': .1, 'assumedWidthM': .2, 'mPerNative': None}}
    with tempfile.TemporaryDirectory(prefix='workcell-report-endpoints-') as directory:
        root = Path(directory)
        scene, objects, endpoints = trimesh.Scene(), [], []
        for x, (ident, height) in enumerate((('post-box-1', .31), ('fence-0', .18), ('post-box-2', .44))):
            point, foot = trimesh.transform_points([[x, 0, height], [x, 0, 0]], np.linalg.inv(transform))
            mesh = trimesh.Trimesh(trimesh.transform_points(
                [[x, 0, height], [x + .1, 0, height], [x, 0, height + 1]], np.linalg.inv(transform)),
                [[0, 1, 2]], process=False)
            scene.add_geometry(mesh, node_name=ident, geom_name=ident)
            objects.append({'id': ident, 'kind': 'fence' if ident == 'fence-0' else 'light curtain',
                            'label': ident, 'representation': 'synthetic observed face', 'observations': [],
                            'measurements': {}, 'model': {'file': 'terminals.glb', 'nodes': [ident]}})
            endpoints.append({'objectId': ident, 'heightNative': height,
                              'pointNative': point.tolist(), 'footNative': foot.tolist()})
        scene.export(root / 'terminals.glb')
        (root / 'objects.json').write_text(json.dumps({'objects': objects, 'coverage': {}}))
        measured = {'sceneTransformNative': transform.tolist(), 'objects': endpoints,
                    'sourceFiles': {'terminals.glb': hashlib.sha256((root / 'terminals.glb').read_bytes()).hexdigest()}}
        rgb = np.full((2, 2, 3), 128, np.uint8)
        frame = {'image': packed(rgb), 'pts3d': packed([[[0., 0, 3], [.1, 0, 3]], [[0, .1, 3], [.1, .1, 3]]]),
                 'non_ambiguous_mask': packed(np.ones((2, 2), bool)), 'camera_poses': packed(np.eye(4)),
                 'intrinsics': packed([[2., 0, 1], [0, 2, 1], [0, 0, 1]])}
        for photo in range(1, 5):
            Image.fromarray(rgb).save(root / f'photo-{photo}.png')
            with gzip.open(root / f'frame_{photo:04d}.json.gz', 'wt') as stream:
                json.dump(frame, stream)
        endpoint_path = root / 'model-endpoint-estimate.json'
        endpoint_path.write_text(json.dumps(measured))
        for factor in (None, .5):
            geometry['anchor'].update(mPerNative=factor, referenceFit={
                'status': 'unsupported' if factor is None else 'available', 'mPerNative': factor, 'camerasFixed': True})
            (root / 'geometry.json').write_text(json.dumps(geometry))
            report = build(root)
            result = report['endpointEstimation']
            assert json.loads((root / 'scene-report.json').read_text())['endpointEstimation'] == result
            assert len(report['revision']['document']['cameras']) == 4
            assert result['scale']['mPerNative'] == factor
            assert result['groundTruthUsedForEstimation'] is False
            rows = {row['objectId']: row for row in result['endpoints']}
            assert rows['post-box-1']['label'] == '右侧光幕底端'
            assert rows['post-box-2']['label'] == '左侧光幕底端'
            assert rows['fence-0']['label'] == '邻近围栏下沿'
            for source in endpoints:
                row = rows[source['objectId']]
                assert np.isclose(row['heightNative'], source['heightNative'])
                assert np.allclose(np.subtract(row['pointNative'], row['footNative']), [0, 0, source['heightNative']])
                assert abs(row['footNative'][2]) < 1e-12
                if factor is None:
                    assert row['estimateCm'] is None and row['rangeCm'] is None
                else:
                    assert np.isclose(row['estimateCm'], source['heightNative'] * factor * 100)
                    assert np.allclose(row['rangeCm'], [row['estimateCm']] * 2)
            difference = result['difference']
            assert np.isclose(difference['valueNative'], .13)
            if factor is None:
                assert difference['valueCm'] is None and difference['rangeCm'] is None
            else:
                assert np.isclose(difference['valueCm'], 6.5)
                assert np.allclose(difference['rangeCm'], [6.5, 6.5])

        visible = deepcopy(measured)
        visible['objects'][2]['measurementScope'] = 'visible_face_lower_terminal'
        endpoint_path.write_text(json.dumps(visible))
        assert build(root)['endpointEstimation']['endpoints'][2]['label'] == '左侧光幕可见面下沿'
        edge = {**endpoints[2], 'id': 'post-box-2', 'status': 'conditional',
                'sourcePhotos': [2], 'surfaceSupportPhotos': [1, 2, 4]}
        geometry['physicalClearances'] = {'objects': [edge]}
        item = {'id': 'post-box-2', 'physicalBottom': {'measurementScope': 'visible_face_lower_terminal'}}
        assert np.isclose(_ground_distance(item, geometry, transform)['feature']['valueNative'], .44)
        assert _ground_distance({'id': 'post-box-2'}, geometry, transform)['feature'] is None
        edge['surfaceSupportPhotos'] = [2]
        assert _ground_distance(item, geometry, transform)['feature'] is None

        for corruption, message in (('transform', 'different floor'), ('foot', 'Invalid model endpoint'), ('hash', 'is stale')):
            invalid = deepcopy(measured)
            if corruption == 'transform':
                invalid['sceneTransformNative'][2][3] += .1
            elif corruption == 'foot':
                invalid['objects'][0]['footNative'] = (np.asarray(endpoints[0]['footNative']) + .1 * normal).tolist()
            else:
                invalid['sourceFiles']['terminals.glb'] = '0' * 64
            endpoint_path.write_text(json.dumps(invalid))
            previous_report = (root / 'scene-report.json').read_bytes()
            try:
                build(root)
            except ValueError as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError('Invalid endpoint evidence accepted: ' + corruption)
            assert (root / 'scene-report.json').read_bytes() == previous_report
    print('PASS: actual four-frame report build; unknown scale stays native; supplied scale converts cm; left/right labels; tilted shared floor; stale hash and wrong plane rejected')


if __name__ == '__main__':
    check()
