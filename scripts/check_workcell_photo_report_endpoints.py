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
        ids = {'post-box-1': 'post-box-1:terminal', 'fence-0': 'fence-0:near:post-box-1', 'post-box-2': 'post-box-2:terminal'}
        for x, (ident, height) in enumerate((('post-box-1', .31), ('fence-0', .18), ('post-box-2', .44))):
            point, foot = trimesh.transform_points([[x, 0, height], [x, 0, 0]], np.linalg.inv(transform))
            mesh = trimesh.Trimesh(trimesh.transform_points(
                [[x, 0, height], [x + .1, 0, height], [x, 0, height + 1]], np.linalg.inv(transform)),
                [[0, 1, 2]], process=False)
            scene.add_geometry(mesh, node_name=ident, geom_name=ident)
            objects.append({'id': ident, 'kind': 'safety fence' if ident == 'fence-0' else 'yellow safety post',
                            'label': ident, 'representation': 'synthetic observed face', 'observations': [],
                            'measurements': {}, 'model': {'file': 'terminals.glb', 'nodes': [ident]}})
            endpoints.append({'id': ids[ident], 'objectId': ident, 'node': ident, 'modelFile': 'terminals.glb', 'heightNative': height,
                              'measurementScope': 'model_lower_rail_near_curtain' if ident == 'fence-0' else 'model_bottom_face_center',
                              **({'pairedEndpointId': ids['fence-0']} if ident == 'post-box-1' else {}),
                              **({'railPart': 'lower_edge', 'memberRole': 'lower-rail continuation'} if ident == 'fence-0' else {}),
                              'pointNative': point.tolist(), 'footNative': foot.tolist()})
        scene.export(root / 'terminals.glb')
        (root / 'objects.json').write_text(json.dumps({'objects': objects, 'coverage': {}}))
        measured = {'schemaVersion': 3, 'sceneTransformNative': transform.tolist(), 'objects': endpoints,
                    'curtainMinusRail': [{'minuendId': ids['post-box-1'], 'subtrahendId': ids['fence-0'], 'valueNative': .13}],
                    'sourceFiles': {'terminals.glb': hashlib.sha256((root / 'terminals.glb').read_bytes()).hexdigest()}}
        rgb = np.full((2, 2, 3), 128, np.uint8)
        # Photo cameras look along -Y of the report floor from y=+5, so report -X is image right.
        report_camera = np.eye(4); report_camera[:3, :3] = [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]; report_camera[:3, 3] = [1, 5, .5]
        frame = {'image': packed(rgb), 'pts3d': packed([[[0., 0, 3], [.1, 0, 3]], [[0, .1, 3], [.1, .1, 3]]]),
                 'non_ambiguous_mask': packed(np.ones((2, 2), bool)), 'camera_poses': packed(np.linalg.inv(transform) @ report_camera),
                 'intrinsics': packed([[2., 0, 1], [0, 2, 1], [0, 0, 1]])}
        for photo in range(1, 5):
            Image.fromarray(rgb).save(root / f'photo-{photo}.png')
            with gzip.open(root / f'frame_{photo:04d}.json.gz', 'wt') as stream:
                json.dump(frame, stream)
        endpoint_path = root / 'model-endpoint-estimate.json'
        endpoint_path.write_text(json.dumps(measured))
        estimates = []
        for factor in (None, .5):
            geometry['anchor'].update(mPerNative=factor, referenceFit={
                'status': 'unsupported' if factor is None else 'available', 'mPerNative': factor, 'camerasFixed': True})
            (root / 'geometry.json').write_text(json.dumps(geometry))
            report = build(root)
            result = report['endpointEstimation']
            estimates.append(result)
            assert json.loads((root / 'scene-report.json').read_text())['endpointEstimation'] == result
            assert len(report['revision']['document']['cameras']) == 4
            assert result['groundTruthUsedForEstimation'] is False
            rows = {row['objectId']: row for row in result['endpoints']}
            assert rows['post-box-1']['label'] == '右侧光幕底端' and rows['post-box-1']['side'] == 'right'
            assert rows['post-box-2']['label'] == '左侧光幕底端' and rows['post-box-2']['side'] == 'left'
            assert rows['fence-0']['label'] == '右侧光幕旁围栏下沿' and rows['fence-0']['side'] == 'right'
            assets = {asset['id']: asset['sha256'] for asset in report['revision']['document']['assets']}
            for source in endpoints:
                row = rows[source['objectId']]
                assert not {'estimateCm', 'rangeCm'} & row.keys(), 'no centimetre snapshot beside the native value'
                assert row['representationId'] == 'rep-' + row['objectId'] and assets[row['assetId']] == row['assetSha256']
                assert np.isclose(row['heightNative'], source['heightNative'])
                assert np.allclose(np.subtract(row['pointNative'], row['footNative']), [0, 0, source['heightNative']])
                assert abs(row['footNative'][2]) < 1e-12
            differences = {row['id']: row for row in result['differences']}
            assert np.isclose(differences['post-box-1:terminal-minus-rail']['valueNative'], .13)
            assert np.isclose(differences['curtain-left-minus-right']['valueNative'], .13)
            assert 'rail-left-minus-right' not in differences, 'one rail has no left/right counterpart'
            assert not any('Cm' in key for row in result['differences'] for key in row)
        # A scale-only change (null or supplied) leaves every native measurement identical.
        assert estimates[0] == estimates[1]

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

        for corruption, message in (('transform', 'different floor'), ('foot', 'Invalid model endpoint'), ('hash', 'is stale'),
                                    ('schema', 'schema 1 instead of 3'), ('schema-2', 'schema 2 instead of 3'),
                                    ('rail-part', 'rail endpoint without railPart')):
            invalid = deepcopy(measured)
            if corruption == 'schema':
                invalid['schemaVersion'] = 1; invalid.pop('curtainMinusRail')
            elif corruption == 'schema-2':  # a stale table that still carries curtainMinusRail is refused by its version alone
                invalid['schemaVersion'] = 2
            elif corruption == 'rail-part':  # a rail point that does not say which part it measures fails closed
                next(row for row in invalid['objects'] if row['objectId'] == 'fence-0').pop('railPart')
            elif corruption == 'transform':
                invalid['sceneTransformNative'][2][3] += .1
            elif corruption == 'foot':
                invalid['objects'][0]['footNative'] = (np.asarray(endpoints[0]['footNative']) + .1 * normal).tolist()
            else:
                invalid['sourceFiles']['terminals.glb'] = '0' * 64
            endpoint_path.write_text(json.dumps(invalid))
            previous = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
            try:
                build(root)
            except ValueError as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError('Invalid endpoint evidence accepted: ' + corruption)
            # A refused build leaves the report and every asset it hashed byte-identical, with no staging left behind.
            assert previous == {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
            assert not any(path.is_dir() for path in root.iterdir())
    print('PASS: actual four-frame report build; endpoints bound to displayed representations; scale-only change leaves native endpoints identical; photo-4 left/right labels and differences; tilted shared floor; stale hash, stale schema and wrong plane rejected without touching any file')


if __name__ == '__main__':
    check()
