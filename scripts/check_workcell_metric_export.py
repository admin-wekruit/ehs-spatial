"""Run with project Python: verify report/export coordinates, scale and provenance."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.workcell_photo_oneshot import _export_metric_scene

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    source = trimesh.creation.box([1, .2, .1])
    source.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.zeros((len(source.vertices), 2)), image=Image.new('RGB', (2, 2), 'white'))
    colors = np.array([[40 + i * 20, 120, 210, 255] for i in range(len(source.vertices))], dtype=np.uint8)
    source.visual.vertex_attributes['color'] = colors
    (root/'beam.glb').write_bytes(source.export(file_type='glb'))
    pose = {'coordinateFrameId': 'floor', 'position': [2, 3, .45],
            'quaternion': [0, 0, 0, 1], 'scale': [1, 1, 1]}
    rep = {'id': 'r', 'kind': 'generated_mesh', 'sourceValidity': 'current',
           'assetId': 'beam', 'coordinateFrameId': 'floor', 'transform': pose}
    report = {'assetURLs': {'beam': 'beam.glb'}, 'revision': {'id': 'test', 'documentSha256': 'fixture-document', 'document': {
        'coordinateFrames': [{'id': 'floor', 'ground': {'normal': [0, 0, 1], 'offset': 0}}],
        'entities': [{'id': 'beam', 'activeModelRepresentationId': 'r', 'representations': [rep]}]}}}
    for status, scale, filename in [('accepted_3d_reference', .5, 'workcell-metric.glb'),
                                    ('conditional_unvalidated', .5, 'workcell-conditional.glb'),
                                    ('conditional_unvalidated', 1., 'workcell-conditional.glb'),
                                    ('uncalibrated', None, 'workcell-native.glb')]:
        report['modelMeasurementScale'] = {'status': status, 'nativeToMeters': scale, 'source': 'test'}
        assert _export_metric_scene(root, report) == filename
        scene = trimesh.load(root/filename, force='scene')
        mesh = next(iter(scene.geometry.values()))
        assert mesh.visual.kind == 'texture', 'Export must retain texture visual'
        assert 'color' in mesh.visual.vertex_attributes, 'Export must retain glTF COLOR_0'
        assert np.array_equal(trimesh.visual.color.to_rgba(mesh.visual.vertex_attributes['color']), colors)
        payload = (root/filename).read_bytes()
        gltf = json.loads(payload[20:20 + int.from_bytes(payload[12:16], 'little')])
        assert all('COLOR_0' in primitive['attributes'] and 'TEXCOORD_0' in primitive['attributes']
                   for mesh_json in gltf['meshes'] for primitive in mesh_json['primitives'])
        points = scene.to_geometry().vertices
        factor = 1 if scale is None else scale
        expected = source.vertices + pose['position']
        expected = expected[:, [0, 2, 1]] * [1, 1, -1] * factor
        assert np.allclose(points.min(0), expected.min(0), atol=1e-6)
        assert np.allclose(points.max(0), expected.max(0), atol=1e-6)
        assert np.isclose(points[:, 1].min(), .4*factor, atol=1e-6)
        assert scene.metadata['scale_status'] == status
        assert scene.metadata['modelMeasurementScale']['nativeToMeters'] == scale
        assert scene.metadata['ground'] == {'normal': [0, 1, 0], 'offset': 0}
        assert scene.metadata['groundTruth'] is False
        assert scene.metadata['reportRevision'] == 'test' and scene.metadata['documentSha256'] == 'fixture-document'
    report['modelMeasurementScale']['nativeToMeters'] = float('nan')
    try: _export_metric_scene(root, report)
    except ValueError: pass
    else: raise AssertionError('NaN scale accepted')
print('PASS: exact report pose -> browser Y-up convention; conditional/accepted/native scale; height .4; texture COLOR_0 retained; no false calibration')
