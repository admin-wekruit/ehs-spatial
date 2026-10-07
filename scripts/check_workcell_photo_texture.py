"""CPU-only checks for planar photo textures; no real-scene accuracy claim."""
import importlib.util
import io
import json
import struct
import tempfile
import gzip
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PIL import Image

assert importlib.util.find_spec('workcell_photo_texture'), 'Missing geometry-preserving photo texture builder'
import workcell_photo_texture
from workcell_photo_texture import texture_planar_mesh, source_texture_frames
from ehs_spatial.platform.blender_export import mesh_from_asset
from workcell_guard_silhouette import _export, _model


def sample(mesh, point):
    # UV is affine on the unchanged planar surface, unlike perspective photo UV.
    weights = np.linalg.lstsq(np.c_[mesh.vertices[:, :2], np.ones(len(mesh.vertices))].T,
                              np.r_[point[:2], 1.], rcond=None)[0]
    uv = weights @ mesh.visual.uv
    image = np.asarray(mesh.visual.material.baseColorTexture)
    return image[int(round((1 - uv[1]) * image.shape[0] - .5)),
                 int(round(uv[0] * image.shape[1] - .5))]


mesh = trimesh.Trimesh(vertices=[[-.8, -.6, 2.], [.8, -.6, 2.], [.8, .6, 2.], [-.8, .6, 2.]],
                       faces=[[0, 1, 2], [0, 2, 3]], process=False)
mesh.visual.vertex_colors = np.tile([30, 40, 50, 255], (4, 1))
vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
# glTF vertex COLOR_0 is linear, whereas its base-color image is sRGB.
linear_old = np.array([30, 40, 50]) / 255.
original_display = np.rint(np.where(linear_old <= .0031308, linear_old * 12.92,
                                    1.055 * linear_old ** (1 / 2.4) - .055) * 255).astype(np.uint8)
A = np.array([[.5, 0, -3.25], [0, .5, 2.75], [0, 0, 1.]])
raw_K = np.array([[40., 0, 64.], [0, 40., 56.], [0, 0, 1.]])
yy, xx = np.indices((113, 129))
rgb = np.stack([xx, yy, np.full_like(xx, 25)], axis=2).astype(np.uint8)
frame = {'K': A @ raw_K, 'A': A, 'pose': np.eye(4), 'textureRgb': rgb}
mask = np.ones((64, 64), bool)
mask[29:33, 27:31] = False  # A maps the center into this excluded foreground hole.
occluder = np.array([[.15, .05, 1.], [.25, .05, 1.], [.25, .15, 1.], [.15, .15, 1.]])
textured, report = texture_planar_mesh(mesh, {1: frame}, {1: mask},
                                      occluder_polygons=[occluder], max_size=128)
assert np.array_equal(mesh.vertices, vertices) and np.array_equal(mesh.faces, faces)
assert np.array_equal(textured.vertices, vertices) and np.array_equal(textured.faces, faces)
assert np.allclose(sample(textured, [-.5, .2, 2]), [54, 60, 25], atol=1), 'Use full-resolution A^-1 K and correct UV orientation'
assert np.array_equal(sample(textured, [0, 0, 2]), original_display), 'An excluded foreground pixel must keep its original mesh display color'
assert np.array_equal(sample(textured, [.4, .2, 2]), original_display), 'The nearer sheet must occlude the far sheet'
assert 0 < report['photoTexelFraction'] < 1
assert report['geometryUnchanged'] and report['sourcePhotos'] == [1]

# A second photo must not fill gaps with a misregistered cross-view patch.
# One coherent source per sheet preserves unsupported original appearance.
second = {**frame, 'textureRgb': np.full_like(rgb, [10, 150, 20])}
second_mask = np.zeros_like(mask); second_mask[27:35, 25:33] = True
filled, coherent_report = texture_planar_mesh(mesh, {1: frame, 2: second}, {1: mask, 2: second_mask}, max_size=128)
assert len(coherent_report['sourcePhotos']) <= 1, 'Each sheet must use at most one photo source'
assert coherent_report['sourcePhotos'] == [1]
assert np.array_equal(sample(filled, [0, 0, 2]), original_display)
assert np.allclose(sample(filled, [-.5, .2, 2]), [54, 60, 25], atol=1)
reverse_pose = np.diag([1., -1., -1., 1.]); reverse_pose[2, 3] = 4
invisible = {**frame, 'pose': reverse_pose}
visible, visible_report = texture_planar_mesh(mesh, {0: invisible, 1: frame},
                                             {0: np.zeros_like(mask), 1: mask}, max_size=128)
assert visible_report['sourcePhotos'] == [1], 'An invisible opposite-side view must not suppress a visible source'

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory); source = root / 'source.jpg'
    Image.fromarray(rgb).save(source, quality=95)
    raw = {'original_image': {'height': len(rgb), 'width': rgb.shape[1]},
           'input_mask_transform': {'input_to_canonical_pixel_centres': A.tolist()}}
    with gzip.open(root / 'frame_0001.json.gz', 'wt') as stream:
        json.dump(raw, stream)
    loaded = source_texture_frames(root, {1: {'K': frame['K'], 'pose': frame['pose']}}, [source] * 4)
    assert loaded[1]['textureRgb'].shape == rgb.shape and np.array_equal(loaded[1]['A'], A)
    raw['original_image']['width'] += 1
    with gzip.open(root / 'frame_0001.json.gz', 'wt') as stream:
        json.dump(raw, stream)
    try:
        source_texture_frames(root, {1: frame}, [source] * 4)
    except ValueError as error:
        assert 'dimensions' in str(error)
    else:
        raise AssertionError('Mismatched raw raster dimensions must be rejected')

blob = textured.export(file_type='glb')
tree = json.loads(blob[20:20 + struct.unpack('<I', blob[12:16])[0]])
assert 'TEXCOORD_0' in tree['meshes'][0]['primitives'][0]['attributes']
assert 'COLOR_0' not in tree['meshes'][0]['primitives'][0]['attributes'], 'Do not multiply the photo by old vertex colors'
assert tree['materials'][0]['pbrMetallicRoughness']['baseColorTexture']
reopened = trimesh.load(io.BytesIO(blob), file_type='glb', force='mesh', process=False)
assert np.array_equal(reopened.vertices, vertices.astype(np.float32))
assert np.array_equal(reopened.faces, faces)
assert np.allclose(sample(reopened, [-.5, .2, 2]), [54, 60, 25], atol=1)
existing_export = mesh_from_asset(blob, {})
assert existing_export.uv is not None and existing_export.texture_bytes
assert np.array_equal(existing_export.vertices, vertices.astype(np.float32))

# More than OpenCV's 32767 remap-row limit must work at the production size.
large, _ = texture_planar_mesh(mesh, {1: frame}, {1: mask}, max_size=512)
assert max(large.visual.material.baseColorTexture.size) == 512
tilted = mesh.copy()
rotation = trimesh.transformations.rotation_matrix(np.deg2rad(35), [0, 1, 0])
tilted.apply_transform(rotation)
target = trimesh.transform_points([[-.5, .2, 2.]], rotation)[0]
perspective, _ = texture_planar_mesh(tilted, {1: frame}, {1: np.ones_like(mask)}, max_size=128)
raw_target = raw_K @ target; raw_target /= raw_target[2]
assert np.allclose(sample(perspective, target), [*raw_target[:2], 25], atol=2), 'Bake perspective projection into affine planar UV'

bad = mesh.copy(); bad.vertices[0, 2] += .1
try:
    texture_planar_mesh(bad, {1: frame}, {1: mask}, max_size=64)
except ValueError as error:
    assert 'planar' in str(error).lower()
else:
    raise AssertionError('A curved/folded mesh must not silently become a planar texture chart')

template = {'origin': np.array([0., 0., 2.]), 'extent': 1.,
            'rotation': trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3],
            'outlines': [np.array([[0., -.5], [.7, -.5], [.7, .5], [0., .5]])] * 2}
polygons, pose = _model(np.deg2rad(105), np.zeros(9), template)
export_frames = {1: {**frame, 'analysisRgb': rgb, 'C': np.linalg.inv(A)}}
board = {'side': 'left', 'masks': {1: np.ones_like(mask)}}
with tempfile.TemporaryDirectory() as directory:
    old_path = Path(directory) / 'old.glb'
    _export(polygons, pose, board, export_frames, old_path)
    assert hasattr(workcell_photo_texture, 'texture_existing_guards'), 'Missing replay-only texture API'
    # Dataset loading is checked separately above; replay must preserve a real
    # transformed scene graph and GLB buffers without invoking any geometry fit.
    replay_root = Path(directory) / 'input'; replay_root.mkdir()
    for side in ('left', 'right'):
        saved = trimesh.load(old_path, force='scene', process=False)
        for node in saved.graph.nodes_geometry:
            matrix, name = saved.graph[node]
            matrix = matrix.copy()
            matrix[0, 3] += .03
            saved.graph.update(frame_to=node, matrix=matrix, geometry=name)
        saved.export(replay_root / f'guard-{side}.glb')
    boards = {side: {**board, 'side': side} for side in ('left', 'right')}
    replay_out = Path(directory) / 'textured'
    with patch('workcell_guard_joint._inputs', return_value=(export_frames, boards, {})), \
         patch('workcell_photo_texture.source_texture_frames', return_value=export_frames):
        replay = workcell_photo_texture.texture_existing_guards(replay_root, [], replay_out, max_size=128)
    assert replay['geometryUnchanged'] and len(replay['objects']) == 2
    assert all(len(p['sourcePhotos']) <= 1 for obj in replay['objects'] for p in obj['panels'])
    assert (replay_out / 'texture-results.json').is_file()
    assert len(list(replay_out.glob('*.png'))) == 4
    for side in ('left', 'right'):
        old = trimesh.load(replay_root / f'guard-{side}.glb', force='scene', process=False)
        new = trimesh.load(replay_out / f'guard-{side}.glb', force='scene', process=False)
        for node in old.graph.nodes_geometry:
            m, g = old.graph[node]; n, h = new.graph[node]
            assert np.array_equal(m, n) and np.array_equal(old.geometry[g].vertices, new.geometry[h].vertices)
            assert np.array_equal(old.geometry[g].faces, new.geometry[h].faces)
            assert new.geometry[h].visual.kind == 'texture'
print('PASS: raw pixel transform, UV orientation, mask holes, sheet occlusion, source selection, unchanged geometry, GLB texture/export roundtrip')
