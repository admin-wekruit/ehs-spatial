"""Run: PYTHONPATH=.:scripts python scripts/check_workcell_post_geometry.py."""
import base64
import gzip
import json
from pathlib import Path
import tempfile

import numpy as np
import trimesh

from workcell_photo_oneshot import _posts


def packed(value):
    value = np.asarray(value)
    return {'data': base64.b64encode(value.tobytes()).decode(),
            'shape': list(value.shape), 'dtype': str(value.dtype)}


def encoded(mask):
    values = mask.ravel(order='F').astype(np.uint8)
    ends = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)]
    counts = np.diff(ends).tolist()
    if values[0]:
        counts.insert(0, 0)
    return json.dumps({'size': list(mask.shape), 'counts': counts})


def check():
    normal = np.array([.2, -.9, .3]); normal /= np.linalg.norm(normal)
    basis = trimesh.geometry.align_vectors([0, 1, 0], normal)[:3, :3]
    points = np.empty((40, 80, 3))
    masks = []
    for index in range(4):
        y, x = np.meshgrid(np.linspace(.37, 2.17, 40), np.linspace(-.12, .12, 20), indexing='ij')
        local = np.stack([x + index * .6, y + index * .13, x * .1 + 3.], axis=-1)
        points[:, index * 20:(index + 1) * 20] = local @ basis.T
        mask = np.zeros((40, 80), bool); mask[:, index * 20:(index + 1) * 20] = True
        masks.append(mask)
    frame = {'pts3d': packed(points), 'image': packed(np.full((40, 80, 3), 160, np.uint8)),
             'non_ambiguous_mask': packed(np.ones((40, 80), bool)),
             'input_mask_transform': {'input_to_canonical_pixel_centres': np.eye(3).tolist()}}
    segmentation = {'prompts': [{'text': word} for word in ('yellow safety post', 'black bollard')],
                    'results': [[]] * 3 + [[{'rle': [encoded(mask) for mask in masks[start:start + 2]],
                                             'scores': [.99, .98]} for start in (0, 2)]]}
    with tempfile.TemporaryDirectory(prefix='post-ground-check-') as directory:
        root = Path(directory)
        with gzip.open(root / 'frame_0004.json.gz', 'wt') as stream:
            json.dump(frame, stream)

        def fit(up):
            (root / 'geometry.json').write_text(json.dumps({'floor': {'normal': list(up), 'offset': .4}}))
            records = _posts(root, segmentation)
            scene = trimesh.load(root / 'posts.glb', force='scene', process=False)
            return records, {node: trimesh.transform_points(scene.geometry[scene.graph[node][1]].vertices,
                                                            scene.graph[node][0]) for node in scene.graph.nodes_geometry}

        records, meshes = fit(normal)
        assert len(records) == len(meshes) == 4
        for index, row in enumerate(records):
            assert np.allclose(row['groundNormalNative'], normal)
            assert 'not a measured' in row['axisStatus']
            vertices = meshes[row['name']]
            axis = np.linalg.svd(vertices - vertices.mean(0), full_matrices=False)[2][0]
            assert abs(axis @ normal) > 1 - 1e-7, (row['name'], axis, normal)
            native = vertices @ basis
            expected = np.percentile((points[masks[index]] @ basis)[:, 1], [2, 98])
            assert np.allclose([native[:, 1].min(), native[:, 1].max()], expected, atol=5e-7)
            assert np.allclose(native.mean(0)[[0, 2]], np.median(points[masks[index]] @ basis, axis=0)[[0, 2]], atol=5e-7)
        # Replacing the plane must refit the original source cloud, not rotate an
        # already fitted model again. This is the physical-clearance update path.
        fit([0, 1, 0])
        _, repeated = fit(normal)
        assert all(np.array_equal(meshes[name], repeated[name]) for name in meshes)
        for invalid in ([0, 0, 0], [0, 1], [0, float('nan'), 0]):
            try:
                fit(invalid)
            except ValueError as error:
                assert 'ground normal' in str(error)
            else:
                raise AssertionError('Invalid ground basis accepted')
    print('PASS: four post axes match the actual ground normal; longitudinal bounds/centers preserved; ground refresh is idempotent; invalid normals rejected')


if __name__ == '__main__':
    check()
