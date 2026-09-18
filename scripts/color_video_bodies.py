"""Project same-frame RGB onto visible human mesh vertices, preserving geometry.

Occluded/unobserved vertices stay neutral. No appearance is carried across time
or track IDs, and no provider is called.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from build_lingbot_replay import read_prediction, resize_mask
from reconstruct_room_rgb import digest


def source_colors(mesh, rgb, mask, k, c2w):
    import open3d as o3d
    if (rgb.dtype != np.uint8 or rgb.shape != (*mask.shape, 3)
            or mask.dtype != bool or k.shape != (3, 3) or c2w.shape != (4, 4)
            or not all(np.isfinite(x).all() for x in [mesh.vertices, k, c2w])):
        raise ValueError('Invalid source-color camera/image domain')
    camera = (mesh.vertices-c2w[:3, 3]) @ c2w[:3, :3]
    q = camera @ k.T
    uv = np.rint(q[:, :2]/np.maximum(q[:, 2:], 1e-8)).astype(np.int64)
    h, w = mask.shape
    valid = (camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    indices = np.flatnonzero(valid)
    # Erode one pixel to keep adjacent background colors out of the silhouette.
    interior = cv2.erode(mask.astype('uint8'), np.ones((3, 3), np.uint8)) > 0
    indices = indices[interior[uv[indices, 1], uv[indices, 0]]]
    cast = o3d.t.geometry.RaycastingScene()
    cast.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(camera.astype('float32')),
        o3d.core.Tensor(mesh.faces.astype('uint32'))))
    # ponytail: face-center visibility avoids ambiguous shared-vertex rays;
    # subtriangle occlusion would require per-pixel texture projection.
    # A face contributes color only when that exact face is the first ray hit.
    centers = camera[mesh.faces].mean(axis=1)
    rays = np.column_stack((np.zeros_like(centers), centers)).astype('float32')
    first = cast.cast_rays(o3d.core.Tensor(rays))['primitive_ids'].numpy()
    visible = np.zeros(len(camera), bool)
    visible[mesh.faces[first == np.arange(len(mesh.faces))].ravel()] = True
    indices = indices[visible[indices]]
    colors = np.tile(np.array([178, 207, 222, 255], np.uint8), (len(camera), 1))
    colors[indices, :3] = rgb[uv[indices, 1], uv[indices, 0]]
    return colors, indices


def build(args):
    import trimesh
    read = lambda p: json.loads(p.read_text())
    scene = read(args.scene); analysis = read(args.analysis); execution = read(args.run/'run.json')
    if (scene['coordinate_frame'] != 'lingbot_native_monocular'
            or scene['source_video_sha256'] != analysis['provenance']['sourceVideoSha256']
            or scene['provenance']['analysis_sha256'] != digest(args.analysis)
            or scene['provenance']['execution_sha256'] != digest(args.run/'run.json')):
        raise ValueError('Color, geometry and video provenance differ')
    if any(r['status'] == 'accepted_model_estimate' for r in scene.get('bodyInterpolation', [])):
        raise ValueError('Same-frame color requires an exported mesh at every displayed sample')
    frames = {f['sourceFrame']: f for f in scene['frames']}
    observations = {f['sourceFrame']: f for f in analysis['frames']}
    native = {f['sourceFrame']: f for f in execution['frames']}
    args.output.mkdir(exist_ok=False, parents=True)
    def relocate(value):
        if isinstance(value, list):
            for item in value: relocate(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if (key.endswith('Url') or key == 'url') and isinstance(item, str) and not item.startswith(('http://', 'https://')):
                    value[key] = os.path.relpath((args.scene.parent/item).resolve(), args.output.resolve())
                else: relocate(item)
    relocate(scene)
    records = []
    for body in scene['bodyKeyframes']:
        index = body['sourceFrame']; entity = body['entityId']; source = args.output/body['meshUrl']
        if digest(source) != body['mesh_sha256']: raise ValueError('Source body changed')
        prediction = args.run/native[index]['file']
        if digest(prediction) != native[index]['sha256']: raise ValueError('Native RGB/camera changed')
        _, _, rgb, k, c2w, _ = read_prediction(prediction)
        if not np.array_equal(c2w, frames[index]['c2w']): raise ValueError('Body camera differs from source RGB')
        obj = next(o for o in observations[index]['objects'] if o['entityId'] == entity)
        mask_path = args.analysis.parent/obj['maskUrl']
        mask = resize_mask(cv2.imread(str(mask_path), -1)[:, :, 3] > 0, (analysis['height'], analysis['width']))
        mesh = trimesh.load(source, force='mesh', process=False)
        vertices = mesh.vertices.copy(); faces = mesh.faces.copy()
        colors, visible = source_colors(mesh, rgb, mask, k, c2w)
        mesh.visual.vertex_colors = colors
        target = args.output/Path(body['meshUrl']).name
        mesh.export(target)
        exported = trimesh.load(target, force='mesh', process=False)
        if not np.array_equal(exported.vertices, vertices) or not np.array_equal(exported.faces, faces):
            raise ValueError('Coloring changed geometry/topology')
        record = {'sourceFrame': index, 'entityId': entity, 'source_body_sha256': body['mesh_sha256'],
            'source_prediction_sha256': native[index]['sha256'], 'source_mask_sha256': digest(mask_path),
            'colored_vertices': len(visible), 'vertices': len(vertices), 'geometry_changed': False,
            'mesh_sha256': digest(target)}
        body.update(meshUrl=target.name, mesh_sha256=record['mesh_sha256'], sourceColor=record)
        records.append(record)
    report = {'source_scene_sha256': digest(args.scene), 'new_provider_calls': 0,
        'method': 'same-frame mask and first-hit visibility, nearest RGB vertex colors',
        'visibility': 'first-hit triangle identity at face center', 'mask_erosion_pixels': 1,
        'hidden_surface_color_inferred': False, 'records': records}
    (args.output/'color-evidence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    scene['provenance']['body_source_color'] = {'url': 'color-evidence.json', 'sha256': digest(args.output/'color-evidence.json')}
    scene['limitations'].append('人体可见部分的颜色来自同一视频帧；背面和遮挡部分保留中性色，未生成衣物几何或跨帧外观。')
    (args.output/'scene.json').write_text(json.dumps(scene, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'bodies': len(records), 'colored_vertices': sum(r['colored_vertices'] for r in records),
        'geometry_changed': False, 'new_provider_calls': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['scene', 'analysis', 'run', 'output']: parser.add_argument('--'+key, type=Path, required=True)
    build(parser.parse_args())
