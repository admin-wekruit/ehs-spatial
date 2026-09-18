"""Export dense RGB-D map and per-frame human surfaces for temporal replay.

Uses the existing validated camera/mask/registered-depth contract. Visible
surfaces remain open; this is not anatomical completion of hidden body parts.
"""
from __future__ import annotations
import argparse, copy, json, os, time
from pathlib import Path
import cv2
import numpy as np
from build_video_object_models import observed_surface
from build_replay_scene import world_points
from reconstruct_room_rgb import digest


def export_surface(path, vertices, faces, colors):
    import trimesh
    trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False).export(path)
    check = trimesh.load(path, force='mesh', process=False)
    if not np.allclose(check.vertices, vertices, atol=1e-6, rtol=0) or not np.array_equal(check.faces, faces):
        raise ValueError('Export changed observed surface')


def build(args):
    import trimesh
    started = time.monotonic()
    scene = json.loads(args.scene.read_text())
    native = json.loads(args.native_scene.read_text())
    analysis = json.loads(args.analysis.read_text())
    run = Path(native['source_run']); meta = json.loads((run / 'run.json').read_text())
    if (scene['provenance']['native_scene_sha256'] != digest(args.native_scene)
            or scene['provenance']['analysis_sha256'] != digest(args.analysis)
            or native['evaluation_alignment_applied'] or scene['units'] != 'meters'
            or not scene['provenance']['sensor_depth'] or not meta['execution_complete']
            or digest(run / 'run.json') != native['source_run_sha256']
            or digest(run / 'input.manifest.json') != meta['artifacts_sha256']['input.manifest.json']
            or analysis['provenance']['sourceVideoSha256'] != scene['source_video_sha256']):
        raise ValueError('Camera, masks or metric sensor source differs')
    inputs = {f['source_index']: f for f in json.loads((run / 'input.manifest.json').read_text())}
    frames = {f['sourceFrame']: f for f in analysis['frames']}
    native_frames = {f['source_index']: f for f in native['frames']}
    k = np.array(scene['provenance']['k']); factor = scene['provenance']['depth_factor']
    args.output.mkdir(parents=True, exist_ok=False)
    result = copy.deepcopy(scene)
    def relocate(url): return os.path.relpath((args.scene.parent / url).resolve(), args.output.resolve())
    if result.get('meshUrl'): result['meshUrl'] = relocate(result['meshUrl'])
    for obj in result.get('staticObjects', []):
        for key in ['meshUrl', 'provenanceUrl']: obj[key] = relocate(obj[key])
        for key in ['maskUrl', 'imageUrl']: obj['source'][key] = relocate(obj['source'][key])
    if result.get('semanticReview'): result['semanticReview']['url'] = relocate(result['semanticReview']['url'])
    all_points, all_colors, all_frames, records = [], [], [], []
    total_faces = total_bytes = surfaces = 0
    # ponytail: 3-pixel surface sampling bounds per-frame transfer; do not bridge
    # missing depth or hold a preceding frame during a temporal gap.
    step = 3
    for frame in result['frames']:
        index = frame['sourceFrame']; item = inputs[index]
        if index not in frames: continue
        observed = frames[index]; c2w = np.array(frame['c2w'])
        if not np.array_equal(c2w, native_frames[index]['c2w']): raise ValueError('Camera changed')
        if abs(item['depth_timestamp'] - native_frames[index]['timestamp']) >= .02:
            raise ValueError('Unassociated depth exposure')
        depth_path, image_path = Path(item['depth_source_path']), Path(item['source_path'])
        if digest(depth_path) != item['depth_sha256'] or digest(image_path) != item['sha256']:
            raise ValueError('Source image or depth changed')
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        color = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        if raw.dtype != np.uint16 or raw.shape != color.shape[:2]: raise ValueError('RGB-D domain differs')
        depth = raw.astype(np.float32) / factor
        excluded = np.zeros(depth.shape, bool)
        masks = {}
        for obj in observed['objects']:
            path = (args.analysis.parent / obj['maskUrl']).resolve()
            if not path.is_relative_to(args.analysis.parent.resolve()): raise ValueError('Mask outside analysis')
            rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if rgba.shape != (*depth.shape, 4): raise ValueError('Mask domain differs')
            mask = rgba[:, :, 3] > 0; excluded |= mask; masks[obj['entityId']] = (mask, digest(path))
        for obj in frame['objects']:
            if obj['entityId'] not in masks: raise ValueError('Object is not from current frame')
            mask, mask_hash = masks[obj['entityId']]
            small_k = k.copy(); small_k[:2] /= step
            v, faces, colors, pixels, count = observed_surface(color[::step, ::step], depth[::step, ::step], mask[::step, ::step], small_k, c2w)
            if not len(faces): continue
            name = f"human-{index:05d}-{obj['entityId'].rsplit('-', 1)[-1]}.glb"
            export_surface(args.output / name, v, faces, colors)
            # Verify the actual GLB vertices against the exact input pixel/depth.
            local = (v - c2w[:3, 3]) @ c2w[:3, :3]; uvw = local @ k.T
            reprojection = float(np.max(np.abs(uvw[:, :2] / uvw[:, 2:] - pixels * step)))
            if reprojection > .001: raise ValueError('Human surface is not registered to source pixels')
            obj['surface'] = {'meshUrl': name, 'representation': 'visible_rgbd_surface', 'sourceFrame': index,
                              'sha256': digest(args.output / name), 'triangles': len(faces)}
            records.append({'sourceFrame': index, 'entityId': obj['entityId'], 'mask_sha256': mask_hash,
                            'rgb_sha256': item['sha256'], 'depth_sha256': item['depth_sha256'],
                            'mesh_sha256': obj['surface']['sha256'], 'max_reprojection_error_px': reprojection})
            total_faces += len(faces); total_bytes += (args.output / name).stat().st_size; surfaces += 1
        if index % 5 == 0 and excluded.any():
            excluded = cv2.dilate(excluded.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            valid = (~excluded) & (depth >= .2) & (depth <= 5)
            sample = np.zeros(depth.shape, bool); sample[::3, ::3] = True; valid &= sample
            y, x = np.where(valid)
            all_points.append(world_points(np.column_stack((x, y)), depth[valid], k, c2w))
            all_colors.append(color[valid]); all_frames.append(np.full(len(x), index, np.int32))
        if index % 100 == 0: print(json.dumps({'frame': index, 'surfaces': surfaces}), flush=True)
    points, colors, source_frames = np.concatenate(all_points), np.concatenate(all_colors), np.concatenate(all_frames)
    # Voxel support counts DISTINCT frames, not pixels: remove isolated depth speckle.
    voxel = .015
    while True:
        _, inverse = np.unique(np.floor(points / voxel).astype(np.int32), axis=0, return_inverse=True)
        count = np.bincount(inverse)
        support = np.bincount(np.unique(np.column_stack((inverse, source_frames)), axis=0)[:, 0], minlength=len(count))
        keep = support >= 2
        if int(keep.sum()) <= 300000: break
        voxel *= 1.2
    xyz = np.column_stack([np.bincount(inverse, weights=points[:, j]) / count for j in range(3)])[keep]
    rgb = np.rint(np.column_stack([np.bincount(inverse, weights=colors[:, j]) / count for j in range(3)])[keep]).astype(np.uint8)
    cloud = trimesh.points.PointCloud(xyz, colors=rgb); cloud.export(args.output / 'dense-static.ply')
    trimesh.Scene(cloud).export(args.output / 'dense-static.glb')
    result['pointCloudUrl'] = 'dense-static.glb'
    result['pointCloudCount'] = len(xyz)
    result['provenance']['visual_surface_source_scene_sha256'] = digest(args.scene)
    result['provenance']['dense_cloud'] = {'sha256': digest(args.output / 'dense-static.glb'), 'voxel_m': voxel,
                                        'minimum_distinct_frames': 2, 'raw_points': len(points), 'display_points': len(xyz)}
    result['humanSurfaces'] = {'representation': 'visible_rgbd_surface', 'samplingPixels': step, 'count': surfaces,
                             'coordinateFrame': scene['coordinate_frame'], 'hidden_body_completed': False}
    result['limitations'] = [s for s in result['limitations'] if '人体为可见表面骨架' not in s]
    result['limitations'].append('人物彩色网格来自每一帧的有效深度和人物掩码；背面和遮挡仍无观测，不能当作完整人体形状。')
    (args.output / 'surface-provenance.json').write_text(json.dumps(records))
    result['provenance']['human_surface_records_sha256'] = digest(args.output / 'surface-provenance.json')
    (args.output / 'scene.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
    report = {'elapsedSeconds': time.monotonic() - started, 'surfaces': surfaces, 'trianglesAcrossTime': total_faces,
              'humanAssetBytes': total_bytes, 'densePoints': len(xyz), 'voxelM': voxel,
              'scene_sha256': digest(args.output / 'scene.json'), 'max_reprojection_error_px': max(r['max_reprojection_error_px'] for r in records)}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['scene', 'native-scene', 'analysis', 'output']: p.add_argument('--' + name, type=Path, required=True)
    build(p.parse_args())
