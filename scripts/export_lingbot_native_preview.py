"""Inspect native model output, including failed geometry, without accepting a fused map."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from build_lingbot_replay import read_prediction
from build_replay_scene import media_spans, world_points
from reconstruct_room_rgb import digest, pointmap_residuals


def build(run, review, output):
    execution = json.loads((run / 'run.json').read_text())
    plan = json.loads((run / 'plan.json').read_text())
    check = json.loads(review.read_text())
    if (execution['status'] != 'inference_complete'
            or execution['plan_sha256'] != digest(run / 'plan.json')
            or check['execution_sha256'] != digest(run / 'run.json')
            or digest(plan['source_video']) != plan['source_video_sha256']):
        raise ValueError('Native preview source or review changed')
    spans, _ = media_spans(plan['source_video'])
    frames, points, colors, contracts = [], [], [], []
    for i, (source, record) in enumerate(zip(plan['frames'], execution['frames'], strict=True)):
        index = source['sourceFrame']
        path = run / record['file']
        if record['sourceFrame'] != index or digest(path) != record['sha256'] or abs(source['timeSec'] - spans[index][0]) > .001:
            raise ValueError('Prediction and source frame differ')
        depth, confidence, rgb, k, camera, valid = read_prediction(path)
        yy, xx = np.indices(depth.shape)
        xyz = world_points(np.column_stack((xx.ravel(), yy.ravel())), depth.ravel(), k, camera).reshape(*depth.shape, 3)
        contracts.append({'sourceFrame': index, **pointmap_residuals(xyz, depth, k, camera, valid)})
        support = (valid & (confidence >= 1.5))[::6, ::6]
        points.append(xyz[::6, ::6][support]); colors.append(rgb[::6, ::6][support])
        frames.append({'sourceFrame': index, 'timeSec': source['timeSec'],
                       'endTimeSec': plan['frames'][i + 1]['timeSec'] if i + 1 < len(plan['frames']) else spans[-1][1],
                       'c2w': camera.tolist(), 'objects': []})
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(points)))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate(colors) / 255)
    if not len(pc.points):
        raise ValueError('No native points')
    spacing = float(np.linalg.norm(pc.get_max_bound() - pc.get_min_bound())) / 1000
    cloud = pc.voxel_down_sample(spacing)
    while len(cloud.points) > 300000:
        spacing *= 1.2
        cloud = pc.voxel_down_sample(spacing)
    output.mkdir(parents=True, exist_ok=False)
    asset = output / 'native-cloud.glb'
    trimesh.Scene(trimesh.points.PointCloud(np.asarray(cloud.points), colors=np.rint(np.asarray(cloud.colors) * 255).astype('uint8'))).export(asset)
    scene = {'schema': 'phase2-replay-scene-v1', 'coordinate_frame': 'lingbot_native_monocular',
             'units': 'uncalibrated_monocular', 'source_video_sha256': plan['source_video_sha256'],
             'method': 'LingBot-Map native predicted depth and cameras · diagnostic point cloud',
             'complete_room_accepted': False, 'pointCloudUrl': asset.name, 'pointCloudCount': len(cloud.points),
             'points': [[i, *p] for i, p in enumerate(np.asarray(cloud.points)[::max(1, len(cloud.points) // 20000)])],
             'frames': frames, 'provenance': {'execution_sha256': digest(run / 'run.json'),
             'plan_sha256': digest(run / 'plan.json'), 'temporal_geometry_check': check,
             'point_cloud_sha256': digest(asset), 'display_voxel_native': spacing, 'sensor_depth_used': False,
             'groundtruth_used_for_geometry': False, 'geometry_contract_per_frame': contracts},
             'limitations': ['原生预测点云供排查；未通过的跨视角结果没有作为已验证地图接受。',
                             '每6像素采样并体素压缩显示，原始逐帧深度和相机仍保留；重复表面与错位未修复。',
                             '此视图尚未做动态对象排除、静态表面融合或完整物体建模；不能量取现场尺寸。']}
    (output / 'scene.json').write_text(json.dumps(scene, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'frames': len(frames), 'display_points': len(cloud.points), 'cross_view_check_passed': check['passed'],
                      'complete_room_accepted': False, 'scene_sha256': digest(output / 'scene.json')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'review', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    build(a.run, a.review, a.output)
