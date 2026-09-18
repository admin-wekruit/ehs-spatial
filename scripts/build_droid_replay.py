"""Export a completed DROID result without GT alignment or metric-scale claims.

Camera replay uses the official full motion-only filler trajectory. TSDF uses
only final native keyframe c2w/depth/K, never filler poses or stale disps_up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from build_replay_scene import media_spans, world_points
from reconstruct_room_rgb import digest, integrate, new_volume, pointmap_residuals, validate_frame
from reconstruct_tum_room import read_rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def rigid_poses(poses, count):
    require(poses.shape == (count, 4, 4) and np.isfinite(poses).all(), 'Invalid pose array')
    rotation = poses[:, :3, :3]
    require(np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            and np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-4, rtol=0)
            and np.allclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0), 'Expected proper rigid c2w poses')


def depth_support(poses, disparity, intrinsics, dtype=np.float64):
    """CPU math of pinned DROID viewer filtering, over exported valid keyframes.

    This is not a CUDA/full-capacity-buffer replay. The unusual forward offsets
    and four-corner OR follow droid_kernels.cu:661,730-733, not bilinear sampling.
    """
    poses, disparity, intrinsics = (np.asarray(a, dtype=dtype) for a in (poses, disparity, intrinsics))
    count, height, width = disparity.shape
    fx, fy, cx, cy = intrinsics[0]
    y, x = np.indices((height, width), dtype=dtype)
    rays = np.stack([(x - cx) / fx, (y - cy) / fy, np.ones_like(x)], axis=-1)
    votes = np.zeros(disparity.shape, np.uint8)
    with np.errstate(divide='ignore', invalid='ignore'):
        depths = 1 / disparity
        for i in range(count):
            for offset in [-1, -2, -3, 3, 4, 5]:
                j = i + offset
                if not 0 <= j < count:
                    continue
                relative = np.linalg.inv(poses[j]) @ poses[i]
                projected = rays @ relative[:3, :3].T + disparity[i, :, :, None] * relative[:3, 3]
                uv = projected[:, :, :2] / projected[:, :, 2:] * [fx, fy] + [cx, cy]
                target_z = projected[:, :, 2] / disparity[i]
                inside = (np.isfinite(uv).all(axis=-1) & (uv[:, :, 0] >= 0) & (uv[:, :, 0] < width - 1)
                          & (uv[:, :, 1] >= 0) & (uv[:, :, 1] < height - 1))
                py, px = np.where(inside)
                u, v = np.floor(uv[inside]).astype(int).T
                nearby = np.stack([depths[j, v, u], depths[j, v, u + 1],
                                   depths[j, v + 1, u], depths[j, v + 1, u + 1]], axis=-1)
                votes[i, py, px] += np.any(np.abs(nearby - target_z[inside, None]) < .005, axis=-1)
    prior = disparity > .5 * disparity.mean(axis=(1, 2), keepdims=True)
    retained = np.isfinite(disparity) & (disparity > 0) & (votes >= 2) & prior
    return votes, prior, retained


def keyframe_arrays(data, count, preprocessing):
    indices = data['keyframe_source_indices']
    require(indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer) and len(indices) > 0
            and np.all(np.diff(indices) > 0) and indices[0] >= 0 and indices[-1] < count,
            'Keyframes must reference unique ordered input frames')
    poses = data['keyframe_c2w']
    rigid_poses(poses, len(indices))
    rigid_poses(data['poses_c2w'], count)
    width, height = preprocessing['model_wh']
    require(width % 8 == 0 and height % 8 == 0, 'Expected the native eightfold depth raster')
    disparity = data['keyframe_final_lowres_inverse_depth']
    depth = data['keyframe_final_lowres_depth']
    valid = data['keyframe_final_lowres_valid']
    require(disparity.shape == depth.shape == valid.shape == (len(indices), height // 8, width // 8)
            and valid.dtype == np.bool_, 'Unexpected final low-resolution depth domain')
    numeric = np.isfinite(disparity) & (disparity > 0)
    require(np.array_equal(valid, numeric) and np.isfinite(depth).all()
            and np.all(depth[~valid] == 0) and np.all(depth[valid] > 0)
            and np.allclose(depth[valid] * disparity[valid], 1, atol=2e-6, rtol=0),
            'Final depth, inverse depth and numeric validity disagree')
    low_k = data['keyframe_final_lowres_intrinsics']
    expected_k = np.asarray(preprocessing['model_intrinsics_fx_fy_cx_cy'])
    require(low_k.shape == (len(indices), 4) and np.isfinite(low_k).all()
            and np.allclose(low_k * 8, expected_k, atol=1e-5, rtol=0), 'Final low-resolution K must be model K / 8')
    bgr = data['keyframe_model_bgr']
    require(bgr.shape == (len(indices), height, width, 3) and bgr.dtype == np.uint8,
            'Missing exact model-domain keyframe colors')
    return indices, poses, depth, valid, low_k, bgr


def fullres_arrays(data, low_depth, low_k, colors):
    """The official offline viewer samples full-resolution results with stride 2."""
    disparity = data['keyframe_final_fullres_inverse_depth']
    depth = data['keyframe_final_fullres_depth']
    valid = data['keyframe_final_fullres_valid']
    full_k = data['keyframe_final_fullres_intrinsics']
    require(depth.shape == disparity.shape == valid.shape == colors.shape[:3]
            and depth.shape[0] == low_depth.shape[0] and valid.dtype == np.bool_,
            'Final upsampled depth must share the exact model RGB raster')
    numeric = np.isfinite(disparity) & (disparity > 0)
    require(np.array_equal(valid, numeric) and np.isfinite(depth).all()
            and np.all(depth[~valid] == 0) and np.all(depth[valid] > 0)
            and np.allclose(depth[valid] * disparity[valid], 1, atol=2e-6, rtol=0),
            'Final full-resolution depth and inverse depth disagree')
    require(full_k.shape == low_k.shape and np.isfinite(full_k).all()
            and np.allclose(full_k, low_k * 8, atol=1e-5, rtol=0),
            'Final full-resolution intrinsics must equal model K')
    return depth[:, ::2, ::2], valid[:, ::2, ::2], full_k / 2, disparity[:, ::2, ::2]


def build(args):
    import open3d as o3d
    import trimesh

    started = time.monotonic()
    require(not args.output.exists(), 'Output must be a new directory')
    require(np.isfinite(args.voxel_length_native) and args.voxel_length_native > 0,
            'Explicit native voxel length must be finite and positive')
    run = args.run.resolve()
    require((run / 'remote-run.json').is_file() and (run / 'prediction.npz').is_file(),
            'A real completed native result is required before export')
    remote, local = read_json(run / 'remote-run.json'), read_json(run / 'run.json')
    require(remote['status'] == 'inference_complete' and remote['scale'] == 'uncalibrated_monocular'
            and remote['groundtruth_uploaded'] is False and remote['sensor_depth_uploaded'] is False
            and remote['groundtruth_alignment_applied'] is False and remote['full_poses_are_filler_output'] is True
            and remote['source_revision'] == local['source_revision'],
            'Require a completed, unaligned, RGB-only native result')
    final_fullres = args.depth_domain == 'final-fullres'
    required = ['prediction.npz', 'frames.jsonl', 'preprocessing.json', 'pose-contract.json', 'inference-config.json']
    if final_fullres:
        require(remote.get('contract_version') == 'droid-final-upsampling-v2',
                'Full-resolution fusion requires verified post-BA learned upsampling')
        required.append('final-upsampling-validation.json')
    for name in required:
        require(digest(run / name) == remote['artifacts_sha256'][name], 'Changed native artifact: ' + name)
    if final_fullres:
        upsampling_check = read_json(run / 'final-upsampling-validation.json')
        require(all(upsampling_check.get(name) is True for name in
                    ['final_lowres_unchanged', 'final_native_poses_unchanged', 'full_keyframe_mask_coverage'])
                and upsampling_check.get('max_abs_recompute_error') == 0,
                'Final learned upsampling validation did not pass')
    require(digest(run / 'input-manifest.json') == local['input_manifest_sha256']
            and digest(run / 'runner-at-execution.py') == local['script_sha256'], 'Changed run input or runner')
    inputs, pre = read_json(run / 'input-manifest.json'), read_json(run / 'preprocessing.json')
    require(inputs['groundtruth_included'] is False and inputs['depth_included'] is False, 'Expected RGB-only input')
    records = inputs['frames']; count = len(records)
    require(count == inputs['frame_count'] == remote['frames_processed'] == remote['full_pose_count'], 'Incomplete native trajectory')
    ledger = [json.loads(line) for line in (run / 'frames.jsonl').read_text().splitlines()]
    require(len(ledger) == count, 'Incomplete native frame ledger')
    video = read_json(args.video_manifest)
    spans, shape = media_spans(args.video)
    require(video['schema'] == 'phase2-source-video-v1'
            and digest(args.video) == video['playback']['sha256']
            and len(spans) == count == video['playback']['input_frames'] == video['playback']['decoded_frames']
            and shape == (video['playback']['height'], video['playback']['width'])
            and video['encoding']['every_source_frame_once_in_original_order'] is True
            and len(video['frames']) == count, 'Video is not the exact declared complete RGB sequence')
    dataset = (args.video_manifest.parent / video['original_root']).resolve()
    require(digest(dataset / 'rgb.txt') == inputs['rgb_index_sha256'] == video['rgb_index_sha256'], 'Changed original RGB index')
    rows = read_rows(dataset / 'rgb.txt', 2)
    require(len(rows) == count, 'Input index length differs')
    original_hashes = {item['path']: item['sha256'] for item in video['original_files']}
    require(np.array_equal(pre['source_K'], inputs['source_K_fx_fy_cx_cy'])
            and np.array_equal(pre['source_distortion'], inputs['source_distortion'])
            and pre['undistort_new_K'] == 'same as source K', 'Preprocessing calibration disagrees')
    fx, fy, cx, cy = pre['source_K']
    source_k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    resize_w, resize_h = pre['resize_wh']; x0, y0, x1, y1 = pre['crop_xyxy']
    require(0 <= x0 < x1 <= resize_w and 0 <= y0 < y1 <= resize_h
            and pre['model_wh'] == [x1 - x0, y1 - y0], 'Invalid preprocessing raster')
    scaled_k = [fx * resize_w / shape[1], fy * resize_h / shape[0],
                cx * resize_w / shape[1] - x0, cy * resize_h / shape[0] - y0]
    require(np.allclose(scaled_k, pre['model_intrinsics_fx_fy_cx_cy'], atol=2e-5, rtol=0),
            'Model K does not follow declared native scale/crop convention')
    # Every domain is explicit. The original pre-BA upsampled snapshot is rejected.
    with np.load(run / 'prediction.npz', allow_pickle=False) as data:
        indices, poses, depth, valid, low_k, colors = keyframe_arrays(data, count, pre)
        disparity = data['keyframe_final_lowres_inverse_depth']
        depth_array = 'keyframe_final_lowres_depth'
        color_sampling = 'model BGR[3::8,3::8,::-1], official visualization convention'
        if final_fullres:
            depth, valid, low_k, disparity = fullres_arrays(data, depth, low_k, colors)
            depth_array = 'keyframe_final_fullres_depth[:,::2,::2]'
            color_sampling = 'model BGR[::2,::2,::-1], final full-resolution disparity[::2,::2], K=model K/2; official offline convention'
        require(len(indices) == remote['keyframe_count'], 'Native keyframe count disagrees')
        depth_statistics = {'quantiles_0_1_10_50_90_99_100_percent_native': np.quantile(depth[valid], [0, .01, .1, .5, .9, .99, 1]).tolist(),
                            'numeric_valid_pixels': int(valid.sum()),
                            'median_depth_pixel_footprint_native': float(np.median(depth[valid]) / np.mean(low_k[:, :2]))}
        full_poses = data['poses_c2w']
        difference = np.abs(full_poses[indices] - poses)
        contract = read_json(run / 'pose-contract.json')
        translation_difference = np.linalg.norm(full_poses[indices, :3, 3] - poses[:, :3, 3], axis=1).max()
        require(np.isclose(difference.max(), contract['full_vs_keyframe_max_matrix_difference'], atol=1e-7, rtol=0)
                and np.isclose(translation_difference, contract['full_vs_keyframe_max_translation_difference'], atol=1e-7, rtol=0),
                'Recorded native/filler pose difference disagrees')
        key_lookup = {int(source): i for i, source in enumerate(indices)}
        for index, (record, row, log, frame, span) in enumerate(zip(records, rows, ledger, video['frames'], spans)):
            source = (dataset / record['relative_path']).resolve()
            require(source.is_relative_to(dataset) and record['relative_path'] == row[1]
                    and record['source_index'] == log['source_index'] == log['native_timestamp'] == frame['source_index'] == frame['media_frame_index'] == index
                    and record['timestamp_text'] == row[0] == log['source_timestamp_text']
                    and abs(float(row[0]) - frame['timestamp']) < 2e-6
                    and Path(frame['source_path']).resolve() == source
                    and digest(source) == record['sha256'] == log['rgb_sha256'] == frame['source_sha256'] == original_hashes[row[1]]
                    and np.allclose(span, [frame['media_pts_seconds'], frame['media_end_seconds']], atol=1e-4, rtol=0),
                    'Source frame/hash/time mapping differs at frame ' + str(index))
            image = cv2.imread(str(source))
            require(image is not None and image.shape == (*shape, 3), 'Changed source raster')
            canonical = cv2.resize(cv2.undistort(image, source_k, np.asarray(pre['source_distortion'])),
                                   (resize_w, resize_h))[y0:y1, x0:x1]
            require(hashlib.sha256(canonical.tobytes()).hexdigest() == log['model_bgr_sha256'], 'Model raster hash differs')
            if index in key_lookup:
                require(np.array_equal(canonical, colors[key_lookup[index]]), 'Saved keyframe colors differ from native input')
        volume = new_volume(args.voxel_length_native)
        support_count, depth_prior, supported = depth_support(poses, disparity, low_k)
        counts32, _, supported32 = depth_support(poses, disparity, low_k, np.float32)
        support_contract = {
            'method': 'CPU mathematical reproduction of DROID visualization defaults on exported final keyframes',
            'source_revision': remote['source_revision'],
            'source': 'droid_slam/visualization.py; src/droid_kernels.cu:629-735',
            'neighbor_offsets': [-1, -2, -3, 3, 4, 5], 'depth_difference_threshold_native': .005,
            'neighbor_test': 'Any of four floor/ceil neighbouring pixels passes strict absolute z-depth difference; at most one vote per view',
            'minimum_votes': 2, 'inverse_depth_prior': 'disparity > 0.5 * mean disparity in this keyframe',
            'active_keyframe_bounds': [0, len(indices)], 'gpu_bit_exact_claim': False,
            'buffer_scope': 'Only exported final keyframes, excluding unused capacity and temporary filler slots',
            'numeric_valid_pixels': int(valid.sum()), 'retained_pixels': int(supported.sum()),
            'rejected_insufficient_votes': int((valid & (support_count < 2)).sum()),
            'rejected_prior_after_votes_pass': int((valid & (support_count >= 2) & ~depth_prior).sum()),
            'float32_vs_float64_count_differences': int((counts32 != support_count).sum()),
            'float32_vs_float64_retained_differences': int((supported32 != supported).sum())}
        geometry, point_blocks, color_blocks, point_count = [], [], [], 0
        for key, source in enumerate(indices):
            fx, fy, cx, cy = low_k[key]
            k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
            # Match official visualization.py sampling; no extra K pixel offset.
            rgb = np.ascontiguousarray(colors[key, ::2, ::2, ::-1] if final_fullres else colors[key, 3::8, 3::8, ::-1])
            if valid[key].any():
                validate_frame(rgb, depth[key], np.ones_like(depth[key]), valid[key], k, poses[key])
                y, x = np.where(valid[key])
                point_blocks.append(world_points(np.column_stack((x, y)), depth[key][valid[key]], k, poses[key]))
                color_blocks.append(rgb[valid[key]])
            if supported[key].any():
                integrate(volume, rgb, np.where(supported[key], depth[key], 0), k, poses[key])
            end_point = point_count + int(valid[key].sum())
            geometry.append({'sourceFrame': int(source), 'nativeKeyframeIndex': key,
                             'c2w': poses[key].tolist(), 'K': k.tolist(),
                             'poseSource': 'droid_final_native_keyframe',
                             'numericValidPixels': int(valid[key].sum()),
                             'fusionSupportedPixels': int(supported[key].sum()),
                             'pointRowsStartStop': [point_count, end_point],
                             'depthArray': depth_array, 'rawArrayIndex': key})
            point_count = end_point
        mesh = volume.extract_triangle_mesh()
        require(not mesh.is_empty() and np.isfinite(np.asarray(mesh.vertices)).all(), 'No finite predicted surface extracted')
        mesh.compute_vertex_normals()
        model = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles),
                               vertex_normals=np.asarray(mesh.vertex_normals),
                               vertex_colors=np.rint(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8), process=False)
        args.output.mkdir(parents=True, exist_ok=False)
        np.savez(args.output / 'depth-support.npz', count=support_count, inverse_depth_prior=depth_prior,
                 retained=supported, count_float32=counts32, retained_float32=supported32)
        write_json(args.output / 'depth-support.json', support_contract)
        model.export(args.output / 'predicted-scene.glb')
        reread = trimesh.load(args.output / 'predicted-scene.glb', force='mesh', process=False)
        require(len(reread.faces) == len(model.faces) and np.allclose(reread.bounds, model.bounds, atol=1e-6), 'GLB export changed geometry')
        points, point_colors = np.concatenate(point_blocks), np.concatenate(color_blocks)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.colors = o3d.utility.Vector3dVector(point_colors.astype(float) / 255)
        require(o3d.io.write_point_cloud(str(args.output / 'native-keyframe-points.ply'), cloud), 'PLY export failed')
        cloud_read = o3d.io.read_point_cloud(str(args.output / 'native-keyframe-points.ply'))
        require(np.array_equal(np.asarray(cloud_read.points), points)
                and np.array_equal(np.rint(np.asarray(cloud_read.colors) * 255).astype(np.uint8), point_colors),
                'PLY export changed native XYZ or RGB')
        # ponytail: deterministic display sampling only; full point order/colors
        # remain in PLY and raw arrays. Larger browser sets need streamed assets.
        display_stride = max(1, int(np.ceil(len(points) / 300000)))
        display_ids = np.arange(0, len(points), display_stride)
        scene = {'schema': 'phase2-replay-scene-v1', 'coordinate_frame': 'droid_final_native_world',
                 'units': 'uncalibrated_monocular', 'source_video_sha256': digest(args.video),
                 'method': f'DROID full motion-only filler camera replay; {args.depth_domain} native keyframe TSDF with viewer-rule depth support',
                 'points': [[int(i), *points[i].tolist()] for i in display_ids],
                 'meshUrl': 'predicted-scene.glb', 'complete_room_accepted': False,
                 'quality_status': 'not_validated',
                 'frames': [{'sourceFrame': i, 'timeSec': span[0], 'endTimeSec': span[1],
                             'c2w': full_poses[i].tolist(), 'poseSource': 'droid_motion_only_filler', 'objects': []}
                            for i, span in enumerate(spans)],
                 'coverage': {'inputFrames': count, 'trajectorySamples': len(full_poses), 'filledTrajectory': True,
                              'geometryKeyframes': len(indices), 'geometryFramesWithNumericDepth': sum(g['numericValidPixels'] > 0 for g in geometry)},
                 'provenance': {'source_run': str(run), 'source_revision': remote['source_revision'],
                                'weights_sha256': remote['weights_sha256'],
                                'source_run_sha256': digest(run / 'run.json'), 'remote_run_sha256': digest(run / 'remote-run.json'),
                                'input_manifest_sha256': digest(run / 'input-manifest.json'),
                                'video_manifest_sha256': digest(args.video_manifest),
                                'raw_prediction_path': str(run / 'prediction.npz'), 'raw_prediction_sha256': digest(run / 'prediction.npz'),
                                'raw_arrays': {name: {'shape': list(data[name].shape), 'dtype': str(data[name].dtype)} for name in data.files},
                                'gt_pose_input': False, 'evaluation_alignment_applied': False, 'sensor_depth_input': False,
                                'geometry_pose_source': 'keyframe_c2w', 'camera_pose_source': 'poses_c2w motion-only filler',
                                'native_filler_pose_difference': contract, 'preprocessing': pre,
                                'local_runtime': {'opencv': cv2.__version__, 'numpy': np.__version__, 'open3d': o3d.__version__},
                                'depth_statistics': depth_statistics,
                                'raw_depth_validity': 'numeric_validity_only',
                                'surface_support': support_contract,
                                'surface_support_arrays_sha256': digest(args.output / 'depth-support.npz'),
                                'depth_domain': args.depth_domain, 'color_sampling': color_sampling,
                                'tsdf': {'voxel_length_native': volume.voxel_length, 'sdf_trunc_native': volume.sdf_trunc,
                                         'voxel_selection_reason': args.voxel_reason},
                                'pointcloud': {'path': 'native-keyframe-points.ply',
                                               'sha256': digest(args.output / 'native-keyframe-points.ply'),
                                               'raw_points': len(points), 'displayed_points': len(display_ids),
                                               'display_stride': display_stride,
                                               'row_order': 'keyframe order, then row-major numeric-valid pixels; display ID is full PLY row'},
                                'mesh_sha256': digest(args.output / 'predicted-scene.glb')},
                 'limitations': ['单目原生尺度未标定；尺寸和距离不是米。',
                                 '全帧相机是官方 motion-only 补全输出，不表示每帧跟踪成功。',
                                 f'表面仅用终态关键帧 {depth.shape[2]}×{depth.shape[1]} 深度；未混入补全位姿或旧上采样深度。',
                                 '原始点云保留全部数值有效深度；表面使用官方可视化规则的 CPU 支持检查，不是 CUDA 位级复现。',
                                 '模型支持筛选不代表独立几何精度或完整房间已验收。',
                                 '没有人体或物体掩码；场景中的运动物体未被剔除。']}
        write_json(args.output / 'geometry-frames.json', geometry)
        scene['provenance']['geometry_frames_sha256'] = digest(args.output / 'geometry-frames.json')
        write_json(args.output / 'scene.json', scene)
        metrics = {'execution_complete': True, 'quality_status': 'not_validated',
                   'elapsed_seconds': time.monotonic() - started, 'trajectorySamples': count,
                   'geometryKeyframes': len(indices), 'vertices': len(model.vertices), 'triangles': len(model.faces),
                   'raw_points': len(points), 'displayed_points': len(display_ids),
                   'surface_supported_pixels': int(supported.sum()),
                   'scene_sha256': digest(args.output / 'scene.json'),
                   'prediction_sha256_unchanged': digest(run / 'prediction.npz') == remote['artifacts_sha256']['prediction.npz']}
        require(metrics['prediction_sha256_unchanged'], 'Native prediction changed during export')
        write_json(args.output / 'metrics.json', metrics)
        (args.output / 'adapter-at-execution.py').write_bytes(Path(__file__).read_bytes())
        write_json(args.output / 'command.json', {'cwd': str(Path.cwd()), 'argv': [sys.executable, *sys.argv],
                                                'environment': {'PYTHONPATH': os.environ.get('PYTHONPATH')},
                                                'adapter_sha256': digest(Path(__file__)),
                                                'helpers_sha256': {name: digest(Path(__file__).parent / name) for name in
                                                                  ['build_replay_scene.py', 'reconstruct_room_rgb.py', 'reconstruct_tum_room.py']}})
        print(json.dumps(metrics))


def self_check():
    planes = np.ones((7, 5, 6), np.float32) * .5
    poses = np.tile(np.eye(4), (7, 1, 1)); ks = np.tile([10, 10, 2, 2], (7, 1))
    votes, prior, retained = depth_support(poses, planes, ks)
    assert np.array_equal(votes[:, 2, 2], [3, 4, 4, 4, 3, 3, 3])
    assert not votes[:, -1].any() and not votes[:, :, -1].any()
    assert retained[:, :4, :5].all() and prior.all()
    for translation in [[100, 0, 0], [0, 0, -4]]:
        moved = poses.copy(); moved[0, :3, 3] = translation
        assert not depth_support(moved, planes, ks)[0][0].any(), 'Off-raster/behind-camera views must not support depth'
    mismatch = planes.copy(); mismatch[0] = 1
    assert not depth_support(poses, mismatch, ks)[0][0].any(), 'Occluding or inconsistent depths must not vote'
    # Three disagreeing corners do not hide the one matching observed corner.
    corners = planes.copy(); corners[3:, 1:3, 1:3] = 1; corners[3:, 2, 2] = .5
    assert depth_support(poses, corners, ks)[0][0, 1, 1] == 3
    k = np.array([[45., 0, 19.5], [0, 45., 14.5], [0, 0, 1]])
    pose = np.eye(4); angle = .3
    pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    pose[:3, 3] = [.7, -.2, .4]
    rigid_poses(pose[None], 1)
    depth = np.full((30, 40), 2., np.float32); yy, xx = np.indices(depth.shape)
    xyz = world_points(np.column_stack((xx.ravel(), yy.ravel())), depth.ravel(), k, pose).reshape(30, 40, 3)
    residual = pointmap_residuals(xyz, depth, k, pose, np.ones_like(depth, bool))
    assert residual['pointmap_xyz_reprojection_error_p95_px'] < 1e-10
    assert pointmap_residuals(xyz, depth, k, np.linalg.inv(pose), np.ones_like(depth, bool))['pointmap_xyz_reprojection_error_p95_px'] > 1
    image = np.zeros((240, 320, 3), np.uint8); image[3::8, 3::8] = [10, 20, 30]
    rgb = np.ascontiguousarray(image[3::8, 3::8, ::-1]); assert np.all(rgb == [30, 20, 10])
    volume = new_volume(.025); integrate(volume, rgb, depth, k, pose)
    vertices = np.asarray(volume.extract_triangle_mesh().vertices)
    local = (vertices - pose[:3, 3]) @ pose[:3, :3]
    assert len(vertices) > 100 and np.quantile(np.abs(local[:, 2] - 2), .95) < .025
    sample = {'keyframe_source_indices': np.array([0]), 'keyframe_c2w': pose[None], 'poses_c2w': pose[None],
              'keyframe_final_lowres_depth': depth[None], 'keyframe_final_lowres_inverse_depth': .5 * np.ones_like(depth[None]),
              'keyframe_final_lowres_valid': np.ones_like(depth[None], bool),
              'keyframe_final_lowres_intrinsics': np.array([[45, 45, 19.5, 14.5]]),
              'keyframe_intrinsics_fx_fy_cx_cy': np.array([[360, 360, 156, 116]]), 'keyframe_model_bgr': image[None]}
    pre = {'model_wh': [320, 240], 'model_intrinsics_fx_fy_cx_cy': [360, 360, 156, 116]}
    keyframe_arrays(sample, 1, pre)
    full_depth = np.full((1, 240, 320), 2., np.float32)
    sample.update(keyframe_final_fullres_depth=full_depth,
                  keyframe_final_fullres_inverse_depth=np.full_like(full_depth, .5),
                  keyframe_final_fullres_valid=np.ones_like(full_depth, bool),
                  keyframe_final_fullres_intrinsics=sample['keyframe_intrinsics_fx_fy_cx_cy'])
    half_depth, _, half_k, _ = fullres_arrays(sample, depth[None], sample['keyframe_final_lowres_intrinsics'], image[None])
    assert half_depth.shape == (1, 120, 160) and np.array_equal(half_k, [[180, 180, 78, 58]])
    # Stride sampling preserves precisely the original full-resolution camera ray.
    full_pixel, half_pixel = np.array([80., 100.]), np.array([40., 50.])
    assert np.allclose((full_pixel - [156, 116]) / [360, 360], (half_pixel - half_k[0, 2:]) / half_k[0, :2])
    sample['keyframe_final_fullres_intrinsics'] = sample['keyframe_final_lowres_intrinsics']
    try:
        fullres_arrays(sample, depth[None], sample['keyframe_final_lowres_intrinsics'], image[None])
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong full-resolution K accepted')
    sample['keyframe_final_lowres_depth'] = depth[None] * 2
    try:
        keyframe_arrays(sample, 1, pre)
    except ValueError:
        pass
    else:
        raise AssertionError('Inconsistent reciprocal depth accepted')
    print('DROID replay check passed: official neighbour/four-corner support, borders, off-raster/behind-camera/occlusion rejection, final depth/K, reciprocal depth rejection, c2w roundtrip, RGB sampling and inverse-c2w TSDF')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--video-manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--voxel-length-native', type=float)
    parser.add_argument('--depth-domain', choices=['final-lowres', 'final-fullres'], default='final-lowres')
    parser.add_argument('--voxel-reason', default='Explicit native-unit parameter; no ground-truth scale used')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif all([args.run, args.video, args.video_manifest, args.output]) and args.voxel_length_native is not None:
        build(args)
    else:
        parser.error('Provide --run, --video, --video-manifest, --output and explicit --voxel-length-native, or --self-check')
