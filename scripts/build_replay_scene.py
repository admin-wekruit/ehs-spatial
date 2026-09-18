"""Bind final estimated camera poses to video time; optionally fuse sensor RGB-D.

The RGB-D experiment uses real sensor depth, never GT poses or scale alignment.
Human surface joints are visible-depth samples, not a fitted anatomical skeleton.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from reconstruct_room_rgb import digest, integrate, new_volume, validate_frame
from reconstruct_tum_room import read_rows


def media_spans(video):
    cap = cv2.VideoCapture(str(video))
    times, size = [], None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            size = frame.shape[:2]
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    finally:
        cap.release()
    # ponytail: supplied public previews are CFR. VFR needs packet end times;
    # never manufacture a final interval for a differently timed upload.
    if len(times) < 2 or not np.isfinite(fps) or fps <= 0 or not np.allclose(
        np.diff(times), 1 / fps, atol=.0001, rtol=0
    ):
        raise ValueError('This exporter requires verified CFR media timestamps')
    return list(zip(times, [*times[1:], times[-1] + 1 / fps])), size


def world_points(uv, z, k, c2w):
    uv = np.asarray(uv, float)
    local = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(k).T
    local *= np.asarray(z)[:, None]
    return local @ c2w[:3, :3].T + c2w[:3, 3]


def surface_joints(joints, depth, mask, k, c2w):
    """Require mask-supported, locally consistent sensor depth at each joint."""
    result = []
    height, width = depth.shape
    for joint in joints:
        if joint is None or not np.isfinite(joint).all() or joint[2] < .3:
            result.append(None)
            continue
        x, y = np.rint(joint[:2]).astype(int)
        if not 0 <= x < width or not 0 <= y < height or not mask[y, x]:
            result.append(None)
            continue
        ys, xs = slice(max(0, y - 2), min(height, y + 3)), slice(max(0, x - 2), min(width, x + 3))
        patch = depth[ys, xs]
        support = patch[(patch > 0) & np.isfinite(patch) & mask[ys, xs]]
        if len(support) < 3 or np.percentile(support, 90) - np.percentile(support, 10) > .12:
            result.append(None)
            continue
        result.append(world_points([joint[:2]], [np.median(support)], k, c2w)[0].tolist())
    return result


def build(args):
    import trimesh

    started = time.monotonic()
    native = json.loads(args.native_scene.read_text())
    run = Path(native['source_run'])
    meta = json.loads((run / 'run.json').read_text())
    if not meta['execution_complete'] or digest(run / 'run.json') != native['source_run_sha256']:
        raise ValueError('Camera execution changed or did not complete')
    if native['evaluation_alignment_applied'] or native['coordinate_frame'] != 'final_native_atlas':
        raise ValueError('Expected final native estimated map without GT alignment')
    spans, shape = media_spans(args.video)
    height, width = shape
    video_manifest = json.loads(args.video_manifest.read_text())
    dataset = Path(meta['dataset']).resolve()
    declared_root = (args.video_manifest.parent / video_manifest['original_root']).resolve()
    originals = {f['path']: f['sha256'] for f in video_manifest['original_files']}
    identity_mapping = ('MP4 frame i corresponds to data row i of original rgb.txt '
                        '(zero-based, excluding comment lines). Every RGB input frame appears once and in order.')
    if (declared_root != dataset or video_manifest['playback']['sha256'] != digest(args.video)
            or video_manifest['playback'].get('frame_mapping') != identity_mapping
            or video_manifest['playback']['input_frames'] != len(spans)
            or video_manifest['playback']['decoded_frames'] != len(spans)
            or originals['rgb.txt'] != meta['rgb_index_sha256']
            or digest(dataset / 'rgb.txt') != meta['rgb_index_sha256']):
        raise ValueError('Video is not the declared complete camera RGB source')
    rgb_rows = read_rows(dataset / 'rgb.txt', 2)
    if len(rgb_rows) != len(spans):
        raise ValueError('Video and source frame counts differ')
    inputs = {f['source_index']: f for f in json.loads((run / 'input.manifest.json').read_text())}
    for name in ['input.manifest.json', 'camera.yaml']:
        if digest(run / name) != meta['artifacts_sha256'][name]:
            raise ValueError('Camera input artifact changed: ' + name)
    for frame in native['frames']:
        if frame['source_index'] not in inputs or not 0 <= frame['source_index'] < len(spans):
            raise ValueError('Pose source frame is not present in the input video')
        if Path(frame['source_image']).resolve() != Path(inputs[frame['source_index']]['source_path']).resolve():
            raise ValueError('Pose source image differs from camera input')
        if digest(frame['source_image']) != inputs[frame['source_index']]['sha256']:
            raise ValueError('Camera source image changed')
        source_row = rgb_rows[frame['source_index']]
        if ((dataset / source_row[1]).resolve() != Path(frame['source_image']).resolve()
                or abs(float(source_row[0]) - frame['timestamp']) > .000002
                or originals[source_row[1]] != inputs[frame['source_index']]['sha256']):
            raise ValueError('Video source-frame mapping differs from camera inputs')
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    scene = {'schema': 'phase2-replay-scene-v1', 'coordinate_frame': native['coordinate_frame'],
             'units': native['units'], 'source_video_sha256': digest(args.video),
             'method': 'ORB-SLAM3 final estimated camera and sparse map',
             'points': native['points'], 'frames': [], 'complete_room_accepted': False,
             'map_id': native['map_id'], 'coverage': native['coverage'],
             'limitations': ['仅显示同一张最终地图中的有效相机；丢失与未合并区域不连接。'],
             'provenance': {'native_scene_sha256': digest(args.native_scene),
                            'input_video_sha256': digest(args.video),
                            'video_manifest_sha256': digest(args.video_manifest), 'gt_pose_input': False}}
    analysis, pairs, volume = None, {}, None
    if args.analysis:
        if native['units'] != 'meters' or meta.get('sensor') != 'rgbd':
            raise ValueError('Sensor-depth fusion requires an explicitly metric RGB-D camera run')
        analysis = json.loads(args.analysis.read_text())
        if (analysis['provenance']['sourceVideoSha256'] != scene['source_video_sha256']
                or (analysis['height'], analysis['width']) != shape):
            raise ValueError('Masks and camera video do not share the same source/domain')
        analysis = {f['sourceFrame']: f for f in analysis['frames']}
        dataset = Path(meta['dataset'])
        if (digest(dataset / 'rgb.txt') != meta['rgb_index_sha256']
                or digest(dataset / 'depth.txt') != meta['depth_index_sha256']):
            raise ValueError('Sensor source index changed')
        rgb = read_rows(dataset / 'rgb.txt', 2)
        if len(rgb) != len(spans):
            raise ValueError('Video must preserve every source RGB frame')
        pairs = {i: f for i, f in inputs.items() if 'depth_source_path' in f}
        settings = cv2.FileStorage(str(run / 'camera.yaml'), cv2.FILE_STORAGE_READ)
        try:
            def value(name):
                n = settings.getNode(name)
                if n.empty():
                    raise ValueError('Missing camera calibration: ' + name)
                return n.real()
            if settings.getNode('File.version').string() != '1.0':
                raise ValueError('Expected ORB camera settings version 1.0')
            k = np.array([[value('Camera1.fx'), 0, value('Camera1.cx')],
                          [0, value('Camera1.fy'), value('Camera1.cy')], [0, 0, 1]])
            factor = value('RGBD.DepthMapFactor')
            if factor <= 0 or any(abs(value('Camera1.' + d)) > 1e-10 for d in ['k1', 'k2', 'p1', 'p2']):
                raise ValueError('This fusion requires registered undistorted RGB-D and positive depth factor')
        finally:
            settings.release()
        volume = new_volume()
        scene['method'] += ' + sensor RGB-D TSDF with observed person masks excluded'
        scene['provenance'].update({'analysis_sha256': digest(args.analysis), 'sensor_depth': True,
                                    'k': k.tolist(), 'depth_factor': factor})
        scene['limitations'] += ['此三维对照使用传感器深度，不代表普通 RGB 视频已有相同精度。',
                                 '人体为可见表面骨架估计；遮挡关节不补全，不是人体形状拟合。',
                                 '静态融合只用有人员分割的帧；尚未发现的后入画对象不在当前种子轨迹中。']
    fused, depth_records = 0, []
    for frame in native['frames']:
        index = frame['source_index']
        start, end = spans[index]
        c2w = np.asarray(frame['c2w'], float)
        record = {'sourceFrame': index, 'timeSec': start, 'endTimeSec': end,
                  'c2w': frame['c2w'], 'objects': []}
        observed = analysis.get(index) if analysis is not None else None
        if observed is not None and index in pairs:
            if abs(observed['timeSec'] - start) > .001 or abs(observed['endTimeSec'] - end) > .001:
                raise ValueError('Object and camera frame times disagree')
            path = Path(pairs[index]['depth_source_path'])
            if digest(path) != pairs[index]['depth_sha256'] or abs(pairs[index]['depth_timestamp'] - frame['timestamp']) >= .02:
                raise ValueError('Sensor depth changed or is not the camera-associated exposure')
            sensor = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            color = cv2.imread(frame['source_image'])
            if sensor.dtype != np.uint16 or sensor.shape != shape or color.shape[:2] != shape:
                raise ValueError('Unexpected RGB-D pixel domain')
            depth = sensor.astype(np.float32) / factor
            depth[(depth > 5) | (depth < .2)] = 0
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            validate_frame(color, depth, np.ones(shape), depth > 0, k, c2w)
            excluded = np.zeros(shape, bool)
            for obj in observed['objects']:
                mask_path = (args.analysis.parent / obj['maskUrl']).resolve()
                if not mask_path.is_relative_to(args.analysis.parent.resolve()):
                    raise ValueError('Mask must belong to the saved analysis directory')
                rgba = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if rgba is None or rgba.shape != (*shape, 4):
                    raise ValueError('Expected a source-domain RGBA mask')
                mask = rgba[:, :, 3] > 0
                excluded |= mask
                joints = surface_joints(obj.get('keypoints', []), depth, mask, k, c2w)
                valid = mask & (depth > 0)
                item = {'entityId': obj['entityId'], 'keypoints3d': joints,
                        'bones': [b for b in obj.get('bones', []) if joints[b[0]] is not None and joints[b[1]] is not None],
                        'representation': 'visible_depth_surface_skeleton', 'world_motion': 'not_classified'}
                if valid.any():
                    y, x = np.where(valid)
                    xyz = world_points(np.column_stack((x, y)), depth[valid], k, c2w)
                    item['centroid'] = np.median(xyz, axis=0).tolist()
                record['objects'].append(item)
            # Exclude a two-pixel boundary band from static fusion only.
            excluded = cv2.dilate(excluded.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            clean = np.where(excluded, 0, depth)
            if clean.any():
                integrate(volume, color, clean, k, c2w)
                fused += 1
            depth_records.append({'sourceFrame': index, 'path': str(path), 'sha256': digest(path),
                                  'excludedPixels': int(excluded.sum())})
        scene['frames'].append(record)
    if volume is not None:
        mesh = volume.extract_triangle_mesh()
        if mesh.is_empty():
            raise ValueError('No observed static surface was reconstructed')
        mesh.compute_vertex_normals()
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        colors = np.rint(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
        model = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
        model.export(output / 'static-scene.glb')
        reread = trimesh.load(output / 'static-scene.glb', force='mesh', process=False)
        if len(reread.faces) != len(faces) or not np.allclose(reread.bounds, model.bounds, atol=1e-6):
            raise ValueError('GLB geometry changed during export')
        scene['meshUrl'] = 'static-scene.glb'
        scene['provenance']['mesh_sha256'] = digest(output / 'static-scene.glb')
        (output / 'depth-inputs.json').write_text(json.dumps(depth_records, indent=2))
        scene['provenance']['depth_inputs_sha256'] = digest(output / 'depth-inputs.json')
    (output / 'scene.json').write_text(json.dumps(scene, ensure_ascii=False, allow_nan=False))
    metrics = {'elapsedSeconds': time.monotonic() - started, 'frames': len(scene['frames']),
               'fusedFrames': fused, 'scene_sha256': digest(output / 'scene.json')}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))


def self_check():
    k = np.array([[100, 0, 10], [0, 100, 10], [0, 0, 1.]])
    c2w = np.eye(4); c2w[:3, 3] = [3, 4, 5]
    assert np.allclose(world_points([[20, 10]], [2], k, c2w), [[3.2, 4, 7]])
    mask = np.ones((21, 21), bool); mask[1, 1] = False
    depth = np.full((21, 21), 2.)
    result = surface_joints([[10, 10, .8], [1, 1, .9], [10, 10, .1], None], depth, mask, k, c2w)
    assert np.allclose(result[0], [3, 4, 7]) and result[1:] == [None, None, None]
    depth[8:11, 8:11] = 3
    assert surface_joints([[10, 10, .9]], depth, mask, k, c2w) == [None]
    print('replay scene check passed: camera/world transform, mask support, missing joints and depth boundaries')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-scene', type=Path)
    p.add_argument('--video', type=Path)
    p.add_argument('--video-manifest', type=Path)
    p.add_argument('--analysis', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--self-check', action='store_true')
    args = p.parse_args()
    if args.self_check:
        self_check()
    elif args.native_scene and args.video and args.video_manifest and args.output:
        build(args)
    else:
        p.error('--native-scene, --video, --video-manifest and --output are required')
