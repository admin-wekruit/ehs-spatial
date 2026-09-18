"""Same-input official-cache comparison; raw RGB/predictions stay on Modal.

Run with --source-run, --source-plan-sha, --run-id and --output.
--review-only never launches GPU inference. Collection is capped at 32 MiB and
retains 2 GiB local headroom; the full-result collector keeps its 10 GiB guard.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lingbot_room import app, volume, image, infer, digest, save, streaming_interval

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
review_image = image.pip_install('trimesh==5.1.0')
image = image.add_local_file(Path(__file__).with_name('lingbot_room.py'), '/root/lingbot_room.py')
review_image = review_image.add_local_file(Path(__file__).with_name('lingbot_room.py'), '/root/lingbot_room.py')
for name in ['build_lingbot_replay.py', 'build_replay_scene.py', 'reconstruct_room_rgb.py',
             'reconstruct_tum_room.py', 'video_motion.py']:
    review_image = review_image.add_local_file(SCRIPTS / name, '/review/' + name)


def rectify_raster(raster, calibration, mask=False):
    import cv2
    import numpy as np
    if calibration is None: return raster
    k = np.asarray(calibration['K'], dtype=float)
    d = np.asarray(calibration['distortion'], dtype=float)
    wh = tuple(calibration['source_wh'])
    if (wh != raster.shape[1::-1] or k.shape != (3, 3) or d.shape != (5,)
            or not np.isfinite(k).all() or not np.isfinite(d).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.array_equal(k[2], [0, 0, 1])):
        raise ValueError('Calibration must match the source raster and pinhole camera')
    if not mask: return cv2.undistort(raster, k, d, None, k)
    maps = cv2.initUndistortRectifyMap(k, d, None, k, wh, cv2.CV_32FC1)
    return cv2.remap(raster, *maps, cv2.INTER_NEAREST)


@app.function(image=image, cpu=1, memory=1024, timeout=180, retries=0,
              volumes={'/artifact': volume})
def prepare(source_run, run_id, expected_sha, calibration=None, windowed=False):
    volume.reload()
    for name in (source_run, run_id):
        if not name or not all(c.isalnum() or c in '-_' for c in name):
            raise ValueError('Invalid run ID')
    source, target = Path('/artifact') / source_run, Path('/artifact') / run_id
    assert digest(source / 'plan.json') == expected_sha
    plan = json.loads((source / 'plan.json').read_text())
    assert 'rgb_rectification' not in plan, 'Do not rectify already rectified inputs'
    if windowed:
        assert calibration is None, 'One inference change per comparison'
        assert plan['configuration']['keyframe_interval'] == 1
        assert plan['configuration'].get('mode', 'streaming') == 'streaming'
        plan['configuration'].update(mode='windowed', window_size=64, overlap_size=16)
    elif calibration is None:
        assert plan['configuration']['keyframe_interval'] == 1
        plan['configuration']['keyframe_interval'] = streaming_interval(len(plan['frames']))
        assert plan['configuration']['keyframe_interval'] > 1, 'No parameter change to compare'
    plan['comparison'] = {'parent_run': source_run, 'parent_plan_sha256': expected_sha,
                          'only_inference_change': 'official_windowed_mode' if windowed else 'calibrated_rgb_rectification' if calibration is not None else 'keyframe_interval'}
    target.mkdir(exist_ok=False)
    assert digest(source / 'rgb.tar') == plan['rgb_archive_sha256']
    if calibration is not None:
        import cv2
        import numpy as np
        import io
        with tarfile.open(source / 'rgb.tar') as original, tarfile.open(target / 'rgb.tar', 'w') as corrected:
            for record in plan['frames']:
                raw = original.extractfile(record['path']).read()
                import hashlib
                assert hashlib.sha256(raw).hexdigest() == record['sha256']
                raster = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                ok, encoded = cv2.imencode('.png', rectify_raster(raster, calibration))
                assert ok
                raw = encoded.tobytes()
                record['unrectified_sha256'] = record['sha256']
                record['sha256'] = hashlib.sha256(raw).hexdigest()
                info = tarfile.TarInfo(record['path']); info.size = len(raw)
                corrected.addfile(info, io.BytesIO(raw))
        plan['rgb_rectification'] = calibration
        plan['rgb_archive_sha256'] = digest(target / 'rgb.tar')
    else:
        shutil.copyfile(source / 'rgb.tar', target / 'rgb.tar')
    save(target / 'plan.json', plan)
    volume.commit()
    return {'plan': plan, 'plan_sha256': digest(target / 'plan.json')}


@app.function(image=review_image, cpu=4, memory=8192, timeout=600, retries=0,
              volumes={'/artifact': volume})
def review(run_id, analysis):
    import numpy as np
    import trimesh
    sys.path.insert(0, '/review')
    sys.path.insert(0, '/opt/lingbot')
    from build_lingbot_replay import read_prediction, check_temporal_geometry
    from build_replay_scene import world_points
    from lingbot_map.utils.geometry import unproject_depth_map_to_point_map
    volume.reload()
    root = Path('/artifact') / run_id
    run = root / 'result'
    plan = json.loads((root / 'plan.json').read_text())
    execution = json.loads((run / 'run.json').read_text())
    assert execution['status'] == 'inference_complete'
    assert execution['plan_sha256'] == digest(root / 'plan.json')
    assert analysis['provenance']['sourceVideoSha256'] == plan['source_video_sha256']
    # Masks arrive as the already processed source-domain PNGs, not new inference.
    mask_root = Path('/tmp/review-masks'); mask_root.mkdir(exist_ok=True)
    import base64
    for frame in analysis['frames']:
        for obj in frame['objects']:
            raw = base64.b64decode(obj.pop('maskPngBase64'), validate=True)
            name = Path(obj['maskUrl']).name
            assert name == obj['maskUrl'], 'Masks must have flat relative paths'
            if plan.get('rgb_rectification'):
                import cv2
                raster = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                ok, encoded = cv2.imencode('.png', rectify_raster(raster, plan['rgb_rectification'], mask=True))
                assert ok
                raw = encoded.tobytes()
            (mask_root / name).write_bytes(raw)
    temporal = check_temporal_geometry(run, {**analysis, '_base': str(mask_root)})
    clouds, colors, frames = [], [], []
    worst_residual = 0.0
    for i, (record, source) in enumerate(zip(execution['frames'], plan['frames'], strict=True)):
        path = run / record['file']
        assert record['sourceFrame'] == source['sourceFrame'] and digest(path) == record['sha256']
        depth, confidence, rgb, k, camera, valid = read_prediction(path)
        w2c = np.linalg.inv(camera)[:3]
        xyz = unproject_depth_map_to_point_map(depth[None, ..., None], w2c[None], k[None])[0]
        yy, xx = np.indices(depth.shape)
        ours = world_points(np.column_stack((xx.ravel(), yy.ravel())), depth.ravel(), k, camera).reshape(xyz.shape)
        residual = float(np.max(np.abs(xyz[valid] - ours[valid])))
        assert residual <= max(1e-5, float(np.max(np.abs(xyz[valid]))) * 1e-5)
        worst_residual = max(worst_residual, residual)
        # Exactly the upstream viewer's confidence filter and per-frame flat stride.
        keep = valid & (confidence > 1.5)
        # Keep only sampled buffers; a stride view retains the full masked array.
        clouds.append(xyz[keep][::10].copy()); colors.append(rgb[keep][::10].copy())
        frames.append({'sourceFrame': source['sourceFrame'], 'timeSec': source['timeSec'],
                       'endTimeSec': plan['frames'][i + 1]['timeSec'] if i + 1 < len(plan['frames']) else analysis['frames'][-1]['endTimeSec'],
                       'c2w': camera.tolist(), 'objects': []})
    xyz, rgb = np.concatenate(clouds), np.concatenate(colors)
    assert len(xyz) > 0
    stride = max(1, (len(xyz) + 299999) // 300000)
    displayed, display_rgb = xyz[::stride], rgb[::stride]
    output = root / 'preview'; output.mkdir(exist_ok=True)
    asset = output / 'native-cloud.glb'
    trimesh.Scene(trimesh.points.PointCloud(displayed, colors=display_rgb)).export(asset)
    summary = {'configuration': plan['configuration'], 'inference_seconds': execution['inference_seconds'],
               'rgb_rectification': plan.get('rgb_rectification'),
               'official_viewer_points': len(xyz), 'display_points': len(displayed), 'display_stride': stride,
               'official_vs_adapter_max_xyz_residual': worst_residual,
               'temporal_check': temporal, 'complete_room_accepted': False}
    scene = {'schema': 'phase2-replay-scene-v1', 'coordinate_frame': 'lingbot_native_monocular',
             'units': 'uncalibrated_monocular', 'source_video_sha256': plan['source_video_sha256'],
             'method': f"LingBot {plan['configuration'].get('mode', 'streaming')} with declared input plan + official depth unprojection; diagnostic preview",
             'complete_room_accepted': False, 'pointCloudUrl': asset.name, 'pointCloudCount': len(displayed),
             'points': [[i, *p.tolist()] for i, p in enumerate(displayed[::max(1, len(displayed)//20000)])],
             'frames': frames, 'provenance': {'plan_sha256': execution['plan_sha256'],
             'execution_sha256': digest(run / 'run.json'), 'cloud_sha256': digest(asset), **summary},
             'limitations': ['原生预测诊断对照；缓存间隔与RGB校正记录在来源中，仍非完整房间验收。',
                 '按官方置信度和采样规则生成，再均匀抽取最多30万点供浏览器查看；原始结果留在云端。',
                 '此预览未做动态对象排除或静态融合，不能当作完整实体或现场量测。']}
    save(output / 'scene.json', scene); save(output / 'review.json', summary)
    save(output / 'plan.json', plan); save(output / 'run.json', execution)
    files = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in output.iterdir() if p.is_file()]
    assert sum(f['bytes'] for f in files) <= 32 * 1024**2
    volume.commit()
    return {'summary': summary, 'files': files}


def collect_preview(run_id, output, result, artifact_volume=None):
    artifact_volume = volume if artifact_volume is None else artifact_volume
    total = sum(f['bytes'] for f in result['files'])
    if total > 32 * 1024**2 or shutil.disk_usage(output).free < 2 * 1024**3 + total:
        raise OSError('Compact preview exceeds 32 MiB or would leave less than 2 GiB free')
    for record in result['files']:
        name = record['name']
        if Path(name).name != name: raise ValueError('Invalid preview path')
        path = output / name
        with path.open('xb') as stream:
            for block in artifact_volume.read_file(f'{run_id}/preview/{name}'):
                if stream.tell() + len(block) > record['bytes']: raise ValueError('Preview exceeded declared size')
                stream.write(block)
        assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256']


def main(a):
    import base64
    analysis = json.loads(a.analysis.read_text())
    for frame in analysis['frames']:
        for obj in frame['objects']:
            source = (a.analysis.parent / obj['maskUrl']).resolve()
            assert source.is_relative_to(a.analysis.parent.resolve())
            obj['maskUrl'] = source.name
            obj['maskPngBase64'] = base64.b64encode(source.read_bytes()).decode()
    if not a.review_only:
        a.output.mkdir(parents=True, exist_ok=False)
    with app.run():
        if not a.review_only:
            calibration = json.loads(a.calibration.read_text()) if a.calibration else None
            prepared = prepare.remote(a.source_run, a.run_id, a.source_plan_sha, calibration, a.windowed)
            save(a.output / 'input-plan.json', prepared)
            deadline = time.time() + 940
            call = infer.spawn(a.run_id, deadline, prepared['plan_sha256'])
            save(a.output / 'submission.json', {'run_id': a.run_id, 'call_id': call.object_id, 'app_id': app.app_id})
            try:
                print(json.dumps(call.get(timeout=940)), flush=True)
            except BaseException:
                call.cancel(terminate_containers=True)
                raise
        result = review.remote(a.run_id, analysis)
        save(a.output / 'collection.json', result)
        collect_preview(a.run_id, a.output, result)
        print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source-run', 'source-plan-sha', 'run-id']: p.add_argument('--' + name, required=True)
    for name in ['output', 'analysis']: p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--review-only', action='store_true')
    p.add_argument('--calibration', type=Path, help='Calibrated RGB-only rectification comparison; K, distortion, source_wh')
    p.add_argument('--windowed', action='store_true', help='Official 64-frame windows with 16-frame overlap; compare against interval1 streaming')
    main(p.parse_args())
