"""Bounded official LingBot-Map RGB inference; sensor depth/GT never uploaded.

prepare writes a reviewable input plan. execute requires that plan and dispatches
one GPU attempt; collect only retrieves saved results, never resubmits inference.
prepare --video V --run-id R decodes the video on Modal (CPU) instead: the frames and
the predictions stay on the volume, and execute neither uploads frames nor collects.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

import modal

REV = '849e690bb086103637e44b1e91878d9d43a8bf0c'
WEIGHTS_REV = '204754b72bb24f561f8d7e7e1e4e4cd9e809adf9'
WEIGHTS_SHA = 'ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72'
app = modal.App('panoptes-lingbot-room-once')
volume = modal.Volume.from_name('panoptes-lingbot-map', create_if_missing=True)
attempts = modal.Dict.from_name('panoptes-lingbot-map-attempts', create_if_missing=True)
image = (modal.Image.from_registry('pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime').entrypoint([])
    .apt_install('git', 'libglib2.0-0', 'libgl1')
    .pip_install('huggingface_hub==0.34.4', 'einops==0.8.1', 'safetensors==0.6.2',
                 'opencv-python-headless==4.12.0.88', 'scipy==1.16.2')
    .run_commands('git init /opt/lingbot',
        'git -C /opt/lingbot remote add origin https://github.com/Robbyant/lingbot-map.git',
        'git -C /opt/lingbot sparse-checkout init --cone',
        'git -C /opt/lingbot sparse-checkout set lingbot_map',
        f'git -C /opt/lingbot fetch --depth=1 --filter=blob:none origin {REV}',
        'git -C /opt/lingbot checkout --detach FETCH_HEAD'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def require_disk_space(path, write_bytes=0):
    path = Path(path).resolve()
    while not path.exists(): path = path.parent
    free = shutil.disk_usage(path).free
    if free < 10 * 1024**3 + write_bytes:
        raise OSError(f'Only {free / 1024**3:.2f} GiB disk space available; keep 10 GiB free plus planned output. Stopped before the next write or GPU submission; existing files retained.')


def validate_indices(indices, count):
    if not indices or any(type(i) is not int or not 0 <= i < count for i in indices) or any(a >= b for a,b in zip(indices, indices[1:])):
        raise ValueError('Source frames must be distinct, chronological and inside video')
    return indices


def streaming_interval(frame_count):
    if type(frame_count) is not int or not 8 <= frame_count <= 768:
        raise ValueError('Bounded streaming requires 8–768 frames')
    # Official demo at REV: retain at most ~320 streaming keyframes.
    return (frame_count + 319) // 320


def infer_sequence(model, images, configuration, output_device):
    mode = configuration.get('mode', 'streaming')
    options = dict(num_scale_frames=configuration['anchor_frames'],
                   keyframe_interval=configuration['keyframe_interval'], output_device=output_device)
    if mode == 'streaming': return model.inference_streaming(images, **options)
    if mode != 'windowed': raise ValueError('Unknown official inference mode')
    size, overlap = configuration['window_size'], configuration['overlap_size']
    if type(size) is not int or type(overlap) is not int or not 8 <= overlap < size <= 64:
        raise ValueError('Bounded windows require 8 <= overlap < window <= 64')
    return model.inference_windowed(images, window_size=size, overlap_size=overlap,
                                    overlap_keyframes=None, **options)


def validate_prediction(depth, confidence, points, k, c2w):
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from reconstruct_room_rgb import validate_frame, pointmap_residuals
    if points.shape != (*depth.shape, 3): raise ValueError('Point map raster differs')
    valid = validate_frame(np.zeros((*depth.shape, 3), np.uint8), depth, confidence,
                           np.isfinite(points).all(-1), k, c2w)
    report = pointmap_residuals(points, depth, k, c2w, valid)
    if report['pointmap_camera_z_residual_p95'] > float(np.median(depth[valid]))*.1:
        raise ValueError('Point/depth heads or camera convention disagree')
    return report


def decode_frames(video, frames, stride, span=None):
    """PNG of every stride-th video frame into frames/; returns the plan records and the video's frame count.
    span (START, END): only frames START..END-1, one continuous shot of an edited video."""
    import cv2
    cap = cv2.VideoCapture(str(video)); records=[]; index=0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok: break
            if index % stride == 0 and (span is None or span[0] <= index < span[1]):
                if len(records) >= 768: raise ValueError('Bounded experiment supports at most 768 input frames, below the 1024-frame positional limit')
                require_disk_space(frames, bgr.nbytes + 131072)
                path=frames/f'{index:06d}.png'; assert cv2.imwrite(str(path), bgr)
                records.append({'sourceFrame':index, 'timeSec':cap.get(cv2.CAP_PROP_POS_MSEC)/1000,
                                'path':path.name, 'sha256':digest(path)})
            index += 1
    finally: cap.release()
    validate_indices([r['sourceFrame'] for r in records],index)
    if len(records)<8 or any(a['timeSec']>=b['timeSec'] for a,b in zip(records,records[1:])):
        raise ValueError('Need at least 8 source frames with increasing media timestamps')
    return records, index


def valid_run_id(run_id):
    if not run_id or not all(c.isalnum() or c in '-_' for c in run_id):raise ValueError('Invalid run ID')
    return run_id


@app.function(image=image, cpu=(4,4), memory=(8192,8192), timeout=900, retries=0, volumes={'/artifact':volume})
def decode_remote(run_id, video_sha, stride, span=None):
    """CPU only: the run's uploaded source.mp4 -> rgb.tar beside it, the same records prepare writes locally."""
    volume.reload(); root=Path('/artifact')/valid_run_id(run_id)
    if digest(root/'source.mp4')!=video_sha:raise ValueError('Uploaded video differs from the local source')
    frames=Path('/tmp/rgb');frames.mkdir()
    records,count=decode_frames(root/'source.mp4',frames,stride,span)
    with tarfile.open(root/'rgb.tar','w') as archive:
        for record in records:archive.add(frames/record['path'],arcname=record['path'])
    volume.commit()
    return records,count,digest(root/'rgb.tar')


def prepare(manifest_path, sample_id, output, stride, video=None, run_id=None, span=None):
    if video is None:
        manifest = json.loads(manifest_path.read_text())
        sample = next(s for s in manifest['samples'] if s['id'] == sample_id)
        video = (manifest_path.parent/sample['video']['url']).resolve()
        if digest(video) != sample['video']['sha256']: raise ValueError('Source video changed')
    video = Path(video).resolve()
    if type(stride) is not int or stride < 1: raise ValueError('Positive integer stride required')
    require_disk_space(output)
    output.mkdir(parents=True, exist_ok=False)
    if run_id is None:
        frames = output/'rgb'; frames.mkdir()
        records, index = decode_frames(video, frames, stride, span)
        require_disk_space(output, sum((frames / r['path']).stat().st_size + 4096 for r in records))
        with tarfile.open(output/'rgb.tar','w') as archive:
            for record in records: archive.add(frames/record['path'],arcname=record['path'])
        archive_sha = digest(output/'rgb.tar')
    else:  # frames of a long full-resolution video would be GBs locally; decode them next to the GPU instead
        with volume.batch_upload() as batch: batch.put_file(video, f'{valid_run_id(run_id)}/source.mp4')
        with app.run(): records, index, archive_sha = decode_remote.remote(run_id, digest(video), stride, span)
        validate_indices([r['sourceFrame'] for r in records], index)
    plan={'source_video':str(video), 'source_video_sha256':digest(video), 'source_frame_count':index,
          'frames':records, 'stride':stride, **({'shot_frames':list(span)} if span else {}), 'code_revision':REV,'weights_revision':WEIGHTS_REV,
          'weights_sha256':WEIGHTS_SHA,'rgb_archive_sha256':archive_sha,'rgb_archive_on_volume':run_id,
          'sensor_depth_uploaded':False,'groundtruth_uploaded':False,'new_training':False,
          'max_gpu_seconds':900,'max_gpu_attempts':1,'reserved_usd':2,
          'configuration':{'mode':'streaming','image_size':518,'patch_size':14,'window':64,'anchor_frames':8,
                           'camera_iterations':4,'backend':'sdpa','keyframe_interval':streaming_interval(len(records))}}
    save(output/'plan.json',plan); print(json.dumps({k:v for k,v in plan.items() if k!='frames'}),flush=True)


@app.function(image=image, gpu=['A100-80GB', 'H100', 'A100-40GB'], cpu=(4,4), memory=(32768,32768),
              timeout=900, startup_timeout=60, retries=0, max_containers=1,
              min_containers=0, scaledown_window=2, volumes={'/artifact':volume})
def infer(run_id, deadline, plan_sha):
    if not attempts.put(run_id, {'claimed_at':time.time()},skip_if_exists=True):
        raise RuntimeError('This GPU attempt was already claimed; collect without resubmitting')
    import signal
    def expired(*_): raise TimeoutError('Bounded LingBot inference deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1,min(880,int(deadline-time.time())-10)))
    started=time.time(); volume.reload(); root=Path('/artifact')/run_id; out=root/'result';out.mkdir(exist_ok=False)
    save(out/'run.json',{'status':'running','run_id':run_id});volume.commit()
    try:
        assert digest(root/'plan.json')==plan_sha
        plan=json.loads((root/'plan.json').read_text());assert plan['code_revision']==REV and plan['weights_sha256']==WEIGHTS_SHA
        interval=plan['configuration']['keyframe_interval']
        mode=plan['configuration'].get('mode','streaming')
        if mode not in ('streaming','windowed'): raise ValueError('Unknown inference mode')
        if type(interval) is not int or not 1 <= interval <= streaming_interval(len(plan['frames'])):
            raise ValueError('Invalid declared streaming keyframe interval')
        assert not plan['sensor_depth_uploaded'] and not plan['groundtruth_uploaded']
        assert digest(root/'rgb.tar')==plan['rgb_archive_sha256']
        import numpy as np
        import torch
        from huggingface_hub import hf_hub_download
        weight=hf_hub_download('robbyant/lingbot-map','lingbot-map.pt',revision=WEIGHTS_REV,cache_dir='/artifact/hf')
        assert digest(weight)==WEIGHTS_SHA;volume.commit()
        rgb=Path('/tmp/lingbot-rgb');rgb.mkdir()
        with tarfile.open(root/'rgb.tar') as archive:
            expected={r['path']:r['sha256'] for r in plan['frames']};members=archive.getmembers()
            assert len(members)==len(expected) and {m.name for m in members}==set(expected)
            for member in members:
                assert member.isfile() and Path(member.name).name==member.name
                p=rgb/member.name;p.write_bytes(archive.extractfile(member).read());assert digest(p)==expected[member.name]
        sys.path.insert(0,'/opt/lingbot')
        from demo import load_model, postprocess, prepare_for_visualization
        from lingbot_map.utils.load_fn import load_and_preprocess_images
        from types import SimpleNamespace
        args=SimpleNamespace(mode=mode,image_size=518,patch_size=14,enable_3d_rope=True,
            max_frame_num=1024,kv_cache_sliding_window=64,num_scale_frames=8,use_sdpa=True,
            camera_num_iterations=4,model_path=weight)
        model=load_model(args,'cuda');model.aggregator=model.aggregator.to(dtype=torch.bfloat16);model.eval()
        images=load_and_preprocess_images([str(rgb/r['path']) for r in plan['frames']],mode='crop',image_size=518,patch_size=14)
        images=images.to('cuda');torch.cuda.reset_peak_memory_stats();begin=time.time()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            prediction=infer_sequence(model,images,plan['configuration'],torch.device('cpu'))
        torch.cuda.synchronize();infer_seconds=time.time()-begin;memory=torch.cuda.max_memory_allocated()
        # Preserve native outputs before any adapter work; an exporter failure
        # must not require another paid forward pass.
        raw=out/'native-prediction.pt'
        torch.save({k:prediction[k] for k in ('depth','depth_conf','pose_enc','images','chunk_scales','chunk_transforms','alignment_mode') if k in prediction},raw)
        save(out/'native-prediction.json',{'keys':list(prediction),'sha256':digest(raw),
            'inference_seconds':infer_seconds,'peak_gpu_bytes':memory})
        volume.commit()
        prediction,colors=postprocess(prediction,prediction['images']);data=prepare_for_visualization(prediction,colors)
        files=[]
        for i,record in enumerate(plan['frames']):
            p=out/f"frame-{record['sourceFrame']:06d}.npz"
            np.savez_compressed(p,depth=data['depth'][i].astype('float32'),depth_conf=data['depth_conf'][i].astype('float32'),
                w2c=data['extrinsic'][i].astype('float32'),k=data['intrinsic'][i].astype('float32'),
                rgb=np.rint(np.moveaxis(data['images'][i],0,-1)*255).astype('uint8'))
            files.append({'sourceFrame':record['sourceFrame'],'file':p.name,'sha256':digest(p)})
        result={'status':'inference_complete','run_id':run_id,'code_revision':REV,'weights_sha256':WEIGHTS_SHA,
                'plan_sha256':plan_sha,'configuration':plan['configuration'],'frames':files,
                'inference_seconds':infer_seconds,'peak_gpu_bytes':memory,'elapsed_seconds':time.time()-started,
                'gpu':torch.cuda.get_device_name(),
                'geometry_contract':'native camera z-depth + predicted K + official demo extrinsic W2C; invert once at map ingestion, no point head',
                'native_prediction_sha256':digest(raw),
                'source_sha256':{str(p.relative_to('/opt/lingbot')):digest(p) for p in Path('/opt/lingbot/lingbot_map').rglob('*.py')},
                'sensor_depth_uploaded':False,'groundtruth_uploaded':False,'new_training':False}
        save(out/'run.json',result);volume.commit();return {k:v for k,v in result.items() if k not in ('frames','source_sha256')}
    except BaseException as error:
        save(out/'run.json',{'status':'failed','run_id':run_id,'type':type(error).__name__,'message':str(error)[:600],
                            'elapsed_seconds':time.time()-started});volume.commit();raise
    finally: signal.alarm(0)


def collect(output, run_id):
    plan=json.loads((output/'plan.json').read_text())
    require_disk_space(output, len(plan['frames']) * 518 * 518 * 12)
    remote=f'{run_id}/result'
    for entry in volume.iterdir(remote,recursive=False):
        name=Path(entry.path).name
        if name=='native-prediction.pt':continue
        require_disk_space(output, 518 * 518 * 12)
        with (output/name).open('wb') as stream:
            for block in volume.read_file(remote+'/'+name):stream.write(block)
    report=json.loads((output/'run.json').read_text())
    if report['status']=='inference_complete':
        assert report['plan_sha256']==digest(output/'plan.json')
        for f in report['frames']:assert digest(output/f['file'])==f['sha256']
    return {k:v for k,v in report.items() if k not in ('frames','source_sha256')}


def execute(output, run_id):
    valid_run_id(run_id)
    plan=json.loads((output/'plan.json').read_text())
    remote=plan.get('rgb_archive_on_volume')  # frames already on the volume (infer re-checks their SHA)
    if remote not in (None,run_id):raise ValueError('Plan frames were decoded for another run ID')
    require_disk_space(output, len(plan['frames']) * 518 * 518 * 12)
    if not remote:assert digest(output/'rgb.tar')==plan['rgb_archive_sha256']
    assert digest(plan['source_video'])==plan['source_video_sha256']
    if (output/'submission.json').exists():raise ValueError('Submission exists; use collect')
    (output/'runner-at-execution.py').write_bytes(Path(__file__).read_bytes())
    with volume.batch_upload() as batch:
        for name in ['plan.json']+([] if remote else ['rgb.tar']):batch.put_file(output/name,f'{run_id}/{name}')
    with app.run():
        deadline=time.time()+940
        save(output/'submission.json',{'run_id':run_id,'status':'submitting','app_id':app.app_id,'deadline':deadline})
        call=infer.spawn(run_id,deadline,digest(output/'plan.json'))
        save(output/'submission.json',{'run_id':run_id,'status':'submitted','app_id':app.app_id,'call_id':call.object_id,'deadline':deadline})
        try:print(json.dumps(call.get(timeout=max(1,deadline-time.time()))),flush=True)
        except BaseException:
            call.cancel(terminate_containers=True);raise
        if remote:print(f'Predictions stay on volume panoptes-lingbot-map:/{run_id}/result; collect them explicitly if they fit locally',flush=True)
        else:print(json.dumps(collect(output,run_id)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','execute','collect'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path);p.add_argument('--sample');p.add_argument('--stride',type=int,default=3);p.add_argument('--run-id')
    p.add_argument('--video',type=Path,help='prepare from this video instead of --manifest/--sample; needs --run-id (decoded on Modal)')
    p.add_argument('--frames',type=lambda s:tuple(int(v) for v in s.split(':')),metavar='START:END',help='prepare: only source frames START..END-1 (one continuous shot)')
    a=p.parse_args()
    if a.mode=='prepare':
        if a.video and not a.run_id:p.error('--video decodes on the volume and needs --run-id')
        prepare(a.manifest,a.sample,a.output,a.stride,a.video,a.run_id if a.video else None,a.frames)
    elif a.mode=='execute':execute(a.output,a.run_id)
    else:print(json.dumps(collect(a.output,a.run_id)),flush=True)
