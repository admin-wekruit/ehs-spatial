"""Propagate saved SAM3 image instances with cached SAM2.1 native video memory.

No discovery calls, manually drawn prompts, second tracker, or guessed identities.
The seed lineage and clip manifest must refer to the same original media frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehs_spatial.providers.sam3 import decode_coco_rle, encode_coco_rle
from build_video_pose_preview import intervals, sha, source_spans


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def decoded_clip(path, destination):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Cannot decode input clip')
    records, shape = [], None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if shape is not None and bgr.shape != shape:
                raise ValueError('Video changes pixel dimensions')
            shape = bgr.shape
            index = len(records)
            path_jpg = destination / f'{index:05d}.jpg'
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(path_jpg, quality=100, subsampling=0)
            records.append({'frame_index': index,
                            'timestamp_seconds': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                            'decoded_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                            'jpeg_sha256': sha(path_jpg)})
    finally:
        cap.release()
    if not records or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Clip has no frames or valid frame rate')
    # ponytail: this entry accepts CFR clips; existing source_spans rejects retiming.
    times = [r['timestamp_seconds'] for r in records]
    if len(times) > 1 and not np.allclose(np.diff(times), 1 / fps, atol=.0001, rtol=0):
        raise ValueError('Expected CFR clip; preserve source PTS when preparing this input')
    return {'sha256': sha(path), 'frame_count': len(records), 'width': shape[1],
            'height': shape[0], 'duration_seconds': times[-1] + 1 / fps,
            'frame_timestamps_seconds': times}, records


def seed_frame_index(observed, lineage, parent_manifest, clip):
    if lineage['clip_sha256'] != observed['source_clip_sha256'] or lineage['source_sha256'] != parent_manifest['sourceSha256']:
        raise ValueError('Seed and target do not share the declared source video')
    local = observed['source_frame_index']
    original = lineage['source_frame_indices'][local]
    target = original - parent_manifest['sourceStartFrame']
    if not 0 <= target < clip['frame_count']:
        raise ValueError('Seed frame is outside the target clip')
    if abs(lineage['source_timestamps_seconds'][local] - parent_manifest['sourceFrameMediaTimesSeconds'][target]) > .001:
        raise ValueError('Seed and target reference different media times')
    if (observed['height'], observed['width']) != (clip['height'], clip['width']):
        raise ValueError('Seed and target pixel domains differ')
    return target


def seed_masks(folder, lineage_path, parent_manifest, clip):
    observed = json.loads((folder / 'input-manifest.json').read_text())
    lineage = json.loads(lineage_path.read_text())
    seed_clip = Path(observed['source_clip'])
    seed_frame = folder / f"frame-{observed['source_frame_index']}.png"
    if sha(seed_clip) != observed['source_clip_sha256'] or sha(seed_frame) != observed['frame_sha256']:
        raise ValueError('Seed source clip or source PNG changed')
    target = seed_frame_index(observed, lineage, parent_manifest, clip)
    local = observed['source_frame_index']
    original = lineage['source_frame_indices'][local]
    cap = cv2.VideoCapture(str(seed_clip))
    try:
        for _ in range(local + 1):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError('Seed source frame is missing')
    finally:
        cap.release()
    if bgr.shape[:2] != (clip['height'], clip['width']) or not np.array_equal(cv2.imread(str(seed_frame)), bgr):
        raise ValueError('Seed PNG is not the declared decoded source frame')
    provider = json.loads((folder / 'provider-output.json').read_text())
    instances = json.loads((folder / 'instances.json').read_text())
    label = observed['prompt']
    if not isinstance(label, str) or not label.strip() or any(i.get('label') != label for i in instances):
        raise ValueError('Seed labels must match the saved discovery prompt')
    seeds = []
    ids = [i['instance_index'] for i in instances]
    if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Expected distinct provider instance indices')
    for instance in instances:
        index = instance['instance_index']
        mask_path = folder / f'instance-{index}-mask.png'
        mask = np.array(Image.open(mask_path).convert('L')) > 0
        native = decode_coco_rle(provider['rle'][index], height=clip['height'], width=clip['width']).astype(bool)
        if mask.shape != (clip['height'], clip['width']) or not mask.any() or not np.array_equal(mask, native):
            raise ValueError('Seed PNG differs from its provider mask or pixel domain')
        seeds.append({'track_id': index, 'frame_index': target, 'mask': mask,
                      'provenance': {'instance_index': index, 'mask_sha256': sha(mask_path),
                                     'source_frame_index': original, 'source_png_sha256': sha(seed_frame)}})
    return seeds, {'prompt': label, 'source_manifest_sha256': sha(lineage_path),
                   'discovery_manifest_sha256': sha(folder / 'input-manifest.json'),
                   'provider_output_sha256': sha(folder / 'provider-output.json'),
                   'seed_clip_sha256': sha(seed_clip), 'target_clip_sha256': clip['sha256'],
                   'pixel_domain': 'same source frame and dimensions; RGB reencoding is explicit',
                   'instances': [s['provenance'] for s in seeds]}


def encode_output(index, native_ids, logits, clip, source_start, expected_ids, label='person'):
    if not isinstance(label, str) or not label.strip():
        raise ValueError('A nonempty discovery label is required')
    ids = list(native_ids)
    values = np.asarray(logits)
    if ids != list(expected_ids) or len(ids) != len(set(ids)):
        raise ValueError('Native predictor changed seed identity order')
    if values.shape != (len(ids), 1, clip['height'], clip['width']) or not np.isfinite(values).all():
        raise ValueError('Native mask logits have invalid dimensions or values')
    objects = []
    for native_id, mask in zip(ids, values[:, 0] > 0, strict=True):
        y, x = np.nonzero(mask)
        box = ([float(x.min() / clip['width']), float(y.min() / clip['height']),
                float((x.max() - x.min() + 1) / clip['width']),
                float((y.max() - y.min() + 1) / clip['height'])] if len(x) else [0, 0, 0, 0])
        rle = encode_coco_rle(mask)
        if not np.array_equal(decode_coco_rle(rle).astype(bool), mask):
            raise ValueError('RLE changed the native mask')
        objects.append({'track_id': native_id, 'label': label, 'score': None,
                        'rle': rle, 'box_xywh_normalized': box, 'mask_area_pixels': int(mask.sum())})
    return {'frame_index': index, 'source_frame_index': source_start + index,
            'timestamp_seconds': clip['frame_timestamps_seconds'][index], 'objects': objects}


def run(args):
    import torch

    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'frames').mkdir()
    clip, decoded = decoded_clip(args.clip, args.output / 'frames')
    parent = json.loads(args.source_manifest.read_text())
    skeleton = {'input': clip, 'frames': [
        {'frame_index': i, 'source_frame_index': parent['sourceStartFrame'] + i,
         'timestamp_seconds': t} for i, t in enumerate(clip['frame_timestamps_seconds'])]}
    source_spans(skeleton, parent)
    seeds, evidence = seed_masks(args.seed_dir, args.seed_lineage, parent, clip)
    if any(s['frame_index'] != 0 for s in seeds):
        raise ValueError('This bounded run requires discovered seeds on its first frame')
    revision = subprocess.check_output(['git', '-C', str(args.vendor), 'rev-parse', 'HEAD'], text=True).strip()
    sys.path.insert(0, str(args.vendor))
    from sam2.build_sam import build_sam2_video_predictor
    method = {'name': 'SAM3 image seeds + SAM2.1 native video memory',
              'script_sha256': sha(Path(__file__)),
              'source_revision': revision, 'weights_sha256': sha(args.weights),
              'device': args.device, 'torch_version': torch.__version__,
              'mps_cpu_fallback_environment': os.getenv('PYTORCH_ENABLE_MPS_FALLBACK'),
              'mask_logit_threshold': 0, 'postprocessing': False,
              'offload_video_to_cpu': True, 'offload_state_to_cpu': False,
              'seed_provenance': evidence, 'native_id_origin': 'SAM3 instance index supplied to SAM2 add_new_mask'}
    save(args.output / 'input-decoding.json', {'clip': clip, 'frames': decoded,
         'source_manifest_sha256': sha(args.source_manifest),
         'encoding': 'OpenCV decoded BGR -> RGB -> JPEG quality100 subsampling0; no resizing or resampling'})
    save(args.output / 'source-manifest.json', parent)
    raw = {'session_id': str(uuid.uuid4()), 'identity_scope': 'video_session_only',
           'method': method, 'input': clip, 'frames': []}
    save(args.output / 'run.json', {'status': 'running', 'session_id': raw['session_id'], 'method': method})
    model = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml', str(args.weights),
                                      device=args.device, apply_postprocessing=False, vos_optimized=False)
    with torch.inference_mode():
        state = model.init_state(str(args.output / 'frames'), offload_video_to_cpu=True,
                                 offload_state_to_cpu=False)
        for seed in seeds:
            model.add_new_mask(state, frame_idx=seed['frame_index'], obj_id=seed['track_id'], mask=seed['mask'])
        expected_ids = [s['track_id'] for s in seeds]
        with (args.output / 'frames.jsonl').open('w') as ledger:
            for index, ids, logits in model.propagate_in_video(state):
                frame = encode_output(index, ids, logits.float().cpu().numpy(), clip,
                                      parent['sourceStartFrame'], expected_ids, evidence['prompt'])
                if index != len(raw['frames']):
                    raise ValueError('Native predictor omitted or reordered a source frame')
                raw['frames'].append(frame)
                ledger.write(json.dumps(frame, allow_nan=False) + '\n')
                ledger.flush()
                if index % 8 == 0:
                    print(json.dumps({'frame': index, 'elapsed_seconds': round(time.monotonic() - start, 2),
                                      'areas': [o['mask_area_pixels'] for o in frame['objects']]}), flush=True)
    intervals(raw)
    save(args.output / 'observations.json', raw)
    result = {'status': 'execution_complete', 'frames': len(raw['frames']),
              'native_ids': expected_ids, 'elapsed_seconds': time.monotonic() - start,
              'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
              'mps_driver_allocated_bytes_at_end': torch.mps.driver_allocated_memory() if args.device == 'mps' else None,
              'quality': 'requires_visual_review', 'raw_sha256': sha(args.output / 'observations.json')}
    save(args.output / 'run.json', result)
    return result


def self_check():
    clip = {'width': 3, 'height': 2, 'frame_count': 2,
            'frame_timestamps_seconds': [0, .1], 'duration_seconds': .2}
    logits = np.full((2, 1, 2, 3), -1., np.float32)
    logits[0, 0, 0, 1] = 2
    logits[0, 0, 1, 1] = 0  # Native threshold is strictly > 0, not >= 0.
    frame = encode_output(0, [9, 2], logits, clip, 7, [9, 2])
    assert [o['track_id'] for o in frame['objects']] == [9, 2]
    assert [o['mask_area_pixels'] for o in frame['objects']] == [1, 0]
    assert all(o['score'] is None for o in frame['objects'])
    assert frame['source_frame_index'] == 7
    vehicle = encode_output(0, [9, 2], logits, clip, 7, [9, 2], 'car')
    assert all(o['label'] == 'car' for o in vehicle['objects'])
    second = encode_output(1, [9, 2], logits, clip, 7, [9, 2])
    assert intervals({'input': clip, 'frames': [frame, second]}) == [(0, .1), (.1, .2)]
    for ids, values in [([2, 9], logits), ([9, 2], logits[:, :, :, :2])]:
        try:
            encode_output(0, ids, values, clip, 0, [9, 2])
        except ValueError:
            pass
        else:
            raise AssertionError('Wrong identity or pixel domain accepted')
    observed = {'source_clip_sha256': 'seed', 'source_frame_index': 0, 'height': 2, 'width': 3}
    lineage = {'clip_sha256': 'seed', 'source_sha256': 'parent',
               'source_frame_indices': [7], 'source_timestamps_seconds': [.7]}
    parent = {'sourceSha256': 'parent', 'sourceStartFrame': 7,
              'sourceFrameMediaTimesSeconds': [.7, .8, .9]}
    assert seed_frame_index(observed, lineage, parent, clip) == 0
    for changed in [observed | {'width': 4}, observed | {'source_clip_sha256': 'other'}]:
        try:
            seed_frame_index(changed, lineage, parent, clip)
        except ValueError:
            pass
        else:
            raise AssertionError('Unrelated seed source or pixel domain accepted')
    try:
        seed_frame_index(observed, lineage, parent | {'sourceFrameMediaTimesSeconds': [.6, .8, .9]}, clip)
    except ValueError:
        pass
    else:
        raise AssertionError('Seed phase mismatch accepted')
    print('seeded video self-check passed: native ID order, empty masks, source time and pixel domain')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['clip', 'source-manifest', 'seed-dir', 'seed-lineage', 'vendor', 'weights', 'output']:
        p.add_argument('--' + name, type=Path)
    p.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    p.add_argument('--self-check', action='store_true')
    args = p.parse_args()
    if args.self_check:
        self_check()
    elif all(getattr(args, n.replace('-', '_')) for n in
             ['clip', 'source-manifest', 'seed-dir', 'seed-lineage', 'vendor', 'weights', 'output']):
        existed = args.output.exists()
        try:
            print(json.dumps(run(args), indent=2))
        except Exception as error:
            if not existed and args.output.is_dir():
                save(args.output / 'run.json', {'status': 'execution_failed',
                     'error_type': type(error).__name__, 'error': str(error)})
            raise
    else:
        p.error('Provide every input path or --self-check')
