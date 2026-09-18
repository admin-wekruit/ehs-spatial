"""Saved periodic SAM3 discoveries with bounded SAM2 propagation, without identity guesses.

Uses one cached model, independent short inference states, and the complete CFR
source. Raw native masks (including collisions and duplicate window boundaries)
are retained separately from the once-per-source-frame observation ledger.
"""
from __future__ import annotations

import argparse
import gc
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

from run_seeded_video import decoded_clip, encode_output, save, sha, decode_coco_rle
from build_video_pose_preview import intervals, source_spans


def mask_iou(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.


def match_discovery(previous, masks, next_id):
    """Mutual best matches must clear both competitors; ties remain unknown."""
    old_ids = [x['track_id'] for x in previous]
    if len(old_ids) != len(set(old_ids)) or any(type(i) is not int or i < 0 for i in old_ids):
        raise ValueError('Invalid previous track IDs')
    if next_id <= max(old_ids, default=-1):
        raise ValueError('New IDs must be strictly monotonic')
    scores = np.array([[mask_iou(mask, old['mask']) for old in previous] for mask in masks])
    assignments = []
    for row, mask in enumerate(masks):
        item = {'instance_index': row, 'previous_track_id': None,
                'accepted_continuation': False, 'reason': 'new_or_unknown',
                'candidates': [{'previous_track_id': old_id, 'iou': float(scores[row, col])}
                               for col, old_id in enumerate(old_ids)],
                'row_margin': None, 'col_margin': None}
        if old_ids:
            col = int(np.argmax(scores[row]))
            best = float(scores[row, col])
            row_second = max(np.delete(scores[row], col), default=0.)
            col_second = max(np.delete(scores[:, col], row), default=0.)
            item.update(row_margin=float(best-row_second), col_margin=float(best-col_second),
                        best_previous_track_id=old_ids[col], best_iou=best)
            if (best >= .5 and item['row_margin'] >= .15 and item['col_margin'] >= .15
                    and int(np.argmax(scores[:, col])) == row):
                item.update(track_id=old_ids[col], previous_track_id=old_ids[col],
                            accepted_continuation=True, reason='mutual_unique_iou')
        if not item['accepted_continuation']:
            item['track_id'] = next_id
            next_id += 1
        assignments.append(item)
    return assignments, next_id


def discoveries(roots, clip, decoded, step):
    found = {}
    for root in roots:
        for path in sorted(root.glob('frame-*/input-manifest.json')):
            manifest = json.loads(path.read_text())
            index = manifest['source_frame_index']
            if type(index) is not int or not 0 <= index < clip['frame_count'] or index in found:
                raise ValueError(f'Duplicate/out-of-range discovery frame: {index}')
            if (manifest.get('prompt') != 'person' or manifest['source_clip_sha256'] != clip['sha256']
                    or (manifest['height'], manifest['width']) != (clip['height'], clip['width'])
                    or abs(manifest['timestamp_seconds']-clip['frame_timestamps_seconds'][index]) > .001):
                raise ValueError('Discovery source, pixel domain, or timestamp differs')
            png = path.parent / f'frame-{index}.png'
            bgr = cv2.imread(str(png))
            if bgr is None or sha(png) != manifest['frame_sha256']:
                raise ValueError('Discovery source PNG changed')
            rgb_hash = hashlib.sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
            if rgb_hash != decoded[index]['decoded_rgb_sha256']:
                raise ValueError('Discovery PNG is not this exact decoded source frame')
            provider_path = path.parent / 'provider-output.json'
            provider = json.loads(provider_path.read_text())
            instances = json.loads((path.parent / 'instances.json').read_text())
            if sorted(i['instance_index'] for i in instances) != list(range(len(provider['rle']))):
                raise ValueError('Discovery instances do not cover every native provider mask')
            masks, provenance = [], []
            for instance in sorted(instances, key=lambda x: x['instance_index']):
                i = instance['instance_index']
                mask_path = path.parent / f'instance-{i}-mask.png'
                image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                native = decode_coco_rle(provider['rle'][i], height=clip['height'], width=clip['width']).astype(bool)
                if (instance.get('label') != 'person' or native.shape != (clip['height'], clip['width'])
                        or image is None or image.ndim != 2 or image.shape != native.shape
                        or sha(mask_path) != instance['mask_sha256']
                        or not np.array_equal(image > 0, native) or not native.any()):
                    raise ValueError('Saved discovery mask differs from native RLE')
                masks.append(native)
                provenance.append({'mask_path': str(mask_path.resolve()), 'mask_sha256': sha(mask_path)})
            found[index] = {'frame_index': index, 'source_png': str(png.resolve()),
                            'source_png_sha256': sha(png), 'manifest_sha256': sha(path),
                            'provider_output_sha256': sha(provider_path), 'masks': masks,
                            'instances': provenance}
    required = list(range(0, clip['frame_count'], step))
    if sorted(found) != required:
        raise ValueError(f'Discoveries must cover exactly every {step} frames; missing {sorted(set(required)-found.keys())}, extra {sorted(found.keys()-set(required))}')
    return dict(sorted(found.items()))


def collision_summary(frames):
    result = {'frames': len(frames), 'any_overlap_frames': 0, 'pair_iou_ge_0_5_frames': 0,
              'pair_iou_ge_0_9_frames': 0, 'empty_union_frames': 0, 'overlap_pixels_sum': 0}
    for frame in frames:
        masks = [decode_coco_rle(o['rle']).astype(bool) for o in frame['objects']]
        result['empty_union_frames'] += not any(m.any() for m in masks)
        if not masks:
            continue
        overlap = int(np.count_nonzero(np.sum(masks, axis=0) > 1))
        largest = max((mask_iou(a, b) for i, a in enumerate(masks) for b in masks[i+1:]), default=0.)
        result['any_overlap_frames'] += overlap > 0
        result['overlap_pixels_sum'] += overlap
        result['pair_iou_ge_0_5_frames'] += largest >= .5
        result['pair_iou_ge_0_9_frames'] += largest >= .9
    return result


def run(args):
    import torch

    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    frame_dir = args.output / 'frames'; frame_dir.mkdir()
    clip, decoded = decoded_clip(args.clip, frame_dir)
    found = discoveries(args.discovery_roots, clip, decoded, args.step)
    parent = {'sourceVideo': str(args.clip.resolve()), 'sourceSha256': clip['sha256'],
              'sourceStartFrame': 0, 'frameCount': clip['frame_count'], 'clipSha256': clip['sha256'],
              'clipDurationSeconds': clip['duration_seconds'],
              'sourceFrameMediaTimesSeconds': [*clip['frame_timestamps_seconds'], clip['duration_seconds']],
              'purpose': 'Complete source; periodic saved SAM3 discovery and bounded SAM2 memory'}
    skeleton = {'input': clip, 'frames': [{'frame_index': i, 'source_frame_index': i,
                 'timestamp_seconds': t} for i, t in enumerate(clip['frame_timestamps_seconds'])]}
    source_spans(skeleton, parent)
    save(args.output / 'source-manifest.json', parent)
    save(args.output / 'input-decoding.json', {'clip': clip, 'frames': decoded,
         'encoding': 'OpenCV decoded RGB -> JPEG quality100 subsampling0; no resizing or retiming'})
    revision = subprocess.check_output(['git', '-C', str(args.vendor), 'rev-parse', 'HEAD'], text=True).strip()
    method = {'name': 'Periodic SAM3 discovery + short-window SAM2.1 propagation',
              'script_sha256': sha(Path(__file__)), 'source_revision': revision,
              'weights_sha256': sha(args.weights), 'device': args.device, 'torch_version': torch.__version__,
              'mps_cpu_fallback_environment': os.getenv('PYTORCH_ENABLE_MPS_FALLBACK'),
              'discovery_step_frames': args.step, 'boundary_overlap_frames': 1,
              'mask_logit_threshold': 0, 'postprocessing': False, 'overlap_suppression': False,
              'identity_rule': 'Mutual best IoU>=0.5 and both competitor margins>=0.15; otherwise new unknown track. No cross-gap re-identification.',
              'model_instances': 1, 'identity_scope': 'video_session_only',
              'offload_video_to_cpu': True, 'offload_state_to_cpu': False}
    raw = {'session_id': str(uuid.uuid4()), 'identity_scope': 'video_session_only',
           'method': method, 'input': clip, 'frames': []}
    save(args.output / 'run.json', {'status': 'running', 'session_id': raw['session_id'], 'method': method})
    sys.path.insert(0, str(args.vendor))
    from sam2.build_sam import build_sam2_video_predictor
    model = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml', str(args.weights),
                                     device=args.device, apply_postprocessing=False, vos_optimized=False)
    if model.non_overlap_masks:
        raise ValueError('Native overlap suppression must remain disabled')
    previous, next_id = [], 0
    with torch.inference_mode(), (args.output / 'frames.jsonl').open('w') as ledger, \
            (args.output / 'native-windows.jsonl').open('w') as native_ledger, \
            (args.output / 'discovery-matches.jsonl').open('w') as match_ledger, \
            (args.output / 'identity-events.jsonl').open('w') as events:
        for start, discovery in found.items():
            last = min(start + args.step, clip['frame_count']-1)
            emit_end = min(start + args.step, clip['frame_count'])
            assignments, next_id = match_discovery(previous, discovery['masks'], next_id)
            for assignment, evidence in zip(assignments, discovery['instances'], strict=True):
                assignment.update(evidence)
            evidence = {k: v for k, v in discovery.items() if k not in ('masks', 'instances')}
            evidence['assignments'] = assignments
            match_ledger.write(json.dumps(evidence, allow_nan=False)+'\n'); match_ledger.flush()
            ids = [a['track_id'] for a in assignments]
            current_ids = dict(zip(ids, ids))
            absent = {i: False for i in ids}
            window = args.output / 'windows' / f'{start:05d}'; window.mkdir(parents=True)
            for local, index in enumerate(range(start, last+1)):
                (window / f'{local:05d}.jpg').symlink_to(frame_dir / f'{index:05d}.jpg')
            if ids:
                # MPS CPU-state offload corrupts single-object predictions in this runtime;
                # device-resident state matches CPU reference masks. Each state is bounded by this window.
                state = model.init_state(str(window), offload_video_to_cpu=True, offload_state_to_cpu=False)
                for track_id, mask in zip(ids, discovery['masks'], strict=True):
                    model.add_new_mask(state, frame_idx=0, obj_id=track_id, mask=mask)
                predictions = model.propagate_in_video(state, start_frame_idx=0)
            else:
                predictions = ((i, [], np.empty((0, 1, clip['height'], clip['width']), np.float32))
                               for i in range(last-start+1))
            previous = []
            processed = 0
            for local, native_ids, logits in predictions:
                if local != processed:
                    raise ValueError('Native short window omitted/reordered a frame')
                processed += 1
                index = start + local
                values = logits.float().cpu().numpy() if ids else logits
                frame = encode_output(index, native_ids, values, clip, 0, ids)
                native_ledger.write(json.dumps({'window_start': start, 'local_frame_index': local, **frame}, allow_nan=False)+'\n')
                for obj in frame['objects']:
                    native_id = obj['track_id']
                    if obj['mask_area_pixels'] > 0 and absent[native_id]:
                        # A zero-mask gap cannot establish re-entry identity, even within one window.
                        current_ids[native_id] = next_id; next_id += 1
                        events.write(json.dumps({'frame_index': index, 'window_start': start,
                            'native_object_id': native_id, 'track_id': current_ids[native_id],
                            'reason': 'new_unknown_after_absence'})+'\n')
                    absent[native_id] = obj['mask_area_pixels'] == 0
                    obj.update(native_object_id=native_id, track_id=current_ids[native_id],
                               window_start=start, identity_status='short_term_only')
                if index < emit_end:
                    if index != len(raw['frames']):
                        raise ValueError('Canonical output duplicated/skipped a source frame')
                    raw['frames'].append(frame)
                    ledger.write(json.dumps(frame, allow_nan=False)+'\n'); ledger.flush()
                if index == last:
                    previous = [{'track_id': o['track_id'], 'mask': decode_coco_rle(o['rle']).astype(bool)}
                                for o in frame['objects']]
            if processed != last-start+1:
                raise ValueError('Incomplete native short window')
            native_ledger.flush(); events.flush()
            if ids:
                model.reset_state(state)
                del state, predictions, logits, values
                gc.collect()
                if args.device == 'mps':
                    torch.mps.empty_cache()
            print(json.dumps({'window_start': start, 'output_frames': len(raw['frames']),
                              'discovered_instances': len(ids), 'next_id': next_id,
                              'elapsed_seconds': round(time.monotonic()-started, 2)}), flush=True)
    intervals(raw); source_spans(raw, parent)
    save(args.output / 'observations.json', raw)
    result = {'status': 'execution_complete', 'frames': len(raw['frames']), 'session_id': raw['session_id'],
              'allocated_track_ids': next_id, 'method': method, 'elapsed_seconds': time.monotonic()-started,
              'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
              'mps_driver_allocated_bytes_at_end': torch.mps.driver_allocated_memory() if args.device == 'mps' else None,
              'quality': 'requires_visual_review', 'raw_sha256': sha(args.output / 'observations.json'),
              'collisions': collision_summary(raw['frames'])}
    if args.compare:
        baseline = json.loads(args.compare.read_text())
        intervals(baseline)
        if any(baseline['input'][k] != clip[k] for k in ['sha256', 'frame_count', 'height', 'width']):
            raise ValueError('Baseline comparison must use identical source video')
        result['baseline_comparison'] = {'source': str(args.compare.resolve()), 'sha256': sha(args.compare),
                                        'collisions': collision_summary(baseline['frames'])}
        prior = json.loads((args.compare.parent / 'run.json').read_text())
        result['baseline_comparison'].update({k: prior.get(k) for k in ['elapsed_seconds', 'peak_process_rss_bytes']})
    save(args.output / 'run.json', result)
    return result


def self_check():
    a = np.zeros((8, 8), bool); a[:3, :3] = True
    b = np.zeros_like(a); b[5:, 5:] = True
    previous = [{'track_id': 7, 'mask': a}, {'track_id': 9, 'mask': b}]
    matched, next_id = match_discovery(previous, [b, a], 10)
    assert [x['track_id'] for x in matched] == [9, 7] and next_id == 10
    tied, next_id = match_discovery(previous, [a, a], 10)
    assert [x['track_id'] for x in tied] == [10, 11] and next_id == 12
    tied, _ = match_discovery([{'track_id': 7, 'mask': a}, {'track_id': 9, 'mask': a}], [a], 10)
    assert not tied[0]['accepted_continuation']
    almost = a.copy(); almost[0, 0] = False
    close, _ = match_discovery([{'track_id': 7, 'mask': a}, {'track_id': 9, 'mask': almost}], [a], 10)
    assert close[0]['best_iou'] == 1 and close[0]['row_margin'] < .15
    assert not close[0]['accepted_continuation']
    empty, next_id = match_discovery(previous, [], 10)
    assert empty == [] and next_id == 10
    fresh, next_id = match_discovery([], [a, b], next_id)
    assert [x['track_id'] for x in fresh] == [10, 11] and next_id == 12
    absent, _ = match_discovery([{'track_id': 1, 'mask': np.zeros_like(a)}], [a], 12)
    assert absent[0]['track_id'] == 12 and not absent[0]['accepted_continuation']
    print('periodic discovery self-check passed: permutation, both ambiguity directions, empty windows, absent masks, monotonic fresh IDs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['clip', 'vendor', 'weights', 'output', 'compare']:
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--discovery-roots', type=Path, nargs='+')
    parser.add_argument('--step', type=int, default=30)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        if any(getattr(args, k) is None for k in ['clip', 'vendor', 'weights', 'output', 'discovery_roots']) or args.step < 1:
            parser.error('clip, vendor, weights, output, discovery-roots and positive step are required')
        args.output = args.output.resolve()
        if args.output.exists():
            parser.error('Output must be a new directory')
        try:
            print(json.dumps(run(args), allow_nan=False))
        except Exception as error:
            if args.output.exists():
                path = args.output / 'run.json'
                result = json.loads(path.read_text()) if path.exists() else {}
                save(path, {**result, 'status': 'execution_failed', 'error': repr(error)})
            raise
