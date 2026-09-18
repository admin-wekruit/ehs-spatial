"""Propose same-session reentry edges; preserve raw native tracklets unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import tempfile

import numpy as np
from PIL import Image

from person_reid import PersonReID, sha, WEIGHTS_SHA, SOURCE_SHA

MIN_COSINE = .6
MIN_MARGIN = .1
MIN_CONSECUTIVE_DISCOVERIES = 2
NATIVE_IOU_MIN = .5
DISCOVERY_COLLISION_IOU = .5


def validate_run_files(raw_path, matches_path):
    raw_path, matches_path = raw_path.resolve(), matches_path.resolve()
    if matches_path != raw_path.parent / 'discovery-matches.jsonl':
        raise ValueError('Raw and discovery ledger must belong to the same run directory')
    raw_digest = sha(raw_path)
    raw = json.loads(raw_path.read_text())
    run_path = raw_path.parent / 'run.json'
    run = json.loads(run_path.read_text())
    if (run['status'] != 'execution_complete' or run['raw_sha256'] != raw_digest
            or run['session_id'] != raw['session_id'] or run['method'] != raw['method']):
        raise ValueError('Completed run, raw hash, session or method provenance does not match')
    return raw, raw_digest, sha(matches_path), sha(run_path)


def discovery_eligibility(assignments, masks):
    reasons = [None if item['accepted_continuation'] or
               max((candidate['iou'] for candidate in item['candidates']), default=0) < NATIVE_IOU_MIN
               else 'ambiguous_native_detection_association' for item in assignments]
    for i, a in enumerate(masks):
        for j in range(i + 1, len(masks)):
            union = np.count_nonzero(a | masks[j])
            if union and np.count_nonzero(a & masks[j]) / union >= DISCOVERY_COLLISION_IOU:
                reasons[i] = reasons[j] = 'overlapping_independent_discovery_masks'
    return reasons


def link(rows, active_by_frame, discovery_interval=30):
    """Rows contain independently discovered instances with normalized descriptors.

    These fixed experimental gates are not calibrated identity probabilities.
    Gallery entries retain their native ID; candidate edges never update a target.
    """
    gallery, ready, births, pending, emitted, edges, evaluations = {}, {}, {}, {}, set(), [], []
    previous_frame = None
    for row in rows:
        frame, observations = row['frame_index'], row['observations']
        if previous_frame is not None and frame <= previous_frame:
            raise ValueError('Discovery frames must strictly increase')
        if previous_frame is not None and frame - previous_frame != discovery_interval:
            pending.clear()
        previous_frame = frame
        ids = [item['track_id'] for item in observations]
        if len(ids) != len(set(ids)):
            raise ValueError('One native tracklet cannot have two detections in a discovery frame')
        if any(not np.isfinite(item['descriptor']).all() or
               not np.isclose(np.linalg.norm(item['descriptor']), 1, atol=1e-5) for item in observations):
            raise ValueError('Expected finite normalized appearance descriptors')
        for item in observations:
            births.setdefault(item['track_id'], frame)
        # ponytail: retain observed prototypes for this bounded offline video;
        # a long-running service should bound gallery prototypes per tracklet.
        scores = {}
        for item in observations:
            own = item['track_id']
            if not item['eligible']:
                continue
            scores[own] = {}
            for other, prototypes in gallery.items():
                if other == own or other not in ready:
                    continue
                values = [float(np.clip(np.dot(item['descriptor'], p['descriptor']), -1, 1)) for p in prototypes]
                best = int(np.argmax(values))
                scores[own][other] = {'cosine': values[best], 'reference_frame': prototypes[best]['frame_index'],
                                     'reference_instance': prototypes[best]['instance_index'],
                                     'target_discovery_evidence': ready[other]}
        next_pending = {}
        for item in observations:
            own = item['track_id']
            evidence = {'frame_index': frame, 'instance_index': item['instance_index'],
                        'discovery_manifest_sha256': row['manifest_sha256'],
                        'source_png_sha256': row['source_png_sha256'],
                        'mask_sha256': item['mask_sha256']}
            record = {'tracklet_id': own, 'evidence': evidence, 'scores_by_tracklet': scores.get(own, {}),
                      'status': 'identity_unresolved', 'candidate_tracklet_id': None}
            ranked = sorted(scores.get(own, {}).items(), key=lambda pair: pair[1]['cosine'], reverse=True)
            if not item['eligible']:
                record['reason'] = item.get('ineligible_reason', 'ambiguous_native_detection_association')
            elif len(ranked) < 2:
                record['reason'] = 'insufficient_competing_gallery_identities'
            else:
                target, best = ranked[0]
                score, margin = best['cosine'], best['cosine'] - ranked[1][1]['cosine']
                competitors = [values[target]['cosine'] for other, values in scores.items()
                               if other != own and target in values]
                column_margin = score - max(competitors) if competitors else None
                record.update(candidate_tracklet_id=target, cosine=score, row_margin=margin,
                              column_margin=column_margin, reference_frame=best['reference_frame'])
                if score < MIN_COSINE or margin < MIN_MARGIN:
                    record['reason'] = 'weak_or_ambiguous_appearance'
                elif target in active_by_frame[frame]:
                    record['reason'] = 'candidate_old_tracklet_is_visible'
                elif any(target in visible and own in visible for visible in active_by_frame.values()):
                    record['reason'] = 'candidate_tracklets_co_visible_in_raw'
                elif births[target] >= births[own]:
                    record['reason'] = 'candidate_does_not_predate_tracklet'
                elif column_margin is not None and column_margin < MIN_MARGIN:
                    record['reason'] = 'competing_detection_for_same_old_tracklet'
                else:
                    prior = pending.get(own)
                    consistent = (prior is not None and prior['target'] == target and
                                  item['accepted_continuation'] and
                                  prior['evidence'][-1]['discovery_manifest_sha256'] != row['manifest_sha256'])
                    chain = prior['evidence'] + [evidence] if consistent else [evidence]
                    next_pending[own] = {'target': target, 'evidence': chain[-MIN_CONSECUTIVE_DISCOVERIES:]}
                    record['consistent_discoveries'] = len(chain)
                    if len(chain) < MIN_CONSECUTIVE_DISCOVERIES:
                        record['reason'] = 'awaiting_independent_consistent_discovery'
                    else:
                        record.update(status='session_reentry_candidate', reason='appearance_and_temporal_gates_passed')
                        key = (own, target)
                        if key not in emitted:
                            edges.append({'type': 'session_reentry_candidate', 'from_tracklet_id': own,
                                          'to_earlier_tracklet_id': target, 'human_confirmed': False,
                                          'native_ids_modified': False, 'evidence': chain[-2:],
                                          'target_discovery_evidence': ready[target],
                                          'cosine': score, 'row_margin': margin, 'column_margin': column_margin})
                            emitted.add(key)
            evaluations.append(record)
        pending = next_pending  # Missing, weak, occupied or competing observations reset confirmation.
        for item in observations:
            if item['eligible']:
                prototypes = gallery.setdefault(item['track_id'], [])
                evidence = {'frame_index': frame, 'instance_index': item['instance_index'],
                            'discovery_manifest_sha256': row['manifest_sha256'],
                            'source_png_sha256': row['source_png_sha256'], 'mask_sha256': item['mask_sha256']}
                if (item['track_id'] not in ready and prototypes and item['accepted_continuation']
                        and prototypes[-1]['frame_index'] + discovery_interval == frame
                        and prototypes[-1]['evidence']['discovery_manifest_sha256'] != row['manifest_sha256']):
                    ready[item['track_id']] = [prototypes[-1]['evidence'], evidence]
                prototypes.append(item | {'frame_index': frame, 'evidence': evidence})
    return edges, evaluations


def run(args):
    start = time.monotonic()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing candidate experiment')
    raw, raw_digest, ledger_digest, run_digest = validate_run_files(args.raw, args.discovery_matches)
    if raw['identity_scope'] != 'video_session_only' or raw['method']['discovery_step_frames'] != args.discovery_interval_frames:
        raise ValueError('Expected same-session periodic raw with the declared discovery schedule')
    decoding_path = args.raw.parent / 'input-decoding.json'
    decoding = json.loads(decoding_path.read_text())
    if decoding['clip']['sha256'] != raw['input']['sha256']:
        raise ValueError('Decoded source RGB ledger belongs to another input')
    decoded_frames = {frame['frame_index']: frame for frame in decoding['frames']}
    records = [json.loads(line) for line in args.discovery_matches.read_text().splitlines() if line.strip()]
    frames = {frame['frame_index']: frame for frame in raw['frames']}
    if len(frames) != len(raw['frames']):
        raise ValueError('Duplicate raw frame indices')
    model = PersonReID(args.model_dir, args.device)
    active = {index: {obj['track_id'] for obj in frame['objects'] if obj['mask_area_pixels'] > 0}
              for index, frame in frames.items()}
    rows, vectors, names = [], [], []
    for record in records:
        index = record['frame_index']
        if index not in frames:
            raise ValueError('Discovery frame has no raw media observation')
        frame = frames[index]
        image_path = Path(record['source_png'])
        manifest_path = image_path.parent / 'input-manifest.json'
        if sha(manifest_path) != record['manifest_sha256']:
            raise ValueError('Independent discovery source manifest changed')
        manifest = json.loads(manifest_path.read_text())
        if (manifest['source_clip_sha256'] != raw['input']['sha256'] or manifest['source_frame_index'] != index
                or frame['source_frame_index'] != index
                or manifest['frame_sha256'] != record['source_png_sha256']
                or abs(manifest['timestamp_seconds'] - frame['timestamp_seconds']) > .001):
            raise ValueError('Independent discovery refers to another source, frame or media time')
        if sha(image_path) != record['source_png_sha256']:
            raise ValueError('Discovery source PNG changed')
        rgb = np.asarray(Image.open(image_path).convert('RGB'))
        if rgb.shape[:2] != (raw['input']['height'], raw['input']['width']):
            raise ValueError('Discovery image and raw video pixel domains differ')
        if hashlib.sha256(rgb.tobytes()).hexdigest() != decoded_frames[index]['decoded_rgb_sha256']:
            raise ValueError('Discovery PNG differs from the actual decoded video frame')
        observed_ids = {obj['track_id'] for obj in frame['objects']}
        observations, masks = [], []
        for assignment in record['assignments']:
            if assignment['track_id'] not in observed_ids:
                raise ValueError('Discovered tracklet is missing from raw native output')
            mask_path = Path(assignment['mask_path'])
            if sha(mask_path) != assignment['mask_sha256']:
                raise ValueError('Independent discovery mask changed')
            mask = np.asarray(Image.open(mask_path).convert('L')) > 0
            if mask.shape != rgb.shape[:2]:
                raise ValueError('Discovery mask and RGB source pixel domains differ')
            masks.append(mask)
        reasons = discovery_eligibility(record['assignments'], masks)
        for assignment, mask, reason in zip(record['assignments'], masks, reasons, strict=True):
            if not mask.any():
                continue  # Empty discovery cannot confirm identity; pending count is cleared by link().
            vector, box = model.describe(rgb, mask)
            continuation = assignment['accepted_continuation']
            eligible = reason is None
            observations.append({'track_id': assignment['track_id'], 'instance_index': assignment['instance_index'],
                                 'descriptor': vector, 'eligible': eligible, 'accepted_continuation': continuation,
                                 'ineligible_reason': reason, 'mask_sha256': assignment['mask_sha256'], 'crop_xyxy': box})
            vectors.append(vector)
            names.append(f"{index}:{assignment['instance_index']}:{assignment['track_id']}")
        rows.append({'frame_index': index, 'manifest_sha256': record['manifest_sha256'],
                     'source_png_sha256': record['source_png_sha256'], 'observations': observations})
    edges, evaluations = link(rows, active, args.discovery_interval_frames)
    if (sha(args.raw) != raw_digest or sha(args.discovery_matches) != ledger_digest
            or sha(args.raw.parent / 'run.json') != run_digest):
        raise ValueError('Native run artifacts changed during postprocessing')
    result = {'session_id': raw['session_id'], 'identity_scope': raw['identity_scope'],
              'method': 'OSNet-AIN appearance plus consecutive independent discovery candidate linking',
              'thresholds': {'cosine_min': MIN_COSINE, 'row_and_column_margin_min': MIN_MARGIN,
                             'consecutive_independent_discoveries': MIN_CONSECUTIVE_DISCOVERIES,
                             'consecutive_clean_target_gallery_discoveries': 2,
                             'native_iou_min': NATIVE_IOU_MIN, 'discovery_mask_collision_iou': DISCOVERY_COLLISION_IOU},
              'threshold_status': 'fixed_experimental_gates_not_calibrated_on_unseen_person_negatives',
              'raw_sha256': raw_digest, 'discovery_matches_sha256': ledger_digest,
              'completed_run_manifest_sha256': run_digest,
              'source_run_directory': str(args.raw.resolve().parent),
              'source_decoding_sha256': sha(decoding_path),
              'weights_sha256': WEIGHTS_SHA, 'model_source_sha256': SOURCE_SHA,
              'script_sha256': sha(__file__), 'descriptor_script_sha256': sha(Path(__file__).with_name('person_reid.py')),
              'discovery_interval_frames': args.discovery_interval_frames,
              'native_ids_modified': False, 'candidate_features_copied_to_old_gallery': False,
              'edges': edges, 'evaluations': evaluations, 'elapsed_seconds': time.monotonic() - start,
              'limitations': ['Session-scoped appearance candidates; not physical identity proof or cross-video re-ID',
                             'Unknown-person negative examples have not calibrated these gates',
                             'Human confirmation remains absent; raw native tracklets remain the default scene IDs']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix('.npz'), descriptors=vectors, observation_keys=names)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps({'edges': len(edges), 'evaluations': len(evaluations), 'elapsed_seconds': result['elapsed_seconds']}))


def self_check():
    def obs(identity, vector, continuation=False):
        return {'track_id': identity, 'instance_index': identity, 'descriptor': np.array(vector),
                'eligible': True, 'accepted_continuation': continuation, 'mask_sha256': 'mask'}
    def row(frame, items):
        return {'frame_index': frame, 'manifest_sha256': f'manifest-{frame}',
                'source_png_sha256': f'image-{frame}', 'observations': items}
    initial = row(0, [obs(0, [1., 0, 0]), obs(1, [0, 1., 0])])
    established = row(30, [obs(0, [1., 0, 0], True), obs(1, [0, 1., 0], True)])
    gallery_rows = [initial, established]
    vector = [0.8, .1, np.sqrt(.35)]
    one = row(60, [obs(2, vector)])
    two = row(90, [obs(2, vector, True)])
    visible = {0: {0, 1}, 30: {0, 1}, 60: {2}, 90: {2}, 120: {2}}
    edges, _ = link([*gallery_rows, one, two], visible)
    assert len(edges) == 1 and edges[0]['from_tracklet_id'] == 2 and edges[0]['to_earlier_tracklet_id'] == 0
    assert not edges[0]['human_confirmed'] and not edges[0]['native_ids_modified']
    assert [e['frame_index'] for e in edges[0]['target_discovery_evidence']] == [0, 30]
    assert not link([initial, one, two], visible)[0]  # A single discovery cannot qualify the target gallery.
    assert not link([*gallery_rows, one, two], visible | {90: {0, 2}})[0]
    assert not link([*gallery_rows, one, row(90, []), row(120, [obs(2, vector, True)])], visible)[0]
    assert not link([*gallery_rows, one, row(120, [obs(2, vector, True)])], visible)[0]
    weak = [.59, .1, np.sqrt(1 - .59 ** 2 - .1 ** 2)]
    assert not link([*gallery_rows, one, row(90, [obs(2, weak, True)])], visible)[0]
    twin = [0.81, .1, np.sqrt(1 - .81 ** 2 - .1 ** 2)]
    crowded = [*gallery_rows, row(60, [obs(2, vector), obs(3, twin)]),
               row(90, [obs(2, vector, True), obs(3, twin, True)])]
    assert not link(crowded, visible | {60: {2, 3}, 90: {2, 3}})[0]
    assert not link([*gallery_rows, one, two | {'manifest_sha256': one['manifest_sha256']}], visible)[0]
    lone_gallery = [row(0, [obs(0, [1., 0, 0])]), row(30, [obs(0, [1., 0, 0], True)])]
    assert not link([*lone_gallery, one, two], visible)[0]
    assert not link([*gallery_rows, one, two], visible | {75: {0, 2}})[0]
    independent = {'accepted_continuation': False, 'candidates': [{'iou': 0}]}
    distinct_masks = [np.array([[True, False]]), np.array([[False, True]])]
    assert discovery_eligibility([independent, independent], distinct_masks) == [None, None]
    duplicate = discovery_eligibility([independent, independent], [distinct_masks[0]] * 2)
    assert duplicate == ['overlapping_independent_discovery_masks'] * 2
    with tempfile.TemporaryDirectory() as folder:
        a, b = Path(folder) / '001', Path(folder) / '002'
        a.mkdir(); b.mkdir()
        raw_path, ledger_path = b / 'observations.json', b / 'discovery-matches.jsonl'
        raw_path.write_text(json.dumps({'session_id': 'session-002', 'method': {'script_sha256': 'code'}}))
        ledger_path.write_text('{}\n'); (a / 'discovery-matches.jsonl').write_text('{}\n')
        run = {'status': 'execution_complete', 'raw_sha256': sha(raw_path),
               'session_id': 'session-002', 'method': {'script_sha256': 'code'}}
        (b / 'run.json').write_text(json.dumps(run))
        assert validate_run_files(raw_path, ledger_path)[1] == run['raw_sha256']
        for wrong_ledger, changed in [(a / 'discovery-matches.jsonl', run),
                                      (ledger_path, run | {'session_id': 'session-001'}),
                                      (ledger_path, run | {'raw_sha256': 'wrong'}),
                                      (ledger_path, run | {'method': {'script_sha256': 'other-code'}})]:
            (b / 'run.json').write_text(json.dumps(changed))
            try:
                validate_run_files(raw_path, wrong_ledger)
            except ValueError:
                pass
            else:
                raise AssertionError('Wrong-run ledger or completion provenance accepted')
    print('tracklet linking self-check passed: clean two-discovery gallery and candidates; active/weak/missing/gap/tie/reused/single-gallery rejection')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'discovery-matches', 'model-dir', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--discovery-interval-frames', type=int, default=30)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.raw and args.discovery_matches and args.model_dir and args.output and args.discovery_interval_frames > 0:
        run(args)
    else:
        parser.error('raw, discovery-matches, model-dir, output and positive discovery interval are required')
