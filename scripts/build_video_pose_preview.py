"""Attach pretrained RTMPose joints to native SAM video IDs and export replay data.

No second detector/tracker, new identity guesses, or 3D pose claims. Requires the
saved SAM result and its exact input clip; source-frame offsets remain explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehs_spatial.providers.sam3 import decode_coco_rle

BONES = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9],
         [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
         [13, 15], [12, 14], [14, 16]]


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def intervals(raw: dict) -> list[tuple[float, float]]:
    frames, clip = raw['frames'], raw['input']
    if [f['frame_index'] for f in frames] != list(range(clip['frame_count'])):
        raise ValueError('Expected every native frame, including empty outputs')
    times = clip['frame_timestamps_seconds']
    if len(times) != len(frames) or not times or times[0] < 0:
        raise ValueError('Invalid source-media timestamps')
    ends = [*times[1:], clip['duration_seconds']]
    if any(not np.isfinite(a) or not np.isfinite(b) or a >= b for a, b in zip(times, ends)):
        raise ValueError('Timestamps must be finite, strictly increasing intervals')
    if any(abs(f['timestamp_seconds'] - t) > 1e-6 for f, t in zip(frames, times)):
        raise ValueError('Frame and clip timestamps disagree')
    return list(zip(times, ends))


def object_input(obj: dict, width: int, height: int) -> tuple[np.ndarray, list[float] | None]:
    mask = decode_coco_rle(obj['rle'], height=height, width=width).astype(bool)
    if mask.shape != (height, width):
        raise ValueError('Mask is outside the source pixel domain')
    box = np.asarray(obj['box_xywh_normalized'], dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0) or np.any(box > 1):
        raise ValueError('Invalid normalized xywh box')
    x, y, w, h = box
    if x + w > 1.00001 or y + h > 1.00001:
        raise ValueError('Normalized xywh box extends outside the image')
    if not mask.any() or w == 0 or h == 0:
        return mask, None
    return mask, [float(x * width), float(y * height),
                  float(min(1, x + w) * width), float(min(1, y + h) * height)]


def source_spans(raw: dict, manifest: dict) -> list[tuple[float, float]]:
    """Bind clip frames to the exact parent media, never a free UI time offset."""
    spans = intervals(raw)
    if manifest['clipSha256'] != raw['input']['sha256'] or manifest['frameCount'] != len(spans):
        raise ValueError('Clip provenance does not match the actual SAM input')
    source = Path(manifest['sourceVideo'])
    if sha(source) != manifest['sourceSha256']:
        raise ValueError('Source video changed since the clip was prepared')
    start = manifest['sourceStartFrame']
    if type(start) is not int or start < 0 or any(
        f['source_frame_index'] != start + i for i, f in enumerate(raw['frames'])
    ):
        raise ValueError('Native outputs are not bound to the declared source frames')
    times = manifest['sourceFrameMediaTimesSeconds']
    if len(times) != len(spans) + 1 or any(
        not np.isfinite(a) or not np.isfinite(b) or a < 0 or a >= b
        for a, b in zip(times, times[1:])
    ):
        raise ValueError('Source frame presentation times must include the final interval end')
    cap = cv2.VideoCapture(str(source))
    actual = []
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(start + len(spans) + 1):
            ok, image = cap.read()
            if not ok:
                # ponytail: a full, byte-identical CFR source has no following
                # frame; derive only that final end from verified frame spacing.
                if (start == 0 and i == count == len(spans) and len(actual) >= 2
                        and manifest['clipSha256'] == manifest['sourceSha256']
                        and np.isfinite(fps) and fps > 0
                        and np.allclose(np.diff(actual), 1 / fps, atol=.0001, rtol=0)):
                    actual.append(actual[-1] + 1 / fps)
                    break
                raise ValueError('The parent video must include the frame after this analysis clip')
            if image.shape[:2] != (raw['input']['height'], raw['input']['width']):
                raise ValueError('Parent video and analysis pixel domains differ')
            if i >= start:
                actual.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    finally:
        cap.release()
    if any(abs(t - a) > .001 for t, a in zip(times, actual, strict=True)):
        raise ValueError('Declared times differ from the actual parent video frames')
    # The prepared MVP clip preserves every frame. Reject retiming, including VFR changes.
    clip_times = [*[a for a, _ in spans], spans[-1][1]]
    if any(abs((a - clip_times[0]) - (t - times[0])) > .002
           for a, t in zip(clip_times, times, strict=True)):
        raise ValueError('Clip was retimed relative to the parent source video')
    return list(zip(times, times[1:]))


def build(raw_path: Path, clip_path: Path, model_path: Path | None, output: Path,
          source_manifest: Path, reentry_candidates: Path | None = None) -> dict:
    from importlib.metadata import version

    started = time.monotonic()
    raw = json.loads(raw_path.read_text())
    reentry = json.loads(reentry_candidates.read_text()) if reentry_candidates else None
    if reentry is not None and (reentry['raw_sha256'] != sha(raw_path)
            or reentry['session_id'] != raw['session_id'] or reentry['native_ids_modified'] is not False):
        raise ValueError('Re-entry candidates must reference this exact unmodified segmentation run')
    if raw['input']['sha256'] != sha(clip_path):
        raise ValueError('Clip hash differs from the actual SAM input')
    manifest = json.loads(source_manifest.read_text())
    spans = source_spans(raw, manifest)
    width, height = raw['input']['width'], raw['input']['height']
    labels = [o.get('label') for f in raw['frames'] for o in f['objects']]
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError('Every observation requires its native discovery label')
    has_people = 'person' in labels
    pose = None
    if has_people:
        if model_path is None:
            raise ValueError('Person observations require --pose-model')
        from rtmlib import RTMPose
        pose = RTMPose(str(model_path), model_input_size=(192, 256),
                       to_openpose=False, backend='onnxruntime', device='cpu')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'masks').mkdir()
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError('Cannot decode the SAM input clip')
    session = raw['session_id']
    if not isinstance(session, str) or not session.strip():
        raise ValueError('A native session or provider request ID is required')
    namespace = 'sam-' + session
    result = {'version': 1, 'coordinateSpace': 'source_pixels', 'width': width,
              'height': height, 'method': raw['method']['name'] + (' + RTMPose COCO17 (CPU ONNX)' if has_people else ''),
              'identityScope': raw['identity_scope'], 'frames': [],
              'limitations': ['观测仅覆盖所记录的帧区间，不为缺失区间补位置。',
                              'ID 是本次视频会话中的短期轨迹，未验证跨遮挡或跨视频持久身份。']}
    if has_people:
        result['limitations'].extend([
                              '此文件保存二维骨架；三维位置需由空间回放中的有效相机和深度另行支持。',
                              '关节分数是模型响应，0.3 仅为显示阈值，不是校准后的正确概率。'])
    else:
        result['limitations'].append('非人物对象仅显示实际掩码和轨迹；不套用人体骨架，也不据二维位置推断三维运动。')
    observations = person_observations = 0
    try:
        for frame, (start, end) in zip(raw['frames'], spans, strict=True):
            ok, bgr = cap.read()
            if not ok or bgr.shape[:2] != (height, width):
                raise ValueError('Decoded clip does not match the native frame domain')
            inputs = [object_input(o, width, height) for o in frame['objects']]
            ids = [o['track_id'] for o in frame['objects']]
            if len(ids) != len(set(ids)) or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError('Native track IDs must be unique nonnegative integers')
            objects, absent = [], []
            # ponytail: top-down pose uses SAM boxes directly; no duplicate tracker.
            eligible = [i for i, (_, box) in enumerate(inputs)
                        if box is not None and frame['objects'][i]['label'] == 'person']
            pose_output = {}
            if eligible:
                # Official model pipeline.json sets to_rgb=true; RTMLib does not swap channels.
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                points, scores = pose(rgb, bboxes=[inputs[i][1] for i in eligible])
                if points.shape != (len(eligible), 17, 2) or scores.shape != (len(eligible), 17):
                    raise ValueError('Expected COCO17 output for every SAM person')
                pose_output = {i: (points[j], scores[j]) for j, i in enumerate(eligible)}
            if inputs:
                for j, (obj, (mask, box)) in enumerate(zip(frame['objects'], inputs, strict=True)):
                    if not mask.any():
                        absent.append(f"{namespace}-{obj['track_id']}")
                        continue
                    name = f"masks/{frame['frame_index']:05d}-{obj['track_id']}.png"
                    rgba = np.zeros((height, width, 4), dtype=np.uint8)
                    rgba[mask] = (190, 226, 101, 255)  # OpenCV BGRA; transparent off-mask.
                    if not cv2.imwrite(str(output / name), rgba):
                        raise IOError('Mask PNG export failed')
                    saved = cv2.imread(str(output / name), cv2.IMREAD_UNCHANGED)
                    if not np.array_equal(saved[:, :, 3] > 0, mask):
                        raise ValueError('Preview mask changed the native segmentation')
                    pts, confidence = pose_output.get(j, ([], []))
                    joints = [[float(x), float(y), float(min(1, max(0, c)))]
                              if np.isfinite([x, y, c]).all() else None
                              for (x, y), c in zip(pts, confidence, strict=True)]
                    exported = {'entityId': f"{namespace}-{obj['track_id']}",
                                    'nativeTrackId': obj['track_id'],
                                    'label': ('人' if obj['label'] == 'person' else obj['label']) + ' · 短期轨迹',
                                    'sourceLabel': obj['label'],
                                    'maskUrl': name,
                                    'keypoints': joints, 'bones': BONES if joints else [],
                                    'poseStatus': ('not_applicable_nonhuman' if obj['label'] != 'person' else
                                                   'estimated_2d' if joints else 'insufficient_mask_support'),
                                    'rawKeypointScores': [float(c) if np.isfinite(c) else None for c in confidence]}
                    if box is not None:
                        exported['bbox'] = box
                    if obj.get('score') is not None:
                        if not np.isfinite(obj['score']) or not 0 <= obj['score'] <= 1:
                            raise ValueError('Object score must be null or a finite value in [0,1]')
                        exported['confidence'] = obj['score']
                    objects.append(exported)
                    person_observations += obj['label'] == 'person'
            result['frames'].append({'timeSec': start,
                                     'endTimeSec': end,
                                     'sourceFrame': frame['source_frame_index'], 'objects': objects,
                                     'absentEntityIds': absent})
            observations += len(objects)
        if cap.read()[0]:
            raise ValueError('Clip contains frames without a native SAM output')
    finally:
        cap.release()
    result['provenance'] = {'rawSamSha256': sha(raw_path), 'clipSha256': sha(clip_path),
                            'poseModelSha256': sha(model_path) if has_people else None, 'samMethod': raw.get('method'),
                            'poseInputColor': 'RGB' if has_people else None, 'poseInputSize': [192, 256] if has_people else None,
                            'rtmlib': version('rtmlib') if has_people else None,
                            'onnxruntime': version('onnxruntime') if has_people else None,
                            'sourceManifestSha256': sha(source_manifest),
                            'samSessionId': session,
                            'sourceVideoSha256': manifest['sourceSha256'],
                            'sourceStartFrame': manifest['sourceStartFrame']}
    if reentry is not None:
        known = {o['nativeTrackId'] for frame in result['frames'] for o in frame['objects']}
        result['identityCandidates'] = []
        for edge in reentry['edges']:
            old, new = edge['to_earlier_tracklet_id'], edge['from_tracklet_id']
            indices = [e['frame_index'] for e in edge['evidence'] + edge['target_discovery_evidence']]
            if (type(old) is not int or type(new) is not int or old not in known or new not in known or old == new
                    or any(type(i) is not int or not 0 <= i < len(raw['frames']) for i in indices)
                    or edge['human_confirmed'] is not False or edge['type'] != 'session_reentry_candidate'):
                raise ValueError('Identity candidate is not an unconfirmed pair of observed tracks')
            result['identityCandidates'].append({'fromEntityId': f'{namespace}-{new}',
                'toEntityId': f'{namespace}-{old}', 'fromTrackId': new, 'toTrackId': old,
                'status': 'unconfirmed_reentry_candidate', 'cosine': edge['cosine'],
                'evidenceFrames': [raw['frames'][e['frame_index']]['source_frame_index'] for e in edge['evidence']],
                'referenceFrames': [raw['frames'][e['frame_index']]['source_frame_index'] for e in edge['target_discovery_evidence']]})
        result['provenance']['identityCandidateSha256'] = sha(reentry_candidates)
        result['limitations'].append('重新出现的外观关联仅是候选；未人工确认，不合并原始轨迹编号。')
    (output / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False))
    metrics = {'status': 'execution_complete', 'quality': 'not_independently_validated',
               'frames': len(spans), 'objectObservations': observations, 'personObservations': person_observations,
               'elapsedSeconds': round(time.monotonic() - started, 3),
               'analysisSha256': sha(output / 'analysis.json')}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    return metrics


def self_check() -> None:
    from ehs_spatial.providers.sam3 import encode_coco_rle
    from tempfile import TemporaryDirectory
    mask = np.array([[0, 1, 1], [0, 1, 0]], dtype=bool)
    out, box = object_input({'rle': encode_coco_rle(mask),
                             'box_xywh_normalized': [1/3, 0, 2/3, 1]}, 3, 2)
    assert np.array_equal(out, mask) and np.allclose(box, [1, 0, 3, 2])
    empty, box = object_input({'rle': encode_coco_rle(np.zeros((2, 3), bool)),
                               'box_xywh_normalized': [0, 0, 0, 0]}, 3, 2)
    assert not empty.any() and box is None
    raw = {'input': {'frame_count': 2, 'frame_timestamps_seconds': [0, .033],
                     'duration_seconds': .067},
           'frames': [{'frame_index': 0, 'timestamp_seconds': 0},
                      {'frame_index': 1, 'timestamp_seconds': .033}]}
    assert intervals(raw) == [(0, .033), (.033, .067)]
    raw['frames'].pop()
    try:
        intervals(raw)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing source frame was accepted')
    with TemporaryDirectory() as folder:
        source = Path(folder) / 'source.avi'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (16, 16))
        assert writer.isOpened()
        for i in range(4):
            writer.write(np.full((16, 16, 3), 20 * i, dtype=np.uint8))
        writer.release()
        raw = {'input': {'sha256': 'clip', 'frame_count': 2, 'width': 16, 'height': 16,
                         'frame_timestamps_seconds': [0, .1], 'duration_seconds': .2},
               'frames': [{'frame_index': i, 'source_frame_index': i + 1,
                           'timestamp_seconds': .1 * i} for i in range(2)]}
        manifest = {'clipSha256': 'clip', 'frameCount': 2, 'sourceVideo': str(source),
                    'sourceSha256': sha(source), 'sourceStartFrame': 1,
                    'sourceFrameMediaTimesSeconds': [.1, .2, .3]}
        assert source_spans(raw, manifest) == [(.1, .2), (.2, .3)]
        for change in [{'sourceStartFrame': 2}, {'sourceSha256': 'wrong'},
                       {'sourceFrameMediaTimesSeconds': [1.1, 1.2, 1.3]},
                       {'sourceFrameMediaTimesSeconds': [.1, .2, 1.3]}]:
            try:
                source_spans(raw, manifest | change)
            except ValueError:
                pass
            else:
                raise AssertionError('Incorrect parent time / source identity was accepted')
        full = {'input': {'sha256': sha(source), 'frame_count': 4, 'width': 16, 'height': 16,
                          'frame_timestamps_seconds': [0, .1, .2, .3], 'duration_seconds': .4},
                'frames': [{'frame_index': i, 'source_frame_index': i,
                            'timestamp_seconds': .1 * i} for i in range(4)]}
        parent = {'sourceVideo': str(source), 'sourceSha256': sha(source), 'clipSha256': sha(source),
                  'frameCount': 4, 'sourceStartFrame': 0,
                  'sourceFrameMediaTimesSeconds': [0, .1, .2, .3, .4]}
        assert source_spans(full, parent)[-1] == (.3, .4)
        try:
            source_spans(full, parent | {'sourceFrameMediaTimesSeconds': [0, .1, .2, .3, .5]})
        except ValueError:
            pass
        else:
            raise AssertionError('Full-video final interval extension was accepted')
        car_mask = encode_coco_rle(np.ones((16, 16), bool))
        full.update(session_id='check-car', identity_scope='video_session_only', method={'name': 'Self-check'})
        for frame in full['frames']:
            frame['objects'] = [{'track_id': 7, 'label': 'car', 'rle': car_mask,
                                 'box_xywh_normalized': [0, 0, 1, 1], 'score': None}]
        raw_path, parent_path = Path(folder) / 'raw.json', Path(folder) / 'parent.json'
        raw_path.write_text(json.dumps(full)); parent_path.write_text(json.dumps(parent))
        output = Path(folder) / 'preview'
        metrics = build(raw_path, source, None, output, parent_path)
        exported = json.loads((output / 'analysis.json').read_text())
        assert metrics['personObservations'] == 0 and metrics['objectObservations'] == 4
        assert exported['provenance']['poseModelSha256'] is None
        assert all(o['sourceLabel'] == 'car' and not o['keypoints'] and not o['bones']
                   and o['poseStatus'] == 'not_applicable_nonhuman'
                   for f in exported['frames'] for o in f['objects'])
    print('video pose input contract passed: RLE pixels, normalized xywh, time and missing frames')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path)
    p.add_argument('--clip', type=Path)
    p.add_argument('--pose-model', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--source-manifest', type=Path)
    p.add_argument('--reentry-candidates', type=Path)
    p.add_argument('--self-check', action='store_true')
    a = p.parse_args()
    if a.self_check:
        self_check()
    elif all([a.raw, a.clip, a.output, a.source_manifest]):
        print(json.dumps(build(a.raw, a.clip, a.pose_model, a.output, a.source_manifest, a.reentry_candidates)))
    else:
        p.error('--raw, --clip, --source-manifest and --output are required; person observations also require --pose-model')
