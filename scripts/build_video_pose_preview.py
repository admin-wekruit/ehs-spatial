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
    mask = decode_coco_rle(obj['rle']).astype(bool)
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
        for i in range(start + len(spans) + 1):
            ok, image = cap.read()
            if not ok:
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


def build(raw_path: Path, clip_path: Path, model_path: Path, output: Path,
          source_manifest: Path) -> dict:
    from rtmlib import RTMPose
    from importlib.metadata import version

    started = time.monotonic()
    raw = json.loads(raw_path.read_text())
    if raw['input']['sha256'] != sha(clip_path):
        raise ValueError('Clip hash differs from the actual SAM input')
    manifest = json.loads(source_manifest.read_text())
    spans = source_spans(raw, manifest)
    width, height = raw['input']['width'], raw['input']['height']
    output.mkdir(parents=True, exist_ok=False)
    (output / 'masks').mkdir()
    pose = RTMPose(str(model_path), model_input_size=(192, 256),
                   to_openpose=False, backend='onnxruntime', device='cpu')
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError('Cannot decode the SAM input clip')
    session = raw['session_id']
    if not isinstance(session, str) or not session.strip():
        raise ValueError('A native session or provider request ID is required')
    namespace = 'sam-' + session
    result = {'version': 1, 'coordinateSpace': 'source_pixels', 'width': width,
              'height': height, 'method': 'SAM native video IDs + RTMPose COCO17 (CPU ONNX)',
              'identityScope': 'video_session_only', 'frames': [],
              'limitations': ['仅分析保存的连续短片，其余时间没有观测。',
                              'ID 是本次视频会话中的短期轨迹，未验证跨遮挡或跨视频持久身份。',
                              '骨架为二维图像估计；尚无三维人体、世界运动或动态对象模型。',
                              '关节分数是模型响应，0.3 仅为显示阈值，不是校准后的正确概率。']}
    observations = 0
    try:
        for frame, (start, end) in zip(raw['frames'], spans, strict=True):
            ok, bgr = cap.read()
            if not ok or bgr.shape[:2] != (height, width):
                raise ValueError('Decoded clip does not match the native frame domain')
            inputs = [object_input(o, width, height) for o in frame['objects']]
            ids = [o['track_id'] for o in frame['objects']]
            if len(ids) != len(set(ids)) or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError('Native track IDs must be unique nonnegative integers')
            objects = []
            # ponytail: top-down pose uses SAM boxes directly; no duplicate tracker.
            eligible = [i for i, (_, box) in enumerate(inputs) if box is not None]
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
                    if obj['label'] != 'person':
                        raise ValueError('Human pose cannot be applied to non-person observations')
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
                                    'nativeTrackId': obj['track_id'], 'label': '人 · 短期轨迹',
                                    'confidence': obj['score'], 'maskUrl': name,
                                    'keypoints': joints, 'bones': BONES if joints else [],
                                    'poseStatus': 'estimated_2d' if joints else 'insufficient_mask_support',
                                    'rawKeypointScores': [float(c) if np.isfinite(c) else None for c in confidence]}
                    if box is not None:
                        exported['bbox'] = box
                    objects.append(exported)
            result['frames'].append({'timeSec': start,
                                     'endTimeSec': end,
                                     'sourceFrame': frame['source_frame_index'], 'objects': objects})
            observations += len(objects)
        if cap.read()[0]:
            raise ValueError('Clip contains frames without a native SAM output')
    finally:
        cap.release()
    result['provenance'] = {'rawSamSha256': sha(raw_path), 'clipSha256': sha(clip_path),
                            'poseModelSha256': sha(model_path), 'samMethod': raw.get('method'),
                            'poseInputColor': 'RGB', 'poseInputSize': [192, 256],
                            'rtmlib': version('rtmlib'), 'onnxruntime': version('onnxruntime'),
                            'sourceManifestSha256': sha(source_manifest),
                            'samSessionId': session,
                            'sourceVideoSha256': manifest['sourceSha256'],
                            'sourceStartFrame': manifest['sourceStartFrame']}
    (output / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False))
    metrics = {'status': 'execution_complete', 'quality': 'not_independently_validated',
               'frames': len(spans), 'personObservations': observations,
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
    print('video pose input contract passed: RLE pixels, normalized xywh, time and missing frames')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path)
    p.add_argument('--clip', type=Path)
    p.add_argument('--pose-model', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--source-manifest', type=Path)
    p.add_argument('--self-check', action='store_true')
    a = p.parse_args()
    if a.self_check:
        self_check()
    elif all([a.raw, a.clip, a.pose_model, a.output, a.source_manifest]):
        print(json.dumps(build(a.raw, a.clip, a.pose_model, a.output, a.source_manifest)))
    else:
        p.error('--raw, --clip, --pose-model, --source-manifest and --output are required')
