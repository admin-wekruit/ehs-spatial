"""Offline fixed-camera video -> flow residual -> seeds -> tracker, with GPU boundaries mocked."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'modal_apps'))
import motion_masks as motion
import sam3_motion_tracks as tracking

flow = np.zeros((24, 32, 2), float)
flow[5:12, 8:16] = [3, 4]
assert np.array_equal(motion.residual(flow, fixed_camera=True), np.linalg.norm(flow, axis=-1))

with TemporaryDirectory() as directory:
    root = Path(directory)
    video = root / 'input.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 10, (854, 482))
    assert writer.isOpened()
    y, x = np.indices((482, 854))
    originals = [np.stack([(x + n * 7) % 256, y % 256, (x + y) % 256], -1).astype(np.uint8) for n in range(4)]
    for frame in originals:
        writer.write(frame)
    writer.release()
    frames, source = motion.fixed_video_frames(video)
    assert len(frames) == source['frame_count'] == 4 and source['fps'] == 10
    assert source['sha256'] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert source['source_size_wh'] == [854, 482]
    assert source['resized_size_wh'] == [640, 361] and source['raster_size_wh'] == [640, 368]
    assert source['resize_factors_xy'] == [640 / 854, 361 / 482]
    assert source['padding_ltrb'] == [0, 0, 0, 7]
    assert source['raster_sha256'] == hashlib.sha256(b''.join(f.tobytes() for f in frames)).hexdigest()
    for original, actual in zip(originals, frames):
        expected = cv2.copyMakeBorder(cv2.resize(original, (640, 361), interpolation=cv2.INTER_AREA), 0, 7, 0, 0, cv2.BORDER_REPLICATE)
        assert np.array_equal(actual, expected), 'preserve aspect ratio with padding, no stretched raster'
    region = np.zeros(frames[0].shape[:2], bool)
    region[60:180, 180:300] = True
    jpeg_by_frame = {}

    def flow_remote(pairs):
        answers = []
        for index, first, second in pairs:
            jpeg_by_frame[index], jpeg_by_frame[index + 1] = first, second
            flow = np.zeros((*region.shape, 2), np.float16)
            flow[region] = [3, 4]
            flow[-7:] = [3, 4]  # padding is never evidence
            flow[:8, :8] = [-20, 0]  # endpoints outside the source content are never evidence
            flow[200:240, 200:240] = [3, 4]
            cycle = np.zeros(region.shape, np.float16)
            cycle[200:240, 200:240] = motion.CYCLE_PX + 1
            buffer = io.BytesIO()
            np.savez_compressed(buffer, flow=flow, cycle=cycle)
            answers.append((index, buffer.getvalue()))
        return answers

    def boxes(requests, work_width):
        assert work_width == 640
        assert all(frame.shape[:2] == region.shape and box == [180, 60, 300, 180] for frame, box in requests.values())
        return {key: (region.copy(), .9) for key in requests}

    def track_remote(jpegs, seeds, text):
        assert jpegs == [jpeg_by_frame[n] for n in range(1, 4)], 'tracker must use the flow raster and exact source frame order'
        assert [s[0] for s in seeds] == [0, 1] and text == ''
        report = {'stages': {'motion': {'shape': list(region.shape)}}}
        buffer = io.BytesIO()
        np.savez_compressed(buffer, report=json.dumps(report), **{f'motion/{n}/1': np.packbits(region) for n in range(3)})
        return buffer.getvalue()

    app = SimpleNamespace(run=contextlib.nullcontext)
    motion_args = SimpleNamespace(droid_run=None, video=video, fixed_camera=True, output=root / 'motion', every=1, gap=1, reference=None)
    track_args = SimpleNamespace(droid_run=None, video=video, fixed_camera=True, output=root / 'tracks', motion=motion_args.output, frames=[1, 4], text='', reference=None)
    with patch.object(motion, 'app', app), patch.object(motion, 'flow_remote', SimpleNamespace(remote=flow_remote)), \
         patch.object(motion.sam2_everything, 'box_masks', boxes), patch.object(tracking, 'app', app), \
         patch.object(tracking, 'track_remote', SimpleNamespace(remote=track_remote)):
        motion.run(motion_args)
        tracking.run(track_args)
    motion_state = json.loads((motion_args.output / 'motion.json').read_text())
    track_state = json.loads((track_args.output / 'tracks.json').read_text())
    assert motion_state['source'] == track_state['source'] == source
    assert motion_state['cameras'] is None and track_state['cameras'] is None
    assert motion_state['retries'] == track_state['retries'] == 0
    assert track_state['objects']['motion'][0]['moving_share_median'] == 1
    with np.load(motion_args.output / '00001-residual.npz') as saved:
        assert np.array_equal(saved['trusted_moving'], region), 'padding, out-of-content endpoints and bad cycles must not vote moving'
    larger_track = region.copy(); larger_track[200:240, 200:240] = True
    assert tracking.moving_share({0: larger_track}, 1, motion_args.output) == region.sum() / larger_track.sum(), 'untrusted track pixels stay in denominator'
    assert sorted(p.name for p in (track_args.output / 'motion').glob('*.png')) == ['00001.png', '00002.png', '00003.png']
    assert all(np.array_equal(cv2.imread(str(p), 0) > 0, region) for p in (track_args.output / 'motion').glob('*.png'))

    def rejected(run, args, message):
        try:
            run(args)
        except ValueError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError('invalid source accepted')

    with patch.object(motion, 'flow_remote', SimpleNamespace(remote=lambda _: (_ for _ in ()).throw(AssertionError('paid flow')))), \
         patch.object(tracking, 'track_remote', SimpleNamespace(remote=lambda *_: (_ for _ in ()).throw(AssertionError('paid tracker')))):
        for run, args in ((motion.run, motion_args), (tracking.run, track_args)):
            rejected(run, SimpleNamespace(**(vars(args) | {'fixed_camera': False})), '--fixed-camera')
            rejected(run, SimpleNamespace(**(vars(args) | {'droid_run': root})), '--droid-run')
            rejected(run, SimpleNamespace(**(vars(args) | {'reference': root})), '--reference')
        for field, wrong in (('sha256', 'wrong'), ('raster_size_wh', [640, 480]), ('raster_sha256', 'wrong')):
            changed = {**motion_state, 'source': {**source, field: wrong}}
            (motion_args.output / 'motion.json').write_text(json.dumps(changed))
            rejected(tracking.run, track_args, 'source')
        (motion_args.output / 'motion.json').write_text(json.dumps(motion_state))
        rejected(tracking.run, SimpleNamespace(**(vars(track_args) | {'frames': [0, 5]})), 'frames')

    for name, stage in (('model-error', {'error': 'mock failure'}), ('no-tracks', {})):
        output = root / name
        def failed_remote(*_):
            buffer = io.BytesIO()
            np.savez_compressed(buffer, report=json.dumps({'stages': {'motion': stage}}))
            return buffer.getvalue()
        with patch.object(tracking, 'app', app), patch.object(tracking, 'track_remote', SimpleNamespace(remote=failed_remote)):
            tracking.run(SimpleNamespace(**(vars(track_args) | {'output': output})))
        assert json.loads((output / 'tracks.json').read_text())['status'] != 'complete'

    for index in range(3):
        np.savez_compressed(motion_args.output / f'{index:05d}-residual.npz', residual=region * 5, floor=2)
    assert tracking.moving_share({0: region}, 1, motion_args.output) is None, 'legacy residual without cycle validity is unverified'
    tracking.run(track_args)  # read cached raw model output without a new GPU call
    unverified = json.loads((track_args.output / 'tracks.json').read_text())
    assert unverified['status'] != 'complete'
    assert not unverified['objects']['motion'][0]['kept'] and unverified['objects']['motion'][0]['reason'] == 'no_trusted_motion_samples'

    with patch.object(motion, 'MAX_VIDEO_BYTES', frames[0].nbytes):
        rejected(motion.fixed_video_frames, video, 'memory')
    real_capture = cv2.VideoCapture
    def truncated_capture(path):
        cap = real_capture(path)
        return SimpleNamespace(isOpened=cap.isOpened, read=cap.read, release=cap.release,
                               get=lambda key: 5 if key == cv2.CAP_PROP_FRAME_COUNT else cap.get(key))
    with patch.object(cv2, 'VideoCapture', truncated_capture):
        rejected(motion.fixed_video_frames, video, 'frame count')

print('PASS: fixed-camera residual, aspect/source provenance, shared flow/tracker raster and indices, source mismatches rejected before GPU')
