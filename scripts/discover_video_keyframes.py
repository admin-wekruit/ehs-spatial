"""Run existing SAM3 image discovery on explicitly selected source video frames.

Each call has its own immutable input/results directory and recorded cost. This
produces independent observations, never an identity assignment or a new tracker.
"""
import argparse
import contextlib
import base64
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehs_spatial.providers.sam3 import decode_coco_rle
from modal_apps.sam3_video_fal import execute
import modal_apps.sam3_app as sam3_app
from build_video_pose_preview import sha


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    selected = sorted(set(args.frames))
    if not selected or selected[0] < 0 or len(selected) != len(args.frames) or len(selected) > 12:
        raise ValueError('Choose up to twelve distinct nonnegative frame indices')
    cap = cv2.VideoCapture(str(args.video))
    video_sha = sha(args.video)
    done = []
    quote = None
    service = sam3_app.Sam3() if args.provider == 'self-hosted' else None  # one warm L4 for every frame; the response keeps fal's shape
    with (sam3_app.app.run() if service else contextlib.nullcontext()):
        try:
            for index in range(selected[-1] + 1):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError('Selected frame is absent from the input video')
                if index not in selected:
                    continue
                folder = args.output / f'frame-{index:05d}'
                folder.mkdir()
                image = folder / f'frame-{index}.png'
                if not cv2.imwrite(str(image), bgr):
                    raise IOError('Could not save the exact decoded source frame')
                height, width = bgr.shape[:2]
                manifest = {'source_clip': str(args.video.resolve()), 'source_clip_sha256': video_sha,
                            'source_frame_index': index, 'frame_sha256': sha(image),
                            'timestamp_seconds': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                            'width': width, 'height': height, 'prompt': args.prompt}
                (folder / 'input-manifest.json').write_text(json.dumps(manifest, indent=2))
                payload = {'mode': 'submit', 'endpoint': 'fal-ai/sam-3-1/image-rle',
                         'billing_units': 1, 'max_fal_usd': .02,
                         'input': {'image_url': 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode(),
                                   'prompt': args.prompt, 'return_multiple_masks': True,
                                   'include_scores': True, 'include_boxes': True}}
                if quote and time.time() - quote['fetched_at'] < 500:
                    payload['batch_pricing_quote'] = quote
                if service:
                    response = service.segment.remote(image.read_bytes(), [{'text': args.prompt}])[0]
                    (folder / 'provider-events.jsonl').write_text(json.dumps({'phase': 'self_hosted', 'data': {'app': sam3_app.app.name, 'model': sam3_app.MODEL_ID, 'gpu': 'L4'}}) + '\n')
                    (folder / 'provider-output.json').write_text(json.dumps({**response, 'provider': 'self-hosted ' + sam3_app.MODEL_ID}))
                else:
                    execute(payload, folder, 'provider-events.jsonl')
                if not service and 'batch_pricing_quote' not in payload:
                    for line in (folder / 'provider-events.jsonl').read_text().splitlines():
                        event = json.loads(line)
                        if event['phase'] == 'pricing':
                            quote = {'pricing': event['data'], 'fetched_at': time.time()}
                data = json.loads((folder / 'provider-output.json').read_text())
                instances = []
                for ordinal, rle in enumerate(data['rle']):
                    mask = decode_coco_rle(rle, height=height, width=width).astype(bool)
                    if mask.shape != (height, width) or not mask.any():
                        raise ValueError('Expected a nonempty source-domain instance mask')
                    y, x = np.where(mask)
                    path = folder / f'instance-{ordinal}-mask.png'
                    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
                        raise IOError('Could not save the instance mask')
                    instances.append({'instance_index': ordinal, 'label': args.prompt,
                                      'mask_area_pixels': int(mask.sum()), 'mask_sha256': sha(path),
                                      'mask_bounds_xyxy_exclusive': [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)]})
                (folder / 'instances.json').write_text(json.dumps(instances, indent=2))
                done.append({'frame': index, 'instances': len(instances), 'directory': str(folder)})
                (args.output / 'manifest.json').write_text(json.dumps(done, indent=2))
                print(json.dumps(done[-1]), flush=True)
        finally:
            cap.release()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--frames', type=int, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prompt', default='person')
    p.add_argument('--provider', choices=['fal', 'self-hosted'], default='fal', help='self-hosted: modal_apps/sam3_app.py (facebook/sam3 on our own L4) instead of the fal endpoint')
    run(p.parse_args())
