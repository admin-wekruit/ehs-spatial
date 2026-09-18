"""Run directly: empty detections are valid; a batch reuses one fresh price quote."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import discover_video_keyframes as discovery

with TemporaryDirectory() as directory:
    root = Path(directory)
    video = root / 'input.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 24))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((24, 32, 3), np.uint8))
    writer.release()
    calls = []
    pricing = {'prices': [{'endpoint_id': 'fal-ai/sam-3-1/image-rle', 'unit_price': .01,
                           'unit': 'units', 'currency': 'USD'}]}

    def provider(payload, output, log_name):
        calls.append(payload)
        assert payload['max_fal_usd'] == .02
        (output / log_name).write_text(json.dumps({'phase': 'pricing', 'data': pricing}) + '\n')
        (output / 'provider-output.json').write_text('{"rle": []}')

    with patch.object(discovery, 'execute', provider):
        discovery.run(SimpleNamespace(video=video, frames=[0, 1, 2], output=root / 'out', prompt='person'))
    assert 'batch_pricing_quote' not in calls[0]
    assert calls[1]['batch_pricing_quote'] == calls[2]['batch_pricing_quote']
    assert calls[1]['batch_pricing_quote']['pricing'] == pricing
    assert all(x['instances'] == 0 for x in json.loads((root / 'out/manifest.json').read_text()))
print('PASS: one fresh quote, unchanged per-call cap, completed empty detections retained')
