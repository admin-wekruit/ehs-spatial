"""Offline CLI check: default SAM3 discovery preserves masks/provenance without fal."""
import contextlib
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import discover_video_keyframes as discovery
from modal_apps import sam3_video_fal
from run_discovered_video import decoded_clip, discoveries

with TemporaryDirectory() as directory:
    root = Path(directory)
    video = root / 'input.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 24))
    assert writer.isOpened()
    for value in (40, 160):
        writer.write(np.full((24, 32, 3), value, np.uint8))
    writer.release()
    decoded_dir = root / 'decoded'
    decoded_dir.mkdir()
    clip, decoded = decoded_clip(video, decoded_dir)
    mask = np.zeros((24, 32), bool)
    mask[3:9, 5:12] = True
    detected = {'rle': [discovery.sam3_app._encode_coco_rle(mask)], 'scores': [.9]}
    empty = {'rle': [], 'scores': []}
    app = SimpleNamespace(name='sam3-inference', run=Mock(side_effect=contextlib.nullcontext))

    for prompt in ('person', 'basket'):
        output = root / prompt
        remote = Mock(side_effect=[[detected], [empty]])
        service = SimpleNamespace(segment=SimpleNamespace(remote=remote))
        argv = [str(ROOT / 'scripts/discover_video_keyframes.py'), '--video', str(video),
                '--frames', '0', '1', '--output', str(output)]
        if prompt != 'person':
            argv += ['--prompt', prompt]
        with patch.object(discovery.sam3_app, 'app', app), \
             patch.object(discovery.sam3_app, 'Sam3', return_value=service) as constructor, \
             patch.object(sam3_video_fal, 'execute', side_effect=AssertionError('fal called')) as fal, \
             patch.object(sys, 'argv', argv):
            runpy.run_path(argv[0], run_name='__main__')
        constructor.assert_called_once_with()
        fal.assert_not_called()
        assert remote.call_count == 2
        found = discoveries([output], clip, decoded, step=1, prompt=prompt)
        assert np.array_equal(found[0]['masks'][0], mask) and found[1]['masks'] == []
        assert [x['instances'] for x in json.loads((output / 'manifest.json').read_text())] == [1, 0]
        for index, call in enumerate(remote.call_args_list):
            folder = output / f'frame-{index:05d}'
            assert call.args == ((folder / f'frame-{index}.png').read_bytes(), [{'text': prompt}])
            manifest = json.loads((folder / 'input-manifest.json').read_text())
            provider = json.loads((folder / 'provider-output.json').read_text())
            event = json.loads((folder / 'provider-events.jsonl').read_text())
            assert manifest['provider'] == 'self-hosted'
            assert provider['provider'] == 'self-hosted facebook/sam3'
            assert event == {'phase': 'self_hosted', 'data': {'app': app.name, 'model': 'facebook/sam3', 'gpu': 'L4'}}

    failure = {**empty, 'error': 'mock inference failure'}
    service.segment.remote = Mock(return_value=[failure])
    output = root / 'failed'
    with patch.object(discovery.sam3_app, 'app', app), \
         patch.object(discovery.sam3_app, 'Sam3', return_value=service), \
         patch.object(discovery, 'execute', side_effect=AssertionError('fal fallback called')) as fal:
        try:
            discovery.run(SimpleNamespace(video=video, frames=[0], output=output,
                                          prompt='person', provider='self-hosted'))
        except RuntimeError as error:
            assert str(error) == 'SAM3 discovery failed: mock inference failure'
        else:
            raise AssertionError('Provider failure recorded as successful empty discovery')
    fal.assert_not_called()
    assert json.loads((output / 'frame-00000/provider-output.json').read_text())['error'] == failure['error']
    assert not (output / 'frame-00000/instances.json').exists()
    assert not (output / 'manifest.json').exists()

print('PASS: self-hosted CLI defaults, exact masks/source/provider provenance, empty detections, errors without fal fallback')
