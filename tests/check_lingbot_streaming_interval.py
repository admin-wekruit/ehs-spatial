"""Run directly: long videos must use the pinned official demo's cache interval."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_apps import lingbot_room

assert callable(getattr(lingbot_room, 'streaming_interval', None)), 'Official automatic keyframe selection is missing'
for frames, expected in [(8, 1), (273, 1), (320, 1), (321, 2), (640, 2), (681, 3), (768, 3)]:
    assert lingbot_room.streaming_interval(frames) == expected, (frames, expected)
for invalid in [0, -1, 769, True, 8.5]:
    try:
        lingbot_room.streaming_interval(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f'Invalid bounded input accepted: {invalid}')
print('PASS: official streaming keyframe schedule and bounded inputs')
