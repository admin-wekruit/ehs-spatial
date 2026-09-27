"""The offline scripts' own self-checks, run by pytest: a regression in the cut, merge, free-space, live map or replay
stream rules turns the suite red instead of waiting for someone to run the script by hand."""

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


@pytest.mark.parametrize("script", ["detect_shot_cuts", "merge_object_models", "box_free_space", "live_map"])
def test_self_check(script):
    importlib.import_module(script).self_check()


def test_replay_people_stream_self_check():
    replay = importlib.import_module("replay_people_stream")
    if not replay.DEPTH.exists():  # it replays research-notes data that only this Mac holds
        pytest.skip(f"no replay data at {replay.PHASE2}")
    replay.self_check()
