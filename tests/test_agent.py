"""Conversational refine: located box flows into refine_region."""
import numpy as np
import pytest
from PIL import Image

from ehs_spatial.agent import LocatedObject, agent_refine
from ehs_spatial.refine import RefineError
import importlib.util as _ilu
from pathlib import Path as _P
_spec = _ilu.spec_from_file_location(
    "_test_refine_helpers", _P(__file__).parent / "test_refine.py"
)
_helpers = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_fake_run, _rle = _helpers._fake_run, _helpers._rle


def test_agent_refine_locates_and_measures(tmp_path):
    _fake_run(tmp_path)
    W, H = 64, 48
    mask = np.zeros((H, W), bool)
    mask[10:40, 20:30] = True

    def locator(image_path, instruction):
        assert "透明" in instruction
        return LocatedObject(
            found=True, label_en="safety fence",
            box_2d=(160, 280, 880, 500), rationale="the glazed panel",
        )

    def subscriber(endpoint, *, arguments):
        b = arguments["box_prompts"][0]
        assert b["x_min"] == int(280 / 1000 * W)
        return {"rle": [_rle(mask)], "scores": [0.88]}

    result = agent_refine(
        "r1", "左边那排透明护板是围栏", runs_root=tmp_path / "runs",
        apply=False, locator=locator, subscriber=subscriber,
    )
    assert result["label"] == "safety fence"
    assert result["sam_score"] == 0.88
    assert result["instruction"].startswith("左边")


def test_agent_refine_honest_when_not_found(tmp_path):
    _fake_run(tmp_path)

    def locator(image_path, instruction):
        return LocatedObject(
            found=False, label_en="", box_2d=(0, 0, 1, 1),
            rationale="nothing matches",
        )

    with pytest.raises(RefineError):
        agent_refine(
            "r1", "看不见的东西", runs_root=tmp_path / "runs", locator=locator
        )
