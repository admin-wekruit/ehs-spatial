"""Conversational refine: located box flows into refine_region."""
import numpy as np
import base64
import json
import shutil
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


def test_boxed_add_object_keeps_small_second_frame_evidence_without_vlm(tmp_path, monkeypatch):
    from ehs_spatial.agent_hub import agent_turn
    from ehs_spatial.object_evidence import build_object_evidence, load_candidate_mask
    import ehs_spatial.refine as refine

    run = _fake_run(tmp_path)
    shutil.copytree(run / "geometry/frames/frame_0001", run / "geometry/frames/frame_0002")
    image2 = run / "input/image_02.png"
    Image.new("RGB", (64, 48), (200, 0, 0)).save(image2)
    mask = np.zeros((48, 64), bool)
    mask[10:17, 20:27] = True  # 49 pixels fails measurement, not object accounting.
    calls = []

    def subscriber(endpoint, *, arguments):
        assert base64.b64decode(arguments["image_url"].split(",", 1)[1]) == image2.read_bytes()
        calls.append(endpoint)
        return {"rle": [_rle(mask)], "scores": [0.91]}

    monkeypatch.setattr(refine, "_default_subscriber", subscriber)
    monkeypatch.setattr("ehs_spatial.agent.GeminiAdapter", lambda: pytest.fail("boxed add called VLM"))
    request = dict(action="add_object", frame_id="frame_0002", box=[19, 9, 28, 18],
                   label="small switch", language="en", runs_root=tmp_path / "runs")
    out = agent_turn("r1", "This is a switch, not a fence", **request)
    assert out["intent"] == "add_object" and out["changed"]
    assert out["geometry_status"] == "unmeasured" and "49" in out["geometry_reason"]
    rows = json.loads((run / "refinements.json").read_text())
    assert len(rows) == 1 and rows[0]["frame_id"] == "frame_0002"
    assert "height_m" not in rows[0] and rows[0]["evidence_id"] == out["evidence_id"]
    assert rows[0]["instruction"] == "This is a switch, not a fence"
    registry = build_object_evidence(run)
    candidates = [o for o in registry["candidates"] if o["label"] == "small switch"]
    assert len(candidates) == 1 and candidates[0]["frame_id"] == "frame_0002"
    assert np.array_equal(load_candidate_mask(run, candidates[0]), mask)
    again = agent_turn("r1", "This is a switch, not a fence", **request)
    assert not again["changed"] and len(calls) == 1
    log = json.loads((run / "chat.jsonl").read_text().splitlines()[-1])
    assert log["language"] == "en" and log["evidence_id"] == out["evidence_id"]
    assert log["box"] == request["box"]
    Image.new("RGB", (64, 48), (0, 200, 0)).save(image2)
    changed_image = agent_turn("r1", "This is a switch, not a fence", **request)
    assert changed_image["evidence_id"] != out["evidence_id"] and len(calls) == 2


def test_description_add_object_locates_once_in_selected_frame_and_persists_rationale(tmp_path, monkeypatch):
    from ehs_spatial.agent_hub import agent_turn
    import ehs_spatial.refine as refine

    run = _fake_run(tmp_path)
    Image.new("RGB", (64, 48), (0, 200, 0)).save(run / "input/image_02.png")
    calls = []

    def locate(image_path, instruction, adapter):
        assert image_path.endswith("image_02.png")
        calls.append(image_path)
        return LocatedObject(found=True, label_en="novel object", box_2d=(100, 100, 600, 600), rationale="visible at the selected frame")

    monkeypatch.setattr("ehs_spatial.agent.GeminiAdapter", lambda: object())
    monkeypatch.setattr("ehs_spatial.agent._gemini_locator", locate)
    monkeypatch.setattr("ehs_spatial.agent._gemini_verify", lambda *a: pytest.fail("second VLM call"))
    monkeypatch.setattr(refine, "_default_subscriber", lambda *a, **kw: {"rle": [_rle(np.ones((48, 64), bool))], "scores": [0.9]})
    out = agent_turn("r1", "Add the small object near the edge", runs_root=tmp_path / "runs",
                     action="add_object", frame_id="frame_0002")
    assert len(calls) == 1 and out["geometry_status"] == "unmeasured"
    row = json.loads((run / "refinements.json").read_text())[0]
    assert row["located_rationale"] == "visible at the selected frame"
    assert "geometry for frame_0002 is missing" in row["geometry_reason"]
