"""The agent hub: five operator verbs, deterministic routing, every turn
logged, edits written back and policies re-evaluated."""

import json
from pathlib import Path

from ehs_spatial.agent_hub import (
    agent_turn,
    chat_history,
    classify_intent,
    parse_policy_adjust,
    parse_relabel,
)


def test_intent_routing_is_deterministic():
    assert classify_intent("围栏离机器人多远？") == "ask"
    assert classify_intent("右边黄色柱子帮我量一下") == "refine"
    assert classify_intent("机器人后面还有一台设备没识别") == "refine"
    assert classify_intent("这块不是围栏，是导向挡板") == "relabel"
    assert classify_intent("p01 改成 0.8m") == "policy_adjust"
    assert classify_intent("这个 cell 用 0.8 米 间距规则") == "policy_adjust"


def test_parse_relabel_maps_operator_words_to_policy_labels():
    assert parse_relabel("这块不是围栏，是导向挡板") == ("safety fence", "sloped surface")
    assert parse_relabel("把小车改成托盘") == ("material cart", "pallet")
    assert parse_relabel("随便说点什么") is None


def test_parse_policy_adjust():
    assert parse_policy_adjust("p01 改成 0.8m") == ("p01", 0.8)
    assert parse_policy_adjust("走道规则用 1.2 米") == ("p03", 1.2)
    assert parse_policy_adjust("阈值 0.5m") == (None, 0.5)


def _synthetic_run(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    scene = {
        "run_id": "r1",
        "entities": [
            {
                "entity_id": "e1", "label": "safety fence",
                "observation_ids": ["o1"], "centroid_xyz": [0, 2, 0.5],
                "footprint_xy": [[-1, 2], [1, 2], [1, 2.1], [-1, 2.1]],
                "height_m": 1.1, "evidence_frame_ids": ["frame_0001"],
            },
            {
                "entity_id": "e2", "label": "material cart",
                "observation_ids": ["o2"], "centroid_xyz": [0, 2.3, 0.5],
                "footprint_xy": [[-0.5, 2.15], [0.5, 2.15], [0.5, 2.6], [-0.5, 2.6]],
                "height_m": 1.0, "evidence_frame_ids": ["frame_0001"],
            },
        ],
        "warnings": [],
        "facts": [],
        "fence_polygon": [[-1, 2], [1, 2], [1, 2.1], [-1, 2.1]],
        "floor_plane": [0.0, 0.0, 1.0, 0.0],
        "scale_source": "moge_anchor",
        "scale_factor": 1.0,
        "scale_confidence": 0.9,
    }
    (run / "scene.json").write_text(json.dumps(scene))
    spec = {
        "policy_id": "p01-movable-equipment-clearance",
        "source_text": "keep 0.6 m clear",
        "predicate": "min_separation",
        "subject_labels": ["material cart"],
        "object_labels": ["safety fence"],
        "threshold": 0.6, "unit": "m", "severity": "major",
    }
    (run / "policies.json").write_text(json.dumps({"specs": [spec], "results": []}))
    return run


def test_relabel_rewrites_scene_and_reevaluates(tmp_path):
    run = _synthetic_run(tmp_path)
    out = agent_turn("r1", "这块不是围栏，是导向挡板", runs_root=tmp_path / "runs")
    assert out["intent"] == "relabel" and out["changed"]
    scene = json.loads((run / "scene.json").read_text())
    assert [e["label"] for e in scene["entities"]] == ["sloped surface", "material cart"]
    results = json.loads((run / "policies.json").read_text())["results"]
    assert results and results[0]["policy_id"].startswith("p01")
    # the fence is gone, so the clearance rule has nothing to measure against
    assert results[0]["status"] != "FAIL"
    log = chat_history(run)
    assert log[0]["content"].startswith("[relabel]") and log[1]["role"] == "assistant"


def test_policy_adjust_is_per_run_and_reversible(tmp_path):
    run = _synthetic_run(tmp_path)
    out = agent_turn("r1", "p01 改成 0.2m", runs_root=tmp_path / "runs")
    assert out["intent"] == "policy_adjust" and out["changed"]
    envelope = json.loads((run / "policies.json").read_text())
    assert envelope["specs"][0]["threshold"] == 0.2
    assert envelope["specs"][0]["adjusted_for_run"] is True
    assert (run / "policies.original.json").exists()
    original = json.loads((run / "policies.original.json").read_text())
    assert original["specs"][0]["threshold"] == 0.6


def test_ask_and_refine_delegate_and_log(tmp_path):
    run = _synthetic_run(tmp_path)
    seen = {}
    out = agent_turn(
        "r1", "围栏离小车多远？", runs_root=tmp_path / "runs",
        answer_fn=lambda q: seen.setdefault("q", q) and "0.05 m [fact-1]",
    )
    assert out["intent"] == "ask" and "fact-1" in out["reply"]
    out2 = agent_turn(
        "r1", "右边黄色柱子帮我量", runs_root=tmp_path / "runs",
        refine_fn=lambda rid, msg, apply: {
            "label": "yellow post", "height_m": 2.1, "extent_m": "0.6x1.1",
            "camera_dist_m": 2.5, "sam_score": 0.97,
        },
    )
    assert out2["intent"] == "refine" and "yellow post" in out2["reply"]
    assert len(chat_history(run)) == 4


def test_preview_mode_changes_nothing(tmp_path):
    run = _synthetic_run(tmp_path)
    before = (run / "scene.json").read_text()
    out = agent_turn("r1", "这块不是围栏，是导向挡板", runs_root=tmp_path / "runs", apply=False)
    assert not out["changed"] and (run / "scene.json").read_text() == before
