"""lab/labels.py: two reviewers' label files -> gold (agreeing rows merged, the conflicting key left out and listed), gold_in rows
carried over, declared inputs extracted, the CLI; L1 scene-json `declared` cfg merges them into Scene.declared_inputs and the
output is byte-identical without it."""
import json
from pathlib import Path

from ehs_spatial.verdict.contracts import Scene
from ehs_spatial.verdict.lab import labels
from ehs_spatial.verdict.layers.l1_scene.scene_json import SceneJson

SCENE = Path(__file__).resolve().parents[2] / "ehs_spatial/verdict/benchmark/v0/scenes/scene-090.json"


def label_file(path, reviewer, rows, declared):
    path.write_text(json.dumps({"schema": "verdict-labels/1", "run_id": "r1", "scene_id": "090-a9a6e0a0", "benchmark": "v0/090", "reviewer": reviewer,
                                "rule_pack": "handwritten-trial@0", "labels": rows, "declared_inputs": declared}))
    return path


def row(rule, subjects, status, reason=""):
    return {"rule_id": rule, "rule_version": "0", "subjects": subjects, "labels": [], "status": status, "reason": reason, "confidence": "sure", "remeasure": False}


def test_merge_two_reviewers_one_conflict(tmp_path):
    a = label_file(tmp_path / "a.json", "ann", [row("fence_height", ["x"], "PASS", "2.0 m"), row("floor_gap", ["y"], "FAIL"), row("crush_gap", ["x", "y"], "NOT_APPLICABLE")],
                   {"stop_time_ms": 500})
    b = label_file(tmp_path / "b.json", "bob", [row("fence_height", ["x"], "PASS"), row("floor_gap", ["y"], "PASS")], {"risk_level": "high"})
    old = {"version": "v0", "provisional": True, "source": "trial", "verdicts": [
        {"scene_id": "090-a9a6e0a0", "rule_id": "floor_gap", "subjects": ["y"], "labels": [], "status": "PASS"},
        {"scene_id": "090-a9a6e0a0", "rule_id": "enclosure", "subjects": ["hazard_zone"], "labels": [], "status": "CANNOT_DETERMINE"}]}
    gold = labels.merge_gold([a, b], old)
    by = {(g["rule_id"], tuple(g["subjects"])): g for g in gold["verdicts"]}
    assert set(by) == {("fence_height", ("x",)), ("crush_gap", ("x", "y")), ("enclosure", ("hazard_zone",))}   # floor_gap: conflict -> out, old row too
    assert by[("fence_height", ("x",))]["reviewer"] == "ann, bob" and by[("fence_height", ("x",))]["reason"] == "2.0 m"
    assert by[("crush_gap", ("x", "y"))]["status"] == "NOT_APPLICABLE"
    assert [(c["reviewer"], c["status"]) for c in gold["conflicts"]] == [("ann", "FAIL"), ("bob", "PASS")]
    assert gold["version"] == "v1" and gold["provisional"] is True and "1 rows still from: trial" in gold["source"]
    fresh = labels.merge_gold([a, b])
    assert fresh["provisional"] is False and fresh["source"] == "human labels" and fresh["version"] == "v1"
    assert labels.declared_inputs([a, b]) == {"090-a9a6e0a0": {"stop_time_ms": 500, "risk_level": "high"}}


def test_cli_merge_and_declared(tmp_path, capsys):
    a = label_file(tmp_path / "a.json", "ann", [row("fence_height", ["x"], "PASS")], {"stop_time_ms": 500})
    labels.main(["merge", "--labels", str(a), "--out", str(tmp_path / "gold.json"), "--version", "v2"])
    labels.main(["declared", "--labels", str(a), "--out", str(tmp_path / "declared.json")])
    assert json.loads((tmp_path / "gold.json").read_text())["verdicts"][0]["status"] == "PASS"
    assert json.loads((tmp_path / "declared.json").read_text()) == {"090-a9a6e0a0": {"stop_time_ms": 500}}
    assert "benchmark/v2/gold.json" in capsys.readouterr().out


def test_scene_json_declared_cfg(tmp_path):
    plain = SceneJson().run({"source": SCENE}, {}, tmp_path)["scene"]
    assert plain.model_dump_json(indent=1) == Scene.load(SCENE).model_dump_json(indent=1)
    dec = tmp_path / "declared.json"
    dec.write_text(json.dumps({"090-a9a6e0a0": {"stop_time_ms": 500, "risk_level": "high"}, "other": {"stop_time_ms": 1}}))
    merged = SceneJson().run({"source": SCENE}, {"declared": str(dec)}, tmp_path)["scene"]
    assert merged.declared_inputs == {**plain.declared_inputs, "stop_time_ms": 500, "risk_level": "high"}
    assert merged.model_dump(exclude={"declared_inputs"}) == plain.model_dump(exclude={"declared_inputs"})
