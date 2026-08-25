"""scripts/policy_compile.py writes the production policies.json envelope.

Review finding: the script used to write a third, incompatible report shape
to a path nothing reads, so compiled policy verdicts never reached the app
history card or report.py. These tests pin the repaired contract: evaluating
against a run writes {"specs", "results"} to runs/<id>/policies.json, the
exact file and shape every reader consumes.
"""

import importlib.util
import json
from pathlib import Path

from ehs_spatial.contracts import Entity3D, SceneMap
from ehs_spatial.policy import PolicySpec, Predicate
from ehs_spatial.report import _policy_rows


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "policy_compile.py"
SPEC = importlib.util.spec_from_file_location("policy_compile_cli", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
policy_compile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy_compile)


def _square(x, y, size):
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]


def _entity(entity_id, label, footprint):
    return Entity3D(
        entity_id=entity_id,
        label=label,
        observation_ids=[f"obs-{entity_id}"],
        centroid_xyz=(0.0, 0.0, 0.5),
        footprint_xy=footprint,
        height_m=1.0,
        evidence_frame_ids=["f1", "f2"],
    )


def _write_fixture_tree(tmp_path):
    """Rules file, a cached compiled spec, and a run scene that violates it."""
    rules = tmp_path / "rules.md"
    rules.write_text("# demo\n- keep pallets 0.6 m from fencing\n")
    spec = PolicySpec(
        policy_id="p01-pallet-clearance",
        source_text="keep pallets 0.6 m from fencing",
        predicate=Predicate.MIN_SEPARATION,
        subject_labels=["pallet"],
        object_labels=["safety fence"],
        threshold=0.6,
    )
    cache = tmp_path / "outputs" / "policies" / "compiled" / "p01.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(spec.model_dump_json(indent=2) + "\n")
    scene = SceneMap(
        run_id="r1",
        floor_plane=(0.0, 0.0, 1.0, 0.0),
        scale_source="camera_height",
        scale_factor=1.0,
        fence_polygon=[],
        entities=[
            _entity("fence", "safety fence", _square(0, 0, 2.0)),
            _entity("near", "pallet", _square(2.3, 0.5, 0.4)),
        ],
        facts=[],
        warnings=[],
    )
    scene_json = tmp_path / "runs" / "r1" / "scene.json"
    scene_json.parent.mkdir(parents=True)
    scene_json.write_text(scene.model_dump_json(indent=2))
    return rules, scene_json


def test_run_evaluation_writes_reader_envelope_to_run_dir(tmp_path, monkeypatch):
    rules, _ = _write_fixture_tree(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert policy_compile.main(["--policies", str(rules), "--run", "r1"]) == 0

    payload = json.loads((tmp_path / "runs" / "r1" / "policies.json").read_text())
    assert set(payload) == {"specs", "results"}
    assert payload["specs"][0]["policy_id"] == "p01-pallet-clearance"
    assert payload["results"][0]["status"] == "FAIL"
    # The exact consumers the old shape starved can now read it.
    rows = _policy_rows(payload)
    assert rows[0]["policy_id"] == "p01-pallet-clearance"
    assert rows[0]["rule"] == "min_separation 0.6 m"
    assert rows[0]["source_text"] == "keep pallets 0.6 m from fencing"


def test_bare_scene_evaluation_writes_same_envelope_to_cache_dir(
    tmp_path, monkeypatch
):
    rules, scene_json = _write_fixture_tree(tmp_path)
    monkeypatch.chdir(tmp_path)

    rc = policy_compile.main(["--policies", str(rules), "--scene", str(scene_json)])

    assert rc == 0
    report = tmp_path / "outputs" / "policies" / "report_scene.json"
    payload = json.loads(report.read_text())
    assert set(payload) == {"specs", "results"}
    assert _policy_rows(payload)[0]["status"] == "FAIL"
