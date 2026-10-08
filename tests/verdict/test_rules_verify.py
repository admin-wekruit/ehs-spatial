"""The verification harness (layers/l5_rules/verify.py) on the trial pack: signature check, differential, metamorphic suite, facts parity."""
from __future__ import annotations

from pathlib import Path

from ehs_spatial.verdict.contracts import Signature
from ehs_spatial.verdict.layers.l5_rules import verify
from ehs_spatial.verdict.layers.l6_engine.python import PythonEngine
from ehs_spatial.verdict.synth import facts as F
from ehs_spatial.verdict.synth import scenes
from ehs_spatial.verdict.synth.pack import trial_pack

SIGNATURE = Signature.load(Path(__file__).resolve().parents[2] / "ehs_spatial" / "verdict" / "signature-v1.json")
ENGINE = verify.engine_of(PythonEngine, {"run_id": "verify-test"})


def test_signature_check_passes_the_trial_pack():
    assert verify.signature_check(trial_pack(), SIGNATURE) == []


def test_signature_check_catches_invented_vocabulary():
    pk = trial_pack()
    pk.rules[1].spec["requirement"]["predicate"] = "floor_gapp"
    pk.rules[0].spec["selection"]["F"] = ["fencee"]
    pk.rules[2].spec["exceptions"] = ["interlocked"]
    pk.rules[3].spec["applicability"] = ["perimeter_of(X, Z, Q)", "nonsense"]
    pk.rules[4].spec["requirement"]["inputs"] = ["stop_time"]
    pk.rules[5].spec["requirement"]["args"] = ["Y"]
    problems = verify.signature_check(pk, SIGNATURE)
    assert any("floor_gapp" in p for p in problems) and any("fencee" in p for p in problems) and any("interlocked" in p for p in problems)
    assert any("perimeter_of takes 2 args" in p for p in problems) and any("'nonsense' is not name(args)" in p for p in problems)
    assert any("'stop_time' not a declared input" in p for p in problems) and any("variable Y is bound by neither" in p for p in problems)


def test_differential_finds_threshold_change_only():
    a, b = trial_pack(), trial_pack()
    b.rules[0].spec["requirement"]["threshold"] = 2500
    scs = [scenes.cell(), scenes.cell(enclosed=False)]
    assert verify.differential(a, trial_pack(), scs, ENGINE, F.facts_of) == []
    d = verify.differential(a, b, scs, ENGINE, F.facts_of)
    assert len(d) == 8 and {x["rule_id"] for x in d} == {"fence_height"} and {(x["a"], x["b"]) for x in d} == {("PASS", "FAIL")}


def test_metamorphic_suite_passes_on_the_trial_pack():
    scs = [scenes.cell(), scenes.cell(enclosed=False), scenes.cell(coverage="none", floor_gap_m=0.17), scenes.cell(robot_fixed_gap_m=0.5, fence_height_m=1.42)]
    report = verify.metamorphic(trial_pack(), scs, ENGINE, F.facts_of)
    assert verify.ok(report), report


def test_metamorphic_catches_a_position_dependent_l2():
    def leaky_facts(scene):                                        # an L2 whose heights depend on where the object stands
        fx = F.facts_of(scene)
        for f in fx.facts:
            if f.pred == "top_height":
                f.value += 1000 * next(o.center_m[0] for o in scene.objects if o.id == f.args[0])
        return fx
    report = verify.metamorphic(trial_pack(), [scenes.cell()], ENGINE, leaky_facts, angles=(180,))
    assert {x["rule_id"] for x in report["rigid"]} == {"fence_height"} and not report["monotone"]


def test_metamorphic_catches_a_non_monotone_engine():
    swap = {"PASS": "FAIL", "FAIL": "PASS"}

    def upside_down(facts, pack, scene=None):                      # an engine that flips floor_gap verdicts
        vs = ENGINE(facts, pack, scene)
        for v in vs.verdicts:
            if v.rule_id == "floor_gap":
                v.status = swap.get(v.status, v.status)
        return vs
    report = verify.metamorphic(trial_pack(), [scenes.cell()], upside_down, F.facts_of, angles=())
    assert {x["rule_id"] for x in report["monotone"]} == {"floor_gap"} and not report["rigid"]


def test_rigid_transform_keeps_geometry():
    sc = scenes.cell()
    moved = verify.transform(sc, 90, (1.0, -2.0))
    assert moved.objects[0].center_m[:2] == [1.0, -2.0] and moved.coverage is not None
    a, b = F.facts_of(sc), F.facts_of(moved)
    parity = verify.facts_parity(a, b)
    assert parity["value"] == [] and parity["u"] == [] and parity["only_a"] == [] and parity["only_b"] == []
    assert len(b.grid.observed) == b.grid.shape[0] * b.grid.shape[1]          # rotated coverage still lands on every cell


def test_facts_parity_reports_differences():
    a = F.facts_of(scenes.cell())
    b = a.model_copy(deep=True)
    b.facts[1].value += 5
    b.facts.pop()
    p = verify.facts_parity(a, b, tol_mm=2)
    assert len(p["value"]) == 1 and p["value"][0]["pred"] == a.facts[1].pred and len(p["only_a"]) == 1
