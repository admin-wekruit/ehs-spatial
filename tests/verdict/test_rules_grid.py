"""Threshold grid (synth/grid.py) expectations under L6 python@2 with the guard band k=2, and the knob the decision variants turn."""
from __future__ import annotations

from ehs_spatial.verdict.layers.l5_rules import verify
from ehs_spatial.verdict.layers.l6_engine.python import PythonEngine
from ehs_spatial.verdict.synth import grid
from ehs_spatial.verdict.synth.pack import trial_pack

NUMERIC = {"fence_height", "floor_gap", "lc_lowest_beam", "crush_gap"}


def test_cases_sit_on_the_guard_band_boundary():
    cases = grid.cases(trial_pack())
    assert {c["rule_id"] for c in cases} == NUMERIC and all("skipped" not in c for c in cases)
    for c in cases:
        if c["offset"] in ("+2s", "-2s"):
            assert abs(c["value_mm"] - c["threshold"]) == c["u_mm"]      # +-2 sigma = +-U: exactly on V -+ U = T
        f = next(f for f in c["facts"].facts if f.pred in ("top_height", "bottom_height", "min_distance_3d") and f.value == round(c["value_mm"]))
        assert f.u == c["u_mm"]


def test_grid_matches_guard_band_k2():
    rep = verify.threshold_grid(trial_pack(), verify.engine_of(PythonEngine))
    assert rep["mismatches"] == [], rep["mismatches"]
    assert {r["rule_id"] for r in rep["rows"]} == NUMERIC
    table = verify.grid_table(rep["rows"])
    assert table.splitlines()[0] == "| rule | +eps | -eps | +2s | -2s | +5s | -5s |"
    assert "| crush_gap | NEEDS_MEASUREMENT | NEEDS_MEASUREMENT | PASS | NEEDS_MEASUREMENT | PASS | FAIL |" in table


def test_simple_decision_never_says_needs_measurement():
    cases = grid.cases(trial_pack())
    rep = verify.threshold_grid(trial_pack(), verify.engine_of(PythonEngine, {"decision": "simple"}), cases)
    bad = {(r["rule_id"], r["offset"]) for r in rep["mismatches"]}
    assert bad == {(c["rule_id"], c["offset"]) for c in cases if c["expected"] == "NEEDS_MEASUREMENT"}
    assert all(r["actual"] in (["PASS"], ["FAIL"]) for r in rep["rows"])


def test_conservative_decision_fails_the_band():
    cases = grid.cases(trial_pack())
    rep = verify.threshold_grid(trial_pack(), verify.engine_of(PythonEngine, {"decision": "conservative"}), cases)
    assert all(r["actual"] == ["FAIL"] for r in rep["rows"] if r["expected"] == "NEEDS_MEASUREMENT")
    assert all(r["ok"] for r in rep["rows"] if r["expected"] != "NEEDS_MEASUREMENT")
