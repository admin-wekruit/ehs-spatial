"""Lab rule 6 for the baseline itself: the ported plugins reproduce the trial (research/verdict-layer-trial-2026-10-07/out/<cell>/
verdicts.json): the status counts and, per (rule, subjects), status / measured / u / margin."""
import json
from pathlib import Path

import pytest

from ehs_spatial.verdict.contracts import VerdictSet
from ehs_spatial.verdict.lab import config, runner

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "ehs_spatial/verdict/configs/baseline.yaml"
TRIAL = ROOT / "research/verdict-layer-trial-2026-10-07/out"
EXPECTED = {"090": {"PASS": 10, "FAIL": 2, "NEEDS_MEASUREMENT": 2, "CANNOT_DETERMINE": 1, "NEEDS_INPUT": 7},
            "030": {"PASS": 9, "FAIL": 1, "NEEDS_MEASUREMENT": 1, "CANNOT_DETERMINE": 1, "NEEDS_INPUT": 6}}


@pytest.mark.parametrize("item", sorted(EXPECTED))
def test_baseline_reproduces_the_trial(tmp_path, item):
    cfg = config.load(BASELINE)
    cfg["item"] = item
    vs = VerdictSet.load(tmp_path / runner.run(cfg, tmp_path, f"parity-{item}") / "L6/verdicts.json")
    counts = {}
    for v in vs.verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1
    assert counts == EXPECTED[item]
    trial = {(r["rule"], tuple(r["subjects"])): r for r in json.loads((TRIAL / item / "verdicts.json").read_text())["verdicts"]}
    mine = {(v.rule_id, tuple(v.subjects)): v for v in vs.verdicts}
    assert mine.keys() == trial.keys()
    for key, r in trial.items():
        v = mine[key]
        assert (v.status, v.measured, v.u, v.margin) == (r["status"], r.get("measured_mm"), r.get("U_mm"), r.get("margin_mm")), key
        assert v.provenance.run_id == f"parity-{item}" and v.provenance.plugins["L6"] == "clingo@2"
