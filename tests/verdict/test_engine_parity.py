"""Parity of L6 python@2 with the trial's clingo engine (research/verdict-layer-trial-2026-10-07: engine.py + rules.lp) on the trial's own
outputs: out/{090,030}/scene-graph.json converted into Facts, the trial's six rules as Rule.spec entries (synth/pack.py), exact status per
(rule, subjects) against out/{090,030}/verdicts.json, plus measured / U / threshold / margin on the numeric rows and the opening count."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ehs_spatial.verdict.contracts import Fact, Facts, Grid
from ehs_spatial.verdict.layers.l6_engine.python import PythonEngine
from ehs_spatial.verdict.synth.pack import trial_pack

TRIAL = Path(__file__).resolve().parents[2] / "research" / "verdict-layer-trial-2026-10-07" / "out"
DEFAULT_SIGMA_M, DEFAULT_SCALE_REL = 0.05, 0.02   # scene_graph.py defaults
COUNTS = {"090": {"PASS": 10, "FAIL": 2, "NEEDS_MEASUREMENT": 2, "CANNOT_DETERMINE": 1, "NEEDS_INPUT": 7},
          "030": {"PASS": 9, "FAIL": 1, "NEEDS_MEASUREMENT": 1, "CANNOT_DETERMINE": 1, "NEEDS_INPUT": 6}}
OPENINGS = {"090": 48, "030": 50}


def mm(x: float) -> int:
    return int(round(x * 1000))


def u_mm(value_m: float, sigmas, rel) -> int:
    terms = [DEFAULT_SIGMA_M if s is None else s for s in sigmas]
    rel = DEFAULT_SCALE_REL if rel is None else rel
    return mm(2 * math.sqrt(sum(t * t for t in terms) + (rel * abs(value_m)) ** 2))


def facts_of_graph(g: dict) -> Facts:
    """scene-graph.json -> Facts: nodes give obj / top_height / bottom_height (U as engine.facts_for), typed edges give pair facts."""
    rel = g.get("scale_rel_unc")
    fs = []
    for n in g["nodes"]:
        if n["cls"] == "floor":
            continue
        views = [str(v) for v in n["views"]]
        fs.append(Fact(pred="obj", args=[n["id"], n["cls"]]))
        fs.append(Fact(pred="bottom_height", args=[n["id"]], value=mm(n["bottom_m"]), unit="mm", u=u_mm(n["bottom_m"], [n["sigma_m"].get("bottom")], rel), views=views))
        fs.append(Fact(pred="top_height", args=[n["id"]], value=mm(n["top_m"]), unit="mm",
                       u=u_mm(n["top_m"], [n["sigma_m"].get("bottom"), n["sigma_m"].get("H")], rel), views=views))
    for e in g["edges"]:
        if e["type"] in ("min_distance_3d", "horizontal_gap", "z_overlap", "reach_over"):
            fs.append(Fact(pred=e["type"], args=[e["a"], e["b"]], value=e["value_mm"], unit="mm", u=e["U_mm"], views=[str(v) for v in e["views"]], flags=e["flags"]))
    occ = g["plan_occupancy"]
    grid = Grid(cell_m=occ["cell_m"], origin_xy=occ["origin"], basis=occ["basis"], shape=occ["shape"], blocked=occ["blocked"], hazard=occ["hazard"],
                outside=occ["outside"], observed=occ["observed"])
    return Facts(scene_id=g["scene_id"], signature_version="1", facts=fs, grid=grid, producer="trial-scene-graph-converter")


@pytest.mark.parametrize("cell", ["090", "030"])
def test_parity_with_trial_verdicts(cell):
    g = json.loads((TRIAL / cell / "scene-graph.json").read_text())
    want = {(r["rule"], tuple(r["subjects"])): r for r in json.loads((TRIAL / cell / "verdicts.json").read_text())["verdicts"]}
    vs = PythonEngine().run({"facts": facts_of_graph(g), "rule_pack": trial_pack(), "scene": None}, {"decision": "guard_band", "k": 2}, Path("."))["verdicts"]
    got = {(v.rule_id, tuple(v.subjects)): v for v in vs.verdicts}
    assert set(got) == set(want)
    for key, r in want.items():
        assert got[key].status == r["status"], (key, got[key].status, r["status"])
        if "measured_mm" in r:
            v = got[key]
            assert (v.measured, v.u, v.threshold, v.margin) == (r["measured_mm"], r["U_mm"], r["threshold_mm"], r["margin_mm"]), key
    assert {s: sum(1 for v in vs.verdicts if v.status == s) for s in COUNTS[cell]} == COUNTS[cell]
    assert got[("enclosure", ("hazard_zone",))].evidence["grid"]["openings"] == OPENINGS[cell]
    assert vs.decision_rule == "guard_band_k2" and vs.provenance.plugins["L6"] == "python@2"


def test_enclosure_parity_clingo_python_on_synth_cells(tmp_path):
    """@2 of both engines + handwritten@2: enclosure is decided only with coverage (PASS closed / FAIL open), CANNOT_DETERMINE without."""
    from ehs_spatial.verdict.contracts import Signature
    from ehs_spatial.verdict.layers.l5_rules.handwritten import Handwritten
    from ehs_spatial.verdict.layers.l6_engine.clingo import Clingo
    from ehs_spatial.verdict.synth import facts as F
    from ehs_spatial.verdict.synth import scenes

    sig = Signature.load(TRIAL.parents[2] / "ehs_spatial/verdict/signature-v1.json")
    pack = Handwritten().run({"signature": sig}, {}, tmp_path)["rule_pack"]
    want = {("full", True): "PASS", ("full", False): "FAIL", ("none", True): "CANNOT_DETERMINE", ("none", False): "CANNOT_DETERMINE"}
    for (cov, enc), expected in want.items():
        scene = scenes.cell(enclosed=enc, coverage=cov)
        fx = F.facts_of(scene)
        for engine in (Clingo, PythonEngine):
            vs = engine().run({"facts": fx, "rule_pack": pack, "scene": scene}, {"k": 2}, tmp_path)["verdicts"]
            got = {v.rule_id: v for v in vs.verdicts}["enclosure"]
            assert got.status == expected, (cov, enc, engine.__name__, got.status, got.notes)
            if expected == "CANNOT_DETERMINE":
                assert "no floor coverage" in got.notes[0] or "unobserved" in got.notes[0], got.notes
