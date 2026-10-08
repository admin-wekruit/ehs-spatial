"""Unit checks of L6 python@2: decision-rule variants, the five statuses, applicability / exceptions, grid enclosure, determinism."""
from __future__ import annotations

from pathlib import Path

from ehs_spatial.verdict.contracts import Fact, Facts, Grid, Rule, RulePack, Scene
from ehs_spatial.verdict.layers.l6_engine.python import PythonEngine, decide, enclosure
from ehs_spatial.verdict.plugins import REGISTRY, tag


def pack(op=">=", T=1400, pred="top_height", args=("F",), selection=None, applicability=(), exceptions=(), inputs=(), **extra) -> RulePack:
    req = {"predicate": pred, "args": list(args), "operator": op, "unit": "mm", "inputs": list(inputs), **extra}
    if T is not None:
        req["threshold"] = T
    r = Rule(rule_id="r", version="1", clause="c", standard="s", edition="e", rule_class="geometry",
             spec={"selection": selection or {"F": ["fence"]}, "applicability": list(applicability), "requirement": req, "exceptions": list(exceptions)})
    return RulePack(pack_id="p", version="1", signature_version="1", rules=[r])


def facts(*fs, grid=None, quality=None) -> Facts:
    return Facts(scene_id="t", signature_version="1", facts=[Fact(pred="obj", args=["f1", "fence"]), *fs], grid=grid, quality=quality or {}, producer="test-l2@0")


def top(v, u=40.0):
    return Fact(pred="top_height", args=["f1"], value=v, unit="mm", u=u)


def run(fx, pk, cfg=None, scene=None):
    return PythonEngine().run({"facts": fx, "rule_pack": pk, "scene": scene}, cfg or {}, Path("."))["verdicts"]


def status(fx, pk, cfg=None, scene=None):
    vs = run(fx, pk, cfg, scene).verdicts
    assert len(vs) == 1, vs
    return vs[0]


def test_registered():
    assert REGISTRY["L6"]["python"] is PythonEngine and tag(PythonEngine) == "python@2"


def test_guard_band_boundaries_match_rules_lp():
    assert decide(1440, 40, ">=", 1400) == "PASS"               # V - U = T passes (inclusive)
    assert decide(1360, 40, ">=", 1400) == "NEEDS_MEASUREMENT"  # V + U = T is not yet FAIL (strict)
    assert decide(1359, 40, ">=", 1400) == "FAIL"
    assert decide(160, 20, "<=", 180) == "PASS"
    assert decide(200, 20, "<=", 180) == "NEEDS_MEASUREMENT"
    assert decide(201, 20, "<=", 180) == "FAIL"
    assert decide(1, None, "==", 1) == "PASS" and decide(0, None, "==", 1) == "FAIL"


def test_decision_variants():
    fx = facts(top(1420))                                        # 1420 +- 40 straddles 1400
    assert status(fx, pack()).status == "NEEDS_MEASUREMENT"
    assert status(fx, pack(), {"decision": "guard_band", "k": 1}).status == "PASS"          # k=1 halves U: 1420 - 20 >= 1400
    assert status(fx, pack(), {"decision": "simple"}).status == "PASS"
    assert status(facts(top(1399)), pack(), {"decision": "simple"}).status == "FAIL"
    assert status(fx, pack(), {"decision": "conservative"}).status == "FAIL"
    assert status(facts(top(1440)), pack(), {"decision": "conservative"}).status == "PASS"
    assert run(fx, pack(), {"decision": "conservative"}).decision_rule == "conservative"
    assert run(fx, pack(), {"k": 1}).decision_rule == "guard_band_k1"


def test_margin_and_evidence():
    v = status(facts(top(1500)), pack())
    assert (v.measured, v.u, v.threshold, v.margin) == (1500, 40, 1400, 100)
    assert v.evidence["facts"][0]["pred"] == "top_height"
    v = status(facts(Fact(pred="bottom_height", args=["f1"], value=100, unit="mm", u=20)), pack("<=", 180, "bottom_height"))
    assert v.status == "PASS" and v.margin == 80


def test_missing_fact_and_untrusted():
    v = status(facts(), pack())
    assert v.status == "CANNOT_DETERMINE" and "no fact top_height(f1)" in v.notes
    assert run(facts(), pack()).gaps["missing_facts"] == ["top_height(f1)"]
    v = status(facts(top(1500), Fact(pred="untrusted", args=["f1"], value=1)), pack())
    assert v.status == "CANNOT_DETERMINE" and v.evidence == {"untrusted": ["f1"]}
    assert status(facts(top(1500), quality={"untrusted": ["f1"]}), pack()).status == "CANNOT_DETERMINE"
    assert status(facts(Fact(pred="top_height", args=["f1"], value=1500, u=40, flags=["untrusted"])), pack()).status == "CANNOT_DETERMINE"


def test_needs_input_formula_and_table():
    pk = pack(">=", None, formula="1.6*T + 8*(d - 14)", inputs=["stop_time_ms", "resolution_mm"], bindings={"T": "stop_time_ms", "d": "resolution_mm"})
    scene = Scene(scene_id="t", objects=[], ground_normal=[0, 0, 1])
    v = status(facts(top(1000, 20)), pk, scene=scene)
    assert v.status == "NEEDS_INPUT" and v.unknown_inputs == ["stop_time_ms", "resolution_mm"] and v.measured == 1000
    scene.declared_inputs = {"stop_time_ms": 500, "resolution_mm": 30}
    v = status(facts(top(1000, 20)), pk, scene=scene)
    assert v.status == "PASS" and v.threshold == 1.6 * 500 + 8 * 16
    assert status(facts(top(1000, 20)), pk).status == "NEEDS_INPUT"                       # no scene at all -> nothing declared
    pk = pack(">=", None, table="ISO13857:2019/Table2")
    v = status(facts(top(1000, 20)), pk, scene=scene)
    assert v.status == "NEEDS_INPUT" and v.unknown_inputs == ["ISO13857:2019/Table2"]
    scene.declared_inputs["ISO13857:2019/Table2"] = 850
    assert status(facts(top(1000, 20)), pk, scene=scene).status == "PASS"


def test_applicability_binds_and_filters():
    pk = pack("<=", 180, "bottom_height", applicability=["perimeter_of(F, Z)"])
    fx = facts(Fact(pred="bottom_height", args=["f1"], value=100, u=20))
    vs = run(fx, pk)
    assert vs.verdicts == [] and vs.coverage["r"]["not_applicable"] == 1
    fx = facts(Fact(pred="bottom_height", args=["f1"], value=100, u=20), Fact(pred="perimeter_of", args=["f1", "z1"], value=1))
    v = status(fx, pk)
    assert v.status == "PASS" and v.subjects == ["f1"]


def test_exception_switches_rule_off():
    fx = facts(top(1500), Fact(pred="interlock", args=["f1"], value=1))
    vs = run(fx, pack(exceptions=["interlock"]))
    assert vs.verdicts == [] and vs.coverage["r"]["exception"] == 1


def test_pair_predicate_symmetric_lookup():
    fx = Facts(scene_id="t", signature_version="1", facts=[Fact(pred="obj", args=["r", "robot"]), Fact(pred="obj", args=["f1", "fence"]),
                                                            Fact(pred="min_distance_3d", args=["f1", "r"], value=700, u=30)])
    v = status(fx, pack(">=", 500, "min_distance_3d", ("R", "X"), selection={"R": ["robot"], "X": ["fence"]}))
    assert v.status == "PASS" and v.subjects == ["r", "f1"]


def grid(open_cell=False, observed="all", hazard=True) -> Grid:
    cells = [(x, y) for x in range(5) for y in range(5)]
    ring = [(x, y) for x, y in cells if max(abs(x - 2), abs(y - 2)) == 1 and not (open_cell and (x, y) == (1, 2))]
    obs = None if observed is None else [list(c) for c in cells if observed == "all" or c != (1, 2)]
    return Grid(cell_m=0.1, origin_xy=[0, 0], basis=[[0, 1, 0], [-1, 0, 0]], shape=[5, 5], blocked=[list(c) for c in ring],
                hazard=[[2, 2]] if hazard else [], outside=[list(c) for c in cells if 0 in c or 4 in c], observed=obs)


def test_enclosure_grid_semantics():
    assert enclosure(grid())[0] == 1
    assert enclosure(grid(open_cell=True))[0] == 0
    assert enclosure(grid(observed=None))[0] is None                            # closed ring but no coverage: not decided (@2)
    assert enclosure(grid(observed=None))[1]["no_coverage"] is True
    assert enclosure(grid(open_cell=True, observed=None))[0] is None
    assert enclosure(grid(open_cell=True, observed="not_opening"))[0] is None   # the only path crosses an unobserved cell
    assert enclosure(grid(hazard=False))[0] is None
    pk = pack("==", 1, "enclosed", ("Z",), selection={"Z": ["hazard_zone"]})
    assert status(facts(grid=grid()), pk).status == "PASS"
    v = status(facts(grid=grid(open_cell=True)), pk)
    assert v.status == "FAIL" and v.subjects == ["hazard_zone"] and v.evidence["grid"]["openings"] == 1
    assert status(facts(grid=grid(open_cell=True, observed=None)), pk).status == "CANNOT_DETERMINE"
    v = status(facts(grid=grid(observed=None)), pk)
    assert v.status == "CANNOT_DETERMINE" and v.notes[0].startswith("no floor coverage")
    assert run(facts(), pk).verdicts == []                                        # no grid -> no implicit zone -> nothing to bind


def test_deterministic_and_provenance():
    fx = facts(top(1500))
    a, b = run(fx, pack(), {"run_id": "x"}), run(fx, pack(), {"run_id": "x"})
    assert a.model_dump_json() == b.model_dump_json()
    assert a.provenance.plugins == {"L2": "test-l2@0", "L6": "python@2"} and a.provenance.run_id == "x" and a.rule_pack == "p@1"


def test_uncompiled_rules_are_skipped():
    pk = pack()
    pk.rules[0].status, pk.rules[0].unsupported_reason = "refused", "photos cannot see it"
    vs = run(facts(top(1500)), pk)
    assert vs.verdicts == [] and vs.coverage["r"]["skipped"] == "photos cannot see it" and vs.gaps["rules_skipped"] == ["r"]
