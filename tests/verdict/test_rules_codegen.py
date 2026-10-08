"""L5 code-synthesis@0 with a fake model (no API): a faithful translation with passing tests compiles, a wrong threshold is refused by the
model's own tests, an ASP the model wrote itself is checked under clingo, a model refusal is kept, llm_calls counts uncached calls."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from ehs_spatial.verdict import llm
from ehs_spatial.verdict.contracts import Signature
from ehs_spatial.verdict.layers.l4_spec.clause_kg import ClauseKG
from ehs_spatial.verdict.layers.l5_rules import asp, common
from ehs_spatial.verdict.layers.l5_rules.code_synthesis import CodeSynthesis

SIGNATURE = Signature.load(Path(__file__).resolve().parents[2] / "ehs_spatial" / "verdict" / "signature-v1.json")
FLOOR_GAP, MIN_HEIGHT, LC_BOTTOM, TRAPPING = "ISO13857:2019/4.4", "ISO13857:2019/Table2-min-height", "TS0011963:Rev10/9.1.2", "ISO10218-2:2011/trapping-clearance"
TESTS = {FLOOR_GAP: [({"floor_gap_m": 0.10}, "PASS"), ({"floor_gap_m": 0.25}, "FAIL"), ({"floor_gap_m": 0.185}, "NEEDS_MEASUREMENT")],
         MIN_HEIGHT: [({"fence_height_m": 2.0}, "PASS"), ({"fence_height_m": 1.2}, "FAIL")],
         LC_BOTTOM: [({"lc_bottom_m": 0.25}, "PASS"), ({"lc_bottom_m": 0.4}, "FAIL")],
         TRAPPING: [({"robot_fixed_gap_m": 1.0}, "NEEDS_INPUT")]}


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    out = ClauseKG().run({"signature": SIGNATURE, "scene": None}, {"clauses_file": ["clauses-v0.json", "clauses-ts0011963-v0.json"]}, tmp_path_factory.mktemp("l4"))
    return {**out, "signature": SIGNATURE, "retrieved": [FLOOR_GAP, MIN_HEIGHT, LC_BOTTOM, TRAPPING]}


def faithful(clause) -> common.Spec:
    """The hand-written spec of the clause file as the model's answer."""
    r = clause.requirement
    req = common.Requirement(predicate=str(r.get("predicate")), args=list(r.get("args", [])), operator=str(r.get("operator")),
                             threshold=r["threshold"] if asp.numeric(r.get("threshold")) else None, table=r.get("table"), formula=r.get("formula"),
                             unit=r.get("unit"), inputs=list(r.get("inputs", [])), bindings=[])
    return common.Spec(selection=[common.Var(var=v, classes=c) for v, c in clause.selection.items()], applicability=list(clause.applicability),
                       requirement=req, exceptions=list(clause.exceptions))


def fake(monkeypatch, graph, edit=None, cached=(), asp_text=None):
    by_id, calls = {c.id: c for c in graph.clauses}, []

    def complete(system, user, schema, model=None, effort=None, cache_dir=None):
        cid = user.split("\n")[0].removeprefix("id: ")
        spec = faithful(by_id[cid])
        out = common.Synthesis(spec=spec, asp=(asp_text or {}).get(cid, ""), needs=[], refuse_reason=None,
                               tests=[common.Test(cell_params=[common.Param(name=k, value=v) for k, v in p.items()], expected=e) for p, e in TESTS.get(cid, [])])
        if edit:
            edit(cid, out)
        calls.append(cid)
        assert schema is common.Synthesis and "Signature classes" in system and "Worked example" in system
        return out, {"model": "fake", "cached": cid in cached, "usage": {}, "stop_reason": "end_turn"}
    monkeypatch.setattr(llm, "complete", complete)
    return calls


def test_faithful_translation_compiles(inputs, tmp_path, monkeypatch):
    calls = fake(monkeypatch, inputs["clauses"], cached=(LC_BOTTOM,))
    out = CodeSynthesis().run(inputs, {}, tmp_path / "L5")
    pack = out["rule_pack"]
    assert out["llm_calls"] == 3 and calls == [c.id for c in inputs["clauses"].clauses if c.id in inputs["retrieved"]] and pack.pack_id == "codegen"
    by = {r.clause: r for r in pack.rules}
    assert {c: by[c].status for c in inputs["retrieved"]} == {FLOOR_GAP: "compiled", MIN_HEIGHT: "compiled", LC_BOTTOM: "compiled", TRAPPING: "needs_input"}
    assert by[FLOOR_GAP].asp.startswith("thr(iso13857_2019_4_4, 180).  dir(iso13857_2019_4_4, le).")
    assert by[FLOOR_GAP].provenance == {"plugin": "code-synthesis@0", "clause": FLOOR_GAP, "model": "fake"}
    assert json.loads(by[FLOOR_GAP].review["llm"])["tests"][1]["expected"] == "FAIL"
    assert by[FLOOR_GAP].spec["requirement"]["threshold"] == 180.0 and by[FLOOR_GAP].spec["selection"] == {"F": ["fence", "guard"], "Z": ["hazard_zone", "restricted_space"]}


def test_wrong_threshold_is_refused_by_the_models_tests(inputs, tmp_path, monkeypatch):
    def wrong(cid, out):
        if cid == FLOOR_GAP:
            out.spec.requirement.threshold = 280
    fake(monkeypatch, inputs["clauses"], edit=wrong)
    by = {r.clause: r for r in CodeSynthesis().run(inputs, {}, tmp_path / "L5")["rule_pack"].rules}
    assert by[FLOOR_GAP].status == "refused" and by[FLOOR_GAP].asp == ""
    assert by[FLOOR_GAP].unsupported_reason == "tests (python): cell {'floor_gap_m': 0.25} expected FAIL, got ['PASS']"
    assert json.loads(by[FLOOR_GAP].review["llm"])["spec"]["requirement"]["threshold"] == 280
    assert {by[c].status for c in (MIN_HEIGHT, LC_BOTTOM)} == {"compiled"}


def test_model_written_asp_is_checked_under_clingo(inputs, tmp_path, monkeypatch):
    wrong_asp = ("thr(iso13857_2019_4_4, 280).  dir(iso13857_2019_4_4, le).\n"
                 'subject(iso13857_2019_4_4, (F, "hazard_zone")) :- obj(F, fence), hazard(_), perimeter_of(F, "hazard_zone").\n'
                 'meas(iso13857_2019_4_4, (F, "hazard_zone"), V, U) :- subject(iso13857_2019_4_4, (F, "hazard_zone")), floor_gap(F, V, U).\n')
    fake(monkeypatch, inputs["clauses"], asp_text={FLOOR_GAP: wrong_asp})
    by = {r.clause: r for r in CodeSynthesis().run(inputs, {}, tmp_path / "L5")["rule_pack"].rules}
    assert by[FLOOR_GAP].status == "refused" and by[FLOOR_GAP].unsupported_reason.startswith("tests (clingo): cell {'floor_gap_m': 0.25} expected FAIL")
    assert (tmp_path / "L5" / "program.lp").exists()                 # clingo wrote its program under the run directory, not the cwd


def test_model_refusal_and_a_missing_test_cell(inputs, tmp_path, monkeypatch):
    def edit(cid, out):
        if cid == MIN_HEIGHT:
            out.refuse_reason = "needs the hazard zone perimeter, which the scene does not declare"
        if cid == LC_BOTTOM:
            out.tests = [common.Test(cell_params=[common.Param(name="no_such_knob", value=1.0)], expected="PASS")]
    fake(monkeypatch, inputs["clauses"], edit=edit)
    by = {r.clause: r for r in CodeSynthesis().run(inputs, {}, tmp_path / "L5")["rule_pack"].rules}
    assert by[MIN_HEIGHT].status == "refused" and by[MIN_HEIGHT].unsupported_reason.startswith("model refused: needs the hazard zone")
    assert by[LC_BOTTOM].status == "refused" and by[LC_BOTTOM].unsupported_reason.startswith("tests: cell {'no_such_knob': 1.0} is not a valid synthetic cell")
    assert Counter(r.status for r in by.values()) == {"refused": 2, "compiled": 1, "needs_input": 1}
