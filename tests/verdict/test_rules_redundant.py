"""L5 redundant-translation@0 with a fake model (no API): two agreeing framings compile, a disagreeing pair is refused with both candidates
in review, normalisation ignores class order / operator spelling, a refusal on either side refuses, llm_calls counts both calls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ehs_spatial.verdict import llm
from ehs_spatial.verdict.contracts import Signature
from ehs_spatial.verdict.layers.l4_spec.clause_kg import ClauseKG
from ehs_spatial.verdict.layers.l5_rules import asp, common
from ehs_spatial.verdict.layers.l5_rules.redundant import Redundant, disagreement

SIGNATURE = Signature.load(Path(__file__).resolve().parents[2] / "ehs_spatial" / "verdict" / "signature-v1.json")
FLOOR_GAP, MIN_HEIGHT, TRAPPING = "ISO13857:2019/4.4", "ISO13857:2019/Table2-min-height", "ISO10218-2:2011/trapping-clearance"


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    out = ClauseKG().run({"signature": SIGNATURE, "scene": None}, {"clauses_file": ["clauses-v0.json", "clauses-ts0011963-v0.json"]}, tmp_path_factory.mktemp("l4"))
    return {**out, "signature": SIGNATURE, "retrieved": [FLOOR_GAP, MIN_HEIGHT, TRAPPING]}


def faithful(clause) -> common.Spec:
    r = clause.requirement
    req = common.Requirement(predicate=str(r.get("predicate")), args=list(r.get("args", [])), operator=str(r.get("operator")),
                             threshold=r["threshold"] if asp.numeric(r.get("threshold")) else None, table=r.get("table"), formula=r.get("formula"),
                             unit=r.get("unit"), inputs=list(r.get("inputs", [])), bindings=[])
    return common.Spec(selection=[common.Var(var=v, classes=c) for v, c in clause.selection.items()], applicability=list(clause.applicability),
                       requirement=req, exceptions=list(clause.exceptions))


def fake(monkeypatch, graph, edit_b=None):
    by_id, calls = {c.id: c for c in graph.clauses}, []

    def complete(system, user, schema, model=None, effort=None, cache_dir=None):
        cid, framing_b = user.split("\n")[0].removeprefix("id: "), "withheld" in user
        assert schema is common.Translation and ("requirement as given" in user) != framing_b
        out = common.Translation(spec=faithful(by_id[cid]), asp="", needs=[], refuse_reason=None)
        if framing_b and edit_b:
            edit_b(cid, out)
        calls.append((cid, "B" if framing_b else "A"))
        return out, {"model": "fake", "cached": False, "usage": {}, "stop_reason": "end_turn"}
    monkeypatch.setattr(llm, "complete", complete)
    return calls


def test_agreeing_translations_compile(inputs, tmp_path, monkeypatch):
    calls = fake(monkeypatch, inputs["clauses"])
    out = Redundant().run(inputs, {}, tmp_path / "L5")
    by = {r.clause: r for r in out["rule_pack"].rules}
    assert out["llm_calls"] == 6 and calls == [(c, f) for c in inputs["retrieved"] for f in "AB"] and out["rule_pack"].pack_id == "redundant"
    assert {c: by[c].status for c in inputs["retrieved"]} == {FLOOR_GAP: "compiled", MIN_HEIGHT: "compiled", TRAPPING: "needs_input"}
    assert by[FLOOR_GAP].asp.startswith("thr(iso13857_2019_4_4, 180).") and set(by[FLOOR_GAP].review) == {"candidate_a", "candidate_b"}
    assert by[FLOOR_GAP].provenance == {"plugin": "redundant-translation@0", "clause": FLOOR_GAP, "model": "fake"}


def test_disagreeing_translations_are_refused_with_both_candidates(inputs, tmp_path, monkeypatch):
    def edit_b(cid, out):
        if cid == FLOOR_GAP:
            out.spec.requirement.threshold = 240
        if cid == MIN_HEIGHT:
            out.spec.selection = [common.Var(var="S", classes=["fence"])]
            out.needs = ["risk_level"]
    fake(monkeypatch, inputs["clauses"], edit_b)
    by = {r.clause: r for r in Redundant().run(inputs, {}, tmp_path / "L5")["rule_pack"].rules}
    assert by[FLOOR_GAP].status == "refused" and by[FLOOR_GAP].unsupported_reason == "redundant translations disagree: requirement.threshold"
    assert by[MIN_HEIGHT].unsupported_reason == "redundant translations disagree: selection, requirement.inputs"
    a, b = (json.loads(by[FLOOR_GAP].review[k]) for k in ("candidate_a", "candidate_b"))
    assert (a["spec"]["requirement"]["threshold"], b["spec"]["requirement"]["threshold"]) == (180, 240) and by[FLOOR_GAP].asp == ""
    assert by[TRAPPING].status == "needs_input"


def test_normalisation_ignores_class_order_and_operator_spelling(inputs, tmp_path, monkeypatch):
    def edit_b(cid, out):
        if cid == FLOOR_GAP:
            out.spec.selection = [common.Var(var="Z", classes=["restricted_space", "hazard_zone"]), common.Var(var="F", classes=["guard", "fence"])]
            out.spec.requirement.operator = "≤"
            out.spec.requirement.threshold = 180.0
    fake(monkeypatch, inputs["clauses"], edit_b)
    by = {r.clause: r for r in Redundant().run(inputs, {}, tmp_path / "L5")["rule_pack"].rules}
    assert by[FLOOR_GAP].status == "compiled"
    assert disagreement({"selection": {}, "applicability": [], "requirement": {"operator": ">="}, "exceptions": ["a"]},
                        {"selection": {}, "applicability": ["x(Y)"], "requirement": {"operator": "<="}, "exceptions": ["a"]}) == ["applicability", "requirement.operator"]


def test_refusal_on_either_side_refuses(inputs, tmp_path, monkeypatch):
    def edit_b(cid, out):
        if cid == MIN_HEIGHT:
            out.refuse_reason = "cannot reconstruct a number from the title alone"
    fake(monkeypatch, inputs["clauses"], edit_b)
    by = {r.clause: r for r in Redundant().run(inputs, {}, tmp_path / "L5")["rule_pack"].rules}
    assert by[MIN_HEIGHT].status == "refused" and by[MIN_HEIGHT].unsupported_reason == "model refused: cannot reconstruct a number from the title alone"
    assert by[FLOOR_GAP].status == "compiled"
