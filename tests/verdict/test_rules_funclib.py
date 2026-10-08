"""L5 function-library@0 on the merged ISO v0 + TS-0011963 graph: status counts, signature-clean rules, asp.render reproduces the
handwritten ASP shape, parity with handwritten@2 under both engines on synthetic cells, vocabulary gaps from the alignment,
`all` vs retrieved, byte-identical re-runs."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ehs_spatial.verdict.contracts import RulePack, Signature
from ehs_spatial.verdict.layers.l4_spec.clause_kg import ClauseKG
from ehs_spatial.verdict.layers.l5_rules import asp, common, verify
from ehs_spatial.verdict.layers.l5_rules.function_library import FunctionLibrary
from ehs_spatial.verdict.layers.l5_rules.handwritten import Handwritten
from ehs_spatial.verdict.layers.l5_rules.handwritten import rules as handwritten_rules
from ehs_spatial.verdict.layers.l6_engine.clingo import Clingo
from ehs_spatial.verdict.layers.l6_engine.python import PythonEngine
from ehs_spatial.verdict.synth import scenes

SIGNATURE = Signature.load(Path(__file__).resolve().parents[2] / "ehs_spatial" / "verdict" / "signature-v1.json")
FILES = ["clauses-v0.json", "clauses-ts0011963-v0.json"]
COMPILED = {"iec60204_1_2016_10_1_2", "iso13857_2019_4_4", "iso13857_2019_table2_min_height", "ts0011963_rev10_4_3_4", "ts0011963_rev10_4_3_4_upper",
            "ts0011963_rev10_8_1_4", "ts0011963_rev10_8_1_5", "ts0011963_rev10_8_1_6", "ts0011963_rev10_9_1_1", "ts0011963_rev10_9_1_2", "ts0011963_rev10_9_1_6", "ts0011963_rev10_9_1_7"}
NEEDS_INPUT = {"iso10218_2_2011_perimeter_safeguarding", "iso10218_2_2011_trapping_clearance", "iso13854_2017_table1", "iso13855_2010_normal_approach",
               "iso13855_2024_9_3", "iso13855_2024_formula13", "iso13857_2019_table2", "iso13857_2019_table4", "ts0011963_rev10_10_6_3"}
PARITY = {"iso13857_2019_4_4": "floor_gap", "iso13857_2019_table2_min_height": "fence_height", "ts0011963_rev10_9_1_2": "lc_lowest_beam"}
CELLS = [scenes.cell(), scenes.cell(floor_gap_m=0.2), scenes.cell(floor_gap_m=0.185), scenes.cell(fence_height_m=1.2), scenes.cell(lc_bottom_m=0.4),
         scenes.cell(robot_fixed_gap_m=0.4)]


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    out = ClauseKG().run({"signature": SIGNATURE, "scene": None}, {"clauses_file": FILES}, tmp_path_factory.mktemp("l4"))
    return {**out, "signature": SIGNATURE}


@pytest.fixture(scope="module")
def pack(inputs, tmp_path_factory):
    return FunctionLibrary().run(inputs, {}, tmp_path_factory.mktemp("l5"))["rule_pack"]


def test_status_counts_on_the_merged_graph(pack):
    assert len(pack.rules) == 38 and Counter(r.status for r in pack.rules) == {"compiled": 12, "needs_input": 9, "refused": 17}   # 12 since attributes may be unary applicability atoms (TS 8.1.4)
    assert {r.rule_id for r in pack.rules if r.status == "compiled"} == COMPILED
    assert {r.rule_id for r in pack.rules if r.status == "needs_input"} == NEEDS_INPUT
    assert [r.rule_id for r in pack.rules] == sorted(r.rule_id for r in pack.rules)
    assert all(r.provenance == {"plugin": "function-library@0", "clause": r.clause} for r in pack.rules)
    assert all(r.unsupported_reason for r in pack.rules if r.status == "refused")
    assert pack.pack_id == "funclib" and pack.common_asp == asp.COMMON_ASP and pack.signature_version == "1"


def test_live_rules_are_signature_clean_and_carry_asp(pack):
    live = [r for r in pack.rules if r.status in ("compiled", "needs_input")]
    assert verify.signature_check(RulePack(pack_id="x", version="0", signature_version="1", rules=live), SIGNATURE) == []
    assert all(r.asp for r in live) and all(not r.asp for r in pack.rules if r.status == "refused")
    assert all("needs_input) :- subject(" in r.asp for r in live if r.status == "needs_input")
    assert all(r.inputs == sorted(r.spec["requirement"].get("inputs", [])) for r in live)
    assert {r.rule_id: r.thresholds for r in live if r.rule_id in ("iso13857_2019_4_4", "ts0011963_rev10_9_1_7")} == \
        {"iso13857_2019_4_4": {"iso13857_2019_4_4": 180.0}, "ts0011963_rev10_9_1_7": {"ts0011963_rev10_9_1_7": 300.0}}


def test_render_reproduces_the_handwritten_shape():
    hw = {r.rule_id: r for r in handwritten_rules()}
    for rid in ("floor_gap", "fence_height", "lc_lowest_beam"):
        assert asp.render(hw[rid]) == hw[rid].asp.replace("bottom(", "bottom_height(").replace("top(", "top_height(")
    assert asp.render(hw["crush_gap"]).count("subject(crush_gap, (R, X)) :- obj(R, robot), obj(X, ") == 4
    assert asp.render(hw["reach_over"]) == ("subject(reach_over, (H, S)) :- obj(H, robot), obj(S, fence).\n"
                                            "subject(reach_over, (H, S)) :- obj(H, robot), obj(S, guard).\n"
                                            "subject(reach_over, (H, S)) :- obj(H, robot), obj(S, bollard).\n"
                                            "subject(reach_over, (H, S)) :- obj(H, robot), obj(S, light_curtain).\n"
                                            "status(reach_over, (H, S), needs_input) :- subject(reach_over, (H, S)).\n")
    assert asp.render(hw["enclosure"]) == ('subject(enclosure, "hazard_zone") :- hazard(_).\n'
                                           'status(enclosure, "hazard_zone", pass) :- subject(enclosure, "hazard_zone"), enclosed("hazard_zone").\n'
                                           'status(enclosure, "hazard_zone", cannot_determine) :- subject(enclosure, "hazard_zone"), not enclosed("hazard_zone").\n')
    assert asp.rule_id("ISO13857:2019/4.4") == "iso13857_2019_4_4" and asp.rule_id("OSHA:1910.36(g)(2)") == "osha_1910_36_g_2"


def test_render_refuses_what_the_engines_cannot_evaluate(pack):
    reasons = {r.rule_id: r.unsupported_reason for r in pack.rules if r.status == "refused"}
    assert "exists" in reasons["ts0011963_rev10_4_3_3"] and "threshold 0" in reasons["ts0011963_rev10_4_3_7"]
    assert "3 selection variables" in reasons["iso13857_2019_table7"] and "is not name(args)" in reasons["iso13855_2010_parallel_approach"]
    assert "zone variable A" in reasons["osha_1910_36_g_2"] and "not photo-checkable" in reasons["ts0011963_rev10_9_1_9"]
    assert reasons["iso13850_2015_4_3"] == "semantic clause without a bool predicate"


@pytest.mark.parametrize("engine_cls", [PythonEngine, Clingo])
def test_parity_with_handwritten_under_both_engines(pack, engine_cls, tmp_path):
    engine = verify.engine_of(engine_cls, {"k": 2}, tmp_path)
    hw = Handwritten().run({"signature": SIGNATURE}, {}, tmp_path)["rule_pack"]
    for scene in CELLS:
        facts = common.synthetic_facts(scene)                       # synthetic-facts@1 + the perimeter_of / covers_opening facts of the cell
        mine, theirs = verify.statuses(engine(facts, pack, scene)), verify.statuses(engine(facts, hw, scene))
        for rid, hw_id in PARITY.items():                            # subjects: (F, hazard_zone) here, F there -> compare on the object
            got = {k[1][0]: s for k, s in mine.items() if k[0] == rid}
            want = {k[1][0]: s for k, s in theirs.items() if k[0] == hw_id}
            assert got and got == want, (scene.scene_id, rid, got, want)
        crush = {k[1]: s for k, s in mine.items() if k[0] == "iso10218_2_2011_trapping_clearance"}
        hw_crush = {k[1]: s for k, s in theirs.items() if k[0] == "crush_gap"}
        assert set(crush) == {k for k in hw_crush if k[1] != "lc"} and set(crush.values()) == {"NEEDS_INPUT"}   # the clause declares restricted_space
    assert {s for s in verify.statuses(engine(common.synthetic_facts(CELLS[1]), pack, CELLS[1])).values()} >= {"NEEDS_MEASUREMENT", "PASS", "NEEDS_INPUT"}


def test_vocabulary_gap_from_the_alignment(inputs, tmp_path):
    alignment = inputs["alignment"].model_copy(deep=True)
    for row in alignment.rows:
        if row.term == "estop":
            row.confidence, row.target = 0.0, ""
    pack = FunctionLibrary().run({**inputs, "alignment": alignment}, {}, tmp_path)["rule_pack"]
    gapped = {r.rule_id: r.unsupported_reason for r in pack.rules if r.status == "vocabulary_gap"}
    assert set(gapped) == {"iec60204_1_2016_10_1_2", "iec60204_1_2016_10_7", "iso13850_2015_4_3", "ts0011963_rev10_4_3_1", "ts0011963_rev10_4_3_3",
                           "ts0011963_rev10_4_3_4", "ts0011963_rev10_4_3_4_upper"}
    assert set(gapped.values()) == {"no Signature entry for: estop"}


def test_retrieved_subset_and_all(inputs, tmp_path):
    ids = inputs["retrieved"][:3]
    assert [r.clause for r in FunctionLibrary().run({**inputs, "retrieved": ids}, {}, tmp_path)["rule_pack"].rules] == sorted(ids, key=asp.rule_id)
    assert len(FunctionLibrary().run({**inputs, "retrieved": []}, {"all": True}, tmp_path)["rule_pack"].rules) == 38


def test_two_runs_dump_byte_identical_packs(inputs, tmp_path):
    a, b = (FunctionLibrary().run(inputs, {}, tmp_path)["rule_pack"] for _ in range(2))
    a.dump(tmp_path / "a.json")
    b.dump(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
