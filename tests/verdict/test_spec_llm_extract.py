"""L4 llm-extract@0 without the API: the splitter on the sample text; the plugin with a fake `llm.complete` that answers from the
reference file (plus one bad candidate: unknown term, ungrounded number, wrong citation, unknown table; and a duplicate); the flags,
the diff, llm_calls, determinism; the run config, and one run through the runner."""
import json
from pathlib import Path

from ehs_spatial.verdict import llm
from ehs_spatial.verdict.contracts import ClauseGraph, Signature
from ehs_spatial.verdict.lab import config, runner
from ehs_spatial.verdict.layers.l4_spec import llm_extract as lx
from ehs_spatial.verdict.plugins import REGISTRY, tag

VERDICT = Path(__file__).resolve().parents[2] / "ehs_spatial/verdict"
SAMPLE = VERDICT / "spec/samples/manual-loading-station-concept-2026-10-08.md"
REFERENCE = ClauseGraph.load(VERDICT / "spec/clauses-ts0011963-v0.json")
SIGNATURE = Signature.load(VERDICT / "signature-v1.json")
REF_IDS = sorted(c.id for c in REFERENCE.clauses)
BAD = lx.Candidate(clause_id="TS0011963:Rev10/8.1.9", title="Mesh panel at least 999 mm", paraphrase="made up", rule_class="geometry",
                   selection=[lx.Binding(var="F", values=["mesh_panel"])], applicability=["perimeter_of(F, Z)"],
                   requirement=lx.Requirement(predicate="top_height", args=["F"], operator=">=", threshold=999, table="ISO13857:2019/Table99",
                                              formula=None, unit="mm", inputs=[]),
                   exceptions=[], tags=["fence"], photo_checkable=True, source_quote="n/a", vocabulary_gaps=["mesh panel"])
BAD_FLAGS = ["uncited", "ungrounded:999", "unknown_table:ISO13857:2019/Table99", "vocabulary_gap:mesh_panel"]


def fake_complete(system, user, schema, model=None, effort=None, cache_dir=None):
    """Every unit gets the reference clauses its citations cover; u01 also BAD, u02 a duplicate of 8.1.4. Device notes count as cached."""
    head, cites = user.splitlines()[:2]
    uid = head.split()[1]
    cands = [lx.candidate_of(c) for c in REFERENCE.clauses if lx.is_cited(c.id, [cites.removeprefix("Citations: ")])]
    cands += [BAD] if uid == "u01" else [lx.candidate_of(REFERENCE.clauses[0])] if uid == "u02" else []
    assert schema is lx.Extraction and "Worked example" in system and "classes: fence" in system
    return lx.Extraction(candidates=cands), {"model": "fake-model", "cached": "device_note" in head, "usage": {}, "stop_reason": "end_turn"}


def test_splitter_on_the_sample():
    units = lx.split_units(SAMPLE.read_text())
    kinds = [u["kind"] for u in units]
    assert kinds.count("requirement") == 9 and kinds.count("device_note") == 2 and len(units) == 11
    assert [u["id"] for u in units] == [f"u{i:02d}" for i in range(1, 12)]
    assert units[0]["citations"] == ["TS-0011963 Rev 10, 8.1.4–8.1.6"] and "1000 mm" in units[0]["text"]
    assert set(lx.standards(SAMPLE.read_text())) == {"TS0011963:Rev10", "ISO13857:2019", "ISO13855:2024", "ISO14120:2015", "ISO13849-1:2023", "ISO12100:2010"}
    cites = {u["id"]: u["citations"] for u in units}
    assert lx.is_cited("TS0011963:Rev10/8.1.5", cites["u01"]) and not lx.is_cited("TS0011963:Rev10/8.1.9", cites["u01"])
    assert lx.is_cited("ISO13855:2024/Formula13", cites["u05"]) and lx.is_cited("ISO13857:2019/Table2", cites["u07"])
    assert lx.is_cited("TS0011963:Rev10/4.3.4-upper", cites["u08"]) and not lx.is_cited("ISO13857:2019/4.1", cites["u02"])
    assert all(any(lx.is_cited(i, c) for c in cites.values()) for i in REF_IDS)   # every reference clause is cited by some unit


def test_plugin_with_fake_model(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "complete", fake_complete)
    assert REGISTRY["L4"]["llm-extract"] is lx.LLMExtract and tag(lx.LLMExtract) == "llm-extract@0"

    def run():
        return lx.LLMExtract().run({"spec_dir": tmp_path / "missing", "signature": SIGNATURE, "scene": None}, {}, tmp_path / "runs" / "r" / "L4")

    out = run()
    graph, report, d = out["clauses"], out["extraction_report"], out["diff"]
    assert isinstance(graph, ClauseGraph) and graph.version == "llm-extract-0+fake-model"
    assert sorted(c.id for c in graph.clauses) == sorted(REF_IDS + [BAD.clause_id])
    assert out["llm_calls"] == report["llm_calls"] == 9                                   # the two device notes were cache hits
    assert report["clauses"][BAD.clause_id]["flags"] == BAD_FLAGS
    assert [i for i, v in report["clauses"].items() if v["flags"]] == [BAD.clause_id]        # reference-derived clauses are clean
    assert report["duplicates"] == [{"clause_id": "TS0011963:Rev10/8.1.4", "unit": "u02"}]
    assert report["vocabulary_gaps"] == ["mesh panel", "mesh_panel"]
    by_id = {c.id: c for c in graph.clauses}
    assert by_id[BAD.clause_id].photo_checkable is False and by_id["TS0011963:Rev10/8.1.4"].requirement == REFERENCE.clauses[0].requirement
    assert by_id["TS0011963:Rev10/9.1.9"].requirement["attribute"] == "resolution_mm" and by_id["TS0011963:Rev10/4.3.4"].standard == "TS-0011963"
    assert {(r.term, r.confidence) for r in out["alignment"].rows if r.term == "mesh_panel"} == {("mesh_panel", 0.0)}
    assert d["same_id_same_requirement"] == REF_IDS and d["ids_only_llm"] == [BAD.clause_id]
    assert d["ids_only_reference"] == [] and d["same_id_different_requirement"] == []
    assert graph.tables == REFERENCE.tables and graph.definitions == REFERENCE.definitions
    assert out["retrieved"] == [c.id for c in graph.clauses]
    again = run()
    assert again["clauses"].model_dump_json() == graph.model_dump_json()
    assert json.dumps(again["extraction_report"], sort_keys=True) == json.dumps(report, sort_keys=True)


def test_config_and_runner(monkeypatch, tmp_path):
    cfg = config.load(VERDICT / "configs/llm-extract-ts.yaml")
    assert cfg["L4"] == {"plugin": "llm-extract", "spec_text": "samples/manual-loading-station-concept-2026-10-08.md", "reference": "clauses-ts0011963-v0.json"}
    assert runner.tags(cfg)["L4"] == "llm-extract@0"
    monkeypatch.setattr(llm, "complete", fake_complete)
    cfg["item"] = "090"
    run_dir = tmp_path / runner.run(cfg, tmp_path, "llm-extract-test")
    l4 = {p.stem: json.loads(p.read_text()) for p in (run_dir / "L4").glob("*.json")}
    assert set(l4) == {"clauses", "alignment", "retrieved", "extraction_report", "diff"}
    assert l4["diff"]["same_id_same_requirement"] == REF_IDS and "TS0011963:Rev10/8.1.5" in l4["retrieved"]
    ledger = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert ledger[-1]["llm_calls"] == 9 and ledger[-1]["plugins"]["L4"] == "llm-extract@0"
