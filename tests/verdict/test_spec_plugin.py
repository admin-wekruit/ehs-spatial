"""Alignment coverage, retrieval on the two benchmark scenes, and the clause-kg@0 plugin (registered, deterministic)."""
import json
from pathlib import Path

from ehs_spatial.verdict.contracts import ClauseGraph, Scene, Signature
from ehs_spatial.verdict.layers.l4_spec.alignment import align, coverage
from ehs_spatial.verdict.layers.l4_spec.clause_kg import SPEC_DIR, ClauseKG
from ehs_spatial.verdict.layers.l4_spec.retrieve import retrieve
from ehs_spatial.verdict.plugins import REGISTRY, tag

ROOT = Path(__file__).resolve().parents[2]
TRIAL_OUT = ROOT / "research/verdict-layer-trial-2026-10-07/out"
GRAPH = ROOT / "ehs_spatial/verdict/spec/clauses-v0.json"
SIGNATURE = ROOT / "ehs_spatial/verdict/signature-v1.json"

GAP_TERMS: set[str] = set()   # every term aligns since signature-v1 gained aisle_width / climbing_aids / agv
GEOMETRY_CLAUSES = {   # what the fence / guard / light-curtain / robot scenes must retrieve
    "ISO13857:2019/4.4", "ISO13857:2019/Table2", "ISO13857:2019/Table2-min-height", "ISO13857:2019/Table4", "ISO13857:2019/Table7",
    "ISO13855:2010/normal-approach", "ISO13855:2010/parallel-approach", "ISO13855:2010/multi-beam",
    "ISO13854:2017/Table1", "ISO10218-2:2011/trapping-clearance", "ISO10218-2:2011/perimeter-safeguarding", "ISO14120:2015/5.18",
}


def load_trial_scene(path: Path) -> Scene:
    """The trial's scene JSON predates the contract: schema tag differs, views are ints; zones / coverage / declared_inputs /
    producer are absent and take the contract defaults."""
    d = json.loads(path.read_text())
    d["schema"] = "verdict/1"
    for o in d["objects"]:
        o["views"] = [str(v) for v in o["views"]]
    return Scene.model_validate(d)


def test_alignment_rows_and_coverage():
    graph, sig = ClauseGraph.load(GRAPH), Signature.load(SIGNATURE)
    al = align(graph, sig)
    known = set(sig.classes) | set(sig.zones) | set(sig.predicates) | set(sig.attributes) | set(sig.declared_inputs)
    assert al.signature_version == sig.version and al.rows
    for r in al.rows:
        if r.confidence == 1.0:
            assert r.target == r.term and r.source == "exact"
        elif r.confidence == 0.8:
            assert r.target in known and r.target in graph.definitions[r.term] and r.source == "synonym"
        else:
            assert r.confidence == 0.0 and r.target == ""
    assert {r.term for r in al.rows if r.confidence == 0.0} == GAP_TERMS
    assert coverage(al) >= 0.9
    # kinds follow the Signature: zones used in a selection are 'zone', aliases take their target's kind
    kinds = {(r.term, r.target): r.kind for r in al.rows}
    assert kinds[("hazard_zone", "hazard_zone")] == "zone" and kinds[("ESPE", "light_curtain")] == "class"
    assert kinds[("floor gap", "floor_gap")] == "predicate" and kinds[("interlock", "interlock")] == "attribute"
    assert {r.kind for r in al.rows if r.term == "resolution_mm"} == {"attribute", "input"}
    assert al.model_dump_json() == align(graph, sig).model_dump_json()


def test_retrieve_on_benchmark_scenes():
    graph = ClauseGraph.load(GRAPH)
    all_ids = [c.id for c in graph.clauses]
    assert retrieve(graph, None) == all_ids
    for name in ("scene-030.json", "scene-090.json"):
        ids = retrieve(graph, load_trial_scene(TRIAL_OUT / name))
        assert GEOMETRY_CLAUSES <= set(ids), name
        assert "ISO3691-4:2020/path-clearance" not in ids, name
        assert not any(i.startswith(("OSHA", "NFPA", "IEC60204", "ISO13850")) for i in ids), name   # no e-stop / aisle in the scene
        assert ids == [i for i in all_ids if i in set(ids)], name                                      # graph order


def test_plugin_registered_and_deterministic(tmp_path):
    assert REGISTRY["L4"]["clause-kg"] is ClauseKG and tag(ClauseKG) == "clause-kg@0"
    assert (SPEC_DIR / "clauses-v0.json").exists()
    sig, scene = Signature.load(SIGNATURE), load_trial_scene(TRIAL_OUT / "scene-090.json")
    out = ClauseKG().run({"spec_dir": None, "signature": sig, "scene": scene}, {}, tmp_path)
    again = ClauseKG().run({"spec_dir": SPEC_DIR, "signature": sig, "scene": scene}, {"clauses_file": "clauses-v0.json"}, tmp_path)
    assert isinstance(out["clauses"], ClauseGraph) and out["clauses"].model_dump_json() == again["clauses"].model_dump_json()
    assert out["alignment"].model_dump_json() == again["alignment"].model_dump_json()
    assert out["retrieved"] == again["retrieved"] == retrieve(out["clauses"], scene)
    assert ClauseKG().run({}, {}, tmp_path)["retrieved"] == [c.id for c in out["clauses"].clauses]   # defaults: everything
