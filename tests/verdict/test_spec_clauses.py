"""spec/clauses-v0.json: validates as a ClauseGraph, references only Signature terms (or the listed gaps), tables resolve."""
import importlib
import inspect
import re
from pathlib import Path

from ehs_spatial.verdict.contracts import ClauseGraph, Signature

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "ehs_spatial/verdict/spec/clauses-v0.json"
SIGNATURE = ROOT / "ehs_spatial/verdict/signature-v1.json"
_CALL = re.compile(r"\b([a-z_][a-z0-9_]*)\(")

# Terms the clauses need that signature-v1 does not have. Each is a deliberate 0.0 row in the alignment (see l4_spec/README.md).
KNOWN_GAPS: set[str] = set()   # aisle_width / climbing_aids are in signature-v1 now
KNOWN_TAG_GAPS: set[str] = set()   # agv is a signature-v1 class now

EXPECTED_IDS = {
    "ISO13857:2019/4.4", "ISO13857:2019/Table2", "ISO13857:2019/Table2-min-height", "ISO13857:2019/Table4", "ISO13857:2019/Table7",
    "ISO13855:2010/normal-approach", "ISO13855:2010/parallel-approach", "ISO13855:2010/multi-beam",
    "ISO13854:2017/Table1", "ISO10218-2:2011/trapping-clearance", "ISO10218-2:2011/perimeter-safeguarding", "ISO14120:2015/5.18",
    "ISO13850:2015/4.3", "IEC60204-1:2016/10.7", "IEC60204-1:2016/10.1.2",
    "OSHA:1910.176(a)-marked", "OSHA:1910.176(a)-clear", "OSHA:1910.36(g)(2)", "NFPA101/aisle-width-existing",
    "ISO3691-4:2020/path-clearance",
}


def test_graph_validates():
    graph = ClauseGraph.load(GRAPH)
    ids = [c.id for c in graph.clauses]
    assert len(ids) == len(set(ids))
    assert EXPECTED_IDS <= set(ids)
    assert graph.version == "0"
    # vendor reproductions only: nothing was checked against a purchased text
    assert not any(c.verified for c in graph.clauses) and not any(t.verified for t in graph.tables)
    assert all(c.paraphrase and c.source_url and c.tags for c in graph.clauses)


def test_clauses_reference_signature_or_known_gaps():
    graph, sig = ClauseGraph.load(GRAPH), Signature.load(SIGNATURE)
    classes_zones, table_ids = set(sig.classes) | set(sig.zones), {t.id for t in graph.tables}
    for c in graph.clauses:
        for var, values in c.selection.items():
            assert set(values) <= classes_zones, (c.id, var, values)
        for a in c.applicability:
            assert set(_CALL.findall(a)) <= set(sig.predicates), (c.id, a)
        r = c.requirement
        assert r.get("predicate") or r.get("attribute"), c.id
        if r.get("predicate"):
            assert r["predicate"] in sig.predicates or r["predicate"] in KNOWN_GAPS, (c.id, r["predicate"])
        if r.get("attribute"):
            assert r["attribute"] in sig.attributes or r["attribute"] in KNOWN_GAPS, (c.id, r["attribute"])
        assert set(r.get("predicates", [])) <= set(sig.predicates), c.id
        assert set(r.get("attributes", [])) <= set(sig.attributes), c.id
        assert set(r.get("inputs", [])) <= set(sig.declared_inputs), (c.id, r.get("inputs"))
        assert r["operator"] in {"<=", ">=", "==", "in", "exists", "not_exists"}, c.id
        assert ("threshold" in r) or ("table" in r) or ("formula" in r), c.id
        if "table" in r:
            assert r["table"] in table_ids, (c.id, r["table"])
        assert set(c.exceptions) <= set(sig.attributes), c.id
        assert set(c.definitions) <= set(graph.definitions), c.id
        assert set(c.tags) <= classes_zones | KNOWN_TAG_GAPS, (c.id, c.tags)


def test_tables_resolve_to_functions():
    graph = ClauseGraph.load(GRAPH)
    for t in graph.tables:
        module, name = t.function.split(":")
        fn = getattr(importlib.import_module(module), name)
        assert callable(fn) and list(inspect.signature(fn).parameters) == t.inputs, t.id
        assert t.output_unit == "mm"
