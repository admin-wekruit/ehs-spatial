"""The customer's TS-0011963 clauses (transcribed 2026-10-08) load, merge with the ISO graph, align and retrieve."""
from pathlib import Path

from ehs_spatial.verdict.contracts import ClauseGraph, Scene, Signature
from ehs_spatial.verdict.layers.l4_spec.alignment import align
from ehs_spatial.verdict.layers.l4_spec.clause_kg import ClauseKG
from ehs_spatial.verdict.layers.l4_spec.tables import ts0011963_table10_3

ROOT = Path(__file__).resolve().parents[2] / "ehs_spatial" / "verdict"


def test_ts_file_validates_and_aligns():
    g = ClauseGraph.load(ROOT / "spec" / "clauses-ts0011963-v0.json")
    assert len(g.clauses) >= 15 and all(not c.verified for c in g.clauses)
    rows = align(g, Signature.load(ROOT / "signature-v1.json")).rows
    terms = {r.term for r in rows}; gaps = sorted({r.term for r in rows if r.confidence == 0})
    assert gaps == [], gaps
    assert len(terms) > 20


def test_table10_3_steps():
    assert [ts0011963_table10_3(k) for k in (0, 149, 150, 299, 300, 1000)] == [150, 150, 500, 500, 700, 700]


def test_merged_graph_retrieves_ts_clauses_for_090(tmp_path):
    scene = Scene.load(ROOT / "benchmark" / "v0" / "scenes" / "scene-090.json")
    out = ClauseKG().run({"spec_dir": ROOT / "spec", "signature": Signature.load(ROOT / "signature-v1.json"), "scene": scene},
                         {"clauses_file": ["clauses-v0.json", "clauses-ts0011963-v0.json"]}, tmp_path)
    ids = out["retrieved"]
    assert "TS0011963:Rev10/8.1.5" in ids and "TS0011963:Rev10/9.1.7" in ids and "ISO13857:2019/4.4" in ids
    assert "TS0011963:Rev10/11.1.2" not in ids   # no stack light in the 090 scene
    assert out["clauses"].version.startswith("0+ts0011963")
