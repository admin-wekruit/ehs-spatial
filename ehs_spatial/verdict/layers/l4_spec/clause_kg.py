"""L4 plugin clause-kg@0: loads the hand-extracted clause graph (spec/clauses-v0.json), aligns its terms to the Signature and
retrieves the clauses whose tags match the scene's classes / zones. See README.md in this directory."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ehs_spatial.verdict.contracts import ClauseGraph, Signature
from ehs_spatial.verdict.layers.l4_spec.alignment import align
from ehs_spatial.verdict.layers.l4_spec.retrieve import retrieve
from ehs_spatial.verdict.plugins import register

VERDICT_DIR = Path(__file__).resolve().parents[2]
SPEC_DIR = VERDICT_DIR / "spec"
SIGNATURE_PATH = VERDICT_DIR / "signature-v1.json"


@register("L4", "clause-kg", "0")
class ClauseKG:
    """inputs: spec_dir (default: the package's spec/), signature (Signature; default: signature-v1.json), scene (Scene | None).
    cfg: clauses_file (default 'clauses-v0.json'). Output: clauses (ClauseGraph), alignment (Alignment), retrieved (clause ids)."""

    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        spec_dir = Path(inputs.get("spec_dir") or SPEC_DIR)
        files = cfg.get("clauses_file", "clauses-v0.json")
        files = [files] if isinstance(files, str) else list(files)
        if not (spec_dir / files[0]).exists():   # the runner offers <benchmark>/spec as an override; the package spec/ is the default
            spec_dir = SPEC_DIR
        graphs = [ClauseGraph.load(spec_dir / f) for f in files]
        clauses = graphs[0]
        for g in graphs[1:]:   # merge: clauses and tables append, standards by id, definitions by term (later files win on a term)
            clauses.clauses += g.clauses; clauses.tables += g.tables
            seen = {st["id"] for st in clauses.standards}; clauses.standards += [st for st in g.standards if st["id"] not in seen]
            clauses.definitions.update(g.definitions); clauses.version = clauses.version + "+" + g.version
        signature = inputs.get("signature") or Signature.load(SIGNATURE_PATH)
        return {"clauses": clauses, "alignment": align(clauses, signature), "retrieved": retrieve(clauses, inputs.get("scene"))}
