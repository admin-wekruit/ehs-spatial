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
        clauses = ClauseGraph.load(spec_dir / cfg.get("clauses_file", "clauses-v0.json"))
        signature = inputs.get("signature") or Signature.load(SIGNATURE_PATH)
        return {"clauses": clauses, "alignment": align(clauses, signature), "retrieved": retrieve(clauses, inputs.get("scene"))}
