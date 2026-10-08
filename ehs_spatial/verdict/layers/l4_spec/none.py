"""L4 none@1: no specification side. An empty clause graph, an empty alignment table and nothing retrieved: what a hand-written
L5 pack needs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ehs_spatial.verdict.contracts import Alignment, ClauseGraph
from ehs_spatial.verdict.plugins import register


@register("L4", "none", "1")
class NoSpec:
    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        return {"clauses": ClauseGraph(version="none", clauses=[]).model_dump(),
                "alignment": Alignment(signature_version=inputs["signature"].version, rows=[]).model_dump(),
                "retrieved": []}
