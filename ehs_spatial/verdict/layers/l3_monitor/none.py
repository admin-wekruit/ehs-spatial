"""L3 none@1: no perception monitor; the Facts pass through unchanged (no quality findings, nothing flagged untrusted)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ehs_spatial.verdict.plugins import register


@register("L3", "none", "1")
class NoMonitor:
    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        return {"facts": inputs["facts"]}
