"""L1 scene-json@1: load an already-converted contract Scene JSON (the benchmark's scenes/*.json). No conversion, no producer change."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ehs_spatial.verdict.contracts import Scene
from ehs_spatial.verdict.plugins import register


@register("L1", "scene-json", "1")
class SceneJson:
    """inputs['source']: path to a Scene JSON (or the dict itself) -> {'scene': Scene}."""

    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        src = inputs["source"]
        return {"scene": Scene.model_validate(src) if isinstance(src, dict) else Scene.load(src)}
