"""L1 scene-json@1: load an already-converted contract Scene JSON (the benchmark's scenes/*.json). No conversion, no producer change.
cfg `declared: <path>` merges that declared.json (`panoptes verdict labels declared`; {scene_id: {input: value}}) into
Scene.declared_inputs, the file winning over the scene's own values."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ehs_spatial.verdict.contracts import Scene
from ehs_spatial.verdict.plugins import register


@register("L1", "scene-json", "1")
class SceneJson:
    """inputs['source']: path to a Scene JSON (or the dict itself) -> {'scene': Scene}."""

    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        src = inputs["source"]
        scene = Scene.model_validate(src) if isinstance(src, dict) else Scene.load(src)
        if cfg.get("declared"):
            extra = json.loads(Path(cfg["declared"]).read_text()).get(scene.scene_id, {})
            scene = Scene.model_validate({**scene.model_dump(), "declared_inputs": {**scene.declared_inputs, **extra}})
        return {"scene": scene}
