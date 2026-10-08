"""L1 measurement-layer@1: a published measurement layer (web/src/measurement-layer.ts shape) -> Scene.

Port of research/verdict-layer-trial-2026-10-07/adapter_measurement_layer.py: the only code that knows the layer's native units
and box encoding. Classes come from the English labels (a semantic placeholder); photo ids become the object's `views`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ehs_spatial.verdict.contracts import Obj, Scene
from ehs_spatial.verdict.plugins import register

CLASS_BY_WORD = (("light curtain", "light_curtain"), ("fence", "fence"), ("guard", "guard"), ("bollard", "bollard"), ("post", "bollard"),
                 ("robot", "robot"), ("cart", "cart"), ("e-stop", "estop"), ("emergency", "estop"))


def classify(label: str) -> str:
    low = label.lower()
    return next((cls for word, cls in CLASS_BY_WORD if word in low), "other")


def convert(layer: dict, scene_id: str, producer: str) -> Scene:
    ntm = float(layer["scale"]["nativeToMeters"])
    objects = []
    for oid, box in layer.get("boxes", {}).items():
        label = (layer.get("labelsEn") or {}).get(oid) or (layer.get("labels") or {}).get(oid) or box["label"]
        sigma = {k: (None if box["dims"][k]["sigmaCm"] is None else box["dims"][k]["sigmaCm"] / 100.0) for k in ("L", "W", "H", "bottom")}
        views = sorted({str(p) for face in box.get("faces", {}).values() for p in face.get("photos", [])})
        objects.append(Obj(id=oid, cls=classify(label), label=label, center_m=[c * ntm for c in box["centerNative"]], axes=box["axes"],
                           size_m=list(box["sizeM"]), bottom_m=float(box["bottomM"]), top_m=float(box["topM"]), sigma_m=sigma,
                           confidence=box["confidence"], floor_contact=bool(box["floorContact"]), views=views))
    ground = layer.get("ground") or {}
    return Scene(scene_id=scene_id, objects=objects, ground_normal=list(ground.get("normal", [0, 0, 1])),
                 scale_rel_unc=layer["scale"].get("uncertaintyRelative"), producer=producer)


@register("L1", "measurement-layer", "1")
class MeasurementLayer:
    """inputs['source']: path to the layer JSON (or the dict). cfg: scene_id (default: the file's stem)."""

    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        src = inputs["source"]
        layer = src if isinstance(src, dict) else json.loads(Path(src).read_text())
        scene_id = cfg.get("scene_id") or ("scene" if isinstance(src, dict) else Path(src).stem)
        return {"scene": convert(layer, scene_id, f"{self.name}@{self.version}")}
