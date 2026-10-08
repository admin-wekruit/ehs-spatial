"""Sigma policy for C1: a 1-sigma for every unmeasured box dimension, from a per-confidence table.

The table is calibrated on the dimensions that DO carry a sigma in the two benchmark scenes
(research/verdict-layer-trial-2026-10-07/out/scene-090.json, scene-030.json): per confidence level the median of all measured
sigma values (L, W, H, bottom pooled); a level with no data takes the next worse level's value; 'unverified' is at least 5 cm.
README-coverage.md ("Sigma policy") has the table and its caveat. tests/verdict/test_sigma.py recomputes it from the files.
"""
from __future__ import annotations

import json

import numpy as np

from ehs_spatial.verdict.contracts import Scene

DIMS = ("L", "W", "H", "bottom")
LEVELS = ("high", "medium", "low", "unverified")      # best -> worst
UNVERIFIED_MIN_M = 0.05

# calibrate() on the two benchmark scenes, 2026-10-08 (17 / 6 / 2 measured values for medium / low / unverified; high has none)
SIGMA_TABLE_M = {"high": 0.0037, "medium": 0.0037, "low": 0.00395, "unverified": 0.0537}


def calibrate(scenes: list[dict]) -> dict[str, float]:
    """Per-confidence median of the measured sigmas (metres, rounded to 0.01 mm). `scenes` are scene dicts (trial or verdict/1 schema)."""
    pool = {level: [] for level in LEVELS}
    for s in scenes:
        for o in s["objects"]:
            pool[o["confidence"]] += [v for v in (o.get("sigma_m") or {}).values() if v is not None]
    table, worse = {}, None
    for level in reversed(LEVELS):                         # worst first, so a level without data inherits the next worse one
        value = float(np.median(pool[level])) if pool[level] else worse
        if level == "unverified":
            value = max(value or 0.0, UNVERIFIED_MIN_M)
        table[level] = worse = value
    return {level: round(table[level], 5) for level in LEVELS}


def fill_sigma(scene: Scene, table: dict[str, float] = SIGMA_TABLE_M) -> Scene:
    """A copy of `scene` with every None sigma replaced by the table value of the object's confidence. The filled keys are
    recorded in scene.declared_inputs['sigma_filled'] as a JSON list of '<obj_id>:<dim>' (Obj has no flags field)."""
    out = scene.model_copy(deep=True)
    filled = set(json.loads(str(out.declared_inputs.get("sigma_filled", "[]"))))
    for o in out.objects:
        for dim in DIMS:
            if o.sigma_m.get(dim) is None:
                o.sigma_m[dim] = table[o.confidence]
                filled.add(f"{o.id}:{dim}")
    out.declared_inputs["sigma_filled"] = json.dumps(sorted(filled))
    return out
