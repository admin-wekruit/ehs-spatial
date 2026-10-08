"""L3 perception monitor `stpl@1`: is each object of the Scene trustworthy enough to be judged? README-stpl.md.

Three checks per object; one failure makes it 'untrusted' (a Fact the rules turn into CANNOT_DETERMINE):
  views   seen by fewer than cfg.min_views photos / frames (default 2)
  floor   floor_contact claimed but |bottom_m| > cfg.floor_tol_m (default 0.05)
  size    a side outside the class's plausible range (SIZE_RANGES_M; classes without a row are not size-checked)
Writes facts.quality = {monitor, untrusted, coverage_ratio, checks} and flags the existing facts of untrusted objects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ehs_spatial.verdict.contracts import Fact, Facts, Scene
from ehs_spatial.verdict.plugins import register, tag

# class -> ((thin side), (long side), (height)) in metres, applied to (min(L, W), max(L, W), H): the box's L / W order is arbitrary
SIZE_RANGES_M: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    "fence":            ((0.05, 0.6), (0.3, 10.0), (0.8, 3.5)),
    "guard":            ((0.02, 3.0), (0.1, 10.0), (0.1, 3.0)),
    "bollard":          ((0.05, 0.5), (0.05, 0.5), (0.3, 1.5)),
    "light_curtain":    ((0.02, 0.3), (0.02, 0.3), (0.1, 2.5)),
    "robot":            ((0.3, 3.0), (0.3, 3.0), (0.3, 4.0)),
    "cart":             ((0.3, 2.0), (0.3, 3.0), (0.3, 2.0)),
    "estop":            ((0.02, 0.3), (0.02, 0.3), (0.02, 0.3)),
    "interlocked_door": ((0.02, 0.3), (0.5, 3.0), (1.0, 3.0)),
    "control_panel":    ((0.1, 1.5), (0.1, 2.0), (0.2, 2.5)),
    "person":           ((0.2, 1.0), (0.2, 1.0), (0.5, 2.2)),
}


def failed_checks(o, min_views: int, floor_tol_m: float) -> list[str]:
    failed = []
    if len(o.views) < min_views:
        failed.append("views")
    if o.floor_contact and abs(o.bottom_m) > floor_tol_m:
        failed.append("floor")
    ranges = SIZE_RANGES_M.get(o.cls)
    sides = (min(o.size_m[:2]), max(o.size_m[:2]), o.size_m[2])
    if ranges and not all(lo <= v <= hi for v, (lo, hi) in zip(sides, ranges)):
        failed.append("size")
    return failed


@register("L3", "stpl", "1")
class Stpl:
    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        scene: Scene = inputs["scene"]
        facts: Facts = inputs["facts"].model_copy(deep=True)
        min_views, floor_tol = int(cfg.get("min_views", 2)), float(cfg.get("floor_tol_m", 0.05))
        checks, untrusted = {}, []
        for o in scene.objects:
            failed = failed_checks(o, min_views, floor_tol)
            checks[o.id] = {"views": len(o.views), "bottom_m": o.bottom_m, "floor_contact": o.floor_contact, "size_m": list(o.size_m), "failed": failed}
            if failed:
                untrusted.append(o.id)
        for f in facts.facts:
            if any(a in untrusted for a in f.args) and "untrusted" not in f.flags:
                f.flags.append("untrusted")
        for oid in untrusted:
            facts.facts.append(Fact(pred="untrusted", args=[oid], value=1, unit="bool", flags=checks[oid]["failed"]))
        cov = scene.coverage
        facts.quality = {"monitor": tag(self), "untrusted": untrusted,
                         "coverage_ratio": len(cov.observed) / (cov.shape[0] * cov.shape[1]) if cov else None, "checks": checks}
        return {"facts": facts}
