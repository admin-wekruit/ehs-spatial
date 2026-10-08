"""synthetic-facts@1: a minimal, documented Scene -> Facts for AXIS-ALIGNED boxes (the scenes of synth/scenes.py).

Not a registered L2 plugin and not a replacement for layers/l2_facts: it exists so the verification harness (layers/l5_rules/verify.py)
and the threshold grid (synth/grid.py) run without the real L2, and as the oracle the real L2 must agree with on axis-aligned scenes
(verify.facts_parity). Everything is deterministic: object order of the Scene, integer millimetres, no randomness.

Predicates emitted (Signature v1):
  obj(id, cls)            args = [id, cls]   (the class rides in args; Fact.value cannot hold a string)
  top_height(id)          top_m                       U from [sigma bottom, sigma H]
  bottom_height(id)       bottom_m (negative = below the fitted floor)   U from [sigma bottom]
  floor_gap(id)           max(bottom_m, 0)            same U
  min_distance_3d(a, b)   closest distance between the two AABBs (0 when they touch / overlap)   U from [plan sigma a, plan sigma b]
  horizontal_gap(a, b)    plan-view AABB gap                                                        same U
  z_overlap(a, b)         overlap of the two height ranges, clamped at 0                            U from [sigma H a, sigma H b]
  reach_over(h, s)        hazard (robot) x fixed structure pairs: value = horizontal gap c            U of the horizontal gap
                          (a = hazard top, b = structure top are NOT carried: Fact has no field for them -> contract request)
  grid                    plan occupancy: blocked = fixed-class footprints, hazard = robot footprints, outside = border cells,
                          observed = resampled from Scene.coverage (None when the scene has no coverage)
Uncertainty mirrors the trial (research/verdict-layer-trial-2026-10-07/scene_graph.py u_mm): U = 2 * sqrt(sum sigma^2 + (rel * |v|)^2),
sigma None -> 0.05 m ('default_sigma'), scale_rel_unc None -> 0.02 ('default_scale_unc'); values and U are rounded to integer mm.
Known differences from the trial's relations.py: a footprint is expanded by half a cell with SQUARE corners (the trial buffers the convex
hull, i.e. round corners) -> grid cells may differ at box corners; pair facts are emitted for every unordered pair, not only hazard/fixed.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from ehs_spatial.verdict.contracts import Fact, Facts, Grid, Scene

PRODUCER = "synthetic-facts@1"
K = 2
DEFAULT_SIGMA_M = 0.05
DEFAULT_SCALE_REL = 0.02
FIXED = ("fence", "guard", "bollard", "light_curtain")
HAZARD = ("robot",)
CELL_M = 0.10
MARGIN_M = 1.0


def mm(x: float) -> int:
    return int(round(x * 1000))


def u_mm(value_m: float, sigmas: list[float | None], rel: float | None) -> tuple[int, list[str]]:
    terms = [DEFAULT_SIGMA_M if s is None else s for s in sigmas]
    flags = sorted({"default_sigma" for s in sigmas if s is None} | ({"default_scale_unc"} if rel is None else set()))
    rel = DEFAULT_SCALE_REL if rel is None else rel
    return mm(K * math.sqrt(sum(t * t for t in terms) + (rel * abs(value_m)) ** 2)), flags


def plan_basis(normal) -> tuple[np.ndarray, np.ndarray]:
    """Same construction as the trial's relations.plan_basis so grids share the frame."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    e1 = np.cross(n, [1, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1, 0])
    e1 = e1 / np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def corners(o) -> np.ndarray:
    c, a, h = np.asarray(o.center_m, float), np.asarray(o.axes, float), np.asarray(o.size_m, float) / 2
    return np.array([c + a[0] * sx * h[0] + a[1] * sy * h[1] + a[2] * sz * h[2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def aabb(o) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(o.axes, float)
    if not np.allclose(np.abs(a).max(axis=1), 1.0, atol=1e-6) or not np.allclose(np.abs(a).sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{PRODUCER} handles axis-aligned boxes only; object {o.id} axes {o.axes}")
    pts = corners(o)
    return pts.min(axis=0), pts.max(axis=0)


def _gap(lo_a, hi_a, lo_b, hi_b) -> np.ndarray:
    return np.maximum(0.0, np.maximum(lo_b - hi_a, lo_a - hi_b))


def plan_sigma(o) -> float | None:
    return max((s for s in (o.sigma_m.get("L"), o.sigma_m.get("W")) if s is not None), default=None)


def grid_frame(objects, normal=(0, 0, 1), cell_m: float = CELL_M, margin_m: float = MARGIN_M) -> dict:
    """Plan grid covering every footprint plus a margin: origin (plan coords), shape, basis. Shared by scenes.py (Coverage) and plan_grid."""
    e1, e2 = plan_basis(normal)
    xy = np.array([[p @ e1, p @ e2] for o in objects for p in corners(o)])
    # origin snapped to the cell lattice (the trial uses min - margin): a 90-degree rotation or a cell-multiple shift of the scene then
    # maps cells onto cells, so a Coverage written on this frame stays cell-exact under verify.transform
    lo = [math.floor(v / cell_m + 1e-9) * cell_m - margin_m for v in xy.min(axis=0)]
    hi = [math.ceil(v / cell_m - 1e-9) * cell_m + margin_m for v in xy.max(axis=0)]
    shape = [int(round((hi[0] - lo[0]) / cell_m)), int(round((hi[1] - lo[1]) / cell_m))]
    return {"cell_m": cell_m, "origin_xy": [float(lo[0]), float(lo[1])], "basis": [e1.tolist(), e2.tolist()], "shape": shape}


def footprint_cells(o, frame, normal) -> set[tuple[int, int]]:
    e1, e2 = plan_basis(normal)
    xy = np.array([[p @ e1, p @ e2] for p in corners(o)])
    (minx, miny), (maxx, maxy) = xy.min(axis=0), xy.max(axis=0)
    cell, (ox, oy) = frame["cell_m"], frame["origin_xy"]
    nx, ny = frame["shape"]
    out = set()
    for ix in range(max(0, int((minx - ox) / cell) - 1), min(nx, int((maxx - ox) / cell) + 2)):
        cx = ox + (ix + 0.5) * cell
        if not (minx - cell / 2 <= cx <= maxx + cell / 2):
            continue
        for iy in range(max(0, int((miny - oy) / cell) - 1), min(ny, int((maxy - oy) / cell) + 2)):
            cy = oy + (iy + 0.5) * cell
            if miny - cell / 2 <= cy <= maxy + cell / 2:
                out.add((ix, iy))
    return out


def observed_cells(scene: Scene, frame) -> list[list[int]] | None:
    """Resample Scene.coverage onto the facts grid: a facts cell is observed when its centre falls in an observed coverage cell."""
    cov = scene.coverage
    if cov is None:
        return None
    seen = {tuple(c) for c in cov.observed}
    e1, e2 = np.asarray(frame["basis"][0]), np.asarray(frame["basis"][1])
    c1, c2 = np.asarray(cov.basis[0]), np.asarray(cov.basis[1])
    cell, (ox, oy) = frame["cell_m"], frame["origin_xy"]
    out = []
    for ix in range(frame["shape"][0]):
        for iy in range(frame["shape"][1]):
            p = e1 * (ox + (ix + 0.5) * cell) + e2 * (oy + (iy + 0.5) * cell)
            jx = int(math.floor((p @ c1 - cov.origin_xy[0]) / cov.cell_m))
            jy = int(math.floor((p @ c2 - cov.origin_xy[1]) / cov.cell_m))
            if (jx, jy) in seen:
                out.append([ix, iy])
    return out


def plan_grid(scene: Scene) -> Grid:
    frame = grid_frame(scene.objects, scene.ground_normal)
    blocked, hazard = set(), set()
    for o in scene.objects:
        if o.cls in FIXED:
            blocked |= footprint_cells(o, frame, scene.ground_normal)
        elif o.cls in HAZARD:
            hazard |= footprint_cells(o, frame, scene.ground_normal)
    nx, ny = frame["shape"]
    outside = {(ix, iy) for ix in range(nx) for iy in range(ny) if ix in (0, nx - 1) or iy in (0, ny - 1)}
    return Grid(cell_m=frame["cell_m"], origin_xy=frame["origin_xy"], basis=frame["basis"], shape=frame["shape"],
                blocked=[list(c) for c in sorted(blocked)], hazard=[list(c) for c in sorted(hazard - blocked)],
                outside=[list(c) for c in sorted(outside)], observed=observed_cells(scene, frame))


def facts_of(scene: Scene) -> Facts:
    rel = scene.scale_rel_unc
    e1, e2 = plan_basis(scene.ground_normal)
    boxes = {o.id: aabb(o) for o in scene.objects}
    fs: list[Fact] = []
    for o in scene.objects:
        views = [str(v) for v in o.views]
        ub, fb = u_mm(o.bottom_m, [o.sigma_m.get("bottom")], rel)
        ut, ft = u_mm(o.top_m, [o.sigma_m.get("bottom"), o.sigma_m.get("H")], rel)
        fs.append(Fact(pred="obj", args=[o.id, o.cls]))
        fs.append(Fact(pred="top_height", args=[o.id], value=mm(o.top_m), unit="mm", u=ut, views=views, flags=ft))
        fs.append(Fact(pred="bottom_height", args=[o.id], value=mm(o.bottom_m), unit="mm", u=ub, views=views, flags=fb))
        fs.append(Fact(pred="floor_gap", args=[o.id], value=mm(max(o.bottom_m, 0.0)), unit="mm", u=ub, views=views, flags=fb))
    for a, b in itertools.combinations(scene.objects, 2):
        views = sorted(set(str(v) for v in a.views) | set(str(v) for v in b.views))
        (la, ha), (lb, hb) = boxes[a.id], boxes[b.id]
        g = _gap(la, ha, lb, hb)
        d = float(math.sqrt((g * g).sum()))
        hg = float(math.sqrt(sum((g @ e) ** 2 for e in (e1, e2))))   # plan-view gap of the AABBs (axes are aligned with the plan basis)
        zo = min(a.top_m, b.top_m) - max(a.bottom_m, b.bottom_m)
        ud, fd = u_mm(d, [plan_sigma(a), plan_sigma(b)], rel)
        uh, fh = u_mm(hg, [plan_sigma(a), plan_sigma(b)], rel)
        uz, fz = u_mm(zo, [a.sigma_m.get("H"), b.sigma_m.get("H")], rel)
        fs.append(Fact(pred="min_distance_3d", args=[a.id, b.id], value=mm(d), unit="mm", u=ud, views=views, flags=fd))
        fs.append(Fact(pred="horizontal_gap", args=[a.id, b.id], value=mm(hg), unit="mm", u=uh, views=views, flags=fh))
        fs.append(Fact(pred="z_overlap", args=[a.id, b.id], value=mm(max(zo, 0.0)), unit="mm", u=uz, views=views, flags=fz))
        if (a.cls in HAZARD and b.cls in FIXED) or (b.cls in HAZARD and a.cls in FIXED):
            hz, st = (a, b) if a.cls in HAZARD else (b, a)
            fs.append(Fact(pred="reach_over", args=[hz.id, st.id], value=mm(hg), unit="mm", u=uh, views=views,
                           flags=fh + [f"a_mm={mm(hz.top_m)}", f"b_mm={mm(st.top_m)}"]))   # a / b ride in flags, as relations@1 does
    return Facts(scene_id=scene.scene_id, signature_version="1", facts=fs, grid=plan_grid(scene), producer=PRODUCER)
