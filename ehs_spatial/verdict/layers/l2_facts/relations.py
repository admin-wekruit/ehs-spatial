"""L2 relations@1: Scene -> Facts, the trial's relation library (research/verdict-layer-trial-2026-10-07/{relations,scene_graph}.py)
emitting the predicates of signature-v1.json. Reads the Scene only (metric oriented boxes); rules never compute geometry.

Per object   top_height, bottom_height, floor_gap (= bottom_height clamped at 0)
Per pair     (a hazard-class object involved, or both fixed-class; pairs in the Scene's object order)
             min_distance_3d   closest surface points of the two oriented boxes, 0 when they overlap
             horizontal_gap    plan-view footprint gap, 0 when the footprints overlap          (ISO 13857 Table 2 "c")
             z_overlap         overlap of the two height ranges, 0 = one entirely above the other (flag separated_vertically)
             above             a entirely above b with overlapping footprints; value = the clearance
             line_of_sight     1 when no third box crosses the segment between the closest points (flags blocked_by=<id>)
             reach_over        hazard x fixed only: value = c (horizontal gap), flags a_mm= (hazard top), b_mm= (structure top)
Grid         plan-view occupancy: blocked = fixed-class footprints, hazard = hazard-class footprints, outside = border cells;
             Scene.coverage.observed is copied onto it by coordinate when present (None otherwise = unknown everywhere)

Units: Scene metres -> integer millimetres. u = k * sqrt(sum sigma^2 + (rel * |value|)^2), expanded (k = 2); a missing sigma is
default_sigma_m (flag default_sigma), a missing Scene.scale_rel_unc is default_scale_rel (flag default_scale_unc).
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import MultiPoint, Point

from ehs_spatial.verdict.contracts import Fact, Facts, Grid, Obj, Scene
from ehs_spatial.verdict.plugins import register

DEFAULTS: dict[str, Any] = dict(k=2, default_sigma_m=0.05, default_scale_rel=0.02, cell_m=0.10, margin_m=1.0, face_samples=9,
                                fixed=["fence", "guard", "bollard", "light_curtain"], hazard=["robot"], signature_version="1")


def mm(x: float) -> int:
    return int(round(x * 1000))


def frame(o: Obj):
    return np.asarray(o.center_m, float), np.asarray(o.axes, float), np.asarray(o.size_m, float) / 2


def corners(o: Obj) -> np.ndarray:
    c, a, h = frame(o)
    return np.array([c + a[0] * sx * h[0] + a[1] * sy * h[1] + a[2] * sz * h[2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def surface_points(o: Obj, n: int) -> np.ndarray:
    """n x n points on each of the six faces of the oriented box."""
    c, a, h = frame(o)
    t = np.linspace(-1, 1, n)
    pts = []
    for fixed in range(3):
        u, v = [k for k in range(3) if k != fixed]
        for side in (-1, 1):
            tu, tv = np.meshgrid(t, t)
            pts.append(c + a[fixed] * side * h[fixed] + np.outer(tu.ravel(), a[u] * h[u]) + np.outer(tv.ravel(), a[v] * h[v]))
    return np.concatenate(pts)


def inside(o: Obj, points: np.ndarray) -> bool:
    """Some point lies inside o: |coordinate| <= 1 on every axis of o's box frame divided by the half extents."""
    c, a, h = frame(o)
    return bool(np.any(np.all(np.abs(((points - c) @ a.T) / np.maximum(h, 1e-9)) <= 1, axis=1)))


def closest(pa: np.ndarray, pb: np.ndarray):
    d2 = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(d2.argmin(), d2.shape)
    return pa[i], pb[j], float(math.sqrt(d2[i, j]))


def segment_blocked(p, q, scene: Scene, exclude: set[str], n: int = 40) -> list[str]:
    pts = np.linspace(p, q, n)[1:-1]
    return [o.id for o in scene.objects if o.id not in exclude and inside(o, pts)]


def plan_basis(normal):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    e1 = np.cross(n, [1, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1, 0])
    e1 = e1 / np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def plan_grid(scene: Scene, foot: dict, e1, e2, blockers: set[str], hazards: set[str], cell_m: float, margin_m: float) -> Grid:
    allpts = np.array([xy for g in foot.values() for xy in g.exterior.coords])
    lo, hi = allpts.min(0) - margin_m, allpts.max(0) + margin_m
    nx, ny = int(np.ceil((hi[0] - lo[0]) / cell_m)), int(np.ceil((hi[1] - lo[1]) / cell_m))
    blocked, hazard = set(), set()
    for o in scene.objects:
        if o.cls not in blockers and o.cls not in hazards:
            continue
        g = foot[o.id]
        minx, miny, maxx, maxy = g.bounds
        band = g.buffer(cell_m * 0.5)
        for ix in range(int((minx - lo[0]) / cell_m), int((maxx - lo[0]) / cell_m) + 1):
            for iy in range(int((miny - lo[1]) / cell_m), int((maxy - lo[1]) / cell_m) + 1):
                if band.contains(Point(lo[0] + (ix + .5) * cell_m, lo[1] + (iy + .5) * cell_m)):
                    (blocked if o.cls in blockers else hazard).add((ix, iy))
    outside = {(ix, iy) for ix in range(nx) for iy in range(ny) if ix in (0, nx - 1) or iy in (0, ny - 1)}
    observed = None
    if scene.coverage is not None:   # ponytail: cell centres mapped by coordinate; resample properly if the two cell sizes ever differ
        cov = scene.coverage
        ce1, ce2 = np.asarray(cov.basis[0], float), np.asarray(cov.basis[1], float)
        observed = set()
        for ix, iy in cov.observed:
            p = (cov.origin_xy[0] + (ix + .5) * cov.cell_m) * ce1 + (cov.origin_xy[1] + (iy + .5) * cov.cell_m) * ce2
            gx, gy = int((p @ e1 - lo[0]) // cell_m), int((p @ e2 - lo[1]) // cell_m)
            if 0 <= gx < nx and 0 <= gy < ny:
                observed.add((gx, gy))
    cells = lambda s: [list(c) for c in sorted(s)]
    return Grid(cell_m=cell_m, origin_xy=[float(lo[0]), float(lo[1])], basis=[e1.tolist(), e2.tolist()], shape=[nx, ny], blocked=cells(blocked),
                hazard=cells(hazard - blocked), outside=cells(outside), observed=None if observed is None else cells(observed))


def build(scene: Scene, cfg: dict[str, Any], producer: str) -> Facts:
    cfg = {**DEFAULTS, **cfg}
    k, dsig, drel = cfg["k"], cfg["default_sigma_m"], cfg["default_scale_rel"]
    fixed, hazard, rel = set(cfg["fixed"]), set(cfg["hazard"]), scene.scale_rel_unc

    def u_mm(value_m: float, sigmas: list) -> tuple[int, list[str]]:
        terms = [dsig if s is None else s for s in sigmas]
        flags = sorted({"default_sigma" for s in sigmas if s is None} | ({"default_scale_unc"} if rel is None else set()))
        r = drel if rel is None else rel
        return mm(k * math.sqrt(sum(t * t for t in terms) + (r * abs(value_m)) ** 2)), flags

    def plan_sigma(o: Obj):
        return max((s for s in (o.sigma_m.get("L"), o.sigma_m.get("W")) if s is not None), default=None)

    facts: list[Fact] = []

    def add(pred, args, value, u, views, flags, unit="mm"):
        facts.append(Fact(pred=pred, args=list(args), value=value, unit=unit, u=u, views=list(views), flags=list(flags)))

    pts = {o.id: surface_points(o, cfg["face_samples"]) for o in scene.objects}
    e1, e2 = plan_basis(scene.ground_normal)
    foot = {o.id: MultiPoint([(p @ e1, p @ e2) for p in corners(o)]).convex_hull for o in scene.objects}
    for o in scene.objects:
        ut, ft = u_mm(o.top_m, [o.sigma_m.get("bottom"), o.sigma_m.get("H")])
        ub, fb = u_mm(o.bottom_m, [o.sigma_m.get("bottom")])
        add("top_height", [o.id], mm(o.top_m), ut, o.views, ft)
        add("bottom_height", [o.id], mm(o.bottom_m), ub, o.views, fb)
        add("floor_gap", [o.id], max(mm(o.bottom_m), 0), ub, o.views, fb)
    for oa, ob in itertools.combinations(scene.objects, 2):
        if not (oa.cls in hazard or ob.cls in hazard or (oa.cls in fixed and ob.cls in fixed)):
            continue
        a, b = oa.id, ob.id
        views = sorted(set(oa.views) | set(ob.views))
        p, q, d = closest(pts[a], pts[b])
        d = 0.0 if inside(ob, pts[a]) or inside(oa, pts[b]) else d
        ud, fd = u_mm(d, [plan_sigma(oa), plan_sigma(ob)])
        add("min_distance_3d", [a, b], mm(d), ud, views, fd)
        hg = float(foot[a].distance(foot[b]))
        uh, fh = u_mm(hg, [plan_sigma(oa), plan_sigma(ob)])
        add("horizontal_gap", [a, b], mm(hg), uh, views, fh)
        zo = min(oa.top_m, ob.top_m) - max(oa.bottom_m, ob.bottom_m)
        uz, fz = u_mm(zo, [oa.sigma_m.get("H"), ob.sigma_m.get("H")])
        add("z_overlap", [a, b], mm(max(zo, 0.0)), uz, views, fz + (["separated_vertically"] if zo < 0 else []))
        for top, bot in ((oa, ob), (ob, oa)):
            if top.bottom_m >= bot.top_m - 0.02 and foot[top.id].intersects(foot[bot.id]):
                add("above", [top.id, bot.id], mm(top.bottom_m - bot.top_m), u_mm(0, [top.sigma_m.get("bottom"), bot.sigma_m.get("H")])[0], views, [])
        blocked = segment_blocked(p, q, scene, {a, b})
        add("line_of_sight", [a, b], 0 if blocked else 1, None, views, [f"blocked_by={x}" for x in blocked], unit="bool")
        if oa.cls in hazard and ob.cls in fixed or ob.cls in hazard and oa.cls in fixed:   # the ISO 13857 Table 2 inputs
            hz, st = (oa, ob) if oa.cls in hazard else (ob, oa)
            add("reach_over", [hz.id, st.id], mm(hg), uh, views, [f"a_mm={mm(hz.top_m)}", f"b_mm={mm(st.top_m)}", *fh])
    grid = plan_grid(scene, foot, e1, e2, fixed, hazard, cfg["cell_m"], cfg["margin_m"])
    return Facts(scene_id=scene.scene_id, signature_version=str(cfg["signature_version"]), facts=facts, grid=grid, producer=producer)


@register("L2", "relations", "1")
class Relations:
    """cfg (all optional, see DEFAULTS): k, default_sigma_m, default_scale_rel, cell_m, margin_m, face_samples, fixed, hazard, signature_version."""

    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        return {"facts": build(inputs["scene"], cfg, f"{self.name}@{self.version}")}


if __name__ == "__main__":   # self-check: two unit boxes 0.3 m apart; a fence ring around a robot; one coverage cell lands on the grid
    from ehs_spatial.verdict.contracts import Coverage

    def mk(i, cls, cx, cy, L, W, H):
        return Obj(id=i, cls=cls, label=i, center_m=[cx, cy, H / 2], axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size_m=[L, W, H],
                   bottom_m=0.0, top_m=H, confidence="high", floor_contact=True)

    ring = [mk("n", "fence", 0, 2, 4.2, 0.1, 2), mk("s", "fence", 0, -2, 4.2, 0.1, 2), mk("e", "fence", 2, 0, 0.1, 4.2, 2), mk("w", "fence", -2, 0, 0.1, 4.2, 2)]
    cov = Coverage(cell_m=0.1, origin_xy=[-3, -3], basis=[[1, 0, 0], [0, 1, 0]], shape=[60, 60], observed=[[30, 30]])
    f = build(Scene(scene_id="t", objects=[mk("r", "robot", 0, 0, 1, 1, 1), mk("b", "fence", 1.3, 0, 1, 1, 1), *ring], ground_normal=[0, 0, 1], coverage=cov), {}, "self-check")
    by = {(x.pred, tuple(x.args)): x for x in f.facts}
    assert abs(by[("min_distance_3d", ("r", "b"))].value - 300) <= 20, by[("min_distance_3d", ("r", "b"))]
    g = f.grid
    assert g.hazard and g.blocked and not (set(map(tuple, g.hazard)) & set(map(tuple, g.outside)))
    (gx, gy), = g.observed
    world = (g.origin_xy[0] + (gx + .5) * g.cell_m) * np.asarray(g.basis[0]) + (g.origin_xy[1] + (gy + .5) * g.cell_m) * np.asarray(g.basis[1])
    assert np.linalg.norm(world - [0.05, 0.05, 0]) < 0.1, world
    print("relations self-check ok:", len(f.facts), "facts, grid", g.shape, "blocked", len(g.blocked), "hazard", len(g.hazard))
