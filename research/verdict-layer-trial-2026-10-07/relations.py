"""Derived relations of a Scene (scene.py): pairwise box distances and a plan-view occupancy grid.

Reads the Scene only (metric boxes). This is the geometry the rules consume as facts; the rules never compute geometry.
"""
import itertools

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point

FACE_SAMPLES = 9     # per face edge: 9x9 points on each of the 6 faces = 486 points per box


def corners(o):
    c, a, h = np.asarray(o.center_m), np.asarray(o.axes, float), np.asarray(o.size_m) / 2
    return np.array([c + a[0] * sx * h[0] + a[1] * sy * h[1] + a[2] * sz * h[2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def surface_points(o, n=FACE_SAMPLES):
    c, a, h = np.asarray(o.center_m), np.asarray(o.axes, float), np.asarray(o.size_m) / 2
    t = np.linspace(-1, 1, n)
    pts = []
    for fixed in range(3):
        u, v = [k for k in range(3) if k != fixed]
        for side in (-1, 1):
            tu, tv = np.meshgrid(t, t)
            pts.append(c + a[fixed] * side * h[fixed] + np.outer(tu.ravel(), a[u] * h[u]) + np.outer(tv.ravel(), a[v] * h[v]))
    return np.concatenate(pts)


def pair_distances(scene, pairs):
    """{(a_id, b_id): metres} minimum surface distance between the two oriented boxes (0 when they touch or overlap)."""
    pts = {o.id: surface_points(o) for o in scene.objects}
    trees = {k: cKDTree(v) for k, v in pts.items()}
    out = {}
    for a, b in pairs:
        d = min(trees[b].query(pts[a])[0].min(), trees[a].query(pts[b])[0].min())
        if np.any(np.all(np.abs((corners_local(scene_obj(scene, b), pts[a]))) <= 1 + 1e-9, axis=1)):
            d = 0.0   # a sampled point of a lies inside b: overlap
        out[(a, b)] = float(d)
    return out


def scene_obj(scene, oid):
    return next(o for o in scene.objects if o.id == oid)


def corners_local(o, points):
    """Points expressed in o's box frame, divided by the half extents: |coord| <= 1 on every axis means inside."""
    c, a, h = np.asarray(o.center_m), np.asarray(o.axes, float), np.asarray(o.size_m) / 2
    return ((np.asarray(points) - c) @ a.T) / np.maximum(h, 1e-9)


def plan_basis(normal):
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    e1 = np.cross(n, [1, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1, 0]); e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def plan_grid(scene, blockers, hazards, cell_m=0.10, margin_m=1.0):
    """Plan-view occupancy: cells blocked by the footprint of any blocker-class object, hazard cells = footprint of hazard-class objects.
    Returns dict(cells, adj, blocked, hazard, outside, origin, cell_m, basis) with cells as (ix, iy)."""
    e1, e2 = plan_basis(scene.ground_normal)
    foot = {o.id: MultiPoint([(p @ e1, p @ e2) for p in corners(o)]).convex_hull for o in scene.objects}
    allpts = np.array([xy for g in foot.values() for xy in g.exterior.coords])
    lo, hi = allpts.min(0) - margin_m, allpts.max(0) + margin_m
    nx, ny = int(np.ceil((hi[0] - lo[0]) / cell_m)), int(np.ceil((hi[1] - lo[1]) / cell_m))
    blocked, hazard = set(), set()
    for o in scene.objects:
        if o.cls not in blockers and o.cls not in hazards:
            continue
        g = foot[o.id]
        minx, miny, maxx, maxy = g.bounds
        for ix in range(int((minx - lo[0]) / cell_m), int((maxx - lo[0]) / cell_m) + 1):
            for iy in range(int((miny - lo[1]) / cell_m), int((maxy - lo[1]) / cell_m) + 1):
                if g.buffer(cell_m * 0.5).contains(Point(lo[0] + (ix + .5) * cell_m, lo[1] + (iy + .5) * cell_m)):
                    (blocked if o.cls in blockers else hazard).add((ix, iy))
    cells = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    adj = [((ix, iy), (ix + dx, iy + dy)) for ix, iy in cells for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= ix + dx < nx and 0 <= iy + dy < ny]
    outside = {(ix, iy) for ix, iy in cells if ix in (0, nx - 1) or iy in (0, ny - 1)}
    return dict(cells=cells, adj=adj, blocked=blocked, hazard=hazard - blocked, outside=outside, origin=lo, cell_m=cell_m, basis=(e1, e2), shape=(nx, ny))


def all_pairs(scene, cls_a, cls_b):
    return [(a.id, b.id) for a in scene.objects for b in scene.objects if a.cls == cls_a and b.cls in cls_b and a.id != b.id]


if __name__ == '__main__':   # self-check: two unit boxes 0.3 m apart; a fence ring around a robot is closed, remove one side and it opens
    from scene import Obj, Scene
    mk = lambda i, cls, cx, cy, L, W, H: Obj(i, cls, i, [cx, cy, H / 2], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [L, W, H], 0.0, H, {}, 'high', True)
    s = Scene('t', [mk('a', 'robot', 0, 0, 1, 1, 1), mk('b', 'fence', 1.3, 0, 1, 1, 1)], [0, 0, 1], None)
    d = pair_distances(s, [('a', 'b')])[('a', 'b')]
    assert abs(d - 0.3) < 0.02, d
    ring = [mk('n', 'fence', 0, 2, 4.2, 0.1, 2), mk('s', 'fence', 0, -2, 4.2, 0.1, 2), mk('e', 'fence', 2, 0, 0.1, 4.2, 2), mk('w', 'fence', -2, 0, 0.1, 4.2, 2)]
    g = plan_grid(Scene('t', [mk('r', 'robot', 0, 0, 1, 1, 1), *ring], [0, 0, 1], None), {'fence'}, {'robot'})
    assert g['hazard'] and g['blocked'] and not (g['hazard'] & g['outside'])
    print('relations self-check ok: distance', round(d, 3), 'grid', g['shape'], 'blocked', len(g['blocked']), 'hazard', len(g['hazard']))
