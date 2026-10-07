"""Thin 3D scene graph: nodes (objects + floor) and typed, measured edges, built from the Scene contract only; persisted as JSON.

Edges (all with 3D endpoints p, q in world metres, value in mm, expanded uncertainty U in mm (k=2), and the photo ids behind them):
  min_distance_3d   closest surface points of two oriented boxes (3D)
  horizontal_gap    plan-view footprint gap (0 when footprints overlap) - ISO 13857 Table 2 "c"
  floor_gap         object bottom -> floor (vertical)
  z_overlap         vertical overlap of the two height ranges (mm; 0 = one is entirely above the other)
  above             a is entirely above b and their footprints overlap (stacking / overhang)
  reach_over        hazard top a, structure top b, horizontal gap c (the three inputs of ISO 13857 Table 2)
  line_of_sight     the segment between the two closest points crosses another box (blocked_by) or not
plus the plan-view occupancy grid (relations.plan_grid) as a layer.  python scene_graph.py out/scene-090.json out/090/scene-graph.json
"""
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import MultiPoint

import relations
from scene import Scene

K = 2
DEFAULT_SIGMA_M = 0.05
DEFAULT_SCALE_REL = 0.02
FIXED = {'fence', 'guard', 'bollard', 'light_curtain'}
HAZARD = {'robot'}


def mm(x):
    return int(round(x * 1000))


def u_mm(value_m, sigmas, rel):
    terms = [DEFAULT_SIGMA_M if s is None else s for s in sigmas]
    flags = sorted({'default_sigma' for s in sigmas if s is None} | ({'default_scale_unc'} if rel is None else set()))
    rel = DEFAULT_SCALE_REL if rel is None else rel
    return mm(K * math.sqrt(sum(t * t for t in terms) + (rel * abs(value_m)) ** 2)), flags


def plan_sigma(o):
    return max((s for s in (o.sigma_m.get('L'), o.sigma_m.get('W')) if s is not None), default=None)


def closest(pa, pb):
    d2 = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(d2.argmin(), d2.shape)
    return pa[i], pb[j], float(math.sqrt(d2[i, j]))


def segment_blocked(p, q, scene, exclude, n=40):
    pts = np.linspace(p, q, n)[1:-1]
    hits = []
    for o in scene.objects:
        if o.id in exclude:
            continue
        if np.any(np.all(np.abs(relations.corners_local(o, pts)) <= 1, axis=1)):
            hits.append(o.id)
    return hits


def build(scene):
    rel = scene.scale_rel_unc
    objs = {o.id: o for o in scene.objects}
    pts = {o.id: relations.surface_points(o) for o in scene.objects}
    e1, e2 = relations.plan_basis(scene.ground_normal)
    foot = {o.id: MultiPoint([(p @ e1, p @ e2) for p in relations.corners(o)]).convex_hull for o in scene.objects}
    nodes = [{'id': 'floor', 'cls': 'floor', 'label': 'floor', 'normal': scene.ground_normal}]
    nodes += [{'id': o.id, 'cls': o.cls, 'label': o.label, 'center_m': o.center_m, 'size_m': o.size_m, 'bottom_m': o.bottom_m, 'top_m': o.top_m,
               'confidence': o.confidence, 'sigma_m': o.sigma_m, 'views': o.views} for o in scene.objects]
    edges = []
    for o in scene.objects:   # vertical: object -> floor
        u, flags = u_mm(o.bottom_m, [o.sigma_m.get('bottom')], rel)
        base = np.asarray(o.center_m) - np.asarray(o.axes[2]) * o.size_m[2] / 2
        edges.append({'type': 'floor_gap', 'a': o.id, 'b': 'floor', 'value_mm': mm(o.bottom_m), 'U_mm': u, 'flags': flags,
                      'p': base.tolist(), 'q': (base - np.asarray(scene.ground_normal) * o.bottom_m).tolist(), 'views': o.views})
    pairs = [(a.id, b.id) for a, b in itertools.combinations(scene.objects, 2) if a.cls in HAZARD or b.cls in HAZARD or (a.cls in FIXED and b.cls in FIXED)]
    for a, b in pairs:
        oa, ob = objs[a], objs[b]
        views = sorted(set(oa.views) | set(ob.views))
        p, q, d = closest(pts[a], pts[b])
        inside = np.any(np.all(np.abs(relations.corners_local(ob, pts[a])) <= 1, axis=1)) or np.any(np.all(np.abs(relations.corners_local(oa, pts[b])) <= 1, axis=1))
        d = 0.0 if inside else d
        u, flags = u_mm(d, [plan_sigma(oa), plan_sigma(ob)], rel)
        edges.append({'type': 'min_distance_3d', 'a': a, 'b': b, 'value_mm': mm(d), 'U_mm': u, 'flags': flags, 'p': p.tolist(), 'q': q.tolist(), 'views': views})
        hg = float(foot[a].distance(foot[b]))
        uh, fh = u_mm(hg, [plan_sigma(oa), plan_sigma(ob)], rel)
        edges.append({'type': 'horizontal_gap', 'a': a, 'b': b, 'value_mm': mm(hg), 'U_mm': uh, 'flags': fh, 'p': p.tolist(), 'q': q.tolist(), 'views': views})
        zo = min(oa.top_m, ob.top_m) - max(oa.bottom_m, ob.bottom_m)
        uz, fz = u_mm(zo, [oa.sigma_m.get('H'), ob.sigma_m.get('H')], rel)
        edges.append({'type': 'z_overlap', 'a': a, 'b': b, 'value_mm': mm(max(zo, 0.0)), 'U_mm': uz, 'flags': fz, 'p': p.tolist(), 'q': q.tolist(), 'views': views,
                      'separated_vertically': zo < 0})
        for top, bot in ((oa, ob), (ob, oa)):
            if top.bottom_m >= bot.top_m - 0.02 and foot[top.id].intersects(foot[bot.id]):
                edges.append({'type': 'above', 'a': top.id, 'b': bot.id, 'value_mm': mm(top.bottom_m - bot.top_m), 'U_mm': u_mm(0, [top.sigma_m.get('bottom'), bot.sigma_m.get('H')], rel)[0],
                              'flags': [], 'p': top.center_m, 'q': bot.center_m, 'views': views})
        blocked = segment_blocked(p, q, scene, {a, b})
        edges.append({'type': 'line_of_sight', 'a': a, 'b': b, 'value_mm': mm(d), 'U_mm': u, 'flags': [], 'p': p.tolist(), 'q': q.tolist(), 'views': views,
                      'blocked_by': blocked})
        if oa.cls in HAZARD and ob.cls in FIXED or ob.cls in HAZARD and oa.cls in FIXED:   # ISO 13857 Table 2 inputs
            hz, st = (oa, ob) if oa.cls in HAZARD else (ob, oa)
            ua, _ = u_mm(hz.top_m, [hz.sigma_m.get('bottom'), hz.sigma_m.get('H')], rel)
            ub, _ = u_mm(st.top_m, [st.sigma_m.get('bottom'), st.sigma_m.get('H')], rel)
            edges.append({'type': 'reach_over', 'a': hz.id, 'b': st.id, 'a_mm': mm(hz.top_m), 'a_U_mm': ua, 'b_mm': mm(st.top_m), 'b_U_mm': ub,
                          'c_mm': mm(hg), 'c_U_mm': uh, 'value_mm': mm(hg), 'U_mm': uh, 'flags': fh, 'p': p.tolist(), 'q': q.tolist(), 'views': views})
    grid = relations.plan_grid(scene, FIXED, HAZARD)
    layer = {'cell_m': grid['cell_m'], 'origin': [float(x) for x in grid['origin']], 'basis': [grid['basis'][0].tolist(), grid['basis'][1].tolist()], 'shape': list(grid['shape']),
             'blocked': sorted(grid['blocked']), 'hazard': sorted(grid['hazard']), 'outside': sorted(grid['outside']), 'observed': None}
    return {'schema': 'panoptes.scene_graph/0-trial', 'scene_id': scene.scene_id, 'nodes': nodes, 'edges': edges, 'plan_occupancy': layer,
            'coverage': {'observed_floor': 'not provided by the reconstruction layer; cells outside footprints are unknown, not empty'}}


def dump(graph, path):
    Path(path).write_text(json.dumps(graph, indent=1, ensure_ascii=False))


def summary(graph):
    by = {}
    for e in graph['edges']:
        by[e['type']] = by.get(e['type'], 0) + 1
    return by


if __name__ == '__main__':
    scene = Scene.load(sys.argv[1])
    g = build(scene)
    dump(g, sys.argv[2])
    print(sys.argv[2], 'nodes', len(g['nodes']), 'edges', summary(g))
