"""Plan-view picture of the thin scene layer: object footprints, occupancy grid (blocked / hazard / reachable from outside / openings),
and the robot<->fixed distance edges the rules consumed. Reads the Scene contract only.  python viz.py out/scene-090.json out/090/scene-graph.png
"""
from collections import deque
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

import relations
from engine import BLOCKERS, HAZARDS, expanded_u
from scene import Scene

COLOR = {'fence': '#1f77b4', 'guard': '#17becf', 'bollard': '#9467bd', 'light_curtain': '#2ca02c', 'robot': '#d62728', 'cart': '#8c564b', 'estop': '#ff7f0e', 'other': '#7f7f7f'}


def reach_set(grid):
    nx, ny = grid['shape']; blocked = grid['blocked']
    seen, q = set(), deque(c for c in grid['outside'] if c not in blocked)
    seen.update(q)
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = (x + dx, y + dy)
            if 0 <= c[0] < nx and 0 <= c[1] < ny and c not in blocked and c not in seen:
                seen.add(c); q.append(c)
    return seen


def main(scene_path, out_png):
    scene = Scene.load(scene_path)
    grid = relations.plan_grid(scene, BLOCKERS, HAZARDS)
    e1, e2 = grid['basis']; lo, cm = grid['origin'], grid['cell_m']
    reach = reach_set(grid)
    openings = {c for c in reach if any((c[0] + dx, c[1] + dy) in grid['hazard'] for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))}
    fig, ax = plt.subplots(figsize=(11, 8))
    for (x, y) in grid['cells']:
        col = None
        if (x, y) in grid['blocked']: col = '#333333'
        elif (x, y) in openings: col = '#ff7f0e'
        elif (x, y) in grid['hazard']: col = '#ffb3b3'
        elif (x, y) in reach: col = '#eeeeee'
        if col:
            ax.add_patch(Rectangle((lo[0] + x * cm, lo[1] + y * cm), cm, cm, facecolor=col, edgecolor='none', alpha=0.9 if col == '#333333' else 0.6))
    pts = {o.id: relations.surface_points(o) for o in scene.objects}
    for o in scene.objects:
        poly = np.array([(p @ e1, p @ e2) for p in relations.corners(o)])
        hull = relations.MultiPoint([tuple(p) for p in poly]).convex_hull
        ax.add_patch(Polygon(np.array(hull.exterior.coords), closed=True, fill=False, edgecolor=COLOR.get(o.cls, '#7f7f7f'), linewidth=2))
        cx, cy = hull.centroid.x, hull.centroid.y
        ax.text(cx, cy, f"{o.label}\n{o.cls} · {o.confidence}\nbottom {o.bottom_m * 100:+.0f} cm", ha='center', va='center', fontsize=7, color=COLOR.get(o.cls, '#7f7f7f'))
    for a, b in relations.all_pairs(scene, 'robot', BLOCKERS):
        pa, pb = pts[a], pts[b]
        d2 = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1); i, j = np.unravel_index(d2.argmin(), d2.shape)
        p, q = pa[i], pb[j]; d = float(np.sqrt(d2[i, j]))
        oa, ob = relations.scene_obj(scene, a), relations.scene_obj(scene, b)
        u, _ = expanded_u(d, [max((s for s in (oa.sigma_m.get('L'), oa.sigma_m.get('W')) if s is not None), default=None),
                              max((s for s in (ob.sigma_m.get('L'), ob.sigma_m.get('W')) if s is not None), default=None)], scene.scale_rel_unc)
        ax.plot([p @ e1, q @ e1], [p @ e2, q @ e2], color='#d62728' if d < 0.5 else '#999999', linewidth=1, linestyle='--')
        ax.text(((p + q) / 2) @ e1, ((p + q) / 2) @ e2, f"{d * 1000:.0f}±{u * 1000:.0f} mm", fontsize=6, color='#d62728' if d < 0.5 else '#555555')
    ax.set_aspect('equal'); ax.set_xlim(lo[0], lo[0] + grid['shape'][0] * cm); ax.set_ylim(lo[1], lo[1] + grid['shape'][1] * cm)
    ax.set_xlabel('e1 (m)'); ax.set_ylabel('e2 (m)')
    ax.set_title(f"{scene.scene_id}: thin scene layer, plan view  |  dark = blocked by guard-class footprint, pink = hazard (robot box), "
                 f"grey = reachable from outside, orange = openings into the hazard\nobserved-floor coverage: NOT PROVIDED by this layer (everything outside footprints is treated as unknown, not as empty)", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); print(out_png, 'blocked', len(grid['blocked']), 'hazard', len(grid['hazard']), 'reachable', len(reach), 'openings', len(openings))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
