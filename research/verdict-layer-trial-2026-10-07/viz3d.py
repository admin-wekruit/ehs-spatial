"""3D picture of the scene graph: boxes as wireframes on the floor, floor-gap edges, 3D closest-point edges robot<->fixed with mm ± U,
reach-over triples (a / b / c) for robot vs fences, line-of-sight blocking. Two viewpoints side by side.
  python viz3d.py out/090/scene-graph.json out/090/scene-graph-3d.png
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLOR = {'fence': '#1f77b4', 'guard': '#17becf', 'bollard': '#9467bd', 'light_curtain': '#2ca02c', 'robot': '#d62728', 'cart': '#8c564b', 'estop': '#ff7f0e', 'other': '#7f7f7f'}
EDGES_OF_BOX = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def box_corners(n, axes):
    c, a, h = np.asarray(n['center_m']), np.asarray(axes, float), np.asarray(n['size_m']) / 2
    return np.array([c + a[0] * sx * h[0] + a[1] * sy * h[1] + a[2] * sz * h[2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def draw(ax, graph, axes_by_id, elev, azim):
    nodes = {n['id']: n for n in graph['nodes'] if n['cls'] != 'floor'}
    floor_n = np.asarray(next(n['normal'] for n in graph['nodes'] if n['cls'] == 'floor'), float); floor_n /= np.linalg.norm(floor_n)
    occ = graph['plan_occupancy']; e1, e2 = np.asarray(occ['basis'][0]), np.asarray(occ['basis'][1]); lo = occ['origin']
    base_pt = min((np.asarray(e['q']) for e in graph['edges'] if e['type'] == 'floor_gap'), key=lambda p: p @ floor_n)
    offset = base_pt @ floor_n
    F = lambda p: np.array([np.asarray(p) @ e1, np.asarray(p) @ e2, np.asarray(p) @ floor_n - offset])   # floor-aligned frame: z = height above floor
    allc = np.array([F(c) for i, n in nodes.items() for c in box_corners(n, axes_by_id[i])])
    for i, n in nodes.items():
        cs = np.array([F(c) for c in box_corners(n, axes_by_id[i])]); col = COLOR.get(n['cls'], '#7f7f7f')
        for a, b in EDGES_OF_BOX:
            ax.plot(*zip(cs[a], cs[b]), color=col, linewidth=1.2 if n['cls'] != 'cart' else 0.6, alpha=0.95)
        ax.text(*cs.mean(0), f"{n['label']}\n{n['confidence']}", fontsize=6, color=col)
    xs = np.linspace(lo[0], lo[0] + occ['shape'][0] * occ['cell_m'], 2); ys = np.linspace(lo[1], lo[1] + occ['shape'][1] * occ['cell_m'], 2)
    X, Y = np.meshgrid(xs, ys); ax.plot_surface(X, Y, np.zeros_like(X), alpha=0.10, color='#999999')   # the floor, z = 0
    for e in graph['edges']:
        p, q = F(e['p']), F(e['q'])
        if e['type'] == 'floor_gap' and e['value_mm'] > 30:
            ax.plot(*zip(p, q), color='#ff7f0e', linewidth=1.5)
            ax.text(*((p + q) / 2), f"gap {e['value_mm']}±{e['U_mm']}", fontsize=5.5, color='#ff7f0e')
        if e['type'] == 'min_distance_3d' and (nodes[e['a']]['cls'] == 'robot' or nodes[e['b']]['cls'] == 'robot'):
            col = '#d62728' if e['value_mm'] < 500 else '#555555'
            ax.plot(*zip(p, q), color=col, linestyle='--', linewidth=0.9)
            ax.text(*((p + q) / 2), f"{e['value_mm']}±{e['U_mm']}", fontsize=5.5, color=col)
        if e['type'] == 'line_of_sight' and e['blocked_by']:
            ax.scatter(*((p + q) / 2), marker='x', color='#000000', s=12)
        if e['type'] == 'reach_over' and nodes[e['b']]['cls'] == 'fence':
            ax.text(*q, f"a={e['a_mm']} b={e['b_mm']} c={e['c_mm']}", fontsize=5.5, color='#1f77b4')
    lo3, hi3 = allc.min(0), allc.max(0); span = (hi3 - lo3).max() / 2; mid = (lo3 + hi3) / 2
    ax.set_xlim(mid[0] - span, mid[0] + span); ax.set_ylim(mid[1] - span, mid[1] + span); ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.view_init(elev=elev, azim=azim); ax.set_xlabel('e1 (m)'); ax.set_ylabel('e2 (m)'); ax.set_zlabel('height above floor (m)')


def main(graph_path, scene_path, out_png):
    graph = json.loads(open(graph_path).read()); scene = json.loads(open(scene_path).read())
    axes_by_id = {o['id']: o['axes'] for o in scene['objects']}
    fig = plt.figure(figsize=(16, 7))
    for k, (elev, azim) in enumerate(((28, -55), (18, 35))):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d'); draw(ax, graph, axes_by_id, elev, azim)
        ax.set_title(f"view {k + 1}: elev {elev}°, azim {azim}°", fontsize=8)
    kinds = {}
    for e in graph['edges']:
        kinds[e['type']] = kinds.get(e['type'], 0) + 1
    fig.suptitle(f"{graph['scene_id']} scene graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges {kinds}\n"
                 "orange = floor gap (> 30 mm), dashed = robot<->fixed closest 3D distance mm ± U (red < 500), x = line of sight blocked, a/b/c = ISO 13857 Table 2 inputs", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); print(out_png, kinds)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
