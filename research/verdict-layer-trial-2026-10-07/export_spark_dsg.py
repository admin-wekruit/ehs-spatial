"""Export the thin scene graph (scene_graph.py JSON) into MIT-SPARK's Spark-DSG (the data structure Hydra / Clio / Khronos produce),
so the verdict layer can later consume a Hydra-built graph from video with no change in format.

Layers used: OBJECTS (our boxes as OBB nodes, typed measured edges between them), PLACES (free-space cells of the plan occupancy grid,
Hydra-style `distance` = clearance to the nearest blocker, 4-adjacency), ROOMS (one workcell node, parent of every place; objects
attached to their nearest place).  python export_spark_dsg.py out/090/scene-graph.json out/scene-090.json out/090/scene-graph.dsg.json
"""
import json
from pathlib import Path
import sys

import numpy as np
import spark_dsg as dsg

CLASS_ID = {'fence': 1, 'guard': 2, 'bollard': 3, 'light_curtain': 4, 'robot': 5, 'cart': 6, 'estop': 7, 'other': 8}
PLACE_CELL_M = 0.5


def set_metadata(attrs, data):
    """Spark-DSG node/edge metadata is a JSON object; the binding differs across versions, so try the known spellings."""
    data = json.loads(json.dumps(data))   # plain JSON types only
    for setter in (lambda: attrs.metadata.set(data), lambda: attrs.metadata.add(data), lambda: attrs.metadata.set(json.dumps(data))):
        try:
            setter(); return True
        except Exception:
            continue
    return False


def obb(size, center, axes):
    R = np.array(axes, float).T   # columns = box axes in world coordinates
    for args in ((dsg.BoundingBoxType.OBB, np.array(size, np.float32), np.array(center, np.float32), R.astype(np.float32)),
                 (np.array(size, np.float32), np.array(center, np.float32), R.astype(np.float32)),
                 (np.array(size, np.float32), np.array(center, np.float32))):
        try:
            return dsg.BoundingBox(*args)
        except TypeError:
            continue
    return dsg.BoundingBox(np.array(size, np.float32))


def export(graph, scene):
    axes = {o['id']: o['axes'] for o in scene['objects']}
    G = dsg.DynamicSceneGraph()
    sym = {}
    objects = [n for n in graph['nodes'] if n['cls'] != 'floor']
    for i, n in enumerate(objects):
        a = dsg.ObjectNodeAttributes()
        a.name = n['label']; a.semantic_label = CLASS_ID.get(n['cls'], 8); a.position = np.array(n['center_m'], float)
        a.bounding_box = obb(n['size_m'], n['center_m'], axes[n['id']])
        set_metadata(a, {'panoptes_id': n['id'], 'cls': n['cls'], 'confidence': n['confidence'], 'bottom_m': n['bottom_m'], 'top_m': n['top_m'],
                         'sigma_m': n['sigma_m'], 'views': n['views']})
        s = dsg.NodeSymbol('O', i); sym[n['id']] = s
        assert G.add_node(dsg.DsgLayers.OBJECTS, s, a)
    # object-object edges: one Spark-DSG edge per pair, weight = 3D closest distance (m); every other measure of the pair in metadata
    pairs = {}
    for e in graph['edges']:
        if e['type'] == 'floor_gap' or e['a'] not in sym or e['b'] not in sym:
            continue
        key = tuple(sorted((e['a'], e['b'])))   # `above` / `reach_over` edges are directed; the pair is the same
        pairs.setdefault(key, {})[e['type']] = {k: v for k, v in e.items() if k not in ('type', 'p', 'q')}
    for (a, b), measures in pairs.items():
        ea = dsg.EdgeAttributes(); ea.weighted = True; ea.weight = measures['min_distance_3d']['value_mm'] / 1000.0
        set_metadata(ea, measures)
        assert G.insert_edge(sym[a], sym[b], ea)
    # places: free cells of the occupancy layer at PLACE_CELL_M, Hydra-style clearance distance; adjacency edges
    occ = graph['plan_occupancy']; e1, e2 = np.array(occ['basis'][0]), np.array(occ['basis'][1]); lo = np.array(occ['origin']); cm = occ['cell_m']
    nx, ny = occ['shape']; blocked = {tuple(c) for c in occ['blocked']}; hazard = {tuple(c) for c in occ['hazard']}
    step = max(1, int(round(PLACE_CELL_M / cm)))
    bl = np.array(sorted(blocked)) if blocked else np.zeros((0, 2))
    floor_n = np.array(next(n['normal'] for n in graph['nodes'] if n['cls'] == 'floor'), float); floor_n /= np.linalg.norm(floor_n)
    floor_pt = min((np.array(e['q']) for e in graph['edges'] if e['type'] == 'floor_gap'), key=lambda p: p @ floor_n)
    place = {}
    for x in range(0, nx, step):
        for y in range(0, ny, step):
            if (x, y) in blocked:
                continue
            d = float(np.sqrt(((bl - [x, y]) ** 2).sum(1).min()) * cm) if len(bl) else float('inf')
            p = dsg.PlaceNodeAttributes(); p.distance = d
            xy = lo + np.array([(x + .5) * cm, (y + .5) * cm])
            world = floor_pt + e1 * (xy[0] - (floor_pt @ e1)) + e2 * (xy[1] - (floor_pt @ e2))
            p.position = world + floor_n * 1.0   # 1 m above the floor, as Hydra places sit in free space
            set_metadata(p, {'cell': [x, y], 'hazard': (x, y) in hazard, 'observed': 'unknown'})
            s = dsg.NodeSymbol('p', len(place)); place[(x, y)] = s
            assert G.add_node(dsg.DsgLayers.PLACES, s, p)
    for (x, y), s in place.items():
        for dx, dy in ((step, 0), (0, step)):
            if (x + dx, y + dy) in place:
                G.insert_edge(s, place[(x + dx, y + dy)])
    # room: one workcell node, parent of every place; each object attached to its nearest place
    r = dsg.RoomNodeAttributes(); r.name = 'workcell'; r.semantic_label = 1
    r.position = np.mean([G.get_node(s).attributes.position for s in place.values()], axis=0) if place else np.zeros(3)
    rs = dsg.NodeSymbol('R', 0); assert G.add_node(dsg.DsgLayers.ROOMS, rs, r)
    for s in place.values():
        G.insert_edge(rs, s)
    positions = {s: G.get_node(s).attributes.position for s in place.values()}
    for n in objects:
        c = np.array(n['center_m']); nearest = min(positions, key=lambda s: np.linalg.norm(positions[s] - c))
        G.insert_edge(nearest, sym[n['id']])
    return G


def main(graph_path, scene_path, out_path):
    graph, scene = json.loads(Path(graph_path).read_text()), json.loads(Path(scene_path).read_text())
    G = export(graph, scene)
    G.save(str(out_path), include_mesh=False)
    H = dsg.DynamicSceneGraph.load(str(out_path))
    per_layer = {name: H.get_layer(getattr(dsg.DsgLayers, name)).num_nodes() for name in ('OBJECTS', 'PLACES', 'ROOMS')}
    print(f"{out_path}: {H.num_nodes()} nodes {H.num_edges()} edges, per layer {per_layer}, {Path(out_path).stat().st_size // 1024} KB")
    o = H.get_node(dsg.NodeSymbol('O', 0)).attributes
    print('  object 0 back:', o.name, 'label', o.semantic_label, 'bbox dims', np.round(o.bounding_box.dimensions, 3), 'type', o.bounding_box.type, 'metadata', (str(o.metadata)[:160] if hasattr(o, 'metadata') else '-'))


if __name__ == '__main__':
    main(*sys.argv[1:4])
