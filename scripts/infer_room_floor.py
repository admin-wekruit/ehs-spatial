"""The floor of a room as a model: the verified floor plane extended over the room's footprint. No model call.

Observed layers only hold what a camera saw, and a handheld camera looks at desks, not at the floor. The model layer
already holds estimates (a generated object's unseen sides); a floor is the easiest of them: one plane, already verified
as one plane across views (mono_room metric), bounded by the room. The footprint is the convex hull of the observed
geometry on the floor plan: floor is assumed between things that were seen, not beyond them (a bounding rectangle
reached well outside the room). The colour is the median of the floor that was seen. It is written as a model
with its basis and the share that was actually observed, never into the observed mesh.

  python scripts/infer_room_floor.py --fused RUN --output NEW_DIR
  python scripts/infer_room_floor.py --self-check
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BELOW, CELL, MIN_COMPONENT = .01, .05, 2000  # the model sits a hair under the plane so the observed floor stays visible; coverage cell; fragments smaller than this do not stretch the room


def footprint(plan_points):
    """Convex outline of the points on the floor plan, without the farthest 0.5% (a stray fragment must not stretch the room)."""
    from shapely.geometry import MultiPoint
    reach = np.linalg.norm(plan_points - np.median(plan_points, 0), axis=1)
    bulk = plan_points[reach <= np.percentile(reach, 99.5)]
    return np.array(MultiPoint(bulk[::max(1, len(bulk) // 50000)]).convex_hull.exterior.coords[:-1])


def build(args):
    import open3d as o3d
    import trimesh
    scale = json.loads((args.fused / "metric-scale.json").read_text())
    up, origin, tolerance, metres = np.array(scale["up_native"]), np.array(scale["plane_point_native"]), 1.5 * scale["plane_tolerance_native"], scale["metres_per_native_unit"]
    a = np.cross(up, [1., 0, 0]); a /= np.linalg.norm(a)
    b = np.cross(up, a)
    mesh = o3d.io.read_triangle_mesh(str(args.fused / "mono-anchored-mesh.ply"))
    vertices, faces, colors = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
    labels, sizes, _ = (np.asarray(x) for x in mesh.cluster_connected_triangles())
    solid = np.zeros(len(vertices), bool)
    solid[faces[sizes[labels] >= MIN_COMPONENT].ravel()] = True
    relative = vertices - origin
    plan = np.stack([relative @ a, relative @ b], 1)
    corners = footprint(plan[solid])
    seen = solid & (np.abs(relative @ up) < tolerance)
    colour = np.median(colors[seen], 0) if seen.any() else np.array([.5, .5, .5])
    world = origin + corners[:, :1] * a + corners[:, 1:] * b - up * BELOW
    fan = [[0, n, n + 1] for n in range(1, len(world) - 1)]  # convex, so a fan from one corner covers it
    floor = trimesh.Trimesh(world, fan + [f[::-1] for f in fan], vertex_colors=np.tile((colour * 255).astype(np.uint8), (len(world), 1)), process=False)  # both sides
    args.output.mkdir(parents=True, exist_ok=False)
    floor.export(args.output / "inferred-floor.glb")
    from shapely.geometry import Polygon
    area = Polygon(corners).area
    cells = len(np.unique(np.floor(plan[seen] / CELL).astype(int), axis=0))
    report = {"kind": "inferred_floor_model", "basis": "verified floor plane (consensus over views) extended over the convex hull of the observed room geometry on the floor plan",
              "observed": False, "observed_share_of_this_floor": round(min(1., cells * CELL * CELL / area), 3),
              "extent_m": [round(float(v) * metres, 2) for v in corners.max(0) - corners.min(0)], "area_m2": round(area * metres ** 2, 1), "metres_per_native_unit": metres,
              "corners_native": world.tolist(), "colour_rgb": (colour * 255).astype(int).tolist(), "colour_source": "median colour of the floor that was seen",
              "below_plane_native": BELOW, "fused_run": str(args.fused), "mesh_sha256": hashlib.sha256((args.output / "inferred-floor.glb").read_bytes()).hexdigest(), "newModelCalls": 0,
              "limits": ["convex: a concave room is over-covered inside its corners, and floor beyond the outermost seen structure is left out", "walls, doors and what stands on the unseen floor are not inferred", "metres rest on the stated 1.6 m carry height"]}
    (args.output / "inferred-floor.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in ("extent_m", "area_m2", "observed_share_of_this_floor", "colour_rgb")}))


def self_check():
    rng = np.random.default_rng(0)
    room = rng.uniform([0, 0], [4, 3], (20000, 2))
    turn = np.array([[np.cos(.4), -np.sin(.4)], [np.sin(.4), np.cos(.4)]])
    stray = np.array([[40., 40.]])  # one far fragment must not stretch the room
    from shapely.geometry import Polygon
    outline = Polygon(footprint(np.vstack([room, stray]) @ turn.T))
    assert 11.5 < outline.area < 12.05, outline.area
    print("inferred floor check passed: the room's outline is recovered and a stray fragment does not stretch it")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--fused", type=Path)
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else build(a)
