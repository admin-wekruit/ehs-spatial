"""The floor of a room as a model: the verified floor plane extended over the room's footprint. No model call.

Observed layers only hold what a camera saw, and a handheld camera looks at desks, not at the floor. The model layer
already holds estimates (a generated object's unseen sides); a floor is the easiest of them: one plane, already verified
as one plane across views (mono_room metric), bounded by the room. The footprint is the convex hull of the observed
geometry on the floor plan: floor is assumed between things that were seen, not beyond them (a bounding rectangle
reached well outside the room). The colour is the median of the floor that was seen. It is written as a model
with its basis and the share that was actually observed, never into the observed mesh.

With --dense (a LingBot dense point map in the same frame) the floor is inferred only where the record supports it:
under things seen standing on it (points 1 cm to 2 m above the plane), in gaps of up to CLOSE_M between those and the
floor that was seen, and inside anything they enclose; never across open ground no camera saw (a convex outline of the
map reached behind walls and down unwalked aisles). The model is a grid over that region whose colour at each point is
the nearest floor the video saw (within REACH, else the median), and every cell where no floor was seen also gets an
inferred point, a little dimmed, for the point-cloud view: the floor is complete where it can be, and still told apart
from what was observed.

  python scripts/infer_room_floor.py --fused RUN --dense LINGBOT_MAP --droid-run RUN --output NEW_DIR
  python scripts/infer_room_floor.py --fused RUN --output NEW_DIR      # no dense map: withheld (--legacy-convex: the old outline)
  python scripts/infer_room_floor.py --self-check
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BELOW, CELL, MIN_COMPONENT = .01, .05, 2000
GRID_M, POINT_M, REACH_M, DIM = .03, .02, .5, .85
REGION_M, CLOSE_M, STANDING_M, MIN_PATCH_M2 = .05, .5, 2., 1.
# a wrong floor is worse than none: each inferred cell must survive every camera that could see it (none saw clearly past
# it), and the gap-closing distance is the largest whose hidden-block test puts floor where the video proves there is none
# in at most MAX_FALSE of cells; if none qualifies, no floor is inferred at all
CLOSINGS_M, MAX_FALSE, HIDE_SHARE, BLOCK_M, SEE_THROUGH_REL, SEE_THROUGH_M, REFUTING_VIEWS = (.25, .5, 1.), .05, .2, 1., .08, .05, 2
# the see-through test is trusted only if it is quiet on floor the video saw (flags <= MAX_NOISE of it; at 4% + 3 cm it
# flagged 8.9% of ME340's seen floor, depth noise at grazing angles) and loud on a deliberately wrong floor: the plane
# raised by CONTROL_M must be flagged in >= MIN_CONTROL of the seen floor's cells
MAX_NOISE, CONTROL_M, MIN_CONTROL = .01, .15, .5  # --dense: region cells, gap closing, things standing on the floor, smallest kept patch  # --dense: model grid and inferred-point spacing, colour reach, brightness of inferred points  # the model sits a hair under the plane so the observed floor stays visible; coverage cell; fragments smaller than this do not stretch the room


def footprint(plan_points):
    """Convex outline of the points on the floor plan, without the farthest 0.5% (a stray fragment must not stretch the room)."""
    from shapely.geometry import MultiPoint
    reach = np.linalg.norm(plan_points - np.median(plan_points, 0), axis=1)
    bulk = plan_points[reach <= np.percentile(reach, 99.5)]
    return np.array(MultiPoint(bulk[::max(1, len(bulk) // 50000)]).convex_hull.exterior.coords[:-1])


def floor_region(plan, height, tolerance, cell, close, standing, min_cells):
    """Where the floor can be inferred, on a `cell` grid of the plan: the floor seen, under things seen standing on it, gaps
    of up to `close` between them, and whatever they enclose. Patches under `min_cells` without any floor seen are dropped
    (stray points). Returns (region, floor seen, grid origin)."""
    from scipy import ndimage
    low = plan.min(0)
    index = ((plan - low) / cell).astype(int)
    shape = tuple(index.max(0)[::-1] + 1)
    floor, things = np.zeros(shape, bool), np.zeros(shape, bool)
    seen, standing_on = np.abs(height) < tolerance, (height >= tolerance) & (height < standing)
    floor[index[seen, 1], index[seen, 0]] = True
    things[index[standing_on, 1], index[standing_on, 0]] = True
    r = int(round(close / cell))
    disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    region = ndimage.binary_fill_holes(ndimage.binary_closing(np.pad(floor | things, r), structure=disk)[r:-r, r:-r])
    labels, count = ndimage.label(region)
    sizes, with_floor = ndimage.sum(region, labels, range(1, count + 1)), ndimage.maximum(floor, labels, range(1, count + 1))
    keep = np.r_[False, (sizes >= min_cells) | (with_floor > 0)]
    return keep[labels], floor, low


def inside(points, region, low, cell):
    """Which plan points fall in a region cell."""
    index = np.floor((points - low) / cell).astype(int)
    ok = (index >= 0).all(1) & (index[:, 0] < region.shape[1]) & (index[:, 1] < region.shape[0])
    out = np.zeros(len(points), bool)
    out[ok] = region[index[ok, 1], index[ok, 0]]
    return out


def camera_depths(droid_run, depth_run, skip):
    """(frame -> (reliable depth on the 640x480 raster, camera-to-world)), raster K [fx, fy, cx, cy]: the depth the map was built from."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_apps"))
    import mono_room
    mono_room.use_clip(droid_run)
    k = mono_room.prepare_image(np.zeros((480, 640, 3), np.uint8), mono_room.CALIBRATION, 2)[1]
    views = {}
    for row in mono_room.load(droid_run, None, depth_run):
        if row["source_index"] not in skip:
            views[row["source_index"]] = (np.where(mono_room.unreliable(row["mono"], None, None, .03), 0, row["scale"] * row["mono"]).astype(np.float32), row["c2w"])
    return views, k


def see_through(points, views, k, slack):
    """Per point: views whose reliable depth at its pixel lies clearly beyond it (the camera saw through where it would be)."""
    count = np.zeros(len(points), int)
    for depth, c2w in views.values():
        local = (points - c2w[:3, 3]) @ c2w[:3, :3]
        z = local[:, 2]
        front = z > 1e-6
        u = np.round(np.where(front, local[:, 0] / np.where(front, z, 1), 0) * k[0] + k[2]).astype(int)
        v = np.round(np.where(front, local[:, 1] / np.where(front, z, 1), 0) * k[1] + k[3]).astype(int)
        ok = front & (u >= 0) & (v >= 0) & (u < depth.shape[1]) & (v < depth.shape[0])
        observed = np.zeros(len(points))
        observed[ok] = depth[v[ok], u[ok]]
        count += ok & (observed > 0) & (observed > z * (1 + SEE_THROUGH_REL) + slack)
    return count


def cell_centres(cells, low, cell):
    rows, cols = np.nonzero(cells)
    return np.stack([cols, rows], 1) * cell + low + cell / 2


def hidden_block_test(plan, height, tolerance, cell, close, standing, min_cells, on_plane, views, k, slack, seed=0):
    """Hide the seen floor in HIDE_SHARE of the BLOCK_M blocks that have some, infer again, and score the hidden blocks: recall
    of the floor that was hidden, and the share of floor put where the video proves there is none (cameras saw through it)."""
    size = cell * BLOCK_M / REGION_M  # BLOCK_M in native units (cell is REGION_M)
    block = np.floor(plan / size).astype(int)
    floor = np.abs(height) < tolerance
    keys = np.unique(block[floor], axis=0)
    hidden_keys = keys[np.random.default_rng(seed).random(len(keys)) < HIDE_SHARE]
    hidden = np.isin(block[:, 0] * 100003 + block[:, 1], hidden_keys[:, 0] * 100003 + hidden_keys[:, 1])
    keep = ~(floor & hidden)
    region, _, low = floor_region(plan[keep], height[keep], tolerance, cell, close, standing, min_cells)
    truth = plan[floor & hidden]
    recall = float(inside(truth, region, low, cell).mean()) if len(truth) else None
    centres = cell_centres(region, low, cell)
    in_hidden = np.isin(np.floor(centres / size).astype(int) @ [100003, 1], hidden_keys @ [100003, 1])
    predicted = centres[in_hidden]
    false_floor = float((see_through(on_plane(predicted), views, k, slack) >= REFUTING_VIEWS).mean()) if len(predicted) else None
    return {"recall": recall, "falseFloor": false_floor, "hiddenBlocks": int(len(hidden_keys)), "predictedCells": int(len(predicted))}


def floor_grid(corners, spacing):
    """Grid points on the floor plan inside the (convex) outline."""
    from matplotlib.path import Path as Outline
    low, high = corners.min(0), corners.max(0)
    xs, ys = np.arange(low[0], high[0] + spacing, spacing), np.arange(low[1], high[1] + spacing, spacing)
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    return grid[Outline(corners).contains_points(grid, radius=spacing)]


def grid_faces(grid):
    """Triangles of a regular grid given as its surviving points (cells with all four corners present)."""
    spacing = np.min(np.diff(np.unique(grid[:, 0])))
    keys = np.round((grid - grid.min(0)) / spacing).astype(np.int64)
    index = {(int(i), int(j)): n for n, (i, j) in enumerate(keys)}
    faces = []
    for (i, j), n in index.items():
        right, up_, diagonal = index.get((i + 1, j)), index.get((i, j + 1)), index.get((i + 1, j + 1))
        if right is not None and up_ is not None and diagonal is not None:
            faces += [[n, right, diagonal], [n, diagonal, up_]]
    return np.array(faces, np.int64).reshape(-1, 3)


def floor_colours(plan_points, seen_plan, seen_colours, reach, fallback, k=24):
    """Median colour of the (up to k) nearest floor points seen within `reach` on the plan, else the fallback colour: one
    nearest point painted blotches (a stain or a stray bright point became a disc of floor)."""
    if not len(seen_plan):
        return np.tile(fallback, (len(plan_points), 1))
    from scipy.spatial import cKDTree
    k = min(k, len(seen_plan))
    distance, nearest = cKDTree(seen_plan).query(plan_points, k=k, distance_upper_bound=reach)
    distance, nearest = distance.reshape(len(plan_points), k), nearest.reshape(len(plan_points), k)
    found = np.isfinite(distance)
    colours = np.where(found[..., None], np.asarray(seen_colours, float)[np.minimum(nearest, len(seen_plan) - 1)], np.nan)
    out = np.tile(np.asarray(fallback, float), (len(plan_points), 1))
    any_found = found.any(1)
    out[any_found] = np.nanmedian(colours[any_found], 1)
    return out


NO_DENSE = ("no validated dense map (the LingBot map was missing or failed its display gate): nothing tests where the floor goes on "
            "unseen ground, and a wrong floor is worse than none")


def build(args):
    if not args.dense and not args.legacy_convex:  # the convex outline was never tested against the video: withheld, with its reason
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "inferred-floor.json").write_text(json.dumps({"kind": "inferred_floor_withheld", "reason": NO_DENSE}, indent=1))
        print(json.dumps({"withheld": True, "reason": NO_DENSE}))
        return
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
    dense = None
    if args.dense:  # the dense map: outline, the floor it saw, and a grid coloured from it
        cloud = next(iter(trimesh.load(args.dense / "dense-points.glb", process=False).geometry.values()))
        points, point_colors = np.asarray(cloud.vertices, float), np.asarray(cloud.colors)[:, :3] / 255
        relative_dense = points - origin
        plan_dense, height = np.stack([relative_dense @ a, relative_dense @ b], 1), relative_dense @ up
        floor_seen = np.abs(height) < tolerance
        colour = np.median(point_colors[floor_seen], 0) if floor_seen.any() else colour
        cell, slack, min_cells = REGION_M / metres, SEE_THROUGH_M / metres, MIN_PATCH_M2 / REGION_M ** 2
        views, k = camera_depths(args.droid_run, args.fused, {i for span in args.skip_frames for i in range(int(span.split("-")[0]), int(span.split("-")[1]) + 1)})
        on_plane = lambda xy: origin + xy[:, :1] * a + xy[:, 1:] * b
        _, seen_cells, seen_low = floor_region(plan_dense, height, tolerance, cell, CLOSINGS_M[0] / metres, STANDING_M / metres, min_cells)
        seen_centres = on_plane(cell_centres(seen_cells, seen_low, cell))  # floor the video saw: where the test must stay quiet
        noise = float((see_through(seen_centres, views, k, slack) >= REFUTING_VIEWS).mean())
        control = float((see_through(seen_centres + up * CONTROL_M / metres, views, k, slack) >= REFUTING_VIEWS).mean())
        trials = {c: hidden_block_test(plan_dense, height, tolerance, cell, c / metres, STANDING_M / metres, min_cells, on_plane, views, k, slack) for c in CLOSINGS_M}
        passing = [c for c, t in trials.items() if t["falseFloor"] is not None and t["falseFloor"] <= MAX_FALSE]
        validation = {"rule": f"gap closing = the largest of {list(CLOSINGS_M)} m whose hidden-block test (floor hidden in {HIDE_SHARE:.0%} of {BLOCK_M} m blocks) puts floor where cameras saw through it in <= {MAX_FALSE:.0%} of cells; then every inferred cell that >= {REFUTING_VIEWS} views saw through (depth beyond it by {SEE_THROUGH_REL:.0%} + {SEE_THROUGH_M} m) is removed",
                      "hiddenBlockTests": {str(c): t for c, t in trials.items()}, "views": len(views),
                      "testNoiseOnSeenFloor": round(noise, 4), "raisedPlaneControlFlagged": round(control, 4),
                      "testTrust": f"flags <= {MAX_NOISE:.0%} of the seen floor and >= {MIN_CONTROL:.0%} of it raised by {CONTROL_M} m"}
        if noise > MAX_NOISE or control < MIN_CONTROL:
            passing = []  # the test cannot tell a wrong floor from a right one here: infer nothing
        if not passing:
            (args.output / "inferred-floor.json").write_text(json.dumps({"kind": "inferred_floor_withheld", "validation": validation,
                "reason": "no gap-closing distance kept wrongly placed floor under the limit: a wrong floor is worse than none"}, indent=1))
            print(json.dumps({"withheld": True, **validation["hiddenBlockTests"]}))
            return
        close = max(passing)
        region, floor_cells, low = floor_region(plan_dense, height, tolerance, cell, close / metres, STANDING_M / metres, min_cells)
        candidates = region & ~floor_cells
        refuted = see_through(on_plane(cell_centres(candidates, low, cell)), views, k, slack) >= REFUTING_VIEWS
        rows, cols = np.nonzero(candidates)
        region[rows[refuted], cols[refuted]] = False
        validation.update(closeM=close, inferredCells=int(candidates.sum()), refutedCells=int(refuted.sum()), refutedShare=round(float(refuted.mean()), 4) if len(refuted) else 0.)
        corners = footprint(plan_dense[inside(plan_dense, region, low, cell)])  # the region's outline, for the record
        grid = floor_grid(corners, GRID_M / metres)
        grid = grid[inside(grid, region, low, cell)]
        tint = floor_colours(grid, plan_dense[floor_seen], point_colors[floor_seen], REACH_M / metres, colour)
        faces_grid = grid_faces(grid)
        world = origin + grid[:, :1] * a + grid[:, 1:] * b - up * BELOW
        floor = trimesh.Trimesh(world, faces_grid, vertex_colors=np.c_[(tint * 255).round(), np.full(len(tint), 255)].astype(np.uint8), process=False)
        cells = floor_grid(corners, POINT_M / metres)  # inferred points only in the region, where no floor was seen within a cell
        cells = cells[inside(cells, region, low, cell)]
        from scipy.spatial import cKDTree
        unseen = cKDTree(plan_dense[floor_seen]).query(cells, distance_upper_bound=POINT_M / metres)[0] == np.inf if floor_seen.any() else np.ones(len(cells), bool)
        cells = cells[unseen]
        cell_colours = floor_colours(cells, plan_dense[floor_seen], point_colors[floor_seen], REACH_M / metres, colour) * DIM
        on_plane = origin + cells[:, :1] * a + cells[:, 1:] * b
        trimesh.PointCloud(on_plane, colors=np.c_[(cell_colours * 255).round(), np.full(len(cells), 255)].astype(np.uint8)).export(args.output / "inferred-floor-points.glb")
        dense = {"points": int(len(cells)), "point_spacing_native": POINT_M / metres, "grid_native": GRID_M / metres,
                 "colour_rule": f"each grid vertex and inferred point takes the median colour of the 24 nearest floor points the video saw within {REACH_M} m, else the median floor colour; inferred points are dimmed to {DIM:.0%} brightness",
                 "region": f"floor seen, under things seen standing on it ({tolerance * metres:.2f}-{STANDING_M} m above the plane), gaps up to {CLOSE_M} m between them, and what they enclose, on a {REGION_M} m grid; never across open ground no camera saw",
                 "region_area_m2": round(float(region.sum()) * REGION_M ** 2, 1), "floor_seen_area_m2": round(float(floor_cells.sum()) * REGION_M ** 2, 1),
                 "floor_points_seen": int(floor_seen.sum()), "validation": validation}
        seen, plan = floor_seen, plan_dense
    floor.export(args.output / "inferred-floor.glb")
    from shapely.geometry import Polygon
    area = Polygon(corners).area if dense is None else dense["region_area_m2"] / metres ** 2
    cells = len(np.unique(np.floor(plan[seen] / CELL).astype(int), axis=0))
    basis = ("floor plane (consensus over views) extended under and between things seen standing on it, and over what they enclose; cells "
             "cameras saw through removed" if dense else "floor plane (consensus over views) extended over the convex hull of the observed room "
             "geometry on the floor plan (--legacy-convex: not tested against the video)")
    report = {**({"dense": dense} if dense else {}), "kind": "inferred_floor_model", "basis": basis,
              "observed": False, "observed_share_of_this_floor": round(min(1., cells * CELL * CELL / area), 3),
              "extent_m": [round(float(v) * metres, 2) for v in corners.max(0) - corners.min(0)], "area_m2": round(area * metres ** 2, 1), "metres_per_native_unit": metres,
              "corners_native": (origin + corners[:, :1] * a + corners[:, 1:] * b - up * BELOW).tolist(), "colour_rgb": (colour * 255).astype(int).tolist(), "colour_source": "median colour of the floor that was seen",
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
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1.]])
    grid = floor_grid(square, .1)
    assert len(grid) == 121 and len(grid_faces(grid)) == 200, (len(grid), len(grid_faces(grid)))
    # region: two seen floor patches with a table standing between them fill in; the open ground beyond them does not
    patch = lambda x0, x1, y0, y1: np.stack(np.meshgrid(np.arange(x0, x1, .02), np.arange(y0, y1, .02)), -1).reshape(-1, 2)
    floors, table, stray = np.vstack([patch(0, 1, 0, 1), patch(1.6, 2.6, 0, 1)]), patch(1.05, 1.55, .2, .8), np.array([[9., 9.]])
    plan_pts, heights = np.vstack([floors, table, stray]), np.r_[np.zeros(len(floors)), np.full(len(table), .7), 0.]
    region, _, low = floor_region(plan_pts, heights, .01, .05, .3, 2., 400)
    assert inside(np.array([[1.3, .5], [.5, .5]]), region, low, .05).all(), "under the table and on the seen floor"
    assert not inside(np.array([[5., 5.], [8.9, 8.9]]), region, low, .05).any(), "not across open ground, not around a stray point"
    # see-through: a camera looking down +z at a wall 3 away saw through a point 2 away (empty space) but not one 3.5 away
    wall = np.full((480, 640), 3., np.float32)
    counts = see_through(np.array([[0, 0, 2.], [0, 0, 3.5], [0, 0, 3.]]), {0: (wall, np.eye(4))}, [500., 500., 319.5, 239.5], .03)
    assert counts.tolist() == [1, 0, 0], counts
    tint = floor_colours(np.array([[0, 0], [5, 5.]]), np.array([[.1, 0]]), np.array([[1, 0, 0.]]), .5, np.array([.5, .5, .5]))
    assert np.allclose(tint, [[1, 0, 0], [.5, .5, .5]]), "nearest seen colour within reach, else the fallback"
    print("inferred floor check passed: the room's outline is recovered and a stray fragment does not stretch it; grid, faces and seen colours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--fused", type=Path)
    parser.add_argument("--dense", type=Path, help="LingBot dense map (lingbot_dense_map.py build) in the same frame: outline, grid colours, inferred points")
    parser.add_argument("--droid-run", type=Path, help="with --dense: the cameras whose depth checks every inferred cell")
    parser.add_argument("--skip-frames", nargs="*", default=[], metavar="A-B", help="frames never used as evidence (a cut-away shot)")
    parser.add_argument("--legacy-convex", action="store_true", help="without --dense: the old convex outline, never tested against the video "
                                                                    "(default: withheld, since a wrong floor is worse than none)")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else build(a)
