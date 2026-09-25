"""Fill the holes of a video report's room surface from each view's own depth.

The fused room surface keeps only depth that several views agree on, minus depth edges, so much of what the camera saw
is missing: the three MVP videos covered 55-85% of each keyframe's own view. Every missing pixel still has that view's
depth estimate. This walks the keyframes, finds the pixels the surface (with the patches added so far) does not cover,
or where it lies clearly behind this view's depth, and adds a triangle grid over them from that depth, coloured from the
frame. No triangle spans a depth jump, and pixels on moving entities stay empty: the moving layer draws them. The
patches are single-view depth, less certain than the fused surface; they are one separate geometry of the output GLB
(single-view-depth-fill), and fill.json records how much was added and the coverage before and after.

  python scripts/fill_scene_holes.py --droid-run D --depth-run F --shell textured-scene.glb [--dynamic-masks M] --output NEW_DIR
  python scripts/fill_scene_holes.py --self-check
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "modal_apps")]
STEP = 4          # grid spacing on the 640x480 raster: one vertex per 4x4 pixels
JUMP = .08        # a grid cell whose corner depths differ by more than this share is a depth edge, left open
BEHIND = 1.3      # the surface counts as missing where the first hit lies this far beyond the view's own depth


def grid(height=480, width=640, step=STEP):
    """Raster coordinates (u, v) of the grid samples, pixel centres of the 640x480 raster."""
    v, u = np.mgrid[step // 2:height:step, step // 2:width:step]
    return u.astype(np.float64), v.astype(np.float64)


def rays(u, v, K, c2w):
    """Unit world directions through raster pixels and the length of the ray to depth 1."""
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    stretch = np.linalg.norm(d, axis=-1)
    return (d / stretch[..., None]) @ c2w[:3, :3].T, stretch


def patch(depth, fill, u, v, K, c2w, colour, jump=JUMP):
    """Triangles over grid cells that touch a fill sample and whose four corners have smooth depth (0 = none).

    A cell counts if any corner is to be filled, so a patch overlaps its covered border by one cell and meets the
    surface without a crack. Returns world vertices, faces and RGB colours of the used samples.
    """
    h, w = depth.shape
    corners = [(slice(0, -1), slice(0, -1)), (slice(0, -1), slice(1, None)), (slice(1, None), slice(0, -1)), (slice(1, None), slice(1, None))]
    z = [depth[c] for c in corners]
    lo, hi = np.minimum.reduce(z), np.maximum.reduce(z)
    keep = (lo > 0) & (hi <= lo * (1 + jump)) & np.logical_or.reduce([fill[c] for c in corners])
    i, j = np.nonzero(keep)
    tl, tr, bl, br = i * w + j, i * w + j + 1, (i + 1) * w + j, (i + 1) * w + j + 1
    faces = np.concatenate([np.stack([tl, bl, tr], 1), np.stack([tr, bl, br], 1)])
    if not len(faces):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 3), np.uint8)
    used, inverse = np.unique(faces, return_inverse=True)
    uu, vv, zz = u.ravel()[used], v.ravel()[used], depth.ravel()[used]
    local = np.stack([(uu - K[0, 2]) / K[0, 0] * zz, (vv - K[1, 2]) / K[1, 1] * zz, zz], 1)
    return local @ c2w[:3, :3].T + c2w[:3, 3], inverse.reshape(-1, 3), colour.reshape(-1, 3)[used]


def ray_scene(vertices, faces):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)), o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return scene


class Surface:
    """The room surface, built into a ray scene once, and the patches so far, rebuilt after each view that adds some,
    so the next view already sees what the last one filled."""

    def __init__(self, vertices, faces):
        self.base, self.vertices, self.faces, self.patches = ray_scene(vertices, faces), [], [], None

    def add(self, vertices, faces):
        if len(faces):
            self.faces.append(np.asarray(faces) + sum(map(len, self.vertices)))
            self.vertices.append(np.asarray(vertices))
            self.patches = ray_scene(np.concatenate(self.vertices), np.concatenate(self.faces))

    def hits(self, origin, directions):
        """Distance to the first hit along each ray (inf where none)."""
        import open3d as o3d
        flat = directions.reshape(-1, 3)
        cast = o3d.core.Tensor(np.hstack([np.broadcast_to(origin, flat.shape), flat]).astype(np.float32))
        t = self.base.cast_rays(cast)["t_hit"].numpy()
        if self.patches is not None:
            t = np.minimum(t, self.patches.cast_rays(cast)["t_hit"].numpy())
        return t.reshape(directions.shape[:-1])


def coverage(surface, views, u, v):
    """Share of each view's grid samples whose ray meets the surface."""
    return np.array([np.isfinite(surface.hits(c2w[:3, 3], rays(u, v, K, c2w)[0])).mean() for _, K, c2w in views])


def run(args):
    import cv2
    import trimesh
    import mono_room
    mono_room.use_clip(args.droid_run)
    manifest = json.loads((args.droid_run / "input-manifest.json").read_text())
    rows = sorted(mono_room.load(args.droid_run, None, args.depth_run), key=lambda r: r["source_index"])
    for r in rows:
        r["conf"] = None  # not used; hundreds of views have to fit in memory
    raster_k = np.asarray(mono_room.prepare_image(np.zeros((480, 640, 3), np.uint8), mono_room.CALIBRATION, 2)[1], np.float64)
    K_of = lambda r: (lambda k: np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.]]))(r["k"] * 2 if mono_room.METRIC_CAMERAS else raster_k)
    shell = trimesh.load(args.shell, process=False)
    pieces = [g for g in shell.geometry.values()]
    offsets = np.cumsum([0] + [len(g.vertices) for g in pieces[:-1]])
    surface = Surface(np.concatenate([g.vertices for g in pieces]), np.concatenate([g.faces + o for g, o in zip(pieces, offsets)]))
    u, v = grid()
    judged = [(r["source_index"], K_of(r), r["c2w"]) for r in rows[::5]]
    before = coverage(surface, judged, u, v)

    vertices, faces, colours, count, filled = [], [], [], 0, 0
    for r in rows[::args.every]:
        index, K, c2w = r["source_index"], K_of(r), r["c2w"]
        depth = np.where(mono_room.unreliable(r["mono"], None, None, .03), 0, r["scale"] * r["mono"])[STEP // 2::STEP, STEP // 2::STEP]
        if args.dynamic_masks:
            depth = np.where(mono_room.moving_mask(args.dynamic_masks, index)[STEP // 2::STEP, STEP // 2::STEP], 0, depth)
        directions, stretch = rays(u, v, K, c2w)
        hit = surface.hits(c2w[:3, 3], directions)
        missing = (depth > 0) & ~(hit <= depth * stretch * BEHIND)
        if not missing.any():
            continue
        bgr = mono_room.prepare_image(cv2.imread(str(mono_room.DATASET / manifest["frames"][index]["relative_path"])), mono_room.CALIBRATION, 2)[0]
        world, tri, rgb = patch(depth, missing, u, v, K, c2w, bgr[STEP // 2::STEP, STEP // 2::STEP][..., ::-1])
        vertices.append(world), faces.append(tri + count), colours.append(rgb)
        surface.add(world, tri)
        count, filled = count + len(world), filled + 1
    after = coverage(surface, judged, u, v)

    args.output.mkdir(parents=True, exist_ok=False)
    added = trimesh.Trimesh(np.concatenate(vertices), np.concatenate(faces), vertex_colors=np.concatenate(colours), process=False) if vertices else None
    if added is not None:
        shell.add_geometry(added, geom_name="single-view-depth-fill")
    shell.export(args.output / "textured-scene.glb")
    stats = lambda s: {"mean": round(float(s.mean()), 4), "median": round(float(np.median(s)), 4), "worst": round(float(s.min()), 4)}
    report = {"views": len(rows), "views_used": len(rows[::args.every]), "views_that_added": filled, "every": args.every,
              "triangles_fused": int(sum(len(g.faces) for g in pieces)), "triangles_added": 0 if added is None else len(added.faces),
              "coverage_of_keyframe_views_before": stats(before), "coverage_of_keyframe_views_after": stats(after),
              "judged_views": len(judged), "grid_step_px": STEP, "depth_jump_left_open": JUMP, "behind_factor": BEHIND,
              "moving_pixels_left_empty": bool(args.dynamic_masks),
              "note": "added triangles are one view's own depth, not confirmed by other views; geometry single-view-depth-fill"}
    (args.output / "fill.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def self_check():
    """A wall with a hole and a step: only the hole is filled, the step stays open, moving pixels stay empty."""
    u, v = grid(40, 40, 4)
    K, c2w = np.array([[40., 0, 20], [0, 40, 20], [0, 0, 1]]), np.eye(4)
    depth = np.full(u.shape, 2.)
    depth[:, 6:] = 3.  # a depth step between columns 5 and 6
    hole = np.zeros(u.shape, bool)
    hole[2:4, 1:3] = True
    colour = np.zeros(u.shape + (3,), np.uint8)
    world, faces, rgb = patch(depth, hole, u, v, K, c2w, colour)
    assert len(faces) == 2 * 9 and len(world) == 16 == len(rgb), (len(faces), len(world))  # the 3x3 cells touching the 2x2 hole
    assert np.allclose(world[:, 2], 2.), "hole samples come back at their own depth"
    step_hole = np.zeros(u.shape, bool)
    step_hole[4, 5] = True
    world, faces, _ = patch(depth, step_hole, u, v, K, c2w, colour)
    assert len(faces) == 4 and set(np.round(world[:, 2], 3)) <= {2., 3.}, "cells across the step are left open"
    assert all(len(set(np.round(world[f, 2], 3))) == 1 for f in faces), "no triangle spans the step"
    moving = depth.copy()
    moving[2:4, 1:3] = 0
    assert len(patch(moving, hole, u, v, K, c2w, colour)[1]) == 0, "moving (depth 0) samples make no triangles"
    surface = Surface(np.array([[-9., -9, 2.5], [9, -9, 2.5], [0, 9, 2.5]]), np.array([[0, 1, 2]]))
    directions, stretch = rays(u, v, K, c2w)
    hit = surface.hits(c2w[:3, 3], directions)
    assert np.isfinite(hit).any() and np.allclose(hit[np.isfinite(hit)] / stretch[np.isfinite(hit)], 2.5, atol=1e-3), "ray distance matches the plane"
    print("fill check passed: holes filled at their own depth, steps and moving pixels left open, rays meet the surface")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--droid-run", type=Path)
    parser.add_argument("--depth-run", type=Path, help="the fused depth run whose mono/*.npz built the room surface")
    parser.add_argument("--shell", type=Path, help="texture_fused_mesh.py textured-scene.glb of that run")
    parser.add_argument("--dynamic-masks", type=Path, help="assemble_dynamic_masks.py masks: pixels of moving entities stay empty")
    parser.add_argument("--every", type=int, default=2, help="fill from every n-th view")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else run(a)
