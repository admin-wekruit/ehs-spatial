"""Fill the holes of a video report's room surface from each view's own depth.

The fused room surface keeps only depth that several views agree on, minus depth edges, so much of what the camera saw
is missing: the three MVP videos covered 55-85% of each keyframe's own view. Every missing pixel still has that view's
depth estimate. This walks the keyframes, finds the pixels the surface (with the patches added so far) does not cover,
or where it lies clearly behind this view's depth, and adds a triangle grid over them from that depth, coloured from the
frame. No triangle spans a depth jump, and pixels on moving entities stay empty: the moving layer draws them. The
patches are single-view depth, less certain than the fused surface; they are separate geometries of the output GLB
(single-view-depth-fill*), and fill.json records how much was added and the coverage before and after.

Without --video a patch carries one vertex colour per grid sample of the 640x480 raster (--step 4: a 160x120 colour
image). With --video (the clip's uncropped source-full.mp4) each patch is textured from its own full-resolution frame
through UVs, packed into atlas pages with the same exposure gain texture_fused_mesh.py found for that view; moving
entities (grown) and --overlay-rows (burned-in subtitles) make no triangles. With --carve a patch point that other views
see through (their own depth lies clearly beyond it) is dropped: the skirts single-view depth draws across depth edges
look right from their own view only and cover the true surface from every other one. Views whose depth disagrees with
the room surface (cameras that drifted from the map) judge no patch, and their own patches are judged by all views.

  python scripts/fill_scene_holes.py --droid-run D --depth-run F --shell textured-scene.glb [--dynamic-masks M] --output NEW_DIR
  python scripts/fill_scene_holes.py ... --video CLIP/source-full.mp4 [--overlay-rows 648:704] [--step 2] [--carve] --output NEW_DIR
  python scripts/fill_scene_holes.py --self-check
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "modal_apps"), str(Path(__file__).resolve().parent)]
STEP = 4          # grid spacing on the 640x480 raster: one vertex per 4x4 pixels
JUMP = .08        # a 4-px grid cell whose corner depths differ by more than this share is a depth edge, left open (scaled with the step: same slope)
BEHIND = 1.3      # the surface counts as missing where the first hit lies this far beyond the view's own depth
GROW = 5          # with --video: raster pixels a moving-entity mask is grown by, so no textured cell reaches onto the entity
CARVE = .08       # with --carve: a view whose depth lies this share beyond a patch point sees through it (fuse --carve: .04, stricter, more holes)
ON_SHELL = .6     # with --carve: share of a view's depth the room surface must agree with; ME340: drifted views .40-.45, the rest .82-.96


def grid(height=480, width=640, step=STEP):
    """Raster coordinates (u, v) of the grid samples, pixel centres of the 640x480 raster."""
    v, u = np.mgrid[step // 2:height:step, step // 2:width:step]
    return u.astype(np.float64), v.astype(np.float64)


def rays(u, v, K, c2w):
    """Unit world directions through raster pixels and the length of the ray to depth 1."""
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    stretch = np.linalg.norm(d, axis=-1)
    return np.einsum("...j,kj->...k", d / stretch[..., None], c2w[:3, :3]), stretch  # einsum: Accelerate's dgemm segfaults on big (n, 3) @ (3, 3)


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
    return np.einsum("nj,kj->nk", local, c2w[:3, :3]) + c2w[:3, 3], inverse.reshape(-1, 3), colour.reshape(-1, 3)[used]


def under_overlay(v, K, video_k, rows):
    """Raster samples whose ray lands in rows [y0, y1) of the uncropped video frame."""
    y = (v - K[1, 2]) / K[1, 1] * video_k[1] + video_k[3]
    return (y >= rows[0]) & (y < rows[1])


def free_space(points, evidence, relative=CARVE):
    """How many views see each point at their own depth, and how many see through it (their depth lies beyond it)."""
    from filter_video_static_surfaces import depth_evidence
    positive, negative = np.zeros(len(points), int), np.zeros(len(points), int)
    for depth, k, c2w in evidence:
        pos, neg = depth_evidence(points, depth, depth <= 0, k, c2w, tolerance=0, depth_range=(0, np.inf), relative_tolerance=relative)
        positive[pos] += 1
        negative[neg] += 1
    return positive, negative


def shell_agreement(surface, depth, k, c2w, relative=CARVE, stride=8):
    """Share of a view's own depth samples the room surface agrees with, where both exist (1 if none)."""
    v, u = np.mgrid[stride // 2:depth.shape[0]:stride, stride // 2:depth.shape[1]:stride].astype(float)
    z = depth[v.astype(int), u.astype(int)]
    directions, stretch = rays(u, v, k, c2w)
    t = surface.hits(c2w[:3, 3], directions)
    both = (z > 0) & np.isfinite(t)
    return float((np.abs(t[both] / (z[both] * stretch[both]) - 1) < relative).mean()) if both.any() else 1.


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
    from texture_fused_mesh import decode, full_k, keep_bytes, photo_atlas, project
    mono_room.use_clip(args.droid_run)
    manifest = json.loads((args.droid_run / "input-manifest.json").read_text())
    rows = sorted(mono_room.load(args.droid_run, None, args.depth_run), key=lambda r: r["source_index"])
    for r in rows:
        r["conf"] = None  # not used; hundreds of views have to fit in memory
    raster_k = np.asarray(mono_room.prepare_image(np.zeros((480, 640, 3), np.uint8), mono_room.CALIBRATION, 2)[1], np.float64)
    K_of = lambda r: (lambda k: np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.]]))(r["k"] * 2 if mono_room.METRIC_CAMERAS else raster_k)
    shell = trimesh.load(args.shell, process=False)
    pieces = [g for g in shell.geometry.values()]
    for g in pieces:  # pass the shell's photos through byte for byte: trimesh would re-encode them at JPEG quality 75
        image = getattr(getattr(g.visual, "material", None), "baseColorTexture", None)
        if getattr(image, "format", None) == "JPEG" and hasattr(getattr(image, "fp", None), "getvalue"):
            keep_bytes(image, image.fp.getvalue())
    offsets = np.cumsum([0] + [len(g.vertices) for g in pieces[:-1]])
    surface = Surface(np.concatenate([g.vertices for g in pieces]), np.concatenate([g.faces + o for g, o in zip(pieces, offsets)]))
    step, jump = args.step, JUMP * args.step / STEP  # a finer grid must not admit steeper cells
    u, v = grid(step=step)
    judged = [(r["source_index"], K_of(r), r["c2w"]) for r in rows[::5]]
    before = coverage(surface, judged, *grid())  # always on the 4-px grid, comparable between steps
    if args.video:
        assert mono_room.RASTER == "tum" and not mono_room.METRIC_CAMERAS and not any(mono_room.SOURCE_D), "--video needs an undistorted video clip (prepare_video_clip.py)"
        record = json.loads(args.video.with_suffix(".json").read_text())
        video_k, size = full_k(mono_room.SOURCE_K, record), (record["width"], record["height"])
        texture_report = args.shell.parent / "texture-report.json"  # same exposure gain per view as the shell's texture
        gains = json.loads(texture_report.read_text()).get("exposure_gain_rgb_by_view", {}) if texture_report.exists() else {}
    moving_of = lambda index: mono_room.moving_mask(args.dynamic_masks, index) if args.dynamic_masks else np.zeros((480, 640), bool)
    if args.carve:  # every view's own reliable depth (half raster), to test patch points against free space
        evidence = [(np.where(mono_room.unreliable(r["mono"], None, None, .03) | moving_of(r["source_index"]), 0, r["scale"] * r["mono"])[::2, ::2],
                     K_of(r) * [[.5], [.5], [1]], r["c2w"]) for r in rows]
        # Cameras that drifted from the map (ME340: the stretch between keyframes 14 and 226) see through all of it, so they
        # judge nothing; their own patches are judged by everyone: right from those cameras only, they misplace the room for the rest.
        on_shell = np.array([shell_agreement(surface, *e, args.carve_tolerance) for e in evidence]) >= ON_SHELL
        judges = [np.flatnonzero(on_shell) if on_shell[n] else np.arange(len(rows)) for n in range(len(rows))]
    carved = 0

    vertices, faces, colours, patches, count, filled = [], [], [], [], 0, 0
    for n, r in list(enumerate(rows))[::args.every]:
        index, K, c2w = r["source_index"], K_of(r), r["c2w"]
        depth = np.where(mono_room.unreliable(r["mono"], None, None, .03), 0, r["scale"] * r["mono"])[step // 2::step, step // 2::step]
        if args.dynamic_masks:
            moving = moving_of(index)
            if args.video and moving.any():
                moving = cv2.dilate(moving.astype(np.uint8), np.ones((2 * GROW + 1, 2 * GROW + 1), np.uint8)).astype(bool)
            depth = np.where(moving[step // 2::step, step // 2::step], 0, depth)
        if args.video and args.overlay_rows:
            depth = np.where(under_overlay(v, K, video_k, args.overlay_rows), 0, depth)
        directions, stretch = rays(u, v, K, c2w)
        hit = surface.hits(c2w[:3, 3], directions)
        missing = (depth > 0) & ~(hit <= depth * stretch * BEHIND)
        if not missing.any():
            continue
        if args.video:  # texture coordinates come from the frame itself; no raster colours
            colour = np.zeros(u.shape + (3,), np.uint8)
        else:
            colour = mono_room.prepare_image(cv2.imread(str(mono_room.DATASET / manifest["frames"][index]["relative_path"])), mono_room.CALIBRATION, 2)[0][step // 2::step, step // 2::step][..., ::-1]
        world, tri, rgb = patch(depth, missing, u, v, K, c2w, colour, jump)
        if args.carve and len(tri):  # skirts across depth edges and floaters lie where other views see through
            positive, negative = free_space(world, [evidence[j] for j in judges[n]], args.carve_tolerance)
            through = negative > np.maximum(2, .15 * positive)
            carved += int(through[tri].any(1).sum())
            used, inverse = np.unique(tri[~through[tri].any(1)], return_inverse=True)
            world, rgb, tri = world[used], rgb[used], inverse.reshape(-1, 3)
        if not len(tri):
            continue
        if args.video:
            patches.append((index, world, tri, project(world, c2w, video_k)[0], gains.get(str(index), [1., 1., 1.])))
        else:
            vertices.append(world), faces.append(tri + count), colours.append(rgb)
            count += len(world)
        surface.add(world, tri)
        filled += 1
    after = coverage(surface, judged, *grid())

    args.output.mkdir(parents=True, exist_ok=False)
    added, atlas = [], {}
    if args.video and patches:
        added, atlas = photo_atlas(lambda wanted: decode(args.video, wanted), patches, "single-view-depth-fill", args.texture_megapixels, args.jpeg_quality, size)
    elif vertices:
        added = [("single-view-depth-fill", trimesh.Trimesh(np.concatenate(vertices), np.concatenate(faces), vertex_colors=np.concatenate(colours), process=False))]
    for name, geometry in added:
        shell.add_geometry(geometry, geom_name=name)
    shell.export(args.output / "textured-scene.glb")
    stats = lambda s: {"mean": round(float(s.mean()), 4), "median": round(float(np.median(s)), 4), "worst": round(float(s.min()), 4)}
    report = {"views": len(rows), "views_used": len(rows[::args.every]), "views_that_added": filled, "every": args.every,
              "triangles_fused": int(sum(len(g.faces) for g in pieces)), "triangles_added": int(sum(len(g.faces) for _, g in added)),
              "coverage_of_keyframe_views_before": stats(before), "coverage_of_keyframe_views_after": stats(after),
              "judged_views": len(judged), "grid_step_px": step, "depth_jump_left_open": round(jump, 4), "behind_factor": BEHIND,
              "moving_pixels_left_empty": bool(args.dynamic_masks), "free_space_carving": {"rule": f"drop patch points more than max(2, .15 x agreeing) views see through ({args.carve_tolerance:.0%} relative); judged by the views that agree with the room surface (at least {ON_SHELL:.0%} of their depth); patches of views that do not are judged by all",
                                                                                     "views_off_the_room_surface": [r["source_index"] for r, ok in zip(rows, on_shell) if not ok],
                                                                                     "triangles_removed": carved} if args.carve else None, "glb_bytes": (args.output / "textured-scene.glb").stat().st_size,
              "texture": (f"{args.video.name} frames through UVs, atlas pages, exposure gain per view from {texture_report.name if gains else 'none'}; "
                          f"moving masks grown {GROW} raster px; overlay rows {args.overlay_rows} left empty" if args.video else "one vertex colour per grid sample of the 640x480 raster"),
              **atlas, "note": "added triangles are one view's own depth, not confirmed by other views; geometries single-view-depth-fill*"}
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
    beside = [np.eye(4) for _ in range(3)]
    for n, pose in enumerate(beside):
        pose[0, 3] = .2 * n
    positive, negative = free_space(np.array([[.1, 0, 2.], [.1, 0, 1.5]]), [(np.full((20, 20), 2.), K * [[.5], [.5], [1]], pose) for pose in beside])
    assert positive.tolist() == [3, 0] and negative.tolist() == [0, 3], "a point on the wall is agreed on; a skirt in front of it is seen through"
    wall = Surface(np.array([[-9., -9, 2.], [9, -9, 2.], [0, 9, 2.]]), np.array([[0, 1, 2]]))
    drifted = np.eye(4); drifted[0, 3] = .1  # a camera whose own depth puts the wall at 1
    assert shell_agreement(wall, np.full((20, 20), 2.), K * [[.5], [.5], [1]], np.eye(4), stride=2) == 1 and \
        shell_agreement(wall, np.full((20, 20), 1.), K * [[.5], [.5], [1]], drifted, stride=2) == 0, "a drifted view disagrees with the room surface"

    # --video: a patch's texture coordinates are its samples' pixels in the uncropped frame (same ray, other K)
    import io
    import trimesh
    from texture_fused_mesh import Renderer, keep_bytes, photo_atlas, project
    video_k = np.array([60., 60., 50.5, 30.5])  # the 40x40 raster seen by a wider 101x61 frame
    fine_u, fine_v = grid(40, 40, 2)
    world, faces, _ = patch(np.full(fine_u.shape, 2.), np.ones(fine_u.shape, bool), fine_u, fine_v, K, c2w, np.zeros(fine_u.shape + (3,), np.uint8))
    pixels = project(world, c2w, video_k)[0]
    raster = np.stack(np.meshgrid(np.arange(1, 40, 2.), np.arange(1, 40, 2.)), -1).reshape(-1, 2)
    assert len(faces) == 2 * 19 * 19 and np.allclose(pixels, (raster - K[[0, 1], [2, 2]]) / 40 * 60 + video_k[2:]), "a 2-px grid; UV = the sample's video pixel"
    assert under_overlay(np.array([0., 35., 39.]), K, video_k, (50, 61)).tolist() == [False, True, True], "overlay rows are found through the ray"
    yy, xx = np.mgrid[:61, :101]
    frame = np.stack([xx * 2, yy * 3, 90 + 0 * xx], -1).astype(np.uint8)
    meshes, _ = photo_atlas(lambda wanted: ((i, frame) for i in sorted(wanted)), [(3, world, faces, pixels, [1., 1., 1.])], "single-view-depth-fill", 1, 95, (101, 61))
    shell = trimesh.Scene()
    for name, geometry in meshes:
        shell.add_geometry(geometry, geom_name=name)
    image, found = Renderer(trimesh.load(io.BytesIO(shell.export(file_type="glb")), file_type="glb", process=False))(c2w, video_k, (101, 61))
    inside = found & (xx > 25) & (xx < 75) & (yy > 8) & (yy < 52)
    assert inside.sum() > 1000 and np.abs(image[inside] - frame[..., ::-1][inside]).mean() < 1.5, "the textured patch shows its own frame"
    loaded = trimesh.load(io.BytesIO(shell.export(file_type="glb")), file_type="glb", process=False)
    photo = next(iter(loaded.geometry.values())).visual.material.baseColorTexture
    original = photo.fp.getvalue()
    keep_bytes(photo, original)
    assert original[:-4] in loaded.export(file_type="glb"), "a shell's JPEG passes through the fill byte for byte"
    print("fill check passed: holes filled at their own depth, steps and moving pixels left open, rays meet the surface; "
          "skirts seen through, drifted views set apart; video patches take their frame's pixels as UVs on a 2-px grid, overlay rows found, the rendered patch shows its frame, shell JPEGs pass through")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--droid-run", type=Path)
    parser.add_argument("--depth-run", type=Path, help="the fused depth run whose mono/*.npz built the room surface")
    parser.add_argument("--shell", type=Path, help="texture_fused_mesh.py textured-scene.glb of that run")
    parser.add_argument("--dynamic-masks", type=Path, help="assemble_dynamic_masks.py masks: pixels of moving entities stay empty")
    parser.add_argument("--every", type=int, default=2, help="fill from every n-th view")
    parser.add_argument("--step", type=int, default=STEP, help="grid spacing on the 640x480 raster")
    parser.add_argument("--carve", action="store_true", help="drop patch triangles in space other views see through (their own depth lies beyond)")
    parser.add_argument("--carve-tolerance", type=float, default=CARVE, help="with --carve: relative depth beyond a patch point at which a view sees through it")
    parser.add_argument("--video", type=Path, help="the clip's uncropped source-full.mp4 (source-full.json beside it): texture patches from its frames")
    parser.add_argument("--overlay-rows", type=lambda s: tuple(int(n) for n in s.split(":")), help="with --video: Y0:Y1 rows of the video frame under burned-in subtitles")
    parser.add_argument("--texture-megapixels", type=float, default=30, help="with --video: atlas pixels the fill may add")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="with --video: atlas page JPEG quality")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else run(a)
