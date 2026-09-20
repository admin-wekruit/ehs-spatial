"""Photo texture for a fused room mesh: every triangle is coloured from the source photo that sees it best. No model call.

The fused mesh carries one colour per voxel corner, so it looks as blurry as the voxel is large however sharp the video.
Here each triangle takes the view that sees all three corners unoccluded (ray test against the mesh itself), most
frontally and from nearest; its texture coordinates are the corners' pixels in that photo. Geometry is not touched and
nothing is painted where no photo looks: triangles no kept view sees fully keep their fused vertex colour.

  python scripts/texture_fused_mesh.py --droid-run RUN --fused FUSED_RUN --output NEW_DIR [--scene REPLAY/scene.json]
  python scripts/texture_fused_mesh.py --self-check
"""
import argparse
import io
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "modal_apps"), str(ROOT / "scripts")]
HIDDEN, MARGIN, MIN_FACING = .03, 2, .2  # a hit 3% nearer hides a corner; pixels from the border; cosine below which a view is too oblique
NEARLY, MIN_CLAIM = .5, .001  # a view at half the best view's facing/distance may still claim a triangle; smallest claim worth a texture, as a share of triangles


def raycaster(vertices, faces):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)), o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return scene


def project(vertices, c2w, k):
    """Pinhole pixels and camera-space depth."""
    local = (vertices - c2w[:3, 3]) @ c2w[:3, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack([local[:, 0] / local[:, 2] * k[0] + k[2], local[:, 1] / local[:, 2] * k[1] + k[3]]), local[:, 2]


def view_scores(vertices, normals, c2w, k, size, scene):
    """Per vertex: how well this camera sees it (0 = not at all; frontal and near is better)."""
    import open3d as o3d
    pixels, depth = project(vertices, c2w, k)
    direction = vertices - c2w[:3, 3]
    distance = np.linalg.norm(direction, axis=1)
    direction = direction / np.maximum(distance, 1e-9)[:, None]
    facing = -(normals * direction).sum(1)
    ids = np.flatnonzero((depth > 1e-6) & (facing > MIN_FACING) & (pixels[:, 0] >= MARGIN) & (pixels[:, 0] < size[0] - MARGIN)
                         & (pixels[:, 1] >= MARGIN) & (pixels[:, 1] < size[1] - MARGIN))
    score = np.zeros(len(vertices), np.float32)
    if len(ids):
        rays = np.hstack([np.repeat(c2w[:3, 3][None], len(ids), 0), direction[ids]]).astype(np.float32)
        seen = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy() >= distance[ids] * (1 - HIDDEN)
        score[ids[seen]] = (facing[ids] / distance[ids])[seen]
    return score


def level(image, pixels, fused_rgb):
    """One gain per channel that brings this photo to the fused colour at its own vertices.

    The fused colour is the average over every view of that spot, so it is a common exposure reference: auto-exposure
    steps between neighbouring photos shrink without touching detail. ponytail: global gain per photo; seam blending if steps still distract.
    """
    height, width = image.shape[:2]
    sample = image[np.clip(pixels[:, 1].astype(int), 0, height - 1), np.clip(pixels[:, 0].astype(int), 0, width - 1)][:, ::-1].astype(np.float32)
    gain = np.clip(np.median((fused_rgb + 1.) / (sample + 1.), 0), .6, 1.6)
    return np.clip(image * gain[::-1], 0, 255).astype(np.uint8), [round(float(g), 3) for g in gain]


def assign(vertices, faces, normals, views, scene, budget):
    """View per face, or -1, as few large patches rather than a per-triangle patchwork (every patch border is a visible seam).

    Greedy cover: the view that sees the most still-unclaimed triangles nearly as well as their best view (score >= NEARLY * best,
    weighted by that ratio) claims them all; repeat until `budget` views are used or a claim gets tiny. What is left goes to
    the best used view that sees it at all.
    """
    def face_score(n):
        score = view_scores(vertices, normals, views[n]["c2w"], views[n]["k"], views[n]["size"], scene)[faces]
        return np.where(score.min(1) > 0, score.mean(1), 0)  # all three corners seen
    top = np.zeros(len(faces), np.float32)
    for n in range(len(views)):
        top = np.maximum(top, face_score(n))
    with np.errstate(divide="ignore", invalid="ignore"):  # one byte per view and face: quality relative to the face's best view, 0 = not nearly as good
        nearly = np.stack([np.where((ratio := face_score(n) / top) >= NEARLY, ratio * 255, 0).astype(np.uint8) for n in range(len(views))])
    winner, kept, open_ = np.full(len(faces), -1), [], top > 0
    while len(kept) < budget and open_.any():
        n = int(nearly[:, open_].sum(1, dtype=np.int64).argmax())
        claim = open_ & (nearly[n] > 0)
        if claim.sum() < max(1, MIN_CLAIM * len(faces)):
            break
        winner[claim], open_ = n, open_ & ~claim
        kept.append(n)
    best = np.zeros(len(faces), np.float32)
    for n in kept if open_.any() else []:
        score = face_score(n)
        take = open_ & (score > best)
        winner[take], best = n, np.maximum(best, score)
    return winner, sorted(kept)


def build(args):
    import open3d as o3d
    from PIL import Image
    import trimesh
    import mono_room
    mono_room.use_clip(args.droid_run)
    data = np.load(args.droid_run / "prediction.npz")
    manifest = json.loads((args.droid_run / "input-manifest.json").read_text())
    keys = {int(index): n for n, index in enumerate(data["keyframe_source_indices"])}
    mesh = o3d.io.read_triangle_mesh(str(args.fused / "mono-anchored-mesh.ply"))
    mesh.compute_vertex_normals()
    vertices, faces, normals = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_normals)
    colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    full = mono_room.RASTER == "resize"  # an undistorted pinhole source: the photo itself is the texture, at up to --texture-width
    width = min(args.texture_width, mono_room.SOURCE_WH[0]) if full else 640
    views = []
    for path in sorted((args.fused / "mono").glob("*.npz")):  # the views that built the mesh
        index = int(path.stem)
        c2w = (data["keyframe_c2w"][keys[index]] if index in keys else data["poses_c2w"][index]).astype(np.float64)
        k640 = data["keyframe_final_fullres_intrinsics"][index].astype(np.float64) * 2 if mono_room.METRIC_CAMERAS else None
        views.append({"index": index, "c2w": c2w, "k640": k640, "size": (width, width * 3 // 4)})
    for view in views:  # TUM: K of the rectified raster comes with the raster, as in fuse
        if view["k640"] is None:
            view["k640"] = np.asarray(mono_room.prepare_image(np.zeros((480, 640, 3), np.uint8), mono_room.CALIBRATION, 2)[1], np.float64)
        view["k"] = view["k640"] * (width / 640)
    winner, kept = assign(vertices, faces, normals, views, raycaster(vertices, faces), max(1, int(args.texture_megapixels * 1e6 / (width * width * 3 // 4))))
    args.output.mkdir(parents=True, exist_ok=False)
    scene, gains = trimesh.Scene(), {}
    for n in kept:
        chosen = faces[winner == n]
        if not len(chosen):
            continue
        view = views[n]
        image = cv2.imread(str(mono_room.DATASET / manifest["frames"][view["index"]]["relative_path"]))
        image = cv2.resize(image, view["size"], interpolation=cv2.INTER_AREA) if full else mono_room.prepare_image(image, mono_room.CALIBRATION, 2)[0]
        used, inverse = np.unique(chosen, return_inverse=True)
        pixels = project(vertices[used], view["c2w"], view["k"])[0]
        image, gains[view["index"]] = level(image, pixels, colors[used])
        uv = np.column_stack([pixels[:, 0] / view["size"][0], 1 - pixels[:, 1] / view["size"][1]])  # trimesh's origin is bottom-left
        jpeg = Image.open(io.BytesIO(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()))  # stays JPEG inside the GLB
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=jpeg, metallicFactor=0., roughnessFactor=1.)
        scene.add_geometry(trimesh.Trimesh(vertices[used], inverse.reshape(-1, 3), visual=trimesh.visual.TextureVisuals(uv=uv, material=material), process=False),
                           geom_name=f"view-{view['index']:05d}")
    rest = faces[winner < 0]
    if len(rest):
        used, inverse = np.unique(rest, return_inverse=True)
        scene.add_geometry(trimesh.Trimesh(vertices[used], inverse.reshape(-1, 3), vertex_colors=colors[used], process=False), geom_name="fused-colour-only")
    scene.export(args.output / "textured-scene.glb")
    area = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1) / 2
    report = {"mesh": str(args.fused / "mono-anchored-mesh.ply"), "triangles": len(faces), "views_available": len(views), "views_used": len(kept),
              "texture_size": list(views[0]["size"]), "textured_triangle_share": float((winner >= 0).mean()), "textured_area_share": float(area[winner >= 0].sum() / area.sum()),
              "glb_bytes": (args.output / "textured-scene.glb").stat().st_size, "geometry_changed": False, "newModelCalls": 0,
              "exposure_gain_rgb_by_view": gains,
              "rule": f"a view may texture a triangle whose three corners it sees (ray test, {HIDDEN:.0%} tolerance, facing cosine > {MIN_FACING}); greedy cover by views within {NEARLY:.0%} of the "
                      "triangle's best facing/distance, so patches are large and seams few; "
                      "one exposure gain per photo towards the fused colour, no seam blending"}
    (args.output / "texture-report.json").write_text(json.dumps(report, indent=1))
    if args.scene:  # the same replay scene, showing the textured mesh
        replay = json.loads(args.scene.read_text())
        relative = lambda url: os.path.relpath((args.scene.parent / url).resolve(), args.output.resolve())
        replay["pointCloudUrl"] = relative(replay["pointCloudUrl"])
        for item in [o["surface"] for f in replay["frames"] for o in f["objects"] if o.get("surface")] + replay.get("staticObjects", []):
            item["meshUrl"] = relative(item["meshUrl"])
        for item in replay.get("staticObjects", []):
            item["provenanceUrl"] = relative(item["provenanceUrl"])
            item["source"].update(maskUrl=relative(item["source"]["maskUrl"]), imageUrl=relative(item["source"]["imageUrl"]))
            if item.get("generatedModel"):
                item["generatedModel"].update(meshUrl=relative(item["generatedModel"]["meshUrl"]), provenanceUrl=relative(item["generatedModel"]["provenanceUrl"]))
        replay["meshUrl"] = "textured-scene.glb"
        replay["limitations"] = replay.get("limitations", []) + ["网格外观取自看得最清楚的那张源照片（几何不变）；相邻照片曝光不同处会有接缝；没有照片完整看到的三角面保留融合颜色。"]
        replay.setdefault("provenance", {})["texture"] = report
        (args.output / "scene.json").write_text(json.dumps(replay, allow_nan=False, separators=(",", ":")))
    print(json.dumps(report))


def self_check():
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9)), -1).reshape(-1, 2)
    vertices = np.column_stack([grid, np.full(len(grid), 3.)])  # a wall 3 units ahead, facing the origin
    quads = [(r * 9 + c, r * 9 + c + 1, r * 9 + c + 10, r * 9 + c + 9) for r in range(8) for c in range(8)]
    faces = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    normals = np.tile([0, 0, -1.], (len(vertices), 1))
    k, size = np.array([200., 200, 160, 120]), (320, 240)
    near, far = np.eye(4), np.eye(4); far[2, 3] = -2.
    views = [{"c2w": far, "k": k, "size": size}, {"c2w": near, "k": k, "size": size}]
    winner, kept = assign(vertices, faces, normals, views, raycaster(vertices, faces), 9)
    assert (winner == 1).all() and kept == [1], "the nearer of two frontal views must win every triangle"
    blocker = np.array([[0, -2, 1.5], [2, -2, 1.5], [2, 2, 1.5], [0, 2, 1.5]])  # hides the right half from the near camera only partly from the far one
    both = raycaster(np.vstack([vertices, blocker]), np.vstack([faces, [[81, 82, 83], [81, 83, 84]]]))
    winner, kept = assign(vertices, faces, normals, views, both, 9)
    centre = vertices[faces].mean(1)[:, 0]
    assert (winner[centre < -.2] == 1).all() and (winner[centre > .6] != 1).all(), "a triangle hidden from the best view must not take its photo"
    assert np.allclose(project(vertices[:1], near, k)[0], [[160 - 200 / 3, 120 - 200 / 3]]), "texture coordinate must be the pinhole pixel"
    assert assign(vertices, faces, normals, views, raycaster(vertices, faces), 1)[1] == [1], "the view budget keeps the views that win most"
    dark = np.full((240, 320, 3), 80, np.uint8)
    assert np.allclose(level(dark, np.array([[10., 10], [300, 200]]), np.full((2, 3), 120.))[0], 120, atol=1), "a photo darker than the fused colour is lifted to it"
    print("texture check passed: nearest frontal view wins, occluded triangles refuse the photo, pinhole texture coordinates, view budget, exposure levelled to the fused colour")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    for name in ("droid-run", "fused", "output", "scene"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--texture-width", type=int, default=1280)
    parser.add_argument("--texture-megapixels", type=float, default=60, help="total texture pixels the browser has to hold; sets how many views may contribute")
    a = parser.parse_args()
    self_check() if a.self_check else build(a)
