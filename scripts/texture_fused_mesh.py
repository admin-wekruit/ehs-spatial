"""Photo texture for a fused room mesh: every triangle is coloured from the source photo that sees it best. No model call.

The fused mesh carries one colour per voxel corner, so it looks as blurry as the voxel is large however sharp the video.
Here each triangle takes the view that sees all three corners unoccluded (ray test against the mesh itself), most
frontally and from nearest; its texture coordinates are the corners' pixels in that photo. Geometry is not touched and
nothing is painted where no photo looks: triangles no kept view sees fully keep their fused vertex colour.

With --video (the clip's uncropped source-full.mp4) the photos are the full video frames instead of the 640x480 clip
raster. A view's claim on a triangle is its effective resolution (projected pixel area / distance) x frontality x the
frame's sharpness (variance of the Laplacian against its neighbours in time, so motion blur and worse-encoded frames
lose). Pixels on moving entities (--dynamic-masks) and burned-in subtitles (--overlay-rows) never lend colour. Only the
pixels the triangles use are cut from the frames and packed into atlas pages, so every view may contribute.

--heldout renders textured GLBs into every n-th video frame (Open3D ray casting, the viewer's texture x vertex colour)
and scores them against the real frame: PSNR, SSIM, coverage, detail (gradient) ratio; moving and overlay pixels excluded.

  python scripts/texture_fused_mesh.py --droid-run RUN --fused FUSED_RUN --output NEW_DIR [--scene REPLAY/scene.json]
  python scripts/texture_fused_mesh.py --droid-run RUN --fused FUSED_RUN --output NEW_DIR --video CLIP/source-full.mp4 \\
      [--dynamic-masks MASKS] [--overlay-rows 648:704]
  python scripts/texture_fused_mesh.py --heldout OLD.glb NEW.glb --droid-run RUN --fused FUSED_RUN --video V \\
      [--dynamic-masks MASKS] [--overlay-rows 648:704] --output NEW_DIR
  python scripts/texture_fused_mesh.py --self-check
"""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "modal_apps"), str(ROOT / "scripts")]
HIDDEN, MARGIN, MIN_FACING = .03, 2, .2  # a hit 3% nearer hides a corner; pixels from the border; cosine below which a view is too oblique
NEARLY, MIN_CLAIM = .5, .001  # a view at half the best view's facing/distance may still claim a triangle; smallest claim worth a texture, as a share of triangles
PAGE, TILE, PAD = 4096, 128, 2  # atlas page side (every WebGL2 device takes 4096); photo tile a crop is grouped by; real photo kept around a crop
GROW, SHARP_WINDOW = 15, 31  # full-frame pixels a moving-entity mask is grown by; frames a frame's sharpness is judged against
AGREE = .08  # with --video: a view's own depth within this share of a vertex's distance shows that vertex


def raycaster(vertices, faces):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)), o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return scene


def project(vertices, c2w, k):
    """Pinhole pixels and camera-space depth."""
    local = np.einsum("nj,jk->nk", vertices - c2w[:3, 3], c2w[:3, :3])  # not `@`: Accelerate's dgemm over-reads big (n, 3) inputs and segfaulted
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack([local[:, 0] / local[:, 2] * k[0] + k[2], local[:, 1] / local[:, 2] * k[1] + k[3]]), local[:, 2]


def full_k(clip_k, record):
    """fx, fy, cx, cy of the uncropped video frame, from the 640x480 clip raster's K and where that raster sits (source-full.json).

    The clip raster is an area resize of a crop, which keeps pixel centres aligned: clip pixel x is video pixel (x + .5) * s - .5 + x0.
    """
    x0, y0, w, h = record["raster_in_video_xywh"]
    sx, sy = w / record["raster_wh"][0], h / record["raster_wh"][1]
    fx, fy, cx, cy = clip_k
    return np.array([fx * sx, fy * sy, (cx + .5) * sx - .5 + x0, (cy + .5) * sy - .5 + y0])


def decode(video, wanted):
    """(index, BGR frame) of the wanted frame indices in order, one frame in memory at a time."""
    wanted = set(wanted)
    capture = cv2.VideoCapture(str(video))
    for index in range(max(wanted) + 1):
        ok, frame = capture.read()
        assert ok, f"{video} has no frame {index}"
        if index in wanted:
            yield index, frame
    capture.release()


def usable_pixels(masks, index, record, overlay=None, grow=GROW):
    """Full-frame pixels a photo may lend the room: off moving entities (clip-raster masks, grown) and off burned-in overlay rows."""
    x0, y0, w, h = (int(round(value)) for value in record["raster_in_video_xywh"])
    bad = np.zeros((record["height"], record["width"]), np.uint8)
    for path in sorted(masks.glob(f"{index:05d}-*.png")) if masks else []:
        bad[y0:y0 + h, x0:x0 + w] |= cv2.resize((cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if bad.any():
        bad = cv2.dilate(bad, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    if overlay:
        bad[overlay[0]:overlay[1]] = 1
    return bad == 0


def sharpness_weight(variance, window=SHARP_WINDOW):
    """Variance of the Laplacian over its running median in time, clipped and squared like a pixel area.

    Content changes slowly along a walk-through, so a frame blurrier than its neighbours is motion blur or a worse-encoded frame.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    median = np.median(sliding_window_view(np.pad(variance, window // 2, mode="edge"), window), axis=1)
    return np.clip(variance / median, .5, 1.5) ** 2


def view_scores(vertices, normals, c2w, k, size, scene, min_facing=MIN_FACING):
    """Per vertex: how well this camera sees it (0 = not at all; frontal and near is better)."""
    import open3d as o3d
    pixels, depth = project(vertices, c2w, k)
    direction = vertices - c2w[:3, 3]
    distance = np.linalg.norm(direction, axis=1)
    direction = direction / np.maximum(distance, 1e-9)[:, None]
    facing = -(normals * direction).sum(1)
    ids = np.flatnonzero((depth > 1e-6) & (facing > min_facing) & (pixels[:, 0] >= MARGIN) & (pixels[:, 0] < size[0] - MARGIN)
                         & (pixels[:, 1] >= MARGIN) & (pixels[:, 1] < size[1] - MARGIN))
    score = np.zeros(len(vertices), np.float32)
    if len(ids):
        rays = np.hstack([np.repeat(c2w[:3, 3][None], len(ids), 0), direction[ids]]).astype(np.float32)
        seen = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy() >= distance[ids] * (1 - HIDDEN)
        score[ids[seen]] = (facing[ids] / distance[ids])[seen]
    return score


def resolution_scores(vertices, faces, normals, c2w, k, size, scene, usable, min_facing, depth=None):
    """Per face: projected pixel area / distance x frontality where all three corners are seen on usable pixels (else 0);
    also every vertex's pixel and whether it is seen.

    depth: (the view's own depth map, its K). Where it knows the depth and disagrees, the pixel shows something else: an
    occluder the mesh lacks, or a camera that drifted from the map (a stretch without keyframes) and would paint misplaced photo.
    """
    score = view_scores(vertices, normals, c2w, k, size, scene, min_facing)  # facing / distance
    pixels = project(vertices, c2w, k)[0]
    seen = np.flatnonzero(score > 0)  # MARGIN keeps their rounded pixels inside the frame
    score[seen[~usable[np.rint(pixels[seen, 1]).astype(int), np.rint(pixels[seen, 0]).astype(int)]]] = 0
    if depth is not None:
        own, own_k = depth
        seen = np.flatnonzero(score > 0)
        at, z = project(vertices[seen], c2w, own_k)
        x, y = np.rint(at).astype(int).T
        inside = (x >= 0) & (x < own.shape[1]) & (y >= 0) & (y < own.shape[0])
        known = np.zeros(len(seen))
        known[inside] = own[y[inside], x[inside]]
        score[seen[(known > 0) & (np.abs(known / z - 1) > AGREE)]] = 0
    corner, per = pixels[faces], score[faces]
    a, b = corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]
    with np.errstate(invalid="ignore"):
        area = .5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        return np.where(per.min(1) > 0, area * per.mean(1), 0).astype(np.float32), pixels, score > 0


def exposure_gain(image, pixels, fused_rgb):
    """One gain per RGB channel that brings this BGR photo to the fused colour at the given vertex pixels."""
    height, width = image.shape[:2]
    sample = image[np.clip(pixels[:, 1].astype(int), 0, height - 1), np.clip(pixels[:, 0].astype(int), 0, width - 1)][:, ::-1].astype(np.float32)
    return np.clip(np.median((fused_rgb + 1.) / (sample + 1.), 0), .6, 1.6)


def level(image, pixels, fused_rgb):
    """One gain per channel that brings this photo to the fused colour at its own vertices.

    The fused colour is the average over every view of that spot, so it is a common exposure reference: auto-exposure
    steps between neighbouring photos shrink without touching detail. ponytail: global gain per photo; seam blending if steps still distract.
    """
    gain = exposure_gain(image, pixels, fused_rgb)
    return np.clip(image * gain[::-1], 0, 255).astype(np.uint8), [round(float(g), 3) for g in gain]


def seen_fully(vertices, faces, normals, views, scene):
    """face_score for assign on the clip raster: mean facing/distance of the corners where all three are seen."""
    def face_score(n):
        score = view_scores(vertices, normals, views[n]["c2w"], views[n]["k"], views[n]["size"], scene)[faces]
        return np.where(score.min(1) > 0, score.mean(1), 0).astype(np.float32)
    return face_score


def assign(face_score, views, budget, claim=MIN_CLAIM, nearly=NEARLY):
    """View per face, or -1, as few large patches rather than a per-triangle patchwork (every patch border is a visible seam).

    face_score(n): how well view n shows every face (0 = not at all). Greedy cover: the view that sees the most
    still-unclaimed triangles nearly as well as their best view (score >= nearly * best, weighted by that ratio) claims
    them all; repeat until `budget` views are used or a claim is under `claim` of the triangles (0: until every seen
    triangle is claimed). What is left goes to the best used view that sees it at all.
    """
    top = face_score(0).astype(np.float32)
    for n in range(1, views):
        top = np.maximum(top, face_score(n))
    with np.errstate(divide="ignore", invalid="ignore"):  # one byte per view and face: quality relative to the face's best view, 0 = not nearly as good
        ratio = np.stack([np.where((r := face_score(n) / top) >= nearly, r * 255, 0).astype(np.uint8) for n in range(views)])
    winner, kept, open_ = np.full(len(top), -1), [], top > 0
    while len(kept) < budget and open_.any():
        n = int(ratio[:, open_].sum(1, dtype=np.int64).argmax())
        take = open_ & (ratio[n] > 0)
        if take.sum() < max(1, claim * len(top)):
            break
        winner[take], open_ = n, open_ & ~take
        kept.append(n)
    best = np.zeros(len(top), np.float32)
    for n in kept if open_.any() else []:
        score = face_score(n)
        take = open_ & (score > best)
        winner[take], best = n, np.maximum(best, score)
    return winner, sorted(kept)


def keep_bytes(image, data):
    """Make trimesh embed exactly `data` for this PIL image: its GLB exporter calls image.save(), which re-encodes a JPEG at PIL's quality 75."""
    image.save = lambda file, format=None, **_: file.write(data)
    return image


def jpeg(bgr, quality):
    from PIL import Image
    data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    return keep_bytes(Image.open(io.BytesIO(data)), data)


def tiles(pixels, faces, size, tile=TILE, pad=PAD):
    """A view's triangles grouped by the photo tile their centre falls in, each group with a crop box (x0, y0, x1, y1)
    that holds all its corners plus `pad` pixels, so bilinear sampling never leaves the crop."""
    centre = pixels[faces].mean(1)
    key = np.floor(centre[:, 1] / tile).astype(np.int64) * 65536 + np.floor(centre[:, 0] / tile).astype(np.int64)
    order = np.argsort(key, kind="stable")
    groups = []
    for ids in np.split(order, np.flatnonzero(np.diff(key[order])) + 1):
        corner = pixels[faces[ids]].reshape(-1, 2)
        x0, y0 = np.maximum(np.floor(corner.min(0)).astype(int) - pad, 0)
        x1, y1 = np.minimum(np.ceil(corner.max(0)).astype(int) + pad + 1, size)
        groups.append((ids, (int(x0), int(y0), int(x1), int(y1))))
    return groups


def shelf(sizes, page=PAGE):
    """(sheet, x, y) for every (w, h) box, tallest first, left to right on shelves of page x page sheets; and each sheet's used (w, h)."""
    places, sheets, x, y, height = [None] * len(sizes), [], page, 0, 0
    for n in sorted(range(len(sizes)), key=lambda n: -sizes[n][1]):
        w, h = sizes[n]
        if x + w > page:
            x, y, height = 0, y + height, 0
        if not sheets or y + h > page:
            sheets.append([0, 0])
            x, y, height = 0, 0, 0
        places[n] = (len(sheets) - 1, x, y)
        sheets[-1] = [max(sheets[-1][0], x + w), max(sheets[-1][1], y + h)]
        x, height = x + w, max(height, h)
    return places, sheets


def photo_atlas(read, patches, name, megapixels, quality, size):
    """Meshes textured with only the photo pixels their triangles use, one mesh per atlas page.

    patches: [(frame index, vertices, faces, the vertices' pixels in that frame, RGB gain)]; read(indices) yields
    (index, BGR frame). Crops are cut from the real photo with a margin, levelled by the patch's gain and shelf-packed;
    all of them shrink together if they would exceed `megapixels`. Returns [(name, trimesh.Trimesh)] and atlas facts.
    """
    import trimesh
    boxes = [(n, ids, box) for n, (_, _, faces, pixels, _) in enumerate(patches) for ids, box in tiles(pixels, faces, size)]
    area = sum((x1 - x0) * (y1 - y0) for _, _, (x0, y0, x1, y1) in boxes)
    scale = min(1., np.sqrt(megapixels * 1e6 / max(area, 1)))
    sizes = [(max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))) for _, _, (x0, y0, x1, y1) in boxes]
    places, sheets = shelf(sizes)
    pages, by_frame = [np.zeros((h, w, 3), np.uint8) for w, h in sheets], {}
    for number, (n, _, _) in enumerate(boxes):
        by_frame.setdefault(patches[n][0], []).append(number)
    for index, frame in read(set(by_frame)):
        for number in by_frame[index]:
            n, _, (x0, y0, x1, y1) = boxes[number]
            crop = np.clip(frame[y0:y1, x0:x1] * np.asarray(patches[n][4], np.float32)[::-1], 0, 255).astype(np.uint8)  # RGB gain, BGR frame
            (w, h), (sheet, px, py) = sizes[number], places[number]
            pages[sheet][py:py + h, px:px + w] = crop if (w, h) == (x1 - x0, y1 - y0) else cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    parts = [[] for _ in pages]
    for number, (n, ids, (x0, y0, x1, y1)) in enumerate(boxes):
        _, vertices, faces, pixels, _ = patches[n]
        used, inverse = np.unique(faces[ids], return_inverse=True)
        (w, h), (sheet, px, py) = sizes[number], places[number]
        texel = (pixels[used] - [x0, y0] + .5) * [w / (x1 - x0), h / (y1 - y0)] + [px, py]  # position on the page; texel centres at +.5
        parts[sheet].append((vertices[used], inverse.reshape(-1, 3), texel))
    meshes = []
    for sheet, (page, part) in enumerate(zip(pages, parts)):
        offsets = np.cumsum([0] + [len(v) for v, _, _ in part[:-1]])
        texel = np.concatenate([t for _, _, t in part])
        uv = np.column_stack([texel[:, 0] / page.shape[1], 1 - texel[:, 1] / page.shape[0]])  # trimesh's origin is bottom-left
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=jpeg(page, quality), metallicFactor=0., roughnessFactor=1.)
        meshes.append((f"{name}-{sheet:02d}", trimesh.Trimesh(np.concatenate([v for v, _, _ in part]), np.concatenate([f + o for (_, f, _), o in zip(part, offsets)]),
                                                             visual=trimesh.visual.TextureVisuals(uv=uv, material=material), process=False)))
    return meshes, {"atlas_pages_wh": sheets, "atlas_crops": len(boxes), "texture_pixels": int(sum(w * h for w, h in sizes)), "texture_scale": round(float(scale), 4)}


def video_texture(args, vertices, faces, normals, colors, views):
    """Winner view per face, kept views, atlas meshes and facts, from the uncropped video frames."""
    import mono_room
    assert mono_room.RASTER == "tum" and not mono_room.METRIC_CAMERAS and not any(mono_room.SOURCE_D), "--video needs an undistorted video clip (prepare_video_clip.py)"
    record = json.loads(args.video.with_suffix(".json").read_text())
    k, size = full_k(mono_room.SOURCE_K, record), (record["width"], record["height"])
    scene, wanted = raycaster(vertices, faces), {view["index"]: n for n, view in enumerate(views)}
    rows = {r["source_index"]: r for r in mono_room.load(args.droid_run, None, args.fused)}  # each view's own depth (native units)
    claims, gains, variance = np.zeros((len(views), len(faces)), np.float32), {}, []
    for index, frame in decode(args.video, range(record["frames"])):  # every frame: sharpness is judged against the neighbours
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance.append(cv2.Laplacian(np.delete(grey, np.s_[slice(*args.overlay_rows)], 0) if args.overlay_rows else grey, cv2.CV_32F).var())
        if index in wanted:
            usable = usable_pixels(args.dynamic_masks, index, record, args.overlay_rows)
            own = np.where(mono_room.unreliable(rows[index]["mono"], None, None, .03), 0, rows[index]["scale"] * rows[index]["mono"])
            claims[wanted[index]], pixels, seen = resolution_scores(vertices, faces, normals, views[wanted[index]]["c2w"], k, size, scene, usable,
                                                                    args.min_facing, (own, views[wanted[index]]["k640"]))
            gains[index] = exposure_gain(frame, pixels[seen], colors[seen]) if seen.any() else np.ones(3, np.float32)
    del rows
    common = np.median(list(gains.values()), 0)  # the fused colour comes from the clip raster, a few % off this video's colours: level views to each other only
    gains = {index: gain / common for index, gain in gains.items()}
    weight = sharpness_weight(np.array(variance))
    for n, view in enumerate(views):
        claims[n] *= weight[view["index"]]
    winner, kept = assign(lambda n: claims[n], len(views), len(views), claim=0, nearly=args.nearly)
    del claims
    patches = []
    for n in kept:
        used, inverse = np.unique(faces[winner == n], return_inverse=True)
        patches.append((views[n]["index"], vertices[used], inverse.reshape(-1, 3), project(vertices[used], views[n]["c2w"], k)[0], gains[views[n]["index"]]))
    meshes, facts = photo_atlas(lambda indices: decode(args.video, indices), patches, "atlas", args.texture_megapixels, args.jpeg_quality, size)
    facts.update(texture_source=f"{args.video.name} {size[0]}x{size[1]} uncropped frames", video_k_fx_fy_cx_cy=[round(float(v), 4) for v in k],
                 fused_over_video_rgb=[round(float(g), 4) for g in common],
                 exposure_gain_rgb_by_view={str(i): [round(float(g), 3) for g in gain] for i, gain in gains.items()},
                 sharpness_weight_by_view={str(v["index"]): round(float(weight[v["index"]]), 3) for v in views},
                 min_facing=args.min_facing, nearly=args.nearly, overlay_rows=args.overlay_rows, dynamic_masks=str(args.dynamic_masks) if args.dynamic_masks else None,
                 rule=f"a view may texture a triangle whose three corners it sees (ray test, {HIDDEN:.0%} tolerance, facing cosine > {args.min_facing}; its own depth, where known, "
                      f"within {AGREE:.0%}) on pixels off moving "
                      f"entities (masks grown {GROW} px) and overlay rows; claim = projected pixel area / distance x facing x sharpness weight (Laplacian variance over its "
                      f"{SHARP_WINDOW}-frame running median, clipped to .5-1.5, squared); greedy cover by views within {args.nearly:.0%} of the triangle's best claim until every "
                      "seen triangle is claimed; one exposure gain per photo towards the fused colour, divided by the median gain of all views; only the used pixels are packed into atlas pages")
    return winner, kept, meshes, facts


def build(args):
    import open3d as o3d
    import trimesh
    import mono_room
    started = time.time()
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
    args.output.mkdir(parents=True, exist_ok=False)
    scene, gains = trimesh.Scene(), {}
    if args.video:
        winner, kept, meshes, facts = video_texture(args, vertices, faces, normals, colors, views)
        for name, geometry in meshes:
            scene.add_geometry(geometry, geom_name=name)
    else:
        winner, kept = assign(seen_fully(vertices, faces, normals, views, raycaster(vertices, faces)), len(views),
                              max(1, int(args.texture_megapixels * 1e6 / (width * width * 3 // 4))))
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
            material = trimesh.visual.material.PBRMaterial(baseColorTexture=jpeg(image, 88), metallicFactor=0., roughnessFactor=1.)
            scene.add_geometry(trimesh.Trimesh(vertices[used], inverse.reshape(-1, 3), visual=trimesh.visual.TextureVisuals(uv=uv, material=material), process=False),
                               geom_name=f"view-{view['index']:05d}")
        facts = {"texture_size": list(views[0]["size"]), "exposure_gain_rgb_by_view": gains,
                 "rule": f"a view may texture a triangle whose three corners it sees (ray test, {HIDDEN:.0%} tolerance, facing cosine > {MIN_FACING}); greedy cover by views within {NEARLY:.0%} of the "
                         "triangle's best facing/distance, so patches are large and seams few; "
                         "one exposure gain per photo towards the fused colour, no seam blending"}
    rest = faces[winner < 0]
    if len(rest):
        if args.video:  # the same colours as the textured triangles next to them
            colors = np.clip(colors / np.asarray(facts["fused_over_video_rgb"]), 0, 255).astype(np.uint8)
        used, inverse = np.unique(rest, return_inverse=True)
        scene.add_geometry(trimesh.Trimesh(vertices[used], inverse.reshape(-1, 3), vertex_colors=colors[used], process=False), geom_name="fused-colour-only")
    scene.export(args.output / "textured-scene.glb")
    area = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1) / 2
    report = {"mesh": str(args.fused / "mono-anchored-mesh.ply"), "triangles": len(faces), "views_available": len(views), "views_used": len(kept),
              "textured_triangle_share": float((winner >= 0).mean()), "textured_area_share": float(area[winner >= 0].sum() / area.sum()),
              "glb_bytes": (args.output / "textured-scene.glb").stat().st_size, "geometry_changed": False, "newModelCalls": 0,
              "seconds": round(time.time() - started, 1), **facts}
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
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("_by_view")}))


def bilinear(image, x, y):
    """Clamp-to-edge bilinear samples of an HxWx3 uint8 image at float pixel positions (texel centres at integers), as float32."""
    height, width = image.shape[:2]
    x, y = np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)
    x0, y0 = np.minimum(x.astype(int), max(width - 2, 0)), np.minimum(y.astype(int), max(height - 2, 0))
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    ax, ay = (x - x0)[:, None], (y - y0)[:, None]
    return ((image[y0, x0] * (1 - ax) + image[y0, x1] * ax) * (1 - ay) + (image[y1, x0] * (1 - ax) + image[y1, x1] * ax) * ay).astype(np.float32)


class Renderer:
    """A textured trimesh scene drawn as the report viewer draws it: nearest surface, both sides, texture x vertex colour, linear filter."""

    def __init__(self, scene):
        import open3d as o3d
        import trimesh
        self.cast, self.parts = o3d.t.geometry.RaycastingScene(), {}
        for node in scene.graph.nodes_geometry:
            transform, name = scene.graph[node]
            geometry = scene.geometry[name]
            vertices = trimesh.transform_points(np.asarray(geometry.vertices), transform)
            gid = self.cast.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(np.asarray(geometry.faces, np.uint32)))
            visual = geometry.visual
            uv, image = getattr(visual, "uv", None), getattr(getattr(visual, "material", None), "baseColorTexture", None)
            texture = np.asarray(image.convert("RGB")) if uv is not None and image is not None else None
            colour = np.asarray(visual.vertex_colors)[:, :3].astype(np.float32) / 255 if getattr(visual, "kind", None) == "vertex" else None
            self.parts[gid] = (np.asarray(geometry.faces), None if texture is None else np.asarray(uv, np.float64), texture, colour)

    def __call__(self, c2w, k, size):
        """RGB float image (0 where nothing is hit) and the hit mask."""
        import open3d as o3d
        width, height = size
        v, u = np.mgrid[:height, :width]
        direction = np.einsum("hwj,kj->hwk", np.stack([(u - k[2]) / k[0], (v - k[3]) / k[1], np.ones((height, width))], -1), c2w[:3, :3])
        rays = np.concatenate([np.broadcast_to(c2w[:3, 3], direction.shape), direction], -1).astype(np.float32).reshape(-1, 6)
        found, gid, pid, bary = [], [], [], []
        hit = self.cast.cast_rays(o3d.core.Tensor(rays))
        found = np.isfinite(hit["t_hit"].numpy()).reshape(height, width)
        gid, pid = hit["geometry_ids"].numpy().reshape(height, width), hit["primitive_ids"].numpy().reshape(height, width)
        bary = hit["primitive_uvs"].numpy().reshape(height, width, 2)
        image = np.zeros((height, width, 3), np.float32)
        for g, (faces, uv, texture, colour) in self.parts.items():
            pick = found & (gid == g)
            if not pick.any():
                continue
            corner, b = faces[pid[pick]], bary[pick]
            weight = np.column_stack([1 - b.sum(1), b])[:, :, None]  # Embree: point = (1-u-v) p0 + u p1 + v p2
            rgb = np.full((len(corner), 3), 255, np.float32)
            if texture is not None:
                t = (weight * uv[corner]).sum(1)
                rgb = bilinear(texture, t[:, 0] * texture.shape[1] - .5, (1 - t[:, 1]) * texture.shape[0] - .5)
            if colour is not None:
                rgb = rgb * (weight * colour[corner]).sum(1)
            image[pick] = rgb
        return image, found


def ssim_map(a, b):
    """SSIM (Wang et al. 2004: 11x11 Gaussian window, sigma 1.5, 8-bit constants) per pixel, averaged over channels."""
    c1, c2 = (.01 * 255) ** 2, (.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    return (((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))).mean(-1)


def scores(real, render, valid):
    """PSNR (dB) over the valid pixels; SSIM and detail ratio over valid pixels whose whole SSIM window is valid."""
    inner = cv2.erode(valid.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    filled = np.where(valid[..., None], render, real)
    mse = float(((real[valid] - render[valid]) ** 2).mean())
    grad = lambda x: np.hypot(cv2.Sobel(x.mean(-1), cv2.CV_32F, 1, 0), cv2.Sobel(x.mean(-1), cv2.CV_32F, 0, 1))
    return {"psnr": 10 * np.log10(255 ** 2 / max(mse, 1e-6)), "ssim": float(ssim_map(real, filled)[inner].mean()) if inner.any() else float("nan"),
            "detail": float(grad(filled)[inner].mean() / max(grad(real)[inner].mean(), 1e-6)) if inner.any() else float("nan")}


def heldout(args):
    """Render every GLB into each --every-th video frame and score it against the real frame; side-by-sides of four frames."""
    import trimesh
    import mono_room
    mono_room.use_clip(args.droid_run)
    record = json.loads(args.video.with_suffix(".json").read_text())
    k, size = full_k(mono_room.SOURCE_K, record), (record["width"], record["height"])
    data = np.load(args.droid_run / "prediction.npz")
    keys = {int(index): n for n, index in enumerate(data["keyframe_source_indices"])}
    sources = {int(path.stem) for path in (args.fused / "mono").glob("*.npz")}  # every frame a texture or fill patch could come from
    frames = list(range(0, record["frames"], args.every))
    unseen = [i for i in frames if i not in sources]
    pictures = {unseen[len(unseen) * q // 8] for q in (1, 3, 5, 7)}
    names = [str(path) for path in args.heldout]
    renderers = [Renderer(trimesh.load(path, process=False)) for path in args.heldout]
    x0, y0, w, h = (int(round(value)) for value in record["raster_in_video_xywh"])
    centre = np.zeros((size[1], size[0]), bool)
    centre[y0:y0 + h, x0:x0 + w] = True  # where the clip raster, hence every depth map, lies
    args.output.mkdir(parents=True, exist_ok=False)
    per_frame, started = [], time.time()
    for index, frame in decode(args.video, frames):
        c2w = (data["keyframe_c2w"][keys[index]] if index in keys else data["poses_c2w"][index]).astype(np.float64)
        real, usable = frame[..., ::-1].astype(np.float32), usable_pixels(args.dynamic_masks, index, record, args.overlay_rows, grow=10)
        renders = [render(c2w, k, size) for render in renderers]
        common = np.logical_and.reduce([found for _, found in renders]) & usable
        row = {"frame": index, "never_a_source": index not in sources, "usable_pixels": int(usable.sum())}
        for name, (image, found) in zip(names, renders):
            valid = found & usable
            row[name] = {"coverage": float(valid.sum() / usable.sum()), "coverage_centre": float((valid & centre).sum() / (usable & centre).sum()),
                         "coverage_sides": float((valid & ~centre).sum() / max((usable & ~centre).sum(), 1)), **(scores(real, image, valid) if valid.any() else {}),
                         **{f"common_{key}": value for key, value in (scores(real, image, common) if common.any() else {}).items()}}
        per_frame.append(row)
        if index in pictures:
            side = np.hstack([real] + [np.where(found[..., None], image, 0) for image, found in renders])
            cv2.imwrite(str(args.output / f"heldout-{index:05d}.png"), np.clip(side, 0, 255).astype(np.uint8)[..., ::-1])
    summary = {}
    for subset, chosen in (("all_heldout_frames", per_frame), ("never_a_source_frames", [r for r in per_frame if r["never_a_source"]])):
        summary[subset] = {"frames": len(chosen)}
        for name in names:
            summary[subset][name] = {key: round(float(np.nanmean([r[name][key] for r in chosen if key in r[name]])), 4)
                                     for key in ("coverage", "coverage_centre", "coverage_sides", "psnr", "ssim", "detail", "common_psnr", "common_ssim", "common_detail")}
    result = {"glbs": names, "every": args.every, "frames": frames, "side_by_side_frames": sorted(pictures), "side_by_side": "real | " + " | ".join(names),
              "video": str(args.video), "video_k_fx_fy_cx_cy": [round(float(v), 4) for v in k], "overlay_rows_excluded": args.overlay_rows,
              "moving_pixels_excluded": str(args.dynamic_masks) if args.dynamic_masks else None,
              "definitions": {"coverage": "share of usable pixels (not moving, not overlay) where the render hits geometry",
                              "coverage_centre": "the same inside the 4:3 clip raster, where depth was estimated; coverage_sides: outside it",
                              "psnr": "per frame over valid pixels (hit and usable), RGB 0-255, then averaged over frames",
                              "ssim": "per frame over valid pixels whose 11x11 window is all valid, Gaussian sigma 1.5, mean of RGB channels",
                              "detail": "mean Sobel gradient magnitude of the render over the real frame on the same pixels (1 = as much fine detail as the video)",
                              "common_*": "the same on pixels every compared GLB covers",
                              "never_a_source_frames": "held-out frames that are not among the depth run's views, so neither texture nor fill could have come from them"},
              "summary": summary, "seconds": round(time.time() - started, 1), "per_frame": per_frame}
    (args.output / "heldout-metrics.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(summary, indent=1))


def self_check():
    import tempfile
    import trimesh
    from PIL import Image
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9)), -1).reshape(-1, 2)
    vertices = np.column_stack([grid, np.full(len(grid), 3.)])  # a wall 3 units ahead, facing the origin
    quads = [(r * 9 + c, r * 9 + c + 1, r * 9 + c + 10, r * 9 + c + 9) for r in range(8) for c in range(8)]
    faces = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    normals = np.tile([0, 0, -1.], (len(vertices), 1))
    k, size = np.array([200., 200, 160, 120]), (320, 240)
    near, far = np.eye(4), np.eye(4); far[2, 3] = -2.
    views = [{"c2w": far, "k": k, "size": size}, {"c2w": near, "k": k, "size": size}]
    winner, kept = assign(seen_fully(vertices, faces, normals, views, raycaster(vertices, faces)), 2, 9)
    assert (winner == 1).all() and kept == [1], "the nearer of two frontal views must win every triangle"
    blocker = np.array([[0, -2, 1.5], [2, -2, 1.5], [2, 2, 1.5], [0, 2, 1.5]])  # hides the right half from the near camera only partly from the far one
    both = raycaster(np.vstack([vertices, blocker]), np.vstack([faces, [[81, 82, 83], [81, 83, 84]]]))
    winner, kept = assign(seen_fully(vertices, faces, normals, views, both), 2, 9)
    centre = vertices[faces].mean(1)[:, 0]
    assert (winner[centre < -.2] == 1).all() and (winner[centre > .6] != 1).all(), "a triangle hidden from the best view must not take its photo"
    assert np.allclose(project(vertices[:1], near, k)[0], [[160 - 200 / 3, 120 - 200 / 3]]), "texture coordinate must be the pinhole pixel"
    assert assign(seen_fully(vertices, faces, normals, views, raycaster(vertices, faces)), 2, 1)[1] == [1], "the view budget keeps the views that win most"
    dark = np.full((240, 320, 3), 80, np.uint8)
    assert np.allclose(level(dark, np.array([[10., 10], [300, 200]]), np.full((2, 3), 120.))[0], 120, atol=1), "a photo darker than the fused colour is lifted to it"

    # the uncropped video frame: a 960x720 crop at x=160 was area-resized to the 640x480 clip raster
    record = {"raster_in_video_xywh": [160., 0., 960., 720.], "raster_wh": [640, 480], "width": 1280, "height": 720}
    kv = full_k([377.5, 377.5, 319.5, 239.5], record)
    point = np.array([[.3, -.2, 2.]])
    assert np.allclose(kv, [566.25, 566.25, 639.5, 359.5]) and np.allclose(
        project(point, np.eye(4), kv)[0], (project(point, np.eye(4), [377.5, 377.5, 319.5, 239.5])[0] + .5) * 1.5 - .5 + [160, 0]), "video pixel = clip pixel through the crop"
    usable = usable_pixels(None, 0, record, (648, 704))
    assert usable[647].all() and not usable[648:704].any() and usable[704].all(), "overlay rows never lend colour"
    with tempfile.TemporaryDirectory() as folder:
        mask = np.zeros((480, 640, 3), np.uint8); mask[200:220, 300:310] = 255
        cv2.imwrite(str(Path(folder) / "00007-0.png"), mask)
        moving = ~usable_pixels(Path(folder), 7, record, grow=3)
        assert moving[300:330, 610:625].all() and not moving[290:340, 400:590].any() and not moving[:, :160].any(), "a mask lands through the crop, grown a little"
    variance = np.full(61, 400.); variance[30] = 200.
    weight = sharpness_weight(variance)
    assert weight[30] == .25 and np.allclose(weight[:29], 1), "a frame half as sharp as its neighbours weighs a quarter"

    # effective resolution: nearer and frontal claims more; unusable pixels claim nothing
    scene = raycaster(vertices, faces)
    everywhere = np.ones((240, 320), bool)
    row_near = resolution_scores(vertices, faces, normals, near, k, size, scene, everywhere, .1)[0]
    row_far = resolution_scores(vertices, faces, normals, far, k, size, scene, everywhere, .1)[0]
    assert (row_near > 4 * row_far).all(), "area / distance: 2.5x nearer is ~15x the claim"
    hidden = everywhere.copy(); hidden[:, :160] = False
    row_masked = resolution_scores(vertices, faces, normals, near, k, size, scene, hidden, .1)[0]
    assert (row_masked[centre < -.1] == 0).all() and (row_masked[centre > .1] > 0).all(), "triangles on unusable pixels get no claim"
    own = np.full((240, 320), 3.); own[:, :160] = 1.5  # the view's own depth: something the mesh lacks stands in front of the left half
    row_occluded = resolution_scores(vertices, faces, normals, near, k, size, scene, everywhere, .1, (own, k))[0]
    assert (row_occluded[centre < -.1] == 0).all() and (row_occluded[centre > .1] > 0).all(), "a view whose own depth disagrees does not texture"
    unknown = np.zeros((240, 320))
    assert (resolution_scores(vertices, faces, normals, near, k, size, scene, everywhere, .1, (unknown, k))[0] > 0).all(), "unknown depth refuses nothing"

    # atlas: only the used pixels, packed without overlap; the rendered textured mesh reproduces the photo
    sizes = [(300, 200), (4000, 100), (100, 4000), (3000, 3000)]
    places, _ = shelf(sizes)
    boxes = [(x, y, x + w, y + h, s) for (s, x, y), (w, h) in zip(places, sizes)]
    assert all(x1 <= PAGE and y1 <= PAGE for _, _, x1, y1, _ in boxes) and not any(
        a[4] == b[4] and a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3] for i, a in enumerate(boxes) for b in boxes[i + 1:]), "packed boxes overlap or spill"
    v, u = np.mgrid[:240, :320]
    photo = np.stack([u * .7, v * .9, 128 + 60 * np.sin(u / 9.) * np.cos(v / 7.)], -1).astype(np.uint8)  # BGR, smooth enough for JPEG
    pixels = project(vertices, near, k)[0]
    for megapixels in (1., .02):  # at full resolution, and shrunk to fit a budget
        meshes, facts = photo_atlas(lambda wanted: ((i, photo) for i in sorted(wanted)), [(5, vertices, faces, pixels, np.ones(3))], "atlas", megapixels, 95, size)
        assert facts["texture_pixels"] < 320 * 240 and (megapixels == 1) == (facts["texture_scale"] == 1), facts
        textured = trimesh.Scene()
        for name, geometry in meshes:
            textured.add_geometry(geometry, geom_name=name)
        glb = textured.export(file_type="glb")
        image, found = Renderer(trimesh.load(io.BytesIO(glb), file_type="glb", process=False))(near, k, size)
        inside = found & (u > 100) & (u < 220) & (v > 60) & (v < 180)
        error = np.abs(image[inside] - photo[..., ::-1][inside]).mean()
        assert inside.sum() > 10000 and error < (1. if megapixels == 1 else 12), (megapixels, error)  # a half-texel slip already costs 1.3
    embedded = Image.open(io.BytesIO(glb[glb.find(b"\xff\xd8\xff"):]))
    assert list(embedded.quantization[0])[:4] == list(Image.open(io.BytesIO(cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())).quantization[0])[:4], \
        "the GLB must hold our JPEG, not trimesh's quality-75 re-encode"
    busy = cv2.resize(np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8), (320, 240), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    same = scores(busy, busy, np.ones((240, 320), bool))
    soft = scores(busy, cv2.GaussianBlur(busy, (0, 0), 2), np.ones((240, 320), bool))
    assert same["ssim"] > .999 and same["psnr"] > 100 and abs(same["detail"] - 1) < 1e-6 and soft["detail"] < .9 and soft["ssim"] < .99, (same, soft)
    print("texture check passed: nearest frontal view wins, occluded triangles refuse the photo, pinhole texture coordinates, view budget, exposure levelled to the fused colour; "
          "video K through the crop, overlay rows, grown moving masks and disagreeing own depth refused, sharpness against neighbours, resolution claim, atlas packing, "
          "rendered atlas reproduces the photo (full and budget-shrunk), GLB keeps the JPEG bytes, metric sanity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true")
    for name in ("droid-run", "fused", "output", "scene"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--texture-width", type=int, default=1280)
    parser.add_argument("--texture-megapixels", type=float, default=60, help="total texture pixels the browser has to hold; sets how many views may contribute (with --video: atlas pixels)")
    parser.add_argument("--video", type=Path, help="the clip's uncropped source-full.mp4 (source-full.json beside it): texture from its frames")
    parser.add_argument("--dynamic-masks", type=Path, help="SOURCEINDEX-*.png masks on the clip raster: those pixels (grown) never texture the room")
    parser.add_argument("--overlay-rows", type=lambda s: tuple(int(v) for v in s.split(":")), help="Y0:Y1 rows of the video frame under burned-in subtitles")
    parser.add_argument("--min-facing", type=float, default=MIN_FACING, help="with --video: cosine below which a view is too oblique")
    parser.add_argument("--nearly", type=float, default=.3, help="with --video: share of a triangle's best claim a covering view needs (fewer, larger patches below .5)")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="with --video: atlas page JPEG quality")
    parser.add_argument("--heldout", type=Path, nargs="+", help="textured GLBs to render into held-out video frames and score")
    parser.add_argument("--every", type=int, default=8, help="with --heldout: every n-th video frame")
    a = parser.parse_args()
    self_check() if a.self_check else heldout(a) if a.heldout else build(a)
