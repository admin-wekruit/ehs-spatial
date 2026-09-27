"""Whole-object models for a video report's important objects: the seen side from the video, the rest marked as inferred.

An object map shows each object only as the surface the camera saw, often just its front. For a few important objects
this asks the deployed RecGen generator (lucida-private-assets/generate_object, the route of
build_video_entity_model.py) for one complete mesh and checks it against the video:

  1. select: clearly named objects seen in >= 8 views, EHS equipment first, then support points, one per 3D place.
     VLM names can be wrong, so selection.jpg shows every best-view crop for review before any call; --entities
     ID=note (priority order) and --exclude ID=reason record that review in the manifest;
  2. best view: mask clear of the frame edges, the burnt-in caption band and the moving person; large, sharp (variance
     of Laplacian), frontal, and without an occluder's notch (solidity). An object whose every mask is cut by the 4:3
     crop's side is segmented again on the uncropped 16:9 frame (SAM 2.1, box prompt; kept only if whole and the same);
  3. input: the 1280x720 source frame cropped around the mask (+20% a side), the clip-frame mask carried onto it, and
     that view's posed DA3 depth resampled onto the crop grid (0 = no reliable depth; RecGen's contract, no imputation);
  4. place: the mesh comes back posed in that view's camera, so the view's camera-to-world puts it in
     droid_final_native_world; a trimmed Sim3 ICP against the object's observed points (masked depth of up to 16 views
     that agree with the best view, moving person removed) refines depth, scale and rotation but keeps the source-view
     position, and is dropped if it costs the source view more than 0.05 silhouette IoU;
  5. mark: a vertex within 2 voxels (0.028 native) of an observed point and in clear sight of a camera that saw the
     object is observed (alpha 255); every other vertex is the generator's guess (alpha 90).

A model is accepted only if its visible side matches the video: the source-view gate of build_lingbot_object_model
(silhouette IoU, depth error) plus the multi-view fit residual and the share of observed points it explains. A model
that fails is tried again from the next view (up to ATTEMPT_VIEWS, best first, each at least RETRY_GAP frames from the
others), then once more from the best view with EXTRA_SEED; the first accepted attempt ends the tries, else the one
with the highest silhouette IoU is kept. Every attempt (view, seed, metrics) is listed in validation.json.

Output (world coordinates, identity transform): models/<entity>/{model.glb (the generator's full mesh, COLOR_0 RGBA, no
material), anchor.json, validation.json} for import_video_scene.py --models; glb/<entity>.glb (a light copy with an
alphaMode BLEND material); review/<entity>.jpg; selection.jpg; manifest.json. Each call is journaled by the platform
transport; a received journal is read back (from the Modal volume) without a new call, and an unresolved dispatch is
never re-sent unless Modal shows it never got a container. No call starts unless the spend so far plus one worst-case
call stays under --max-usd. RecGen weights are non-commercial research only.

  python scripts/complete_video_objects.py --droid-run D --depth-run F --object-map M --masks MASKS \
      --dynamic-masks DM --clip CLIP_DIR --output NEW_DIR [--entities object-019=note ...] [--exclude object-031=reason ...] [--invoke]
  python scripts/complete_video_objects.py --self-check
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "modal_apps"), str(ROOT / "scripts")]

VOXEL = .014                 # native units; observed = within 2 voxels of a point the video saw
OCCLUSION = VOXEL / 2        # a ray that stops this much short of a vertex is blocked by the mesh itself
OBSERVED_ALPHA, INFERRED_ALPHA = 255, 90
FIT_TRIANGLES, LIGHT_TRIANGLES = 100_000, 50_000  # the generator returns 0.3-1.2 M; ICP runs on, and glb/ holds, a light copy
SEED, MIN_VIEWS, FIT_VIEWS, COUNT = 42, 8, 16, 10
ALL_MIN_VIEWS, ALL_MIN_EXTENT_M, ALL_STATUSES = 5, .10, ("clear", "partial")  # --all: every named physical object this size
PART_SHARE = .5              # --all: an entity with this share of its points on an accepted model is a part of it
REDISPATCH = 2               # re-sends of an input whose call Modal shows never got a container (it only waited for a GPU)
GPUS = ("A100-40GB", "A100-80GB")  # the transport pins A100-80GB; a re-send alternates (same sm_80 image, RecGen peaks at 14.3 GB)
IDLE_STOP = 4                # consecutive calls that never got a container: no more dispatches this run
RETRY_GAP = 30               # frames between any two views an object is generated from
ATTEMPT_VIEWS, EXTRA_SEED = 4, 43  # views tried per object, best first; then the best view once more with another seed
MIN_SOURCE_SIDE = 64         # source-frame pixels: the short side of a view's mask box (the generator's crop comes from that frame)
FIT_GATE = {"max_fit_median_native": VOXEL, "min_observed_coverage": .6, "scale_range": [.8, 1.25], "min_entity_coverage": .25,
            "min_agreeing_views": 3, "max_icp_rotation_deg": 15, "min_judged_points": 100,
            "judgedOn": "every other agreeing view (never the source view) is held out of the ICP and alone judges fit and coverage"}
RECGEN = {"modalApp": "lucida-private-assets", "modalFunction": "generate_object", "modalVolume": "panoptes-lucida-weights"}
GENERATOR = "RecGen (non-commercial research licence)"
SAM3D = {"modalApp": "panoptes-sam3d-objects-research", "modalClass": "SAM3DObjects", "model": "facebook/sam-3d-objects",
         "modelRevision": "2e73555018d2741ccd486e56c24fac41155a1dc6", "codeRevision": "f91db411c50efee93d8db7aeb323885650f6f722",
         "generator": "SAM 3D Objects (SAM License: commercial use allowed, no ITAR/military/nuclear uses)"}
BOX = "gravity-aligned box fitted to the video's own points (no learned model): kept only where the object is box-shaped enough to pass the same gate"
CYLINDER = "upright cylinder fitted to the video's own points (no learned model): kept only where the object is an upright cylinder enough to pass the same gate"
GENERATORS = {"recgen": GENERATOR, "sam3d": SAM3D["generator"], "box": BOX, "cylinder": CYLINDER}
SAM3D_USD_PER_SECOND, SAM3D_WORST_SECONDS = .000694 + 4 * .0000131 + 32 * .00000222, 900  # modal_apps/sam3d_research.py's container and timeout
# Modal list prices of the transport's container (A100-80GB + 8 cores + 64 GiB), per second; its client deadline is 240 s.
USD_PER_SECOND, WORST_CALL_SECONDS = .000694 + 8 * .0000131 + 64 * .00000222, 240
EXCLUDED = ("floor", "wall", "ceiling", "unnamed surface", "light", "lamp", "fixture", "man", "woman", "person", "people", "worker", "human")
RELEVANT = {"machine": ("machine", "mill", "lathe", "drill", "press", "saw", "grinder", "cnc", "motor", "chuck", "guard"),
            "control panel": ("panel", "controller", "control", "switch"),
            "cabinet or rack": ("cabinet", "drawer", "locker", "shelf", "rack"),
            "bin or box": ("bin", "box", "case", "container", "tub", "crate", "tray"),
            "cart": ("cart", "trolley"), "bench": ("bench", "workbench", "table", "tabletop"),
            "vise": ("vise", "clamp"), "monitor": ("monitor", "computer")}
BOX_VIEWS = False            # --generator box or cylinder sets it: views are not generator crops, so edge cuts and size do not disqualify them
BORDER = 4                   # clip-frame pixels a best-view mask keeps clear of the frame edge


def frame_set(spans):
    """'14-225' style inclusive ranges (a single number is one frame) -> frozenset of frame indices."""
    return frozenset(f for span in spans for a, b in [map(int, (span + "-" + span).split("-")[:2])] for f in range(a, b + 1))


def matches(label, terms):
    """First term naming the label: a phrase anywhere, a single word only as a whole word."""
    words = label.lower().replace("-", " ").split()
    return next((t for t in terms if (t in label.lower() if " " in t else t in words)), None)


def candidates(entities, statuses=("clear",), min_views=MIN_VIEWS, metres=None, skip=frozenset()):
    """Named objects seen in enough views (and, given `metres`, at least ALL_MIN_EXTENT_M across), EHS equipment first,
    then support points; with the rule each passed."""
    ranked = []
    for e in entities:
        views, group = len(set(e["sourceFrames"]) - skip), next((g for g, terms in RELEVANT.items() if matches(e["label"], terms)), None)
        extent = float(np.ptp(np.array(e["boundsNative"]), 0).max()) * metres if metres else None
        if (e.get("labelStatus") in statuses and not matches(e["label"], EXCLUDED) and views >= min_views
                and (extent is None or extent >= ALL_MIN_EXTENT_M)):
            ranked.append((group is None, -e["supportPoints"], e["entityId"], e, group, views, extent))
    return [(e, f"'{e['label']}' ({group or 'not listed as EHS equipment'}): name {e.get('labelStatus')}, {views} views, "
                f"{e['supportPoints']} support points" + (f", {extent:.2f} m across" if extent else ""))
            for *_, e, group, views, extent in sorted(ranked, key=lambda r: r[:3])]


def box_iou(box, other):
    """3D IoU of two [[lo], [hi]] boxes: a high value is one physical object entered twice (a thing resting on another is low)."""
    lo, hi = np.maximum(box[0], other[0]), np.minimum(box[1], other[1])
    shared = np.prod(np.clip(hi - lo, 0, None))
    volume = lambda b: np.prod(np.clip(np.subtract(b[1], b[0]), 0, None))
    return float(shared / max(volume(box) + volume(other) - shared, 1e-12))


def subtitle_box(rgb, band=420, white=200, minimum=150, pad=6):
    """Clip-frame box [x0, y0, x1, y1] of burnt-in caption text (white glyphs in the bottom band), or None.

    Captions are the sharpest thing in a frame, so a mask of caption text wins any sharpness contest; and caption text
    over an object is not the object's texture.
    """
    ys, xs = np.nonzero(rgb[band:].min(2) >= white)
    if len(xs) < minimum:
        return None
    return [int(xs.min()) - pad, int(ys.min()) + band - pad, int(xs.max()) + pad, int(ys.max()) + band + pad]


class Clip:
    """The clip's rasters of one camera: 640x480 clip frame (masks), prepared depth raster, 1280x720 source frame."""

    def __init__(self, droid_run, clip_dir):
        import mono_room
        mono_room.use_clip(droid_run)
        self.mono_room = mono_room
        fx, fy, cx, cy = mono_room.SOURCE_K
        video = json.loads((clip_dir / "source-full.json").read_text())
        x0, y0, width, height = video["raster_in_video_xywh"]
        sx, sy = width / video["raster_wh"][0], height / video["raster_wh"][1]
        # pixel centres: a clip pixel x covers source pixels sx*x .. sx*x+sx-1, so its centre is sx*x + (sx-1)/2 + x0
        self.clip_to_full = np.array([[sx, 0, (sx - 1) / 2 + x0], [0, sy, (sy - 1) / 2 + y0], [0, 0, 1.]])
        self.k_full = self.clip_to_full @ np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        k = mono_room.prepare_image(np.zeros((480, 640, 3), np.uint8), mono_room.CALIBRATION, 2)[1]
        self.k_raster = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.]])
        self.full_to_raster = self.k_raster @ np.linalg.inv(self.k_full)  # one pinhole, two rasters, no distortion
        self.video, self.full_size, self.clip_size = clip_dir / video["video"], (video["width"], video["height"]), tuple(video["raster_wh"])
        self.clip_frames = [clip_dir / f["relative_path"] for f in json.loads((droid_run / "input-manifest.json").read_text())["frames"]]

    def full_mask(self, clip_mask):
        """Clip-frame mask -> source-frame mask: bilinear at source pixel centres, half coverage counts."""
        v, u = np.indices(self.full_size[::-1], dtype=np.float64)
        back = np.linalg.inv(self.clip_to_full)
        mx, my = (back[0, 0] * u + back[0, 2]).astype(np.float32), (back[1, 1] * v + back[1, 2]).astype(np.float32)
        return cv2.remap(clip_mask.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=0) >= .5

    def clip_mask(self, full_mask):
        """Source-frame mask -> clip-frame mask (bilinear at clip pixel centres, half coverage counts)."""
        v, u = np.indices(self.clip_size[::-1], dtype=np.float64)
        mx = (self.clip_to_full[0, 0] * u + self.clip_to_full[0, 2]).astype(np.float32)
        my = (self.clip_to_full[1, 1] * v + self.clip_to_full[1, 2]).astype(np.float32)
        return cv2.remap(full_mask.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=0) >= .5

    def raster_mask(self, path):
        return self.mono_room.prepare_image(cv2.imread(str(path), cv2.IMREAD_COLOR), self.mono_room.CALIBRATION, 2)[0][..., 0] > 0

    def frames(self, wanted):
        """(index, RGB source frame) for the wanted indices, in order; MP4 frame i is clip frame i. Decoded, never all held."""
        capture = cv2.VideoCapture(str(self.video))
        for index in range(max(wanted, default=-1) + 1):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"source video ends before frame {index}")
            if index in wanted:
                yield index, np.ascontiguousarray(bgr[..., ::-1])


def lift(depth, k, c2w, pixels=None):
    """World points of raster depth (all positive pixels, or the given boolean selection)."""
    v, u = np.nonzero((depth > 0) if pixels is None else pixels & (depth > 0))
    z = depth[v, u].astype(np.float64)
    local = np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z], 1)
    return local @ c2w[:3, :3].T + c2w[:3, 3]


def facing(depth, k):
    """Per pixel |cos| between the depth surface normal and the viewing ray (1 = seen face-on; NaN without depth)."""
    v, u = np.indices(depth.shape)
    points = np.stack([(u - k[0, 2]) / k[0, 0] * depth, (v - k[1, 2]) / k[1, 1] * depth, depth], -1)
    normal = np.cross(points[1:-1, 2:] - points[1:-1, :-2], points[2:, 1:-1] - points[:-2, 1:-1])
    inner = (depth[1:-1, 1:-1] > 0) & (depth[1:-1, 2:] > 0) & (depth[1:-1, :-2] > 0) & (depth[2:, 1:-1] > 0) & (depth[:-2, 1:-1] > 0)
    p = points[1:-1, 1:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.abs(np.sum(normal * p, -1)) / (np.linalg.norm(normal, axis=-1) * np.linalg.norm(p, axis=-1))
    return np.pad(np.where(inner, cos, np.nan), 1, constant_values=np.nan)


def main_parts(mask, share=.1):
    """The mask without specks: its connected parts of at least `share` of the largest (an occluder may split an object in two)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return np.isin(labels, 1 + np.flatnonzero(areas >= share * areas.max())) if count > 2 else mask


def mask_path(masks, observation):
    label, frame, instance = observation.rsplit(":", 2)
    found = sorted(masks.glob(f"{label}-*/frame-{int(frame):05d}/instance-{instance}-mask.png"))
    return found[0] if len(found) == 1 else None


def reliable(row, clip, dynamic):
    """The view's reliable posed depth with the moving person (dilated) removed, and that person's dilated mask."""
    mono_room = clip.mono_room
    depth = np.where(mono_room.unreliable(row["mono"], None, None, .03), 0, row["scale"] * row["mono"]).astype(np.float32)
    moving = cv2.dilate(mono_room.moving_mask(dynamic, row["source_index"]).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return np.where(moving, 0, depth).astype(np.float32), moving


def view_metrics(entity, rows, clip, args, cache):
    """Every observation of the entity with depth: hard filters and the geometric parts of the score."""
    out = []
    for observation in entity["observations"]:
        frame = int(observation.rsplit(":", 2)[1])
        path = mask_path(args.masks, observation)
        if frame not in rows or path is None:
            continue
        if frame not in cache:
            cache.clear()  # observations come in frame order; one view is held at a time
            depth, moving = reliable(rows[frame], clip, args.dynamic_masks)
            cache[frame] = depth, moving, facing(depth, clip.k_raster), None if args.no_captions else subtitle_box(cv2.imread(str(clip.clip_frames[frame])))
        depth, moving, cos, caption = cache[frame]
        clip_mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
        raster = clip.raster_mask(path)
        ys, xs = np.nonzero(clip_mask)
        if not len(xs) or not raster.any():
            continue
        ry, rx = np.nonzero(raster)
        py, px = np.nonzero(main_parts(clip_mask))  # the part build_input carries onto the source frame
        out.append({"observation": observation, "frame": frame, "mask": str(path), "area": int(raster.sum()),
                    "clipBox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "sourceShortSide": float(min((np.ptp(px) + 1) * clip.clip_to_full[0, 0], (np.ptp(py) + 1) * clip.clip_to_full[1, 1])),
                    "solidity": float(raster.sum() / max(cv2.contourArea(cv2.convexHull(np.column_stack([rx, ry]).astype(np.int32))), 1)),
                    "cut": [side for side, hit in (("left", xs.min() < BORDER), ("right", xs.max() >= 640 - BORDER), ("top", ys.min() < BORDER),
                                                   ("bottom", ys.max() >= 480 - BORDER), ("raster edge", rx.min() < 2 or ry.min() < 2 or rx.max() >= 638 or ry.max() >= 478)) if hit],
                    "depthShare": float((depth[raster] > 0).mean()), "personShare": float(moving[raster].mean()),
                    "captionShare": float(clip_mask[caption[1]:caption[3] + 1, max(caption[0], 0):caption[2] + 1].sum() / clip_mask.sum()) if caption else 0.,
                    "frontal": float(np.nanmedian(cos[raster])) if np.isfinite(cos[raster]).sum() >= 20 else 0.})
    return out


def failing(m):
    """The filters a view fails. 'side' is the 4:3 crop's left/right edge: the uncropped 16:9 frame may hold the rest;
    the source frame is cut at 'top/bottom' as well. 'small': under 1500 raster px and, on the source frame the
    generator's crop is cut from, a mask box under MIN_SOURCE_SIDE px on its short side (either size is enough)."""
    cut = set(m["cut"])
    if BOX_VIEWS:  # the box is built from every agreeing view's points, not from this view's crop: only what spoils the view as a judge counts
        return {name for name, bad in (("depth", m["depthShare"] < .5), ("person", m["personShare"] >= .02), ("caption", m["captionShare"] >= .02)) if bad}
    return {name for name, bad in (("side", cut & {"left", "right"}), ("top/bottom", cut & {"top", "bottom"}), ("raster edge", cut == {"raster edge"}),
                                   ("depth", m["depthShare"] < .5), ("person", m["personShare"] >= .02), ("caption", m["captionShare"] >= .02),
                                   ("small", m["area"] < 1500 and m["sourceShortSide"] < MIN_SOURCE_SIDE)) if bad}


def usable(metrics):
    """Views that can show the whole object: mask clear of the frame edges, the caption band and the person, mostly reliable depth, not small."""
    return [m for m in metrics if not failing(m)]


def spread(views, count=ATTEMPT_VIEWS, gap=RETRY_GAP):
    """Up to `count` views, highest score first, each at least `gap` frames from every view taken before it."""
    taken = []
    for m in sorted(views, key=lambda m: -m["score"]):
        if len(taken) < count and all(abs(m["frame"] - t["frame"]) >= gap for t in taken):
            taken.append(m)
    return taken


def pick_views(views, clip):
    """Per entity, the views to generate from (spread): score = area x relative sharpness (variance of Laplacian) x
    frontality x solidity^2; the first is the best view."""
    by_frame = {}
    for ms in views.values():
        for m in ms:
            by_frame.setdefault(m["frame"], []).append(m)
    for index, rgb in clip.frames(set(by_frame)):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        for m in by_frame[index]:
            (x0, y0), (x1, y1) = (np.reshape(m["clipBox"], (2, 2)) * np.diag(clip.clip_to_full)[:2] + clip.clip_to_full[:2, 2]).round().astype(int)
            m["sharpness"] = float(cv2.Laplacian(gray[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1], cv2.CV_64F).var())
    tries = {}
    for eid, ms in views.items():
        top = max(m["sharpness"] for m in ms)
        for m in ms:
            m["score"] = m["area"] * m["sharpness"] / max(top, 1e-9) * max(m["frontal"], .1) * min(m["solidity"], 1) ** 2  # a notch is usually an occluder
        tries[eid] = spread(ms)
    return tries


def build_input(entity, view, row, frame_rgb, clip, args, video_sha):
    """RecGen payload for one view: source-frame crop, carried mask, posed depth on the crop grid, crop K."""
    from ehs_spatial.platform.recgen import RecGenRequest, source_grid_crop
    from reconstruct_room_rgb import digest
    depth, _ = reliable(row, clip, args.dynamic_masks)
    full = (main_parts(cv2.imread(view["fullMask"], cv2.IMREAD_GRAYSCALE) > 0) if view.get("fullMask")
            else clip.full_mask(main_parts(cv2.imread(view["mask"], cv2.IMREAD_GRAYSCALE) > 0)))
    crop = source_grid_crop(frame_rgb, full, depth, clip.k_raster, clip.full_to_raster)
    # source_grid_crop keeps mask pixels only where the depth raster reaches; a re-segmented object continues past it
    left, top, right, bottom = crop["pixelMapping"]["sourceCropXYXY"]
    yy, xx = np.mgrid[top:bottom, left:right]
    inside = (xx >= 0) & (yy >= 0) & (xx < full.shape[1]) & (yy < full.shape[0])
    crop["mask"] = np.zeros(inside.shape, bool)
    crop["mask"][inside] = full[yy[inside], xx[inside]]
    payload = {"entityId": entity["entityId"], "anchorObservationId": view["observation"], "seed": SEED, "views": [{
        "observationId": view["observation"], "observationRevision": 1, "imageId": f"{video_sha}:{view['frame']}",
        "imageSha256": hashlib.sha256(frame_rgb.tobytes()).hexdigest(), "maskSha256": digest(view.get("fullMask") or view["mask"]),
        "geometrySolutionSha256": digest(args.depth_run / "mono" / f"{view['frame']:05d}.npz"),
        "coordinateFrameId": "droid_final_native_world", "cameraToWorld": row["c2w"], **crop}]}
    RecGenRequest.from_payload(payload)
    if args.generator == "sam3d":  # SAM 3D takes the whole frame: it crops around the mask itself, and the intrinsics it infers
        mapping = np.asarray(clip.full_to_raster, float)  # from the pointmap assume a centred principal point, which a crop is not
        v, u = np.indices(full.shape, dtype=np.float64)  # the affine map element-wise: a BLAS matmul here segfaulted once Modal was up
        cx = np.floor(mapping[0, 0] * u + mapping[0, 1] * v + mapping[0, 2] + .5).astype(int)
        cy = np.floor(mapping[1, 0] * u + mapping[1, 1] * v + mapping[1, 2] + .5).astype(int)
        inside = (cx >= 0) & (cy >= 0) & (cx < depth.shape[1]) & (cy < depth.shape[0])
        full_depth = np.zeros(full.shape, np.float32)
        full_depth[inside] = depth[cy[inside], cx[inside]]
        payload["views"][0].update(fullRgb=frame_rgb, fullMask=full, fullDepth=full_depth, fullK=np.linalg.inv(mapping) @ clip.k_raster)
    return payload


def agreement(points, depth, mask, k, c2w):
    """How another view's points sit in the anchor view: share landing inside its mask, median |relative depth difference| there."""
    local = (points - c2w[:3, 3]) @ c2w[:3, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = np.round(local[:, 0] / local[:, 2] * k[0, 0] + k[0, 2]), np.round(local[:, 1] / local[:, 2] * k[1, 1] + k[1, 2])
    ok = (local[:, 2] > 0) & (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
    u, v, z = u[ok].astype(int), v[ok].astype(int), local[ok, 2]
    hit = mask[v, u] & (depth[v, u] > 0)
    return float(hit.sum() / max(len(points), 1)), float(np.median(np.abs(z[hit] / depth[v[hit], u[hit]] - 1))) if hit.sum() >= 100 else np.inf


def observed_points(entity, view, rows, clip, args, inside=.6, relative=.05):
    """World points the video saw of the entity, from the views that agree with the best view, and those views.

    Camera drift over the clip (or a look-alike merged into the entity) can put another view's points of the same object
    a tenth of the range away; fitting to them would pull the model off the view it was generated from. So a view counts
    only if its points land mostly inside the best view's mask at the depth seen there. That keeps views of the same
    side; sides only other views saw cannot be checked and are left out. The points are cut to the cells at least two of
    the entity's usable views agree on (a mask that spills past the object lands on different background in each view);
    those agreed cells are returned too, as the entity's extent for entity_coverage.
    """
    from build_video_object_map import consensus, pack
    anchor = rows[view["frame"]]
    depth, _ = reliable(anchor, clip, args.dynamic_masks)
    mask = main_parts(clip.raster_mask(Path(view["mask"])))
    cell = float(entity["cells"]["cellNative"]) if entity.get("cells") else 2 * VOXEL
    kept, checked, occupied = {}, 0, []
    for observation in entity["observations"]:
        frame = int(observation.rsplit(":", 2)[1])
        path = mask_path(args.masks, observation)
        if frame not in rows or path is None:
            continue
        own, _ = reliable(rows[frame], clip, args.dynamic_masks)
        points = lift(own, clip.k_raster, rows[frame]["c2w"], clip.raster_mask(path))
        share, error = agreement(points, depth, mask, clip.k_raster, anchor["c2w"])
        checked += 1
        occupied.append(np.unique(np.floor(points / cell).astype(np.int64), axis=0))
        if observation == view["observation"] or share >= inside and error <= relative:
            kept[observation] = points
    names = sorted(kept, key=lambda o: int(o.rsplit(":", 2)[1]))
    names = sorted({names[i] for i in np.linspace(0, len(names) - 1, min(len(names), FIT_VIEWS)).round().astype(int)} | {view["observation"]})
    points = np.concatenate([kept[o] for o in names])
    owner = np.concatenate([np.full(len(kept[o]), i) for i, o in enumerate(names)])  # which of `names` each point came from
    agreed = consensus([(c + .5) * cell for c in occupied], cell)[1] if len(occupied) > 1 else occupied[0]
    voted = np.isin(pack(np.floor(points / cell).astype(np.int64)), pack(agreed))
    if voted.sum() >= 100:  # too few votes to cut by: keep what the views saw
        points, owner = points[voted], owner[voted]
    first = np.unique(np.floor(points / (VOXEL / 4)).astype(np.int64), axis=0, return_index=True)[1]
    return points[first], names, (agreed + .5) * cell, cell, owner[first], {"viewsChecked": checked, "viewsAgreeing": len(kept), "fitViews": names,
                           "agreementRule": f">= {inside:.0%} of a view's points inside the best view's mask, median |relative depth difference| <= {relative}"}


def ray_scene(vertices, faces):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)), o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return scene


def closest(vertices, faces, points):
    import open3d as o3d
    found = ray_scene(vertices, faces).compute_closest_points(o3d.core.Tensor(np.asarray(points, np.float32)))["points"].numpy()
    return found.astype(np.float64), np.linalg.norm(found - points, axis=1)


def fit(vertices, faces, observed, eye=None, iterations=100, keep=.8):
    """Trimmed Sim3 ICP moving the mesh onto the observed points.

    Correspondences run from each observed point to its closest point on the mesh, so the sides the generator added
    and the video never saw pull nothing. A step that would take the accumulated scale out of FIT_GATE's range is
    rigid instead. Depth points on a flat front do not pin sliding across it, so with `eye` (the source camera centre)
    the mesh centroid stays on that camera's ray through where it started: depth, scale and rotation are refined, the
    source-view position is kept. Returns the 4x4 applied and the moved vertices.
    """
    from ehs_spatial.platform.contracts import PlatformError
    from ehs_spatial.platform.spatial import similarity_transform
    total, current = np.eye(4), np.asarray(vertices, np.float64)
    low, high = FIT_GATE["scale_range"]
    ray = None if eye is None else (current.mean(0) - eye) / np.linalg.norm(current.mean(0) - eye)
    for _ in range(iterations):
        target, distance = closest(current, faces, observed)
        inliers = distance <= np.quantile(distance, keep)
        try:
            step = similarity_transform(target[inliers], observed[inliers])
        except PlatformError:  # the closest mesh points collapse onto a line or a point: no further step, the gate judges the model as it stands
            break
        scale = np.cbrt(np.linalg.det(step[:3, :3]))
        if not low <= scale * np.cbrt(np.linalg.det(total[:3, :3])) <= high:  # rigid: same rotation, centroids matched
            step[:3, :3] /= scale
            step[:3, 3] = observed[inliers].mean(0) - step[:3, :3] @ target[inliers].mean(0)
        if ray is not None:
            centre = current.mean(0) @ step[:3, :3].T + step[:3, 3]
            step[:3, 3] += eye + ray * ((centre - eye) @ ray) - centre
        current, total = current @ step[:3, :3].T + step[:3, 3], step @ total
        if np.abs(step - np.eye(4)).max() < 1e-6:
            break
    return total, current


def seen(vertices, faces, observed, cameras, k, size=(640, 480), radius=None):
    """Observed vertices: an observed point within `radius`, facing a camera that saw the object and in its clear sight.

    Facing matters: the back of a thin part and the inner wall of a shell lie within `radius` of the front's points and in
    sight lines that stop only OCCLUSION short, so without it they counted as seen.
    """
    import open3d as o3d
    import trimesh
    from scipy.spatial import cKDTree
    radius = 2 * VOXEL if radius is None else radius  # VOXEL is set per clip in main()
    near = cKDTree(observed).query(vertices, distance_upper_bound=radius)[0] < np.inf
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    normals = mesh.vertex_normals * (1. if mesh.volume >= 0 else -1.)  # outward, whichever way the generator wound its faces
    scene, clear = ray_scene(vertices, faces), np.zeros(len(vertices), bool)
    for c2w in cameras:
        local = (vertices - c2w[:3, 3]) @ c2w[:3, :3]
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = local[:, 0] / local[:, 2] * k[0, 0] + k[0, 2], local[:, 1] / local[:, 2] * k[1, 1] + k[1, 2]
        facing = ((c2w[:3, 3] - vertices) * normals).sum(1) > 0
        todo = near & ~clear & facing & (local[:, 2] > 0) & (u >= 0) & (u < size[0]) & (v >= 0) & (v < size[1])
        if todo.any():
            direction = vertices[todo] - c2w[:3, 3]
            distance = np.linalg.norm(direction, axis=1)
            rays = np.hstack([np.broadcast_to(c2w[:3, 3], direction.shape), direction / distance[:, None]]).astype(np.float32)
            hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            clear[np.flatnonzero(todo)[hit >= distance - OCCLUSION]] = True
    return near & clear


def decimate(vertices, faces, colors, target):
    """Quadric decimation that keeps vertex colours (float RGB 0..1)."""
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(np.asarray(faces, np.int32)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    if len(faces) > target:
        mesh = mesh.simplify_quadric_decimation(target)
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)


def glb(vertices, faces, rgba, blend):
    """One-buffer GLB with COLOR_0 RGBA; `blend` adds one double-sided material with alphaMode BLEND."""
    import trimesh
    data = trimesh.Trimesh(vertices, faces, vertex_colors=rgba, process=False).export(file_type="glb")
    if not blend:
        return data
    length = struct.unpack_from("<I", data, 12)[0]
    spec = json.loads(data[20:20 + length])
    spec["materials"] = [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1},
                          "alphaMode": "BLEND", "doubleSided": True}]
    for mesh in spec["meshes"]:
        for primitive in mesh["primitives"]:
            primitive["material"] = 0
    text = json.dumps(spec, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text + data[20 + length:]
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def render(vertices, faces, rgb, observed, K, c2w, size, background):
    """Raycast the mesh into a pinhole view over `background`: shaded model colour, inferred faces tinted magenta."""
    import open3d as o3d
    scene = ray_scene(vertices, faces)
    hit = scene.cast_rays(scene.create_rays_pinhole(o3d.core.Tensor(K), o3d.core.Tensor(np.linalg.inv(c2w)), size[0], size[1]))
    ids = hit["primitive_ids"].numpy().astype(np.int64)
    inside = ids != scene.INVALID_ID
    face = ids[inside]
    colour = rgb[faces[face]].mean(1)
    shade = .35 + .65 * np.abs(hit["primitive_normals"].numpy()[inside] @ (c2w[:3, 2]))[:, None]
    colour = colour * shade
    guessed = ~observed[faces[face]].all(1)
    colour[guessed] = .45 * colour[guessed] + .55 * np.array([255, 40, 220])
    image = background.astype(np.float64).copy()
    image[inside] = colour
    return image.astype(np.uint8), inside


def look_at(eye, target, up):
    """OpenCV camera-to-world looking from eye at target, image y along -up."""
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(-up, z)
    x /= np.linalg.norm(x)
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = np.stack([x, np.cross(z, x), z], 1), eye
    return c2w


def review(path, crop, vertices, faces, rgb, observed, c2w, up, caption):
    """best-view crop | model in that view over the dimmed crop, mask outline yellow | model from the far side."""
    size = crop["rgb"].shape[1], crop["rgb"].shape[0]
    K = np.asarray(crop["K"], float)
    placed, _ = render(vertices, faces, rgb, observed, K, c2w, size, crop["rgb"] * .4)
    cv2.drawContours(placed, cv2.findContours(crop["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 230, 0), 2)
    centre = vertices.mean(0)
    arm = c2w[:3, 3] - centre
    arm -= 2 * (arm - (arm @ up) * up)  # half a turn about the vertical through the object's middle
    centred = K.copy()
    centred[:2, 2] = np.array(size) / 2  # the crop's own principal point can lie outside the crop; a look-at camera aims at the middle
    back, _ = render(vertices, faces, rgb, observed, centred, look_at(centre + arm, centre, up), size, np.full(crop["rgb"].shape, 90))
    panels = [cv2.resize(np.ascontiguousarray(p), (480, 480), interpolation=cv2.INTER_AREA) for p in (crop["rgb"], placed, back)]
    sheet = np.vstack([np.full((56, 1440, 3), 255, np.uint8), np.hstack(panels)])
    for row, text in enumerate(caption):
        cv2.putText(sheet, text, (8, 22 + 24 * row), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])


def spent(journal):
    """(provider GPU seconds, local wall seconds, estimated USD) of every dispatch journaled so far, failures included.

    Local wall time (dispatch to result, queueing included) bounds the billed container time from above; a dispatch
    without a recorded end counts as a worst-case call.
    """
    gpu = wall = 0.
    for state_path in journal.rglob("dispatch.json"):
        state = json.loads(state_path.read_text())
        wall += state.get("wallSeconds", 0 if state.get("status") == "preparing" else WORST_CALL_SECONDS)
        record = state_path.parent / "output.json"
        gpu += (json.loads(record.read_text()).get("gpu_function_seconds") or 0) if record.exists() else 0
    return gpu, wall, wall * USD_PER_SECOND


def modal_state(state_path):
    """Modal's own record of a journaled call: its status and whether it ever got a container (task)."""
    import modal
    try:
        nodes = modal.FunctionCall.from_id(json.loads(state_path.read_text())["providerRequestId"]).get_call_graph()
        return {"status": [n.status.name for n in nodes], "gotContainer": any(n.task_id for n in nodes)}
    except Exception as error:
        return {"error": type(error).__name__}


def never_started(states):
    """True when Modal shows every given dispatch still queued without a container: nothing ran, nothing was billed."""
    import modal
    from modal.call_graph import InputStatus
    try:
        ids = [json.loads(s.read_text())["providerRequestId"] for s in states]
        nodes = [n for i in ids for n in modal.FunctionCall.from_id(i).get_call_graph()]
    except Exception:  # an unknown outcome is not a call that never ran
        return False
    return bool(nodes) and all(n.status == InputStatus.PENDING and not n.task_id for n in nodes)


def use_gpu(gpu):
    """Send the transport's calls to `gpu` instead of its pinned A100-80GB: same deployed function, image, code, weights.

    Only this process's modal.Function.with_options is wrapped; recgen_transport.py is untouched.
    """
    import modal
    original = getattr(modal.Function.with_options, "original", modal.Function.with_options)

    def with_options(self, *args, **kwargs):
        if kwargs.get("gpu") == "A100-80GB":
            kwargs["gpu"] = gpu
        return original(self, *args, **kwargs)
    with_options.original = original
    modal.Function.with_options = with_options


def call_identity(payload, function_id):
    """The journal directory name recgen_transport.invoke gives this input (same digest, same fields; checked after each new call)."""
    from ehs_spatial.platform.contracts import digest
    from ehs_spatial.platform.recgen import RECGEN_PINS, RecGenRequest
    parsed, protocol = RecGenRequest.from_payload(payload), payload.get("_researchProtocol", {})
    return digest({"payloadSha256": hashlib.sha256(parsed.to_npz()).hexdigest(), "entityId": parsed.entity_id, "seed": parsed.seed,
                   "pins": RECGEN_PINS, "runtime": protocol.get("runtimeManifest"), "functionId": function_id,
                   **({"dispatchAttemptId": protocol["dispatchAttemptId"]} if "dispatchAttemptId" in protocol else {})})


def journal_folder(journal, payload, frame, function_id):
    """Where one input's calls are journaled: its entity's folder if that already holds one (earlier runs journaled an
    object's first view there), else view-<frame> under it, suffixed -seed-<seed> off the default seed."""
    entity = journal / payload["entityId"]
    if (entity / call_identity(payload, function_id)).exists():
        return entity
    return entity / (f"view-{frame:05d}" + ("" if payload["seed"] == SEED else f"-seed-{payload['seed']}"))


def obtain(payload, base, args):
    """The generator's mesh for one input: read back from its journal once received, else one new call within the cap.

    Only this input's own journal entries count (a folder may hold other inputs' calls), and one received under any
    dispatch attempt is read back. An unresolved dispatch is never sent again, unless Modal shows it never got a
    container (it waited for a GPU past the transport's deadline); then the same input goes out under a new dispatch
    attempt id, at most REDISPATCH times, on the other GPU type. After IDLE_STOP such calls in a row nothing more is
    dispatched. Returns (mesh or None, record or None).
    """
    tries = [(base, payload)] + [(base / f"redispatch-{n}", {**payload, "_researchProtocol": {"dispatchAttemptId": f"redispatch-{n}"}})
                                 for n in range(1, REDISPATCH + 1)]
    roots = [folder / call_identity(attempt, args.function_id) for folder, attempt in tries]
    for (folder, attempt), root in zip(tries, roots):
        if (root / "manifest.json").exists():
            return request(attempt, folder, args.function_id)  # already paid: read back, no new call
    n = 0
    while n <= REDISPATCH:
        (folder, attempt), state = tries[n], roots[n] / "dispatch.json"
        if state.exists():
            if not never_started([state]):
                return None, {"error": "unresolved dispatch that may have run: never sent again"}
            n += 1
            continue
        if not args.invoke:
            return None, None
        if args.idle >= IDLE_STOP:
            return None, {"error": f"not requested: the last {args.idle} calls never got a GPU container"}
        usd = spent(args.output / "journal")[2]
        if usd + WORST_CALL_SECONDS * USD_PER_SECOND > args.max_usd:
            return None, {"error": f"not requested: {usd:.2f} USD spent, one more worst-case call could pass the cap"}
        use_gpu(args.gpu)
        folder.mkdir(parents=True, exist_ok=True)
        save(folder / "requested-gpu.json", {"gpu": args.gpu, "dispatchAttemptId": f"redispatch-{n}" if n else None})
        mesh, record = request(attempt, folder, args.function_id)
        if not state.exists():  # the transport named the journal differently: stop before a rerun could pay for it again
            raise RuntimeError(f"call_identity no longer matches recgen_transport: {folder} holds a call this script cannot find")
        print(json.dumps({"entity": payload["entityId"], "view": payload["anchorObservationId"], "seed": payload["seed"], "dispatch": n, "gpu": args.gpu,
                          "status": "received" if mesh is not None else record.get("error") or "provider error",
                          **(record.get("telemetry") or {})}, default=str), flush=True)
        if mesh is not None or record.get("error") != "provider_outcome_unknown":
            args.idle = 0
            return mesh, record
        if never_started([state]):
            args.idle += 1
            args.gpu = GPUS[(GPUS.index(args.gpu) + 1) % len(GPUS)]
    return None, {"error": f"no GPU container after {REDISPATCH + 1} dispatches"}


def sam3d_pointmap(depth, k):
    """This view's own depth on the crop grid as SAM 3D's pointmap: PyTorch3D camera (OpenCV x and y negated), NaN = no depth."""
    v, u = np.indices(depth.shape, dtype=np.float64)
    z = np.where(depth > 0, depth, np.nan)
    return np.stack([-(u - k[0, 2]) / k[0, 0] * z, -(v - k[1, 2]) / k[1, 1] * z, z], -1).astype(np.float32)


def sam3d_to_world(vertices, object_to_camera, c2w):
    """Mesh vertices from SAM 3D's object frame to the scene: its pose puts them in the PyTorch3D camera of the pointmap."""
    camera = (np.c_[vertices, np.ones(len(vertices))] @ np.asarray(object_to_camera).T)[:, :3] * [-1, -1, 1]  # -> OpenCV camera
    return camera @ np.asarray(c2w)[:3, :3].T + np.asarray(c2w)[:3, 3]


def box_mesh(points, up, step):
    """Gravity-aligned box around `points`: yaw of the smallest rectangle on the floor plane, extents at the 1st/99th
    percentiles, at least one `step` thick (a face seen straight on), faces cut to edges <= `step` so each vertex can
    carry its own colour. Returns world vertices and faces."""
    import trimesh
    up = np.asarray(up, float) / np.linalg.norm(up)
    a = np.cross(up, [1., 0, 0] if abs(up[0]) < .9 else [0, 1., 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    angle = np.radians(cv2.minAreaRect(np.c_[points @ a, points @ b].astype(np.float32))[2])
    x = np.cos(angle) * a + np.sin(angle) * b
    axes = np.stack([x, np.cross(up, x), up])  # rows, right-handed
    local = points @ axes.T
    lo, hi = np.percentile(local, 1, 0), np.percentile(local, 99, 0)
    hi = np.maximum(hi, lo + step)
    unit = trimesh.creation.box(bounds=[lo, hi])
    vertices, faces = trimesh.remesh.subdivide_to_size(unit.vertices, unit.faces, step)
    return vertices @ axes, faces


def cylinder_mesh(points, up, step, trials=500, sample=4000):
    """Upright (gravity-aligned) cylinder around `points`, or None when their floor-plane projection is not an arc.

    Circle on the floor plane by RANSAC (3-point circles, most points within `step` of the circle; seeded, so the same
    points give the same model), then least squares on those points: floor spill, a lid or a neighbour projects off
    the circle and pulls nothing. Extents at the 1st/99th height percentiles, at least one `step` tall; faces cut to
    edges <= `step` for per-vertex colour. A radius above the arc's own chord (under ~60 degrees of arc) is refused: a
    plane fits such points as well, and the circle's size would be a guess. Returns world vertices and faces.
    """
    import trimesh
    from scipy.optimize import least_squares
    up = np.asarray(up, float) / np.linalg.norm(up)
    a = np.cross(up, [1., 0, 0] if abs(up[0]) < .9 else [0, 1., 0])
    a /= np.linalg.norm(a)
    axes = np.stack([a, np.cross(up, a), up])  # rows, right-handed
    xy, height = points @ axes[:2].T, points @ up
    rng = np.random.default_rng(SEED)
    pool = xy[rng.choice(len(xy), min(len(xy), sample), replace=False)]
    tri = pool[rng.integers(0, len(pool), (trials, 3))]
    lhs, rhs = 2 * (tri[:, 1:] - tri[:, :1]), (tri[:, 1:] ** 2).sum(2) - (tri[:, :1] ** 2).sum(2)
    tri, lhs, rhs = (x[np.abs(np.linalg.det(lhs)) > 1e-12] for x in (tri, lhs, rhs))  # collinear triples have no circle
    centres = np.linalg.solve(lhs, rhs[..., None])[..., 0]
    radii = np.linalg.norm(tri[:, 0] - centres, axis=1)
    best = (np.abs(np.linalg.norm(pool[None] - centres[:, None], axis=2) - radii[:, None]) <= step).sum(1).argmax()
    ring = np.abs(np.linalg.norm(xy - centres[best], axis=1) - radii[best]) <= step
    cx, cy, radius = least_squares(lambda p: np.linalg.norm(xy[ring] - p[:2], axis=1) - p[2], [*centres[best], radii[best]]).x
    radius = abs(radius)
    ring = np.abs(np.linalg.norm(xy - [cx, cy], axis=1) - radius) <= step
    if ring.sum() < 10:
        return None
    along = xy[ring] @ np.linalg.svd(xy[ring] - xy[ring].mean(0), full_matrices=False)[2][0]
    if not radius <= np.subtract(*np.percentile(along, [99, 1])):
        return None
    lo, hi = np.percentile(height, [1, 99])
    unit = trimesh.creation.cylinder(radius=radius, segment=[[cx, cy, lo], [cx, cy, max(hi, lo + step)]],
                                     sections=max(16, int(np.ceil(2 * np.pi * radius / step))))
    vertices, faces = trimesh.remesh.subdivide_to_size(unit.vertices, unit.faces, step)
    return vertices @ axes, faces


def box_obtain(payload, args):
    """--generator box or cylinder: the shape of the points the ICP half of the agreeing views saw (the held-out half
    never shapes it), coloured from the source crop where it faces that camera, elsewhere the mask's median colour.
    Deterministic, so a second seed of the same view is not tried."""
    import trimesh
    if payload["seed"] != SEED:
        return None, {"reason": f"a {args.generator} does not depend on the seed"}
    entity, view = args.box_lookup[payload["entityId"], payload["anchorObservationId"]]
    observed, names, _, _, owner, _ = observed_points(entity, view, args.rows, args.clip, args)
    source = names.index(view["observation"])
    shaping = ~np.isin(owner, [i for i in range(len(names)) if i % 2 and i != source])  # assess()'s held-out split, inverted
    if shaping.sum() < 10:
        return None, {"reason": f"too few points to shape a {args.generator}"}
    shape = (cylinder_mesh if args.generator == "cylinder" else box_mesh)(observed[shaping], args.plan_up, 2 * VOXEL)
    if shape is None:
        return None, {"reason": "the points' floor-plane projection is not an arc: no cylinder"}
    vertices, faces = shape
    crop = payload["views"][0]
    rgb, mask, k, c2w = crop["rgb"], np.asarray(crop["mask"], bool), np.asarray(crop["K"], float), np.asarray(crop["cameraToWorld"], float)
    colours = np.tile(np.median(rgb[mask], 0) if mask.any() else rgb.reshape(-1, 3).mean(0), (len(vertices), 1))
    local = (vertices - c2w[:3, 3]) @ c2w[:3, :3]
    z = np.maximum(local[:, 2], 1e-9)
    u, v = np.round(k[0, 0] * local[:, 0] / z + k[0, 2]).astype(int), np.round(k[1, 1] * local[:, 1] / z + k[1, 2]).astype(int)
    toward = ((c2w[:3, 3] - vertices) * trimesh.Trimesh(vertices, faces, process=False).vertex_normals).sum(1) > 0
    ok = toward & (local[:, 2] > 0) & (u >= 0) & (v >= 0) & (u < rgb.shape[1]) & (v < rgb.shape[0])
    colours[ok] = rgb[v[ok], u[ok]]
    return (vertices, faces, colours.astype(np.float64)), {"generator": GENERATORS[args.generator], "shapedBy": int(shaping.sum())}


def sam3d_spent(journal):
    """GPU seconds and USD of every SAM 3D call journaled under `journal` (container wall time x list price)."""
    seconds = sum(json.loads(p.read_text()).get("seconds", 0) for p in journal.rglob("sam3d-*/record.json")) if journal.exists() else 0
    return seconds, seconds * SAM3D_USD_PER_SECOND


def sam3d_obtain(payload, folder, args):
    """SAM 3D Objects for one input (self-hosted, modal_apps/sam3d_research.py): read back once received, else one call within
    --max-usd. A dispatch without an output is never sent again. Returns (world-frame mesh or None, record or None)."""
    view = payload["views"][0]
    names = ("fullRgb", "fullMask", "fullDepth", "fullK") if "fullRgb" in view else ("rgb", "mask", "depth", "K")
    rgb, mask, depth, k = (np.ascontiguousarray(np.asarray(view[n])) for n in names)
    key = hashlib.sha256(b"".join([json.dumps({"seed": payload["seed"], **{k: SAM3D[k] for k in ("modelRevision", "codeRevision")}}).encode(),
                                   rgb.tobytes(), mask.astype(bool).tobytes(), depth.astype(np.float32).tobytes()])).hexdigest()[:24]
    root = folder / f"sam3d-{key}"
    if not (root / "output.npz").exists():
        if (root / "dispatch.json").exists():
            return None, {"error": "unresolved SAM 3D dispatch that may have run: never sent again"}
        if not args.invoke:
            return None, None
        usd = sam3d_spent(args.output / "journal-sam3d")[1]
        if usd + SAM3D_WORST_SECONDS * SAM3D_USD_PER_SECOND > args.max_usd:
            return None, {"error": f"not requested: {usd:.2f} USD spent on SAM 3D, one more worst-case call could pass the cap"}
        import modal
        root.mkdir(parents=True, exist_ok=True)
        save(root / "dispatch.json", {"entity": payload["entityId"], "observation": payload["anchorObservationId"], "seed": payload["seed"], **SAM3D})
        started = time.time()
        try:
            out = modal.Cls.from_name(SAM3D["modalApp"], SAM3D["modalClass"])().run.remote(rgb, mask.astype(bool), sam3d_pointmap(depth, k.astype(float)), payload["seed"])
        except Exception as error:  # the call may have run: kept as dispatched, never re-sent
            save(root / "record.json", {"error": type(error).__name__, "detail": str(error)[:300], "seconds": time.time() - started})
            return None, {"error": type(error).__name__}
        if "error" in out:  # the worker's traceback: journaled with the dispatch, this input is never sent again
            save(root / "record.json", {"error": out["error"][-2000:], "seconds": time.time() - started, "gpu": out.get("gpu")})
            return None, {"error": "sam3d_worker_error"}
        np.savez_compressed(root / "output.npz", **{k: out[k] for k in ("vertices", "faces", "colors", "objectToCamera")})
        save(root / "record.json", {"seconds": time.time() - started, "workerSeconds": out["seconds"], "gpu": out["gpu"], "pins": out["pins"]})
    out, record = np.load(root / "output.npz"), json.loads((root / "record.json").read_text())
    vertices = sam3d_to_world(out["vertices"], out["objectToCamera"], view["cameraToWorld"])
    telemetry = {"workerElapsedSeconds": record["seconds"], "gpuElapsedSeconds": record.get("workerSeconds")}
    return (vertices, np.asarray(out["faces"]), np.asarray(out["colors"])), {"pins": record.get("pins"), "telemetry": telemetry, "journal": str(root),
                                                                             "runtimeEvidence": {"hardware": {"gpu": record.get("gpu")}}}


def request(payload, folder, function_id):
    """One journaled RecGen call, or the read-back of a received one: the full mesh placed in the scene frame."""
    from ehs_spatial.platform.contracts import PlatformError
    from ehs_spatial.platform.recgen import RecGenRequest, adapt_output
    from ehs_spatial.platform.recgen_transport import invoke
    from ehs_spatial.platform.reconstruction import ProviderResponseError
    from ehs_spatial.platform.spatial import transform_points
    os.environ["PANOPTES_RECGEN_JOURNAL"] = str(folder)
    try:
        response = invoke(payload, {**RECGEN, "modalFunctionId": function_id})
    except ProviderResponseError as error:  # journaled as outcome unknown
        return None, {"error": "provider_outcome_unknown", "detail": str(error)[:300]}
    record = {k: v for k, v in response.items() if k not in ("vertices", "faces", "colors", "officialPosedVertices")}
    if response.get("providerError"):
        return None, record
    try:
        result = adapt_output(RecGenRequest.from_payload(payload), response)
    except (PlatformError, ValueError) as error:  # kept on record; the journal still holds the paid files
        return None, {**record, "error": getattr(error, "code", type(error).__name__)}
    colors = np.asarray(result["colors"], np.float64)[:, :3]
    colors = colors / 255 if colors.max() > 1 else colors
    record.update(provenance=result["provenance"], proposedObjectToNative=result["proposedObjectToNative"].tolist(), journal=str(folder))
    return (transform_points(result["vertices"], result["proposedObjectToNative"]), np.asarray(result["faces"]),
            (colors * 255).round().astype(np.uint8)), record


def prune(folder):
    """Drop a journal's uploaded input and downloaded meshes once the model is written: the volume keeps them, and a
    received journal reads them back without a new call."""
    for name in ("input.npz", "object.glb", "object.ply", "posed-object.glb", "posed-object.ply"):
        for path in folder.rglob(name):
            path.unlink()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def entity_coverage(centres, cell, vertices, faces):
    """Share of the entity's agreed cells (centres, from its usable views) lying on the model, within two cells.

    The object map can merge a bench with the things on it; a best-view mask may then be one of those things, and a
    good model of it is still not a model of the entity.
    """
    return float((closest(vertices, faces, centres)[1] <= 2 * cell).mean()) if len(centres) else None


def coverage_reason(coverage):
    if coverage is not None and coverage < FIT_GATE["min_entity_coverage"]:
        return f"model covers only {coverage:.0%} of its entity's agreed cells: the best-view mask is a part of the entity or another object merged into it"
    return None


def assess(entity, view, crop, mesh, record, rows, clip, args):
    """Place, mark and judge one generated mesh at full resolution; nothing is written."""
    import io
    import trimesh
    from build_lingbot_object_model import evaluate
    vertices, faces, colors = mesh
    observed, fit_views, centres, cell, owner, views_report = observed_points(entity, view, rows, clip, args)
    source = fit_views.index(view["observation"])
    judged = np.isin(owner, [i for i in range(len(fit_views)) if i % 2 and i != source])  # held out of the ICP, the only judges of fit
    row = rows[view["frame"]]
    depth, moving = reliable(row, clip, args.dynamic_masks)
    mask = main_parts(clip.raster_mask(Path(view["mask"]))) & ~moving
    source_view = lambda v, f: evaluate(v, f, depth, np.full(depth.shape, 2., np.float32), depth > 0, mask, clip.k_raster, row["c2w"])
    light, light_faces, _ = decimate(vertices, faces, colors / 255, FIT_TRIANGLES)  # ICP on a light copy; its transform moves the full mesh
    moved_by, _ = fit(light, light_faces, observed[~judged], eye=row["c2w"][:3, 3])
    refined = vertices @ moved_by[:3, :3].T + moved_by[:3, 3]
    unrefined, before = source_view(vertices, faces), vertices.mean(0)
    kept = source_view(refined, faces)["silhouette_iou"] >= unrefined["silhouette_iou"] - .05
    vertices, moved_by = (refined, moved_by) if kept else (vertices, np.eye(4))  # else the refinement cost the source view its silhouette
    distance = closest(vertices, faces, observed[judged])[1] if judged.any() else np.empty(0)
    median = float(np.median(distance)) if len(distance) else None  # None: nothing held out to judge the fit by (a reason below)
    flags = seen(vertices, faces, observed, [rows[int(o.rsplit(":", 2)[1])]["c2w"] for o in fit_views], clip.k_raster)
    rgba = np.column_stack([colors, np.where(flags, OBSERVED_ALPHA, INFERRED_ALPHA)]).astype(np.uint8)
    plain = glb(vertices, faces, rgba, blend=False)
    exported = trimesh.load(io.BytesIO(plain), file_type="glb", force="mesh", process=False)  # judge the serialized deliverable
    gate = source_view(np.asarray(exported.vertices), np.asarray(exported.faces))
    area = trimesh.Trimesh(vertices, faces, process=False).area_faces
    metres, telemetry = args.metres_per_native, record.get("telemetry") or {}
    p90 = float(np.percentile(distance, 90)) if len(distance) else None
    fitted = {"fitResidualNative": median, "fitResidualP90Native": p90,
              "fitResidualCm": None if median is None else round(median * metres * 100, 2), "fitResidualP90Cm": None if p90 is None else round(p90 * metres * 100, 2),
              "observedCoverage": float((distance <= 2 * VOXEL).mean()) if len(distance) else 0., "observedPoints": len(observed), "judgedPoints": int(judged.sum()), **views_report,
              "icpScale": float(np.cbrt(np.linalg.det(moved_by[:3, :3]))), "icpCentroidMoveNative": float(np.linalg.norm(vertices.mean(0) - before)),
              "icpTransform": moved_by.tolist(), "icpKept": bool(kept),
              "icpRotationDeg": float(np.degrees(np.arccos(np.clip((np.trace(moved_by[:3, :3] / np.cbrt(np.linalg.det(moved_by[:3, :3]))) - 1) / 2, -1, 1)))),
              "unrefined": {k: unrefined[k] for k in ("silhouette_iou", "relative_depth_median", "relative_depth_p95")}}
    coverage = entity_coverage(centres, cell, vertices, faces)
    low, high = FIT_GATE["scale_range"]
    reasons = [r for r, bad in (("source-view gate failed (silhouette/depth)", not gate["accepted_source_consistency"]),
                                (f"fit median {median or 0:.4f} > {FIT_GATE['max_fit_median_native']} native", median is not None and median > FIT_GATE["max_fit_median_native"]),
                                (f"model explains only {fitted['observedCoverage']:.0%} of the observed points", fitted["observedCoverage"] < FIT_GATE["min_observed_coverage"]),
                                (coverage_reason(coverage), coverage_reason(coverage) is not None),
                                (f"only {views_report['viewsAgreeing']} views agree with the best view: the fit cannot check more than the view it was made from",
                                 views_report["viewsAgreeing"] < FIT_GATE["min_agreeing_views"]),
                                (f"only {int(judged.sum())} held-out points to judge the fit", judged.sum() < FIT_GATE["min_judged_points"]),
                                (f"ICP scale {fitted['icpScale']:.2f} at its clamp: the generated size is off by more than the fit may correct",
                                 not low * 1.01 < fitted["icpScale"] < high / 1.01),
                                (f"ICP turned the model {fitted['icpRotationDeg']:.0f} degrees: the generated pose was far off",
                                 fitted["icpRotationDeg"] > FIT_GATE["max_icp_rotation_deg"])) if bad]
    validation = {**gate, "entity": entity["entityId"], "observation": view["observation"], "accepted_source_consistency": not reasons,
                  "rejectionReasons": reasons, **fitted, "entityCoverage": coverage, "entityCoverageBasis": "agreed cells of the usable views", "viewMask": view["mask"],
                  "fitGate": FIT_GATE,
                  "observedShare": float(flags.mean()), "observedAreaShare": float(area[flags[faces].all(1)].sum() / area.sum()),
                  "observedRule": f"within {2 * VOXEL} native of an observed point, facing one of the fit views' cameras and in its clear sight",
                  "skipFrames": sorted(args.skip),
                  "alpha": {"observed": OBSERVED_ALPHA, "inferred": INFERRED_ALPHA}, "triangles": len(faces), "seed": SEED, "sourceFrame": view["frame"],
                  "generator": GENERATORS[args.generator], "pins": record.get("pins"), "providerRequestId": record.get("providerRequestId"),
                  "gpuType": ((record.get("runtimeEvidence") or {}).get("hardware") or {}).get("gpu"),
                  "gpuSeconds": telemetry.get("gpuElapsedSeconds"), "wallSeconds": telemetry.get("workerElapsedSeconds"),
                  "estimatedUsd": round((telemetry.get("workerElapsedSeconds") or 0) * (SAM3D_USD_PER_SECOND if args.generator == "sam3d" else USD_PER_SECOND), 4),
                  "coordinateFrame": "droid_final_native_world", "metresPerNativeUnit": metres,
                  "maskSource": ({"method": "SAM 2.1 box prompt on the uncropped 1280x720 frame (the observation's mask was cut by the 4:3 crop)", **view["resegmented"]}
                                 if view.get("resegmented") else "the observation's mask")}
    return {"view": view, "crop": crop, "vertices": vertices, "faces": faces, "colors": colors, "flags": flags, "plain": plain, "validation": validation}


def write(entity, result, args, plan_up):
    """models/<entity>/ (full mesh, plain vertex colours), glb/<entity>.glb (light copy, BLEND) and the review sheet."""
    from scipy.spatial import cKDTree
    eid, out, validation, view = entity["entityId"], args.output, result["validation"], result["view"]
    folder = out / "models" / eid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.glb").write_bytes(result["plain"])
    validation["mesh_sha256"] = hashlib.sha256(result["plain"]).hexdigest()
    save(folder / "validation.json", validation)
    save(folder / "anchor.json", {"entity": eid, "observation": view["observation"], "sourceFrame": view["frame"]})
    light, light_faces, light_colors = decimate(result["vertices"], result["faces"], result["colors"] / 255, LIGHT_TRIANGLES)
    alpha = np.where(result["flags"], OBSERVED_ALPHA, INFERRED_ALPHA)[cKDTree(result["vertices"]).query(light)[1]]
    (out / "glb").mkdir(exist_ok=True)
    (out / "glb" / f"{eid}.glb").write_bytes(glb(light, light_faces, np.column_stack([(light_colors * 255).round(), alpha]).astype(np.uint8), blend=True))
    (out / "review").mkdir(exist_ok=True)
    review(out / "review" / f"{eid}.jpg", result["crop"], result["vertices"], result["faces"], result["colors"].astype(np.float64), result["flags"],
           args.rows[view["frame"]]["c2w"], plan_up,
           [f"{eid} '{entity['label']}' frame {view['frame']} seed {validation['seed']} ({len(validation.get('attempts') or [1])} tries)  "
            + ("ACCEPTED" if validation["accepted_source_consistency"] else "REJECTED: " + "; ".join(validation["rejectionReasons"])),
            f"IoU {validation['silhouette_iou']:.2f}  depth err med {validation['relative_depth_median'] or 0:.3f}  fit med {validation['fitResidualCm']} cm  "
            f"coverage {validation['observedCoverage']:.0%}  observed {validation['observedShare']:.0%}  | crop | model (magenta = inferred) | far side"])
    return len(light_faces)


def prepare(args):
    """Selection, best views and generator inputs; no model call."""
    import mono_room
    clip = Clip(args.droid_run, args.clip)
    rows = {r["source_index"]: r for r in mono_room.load(args.droid_run, None, args.depth_run)}
    rows = {i: r for i, r in rows.items() if i not in args.skip}  # e.g. a cut-away shot whose cameras are wrong: never a view
    for row in rows.values():
        for key in ("conf", "droid", "retained"):
            row.pop(key, None)  # hundreds of views have to fit in memory
    document = json.loads((args.object_map / "object-map.json").read_text())
    entities = {e["entityId"]: e for e in document["entities"]}
    excluded = dict((item + "=").split("=")[:2] for item in args.exclude)
    rule = {e["entityId"]: why for e, why in candidates(document["entities"], skip=args.skip)}
    explicit = [(item + "=").split("=")[:2] for item in args.entities]  # "id" or "id=review note", in priority order
    ranked = ([(entities[e], (rule.get(e, "outside the selection rule") + "; review: " + (note or "chosen")).strip()) for e, note in explicit]
              if explicit else [(entities[e], why) for e, why in rule.items()])
    args.explicit = {e for e, _ in explicit}
    if args.all:  # then every other named physical object: the manifest counts them as qualified
        extended = candidates(document["entities"], ALL_STATUSES, ALL_MIN_VIEWS, args.metres_per_native, args.skip)
        args.qualified = [e["entityId"] for e, _ in extended]
        ranked += [(e, why) for e, why in extended if e["entityId"] not in args.explicit]
    rejected = [{"entityId": e, "label": entities[e]["label"], "reason": "excluded at review: " + why} for e, why in excluded.items()]
    if args.all:  # seen only in skipped frames: not a separate object of the 3D map
        rejected += [{"entityId": e["entityId"], "label": e["label"], "reason": "skipped: cut-away shot only"} for e in document["entities"]
                     if e["sourceFrames"] and not set(e["sourceFrames"]) - args.skip and e.get("labelStatus") in ALL_STATUSES
                     and not matches(e["label"], EXCLUDED) and e["entityId"] not in args.explicit]
    chosen, cache = [], {}
    for entity, why in ranked:
        eid = entity["entityId"]
        if not args.all and len(chosen) == args.count:
            break
        if eid in excluded:
            continue
        if not set(entity["sourceFrames"]) - args.skip:
            rejected.append({"entityId": eid, "label": entity["label"], "reason": "skipped: cut-away shot only"})
            continue
        box = np.array(entity["boundsNative"])
        twin = next((c["entity"]["entityId"] for c in chosen if box_iou(box, np.array(c["entity"]["boundsNative"])) >= .25), None)
        if twin and eid not in args.explicit:
            rejected.append({"entityId": eid, "label": entity["label"], "reason": f"3D box overlaps selected {twin} (IoU >= 0.25): likely the same object"})
            continue
        metrics = view_metrics(entity, rows, clip, args, cache)
        views, side = usable(metrics), [m for m in metrics if "side" in failing(m) and failing(m) <= {"side", "raster edge"}]
        if not views and not side:
            least = min((failing(m) for m in metrics), key=len, default={"no mask with depth"})
            rejected.append({"entityId": eid, "label": entity["label"], "reason": "no usable view: the least-failing view fails " + ", ".join(sorted(least)),
                             "largestMaskPx": max((m["area"] for m in metrics), default=0)})
            continue
        chosen.append({"entity": entity, "why": why, "views": views or side, "resegment": not views})
    tries = pick_views({c["entity"]["entityId"]: c.pop("views") for c in chosen}, clip)
    for c in chosen:  # after a failed fit the next try is the best view from another moment of the walk
        c["tries"] = tries[c["entity"]["entityId"]]
    chosen, lost = resegment(chosen, clip, args)
    rejected += lost
    args.video_sha = hashlib.sha256(clip.video.read_bytes()).hexdigest()
    wanted = {}
    for c in chosen:
        c["payloads"] = [None] * len(c["tries"])
        for n, view in enumerate(c["tries"]):
            wanted.setdefault(view["frame"], []).append((c, n))
    for index, rgb in clip.frames(set(wanted)):  # only the crops are kept, never the frames
        for c, n in wanted[index]:
            c["payloads"][n] = build_input(c["entity"], c["tries"][n], rows[index], rgb, clip, args, args.video_sha)
    return clip, rows, document, chosen, rejected


def resegment(chosen, clip, args):
    """Views cut by the 4:3 crop's side, segmented again on the uncropped 16:9 frame.

    SAM 2.1 (sam2_everything.box_masks, one call) gets the mask's box widened past each cut side by the mask's own
    width. A new mask is kept only if it is whole (clear of the frame edge) and the same object (IoU >= 0.6 with the old
    mask inside the crop); a view whose mask fails is dropped from the entity's tries, and an entity left without one is
    dropped with the reason. Every answer is kept under resegmented/<entity>/ and reused, so a rerun asks SAM nothing twice.
    Returns (kept entities, dropped ones).
    """
    record = lambda eid, view: args.output / "resegmented" / eid / f"frame-{view['frame']:05d}.json"
    todo = [(c["entity"]["entityId"], view) for c in chosen if c.get("resegment") for view in c["tries"]]
    ask = [(eid, view) for eid, view in todo if not record(eid, view).exists()]
    requests, olds, answers = {}, {}, {}
    frames = dict(clip.frames({view["frame"] for _, view in ask}))
    for eid, view in todo:
        name = f"{eid}:{view['frame']}"
        olds[name] = clip.full_mask(main_parts(cv2.imread(view["mask"], cv2.IMREAD_GRAYSCALE) > 0))
        ys, xs = np.nonzero(olds[name])
        x0, x1 = int(xs.min()), int(xs.max())
        width = x1 - x0
        box = [max(0, x0 - width) if "left" in view["cut"] else x0, int(ys.min()), min(clip.full_size[0] - 1, x1 + width) if "right" in view["cut"] else x1, int(ys.max())]
        if not record(eid, view).exists():
            requests[name] = np.ascontiguousarray(frames[view["frame"]][..., ::-1]), box
        else:
            saved = json.loads(record(eid, view).read_text())
            source = record(eid, view).with_name(record(eid, view).stem + "-source-mask.png")
            answers[name] = (cv2.imread(str(source), cv2.IMREAD_GRAYSCALE) > 0 if source.exists() else np.zeros(clip.full_size[::-1], bool)), saved["samScore"]
    if requests and not args.invoke:  # a SAM call costs GPU: only with --invoke; this run treats those views as not re-segmented
        answers.update({name: (np.zeros(clip.full_size[::-1], bool), 0.) for name in requests})
        requests = {}
    if requests:
        import time
        import sam2_everything
        started = time.time()
        answers.update(sam2_everything.box_masks(requests))
        ledger = args.output / "resegmented" / "sam2-calls.json"  # every SAM call of this output, for the spend
        calls = json.loads(ledger.read_text()) if ledger.exists() else []
        save(ledger, calls + [{"wallSeconds": round(time.time() - started, 1), "masks": len(requests)}])
    inside = np.zeros(clip.full_size[::-1], bool)
    inside[:, int(clip.clip_to_full[0, 2]):int(clip.clip_to_full[0, 2] + clip.clip_to_full[0, 0] * clip.clip_size[0])] = True
    kept, lost = [], []
    for c in chosen:
        if not c.get("resegment"):
            kept.append(c)
            continue
        eid, notes, tries = c["entity"]["entityId"], [], []
        for view in c["tries"]:
            name = f"{eid}:{view['frame']}"
            mask, score = answers[name]
            ys, xs = np.nonzero(mask)
            iou = float((mask & olds[name] & inside).sum() / max(((mask & inside) | olds[name]).sum(), 1))
            whole = len(xs) and xs.min() >= 2 and ys.min() >= 2 and xs.max() < clip.full_size[0] - 2 and ys.max() < clip.full_size[1] - 2
            notes.append({"view": view["frame"], "iouWithCroppedMask": round(iou, 3), "whole": bool(whole), "samScore": None if score is None else round(score, 3)})
            folder = args.output / "resegmented" / eid
            full_path, clip_path = folder / f"frame-{view['frame']:05d}-source-mask.png", folder / f"frame-{view['frame']:05d}-clip-mask.png"
            if not record(eid, view).exists():
                save(record(eid, view), notes[-1])
                if whole and iou >= .6:
                    cv2.imwrite(str(full_path), mask.astype(np.uint8) * 255)
                    cv2.imwrite(str(clip_path), clip.clip_mask(mask).astype(np.uint8) * 255)
            if whole and iou >= .6:
                tries.append({**view, "mask": str(clip_path), "fullMask": str(full_path), "resegmented": notes[-1]})
        c["tries"] = tries
        if tries:
            kept.append(c)
        else:
            lost.append({"entityId": eid, "label": c["entity"]["label"], "resegmentation": notes,
                         "reason": "no usable view: cut by the 4:3 crop's side, and SAM 2.1 on the uncropped frame gave no whole mask of the same object"})
    return kept, lost


def sheet(path, chosen):
    """Contact sheet of the chosen best-view crops, mask outlined: the evidence to check a VLM name before paying for it."""
    side, per_row = (360, 5) if len(chosen) <= 20 else (240, 8)
    tiles = []
    for c in chosen:
        crop = c["payloads"][0]["views"][0]
        tile = np.ascontiguousarray(crop["rgb"][..., ::-1])
        cv2.drawContours(tile, cv2.findContours(crop["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 230, 255), 3)
        tile = cv2.resize(tile, (side, side), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"{c['entity']['entityId']} {c['entity']['label']} f{c['tries'][0]['frame']}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % per_row)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {k: str(getattr(args, k)) for k in ("droid_run", "depth_run", "object_map", "masks", "dynamic_masks", "clip")}  # before args.clip becomes the Clip
    args.metres_per_native = json.loads((args.depth_run / "metric-scale.json").read_text())["metres_per_native_unit"]
    clip, rows, document, chosen, rejected = prepare(args)
    args.rows, args.gpu, args.idle = rows, GPUS[0], 0  # A100-80GB had no capacity when this ran; the first re-send goes to A100-40GB
    sheet(args.output / "selection.jpg", chosen)
    plan_up = np.array(document["plan"]["up"]) if document.get("plan") else np.array([0, -1., 0])  # floor normal; image y is down
    from concurrent.futures import ThreadPoolExecutor
    journal, entries, placed = args.output / "journal", [], []  # placed: (entity, light vertices, light faces) of accepted models
    pool = ThreadPoolExecutor(max_workers=1)  # one GPU call at a time, running ahead while the previous object is checked
    ahead = {}
    # the generator's call for one input: RecGen through its journaled transport, or self-hosted SAM 3D with its own journal
    call = (lambda payload, frame: (obtain, payload, journal_folder(journal, payload, frame, args.function_id), args)) if args.generator == "recgen" else \
           (lambda payload, frame: (sam3d_obtain, payload, args.output / "journal-sam3d" / payload["entityId"], args)) if args.generator == "sam3d" else \
           (lambda payload, frame: (box_obtain, payload, args))
    args.clip, args.plan_up = clip, plan_up
    args.box_lookup = {(c["entity"]["entityId"], view["observation"]): (c["entity"], view) for c in chosen for view in c["tries"]}

    def parent_of(c):
        """The accepted model an extended entity mostly lies on (it is a part of that object), or None."""
        if c["entity"]["entityId"] in args.explicit:
            return None
        crop = c["payloads"][0]["views"][0]
        points = lift(np.where(crop["mask"], crop["depth"], 0), np.asarray(crop["K"], float), np.asarray(crop["cameraToWorld"], float))
        for eid, vertices, faces in placed:
            low, high = vertices.min(0) - 2 * VOXEL, vertices.max(0) + 2 * VOXEL
            if len(points) and ((points >= low) & (points <= high)).all(1).mean() >= PART_SHARE:
                share = float((closest(vertices, faces, points)[1] <= 2 * VOXEL).mean())
                if share >= PART_SHARE:
                    return {"parent": eid, "shareOnParent": round(share, 3)}
        return None

    superseded = []
    for validation in sorted((args.output / "models").glob("*/validation.json")):
        v = json.loads(validation.read_text())
        used = [v["sourceFrame"]] + [int(o.rsplit(":", 2)[1]) for o in v.get("fitViews", [])]
        if any(f in args.skip for f in used):  # its view or fit points came from frames that are never to be used
            eid, target = v["entity"], args.output / "superseded" / v["entity"]
            target.mkdir(parents=True, exist_ok=True)
            for source in (args.output / "models" / eid, args.output / "glb" / f"{eid}.glb", args.output / "review" / f"{eid}.jpg"):
                if source.exists():
                    source.rename(target / source.name)
            for mesh in (target / eid / "model.glb", target / f"{eid}.glb"):  # placed with wrong cameras: only the record and review stay
                mesh.unlink(missing_ok=True)
            superseded.append({"entityId": eid, "sourceFrame": v["sourceFrame"], "skippedFramesUsed": sorted({f for f in used if f in args.skip})})
            save(target / "superseded.json", superseded[-1])
    superseded = [json.loads(p.read_text()) for p in sorted((args.output / "superseded").glob("*/superseded.json"))]  # earlier runs' too

    def written(eid):
        """The validation of a model this output already holds (its hash still matching), or None; with --reassess always None,
        so each try is read back from the journal and judged again."""
        if args.reassess:
            return None
        folder = args.output / "models" / eid
        try:
            v = json.loads((folder / "validation.json").read_text())
            return v if v.get("mesh_sha256") == hashlib.sha256((folder / "model.glb").read_bytes()).hexdigest() else None
        except OSError:
            return None

    def queue(i):
        if i < len(chosen) and i not in ahead:
            c = chosen[i]
            previous = written(c["entity"]["entityId"])
            parent = None if previous else parent_of(c)
            ahead[i] = parent, None if parent or previous else pool.submit(*call(c["payloads"][0], c["tries"][0]["frame"])), previous

    queue(0)
    for i, c in enumerate(chosen):
        eid = c["entity"]["entityId"]
        parent, first, previous = ahead.pop(i)
        parent = parent or (parent_of(c) if first else None)  # the object just before may have become its parent
        queue(i + 1)
        entry = {"entityId": eid, "label": c["entity"]["label"], "labelStatus": c["entity"].get("labelStatus"), "selection": c["why"], "seed": SEED}
        if c["entity"].get("labelStatus") != "clear":
            entry["labelUncertain"] = f"the VLM marked the name '{c['entity']['label']}' as {c['entity'].get('labelStatus')}"
        if previous:  # already modelled by an earlier run of this output: reused, never regenerated
            import trimesh
            previous.update({k: entry[k] for k in ("labelStatus", "labelUncertain") if k in entry})
            if previous.get("entityCoverageBasis") != "agreed cells of the usable views":  # checked before this rule existed: judged on its display copy
                light = trimesh.load(args.output / "glb" / f"{eid}.glb", force="mesh", process=False)
                view = {"frame": previous["sourceFrame"], "observation": previous["observation"],
                        "mask": previous.get("viewMask") or str(mask_path(args.masks, previous["observation"]))}
                _, _, centres, cell, _, _ = observed_points(c["entity"], view, rows, clip, args)
                previous.update(entityCoverage=entity_coverage(centres, cell, np.asarray(light.vertices), np.asarray(light.faces)),
                                entityCoverageBasis="agreed cells of the usable views", fitGate=FIT_GATE)
                previous["rejectionReasons"] = [r for r in previous["rejectionReasons"] if not r.startswith("model covers only")]
                reason = coverage_reason(previous["entityCoverage"])
                if reason:
                    previous["rejectionReasons"].append(reason)
                previous["accepted_source_consistency"] = not previous["rejectionReasons"]
            save(args.output / "models" / eid / "validation.json", previous)
            entries.append({**entry, "attempts": previous.get("attempts"), "view": previous["sourceFrame"], "observation": previous["observation"],
                            "glb": f"glb/{eid}.glb", "model": f"models/{eid}/model.glb", "review": f"review/{eid}.jpg", "triangles": previous["triangles"],
                            **{k: previous[k] for k in ("observedShare", "fitResidualNative", "fitResidualCm", "gpuType", "gpuSeconds")},
                            "accepted": previous["accepted_source_consistency"], "rejectionReasons": previous["rejectionReasons"], "silhouetteIou": previous["silhouette_iou"],
                            "maskSource": previous.get("maskSource")})
            if previous["accepted_source_consistency"]:
                light = trimesh.load(args.output / "glb" / f"{eid}.glb", force="mesh", process=False)
                placed.append((eid, np.asarray(light.vertices), np.asarray(light.faces)))
            continue
        if parent:
            entries.append({**entry, "status": f"skipped: part of {parent['parent']}", **parent})
            print(json.dumps({"entity": eid, "skipped": parent}), flush=True)
            continue
        # the views best first, then the best view with another seed; the first accepted model ends the tries
        tries = [(view, payload) for view, payload in zip(c["tries"], c["payloads"])] + [(c["tries"][0], {**c["payloads"][0], "seed": EXTRA_SEED})]
        best, attempts = None, []
        judge = lambda r: (r["validation"]["accepted_source_consistency"], r["validation"]["silhouette_iou"])
        for n, (view, payload) in enumerate(tries):
            if best and best["validation"]["accepted_source_consistency"]:
                break
            mesh, record = (first if n == 0 else pool.submit(*call(payload, view["frame"]))).result()
            attempt = {"attempt": n + 1, "view": view["frame"], "observation": view["observation"], "seed": payload["seed"],
                       "viewScore": view.get("score"), "providerRequestId": (record or {}).get("providerRequestId")}
            if mesh is None:  # not requested (no --invoke, or the cap), or no model: the next try may still be read back
                attempts.append({**attempt, "generated": False, "record": record})
                continue
            result = assess(c["entity"], view, payload["views"][0], mesh, record, rows, clip, args)
            v = result["validation"]
            v["seed"] = payload["seed"]
            attempts.append({**attempt, "generated": True, **{k: v[k] for k in ("accepted_source_consistency", "rejectionReasons", "silhouette_iou",
                                                                                  "relative_depth_median", "relative_depth_p95", "fitResidualNative",
                                                                                  "observedCoverage", "entityCoverage", "gpuType", "gpuSeconds", "wallSeconds", "estimatedUsd")}})
            print(json.dumps({"entity": eid, "attempt": n + 1, "view": view["frame"], "seed": payload["seed"], "accepted": v["accepted_source_consistency"],
                              "iou": round(v["silhouette_iou"], 3), "fitCm": v["fitResidualCm"], "observedShare": round(v["observedShare"], 3)}), flush=True)
            best = result if best is None or judge(result) > judge(best) else best
            del result, mesh
        entry["attempts"] = attempts
        if best:
            best["validation"].update(attempts=attempts, labelStatus=entry["labelStatus"], **({"labelUncertain": entry["labelUncertain"]} if "labelUncertain" in entry else {}))
            light = write(c["entity"], best, args, plan_up)
            if best["validation"]["accepted_source_consistency"]:
                placed.append((eid, *decimate(best["vertices"], best["faces"], best["colors"] / 255, LIGHT_TRIANGLES)[:2]))
            v, view = best["validation"], best["view"]
            entry.update(view=view["frame"], observation=view["observation"], seed=v["seed"], viewScore={k: view.get(k) for k in ("area", "sharpness", "frontal", "solidity", "score")},
                         glb=f"glb/{eid}.glb", glbTriangles=light, model=f"models/{eid}/model.glb", review=f"review/{eid}.jpg", triangles=v["triangles"],
                         observedShare=v["observedShare"], fitResidualNative=v["fitResidualNative"], fitResidualCm=v["fitResidualCm"],
                         accepted=v["accepted_source_consistency"], rejectionReasons=v["rejectionReasons"], silhouetteIou=v["silhouette_iou"],
                         gpuType=v["gpuType"], gpuSeconds=v["gpuSeconds"], maskSource=v["maskSource"])
            prune(journal / eid)
        else:
            entry.update(view=c["tries"][0]["frame"], observation=c["tries"][0]["observation"], status="not generated")
        entries.append(entry)
        del best
    pool.shutdown()
    prune(journal)  # inputs of calls that never produced a model; nothing reads them again
    from collections import Counter
    gpu, wall, usd = spent(journal)
    sam_ledger = args.output / "resegmented" / "sam2-calls.json"
    sam_seconds = sum(call["wallSeconds"] for call in json.loads(sam_ledger.read_text())) if sam_ledger.exists() else 0
    sam_usd = sam_seconds * (.000222 + .0000131 + 8 * .00000222)
    records = [json.loads(p.read_text()) for p in journal.rglob("output.json")]
    dispatches = []
    for state_path in sorted(journal.rglob("dispatch.json")):
        state, output, asked = (json.loads(p.read_text()) if p.exists() else {} for p in
                                (state_path, state_path.parent / "output.json", state_path.parent.parent / "requested-gpu.json"))
        dispatches.append({"journal": str(state_path.parent.relative_to(journal)), "status": state.get("status"), "providerRequestId": state.get("providerRequestId"),
                           "requestedGpu": asked.get("gpu", "A100-80GB"), "ranOn": (output.get("hardware") or {}).get("gpu"),
                           "gpuSeconds": output.get("gpu_function_seconds"), "wallSeconds": state.get("wallSeconds"),
                           **({} if state.get("status") == "received" else {"modal": modal_state(state_path)})})
    save(args.output / "manifest.json", {
        "schema": "phase2-video-object-models-v1", "coordinateFrame": "droid_final_native_world", "metresPerNativeUnit": args.metres_per_native,
        "generator": GENERATORS[args.generator], "pins": SAM3D if args.generator == "sam3d" else ({k: records[0].get(k) for k in ("model_id", "model_revision", "code_revision", "weights_manifest_sha256")} if records else None),
        "route": {**RECGEN, "modalFunctionId": args.function_id, "transport": "ehs_spatial.platform.recgen_transport.invoke", "views": 1},
        "inputs": inputs,
        "selectionRule": {"labelStatus": "clear", "excludedLabels": EXCLUDED, "minViews": MIN_VIEWS, "ehsEquipment": RELEVANT,
                          "order": "EHS equipment first, then support points; an entity whose 3D box has IoU >= 0.25 with an already chosen one is skipped",
                          "bestView": "mask clear of the border and of caption text, >= 50% reliable depth, not next to the moving person; max area x relative sharpness x frontality x solidity^2",
                          "small": f"under 1500 px on the 640x480 raster and a mask box under {MIN_SOURCE_SIDE} px on its short side on the 1280x720 source frame (the generator's crop)",
                          "attempts": f"up to {ATTEMPT_VIEWS} views, highest score first, each at least {RETRY_GAP} frames from the others, then the best view "
                                      f"with seed {EXTRA_SEED}; the first accepted model ends the tries, else the highest silhouette IoU is kept"},
        "observedRule": f"vertex within {2 * VOXEL} native of an observed point and in clear sight of a camera that saw the object: alpha {OBSERVED_ALPHA}; otherwise inferred, alpha {INFERRED_ALPHA}",
        "acceptance": {"sourceView": "build_lingbot_object_model.evaluate gate", **FIT_GATE},
        "files": {"model": "models/<entity>/model.glb: the generator's full mesh, plain COLOR_0 RGBA, no material (import_video_scene.py --models)",
                  "glb": f"glb/<entity>.glb: the same model decimated to {LIGHT_TRIANGLES} triangles with a BLEND material, for direct viewing"},
        "summary": {"qualified": len(args.qualified) if args.all else None,
                    "qualifyingRule": (f"--all: labelStatus in {ALL_STATUSES}, not {EXCLUDED}, >= {ALL_MIN_VIEWS} views, >= {ALL_MIN_EXTENT_M} m across"
                                       if args.all else "the explicit --entities list"),
                    "noUsableView": sum(r["reason"].startswith("no usable view") for r in rejected),
                    "sameObjectAsSelected": sum(r["reason"].startswith("3D box overlaps") for r in rejected),
                    "partsSkipped": sum(e.get("status", "").startswith("skipped") for e in entries),
                    "modelled": sum("model" in e for e in entries), "accepted": sum(bool(e.get("accepted")) for e in entries),
                    "rejectedAfterFit": [e["entityId"] for e in entries if "model" in e and not e["accepted"]],
                    "notGenerated": [e["entityId"] for e in entries if e.get("status") == "not generated"],
                    "cutAwayOnly": [r["entityId"] for r in rejected if r["reason"] == "skipped: cut-away shot only"],
                    "superseded": superseded, "redone": [s["entityId"] for s in superseded if (args.output / "models" / s["entityId"]).exists()],
                    "noUsableViewReasons": dict(Counter(r["reason"].split("fails ")[-1] for r in rejected if r["reason"].startswith("no usable view"))),
                    "resegmented": [e["entityId"] for e in entries if isinstance(e.get("maskSource"), dict)]},
        "objects": entries, "rejected": rejected,
        "spend": {"gpuSeconds": round(gpu, 1), "wallSeconds": round(wall, 1), "estimatedUsd": round(usd + sam_usd, 3), "capUsd": args.max_usd, "dispatches": dispatches,
                  "sam2": {"wallSeconds": round(sam_seconds, 1), "estimatedUsd": round(sam_usd, 4), "rate": "L4 + 1 core + 8 GiB list price x wall seconds"},
                  "sam3d": dict(zip(("wallSeconds", "estimatedUsd"), (round(x, 3) for x in sam3d_spent(args.output / "journal-sam3d")))),
                  "rate": f"USD {USD_PER_SECOND:.7f}/s = Modal list price of A100-80GB + 8 cores + 64 GiB, times local wall seconds (an upper bound on billed time)"},
        "limitations": ["Unseen sides are the generator's estimate; the observed flag only says a video point lies near and a camera had sight of it.",
                        "Observed points come only from views that agree with the best view, so sides seen only from other, drifted views stay inferred.",
                        "Metric scale is assumed (camera height), not measured; model scale follows the posed depth.",
                        "Vertex colours are the generator's own, not re-projected from the video.",
                        "RecGen weights are CC-BY-NC-4.0 and its code TRI non-commercial: research use only."]})


def self_check():
    from ehs_spatial.platform.recgen import source_grid_crop
    from ehs_spatial.platform.spatial import transform_points
    import trimesh
    # clip -> source frame: a clip rectangle lands on its 1.5x image plus the crop offset; both Ks see a point at the same place
    clip = Clip.__new__(Clip)
    clip.clip_to_full = np.array([[1.5, 0, 160.25], [0, 1.5, .25], [0, 0, 1.]])
    clip.full_size, clip.clip_size = (1280, 720), (640, 480)
    k_clip = np.array([[377.5, 0, 319.5], [0, 377.5, 239.5], [0, 0, 1.]])
    clip.k_full = clip.clip_to_full @ k_clip
    mask = np.zeros((480, 640), bool)
    mask[100:200, 300:400] = True
    ys, xs = np.nonzero(clip.full_mask(mask))
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (610, 759, 150, 299), (xs.min(), xs.max(), ys.min(), ys.max())
    assert np.array_equal(clip.clip_mask(clip.full_mask(mask)), mask), "clip -> source -> clip is the identity on a mask"
    view = {"cut": ["left"], "depthShare": .9, "personShare": 0., "captionShare": 0., "area": 2000, "sourceShortSide": 50.}
    assert failing(view) == {"side"} and failing({**view, "cut": ["raster edge"], "area": 900}) == {"raster edge", "small"}
    assert not failing({**view, "cut": [], "area": 900, "sourceShortSide": 80.}), "a compact view is large enough on the source frame"
    assert failing({**view, "cut": [], "personShare": .5}) == {"person"} and not usable([{**view, "cut": ["bottom"]}])
    global BOX_VIEWS
    BOX_VIEWS = True
    assert not failing({**view, "cut": ["left", "top"], "area": 900}) and failing({**view, "depthShare": .2}) == {"depth"}, "box views: only judge-spoiling filters"
    BOX_VIEWS = False
    # box: a yawed 0.4 x 0.2 x 0.6 block's surface points (image y down, so up is -y) give back its size and yaw, closed and outward
    rng = np.random.default_rng(0)
    yaw = np.radians(30)
    turn = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    block = trimesh.creation.box(extents=[.4, .6, .2])
    surface = block.sample(20000, seed=0) @ turn.T + [1, -.3, 2] + rng.normal(0, .002, (20000, 3))
    bv, bf = box_mesh(surface, [0, -1., 0], .02)
    fitted = trimesh.Trimesh(bv, bf)
    assert fitted.is_watertight and abs(fitted.volume - .048) < .003, fitted.volume  # 0.4 x 0.6 x 0.2: size and yaw both right
    assert closest(bv, bf, surface[:2000])[1].max() < .02, "every surface point lies on the fitted box"
    flat = trimesh.Trimesh(*box_mesh(surface[np.abs(surface @ turn[:, 2] - [1, -.3, 2] @ turn[:, 2] - .1) < .003], [0, -1., 0], .02))
    assert flat.is_watertight and .4 * .6 * .019 < flat.volume < .4 * .6 * .03, flat.volume  # a face seen straight on: one step thick
    # cylinder: a drum (radius 0.15, 0.1..0.9 up) seen from one side, with its lid, 10% floor spill and a few stray points
    # 0.6 m above it (a cable), is given back and fits it better than a box does; a 0.3 m square post seen from the same
    # side (two faces and its top) fits a cylinder at least twice worse than the drum and worse than its own box (medians
    # of the held-out half). A flat face gets no cylinder.
    arc, rise = rng.uniform(np.pi + .2, 2 * np.pi - .2, 4000), rng.uniform(.1, .9, 4000)
    side = np.c_[1 + .15 * np.cos(arc), -rise, 2 + .15 * np.sin(arc)] + rng.normal(0, .002, (4000, 3))
    spoke, around = .15 * np.sqrt(rng.uniform(0, 1, 800)), rng.uniform(0, 2 * np.pi, 800)
    lid = np.c_[1 + spoke * np.cos(around), np.full(800, -.9), 2 + spoke * np.sin(around)]
    spill = np.r_[np.c_[rng.uniform(.7, 1.3, 400), np.full(400, -.1), rng.uniform(1.6, 1.84, 400)], np.tile([1, -1.5, 1.9], (20, 1))]
    cv, cf = cylinder_mesh(np.r_[side[1::2], lid, spill], [0, -1., 0], .02)  # shaped by half the side, judged by the other half
    across = np.hypot(cv[:, 0] - 1, cv[:, 2] - 2)
    assert trimesh.Trimesh(cv, cf).is_watertight and abs(across.max() - .15) < .005 and np.abs(np.sort(-cv[:, 1])[[0, -1]] - [.1, .9]).max() < .02, (across.max(), cv[:, 1].min())
    held_out = lambda shape, points: np.median(closest(*shape(points[1::2], [0, -1., 0], .02), points[::2])[1])
    drum_fit = np.median(closest(cv, cf, side[::2])[1])
    post = trimesh.creation.box(extents=[.3, .8, .3])
    samples, index = trimesh.sample.sample_surface_even(post, 12000, seed=1)
    normals = post.face_normals[index] @ turn.T
    front = (normals[:, 2] < -.1) | (normals[:, 1] < -.5)  # the faces a camera at -z and above sees
    samples = samples[front] @ turn.T + [1, -.5, 2] + rng.normal(0, .002, (front.sum(), 3))
    post_fit = held_out(cylinder_mesh, samples)
    assert drum_fit < .005 and post_fit > 2 * drum_fit and held_out(box_mesh, samples) < post_fit, (drum_fit, post_fit)
    assert held_out(box_mesh, np.r_[side, side]) > 2 * drum_fit, "a box fits the drum worse than its cylinder"
    assert cylinder_mesh(surface[np.abs(surface @ turn[:, 2] - [1, -.3, 2] @ turn[:, 2] - .1) < .003], [0, -1., 0], .02) is None, "a flat face is no arc"
    point = np.array([.3, -.2, 2.])
    p_clip, p_full = k_clip @ point, clip.k_full @ point
    assert np.allclose(clip.clip_to_full @ (p_clip / p_clip[2]), p_full / p_full[2])
    # crop: a tilted plane's raster depth carried onto the source-frame crop lifts back onto the same plane
    k_raster = np.array([[415.26, 0, 319.45], [0, 402.68, 239.47], [0, 0, 1.]])
    v, u = np.indices((480, 640))
    normal, offset = np.array([.2, -.1, 1.]), 2.
    rays = np.stack([(u - k_raster[0, 2]) / k_raster[0, 0], (v - k_raster[1, 2]) / k_raster[1, 1], np.ones(u.shape)], -1)
    depth = (offset / (rays @ normal)).astype(np.float32)
    full_mask = np.zeros((720, 1280), bool)
    full_mask[300:420, 500:700] = True
    crop = source_grid_crop(np.zeros((720, 1280, 3), np.uint8), full_mask, depth, k_raster, k_raster @ np.linalg.inv(clip.k_full))
    cv_, cu = np.nonzero(crop["depth"] > 0)
    z = crop["depth"][cv_, cu]
    lifted = np.stack([(cu - crop["K"][0][2]) / crop["K"][0][0] * z, (cv_ - crop["K"][1][2]) / crop["K"][1][1] * z, z], 1)
    assert crop["mask"].sum() == full_mask.sum() and np.abs(lifted @ normal - offset).max() < .02, np.abs(lifted @ normal - offset).max()
    # observed vs inferred: a 2 cm slab seen from the front only. Its back face lies within 2 voxels of the front points
    # but behind the slab, so it must stay inferred while the front face counts as observed.
    slab = trimesh.creation.box(extents=[.4, .3, .02]).subdivide().subdivide().subdivide()
    camera = np.eye(4)
    camera[2, 3] = -1.5
    grid = np.stack(np.meshgrid(np.linspace(-.19, .19, 40), np.linspace(-.14, .14, 30)), -1).reshape(-1, 2)
    flags = seen(slab.vertices, slab.faces, np.column_stack([grid, np.full(len(grid), -.01)]), [camera], k_raster)
    z = slab.vertices[:, 2]
    assert flags[z < -.009].all() and not flags[z > .009].any(), "front face observed, back face inferred"
    thin = trimesh.creation.box(extents=[.4, .3, .002]).subdivide().subdivide().subdivide()  # thinner than OCCLUSION: only facing tells
    flags = seen(thin.vertices, thin.faces, np.column_stack([grid, np.full(len(grid), -.001)]), [camera], k_raster)
    x, y, z = np.asarray(thin.vertices).T  # the rim's side-face vertices face sideways, away from the camera: neither side
    assert flags[(z < -5e-4) & (np.abs(x) < .199) & (np.abs(y) < .149)].all() and not flags[z > 5e-4].any(), "a thin part's back is never seen"
    # fit: a known similarity is undone from the front, top and both side faces of a box (all seven degrees of freedom seen)
    box = trimesh.creation.box(extents=[.4, .3, .2])
    samples, index = trimesh.sample.sample_surface_even(box, 4000, seed=1)
    samples = samples[box.face_normals[index][:, 2] < .5]  # everything but the back face
    truth = trimesh.transformations.compose_matrix(scale=[1.1] * 3, angles=[.05, -.08, .03], translate=[.03, -.02, .04])
    moved = transform_points(box.vertices, truth)
    _, result = fit(moved, box.faces, samples)
    assert np.abs(result - box.vertices).max() < 2e-3, np.abs(result - box.vertices).max()
    eye = np.array([0, 0, -2.])  # with a source camera the centroid may only move along that camera's ray
    _, result = fit(moved, box.faces, samples, eye=eye)
    start = (moved.mean(0) - eye) / np.linalg.norm(moved.mean(0) - eye)
    assert np.linalg.norm(np.cross(start, result.mean(0) - eye)) < 1e-9
    assert np.allclose(fit(box.vertices, box.faces, np.zeros((50, 3)))[0], np.eye(4)), "a degenerate registration ends the fit, not the run"
    # GLB: one buffer, BLEND material, RGBA read back unchanged; the plain variant has no material
    rgba = np.column_stack([np.full((len(slab.vertices), 3), 200), np.where(flags, OBSERVED_ALPHA, INFERRED_ALPHA)]).astype(np.uint8)
    blended, plain = glb(slab.vertices, slab.faces, rgba, True), glb(slab.vertices, slab.faces, rgba, False)
    spec = json.loads(blended[20:20 + struct.unpack_from("<I", blended, 12)[0]])
    assert spec["materials"][0]["alphaMode"] == "BLEND" and len(spec["buffers"]) == 1 and struct.unpack_from("<I", blended, 8)[0] == len(blended)
    assert "materials" not in json.loads(plain[20:20 + struct.unpack_from("<I", plain, 12)[0]])
    back = trimesh.load(trimesh.util.wrap_as_stream(blended), file_type="glb", force="mesh", process=False)
    assert np.allclose(back.vertices, slab.vertices, atol=1e-6)
    back = trimesh.load(trimesh.util.wrap_as_stream(plain), file_type="glb", force="mesh", process=False)
    assert np.array_equal(back.visual.vertex_colors, rgba)
    assert matches("milling machine guard", RELEVANT["machine"]) and not matches("manual", EXCLUDED) and matches("light fixture", EXCLUDED)
    captioned = np.zeros((480, 640, 3), np.uint8)
    captioned[440:455, 200:420] = 255
    assert subtitle_box(captioned) == [194, 434, 425, 460] and subtitle_box(np.zeros_like(captioned)) is None
    plane, patch = np.full((480, 640), 2., np.float32), np.zeros((480, 640), bool)
    patch[100:300, 200:400] = True
    points = lift(plane, k_raster, np.eye(4), patch)
    assert agreement(points, plane, patch, k_raster, np.eye(4)) == (1., 0.)
    assert abs(agreement(points * 1.1, plane, patch, k_raster, np.eye(4))[1] - .1) < 1e-6, "a view 10% deeper disagrees by 10%"
    plane_depth = np.full((48, 64), 2.)
    plane_depth[:4] = 0  # no depth: NaN in the pointmap
    pointmap = sam3d_pointmap(plane_depth, k_raster)
    assert np.isnan(pointmap[:4]).all() and (pointmap[4:, :, 2] == 2).all() and (pointmap[4:, 40, 0] > 0).all(), "PyTorch3D: a pixel left of the principal point has positive x"
    c2w_test = np.eye(4)
    c2w_test[:3, :3], c2w_test[:3, 3] = trimesh.transformations.rotation_matrix(.3, [0, 1, 0])[:3, :3], [1, 2, 3]
    grid_points = lift(plane_depth, k_raster, c2w_test)  # OpenCV lift of the valid pixels, row-major
    assert np.allclose(sam3d_to_world(pointmap[4:].reshape(-1, 3), np.eye(4), c2w_test), grid_points), "pointmap -> world agrees with lift()"
    assert frame_set(["14-225", "3"]) == frozenset(range(14, 226)) | {3} and frame_set([]) == frozenset()
    speck = np.zeros((50, 50), bool)
    speck[5:30, 5:30], speck[40:42, 40:42], speck[5:30, 33:40] = True, True, True
    assert main_parts(speck).sum() == 25 * 25 + 25 * 7, "the 2x2 speck goes, the 25x7 part split off by an occluder stays"
    assert box_iou([[0] * 3, [1] * 3], [[0] * 3, [1] * 3]) == 1 and box_iou([[0] * 3, [1] * 3], [[.9] * 3, [1.9] * 3]) < .01
    # tries: highest score first, every view RETRY_GAP frames from each one taken before it
    scored = [{"frame": f, "score": s} for f, s in ((100, 10), (110, 9), (140, 8), (131, 7.5), (60, 5), (200, 1))]
    assert [m["frame"] for m in spread(scored)] == [100, 140, 60, 200] and [m["frame"] for m in spread(scored, 2)] == [100, 140]
    # journal: a call is found under the transport's own name; an earlier run's entity-level journal of the same input
    # (unresolved first dispatch, received re-send) is read back, never sent again; another seed is another input
    import tempfile
    payload = {"entityId": "e", "anchorObservationId": "o", "seed": SEED, "views": [{
        "observationId": "o", "observationRevision": 1, "imageId": "i", "imageSha256": "0" * 64, "maskSha256": "0" * 64,
        "geometrySolutionSha256": "0" * 64, "coordinateFrameId": "f", "cameraToWorld": np.eye(4).tolist(), **crop}]}
    seeded, resent, fid = {**payload, "seed": EXTRA_SEED}, {**payload, "_researchProtocol": {"dispatchAttemptId": "redispatch-1"}}, "fu-check"
    name = call_identity(payload, fid)
    assert name == call_identity(payload, fid) and len({name, call_identity(seeded, fid), call_identity(resent, fid), call_identity(payload, "fu-other")}) == 4
    real = globals()["request"]
    with tempfile.TemporaryDirectory() as tmp:
        journal, calls = Path(tmp), []
        assert journal_folder(journal, payload, 7, fid) == journal / "e" / "view-00007"
        for folder, stem, record in ((journal / "e", name, "dispatch.json"), (journal / "e" / "redispatch-1", call_identity(resent, fid), "manifest.json")):
            (folder / stem).mkdir(parents=True)
            (folder / stem / record).write_text("{}")
        assert journal_folder(journal, payload, 7, fid) == journal / "e" and journal_folder(journal, seeded, 7, fid) == journal / "e" / f"view-00007-seed-{EXTRA_SEED}"
        globals()["request"] = lambda attempt, folder, _: calls.append((attempt.get("_researchProtocol"), folder)) or ("mesh", {})
        try:
            args = argparse.Namespace(function_id=fid, invoke=False)
            assert obtain(payload, journal / "e", args) == ("mesh", {}) and calls == [({"dispatchAttemptId": "redispatch-1"}, journal / "e" / "redispatch-1")]
            assert obtain(seeded, journal_folder(journal, seeded, 7, fid), args) == (None, None) and len(calls) == 1
        finally:
            globals()["request"] = real
    print("complete_video_objects self-check passed: clip->source mask and K, crop depth lifts onto its plane, "
          "slab back face inferred / front observed, Sim3 fit recovered, BLEND and plain GLB round-trip RGBA, captions found, box IoU, "
          "drum cylinder recovered through lid and spill, square post fits it worse, flat face no cylinder, "
          "small on the source frame, spread tries, journal found by call identity and read back")


def main():
    global VOXEL, OCCLUSION, BOX_VIEWS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true")
    for name in ("droid-run", "depth-run", "object-map", "masks", "dynamic-masks", "clip", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--entities", nargs="*", default=[], help="ENTITY or ENTITY=review note, in priority order, instead of the selection rule's order")
    parser.add_argument("--exclude", nargs="*", default=[], help="ENTITY=reason: dropped at review, reason recorded")
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--all", action="store_true", help="after --entities, every named physical object of the qualifying rule (no count limit)")
    parser.add_argument("--generator", choices=("recgen", "sam3d", "box", "cylinder"), default="recgen",
                        help="recgen: the deployed RecGen (non-commercial); sam3d: self-hosted SAM 3D Objects (modal deploy modal_apps/sam3d_research.py first); "
                             "box: a gravity-aligned box of the video's own points; cylinder: an upright cylinder of them (both CPU only, same gate)")
    parser.add_argument("--reassess", action="store_true", help="judge every journaled model again under the current gate (CPU; no call without --invoke)")
    parser.add_argument("--invoke", action="store_true", help="request missing models; without it only inputs, selection sheet and existing models are processed")
    parser.add_argument("--function-id", default="fu-Hh2leT3x1kprDaWpWsZ09l", help="the deployed generate_object the transport must find")
    parser.add_argument("--max-usd", type=float, default=30.)
    parser.add_argument("--skip-frames", nargs="*", default=[], metavar="A-B", help="frame ranges never used as a view or for points (e.g. a cut-away shot whose cameras are wrong)")
    parser.add_argument("--voxel-native", type=float, default=VOXEL, help="the depth run's fused voxel in native units (fuse-metrics.json voxel_native): observed and fit "
                        "tolerances are counted in it (default: ME340's)")
    parser.add_argument("--no-captions", action="store_true", help="the video has no burned-in captions: skip the white-text caption test (a bright floor sets it off)")
    args = parser.parse_args()
    BOX_VIEWS = args.generator in ("box", "cylinder")
    VOXEL, OCCLUSION = args.voxel_native, args.voxel_native / 2  # ponytail: module constants re-set once per run, as every helper reads them
    FIT_GATE["max_fit_median_native"] = VOXEL
    if args.self_check:
        return self_check()
    missing = [n for n in ("droid_run", "depth_run", "object_map", "masks", "dynamic_masks", "clip", "output") if getattr(args, n) is None]
    if missing:
        parser.error("missing " + ", ".join("--" + m.replace("_", "-") for m in missing))
    args.skip = frame_set(args.skip_frames)
    run(args)


if __name__ == "__main__":
    main()
