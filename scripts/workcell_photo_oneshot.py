"""One command: N >= 2 raw photos of ONE workcell scene -> calibrated candidate and interactive 3D report.

Every photo must show the same physical scene (one cell): MapAnything fuses them into one world.
The reference photo (--reference-photo K, 1-based, default the last) fixes the left/right
convention, the post primitives and the emergency-button scale reference.
Heavy stages share one ephemeral two-A100 Modal container, with GPU 0 for
geometry/OWLv2 and GPU 1 for SAM 3, then both GPUs for RecGen (one multi-view model per object).
Run with the project venv: python scripts/workcell_photo_oneshot.py --images a.jpg b.jpg --out NEW_DIR --viewer-assets THREE_0_178_0_DIR
"""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import trimesh


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# Three.js 0.178.0 npm package, MIT. See docs/workcell-photo/HANDOFF.md for acquisition.
VIEWER_ASSET_SHA256 = {
    "LICENSE": "bfe119ea4fd413f5f7ca3fcd63adb0c4a073ed39daa2fe7d3e6b769e21272601",
    "three.core.js": "562b72799ef1145f77997ece49a34f578422873757b0a13e41d76dcbfb776f06",
    "three.module.js": "bc0d236927f5163414e7c59a5567257dfe925f1929ce0a151ac4185dc45ca5a2",
    "addons/controls/OrbitControls.js": "b97879c748170baadeb3fb84cea1ffdf4674e283dc06042f34e2acb95a76042c",
    "addons/loaders/GLTFLoader.js": "caba6c51cfd8c7d5313bd7705a54b76bc0a7199d9822ecc497c5311eaffe8e5e",
    "addons/utils/BufferGeometryUtils.js": "cbcfe1864abedcc0122cb893373918fe14469491717a34ca6efcc38551805765",
    "addons/exporters/GLTFExporter.js": "3d91af558632f8ced2ac9bb4230f108c4004ef82966338ae001b9fa84be59550",
}
GUARD_WORD = "yellow and black striped panel"
WORDS = ("industrial robot arm", "safety fence", "yellow safety post", "black bollard",
         "emergency stop button", "red emergency stop switch", "light curtain",
         "work platform", "cart", "control cabinet", "signal light", "stack light",
         "warning sign", "workcell sign", "folding safety barrier", "cable tray",
         "instruction poster", "transparent safety panel", "floor marking", GUARD_WORD,
         # Entrance gantry members and its top pipe (scripts/workcell_gantry.GANTRY_WORDS; prompt probe 2026-10-04).
         "white steel beam", "blue pipe")


CAPTURE = "capture.json"


def capture_record(images, reference_photo=None):
    """The scene's photo set: photo i is images[i-1]; one reference photo (default the last)."""
    images = [Path(p) for p in images]
    if len(images) < 2 or len({p.resolve() for p in images}) != len(images) or not all(p.is_file() for p in images):
        raise ValueError("At least two distinct, readable photos of one scene are required")
    reference = len(images) if reference_photo in (None, 0) else reference_photo
    if type(reference) is not int or not 1 <= reference <= len(images):
        raise ValueError(f"--reference-photo must name one of photos 1..{len(images)}")
    return {"schemaVersion": 1, "photoCount": len(images), "referencePhoto": reference,
            "referenceRole": "left/right convention, post primitives and emergency-button scale reference",
            "scope": "all photos show one physical scene; MapAnything fuses them into one world",
            "sources": [{"photo": i, "name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for i, p in enumerate(images, 1)]}


def scene_photos(root):
    """(photo count, reference photo) of a run: capture.json, else a legacy run's frames 1..N with N the reference."""
    path = Path(root) / CAPTURE
    if path.is_file():
        record = json.loads(path.read_text())
        count, reference = record["photoCount"], record["referencePhoto"]
    else:
        # ponytail: pre-capture runs were four photos with photo 4 the reviewer's view; frames are numbered 1..N.
        count = reference = max((int(p.name[6:10]) for p in Path(root).glob("frame_*.json.gz")), default=0)
    if type(count) is not int or type(reference) is not int or count < 1 or not 1 <= reference <= count:
        raise ValueError("Run has no valid photo set (capture.json or frame_0001..N)")
    return count, reference


def _array(spec):
    return np.frombuffer(base64.b64decode(spec["data"]), np.dtype(spec["dtype"])).reshape(spec["shape"])


def _frame(root, index):
    with gzip.open(root / f"frame_{index:04d}.json.gz", "rt") as stream:
        return json.load(stream)


def _mask(frame, response):
    shape = frame["image"]["shape"][:2]
    affine = np.asarray(frame["input_mask_transform"]["input_to_canonical_pixel_centres"], float)[:2]
    mask = np.zeros(shape, bool)
    for encoded in response.get("rle", []):
        item = json.loads(encoded)
        counts = np.asarray(item["counts"], np.int64)
        if counts.sum() != np.prod(item["size"]):
            raise ValueError("SAM 3 mask RLE length mismatch")
        flat = np.repeat(np.arange(len(counts), dtype=np.uint8) % 2, counts)
        decoded = flat.reshape(item["size"], order="F")
        mask |= (decoded.astype(bool) if list(item["size"]) == list(shape) else
                 cv2.warpAffine(decoded, affine, (shape[1], shape[0]),
                                flags=cv2.INTER_NEAREST).astype(bool))
    return mask


def _response(seg, index, word):
    j = next(j for j, prompt in enumerate(seg["prompts"]) if prompt["text"] == word)
    return seg["results"][index - 1][j]


def _job(name, argv, out):
    started = time.monotonic()
    result = subprocess.run([sys.executable, "-m", "modal", "run", *argv], cwd=REPO,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=1800, check=False)
    elapsed = time.monotonic() - started
    run_ids = re.findall(r"ap-[A-Za-z0-9]+", result.stdout)
    record = {"stage": name, "wallSeconds": round(elapsed, 2), "runIds": sorted(set(run_ids)),
              "returncode": result.returncode, "result": str(out)}
    (out / "modal-call.json").write_text(json.dumps(record, indent=2))
    if result.returncode:
        safe = [line for line in result.stdout.splitlines()
                if not re.search(r"capabilit|token|secret", line, re.I)]
        raise RuntimeError(f"{name} failed ({result.returncode}): " + "\n".join(safe[-10:])[-1800:])
    return record


ROBOT_MIN_PIXELS, CART_MIN_PIXELS, GUARD_MIN_PIXELS = 500, 2500, 100
RECGEN_MAX_VIEWS = 4  # fast_report.recgen_fast DEFAULT max_views: RecGen conditions on the first four views, anchor first


def _prepare_inputs(root, seg, cart_seg):
    """Per-view RecGen inputs of one scene; a view that does not show an object keeps an empty mask."""
    robot, cart, guard, robot_picks, cart_picks, guard_picks = {}, {}, {}, [], [], []
    floor = json.loads((root / "floor-reference.json").read_text())
    ground_n = np.asarray(floor["normal"])
    ground_d = floor["offset"]
    for i in range(1, scene_photos(root)[0] + 1):
        frame = _frame(root, i)
        rgb, points = _array(frame["image"]).copy(), _array(frame["pts3d"])
        pose, K = _array(frame["camera_poses"]), _array(frame["intrinsics"])
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        depth = local[..., 2].astype(np.float32)
        good = _array(frame["non_ambiguous_mask"]).astype(bool) & np.isfinite(depth) & (depth > 0) & (depth < 30)
        yy, xx = np.indices(good.shape)
        u = K[0, 0] * local[..., 0] / np.maximum(depth, 1e-6) + K[0, 2]
        v = K[1, 1] * local[..., 1] / np.maximum(depth, 1e-6) + K[1, 2]
        if np.nanmedian(np.hypot(u[good] - xx[good], v[good] - yy[good])) >= 3:
            raise ValueError(f"photo {i}: camera/point projection mismatch")
        robot_mask = _mask(frame, _response(seg, i, "industrial robot arm")) & good
        if robot_mask.sum() < ROBOT_MIN_PIXELS:
            robot_mask[:] = False  # this view does not show the robot well enough to condition or place its model
        guard_response = _response(seg, i, GUARD_WORD)
        panels = []
        for encoded, confidence in zip(guard_response['rle'], guard_response['scores']):
            candidate = _mask(frame, {'rle':[encoded]}) & good
            if confidence < .6 or not candidate.any():
                continue
            heights = points[candidate] @ ground_n + ground_d
            if np.median(heights) <= 3 * floor['residualP95Native']:
                continue  # Floor hazard stripes are a separate physical object.
            panels.append(candidate)
        # ponytail: retain substantial supported faces; tiny striped fittings are
        # not evidence for the large folded guard. Instance-level routing can
        # replace this relative-area gate when smaller guards are in scope.
        largest = max((int(m.sum()) for m in panels), default=0)
        panels = [m for m in panels if m.sum() >= .2 * largest]
        guard_mask = np.logical_or.reduce(panels) if panels else np.zeros_like(good)
        cart_response = cart_seg["results"][i - 1]
        choices = []
        for k, (rle, score) in enumerate(zip(cart_response["rle"], cart_response["scores"])):
            mask = _mask(frame, {"rle": [rle]}) & good
            retained = mask & ~guard_mask
            guard_fraction = (mask & guard_mask).sum() / max(1, mask.sum())
            mask = retained
            overlap = (mask & robot_mask).sum() / max(1, mask.sum())
            if mask.sum() > CART_MIN_PIXELS and guard_fraction < .5 and overlap < .25:
                choices.append((score, k, mask, overlap, guard_fraction))
        guard_fraction = None
        if choices:
            score, picked, cart_mask, overlap, guard_fraction = max(choices, key=lambda row: row[0])
        else:  # this view shows no cart mask separate from robot and guard
            score, picked, cart_mask, overlap = None, None, np.zeros_like(good), None
        guard_picks.append({"photo": i, "supportedPixels": int(guard_mask.sum()),
                            "source": GUARD_WORD, "selectedPanels": len(panels), "routing": "score >= .6; above floor residual; area >= .2 largest supported panel",
                            "excludedFromCartFraction": None if guard_fraction is None else float(guard_fraction)})
        cart_picks.append({"photo": i, "instance": picked, "score": score,
                           "robotOverlap": None if overlap is None else round(float(overlap), 3), "supportedPixels": int(cart_mask.sum())})
        robot_picks.append({"photo": i, "supportedPixels": int(robot_mask.sum()), "source": "industrial robot arm",
                            "routing": f"union of SAM instances; at least {ROBOT_MIN_PIXELS} valid pixels, else not visible"})
        shared = {f"v{i}_rgb": rgb, f"v{i}_depth": np.where(good, depth, 0),
                  f"v{i}_K": K, f"v{i}_c2w": pose}
        robot.update(shared | {f"v{i}_mask": robot_mask.astype(np.uint8) * 255})
        cart.update(shared | {f"v{i}_mask": cart_mask.astype(np.uint8) * 255})
        guard.update(shared | {f"v{i}_mask": guard_mask.astype(np.uint8) * 255})
    if not _visible_views(robot_picks, ROBOT_MIN_PIXELS):
        raise ValueError(f"No photo shows {ROBOT_MIN_PIXELS} valid robot pixels")
    if not _visible_views(cart_picks, CART_MIN_PIXELS):
        raise ValueError("No photo shows a cart mask separate from robot and guard")
    np.savez_compressed(root / "robot-input.npz", **robot)
    np.savez_compressed(root / "cart-input.npz", **cart)
    np.savez_compressed(root / "guard-input.npz", **guard)
    (root / "robot-mask-selection.json").write_text(json.dumps(robot_picks, indent=2) + "\n")
    (root / "guard-mask-selection.json").write_text(json.dumps(guard_picks, indent=2) + "\n")
    (root / "cart-mask-selection.json").write_text(json.dumps(cart_picks, indent=2) + "\n")


def _visible_views(rows, minimum):
    """Photos with at least `minimum` supported mask pixels, most support first.

    RecGen's shape/appearance conditioning anchors on the first view; occluded
    capture-order views erased the guard's connecting face, so rank actual support."""
    return [r['photo'] for r in sorted(rows, key=lambda r: (-r['supportedPixels'], r['photo'])) if r['supportedPixels'] >= minimum]


def recgen_plans(root):
    """One multi-view RecGen model per object; returns (per-GPU job lists, views each model is placed against).

    Robot and cart: one group over every view that shows them (RecGen reads the first four, anchor first), placed
    against all of those views. Guard: its two best-supported views, which also place it (unchanged recipe)."""
    rows = {kind: json.loads((Path(root) / f"{kind}-mask-selection.json").read_text()) for kind in ("robot", "cart", "guard")}
    views = {"robot": _visible_views(rows["robot"], ROBOT_MIN_PIXELS), "cart": _visible_views(rows["cart"], CART_MIN_PIXELS)}
    generation = {kind: views[kind][:RECGEN_MAX_VIEWS] for kind in views}
    generation["guard"] = views["guard"] = _visible_views(rows["guard"], GUARD_MIN_PIXELS)[:2]
    if len(generation["guard"]) < 2:
        raise ValueError("V-guard lacks independent support in at least two views")
    plans = [[("robot", [["multi", generation["robot"]]])],
             [("cart", [["multi", generation["cart"]]]), ("guard", [["multi", generation["guard"]]])]]
    return plans, views


def place_model(mesh, inputs, generation, refine_views):
    """Place one RecGen mesh (anchor-camera frame) in the scene world, then similarity-refine it (fast_report.x7.refine,
    shape-preserving Sim(3)) against the masks and depth of refine_views; score its mask IoU in every photo."""
    from fast_report.x7 import Caster, light, rays, refine, score_view
    mesh = mesh.copy()
    mesh.apply_transform(inputs[f"v{generation[0]}_c2w"])
    v, f, _ = light(mesh.vertices, mesh.faces, mesh.visual.vertex_colors[:, :3])
    photos = sorted(int(key[1:-5]) for key in inputs.keys() if key.endswith("_mask"))
    views = {}
    for i in photos:
        depth = inputs[f"v{i}_depth"][::2, ::2]
        K = inputs[f"v{i}_K"].copy(); K[:2] /= 2
        views[i] = {"rays": rays(K, inputs[f"v{i}_c2w"], depth.shape[1], depth.shape[0]),
                    "target": inputs[f"v{i}_mask"][::2, ::2] > 0, "depth": depth}
    transform, placement = refine(v, f, [views[i] for i in refine_views], uniform_scale=True)
    mesh.apply_transform(transform)
    caster = Caster(v, f)
    placement.update(generationViews=list(generation), refineViews=list(refine_views), transform=transform.tolist(),
                     sourceChecks=[{"photo": i, "targetPixels": int(views[i]["target"].sum()), **score_view(caster, transform, views[i])}
                                   for i in photos],
                     sourceCheckScope="half-resolution SAM mask of every photo; the model is hidden only where scene depth is in front of it outside the mask",
                     basis="shape-preserving similarity alignment to source masks and estimated depth; not surveyed physical accuracy")
    return mesh, placement


def _surface(root, seg, name, index, word, mask_override=None):
    frame = _frame(root, index)
    p, color = _array(frame["pts3d"]), _array(frame["image"])
    mask = (_mask(frame, _response(seg, index, word)) if mask_override is None else mask_override)
    mask &= _array(frame["non_ambiguous_mask"]).astype(bool) & np.isfinite(p).all(2)
    step = 2
    ys, xs = np.arange(0, mask.shape[0], step), np.arange(0, mask.shape[1], step)
    points, rgb, keep = p[np.ix_(ys, xs)], color[np.ix_(ys, xs)], mask[np.ix_(ys, xs)]
    h, w = keep.shape
    ids = np.arange(h * w).reshape(h, w)
    valid = keep[:-1, :-1] & keep[1:, :-1] & keep[:-1, 1:] & keep[1:, 1:]
    edge = np.maximum(np.linalg.norm(points[1:, :-1] - points[:-1, :-1], axis=2),
                      np.linalg.norm(points[:-1, 1:] - points[:-1, :-1], axis=2))
    y, x = np.nonzero(valid & (edge < .35))
    if len(x) < 1000:
        raise ValueError(f"{name}: insufficient observed surface")
    a, b, c, d = ids[y, x], ids[y + 1, x], ids[y + 1, x + 1], ids[y, x + 1]
    mesh = trimesh.Trimesh(vertices=points.reshape(-1, 3),
                           faces=np.r_[np.stack([a, b, c], 1), np.stack([a, c, d], 1)],
                           vertex_colors=rgb.reshape(-1, 3), process=False)
    mesh.remove_unreferenced_vertices()
    (root / f"{name}.glb").write_bytes(mesh.export(file_type="glb"))
    return {"sourcePhoto": index, "vertices": len(mesh.vertices), "faces": len(mesh.faces)}


def _posts(root, seg):
    """Up to two yellow posts and two black bollards from the scene's reference photo (photo 4 in legacy runs)."""
    reference = scene_photos(root)[1]
    frame = _frame(root, reference)
    points, colors = _array(frame["pts3d"]), _array(frame["image"])
    valid = _array(frame["non_ambiguous_mask"]).astype(bool)
    normal = np.asarray(json.loads((root / "geometry.json").read_text())["floor"]["normal"], float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 0:
        raise ValueError("Post models require a finite nonzero ground normal")
    normal /= np.linalg.norm(normal)
    transform = trimesh.geometry.align_vectors([0, 1, 0], normal)
    # ponytail: upright is the existing display prior, not a measured post axis.
    # Fit in the actual ground frame; source multiview edges must replace this prior for metrology.
    local_points = points @ transform[:3, :3]
    scene, records = trimesh.Scene(), []
    for word, kind in (("yellow safety post", "box"), ("black bollard", "cylinder")):
        response = _response(seg, reference, word)
        candidates = []
        for encoded, score in zip(response["rle"], response["scores"]):
            mask = _mask(frame, {"rle": [encoded]}) & valid
            if score >= .8 and mask.sum() >= 200:
                candidates.append((score, mask))
        for number, (score, mask) in enumerate(sorted(candidates, key=lambda row: -row[0])[:2], 1):
            cloud = local_points[mask]
            low, high = np.percentile(cloud, [2, 98], axis=0)
            center = np.median(cloud, axis=0)
            if kind == "box":
                extent = np.maximum((high - low) * [.85, 1, .3], [.07, .2, .07])
                mesh = trimesh.creation.box(extents=extent)
            else:
                radius = max(.04, float(np.median((high - low)[[0, 2]]) / 2))
                mesh = trimesh.creation.cylinder(radius=radius, height=float(high[1] - low[1]), sections=32)
                mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            mesh.apply_translation([center[0], (low[1] + high[1]) / 2, center[2]])
            mesh.apply_transform(transform)
            color = np.median(colors[mask], axis=0).astype(np.uint8)
            mesh.visual.vertex_colors = np.tile(np.r_[color, 255], (len(mesh.vertices), 1))
            label = f"{kind}-{number}"
            scene.add_geometry(mesh, node_name=label, geom_name=label)
            records.append({"name": label, "sourcePhoto": reference, "score": score,
                            "supportedPixels": int(mask.sum()), "kind": kind,
                            "groundNormalNative": normal.tolist(),
                            "axisStatus": "upright display prior; not a measured physical axis"})
    if not records:
        raise ValueError(f"Photo {reference}: no yellow post or black bollard with score >= .8 and 200 pixels")
    (root / "posts.glb").write_bytes(scene.export(file_type="glb"))
    (root / "posts-source.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def _mask_sheet(root, seg, groups, filename, cart=None):
    """Contact sheet of every photo, two per column (four photos: the historical 2 x 2)."""
    tiles = []
    colors = ((20, 80, 255), (0, 210, 20), (240, 100, 20), (200, 0, 180), (0, 180, 220))
    for i in range(1, scene_photos(root)[0] + 1):
        frame = _frame(root, i)
        image = _array(frame["image"]).copy()
        for j, word in enumerate(groups):
            response = cart["results"][i - 1] if cart is not None else _response(seg, i, word)
            for encoded in response["rle"]:
                mask = _mask(frame, {"rle": [encoded]})
                if mask.sum() < 50:
                    continue
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, colors[j % len(colors)], 2)
        header = np.full((28, image.shape[1], 3), 20, np.uint8)
        cv2.putText(header, f"Photo {i}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (255, 255, 255), 1)
        tiles.append(np.vstack([header, image]))
    height, width = max(t.shape[0] for t in tiles), max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, height - t.shape[0]), (0, width - t.shape[1]), (0, 0))) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (len(tiles) % 2)
    sheet = np.hstack([np.vstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    Image.fromarray(sheet).save(root / filename, quality=86)


def _bbox_quality(root, seg, kind):
    """The one multi-view model's projected box against its mask box, in every photo that shows the object."""
    rows = []
    for i in range(1, scene_photos(root)[0] + 1):
        frame = _frame(root, i)
        pose, K = _array(frame["camera_poses"]), _array(frame["intrinsics"])
        if kind == "robot":
            mask = _mask(frame, _response(seg, i, "industrial robot arm"))
        else:
            z = np.load(root / "cart-input.npz")
            mask = z[f"v{i}_mask"] > 0
        yy, xx = np.nonzero(mask)
        if not len(xx):
            rows.append({"photo": i, "model": "multi", "maskBBox": None, "reason": "object not visible in this photo"})
            continue
        truth = np.array([xx.min(), xx.max(), yy.min(), yy.max()], float)
        for model, label in ((f"{kind}-multi", "multi"),):
            verts = trimesh.load(root / f"{model}.glb", force="mesh", process=False).vertices
            local = (verts - pose[:3, 3]) @ pose[:3, :3]
            q = local[:, 2] > .01
            u = K[0, 0] * local[q, 0] / local[q, 2] + K[0, 2]
            v = K[1, 1] * local[q, 1] / local[q, 2] + K[1, 2]
            pred = np.array([*np.percentile(u, [2, 98]), *np.percentile(v, [2, 98])])
            intersection = max(0, min(truth[1], pred[1]) - max(truth[0], pred[0])) * max(0, min(truth[3], pred[3]) - max(truth[2], pred[2]))
            area = lambda b: max(0, b[1] - b[0]) * max(0, b[3] - b[2])
            rows.append({"photo": i, "model": label,
                         "maskBBox": truth.round(1).tolist(), "modelProjectedBBox": pred.round(1).tolist(),
                         "bboxIoU": round(float(intersection / max(1, area(truth) + area(pred) - intersection)), 3),
                         "centerOffsetPx": round(float(np.linalg.norm([pred[:2].mean() - truth[:2].mean(),
                                                                          pred[2:].mean() - truth[2:].mean()])), 1)})
    return rows


SCENE_EXPORTS = ('workcell-conditional.glb', 'workcell-metric.glb', 'workcell-native.glb')


def _export_metric_scene(root, report):
    """Export the exact report assets/poses and its single model measurement scale."""
    from ehs_spatial.platform.spatial import transform_matrix
    root = Path(root)
    measurement_scale = report['modelMeasurementScale']
    scale = measurement_scale['nativeToMeters']
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError("No finite positive reference scale for scene export")
    factor = scale if scale is not None else 1.0
    # Same Z-up -> glTF Y-up rotation as the browser's downloadModel.
    transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    transform[:3, :3] *= factor
    scene = trimesh.Scene()
    document = report['revision']['document']
    frames = {frame['id']: frame for frame in document['coordinateFrames']}
    for item in document['entities']:
        if item.get('sourceContext') or item.get('visible') is False:
            continue
        spec = next((r for r in item['representations'] if r['id'] == item.get('activeModelRepresentationId')
                     and r['kind'] == 'generated_mesh' and r.get('sourceValidity') != 'stale'), None)
        if spec is None:
            continue
        frame = frames[spec['coordinateFrameId']]
        if not np.allclose(frame['ground']['normal'], [0, 0, 1]) or frame['ground']['offset'] != 0:
            raise ValueError('Workcell export requires the report floor Z-up frame')
        placed = transform_matrix(item.get('currentModelTransform') or spec['transform'])
        source = trimesh.load(root / report['assetURLs'][spec['assetId']], force='scene', process=False)
        for node in source.graph.nodes_geometry:
            matrix, geometry_id = source.graph.get(node)
            original = source.geometry[geometry_id]
            mesh = original.copy()
            if original.visual.kind == 'texture' and 'color' in original.visual.vertex_attributes:
                mesh.visual.vertex_attributes['color'] = original.visual.vertex_attributes['color'].copy()
            name = item['id'] + ':' + node
            scene.add_geometry(mesh, node_name=name, geom_name=name, transform=transform @ placed @ matrix)
    scene.metadata.update(units='meters' if scale is not None else 'native', upAxis='Y',
                          metricScaleMPerNative=scale, scale_status=measurement_scale['status'],
                          modelMeasurementScale=measurement_scale, ground={'normal': [0, 1, 0], 'offset': 0},
                          reportRevision=report['revision']['id'], documentSha256=report['revision']['documentSha256'],
                          groundTruth=False)
    name = ('workcell-conditional.glb' if measurement_scale['status'] == 'conditional_unvalidated'
            else 'workcell-metric.glb' if scale is not None else 'workcell-native.glb')
    (root / name).write_bytes(scene.export(file_type='glb'))
    for other in SCENE_EXPORTS:
        # An export under another scale status belongs to an earlier revision; never package it beside this one.
        if other != name and (root / other).is_file():
            (root / other).unlink()
    return name


def _anchor_sheet(root, sources, anchor):
    """Show exactly which physical component the editable dimensions refer to."""
    views = anchor["views"]
    if not views:  # no button reference observed in this scene: nothing to show, scale stays native
        return
    sheet = Image.new("RGB", (160 * len(views), 188), "#152222")
    draw = ImageDraw.Draw(sheet)
    for number, view in enumerate(views):
        with Image.open(sources[view["photo"] - 1]) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            crop = source.crop(tuple(view["boxRaw"]))
            crop.thumbnail((144, 156))
            sheet.paste(crop, (number * 160 + (160 - crop.width) // 2, 8))
        draw.text((number * 160 + 8, 169), f"Photo {view['photo']}", fill="white")
    sheet.save(root / "geometry-anchor.jpg", quality=90)


def _freeze_report_ui(root, viewer_assets):
    built = REPO / "web/dist-photo"
    if not (built / "photo.html").is_file():
        raise FileNotFoundError("Build the shared report UI first: cd web && node node_modules/vite/bin/vite.js build --config vite.photo.config.ts")
    assets = {name: (viewer_assets / name).read_bytes() for name in VIEWER_ASSET_SHA256}
    for name, payload in assets.items():
        if hashlib.sha256(payload).hexdigest() != VIEWER_ASSET_SHA256[name]:
            raise ValueError(f"Viewer asset is not the required Three.js 0.178.0 file: {viewer_assets / name}")
    shutil.copytree(built, root / "report-ui")
    for name, payload in assets.items():
        target = root / "report-ui/viewer-assets" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def _build_page(root, metrics):
    built = root / "report-ui"
    if not (built / "photo.html").is_file() or any(
            not (built / "viewer-assets" / name).is_file() for name in VIEWER_ASSET_SHA256):
        raise FileNotFoundError("Freeze the built report UI and its Three.js assets before packaging")
    page = root / "page"
    page.mkdir()
    shutil.copytree(built, page, dirs_exist_ok=True)
    shutil.copyfile(built / "photo.html", page / "index.html")
    assets = ("guard-multi.glb", "guard-left.glb", "guard-center.glb", "guard-right.glb",
              "guard-partition.json", "cart-observed.glb", "posts.glb", "fence-observed.glb",
              "mask-contact-sheet.jpg", "extra-mask-contact-sheet.jpg", "cart-mask-sheet.jpg",
              "fence-fitted.glb", "floor-fitted.glb", "geometry.json",
              "objects.json", "scene-report.json")
    for name in assets:
        shutil.copyfile(root / name, page / name)
    # One multi-view model per object (robot-multi, cart-multi) and its placement; legacy runs carry per-photo robots.
    for path in [*root.glob("robot-*.glb"), *root.glob("cart-multi.glb"), *root.glob("cart-single.glb"),
                 *root.glob("*-placement.json"), *root.glob(CAPTURE), *root.glob("object-extras.glb")]:
        shutil.copyfile(path, page / path.name)
    for name in ('measurements.json', 'measurement-evaluation.json', 'physical-clearances.json', 'structural-result.json', 'housing-models.json', 'model-endpoint-estimate.json', 'workcell-metric.glb', 'workcell-native.glb', 'workcell-conditional.glb'):
        if (root/name).is_file():
            shutil.copyfile(root/name, page/name)
    for model in [*root.glob("entity-*.glb"), *root.glob("post-box-*-physical.glb"), *root.glob("guard-*-initializer.glb"), *root.glob("structural-*.jpg"), *root.glob("raw-image-features-*.jpg")]:
        shutil.copyfile(model, page / model.name)
    for evidence in [*root.glob("geometry-*.jpg"), *root.glob("geometry-*.png")]:
        shutil.copyfile(evidence, page / evidence.name)
    frames = []
    for i in range(1, scene_photos(root)[0] + 1):
        shutil.copyfile(root / f"photo-{i}.png", page / f"photo-{i}.png")
        frame = _frame(root, i)
        shape = frame["image"]["shape"]
        frames.append({"id": i, "image": f"photo-{i}.png", "width": shape[1], "height": shape[0],
                       "K": _array(frame["intrinsics"]).tolist(),
                       "cameraToWorld": _array(frame["camera_poses"]).tolist()})
    times = metrics["stageTiming"]
    data = {"frames": frames, "quality": metrics["robotQuality"], "cartQuality": metrics["cartQuality"],
            "geometry": json.loads((root/"geometry.json").read_text()),
            "timing": {"oneShotSeconds": metrics["oneShotWallSeconds"],
                       "geometrySeconds": times["geometrySeconds"],
                       "segmentationSeconds": times["segmentationSeconds"],
                       "segmentationExtraSeconds": 0,
                       "owlSeconds": times["owlSeconds"], "cartMaskSeconds": times["cartMaskSeconds"],
                       "cartModelSeconds": times["modelSeconds"], "modelSeconds": times["modelSeconds"]}}
    (page / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    shared = json.loads((page / "scene-report.json").read_text())
    shared["timing"] = data["timing"]
    from scripts.workcell_policy_evidence import policy_evidence
    shared["policyEvidence"] = policy_evidence(shared, json.loads((root / "objects.json").read_text())["objects"])
    if shared.get("semanticExperiment"):
        from scripts.workcell_semantic_report import package
        package(root, page, shared["semanticExperiment"])
    (page / "scene-report.json").write_text(json.dumps(shared, ensure_ascii=False, indent=2) + "\n")
    return page


def _self_check():
    assert np.array_equal(_mask({"image": {"shape": [2, 2, 3]},
                                 "input_mask_transform": {"input_to_canonical_pixel_centres": np.eye(3).tolist()}},
                                {"rle": [json.dumps({"size": [2, 2], "counts": [0, 2, 1, 1]})]}),
                          np.array([[1, 0], [1, 1]], bool))
    assert len(WORDS) == len(set(WORDS))
    print("workcell_photo_oneshot self-check passed")


def _semantic_stage(out, protocol, ledger):
    """Semantics run on this exact finished revision; the shared tail then binds them."""
    from scripts.workcell_photo_report import finalize
    destination = out / "semantic-experiment"
    destination.mkdir()
    try:
        ledger["runs"].append(_job("semantic", ["modal_apps/workcell_semantic_match.py", "--root", str(out),
                                                "--out", str(destination), "--config", str(protocol)], destination))
    except Exception as error:  # Spend and failure stay recorded; the report says semantics are not bound.
        ledger["semanticFailure"] = str(error)
    if (destination / "spend-ledger.json").is_file():
        ledger["semanticLedger"] = json.loads((destination / "spend-ledger.json").read_text())
    (out / "spend-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    return finalize(out)


def run(images, out, diameter_m, height_m, viewer_assets, measurements=None, semantic_protocol=None, reference_photo=None):
    from scripts.workcell_photo_calibration import load_measurements, resolve_dimensions
    measured = load_measurements(measurements) if measurements else None
    if semantic_protocol is not None:
        from scripts.workcell_semantic_match import config_checked, read_json
        config_checked(read_json(semantic_protocol))
    diameter_m, height_m = resolve_dimensions(measured, diameter_m, height_m)
    capture = capture_record(images, reference_photo)  # N >= 2 distinct readable photos of one scene
    if not all(np.isfinite(v) and v > 0 for v in (diameter_m, height_m)):
        raise ValueError("Button dimensions must be finite and positive")
    if out.exists():
        raise ValueError("Output must be a new directory; a one-shot run never mutates earlier evidence")
    out.mkdir(parents=True)
    (out / CAPTURE).write_text(json.dumps(capture, indent=2) + "\n")
    # Freeze the built viewer before compute; concurrent rebuilds must not change a running report.
    _freeze_report_ui(out, viewer_assets)
    began = time.monotonic()
    ledger = {"hardware": "one ephemeral 2 x A100-80GB container", "mode": "ephemeral modal run",
              "actualBilledUsd": None, "runs": []}
    paths = ",".join(str(p) for p in images)
    reference_args = []
    if measured:
        reference_path = out / "reference-input.json"
        reference_path.write_text(json.dumps(measured['reference']))
        reference_args = ['--reference', str(reference_path)]
    def estimate():
        # A failed cloud stage still returns its timing: its spend is recorded the same way.
        if not (out / "modal-timing.json").is_file():
            return
        modal_timing = json.loads((out / "modal-timing.json").read_text())
        usd_per_second = 2 * .000694 + 16 * .0000131 + 80 * .00000222
        ledger["estimate"] = {
            "usdPerSecond": round(usd_per_second, 7),
            "functionWindowEstimateUsd": round(usd_per_second * modal_timing["containerWallSeconds"], 3),
            "callWindowEstimateUsd": round(usd_per_second * modal_timing["wallSecondsIncludingColdStart"], 3),
            "rateSource": "https://modal.com/pricing",
            "rateCheckedDate": "2026-09-30",
            "basis": "reserved-resource list-rate estimates, not invoice amounts; call window includes possible scheduling time"}
    try:
        record = _job("one-container", ["modal_apps/workcell_photo_all.py", "--images", paths,
                  "--out", str(out), "--words", ",".join(WORDS), "--reference-photo", str(capture["referencePhoto"]),
                  "--button-diameter-m", str(diameter_m), "--button-height-m", str(height_m), *reference_args], out)
    except Exception as error:
        ledger["failure"] = str(error)
        if (out / "modal-call.json").is_file():
            ledger["runs"].append(json.loads((out / "modal-call.json").read_text()))
        estimate()
        (out / "spend-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
        raise
    ledger["runs"].append(record)
    estimate()
    (out / "spend-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    local, tick = {}, time.monotonic()
    def lap(name):
        nonlocal tick
        local[name] = round(time.monotonic() - tick, 2)
        tick = time.monotonic()
    seg = json.loads((out / "sam3.json").read_text())
    cart_seg = json.loads((out / "cart-masks.json").read_text())
    count, reference = capture["photoCount"], capture["referencePhoto"]
    if len(seg["results"]) != count or len(cart_seg["results"]) != count:
        raise ValueError(f"Modal result must include all {count} photos")
    robot_quality = _bbox_quality(out, seg, "robot")
    cart_quality = _bbox_quality(out, seg, "cart")
    (out / "bbox-eval.json").write_text(json.dumps(robot_quality, indent=2) + "\n")
    (out / "cart-bbox-eval.json").write_text(json.dumps(cart_quality, indent=2) + "\n")
    cart_z = np.load(out / "cart-input.npz")
    cart_view = json.loads((out / "cart-placement.json").read_text())["generationViews"][0]
    surfaces = {"cart": _surface(out, seg, "cart-observed", cart_view, "cart", cart_z[f"v{cart_view}_mask"] > 0)}
    for photo in [reference, *(i for i in range(1, count + 1) if i != reference)]:
        try:  # the reference photo first, then the others: the first with enough observed fence surface
            surfaces["fence"] = _surface(out, seg, "fence-observed", photo, "safety fence")
            break
        except ValueError:
            continue
    else:
        raise ValueError("fence-observed: insufficient observed fence surface in every photo")
    posts = json.loads((out / "posts-source.json").read_text())
    _mask_sheet(out, seg, ("industrial robot arm", "safety fence", "work platform"), "mask-contact-sheet.jpg")
    _mask_sheet(out, seg, ("yellow safety post", "black bollard", "emergency stop button"), "extra-mask-contact-sheet.jpg")
    _mask_sheet(out, seg, ("cart",), "cart-mask-sheet.jpg", cart_seg)
    lap("evidenceSheetsSeconds")
    # Always re-run the shared tail locally: the cloud built under a temporary directory name, and
    # evaluation targets (when supplied) join here. It re-measures the current models and rebuilds.
    from scripts.workcell_photo_report import finalize
    finalize(out, measured)
    lap("finalizeSeconds")
    if semantic_protocol is not None:
        _semantic_stage(out, semantic_protocol, ledger)
        lap("semanticStageSeconds")
    geometry = json.loads((out / "geometry.json").read_text())
    _anchor_sheet(out, images, geometry["anchor"])
    _export_metric_scene(out, json.loads((out / 'scene-report.json').read_text()))
    lap("exportSeconds")
    metrics = {"oneShotWallSeconds": round(time.monotonic() - began, 2), "capture": capture,
               "robotQuality": robot_quality, "cartQuality": cart_quality,
               "surfaces": surfaces, "posts": posts, "runs": ledger["runs"], "geometry": geometry,
               "stageTiming": json.loads((out / "stage-timing.json").read_text()), "localStageTiming": local}
    page = _build_page(out, metrics)
    lap("pageSeconds")
    metrics["oneShotWallSeconds"] = round(time.monotonic() - began, 2)
    page_data = json.loads((page / "data.json").read_text())
    page_data["timing"]["oneShotSeconds"] = metrics["oneShotWallSeconds"]
    (page / "data.json").write_text(json.dumps(page_data, ensure_ascii=False, indent=2) + "\n")
    shared = json.loads((page / "scene-report.json").read_text())
    shared["timing"]["oneShotSeconds"] = metrics["oneShotWallSeconds"]
    (page / "scene-report.json").write_text(json.dumps(shared, ensure_ascii=False, indent=2) + "\n")
    (out / "scene-report.json").write_text(json.dumps(shared, ensure_ascii=False, indent=2) + "\n")
    (out / "one-shot.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    (out / "spend-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({"oneShotWallSeconds": metrics["oneShotWallSeconds"], "photos": count, "referencePhoto": reference,
                      "buttonStatus": geometry["anchor"]["status"], "output": str(out)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", help="Two or more photos of ONE scene, in photo order")
    parser.add_argument("--reference-photo", type=int, help="1-based photo for left/right, posts and the button scale reference (default: the last)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--viewer-assets", type=Path, help="Three.js 0.178.0 runtime directory; see docs/workcell-photo/HANDOFF.md")
    parser.add_argument("--button-diameter-m", type=float, help="Legacy whole-envelope width; cannot combine with --measurements")
    parser.add_argument("--button-height-m", type=float)
    parser.add_argument("--measurements", type=Path, help="Measured reference and independent evaluation JSON")
    parser.add_argument("--semantic-protocol", type=Path, help="Frozen semantic protocol; runs the semantic stage on the finished revision")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if not args.images or not args.out or not args.viewer_assets:
        parser.error("--images, --out and --viewer-assets are required")
    if len(args.images) < 2:
        parser.error("--images needs at least two photos of one scene")
    run([p.resolve() for p in args.images], args.out.resolve(),
        args.button_diameter_m, args.button_height_m, args.viewer_assets.resolve(), args.measurements,
        args.semantic_protocol.resolve() if args.semantic_protocol else None, args.reference_photo)


if __name__ == "__main__":
    main()
