"""One command: four raw workcell photos -> calibrated candidate and interactive 3D report.

Heavy stages share one ephemeral two-A100 Modal container, with GPU 0 for
geometry/OWLv2 and GPU 1 for SAM 3, then both GPUs for RecGen.
Run with the project venv: python scripts/workcell_photo_oneshot.py --images a.jpg b.jpg c.jpg d.jpg --out NEW_DIR --viewer-assets THREE_0_178_0_DIR
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
         "instruction poster", "transparent safety panel", "floor marking", GUARD_WORD)


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
    if result.returncode:
        safe = [line for line in result.stdout.splitlines()
                if not re.search(r"capabilit|token|secret", line, re.I)]
        raise RuntimeError(f"{name} failed ({result.returncode}): " + "\n".join(safe[-10:])[-1800:])
    return {"stage": name, "wallSeconds": round(elapsed, 2), "runIds": sorted(set(run_ids)),
            "result": str(out)}


def _foreground(root, seg, index, word, minimum=300):
    frame = _frame(root, index)
    mask = _mask(frame, _response(seg, index, word))
    good = _array(frame["non_ambiguous_mask"]).astype(bool)
    result = mask & good
    if result.sum() < minimum:
        raise ValueError(f"photo {index}: {word} has only {result.sum()} valid pixels")
    return frame, result


def _prepare_inputs(root, seg, cart_seg):
    robot, cart, guard, cart_picks, guard_picks = {}, {}, {}, [], []
    floor = json.loads((root / "floor-reference.json").read_text())
    ground_n = np.asarray(floor["normal"])
    ground_d = floor["offset"]
    for i in range(1, 5):
        frame, robot_mask = _foreground(root, seg, i, "industrial robot arm", 500)
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
        robot_mask &= good
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
            if mask.sum() > 2500 and guard_fraction < .5 and overlap < .25:
                choices.append((score, k, mask, overlap, guard_fraction))
        if not choices:
            raise ValueError(f"photo {i}: no cart mask separate from robot")
        score, picked, cart_mask, overlap, guard_fraction = max(choices, key=lambda row: row[0])
        guard_picks.append({"photo": i, "supportedPixels": int(guard_mask.sum()),
                            "source": GUARD_WORD, "selectedPanels": len(panels), "routing": "score >= .6; above floor residual; area >= .2 largest supported panel", "excludedFromCartFraction": float(guard_fraction)})
        cart_picks.append({"photo": i, "instance": picked, "score": score,
                           "robotOverlap": round(float(overlap), 3), "supportedPixels": int(cart_mask.sum())})
        shared = {f"v{i}_rgb": rgb, f"v{i}_depth": np.where(good, depth, 0),
                  f"v{i}_K": K, f"v{i}_c2w": pose}
        robot.update(shared | {f"v{i}_mask": robot_mask.astype(np.uint8) * 255})
        cart.update(shared | {f"v{i}_mask": cart_mask.astype(np.uint8) * 255})
        guard.update(shared | {f"v{i}_mask": guard_mask.astype(np.uint8) * 255})
    np.savez_compressed(root / "robot-input.npz", **robot)
    np.savez_compressed(root / "cart-input.npz", **cart)
    np.savez_compressed(root / "guard-input.npz", **guard)
    (root / "guard-mask-selection.json").write_text(json.dumps(guard_picks, indent=2) + "\n")
    (root / "cart-mask-selection.json").write_text(json.dumps(cart_picks, indent=2) + "\n")


def _guard_views(rows):
    # RecGen's shape/appearance conditioning uses the first pair. Occluded
    # capture-order views erased the connecting face; rank actual mask support.
    return [r['photo'] for r in sorted(rows, key=lambda r: (-r['supportedPixels'], r['photo']))
            if r['supportedPixels'] >= 100][:2]


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
    frame = _frame(root, 4)
    points, colors = _array(frame["pts3d"]), _array(frame["image"])
    valid = _array(frame["non_ambiguous_mask"]).astype(bool)
    scene, records = trimesh.Scene(), []
    for word, kind in (("yellow safety post", "box"), ("black bollard", "cylinder")):
        response = _response(seg, 4, word)
        candidates = []
        for encoded, score in zip(response["rle"], response["scores"]):
            mask = _mask(frame, {"rle": [encoded]}) & valid
            if score >= .8 and mask.sum() >= 200:
                candidates.append((score, mask))
        for number, (score, mask) in enumerate(sorted(candidates, key=lambda row: -row[0])[:2], 1):
            cloud = points[mask]
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
            color = np.median(colors[mask], axis=0).astype(np.uint8)
            mesh.visual.vertex_colors = np.tile(np.r_[color, 255], (len(mesh.vertices), 1))
            label = f"{kind}-{number}"
            scene.add_geometry(mesh, node_name=label, geom_name=label)
            records.append({"name": label, "sourcePhoto": 4, "score": score,
                            "supportedPixels": int(mask.sum()), "kind": kind})
    if len(records) != 4:
        raise ValueError(f"Expected two yellow and two black posts, got {len(records)}")
    (root / "posts.glb").write_bytes(scene.export(file_type="glb"))
    (root / "posts-source.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def _mask_sheet(root, seg, groups, filename, cart=None):
    tiles = []
    colors = ((20, 80, 255), (0, 210, 20), (240, 100, 20), (200, 0, 180), (0, 180, 220))
    for i in range(1, 5):
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
    sheet = np.hstack([np.vstack(tiles[:2]), np.vstack(tiles[2:])])
    Image.fromarray(sheet).save(root / filename, quality=86)


def _bbox_quality(root, seg, kind):
    rows = []
    for i in range(1, 5):
        frame = _frame(root, i)
        pose, K = _array(frame["camera_poses"]), _array(frame["intrinsics"])
        if kind == "robot":
            mask = _mask(frame, _response(seg, i, "industrial robot arm"))
        else:
            z = np.load(root / "cart-input.npz")
            mask = z[f"v{i}_mask"] > 0
        yy, xx = np.nonzero(mask)
        truth = np.array([xx.min(), xx.max(), yy.min(), yy.max()], float)
        models = ((f"robot-v{i}", f"v{i}"), ("robot-multi", "multi")) if kind == "robot" else (("cart-single", "single"),)
        for model, label in models:
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


def _export_metric_scene(root, geometry):
    """Bake one common scale and floor transform into the downloadable scene."""
    scale = float(geometry["anchor"]["mPerNative"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("No finite positive reference scale for scene export")
    normal = np.asarray(geometry["floor"]["normal"], float)
    transform = trimesh.geometry.align_vectors(normal, [0, 1, 0])
    transform[:3, :3] *= scale
    transform[1, 3] = scale * float(geometry["floor"]["offset"]) / np.linalg.norm(normal)
    scene = trimesh.Scene()
    objects = json.loads((root / "objects.json").read_text())["objects"]
    cached = {}
    for item in objects:
        spec = item.get("modelsByPhoto", {}).get("4", item["model"])
        if not spec or not spec.get("nodes"):
            continue
        if spec["file"] not in cached:
            cached[spec["file"]] = trimesh.load(root / spec["file"], force="scene")
        source = cached[spec["file"]]
        for node in spec["nodes"]:
            matrix, geometry_id = source.graph.get(node)
            original = source.geometry[geometry_id]
            mesh = original.copy()
            if original.visual.kind == 'texture' and 'color' in original.visual.vertex_attributes:
                mesh.visual.vertex_attributes['color'] = original.visual.vertex_attributes['color'].copy()
            name = item["id"] + ":" + node
            scene.add_geometry(mesh, node_name=name, geom_name=name, transform=transform @ matrix)
    scene.metadata.update(units="meters")
    if geometry.get('calibration'):
        scene.metadata.update(scale_status='user_measured_reference', calibration=geometry['calibration'])
    else:
        scene.metadata.update(scale_status="user_dimension_hypothesis",
                              button_height_m=geometry["anchor"]["assumedHeightM"],
                              button_width_m=geometry["anchor"]["assumedWidthM"])
    (root / "workcell-metric.glb").write_bytes(scene.export(file_type="glb"))


def _anchor_sheet(root, sources, anchor):
    """Show exactly which physical component the editable dimensions refer to."""
    views = anchor["views"]
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
    assets = ("robot-v1.glb", "robot-v2.glb", "robot-v3.glb", "robot-v4.glb", "robot-multi.glb",
              "cart-single.glb", "guard-multi.glb", "guard-left.glb", "guard-center.glb", "guard-right.glb",
              "guard-partition.json", "cart-observed.glb", "posts.glb", "fence-observed.glb",
              "mask-contact-sheet.jpg", "extra-mask-contact-sheet.jpg", "cart-mask-sheet.jpg",
              "fence-fitted.glb", "floor-fitted.glb", "workcell-metric.glb", "geometry.json",
              "objects.json", "object-extras.glb", "scene-report.json")
    for name in assets:
        shutil.copyfile(root / name, page / name)
    for name in ('measurements.json', 'measurement-evaluation.json'):
        if (root/name).is_file():
            shutil.copyfile(root/name, page/name)
    for model in root.glob("entity-*.glb"):
        shutil.copyfile(model, page / model.name)
    for evidence in [*root.glob("geometry-*.jpg"), *root.glob("geometry-*.png")]:
        shutil.copyfile(evidence, page / evidence.name)
    frames = []
    for i in range(1, 5):
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
    (page / "scene-report.json").write_text(json.dumps(shared, ensure_ascii=False, indent=2) + "\n")
    return page


def _self_check():
    assert np.array_equal(_mask({"image": {"shape": [2, 2, 3]},
                                 "input_mask_transform": {"input_to_canonical_pixel_centres": np.eye(3).tolist()}},
                                {"rle": [json.dumps({"size": [2, 2], "counts": [0, 2, 1, 1]})]}),
                          np.array([[1, 0], [1, 1]], bool))
    assert len(WORDS) == len(set(WORDS))
    print("workcell_photo_oneshot self-check passed")


def run(images, out, diameter_m, height_m, viewer_assets, measurements=None):
    from scripts.workcell_photo_calibration import apply_measurements, load_measurements, resolve_dimensions
    measured = load_measurements(measurements) if measurements else None
    diameter_m, height_m = resolve_dimensions(measured, diameter_m, height_m)
    if len(images) != 4 or len(set(images)) != 4 or any(not p.is_file() for p in images):
        raise ValueError("Exactly four distinct, readable source photos are required")
    if not all(np.isfinite(v) and v > 0 for v in (diameter_m, height_m)):
        raise ValueError("Button dimensions must be finite and positive")
    if out.exists():
        raise ValueError("Output must be a new directory; a one-shot run never mutates earlier evidence")
    out.mkdir(parents=True)
    # Freeze the built viewer before compute; concurrent rebuilds must not change a running report.
    _freeze_report_ui(out, viewer_assets)
    began = time.monotonic()
    ledger = {"hardware": "one ephemeral 2 x A100-80GB container", "mode": "ephemeral modal run",
              "actualBilledUsd": None, "runs": []}
    paths = ",".join(str(p) for p in images)
    try:
        record = _job("one-container", ["modal_apps/workcell_photo_all.py", "--images", paths,
                  "--out", str(out), "--words", ",".join(WORDS),
                  "--button-diameter-m", str(diameter_m), "--button-height-m", str(height_m)], out)
    except Exception as error:
        ledger["failure"] = str(error)
        (out / "spend-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
        raise
    ledger["runs"].append(record)
    modal_timing = json.loads((out / "modal-timing.json").read_text())
    usd_per_second = 2 * .000694 + 16 * .0000131 + 80 * .00000222
    ledger["estimate"] = {
        "usdPerSecond": round(usd_per_second, 7),
        "functionWindowEstimateUsd": round(usd_per_second * modal_timing["containerWallSeconds"], 3),
        "callWindowEstimateUsd": round(usd_per_second * modal_timing["wallSecondsIncludingColdStart"], 3),
        "rateSource": "https://modal.com/pricing",
        "rateCheckedDate": "2026-09-30",
        "basis": "reserved-resource list-rate estimates, not invoice amounts; call window includes possible scheduling time"}
    (out / "spend-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    seg = json.loads((out / "sam3.json").read_text())
    cart_seg = json.loads((out / "cart-masks.json").read_text())
    if len(seg["results"]) != 4 or len(cart_seg["results"]) != 4:
        raise ValueError("Modal result must include four images")
    robot_quality = _bbox_quality(out, seg, "robot")
    cart_quality = _bbox_quality(out, seg, "cart")
    (out / "bbox-eval.json").write_text(json.dumps(robot_quality, indent=2) + "\n")
    (out / "cart-bbox-eval.json").write_text(json.dumps(cart_quality, indent=2) + "\n")
    cart_z = np.load(out / "cart-input.npz")
    surfaces = {"cart": _surface(out, seg, "cart-observed", 1, "cart", cart_z["v1_mask"] > 0),
                "fence": _surface(out, seg, "fence-observed", 4, "safety fence")}
    posts = json.loads((out / "posts-source.json").read_text())
    _mask_sheet(out, seg, ("industrial robot arm", "safety fence", "work platform"), "mask-contact-sheet.jpg")
    _mask_sheet(out, seg, ("yellow safety post", "black bollard", "emergency stop button"), "extra-mask-contact-sheet.jpg")
    _mask_sheet(out, seg, ("cart",), "cart-mask-sheet.jpg", cart_seg)
    geometry = json.loads((out / "geometry.json").read_text())
    if measured:
        geometry = apply_measurements(out, measured)
        from scripts.workcell_photo_report import build as build_report
        build_report(out)
    _anchor_sheet(out, images, geometry["anchor"])
    _export_metric_scene(out, geometry)
    metrics = {"oneShotWallSeconds": round(time.monotonic() - began, 2),
               "robotQuality": robot_quality, "cartQuality": cart_quality,
               "surfaces": surfaces, "posts": posts, "runs": ledger["runs"], "geometry": geometry,
               "stageTiming": json.loads((out / "stage-timing.json").read_text())}
    page = _build_page(out, metrics)
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
    print(json.dumps({"oneShotWallSeconds": metrics["oneShotWallSeconds"],
                      "buttonStatus": geometry["anchor"]["status"], "output": str(out)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs=4)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--viewer-assets", type=Path, help="Three.js 0.178.0 runtime directory; see docs/workcell-photo/HANDOFF.md")
    parser.add_argument("--button-diameter-m", type=float, help="Legacy whole-envelope width; cannot combine with --measurements")
    parser.add_argument("--button-height-m", type=float)
    parser.add_argument("--measurements", type=Path, help="Measured reference and independent evaluation JSON")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if not args.images or not args.out or not args.viewer_assets:
        parser.error("--images, --out and --viewer-assets are required")
    run([p.resolve() for p in args.images], args.out.resolve(),
        args.button_diameter_m, args.button_height_m, args.viewer_assets.resolve(), args.measurements)


if __name__ == "__main__":
    main()
