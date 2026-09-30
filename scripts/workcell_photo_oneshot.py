"""One command: four raw workcell photos -> calibrated candidate and interactive 3D report.

Heavy stages share one ephemeral two-A100 Modal container, with GPU 0 for
geometry/OWLv2 and GPU 1 for SAM 3, then both GPUs for RecGen.
Run with the project venv: python scripts/workcell_photo_oneshot.py --images a.jpg b.jpg c.jpg d.jpg --out NEW_DIR
"""

import argparse
import base64
import gzip
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import trimesh


REPO = Path(__file__).resolve().parents[1]
MODAL = Path("/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal")
TEMPLATE = Path("/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages/workcell-photo-direct/index.html")
WORDS = ("industrial robot arm", "safety fence", "yellow safety post", "black bollard",
         "emergency stop button", "red emergency stop switch", "light curtain",
         "work platform", "cart", "control cabinet")


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
    result = subprocess.run([str(MODAL), "run", *argv], cwd=REPO,
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
    robot, cart, cart_picks = {}, {}, []
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
        cart_response = cart_seg["results"][i - 1]
        choices = []
        for k, (rle, score) in enumerate(zip(cart_response["rle"], cart_response["scores"])):
            mask = _mask(frame, {"rle": [rle]}) & good
            overlap = (mask & robot_mask).sum() / max(1, mask.sum())
            if mask.sum() > 2500 and overlap < .25:
                choices.append((score, k, mask, overlap))
        if not choices:
            raise ValueError(f"photo {i}: no cart mask separate from robot")
        score, picked, cart_mask, overlap = max(choices, key=lambda row: row[0])
        cart_picks.append({"photo": i, "instance": picked, "score": score,
                           "robotOverlap": round(float(overlap), 3), "supportedPixels": int(cart_mask.sum())})
        shared = {f"v{i}_rgb": rgb, f"v{i}_depth": np.where(good, depth, 0),
                  f"v{i}_K": K, f"v{i}_c2w": pose}
        robot.update(shared | {f"v{i}_mask": robot_mask.astype(np.uint8) * 255})
        cart.update(shared | {f"v{i}_mask": cart_mask.astype(np.uint8) * 255})
    np.savez_compressed(root / "robot-input.npz", **robot)
    np.savez_compressed(root / "cart-input.npz", **cart)
    (root / "cart-mask-selection.json").write_text(json.dumps(cart_picks, indent=2) + "\n")


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


def _color_components(mask, area_min, area_max):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return [(int(x), int(y), int(w), int(h), int(area))
            for x, y, w, h, area in stats[1:n] if area_min <= area <= area_max]


def _button_candidate(root, seg, diameter_m, height_m):
    """Find a red mushroom cap directly above a compact yellow housing."""
    candidates = []
    for i in range(1, 5):
        frame = _frame(root, i)
        rgb = _array(frame["image"])
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        red = cv2.inRange(hsv, (0, 80, 80), (12, 255, 255)) | cv2.inRange(hsv, (170, 80, 80), (179, 255, 255))
        yellow = cv2.inRange(hsv, (16, 70, 80), (45, 255, 255))
        reds = _color_components(red, 8, 180)
        yellows = _color_components(yellow, 18, 350)
        points, K, pose = (_array(frame[k]) for k in ("pts3d", "intrinsics", "camera_poses"))
        valid = _array(frame["non_ambiguous_mask"]).astype(bool)
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        depth = local[..., 2]
        for rx, ry, rw, rh, ra in reds:
            if not (3 <= rw <= 20 and 2 <= rh <= 18):
                continue
            for yx, yy, yw, yh, ya in yellows:
                if not (5 <= yw <= 35 and 3 <= yh <= 35):
                    continue
                overlap = max(0, min(rx + rw, yx + yw) - max(rx, yx))
                if overlap < .55 * rw or abs(rx + rw / 2 - yx - yw / 2) > .35 * yw:
                    continue
                if not (ry - 2 <= yy <= ry + rh + 12 and ry + rh <= yy + yh + 3):
                    continue
                x0, y0, x1, y1 = min(rx, yx), min(ry, yy), max(rx + rw, yx + yw), max(ry + rh, yy + yh)
                if x1 - x0 > 35 or y1 - y0 > 42:
                    continue
                if x0 == 0 or y0 == 0 or x1 == rgb.shape[1] or y1 == rgb.shape[0]:
                    continue  # A cropped diameter is not a scale measurement.
                patch = valid[y0:y1, x0:x1] & np.isfinite(depth[y0:y1, x0:x1]) & (depth[y0:y1, x0:x1] > 0)
                if patch.sum() < 12:
                    continue
                native_depth = float(np.median(depth[y0:y1, x0:x1][patch]))
                native_d = float((x1 - x0) * native_depth / K[0, 0])
                native_h = float((y1 - y0) * native_depth / K[1, 1])
                world = np.median(points[y0:y1, x0:x1][patch], axis=0)
                if min(native_d, native_h) <= 0:
                    continue
                candidates.append({"photo": i, "box": [x0, y0, x1, y1],
                                   "redPixels": ra, "yellowPixels": ya,
                                   "centerNative": world.tolist(), "nativeDiameter": native_d,
                                   "nativeHeight": native_h, "scaleFromDiameter": diameter_m / native_d,
                                   "scaleFromHeight": height_m / native_h})
    # The anchor is one stationary physical component in several camera views.
    groups = []
    for candidate in sorted(candidates, key=lambda row: -(row["redPixels"] + row["yellowPixels"])):
        center = np.asarray(candidate["centerNative"])
        group = next((g for g in groups if np.linalg.norm(center - np.asarray(g[0]["centerNative"])) < .35
                      and all(v["photo"] != candidate["photo"] for v in g)), None)
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)
    views = max(groups, key=lambda g: (len(g), sum(v["redPixels"] + v["yellowPixels"] for v in g)), default=[])
    scales = np.array([x[k] for x in views for k in ("scaleFromDiameter", "scaleFromHeight")], float)
    med = float(np.median(scales)) if len(scales) else None
    spread = float(np.median(np.abs(scales - med)) / med) if med else None
    shape_disagreement = (float(np.median([abs(x["scaleFromDiameter"] - x["scaleFromHeight"]) /
                                           np.mean([x["scaleFromDiameter"], x["scaleFromHeight"]])
                                           for x in views])) if views else None)
    usable = (len(views) >= 2 and spread is not None and spread <= .15 and
              shape_disagreement is not None and shape_disagreement <= .15)
    return {"assumedDiameterM": diameter_m, "assumedHeightM": height_m,
            "source": "user-supplied provisional dimensions; red/yellow color pair in raw images",
            "views": views, "candidateCount": len(candidates),
            "status": "candidate" if usable else "unverified",
            "nativeToMetres": med if usable else None,
            "relativeMedianAbsoluteDeviation": spread,
            "diameterHeightScaleDisagreement": shape_disagreement}


def _button_preview(root, button):
    """Show the supplied 20x20 cm shape as a hypothesis, even if calibration fails."""
    views = button["views"]
    if len(views) < 2:
        return None
    scale = float(np.median([view["scaleFromDiameter"] for view in views]))
    diameter_native = button["assumedDiameterM"] / scale
    height_native = button["assumedHeightM"] / scale
    center = np.median([view["centerNative"] for view in views], axis=0)
    scene = trimesh.Scene()
    for label, radius, height, offset, color in (
        ("yellow_housing", diameter_native / 2, height_native * .8, height_native * .1, [235, 197, 28, 255]),
        ("red_cap", diameter_native * .24, height_native * .2, -height_native * .4, [191, 35, 30, 255]),
    ):
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        mesh.apply_translation(center + [0, offset, 0])
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        scene.add_geometry(mesh, node_name=label, geom_name=label)
    (root / "button.glb").write_bytes(scene.export(file_type="glb"))
    return {"kind": "assumed dimension preview", "sourcePhotos": [x["photo"] for x in views],
            "scaleUsedForPreviewOnly": scale}


def _build_page(root, metrics):
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f"Report template missing: {TEMPLATE}")
    page = root / "page"
    page.mkdir()
    (page / "index.html").write_text(
        TEMPLATE.read_text().replace("../observed/viewer-assets/", "./viewer-assets/"))
    shutil.copytree(TEMPLATE.parent.parent / "observed/viewer-assets", page / "viewer-assets")
    assets = ("robot-v1.glb", "robot-v2.glb", "robot-v3.glb", "robot-v4.glb", "robot-multi.glb",
              "cart-single.glb", "cart-observed.glb", "posts.glb", "fence-observed.glb",
              "mask-contact-sheet.jpg", "extra-mask-contact-sheet.jpg", "cart-mask-sheet.jpg")
    for name in assets:
        shutil.copyfile(root / name, page / name)
    if (root / "button.glb").is_file():
        shutil.copyfile(root / "button.glb", page / "button.glb")
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
            "button": metrics["button"], "scale": metrics["button"]["status"],
            "timing": {"oneShotSeconds": metrics["oneShotWallSeconds"],
                       "geometrySeconds": times["geometrySeconds"],
                       "segmentationSeconds": times["segmentationSeconds"],
                       "segmentationExtraSeconds": 0,
                       "owlSeconds": times["owlSeconds"], "cartMaskSeconds": times["cartMaskSeconds"],
                       "cartModelSeconds": times["modelSeconds"], "modelSeconds": times["modelSeconds"]}}
    (page / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return page


def _self_check():
    assert np.array_equal(_mask({"image": {"shape": [2, 2, 3]},
                                 "input_mask_transform": {"input_to_canonical_pixel_centres": np.eye(3).tolist()}},
                                {"rle": [json.dumps({"size": [2, 2], "counts": [0, 2, 1, 1]})]}),
                          np.array([[1, 0], [1, 1]], bool))
    assert len(WORDS) == len(set(WORDS))
    print("workcell_photo_oneshot self-check passed")


def run(images, out, diameter_m, height_m):
    if len(images) != 4 or len(set(images)) != 4 or any(not p.is_file() for p in images):
        raise ValueError("Exactly four distinct, readable source photos are required")
    if diameter_m <= 0 or height_m <= 0:
        raise ValueError("Button dimensions must be positive")
    if out.exists():
        raise ValueError("Output must be a new directory; a one-shot run never mutates earlier evidence")
    out.mkdir(parents=True)
    began = time.monotonic()
    ledger = {"hardware": "one ephemeral 2 x A100-80GB container", "mode": "ephemeral modal run",
              "actualBilledUsd": None, "runs": []}
    paths = ",".join(str(p) for p in images)
    try:
        record = _job("one-container", ["modal_apps/workcell_photo_all.py", "--images", paths,
                  "--out", str(out), "--words", ",".join(WORDS)], out)
    except Exception as error:
        ledger["failure"] = str(error)
        (out / "spend-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
        raise
    ledger["runs"].append(record)
    modal_timing = json.loads((out / "modal-timing.json").read_text())
    usd_per_second = 2 * .000694 + 16 * .0000131 + 80 * .00000222
    ledger["estimate"] = {
        "usdPerSecond": round(usd_per_second, 7),
        "lowerBoundUsd": round(usd_per_second * modal_timing["containerWallSeconds"], 3),
        "wallClockUpperBoundUsd": round(usd_per_second * modal_timing["wallSecondsIncludingColdStart"], 3),
        "basis": "unit-rate estimate, not an invoice; upper bound includes possible scheduling time"}
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
    posts = _posts(out, seg)
    _mask_sheet(out, seg, ("industrial robot arm", "safety fence", "work platform"), "mask-contact-sheet.jpg")
    _mask_sheet(out, seg, ("yellow safety post", "black bollard", "emergency stop button"), "extra-mask-contact-sheet.jpg")
    _mask_sheet(out, seg, ("cart",), "cart-mask-sheet.jpg", cart_seg)
    button = _button_candidate(out, seg, diameter_m, height_m)
    button["preview"] = _button_preview(out, button)
    metrics = {"oneShotWallSeconds": round(time.monotonic() - began, 2),
               "button": button, "robotQuality": robot_quality, "cartQuality": cart_quality,
               "surfaces": surfaces, "posts": posts, "runs": ledger["runs"],
               "stageTiming": json.loads((out / "stage-timing.json").read_text())}
    page = _build_page(out, metrics)
    metrics["oneShotWallSeconds"] = round(time.monotonic() - began, 2)
    page_data = json.loads((page / "data.json").read_text())
    page_data["timing"]["oneShotSeconds"] = metrics["oneShotWallSeconds"]
    (page / "data.json").write_text(json.dumps(page_data, ensure_ascii=False, indent=2) + "\n")
    (out / "one-shot.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    (out / "spend-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({"oneShotWallSeconds": metrics["oneShotWallSeconds"],
                      "buttonStatus": button["status"], "output": str(out)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs=4)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--button-diameter-m", type=float, default=.2)
    parser.add_argument("--button-height-m", type=float, default=.2)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if not args.images or not args.out:
        parser.error("--images and --out are required")
    run([p.resolve() for p in args.images], args.out.resolve(),
        args.button_diameter_m, args.button_height_m)


if __name__ == "__main__":
    main()
