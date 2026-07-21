"""ARKitScenes eval: real iPhone photos + laser-era GT, mono vs multi-view.

Product-form exam: 1920x1440 wide RGB frames from handheld iPhone captures,
GT = metric 3D object boxes (label, centroid, axes) in the same world frame
as the camera trajectory. For each scene we pick object pairs 0.5-7 m
apart, find a frame seeing both, and measure the pair distance from
MapAnything geometry using the projected GT boxes as regions (no detector,
no SAM — isolates geometry). Two modes on the SAME scenes:

  mono       one frame  -> MapAnything native mono scale
  multiview  four frames -> MapAnything multi-view native scale

Distances are frame-invariant, so no cross-frame alignment is needed.
The traj rotation convention is auto-detected per scene by projecting GT
centroids under both interpretations and keeping the one that lands more
centroids in-image (logged, deterministic).

Usage:
  uv run --env-file .env python scripts/arkitscenes_eval.py --limit 3
  uv run --env-file .env python scripts/arkitscenes_eval.py --live --vlm
"""

import argparse
import json
import re
import sys
import time
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np

DATA = Path("outputs/datasets/arkitscenes")
WORK = Path("outputs/arkitscenes_v1")
PAIR_RANGE = (0.5, 7.0)
MAX_PAIRS_PER_SCENE = 3
UPSCALE = 1920 / 256  # lowres_wide intrinsics -> wide frame


def _axis_angle_matrix(vector: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = vector / angle
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _load_scene(vid: str) -> dict | None:
    root = DATA / "scenes" / vid
    annotation = json.load(open(root / "annotation.json"))
    objects = []
    for item in annotation.get("data", []):
        segments = item.get("segments", {}).get("obbAligned", {})
        centroid = segments.get("centroid")
        if centroid:
            objects.append(
                {
                    "label": item["label"],
                    "centroid": np.asarray(centroid),
                    "axes": np.asarray(
                        segments.get("normalizedAxes", np.eye(3).ravel().tolist())
                    ).reshape(3, 3),
                    "lengths": np.asarray(
                        segments.get("axesLengths", [0.5, 0.5, 0.5])
                    ),
                }
            )
    if len(objects) < 2:
        return None

    poses = {}
    for line in open(root / "lowres_wide.traj"):
        parts = line.split()
        if len(parts) < 7:
            continue
        ts = float(parts[0])
        rotation = _axis_angle_matrix(np.asarray([float(v) for v in parts[1:4]]))
        translation = np.asarray([float(v) for v in parts[4:7]])
        poses[round(ts, 3)] = (rotation, translation)

    intrinsics = {}
    with zipfile.ZipFile(root / "lowres_wide_intrinsics.zip") as archive:
        for name in archive.namelist():
            match = re.search(r"_([\d.]+)\.pincam$", name)
            if not match:
                continue
            values = archive.read(name).decode().split()
            intrinsics[round(float(match.group(1)), 3)] = [
                float(v) * UPSCALE for v in values[2:6]
            ]

    frames = []
    with zipfile.ZipFile(root / "upsampling.zip") as archive:
        for name in archive.namelist():
            match = re.search(r"/wide/" + vid + r"_([\d.]+)\.png$", name)
            if match:
                frames.append((round(float(match.group(1)), 3), name))
    frames.sort()
    return {
        "vid": vid,
        "objects": objects,
        "poses": poses,
        "intrinsics": intrinsics,
        "frames": frames,
    }


def _nearest(mapping: dict, ts: float):
    if ts in mapping:
        return mapping[ts]
    key = min(mapping, key=lambda value: abs(value - ts))
    return mapping[key] if abs(key - ts) < 0.1 else None


def _project(scene: dict, ts: float, points: np.ndarray, w2c: bool):
    pose = _nearest(scene["poses"], ts)
    pin = _nearest(scene["intrinsics"], ts)
    if pose is None or pin is None:
        return None
    rotation, translation = pose
    if w2c:
        camera = points @ rotation.T + translation
    else:
        camera = (points - translation) @ rotation
    fx, fy, cx, cy = pin
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * camera[:, 0] / camera[:, 2] + cx
        v = fy * camera[:, 1] / camera[:, 2] + cy
    return np.column_stack([u, v, camera[:, 2]])


def _detect_convention(scene: dict) -> bool:
    centroids = np.asarray([o["centroid"] for o in scene["objects"]])
    scores = {}
    sample = scene["frames"][:: max(1, len(scene["frames"]) // 12)]
    for w2c in (True, False):
        hits = 0
        for ts, _ in sample:
            projected = _project(scene, ts, centroids, w2c)
            if projected is None:
                continue
            ok = (
                (projected[:, 2] > 0.2)
                & (projected[:, 0] > 0) & (projected[:, 0] < 1920)
                & (projected[:, 1] > 0) & (projected[:, 1] < 1440)
            )
            hits += int(ok.sum())
        scores[w2c] = hits
    return scores[True] >= scores[False]


def _depth_at(scene: dict, frame_name: str, u: float, v: float) -> float | None:
    """FARO-projected highres depth (uint16 mm) at a pixel, or None."""
    from PIL import Image
    import io

    root = DATA / "scenes" / scene["vid"]
    depth_name = frame_name.replace("/wide/", "/highres_depth/")
    try:
        with zipfile.ZipFile(root / "upsampling.zip") as archive:
            raw = archive.read(depth_name)
    except KeyError:
        return None
    with Image.open(io.BytesIO(raw)) as image:
        depth = np.asarray(image)
    y, x = int(v), int(u)
    if not (0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]):
        return None
    patch = depth[max(0, y - 8) : y + 8, max(0, x - 8) : x + 8]
    patch = patch[patch > 0]
    if not len(patch):
        return None
    return float(np.median(patch)) / 1000.0


def _select_cases(scene: dict, w2c: bool) -> list[dict]:
    cases = []
    pairs = [
        (a, b)
        for a, b in combinations(range(len(scene["objects"])), 2)
        if PAIR_RANGE[0]
        <= np.linalg.norm(
            scene["objects"][a]["centroid"] - scene["objects"][b]["centroid"]
        )
        <= PAIR_RANGE[1]
        and scene["objects"][a]["label"] != scene["objects"][b]["label"]
    ]
    for a, b in pairs:
        best = None
        targets = np.asarray(
            [scene["objects"][a]["centroid"], scene["objects"][b]["centroid"]]
        )
        # sample every ~4th frame: the visibility check reads depth PNGs
        step = max(1, len(scene["frames"]) // 24)
        for ts, name in scene["frames"][::step]:
            projected = _project(scene, ts, targets, w2c)
            if projected is None:
                continue
            inside = (
                (projected[:, 2] > 0.5)
                & (projected[:, 0] > 96) & (projected[:, 0] < 1824)
                & (projected[:, 1] > 72) & (projected[:, 1] < 1368)
            )
            if not inside.all():
                continue
            # occlusion check against laser-projected depth: an object is
            # visible only if measured depth matches its projected depth.
            visible = True
            for u, v, depth in projected:
                measured = _depth_at(scene, name, u, v)
                if measured is None or abs(measured - depth) / depth > 0.35:
                    visible = False
                    break
            if not visible:
                continue
            # reject frames where the two projected GT boxes overlap: box
            # regions on overlapping objects measure the wrong thing.
            probe = {"pair": (a, b), "ts": ts}
            box_a, _ = _bbox_region(scene, probe, 0, w2c)
            box_b, _ = _bbox_region(scene, probe, 1, w2c)
            ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
            ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_min = min(
                (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]),
                (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]),
            )
            if area_min <= 0 or inter / area_min > 0.2:
                continue
            separation = float(
                np.hypot(
                    projected[0, 0] - projected[1, 0],
                    projected[0, 1] - projected[1, 1],
                )
            )
            if best is None or separation > best[0]:
                best = (separation, ts, name)
        if best is not None:
            cases.append(
                {
                    "pair": (a, b),
                    "labels": (
                        scene["objects"][a]["label"],
                        scene["objects"][b]["label"],
                    ),
                    "gt_m": float(
                        np.linalg.norm(
                            scene["objects"][a]["centroid"]
                            - scene["objects"][b]["centroid"]
                        )
                    ),
                    "ts": best[1],
                    "frame_name": best[2],
                }
            )
        if len(cases) >= MAX_PAIRS_PER_SCENE:
            break
    return cases


def _extract_frames(vid: str, names: list[str]) -> list[Path]:
    out = []
    root = DATA / "scenes" / vid
    target_dir = WORK / "frames" / vid
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "upsampling.zip") as archive:
        for name in names:
            target = target_dir / Path(name).name
            if not target.exists():
                target.write_bytes(archive.read(name))
            out.append(target)
    return out


def _geometry_dir(vid: str, mode: str, key: str) -> Path:
    return WORK / "geometry" / f"{vid}__{mode}__{key}"


def _run_geometry(images: list[Path], target: Path) -> None:
    if (target / "frames").is_dir():
        return
    from ehs_spatial.providers.base import ProviderError
    from ehs_spatial.providers.map_anything import MapAnythingAdapter

    for attempt in range(6):
        try:
            MapAnythingAdapter().run([str(p) for p in images], target)
            return
        except ProviderError as exc:
            message = str(exc)
            if not any(c in message for c in ("429", "throttled", "500", "502", "503")):
                raise
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"MapAnything kept failing for {target.name}")


def _box_corners(obj: dict) -> np.ndarray:
    half = obj["lengths"] / 2.0
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    return obj["centroid"] + (signs * half) @ obj["axes"]


def _bbox_region(scene: dict, case: dict, index: int, w2c: bool):
    """Tight 2D box from the projected GT 3D box corners, plus the object's
    expected depth for region filtering."""
    obj = scene["objects"][case["pair"][index]]
    corners = _project(scene, case["ts"], _box_corners(obj), w2c)
    centroid = _project(scene, case["ts"], obj["centroid"][None, :], w2c)[0]
    u1 = float(np.clip(corners[:, 0].min(), 0, 1920))
    u2 = float(np.clip(corners[:, 0].max(), 0, 1920))
    v1 = float(np.clip(corners[:, 1].min(), 0, 1440))
    v2 = float(np.clip(corners[:, 1].max(), 0, 1440))
    return (u1, v1, u2, v2), float(centroid[2])


def _region_centroid(
    geometry_dir: Path, frame_index: int, box, expected_depth: float | None = None
) -> np.ndarray | None:
    frame_dirs = sorted((geometry_dir / "frames").iterdir())
    frame_dir = frame_dirs[frame_index]
    points = np.load(frame_dir / "pts3d.npy")
    valid = np.load(frame_dir / "valid_mask.npy").astype(bool)
    height, width = valid.shape
    sx, sy = width / 1920.0, height / 1440.0
    x1, y1, x2, y2 = box
    mask = np.zeros_like(valid)
    mask[
        max(0, int(y1 * sy)) : min(height, int(y2 * sy)),
        max(0, int(x1 * sx)) : min(width, int(x2 * sx)),
    ] = True
    selected = mask & valid & np.isfinite(points).all(axis=2)
    cloud = points[selected]
    if len(cloud) < 25:
        return None
    ranges = np.linalg.norm(cloud, axis=1)
    cloud = cloud[ranges > 0.15]
    if len(cloud) < 25:
        return None
    if expected_depth is not None:
        # Keep only points near the object's laser-verified depth: the box
        # region includes background/foreground the object does not occupy.
        # Scale-free selection: normalise ranges by their median so a global
        # mono scale error does not evict the object itself.
        ranges = np.linalg.norm(cloud, axis=1)
        rel = ranges / (np.median(ranges) + 1e-6)
        cloud = cloud[np.abs(rel - 1.0) < 0.35]
        if len(cloud) < 25:
            return None
    ranges = np.linalg.norm(cloud, axis=1)
    median = np.median(ranges)
    mad = np.median(np.abs(ranges - median)) + 1e-6
    cloud = cloud[np.abs(ranges - median) < 2.0 * mad]
    if len(cloud) < 25:
        return None
    return np.median(cloud, axis=0)


def _ask_vlm(scene: dict, case: dict, frame: Path, boxes) -> float | None:
    cache = WORK / "vlm" / f"{scene['vid']}_{case['pair'][0]}_{case['pair'][1]}.json"
    if cache.exists():
        return json.loads(cache.read_text()).get("value")
    import io

    from PIL import Image, ImageDraw
    from google import genai
    from google.genai import types

    with Image.open(frame) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for box, color in zip(boxes, ("red", "blue")):
        draw.rectangle(box, outline=color, width=6)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"),
            f"Estimate the distance in metres between the {case['labels'][0]} "
            f"(red box) and the {case['labels'][1]} (blue box) centres. "
            "Respond with ONLY a number in meters.",
        ],
    )
    match = re.search(r"[\d.]+", response.text or "")
    value = float(match.group(0)) if match else None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"value": value}) + "\n")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vlm", action="store_true")
    args = parser.parse_args(argv)

    vids = (DATA / "selected_ids.txt").read_text().split()
    if args.limit:
        vids = vids[: args.limit]

    planned = []
    for vid in vids:
        if not (DATA / "scenes" / vid / "upsampling.zip").exists():
            continue
        scene = _load_scene(vid)
        if scene is None:
            continue
        w2c = _detect_convention(scene)
        cases = _select_cases(scene, w2c)
        if cases:
            planned.append((scene, w2c, cases))
    total_calls = sum(
        (0 if (_geometry_dir(s["vid"], "mono", c["frame_name"][-9:-4]) / "frames").is_dir() else 1)
        + (0 if (_geometry_dir(s["vid"], "multi", c["frame_name"][-9:-4]) / "frames").is_dir() else 1)
        for s, _, cases in planned
        for c in cases[:1]
    )
    n_cases = sum(len(c) for _, _, c in planned)
    print(f"{len(planned)} scenes | {n_cases} pairs | ~{total_calls} MapAnything calls")
    if total_calls and not args.live:
        print("pass --live to spend them (cached reruns are free)", file=sys.stderr)
        return 2

    rows = []
    for scene, w2c, cases in planned:
        vid = scene["vid"]
        # one reconstruction per scene per mode, anchored on the first case's frame
        primary = cases[0]
        key = primary["frame_name"][-9:-4]
        mono_dir = _geometry_dir(vid, "mono", key)
        multi_dir = _geometry_dir(vid, "multi", key)
        try:
            mono_frame = _extract_frames(vid, [primary["frame_name"]])[0]
            step = max(1, len(scene["frames"]) // 4)
            extra_names = [name for _, name in scene["frames"][::step][:3]]
            multi_names = [primary["frame_name"]] + [
                n for n in extra_names if n != primary["frame_name"]
            ][:3]
            multi_frames = _extract_frames(vid, multi_names)
            _run_geometry([mono_frame], mono_dir)
            _run_geometry(multi_frames, multi_dir)
        except Exception as exc:
            print(f"skipping scene {vid}: {exc}", file=sys.stderr)
            continue

        for case in cases:
            if case["frame_name"] != primary["frame_name"]:
                continue  # score only pairs on the reconstructed frame
            regions = [_bbox_region(scene, case, i, w2c) for i in (0, 1)]
            boxes = [r[0] for r in regions]
            row = {
                "vid": vid,
                "labels": case["labels"],
                "gt_m": round(case["gt_m"], 3),
                "w2c": w2c,
            }
            for mode, gdir in (("mono", mono_dir), ("multi", multi_dir)):
                centroids = [
                    _region_centroid(gdir, 0, box, depth)
                    for box, depth in regions
                ]
                if any(c is None for c in centroids):
                    continue
                predicted = float(np.linalg.norm(centroids[0] - centroids[1]))
                row[f"{mode}_m"] = round(predicted, 3)
                row[f"{mode}_rel"] = round(abs(predicted - case["gt_m"]) / case["gt_m"], 3)
            if args.vlm:
                value = _ask_vlm(scene, case, mono_frame, boxes)
                if value is not None:
                    row["vlm_m"] = round(value, 3)
                    row["vlm_rel"] = round(abs(value - case["gt_m"]) / case["gt_m"], 3)
            rows.append(row)

    def summarize(key: str) -> dict:
        errors = [r[key] for r in rows if r.get(key) is not None]
        if not errors:
            return {"answered": 0, "of": len(rows)}
        return {
            "answered": len(errors),
            "of": len(rows),
            "median_rel_err": round(float(np.median(errors)), 3),
            "success_at_25pct": round(
                sum(1 for e in errors if e <= 0.25) / len(errors), 3
            ),
        }

    summary = {
        "rows": rows,
        "mono": summarize("mono_rel"),
        "multi": summarize("multi_rel"),
        "vlm": summarize("vlm_rel") if args.vlm else None,
        "protocol": "GT = 3D box centroid distance; regions = projected GT "
        "centroids (detector-free); success = rel err <= 25%",
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in rows[-6:]:
        print(row)
    print("mono:", summary["mono"])
    print("multi:", summary["multi"])
    if args.vlm:
        print("vlm:", summary["vlm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
