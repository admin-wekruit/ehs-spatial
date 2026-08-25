"""Whole-scene inventory: enumerate every object, then map the room to CAD.

Three stages over a run's cached geometry:
  1. a VLM lists every distinct object it can see, as short noun phrases
  2. SAM segments each phrase; masks are lifted into the run's floor frame
  3. wall planes are fitted from the non-floor cloud, and the result is
     written as a dimensioned floor plan (PNG) and a real DXF that opens
     in any CAD tool

Only stages 1-2 spend money, and both are disk-cached, so re-runs are free.
Nothing here feeds a verdict: this is the scene-understanding surface, not
the compliance path.

Usage:
  uv run --env-file .env python scripts/scene_inventory.py --run demo-real-factory
  uv run --env-file .env python scripts/scene_inventory.py --run demo-real-factory --live
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint, Point, Polygon

from ehs_spatial.contracts import GeometryFrame, Observation2D
from ehs_spatial.geometry import _build_geometry, _spatial_state
from ehs_spatial.providers.sam3 import SAM3_ENDPOINT, decode_coco_rle

MAX_PHRASES = 26
MIN_MASK_PIXELS = 50
MIN_CLOUD_POINTS = 50
WALL_MIN_INLIERS = 400
WALL_MAX_COUNT = 6
# Beyond this range single-view depth collapses (the boundary map measured
# negative heights at ~20 m), so far entries are inventoried but kept off
# the dimensioned plan rather than drawn as if they were surveyed.
PLAN_MAX_RANGE_M = 12.0
# Structure is drawn as structure, not as furniture.
STRUCTURE_LABELS = {"floor", "ceiling", "wall", "window", "ground", "roof"}


def _frames(run: Path) -> list[GeometryFrame]:
    out = []
    for frame_dir in sorted((run / "geometry" / "frames").iterdir()):
        if not frame_dir.is_dir():
            continue
        out.append(
            GeometryFrame(
                frame_id=frame_dir.name,
                canonical_image_path=str(frame_dir / "canonical.png"),
                pts3d_path=str(frame_dir / "pts3d.npy"),
                conf_path=str(frame_dir / "conf.npy"),
                valid_mask_path=str(frame_dir / "valid_mask.npy"),
                camera_to_world=np.load(frame_dir / "camera_to_world.npy").tolist(),
                intrinsics=np.load(frame_dir / "intrinsics.npy").tolist(),
            )
        )
    return out


def _enumerate_objects(run: Path, frame: GeometryFrame, *, live: bool):
    """Stage 1 — the VLM only supplies vocabulary, never geometry."""
    cache = run / "inventory" / "phrases.json"
    if cache.exists():
        return json.loads(cache.read_text())
    if not live:
        return None
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            types.Part.from_bytes(
                data=Path(frame.canonical_image_path).read_bytes(),
                mime_type="image/png",
            ),
            "List every distinct physical object and structure visible in "
            "this industrial photo. Respond with ONLY a JSON array of short "
            "singular noun phrases suitable as segmentation prompts (e.g. "
            '["machine", "safety fence", "worker", "floor marking"]). '
            f"At most {MAX_PHRASES} entries, most prominent first. No "
            "duplicates, no adjectives about colour or count.",
        ],
    )
    match = re.search(r"\[.*\]", response.text or "", re.S)
    phrases = [str(p).strip().lower() for p in json.loads(match.group(0))]
    phrases = list(dict.fromkeys(p for p in phrases if p))[:MAX_PHRASES]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(phrases, indent=2) + "\n")
    return phrases


def _segment(run: Path, frame: GeometryFrame, phrase: str, *, live: bool):
    slug = re.sub(r"[^a-z0-9]+", "_", phrase).strip("_")
    cache = run / "inventory" / "sam" / f"{frame.frame_id}__{slug}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    if not live:
        return None
    import fal_client

    response = fal_client.subscribe(
        SAM3_ENDPOINT,
        arguments={
            "image_url": "data:image/png;base64,"
            + base64.b64encode(
                Path(frame.canonical_image_path).read_bytes()
            ).decode("ascii"),
            "prompt": phrase,
            "return_multiple_masks": True,
            "include_scores": True,
            "include_boxes": True,
            "max_masks": 12,
        },
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(response) + "\n")
    return response


def _clean(points: np.ndarray) -> np.ndarray | None:
    if len(points) < MIN_CLOUD_POINTS:
        return None
    radius = np.linalg.norm(points[:, :2], axis=1)
    points = points[radius > 0.15]
    if len(points) < MIN_CLOUD_POINTS:
        return None
    radius = np.linalg.norm(points[:, :2], axis=1)
    centre = np.median(radius)
    spread = np.median(np.abs(radius - centre)) + 1e-6
    points = points[np.abs(radius - centre) < 2.5 * spread]
    return points if len(points) >= MIN_CLOUD_POINTS else None


def _fit_walls(points: np.ndarray) -> list[dict]:
    """Vertical planes above the floor, as 2D wall segments. Deterministic
    RANSAC on the horizontal projection: a wall is a line in plan view."""
    rng = np.random.default_rng(0)
    remaining = points[(points[:, 2] > 0.5) & (points[:, 2] < 6.0)]
    if len(remaining) > 40000:
        remaining = remaining[:: len(remaining) // 40000]
    walls = []
    for _ in range(WALL_MAX_COUNT):
        if len(remaining) < WALL_MIN_INLIERS:
            break
        best = None
        for _ in range(400):
            a, b = remaining[rng.choice(len(remaining), 2, replace=False)][:, :2]
            direction = b - a
            length = np.linalg.norm(direction)
            if length < 1.0:
                continue
            normal = np.array([-direction[1], direction[0]]) / length
            distance = np.abs((remaining[:, :2] - a) @ normal)
            inliers = distance < 0.12
            count = int(inliers.sum())
            if best is None or count > best[0]:
                best = (count, inliers, a, normal)
        if best is None or best[0] < WALL_MIN_INLIERS:
            break
        count, inliers, anchor, normal = best
        cloud = remaining[inliers]
        along = np.array([-normal[1], normal[0]])
        projection = (cloud[:, :2] - anchor) @ along
        low, high = np.quantile(projection, [0.02, 0.98])
        if high - low < 1.0:
            remaining = remaining[~inliers]
            continue
        walls.append(
            {
                "start": (anchor + along * low).tolist(),
                "end": (anchor + along * high).tolist(),
                "length_m": round(float(high - low), 2),
                "height_m": round(float(np.quantile(cloud[:, 2], 0.95)), 2),
                "points": count,
            }
        )
        remaining = remaining[~inliers]
    return walls


def _write_dxf(path: Path, walls: list[dict], objects: list[dict]) -> None:
    """Minimal but valid DXF R12: walls on WALLS, footprints on OBJECTS,
    labels on TEXT. Opens in AutoCAD/LibreCAD/QCAD. Units are metres."""
    out = ["0", "SECTION", "2", "ENTITIES"]

    def line(x1, y1, x2, y2, layer):
        out.extend(
            ["0", "LINE", "8", layer,
             "10", f"{x1:.4f}", "20", f"{y1:.4f}", "30", "0.0",
             "11", f"{x2:.4f}", "21", f"{y2:.4f}", "31", "0.0"]
        )

    for wall in walls:
        line(*wall["start"], *wall["end"], "WALLS")
    for obj in objects:
        ring = obj["footprint"] + [obj["footprint"][0]]
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            line(x1, y1, x2, y2, "OBJECTS")
        cx, cy = obj["centroid_xy"]
        out.extend(
            ["0", "TEXT", "8", "TEXT",
             "10", f"{cx:.4f}", "20", f"{cy:.4f}", "30", "0.0",
             "40", "0.18", "1", f"{obj['label']} H={obj['height_m']:.2f}"]
        )
    out.extend(["0", "ENDSEC", "0", "EOF"])
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _clip_segment(start, end, min_x, min_y, max_x, max_y):
    """Liang-Barsky clip of a wall segment to the drawing frame."""
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x1 - min_x), (dx, max_x - x1),
        (-dy, y1 - min_y), (dy, max_y - y1),
    ):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return (
        (x1 + t0 * dx, y1 + t0 * dy),
        (x1 + t1 * dx, y1 + t1 * dy),
    )


def _render_plan(path: Path, walls: list[dict], objects: list[dict]) -> None:
    size, margin = 1400, 90
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    # Frame on the measured objects and the camera. A long wall must not set
    # the scale, or the workcell shrinks into a corner — walls get clipped to
    # the frame instead.
    xs = [0.0] + [x for o in objects for x, _ in o["footprint"]]
    ys = [0.0] + [y for o in objects for _, y in o["footprint"]]
    min_x, max_x = min(xs) - 1.2, max(xs) + 1.2
    min_y, max_y = min(ys) - 1.2, max(ys) + 1.2
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    min_x -= (span - (max_x - min_x)) / 2
    max_x = min_x + span
    min_y -= (span - (max_y - min_y)) / 2
    max_y = min_y + span
    scale = (size - 2 * margin) / span

    def px(point):
        return (
            margin + (point[0] - min_x) * scale,
            size - margin - (point[1] - min_y) * scale,
        )

    for gx in np.arange(np.floor(min_x), np.ceil(max_x) + 1):
        draw.line([px((gx, min_y)), px((gx, max_y))], fill="#ececec")
    for gy in np.arange(np.floor(min_y), np.ceil(max_y) + 1):
        draw.line([px((min_x, gy)), px((max_x, gy))], fill="#ececec")

    for wall in walls:
        clipped = _clip_segment(
            wall["start"], wall["end"], min_x, min_y, max_x, max_y
        )
        if clipped is None:
            continue
        start, end = clipped
        draw.line([px(start), px(end)], fill="#333333", width=7)
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        draw.text(px(mid), f"wall {wall['length_m']:.1f}m", fill="#333333")

    palette = [
        "#e5194b", "#2f7de1", "#12a15a", "#c86a00", "#8a4fd0",
        "#0f9aa8", "#b8860b", "#d4457f", "#5a7d2a", "#7a6a5a",
    ]
    for index, obj in enumerate(objects):
        colour = palette[index % len(palette)]
        draw.polygon([px(p) for p in obj["footprint"]], outline=colour, width=3)
        cx, cy = obj["centroid_xy"]
        anchor_px = px((cx, cy))
        # Fan labels down the right margin with leader lines: plan symbols
        # overlap in a crowded workcell, their captions must not.
        label_y = margin + 26 * index + 10
        label_x = size - margin - 250
        draw.line([anchor_px, (label_x - 8, label_y + 8)], fill=colour, width=1)
        draw.ellipse(
            [anchor_px[0] - 3, anchor_px[1] - 3, anchor_px[0] + 3, anchor_px[1] + 3],
            fill=colour,
        )
        draw.text(
            (label_x, label_y),
            f"{obj['label']}  H {obj['height_m']:.2f} m  "
            f"{obj['size_m']} m  d {obj['camera_dist_m']:.2f} m",
            fill=colour,
        )

    camera = px((0.0, 0.0))
    draw.ellipse(
        [camera[0] - 8, camera[1] - 8, camera[0] + 8, camera[1] + 8], fill="#c62828"
    )
    draw.text((camera[0] + 12, camera[1] - 6), "camera", fill="#c62828")
    draw.line(
        [(margin, size - 40), (margin + scale, size - 40)], fill="black", width=4
    )
    draw.text((margin, size - 34), "1 m", fill="black")
    image.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--camera-height", type=float, default=1.5)
    args = parser.parse_args(argv)

    run = Path("runs") / args.run
    frames = _frames(run)
    observations = [
        Observation2D.model_validate(item)
        for item in json.loads((run / "observations.json").read_text())
    ]
    transform = _build_geometry(frames, observations, args.camera_height).transform
    if transform is None:
        print("run has no floor transform", file=sys.stderr)
        return 1

    phrases = _enumerate_objects(run, frames[0], live=args.live)
    if phrases is None:
        print("needs 1 VLM call + up to 26 SAM calls; pass --live", file=sys.stderr)
        return 2
    print(f"VLM listed {len(phrases)} objects: {', '.join(phrases)}")

    entries: list[dict] = []
    for frame in frames:
        points3d = np.load(frame.pts3d_path)
        valid = np.load(frame.valid_mask_path).astype(bool)
        finite = valid & np.isfinite(points3d).all(axis=2)
        height, width = valid.shape
        for phrase in phrases:
            response = _segment(run, frame, phrase, live=args.live)
            if response is None:
                continue
            rles = response.get("rle") or []
            if isinstance(rles, str):
                rles = [rles]
            scores = response.get("scores") or [1.0] * len(rles)
            for index, rle in enumerate(rles):
                mask = decode_coco_rle(rle, height=height, width=width).astype(bool)
                if int(mask.sum()) < MIN_MASK_PIXELS:
                    continue
                cloud = _clean(transform.apply(points3d[mask & finite]))
                if cloud is None:
                    continue
                hull = MultiPoint([(x, y) for x, y in cloud[:, :2]]).convex_hull
                if not isinstance(hull, Polygon) or hull.area < 0.01:
                    continue
                top = float(np.quantile(cloud[:, 2], 0.95))
                orientation, tilt, _ = _spatial_state(cloud, max(top, 0.0))
                box = hull.minimum_rotated_rectangle
                corners = list(box.exterior.coords)[:4]
                sides = sorted(
                    np.hypot(
                        corners[i][0] - corners[i - 1][0],
                        corners[i][1] - corners[i - 1][1],
                    )
                    for i in range(1, 3)
                )
                entries.append(
                    {
                        "label": phrase,
                        "instance": index,
                        "frame": frame.frame_id,
                        "score": round(
                            float(scores[index]) if index < len(scores) else 1.0, 3
                        ),
                        "points": int(len(cloud)),
                        "height_m": round(top, 2),
                        "size_m": f"{sides[1]:.2f}x{sides[0]:.2f}",
                        "footprint_area_m2": round(float(hull.area), 2),
                        "centroid_xy": [
                            round(float(hull.centroid.x), 2),
                            round(float(hull.centroid.y), 2),
                        ],
                        "camera_dist_m": round(
                            float(Point(0.0, 0.0).distance(hull)), 2
                        ),
                        "orientation_deg": None
                        if orientation is None
                        else round(orientation, 1),
                        "tilt_deg": None if tilt is None else round(tilt, 1),
                        "footprint": [
                            (round(float(x), 3), round(float(y), 3))
                            for x, y in list(hull.exterior.coords)[:-1]
                        ],
                    }
                )

    # One symbol per label: nearest reliable instance, not the biggest —
    # the biggest is usually a far-field smear.
    best_per_label: dict[str, dict] = {}
    for entry in entries:
        if entry["label"] in STRUCTURE_LABELS:
            continue
        if entry["camera_dist_m"] > PLAN_MAX_RANGE_M or entry["height_m"] <= 0.0:
            entry["off_plan_reason"] = (
                "beyond reliable single-view range"
                if entry["camera_dist_m"] > PLAN_MAX_RANGE_M
                else "non-positive height (depth collapse)"
            )
            continue
        current = best_per_label.get(entry["label"])
        if current is None or entry["camera_dist_m"] < current["camera_dist_m"]:
            best_per_label[entry["label"]] = entry
    plan_objects = sorted(
        best_per_label.values(), key=lambda e: -e["footprint_area_m2"]
    )
    off_plan = [e for e in entries if e.get("off_plan_reason")]

    scene_cloud = []
    for frame in frames:
        points3d = np.load(frame.pts3d_path)
        valid = np.load(frame.valid_mask_path).astype(bool)
        finite = valid & np.isfinite(points3d).all(axis=2)
        scene_cloud.append(transform.apply(points3d[finite]))
    cloud = np.vstack(scene_cloud)
    walls = [
        w
        for w in _fit_walls(cloud)
        if min(
            float(np.hypot(*w["start"])), float(np.hypot(*w["end"]))
        )
        < PLAN_MAX_RANGE_M
    ]

    out_dir = run / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inventory.json").write_text(
        json.dumps(
            {"phrases": phrases, "objects": entries, "walls": walls}, indent=2
        )
        + "\n"
    )
    _render_plan(out_dir / "floor_plan.png", walls, plan_objects)
    _write_dxf(out_dir / "floor_plan.dxf", walls, plan_objects)

    print(f"\n{len(entries)} instances | {len(plan_objects)} on the plan | "
          f"{len(off_plan)} rejected | {len(walls)} wall plane(s)")
    for obj in plan_objects:
        print(
            f"  {obj['label']:<22} H={obj['height_m']:>5.2f}m  "
            f"{obj['size_m']:>12}  d_cam={obj['camera_dist_m']:>5.2f}m  "
            f"pts={obj['points']}"
        )
    for wall in walls:
        print(f"  WALL len={wall['length_m']}m h={wall['height_m']}m pts={wall['points']}")
    for entry in off_plan[:6]:
        print(f"  [off-plan] {entry['label']}: {entry['off_plan_reason']}")
    print("\nwrote", out_dir / "floor_plan.png", "and floor_plan.dxf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
