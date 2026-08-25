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


def _render_plan(path: Path, walls: list[dict], objects: list[dict], run_id: str) -> None:
    """A drawing, not a scatter plot: title block, coordinate grid, oriented
    object rectangles with dimension strings, dimensioned clearances between
    the closest pairs, and hatched walls."""
    from shapely.geometry import LineString
    from shapely.ops import nearest_points as _nearest

    W, H = 1600, 1240
    plot_w, plot_h = 1180, 1120
    left, top = 40, 60
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)

    xs = [0.0] + [x for o in objects for x, _ in o["footprint"]]
    ys = [0.0] + [y for o in objects for _, y in o["footprint"]]
    min_x, max_x = min(xs) - 1.2, max(xs) + 1.2
    min_y, max_y = min(ys) - 1.2, max(ys) + 1.2
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    min_x -= (span - (max_x - min_x)) / 2
    min_y -= (span - (max_y - min_y)) / 2
    max_x, max_y = min_x + span, min_y + span
    scale = min(plot_w, plot_h) / span

    def px(point):
        return (
            left + (point[0] - min_x) * scale,
            top + plot_h - (point[1] - min_y) * scale,
        )

    # sheet border and plot frame
    draw.rectangle([12, 12, W - 12, H - 12], outline="#111111", width=3)
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline="#555555", width=1)

    for gx in np.arange(np.ceil(min_x), np.floor(max_x) + 1):
        a, b = px((gx, min_y)), px((gx, max_y))
        draw.line([a, b], fill="#e8e8e8")
        draw.text((a[0] - 8, top + plot_h + 4), f"{gx:.0f}", fill="#9a9a9a")
    for gy in np.arange(np.ceil(min_y), np.floor(max_y) + 1):
        a, b = px((min_x, gy)), px((max_x, gy))
        draw.line([a, b], fill="#e8e8e8")
        draw.text((left - 26, a[1] - 6), f"{gy:.0f}", fill="#9a9a9a")

    # walls: thick line plus hatch ticks on the occupied side
    for wall in walls:
        clipped = _clip_segment(wall["start"], wall["end"], min_x, min_y, max_x, max_y)
        if clipped is None:
            continue
        start, end = clipped
        draw.line([px(start), px(end)], fill="#1a1a1a", width=9)
        direction = np.asarray(end) - np.asarray(start)
        length = float(np.hypot(*direction))
        if length < 1e-6:
            continue
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        for offset in np.arange(0.12, length, 0.32):
            base = np.asarray(start) + unit * offset
            draw.line([px(base), px(base + normal * 0.18)], fill="#7a7a7a", width=2)
        mid = np.asarray(start) + unit * (length / 2) + normal * 0.35
        draw.text(px(mid), f"WALL  {wall['length_m']:.2f} m  h {wall['height_m']:.2f}",
                  fill="#1a1a1a", anchor="mm")

    def dimension(a, b, text, colour, offset=0.0):
        """Extension lines + arrowed dimension line, CAD convention."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        direction = b - a
        length = float(np.hypot(*direction))
        if length < 1e-6:
            return
        unit = direction / length
        normal = np.array([-unit[1], unit[0]]) * offset
        pa, pb = px(a + normal), px(b + normal)
        if offset:
            draw.line([px(a), pa], fill=colour, width=1)
            draw.line([px(b), pb], fill=colour, width=1)
        draw.line([pa, pb], fill=colour, width=2)
        for point, sign in ((pa, 1), (pb, -1)):
            tip = np.asarray(point)
            back = tip + sign * np.array([unit[0], -unit[1]]) * 11
            side = np.array([-unit[1], -unit[0]]) * 4.5
            draw.polygon([tuple(tip), tuple(back + side), tuple(back - side)], fill=colour)
        mid = (a + b) / 2 + normal
        draw.text(px(mid), text, fill=colour, anchor="mm")

    palette = ["#c1121f", "#1d4ed8", "#047857", "#b45309", "#6d28d9",
               "#0e7490", "#9d174d", "#4d7c0f", "#7c2d12", "#334155"]
    legend = []
    for index, obj in enumerate(objects):
        colour = palette[index % len(palette)]
        box = Polygon(obj["footprint"]).minimum_rotated_rectangle
        ring = list(box.exterior.coords)
        draw.polygon([px(p) for p in ring], outline=colour, width=3)
        draw.line([px(p) for p in obj["footprint"] + [obj["footprint"][0]]],
                  fill=colour, width=1)
        cx, cy = obj["centroid_xy"]
        draw.text(px((cx, cy)), f"{index + 1}", fill=colour, anchor="mm")
        # dimension the two sides of the oriented box
        for i in (0, 1):
            p0, p1 = np.asarray(ring[i]), np.asarray(ring[i + 1])
            side = float(np.hypot(*(p1 - p0)))
            if side * scale > 46:
                dimension(p0, p1, f"{side:.2f}", colour, offset=0.14)
        legend.append((index + 1, colour, obj))

    # clearance dimensions for the closest object pairs — what a rule reads
    pairs = []
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            pa, pb = Polygon(a["footprint"]), Polygon(b["footprint"])
            gap = pa.distance(pb)
            # Sub-decimetre gaps are parts of one thing (worker / jumpsuit /
            # hard hat); dimensioning them buries the drawing in arrows.
            if pa.intersects(pb) or gap < 0.15:
                continue
            pairs.append((gap, a, b, pa, pb))
    pairs.sort(key=lambda item: item[0])
    for gap, a, b, pa, pb in pairs[:3]:
        p1, p2 = _nearest(pa, pb)
        dimension((p1.x, p1.y), (p2.x, p2.y), f"{gap:.2f} m", "#047857")
    camera = px((0.0, 0.0))
    draw.ellipse([camera[0] - 7, camera[1] - 7, camera[0] + 7, camera[1] + 7],
                 fill="#c1121f")
    draw.line([(camera[0] - 14, camera[1]), (camera[0] + 14, camera[1])],
              fill="#c1121f", width=1)
    draw.line([(camera[0], camera[1] - 14), (camera[0], camera[1] + 14)],
              fill="#c1121f", width=1)
    draw.text((camera[0] + 12, camera[1] + 8), "CAM (0,0)", fill="#c1121f")

    # legend + title block on the right rail
    rail = left + plot_w + 24
    draw.text((rail, top), "OBJECT SCHEDULE", fill="#111111")
    draw.line([(rail, top + 16), (W - 24, top + 16)], fill="#111111", width=2)
    row = top + 26
    for number, colour, obj in legend:
        draw.rectangle([rail, row + 3, rail + 9, row + 12], fill=colour)
        draw.text((rail + 16, row), f"{number}. {obj['label']}", fill="#111111")
        draw.text((rail + 16, row + 14),
                  f"H {obj['height_m']:.2f}  {obj['size_m']}  d {obj['camera_dist_m']:.2f}",
                  fill="#5b5b5b")
        row += 34
    block_top = H - 150
    draw.rectangle([rail, block_top, W - 24, H - 24], outline="#111111", width=2)
    draw.line([(rail, block_top + 26), (W - 24, block_top + 26)], fill="#111111")
    draw.text((rail + 8, block_top + 6), "MEASURED FLOOR PLAN", fill="#111111")
    for offset, line in enumerate([
        f"run      {run_id}",
        f"objects  {len(objects)}   walls {len(walls)}",
        "units    metres, floor frame",
        "origin   camera position",
        "source   photogrammetry, not a survey",
    ]):
        draw.text((rail + 8, block_top + 34 + offset * 17), line, fill="#333333")
    bar_m = 1.0
    bar = px((min_x + 0.4, min_y + 0.35))
    draw.line([bar, (bar[0] + bar_m * scale, bar[1])], fill="#111111", width=5)
    draw.text((bar[0], bar[1] + 8), "1 m", fill="#111111")
    image.save(path)


def _write_scene(path: Path, run_id: str, entries: list[dict]) -> None:
    """A SceneMap over the whole inventory, so compiled policies can be
    evaluated against every enumerated class rather than only the production
    vocabulary."""
    from ehs_spatial.contracts import Entity3D, SceneMap

    entities = []
    for index, entry in enumerate(entries, start=1):
        if entry["height_m"] <= 0:
            continue
        entities.append(
            Entity3D(
                entity_id=f"inv-{index:03d}",
                label=entry["label"],
                observation_ids=[f"inv-obs-{index:03d}"],
                centroid_xyz=(
                    float(entry["centroid_xy"][0]),
                    float(entry["centroid_xy"][1]),
                    float(entry["height_m"]) / 2,
                ),
                footprint_xy=[(float(x), float(y)) for x, y in entry["footprint"]],
                height_m=float(entry["height_m"]),
                evidence_frame_ids=[entry["frame"]],
                orientation_deg=entry.get("orientation_deg"),
                tilt_deg=entry.get("tilt_deg"),
            )
        )
    scene = SceneMap(
        run_id=f"{run_id}-inventory",
        floor_plane=(0.0, 0.0, 1.0, 0.0),
        scale_source="camera_height",
        scale_factor=1.0,
        fence_polygon=[],
        entities=entities,
        facts=[],
        warnings=["inventory scene: exploration vocabulary, not the production path"],
    )
    path.write_text(scene.model_dump_json(indent=2) + "\n")


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
    _render_plan(out_dir / "floor_plan.png", walls, plan_objects, args.run)
    _write_dxf(out_dir / "floor_plan.dxf", walls, plan_objects)
    _write_scene(out_dir / "scene.json", args.run, entries)

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
