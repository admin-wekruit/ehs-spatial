"""Video POC: deterministic state/trajectory verdicts on a real MEVA clip.

Pipeline (every stage cached on disk, pay once, iterate free):
  1. extract  — sample the AVI at 1 fps over the chosen segment (full-res JPEG)
  2. sam      — fal SAM 3.1 image-rle per frame, prompts "person" and
                "car"->"vehicle" (ordered fallback, LABEL_PROMPTS style),
                disk cache + --live gate + budget cap + 429/5xx backoff
  3. track    — ByteTrack (Apache `trackers` package) over mask bboxes
  4. lift     — ground-plane ray-cast through the clip's real KRTD camera
                model: undistort the bbox bottom-center pixel, cast the ray
                from the camera center to z=0 -> world metres.
                ASSUMPTION: feet on ground. Fails for occluded feet and for
                anything not touching the floor.
  5. judge    — deterministic shapely rules with the mono tier band (0.35 m):
                R1 person-in-keep-clear-zone state timeline,
                R2 person-to-vehicle min distance, R3 person speed.
                Every rule runs twice: on estimated tracks AND on GT geom
                boxes lifted through the SAME ray-cast. The verdict agreement
                rate between the two is the headline honesty number.
  6. render   — report JSON + top-down trajectory PNG + overlay GIF.

GT notes: MEVA geom ids are per-activity actor ids, not global identities
(the same physical car appears under several ids), and geom boxes only exist
while an annotated activity is running, so GT has gaps while people walk
between activities. Both quirks are handled by comparing scene-level states
per timestamp, and disagreements caused by GT gaps are counted separately.
"""

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehs_spatial.providers.sam3 import SAM3_ENDPOINT, decode_coco_rle  # noqa: E402
from ehs_spatial.rules import ERROR_BUDGET_MONO_M  # noqa: E402

COST_PER_CALL_USD = 0.01
MAX_LIVE_CALLS = 150  # hard budget cap for the whole POC
MIN_BOX_HEIGHT_PX = 12  # drop sub-noise masks before tracking

# Ordered prompt candidates per class, canonical first (LABEL_PROMPTS style):
# the first prompt that yields any mask on a frame wins for that frame.
CLASS_PROMPTS: dict[str, tuple[str, ...]] = {
    "person": ("person",),
    "vehicle": ("car", "vehicle"),
}

BAND_M = ERROR_BUDGET_MONO_M  # single-camera tier: ±0.35 m
R2_MIN_SEPARATION_M = 2.0  # person-to-vehicle keep-apart (forklift analog)
R3_MAX_SPEED_MPS = 1.5  # walking-pace limit
ZONE_BUFFER_M = 0.75
# Zone is authored from the GT unload/trunk activity footprint (see
# derive_zone); in production the site owner draws it on the SceneMap.
ZONE_ACTIVITIES = (
    "person_opens_trunk",
    "person_unloads_vehicle",
    "person_closes_trunk",
)

PASS, FAIL, REVIEW, NO_DATA = "PASS", "FAIL", "NEEDS_REVIEW", "NO_DATA"
_SEVERITY = {FAIL: 3, REVIEW: 2, PASS: 1, NO_DATA: 0}


# ---------------------------------------------------------------- KRTD lift
def parse_krtd(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """KRTD: K (3x3), R (3x3), T (3), distortion (k1 k2 p1 p2 k3)."""
    values = [float(token) for token in text.split()]
    if len(values) < 21:
        raise ValueError(f"KRTD needs >=21 values, got {len(values)}")
    K = np.array(values[0:9]).reshape(3, 3)
    R = np.array(values[9:18]).reshape(3, 3)
    T = np.array(values[18:21])
    dist = np.array(values[21:] + [0.0] * (5 - len(values[21:])))
    if not (dist[2] == dist[3] == dist[4] == 0.0):
        raise ValueError("only radial k1/k2 distortion is supported")
    return K, R, T, dist


def camera_center(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    return -R.T @ T


def distort_normalized(x: float, y: float, dist: np.ndarray) -> tuple[float, float]:
    r2 = x * x + y * y
    factor = 1.0 + dist[0] * r2 + dist[1] * r2 * r2
    return x * factor, y * factor


def undistort_normalized(
    xd: float, yd: float, dist: np.ndarray, iterations: int = 25
) -> tuple[float, float]:
    # Fixed-point inversion of the radial model; converges fast for |k1|<0.5.
    x, y = xd, yd
    for _ in range(iterations):
        r2 = x * x + y * y
        factor = 1.0 + dist[0] * r2 + dist[1] * r2 * r2
        x, y = xd / factor, yd / factor
    return x, y


def project_ground_point(
    world_xy: tuple[float, float],
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    dist: np.ndarray,
) -> tuple[float, float]:
    """World (x, y, 0) -> distorted pixel. Inverse of lift; used by tests."""
    point = R @ np.array([world_xy[0], world_xy[1], 0.0]) + T
    if point[2] <= 0:
        raise ValueError("point behind camera")
    xd, yd = distort_normalized(point[0] / point[2], point[1] / point[2], dist)
    return K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]


def lift_pixel(
    u: float,
    v: float,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    dist: np.ndarray,
) -> tuple[float, float] | None:
    """Distorted pixel -> world (x, y) on the z=0 ground plane, or None."""
    x, y = undistort_normalized((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], dist)
    center = camera_center(R, T)
    direction = R.T @ np.array([x, y, 1.0])
    if abs(direction[2]) < 1e-9:
        return None
    t = -center[2] / direction[2]
    if t <= 0:
        return None
    hit = center + t * direction
    return float(hit[0]), float(hit[1])


# ------------------------------------------------------------------ MEVA GT
def parse_meva_lines(path: Path, key: str) -> list[dict]:
    """MEVA Kitware YML lines are `- {...}` python-literal dicts."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        row = ast.literal_eval(line[2:])
        if key in row:
            rows.append(row[key])
    return rows


def load_gt_boxes(
    annotations_dir: Path, clip_stem: str
) -> tuple[dict[int, str], dict[int, list[tuple[int, tuple[float, float, float, float]]]]]:
    """Return (actor_id -> class, frame -> [(actor_id, xyxy), ...])."""
    classes = {}
    for row in parse_meva_lines(annotations_dir / f"{clip_stem}.types.yml", "types"):
        cset = row["cset3"]
        classes[row["id1"]] = max(cset, key=cset.get)
    frames: dict[int, list] = {}
    for row in parse_meva_lines(annotations_dir / f"{clip_stem}.geom.yml", "geom"):
        x1, y1, x2, y2 = (float(value) for value in row["g0"].split())
        frames.setdefault(row["ts0"], []).append((row["id1"], (x1, y1, x2, y2)))
    return classes, frames


def load_gt_activities(annotations_dir: Path, clip_stem: str) -> list[dict]:
    rows = []
    for row in parse_meva_lines(
        annotations_dir / f"{clip_stem}.activities.yml", "act"
    ):
        name = max(row["act2"], key=row["act2"].get)
        span = row["timespan"][0]["tsr0"]
        actors = [
            {"id": actor["id1"], "span": actor["timespan"][0]["tsr0"]}
            for actor in row["actors"]
        ]
        rows.append({"activity": name, "span": span, "actors": actors})
    return rows


def derive_zone(
    activities: list[dict],
    gt_classes: dict[int, str],
    gt_frames: dict[int, list],
    camera: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Polygon:
    """Keep-clear zone = footprint of where the GT unload actually happened.

    The rectangle over the lifted positions of every person actor of the
    trunk/unload activities, buffered ZONE_BUFFER_M. Deterministic and
    documented; a real deployment gets this polygon from the site owner.
    """
    K, R, T, dist = camera
    actor_spans = {}
    for activity in activities:
        if activity["activity"] not in ZONE_ACTIVITIES:
            continue
        for actor in activity["actors"]:
            if gt_classes.get(actor["id"]) == "person":
                actor_spans[actor["id"]] = actor["span"]
    points = []
    for frame, rows in gt_frames.items():
        for actor_id, (x1, y1, x2, y2) in rows:
            span = actor_spans.get(actor_id)
            if span is None or not span[0] <= frame <= span[1]:
                continue
            hit = lift_pixel((x1 + x2) / 2.0, y2, K, R, T, dist)
            if hit is not None:
                points.append(hit)
    if not points:
        raise ValueError("no GT person positions found for the zone activities")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return box(min(xs), min(ys), max(xs), max(ys)).buffer(
        ZONE_BUFFER_M, join_style="mitre"
    )


# ------------------------------------------------------------- frame stages
def extract_frames(video: Path, frame_ids: list[int], frames_dir: Path) -> None:
    missing = [f for f in frame_ids if not (frames_dir / f"f{f:06d}.jpg").is_file()]
    if not missing:
        return
    import cv2  # via `uv run --with trackers` (trackers pulls opencv)

    frames_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(missing)
    capture = cv2.VideoCapture(str(video))
    index = 0
    # Sequential grab: AVI seeking is unreliable, decoding 4k frames is cheap.
    while wanted and capture.grab():
        if index in wanted:
            ok, frame = capture.retrieve()
            if not ok:
                raise RuntimeError(f"failed to decode frame {index}")
            cv2.imwrite(
                str(frames_dir / f"f{index:06d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            wanted.discard(index)
        index += 1
    capture.release()
    if wanted:
        raise RuntimeError(f"video ended before frames {sorted(wanted)}")


def _cache_path(cache_dir: Path, frame_id: int, prompt: str) -> Path:
    return cache_dir / f"f{frame_id:06d}__{prompt.replace(' ', '_')}.json"


def _subscribe_with_backoff(prompt: str, image_uri: str) -> dict:
    import fal_client

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return fal_client.subscribe(
                SAM3_ENDPOINT,
                arguments={
                    "image_url": image_uri,
                    "prompt": prompt,
                    "return_multiple_masks": True,
                    "include_scores": True,
                    "include_boxes": True,
                    "max_masks": 32,
                },
            )
        except Exception as exc:  # 429/5xx and transport errors
            last_error = exc
            time.sleep(2**attempt * 2)
    raise last_error  # type: ignore[misc]


def _image_uri(path: Path) -> str:
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def sam_stage(
    frames_dir: Path, cache_dir: Path, frame_ids: list[int], live: bool
) -> tuple[int, list[str]]:
    """Fill the SAM cache. Returns (live_calls_made, failures)."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    def planned_calls() -> list[tuple[int, str]]:
        calls = []
        for frame_id in frame_ids:
            for prompts in CLASS_PROMPTS.values():
                for ordinal, prompt in enumerate(prompts):
                    path = _cache_path(cache_dir, frame_id, prompt)
                    if path.is_file():
                        # fallback prompt only when every earlier one is empty
                        if json.loads(path.read_text()).get("rle"):
                            break
                        continue
                    calls.append((frame_id, prompt))
                    break  # later fallbacks depend on this result
        return calls

    failures: list[str] = []
    calls_made = 0
    while True:
        calls = planned_calls()
        calls = [call for call in calls if f"{call[0]}:{call[1]}" not in failures]
        if not calls:
            break
        if not live:
            print(
                f"sam: {len(calls)} uncached calls planned "
                f"(~${len(calls) * COST_PER_CALL_USD:.2f}); rerun with --live",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if calls_made + len(calls) > MAX_LIVE_CALLS:
            raise SystemExit(
                f"budget cap: {calls_made} made + {len(calls)} planned > "
                f"{MAX_LIVE_CALLS} max calls"
            )
        for frame_id, prompt in calls:
            uri = _image_uri(frames_dir / f"f{frame_id:06d}.jpg")
            try:
                response = _subscribe_with_backoff(prompt, uri)
            except Exception as exc:
                failures.append(f"{frame_id}:{prompt}")
                print(f"sam: frame {frame_id} prompt {prompt!r} failed: {exc}")
                continue
            calls_made += 1
            with Image.open(frames_dir / f"f{frame_id:06d}.jpg") as image:
                width, height = image.size
            _cache_path(cache_dir, frame_id, prompt).write_text(
                json.dumps(
                    {
                        "rle": response.get("rle"),
                        "scores": response.get("scores"),
                        "width": width,
                        "height": height,
                    }
                ),
                encoding="utf-8",
            )
    return calls_made, failures


def load_detections(
    cache_dir: Path, frame_ids: list[int]
) -> dict[int, dict[str, list[dict]]]:
    """frame -> class -> [{bbox, score}] from the winning cached prompt."""
    detections: dict[int, dict[str, list[dict]]] = {}
    for frame_id in frame_ids:
        per_class: dict[str, list[dict]] = {}
        for class_name, prompts in CLASS_PROMPTS.items():
            rows: list[dict] = []
            for prompt in prompts:
                path = _cache_path(cache_dir, frame_id, prompt)
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text())
                rles = payload.get("rle") or []
                if isinstance(rles, str):
                    rles = [rles]
                scores = payload.get("scores") or []
                for ordinal, serialized in enumerate(rles):
                    mask = decode_coco_rle(
                        serialized,
                        height=payload["height"],
                        width=payload["width"],
                    )
                    ys, xs = np.nonzero(mask)
                    if not len(ys) or ys.max() - ys.min() < MIN_BOX_HEIGHT_PX:
                        continue
                    rows.append(
                        {
                            "bbox": (
                                float(xs.min()),
                                float(ys.min()),
                                float(xs.max() + 1),
                                float(ys.max() + 1),
                            ),
                            "score": float(scores[ordinal])
                            if ordinal < len(scores)
                            else 1.0,
                            "prompt": prompt,
                        }
                    )
                if rows:
                    break  # first prompt with masks wins (ordered fallback)
            per_class[class_name] = rows
        detections[frame_id] = per_class
    return detections


# ----------------------------------------------------------------- tracking
def track_stage(
    detections: dict[int, dict[str, list[dict]]],
    frame_ids: list[int],
    fps: float,
) -> dict[str, list[tuple[int, tuple[float, float, float, float]]]]:
    """ByteTrack per class over mask bboxes -> track -> [(frame, xyxy)]."""
    import supervision as sv
    from trackers import ByteTrackTracker

    tracks: dict[str, list] = {}
    for class_name in CLASS_PROMPTS:
        tracker = ByteTrackTracker(
            # The library counts the lost-track budget in 30 fps frame units
            # whatever frame_rate says (budget = buffer / 30 seconds), so a
            # 5-second keepalive across 1 fps samples needs 150, not 5.
            lost_track_buffer=150,
            frame_rate=1.0,  # tracker steps once per sampled frame
            minimum_consecutive_frames=2,  # kill one-frame ghost masks
            track_activation_threshold=0.0,  # SAM prompt output is precise
            high_conf_det_threshold=0.0,
        )
        for frame_id in frame_ids:
            rows = detections.get(frame_id, {}).get(class_name, [])
            if rows:
                current = sv.Detections(
                    xyxy=np.array([row["bbox"] for row in rows], dtype=float),
                    confidence=np.array([row["score"] for row in rows]),
                )
            else:
                current = sv.Detections.empty()
            tracked = tracker.update(current, timestamp=frame_id / fps)
            for bbox, track_id in zip(
                tracked.xyxy, tracked.tracker_id, strict=True
            ):
                if int(track_id) < 0:
                    continue  # immature (first-frame) detection, no identity yet
                tracks.setdefault(f"{class_name}-{int(track_id)}", []).append(
                    (frame_id, tuple(float(value) for value in bbox))
                )
    return tracks


def lift_tracks(
    tracks: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    camera: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, dict[int, dict]]:
    """track -> frame -> {xy, edge}; edge = lifted bottom bbox edge (vehicles).

    Bottom-center lift assumes feet/wheels on the ground plane; the bottom
    corners give a ground segment approximating the vehicle's near side.
    """
    K, R, T, dist = camera
    lifted: dict[str, dict[int, dict]] = {}
    for track_id, samples in tracks.items():
        for frame_id, (x1, y1, x2, y2) in samples:
            center = lift_pixel((x1 + x2) / 2.0, y2, K, R, T, dist)
            if center is None:
                continue
            entry: dict = {"xy": center}
            if track_id.startswith("vehicle"):
                left = lift_pixel(x1, y2, K, R, T, dist)
                right = lift_pixel(x2, y2, K, R, T, dist)
                if left is not None and right is not None and left != right:
                    entry["edge"] = (left, right)
            lifted.setdefault(track_id, {})[frame_id] = entry
    return lifted


def gt_to_tracks(
    gt_classes: dict[int, str],
    gt_frames: dict[int, list],
    frame_ids: list[int],
) -> dict[str, list[tuple[int, tuple[float, float, float, float]]]]:
    """GT geom boxes at the sampled frames, person/vehicle only (bags skipped),
    shaped like estimated tracks so they run through the same lift + rules."""
    wanted = set(frame_ids)
    tracks: dict[str, list] = {}
    for frame_id in sorted(gt_frames):
        if frame_id not in wanted:
            continue
        for actor_id, bbox in gt_frames[frame_id]:
            class_name = gt_classes.get(actor_id)
            if class_name not in ("person", "vehicle"):
                continue
            tracks.setdefault(f"{class_name}-gt{actor_id}", []).append(
                (frame_id, bbox)
            )
    return tracks


# ------------------------------------------------------------------ judging
def banded_verdict(value: float, threshold: float, band: float, fail_low: bool) -> str:
    """fail_low: values below threshold fail (min separation); else above."""
    if fail_low:
        if value < threshold - band:
            return FAIL
        if value > threshold + band:
            return PASS
        return REVIEW
    if value > threshold + band:
        return FAIL
    if value < threshold - band:
        return PASS
    return REVIEW


def zone_verdict(zone: Polygon, xy: tuple[float, float], band: float) -> str:
    point = Point(xy)
    boundary_distance = point.distance(zone.exterior)
    if zone.contains(point):
        return FAIL if boundary_distance > band else REVIEW
    return PASS if boundary_distance > band else REVIEW


def _points_at(
    lifted: dict[str, dict[int, dict]], frame_id: int, prefix: str
) -> list[tuple[str, dict]]:
    return [
        (track_id, samples[frame_id])
        for track_id, samples in lifted.items()
        if track_id.startswith(prefix) and frame_id in samples
    ]


def judge(
    lifted: dict[str, dict[int, dict]],
    zone: Polygon,
    frame_ids: list[int],
    fps: float,
) -> dict:
    step_seconds = (frame_ids[1] - frame_ids[0]) / fps
    speed_band = 2 * BAND_M / step_seconds  # both endpoints off by the band
    r1, r2, r3 = {}, {}, {}
    r2_values, r3_values = {}, {}
    for ordinal, frame_id in enumerate(frame_ids):
        persons = _points_at(lifted, frame_id, "person")
        vehicles = _points_at(lifted, frame_id, "vehicle")

        # R1: nobody inside the keep-clear zone. No person -> zone clear.
        if persons:
            verdicts = [zone_verdict(zone, entry["xy"], BAND_M) for _, entry in persons]
            r1[frame_id] = max(verdicts, key=lambda v: _SEVERITY[v])
        else:
            r1[frame_id] = PASS

        # R2: min person-to-vehicle distance (person point to the vehicle's
        # lifted bottom-edge segment when available, else its point).
        if persons and vehicles:
            distances = []
            for _, person in persons:
                point = Point(person["xy"])
                for _, vehicle in vehicles:
                    geometry = (
                        LineString(vehicle["edge"])
                        if "edge" in vehicle
                        else Point(vehicle["xy"])
                    )
                    distances.append(point.distance(geometry))
            minimum = min(distances)
            r2_values[frame_id] = round(minimum, 3)
            r2[frame_id] = banded_verdict(
                minimum, R2_MIN_SEPARATION_M, BAND_M, fail_low=True
            )
        else:
            r2[frame_id] = NO_DATA

        # R3: person speed between consecutive samples of the same track.
        speeds = []
        if ordinal:
            previous_frame = frame_ids[ordinal - 1]
            for track_id, entry in persons:
                before = lifted[track_id].get(previous_frame)
                if before is None:
                    continue
                dx = entry["xy"][0] - before["xy"][0]
                dy = entry["xy"][1] - before["xy"][1]
                speeds.append((dx * dx + dy * dy) ** 0.5 / step_seconds)
        if speeds:
            fastest = max(speeds)
            r3_values[frame_id] = round(fastest, 3)
            r3[frame_id] = banded_verdict(
                fastest, R3_MAX_SPEED_MPS, speed_band, fail_low=False
            )
        else:
            r3[frame_id] = NO_DATA
    return {
        "R1_zone": r1,
        "R2_min_distance": r2,
        "R3_speed": r3,
        "R2_values_m": r2_values,
        "R3_values_mps": r3_values,
        "speed_band_mps": round(speed_band, 3),
    }


def agreement(
    estimated: dict[int, str],
    reference: dict[int, str],
    frame_ids: list[int],
    gt_presence: dict[int, dict[str, int]] | None = None,
) -> dict:
    matches = 0
    both_observed = both_observed_matches = 0
    disagreements = []
    for frame_id in frame_ids:
        est, ref = estimated[frame_id], reference[frame_id]
        if est == ref:
            matches += 1
        else:
            entry = {"frame": frame_id, "estimated": est, "gt": ref}
            if gt_presence is not None:
                # How many GT boxes existed at this frame: 0 persons usually
                # means a MEVA annotation gap (actors are only boxed while an
                # activity runs), not an empty scene.
                entry["gt_boxes"] = gt_presence.get(frame_id, {})
            disagreements.append(entry)
        if NO_DATA not in (est, ref):
            both_observed += 1
            both_observed_matches += est == ref
    return {
        "rate": round(matches / len(frame_ids), 3),
        "rate_both_observed": round(both_observed_matches / both_observed, 3)
        if both_observed
        else None,
        "n": len(frame_ids),
        "n_both_observed": both_observed,
        "disagreements": disagreements,
    }


ATE_GATE_M = 3.0


def ate_stats(
    estimated: dict[str, dict[int, dict]],
    reference: dict[str, dict[int, dict]],
    frame_ids: list[int],
) -> dict:
    """Nearest-GT-per-estimate position error, gated at ATE_GATE_M.

    GT ids are per-activity duplicates, so per-point nearest-neighbour is the
    honest match, not track pairing. The gate keeps the error statistic to
    objects GT actually annotates: MEVA only boxes activity actors, so an
    estimate with no GT inside the gate is either a false track OR a real but
    unannotated object (e.g. the parked cars across the street) — counted, not
    scored.
    """
    stats = {}
    for class_name in CLASS_PROMPTS:
        errors, beyond_gate, missed_gt = [], 0, 0
        for frame_id in frame_ids:
            est_points = [
                entry["xy"] for _, entry in _points_at(estimated, frame_id, class_name)
            ]
            gt_points = [
                entry["xy"] for _, entry in _points_at(reference, frame_id, class_name)
            ]
            for est_xy in est_points:
                nearest = min(
                    (
                        ((est_xy[0] - g[0]) ** 2 + (est_xy[1] - g[1]) ** 2) ** 0.5
                        for g in gt_points
                    ),
                    default=None,
                )
                if nearest is None or nearest > ATE_GATE_M:
                    beyond_gate += 1
                else:
                    errors.append(nearest)
            for gt_xy in gt_points:
                if not est_points or (
                    min(
                        ((gt_xy[0] - e[0]) ** 2 + (gt_xy[1] - e[1]) ** 2) ** 0.5
                        for e in est_points
                    )
                    > ATE_GATE_M
                ):
                    missed_gt += 1
        stats[class_name] = {
            "median_m": round(float(np.median(errors)), 3) if errors else None,
            "p90_m": round(float(np.percentile(errors, 90)), 3) if errors else None,
            "n_points": len(errors),
            "est_points_beyond_gate": beyond_gate,
            "gt_points_missed_beyond_gate": missed_gt,
            "gate_m": ATE_GATE_M,
        }
    return stats


# ---------------------------------------------------------------- rendering
_TRACK_COLORS = [
    (214, 69, 65),
    (65, 131, 215),
    (38, 166, 91),
    (243, 156, 18),
    (155, 89, 182),
    (22, 160, 133),
    (211, 84, 0),
    (52, 73, 94),
]


def _color(track_id: str) -> tuple[int, int, int]:
    return _TRACK_COLORS[abs(hash(track_id)) % len(_TRACK_COLORS)]


def render_topdown(
    estimated: dict[str, dict[int, dict]],
    reference: dict[str, dict[int, dict]],
    zone: Polygon,
    r1_estimated: dict[int, str],
    camera_xy: tuple[float, float],
    path: Path,
) -> None:
    # Viewport: the annotated action area (GT points + zone). Estimated
    # far-field tracks (the unannotated parked cars across the street) would
    # compress the action into a corner, so tracks entirely outside the
    # viewport are counted in a footnote instead of drawn.
    points = [
        entry["xy"] for samples in reference.values() for entry in samples.values()
    ]
    xs = [p[0] for p in points] + [x for x, _ in zone.exterior.coords]
    ys = [p[1] for p in points] + [y for _, y in zone.exterior.coords]
    margin = 3.0
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    size = 900
    scale = (size - 40) / max(x1 - x0, y1 - y0)

    def pixel(xy: tuple[float, float]) -> tuple[float, float]:
        return 20 + (xy[0] - x0) * scale, size - 20 - (xy[1] - y0) * scale

    def in_view(xy: tuple[float, float]) -> bool:
        return x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1

    image = Image.new("RGB", (size, size), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [pixel(p) for p in zone.exterior.coords], outline=(200, 60, 60), width=3
    )
    zone_pixel = pixel((zone.centroid.x, zone.centroid.y))
    draw.text(
        (zone_pixel[0] - 30, zone_pixel[1]), "keep-clear", fill=(200, 60, 60)
    )
    omitted = 0
    for lifted, is_gt in ((reference, True), (estimated, False)):
        for track_id, samples in sorted(lifted.items()):
            visible = [
                samples[f]["xy"] for f in sorted(samples) if in_view(samples[f]["xy"])
            ]
            if not is_gt and not visible:
                omitted += 1
                continue
            trail = [pixel(xy) for xy in visible]
            if len(trail) < 2:
                continue
            color = (170, 170, 170) if is_gt else _color(track_id)
            width = 2 if is_gt else 3
            draw.line(trail, fill=color, width=width)
            marker = "gt" if is_gt else track_id
            draw.text(trail[-1], marker, fill=color)
    for track_id, samples in estimated.items():
        if not track_id.startswith("person"):
            continue
        for frame_id, entry in samples.items():
            if r1_estimated.get(frame_id) == FAIL and zone.contains(Point(entry["xy"])):
                px, py = pixel(entry["xy"])
                draw.ellipse([px - 4, py - 4, px + 4, py + 4], outline=(200, 30, 30), width=2)
    draw.text((20, 8), "estimated = color, GT-lifted = gray, red rings = R1 FAIL", fill=(60, 60, 60))
    if omitted:
        draw.text(
            (20, size - 14),
            f"{omitted} estimated far-field track(s) outside view "
            "(unannotated parked cars across the street)",
            fill=(120, 120, 120),
        )
    bar = pixel((x1 - margin - 5.0, y0 + margin / 2)), pixel((x1 - margin, y0 + margin / 2))
    draw.line([bar[0], bar[1]], fill=(0, 0, 0), width=3)
    draw.text((bar[0][0], bar[0][1] - 16), "5 m", fill=(0, 0, 0))
    if in_view(camera_xy):
        px, py = pixel(camera_xy)
        draw.regular_polygon((px, py, 7), 3, fill=(0, 0, 0))
        draw.text((px + 8, py - 6), "camera", fill=(0, 0, 0))
    image.save(path)


_CLASS_TINTS = {"person": (230, 60, 40), "vehicle": (40, 110, 230)}


def _paint_masks(frame: Image.Image, cache_dir: Path, frame_id: int) -> None:
    """Alpha-blend the winning-prompt SAM masks, tinted per class."""
    pixels = np.asarray(frame, dtype=np.uint16)
    for class_name, prompts in CLASS_PROMPTS.items():
        tint = np.array(_CLASS_TINTS[class_name], dtype=np.uint16)
        for prompt in prompts:
            path = _cache_path(cache_dir, frame_id, prompt)
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            rles = payload.get("rle") or []
            if isinstance(rles, str):
                rles = [rles]
            for serialized in rles:
                mask = decode_coco_rle(
                    serialized, height=payload["height"], width=payload["width"]
                ).astype(bool)
                pixels[mask] = (pixels[mask] * 3 + tint * 2) // 5
            if rles:
                break  # same ordered-fallback rule as load_detections
    frame.paste(Image.fromarray(pixels.astype(np.uint8)))


def render_overlay(
    frames_dir: Path,
    cache_dir: Path,
    tracks: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    judged: dict,
    frame_ids: list[int],
    fps: float,
    gif_path: Path,
    strip_path: Path,
    strip_frames: list[int],
    strip_crop: tuple[int, int, int, int],
) -> None:
    boxes_at: dict[int, list[tuple[str, tuple]]] = {}
    for track_id, samples in tracks.items():
        for frame_id, bbox in samples:
            boxes_at.setdefault(frame_id, []).append((track_id, bbox))
    rendered: dict[int, Image.Image] = {}
    for frame_id in frame_ids:
        with Image.open(frames_dir / f"f{frame_id:06d}.jpg") as source:
            frame = source.convert("RGB")
        _paint_masks(frame, cache_dir, frame_id)
        draw = ImageDraw.Draw(frame)
        for track_id, bbox in boxes_at.get(frame_id, []):
            color = _color(track_id)
            draw.rectangle(bbox, outline=color, width=4)
            draw.text((bbox[0], max(0, bbox[1] - 16)), track_id, fill=color)
        r2_value = judged["R2_values_m"].get(frame_id)
        status = " | ".join(
            f"{rule.split('_')[0]}:{judged[rule][frame_id]}"
            for rule in ("R1_zone", "R2_min_distance", "R3_speed")
        )
        banner = f"t={frame_id / fps:.1f}s  frame {frame_id}  {status}"
        if r2_value is not None:
            banner += f"  person-vehicle {r2_value:.2f} m"
        draw.rectangle([0, 0, frame.width, 26], fill=(0, 0, 0))
        draw.text((8, 6), banner, fill=(255, 255, 255))
        rendered[frame_id] = frame
    small = [
        rendered[f].resize((720, round(720 * rendered[f].height / rendered[f].width)))
        for f in frame_ids
    ]
    small[0].save(
        gif_path,
        save_all=True,
        append_images=small[1:],
        duration=500,
        loop=0,
    )
    # The strip zooms into the action region — at 1080p the activity occupies
    # a small corner of the exterior view, so full frames hide the masks.
    columns = [rendered[f].crop(strip_crop) for f in strip_frames if f in rendered]
    if columns:
        width = 960
        tiles = [
            column.resize((width, round(width * column.height / column.width)))
            for column in columns
        ]
        strip = Image.new("RGB", (width, sum(tile.height for tile in tiles)))
        offset = 0
        for tile in tiles:
            strip.paste(tile, (0, offset))
            offset += tile.height
        strip.save(strip_path)


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        default="outputs/datasets/meva/videos/"
        "2018-03-05.13-15-00.13-20-00.bus.G506.r13.avi",
    )
    parser.add_argument(
        "--krtd",
        default="outputs/datasets/meva/krtd/"
        "2018-03-05.13-15-00.13-20-00.bus.G506.krtd",
    )
    parser.add_argument(
        "--annotations", default="outputs/datasets/meva/annotations"
    )
    parser.add_argument(
        "--clip-stem", default="2018-03-05.13-15-00.13-20-00.bus.G506"
    )
    parser.add_argument("--out", default="outputs/video_poc_v1")
    parser.add_argument("--start", type=int, default=2500)
    parser.add_argument("--end", type=int, default=3800)
    parser.add_argument("--step", type=int, default=30, help="30 = 1 fps")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame_ids = list(range(args.start, args.end, args.step))
    camera = parse_krtd(Path(args.krtd).read_text(encoding="utf-8"))
    center = camera_center(camera[1], camera[2])

    extract_frames(Path(args.video), frame_ids, out / "frames")
    calls_made, failures = sam_stage(
        out / "frames", out / "sam_cache", frame_ids, args.live
    )
    detections = load_detections(out / "sam_cache", frame_ids)

    estimated_tracks = track_stage(detections, frame_ids, args.fps)
    estimated = lift_tracks(estimated_tracks, camera)

    gt_classes, gt_frames = load_gt_boxes(Path(args.annotations), args.clip_stem)
    activities = load_gt_activities(Path(args.annotations), args.clip_stem)
    zone = derive_zone(activities, gt_classes, gt_frames, camera)
    gt_tracks = gt_to_tracks(gt_classes, gt_frames, frame_ids)
    reference = lift_tracks(gt_tracks, camera)

    judged_estimated = judge(estimated, zone, frame_ids, args.fps)
    judged_reference = judge(reference, zone, frame_ids, args.fps)
    gt_presence = {
        frame_id: {
            class_name: len(_points_at(reference, frame_id, class_name))
            for class_name in CLASS_PROMPTS
        }
        for frame_id in frame_ids
    }
    agreements = {
        rule: agreement(
            judged_estimated[rule], judged_reference[rule], frame_ids, gt_presence
        )
        for rule in ("R1_zone", "R2_min_distance", "R3_speed")
    }
    ate = ate_stats(estimated, reference, frame_ids)

    render_topdown(
        estimated,
        reference,
        zone,
        judged_estimated["R1_zone"],
        (float(center[0]), float(center[1])),
        out / "trajectories_topdown.png",
    )
    render_overlay(
        out / "frames",
        out / "sam_cache",
        estimated_tracks,
        judged_estimated,
        frame_ids,
        args.fps,
        out / "overlay.gif",
        out / "overlay_strip.png",
        # exit-vehicle / unload FAIL / talkers in zone / walk away
        strip_frames=[2650, 2920, 3220, 3760],
        strip_crop=(1400, 0, 1920, 520),
    )

    report = {
        "clip": args.clip_stem,
        "segment_frames": [args.start, args.end, args.step],
        "camera_center_world": [round(float(v), 3) for v in center],
        "band_m": BAND_M,
        "thresholds": {
            "R2_min_separation_m": R2_MIN_SEPARATION_M,
            "R3_max_speed_mps": R3_MAX_SPEED_MPS,
            "speed_band_mps": judged_estimated["speed_band_mps"],
        },
        "zone_wkt": zone.wkt,
        "sam": {
            "live_calls_this_run": calls_made,
            "cost_this_run_usd": round(calls_made * COST_PER_CALL_USD, 2),
            "cached_files": len(list((out / "sam_cache").glob("*.json"))),
            "failed_calls": failures,
        },
        "tracks": {
            "estimated": {
                track_id: len(samples)
                for track_id, samples in sorted(estimated_tracks.items())
            },
            "gt": {
                track_id: len(samples)
                for track_id, samples in sorted(gt_tracks.items())
            },
        },
        "trajectories_estimated": {
            track_id: {
                str(frame): [round(v, 3) for v in entry["xy"]]
                for frame, entry in sorted(samples.items())
            }
            for track_id, samples in sorted(estimated.items())
        },
        "timelines_estimated": {
            rule: {str(f): judged_estimated[rule][f] for f in frame_ids}
            for rule in ("R1_zone", "R2_min_distance", "R3_speed")
        },
        "timelines_gt": {
            rule: {str(f): judged_reference[rule][f] for f in frame_ids}
            for rule in ("R1_zone", "R2_min_distance", "R3_speed")
        },
        "R2_min_distance_m": {
            "estimated": {str(f): v for f, v in judged_estimated["R2_values_m"].items()},
            "gt": {str(f): v for f, v in judged_reference["R2_values_m"].items()},
        },
        "R3_speed_mps": {
            "estimated": {str(f): v for f, v in judged_estimated["R3_values_mps"].items()},
            "gt": {str(f): v for f, v in judged_reference["R3_values_mps"].items()},
        },
        "verdict_agreement": agreements,
        "ate": ate,
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report: {(out / 'report.json').resolve()}")
    for rule, stats in agreements.items():
        print(
            f"{rule}: agreement {stats['rate']} "
            f"(both-observed {stats['rate_both_observed']}, "
            f"{len(stats['disagreements'])} disagreements)"
        )
    for class_name, stats in ate.items():
        print(f"ATE {class_name}: median {stats['median_m']} m over {stats['n_points']} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
