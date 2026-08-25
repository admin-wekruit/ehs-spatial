"""MTMC trajectory pilot — does the harness thesis extend to TIME?

Three layers on Warehouse_016 (nvidia/PhysicalAI-SmartSpaces, CC-BY-4.0,
SYNTHETIC Omniverse/Cosmos imagery — no real photos with GT exist in this
collection):

1. Denser sampling: the mtmc_eval distance protocol re-scored at 15
   timestamps per camera (60 mono frames, 5x the n=140 baseline).
2. Trajectories: per-object world positions lifted from our geometry
   (MapAnything mono + MoGe-2 anchor, camera->world via calibration.json)
   compared against the dataset's world-coordinate GT trajectories,
   ATE-style, per camera and pooled; plus a multiview variant that feeds
   all 4 same-timestamp frames to MapAnything jointly.
3. Deterministic temporal judgment: three rules evaluated per timestamp
   from BOTH the GT trajectories and OUR estimated trajectories, with the
   capture tier's error band as the honesty buffer. The GT-vs-estimated
   verdict delta is the finding.

Honesty notes (owner red line):
- Object identity comes from GT in this pilot. We validate metrology +
  judgment, NOT a tracker; the tracker slot is a commodity scanned
  separately.
- Estimated positions are medians of the VISIBLE surface inside a GT box;
  GT locations are object centres. The surface-to-centre offset is part of
  the measured error, not corrected away.
- Camera_07's known mono distortion is reported, not tuned around.

Map convention (verified against the floor plan: inverse-homography of
the taped-zone corners in Camera_000000.png lands on the same corners in
map.png only with the y-flip):
  px = (x + tx) * scaleFactor,  py = MAP_H - (y + ty) * scaleFactor

House pattern: paid calls are disk-cached under outputs/mtmc_v1 in the
MAIN repo, spend is gated behind --live with printed counts, per-image
failures skip. ZERO SAM, ZERO Gemini.

  uv run --env-file .env --with imageio-ffmpeg python scripts/trajectory_eval.py
  uv run --env-file .env --with imageio-ffmpeg python scripts/trajectory_eval.py --live
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).parent))
import mtmc_eval as base  # noqa: E402  (house helpers: GT, frames, geometry)

from ehs_spatial.rules import (  # noqa: E402
    ERROR_BUDGET_MONO_M,
    ERROR_BUDGET_MULTIVIEW_M,
)

DATA = base.DATA
WORK = base.WORK
CAMERAS = base.CAMERAS
# 15 evenly spread timestamps (20 s apart over the 5-minute sequence);
# includes the baseline's 0/3600/7200 so their geometry cache is reused.
FRAME_IDS = tuple(range(0, 9000, 600))
STEP_S = 600 / base.FPS  # 20 s between samples
# Multiview joint reconstructions at 4 of the 15 timestamps.
MV_FRAME_IDS = (0, 2400, 4800, 7200)

# --- Layer 3 rule constants (choices documented in the review doc) -------
# R1 keep-clear zone: a 4x4 m square centred on the parked Transporter
# (AMR, object 349, static at (-1.77, -9.13) for the whole sequence) —
# "pedestrians keep clear of the AMR staging area". Chosen because it is
# a plausible EHS zone AND GT traffic actually crosses it at sampled
# timestamps, so all three verdict classes get exercised.
ZONE = Polygon(
    [(-3.77, -11.13), (0.23, -11.13), (0.23, -7.13), (-3.77, -7.13)]
)
ZONE_EXEMPT_IDS = {349}  # the AMR belongs in its own staging area
# R2: min distance over time between Person 699 and Transporter 349
# (their GT distance sweeps 1.7-11 m across the samples, crossing the
# threshold in both directions).
R2_PAIR = (699, 349)
R2_THRESHOLD_M = 2.0
# R3: average speed over each 20 s window against a movement limit.
# GT window speeds span 0-0.58 m/s, so 0.25 m/s exercises both sides;
# the rule semantics (displacement/dt, banded) are what is validated,
# not any specific site speed limit.
R3_SPEED_LIMIT_MPS = 0.25
# Two banded endpoints 20 s apart: worst-case speed error is
# 2 * position band / dt.
SPEED_BAND_MPS = 2 * ERROR_BUDGET_MONO_M / STEP_S

PASS, FAIL, NEEDS_REVIEW = "PASS", "FAIL", "NEEDS_REVIEW"


# --- pure geometry -------------------------------------------------------

def extrinsic_parts(sensor: dict) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(sensor["extrinsicMatrix"], dtype=float)
    return matrix[:, :3], matrix[:, 3]


def camera_from_world(rotation: np.ndarray, translation: np.ndarray,
                      points: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation


def world_from_camera(rotation: np.ndarray, translation: np.ndarray,
                      points: np.ndarray) -> np.ndarray:
    return (points - translation) @ rotation


def world_to_map(xy: np.ndarray, scale: float, tx: float, ty: float,
                 map_h: int) -> tuple[float, float]:
    return ((xy[0] + tx) * scale, map_h - (xy[1] + ty) * scale)


def zone_depth(xy, polygon: Polygon) -> float:
    """Signed distance to the zone boundary: positive inside."""
    point = Point(float(xy[0]), float(xy[1]))
    distance = point.distance(polygon.exterior)
    return distance if polygon.contains(point) else -distance


def window_speeds(series: dict[int, np.ndarray]) -> dict[int, float]:
    """Average speed per consecutive-sample window, keyed by end frame."""
    frames = sorted(series)
    return {
        b: float(np.linalg.norm(series[b] - series[a]) / ((b - a) / base.FPS))
        for a, b in zip(frames, frames[1:])
        if b - a == FRAME_IDS[1] - FRAME_IDS[0]
    }


# --- pure judgment -------------------------------------------------------

def banded_verdict(measured: float, threshold: float, band: float,
                   fail_when: str) -> str:
    """PASS/FAIL/NEEDS_REVIEW with the tier band as honesty buffer.

    fail_when='above': measured > threshold violates (zone depth, speed).
    fail_when='below': measured < threshold violates (min distance).
    band=0 reproduces the perfect-tracking binary verdict."""
    if fail_when == "below":
        measured, threshold = -measured, -threshold
    if measured > threshold + band:
        return FAIL
    if measured >= threshold - band:
        return NEEDS_REVIEW if band > 0 else PASS
    return PASS


def rule_agreement(gt_verdicts: list[str], est_verdicts: list[str]) -> dict:
    """How much does our metrology degrade the judgment vs perfect tracking?"""
    decided = [
        (g, e) for g, e in zip(gt_verdicts, est_verdicts)
        if e != NEEDS_REVIEW
    ]
    agree = sum(1 for g, e in decided if g == e)
    return {
        "n": len(gt_verdicts),
        "est_needs_review": sum(1 for e in est_verdicts if e == NEEDS_REVIEW),
        "decided": len(decided),
        "decided_agree": agree,
        "agreement_rate_decided": (
            round(agree / len(decided), 3) if decided else None
        ),
        "hard_disagreements": sum(1 for g, e in decided if g != e),
    }


def summarize_errors(errors: list[float]) -> dict:
    if not errors:
        return {"n": 0}
    values = np.asarray(errors)
    return {
        "n": len(values),
        "median_m": round(float(np.median(values)), 3),
        "p90_m": round(float(np.quantile(values, 0.9)), 3),
        "mean_m": round(float(np.mean(values)), 3),
    }


# --- lifting -------------------------------------------------------------

def lift_box_world(points: np.ndarray, valid: np.ndarray,
                   camera_to_world: np.ndarray, box: list[float],
                   scale: float, rotation: np.ndarray,
                   translation: np.ndarray) -> np.ndarray | None:
    """GT-box region -> robust representative point -> dataset world frame.

    MapAnything points (its own frame) -> its camera frame via the run's
    camera_to_world -> metres via the anchor scale -> dataset world via the
    calibration extrinsic inverse."""
    cloud = base.box_points(points, valid, box)
    if cloud is None:
        return None
    representative = np.median(cloud, axis=0)
    camera = (
        np.linalg.inv(camera_to_world) @ np.append(representative, 1.0)
    )[:3] * scale
    return world_from_camera(rotation, translation, camera)


def _load_frame_arrays(frame_dir: Path):
    return (
        np.load(frame_dir / "pts3d.npy"),
        np.load(frame_dir / "valid_mask.npy").astype(bool),
        np.load(frame_dir / "camera_to_world.npy"),
    )


def _camera_median_range(points, valid, camera_to_world) -> float | None:
    finite = valid & np.isfinite(points).all(axis=2)
    cloud = points[finite]
    if len(cloud) < 100:
        return None
    camera = (
        np.linalg.inv(camera_to_world) @ np.c_[cloud, np.ones(len(cloud))].T
    )[:3].T
    return float(np.median(np.linalg.norm(camera, axis=1)))


def _mono_moge_median(image: str) -> float | None:
    """Cached MoGe metric median range for a mono image (no new spend)."""
    moge_dir = base._geometry_dir(image) / "moge"
    if not moge_dir.is_dir():
        return None
    caches = sorted(moge_dir.glob("*.json"))
    if not caches:
        return None
    return json.loads(caches[0].read_text()).get("moge_median_range_m")


# --- multiview -----------------------------------------------------------

def _mv_dir(frame_index: int) -> Path:
    return WORK / "geometry_mv" / f"{frame_index:06d}"


def _run_multiview(frame_index: int, image_paths: list[Path]) -> None:
    target = _mv_dir(frame_index)
    if (target / "frames").is_dir():
        return
    from ehs_spatial.providers.base import ProviderError
    from ehs_spatial.providers.map_anything import MapAnythingAdapter

    for attempt in range(6):
        try:
            MapAnythingAdapter().run([str(p) for p in image_paths], target)
            (target / "cameras.json").write_text(
                json.dumps(list(CAMERAS)) + "\n"
            )
            return
        except ProviderError as exc:
            message = str(exc)
            if not any(
                code in message
                for code in ("429", "throttled", "502", "503", "500")
            ):
                raise
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"MapAnything multiview kept failing for {frame_index}")


# --- layers --------------------------------------------------------------

def score_layer1(questions: list[dict], scales: dict[str, float | None]) -> dict:
    """The mtmc_eval distance protocol, re-scored at the larger n."""
    rows = []
    for question in questions:
        frame_dir = next((base._geometry_dir(question["image"]) / "frames").iterdir())
        points = np.load(frame_dir / "pts3d.npy")
        valid = np.load(frame_dir / "valid_mask.npy").astype(bool)
        clouds = [base.box_points(points, valid, b) for b in question["boxes"]]
        row = dict(question)
        if all(cloud is not None for cloud in clouds):
            centroid = float(np.linalg.norm(
                np.median(clouds[0], axis=0) - np.median(clouds[1], axis=0)
            ))
            minimum = base.min_distance(clouds[0], clouds[1])
            scale = scales.get(question["image"])
            row["centroid_rel"] = round(
                abs(centroid - question["gt_centroid_m"]) / question["gt_centroid_m"], 3
            )
            if question["gt_min_m"] > 0:
                row["min_rel"] = round(
                    abs(minimum - question["gt_min_m"]) / question["gt_min_m"], 3
                )
            if scale is not None:
                row["moge_centroid_rel"] = round(
                    abs(centroid * scale - question["gt_centroid_m"])
                    / question["gt_centroid_m"], 3,
                )
                if question["gt_min_m"] > 0:
                    row["moge_min_rel"] = round(
                        abs(minimum * scale - question["gt_min_m"])
                        / question["gt_min_m"], 3,
                    )
        rows.append(row)

    per_camera = {}
    for camera in CAMERAS:
        errors = [
            r["moge_centroid_rel"] for r in rows
            if r["camera"] == camera and "moge_centroid_rel" in r
        ]
        per_camera[camera] = {
            "n": len(errors),
            "median_rel_err": round(float(np.median(errors)), 3) if errors else None,
        }
    return {
        "n_questions": len(rows),
        "unanchored_centroid": base.summarize(rows, "centroid_rel"),
        "unanchored_min": base.summarize(rows, "min_rel"),
        "moge_centroid": base.summarize(rows, "moge_centroid_rel"),
        "moge_min": base.summarize(rows, "moge_min_rel"),
        "moge_centroid_per_camera": per_camera,
        "baseline_n140": {
            "moge_centroid_median": 0.174,
            "moge_min_median": 0.253,
            "per_camera_moge_centroid_median": {
                "Camera": 0.104, "Camera_01": 0.100,
                "Camera_03": 0.183, "Camera_07": 1.079,
            },
        },
    }


def visible_objects(objects: list[dict], camera: str) -> list[dict]:
    result = []
    for entry in objects:
        box = entry["2d bounding box visible"].get(camera)
        if box is None:
            continue
        if (box[2] - box[0]) * (box[3] - box[1]) < base.MIN_BOX_AREA_PX:
            continue
        result.append(entry)
    return result


def build_estimates(
    ground_truth: dict,
    extrinsics: dict[str, tuple[np.ndarray, np.ndarray]],
    frame_ids: tuple[int, ...],
    geometry_dir_for,
    scale_for,
) -> dict[tuple[int, str, int], np.ndarray]:
    """(frame, camera, object) -> estimated world position."""
    estimates = {}
    for frame_index in frame_ids:
        objects = ground_truth.get(str(frame_index), [])
        for camera in CAMERAS:
            frames_dir = geometry_dir_for(frame_index, camera)
            if frames_dir is None or not frames_dir.is_dir():
                continue
            scale = scale_for(frame_index, camera)
            if scale is None:
                continue
            # Mono runs hold exactly one frame directory.
            points, valid, camera_to_world = _load_frame_arrays(
                next(frames_dir.iterdir())
            )
            rotation, translation = extrinsics[camera]
            for entry in visible_objects(objects, camera):
                world = lift_box_world(
                    points, valid, camera_to_world,
                    entry["2d bounding box visible"][camera],
                    scale, rotation, translation,
                )
                if world is not None:
                    estimates[(frame_index, camera, entry["object id"])] = world
    return estimates


def trajectory_errors(
    estimates: dict[tuple[int, str, int], np.ndarray],
    gt_positions: dict[tuple[int, int], np.ndarray],
) -> dict:
    """ATE-style summaries: XY (floor) error per camera, per object, pooled."""
    per_camera = defaultdict(list)
    per_object = defaultdict(list)
    errors_3d = []
    for (frame_index, camera, object_id), world in estimates.items():
        gt = gt_positions.get((frame_index, object_id))
        if gt is None:
            continue
        error_xy = float(np.linalg.norm(world[:2] - gt[:2]))
        per_camera[camera].append(error_xy)
        per_object[(camera, object_id)].append(error_xy)
        errors_3d.append(float(np.linalg.norm(world - gt)))
    pooled = [e for errors in per_camera.values() for e in errors]
    object_ates = sorted(
        round(float(np.median(v)), 3) for v in per_object.values()
    )
    return {
        "pooled_xy": summarize_errors(pooled),
        "pooled_3d": summarize_errors(errors_3d),
        "per_camera_xy": {
            camera: summarize_errors(per_camera.get(camera, []))
            for camera in CAMERAS
        },
        "per_object_ate_xy_median": (
            round(float(np.median(object_ates)), 3) if object_ates else None
        ),
        "n_object_series": len(object_ates),
    }


def fuse_estimates(
    estimates: dict[tuple[int, str, int], np.ndarray]
) -> dict[tuple[int, int], np.ndarray]:
    """One position per (frame, object): median across cameras that saw it."""
    grouped = defaultdict(list)
    for (frame_index, _, object_id), world in estimates.items():
        grouped[(frame_index, object_id)].append(world)
    return {
        key: np.median(np.stack(values), axis=0)
        for key, values in grouped.items()
    }


def judge_rules(
    gt_series: dict[int, dict[int, np.ndarray]],
    est_series: dict[int, dict[int, np.ndarray]],
    band: float,
) -> dict:
    """R1/R2/R3 from GT (band 0 = perfect tracking) and from estimates."""
    # R1 keep-clear zone: judged over the objects the estimate covers at
    # each timestamp, so the two verdicts see the same evidence set.
    r1_rows, r1_gt, r1_est = [], [], []
    for frame_index in FRAME_IDS:
        common = [
            object_id for object_id in est_series.get(frame_index, {})
            if object_id in gt_series.get(frame_index, {})
            and object_id not in ZONE_EXEMPT_IDS
        ]
        if not common:
            continue
        gt_depth = max(
            zone_depth(gt_series[frame_index][o][:2], ZONE) for o in common
        )
        est_depth = max(
            zone_depth(est_series[frame_index][o][:2], ZONE) for o in common
        )
        gt_verdict = banded_verdict(gt_depth, 0.0, 0.0, "above")
        est_verdict = banded_verdict(est_depth, 0.0, band, "above")
        r1_gt.append(gt_verdict)
        r1_est.append(est_verdict)
        r1_rows.append({
            "t_s": frame_index / base.FPS,
            "gt_depth_m": round(gt_depth, 3),
            "est_depth_m": round(est_depth, 3),
            "gt": gt_verdict, "est": est_verdict,
            "n_objects": len(common),
        })

    # R2 min distance over time for the chosen pair.
    subject, other = R2_PAIR
    r2_rows, r2_gt, r2_est = [], [], []
    for frame_index in FRAME_IDS:
        gt_frame = gt_series.get(frame_index, {})
        est_frame = est_series.get(frame_index, {})
        if not all(o in gt_frame and o in est_frame for o in R2_PAIR):
            continue
        gt_gap = float(np.linalg.norm(
            gt_frame[subject][:2] - gt_frame[other][:2]
        ))
        est_gap = float(np.linalg.norm(
            est_frame[subject][:2] - est_frame[other][:2]
        ))
        gt_verdict = banded_verdict(gt_gap, R2_THRESHOLD_M, 0.0, "below")
        est_verdict = banded_verdict(est_gap, R2_THRESHOLD_M, band, "below")
        r2_gt.append(gt_verdict)
        r2_est.append(est_verdict)
        r2_rows.append({
            "t_s": frame_index / base.FPS,
            "gt_gap_m": round(gt_gap, 3), "est_gap_m": round(est_gap, 3),
            "gt": gt_verdict, "est": est_verdict,
        })

    # R3 speed per 20 s window, every object with both series.
    r3_rows, r3_gt, r3_est = [], [], []
    gt_tracks, est_tracks = defaultdict(dict), defaultdict(dict)
    for frame_index, frame in gt_series.items():
        for object_id, position in frame.items():
            gt_tracks[object_id][frame_index] = position[:2]
    for frame_index, frame in est_series.items():
        for object_id, position in frame.items():
            est_tracks[object_id][frame_index] = position[:2]
    for object_id in sorted(gt_tracks):
        gt_speeds = window_speeds(gt_tracks[object_id])
        est_speeds = window_speeds(est_tracks.get(object_id, {}))
        for end_frame in sorted(gt_speeds):
            if end_frame not in est_speeds:
                continue
            gt_verdict = banded_verdict(
                gt_speeds[end_frame], R3_SPEED_LIMIT_MPS, 0.0, "above"
            )
            est_verdict = banded_verdict(
                est_speeds[end_frame], R3_SPEED_LIMIT_MPS,
                SPEED_BAND_MPS, "above",
            )
            r3_gt.append(gt_verdict)
            r3_est.append(est_verdict)
            r3_rows.append({
                "object_id": object_id,
                "t_s": end_frame / base.FPS,
                "gt_mps": round(gt_speeds[end_frame], 3),
                "est_mps": round(est_speeds[end_frame], 3),
                "gt": gt_verdict, "est": est_verdict,
            })

    return {
        "R1_keep_clear_zone": {
            "zone_world_xy": [list(c) for c in ZONE.exterior.coords][:-1],
            "exempt_object_ids": sorted(ZONE_EXEMPT_IDS),
            "band_m": band,
            "timeline": r1_rows,
            "agreement": rule_agreement(r1_gt, r1_est),
        },
        "R2_min_distance": {
            "pair": list(R2_PAIR),
            "threshold_m": R2_THRESHOLD_M,
            "band_m": band,
            "timeline": r2_rows,
            "agreement": rule_agreement(r2_gt, r2_est),
        },
        "R3_speed": {
            "limit_mps": R3_SPEED_LIMIT_MPS,
            "band_mps": round(SPEED_BAND_MPS, 4),
            "window_s": STEP_S,
            "timeline": r3_rows,
            "agreement": rule_agreement(r3_gt, r3_est),
        },
    }


def build_scorecard(meta: dict, layer1: dict, layer2: dict, layer3: dict,
                    spend: dict) -> dict:
    """Schema is load-bearing: tests pin these keys."""
    return {
        "meta": meta,
        "layer1_distance_rescore": layer1,
        "layer2_trajectories": layer2,
        "layer3_temporal_judgment": layer3,
        "spend": spend,
    }


# --- rendering -----------------------------------------------------------

def render_map(
    gt_series, est_series, layer3, map_path: Path, scale, tx, ty, out: Path
) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(map_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    map_h = image.size[1]

    def px(xy):
        return world_to_map(np.asarray(xy, dtype=float), scale, tx, ty, map_h)

    zone_px = [px(c) for c in ZONE.exterior.coords]
    draw.line(zone_px, fill="red", width=5)

    palette = [
        "orange", "yellow", "lime", "cyan", "magenta", "white",
        "deepskyblue", "springgreen", "gold", "violet", "salmon",
    ]
    object_ids = sorted({
        o for frame in gt_series.values() for o in frame
    })
    for color, object_id in zip(palette * 3, object_ids):
        gt_points = [
            px(gt_series[f][object_id][:2])
            for f in FRAME_IDS if object_id in gt_series.get(f, {})
        ]
        if len(gt_points) >= 2:
            draw.line(gt_points, fill=color, width=3)
        est_points = [
            px(est_series[f][object_id][:2])
            for f in FRAME_IDS if object_id in est_series.get(f, {})
        ]
        if len(est_points) >= 2:
            draw.line(est_points, fill=color, width=1)
        for point in est_points:
            draw.ellipse(
                [point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4],
                outline=color, width=2,
            )
    # Mark R1 violation timestamps beside the zone.
    anchor = px((ZONE.centroid.x, ZONE.centroid.y))
    violations = [
        row for row in layer3["R1_keep_clear_zone"]["timeline"]
        if FAIL in (row["gt"], row["est"])
    ]
    for index, row in enumerate(violations):
        label = f"t={row['t_s']:.0f}s gt:{row['gt'][0]} est:{row['est'][0]}"
        draw.text((anchor[0] - 55, anchor[1] - 80 + 14 * index), label, fill="red")
    draw.text(
        (20, 20),
        "GT thick / estimate thin+circles, per-object colour; "
        "red box = keep-clear zone (R1)",
        fill="white",
    )
    image.save(out)


# --- main ----------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--scene", default=base.SCENE)
    args = parser.parse_args(argv)

    ground_truth = json.loads(
        (DATA / args.scene / "ground_truth.json").read_text()
    )
    calibration = json.loads(
        (DATA / args.scene / "calibration.json").read_text()
    )
    sensors = {s["id"]: s for s in calibration["sensors"]}
    extrinsics = {c: extrinsic_parts(sensors[c]) for c in CAMERAS}
    map_scale = sensors[CAMERAS[0]]["scaleFactor"]
    map_tx = sensors[CAMERAS[0]]["translationToGlobalCoordinates"]["x"]
    map_ty = sensors[CAMERAS[0]]["translationToGlobalCoordinates"]["y"]

    questions = base.build_questions(ground_truth, CAMERAS, FRAME_IDS)
    images = sorted(
        f"{camera}_{frame_index:06d}"
        for camera in CAMERAS for frame_index in FRAME_IDS
    )
    uncached_geometry = [
        i for i in images if not (base._geometry_dir(i) / "frames").is_dir()
    ]
    uncached_moge = [i for i in images if not base._moge_cached(i)]
    uncached_mv = [
        f for f in MV_FRAME_IDS if not (_mv_dir(f) / "frames").is_dir()
    ]
    print(
        f"{len(questions)} distance questions | {len(images)} mono images | "
        f"{len(uncached_geometry)} uncached MapAnything mono + "
        f"{len(uncached_moge)} uncached MoGe + "
        f"{len(uncached_mv)} uncached MapAnything multiview calls"
    )
    if (uncached_geometry or uncached_moge or uncached_mv) and not args.live:
        print("pass --live to spend them (cached reruns are free)", file=sys.stderr)
        return 2

    failures: list[str] = []
    scales: dict[str, float | None] = {}
    for image in images:
        camera, frame_index = image.rsplit("_", 1)
        try:
            png = base._extract_frame(args.scene, camera, int(frame_index))
            base._run_geometry(image, png)
            scales[image] = base._moge_scale(image)
        except Exception as exc:  # one bad image must not kill the batch
            failures.append(f"{image}: {exc}")
            print(f"skipping {image}: {exc}", file=sys.stderr)

    for frame_index in MV_FRAME_IDS:
        try:
            paths = [
                base._extract_frame(args.scene, camera, frame_index)
                for camera in CAMERAS
            ]
            _run_multiview(frame_index, paths)
        except Exception as exc:
            failures.append(f"multiview_{frame_index}: {exc}")
            print(f"skipping multiview {frame_index}: {exc}", file=sys.stderr)

    # ---- Layer 1
    questions = [
        q for q in questions
        if (base._geometry_dir(q["image"]) / "frames").is_dir()
    ]
    layer1 = score_layer1(questions, scales)

    # ---- Layer 2
    gt_positions = {
        (frame_index, entry["object id"]): np.asarray(entry["3d location"])
        for frame_index in FRAME_IDS
        for entry in ground_truth.get(str(frame_index), [])
    }

    def mono_dir(frame_index, camera):
        return base._geometry_dir(f"{camera}_{frame_index:06d}") / "frames"

    mono_estimates = build_estimates(
        ground_truth, extrinsics, FRAME_IDS,
        geometry_dir_for=mono_dir,
        scale_for=lambda f, c: scales.get(f"{c}_{f:06d}"),
    )

    # Multiview: view i of the joint run corresponds to CAMERAS[i].
    mv_estimates_native: dict[tuple[int, str, int], np.ndarray] = {}
    mv_estimates_moge: dict[tuple[int, str, int], np.ndarray] = {}
    mv_scales: dict[int, dict] = {}
    for frame_index in MV_FRAME_IDS:
        frames_root = _mv_dir(frame_index) / "frames"
        if not frames_root.is_dir():
            continue
        frame_dirs = sorted(frames_root.iterdir())
        if len(frame_dirs) != len(CAMERAS):
            failures.append(
                f"multiview_{frame_index}: {len(frame_dirs)} views, expected "
                f"{len(CAMERAS)}"
            )
            continue
        # MoGe anchor for the joint gauge, from cached mono MoGe medians
        # (same source frames — zero extra spend).
        ratios = []
        arrays = {}
        for camera, frame_dir in zip(CAMERAS, frame_dirs):
            points, valid, camera_to_world = _load_frame_arrays(frame_dir)
            arrays[camera] = (points, valid, camera_to_world)
            native = _camera_median_range(points, valid, camera_to_world)
            moge = _mono_moge_median(f"{camera}_{frame_index:06d}")
            if native and moge:
                ratios.append(moge / native)
        moge_scale = float(np.median(ratios)) if ratios else None
        mv_scales[frame_index] = {
            "moge_scale": round(moge_scale, 4) if moge_scale else None,
            "n_ratios": len(ratios),
        }
        objects = ground_truth.get(str(frame_index), [])
        for camera in CAMERAS:
            points, valid, camera_to_world = arrays[camera]
            rotation, translation = extrinsics[camera]
            for entry in visible_objects(objects, camera):
                box = entry["2d bounding box visible"][camera]
                key = (frame_index, camera, entry["object id"])
                native_world = lift_box_world(
                    points, valid, camera_to_world, box, 1.0,
                    rotation, translation,
                )
                if native_world is not None:
                    mv_estimates_native[key] = native_world
                if moge_scale is not None:
                    moge_world = lift_box_world(
                        points, valid, camera_to_world, box, moge_scale,
                        rotation, translation,
                    )
                    if moge_world is not None:
                        mv_estimates_moge[key] = moge_world

    mono_at_mv = {
        key: value for key, value in mono_estimates.items()
        if key[0] in MV_FRAME_IDS
    }
    layer2 = {
        "protocol": (
            "position = median 3D point of the GT visible-box region, "
            "camera->world via calibration extrinsics; error vs GT object "
            "centre (surface-to-centre offset included, not corrected)"
        ),
        "mono": trajectory_errors(mono_estimates, gt_positions),
        "mono_at_mv_timestamps": trajectory_errors(mono_at_mv, gt_positions),
        "multiview_native_scale": trajectory_errors(
            mv_estimates_native, gt_positions
        ),
        "multiview_moge_anchored": trajectory_errors(
            mv_estimates_moge, gt_positions
        ),
        "multiview_scales": mv_scales,
        "coverage": {
            "mono_estimates": len(mono_estimates),
            "gt_slots_visible": sum(
                1 for frame_index in FRAME_IDS
                for camera in CAMERAS
                for _ in visible_objects(
                    ground_truth.get(str(frame_index), []), camera
                )
            ),
        },
    }

    # ---- Layer 3
    gt_series: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for (frame_index, object_id), position in gt_positions.items():
        gt_series[frame_index][object_id] = position
    est_series: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for (frame_index, object_id), position in fuse_estimates(mono_estimates).items():
        est_series[frame_index][object_id] = position
    layer3 = judge_rules(gt_series, est_series, ERROR_BUDGET_MONO_M)

    meta = {
        "scene": args.scene,
        "license": "CC-BY-4.0 (nvidia/PhysicalAI-SmartSpaces, ungated)",
        "imagery": "synthetic (Omniverse/Cosmos Transfer)",
        "identity_source": (
            "object identity and 2D regions come from GT: this validates "
            "metrology + judgment, not a tracker"
        ),
        "cameras": list(CAMERAS),
        "frame_indices": list(FRAME_IDS),
        "multiview_frame_indices": list(MV_FRAME_IDS),
        "error_band_mono_m": ERROR_BUDGET_MONO_M,
        "error_band_multiview_m": ERROR_BUDGET_MULTIVIEW_M,
        "image_failures": failures,
    }
    spend = {
        "mapanything_mono_calls_this_run": len(uncached_geometry),
        "mapanything_multiview_calls_this_run": len(uncached_mv),
        "moge_calls_this_run": len(uncached_moge),
        "sam_calls": 0,
        "gemini_calls": 0,
    }
    scorecard = build_scorecard(meta, layer1, layer2, layer3, spend)
    WORK.mkdir(parents=True, exist_ok=True)
    report_path = WORK / "trajectory_report.json"
    report_path.write_text(json.dumps(scorecard, indent=2) + "\n")
    overlay_path = WORK / "trajectory_map.png"
    render_map(
        gt_series, est_series, layer3,
        DATA / args.scene / "map.png", map_scale, map_tx, map_ty,
        overlay_path,
    )

    print("layer1 moge_centroid:", layer1["moge_centroid"])
    print("layer1 per-camera:", layer1["moge_centroid_per_camera"])
    print("layer2 mono pooled xy:", layer2["mono"]["pooled_xy"])
    print("layer2 mono@mv xy:", layer2["mono_at_mv_timestamps"]["pooled_xy"])
    print("layer2 mv native xy:", layer2["multiview_native_scale"]["pooled_xy"])
    print("layer2 mv moge xy:", layer2["multiview_moge_anchored"]["pooled_xy"])
    for name, rule in layer3.items():
        print(f"layer3 {name}:", rule["agreement"])
    print("report:", report_path)
    print("overlay:", overlay_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
