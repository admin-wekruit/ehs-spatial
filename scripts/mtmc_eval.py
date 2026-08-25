"""NVIDIA PhysicalAI-SmartSpaces MTMC — calibrated multi-camera distance eval.

Out-of-domain exam on a warehouse the harness has never seen: the
MTMC_Tracking_2025 val scenes ship per-frame object world coordinates,
oriented 3D boxes, and per-camera visible 2D boxes, so distance questions
need no detector, no phrase parsing, and ZERO SAM spend — the GT boxes are
the regions. We score MapAnything mono geometry, unanchored and MoGe-2
auto-anchored, against GT under BOTH distance protocols (centroid-to-centroid
and min surface distance): the boundary map showed protocol choice alone
moves the score, so each prediction is compared to its matching GT semantic.

License: CC-BY-4.0, ungated, no registration (verified via the HF API
2026-08-25: "gated": false, "license": "cc-by-4.0"). Honesty note: every
GT-labelled scene is SYNTHETIC (Omniverse/Cosmos Transfer); the only
real-world captures in the collection (2026 Warehouse_026/027) ship without
ground truth, so "real warehouse imagery with GT" does not exist here.

House pattern: files download with curl (this env's urllib lacks SSL certs),
every paid response is disk-cached under outputs/mtmc_v1 in the MAIN repo,
spend is gated behind --live with a printed count, per-image failures skip.
Frame extraction shells out to the imageio-ffmpeg bundled binary (no system
ffmpeg on this machine):

  uv run --env-file .env --with imageio-ffmpeg python scripts/mtmc_eval.py
  uv run --env-file .env --with imageio-ffmpeg python scripts/mtmc_eval.py --live
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/adam/Desktop/Tesla/ehs-spatial")
DATA = REPO / "outputs/datasets/mtmc/MTMC_Tracking_2025/val"
WORK = REPO / "outputs/mtmc_v1"
BASE = (
    "https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/"
    "resolve/main/MTMC_Tracking_2025/val"
)
SCENE = "Warehouse_016"
# The 4 cameras that see the most simultaneous objects, and 3 well-separated
# timestamps: 12 mono images, 12 MapAnything + 12 MoGe calls, cents total.
CAMERAS = ("Camera", "Camera_01", "Camera_03", "Camera_07")
FRAME_IDS = (0, 3600, 7200)
SOURCE_W, SOURCE_H = 1920, 1080
FPS = 30.0
MIN_BOX_AREA_PX = 2000
MAX_PAIRS_PER_IMAGE = 12
MIN_GT_DISTANCE_M = 0.5


def _fetch(scene: str, relative: str) -> Path:
    target = DATA / scene / relative
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-sSfL", "-o", str(target), f"{BASE}/{scene}/{relative}"],
        check=True,
    )
    return target


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError:
        raise SystemExit(
            "frame extraction needs imageio-ffmpeg: rerun via "
            "`uv run --with imageio-ffmpeg ...`"
        )
    return imageio_ffmpeg.get_ffmpeg_exe()


def _extract_frame(scene: str, camera: str, frame_index: int) -> Path:
    target = DATA / scene / "frames_png" / f"{camera}_{frame_index:06d}.png"
    if target.exists():
        return target
    video = _fetch(scene, f"videos/{camera}.mp4")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Input seek is frame-accurate in modern ffmpeg (decodes forward from the
    # preceding keyframe to the requested PTS).
    subprocess.run(
        [
            _ffmpeg_exe(), "-y", "-loglevel", "error",
            "-ss", f"{frame_index / FPS:.6f}",
            "-i", str(video),
            "-frames:v", "1",
            str(target),
        ],
        check=True,
    )
    return target


def _rotation_matrix(pitch: float, roll: float, yaw: float) -> np.ndarray:
    # Yaw about +Z dominates in this dataset (pitch/roll are ~0 for every
    # object class); the exact pitch/roll composition order is therefore
    # immaterial at the reported precision.
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def box_surface_points(
    location: list[float],
    scale: list[float],
    rotation: list[float],
    grid: int = 8,
) -> np.ndarray:
    """Sample the surface of an oriented 3D box (GT min-distance protocol).

    Grid sampling quantises the answer by ~scale/(grid-1) when the true
    closest points fall between samples; at grid=8 that is <=9 cm for a
    person-sized box, well inside the model error being measured."""
    half = np.asarray(scale, dtype=float) / 2.0
    lin = np.linspace(-1.0, 1.0, grid)
    faces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            u, v = np.meshgrid(lin, lin)
            face = np.zeros((grid * grid, 3))
            others = [i for i in range(3) if i != axis]
            face[:, axis] = sign
            face[:, others[0]] = u.ravel()
            face[:, others[1]] = v.ravel()
            faces.append(face)
    unit = np.vstack(faces) * half
    rot = _rotation_matrix(*rotation)
    return unit @ rot.T + np.asarray(location, dtype=float)


def gt_min_distance(a: dict, b: dict) -> float:
    pa = box_surface_points(
        a["3d location"], a["3d bounding box scale"], a["3d bounding box rotation"]
    )
    pb = box_surface_points(
        b["3d location"], b["3d bounding box scale"], b["3d bounding box rotation"]
    )
    return float(
        np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)).min()
    )


def build_questions(
    ground_truth: dict, cameras: tuple[str, ...], frame_ids: tuple[int, ...]
) -> list[dict]:
    """Pair-distance questions from the dataset's own world-coordinate GT."""
    questions = []
    for frame_index in frame_ids:
        objects = ground_truth.get(str(frame_index), [])
        for camera in cameras:
            visible = []
            for entry in objects:
                box = entry["2d bounding box visible"].get(camera)
                if box is None:
                    continue
                if (box[2] - box[0]) * (box[3] - box[1]) < MIN_BOX_AREA_PX:
                    continue
                visible.append(entry)
            visible.sort(key=lambda e: e["object id"])
            pairs = []
            for a, b in itertools.combinations(visible, 2):
                gt_centroid = float(
                    np.linalg.norm(
                        np.asarray(a["3d location"]) - np.asarray(b["3d location"])
                    )
                )
                if gt_centroid < MIN_GT_DISTANCE_M:
                    continue
                pairs.append(
                    {
                        "image": f"{camera}_{frame_index:06d}",
                        "camera": camera,
                        "frame_index": frame_index,
                        "ids": [a["object id"], b["object id"]],
                        "types": [a["object type"], b["object type"]],
                        "boxes": [
                            a["2d bounding box visible"][camera],
                            b["2d bounding box visible"][camera],
                        ],
                        "gt_centroid_m": round(gt_centroid, 3),
                        "gt_min_m": round(gt_min_distance(a, b), 3),
                    }
                )
            questions.extend(pairs[:MAX_PAIRS_PER_IMAGE])
    return questions


def _geometry_dir(image: str) -> Path:
    return WORK / "geometry" / image


def _run_geometry(image: str, image_path: Path) -> None:
    target = _geometry_dir(image)
    if (target / "frames").is_dir():
        return
    from ehs_spatial.providers.base import ProviderError
    from ehs_spatial.providers.map_anything import MapAnythingAdapter

    for attempt in range(6):
        try:
            MapAnythingAdapter().run([str(image_path)], target)
            return
        except ProviderError as exc:
            message = str(exc)
            transient = any(
                code in message
                for code in ("429", "throttled", "502", "503", "500")
            )
            if not transient:
                raise
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"MapAnything kept failing transiently for {image}")


def _disk_frame(image: str):
    from ehs_spatial.contracts import GeometryFrame

    frame_dir = next((_geometry_dir(image) / "frames").iterdir())
    return GeometryFrame(
        frame_id=frame_dir.name,
        canonical_image_path=str(frame_dir / "canonical.png"),
        pts3d_path=str(frame_dir / "pts3d.npy"),
        conf_path=str(frame_dir / "conf.npy"),
        valid_mask_path=str(frame_dir / "valid_mask.npy"),
        camera_to_world=np.load(frame_dir / "camera_to_world.npy").tolist(),
        intrinsics=np.load(frame_dir / "intrinsics.npy").tolist(),
    )


def _moge_cached(image: str) -> bool:
    geometry = _geometry_dir(image) / "frames"
    if not geometry.is_dir():
        return False
    frame_dir = next(geometry.iterdir())
    return (_geometry_dir(image) / "moge" / f"{frame_dir.name}.json").exists()


def _moge_scale(image: str) -> float | None:
    from ehs_spatial.providers.moge import MoGeAnchorAdapter

    anchor = MoGeAnchorAdapter().anchor_scale(
        [_disk_frame(image)], _geometry_dir(image)
    )
    return None if anchor is None else anchor.scale


def box_points(
    points: np.ndarray, valid: np.ndarray, box: list[float]
) -> np.ndarray | None:
    """Robust 3D points inside a (source-resolution) 2D box.

    Boxes are regions, not masks, so background pixels ride along; the range
    MAD trim keeps the dominant surface, exactly as nvwarehouse_eval does
    for its RLE masks. Selection noise is part of the measured protocol.
    """
    height, width = valid.shape
    sx, sy = width / SOURCE_W, height / SOURCE_H
    x1, y1, x2, y2 = box
    region = np.zeros_like(valid, dtype=bool)
    region[
        max(0, int(y1 * sy)) : min(height, int(np.ceil(y2 * sy))),
        max(0, int(x1 * sx)) : min(width, int(np.ceil(x2 * sx))),
    ] = True
    selected = region & valid & np.isfinite(points).all(axis=2)
    cloud = points[selected]
    if len(cloud) < 25:
        return None
    cloud = cloud[np.linalg.norm(cloud, axis=1) > 0.15]
    if len(cloud) < 25:
        return None
    ranges = np.linalg.norm(cloud, axis=1)
    median = np.median(ranges)
    mad = np.median(np.abs(ranges - median)) + 1e-6
    cloud = cloud[np.abs(ranges - median) < 3.0 * mad]
    return cloud if len(cloud) >= 25 else None


def min_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a[:: max(1, len(a) // 3000)]
    b = b[:: max(1, len(b) // 3000)]
    nn = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(np.percentile(nn, 0.5))


def _projection_sanity(
    calibration: dict, questions: list[dict], ground_truth: dict
) -> dict:
    """Median px gap between projected GT centers and visible-box centers.

    A loose sync/calibration check only: visible boxes are occlusion-cropped,
    so even perfect calibration leaves a residual."""
    matrices = {
        sensor["id"]: np.asarray(sensor["cameraMatrix"], dtype=float)
        for sensor in calibration["sensors"]
    }
    errors = []
    for question in questions:
        matrix = matrices.get(question["camera"])
        if matrix is None:
            continue
        objects = {
            entry["object id"]: entry
            for entry in ground_truth[str(question["frame_index"])]
        }
        for object_id, box in zip(question["ids"], question["boxes"]):
            location = np.asarray(objects[object_id]["3d location"] + [1.0])
            projected = matrix @ location
            if abs(projected[2]) < 1e-9:
                continue
            u, v = projected[0] / projected[2], projected[1] / projected[2]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            errors.append(float(np.hypot(u - cx, v - cy)))
    return {
        "n": len(errors),
        "median_px": round(float(np.median(errors)), 1) if errors else None,
    }


def summarize(rows: list[dict], key: str) -> dict:
    errors = [row[key] for row in rows if key in row]
    if not errors:
        return {"answered": 0}
    return {
        "answered": len(errors),
        "of": len(rows),
        "median_rel_err": round(float(np.median(errors)), 3),
        "success_at_10pct": round(
            sum(1 for e in errors if e <= 0.10) / len(errors), 3
        ),
        "success_at_25pct": round(
            sum(1 for e in errors if e <= 0.25) / len(errors), 3
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="max images")
    parser.add_argument("--scene", default=SCENE)
    args = parser.parse_args(argv)

    _fetch(args.scene, "calibration.json")
    _fetch(args.scene, "ground_truth.json")
    ground_truth = json.loads(
        (DATA / args.scene / "ground_truth.json").read_text()
    )
    calibration = json.loads(
        (DATA / args.scene / "calibration.json").read_text()
    )
    questions = build_questions(ground_truth, CAMERAS, FRAME_IDS)
    images = sorted({q["image"] for q in questions})[: args.limit]
    questions = [q for q in questions if q["image"] in set(images)]

    uncached_geometry = [
        image for image in images if not (_geometry_dir(image) / "frames").is_dir()
    ]
    uncached_moge = [image for image in images if not _moge_cached(image)]
    print(
        f"{len(questions)} distance questions | {len(images)} images | "
        f"{len(uncached_geometry)} uncached MapAnything + "
        f"{len(uncached_moge)} uncached MoGe calls"
    )
    if (uncached_geometry or uncached_moge) and not args.live:
        print("pass --live to spend them (cached reruns are free)", file=sys.stderr)
        return 2

    failures = []
    scales: dict[str, float | None] = {}
    for image in images:
        camera, frame_index = image.rsplit("_", 1)
        try:
            png = _extract_frame(args.scene, camera, int(frame_index))
            _run_geometry(image, png)
            scales[image] = _moge_scale(image)
        except Exception as exc:  # one bad image must not kill the batch
            failures.append(f"{image}: {exc}")
            print(f"skipping {image}: {exc}", file=sys.stderr)
    questions = [
        q for q in questions if (_geometry_dir(q["image"]) / "frames").is_dir()
    ]

    rows = []
    for question in questions:
        frame_dir = next((_geometry_dir(question["image"]) / "frames").iterdir())
        points = np.load(frame_dir / "pts3d.npy")
        valid = np.load(frame_dir / "valid_mask.npy").astype(bool)
        clouds = [box_points(points, valid, box) for box in question["boxes"]]
        row = dict(question)
        if all(cloud is not None for cloud in clouds):
            centroid = float(
                np.linalg.norm(
                    np.median(clouds[0], axis=0) - np.median(clouds[1], axis=0)
                )
            )
            minimum = min_distance(clouds[0], clouds[1])
            scale = scales.get(question["image"])
            row["centroid_m"] = round(centroid, 3)
            row["min_m"] = round(minimum, 3)
            row["centroid_rel"] = round(
                abs(centroid - question["gt_centroid_m"])
                / question["gt_centroid_m"],
                3,
            )
            if question["gt_min_m"] > 0:
                row["min_rel"] = round(
                    abs(minimum - question["gt_min_m"]) / question["gt_min_m"], 3
                )
            if scale is not None:
                row["moge_scale"] = round(scale, 4)
                row["moge_centroid_m"] = round(centroid * scale, 3)
                row["moge_min_m"] = round(minimum * scale, 3)
                row["moge_centroid_rel"] = round(
                    abs(centroid * scale - question["gt_centroid_m"])
                    / question["gt_centroid_m"],
                    3,
                )
                if question["gt_min_m"] > 0:
                    row["moge_min_rel"] = round(
                        abs(minimum * scale - question["gt_min_m"])
                        / question["gt_min_m"],
                        3,
                    )
        rows.append(row)

    summary = {
        "scene": args.scene,
        "license": "CC-BY-4.0 (nvidia/PhysicalAI-SmartSpaces, ungated)",
        "imagery": "synthetic (Omniverse/Cosmos Transfer)",
        "cameras": CAMERAS,
        "frame_indices": FRAME_IDS,
        "rows": rows,
        "unanchored_centroid": summarize(rows, "centroid_rel"),
        "unanchored_min": summarize(rows, "min_rel"),
        "moge_centroid": summarize(rows, "moge_centroid_rel"),
        "moge_min": summarize(rows, "moge_min_rel"),
        "moge_scales": {
            image: round(scale, 4)
            for image, scale in scales.items()
            if scale is not None
        },
        "gt_projection_check": _projection_sanity(
            calibration, questions, ground_truth
        ),
        "scale_source": "MapAnything mono; MoGe-2 auto anchor per image",
        "image_failures": failures,
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name in (
        "unanchored_centroid",
        "unanchored_min",
        "moge_centroid",
        "moge_min",
    ):
        print(f"{name}: {summary[name]}")
    print("gt_projection_check:", summary["gt_projection_check"])
    print("report:", WORK / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
