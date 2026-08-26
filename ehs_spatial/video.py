"""Uncalibrated video tier: states + trajectories + banded verdicts.

Productization of scripts/video_poc.py for footage WITHOUT a camera
calibration file: sample frames, SAM masks per frame (disk-cached under the
run dir), ByteTrack identity, then a 3D lift that fits the floor plane from
MoGe-2 metric mono depth on a few keyframes and ray-casts mask bottom-centers
to that plane. The three banded temporal rules (keep-clear zone, person-to-
vehicle min distance, person speed) run on the lifted tracks with the mono
band (±0.35 m).

Accuracy tiers — be honest about which one this is:

- **video-mono (this module, uncalibrated)**: floor plane and pinhole
  intrinsics are recovered from MoGe-2's own metric point map, not measured.
  Position error is bounded below by the mono band and grows with range and
  floor-fit error; feet-on-ground is a load-bearing assumption.
- **calibrated fixed camera (the measurement tier)**: the same pipeline
  through a real KRTD model measured 0.45 m median person ATE on MEVA
  (docs/demos/2026-08-25-poc-video.md). An installed camera with a real
  calibration is what turns states into measurements.
"""

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, Point, Polygon

from .artifacts import ArtifactStore
from .contracts import ProviderManifest, RunManifest
from .providers.base import ProviderError
from .providers.moge import MOGE_VERSION
from .providers.sam3 import SAM3_ENDPOINT, decode_coco_rle
from .rules import ERROR_BUDGET_MONO_M

COST_PER_SAM_CALL_USD = 0.01
MAX_LIVE_CALLS = 150  # hard budget cap per run, same as the POC
MIN_BOX_HEIGHT_PX = 12
MOGE_KEYFRAMES = 3
MIN_FLOOR_INLIER_FRACTION = 0.25
BAND_M = ERROR_BUDGET_MONO_M  # single-camera tier: ±0.35 m
R2_MIN_SEPARATION_M = 2.0  # person-to-vehicle keep-apart
R3_MAX_SPEED_MPS = 1.5  # walking-pace limit
PERSON_LABEL = "person"

PASS, FAIL, REVIEW, NO_DATA = "PASS", "FAIL", "NEEDS_REVIEW", "NO_DATA"
_SEVERITY = {FAIL: 3, REVIEW: 2, PASS: 1, NO_DATA: 0}
RULE_NAMES = ("R1_zone", "R2_min_distance", "R3_speed")

CALIBRATED_REFERENCE = (
    "calibrated fixed-camera tier reference: 0.45 m median person ATE on "
    "MEVA with a real KRTD model (docs/demos/2026-08-25-poc-video.md)"
)
TIER_NOTE = (
    "uncalibrated video-mono tier: floor plane and intrinsics are recovered "
    "from MoGe-2 metric mono depth, not measured; positions carry at least "
    f"the ±{BAND_M} m mono band and assume feet on the floor"
)


@dataclass(frozen=True)
class VideoRunPaths:
    root: Path
    frames_dir: Path
    sam_cache_dir: Path
    moge_dir: Path
    overlay_gif: Path
    overlay_strip_png: Path
    topdown_png: Path
    report_json: Path
    manifest_json: Path


def video_paths(store: ArtifactStore, run_id: str) -> VideoRunPaths:
    root = store.paths(run_id).root
    return VideoRunPaths(
        root=root,
        frames_dir=root / "frames",
        sam_cache_dir=root / "sam_cache",
        moge_dir=root / "moge",
        overlay_gif=root / "overlay.gif",
        overlay_strip_png=root / "overlay_strip.png",
        topdown_png=root / "trajectories_topdown.png",
        report_json=root / "video_report.json",
        manifest_json=root / "manifest.json",
    )


# ------------------------------------------------------------------ sampling
def sample_schedule(
    total_frames: int,
    native_fps: float,
    sample_fps: float,
    max_frames: int,
    start_frame: int = 0,
) -> list[int]:
    """Native frame indices to sample: every native_fps/sample_fps frames
    from start_frame, capped at max_frames (so long clips cover the first N
    samples of the chosen segment)."""
    if total_frames <= 0 or native_fps <= 0 or sample_fps <= 0 or max_frames <= 0:
        return []
    step = max(1, round(native_fps / sample_fps))
    return list(range(max(0, start_frame), total_frames, step))[:max_frames]


def _cv2_extract(
    video_path: Path,
    frames_dir: Path,
    sample_fps: float,
    max_frames: int,
    start_s: float = 0.0,
) -> tuple[list[int], float]:
    """Default extractor: OpenCV sequential decode (AVI seeking is
    unreliable), JPEGs at f{frame:06d}.jpg. Returns (frame_ids, native_fps)."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = sample_schedule(
        total, native_fps, sample_fps, max_frames, round(start_s * native_fps)
    )
    wanted = {
        f for f in frame_ids if not (frames_dir / f"f{f:06d}.jpg").is_file()
    }
    frames_dir.mkdir(parents=True, exist_ok=True)
    index = 0
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
    return frame_ids, float(native_fps)


# ----------------------------------------------------------------- SAM stage
def _cache_path(cache_dir: Path, frame_id: int, label: str) -> Path:
    return cache_dir / f"f{frame_id:06d}__{label.replace(' ', '_')}.json"


def _default_sam_subscriber(endpoint: str, *, arguments: dict) -> dict:
    import fal_client

    return fal_client.subscribe(endpoint, arguments=arguments)


def _subscribe_with_backoff(subscriber, prompt: str, image_uri: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return subscriber(
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


def sam_stage(
    frames_dir: Path,
    cache_dir: Path,
    frame_ids: list[int],
    labels: tuple[str, ...],
    subscriber: Callable[..., dict],
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[int, list[str]]:
    """Fill the per-frame SAM cache (one prompt per label). Returns
    (live_calls_made, failure keys). Rerunning a cached run costs $0."""
    import base64

    cache_dir.mkdir(parents=True, exist_ok=True)
    planned = [
        (frame_id, label)
        for frame_id in frame_ids
        for label in labels
        if not _cache_path(cache_dir, frame_id, label).is_file()
    ]
    if len(planned) > MAX_LIVE_CALLS:
        raise ValueError(
            f"budget cap: {len(planned)} planned SAM calls exceed "
            f"{MAX_LIVE_CALLS}; reduce sample rate, frames, or labels"
        )
    failures: list[str] = []
    calls_made = 0
    for ordinal, (frame_id, label) in enumerate(planned):
        if progress_cb is not None:
            progress_cb(f"SAM {ordinal + 1}/{len(planned)} (frame {frame_id})")
        frame_path = frames_dir / f"f{frame_id:06d}.jpg"
        uri = (
            "data:image/jpeg;base64,"
            + base64.b64encode(frame_path.read_bytes()).decode()
        )
        try:
            response = _subscribe_with_backoff(subscriber, label, uri)
        except Exception as exc:
            failures.append(f"{frame_id}:{label}: {exc}")
            continue
        calls_made += 1
        with Image.open(frame_path) as image:
            width, height = image.size
        _cache_path(cache_dir, frame_id, label).write_text(
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
    cache_dir: Path, frame_ids: list[int], labels: tuple[str, ...]
) -> dict[int, dict[str, list[dict]]]:
    """frame -> label -> [{bbox, score}] from the cached SAM responses."""
    detections: dict[int, dict[str, list[dict]]] = {}
    for frame_id in frame_ids:
        per_label: dict[str, list[dict]] = {}
        for label in labels:
            rows: list[dict] = []
            path = _cache_path(cache_dir, frame_id, label)
            if path.is_file():
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
                        }
                    )
            per_label[label] = rows
        detections[frame_id] = per_label
    return detections


# ----------------------------------------------------------------- tracking
def bytetrack(
    detections: dict[int, dict[str, list[dict]]],
    frame_ids: list[int],
    native_fps: float,
    labels: tuple[str, ...],
) -> dict[str, list[tuple[int, tuple[float, float, float, float]]]]:
    """ByteTrack per label over mask bboxes -> "label-N" -> [(frame, xyxy)].
    Settings proven on the MEVA POC (lost_track_buffer is counted in 30 fps
    frame units by the library, so 150 = a 5 s keepalive)."""
    import supervision as sv
    from trackers import ByteTrackTracker

    tracks: dict[str, list] = {}
    for label in labels:
        tracker = ByteTrackTracker(
            lost_track_buffer=150,
            frame_rate=1.0,
            minimum_consecutive_frames=2,
            track_activation_threshold=0.0,
            high_conf_det_threshold=0.0,
        )
        for frame_id in frame_ids:
            rows = detections.get(frame_id, {}).get(label, [])
            if rows:
                current = sv.Detections(
                    xyxy=np.array([row["bbox"] for row in rows], dtype=float),
                    confidence=np.array([row["score"] for row in rows]),
                )
            else:
                current = sv.Detections.empty()
            tracked = tracker.update(current, timestamp=frame_id / native_fps)
            for bbox, track_id in zip(
                tracked.xyxy, tracked.tracker_id, strict=True
            ):
                if int(track_id) < 0:
                    continue  # immature first-frame detection, no identity yet
                tracks.setdefault(f"{label}-{int(track_id)}", []).append(
                    (frame_id, tuple(float(value) for value in bbox))
                )
    return tracks


# --------------------------------------------------------- MoGe floor model
def fit_pinhole_from_grid(grid: np.ndarray) -> dict | None:
    """Least-squares pinhole intrinsics from an organized point map.

    grid is (H, W, 3) camera-frame points; u = fx*(x/z) + cx per column and
    v = fy*(y/z) + cy per row. Everything downstream (plane, rays) lives in
    the same frame as the grid, so axis conventions cancel."""
    height, width = grid.shape[:2]
    z = grid[:, :, 2]
    valid = np.isfinite(grid).all(axis=2) & (np.abs(z) > 1e-6)
    if valid.sum() < 100:
        return None
    vv, uu = np.nonzero(valid)
    x_over_z = grid[vv, uu, 0] / grid[vv, uu, 2]
    y_over_z = grid[vv, uu, 1] / grid[vv, uu, 2]
    if np.ptp(x_over_z) < 1e-6 or np.ptp(y_over_z) < 1e-6:
        return None
    fx, cx = np.polyfit(x_over_z, uu.astype(float), 1)
    fy, cy = np.polyfit(y_over_z, vv.astype(float), 1)
    if not all(np.isfinite(v) for v in (fx, fy, cx, cy)):
        return None
    if abs(fx) < 1e-3 or abs(fy) < 1e-3:
        return None
    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


def ransac_plane(
    points: np.ndarray,
    iterations: int = 200,
    threshold_m: float = 0.08,
    seed: int = 7,
) -> tuple[np.ndarray, float, float] | None:
    """RANSAC plane n·X + d = 0 (|n| = 1, d > 0 so the camera at the origin
    is on the positive side). Returns (normal, d, inlier_fraction)."""
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 50:
        return None
    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        distances = np.abs((points - sample[0]) @ normal)
        inliers = distances < threshold_m
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < 50:
        return None
    # Refine on the inlier set via SVD.
    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vt[2]
    d = -float(normal @ centroid)
    if d < 0:  # orient so the camera (origin) sits on the positive side
        normal, d = -normal, -d
    return normal, d, float(best_inliers.mean())


def _organize_cloud(
    points: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray | None:
    """Reshape a per-pixel point cloud back into its (H, W, 3) grid by
    factoring N into a grid with the source image's aspect ratio. Fails
    (None) when the cloud dropped invalid pixels and is no longer a grid."""
    count = len(points)
    if count < 100:
        return None
    aspect = image_size[0] / image_size[1]
    for height in range(int(np.sqrt(count / aspect) * 0.8) or 1, int(np.sqrt(count / aspect) * 1.25) + 2):
        if height <= 0 or count % height:
            continue
        width = count // height
        if abs(width / height - aspect) < 0.02:
            return points.reshape(height, width, 3)
    return None


class FloorCamera:
    """Floor plane + pinhole model recovered from one MoGe keyframe, with
    intrinsics rescaled to full-frame pixels."""

    def __init__(
        self,
        normal: np.ndarray,
        d: float,
        intrinsics: dict,
        inlier_fraction: float,
    ) -> None:
        self.normal = np.asarray(normal, dtype=float)
        self.d = float(d)
        self.intrinsics = intrinsics
        self.inlier_fraction = inlier_fraction
        # Plane frame for top-down coordinates: origin at the camera's foot
        # point, forward = optical axis projected onto the plane.
        self.origin = -self.d * self.normal
        z_axis = np.array([0.0, 0.0, 1.0])
        forward = z_axis - (self.normal @ z_axis) * self.normal
        norm = np.linalg.norm(forward)
        if norm < 1e-6:  # camera looking straight down: use image-down instead
            fallback = np.array([0.0, 1.0, 0.0])
            forward = fallback - (self.normal @ fallback) * self.normal
            norm = np.linalg.norm(forward)
        self.forward = forward / norm
        self.lateral = np.cross(self.normal, self.forward)

    @property
    def camera_height_m(self) -> float:
        return self.d

    def lift_pixel(self, u: float, v: float) -> tuple[float, float] | None:
        """Full-frame pixel -> (x, y) metres on the floor plane, or None
        when the ray misses the floor (above the horizon)."""
        k = self.intrinsics
        direction = np.array(
            [(u - k["cx"]) / k["fx"], (v - k["cy"]) / k["fy"], 1.0]
        )
        denominator = self.normal @ direction
        if abs(denominator) < 1e-9:
            return None
        t = -self.d / denominator
        if t <= 0:
            return None
        hit = t * direction
        offset = hit - self.origin
        return float(offset @ self.lateral), float(offset @ self.forward)


def _default_moge_runner(model_identifier: str, *, input: dict) -> object:
    # Shares the photo chain's backend switch (MoGe-3 on Modal by default,
    # MOGE_BACKEND=replicate for the pinned MoGe-2).
    from .providers.moge import _default_runner

    return _default_runner(model_identifier, input=input)


def _moge_cloud(image_path: Path, cache_ply: Path, runner) -> np.ndarray | None:
    """MoGe-2 metric point cloud for one keyframe, cached as a .ply beside
    the run. Any failure returns None — callers abstain, never fabricate."""
    import open3d as o3d

    if not cache_ply.is_file():
        import base64

        payload = "data:image/jpeg;base64," + base64.b64encode(
            image_path.read_bytes()
        ).decode("ascii")
        try:
            output = runner(MOGE_VERSION, input={"image": payload, "fp16": True})
            cloud_source = output["pointcloud_ply"]
            data = (
                cloud_source.read()
                if hasattr(cloud_source, "read")
                else Path(str(cloud_source)).read_bytes()
            )
        except Exception:
            return None
        cache_ply.parent.mkdir(parents=True, exist_ok=True)
        cache_ply.write_bytes(data)
        try:
            # MoGe's own normalized 3x3 intrinsics — required downstream when
            # the cloud dropped invalid pixels and is no longer a full grid.
            intr_source = output["intrinsics_json"]
            _intrinsics_path(cache_ply).write_bytes(
                intr_source.read()
                if hasattr(intr_source, "read")
                else Path(str(intr_source)).read_bytes()
            )
        except Exception:
            pass
    try:
        points = np.asarray(o3d.io.read_point_cloud(str(cache_ply)).points)
    except Exception:
        return None
    if not len(points):
        return None
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) and np.median(finite[:, 2]) < 0:
        # MoGe exports its cloud for GL viewers (y up, z back); flip to
        # OpenCV convention (x right, y down, z forward) — same auto-detect
        # as scripts/arm_poc.py cloud_to_depth.
        points = points * np.array([1.0, -1.0, -1.0])
    return points


def _intrinsics_path(cache_ply: Path) -> Path:
    return cache_ply.with_suffix(".intrinsics.json")


def _moge_pixel_intrinsics(
    cache_ply: Path, image_size: tuple[int, int]
) -> dict | None:
    """MoGe's cached normalized intrinsics, scaled to full-frame pixels."""
    path = _intrinsics_path(cache_ply)
    if not path.is_file():
        return None
    try:
        matrix = json.loads(path.read_text())["intrinsics"]
    except Exception:
        return None
    width, height = image_size
    return {
        "fx": float(matrix[0][0]) * width,
        "cx": float(matrix[0][2]) * width,
        "fy": float(matrix[1][1]) * height,
        "cy": float(matrix[1][2]) * height,
    }


def _lower_image_points(
    points: np.ndarray, intrinsics: dict, image_size: tuple[int, int]
) -> np.ndarray:
    """Points that project into the lower part of the image (where floors
    live), for clouds that lost their pixel-grid ordering."""
    finite = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    points = points[finite]
    v = intrinsics["fy"] * points[:, 1] / points[:, 2] + intrinsics["cy"]
    return points[v > image_size[1] * 0.55]


def fit_floor_from_keyframes(
    frames_dir: Path,
    moge_dir: Path,
    keyframe_ids: list[int],
    runner,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[FloorCamera | None, dict]:
    """Fit a FloorCamera per keyframe and keep the best-inlier fit as the
    camera model (the camera is fixed, so one good keyframe suffices); the
    per-keyframe height spread is reported as a stability signal."""
    keyframes: list[dict] = []
    best: FloorCamera | None = None
    for frame_id in keyframe_ids:
        if progress_cb is not None:
            progress_cb(f"MoGe floor fit (frame {frame_id})")
        entry: dict = {"frame": frame_id}
        keyframes.append(entry)
        frame_path = frames_dir / f"f{frame_id:06d}.jpg"
        with Image.open(frame_path) as image:
            image_size = image.size
        points = _moge_cloud(
            frame_path, moge_dir / f"f{frame_id:06d}.ply", runner
        )
        if points is None:
            entry["error"] = "MoGe call or point cloud read failed"
            continue
        grid = _organize_cloud(points, image_size)
        if grid is not None:
            intrinsics = fit_pinhole_from_grid(grid)
            if intrinsics is None:
                entry["error"] = "pinhole fit failed"
                continue
            grid_height, grid_width = grid.shape[:2]
            # Rescale grid-pixel intrinsics to full-frame pixels.
            scale = grid_width / image_size[0]
            intrinsics = {
                "fx": intrinsics["fx"] / scale,
                "cx": intrinsics["cx"] / scale,
                "fy": intrinsics["fy"] / (grid_height / image_size[1]),
                "cy": intrinsics["cy"] / (grid_height / image_size[1]),
            }
            # Floor candidates: lower part of the image, where floors live.
            candidates = grid[int(grid_height * 0.55) :].reshape(-1, 3)
        else:
            # Real MoGe clouds drop invalid pixels (sky, no-depth) and are
            # rarely full grids: fall back to the model's own intrinsics and
            # project points to select the lower-image floor candidates.
            intrinsics = _moge_pixel_intrinsics(
                moge_dir / f"f{frame_id:06d}.ply", image_size
            )
            if intrinsics is None:
                entry["error"] = (
                    "point cloud is not an organized pixel grid and no "
                    "MoGe intrinsics are cached"
                )
                continue
            candidates = _lower_image_points(points, intrinsics, image_size)
        plane = ransac_plane(candidates)
        if plane is None:
            entry["error"] = "no dominant plane in the lower image"
            continue
        normal, d, inlier_fraction = plane
        if abs(normal[1]) < 0.6:
            entry["error"] = (
                "dominant plane is not floor-like "
                f"(|n_y| = {abs(normal[1]):.2f} < 0.6)"
            )
            continue
        entry.update(
            {
                "camera_height_m": round(d, 3),
                "inlier_fraction": round(inlier_fraction, 3),
            }
        )
        camera = FloorCamera(normal, d, intrinsics, inlier_fraction)
        if best is None or camera.inlier_fraction > best.inlier_fraction:
            best = camera
    heights = [
        entry["camera_height_m"]
        for entry in keyframes
        if "camera_height_m" in entry
    ]
    summary: dict = {"keyframes": keyframes}
    if best is None:
        summary.update(
            {"fitted": False, "reason": "no keyframe produced a floor fit"}
        )
        return None, summary
    if best.inlier_fraction < MIN_FLOOR_INLIER_FRACTION:
        summary.update(
            {
                "fitted": False,
                "reason": (
                    f"best floor inlier fraction {best.inlier_fraction:.2f} "
                    f"< {MIN_FLOOR_INLIER_FRACTION} — the fit is not "
                    "trustworthy enough to measure against"
                ),
            }
        )
        return None, summary
    summary.update(
        {
            "fitted": True,
            "camera_height_m": round(best.camera_height_m, 3),
            "inlier_fraction": round(best.inlier_fraction, 3),
            "height_spread_m": round(max(heights) - min(heights), 3)
            if len(heights) > 1
            else None,
        }
    )
    return best, summary


def lift_tracks(
    tracks: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    camera: FloorCamera,
) -> dict[str, dict[int, dict]]:
    """track -> frame -> {xy, edge}. Bottom-center lift assumes feet/wheels
    on the floor; non-person bottom corners give a ground segment for
    distance measurement (near-side edge), as in the POC."""
    lifted: dict[str, dict[int, dict]] = {}
    for track_id, samples in tracks.items():
        for frame_id, (x1, y1, x2, y2) in samples:
            center = camera.lift_pixel((x1 + x2) / 2.0, y2)
            if center is None:
                continue
            entry: dict = {"xy": center}
            if not track_id.startswith(f"{PERSON_LABEL}-"):
                left = camera.lift_pixel(x1, y2)
                right = camera.lift_pixel(x2, y2)
                if left is not None and right is not None and left != right:
                    entry["edge"] = (left, right)
            lifted.setdefault(track_id, {})[frame_id] = entry
    return lifted


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
    lifted: dict[str, dict[int, dict]], frame_id: int, persons: bool
) -> list[tuple[str, dict]]:
    prefix = f"{PERSON_LABEL}-"
    return [
        (track_id, samples[frame_id])
        for track_id, samples in lifted.items()
        if track_id.startswith(prefix) == persons and frame_id in samples
    ]


def judge(
    lifted: dict[str, dict[int, dict]],
    zone: Polygon | None,
    frame_ids: list[int],
    step_seconds: float,
) -> dict:
    """The three banded temporal rules from the video POC. Rules are
    person-centric: without a "person" label every timeline is NO_DATA."""
    speed_band = 2 * BAND_M / step_seconds  # both endpoints off by the band
    r1, r2, r3 = {}, {}, {}
    r2_values, r3_values = {}, {}
    for ordinal, frame_id in enumerate(frame_ids):
        persons = _points_at(lifted, frame_id, persons=True)
        vehicles = _points_at(lifted, frame_id, persons=False)

        # R1: nobody inside the keep-clear zone. No person -> zone clear;
        # no zone authored -> nothing to judge.
        if zone is None:
            r1[frame_id] = NO_DATA
        elif persons:
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


def worst_verdict(timeline: dict[int, str]) -> str:
    if not timeline:
        return NO_DATA
    return max(timeline.values(), key=lambda v: _SEVERITY[v])


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
    lifted: dict[str, dict[int, dict]],
    zone: Polygon | None,
    r1: dict[int, str],
    path: Path,
) -> None:
    points = [
        entry["xy"] for samples in lifted.values() for entry in samples.values()
    ]
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if zone is not None:
        xs += [x for x, _ in zone.exterior.coords]
        ys += [y for _, y in zone.exterior.coords]
    xs.append(0.0)  # the camera foot point is the plane-frame origin
    ys.append(0.0)
    margin = 3.0
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    size = 900
    scale = (size - 40) / max(x1 - x0, y1 - y0)

    def pixel(xy: tuple[float, float]) -> tuple[float, float]:
        return 20 + (xy[0] - x0) * scale, size - 20 - (xy[1] - y0) * scale

    image = Image.new("RGB", (size, size), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    if zone is not None:
        draw.polygon(
            [pixel(p) for p in zone.exterior.coords], outline=(200, 60, 60), width=3
        )
        zone_pixel = pixel((zone.centroid.x, zone.centroid.y))
        draw.text((zone_pixel[0] - 30, zone_pixel[1]), "keep-clear", fill=(200, 60, 60))
    for track_id, samples in sorted(lifted.items()):
        trail = [pixel(samples[f]["xy"]) for f in sorted(samples)]
        if len(trail) < 2:
            continue
        color = _color(track_id)
        draw.line(trail, fill=color, width=3)
        draw.text(trail[-1], track_id, fill=color)
    if zone is not None:
        for track_id, samples in lifted.items():
            if not track_id.startswith(f"{PERSON_LABEL}-"):
                continue
            for frame_id, entry in samples.items():
                if r1.get(frame_id) == FAIL and zone.contains(Point(entry["xy"])):
                    px, py = pixel(entry["xy"])
                    draw.ellipse(
                        [px - 4, py - 4, px + 4, py + 4],
                        outline=(200, 30, 30),
                        width=2,
                    )
    draw.text(
        (20, 8),
        "estimated tracks (uncalibrated video-mono lift), red rings = R1 FAIL",
        fill=(60, 60, 60),
    )
    bar_y = y0 + margin / 2
    bar = pixel((x1 - margin - 5.0, bar_y)), pixel((x1 - margin, bar_y))
    draw.line([bar[0], bar[1]], fill=(0, 0, 0), width=3)
    draw.text((bar[0][0], bar[0][1] - 16), "5 m", fill=(0, 0, 0))
    px, py = pixel((0.0, 0.0))
    draw.regular_polygon((px, py, 7), 3, fill=(0, 0, 0))
    draw.text((px + 8, py - 6), "camera", fill=(0, 0, 0))
    image.save(path)


_LABEL_TINTS = [(230, 60, 40), (40, 110, 230), (38, 166, 91), (243, 156, 18)]


def _paint_masks(
    frame: Image.Image, cache_dir: Path, frame_id: int, labels: tuple[str, ...]
) -> None:
    pixels = np.asarray(frame, dtype=np.uint16)
    for ordinal, label in enumerate(labels):
        tint = np.array(_LABEL_TINTS[ordinal % len(_LABEL_TINTS)], dtype=np.uint16)
        path = _cache_path(cache_dir, frame_id, label)
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
    frame.paste(Image.fromarray(pixels.astype(np.uint8)))


def render_overlay(
    frames_dir: Path,
    cache_dir: Path,
    tracks: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    judged: dict,
    frame_ids: list[int],
    native_fps: float,
    labels: tuple[str, ...],
    gif_path: Path,
    strip_path: Path,
) -> None:
    boxes_at: dict[int, list[tuple[str, tuple]]] = {}
    for track_id, samples in tracks.items():
        for frame_id, bbox in samples:
            boxes_at.setdefault(frame_id, []).append((track_id, bbox))
    rendered: dict[int, Image.Image] = {}
    for frame_id in frame_ids:
        with Image.open(frames_dir / f"f{frame_id:06d}.jpg") as source:
            frame = source.convert("RGB")
        _paint_masks(frame, cache_dir, frame_id, labels)
        draw = ImageDraw.Draw(frame)
        for track_id, bbox in boxes_at.get(frame_id, []):
            color = _color(track_id)
            draw.rectangle(bbox, outline=color, width=4)
            draw.text((bbox[0], max(0, bbox[1] - 16)), track_id, fill=color)
        r2_value = judged["R2_values_m"].get(frame_id)
        status = " | ".join(
            f"{rule.split('_')[0]}:{judged[rule][frame_id]}" for rule in RULE_NAMES
        )
        banner = f"t={frame_id / native_fps:.1f}s  frame {frame_id}  {status}"
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
        gif_path, save_all=True, append_images=small[1:], duration=500, loop=0
    )
    # Strip: up to 6 evenly spaced full frames (no clip-specific crop —
    # productized runs have no known "action region").
    strip_ids = [
        frame_ids[i]
        for i in sorted(
            {
                round(j * (len(frame_ids) - 1) / max(1, min(6, len(frame_ids)) - 1))
                for j in range(min(6, len(frame_ids)))
            }
        )
    ]
    width = 960
    tiles = [
        rendered[f].resize(
            (width, round(width * rendered[f].height / rendered[f].width))
        )
        for f in strip_ids
    ]
    strip = Image.new("RGB", (width, sum(tile.height for tile in tiles)))
    offset = 0
    for tile in tiles:
        strip.paste(tile, (0, offset))
        offset += tile.height
    strip.save(strip_path)


# --------------------------------------------------------------------- run
def _write_manifest(store: ArtifactStore, run_id: str, paths: VideoRunPaths) -> None:
    try:
        code_version = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        code_version = "unknown"
    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        capture_tier="video-mono",
        providers=ProviderManifest(
            mapanything_model_id="unused",
            sam_endpoint=SAM3_ENDPOINT,
            gemini_model="unused",
            moge_version=MOGE_VERSION,
            code_version=code_version,
        ),
    )
    store.save_json(paths.manifest_json, manifest)


def run_video_assessment(
    video_path: str | Path,
    *,
    store: ArtifactStore,
    run_id: str,
    sample_fps: float = 1.0,
    max_frames: int = 60,
    labels: tuple[str, ...] = ("person", "car"),
    zone_wkt: str | None = None,
    start_s: float = 0.0,
    progress_cb: Callable[[str], None] | None = None,
    sam_subscriber: Callable[..., dict] | None = None,
    moge_runner: Callable[..., object] | None = None,
    track_fn: Callable[..., dict] | None = None,
    frame_extractor: Callable[..., tuple[list[int], float]] | None = None,
) -> dict:
    """Full uncalibrated-tier video assessment into runs/<run_id>/.

    Adapters are injectable exactly like the photo pipeline's: sam_subscriber
    (fal_client.subscribe-shaped), moge_runner (replicate.run-shaped),
    track_fn (bytetrack-shaped), frame_extractor (_cv2_extract-shaped).
    Quality failures (floor fit, zero masks) abstain with NO_DATA verdicts
    and a recorded reason — never a fallback result. Provider/transport
    problems raise instead.
    """
    labels = tuple(label.strip() for label in labels if label.strip())
    if not labels:
        raise ValueError("at least one object label is required")
    zone = None
    if zone_wkt:
        zone = shapely_wkt.loads(zone_wkt)
        if not isinstance(zone, Polygon):
            raise ValueError("zone_wkt must be a POLYGON")
    paths = video_paths(store, run_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    _write_manifest(store, run_id, paths)

    if progress_cb is not None:
        progress_cb("extracting frames")
    extract = frame_extractor or _cv2_extract
    extract_args = (Path(video_path), paths.frames_dir, sample_fps, max_frames)
    # start_s is passed only when set, so injected extractors keep the
    # 4-argument POC shape unless they opt into segments.
    frame_ids, native_fps = (
        extract(*extract_args, start_s) if start_s else extract(*extract_args)
    )
    if len(frame_ids) < 2:
        raise ValueError(
            f"video yielded {len(frame_ids)} sampled frame(s); need at least 2"
        )
    step_seconds = (frame_ids[1] - frame_ids[0]) / native_fps

    calls_made, failures = sam_stage(
        paths.frames_dir,
        paths.sam_cache_dir,
        frame_ids,
        labels,
        sam_subscriber or _default_sam_subscriber,
        progress_cb,
    )
    if failures and not calls_made:
        # Every uncached call failed: that is a provider outage, not an
        # empty scene — raise instead of abstaining on "no masks".
        raise ProviderError(
            "fal", "sam3.video", f"all {len(failures)} SAM calls failed"
        )
    detections = load_detections(paths.sam_cache_dir, frame_ids, labels)
    n_masks = sum(
        len(rows) for per_label in detections.values() for rows in per_label.values()
    )

    tracks = (track_fn or bytetrack)(detections, frame_ids, native_fps, labels)

    keyframe_ids = sorted(
        {frame_ids[0], frame_ids[len(frame_ids) // 2], frame_ids[-1]}
    )[:MOGE_KEYFRAMES]
    camera, floor = fit_floor_from_keyframes(
        paths.frames_dir,
        paths.moge_dir,
        keyframe_ids,
        moge_runner or _default_moge_runner,
        progress_cb,
    )

    abstained = None
    if n_masks == 0:
        abstained = "SAM returned no usable masks on any sampled frame"
    elif camera is None:
        abstained = f"floor fit failed: {floor.get('reason', 'unknown')}"

    if abstained is None:
        lifted = lift_tracks(tracks, camera)
        judged = judge(lifted, zone, frame_ids, step_seconds)
        if progress_cb is not None:
            progress_cb("rendering evidence")
        render_topdown(lifted, zone, judged["R1_zone"], paths.topdown_png)
    else:
        lifted = {}
        judged = {
            "R1_zone": {f: NO_DATA for f in frame_ids},
            "R2_min_distance": {f: NO_DATA for f in frame_ids},
            "R3_speed": {f: NO_DATA for f in frame_ids},
            "R2_values_m": {},
            "R3_values_mps": {},
            "speed_band_mps": round(2 * BAND_M / step_seconds, 3),
        }
    if n_masks:
        render_overlay(
            paths.frames_dir,
            paths.sam_cache_dir,
            tracks,
            judged,
            frame_ids,
            native_fps,
            labels,
            paths.overlay_gif,
            paths.overlay_strip_png,
        )

    verdicts = {rule: worst_verdict(judged[rule]) for rule in RULE_NAMES}
    verdicts["overall"] = max(verdicts.values(), key=lambda v: _SEVERITY[v])
    report = {
        "run_id": run_id,
        "video": Path(video_path).name,
        "native_fps": round(native_fps, 3),
        "sample_fps": sample_fps,
        "start_s": start_s,
        "step_seconds": round(step_seconds, 3),
        "sampled_frame_ids": frame_ids,
        "labels": list(labels),
        "tier": {
            "capture_tier": "video-mono",
            "band_m": BAND_M,
            "note": TIER_NOTE,
            "calibrated_reference": CALIBRATED_REFERENCE,
        },
        "floor": floor,
        "abstained": abstained,
        "thresholds": {
            "R2_min_separation_m": R2_MIN_SEPARATION_M,
            "R3_max_speed_mps": R3_MAX_SPEED_MPS,
            "speed_band_mps": judged["speed_band_mps"],
        },
        "zone_wkt": zone.wkt if zone is not None else None,
        "spend": {
            "sam_calls": calls_made,
            "sam_cost_usd": round(calls_made * COST_PER_SAM_CALL_USD, 2),
            "moge_keyframes": len(keyframe_ids),
            "failed_sam_calls": failures,
        },
        "tracks": {
            track_id: len(samples) for track_id, samples in sorted(tracks.items())
        },
        "trajectories": {
            track_id: {
                str(frame): [round(v, 3) for v in entry["xy"]]
                for frame, entry in sorted(samples.items())
            }
            for track_id, samples in sorted(lifted.items())
        },
        "timelines": {
            rule: {str(f): judged[rule][f] for f in frame_ids}
            for rule in RULE_NAMES
        },
        "R2_min_distance_m": {
            str(f): v for f, v in judged["R2_values_m"].items()
        },
        "R3_speed_mps": {str(f): v for f, v in judged["R3_values_mps"].items()},
        "verdicts": verdicts,
    }
    paths.report_json.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _main(argv: list[str] | None = None) -> int:
    """Headless runner (the app's Video tab is the primary UI): extract,
    segment, track, lift, judge one clip into runs/<run-id>/."""
    import argparse

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("video")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--labels", default="person, car")
    parser.add_argument("--zone-wkt", default=None)
    args = parser.parse_args(argv)
    report = run_video_assessment(
        args.video,
        store=ArtifactStore(args.runs_root),
        run_id=args.run_id,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        start_s=args.start_s,
        labels=tuple(part.strip() for part in args.labels.split(",")),
        zone_wkt=args.zone_wkt,
        progress_cb=lambda message: print(message, flush=True),
    )
    print(json.dumps({"verdicts": report["verdicts"], "floor": report["floor"], "spend": report["spend"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "BAND_M",
    "COST_PER_SAM_CALL_USD",
    "FloorCamera",
    "MAX_LIVE_CALLS",
    "VideoRunPaths",
    "banded_verdict",
    "fit_floor_from_keyframes",
    "fit_pinhole_from_grid",
    "judge",
    "lift_tracks",
    "load_detections",
    "ransac_plane",
    "run_video_assessment",
    "sample_schedule",
    "sam_stage",
    "video_paths",
    "worst_verdict",
    "zone_verdict",
]
