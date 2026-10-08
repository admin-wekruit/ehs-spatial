"""Uncalibrated video tier: states + trajectories + banded verdicts.

Productization of scripts/video_poc.py for footage WITHOUT a camera
calibration file: sample frames, SAM masks per frame (disk-cached under the
run dir), ByteTrack identity, then a 3D lift that fits the floor plane from
MoGe-2 metric mono depth on a few keyframes and ray-casts mask bottom-centers
to that plane. The three banded temporal rules (keep-clear zone, person-to-
vehicle min distance, person speed) run on the lifted tracks with the mono
band (±0.35 m). A rule PASSes only on frames that covered it (zone floor observed,
frame live, detector answered), a foot point decides only after lift_foot's checks, a
person who cannot be placed still makes the rules they could touch NEEDS_REVIEW, and
every verdict in metres needs measured metres (scale_gated) — the same judge_frame
serves the live people loop (ehs_spatial/live_people.py).

Accuracy tiers — be honest about which one this is:

- **video-mono (this module, uncalibrated)**: floor plane and pinhole
  intrinsics are recovered from MoGe-2's own metric point map, not measured.
  Position error is bounded below by the mono band and grows with range and
  floor-fit error; feet-on-ground is a load-bearing assumption.
- **calibrated fixed camera (the measurement tier)**: the same pipeline
  through a real KRTD model measured 0.45 m median person ATE on MEVA
  (docs/archive/demos/2026-08-25-poc-video.md). An installed camera with a real
  calibration is what turns states into measurements.
"""

import json
import subprocess
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import shapely
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box

from .artifacts import ArtifactStore
from .contracts import ProviderManifest, RunManifest
from .providers.base import ProviderError
from .providers.moge import MOGE_VERSION
from .providers.sam3 import SAM3_ENDPOINT, decode_coco_rle, sam_backend_revision, sam_subscribe
from .rules import ERROR_BUDGET_MONO_M

COST_PER_SAM_CALL_USD = 0.01  # fal list price per call
# Our Modal L4 (modal_apps/sam3_app.py, one container) bills uptime, not calls: L4 + 1 core + 8 GiB list price, as
# scripts/complete_video_objects.py prices it, over the call plus the idle time the container stays up (E).
MODAL_L4_USD_PER_S = .000222 + .0000131 + 8 * .00000222
MODAL_SCALEDOWN_S = 60.0  # Modal's default idle window; sam3_app.py does not set one
# Spend in dollars per rolling hour. A 1 Hz stream on the Modal L4 keeps one container up, about $0.91/h (E), so it
# is never stopped; the same budget buys 150 fal calls. Running out is a coverage gap, never a crash (live_people).
# ponytail: in-process only; a shared ledger once more than one worker spends on one account.
SAM_SPEND_USD_PER_HOUR = 1.50
MIN_BOX_HEIGHT_PX = 12  # a smaller mask is too small to lift: kept, but only as a person to review
MOGE_KEYFRAMES = 3
MIN_FLOOR_INLIER_FRACTION = 0.25
BAND_M = ERROR_BUDGET_MONO_M  # single-camera tier: ±0.35 m
R2_MIN_SEPARATION_M = 2.0  # person-to-vehicle keep-apart
R3_MAX_SPEED_MPS = 1.5  # walking-pace limit
SPEED_BASELINE_S = 2.0  # speed is a line fitted over at least this long, whatever the sample rate
SPEED_BAND_MPS = 2 * BAND_M / SPEED_BASELINE_S  # both ends of the baseline off by the band
MAX_HUMAN_SPEED_MPS = 4.0  # faster than this between two samples is a tracking error, not a person
PERSON_HEIGHT_M = (1.3, 2.1)  # implied standing height a foot point must explain
OCCLUSION_MARGIN_M = 0.3  # depth below the foot may be this much nearer than the floor there
FLOOR_TOLERANCE_M = 0.1  # an observed depth this close to the floor's along its ray is floor
# Both depth tolerances grow with range: a depth map is off by a share of the range, not a fixed distance.
# ponytail: about twice the median depth error measured on the test clips (was evaluate_video_policy's own copy);
# a calibration study replaces it.
DEPTH_RELATIVE = 0.05
PERSON_FOOTPRINT_M = 0.5  # side of the floor square a person stands on; an unobserved patch this big can hide one
# Zone floor grid: a PERSON_FOOTPRINT_M square, however turned, holds an axis-aligned square of side 0.5 / sqrt(2) =
# 0.35 m, and that always holds 2 x 2 samples 0.125 m apart. So a hidden person leaves a 2 x 2 block of unobserved samples.
ZONE_SAMPLE_M = 0.125
BORDER_PX = 2
DARK_MEAN = 12.0  # mean grey level (0-255) below which a frame shows nothing
FROZEN_MEAN_ABS_DIFF = 0.25  # ponytail: uncalibrated; a real stream's noise floor sets it
PERSON_LABEL = "person"

PASS, FAIL, REVIEW, NO_DATA = "PASS", "FAIL", "NEEDS_REVIEW", "NO_DATA"
_SEVERITY = {FAIL: 3, REVIEW: 2, PASS: 1, NO_DATA: 0}
RULE_NAMES = ("R1_zone", "R2_min_distance", "R3_speed")

CALIBRATED_REFERENCE = (
    "calibrated fixed-camera tier reference: 0.45 m median person ATE on "
    "MEVA with a real KRTD model (docs/archive/demos/2026-08-25-poc-video.md)"
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


class BudgetExhausted(ValueError):
    """The rolling-hour spend guard has no money left for this call."""


class HourlySpendGuard:
    """Rolling one-hour cap on paid detector spend, checked before money is spent."""

    def __init__(self, usd_per_hour: float = SAM_SPEND_USD_PER_HOUR, clock=time.monotonic) -> None:
        self.usd_per_hour, self.clock, self.spent = usd_per_hour, clock, deque()

    def remaining(self) -> float:
        now = self.clock()
        while self.spent and now - self.spent[0][0] >= 3600:
            self.spent.popleft()
        return self.usd_per_hour - sum(cost for _, cost in self.spent)

    def charge(self, usd: float) -> None:
        """Book usd before a call; nothing left (even for a call billed afterwards, usd=0) raises."""
        left = self.remaining()
        if left <= 1e-9 or usd > left + 1e-9:
            raise BudgetExhausted(
                f"budget cap: ${self.usd_per_hour - left:.2f} of the ${self.usd_per_hour:.2f}/hour SAM spend guard spent "
                "in the last hour" + (f", ${left:.2f} left for a ${usd:.2f} call" if left > 1e-9 else "")
            )
        if usd:
            self.record(usd)

    def record(self, usd: float) -> None:
        """Book money already spent (a call billed by the second, known only after it)."""
        self.spent.append((self.clock(), usd))


SAM_SPEND = HourlySpendGuard()  # one per process: the app and a live loop share it


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
    guard: HourlySpendGuard | None = None,
) -> tuple[int, list[str]]:
    """Fill the per-frame SAM cache (one prompt per label). Returns
    (live_calls_made, failure keys). Rerunning a cached run costs $0."""
    guard = guard or SAM_SPEND
    import base64

    cache_dir.mkdir(parents=True, exist_ok=True)
    planned = [
        (frame_id, label)
        for frame_id in frame_ids
        for label in labels
        if not _cache_path(cache_dir, frame_id, label).is_file()
    ]
    if len(planned) * COST_PER_SAM_CALL_USD > guard.remaining() + 1e-9:
        raise ValueError(
            f"budget cap: {len(planned)} planned SAM calls exceed the "
            f"${guard.usd_per_hour:.2f}/hour spend guard (${guard.remaining():.2f} left); "
            "reduce sample rate, frames, or labels"
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
        guard.charge(COST_PER_SAM_CALL_USD)
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
                    if not len(ys):  # a small mask is still a detection: lift_foot flags it, never drops it
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
        self.depth: np.ndarray | None = None  # static depth of the keyframes, set by the floor fit

    @property
    def camera_height_m(self) -> float:
        return self.d

    @property
    def K(self) -> np.ndarray:
        k = self.intrinsics
        return np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])

    def to_floor(self, point: np.ndarray) -> tuple[float, float]:
        offset = np.asarray(point) - self.origin
        return float(offset @ self.lateral), float(offset @ self.forward)

    def camera_points(self, floor_points: np.ndarray) -> np.ndarray:
        """(n, 3) floor (x, y, height above the floor) -> camera frame."""
        x, y, h = np.asarray(floor_points, float).T
        return self.origin + np.outer(x, self.lateral) + np.outer(y, self.forward) + np.outer(h, self.normal)

    def project(self, xy: tuple[float, float]) -> tuple[float, float] | None:
        """Floor (x, y) metres -> full-frame pixel, None behind the camera."""
        point = self.origin + xy[0] * self.lateral + xy[1] * self.forward
        if point[2] <= 1e-6:
            return None
        pixel = self.K @ (point / point[2])
        return float(pixel[0]), float(pixel[1])

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


def _static_depth(clouds: list[np.ndarray], K: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Farthest surface per pixel over the keyframe clouds: the fixed camera's static depth,
    so a person standing in one keyframe does not hide the floor behind them."""
    width, height = image_size
    depth = np.zeros((height, width))
    for points in clouds:
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)]
        pixels = np.rint(points @ K.T / points[:, 2:3]).astype(int)
        inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        np.maximum.at(depth, (pixels[inside, 1], pixels[inside, 0]), points[inside, 2])
    return depth


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
    clouds: list[np.ndarray] = []
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
        clouds.append(points)
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
    best.depth = _static_depth(clouds, best.K, image_size)
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


def lift_foot(
    bbox: tuple[float, float, float, float],
    image_wh: tuple[int, int],
    K: np.ndarray,
    plane: tuple[np.ndarray, float],
    depth: np.ndarray | None,
    exclude: np.ndarray | None = None,
    metres_per_unit: float = 1.0,
    person: bool = True,
    foot_u: float | None = None,
) -> tuple[np.ndarray | None, float | None, list[str]]:
    """The foot pixel (box bottom, at foot_u or the box centre) on the floor, in the camera frame,
    and why not to trust it yet.

    plane is (n, d), n.X + d = 0 with n pointing up to the camera. A foot point is accepted only if
    the box is at least MIN_BOX_HEIGHT_PX tall and off the image border, the box top implies a
    standing height of 1.3-2.1 m, and the depth just below the foot pixel is floor at the expected
    range: nothing nearer by more than 0.3 m or 5% of the range (an occluder hides the feet, so the
    box bottom is not the foot) and the floor was observed there, within 0.1 m or 5% of the range.
    Returns (foot, implied height m, reasons); empty reasons = accepted. A foot above the horizon
    is (None, None, ["foot_above_horizon"]): callers keep that detection as a person to review.
    """
    x1, y1, x2, y2 = bbox
    width, height = image_wh
    normal, offset = np.asarray(plane[0], float), float(plane[1])
    k_inv = np.linalg.inv(K)
    u = (x1 + x2) / 2.0 if foot_u is None else foot_u
    foot_ray = k_inv @ np.array([u, y2, 1.0])
    if normal @ foot_ray >= -1e-9:
        return None, None, ["foot_above_horizon"]
    foot = (-offset / (normal @ foot_ray)) * foot_ray
    reasons = ["box_too_small"] if y2 - y1 < MIN_BOX_HEIGHT_PX else []
    if x1 <= BORDER_PX or y1 <= BORDER_PX or x2 >= width - BORDER_PX or y2 >= height - BORDER_PX:
        reasons.append("box_touches_border")
    # Height: where the vertical through the foot passes closest to the ray through the box top.
    top_ray = k_inv @ np.array([u, y1, 1.0])
    b, c = normal @ top_ray, top_ray @ top_ray
    implied = (b * (top_ray @ foot) - c * (normal @ foot)) / (c - b * b) * metres_per_unit
    if person and not PERSON_HEIGHT_M[0] <= implied <= PERSON_HEIGHT_M[1]:
        reasons.append("implied_height")
    patch = None
    if depth is not None:
        rows = slice(int(np.ceil(y2)), min(height, int(np.ceil(y2)) + 4))
        cols = slice(max(0, int(u) - 6), min(width, int(u) + 7))  # the foot pixel's neighbourhood, not the box width
        z = depth[rows, cols]
        ok = np.isfinite(z) & (z > 0)
        if exclude is not None:
            ok &= ~exclude[rows, cols]
        if ok.sum() >= 3:
            vs, us = np.nonzero(ok)
            rays = np.c_[us + cols.start, vs + rows.start, np.ones(len(us))] @ k_inv.T
            expected = -offset / (rays @ normal) * metres_per_unit  # z of the floor along each ray
            nearer = expected - z[ok] * metres_per_unit
            patch = (np.median(nearer) > max(OCCLUSION_MARGIN_M, DEPTH_RELATIVE * np.median(expected)),
                     np.mean(np.abs(nearer) <= np.maximum(FLOOR_TOLERANCE_M, DEPTH_RELATIVE * expected)))
    if patch is None:
        reasons.append("no_depth_below_foot")
    else:
        if patch[0]:
            reasons.append("foot_occluded")
        if patch[1] < 0.5:
            reasons.append("foot_off_observed_floor")
    return foot, float(implied), reasons


def lift_tracks(
    tracks: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    camera: FloorCamera,
    image_wh: tuple[int, int] | None = None,
    zone: Polygon | None = None,
) -> dict[str, dict[int, dict]]:
    """track -> frame -> {xy, edge, review, near_zone}. Bottom-centre lift assumes feet/wheels
    on the floor, so every foot passes lift_foot's checks against the fixed camera's
    static depth before it can decide a rule; non-person bottom corners give a ground
    segment for distance measurement (near-side edge), as in the POC. A box that cannot be
    placed (foot above the horizon) stays, with xy None and its reason: a person nobody can
    place is still a person. near_zone: the box overlaps where a person in the zone could appear."""
    if image_wh is None:
        image_wh = (int(round(2 * camera.intrinsics["cx"])), int(round(2 * camera.intrinsics["cy"])))
    plane = (camera.normal, camera.d)
    region = image_region(camera.camera_points(zone_prism_points(zone)), camera.K, image_wh) if zone is not None else None
    lifted: dict[str, dict[int, dict]] = {}
    for track_id, samples in tracks.items():
        person = track_id.startswith(f"{PERSON_LABEL}-")
        for frame_id, bbox in samples:
            foot, height_m, reasons = lift_foot(bbox, image_wh, camera.K, plane, camera.depth, person=person)
            entry: dict = {"xy": None if foot is None else camera.to_floor(foot), "review": reasons,
                           "near_zone": region is None or region.intersects(box(*bbox))}
            lifted.setdefault(track_id, {})[frame_id] = entry
            if foot is None:
                continue
            if height_m is not None:
                entry["height_m"] = round(height_m, 3)
            if not person:
                x1, _, x2, y2 = bbox
                left = camera.lift_pixel(x1, y2)
                right = camera.lift_pixel(x2, y2)
                if left is not None and right is not None and left != right:
                    entry["edge"] = (left, right)
    return lifted


def keep_untracked(
    tracks: dict[str, list[tuple[int, tuple[float, float, float, float]]]],
    detections: dict[int, dict[str, list[dict]]],
    min_iou: float = 0.5,
) -> dict[str, list[tuple[int, tuple[float, float, float, float]]]]:
    """A detection the tracker gave no identity (ByteTrack holds back a first sighting) is still in that frame: it
    joins as a one-frame track, so it can decide or block a rule but never gives a speed."""
    tracked: dict[tuple[str, int], list] = {}
    for track_id, samples in tracks.items():
        for frame_id, bbox in samples:
            tracked.setdefault((track_id.rsplit("-", 1)[0], frame_id), []).append(box(*bbox))
    kept = dict(tracks)
    for frame_id, per_label in detections.items():
        for label, rows in per_label.items():
            for ordinal, row in enumerate(rows):
                shape = box(*row["bbox"])
                if all(shape.intersection(other).area < min_iou * shape.union(other).area
                       for other in tracked.get((label, frame_id), [])):
                    kept[f"{label}-untracked-{frame_id}-{ordinal}"] = [(frame_id, row["bbox"])]
    return kept


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


def contract_scale(scale: dict) -> dict:
    """The platform's scale record for one metric-scale.json, decided by the scale that was actually written.

    The platform accepts three states and its rule engine only gives PASS/FAIL on operator_anchored. Only a measurement
    may take that state: metric poses from the capture device. A carry height the operator stated but nobody measured is
    an assumption and becomes model_estimated, so the engine asks for calibration instead of deciding (PLAN.md section 6:
    an assumed value is never stored as a measurement). A clip with no metres at all stays uncalibrated, and so does a clip
    whose lens failed its floor-plane gate (intrinsics_uncertain): its geometry stays, in native units, and no rule decides.
    """
    status = scale.get("scale_status")
    if status == "device_metric":
        return {"status": "operator_anchored", "nativeToMeters": scale["metres_per_native_unit"],
                "anchor": {"kind": "device_metric_poses", "measured": True, "note": "metres come from the capture device's own poses; nothing was assumed"}}
    if status in ("assumed_camera_height", "assumed_camera_height_floor_views_disagree"):
        stated = scale["metres_per_native_unit"] * scale["camera_height_native_median"]  # the height the run was told, recovered exactly
        return {"status": "model_estimated", "nativeToMeters": scale["metres_per_native_unit"],
                "anchor": {"kind": "stated_carry_height", "metres": round(stated, 4), "measured": False,
                           "floorViewsAgree": status == "assumed_camera_height",
                           "modelEstimateMetresPerNative": scale.get("model_estimated_metres_per_native_unit"),
                           "sourcesDisagreeOver10pct": scale.get("scale_sources_disagree_over_10pct")}}
    if status == "model_metric":  # a model's own metric depth (MoGe): metres nobody measured
        return {"status": "model_estimated", "nativeToMeters": 1.0, "anchor": {"kind": "model_metric_depth", "measured": False}}
    if status == "uncalibrated":
        return {"status": "uncalibrated"}
    if status == "intrinsics_uncertain":  # the shot's lens failed its floor-plane gate (M2 lens rule): no metres at all, whatever was stated
        return {"status": "uncalibrated", "intrinsicsUncertain": True, "planeInlierFraction": scale.get("plane_inlier_fraction")}
    raise ValueError(f"unknown scale_status {status!r}: refuse rather than guess what it means")


def scale_gated(verdict: str, scale_record: dict) -> str:
    """The one gate every distance and speed verdict passes: without measured metres a
    PASS or FAIL can only ask for review (需复核)."""
    if verdict in (PASS, FAIL) and scale_record.get("status") != "operator_anchored":
        return REVIEW
    return verdict


def frame_health(gray: np.ndarray, previous: np.ndarray | None) -> str | None:
    """Why this frame cannot show a person (dark / frozen), None when it can."""
    gray = np.asarray(gray, dtype=np.float32)
    if gray.mean() < DARK_MEAN:
        return "dark"
    if previous is not None and previous.shape == gray.shape and np.abs(gray - previous).mean() < FROZEN_MEAN_ABS_DIFF:
        return "frozen"
    return None


def zone_floor_points(zone: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """The zone floor's samples: (n, 3) points (x, y, height 0), its outline ZONE_SAMPLE_M apart and then the centres of
    the ZONE_SAMPLE_M grid cells inside it; and (n, 2) int grid cells of those points. An outline sample sits in the cell
    it falls in, so a part of the zone narrower than two cells, which few or no cell centres fall in, is still sampled.
    ponytail: every cell of the zone, so the cost grows with its area (64 a square metre); fine for keep-clear zones."""
    step = ZONE_SAMPLE_M
    outline = shapely.get_coordinates(shapely.line_interpolate_point(zone.exterior, np.arange(0, zone.exterior.length, step)))
    minx, miny, maxx, maxy = zone.bounds
    shape = np.maximum(np.ceil([(maxx - minx) / step, (maxy - miny) / step]).astype(int), 1)
    cols, rows = (g.ravel() for g in np.meshgrid(np.arange(shape[0]), np.arange(shape[1])))
    xs, ys = minx + (cols + 0.5) * step, miny + (rows + 0.5) * step
    inside = shapely.contains_xy(zone, xs, ys)
    xy = np.vstack([outline, np.c_[xs[inside], ys[inside]]])
    edge = np.clip(np.floor((outline - [minx, miny]) / step).astype(int), 0, shape - 1)
    cells = np.vstack([edge, np.c_[cols[inside], rows[inside]]]).astype(int)
    return np.c_[xy, np.zeros(len(xy))], cells


def zone_prism_points(zone: Polygon) -> np.ndarray:
    """(n, 3) corners of the space a person standing in the zone can fill: the zone grown by the position band, from the
    floor up to the tallest person."""
    corners = np.array(zone.buffer(BAND_M).exterior.coords)
    return np.vstack([np.c_[corners, np.full(len(corners), h)] for h in (0.0, PERSON_HEIGHT_M[1])])


def image_region(points: np.ndarray, K: np.ndarray, image_wh: tuple[int, int]) -> Polygon:
    """Where camera-frame points can appear in the image: the hull of their projections, or the whole image when any
    is behind the camera (ponytail: clip at the near plane if that asks for review too often)."""
    frame = box(0, 0, *image_wh)
    if (points[:, 2] <= 1e-6).any():
        return frame
    pixels = points @ K.T / points[:, 2:3]
    return MultiPoint(pixels[:, :2]).convex_hull.intersection(frame)


def zone_floor_seen(points: np.ndarray, cells: np.ndarray, K: np.ndarray, depth: np.ndarray | None,
                    image_wh: tuple[int, int], blocked: np.ndarray | None = None) -> bool:
    """The zone floor was observed in this frame well enough that nobody can stand on it unseen. points and cells are
    zone_floor_points' (points moved into the camera frame). Every point must project in front of the camera and inside
    the image. A sample is observed when the depth there reaches the floor (nothing nearer by more than lift_foot's
    occlusion margin) and no detection covers its pixel (blocked, for a static depth plate that cannot see a passing
    occluder). A cell is hidden when any of its samples is. An isolated hidden cell is sensor noise; a 2 x 2 block of cells
    whose sampled cells are all hidden is floor where a PERSON_FOOTPRINT_M footprint could hide, so the zone is not seen.
    Cells no sample falls in are not zone, so a part of the zone one cell wide (a zone one cell wide: 1 x 2 blocks) is
    judged by its own samples, outline included, and never passes for want of four hidden cells.
    ponytail: a footprint straddling the outline is judged by its part inside the zone only."""
    width, height = image_wh
    if depth is None or (points[:, 2] <= 1e-6).any():
        return False
    pixels = points @ K.T / points[:, 2:3]
    u, v = pixels[:, 0], pixels[:, 1]
    if not ((u >= 0) & (u < width) & (v >= 0) & (v < height)).all():
        return False
    u, v = u.astype(int), v.astype(int)
    observed, expected = depth[v, u], points[:, 2]  # metres, as every caller's depth is
    seen = (observed > 0) & (expected - observed <= np.maximum(OCCLUSION_MARGIN_M, DEPTH_RELATIVE * expected))  # NaN: unseen
    if blocked is not None:
        seen &= ~blocked[v, u]
    sampled, clear = np.zeros((2, *np.maximum(cells.max(0) + 1, 2)), bool)  # at least 2 x 2: a one-cell-wide zone has blocks
    sampled[tuple(cells.T)] = clear[tuple(cells.T)] = True
    clear[tuple(cells[~seen].T)] = False
    block = lambda a: a[:-1, :-1] | a[1:, :-1] | a[:-1, 1:] | a[1:, 1:]  # noqa: E731  any cell of each 2 x 2 block
    return not (block(sampled) & ~block(clear)).any()


def track_speed(samples: Sequence[tuple[float, tuple[float, float]]], t: float) -> tuple[float | None, str | None]:
    """One track's speed at time t: a least-squares line through its accepted positions over
    the last SPEED_BASELINE_S seconds (from the newest sample at or before t - baseline).

    The baseline is fixed in seconds, so the value and its band do not move with the sample
    rate. (None, None) until a full baseline exists; (None, "jump") when two samples imply more
    than MAX_HUMAN_SPEED_MPS beyond the position band: the segment is a tracking error to
    review, never a FAIL. samples are in time order; only the window is read, newest first, so
    the cost does not grow with the track's history.
    """
    window = []
    for s, xy in reversed(samples):
        if s > t + 1e-6:
            continue
        window.append((s, xy))
        if s <= t - SPEED_BASELINE_S + 1e-6:
            break
    else:
        return None, None
    window.reverse()
    if len(window) < 2 or window[-1][0] - window[0][0] > 2 * SPEED_BASELINE_S or window[-1][0] < t - 1e-6:
        return None, None  # too sparse, a gap too long to bridge, or the track is not seen now
    for (s0, a), (s1, b) in zip(window, window[1:]):
        if (np.hypot(b[0] - a[0], b[1] - a[1]) - 2 * BAND_M) / max(s1 - s0, 1e-6) > MAX_HUMAN_SPEED_MPS:
            return None, "jump"
    times = np.array([s for s, _ in window])
    slope = np.polyfit(times - times.mean(), np.array([xy for _, xy in window]), 1)[0]
    return float(np.hypot(*slope)), None


def judge_frame(
    persons: list[dict],
    movers: list[dict],
    zone: Polygon | None,
    gap: str | None,
    zone_seen: bool,
    speeds: list[tuple[float | None, str | None]],
    scale_record: dict,
) -> dict:
    """The three rules at one instant; the batch judge and the live loop both call this.

    gap: why this instant shows nothing (no pose, dark, frozen, detector offline) -> every rule
    NO_DATA. zone_seen: the zone floor was observed; without it nobody-in-the-zone is NO_DATA,
    never PASS (a person seen inside still FAILs). Entries carry "review" reasons from
    lift_foot and xy None when they could not be placed at all: such a person can only make a
    rule NEEDS_REVIEW, and for R1 only if near_zone (their box overlaps where a person in the
    zone could appear; missing counts as near). Zone membership is judged in metres with a
    metre band, so R1 goes through scale_gated like R2/R3; the pre-gate verdict ("raw") is kept.
    """
    if gap is not None:
        return {rule: {"verdict": NO_DATA, "reason": gap} for rule in RULE_NAMES}
    out = {}
    if zone is None:
        out["R1_zone"] = {"verdict": NO_DATA, "reason": "no zone"}
    else:
        verdicts = [REVIEW if p.get("review") else zone_verdict(zone, p["xy"], BAND_M)
                    for p in persons if p.get("near_zone", True) or not p.get("review")]
        raw = max(verdicts, key=lambda v: _SEVERITY[v]) if verdicts else PASS
        raw = NO_DATA if raw == PASS and not zone_seen else raw
        out["R1_zone"] = {"verdict": scale_gated(raw, scale_record), "raw": raw,
                          "reason": None if zone_seen else "zone floor not observed"}
    if persons and movers:
        pairs = []
        for person in persons:
            for mover in movers:
                if person["xy"] is None or mover["xy"] is None:
                    pairs.append((None, REVIEW))  # someone who could not be placed: the distance is unknown
                    continue
                geometry = LineString(mover["edge"]) if "edge" in mover else Point(mover["xy"])
                distance = Point(person["xy"]).distance(geometry)
                raw = REVIEW if person.get("review") or mover.get("review") else banded_verdict(
                    distance, R2_MIN_SEPARATION_M, BAND_M, fail_low=True)
                pairs.append((distance, raw))
        raw = max((v for _, v in pairs), key=lambda v: _SEVERITY[v])
        known = [d for d, _ in pairs if d is not None]
        out["R2_min_distance"] = {"verdict": scale_gated(raw, scale_record), "raw": raw,
                                  "value": round(min(known), 3) if known else None}
    else:
        out["R2_min_distance"] = {"verdict": NO_DATA, "reason": "needs a person and a mover"}
    raws = [REVIEW if reason else banded_verdict(speed, R3_MAX_SPEED_MPS, SPEED_BAND_MPS, fail_low=False)
            for speed, reason in speeds if speed is not None or reason]
    if raws:
        raw = max(raws, key=lambda v: _SEVERITY[v])
        known = [speed for speed, _ in speeds if speed is not None]
        out["R3_speed"] = {"verdict": scale_gated(raw, scale_record), "raw": raw,
                           "value": round(max(known), 3) if known else None}
    else:
        out["R3_speed"] = {"verdict": NO_DATA, "reason": f"no track with a {SPEED_BASELINE_S:g} s baseline"}
    return out


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
    frame_times: dict[int, float],
    coverage: dict[int, dict] | None = None,
    scale_record: dict | None = None,
) -> dict:
    """The three rules over a sampled clip. Rules are person-centric: without a person
    detector every timeline is NO_DATA. coverage: frame -> {"gap": reason|None,
    "zone_seen": bool}; a frame without an entry never saw the zone, so it cannot PASS it. scale_record defaults to
    the video-mono tier's model-metric depth, so every rule gives NEEDS_REVIEW, never PASS/FAIL; the verdicts before
    that gate are kept under "before_scale_gate"."""
    scale_record = scale_record or contract_scale({"scale_status": "model_metric"})
    coverage = coverage or {}
    person_samples = {
        track_id: [(frame_times[f], entry["xy"]) for f, entry in sorted(samples.items())
                   if f in frame_times and not entry.get("review")]
        for track_id, samples in lifted.items() if track_id.startswith(f"{PERSON_LABEL}-")
    }
    timelines: dict[str, dict] = {rule: {} for rule in RULE_NAMES}
    before_gate: dict[str, dict] = {rule: {} for rule in RULE_NAMES}
    r2_values, r3_values = {}, {}
    for frame_id in sorted(frame_times):
        t = frame_times[frame_id]
        persons = _points_at(lifted, frame_id, persons=True)
        speeds = [(None, "unaccepted foot") if entry.get("review") else track_speed(person_samples[track_id], t)
                  for track_id, entry in persons]
        state = coverage.get(frame_id, {})
        result = judge_frame([e for _, e in persons], [e for _, e in _points_at(lifted, frame_id, persons=False)],
                             zone, state.get("gap"), state.get("zone_seen", False), speeds, scale_record)
        for rule in RULE_NAMES:
            timelines[rule][frame_id] = result[rule]["verdict"]
            before_gate[rule][frame_id] = result[rule].get("raw", result[rule]["verdict"])
        if result["R2_min_distance"].get("value") is not None:
            r2_values[frame_id] = result["R2_min_distance"]["value"]
        if result["R3_speed"].get("value") is not None:
            r3_values[frame_id] = result["R3_speed"]["value"]
    return {
        **timelines,
        "R2_values_m": r2_values,
        "R3_values_mps": r3_values,
        "before_scale_gate": before_gate,
        "speed_band_mps": round(SPEED_BAND_MPS, 3),
        "scale": scale_record,
    }


_SUMMARY = {FAIL: 3, REVIEW: 2, NO_DATA: 1, PASS: 0}  # over time: an unobserved stretch outranks a PASS


def worst_verdict(timeline: dict[int, str]) -> str:
    """FAIL > NEEDS_REVIEW > NO_DATA > PASS: PASS only when every instant passed (one PASS and 59 NO_DATA used to sum to PASS)."""
    if not timeline:
        return NO_DATA
    return max(timeline.values(), key=lambda v: _SUMMARY[v])


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
        entry["xy"] for samples in lifted.values() for entry in samples.values() if entry["xy"] is not None
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
        trail = [pixel(samples[f]["xy"]) for f in sorted(samples) if samples[f]["xy"] is not None]
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
                if r1.get(frame_id) == FAIL and entry["xy"] is not None and zone.contains(Point(entry["xy"])):
                    px, py = pixel(entry["xy"])
                    draw.ellipse(
                        [px - 4, py - 4, px + 4, py + 4],
                        outline=(200, 30, 30),
                        width=2,
                    )
    draw.text(
        (20, 8),
        "estimated tracks (uncalibrated video-mono lift), red rings = R1 FAIL before the scale gate",
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
def _write_manifest(store: ArtifactStore, run_id: str, paths: VideoRunPaths, sam: dict) -> None:
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
            sam_backend=sam["backend"],
            sam_model_revision=sam["model_revision"],
        ),
    )
    store.save_json(paths.manifest_json, manifest)


def frame_coverage(
    frames_dir: Path,
    cache_dir: Path,
    frame_ids: list[int],
    zone: Polygon | None,
    camera: FloorCamera | None,
    detections: dict[int, dict[str, list[dict]]] | None = None,
) -> dict[int, dict]:
    """frame -> {"gap", "zone_seen"} for judge_frame: a sampled frame covers the rules only
    if it is neither dark nor frozen and the person detector answered for it; it saw the zone
    only if the static depth plate reaches the zone floor and no detected box of that frame
    stands in front of it."""
    coverage: dict[int, dict] = {}
    previous = None
    floor_points, floor_cells = zone_floor_points(zone) if zone is not None and camera is not None else (None, None)
    floor_points = None if floor_points is None else camera.camera_points(floor_points)
    for frame_id in frame_ids:
        with Image.open(frames_dir / f"f{frame_id:06d}.jpg") as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32)
        gap = frame_health(gray, previous)
        previous = gray
        if gap is None and not _cache_path(cache_dir, frame_id, PERSON_LABEL).is_file():
            gap = "person detector offline"
        zone_seen = False
        if floor_points is not None:
            blocked = np.zeros(gray.shape, bool)
            for rows in (detections or {}).get(frame_id, {}).values():
                for row in rows:
                    x1, y1, x2, y2 = (int(round(v)) for v in row["bbox"])
                    blocked[max(0, y1):y2, max(0, x1):x2] = True
            zone_seen = zone_floor_seen(floor_points, floor_cells, camera.K, camera.depth, (gray.shape[1], gray.shape[0]), blocked)
        coverage[frame_id] = {"gap": gap, "zone_seen": zone_seen}
    return coverage


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
    sam = sam_backend_revision() if sam_subscriber is None else {"backend": "injected", "model_revision": "unknown"}
    _write_manifest(store, run_id, paths, sam)

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
        sam_subscriber or sam_subscribe,
        progress_cb,
    )
    if failures and not calls_made:
        # Every uncached call failed: that is a provider outage, not an
        # empty scene — raise instead of abstaining on "no masks".
        raise ProviderError(
            sam["backend"], "sam3.video", f"all {len(failures)} SAM calls failed"
        )
    detections = load_detections(paths.sam_cache_dir, frame_ids, labels)
    n_masks = sum(
        len(rows) for per_label in detections.values() for rows in per_label.values()
    )

    tracks = keep_untracked((track_fn or bytetrack)(detections, frame_ids, native_fps, labels), detections)

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

    frame_times = {f: f / native_fps for f in frame_ids}
    coverage = frame_coverage(paths.frames_dir, paths.sam_cache_dir, frame_ids, zone, camera, detections)
    if abstained is None:
        with Image.open(paths.frames_dir / f"f{frame_ids[0]:06d}.jpg") as image:
            image_wh = image.size
        lifted = lift_tracks(tracks, camera, image_wh, zone)
        judged = judge(lifted, zone, frame_times, coverage)
        if progress_cb is not None:
            progress_cb("rendering evidence")
        render_topdown(lifted, zone, judged["before_scale_gate"]["R1_zone"], paths.topdown_png)
    else:
        lifted = {}
        judged = {
            "R1_zone": {f: NO_DATA for f in frame_ids},
            "R2_min_distance": {f: NO_DATA for f in frame_ids},
            "R3_speed": {f: NO_DATA for f in frame_ids},
            "before_scale_gate": {rule: {f: NO_DATA for f in frame_ids} for rule in RULE_NAMES},
            "R2_values_m": {},
            "R3_values_mps": {},
            "speed_band_mps": round(SPEED_BAND_MPS, 3),
            "scale": contract_scale({"scale_status": "model_metric"}),
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
            "speed_baseline_s": SPEED_BASELINE_S,
            "max_human_speed_mps": MAX_HUMAN_SPEED_MPS,
            "person_height_m": list(PERSON_HEIGHT_M),
            "occlusion_margin_m": OCCLUSION_MARGIN_M,
        },
        "scale": judged["scale"],
        "coverage": {str(f): c for f, c in coverage.items()},
        "zone_wkt": zone.wkt if zone is not None else None,
        "spend": {
            "sam_calls": calls_made,
            "sam_cost_usd": round(calls_made * COST_PER_SAM_CALL_USD, 2),
            "moge_keyframes": len(keyframe_ids),
            "failed_sam_calls": failures,
            "sam_backend": sam,
        },
        "tracks": {
            track_id: len(samples) for track_id, samples in sorted(tracks.items())
        },
        "trajectories": {
            track_id: {
                str(frame): [round(v, 3) for v in entry["xy"]]
                for frame, entry in sorted(samples.items()) if entry["xy"] is not None
            }
            for track_id, samples in sorted(lifted.items())
        },
        "foot_review": {
            track_id: {str(frame): entry["review"] for frame, entry in sorted(samples.items()) if entry.get("review")}
            for track_id, samples in sorted(lifted.items())
        },
        "timelines": {
            rule: {str(f): judged[rule][f] for f in frame_ids}
            for rule in RULE_NAMES
        },
        "timelines_before_scale_gate": {
            rule: {str(f): judged["before_scale_gate"][rule][f] for f in frame_ids}
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
    "HourlySpendGuard",
    "SAM_SPEND_USD_PER_HOUR",
    "VideoRunPaths",
    "banded_verdict",
    "contract_scale",
    "frame_health",
    "judge_frame",
    "lift_foot",
    "scale_gated",
    "track_speed",
    "BudgetExhausted",
    "image_region",
    "keep_untracked",
    "zone_floor_points",
    "zone_floor_seen",
    "zone_prism_points",
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
