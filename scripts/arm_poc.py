"""Robot-arm POC: arm states + sweep envelopes on a DROID episode, FK-validated.

Ground truth ($0, deterministic): Franka Panda forward kinematics from the
episode's recorded joint positions. The standard modified-DH table reproduces
the h5's own cartesian_position record with 0.00 mm error on every frame
(docs/research/2026-08-25-robotcell-data.md), so FK link positions ARE the
exact arm geometry. Camera extrinsics come from the h5 (camera pose in the
robot base frame, extrinsic-xyz euler — the convention was proven by the
wrist camera, whose extrinsics stay a constant rigid transform from the FK
flange across all frames only under 'xyz').

Estimate (the commodity stack): sampled exterior frames -> SAM 3.1 "robot
arm" masks (disk cache, --live gate, budget cap) -> MoGe-2 metric mono depth
(replicate, cached) -> masked pixels lifted to a camera-frame point cloud ->
base frame via the h5 extrinsics.

Judgment, estimated vs FK truth:
  R-A  ACTIVE/PARKED state timeline (mask motion energy vs joint-velocity norm)
  R-B  plan-view sweep envelope: IoU + boundary discrepancy in metres
  R-C  banded keep-clear rules: (1) "arm never exits TRUE envelope + 0.3 m",
       (2) a keep-out line the arm genuinely crosses, so GT contains FAILs
       and silent misses are measurable.

Honesty: every threshold is a stated constant below; nothing is tuned to
close gaps. If the MoGe metric lift is off, the FK-anchored scale fit says
by how much — that number is itself the finding (does an arm cell need a
scale reference?).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint, Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehs_spatial.providers.sam3 import SAM3_ENDPOINT, decode_coco_rle  # noqa: E402
from ehs_spatial.providers.moge import MOGE_VERSION  # noqa: E402
from ehs_spatial.rules import ERROR_BUDGET_MONO_M  # noqa: E402

# ------------------------------------------------------------ stated constants
DATA_ROOT = Path(
    "/Users/adam/Desktop/Tesla/ehs-spatial/outputs/datasets/robotcell/droid"
)
EPISODE = "AUTOLab__Tue_Nov__7_13:53:20_2023"
CAMERA_SERIAL = "22008760"  # ext1: the only AUTOLab exterior view not curtained

SAMPLE_STRIDE = 2  # every 2nd control step (~7 fps at the ~14.3 Hz record)
MAX_SAMPLES = 90  # == the MoGe call budget

SAM_PROMPTS = ("robot arm", "robotic arm")  # ordered fallback
SAM_COST_PER_CALL_USD = 0.01
MOGE_COST_PER_CALL_EST_USD = 0.005  # replicate bills GPU-seconds; estimate
MAX_SAM_CALLS = 120
MAX_MOGE_CALLS = 90

GT_ACTIVE_RADPS = 0.05  # joint-velocity norm above this = ACTIVE (truth)
EST_ACTIVE_MASK_CHANGE = 0.05  # mask XOR/union above this = ACTIVE (estimate)

BAND_M = ERROR_BUDGET_MONO_M  # mono tier: ±0.35 m
ZONE_MARGIN_M = 0.30  # keep-clear zone = TRUE envelope + this
KEEPOUT_INSET_M = 0.15  # keep-out line sits this far inside the TRUE max-x

FK_DENSE_STEP_M = 0.05  # sample spacing along FK link segments
MASK_ERODE_PX = 2  # cut mask-boundary depth bleed
PIXEL_STRIDE = 3  # subsample mask pixels for the lift
MAX_RANGE_M = 2.0  # cell interior; masked depths beyond this are bleed

PASS, FAIL, REVIEW, NO_DATA = "PASS", "FAIL", "NEEDS_REVIEW", "NO_DATA"
ACTIVE, PARKED = "ACTIVE", "PARKED"

# ------------------------------------------------------------------ Panda FK
# Modified DH (Craig): rows of (a_{i-1}, d_i, alpha_{i-1}) for joints 1..7.
PANDA_DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
PANDA_FLANGE_D = 0.107


def _dh_step(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def fk_flange(q: np.ndarray) -> np.ndarray:
    """7 joint angles -> 4x4 flange pose in the robot base frame."""
    T = np.eye(4)
    for (a, d, alpha), theta in zip(PANDA_DH, q, strict=True):
        T = T @ _dh_step(a, d, alpha, theta)
    return T @ _dh_step(0.0, PANDA_FLANGE_D, 0.0, 0.0)


def fk_origins(q: np.ndarray) -> np.ndarray:
    """Base + 7 joint-frame origins + flange origin, (9, 3), base frame."""
    T = np.eye(4)
    origins = [T[:3, 3].copy()]
    for (a, d, alpha), theta in zip(PANDA_DH, q, strict=True):
        T = T @ _dh_step(a, d, alpha, theta)
        origins.append(T[:3, 3].copy())
    T = T @ _dh_step(0.0, PANDA_FLANGE_D, 0.0, 0.0)
    origins.append(T[:3, 3].copy())
    return np.array(origins)


def densify_skeleton(origins: np.ndarray, step: float = FK_DENSE_STEP_M) -> np.ndarray:
    """Points every `step` metres along the link segments (kinematic skeleton)."""
    points = []
    for a, b in zip(origins, origins[1:], strict=False):
        length = float(np.linalg.norm(b - a))
        n = max(2, int(length / step) + 1)
        for s in np.linspace(0.0, 1.0, n):
            points.append(a + s * (b - a))
    return np.array(points)


def euler_xyz_to_matrix(angles: np.ndarray) -> np.ndarray:
    """Extrinsic x-y-z euler -> rotation matrix (== scipy from_euler('xyz'))."""
    ax, ay, az = angles
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Inverse of euler_xyz_to_matrix (non-degenerate case)."""
    ay = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    ax = np.arctan2(R[2, 1], R[2, 2])
    az = np.arctan2(R[1, 0], R[0, 0])
    return np.array([ax, ay, az])


# ---------------------------------------------------------------- h5 loading
def load_episode(episode_dir: Path, camera_serial: str) -> dict:
    import h5py

    with h5py.File(episode_dir / "trajectory.h5", "r") as f:
        q = f["observation/robot_state/joint_positions"][:]
        qd = f["observation/robot_state/joint_velocities"][:]
        cart = f["observation/robot_state/cartesian_position"][:]
        ext = f[f"observation/camera_extrinsics/{camera_serial}_left"][:]
        ts_ms = f["observation/timestamp/control/step_start"][:]
    # Exterior extrinsics are constant over the episode; take row 0.
    cam_pos = ext[0, :3]
    cam_rot = euler_xyz_to_matrix(ext[0, 3:])  # camera -> base
    return {
        "q": q,
        "qd": qd,
        "cart": cart,
        "cam_pos": cam_pos,
        "cam_rot": cam_rot,
        "t_s": (ts_ms - ts_ms[0]) / 1000.0,
    }


def validate_fk(q: np.ndarray, cart: np.ndarray) -> float:
    """Max |FK flange - recorded cartesian| over the episode, metres."""
    errors = [
        float(np.linalg.norm(fk_flange(q[i])[:3, 3] - cart[i, :3]))
        for i in range(len(q))
    ]
    return max(errors)


# ------------------------------------------------------------- frame extract
def extract_frames(video: Path, frame_ids: list[int], frames_dir: Path) -> None:
    missing = [f for f in frame_ids if not (frames_dir / f"f{f:06d}.jpg").is_file()]
    if not missing:
        return
    import cv2

    frames_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(missing)
    capture = cv2.VideoCapture(str(video))
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


# ------------------------------------------------------------------ SAM stage
def _sam_cache_path(cache_dir: Path, frame_id: int, prompt: str) -> Path:
    return cache_dir / f"f{frame_id:06d}__{prompt.replace(' ', '_')}.json"


def _image_uri(path: Path) -> str:
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


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
                    "max_masks": 8,
                },
            )
        except Exception as exc:  # 429/5xx and transport errors
            last_error = exc
            time.sleep(2**attempt * 2)
    raise last_error  # type: ignore[misc]


def sam_planned_calls(cache_dir: Path, frame_ids: list[int]) -> list[tuple[int, str]]:
    calls = []
    for frame_id in frame_ids:
        for prompt in SAM_PROMPTS:
            path = _sam_cache_path(cache_dir, frame_id, prompt)
            if path.is_file():
                if json.loads(path.read_text()).get("rle"):
                    break  # this prompt already hit; no fallback needed
                continue
            calls.append((frame_id, prompt))
            break  # later fallbacks depend on this result
    return calls


def sam_stage(
    frames_dir: Path, cache_dir: Path, frame_ids: list[int], live: bool
) -> tuple[int, list[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    calls_made = 0
    while True:
        calls = [
            call
            for call in sam_planned_calls(cache_dir, frame_ids)
            if f"{call[0]}:{call[1]}" not in failures
        ]
        if not calls:
            break
        if not live:
            print(
                f"sam: {len(calls)} uncached calls planned "
                f"(~${len(calls) * SAM_COST_PER_CALL_USD:.2f}); rerun with --live",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if calls_made + len(calls) > MAX_SAM_CALLS:
            raise SystemExit(
                f"sam budget cap: {calls_made} made + {len(calls)} planned "
                f"> {MAX_SAM_CALLS}"
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
            _sam_cache_path(cache_dir, frame_id, prompt).write_text(
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


def load_arm_mask(cache_dir: Path, frame_id: int) -> np.ndarray | None:
    """Union of the winning prompt's masks, bool HxW; None if no cache/mask."""
    for prompt in SAM_PROMPTS:
        path = _sam_cache_path(cache_dir, frame_id, prompt)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        rles = payload.get("rle") or []
        if isinstance(rles, str):
            rles = [rles]
        if not rles:
            continue
        union = np.zeros((payload["height"], payload["width"]), dtype=bool)
        for serialized in rles:
            union |= decode_coco_rle(
                serialized, height=payload["height"], width=payload["width"]
            ).astype(bool)
        return union
    return None


def erode_mask(mask: np.ndarray, pixels: int = MASK_ERODE_PX) -> np.ndarray:
    """Binary erosion by `pixels` via numpy shifts (no scipy/cv2 needed)."""
    out = mask.copy()
    for _ in range(pixels):
        shrunk = out.copy()
        shrunk[1:, :] &= out[:-1, :]
        shrunk[:-1, :] &= out[1:, :]
        shrunk[:, 1:] &= out[:, :-1]
        shrunk[:, :-1] &= out[:, 1:]
        out = shrunk
    return out


# ----------------------------------------------------------------- MoGe stage
def parse_ply_xyz(data: bytes) -> np.ndarray:
    """Minimal PLY vertex reader (binary little-endian or ascii), xyz only."""
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii", errors="replace")
    fmt = "ascii" if "format ascii" in header else "binary_little_endian"
    count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            properties.append((parts[1], parts[2]))
    sizes = {"float": 4, "float32": 4, "double": 8, "uchar": 1, "uint8": 1, "int": 4}
    if fmt == "ascii":
        rows = np.loadtxt(
            [line for line in data[header_end:].decode().splitlines() if line.strip()],
            ndmin=2,
        )[:count]
        names = [name for _, name in properties]
        return rows[:, [names.index("x"), names.index("y"), names.index("z")]]
    offset_map = {}
    stride = 0
    for type_name, name in properties:
        offset_map[name] = (stride, type_name)
        stride += sizes[type_name]
    body = data[header_end : header_end + count * stride]
    xyz = np.empty((count, 3), dtype=np.float64)
    for column, name in enumerate(("x", "y", "z")):
        start, type_name = offset_map[name]
        kind = {"float": "<f4", "float32": "<f4", "double": "<f8"}[type_name]
        xyz[:, column] = np.ndarray(
            (count,), dtype=kind, buffer=body, offset=start, strides=(stride,)
        )
    return xyz


def cloud_to_depth(
    cloud: np.ndarray, fov_x_deg: float, width: int, height: int
) -> np.ndarray:
    """Camera-frame cloud -> nearest-hit depth map (z metres; nan = empty).

    MoGe exports its cloud for GL viewers (y up, z back); auto-detect via the
    median z sign and flip to OpenCV convention (x right, y down, z forward).
    """
    points = np.asarray(cloud, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) and np.median(points[:, 2]) < 0:
        points = points * np.array([1.0, -1.0, -1.0])
    points = points[points[:, 2] > 1e-6]
    fx = (width / 2.0) / np.tan(np.radians(fov_x_deg) / 2.0)
    u = np.round(fx * points[:, 0] / points[:, 2] + width / 2.0).astype(int)
    v = np.round(fx * points[:, 1] / points[:, 2] + height / 2.0).astype(int)
    keep = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[keep], v[keep], points[keep, 2]
    depth = np.full((height, width), np.nan, dtype=np.float32)
    order = np.argsort(-z)  # nearest last wins the collision
    depth[v[order], u[order]] = z[order].astype(np.float32)
    return depth


def depth_to_points(
    depth: np.ndarray, mask: np.ndarray, fov_x_deg: float, stride: int = PIXEL_STRIDE
) -> np.ndarray:
    """Masked depth pixels -> camera-frame points (OpenCV convention)."""
    height, width = depth.shape
    fx = (width / 2.0) / np.tan(np.radians(fov_x_deg) / 2.0)
    vs, us = np.nonzero(mask)
    keep = (vs % stride == 0) & (us % stride == 0)
    vs, us = vs[keep], us[keep]
    zs = depth[vs, us]
    ok = np.isfinite(zs)
    vs, us, zs = vs[ok], us[ok], zs[ok].astype(np.float64)
    xs = (us - width / 2.0) * zs / fx
    ys = (vs - height / 2.0) * zs / fx
    return np.column_stack([xs, ys, zs])


def _moge_cache_path(cache_dir: Path, frame_id: int) -> Path:
    return cache_dir / f"f{frame_id:06d}.npz"


def moge_stage(
    frames_dir: Path, cache_dir: Path, frame_ids: list[int], live: bool
) -> tuple[int, list[int]]:
    """Fill the MoGe depth cache. Returns (live_calls_made, failed_frames)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        f for f in frame_ids if not _moge_cache_path(cache_dir, f).is_file()
    ]
    if not missing:
        return 0, []
    if not live:
        print(
            f"moge: {len(missing)} uncached calls planned "
            f"(~${len(missing) * MOGE_COST_PER_CALL_EST_USD:.2f} est); "
            "rerun with --live",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if len(missing) > MAX_MOGE_CALLS:
        raise SystemExit(
            f"moge budget cap: {len(missing)} planned > {MAX_MOGE_CALLS}"
        )
    import replicate

    calls_made = 0
    failures: list[int] = []
    for frame_id in missing:
        payload = _image_uri(frames_dir / f"f{frame_id:06d}.jpg")
        output = None
        for attempt in range(4):
            try:
                output = replicate.run(
                    MOGE_VERSION, input={"image": payload, "fp16": True}, wait=False
                )
                break
            except Exception as exc:
                if not any(
                    code in str(exc)
                    for code in ("429", "throttled", "500", "502", "503")
                ):
                    print(f"moge: frame {frame_id} failed: {exc}")
                    break
                time.sleep(8 * (attempt + 1))
        if output is None:
            failures.append(frame_id)
            continue
        calls_made += 1
        try:
            source = output["pointcloud_ply"]
            ply = source.read() if hasattr(source, "read") else Path(
                str(source)
            ).read_bytes()
            fov_x = float(output.get("fov_x_deg") or 0.0)
            cloud = parse_ply_xyz(ply)
            with Image.open(frames_dir / f"f{frame_id:06d}.jpg") as image:
                width, height = image.size
            depth = cloud_to_depth(cloud, fov_x, width, height)
        except Exception as exc:
            print(f"moge: frame {frame_id} postprocess failed: {exc}")
            failures.append(frame_id)
            continue
        np.savez_compressed(
            _moge_cache_path(cache_dir, frame_id),
            depth=depth.astype(np.float16),
            fov_x_deg=fov_x,
        )
    return calls_made, failures


# --------------------------------------------------------------- 3D estimate
def lift_frame(
    sam_dir: Path, moge_dir: Path, frame_id: int, cam_pos: np.ndarray,
    cam_rot: np.ndarray,
) -> np.ndarray | None:
    """Estimated arm points in the robot base frame for one frame, or None."""
    mask = load_arm_mask(sam_dir, frame_id)
    cache = _moge_cache_path(moge_dir, frame_id)
    if mask is None or not mask.any() or not cache.is_file():
        return None
    data = np.load(cache)
    depth = data["depth"].astype(np.float32)
    fov_x = float(data["fov_x_deg"])
    if fov_x <= 0:
        return None
    camera_points = depth_to_points(depth, erode_mask(mask), fov_x)
    if not len(camera_points):
        return None
    ranges = np.linalg.norm(camera_points, axis=1)
    camera_points = camera_points[ranges < MAX_RANGE_M]
    if not len(camera_points):
        return None
    return (cam_rot @ camera_points.T).T + cam_pos


# ------------------------------------------------------------------- states
def gt_state_labels(qd: np.ndarray, threshold: float = GT_ACTIVE_RADPS) -> list[str]:
    return [ACTIVE if float(np.linalg.norm(row)) > threshold else PARKED for row in qd]


def mask_change_ratio(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    union = (a | b).sum()
    if union == 0:
        return None
    return float((a ^ b).sum() / union)


# ------------------------------------------------------------------ envelopes
def hull_polygon(points_xy: np.ndarray) -> Polygon:
    hull = MultiPoint([tuple(p) for p in points_xy]).convex_hull
    if not isinstance(hull, Polygon):
        raise ValueError("degenerate hull (colinear points)")
    return hull


def planview_extremes(points_xy: np.ndarray) -> np.ndarray:
    """Convex-hull vertices of a 2D point set (fallback: the points).

    The zone polygons are convex, and the signed distance to a convex set is
    a convex function, so its max over a point set is attained at a hull
    vertex — evaluating rules on the vertices is exact and ~1000x cheaper.
    """
    if len(points_xy) < 4:
        return np.asarray(points_xy, dtype=float)
    hull = MultiPoint([tuple(p) for p in points_xy]).convex_hull
    if not isinstance(hull, Polygon):
        return np.asarray(points_xy, dtype=float)
    return np.array(hull.exterior.coords[:-1])


def envelope_iou(a: Polygon, b: Polygon) -> float:
    union = a.union(b).area
    return float(a.intersection(b).area / union) if union else 0.0


def boundary_discrepancy(estimated: Polygon, true: Polygon, step: float = 0.01) -> dict:
    """Boundary-to-boundary distances in metres, sampled every `step`.

    under_estimate_max/mean: how far the TRUE envelope pokes outside the
    ESTIMATED one — the amount a fence based on the camera alone would be
    too small. over_estimate_*: the opposite (estimate too big).
    """
    def _samples(poly: Polygon) -> list[Point]:
        boundary = poly.exterior
        n = max(8, int(boundary.length / step))
        return [boundary.interpolate(i / n, normalized=True) for i in range(n)]

    under = [
        point.distance(estimated.exterior) if not estimated.contains(point) else 0.0
        for point in _samples(true)
    ]
    over = [
        point.distance(true.exterior) if not true.contains(point) else 0.0
        for point in _samples(estimated)
    ]
    both = [point.distance(estimated.exterior) for point in _samples(true)]
    return {
        "under_estimate_max_m": round(max(under), 3),
        "under_estimate_mean_m": round(float(np.mean(under)), 3),
        "over_estimate_max_m": round(max(over), 3),
        "over_estimate_mean_m": round(float(np.mean(over)), 3),
        "boundary_mean_abs_m": round(float(np.mean(both)), 3),
        "boundary_max_abs_m": round(max(both), 3),
    }


# -------------------------------------------------------------------- rules
def banded_exit_verdict(max_signed_out_m: float, band: float = BAND_M) -> str:
    """Rule 'never exits region': value = max signed distance outside it."""
    if max_signed_out_m > band:
        return FAIL
    if max_signed_out_m < -band:
        return PASS
    return REVIEW


def signed_outside_zone(points_xy: np.ndarray, zone: Polygon) -> float:
    """Max over points of signed plan-view distance outside `zone`."""
    best = -np.inf
    exterior = zone.exterior
    for xy in points_xy:
        point = Point(xy)
        distance = point.distance(exterior)
        best = max(best, -distance if zone.contains(point) else distance)
    return float(best)


def confusion(
    estimated: list[str], reference: list[str]
) -> dict:
    pairs: dict[str, int] = {}
    for est, ref in zip(estimated, reference, strict=True):
        key = f"est_{est}|gt_{ref}"
        pairs[key] = pairs.get(key, 0) + 1
    observed = [
        (e, r) for e, r in zip(estimated, reference, strict=True) if e != NO_DATA
    ]
    matches = sum(e == r for e, r in observed)
    return {
        "agreement": round(matches / len(observed), 3) if observed else None,
        "n_observed": len(observed),
        "n_total": len(estimated),
        "pairs": dict(sorted(pairs.items())),
        "silent_misses": sum(
            1 for e, r in observed if e == PASS and r == FAIL
        ),
    }


# ------------------------------------------------------- FK-anchored scale
def _point_segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denominator = float(ab @ ab)
    if denominator < 1e-12:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(((points - a) @ ab) / denominator, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def skeleton_residual(points: np.ndarray, origins: np.ndarray) -> float:
    """Median distance from points to the FK skeleton segments, metres."""
    distances = np.full(len(points), np.inf)
    for a, b in zip(origins, origins[1:], strict=False):
        distances = np.minimum(distances, _point_segment_distances(points, a, b))
    return float(np.median(distances))


def fit_fk_scale(
    per_frame_points: dict[int, np.ndarray],
    origins_by_step: dict[int, np.ndarray],
    cam_pos: np.ndarray,
    max_points: int = 400,
) -> dict:
    """Golden-section fit of one range scale s (about the camera centre)
    minimising the median point-to-FK-skeleton residual. This is the
    'what would a scale reference buy' number — reported, never silently
    applied to the primary results."""
    frames = sorted(per_frame_points)
    subset = frames[:: max(1, len(frames) // 15)]
    clouds = []
    for frame_id in subset:
        points = per_frame_points[frame_id]
        if len(points) > max_points:
            points = points[:: len(points) // max_points]
        clouds.append((points, origins_by_step[frame_id]))

    def score(s: float) -> float:
        residuals = [
            skeleton_residual(cam_pos + s * (points - cam_pos), origins)
            for points, origins in clouds
        ]
        return float(np.median(residuals))

    low, high = 0.3, 3.0
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    c = high - golden * (high - low)
    d = low + golden * (high - low)
    for _ in range(40):
        if score(c) < score(d):
            high = d
        else:
            low = c
        c = high - golden * (high - low)
        d = low + golden * (high - low)
    best = (low + high) / 2.0
    return {
        "scale": round(best, 4),
        "residual_raw_m": round(score(1.0), 4),
        "residual_scaled_m": round(score(best), 4),
        "frames_used": len(clouds),
    }


# ---------------------------------------------------------------- rendering
def render_planview(
    true_hull: Polygon,
    est_hull: Polygon | None,
    est_hull_scaled: Polygon | None,
    zone: Polygon,
    keepout_x: float,
    cam_pos: np.ndarray,
    path: Path,
) -> None:
    geoms = [zone, true_hull] + [g for g in (est_hull, est_hull_scaled) if g]
    xs = [x for g in geoms for x, _ in g.exterior.coords] + [float(cam_pos[0]), 0.0]
    ys = [y for g in geoms for _, y in g.exterior.coords] + [float(cam_pos[1]), 0.0]
    margin = 0.15
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    size = 900
    scale = (size - 60) / max(x1 - x0, y1 - y0)

    def pixel(xy) -> tuple[float, float]:
        return 30 + (xy[0] - x0) * scale, size - 30 - (xy[1] - y0) * scale

    image = Image.new("RGB", (size, size), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    for polygon, color, label in (
        (zone, (240, 160, 40), "zone = TRUE envelope + 0.3 m"),
        (true_hull, (40, 140, 60), "TRUE sweep (FK)"),
        (est_hull, (200, 50, 50), "estimated sweep (SAM+MoGe raw)"),
        (est_hull_scaled, (60, 90, 200), "estimated, FK-scale-calibrated"),
    ):
        if polygon is None:
            continue
        draw.polygon([pixel(p) for p in polygon.exterior.coords], outline=color, width=3)
    # keep-out line x = keepout_x (vertical in plan view)
    draw.line([pixel((keepout_x, y0)), pixel((keepout_x, y1))], fill=(140, 60, 180), width=2)
    draw.text(pixel((keepout_x, y1 - 0.05)), "keep-out", fill=(140, 60, 180))
    px, py = pixel((float(cam_pos[0]), float(cam_pos[1])))
    draw.regular_polygon((px, py, 7), 3, fill=(0, 0, 0))
    draw.text((px + 8, py - 6), "camera", fill=(0, 0, 0))
    bx, by = pixel((0.0, 0.0))
    draw.ellipse([bx - 5, by - 5, bx + 5, by + 5], outline=(0, 0, 0), width=2)
    draw.text((bx + 8, by - 6), "robot base", fill=(0, 0, 0))
    legend = [
        ((240, 160, 40), "zone = TRUE envelope + 0.3 m"),
        ((40, 140, 60), "TRUE sweep envelope (FK, exact)"),
        ((200, 50, 50), "estimated sweep (SAM+MoGe, raw metric)"),
        ((60, 90, 200), "estimated, FK-scale-calibrated"),
        ((140, 60, 180), f"keep-out line x = {keepout_x:.3f} m"),
    ]
    for row, (color, text) in enumerate(legend):
        draw.rectangle([20, 16 + 18 * row, 32, 26 + 18 * row], fill=color)
        draw.text((38, 14 + 18 * row), text, fill=(40, 40, 40))
    bar0 = pixel((x1 - margin - 0.5, y0 + margin / 2))
    bar1 = pixel((x1 - margin, y0 + margin / 2))
    draw.line([bar0, bar1], fill=(0, 0, 0), width=3)
    draw.text((bar0[0], bar0[1] - 16), "0.5 m", fill=(0, 0, 0))
    image.save(path)


_MASK_TINT = np.array((230, 60, 40), dtype=np.uint16)


def render_overlay(
    frames_dir: Path,
    sam_dir: Path,
    frame_ids: list[int],
    t_s: np.ndarray,
    est_states: dict[int, str],
    gt_states: list[str],
    rc_est: dict[int, str],
    rc_gt: list[str],
    gif_path: Path,
    strip_path: Path,
    strip_frames: list[int],
) -> None:
    rendered: dict[int, Image.Image] = {}
    for frame_id in frame_ids:
        with Image.open(frames_dir / f"f{frame_id:06d}.jpg") as source:
            frame = source.convert("RGB")
        mask = load_arm_mask(sam_dir, frame_id)
        if mask is not None:
            pixels = np.asarray(frame, dtype=np.uint16)
            pixels[mask] = (pixels[mask] * 3 + _MASK_TINT * 2) // 5
            frame = Image.fromarray(pixels.astype(np.uint8))
        draw = ImageDraw.Draw(frame)
        est = est_states.get(frame_id, NO_DATA)
        banner = (
            f"t={t_s[frame_id]:.2f}s step {frame_id}  "
            f"state est:{est} gt:{gt_states[frame_id]}  "
            f"zone est:{rc_est.get(frame_id, NO_DATA)} gt:{rc_gt[frame_id]}"
        )
        draw.rectangle([0, 0, frame.width, 26], fill=(0, 0, 0))
        draw.text((8, 6), banner, fill=(255, 255, 255))
        rendered[frame_id] = frame
    small = [
        rendered[f].resize((720, round(720 * rendered[f].height / rendered[f].width)))
        for f in frame_ids
    ]
    small[0].save(
        gif_path, save_all=True, append_images=small[1:], duration=180, loop=0
    )
    columns = [rendered[f] for f in strip_frames if f in rendered]
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


# ------------------------------------------------- FANUC low-res cross-check
FANUC_ROOT = Path(
    "/Users/adam/Desktop/Tesla/ehs-spatial/outputs/datasets/robotcell/fanuc_berkeley"
)
FANUC_EPISODE = 0
FANUC_FPS = 10.0
FANUC_STRIDE = 8  # every 0.8 s


def run_fanuc(out: Path, live: bool) -> dict:
    """Arm-state axis only on one 224x224 Berkeley FANUC episode.

    No camera calibration exists for this dataset, so the envelope and zone
    axes cannot run; this measures (a) whether SAM masks survive 224 px and
    (b) state agreement vs finite-difference joint velocities.
    """
    import cv2
    import pyarrow.parquet as pq

    episodes = pq.read_table(
        FANUC_ROOT / "meta/episodes/chunk-000/file-000.parquet",
        columns=[
            "episode_index",
            "dataset_from_index",
            "dataset_to_index",
            "videos/observation.images.image/from_timestamp",
            "length",
        ],
    ).to_pylist()
    row = next(r for r in episodes if r["episode_index"] == FANUC_EPISODE)
    states = pq.read_table(
        FANUC_ROOT / "data/chunk-000/file-000.parquet",
        columns=["observation.state"],
    )["observation.state"].to_pylist()
    q = np.array(states[row["dataset_from_index"] : row["dataset_to_index"]])[:, :7]
    # central finite-difference joint velocity (rad/s) — the parquet has no
    # recorded velocities, unlike the DROID h5
    qd = np.gradient(q, 1.0 / FANUC_FPS, axis=0)
    gt_states = gt_state_labels(qd)

    start_frame = int(round(row["videos/observation.images.image/from_timestamp"] * FANUC_FPS))
    local_ids = list(range(0, row["length"], FANUC_STRIDE))
    frames_dir = out / "fanuc_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        i for i in local_ids if not (frames_dir / f"f{i:06d}.jpg").is_file()
    ]
    if missing:
        capture = cv2.VideoCapture(
            str(FANUC_ROOT / "videos/observation.images.image/chunk-000/file-000.mp4")
        )
        wanted = {start_frame + i: i for i in missing}
        index = 0
        while wanted and capture.grab():
            if index in wanted:
                ok, frame = capture.retrieve()
                if not ok:
                    raise RuntimeError(f"failed to decode video frame {index}")
                cv2.imwrite(
                    str(frames_dir / f"f{wanted[index]:06d}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                del wanted[index]
            index += 1
        capture.release()
        if wanted:
            raise RuntimeError(f"video ended before frames {sorted(wanted)}")

    sam_calls, sam_failures = sam_stage(
        frames_dir, out / "fanuc_sam_cache", local_ids, live
    )
    masks = {i: load_arm_mask(out / "fanuc_sam_cache", i) for i in local_ids}
    est_states: dict[int, str] = {}
    change_values: dict[int, float] = {}
    for previous, current in zip(local_ids, local_ids[1:]):
        ratio = mask_change_ratio(masks[previous], masks[current])
        if ratio is None:
            est_states[current] = NO_DATA
        else:
            change_values[current] = round(ratio, 4)
            est_states[current] = ACTIVE if ratio > EST_ACTIVE_MASK_CHANGE else PARKED
    pairs = [(est_states.get(i, NO_DATA), gt_states[i]) for i in local_ids[1:]]
    state_confusion = confusion(*map(list, zip(*pairs)))
    mask_pixels = [int(m.sum()) for m in masks.values() if m is not None]
    report = {
        "episode_index": FANUC_EPISODE,
        "n_steps": row["length"],
        "sampled_steps": local_ids,
        "sample_dt_s": FANUC_STRIDE / FANUC_FPS,
        "frames_with_mask": sum(1 for m in masks.values() if m is not None),
        "frames_sampled": len(local_ids),
        "mask_pixels_min_median_max": (
            [min(mask_pixels), int(np.median(mask_pixels)), max(mask_pixels)]
            if mask_pixels
            else None
        ),
        "state_timeline": {
            "estimated": {str(i): est_states.get(i, NO_DATA) for i in local_ids[1:]},
            "gt": {str(i): gt_states[i] for i in local_ids[1:]},
            "mask_change_values": {str(i): v for i, v in change_values.items()},
            "confusion": state_confusion,
        },
        "spend": {
            "sam_calls_this_run": sam_calls,
            "sam_cost_this_run_usd": round(sam_calls * SAM_COST_PER_CALL_USD, 2),
            "sam_failed": sam_failures,
        },
    }
    (out / "fanuc_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"fanuc report: {(out / 'fanuc_report.json').resolve()}")
    print(
        f"fanuc R-A agreement: {state_confusion['agreement']} "
        f"({report['frames_with_mask']}/{report['frames_sampled']} frames masked)"
    )
    return report


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", default=str(DATA_ROOT / EPISODE))
    parser.add_argument("--camera", default=CAMERA_SERIAL)
    parser.add_argument(
        "--out", default="/Users/adam/Desktop/Tesla/ehs-spatial/outputs/arm_poc_v1"
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--fanuc", action="store_true",
        help="run only the Berkeley FANUC 224px cross-check",
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.fanuc:
        run_fanuc(out, args.live)
        return 0
    episode_dir = Path(args.episode_dir)

    data = load_episode(episode_dir, args.camera)
    q, qd, t_s = data["q"], data["qd"], data["t_s"]
    cam_pos, cam_rot = data["cam_pos"], data["cam_rot"]
    fk_error = validate_fk(q, data["cart"])
    print(f"FK vs recorded cartesian: max error {fk_error * 1000:.3f} mm")

    n_steps = len(q)
    n_video_frames = n_steps - 1  # the MP4s carry one frame fewer than the h5
    frame_ids = list(range(0, n_video_frames, SAMPLE_STRIDE))[:MAX_SAMPLES]

    # ---- ground truth layer
    origins_by_step = {i: fk_origins(q[i]) for i in range(n_steps)}
    dense_by_step = {i: densify_skeleton(origins_by_step[i]) for i in range(n_steps)}
    all_fk = np.vstack([dense_by_step[i] for i in range(n_steps)])
    true_hull = hull_polygon(all_fk[:, :2])
    # round join = a true 0.3 m offset everywhere (mitre overshoots at the
    # hull's sharp corners, silently widening the zone)
    zone = true_hull.buffer(ZONE_MARGIN_M, join_style="round")
    keepout_x = float(all_fk[:, 0].max()) - KEEPOUT_INSET_M
    gt_states = gt_state_labels(qd)

    # ---- estimated layer
    extract_frames(
        episode_dir / "recordings" / "MP4" / f"{args.camera}.mp4",
        frame_ids,
        out / "frames",
    )
    sam_calls, sam_failures = sam_stage(
        out / "frames", out / "sam_cache", frame_ids, args.live
    )
    moge_calls, moge_failures = moge_stage(
        out / "frames", out / "moge_cache", frame_ids, args.live
    )

    masks = {f: load_arm_mask(out / "sam_cache", f) for f in frame_ids}
    clouds: dict[int, np.ndarray] = {}
    for frame_id in frame_ids:
        points = lift_frame(
            out / "sam_cache", out / "moge_cache", frame_id, cam_pos, cam_rot
        )
        if points is not None:
            clouds[frame_id] = points

    # ---- R-A arm state
    est_states: dict[int, str] = {}
    change_values: dict[int, float] = {}
    for previous, current in zip(frame_ids, frame_ids[1:]):
        ratio = mask_change_ratio(masks[previous], masks[current])
        if ratio is None:
            est_states[current] = NO_DATA
        else:
            change_values[current] = round(ratio, 4)
            est_states[current] = (
                ACTIVE if ratio > EST_ACTIVE_MASK_CHANGE else PARKED
            )
    state_pairs = [
        (est_states.get(f, NO_DATA), gt_states[f]) for f in frame_ids[1:]
    ]
    state_confusion = confusion(*map(list, zip(*state_pairs)))
    state_disagreements = [
        {
            "step": f,
            "estimated": est_states.get(f, NO_DATA),
            "gt": gt_states[f],
            "mask_change": change_values.get(f),
            "qd_norm": round(float(np.linalg.norm(qd[f])), 4),
        }
        for f in frame_ids[1:]
        if est_states.get(f, NO_DATA) != gt_states[f]
    ]

    # ---- R-B sweep envelope
    est_hull = est_hull_scaled = None
    envelope_report: dict = {"status": "no estimated points"}
    scale_fit: dict = {}
    extremes = {
        f: planview_extremes(points[:, :2]) for f, points in clouds.items()
    }
    if clouds:
        all_est = np.vstack(list(extremes.values()))
        est_hull = hull_polygon(all_est)
        envelope_report = {
            "iou_planview": round(envelope_iou(est_hull, true_hull), 3),
            "true_area_m2": round(true_hull.area, 3),
            "estimated_area_m2": round(est_hull.area, 3),
            "discrepancy_m": boundary_discrepancy(est_hull, true_hull),
        }
        scale_fit = fit_fk_scale(clouds, origins_by_step, cam_pos)
        scaled_extremes = {
            f: cam_pos[:2] + scale_fit["scale"] * (points - cam_pos[:2])
            for f, points in extremes.items()
        }
        est_hull_scaled = hull_polygon(np.vstack(list(scaled_extremes.values())))
        envelope_report["fk_scale_calibrated"] = {
            "scale": scale_fit["scale"],
            "iou_planview": round(envelope_iou(est_hull_scaled, true_hull), 3),
            "estimated_area_m2": round(est_hull_scaled.area, 3),
            "discrepancy_m": boundary_discrepancy(est_hull_scaled, true_hull),
        }

    # ---- R-C banded zone rules (est banded, GT exact)
    rc_est: dict[int, str] = {}
    rc_values: dict[int, float] = {}
    keepout_est: dict[int, str] = {}
    keepout_values: dict[int, float] = {}
    for frame_id in frame_ids:
        points = extremes.get(frame_id)
        if points is None:
            rc_est[frame_id] = NO_DATA
            keepout_est[frame_id] = NO_DATA
            continue
        out_zone = signed_outside_zone(points, zone)
        rc_values[frame_id] = round(out_zone, 3)
        rc_est[frame_id] = banded_exit_verdict(out_zone)
        out_keep = float(points[:, 0].max() - keepout_x)
        keepout_values[frame_id] = round(out_keep, 3)
        keepout_est[frame_id] = banded_exit_verdict(out_keep)
    rc_gt = [
        FAIL if signed_outside_zone(dense_by_step[i][:, :2], zone) > 0 else PASS
        for i in range(n_steps)
    ]
    keepout_gt = [
        FAIL if float(dense_by_step[i][:, 0].max() - keepout_x) > 0 else PASS
        for i in range(n_steps)
    ]
    rc_confusion = confusion(
        [rc_est[f] for f in frame_ids], [rc_gt[f] for f in frame_ids]
    )
    keepout_confusion = confusion(
        [keepout_est[f] for f in frame_ids], [keepout_gt[f] for f in frame_ids]
    )

    # ---- rendering
    if est_hull is not None:
        render_planview(
            true_hull, est_hull, est_hull_scaled, zone, keepout_x, cam_pos,
            out / "envelope_planview.png",
        )
    strip_frames = [frame_ids[i] for i in (0, len(frame_ids) // 3, 2 * len(frame_ids) // 3, len(frame_ids) - 1)]
    render_overlay(
        out / "frames", out / "sam_cache", frame_ids, t_s,
        est_states, gt_states, rc_est, rc_gt,
        out / "overlay.gif", out / "overlay_strip.png", strip_frames,
    )

    spend = {
        "sam_calls_this_run": sam_calls,
        "sam_cost_this_run_usd": round(sam_calls * SAM_COST_PER_CALL_USD, 2),
        "sam_failed": sam_failures,
        "moge_calls_this_run": moge_calls,
        "moge_cost_this_run_est_usd": round(
            moge_calls * MOGE_COST_PER_CALL_EST_USD, 2
        ),
        "moge_failed_frames": moge_failures,
    }
    report = {
        "episode": episode_dir.name,
        "camera_serial": args.camera,
        "n_steps": n_steps,
        "sampled_steps": frame_ids,
        "fk_max_error_vs_recorded_cartesian_mm": round(fk_error * 1000, 4),
        "thresholds": {
            "gt_active_radps": GT_ACTIVE_RADPS,
            "est_active_mask_change": EST_ACTIVE_MASK_CHANGE,
            "band_m": BAND_M,
            "zone_margin_m": ZONE_MARGIN_M,
            "keepout_inset_m": KEEPOUT_INSET_M,
            "mask_erode_px": MASK_ERODE_PX,
            "max_range_m": MAX_RANGE_M,
            "fk_dense_step_m": FK_DENSE_STEP_M,
        },
        "state_timeline": {
            "estimated": {str(f): est_states.get(f, NO_DATA) for f in frame_ids[1:]},
            "gt": {str(f): gt_states[f] for f in frame_ids[1:]},
            "mask_change_values": {str(f): v for f, v in change_values.items()},
            "confusion": state_confusion,
            "disagreements": state_disagreements,
        },
        "envelope": envelope_report,
        "fk_scale_fit": scale_fit,
        "zone_rule": {
            "zone_wkt": zone.wkt,
            "estimated": {str(f): rc_est[f] for f in frame_ids},
            "gt": {str(f): rc_gt[f] for f in frame_ids},
            "values_max_signed_out_m": {str(f): v for f, v in rc_values.items()},
            "confusion": rc_confusion,
        },
        "keepout_rule": {
            "keepout_x_m": round(keepout_x, 4),
            "estimated": {str(f): keepout_est[f] for f in frame_ids},
            "gt": {str(f): keepout_gt[f] for f in frame_ids},
            "values_max_signed_out_m": {str(f): v for f, v in keepout_values.items()},
            "confusion": keepout_confusion,
        },
        "frames_with_cloud": len(clouds),
        "spend": spend,
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report: {(out / 'report.json').resolve()}")
    print(
        f"R-A state agreement: {state_confusion['agreement']} "
        f"({len(state_disagreements)} disagreements)"
    )
    if "iou_planview" in envelope_report:
        print(
            f"R-B envelope IoU: {envelope_report['iou_planview']} "
            f"(FK-scale-calibrated: "
            f"{envelope_report['fk_scale_calibrated']['iou_planview']})"
        )
    print(
        f"R-C zone agreement: {rc_confusion['agreement']} "
        f"silent misses: {rc_confusion['silent_misses']} | "
        f"keep-out agreement: {keepout_confusion['agreement']} "
        f"silent misses: {keepout_confusion['silent_misses']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
