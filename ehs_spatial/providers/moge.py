"""MoGe-2 automatic scale anchor.

MapAnything's mono geometry has the right shape but an unreliable absolute
scale; MoGe-2 (MIT) is an independent metric model. The ratio of their
median camera-frame ranges over the same view is a single scalar that
re-gauges MapAnything to metres with no operator input. Validated against
laser/synthetic ground truth on three packs
(docs/reviews/2026-08-24-moge-auto-anchor.md): warehouse 9.0% median
relative error vs the 8.5% operator-anchor upper bound; ETH3D 14.7 cm vs
14.4; Redwood 22.9 cm vs 27.5 (the auto anchor WINS there — the measured
camera height was the weaker number).

Per-frame responses cache under <geometry_dir>/moge/, so re-runs are free.
Every failure path returns None: the scale chain falls back to the
operator's camera height, never crashes a run.
"""

import base64
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..contracts import GeometryFrame

MOGE_VERSION = (
    "jasonod888/moge2:"
    "daa7a9329b3d6bb513f3a20def9451b6e3cf234afe2bb60ba38f96ecd33c81f7"
)
MOGE3_MODAL_APP = "moge3-inference"
_TRANSIENT = ("429", "throttled", "500", "502", "503", "timeout", "connection")


def _replicate_runner(model_identifier: str, *, input: dict[str, object]) -> object:
    import replicate

    return replicate.run(model_identifier, input=input, wait=False)


def _modal_runner(model_identifier: str, *, input: dict[str, object]) -> object:
    """MoGe-3 on our own Modal endpoint (A/B: depth-edge width -40% vs
    MoGe-2 at identical metric scale, ~6x faster than Replicate). Returns
    the same shape the Replicate consumers read: 'pointcloud_ply' and
    'intrinsics_json' as file-like objects."""
    import io

    import modal

    payload = str(input.get("image", ""))
    prefix, _, encoded = payload.partition(",")
    image_bytes = base64.b64decode(encoded if _ else prefix)
    MoGe3 = modal.Cls.from_name(MOGE3_MODAL_APP, "MoGe3")
    result = MoGe3().infer.remote(image_bytes)
    return {
        "pointcloud_ply": io.BytesIO(result["ply"]),
        "intrinsics_json": io.BytesIO(
            json.dumps({"intrinsics": result["intrinsics"]}).encode("utf-8")
        ),
        "fov_x_deg": result.get("fov_x_deg"),
    }


def _default_runner(model_identifier: str, *, input: dict[str, object]) -> object:
    from ..backends import http_json, service_backend

    backend = service_backend("MOGE_BACKEND", "modal")
    if backend == "replicate":
        return _replicate_runner(model_identifier, input=input)
    if backend == "http":
        import io
        import os

        payload = str(input.get("image", ""))
        prefix, _, encoded = payload.partition(",")
        response = http_json(
            os.environ["MOGE_HTTP_URL"],
            {"image_b64": encoded if _ else prefix},
        )
        return {
            "pointcloud_ply": io.BytesIO(
                base64.b64decode(response["ply_b64"])
            ),
            "intrinsics_json": io.BytesIO(
                json.dumps({"intrinsics": response["intrinsics"]}).encode()
            ),
            "fov_x_deg": response.get("fov_x_deg"),
        }
    return _modal_runner(model_identifier, input=input)


class ScaleAnchor:
    def __init__(self, scale: float, confidence: float, ratios: list[float]):
        self.scale = scale
        self.confidence = confidence
        self.ratios = ratios


def _mapanything_median_range(frame: GeometryFrame) -> float | None:
    """Median camera-frame range of MapAnything's cloud for one frame."""
    points = np.load(frame.pts3d_path)
    valid = np.load(frame.valid_mask_path).astype(bool)
    finite = valid & np.isfinite(points).all(axis=2)
    world = points[finite]
    if len(world) < 100:
        return None
    camera_to_world = np.asarray(frame.camera_to_world, dtype=float)
    camera = (
        np.linalg.inv(camera_to_world) @ np.c_[world, np.ones(len(world))].T
    )[:3].T
    median = float(np.median(np.linalg.norm(camera, axis=1)))
    return median if np.isfinite(median) and median > 0 else None


class MoGeAnchorAdapter:
    def __init__(self, runner: Callable[..., object] | None = None) -> None:
        self.runner = runner or _default_runner

    def _moge_median_range(
        self, image_path: str, cache_path: Path
    ) -> float | None:
        if cache_path.exists():
            return json.loads(cache_path.read_text()).get("moge_median_range_m")
        payload = "data:image/png;base64," + base64.b64encode(
            Path(image_path).read_bytes()
        ).decode("ascii")
        output = None
        for attempt in range(4):
            try:
                output = self.runner(
                    MOGE_VERSION, input={"image": payload, "fp16": True}
                )
                break
            except Exception as exc:
                if not any(code in str(exc) for code in _TRANSIENT):
                    return None
                time.sleep(8 * (attempt + 1))
        if output is None:
            return None
        try:
            import open3d as o3d

            cloud_source = output["pointcloud_ply"]
            with tempfile.NamedTemporaryFile(suffix=".ply") as handle:
                handle.write(
                    cloud_source.read()
                    if hasattr(cloud_source, "read")
                    else Path(str(cloud_source)).read_bytes()
                )
                handle.flush()
                cloud = np.asarray(o3d.io.read_point_cloud(handle.name).points)
        except Exception:
            return None
        if not len(cloud):
            return None
        median = float(np.median(np.linalg.norm(cloud, axis=1)))
        if not (np.isfinite(median) and median > 0):
            return None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"moge_median_range_m": median, "points": int(len(cloud))}
            )
            + "\n"
        )
        return median

    def anchor_scale(
        self, frames: list[GeometryFrame], geometry_dir: str | Path
    ) -> ScaleAnchor | None:
        """One scalar for the whole capture: median of per-frame ratios.
        Returns None on any failure — callers fall back, never crash."""
        cache_dir = Path(geometry_dir) / "moge"

        def _frame_ratio(frame: GeometryFrame) -> float | None:
            native = _mapanything_median_range(frame)
            if native is None:
                return None
            moge = self._moge_median_range(
                frame.canonical_image_path,
                cache_dir / f"{frame.frame_id}.json",
            )
            if moge is None:
                return None
            return moge / native

        # per-frame inferences are independent remote calls; run them
        # concurrently (a 3-view capture spent ~1 serial minute here)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(4, len(frames) or 1)) as pool:
            ratios = [r for r in pool.map(_frame_ratio, frames) if r is not None]
        if not ratios:
            return None
        values = np.asarray(ratios)
        scale = float(np.median(values))
        if not (np.isfinite(scale) and scale > 0):
            return None
        spread = float(np.median(np.abs(values - scale)))
        # One frame gives zero spread by construction, which says nothing
        # about quality — cap, don't fabricate, certainty.
        if len(ratios) < 2:
            confidence = 0.5
        else:
            confidence = float(np.clip(1.0 - spread / scale, 0.0, 1.0))
        return ScaleAnchor(scale, confidence, [round(r, 4) for r in ratios])


__all__ = ["MOGE_VERSION", "MoGeAnchorAdapter", "ScaleAnchor"]
