"""Offline tests for the uncalibrated video tier (ehs_spatial/video.py).

No network: SAM, MoGe, tracking, and frame extraction are all injected,
mirroring how the photo pipeline fakes its adapters.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ehs_spatial.artifacts import ArtifactStore
from ehs_spatial.contracts import RunManifest
from ehs_spatial.providers.base import ProviderError
from ehs_spatial import video as video_module
from ehs_spatial.video import (
    FloorCamera,
    banded_verdict,
    fit_floor_from_keyframes,
    fit_pinhole_from_grid,
    judge,
    lift_tracks,
    ransac_plane,
    run_video_assessment,
    sam_stage,
    sample_schedule,
    video_paths,
    worst_verdict,
    zone_verdict,
)
from shapely import wkt as shapely_wkt


# ---------------------------------------------------------------- sampling
def test_sample_schedule_steps_and_caps():
    assert sample_schedule(300, 30.0, 1.0, 60) == list(range(0, 300, 30))
    assert sample_schedule(9000, 30.0, 0.5, 40) == list(range(0, 2400, 60))
    assert sample_schedule(100, 30.0, 2.0, 5) == [0, 15, 30, 45, 60]
    # Faster than native degrades to every frame, never interpolation.
    assert sample_schedule(4, 30.0, 120.0, 10) == [0, 1, 2, 3]
    assert sample_schedule(0, 30.0, 1.0, 60) == []
    assert sample_schedule(300, 30.0, 1.0, 0) == []


# ------------------------------------------------------------- camera model
_FX, _FY, _CX, _CY = 80.0, 80.0, 32.0, 24.0
_W, _H = 64, 48


def _synthetic_grid(floor_y: float = 1.5, wall_z: float = 10.0) -> np.ndarray:
    """Organized point map of a floor (y = floor_y, camera y points down)
    with a far wall above the horizon, from a known pinhole."""
    grid = np.zeros((_H, _W, 3))
    for v in range(_H):
        for u in range(_W):
            direction = np.array(
                [(u - _CX) / _FX, (v - _CY) / _FY, 1.0]
            )
            if direction[1] > 1e-3:  # below the horizon: floor
                t = floor_y / direction[1]
            else:  # wall at wall_z
                t = wall_z
            grid[v, u] = direction * t
    return grid


def test_fit_pinhole_recovers_known_intrinsics():
    fitted = fit_pinhole_from_grid(_synthetic_grid())
    assert fitted is not None
    assert fitted["fx"] == pytest.approx(_FX, abs=0.5)
    assert fitted["fy"] == pytest.approx(_FY, abs=0.5)
    assert fitted["cx"] == pytest.approx(_CX, abs=0.5)
    assert fitted["cy"] == pytest.approx(_CY, abs=0.5)


def test_fit_pinhole_rejects_degenerate_grids():
    assert fit_pinhole_from_grid(np.zeros((4, 4, 3))) is None
    flat = np.ones((_H, _W, 3))  # every ray identical: no spread to fit
    assert fit_pinhole_from_grid(flat) is None


def test_ransac_plane_finds_floor_among_outliers():
    rng = np.random.default_rng(3)
    floor = np.c_[
        rng.uniform(-5, 5, 400), np.full(400, 2.0), rng.uniform(1, 20, 400)
    ]
    floor += rng.normal(0, 0.01, floor.shape)
    outliers = rng.uniform(-5, 5, (100, 3))
    result = ransac_plane(np.vstack([floor, outliers]))
    assert result is not None
    normal, d, inlier_fraction = result
    assert abs(normal[1]) > 0.99
    assert d == pytest.approx(2.0, abs=0.05)
    assert d > 0  # camera (origin) on the positive side
    assert inlier_fraction > 0.6


def test_ransac_plane_needs_enough_points():
    assert ransac_plane(np.zeros((10, 3))) is None


def test_floor_fit_falls_back_to_moge_intrinsics_on_non_grid_cloud(tmp_path):
    """Real MoGe clouds drop invalid pixels: the grid path must fail and the
    cached-intrinsics fallback must still recover the floor."""
    import json as json_module

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    moge_dir = tmp_path / "moge"
    moge_dir.mkdir()
    Image.new("RGB", (_W, _H)).save(frames_dir / "f000000.jpg")

    grid = _synthetic_grid()
    points = grid.reshape(-1, 3)
    points = np.delete(points, np.arange(0, 37), axis=0)  # 37 is prime: no grid

    def runner(model_identifier, *, input):
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        ply_path = tmp_path / "nongrid.ply"
        o3d.io.write_point_cloud(str(ply_path), cloud, write_ascii=True)
        intr_path = tmp_path / "nongrid_intrinsics.json"
        intr_path.write_text(
            json_module.dumps(
                {
                    "intrinsics": [
                        [_FX / _W, 0.0, _CX / _W],
                        [0.0, _FY / _H, _CY / _H],
                        [0.0, 0.0, 1.0],
                    ]
                }
            )
        )
        return {
            "pointcloud_ply": str(ply_path),
            "intrinsics_json": str(intr_path),
        }

    camera, summary = fit_floor_from_keyframes(frames_dir, moge_dir, [0], runner)
    assert summary["fitted"] is True
    assert camera is not None
    assert camera.camera_height_m == pytest.approx(1.5, abs=0.05)


def _camera(height: float = 2.0) -> FloorCamera:
    return FloorCamera(
        normal=np.array([0.0, -1.0, 0.0]),
        d=height,
        intrinsics={"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
        inlier_fraction=0.9,
    )


def test_lift_pixel_hits_known_ground_points():
    camera = _camera()
    assert camera.camera_height_m == 2.0
    # Straight ahead, 5 m out: pixel (cx, cy + fy * height/5).
    assert camera.lift_pixel(320.0, 440.0) == pytest.approx((0.0, 5.0))
    # 1 m to the camera's right, 4 m out (lateral axis sign is consistent
    # within a run; the exact handedness is unobservable without calibration).
    x, y = camera.lift_pixel(445.0, 490.0)
    assert abs(x) == pytest.approx(1.0)
    assert y == pytest.approx(4.0)


def test_lift_pixel_rejects_rays_above_the_horizon():
    assert _camera().lift_pixel(320.0, 100.0) is None


def test_lift_tracks_bottom_center_and_vehicle_edges():
    camera = _camera()
    tracks = {
        "person-1": [(0, (310.0, 400.0, 330.0, 440.0))],
        "forklift-1": [(0, (300.0, 400.0, 340.0, 440.0))],
    }
    lifted = lift_tracks(tracks, camera)
    assert lifted["person-1"][0]["xy"] == pytest.approx((0.0, 5.0))
    assert "edge" not in lifted["person-1"][0]
    assert "edge" in lifted["forklift-1"][0]
    left, right = lifted["forklift-1"][0]["edge"]
    assert left != right


# ------------------------------------------------------------------ judging
def test_banded_verdicts_both_directions():
    band = 0.35
    assert banded_verdict(1.0, 2.0, band, fail_low=True) == "FAIL"
    assert banded_verdict(2.1, 2.0, band, fail_low=True) == "NEEDS_REVIEW"
    assert banded_verdict(2.5, 2.0, band, fail_low=True) == "PASS"
    assert banded_verdict(2.5, 1.5, 0.7, fail_low=False) == "FAIL"
    assert banded_verdict(1.6, 1.5, 0.7, fail_low=False) == "NEEDS_REVIEW"
    assert banded_verdict(0.5, 1.5, 0.7, fail_low=False) == "PASS"


def test_zone_verdict_banded():
    zone = shapely_wkt.loads("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))")
    assert zone_verdict(zone, (2.0, 2.0), 0.35) == "FAIL"
    assert zone_verdict(zone, (0.1, 2.0), 0.35) == "NEEDS_REVIEW"
    assert zone_verdict(zone, (10.0, 10.0), 0.35) == "PASS"


def _lifted_fixture():
    return {
        "person-1": {
            0: {"xy": (0.0, 0.0)},
            30: {"xy": (0.0, 4.0)},  # 4 m in one second: FAIL speed
            60: {"xy": (0.0, 4.2)},
        },
        "car-1": {
            0: {"xy": (0.0, 1.0)},  # 1 m from the person: FAIL separation
            30: {"xy": (0.0, 1.0)},
        },
    }


def test_judge_timelines_on_synthetic_tracks():
    frame_ids = [0, 30, 60]
    zone = shapely_wkt.loads("POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))")
    judged = judge(_lifted_fixture(), zone, frame_ids, step_seconds=1.0)
    assert judged["speed_band_mps"] == pytest.approx(0.7)
    assert judged["R1_zone"][0] == "FAIL"  # person inside the zone
    assert judged["R1_zone"][60] == "PASS"
    assert judged["R2_min_distance"][0] == "FAIL"
    assert judged["R2_min_distance"][60] == "NO_DATA"  # vehicle vanished
    assert judged["R3_speed"][0] == "NO_DATA"  # no previous sample
    assert judged["R3_speed"][30] == "FAIL"
    assert judged["R3_speed"][60] == "PASS"
    assert judged["R2_values_m"][0] == pytest.approx(1.0)
    assert judged["R3_values_mps"][30] == pytest.approx(4.0)


def test_judge_without_zone_or_person_abstains():
    frame_ids = [0, 30]
    judged = judge(_lifted_fixture(), None, frame_ids, step_seconds=1.0)
    assert set(judged["R1_zone"].values()) == {"NO_DATA"}
    lifted = {"car-1": {0: {"xy": (0.0, 1.0)}, 30: {"xy": (0.0, 1.0)}}}
    zone = shapely_wkt.loads("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
    judged = judge(lifted, zone, frame_ids, step_seconds=1.0)
    assert set(judged["R1_zone"].values()) == {"PASS"}  # nobody in the zone
    assert set(judged["R2_min_distance"].values()) == {"NO_DATA"}
    assert set(judged["R3_speed"].values()) == {"NO_DATA"}


def test_worst_verdict_orders_by_severity():
    assert worst_verdict({}) == "NO_DATA"
    assert worst_verdict({0: "PASS", 1: "NEEDS_REVIEW"}) == "NEEDS_REVIEW"
    assert worst_verdict({0: "PASS", 1: "FAIL", 2: "NO_DATA"}) == "FAIL"


# ---------------------------------------------------------------- SAM stage
def _encode_mask(mask: np.ndarray) -> str:
    """Uncompressed COCO RLE counts (column-major), the simplest format
    decode_coco_rle accepts alongside height/width."""
    flat = mask.flatten(order="F")
    counts, value, run = [], 0, 0
    for pixel in flat:
        if pixel == value:
            run += 1
        else:
            counts.append(run)
            value, run = int(pixel), 1
    counts.append(run)
    return json.dumps(counts)


def _rect_mask(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros((_H, _W), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def test_sam_stage_budget_cap_blocks_before_spending(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="budget cap"):
        sam_stage(
            tmp_path,
            tmp_path / "cache",
            list(range(80)),
            ("person", "car"),
            lambda *a, **k: calls.append(1),
        )
    assert not calls


def test_sam_stage_caches_and_reruns_free(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for frame_id in (0, 30):
        Image.new("RGB", (_W, _H), "gray").save(
            frames_dir / f"f{frame_id:06d}.jpg"
        )
    calls = []

    def subscriber(endpoint, *, arguments):
        calls.append(arguments["prompt"])
        return {"rle": [_encode_mask(_rect_mask(10, 20, 20, 40))], "scores": [0.9]}

    cache_dir = tmp_path / "cache"
    made, failures = sam_stage(
        frames_dir, cache_dir, [0, 30], ("person",), subscriber
    )
    assert made == 2 and not failures
    made, failures = sam_stage(
        frames_dir, cache_dir, [0, 30], ("person",), subscriber
    )
    assert made == 0 and len(calls) == 2  # fully cached: zero new spend


# ------------------------------------------------- full run, injected fakes
def _fake_extractor(frame_ids):
    def extract(video_path, frames_dir, sample_fps, max_frames):
        frames_dir.mkdir(parents=True, exist_ok=True)
        for frame_id in frame_ids:
            Image.new("RGB", (_W, _H), (120, 120, 120)).save(
                frames_dir / f"f{frame_id:06d}.jpg"
            )
        return list(frame_ids), 30.0

    return extract


def _fake_sam_subscriber():
    state = {"calls": 0}

    def subscriber(endpoint, *, arguments):
        state["calls"] += 1
        if arguments["prompt"] == "person":
            # Walks right along the bottom rows as calls advance.
            offset = min(40, 4 * state["calls"])
            mask = _rect_mask(offset, 26, offset + 8, 46)
        else:
            mask = _rect_mask(2, 30, 14, 46)
        return {"rle": [_encode_mask(mask)], "scores": [0.9]}

    return subscriber


def _fake_track_fn(detections, frame_ids, native_fps, labels):
    tracks = {}
    for label in labels:
        for frame_id in frame_ids:
            for row in detections.get(frame_id, {}).get(label, [])[:1]:
                tracks.setdefault(f"{label}-1", []).append(
                    (frame_id, row["bbox"])
                )
    return tracks


def _fake_moge_runner(tmp_path):
    def runner(model_identifier, *, input):
        import open3d as o3d

        grid = _synthetic_grid()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(grid.reshape(-1, 3))
        ply_path = tmp_path / "moge_fixture.ply"
        o3d.io.write_point_cloud(str(ply_path), cloud, write_ascii=True)
        return {"pointcloud_ply": str(ply_path)}

    return runner


def test_run_video_assessment_offline_end_to_end(tmp_path):
    store = ArtifactStore(tmp_path / "runs")
    report = run_video_assessment(
        tmp_path / "clip.avi",
        store=store,
        run_id="video-run-1",
        sample_fps=1.0,
        max_frames=8,
        labels=("person", "car"),
        zone_wkt="POLYGON ((-50 -50, 50 -50, 50 50, -50 50, -50 -50))",
        sam_subscriber=_fake_sam_subscriber(),
        moge_runner=_fake_moge_runner(tmp_path),
        track_fn=_fake_track_fn,
        frame_extractor=_fake_extractor([0, 30, 60, 90]),
    )
    paths = video_paths(store, "video-run-1")
    # Artifacts on disk.
    assert paths.report_json.is_file()
    assert paths.overlay_gif.is_file()
    assert paths.overlay_strip_png.is_file()
    assert paths.topdown_png.is_file()
    manifest = RunManifest.model_validate_json(
        paths.manifest_json.read_text(encoding="utf-8")
    )
    assert manifest.capture_tier == "video-mono"
    # Report schema and honesty markers.
    assert report["abstained"] is None
    assert report["floor"]["fitted"] is True
    assert report["floor"]["camera_height_m"] == pytest.approx(1.5, abs=0.05)
    assert report["tier"]["capture_tier"] == "video-mono"
    assert "calibrated" in report["tier"]["calibrated_reference"]
    assert report["spend"]["sam_calls"] == 8  # 4 frames x 2 labels
    assert report["spend"]["sam_cost_usd"] == pytest.approx(0.08)
    for rule in ("R1_zone", "R2_min_distance", "R3_speed"):
        assert len(report["timelines"][rule]) == 4
    assert set(report["verdicts"]) == {
        "R1_zone",
        "R2_min_distance",
        "R3_speed",
        "overall",
    }
    assert report["trajectories"]["person-1"]
    # The synthetic person walks: consecutive lifted positions differ.
    positions = list(report["trajectories"]["person-1"].values())
    assert positions[0] != positions[-1]
    assert report["zone_wkt"].startswith("POLYGON")


def test_run_video_assessment_abstains_when_floor_fit_fails(tmp_path):
    store = ArtifactStore(tmp_path / "runs")

    def broken_moge(model_identifier, *, input):
        raise RuntimeError("moge outage")

    report = run_video_assessment(
        tmp_path / "clip.avi",
        store=store,
        run_id="video-run-2",
        labels=("person", "car"),
        sam_subscriber=_fake_sam_subscriber(),
        moge_runner=broken_moge,
        track_fn=_fake_track_fn,
        frame_extractor=_fake_extractor([0, 30, 60]),
    )
    assert report["abstained"].startswith("floor fit failed")
    assert report["floor"]["fitted"] is False
    assert set(report["verdicts"].values()) == {"NO_DATA"}
    assert report["trajectories"] == {}
    paths = video_paths(store, "video-run-2")
    assert paths.report_json.is_file()  # the abstention is a real artifact
    assert paths.overlay_gif.is_file()  # masks still render as evidence
    assert not paths.topdown_png.exists()  # no metric floor, no top-down


def test_run_video_assessment_raises_provider_error_on_sam_outage(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(video_module.time, "sleep", lambda seconds: None)
    store = ArtifactStore(tmp_path / "runs")

    def broken_sam(endpoint, *, arguments):
        raise RuntimeError("fal outage")

    with pytest.raises(ProviderError):
        run_video_assessment(
            tmp_path / "clip.avi",
            store=store,
            run_id="video-run-3",
            labels=("person",),
            sam_subscriber=broken_sam,
            moge_runner=_fake_moge_runner(tmp_path),
            track_fn=_fake_track_fn,
            frame_extractor=_fake_extractor([0, 30]),
        )


def test_run_video_assessment_validates_inputs(tmp_path):
    store = ArtifactStore(tmp_path / "runs")
    with pytest.raises(ValueError, match="label"):
        run_video_assessment(
            tmp_path / "clip.avi",
            store=store,
            run_id="video-run-4",
            labels=(" ",),
            frame_extractor=_fake_extractor([0, 30]),
        )
    with pytest.raises(ValueError, match="POLYGON"):
        run_video_assessment(
            tmp_path / "clip.avi",
            store=store,
            run_id="video-run-5",
            zone_wkt="POINT (0 0)",
            frame_extractor=_fake_extractor([0, 30]),
        )
    with pytest.raises(ValueError, match="at least 2"):
        run_video_assessment(
            tmp_path / "clip.avi",
            store=store,
            run_id="video-run-6",
            sam_subscriber=_fake_sam_subscriber(),
            frame_extractor=_fake_extractor([0]),
        )
