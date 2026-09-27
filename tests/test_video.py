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
    HourlySpendGuard,
    banded_verdict,
    contract_scale,
    fit_floor_from_keyframes,
    fit_pinhole_from_grid,
    frame_health,
    judge,
    judge_frame,
    lift_foot,
    lift_tracks,
    track_speed,
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
    lifted = lift_tracks(tracks, camera, (640, 480))
    assert lifted["person-1"][0]["xy"] == pytest.approx((0.0, 5.0))
    assert "edge" not in lifted["person-1"][0]
    assert "edge" in lifted["forklift-1"][0]
    left, right = lifted["forklift-1"][0]["edge"]
    assert left != right
    # A 0.4 m tall "person" with no static depth below it is never accepted.
    assert lifted["person-1"][0]["height_m"] == pytest.approx(0.4)
    assert set(lifted["person-1"][0]["review"]) == {"implied_height", "no_depth_below_foot"}


def _floor_depth(camera: FloorCamera, size=(640, 480)) -> np.ndarray:
    """Static depth of an empty floor (and far wall above the horizon)."""
    width, height = size
    vv, uu = np.mgrid[0:height, 0:width]
    rays = np.stack([(uu - 320.0) / 500.0, (vv - 240.0) / 500.0, np.ones_like(uu, float)], -1)
    down = rays[..., 1]
    return np.where(down > 1e-3, camera.d / np.maximum(down, 1e-3), 50.0)


def test_lift_foot_accepts_only_a_plausible_visible_standing_foot():
    camera = _camera(2.0)
    plane, depth = (camera.normal, camera.d), _floor_depth(camera)
    # 1.7 m person 5 m out: foot at v = 440, head at v = 240 + 500 * 0.3 / 5 = 270.
    person = (310.0, 270.0, 330.0, 440.0)
    foot, height, reasons = lift_foot(person, (640, 480), camera.K, plane, depth)
    assert height == pytest.approx(1.7) and reasons == []
    assert foot == pytest.approx([0.0, 2.0, 5.0])
    assert "implied_height" in lift_foot((310.0, 390.0, 330.0, 440.0), (640, 480), camera.K, plane, depth)[2]
    assert "box_touches_border" in lift_foot((310.0, 270.0, 330.0, 479.0), (640, 480), camera.K, plane, depth)[2]
    # A bench 3 m out hides the feet: depth below the box is 2 m nearer than the floor there.
    occluded = depth.copy()
    occluded[440:450, 300:340] = 3.0
    reasons = lift_foot(person, (640, 480), camera.K, plane, occluded)[2]
    assert "foot_occluded" in reasons and "foot_off_observed_floor" in reasons
    # A hole in the observed floor (a pit 0.5 m deep): not nearer, but not floor either.
    pit = depth * 1.25
    assert lift_foot(person, (640, 480), camera.K, plane, pit)[2] == ["foot_off_observed_floor"]
    # MoGe's metres are a model's: height needs metres_per_unit to judge in metres.
    assert lift_foot(person, (640, 480), camera.K, plane, depth, metres_per_unit=0.5)[1] == pytest.approx(0.85)


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


_TIMES = {0: 0.0, 30: 1.0, 60: 2.0}
_MEASURED = contract_scale({"scale_status": "device_metric", "metres_per_native_unit": 1.0})


def test_judge_timelines_on_synthetic_tracks():
    zone = shapely_wkt.loads("POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))")
    covered = {f: {"gap": None, "zone_seen": True} for f in _TIMES}
    judged = judge(_lifted_fixture(), zone, _TIMES, covered, _MEASURED)
    assert judged["speed_band_mps"] == pytest.approx(0.35)  # 2 x 0.35 m over the 2 s baseline
    assert judged["R1_zone"][0] == "FAIL"  # person inside the zone
    assert judged["R1_zone"][60] == "PASS"
    assert judged["R2_min_distance"][0] == "FAIL"
    assert judged["R2_min_distance"][60] == "NO_DATA"  # vehicle vanished
    assert judged["R3_speed"][0] == "NO_DATA"  # no 2 s baseline yet
    assert judged["R3_speed"][30] == "NO_DATA"
    assert judged["R3_speed"][60] == "FAIL"  # line through 0, 4, 4.2 m: 2.1 m/s
    assert judged["R2_values_m"][0] == pytest.approx(1.0)
    assert judged["R3_values_mps"][60] == pytest.approx(2.1)


def test_judge_scale_gate_turns_unmeasured_metres_into_review():
    zone = shapely_wkt.loads("POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))")
    judged = judge(_lifted_fixture(), zone, _TIMES)  # default: MoGe metres, a model's estimate
    assert judged["scale"]["status"] == "model_estimated"
    assert judged["R2_min_distance"][0] == "NEEDS_REVIEW"
    assert judged["R3_speed"][60] == "NEEDS_REVIEW" and judged["R3_before_scale_gate"][60] == "FAIL"
    assert judged["R1_zone"][0] == "FAIL"  # a zone entry is not a distance or speed verdict
    stated = contract_scale({"scale_status": "assumed_camera_height", "metres_per_native_unit": 2.0,
                             "camera_height_native_median": 0.8})
    assert judge(_lifted_fixture(), zone, _TIMES, scale_record=stated)["R2_min_distance"][0] == "NEEDS_REVIEW"


def test_judge_without_zone_or_person_abstains():
    judged = judge(_lifted_fixture(), None, _TIMES)
    assert set(judged["R1_zone"].values()) == {"NO_DATA"}
    lifted = {"car-1": {0: {"xy": (0.0, 1.0)}, 30: {"xy": (0.0, 1.0)}}}
    zone = shapely_wkt.loads("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
    covered = {f: {"gap": None, "zone_seen": True} for f in _TIMES}
    judged = judge(lifted, zone, _TIMES, covered)
    assert set(judged["R1_zone"].values()) == {"PASS"}  # nobody in a zone fully seen
    assert set(judged["R2_min_distance"].values()) == {"NO_DATA"}
    assert set(judged["R3_speed"].values()) == {"NO_DATA"}
    assert set(judge(lifted, zone, _TIMES)["R1_zone"].values()) == {"NO_DATA"}  # no coverage evidence, no PASS


def test_no_detection_is_no_data_unless_the_zone_was_covered():
    """video.py used to PASS the zone rule whenever nobody was detected."""
    zone = shapely_wkt.loads("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
    for gap in ("frozen", "dark", "person detector offline", "no camera pose"):
        assert judge_frame([], [], zone, gap, True, [], _MEASURED)["R1_zone"]["verdict"] == "NO_DATA"
    assert judge_frame([], [], zone, None, False, [], _MEASURED)["R1_zone"]["verdict"] == "NO_DATA"
    assert judge_frame([], [], zone, None, True, [], _MEASURED)["R1_zone"]["verdict"] == "PASS"
    inside = [{"xy": (0.5, 0.5)}]
    assert judge_frame(inside, [], zone, None, False, [], _MEASURED)["R1_zone"]["verdict"] == "FAIL"  # seen inside
    unsure = [{"xy": (5.0, 5.0), "review": ["foot_occluded"]}]
    assert judge_frame(unsure, [], zone, None, True, [], _MEASURED)["R1_zone"]["verdict"] == "NEEDS_REVIEW"


def test_frame_health_flags_dark_and_frozen_frames():
    rng = np.random.default_rng(0)
    frame = rng.uniform(0, 255, (48, 64))
    assert frame_health(frame, None) is None
    assert frame_health(frame, frame.copy()) == "frozen"
    assert frame_health(frame * 0.02, None) == "dark"
    assert frame_health(frame, rng.uniform(0, 255, (48, 64))) is None


def _walker(rate_hz, speed=1.2, noise=0.1, seconds=10.0, seed=1):
    rng = np.random.default_rng(seed)
    times = np.arange(0, seconds + 1e-9, 1 / rate_hz)
    return [(float(t), (float(speed * t + rng.normal(0, noise)), float(rng.normal(0, noise)))) for t in times]


def test_speed_verdict_does_not_depend_on_the_sample_rate():
    """The old band 2 x 0.35 m / step widened to 3.5 m/s at 5 Hz: any walker passed."""
    for speed, expected in ((1.0, "PASS"), (2.4, "FAIL"), (1.55, "NEEDS_REVIEW")):
        verdicts = set()
        for rate in (1, 5, 10):
            samples = _walker(rate, speed)
            for t in (4.0, 7.0, 10.0):
                value, reason = track_speed(samples, t)
                assert reason is None and value == pytest.approx(speed, abs=0.2), (rate, t, value)
                verdicts.add(banded_verdict(value, 1.5, 0.35, fail_low=False))
        assert verdicts == {expected}, (speed, verdicts)
    assert track_speed(_walker(5), 1.5) == (None, None)  # no 2 s baseline yet


def test_an_impossible_jump_is_review_not_fail():
    samples = _walker(5, 1.0, noise=0.0)
    samples = [(t, (x + (6.0 if t >= 5.0 else 0.0), y)) for t, (x, y) in samples]  # identity swap at 5 s
    assert track_speed(samples, 6.0) == (None, "jump")
    assert track_speed(samples, 7.2)[1] is None  # the segment after the jump is clean again
    result = judge_frame([{"xy": (0.0, 0.0)}], [], None, None, True, [track_speed(samples, 6.0)], _MEASURED)
    assert result["R3_speed"]["verdict"] == "NEEDS_REVIEW"


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
            guard=HourlySpendGuard(),
        )
    assert not calls


def test_hourly_spend_guard_rolls_over_instead_of_capping_the_run():
    now = [0.0]
    guard = HourlySpendGuard(1.50, clock=lambda: now[0])
    for _ in range(150):
        guard.charge(0.01)
    with pytest.raises(ValueError, match="budget cap"):
        guard.charge(0.01)
    now[0] = 3600.0  # an hour later a live stream keeps going
    guard.charge(0.01)
    assert guard.remaining() == pytest.approx(1.49)


def test_default_detector_routes_through_the_backend_switch(tmp_path, monkeypatch):
    """_default_sam_subscriber called fal_client directly and ignored SAM3_BACKEND."""
    routed = []

    def switch(endpoint, *, arguments):
        routed.append(endpoint)
        return {"rle": [_encode_mask(_rect_mask(10, 20, 20, 40))], "scores": [0.9]}

    monkeypatch.setattr(video_module, "sam_subscribe", switch)
    monkeypatch.setenv("SAM3_BACKEND", "modal")
    store = ArtifactStore(tmp_path / "runs")
    run_video_assessment(
        tmp_path / "clip.avi", store=store, run_id="video-run-switch", labels=("person",),
        moge_runner=_fake_moge_runner(tmp_path), track_fn=_fake_track_fn,
        frame_extractor=_fake_extractor([0, 30]),
    )
    assert len(routed) == 2
    manifest = RunManifest.model_validate_json(video_paths(store, "video-run-switch").manifest_json.read_text())
    assert manifest.providers.sam_backend == "modal"
    assert "facebook/sam3" in manifest.providers.sam_model_revision


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
    # The fake extractor writes identical grey frames: a frozen feed covers nothing.
    assert [report["coverage"][str(f)]["gap"] for f in (0, 30)] == [None, "frozen"]
    assert {report["timelines"][rule]["30"] for rule in ("R1_zone", "R2_min_distance", "R3_speed")} == {"NO_DATA"}
    assert report["timelines"]["R1_zone"]["0"] == "NEEDS_REVIEW"  # a 20 px "person" never passes lift_foot
    assert report["scale"]["status"] == "model_estimated"  # MoGe metres never decide R2/R3


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


def test_live_people_loop_self_check():
    """The streaming loop's own replay checks (1 Hz vs 5 Hz, gaps, offline detector, scale gate)."""
    from ehs_spatial.live_people import self_check

    self_check()
