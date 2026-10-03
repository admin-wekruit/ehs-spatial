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
    BudgetExhausted,
    FloorCamera,
    HourlySpendGuard,
    banded_verdict,
    contract_scale,
    fit_floor_from_keyframes,
    fit_pinhole_from_grid,
    frame_health,
    image_region,
    judge,
    judge_frame,
    keep_untracked,
    lift_foot,
    lift_tracks,
    track_speed,
    ransac_plane,
    run_video_assessment,
    sam_stage,
    sample_schedule,
    video_paths,
    worst_verdict,
    zone_floor_points,
    zone_floor_seen,
    zone_verdict,
)
from shapely import wkt as shapely_wkt

_PROCESS_GUARD = video_module.SAM_SPEND


@pytest.fixture(autouse=True)
def _own_spend_guard(monkeypatch):
    """Fake SAM calls charge a guard of their own, never the process's (they did: 20 charges a run)."""
    monkeypatch.setattr(video_module, "SAM_SPEND", HourlySpendGuard())


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


def test_lift_foot_floor_tolerance_grows_with_range():
    """A fixed 0.1 m floor tolerance rejected nearly every far foot on real fixed cameras (3 of 56 on bus g506)."""
    camera = _camera(3.4)
    plane = (camera.normal, camera.d)
    person = (310.0, 282.5, 330.0, 325.0)  # 1.7 m person 20 m out: foot v = 240 + 500 * 3.4 / 20
    long_4pct = _floor_depth(camera) * 1.04  # a mono depth model 4% long, inside its measured error
    assert lift_foot(person, (640, 480), camera.K, plane, long_4pct)[2] == []
    assert "foot_off_observed_floor" in lift_foot(person, (640, 480), camera.K, plane, _floor_depth(camera) * 1.2)[2]


def test_lift_tracks_keeps_people_it_cannot_place():
    """lift_tracks dropped a box whose foot is above the horizon, so a head over a shelf counted as nobody and the
    zone PASSed. It stays, unplaced, and blocks a PASS when it overlaps where someone in the zone could appear."""
    camera = _camera(1.2)
    camera.depth = _floor_depth(camera)
    zone = shapely_wkt.loads("POLYGON ((-1 4, 1 4, 1 6, -1 6, -1 4))")
    tracks = {"person-1": [(0, (300.0, 185.0, 340.0, 233.0))],  # head over a 1.25 m shelf, standing 5 m out
              "person-2": [(0, (1.0, 185.0, 40.0, 233.0))]}  # the same, 3 m to the side of the zone
    lifted = lift_tracks(tracks, camera, (640, 480), zone)
    assert lifted["person-1"][0] == {"xy": None, "review": ["foot_above_horizon"], "near_zone": True}
    assert lifted["person-2"][0]["xy"] is None and lifted["person-2"][0]["near_zone"] is False
    covered = {0: {"gap": None, "zone_seen": True}}
    assert judge(lifted, zone, {0: 0.0}, covered, _MEASURED)["R1_zone"][0] == "NEEDS_REVIEW"
    aside = {"person-2": lifted["person-2"]}
    assert judge(aside, zone, {0: 0.0}, covered, _MEASURED)["R1_zone"][0] == "PASS"
    assert "box_too_small" in lift_foot((310.0, 300.0, 330.0, 310.0), (640, 480), camera.K, (camera.normal, camera.d),
                                        camera.depth, person=False)[2]  # a 10 px mover is not placed either


def test_static_depth_keeps_the_farthest_surface():
    """The fixed camera's depth plate: a person standing in one keyframe must not hide the floor behind them."""
    camera = _camera(2.0)
    floor = np.array([[0.0, 2.0, 5.0], [1.0, 2.0, 5.0]])  # two floor points 5 m out, at pixels (320, 440), (420, 440)
    with_person = np.array([[0.0, 1.6, 4.0], [1.0, 2.0, 5.0]])  # someone 4 m out on the first point's ray
    depth = video_module._static_depth([with_person, floor], camera.K, (640, 480))
    assert depth[440, 320] == pytest.approx(5.0) and depth[440, 420] == pytest.approx(5.0) and (depth > 0).sum() == 2


def test_untracked_detections_are_kept_as_one_frame_tracks():
    """ByteTrack holds back a first sighting: a person seen in one sampled frame had no track and counted as nobody."""
    detections = {0: {"person": [{"bbox": (10, 20, 20, 40), "score": 0.9}, {"bbox": (40, 20, 50, 40), "score": 0.9}]}}
    tracks = {"person-1": [(0, (10.5, 20.0, 20.0, 40.0))]}
    kept = keep_untracked(tracks, detections)
    assert kept == {**tracks, "person-untracked-0-1": [(0, (40, 20, 50, 40))]}


def test_zone_floor_must_be_observed_not_just_in_view(tmp_path):
    """A zone behind a wall PASSed: only its corners were checked against the image."""
    camera = _camera(2.0)
    camera.depth = _floor_depth(camera)
    zone = shapely_wkt.loads("POLYGON ((-1 5, 1 5, 1 7, -1 7, -1 5))")
    points, cells = zone_floor_points(zone)
    points = camera.camera_points(points)
    assert zone_floor_seen(points, cells, camera.K, camera.depth, (640, 480))
    assert not zone_floor_seen(points, cells, camera.K, np.minimum(camera.depth, 3.5), (640, 480))  # a wall 3.5 m out
    frames_dir, cache_dir = tmp_path / "frames", tmp_path / "cache"
    frames_dir.mkdir()
    cache_dir.mkdir()
    for frame_id in (0, 1):
        Image.fromarray(np.random.default_rng(frame_id).integers(20, 235, (480, 640, 3), dtype=np.uint8)).save(
            frames_dir / f"f{frame_id:06d}.jpg")
        (cache_dir / f"f{frame_id:06d}__person.json").write_text("{}")
    truck = {"car": [{"bbox": (150.0, 200.0, 500.0, 400.0), "score": 0.9}]}  # parked in front of the zone in frame 1
    coverage = video_module.frame_coverage(frames_dir, cache_dir, [0, 1], zone, camera, {1: truck})
    assert [coverage[f]["zone_seen"] for f in (0, 1)] == [True, False]


def test_a_hidden_footprint_anywhere_in_the_zone_is_not_seen():
    """ZONE_SEEN_FRACTION 0.95 let 5% of the zone floor go unobserved, more than a person covers on any zone over 2-5 m2.
    Now any 0.5 m footprint, placed and turned anywhere inside the zone, hidden from a camera straight above, is found;
    one unobserved sample alone (sensor noise) is not."""
    import cv2
    zone = shapely_wkt.loads("POLYGON ((0 0, 3 0, 3 3, 0 3, 0 0))")
    points, cells = zone_floor_points(zone)
    above = points - [1.5, 1.5, -4.0]  # camera 4 m over the zone's middle, looking down: x, y and 4 m depth
    K, size = np.array([[100.0, 0, 160], [0, 100.0, 120], [0, 0, 1]]), (320, 240)
    floor = np.full(size[::-1], 4.0)
    assert zone_floor_seen(above, cells, K, floor, size)
    rng = np.random.default_rng(0)
    for _ in range(300):
        angle, centre = rng.uniform(0, np.pi / 2), rng.uniform(0.36, 2.64, 2)  # 0.36 m: the turned footprint stays inside
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        corners = (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * video_module.PERSON_FOOTPRINT_M / 2) @ rotation.T + centre
        pixels = (corners - 1.5) * 100 / 4 + [160, 120]
        hidden = cv2.dilate(cv2.fillPoly(np.zeros(size[::-1], np.uint8), [np.round(pixels).astype(np.int32)], 1), np.ones((3, 3)))
        occluded = np.where(hidden > 0, 3.0, floor)  # something 1 m tall over the footprint: its floor is not observed
        assert not zone_floor_seen(above, cells, K, occluded, size), (angle, centre)
    noisy = floor.copy()
    lone = (above[cells[:, 0] == 5][:1] @ K.T)[0]
    noisy[int(lone[1] / lone[2]), int(lone[0] / lone[2])] = 0.0  # one sample without depth
    assert zone_floor_seen(above, cells, K, noisy, size)


def test_an_unplaced_person_makes_distance_and_speed_review():
    """judge_frame gave R2 PASS for a placed person 10 m from a mover while a person nobody could place stood beside it."""
    placed, unplaced, mover = {"xy": (0.0, 0.0), "review": []}, {"xy": None, "review": ["foot_above_horizon"]}, {"xy": (10.0, 0.0)}
    assert judge_frame([placed], [mover], None, None, True, [], _MEASURED)["R2_min_distance"]["verdict"] == "PASS"
    assert judge_frame([placed, unplaced], [mover], None, None, True, [], _MEASURED)["R2_min_distance"]["verdict"] == "NEEDS_REVIEW"
    lifted = {"person-1": {0: unplaced, 30: unplaced, 60: unplaced}}  # the batch judge: seen three times, never placed
    assert judge(lifted, None, _TIMES, None, _MEASURED)["R3_speed"] == {0: "NEEDS_REVIEW", 30: "NEEDS_REVIEW", 60: "NEEDS_REVIEW"}


def test_lift_foot_occlusion_margin_grows_with_range():
    """At 20 m a depth 5% short is inside the depth error, not an occluder; a fixed 0.3 m margin flagged it."""
    camera = _camera(3.4)
    plane = (camera.normal, camera.d)
    person = (310.0, 282.5, 330.0, 325.0)  # 1.7 m person 20 m out: foot v = 240 + 500 * 3.4 / 20
    depth = _floor_depth(camera)
    short = depth.copy()
    short[325:329] -= 0.5  # 0.5 m nearer below the foot: 2.5% of the range
    assert "foot_occluded" not in lift_foot(person, (640, 480), camera.K, plane, short)[2]
    short[325:329] = depth[325:329] - 1.5  # 7.5% of the range: something stands there
    assert "foot_occluded" in lift_foot(person, (640, 480), camera.K, plane, short)[2]


def test_image_region_behind_the_camera_is_the_whole_frame():
    """A zone prism with a corner behind the camera can appear anywhere in the image, never nowhere."""
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    in_front = np.array([[-1.0, 0, 5], [1, 0, 5], [0, 1, 6]])
    assert image_region(in_front, K, (640, 480)).area < 640 * 480 / 10
    assert image_region(np.vstack([in_front, [0, 0, -1.0]]), K, (640, 480)).area == 640 * 480


def test_track_speed_reads_only_its_window():
    """Live tracks kept every sample and track_speed scanned them all each step (32.8 ms a step at 30k frames)."""
    from collections.abc import Sequence

    class Counted(Sequence):
        def __init__(self, items):
            self.items, self.reads = items, 0

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            self.reads += 1
            return self.items[index]

    samples = Counted(_walker(10, 1.0, noise=0.0, seconds=10_000.0))
    value, reason = track_speed(samples, 10_000.0)
    assert reason is None and value == pytest.approx(1.0) and samples.reads < 30, samples.reads


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
    assert judged["R3_speed"][60] == "NEEDS_REVIEW" and judged["before_scale_gate"]["R3_speed"][60] == "FAIL"
    # A zone entry is judged in metres with a metre band: without measured metres it only asks for review.
    assert judged["R1_zone"][0] == "NEEDS_REVIEW" and judged["before_scale_gate"]["R1_zone"][0] == "FAIL"
    stated = contract_scale({"scale_status": "assumed_camera_height", "metres_per_native_unit": 2.0,
                             "camera_height_native_median": 0.8})
    assert judge(_lifted_fixture(), zone, _TIMES, scale_record=stated)["R2_min_distance"][0] == "NEEDS_REVIEW"


def test_judge_without_zone_or_person_abstains():
    judged = judge(_lifted_fixture(), None, _TIMES)
    assert set(judged["R1_zone"].values()) == {"NO_DATA"}
    lifted = {"car-1": {0: {"xy": (0.0, 1.0)}, 30: {"xy": (0.0, 1.0)}}}
    zone = shapely_wkt.loads("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
    covered = {f: {"gap": None, "zone_seen": True} for f in _TIMES}
    judged = judge(lifted, zone, _TIMES, covered, _MEASURED)
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
    assert worst_verdict({0: "PASS", 1: "NO_DATA"}) == "NO_DATA"  # PASS needs every instant to pass


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
    guard.record(1.49)  # a per-second bill booked after the call
    with pytest.raises(BudgetExhausted):
        guard.charge(0.0)  # nothing left: even a call billed afterwards is refused


def test_fake_sam_calls_never_charge_the_process_guard(tmp_path):
    before = _PROCESS_GUARD.remaining()
    run_video_assessment(
        tmp_path / "clip.avi", store=ArtifactStore(tmp_path / "runs"), run_id="video-run-guard", labels=("person",),
        sam_subscriber=_fake_sam_subscriber(), moge_runner=_fake_moge_runner(tmp_path), track_fn=_fake_track_fn,
        frame_extractor=_fake_extractor([0, 30]),
    )
    assert _PROCESS_GUARD.remaining() == before


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
        for frame_id in frame_ids:  # frames that differ, as a live camera's do: identical ones are a frozen feed
            noise = np.random.default_rng(frame_id).integers(60, 180, (_H, _W, 3), dtype=np.uint8)
            Image.fromarray(noise).save(frames_dir / f"f{frame_id:06d}.jpg")
        return list(frame_ids), 30.0

    return extract


def _fake_sam_subscriber():
    state = {"calls": 0}

    def subscriber(endpoint, *, arguments):
        state["calls"] += 1
        if arguments["prompt"] == "person":
            # A 1.7 m person 6 m out walking right as calls advance: foot at v = 24 + 80 * 1.5 / 6 = 44.
            offset = min(40, 4 * state["calls"])
            mask = _rect_mask(offset, 21, offset + 6, 44)
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
        zone_wkt="POLYGON ((-0.5 5.5, 0.5 5.5, 0.5 7.5, -0.5 7.5, -0.5 5.5))",
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
    assert [report["coverage"][str(f)]["gap"] for f in (0, 30, 60, 90)] == [None] * 4
    # The walker's feet pass lift_foot; it starts 1.9 m right of the zone (seen: PASS) and ends inside it (FAIL).
    assert report["foot_review"]["person-1"] == {}
    assert report["coverage"]["0"]["zone_seen"] is True
    assert report["timelines_before_scale_gate"]["R1_zone"]["0"] == "PASS"
    assert report["timelines_before_scale_gate"]["R1_zone"]["90"] == "FAIL"
    assert report["scale"]["status"] == "model_estimated"  # MoGe metres decide nothing: every rule asks for review
    assert set(report["timelines"]["R1_zone"].values()) == {"NEEDS_REVIEW"}


def test_load_detections_keeps_small_masks(tmp_path):
    """Masks under 12 px were dropped before tracking: a distant or half-hidden person counted as nobody."""
    (tmp_path / "f000000__person.json").write_text(json.dumps(
        {"rle": [_encode_mask(_rect_mask(10, 20, 14, 28))], "scores": [0.8], "width": _W, "height": _H}))
    assert [row["bbox"] for row in video_module.load_detections(tmp_path, [0], ("person",))[0]["person"]] == [
        (10.0, 20.0, 14.0, 28.0)]


def test_run_video_assessment_keeps_people_the_tracker_held_back(tmp_path):
    """A detection the tracker gives no identity still decides R1: here the tracker names nobody at all."""
    report = run_video_assessment(
        tmp_path / "clip.avi", store=ArtifactStore(tmp_path / "runs"), run_id="video-run-untracked",
        labels=("person", "car"), zone_wkt="POLYGON ((-0.5 5.5, 0.5 5.5, 0.5 7.5, -0.5 7.5, -0.5 5.5))",
        sam_subscriber=_fake_sam_subscriber(), moge_runner=_fake_moge_runner(tmp_path),
        track_fn=lambda *args: {}, frame_extractor=_fake_extractor([0, 30, 60, 90]),
    )
    assert report["timelines_before_scale_gate"]["R1_zone"]["90"] == "FAIL"
    assert "person-untracked-90-0" in report["tracks"]


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


def test_a_zone_cell_is_hidden_when_any_of_its_samples_is():
    """Two samples per cell, one seen and one behind something 1 m nearer: every cell is hidden, so the 2 x 2 block is
    floor a person could stand on unseen. Counting a cell as clear when any sample was seen would pass it."""
    K = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    points, cells = [], []
    for ci, (a, b) in enumerate(((-.3, -.2), (.2, .3))):
        for cj, (c, d) in enumerate(((-.3, -.2), (.2, .3))):
            points += [(a, c, 2.), (b, d, 2.)]
            cells += [(ci, cj), (ci, cj)]
    points, cells = np.array(points), np.array(cells)
    depth = np.full((480, 640), 2., np.float32)
    assert video_module.zone_floor_seen(points, cells, K, depth, (640, 480))
    for u, v in (points @ K.T / points[:, 2:3])[1::2, :2].astype(int):
        depth[v, u] = 1.
    assert not video_module.zone_floor_seen(points, cells, K, depth, (640, 480))
