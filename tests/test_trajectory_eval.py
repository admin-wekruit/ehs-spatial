"""Offline tests for the trajectory pilot's pure logic: calibration
transforms, banded temporal verdicts, speed windows, agreement math, and
the scorecard schema — no network, no dataset files."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

SCRIPT = Path(__file__).parents[1] / "scripts" / "trajectory_eval.py"
SPEC = importlib.util.spec_from_file_location("trajectory_eval", SCRIPT)
trajectory_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trajectory_eval)


def _rotation(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_calibration_transform_round_trip():
    rotation = _rotation(0.7) @ np.array(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float
    )
    translation = np.array([1.5, -2.0, 4.0])
    world = np.array([[0.0, 0.0, 0.0], [3.2, -7.1, 0.9], [-8.0, -18.0, 1.6]])
    camera = trajectory_eval.camera_from_world(rotation, translation, world)
    back = trajectory_eval.world_from_camera(rotation, translation, camera)
    assert np.allclose(back, world, atol=1e-9)


def test_extrinsic_parts_shapes():
    sensor = {"extrinsicMatrix": np.hstack(
        [np.eye(3), [[1.0], [2.0], [3.0]]]
    ).tolist()}
    rotation, translation = trajectory_eval.extrinsic_parts(sensor)
    assert rotation.shape == (3, 3) and translation.shape == (3,)
    assert np.allclose(translation, [1.0, 2.0, 3.0])


def test_world_to_map_flips_y():
    # Verified convention: py = map_h - (y + ty) * scale.
    px, py = trajectory_eval.world_to_map(
        np.array([1.0, -2.0]), scale=10.0, tx=4.0, ty=5.0, map_h=100
    )
    assert (px, py) == (50.0, 70.0)


def test_zone_depth_signed():
    square = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    assert trajectory_eval.zone_depth((2, 2), square) == pytest.approx(2.0)
    assert trajectory_eval.zone_depth((2, 3.9), square) == pytest.approx(0.1)
    assert trajectory_eval.zone_depth((2, 5), square) == pytest.approx(-1.0)


def test_banded_verdicts_zone_and_distance_and_speed():
    verdict = trajectory_eval.banded_verdict
    band = 0.35
    # Zone/speed style: measured above threshold violates.
    assert verdict(0.5, 0.0, band, "above") == "FAIL"
    assert verdict(0.2, 0.0, band, "above") == "NEEDS_REVIEW"
    assert verdict(-0.2, 0.0, band, "above") == "NEEDS_REVIEW"
    assert verdict(-0.5, 0.0, band, "above") == "PASS"
    # Min-distance style: measured below threshold violates.
    assert verdict(1.0, 2.0, band, "below") == "FAIL"
    assert verdict(1.9, 2.0, band, "below") == "NEEDS_REVIEW"
    assert verdict(2.2, 2.0, band, "below") == "NEEDS_REVIEW"
    assert verdict(3.0, 2.0, band, "below") == "PASS"
    # band=0 is the perfect-tracking binary verdict, no NEEDS_REVIEW.
    assert verdict(0.01, 0.0, 0.0, "above") == "FAIL"
    assert verdict(-0.01, 0.0, 0.0, "above") == "PASS"
    assert verdict(1.99, 2.0, 0.0, "below") == "FAIL"
    assert verdict(2.01, 2.0, 0.0, "below") == "PASS"


def test_window_speeds_constant_velocity():
    step = trajectory_eval.FRAME_IDS[1] - trajectory_eval.FRAME_IDS[0]
    dt = step / trajectory_eval.base.FPS
    series = {
        0: np.array([0.0, 0.0]),
        step: np.array([2.0, 0.0]),
        2 * step: np.array([2.0, 2.0]),
        # A gap (non-consecutive sample) must not produce a window.
        4 * step: np.array([10.0, 10.0]),
    }
    speeds = trajectory_eval.window_speeds(series)
    assert set(speeds) == {step, 2 * step}
    assert speeds[step] == pytest.approx(2.0 / dt)
    assert speeds[2 * step] == pytest.approx(2.0 / dt)


def test_rule_agreement_counts():
    gt = ["PASS", "FAIL", "PASS", "FAIL"]
    est = ["PASS", "NEEDS_REVIEW", "FAIL", "FAIL"]
    result = trajectory_eval.rule_agreement(gt, est)
    assert result["n"] == 4
    assert result["est_needs_review"] == 1
    assert result["decided"] == 3
    assert result["decided_agree"] == 2
    assert result["agreement_rate_decided"] == pytest.approx(0.667)
    assert result["hard_disagreements"] == 1


def test_lift_box_world_recovers_known_point():
    # A synthetic camera 3 m up looking straight down +x; MapAnything frame
    # equals its camera frame (identity pose) at half metric scale.
    rotation = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    world_point = np.array([5.0, 0.0, 0.0])
    translation = -rotation @ np.array([0.0, 0.0, 3.0])
    camera_point = rotation @ world_point + translation  # camera frame, metres
    height, width = 64, 64
    points = np.tile(camera_point / 2.0, (height, width, 1))  # half scale
    valid = np.ones((height, width), dtype=bool)
    lifted = trajectory_eval.lift_box_world(
        points, valid, np.eye(4), [0, 0, 1920, 1080], 2.0,
        rotation, translation,
    )
    assert lifted is not None
    assert np.allclose(lifted, world_point, atol=1e-6)


def test_scorecard_schema():
    scorecard = trajectory_eval.build_scorecard(
        meta={"scene": "x"}, layer1={"n_questions": 0},
        layer2={"mono": {}}, layer3={"R1_keep_clear_zone": {}},
        spend={"sam_calls": 0},
    )
    assert set(scorecard) == {
        "meta",
        "layer1_distance_rescore",
        "layer2_trajectories",
        "layer3_temporal_judgment",
        "spend",
    }
    assert scorecard["spend"]["sam_calls"] == 0
