"""Offline tests for the MTMC eval's pure logic: GT box-distance math,
question building, and robust box-region point selection — no network."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "mtmc_eval.py"
SPEC = importlib.util.spec_from_file_location("mtmc_eval", SCRIPT)
mtmc_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mtmc_eval)


def _gt_object(object_id, location, scale=(1.0, 1.0, 1.0), yaw=0.0, boxes=None):
    return {
        "object id": object_id,
        "object type": "Person",
        "3d location": list(location),
        "3d bounding box scale": list(scale),
        "3d bounding box rotation": [0.0, 0.0, yaw],
        "2d bounding box visible": boxes or {},
    }


def test_gt_min_distance_between_unit_boxes():
    a = _gt_object(1, (0.0, 0.0, 0.5))
    b = _gt_object(2, (2.0, 0.0, 0.5))
    # Unit cubes centred 2 m apart on x: faces at 0.5 and 1.5 -> gap 1.0.
    assert mtmc_eval.gt_min_distance(a, b) == pytest.approx(1.0, abs=1e-6)


def test_gt_min_distance_respects_yaw():
    # A 4 m long box touches its neighbour face-to-face (gap 0); rotated
    # 90 deg about z it extends along y instead, opening a 1.5 m x-gap.
    long_box = _gt_object(1, (0.0, 0.0, 0.5), scale=(4.0, 1.0, 1.0))
    neighbour = _gt_object(2, (2.5, 0.0, 0.5))
    assert mtmc_eval.gt_min_distance(long_box, neighbour) == pytest.approx(
        0.0, abs=1e-6
    )
    rotated = _gt_object(
        1, (0.0, 0.0, 0.5), scale=(4.0, 1.0, 1.0), yaw=np.pi / 2
    )
    # Grid sampling quantises by up to ~scale/(grid-1); 5 cm covers it here.
    assert mtmc_eval.gt_min_distance(rotated, neighbour) == pytest.approx(
        1.5, abs=0.05
    )


def test_build_questions_filters_and_pairs():
    big = [100, 100, 300, 400]  # 200x300 px, above the area floor
    tiny = [0, 0, 20, 20]  # below MIN_BOX_AREA_PX -> dropped
    ground_truth = {
        "0": [
            _gt_object(1, (0.0, 0.0, 0.5), boxes={"Camera": big}),
            _gt_object(2, (3.0, 0.0, 0.5), boxes={"Camera": big}),
            _gt_object(3, (6.0, 0.0, 0.5), boxes={"Camera": tiny}),
            # Closer than MIN_GT_DISTANCE_M to object 1 -> pair dropped.
            _gt_object(4, (0.2, 0.0, 0.5), boxes={"Camera": big}),
            _gt_object(5, (9.0, 0.0, 0.5), boxes={"Camera_99": big}),
        ]
    }
    questions = mtmc_eval.build_questions(ground_truth, ("Camera",), (0,))
    pairs = {tuple(q["ids"]) for q in questions}
    assert pairs == {(1, 2), (2, 4)}
    question = next(q for q in questions if q["ids"] == [1, 2])
    assert question["gt_centroid_m"] == pytest.approx(3.0)
    assert question["gt_min_m"] == pytest.approx(2.0)
    assert question["image"] == "Camera_000000"


def test_box_points_selects_region_and_trims_outliers():
    height, width = 108, 192  # 1/10th of the 1920x1080 source
    points = np.zeros((height, width, 3))
    points[..., 2] = 50.0  # far background everywhere
    points[40:60, 40:60, 2] = 5.0  # the object surface
    valid = np.ones((height, width), dtype=bool)
    # Source-resolution box covering the object region (x10 scale).
    cloud = mtmc_eval.box_points(points, valid, [400, 400, 600, 600])
    assert cloud is not None
    # The MAD trim keeps the dominant 5 m surface only.
    assert np.median(np.linalg.norm(cloud, axis=1)) == pytest.approx(5.0)
    assert np.linalg.norm(cloud, axis=1).max() < 10.0


def test_box_points_rejects_sparse_regions():
    points = np.zeros((10, 10, 3))
    valid = np.zeros((10, 10), dtype=bool)
    assert mtmc_eval.box_points(points, valid, [0, 0, 1920, 1080]) is None


def test_min_distance_between_clusters():
    a = np.tile([0.0, 0.0, 5.0], (100, 1))
    b = np.tile([3.0, 0.0, 5.0], (100, 1))
    assert mtmc_eval.min_distance(a, b) == pytest.approx(3.0)
