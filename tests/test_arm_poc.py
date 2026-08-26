"""Offline tests for the robot-arm POC (scripts/arm_poc.py). No network."""

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import arm_poc  # noqa: E402


# ------------------------------------------------------------------- FK
def test_fk_zero_pose_matches_franka_datasheet():
    # Known Panda zero-configuration flange pose: (0.088, 0, 0.926), z down.
    T = arm_poc.fk_flange(np.zeros(7))
    assert np.allclose(T[:3, 3], [0.088, 0.0, 0.926], atol=1e-12)
    assert np.allclose(T[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)


def test_fk_matches_h5_recorded_cartesian_sample():
    # First timestep of the AUTOLab episode, copied verbatim from trajectory.h5:
    # observation/robot_state/{joint_positions,cartesian_position}[0].
    q0 = np.array(
        [
            0.1727488934993744, -0.7544930577278137, 0.23161287605762482,
            -2.757798433303833, 0.04239807277917862, 1.8208118677139282,
            0.1214694231748581,
        ]
    )
    cart0_xyz = np.array(
        [0.27276310324668884, 0.11364641040563583, 0.40711116790771484]
    )
    cart0_euler = np.array(
        [2.9973092209816716, 0.17590440589840317, 0.1786799974429964]
    )
    T = arm_poc.fk_flange(q0)
    assert np.linalg.norm(T[:3, 3] - cart0_xyz) < 1e-6  # the 0.00 mm claim
    R_recorded = arm_poc.euler_xyz_to_matrix(cart0_euler)
    assert np.allclose(R_recorded, T[:3, :3], atol=1e-6)


def test_fk_origins_shape_and_chain():
    origins = arm_poc.fk_origins(np.zeros(7))
    assert origins.shape == (9, 3)
    assert np.allclose(origins[0], 0.0)  # base
    assert np.allclose(origins[1], [0.0, 0.0, 0.333])  # joint-1 origin
    assert np.allclose(origins[-1], [0.088, 0.0, 0.926])  # flange


def test_densify_skeleton_covers_segments():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    dense = arm_poc.densify_skeleton(origins, step=0.1)
    assert len(dense) >= 11
    assert np.allclose(dense[0], origins[0]) and np.allclose(dense[-1], origins[1])


def test_euler_xyz_roundtrip():
    angles = np.array([-1.838, 0.017, -1.984])
    R = arm_poc.euler_xyz_to_matrix(angles)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.allclose(arm_poc.matrix_to_euler_xyz(R), angles, atol=1e-12)


# ------------------------------------------------------------ state labels
def test_gt_state_labels_threshold():
    qd = np.array([[0.0] * 7, [0.01] * 7, [0.5] + [0.0] * 6])
    labels = arm_poc.gt_state_labels(qd, threshold=0.05)
    assert labels == ["PARKED", "PARKED", "ACTIVE"]


def test_mask_change_ratio():
    a = np.zeros((4, 4), dtype=bool)
    a[:2] = True
    b = np.zeros((4, 4), dtype=bool)
    b[1:3] = True
    assert arm_poc.mask_change_ratio(a, b) == pytest.approx(8 / 12)
    assert arm_poc.mask_change_ratio(a, a) == 0.0
    assert arm_poc.mask_change_ratio(None, a) is None
    empty = np.zeros((4, 4), dtype=bool)
    assert arm_poc.mask_change_ratio(empty, empty) is None


# --------------------------------------------------------------- envelopes
def test_envelope_iou_known_squares():
    a = box(0, 0, 2, 2)
    b = box(1, 0, 3, 2)
    # intersection 2, union 6
    assert arm_poc.envelope_iou(a, b) == pytest.approx(1 / 3)
    assert arm_poc.envelope_iou(a, a) == pytest.approx(1.0)


def test_boundary_discrepancy_nested_squares():
    inner = box(0.1, 0.1, 0.9, 0.9)
    outer = box(0.0, 0.0, 1.0, 1.0)
    # estimate = inner, truth = outer: pure under-estimate of 0.1 at edges
    result = arm_poc.boundary_discrepancy(inner, outer, step=0.01)
    assert result["under_estimate_max_m"] == pytest.approx(0.141, abs=0.005)
    assert result["over_estimate_max_m"] == 0.0
    # estimate = outer, truth = inner: pure over-estimate
    result = arm_poc.boundary_discrepancy(outer, inner, step=0.01)
    assert result["under_estimate_max_m"] == 0.0
    assert result["over_estimate_max_m"] == pytest.approx(0.141, abs=0.005)


def test_planview_extremes_matches_full_set_on_rules():
    rng = np.random.default_rng(11)
    points = rng.uniform(-1.5, 1.5, (500, 2))
    zone = box(-1.0, -1.0, 1.0, 1.0)
    vertices = arm_poc.planview_extremes(points)
    assert len(vertices) < 30
    assert arm_poc.signed_outside_zone(vertices, zone) == pytest.approx(
        arm_poc.signed_outside_zone(points, zone)
    )


def test_hull_polygon_rejects_degenerate():
    with pytest.raises(ValueError):
        arm_poc.hull_polygon(np.array([[0.0, 0.0], [1.0, 1.0]]))


# ------------------------------------------------------------------- rules
def test_banded_exit_verdict():
    band = arm_poc.BAND_M
    assert arm_poc.banded_exit_verdict(band + 0.01) == "FAIL"
    assert arm_poc.banded_exit_verdict(-band - 0.01) == "PASS"
    assert arm_poc.banded_exit_verdict(0.0) == "NEEDS_REVIEW"
    assert arm_poc.banded_exit_verdict(band) == "NEEDS_REVIEW"


def test_signed_outside_zone():
    zone = box(0, 0, 2, 2)
    inside = np.array([[1.0, 1.0]])
    outside = np.array([[3.0, 1.0]])
    both = np.array([[1.0, 1.0], [3.0, 1.0]])
    assert arm_poc.signed_outside_zone(inside, zone) == pytest.approx(-1.0)
    assert arm_poc.signed_outside_zone(outside, zone) == pytest.approx(1.0)
    assert arm_poc.signed_outside_zone(both, zone) == pytest.approx(1.0)


def test_confusion_counts_silent_misses():
    est = ["PASS", "FAIL", "NEEDS_REVIEW", "NO_DATA", "PASS"]
    ref = ["PASS", "FAIL", "PASS", "PASS", "FAIL"]
    result = arm_poc.confusion(est, ref)
    assert result["n_observed"] == 4
    assert result["agreement"] == pytest.approx(0.5)
    assert result["silent_misses"] == 1
    assert result["pairs"]["est_PASS|gt_FAIL"] == 1


# ---------------------------------------------------------- depth machinery
def test_depth_roundtrip_synthetic_cloud():
    width, height, fov_x = 64, 48, 60.0
    fx = (width / 2.0) / np.tan(np.radians(fov_x) / 2.0)
    rng = np.random.default_rng(7)
    zs = rng.uniform(0.5, 2.0, 200)
    us = rng.integers(2, width - 2, 200)
    vs = rng.integers(2, height - 2, 200)
    cloud = np.column_stack(
        [(us - width / 2.0) * zs / fx, (vs - height / 2.0) * zs / fx, zs]
    )
    depth = arm_poc.cloud_to_depth(cloud, fov_x, width, height)
    mask = np.isfinite(depth)
    points = arm_poc.depth_to_points(depth, mask, fov_x, stride=1)
    assert len(points) == mask.sum()
    # every reconstructed point sits on an input ray with the input depth
    reconstructed_z = np.sort(points[:, 2])
    original_z = np.sort(depth[mask])
    assert np.allclose(reconstructed_z, original_z, atol=1e-3)


def test_cloud_to_depth_flips_gl_convention():
    width, height, fov_x = 32, 32, 60.0
    cloud_cv = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 1.5]])
    cloud_gl = cloud_cv * np.array([1.0, -1.0, -1.0])
    depth_cv = arm_poc.cloud_to_depth(cloud_cv, fov_x, width, height)
    depth_gl = arm_poc.cloud_to_depth(cloud_gl, fov_x, width, height)
    assert np.array_equal(
        np.isfinite(depth_cv), np.isfinite(depth_gl)
    ) and np.allclose(
        depth_cv[np.isfinite(depth_cv)], depth_gl[np.isfinite(depth_gl)]
    )


def test_parse_ply_binary_and_ascii():
    xyz = np.array([[0.1, -0.2, 0.3], [1.0, 2.0, 3.0]], dtype=np.float32)
    header = (
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        b"end_header\n"
    )
    body = b""
    for row in xyz:
        body += row.tobytes() + bytes([255, 0, 0])
    parsed = arm_poc.parse_ply_xyz(header + body)
    assert np.allclose(parsed, xyz, atol=1e-6)
    ascii_data = (
        b"ply\nformat ascii 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
        b"0.1 -0.2 0.3\n1.0 2.0 3.0\n"
    )
    assert np.allclose(arm_poc.parse_ply_xyz(ascii_data), xyz, atol=1e-6)


def test_erode_mask_shrinks_but_keeps_interior():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    eroded = arm_poc.erode_mask(mask, pixels=2)
    assert eroded.sum() == 4  # 6x6 -> 2x2 after two erosions
    assert eroded[4:6, 4:6].all()


def test_skeleton_residual_and_scale_fit_recovers_planted_scale():
    origins = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    rng = np.random.default_rng(3)
    on_skeleton = []
    for a, b in zip(origins, origins[1:]):
        for s in rng.uniform(0, 1, 60):
            on_skeleton.append(a + s * (b - a))
    on_skeleton = np.array(on_skeleton)
    assert arm_poc.skeleton_residual(on_skeleton, origins) < 1e-9
    cam = np.array([2.0, 2.0, 1.0])
    shrunk = cam + (on_skeleton - cam) / 1.6  # range scale error of 1.6x
    fit = arm_poc.fit_fk_scale({0: shrunk}, {0: origins}, cam)
    assert fit["scale"] == pytest.approx(1.6, abs=0.01)
    assert fit["residual_scaled_m"] < fit["residual_raw_m"]


# -------------------------------------------------------------- budget gate
def test_sam_stage_dry_run_exits_2(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    import PIL.Image

    PIL.Image.new("RGB", (8, 8)).save(frames / "f000000.jpg")
    with pytest.raises(SystemExit) as excinfo:
        arm_poc.sam_stage(frames, tmp_path / "cache", [0], live=False)
    assert excinfo.value.code == 2


def test_moge_stage_dry_run_exits_2(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        arm_poc.moge_stage(frames, tmp_path / "cache", [0], live=False)
    assert excinfo.value.code == 2
