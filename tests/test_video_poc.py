"""Offline tests for the video POC: KRTD lift, banded rules, GT parsing.

No network, no video, no tracker dependency (those imports are lazy in the
script); everything here runs on synthetic data.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

SCRIPT = Path(__file__).parents[1] / "scripts" / "video_poc.py"
SPEC = importlib.util.spec_from_file_location("video_poc", SCRIPT)
video_poc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(video_poc)


def _synthetic_camera():
    """Camera at (0, 0, 5) looking down 45 degrees at the ground plane."""
    forward = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    C = np.array([0.0, 0.0, 5.0])
    T = -R @ C
    K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
    dist = np.array([-0.3, 0.1, 0.0, 0.0, 0.0])
    return K, R, T, dist


def test_parse_krtd_shapes_and_values():
    text = """
    1500.4 0 998.2
    0 1068.2 519.6
    0 0 1

    1 0 0
    0 1 0
    0 0 1

    25.4 -42.9 81.3

    -0.39 0.13 0 0 0
    """
    K, R, T, dist = video_poc.parse_krtd(text)
    assert K[0, 0] == 1500.4 and K[1, 2] == 519.6
    assert np.allclose(R, np.eye(3))
    assert np.allclose(T, [25.4, -42.9, 81.3])
    assert np.allclose(dist, [-0.39, 0.13, 0.0, 0.0, 0.0])


def test_parse_krtd_rejects_tangential_distortion():
    values = list(range(1, 22)) + [0.1, 0.0, 0.2, 0.0, 0.0]
    with pytest.raises(ValueError):
        video_poc.parse_krtd(" ".join(str(v) for v in values))


def test_raycast_roundtrip_recovers_ground_points():
    K, R, T, dist = _synthetic_camera()
    for world in [(0.5, 4.0), (-1.2, 6.0), (2.0, 8.5), (0.0, 5.0)]:
        u, v = video_poc.project_ground_point(world, K, R, T, dist)
        lifted = video_poc.lift_pixel(u, v, K, R, T, dist)
        assert lifted is not None
        assert abs(lifted[0] - world[0]) < 1e-6
        assert abs(lifted[1] - world[1]) < 1e-6


def test_lift_rejects_rays_missing_the_ground():
    K, R, T, _ = _synthetic_camera()
    no_dist = np.zeros(5)
    # A pixel far above the principal point looks up over the horizon.
    # (Zero distortion: the radial model only holds inside the image.)
    assert video_poc.lift_pixel(640.0, -20000.0, K, R, T, no_dist) is None


def test_banded_verdict_min_separation():
    band = video_poc.BAND_M
    check = lambda value: video_poc.banded_verdict(value, 2.0, band, fail_low=True)
    assert check(1.0) == video_poc.FAIL
    assert check(2.0) == video_poc.REVIEW
    assert check(2.0 + band + 0.01) == video_poc.PASS
    assert check(2.0 - band + 0.01) == video_poc.REVIEW


def test_banded_verdict_max_speed():
    check = lambda value: video_poc.banded_verdict(value, 1.5, 0.7, fail_low=False)
    assert check(2.5) == video_poc.FAIL
    assert check(0.5) == video_poc.PASS
    assert check(1.6) == video_poc.REVIEW


def test_zone_verdict_banded():
    zone = box(0.0, 0.0, 10.0, 10.0)
    band = 0.35
    assert video_poc.zone_verdict(zone, (5.0, 5.0), band) == video_poc.FAIL
    assert video_poc.zone_verdict(zone, (5.0, 20.0), band) == video_poc.PASS
    assert video_poc.zone_verdict(zone, (5.0, 10.2), band) == video_poc.REVIEW
    assert video_poc.zone_verdict(zone, (5.0, 9.8), band) == video_poc.REVIEW


def test_parse_meva_lines_and_gt_loading(tmp_path):
    (tmp_path / "clip.types.yml").write_text(
        "- {'meta': 'header'}\n"
        "- {'types': {'cset3': {'person': 1.0}, 'id1': 7}}\n"
        "- {'types': {'cset3': {'vehicle': 1.0}, 'id1': 8}}\n"
        "- {'types': {'cset3': {'bag': 1.0}, 'id1': 9}}\n"
    )
    (tmp_path / "clip.geom.yml").write_text(
        "- {'geom': {'g0': '10 20 30 60', 'id0': 1, 'id1': 7, 'ts0': 100}}\n"
        "- {'geom': {'g0': '50 20 90 40', 'id0': 2, 'id1': 8, 'ts0': 100}}\n"
        "- {'geom': {'g0': '11 21 31 61', 'id0': 3, 'id1': 9, 'ts0': 100}}\n"
        "- {'geom': {'g0': '12 22 32 62', 'id0': 4, 'id1': 7, 'ts0': 130}}\n"
    )
    classes, frames = video_poc.load_gt_boxes(tmp_path, "clip")
    assert classes == {7: "person", 8: "vehicle", 9: "bag"}
    assert sorted(frames) == [100, 130]

    tracks = video_poc.gt_to_tracks(classes, frames, [100, 130])
    assert set(tracks) == {"person-gt7", "vehicle-gt8"}  # bag excluded
    assert tracks["person-gt7"] == [
        (100, (10.0, 20.0, 30.0, 60.0)),
        (130, (12.0, 22.0, 32.0, 62.0)),
    ]


def test_judge_timelines_on_synthetic_tracks():
    zone = box(0.0, 0.0, 4.0, 4.0)
    frame_ids = [0, 30, 60]
    lifted = {
        # person walks from far outside the zone to deep inside it, fast
        "person-1": {
            0: {"xy": (10.0, 10.0)},
            30: {"xy": (8.0, 8.0)},
            60: {"xy": (2.0, 2.0)},
        },
        "vehicle-1": {
            0: {"xy": (10.0, 0.0), "edge": ((9.0, 0.0), (11.0, 0.0))},
            30: {"xy": (10.0, 0.0), "edge": ((9.0, 0.0), (11.0, 0.0))},
        },
    }
    judged = video_poc.judge(lifted, zone, frame_ids, fps=30.0)
    assert judged["R1_zone"] == {0: "PASS", 30: "PASS", 60: "FAIL"}
    # frame 0: person (10,10) to edge segment y=0 -> 10 m -> PASS;
    # frame 60: vehicle gone -> NO_DATA
    assert judged["R2_min_distance"][0] == "PASS"
    assert judged["R2_min_distance"][60] == "NO_DATA"
    # frame 30 speed: sqrt(8)/1s = 2.83 > 1.5 + 0.7 -> FAIL
    assert judged["R3_speed"] == {0: "NO_DATA", 30: "FAIL", 60: "FAIL"}
    assert judged["R2_values_m"][0] == pytest.approx(10.0, abs=0.01)


def test_agreement_counts_and_separates_no_data():
    frame_ids = [0, 30, 60, 90]
    estimated = {0: "PASS", 30: "FAIL", 60: "PASS", 90: "PASS"}
    reference = {0: "PASS", 30: "FAIL", 60: "NO_DATA", 90: "FAIL"}
    stats = video_poc.agreement(estimated, reference, frame_ids)
    assert stats["rate"] == 0.5
    assert stats["n_both_observed"] == 3
    assert stats["rate_both_observed"] == pytest.approx(2 / 3, abs=0.01)
    assert [d["frame"] for d in stats["disagreements"]] == [60, 90]


def test_sam_stage_offline_gate_blocks_spend(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        video_poc.sam_stage(tmp_path, tmp_path / "cache", [1, 2], live=False)
    assert excinfo.value.code == 2
    assert not list((tmp_path / "cache").glob("*.json"))


def test_ate_stats_nearest_neighbour():
    frame_ids = [0]
    estimated = {"person-1": {0: {"xy": (1.0, 1.0)}}}
    reference = {
        "person-gt1": {0: {"xy": (1.3, 1.4)}},
        "person-gt2": {0: {"xy": (9.0, 9.0)}},
    }
    stats = video_poc.ate_stats(estimated, reference, frame_ids)
    assert stats["person"]["median_m"] == pytest.approx(0.5, abs=0.001)
    assert stats["person"]["gt_points_missed_beyond_gate"] == 1
    assert stats["person"]["est_points_beyond_gate"] == 0
    assert stats["vehicle"]["median_m"] is None


def test_ate_stats_gates_unannotated_objects():
    # An estimate far from every GT point (e.g. a real but unannotated parked
    # car) is counted, not scored as position error.
    estimated = {"vehicle-1": {0: {"xy": (80.0, 80.0)}}}
    reference = {"vehicle-gt1": {0: {"xy": (0.0, 0.0)}}}
    stats = video_poc.ate_stats(estimated, reference, [0])
    assert stats["vehicle"]["median_m"] is None
    assert stats["vehicle"]["est_points_beyond_gate"] == 1
    assert stats["vehicle"]["gt_points_missed_beyond_gate"] == 1
