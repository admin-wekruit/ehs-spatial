"""Small offline check: timestamp uniqueness, pose direction, and no fitted ATE scale."""

import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location("reconstruct_tum_room", Path(__file__).parents[1] / "scripts/reconstruct_tum_room.py")
room = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(room)


def test_tum_control_geometry_contract():
    # Two RGB frames compete for one depth: closest wins, no duplicated depth.
    pairs = room.associate([1., 1.01, 1.04, 2.], [1.009, 1.041], .02)
    assert pairs == [(1, 0), (2, 1)]
    assert room.associate([1., 2.], [1.1, 2.1], .02) == []
    assert room.associate([0.], [.02], .02) == []
    try:
        room.associate([1., 1.], [1.], .02)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate timestamps must be rejected")

    angle = .4
    first = np.eye(4)
    first[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    first[:3, 3] = [2., 1., -.5]
    second = first.copy()
    second[:3, 3] += [.3, -.2, .1]
    previous_to_current = np.linalg.inv(second) @ first
    actual = room.advance_pose(first, previous_to_current)
    np.testing.assert_allclose(actual, second, atol=1e-10)
    point_previous = np.array([.2, .1, 2., 1.])
    np.testing.assert_allclose(actual @ previous_to_current @ point_previous, first @ point_previous, atol=1e-10)
    assert not np.allclose(first @ previous_to_current, second)

    xyz = np.array([[0., 0, 0], [1., 0, 0], [0, 2., 0], [0, 0, 3.], [1., 2., 3.]])
    reference = xyz @ first[:3, :3].T + first[:3, 3]
    exact = room.metric_ate(xyz, reference)
    assert exact["rmse_m"] < 1e-10 and exact["scale"] == 1 and not exact["scale_fit"]
    assert room.metric_ate(xyz * 2, reference)["rmse_m"] > 1


if __name__ == "__main__":
    test_tum_control_geometry_contract()
    print("Passed: unique nearest timestamp association, c2w direction, metric ATE without scale fitting")
