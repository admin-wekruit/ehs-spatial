"""Cell-rectangle premise: a workcell's walls and guard structures form a
closed rectangle in the Manhattan frame. The fit must recover the rectangle
from fanned/noisy boundary evidence, re-seat walls exactly on the fitted
sides, tag far-away fence structure as another cell's, and never invent a
side it has no evidence for."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scene_inventory import (  # noqa: E402
    CELL_SNAP_BAND,
    _apply_cell_rectangle,
    _cell_frame,
    _fit_cell_rectangle,
)

THETA = np.radians(20.0)


def _wall(uv1, uv2, points=2000):
    _, to_xy = _cell_frame(THETA)
    x1, y1 = to_xy(uv1)
    x2, y2 = to_xy(uv2)
    return {
        "start": [x1, y1],
        "end": [x2, y2],
        "length_m": round(float(np.hypot(x2 - x1, y2 - y1)), 2),
        "height_m": 3.0,
        "points": points,
    }


def _fence(uv1, uv2, label="safety fence"):
    _, to_xy = _cell_frame(THETA)
    a, b = to_xy(uv1), to_xy(uv2)
    normal = np.array([-(b[1] - a[1]), b[0] - a[0]])
    normal = 0.05 * normal / np.linalg.norm(normal)
    quad = [list(a), list(b), list(b + normal), list(a + normal)]
    mid = (np.asarray(a) + np.asarray(b)) / 2
    return {
        "label": label,
        "footprint": quad,
        "rect_snapped": quad,
        "centroid_xy": [float(mid[0]), float(mid[1])],
    }


def _machine(uv):
    _, to_xy = _cell_frame(THETA)
    x, y = to_xy(uv)
    return {"label": "robotic arm", "footprint": [], "centroid_xy": [x, y]}


def _closed_cell():
    # 4 x 6 m cell: left/right walls fanned a few degrees, front/back
    # fences slightly off their true offsets — realistic RANSAC output
    walls = [
        _wall((-2.05, 0.1), (-1.95, 5.9)),   # u_min, fanned
        _wall((2.02, 0.0), (1.9, 6.1)),      # u_max, fanned other way
        _wall((-1.8, 6.05), (1.7, 5.95)),    # v_max
    ]
    entries = [
        _fence((-1.6, 0.02), (1.5, -0.03)),  # v_min fence line
        _fence((-1.98, 1.0), (-2.02, 4.0)),  # reinforces u_min
        _machine((0.0, 3.0)),                # the hazard the cell encloses
    ]
    return walls, entries


def test_closed_cell_recovers_rectangle():
    walls, entries = _closed_cell()
    cell = _fit_cell_rectangle(walls, entries, THETA)
    assert cell is not None and cell["corners"] is not None
    du, dv = cell["size_m"]
    assert du == pytest.approx(4.0, abs=0.25)
    assert dv == pytest.approx(6.0, abs=0.25)
    assert all(cell["sides"][k] for k in ("u_min", "u_max", "v_min", "v_max"))


def test_walls_snap_parallel_onto_sides():
    walls, entries = _closed_cell()
    cell = _fit_cell_rectangle(walls, entries, THETA)
    _apply_cell_rectangle(cell, walls, entries, THETA)
    to_uv, _ = _cell_frame(THETA)
    for wall in walls:
        assert wall.get("cell_side"), wall
        (u1, v1), (u2, v2) = to_uv(wall["start"]), to_uv(wall["end"])
        side = cell["sides"][wall["cell_side"]]
        # exactly on the side line (2e-3: endpoints rounded to mm in
        # world coords) -> zero fan by construction
        if wall["cell_side"].startswith("u"):
            assert abs(u1 - side["offset"]) < 2e-3
            assert abs(u2 - side["offset"]) < 2e-3
        else:
            assert abs(v1 - side["offset"]) < 2e-3
            assert abs(v2 - side["offset"]) < 2e-3


def test_far_fence_tagged_outside_cell_not_moved():
    walls, entries = _closed_cell()
    outlier = _fence((-1.5, 12.0), (1.5, 12.0))
    before = [list(p) for p in outlier["footprint"]]
    entries.append(outlier)
    cell = _fit_cell_rectangle(walls, entries, THETA)
    _apply_cell_rectangle(cell, walls, entries, THETA)
    assert outlier.get("outside_cell") is True
    assert outlier["footprint"] == before
    assert not any(e.get("outside_cell") for e in entries[:-1])


def test_open_cell_keeps_missing_side_open():
    walls, entries = _closed_cell()
    # remove the v_min fence -> only 3 sides evidenced
    entries = [entries[1], entries[2]]
    cell = _fit_cell_rectangle(walls, entries, THETA)
    assert cell is not None
    assert cell["sides"]["v_min"] is None
    assert cell["corners"] is None and cell["size_m"] is None


def test_no_theta_or_thin_evidence_yields_none():
    walls, entries = _closed_cell()
    assert _fit_cell_rectangle(walls, entries, None) is None
    assert _fit_cell_rectangle([walls[0]], [], THETA) is None


def test_neighbour_cell_wall_not_swallowed_as_side():
    # a second cell's wall 3 m beyond u_max, weaker support than the true
    # pair -> the fit must keep the tight, well-supported enclosure
    walls, entries = _closed_cell()
    walls.append(_wall((5.0, 1.0), (5.0, 3.2), points=500))
    cell = _fit_cell_rectangle(walls, entries, THETA)
    assert cell["size_m"][0] == pytest.approx(4.0, abs=0.3)
    to_uv, _ = _cell_frame(THETA)
    neighbour_u = 5.0
    for key in ("u_min", "u_max"):
        assert abs(cell["sides"][key]["offset"] - neighbour_u) > 1.0


def test_snap_band_respected():
    walls, entries = _closed_cell()
    stray = _wall((-2.0 - CELL_SNAP_BAND - 0.2, 1.0),
                  (-2.0 - CELL_SNAP_BAND - 0.2, 4.0), points=450)
    walls.append(stray)
    start_before = list(stray["start"])
    cell = _fit_cell_rectangle(walls[:-1], entries, THETA)
    _apply_cell_rectangle(cell, walls, entries, THETA)
    assert stray["start"] == start_before or stray.get("cell_side") is None


def test_second_guard_row_stays_interior_divider():
    # two guard rows both between camera and machine: the far boundary
    # must come from beyond the machine, never from the second row
    walls, entries = _closed_cell()
    entries.append(_fence((-1.4, 1.8), (1.4, 1.8)))   # second row
    cell = _fit_cell_rectangle(walls, entries, THETA)
    assert cell["sides"]["v_min"]["offset"] < 1.0     # front row wins min
    assert cell["sides"]["v_max"]["offset"] > 5.0     # back wall wins max


def test_camera_standing_on_front_boundary_keeps_it():
    # interior fallback = camera origin ON the v=0 wall: the near wall
    # must still be v_min, not vanish into the "above" bucket
    walls = [
        _wall((-2.0, 0.0), (2.0, 0.0)),
        _wall((-2.0, 6.0), (2.0, 6.0)),
        _wall((-2.0, 0.5), (-2.0, 5.5)),
        _wall((2.0, 0.5), (2.0, 5.5)),
    ]
    cell = _fit_cell_rectangle(walls, [], THETA)
    assert cell["sides"]["v_min"] is not None
    assert cell["sides"]["v_min"]["offset"] == pytest.approx(0.0, abs=0.1)
    assert cell["sides"]["v_max"]["offset"] == pytest.approx(6.0, abs=0.1)


def test_robot_named_fence_is_not_the_hazard_anchor():
    walls, entries = _closed_cell()
    entries = [e for e in entries if e["label"] != "robotic arm"]
    front = _fence((-1.6, 0.02), (1.5, -0.03), label="robot safety fence")
    entries.append(front)
    cell = _fit_cell_rectangle(walls, entries, THETA)
    # front fence stays a side; the "robot" in its label must not have
    # anchored the interior onto the boundary itself
    assert cell["sides"]["v_min"] is not None
    assert cell["sides"]["v_min"]["offset"] < 0.5


def test_strong_neighbour_wall_across_aisle_not_swallowed():
    # neighbour's LONG wall 4 m beyond our fence line: credible by
    # support, rejected by the aisle gap
    walls, entries = _closed_cell()
    walls.append(_wall((6.0, 0.0), (6.0, 8.0), points=3000))
    cell = _fit_cell_rectangle(walls, entries, THETA)
    assert cell["sides"]["u_max"]["offset"] == pytest.approx(1.96, abs=0.3)


def test_interior_guard_chain_does_not_bridge_walls():
    # guards every 0.5 m across the cell: single-linkage would weld the
    # two real walls into one cluster; the span split keeps them apart
    walls, entries = _closed_cell()
    for u in np.arange(-1.5, 1.6, 0.5):
        entries.append(_fence((u, 1.0), (u, 3.0), label="machine guard fence"))
    cell = _fit_cell_rectangle(walls, entries, THETA)
    assert cell is not None
    assert cell["sides"]["u_min"] is not None
    assert cell["sides"]["u_min"]["offset"] == pytest.approx(-2.0, abs=0.3)
    assert cell["sides"]["u_max"]["offset"] == pytest.approx(1.96, abs=0.3)


def test_wall_length_recomputed_after_snap():
    walls, entries = _closed_cell()
    cell = _fit_cell_rectangle(walls, entries, THETA)
    _apply_cell_rectangle(cell, walls, entries, THETA)
    for wall in walls:
        actual = float(np.hypot(
            wall["end"][0] - wall["start"][0],
            wall["end"][1] - wall["start"][1],
        ))
        assert wall["length_m"] == pytest.approx(actual, abs=0.02)
