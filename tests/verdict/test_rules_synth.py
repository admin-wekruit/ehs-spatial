"""Sanity of the synthetic cell builder (synth/scenes.py) and synthetic-facts@1 (synth/facts.py)."""
from __future__ import annotations

import numpy as np
import pytest

from ehs_spatial.verdict.layers.l6_engine.python import enclosure
from ehs_spatial.verdict.synth import facts as F
from ehs_spatial.verdict.synth import scenes


def fact(fx, pred, *args):
    return next(f for f in fx.facts if f.pred == pred and f.args == list(args))


def test_cell_layout_and_facts():
    sc = scenes.cell(fence_height_m=2.0, floor_gap_m=0.1, robot_fixed_gap_m=1.0, lc_bottom_m=0.25)
    assert [o.id for o in sc.objects] == ["robot", "fence_e", "fence_w", "fence_n", "fence_s", "bollard_1", "bollard_2", "lc", "cart"]
    assert all(o.axes == scenes.AXES for o in sc.objects) and sc.scale_rel_unc == 0.0
    fx = F.facts_of(sc)
    assert fx.producer == "synthetic-facts@1" and fx.signature_version == "1"
    for fence in ("fence_e", "fence_w", "fence_n", "fence_s"):
        assert (fact(fx, "top_height", fence).value, fact(fx, "top_height", fence).u) == (2000, 28)     # U = 2 * sqrt(0.01^2 + 0.01^2) m
        assert (fact(fx, "bottom_height", fence).value, fact(fx, "bottom_height", fence).u) == (100, 20)
        assert fact(fx, "min_distance_3d", "robot", fence).value == 1000
        assert fact(fx, "reach_over", "robot", fence).value == 1000
    assert fact(fx, "bottom_height", "lc").value == 250
    assert fact(fx, "min_distance_3d", "robot", "lc").value > 1500 and fact(fx, "min_distance_3d", "robot", "bollard_1").value > 2000
    assert fact(fx, "z_overlap", "robot", "fence_e").value == 1900 and fact(fx, "horizontal_gap", "robot", "fence_e").value == 1000
    assert fact(fx, "obj", "cart", "cart") is not None
    assert fact(fx, "obj", "robot", "robot").args == ["robot", "robot"]


def test_negative_bottom_and_floor_gap():
    fx = F.facts_of(scenes.cell(floor_gap_m=-0.02))
    assert fact(fx, "bottom_height", "fence_e").value == -20 and fact(fx, "floor_gap", "fence_e").value == 0


def test_enclosure_and_coverage():
    closed, opened = F.facts_of(scenes.cell()), F.facts_of(scenes.cell(enclosed=False, opening_width_m=0.8))
    assert enclosure(closed.grid)[0] == 1 and enclosure(opened.grid)[0] == 0
    nx, ny = closed.grid.shape
    assert len(closed.grid.observed) == nx * ny                                  # 'full' coverage lands on every facts cell
    assert len(closed.grid.blocked) > len(opened.grid.blocked) and closed.grid.hazard == opened.grid.hazard
    none = F.facts_of(scenes.cell(enclosed=False, coverage="none"))
    assert none.grid.observed is None and enclosure(none.grid)[0] is None
    assert scenes.cell(coverage="none").coverage is None


def test_sigma_dict_and_defaults():
    fx = F.facts_of(scenes.cell(sigma_m={"L": 0.02, "W": 0.03, "H": None, "bottom": 0.0}, scale_rel_unc=None))
    f = fact(fx, "top_height", "fence_e")
    assert f.flags == ["default_scale_unc", "default_sigma"] and f.u == F.mm(2 * np.hypot(0.05, 0.02 * 2.0))
    assert fact(fx, "min_distance_3d", "robot", "fence_e").u == F.mm(2 * np.sqrt(2 * 0.03 ** 2 + (0.02 * 1.0) ** 2))


def test_deterministic_and_small():
    sc = scenes.cell()
    assert F.facts_of(sc).model_dump_json() == F.facts_of(sc).model_dump_json()
    nx, ny = F.facts_of(sc).grid.shape
    assert nx * ny < 6000


def test_rejects_rotated_boxes_and_bad_params():
    sc = scenes.cell()
    c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    sc.objects[0].axes = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
    with pytest.raises(ValueError):
        F.facts_of(sc)
    with pytest.raises(ValueError):
        scenes.cell(fence_height_m=0.1, floor_gap_m=0.2)
    with pytest.raises(ValueError):
        scenes.cell(enclosed=False, opening_width_m=0)
