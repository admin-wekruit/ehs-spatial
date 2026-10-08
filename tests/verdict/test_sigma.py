"""The sigma table is what the two benchmark scenes say, and fill_sigma records what it filled."""
import json
from pathlib import Path

import pytest

from ehs_spatial.verdict.contracts import Obj, Scene
from ehs_spatial.verdict.layers.l1_scene.sigma import LEVELS, SIGMA_TABLE_M, UNVERIFIED_MIN_M, calibrate, fill_sigma

OUT = Path(__file__).resolve().parents[2] / "research/verdict-layer-trial-2026-10-07/out"
BENCHMARK = [OUT / "scene-090.json", OUT / "scene-030.json"]


@pytest.mark.skipif(not all(p.exists() for p in BENCHMARK), reason="benchmark scenes not checked out")
def test_table_is_the_calibration_of_the_benchmark_scenes_and_is_deterministic():
    scenes = [json.loads(p.read_text()) for p in BENCHMARK]
    assert calibrate(scenes) == SIGMA_TABLE_M == calibrate(scenes[::-1])


def test_levels_without_data_inherit_the_next_worse_level():
    one = [{"objects": [{"confidence": "low", "sigma_m": {"L": 0.01, "W": None}}]}]
    assert calibrate(one) == {"high": 0.01, "medium": 0.01, "low": 0.01, "unverified": UNVERIFIED_MIN_M}
    assert calibrate([{"objects": []}]) == {level: UNVERIFIED_MIN_M for level in LEVELS}


def obj(oid, confidence, sigma):
    return Obj(id=oid, cls="fence", center_m=[0, 0, 1], axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size_m=[2, 0.1, 2], bottom_m=0, top_m=2,
               sigma_m=sigma, confidence=confidence)


def test_fill_sigma_fills_only_missing_dimensions_and_records_them():
    scene = Scene(scene_id="t", ground_normal=[0, 0, 1], objects=[obj("a", "low", {"L": 0.002, "W": None, "H": None, "bottom": None}),
                                                                   obj("b", "unverified", {})])
    out = fill_sigma(scene)
    assert out.objects[0].sigma_m == {"L": 0.002, "W": SIGMA_TABLE_M["low"], "H": SIGMA_TABLE_M["low"], "bottom": SIGMA_TABLE_M["low"]}
    assert out.objects[1].sigma_m == {dim: SIGMA_TABLE_M["unverified"] for dim in ("L", "W", "H", "bottom")}
    assert json.loads(out.declared_inputs["sigma_filled"]) == ["a:H", "a:W", "a:bottom", "b:H", "b:L", "b:W", "b:bottom"]
    assert scene.objects[0].sigma_m["W"] is None and "sigma_filled" not in scene.declared_inputs      # input untouched
    assert fill_sigma(out).declared_inputs["sigma_filled"] == out.declared_inputs["sigma_filled"]     # idempotent
