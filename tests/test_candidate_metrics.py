"""Local artifact scoring: known geometry, missing evidence, and replay safety."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "candidate_metrics.py"
SPEC = importlib.util.spec_from_file_location("candidate_metrics", SCRIPT)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def _write(run, relative, value):
    path = run / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _fence(instance, local_centroid, angle=30.0):
    theta = np.radians(30.0)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    centre = rotation @ local_centroid
    alpha = np.radians(angle)
    axes = np.array([[np.cos(alpha), -np.sin(alpha)],
                     [np.sin(alpha), np.cos(alpha)]])
    rect = np.array([[-1, -.05], [1, -.05], [1, .05], [-1, .05]]) @ axes.T + centre
    return {
        "frame": f"frame_{instance + 1:04d}", "label": "safety fence",
        "instance": instance, "refine_slug": None, "guard_chain": 4,
        "footprint_method": "guard-line", "guard_axis_snapped": True,
        "centroid_xy": centre.tolist(), "rect_snapped": rect.tolist(),
        "footprint": rect.tolist(), "footprint_area_m2": .2,
    }


def test_known_rotated_chain_and_enumeration(tmp_path):
    objects = [_fence(0, [0, 0]), _fence(1, [3, .4], angle=31)]
    objects[1]["outside_cell"] = True
    _write(tmp_path, "inventory/inventory.json", {
        "objects": objects, "manhattan_theta_deg": 25,
        "cell_rect": {"sides": {"u_min": {"offset": 0}, "u_max": None}},
    })
    _write(tmp_path, "inventory/phrases.json", ["safety fence", "panel", "crate"])
    _write(tmp_path, "inventory/unresolved.json", [{"phrase": "panel"}])
    result = metrics.collect_metrics(tmp_path)
    geometry = result["geometry"]
    assert geometry["collinearity_max_residual_m"] == pytest.approx(.2)
    assert geometry["parallel_max_spread_deg"] == pytest.approx(1)
    assert geometry["manhattan_max_off_axis_deg"] == pytest.approx(5)
    assert geometry["thin_max_short_side_m"] == pytest.approx(.1)
    assert geometry["chains"][0]["members"][1]["frame_id"] == "frame_0002"
    assert result["inventory"]["outside_cell_count"] == 1
    assert result["inventory"]["cell_side_count"] == 1
    assert result["inventory"]["silently_dropped_phrases"] == ["crate"]
    assert result["fence_footprints"][0]["polygon_area_m2"] == pytest.approx(.2)
    assert result["fence_footprints"][1]["snapped_area_m2"] == pytest.approx(.2)
    assert result["canonical_test_run"] is False
    assert result["existing_test_thresholds"] is None


def test_unscored_is_not_zero_and_duplicate_keys_are_resolved_by_sequence(tmp_path):
    objects = [_fence(0, [0, 0]), _fence(0, [3, 0])]
    objects[1]["frame"] = "frame_0002"
    _write(tmp_path, "inventory/inventory.json", {"objects": objects})
    _write(tmp_path, "inventory/reprojection.json", {"scores": [
        {"label": "safety fence", "instance": 0, "status": "base-occluded"},
        {"label": "safety fence", "instance": 0, "status": "unprojectable"},
    ]})
    reprojection = metrics.collect_metrics(tmp_path)["reprojection"]
    assert reprojection["worst_frac"] is None
    assert reprojection["median_frac"] is None
    assert reprojection["scored_count"] == 0
    assert reprojection["status_counts"] == {"base-occluded": 1, "unprojectable": 1}
    assert [s["key"]["frame_id"] for s in reprojection["scores"]] == [
        "frame_0001", "frame_0002",
    ]
    # A partial score list cannot safely recover one of two identical IDs.
    _write(tmp_path, "inventory/reprojection.json", {"scores": [
        {"label": "safety fence", "instance": 0, "mean_dv_frac": .15},
    ]})
    reprojection = metrics.collect_metrics(tmp_path)["reprojection"]
    assert reprojection["worst_frac"] == .15
    assert reprojection["scores"][0]["key"]["frame_id"] is None
    assert len(reprojection["scores"][0]["candidate_entity_keys"]) == 2


def test_missing_artifacts_are_null(tmp_path):
    result = metrics.collect_metrics(tmp_path)
    assert result["geometry"]["collinearity_max_residual_m"] is None
    assert result["reprojection"]["scored_count"] is None
    assert result["inventory"]["object_count"] is None
    assert result["inventory"]["silently_dropped_phrases"] is None
    assert result["fence_footprints"] is None
    assert result["scale"]["ratios"] is None
    assert result["ground_truth"] is None
    assert result["latency_s"] is None
    assert result["cost_usd"] is None


def test_anchor_ratio_and_cli_do_not_modify_run(tmp_path):
    run = tmp_path / "run"
    frame = run / "geometry/frames/frame_0001"
    frame.mkdir(parents=True)
    np.save(frame / "pts3d.npy", np.tile([0., 0., 2.], (10, 10, 1)))
    np.save(frame / "valid_mask.npy", np.ones((10, 10), bool))
    np.save(frame / "camera_to_world.npy", np.eye(4))
    _write(run, "geometry/moge/frame_0001.json", {"moge_median_range_m": 4.})
    _write(run, "scene.json", {
        "scale_source": "moge_anchor", "scale_factor": 2., "scale_confidence": .5,
    })
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    output = tmp_path / "metrics.json"
    assert metrics.main(["--run", str(run), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["scale"]["ratios"] == [2.]
    assert result["scale"]["confidence"] == .5
    assert result["scale"]["ground_truth_error"] is None
    assert before == {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="outside the run"):
        metrics.main(["--run", str(run), "--output", str(run / "metrics.json")])
