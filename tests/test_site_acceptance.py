"""Offline fixture tests for the site-acceptance protocol: scoring math,
schema, tier bands, and the footprint-distance semantic — no network."""

import importlib.util
import json
from pathlib import Path

import pytest

from ehs_spatial.contracts import Entity3D
from ehs_spatial.rules import ERROR_BUDGET_MONO_M, ERROR_BUDGET_MULTIVIEW_M

SCRIPT = Path(__file__).parents[1] / "scripts" / "site_acceptance.py"
SPEC = importlib.util.spec_from_file_location("site_acceptance", SCRIPT)
site_acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(site_acceptance)


TRUTH = {
    "camera_height_m": 1.5,
    "pairs": [
        {
            "name": "pallet-to-fence",
            "measured_m": 1.20,
            "subject_label": "pallet",
            "object_label": "safety fence",
        },
        {
            "name": "crate-to-fence",
            "measured_m": 2.00,
            "subject_label": "crate",
            "object_label": "safety fence",
        },
    ],
}


def _entity(entity_id: str, label: str, footprint) -> Entity3D:
    return Entity3D(
        entity_id=entity_id,
        label=label,
        observation_ids=["obs-1"],
        centroid_xyz=(0.0, 0.0, 0.5),
        footprint_xy=footprint,
        height_m=1.0,
        evidence_frame_ids=["frame-0"],
    )


SQUARE_AT_0 = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
SQUARE_AT_2 = [(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]


def test_score_pairs_math_and_schema():
    scorecard = site_acceptance.score_pairs(
        TRUTH,
        {"pallet-to-fence": 1.30, "crate-to-fence": 2.50},
        tier="mono",
        scale_source="moge_anchor",
    )
    assert scorecard["tier"] == "mono"
    assert scorecard["pass_band_m"] == ERROR_BUDGET_MONO_M
    first, second = scorecard["pairs"]
    assert first["abs_err_m"] == pytest.approx(0.10)
    assert first["rel_err"] == pytest.approx(0.083)
    assert first["passed"] is True
    assert second["abs_err_m"] == pytest.approx(0.50)
    assert second["passed"] is False  # 0.50 > 0.35 mono band
    assert scorecard["n_pass"] == 1
    assert scorecard["n_total"] == 2
    assert scorecard["passed"] is False
    # Schema the operator-facing table relies on.
    assert set(first) == {
        "name",
        "subject_label",
        "object_label",
        "measured_m",
        "predicted_m",
        "abs_err_m",
        "rel_err",
        "passed",
    }
    json.dumps(scorecard)  # must be JSON-serialisable as-is


def test_score_pairs_multiview_band_is_tighter():
    scorecard = site_acceptance.score_pairs(
        TRUTH,
        {"pallet-to-fence": 1.45, "crate-to-fence": 2.10},
        tier="multiview",
        scale_source="camera_height",
    )
    assert scorecard["pass_band_m"] == ERROR_BUDGET_MULTIVIEW_M
    # 0.25 abs err passes mono (0.35) but fails multiview (0.20).
    assert scorecard["pairs"][0]["passed"] is False
    assert scorecard["pairs"][1]["passed"] is True


def test_score_pairs_missing_prediction_fails():
    scorecard = site_acceptance.score_pairs(
        TRUTH,
        {"pallet-to-fence": None},
        tier="mono",
        scale_source="model_native",
    )
    row = scorecard["pairs"][0]
    assert row["predicted_m"] is None
    assert row["abs_err_m"] is None
    assert row["passed"] is False
    assert scorecard["passed"] is False


def test_pair_distance_uses_footprint_gap():
    entities = [
        _entity("e1", "pallet", SQUARE_AT_0),
        _entity("e2", "safety fence", SQUARE_AT_2),
    ]
    distance, warnings = site_acceptance.pair_distance_m(
        entities, "pallet", "safety fence"
    )
    assert distance == pytest.approx(1.0)
    assert warnings == []


def test_pair_distance_touching_is_zero_and_missing_is_none():
    touching = [
        _entity("e1", "pallet", SQUARE_AT_0),
        _entity("e2", "safety fence", [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
    ]
    distance, _ = site_acceptance.pair_distance_m(touching, "pallet", "safety fence")
    assert distance == 0.0

    distance, warnings = site_acceptance.pair_distance_m(
        touching, "pallet", "crate"
    )
    assert distance is None
    assert "crate" in warnings[0]


def test_pair_distance_multiple_candidates_reports_closest():
    entities = [
        _entity("e1", "pallet", SQUARE_AT_0),
        _entity("e2", "safety fence", SQUARE_AT_2),
        _entity(
            "e3",
            "safety fence",
            [(5.0, 0.0), (6.0, 0.0), (6.0, 1.0), (5.0, 1.0)],
        ),
    ]
    distance, warnings = site_acceptance.pair_distance_m(
        entities, "pallet", "safety fence"
    )
    assert distance == pytest.approx(1.0)
    assert any("multiple candidates" in w for w in warnings)


def test_load_truth_rejects_unknown_label(tmp_path):
    bad = dict(TRUTH)
    bad["pairs"] = [
        {
            "name": "x",
            "measured_m": 1.0,
            "subject_label": "unicorn",
            "object_label": "safety fence",
        }
    ]
    path = tmp_path / "truth.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unicorn"):
        site_acceptance.load_truth(path)


def test_load_truth_accepts_valid_file(tmp_path):
    path = tmp_path / "truth.json"
    path.write_text(json.dumps(TRUTH))
    truth = site_acceptance.load_truth(path)
    assert truth["camera_height_m"] == 1.5
    assert len(truth["pairs"]) == 2
