import json
import os
from pathlib import Path

import pytest

from ehs_spatial.eval_pack import generate_eval_pack, run_live_benchmark_case


pytestmark = pytest.mark.skipif(
    os.getenv("EHS_LIVE_BENCHMARK") != "1",
    reason="set EHS_LIVE_BENCHMARK=1 to authorize paid provider calls",
)

CASE_IDS = (
    "ladder_050",
    "ladder_070",
    "platform_inside",
    "fence_occluded",
)
SELECTED_CASES = tuple(
    case_id.strip()
    for case_id in os.getenv("EHS_BENCHMARK_CASES", "ladder_050").split(",")
    if case_id.strip()
)


@pytest.fixture(scope="module")
def live_pack() -> Path:
    root = Path(os.getenv("EHS_BENCHMARK_PACK", "outputs/ehs_v1")).resolve()
    if not (root / "manifest.json").is_file():
        generate_eval_pack(root)
    return root


@pytest.mark.parametrize("case_id", SELECTED_CASES)
def test_paid_provider_chain_matches_calibrated_case(live_pack, case_id):
    assert case_id in CASE_IDS, f"unknown EHS_BENCHMARK_CASES value: {case_id}"
    missing = [
        name
        for name in ("REPLICATE_API_TOKEN", "FAL_KEY", "GEMINI_API_KEY")
        if not os.getenv(name)
    ]
    assert not missing, f"missing provider variables: {missing}"

    report = run_live_benchmark_case(live_pack, case_id)

    assert "error" not in report, json.dumps(report, indent=2)
    assert report["passed"] is True, json.dumps(report, indent=2)
    assert report["actual_status"] == report["expected_status"]
    assert report["fact_grounding_passed"] is True
    assert all(
        (live_pack / relative_path).is_file()
        and (live_pack / relative_path).stat().st_size > 0
        for relative_path in report["provider_artifacts"].values()
    )

    if report["expected_status"] != "INSUFFICIENT_EVIDENCE":
        assert report["threshold_side_matches"] is True
        assert report["entity_evidence_passed"] is True
    else:
        assert report["threshold_side_matches"] is None
    if case_id == "ladder_050":
        assert report["chat"]["grounded"] is True
        assert report["chat"]["fact_ids"]
