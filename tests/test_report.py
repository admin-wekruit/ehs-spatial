import importlib.util
import json
from pathlib import Path

from PIL import Image

from ehs_spatial import report


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "drift_check.py"
SPEC = importlib.util.spec_from_file_location("drift_check_cli", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
drift_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drift_check)


def _synthetic_run(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "run-r1"
    (run / "evidence").mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-r1",
                "created_at": "2026-08-25T10:00:00+00:00",
                "operator": "inspector-a",
                "capture_tier": "mono",
                "providers": {
                    "gemini_model": "gemini-3.5-flash",
                    "code_version": "abc1234",
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "assessment.json").write_text(
        json.dumps(
            {
                "status": "NEEDS_REVIEW",
                "approximate_distance_m": 0.58,
                "distance_error_budget_m": 0.05,
            }
        ),
        encoding="utf-8",
    )
    (run / "policies.json").write_text(
        json.dumps(
            {
                "specs": [
                    {
                        "policy_id": "p01-clearance",
                        "predicate": "min_separation",
                        "threshold": 0.6,
                        "unit": "m",
                        "source_text": "Keep movable equipment 0.6 m clear.",
                    }
                ],
                "results": [
                    {
                        "policy_id": "p01-clearance",
                        "status": "FAIL",
                        "violations": [
                            {
                                "measured": 0.41,
                                "threshold": 0.6,
                                "unit": "m",
                            }
                        ],
                        "warnings": ["single-view evidence"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "scene.json").write_text(
        json.dumps({"warnings": ["floor plane fit is weak"]}),
        encoding="utf-8",
    )
    (run / "review.json").write_text(
        json.dumps(
            {
                "run_id": "run-r1",
                "reviewer": "casey",
                "decision": "overridden",
                "overridden_status": "FAIL",
                "reason": "ladder is actually touching the fence",
                "created_at": "2026-08-25T11:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    for name in ("topdown.png", "plan_view.png", "evidence/frame_0001_overlay.png"):
        Image.new("RGB", (8, 8), "white").save(run / name)
    return run


def test_build_report_html_renders_every_section(tmp_path):
    run = _synthetic_run(tmp_path)

    html = report.build_report_html(run)

    assert "run-r1" in html
    assert "NEEDS_REVIEW" in html
    assert "0.58 m ± 0.05 m" in html
    assert report.DEMO_RULE_COPY in html
    assert "p01-clearance" in html
    assert "min_separation 0.6 m" in html
    assert "Keep movable equipment 0.6 m clear." in html
    assert "0.41 m (limit 0.6 m)" in html
    assert "floor plane fit is weak" in html
    assert "ladder is actually touching the fence" in html
    assert "gemini_model=gemini-3.5-flash" in html
    assert html.count("data:image/png;base64,") == 3


def test_report_demo_rule_copy_matches_the_app():
    from ehs_spatial.app import DEMO_RULE_COPY

    assert report.DEMO_RULE_COPY == DEMO_RULE_COPY


def test_report_handles_bare_list_policies_and_missing_artifacts(tmp_path):
    empty = tmp_path / "runs" / "run-empty"
    empty.mkdir(parents=True)

    html = report.build_report_html(empty)

    assert "run-empty" in html
    assert "No readable assessment.json" in html
    assert "No compiled policies" in html
    assert "No evidence images on disk" in html

    legacy = tmp_path / "runs" / "run-legacy"
    legacy.mkdir(parents=True)
    (legacy / "policies.json").write_text(
        json.dumps([{"policy_id": "p99", "status": "PASS"}]),
        encoding="utf-8",
    )

    assert "p99" in report.build_report_html(legacy)


def test_write_report_and_cli_produce_report_html(tmp_path, capsys):
    run = _synthetic_run(tmp_path)

    destination = report.write_report(run)

    assert destination == run / "report.html"
    assert "run-r1" in destination.read_text(encoding="utf-8")

    assert report.main([str(run)]) == 0
    assert str(destination) in capsys.readouterr().out


def test_report_escapes_artifact_text(tmp_path):
    run = tmp_path / "runs" / "run-x"
    run.mkdir(parents=True)
    (run / "scene.json").write_text(
        json.dumps({"warnings": ["<script>alert(1)</script>"]}),
        encoding="utf-8",
    )

    html = report.build_report_html(run)

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_drift_diff_case_is_silent_when_stable():
    assert drift_check.diff_case("c", "PASS", "PASS", 0.70, 0.72, 0.05) == []
    assert drift_check.diff_case(
        "c", "INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE", None, None, 0.05
    ) == []
    # The pack stores the ground-truth distance even for occluded cases,
    # but an INSUFFICIENT_EVIDENCE expectation means "no distance".
    assert drift_check.diff_case(
        "c", "INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE", 0.5, None, 0.05
    ) == []


def test_drift_diff_case_flags_status_and_distance_drift():
    findings = drift_check.diff_case("c", "PASS", "FAIL", 0.70, 0.50, 0.05)
    assert findings == [
        "c: status drift PASS -> FAIL",
        "c: distance drift 0.700 m -> 0.500 m (|delta| 0.200 m > 0.050 m)",
    ]


def test_drift_diff_case_flags_appearing_and_disappearing_distances():
    assert drift_check.diff_case("c", "PASS", "PASS", None, 0.5, 0.05) == [
        "c: distance appeared (0.500 m) where none was expected"
    ]
    assert drift_check.diff_case("c", "PASS", "PASS", 0.5, None, 0.05) == [
        "c: distance disappeared (expected 0.500 m)"
    ]


def test_drift_diff_report_aggregates_cases_in_order():
    cases = {
        "b_case": {
            "expected_status": "PASS",
            "actual_status": "FAIL",
            "expected_distance_m": 0.7,
            "actual_distance_m": 0.7,
        },
        "a_case": {
            "expected_status": "FAIL",
            "actual_status": "FAIL",
            "expected_distance_m": 0.5,
            "actual_distance_m": 0.5,
        },
    }

    assert drift_check.diff_report(cases, 0.05) == [
        "b_case: status drift PASS -> FAIL"
    ]


def test_drift_check_main_gates_without_a_pack_or_baseline(tmp_path, capsys):
    # No pack at all: both modes refuse before any work.
    assert drift_check.main(["--pack", str(tmp_path / "absent")]) == 2
    assert (
        drift_check.main(["--pack", str(tmp_path / "absent"), "--live"]) == 2
    )

    # A pack without a cached offline baseline refuses --live before any
    # provider call is attempted.
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "manifest.json").write_text("{}", encoding="utf-8")

    assert (
        drift_check.main(
            ["--pack", str(pack), "--case", "ladder_050", "--live"]
        )
        == 2
    )
    assert "offline" in capsys.readouterr().err
