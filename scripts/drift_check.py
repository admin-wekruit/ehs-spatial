"""Detect provider drift: same cached inputs, are today's answers the same?

Offline (default, free): replay the calibrated oracle pack through the
deterministic geometry path via run_offline_benchmark, then diff each case's
status and distance against the pack's stored expectations with a tolerance.
The replay spends nothing — it reads the pack's cached pointmaps and masks.

--live (paid, one case): re-run ONE pack case through the real providers via
run_live_benchmark_case and diff today's answer against the cached offline
replay for the same case. Gated the house way: the planned spend is printed
and nothing runs without the explicit flag.

Usage:
  uv run python scripts/drift_check.py --pack outputs/ehs_v1
  uv run --env-file .env python scripts/drift_check.py --pack outputs/ehs_v1 \\
      --case ladder_050 --live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ehs_spatial.eval_pack import run_live_benchmark_case, run_offline_benchmark


CASE_IDS = ("ladder_050", "ladder_070", "platform_inside", "fence_occluded")
# Matches the pack's own offline gate (absolute_error_m <= 0.1): drift means
# exceeding the accuracy the pack was calibrated to, not normal replay noise.
DEFAULT_TOLERANCE_M = 0.1


def diff_case(
    case_id: str,
    expected_status: str | None,
    actual_status: str | None,
    expected_distance_m: float | None,
    actual_distance_m: float | None,
    tolerance_m: float,
) -> list[str]:
    """Pure comparison: human-readable drift findings, [] when stable."""
    findings = []
    if actual_status != expected_status:
        findings.append(
            f"{case_id}: status drift {expected_status} -> {actual_status}"
        )
    if expected_status == "INSUFFICIENT_EVIDENCE":
        # The pack's own rule: an insufficient-evidence verdict must not
        # report a distance, whatever the ground-truth geometry was.
        expected_distance_m = None
    if expected_distance_m is None and actual_distance_m is not None:
        findings.append(
            f"{case_id}: distance appeared ({actual_distance_m:.3f} m) "
            "where none was expected"
        )
    elif expected_distance_m is not None and actual_distance_m is None:
        findings.append(
            f"{case_id}: distance disappeared "
            f"(expected {expected_distance_m:.3f} m)"
        )
    elif (
        expected_distance_m is not None
        and actual_distance_m is not None
        and abs(actual_distance_m - expected_distance_m) > tolerance_m
    ):
        findings.append(
            f"{case_id}: distance drift {expected_distance_m:.3f} m -> "
            f"{actual_distance_m:.3f} m "
            f"(|delta| {abs(actual_distance_m - expected_distance_m):.3f} m "
            f"> {tolerance_m:.3f} m)"
        )
    return findings


def diff_report(cases: dict, tolerance_m: float) -> list[str]:
    """Diff every case of an offline_report.json-shaped mapping against its
    stored expectations."""
    return [
        finding
        for case_id in sorted(cases)
        for finding in diff_case(
            case_id,
            cases[case_id].get("expected_status"),
            cases[case_id].get("actual_status"),
            cases[case_id].get("expected_distance_m"),
            cases[case_id].get("actual_distance_m"),
            tolerance_m,
        )
    ]


def _print_findings(findings: list[str], label: str) -> int:
    if findings:
        print(f"{label}: DRIFT ({len(findings)} finding(s))")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print(f"{label}: stable, no drift")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("outputs/ehs_v1"))
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE_M, metavar="M"
    )
    parser.add_argument("--case", choices=CASE_IDS, default="ladder_050")
    parser.add_argument(
        "--live",
        action="store_true",
        help="explicitly authorize re-running --case through paid providers",
    )
    args = parser.parse_args(argv)

    if not (args.pack / "manifest.json").is_file():
        print(
            f"no eval pack at {args.pack} — generate it first:\n"
            f"  uv run python scripts/ehs_eval.py generate --output {args.pack}",
            file=sys.stderr,
        )
        return 2

    if not args.live:
        report = run_offline_benchmark(args.pack)
        return _print_findings(
            diff_report(report["cases"], args.tolerance), "offline replay"
        )

    offline_path = args.pack / "offline_report.json"
    try:
        cached = json.loads(offline_path.read_text(encoding="utf-8"))
        baseline = cached["cases"][args.case]
    except (OSError, ValueError, KeyError):
        print(
            f"no cached offline baseline for {args.case} in {offline_path} — "
            "run the offline mode first",
            file=sys.stderr,
        )
        return 2

    print(
        f"1 live case ({args.case}): paid MapAnything + SAM + Gemini calls"
    )
    live = run_live_benchmark_case(args.pack, args.case)
    if "error" in live:
        error = live["error"]
        print(
            f"live run failed before comparison: {error.get('provider')} "
            f"{error.get('operation')}: {error.get('message')}",
            file=sys.stderr,
        )
        return 2
    findings = diff_case(
        args.case,
        baseline.get("actual_status"),
        live.get("actual_status"),
        baseline.get("actual_distance_m"),
        live.get("actual_distance_m"),
        args.tolerance,
    )
    return _print_findings(findings, f"live vs cached ({args.case})")


if __name__ == "__main__":
    raise SystemExit(main())
