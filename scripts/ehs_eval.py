"""Generate and evaluate the calibrated EHS proof pack."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from ehs_spatial.eval_pack import (
    generate_eval_pack,
    run_live_benchmark_case,
    run_offline_benchmark,
)


PROVIDER_KEYS = ("REPLICATE_API_TOKEN", "FAL_KEY", "GEMINI_API_KEY")
CASE_IDS = (
    "ladder_050",
    "ladder_070",
    "platform_inside",
    "fence_occluded",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate the CPU eval pack")
    generate.add_argument("--output", required=True, type=Path)

    offline = commands.add_parser("offline", help="run oracle geometry eval")
    offline.add_argument("--pack", required=True, type=Path)

    live = commands.add_parser("live", help="run one paid provider case")
    live.add_argument("--pack", required=True, type=Path)
    live.add_argument("--case", required=True, choices=CASE_IDS)
    live.add_argument(
        "--live",
        action="store_true",
        help="explicitly authorize paid provider calls",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        generate_eval_pack(args.output)
        print((args.output / "manifest.json").resolve())
        return 0

    if args.command == "offline":
        report = run_offline_benchmark(args.pack)
        print((args.pack / "offline_report.json").resolve())
        return 0 if report["passed"] else 1

    if not args.live:
        print("live eval requires the explicit --live flag", file=os.sys.stderr)
        return 2
    missing = [name for name in PROVIDER_KEYS if not os.getenv(name)]
    if missing:
        print(
            "live eval is missing provider variables: " + ", ".join(missing),
            file=os.sys.stderr,
        )
        return 2

    report = run_live_benchmark_case(args.pack, args.case)
    print((args.pack / "live_report.json").resolve())
    error = report.get("error")
    if not report["passed"] and isinstance(error, dict):
        message = str(error.get("message", "live provider evaluation failed"))
        for name in PROVIDER_KEYS:
            value = os.getenv(name)
            if value:
                message = message.replace(value, "[REDACTED]")
        layer = " ".join(
            str(error[name])
            for name in ("provider", "operation")
            if error.get(name)
        )
        print(f"{layer + ': ' if layer else ''}{message}", file=os.sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
