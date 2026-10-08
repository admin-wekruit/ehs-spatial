"""panoptes verdict {plugins | run | matrix | scorecard}: the lab's command line (also `python -m ehs_spatial.verdict.lab.cli`).
Only argparse at import time: ehs_spatial.cli registers this subcommand without needing the `verdict` extra installed."""
from __future__ import annotations

import argparse
from pathlib import Path


def add_arguments(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="verdict_command", required=True)
    sub.add_parser("plugins", help="registered plugins per layer")
    r = sub.add_parser("run", help="one config x one benchmark item -> runs/<run_id>/")
    r.add_argument("--config", required=True, help="run YAML (see ehs_spatial/verdict/configs/baseline.yaml)")
    r.add_argument("--item", help="benchmark item id (default: the first of items.yaml)")
    m = sub.add_parser("matrix", help="benchmark items x configs, one process per task, then the scorecard")
    m.add_argument("--configs", nargs="+", required=True)
    m.add_argument("--jobs", type=int, default=4)
    for q in (r, m):
        q.add_argument("--benchmark", help="override the configs' benchmark directory")
        q.add_argument("--runs-dir", default="runs")
        q.add_argument("--run-id", help="default: <YYYYmmdd-HHMMSS>-<git short sha>")
    sub.add_parser("scorecard", help="scorecard.md / scorecard.json from the ledger").add_argument("--runs-dir", default="runs")


def dispatch(args: argparse.Namespace) -> None:
    from ehs_spatial.verdict import plugins
    from ehs_spatial.verdict.lab import config, matrix, runner, scorecard

    if args.verdict_command == "plugins":
        for layer, names in plugins.available().items():
            print(f"{layer}: {' '.join(names) or '-'}")
    elif args.verdict_command == "run":
        cfg = config.load(args.config)
        if args.benchmark:
            cfg["benchmark"] = str(Path(args.benchmark).resolve())
        if args.item:
            cfg["item"] = args.item
        print(runner.run(cfg, args.runs_dir, args.run_id))
    elif args.verdict_command == "matrix":
        bench = Path(args.benchmark).resolve() if args.benchmark else None
        for run_id in matrix.run([Path(c) for c in args.configs], Path(args.runs_dir), args.run_id or runner.new_run_id(), bench, args.jobs):
            print(run_id)
        print(scorecard.write(args.runs_dir))
    else:
        print(scorecard.write(args.runs_dir))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="panoptes verdict", description=__doc__)
    add_arguments(p)
    dispatch(p.parse_args(argv))


if __name__ == "__main__":
    main()
