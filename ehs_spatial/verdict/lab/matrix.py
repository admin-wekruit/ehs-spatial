"""Benchmark items x config files: each (config, item) is one independent run in its own process (no shared state beyond the
runs directory: one run dir per task, one ledger line each)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ehs_spatial.verdict.lab import config as config_mod
from ehs_spatial.verdict.lab import runner


def _task(args: tuple) -> str:
    config_path, benchmark, item_id, runs_dir, run_id = args
    cfg = config_mod.load(config_path)
    cfg["benchmark"], cfg["item"] = str(benchmark), item_id
    return runner.run(cfg, runs_dir, run_id)


def run(configs: list[Path], runs_dir: Path, run_id: str, benchmark: Path | None = None, jobs: int = 4) -> list[str]:
    """Run ids are <run_id>-<config stem>-<item id>; `benchmark` overrides the configs' own."""
    tasks = []
    for path in configs:
        bench = Path(benchmark or config_mod.load(path)["benchmark"])
        tasks += [(Path(path), bench, str(item["id"]), Path(runs_dir), f"{run_id}-{Path(path).stem}-{item['id']}") for item in runner.items(bench)]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_task, tasks))
