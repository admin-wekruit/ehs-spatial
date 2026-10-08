"""Lab rule 3: same config, same inputs, same run id -> byte-identical outputs. Two runs into two runs directories; everything under
the run directory is compared (layer JSON, program.lp, verdicts.md, config.json). Only the ledger carries a timestamp."""
from pathlib import Path

from ehs_spatial.verdict.lab import config, runner

BASELINE = Path(__file__).resolve().parents[2] / "ehs_spatial/verdict/configs/baseline.yaml"


def files(run_dir: Path) -> dict[str, bytes]:
    return {str(p.relative_to(run_dir)): p.read_bytes() for p in sorted(run_dir.rglob("*")) if p.is_file()}


def test_two_runs_are_byte_identical(tmp_path):
    cfg = config.load(BASELINE)
    cfg["item"] = "090"
    a = files(tmp_path / "a" / runner.run(dict(cfg), tmp_path / "a", "fixed-run"))
    b = files(tmp_path / "b" / runner.run(dict(cfg), tmp_path / "b", "fixed-run"))
    assert a and a.keys() == b.keys(), sorted(a)
    assert [name for name in a if a[name] != b[name]] == []
