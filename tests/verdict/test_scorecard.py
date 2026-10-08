"""Two configs -> two ledger rows; the diff column names the layer whose params changed (L6 k 2 -> 1). The second config composes
the baseline through `extends` and reuses L2 from the first run; gold agreement is reported against benchmark/v0/gold.json."""
import json
from pathlib import Path

from ehs_spatial.verdict.lab import config, runner, scorecard

CONFIGS = Path(__file__).resolve().parents[2] / "ehs_spatial/verdict/configs"


def test_two_rows_and_the_diff_column(tmp_path):
    base = config.load(CONFIGS / "baseline.yaml")
    base["item"] = "090"
    runner.run(base, tmp_path, "sc-1")
    (tmp_path / "k1.yaml").write_text(f"extends: {CONFIGS / 'baseline.yaml'}\nitem: '090'\nL2: {{reuse: sc-1}}\nL6: {{plugin: clingo, k: 1}}\n")
    runner.run(config.load(tmp_path / "k1.yaml"), tmp_path, "sc-2")
    md = scorecard.write(tmp_path)
    rows = [line for line in md.splitlines() if line.startswith("| sc-")]
    assert len(rows) == 2 and rows[0].startswith("| sc-1 |") and rows[1].startswith("| sc-2 |"), md
    data = json.loads((tmp_path / "scorecard.json").read_text())
    assert [r["diff"] for r in data] == ["—", "L6 k 2→1"]
    assert data[0]["gold"] == {"PASS": "10/10", "FAIL": "2/2", "NEEDS_MEASUREMENT": "2/2", "NEEDS_INPUT": "7/7", "CANNOT_DETERMINE": "1/1"}
    assert data[0]["gold_provisional"] is True
    assert (tmp_path / "sc-2/L2/facts.json").read_bytes() == (tmp_path / "sc-1/L2/facts.json").read_bytes()   # reused, copied into the run
