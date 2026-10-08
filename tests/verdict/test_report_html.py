"""L7 html@1: one self-contained verdicts.html (nothing external) that names the run id, every rule id, the hotkey legend and the
labels schema; deterministic (two runs byte-identical)."""
from pathlib import Path

from ehs_spatial.verdict.contracts import VerdictSet
from ehs_spatial.verdict.lab import config, runner

CONFIG = Path(__file__).resolve().parents[2] / "ehs_spatial/verdict/configs/html-report.yaml"


def test_html_report_is_self_contained_and_deterministic(tmp_path):
    cfg = config.load(CONFIG)
    cfg["item"] = "090"
    a = tmp_path / "a" / runner.run(dict(cfg), tmp_path / "a", "html-090")
    b = tmp_path / "b" / runner.run(dict(cfg), tmp_path / "b", "html-090")
    html = (a / "L7/verdicts.html").read_text()
    assert html == (b / "L7/verdicts.html").read_text()
    vs = VerdictSet.load(a / "L6/verdicts.json")
    assert vs.verdicts and "html-090" in html and all(v.rule_id in html for v in vs.verdicts)
    assert "verdict-labels/1" in html and "<kbd>Enter</kbd> save + next" in html and "<kbd>1 … 6</kbd>" in html
    assert 'data-f="stop_time_ms"' in html and 'data-f="restricted_space"' in html
    assert "src=" not in html and "<link" not in html and "@import" not in html and "fetch(" not in html
