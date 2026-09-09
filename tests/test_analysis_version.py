"""One generation of analysis across the product: inventories carry the
rule-set version stamp, and the app classifies every run as current /
stale / missing so stale ones get upgraded on open instead of shown."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ehs_spatial.app import analysis_state, current_analysis_version  # noqa: E402


def _run_with_inventory(tmp_path: Path, payload: dict | None) -> Path:
    run = tmp_path / "run-x"
    (run / "inventory").mkdir(parents=True)
    if payload is not None:
        (run / "inventory" / "inventory.json").write_text(json.dumps(payload))
    return run


def test_version_stamp_matches_inventory_module():
    from scene_inventory import ANALYSIS_VERSION

    assert current_analysis_version() == ANALYSIS_VERSION
    assert ANALYSIS_VERSION  # never empty — every inventory must be stamped


def test_missing_inventory_is_missing(tmp_path):
    run = _run_with_inventory(tmp_path, None)
    assert analysis_state(run) == "missing"


def test_unstamped_or_old_inventory_is_stale(tmp_path):
    assert analysis_state(_run_with_inventory(tmp_path, {"objects": []})) == "stale"
    old = _run_with_inventory(tmp_path / "b", {"analysis_version": "0000"})
    assert analysis_state(old) == "stale"


def test_current_inventory_is_current(tmp_path):
    run = _run_with_inventory(
        tmp_path, {"analysis_version": current_analysis_version()}
    )
    assert analysis_state(run) == "current"


def test_corrupt_inventory_is_stale_not_crash(tmp_path):
    run = tmp_path / "run-c"
    (run / "inventory").mkdir(parents=True)
    (run / "inventory" / "inventory.json").write_text("{not json")
    assert analysis_state(run) == "stale"
