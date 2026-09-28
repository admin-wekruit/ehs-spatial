"""The fixes the first Lightning one-shot run and its independent check asked for (2026-09-28). CPU only, no Modal call:
synthetic inputs, plus the cached Lightning runs read-only where they exist."""
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "modal_apps")]
from report_runner import decide, stages  # noqa: E402
from test_report_runner_stages import RESEARCH, SITES, build, ctx, flag  # noqa: E402

RUNS = stages.art() / "runs"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


# ---------------------------------------------------------------- 1. the inferred floor: never the untested convex outline
def test_floor_without_a_dense_map_is_withheld_with_its_reason(tmp_path):
    out = tmp_path / "floor"
    done = subprocess.run([sys.executable, str(REPO / "scripts/infer_room_floor.py"), "--fused", str(tmp_path / "no-such-run"), "--output", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    record = json.loads((out / "inferred-floor.json").read_text())
    assert record["kind"] == "inferred_floor_withheld" and "dense map" in record["reason"]
    assert not (out / "inferred-floor.glb").exists(), "the convex outline needs --legacy-convex"


def test_the_floor_decision_keeps_only_a_dense_validated_floor_on_a_verified_plane(tmp_path):
    model = {"kind": "inferred_floor_model", "basis": "b", "area_m2": 463.2}
    lens = lambda status: write(tmp_path / status / "lens.json", {"value": {"scale_status": status}})
    convex = write(tmp_path / "convex/inferred-floor.json", model)
    dense = write(tmp_path / "dense/inferred-floor.json", {**model, "dense": {"validation": {"closeM": .5}}})
    withheld = write(tmp_path / "withheld/inferred-floor.json", {"kind": "inferred_floor_withheld", "reason": "no validated dense map"})
    assert decide.inferred_floor(convex)["value"] is False, "the Lightning floor: a convex outline, never tested"
    assert decide.inferred_floor(dense, lens("intrinsics_uncertain"))["value"] is False, "Lightning's plane failed the lens gate (0.835 < 0.90)"
    assert decide.inferred_floor(withheld)["value"] is False and "dense" in decide.inferred_floor(withheld)["evidence"]["withheld"]
    assert decide.inferred_floor(dense, lens("assumed_camera_height"))["value"] is True and decide.inferred_floor(dense)["value"] is True


def test_the_runner_never_asks_for_the_convex_outline():
    values = {**SITES["walmart"][2], "dense_gate": {"use": None, "dir": None}}
    by = build("walmart", decisions=values)
    assert flag(by["floor_infer"], "--dense") is None and flag(by["floor_infer"], "--legacy-convex") is None
    assert "lens=@lens_gate:decision" in by["inferred_floor"].commands[0] and by["inferred_floor"].inputs["lens_gate"] == ("lens_gate", ())
    assert by["floor_infer"].version == 2
    delivered = build("walmart", profile=types.SimpleNamespace(**{**vars(RESEARCH), "name": "delivered", "omit": ("static_filter", "lens_gate")}))
    assert delivered["floor_infer"].version == 1 and "lens_gate" not in delivered["inferred_floor"].inputs, \
        "the delivered profile keys its adopted runs under the versions they were adopted with"
