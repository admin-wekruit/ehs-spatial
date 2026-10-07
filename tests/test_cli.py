"""panoptes run --dry-run / status against the repo's frozen research data: the steps of run_all.sh, their done / todo
state and their commands, without running anything heavy and without writing a ledger."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ehs_spatial import cli

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/module-swap-2026-10-07/data"


def panoptes(*args, env=None, tmp=None):
    # a private platform / workcell root: S7's done-check looks at $PANOPTES_PLATFORM/.platform/swap-20261007/<variant>/result.json,
    # which exists on the machine that published the reports (and PANOPTES_WORKCELL may point at that checkout)
    extra = {"PANOPTES_PAGES": str(tmp / "pages"), "PANOPTES_PLATFORM": str(tmp / "platform"), "PANOPTES_WORKCELL": str(tmp / "workcell")} if tmp else {}
    return subprocess.run([sys.executable, "-m", "ehs_spatial.cli", *args], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, **extra, **(env or {})})


def states(stdout):
    out = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in cli.STEPS:
            out[(parts[0], parts[1])] = parts[2]
    return out


def test_load_env_sources_env_sh():
    env = cli.load_env()
    assert Path(env["SWAP_SCRATCH"]) == DATA and Path(env["SWAP_NOTES"]) == ROOT / "research/module-swap-2026-10-07/notes"
    assert env["PANOPTES_ONPREM"] == "1" and "run_stage.py" in env["MODAL_RUN"] and env["PY"].endswith("python")


def test_dry_run_lists_steps_against_the_repo_data(tmp_path):
    result = panoptes("run", "--cell", "090", "--dry-run", tmp=tmp_path)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "dry run: nothing is executed" in out and "DONE 090" in out
    s2a, s4a = out.split("S2b mvs")[0], out.split("S4a sam3d")[1].split("S4b")[0]
    assert "done, skipped" in s2a  # checks/clean-gpu/090 is frozen in the repo
    assert "completion_ab.py --stage sam3d" in s4a and "AB_FRAME=all" in s4a and "AB_CELL=090" in s4a  # swap-runs/ is not
    assert "mvs_route.py" not in out  # S2b is done too
    assert "compare.py left_light_curtain" in out and "build_capture_report.py" in out and "compare_layers.py" in out
    assert not (DATA / "swap-runs/090/ledger.json").exists()


def test_status_marks_done_todo_and_always(tmp_path):
    result = panoptes("status", "--cell", "030", tmp=tmp_path)
    assert result.returncode == 0, result.stderr
    s = states(result.stdout)
    assert s[("S2a", "route")] == "done" and s[("S2d", "export-fill")] == "done"
    assert s[("S4a", "sam3d")] == "todo" and s[("S7", "publish")] == "todo"
    assert s[("S2d", "field-values")] == "always" and s[("S8", "checks")] == "always"
    assert len(s) == len(cli.ITEMS)


def test_only_and_from_select_steps(tmp_path):
    only = panoptes("run", "--cell", "090", "--dry-run", "--only", "S4c", tmp=tmp_path).stdout
    assert "S4c generation" in only and "S4c pins" in only and "S4a" not in only and "S5" not in only
    tail = panoptes("run", "--cell", "090", "--dry-run", "--from", "S7", tmp=tmp_path).stdout
    assert "S7 publish" in tail and "S8 checks" in tail and "S6 report" not in tail and "tables_030.py" not in tail
    bad = panoptes("run", "--cell", "090", "--dry-run", "--only", "S99", tmp=tmp_path)
    assert bad.returncode != 0 and "unknown step" in bad.stderr


def test_provider_branches_show_in_the_dry_run(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = {"SWAP_SCRATCH": str(scratch), "GEOMETRY_MVS_BACKEND": "http", "SAM3D_BACKEND": "http"}
    out = panoptes("run", "--cell", "090", "--dry-run", "--only", "S2a", env=env, tmp=tmp_path).stdout
    assert "providers.geometry_mvs.run('090'" in out and "GEOMETRY_MVS_BACKEND=http" in out and "prod_route_modal.py" not in out
    out = panoptes("run", "--cell", "090", "--dry-run", "--only", "S2a", env={"SWAP_SCRATCH": str(scratch)}, tmp=tmp_path).stdout
    assert "prod_route_modal.py --stage route --cells 090" in out


def test_ctx_drops_weights_flag_without_a_mirror(tmp_path):
    env = cli.load_env()
    env["MODAL_RUN"] = f"{env['PY']} run_stage.py --weights {tmp_path / 'no-mirror'}"
    assert "--weights" not in cli.Ctx("090", env).MODAL_RUN
    (tmp_path / "mirror").mkdir()
    (tmp_path / "mirror/manifest.json").write_text("{}")
    env["MODAL_RUN"] = f"{env['PY']} run_stage.py --weights {tmp_path / 'mirror'}"
    assert cli.Ctx("090", env).MODAL_RUN[-2:] == ["--weights", str(tmp_path / "mirror")]
    with pytest.raises(SystemExit):
        cli.Ctx("091", env)
