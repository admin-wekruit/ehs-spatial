"""panoptes: the module-swap pipeline (research/module-swap-2026-10-07/run_all.sh) as a CLI. It supersedes run_all.sh,
which stays unchanged as the reference.

  panoptes run --cell 090|030 [--from STEP] [--only STEP] [--dry-run]
  panoptes status --cell 090

Same steps, same scripts (run through subprocess with env.sh's variables: env.sh is sourced, so PANOPTES_WORKCELL,
SWAP_SCRATCH, MODAL_RUN, PY, ... mean what they mean there), same idempotent skip rules. The GPU stages go through the
providers when a backend is set: S2a through providers.geometry_mvs when GEOMETRY_MVS_BACKEND is set, S4a through
providers.sam3d (inside completion_ab.py) when SAM3D_BACKEND is set (docs/BACKENDS-v1.md); unset = run_all.sh's path.
Every step that ran appends to $SWAP_SCRATCH/swap-runs/<cell>/ledger.json: inputs sha256, start / end, seconds,
backend, model_info when a provider answered, host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
ENV_SH = ROOT / "research/module-swap-2026-10-07/env.sh"
CELLS = {
    "090": dict(
        run="lucida-replica-01",
        objects="left_light_curtain right_light_curtain left_fence right_fence left_post right_post robot cart guard".split(),
        frames=["sam3d", "sam3d-frame_0001", "sam3d-frame_0002", "sam3d-frame_0003"],
        scale="mvs090/estop-scale2.json",
    ),
    "030": dict(
        run="bor1-030-01",
        objects="cart guard left_light_curtain left_post right_fence right_light_curtain right_post robot".split(),
        frames=["sam3d", "sam3d-frame_0001", "sam3d-frame_0002"],
        scale="mvs030/estop-scale.json",
    ),
}
GATE_KEYS = ("housing090R_cm", "fence090R_cm", "housing030L_cm", "housing030R_cm", "estopNativeToMeters090", "estopNativeToMeters030")


def load_env() -> dict:
    """env.sh sourced by the shell (zsh, else bash): every variable as run_all.sh sees it; values already in the
    environment win, as in env.sh itself."""
    shell = shutil.which("zsh") or shutil.which("bash")
    if not shell:
        raise SystemExit("panoptes: zsh or bash is needed to source env.sh")
    env = {**os.environ, "PANOPTES_SERVING": os.environ.get("PANOPTES_SERVING", str(ROOT))}
    out = subprocess.run([shell, "-c", 'source "$1" && env -0', "panoptes", str(ENV_SH)], capture_output=True, env=env, check=True)
    return dict(item.split("=", 1) for item in out.stdout.decode().split("\0") if "=" in item)


class Ctx:
    def __init__(self, cell: str, env: dict, dry_run: bool = False):
        if cell not in CELLS:
            raise SystemExit(f"cell: {' | '.join(CELLS)}")
        self.cell, self.env, self.dry = cell, env, dry_run
        self.SP, self.N = Path(env["SWAP_SCRATCH"]), Path(env["SWAP_NOTES"])
        self.SWAP_ROOT, self.SERVING = Path(env["SWAP_ROOT"]), Path(env["PANOPTES_SERVING"])
        self.WORKCELL, self.PLATFORM, self.RUNS = Path(env["PANOPTES_WORKCELL"]), Path(env["PANOPTES_PLATFORM"]), Path(env["PANOPTES_RUNS"])
        self.PY, self.PG = env["PY"], Path(env["PG"])
        self.MODAL_RUN = shlex.split(env["MODAL_RUN"])
        if "--weights" in self.MODAL_RUN:  # ponytail: a CPU-only flow machine (every GPU model behind http) has no weights mirror;
            i = self.MODAL_RUN.index("--weights")  # run_stage.py would refuse to start without one, so the flag is dropped here
            if not (Path(self.MODAL_RUN[i + 1]) / "manifest.json").exists():
                del self.MODAL_RUN[i : i + 2]
        c = CELLS[cell]
        self.RUNNAME, self.OBJS, self.FRAMES = c["run"], c["objects"], c["frames"]
        self.FILLX = self.SP / f"checks/bbab-export-{cell}-mvs-fill"
        self.AB = self.SP / f"swap-runs/{cell}/mvs-fill-ab"
        self.V = f"{cell}-v2-mvs-fill-sam3d"
        self.RUN = self.SP / f"swap-runs/{cell}/mvs-fill-sam3d"
        self.CMP = self.N / f"module-swap-090-2026-10-07/cmp-{cell}-mvs-fill"
        self.SCALE = self.SP / c["scale"]
        self.ledger = self.SP / f"swap-runs/{cell}/ledger.json"
        self.model_info: dict | None = None
        self.backend: str | None = None

    # ---------------------------------------------------------------- notes directories
    @property
    def geometry_notes(self) -> Path:
        return self.N / "geometry-backbone-ab-2026-10-06"

    @property
    def swap_notes(self) -> Path:
        return self.N / "module-swap-090-2026-10-07"

    @property
    def completion_notes(self) -> Path:
        return self.N / "completion-licence-ab-2026-10-05"

    @property
    def compare_notes(self) -> Path:
        return self.N / "completion-ab-090-2026-10-06"

    # ---------------------------------------------------------------- running things
    def sh(self, argv: list, cwd: Path, env: dict | None = None) -> None:
        argv, env = [str(a) for a in argv], {k: str(v) for k, v in (env or {}).items()}
        shown = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + (" " if env else "") + shlex.join(argv)
        print(f"  $ (cd {cwd}) {shown}", flush=True)
        if self.dry:
            return
        self.backend = self.backend or ("run_stage" if any(a.endswith("run_stage.py") for a in argv) else "python")
        subprocess.run(argv, cwd=cwd, env={**self.env, **env}, check=True)

    def modal_run(self, argv: list, cwd: Path, env: dict | None = None) -> None:
        self.sh([*self.MODAL_RUN, *argv], cwd, env)

    def py(self, argv: list, cwd: Path, env: dict | None = None) -> None:
        self.sh([self.PY, *argv], cwd, env)

    def do(self, what: str, fn: Callable[[], None]) -> None:
        print(f"  + {what}", flush=True)
        if not self.dry:
            fn()

    def ab_env(self, **extra) -> dict:
        return dict(AB_CELL=self.cell, AB_RUN=self.FILLX, AB_OUT=self.AB, **extra)


# ---------------------------------------------------------------- the steps of run_all.sh, one item per skip rule
@dataclass
class Item:
    step: str
    name: str
    title: str
    done: Callable[[Ctx], bool] | None  # None: runs every time
    run: Callable[[Ctx], None]
    inputs: Callable[[Ctx], list[Path]]


def s2a_route(ctx: Ctx) -> None:
    backend = os.environ.get("GEOMETRY_MVS_BACKEND")
    if not backend:
        ctx.modal_run(["prod_route_modal.py", "--stage", "route", "--cells", ctx.cell], ctx.geometry_notes)
        return
    ctx.backend = backend

    def provider():
        from .providers import geometry_mvs, service_client

        geometry_mvs.run(ctx.cell, geometry_mvs.frames_for(ctx.RUNS / ctx.RUNNAME), dest=ctx.SP)
        ctx.model_info = service_client.last_model_info

    ctx.do(f"providers.geometry_mvs.run({ctx.cell!r}, frames_for({ctx.RUNS / ctx.RUNNAME}), dest={ctx.SP})  [GEOMETRY_MVS_BACKEND={backend}]", provider)


def s2c_export(ctx: Ctx) -> None:
    ctx.py(["backbone_ab_modal.py", "export", "mvs-da3-base", ctx.cell], ctx.geometry_notes)
    src, dst = ctx.SP / f"checks/bbab-export-{ctx.cell}-mvs-da3-base", ctx.SP / f"checks/bbab-export-{ctx.cell}-mvs-scipyba"
    ctx.do(f"mv {src} {dst}", lambda: src.rename(dst))


def s2d_gate(ctx: Ctx) -> None:
    ctx.py(["field_values_fill.py"], ctx.swap_notes)

    def gate():
        f = json.loads((ctx.swap_notes / "field-values-mvs-fill.json").read_text())
        a, b = f["mvs-da3-base"], f["mvs-fill"]
        for k in GATE_KEYS:
            print(f"  {k:24s} mvs {a.get(k)}  fill {b.get(k)}")
        if not (b["maeCm4values"] <= 1.56 and b["maxAbsErrCm"] <= 3.0):
            raise SystemExit("field-value gate failed: the fill changed a measurement")
        print("  GATE PASSED (MAE %.2f cm, max %.2f cm)" % (b["maeCm4values"], b["maxAbsErrCm"]))

    ctx.do("field-value gate (MAE <= 1.56 cm, max <= 3.0 cm)", gate)


def s4a_sam3d(ctx: Ctx) -> None:
    ctx.do(f"mkdir -p {ctx.AB}", lambda: ctx.AB.mkdir(parents=True, exist_ok=True))
    backend = os.environ.get("SAM3D_BACKEND")
    if backend:
        ctx.backend = backend
    ctx.modal_run(["completion_ab.py", "--stage", "sam3d"], ctx.completion_notes, ctx.ab_env(AB_FRAME="all"))
    if backend and not ctx.dry:  # the provider's model_info, as completion_ab.py journals it per object
        record = json.loads((ctx.AB / "sam3d/record.json").read_text())
        ctx.model_info = next((o["model_info"] for o in record["objects"].values() if o.get("model_info")), None)


def variants(ctx: Ctx) -> str:
    return ",".join(d for d in ctx.FRAMES if any((ctx.AB / d).glob("*.npz")))


def s4b_assemble(ctx: Ctx) -> None:
    found = variants(ctx) or ("<sam3d variants holding .npz>" if ctx.dry else "")
    ctx.modal_run(["completion_ab.py", "--stage", "assemble", "--variants", found], ctx.completion_notes, ctx.ab_env())


def s4b_compare(ctx: Ctx) -> None:
    ctx.do(f"mkdir -p {ctx.CMP}", lambda: ctx.CMP.mkdir(parents=True, exist_ok=True))
    ctx.py(["compare.py", *ctx.OBJS], ctx.compare_notes, ctx.ab_env(CMP_NOTES=ctx.CMP, CMP_SCALE=ctx.SCALE))


def s7_publish(ctx: Ctx) -> None:
    ready = [ctx.PG / "pg_isready", "-h", "127.0.0.1", "-p", "55432", "-q"]

    def start_postgres():
        if subprocess.run([str(a) for a in ready]).returncode == 0:
            return
        subprocess.run(
            [str(ctx.PG / "pg_ctl"), "-D", ctx.env["PGDATA_DIR"], "-o", "-p 55432 -c listen_addresses=127.0.0.1", "-l", str(ctx.SP / "pg55432.log"), "start"],
            env={**ctx.env, "LC_ALL": "en_US.UTF-8"},
            check=True,
        )
        for _ in range(40):
            if subprocess.run([str(a) for a in ready]).returncode == 0:
                return
            time.sleep(1)
        raise SystemExit("postgres at 127.0.0.1:55432 did not come up")

    ctx.do("postgres at 127.0.0.1:55432 (pg_isready || pg_ctl start)", start_postgres)
    ctx.py(
        [ctx.SWAP_ROOT / "platform/publish-swap-20261007.py", ctx.V, ctx.RUN, ctx.SCALE, f"{ctx.cell} module swap: MVS + MoGe-3 fill + SAM 3D, assembly v2"],
        ctx.PLATFORM,
    )


def s8_checks(ctx: Ctx) -> None:
    ctx.py(["serve_export.py", ctx.V], ctx.swap_notes)
    ctx.py(["run_stages.py", ctx.V], ctx.swap_notes, dict(STAGES_CELL=ctx.cell))
    ctx.py(["build_swap_layer.py", ctx.V], ctx.swap_notes)
    if ctx.cell == "090":
        ctx.py(["compare_json.py"], ctx.swap_notes)
        ctx.py(["compare_layers.py"], ctx.swap_notes)
        print("  tables: table.md layers-table.md")
    else:
        ctx.py(["tables_030.py"], ctx.swap_notes)
        print("  table: table-030.md")


ITEMS: list[Item] = [
    Item("S2a", "route", "RoMa matches + DA3-BASE start + MoGe-3 depth (GPU) -> checks/clean-gpu/<cell>",
         lambda c: (c.SP / f"checks/clean-gpu/{c.cell}/moge-frame_0001.npz").exists(), s2a_route,
         lambda c: [c.RUNS / c.RUNNAME]),
    Item("S2b", "mvs", "bundle adjustment + two-view triangulation (CPU) -> checks/bbab-geom/<cell>-mvs-da3-base-padded",
         lambda c: (c.SP / f"checks/bbab-geom/{c.cell}-mvs-da3-base-padded/geometry").is_dir(),
         lambda c: c.py(["mvs_route.py", "da3-base"], c.geometry_notes),
         lambda c: [c.SP / f"checks/clean-gpu/{c.cell}", c.SP / f"checks/clean-geom/{c.cell}-da3-base-ba-f"]),
    Item("S2c", "analyse", "fair evaluator (floor, e-stop scale, field values)",
         lambda c: (c.SP / f"checks/bbab-analyse/{c.cell}-mvs-da3-base-padded.json").exists(),
         lambda c: c.modal_run(["backbone_ab_modal.py", "analyse", f"{c.cell}-mvs-da3-base-padded"], c.geometry_notes),
         lambda c: [c.SP / f"checks/bbab-geom/{c.cell}-mvs-da3-base-padded"]),
    Item("S2c", "export", "export -> checks/bbab-export-<cell>-mvs-scipyba",
         lambda c: (c.SP / f"checks/bbab-export-{c.cell}-mvs-scipyba").is_dir(), s2c_export,
         lambda c: [c.SP / f"checks/bbab-geom/{c.cell}-mvs-da3-base-padded"]),
    Item("S2d", "fill", "MoGe-3 in-mask fill (CPU) -> checks/bbab-geom/<cell>-mvs-fill-padded",
         lambda c: (c.SP / f"checks/bbab-geom/{c.cell}-mvs-fill-padded/geometry").is_dir(),
         lambda c: c.py(["fill_geometry.py"], c.swap_notes, dict(FILL_CELL=c.cell)),
         lambda c: [c.SP / f"checks/bbab-geom/{c.cell}-mvs-da3-base-padded", c.SP / f"checks/clean-gpu/{c.cell}"]),
    Item("S2d", "analyse-fill", "fair evaluator on the filled geometry",
         lambda c: (c.SP / f"checks/bbab-analyse/{c.cell}-mvs-fill-padded.json").exists(),
         lambda c: c.modal_run(["backbone_ab_modal.py", "analyse", f"{c.cell}-mvs-fill-padded"], c.geometry_notes),
         lambda c: [c.SP / f"checks/bbab-geom/{c.cell}-mvs-fill-padded"]),
    Item("S2d", "export-fill", "export -> checks/bbab-export-<cell>-mvs-fill",
         lambda c: (c.SP / f"checks/bbab-export-{c.cell}-mvs-fill").is_dir(),
         lambda c: c.py(["backbone_ab_modal.py", "export", "mvs-fill", c.cell], c.geometry_notes),
         lambda c: [c.SP / f"checks/bbab-geom/{c.cell}-mvs-fill-padded"]),
    Item("S2d", "field-values", "field values + gate: the fill must not change a measurement", None, s2d_gate,
         lambda c: [c.SP / f"checks/bbab-analyse/{c.cell}-mvs-fill-padded.json"]),
    Item("S4a", "sam3d", "SAM 3D Objects candidates from every masked photo (GPU) -> swap-runs/<cell>/mvs-fill-ab/sam3d*",
         lambda c: (c.AB / "sam3d/record.json").exists(), s4a_sam3d, lambda c: [c.FILLX]),
    Item("S4b", "assemble", "candidate assembly (CPU)",
         lambda c: (c.AB / "assembly/sam3d/comparisons.json").exists(), s4b_assemble,
         lambda c: [c.AB / d for d in c.FRAMES if (c.AB / d).is_dir()]),
    Item("S4b", "compare", "uniform selection -> cmp-<cell>-mvs-fill/results.json",
         lambda c: (c.CMP / "results.json").exists(), s4b_compare, lambda c: [c.AB / "assembly"]),
    Item("S4c", "generation", "generation/ contract -> swap-runs/<cell>/mvs-fill-sam3d",
         lambda c: (c.RUN / "generation").is_dir(),
         lambda c: c.py(["swap_generation.py", f"{c.cell}/mvs-fill-sam3d"], c.swap_notes), lambda c: [c.CMP / "results.json"]),
    Item("S4c", "pins", "importer pins on the run directory", None,
         lambda c: c.py(["pin_run.py", c.RUN], c.swap_notes), lambda c: [c.RUN / "generation"]),
    Item("S5", "assembly-v2", "assembly v2 (CPU, floor-contact hinge)",
         lambda c: (c.RUN / "result/comparisons.json").exists(),
         lambda c: c.modal_run(["modal_apps/assemble_scene.py", "--run", c.RUN], c.SERVING), lambda c: [c.RUN / "generation"]),
    Item("S6", "report", "capture report -> swap-runs/<cell>/mvs-fill-sam3d/public",
         lambda c: (c.RUN / "public/scene.json").exists(),
         lambda c: c.py(["scripts/research/build_capture_report.py", "--run", c.RUN, "--label", f"{c.cell}: MVS + MoGe-3 fill + SAM 3D, assembly v2",
                         "--pages-root", c.WORKCELL / "web"], c.SERVING),
         lambda c: [c.RUN / "result"]),
    Item("S7", "publish", "platform import + export (Postgres at 127.0.0.1:55432)",
         lambda c: (c.PLATFORM / f".platform/swap-20261007/{c.V}/result.json").exists(), s7_publish, lambda c: [c.RUN / "public"]),
    Item("S8", "checks", "S8-S14 original checks + layer + tables", None, s8_checks, lambda c: [c.RUN / "public"]),
]
STEPS = list(dict.fromkeys(i.step for i in ITEMS))


def select(from_step: str | None, only: str | None) -> list[Item]:
    for key in (from_step, only):
        if key and key not in STEPS:
            raise SystemExit(f"unknown step {key!r}; steps: {' '.join(STEPS)}")
    if only:
        return [i for i in ITEMS if i.step == only]
    if from_step:
        return [i for i in ITEMS if STEPS.index(i.step) >= STEPS.index(from_step)]
    return ITEMS


def state(item: Item, ctx: Ctx) -> str:
    return "always" if item.done is None else ("done" if item.done(ctx) else "todo")


# ---------------------------------------------------------------- the ledger
def sha256_path(path: Path) -> str | None:
    """A file's sha256, or for a directory the sha256 of 'relative path\\0sha256\\n' over its files in sorted order."""
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(f"{f.relative_to(path)}\0{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    return digest.hexdigest()


def record(ctx: Ctx, item: Item, started: float, inputs: dict, status: str, error: str | None) -> None:
    entry = dict(
        step=item.step, item=item.name, status=status, error=error,
        started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)), ended=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        seconds=round(time.time() - started, 3), host=socket.gethostname(), backend=ctx.backend, inputs_sha256=inputs,
        model_info=ctx.model_info,
    )
    ledger = json.loads(ctx.ledger.read_text()) if ctx.ledger.exists() else {"cell": ctx.cell, "entries": []}
    ledger["entries"].append(entry)
    ctx.ledger.parent.mkdir(parents=True, exist_ok=True)
    ctx.ledger.write_text(json.dumps(ledger, indent=1) + "\n")


# ---------------------------------------------------------------- commands
def cmd_status(ctx: Ctx, items: list[Item]) -> None:
    print(f"panoptes {ctx.cell}  SWAP_SCRATCH={ctx.SP}")
    for item in items:
        print(f"  {item.step:4s} {item.name:14s} {state(item, ctx):7s} {item.title}")


def cmd_run(ctx: Ctx, items: list[Item]) -> None:
    print(f"panoptes {ctx.cell}  SWAP_SCRATCH={ctx.SP}" + ("  (dry run: nothing is executed)" if ctx.dry else ""))
    for item in items:
        print(f"\n===== {ctx.cell} {item.step} {item.name}  {time.strftime('%H:%M:%S')}  {item.title}")
        if state(item, ctx) == "done":
            print("  done, skipped")
            continue
        ctx.model_info, ctx.backend = None, None
        started = time.time()
        inputs = {} if ctx.dry else {str(p): sha256_path(p) for p in item.inputs(ctx)}
        try:
            item.run(ctx)
            status, error = "ok", None
        except (Exception, SystemExit) as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
        if not ctx.dry:
            record(ctx, item, started, inputs, status, error)
        if error:
            raise SystemExit(f"{item.step} {item.name} failed: {error}")
    print(f"\nDONE {ctx.cell} {time.strftime('%H:%M:%S')}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="panoptes", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status"):
        p = sub.add_parser(name)
        p.add_argument("--cell", required=True, choices=sorted(CELLS))
        p.add_argument("--from", dest="from_step", metavar="STEP", help=f"start at this step ({' '.join(STEPS)})")
        p.add_argument("--only", metavar="STEP", help="this step only")
        if name == "run":
            p.add_argument("--dry-run", action="store_true", help="list the steps, their done / todo state and their commands; run nothing")
    args = parser.parse_args(argv)
    ctx = Ctx(args.cell, load_env(), dry_run=getattr(args, "dry_run", False))
    items = select(args.from_step, args.only)
    (cmd_run if args.command == "run" else cmd_status)(ctx, items)


if __name__ == "__main__":
    main()
