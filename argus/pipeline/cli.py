"""Capture-to-publication pipeline: panoptes run --cell 090|030 [--from STEP] [--only STEP] [--dry-run]."""

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

from argus import ROOT
CELLS = {p.stem: json.loads(p.read_text()) for p in (ROOT / 'argus/pipeline/cells').glob('*.json')}
GATE_KEYS = ("housing090R_cm", "fence090R_cm", "housing030L_cm", "housing030R_cm", "estopNativeToMeters090", "estopNativeToMeters030")


def load_env() -> dict:
    """The process environment is the one config boundary; the CLI supplies code/data defaults."""
    env = dict(os.environ)
    data = Path(env.get('PANOPTES_DATA_ROOT', './data')).resolve()
    for key, value in {'PANOPTES_DATA_ROOT': data, 'PANOPTES_RUNS': data / 'runs', 'PANOPTES_PAGES': data / 'measurement-layer',
                       'PY': sys.executable}.items():
        env.setdefault(key, str(value))
    for key in ('PANOPTES_DATA_ROOT', 'PANOPTES_RUNS', 'PANOPTES_PAGES'):
        env[key] = str(Path(env[key]).resolve())
    return env


class Ctx:
    def __init__(self, cell: str, env: dict, dry_run: bool = False):
        if cell not in CELLS:
            raise SystemExit(f"cell: {' | '.join(CELLS)}")
        self.cell, self.env, self.dry = cell, env, dry_run
        self.SP = Path(env['PANOPTES_DATA_ROOT'])
        self.N = self.SP / 'pipeline'
        self.RUNS = Path(env['PANOPTES_RUNS'])
        self.PY = env['PY']
        self.MODAL_RUN = [shutil.which('modal') or str(Path(sys.executable).parent / 'modal'), 'run']
        c = CELLS[cell]
        self.RUNNAME, self.OBJS, self.FRAMES = c["run"], c["objects"], c["frames"]
        self.FILLX = self.SP / f"checks/bbab-export-{cell}-mvs-fill"
        self.AB = self.SP / f"swap-runs/{cell}/mvs-fill-ab"
        self.V = f"{cell}-v2-mvs-fill-sam3d"
        self.RUN = self.SP / f"swap-runs/{cell}/mvs-fill-sam3d"
        self.CMP = self.N / f"cmp-{cell}-mvs-fill"
        self.SCALE = self.SP / c["scale"]
        self.ledger = self.SP / f"swap-runs/{cell}/ledger.json"
        self.model_info: dict | None = None
        self.backend: str | None = None

    # ---------------------------------------------------------------- running things
    def sh(self, argv: list, cwd: Path, env: dict | None = None) -> None:
        argv, env = [str(a) for a in argv], {k: str(v) for k, v in (env or {}).items()}
        shown = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + (" " if env else "") + shlex.join(argv)
        print(f"  $ (cd {cwd}) {shown}", flush=True)
        if self.dry:
            return
        self.backend = self.backend or ("modal" if Path(argv[0]).name == "modal" else "python")
        subprocess.run(argv, cwd=cwd, env={**self.env, **env}, check=True)

    def modal_run(self, argv: list, cwd: Path, env: dict | None = None) -> None:
        self.sh([*self.MODAL_RUN, *argv], cwd, env)

    def py(self, argv: list, cwd: Path, env: dict | None = None) -> None:
        self.sh([self.PY, *argv], ROOT, env)

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
    backend = os.environ.get("GEOMETRY_MVS_BACKEND") or "modal"
    ctx.backend = backend

    def provider():
        from argus.providers import geometry_mvs, service_client

        geometry_mvs.run(ctx.cell, geometry_mvs.frames_for(ctx.RUNS / ctx.RUNNAME), dest=ctx.SP)
        ctx.model_info = service_client.last_model_info

    ctx.do(f"providers.geometry_mvs.run({ctx.cell!r}, frames_for({ctx.RUNS / ctx.RUNNAME}), dest={ctx.SP})  [GEOMETRY_MVS_BACKEND={backend}]", provider)


def s2c_export(ctx: Ctx) -> None:
    ctx.py(["-m", "argus.pipeline.geometry_evaluator", "export", "mvs-da3-base", ctx.cell], ROOT)
    src, dst = ctx.SP / f"checks/bbab-export-{ctx.cell}-mvs-da3-base", ctx.SP / f"checks/bbab-export-{ctx.cell}-mvs-scipyba"
    ctx.do(f"mv {src} {dst}", lambda: src.rename(dst))


def s2d_gate(ctx: Ctx) -> None:
    ctx.py(["-m", "argus.pipeline.field_values_fill"], ROOT)

    def gate():
        f = json.loads((ctx.N / "field-values-mvs-fill.json").read_text())
        a, b = f["mvs-da3-base"], f["mvs-fill"]
        for k in GATE_KEYS:
            print(f"  {k:24s} mvs {a.get(k)}  fill {b.get(k)}")
        if not (b["maeCm4values"] <= 1.56 and b["maxAbsErrCm"] <= 3.0):
            raise SystemExit("field-value gate failed: the fill changed a measurement")
        print("  GATE PASSED (MAE %.2f cm, max %.2f cm)" % (b["maeCm4values"], b["maxAbsErrCm"]))

    ctx.do("field-value gate (MAE <= 1.56 cm, max <= 3.0 cm)", gate)


def s4a_sam3d(ctx: Ctx) -> None:
    ctx.do(f"mkdir -p {ctx.AB}", lambda: ctx.AB.mkdir(parents=True, exist_ok=True))
    backend = os.environ.get("SAM3D_BACKEND", "modal")
    ctx.backend = backend
    if backend == "http":
        ctx.py(["-m", "argus.pipeline.completion", "--stage", "sam3d"], ROOT, ctx.ab_env(AB_FRAME="all"))
    else:
        ctx.modal_run([ROOT / "argus/pipeline/completion.py", "--stage", "sam3d"], ROOT, ctx.ab_env(AB_FRAME="all"))
    if backend and not ctx.dry:  # the provider's model_info, as completion_ab.py journals it per object
        record = json.loads((ctx.AB / "sam3d/record.json").read_text())
        ctx.model_info = next((o["model_info"] for o in record["objects"].values() if o.get("model_info")), None)


def variants(ctx: Ctx) -> str:
    return ",".join(d for d in ctx.FRAMES if any((ctx.AB / d).glob("*.npz")))


def missing_candidates(out: Path, objects: list, frames: list) -> list:
    return [oid for oid in objects if not any((out / variant / f'{oid}.npz').is_file() for variant in frames)]


def current_selection(ctx: Ctx) -> bool:
    path = ctx.CMP / 'results.json'
    if not path.is_file():
        return False
    try:
        results = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return isinstance(results, dict) and all(isinstance(results.get(oid), dict)
               and isinstance(results[oid].get('decision'), dict)
               and isinstance(results[oid]['decision'].get('code'), str)
               and results[oid]['decision']['code'] for oid in ctx.OBJS)


def s4b_assemble(ctx: Ctx) -> None:
    found = variants(ctx) or ("<sam3d variants holding .npz>" if ctx.dry else "")
    ctx.py(["-m", "argus.pipeline.completion", "--stage", "assemble", "--variants", found], ROOT, ctx.ab_env())


def s4b_compare(ctx: Ctx) -> None:
    ctx.do(f"mkdir -p {ctx.CMP}", lambda: ctx.CMP.mkdir(parents=True, exist_ok=True))
    ctx.py(["-m", "argus.pipeline.select_sam3d", *ctx.OBJS], ROOT, ctx.ab_env(CMP_NOTES=ctx.CMP, CMP_SCALE=ctx.SCALE))


def s7_publish(ctx: Ctx) -> None:
    ctx.py(
        ["-m", "argus.platform.publish_capture", ctx.V, ctx.RUN, ctx.SCALE, f"{ctx.cell} module swap: MVS + MoGe-3 fill + SAM 3D, assembly v2"],
        ROOT,
    )


def s8_checks(ctx: Ctx) -> None:
    ctx.py(["-m", "argus.pipeline.serve_export", ctx.V], ROOT)
    ctx.py(["-m", "argus.pipeline.run_stages", ctx.V], ROOT, dict(STAGES_CELL=ctx.cell))
    ctx.py(["-m", "argus.pipeline.build_swap_layer", ctx.V], ROOT)


ITEMS: list[Item] = [
    Item("S2a", "route", "RoMa matches + DA3-BASE start + MoGe-3 depth (GPU) -> checks/clean-gpu/<cell>",
         lambda c: (c.SP / f"checks/clean-gpu/{c.cell}/moge-frame_0001.npz").exists(), s2a_route,
         lambda c: [c.RUNS / c.RUNNAME]),
    Item("S2b", "mvs", "bundle adjustment + two-view triangulation (CPU) -> checks/bbab-geom/<cell>-mvs-da3-base-padded",
         lambda c: (c.SP / f"checks/bbab-geom/{c.cell}-mvs-da3-base-padded/geometry").is_dir(),
         lambda c: c.py(["-m", "argus.pipeline.mvs_route", "da3-base"], ROOT, dict(AB_CELL=c.cell)),
         lambda c: [c.SP / f"checks/clean-gpu/{c.cell}", c.SP / f"checks/clean-geom/{c.cell}-da3-base-ba-f"]),
    Item("S2c", "analyse", "fair evaluator (floor, e-stop scale, field values)",
         lambda c: (c.SP / f"checks/bbab-analyse/{c.cell}-mvs-da3-base-padded.json").exists(),
         lambda c: c.py(["-m", "argus.pipeline.geometry_evaluator", "analyse", f"{c.cell}-mvs-da3-base-padded"], ROOT),
         lambda c: [c.SP / f"checks/bbab-geom/{c.cell}-mvs-da3-base-padded"]),
    Item("S2c", "export", "export -> checks/bbab-export-<cell>-mvs-scipyba",
         lambda c: (c.SP / f"checks/bbab-export-{c.cell}-mvs-scipyba").is_dir(), s2c_export,
         lambda c: [c.SP / f"checks/bbab-geom/{c.cell}-mvs-da3-base-padded"]),
    Item("S2d", "fill", "MoGe-3 in-mask fill (CPU) -> checks/bbab-geom/<cell>-mvs-fill-padded",
         lambda c: (c.SP / f"checks/bbab-geom/{c.cell}-mvs-fill-padded/geometry").is_dir(),
         lambda c: c.py(["-m", "argus.pipeline.fill_geometry"], ROOT, dict(FILL_CELL=c.cell)),
         lambda c: [c.SP / f"checks/bbab-geom/{c.cell}-mvs-da3-base-padded", c.SP / f"checks/clean-gpu/{c.cell}"]),
    Item("S2d", "analyse-fill", "fair evaluator on the filled geometry",
         lambda c: (c.SP / f"checks/bbab-analyse/{c.cell}-mvs-fill-padded.json").exists(),
         lambda c: c.py(["-m", "argus.pipeline.geometry_evaluator", "analyse", f"{c.cell}-mvs-fill-padded"], ROOT),
         lambda c: [c.SP / f"checks/bbab-geom/{c.cell}-mvs-fill-padded"]),
    Item("S2d", "export-fill", "export -> checks/bbab-export-<cell>-mvs-fill",
         lambda c: (c.SP / f"checks/bbab-export-{c.cell}-mvs-fill").is_dir(),
         lambda c: c.py(["-m", "argus.pipeline.geometry_evaluator", "export", "mvs-fill", c.cell], ROOT),
         lambda c: [c.SP / f"checks/bbab-geom/{c.cell}-mvs-fill-padded"]),
    Item("S2d", "field-values", "field values + gate: the fill must not change a measurement", None, s2d_gate,
         lambda c: [c.SP / f"checks/bbab-analyse/{c.cell}-mvs-fill-padded.json"]),
    Item("S4a", "sam3d", "SAM 3D Objects candidates from every masked photo (GPU) -> swap-runs/<cell>/mvs-fill-ab/sam3d*",
         lambda c: not missing_candidates(c.AB, c.OBJS, c.FRAMES), s4a_sam3d, lambda c: [c.FILLX]),
    Item("S4b", "assemble", "candidate assembly (CPU)",
         lambda c: (c.AB / "assembly/sam3d/comparisons.json").exists(), s4b_assemble,
         lambda c: [c.AB / d for d in c.FRAMES if (c.AB / d).is_dir()]),
    Item("S4b", "compare", "uniform selection -> cmp-<cell>-mvs-fill/results.json",
         current_selection, s4b_compare, lambda c: [c.AB / "assembly"]),
    Item("S4c", "generation", "generation/ contract -> swap-runs/<cell>/mvs-fill-sam3d",
         None,
         lambda c: c.py(["-m", "argus.pipeline.swap_generation", f"{c.cell}/mvs-fill-sam3d"], ROOT), lambda c: [c.CMP / "results.json"]),
    Item("S4c", "pins", "importer pins on the run directory", None,
         lambda c: c.py(["-m", "argus.pipeline.pin_run", c.RUN], ROOT), lambda c: [c.RUN / "generation"]),
    Item("S5", "assembly-v2", "assembly v2 (CPU, floor-contact hinge)",
         lambda c: (c.RUN / "result/comparisons.json").exists(),
         lambda c: c.py(["-m", "argus.pipeline.assemble_scene", "--run", c.RUN], ROOT), lambda c: [c.RUN / "generation"]),
    Item("S6", "report", "capture report -> swap-runs/<cell>/mvs-fill-sam3d/public",
         lambda c: (c.RUN / "public/scene.json").exists(),
         lambda c: c.py(["-m", "argus.pipeline.build_capture_report", "--run", c.RUN, "--label", f"{c.cell}: MVS + MoGe-3 fill + SAM 3D, assembly v2"], ROOT),
         lambda c: [c.RUN / "result"]),
    Item("S7", "publish", "platform import + export (configured database)",
         lambda c: (c.SP / f"publications/{c.V}/result.json").exists(), s7_publish, lambda c: [c.RUN / "public"]),
    Item("S8", "checks", "S8-S14 original checks + layer", None, s8_checks, lambda c: [c.RUN / "public"]),
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
    print(f"panoptes {ctx.cell}  PANOPTES_DATA_ROOT={ctx.SP}")
    for item in items:
        print(f"  {item.step:4s} {item.name:14s} {state(item, ctx):7s} {item.title}")


def cmd_run(ctx: Ctx, items: list[Item]) -> None:
    print(f"panoptes {ctx.cell}  PANOPTES_DATA_ROOT={ctx.SP}" + ("  (dry run: nothing is executed)" if ctx.dry else ""))
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
    env = load_env()
    os.environ.update(env)
    ctx = Ctx(args.cell, env, dry_run=getattr(args, "dry_run", False))
    items = select(args.from_step, args.only)
    (cmd_run if args.command == "run" else cmd_status)(ctx, items)


if __name__ == "__main__":
    main()
