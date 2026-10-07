"""The GPU half of the licence-clean geometry route: run(cell, frames, options) -> a directory holding

  checks/clean-gpu/<cell>/{roma-i-j, dense-i-j, moge-frame_000N}.npz      RoMa matches + dense warps, MoGe-3 depth
  checks/da3fair-geom/<cell>-da3-base-padded/geometry/...                  the DA3-BASE start geometry
  checks/clean-geom/<cell>-da3-base-ba-f/geometry/...                      BA-f cameras

= what research/module-swap-2026-10-07's mvs_route.py / prod_route_modal.py read (docs/BACKENDS-v1.md, geometry-mvs).

  GEOMETRY_MVS_BACKEND  http   POST /v1/geometry-mvs/jobs + polling; the result tar.gz is extracted into dest
                               (GEOMETRY_MVS_HTTP_URLS, PANOPTES_SERVICE_API_KEY)
                        local  geometry_clean_ab.py's infer / refine / dense-infer bodies (and fair_ab_modal's DA3-BASE
                               infer when the start is missing) in this process through run_stage.py's stub
                        modal  the same bodies on our Modal apps, in an ephemeral app.run() as `modal run` does; default

frames = [{frame_id, canonical_png: bytes (518x518 RGB), alpha: bool HxW}] - frames_for(run_dir) reads them from a
run's manifest.json. dest defaults to $SWAP_SCRATCH, the research data root, so the three directories land where the
CPU route reads them; the local / modal bodies write there directly and also read the frozen fair inputs
(checks/da3fair-data/<cell>) from it.
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from ..backends import service_backend
from . import service_client
from .base import ProviderError, onprem, serving_root, workcell_root

MODEL = "geometry-mvs"
OPTIONS = {"start": "da3-base", "roma": "outdoor", "pairs": "all"}
OUTPUTS = (
    "checks/clean-gpu/{cell}",
    "checks/da3fair-geom/{cell}-da3-base-padded/geometry",
    "checks/clean-geom/{cell}-da3-base-ba-f/geometry",
)


def frames_for(run: Path) -> list[dict]:
    """The frozen canonical frames of a run (manifest.json frames[]: frame_id, canonical, alpha)."""
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    return [
        {
            "frame_id": f["frame_id"],
            "canonical_png": (run / f["canonical"]).read_bytes(),
            "alpha": np.load(run / f["alpha"]).astype(bool),
        }
        for f in manifest["frames"]
    ]


def outputs(cell: str, dest: Path) -> list[Path]:
    return [Path(dest) / p.format(cell=cell) for p in OUTPUTS]


def run(cell: str, frames: list[dict], options: dict | None = None, *, dest: Path | str | None = None) -> Path:
    backend = service_backend("GEOMETRY_MVS_BACKEND", "modal")
    options = {**OPTIONS, **(options or {})}
    dest = Path(dest or os.environ.get("SWAP_SCRATCH") or tempfile.mkdtemp(prefix=f"geometry-mvs-{cell}-"))
    if backend == "http":
        _http(cell, frames, options, dest)
    elif backend in ("local", "modal"):
        _stages(cell, frames, options, dest, backend)
    else:
        raise ValueError(f"unknown GEOMETRY_MVS_BACKEND {backend!r}")
    missing = [str(p.relative_to(dest)) for p in outputs(cell, dest) if not p.is_dir()]
    if missing:
        raise ProviderError("geometry-mvs", backend, f"{dest} lacks {missing}")
    return dest


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _http(cell, frames, options, dest: Path) -> None:
    body = {
        "cell": cell,
        "options": options,
        "frames": [
            {
                "frame_id": f["frame_id"],
                "canonical_png_b64": _b64(f["canonical_png"]),
                "alpha_npy_b64": _b64(_npy(np.asarray(f["alpha"], bool))),
            }
            for f in frames
        ],
    }
    result = service_client.call(MODEL, body)
    archive = service_client.fetch_result(result, operation=MODEL)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")


def _import(path: Path):
    """The app module under the real modal client (what `modal run` imports)."""
    for p in (str(path.parent.parent), str(path.parent)):
        if p not in sys.path:
            sys.path.insert(1, p)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _stages(cell, frames, options, dest: Path, backend: str) -> None:
    """geometry_clean_ab.main(stage=infer | refine | dense-infer) on the request's frames, function bodies unchanged:
    fair_ab_modal.frames_for is pointed at the request (not at outputs/candidate-evaluation), everything else is read
    from and written to dest = SWAP_SCRATCH. The research scripts' own guards stay (infer refuses an existing
    checks/clean-gpu; delete it for a full recompute, as REPRODUCE-PROMPT.md says)."""
    if options != OPTIONS:
        raise ProviderError("geometry-mvs", backend, f"the function bodies only know {OPTIONS}, not {options}")
    previous = os.environ.get("SWAP_SCRATCH")
    os.environ["SWAP_SCRATCH"] = str(dest)  # fair_ab_modal / geometry_clean_ab read it at import
    try:
        _run_stages(cell, frames, dest, backend)
    finally:
        if previous is None:
            os.environ.pop("SWAP_SCRATCH", None)
        else:
            os.environ["SWAP_SCRATCH"] = previous


def _run_stages(cell, frames, dest: Path, backend: str) -> None:
    from PIL import Image

    notes = Path(os.environ.get("SWAP_NOTES") or serving_root() / "research/module-swap-2026-10-07/notes")
    kit = workcell_root()
    sys.path[:0] = [str(notes / "geometry-licence-ab-fair-2026-10-05"), str(kit / "modal_apps"), str(kit / "scripts/onprem")]
    app_path = kit / "modal_apps/geometry_clean_ab.py"
    gc = onprem().load_app(app_path) if backend == "local" else _import(app_path)
    fam = sys.modules["fair_ab_modal"]
    x0, xw = fam.X0, fam.XW

    def request_frames(_cell):
        out = {"padded": {}, "unpadded": {}}
        for f in frames:
            name = f["frame_id"] + ".png"
            out["padded"][name] = f["canonical_png"]
            buffer = io.BytesIO()
            Image.open(io.BytesIO(f["canonical_png"])).convert("RGB").crop((x0, 0, x0 + xw, 518)).save(buffer, format="PNG")
            out["unpadded"][name] = buffer.getvalue()
        return out

    fam.frames_for = request_frames
    with contextlib.ExitStack() as stack:
        for app in (gc.app, fam.app):  # ephemeral apps, as `modal run`; the stub's run() is a no-op
            stack.enter_context(app.run())
        if not gc.INIT["da3-base"](cell).exists():  # the DA3-BASE start: the fair A/B's own GPU stage
            fam.app.registered_entrypoints["main"](stage="infer", cells=cell)
        main = gc.app.registered_entrypoints["main"]
        main(stage="infer", cells=cell, vggt=False)
        main(stage="refine", cells=cell, bases="da3-base", only=f"{cell}-da3-base-ba-f")
        main(stage="dense-infer", cells=cell)


__all__ = ["MODEL", "OPTIONS", "OUTPUTS", "frames_for", "outputs", "run"]
