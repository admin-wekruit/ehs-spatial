"""Selected geometry route. GEOMETRY_MVS_BACKEND=modal|http; request frames in, clean GPU/start/refined geometry directories out."""

from __future__ import annotations

from argus import ROOT
import base64
import contextlib
import importlib
import io
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from argus.pipeline.backends import service_backend
from argus.providers import service_client
from argus.providers.base import ProviderError

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
    dest = Path(dest or os.environ.get("PANOPTES_DATA_ROOT") or tempfile.mkdtemp(prefix=f"geometry-mvs-{cell}-"))
    if backend == "http":
        _http(cell, frames, options, dest)
    elif backend == "modal":
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




def _stages(cell, frames, options, dest: Path, backend: str) -> None:
    """geometry_clean_ab.main(stage=infer | refine | dense-infer) on the request's frames, function bodies unchanged:
    fair_ab_modal.frames_for is pointed at the request (not at outputs/candidate-evaluation), everything else is read
    from and written to dest = PANOPTES_DATA_ROOT. The research scripts' own guards stay (infer refuses an existing
    checks/clean-gpu; delete it for a full recompute, as REPRODUCE-PROMPT.md says)."""
    if options != OPTIONS:
        raise ProviderError("geometry-mvs", backend, f"the function bodies only know {OPTIONS}, not {options}")
    previous = os.environ.get("PANOPTES_DATA_ROOT")
    os.environ["PANOPTES_DATA_ROOT"] = str(dest)  # fair_ab_modal / geometry_clean_ab read it at import
    try:
        _run_stages(cell, frames, dest, backend)
    finally:
        if previous is None:
            os.environ.pop("PANOPTES_DATA_ROOT", None)
        else:
            os.environ["PANOPTES_DATA_ROOT"] = previous


def _run_stages(cell, frames, dest: Path, backend: str) -> None:
    from PIL import Image

    # The request data root is set before loading the two canonical app modules.
    fam = importlib.reload(importlib.import_module("argus.pipeline.field_evaluator"))
    gc = importlib.reload(importlib.import_module("argus.pipeline.geometry_clean_ab"))
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
        for app in (fam.app, gc.app):  # ephemeral apps, as `modal run`
            stack.enter_context(app.run())
        if not gc.INIT["da3-base"](cell).exists():  # the DA3-BASE start: the fair A/B's own GPU stage
            fam.app.registered_entrypoints["main"](stage="infer", cells=cell)
        main = gc.app.registered_entrypoints["main"]
        main(stage="infer", cells=cell)
        main(stage="refine", cells=cell, bases="da3-base", only=f"{cell}-da3-base-ba-f")
        main(stage="dense-infer", cells=cell)


__all__ = ["MODEL", "OPTIONS", "OUTPUTS", "frames_for", "outputs", "run"]
