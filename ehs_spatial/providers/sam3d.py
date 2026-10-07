"""SAM 3D Objects, one candidate per call: generate(rgb, mask, pointmap, seed) -> what
modal_apps/sam3d_research.SAM3DObjects.run returns, plus model_info.

  SAM3D_BACKEND  http   the sam3d service (docs/BACKENDS-v1.md): POST /v1/sam3d (sync, the default for one candidate)
                        or /v1/sam3d/jobs + polling (sync=False); SAM3D_HTTP_URLS, PANOPTES_SERVICE_API_KEY
                 local  the same class in this process through ehs-spatial/scripts/onprem/run_stage.py's stub
                        (PANOPTES_WORKCELL = the ehs-spatial checkout, WEIGHTS = its weights mirror); no Modal
                 modal  the class of modal_apps/sam3d_research as completion_ab.py uses it (inside `modal run`,
                        or under run_stage.py where `modal` is the stub); the default when unset

Result: {vertices float32 Nx3, faces Mx3, colors uint8 Nx3, object_to_camera_p3d float64 4x4, pins, seconds, gpu,
model_info}; a failed generation comes back as the class reports it, {error: <message>, seconds, gpu}: the caller
journals it and never re-sends that input. Inputs: rgb HxWx3 uint8, mask HxW bool, pointmap HxWx3 float32
(PyTorch3D camera, NaN = no depth), seed. The weights load once per process (the instance is cached).
"""

from __future__ import annotations

import base64
import functools
import io
import sys

import numpy as np

from ..backends import service_backend
from . import service_client
from .base import ProviderError, onprem, workcell_root

MODEL = "sam3d"


def generate(rgb: np.ndarray, mask: np.ndarray, pointmap: np.ndarray, seed: int, *, sync: bool = True) -> dict:
    backend = service_backend("SAM3D_BACKEND", "modal")
    rgb, mask, pointmap = np.asarray(rgb, np.uint8), np.asarray(mask, bool), np.asarray(pointmap, np.float32)
    if rgb.shape[:2] != mask.shape or pointmap.shape != rgb.shape[:2] + (3,):
        raise ValueError("image, mask and pointmap grids differ")
    if backend == "http":
        return _http(rgb, mask, pointmap, int(seed), sync)
    if backend in ("local", "modal"):
        return _normalise(_runner(backend).run.remote(rgb, mask, pointmap, int(seed)), backend)
    raise ValueError(f"unknown SAM3D_BACKEND {backend!r}")


def _png(array: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _http(rgb, mask, pointmap, seed, sync) -> dict:
    pointmap_npz = io.BytesIO()
    np.savez_compressed(pointmap_npz, pointmap=pointmap)
    body = {
        "seed": seed,
        "image_b64": _b64(_png(rgb)),
        "mask_b64": _b64(_png(mask.astype(np.uint8) * 255)),
        "pointmap_npz_b64": _b64(pointmap_npz.getvalue()),
    }
    try:
        result = service_client.call(MODEL, body, sync=sync)
    except service_client.ServiceError as error:
        if error.code != "model_error":
            raise
        return {"error": error.original_message, "seconds": error.job.get("seconds"), "gpu": error.job.get("gpu")}
    if not result.get("mesh_npz_b64"):
        raise ProviderError("sam3d", "http", f"no mesh_npz_b64 in the result (keys {sorted(result)})")
    mesh = np.load(io.BytesIO(base64.b64decode(result["mesh_npz_b64"])))
    return {
        "vertices": mesh["vertices"],
        "faces": mesh["faces"],
        "colors": mesh["colors"],
        "object_to_camera_p3d": mesh["object_to_camera_p3d"],
        "pins": result.get("pins"),
        "seconds": result.get("seconds"),
        "gpu": result.get("gpu"),
        "model_info": result.get("model_info"),
    }


@functools.lru_cache(maxsize=None)
def _runner(backend: str):
    """One SAM3DObjects instance per process: the weights load once, as completion_ab.py's warm container."""
    if backend == "local":
        return onprem().load_app(workcell_root() / "modal_apps/sam3d_research.py").SAM3DObjects()
    sys.path.insert(0, str(workcell_root() / "modal_apps"))
    import sam3d_research  # under `modal run completion_ab.py` this is the module it imported and included in its app

    return sam3d_research.SAM3DObjects()


def _normalise(out: dict, backend: str) -> dict:
    if "error" in out:
        return out
    pins = out.get("pins") or {}
    return {
        "vertices": out["vertices"],
        "faces": out["faces"],
        "colors": out["colors"],
        "object_to_camera_p3d": out["objectToCamera"],
        "pins": out.get("pins"),
        "seconds": out.get("seconds"),
        "gpu": out.get("gpu"),
        "model_info": {
            "model_id": pins.get("model"),
            "model_revision": pins.get("modelRevision"),
            "code_revision": pins.get("codeRevision"),
            "backend": backend,
        },
    }


__all__ = ["MODEL", "generate"]
