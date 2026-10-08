"""SAM3D_BACKEND=modal|http; RGB, mask, external pointmap and seed in, posed mesh and provenance out."""

from __future__ import annotations

import base64
import functools
import io
import sys

import numpy as np

from argus.pipeline.backends import service_backend
from argus.providers import service_client
from argus.providers.base import ProviderError

MODEL = "sam3d"


def generate(rgb: np.ndarray, mask: np.ndarray, pointmap: np.ndarray, seed: int, *, sync: bool = True) -> dict:
    backend = service_backend("SAM3D_BACKEND", "modal")
    rgb, mask, pointmap = np.asarray(rgb, np.uint8), np.asarray(mask, bool), np.asarray(pointmap, np.float32)
    if rgb.shape[:2] != mask.shape or pointmap.shape != rgb.shape[:2] + (3,):
        raise ValueError("image, mask and pointmap grids differ")
    if backend == "http":
        return _http(rgb, mask, pointmap, int(seed), sync)
    if backend == "modal":
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
    import argus.providers.sam3d_modal as sam3d_research

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
