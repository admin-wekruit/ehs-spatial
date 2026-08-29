"""One switchboard for every model service backend.

Each heavyweight service routes through an env-selected backend, and every
backend speaks the SAME request/response contract — so moving from a
vendor API to our Modal deployment to an internal GPU cluster is a config
change, never a code change. An internal serving team implements the
contract below and sets the env vars; nothing else moves.

Services and their env vars:

  SAM3_BACKEND       fal (default) | modal | http
  GEOMETRY_BACKEND   replicate (default) | modal | http   (MapAnything)
  MOGE_BACKEND       modal (default) | replicate | http   (MoGe-3)
  (VLM stays on the Gemini adapter; an OpenAI-compatible internal
   endpoint slots in behind ehs_spatial.providers.gemini later — the
   adapter boundary is already the seam.)

HTTP contract (the internal-GPU case), one POST per call, JSON body:

  SAM3_HTTP_URL      {"image_b64": ..., "prompt": str}            -> {"rle": [...], "scores": [...]}
                     {"image_b64": ..., "box": [x1,y1,x2,y2]}     -> same
  GEOMETRY_HTTP_URL  {"inputs": [dataURI,...], **flags}           -> replicate MapAnything response schema
  MOGE_HTTP_URL      {"image_b64": ...}                           -> {"ply_b64": ..., "intrinsics": [[...]], "fov_x_deg": float}

RLE format everywhere: COCO object form {"size": [H, W], "counts": [...]}
(column-major, zero run first) — ehs_spatial.providers.sam3.decode_coco_rle
/ encode_coco_rle are the reference codecs.
"""

import json
import os
import urllib.request


def service_backend(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default).strip().lower() or default


def http_json(url: str, payload: dict, *, timeout: float = 600.0) -> dict:
    """Minimal JSON POST for internal serving endpoints — stdlib only so
    the contract has zero client dependencies."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


__all__ = ["service_backend", "http_json"]
