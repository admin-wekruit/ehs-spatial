"""Contract v1 client (docs/BACKENDS-v1.md), stdlib only, shared by providers/sam3d.py and providers/geometry_mvs.py.

  auth      X-API-Key = PANOPTES_SERVICE_API_KEY on every request
  routing   <MODEL>_HTTP_URLS (comma list, e.g. SAM3D_HTTP_URLS): the root with the smallest /healthz queue_depth gets
            the call; /healthz answers are cached HEALTH_TTL_S; a root that is down or ok:false is skipped
  retries   PANOPTES_SERVICE_RETRIES (default 3) retries with exponential backoff on connection errors, 429 busy and 503
            only; unauthorized / bad_input / model_error / not_found raise ServiceError at once
  jobs      POST /v1/<model>/jobs -> 202 {job_id, input_sha256, cached}; GET /v1/jobs/{id} polled with backoff until
            done | error, bounded by PANOPTES_SERVICE_TIMEOUT_S (default 1800, also the per-request timeout)
  key       input_sha256(body): the idempotency key over the canonical input bytes
  ledger    every result carries model_info (the /v1/info document); last_model_info keeps the latest for the caller
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

from .base import ProviderError

RETRY_STATUS = (429, 503)
HEALTH_TTL_S = 2.0
_health: dict[str, tuple[float, int | None]] = {}  # root -> (checked at, queue_depth | None when unusable)
last_model_info: dict | None = None


class ServiceError(ProviderError):
    """A contract error document ({"error": {"code", "message"}}), a failed job, or an exhausted retry."""

    def __init__(self, operation: str, code: str, message: str, status: int | None = None, job: dict | None = None):
        self.code, self.status, self.job = code, status, job or {}
        super().__init__("service", operation, f"{code}: {message}")


def timeout_s() -> float:
    return float(os.environ.get("PANOPTES_SERVICE_TIMEOUT_S") or 1800)


def retries() -> int:
    return int(os.environ.get("PANOPTES_SERVICE_RETRIES") or 3)


def input_sha256(body: dict) -> str:
    """The idempotency key: sha256 of the compact, key-sorted JSON of the non-binary fields (input_sha256 itself left
    out) followed by the raw bytes of every *_b64 field, in field-name order (nested: by sorted key within an object,
    by index within a list, e.g. frames[0].alpha_npy_b64, frames[0].canonical_png_b64, frames[1]...)."""
    blobs: list[bytes] = []

    def plain(node, top=False):
        if isinstance(node, dict):
            out = {}
            for key in sorted(node):
                if top and key == "input_sha256":
                    continue
                if key.endswith("_b64"):
                    blobs.append(base64.b64decode(node[key]))
                    continue
                out[key] = plain(node[key])
            return out
        if isinstance(node, list):
            return [plain(v) for v in node]
        return node

    digest = hashlib.sha256(json.dumps(plain(body, top=True), sort_keys=True, separators=(",", ":")).encode())
    for data in blobs:
        digest.update(data)
    return digest.hexdigest()


def urls(model: str) -> list[str]:
    env = model.upper().replace("-", "_") + "_HTTP_URLS"
    out = [u.strip().rstrip("/") for u in os.environ.get(env, "").split(",") if u.strip()]
    if not out:
        raise ProviderError("service", model, f"{env} is not set")
    return out


def _http(method: str, url: str, body: dict | None = None, *, timeout: float | None = None) -> tuple[int, object]:
    """One exchange -> (status, json). Connection problems raise OSError (urllib wraps them); HTTP errors return
    their status with the error document (or a synthetic one when the body is not JSON)."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"X-API-Key": os.environ.get("PANOPTES_SERVICE_API_KEY", ""), "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout or timeout_s()) as response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, {"error": {"code": f"http_{error.code}", "message": raw[:500]}}


def request(method: str, url: str, body: dict | None = None, *, operation: str, timeout: float | None = None) -> dict:
    """The exchange with the contract's retry policy; an error document becomes ServiceError."""
    delay = 1.0
    for attempt in range(retries() + 1):
        try:
            status, doc = _http(method, url, body, timeout=timeout)
        except OSError as error:  # connection refused / reset / DNS / timeout
            if attempt == retries():
                raise ServiceError(operation, "connection", f"{url}: {error}") from error
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        if status in RETRY_STATUS and attempt < retries():
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        if status >= 400 or (isinstance(doc, dict) and set(doc) == {"error"}):  # a job document carries error beside status
            error = doc.get("error") if isinstance(doc, dict) else None
            error = error if isinstance(error, dict) else {}
            raise ServiceError(operation, error.get("code", f"http_{status}"), error.get("message", str(doc)[:500]), status)
        return doc
    raise AssertionError("unreachable")


def pick(model: str) -> str:
    """The service root with the smallest queue_depth (ties: first in the list); roots that are down or report ok:false
    are skipped; when every root is down the first one gets the call (its own retries then decide)."""
    roots = urls(model)
    now = time.monotonic()
    depths: dict[str, int] = {}
    for root in roots:
        checked, depth = _health.get(root, (None, None))
        if checked is None or now - checked > HEALTH_TTL_S:
            try:
                status, doc = _http("GET", root + "/healthz", timeout=10)
                depth = int(doc.get("queue_depth", 0)) if status == 200 and isinstance(doc, dict) and doc.get("ok") else None
            except OSError:
                depth = None
            _health[root] = (now, depth)
        if depth is not None:
            depths[root] = depth
    return min(depths, key=depths.get) if depths else roots[0]


def wait(root: str, job_id: str, *, operation: str) -> dict:
    """GET /v1/jobs/{id} until done (the job document) or error (ServiceError with the job attached)."""
    deadline = time.monotonic() + timeout_s()
    delay = 1.0
    while True:
        job = request("GET", f"{root}/v1/jobs/{job_id}", operation=operation, timeout=60)
        status = job.get("status")
        if status == "done":
            return job
        if status == "error":
            error = job.get("error") or {}
            raise ServiceError(operation, error.get("code", "model_error"), error.get("message", ""), job=job)
        if time.monotonic() > deadline:
            raise ServiceError(operation, "timeout", f"job {job_id} still {status} after {timeout_s():g} s")
        time.sleep(delay)
        delay = min(delay * 2, 15)


def call(model: str, body: dict, *, sync: bool = False, operation: str | None = None) -> dict:
    """One contract call: the body gets its input_sha256; sync -> POST /v1/<model>, else /v1/<model>/jobs + polling.
    Returns the result with model_info / seconds / gpu, input_sha256, cached and service_url filled in."""
    global last_model_info
    operation = operation or model
    body = dict(body)
    body["input_sha256"] = body.get("input_sha256") or input_sha256(body)
    root = pick(model)
    if sync:
        job: dict = {}
        result = dict(request("POST", f"{root}/v1/{model}", body, operation=operation))
    else:
        job = request("POST", f"{root}/v1/{model}/jobs", body, operation=operation, timeout=min(timeout_s(), 300))
        done = wait(root, job["job_id"], operation=operation)
        result = dict(done.get("result") or {})
        for key in ("model_info", "seconds", "gpu"):
            result.setdefault(key, done.get(key))
    result.update(input_sha256=body["input_sha256"], cached=bool(job.get("cached", False)), service_url=root)
    last_model_info = result.get("model_info")
    return result


def fetch_result(result: dict, *, operation: str = "result") -> bytes:
    """A large output: result_b64 inline, or result_uri (http(s) presigned, or s3:// through boto3 with the ambient
    AWS_ENDPOINT_URL / credentials)."""
    if result.get("result_b64"):
        return base64.b64decode(result["result_b64"])
    uri = result.get("result_uri")
    if not uri:
        raise ServiceError(operation, "bad_result", "neither result_b64 nor result_uri in the result")
    if uri.startswith(("http://", "https://")):
        with urllib.request.urlopen(uri, timeout=timeout_s()) as response:
            return response.read()
    if uri.startswith("s3://"):
        import boto3  # ponytail: only the s3 result path needs it; the http path stays dependency-free

        bucket, _, key = uri[5:].partition("/")
        return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    raise ServiceError(operation, "bad_result", f"unsupported result_uri {uri[:80]}")


__all__ = ["ServiceError", "call", "fetch_result", "input_sha256", "pick", "request", "urls", "wait"]
