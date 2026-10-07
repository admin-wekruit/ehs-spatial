"""Shared pieces of the /v1 model services (docs/BACKENDS-v1.md): X-API-Key auth, /healthz, /v1/info, the in-process job
queue (one worker thread = the one GPU of the container), /v1/jobs/{id}, idempotency by input_sha256 (finished results kept
in memory and, with PANOPTES_SERVICE_CACHE_DIR, on disk), the error shape, one JSON log line per job (no payloads, no keys),
and the result hand-off: s3://PANOPTES_RESULT_BUCKET/<prefix>/<model>/<input_sha256>.tar.gz as result_uri, else result_b64.

A service module does
    service = Service('sam3d', work=run, loaded=True)   # registry.yaml entry 'sam3d'; work(body) -> result dict
    app = service.app                                   # + its own POST routes calling service.submit(...)
Env: PANOPTES_SERVICE_API_KEY (required unless PANOPTES_SERVICE_ALLOW_NO_KEY=1), PANOPTES_SERVICE_MAX_QUEUE (4),
PANOPTES_SERVICE_CACHE_DIR, PANOPTES_RESULT_BUCKET (+ AWS_ENDPOINT_URL and the AWS credentials boto3 reads),
PANOPTES_SERVICE_SYNC_TIMEOUT_S (600), PANOPTES_FAKE_MODEL=1 (deterministic fake backends: CI and the client tests),
PANOPTES_WEIGHTS_DIR (/weights; the fetch_weights*.py cache, SHA-256-checked against registry.yaml at start).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import io
import json
import logging
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
from starlette.exceptions import HTTPException as StarletteHTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry  # noqa: E402

LOG = logging.getLogger('panoptes.v1')
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.WARNING, format='%(message)s', stream=sys.stderr)
LOG.setLevel(logging.INFO)  # the one JSON line per job
FAKE = os.environ.get('PANOPTES_FAKE_MODEL') == '1'
SHA256 = re.compile(r'^[0-9a-f]{64}$')
HERE = Path(__file__).resolve().parent


class ServiceError(Exception):
    """-> {"error": {"code", "message"}} with the HTTP status; raised in a route (now) or in a job (status 'error')."""
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def bad_input(message: str) -> ServiceError:
    return ServiceError(400, 'bad_input', message)


def model_error(message: str) -> ServiceError:
    return ServiceError(500, 'model_error', message)


# ---------------------------------------------------------------- environment facts
def code_sha() -> str:
    """git sha of this service's checkout; PANOPTES_CODE_SHA in the images (the build context carries no .git)."""
    if os.environ.get('PANOPTES_CODE_SHA'):
        return os.environ['PANOPTES_CODE_SHA']
    try:
        return subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def gpu_name() -> str:
    if FAKE:
        return 'fake'
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return 'cpu'


def vram_free_mb() -> int | None:
    if FAKE:
        return None
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info()[0] // 2 ** 20)
    except ImportError:
        pass
    return None


def weights_dir() -> Path:
    return Path(os.environ.get('PANOPTES_WEIGHTS_DIR', '/weights'))


def verify_weights(entry: dict) -> None:
    """Every file registry.yaml pins for the model must sit under PANOPTES_WEIGHTS_DIR with the pinned SHA-256 (fetch_weights*.py
    --cache DIR layout); otherwise the service refuses to start. Skipped in fake mode (no weights, no model)."""
    if FAKE:
        return
    root = weights_dir()
    for ws in entry.get('weights', []):
        for rel, want in ws.get('sha256', {}).items():
            path = root / ws['path'] / rel
            if not path.is_file():
                raise SystemExit(f'{entry["name"]}: weight file {path} missing (fetch it with {ws.get("fetch", "scripts/onprem/fetch_weights*.py")})')
            with path.open('rb') as f:
                got = hashlib.file_digest(f, 'sha256').hexdigest()
            if got != want:
                raise SystemExit(f'{entry["name"]}: {path} SHA-256 {got} is not the pinned {want}; refusing to start')


# ---------------------------------------------------------------- binary fields: <name>_b64 | <name>_uri (s3://)
def b64decode(text: str, name: str) -> bytes:
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        raise bad_input(f'{name}: not valid base64 ({error})') from None


def _s3():
    import boto3  # only installed / needed where buckets are used
    return boto3.client('s3', endpoint_url=os.environ.get('AWS_ENDPOINT_URL') or None)


def _split_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith('s3://') or '/' not in uri[5:]:
        raise bad_input(f'{uri!r}: only s3://bucket/key is accepted')
    bucket, _, key = uri[5:].partition('/')
    return bucket, key


def read_uri(uri: str) -> bytes:
    bucket, key = _split_uri(uri)
    try:
        return _s3().get_object(Bucket=bucket, Key=key)['Body'].read()
    except Exception as error:  # noqa: BLE001 - the client sees the type, never the credentials
        raise bad_input(f'{uri}: {type(error).__name__}') from None


def read_binary(body: dict, name: str, required: bool = True) -> bytes | None:
    """body[name_b64] (base64 of the file bytes) or body[name_uri] (s3://bucket/key, read with the service's credentials)."""
    if body.get(f'{name}_b64') is not None:
        return b64decode(body[f'{name}_b64'], f'{name}_b64')
    if body.get(f'{name}_uri') is not None:
        return read_uri(str(body[f'{name}_uri']))
    if required:
        raise bad_input(f'{name}_b64 or {name}_uri is required')
    return None


def npz(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def load_npz(data: bytes, name: str) -> dict:
    try:
        with np.load(io.BytesIO(data)) as d:
            return {k: d[k] for k in d.files}
    except (OSError, ValueError, EOFError) as error:
        raise bad_input(f'{name}: not an npz ({error})') from None


def publish_result(model: str, input_sha256: str, data: bytes, suffix: str = '.tar.gz') -> dict:
    """Large results: uploaded as s3://<bucket>/<prefix>/<model>/<input_sha256><suffix> when PANOPTES_RESULT_BUCKET is set
    (bucket or bucket/prefix), else inline result_b64."""
    target = os.environ.get('PANOPTES_RESULT_BUCKET')
    if not target:
        return {'result_b64': base64.b64encode(data).decode('ascii')}
    bucket, _, prefix = target.partition('/')
    key = '/'.join(filter(None, [prefix.strip('/'), model, f'{input_sha256}{suffix}']))
    _s3().put_object(Bucket=bucket, Key=key, Body=data)
    return {'result_uri': f's3://{bucket}/{key}'}


# ---------------------------------------------------------------- jobs
class Job:
    __slots__ = ('job_id', 'input_sha256', 'status', 'result', 'error', 'seconds', 'done')

    def __init__(self, input_sha256: str):
        self.job_id, self.input_sha256 = uuid.uuid4().hex, input_sha256
        self.status, self.result, self.error, self.seconds = 'queued', None, None, None
        self.done = threading.Event()


class Service:
    def __init__(self, model: str, loaded: bool = True):
        self.model, self.entry = model, registry.entry(model)
        self.key = os.environ.get('PANOPTES_SERVICE_API_KEY') or ''
        if not self.key and os.environ.get('PANOPTES_SERVICE_ALLOW_NO_KEY') != '1':
            raise SystemExit('PANOPTES_SERVICE_API_KEY is not set (PANOPTES_SERVICE_ALLOW_NO_KEY=1 to run without a key)')
        self.max_queue = int(os.environ.get('PANOPTES_SERVICE_MAX_QUEUE', '4'))
        self.sync_timeout = float(os.environ.get('PANOPTES_SERVICE_SYNC_TIMEOUT_S', '600'))
        cache = os.environ.get('PANOPTES_SERVICE_CACHE_DIR')
        self.cache_dir = Path(cache) / model if cache else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaded, self.gpu = loaded, gpu_name()
        self.info = registry.info(self.entry, code_sha())
        self.jobs: dict[str, Job] = {}
        self.by_sha: dict[str, Job] = {}
        self.queue: queue.Queue = queue.Queue()
        self.running, self.lock = 0, threading.Lock()
        threading.Thread(target=self._worker, name=f'{model}-worker', daemon=True).start()
        self.app = self._app()

    # ---- the queue
    def submit(self, input_sha256: str, work) -> tuple[Job, bool]:
        """Idempotent by input_sha256: the same key returns the finished (or still queued / running) job; (job, cached)."""
        if not isinstance(input_sha256, str) or not SHA256.match(input_sha256):
            raise bad_input('input_sha256 must be 64 lowercase hex characters')
        with self.lock:
            job = self.by_sha.get(input_sha256) or self._from_disk(input_sha256)
            if job is not None:
                return job, job.status == 'done'
            if self.queue.qsize() >= self.max_queue:
                raise ServiceError(429, 'busy', f'queue_depth {self.queue.qsize()} >= PANOPTES_SERVICE_MAX_QUEUE {self.max_queue}')
            job = Job(input_sha256)
            self.jobs[job.job_id] = self.by_sha[input_sha256] = job
            self.queue.put((job, work))
            return job, False

    def wait(self, job: Job) -> dict:
        """Synchronous calls: the job's result, or its error raised; the queue still serialises the GPU."""
        if not job.done.wait(self.sync_timeout):
            raise ServiceError(504, 'model_error', f'sync call exceeded PANOPTES_SERVICE_SYNC_TIMEOUT_S={self.sync_timeout:g}; poll /v1/jobs/{job.job_id}')
        if job.status == 'error':
            raise ServiceError(500 if job.error['code'] == 'model_error' else 400, job.error['code'], job.error['message'])
        return job.result

    def record(self, job: Job) -> dict:
        return dict(job_id=job.job_id, input_sha256=job.input_sha256, status=job.status, result=job.result, error=job.error,
                    seconds=job.seconds, model_info=self.info, gpu=self.gpu)

    def _worker(self):
        while True:
            job, work = self.queue.get()
            with self.lock:
                self.running, job.status = self.running + 1, 'running'
            started = time.monotonic()
            try:
                result = work()
                job.seconds = round(time.monotonic() - started, 3)
                result.update(seconds=job.seconds, gpu=self.gpu, model_info=self.info)
                job.result, job.status = result, 'done'
            except ServiceError as error:
                job.error, job.status = {'code': error.code, 'message': error.message}, 'error'
            except Exception as error:  # noqa: BLE001 - reported to the client as model_error, never re-raised into the worker
                LOG.exception('job %s failed', job.job_id)
                job.error, job.status = {'code': 'model_error', 'message': f'{type(error).__name__}: {error}'[-2000:]}, 'error'
            if job.seconds is None:
                job.seconds = round(time.monotonic() - started, 3)
            with self.lock:
                self.running -= 1
                if job.status == 'error' and self.by_sha.get(job.input_sha256) is job:
                    del self.by_sha[job.input_sha256]  # errors are not cached: the same input may be sent again
            if job.status == 'done' and self.cache_dir:
                try:
                    (self.cache_dir / f'{job.input_sha256}.json').write_text(json.dumps(self.record(job)))
                except OSError as error:
                    LOG.warning('result cache write failed: %s', error)
            LOG.info(json.dumps(dict(job_id=job.job_id, input_sha256=job.input_sha256, seconds=job.seconds, gpu=self.gpu, status=job.status)))
            job.done.set()

    def _from_disk(self, input_sha256: str) -> Job | None:
        path = self.cache_dir / f'{input_sha256}.json' if self.cache_dir else None
        if not path or not path.is_file():
            return None
        row = json.loads(path.read_text())
        job = Job(input_sha256)
        job.job_id, job.status, job.result, job.seconds = row['job_id'], 'done', row['result'], row['seconds']
        job.done.set()
        self.jobs[job.job_id] = self.by_sha[input_sha256] = job
        return job

    # ---- the app
    def _app(self) -> FastAPI:
        app = FastAPI(title=f'{self.model}-service', docs_url=None, redoc_url=None)
        service = self

        @app.middleware('http')
        async def auth(request: Request, call_next):
            given = request.headers.get('x-api-key', '')
            if service.key and not hmac.compare_digest(given.encode(), service.key.encode()):
                return error_response(401, 'unauthorized', 'X-API-Key header missing or wrong')
            return await call_next(request)

        @app.exception_handler(ServiceError)
        async def service_error(request, error: ServiceError):
            return error_response(error.status, error.code, error.message)

        @app.exception_handler(RequestValidationError)
        async def validation_error(request, error: RequestValidationError):
            first = error.errors()[0] if error.errors() else {}
            where = '.'.join(str(p) for p in first.get('loc', ()) if p != 'body')
            return error_response(400, 'bad_input', f'{where}: {first.get("msg", "invalid")}' if where else 'invalid body')

        @app.exception_handler(StarletteHTTPException)
        async def http_error(request, error: StarletteHTTPException):
            code = {404: 'not_found', 401: 'unauthorized', 429: 'busy'}.get(error.status_code, 'bad_input' if error.status_code < 500 else 'model_error')
            return error_response(error.status_code, code, str(error.detail))

        @app.exception_handler(Exception)
        async def unhandled(request, error: Exception):
            LOG.exception('unhandled error')
            return error_response(500, 'model_error', f'{type(error).__name__}: {error}'[-2000:])

        @app.get('/healthz')
        def healthz():
            body = dict(ok=service.loaded, model=service.model, revision=service.info['model_revision'], gpu=service.gpu,
                        vram_free_mb=vram_free_mb(), queue_depth=service.queue.qsize(), running=service.running)
            return JSONResponse(body, status_code=200 if service.loaded else 503)

        @app.get('/v1/info')
        def info():
            return service.info

        @app.get('/v1/jobs/{job_id}')
        def job(job_id: str):
            found = service.jobs.get(job_id)
            if found is None:
                raise ServiceError(404, 'not_found', f'job {job_id}')
            return service.record(found)

        return app

    def accepted(self, job: Job, cached: bool) -> JSONResponse:
        body = dict(job_id=job.job_id, input_sha256=job.input_sha256, cached=cached)
        if cached:
            body.update(status='done', result=job.result)
        return JSONResponse(body, status_code=202)


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({'error': {'code': code, 'message': message}}, status_code=status)


def _check():
    """Self-check of the shared layer in fake mode: auth, healthz, idempotency, busy, error shape, result cache on disk."""
    import tempfile
    from fastapi.testclient import TestClient
    os.environ.update(PANOPTES_SERVICE_API_KEY='k', PANOPTES_SERVICE_MAX_QUEUE='1', PANOPTES_FAKE_MODEL='1')
    with tempfile.TemporaryDirectory() as tmp:
        os.environ['PANOPTES_SERVICE_CACHE_DIR'] = tmp
        s = Service('sam3d')
        gate = threading.Event()

        @s.app.post('/v1/x/jobs')
        def post(body: dict):
            job, cached = s.submit(body['input_sha256'], lambda: (gate.wait(5), {'v': body['input_sha256'][:4]})[1])
            return s.accepted(job, cached)
        c = TestClient(s.app, raise_server_exceptions=False)
        assert c.get('/healthz').status_code == 401 and c.get('/healthz').json()['error']['code'] == 'unauthorized'
        h = {'X-API-Key': 'k'}
        assert c.get('/healthz', headers=h).json()['ok'] is True and c.get('/v1/jobs/nope', headers=h).status_code == 404
        a, b, d = 'a' * 64, 'b' * 64, 'd' * 64
        r1 = c.post('/v1/x/jobs', json={'input_sha256': a}, headers=h); assert r1.status_code == 202 and r1.json()['cached'] is False
        time.sleep(.1)  # the worker takes job a (blocked on the gate); b queues; c is busy
        assert c.post('/v1/x/jobs', json={'input_sha256': b}, headers=h).status_code == 202
        busy = c.post('/v1/x/jobs', json={'input_sha256': 'c' * 64}, headers=h); assert busy.status_code == 429 and busy.json()['error']['code'] == 'busy', busy.text
        assert c.post('/v1/x/jobs', json={'input_sha256': 'zz'}, headers=h).json()['error']['code'] == 'bad_input'
        gate.set(); s.jobs[r1.json()['job_id']].done.wait(5); time.sleep(.2)
        rec = c.get(f"/v1/jobs/{r1.json()['job_id']}", headers=h).json()
        assert rec['status'] == 'done' and rec['result']['v'] == 'aaaa' and rec['model_info'] == s.info and rec['gpu'] == s.gpu, rec
        again = c.post('/v1/x/jobs', json={'input_sha256': a}, headers=h).json()
        assert again['cached'] is True and again['job_id'] == r1.json()['job_id'] and again['result']['v'] == 'aaaa'
        assert (Path(tmp) / 'sam3d' / f'{a}.json').is_file()
        s.by_sha.clear(); s.jobs.clear()  # a fresh process finds it on disk
        assert c.post('/v1/x/jobs', json={'input_sha256': a}, headers=h).json()['cached'] is True
        assert c.post('/v1/x/jobs', json={'input_sha256': d}, headers=h).json()['cached'] is False
    print('v1_common self-test passed')


if __name__ == '__main__' and sys.argv[1:] == ['check']:
    _check()
