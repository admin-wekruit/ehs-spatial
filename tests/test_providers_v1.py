"""Contract v1 providers (docs/BACKENDS-v1.md) against an in-process fake service: least-queue selection across two
roots, retry on 429 then success, no retry on model_error / unauthorized, idempotent resubmission, job polling to done,
the geometry tar's layout; and the local / modal backends' plumbing with fake run_stage / app modules (no GPU here)."""

import base64
import http.server
import io
import json
import sys
import tarfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from ehs_spatial.providers import geometry_mvs, sam3d, service_client

INFO = {"model_id": "fake/model", "model_revision": "abc", "code_revision": "def", "weights_sha256": {}, "licence": "test",
        "code_sha": "0", "seed_policy": "seed in request"}


class FakeService:
    """One contract-v1 model service on 127.0.0.1 (thread + http.server): /healthz, /v1/info, POST /v1/<model>,
    POST /v1/<model>/jobs (input_sha256 = idempotency key), GET /v1/jobs/{id} (queued -> running -> done | error)."""

    def __init__(self, model, *, key="test-key", queue_depth=0, busy_first=0, model_error=False, polls=2):
        self.model, self.key, self.queue_depth, self.busy_left, self.model_error, self.polls = model, key, queue_depth, busy_first, model_error, polls
        self.posts, self.jobs, self.by_key, self.seen = [], {}, {}, []
        service = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send(self, status, doc):
                body = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.headers.get("X-API-Key") != service.key:
                    return self.send(401, {"error": {"code": "unauthorized", "message": "bad key"}})
                if self.path == "/healthz":
                    return self.send(200, {"ok": True, "model": service.model, "revision": "abc", "gpu": "FAKE", "vram_free_mb": 1,
                                           "queue_depth": service.queue_depth, "running": 0})
                if self.path == "/v1/info":
                    return self.send(200, INFO)
                if self.path.startswith("/v1/jobs/"):
                    job = service.jobs.get(self.path.rsplit("/", 1)[1])
                    if not job:
                        return self.send(404, {"error": {"code": "not_found", "message": "no such job"}})
                    job["polls"] += 1
                    common = {"seconds": 9.8, "model_info": INFO, "gpu": "FAKE"}
                    if job["polls"] < service.polls:
                        return self.send(200, {"status": "queued" if job["polls"] == 1 else "running", "result": None, "error": None, **common})
                    if service.model_error:
                        return self.send(200, {"status": "error", "result": None, "error": {"code": "model_error", "message": "Traceback: boom"}, **common})
                    return self.send(200, {"status": "done", "result": service.result(job["body"]), "error": None, **common})
                self.send(404, {"error": {"code": "not_found", "message": self.path}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                service.posts.append((self.path, body))
                if self.headers.get("X-API-Key") != service.key:
                    return self.send(401, {"error": {"code": "unauthorized", "message": "bad key"}})
                if service.busy_left > 0:
                    service.busy_left -= 1
                    return self.send(429, {"error": {"code": "busy", "message": "queue full"}})
                if self.path == f"/v1/{service.model}":
                    if service.model_error:
                        return self.send(500, {"error": {"code": "model_error", "message": "Traceback: boom"}})
                    return self.send(200, {**service.result(body), "seconds": 9.8, "gpu": "FAKE", "model_info": INFO})
                if self.path == f"/v1/{service.model}/jobs":
                    key = body["input_sha256"]
                    if key in service.by_key:
                        return self.send(202, {"job_id": service.by_key[key], "input_sha256": key, "cached": True})
                    job_id = f"job{len(service.jobs) + 1}"
                    service.jobs[job_id], service.by_key[key] = {"body": body, "polls": 0}, job_id
                    return self.send(202, {"job_id": job_id, "input_sha256": key, "cached": False})
                self.send(404, {"error": {"code": "not_found", "message": self.path}})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def result(self, body):
        if self.model == "sam3d":
            from PIL import Image

            rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["image_b64"]))))
            mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["mask_b64"])))) > 0
            pointmap = np.load(io.BytesIO(base64.b64decode(body["pointmap_npz_b64"])))["pointmap"]
            self.seen.append((rgb.shape, mask.sum(), pointmap.shape, body["seed"]))
            buffer = io.BytesIO()
            np.savez(buffer, vertices=pointmap[mask][:3].astype(np.float32), faces=np.array([[0, 1, 2]], np.int32),
                     colors=rgb[mask][:3].astype(np.uint8), object_to_camera_p3d=np.eye(4) * body["seed"])
            return {"mesh_npz_b64": base64.b64encode(buffer.getvalue()).decode(), "vertices": 3, "faces": 1,
                    "pins": {"model": "facebook/sam-3d-objects"}}
        cell = body["cell"]
        self.seen.append(body)
        files = {}
        for i, j in ((0, 1),):
            files[f"checks/clean-gpu/{cell}/roma-{i}-{j}.npz"] = npz(uvA=np.zeros((2, 2)), uvB=np.zeros((2, 2)))
            files[f"checks/clean-gpu/{cell}/dense-{i}-{j}.npz"] = npz(uvAB=np.zeros((2, 2)))
        for f in body["frames"]:
            files[f"checks/clean-gpu/{cell}/moge-{f['frame_id']}.npz"] = npz(points=np.zeros((2, 2, 3)))
            for geom in (f"da3fair-geom/{cell}-da3-base-padded", f"clean-geom/{cell}-da3-base-ba-f"):
                for name in ("pts3d", "conf", "valid_mask", "intrinsics", "camera_to_world"):
                    files[f"checks/{geom}/geometry/frames/{f['frame_id']}/{name}.npy"] = npy(np.zeros(3))
        for geom in (f"da3fair-geom/{cell}-da3-base-padded", f"clean-geom/{cell}-da3-base-ba-f"):
            files[f"checks/{geom}/geometry/candidate_manifest.json"] = b"{}"
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return {"result_b64": base64.b64encode(buffer.getvalue()).decode()}

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def npz(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def npy(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


@pytest.fixture
def sleeps(monkeypatch):
    """No real waiting: every backoff / poll sleep is recorded instead."""
    recorded = []
    monkeypatch.setattr(service_client.time, "sleep", recorded.append)
    monkeypatch.setenv("PANOPTES_SERVICE_API_KEY", "test-key")
    monkeypatch.setenv("PANOPTES_SERVICE_RETRIES", "3")
    monkeypatch.setenv("PANOPTES_SERVICE_TIMEOUT_S", "30")
    service_client._health.clear()
    service_client.last_model_info = None
    return recorded


@pytest.fixture
def services():
    started = []

    def start(*args, **kwargs):
        started.append(FakeService(*args, **kwargs))
        return started[-1]

    yield start
    for s in started:
        s.close()


def inputs(seed=42):
    rgb = np.full((8, 6, 3), 90, np.uint8)
    mask = np.zeros((8, 6), bool)
    mask[2:6, 1:5] = True
    pointmap = np.full((8, 6, 3), np.nan, np.float32)
    pointmap[mask] = [0.1, 0.2, 2.0]
    return rgb, mask, pointmap, seed


# ---------------------------------------------------------------- the key
def test_input_sha256_is_canonical():
    blob = base64.b64encode(b"\x00\x01binary").decode()
    a = {"seed": 42, "image_b64": blob, "options": {"roma": "outdoor", "start": "da3-base"}, "input_sha256": "ignored"}
    b = {"options": {"start": "da3-base", "roma": "outdoor"}, "image_b64": blob, "seed": 42}
    assert service_client.input_sha256(a) == service_client.input_sha256(b)
    assert service_client.input_sha256({**b, "seed": 43}) != service_client.input_sha256(b)
    assert service_client.input_sha256({**b, "image_b64": base64.b64encode(b"\x00\x02binary").decode()}) != service_client.input_sha256(b)
    nested = {"cell": "090", "frames": [{"frame_id": "f1", "alpha_npy_b64": blob, "canonical_png_b64": blob}]}
    assert len(service_client.input_sha256(nested)) == 64


# ---------------------------------------------------------------- http backend, sam3d
def test_least_queue_selection_across_two_roots(monkeypatch, sleeps, services):
    busy, idle = services("sam3d", queue_depth=3), services("sam3d", queue_depth=0)
    monkeypatch.setenv("SAM3D_BACKEND", "http")
    monkeypatch.setenv("SAM3D_HTTP_URLS", f"{busy.url}, {idle.url}/")
    out = sam3d.generate(*inputs())
    assert [p[0] for p in idle.posts] == ["/v1/sam3d"] and not busy.posts
    assert idle.seen == [((8, 6, 3), 16, (8, 6, 3), 42)]
    assert set(out) == {"vertices", "faces", "colors", "object_to_camera_p3d", "pins", "seconds", "gpu", "model_info"}
    assert out["object_to_camera_p3d"].shape == (4, 4) and out["object_to_camera_p3d"][0, 0] == 42
    assert out["vertices"].dtype == np.float32 and out["colors"].dtype == np.uint8
    assert out["model_info"] == INFO and service_client.last_model_info == INFO and out["seconds"] == 9.8
    assert sleeps == []
    # the /healthz answers are cached for HEALTH_TTL_S; once they expire the roots are re-ranked
    idle.queue_depth, busy.queue_depth = 5, 0
    sam3d.generate(*inputs())
    assert len(idle.posts) == 2 and not busy.posts
    service_client._health.clear()
    sam3d.generate(*inputs())
    assert len(idle.posts) == 2 and [p[0] for p in busy.posts] == ["/v1/sam3d"]


def test_retry_on_busy_then_success(monkeypatch, sleeps, services):
    service = services("sam3d", busy_first=1)
    monkeypatch.setenv("SAM3D_BACKEND", "http")
    monkeypatch.setenv("SAM3D_HTTP_URLS", service.url)
    out = sam3d.generate(*inputs())
    assert "error" not in out and len(service.posts) == 2 and sleeps == [1.0]


def test_connection_error_is_retried_then_raised(monkeypatch, sleeps):
    monkeypatch.setenv("SAM3D_HTTP_URLS", "http://127.0.0.1:9")  # nothing listens on the discard port
    monkeypatch.setenv("PANOPTES_SERVICE_RETRIES", "2")
    with pytest.raises(service_client.ServiceError) as error:
        service_client.call("sam3d", {"seed": 1}, sync=True)
    assert error.value.code == "connection" and sleeps == [1.0, 2.0]


def test_no_retry_on_model_error(monkeypatch, sleeps, services):
    service = services("sam3d", model_error=True)
    monkeypatch.setenv("SAM3D_BACKEND", "http")
    monkeypatch.setenv("SAM3D_HTTP_URLS", service.url)
    out = sam3d.generate(*inputs())
    assert set(out) == {"error", "seconds", "gpu"} and "model_error" in out["error"]  # journalled by the caller, as the class reports it
    assert len(service.posts) == 1 and sleeps == []
    with pytest.raises(service_client.ServiceError) as error:
        service_client.call("sam3d", {"seed": 1}, sync=True)
    assert error.value.code == "model_error" and error.value.status == 500 and len(service.posts) == 2 and sleeps == []


def test_unauthorized_is_not_retried(monkeypatch, sleeps, services):
    service = services("sam3d")
    monkeypatch.setenv("PANOPTES_SERVICE_API_KEY", "wrong")
    monkeypatch.setenv("SAM3D_BACKEND", "http")
    monkeypatch.setenv("SAM3D_HTTP_URLS", service.url)
    with pytest.raises(service_client.ServiceError) as error:
        sam3d.generate(*inputs())
    assert error.value.code == "unauthorized" and error.value.status == 401
    assert len(service.posts) == 1 and sleeps == []


def test_sam3d_jobs_polling_to_done(monkeypatch, sleeps, services):
    service = services("sam3d", polls=3)
    monkeypatch.setenv("SAM3D_BACKEND", "http")
    monkeypatch.setenv("SAM3D_HTTP_URLS", service.url)
    out = sam3d.generate(*inputs(seed=7), sync=False)
    assert [p[0] for p in service.posts] == ["/v1/sam3d/jobs"]
    assert service.jobs["job1"]["polls"] == 3 and sleeps == [1.0, 2.0]  # queued, running, done
    assert out["object_to_camera_p3d"][1, 1] == 7 and out["model_info"] == INFO and out["gpu"] == "FAKE"
    # a failed job: model_error comes back as data, never re-sent
    service.model_error, service.posts[:] = True, []
    out = sam3d.generate(*inputs(seed=8), sync=False)
    assert "model_error" in out["error"] and out["seconds"] == 9.8 and out["gpu"] == "FAKE" and len(service.posts) == 1


def test_sam3d_rejects_mismatched_grids(monkeypatch):
    monkeypatch.setenv("SAM3D_BACKEND", "http")
    rgb, mask, pointmap, seed = inputs()
    with pytest.raises(ValueError, match="grids differ"):
        sam3d.generate(rgb, mask[:4], pointmap, seed)


def test_unknown_backend_fails_loudly(monkeypatch):
    monkeypatch.setenv("SAM3D_BACKEND", "banana")
    monkeypatch.setenv("GEOMETRY_MVS_BACKEND", "banana")
    with pytest.raises(ValueError, match="banana"):
        sam3d.generate(*inputs())
    with pytest.raises(ValueError, match="banana"):
        geometry_mvs.run("090", [])


# ---------------------------------------------------------------- http backend, geometry-mvs
def frames(n=2, size=8):
    from PIL import Image

    out = []
    for i in range(1, n + 1):
        buffer = io.BytesIO()
        Image.fromarray(np.full((size, size, 3), 10 * i, np.uint8)).save(buffer, format="PNG")
        alpha = np.zeros((size, size), bool)
        alpha[:, 1:-1] = True
        out.append({"frame_id": f"frame_000{i}", "canonical_png": buffer.getvalue(), "alpha": alpha})
    return out


def test_geometry_jobs_idempotent_resubmission_and_tar_layout(monkeypatch, sleeps, services, tmp_path):
    service = services("geometry-mvs", polls=2)
    monkeypatch.setenv("GEOMETRY_MVS_BACKEND", "http")
    monkeypatch.setenv("GEOMETRY_MVS_HTTP_URLS", service.url)
    dest = geometry_mvs.run("090", frames(), dest=tmp_path / "a")
    assert dest == tmp_path / "a"
    for rel in ("checks/clean-gpu/090/roma-0-1.npz", "checks/clean-gpu/090/dense-0-1.npz", "checks/clean-gpu/090/moge-frame_0001.npz",
                "checks/clean-gpu/090/moge-frame_0002.npz", "checks/da3fair-geom/090-da3-base-padded/geometry/frames/frame_0001/pts3d.npy",
                "checks/da3fair-geom/090-da3-base-padded/geometry/candidate_manifest.json",
                "checks/clean-geom/090-da3-base-ba-f/geometry/frames/frame_0002/camera_to_world.npy"):
        assert (dest / rel).is_file(), rel
    body = service.seen[0]
    assert body["cell"] == "090" and body["options"] == geometry_mvs.OPTIONS and len(body["input_sha256"]) == 64
    assert [f["frame_id"] for f in body["frames"]] == ["frame_0001", "frame_0002"]
    assert base64.b64decode(body["frames"][0]["canonical_png_b64"]) == frames()[0]["canonical_png"]
    alpha = np.load(io.BytesIO(base64.b64decode(body["frames"][1]["alpha_npy_b64"])))
    assert alpha.dtype == bool and alpha.shape == (8, 8) and alpha.sum() == 48
    assert service.jobs["job1"]["polls"] == 2 and sleeps == [1.0]
    assert service_client.last_model_info == INFO
    # the same frames again: the same key, the finished job comes back (cached), no new GPU work
    geometry_mvs.run("090", frames(), dest=tmp_path / "b")
    assert [p[0] for p in service.posts] == ["/v1/geometry-mvs/jobs"] * 2
    assert len(service.jobs) == 1 and (tmp_path / "b/checks/clean-gpu/090/roma-0-1.npz").is_file()
    # a different seed-equivalent input (another frame) is a different key
    geometry_mvs.run("090", frames(n=1), dest=tmp_path / "c")
    assert len(service.jobs) == 2


def test_geometry_incomplete_tar_is_refused(monkeypatch, sleeps, services, tmp_path):
    service = services("geometry-mvs")
    service.result = lambda body: {"result_b64": base64.b64encode(b"").decode()}
    monkeypatch.setenv("GEOMETRY_MVS_BACKEND", "http")
    monkeypatch.setenv("GEOMETRY_MVS_HTTP_URLS", service.url)
    with pytest.raises(Exception):
        geometry_mvs.run("090", frames(), dest=tmp_path / "x")


def test_frames_for_reads_a_run_manifest(tmp_path):
    run = tmp_path / "run"
    (run / "evidence/canonical").mkdir(parents=True)
    (run / "evidence/canonical/frame_0001.png").write_bytes(frames(n=1)[0]["canonical_png"])
    np.save(run / "evidence/canonical/frame_0001_alpha.npy", np.ones((8, 8), bool))
    (run / "manifest.json").write_text(json.dumps({"frames": [{"frame_id": "frame_0001", "canonical": "evidence/canonical/frame_0001.png",
                                                               "alpha": "evidence/canonical/frame_0001_alpha.npy"}]}))
    out = geometry_mvs.frames_for(run)
    assert [f["frame_id"] for f in out] == ["frame_0001"] and out[0]["alpha"].dtype == bool and out[0]["canonical_png"][:4] == b"\x89PNG"


# ---------------------------------------------------------------- local / modal backends: the plumbing, with fakes
FAKE_RUN_STAGE = '''
import importlib.util, sys
CALLS = []
def use_stub(): CALLS.append("use_stub")
def pin_torch_hub(): CALLS.append("pin_torch_hub")
def link_weights(cache): CALLS.append(("link_weights", str(cache))); return {}
def load_app(path):
    CALLS.append(("load_app", path.name))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec); sys.modules[path.stem] = module; spec.loader.exec_module(module)
    return module
'''
FAKE_SAM3D_RESEARCH = '''
import numpy as np
LOADS = []
class _Method:
    def __init__(self, f): self.f = f
    def remote(self, *a, **k): return self.f(*a, **k)
class SAM3DObjects:
    def __init__(self): LOADS.append(1)
    @property
    def run(self): return _Method(self._run)
    def _run(self, rgb, mask, pointmap, seed):
        assert rgb.dtype == np.uint8 and mask.dtype == bool and pointmap.dtype == np.float32
        return {"vertices": np.zeros((3, 3), np.float32), "faces": np.zeros((1, 3), np.uint32), "colors": np.zeros((3, 3), np.uint8),
                "objectToCamera": np.eye(4) * seed, "pins": {"model": "facebook/sam-3d-objects", "modelRevision": "m", "codeRevision": "c"},
                "gpu": "FAKE-GPU", "seconds": 1.5}
'''
FAKE_FAIR = '''
import contextlib, os
from pathlib import Path
SCR = Path(os.environ["SWAP_SCRATCH"]); X0, XW = 63, 392
CALLS = []
def frames_for(cell): raise AssertionError("the provider must point frames_for at the request")
def main(stage, cells="090,030", only=""):
    fr = frames_for(cells); CALLS.append((stage, cells, sorted(fr["padded"])))
    for c in cells.split(","):
        d = SCR / f"checks/da3fair-geom/{c}-da3-base-padded/geometry"; (d / "frames").mkdir(parents=True); (d / "candidate_manifest.json").write_text("{}")
class App:
    def __init__(self): self.registered_entrypoints = {"main": main}; self.runs = 0
    def run(self): self.runs += 1; return contextlib.nullcontext(self)
app = App()
'''
FAKE_GEOMETRY_CLEAN_AB = '''
import contextlib, io
from pathlib import Path
import fair_ab_modal as fam
from PIL import Image
SCR = fam.SCR; GPUD, GEOM = SCR / "checks/clean-gpu", SCR / "checks/clean-geom"
INIT = {"da3-base": lambda c: SCR / f"checks/da3fair-geom/{c}-da3-base-padded/geometry"}
CALLS = []
def main(stage, cells="090,030", bases="moge,da3-base,mapanything,pi3x", only="", vggt=False, ba="numpy"):
    fr = fam.frames_for(cells); CALLS.append((stage, cells, bases, only, vggt))
    for c in cells.split(","):
        if stage == "infer":
            (GPUD / c).mkdir(parents=True); (GPUD / c / "roma-0-1.npz").write_bytes(b"roma")
            for name, data in fr["unpadded"].items():
                (GPUD / c / f"moge-{Path(name).stem}.npz").write_bytes(repr(Image.open(io.BytesIO(data)).size).encode())
        if stage == "refine":
            assert INIT["da3-base"](c).exists()
            for name in only.split(","):
                d = GEOM / name / "geometry"; (d / "frames/frame_0001").mkdir(parents=True); (d / "candidate_manifest.json").write_text("{}")
        if stage == "dense-infer":
            (GPUD / c / "dense-0-1.npz").write_bytes(b"dense")
class App:
    def __init__(self): self.registered_entrypoints = {"main": main}; self.runs = 0
    def run(self): self.runs += 1; return contextlib.nullcontext(self)
app = App()
'''


@pytest.fixture
def fake_kit(tmp_path, monkeypatch):
    """A fake ehs-spatial checkout (run_stage.py, modal_apps) and fake swap notes; sys.modules / sys.path restored after."""
    kit, notes = tmp_path / "kit", tmp_path / "notes"
    (kit / "scripts/onprem").mkdir(parents=True)
    (kit / "modal_apps").mkdir()
    (notes / "geometry-licence-ab-fair-2026-10-05").mkdir(parents=True)
    (kit / "scripts/onprem/run_stage.py").write_text(FAKE_RUN_STAGE)
    (kit / "modal_apps/sam3d_research.py").write_text(FAKE_SAM3D_RESEARCH)
    (kit / "modal_apps/geometry_clean_ab.py").write_text(FAKE_GEOMETRY_CLEAN_AB)
    (notes / "geometry-licence-ab-fair-2026-10-05/fair_ab_modal.py").write_text(FAKE_FAIR)
    monkeypatch.setenv("PANOPTES_WORKCELL", str(kit))
    monkeypatch.setenv("SWAP_NOTES", str(notes))
    monkeypatch.setenv("WEIGHTS", str(tmp_path / "weights"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = dict(sys.modules)   # the fakes may REPLACE real modules another test imported (run_stage, sam3d_research, ...): restore those too
    sam3d._runner.cache_clear()
    yield kit
    fakes = ('run_stage', 'sam3d_research', 'geometry_clean_ab', 'fair_ab_modal', 'completion_ab', 'modal')
    for name in [k for k in sys.modules if k not in before and (k in fakes or k.startswith('modal.'))]:
        sys.modules.pop(name, None)  # only the fakes: third-party modules first imported here stay (C extensions load once per process)
    sys.modules.update(before)   # every module object exactly as it was (the fakes replace run_stage, sam3d_research, modal, ...)
    sam3d._runner.cache_clear()


def test_sam3d_local_backend_runs_the_class_in_process(monkeypatch, fake_kit):
    monkeypatch.setenv("SAM3D_BACKEND", "local")
    out = sam3d.generate(*inputs(seed=3))
    assert out["object_to_camera_p3d"][2, 2] == 3 and out["gpu"] == "FAKE-GPU" and out["seconds"] == 1.5
    assert out["model_info"] == {"model_id": "facebook/sam-3d-objects", "model_revision": "m", "code_revision": "c", "backend": "local"}
    assert set(out) == {"vertices", "faces", "colors", "object_to_camera_p3d", "pins", "seconds", "gpu", "model_info"}
    run_stage = sys.modules["run_stage"]
    assert run_stage.CALLS == ["use_stub", "pin_torch_hub", ("link_weights", str(fake_kit.parent / "weights")), ("load_app", "sam3d_research.py")]
    sam3d.generate(*inputs(seed=4))
    assert sys.modules["sam3d_research"].LOADS == [1]  # the weights load once per process


def test_sam3d_modal_backend_uses_the_research_module(monkeypatch, fake_kit):
    monkeypatch.setenv("SAM3D_BACKEND", "modal")
    out = sam3d.generate(*inputs(seed=5))
    assert out["object_to_camera_p3d"][0, 0] == 5 and out["model_info"]["backend"] == "modal"
    assert "run_stage" not in sys.modules


@pytest.mark.parametrize("backend", ["local", "modal"])
def test_geometry_local_and_modal_backends_run_the_stage_bodies(monkeypatch, fake_kit, tmp_path, backend):
    monkeypatch.setenv("GEOMETRY_MVS_BACKEND", backend)
    dest = tmp_path / f"scratch-{backend}"
    out = geometry_mvs.run("090", frames(size=518), dest=dest)
    assert out == dest and all(p.is_dir() for p in geometry_mvs.outputs("090", dest))
    gc, fam = sys.modules["geometry_clean_ab"], sys.modules["fair_ab_modal"]
    assert fam.CALLS == [("infer", "090", ["frame_0001.png", "frame_0002.png"])]  # the DA3-BASE start was missing
    assert gc.CALLS == [("infer", "090", "moge,da3-base,mapanything,pi3x", "", False),
                        ("refine", "090", "da3-base", "090-da3-base-ba-f", False),
                        ("dense-infer", "090", "moge,da3-base,mapanything,pi3x", "", False)]
    assert (dest / "checks/clean-gpu/090/moge-frame_0001.npz").read_bytes() == b"(392, 518)"  # the content crop of the request's frame
    assert gc.app.runs == 1 and fam.app.runs == 1
    assert ("run_stage" in sys.modules) == (backend == "local")
    with pytest.raises(Exception, match="function bodies only know"):
        geometry_mvs.run("090", frames(size=518), {"roma": "indoor"}, dest=tmp_path / "other")
