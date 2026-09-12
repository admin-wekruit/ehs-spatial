import io
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from ehs_spatial.artifacts import ArtifactStore
from ehs_spatial.report_workspace import register_workspace


def test_report_library_upload_and_source_identity(tmp_path, monkeypatch):
    from ehs_spatial import report_workspace
    dispatched = []
    monkeypatch.setattr(report_workspace.JOBS, "submit", lambda *args: dispatched.append(args))
    service = SimpleNamespace(store=ArtifactStore(tmp_path))
    api = FastAPI()
    register_workspace(api, service)
    client = TestClient(api)
    stream = io.BytesIO()
    Image.new("RGB", (317, 193), "navy").save(stream, "PNG")
    created = client.post("/api/reports", data={"title": "Fixture one"}, files={"images": ("photo.png", stream.getvalue(), "image/png")})
    assert created.status_code == 202
    rid = created.json()["run_id"]
    capture = dispatched[0][2]
    assert capture.camera_height_m is None
    assert not (tmp_path / rid / 'calibration.json').exists()
    assert Path(capture.image_paths[0]).is_file()
    service.store.prepare_run(capture)
    history = client.get("/api/reports").json()["reports"]
    assert history[0]["title"] == "Fixture one" and history[0]["run_id"] == rid
    detail = client.get(f"/api/reports/{rid}").json()
    assert detail["evidence"]["frames"][0]["width"] == 317
    assert detail["evidence"]["frames"][0]["height"] == 193
    assert detail["evidence"]["candidates"] == []
    # Ordinary report reads consume the completed evidence snapshot without
    # loading large provider arrays again.
    from ehs_spatial import object_evidence
    monkeypatch.setattr(object_evidence, "build_object_evidence", lambda _: (_ for _ in ()).throw(AssertionError("unexpected rebuild")))
    assert client.get(f"/api/reports/{rid}").json()["revision"] == detail["revision"]
    image = client.get(detail["evidence"]["frames"][0]["url"])
    assert image.status_code == 200 and Image.open(io.BytesIO(image.content)).size == (317, 193)
    assert client.get(f"/reports/{rid}").status_code == 200
    assert client.get("/reports/missing").status_code == 404
    assert client.get(f"/api/reports/{rid}/images/nonexistent").status_code == 404
    assert client.post("/api/reports", files={"images": ("fake.jpg", b"invalid", "image/jpeg")}).status_code == 400
    assert client.post("/api/reports", data={"camera_height_m": "nan"}, files={"images": ("photo.png", stream.getvalue(), "image/png")}).status_code == 400
    assert len(dispatched) == 1


def test_explicit_upload_height_has_saved_calibration_provenance(tmp_path, monkeypatch):
    from ehs_spatial import report_workspace
    dispatched = []
    monkeypatch.setattr(report_workspace.JOBS, 'submit', lambda *args: dispatched.append(args))
    api = FastAPI()
    register_workspace(api, SimpleNamespace(store=ArtifactStore(tmp_path)))
    stream = io.BytesIO()
    Image.new('RGB', (17, 23), 'gray').save(stream, 'PNG')
    response = TestClient(api).post('/api/reports', data={'camera_height_m':'1.64'},
        files={'images':('photo.png',stream.getvalue(),'image/png')})
    assert response.status_code == 202
    run = tmp_path / response.json()['run_id']
    assert json.loads((run/'calibration.json').read_text()) == {'camera_height_provided':True,'camera_height_m':1.64}
    assert dispatched[0][2].camera_height_m == 1.64


def test_history_keeps_failed_and_distinct_reports(tmp_path):
    for rid, state in [("new", "queued"), ("failed", "failed")]:
        run = tmp_path / rid
        run.mkdir()
        (run / "workspace.json").write_text(json.dumps({"title": "same title", "state": state, "created_at": "2026-09-09T00:00:00Z"}))
    api = FastAPI()
    register_workspace(api, SimpleNamespace(store=ArtifactStore(tmp_path)))
    rows = TestClient(api).get("/api/reports").json()["reports"]
    assert {row["run_id"] for row in rows} == {"new", "failed"}
    assert {row["phase"] for row in rows} == {"queued", "failed"}
    assert all(not row["has_report"] for row in rows)


def test_durable_queue_accepts_a_volume_symlink(tmp_path):
    import ast
    from ehs_spatial.report_workspace import run_upload
    from ehs_spatial.report_generation import run_generation
    target = tmp_path / 'volume'
    target.mkdir()
    mounted = tmp_path / 'runs'
    mounted.symlink_to(target, target_is_directory=True)
    # Execute the actual dispatcher without importing Modal image definitions
    # or invoking providers. The mount models Modal's real symlink behavior.
    source = Path('modal_apps/report_workspace_app.py').read_text()
    definition = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'DurableUploads')
    dispatched = []
    worker = SimpleNamespace(spawn=lambda *args: (dispatched.append(args) or SimpleNamespace(object_id='call-test')))
    env = {'Path':lambda value: mounted if str(value)=='/app/runs' else Path(value),
           'reports':SimpleNamespace(commit=lambda: None), 'process_upload':worker, 'process_generation':worker}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<dispatcher>', 'exec'), env)
    queue = env['DurableUploads']()
    service = SimpleNamespace(store=ArtifactStore(mounted))
    capture = SimpleNamespace(run_id='upload', model_dump=lambda **kwargs:{'run_id':'upload'})
    queue.submit(run_upload, service, capture)
    queue.submit(run_generation, mounted, 'upload', 'lamp')
    assert dispatched == [({'run_id':'upload'},), ('upload','lamp')]


def test_failed_prepare_keeps_original_upload(tmp_path):
    from ehs_spatial.contracts import CaptureRun
    from ehs_spatial.report_workspace import run_upload
    staging = tmp_path/'partial/upload'
    staging.mkdir(parents=True)
    source = staging/'01.jpg'
    source.write_bytes(b'original-upload-not-yet-copied')
    service = SimpleNamespace(store=ArtifactStore(tmp_path), run_assessment=lambda _: (_ for _ in ()).throw(OSError('copy failed')))
    run_upload(service, CaptureRun(run_id='partial', image_paths=[str(source)]))
    assert source.read_bytes() == b'original-upload-not-yet-copied'
    assert json.loads((staging.parent/'workspace.json').read_text())['state'] == 'failed'
