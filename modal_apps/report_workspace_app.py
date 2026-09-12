"""CPU report workspace with durable, serialized uploads and persistent artifacts."""
import json
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
PUBLIC = Path(os.environ.get("PANOPTES_PUBLIC_BUNDLE", "/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages"))
RESEARCH = Path(os.environ.get("PANOPTES_RESEARCH_BUNDLE", "/Users/adam/Desktop/panoptes-public/panoptes-serving"))
APP_NAME = "panoptes-report-workspace"
app = modal.App(APP_NAME)
reports = modal.Volume.from_name("panoptes-report-workspace-data", create_if_missing=True)


def public_ignore(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts) or path.suffix in {".py", ".cjs", ".md", ".pyc"}


def deployment_secret():
    if not modal.is_local():
        return modal.Secret.from_name("panoptes-report-workspace")
    # Only required credentials cross the boundary; never mount .env or the repo root.
    values = {}
    for line in (REPO / ".env").read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"GEMINI_API_KEY", "FAL_KEY"}:
            values[key] = value.strip().strip("\"'")
    access = REPO / ".workspace-access.json"
    if access.stat().st_mode & 0o077:
        raise ValueError("Workspace access file must have mode 600")
    values["PANOPTES_WRITE_TOKEN"] = json.loads(access.read_text())["token"]
    assert values.get("GEMINI_API_KEY") and values.get('FAL_KEY') and len(values["PANOPTES_WRITE_TOKEN"]) >= 32
    return modal.Secret.from_dict(values)


if modal.is_local():
    image = (modal.Image.debian_slim(python_version="3.12")
        .apt_install("libgl1", "libgomp1")
        .pip_install_from_pyproject(str(REPO / "pyproject.toml"))
        .pip_install("opencv-python-headless==5.0.0.93", "trimesh==5.1.0")
        .env({"PYTHONPATH": "/app", "PANOPTES_NO_RESUME": "1", "PANOPTES_PUBLIC_RUNS": "user-bor1-02,bor1-components-20260909",
              "PANOPTES_PUBLISHED_ROOT": "/published", "PANOPTES_SITE_ORIGIN": "https://admin-wekruit.github.io", "SAM3_BACKEND": "fal",
              "GEOMETRY_BACKEND": "modal", "MOGE_BACKEND": "modal", "GRADIO_ANALYTICS_ENABLED": "False",
              "PANOPTES_RESEARCH_ROOT":"/research", "LUCIDA_DEPLOYED_APP":"lucida-private-assets",
              "PANOPTES_RECGEN_ENVIRONMENT":"/research/recgen-environment.json"})
        .workdir("/app")
        .add_local_dir(REPO / "ehs_spatial", "/app/ehs_spatial", ignore=["**/__pycache__/**", "**/*.pyc"])
        .add_local_dir(REPO / "scripts", "/app/scripts", ignore=["**/__pycache__/**", "**/*.pyc"])
        .add_local_dir(REPO / "outputs/policies/compiled", "/app/outputs/policies/compiled")
        .add_local_dir(PUBLIC, "/published", ignore=public_ignore)
        .add_local_file(PUBLIC / "pack-model.py", "/published/pack-model.py")
        .add_local_file(RESEARCH / "modal_apps/lucida_assets.py", "/research/modal_apps/lucida_assets.py")
        .add_local_file(RESEARCH / "outputs/candidate-evaluation/lucida-replica-01/generation/environment-recgen/output.json", "/research/recgen-environment.json"))
    for name in ("export_object_evidence.py", "generate_lucida_assets.py", "prepare_product_floor.py",
                 "assemble_lucida_scene.py", "render_lucida_comparisons.py", "lucida_viewer.html"):
        image = image.add_local_file(RESEARCH / "scripts/research" / name, "/research/scripts/research/" + name)
else:
    image = modal.Image.debian_slim(python_version="3.12")
secret = deployment_secret()


@app.function(image=image, volumes={"/app/runs": reports}, secrets=[secret],
              cpu=2, memory=16384, timeout=3600, max_containers=1, retries=0)
def process_upload(capture_data: dict, resume_geometry: dict | None = None):
    """The invocation lives independently of the HTTP container lifecycle."""
    from ehs_spatial.contracts import CaptureRun
    from ehs_spatial.pipeline import EHSAssessmentPipeline
    from ehs_spatial.report_workspace import run_upload
    reports.reload()
    capture = CaptureRun.model_validate(capture_data)
    staging = (Path("runs") / capture.run_id / "upload").resolve()
    if not all(Path(p).resolve().is_relative_to(staging) for p in capture.image_paths):
        raise ValueError("Upload source is outside this run's staging directory")
    try:
        service = EHSAssessmentPipeline()
        if resume_geometry is not None:
            import hashlib
            from ehs_spatial.providers.map_anything import MapAnythingAdapter, saved_geometry_runner
            run = Path('runs') / capture.run_id
            expected = resume_geometry['input_sha256']
            if len(expected) != len(capture.image_paths):
                raise ValueError('Resume source count changed')
            for i, (source, sha) in enumerate(zip(capture.image_paths, expected), 1):
                original = run / 'input' / f'image_{i:02d}{Path(source).suffix.lower()}'
                if any(hashlib.sha256(p.read_bytes()).hexdigest() != sha for p in [Path(source), original]):
                    raise ValueError('Resume staging/input SHA256 mismatch')
            service.map_anything = MapAnythingAdapter(runner=saved_geometry_runner(run / 'geometry', **resume_geometry))
        run_upload(service, capture)
        return json.loads((Path("runs") / capture.run_id / "workspace.json").read_text())
    finally:
        reports.commit()


@app.function(image=image, volumes={"/app/runs": reports}, secrets=[secret],
              cpu=4, memory=16384, timeout=2400, max_containers=1, retries=0)
def process_generation(run_id: str, candidate_id: str):
    from ehs_spatial.report_generation import run_generation
    from ehs_spatial.path_safety import validate_safe_path_segment
    for value in (run_id, candidate_id):
        validate_safe_path_segment(value, 'id')
        if value.startswith('.'):
            raise ValueError('Invalid source ID')
    reports.reload()
    try:
        return run_generation(Path('/app/runs'), run_id, candidate_id)
    finally:
        reports.commit()


class DurableUploads:
    def submit(self, fn, service, capture, candidate_id=None):
        from ehs_spatial.report_workspace import run_upload, write_json
        from ehs_spatial.report_generation import run_generation, job_root
        if fn is run_generation:
            if Path(service).resolve() != Path('/app/runs').resolve():
                raise ValueError('Unexpected generation store')
            reports.commit()
            call = process_generation.spawn(capture, candidate_id)
            write_json(job_root(Path(service), capture, candidate_id) / 'job.json', {'modal_call_id':call.object_id})
            reports.commit()
            return call
        if fn is not run_upload or service.store.root.resolve() != Path("/app/runs").resolve():
            raise ValueError("Only report upload jobs may use this queue")
        reports.commit()
        call = process_upload.spawn(capture.model_dump(mode="json"))
        write_json(Path("runs") / capture.run_id / "job.json", {"modal_call_id": call.object_id})
        reports.commit()
        return call


def reconcile_finished_jobs():
    """Persist terminal Modal errors; never retry a potentially billable model call."""
    from ehs_spatial.report_workspace import write_json
    for path in Path("runs").glob("*/job.json"):
        state_path = path.with_name("workspace.json")
        state = json.loads(state_path.read_text())
        if state.get("state") not in {"queued", "analyzing"}:
            continue
        call = modal.FunctionCall.from_id(json.loads(path.read_text())["modal_call_id"])
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            continue
        except (modal.exception.ConnectionError, modal.exception.InternalError, modal.exception.ServiceError):
            # A failed status lookup is not evidence that the model job failed.
            continue
        except Exception as error:
            state.update(state="failed", error=type(error).__name__)
            write_json(state_path, state)
        else:
            if isinstance(result, dict) and result.get("state") in {"done", "failed"}:
                write_json(state_path, result)
    for path in Path('runs/.generations').glob('*/*/job.json'):
        state_path = path.with_name('status.json')
        state = json.loads(state_path.read_text())
        if state.get('state') not in {'queued','running'}:
            continue
        try:
            result = modal.FunctionCall.from_id(json.loads(path.read_text())['modal_call_id']).get(timeout=0)
        except (TimeoutError, modal.exception.ConnectionError, modal.exception.InternalError, modal.exception.ServiceError):
            continue
        except Exception as error:
            write_json(state_path, dict(state, state='failed', error=type(error).__name__))
        else:
            if isinstance(result, dict) and result.get('state') in {'done','failed'}:
                write_json(state_path, result)


@app.function(image=image, volumes={"/app/runs": reports}, secrets=[secret],
              cpu=2, memory=8192, timeout=1200, max_containers=1)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def web():
    import asyncio
    from ehs_spatial import report_workspace
    report_workspace.JOBS.shutdown(wait=False)
    report_workspace.JOBS = DurableUploads()
    from ehs_spatial.serve import app as server
    if not os.environ.get("PANOPTES_WRITE_TOKEN") or not getattr(server.state, "workspace_write_auth", False):
        raise RuntimeError("Cloud workspace requires the product write-auth middleware")

    volume_lock = asyncio.Lock()

    async def volume_call(operation):
        # Cancellation must not release the lock while its synchronous volume
        # operation is still running in a worker thread.
        task = asyncio.create_task(asyncio.to_thread(operation))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        result = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def persistent_app(scope, receive, send):
        path = scope.get("path", "")
        if (scope["type"] != "http" or path in {"/api/session", "/published", "/workspace-assets"}
                or path.startswith(("/published/", "/workspace-assets/"))):
            return await server(scope, receive, send)
        # ponytail: volume throughput stays serialized, including complete streamed
        # responses. Stateless session/static requests can run concurrently; move
        # report writes to coordinated jobs before allowing parallel volume access.
        async with volume_lock:
            await volume_call(reports.reload)
            try:
                if path.startswith("/api/reports"):
                    await volume_call(reconcile_finished_jobs)
                await server(scope, receive, send)
            finally:
                await volume_call(reports.commit)
    return persistent_app


@app.local_entrypoint()
def seed(registry_only: bool = False, run_id: str = 'user-bor1-02'):
    """Seed only the authorized run, without hidden files, credentials or chat cursors."""
    import io
    from ehs_spatial.object_evidence import build_object_evidence
    from ehs_spatial.path_safety import validate_safe_path_segment
    validate_safe_path_segment(run_id, 'run_id')
    if run_id not in {'user-bor1-02', 'bor1-components-20260909'}:
        raise ValueError('Select an authorized published report')
    source = REPO / 'runs' / run_id
    registry = build_object_evidence(source)
    files = [p for p in source.rglob("*") if p.is_file() and not p.is_symlink()
             and not any(part.startswith(".") for part in p.relative_to(source).parts)
             and p.name not in {"chat.jsonl", "job.json", "workspace.json", "object-evidence.json"}]
    assert files and all(p.resolve().is_relative_to(source.resolve()) for p in files)
    with reports.batch_upload() as batch:
        if not registry_only:
            for path in files:
                batch.put_file(path, '/' + run_id + '/' + path.relative_to(source).as_posix())
        batch.put_file(io.BytesIO(json.dumps(registry, ensure_ascii=False, allow_nan=False).encode()),
                       '/' + run_id + '/object-evidence.json')
    print(json.dumps({"seed_run": source.name, "files": 0 if registry_only else len(files),
                      "bytes": 0 if registry_only else sum(p.stat().st_size for p in files),
                      "registry_candidates": len(registry["candidates"]),
                      "excluded": ["other runs", "chat cursors", "hidden files", "credentials"]}))


if __name__ == "__main__":
    # Local-only queue contract check; no remote function or provider is invoked.
    from types import SimpleNamespace
    from unittest.mock import patch
    from ehs_spatial.report_workspace import run_upload
    steps = []
    fake_reports = SimpleNamespace(commit=lambda: steps.append("commit"))
    fake_worker = SimpleNamespace(spawn=lambda value: (steps.append("spawn") or SimpleNamespace(object_id="test-call")))
    capture = SimpleNamespace(run_id="test-run", model_dump=lambda **kw: {"run_id": "test-run"})
    service = SimpleNamespace(store=SimpleNamespace(root=Path("/app/runs")))
    with patch.dict(globals(), reports=fake_reports, process_upload=fake_worker), \
         patch("ehs_spatial.report_workspace.write_json", side_effect=lambda p, v: steps.append("write-id")):
        assert DurableUploads().submit(run_upload, service, capture).object_id == "test-call"
    assert steps == ["commit", "spawn", "write-id", "commit"]
    assert public_ignore(Path(".git/config")) and public_ignore(Path(".env"))
    assert not public_ignore(Path("model/objects/robot.bin.gz"))
    print("PASS: durable queue ordering and public bundle exclusions; zero remote calls")
