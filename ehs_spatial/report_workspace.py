"""Report library and upload lifecycle, using the existing assessment pipeline."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import time
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .artifacts import ArtifactStore, _read_json_dict
from .contracts import CaptureRun
from .path_safety import validate_safe_path_segment

STATIC = Path(__file__).with_name("static")
# ponytail: one provider job at a time in this process. A multi-worker deployment
# must replace this lane with a durable shared queue before raising concurrency.
JOBS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="report-analysis")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        temp = Path(f.name)
    temp.replace(path)


def report_summary(root: Path, row: dict) -> dict:
    run = root / row["run_id"]
    workspace = _read_json_dict(run / "workspace.json")
    phase = run / "deep_report.status"
    state = ("failed" if workspace.get("state") == "failed" else
             phase.read_text().strip() if phase.is_file() else workspace.get("state", "saved"))
    return {**row, "title": workspace.get("title") or row["run_id"],
            "created_at": row.get("created_at") or workspace.get("created_at"),
            "phase": state, "error": workspace.get("error"),
            "report_url": f"/reports/{row['run_id']}",
            "has_report": (run / "assessment.json").is_file(),
            "image_count": len(list((run / "input").glob("image_*")))}


def run_upload(service, capture: CaptureRun) -> None:
    started = time.monotonic()
    timing = {}
    run = service.store.paths(capture.run_id).root
    path = run / "workspace.json"
    state = _read_json_dict(path)
    state["state"] = "analyzing"
    write_json(path, state)
    try:
        stage_started = time.monotonic()
        service.run_assessment(capture)
        timing['assessmentSeconds'] = time.monotonic() - stage_started
        from .app import _run_deep_report_chain
        stage_started = time.monotonic()
        _run_deep_report_chain(capture.run_id)
        timing['reportSeconds'] = time.monotonic() - stage_started
        phase = (run / "deep_report.status").read_text().strip()
        if phase != "done":
            raise RuntimeError("The report pipeline did not complete")
        state["state"] = "done"
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception("Report %s failed", capture.run_id)
        state.update(state="failed", error=type(exc).__name__)
    finally:
        state['timing'] = {**timing, 'totalSeconds': time.monotonic() - started}
        write_json(path, state)
        # Staged upload bytes are copied into the immutable input by prepare_run.
        if state['state'] == 'done':
            for source in capture.image_paths:
                Path(source).unlink(missing_ok=True)
            if capture.image_paths:
                Path(capture.image_paths[0]).parent.rmdir()


def register_workspace(api, service, *, published_root: Path | None = None) -> None:
    root = service.store.root

    def resolve(run_id: str) -> Path:
        try:
            validate_safe_path_segment(run_id, "run_id")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        run = root / run_id
        if run_id.startswith('.') or not run.is_dir() or not run.resolve().is_relative_to(root.resolve()):
            raise HTTPException(404, "Report not found")
        return run

    api.mount("/workspace-assets", StaticFiles(directory=STATIC), name="workspace-assets")
    if published_root and published_root.is_dir():
        @api.get("/published/site-config.js")
        def viewer_site_config():
            return FileResponse(STATIC / "site-config.js", media_type="application/javascript", headers={"Cache-Control": "no-cache"})

        api.mount("/published", StaticFiles(directory=published_root, html=True), name="published-report")
    from .report_generation import register_generation
    register_generation(api, service)

    @api.get("/reports")
    @api.get("/reports/{run_id}")
    def workspace_page(run_id: str | None = None):
        if run_id:
            resolve(run_id)
        return FileResponse(STATIC / "workspace.html", headers={"Cache-Control": "no-cache"})

    @api.get("/api/reports")
    def reports(request: Request):
        rows = [report_summary(root, row) for row in ArtifactStore(root).list_runs()]
        if not getattr(request.state, "can_write", True):
            rows = [row for row in rows if row["run_id"] in request.state.public_runs]
        rows.sort(key=lambda row: (row.get("created_at") or "", row["run_id"]), reverse=True)
        return {"reports": rows, "mode": "live"}

    @api.get("/api/reports/{run_id}")
    def report_detail(run_id: str):
        run = resolve(run_id)
        from .object_evidence import write_object_evidence
        # Every completed pipeline/correction writes this snapshot. Reading a
        # report must not re-hash hundreds of MB of provider data on each visit.
        evidence = _read_json_dict(run / "object-evidence.json")
        if not evidence:
            evidence = _read_json_dict(write_object_evidence(run))
        from .observed_scene import spatial_state
        spatial, spatial_candidates = spatial_state(run, evidence)
        from .report_generation import generation_state, job_root
        for candidate in evidence.get('candidates', []):
            candidate['spatial'] = spatial_candidates[candidate['id']]
            job = generation_state(run_id, candidate['id'], job_root(root,run_id,candidate['id']), evidence.get('source_sha256'))
            retry_with_new_evidence = job['state'] == 'failed' and job['source_revision'] != job['current_source_revision']
            if job['state'] != 'not_started' and not retry_with_new_evidence:
                candidate['generation'] = {**job, 'status':job['state'], 'reason':job['error']}
        for frame in evidence.get("frames", []):
            frame["url"] = f"/api/reports/{run_id}/images/{frame['frame_id']}"
        # The installed public snapshot declares its own inspection run. No
        # labels or coordinates are used to guess attachment correspondence.
        playgrounds = []
        if spatial['viewer_url']:
            stale = spatial['status'] == 'stale'
            playgrounds.append({'id':'observed', 'url':spatial['viewer_url'], 'revision':spatial['revision'],
                'title':{'zh':'上一版观测空间' if stale else '观测空间',
                         'en':'Previous observed scene' if stale else 'Observed scene'}})
        if published_root:
            data = _read_json_dict(published_root / "unified-data.json")
            source = data.get("source_run_id")
            if source == run_id:
                playgrounds.append({"id": "combined", "title": {"zh": "重建实验场", "en": "Reconstruction playgrounds"}, "url": "/published/"})
        artifact = _read_json_dict(run / "playgrounds.json")
        for item in artifact.get("items", []):
            url = item.get("url", "")
            if url.startswith(f"/reports/{run_id}/") and not url.startswith("//"):
                playgrounds.append(item)
        for candidate in evidence.get('candidates', []):
            generation = candidate['generation']
            if generation.get('viewer_url') and generation['status'] in {'done','stale'}:
                stale = generation['status'] == 'stale'
                label = candidate.get('label', candidate['id'])
                playgrounds.append({'id':candidate['id'], 'url':generation['viewer_url'],
                    'title':{'zh':label+(' · 上一版生成物体' if stale else ' · 生成物体'),
                             'en':label+(' · Previous generated object' if stale else ' · Generated object')}})
        row = next((r for r in ArtifactStore(root).list_runs() if r["run_id"] == run_id), {"run_id": run_id})
        evidence_bytes = json.dumps(evidence, sort_keys=True).encode()
        return {**report_summary(root, row), "evidence": evidence, "playgrounds": playgrounds, "spatial":spatial,
                "revision": hashlib.sha256(evidence_bytes).hexdigest()[:16],
                "inspection_url": f"/report/{run_id}?embedded=1", "download_url": f"/report/{run_id}/file"}

    @api.get('/api/reports/{run_id}/observed/revisions/{revision}/assets/{asset:path}')
    def observed_asset(run_id: str, revision: str, asset: str):
        run = resolve(run_id)
        if not re.fullmatch(r'[0-9a-f]{24}', revision):
            raise HTTPException(404, 'Observed scene revision not found')
        directory = (run / 'observed' / revision).resolve()
        if not directory.is_relative_to(run.resolve()):
            raise HTTPException(404, 'Observed scene revision not found')
        manifest = _read_json_dict(directory / 'manifest.json')
        path = (directory / asset).resolve()
        if (manifest.get('revision') != revision or asset not in manifest.get('artifacts', {})
                or not path.is_relative_to(directory) or not path.is_file()
                or path.suffix.lower() not in {'.json','.glb','.gz','.bin','.png','.jpg','.jpeg','.webp'}):
            raise HTTPException(404, 'Observed scene asset not found')
        return FileResponse(path)

    @api.get("/api/reports/{run_id}/images/{frame_id}")
    def report_image(run_id: str, frame_id: str):
        run = resolve(run_id)
        match = re.fullmatch(r"frame_(\d{4})", frame_id)
        images = sorted((run / "input").glob("image_*"))
        if not match or not 1 <= int(match[1]) <= len(images):
            raise HTTPException(404, "Frame not found")
        path = images[int(match[1]) - 1]
        if not path.is_file() or not path.resolve().is_relative_to(run.resolve()):
            raise HTTPException(404, "Source image unavailable")
        return FileResponse(path)

    @api.post("/api/reports", status_code=202)
    async def upload_report(request: Request):
        form = await request.form(max_files=4, max_fields=4, max_part_size=32 * 1024 * 1024)
        files = form.getlist("images")
        if not 1 <= len(files) <= 4:
            raise HTTPException(400, "Upload one to four photos")
        title = str(form.get("title") or "").strip()[:160]
        height_text = str(form.get("camera_height_m") or "").strip()
        try:
            camera_height = float(height_text) if height_text else None
            if camera_height is not None and not 0.1 <= camera_height <= 10:
                raise ValueError()
        except ValueError as exc:
            raise HTTPException(400, "Camera height must be between 0.1 and 10 metres") from exc
        contents = []
        for upload in files:
            if not hasattr(upload, "read"):
                raise HTTPException(400, "Invalid image upload")
            data = await upload.read(32 * 1024 * 1024 + 1)
            await upload.close()
            if len(data) > 32 * 1024 * 1024:
                raise HTTPException(413, "Each photo must be under 32 MB")
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                    if image.format not in {"JPEG", "PNG", "WEBP"}:
                        raise ValueError()
                    suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[image.format]
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
                raise HTTPException(400, "Upload a valid JPEG, PNG or WebP photo") from exc
            contents.append((data, suffix))
        run_id = uuid4().hex
        run = root / run_id
        run.mkdir(parents=True)
        if camera_height is not None:
            write_json(run / 'calibration.json', {'camera_height_provided': True, 'camera_height_m': camera_height})
        staging = run / "upload"
        staging.mkdir()
        paths = []
        for index, (data, suffix) in enumerate(contents, 1):
            path = staging / f"{index:02d}{suffix}"
            path.write_bytes(data)
            paths.append(str(path))
        write_json(run / "workspace.json", {"title": title or run_id, "state": "queued",
                   "created_at": datetime.now(timezone.utc).isoformat()})
        capture = CaptureRun(run_id=run_id, image_paths=paths, camera_height_m=camera_height)
        try:
            JOBS.submit(run_upload, service, capture)
        except Exception as error:
            state = _read_json_dict(run / 'workspace.json')
            write_json(run / 'workspace.json', dict(state, state='failed', error=type(error).__name__))
            raise HTTPException(503, f'Report {run_id} was saved, but could not start. It remains in report history.') from error
        return JSONResponse({"run_id": run_id, "url": f"/reports/{run_id}", "phase": "queued"}, status_code=202)
