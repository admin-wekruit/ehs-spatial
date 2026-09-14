"""Injected FastAPI application for the new product; no historical-report writes."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import io
from typing import Annotated, Literal
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
import psycopg

from .contracts import (AgentRequest, CreateBranch, CreateProject, EditRequest, ForkRequest,
                        JobRequest, PlatformError, PublicationRequest, capability_sha,
                        AgentTurn, Asset, AssetRecord, Branch, Capture, CaptureCreated,
                        Commit, EditBatch, ErrorResponse, Health, Items, Job, Project,
                        ProjectDetail, Publication, PublicationSummary, Revision)
from .executor import dispatch_pending

MAX_UPLOAD_BYTES = 32 * 1024 * 1024


def _capability(authorization):
    if not authorization or not authorization.startswith("Capability "):
        raise PlatformError("capability_required", 401)
    token = authorization[len("Capability "):]
    capability_sha(token)
    return token


def _public(value):
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if key not in ("attemptToken", "lateResults", "requestSha256", "capabilitySha256", "executorRef")}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


def _image_metadata(data):
    try:
        with Image.open(io.BytesIO(data)) as check:
            check.verify()
        with Image.open(io.BytesIO(data)) as original:
            if getattr(original, "n_frames", 1) != 1:
                raise PlatformError("animated_image_not_supported", 422)
            width, height = original.size
            orientation = original.getexif().get(274, 1)
            if orientation not in range(1, 9):
                orientation = 1
            normalized = ImageOps.exif_transpose(original)
            normalized = normalized.convert("RGBA" if "A" in normalized.getbands() else "RGB")
            output = io.BytesIO()
            normalized.save(output, format="PNG")
            mappings = {1: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 2: [[-1, 0, width-1], [0, 1, 0], [0, 0, 1]],
                        3: [[-1, 0, width-1], [0, -1, height-1], [0, 0, 1]], 4: [[1, 0, 0], [0, -1, height-1], [0, 0, 1]],
                        5: [[0, 1, 0], [1, 0, 0], [0, 0, 1]], 6: [[0, -1, height-1], [1, 0, 0], [0, 0, 1]],
                        7: [[0, -1, height-1], [-1, 0, width-1], [0, 0, 1]], 8: [[0, 1, 0], [-1, 0, width-1], [0, 0, 1]]}
            return output.getvalue(), {"width": normalized.width, "height": normalized.height,
                "originalWidth": width, "originalHeight": height, "exifOrientation": orientation,
                "originalSha256": hashlib.sha256(data).hexdigest(), "originalMediaType": Image.MIME.get(original.format, "application/octet-stream"),
                "pixelMapping": [{"source": "original_pixels", "target": "normalized_pixels", "coordinateConvention": "pixel_centers", "matrix": mappings[orientation]}]}
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise PlatformError("invalid_image", 422) from None


def create_app(*, repository, blobs, executor=None, agent_service=None, policy_service=None):
    repository.blobs = blobs
    @asynccontextmanager
    async def lifespan(app):
        stop = asyncio.Event()

        async def outbox():
            while not stop.is_set():
                await asyncio.to_thread(repository.recover_expired_jobs)
                if executor:
                    await asyncio.to_thread(dispatch_pending, repository, executor)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=2)
                except TimeoutError:
                    pass

        task = asyncio.create_task(outbox()) if executor or agent_service else None
        yield
        stop.set()
        if task:
            await task

    app = FastAPI(title="Panoptes platform", lifespan=lifespan,
                  responses={status: {"model": ErrorResponse} for status in (400, 401, 403, 404, 409, 413, 422, 503)})
    app.state.repository, app.state.blobs, app.state.executor = repository, blobs, executor

    @app.exception_handler(PlatformError)
    async def platform_error(request, exc):
        return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "params": exc.params}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request", "params": {
            "fields": [{"path": list(item["loc"]), "type": item["type"]} for item in exc.errors()]}}})

    @app.exception_handler(psycopg.Error)
    async def database_error(request, exc):
        return JSONResponse(status_code=503, content={"error": {"code": "database_unavailable", "params": {}}})

    @app.get("/api/health", response_model=Health, response_model_exclude_unset=True)
    def health():
        with repository._connect() as connection:
            connection.execute("SELECT 1")
        return {"status": "ok", "schemaVersion": 1}

    @app.get("/api/projects", response_model=Items[Project], response_model_exclude_unset=True)
    def projects():
        return repository.list_projects()

    @app.post("/api/projects", status_code=201, response_model=ProjectDetail, response_model_exclude_unset=True)
    def create_project(body: CreateProject, authorization: Annotated[str | None, Header()] = None):
        return repository.create_project(_capability(authorization), body.model_dump(mode="json"))

    @app.get("/api/projects/{project_id}", response_model=ProjectDetail, response_model_exclude_unset=True)
    def get_project(project_id: UUID):
        return repository.get_project(str(project_id))

    @app.post("/api/projects/{project_id}/branches", status_code=201, response_model=Branch, response_model_exclude_unset=True)
    def branches(project_id: UUID, body: CreateBranch, authorization: Annotated[str | None, Header()] = None):
        return repository.create_branch(str(project_id), _capability(authorization), body.model_dump(mode="json"))

    @app.post("/api/projects/{project_id}/forks", status_code=201, response_model=ProjectDetail, response_model_exclude_unset=True)
    def forks(project_id: UUID, body: ForkRequest, authorization: Annotated[str | None, Header()] = None):
        return repository.fork_project(str(project_id), _capability(authorization), body.model_dump(mode="json"))

    @app.get("/api/revisions/{revision_id}", response_model=Revision, response_model_exclude_unset=True)
    def revision(revision_id: UUID):
        return repository.get_revision(str(revision_id))

    @app.post("/api/projects/{project_id}/edits", status_code=201, response_model=Commit, response_model_exclude_unset=True)
    def edits(project_id: UUID, body: EditRequest, authorization: Annotated[str | None, Header()] = None):
        return repository.commit_edits(str(project_id), _capability(authorization), body.model_dump(mode="json"))

    @app.post("/api/projects/{project_id}/captures", status_code=201, response_model=CaptureCreated, response_model_exclude_unset=True)
    async def captures(project_id: UUID, files: Annotated[list[UploadFile], File()],
                       target: Annotated[Literal["scene", "standalone_object"], Form()],
                       requestId: Annotated[UUID, Form()], branchId: Annotated[UUID, Form()], baseRevisionId: Annotated[UUID, Form()],
                       captureMode: Annotated[Literal["initial", "append"], Form()] = "initial",
                       authorization: Annotated[str | None, Header()] = None):
        capability = _capability(authorization)
        repository.authorize(str(project_id), capability)
        if not 1 <= len(files) <= 4:
            raise PlatformError("image_count_out_of_range", 422, minimum=1, maximum=4)
        images = []
        for upload in files:
            data = await upload.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise PlatformError("upload_too_large", 413, maximumBytes=MAX_UPLOAD_BYTES)
            normalized, metadata = _image_metadata(data)
            original_blob = blobs.put(data, metadata.pop("originalMediaType"))
            original_blob["metadata"] = {"kind": "original_image", "width": metadata["originalWidth"], "height": metadata["originalHeight"]}
            original = repository.register_asset(str(project_id), original_blob)
            image = blobs.put(normalized, "image/png")
            image["metadata"] = {**metadata, "originalAssetId": original["id"], "kind": "source_image"}
            images.append(image)
        return _public(repository.create_capture(str(project_id), capability,
            {"requestId": str(requestId), "branchId": str(branchId), "baseRevisionId": str(baseRevisionId), "target": target, "captureMode": captureMode}, images))

    @app.post("/api/projects/{project_id}/jobs", status_code=201, response_model=Job, response_model_exclude_unset=True)
    def jobs(project_id: UUID, body: JobRequest, tasks: BackgroundTasks, authorization: Annotated[str | None, Header()] = None):
        job = repository.create_job(str(project_id), _capability(authorization), body.model_dump(mode="json"))
        if executor:
            tasks.add_task(dispatch_pending, repository, executor)
        return _public(job)

    @app.get("/api/jobs/{job_id}", response_model=Job, response_model_exclude_unset=True)
    def get_job(job_id: UUID):
        return _public(repository.get_job(str(job_id)))

    @app.post("/api/jobs/{job_id}/cancel", response_model=Job, response_model_exclude_unset=True)
    def cancel_job(job_id: UUID, authorization: Annotated[str | None, Header()] = None):
        return _public(repository.cancel_job(str(job_id), _capability(authorization)))

    @app.get("/api/assets/{asset_id}", response_model=Asset, response_model_exclude_unset=True)
    def asset(asset_id: UUID):
        value = repository.get_asset(str(asset_id))
        return {**value, "url": blobs.url(value["storageKey"], value["id"])}

    @app.get("/api/assets/{asset_id}/content", response_class=Response,
             responses={200: {"description": "Immutable asset bytes; Content-Type is the stored media type.", "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}})
    def content(asset_id: UUID):
        value = repository.get_asset(str(asset_id))
        data = blobs.get(value["storageKey"], value["sha256"], value["sizeBytes"])
        return Response(data, media_type=value["mediaType"], headers={"ETag": '"' + value["sha256"] + '"', "Cache-Control": "public,max-age=31536000,immutable", "X-Content-Type-Options": "nosniff"})

    @app.get("/api/blobs/sha256/{sha256}", response_class=Response,
             responses={200: {"description": "Immutable content-addressed bytes.", "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}})
    def blob_content(sha256: str):
        metadata = blobs.head("sha256/" + sha256)
        if metadata is None:
            raise PlatformError("blob_not_found", 404)
        data = blobs.get(metadata["storageKey"], metadata["sha256"], metadata["sizeBytes"])
        return Response(data, media_type=metadata["mediaType"], headers={"ETag": '"' + metadata["sha256"] + '"', "Cache-Control": "public,max-age=31536000,immutable", "X-Content-Type-Options": "nosniff"})

    @app.get("/api/publications", response_model=Items[PublicationSummary], response_model_exclude_unset=True)
    def publications():
        return repository.list_publications()

    @app.get("/api/publications/{publication_id}", response_model=Publication, response_model_exclude_unset=True)
    def publication(publication_id: UUID):
        return repository.get_publication(str(publication_id))

    @app.post("/api/projects/{project_id}/publications", status_code=201, response_model=Publication, response_model_exclude_unset=True)
    def publish(project_id: UUID, body: PublicationRequest, authorization: Annotated[str | None, Header()] = None):
        capability = _capability(authorization)
        repository.authorize(str(project_id), capability)
        evidence = {}
        if policy_service:
            evidence = policy_service.publication_evidence(str(project_id), str(body.sceneRevisionId), [str(x) for x in body.evaluationIds], [str(x) for x in body.reviewIds])
        return repository.create_publication(str(project_id), capability, body.model_dump(mode="json"), **evidence)

    @app.post("/api/projects/{project_id}/agent-turns", status_code=201, response_model=AgentTurn, response_model_exclude_unset=True)
    def agent_turn(project_id: UUID, body: AgentRequest, tasks: BackgroundTasks, authorization: Annotated[str | None, Header()] = None):
        capability = _capability(authorization)
        if agent_service is None:
            raise PlatformError("agent_not_configured", 503)
        turn = repository.create_agent_turn(str(project_id), capability, body.model_dump(mode="json"))
        if turn["status"] == "pending":
            tasks.add_task(agent_service.run_turn, turn, capability)
        return _public(turn)

    @app.get("/api/projects/{project_id}/agent-turns", response_model=Items[AgentTurn], response_model_exclude_unset=True)
    def agent_turns(project_id: UUID, conversationId: Annotated[UUID | None, Query()] = None, afterSequence: Annotated[int, Query(ge=0)] = 0):
        return _public(repository.list_agent_turns(str(project_id), conversation_id=str(conversationId) if conversationId else None, after_sequence=afterSequence))

    def records_endpoint(kind):
        def records(project_id: UUID):
            return _public(repository.list_project_records(str(project_id), kind))
        return records

    for kind, model in (("revisions", Revision), ("captures", Capture), ("assets", AssetRecord), ("jobs", Job), ("edits", EditBatch)):
        app.add_api_route("/api/projects/{project_id}/" + kind, records_endpoint(kind), methods=["GET"], response_model=Items[model], response_model_exclude_unset=True)

    if policy_service:
        policy_service.register_routes(app)
    return app
