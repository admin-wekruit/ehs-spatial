"""Verified immutable publications and optionally isolated visitor feedback."""
from __future__ import annotations

from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.exceptions import RequestValidationError

from .contracts import PlatformError
from .feedback import FeedbackRequest


FEEDBACK_PATH = re.compile(r"/api/publications/[^/]+/entities/[^/]+/feedback")
IDENTITY_SUGGESTIONS_PATH = re.compile(r"/api/publications/[^/]+/identity-suggestions")


def immutable_route(path: str):
    return path.startswith(("/api/publications/", "/api/revisions/", "/api/assets/"))


def read_catalog(catalog_dir: str | Path):
    """Verify every immutable payload and blob before publishing a serving index."""
    root = Path(catalog_dir).resolve()
    if (root / "bundle.json").exists():
        raise ValueError("Expected a publication catalog, not a single bundle")
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    if not directories:
        raise ValueError("Publication catalog is empty")
    responses, files, bundles, revisions = {}, {}, [], {}
    from .publication_view import publication_view

    def shared_revision(revision):
        # Exports repeat frozen revisions across routes and publications. Share
        # only after comparing the complete payload; these records stay read-only.
        shared = revisions.setdefault(revision["id"], revision)
        if shared != revision:
            raise ValueError(f"Conflicting immutable route payload: /api/revisions/{revision['id']}")
        return shared

    for directory in directories:
        if str(UUID(directory.name)) != directory.name:
            raise ValueError("Catalog directories must be publication IDs")
        bundle = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
        if bundle["schemaVersion"] != 1:
            raise ValueError("Unsupported publication bundle schema")
        saved = bundle["responses"]
        publication = saved["/api/publications/" + bundle["publicationId"]]
        if directory.name != bundle["publicationId"] or publication["id"] != bundle["publicationId"] or publication["projectId"] != bundle["projectId"]:
            raise ValueError("Publication bundle identity mismatch")
        created_at = datetime.fromisoformat(publication["createdAt"])
        if created_at.tzinfo is None:
            raise ValueError("Publication createdAt must contain a timezone")
        revision = publication["snapshot"]["revision"]
        if revision["id"] != publication["sceneRevisionId"] or revision["projectId"] != bundle["projectId"] or saved["/api/revisions/" + revision["id"]] != revision:
            raise ValueError("Publication revision mismatch")
        for path, value in saved.items():
            if path.startswith("/api/revisions/"):
                saved[path] = shared_revision(value)
        for key in ("revision", "reconstructionRevision"):
            if publication["snapshot"].get(key) is not None:
                publication["snapshot"][key] = shared_revision(publication["snapshot"][key])
        detail = saved["/api/projects/" + bundle["projectId"]]
        if detail.get("revision") is not None:
            detail["revision"] = shared_revision(detail["revision"])
        history = saved["/api/projects/" + bundle["projectId"] + "/revisions"]
        history["items"] = [shared_revision(item) for item in history["items"]]
        summaries = saved["/api/publications"]["items"]
        if len(summaries) != 1 or any(summaries[0][key] != publication[key] for key in ("id", "projectId", "sceneRevisionId", "createdAt")):
            raise ValueError("Publication summary mismatch")
        manifest_ids, checked = set(), set()
        for entry in publication["snapshot"]["assetManifest"]:
            asset_id, sha = entry["assetId"], entry["sha256"]
            if asset_id in manifest_ids or not re.fullmatch(r"[0-9a-f]{64}", sha) or type(entry["sizeBytes"]) is not int or entry["sizeBytes"] < 0:
                raise ValueError("Invalid publication asset manifest")
            manifest_ids.add(asset_id)
            record = saved["/api/assets/" + asset_id]
            if record["id"] != asset_id or record["projectId"] != bundle["projectId"] or record["url"] != f"/api/assets/{asset_id}/content" or any(record[key] != entry[key] for key in ("sha256", "sizeBytes", "mediaType")):
                raise ValueError("Publication asset metadata mismatch")
            path = directory / "blobs" / sha
            if path.stat().st_size != entry["sizeBytes"]:
                raise ValueError(f"Publication asset size mismatch: {asset_id}")
            if sha not in checked:
                with path.open("rb") as source:
                    if hashlib.file_digest(source, "sha256").hexdigest() != sha:
                        raise ValueError(f"Publication asset hash mismatch: {asset_id}")
                checked.add(sha)
            files.setdefault(asset_id, (path, record))
        if {path.removeprefix("/api/assets/") for path in saved if path.startswith("/api/assets/")} != manifest_ids:
            raise ValueError("Publication asset routes differ from its manifest")
        for path, value in saved.items():
            if immutable_route(path):
                if path in responses and responses[path] != value:
                    raise ValueError(f"Conflicting immutable route payload: {path}")
                responses[path] = value
        responses[f"/api/publications/{publication['id']}/view"] = publication_view(publication, detail)
        for edit in publication["snapshot"].get("editBatches") or []:
            responses[f"/api/publications/{publication['id']}/edits/{edit['id']}"] = edit
        bundles.append((created_at, publication["id"], bundle))

    # Choose a whole project projection by publication time, never filesystem
    # order or a mix of metadata from different releases. UUID breaks exact ties.
    bundles.sort(key=lambda item: item[:2], reverse=True)
    latest = {}
    for _, _, bundle in bundles:
        latest.setdefault(bundle["projectId"], bundle)
    projects, policies = [], {}
    for project_id, bundle in latest.items():
        saved = bundle["responses"]
        detail = saved["/api/projects/" + project_id]
        project = detail["project"]
        if project["id"] != project_id or saved["/api/projects"]["items"] != [project]:
            raise ValueError("Publication project catalog mismatch")
        projects.append(project)
        for policy in saved["/api/policies"]["items"]:
            if policy["projectId"] != project_id or policy["id"] in policies:
                raise ValueError("Publication policy catalog mismatch")
            policies[policy["id"]] = policy
        for path, value in saved.items():
            if immutable_route(path) or path in ("/api/publications", "/api/projects", "/api/policies"):
                continue
            if path in responses and responses[path] != value:
                raise ValueError(f"Conflicting project route payload: {path}")
            responses[path] = value
    responses["/api/publications"] = {"items": [bundle["responses"]["/api/publications"]["items"][0] for _, _, bundle in bundles]}
    responses["/api/projects"] = {"items": projects}
    responses["/api/policies"] = {"items": list(policies.values())}
    return responses, files


def compile_catalog(catalog_dir: str | Path, output_dir: str | Path):
    """Run verification once at deployment, then pre-encode immutable HTTP bodies.

    The deployment image is immutable. Runtime serves those exact verified bytes
    instead of parsing all historical snapshots and rehashing all blobs on boot.
    """
    root, output = Path(catalog_dir).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    responses, files = read_catalog(root)
    index = {"schemaVersion": 1, "routes": {}, "assets": {}}
    for route, value in responses.items():
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        sha = hashlib.sha256(body).hexdigest()
        name = sha + ".json.gz"
        if not (output / name).exists():
            temporary = output / (name + ".pending")
            temporary.write_bytes(gzip.compress(body, compresslevel=3, mtime=0))
            temporary.replace(output / name)
        index["routes"][route] = {"file": name, "sha256": sha, "sizeBytes": len(body)}
    for asset_id, (path, record) in files.items():
        index["assets"][asset_id] = {"path": str(path.relative_to(root)),
            **{key: record[key] for key in ("sha256", "sizeBytes", "mediaType")}}
    temporary = output / "index.pending.json"
    temporary.write_text(json.dumps(index, separators=(",", ":")))
    temporary.replace(output / "index.json")
    return {"routes": len(responses), "assets": len(files)}


def create_app(catalog_dir: str | Path, *, allowed_origins: list[str], feedback=None, prepared_dir: str | Path | None = None):
    """Each immediate publication-ID directory contains one unchanged export."""
    for origin in allowed_origins:
        parsed = urlsplit(origin)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise ValueError("CORS requires explicit HTTP(S) origins without paths")
    prepared = Path(prepared_dir).resolve() if prepared_dir else None
    index = None
    if prepared:
        index = json.loads((prepared / "index.json").read_text())
        if index["schemaVersion"] != 1:
            raise ValueError("Unsupported prepared publication index")
        responses = {}
        files = {asset_id: (Path(catalog_dir) / item["path"], item) for asset_id, item in index["assets"].items()}
    else:
        responses, files = read_catalog(catalog_dir)

    def get_record(path):
        if not prepared:
            return responses.get(path)
        record = index["routes"].get(path)
        if record is None:
            return None
        with gzip.open(prepared / record["file"], "rt", encoding="utf-8") as stream:
            return json.load(stream)

    app = FastAPI(title="Panoptes published report", docs_url=None, redoc_url=None, openapi_url=None)

    def error(status, code):
        return JSONResponse({"error": {"code": code, "params": {}}}, status_code=status, headers={"Cache-Control": "no-store"})

    @app.exception_handler(PlatformError)
    async def feedback_error(request, exc):
        return error(exc.status, exc.code)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return error(422, "invalid_request")

    @app.middleware("http")
    async def read_only(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS") and not (feedback is not None and request.method == "POST" and FEEDBACK_PATH.fullmatch(request.url.path)):
            return error(403, "publication_read_only")
        return await call_next(request)

    if feedback is not None:
        def feedback_scope(publication_id, request):
            publication = get_record("/api/publications/" + str(publication_id))
            if publication is None:
                raise PlatformError("publication_not_found", 404)
            authorization = request.headers.get("authorization", "")
            if not authorization.startswith("Feedback "):
                raise PlatformError("feedback_capability_required", 401)
            return publication, authorization.removeprefix("Feedback ")

        @app.get("/api/publications/{publication_id}/entities/{entity_id}/feedback")
        def feedback_history(publication_id: UUID, entity_id: str, conversationId: UUID, request: Request):
            publication, capability = feedback_scope(publication_id, request)
            return JSONResponse(feedback.history(publication, entity_id, capability, str(conversationId)), headers={"Cache-Control": "no-store"})

        @app.post("/api/publications/{publication_id}/entities/{entity_id}/feedback")
        def feedback_turn(publication_id: UUID, entity_id: str, body: FeedbackRequest, request: Request):
            publication, capability = feedback_scope(publication_id, request)
            return JSONResponse(feedback.submit(publication, entity_id, capability, body.model_dump(mode="json")), headers={"Cache-Control": "no-store"})

        @app.get("/api/publications/{publication_id}/identity-suggestions")
        def identity_suggestions(publication_id: UUID):
            publication = get_record("/api/publications/" + str(publication_id))
            if publication is None:
                raise PlatformError("publication_not_found", 404)
            return JSONResponse(feedback.identity_suggestions(publication), headers={"Cache-Control": "no-store"})

    @app.api_route("/api/assets/{asset_id}/content", methods=["GET", "HEAD"])
    def content(asset_id: str):
        if asset_id not in files:
            return error(404, "asset_not_found")
        path, record = files[asset_id]
        return FileResponse(path, media_type=record["mediaType"], headers={
            "ETag": '"' + record["sha256"] + '"',
            "Cache-Control": "public,max-age=31536000,immutable",
            "X-Content-Type-Options": "nosniff",
        })

    @app.api_route("/api/{path:path}", methods=["GET", "HEAD"])
    def record(path: str, request: Request):
        route = "/api/" + path
        headers = {"Cache-Control": "public,max-age=31536000,immutable" if immutable_route(route) else "no-cache",
                   "X-Content-Type-Options": "nosniff"}
        if prepared:
            entry = index["routes"].get(route)
            if entry is None:
                return error(404, "record_not_found")
            headers.update({"ETag": '"' + entry["sha256"] + '"', "Vary": "Accept-Encoding"})
            if request.headers.get("if-none-match") == headers["ETag"]:
                return Response(status_code=304, headers=headers)
            body_path = prepared / entry["file"]
            encodings = request.headers.get("accept-encoding", "").lower().split(",")
            accepts_gzip = any(part.strip().split(";")[0] in ("gzip", "*") and
                               not re.search(r";\s*q=0(?:\.0*)?(?:\s*;|$)", part) for part in encodings)
            if accepts_gzip:
                headers["Content-Encoding"] = "gzip"
                return FileResponse(body_path, media_type="application/json", headers=headers)
            def decoded():
                with gzip.open(body_path, "rb") as stream:
                    while chunk := stream.read(64 * 1024):
                        yield chunk
            headers["Content-Length"] = str(entry["sizeBytes"])
            return StreamingResponse(iter(()) if request.method == "HEAD" else decoded(), media_type="application/json", headers=headers)
        value = responses.get(route)
        if value is None:
            return error(404, "record_not_found")
        return JSONResponse(value, headers=headers)

    if not prepared:
        app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=3)
    class PublicationCORS:
        def __init__(self, app):
            self.read = CORSMiddleware(app, allow_origins=allowed_origins, allow_methods=["GET", "HEAD", "OPTIONS"],
                allow_headers=["Range", "If-Range", "If-None-Match"], expose_headers=["ETag", "Content-Length", "Content-Range"], allow_credentials=False)
            self.feedback = CORSMiddleware(app, allow_origins=allowed_origins, allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["Authorization", "Content-Type"], allow_credentials=False)

        async def __call__(self, scope, receive, send):
            feedback_path = feedback is not None and (FEEDBACK_PATH.fullmatch(scope.get("path", "")) or IDENTITY_SUGGESTIONS_PATH.fullmatch(scope.get("path", "")))
            target = self.feedback if feedback_path else self.read
            if feedback_path and scope.get("method") == "OPTIONS":
                headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in scope.get("headers", [])}

                async def diagnostic_send(message):
                    if message["type"] == "http.response.start":
                        print(json.dumps({"event": "feedback_cors_preflight", "request": {
                            key: headers.get(key) for key in ("origin", "access-control-request-method", "access-control-request-headers")},
                            "response": {key.decode("latin1"): value.decode("latin1") for key, value in message.get("headers", [])
                                         if key.lower().startswith(b"access-control-")}}, ensure_ascii=False), flush=True)
                    await send(message)

                await target(scope, receive, diagnostic_send)
            else:
                await target(scope, receive, send)

    app.add_middleware(PublicationCORS)
    return app
