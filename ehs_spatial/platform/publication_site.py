"""Serve a verified publication export; no database, provider, or write credentials."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse


def create_app(bundle_dir: str | Path, *, allowed_origins: list[str]):
    """Load an immutable bundle and expose only its saved public GET responses."""
    for origin in allowed_origins:
        parsed = urlsplit(origin)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise ValueError("CORS requires explicit HTTP(S) origins without paths")
    root = Path(bundle_dir).resolve()
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if bundle["schemaVersion"] != 1:
        raise ValueError("Unsupported publication bundle schema")
    responses = bundle["responses"]
    publication = responses["/api/publications/" + bundle["publicationId"]]
    if publication["id"] != bundle["publicationId"] or publication["projectId"] != bundle["projectId"]:
        raise ValueError("Publication bundle identity mismatch")
    files, checked = {}, set()
    for entry in publication["snapshot"]["assetManifest"]:
        asset_id, sha = entry["assetId"], entry["sha256"]
        if asset_id in files or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("Invalid publication asset manifest")
        record = responses["/api/assets/" + asset_id]
        if record["id"] != asset_id or record["projectId"] != bundle["projectId"] or record["url"] != f"/api/assets/{asset_id}/content" or any(record[key] != entry[key] for key in ("sha256", "sizeBytes", "mediaType")):
            raise ValueError("Publication asset metadata mismatch")
        path = root / "blobs" / sha
        if path.stat().st_size != entry["sizeBytes"]:
            raise ValueError(f"Publication asset size mismatch: {asset_id}")
        if sha not in checked:
            with path.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != sha:
                    raise ValueError(f"Publication asset hash mismatch: {asset_id}")
            checked.add(sha)
        files[asset_id] = (path, record)
    app = FastAPI(title="Panoptes published report", docs_url=None, redoc_url=None, openapi_url=None)

    def error(status, code):
        return JSONResponse({"error": {"code": code, "params": {}}}, status_code=status)

    @app.middleware("http")
    async def read_only(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            return error(403, "publication_read_only")
        return await call_next(request)

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
    def record(path: str):
        value = responses.get("/api/" + path)
        if value is None:
            return error(404, "record_not_found")
        immutable = path.startswith(("publications/", "revisions/", "assets/"))
        return JSONResponse(value, headers={"Cache-Control": "public,max-age=31536000,immutable" if immutable else "no-cache", "X-Content-Type-Options": "nosniff"})

    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=3)
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET", "HEAD", "OPTIONS"],
                       allow_headers=["Range", "If-Range", "If-None-Match"], expose_headers=["ETag", "Content-Length", "Content-Range"], allow_credentials=False)
    return app
