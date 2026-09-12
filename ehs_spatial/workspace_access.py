"""One private editing workspace; only explicitly published reports are public."""
import hashlib
import hmac
import os
import re
import secrets
import time
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

SESSION_SECONDS = 24 * 60 * 60


class AccessIn(BaseModel):
    token: str = Field(min_length=1, max_length=256)


def _session_token(secret: str) -> tuple[str, int]:
    issued = int(time.time())
    payload = f"v1.{issued}.{secrets.token_urlsafe(24)}"
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return payload + "." + signature, issued + SESSION_SECONDS


def _valid_bearer(header: str, secret: str) -> bool:
    # Each login has independent random entropy; the permanent cookie digest is
    # never accepted as a bearer credential. The browser treats this as opaque.
    if not secret or len(header) > 256:
        return False
    match = re.fullmatch(r"(?i:Bearer) (v1\.([0-9]{1,12})\.[A-Za-z0-9_-]{32})\.([a-f0-9]{64})", header)
    if not match:
        return False
    payload, issued, signature = match.groups()
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected) and int(issued) <= time.time() < int(issued) + SESSION_SECONDS


def _site_origins() -> list[str]:
    origins = list(dict.fromkeys(filter(None, (value.strip() for value in os.environ.get("PANOPTES_SITE_ORIGIN", "").split(",")))))
    for origin in origins:
        parsed = urlsplit(origin)
        loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if (not parsed.hostname or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment
                or parsed.scheme not in ({"https", "http"} if loopback else {"https"})
                or "*" in origin or parsed.netloc != parsed.netloc.lower()):
            raise ValueError("PANOPTES_SITE_ORIGIN must contain exact HTTPS origins (HTTP loopback is allowed for local testing)")
        try:
            parsed.port
        except ValueError as error:
            raise ValueError("PANOPTES_SITE_ORIGIN has an invalid port") from error
    return origins


def register_access(api):
    secret = os.environ.get("PANOPTES_WRITE_TOKEN", "")
    public_runs = set(filter(None, os.environ.get("PANOPTES_PUBLIC_RUNS", "").split(",")))
    origins = _site_origins()
    session = hmac.new(secret.encode(), b"panoptes-workspace-session-v1", hashlib.sha256).hexdigest()
    api.state.workspace_write_auth = bool(secret)

    @api.middleware("http")
    async def access(request: Request, call_next):
        origin = request.headers.get("origin")
        same_origin = f"{request.url.scheme}://{request.url.netloc}"
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin is not None and origin not in {*origins, same_origin}:
            return JSONResponse({"detail": "This website is not allowed to modify the workspace", "code": "workspace_origin_denied"}, status_code=403)
        authorization = request.headers.get("authorization")
        unlocked = not secret or (_valid_bearer(authorization, secret) if authorization is not None
                                 else hmac.compare_digest(request.cookies.get("panoptes_session", "").encode(), session.encode()))
        request.state.can_write = unlocked
        request.state.public_runs = public_runs
        path = request.url.path
        if not unlocked and path != "/api/session":
            denied = request.method not in {"GET", "HEAD", "OPTIONS"} or path.startswith(("/api/chat/", "/workbench"))
            for prefix in ("/api/reports/", "/reports/", "/report/"):
                if path.startswith(prefix):
                    run_id = path[len(prefix):].split("/", 1)[0]
                    denied |= run_id not in public_runs
            if denied:
                return JSONResponse({"detail": "Workspace access required", "code": "workspace_access_required"}, status_code=401)
        return await call_next(request)

    @api.get("/api/session")
    def state(request: Request):
        return JSONResponse({"can_write": request.state.can_write, "requires_access": bool(secret)}, headers={"Cache-Control": "no-store"})

    @api.post("/api/session")
    def unlock(body: AccessIn, request: Request):
        if not secret or not hmac.compare_digest(body.token.encode(), secret.encode()):
            raise HTTPException(401, "Invalid workspace access code")
        token, expires_at = _session_token(secret)
        response = JSONResponse({"can_write": True, "session_token": token, "expires_at": expires_at,
                                 "expires_in": SESSION_SECONDS}, headers={"Cache-Control": "no-store"})
        response.set_cookie("panoptes_session", session, httponly=True,
                            secure=request.url.scheme == "https", samesite="lax", max_age=SESSION_SECONDS)
        return response

    # Added last so preflights and authentication failures receive the same
    # precise CORS policy. A CORS header alone never authorizes a mutation.
    api.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True,
                       allow_methods=["GET", "HEAD", "POST", "OPTIONS"],
                       allow_headers=["Authorization", "Content-Type"], max_age=600)
