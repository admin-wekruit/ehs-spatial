import json
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from ehs_spatial.artifacts import ArtifactStore
from ehs_spatial.report_workspace import register_workspace
from ehs_spatial.workspace_access import register_access


def test_public_snapshot_private_history_and_paid_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("PANOPTES_WRITE_TOKEN", "test-workspace-secret-not-a-real-credential")
    monkeypatch.setenv("PANOPTES_PUBLIC_RUNS", "published")
    for name in ["published", "new-private-upload"]:
        run = tmp_path / name
        run.mkdir()
        (run / "workspace.json").write_text(json.dumps({"title": name, "state": "queued"}))
    api = FastAPI()
    register_access(api)
    register_workspace(api, SimpleNamespace(store=ArtifactStore(tmp_path)))
    calls = []

    @api.post("/api/agent")
    def agent():
        calls.append("agent")
        return {"changed": True}

    client = TestClient(api, base_url="https://workspace.example")
    assert api.state.workspace_write_auth is True
    assert client.get("/api/session").json()["can_write"] is False
    assert client.post("/api/agent", json={}).status_code == 401 and calls == []
    assert client.get("/api/chat/published").status_code == 401
    assert client.get("/reports/new-private-upload").status_code == 401
    assert [r["run_id"] for r in client.get("/api/reports").json()["reports"]] == ["published"]
    assert client.post("/api/session", json={"token": "wrong"}).status_code == 401
    login = client.post("/api/session", json={"token": "test-workspace-secret-not-a-real-credential"})
    assert login.status_code == 200
    assert "HttpOnly" in login.headers["set-cookie"] and "Secure" in login.headers["set-cookie"]
    assert "test-workspace-secret" not in login.headers["set-cookie"]
    assert len(client.get("/api/reports").json()["reports"]) == 2
    assert client.post("/api/agent", json={}).status_code == 200 and calls == ["agent"]


def test_cross_site_cors_and_expiring_bearer_preserve_private_scope(tmp_path, monkeypatch):
    import hashlib
    import hmac
    from ehs_spatial import workspace_access
    secret = "test-only-access-code"
    origin = "https://admin-wekruit.github.io"
    monkeypatch.setenv("PANOPTES_WRITE_TOKEN", secret)
    monkeypatch.setenv("PANOPTES_PUBLIC_RUNS", "published")
    monkeypatch.setenv("PANOPTES_SITE_ORIGIN", origin)
    clock = [1_800_000_000]
    monkeypatch.setattr(workspace_access.time, "time", lambda: clock[0])
    for name in ["published", "private"]:
        run = tmp_path / name; run.mkdir()
        (run / "workspace.json").write_text(json.dumps({"title": name, "state": "done"}))
    api = FastAPI()
    register_access(api)
    register_workspace(api, SimpleNamespace(store=ArtifactStore(tmp_path)))
    calls = []

    @api.post("/api/agent")
    def agent():
        calls.append("agent")
        return {"changed": True}

    client = TestClient(api, base_url="https://backend.example")
    denied = client.get("/api/reports/private", headers={"Origin": origin})
    assert denied.status_code == 401 and denied.headers["access-control-allow-origin"] == origin
    assert client.get("/api/reports/published", headers={"Origin": origin}).status_code == 200
    preflight = client.options("/api/agent", headers={"Origin": origin,
        "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"})
    assert preflight.status_code == 200 and preflight.headers["access-control-allow-origin"] == origin
    assert "authorization" in preflight.headers["access-control-allow-headers"].lower()
    assert client.options("/api/agent", headers={"Origin": "https://evil.example",
        "Access-Control-Request-Method": "POST"}).status_code == 400
    login = client.post("/api/session", json={"token": secret}, headers={"Origin": origin})
    assert login.status_code == 200 and login.headers["Cache-Control"] == "no-store"
    token = login.json()["session_token"]
    permanent = hmac.new(secret.encode(), b"panoptes-workspace-session-v1", hashlib.sha256).hexdigest()
    assert secret not in token and token != permanent and token != secret
    assert login.json()["expires_at"] == clock[0] + 86400
    assert client.post("/api/session", json={"token": secret}, headers={"Origin": origin}).json()["session_token"] != token
    assert client.post("/api/session", json={"token": "错误"}, headers={"Origin": origin}).status_code == 401
    client.cookies.clear()  # Cross-site browsers may block third-party cookies.
    bearer = {"Origin": origin, "Authorization": "Bearer " + token}
    assert client.get("/api/session", headers=bearer).json()["can_write"] is True
    assert client.get("/api/reports/private", headers=bearer).status_code == 200
    assert {r["run_id"] for r in client.get("/api/reports", headers=bearer).json()["reports"]} == {"published", "private"}
    assert client.post("/api/agent", json={}, headers=bearer).status_code == 200 and calls == ["agent"]
    for bad in ["", "Basic " + token, "Bearer " + permanent, "Bearer " + secret,
                "Bearer " + token[:-1] + ("0" if token[-1] != "0" else "1"), "Bearer malformed", "Bearer " + "x"*500]:
        invalid = {"Origin": origin, "Authorization": bad}
        assert client.get("/api/reports/private", headers=invalid).status_code == 401
        assert client.post("/api/agent", json={}, headers=invalid).status_code == 401
        assert client.get("/api/reports/published", headers=invalid).status_code == 200
    assert client.get("/api/reports/private?session_token=" + token, headers={"Origin": origin}).status_code == 401
    clock[0] += 86400
    expired = client.post("/api/agent", json={}, headers=bearer)
    assert expired.status_code == 401 and expired.headers["access-control-allow-origin"] == origin
    clock[0] -= 86401
    assert client.get("/api/reports/private", headers=bearer).status_code == 401  # future-issued token
    clock[0] += 1
    assert client.post("/api/session", json={"token": secret}).status_code == 200
    # A malformed explicit bearer must not quietly fall back to a valid cookie.
    assert client.get("/api/reports/private", headers={"Authorization": "Bearer malformed"}).status_code == 401
    for bad_origin in ["https://evil.example", origin + ".evil.example", "null"]:
        response = client.post("/api/session", json={"token": secret}, headers={"Origin": bad_origin})
        assert response.status_code == 403 and "access-control-allow-origin" not in response.headers
        assert client.post("/api/agent", data="simple body", headers={"Origin": bad_origin,
            "Content-Type": "text/plain", "Authorization": "Bearer " + token}).status_code == 403
    assert client.post("/api/agent", json={}, headers={"Origin": "https://backend.example"}).status_code == 200
    assert client.post("/api/agent", json={}).status_code == 200  # authenticated non-browser clients


def test_site_origins_are_explicit_https_or_configured_loopback(monkeypatch):
    import pytest
    from ehs_spatial.workspace_access import _site_origins
    for invalid in ["*", "http://example.com", "https://example.com/path", "https://user@example.com", "null", "https://example.com:bad"]:
        monkeypatch.setenv("PANOPTES_SITE_ORIGIN", invalid)
        with pytest.raises(ValueError):
            _site_origins()
    monkeypatch.setenv("PANOPTES_SITE_ORIGIN", "https://admin-wekruit.github.io,http://127.0.0.1:8792,http://localhost:8806")
    assert _site_origins() == ["https://admin-wekruit.github.io", "http://127.0.0.1:8792", "http://localhost:8806"]
