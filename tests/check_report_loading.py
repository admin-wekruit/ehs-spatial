"""Run with .venv/bin/python tests/check_report_loading.py."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from check_publication_site import fixture
from fastapi.testclient import TestClient
from argus.platform.api import create_app as create_api
from argus.platform.publication_site import compile_catalog, create_app


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    catalog, prepared = root / "catalog", root / "http"
    publication_id = str(UUID(int=1))
    directory, bundle = fixture(catalog, publication_id, "project", "revision", "asset", "2026-09-16T00:00:00+00:00", "Report")
    route = "/api/publications/" + publication_id
    publication = bundle["responses"][route]
    edit = {"id": str(UUID(int=2)), "createdAt": publication["createdAt"], "baseRevisionId": "before",
            "revisionId": "revision", "operations": [{"type": "setLabel", "label": "Object"}],
            "inverseOperations": [{"type": "restoreScene", "document": {"evidence": "x" * 1_000_000}}]}
    publication["snapshot"]["editBatches"] = [edit]
    with TestClient(create_api(repository=SimpleNamespace(
        get_publication=lambda _: publication,
        get_project=lambda _: bundle["responses"]["/api/projects/project"],
    ), blobs=None)) as client:
        view = client.get(route + "/view")
        assert view.status_code == 200, view.text
        assert view.json()["edits"][0]["operationTypes"] == ["setLabel"]
        assert client.get(route + "/edits/" + edit["id"]).json() == edit
        assert client.get(route + "/edits/" + str(UUID(int=3))).status_code == 404
    (directory / "bundle.json").write_text(json.dumps(bundle))
    compile_catalog(catalog, prepared)
    # Serving must not scan/parse history or rehash the catalog at startup.
    with patch("argus.platform.publication_site.read_catalog", side_effect=AssertionError("Runtime catalog scan")):
        app = create_app(catalog, prepared_dir=prepared, allowed_origins=["https://report.example"])
    with TestClient(app) as client:
        for encoding in ("gzip", "identity", "gzip;q=0"):
            response = client.get(route + "/view", headers={"Accept-Encoding": encoding, "Origin": "https://report.example"})
            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == "https://report.example"
            assert (response.headers.get("content-encoding") == "gzip") == (encoding == "gzip")
            view = response.json()
            assert view["publication"]["snapshot"]["revision"] == publication["snapshot"]["revision"]
            assert view["publication"]["snapshot"]["editBatches"] is None
            assert view["edits"][0]["operationTypes"] == ["setLabel"]
            assert "inverseOperations" not in view["edits"][0]
            assert len(response.content) < 5_000, "History leaked into initial render payload"
            assert client.head(route + "/view", headers={"Accept-Encoding": encoding}).content == b""
            assert client.get(route + "/view", headers={"If-None-Match": response.headers["etag"]}).status_code == 304
        assert client.get(route).json() == publication, "Original publication changed"
        assert client.get(route + "/edits/" + edit["id"]).json() == edit, "Full audit event lost"
        assert client.get(route + "/edits/missing").status_code == 404
        for path, expected in bundle["responses"].items():
            assert client.get(path).json() == expected, path
        assert client.get("/api/assets/asset/content").content == b"Report artifact\n"
        assert client.post(route + "/view").status_code == 403
    blob = next((directory / "blobs").iterdir())
    blob.write_bytes(b"broken")
    try:
        compile_catalog(catalog, prepared)
    except ValueError as error:
        assert "mismatch" in str(error)
    else:
        raise AssertionError("Build accepted corrupt asset")

print("PASS: lean report view, lossless audit downloads, preverified startup, original routes, gzip/identity/HEAD/ETag, CORS and corruption rejection")
