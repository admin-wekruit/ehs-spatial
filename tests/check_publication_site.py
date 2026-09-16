"""Run catalog regressions; optionally verify exports with --catalog PATH --source-api URL."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from urllib.request import urlopen
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from ehs_spatial.platform.contracts import Asset, ProjectDetail, Publication, PublicationSummary
from ehs_spatial.platform.publication_site import create_app


def check(catalog_dir: Path, source_api: str | None = None):
    bundles = [json.loads((directory / "bundle.json").read_text()) for directory in catalog_dir.iterdir() if directory.is_dir()]
    bundles.sort(key=lambda bundle: (datetime.fromisoformat(bundle["responses"]["/api/publications/" + bundle["publicationId"]]["createdAt"]), bundle["publicationId"]), reverse=True)
    latest = {}
    for bundle in bundles:
        latest.setdefault(bundle["projectId"], bundle)
    expected, assets = {}, {}
    for bundle in bundles:
        responses = bundle["responses"]
        route = "/api/publications/" + bundle["publicationId"]
        publication = responses[route]
        Publication.model_validate(publication)
        ProjectDetail.model_validate(responses["/api/projects/" + bundle["projectId"]])
        for item in responses["/api/publications"]["items"]:
            PublicationSummary.model_validate(item)
        if source_api:
            with urlopen(source_api.rstrip("/") + route) as source:
                assert publication == json.load(source), "Frozen publication was changed during export"
        for path, value in responses.items():
            if path.startswith(("/api/publications/", "/api/revisions/", "/api/assets/")):
                assert path not in expected or expected[path] == value
                expected[path] = value
        for entry in publication["snapshot"]["assetManifest"]:
            record = responses["/api/assets/" + entry["assetId"]]
            Asset.model_validate(record)
            assets[entry["assetId"]] = entry
    for project_id, bundle in latest.items():
        for path, value in bundle["responses"].items():
            if path not in ("/api/publications", "/api/projects", "/api/policies"):
                expected[path] = value
    expected["/api/publications"] = {"items": [bundle["responses"]["/api/publications"]["items"][0] for bundle in bundles]}
    expected["/api/projects"] = {"items": [bundle["responses"]["/api/projects"]["items"][0] for bundle in latest.values()]}
    expected["/api/policies"] = {"items": [policy for bundle in latest.values() for policy in bundle["responses"]["/api/policies"]["items"]]}
    origin = "https://report.example"
    with TestClient(create_app(catalog_dir, allowed_origins=[origin])) as client:
        for route, value in expected.items():
            response = client.get(route, headers={"Origin": origin})
            assert response.status_code == 200 and response.json() == value, route
            assert response.headers["access-control-allow-origin"] == origin
        for asset_id, entry in assets.items():
            response = client.get(f"/api/assets/{asset_id}/content")
            assert response.status_code == 200
            assert len(response.content) == entry["sizeBytes"]
            assert hashlib.sha256(response.content).hexdigest() == entry["sha256"]
            assert response.headers["etag"] == '"' + entry["sha256"] + '"'
            assert "immutable" in response.headers["cache-control"]
        selected = next((entry for entry in assets.values() if entry["sizeBytes"] >= 4), None)
        if selected:
            asset_path = f"/api/assets/{selected['assetId']}/content"
            ranged = client.get(asset_path, headers={"Range": "bytes=0-3", "Accept-Encoding": "identity"})
            assert ranged.status_code == 206 and len(ranged.content) == 4
            assert client.head(asset_path).content == b""
        fixed_path = "/api/publications/" + bundles[0]["publicationId"]
        assert "immutable" in client.get(fixed_path).headers["cache-control"]
        assert client.get("/api/publications").headers["cache-control"] == "no-cache"
        assert client.get("/api/projects/" + bundles[0]["projectId"]).headers["cache-control"] == "no-cache"
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            denied = client.request(method, fixed_path, headers={"Origin": origin})
            assert denied.status_code == 403 and denied.json()["error"]["code"] == "publication_read_only"
            assert denied.headers["access-control-allow-origin"] == origin
        assert client.get("/api/projects/unrelated").status_code == 404
        assert client.get("/api/assets/unrelated/content").status_code == 404
        assert "access-control-allow-origin" not in client.get(fixed_path, headers={"Origin": "https://unrelated.example"}).headers
        assert client.options(fixed_path, headers={"Origin": origin, "Access-Control-Request-Method": "GET"}).status_code == 200
        assert client.options(fixed_path, headers={"Origin": origin, "Access-Control-Request-Method": "POST"}).status_code == 400
    return len(bundles), len(expected), len(assets)


def fixture(root, publication_id, project_id, revision_id, asset_id, created_at, title):
    """Small valid exported records; never writes a production catalog."""
    branch_id = "branch-" + project_id
    raw = (title + " artifact\n").encode()
    sha = hashlib.sha256(raw).hexdigest()
    asset = {"id": asset_id, "projectId": project_id, "createdAt": created_at, "jobId": None,
             "storageKey": "sha256/" + sha, "sha256": sha, "sizeBytes": len(raw), "mediaType": "text/plain",
             "metadata": {}, "url": f"/api/assets/{asset_id}/content"}
    document = {"schemaVersion": 1, "target": "scene", "coordinateFrames": [], "cameras": [], "observations": [],
                "entities": [], "assets": [{"id": asset_id}], "annotations": []}
    revision = {"id": revision_id, "projectId": project_id, "createdAt": created_at, "branchId": branch_id,
                "parentRevisionId": None, "sourceRevisionId": None, "document": document,
                "documentSha256": hashlib.sha256(json.dumps(document).encode()).hexdigest(), "label": title}
    project = {"id": project_id, "createdAt": created_at, "title": title, "requestId": "request",
               "defaultBranchId": branch_id, "forkSourceRevisionId": None}
    branch = {"id": branch_id, "projectId": project_id, "createdAt": created_at, "kind": "reconstruction",
              "title": title, "sourceRevisionId": None, "headRevisionId": revision_id, "requestId": "request"}
    entry = {"assetId": asset_id, "sha256": sha, "sizeBytes": len(raw), "mediaType": "text/plain"}
    publication = {"id": publication_id, "projectId": project_id, "createdAt": created_at, "sceneRevisionId": revision_id,
                   "title": title, "requestId": "request", "evaluationIds": [], "reviewIds": [],
                   "snapshot": {"schemaVersion": 1, "revision": revision, "evaluations": [], "reviews": [], "jobs": [], "assetManifest": [entry]}}
    summary = {key: publication[key] for key in ("id", "projectId", "createdAt", "sceneRevisionId", "title")}
    summary.update(previewImageAssetId=None, photoCount=0, objectCount=0, spatialObjectCount=0, modelObjectCount=0, observedSurfaceObjectCount=0)
    policy = {"id": "policy-" + project_id, "projectId": project_id, "createdAt": created_at, "requestId": "request",
              "title": title, "activeRevisionId": None, "draftRevisionId": None, "activationRequests": {}}
    responses = {
        "/api/publications": {"items": [summary]}, "/api/publications/" + publication_id: publication,
        "/api/projects": {"items": [project]}, "/api/projects/" + project_id: {"project": project, "branch": branch, "branches": [branch], "revision": revision},
        "/api/projects/" + project_id + "/revisions": {"items": [revision]}, "/api/revisions/" + revision_id: revision,
        "/api/projects/" + project_id + "/assets": {"items": [{key: value for key, value in asset.items() if key != "url"}]},
        "/api/assets/" + asset_id: asset, "/api/policies": {"items": [policy]},
        "/api/policies/" + policy["id"]: {"policy": policy, "revisions": [], "sources": []},
    }
    bundle = {"schemaVersion": 1, "publicationId": publication_id, "projectId": project_id, "responses": responses}
    directory = root / publication_id
    (directory / "blobs").mkdir(parents=True)
    (directory / "blobs" / sha).write_bytes(raw)
    (directory / "bundle.json").write_text(json.dumps(bundle))
    return directory, bundle


def regressions():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        # Newest time is one microsecond later, but sorts earlier by date text
        # and directory name. File order/export time must not choose the head.
        older, old = fixture(root, str(UUID(int=900)), "project-a", "revision-old", "asset-old", "2026-09-13T05:00:00.100000+01:00", "Older release")
        newer, new = fixture(root, str(UUID(int=2)), "project-a", "revision-new", "asset-new", "2026-09-12T23:00:00.100001-05:00", "Newer release")
        old_revision = old["responses"]["/api/revisions/revision-old"]
        new["responses"]["/api/publications/" + new["publicationId"]]["snapshot"]["reconstructionRevision"] = old_revision
        new["responses"]["/api/revisions/revision-old"] = old_revision
        new["responses"]["/api/projects/project-a/revisions"]["items"].append(old_revision)
        (newer / "bundle.json").write_text(json.dumps(new))
        fixture(root, str(UUID(int=3)), "project-b", "revision-other", "asset-other", "2026-09-13T03:00:00+00:00", "Another project")
        publication_count, _, asset_count = check(root)
        assert publication_count == 3 and asset_count == 3
        app = create_app(root, allowed_origins=[])
        record = next(route.endpoint for route in app.routes if route.path == "/api/{path:path}")
        saved = inspect.getclosurevars(record).nonlocals["responses"]
        current = saved["/api/revisions/revision-new"]
        assert saved["/api/publications/" + new["publicationId"]]["snapshot"]["revision"] is current
        assert saved["/api/projects/project-a"]["revision"] is current
        assert saved["/api/projects/project-a/revisions"]["items"][0] is current
        historical = saved["/api/revisions/revision-old"]
        assert saved["/api/publications/" + old["publicationId"]]["snapshot"]["revision"] is historical
        assert saved["/api/publications/" + new["publicationId"]]["snapshot"]["reconstructionRevision"] is historical
        assert saved["/api/projects/project-a/revisions"]["items"][1] is historical
        with TestClient(app) as client:
            assert [item["id"] for item in client.get("/api/publications").json()["items"]] == [new["publicationId"], old["publicationId"], str(UUID(int=3))]
            assert client.get("/api/projects/project-a").json()["revision"]["id"] == "revision-new"
            assert client.get("/api/projects/project-a/revisions").json()["items"][0]["id"] == "revision-new"
            assert client.get("/api/policies/policy-project-a").json()["policy"]["title"] == "Newer release"
            assert client.get("/api/revisions/revision-old").json() == old["responses"]["/api/revisions/revision-old"]
            assert client.get("/api/assets/asset-old/content").content == b"Older release artifact\n", "An asset omitted from the new snapshot must remain downloadable"
        try:
            create_app(older, allowed_origins=[])
        except ValueError as error:
            assert "catalog" in str(error)
        else:
            raise AssertionError("A single bundle was accepted as a catalog")
        for location in (
            ("/api/projects/project-a", "revision"),
            ("/api/projects/project-a/revisions", "items", 0),
            ("/api/publications/" + new["publicationId"], "snapshot", "reconstructionRevision"),
        ):
            broken = json.loads(json.dumps(new))
            altered = broken["responses"]
            for key in location:
                altered = altered[key]
            altered["document"]["annotations"].append({"id": "conflicting-content"})
            (newer / "bundle.json").write_text(json.dumps(broken))
            try:
                create_app(root, allowed_origins=[])
            except ValueError as error:
                assert "Conflicting immutable route payload" in str(error)
            else:
                raise AssertionError(f"Conflicting embedded revision was shared: {location}")
        broken = deepcopy(new)
        broken["responses"]["/api/revisions/revision-old"] = {**old["responses"]["/api/revisions/revision-old"], "label": "silently replaced"}
        (newer / "bundle.json").write_text(json.dumps(broken))
        try:
            create_app(root, allowed_origins=[])
        except ValueError as error:
            assert "Conflicting immutable route payload" in str(error)
        else:
            raise AssertionError("Conflicting immutable history was accepted")
        (newer / "bundle.json").write_text(json.dumps(new))
        entry = old["responses"]["/api/publications/" + old["publicationId"]]["snapshot"]["assetManifest"][0]
        blob = older / "blobs" / entry["sha256"]
        original = blob.read_bytes()
        blob.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        try:
            create_app(root, allowed_origins=[])
        except ValueError as error:
            assert "hash mismatch" in str(error)
        else:
            raise AssertionError("Corrupt historical asset was accepted")
    print("PASS: shared immutable revisions, full-content conflicts, multi-publication history, precise time ordering, latest project projection, catalog union, old assets, corruption, CORS, ranges and write rejection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--source-api")
    args = parser.parse_args()
    regressions()
    if args.catalog:
        publications, responses, assets = check(args.catalog, args.source_api)
        print(f"PASS: {publications} published bundles, {responses} exact read responses, {assets} byte-verified assets")
