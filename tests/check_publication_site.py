"""Runnable end-to-end check: python tests/check_publication_site.py --bundle PATH."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from ehs_spatial.platform.contracts import Asset, ProjectDetail, Publication, PublicationSummary
from ehs_spatial.platform.publication_site import create_app


def check(bundle_dir: Path, source_api: str | None):
    bundle = json.loads((bundle_dir / "bundle.json").read_text())
    responses = bundle["responses"]
    path = "/api/publications/" + bundle["publicationId"]
    publication = responses[path]
    Publication.model_validate(publication)
    ProjectDetail.model_validate(responses["/api/projects/" + bundle["projectId"]])
    for item in responses["/api/publications"]["items"]:
        PublicationSummary.model_validate(item)
    if source_api:
        with urlopen(source_api.rstrip("/") + path) as source:
            assert publication == json.load(source), "Frozen publication was changed during export"
    manifest = publication["snapshot"]["assetManifest"]
    asset_ids = {entry["assetId"] for entry in manifest}
    assert {p.removeprefix("/api/assets/") for p in responses if p.startswith("/api/assets/")} == asset_ids
    assert [item["id"] for item in responses["/api/projects"]["items"]] == [bundle["projectId"]]
    assert all(item["projectId"] == bundle["projectId"] for item in responses["/api/policies"]["items"])
    origin = "https://report.example"
    with TestClient(create_app(bundle_dir, allowed_origins=[origin])) as client:
        for route, value in responses.items():
            response = client.get(route, headers={"Origin": origin})
            assert response.status_code == 200 and response.json() == value, route
            assert response.headers["access-control-allow-origin"] == origin
        for entry in manifest:
            asset = responses["/api/assets/" + entry["assetId"]]
            Asset.model_validate(asset)
            response = client.get(asset["url"])
            assert response.status_code == 200
            assert len(response.content) == entry["sizeBytes"]
            assert hashlib.sha256(response.content).hexdigest() == entry["sha256"]
            assert response.headers["etag"] == '"' + entry["sha256"] + '"'
            assert "immutable" in response.headers["cache-control"]
        selected = next(entry for entry in manifest if entry["sizeBytes"] >= 4)
        asset_path = responses["/api/assets/" + selected["assetId"]]["url"]
        ranged = client.get(asset_path, headers={"Range": "bytes=0-3", "Accept-Encoding": "identity"})
        assert ranged.status_code == 206 and len(ranged.content) == 4
        assert client.head(asset_path).content == b""
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            denied = client.request(method, path, headers={"Origin": origin})
            assert denied.status_code == 403 and denied.json()["error"]["code"] == "publication_read_only"
            assert denied.headers["access-control-allow-origin"] == origin
        assert client.get("/api/projects/unrelated").status_code == 404
        assert client.get("/api/assets/unrelated/content").status_code == 404
        assert "access-control-allow-origin" not in client.get(path, headers={"Origin": "https://unrelated.example"}).headers
        assert client.options(path, headers={"Origin": origin, "Access-Control-Request-Method": "GET"}).status_code == 200
        assert client.options(path, headers={"Origin": origin, "Access-Control-Request-Method": "POST"}).status_code == 400

    # A same-size corrupted file must fail startup, before any response can be served.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "blobs").mkdir()
        entry = min((item for item in manifest if item["sizeBytes"]), key=lambda item: item["sizeBytes"])
        broken = deepcopy(bundle)
        broken["responses"][path]["snapshot"]["assetManifest"] = [entry]
        (root / "bundle.json").write_text(json.dumps(broken))
        original = (bundle_dir / "blobs" / entry["sha256"]).read_bytes()
        (root / "blobs" / entry["sha256"]).write_bytes(bytes([original[0] ^ 1]) + original[1:])
        try:
            create_app(root, allowed_origins=[])
        except ValueError as error:
            assert "hash mismatch" in str(error)
        else:
            raise AssertionError("Corrupted asset was accepted")
    print(f"PASS: {len(responses)} exact read responses, {len(manifest)} byte-verified assets, CORS, ranges, write rejection, and corruption rejection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--source-api")
    args = parser.parse_args()
    check(args.bundle, args.source_api)
