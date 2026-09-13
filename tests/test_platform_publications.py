"""A publication owns its frozen process/download graph, never future live jobs."""
from copy import deepcopy
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from ehs_spatial.platform.api import create_app
from ehs_spatial.platform.contracts import PlatformError, PublicationSnapshot
from ehs_spatial.platform.storage import LocalBlobStore
from panoptes_worker.__main__ import run_job
from test_platform_backend import repo, project, identity


def enqueue(repo, cap, scene, kind="export_json"):
    return repo.create_job(scene["project"]["id"], cap, {"requestId": identity(), "branchId": scene["branch"]["id"],
        "baseRevisionId": scene["revision"]["id"], "kind": kind, "inputs": {}, "config": {}})


def publish_body(scene):
    return {"requestId": identity(), "sceneRevisionId": scene["revision"]["id"], "title": "Fixed report", "evaluationIds": [], "reviewIds": []}


def test_publication_freezes_producer_and_consumer_jobs_output_assets_and_public_dto(repo, tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo.blobs = blobs
    cap, scene = project(repo)
    old_export = run_job(repo, blobs, enqueue(repo, cap, scene)["id"], {})
    producer = repo.claim_job(enqueue(repo, cap, scene, "analyze_capture")["id"])
    checkpoint = repo.register_asset(scene["project"]["id"], blobs.put(b"checkpoint", "application/octet-stream"), producer["id"])
    document = deepcopy(scene["revision"]["document"])
    document["annotations"] = [{"id": identity(), "kind": "analysis_completed"}]
    producer = repo.finish_job(producer["id"], producer["attemptToken"], "succeeded", document=document,
        result={"stages": [{"stage": "inventory", "assetId": checkpoint["id"]}], "checkpointAssetId": checkpoint["id"]})
    scene["revision"] = repo.get_revision(producer["resultRevisionId"])
    exported = run_job(repo, blobs, enqueue(repo, cap, scene)["id"], {})
    pending = enqueue(repo, cap, scene, "generate_scene")
    body = publish_body(scene)
    publication = repo.create_publication(scene["project"]["id"], cap, body)
    snapshot = publication["snapshot"]
    assert [job["id"] for job in snapshot["jobs"]] == [producer["id"], exported["id"], pending["id"]]
    assert snapshot["jobs"][2]["status"] == "pending_dispatch"
    assets = {asset["assetId"] for asset in snapshot["assetManifest"]}
    assert assets == {checkpoint["id"], exported["result"]["assets"][0]["id"]}
    assert old_export["result"]["assets"][0]["id"] not in assets
    assert all(not ({"attemptToken", "executorRef", "lateResults", "requestSha256"} & job.keys()) for job in snapshot["jobs"])
    started = repo.claim_job(pending["id"])
    repo.finish_job(started["id"], started["attemptToken"], "failed", result={"error": {"code": "later_failure"}})
    enqueue(repo, cap, scene)
    assert repo.get_publication(publication["id"]) == publication
    assert repo.create_publication(scene["project"]["id"], cap, body) == publication
    with TestClient(create_app(repository=repo, blobs=blobs)) as client:
        response = client.get("/api/publications/" + publication["id"])
        assert response.status_code == 200 and response.json() == publication
        schema = client.get("/openapi.json").json()["components"]["schemas"]["PublicationSnapshot"]
        assert schema["properties"]["jobs"]["items"]["$ref"].endswith("/Job")
    old_snapshot = {key: value for key, value in snapshot.items() if key != "jobs"}
    assert PublicationSnapshot.model_validate(old_snapshot).jobs == []


@pytest.mark.parametrize("bad_ref,code", [("missing", "publication_asset_not_found"),
    ("foreign", "publication_job_asset_forbidden"), ("sha256", "publication_job_asset_integrity_conflict"),
    ("sizeBytes", "publication_job_asset_integrity_conflict"), ("invalid", "publication_job_asset_invalid")])
def test_publication_rejects_unverified_job_outputs_atomically(repo, tmp_path, bad_ref, code):
    blobs = LocalBlobStore(tmp_path)
    repo.blobs = blobs
    cap, scene = project(repo)
    _, foreign = project(repo)
    job = repo.claim_job(enqueue(repo, cap, scene)["id"])
    owner = foreign["project"]["id"] if bad_ref == "foreign" else scene["project"]["id"]
    asset = repo.register_asset(owner, blobs.put(b"artifact", "application/octet-stream"))
    ref = deepcopy(asset)
    if bad_ref == "missing":
        ref["id"] = str(uuid4())
    elif bad_ref == "sha256":
        ref["sha256"] = "0" * 64
    elif bad_ref == "sizeBytes":
        ref["sizeBytes"] += 1
    elif bad_ref == "invalid":
        ref["id"] = "not-an-asset"
    repo.finish_job(job["id"], job["attemptToken"], "succeeded", result={"assets": [ref]})
    with pytest.raises(PlatformError, match=code):
        repo.create_publication(scene["project"]["id"], cap, publish_body(scene))
    assert repo.list_publications()["items"] == []


def test_publication_rechecks_output_blob_bytes(repo, tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo.blobs = blobs
    cap, scene = project(repo)
    exported = run_job(repo, blobs, enqueue(repo, cap, scene)["id"], {})
    output = exported["result"]["assets"][0]
    (tmp_path / output["storageKey"]).write_bytes(b"corrupted after job completion")
    with pytest.raises(PlatformError, match="blob_integrity_error"):
        repo.create_publication(scene["project"]["id"], cap, publish_body(scene))
    assert repo.list_publications()["items"] == []
