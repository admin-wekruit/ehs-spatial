"""A publication owns its frozen process/download graph, never future live jobs."""
from copy import deepcopy
import io
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from argus.platform.api import create_app
from argus.platform.contracts import PlatformError, PublicationSnapshot
from argus.platform.storage import LocalBlobStore
from argus.platform.worker import run_job
from test_platform_backend import repo, project, identity, entity


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


@pytest.mark.parametrize('nested', [None, 'analysis', 'review', 'generation', 'captureAnalysis'])
@pytest.mark.parametrize("bad_ref,code", [("missing", "publication_asset_not_found"),
    ("foreign", "publication_job_asset_forbidden"), ("sha256", "publication_job_asset_integrity_conflict"),
    ("sizeBytes", "publication_job_asset_integrity_conflict"), ("invalid", "publication_job_asset_invalid")])
def test_publication_rejects_unverified_job_outputs_atomically(repo, tmp_path, bad_ref, code, nested):
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
    result = {'assets': [ref]}
    if nested:
        result = {nested: {'result': result} if nested == 'captureAnalysis' else result}
    repo.finish_job(job["id"], job["attemptToken"], "succeeded", result=result)
    with pytest.raises(PlatformError, match=code):
        repo.create_publication(scene["project"]["id"], cap, publish_body(scene))
    assert repo.list_publications()["items"] == []


def test_publication_keeps_nested_analysis_proof_without_following_configuration(repo, tmp_path):
    repo.blobs = LocalBlobStore(tmp_path)
    cap, scene = project(repo)
    job = repo.claim_job(enqueue(repo, cap, scene)['id'])
    checkpoint, stage, review, mesh, prepared, unrelated = [repo.register_asset(scene['project']['id'],
        repo.blobs.put(name.encode(), 'application/octet-stream'), job['id'])
        for name in ('checkpoint', 'stage', 'review', 'mesh', 'prepared', 'unrelated-private-configuration')]
    analysis = {'checkpointAssetId': checkpoint['id'], 'stages': [{'stage': 'discovery', 'assetId': stage['id']}]}
    result = {'captureAnalysis': {'jobId': identity(), 'baseRevisionId': scene['revision']['id'], 'result': analysis},
        'analysis': deepcopy(analysis), 'review': {'outputAssetId': review['id']}, 'generation': {'assets': [mesh]},
        'validationAssetId': prepared['id'], 'validationSha256': prepared['sha256'],
        'config': {'assets': [unrelated]}, 'error': {'params': {'outputAssetId': unrelated['id']}}}
    repo.finish_job(job['id'], job['attemptToken'], 'incomplete', result=result)
    publication = repo.create_publication(scene['project']['id'], cap, publish_body(scene))
    snapshot = publication['snapshot']
    ids = [asset['assetId'] for asset in snapshot['assetManifest']]
    assert len(ids) == len(set(ids)) == 5
    assert set(ids) == {checkpoint['id'], stage['id'], review['id'], mesh['id'], prepared['id']}
    assert snapshot['jobs'][0]['result'] == result
    assert not ({'attemptToken', 'executorRef', 'lateResults', 'requestSha256'} & snapshot['jobs'][0].keys())


def test_publication_checks_declared_prepared_input_hash(repo, tmp_path):
    repo.blobs = LocalBlobStore(tmp_path)
    cap, scene = project(repo)
    job = repo.claim_job(enqueue(repo, cap, scene)['id'])
    prepared = repo.register_asset(scene['project']['id'], repo.blobs.put(b'prepared', 'application/json'), job['id'])
    repo.finish_job(job['id'], job['attemptToken'], 'incomplete',
        result={'validationAssetId': prepared['id'], 'validationSha256': '0' * 64})
    with pytest.raises(PlatformError, match='publication_job_asset_integrity_conflict'):
        repo.create_publication(scene['project']['id'], cap, publish_body(scene))
    assert repo.list_publications()['items'] == []


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


def test_publication_library_describes_fixed_content_without_returning_snapshots(repo, tmp_path):
    repo.blobs = LocalBlobStore(tmp_path)
    summaries = []
    # A two-object acceptance scene and a larger observed workcell must have
    # visibly different summaries, even after their live branches advance.
    for photo_count, model_count, observed_count, missing_count in [(1, 1, 0, 1), (3, 9, 46, 13)]:
        cap, scene = project(repo)
        document = deepcopy(scene["revision"]["document"])
        document["coordinateFrames"] = [{"id": "native", "convention": "opencv", "scale": {"status": "uncalibrated", "nativeToMeters": None}, "ground": None}]
        for index in range(photo_count):
            data = io.BytesIO()
            Image.new("RGB", (8, 6), (index, 0, 0)).save(data, "PNG")
            asset = repo.register_asset(scene["project"]["id"], repo.blobs.put(data.getvalue(), "image/png"))
            document["assets"].append({**asset, "kind": "source_image"})
        pose = {"coordinateFrameId": "native", "position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "scale": [1, 1, 1]}
        kinds = ["primitive"] * model_count + ["observed_surface"] * observed_count + [None] * missing_count
        for kind in kinds + ["point_cloud"]:
            item = entity()
            if kind:
                item["representations"] = [{"id": identity(), "kind": kind, "coordinateFrameId": "native", "transform": pose, "placementState": "unconfirmed",
                    **({"primitive": {"kind": "box", "dimensions": [1, 1, 1]}} if kind == "primitive" else {})}]
            if kind == "point_cloud":
                item["sourceContext"] = True
            document["entities"].append(item)
        job = repo.claim_job(enqueue(repo, cap, scene, "analyze_capture")["id"])
        job = repo.finish_job(job["id"], job["attemptToken"], "succeeded", document=document)
        scene["revision"] = repo.get_revision(job["resultRevisionId"])
        publication = repo.create_publication(scene["project"]["id"], cap, publish_body(scene))
        summaries.append({"id": publication["id"], "previewImageAssetId": document["assets"][0]["id"], "photoCount": photo_count,
            "objectCount": len(kinds), "spatialObjectCount": model_count + observed_count, "modelObjectCount": model_count, "observedSurfaceObjectCount": observed_count})
        repo.commit_edits(scene["project"]["id"], cap, {"requestId": identity(), "branchId": scene["branch"]["id"], "baseRevisionId": scene["revision"]["id"],
            "operations": [{"type": "addEntity", "entity": entity()}]})
    cap, empty = project(repo)
    blank = repo.create_publication(empty["project"]["id"], cap, publish_body(empty))
    with TestClient(create_app(repository=repo, blobs=repo.blobs)) as client:
        response = client.get("/api/publications")
        assert response.status_code == 200
        rows = {row["id"]: row for row in response.json()["items"]}
        for expected in summaries:
            assert {key: rows[expected["id"]][key] for key in expected} == expected
        assert rows[blank["id"]]["previewImageAssetId"] is None
        assert rows[blank["id"]]["objectCount"] == rows[blank["id"]]["photoCount"] == 0
        assert all("snapshot" not in row and "document" not in row for row in rows.values())
        schema = client.get("/openapi.json").json()["components"]["schemas"]["PublicationSummary"]
        assert set(summaries[0]) <= set(schema["required"])
