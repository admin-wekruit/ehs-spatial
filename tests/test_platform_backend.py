"""Run with PANOPTES_TEST_DATABASE_URL to exercise actual PostgreSQL transactions."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
import io
import os
import secrets
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image
import psycopg
from psycopg import sql
import pytest

from ehs_spatial.platform.api import create_app
from ehs_spatial.platform.contracts import PlatformError, digest
from ehs_spatial.platform.postgres import PostgresRepository
from ehs_spatial.platform.storage import LocalBlobStore


def identity():
    return str(uuid4())


def capability():
    return "pcap_v1_" + secrets.token_urlsafe(32)


@pytest.fixture
def repo():
    dsn = os.environ.get("PANOPTES_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("PANOPTES_TEST_DATABASE_URL required for real PostgreSQL")
    schema = "backend_test_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    repository = PostgresRepository(dsn, paid_budget=Decimal("1"), schema=schema)
    repository.migrate()
    yield repository
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def project(repo):
    cap = capability()
    return cap, repo.create_project(cap, {"requestId": identity(), "title": "Fixture", "target": "scene"})


def entity():
    return {"id": identity(), "label": "Small object", "observationRefs": [], "associationState": "association_pending",
            "representations": [], "currentModelTransform": None, "measurements": {}, "groupId": None, "lineage": []}


def edit_body(scene, operations=None):
    return {"requestId": identity(), "branchId": scene["branch"]["id"], "baseRevisionId": scene["revision"]["id"],
            "operations": operations or [{"type": "addEntity", "entity": entity()}]}


def test_create_lost_response_capability_hash_and_mismatch(repo):
    cap = capability()
    body = {"requestId": identity(), "title": "Stable", "target": "scene"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: repo.create_project(cap, body), range(2)))
    assert results[0] == results[1]
    with repo._connect() as connection:
        stored = connection.execute("SELECT capability_sha256 FROM projects").fetchone()["capability_sha256"]
    assert stored == hashlib.sha256(cap.encode()).hexdigest()
    assert cap not in str(results)
    with pytest.raises(PlatformError, match="idempotency_mismatch"):
        repo.create_project(cap, {**body, "title": "Different"})


def test_branch_cas_cross_project_ownership_and_planning_independence(repo):
    cap, scene = project(repo)
    other_cap, other = project(repo)
    pid = scene["project"]["id"]
    with pytest.raises(PlatformError, match="capability_forbidden"):
        repo.commit_edits(pid, other_cap, edit_body(scene))
    with pytest.raises(PlatformError, match="branch_not_found"):
        repo.commit_edits(pid, cap, edit_body(other))
    planning = repo.create_branch(pid, cap, {"requestId": identity(), "sourceRevisionId": scene["revision"]["id"], "title": "Plan", "kind": "planning"})

    def commit(_):
        try:
            return repo.commit_edits(pid, cap, edit_body(scene))
        except PlatformError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(commit, range(2)))
    assert sum(isinstance(result, dict) for result in results) == 1
    error = next(result for result in results if isinstance(result, PlatformError))
    assert error.status == 409 and error.code == "revision_conflict"
    branch = next(item for item in repo.get_project(pid)["branches"] if item["id"] == planning["id"])
    assert branch["headRevisionId"] == planning["headRevisionId"] != scene["revision"]["id"]
    planning_initial=repo.get_revision(planning['headRevisionId'])
    assert planning_initial['branchId']==planning['id'] and planning_initial['sourceRevisionId']==scene['revision']['id']
    assert planning_initial['document']==scene['revision']['document']
    planning_edit = repo.commit_edits(pid, cap, {**edit_body(scene), "branchId": planning["id"], "baseRevisionId": planning['headRevisionId']})
    assert planning_edit["revision"]["branchId"] == planning["id"]


def test_edit_idempotency_undo_redo_append_immutable_revision(repo):
    cap, scene = project(repo)
    pid = scene["project"]["id"]
    body = edit_body(scene)
    committed = repo.commit_edits(pid, cap, body)
    assert committed == repo.commit_edits(pid, cap, body)
    assert committed["revision"]["document"]["entities"][0]["representations"] == []
    undo = repo.commit_edits(pid, cap, {"requestId": identity(), "branchId": scene["branch"]["id"], "baseRevisionId": committed["revision"]["id"], "undoOf": committed["editBatch"]["id"]})
    assert undo["revision"]["document"]["entities"] == []
    redo = repo.commit_edits(pid, cap, {"requestId": identity(), "branchId": scene["branch"]["id"], "baseRevisionId": undo["revision"]["id"], "redoOf": committed["editBatch"]["id"]})
    assert redo["revision"]["document"] == committed["revision"]["document"]
    assert len({scene["revision"]["id"], committed["revision"]["id"], undo["revision"]["id"], redo["revision"]["id"]}) == 4
    with pytest.raises(psycopg.errors.RaiseException, match="immutable_platform_record"):
        with repo._connect() as connection:
            connection.execute("UPDATE scene_revisions SET label='changed' WHERE id=%s", (scene["revision"]["id"],))


def test_v2_identity_migration_merge_cas_undo_and_frozen_publication(repo, tmp_path):
    from test_platform_identity import source_scene, decision
    cap, scene = project(repo)
    pid, branch_id = scene["project"]["id"], scene["branch"]["id"]
    repo.blobs = LocalBlobStore(tmp_path)
    images = []
    for color in ("red", "blue"):
        output = io.BytesIO()
        Image.new("RGB", (20, 20), color).save(output, format="PNG")
        images.append({**repo.blobs.put(output.getvalue(), "image/png"), "metadata": {"width": 20, "height": 20}})
    capture = repo.create_capture(pid, cap, {"requestId": identity(), "branchId": branch_id, "baseRevisionId": scene["revision"]["id"], "target": "scene"}, images)
    doc = source_scene()
    doc["captureId"] = capture["capture"]["id"]
    for index, image in enumerate(capture["capture"]["images"]):
        asset = repo.get_asset(image["assetId"])
        doc["assets"][index].update(id=asset["id"], sha256=asset["sha256"])
        doc["cameras"][index]["imageId"] = asset["id"]
        doc["observations"][index]["imageId"] = asset["id"]
    claimed = repo.claim_job(capture["job"]["id"])
    produced = repo.finish_job(claimed["id"], claimed["attemptToken"], "succeeded", document=doc)
    source = repo.get_revision(produced["resultRevisionId"])
    publication = repo.create_publication(pid, cap, {"requestId": identity(), "sceneRevisionId": source["id"], "evaluationIds": [], "reviewIds": [], "title": "Frozen v1"})
    frozen_sha = digest(publication)
    migration_body = {"requestId": identity(), "branchId": branch_id, "baseRevisionId": source["id"], "operations": [{"type": "migrateScene", "schemaVersion": 2}]}
    migrated = repo.commit_edits(pid, cap, migration_body)
    assert repo.commit_edits(pid, cap, migration_body) == migrated
    v2 = migrated["revision"]
    assert v2["document"]["schemaVersion"] == 2 and repo.get_revision(source["id"])["document"] == doc
    d = decision(v2["document"], base=v2["id"])
    merge_body = {"requestId": identity(), "branchId": branch_id, "baseRevisionId": v2["id"], "operations": [
        {"type": "recordIdentityDecision", "decision": d},
        {"type": "mergeEntities", "entityIds": d["entityIds"], "survivorId": d["survivorId"], "decisionId": d["id"]}]}
    saved = repo.commit_edits(pid, cap, merge_body)
    assert repo.commit_edits(pid, cap, merge_body) == saved
    assert len(saved["revision"]["document"]["entities"]) == 1
    with pytest.raises(PlatformError, match="revision_conflict"):
        repo.commit_edits(pid, cap, {**merge_body, "requestId": identity()})
    with pytest.raises(PlatformError, match="capability_forbidden"):
        repo.commit_edits(pid, capability(), merge_body)
    undo = repo.commit_edits(pid, cap, {"requestId": identity(), "branchId": branch_id, "baseRevisionId": saved["revision"]["id"], "undoOf": saved["editBatch"]["id"]})
    assert undo["revision"]["document"] == v2["document"]
    redo = repo.commit_edits(pid, cap, {"requestId": identity(), "branchId": branch_id, "baseRevisionId": undo["revision"]["id"], "redoOf": saved["editBatch"]["id"]})
    assert redo["revision"]["document"] == saved["revision"]["document"]
    assert digest(repo.get_publication(publication["id"])) == frozen_sha
    client = TestClient(create_app(repository=repo, blobs=repo.blobs))
    response = client.get("/api/revisions/" + redo["revision"]["id"])
    assert response.status_code == 200 and response.json()["document"] == redo["revision"]["document"]


def test_append_capture_preserves_scene_deduplicates_sources_and_fences_concurrent_jobs(repo, tmp_path):
    cap, scene = project(repo)
    pid, branch = scene["project"]["id"], scene["branch"]["id"]
    repo.blobs = LocalBlobStore(tmp_path)
    images = []
    for color in ("red", "blue", "green"):
        output = io.BytesIO()
        Image.new("RGB", (20, 20), color).save(output, format="PNG")
        images.append({**repo.blobs.put(output.getvalue(), "image/png"), "metadata": {"width": 20, "height": 20, "pixelMapping": []}})
    body = {"requestId": identity(), "branchId": branch, "baseRevisionId": scene["revision"]["id"], "target": "scene"}
    first = repo.create_capture(pid, cap, body, images[:2])
    item = entity()
    observations = [{"id": identity(), "revision": 1, "imageId": image["assetId"], "originalPixelBox": [1, 2, 4, 6], "maskAssetId": None} for image in first["capture"]["images"]]
    edited = repo.commit_edits(pid, cap, {"requestId": identity(), "branchId": branch, "baseRevisionId": first["revision"]["id"], "operations": [
        {"type": "addEntity", "entity": item}, *[{"type": "addObservation", "entityId": item["id"], "observation": obs} for obs in observations]]})
    migrated = repo.commit_edits(pid, cap, {"requestId": identity(), "branchId": branch, "baseRevisionId": edited["revision"]["id"], "operations": [{"type": "migrateScene"}]})
    before = migrated["revision"]["document"]
    append_body = {**body, "requestId": identity(), "baseRevisionId": migrated["revision"]["id"], "captureMode": "append"}
    appended = repo.create_capture(pid, cap, append_body, [images[0], images[2]])
    assert repo.create_capture(pid, cap, append_body, [images[0], images[2]]) == appended
    after = appended["revision"]["document"]
    for key in ("observations", "entities", "annotations", "identityDecisions", "coordinateFrames", "cameras"):
        assert after[key] == before[key]
    assert after["captureIds"] == [first["capture"]["id"], appended["capture"]["id"]]
    assert len(after["assets"]) == 3
    assert len(appended["job"]["inputs"]["newImageIds"]) == 1
    assert appended["job"]["inputs"]["reusedImageIds"] == [first["capture"]["images"][0]["assetId"]]
    assert after["geometryBindings"][appended["job"]["inputs"]["newImageIds"][0]] is None
    assert repo.get_revision(migrated["revision"]["id"])["document"] == before
    with pytest.raises(PlatformError, match="capture_append_required"):
        repo.create_capture(pid, cap, {**body, "requestId": identity(), "baseRevisionId": appended["revision"]["id"]}, [images[2]])
    with pytest.raises(PlatformError, match="image_count_out_of_range"):
        repo.create_capture(pid, cap, {**append_body, "requestId": identity(), "baseRevisionId": appended["revision"]["id"]}, images * 2)
    with pytest.raises(PlatformError, match="capture_image_mapping_conflict"):
        repo.create_capture(pid, cap, {**append_body, "requestId": identity(), "baseRevisionId": appended["revision"]["id"]}, [{**images[0], "metadata": {**images[0]["metadata"], "pixelMapping": [{"matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]}]}}])
    def competing_append(_):
        try:
            return repo.create_capture(pid, cap, {**append_body, "requestId": identity(), "baseRevisionId": appended["revision"]["id"]}, [images[2]])
        except PlatformError as error:
            return error
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(competing_append, (1, 2)))
    winner = next(result for result in results if isinstance(result, dict))
    loser = next(result for result in results if isinstance(result, PlatformError))
    assert loser.code == "revision_conflict"
    assert winner["job"]["inputs"]["newImageIds"] == [] and len(winner["job"]["inputs"]["reusedImageIds"]) == 1
    running = repo.claim_job(winner["job"]["id"])
    latest = repo.commit_edits(pid, cap, {"requestId": identity(), "branchId": branch, "baseRevisionId": winner["revision"]["id"], "operations": [{"type": "setLabel", "entityId": item["id"], "label": "Owner correction"}]})
    late = repo.finish_job(running["id"], running["attemptToken"], "succeeded", document=winner["revision"]["document"])
    assert late["headAdvanced"] is False and late["resultRevisionId"]
    assert repo.get_project(pid)["revision"]["id"] == latest["revision"]["id"]


def test_fork_independent_capability_publication_is_snapshot(repo):
    cap, scene = project(repo)
    pid = scene["project"]["id"]
    publication = repo.create_publication(pid, cap, {"requestId": identity(), "sceneRevisionId": scene["revision"]["id"], "evaluationIds": [], "reviewIds": [], "title": "Original"})
    assert publication['snapshot']['reportSchemaVersion']==1 and publication['snapshot']['rendererVersion']=='native-webgl-v1'
    assert publication['snapshot']['assetManifest']==[]
    assert publication['snapshot']['playgroundDefinitions']==[{'kind':'observed','revisionId':scene['revision']['id']},{'kind':'model','revisionId':scene['revision']['id']}]
    fork_cap = capability()
    fork = repo.fork_project(pid, fork_cap, {"requestId": identity(), "sourceRevisionId": scene["revision"]["id"], "title": "Visitor", "operations": []})
    with pytest.raises(PlatformError, match="capability_forbidden"):
        repo.commit_edits(fork["project"]["id"], cap, edit_body(fork))
    repo.commit_edits(fork["project"]["id"], fork_cap, edit_body(fork))
    repo.commit_edits(pid, cap, edit_body(scene))
    assert repo.get_publication(publication["id"])["snapshot"]["revision"]["document"] == scene["revision"]["document"]
    assert repo.get_revision(scene["revision"]["id"])["document"]["entities"] == []


def make_job(repo, cap, scene):
    return repo.create_job(scene["project"]["id"], cap, {"requestId": identity(), "branchId": scene["branch"]["id"],
        "baseRevisionId": scene["revision"]["id"], "kind": "reconstruct", "inputs": {}, "config": {"native_setting": 1}})


def test_durable_outbox_cancel_old_attempt_and_late_branch_result(repo):
    cap, scene = project(repo)
    job = make_job(repo, cap, scene)
    assert job["status"] == "pending_dispatch"
    assert job["config"] == {"native_setting": 1, "providerManifest": {}}
    repo.mark_dispatched(job["id"], "executor:1")
    claimed = repo.claim_job(job["id"])
    with pytest.raises(PlatformError, match="job_not_claimable"):
        repo.claim_job(job["id"])
    edited = repo.commit_edits(scene["project"]["id"], cap, edit_body(scene))
    late = repo.finish_job(job["id"], claimed["attemptToken"], "succeeded", document=scene["revision"]["document"], result={"ok": True})
    assert late["resultRevisionId"] and late["headAdvanced"] is False
    assert repo.get_project(scene["project"]["id"])["revision"]["id"] == edited["revision"]["id"]
    duplicate = repo.finish_job(job["id"], claimed["attemptToken"], "succeeded", document=scene["revision"]["document"])
    assert duplicate["lateResultSaved"]
    next_scene = {**scene, "revision": edited["revision"]}
    cancelled = make_job(repo, cap, next_scene)
    running = repo.claim_job(cancelled["id"])
    requested = repo.cancel_job(cancelled["id"], cap)
    assert requested["status"] == "running" and requested["cancelRequested"]
    result = repo.finish_job(cancelled["id"], running["attemptToken"], "succeeded", document=scene["revision"]["document"])
    assert result["lateResultSaved"] and repo.get_job(cancelled["id"])["status"] == "cancelled"
    assert repo.get_project(scene["project"]["id"])["revision"]["id"] == edited["revision"]["id"]


def test_model_reservation_budget_and_unknown_outcome_never_replayed(repo):
    cap, scene = project(repo)
    job = repo.claim_job(make_job(repo, cap, scene)["id"])
    hashes = {key: digest(key) for key in ("code_sha256", "model_sha256", "adapter_sha256", "input_sha256")}
    call = repo.reserve_model_call(job["id"], job["attemptToken"], "test", "test-model", "input-1", ".75", **hashes)
    repo.complete_model_call(call["id"], "outcome_unknown")
    with pytest.raises(PlatformError, match="model_call_already_reserved"):
        repo.reserve_model_call(job["id"], job["attemptToken"], "test", "test-model", "input-1", ".75", **hashes)
    with pytest.raises(PlatformError, match="paid_budget_exceeded"):
        repo.reserve_model_call(job["id"], job["attemptToken"], "test", "test-model", "input-2", ".3", **hashes)
    repo.paid_budget = None
    with pytest.raises(PlatformError, match="paid_budget_not_configured"):
        repo.reserve_model_call(job["id"], job["attemptToken"], "test", "test-model", "input-3", "0", **hashes)
    free = repo.reserve_model_call(job["id"], job["attemptToken"], "local", "primitive", "free", "0", paid=False, **hashes)
    assert free["status"] == "reserved"


def test_failed_provider_usage_counts_actual_cost_before_next_reservation(repo):
    cap,scene=project(repo)
    job=repo.claim_job(make_job(repo,cap,scene)['id'])
    call=repo.reserve_model_call(job['id'],job['attemptToken'],'test','test','first','.1')
    repo.complete_model_call(call['id'],'failed',actual_cost='.9')
    with pytest.raises(PlatformError,match='paid_budget_exceeded'):
        repo.reserve_model_call(job['id'],job['attemptToken'],'test','test','second','.2')


def test_expired_call_preserves_late_usage_without_reopening_job(repo):
    cap,scene=project(repo)
    job=repo.claim_job(make_job(repo,cap,scene)['id'])
    call=repo.reserve_model_call(job['id'],job['attemptToken'],'test','test','first','.1')
    with repo._connect() as connection:
        connection.execute("UPDATE jobs SET lease_expires_at=now()-interval '1 minute' WHERE id=%s",(job['id'],))
    repo.recover_expired_jobs()
    late=repo.complete_model_call(call['id'],'succeeded',actual_cost='.9',response={'usage':42})
    assert late['status']=='outcome_unknown' and late['actualCost']==0.9
    assert late['response']['lateOutcome']['response']['usage']==42
    assert repo.complete_model_call(call['id'],'succeeded',actual_cost='.9',response={'usage':42})==late
    assert repo.get_job(job['id'])['status']=='outcome_unknown'
    with pytest.raises(PlatformError,match='model_call_already_reserved'):
        repo.reserve_model_call(job['id'],job['attemptToken'],'test','test','first','.1')


def test_upload_api_real_image_exif_idempotency_blob_integrity(repo, tmp_path):
    blobs = LocalBlobStore(tmp_path)
    client = TestClient(create_app(repository=repo, blobs=blobs))
    cap = capability()
    headers = {"Authorization": "Capability " + cap}
    created = client.post("/api/projects", headers=headers, json={"requestId": identity(), "title": "Images", "target": "standalone_object"})
    assert created.status_code == 201
    scene = created.json()
    pid = scene["project"]["id"]
    image = Image.new("RGB", (20, 10), "red")
    exif = Image.Exif()
    exif[274] = 6
    output = io.BytesIO()
    image.save(output, format="JPEG", exif=exif)
    form = {"requestId": identity(), "branchId": scene["branch"]["id"], "baseRevisionId": scene["revision"]["id"], "target": "standalone_object"}
    result = client.post(f"/api/projects/{pid}/captures", headers=headers, data=form, files=[("files", ("rotated.jpg", output.getvalue(), "image/jpeg"))])
    assert result.status_code == 201, result.text
    retried = client.post(f"/api/projects/{pid}/captures", headers=headers, data=form, files=[("files", ("renamed.jpg", output.getvalue(), "image/jpeg"))])
    assert retried.status_code == 201 and retried.json() == result.json()
    capture = result.json()["capture"]
    assert result.json()["job"]["kind"] == "analyze_capture"
    assert result.json()["job"]["baseRevisionId"] == result.json()["revision"]["id"]
    assert capture["task"]["target"] == "standalone_object"
    uploaded = capture["images"][0]
    assert (uploaded["width"], uploaded["height"], uploaded["exifOrientation"]) == (10, 20, 6)
    assert uploaded["pixelMapping"][0]["matrix"] == [[0, -1, 9], [1, 0, 0], [0, 0, 1]]
    asset = client.get("/api/assets/" + uploaded["assetId"]).json()
    downloaded = client.get(asset["url"])
    assert downloaded.status_code == 200 and hashlib.sha256(downloaded.content).hexdigest() == asset["sha256"]
    (tmp_path / asset["storageKey"]).write_bytes(b"corruption")
    assert client.get(asset["url"]).json()["error"]["code"] == "blob_integrity_error"
    invalid = client.post(f"/api/projects/{pid}/captures", headers=headers, data={**form, "requestId": identity()}, files=[("files", ("fake.png", b"not-an-image", "image/png"))])
    assert invalid.status_code == 422
    assert client.get("/api/revisions/" + identity()).status_code == 404
    assert client.get("/api/projects").text.find(cap) == -1


def test_local_blob_store_immutable_bytes(tmp_path):
    store = LocalBlobStore(tmp_path)
    metadata = store.put(b"bytes", "application/octet-stream")
    assert store.put(b"bytes", "application/octet-stream") == metadata
    assert store.get(metadata["storageKey"], metadata["sha256"], metadata["sizeBytes"]) == b"bytes"
    with pytest.raises(PlatformError, match="invalid_storage_key"):
        store.get("../../etc/passwd", metadata["sha256"], metadata["sizeBytes"])


def test_agent_atomic_ledger_and_expired_paid_call_not_retried(repo):
    cap, scene = project(repo)
    body = {"requestId": identity(), "conversationId": identity(), "branchId": scene["branch"]["id"], "baseRevisionId": scene["revision"]["id"], "message": "Inspect selected object", "language": "en"}
    turn = repo.create_agent_turn(scene["project"]["id"], cap, body)
    assert repo.create_agent_turn(scene["project"]["id"], cap, body)["jobId"] == turn["jobId"]
    assert not repo.pending_jobs()
    job = repo.claim_job(turn["jobId"])
    repo.claim_agent_turn(turn["id"])
    assert repo.heartbeat_job(job["id"], job["attemptToken"])
    call = repo.reserve_model_call(job["id"], job["attemptToken"], "test", "test-model", "agent-input", ".5")
    with repo._connect() as c:
        c.execute("UPDATE jobs SET lease_expires_at=now()-interval '1 second' WHERE id=%s", (job["id"],))
    assert repo.recover_expired_jobs() == [job["id"]]
    assert repo.get_job(job["id"])["status"] == "outcome_unknown"
    assert repo.list_agent_turns(scene["project"]["id"], conversation_id=body["conversationId"])["items"][0]["status"] == "outcome_unknown"
    with repo._connect() as c:
        assert c.execute("SELECT status FROM model_calls WHERE id=%s", (call["id"],)).fetchone()["status"] == "outcome_unknown"
    assert not repo.pending_jobs()


def test_never_claimed_queued_job_can_recover_without_replaying_paid_work(repo):
    cap, scene = project(repo)
    job = make_job(repo, cap, scene)
    repo.mark_dispatched(job['id'],'dead-process')
    with repo._connect() as c:
        c.execute("UPDATE jobs SET dispatched_at=now()-interval '6 minutes' WHERE id=%s",(job['id'],))
    repo.recover_expired_jobs()
    assert repo.get_job(job['id'])['status']=='pending_dispatch'
    claimed=repo.claim_job(job['id'])
    assert claimed['attempt']==1


def test_offline_import_retries_and_missing_blob_cannot_commit(repo,tmp_path,monkeypatch):
    from scripts import import_public_scene
    from scripts.import_public_scene import run_import
    from test_platform_import import make_public_scene
    source=tmp_path/'source';source.mkdir()
    path=make_public_scene(source)
    blobs=LocalBlobStore(tmp_path/'blobs')
    output=tmp_path/'imports'
    first=run_import(path,repo,blobs,output)
    assert first==run_import(path,repo,blobs,output)
    management=next(output.glob('*.management.json'))
    assert management.stat().st_mode & 0o777 == 0o600
    capability_value=__import__('json').loads(management.read_text())['capability']
    assert capability_value not in str(first)
    next(output.glob('*.manifest.json')).unlink()
    retried=run_import(path,repo,blobs,output)
    assert retried['publicationId']==first['publicationId']
    document=repo.get_revision(first['sceneRevisionId'])['document']
    capture=repo.list_project_records(first['projectId'],'captures')['items'][0]
    assert document['captureId']==capture['id']
    assert capture['task']['recomputeRequiresNewCapture'] and len(capture['images'])==1
    assert len(repo.get_publication(first['publicationId'])['snapshot']['assetManifest'])==len(document['assets'])
    edited=repo.commit_edits(first['projectId'],capability_value,{'requestId':identity(),'branchId':first['branchId'],'baseRevisionId':first['sceneRevisionId'],'operations':[{'type':'setLabel','entityId':document['entities'][0]['id'],'label':'User edit'}]})
    monkeypatch.setattr(import_public_scene,'CONVERTER_VERSION','test-next-converter')
    upgraded=run_import(path,repo,blobs,output)
    assert upgraded['projectId']==first['projectId'] and upgraded['publicationId']!=first['publicationId']
    assert upgraded['branchId']!=first['branchId']
    assert repo.get_project(first['projectId'])['revision']['id']==edited['revision']['id']
    assert repo.get_publication(first['publicationId'])['snapshot']['revision']['id']==first['sceneRevisionId']
    asset=repo.get_asset(document['assets'][0]['id'])
    (blobs.root/asset['storageKey']).write_bytes(b'corrupt')
    body={'requestId':identity(),'branchId':first['branchId'],'baseRevisionId':edited['revision']['id'],'operations':[{'type':'setLabel','entityId':document['entities'][0]['id'],'label':'changed'}]}
    with pytest.raises(PlatformError,match='blob_integrity_error'):
        repo.commit_edits(first['projectId'],capability_value,body)



def test_policy_service_real_database_unknown_applicability_and_typed_invalid(repo, tmp_path):
    from ehs_spatial.platform.policy_service import PolicyService, templates
    cap, scene = project(repo)
    from ehs_spatial.platform.policy_repository import PostgresPolicyRepository
    service = PolicyService(PostgresPolicyRepository(repo), LocalBlobStore(tmp_path))
    client = TestClient(create_app(repository=repo, blobs=service.blobs, policy_service=service))
    headers = {"Authorization": "Capability " + cap}
    pid = scene["project"]["id"]
    invalid = client.post(f"/api/projects/{pid}/policies", json={}, headers=headers)
    assert invalid.status_code == 422
    assert client.post("/api/policies/test", json={"jdm": {"nodes": [None], "edges": []}, "tests": [{}]}).status_code == 422
    template = templates()[1]
    created = service.create_policy(pid, cap, {**template, "requestId": identity()})
    activated = service.activate(pid, created["policy"]["id"], cap, {"requestId": identity(), "policyRevisionId": created["revision"]["id"], "expectedActiveRevisionId": None})
    assert activated["active"]
    evaluation = service.evaluate(pid, cap, {"requestId": identity(), "sceneRevisionId": scene["revision"]["id"], "policyRevisionIds": [created["revision"]["id"]], "context": "observed"})
    finding = evaluation["document"]["findings"][0]
    assert finding['policyTitle']==created['policy']['title']
    assert finding["applicability"] == "unknown" and finding["machineResult"] is None
    with pytest.raises(PlatformError, match="review_evidence_invalid"):
        service.review(pid, finding["id"], cap, {"requestId": identity(), "evaluationId": evaluation["id"], "decision": "confirmed", "reason": "Reviewed", "displayName": "Reviewer", "evidenceRefs": [identity()]})
    review=service.review(pid,finding['id'],cap,{'requestId':identity(),'evaluationId':evaluation['id'],'decision':'needs_evidence','reason':'Need source evidence','displayName':'Reviewer'})
    assert client.get(f'/api/projects/{pid}/reviews').json()['items'][0]['id']==review['id']
    evidence_request=service.review(pid,finding['id'],cap,{'requestId':identity(),'evaluationId':evaluation['id'],'action':'Attach evidence'},evidence_request=True)
    evidence_id=identity()
    changed=repo.commit_edits(pid,cap,edit_body(scene,[{'type':'addAnnotation','annotation':{'id':evidence_id,'kind':'operator_note','text':'Additional evidence'}}]))
    evaluated=service.evaluate(pid,cap,{'requestId':identity(),'sceneRevisionId':changed['revision']['id'],'policyRevisionIds':[created['revision']['id']],'context':'observed'})
    followup={'requestId':identity(),'evaluationId':evaluated['id'],'findingId':evaluated['document']['findings'][0]['id'],'evidenceRefs':[evidence_id]}
    invalid=client.post(f"/api/projects/{pid}/evidence-requests/{evidence_request['id']}/fulfill",json={**followup,'evidenceRefs':[identity()]},headers=headers)
    assert invalid.status_code==422
    fulfilled=client.post(f"/api/projects/{pid}/evidence-requests/{evidence_request['id']}/fulfill",json=followup,headers=headers)
    assert fulfilled.status_code==200 and fulfilled.json()['document']['status']=='fulfilled'
    assert client.post(f"/api/projects/{pid}/evidence-requests/{evidence_request['id']}/fulfill",json=followup,headers=headers).json()==fulfilled.json()
    records=client.get(f'/api/projects/{pid}/evidence-requests').json()['items']
    assert len(records)==2 and {r['document']['status'] for r in records}=={'open','fulfilled'}
    assert service.publication_evidence(pid,scene['revision']['id'],[evaluation['id']],[review['id']])['reviews'][0]['id']==review['id']
    planning=repo.create_branch(pid,cap,{'requestId':identity(),'sourceRevisionId':changed['revision']['id'],'title':'Unedited plan','kind':'planning'})
    planned=service.evaluate(pid,cap,{'requestId':identity(),'sceneRevisionId':planning['headRevisionId'],'policyRevisionIds':[created['revision']['id']],'context':'planning'})
    assert planned['context']=='planning' and planned['sceneRevisionId']==planning['headRevisionId']
    published=repo.create_publication(pid,cap,{'requestId':identity(),'sceneRevisionId':planning['headRevisionId'],'title':'Plan','evaluationIds':[],'reviewIds':[]})
    assert published['snapshot']['branchKind']=='planning'
    assert published['snapshot']['reconstructionRevision']['id']==changed['revision']['id']


def test_openapi_response_contracts_preserve_stored_documents_and_cover_routes(repo,tmp_path):
    from ehs_spatial.platform.api import _public
    from ehs_spatial.platform.contracts import canonical
    from ehs_spatial.platform.policy_service import PolicyService,templates
    from ehs_spatial.platform.policy_repository import PostgresPolicyRepository
    from scripts.import_public_scene import run_import
    from test_platform_import import make_public_scene
    source=tmp_path/'source';source.mkdir()
    blobs=LocalBlobStore(tmp_path/'blobs')
    imported=run_import(make_public_scene(source),repo,blobs,tmp_path/'imports')
    service=PolicyService(PostgresPolicyRepository(repo),blobs)
    app=create_app(repository=repo,blobs=blobs,policy_service=service)
    client=TestClient(app)
    schema=app.openapi()
    binary={'/api/assets/{asset_id}/content','/api/blobs/sha256/{sha256}'}
    for path,methods in schema['paths'].items():
        for operation in methods.values():
            for status,response in operation['responses'].items():
                if status.startswith('2'):
                    media='application/octet-stream' if path in binary else 'application/json'
                    shape=response['content'][media]['schema']
                    assert shape=={'type':'string','format':'binary'} if path in binary else '$ref' in shape
            assert operation['responses']['422']['content']['application/json']['schema']['$ref'].endswith('/ErrorResponse')
    for name in ('ProjectDetail','SceneDocument','Revision','Job','Publication','Asset','Commit','PolicyDetail','PolicyRevision','Evaluation','Review'):
        assert name in schema['components']['schemas']
    pid,rid=imported['projectId'],imported['sceneRevisionId']
    for url,expected in [
        (f'/api/projects/{pid}',repo.get_project(pid)),
        (f'/api/revisions/{rid}',repo.get_revision(rid)),
        (f"/api/publications/{imported['publicationId']}",repo.get_publication(imported['publicationId'])),
        ('/api/publications',repo.list_publications()),
        ('/api/projects',repo.list_projects()),
    ]:
        response=client.get(url)
        assert response.status_code==200
        assert canonical(response.json())==canonical(expected)
    for kind in ('revisions','captures','assets','jobs','edits'):
        response=client.get(f'/api/projects/{pid}/{kind}')
        assert response.status_code==200
        assert canonical(response.json())==canonical(_public(repo.list_project_records(pid,kind)))
    scene=repo.get_revision(rid)
    asset=repo.get_asset(scene['document']['assets'][0]['id'])
    response=client.get(f"/api/assets/{asset['id']}")
    assert response.json()=={**asset,'url':blobs.url(asset['storageKey'],asset['id'])}
    assert client.get(f"/api/assets/{asset['id']}/content").content==blobs.get(asset['storageKey'],asset['sha256'],asset['sizeBytes'])
    assert client.get('/api/policy-templates').json()=={'items':templates()}
    cap,created=project(repo)
    headers={'Authorization':'Capability '+cap}
    policy=client.post(f"/api/projects/{created['project']['id']}/policies",json={**templates()[0],'requestId':identity()},headers=headers)
    assert policy.status_code==200
    policy_id=policy.json()['policy']['id']
    assert client.get(f'/api/policies/{policy_id}').json()==service.get_policy(policy_id)
    assert client.get('/api/policies').json()==service.list_policies()
    body={'requestId':identity(),'conversationId':identity(),'branchId':created['branch']['id'],'baseRevisionId':created['revision']['id'],'message':'What evidence is missing?'}
    turn=repo.create_agent_turn(created['project']['id'],cap,body)
    repo.claim_agent_turn(turn['id'])
    repo.finish_agent_turn(turn['id'],'succeeded',{'kind':'answer','message':'Source photo required.','provider_evidence':{'frame_id':'original'}})
    response=client.get(f"/api/projects/{created['project']['id']}/agent-turns")
    assert response.status_code==200
    assert canonical(response.json())==canonical(_public(repo.list_agent_turns(created['project']['id'])))


def test_identity_agent_scope_and_evidence_fulfillment_keep_historical_evaluation(repo, tmp_path):
    from test_platform_identity import source_scene, decision
    from ehs_spatial.platform.contracts import EvidenceRequest
    from ehs_spatial.platform.policy_repository import PostgresPolicyRepository
    from ehs_spatial.platform.policy_service import PolicyService, templates
    cap, scene = project(repo)
    pid, branch = scene['project']['id'], scene['branch']['id']
    repo.blobs = LocalBlobStore(tmp_path)
    service = PolicyService(PostgresPolicyRepository(repo), repo.blobs)
    policy = service.create_policy(pid, cap, {**templates()[0], 'requestId': identity()})
    service.activate(pid, policy['policy']['id'], cap, {'requestId': identity(), 'policyRevisionId': policy['revision']['id'], 'expectedActiveRevisionId': None})
    images = []
    for color in ('red', 'blue'):
        output = io.BytesIO()
        Image.new('RGB', (20, 20), color).save(output, format='PNG')
        images.append({**repo.blobs.put(output.getvalue(), 'image/png'), 'metadata': {'width': 20, 'height': 20}})
    capture = repo.create_capture(pid, cap, {'requestId': identity(), 'branchId': branch, 'baseRevisionId': scene['revision']['id'], 'target': 'scene'}, images)
    doc = source_scene()
    doc['captureId'] = capture['capture']['id']
    for index, image in enumerate(capture['capture']['images']):
        asset = repo.get_asset(image['assetId'])
        doc['assets'][index].update(id=asset['id'], sha256=asset['sha256'])
        doc['cameras'][index]['imageId'] = asset['id']
        doc['observations'][index]['imageId'] = asset['id']
    context = {**entity(), 'sourceContext': True}
    doc['entities'].append(context)
    doc['annotations'].append({'id': identity(), 'kind': 'policy_applicability', 'policyId': policy['policy']['id'], 'value': 'applicable', 'sourceRefs': [{'observationId': 'observation-2'}]})
    claimed = repo.claim_job(capture['job']['id'])
    result = repo.finish_job(claimed['id'], claimed['attemptToken'], 'succeeded', document=doc)
    migrated = repo.commit_edits(pid, cap, {'requestId': identity(), 'branchId': branch, 'baseRevisionId': result['resultRevisionId'], 'operations': [{'type': 'migrateScene'}]})['revision']
    agent = {'requestId': identity(), 'conversationId': identity(), 'branchId': branch, 'baseRevisionId': migrated['id'], 'message': 'Compare', 'entityId': 'entity-1', 'identityEntityIds': ['entity-1', 'entity-2']}
    turn = repo.create_agent_turn(pid, cap, agent)
    assert repo.create_agent_turn(pid, cap, agent) == turn
    for invalid_pair in (['entity-1'], ['entity-1', 'entity-1'], ['entity-1', identity()], ['entity-1', context['id']]):
        with pytest.raises(PlatformError, match='agent_identity_scope_invalid'):
            repo.create_agent_turn(pid, cap, {**agent, 'requestId': identity(), 'identityEntityIds': invalid_pair})
    with pytest.raises(PlatformError, match='agent_identity_scope_invalid'):
        repo.create_agent_turn(pid, cap, {**agent, 'requestId': identity(), 'entityId': context['id']})
    evaluate = {'requestId': identity(), 'sceneRevisionId': migrated['id'], 'policyRevisionIds': [policy['revision']['id']], 'context': 'observed'}
    original = service.evaluate(pid, cap, evaluate)
    original_hash = digest(original)
    finding = next(f for f in original['document']['findings'] if f['entityId'] == 'entity-2')
    request = service.review(pid, finding['id'], cap, {'requestId': identity(), 'evaluationId': original['id'], 'action': 'Inspect this observation'}, evidence_request=True)
    d = decision(migrated['document'], ids=['entity-1', 'entity-2'], base=migrated['id'])
    changed = repo.commit_edits(pid, cap, {'requestId': identity(), 'branchId': branch, 'baseRevisionId': migrated['id'], 'operations': [
        {'type': 'recordIdentityDecision', 'decision': d}, {'type': 'mergeEntities', 'entityIds': d['entityIds'], 'survivorId': d['survivorId'], 'decisionId': d['id']},
        {'type': 'addAnnotation', 'annotation': {'id': identity(), 'kind': 'manual_evidence', 'entityId': 'entity-1', 'requirement': 'walking_surface_hazard_review', 'passed': True, 'sourceRefs': [{'observationId': 'observation-2'}]}}]})
    followup = service.evaluate(pid, cap, {**evaluate, 'requestId': identity(), 'sceneRevisionId': changed['revision']['id']})
    new_finding = next(f for f in followup['document']['findings'] if f['entityId'] == 'entity-1')
    body = {'requestId': identity(), 'evaluationId': followup['id'], 'findingId': new_finding['id'], 'evidenceRefs': [{'observationId': 'observation-2'}]}
    with pytest.raises(PlatformError, match='evidence_identity_scope_mismatch'):
        service.fulfill_evidence_request(pid, request['id'], cap, {**body, 'requestId': identity(), 'evidenceRefs': [{'observationId': 'observation-1'}]})
    fulfillment = service.fulfill_evidence_request(pid, request['id'], cap, body)
    assert fulfillment['document']['identityBinding']['observationIds'] == ['observation-2']
    assert EvidenceRequest.model_validate(fulfillment).document.identityBinding.sourceFindingId == finding['id']
    assert service.fulfill_evidence_request(pid, request['id'], cap, body) == fulfillment
    assert digest(next(e for e in service.repository.list_evaluations(pid)['items'] if e['id'] == original['id'])) == original_hash
    assert next(r for r in service.repository.list_evidence_requests(pid)['items'] if r['id'] == request['id'])['document']['status'] == 'open'


def test_jsonb_roundtrip_hashes_are_stable_without_changing_existing_revisions(repo):
    from ehs_spatial.platform.contracts import canonical
    from psycopg.types.json import Jsonb
    from copy import deepcopy
    cap, scene = project(repo)
    pid = scene['project']['id']
    old_revision = deepcopy(repo.get_revision(scene['revision']['id']))
    publication = repo.create_publication(pid, cap, {'requestId': identity(), 'sceneRevisionId': scene['revision']['id'], 'evaluationIds': [], 'reviewIds': [], 'title': 'Immutable before numeric revision'})
    values = [-0.0, 0.0, 1.0, 1e16, 1e20, -1.2345678901234568e20, 1.2345678901234568e20, 1e-7, 5e-324, 1.7976931348623157e308, 123456789012345680000]
    payload = {'nested': [{'values': values, 'literal': '-0.0 1e+20'}], 'one': 1.0}
    source = deepcopy(payload)
    with repo._connect() as c:
        restored = c.execute('SELECT %s::jsonb AS value', (Jsonb(payload),)).fetchone()['value']
    assert digest(payload) == digest(restored)
    assert payload == source and canonical({'one': 1.0}) == b'{"one":1.0}'
    assert canonical(1.2345678901234568e20) == b'123456789012345680000'
    assert canonical(payload) == canonical(restored)
    for bad in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ValueError):
            canonical({'bad': bad})
    body = edit_body(scene, [{'type': 'addAnnotation', 'annotation': {'id': identity(), 'kind': 'numeric_evidence', 'payload': payload}}])
    committed = repo.commit_edits(pid, cap, body)
    fetched = repo.get_revision(committed['revision']['id'])
    assert fetched['documentSha256'] == digest(fetched['document'])
    assert repo.commit_edits(pid, cap, body) == committed
    assert repo.get_revision(old_revision['id']) == old_revision
    assert repo.get_publication(publication['id']) == publication
