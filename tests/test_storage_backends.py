"""One storage contract, every backend.

Repositories: MongoDB via mongomock (always), a real MongoDB when PANOPTES_TEST_MONGO_URL is set,
PostgreSQL when PANOPTES_TEST_DATABASE_URL is set. Blob stores: local (always), S3 via moto
(always), a real S3-compatible bucket when PANOPTES_TEST_S3_URL=s3://bucket/prefix is set
(with AWS_* / AWS_ENDPOINT_URL in the environment). No server is started here.
"""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
import inspect
import io
import json
import os
import time
from uuid import uuid4

import boto3
from moto import mock_aws
from PIL import Image
import pytest

from ehs_spatial.platform import policy_repository as policy_repository_module
from ehs_spatial.platform.config import PlatformConfig
from ehs_spatial.platform.contracts import PlatformError, digest
from ehs_spatial.platform.mongo import MongoRepository
from ehs_spatial.platform.mongo_policy import MongoPolicyRepository
from ehs_spatial.platform.policy_service import PolicyService, templates
from ehs_spatial.platform.postgres import PostgresRepository
from ehs_spatial.platform.policy_repository import PostgresPolicyRepository
from ehs_spatial.platform.runtime import blob_store, policy_repository, repository_from_url, services
from ehs_spatial.platform.s3_storage import S3BlobStore, parse_s3_url, s3_client
from ehs_spatial.platform.storage import LocalBlobStore
import test_platform_backend as legacy
from test_platform_backend import capability, edit_body, entity, identity, make_job, project

MB = 1024 * 1024
MOTO_ENV = {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "AWS_DEFAULT_REGION": "us-east-1"}


def _moto_env(monkeypatch):
    for key, value in MOTO_ENV.items():
        monkeypatch.setenv(key, value)
    for key in ("AWS_ENDPOINT_URL", "PANOPTES_S3_ENDPOINT_URL", "PANOPTES_BLOB_PUBLIC_BASE", "AWS_PROFILE"):
        monkeypatch.delenv(key, raising=False)


# ---- fixtures -----------------------------------------------------------------------------

@pytest.fixture(params=["mongomock", "mongo", "postgres"])
def repo(request, tmp_path):
    blobs = LocalBlobStore(tmp_path / "blobs")
    if request.param == "mongomock":
        import mongomock
        repository = MongoRepository("mongodb://localhost/contract", client=mongomock.MongoClient(tz_aware=True), paid_budget=Decimal("1"), prefix="t_", blob_store=blobs)
        repository.migrate()
        yield repository
    elif request.param == "mongo":
        url = os.environ.get("PANOPTES_TEST_MONGO_URL")
        if not url:
            pytest.skip("PANOPTES_TEST_MONGO_URL required for real MongoDB")
        repository = MongoRepository(url, paid_budget=Decimal("1"), prefix="contract_" + uuid4().hex[:8] + "_", blob_store=blobs)
        repository.migrate()
        yield repository
        for name in repository.db.list_collection_names():
            if name.startswith(repository.prefix):
                repository.db.drop_collection(name)
    else:
        dsn = os.environ.get("PANOPTES_TEST_DATABASE_URL")
        if not dsn:
            pytest.skip("PANOPTES_TEST_DATABASE_URL required for real PostgreSQL")
        import psycopg
        from psycopg import sql
        schema = "contract_test_" + uuid4().hex
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        repository = PostgresRepository(dsn, paid_budget=Decimal("1"), schema=schema, blob_store=blobs)
        repository.migrate()
        yield repository
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def mongo(repo):
    if not isinstance(repo, MongoRepository):
        pytest.skip("MongoDB-specific check")
    return repo


@pytest.fixture(params=["local", "s3-moto", "s3"])
def blobs(request, tmp_path, monkeypatch):
    if request.param == "local":
        yield LocalBlobStore(tmp_path / "store")
    elif request.param == "s3-moto":
        _moto_env(monkeypatch)
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="contract-bucket")
            yield S3BlobStore.from_url("s3://contract-bucket/blobs", multipart_threshold=5 * MB)
    else:
        url = os.environ.get("PANOPTES_TEST_S3_URL")
        if not url:
            pytest.skip("PANOPTES_TEST_S3_URL=s3://bucket/prefix required for a real S3-compatible store")
        store = S3BlobStore.from_url(url.rstrip("/") + "/contract-" + uuid4().hex[:8], multipart_threshold=5 * MB)
        yield store
        _delete_prefix(store.client, store.bucket, store.prefix)


def _delete_prefix(client, bucket, prefix):
    listing = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [{"Key": item["Key"]} for item in listing.get("Contents", [])]
    if keys:
        client.delete_objects(Bucket=bucket, Delete={"Objects": keys})


@pytest.fixture(params=["s3-moto", "s3"])
def artifact_bucket(request, monkeypatch):
    """(client, s3 url) for the artifacts sync; same client configuration as S3BlobStore."""
    if request.param == "s3-moto":
        _moto_env(monkeypatch)
        with mock_aws():
            client = s3_client()
            client.create_bucket(Bucket="artifacts")
            yield client, "s3://artifacts/runs/cell-090"
    else:
        url = os.environ.get("PANOPTES_TEST_S3_URL")
        if not url:
            pytest.skip("PANOPTES_TEST_S3_URL=s3://bucket/prefix required for a real S3-compatible store")
        client, target = s3_client(), url.rstrip("/") + "/artifacts-" + uuid4().hex[:8]
        yield client, target
        bucket, prefix = parse_s3_url(target)
        _delete_prefix(client, bucket, prefix)


def service(repo):
    return PolicyService(policy_repository(repo), repo.blobs)


def png(color, size=(20, 20)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


# ---- repositories -------------------------------------------------------------------------

def test_project_create_idempotent_capability_hash_and_lookup(repo):
    cap = capability()
    body = {"requestId": identity(), "title": "Stable", "target": "scene"}
    created = repo.create_project(cap, body)
    assert repo.create_project(cap, body) == created
    assert cap not in json.dumps(created)
    with pytest.raises(PlatformError, match="idempotency_mismatch"):
        repo.create_project(cap, {**body, "title": "Different"})
    pid = created["project"]["id"]
    assert repo.authorize(pid, cap)["id"] == pid
    with pytest.raises(PlatformError, match="capability_forbidden"):
        repo.authorize(pid, capability())
    assert repo.get_project(pid) == created and created["branches"] == [created["branch"]]
    assert created["branch"]["headRevisionId"] == created["revision"]["id"] and created["revision"]["label"] == "Initial"
    assert created["revision"]["documentSha256"] == digest(created["revision"]["document"])
    assert repo.get_revision(created["revision"]["id"]) == created["revision"]
    assert pid in {item["id"] for item in repo.list_projects()["items"]}
    with pytest.raises(PlatformError, match="project_not_found"):
        repo.get_project(identity())
    with pytest.raises(PlatformError, match="revision_not_found"):
        repo.get_revision(identity())
    with pytest.raises(PlatformError, match="invalid_capability"):
        repo.create_project("not-a-capability", body)


def test_branch_commit_idempotency_base_revision_conflict_and_undo_redo(repo):
    cap, scene = project(repo)
    pid, branch_id = scene["project"]["id"], scene["branch"]["id"]
    body = edit_body(scene)
    committed = repo.commit_edits(pid, cap, body)
    assert repo.commit_edits(pid, cap, body) == committed
    assert committed["headAdvanced"] and committed["editBatch"]["revisionId"] == committed["revision"]["id"]
    assert committed["revision"]["parentRevisionId"] == scene["revision"]["id"]
    with pytest.raises(PlatformError, match="idempotency_mismatch"):
        repo.commit_edits(pid, cap, {**body, "operations": []})
    with pytest.raises(PlatformError) as conflict:
        repo.commit_edits(pid, cap, {**body, "requestId": identity()})
    assert conflict.value.code == "revision_conflict" and conflict.value.status == 409
    assert conflict.value.params == {"baseRevisionId": scene["revision"]["id"], "currentRevisionId": committed["revision"]["id"]}
    assert repo.get_project(pid)["revision"]["id"] == committed["revision"]["id"]
    with pytest.raises(PlatformError, match="capability_forbidden"):
        repo.commit_edits(pid, capability(), body)
    with pytest.raises(PlatformError, match="branch_not_found"):
        repo.commit_edits(pid, cap, {**body, "requestId": identity(), "branchId": identity()})
    undo = repo.commit_edits(pid, cap, {"requestId": identity(), "branchId": branch_id, "baseRevisionId": committed["revision"]["id"], "undoOf": committed["editBatch"]["id"]})
    assert undo["revision"]["document"]["entities"] == [] and undo["editBatch"]["undoOf"] == committed["editBatch"]["id"]
    with pytest.raises(PlatformError, match="undo_conflict"):
        repo.commit_edits(pid, cap, {"requestId": identity(), "branchId": branch_id, "baseRevisionId": undo["revision"]["id"], "undoOf": committed["editBatch"]["id"]})
    redo = repo.commit_edits(pid, cap, {"requestId": identity(), "branchId": branch_id, "baseRevisionId": undo["revision"]["id"], "redoOf": committed["editBatch"]["id"]})
    assert redo["revision"]["document"] == committed["revision"]["document"]
    edits = repo.list_project_records(pid, "edits")["items"]
    assert [item["id"] for item in edits] == [redo["editBatch"]["id"], undo["editBatch"]["id"], committed["editBatch"]["id"]]
    assert len(repo.list_project_records(pid, "revisions")["items"]) == 4
    planning_body = {"requestId": identity(), "sourceRevisionId": redo["revision"]["id"], "title": "Plan", "kind": "planning"}
    planning = repo.create_branch(pid, cap, planning_body)
    assert repo.create_branch(pid, cap, planning_body) == planning
    head = repo.get_revision(planning["headRevisionId"])
    assert head["branchId"] == planning["id"] and head["document"] == redo["revision"]["document"] and head["sourceRevisionId"] == redo["revision"]["id"]
    assert [item["id"] for item in repo.get_project(pid)["branches"]] == [branch_id, planning["id"]]
    planned = repo.commit_edits(pid, cap, {**edit_body(scene), "branchId": planning["id"], "baseRevisionId": planning["headRevisionId"]})
    assert planned["revision"]["branchId"] == planning["id"] and repo.get_project(pid)["revision"]["id"] == redo["revision"]["id"]
    with pytest.raises(ValueError):
        repo.list_project_records(pid, "nope")


def test_assets_register_get_integrity_and_stage_cache(repo):
    cap, scene = project(repo)
    pid = scene["project"]["id"]
    metadata = {**repo.blobs.put(png("red"), "image/png"), "metadata": {"width": 20, "height": 20}}
    asset = repo.register_asset(pid, metadata)
    assert repo.register_asset(pid, metadata) == asset
    assert repo.get_asset(asset["id"]) == asset and asset["projectId"] == pid and asset["jobId"] is None
    assert asset["metadata"] == {"width": 20, "height": 20} and asset["mediaType"] == "image/png"
    with pytest.raises(PlatformError, match="blob_integrity_error"):  # bytes are verified before the registry is consulted
        repo.register_asset(pid, {**metadata, "sizeBytes": metadata["sizeBytes"] + 1})
    with pytest.raises(PlatformError, match="invalid_asset"):
        repo.register_asset(pid, {**metadata, "storageKey": "other/" + metadata["sha256"]})
    missing = hashlib.sha256(b"never stored").hexdigest()
    with pytest.raises(PlatformError, match="blob_not_found"):
        repo.register_asset(pid, {"storageKey": "sha256/" + missing, "sha256": missing, "sizeBytes": 12, "mediaType": "text/plain"})
    with pytest.raises(PlatformError, match="asset_not_found"):
        repo.get_asset(identity())
    assert repo.get_stage_cache(pid, "key-1") is None
    cached = repo.register_asset(pid, {**repo.blobs.put(b'{"stage":1}', "application/json"), "metadata": {"kind": "stage_cache", "cacheKey": "key-1"}})
    assert repo.get_stage_cache(pid, "key-1") == cached
    assert [item["id"] for item in repo.list_project_records(pid, "assets")["items"]] == [cached["id"], asset["id"]]


def test_job_lifecycle_outbox_lease_expiry_recovery_and_cancel(repo):
    cap, scene = project(repo)
    pid = scene["project"]["id"]
    body = {"requestId": identity(), "branchId": scene["branch"]["id"], "baseRevisionId": scene["revision"]["id"], "kind": "reconstruct", "inputs": {}, "config": {"native_setting": 1}}
    job = repo.create_job(pid, cap, body)
    assert repo.create_job(pid, cap, body) == job and job["status"] == "pending_dispatch" and job["config"] == {"native_setting": 1, "providerManifest": {}}
    with pytest.raises(PlatformError, match="admin_job_required"):
        repo.create_job(pid, cap, {**body, "requestId": identity(), "kind": "validate_model"})
    with pytest.raises(PlatformError, match="asset_not_found"):
        repo.create_job(pid, cap, {**body, "requestId": identity(), "inputs": {"assetIds": [identity()]}})
    assert [item["id"] for item in repo.pending_jobs()] == [job["id"]]
    queued = repo.mark_dispatched(job["id"], "executor:1")
    assert queued["status"] == "queued" and queued["executorRef"] == "executor:1" and queued["dispatchedAt"] and not repo.pending_jobs()
    with pytest.raises(ValueError):
        repo.claim_job(job["id"], lease_seconds=0)
    claimed = repo.claim_job(job["id"])
    assert claimed["status"] == "running" and claimed["attempt"] == 1 and claimed["attemptToken"] and claimed["leaseExpiresAt"] > claimed["heartbeatAt"]
    with pytest.raises(PlatformError, match="job_not_claimable"):
        repo.claim_job(job["id"])
    with pytest.raises(PlatformError, match="job_not_found"):
        repo.claim_job(identity())
    assert repo.heartbeat_job(job["id"], claimed["attemptToken"]) and not repo.heartbeat_job(job["id"], identity())
    with pytest.raises(PlatformError, match="invalid_job_outcome"):
        repo.finish_job(job["id"], claimed["attemptToken"], "done")
    finished = repo.finish_job(job["id"], claimed["attemptToken"], "succeeded", document=scene["revision"]["document"], result={"ok": True})
    assert finished["status"] == "succeeded" and finished["headAdvanced"] and finished["result"] == {"ok": True}
    assert repo.get_project(pid)["revision"]["id"] == finished["resultRevisionId"]
    assert repo.get_revision(finished["resultRevisionId"])["label"] == "reconstruct"
    late = repo.finish_job(job["id"], claimed["attemptToken"], "succeeded", document=scene["revision"]["document"])
    assert late["lateResultSaved"] and len(late["lateResults"]) == 1 and repo.get_job(job["id"])["status"] == "succeeded"
    assert repo.finish_job(job["id"], claimed["attemptToken"], "succeeded", document=scene["revision"]["document"])["lateResults"] == late["lateResults"]
    # Lease expiry: the lease is the only clock the recovery sweep looks at.
    expiring = repo.claim_job(make_job(repo, cap, {**scene, "revision": repo.get_revision(finished["resultRevisionId"])})["id"], lease_seconds=1)
    call = repo.reserve_model_call(expiring["id"], expiring["attemptToken"], "test", "model", "input-1", ".1")
    assert repo.recover_expired_jobs() == []
    time.sleep(1.2)
    assert not repo.heartbeat_job(expiring["id"], expiring["attemptToken"])
    assert repo.recover_expired_jobs() == [expiring["id"]] and repo.recover_expired_jobs() == []
    assert repo.get_job(expiring["id"])["status"] == "outcome_unknown"
    with pytest.raises(PlatformError, match="stale_job_attempt"):
        repo.reserve_model_call(expiring["id"], expiring["attemptToken"], "test", "model", "input-2", ".1")
    with pytest.raises(PlatformError, match="model_call_already_reserved"):
        repo.reserve_model_call(expiring["id"], expiring["attemptToken"], "test", "model", "input-1", ".1")
    expired = repo.finish_job(expiring["id"], expiring["attemptToken"], "succeeded", document=scene["revision"]["document"])
    assert expired["lateResultSaved"] and repo.get_job(expiring["id"])["status"] == "outcome_unknown"
    outcome = repo.complete_model_call(call["id"], "succeeded", actual_cost=".9", response={"usage": 42})
    assert outcome["status"] == "outcome_unknown" and outcome["actualCost"] == 0.9 and outcome["response"]["lateOutcome"]["response"]["usage"] == 42
    assert repo.complete_model_call(call["id"], "succeeded", actual_cost=".9", response={"usage": 42}) == outcome
    with pytest.raises(PlatformError, match="model_outcome_conflict"):
        repo.complete_model_call(call["id"], "failed")
    # Cancellation keeps a running attempt's late outcome and ends in cancelled.
    cancelled = make_job(repo, cap, {**scene, "revision": repo.get_revision(finished["resultRevisionId"])})
    running = repo.claim_job(cancelled["id"])
    requested = repo.cancel_job(cancelled["id"], cap)
    assert requested["status"] == "running" and requested["cancelRequested"]
    with pytest.raises(PlatformError, match="capability_forbidden"):
        repo.cancel_job(cancelled["id"], capability())
    assert not repo.heartbeat_job(cancelled["id"], running["attemptToken"])
    result = repo.finish_job(cancelled["id"], running["attemptToken"], "succeeded", document=scene["revision"]["document"])
    assert result["lateResultSaved"] and repo.get_job(cancelled["id"])["status"] == "cancelled"
    pending = make_job(repo, cap, {**scene, "revision": repo.get_revision(finished["resultRevisionId"])})
    assert repo.cancel_job(pending["id"], cap)["status"] == "cancelled"
    with pytest.raises(PlatformError, match="job_not_claimable"):
        repo.claim_job(pending["id"])
    assert [item["id"] for item in repo.list_project_records(pid, "jobs")["items"]] == [pending["id"], cancelled["id"], expiring["id"], job["id"]]


def test_model_call_budget_and_dispatch_receipts(repo):
    cap, scene = project(repo)
    job = repo.claim_job(make_job(repo, cap, scene)["id"])
    with pytest.raises(PlatformError, match="invalid_cost"):
        repo.reserve_model_call(job["id"], job["attemptToken"], "test", "model", "k", "-1")
    call = repo.reserve_model_call(job["id"], job["attemptToken"], "test", "model", "k", ".75", code_sha256="c", model_sha256="m", adapter_sha256="a", input_sha256="i")
    assert call["status"] == "reserved" and call["estimatedCost"] == 0.75 and call["actualCost"] is None and call["codeSha256"] == "c"
    with pytest.raises(PlatformError, match="paid_budget_exceeded"):
        repo.reserve_model_call(job["id"], job["attemptToken"], "test", "model", "k2", ".3")
    free = repo.reserve_model_call(job["id"], job["attemptToken"], "local", "primitive", "free", "0", paid=False)
    assert free["estimatedCost"] == 0
    with pytest.raises(PlatformError, match="invalid_provider_request_id"):
        repo.record_model_call_dispatch(call["id"], job["attemptToken"], " ")
    dispatched = repo.record_model_call_dispatch(call["id"], job["attemptToken"], "req-1")
    assert dispatched["response"] == {"providerRequestId": "req-1"}
    assert repo.record_model_call_dispatch(call["id"], job["attemptToken"], "req-1") == dispatched
    with pytest.raises(PlatformError, match="model_dispatch_conflict"):
        repo.record_model_call_dispatch(call["id"], job["attemptToken"], "req-2")
    with pytest.raises(PlatformError, match="stale_job_attempt"):
        repo.record_model_call_dispatch(call["id"], identity(), "req-1")
    with pytest.raises(PlatformError, match="model_dispatch_conflict"):
        repo.complete_model_call(call["id"], "succeeded", actual_cost=".5", response={"providerRequestId": "req-2"})
    done = repo.complete_model_call(call["id"], "succeeded", actual_cost=".5", response={"usage": 1})
    assert done["status"] == "succeeded" and done["actualCost"] == 0.5 and done["response"] == {"providerRequestId": "req-1", "usage": 1}
    with pytest.raises(PlatformError, match="model_outcome_conflict"):
        repo.complete_model_call(call["id"], "failed")
    assert repo.complete_model_call(call["id"], "succeeded") == done
    with pytest.raises(PlatformError, match="model_call_not_found"):
        repo.complete_model_call(identity(), "succeeded")
    second = repo.reserve_model_call(job["id"], job["attemptToken"], "test", "model", "k3", ".5")
    assert second["status"] == "reserved"
    repo.paid_budget = None
    with pytest.raises(PlatformError, match="paid_budget_not_configured"):
        repo.reserve_model_call(job["id"], job["attemptToken"], "test", "model", "k4", "0")


def test_capture_revision_job_and_publication_snapshot(repo):
    cap, scene = project(repo)
    pid, branch_id = scene["project"]["id"], scene["branch"]["id"]
    images = [{**repo.blobs.put(png(color), "image/png"), "metadata": {"width": 20, "height": 20}} for color in ("red", "blue")]
    body = {"requestId": identity(), "branchId": branch_id, "baseRevisionId": scene["revision"]["id"], "target": "scene"}
    capture = repo.create_capture(pid, cap, body, images)
    assert repo.create_capture(pid, cap, body, images) == capture
    assert capture["job"]["kind"] == "analyze_capture" and capture["job"]["baseRevisionId"] == capture["revision"]["id"]
    assert capture["capture"]["task"]["imageIds"] == [image["assetId"] for image in capture["capture"]["images"]]
    assert len(capture["revision"]["document"]["assets"]) == 2 and repo.get_project(pid)["revision"]["id"] == capture["revision"]["id"]
    with pytest.raises(PlatformError, match="image_count_out_of_range"):
        repo.create_capture(pid, cap, {**body, "requestId": identity(), "baseRevisionId": capture["revision"]["id"]}, [])
    with pytest.raises(PlatformError, match="capture_append_required"):
        repo.create_capture(pid, cap, {**body, "requestId": identity(), "baseRevisionId": capture["revision"]["id"]}, images[:1])
    with pytest.raises(PlatformError, match="revision_conflict"):
        repo.create_capture(pid, cap, {**body, "requestId": identity()}, images[:1])
    assert [item["id"] for item in repo.list_project_records(pid, "captures")["items"]] == [capture["capture"]["id"]]
    publication_body = {"requestId": identity(), "sceneRevisionId": capture["revision"]["id"], "evaluationIds": [], "reviewIds": [], "title": "Frozen"}
    publication = repo.create_publication(pid, cap, publication_body)
    assert repo.create_publication(pid, cap, publication_body) == publication
    with pytest.raises(PlatformError, match="idempotency_mismatch"):
        repo.create_publication(pid, cap, {**publication_body, "title": "Renamed"})
    with pytest.raises(PlatformError, match="revision_not_found"):
        repo.create_publication(pid, cap, {**publication_body, "requestId": identity(), "sceneRevisionId": identity()})
    with pytest.raises(PlatformError, match="publication_evidence_unverified"):
        repo.create_publication(pid, cap, {**publication_body, "requestId": identity(), "evaluationIds": [identity()]})
    snapshot = publication["snapshot"]
    assert snapshot["revision"] == capture["revision"] and snapshot["branchKind"] == "reconstruction" and snapshot["editBatches"] == []
    assert [job["id"] for job in snapshot["jobs"]] == [capture["job"]["id"]] and "attemptToken" not in snapshot["jobs"][0]
    assert sorted(item["assetId"] for item in snapshot["assetManifest"]) == sorted(image["assetId"] for image in capture["capture"]["images"])
    assert repo.get_publication(publication["id"]) == publication
    listing = repo.list_publications()["items"]
    assert listing[0] == {"id": publication["id"], "projectId": pid, "sceneRevisionId": capture["revision"]["id"], "title": "Frozen", "createdAt": publication["createdAt"],
                          "previewImageAssetId": capture["capture"]["images"][0]["assetId"], "photoCount": 2, "objectCount": 0, "spatialObjectCount": 0,
                          "modelObjectCount": 0, "observedSurfaceObjectCount": 0}
    with pytest.raises(PlatformError, match="publication_not_found"):
        repo.get_publication(identity())
    edited = repo.commit_edits(pid, cap, edit_body({**scene, "revision": capture["revision"]}))
    later = repo.create_publication(pid, cap, {**publication_body, "requestId": identity(), "sceneRevisionId": edited["revision"]["id"]})
    assert [batch["id"] for batch in later["snapshot"]["editBatches"]] == [edited["editBatch"]["id"]]
    assert repo.list_publications()["items"][0]["objectCount"] == 1
    assert repo.get_publication(publication["id"]) == publication


def test_policy_create_revision_activate_evaluate_and_review(repo):
    cap, scene = project(repo)
    pid = scene["project"]["id"]
    policies = service(repo)
    assert isinstance(policies.repository, MongoPolicyRepository if isinstance(repo, MongoRepository) else PostgresPolicyRepository)
    template = templates()[0]
    body = {**template, "requestId": identity()}
    created = policies.create_policy(pid, cap, body)
    assert policies.create_policy(pid, cap, body) == created
    with pytest.raises(PlatformError, match="idempotency_mismatch"):
        policies.create_policy(pid, cap, {**body, "title": "Changed"})
    policy_id, first = created["policy"]["id"], created["revision"]["id"]
    assert created["policy"]["draftRevisionId"] == first and created["policy"]["activeRevisionId"] is None
    detail = policies.get_policy(policy_id)
    assert [r["id"] for r in detail["revisions"]] == [first] and detail["sources"][0]["id"] == created["revision"]["sourceId"]
    assert policy_id in {item["id"] for item in policies.list_policies()["items"]}
    edit = {"requestId": identity(), "basePolicyRevisionId": first, "jdm": template["jdm"], "tests": template["tests"], "limitations": ["Edited"], "sourceRefs": []}
    saved = policies.save_revision(pid, policy_id, cap, edit)
    assert policies.save_revision(pid, policy_id, cap, edit) == saved and saved["parentRevisionId"] == first
    assert policies.get_policy(policy_id)["policy"]["draftRevisionId"] == saved["id"]
    with pytest.raises(PlatformError) as stale:
        policies.save_revision(pid, policy_id, cap, {**edit, "requestId": identity()})
    assert stale.value.code == "policy_revision_conflict" and stale.value.params["currentRevisionId"] == saved["id"]
    with pytest.raises(PlatformError, match="policy_activation_conflict"):
        policies.activate(pid, policy_id, cap, {"requestId": identity(), "policyRevisionId": saved["id"], "expectedActiveRevisionId": first})
    activation = {"requestId": identity(), "policyRevisionId": saved["id"], "expectedActiveRevisionId": None}
    activated = policies.activate(pid, policy_id, cap, activation)
    assert activated["active"] and activated["policyRevisionId"] == saved["id"] and all(t["passed"] for t in activated["tests"])
    assert policies.activate(pid, policy_id, cap, activation) == {"policyRevisionId": saved["id"], "active": True}
    with pytest.raises(PlatformError, match="idempotency_mismatch"):
        policies.activate(pid, policy_id, cap, {**activation, "expectedActiveRevisionId": saved["id"]})
    assert policies.get_policy(policy_id)["policy"]["activeRevisionId"] == saved["id"]
    evaluation_body = {"requestId": identity(), "sceneRevisionId": scene["revision"]["id"], "policyRevisionIds": [saved["id"]], "context": "observed"}
    with pytest.raises(PlatformError, match="policy_revision_not_active"):
        policies.evaluate(pid, cap, {**evaluation_body, "policyRevisionIds": [first]})
    with pytest.raises(PlatformError, match="evaluation_context_mismatch"):
        policies.evaluate(pid, cap, {**evaluation_body, "context": "planning"})
    evaluation = policies.evaluate(pid, cap, evaluation_body)
    assert policies.evaluate(pid, cap, evaluation_body) == evaluation
    assert evaluation["document"]["policyRevisionIds"] == [saved["id"]] and evaluation["sceneRevisionId"] == scene["revision"]["id"]
    findings = evaluation["document"]["findings"]
    assert findings and findings[0]["policyTitle"] == template["title"] and findings[0]["policyRevisionId"] == saved["id"]
    review_body = {"requestId": identity(), "evaluationId": evaluation["id"], "decision": "needs_evidence", "reason": "Need source evidence", "displayName": "Reviewer"}
    review = policies.review(pid, findings[0]["id"], cap, review_body)
    assert policies.review(pid, findings[0]["id"], cap, review_body) == review and review["document"]["decision"] == "needs_evidence"
    with pytest.raises(PlatformError, match="review_evidence_invalid"):
        policies.review(pid, findings[0]["id"], cap, {**review_body, "requestId": identity(), "evidenceRefs": [identity()]})
    request = policies.review(pid, findings[0]["id"], cap, {"requestId": identity(), "evaluationId": evaluation["id"], "action": "Attach evidence"}, evidence_request=True)
    assert request["document"]["status"] == "open"
    assert [item["id"] for item in policies.repository.list_evaluations(pid)["items"]] == [evaluation["id"]]
    assert [item["id"] for item in policies.repository.list_reviews(pid)["items"]] == [review["id"]]
    assert [item["id"] for item in policies.repository.list_evidence_requests(pid)["items"]] == [request["id"]]
    evidence = policies.publication_evidence(pid, scene["revision"]["id"], [evaluation["id"]], [review["id"]])
    assert evidence == {"evaluations": [evaluation], "reviews": [review]}
    with pytest.raises(PlatformError, match="publication_evaluation_mismatch"):
        policies.publication_evidence(pid, identity(), [evaluation["id"]], [])
    published = repo.create_publication(pid, cap, {"requestId": identity(), "sceneRevisionId": scene["revision"]["id"], "evaluationIds": [evaluation["id"]], "reviewIds": [review["id"]], "title": "Reviewed"},
                                        evaluations=evidence["evaluations"], reviews=evidence["reviews"])
    assert published["snapshot"]["evaluations"] == [evaluation] and published["evaluationIds"] == [evaluation["id"]]


def test_agent_turn_lifecycle_and_listing(repo):
    cap, scene = project(repo)
    pid = scene["project"]["id"]
    body = {"requestId": identity(), "conversationId": identity(), "branchId": scene["branch"]["id"], "baseRevisionId": scene["revision"]["id"], "message": "Inspect", "language": "en"}
    turn = repo.create_agent_turn(pid, cap, body)
    assert repo.create_agent_turn(pid, cap, body) == turn and turn["status"] == "pending" and turn["request"] == body
    assert not repo.pending_jobs() and repo.get_job(turn["jobId"])["kind"] == "agent_turn"
    with pytest.raises(PlatformError, match="agent_scope_not_found"):
        repo.create_agent_turn(pid, cap, {**body, "requestId": identity(), "entityId": identity()})
    with pytest.raises(PlatformError, match="agent_policy_scope_invalid"):
        repo.create_agent_turn(pid, cap, {**body, "requestId": identity(), "policyId": identity()})
    with pytest.raises(PlatformError, match="agent_image_scope_not_found"):
        repo.create_agent_turn(pid, cap, {**body, "requestId": identity(), "imageId": identity()})
    job = repo.claim_job(turn["jobId"])
    claimed = repo.claim_agent_turn(turn["id"])
    assert claimed["status"] == "running"
    with pytest.raises(PlatformError, match="agent_turn_not_claimable"):
        repo.claim_agent_turn(turn["id"])
    second = repo.create_agent_turn(pid, cap, {**body, "requestId": identity(), "message": "Again"})
    listed = repo.list_agent_turns(pid, conversation_id=body["conversationId"])["items"]
    assert [item["id"] for item in listed] == [turn["id"], second["id"]] and listed[0]["sequence"] < listed[1]["sequence"]
    assert listed[0]["appliedEditBatchId"] is None and listed[0]["appliedPolicyRevisionId"] is None
    assert [item["id"] for item in repo.list_agent_turns(pid, after_sequence=listed[0]["sequence"])["items"]] == [second["id"]]
    assert repo.list_agent_turns(pid, conversation_id=identity())["items"] == []
    added = entity()
    proposal = {"kind": "proposal", "message": "Proposed", "operations": [{"type": "addEntity", "entity": added}]}
    finished = repo.finish_agent_turn(turn["id"], "succeeded", proposal)
    assert finished["response"] == proposal and finished["status"] == "succeeded"
    with pytest.raises(PlatformError, match="agent_outcome_conflict"):
        repo.finish_agent_turn(turn["id"], "failed", {})
    repo.finish_job(job["id"], job["attemptToken"], "succeeded")
    with pytest.raises(PlatformError, match="agent_proposal_mismatch"):
        repo.commit_edits(pid, cap, {**edit_body(scene), "agentTurnId": turn["id"]})
    applied = repo.commit_edits(pid, cap, {**edit_body(scene, proposal["operations"]), "agentTurnId": turn["id"]})
    assert applied["editBatch"]["agentTurnId"] == turn["id"]
    assert repo.list_agent_turns(pid)["items"][0]["appliedEditBatchId"] == applied["editBatch"]["id"]
    with pytest.raises(PlatformError, match="project_not_found"):
        repo.list_agent_turns(identity())


def test_fork_inherits_assets_and_keeps_capabilities_separate(repo):
    cap, scene = project(repo)
    pid = scene["project"]["id"]
    images = [{**repo.blobs.put(png("green"), "image/png"), "metadata": {"width": 20, "height": 20}}]
    capture = repo.create_capture(pid, cap, {"requestId": identity(), "branchId": scene["branch"]["id"], "baseRevisionId": scene["revision"]["id"], "target": "scene"}, images)
    fork_cap = capability()
    fork = repo.fork_project(pid, fork_cap, {"requestId": identity(), "sourceRevisionId": capture["revision"]["id"], "title": "Visitor", "operations": []})
    assert fork["project"]["forkSourceRevisionId"] == capture["revision"]["id"] and fork["revision"]["label"] == "Fork"
    assert fork["revision"]["document"] == capture["revision"]["document"]
    with pytest.raises(PlatformError, match="capability_forbidden"):
        repo.commit_edits(fork["project"]["id"], cap, edit_body(fork))
    edited = repo.commit_edits(fork["project"]["id"], fork_cap, edit_body(fork))
    assert len(edited["revision"]["document"]["assets"]) == 1
    with pytest.raises(PlatformError, match="revision_not_found"):
        repo.fork_project(pid, capability(), {"requestId": identity(), "sourceRevisionId": identity(), "title": "Missing", "operations": []})


# ---- MongoDB-specific checks --------------------------------------------------------------

def test_mongo_collections_prefix_indexes_and_hashed_capability(mongo):
    cap, scene = project(mongo)
    names = set(mongo.db.list_collection_names())
    assert {mongo.prefix + table for table in ("projects", "scene_branches", "scene_revisions", "edit_batches", "jobs")} <= names
    assert all(name.startswith(mongo.prefix) for name in names)
    for table in ("edit_batches", "jobs", "captures", "publications", "agent_turns", "policies", "policy_revisions"):
        index = mongo._c(table).index_information()["project_id_request_id_unique"]
        assert index["unique"] and index["key"] == [("project_id", 1), ("request_id", 1)]
    assert mongo._c("projects").index_information()["capability_sha256_unique"]["unique"]
    stored = mongo._c("projects").find_one({"_id": scene["project"]["id"]})
    assert stored["capability_sha256"] == hashlib.sha256(cap.encode()).hexdigest() and cap not in json.dumps(mongo.get_project(scene["project"]["id"]))
    assert mongo._c("locks").count_documents({}) == 0


def test_mongo_recovers_never_claimed_queued_jobs_and_rejects_research_continuation(mongo, monkeypatch):
    from datetime import timedelta
    cap, scene = project(mongo)
    job = make_job(mongo, cap, scene)
    mongo.mark_dispatched(job["id"], "dead-process")
    assert mongo.recover_expired_jobs() == [] and mongo.get_job(job["id"])["status"] == "queued"
    real_now = mongo._now
    monkeypatch.setattr(mongo, "_now", lambda: real_now() + timedelta(minutes=6))
    assert mongo.recover_expired_jobs() == []
    recovered = mongo.get_job(job["id"])
    assert recovered["status"] == "pending_dispatch" and recovered["executorRef"] is None and recovered["dispatchedAt"] is None
    claimed = mongo.claim_job(job["id"])
    assert claimed["attempt"] == 1
    with pytest.raises(PlatformError) as unsupported:
        mongo.finish_job(job["id"], claimed["attemptToken"], "succeeded", continuation={"kind": "validate_model", "inputs": {}, "config": {}})
    assert unsupported.value.code == "research_authority_unsupported" and unsupported.value.status == 501
    assert mongo.get_job(job["id"])["status"] == "running"


# ---- blob stores --------------------------------------------------------------------------

def test_blob_put_get_head_open_url_and_integrity(blobs, monkeypatch):
    data = b"immutable spatial asset"
    metadata = blobs.put(data, "application/octet-stream")
    sha = hashlib.sha256(data).hexdigest()
    assert metadata == {"storageKey": "sha256/" + sha, "sha256": sha, "sizeBytes": len(data), "mediaType": "application/octet-stream"}
    assert blobs.put(data, "application/octet-stream") == metadata
    assert blobs.put_immutable(metadata["storageKey"], io.BytesIO(data), sha, len(data), "application/octet-stream") == metadata
    assert blobs.get(metadata["storageKey"], sha, len(data)) == data
    head = blobs.head(metadata["storageKey"])
    assert head["sha256"] == sha and head["sizeBytes"] == len(data) and head["storageKey"] == metadata["storageKey"]
    assert blobs.open(metadata["storageKey"]).read() == data
    missing = hashlib.sha256(b"missing").hexdigest()
    assert blobs.head("sha256/" + missing) is None
    with pytest.raises(PlatformError, match="blob_not_found"):
        blobs.get("sha256/" + missing, missing, 7)
    with pytest.raises(PlatformError, match="blob_not_found"):
        blobs.open("sha256/" + missing)
    with pytest.raises(PlatformError, match="blob_integrity_error"):
        blobs.get(metadata["storageKey"], sha, len(data) + 1)
    with pytest.raises(PlatformError, match="blob_integrity_error"):
        blobs.put_immutable(metadata["storageKey"], b"other payload", sha, len(data), "application/octet-stream")
    with pytest.raises(PlatformError, match="invalid_storage_key"):
        blobs.get("../../etc/passwd", sha, len(data))
    url = blobs.url(metadata["storageKey"], "asset-1")
    if isinstance(blobs, S3BlobStore):
        assert url == blobs.public_url(metadata["storageKey"]) and blobs.prefix + metadata["storageKey"] in url and "X-Amz-Signature" in url
        monkeypatch.setenv("PANOPTES_BLOB_PUBLIC_BASE", "https://assets.example.com/")
        assert blobs.url(metadata["storageKey"], "asset-1") == "https://assets.example.com/" + blobs.prefix + metadata["storageKey"]
        monkeypatch.delenv("PANOPTES_BLOB_PUBLIC_BASE")
        monkeypatch.setenv("PANOPTES_S3_URL_EXPIRY_S", "60")
        assert "X-Amz-Expires=60" in blobs.public_url(metadata["storageKey"])
    else:
        assert url == "/api/assets/asset-1/content" and blobs.public_url(metadata["storageKey"]) == "/api/blobs/" + metadata["storageKey"]


def test_blob_large_object_round_trip(blobs):
    data = os.urandom(6 * MB)  # above the 5 MB multipart threshold of the S3 fixtures
    metadata = blobs.put(data, "application/octet-stream")
    assert metadata["sizeBytes"] == len(data) and blobs.get(metadata["storageKey"], metadata["sha256"], len(data)) == data
    assert blobs.head(metadata["storageKey"])["sizeBytes"] == len(data)


# ---- artifacts sync -----------------------------------------------------------------------

def test_artifacts_push_pull_round_trip_skips_unchanged_and_verifies(artifact_bucket, tmp_path):
    from scripts import panoptes_artifacts as artifacts
    client, url = artifact_bucket
    source = tmp_path / "run"
    (source / "frames").mkdir(parents=True)
    (source / "ledger.json").write_text('{"step": 1}')
    (source / "frames" / "a.bin").write_bytes(os.urandom(1024))
    (source / "frames" / "b.bin").write_bytes(b"b" * 2048)
    assert artifacts.push(client, source, url) == {"files": 3, "uploaded": 3, "skipped": 0}
    assert artifacts.push(client, source, url) == {"files": 3, "uploaded": 0, "skipped": 3}
    (source / "ledger.json").write_text('{"step": 2}')
    assert artifacts.push(client, source, url) == {"files": 3, "uploaded": 1, "skipped": 2}
    bucket, prefix = parse_s3_url(url)
    manifest = json.loads(client.get_object(Bucket=bucket, Key=prefix + "manifest.json")["Body"].read())
    assert set(manifest) == {"ledger.json", "frames/a.bin", "frames/b.bin"} and manifest["frames/b.bin"]["size"] == 2048
    target = tmp_path / "pulled"
    assert artifacts.pull(client, url, target) == {"files": 3, "downloaded": 3, "skipped": 0}
    assert artifacts.local_manifest(target) == artifacts.local_manifest(source) == {k: v for k, v in manifest.items()}
    assert artifacts.pull(client, url, target) == {"files": 3, "downloaded": 0, "skipped": 3}
    client.put_object(Bucket=bucket, Key=prefix + "frames/b.bin", Body=b"corrupt")
    (target / "frames" / "b.bin").unlink()
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        artifacts.pull(client, url, target)
    assert not (target / "frames" / "b.bin").exists()
    with pytest.raises(SystemExit, match="no manifest.json"):
        artifacts.pull(client, url + "-missing", tmp_path / "nothing")


def test_offline_import_and_api_read_parity(repo, tmp_path):
    """The import script, the API read routes and the policy/agent routes agree with the repository on every backend."""
    from fastapi.testclient import TestClient
    from ehs_spatial.platform.api import _public, create_app
    from ehs_spatial.platform.contracts import canonical
    from scripts.import_public_scene import run_import
    from test_platform_import import make_public_scene
    source = tmp_path / "source"
    source.mkdir()
    blobs = LocalBlobStore(tmp_path / "blobs")
    imported = run_import(make_public_scene(source), repo, blobs, tmp_path / "imports")
    assert run_import(make_public_scene(source), repo, blobs, tmp_path / "imports") == imported
    policies = PolicyService(policy_repository(repo), blobs)
    client = TestClient(create_app(repository=repo, blobs=blobs, policy_service=policies))
    assert client.get("/api/health").json() == {"status": "ok", "schemaVersion": 1}
    pid, rid = imported["projectId"], imported["sceneRevisionId"]
    for url, expected in [(f"/api/projects/{pid}", repo.get_project(pid)), (f"/api/revisions/{rid}", repo.get_revision(rid)),
                          (f"/api/publications/{imported['publicationId']}", repo.get_publication(imported["publicationId"])),
                          ("/api/publications", repo.list_publications()), ("/api/projects", repo.list_projects())]:
        response = client.get(url)
        assert response.status_code == 200 and canonical(response.json()) == canonical(expected)
    for kind in ("revisions", "captures", "assets", "jobs", "edits"):
        response = client.get(f"/api/projects/{pid}/{kind}")
        assert response.status_code == 200 and canonical(response.json()) == canonical(_public(repo.list_project_records(pid, kind)))
    scene = repo.get_revision(rid)
    assert scene["document"]["captureId"] == repo.list_project_records(pid, "captures")["items"][0]["id"]
    asset = repo.get_asset(scene["document"]["assets"][0]["id"])
    assert client.get(f"/api/assets/{asset['id']}").json() == {**asset, "url": blobs.url(asset["storageKey"], asset["id"])}
    assert client.get(f"/api/assets/{asset['id']}/content").content == blobs.get(asset["storageKey"], asset["sha256"], asset["sizeBytes"])
    assert len(repo.get_publication(imported["publicationId"])["snapshot"]["assetManifest"]) == len(scene["document"]["assets"])
    cap, created = project(repo)
    headers = {"Authorization": "Capability " + cap}
    policy = client.post(f"/api/projects/{created['project']['id']}/policies", json={**templates()[0], "requestId": identity()}, headers=headers)
    assert policy.status_code == 200
    assert client.get(f"/api/policies/{policy.json()['policy']['id']}").json() == policies.get_policy(policy.json()["policy"]["id"])
    body = {"requestId": identity(), "conversationId": identity(), "branchId": created["branch"]["id"], "baseRevisionId": created["revision"]["id"], "message": "What evidence is missing?"}
    turn = repo.create_agent_turn(created["project"]["id"], cap, body)
    repo.claim_agent_turn(turn["id"])
    repo.finish_agent_turn(turn["id"], "succeeded", {"kind": "answer", "message": "Source photo required.", "provider_evidence": {"frame_id": "original"}})
    response = client.get(f"/api/projects/{created['project']['id']}/agent-turns")
    assert response.status_code == 200 and canonical(response.json()) == canonical(_public(repo.list_agent_turns(created["project"]["id"])))


# ---- selection by URL ---------------------------------------------------------------------

def test_services_select_backends_from_urls(monkeypatch, tmp_path):
    _moto_env(monkeypatch)
    monkeypatch.setenv("PANOPTES_DATABASE_URL", "mongodb://localhost:27017/panoptes_contract")
    monkeypatch.setenv("PANOPTES_BLOB_ROOT", "s3://contract-bucket/blobs")
    monkeypatch.setenv("PANOPTES_MONGO_PREFIX", "wp1_")
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="contract-bucket")
        repository, blobs = services()
        assert isinstance(repository, MongoRepository) and repository.prefix == "wp1_" and repository.db.name == "panoptes_contract"
        assert isinstance(blobs, S3BlobStore) and (blobs.bucket, blobs.prefix) == ("contract-bucket", "blobs/") and repository.blobs is blobs
        assert isinstance(policy_repository(repository), MongoPolicyRepository)
        assert blobs.put(b"x")["storageKey"].startswith("sha256/")
    monkeypatch.setenv("PANOPTES_DATABASE_URL", "postgresql://panoptes@localhost/panoptes")
    monkeypatch.setenv("PANOPTES_BLOB_ROOT", str(tmp_path / "blobs"))
    config = PlatformConfig.from_env()
    assert (config.database_backend, config.blob_backend) == ("postgres", "local")
    repository, blobs = services(config)
    assert isinstance(repository, PostgresRepository) and isinstance(blobs, LocalBlobStore) and isinstance(policy_repository(repository), PostgresPolicyRepository)
    assert isinstance(repository_from_url("mongodb://user:secret@db.internal:27017/panoptes"), MongoRepository)  # mongodb+srv:// resolves DNS at construction
    assert isinstance(blob_store("s3://bucket"), S3BlobStore) and blob_store("s3://bucket").prefix == ""
    monkeypatch.setenv("PANOPTES_BLOB_BACKEND", "s3")
    with pytest.raises(ValueError, match="PANOPTES_BLOB_ROOT=s3://"):
        PlatformConfig.from_env()
    monkeypatch.delenv("PANOPTES_BLOB_BACKEND")
    monkeypatch.setenv("PANOPTES_DATABASE_URL", "mysql://localhost/panoptes")
    with pytest.raises(ValueError, match="mongodb"):
        PlatformConfig.from_env()
    with pytest.raises(ValueError):
        parse_s3_url("s3://")


# ---- the PostgreSQL backend suite, replayed against MongoDB ----------------------------------

PORTED = ["test_v2_identity_migration_merge_cas_undo_and_frozen_publication", "test_fork_independent_capability_publication_is_snapshot",
          "test_durable_outbox_cancel_old_attempt_and_late_branch_result", "test_model_reservation_budget_and_unknown_outcome_never_replayed",
          "test_failed_provider_usage_counts_actual_cost_before_next_reservation", "test_upload_api_real_image_exif_idempotency_blob_integrity",
          "test_offline_import_retries_and_missing_blob_cannot_commit", "test_policy_service_real_database_unknown_applicability_and_typed_invalid",
          "test_identity_agent_scope_and_evidence_fulfillment_keep_historical_evaluation"]
CONCURRENT = ["test_branch_cas_cross_project_ownership_and_planning_independence", "test_append_capture_preserves_scene_deduplicates_sources_and_fences_concurrent_jobs"]


def _replay(name, repo, tmp_path, monkeypatch):
    if isinstance(repo, PostgresRepository):
        pytest.skip("already covered by test_platform_backend.py")
    monkeypatch.setattr(policy_repository_module, "PostgresPolicyRepository", policy_repository)  # the legacy tests import it inside the function
    case = getattr(legacy, name)
    available = {"repo": repo, "tmp_path": tmp_path, "monkeypatch": monkeypatch}
    case(**{key: available[key] for key in inspect.signature(case).parameters})


@pytest.mark.parametrize("name", PORTED)
def test_backend_suite_replayed(name, repo, tmp_path, monkeypatch):
    _replay(name, repo, tmp_path, monkeypatch)


@pytest.mark.parametrize("name", CONCURRENT)
def test_backend_suite_replayed_concurrently(name, repo, tmp_path, monkeypatch):
    if getattr(repo.client, "__module__", "").startswith("mongomock"):
        pytest.skip("mongomock is not thread-safe; runs against a real server")
    _replay(name, repo, tmp_path, monkeypatch)


def test_concurrent_identical_commits_resolve_to_one_revision(repo):
    if isinstance(repo, MongoRepository) and repo.client.__module__.startswith("mongomock"):
        pytest.skip("mongomock is not thread-safe; runs against a real server")
    cap, scene = project(repo)
    pid, body = scene["project"]["id"], edit_body(scene)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: repo.commit_edits(pid, cap, body), range(4)))
    assert all(result == results[0] for result in results)
    assert len(repo.list_project_records(pid, "edits")["items"]) == 1
