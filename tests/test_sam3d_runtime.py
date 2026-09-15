"""Exercise the actual SAM3D loader and error boundary without GPU/model calls."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_platform_backend import repo


def load_worker(monkeypatch, tmp_path, *, stub_distribution=True):
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps({"generation": {
        "runtimeImage": "registry.example/sam3d@sha256:" + "a" * 64,
        "pins": {"model": "facebook/sam-3d-objects", "codeRevision": "b" * 40, "modelRevision": "c" * 40},
        "distribution": "sam3d_objects", "checkpointConfig": "checkpoints/pipeline.yaml"}}))
    monkeypatch.setenv("PANOPTES_MODEL_RUNTIME_MANIFEST", str(manifest))
    decorator = lambda **kwargs: lambda value: value
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(
        App=lambda name: SimpleNamespace(cls=decorator), enter=decorator, method=decorator,
        Image=SimpleNamespace(from_registry=lambda name: SimpleNamespace(env=lambda values: None)),
        Secret=SimpleNamespace(from_name=lambda name: name)))
    spec = importlib.util.spec_from_file_location("sam3d_worker_test", Path(__file__).parents[1] / "modal_apps/platform_models.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    if stub_distribution:
        monkeypatch.setattr(worker, "_verify_distribution", lambda stage: None)
    return worker


def test_loader_removes_depth_and_gaussian_factories_before_hydra_recurses(monkeypatch, tmp_path):
    worker = load_worker(monkeypatch, tmp_path)
    constructed = []

    def depth_factory():
        constructed.append(True)
        raise AssertionError("An internal depth model would be constructed/downloaded")

    settings = SimpleNamespace(depth_model=depth_factory)

    def instantiate(config, **overrides):
        # Hydra resolves overrides before recursively constructing nested targets.
        depth = overrides.get("depth_model", config.depth_model)
        for key in ('slat_decoder_gs_config_path', 'slat_decoder_gs_ckpt_path',
                    'slat_decoder_gs_4_config_path', 'slat_decoder_gs_4_ckpt_path'):
            if overrides.get(key, 'configured upstream decoder') is not None:
                depth_factory()
        assert overrides['decode_formats'] == ['mesh']
        return SimpleNamespace(depth_model=depth() if callable(depth) else depth)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **k: str(tmp_path)))
    monkeypatch.setitem(sys.modules, "omegaconf", SimpleNamespace(OmegaConf=SimpleNamespace(load=lambda path: settings)))
    monkeypatch.setitem(sys.modules, "hydra.utils", SimpleNamespace(instantiate=instantiate))
    instance = worker.SAM3DObjects()
    instance.load()
    assert constructed == [] and instance.depth_calls == 0
    assert settings.rendering_engine == "pytorch3d" and settings.compile_model is False
    with pytest.raises(RuntimeError, match="External pointmap required"):
        instance.pipeline.depth_model()
    assert instance.depth_calls == 1
    with pytest.raises(AssertionError, match="constructed/downloaded"):
        instantiate(settings)  # Negative control: original construction fails.


def test_external_pointmap_is_preserved_and_failure_returns_telemetry(monkeypatch, tmp_path):
    worker = load_worker(monkeypatch, tmp_path)

    class Event:
        def __init__(self, **kwargs): pass
        def record(self): pass
        def elapsed_time(self, end): return 1250

    class Tensor:
        def __init__(self, data): self.data = data
        def float(self): return self
        def cuda(self): return self

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        from_numpy=Tensor, cuda=SimpleNamespace(Event=Event, synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, "pytorch3d.transforms", SimpleNamespace(quaternion_to_matrix=None))
    monkeypatch.setitem(sys.modules, "sam3d_objects.data.dataset.tdfy.transforms_3d", SimpleNamespace(compose_transform=None))
    received = {}

    def run(image, mask, **kwargs):
        received.update(image=image, mask=mask, **kwargs)
        raise RuntimeError("GPU inference failed after receiving the pointmap")

    instance = worker.SAM3DObjects()
    instance.depth_calls = 0
    instance.pipeline = SimpleNamespace(run=run)
    pointmap = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 10
    payload = {"image": np.full((2, 3, 3), 123, np.uint8), "mask": np.ones((2, 3), bool),
        "pointmap": pointmap, "seed": 7, "decode_formats": ["mesh"], "with_mesh_postprocess": False,
        "with_texture_baking": False, "with_layout_postprocess": False, "use_vertex_color": True}
    result = instance.run(payload)
    np.testing.assert_array_equal(received["pointmap"].data, pointmap)
    np.testing.assert_array_equal(received["image"][..., :3], payload["image"])
    assert received["seed"] == 7 and received["estimate_plane"] is False
    assert received["decode_formats"] == ["mesh"] and instance.depth_calls == 0
    assert result["providerError"]["code"] == "gpu_inference_failed"
    assert result["telemetry"]["gpuElapsedSeconds"] == 1.25
    assert result["telemetry"]["workerElapsedSeconds"] >= 0
    assert result["telemetry"]["actualCostUsd"] is None


def test_runtime_rejects_changed_mesh_source_or_receipt(monkeypatch, tmp_path):
    import hashlib
    worker = load_worker(monkeypatch, tmp_path, stub_distribution=False)
    config = worker.CONFIG['generation']
    relative = 'sam3d_objects/pipeline/inference_pipeline.py'
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_text('reviewed test source\n')
    build = {'codeRevision': config['pins']['codeRevision'], 'files': {
        relative: {'patchedSha256': hashlib.sha256(source.read_bytes()).hexdigest()}}}
    receipt = tmp_path / 'sam3d_objects/panoptes_mesh_build.json'
    receipt.write_text(json.dumps(build))
    config['meshSourceBuildSha256'] = hashlib.sha256(receipt.read_bytes()).hexdigest()
    monkeypatch.setattr(worker.importlib.metadata, 'distribution', lambda name: SimpleNamespace(
        read_text=lambda name: json.dumps({'vcs_info': {'commit_id': config['pins']['codeRevision']}}),
        locate_file=lambda relative: tmp_path / relative))
    worker._verify_distribution('generation')
    source.write_text('unexpected source edit\n')
    with pytest.raises(RuntimeError, match='patched source differs'):
        worker._verify_distribution('generation')
    receipt.write_text('{}')
    with pytest.raises(RuntimeError, match='source build differs'):
        worker._verify_distribution('generation')


def test_remote_research_runtime_mismatch_rejects_before_pipeline_imports(monkeypatch, tmp_path):
    worker = load_worker(monkeypatch, tmp_path)
    instance = worker.SAM3DObjects()
    # No torch/pytorch3d/pipeline setup: the body must reject the frozen pin first.
    with pytest.raises(ValueError, match="Frozen runtime manifest differs"):
        instance.run.__wrapped__(instance, {}, expectedRuntimeManifestSha256="0"*64)


@pytest.mark.parametrize("failure", [None, "license", "runtime_for_quality", "purpose", "payload", "runtime_image", "source_audit", "budget", "insufficient_budget", "public_kind"])
def test_research_worker_enforces_frozen_prerequisites_and_never_writes_scene(monkeypatch, tmp_path, failure):
    from copy import deepcopy
    from ehs_spatial.platform.contracts import PlatformError, digest
    from ehs_spatial.platform.reconstruction import ProviderSpec
    from ehs_spatial.platform.storage import LocalBlobStore
    from panoptes_worker.__main__ import run_job
    from test_platform_reconstruction import Repo
    from test_sam3d_preflight import research_configuration

    class WorkerRepo(Repo):
        def claim_job(self, identity):
            if self.job.get("status") != "queued":
                raise PlatformError("job_not_claimable", 409)
            self.job["status"] = "running"
            return deepcopy(self.job)
        def get_job(self, identity): return deepcopy(self.job)
        def heartbeat_job(self, *args): return True
        def finish_job(self, identity, token, status, document=None, result=None):
            assert document is None, "Research must never save a scene revision"
            assert status in ("succeeded", "failed", "outcome_unknown")
            self.job.update(status=status, result=result, headAdvanced=False, resultRevisionId=None)
            return deepcopy(self.job)

    blobs = LocalBlobStore(tmp_path)
    repo = WorkerRepo(blobs)
    runtime, manifest, protocol = research_configuration()
    image = repo.capture["images"][0]
    image = {**image,"sha256":repo.get_asset(image["assetId"])["sha256"]}
    payload = {"entityId":protocol["entityId"]}
    protocol.update(baselineRevision=repo.rid, inputHashes=[image["sha256"]], payloadSha256=digest(payload),
                    inputAssetHashes=[{"assetId":image["assetId"],"sha256":image["sha256"]}])
    if failure == "license": manifest["generation"]["releaseEvidence"]["license"] = {"status":"unverified"}
    if failure == "runtime_for_quality": protocol["purpose"] = "quality_validation"
    if failure == "purpose": protocol["purpose"] = "product"
    if failure == "payload": payload["changedAfterFreeze"] = True
    if failure == "runtime_image": runtime["generation"]["runtimeImage"] = "mutable:latest"
    if failure == "source_audit": manifest["generation"]["nativePoseEvidence"]["meshOnlyDependencyAudit"] = {"status":"unverified"}
    protocol["providerManifestSha256"] = digest(manifest)
    repo.paid_budget = None if failure == "budget" else .009 if failure == "insufficient_budget" else .01
    frozen = {"schemaVersion":1,"projectId":repo.pid,"branchId":"test-branch","baseRevisionId":repo.rid,
        "baseDocumentSha256":digest(repo.document),"authority":{"source":"database_admin"},
        "protocol":protocol,"payload":payload,"images":[image],"providerManifest":manifest}
    from ehs_spatial.platform.contracts import canonical
    asset = repo.register_asset(repo.pid,{**blobs.put(canonical(frozen),"application/json"),"metadata":{"kind":"sam3d_validation_input"}})
    repo.job.update(kind="generate_object" if failure == "public_kind" else "validate_model", branchId="test-branch", status="queued",
        inputs={"validationAssetId":asset["id"],"validationSha256":asset["sha256"]}, config={"researchProtocolSha256":digest(protocol)})
    calls = []
    config = manifest["generation"]
    provider = ProviderSpec("test-only",config["pins"],lambda p: calls.append(p) or {"vertices":[[0,0,0]],"telemetry":{"actualCostUsd":None}},.01,config["releaseEvidence"])
    monkeypatch.setattr("ehs_spatial.platform.reconstruction.providers_from_manifest",lambda *a,**k:{"generation":provider})
    before = deepcopy(repo.document)
    if failure == "public_kind":
        from ehs_spatial.platform.reconstruction import run_research_job
        with pytest.raises(PlatformError,match="admin_research_job_required"):
            run_research_job(repo,blobs,repo.job)
    else:
        result = run_job(repo,blobs,repo.jid)
        assert result["status"] == ("failed" if failure else "succeeded")
        assert result["result"]["scope"] == "research_only" and result["resultRevisionId"] is None and not result["headAdvanced"]
        assert "output" not in result["result"]
        assert run_job(repo,blobs,repo.jid) == result
    assert len(calls) == len(repo.calls) == (0 if failure else 1)
    assert before == repo.document


def test_unknown_research_call_cannot_charge_again(monkeypatch, tmp_path):
    from ehs_spatial.platform.contracts import PlatformError, digest
    from ehs_spatial.platform.reconstruction import ProviderSpec, run_research_stage
    from ehs_spatial.platform.storage import LocalBlobStore
    from test_platform_reconstruction import Repo
    from test_sam3d_preflight import research_configuration
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    _, manifest, protocol = research_configuration()
    image = repo.capture["images"][0]
    image = {**image,"sha256":repo.get_asset(image["assetId"])["sha256"]}
    payload = {"entityId":protocol["entityId"]}
    protocol.update(baselineRevision=repo.rid,inputHashes=[image["sha256"]],payloadSha256=digest(payload))
    job = {**repo.job,"kind":"validate_model","config":{"researchProtocolSha256":digest(protocol)}}
    repo.paid_budget = 1
    calls = []
    def lost(_):
        calls.append(True)
        raise TimeoutError("credential-bearing endpoint must not enter result")
    config = manifest["generation"]
    provider = ProviderSpec("test-only",config["pins"],lost,.01,config["releaseEvidence"])
    monkeypatch.setattr("ehs_spatial.platform.reconstruction.providers_from_manifest",lambda *a,**k:{"generation":provider})
    for _ in range(2):
        with pytest.raises(PlatformError,match="provider_outcome_unknown"):
            run_research_stage(repo,blobs,job,"generation",payload,[image],manifest,protocol)
    assert len(calls) == len(repo.calls) == 1 and repo.calls[0]["status"] == "outcome_unknown"
    assert "credential-bearing" not in str(repo.calls)


@pytest.mark.parametrize("invalid_pose", [False, True])
def test_research_adapter_sends_frozen_runtime_digest_and_retains_received_evidence(monkeypatch, invalid_pose):
    from ehs_spatial.platform.contracts import digest
    from ehs_spatial.platform.reconstruction import providers_from_manifest
    from test_sam3d_preflight import research_configuration
    runtime, manifest, protocol = research_configuration()
    requests = []
    vertices = np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    def remote(payload, **options):
        requests.append({**payload, **options})
        return {"pins":manifest["generation"]["pins"],"runtimeManifestSha256":digest(runtime["generation"]),
            "internalDepthCalls":0,"decodeFormats":["mesh"],"vertices":vertices,"faces":[[0,1,2]],
            "officialPosedVertices":vertices + (1 if invalid_pose else 0),"objectToProvider":np.eye(4)}
    monkeypatch.setitem(sys.modules,"modal",SimpleNamespace(Cls=SimpleNamespace(from_name=lambda *a: lambda: SimpleNamespace(run=SimpleNamespace(remote=remote)))))
    payload = {"entityId":protocol["entityId"],"imageId":"image","coordinateFrameId":"frame","imageSha256":"f"*64,
        "image":np.ones((3,3,3),np.uint8),"mask":np.ones((3,3),bool),"points":np.ones((3,3,3)),
        "valid":np.ones((3,3),bool),"K":np.eye(3),"cameraToWorld":np.eye(4),"seed":0,"_researchProtocol":protocol}
    output = providers_from_manifest(manifest,_research=True)["generation"].invoke(payload)
    assert requests[0]["expectedRuntimeManifestSha256"] == digest(runtime["generation"])
    np.testing.assert_array_equal(output["runtimeEvidence"]["officialPosedVertices"],vertices + (1 if invalid_pose else 0))
    assert output["runtimeEvidence"]["internalDepthCalls"] == 0 and output["runtimeEvidence"]["decodeFormats"] == ["mesh"]
    if invalid_pose:
        assert output["providerError"]["code"] == "research_response_invalid"
    else:
        assert output["provenance"]["shapeStatus"] == "research_only"
        assert output["provenance"]["releaseEvidenceSha256"] is None


def test_public_research_kind_is_rejected_before_auth_or_idempotency(monkeypatch):
    from ehs_spatial.platform.contracts import PlatformError
    from ehs_spatial.platform.postgres import PostgresRepository
    repo = PostgresRepository("unused")
    monkeypatch.setattr(repo,"_connect",lambda: pytest.fail("Public research kind reached database/idempotency"))
    with pytest.raises(PlatformError,match="admin_job_required"):
        repo.create_job("project","capability",{"kind":"validate_model","requestId":"already-existing-admin-request"})


def test_prepare_freezes_native_inputs_without_database_writes_or_reservations(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from ehs_spatial.platform.contracts import digest
    from ehs_spatial.platform.reconstruction import run_analysis, _unpacked
    from ehs_spatial.platform.storage import LocalBlobStore
    from scripts.research import validate_sam3d as cli
    from test_platform_reconstruction import Repo, bundle
    from test_sam3d_preflight import research_configuration
    blobs = LocalBlobStore(tmp_path)
    repository = Repo(blobs)
    repository.document, _ = run_analysis(repository, blobs, repository.job, bundle(repository))
    for asset in repository.document["assets"]:
        asset.pop("storageKey", None)  # Normal scene references do not expose the blob storage key.
    monkeypatch.setattr(repository, "_connect", lambda: nullcontext(None), raising=False)
    monkeypatch.setattr(cli, "admin_context", lambda *a: ({"source":"database_admin"},{"document":repository.document}))
    monkeypatch.setattr(cli, "check_budget", lambda *a: {"configuredBudgetUsd":"1"})
    runtime, manifest, sample = research_configuration()
    protocol = {k:sample[k] for k in ("id","purpose","metricDefinitions","policyThresholds","split","callLimits")}
    protocol.update(projectId=repository.pid,branchId="test-branch",baselineRevision=repository.rid,
                    entityId=repository.document["entities"][0]["id"])
    before = (len(repository.calls),len(repository.assets),digest(repository.document))
    frozen = cli.prepare(protocol,repository,blobs,manifest,runtime)
    assert before == (len(repository.calls),len(repository.assets),digest(repository.document))
    assert frozen["baseDocumentSha256"] == digest(repository.document)
    assert digest(frozen["payload"]) == frozen["protocol"]["payloadSha256"]
    assert _unpacked(frozen["payload"])["points"].shape == (12,12,3)
    assert len(frozen["images"]) == 1 and len(frozen["protocol"]["inputAssetHashes"]) >= 3


@pytest.mark.parametrize("outcome", ["success", "unknown", "cancelled", "expired"])
def test_admin_submit_and_worker_use_real_sql_idempotency_budget_and_fencing(repo, tmp_path, monkeypatch, outcome):
    from copy import deepcopy
    from decimal import Decimal
    from ehs_spatial.platform.contracts import PlatformError, digest
    from ehs_spatial.platform.reconstruction import ProviderSpec
    from ehs_spatial.platform.storage import LocalBlobStore
    from scripts.research.validate_sam3d import submit
    from panoptes_worker.__main__ import run_job
    from test_platform_backend import project
    from test_sam3d_preflight import research_configuration
    cap, scene = project(repo)
    blobs = LocalBlobStore(tmp_path)
    repo.blobs = blobs
    pid, bid = scene["project"]["id"], scene["branch"]["id"]
    source = deepcopy(scene["revision"]["document"])
    asset = repo.register_asset(pid,blobs.put(b"frozen-source-image","image/png"))
    source["assets"].append(asset)
    with repo._connect() as connection:
        revision = repo._insert_revision(connection,pid,bid,source,parent=scene["revision"]["id"])
    base_id = str(revision["id"])
    _, manifest, protocol = research_configuration()
    payload = {"entityId":protocol["entityId"]}
    protocol.update(baselineRevision=base_id,inputHashes=[asset["sha256"]],payloadSha256=digest(payload),
                    inputAssetHashes=[{"assetId":asset["id"],"sha256":asset["sha256"]}])
    frozen = {"schemaVersion":1,"projectId":pid,"branchId":bid,"baseRevisionId":base_id,
        "baseDocumentSha256":digest(source),"authority":{"source":"database_admin"},"protocol":protocol,
        "providerManifest":manifest,"payload":payload,"images":[{"id":asset["id"],"sha256":asset["sha256"]}]}
    prepared = {"validation":frozen,"sha256":digest(frozen)}
    for budget, code in ((None,"paid_budget_not_configured"),(Decimal(".009"),"paid_budget_exceeded")):
        repo.paid_budget = budget
        with pytest.raises(PlatformError,match=code): submit(prepared,repo,blobs)
    repo.paid_budget = Decimal("1")
    job = submit(prepared,repo,blobs)
    assert submit(prepared,repo,blobs)["id"] == job["id"]
    changed = deepcopy(frozen)
    changed["authority"]["note"] = "different reviewed envelope"
    with pytest.raises(PlatformError,match="research_idempotency_mismatch"):
        submit({"validation":changed,"sha256":digest(changed)},repo,blobs)
    calls = []
    def invoke(value):
        calls.append(value)
        if outcome == "unknown": raise TimeoutError("private provider endpoint")
        if outcome == "cancelled": repo.cancel_job(job["id"],cap)
        if outcome == "expired":
            with repo._connect() as connection:
                connection.execute("UPDATE jobs SET lease_expires_at=now()-interval '1 second' WHERE id=%s",(job["id"],))
        return {"vertices":[[0,0,0]],"telemetry":{"actualCostUsd":None}}
    config = manifest["generation"]
    provider = ProviderSpec("test-only",config["pins"],invoke,.01,config["releaseEvidence"])
    monkeypatch.setattr("ehs_spatial.platform.reconstruction.providers_from_manifest",lambda *a,**k:{"generation":provider})
    result = run_job(repo,blobs,job["id"])
    if outcome == "expired":
        assert result["lateResultSaved"]
        repo.recover_expired_jobs()
        result = repo.get_job(job["id"])
    assert result["status"] == {"success":"succeeded","unknown":"outcome_unknown","cancelled":"cancelled","expired":"outcome_unknown"}[outcome]
    assert not result["headAdvanced"] and result["resultRevisionId"] is None
    assert repo.get_project(pid)["branches"][0]["headRevisionId"] == scene["revision"]["id"]
    assert submit(prepared,repo,blobs)["id"] == job["id"]
    run_job(repo,blobs,job["id"])
    assert len(calls) == 1
    with repo._connect() as connection:
        assert connection.execute("SELECT count(*) AS count FROM model_calls").fetchone()["count"] == 1
