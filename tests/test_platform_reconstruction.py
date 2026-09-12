from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from uuid import uuid4

import numpy as np
from PIL import Image
import pytest

from ehs_spatial.platform.contracts import PlatformError,empty_document,validate_document
from ehs_spatial.platform.reconstruction import (
    MAP_PINS,ProviderSpec,providers_from_env,run_analysis,run_generation,run_segmentation,
    provider_snapshot_from_env,providers_from_manifest,run_research_stage,_Stages,ProviderResponseError,
)
from ehs_spatial.platform.storage import LocalBlobStore


class Repo:
    def __init__(self,blobs,size=12):
        self.assets,self.calls,self.events = [],[],[]
        self.pid,self.cid,self.rid,self.jid = [str(uuid4()) for _ in range(4)]
        self.document = empty_document()
        self.document.update(captureId=self.cid,target="standalone_object")
        images = []
        for i in range(2):
            raw = io.BytesIO()
            Image.fromarray(np.full((size,size,3),40+i,np.uint8)).save(raw,format="PNG")
            asset = self.register_asset(self.pid,blobs.put(raw.getvalue(),"image/png"))
            images.append({"id":asset["id"],"assetId":asset["id"],"width":size,"height":size,"pixelMapping":[]})
            self.document["assets"].append(asset)
        self.capture = {"id":self.cid,"images":images,"target":"standalone_object"}
        self.job = {"id":self.jid,"projectId":self.pid,"baseRevisionId":self.rid,"inputs":{"captureId":self.cid},"attemptToken":str(uuid4()),"kind":"analyze_capture","config":{}}
        self.paid_budget = None

    def register_asset(self,project_id,metadata,job_id=None):
        prior = next((a for a in self.assets if a["sha256"] == metadata["sha256"]),None)
        if prior:
            return deepcopy(prior)
        asset = {"id":str(uuid4()),"projectId":project_id,**metadata}
        self.assets.append(asset)
        self.events.append(("asset",metadata.get("metadata",{}).get("kind")))
        return deepcopy(asset)
    def list_project_records(self,pid,kind):
        return {"items":deepcopy(self.assets if kind == "assets" else [self.capture])}
    def get_revision(self,rid):
        return {"id":rid,"document":deepcopy(self.document)}
    def get_asset(self,aid):
        return deepcopy(next(a for a in self.assets if a["id"] == aid))
    def reserve_model_call(self,*args,**kwargs):
        if kwargs.get("paid") and self.paid_budget is None:
            raise PlatformError("paid_budget_not_configured",409)
        if any(x["key"] == args[4] for x in self.calls):
            raise PlatformError("model_call_already_reserved",409)
        call = {"id":str(uuid4()),"key":args[4],"status":"reserved"}
        self.calls.append(call)
        self.events.append(("reserve",args[2]))
        return call
    def complete_model_call(self,cid,status,**kwargs):
        next(c for c in self.calls if c["id"] == cid).update(status=status,**kwargs)
        self.events.append(("complete",status))


def provider(stage,fn,model=None,paid=False):
    pins = MAP_PINS if stage == "geometry" else {"model":model or stage,"modelRevision":"test-only","adapter":"test-only"}
    evidence = {"pins":pins,**{g:{"status":"passed","artifactSha256":"0"*64} for g in ("license","quality","runtime")}}
    return ProviderSpec(stage,pins,fn,.01,evidence,paid)


def bundle(repo):
    def discover(payload):
        assert repo.calls[-1]["status"] == "reserved"
        return {"items":[{"label":"unlisted ceramic fixture","box":[0,0,5,12]},{"label":"unlisted transparent bin","box":[7,0,12,12]},{"label":"tiny control","box":[5,0,6,1]}]}
    def geometry(payload):
        assert ("asset","analysis_checkpoint") in repo.events
        frames = []
        y,x = np.mgrid[:12,:12]
        points = np.stack(((x-5.5)/5,(y-5.5)/5,np.full_like(x,2)),axis=-1).astype(float)
        valid = np.ones((12,12),bool)
        valid[0,5] = False
        for image in payload["images"]:
            frames.append({"imageId":image["imageId"],"points":points,"valid":valid,"K":np.array([[10,0,5.5],[0,10,5.5],[0,0,1]]),"cameraToWorld":np.eye(4),
                           "rgb":np.full((12,12,3),100,np.uint8),"inputToCanonical":np.eye(3)})
        return {"frames":frames}
    def segmentation(payload):
        x0,y0,x1,y1 = map(int,payload["box"])
        mask = np.zeros((12,12),bool)
        mask[y0:y1,x0:x1] = True
        return {"mask":mask}
    return {"discovery":provider("discovery",discover),"geometry":provider("geometry",geometry),
            "depth":provider("depth",lambda payload:{"imageSha256":payload["image"]["sha256"]},"Ruicheng/moge-3-vitl"),
            "segmentation":provider("segmentation",segmentation)}


def test_generic_multiphoto_analysis_retains_tiny_objects_caches_frames_and_uses_no_floor(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = bundle(repo)
    document,result = run_analysis(repo,blobs,repo.job,providers)
    assert result["status"] == "succeeded",result["errors"]
    assert len(document["observations"]) == 6
    assert len(document["entities"]) == 5  # two objects, two tiny observations, observed context
    assert all(f["ground"] is None for f in document["coordinateFrames"])
    assert document["target"] == "standalone_object"
    tiny = [e for e in document["entities"] if e["label"] == "tiny control"]
    assert len(tiny) == 2 and all(not e["representations"] for e in tiny)
    assert len([r for e in document["entities"] for r in e["representations"]]) == 3
    context = next(e for e in document["entities"] if e.get("kind") == "capture_context")
    assert context["editable"] is False and context["representations"][0]["coverage"] == "observed_camera_state_only"
    for entity in document["entities"]:
        for rep in entity["representations"]:
            assert np.asarray(rep["bounds"]["min"]).shape == (3,)
        if entity["measurements"]:
            assert entity["measurements"]["dimensionBasis"] == "native_axes_not_ground_aligned"
    validate_document(document)
    calls = len(repo.calls)
    second,again = run_analysis(repo,blobs,repo.job,providers)
    assert again["status"] == "succeeded" and len(repo.calls) == calls
    assert all(s["status"] == "cached" for s in again["stages"])
    assert [e["id"] for e in second["entities"]] == [e["id"] for e in document["entities"]]
    depth = [a for a in repo.assets if a.get("metadata",{}).get("stage") == "depth"]
    assert len(depth) == 2 and depth[0]["metadata"]["cacheKey"] != depth[1]["metadata"]["cacheKey"]


def test_discovery_persisted_even_if_geometry_fails_no_retry_on_unknown_outcome(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = bundle(repo)
    def fail(_):
        raise TimeoutError("private endpoint detail")
    providers["geometry"] = provider("geometry",fail)
    document,result = run_analysis(repo,blobs,repo.job,providers)
    assert result["status"] == "incomplete" and len(document["observations"]) == 6
    assert len(document["entities"]) == 6
    assert any(c["status"] == "outcome_unknown" for c in repo.calls)
    assert "private endpoint" not in str(result)
    calls = len(repo.calls)
    _,second = run_analysis(repo,blobs,repo.job,providers)
    assert len(repo.calls) == calls
    assert any(e["code"] == "model_call_already_reserved" for e in second["errors"])


def test_missing_budget_and_release_gates_prevent_any_live_call(tmp_path,monkeypatch):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = bundle(repo)
    providers = {k:ProviderSpec(v.name,v.pins,lambda _:pytest.fail("paid provider called"),v.estimated_cost_usd,v.release_evidence,True) for k,v in providers.items()}
    document,result = run_analysis(repo,blobs,repo.job,providers)
    assert result["status"] == "incomplete" and not repo.calls
    assert all(e["code"] == "paid_budget_not_configured" for e in result["errors"])
    assert not document["entities"]
    monkeypatch.delenv("PANOPTES_PROVIDER_MANIFEST",raising=False)
    assert providers_from_env() == {}
    with pytest.raises(PlatformError,match="geometry_requires_apache_weights"):
        provider("bad",lambda _:None,"facebook/map-anything").validate("geometry")


def test_shape_ready_does_not_invent_button_placement_and_segmentation_keeps_identity(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = bundle(repo)
    document,_ = run_analysis(repo,blobs,repo.job,providers)
    repo.document = document
    tiny = next(e for e in document["entities"] if e["label"] == "tiny control")
    def generation(payload):
        from ehs_spatial.platform.spatial import primitive_mesh
        mesh = primitive_mesh({"type":"box","dimensions":[.1,.1,.1]})
        return {"vertices":mesh.vertices,"faces":mesh.faces,"proposedObjectToNative":None}
    providers["generation"] = provider("generation",generation)
    job = {**repo.job,"id":str(uuid4()),"kind":"generate_object","inputs":{"entityId":tiny["id"]}}
    generated,result = run_generation(repo,blobs,job,providers)
    assert result["status"] == "incomplete" and result["shapeReadyEntityIds"] == [tiny["id"]]
    entity = next(e for e in generated["entities"] if e["id"] == tiny["id"])
    assert entity["representations"][0]["placementState"] == "unconfirmed"
    assert entity["representations"][0]["placementReason"] == "insufficient_observed_depth"
    assert entity["currentModelTransform"] is None
    validate_document(generated)
    oid = tiny["observationRefs"][0]
    segmented,result = run_segmentation(repo,blobs,{**repo.job,"inputs":{"observationId":oid}},providers)
    assert any(e["id"] == tiny["id"] for e in segmented["entities"])
    assert next(o for o in segmented["observations"] if o["id"] == oid)["revision"] == 2


def test_frozen_provider_snapshot_not_mutable_deployment_and_research_does_not_publish(tmp_path,monkeypatch):
    from ehs_spatial.providers.gemini import GEMINI_MODEL_ID
    spec = provider("discovery",lambda _: {},GEMINI_MODEL_ID)
    config = {"provider":"gemini","pins":dict(spec.pins),"estimatedCostUsd":.01,"releaseEvidence":dict(spec.release_evidence)}
    path = tmp_path/"providers.json"
    path.write_text(json.dumps({"discovery":config}))
    monkeypatch.setenv("PANOPTES_PROVIDER_MANIFEST",str(path))
    frozen = provider_snapshot_from_env()
    path.write_text('{}')
    assert provider_snapshot_from_env() == {} and providers_from_manifest(frozen)["discovery"].pins["model"] == GEMINI_MODEL_ID
    path.write_text(json.dumps({"discovery":{**config,"releaseEvidence":{"credential":"do-not-copy"}}}))
    with pytest.raises(PlatformError,match="secret_in_provider_manifest"):
        provider_snapshot_from_env()
    blobs = LocalBlobStore(tmp_path/"blobs")
    repo = Repo(blobs)
    evidence = deepcopy(spec.release_evidence)
    evidence["quality"] = {"status":"unverified"}
    candidate = replace(spec,invoke=lambda _:{"items":[]},release_evidence=evidence)
    with pytest.raises(PlatformError,match="provider_release_gate_unverified"):
        candidate.validate("discovery")
    monkeypatch.setattr("ehs_spatial.platform.reconstruction.providers_from_manifest",lambda *a,**kw:{"discovery":candidate})
    image = repo.capture["images"][0]
    image = {**image,"sha256":repo.get_asset(image["assetId"])["sha256"]}
    protocol = {"id":"heldout-fixture-only","inputHashes":[image["sha256"]],"baselineRevision":"frozen-baseline",
                "metricDefinitions":{"retention":"discovered count"},"policyThresholds":{},"split":"heldout"}
    result = run_research_stage(repo,blobs,repo.job,"discovery",{},[image],frozen,protocol)
    assert result["status"] == "research_only" and result["sceneRevision"] is None
    assert result["productReleaseStatus"] == "not_changed" and evidence["quality"]["status"] == "unverified"


@pytest.mark.parametrize("outcome",["success","decode_failed","persistence_failed","gpu_failed"])
def test_usage_ledger_survives_received_response_failure_without_inventing_cost(tmp_path,monkeypatch,outcome):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    usage = {"total_input_tokens":123,"total_output_tokens":7}
    telemetry = {"usage":usage,"gpuElapsedSeconds":.75,"actualCostUsd":None}
    def invoke(_):
        if outcome == "decode_failed":
            raise ProviderResponseError({**telemetry,"providerRequestId":"provider-request"})
        return {"telemetry":telemetry,"providerRequestId":"provider-request",**({"providerError":{"code":"inference_failed"}} if outcome == "gpu_failed" else {})}
    stages = _Stages(repo,blobs,repo.job,{"discovery":provider("discovery",invoke)})
    if outcome == "persistence_failed":
        monkeypatch.setattr(stages,"put",lambda *a,**kw:(_ for _ in ()).throw(OSError("disk failure")))
    if outcome == "success":
        stages.call("discovery",[],{})
    else:
        with pytest.raises(PlatformError):
            stages.call("discovery",[],{})
    assert len(repo.calls) == 1
    call = repo.calls[0]
    assert call["response"]["usage"] == usage and call["response"]["gpuElapsedSeconds"] == .75
    assert call["actual_cost"] is None and call["response"]["providerRequestId"] == "provider-request"
    assert call["status"] == {"success":"succeeded","decode_failed":"failed","persistence_failed":"outcome_unknown","gpu_failed":"failed"}[outcome]


def test_capture_source_cameras_cannot_silently_rebind_to_different_geometry(tmp_path):
    from ehs_spatial.platform.reconstruction import _capture,_geometry
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    document,_ = run_analysis(repo,blobs,repo.job,bundle(repo))
    _,_,images = _capture(repo,blobs,repo.job)
    stages = _Stages(repo,blobs,repo.job,{})
    original = next(a for a in repo.assets if a.get("metadata",{}).get("stage") == "geometry")
    changed = stages.load(original)["output"]
    changed["frames"][0]["K"][0,0] *= 2
    changed_asset = stages.put(changed,{"kind":"test_only_different_geometry"})
    before = deepcopy(document)
    with pytest.raises(PlatformError,match="capture_geometry_already_bound"):
        _geometry(document,images,changed,changed_asset)
    assert document == before


def test_analysis_binds_estimated_native_ground_only_to_explicit_mask_evidence(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs,size=36)
    providers = bundle(repo)
    providers["discovery"] = provider("discovery",lambda _:{"items":[{"label":"visible walkway","geometryRole":"floor","box":[0,0,36,36]}]})
    def geometry(payload):
        y,x = np.mgrid[:36,:36]
        points = np.stack(((x-17.5)*.08,np.full_like(x,1.5,dtype=float),2+y*.08),axis=-1)
        return {"frames":[{"imageId":image["imageId"],"points":points,"valid":np.ones((36,36),bool),"K":np.array([[30,0,17.5],[0,30,17.5],[0,0,1.]]),
                          "cameraToWorld":np.eye(4),"rgb":np.full((36,36,3),100,np.uint8),"inputToCanonical":np.eye(3)} for image in payload["images"]]}
    providers["geometry"] = provider("geometry",geometry)
    providers["segmentation"] = provider("segmentation",lambda _:{"mask":np.ones((36,36),bool)})
    document,result = run_analysis(repo,blobs,repo.job,providers)
    assert result["status"] == "succeeded",result["errors"]
    ground = document["coordinateFrames"][0]["ground"]
    assert ground["status"] == "estimated" and ground["sourceRefs"]
    assert ground["plane"] == pytest.approx([0,-1,0,1.5],abs=1e-6)
    assert document["coordinateFrames"][0]["scale"]["nativeToMeters"] is None
    assert any(a.get("metadata",{}).get("kind") == "ground_fit_evidence" for a in document["assets"])
    assert result["groundFit"]["views"] and all(e["measurements"].get("groundHeightNative",0) == pytest.approx(0,abs=1e-6) for e in document["entities"])
    validate_document(document)
