from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from uuid import uuid4

import numpy as np
from PIL import Image
import pytest

from ehs_spatial.platform.contracts import PlatformError,empty_document,validate_document,digest
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
        height,width = (size,size) if isinstance(size,int) else size
        images = []
        for i in range(2):
            raw = io.BytesIO()
            Image.fromarray(np.full((height,width,3),40+i,np.uint8)).save(raw,format="PNG")
            asset = self.register_asset(self.pid,blobs.put(raw.getvalue(),"image/png"))
            images.append({"id":asset["id"],"assetId":asset["id"],"width":width,"height":height,"pixelMapping":[]})
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
    def get_stage_cache(self,pid,key):
        return deepcopy(next((a for a in self.assets if a['projectId'] == pid
            and a.get('metadata', {}).get('kind') == 'stage_cache'
            and a['metadata'].get('cacheKey') == key), None))
    def reserve_model_call(self,*args,**kwargs):
        if kwargs.get("paid") and self.paid_budget is None:
            raise PlatformError("paid_budget_not_configured",409)
        if kwargs.get("paid") and args[5] + sum(c["estimate"] for c in self.calls) > self.paid_budget:
            raise PlatformError("paid_budget_exceeded",409)
        if any(x["key"] == args[4] for x in self.calls):
            existing = next(x for x in self.calls if x["key"] == args[4])
            raise PlatformError("model_call_already_reserved",409,status=existing["status"])
        call = {"id":str(uuid4()),"key":args[4],"status":"reserved","estimate":args[5]}
        self.calls.append(call)
        self.events.append(("reserve",args[2]))
        return call
    def complete_model_call(self,cid,status,**kwargs):
        next(c for c in self.calls if c["id"] == cid).update(status=status,**kwargs)
        self.events.append(("complete",status))


def test_generation_batch_requires_explicit_targets_and_preserves_existing_models():
    from ehs_spatial.platform.reconstruction import _generation_targets
    entities = [{'id': 'floor', 'geometryRole': 'floor'}, {'id':'context','sourceContext': True},
        {'id':'modeled','activeModelRepresentationId':'mesh','representations':[{'id':'mesh','kind':'generated_mesh'}]},
        {'id':'parent'}, {'id':'part','parentEntityId':'parent'}, {'id':'a'}, {'id':'b'}]
    document = {'entities':entities, 'observations':[]}
    job = {'kind':'generate_scene', 'inputs':{}}
    for ids, code in [(None,'generation_targets_required'), ([], 'generation_targets_required'),
            (['a','a'], 'generation_targets_required'), (['unknown'],'entity_not_found'),
            (['floor'],'generation_reference_surface'), (['context'],'generation_reference_surface'),
            (['modeled'],'generation_model_already_present'), (['parent'],'generation_part_workflow_required'),
            (['part'],'generation_part_workflow_required')]:
        with pytest.raises(PlatformError, match=code):
            _generation_targets(document, {**job, 'inputs':{'entityIds':ids}})
    requested, targets = _generation_targets(document, {**job, 'inputs':{'entityIds':['b','a']}})
    assert requested == [entity['id'] for entity in targets] == ['b','a']
    # Explicit regeneration remains available and leaves the active pose intact
    # (covered by the existing end-to-end regeneration cases).
    assert _generation_targets(document, {'kind':'generate_object','inputs':{'entityId':'modeled'}})[0] == ['modeled']


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
        return geometry_response(payload['images'])
    def segmentation(payload):
        x0,y0,x1,y1 = map(int,payload["box"])
        mask = np.zeros((12,12),bool)
        mask[y0:y1,x0:x1] = True
        return {"mask":mask}
    return {"discovery":provider("discovery",discover),"geometry":provider("geometry",geometry),
            "depth":provider("depth",lambda payload:depth_response(payload['image']),"Ruicheng/moge-3-vitl"),
            "segmentation":provider("segmentation",segmentation)}


def geometry_response(images):
    frames = []
    y,x = np.mgrid[:12,:12]
    points = np.stack(((x-5.5)/5,(y-5.5)/5,np.full_like(x,2)),axis=-1).astype(float)
    valid = np.ones((12,12),bool)
    valid[0,5] = False
    for image in images:
        frames.append({'imageId':image['imageId'],'points':points,'valid':valid,
            'K':np.array([[10,0,5.5],[0,10,5.5],[0,0,1]]),'cameraToWorld':np.eye(4),
            'rgb':np.full((12,12,3),100,np.uint8),'inputToCanonical':np.eye(3)})
    return {'frames':frames}


def depth_response(image):
    shape = (image['height'],image['width'])
    points = np.zeros((*shape,3))
    points[...,2] = 2
    return {'imageId':image['imageId'], 'imageSha256':image['sha256'],
            'points':points, 'mask':np.ones(shape,bool), 'depth':points[...,2],
            'intrinsics':np.eye(3), 'inputToCanonical':np.eye(3)}


@pytest.mark.parametrize('bad', ['photo','grid','nonfinite','negative','different_z'])
def test_auxiliary_depth_must_belong_to_the_source_grid(bad):
    from ehs_spatial.platform.reconstruction import _depth_response
    image = {'id':'photo-a','imageId':'photo-a','sha256':'a'*64,'width':12,'height':12}
    response = depth_response(image)
    if bad == 'photo': response['imageId'] = 'photo-b'
    elif bad == 'grid': response['mask'] = response['mask'][:-1]
    elif bad == 'nonfinite': response['points'][0,0,0] = np.nan
    elif bad == 'negative': response['depth'][0,0] = -1
    else: response['depth'] = response['depth'].copy() + 1
    with pytest.raises(PlatformError):
        _depth_response(image,response)


def test_generic_multiphoto_analysis_retains_tiny_objects_caches_frames_and_uses_no_floor(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = bundle(repo)
    document,result = run_analysis(repo,blobs,repo.job,providers)
    assert result["status"] == "succeeded",result["errors"]
    assert len(document["observations"]) == 6
    assert len(document["entities"]) == 6  # two objects, two tiny observations, two photo contexts
    assert all(f["ground"] is None for f in document["coordinateFrames"])
    assert document["target"] == "standalone_object"
    tiny = [e for e in document["entities"] if e["label"] == "tiny control"]
    assert len(tiny) == 2 and all(not e["representations"] for e in tiny)
    assert all(e["associationEvidence"]["status"] == "insufficient_support" for e in tiny)
    linked = [e for e in document["entities"] if e.get("associationEvidence", {}).get("status") == "confirmed"]
    assert len(linked) == 2 and all(len(e["observationRefs"]) == 2 for e in linked)
    assert all(any(link["accepted"] for link in e["associationEvidence"]["geometryVerification"]["links"]) for e in linked)
    assert len([r for e in document["entities"] for r in e["representations"]]) == 6
    context = next(e for e in document["entities"] if e.get("kind") == "capture_context")
    assert context["sourceContext"] is True
    assert len([e for e in document["entities"] if not e.get("sourceContext")]) == 4
    assert all(e['cadReference']['status'] == 'resolved' for e in document['entities'] if not e.get('sourceContext'))
    assert context["editable"] is False and context["representations"][0]["coverage"] == "observed_camera_state_only"
    for entity in document["entities"]:
        for rep in entity["representations"]:
            assert np.asarray(rep["bounds"]["min"]).shape == (3,)
        if entity["measurements"]:
            assert entity["measurements"]["dimensionBasis"] == "native_axes_not_ground_aligned"
            bounds = entity["measurements"]["observedBounds"]
            assert isinstance(entity["measurements"]["dimensionsNative"],list)
            assert entity["measurements"]["dimensionsNative"] == pytest.approx(np.asarray(bounds["max"])-np.asarray(bounds["min"]))
    validate_document(document)
    calls = len(repo.calls)
    second,again = run_analysis(repo,blobs,repo.job,providers)
    assert again["status"] == "succeeded" and len(repo.calls) == calls
    assert all(s["status"] == "cached" for s in again["stages"])
    assert [e["id"] for e in second["entities"]] == [e["id"] for e in document["entities"]]
    depth = [a for a in repo.assets if a.get("metadata",{}).get("stage") == "depth"]
    assert len(depth) == 2 and depth[0]["metadata"]["cacheKey"] != depth[1]["metadata"]["cacheKey"]
    repo.document = deepcopy(document)
    # Context identity follows the shared field, including imported contexts.
    next(e for e in repo.document["entities"] if e.get("sourceContext")).pop("kind")
    with pytest.raises(PlatformError, match='generation_targets_required'):
        run_generation(repo,blobs,{**repo.job,"kind":"generate_scene","inputs":{}},{})
    requested = [e['id'] for e in document['entities'] if not e.get('sourceContext')]
    _, generation = run_generation(repo,blobs,{**repo.job,"kind":"generate_scene","inputs":{"entityIds": requested}},{})
    assert set(generation["manifest"]["entityIds"]) == {e["id"] for e in document["entities"] if not e.get("sourceContext")}
    assert len(repo.calls) == calls


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
    assert all(e["associationEvidence"]["status"] == "geometry_missing" for e in document["entities"])
    assert any(c["status"] == "outcome_unknown" for c in repo.calls)
    assert "private endpoint" not in str(result)
    calls = len(repo.calls)
    _,second = run_analysis(repo,blobs,repo.job,providers)
    assert len(repo.calls) == calls
    assert any(e["code"] == "provider_outcome_unknown" for e in second["errors"])


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
    document['coordinateFrames'][0]['ground'] = {'normal': [0., 0., 1.], 'plane': [0., 0., 1., 0.], 'source': 'manual'}
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
    assert entity['activeModelRepresentationId'] is None and entity['representations'][0]['planProjection']['polygons']
    validate_document(generated)
    oid = tiny["observationRefs"][0]
    segmented,result = run_segmentation(repo,blobs,{**repo.job,"inputs":{"observationId":oid}},providers)
    assert any(e["id"] == tiny["id"] for e in segmented["entities"])
    assert next(o for o in segmented["observations"] if o["id"] == oid)["revision"] == 2
    assert next(e for e in segmented['entities'] if e['id'] == tiny['id'])['cadReference']['sourceRefs'] == [{'observationId': oid, 'revision': 2}]


def test_generation_uses_reviewed_observation_and_rejects_invalid_batch_before_calls(tmp_path):
    from ehs_spatial.platform.spatial import primitive_mesh

    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = bundle(repo)
    document, _ = run_analysis(repo, blobs, repo.job, providers)
    targets = [e for e in document['entities'] if e['label'] in ('unlisted ceramic fixture', 'unlisted transparent bin')]
    observations = {o['id']: o for o in document['observations']}
    target, other = targets[:2]
    preferred, broad = (observations[oid] for oid in target['observationRefs'])
    preferred['geometrySupport']['validPixelCount'] = 1
    broad['geometrySupport']['validPixelCount'] = 999
    repo.document = document
    seen = []
    mesh = primitive_mesh({'type':'box', 'dimensions':[.1, .2, .3]})
    def generate(payload):
        seen.append(payload['imageId'])
        return {'vertices':mesh.vertices, 'faces':mesh.faces, 'proposedObjectToNative':np.eye(4)}
    providers['generation'] = provider('generation', generate)
    job = {**repo.job, 'id':str(uuid4()), 'kind':'generate_scene', 'inputs':{
        'entityIds':[target['id']], 'observationIds':[preferred['id']]}}
    generated, result = run_generation(repo, blobs, job, providers)
    assert seen == [preferred['imageId']], 'Reviewed input must not be replaced by a larger contaminated mask'
    assert result['shapeReadyEntityIds'] == [target['id']]
    model = next(r for e in generated['entities'] if e['id'] == target['id'] for r in e['representations'] if r['kind'] == 'generated_mesh')
    assert {'observationId':preferred['id'], 'revision':preferred['revision']} in model['sourceRefs']
    other_oid = other['observationRefs'][0]
    calls = len(repo.calls)
    for ids in ([], [preferred['id']], [preferred['id'], broad['id']],
                [preferred['id'], preferred['id']], [preferred['id'], 'unknown'],
                [preferred['id'], other_oid, broad['id']], 'not-a-list'):
        with pytest.raises(PlatformError, match='generation_anchors_invalid'):
            run_generation(repo, blobs, {**job, 'inputs':{'entityIds':[target['id'],other['id']], 'observationIds':ids}}, providers)
        assert len(repo.calls) == calls, 'Invalid batch anchors must fail before any model reservation'
    del observations[other_oid]['maskAssetId']
    repo.document = document
    with pytest.raises(PlatformError, match='generation_anchors_invalid'):
        run_generation(repo, blobs, {**job, 'inputs':{'entityIds':[target['id'],other['id']], 'observationIds':[preferred['id'],other_oid]}}, providers)
    assert len(repo.calls) == calls


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
    candidate = replace(spec,invoke=lambda _:{"items":[]},release_evidence=evidence,paid=True)
    with pytest.raises(PlatformError,match="provider_release_gate_unverified"):
        candidate.validate("discovery")
    monkeypatch.setattr("ehs_spatial.platform.reconstruction.providers_from_manifest",lambda *a,**kw:{"discovery":candidate})
    image = repo.capture["images"][0]
    image = {**image,"sha256":repo.get_asset(image["assetId"])["sha256"]}
    protocol = {"id":"heldout-fixture-only","stage":"discovery","purpose":"quality_validation","inputHashes":[image["sha256"]],"baselineRevision":repo.rid,
                "metricDefinitions":{"retention":"discovered count"},"policyThresholds":{},"split":"heldout","entityId":"fixture",
                "inputAssetHashes":[{"assetId":image["assetId"],"sha256":image["sha256"]}],"payloadSha256":digest({}),
                "providerManifestSha256":digest(frozen),"callLimits":{"maxCalls":1,"maxCostPerCallUsd":.01,"maxTotalCostUsd":.01}}
    repo.paid_budget = .01
    job = {**repo.job,"kind":"validate_model","config":{"researchProtocolSha256":digest(protocol)}}
    result = run_research_stage(repo,blobs,job,"discovery",{},[image],frozen,protocol)
    assert result["status"] == "succeeded" and result["scope"] == "research_only" and result["sceneRevision"] is None
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
    _geometry(document,images,changed,changed_asset)
    assert all(c in document["cameras"] for c in before["cameras"])
    assert all(f in document["coordinateFrames"] for f in before["coordinateFrames"])
    assert len(document["cameras"]) == 2 * len(before["cameras"])
    assert {b["geometrySolutionId"] for b in document["geometryBindings"].values()} == {changed_asset["id"]}
    assert document["geometryBindings"] != before["geometryBindings"]
    validate_document(document)


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
    assert all('planProjection' in r for e in document['entities'] if not e.get('sourceContext') for r in e['representations'])
    validate_document(document)


@pytest.mark.parametrize("pose_source", ["imported_proposal", "requires_alignment_confirmation", "manual", "primitive", "unattributed"])
def test_regeneration_retains_source_models_and_preserves_active_pose(tmp_path, pose_source):
    from ehs_spatial.platform.repository import apply_operations
    from ehs_spatial.platform.spatial import primitive_mesh

    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = bundle(repo)
    document, _ = run_analysis(repo, blobs, repo.job, providers)
    target = next(e for e in document["entities"] if e["label"] == "unlisted ceramic fixture")
    old = deepcopy(target["representations"][0])
    old.update(id=str(uuid4()), kind="generated_mesh", placementState="unconfirmed", placementReason=pose_source)
    old["transform"]["position"] = [8., 9., 10.]
    target["representations"].append(old)
    target["activeModelRepresentationId"] = old["id"]
    target["currentModelTransform"] = deepcopy(old["transform"])
    if pose_source == "manual":
        document, _ = apply_operations(document, [{"type":"setTransform", "entityId":target["id"], "transform":old["transform"]}])
    elif pose_source == "primitive":
        document, _ = apply_operations(document, [{"type":"setPrimitive", "entityId":target["id"], "primitive":{"type":"box", "dimensions":[1, 2, 3]}, "transform":old["transform"]}])
    elif pose_source == "unattributed":
        target["currentModelTransform"]["position"] = [18., 19., 20.]
    repo.document = document
    before = deepcopy(document)
    mesh = primitive_mesh({"type":"box", "dimensions":[.1, .2, .3]})
    proposed = np.eye(4)
    proposed[:3, 3] = [1., 2., 3.]
    providers["generation"] = provider("generation", lambda _: {"vertices":mesh.vertices, "faces":mesh.faces, "proposedObjectToNative":proposed})
    generated, result = run_generation(repo, blobs, {**repo.job, "id":str(uuid4()), "kind":"generate_object", "inputs":{"entityId":target["id"]}}, providers)
    entity = next(e for e in generated["entities"] if e["id"] == target["id"])
    new = next(r for r in entity["representations"] if r["kind"] == "generated_mesh" and r["id"] != old["id"])
    assert new["transform"]["position"] == pytest.approx([1., 2., 3.])
    assert new["placementState"] == "unconfirmed"
    assert result["placementConfirmedEntityIds"] == []
    assert entity["currentModelTransform"] == next(e for e in before["entities"] if e["id"] == target["id"])["currentModelTransform"]
    assert next(r for r in entity["representations"] if r["id"] == old["id"]) == next(r for e in before["entities"] for r in e["representations"] if r["id"] == old["id"])
    assert repo.document == before
    validate_document(generated)


def test_original_mask_rings_survive_crop_resize_holes_single_pixels_and_resegmentation(tmp_path, monkeypatch):
    from shapely import Polygon, contains_xy

    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs,size=(9,15))
    providers = bundle(repo)
    providers["discovery"] = provider("discovery",lambda _:{"items":[{"label":"fixture","box":[0,0,15,9]}]})
    mask = np.zeros((9,15),bool)
    mask[1:8,2:13] = True
    mask[3:6,5:9] = False
    mask[0,0] = mask[8,14] = True  # Source pixels outside the geometry crop survive.
    providers["segmentation"] = provider("segmentation",lambda _:{"mask":mask})
    mapping = np.array([[.5,0,-1],[0,.5,-1],[0,0,1.]])
    def geometry(payload):
        y,x = np.mgrid[:4,:6]
        return {"frames":[{"imageId":image["imageId"],"points":np.stack((x,y,np.ones_like(x)*2),axis=-1),
            "valid":np.ones((4,6),bool),"K":np.array([[5,0,2.5],[0,5,1.5],[0,0,1.]]),"cameraToWorld":np.eye(4),
            "rgb":np.full((4,6,3),100,np.uint8),"inputToCanonical":mapping} for image in payload["images"]]}
    providers["geometry"] = provider("geometry",geometry)
    document,result = run_analysis(repo,blobs,repo.job,providers)
    assert result["status"] == "succeeded",result["errors"]
    def check(observation, expected):
        assert observation["polygonCoordinateConvention"] == "pixel_edges" and observation["fillRule"] == "evenodd"
        assert observation["maskPolygonization"]["methodVersion"] == "pixel-runs-union-v1"
        y,x = np.mgrid[:9,:15]
        restored = np.zeros(expected.shape,bool)
        for ring in observation["originalPixelPolygons"]:
            restored ^= contains_xy(Polygon(ring),x+.5,y+.5)
        np.testing.assert_array_equal(restored,expected)
        asset = repo.get_asset(observation["maskAssetId"])
        with Image.open(io.BytesIO(blobs.get(asset["storageKey"],asset["sha256"],asset["sizeBytes"]))) as image:
            np.testing.assert_array_equal(np.asarray(image)>0,expected)
        assert observation["originalPixelBox"] == [0,0,15,9]
    for observation in document["observations"]:
        check(observation,mask)
        assert len(observation["originalPixelPolygons"]) == 4  # Outer, hole, two single-pixel components.
    repo.document = document
    original = deepcopy(document)
    oid = document["observations"][0]["id"]
    mask = np.zeros_like(mask)
    empty,_ = run_segmentation(repo,blobs,{**repo.job,"inputs":{"observationId":oid}},providers)
    observation = next(o for o in empty["observations"] if o["id"] == oid)
    check(observation,mask)
    assert observation["revision"] == 2 and observation["maskStatus"] == "empty"
    assert observation["originalPixelPolygons"] == [] and "segmentation_empty" in observation["missingEvidence"]
    assert any(oid in entity["observationRefs"] for entity in empty["entities"])
    assert repo.document == original
    repo.document = empty
    mask[::2,::2] = True
    monkeypatch.setattr("ehs_spatial.platform.reconstruction.MAX_MASK_POLYGON_RUNS",2)
    limited,_ = run_segmentation(repo,blobs,{**repo.job,"inputs":{"observationId":oid}},providers)
    observation = next(o for o in limited["observations"] if o["id"] == oid)
    assert observation["maskStatus"] == "present" and "originalPixelPolygons" not in observation
    assert observation["maskPolygonization"]["status"] == "complexity_limit"
    assert "mask_polygon_complexity_limit" in observation["missingEvidence"] and "segmentation_empty" not in observation["missingEvidence"]
    asset = repo.get_asset(observation["maskAssetId"])
    with Image.open(io.BytesIO(blobs.get(asset["storageKey"],asset["sha256"],asset["sizeBytes"]))) as image:
        np.testing.assert_array_equal(np.asarray(image)>0,mask)
    validate_document(limited)


def test_append_photos_registers_new_solution_preserves_source_and_reuses_duplicates(tmp_path):
    from ehs_spatial.platform.reconstruction import _load_geometry
    from ehs_spatial.platform.spatial import transform_points
    from scipy.spatial.transform import Rotation
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs,size=(24,32))
    y,x = np.mgrid[:24,:32]
    depth = 3+.003*x+.005*y
    points = np.stack(((x-15.5)*depth/30,(y-11.5)*depth/30,depth),-1)
    k = np.array([[30.,0,15.5],[0,30.,11.5],[0,0,1]])
    providers = bundle(repo)
    providers['discovery'] = provider('discovery',lambda _:{'items':[{'label':'independent fixture','box':[8,6,20,18]}]})
    def segment(payload):
        mask = np.zeros((24,32),bool); mask[6:18,8:20]=True
        return {'mask':mask}
    providers['segmentation'] = provider('segmentation',segment)
    providers['geometry'] = provider('geometry',lambda payload:{'frames':[{'imageId':i['imageId'],'points':points,'valid':np.ones((24,32),bool),'K':k,'cameraToWorld':np.eye(4),'rgb':np.full((24,32,3),100,np.uint8),'inputToCanonical':np.eye(3)} for i in payload['images']]})
    original,result = run_analysis(repo,blobs,repo.job,providers)
    assert result['status']=='succeeded',result
    before = deepcopy(original)
    raw = io.BytesIO(); Image.fromarray(np.full((24,32,3),80,np.uint8)).save(raw,format='PNG')
    asset = repo.register_asset(repo.pid,blobs.put(raw.getvalue(),'image/png'))
    new_image = {'id':asset['id'],'assetId':asset['id'],'width':32,'height':24,'pixelMapping':[]}
    new_capture = str(uuid4())
    repo.capture = {'id':new_capture,'target':'standalone_object','images':[new_image]}
    repo.document = deepcopy(original)
    repo.document['captureId']=new_capture;repo.document['captureIds'].append(new_capture)
    repo.document['assets'].append(asset)
    repo.document['geometryBindings'][asset['id']]=None
    # Each joint solve has a different scale/orientation. Native source remains frozen.
    similarity = np.eye(4);similarity[:3,:3]=1.8*Rotation.from_euler('xyz',[.2,.3,-.4]).as_matrix();similarity[:3,3]=[2.,4.,-1.]
    rotation = similarity[:3,:3]/1.8
    c2w = np.eye(4);c2w[:3,:3]=rotation;c2w[:3,3]=similarity[:3,3]
    geometry_calls=[]
    def joint(payload):
        geometry_calls.append([i['imageId'] for i in payload['images']])
        return {'frames':[{'imageId':i['imageId'],'points':transform_points(points,similarity),'valid':np.ones((24,32),bool),'K':k,'cameraToWorld':c2w,'rgb':np.full((24,32,3),100,np.uint8),'inputToCanonical':np.eye(3)} for i in payload['images']]}
    providers['geometry']=provider('geometry',joint)
    job={**repo.job,'id':str(uuid4()),'inputs':{'captureMode':'append','captureId':new_capture,'newImageIds':[asset['id']],'reusedImageIds':[]}}
    updated,result=run_analysis(repo,blobs,job,providers)
    assert result['status']=='succeeded',result['errors']
    assert geometry_calls and all(2<=len(ids)<=4 for ids in geometry_calls)
    assert all(c in updated['cameras'] for c in before['cameras'])
    assert all(o['id'] in {v['id'] for v in updated['observations']} for o in before['observations'])
    assert len(updated['observations'])==3
    objects=[e for e in updated['entities'] if not e.get('sourceContext')]
    assert len(objects)==1 and len(objects[0]['observationRefs'])==3
    old_entity=next(e for e in before['entities'] if not e.get('sourceContext'))
    assert objects[0]['id']==old_entity['id']
    assert objects[0]['measurementEvidence']==old_entity['measurementEvidence']
    assert objects[0]['cadReference']['referenceImageId'] == old_entity['cadReference']['referenceImageId']
    contexts = [e for e in updated['entities'] if e.get('sourceContext')]
    assert all(e in contexts for e in before['entities'] if e.get('sourceContext'))
    assert any(r['sourceRefs'][0].get('imageId') == asset['id'] for e in contexts for r in e['representations'])
    assert updated['registrationEvidence'][0]['metrics']['holdoutP95Relative']<1e-6
    loaded,_=_load_geometry(updated,[],_Stages(repo,blobs,job,{}))
    assert len({f.coordinate_frame_id for f in loaded.values()})==1
    calls=len(repo.calls)
    repo.document=updated
    _,repeat=run_analysis(repo,blobs,{**job,'id':str(uuid4()),'inputs':{**job['inputs'],'newImageIds':[],'reusedImageIds':[asset['id']]}},{})
    assert repeat['newModelCalls']==0 and len(repo.calls)==calls
    validate_document(updated)


def test_resegmentation_uses_replacement_mask_and_only_invalidates_its_observation_sources(tmp_path):
    from ehs_spatial.platform.reconstruction import _load_geometry, _load_masks
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = bundle(repo)
    document, _ = run_analysis(repo, blobs, repo.job, providers)
    target = next(e for e in document['entities'] if e['label'] == 'unlisted ceramic fixture')
    oid, other_oid = target['observationRefs']
    observation = next(o for o in document['observations'] if o['id'] == oid)
    stages = _Stages(repo, blobs, repo.job, {})
    _, canonical = _load_geometry(document, [], stages)
    loaded, _ = _load_masks(document, canonical, stages)
    old_mask = loaded[oid]
    raw = io.BytesIO(); np.save(raw, old_mask, allow_pickle=False)
    mask_asset = stages.put(raw.getvalue(), {'kind': 'canonical_mask'})
    document['assets'].append(mask_asset)
    observation['maskEvidence'] = {'canonicalMaskAssetId': mask_asset['id'], 'geometryManifestAssetId': canonical[observation['imageId']].get('geometryManifestAssetId'), 'inputToCanonical': canonical[observation['imageId']]['inputToCanonical'].tolist()}
    source_rep = next(r for r in target['representations'] if r['kind'] == 'observed_surface')
    source_rep['sourceRefs'] = [{'observationId': oid, 'revision': observation['revision']}]
    alternative = deepcopy(source_rep)
    alternative.update(id=str(uuid4()), sourceRefs=[{'observationId': other_oid, 'revision': 1}], kind='generated_mesh')
    target['representations'].append(alternative)
    target['activeModelRepresentationId'] = alternative['id']
    target['currentModelTransform'] = deepcopy(alternative['transform'])
    original_evidence = deepcopy(target['measurementEvidence'])
    unrelated = {e['id']:deepcopy(e) for e in document['entities'] if e['id'] != target['id']}
    repo.document = document
    # A valid replacement with the same byte geometry must create a new source
    # representation, while the original stays stale and the other view stays valid.
    providers['segmentation'] = provider('segmentation', lambda _: {'mask': old_mask.copy()}, model='replacement-mask')
    job = {**repo.job, 'id': str(uuid4()), 'kind': 'segment_observation', 'inputs': {'captureId': repo.cid, 'observationId': oid}}
    updated, result = run_segmentation(repo, blobs, job, providers)
    assert not result['errors'], result
    revised = next(o for o in updated['observations'] if o['id'] == oid)
    assert revised['revision'] == observation['revision'] + 1 and 'maskEvidence' not in revised
    assert mask_asset['id'] in {a['id'] for a in updated['assets']}
    kept = next(e for e in updated['entities'] if e['id'] == target['id'])
    assert next(r for r in kept['representations'] if r['id'] == source_rep['id'])['sourceValidity'] == 'stale'
    assert next(r for r in kept['representations'] if r['id'] == alternative['id']) == alternative
    new_reps = [r for r in kept['representations'] if r['kind'] == 'observed_surface' and any(ref.get('observationId') == oid and ref.get('revision') == revised['revision'] for ref in r.get('sourceRefs', []))]
    assert new_reps and all(r['id'] != source_rep['id'] and r.get('sourceValidity') != 'stale' for r in new_reps)
    assert all(record in kept['measurementEvidence'] for record in original_evidence)
    for eid, old in unrelated.items():
        current = next(e for e in updated['entities'] if e['id'] == eid)
        assert current['representations'] == old['representations'] and current['measurements'] == old['measurements']
    # A genuinely different mask must bypass the retained old canonical bytes.
    repo.document = updated
    replacement = old_mask.copy(); replacement[:4] = False
    providers['segmentation'] = provider('segmentation', lambda _: {'mask': replacement}, model='replacement-mask-2')
    again, _ = run_segmentation(repo, blobs, {**job, 'id': str(uuid4())}, providers)
    _, records = _load_geometry(again, [], _Stages(repo, blobs, job, {}))
    masks, _ = _load_masks(again, records, _Stages(repo, blobs, job, {}))
    assert np.array_equal(masks[oid], replacement) and not np.array_equal(masks[oid], old_mask)
    validate_document(again)


def test_resegmentation_ground_fit_stays_in_affected_coordinate_frame(tmp_path):
    from ehs_spatial.platform.reconstruction import _ground
    from ehs_spatial.platform.spatial import FrameGeometry
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    document, _ = run_analysis(repo, blobs, repo.job, bundle(repo))
    old_frame = document['coordinateFrames'][0]
    old_frame['ground'] = {'normal': [0., -1., 0.], 'plane': [0., -1., 0., 1.5], 'source': 'observed_floor_mask_and_estimated_depth', 'unit': 'native'}
    old_frame['groundFit'] = {'status': 'estimated', 'source': 'old'}
    old_snapshot = deepcopy(old_frame)
    new_frame = {**deepcopy(old_frame), 'id': 'unregistered-new-frame', 'ground': None, 'groundFit': None}
    document['coordinateFrames'].append(new_frame)
    old_oid, new_oid = document['observations'][:2]
    old_oid['labelEvidence'] = [{'geometryRole': 'floor'}]
    new_oid['labelEvidence'] = [{'geometryRole': 'floor'}]
    # Different image IDs are the important boundary; the geometry arrays are
    # intentionally simple because no successful floor fit is required here.
    new_oid['imageId'] = repo.capture['images'][1]['id']
    y,x = np.mgrid[:12,:12]
    points = np.stack((x*.1,y*.1,np.full_like(x,2,dtype=float)),-1)
    k = np.array([[10.,0,5.5],[0,10.,5.5],[0,0,1.]])
    frames = {old_oid['imageId']: FrameGeometry(old_oid['imageId'],old_frame['id'],'a'*64,points,np.ones((12,12),bool),k,np.eye(4)),
              new_oid['imageId']: FrameGeometry(new_oid['imageId'],new_frame['id'],'b'*64,points,np.ones((12,12),bool),k,np.eye(4))}
    _ground(document,frames,{new_oid['id']:np.ones((12,12),bool)},_Stages(repo,blobs,repo.job,{}),affected_observation_ids={new_oid['id']})
    assert old_frame == old_snapshot and new_frame['ground'] is None
    assert new_frame['groundFit']['reason'] != 'unregistered_coordinate_frames'


def source_equivalence_case(tmp_path, *, native_image_sha256=None):
    """Real immutable JSON/RLE blobs; no model provider or network needed."""
    from ehs_spatial.platform.identity import migrate_document
    from ehs_spatial.platform.spatial import FrameGeometry
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    stages = _Stages(repo, blobs, repo.job, {})
    document = deepcopy(repo.document)
    def asset(value):
        saved = stages.put(json.dumps(value, sort_keys=True).encode(), {'kind':'source_identity_evidence'}, 'application/json')
        if not any(a['id'] == saved['id'] for a in document['assets']):
            document['assets'].append(saved)
        return {'assetId':saved['id'], 'sha256':saved['sha256']}
    image_id, other_image = [i['id'] for i in repo.capture['images']]
    image_sha = next(a['sha256'] for a in document['assets'] if a['id'] == image_id)
    mask = np.zeros((12,12), bool); mask[:,2:10] = True
    sam = asset({'rle':[{'size':[12,12], 'counts':[24,96,24]}, {'size':[12,12], 'counts':[144]}]})
    source_ref = {'encoding':'rle', 'sha256':sam['sha256'], 'pointer':['rle',0], 'shape_hw':[12,12]}
    native = {'source_sha256':sam['sha256'], 'source_instance':0, 'source_frame':'original-frame'}
    provenance = {'objects':[{'id':'native-object', 'views':[{'frame_id':'mapped-frame', 'provenance':native}]},
                             {'id':'unrelated-object', 'views':[{'frame_id':'mapped-frame', 'provenance':deepcopy(native)}]}],
                  'observed_regions':[{'id':'raw-object', 'provenance':{'source_image_sha256':image_sha,'source_record':{'source_frame_id':'original-frame','source_mask':{'ref':source_ref}}}}]}
    if native_image_sha256 is not None:
        provenance['objects'][0]['views'][0]['image_sha256'] = native_image_sha256
    evidence = asset(provenance)
    pair = {'kind':'canonical_sam_rle', 'observationRefs':[{'observationId':'native-observation','revision':1},{'observationId':'raw-observation','revision':1}],
            'imageId':image_id, 'imageSha256':image_sha, 'sourceRef':{**sam, 'jsonPointer':'/rle/0'},
            'canonicalShape':[12,12], 'canonicalMaskSha256':hashlib.sha256(mask.tobytes(order='C')).hexdigest(),
            'evidenceRefs':[{**evidence, 'jsonPointer':'/objects/0/views/0/provenance', 'role':'native_mask_provenance'},
                            {**evidence, 'jsonPointer':'/observed_regions/0/provenance/source_record/source_mask/ref', 'role':'raw_source_reference'}]}
    document['coordinateFrames'] = [{'id':'frame','convention':'opencv','scale':{'status':'uncalibrated','nativeToMeters':None},'ground':None,'sourceRefs':[{'assetId':sam['assetId']}]}]
    k = [[10.,0,5.5],[0,10.,5.5],[0,0,1]]
    for index, image in enumerate((image_id, other_image)):
        document['cameras'].append({'id':f'camera-{index}','imageId':image,'coordinateFrameId':'frame','width':12,'height':12,'K':k,'cameraToWorld':np.eye(4).tolist()})
    for oid, image, record in [('native-observation',image_id,'native-object'),('raw-observation',image_id,'raw-object'),('other-photo-observation',other_image,'other-object')]:
        document['observations'].append({'id':oid,'revision':1,'imageId':image,'originalPixelBox':[2,0,10,12],'maskAssetId':None,
                                        'sourceRefs':[{'assetId':evidence['assetId'],'sourceRecordId':record,'imageSha256':next(a['sha256'] for a in document['assets'] if a['id']==image)}]})
        document['entities'].append({'id':f'entity-{oid}','label':'not used as evidence','observationRefs':[oid],'representations':[],
                                    'associationState':'association_pending','measurements':{},'currentModelTransform':None})
    document['observations'][0]['maskEvidence'] = {'sourceRefs':[{**evidence,'jsonPointer':'/objects/0/views/0','imageSha256':image_sha,'sourceFrameId':'mapped-frame'}]}
    document = migrate_document(document, base_revision_id=repo.rid)
    masks = {o['id']:mask.copy() for o in document['observations']}
    yy,xx = np.mgrid[:12,:12]
    points = np.stack(((xx-5.5)*.2,(yy-5.5)*.2,np.full_like(xx,2,dtype=float)),-1)
    frames = {image:FrameGeometry(image,'frame',next(a['sha256'] for a in document['assets'] if a['id']==image),points,np.ones((12,12),bool),np.array(k),np.eye(4)) for image in (image_id,other_image)}
    def save_proof(value):
        proof = {'schemaVersion':1,'kind':'same_source_observation_equivalences','pairs':[value]}
        document['sourceIdentityEvidence'] = [asset(proof)]
    save_proof(pair)
    return document, masks, stages, pair, save_proof, frames


def test_source_equivalence_verifier_reads_exact_source_rle_and_seeds_geometry(tmp_path):
    from ehs_spatial.platform.reconstruction import _verified_source_equivalences, _associate_and_surfaces
    document, masks, stages, pair, _, frames = source_equivalence_case(tmp_path)
    before = deepcopy(document)
    verified, skipped = _verified_source_equivalences(document, masks, stages)
    assert len(verified) == 1 and skipped == []
    assert document == before and not stages.repo.calls
    assert verified[0]['evidenceRefs'][-1] == {**document['sourceIdentityEvidence'][0], 'jsonPointer':'/pairs/0'}
    unseeded = deepcopy(document)
    unseeded.pop('sourceIdentityEvidence')
    _associate_and_surfaces(unseeded, frames, {}, masks, stages, rebuild_surfaces=False)
    assert len(unseeded['entities']) == 3
    result = _associate_and_surfaces(document, frames, {}, masks, stages, rebuild_surfaces=False)
    assert result['sourceEquivalences']['verifiedPairCount'] == 1
    assert len(result['sourceEquivalences']['merges']) == 1
    assert len(document['entities']) == 1
    assert set(document['entities'][0]['observationRefs']) == set(masks)
    assert sorted(o['id'] for o in document['observations']) == sorted(o['id'] for o in before['observations'])
    assert document['assets'] == before['assets'] and not stages.repo.calls
    validate_document(document)


@pytest.mark.parametrize('changed', ['sha','pointer','instance','mask','image','native_image'])
def test_source_equivalence_verifier_rejects_mismatched_proof_inputs(tmp_path, changed):
    from ehs_spatial.platform.reconstruction import _verified_source_equivalences
    document, masks, stages, pair, save_proof, _ = source_equivalence_case(tmp_path, native_image_sha256='f'*64 if changed == 'native_image' else None)
    if changed == 'sha':
        pair['sourceRef']['sha256'] = 'f'*64
    elif changed == 'pointer':
        pair['evidenceRefs'][0]['jsonPointer'] = '/objects/999/views/0/provenance'
    elif changed == 'instance':
        pair['sourceRef']['jsonPointer'] = '/rle/1'
    elif changed == 'image':
        pair['imageSha256'] = 'f'*64
    elif changed == 'mask':
        masks['raw-observation'][0,2] = False
    save_proof(pair)
    before = deepcopy(document)
    with pytest.raises(PlatformError):
        _verified_source_equivalences(document, masks, stages)
    assert document == before and not stages.repo.calls


def test_source_equivalence_verifier_skips_revised_or_missing_observation_masks(tmp_path):
    from ehs_spatial.platform.reconstruction import _verified_source_equivalences
    document, masks, stages, _, _, _ = source_equivalence_case(tmp_path)
    document['observations'][0]['revision'] += 1
    verified, skipped = _verified_source_equivalences(document, masks, stages)
    assert not verified and skipped == [{'observationIds':['native-observation','raw-observation'],'code':'source_equivalence_observation_revised'}]
    document['observations'][0]['revision'] -= 1
    del masks['raw-observation']
    verified, skipped = _verified_source_equivalences(document, masks, stages)
    assert not verified and skipped[0]['code'] == 'source_equivalence_mask_unavailable'
    assert not stages.repo.calls


def test_source_equivalence_verifier_rejects_provenance_from_an_unrelated_object(tmp_path):
    from ehs_spatial.platform.reconstruction import _verified_source_equivalences
    document, masks, stages, pair, save_proof, _ = source_equivalence_case(tmp_path)
    # This pointer exists and repeats exactly the same SHA/instance/mask. It is
    # still not a provenance edge from either named observation.
    pair['evidenceRefs'][0]['jsonPointer'] = '/objects/1/views/0/provenance'
    save_proof(pair)
    with pytest.raises(PlatformError):
        _verified_source_equivalences(document, masks, stages)
    pair['evidenceRefs'][0]['jsonPointer'] = '/objects/0/views/0/provenance'
    save_proof(pair)
    document['observations'][1]['sourceRefs'][0]['sourceRecordId'] = 'different-raw-source-owner'
    with pytest.raises(PlatformError):
        _verified_source_equivalences(document, masks, stages)


def test_source_equivalence_verifier_rejects_nonobject_proof_document(tmp_path):
    from ehs_spatial.platform.reconstruction import _verified_source_equivalences
    document, masks, stages, _, _, _ = source_equivalence_case(tmp_path)
    asset = stages.put(b'[]', {'kind':'source_identity_evidence'}, 'application/json')
    document['assets'].append(asset)
    document['sourceIdentityEvidence'] = [{'assetId':asset['id'], 'sha256':asset['sha256']}]
    with pytest.raises(PlatformError, match='invalid_source_identity_proof'):
        _verified_source_equivalences(document, masks, stages)


def test_observed_meshing_rejects_degenerate_faces_without_losing_supported_objects(tmp_path):
    from ehs_spatial.platform.reconstruction import _mesh, _save_mesh, _include, run_reassociation
    from ehs_spatial.platform.spatial import MeshData

    valid, rgb = np.ones((2, 2), bool), np.full((2, 2, 3), 100, np.uint8)
    for scale in (1e-30, 1., 1e30):
        assert _mesh(np.zeros((2, 2, 3)), valid, rgb, valid) is None
        line = np.array([[[0., 0., 0.], [1., 0., 0.]], [[2., 0., 0.], [3., 0., 0.]]]) * scale
        assert _mesh(line, valid, rgb, valid) is None
        mixed = np.array([[[0., 0., 0.], [0., 0., 0.]], [[0., 1., 0.], [1., 1., 0.]]]) * scale
        mesh = _mesh(mixed, valid, rgb, valid)
        assert mesh is not None and mesh.faces.tolist() == [[0, 2, 1]]
        assert np.array_equal(mesh.vertices, mixed.reshape(-1, 3)[[1, 2, 3]].astype(np.float32))
    discontinuous = np.zeros((4, 4, 3))
    discontinuous[-1, -2:] = [[1., 0., 10.], [0., 1., 10.]]
    assert _mesh(discontinuous, np.ones((4, 4), bool), np.zeros((4, 4, 3)), np.ones((4, 4), bool)) is None
    y, x = np.mgrid[:4, :4]
    isolated = np.stack((x, y, np.full_like(x, 2)), axis=-1).astype(float)
    isolated[0, 0] = [1e20, 1e20, 1e20]
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = mask[2:, 2:] = True
    mesh = _mesh(isolated, mask, np.zeros((4, 4, 3)), mask)
    assert mesh is not None and len(mesh.vertices) == 4
    assert mesh.vertices.max(axis=0).tolist() == [3., 3., 2.]
    assert mesh.colors.shape == (4, 3)

    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = bundle(repo)
    providers['discovery'] = provider('discovery', lambda _: {'items': [
        {'label':'collapsed support', 'box':[0, 0, 5, 12]}, {'label':'supported', 'box':[7, 0, 12, 12]}]})
    def geometry(payload):
        response = geometry_response(payload['images'])
        for frame in response['frames']:
            frame['points'][:, :5] = [0., 0., 2.]
        return response
    providers['geometry'] = provider('geometry', geometry)
    source, analysis = run_analysis(repo, blobs, repo.job, providers)
    bad = [o for o in source['observations'] if o['originalPixelBox'][0] == 0]
    bad_ids = {o['id'] for o in bad}
    assert analysis['status'] == 'incomplete'
    assert {error['observationId'] for error in analysis['errors']} == bad_ids
    assert all(o['geometrySupport']['boundsNative'] is None and
               'observed_surface_unavailable' in o['missingEvidence'] for o in bad)
    bad_entities = [e for e in source['entities'] if set(e['observationRefs']) & bad_ids]
    assert all(not e['representations'] and not e['measurements'].get('observedBounds') for e in bad_entities)

    # A previous version could have persisted degenerate surfaces. Keep their
    # immutable bytes, but retire only the failed observation's observed reps.
    stages = _Stages(repo, blobs, repo.job, {})
    asset = _save_mesh(stages, MeshData(np.zeros((3, 3), np.float32), np.array([[0, 1, 2]], np.uint32)), {})
    _include(source, asset)
    good = next(r for e in source['entities'] if not e.get('sourceContext') for r in e['representations'])
    for entity in bad_entities:
        refs = [{'observationId':oid, 'revision':1} for oid in entity['observationRefs']]
        old = {**deepcopy(good), 'id':str(uuid4()), 'assetId':asset['id'], 'sourceRefs':refs,
               'bounds':{'min':[0., 0., 0.], 'max':[0., 0., 0.]}}
        generated = {**deepcopy(good), 'id':str(uuid4()), 'kind':'generated_mesh', 'sourceRefs':deepcopy(refs)}
        entity['representations'].extend([old, generated])
        entity['activeModelRepresentationId'] = generated['id']
        entity['currentModelTransform'] = deepcopy(generated['transform'])
    repo.document = deepcopy(source)
    call_count = len(repo.calls)
    job = {**repo.job, 'id':str(uuid4()), 'kind':'reassociate_scene', 'inputs':{'rebuildObservedSurfaces':True}}
    rebuilt, result = run_reassociation(repo, blobs, job, {})
    assert result['status'] == 'incomplete' and len(repo.calls) == call_count
    assert {error['observationId'] for error in result['errors']} == bad_ids
    assert result['surfaceRebuild']['observationCount'] == len(source['observations']) - len(bad_ids)
    assert {e['id'] for e in rebuilt['entities']} == {e['id'] for e in source['entities']}
    for before, after in zip(source['entities'], rebuilt['entities']):
        assert before['id'] == after['id']
        for key in ('observationRefs', 'measurements', 'measurementEvidence', 'measurementSelections',
                    'activeModelRepresentationId', 'currentModelTransform'):
            assert before[key] == after[key]
        for rep in after['representations']:
            if rep['kind'] == 'generated_mesh':
                assert rep == next(r for r in before['representations'] if r['id'] == rep['id'])
            elif not after.get('sourceContext'):
                assert (rep.get('sourceValidity') == 'stale') == bool(set(after['observationRefs']) & bad_ids)
    validate_document(rebuilt)


def test_full_observed_rebuild_preserves_identity_measurements_and_every_photo(tmp_path):
    from ehs_spatial.platform.reconstruction import run_reassociation
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = bundle(repo)
    providers['discovery'] = provider('discovery', lambda _: {'items': [{'label': 'fixture', 'box': [0, 0, 5, 12]}]})
    source, _ = run_analysis(repo, blobs, repo.job, providers)
    entity = next(e for e in source['entities'] if not e.get('sourceContext'))
    assert len(entity['observationRefs']) == 2
    assert {r['sourceRefs'][0]['observationId'] for r in entity['representations']} == set(entity['observationRefs'])
    # Simulate an imported carved mesh: retain its immutable asset, then replace
    # only its current representation selection with full native mask support.
    entity['representations'] = entity['representations'][:1]
    entity['representations'][0]['id'] = str(uuid4())
    generated = {**deepcopy(entity['representations'][0]), 'id': str(uuid4()), 'kind': 'generated_mesh'}
    entity['representations'].append(generated)
    entity['activeModelRepresentationId'] = generated['id']
    entity['currentModelTransform'] = deepcopy(generated['transform'])
    repo.document = deepcopy(source)
    calls = len(repo.calls)
    # Asset deduplication can return frozen import metadata without bounds.
    for asset in repo.assets:
        asset.get('metadata', {}).pop('bounds', None)
    job = {**repo.job, 'id': str(uuid4()), 'kind': 'reassociate_scene', 'inputs': {'rebuildObservedSurfaces': True}}
    rebuilt, result = run_reassociation(repo, blobs, job, {})
    assert result['status'] == 'succeeded' and result['newModelCalls'] == 0 and len(repo.calls) == calls
    assert result['surfaceRebuild']['observationCount'] == 2
    assert rebuilt['observations'] == source['observations'] and rebuilt['identityDecisions'] == source['identityDecisions']
    current = next(e for e in rebuilt['entities'] if e['id'] == entity['id'])
    for key in ('measurementEvidence', 'measurementSelections', 'measurements', 'activeModelRepresentationId', 'currentModelTransform'):
        assert current[key] == entity[key]
    assert next(r for r in current['representations'] if r['id'] == generated['id']) == generated
    old = next(r for r in current['representations'] if r['id'] == entity['representations'][0]['id'])
    assert old['sourceValidity'] == 'stale' and old['assetId'] == entity['representations'][0]['assetId']
    surfaces = [r for r in current['representations'] if r['kind'] == 'observed_surface' and r.get('sourceValidity') != 'stale']
    assert {r['sourceRefs'][0]['observationId'] for r in surfaces} == set(entity['observationRefs'])
    contexts = [r for e in rebuilt['entities'] if e.get('sourceContext') for r in e['representations'] if r['kind'] == 'observed_surface' and r.get('sourceValidity') != 'stale']
    assert {r['sourceRefs'][0]['imageId'] for r in contexts} == {o['imageId'] for o in source['observations']}
    assert all(a in rebuilt['assets'] for a in source['assets'])
    validate_document(rebuilt)
    repo.document = deepcopy(rebuilt)
    repeated, _ = run_reassociation(repo, blobs, job, {})
    assert [e['representations'] for e in repeated['entities']] == [e['representations'] for e in rebuilt['entities']]
    # An incomplete source cannot retire usable geometry or claim full coverage.
    repo.document['observations'][0]['maskAssetId'] = None
    unchanged, failed = run_reassociation(repo, blobs, job, {})
    assert failed['status'] == 'incomplete'
    assert unchanged['entities'] == repo.document['entities']


@pytest.mark.parametrize('cells,parts,holes', [
    ([(0, 0), (1, 0), (0, 1)], 1, 0),
    ([(x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1)], 1, 1),
    ([(0, 0), (3, 0)], 2, 0),
])
def test_observed_plan_uses_indexed_triangles_preserving_concavity_holes_and_islands(tmp_path, monkeypatch, cells, parts, holes):
    from types import SimpleNamespace
    from shapely import Polygon, union_all
    from ehs_spatial.platform import reconstruction
    from ehs_spatial.platform.spatial import MeshData

    vertices, faces = [], []
    for x, y in cells:
        i = len(vertices)
        vertices.extend([(x, y, 0), (x + 1, y, 0), (x + 1, y + 1, 0), (x, y + 1, 0)])
        faces.extend([(i, i + 1, i + 2), (i, i + 2, i + 3)])
    vertices.append((100, 100, 100))  # Retained in the asset but never drawn.
    mesh = MeshData(np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32), None)
    monkeypatch.setattr(reconstruction, '_mesh', lambda *args: mesh)
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    document = deepcopy(repo.document)
    frame_id = str(uuid4())
    document['coordinateFrames'] = [{'id': frame_id, 'ground': {'normal': [0., 0., 1.]}}]
    document['reportEvidence'] = {'plan': {'coordinateFrameId': frame_id, 'nativeToFloor': np.eye(4).tolist()}}
    observation = {'id': str(uuid4()), 'revision': 2}
    frame = SimpleNamespace(image_id=repo.capture['images'][0]['id'], coordinate_frame_id=frame_id, points=None, support=lambda: None)
    entity = {'id': str(uuid4()), 'representations': []}
    reconstruction._save_observed_surface(document, entity, frame, {'rgb': None}, None, _Stages(repo, blobs, repo.job, {}), observation)
    rep = entity['representations'][0]
    saved = rep['planProjection']
    polygons = [Polygon(p['exterior'], p['holes']) for p in saved['polygons']]
    assert len(polygons) == parts and sum(len(p.interiors) for p in polygons) == holes
    assert union_all(polygons).area == pytest.approx(len(cells))
    assert rep['bounds']['max'] == [max(x for x, _ in cells) + 1, max(y for _, y in cells) + 1, 0]
    assert saved['assetSha256'] == repo.get_asset(rep['assetId'])['sha256']
    assert saved['assetId'] == rep['assetId'] and saved['imageId'] == frame.image_id
    assert saved['observationId'] == observation['id'] and saved['observationRevision'] == 2
    assert saved['transformSnapshot'] == rep['transform'] and saved['nativeToPlane'] == np.eye(4).tolist()
    if len(cells) == 3:
        from shapely import Point
        assert not union_all(polygons).covers(Point(1.5, 1.5))
    pose = {**deepcopy(rep['transform']), 'position': [4., 5., 0.], 'scale': [2., 3., 1.],
            'quaternion': [0., 0., 2 ** -.5, 2 ** -.5]}
    moved = reconstruction._plan_projection(document, rep, mesh, saved['assetSha256'], pose)
    moved_union = union_all([Polygon(p['exterior'], p['holes']) for p in moved['polygons']])
    assert moved_union.area == pytest.approx(len(cells) * 6)
    assert moved_union.bounds == pytest.approx([4 - 3 * (max(y for _, y in cells) + 1), 5, 4, 5 + 2 * (max(x for x, _ in cells) + 1)])
    assert moved['transformSnapshot'] == pose and rep['transform'] == saved['transformSnapshot']


def test_plan_projection_keeps_vertical_edges_and_current_model_pose_without_confirming_it(tmp_path):
    from shapely import LineString, Polygon, union_all
    from ehs_spatial.platform.reconstruction import _plan_projection, _refresh_plan_projections
    from ehs_spatial.platform.spatial import MeshData

    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    doc = deepcopy(repo.document)
    frame_id = str(uuid4())
    doc['coordinateFrames'] = [{'id': frame_id, 'ground': {'normal': [0., 0., 1.]}}]
    doc['reportEvidence'] = {'plan': {'coordinateFrameId': frame_id, 'nativeToFloor': np.eye(4).tolist()}}
    pose = {'coordinateFrameId': frame_id, 'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.], 'scale': [1., 1., 1.]}
    rep = {'id': str(uuid4()), 'kind': 'primitive', 'assetId': None, 'coordinateFrameId': frame_id,
           'transform': pose, 'primitive': {'type': 'box', 'dimensions': [1., 2., 3.]}, 'placementState': 'unconfirmed', 'sourceRefs': []}
    wall = MeshData(np.array([[2., 0., 0.], [2., 3., 0.], [2., 3., 4.], [2., 0., 4.]]), np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32))
    flat = _plan_projection(doc, rep, wall, None)
    assert flat['polygons'] == []
    assert union_all([LineString(line) for line in flat['lines']]).equals(LineString([(2, 0), (2, 3)]))
    inactive = {**deepcopy(rep), 'id': str(uuid4())}
    current = {**deepcopy(pose), 'position': [10., 20., 0.], 'scale': [2., 3., 1.]}
    entity = {'id': str(uuid4()), 'representations': [rep, inactive], 'activeModelRepresentationId': rep['id'], 'currentModelTransform': current}
    doc['entities'] = [entity]
    before = deepcopy(entity)
    _refresh_plan_projections(doc, _Stages(repo, blobs, repo.job, {}))
    saved = rep['planProjection']
    shape = union_all([Polygon(p['exterior'], p['holes']) for p in saved['polygons']])
    assert shape.bounds == pytest.approx([9, 17, 11, 23]) and shape.area == pytest.approx(12)
    assert saved['transformSnapshot'] == current and saved['primitiveSnapshot'] == rep['primitive']
    assert {k: v for k, v in rep.items() if k != 'planProjection'} == before['representations'][0]
    assert inactive == before['representations'][1] and entity['currentModelTransform'] == before['currentModelTransform']
    doc['coordinateFrames'][0]['ground'] = None
    _refresh_plan_projections(doc, _Stages(repo, blobs, repo.job, {}))
    assert 'planProjection' not in rep


def test_projection_refresh_uses_frozen_scene_mesh_layout_and_checks_hash(tmp_path):
    from ehs_spatial.platform.reconstruction import _save_mesh, _include, _refresh_plan_projections
    from ehs_spatial.platform.spatial import primitive_mesh
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    stages = _Stages(repo, blobs, repo.job, {})
    asset = _save_mesh(stages, primitive_mesh({'type':'box', 'dimensions':[1.,2.,3.]}), {})
    doc = deepcopy(repo.document)
    _include(doc, asset)
    # Partition assets retain their layout in the immutable scene declaration.
    next(a for a in repo.assets if a['id'] == asset['id'])['metadata'] = {}
    frame_id = str(uuid4())
    doc['coordinateFrames'] = [{'id':frame_id, 'ground':{'normal':[0.,0.,1.]}}]
    doc['reportEvidence'] = {'plan':{'coordinateFrameId':frame_id, 'nativeToFloor':np.eye(4).tolist()}}
    pose = {'coordinateFrameId':frame_id, 'position':[0.,0.,0.], 'quaternion':[0.,0.,0.,1.], 'scale':[1.,1.,1.]}
    rep = {'id':str(uuid4()), 'kind':'generated_mesh', 'assetId':asset['id'], 'coordinateFrameId':frame_id, 'transform':pose}
    doc['entities'] = [{'id':str(uuid4()), 'representations':[rep], 'activeModelRepresentationId':rep['id'], 'currentModelTransform':pose}]
    _refresh_plan_projections(doc, stages)
    assert rep['planProjection']['polygons']
    assert rep['planProjection']['assetSha256'] == asset['sha256']
    next(a for a in doc['assets'] if a['id'] == asset['id'])['sha256'] = '0' * 64
    with pytest.raises(PlatformError, match='scene_asset_hash_mismatch'):
        _refresh_plan_projections(doc, stages)


def test_scene_cad_references_preserve_sources_and_do_not_rebuild_geometry(tmp_path, monkeypatch):
    from ehs_spatial.platform import reconstruction
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    doc, _ = run_analysis(repo, blobs, repo.job, bundle(repo))
    for entity in doc['entities']:
        entity.pop('cadReference', None)
    own = [e for e in doc['entities'] if not e.get('sourceContext')]
    linked = next(e for e in own if len(e['observationRefs']) == 2)
    observations = {o['id']: o for o in doc['observations']}
    preferred = repo.capture['images'][0]['id']
    anchor = next(observations[oid] for oid in linked['observationRefs'] if observations[oid]['imageId'] != preferred)
    model = {**deepcopy(linked['representations'][0]), 'id': str(uuid4()), 'kind': 'generated_mesh',
             'sourceRefs': [{'observationId': anchor['id'], 'revision': anchor['revision']}]}
    linked['representations'].append(model)
    linked.update(activeModelRepresentationId=model['id'], currentModelTransform=deepcopy(model['transform']))
    before = deepcopy(doc)
    stages = _Stages(repo, blobs, repo.job, {})
    audit = reconstruction._establish_cad_references(doc, stages, preferred)
    assert audit['unresolvedEntityCount'] == 0 and audit['resolvedEntityCount'] == len(own)
    assert linked['cadReference']['referenceImageId'] == anchor['imageId']
    assert linked['cadReference']['source'] == 'existing_source_reference'
    for old, current in zip(before['entities'], doc['entities']):
        assert {k: v for k, v in current.items() if k != 'cadReference'} == old
        if not current.get('sourceContext'):
            ref = current['cadReference']
            assert ref['sourceRefs'] == [{'observationId': oid, 'revision': observations[oid]['revision']}
                for oid in current['observationRefs'] if observations[oid]['imageId'] == ref['referenceImageId']]
    unresolved = deepcopy(before)
    for entity in unresolved['entities']:
        entity['activeModelRepresentationId'] = None
    uncertain = reconstruction._establish_cad_references(unresolved, stages)
    assert uncertain['unresolvedEntityCount'] == 2
    assert all(e['cadReference']['referenceImageId'] is None for e in unresolved['entities'] if len(e['observationRefs']) == 2)
    repo.document = deepcopy(doc)
    monkeypatch.setattr(reconstruction, '_load_geometry', lambda *args: pytest.fail('Reference selection cannot rebuild geometry'))
    monkeypatch.setattr(reconstruction, '_refresh_plan_projections', lambda *args, **kwargs: pytest.fail('Existing CAD projections must be reused'))
    updated, result = reconstruction.run_reassociation(repo, blobs, {**repo.job, 'kind': 'reassociate_scene',
        'inputs': {'establishCadReferences': True, 'referenceImageId': preferred}}, {})
    assert result['status'] == 'succeeded' and result['cadReferences']['resolvedEntityCount'] == len(own)
    assert updated['entities'] == doc['entities'] and updated['observations'] == doc['observations']
    validate_document(updated)
    with pytest.raises(PlatformError, match='cad_reference_image_not_found'):
        reconstruction._establish_cad_references(doc, stages, str(uuid4()))
    # Imported model references must bind the source camera in the same artifact.
    imported = deepcopy(before)
    raw = stages.put({'objects': [{'id': 'source-model', 'reference_frame': 'source-camera'}]}, {'kind': 'test_source'})
    imported['assets'].append(raw)
    imported_model = next(r for e in imported['entities'] for r in e['representations'] if r['id'] == model['id'])
    imported_model['sourceRefs'] = [{'assetId': raw['id'], 'sourceRecordId': 'source-model'}]
    next(c for c in imported['cameras'] if c['imageId'] == anchor['imageId'])['sourceRefs'].append({'assetId': raw['id'], 'sourceCameraId': 'source-camera'})
    reconstruction._establish_cad_references(imported, stages, preferred)
    assert next(e for e in imported['entities'] if e['id'] == linked['id'])['cadReference']['referenceImageId'] == anchor['imageId']


def test_cad_reference_job_pins_same_project_source_manifest(tmp_path, monkeypatch):
    from ehs_spatial.platform import reconstruction, source_cad
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    repo.document, _ = run_analysis(repo, blobs, repo.job, bundle(repo))
    stages = _Stages(repo, blobs, repo.job, {})
    inventory = stages.put({'objects': []}, {'kind': 'source_inventory'})
    manifest = stages.put({'files': []}, {'kind': 'source_cad_segmentation_manifest'})
    repo.document['reportEvidence'] = {'historical': {'inventoryAssetId': inventory['id']}}
    calls = []
    def refresh(document, read_asset, masks, *, inventory_asset_id=None, source_manifest_asset_id=None):
        assert {inventory_asset_id, source_manifest_asset_id} <= {a['id'] for a in document['assets']}
        assert json.loads(read_asset(source_manifest_asset_id)) == {'files': []}
        calls.append((inventory_asset_id, source_manifest_asset_id))
        return {'linkedRecordCount': 0}
    monkeypatch.setattr(source_cad, 'refresh_source_cad_links', refresh)
    monkeypatch.setattr(reconstruction, '_load_geometry', lambda *args: ({}, {}))
    monkeypatch.setattr(reconstruction, '_load_masks', lambda *args: ({}, []))
    job = {**repo.job, 'kind': 'reassociate_scene', 'inputs': {'establishCadReferences': True,
        'sourceCadManifestAssetId': manifest['id']}}
    _, result = reconstruction.run_reassociation(repo, blobs, job, {})
    assert calls == [(inventory['id'], manifest['id'])]
    assert result['sourceCadCoverage'] == {'linkedRecordCount': 0}
    next(a for a in repo.assets if a['id'] == manifest['id'])['projectId'] = str(uuid4())
    with pytest.raises(PlatformError, match='source_cad_manifest_scope_mismatch'):
        reconstruction.run_reassociation(repo, blobs, job, {})
    assert len(calls) == 1
