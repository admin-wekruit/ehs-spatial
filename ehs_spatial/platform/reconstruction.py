"""Generic capture stages with durable evidence and no provider fallback.

Paid calls are reserved before dispatch. A transport failure is outcome_unknown,
never an invitation to submit another paid call. Historical experiments remain
separate from this corrected, per-image cache protocol.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Literal, Mapping
from uuid import UUID, uuid5

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .contracts import PlatformError, canonical, digest
from .spatial import (FrameGeometry, MaskObservation, MeshData, SAM3DMeshAdapter,
                      GenerationRequest, associate_observations, camera_intrinsics,
                      stage_cache_key, transform_points, matrix_to_transform,estimate_native_ground)

PIPELINE_VERSION = "capture-v1-per-image-cache"
MAP_PINS = {"model": "facebook/map-anything-apache", "modelRevision": "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a",
            "codeRevision": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9", "adapter": PIPELINE_VERSION}
MAX_MASK_POLYGON_RUNS = 100_000


def _packed(value):
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        return {"__ndarray__": True, "dtype": str(a.dtype), "shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode()}
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode()}
    if isinstance(value, dict):
        return {k: _packed(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [_packed(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unpacked(value):
    if isinstance(value, dict) and value.get("__ndarray__"):
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject or dtype.kind not in "biuf":
            raise PlatformError("invalid_provider_array")
        raw = base64.b64decode(value["data"],validate=True)
        shape = value["shape"]
        if not isinstance(shape,list) or any(type(x) is not int or x < 0 for x in shape) or len(shape) > 4 or int(np.prod(shape)) * dtype.itemsize != len(raw):
            raise PlatformError("invalid_provider_array")
        return np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
    if isinstance(value, dict) and "__bytes__" in value:
        return base64.b64decode(value["__bytes__"],validate=True)
    if isinstance(value, dict):
        return {k:_unpacked(v) for k,v in value.items()}
    if isinstance(value,list):
        return [_unpacked(v) for v in value]
    return value


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    pins: Mapping[str,str]
    invoke: Callable[[dict],dict]
    estimated_cost_usd: float
    release_evidence: Mapping[str,Any]
    paid: bool = True

    def validate(self, stage: str, *, require_quality=True):
        evidence = self.release_evidence
        if not self.pins or any(not isinstance(v,str) or not v for v in self.pins.values()) or evidence.get("pins") != dict(self.pins):
            raise PlatformError("provider_pins_unverified",409,stage=stage)
        for gate in (("license","quality","runtime") if require_quality else ("license","runtime")):
            record = evidence.get(gate,{})
            if record.get("status") != "passed" or len(record.get("artifactSha256","")) != 64:
                raise PlatformError("provider_release_gate_unverified",409,stage=stage,gate=gate)
        if stage == "geometry" and dict(self.pins) != MAP_PINS:
            raise PlatformError("geometry_requires_apache_weights",409)
        if stage == "depth" and self.pins.get("model") != "Ruicheng/moge-3-vitl":
            raise PlatformError("depth_requires_moge3",409)


class ProviderResponseError(Exception):
    """A received response failed decoding; its measured usage is still billable."""
    def __init__(self, telemetry, *, outcome="failed"):
        self.telemetry,self.outcome = telemetry,outcome
        super().__init__("provider_response_invalid")


def _telemetry(response):
    """Only provider-reported usage/cost, never a price inferred from elapsed time."""
    value = response.get("telemetry",{}) if isinstance(response,dict) else {}
    result = {k:value.get(k) for k in ("usage","gpuElapsedSeconds","workerElapsedSeconds","timingMethod","actualCostUsd")}
    result["providerRequestId"] = response.get("providerRequestId") if isinstance(response,dict) else None
    cost = result["actualCostUsd"]
    if cost is not None and (isinstance(cost,bool) or not isinstance(cost,(int,float)) or not np.isfinite(cost) or cost < 0):
        result["actualCostUsd"] = None
    return result


class _Stages:
    def __init__(self, repository, blobs, job, providers):
        self.repo,self.blobs,self.job,self.providers = repository,blobs,job,providers
        self.records, self.assets = [], repository.list_project_records(job["projectId"],"assets")["items"]

    def put(self, value, metadata, media_type="application/json"):
        data = value if isinstance(value,bytes) else canonical(_packed(value))
        blob = self.blobs.put(data,media_type)
        blob["metadata"] = metadata
        asset = self.repo.register_asset(self.job["projectId"],blob,self.job["id"])
        self.assets.append(asset)
        return asset

    def load(self, asset):
        return _unpacked(json.loads(self.blobs.get(asset["storageKey"],asset["sha256"],asset["sizeBytes"])))

    def checkpoint(self, document, stage):
        return self.put(document,{"kind":"analysis_checkpoint","captureId":document["captureId"],"stage":stage,"pipelineVersion":PIPELINE_VERSION})

    def call(self, stage, images, payload, refs=(), *, research_protocol=None):
        provider = self.providers.get(stage)
        if provider is None:
            raise PlatformError("provider_not_configured",409,stage=stage)
        provider.validate(stage,require_quality=research_protocol is None)
        key = stage_cache_key(stage,[{"imageId":x["id"],"sha256":x["sha256"],"pixelMapping":x.get("pixelMapping",[])} for x in images],provider.pins,{"payloadSha256":digest(_packed(payload))},refs)
        cached = next((a for a in self.assets if a.get("metadata",{}).get("kind") == "stage_cache" and a["metadata"].get("cacheKey") == key),None)
        if cached:
            self.records.append({"stage":stage,"cacheKey":key,"status":"cached","assetId":cached["id"],"newModelCalls":0})
            envelope = self.load(cached)
            if envelope.get("cacheKey") != key or envelope.get("stage") != stage:
                raise PlatformError("stage_cache_integrity_error",409)
            if envelope["output"].get("providerError"):
                raise PlatformError("provider_failed",502,stage=stage)
            return envelope["output"],cached
        call = self.repo.reserve_model_call(self.job["id"],self.job["attemptToken"],provider.name,provider.pins["model"],key,provider.estimated_cost_usd,
            code_sha256=digest({"code":provider.pins.get("codeRevision")}),model_sha256=digest({"model":provider.pins}),adapter_sha256=digest({"adapter":PIPELINE_VERSION}),input_sha256=digest(_packed(payload)),paid=provider.paid)
        started = time.monotonic()
        try:
            result = provider.invoke(payload)
            if not isinstance(result,dict):
                raise ProviderResponseError(_telemetry({}))
        except Exception as exc:
            metadata = exc.telemetry if isinstance(exc,ProviderResponseError) else _telemetry({})
            outcome = exc.outcome if isinstance(exc,ProviderResponseError) else "outcome_unknown"
            self.repo.complete_model_call(call["id"],outcome,actual_cost=metadata.get("actualCostUsd"),response={**metadata,"elapsedSeconds":time.monotonic()-started,"error":{"code":"provider_response_invalid" if outcome == "failed" else "provider_outcome_unknown"},"stage":stage})
            raise PlatformError("provider_response_invalid" if outcome == "failed" else "provider_outcome_unknown",502,stage=stage) from None
        metadata = _telemetry(result)
        # A response has arrived; even invalid data must never trigger a duplicate call.
        try:
            asset = self.put({"stage":stage,"cacheKey":key,"output":result},{"kind":"stage_cache","stage":stage,"cacheKey":key,"providerPins":dict(provider.pins),"pipelineVersion":PIPELINE_VERSION})
        except Exception:
            self.repo.complete_model_call(call["id"],"outcome_unknown",actual_cost=metadata.get("actualCostUsd"),response={**metadata,"elapsedSeconds":time.monotonic()-started,"error":{"code":"response_persistence_failed"},"stage":stage})
            raise PlatformError("response_persistence_failed",502,stage=stage) from None
        elapsed = time.monotonic()-started
        status = "failed" if result.get("providerError") else "succeeded"
        self.repo.complete_model_call(call["id"],status,actual_cost=metadata.get("actualCostUsd"),response={**metadata,"assetId":asset["id"],"elapsedSeconds":elapsed})
        self.records.append({**metadata,"stage":stage,"cacheKey":key,"status":status,"assetId":asset["id"],"elapsedSeconds":elapsed,"newModelCalls":1})
        if status == "failed":
            raise PlatformError("provider_failed",502,stage=stage)
        return result,asset


def _capture(repository,blobs,job):
    document = repository.get_revision(job["baseRevisionId"])["document"]
    capture_id = job["inputs"].get("captureId") or document.get("captureId")
    capture = next((x for x in repository.list_project_records(job["projectId"],"captures")["items"] if x["id"] == capture_id),None)
    if capture is None or not 1 <= len(capture["images"]) <= 4:
        raise PlatformError("capture_not_found",404)
    images = []
    for item in capture["images"]:
        asset = repository.get_asset(item["assetId"])
        if asset["projectId"] != job["projectId"]:
            raise PlatformError("asset_project_mismatch",403)
        raw = blobs.get(asset["storageKey"],asset["sha256"],asset["sizeBytes"])
        with Image.open(io.BytesIO(raw)) as im:
            rgb = np.asarray(im.convert("RGB"))
        images.append({**item,"sha256":asset["sha256"],"bytes":raw,"rgb":rgb,"width":rgb.shape[1],"height":rgb.shape[0]})
    return capture,json.loads(canonical(document)),images


def _id(capture_id,*parts):
    return str(uuid5(UUID(capture_id),":".join(map(str,parts))))


def _ref(asset):
    return {"assetId":asset["id"],"sha256":asset["sha256"]}


def _include(document,asset):
    if asset["id"] not in {a["id"] for a in document["assets"]}:
        document["assets"].append(asset)


def _image_payload(image):
    return {"imageId":image["id"],"dataUri":"data:image/png;base64,"+base64.b64encode(image["bytes"]).decode(),"width":image["width"],"height":image["height"],"sha256":image["sha256"]}


def _discover(document,image,response,evidence):
    _include(document,evidence)
    items = response.get("items")
    if not isinstance(items,list):
        raise PlatformError("invalid_discovery_response")
    for i,item in enumerate(items):
        box = np.asarray(item.get("box"),dtype=float)
        label = item.get("label")
        geometry_role = item.get("geometryRole","unknown")
        if geometry_role not in ("floor","object","unknown"):
            raise PlatformError("invalid_discovery_geometry_role")
        if box.shape != (4,) or not np.isfinite(box).all() or min(box) < 0 or box[0] >= box[2] or box[1] >= box[3] or box[2] > image["width"] or box[3] > image["height"] or not isinstance(label,str) or not label.strip():
            raise PlatformError("invalid_discovery_item")
        oid = _id(document["captureId"],"observation",image["id"],evidence["sha256"],i)
        if any(x["id"] == oid for x in document["observations"]):
            continue
        document["observations"].append({"id":oid,"revision":1,"imageId":image["id"],"originalPixelBox":box.tolist(),"maskAssetId":None,
            "pixelMapping":image.get("pixelMapping",[]),"labelEvidence":[{"label":label,"sourceRefs":[_ref(evidence)],"kind":"visual_hypothesis","geometryRole":geometry_role}],"geometrySupport":None,"sourceRefs":[_ref(evidence)]})
        document["entities"].append({"id":_id(document["captureId"],"entity",oid),"label":label,"observationRefs":[oid],
            "associationState":"association_pending","representations":[],"currentModelTransform":None,"measurements":{},
            "groupId":None,"lineage":[]})


def _geometry(document,images,response,evidence):
    records = response.get("frames",[])
    if len(records) != len(images) or {x.get("imageId") for x in records} != {x["id"] for x in images}:
        raise PlatformError("geometry_frame_count_mismatch")
    frame_id = _id(document["captureId"],"native")
    existing = next((f for f in document["coordinateFrames"] if f["id"] == frame_id),None)
    if existing is not None and existing.get("sourceRefs") != [_ref(evidence)]:
        raise PlatformError("capture_geometry_already_bound",409)
    frames,canonical = {},{}
    cameras = []
    lookup = {x["id"]:x for x in images}
    for record in records:
        image = lookup[record["imageId"]]
        a = np.asarray(record["inputToCanonical"],dtype=float)
        if a.shape != (3,3) or not np.isfinite(a).all() or not np.allclose(a[2],[0,0,1]) or abs(np.linalg.det(a)) < 1e-10:
            raise PlatformError("invalid_geometry_pixel_mapping")
        points,valid = np.asarray(record["points"]),np.asarray(record["valid"],dtype=bool)
        f = FrameGeometry(image["id"],frame_id,image["sha256"],points,valid,np.asarray(record["K"]),np.asarray(record["cameraToWorld"]))
        rgb = np.asarray(record["rgb"])
        if rgb.shape != points.shape or rgb.dtype != np.uint8:
            raise PlatformError("canonical_rgb_mismatch")
        frames[image["id"]] = f
        canonical[image["id"]] = {**record,"rgb":rgb,"inputToCanonical":a}
        camera = {"id":_id(document["captureId"],"camera",image["id"]),"imageId":image["id"],"coordinateFrameId":frame_id,
                  "width":image["width"],"height":image["height"],"K":camera_intrinsics(np.linalg.inv(a)@f.K).tolist(),"cameraToWorld":f.camera_to_world.tolist(),"sourceRefs":[_ref(evidence)]}
        cameras.append(camera)
    # Validate the entire joint solution before binding immutable capture cameras.
    _include(document,evidence)
    if existing is None:
        document["coordinateFrames"].append({"id":frame_id,"convention":"opencv","scale":{"status":"uncalibrated","nativeToMeters":None,"sourceRefs":[]},"ground":None,"sourceRefs":[_ref(evidence)]})
    document["cameras"] = [c for c in document["cameras"] if c["imageId"] not in lookup] + cameras
    return frames,canonical


def _canonical_mask(mask,record):
    shape = np.asarray(record["points"]).shape[:2]
    y,x = np.mgrid[:shape[0],:shape[1]]
    inverse = np.linalg.inv(record["inputToCanonical"])
    xy = np.stack((x,y,np.ones_like(x)),axis=-1)@inverse.T
    xy = np.floor(xy[...,:2]+.5).astype(int)
    valid = (xy[...,0]>=0)&(xy[...,0]<mask.shape[1])&(xy[...,1]>=0)&(xy[...,1]<mask.shape[0])
    result = np.zeros(shape,bool)
    result[valid] = mask[xy[...,1][valid],xy[...,0][valid]]
    return result


def _original_mask_polygons(mask):
    from shapely import box, get_parts, union_all

    edges = np.diff(np.pad(mask.astype(np.int8),((0,0),(1,1))),axis=1)
    count = int(np.count_nonzero(edges == 1))
    metadata = {"methodVersion":"pixel-runs-union-v1","runCount":count,"status":"complete"}
    # ponytail: bound GEOS allocation at 100k row runs; larger masks keep their
    # complete PNG. Streaming contour extraction can lift this ceiling later.
    if count > MAX_MASK_POLYGON_RUNS:
        return None,{**metadata,"status":"complexity_limit","maxRuns":MAX_MASK_POLYGON_RUNS}
    if not count:
        return [],metadata
    rows,starts = np.where(edges == 1)
    _,ends = np.where(edges == -1)
    geometry = union_all(box(starts,rows,ends,rows+1))
    # Union exact pixel cells, with no hull, area threshold or simplification.
    rings = [np.asarray(ring.coords).tolist() for polygon in get_parts(geometry)
             for ring in (polygon.exterior,*polygon.interiors)]
    return rings,metadata


def _save_observation_mask(document,observation,image,response,evidence,stages):
    mask = np.asarray(response.get("mask"),dtype=bool)
    if mask.shape != (image["height"],image["width"]):
        raise PlatformError("mask_image_grid_mismatch",observationId=observation["id"])
    stream = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8)*255).save(stream,format="PNG")
    asset = stages.put(stream.getvalue(),{"kind":"observation_mask","observationId":observation["id"],"sourceRefs":[_ref(evidence)]},"image/png")
    polygons,polygonization = _original_mask_polygons(mask)
    observation.update(maskAssetId=asset["id"],maskStatus="present" if mask.any() else "empty",geometrySupport=None,
        maskPolygonization=polygonization,polygonCoordinateConvention="pixel_edges",fillRule="evenodd")
    if polygons is None:
        observation.pop("originalPixelPolygons",None)
    else:
        observation["originalPixelPolygons"] = polygons
    missing = [x for x in observation.get("missingEvidence",[]) if x not in ("segmentation_empty","mask_polygon_complexity_limit")]
    if not mask.any():
        missing.append("segmentation_empty")
    if polygons is None:
        missing.append("mask_polygon_complexity_limit")
    observation["missingEvidence"] = missing
    observation["sourceRefs"].append(_ref(evidence))
    _include(document,asset)
    # The retained box is discovery evidence, including for empty/complex masks.
    return mask


def _mesh(points,valid,rgb,mask):
    keep = valid & mask & np.isfinite(points).all(axis=-1)
    index = np.full(keep.shape,-1,dtype=np.int64)
    index[keep] = np.arange(keep.sum())
    corners = [index[:-1,:-1],index[:-1,1:],index[1:,:-1],index[1:,1:]]
    faces = np.concatenate([np.stack((corners[0],corners[1],corners[2]),axis=-1).reshape(-1,3),np.stack((corners[1],corners[3],corners[2]),axis=-1).reshape(-1,3)])
    faces = faces[(faces>=0).all(axis=1)]
    vertices = points[keep]
    if not len(faces):
        return None
    # Same observed-surface principle as research assembly: no hole/back completion.
    lengths = np.linalg.norm(vertices[faces]-vertices[np.roll(faces,1,axis=1)],axis=-1).max(axis=1)
    spacing = np.median(lengths)
    faces = faces[lengths <= max(spacing*4,1e-12)]
    return MeshData(vertices.astype(np.float32),faces.astype(np.uint32),rgb[keep].astype(np.float32)/255) if len(faces) else None


def _save_mesh(stages,mesh,metadata):
    normals = np.zeros_like(mesh.vertices,dtype=np.float32)
    triangles = mesh.vertices[mesh.faces]
    normal = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    for i in range(3):
        np.add.at(normals,mesh.faces[:,i],normal)
    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
    colors = mesh.colors if mesh.colors is not None else np.ones_like(mesh.vertices)
    vertices = np.column_stack((mesh.vertices,normals,colors)).astype("<f4")
    indices = np.asarray(mesh.faces,dtype="<u4").ravel()
    layout = {"stride":9,"byteOffset":0,"vertexCount":len(vertices),"indexByteOffset":vertices.nbytes,"indexCount":len(indices),"indexType":"uint32"}
    bounds = {"min":mesh.vertices.min(axis=0).tolist(),"max":mesh.vertices.max(axis=0).tolist()}
    asset = stages.put(vertices.tobytes()+indices.tobytes(),{**metadata,"format":"panoptes-mesh-v1","byteLayout":layout,"bounds":bounds},"application/octet-stream")
    return {**asset,"format":"panoptes-mesh-v1","byteLayout":layout}


def _association_evidence(document, associations, frames, masks):
    """Record why identity is unresolved without conflating it with object discovery."""
    lookup = {o["id"]:o for o in document["observations"]}
    scene_images = {o["imageId"] for o in lookup.values()} | {c["imageId"] for c in document["cameras"]}
    for entity in document["entities"]:
        ids = set(entity.get("observationRefs", []))
        if not ids or entity.get("sourceContext"):
            continue
        if entity["associationState"] == "confirmed":
            # Explicit imported/manual ownership survives a later geometry attempt.
            if entity.get("associationEvidence", {}).get("method") != "bidirectional_depth_mask" and entity.get("associationEvidence"):
                continue
            status = "confirmed"
        elif len(scene_images) <= 1:
            status = "single_view"
        elif not associations:
            status = "geometry_missing" if any(lookup[oid]["imageId"] not in frames for oid in ids) else "not_evaluated"
        elif any(oid not in masks for oid in ids):
            status = "mask_missing"
        elif any(lookup[oid]["imageId"] not in frames for oid in ids):
            status = "geometry_missing"
        elif any(set(link["observationIds"]) & ids for link in associations["ambiguous"]):
            status = "competing_candidates"
        elif not any(lookup[oid].get("geometrySupport", {}).get("validPixelCount", 0) >= associations["config"]["min_support"] for oid in ids):
            status = "insufficient_support"
        else:
            status = "no_supported_match"
        links = [link for link in (associations or {}).get("links", []) if set(link["observationIds"]) & ids and link["eligible"]]
        entity["associationEvidence"] = {"method":"bidirectional_depth_mask", "status":status,
            "observationIds":sorted(ids), "candidates":links, "config":(associations or {}).get("config"),
            "sourceRefs":[{"observationId":oid,"revision":lookup[oid].get("revision")} for oid in sorted(ids)],
            "meaning":"Geometric identity evidence; not semantic correctness or verified physical truth"}


def _associate_and_surfaces(document,frames,canonical,masks,stages):
    observations = [MaskObservation(o["id"],o["imageId"],masks[o["id"]]) for o in document["observations"] if o["id"] in masks]
    associations = associate_observations(observations,frames)
    # Existing durable IDs survive merges; no identity is recomputed from a mask.
    for group in associations["groups"]:
        entities = [e for e in document["entities"] if set(e["observationRefs"]) & set(group)]
        if len(group) > 1 and len(entities) > 1:
            survivor = entities[0]
            if any(not set(e["observationRefs"]) <= set(group) for e in entities):
                continue
            survivor["observationRefs"] = list(group)
            survivor["associationState"] = "confirmed"
            survivor["lineage"] += [{"type":"merge","entityIds":[e["id"] for e in entities[1:]],"method":"bidirectional_depth_mask"}]
            document["entities"] = [e for e in document["entities"] if e not in entities[1:]]
    lookup = {o["id"]:o for o in document["observations"]}
    for entity in document["entities"]:
        candidates = []
        for oid in entity["observationRefs"]:
            observation = lookup[oid]
            if oid not in masks or observation["imageId"] not in frames:
                continue
            f = frames[observation["imageId"]]
            support = masks[oid] & f.support()
            points = f.points[support]
            frame_record = next(x for x in document["coordinateFrames"] if x["id"] == f.coordinate_frame_id)
            observation["geometrySupport"] = {"validPixelCount":len(points),"coordinateFrameId":f.coordinate_frame_id,"boundsNative":{"min":points.min(axis=0).tolist(),"max":points.max(axis=0).tolist()} if len(points) >= 8 else None,"sourceRefs":frame_record["sourceRefs"],
                "coverage":"visible_support_only","uncertainty":{"status":"not_quantified","causes":["estimated_depth","occlusion","mask_boundary"]}}
            if len(points) >= 8:
                candidates.append((len(points),oid,f,support))
        if not candidates:
            continue
        _,oid,f,support = max(candidates,key=lambda x:(x[0],x[1]))
        observation = lookup[oid]
        entity["measurements"]["observedBounds"] = {**observation["geometrySupport"]["boundsNative"],"coordinateFrameId":f.coordinate_frame_id,
            "source":"observed_measurement","sourceRefs":[{"observationId":oid,"revision":observation["revision"]}],"unit":"native","validPixelCount":int(support.sum()),
            "coverage":"visible_support_only","uncertainty":{"status":"not_quantified","causes":["estimated_depth","occlusion","mask_boundary"]}}
        bounds = observation["geometrySupport"]["boundsNative"]
        dimensions = np.asarray(bounds["max"])-np.asarray(bounds["min"])
        entity["measurements"].update({"dimensionsNative":dimensions.tolist(),
            "coordinateFrameId":f.coordinate_frame_id,"dimensionBasis":"native_axes_not_ground_aligned"})
        mesh = _mesh(f.points,f.support(),canonical[f.image_id]["rgb"],masks[oid])
        if mesh is None:
            continue
        asset = _save_mesh(stages,mesh,{"kind":"observed_surface","entityId":entity["id"],"sourceObservationId":oid})
        if asset["id"] not in {a["id"] for a in document["assets"]}:
            document["assets"].append(asset)
        rep = {"id":_id(document["captureId"],"observed",entity["id"],asset["sha256"]),"kind":"observed_surface","assetId":asset["id"],"coordinateFrameId":f.coordinate_frame_id,
               "transform":{"coordinateFrameId":f.coordinate_frame_id,"position":[0.,0.,0.],"quaternion":[0.,0.,0.,1.],"scale":[1.,1.,1.]},"bounds":asset["metadata"]["bounds"],"primitive":None,"placementState":"confirmed","sourceRefs":[{"observationId":oid,"revision":observation["revision"]}]}
        entity["representations"] = [r for r in entity["representations"] if r["kind"] != "observed_surface"] + [rep]
    _association_evidence(document,associations,frames,masks)
    return associations


def _capture_context(document,frames,canonical,stages):
    if not frames:
        return
    # One fixed camera state avoids inventing a fused state of moving machinery.
    anchor = max(frames.values(),key=lambda f:(int(f.support().sum()),f.image_id))
    mesh = _mesh(anchor.points,anchor.support(),canonical[anchor.image_id]["rgb"],np.ones(anchor.valid.shape,bool))
    if mesh is None:
        return
    asset = _save_mesh(stages,mesh,{"kind":"capture_context","imageId":anchor.image_id})
    _include(document,asset)
    identity = _id(document["captureId"],"capture_context")
    entity = {"id":identity,"label":"Observed capture context","kind":"capture_context","sourceContext":True,"editable":False,"observationRefs":[],"associationState":"confirmed",
        "representations":[{"id":_id(document["captureId"],"context",asset["sha256"]),"kind":"observed_surface","assetId":asset["id"],"coordinateFrameId":anchor.coordinate_frame_id,
            "transform":{"coordinateFrameId":anchor.coordinate_frame_id,"position":[0.,0.,0.],"quaternion":[0.,0.,0.,1.],"scale":[1.,1.,1.]},
            "primitive":None,"bounds":asset["metadata"]["bounds"],"placementState":"confirmed","editable":False,"sourceRefs":[{"imageId":anchor.image_id},_ref(asset)],"coverage":"observed_camera_state_only"}],
        "currentModelTransform":None,"measurements":{},"groupId":None,"lineage":[]}
    document["entities"] = [e for e in document["entities"] if e["id"] != identity] + [entity]


def _ground(document,frames,masks,stages):
    observations = [o for o in document["observations"] if o["id"] in masks and any(e.get("geometryRole") == "floor" for e in o["labelEvidence"])]
    ground,report = estimate_native_ground(frames,[MaskObservation(o["id"],o["imageId"],masks[o["id"]]) for o in observations])
    refs = [{"observationId":o["id"],"revision":o["revision"],"maskAssetId":o["maskAssetId"]} for o in observations]
    if ground is not None:
        evidence = stages.put({"fit":report,"sourceRefs":refs,"coordinateFrameId":next(iter(frames.values())).coordinate_frame_id},
            {"kind":"ground_fit_evidence","pipelineVersion":PIPELINE_VERSION})
        _include(document,evidence)
        ground["sourceRefs"] = refs+[_ref(evidence)]
    for frame in document["coordinateFrames"]:
        if frame["id"] in {f.coordinate_frame_id for f in frames.values()} and (frame.get("ground") or {}).get("source") not in ("manual","manual_assertion"):
            frame["ground"],frame["groundFit"] = ground,report
    lookup = {o["id"]:o for o in document["observations"]}
    for entity in document["entities"]:
        measurement = entity["measurements"]
        if "observedBounds" not in measurement:
            continue
        measurement["groundHeightNative"],measurement["groundSupportRangeNative"] = None,None
        if ground is None:
            continue
        oid = measurement["observedBounds"]["sourceRefs"][0]["observationId"]
        observation = lookup[oid]
        frame = frames[observation["imageId"]]
        points = frame.points[masks[oid] & frame.support()]
        heights = points@np.asarray(ground["normal"])+ground["plane"][3]
        measurement["groundHeightNative"] = float(heights.max())
        measurement["groundSupportRangeNative"] = {"min":float(heights.min()),"max":float(heights.max()),"span":float(np.ptp(heights)),
            "source":"observed_surface_only","meaning":"Visible support relative to estimated floor; not full object dimensions",
            "sourceRefs":[{"observationId":oid,"revision":observation["revision"]}]+ground["sourceRefs"],"unit":"native","uncertainty":ground["uncertainty"]}
    return report


def run_analysis(repository,blobs,job,providers):
    capture,document,images = _capture(repository,blobs,job)
    stages = _Stages(repository,blobs,job,providers)
    errors,frames,canonical,masks = [],{},{},{}
    def attempt(stage,fn):
        try:
            return fn()
        except PlatformError as exc:
            errors.append({"stage":stage,"code":exc.code,"params":exc.params})
            return None
        except (ValueError,TypeError,KeyError):
            errors.append({"stage":stage,"code":"provider_response_invalid"})
            return None
    for image in images:
        output = attempt("discovery",lambda:stages.call("discovery",[image],{"image":_image_payload(image),"discoveryContractVersion":"geometry_role_v1"}))
        if output:
            response,evidence = output
            attempt("discovery",lambda:_discover(document,image,response,evidence))
            stages.checkpoint(document,"discovery")
    geometry = attempt("geometry",lambda:stages.call("geometry",images,{"images":[_image_payload(i) for i in images]}))
    if geometry:
        parsed = attempt("geometry",lambda:_geometry(document,images,*geometry))
        if parsed:
            frames,canonical = parsed
    for image in images:
        # Separate corrected product baseline: each call/cache sees one photo.
        attempt("depth",lambda:stages.call("depth",[image],{"image":_image_payload(image)}))
    lookup = {i["id"]:i for i in images}
    for observation in document["observations"]:
        image = lookup[observation["imageId"]]
        output = attempt("segmentation",lambda:stages.call("segmentation",[image],{"image":_image_payload(image),"box":observation["originalPixelBox"]},observation["sourceRefs"]))
        if not output:
            continue
        response,evidence = output
        _include(document,evidence)
        mask = attempt("segmentation",lambda:_save_observation_mask(document,observation,image,response,evidence,stages))
        if mask is None:
            continue
        if image["id"] in canonical:
            masks[observation["id"]] = _canonical_mask(mask,canonical[image["id"]])
        stages.checkpoint(document,"segmentation")
    association = attempt("association",lambda:_associate_and_surfaces(document,frames,canonical,masks,stages)) if frames else None
    if association is None:
        _association_evidence(document,None,frames,masks)
    attempt("capture_context",lambda:_capture_context(document,frames,canonical,stages))
    ground_report = attempt("ground",lambda:_ground(document,frames,masks,stages))
    checkpoint = stages.checkpoint(document,"analysis_complete" if not errors else "analysis_incomplete")
    result = {"status":"incomplete" if errors else "succeeded","pipelineVersion":PIPELINE_VERSION,"stages":stages.records,"errors":errors,"checkpointAssetId":checkpoint["id"],
              "association":association,"entityCount":sum(e.get("kind") != "capture_context" for e in document["entities"]),"observationCount":len(document["observations"]),"generatedAssetCount":0,
              "groundFit":ground_report,
              "qualityStatus":"not_evaluated_against_physical_ground_truth","baselineProtocol":"corrected_per_image_cache; historical fixed runs must be rerun separately"}
    return document,result


def _load_geometry(document,images,stages):
    reference = next((r for frame in document["coordinateFrames"] for r in frame.get("sourceRefs",[]) if r.get("assetId")),None)
    if not reference:
        raise PlatformError("geometry_evidence_unavailable",409)
    asset = stages.repo.get_asset(reference["assetId"])
    if asset["projectId"] != stages.job["projectId"] or asset["sha256"] != reference["sha256"]:
        raise PlatformError("geometry_evidence_mismatch",409)
    envelope = stages.load(asset)
    if envelope.get("stage") != "geometry":
        raise PlatformError("geometry_evidence_mismatch",409)
    return _geometry(document,images,envelope["output"],asset)


def run_segmentation(repository,blobs,job,providers):
    _,document,images = _capture(repository,blobs,job)
    stages = _Stages(repository,blobs,job,providers)
    requested = job["inputs"].get("observationIds") or [job["inputs"].get("observationId")]
    observations = [o for o in document["observations"] if o["id"] in requested]
    if not observations or len(observations) != len(set(requested)):
        raise PlatformError("observation_not_found",404)
    errors = []
    for observation in observations:
        image = next(x for x in images if x["id"] == observation["imageId"])
        try:
            response,evidence = stages.call("segmentation",[image],{"image":_image_payload(image),"box":observation["originalPixelBox"]},observation["sourceRefs"])
            _include(document,evidence)
            _save_observation_mask(document,observation,image,response,evidence,stages)
            observation["revision"] += 1
            for entity in document["entities"]:
                if observation["id"] in entity["observationRefs"]:
                    # Retain old meshes as explicit stale evidence; never silently
                    # substitute them for this revised observation.
                    for rep in entity["representations"]:
                        rep["sourceValidity"] = "stale"
                    entity["measurements"] = {}
        except PlatformError as exc:
            errors.append({"observationId":observation["id"],"code":exc.code})
    try:
        frames,canonical = _load_geometry(document,images,stages)
        masks = {}
        for o in document["observations"]:
            if o.get("maskAssetId"):
                asset = repository.get_asset(o["maskAssetId"])
                with Image.open(io.BytesIO(blobs.get(asset["storageKey"],asset["sha256"],asset["sizeBytes"]))) as im:
                    masks[o["id"]] = _canonical_mask(np.asarray(im)>0,canonical[o["imageId"]])
        _associate_and_surfaces(document,frames,canonical,masks,stages)
        _ground(document,frames,masks,stages)
    except PlatformError as exc:
        errors.append({"stage":"geometry_support","code":exc.code})
    stages.checkpoint(document,"segmentation")
    return document,{"status":"incomplete" if errors else "succeeded","stages":stages.records,"errors":errors}


def run_generation(repository,blobs,job,providers):
    _,document,images = _capture(repository,blobs,job)
    stages = _Stages(repository,blobs,job,providers)
    requested = job["inputs"].get("entityIds") or ([job["inputs"]["entityId"]] if job["inputs"].get("entityId") else [e["id"] for e in document["entities"] if not e.get("sourceContext")] if job["kind"] == "generate_scene" else [])
    entities = [e for e in document["entities"] if e["id"] in requested]
    if not requested or len(entities) != len(set(requested)):
        raise PlatformError("entity_not_found",404)
    errors,ready = [],[]
    try:
        frames,canonical = _load_geometry(document,images,stages)
    except PlatformError as exc:
        return document,{"status":"incomplete","errors":[{"code":exc.code}],"shapeReadyEntityIds":[],"placementConfirmedEntityIds":[]}
    observations = {o["id"]:o for o in document["observations"]}
    for entity in entities:
        try:
            candidates = [observations[x] for x in entity["observationRefs"] if observations[x].get("maskAssetId")]
            if not candidates:
                raise PlatformError("generation_mask_required",409)
            anchor = max(candidates,key=lambda x:((x.get("geometrySupport") or {}).get("validPixelCount",0),x["id"]))
            image = next(i for i in images if i["id"] == anchor["imageId"])
            asset = repository.get_asset(anchor["maskAssetId"])
            if asset["projectId"] != job["projectId"]:
                raise PlatformError("asset_project_mismatch",403)
            with Image.open(io.BytesIO(blobs.get(asset["storageKey"],asset["sha256"],asset["sizeBytes"]))) as im:
                mask = _canonical_mask(np.asarray(im)>0,canonical[image["id"]])
            f = frames[image["id"]]
            payload = {"entityId":entity["id"],"image":canonical[image["id"]]["rgb"],"mask":mask,"points":f.points,"valid":f.valid,"K":f.K,"cameraToWorld":f.camera_to_world,
                       "coordinateFrameId":f.coordinate_frame_id,"imageId":image["id"],"imageSha256":image["sha256"],"seed":job.get("config",{}).get("seed",0)}
            response,evidence = stages.call("generation",[image],payload,[{"observationId":anchor["id"],"revision":anchor["revision"],"maskSha256":asset["sha256"]}])
            _include(document,evidence)
            mesh = MeshData(np.asarray(response["vertices"]),np.asarray(response["faces"]),np.asarray(response["colors"]) if response.get("colors") is not None else None)
            mesh_asset = _save_mesh(stages,mesh,{"kind":"generated_mesh","entityId":entity["id"],"sourceRefs":[_ref(evidence)]})
            document["assets"] = [a for a in document["assets"] if a["id"] != mesh_asset["id"]] + [mesh_asset]
            proposed = response.get("proposedObjectToNative")
            transform = matrix_to_transform(np.asarray(proposed) if proposed is not None else np.eye(4),f.coordinate_frame_id)
            rep = {"id":_id(document["captureId"],"generated",entity["id"],mesh_asset["sha256"]),"kind":"generated_mesh","assetId":mesh_asset["id"],"coordinateFrameId":f.coordinate_frame_id,
                   "transform":transform,"bounds":mesh_asset["metadata"]["bounds"],"primitive":None,"placementState":"unconfirmed","sourceRefs":[_ref(evidence),{"observationId":anchor["id"],"revision":anchor["revision"]}],
                   "placementReason":"requires_alignment_confirmation" if proposed is not None else "insufficient_observed_depth","shapeStatus":"ready"}
            previous = entity["representations"]
            # Clear only a copied, unconfirmed generated proposal. Manual/primitive
            # edits and unattributed entity poses cannot be classified as stale.
            if not any(r["kind"] == "primitive" or (r.get("placementSource") or {}).get("type") == "manual_assertion" for r in previous) and any(
                r["kind"] == "generated_mesh" and r.get("placementState") == "unconfirmed" and
                r.get("placementReason") in ("imported_proposal", "requires_alignment_confirmation", "insufficient_observed_depth") and
                r["transform"] == entity.get("currentModelTransform") for r in previous):
                entity["currentModelTransform"] = None
            entity["representations"] = [r for r in entity["representations"] if r["kind"] != "generated_mesh"] + [rep]
            ready.append(entity["id"])
        except PlatformError as exc:
            errors.append({"entityId":entity["id"],"code":exc.code})
        except (ValueError,TypeError,KeyError):
            errors.append({"entityId":entity["id"],"code":"generation_response_invalid"})
    stages.checkpoint(document,"generation")
    return document,{"status":"incomplete","stages":stages.records,"errors":errors,"shapeReadyEntityIds":ready,"placementConfirmedEntityIds":[],
                     "placementStatus":"requires_alignment_confirmation","manifest":{"baseSceneRevisionId":job["baseRevisionId"],"entityIds":requested,"sourceDocumentSha256":digest(repository.get_revision(job["baseRevisionId"])["document"])}}


class _DiscoveredItem(BaseModel):
    label: str = Field(min_length=1)
    box_2d: tuple[int,int,int,int]
    evidence: str = ""
    geometry_role: Literal["floor","object","unknown"] = "unknown"


class _DiscoveryResponse(BaseModel):
    items: list[_DiscoveredItem]


def _discovery_invoke(payload):
    from ..providers.gemini import GeminiAdapter, GEMINI_MODEL_ID, _response_format, _text_block
    from google.genai import interactions
    image = payload["image"]
    adapter = GeminiAdapter()
    response = adapter._create("platform.discovery",model=GEMINI_MODEL_ID,input=[
        _text_block("List distinct visible physical objects and separately identifiable components in this image. Use ordinary names, not a fixed taxonomy. Include small objects and partially occluded objects. Do not claim hidden geometry, dimensions, safety compliance, or certainty. Return one tight box_2d [ymin,xmin,ymax,xmax] in 0-1000 coordinates per visible instance, plus the visual evidence for its label. Do not merge a small attached component into its supporting assembly. Set geometry_role=floor only for a visibly supported hypothesis of the walking floor beneath the scene; tabletop, shelf, platform, and other flat object surfaces are not floor. Use unknown if ambiguous. This role is a hypothesis and does not establish a physical ground plane."),
        interactions.ImageContent(data=image["dataUri"].split(",",1)[1],mime_type="image/png")],response_format=_response_format(_DiscoveryResponse))
    usage = getattr(response,"usage",None)
    telemetry = {"usage":usage.model_dump(mode="json",exclude_none=True) if usage is not None else None}
    metadata = _telemetry({"telemetry":telemetry,"providerRequestId":getattr(response,"id",None)})
    try:
        parsed,request_id = adapter._parse(response,_DiscoveryResponse,"platform.discovery")
    except Exception:
        raise ProviderResponseError(metadata) from None
    items = []
    for item in parsed.items:
        y0,x0,y1,x1 = item.box_2d
        items.append({"label":item.label,"box":[x0*image["width"]/1000,y0*image["height"]/1000,x1*image["width"]/1000,y1*image["height"]/1000],"evidence":item.evidence,"geometryRole":item.geometry_role})
    return {"items":items,"providerRequestId":request_id,"telemetry":telemetry}


def _sam_invoke(payload):
    import fal_client
    from ..providers.sam3 import decode_coco_rle
    image = payload["image"]
    x0,y0,x1,y1 = payload["box"]
    # One explicit submission; the legacy SAM adapter retries unknown timeouts.
    handle = fal_client.submit("fal-ai/sam-3-1/image-rle",arguments={"image_url":image["dataUri"],"box_prompts":[{"x_min":x0,"y_min":y0,"x_max":x1,"y_max":y1}],
        "return_multiple_masks":True,"include_scores":True,"max_masks":3})
    response = handle.get()
    rles = response.get("rle") or []
    rles = [rles] if isinstance(rles,str) else rles
    scores = response.get("scores") or []
    if not rles or len(scores) != len(rles):
        raise ProviderResponseError(_telemetry({"providerRequestId":handle.request_id,"telemetry":{"usage":response.get("usage")}}))
    selected = int(np.argmax(scores))
    try:
        mask = decode_coco_rle(rles[selected],height=image["height"],width=image["width"])
    except Exception:
        raise ProviderResponseError(_telemetry({"providerRequestId":handle.request_id,"telemetry":{"usage":response.get("usage")}})) from None
    return {"mask":mask.astype(bool),"selectedCandidate":selected,"candidateRles":rles,"scores":scores,"providerRequestId":handle.request_id,
            "telemetry":{"usage":response.get("usage")}}


def provider_snapshot_from_env() -> dict:
    """Secret-free deployment inputs; the repository freezes this at job creation."""
    path = os.environ.get("PANOPTES_PROVIDER_MANIFEST")
    if not path:
        return {}
    manifest = json.loads(Path(path).read_text())
    allowed = {"provider","pins","estimatedCostUsd","releaseEvidence","paid","modalApp","modalClass","modalMethod","nativePoseEvidence","providerToOpenCV"}
    for stage,config in manifest.items():
        if stage not in {"discovery","geometry","depth","segmentation","generation"} or not isinstance(config,dict) or set(config)-allowed:
            raise PlatformError("invalid_provider_manifest",stage=stage)
    def reject_secrets(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if any(part in key.lower().replace("_","") for part in ("apikey","password","authorization","secret","credential","capability","accesstoken")):
                    raise PlatformError("secret_in_provider_manifest")
                reject_secrets(item)
        elif isinstance(value,list):
            for item in value:
                reject_secrets(item)
    reject_secrets(manifest)
    # Only these reviewed fields can enter public job configuration.
    return json.loads(canonical(manifest))


def providers_from_env() -> dict[str,ProviderSpec]:
    """Convenience for startup only. Jobs use their persisted manifest snapshot."""
    return providers_from_manifest(provider_snapshot_from_env())


def providers_from_manifest(snapshot: Mapping[str,Any], *, _research=False) -> dict[str,ProviderSpec]:
    manifest = json.loads(canonical(snapshot))
    providers = {}
    for stage,config in manifest.items():
        if config.get("paid",True) is not True:
            raise PlatformError("external_provider_requires_paid_reservation",stage=stage)
        pins = config["pins"]
        if stage == "discovery":
            from ..providers.gemini import GEMINI_MODEL_ID
            if pins.get("model") != GEMINI_MODEL_ID:
                raise PlatformError("discovery_model_pin_mismatch")
            invoke = _discovery_invoke
        elif stage == "segmentation":
            if pins.get("model") != "fal-ai/sam-3-1/image-rle":
                raise PlatformError("segmentation_model_pin_mismatch")
            invoke = _sam_invoke
        elif stage in ("geometry","depth"):
            def invoke(payload, *, stage=stage, config=config, pins=pins):
                import modal
                klass = modal.Cls.from_name(config["modalApp"],config["modalClass"])
                result = getattr(klass(),config["modalMethod"]).remote(payload)
                if result.get("pins") != pins:
                    raise ProviderResponseError(_telemetry(result))
                return _unpacked(result)
        elif stage == "generation":
            if pins.get("model") != "facebook/sam-3d-objects":
                raise PlatformError("generation_requires_sam3d")
            def invoke(payload, *, config=config,pins=pins,research=_research):
                import modal
                received = {}
                def runtime(**kwargs):
                    klass = modal.Cls.from_name(config["modalApp"],config["modalClass"])
                    result = getattr(klass(),config["modalMethod"]).remote(kwargs)
                    received.update(_telemetry(result))
                    if result.get("pins") != pins:
                        raise ProviderResponseError(received)
                    return _unpacked(result)
                adapter = SAM3DMeshAdapter.for_research(runtime,pins,config["providerToOpenCV"],payload["_researchProtocol"]) if research else SAM3DMeshAdapter(runtime,pins,config["nativePoseEvidence"],config["providerToOpenCV"])
                frame = FrameGeometry(payload["imageId"],payload["coordinateFrameId"],payload["imageSha256"],payload["points"],payload["valid"],payload["K"],payload["cameraToWorld"])
                try:
                    result = adapter.generate(GenerationRequest(payload["entityId"],payload["image"],payload["mask"],frame,payload["seed"]))
                except Exception:
                    if received:
                        raise ProviderResponseError(received) from None
                    raise
                return {"vertices":result.mesh.vertices,"faces":result.mesh.faces,"colors":result.mesh.colors,
                        "proposedObjectToNative":result.proposed_object_to_native,"provenance":result.provenance,
                        "telemetry":{k:v for k,v in received.items() if k != "providerRequestId"},"providerRequestId":received.get("providerRequestId")}
        else:
            raise PlatformError("unknown_provider_stage",stage=stage)
        providers[stage] = ProviderSpec(config["provider"],pins,invoke,config["estimatedCostUsd"],config["releaseEvidence"],config.get("paid",True))
        # Gate each independent stage immediately before reserve/dispatch. An
        # unapproved geometry stage must not discard approved discovery evidence.
    return providers


def run_research_stage(repository,blobs,job,stage,payload,images,provider_manifest,protocol):
    """Explicit experiment entrypoint; returns an artifact, never a product scene.

    The experiment can measure a not-yet-passed quality gate; licensing and pinned
    runtime evidence are still required. No automatic call from product handlers.
    """
    required = {"id","inputHashes","baselineRevision","metricDefinitions","policyThresholds","split"}
    if not isinstance(protocol,dict) or not required <= set(protocol) or not protocol["inputHashes"] or not protocol["metricDefinitions"]:
        raise PlatformError("frozen_research_protocol_required",409)
    if [i["sha256"] for i in images] != protocol["inputHashes"]:
        raise PlatformError("research_input_hash_mismatch",409)
    providers = providers_from_manifest(provider_manifest,_research=True)
    stages = _Stages(repository,blobs,job,providers)
    protocol_asset = stages.put({"protocol":protocol,"providerManifest":provider_manifest},{"kind":"frozen_research_protocol"})
    if stage == "generation":
        payload = {**payload,"_researchProtocol":protocol}
    output,asset = stages.call(stage,images,payload,[_ref(protocol_asset)],research_protocol=protocol)
    return {"status":"research_only","protocolAssetId":protocol_asset["id"],"outputAssetId":asset["id"],"stages":stages.records,"output":output,
            "productReleaseStatus":"not_changed","sceneRevision":None}
