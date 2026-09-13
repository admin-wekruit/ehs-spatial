"""Runnable spatial contracts, including the actual Blender save/reopen path."""
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
from uuid import uuid4
from PIL import Image

import numpy as np
import pytest

from ehs_spatial.platform.contracts import PlatformError, empty_document
from ehs_spatial.platform.spatial import (
    AssociationConfig, FrameGeometry, MaskObservation, SAM3DMeshAdapter,
    GenerationRequest, MeshData, associate_observations, ground_measurements,
    pixel_center_mapping, primitive_mesh, project_native, stage_cache_key,
    transform_matrix, transform_points, verify_native_pose,
    unproject_pixels, intersect_camera_plane,estimate_native_ground,
)
from ehs_spatial.platform.blender_export import (
    BLENDER_SCRIPT, blender_camera_parameters, export_scene_revision,
    mesh_from_asset, prepare_export, write_glb,
)


def frame(image_id="a", offset=0):
    y,x = np.mgrid[:12,:12]
    k = np.array([[10.,0,5.5],[0,10.,5.5],[0,0,1.]])
    c2w = np.eye(4)
    c2w[0,3] = offset
    points = np.stack(((x-5.5)/5+offset,(y-5.5)/5,np.full_like(x,2)),axis=-1).astype(float)
    return FrameGeometry(image_id,"native",hashlib.sha256(image_id.encode()).hexdigest(),points,np.ones((12,12),dtype=bool),k,c2w)


def mask(x0=0,x1=12):
    result = np.zeros((12,12),dtype=bool)
    result[:,x0:x1] = True
    return result


def test_crossview_identity_positive_repeated_objects_and_no_overmerge():
    frames = {i:frame(i) for i in ("a","b","c")}
    observations = [MaskObservation(f"{i}-left",i,mask(0,5)) for i in frames]
    observations += [MaskObservation(f"{i}-right",i,mask(7,12)) for i in frames]
    result = associate_observations(observations,frames)
    assert result["groups"] == [["a-left","b-left","c-left"],["a-right","b-right","c-right"]]
    assert not result["ambiguous"]
    # Nested duplicate candidates in one view must not be greedily merged.
    duplicate = MaskObservation("b-duplicate","b",mask(0,5))
    result = associate_observations(observations+[duplicate],frames)
    assert all(not ("b-left" in g and "b-duplicate" in g) for g in result["groups"])
    assert any("b-duplicate" in x["observationIds"] for x in result["ambiguous"])
    assert sorted(x for g in result["groups"] for x in g) == sorted(x.id for x in observations+[duplicate])


def test_tiny_missing_depth_different_frames_and_label_free_identity():
    a,b = frame(),frame("b")
    tiny = np.zeros((12,12),bool)
    tiny[0,0] = True
    result = associate_observations([MaskObservation("small","a",tiny),MaskObservation("unknown","missing",tiny)],{"a":a})
    assert result["groups"] == [["small"],["unknown"]]
    a.valid[:] = False
    result = associate_observations([MaskObservation("a","a",mask()),MaskObservation("b","b",mask())],{"a":a,"b":b})
    assert result["groups"] == [["a"],["b"]]
    with pytest.raises(PlatformError,match="mask_geometry_grid_mismatch"):
        associate_observations([MaskObservation("wrong","b",np.ones((5,5)))],{"b":b})


def test_occlusion_depth_and_independent_support_do_not_create_identity():
    a,b = frame(),frame("b")
    b.points[:] *= .5  # nearer surface in same rays occludes object in frame a
    result = associate_observations([MaskObservation("a","a",mask()),MaskObservation("b","b",mask())],{"a":a,"b":b})
    assert result["groups"] == [["a"],["b"]]
    with pytest.raises(PlatformError,match="invalid_association_config"):
        AssociationConfig(min_support=0)


def test_camera_pixel_centres_skew_rotation_and_nonunit_scale():
    angle = .37
    pose = {"coordinateFrameId":"g","position":[1.,2.,3.],"quaternion":[0.,math.sin(angle/2),0.,math.cos(angle/2)],"scale":[2.,3.,4.]}
    matrix = transform_matrix(pose)
    assert np.allclose(np.linalg.norm(matrix[:3,:3],axis=0),[2,3,4])
    c2w = transform_matrix({**pose,"scale":[1.,1.,1.]})
    k = np.array([[600.,17.,301.],[0.,500.,205.],[0.,0.,1.]])
    local = np.array([[.1,.2,2.],[-.3,.1,4.]])
    world = transform_points(local,c2w)
    uv,depth = project_native(world,{"K":k,"cameraToWorld":c2w})
    assert np.allclose(uv,(local@k.T)[:,:2]/local[:,2,None])
    assert np.allclose(depth,local[:,2])
    assert np.allclose(unproject_pixels(uv,depth,{"K":k,"cameraToWorld":c2w}),world)
    assert np.allclose(intersect_camera_plane(uv[0],{"K":k,"cameraToWorld":c2w},[0,0,1,-world[0,2]]),world[0])
    mapping = pixel_center_mapping((640,480),(320,120))
    assert np.allclose(mapping,[[.5,0,-.25],[0,.25,-.375],[0,0,1]])
    with pytest.raises(PlatformError,match="camera_projection_unrepresentable"):
        blender_camera_parameters({"id":"camera","K":k,"cameraToWorld":c2w,"width":640,"height":480})
    with pytest.raises(PlatformError,match="invalid_rigid_camera"):
        project_native(world,{"K":k,"cameraToWorld":matrix})


def test_cache_actual_perframe_bytes_pins_and_payload_refs():
    a = {"imageId":"a","sha256":"a"*64}
    b = {"imageId":"b","sha256":"b"*64}
    pins = {"model":"abc","adapter":"1"}
    first = stage_cache_key("moge",[a],pins)
    assert first != stage_cache_key("moge",[b],pins)
    assert first != stage_cache_key("moge",[{**a,"sha256":"c"*64}],pins)
    assert first != stage_cache_key("moge",[a],{**pins,"model":"other"})
    assert first != stage_cache_key("moge",[a],pins,source_refs=[{"maskSha256":"c"*64}])
    assert stage_cache_key("joint",[a,b],pins) != stage_cache_key("joint",[b,a],pins)
    with pytest.raises(PlatformError,match="invalid_image_hash"):
        stage_cache_key("moge",[{"imageId":"a","sha256":"run-level-cache"}],pins)


def test_primitives_standalone_without_ground_or_metres():
    box = primitive_mesh({"type":"box","dimensions":[2,3,4]})
    cylinder = primitive_mesh({"type":"cylinder","radius":.5,"height":3,"segments":16})
    assert np.allclose(np.ptp(box.vertices,axis=0),[2,3,4])
    assert np.allclose(np.ptp(cylinder.vertices,axis=0),[1,1,3])
    assert len(cylinder.faces) == 64
    result = ground_measurements(box.vertices,np.eye(4),None)
    assert result["groundHeightNative"] is None and result["groundHeightMeters"] is None and result["groundTiltDegrees"] is None
    result = ground_measurements(box.vertices,np.eye(4),{"plane":[0,0,1,2]},.1)
    assert result["groundHeightNative"] == 4 and result["groundHeightMeters"] == .4
    with pytest.raises(PlatformError,match="invalid_primitive_dimensions"):
        primitive_mesh({"type":"cylinder","radius":0,"height":3})


def test_sam3d_pose_basis_gate_and_double_pose_rejection():
    raw = primitive_mesh({"type":"box","dimensions":[1,2,3]}).vertices
    t = np.diag([2.,3.,4.,1.])
    t[:3,3] = [1,2,3]
    basis = np.diag([-1.,-1.,1.,1.])
    c2w = np.eye(4)
    c2w[:3,3] = [.2,.3,.4]
    official = transform_points(raw,t)
    pose,residual = verify_native_pose(raw,official,t,basis,c2w)
    assert residual == 0 and np.allclose(pose,c2w@basis@t)
    with pytest.raises(PlatformError,match="sam3d_pose_mismatch"):
        verify_native_pose(raw,official,t@t,basis,c2w)
    with pytest.raises(PlatformError,match="sam3d_release_gate_unverified"):
        SAM3DMeshAdapter(lambda **kw: None,{"model":"test"},{},basis)


def test_synthetic_sam3d_contract_is_shape_ready_but_never_placement_proof():
    pins = {"model":"synthetic-test-only","code":"test-only"}
    basis = np.diag([-1.,-1.,1.,1.])
    evidence = {"pins":pins,"providerToOpenCV":basis.tolist(),**{name:{"status":"passed","artifactSha256":"0"*64} for name in ("licenseAudit","meshOnlyDependencyAudit","officialPoseFixture","externalPointmapNoDepth")}}
    mesh = primitive_mesh({"type":"box","dimensions":[1,1,1]})
    calls = []
    def runtime(**kwargs):
        calls.append(kwargs)
        return {"vertices":mesh.vertices,"faces":mesh.faces,"officialPosedVertices":mesh.vertices,"objectToProvider":np.eye(4),"internalDepthCalls":0,"decodeFormats":["mesh"]}
    adapter = SAM3DMeshAdapter(runtime,pins,evidence,basis)
    f = frame()
    f.valid[:] = False
    result = adapter.generate(GenerationRequest("button",np.zeros((12,12,3),np.uint8),mask(),f))
    assert result.placement_state == "unconfirmed" and result.proposed_object_to_native is None
    assert result.provenance["shapeStatus"] == "ready"
    assert calls[0]["decode_formats"] == ["mesh"] and not calls[0]["with_texture_baking"]


def scene_document():
    document = empty_document()
    g = str(uuid4())
    document.update(captureId=str(uuid4()),target="standalone_object",
                    coordinateFrames=[{"id":g,"convention":"opencv","scale":{"status":"uncalibrated","nativeToMeters":None,"sourceRefs":[]},"ground":None}])
    for i in range(4):
        t = {"coordinateFrameId":g,"position":[float(i),.1,2.],"quaternion":[0.,0.,math.sin(.2),math.cos(.2)],"scale":[1.,2.,.7]}
        primitive = {"type":"box","dimensions":[.2,.3,.4]} if i%2 else {"type":"cylinder","radius":.15,"height":.6,"segments":16}
        document["entities"].append({"id":str(uuid4()),"label":"generic object","observationRefs":[],"associationState":"confirmed",
            "representations":[{"id":str(uuid4()),"kind":"primitive","assetId":None,"coordinateFrameId":g,"transform":t,"primitive":primitive,"placementState":"confirmed","sourceRefs":[]}],
            "currentModelTransform":None,"measurements":{},"groupId":None,"lineage":[]})
    document["entities"].append({"id":str(uuid4()),"label":"small unresolved button","observationRefs":[],"associationState":"association_pending","representations":[],"currentModelTransform":None,"measurements":{},"groupId":None,"lineage":[]})
    c2w = transform_matrix({"coordinateFrameId":g,"position":[1.,.2,-.3],"quaternion":[0.,math.sin(.1),0.,math.cos(.1)],"scale":[1.,1.,1.]})
    document["cameras"] = [{"id":str(uuid4()),"imageId":str(uuid4()),"coordinateFrameId":g,"width":640,"height":480,"K":[[600.,0.,260.],[0.,550.,210.],[0.,0.,1.]],"cameraToWorld":c2w.tolist()}]
    document["assets"].append({"id":document["cameras"][0]["imageId"],"kind":"source_image","mediaType":"image/png"})
    return document


def test_glb_revision_roundtrip_missing_mesh_and_model_transform(tmp_path):
    doc = scene_document()
    doc["entities"][0]["currentModelTransform"] = {**doc["entities"][0]["representations"][0]["transform"],"position":[9.,8.,7.]}
    prepared = prepare_export("revision",doc,lambda _: pytest.fail("primitive must not resolve an asset"))
    assert len(prepared["objects"]) == 4 and len(prepared["unplacedEntities"]) == 1
    assert prepared["objects"][0]["matrix"][0][3] == 9
    validation = write_glb(prepared,tmp_path/"scene.glb")
    assert validation["status"] == "passed" and validation["objects"] == 4
    mesh = mesh_from_asset((tmp_path/"scene.glb").read_bytes(),{})
    assert len(mesh.faces) == 2*64+2*12
    assert mesh.vertices[:,0].max() > 9
    assert not doc["entities"][0]["representations"][0]["transform"]["position"][0] == 9


def test_export_asset_allowlist_hash_and_unconfirmed_placement(tmp_path):
    doc = scene_document()
    rep = doc["entities"][0]["representations"][0]
    rep.update(kind="generated_mesh",assetId="unknown")
    called = []
    with pytest.raises(PlatformError,match="representation_asset_not_found"):
        prepare_export("rev",doc,lambda x:called.append(x))
    assert not called
    doc["assets"].append({"id":"unknown","kind":"generated_mesh"})
    rep["placementState"] = "unconfirmed"
    prepared = prepare_export("rev",doc,lambda _: pytest.fail("unplaced asset must not be resolved"))
    assert len(prepared["objects"]) == 3
    with pytest.raises(PlatformError,match="asset_hash_mismatch"):
        mesh_from_asset(b"wrong",{"sha256":"a"*64})


BLENDER = Path(os.environ.get("BLENDER_EXECUTABLE","/Users/adam/Desktop/panoptes-public/.tools/blender-4.5.9/Blender.app/Contents/MacOS/Blender"))


def test_export_point_cloud_context_excluded_but_object_stays_unplaced(tmp_path):
    doc = scene_document()
    doc["entities"][0]["sourceContext"] = True  # Mesh context still exports normally.
    original = prepare_export("revision", doc, lambda _: pytest.fail("no mesh asset needed"))
    cloud = deepcopy(doc["entities"][0])
    cloud.update(id=str(uuid4()), sourceContext=True)
    rep = cloud["representations"][0]
    asset_id = str(uuid4())
    rep.update(id=str(uuid4()), kind="point_cloud", assetId=asset_id, primitive=None)
    doc["assets"].append({"id":asset_id, "kind":"point_cloud"})
    doc["entities"].append(cloud)
    context = prepare_export("revision", doc, lambda _: pytest.fail("point cloud must not be read as mesh"))
    expected = [{"entityId":cloud["id"], "representationId":rep["id"], "reason":"point_cloud_context"}]
    assert context["unplacedEntities"] == original["unplacedEntities"]
    assert context["excludedRepresentations"] == expected
    assert context["objects"] == original["objects"] and context["cameras"] == original["cameras"]
    object_cloud = deepcopy(cloud)
    object_cloud.update(id=str(uuid4()), sourceContext=False)
    object_cloud["representations"][0]["id"] = str(uuid4())
    doc["entities"].append(object_cloud)
    prepared = prepare_export("revision", doc, lambda _: pytest.fail("point cloud must not be read as mesh"))
    assert prepared["unplacedEntities"] == original["unplacedEntities"] + [{"entityId":object_cloud["id"], "representationId":object_cloud["representations"][0]["id"], "reason":"not_a_mesh"}]
    assert prepared["excludedRepresentations"] == expected
    if BLENDER.exists():
        result = export_scene_revision("revision", doc, lambda _: pytest.fail("no mesh asset needed"), tmp_path/"export", BLENDER)
        assert result["excludedRepresentations"] == expected
        assert result["unplacedEntities"] == prepared["unplacedEntities"] and result["status"] == "incomplete"
        assert len(result["validation"]["blender"]["objects"]) == len(original["objects"])
        assert len(result["validation"]["blender"]["cameras"]) == len(original["cameras"])
        assert json.loads((tmp_path/"export/manifest.json").read_text())["excludedRepresentations"] == expected


@pytest.mark.skipif(not BLENDER.exists(),reason="real Blender executable not installed")
def test_real_blender_generic_save_reopen_and_parametric_edit(tmp_path):
    doc = scene_document()
    result = export_scene_revision(str(uuid4()),doc,lambda _: pytest.fail("no assets"),tmp_path/"export",BLENDER)
    assert result["newModelCalls"] == 0
    assert result["validation"]["blender"]["status"] == "passed"
    assert len(result["validation"]["blender"]["objects"]) == 4
    assert len(result["validation"]["blender"]["parameters"]) == 4
    assert result["validation"]["blender"]["cameras"][0]["maxConversionErrorPixels"] < .001
    assert (tmp_path/"export/scene.blend").stat().st_size > 1000
    assert (tmp_path/"export/scene.glb").stat().st_size > 1000
    with pytest.raises(PlatformError,match="export_destination_exists"):
        export_scene_revision("other",doc,lambda _:None,tmp_path/"export",BLENDER)


@pytest.mark.skipif(not BLENDER.exists(),reason="real Blender executable not installed")
def test_blender_portrait_camera_fx_below_fy_preserves_pixel_aspect_ratio(tmp_path):
    doc = scene_document()
    camera = doc["cameras"][0]
    camera.update(width=2880,height=3840,K=[[2795.37455656,0,1439.5],[0,2816.24049831,1919.5],[0,0,1]])
    params = blender_camera_parameters(camera)
    assert params["pixelAspectX"] >= 1 and params["pixelAspectY"] >= 1
    assert params["pixelAspectY"]/params["pixelAspectX"] == pytest.approx(camera["K"][0][0]/camera["K"][1][1])
    result = export_scene_revision("portrait-fx-less-than-fy",doc,lambda _:pytest.fail("no assets"),tmp_path/"portrait",BLENDER)
    assert result["validation"]["blender"]["cameras"][0]["maxConversionErrorPixels"] < .001
    assert params["intrinsicConversionErrorPixels"] < 1e-8
    assert params["reopenTolerancePixels"] == pytest.approx(16*np.spacing(np.float32(3840)))
    assert params["reopenTolerancePixels"] < .004


@pytest.mark.skipif(not BLENDER.exists(),reason="real Blender executable not installed")
def test_embedded_texture_and_uv_survive_glb_and_blender_reopen(tmp_path):
    doc = scene_document()
    asset_id = str(uuid4())
    doc["assets"].append({"id":asset_id,"kind":"generated_mesh"})
    rep = doc["entities"][0]["representations"][0]
    rep.update(kind="generated_mesh",primitive=None,assetId=asset_id)
    box = primitive_mesh({"type":"box","dimensions":[.2,.3,.4]})
    raw = io.BytesIO()
    Image.fromarray(np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,0]]],np.uint8)).save(raw,format="PNG")
    uv = np.tile(np.array([[0,0],[1,0],[1,1],[0,1]],dtype=np.float32),(2,1))
    mesh = MeshData(box.vertices,box.faces,np.ones_like(box.vertices),uv,raw.getvalue(),"image/png")
    result = export_scene_revision("texture-revision",doc,lambda aid:mesh,tmp_path/"textured",BLENDER)
    assert result["validation"]["blender"]["status"] == "passed"
    prepared = prepare_export("asset-only",{**doc,"entities":[doc["entities"][0]]},lambda aid:mesh)
    write_glb(prepared,tmp_path/"asset.glb")
    reopened = mesh_from_asset((tmp_path/"asset.glb").read_bytes(),{})
    assert reopened.texture_bytes == raw.getvalue()
    assert np.array_equal(reopened.uv,uv)


@pytest.mark.skipif(not BLENDER.exists(),reason="real Blender executable not installed")
def test_multiple_material_slots_png_jpeg_and_plain_parts_roundtrip(tmp_path):
    from ehs_spatial.platform.blender_export import _read_glb
    box = primitive_mesh({"type":"box","dimensions":[.2,.3,.4]})
    uv = np.tile(np.array([[0,0],[1,0],[1,1],[0,1]],np.float32),(2,1))
    parts = []
    for i,encoding in enumerate(("PNG","JPEG",None)):
        image = io.BytesIO()
        if encoding:
            Image.fromarray(np.full((3,4,3),70+i*50,np.uint8)).save(image,format=encoding)
        material = {"name":"part-"+str(i),"baseColorFactor":[.5,.7,.9,.8],"roughness":.2+i*.2,"metallic":.1+i*.1,
                    "doubleSided":i == 0,"alphaMode":"BLEND" if i == 0 else "OPAQUE","emissiveFactor":[.02,.03,.04],"colorEncoding":"linear"}
        parts.append(MeshData(box.vertices+[i*.6,0,0],box.faces,np.full((len(box.vertices),4),.7,np.float32),
            uv if encoding else None,image.getvalue() if encoding else None,"image/png" if encoding == "PNG" else "image/jpeg" if encoding else None,material))
    mesh = MeshData(np.concatenate([p.vertices for p in parts]),np.concatenate([p.faces+i*len(box.vertices) for i,p in enumerate(parts)]),primitives=tuple(parts))
    doc = scene_document()
    doc["entities"] = [doc["entities"][0]]
    aid = str(uuid4())
    doc["assets"].append({"id":aid,"kind":"generated_mesh"})
    doc["entities"][0]["representations"][0].update(kind="generated_mesh",assetId=aid,primitive=None)
    prepared = prepare_export("source-multi",doc,lambda _:mesh)
    write_glb(prepared,tmp_path/"source.glb")
    source = (tmp_path/"source.glb").read_bytes()
    loaded = mesh_from_asset(source,{})
    assert len(loaded.primitives) == 3
    for actual,expected in zip(loaded.primitives,parts):
        assert actual.texture_bytes == expected.texture_bytes
        assert actual.material["roughness"] == expected.material["roughness"]
        assert actual.material["baseColorFactor"] == expected.material["baseColorFactor"]
    result = export_scene_revision("multi-material",doc,lambda _:source,tmp_path/"multi",BLENDER)
    assert result["validation"]["blender"]["objects"][0]["materials"] == 3
    exported,_ = _read_glb((tmp_path/"multi/scene.glb").read_bytes())
    assert len(exported["meshes"][0]["primitives"]) == 3
    assert [i["mimeType"] for i in exported["images"]] == ["image/png","image/jpeg"]


def floor_frames(scale=1.,second_floor=1.5):
    y,x = np.mgrid[:36,:36]
    frames,observations = {},[]
    for i,height in enumerate((1.5,second_floor)):
        points = np.stack(((x-17.5)*.08,np.full_like(x,height,dtype=float),2+y*.08),axis=-1)*scale
        camera = np.eye(4)
        camera[0,3] = i*.1*scale
        key = str(i)
        frames[key] = FrameGeometry(key,"native",str(i)*64,points,np.ones((36,36),bool),np.array([[30,0,17.5],[0,30,17.5],[0,0,1.]]),camera)
        observations.append(MaskObservation("floor-"+key,key,np.ones((36,36),bool)))
    return frames,observations


def test_native_floor_fit_is_scale_equivariant_without_camera_height_or_axis_guess():
    frames,observations = floor_frames()
    ground,report = estimate_native_ground(frames,observations)
    assert ground is not None and report["status"] == "estimated"
    assert ground["normal"] == pytest.approx([0,-1,0],abs=1e-6)
    assert ground["plane"][3] == pytest.approx(1.5)
    assert ground["unit"] == "native" and "nativeToMeters" not in ground
    large,_ = estimate_native_ground(*floor_frames(scale=10))
    assert large["plane"][3] == pytest.approx(ground["plane"][3]*10)
    assert large["normal"] == pytest.approx(ground["normal"])
    # A shared native-world rotation must rotate the result, never pin it to Z.
    angle = .43
    rotation = np.array([[1,0,0],[0,math.cos(angle),-math.sin(angle)],[0,math.sin(angle),math.cos(angle)]])
    for frame in frames.values():
        frame.points[:] = frame.points@rotation.T
        frame.camera_to_world[:3,:3] = rotation
        frame.camera_to_world[:3,3] = rotation@frame.camera_to_world[:3,3]
    rotated,_ = estimate_native_ground(frames,observations)
    assert rotated["normal"] == pytest.approx(rotation@ground["normal"],abs=1e-6)


def test_floor_abstains_without_semantic_support_or_consistent_planar_views():
    frames,observations = floor_frames()
    assert estimate_native_ground(frames,[])[1]["reason"] == "no_explicit_floor_support"
    assert estimate_native_ground(*floor_frames(second_floor=1.8))[0] is None
    for observation in observations:
        observation.mask[:,:] = False
        observation.mask[:,0] = True
    assert estimate_native_ground(frames,observations)[0] is None
