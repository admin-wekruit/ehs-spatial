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


def test_proven_same_photo_aliases_do_not_compete_with_their_own_identity():
    frames = {name: frame(name) for name in ('a', 'b', 'c')}
    observations = [MaskObservation(name, name, mask(0,5)) for name in frames]
    observations.append(MaskObservation('a-source-copy', 'a', mask(0,5)))
    result = associate_observations(observations, frames, confirmed_groups=[['a','a-source-copy']])
    assert result['groups'] == [['a','a-source-copy','b','c']]
    # Equal masks alone are not identity proof, and a third unknown competitor
    # must still block automatic joining.
    baseline = associate_observations(observations, frames)
    assert not any({'a','a-source-copy'} <= set(g) for g in baseline['groups'])
    observations.append(MaskObservation('a-unknown', 'a', mask(0,5)))
    result = associate_observations(observations, frames, confirmed_groups=[['a','a-source-copy']])
    assert ['a','a-source-copy'] in result['groups']
    assert ['a-unknown'] in result['groups']


def test_occlusion_depth_and_independent_support_do_not_create_identity():
    a,b = frame(),frame("b")
    b.points[:] *= .5  # nearer surface in same rays occludes object in frame a
    result = associate_observations([MaskObservation("a","a",mask()),MaskObservation("b","b",mask())],{"a":a,"b":b})
    assert result["groups"] == [["a"],["b"]]
    with pytest.raises(PlatformError,match="invalid_association_config"):
        AssociationConfig(min_support=0)


def test_confirmed_identity_extends_across_visible_views_but_not_unknown_chains():
    frames = {name: frame(name, offset) for name, offset in [('a', 0), ('b', 1), ('c', 2)]}
    observations = [MaskObservation(name, name, mask()) for name in frames]
    # A/C have too little overlap. B is independently a view of the confirmed A/B object.
    baseline = associate_observations(observations, frames)
    assert len(baseline['groups']) == 2
    result = associate_observations(observations, frames, confirmed_groups=[['a', 'b']])
    assert result['groups'] == [['a', 'b', 'c']]
    separated = associate_observations(observations, frames, confirmed_groups=[['a', 'b']],
                                      excluded_groups=[[['a', 'b'], ['c']]])
    assert separated['groups'] == [['a', 'b'], ['c']]
    assert any(link.get('reason') == 'identity_exclusion' for link in separated['links'])


def test_small_depth_consistent_fragment_is_not_complete_identity_support():
    a, b = frame(), frame('b')
    b.points[4:] *= 2
    result = associate_observations([MaskObservation('a', 'a', mask()), MaskObservation('b', 'b', mask())], {'a': a, 'b': b})
    assert result['groups'] == [['a'], ['b']]
    direction = result['links'][0]['directions'][0]
    assert direction['support'] == 48
    assert direction['visibleSupport'] == 144
    assert direction['depthAgreement'] == pytest.approx(1 / 3)


def test_confirmed_group_cannot_absorb_a_verifiable_conflict():
    frames = {name: frame(name) for name in ('a', 'b', 'c')}
    observations = [MaskObservation('a', 'a', mask(0, 5)), MaskObservation('b', 'b', mask()), MaskObservation('c', 'c', mask(7, 12))]
    result = associate_observations(observations, frames, confirmed_groups=[['a', 'b']])
    assert result['groups'] == [['a', 'b'], ['c']]


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


def scene_document(*, schema_version=2):
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
    if schema_version == 2:
        from ehs_spatial.platform.identity import migrate_document
        return migrate_document(document, base_revision_id=str(uuid4()))
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


@pytest.mark.parametrize('kind', ['observed_surface', 'generated_mesh', 'primitive'])
def test_export_excludes_stale_before_loading_but_retains_frozen_source(kind, tmp_path):
    doc = scene_document()
    entity = doc['entities'][0]
    rep = entity['representations'][0]
    rep.update(kind=kind, sourceValidity='stale')
    if kind != 'primitive':
        asset_id = str(uuid4())
        rep.update(assetId=asset_id, primitive=None)
        doc['assets'].append({'id':asset_id, 'kind':kind})
    if kind == 'observed_surface':
        entity.update(activeModelRepresentationId=None, currentModelTransform=None)
    before = deepcopy(doc)
    prepared = prepare_export('stale-source', doc, lambda _: pytest.fail('Stale geometry must be excluded before any asset is loaded'))
    assert prepared['excludedRepresentations'] == [{'entityId':entity['id'], 'representationId':rep['id'], 'reason':'source_geometry_stale'}]
    assert len(prepared['objects']) == 3 and all(item['id'] != rep['id'] for item in prepared['objects'])
    assert prepared['document'] == before and doc == before
    assert write_glb(prepared, tmp_path/'current.glb')['objects'] == 3


BLENDER = Path(os.environ.get("BLENDER_EXECUTABLE","/Users/adam/Desktop/panoptes-public/.tools/blender-4.5.9/Blender.app/Contents/MacOS/Blender"))


def candidate_scene():
    doc = scene_document()
    meshes = {}
    first, second, observed, context, _ = doc['entities']
    for entity in (first, second):
        rep = entity['representations'][0]
        rep.update(placementState='unconfirmed', placementReason='imported_proposal' if entity is first else 'requires_alignment_confirmation',
                   coverage='shape_proposal', sourceValidity='current', shapeStatus='ready',
                   sourceRefs=[{'assetId':doc['cameras'][0]['imageId'], 'sourceRecordId':entity['id']}],
                   placementSource={'type':'source_alignment_proposal'})
    first['currentModelTransform'] = {**deepcopy(first['currentModelTransform']), 'position':[9.,8.,7.]}
    alternative = deepcopy(first['representations'][0])
    alternative['id'] = str(uuid4())
    first['representations'].append(alternative)
    evidence = deepcopy(first['representations'][0])
    evidence['id'] = str(uuid4())
    first['representations'].append(evidence)
    for rep, kind in ((second['representations'][0], 'generated_mesh'), (evidence, 'observed_surface'),
                      (observed['representations'][0], 'observed_surface'), (context['representations'][0], 'observed_surface')):
        aid = str(uuid4())
        meshes[aid] = primitive_mesh(rep['primitive'])
        doc['assets'].append({'id':aid, 'kind':kind})
        rep.update(kind=kind, assetId=aid, primitive=None)
        if kind == 'observed_surface':
            rep.update(placementState='confirmed', placementReason='observed_surface', coverage='observed_partial')
    for entity in (observed, context):
        entity.update(activeModelRepresentationId=None, currentModelTransform=None)
    context['sourceContext'] = True
    return doc, meshes


def test_model_scene_exports_active_candidates_without_promoting_observed_evidence(tmp_path):
    from ehs_spatial.platform.blender_export import _read_glb
    doc, meshes = candidate_scene()
    before, calls = deepcopy(doc), []
    def resolve(aid):
        calls.append(aid)
        return meshes[aid]
    prepared = prepare_export('candidate-scene', doc, resolve)
    assert prepared['sceneMode'] == 'models' and prepared['status'] == 'incomplete'
    assert prepared['exportedModelCount'] == 2 and prepared['exportedObservedRepresentationCount'] == 0
    assert [item['id'] for item in prepared['objects']] == [entity['activeModelRepresentationId'] for entity in doc['entities'][:2]]
    assert calls == [doc['entities'][1]['representations'][0]['assetId']]
    assert prepared['missingModelEntities'] == [{'entityId':doc['entities'][2]['id'], 'reason':'active_model_missing'},
                                               {'entityId':doc['entities'][4]['id'], 'reason':'no_representation'}]
    assert len(prepared['placementPendingEntities']) == 2
    assert prepared['objects'][0]['transform'] == before['entities'][0]['currentModelTransform']
    for item in prepared['objects']:
        assert item['placementState'] == 'unconfirmed' and item['editable']
        assert item['coordinateFrameScale']['status'] == 'uncalibrated' and item['coordinateFrameScale']['nativeToMeters'] is None
        assert item['coverage'] == 'shape_proposal' and item['placementSource'] == {'type':'source_alignment_proposal'}
    validation = write_glb(prepared, tmp_path/'models.glb')
    glb, _ = _read_glb((tmp_path/'models.glb').read_bytes())
    assert validation['status'] == 'passed' and validation['objects'] == 2
    assert glb['extras']['sourceDocument'] == before
    assert glb['extras']['placementPendingEntities'] == prepared['placementPendingEntities']
    assert all(node['extras']['placementState'] == 'unconfirmed' for node in glb['nodes'])
    observed = prepare_export('candidate-scene', doc, meshes.__getitem__, scene_mode='observed')
    assert len(observed['objects']) == 3 and all(item['kind'] == 'observed_surface' and not item['editable'] for item in observed['objects'])
    assert doc == before and prepared['manifest']['sourceDocument'] == before
    with pytest.raises(PlatformError, match='invalid_export_scene_mode'):
        prepare_export('candidate-scene', doc, resolve, scene_mode='combined')


def test_model_scene_never_uses_unplaced_or_inactive_geometry():
    doc, _ = candidate_scene()
    doc['entities'][0]['representations'][0]['sourceValidity'] = 'stale'
    doc['entities'][1]['representations'][0]['placementReason'] = 'insufficient_observed_depth'
    prepared = prepare_export('unplaced', doc, lambda _:pytest.fail('No eligible model asset can be loaded'))
    assert prepared['objects'] == [] and prepared['placementPendingEntities'] == []
    assert {row['entityId'] for row in prepared['missingModelEntities']} == {entity['id'] for entity in doc['entities'] if not entity.get('sourceContext')}
    assert prepared['status'] == 'incomplete'


def test_model_scene_exempts_explicit_floor_evidence_but_never_uses_display_labels():
    doc, meshes = candidate_scene()
    floor = doc['entities'][2]
    floor.update(geometryRole='floor', geometryRoleSourceRefs=[{'assetId':doc['assets'][0]['id'], 'sourceRecordId':'verified-floor'}])
    doc['entities'][4]['label'] = 'floor'
    prepared = prepare_export('floor-reference', doc, meshes.__getitem__)
    assert prepared['missingModelEntities'] == [{'entityId':doc['entities'][4]['id'], 'reason':'no_representation'}]
    assert prepared['modelNotRequiredEntities'] == [{'entityId':floor['id'], 'reason':'reference_surface'}]
    assert prepared['manifest']['sourceDocument']['entities'][2] == floor
    assert all(item['entityId'] != floor['id'] for item in prepared['objects'])
    floor_only = prepare_export('floor-only', {**doc, 'entities':[floor]}, lambda _:pytest.fail('Reference evidence is retained without loading a model'))
    assert floor_only['status'] == 'succeeded' and floor_only['missingModelEntities'] == [] and floor_only['exportedModelCount'] == 0


@pytest.mark.parametrize('inputs', [{}, {'sceneMode':'observed'}])
def test_export_worker_forwards_scene_mode_and_keeps_candidate_status_separate(monkeypatch, inputs):
    from types import SimpleNamespace
    from ehs_spatial.platform import blender_export
    from panoptes_worker.__main__ import export_job
    doc, meshes = candidate_scene()
    doc['entities'] = doc['entities'][:2]
    mode = inputs.get('sceneMode', 'models')
    def fake_blender(revision_id, document, resolve, output_dir, executable, *, scene_mode):
        assert scene_mode == mode and document == doc
        output_dir.mkdir()
        return {**prepare_export(revision_id, document, meshes.__getitem__, scene_mode=scene_mode)['manifest'], 'validation':{'blender':{'status':'passed'}}}
    monkeypatch.setattr(blender_export, 'export_scene_revision', fake_blender)
    repository = SimpleNamespace(get_revision=lambda _: {'id':'fixed-revision', 'document':doc})
    result = export_job(repository, None, {'baseRevisionId':'fixed-revision', 'kind':'export_blender', 'inputs':inputs})
    assert result['sceneMode'] == mode and result['unplacedEntities'] == []
    assert result['validation']['validation']['blender']['status'] == 'passed'
    assert result['status'] == ('incomplete' if mode == 'models' else 'succeeded')
    assert len(result['placementPendingEntities']) == (2 if mode == 'models' else 0)


@pytest.mark.skipif(not BLENDER.exists(), reason='real Blender executable not installed')
def test_real_blender_candidate_scene_retains_status_source_and_parametric_editability(tmp_path):
    doc, meshes = candidate_scene()
    before = deepcopy(doc)
    result = export_scene_revision('candidate-scene', doc, meshes.__getitem__, tmp_path/'candidate', BLENDER)
    assert result['status'] == 'incomplete' and result['exportedModelCount'] == 2
    assert len(result['missingModelEntities']) == 2 and len(result['placementPendingEntities']) == 2
    assert result['sourceDocument'] == before and doc == before
    validation = result['validation']['blender']
    assert validation['status'] == 'passed' and result['validation']['glb']['status'] == 'passed'
    assert len(validation['objects']) == 2 and len(validation['parameters']) == 1
    assert all(item['editable'] and item['placementState'] == 'unconfirmed' and item['metadataReopened'] == 'passed' for item in validation['objects'])
    assert json.loads((tmp_path/'candidate/manifest.json').read_text()) == result


def test_export_point_cloud_context_excluded_but_object_stays_unplaced(tmp_path):
    doc = scene_document()
    meshes = {}
    for entity in doc['entities']:
        for rep in entity['representations']:
            aid = str(uuid4())
            meshes[aid] = primitive_mesh(rep['primitive'])
            doc['assets'].append({'id':aid,'kind':'observed_surface'})
            rep.update(kind='observed_surface', assetId=aid, primitive=None)
        entity.update(activeModelRepresentationId=None, currentModelTransform=None)
    doc["entities"][0]["sourceContext"] = True  # Mesh context still exports normally.
    original = prepare_export("revision", doc, meshes.__getitem__, scene_mode='observed')
    cloud = deepcopy(doc["entities"][0])
    cloud.update(id=str(uuid4()), sourceContext=True)
    rep = cloud["representations"][0]
    asset_id = str(uuid4())
    rep.update(id=str(uuid4()), kind="point_cloud", assetId=asset_id, primitive=None)
    doc["assets"].append({"id":asset_id, "kind":"point_cloud"})
    doc["entities"].append(cloud)
    context = prepare_export("revision", doc, meshes.__getitem__, scene_mode='observed')
    expected = [{"entityId":cloud["id"], "representationId":rep["id"], "reason":"point_cloud_context"}]
    assert context["unplacedEntities"] == original["unplacedEntities"]
    assert context["excludedRepresentations"] == expected
    assert context["objects"] == original["objects"] and context["cameras"] == original["cameras"]
    object_cloud = deepcopy(cloud)
    object_cloud.update(id=str(uuid4()), sourceContext=False)
    object_cloud["representations"][0]["id"] = str(uuid4())
    doc["entities"].append(object_cloud)
    prepared = prepare_export("revision", doc, meshes.__getitem__, scene_mode='observed')
    assert prepared["unplacedEntities"] == original["unplacedEntities"] + [{"entityId":object_cloud["id"], "representationId":object_cloud["representations"][0]["id"], "reason":"not_a_mesh"}]
    assert prepared["excludedRepresentations"] == expected
    if BLENDER.exists():
        result = export_scene_revision("revision", doc, meshes.__getitem__, tmp_path/"export", BLENDER, scene_mode='observed')
        assert result["excludedRepresentations"] == expected
        assert result["unplacedEntities"] == prepared["unplacedEntities"] and result["status"] == "incomplete"
        assert len(result["validation"]["blender"]["objects"]) == len(original["objects"])
        assert len(result["validation"]["blender"]["cameras"]) == len(original["cameras"])
        assert all(not item['editable'] for item in result['validation']['blender']['objects'])
        assert result['sceneMode'] == 'observed' and result['exportedModelCount'] == 0
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


def test_shared_reference_registration_retains_native_camera_and_rejects_bad_geometry():
    from ehs_spatial.platform.spatial import FrameGeometry, register_reference, registered_frame
    from scipy.spatial.transform import Rotation
    yy, xx = np.mgrid[:24,:32]
    depth = 3 + .003 * xx + .005 * yy
    k = np.array([[30.,0,15.5],[0,30.,11.5],[0,0,1]])
    points = np.stack(((xx-15.5)*depth/30,(yy-11.5)*depth/30,depth),-1)
    valid = np.ones(depth.shape,bool)
    source = FrameGeometry('photo','native','same-hash',points,valid,k,np.eye(4))
    expected = np.eye(4)
    expected[:3,:3] = 2.7 * Rotation.from_euler('xyz',[.3,-.2,.8]).as_matrix()
    expected[:3,3] = [4,-2,1]
    target = registered_frame(source,expected,'project')
    matrix, report = register_reference(source,target,valid, source_input_to_canonical=np.eye(3), target_input_to_canonical=np.eye(3))
    assert np.allclose(matrix,expected,atol=1e-8)
    assert report['holdoutCount'] > 32 and report['holdoutP95Relative'] < 1e-8
    assert np.allclose(target.camera_to_world[:3,:3].T @ target.camera_to_world[:3,:3],np.eye(3))
    assert np.array_equal(source.camera_to_world,np.eye(4))
    with pytest.raises(PlatformError,match='background_insufficient'):
        register_reference(source,target,np.zeros_like(valid), source_input_to_canonical=np.eye(3), target_input_to_canonical=np.eye(3))
    warped = target.points.copy()
    warped[::2] += [1,2,3]
    with pytest.raises(PlatformError,match='registration_residual_failed'):
        register_reference(source,FrameGeometry('photo','project','same-hash',warped,valid,k,target.camera_to_world),valid, source_input_to_canonical=np.eye(3), target_input_to_canonical=np.eye(3))
    line = points.copy(); line[:] = np.stack((xx,np.zeros_like(xx),np.ones_like(xx)),-1)
    with pytest.raises(PlatformError,match='registration_degenerate'):
        from ehs_spatial.platform.spatial import similarity_transform
        similarity_transform(line.reshape(-1,3),line.reshape(-1,3))


def test_reference_registration_uses_pixel_mapping_and_cross_checks_reference_camera():
    from ehs_spatial.platform.spatial import FrameGeometry, register_reference
    yy, xx = np.mgrid[:32,:32]
    valid = np.ones(xx.shape,bool)
    def plane(focal, frame):
        k = np.array([[focal,0,15.5],[0,focal,15.5],[0,0,1]])
        points = np.stack(((xx-15.5)*2/focal,(yy-15.5)*2/focal,np.full_like(xx,2)),axis=-1)
        return FrameGeometry('same-photo',frame,'same-hash',points,valid,k,np.eye(4))
    # Identical input pixels, different predicted intrinsics. A plane alone lets
    # a misleading similarity fit pass; the same reference camera disproves it.
    with pytest.raises(PlatformError,match='registration_camera_reprojection_failed'):
        register_reference(plane(20,'new'),plane(10,'old'),valid,
            source_input_to_canonical=np.eye(3),target_input_to_canonical=np.eye(3))
    # A real crop has an explicit pixel translation, including changed K.
    target = plane(10,'old')
    crop = np.array([[1.,0,-4],[0,1,0],[0,0,1]])
    source = FrameGeometry('same-photo','new','same-hash',target.points[:,4:],valid[:,4:],crop@target.K,np.eye(4))
    matrix, result = register_reference(source,target,valid,
        source_input_to_canonical=crop,target_input_to_canonical=np.eye(3))
    assert np.allclose(matrix,np.eye(4))
    assert result['cameraReprojectionP95Pixels'] < 1e-8
    assert result['referenceCameraPositionRelative'] < 1e-8


def test_frozen_geometry_cannot_override_explicitly_unbound_or_wrong_solution():
    from types import SimpleNamespace
    from ehs_spatial.platform.reconstruction import _load_geometry
    document = {'schemaVersion':2, 'assets':[], 'cameras':[{'id':'old-camera','imageId':'photo','coordinateFrameId':'frame'}],
        'geometryBindings':{'photo':None},
        'geometryEvidence':{'coordinateFrameId':'frame','manifestAssetId':'manifest',
            'frames':[{'assets':{'input':'photo'},'cameraId':'old-camera'}]}}
    with pytest.raises(PlatformError,match='geometry_evidence_unavailable'):
        _load_geometry(document,[],SimpleNamespace())
    document['geometryBindings']['photo'] = {'cameraId':'old-camera','geometrySolutionId':'different-solve'}
    with pytest.raises(PlatformError,match='geometry_solution_binding_mismatch'):
        _load_geometry(document,[],SimpleNamespace())


@pytest.mark.skipif(not BLENDER.exists(), reason='real Blender executable not installed')
def test_large_revision_source_text_preserves_full_document_on_reopen(tmp_path):
    import time
    from ehs_spatial.platform.contracts import digest
    doc = scene_document()
    # A real source revision carries large evidence tables. Single-line Text.write
    # makes this CPU-bound before saving, independent of mesh size or UVs.
    doc['reportEvidence'] = {'records': [{'observationId': f'observation-{i}', 'support': [0.25, 0.5, 0.75], 'reason': '保留完整原始来源'} for i in range(25000)]}
    start = time.monotonic()
    result = export_scene_revision('large-source-text', doc, lambda _: pytest.fail('no assets'), tmp_path/'large-source', BLENDER)
    assert time.monotonic() - start < 30, 'Source text insertion regressed to the long-line slow path'
    assert result['validation']['blender']['status'] == 'passed'
    assert result['validation']['glb']['status'] == 'passed'
    assert result['documentSha256'] == digest(doc)


def test_v2_active_material_edit_exports_its_color_and_preserves_source_candidates(tmp_path):
    from ehs_spatial.platform.blender_export import _read_glb
    from ehs_spatial.platform.identity import migrate_document
    from ehs_spatial.platform.repository import apply_operations
    source = scene_document(schema_version=1)
    source['entities'] = [source['entities'][0]]
    entity = source['entities'][0]
    entity['material'] = {'baseColorFactor': [0.9, 0.8, 0.7, 1.0], 'roughness': 0.8}
    active = entity['representations'][0]
    shape = primitive_mesh(active['primitive'])
    source_mesh = MeshData(shape.vertices, shape.faces, shape.colors, material={'baseColorFactor': [0.4, 0.6, 0.8, 0.7]})
    asset_id = str(uuid4())
    source['assets'].append({'id': asset_id, 'kind': 'generated_mesh'})
    active.update(kind='generated_mesh', assetId=asset_id, primitive=None)
    active['material'] = {'baseColorFactor': [0.4, 0.6, 0.8, 0.7], 'roughness': 0.2}
    entity['currentModelTransform'] = deepcopy(active['transform'])
    candidate = deepcopy(active)
    candidate.update(id=str(uuid4()), material={'baseColorFactor': [0.1, 0.2, 0.3, 1.0]})
    candidate['transform']['position'][0] += 10
    entity['representations'].append(candidate)
    base = str(uuid4())
    doc = migrate_document(source, base_revision_id=base)
    updated, _ = apply_operations(doc, [{'type': 'setMaterial', 'entityId': entity['id'], 'material': {'color': [0.5, 0.25, 1.0], 'roughness': 0.3}}], base_revision_id=base)
    kept = updated['entities'][0]
    assert kept['material'] == entity['material']
    assert next(r for r in kept['representations'] if r['id'] == candidate['id']) == candidate
    prepared = prepare_export('active-color', updated, lambda _: source_mesh)
    assert len(prepared['objects']) == 1
    assert prepared['objects'][0]['material'] == {'color': [0.5, 0.25, 1.0], 'roughness': 0.3}
    assert prepared['objects'][0]['parts'][0]['material']['baseColorFactor'] == [0.2, 0.15, 0.8, 0.7]
    assert prepared['objects'][0]['transform'] == kept['currentModelTransform'] == next(r for r in kept['representations'] if r['id'] == kept['activeModelRepresentationId'])['transform']
    write_glb(prepared, tmp_path/'active.glb')
    glb, _ = _read_glb((tmp_path/'active.glb').read_bytes())
    assert glb['materials'][0]['pbrMetallicRoughness']['baseColorFactor'] == [0.2, 0.15, 0.8, 0.7]
    assert glb['materials'][0]['pbrMetallicRoughness']['roughnessFactor'] == 0.3
    legacy = prepare_export('legacy-color', source, lambda _: source_mesh)
    assert legacy['objects'] == [] and legacy['missingModelEntities'] == [{'entityId':entity['id'], 'reason':'active_model_missing'}]
    assert source['entities'][0]['material'] == entity['material'] and doc['entities'][0]['representations'][0]['material'] == active['material']
