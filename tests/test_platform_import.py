"""Generic public-scene import invariants, without using a named demo in logic."""
import gzip
import hashlib
import io
import json
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from PIL import Image
import pytest

from ehs_spatial.platform.contracts import PlatformError
from ehs_spatial.platform.spatial import primitive_mesh, transform_matrix, transform_points
from ehs_spatial.platform.repository import apply_operations
from scripts.import_public_scene import floor_evidence, floor_mesh_members, import_document, legacy_transform, packed_asset, parametric_definition, source_path, unpack_mesh


def make_public_scene(root):
    vertices = np.array([[0,0,2,0,0,1,1,0,0],[1,0,2,0,0,1,0,1,0],[0,1,2,0,0,1,0,0,1]], dtype="<f4")
    faces = np.array([0,1,2], dtype="<u4")
    raw = vertices.tobytes()+faces.tobytes()
    def packed(name, data):
        value = gzip.compress(data)
        (root/name).write_bytes(value)
        return {"path":name,"bytes":len(data),"packed_bytes":len(value),"sha256":hashlib.sha256(data).hexdigest()}
    asset = packed("context.bin.gz",raw)
    region = packed("region.faces.gz",np.array([0],dtype="<u4").tobytes())
    for name,mode in (("photo.png","RGB"),("mask.png","L")):
        Image.new(mode,(8,6),255).save(root/name)
    mask = (root/'mask.png').read_bytes()
    transform = {"position":[0.,0.,0.],"rotation_deg":[0.,0.,0.],"scale":[1.,1.,1.]}
    mesh = {"byte_offset":0,"vertex_count":3,"stride":9,"index_byte_offset":108,"index_count":3,"index_type":"uint32","asset":asset}
    scene = {"version":1,"run_id":"arbitrary-source","label":"Generic fixture","target":"standalone_object","units":"uncalibrated",
        "cameras":[{"id":"camera-a","image":"photo.png","width":8,"height":6,"K":[[9,.2,3.5],[0,8,2.5],[0,0,1]],"camera_to_world":np.eye(4).tolist()}],
        "objects":[{"id":"context","label":"Background","source":"observed","role":"context","frame_ids":["camera-a"],"transform":transform,"mesh":mesh},
            {"id":"generated","label":"Proposal","source":"generated","frame_ids":["camera-a"],"transform":{**transform,"rotation_deg":[17,-23,31],"scale":[2,3,4]},"mesh":mesh}],
        "observed_regions":[{"id":"tiny","label":"Tiny object","source":"observed","reference_frame":"camera-a","context_id":"context",
            "mask":{"path":"mask.png","sha256":hashlib.sha256(mask).hexdigest(),"bbox_xyxy":[1,1,2,2],"shape_hw":[6,8],"resolution":"canonical"},
            "source_bbox":{"bbox_xyxy":[1,1,2,2],"resolution":"canonical","shape_hw":[6,8]},"faces":{"count":1,"asset":region}}],
        "unavailable_objects":[{"id":f"missing-{i}","label":"Same label","reason":"No mesh"} for i in range(27)]}
    path = root/'scene.json'
    path.write_text(json.dumps(scene))
    return path


def test_generic_import_retains_context_small_masks_missing_mesh_and_native_camera(tmp_path):
    path = make_public_scene(tmp_path)
    stored = {}
    def put(data,media_type,metadata):
        sha=hashlib.sha256(data).hexdigest()
        identity=str(uuid5(NAMESPACE_URL,sha))
        stored[identity]=data
        return {"id":identity,"sha256":sha,"sizeBytes":len(data),"mediaType":media_type,"metadata":metadata}
    document,manifest = import_document(path,put)
    assert len(document['entities']) == 30
    assert sum(not e['representations'] for e in document['entities']) == 27
    assert document['observations'][0]['originalPixelBox'] == [1,1,2,2]
    assert document['observations'][0]['maskAssetId'] in stored
    assert document['cameras'][0]['K'] == [[9,.2,3.5],[0,8,2.5],[0,0,1]]
    assert document['target']=='standalone_object' and document['coordinateFrames'][0]['ground'] is None
    assert all(e['associationState']=='association_pending' for e in document['entities'])
    assert len(set(manifest['entityIds'].values()))==30


def test_model_report_inputs_pin_exact_observation_versions_without_rewriting_sources(tmp_path):
    path = make_public_scene(tmp_path)
    source = json.loads(path.read_text())
    Image.new('RGB', (8, 6), 'blue').save(tmp_path/'second.png')
    source['cameras'].append({**source['cameras'][0], 'id': 'camera-b', 'image': 'second.png'})
    path.write_text(json.dumps(source))
    view = {'frame_id': 'camera-a', 'bbox': [1, 1, 2, 2], 'polygons': [[[1, 1], [1, 2], [2, 2]]]}
    report = {'reconstruction_run_id': source['run_id'], 'scene_url': path.name,
        'frames': [{'id': 'camera-a', 'url': 'photo.png'}, {'id': 'camera-b', 'url': 'second.png'}],
        'objects': [{'id': 'report-object', 'scene_object_id': 'generated', 'views': [view, view, {**view, 'frame_id': 'camera-b'}]},
                    {'id': 'unmapped', 'scene_object_id': 'absent', 'label': 'Proposal', 'views': [view]}], 'plan': {}}
    report_path = tmp_path/'report.json'
    report_path.write_text(json.dumps(report))
    before = {p: p.read_bytes() for p in (path, report_path)}
    stored = {}
    def put(data, media_type, metadata):
        sha = hashlib.sha256(data).hexdigest()
        aid = str(uuid5(NAMESPACE_URL, sha))
        stored[aid] = data
        return {'id': aid, 'sha256': sha, 'sizeBytes': len(data), 'mediaType': media_type, 'metadata': metadata}
    document, manifest = import_document(path, put)
    entity = next(e for e in document['entities'] if e['id'] == manifest['entityIds']['generated'])
    refs = entity['representations'][0]['sourceRefs']
    artifact, evidence = refs
    assert artifact['role'] == 'model_artifact' and artifact['sourceRecordId'] == 'generated'
    assert artifact['sha256'] == hashlib.sha256(before[path]).hexdigest()
    observation = next(o for o in document['observations'] if o['id'] == evidence['observationId'])
    assert evidence['revision'] == observation['revision'] == 1
    assert evidence['imageId'] == observation['imageId']
    assert evidence['imageSha256'] == hashlib.sha256(stored[observation['imageId']]).hexdigest()
    assert evidence['sha256'] == hashlib.sha256(before[report_path]).hexdigest()
    assert evidence['sourceRecordId'] == 'generated' and evidence['sourceFrameId'] == 'camera-a'
    assert evidence['jsonPointer'] == '/objects/0/views/0'
    # A report-only camera is still available as evidence, but was not a declared model input.
    assert len(entity['observationRefs']) == 2 and len(refs) == 2
    assert document['reportEvidence']['objects'][1]['entityId'] is None
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert import_document(path, put)[1]['documentSha256'] == manifest['documentSha256']
    from ehs_spatial.platform.identity import migrate_document
    from ehs_spatial.platform.correspondence import audit_correspondence
    document['captureId'] = str(uuid5(NAMESPACE_URL, 'test-import-capture'))
    current = migrate_document(document, base_revision_id=str(uuid5(NAMESPACE_URL, 'test-import-revision')))
    audited = next(r for r in audit_correspondence(current)['rows'] if r['entityId'] == entity['id'])
    assert audited['modelCurrent'] and audited['sourceErrors'] == []


def test_explicit_report_views_share_entity_reuse_anchor_and_retain_unmapped_evidence(tmp_path):
    path = make_public_scene(tmp_path)
    source = json.loads(path.read_text())
    Image.new('RGB', (8, 6), 'blue').save(tmp_path/'second-canonical.png')
    Image.new('RGB', (16, 12), 'blue').save(tmp_path/'second-original.png')
    mapping = np.array([[.5, 0, -.25], [0, .5, -.25], [0, 0, 1]])
    source['cameras'].append({**source['cameras'][0], 'id': 'camera-b', 'image': 'second-canonical.png',
        'original_image': 'second-original.png', 'original_width': 16, 'original_height': 12,
        'original_K': (np.linalg.inv(mapping) @ np.asarray(source['cameras'][0]['K'])).tolist(),
        'input_to_canonical_pixel_centres': mapping.tolist()})
    source['objects'].append({**source['objects'][0], 'id': 'floor-observed', 'label': 'Observed floor', 'role': None})
    path.write_text(json.dumps(source))
    def put(data, media_type, metadata):
        sha = hashlib.sha256(data).hexdigest()
        return {'id': str(uuid5(NAMESPACE_URL, sha)), 'sha256': sha, 'sizeBytes': len(data), 'mediaType': media_type, 'metadata': metadata}
    baseline, before = import_document(path, put)
    anchor = next(o for o in baseline['observations'] if o['id'] == before['observationIds']['tiny'])
    view = {'frame_id': 'camera-a', 'bbox': [1, 1, 2, 2], 'polygons': [[[1, 1], [1, 2], [2, 2]]]}
    report = {'reconstruction_run_id': source['run_id'], 'scene_url': path.name,
        'frames': [{'id': 'camera-a', 'url': 'photo.png'}, {'id': 'camera-b', 'url': 'second-canonical.png'}],
        'objects': [
            {'id': 'report-row', 'scene_object_id': 'tiny', 'views': [view, {**view, 'frame_id': 'camera-b'}, view]},
            {'id': 'floor-row', 'scene_object_id': 'floor-observed', 'views': [view]},
            {'id': 'unmapped-row', 'scene_object_id': 'absent', 'label': 'Tiny object', 'views': [view]},
        ], 'plan': {}}
    (tmp_path/'report.json').write_text(json.dumps(report))
    document, manifest = import_document(path, put)
    entities = {entity['id']: entity for entity in document['entities']}
    tiny = entities[manifest['entityIds']['tiny']]
    observations = {o['id']: o for o in document['observations']}
    assert manifest['entityIds'] == before['entityIds']
    assert manifest['observationIds']['tiny'] == anchor['id']
    assert len(document['observations']) == 3 and len(tiny['observationRefs']) == 2
    reused = observations[anchor['id']]
    assert reused['maskAssetId'] == anchor['maskAssetId'] and reused['originalPixelBox'] == anchor['originalPixelBox']
    second = next(observations[oid] for oid in tiny['observationRefs'] if oid != anchor['id'])
    assert second['originalPixelBox'] == [2, 2, 4, 4]
    assert second['originalPixelPolygons'] == [[[2.5, 2.5], [2.5, 4.5], [4.5, 4.5]]]
    assert second['maskAssetId'] is None and second['geometrySupport'] is None
    assert second['missingEvidence'] == ['source_mask_not_packaged']
    binding = second['sourceRefs'][0]
    assert binding['sourceRecordId'] == 'tiny' and binding['sourceFrameId'] == 'camera-b'
    assert binding['imageSha256'] == hashlib.sha256((tmp_path/'second-original.png').read_bytes()).hexdigest()
    assert binding['canonicalImageSha256'] == hashlib.sha256((tmp_path/'second-canonical.png').read_bytes()).hexdigest()
    assert binding['jsonPointer'] == '/objects/0/views/1'
    assert tiny['associationState'] == 'confirmed'
    assert tiny['associationEvidence']['method'] == 'explicit_source_id'
    assert tiny['associationEvidence']['observationIds'] == tiny['observationRefs']
    assert tiny['lineage'][0] == next(e for e in baseline['entities'] if e['id'] == tiny['id'])['lineage'][0]
    assert tiny['lineage'][-1]['observationIds'] == tiny['observationRefs']
    views = document['reportEvidence']['objects'][0]['views']
    assert [v['observationId'] for v in views] == [anchor['id'], second['id'], anchor['id']]
    floor = entities[manifest['entityIds']['floor-observed']]
    assert len(floor['observationRefs']) == 1 and floor['associationState'] == 'association_pending'
    assert len(floor['representations']) == 1 and 'associationEvidence' not in floor
    unknown = document['reportEvidence']['objects'][2]
    assert unknown['entityId'] is None and 'observationId' not in unknown['views'][0]
    assert import_document(path, put)[1]['documentSha256'] == manifest['documentSha256']
    assert manifest['converter']['version'] == 'public-scene-v9'
    report['frames'][1]['url'] = 'photo.png'
    (tmp_path/'report.json').write_text(json.dumps(report))
    with pytest.raises(PlatformError, match='import_report_image_mismatch'):
        import_document(path, put)
    source['cameras'][1]['input_to_canonical_pixel_centres'] = np.eye(3).tolist()
    path.write_text(json.dumps(source))
    with pytest.raises(PlatformError, match='import_pixel_mapping_mismatch'):
        import_document(path, put)


def test_native_euler_exact_order_and_content_hash_validation(tmp_path):
    value={"position":[1,2,3],"rotation_deg":[17,-23,31],"scale":[2,3,4]}
    x,y,z=np.deg2rad(value['rotation_deg'])
    rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    rz=np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
    expected=np.eye(4);expected[:3,:3]=rz@ry@rx@np.diag(value['scale']);expected[:3,3]=value['position']
    assert np.allclose(transform_matrix(legacy_transform(value,'frame')),expected,atol=1e-14)
    path=make_public_scene(tmp_path)
    asset=json.loads(path.read_text())['objects'][0]['mesh']['asset']
    assert packed_asset(tmp_path,asset)
    with pytest.raises(PlatformError,match='import_asset_hash_mismatch'):
        packed_asset(tmp_path,{**asset,'sha256':'0'*64})
    with pytest.raises(PlatformError,match='import_asset_path_invalid'):
        source_path(tmp_path,'../outside.json')


def make_parametric_scene(root):
    path = make_public_scene(root)
    source = json.loads(path.read_text())
    record = source['objects'][1]
    spec = {'kind':'cylinder', 'radius':.3, 'height':1.7, 'segments':16}
    mesh = primitive_mesh(spec)
    pose = legacy_transform({'position':[2.,-1.,4.], 'rotation_deg':[-37.,15.,22.], 'scale':[1.,1.,1.]}, 'native')
    native = transform_matrix(pose)
    # Baked source vertices use a different rotated, non-unit-scaled object frame.
    local = transform_points(transform_points(mesh.vertices, native), np.linalg.inv(transform_matrix(legacy_transform(record['transform'], 'native'))))
    vertices = np.column_stack((local, np.zeros_like(local), np.tile([.18,.24,.3], (len(local),1)))).astype('<f4')
    faces = mesh.faces.copy()
    n = spec['segments']
    for i in range(n):
        j = (i+1) % n
        faces[4*i:4*i+2] = [[i,j,n+i], [j,n+j,n+i]]
    raw = vertices.tobytes() + faces.astype('<u4').tobytes()
    packed = gzip.compress(raw)
    (root/'cylinder.bin.gz').write_bytes(packed)
    params = {'radius_native':spec['radius'], 'height_native':spec['height']}
    definition = {'primitive':{'type':'cylinder','radius':1,'height':1,'segments':n,'local_axis':'+Z'},
                  'parameter_source':'Frozen generic fit', 'limitations':['Circular cross-section prior; no physical scale'],
                  'objects':[{'object_id':record['id'], 'fitted_parameters':params,
                              'native_object_to_world':(native @ np.diag([spec['radius'],spec['radius'],spec['height'],1])).tolist()}]}
    (root/'parameters.json').write_text(json.dumps(definition))
    record.update(source='parametric', parameters=params, metrics={'parameter_source':'parameters.json'},
                  mesh={'byte_offset':0,'vertex_count':len(vertices),'stride':9,'index_byte_offset':vertices.nbytes,
                        'index_count':faces.size,'index_type':'uint32',
                        'asset':{'path':'cylinder.bin.gz','bytes':len(raw),'packed_bytes':len(packed),'sha256':hashlib.sha256(raw).hexdigest()}})
    path.write_text(json.dumps(source))
    return path, spec, native


def test_parametric_import_restores_editable_dimensions_and_proves_source_surface(tmp_path):
    path, spec, native = make_parametric_scene(tmp_path)
    source_bytes, parameter_bytes = path.read_bytes(), (tmp_path/'parameters.json').read_bytes()
    stored = {}
    def put(data, media_type, metadata):
        sha = hashlib.sha256(data).hexdigest()
        identity = str(uuid5(NAMESPACE_URL, sha))
        stored[identity] = data
        return {'id':identity,'sha256':sha,'sizeBytes':len(data),'mediaType':media_type,'metadata':metadata}
    document, manifest = import_document(path, put)
    entity = next(item for item in document['entities'] if item['id'] == manifest['entityIds']['generated'])
    rep = entity['representations'][0]
    namespace = uuid5(NAMESPACE_URL, 'panoptes-public:' + hashlib.sha256(source_bytes).hexdigest())
    assert entity['id'] == str(uuid5(namespace, 'entity:generated')) and rep['id'] == str(uuid5(namespace, 'representation:generated'))
    assert rep['kind'] == 'primitive' and rep['assetId'] is None and rep['primitive'] == spec
    assert rep['placementState'] == 'unconfirmed' and rep['placementReason'] == 'imported_proposal'
    assert rep['transform'] == entity['currentModelTransform'] and rep['transform']['scale'] == [1.,1.,1.]
    assert np.allclose(transform_matrix(rep['transform']), native, atol=1e-12)
    assert np.allclose(rep['material']['color'], [.18,.24,.3])
    mesh = primitive_mesh(spec)
    assert rep['bounds'] == {'min':mesh.vertices.min(axis=0).tolist(), 'max':mesh.vertices.max(axis=0).tolist()}
    source_ref, mesh_ref, parameter_ref = rep['sourceRefs']
    assert all(ref['role'] == 'model_artifact' for ref in rep['sourceRefs'])
    assert all(ref['sha256'] == hashlib.sha256(stored[ref['assetId']]).hexdigest() for ref in rep['sourceRefs'])
    assert stored[source_ref['assetId']] == source_bytes
    assert mesh_ref['assetId'] == manifest['representationAssetIds']['generated']
    assert stored[parameter_ref['assetId']] == parameter_bytes
    assert parameter_ref['sha256'] == hashlib.sha256(parameter_bytes).hexdigest()
    assert parameter_ref['proof']['surfaceCoverage'] == 'equivalent' and parameter_ref['proof']['surfaceFacetCount'] == 18
    assert parameter_ref['proof']['maxVertexErrorNative'] <= parameter_ref['proof']['toleranceNative']
    assert path.read_bytes() == source_bytes and (tmp_path/'parameters.json').read_bytes() == parameter_bytes
    assert import_document(path, put)[1]['documentSha256'] == manifest['documentSha256']


def test_parametric_import_rejects_changed_parameters_and_equal_vertex_surface_corruption(tmp_path):
    path, spec, _ = make_parametric_scene(tmp_path)
    source = json.loads(path.read_text())
    record = source['objects'][1]
    vertices, faces = unpack_mesh(tmp_path, record['mesh'])
    corrupted = faces.copy()
    corrupted[0] = corrupted[4]  # Same vertex set and triangle count, but one missing side triangle.
    with pytest.raises(PlatformError, match='import_parameter_surface_mismatch'):
        parametric_definition(tmp_path, record, (vertices, corrupted), 'native')
    definition = json.loads((tmp_path/'parameters.json').read_text())
    definition['objects'][0]['fitted_parameters']['radius_native'] = spec['radius'] * 1.1
    (tmp_path/'parameters.json').write_text(json.dumps(definition))
    with pytest.raises(PlatformError, match='import_parameter_values_mismatch'):
        parametric_definition(tmp_path, record, (vertices, faces), 'native')
    record['parameters']['radius_native'] *= 1.1
    definition['objects'][0]['native_object_to_world'] = (np.asarray(definition['objects'][0]['native_object_to_world']) @ np.diag([1.1,1.1,1,1])).tolist()
    (tmp_path/'parameters.json').write_text(json.dumps(definition))
    with pytest.raises(PlatformError, match='import_parameter_mesh_mismatch'):
        parametric_definition(tmp_path, record, (vertices, faces), 'native')
    path, _, _ = make_parametric_scene(tmp_path)
    source = json.loads(path.read_text())
    asset = source['objects'][1]['mesh']['asset']
    raw = bytearray(gzip.decompress((tmp_path/asset['path']).read_bytes()))
    np.frombuffer(raw, dtype='<f4')[6] = .8
    packed = gzip.compress(raw)
    (tmp_path/asset['path']).write_bytes(packed)
    asset.update(packed_bytes=len(packed), sha256=hashlib.sha256(raw).hexdigest())
    path.write_text(json.dumps(source))
    with pytest.raises(PlatformError, match='import_parameter_material_not_uniform'):
        import_document(path, lambda data, media_type, metadata: {'id':str(uuid5(NAMESPACE_URL, hashlib.sha256(data).hexdigest())), 'sha256':hashlib.sha256(data).hexdigest(), 'sizeBytes':len(data), 'mediaType':media_type})


def test_import_normalizes_observed_basis_without_report_and_retains_floor_query(tmp_path):
    path = make_public_scene(tmp_path)
    source = json.loads(path.read_text())
    region = source['observed_regions'][0]
    region['label'] = 'Arbitrary display name'
    region['measurements'] = {'status':'available','dimensions_native':{'height':0,'width':2,'depth':3},
        'basis':{'kind':'floor_aligned_native','axes_native':np.eye(3).tolist(),'corners_native':[[0,0,2],[1,0,2],[0,1,2]]}}
    query = {'type':'text-sam','label':'floor','path':'saved-query.json','sha256':'a'*64,'pointer':['rle',0]}
    region['provenance'] = {'source_record':{'source_refs':[query]}}
    source['objects'][1]['label'] = 'floor'
    path.write_text(json.dumps(source))
    def put(data,media_type,metadata):
        sha=hashlib.sha256(data).hexdigest()
        return {'id':str(uuid5(NAMESPACE_URL,sha)),'sha256':sha,'sizeBytes':len(data),'mediaType':media_type,'metadata':metadata}
    document,manifest = import_document(path,put)
    assert 'reportEvidence' not in document
    observed = next(e for e in document['entities'] if e['id']==manifest['entityIds']['tiny'])
    assert observed['measurements']['basis'] == {'kind':'floor_aligned_native','axesNative':np.eye(3).tolist(),'cornersNative':[[0,0,2],[1,0,2],[0,1,2]]}
    assert observed['measurements']['groundHeightNative'] == 0
    assert observed['geometryRole'] == 'floor'
    assert observed['geometryRoleSourceRefs'][0]['sourceEvidence'] == query
    observation = next(o for o in document['observations'] if o['id'] in observed['observationRefs'])
    assert observation['labelEvidence'][0]['geometryRole'] == 'floor'
    assert observation['labelEvidence'][0]['source'] == 'imported_segmentation_query'
    assert not next(e for e in document['entities'] if e['label']=='floor').get('geometryRole'), 'A display label never establishes a geometric role'
    assert json.loads(path.read_text()) == source, 'Conversion must not mutate source artifacts'


def test_imported_bounds_measures_exact_photo_binding_without_automatic_acceptance(tmp_path):
    path = make_public_scene(tmp_path)
    source = json.loads(path.read_text())
    source['floor_plane'] = [0,-2,0,4]
    source['object_evidence'] = 'evidence.json'
    source['objects'][1]['measurements'] = {'status':'available','dimensions_native':{'height':1,'width':2,'depth':3},
        'source':{'candidate_id':'generated','frame_id':'camera-a','image_sha256':hashlib.sha256((tmp_path/'photo.png').read_bytes()).hexdigest(),'mask_sha256':'a'*64}}
    (tmp_path/'evidence.json').write_text(json.dumps({'candidates':[{'id':'generated','frame_id':'camera-a','mask':{'sha256':'a'*64,'bbox':[1,1,3,4],'resolution':'original','shape_hw':[6,8]}}]}))
    path.write_text(json.dumps(source))
    def put(data,media_type,metadata):
        sha=hashlib.sha256(data).hexdigest()
        return {'id':str(uuid5(NAMESPACE_URL,sha)),'sha256':sha,'sizeBytes':len(data),'mediaType':media_type,'metadata':metadata}
    document,_ = import_document(path,put)
    generated = document['entities'][1]
    rep = generated['representations'][0]
    assert rep['bounds'] == {'min':[0,0,2],'max':[1,1,2]}
    assert rep['placementState']=='unconfirmed' and rep['placementReason']=='imported_proposal'
    assert generated['measurements']['unit']=='native' and generated['measurements']['uncertaintyNative'] is None
    assert document['coordinateFrames'][0]['ground']['normal']==[0,-1,0]
    observation = next(o for o in document['observations'] if o['id'] in generated['observationRefs'])
    assert observation['originalPixelBox']==[1,1,3,4] and observation['maskAssetId'] is None
    assert observation['missingEvidence']==['source_mask_not_packaged']
    accepted,_=apply_operations(document,[{'type':'setTransform','entityId':generated['id'],'transform':rep['transform']}])
    assert accepted['entities'][1]['representations'][0]['placementState']=='unconfirmed'
    assert accepted['entities'][1]['representations'][0].get('placementSource') is None
    assert generated['representations'][0]['placementState']=='unconfirmed'
    source['objects'][1]['measurements']['source']['image_sha256']='b'*64
    path.write_text(json.dumps(source))
    rejected,_=import_document(path,put)
    assert rejected['entities'][1]['observationRefs']==[]
    assert rejected['entities'][1]['missingEvidence']==['source_observation_binding_pending']


@pytest.mark.parametrize('parametric', [False, True])
@pytest.mark.parametrize('with_report', [False, True])
def test_exact_candidate_inputs_pin_owned_observations_with_or_without_report(tmp_path, parametric, with_report):
    from copy import deepcopy
    from ehs_spatial.platform.correspondence import audit_correspondence
    from ehs_spatial.platform.identity import migrate_document

    path = make_parametric_scene(tmp_path)[0] if parametric else make_public_scene(tmp_path)
    source = json.loads(path.read_text())
    image_sha = hashlib.sha256((tmp_path/'photo.png').read_bytes()).hexdigest()
    source['object_evidence'] = 'evidence.json'
    source['objects'][1]['measurements'] = {'source': {
        'candidate_id': 'generated', 'frame_id': 'camera-a', 'image_sha256': image_sha, 'mask_sha256': 'a'*64}}
    evidence = {'candidates': [{'id': 'generated', 'label': 'Different display name', 'frame_id': 'camera-a',
        'mask': {'sha256': 'a'*64, 'bbox': [1, 1, 3, 4], 'resolution': 'original', 'shape_hw': [6, 8]}}]}
    evidence_path = tmp_path/'evidence.json'
    evidence_path.write_text(json.dumps(evidence))
    path.write_text(json.dumps(source))
    if with_report:
        view = {'frame_id': 'camera-a', 'bbox': [1, 1, 3, 4], 'polygons': []}
        (tmp_path/'report.json').write_text(json.dumps({'reconstruction_run_id': source['run_id'], 'scene_url': path.name,
            'frames': [{'id': 'camera-a', 'url': 'photo.png'}],
            'objects': [{'id': 'report-row', 'scene_object_id': 'generated', 'views': [view, view]}], 'plan': {}}))
    before = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    def put(data, media_type, metadata):
        sha = hashlib.sha256(data).hexdigest()
        return {'id': str(uuid5(NAMESPACE_URL, sha)), 'sha256': sha, 'sizeBytes': len(data), 'mediaType': media_type, 'metadata': metadata}

    document, manifest = import_document(path, put)
    entity = next(e for e in document['entities'] if e['id'] == manifest['entityIds']['generated'])
    rep = entity['representations'][0]
    inputs = [ref for ref in rep['sourceRefs'] if ref.get('observationId')]
    assert len(inputs) == 1  # Repeated report views reuse the already pinned candidate input.
    observation = next(o for o in document['observations'] if o['id'] == inputs[0]['observationId'])
    assert entity['observationRefs'] == [observation['id']]
    assert inputs[0] == {**observation['sourceRefs'][0], 'observationId': observation['id'], 'revision': 1,
        'imageId': observation['imageId'], 'sha256': hashlib.sha256(before[evidence_path]).hexdigest()}
    assert inputs[0]['binding'] == 'exact_candidate_id_and_image_sha256' and inputs[0]['imageSha256'] == image_sha
    assert all(ref['role'] == 'model_artifact' for ref in rep['sourceRefs'] if not ref.get('observationId'))
    document['captureId'] = str(uuid5(NAMESPACE_URL, 'exact-candidate-capture'))
    current = migrate_document(document, base_revision_id=str(uuid5(NAMESPACE_URL, 'exact-candidate-revision')))
    audited = next(row for row in audit_correspondence(current)['rows'] if row['entityId'] == entity['id'])
    assert audited['modelCurrent'] and audited['sourceErrors'] == []
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert import_document(path, put)[1]['documentSha256'] == manifest['documentSha256']
    if not with_report:
        for key, invalid in [('candidate_id', 'unrelated'), ('frame_id', 'unrelated'),
                             ('image_sha256', 'b'*64), ('mask_sha256', 'b'*64)]:
            changed = deepcopy(source)
            changed['objects'][1]['measurements']['source'][key] = invalid
            path.write_text(json.dumps(changed))
            rejected, _ = import_document(path, put)
            model = rejected['entities'][1]
            assert model['observationRefs'] == []
            assert all(not ref.get('observationId') for ref in model['representations'][0]['sourceRefs'])
            assert model['missingEvidence'] == ['source_observation_binding_pending']


def test_floor_role_requires_pinned_native_mesh_membership(tmp_path, monkeypatch):
    path = make_public_scene(tmp_path)
    source = json.loads(path.read_text())
    fid = source['cameras'][0]['id']
    points = np.array([[[0,0,2],[.01,0,2]],[[0,.01,2],[.01,.01,2]]], dtype='<f4')
    faces = np.array([[0,1,2],[1,3,2]], dtype='<u4')
    vertices = np.c_[points.reshape(-1,3),np.tile([0,0,1,.5,.5,.5],(4,1))].astype('<f4')
    raw = vertices.tobytes()+faces.tobytes()
    packed = gzip.compress(raw)
    (tmp_path/'context.bin.gz').write_bytes(packed)
    for record in source['objects']:
        record['mesh'].update(vertex_count=4,index_byte_offset=144,index_count=6)
        record['mesh']['asset'].update(bytes=len(raw),packed_bytes=len(packed),sha256=hashlib.sha256(raw).hexdigest())
    source['objects'][0]['label'] = 'Unrelated display name'
    values = {'pts3d.npy':points,'valid_mask.npy':np.ones((2,2),bool),'content_valid_mask.npy':np.ones((2,2),bool),
        'conf.npy':np.ones((2,2)),'intrinsics.npy':np.array(source['cameras'][0]['K']),
        'camera_to_world.npy':np.eye(4),'mask.npy':np.ones((2,2),bool),'points.npy':points.reshape(-1,3)}
    hashes = {}
    for name,value in values.items():
        destination=tmp_path/'geometry'/'frames'/fid/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        np.save(destination,value,allow_pickle=False)
        hashes[name]=hashlib.sha256(destination.read_bytes()).hexdigest()
    keys = {'pointmap_path':'pts3d.npy','valid_path':'valid_mask.npy','content_valid_path':'content_valid_mask.npy',
        'conf_path':'conf.npy','K_path':'intrinsics.npy','c2w_path':'camera_to_world.npy','canonical_mask_path':'mask.npy','points_path':'points.npy'}
    view={'frame_id':fid,'sha256':hashes,**{key:f'geometry/frames/{fid}/{name}' for key,name in keys.items()}}
    floor_raw=json.dumps({'views':[view],'coordinate_system':'Descriptive source wording'}).encode()
    (tmp_path/'floor.json').write_bytes(floor_raw)
    floor_sha=hashlib.sha256(floor_raw).hexdigest()
    frozen={'experiment':source['run_id'],'evidence':{'floor_sha256':floor_sha},'geometry':{'frames':[{'frame_id':fid,'files':hashes}]}}
    frozen_raw=json.dumps(frozen).encode()
    (tmp_path/'manifest.json').write_bytes(frozen_raw)
    source['provenance']={'observed_ranges':{'source_sha256':{'manifest.json':hashlib.sha256(frozen_raw).hexdigest()}}}
    source['floor_reference']={'path':'floor.json','sha256':floor_sha,'coordinate_system':'Another descriptive wording'}
    path.write_text(json.dumps(source))
    # Geometry-cloud import has its own end-to-end test; isolate floor binding here.
    monkeypatch.setattr('scripts.import_geometry_evidence.import_geometry_evidence',lambda *args:{'frames':[]})
    def put(data,media_type,metadata):
        sha=hashlib.sha256(data).hexdigest()
        return {'id':str(uuid5(NAMESPACE_URL,sha)),'sha256':sha,'sizeBytes':len(data),'mediaType':media_type,'metadata':metadata}
    document,manifest=import_document(path,put,geometry_root=tmp_path)
    entity=next(e for e in document['entities'] if e['id']==manifest['entityIds']['context'])
    assert entity['geometryRole']=='floor' and not entity['observationRefs']
    assert len(entity['geometryRoleSourceRefs'])==11
    assert manifest['floorBinding']['sourceRecordIds']==['context']
    assert manifest['floorBinding']['vertexCount']==4 and manifest['floorBinding']['faceCount']==2
    assert 'geometryRole' not in document['entities'][1], 'Generated geometry never proves floor membership'
    evidence=floor_evidence(tmp_path,source)
    changed=vertices.copy();changed[0,0]=.001
    assert floor_mesh_members(evidence,{'context':source['objects'][0]},{'context':(changed,faces)},'native')[0]==[]
    evidence['arrays']['points_path']=points.reshape(-1,3).copy();evidence['arrays']['points_path'][0,0]=.001
    with pytest.raises(PlatformError,match='import_floor_sample_mismatch'):
        floor_mesh_members(evidence,{}, {},'native')
    source['cameras'][0]['camera_to_world'][0][3]=1
    with pytest.raises(PlatformError,match='import_floor_camera_mismatch'):
        floor_evidence(tmp_path,source)
    (tmp_path/'floor.json').write_bytes(floor_raw+b' ')
    with pytest.raises(PlatformError,match='import_floor_evidence_hash_mismatch'):
        floor_evidence(tmp_path,source)


def native_representation_source_fixture(tmp_path):
    """An unnamed imported mesh and one already-owned exact source observation."""
    from copy import deepcopy
    path = make_public_scene(tmp_path)
    source = json.loads(path.read_bytes())
    points = np.stack([*np.meshgrid(np.arange(8) * .01, np.arange(6) * .01), np.full((6, 8), 2)], -1).astype('<f4')
    grid = np.arange(48).reshape(6, 8)
    faces = np.concatenate([np.stack([grid[:-1,:-1], grid[:-1,1:], grid[1:,:-1]], -1).reshape(-1,3),
                            np.stack([grid[:-1,1:], grid[1:,1:], grid[1:,:-1]], -1).reshape(-1,3)]).astype('<u4')
    vertices = np.c_[points.reshape(-1,3), np.tile([0,0,1,.5,.5,.5], (48,1))].astype('<f4')
    raw = vertices.tobytes() + faces.tobytes()
    packed = gzip.compress(raw); (tmp_path/'context.bin.gz').write_bytes(packed)
    mesh = source['objects'][0]['mesh']
    mesh.update(vertex_count=48, index_byte_offset=vertices.nbytes, index_count=faces.size)
    mesh['asset'].update(bytes=len(raw), packed_bytes=len(packed), sha256=hashlib.sha256(raw).hexdigest())
    source['objects'] = [source['objects'][0]]
    source['objects'][0].update(id='unrelated-import-id', label='No semantic hint', role=None)
    source['unavailable_objects'] = []
    source['floor_plane'] = [0,0,1,-2]
    region = source['observed_regions'][0]
    face_ids = np.arange(len(faces),dtype='<u4').tobytes()
    packed_faces = gzip.compress(face_ids); (tmp_path/'region.faces.gz').write_bytes(packed_faces)
    region['context_id'] = 'unrelated-import-id'
    region['faces'] = {'count':len(faces),'asset':{'path':'region.faces.gz','bytes':len(face_ids),
        'packed_bytes':len(packed_faces),'sha256':hashlib.sha256(face_ids).hexdigest()}}
    region['mask']['bbox_xyxy'] = [0,0,8,6]
    region['source_bbox']['bbox_xyxy'] = [0,0,8,6]
    sam = json.dumps({'rle': [json.dumps({'size':[6,8], 'counts':[0,48]})]}).encode()
    (tmp_path/'one-instance.json').write_bytes(sam)
    region['provenance'] = {'source_image_sha256':hashlib.sha256((tmp_path/'photo.png').read_bytes()).hexdigest(),
        'source_refs':[{'type':'text-sam','path':'one-instance.json','pointer':['rle',0], 'instance':0,
                       'sha256':hashlib.sha256(sam).hexdigest(),'label':'floor'}]}
    path.write_text(json.dumps(source))
    stored = {}
    def put(data, media_type, metadata):
        sha = hashlib.sha256(data).hexdigest(); aid = str(uuid5(NAMESPACE_URL, sha)); stored[aid] = data
        return {'id':aid, 'sha256':sha, 'sizeBytes':len(data), 'mediaType':media_type, 'metadata':metadata}
    document, manifest = import_document(path, put)
    def include(data, media, metadata, key=None):
        asset = put(data, media, metadata)
        if asset['id'] not in {a['id'] for a in document['assets']}:
            document['assets'].append({**asset, **metadata})
        return asset['id']
    refs = {}
    for name, array in {'pts3d.npy':points,'valid_mask.npy':np.ones((6,8),bool),
            'content_valid_mask.npy':np.ones((6,8),bool),'conf.npy':np.ones((6,8)),
            'intrinsics.npy':np.asarray(source['cameras'][0]['K']), 'camera_to_world.npy':np.eye(4)}.items():
        buffer = io.BytesIO(); np.save(buffer,array,allow_pickle=False)
        refs[name] = include(buffer.getvalue(),'application/x-npy',{'kind':'native_geometry_evidence'})
    camera = document['cameras'][0]; refs['input'] = camera['imageId']
    frame = {'sourceFrameId':'camera-a','cameraId':camera['id'],'assets':refs}
    geometry = {'coordinateFrameId':camera['coordinateFrameId'],'frames':[frame]}
    geometry['manifestAssetId'] = include(json.dumps(geometry).encode(),'application/json',{'kind':'native_geometry_manifest'})
    document['geometryEvidence'] = geometry
    document['captureId'] = 'source-capture'
    from ehs_spatial.platform.identity import migrate_document
    document = migrate_document(document, base_revision_id='source-revision')
    # include remains attached to this exact prepared document, as production preparation does.
    return document, source, stored, include, manifest, {'points':points, 'mask':np.ones((6,8),bool)}, deepcopy(document)


def test_exact_imported_mesh_source_equivalence_keeps_one_observation_and_actual_cad(tmp_path):
    from copy import deepcopy
    from ehs_spatial.platform.identity import apply_source_equivalences, resolve_entity_id
    from ehs_spatial.platform.contracts import validate_document
    from ehs_spatial.platform.reconstruction import _verified_source_equivalences, _plan_projection
    from ehs_spatial.platform.blender_export import mesh_from_asset
    from scripts.import_report_evidence import import_source_equivalences
    from types import SimpleNamespace
    document, source, stored, include, manifest, arrays, before = native_representation_source_fixture(tmp_path)
    observation = document['observations'][0]
    current_owner = next(e for e in document['entities'] if observation['id'] in e['observationRefs'])
    current_owner['representations'][0]['sourceValidity'] = 'stale'
    rebuilt = deepcopy(current_owner['representations'][0])
    rebuilt.update(id='current-rebuilt-observation',sourceValidity='current',sourceRefs=[{
        'observationId':observation['id'],'revision':observation['revision'],'imageId':observation['imageId']}])
    current_owner['representations'].append(rebuilt)
    before = deepcopy(document)
    masks = {observation['id']:arrays['mask']}
    source_id = observation['sourceRefs'][0]['assetId']
    result = import_source_equivalences(document, source, source_id, [], masks, include,
        read_asset=stored.__getitem__, read_source=lambda name, sha:(tmp_path/name).read_bytes(), read_import_asset=lambda record:packed_asset(tmp_path,record))
    assert result['representationPairCount'] == 1
    assert document['entities'] == before['entities'] and document['observations'] == before['observations']
    assets = {a['id']:a for a in document['assets']}
    stages = SimpleNamespace(repo=SimpleNamespace(get_asset=lambda aid:{**assets[aid],'projectId':'project','storageKey':aid}),
        blobs=SimpleNamespace(get=lambda key,sha,size:stored[key]), job={'projectId':'project'})
    verified, skipped = _verified_source_equivalences(document,masks,stages)
    assert not skipped and len(verified) == 1
    merged = apply_source_equivalences(document,verified,base_revision_id='source-revision',masks=masks,read_asset=stored.__getitem__)
    owner_id, old_id = manifest['entityIds']['tiny'], manifest['entityIds']['unrelated-import-id']
    assert len(merged) == 1 and resolve_entity_id(document,old_id) == owner_id
    owner = next(e for e in document['entities'] if e['id'] == owner_id)
    assert owner['observationRefs'] == [observation['id']] and owner['activeModelRepresentationId'] is None
    assert next(r for r in owner['representations'] if r['id']==rebuilt['id']) == rebuilt
    assert next(r for r in owner['representations'] if r['id']==verified[0]['sourceRepresentationId'])['sourceValidity']=='stale'
    assert document['observations'] == before['observations'] and document['cameras'] == before['cameras']
    assert document['coordinateFrames'] == before['coordinateFrames']
    assert document['identityDecisions'][-1]['source'] == 'source_binding'
    previous = next(e for e in before['entities'] if e['id']==old_id)['representations'][0]
    rep = next(r for r in owner['representations'] if r['id']==previous['id'])
    assert {k:v for k,v in rep.items() if k not in ('sourceRefs','sourceKind')} == {k:v for k,v in previous.items() if k not in ('sourceRefs','sourceKind')}
    assert rep['sourceRefs'][:len(previous['sourceRefs'])] == previous['sourceRefs']
    assert rep['sourceKind'] == 'observed_reference_surface'
    asset=assets[rep['assetId']]; mesh=mesh_from_asset(stored[asset['id']],{**asset,**asset['metadata']})
    projection=_plan_projection(document,rep,mesh,asset['sha256'])
    assert projection['imageId']==observation['imageId'] and projection['observationId']==observation['id'] and projection['polygons']
    validate_document(document)
    unchanged=deepcopy(document)
    assert apply_source_equivalences(document,verified,base_revision_id='source-revision',masks=masks,read_asset=stored.__getitem__)==[]
    assert document==unchanged
    # The consumed immutable proof remains lineage after a legitimate new mask
    # revision or geometry solution. It cannot reactivate historical geometry.
    observation['revision'] += 1
    current_observation = next(o for o in document['observations'] if o['id']==observation['id'])
    current_observation['revision'] = observation['revision']
    rep['sourceValidity'] = 'stale'
    document['geometryBindings'][observation['imageId']]['geometrySolutionId']='new-geometry-solution'
    validate_document(document)
    verified,skipped = _verified_source_equivalences(document,masks,stages)
    assert verified==[] and skipped[0]['code']=='source_equivalence_already_applied'


@pytest.mark.parametrize('tamper', ['stale_observation','forged_asset','different_mesh','generated_mesh',
    'translated_native_meshes','foreign_image','different_frame','ambiguous_owner','mask','no_proof','source_face_selection'])
def test_imported_representation_proof_rejects_forged_or_changed_sources_atomically(tmp_path,tamper):
    from copy import deepcopy
    from ehs_spatial.platform.identity import apply_source_equivalences
    from scripts.import_report_evidence import import_source_equivalences
    document,source,stored,include,manifest,arrays,_ = native_representation_source_fixture(tmp_path)
    observation=document['observations'][0]; masks={observation['id']:arrays['mask']}
    import_source_equivalences(document,source,observation['sourceRefs'][0]['assetId'],[],masks,include,
        read_asset=stored.__getitem__,read_source=lambda path,sha:(tmp_path/path).read_bytes(),read_import_asset=lambda record:packed_asset(tmp_path,record))
    proof_ref=document['sourceIdentityEvidence'][-1]
    pair=json.loads(stored[proof_ref['assetId']])['pairs'][0]
    pair['evidenceRefs'].append({**proof_ref,'jsonPointer':'/pairs/0'})
    reps={r['id']:r for e in document['entities'] for r in e['representations']}
    if tamper=='stale_observation': observation['revision']+=1
    elif tamper=='forged_asset': pair['representationAsset']['sha256']='0'*64
    elif tamper=='different_mesh':
        aid=pair['sourceRepresentationAsset']['assetId']; data=bytearray(stored[aid]); data[0:4]=np.float32(.02).tobytes()
        newid=include(bytes(data),'application/octet-stream',next(a for a in document['assets'] if a['id']==aid)['metadata'])
        reps[pair['sourceRepresentationId']]['assetId']=newid
        pair['sourceRepresentationAsset']={'assetId':newid,'sha256':hashlib.sha256(data).hexdigest()}
    elif tamper=='generated_mesh':
        for rid in (pair['representationId'],pair['sourceRepresentationId']): reps[rid]['kind']='generated_mesh'
    elif tamper=='translated_native_meshes':
        for rid,key in ((pair['representationId'],'transformSnapshot'),(pair['sourceRepresentationId'],'sourceTransformSnapshot')):
            reps[rid]['transform']['position'][0]=1
            pair[key]=deepcopy(reps[rid]['transform'])
    elif tamper=='foreign_image': pair['imageId']='unrelated-image'
    elif tamper=='different_frame': reps[pair['representationId']]['coordinateFrameId']='unrelated-frame'
    elif tamper=='ambiguous_owner': document['entities'][0]['observationRefs']=[observation['id']]
    elif tamper=='mask': masks[observation['id']]=~arrays['mask']
    elif tamper=='no_proof': pair['evidenceRefs']=[]
    elif tamper=='source_face_selection':
        # Both current mesh assets still agree. Only the immutable source record's
        # face selection differs, so matching current metadata cannot prove lineage.
        old_source=pair['sourceRecordRef']['assetId']
        data=stored[pair['sourceFaceAsset']['assetId']]
        data=np.frombuffer(data,dtype='<u4')[::-1].copy().tobytes()
        aid=include(data,'application/octet-stream',{'kind':'source_identity_geometry'})
        checksum=hashlib.sha256(data).hexdigest()
        pair['sourceFaceAsset']={'assetId':aid,'sha256':checksum}
        source['observed_regions'][0]['faces']['asset']['sha256']=checksum
        data=json.dumps(source).encode(); aid=include(data,'application/json',{'kind':'import_source'})
        def remap(value):
            if isinstance(value,dict):
                for key,child in value.items():
                    if key in ('assetId','sourceAssetId') and child==old_source: value[key]=aid
                    else: remap(child)
            elif isinstance(value,list):
                for child in value: remap(child)
        for key in ('entities','observations','cameras'): remap(document[key])
        for key in ('sourceRecordRef','observationRecordRef'):
            pair[key].update(assetId=aid,sha256=hashlib.sha256(data).hexdigest())
    before=deepcopy(document)
    with pytest.raises(PlatformError):
        apply_source_equivalences(document,[pair],base_revision_id='source-revision',masks=masks,read_asset=stored.__getitem__)
    assert document==before


def test_new_import_packages_same_source_mesh_proof_with_existing_native_readers(tmp_path,monkeypatch):
    from copy import deepcopy
    document,source,stored,_,_,_,_ = native_representation_source_fixture(tmp_path)
    geometry=deepcopy(document['geometryEvidence'])
    assets={a['id']:a for a in document['assets']}
    def import_geometry(root,path,source,document,manifest,include,ident,frame_id):
        ids={geometry['manifestAssetId'],*(aid for frame in geometry['frames'] for aid in frame['assets'].values())}
        for aid in ids:
            asset=assets[aid]
            include(stored[aid],asset['mediaType'],asset.get('metadata',{}))
        return deepcopy(geometry)
    monkeypatch.setattr('scripts.import_geometry_evidence.import_geometry_evidence',import_geometry)
    monkeypatch.setattr('scripts.import_public_scene.import_observation_masks',lambda *args:{})
    monkeypatch.setattr('scripts.import_public_scene.observation_mask_sources',lambda *args:({},[],[]))
    saved={}
    def put(data,media,metadata):
        sha=hashlib.sha256(data).hexdigest(); aid=str(uuid5(NAMESPACE_URL,sha)); saved[aid]=data
        return {'id':aid,'sha256':sha,'sizeBytes':len(data),'mediaType':media,'metadata':metadata}
    imported,manifest=import_document(tmp_path/'scene.json',put,geometry_root=tmp_path)
    assert manifest['sourceIdentity']['representationPairCount']==1
    proof=json.loads(saved[imported['sourceIdentityEvidence'][0]['assetId']])
    assert proof['pairs'][0]['kind']=='same_source_indexed_mesh'
    assert len(imported['observations'])==1 and len(imported['entities'])==2
    assert not next(e for e in imported['entities'] if e['id']==proof['pairs'][0]['entityId'])['observationRefs']


def test_source_equivalence_refreshes_cad_in_normal_reassociation(tmp_path):
    from ehs_spatial.platform import reconstruction
    from ehs_spatial.platform.storage import LocalBlobStore
    from scripts.import_report_evidence import import_source_equivalences
    from scripts.research.reprocess_object_identity import assert_source_conserved
    from test_platform_reconstruction import Repo
    document, source, stored, include, manifest, arrays, before = native_representation_source_fixture(tmp_path)
    observation = document['observations'][0]
    source_id = observation['sourceRefs'][0]['assetId']
    masks = {observation['id']:arrays['mask']}
    import_source_equivalences(document,source,source_id,[],masks,include,read_asset=stored.__getitem__,
        read_source=lambda name,sha:(tmp_path/name).read_bytes(),read_import_asset=lambda record:packed_asset(tmp_path,record))
    document['geometryEvidence']['frames'][0]['assets']['canonical.png'] = observation['imageId']
    blobs = LocalBlobStore(tmp_path/'blobs')
    repo = Repo(blobs)
    repo.assets = [{**a,**blobs.put(stored[a['id']],a['mediaType']),'projectId':repo.pid} for a in document['assets']]
    repo.document = document
    updated,result = reconstruction.run_reassociation(repo,blobs,{**repo.job,'kind':'reassociate_scene','inputs':{}},{})
    assert result['status']=='succeeded' and result['newModelCalls']==0 and not repo.calls
    assert result['association']['sourceEquivalences']['verifiedPairCount']==1
    assert_source_conserved(document,updated)
    owner = next(e for e in updated['entities'] if observation['id'] in e['observationRefs'])
    reference = next(r for r in owner['representations'] if r.get('sourceKind')=='observed_reference_surface')
    projection = reference.get('planProjection')
    assert projection and projection['imageId']==observation['imageId']
    assert projection['observationId']==observation['id'] and projection['observationRevision']==observation['revision']
    assert projection['assetId']==reference['assetId'] and projection['transformSnapshot']==reference['transform']
    assert projection['polygons']
