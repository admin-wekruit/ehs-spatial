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
    monkeypatch.setattr('scripts.import_geometry_evidence.import_geometry_evidence',lambda *args:{})
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
