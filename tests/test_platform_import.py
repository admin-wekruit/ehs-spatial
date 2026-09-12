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
from ehs_spatial.platform.spatial import transform_matrix
from ehs_spatial.platform.repository import apply_operations
from scripts.import_public_scene import import_document, legacy_transform, packed_asset, source_path


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


def test_imported_bounds_measures_exact_photo_binding_and_manual_acceptance(tmp_path):
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
    assert accepted['entities'][1]['representations'][0]['placementState']=='confirmed'
    assert accepted['entities'][1]['representations'][0]['placementSource']['type']=='manual_assertion'
    assert generated['representations'][0]['placementState']=='unconfirmed'
    source['objects'][1]['measurements']['source']['image_sha256']='b'*64
    path.write_text(json.dumps(source))
    rejected,_=import_document(path,put)
    assert rejected['entities'][1]['observationRefs']==[]
    assert rejected['entities'][1]['missingEvidence']==['source_observation_binding_pending']
