"""Hosted SAM3D transport and source-coordinate contracts; no paid calls."""
import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
from PIL import Image
import pytest

from ehs_spatial.platform import hosted_sam3d as adapter
from ehs_spatial.platform.contracts import PlatformError


def payload():
    camera = np.eye(4)
    camera[:3, 3] = [4, 5, 6]
    y, x = np.mgrid[:3, :5]
    camera_points = np.stack((x / 5, y / 3, np.full_like(x, 2)), -1)
    points = camera_points + camera[:3, 3]
    return {'image':np.full((3,5,3), 120, np.uint8), 'mask':x > 0, 'points':points,
        'valid':np.ones((3,5),bool), 'K':np.eye(3), 'cameraToWorld':camera,
        'coordinateFrameId':'native', 'imageId':'image', 'imageSha256':'a'*64, 'seed':0,
        '_researchProtocol':{'purpose':'runtime_validation','runtimeManifest':{'generation':{
            'pins':adapter.PINS,'provider':'fal','endpoint':adapter.ENDPOINT,
            'schemaSha256':adapter.SCHEMA_SHA256,
            'adapterSourceSha256':hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest(),
            'falClientVersion':'1.0.0','pointmapConvention':adapter.POINTMAP_CONVENTION}}}}


def test_owned_grid_and_camera_basis_are_preserved_without_resizing():
    source = payload()
    request, evidence = adapter.arguments(source)
    decode = lambda uri: base64.b64decode(uri.split(',',1)[1])
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(decode(request['image_url'])))),source['image'])
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(decode(request['mask_urls'][0])))) > 0,source['mask'])
    assert 'pointmap_url' not in request
    actual = np.load(io.BytesIO(evidence['files']['pointmap']['bytes']), allow_pickle=False)
    assert actual.shape == (3,5,3) and actual.dtype == np.float32
    assert np.allclose(actual,(source['points']-[4,5,6])*[-1,-1,1])
    assert evidence['conventionStatus'] == 'native_support_retained_for_placement'
    assert evidence['pointmapSentToProvider'] is False
    assert request['export_textured_glb'] is False


def test_pose_seed_preserves_source_camera_and_uses_owned_depth():
    from scipy.spatial.transform import Rotation
    from ehs_spatial.platform.spatial import transform_points
    camera = np.eye(4)
    camera[:3,:3] = Rotation.from_euler('xyz',[.2,-.3,.4]).as_matrix()
    camera[:3,3] = [4,-2,1]
    view = {'cameraToWorld':camera.copy(), 'depth':np.full((3,5),6.),
            'mask':np.ones((3,5),bool), 'valid':np.ones((3,5),bool)}
    metadata = {'rotation':[[1,0,0,0]], 'scale':[[2,2,2]], 'translation':[[.5,-.25,3]]}
    actual = adapter.native_pose_hint(metadata,view)
    # Known object basis: local Y becomes provider Z; native source depth is 6.
    expected_camera = np.array([[-1,.5,6],[-1,.5,10],[-1,4.5,6]])
    expected_native = transform_points(expected_camera,camera)
    assert np.allclose(transform_points(np.array([[0,0,0],[0,1,0],[0,0,1]]),actual),expected_native)
    assert np.array_equal(camera,view['cameraToWorld'])
    for bad in ({**metadata,'scale':[[0,1,1]]},{**metadata,'translation':[[0,0,-1]]}):
        with pytest.raises(PlatformError,match='hosted_sam3d_pose_seed_invalid'):
            adapter.native_pose_hint(bad,view)


@pytest.mark.parametrize('lost_status',[False,True])
def test_one_charged_post_persists_receipt_before_poll_and_keeps_unverified_pose(monkeypatch,tmp_path,lost_status):
    import fal_client
    from ehs_spatial.platform.blender_export import prepare_export, write_glb
    from test_platform_spatial import scene_document
    from ehs_spatial.platform.reconstruction import ProviderResponseError
    document = scene_document()
    document['entities'] = document['entities'][1:2]
    write_glb(prepare_export('test',document,lambda _:pytest.fail('primitive has no asset')),tmp_path/'box.glb')
    glb = (tmp_path/'box.glb').read_bytes()
    events = []
    queue = {'request_id':'request-1','status_url':'https://queue.fal.run/status',
             'response_url':'https://queue.fal.run/result'}
    class Client:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def post(self,url,**kwargs):
            events.append('post')
            return httpx.Response(200,json=queue,request=httpx.Request('POST',url))
    def upload(data,media_type,*,file_name):
        assert file_name == 'pointmap.npy'
        assert np.load(io.BytesIO(data),allow_pickle=False).shape == (3,5,3)
        return 'https://v3.fal.media/pointmap.npy'
    monkeypatch.setattr(fal_client,'SyncClient',lambda:SimpleNamespace(_client=Client(),upload=upload))
    monkeypatch.setattr(httpx,'Client',Client)
    monkeypatch.setattr(adapter.time,'sleep',lambda _:None)
    def read(client,url,maximum):
        assert events[:2] == ['post','receipt:request-1']
        events.append(url)
        if lost_status: raise TimeoutError('unavailable')
        if url.endswith('/status'):
            return (202,b'{"status":"IN_PROGRESS"}') if events.count(url) == 1 else (200,b'{"status":"COMPLETED"}')
        if url.endswith('/result'):
            return 200,json.dumps({'metadata':[{'object_index':0}],
                                  'model_glb':{'url':'https://v3.fal.media/mesh.glb'}}).encode()
        return 200,glb
    monkeypatch.setattr(adapter,'_read',read)
    if lost_status:
        with pytest.raises(ProviderResponseError):
            adapter.invoke(payload(),on_dispatched=lambda identity:events.append('receipt:'+identity))
    else:
        result = adapter.invoke(payload(),on_dispatched=lambda identity:events.append('receipt:'+identity))
        assert 'providerError' not in result
        assert len(result['vertices']) == 8 and len(result['faces']) == 12
        assert result['proposedObjectToNative'] is None
        assert result['runtimeEvidence']['nativePoseStatus'] == 'unverified'
        assert result['runtimeEvidence']['modelGlb']['bytes'] == glb
    assert events.count('post') == 1


def test_rehashed_foreign_mask_cannot_replace_owned_source(tmp_path):
    from ehs_spatial.platform.storage import LocalBlobStore
    from ehs_spatial.platform.contracts import digest
    from ehs_spatial.platform.reconstruction import _packed, _validate_research_inputs
    from test_platform_reconstruction import Repo
    from test_sam3d_runtime import owned_research_input
    from test_sam3d_preflight import research_configuration
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    _, manifest, protocol = research_configuration()
    source, images = owned_research_input(repo,blobs,protocol)
    source['mask'] = ~source['mask']
    protocol['payloadSha256'] = digest(_packed(source))
    job = {**repo.job,'kind':'validate_model','config':{'researchProtocolSha256':digest(protocol)}}
    with pytest.raises(PlatformError,match='research_input_hash_mismatch'):
        _validate_research_inputs(job,'generation',source,images,manifest,protocol,repo.document,
                                 repository=repo,blobs=blobs)
