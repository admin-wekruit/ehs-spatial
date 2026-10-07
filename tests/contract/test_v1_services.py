"""docs/BACKENDS-v1.md against the fake backends (PANOPTES_FAKE_MODEL=1, conftest.py): auth, /healthz, /v1/info, the sam3d
sync call and job lifecycle, idempotency, busy, errors, and the geometry-mvs tar layout."""
import base64
import io
import tarfile
import threading
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

import inputs

KEY = {'X-API-Key': 'contract-test-key'}
INFO_KEYS = {'model_id', 'model_revision', 'code_revision', 'weights_sha256', 'licence', 'code_sha', 'seed_policy'}
HEALTH_KEYS = {'ok', 'model', 'revision', 'gpu', 'vram_free_mb', 'queue_depth', 'running'}
JOB_KEYS = {'status', 'result', 'error', 'seconds', 'model_info', 'gpu'}


@pytest.fixture(scope='module')
def sam3d():
    import sam3d_service
    return sam3d_service


@pytest.fixture(scope='module')
def geometry():
    import geometry_mvs_service
    return geometry_mvs_service


@pytest.fixture
def client(sam3d):
    return TestClient(sam3d.app, raise_server_exceptions=False)


@pytest.fixture
def gclient(geometry):
    return TestClient(geometry.app, raise_server_exceptions=False)


def poll(client, job_id, timeout=30) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f'/v1/jobs/{job_id}', headers=KEY); assert r.status_code == 200, r.text
        if r.json()['status'] in ('done', 'error'):
            return r.json()
        time.sleep(.05)
    raise AssertionError('job did not finish')


# ---------------------------------------------------------------- common
@pytest.mark.parametrize('path', ['/healthz', '/v1/info', '/v1/jobs/x'])
def test_auth_401(client, gclient, path):
    for c in (client, gclient):
        for headers in ({}, {'X-API-Key': 'wrong'}):
            r = c.get(path, headers=headers)
            assert r.status_code == 401 and r.json() == {'error': {'code': 'unauthorized', 'message': r.json()['error']['message']}}


def test_healthz_shape(client, gclient):
    for c, model in ((client, 'sam3d'), (gclient, 'geometry-mvs')):
        r = c.get('/healthz', headers=KEY); body = r.json()
        assert r.status_code == 200 and set(body) == HEALTH_KEYS, body
        assert body['ok'] is True and body['model'] == model and isinstance(body['gpu'], str)
        assert body['queue_depth'] == 0 and body['running'] == 0 and isinstance(body['revision'], str)


def test_info_shape(client, gclient):
    for c in (client, gclient):
        body = c.get('/v1/info', headers=KEY).json()
        assert INFO_KEYS <= set(body), body
        assert isinstance(body['weights_sha256'], dict) and all(len(v) == 64 for v in body['weights_sha256'].values())
        assert body['code_sha'] and body['licence'] and body['model_id']
    sam = client.get('/v1/info', headers=KEY).json()
    assert sam['model_id'] == 'facebook/sam-3d-objects' and 'ss_generator.ckpt' in sam['weights_sha256']
    geo = gclient.get('/v1/info', headers=KEY).json()
    assert set(geo['components']) == {'da3-base', 'roma_outdoor', 'roma_dinov2', 'moge3'}


def test_unknown_job_404(client):
    r = client.get('/v1/jobs/does-not-exist', headers=KEY)
    assert r.status_code == 404 and r.json()['error']['code'] == 'not_found'


def test_bad_input_400(client, gclient):
    r = client.post('/v1/sam3d', json={'input_sha256': 'nope', 'seed': 1}, headers=KEY)
    assert r.status_code == 400 and r.json()['error']['code'] == 'bad_input', r.text
    body = inputs.sam3d_body(size=16); body.pop('mask_b64')
    r = client.post('/v1/sam3d', json=body, headers=KEY)
    assert r.status_code == 400 and 'mask' in r.json()['error']['message']
    body = inputs.sam3d_body(size=16, salt='shape'); body['mask_b64'] = inputs.b64(inputs.png(np.zeros((8, 8), np.uint8)))
    r = client.post('/v1/sam3d', json=body, headers=KEY)
    assert r.status_code == 400 and 'differ' in r.json()['error']['message']
    r = gclient.post('/v1/geometry-mvs/jobs', json={'input_sha256': 'a' * 64, 'cell': '090', 'frames': []}, headers=KEY)
    assert r.status_code == 400 and r.json()['error']['code'] == 'bad_input'
    body = inputs.geometry_body(size=16, x0=2, width=8, salt='opt'); body['options'] = {'start': 'pi3x'}
    r = gclient.post('/v1/geometry-mvs/jobs', json=body, headers=KEY)
    assert r.status_code == 400 and 'options' in r.json()['error']['message']


# ---------------------------------------------------------------- sam3d
def check_sam3d_result(result: dict, body: dict):
    assert {'mesh_npz_b64', 'pins', 'vertices', 'faces', 'seconds', 'gpu', 'model_info'} <= set(result), set(result)
    with np.load(io.BytesIO(base64.b64decode(result['mesh_npz_b64']))) as d:
        assert set(d.files) == {'vertices', 'faces', 'colors', 'object_to_camera_p3d'}
        assert d['vertices'].dtype == np.float32 and d['vertices'].shape == (result['vertices'], 3)
        assert d['faces'].dtype == np.int32 and d['faces'].shape == (result['faces'], 3) and d['faces'].max() < result['vertices']
        assert d['colors'].dtype == np.uint8 and d['colors'].shape == (result['vertices'], 3)
        assert d['object_to_camera_p3d'].dtype == np.float64 and d['object_to_camera_p3d'].shape == (4, 4)
    assert result['pins']['input_sha256'] == body['input_sha256'] and result['pins']['seed'] == body['seed']
    assert INFO_KEYS <= set(result['model_info']) and isinstance(result['seconds'], float) and result['gpu'] == 'fake'


def test_sam3d_sync(client):
    body = inputs.sam3d_body(size=32)
    r = client.post('/v1/sam3d', json=body, headers=KEY)
    assert r.status_code == 200, r.text
    check_sam3d_result(r.json(), body)
    again = client.post('/v1/sam3d', json=body, headers=KEY).json()  # cached: the same finished result
    assert again['mesh_npz_b64'] == r.json()['mesh_npz_b64'] and again['seconds'] == r.json()['seconds']


def test_sam3d_job_lifecycle_and_idempotency(client):
    body = inputs.sam3d_body(size=32, salt='job')
    r = client.post('/v1/sam3d/jobs', json=body, headers=KEY)
    assert r.status_code == 202 and set(r.json()) >= {'job_id', 'input_sha256', 'cached'}, r.text
    assert r.json()['cached'] is False and r.json()['input_sha256'] == body['input_sha256']
    rec = poll(client, r.json()['job_id'])
    assert JOB_KEYS <= set(rec) and rec['status'] == 'done' and rec['error'] is None, rec
    check_sam3d_result(rec['result'], body)
    assert rec['model_info'] == client.get('/v1/info', headers=KEY).json()
    again = client.post('/v1/sam3d/jobs', json=body, headers=KEY)
    assert again.status_code == 202 and again.json()['cached'] is True and again.json()['job_id'] == r.json()['job_id']
    assert again.json()['result'] == rec['result']


def test_busy_429(sam3d, client, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(sam3d.service, 'max_queue', 1)
    monkeypatch.setattr(sam3d, 'fake_run', lambda *a: (gate.wait(10), _cube(*a))[1])
    try:
        first = client.post('/v1/sam3d/jobs', json=inputs.sam3d_body(size=16, salt='busy1'), headers=KEY)
        assert first.status_code == 202
        time.sleep(.2)  # the worker holds job 1 on the gate; job 2 queues; job 3 is busy
        assert client.post('/v1/sam3d/jobs', json=inputs.sam3d_body(size=16, salt='busy2'), headers=KEY).status_code == 202
        r = client.post('/v1/sam3d/jobs', json=inputs.sam3d_body(size=16, salt='busy3'), headers=KEY)
        assert r.status_code == 429 and r.json()['error']['code'] == 'busy', r.text
        assert client.get('/healthz', headers=KEY).json()['queue_depth'] == 1
        assert client.get('/healthz', headers=KEY).json()['running'] == 1
    finally:
        gate.set()
    assert poll(client, first.json()['job_id'])['status'] == 'done'


def _cube(rgb, mask, pointmap, seed, sha):
    return dict(vertices=np.zeros((8, 3), np.float32), faces=np.zeros((12, 3), np.uint32), colors=np.zeros((8, 3), np.uint8),
                objectToCamera=np.eye(4), pins={'model': 'fake', 'input_sha256': sha, 'seed': seed}, gpu='fake', seconds=0.)


def test_model_error_not_cached(sam3d, client, monkeypatch):
    monkeypatch.setattr(sam3d, 'fake_run', lambda *a, **k: {'error': 'Traceback: boom'})
    body = inputs.sam3d_body(size=16, salt='err')
    r = client.post('/v1/sam3d', json=body, headers=KEY)
    assert r.status_code == 500 and r.json()['error'] == {'code': 'model_error', 'message': 'Traceback: boom'}
    j = client.post('/v1/sam3d/jobs', json=body, headers=KEY)
    assert j.json()['cached'] is False  # errors are not cached: the retry runs again
    rec = poll(client, j.json()['job_id'])
    assert rec['status'] == 'error' and rec['result'] is None and rec['error']['code'] == 'model_error'


# ---------------------------------------------------------------- geometry-mvs
def test_geometry_job_tar_layout(gclient):
    body = inputs.geometry_body(cell='090', n=3, size=24, x0=4, width=16)
    r = gclient.post('/v1/geometry-mvs/jobs', json=body, headers=KEY)
    assert r.status_code == 202 and r.json()['cached'] is False, r.text
    rec = poll(gclient, r.json()['job_id'])
    assert rec['status'] == 'done', rec['error']
    result = rec['result']
    assert 'result_b64' in result and 'result_uri' not in result and result['cell'] == '090'
    assert result['content_rect_xyxy'] == [4, 0, 20, 24]
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(result['result_b64'])), mode='r:gz') as tar:
        names = sorted(m.name for m in tar.getmembers() if m.isfile())
        arrays = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile() and m.name.endswith('.npz')}
    frames = ['frame_0001', 'frame_0002', 'frame_0003']
    gpu = [f'checks/clean-gpu/090/{k}-{i}-{j}.npz' for i in range(3) for j in range(i + 1, 3) for k in ('dense', 'roma')]
    gpu += [f'checks/clean-gpu/090/moge-{f}.npz' for f in frames]
    geom = []
    for root in ('checks/da3fair-geom/090-da3-base-padded/geometry', 'checks/clean-geom/090-da3-base-ba-f/geometry'):
        geom.append(f'{root}/candidate_manifest.json')
        geom += [f'{root}/frames/{f}/{n}' for f in frames for n in ('camera_to_world.npy', 'canonical.png', 'conf.npy', 'intrinsics.npy', 'pts3d.npy', 'valid_mask.npy')]
    assert names == sorted(gpu + geom), names
    assert names == result['files']
    with np.load(io.BytesIO(arrays['checks/clean-gpu/090/roma-0-1.npz'])) as d:
        assert set(d.files) == {'uvA', 'uvB', 'certainty'}
    with np.load(io.BytesIO(arrays['checks/clean-gpu/090/dense-0-2.npz'])) as d:
        assert set(d.files) == {'uvAB', 'certA', 'uvBA', 'certB'} and d['certA'].dtype == np.float16
    with np.load(io.BytesIO(arrays['checks/clean-gpu/090/moge-frame_0002.npz'])) as d:
        assert set(d.files) == {'points', 'mask', 'intrinsicsNormalised'}
    again = gclient.post('/v1/geometry-mvs/jobs', json=body, headers=KEY).json()
    assert again['cached'] is True and again['job_id'] == r.json()['job_id']


def test_geometry_uri_inputs(geometry, gclient, monkeypatch):
    body = inputs.geometry_body(cell='030', n=2, size=16, x0=2, width=8, salt='uri')
    store = {}
    for f in body['frames']:
        store[f's3://bucket/{f["frame_id"]}.png'] = base64.b64decode(f.pop('canonical_png_b64')); f['canonical_png_uri'] = f's3://bucket/{f["frame_id"]}.png'
        store[f's3://bucket/{f["frame_id"]}.npy'] = base64.b64decode(f.pop('alpha_npy_b64')); f['alpha_npy_uri'] = f's3://bucket/{f["frame_id"]}.npy'
    monkeypatch.setattr(geometry.v1, 'read_uri', store.__getitem__)
    r = gclient.post('/v1/geometry-mvs/jobs', json=body, headers=KEY)
    assert r.status_code == 202, r.text
    rec = poll(gclient, r.json()['job_id'])
    assert rec['status'] == 'done' and rec['result']['frames'] == ['frame_0001', 'frame_0002'], rec['error']


def test_result_upload_to_bucket(geometry, gclient, monkeypatch):
    uploaded = {}

    class S3:
        def put_object(self, Bucket, Key, Body):
            uploaded[(Bucket, Key)] = len(Body)
    monkeypatch.setenv('PANOPTES_RESULT_BUCKET', 'results/prefix')
    monkeypatch.setattr(geometry.v1, '_s3', lambda: S3())
    body = inputs.geometry_body(cell='030', n=2, size=16, x0=2, width=8, salt='bucket')
    rec = poll(gclient, gclient.post('/v1/geometry-mvs/jobs', json=body, headers=KEY).json()['job_id'])
    assert rec['status'] == 'done', rec['error']
    key = f"prefix/geometry-mvs/{body['input_sha256']}.tar.gz"
    assert rec['result']['result_uri'] == f's3://results/{key}' and 'result_b64' not in rec['result']
    assert list(uploaded) == [('results', key)] and uploaded[('results', key)] > 0
