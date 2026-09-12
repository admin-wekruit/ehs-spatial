import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ehs_spatial.artifacts import ArtifactStore
from ehs_spatial.report_workspace import register_workspace


def test_single_object_generation_is_idempotent_and_run_scoped(tmp_path, monkeypatch):
    from ehs_spatial import report_workspace
    monkeypatch.setenv('PANOPTES_RESEARCH_ROOT', str(tmp_path / 'research'))
    monkeypatch.setenv('PANOPTES_RECGEN_ENVIRONMENT', str(tmp_path / 'environment.json'))
    calls = []
    monkeypatch.setattr(report_workspace.JOBS, 'submit', lambda *args: calls.append(args))
    run = tmp_path / 'photos'
    run.mkdir()
    (run / 'object-evidence.json').write_text(json.dumps({'candidates': [
        {'id':'lamp','generation':{'status':'pending_validation'}},
        {'id':'no-mask','generation':{'status':'blocked','reason':'No original mask'}}]}))
    api = FastAPI()
    register_workspace(api, SimpleNamespace(store=ArtifactStore(tmp_path)))
    client = TestClient(api)
    url = '/api/reports/photos/generations/lamp'
    assert client.post('/api/reports/photos/generations/no-mask').status_code == 422
    assert client.post('/api/reports/photos/generations/missing').status_code == 404
    assert client.post(url).json()['state'] == 'queued'
    assert client.post(url).json()['state'] == 'queued'
    assert len(calls) == 1
    assert client.get(url + '/assets/scene.json').status_code == 404
    assert [r['run_id'] for r in client.get('/api/reports').json()['reports']] == ['photos']
    job = tmp_path / '.generations/photos/lamp'
    (job/'status.json').write_text('{"state":"done"}')
    result = job/'artifact/run/result'
    result.mkdir(parents=True)
    (result/'scene.json').write_text('{"run_id":"lamp"}')
    assert client.get(url + '/assets/scene.json').json()['run_id'] == 'lamp'
    assert client.get(url + '/assets/../../../../object-evidence.json').status_code == 404
    assert client.get(url).json()['viewer_url'].startswith('/published/viewer.html?scene=/api/reports/photos/')
    # A same-mask object can acquire new floor/depth evidence. Preserve a failed
    # attempt, but permit the corrected input to start exactly one new job.
    (job/'status.json').write_text(json.dumps({'state':'failed','source_sha256':{'depth':'old'}}))
    registry = json.loads((run/'object-evidence.json').read_text())
    registry['source_sha256'] = {'depth':'new'}
    (run/'object-evidence.json').write_text(json.dumps(registry))
    assert client.post(url).json()['state'] == 'queued'
    assert client.post(url).json()['state'] == 'queued'
    assert len(calls) == 2
    assert len(list((job.parent/'.attempts/lamp').glob('*/status.json'))) == 1


@pytest.mark.parametrize('changed_input', ['inventory', 'depth'])
def test_stale_result_is_explicit_and_revision_assets_survive_retry(tmp_path, monkeypatch, changed_input):
    from ehs_spatial import report_workspace
    monkeypatch.setenv('PANOPTES_RESEARCH_ROOT', '/research')
    monkeypatch.setenv('PANOPTES_RECGEN_ENVIRONMENT', '/environment.json')
    calls=[]
    monkeypatch.setattr(report_workspace.JOBS, 'submit', lambda *args:calls.append(args))
    run=tmp_path/'photos';run.mkdir()
    source={'inventory':'v1','depth':'v1'}
    registry={'source_sha256':source.copy(),'candidates':[{'id':'lamp','generation':{'status':'pending_validation'}}]}
    (run/'object-evidence.json').write_text(json.dumps(registry))
    job=tmp_path/'.generations/photos/lamp';result=job/'artifact/run/result';result.mkdir(parents=True)
    (result/'scene.json').write_text('{"frame_id":"original-frame"}')
    (result/'model.bin.gz').write_bytes(b'original mesh')
    # Full source photos remain accessible in historical, report-scoped assets.
    photos = {'original.jpg': b'jpeg source', 'original.JPEG': b'jpeg source 2',
              'original.webp': b'webp source'}
    for name, content in photos.items():
        (result/name).write_bytes(content)
    (result/'private.txt').write_text('not a published asset')
    (job/'status.json').write_text(json.dumps({'state':'done','result':{'source_sha256':source,'artifacts':{'scene.json':'sha'}}}))
    api=FastAPI();register_workspace(api,SimpleNamespace(store=ArtifactStore(tmp_path)));client=TestClient(api)
    url='/api/reports/photos/generations/lamp'
    original=client.get(url).json()
    scene_url=parse_qs(urlsplit(original['viewer_url']).query)['scene'][0]
    for name, content in photos.items():
        response = client.get(scene_url.replace('scene.json', name))
        assert response.status_code == 200 and response.content == content
        assert response.headers['content-type'].startswith('image/')
    assert client.get(scene_url.replace('scene.json', 'private.txt')).status_code == 404
    initial=client.get('/api/reports/photos').json()
    assert initial['playgrounds']==[{'id':'lamp','url':original['viewer_url'],
                                    'title':{'zh':'lamp · 生成物体','en':'lamp · Generated object'}}]
    registry['source_sha256'][changed_input]='v2'
    (run/'object-evidence.json').write_text(json.dumps(registry))
    old_status=(job/'status.json').read_bytes()
    stale=client.get(url).json()
    assert stale['state']=='stale' and stale['viewer_url']==original['viewer_url']
    assert stale['source_revision']!=stale['current_source_revision']
    assert 'model input may be unchanged' in stale['error']
    page=client.get('/api/reports/photos').json()
    detail=page['evidence']['candidates'][0]['generation']
    assert detail['status']=='stale' and detail['viewer_url']==stale['viewer_url'] and detail['metrics_url']==stale['metrics_url']
    assert page['playgrounds'][0]['url']==stale['viewer_url'] and '上一版' in page['playgrounds'][0]['title']['zh']
    assert not calls and (job/'status.json').read_bytes()==old_status
    busy=job.parent/'other/status.json';busy.parent.mkdir();busy.write_text('{"state":"queued"}')
    assert client.post(url).status_code==409 and (job/'status.json').read_bytes()==old_status
    busy.unlink()
    assert client.post(url).json()['state']=='queued'
    assert client.post(url).json()['state']=='queued' and len(calls)==1
    assert client.get(scene_url).json()=={'frame_id':'original-frame'}
    assert client.get(scene_url.replace('scene.json','model.bin.gz')).content==b'original mesh'
    for name, content in photos.items():
        assert client.get(scene_url.replace('scene.json', name)).content == content
    assert client.get(scene_url.replace('/assets/scene.json','/assets/../../status.json')).status_code==404
