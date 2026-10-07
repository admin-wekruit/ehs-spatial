"""Cached replay is bound to exact input bytes and model pins, never labels."""
import hashlib
import json
from pathlib import Path

import pytest
from ehs_spatial.object_generation import generate_candidate, validate_cached_generation


def test_cache_rejects_different_candidate_payload_and_changed_mesh(tmp_path):
    payload=b'exact accepted mask, RGB, camera-z and intrinsics'
    sha=lambda data:hashlib.sha256(data).hexdigest()
    (tmp_path/'input.npz').write_bytes(payload)
    (tmp_path/'source.json').write_text('{"frame_id":"frame_0002"}')
    (tmp_path/'object.ply').write_bytes(b'native mesh')
    (tmp_path/'posed.ply').write_bytes(b'posed mesh')
    pins={'code_revision':'code-pin','model_revision':'weights-pin','seed':42}
    record={**pins,'status':'complete','object_id':'object_exact','source_payload_sha256':sha(payload),
            'source_json_sha256':sha((tmp_path/'source.json').read_bytes()),
            'paths':{'mesh':'object.ply','posed_mesh':'posed.ply'},
            'output_sha256':{'object.ply':sha(b'native mesh'),'posed.ply':sha(b'posed mesh')}}
    (tmp_path/'output.json').write_text(json.dumps(record))
    actual,files=validate_cached_generation(tmp_path,'object_exact',payload,pins)
    assert actual==record and set(files)=={'input.npz','source.json','output.json','object.ply','posed.ply'}
    with pytest.raises(ValueError,match='exact candidate'):
        validate_cached_generation(tmp_path,'object_sibling_same_label',payload,pins)
    with pytest.raises(ValueError,match='exact candidate'):
        validate_cached_generation(tmp_path,'object_exact',payload+b'changed camera',pins)
    (tmp_path/'posed.ply').write_bytes(b'changed pose')
    with pytest.raises(ValueError,match='asset changed'):
        validate_cached_generation(tmp_path,'object_exact',payload,pins)


def test_blocked_candidate_never_starts_a_research_command(tmp_path, monkeypatch):
    source=tmp_path/'source';source.mkdir()
    monkeypatch.setattr('ehs_spatial.object_generation.build_object_evidence',lambda p:{'source_sha256':{},'candidates':[
        {'id':'object_without_depth','frame_id':'frame_0001','label':'button',
         'generation':{'status':'blocked','reason':'Only 3 independent depth pixels'}}]})
    monkeypatch.setattr('ehs_spatial.object_generation.subprocess.run',lambda *a,**kw:pytest.fail('No GPU/CPU generation for blocked evidence'))
    result=generate_candidate(source,'object_without_depth',tmp_path/'job',tmp_path/'research',gpu_budget_seconds=630)
    assert result['state']=='blocked' and result['new_gpu_calls']==0
    assert json.loads((tmp_path/'job/status.json').read_text())==result


@pytest.mark.parametrize('case', ['same_input_after_assembly_failure','unrelated_inventory_change','depth_changed','model_changed','matching_mesh_corrupt','matching_input_corrupt'])
def test_explicit_revision_retry_uses_exact_native_cache_before_any_paid_stage(tmp_path,monkeypatch,case):
    """Real API -> worker -> cache validation; research model/renderer calls are injected."""
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ehs_spatial.artifacts import ArtifactStore
    from ehs_spatial.report_workspace import register_workspace
    from ehs_spatial import report_workspace
    from ehs_spatial.report_generation import run_generation
    from ehs_spatial import object_generation
    runs=tmp_path/'runs';source=runs/'photos';source.mkdir(parents=True)
    (source/'scene.json').write_text('new floor or source inventory revision')
    (source/'report.html').write_text('original report must stay byte-identical')
    sha=lambda data:hashlib.sha256(data).hexdigest()
    current={'scene.json':sha((source/'scene.json').read_bytes())}
    registry={'source_sha256':current,'candidates':[{'id':'object_target','frame_id':'frame_0001','label':'button',
                                                  'generation':{'status':'pending_validation'}}]}
    (source/'object-evidence.json').write_text(json.dumps(registry))
    monkeypatch.setattr(object_generation,'build_object_evidence',lambda _:registry)
    research=tmp_path/'research';scripts=research/'scripts/research';scripts.mkdir(parents=True)
    payload=b'changed depth payload' if case=='depth_changed' else b'exact original payload'
    (scripts/'generate_lucida_assets.py').write_text('def payload_for_object(root,obj):\n    return '+repr(payload)+', {"frame_id":"frame_0001"}\n')
    for name in ['export_object_evidence','prepare_product_floor','assemble_lucida_scene','render_lucida_comparisons']:
        (scripts/(name+'.py')).write_text('# injected offline research stage\n')
    (research/'modal_apps').mkdir()
    (research/'modal_apps/lucida_assets.py').write_text("CODE_REV='code-pin'\nMODEL_REV='weights-pin'\n")
    pins={'code_revision':'code-pin','model_revision':'weights-pin','seed':42}
    environment=tmp_path/'environment.json';environment.write_text(json.dumps({'status':'complete',**pins}))
    viewer=tmp_path/'viewer';viewer.mkdir()
    for name in ['pack-model.py','viewer.html','public-assets.js','i18n-catalog.js','i18n.js','site-config.js','site-client.js']:(viewer/name).write_text('offline fixture: '+name)
    def native(directory,model_input):
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'input.npz').write_bytes(model_input);(directory/'source.json').write_text('{"frame_id":"frame_0001"}')
        (directory/'mesh.ply').write_bytes(b'exact native mesh')
        record={**pins,'status':'complete','object_id':'object_target','source_payload_sha256':sha(model_input),
                'source_json_sha256':sha((directory/'source.json').read_bytes()),'paths':{'mesh':'mesh.ply'},
                'output_sha256':{'mesh.ply':sha(b'exact native mesh')}}
        (directory/'output.json').write_text(json.dumps(record))
    job=runs/'.generations/photos/object_target'
    cached=job/'artifact/run/generation/object_target';native(cached,b'exact original payload')
    if case=='matching_mesh_corrupt':(cached/'mesh.ply').write_bytes(b'corrupt mesh')
    if case=='matching_input_corrupt':(cached/'input.npz').write_bytes(b'corrupt input')
    if case=='model_changed':
        record=json.loads((cached/'output.json').read_text());record['model_revision']='previous-model'
        (cached/'output.json').write_text(json.dumps(record))
    previous='failed' if case=='same_input_after_assembly_failure' else 'done'
    (job/'status.json').write_text(json.dumps({'state':previous,'result':{'source_sha256':{'scene.json':'old-revision'},
                                                                        'reason':'assembling failed' if previous=='failed' else None}}))
    paid=[];scheduled=[]
    monkeypatch.setattr(report_workspace.JOBS,'submit',lambda *args:scheduled.append(args))
    for name,value in [('PANOPTES_RESEARCH_ROOT',research),('PANOPTES_RECGEN_ENVIRONMENT',environment),
                       ('PANOPTES_PUBLISHED_ROOT',viewer)]:monkeypatch.setenv(name,str(value))
    def research_step(command,**kwargs):
        script=Path(command[1]).stem;derived=job/'artifact/run'
        if script=='export_object_evidence':
            (derived/'evidence').mkdir(parents=True)
            (derived/'manifest.json').write_text('{}')
            (derived/'evidence/objects.json').write_text(json.dumps({'objects':[{'object_id':'object_target','preflight':{'payload_sha256':sha(payload)}}],'unavailable_objects':[]}))
        elif script=='generate_lucida_assets':
            paid.append(script);native(derived/'generation/object_target',payload)
            (derived/'generation/gpu-budget.json').write_text('{"calls":[{"charged_seconds":42}]}')
        elif script=='assemble_lucida_scene':
            if '--add-context' not in command:
                (derived/'result').mkdir();(derived/'result/scene.json').write_text('{"objects":[{"id":"object_target","source":"generated"}]}')
        elif script=='render_lucida_comparisons':(derived/'result/metrics.html').write_text('offline metrics')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(object_generation.subprocess,'run',research_step)
    api=FastAPI();register_workspace(api,SimpleNamespace(store=ArtifactStore(runs)));client=TestClient(api)
    before={p.name:p.read_bytes() for p in source.iterdir()}
    assert client.post('/api/reports/photos/generations/object_target').json()['state']=='queued'
    assert len(scheduled)==1
    result=run_generation(runs,'photos','object_target')
    assert result['elapsedSeconds'] >= 0
    if case.startswith('matching_'):
        assert result['state']=='failed' and 'changed' in result['result']['reason'] and not paid
    else:
        assert result['state']=='done'
        for name in ['viewer.html','public-assets.js','i18n-catalog.js','i18n.js','site-config.js','site-client.js']:
            copied=job/'artifact/run/result'/name
            assert copied.read_bytes()==(viewer/name).read_bytes()
            assert result['result']['artifacts']['run/result/'+name]==sha(copied.read_bytes())
        assert (job/'artifact/run/result/index.html').read_bytes()==(viewer/'viewer.html').read_bytes()
        assert len(paid)==(1 if case in {'depth_changed','model_changed'} else 0)
        assert result['result']['new_gpu_calls']==len(paid)
        if not paid:assert any(s['stage']=='assembling' for s in result['result']['stages'])
    assert all((source/name).read_bytes()==data for name,data in before.items())
