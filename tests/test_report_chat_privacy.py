"""Raw conversations have one authenticated source, including old report downloads."""
import hashlib
import html
import json
import os

import pytest
from fastapi.testclient import TestClient

os.environ['PANOPTES_NO_RESUME']='1'


@pytest.mark.parametrize('with_surface',[False,True])
def test_public_saved_report_and_download_strip_legacy_chat_but_authorized_history_works(tmp_path,monkeypatch,with_surface):
    from ehs_spatial import serve
    run=tmp_path/'published';run.mkdir()
    user='PRIVATE_USER_CANARY_a52d4a'
    answer='PRIVATE_AGENT_CANARY_b014d9'
    (run/'chat.jsonl').write_text(json.dumps({'type':'agent_turn','user':user,'assistant':answer,'intent':'refine'})+'\n')
    source_chat=(run/'chat.jsonl').read_bytes()
    iframe=''
    if with_surface:
        surface=run/'surface';surface.mkdir();(run/'inventory').mkdir()
        inventory=run/'inventory/inventory.json';inventory.write_text('{"objects":[]}')
        (surface/'surface.glb').write_bytes(b'glTF-unchanged-test-surface')
        (surface/'face-inv.bin').write_bytes(b'\0\0\0\0')
        data={'asset_url':'surface.glb','face_map_url':'face-inv.bin','supported_inv':[],
              'inventory_sha256':hashlib.sha256(inventory.read_bytes()).hexdigest()}
        (surface/'surface.json').write_text(json.dumps(data))
        viewer='<script id="surface-data" type="application/json">'+json.dumps(data)+'</script>'
        (run/'viewer.html').write_text(viewer)
        iframe='<iframe class="v3d" srcdoc="'+html.escape(viewer,quote=True)+'"></iframe>'
    original=('<style></style><details class="case">'+iframe+'<h4 data-section="chat">Saved chat</h4>'
              '<div class="policy">'+user+'<div>'+answer+'</div></div>'
              '<h4 data-section="review">Review</h4><p>REVIEW_STAYS</p>'
              '<h4 data-section="appendix">Evidence</h4><p>EVIDENCE_STAYS</p></details>')
    (run/'report.html').write_text(original)
    monkeypatch.setattr(serve,'RUNS',tmp_path)
    monkeypatch.setattr(serve,'_ensure_report',lambda _:run/'report.html')
    monkeypatch.setenv('PANOPTES_WRITE_TOKEN','private-workspace-test-token')
    monkeypatch.setenv('PANOPTES_PUBLIC_RUNS','published')
    client=TestClient(serve.create_server(),base_url='https://workspace.example')
    for url in ['/report/published','/report/published?embedded=1','/report/published/file']:
        response=client.get(url)
        assert response.status_code==200
        assert user not in response.text and answer not in response.text
        assert 'REVIEW_STAYS' in response.text and 'EVIDENCE_STAYS' in response.text
        if url.endswith('/file'):
            assert 'attachment' in response.headers['content-disposition']
            if with_surface:assert 'data:model/gltf-binary;base64,' in html.unescape(response.text)
    assert client.get('/api/chat/published').status_code==401
    assert client.post('/api/session',json={'token':'private-workspace-test-token'}).status_code==200
    private=client.get('/api/chat/published')
    assert private.status_code==200 and user in private.text and answer in private.text
    assert (run/'report.html').read_text()==original and (run/'chat.jsonl').read_bytes()==source_chat


def test_legacy_chat_removal_handles_missing_boundary_and_multiple_cases():
    from ehs_spatial.interactive_report import strip_report_chat
    page=("<h4 class='x' data-section='chat'>chat</h4><div>FIRST_PRIVATE</div>"
          "<h4 data-section='review'>Review</h4>KEEP_REVIEW</details>"
          '<h4 data-section="chat">chat</h4>SECOND_PRIVATE')
    result=strip_report_chat(page)
    assert 'FIRST_PRIVATE' not in result and 'SECOND_PRIVATE' not in result
    assert 'KEEP_REVIEW' in result
    assert strip_report_chat(result)==result
