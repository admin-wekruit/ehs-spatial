"""The standalone report page: one URL per run carrying the full report
plus the review and agent panels, served by the same process as the
workbench."""

import hashlib
import os
from pathlib import Path

import pytest

os.environ["PANOPTES_NO_RESUME"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

BOR1 = Path("runs/user-bor1-02")
needs_bor1 = pytest.mark.skipif(
    not (BOR1 / "inventory" / "inventory.json").exists(), reason="BOR1 run missing"
)


@pytest.fixture(scope="module")
def client():
    from ehs_spatial.serve import create_server

    return TestClient(create_server())


def test_processing_report_rejects_concurrent_writes(client, tmp_path, monkeypatch):
    import json
    from ehs_spatial import serve
    run = tmp_path / 'processing'
    run.mkdir()
    (run / 'workspace.json').write_text(json.dumps({'state': 'analyzing'}))
    monkeypatch.setattr(serve, 'RUNS', tmp_path)
    monkeypatch.setattr(serve, 'agent_turn', lambda *a, **k: pytest.fail('provider must not run'))
    assert client.post('/api/agent', json={'run_id': 'processing', 'message': 'add a button'}).status_code == 409
    assert client.post('/api/review', json={'run_id': 'processing', 'reviewer': 'test', 'decision': 'accept'}).status_code == 409


@pytest.mark.parametrize('changed', [True, False])
def test_agent_returns_persisted_object_identity_after_refresh(client, tmp_path, monkeypatch, changed):
    import json
    from ehs_spatial import serve, report_refresh
    run = tmp_path / 'new-workcell'
    run.mkdir()
    evidence = {'candidates': [
        {'id': 'same-label-wrong-object', 'label': 'switch', 'source_refs': [{'evidence_id': 'refine:other'}]},
        {'id': 'correct-source-object', 'label': 'switch', 'source_refs': [{'evidence_id': 'refine:new'}]},
    ]}
    monkeypatch.setattr(serve, 'RUNS', tmp_path)
    def turn(run_id, message, **kwargs):
        assert kwargs['runs_root'] == tmp_path
        assert kwargs['frame_id'] == 'frame_0002'
        return {'changed': changed, 'applied': True, 'evidence_id': 'refine:new', 'reply': 'Saved'}
    refreshed = []
    def refresh(path):
        assert path == run
        refreshed.append(path)
        (run / 'object-evidence.json').write_text(json.dumps(evidence))
        return {'updated': True}
    monkeypatch.setattr(serve, 'agent_turn', turn)
    monkeypatch.setattr(report_refresh, 'refresh_report_after_correction', refresh)
    response = client.post('/api/agent', json={'run_id': run.name, 'message': 'add this switch',
                           'action': 'add_object', 'frame_id': 'frame_0002', 'label': 'switch'})
    assert response.status_code == 200
    assert response.json()['candidate_ids'] == ['correct-source-object']
    assert refreshed == [run]  # cached evidence also repairs a previous failed refresh


@needs_bor1
def test_report_page_is_one_page_with_hub(client):
    response = client.get("/report/user-bor1-02")
    assert response.status_code == 200
    page = response.text
    # the full 14-section report ...
    assert page.count('data-section="') == 14
    # Review stays here; all Agent entry points use the source-aware workspace.
    assert 'id="hub"' in page and 'id="rv-save"' in page
    assert 'id="ag-workspace" href="/reports/user-bor1-02"' in page
    assert 'id="ag-send"' not in page
    assert "<iframe" not in page.split('data-section="viewer"')[0]


@needs_bor1
def test_report_file_download_has_no_live_panels(client):
    response = client.get("/report/user-bor1-02/file")
    assert response.status_code == 200
    assert 'id="hub"' not in response.text


@needs_bor1
def test_chat_endpoint_returns_list(client):
    response = client.get("/api/chat/user-bor1-02")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_unknown_run_is_404_and_bad_id_is_400(client):
    assert client.get("/report/does-not-exist").status_code == 404
    assert client.get("/report/..%2Fetc").status_code in (400, 404)


@needs_bor1
def test_review_requires_reviewer(client):
    response = client.post(
        "/api/review",
        json={"run_id": "user-bor1-02", "reviewer": "", "decision": "confirmed"},
    )
    assert response.status_code == 400


def test_workbench_still_mounted(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="report-list"' in response.text
    assert "gradio" in client.get("/workbench/").text.lower()


def test_report_serves_the_same_full_viewer_by_url(client, tmp_path, monkeypatch):
    from ehs_spatial import serve

    run = tmp_path / "linked-run"
    run.mkdir()
    viewer = '<!doctype html><p>600000 points; panoptes:ready</p>'
    (run / "viewer.html").write_text(viewer)
    report = run / "report.html"
    report.write_text(
        '<style></style><details class="case"><iframe id="v3d" class="v3d" '
        'title="linked 3D" srcdoc="&lt;p&gt;embedded viewer&lt;/p&gt;"></iframe></details>'
    )
    monkeypatch.setattr(serve, "RUNS", tmp_path)
    monkeypatch.setattr(serve, "_ensure_report", lambda run_id: report)

    page = client.get("/report/linked-run")
    assert page.status_code == 200
    revision = hashlib.sha256(viewer.encode()).hexdigest()[:16]
    assert f'src="/report/linked-run/viewer?v={revision}"' in page.text
    assert "srcdoc=" not in page.text and "embedded viewer" not in page.text
    rendered = client.get("/report/linked-run/viewer")
    assert rendered.status_code == 200 and rendered.text == viewer
    assert rendered.headers["content-type"].startswith("text/html")
    assert rendered.headers["cache-control"] == "no-cache"
    assert "srcdoc=" in client.get("/report/linked-run/file").text

    (run / "viewer.html").write_text(viewer + "<p>new internal model</p>")
    assert f'/viewer?v={revision}"' not in client.get("/report/linked-run").text

    (run / "viewer.html").unlink()
    missing = client.get("/report/linked-run/viewer")
    assert missing.status_code == 404 and "工作台" in missing.text


def test_surface_route_serves_only_run_surface_assets(client, tmp_path, monkeypatch):
    from ehs_spatial import serve

    surface = tmp_path / "surface-run" / "surface"
    surface.mkdir(parents=True)
    (surface / "surface.glb").write_bytes(b"glTF-test")
    (surface / "surface.json").write_text('{"asset":"surface.glb"}')
    (surface / "face-inv.bin").write_bytes(b"\x0e\x00\x00\x00")
    (surface / "private.txt").write_text("not a viewer asset")
    monkeypatch.setattr(serve, "RUNS", tmp_path)
    response = client.get("/report/surface-run/surface/surface.glb")
    assert response.status_code == 200 and response.content == b"glTF-test"
    assert response.headers["content-type"] == "model/gltf-binary"
    assert client.get("/report/surface-run/surface/face-inv.bin").content == b"\x0e\x00\x00\x00"
    assert client.get("/report/surface-run/surface/surface.json").json()["asset"] == "surface.glb"
    assert client.get("/report/surface-run/surface/private.txt").status_code == 404
    assert client.get("/report/absent/surface/surface.glb").status_code == 404
    (surface / "surface.glb").unlink()
    assert client.get("/report/surface-run/surface/surface.glb").status_code == 404


def test_surface_download_embeds_the_exact_glb_and_face_ids(client, tmp_path, monkeypatch):
    import base64
    import hashlib
    import html
    import json
    import re
    from ehs_spatial import serve

    run = tmp_path / "surface-run"
    surface = run / "surface"
    surface.mkdir(parents=True)
    (run / "inventory").mkdir()
    inventory = run / "inventory/inventory.json"
    inventory.write_text('{"objects":[]}')
    (surface / "surface.glb").write_bytes(b"glTF-real-surface")
    (surface / "face-inv.bin").write_bytes(b"\x0e\x00\x00\x00")
    data = {"asset_url":"/report/surface-run/surface/surface.glb",
            "face_map_url":"/report/surface-run/surface/face-inv.bin", "supported_inv":[13],
            "inventory_sha256":hashlib.sha256(inventory.read_bytes()).hexdigest()}
    (surface / "surface.json").write_text(json.dumps(data))
    viewer = '<script id="surface-data" type="application/json">'+json.dumps(data)+'</script>'
    original = '<iframe class="v3d" srcdoc="'+html.escape(viewer, quote=True)+'"></iframe>'
    report = run / "report.html"
    report.write_text(original)
    monkeypatch.setattr(serve, "RUNS", tmp_path)
    monkeypatch.setattr(serve, "_ensure_report", lambda _: report)
    response = client.get("/report/surface-run/file")
    assert response.status_code == 200 and 'attachment' in response.headers['content-disposition']
    embedded = html.unescape(re.search(r'srcdoc="([^"]*)"', response.text)[1])
    packed = json.loads(re.search(r'<script[^>]*>(.*?)</script>', embedded)[1])
    assert base64.b64decode(packed['asset_url'].split(',')[1]) == b"glTF-real-surface"
    assert base64.b64decode(packed['face_map_url'].split(',')[1]) == b"\x0e\x00\x00\x00"
    assert packed['supported_inv'] == [13] and report.read_text() == original
    (surface / 'surface.json').write_text(json.dumps(dict(data, supported_inv=[15])))
    assert client.get("/report/surface-run/file").status_code == 409
    (surface / 'surface.json').write_text(json.dumps(data))
    (surface / 'face-inv.bin').unlink()
    assert client.get("/report/surface-run/file").status_code == 409


def test_stale_surface_cannot_claim_new_inventory_objects(tmp_path):
    import hashlib
    import json
    from fastapi import HTTPException
    from ehs_spatial.serve import _check_surface_inventory

    (tmp_path / 'surface').mkdir()
    (tmp_path / 'inventory').mkdir()
    inventory = tmp_path / 'inventory/inventory.json'
    inventory.write_text('{"objects":[]}')
    manifest = {'inventory_sha256':hashlib.sha256(inventory.read_bytes()).hexdigest(),
                'asset_sha256':'frozen-model','face_map_sha256':'frozen-faces','supported_inv':[13]}
    (tmp_path / 'surface/surface.json').write_text(json.dumps(manifest))
    viewer = tmp_path / 'viewer.html'
    viewer.write_text('<script id="surface-data" type="application/json">'+json.dumps(manifest)+'</script>')
    _check_surface_inventory(tmp_path)
    viewer.write_text('<script id="surface-data" type="application/json">'+json.dumps(dict(manifest, face_map_sha256='old-faces'))+'</script>')
    with pytest.raises(HTTPException) as stale:
        _check_surface_inventory(tmp_path)
    assert stale.value.status_code == 409
    inventory.write_text('{"objects":[{"label":"new object"}]}')
    with pytest.raises(HTTPException) as error:
        _check_surface_inventory(tmp_path)
    assert error.value.status_code == 409
