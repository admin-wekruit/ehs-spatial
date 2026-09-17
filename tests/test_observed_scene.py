"""Normal analysis publishes source-bound spatial evidence without a model job."""
import hashlib
import json
import gzip
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from ehs_spatial import observed_scene
from ehs_spatial.artifacts import ArtifactStore
from ehs_spatial.report_workspace import register_workspace, write_json
from ehs_spatial.workspace_access import register_access


def fixture_run(tmp_path, monkeypatch):
    run = tmp_path / 'runs/private-example'
    run.mkdir(parents=True)
    source = run / 'source.bin'
    source.write_bytes(b'exact native evidence')
    evidence = {'run_id': run.name, 'frames': [],
        'source_sha256': {'source.bin': hashlib.sha256(source.read_bytes()).hexdigest()},
        'candidates': [{'id': 'tiny', 'frame_id': 'frame_0001', 'label': 'button',
                        'geometry': {'status': 'unmeasured'}, 'generation': {'status': 'blocked'}},
                       {'id': 'claim', 'frame_id': None, 'label': 'light', 'generation': {'status': 'blocked'}}]}
    write_json(run / 'object-evidence.json', evidence)
    research = tmp_path / 'research'
    exporter = research / 'scripts/research/assemble_lucida_scene.py'
    exporter.parent.mkdir(parents=True)
    exporter.write_text('# fixture exporter revision')
    published = tmp_path / 'published'
    published.mkdir()
    (published / 'pack-model.py').write_text('# fixture packer revision')
    monkeypatch.setenv('PANOPTES_RESEARCH_ROOT', str(research))
    monkeypatch.setenv('PANOPTES_PUBLISHED_ROOT', str(published))
    calls = []
    def cpu_export(command, **kwargs):
        calls.append(command)
        assert kwargs['check'] is True
        assert not any('generate_lucida_assets' in arg for arg in command)
        if '--observed-output' in command:
            out = Path(command[command.index('--observed-output') + 1])
            out.mkdir()
            current = json.loads((run / 'object-evidence.json').read_text())
            def asset(name, raw):
                packed = gzip.compress(raw)
                (out / name).write_bytes(packed)
                return {'path': name, 'bytes': len(raw), 'packed_bytes': len(packed),
                        'sha256': hashlib.sha256(raw).hexdigest()}
            vertices = np.zeros((4, 9), dtype='<f4')
            vertices[:, :3] = [[0, 0, 0], [1, 0, 0], [1, .5, 3**.5/2], [0, .5, 3**.5/2]]
            faces = np.array([[0, 1, 2], [0, 2, 3]], dtype='<u4')
            mesh = {'vertex_count': 4, 'index_count': 6, 'stride': 9, 'index_type': 'uint32',
                    'byte_offset': 0, 'index_byte_offset': vertices.nbytes,
                    'asset': asset('mesh.gz', vertices.tobytes() + faces.tobytes())}
            regions = [{'id': 'tiny', 'reference_frame': 'frame_0001', 'context_id': 'context', 'selectable': True,
                        'faces': {'count': 2, 'encoding': 'uint32-triangle-indices',
                                  'asset': asset('region.gz', np.array([0, 1], dtype='<u4').tobytes())}}] if current.get('geometry_available', True) else []
            missing = [{'id': c['id'], 'reference_frame': c.get('frame_id'), 'reason': 'No connected native points'}
                       for c in current['candidates'] if c['id'] not in {r['id'] for r in regions}]
            scene = {'floor_plane': [0, 0, 1, 0], 'floor_reference': {'source_sha256': 'fixture-ground'},
                     'objects': [{'id': 'context', 'source': 'observed', 'mesh': mesh,
                                  'transform': {'position': [0, 0, 0], 'rotation_deg': [0, 0, 0], 'scale': [1, 1, 1]}}] if regions else [],
                     'observed_regions': regions, 'unavailable_regions': missing,
                     'observed_regions_summary': {'registry_candidates': len(current['candidates']), 'observed_regions': len(regions)}}
            write_json(out / 'scene.json', scene)
            (out / 'photo.jpg').write_bytes(b'fixture original image bytes')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(observed_scene.subprocess, 'run', cpu_export)
    return run, evidence, calls


def test_revisions_membership_source_gate_and_empty_scene(tmp_path, monkeypatch):
    run, evidence, calls = fixture_run(tmp_path, monkeypatch)
    first = observed_scene.build_observed_scene(run)
    assert len(calls) == 2 and first['status'] == 'ready'
    assert first['candidates']['tiny']['status'] == 'ready'  # Unmeasured does not mean no observed surface.
    assert first['candidates']['claim']['status'] == 'unavailable'
    analysis = first['candidates']['tiny']['inclination_analysis']
    assert analysis['status'] == 'measured' and len(analysis['surfaces']) == 1
    assert abs(analysis['surfaces'][0]['inclinationDeg'] - 60) < .01
    assert analysis['surfaces'][0]['classification'] == 'direction_unverified'
    assert first['candidates']['claim']['inclination_analysis']['status'] == 'skipped'
    assert observed_scene.build_observed_scene(run) == first and len(calls) == 2
    old_bytes = {p: p.read_bytes() for p in (run / 'observed' / first['revision']).rglob('*') if p.is_file()}
    evidence['geometry_available'] = False
    write_json(run / 'object-evidence.json', evidence)
    summary, candidates = observed_scene.spatial_state(run, evidence)
    assert summary['status'] == 'stale' and candidates['tiny']['viewer_url'] is None
    empty = observed_scene.build_observed_scene(run)
    assert empty['status'] == 'unavailable' and empty['revision'] != first['revision']
    assert len(calls) == 3  # No packing/mesh construction for an empty scene.
    assert all(p.read_bytes() == data for p, data in old_bytes.items())
    assert observed_scene.spatial_state(run, evidence)[0]['viewer_url'] is None
    (run / 'source.bin').write_bytes(b'changed without updating evidence')
    with pytest.raises(ValueError, match='source or artifact changed'):
        observed_scene.build_observed_scene(run)
    assert json.loads((run / 'observed-scene.json').read_text()) == empty


def test_assets_privacy_stale_candidate_and_historical_revision(tmp_path, monkeypatch):
    run, evidence, _ = fixture_run(tmp_path, monkeypatch)
    first = observed_scene.build_observed_scene(run)
    monkeypatch.setenv('PANOPTES_WRITE_TOKEN', 'test-not-a-real-token')
    monkeypatch.setenv('PANOPTES_PUBLIC_RUNS', '')
    api = FastAPI()
    register_access(api)
    register_workspace(api, SimpleNamespace(store=ArtifactStore(run.parent)))
    client = TestClient(api, base_url='https://workspace.example')
    asset = f'/api/reports/{run.name}/observed/revisions/{first["revision"]}/assets/'
    assert client.get(asset + 'scene.json').status_code == 401
    assert client.get(f'/api/reports/{run.name}').status_code == 401
    assert client.post('/api/session', json={'token': 'test-not-a-real-token'}).status_code == 200
    detail = client.get(f'/api/reports/{run.name}').json()
    assert detail['playgrounds'][0]['id'] == 'observed'
    assert detail['spatial']['status'] == 'ready'
    assert detail['evidence']['candidates'][0]['spatial']['viewer_url'].endswith('&object=tiny')
    assert detail['evidence']['candidates'][1]['spatial']['viewer_url'] is None
    assert client.get(asset + 'photo.jpg').content == b'fixture original image bytes'
    assert client.get(asset + 'manifest.json').status_code == 404
    assert client.get(asset + '%2e%2e%2f%2e%2e%2fsource.bin').status_code == 404
    # Even a manifest-listed symlink cannot serve bytes outside the revision.
    directory = run / 'observed' / first['revision']
    (directory / 'escape.json').symlink_to(run / 'object-evidence.json')
    first['artifacts']['escape.json'] = 'not-trusted'
    write_json(directory / 'manifest.json', first)
    assert client.get(asset + 'escape.json').status_code == 404
    evidence['changed'] = True
    write_json(run / 'object-evidence.json', evidence)
    stale = client.get(f'/api/reports/{run.name}').json()
    assert stale['spatial']['status'] == 'stale'
    assert all(c['spatial']['viewer_url'] is None for c in stale['evidence']['candidates'])
    assert client.get(asset + 'scene.json').status_code == 200


def test_normal_analysis_and_correction_call_observed_builder_after_registry(tmp_path, monkeypatch):
    import sys
    from ehs_spatial import app, interactive_report, object_evidence, report_refresh, viewer
    from ehs_spatial.contracts import CaptureRun
    from ehs_spatial.report_workspace import run_upload
    import scripts.scene_inventory as inventory
    monkeypatch.chdir(tmp_path)
    run = tmp_path / 'runs/example'
    (run / 'upload').mkdir(parents=True)
    upload = run / 'upload/01.jpg'
    upload.write_bytes(b'input')
    (run / 'inventory').mkdir()
    write_json(run / 'inventory/inventory.json', {'objects': []})
    events = []
    def registry(path):
        events.append('registry')
        output = path / 'object-evidence.json'
        write_json(output, {'candidates': [{'id': 'small-unmeasured'}]})
        return output
    def observed(path):
        assert events[-1] == 'registry' and (path / 'object-evidence.json').is_file()
        events.append('observed')
        return {'status': 'ready', 'revision': 'synthetic'}
    monkeypatch.setattr(app, '_scripts_on_path', lambda: None)
    monkeypatch.setitem(sys.modules, 'detect_devices', SimpleNamespace(main=lambda _: events.append('detect')))
    monkeypatch.setitem(sys.modules, 'scene_inventory', SimpleNamespace(main=lambda _: events.append('inventory')))
    monkeypatch.setattr(inventory, 'main', lambda _: events.append('inventory') or 0)
    monkeypatch.setattr(object_evidence, 'write_object_evidence', registry)
    monkeypatch.setattr(observed_scene, 'build_observed_scene', observed)
    monkeypatch.setattr(viewer, 'build_viewer_html', lambda _: events.append('viewer'))
    monkeypatch.setattr(interactive_report, 'build_interactive_run_report', lambda _: events.append('report'))
    service = SimpleNamespace(store=ArtifactStore(tmp_path / 'runs'), run_assessment=lambda _: events.append('assessment'))
    run_upload(service, CaptureRun(run_id='example', image_paths=[str(upload)]))
    assert events == ['assessment', 'detect', 'inventory', 'registry', 'observed', 'viewer', 'report']
    assert json.loads((run / 'workspace.json').read_text())['state'] == 'done'
    events.clear()
    result = report_refresh._build_stage('example')
    assert events == ['inventory', 'registry', 'observed', 'viewer', 'report']
    assert result['spatial_revision'] == 'synthetic'


def test_historical_rgb_mapping_requires_full_pixel_replay(tmp_path):
    import base64
    from PIL import Image
    from scripts.restore_verified_image_mapping import recover
    from ehs_spatial.providers.map_anything import _input_mask_mapping
    run = tmp_path / 'historical'
    for directory in ['input', 'geometry/provider', 'geometry/frames/frame_0001']:
        (run / directory).mkdir(parents=True)
    image = Image.fromarray(np.random.default_rng(2).integers(0, 256, (90, 120, 3), dtype=np.uint8))
    image.save(run / 'input/image_01.png')
    canonical = image.resize((40, 30), Image.Resampling.BICUBIC)
    canonical.save(run / 'geometry/frames/frame_0001/canonical.png')
    payload = {'original_image': {'width': 120, 'height': 90}, 'image': {
        'shape': [30, 40, 3], 'dtype': 'uint8', 'data': base64.b64encode(np.asarray(canonical).tobytes()).decode()}}
    provider = run / 'geometry/provider/frame_0001.json'
    write_json(provider, payload)
    original = provider.read_bytes()
    proof = recover(run)
    assert proof['maxChannelErrorUint8'] == 0 and provider.with_suffix('.before-mapping.json').read_bytes() == original
    _, shape, _, transform = _input_mask_mapping(provider, provider.stat().st_mtime_ns, provider.stat().st_size)
    assert shape == (30, 40) and transform['resized_shape_hw'] == [30, 40]
    with pytest.raises(ValueError, match='already exists'):
        recover(run)
    provider.write_bytes(original)
    Image.fromarray(np.roll(np.asarray(image), 10, axis=1)).save(run / 'input/image_01.png')
    with pytest.raises(ValueError, match='RGB replay failed'):
        recover(run)
    assert provider.read_bytes() == original
