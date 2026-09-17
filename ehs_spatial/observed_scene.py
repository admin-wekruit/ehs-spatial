"""Publish native, evidence-bound observation scenes without generating objects."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlencode

from .artifacts import _read_json_dict
from .report_workspace import write_json


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evidence_revision(evidence: dict) -> str:
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _verify_files(root: Path, files: dict) -> None:
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ValueError('Observed-scene source or artifact changed: ' + relative)


def analyze_observed_inclinations(scene: dict, root: Path) -> dict:
    """Process every source-mask surface before publishing; never on report GET."""
    import numpy as np
    from scripts.import_public_scene import packed_asset, unpack_mesh, legacy_transform
    from .platform.spatial import transform_matrix, transform_points
    from .platform.planar_surfaces import VERSION, DEFAULT_CONFIG, extract_planar_surfaces, surface_inclinations

    meshes = {}
    objects = {o['id']: o for o in scene.get('objects', [])}
    plane = np.asarray(scene.get('floor_plane'), dtype=float)
    ground = {'normal': plane[:3].tolist() if plane.shape == (4,) else [],
              'source': scene.get('floor_reference'), 'angularErrorDeg': None}
    analyses = {}
    for region in scene.get('observed_regions', []) + scene.get('unavailable_regions', []):
        faces = region.get('faces')
        result = {'algorithmVersion': VERSION, 'config': dict(DEFAULT_CONFIG), 'surfaces': [],
                  'source': region.get('provenance'), 'frameId': region.get('reference_frame'),
                  'groundReference': ground, 'status': 'skipped', 'reason': 'measurement_no_observed_triangles'}
        analyses[region['id']] = result
        if not faces:
            continue
        try:
            surface_inclinations([], ground)
        except ValueError:
            result['reason'] = 'measurement_ground_missing'
            continue
        context_id = region['context_id']
        if context_id not in meshes:
            obj = objects[context_id]
            vertices, indices = unpack_mesh(root, obj['mesh'])
            pose = legacy_transform(obj['transform'], region['reference_frame'])
            meshes[context_id] = transform_points(vertices[:, :3], transform_matrix(pose)), indices
        vertices, indices = meshes[context_id]
        raw = packed_asset(root, faces['asset'])
        if faces.get('encoding') != 'uint32-triangle-indices' or len(raw) != faces['count'] * 4:
            raise ValueError('Invalid observed triangle membership')
        selected = np.frombuffer(raw, dtype='<u4')
        if not len(selected) or selected.max() >= len(indices) or len(np.unique(selected)) != len(selected):
            raise ValueError('Invalid observed triangle membership')
        diagnostics = {}
        surfaces = surface_inclinations(extract_planar_surfaces(vertices[indices[selected]], DEFAULT_CONFIG,
                                                               diagnostics=diagnostics), ground)
        # Keep exact face support in the immutable derivative, not in every GET's
        # candidate summary. The region asset binds indices to source geometry.
        for surface in surfaces:
            surface['triangleIndices'] = selected[surface['triangleIndices']].tolist()
        result.update(status='measured' if surfaces else 'unsupported',
                      reason=None if surfaces else 'measurement_no_stable_local_plane',
                      surfaces=surfaces, diagnostics=diagnostics, supportAsset=faces['asset'])
    write_json(root / 'inclination-analysis.json', {'algorithmVersion': VERSION, 'entities': analyses})
    return {key: {**row, 'surfaces': [{k: v for k, v in surface.items() if k != 'triangleIndices'}
                                    for surface in row['surfaces']]} for key, row in analyses.items()}


def build_observed_scene(run: Path) -> dict:
    """CPU-only build; publish one complete immutable revision or leave the old one."""
    run = Path(run).resolve()
    evidence = _read_json_dict(run / 'object-evidence.json')
    if not evidence or evidence.get('run_id') != run.name:
        raise ValueError('A complete evidence registry for this run is required')
    research = Path(os.environ['PANOPTES_RESEARCH_ROOT']).resolve()
    published = Path(os.environ['PANOPTES_PUBLISHED_ROOT']).resolve()
    exporter = research / 'scripts/research/assemble_lucida_scene.py'
    packer = published / 'pack-model.py'
    source = evidence['source_sha256']
    code = {'exporter': digest(exporter), 'packer': digest(packer), 'wrapper': digest(Path(__file__))}
    measurements = Path(__file__).with_name('measurements.py')
    if measurements.is_file():
        code['measurements'] = digest(measurements)
    code['planar_surfaces'] = digest(Path(__file__).parent / 'platform/planar_surfaces.py')
    code['mesh_reader'] = digest(Path(__file__).parents[1] / 'scripts/import_public_scene.py')
    evidence_sha = evidence_revision(evidence)
    revision = evidence_revision({'evidence': evidence_sha, 'code': code})[:24]
    parent = run / 'observed'
    destination = parent / revision
    _verify_files(run, source)
    if destination.exists():
        state = _read_json_dict(destination / 'manifest.json')
        if state.get('revision') != revision or state.get('evidence_sha256') != evidence_sha:
            raise ValueError('Existing observed revision does not match its source')
        _verify_files(destination, state['artifacts'])
    else:
        parent.mkdir(exist_ok=True)
        python = os.environ.get('PANOPTES_RESEARCH_PYTHON', sys.executable)
        # ponytail: one synchronous CPU build in the existing serialized report
        # lane; immutable assets keep old revision URLs readable after refresh.
        with tempfile.TemporaryDirectory(prefix='.building-', dir=parent) as temporary:
            output = Path(temporary) / 'result'
            subprocess.run([python, str(exporter), '--source-run', str(run),
                            '--observed-output', str(output), '--ehs-repo', str(Path(__file__).resolve().parents[1])],
                           check=True, capture_output=True, text=True)
            scene = json.loads((output / 'scene.json').read_text())
            if any(o.get('source') == 'generated' for o in scene.get('objects', [])):
                raise ValueError('Observed analysis must not contain generated assets')
            regions = scene.get('observed_regions', [])
            unavailable = scene.get('unavailable_regions', [])
            identities = [o['id'] for o in regions + unavailable]
            expected = [o['id'] for o in evidence['candidates']]
            if len(set(identities)) != len(identities) or set(identities) != set(expected):
                raise ValueError('Observed scene lost or merged evidence candidates')
            if scene.get('metrics_report'):
                raise ValueError('Observed analysis cannot claim generated-model metrics')
            ready = bool(scene.get('objects'))
            if regions and not ready:
                raise ValueError('Selectable observed regions require actual context geometry')
            if ready:
                subprocess.run([python, str(packer), str(output), str(output)],
                               check=True, capture_output=True, text=True)
            inclinations = analyze_observed_inclinations(json.loads((output / 'scene.json').read_text()), output)
            candidates = {}
            for item in regions:
                faces = item.get('faces')
                if not item.get('selectable') or (faces is not None and not faces.get('count', 0)):
                    raise ValueError('Published region has no selectable spatial support')
                if faces is None and (item.get('surface_status') != 'covered_by_generated'
                                      or item.get('provenance', {}).get('supported_points', 0) < 8):
                    raise ValueError('Bounds-only selection lacks the recorded native support')
                candidates[item['id']] = {'status': 'ready', 'reason': item.get('reason'),
                    'frame_id': item.get('reference_frame'), 'support': 'surface' if faces else 'bounds_only'}
            for item in unavailable:
                candidates[item['id']] = {'status': 'unavailable', 'reason': item.get('reason') or 'No supported native geometry',
                    'frame_id': item.get('reference_frame'), 'support': None}
            for candidate_id, item in candidates.items():
                item['inclination_analysis'] = inclinations[candidate_id]
            state = {'version': 1, 'revision': revision, 'evidence_sha256': evidence_sha,
                'source_sha256': source, 'exporter_sha256': code, 'status': 'ready' if ready else 'unavailable',
                'reason': None if ready else scene.get('reason') or 'No valid connected native surface is available',
                'summary': scene.get('observed_regions_summary', {}), 'candidates': candidates,
                'artifacts': {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}}
            write_json(output / 'manifest.json', state)
            _verify_files(run, source)
            if evidence_revision(_read_json_dict(run / 'object-evidence.json')) != evidence_sha:
                raise ValueError('Evidence changed during observed-scene construction')
            output.rename(destination)
    if evidence_revision(_read_json_dict(run / 'object-evidence.json')) != evidence_sha:
        raise ValueError('Evidence changed before observed-scene publication')
    write_json(run / 'observed-scene.json', state)
    return state


def spatial_state(run: Path, evidence: dict) -> tuple[dict, dict]:
    """Read the published membership only; GET never builds or estimates geometry."""
    state = _read_json_dict(run / 'observed-scene.json')
    if not state:
        missing = {'status': 'not_built', 'reason': 'Observed scene has not been built',
                   'revision': None, 'viewer_url': None, 'summary': {}}
        return missing, {c['id']: dict(missing, frame_id=c.get('frame_id'), support=None) for c in evidence.get('candidates', [])}
    revision = state['revision']
    stale = state['evidence_sha256'] != evidence_revision(evidence)
    reason = 'Saved observed scene belongs to an earlier evidence revision' if stale else state.get('reason')
    base = f'/api/reports/{run.name}/observed/revisions/{revision}/assets/scene.json'
    url = '/published/viewer.html?' + urlencode({'scene': base}) if state['status'] == 'ready' else None
    summary = {'status': 'stale' if stale else state['status'], 'reason': reason,
               'revision': revision, 'viewer_url': url, 'summary': state.get('summary', {})}
    candidates = {}
    for candidate in evidence.get('candidates', []):
        item = state['candidates'].get(candidate['id'], {'status': 'unavailable', 'reason': 'Candidate is absent from the published scene',
                                                      'frame_id': candidate.get('frame_id'), 'support': None})
        candidates[candidate['id']] = dict(item, revision=revision,
            status='stale' if stale else item['status'], reason=reason if stale else item.get('reason'),
            viewer_url=(url + '&' + urlencode({'object': candidate['id']})) if url and not stale and item['status'] == 'ready' else None)
    return summary, candidates
