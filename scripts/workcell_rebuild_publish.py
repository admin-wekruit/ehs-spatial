"""Publish a rebuilt report (scripts/workcell_rebuild_report.py compose -> RUN/public) as a NEW publication, or dry-run it.

Run from the platform checkout with its venv (its ehs_spatial and scripts packages):
  cd panoptes-platform && .venv/bin/python WORKTREE/scripts/workcell_rebuild_publish.py trial --config C --run RUN --scale S --out DIR
  ... publish --config C --run RUN --scale S --title TITLE --out DIR        (the September platform database running)

Edit operations (one batch on the imported revision, both modes): migrateScene v2; setCalibration from the e-stop scale file
(re-checked by onprem_publish.checked_scale: it must have passed and recompute from its features); setPrimitive: the e-stop as
its specification cylinder at the triangulated position (axis = floor normal); setPartRelation for every configured part
(evidence: the part's and its parent's observations); the floor's drawn reference surface: the scene's 'floor_surface' record
becomes an observed_reference_surface of a floor entity that owns copies of the record's photo observations (the viewer draws a
floor entity's reference surface; a model on the floor would count as an object in every check); the record is set out of the
workcell scope (hidden context).
trial    import_document in memory with content-addressed stand-in asset ids (as onprem_publish --dry-run), the same operations
         (apply_operations) -> DIR/view.json (a publication view with that document) and DIR/served/ (every model, surface,
         point-cloud and geometry asset, as workcell_layer_trial.py --served-dir serves them: api/api/assets/<id> ->
         api/blobs/<sha256>). Nothing is stored, nothing published.
publish  run_import (a new project; the importer keeps the management capability in .platform/imports, read here internally and
         never printed) -> commit_edits -> create_publication -> export (the platform app in process, one request at a time) into
         .platform/publication-catalog/<id> -> DIR/result.json. Re-runs are idempotent.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import NAMESPACE_URL, uuid5

import numpy as np

PLATFORM = Path.cwd()
sys.path.insert(0, str(PLATFORM))


def operations(document, manifest, cfg, scale, base_revision_id):
    from scripts.onprem_publish import checked_scale
    value = checked_scale(scale)
    eid = manifest['entityIds']
    frame = document['coordinateFrames'][0]['id']
    ref = {'kind': 'reference_object', 'object': cfg['estop'].get('object', 'emergency stop (photos 1-3)'),
           'specification': cfg['estop']['specification'], 'imageIds': [c['imageId'] for c in document['cameras']],
           **{k: scale[k] for k in ('method', 'edgeRule', 'limit', 'maxDeviation', 'passed', 'features', 'floorRule', 'P3',
                                     'redOnly', 'yellowOnly', 'uncertaintyRelative') if k in scale}}
    ops = [{'type': 'migrateScene', 'schemaVersion': 2},
           {'type': 'setCalibration', 'coordinateFrameId': frame, 'scale': {'status': 'operator_anchored', 'nativeToMeters': value, 'sourceRefs': [ref]}}]
    spec = cfg['estop']['specification']
    up = np.asarray(document['coordinateFrames'][0]['ground']['normal'], float)
    cams = np.array([c['cameraToWorld'] for c in document['cameras']], float)[:, :3, 3]
    up *= np.sign(np.median(cams @ up + document['coordinateFrames'][0]['ground']['offset']))
    from scipy.spatial.transform import Rotation
    x = np.cross(up, [1., 0, 0]); x /= np.linalg.norm(x); R = np.column_stack([x, np.cross(up, x), up])
    estop = next(e['id'] for e in cfg['entities'] if e['method'] == 'estop')
    ops.append({'type': 'setPrimitive', 'entityId': eid[estop],
                'primitive': {'kind': 'cylinder', 'radius': spec['yellowBodyMaxDiameterM'] / 2 / value, 'height': spec['heightM'] / value, 'segments': 64},
                'transform': {'coordinateFrameId': frame, 'position': [float(v) for v in scale['P3']],
                              'quaternion': Rotation.from_matrix(R).as_quat().tolist(), 'scale': [1., 1., 1.]}})
    entities = {e['id']: e for e in document['entities']}
    obs = {o['id']: o for o in document['observations']}
    for ent in cfg['entities']:
        if ent.get('parent') and ent['id'] in eid and ent['parent'] in eid:
            child, parent = entities[eid[ent['id']]], entities[eid[ent['parent']]]
            if not child.get('representations') or not parent.get('representations'):
                continue
            refs = [{'observationId': o, 'revision': obs[o]['revision']} for o in child['observationRefs'] + parent['observationRefs']]
            ops.append({'type': 'setPartRelation', 'entityId': child['id'], 'parentEntityId': parent['id'], 'evidenceRefs': refs,
                        'reason': f"rebuilt report: '{child['label']}' is a part of '{parent['label']}' as in publication {cfg['publicationId']}"})
    if 'floor_surface' in eid and 'observed_floor' in eid:
        surface, bound = entities[eid['floor_surface']], entities[eid['observed_floor']]
        rep = next(r for r in surface['representations'] if r['kind'] == 'observed_surface')
        new_id = str(uuid5(NAMESPACE_URL, cfg['experiment'] + ':floor-reference-entity'))
        copies = []
        for o in (obs[x] for x in surface['observationRefs']):  # the floor masks of every floor photo (build_capture_report)
            c = json.loads(json.dumps(o)); c['id'] = str(uuid5(NAMESPACE_URL, cfg['experiment'] + ':floor-reference:' + o['id'])); c['revision'] = 1
            c['captureId'] = document['captureId']  # migrateScene stamps the imported ones; these are added after it
            c['labelEvidence'] = [{'label': '工位地面（参考面）', 'source': 'rebuild_floor_reference', 'geometryRole': 'floor',
                                   'sourceRefs': [{'observationId': o['id'], 'revision': o['revision']}]}]
            copies.append(c)
        refs = [{'observationId': c['id'], 'revision': 1, 'imageId': c['imageId']} for c in copies]
        entity = {'id': new_id, 'label': '工位地面（参考面，拟合平面）', 'observationRefs': [], 'associationState': 'confirmed',
                  'representations': [dict(rep, id=str(uuid5(NAMESPACE_URL, cfg['experiment'] + ':floor-reference-representation')), sourceKind='observed_reference_surface',
                                           coverage='observed_visible_surface_only', placementState='confirmed', placementReason='imported_observed_surface',
                                           shapeStatus='source_derived_reference_surface', sourceRefs=refs + rep.get('sourceRefs', []))],
                  'currentModelTransform': None, 'measurements': {}, 'measurementEvidence': [], 'measurementSelections': {}, 'activeModelRepresentationId': None,
                  'groupId': None, 'visible': True, 'sourceContext': False, 'geometryRole': 'floor',
                  'geometryRoleSourceRefs': [{'entityId': bound['id'], 'observationIds': bound['observationRefs'], 'binding': 'the bound observed floor entity'}],
                  'lineage': [{'operation': 'rebuild_floor_reference', 'sourceEntityId': surface['id'], 'sourceRepresentationId': rep['id'],
                               'meaning': 'floor-mask rays on the report floor plane (max-inlier fit of the new point maps)'}]}
        ops += [{'type': 'setWorkcellScope', 'entityId': surface['id'], 'included': False,  # its identity decision stays; hidden context
                 'reason': 'carrier of the floor reference-surface mesh; drawn as the floor entity\'s reference surface instead'},
                {'type': 'addEntity', 'entity': entity}]
        ops += [{'type': 'addObservation', 'entityId': new_id, 'observation': c} for c in copies]
    return ops


def trial(cfg, run, scale, out):
    from ehs_spatial.platform.repository import apply_operations
    from scripts.import_public_scene import import_document
    out.mkdir(parents=True)
    store = out / 'served/api/blobs'; store.mkdir(parents=True); index = out / 'served/api/api/assets'; index.mkdir(parents=True)

    def put_asset(data, media_type, metadata):
        sha = hashlib.sha256(data).hexdigest(); aid = str(uuid5(NAMESPACE_URL, 'dry-run-asset:' + sha))
        if metadata.get('kind') != 'source_image':  # the photos go to the checks separately
            (store / sha).write_bytes(data); (index / aid).write_text(json.dumps({'url': f'/blobs/{sha}'}))
        return {'id': aid, 'sha256': sha, 'sizeBytes': len(data), 'mediaType': media_type, 'storageKey': 'sha256/' + sha,
                'metadata': metadata}  # as run_import's put_asset: the checks read logicalName etc. from 'metadata'
    document, manifest = import_document(run / 'public/scene.json', put_asset, geometry_root=run)
    document['captureId'] = str(uuid5(NAMESPACE_URL, 'trial-capture:' + manifest['sourceSha256']))  # run_import sets the job's capture
    document, _ = apply_operations(document, operations(document, manifest, cfg, scale, 'trial-import'), base_revision_id='trial-import')
    view = {'publication': {'id': 'trial-' + cfg['experiment'], 'sceneRevisionId': 'trial-revision', 'title': 'trial',
                            'snapshot': {'revision': {'id': 'trial-revision', 'document': document}}}}
    (out / 'view.json').write_text(json.dumps(view, ensure_ascii=False))
    (out / 'manifest.json').write_text(json.dumps({k: manifest[k] for k in ('entityIds', 'observationIds', 'cameraIds', 'floorBinding', 'entityCount', 'observationCount', 'assetCount') if k in manifest}, indent=1, ensure_ascii=False))
    print(json.dumps({'entities': len(document['entities']), 'observations': len(document['observations']), 'assets': len(document['assets']),
                      'servedMB': round(sum(p.stat().st_size for p in store.iterdir()) / 1e6, 1), 'floorBinding': manifest.get('floorBinding', {}).get('status'),
                      'photos': ','.join(f"{c['imageId']}={Path(c['sourceRefs'][0]['sourceCameraId']).name}" for c in document['cameras'])}))


def publish(cfg, run, scale, title, out):
    os.environ.update(json.loads((PLATFORM / '.platform/identity-runtime-env.json').read_text()))
    from ehs_spatial.platform.config import PlatformConfig
    from ehs_spatial.platform.runtime import services
    from scripts.import_public_scene import run_import
    out.mkdir(parents=True, exist_ok=True)
    scene = run / 'public/scene.json'
    repo, blobs = services(PlatformConfig.from_env()); repo.migrate()
    imported = run_import(scene, repo, blobs, PLATFORM / '.platform/imports', title=title, geometry_root=run)
    project, branch, base = imported['projectId'], imported['branchId'], imported['sceneRevisionId']
    cap = json.loads((PLATFORM / '.platform/imports' / (hashlib.sha256(scene.read_bytes()).hexdigest() + '.management.json')).read_text())['capability']
    document = repo.get_revision(base)['document']
    ops = operations(document, imported, cfg, scale, str(base))
    edit = repo.commit_edits(project, cap, {'requestId': str(uuid5(NAMESPACE_URL, cfg['experiment'] + ':edit:' + str(base))), 'branchId': branch,
                                            'baseRevisionId': base, 'operations': ops, 'label': 'v2 scene; e-stop reference scale and model; part relations'})
    revision = edit['revision']
    assert revision['document']['coordinateFrames'][0]['scale']['status'] == 'operator_anchored'
    publication = repo.create_publication(project, cap, {'requestId': str(uuid5(NAMESPACE_URL, cfg['experiment'] + ':publication:' + str(revision['id']))),
                                                         'sceneRevisionId': str(revision['id']), 'title': title})
    cap = None
    # Export in process: the platform's own app (no job executor, no outbox thread) behind a TestClient, read through the unchanged
    # export_platform_publication.export_publication. A uvicorn server of the full runtime precompiles every publication on this
    # machine with several 1 GB database backends at once (swap filled the disk); one request at a time stays small.
    from fastapi.testclient import TestClient
    from ehs_spatial.platform.api import create_app
    from ehs_spatial.platform.policy_repository import PostgresPolicyRepository
    from ehs_spatial.platform.policy_service import PolicyService
    import importlib.util
    import io
    client = TestClient(create_app(repository=repo, blobs=blobs, policy_service=PolicyService(PostgresPolicyRepository(repo), blobs)))
    api = 'http://platform.invalid'; export = PLATFORM / '.platform/publication-catalog' / str(publication['id'])
    spec = importlib.util.spec_from_file_location('export_platform_publication', PLATFORM / 'scripts/export_platform_publication.py')
    exporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(exporter)

    class InProcess:
        def open(self, url, timeout=None):
            response = client.get(url[len(api):])
            response.raise_for_status()
            stream = io.BytesIO(response.content); stream.__enter__ = lambda *a: stream; stream.__exit__ = lambda *a: False
            return stream
    exporter.build_opener = lambda *handlers: InProcess()
    if not export.exists():  # re-runs: the import, edit and publication are idempotent, an existing export is kept
        exporter.export_publication(api, str(publication['id']), export)
    result = {'projectId': str(project), 'importRevisionId': str(base), 'revisionId': str(revision['id']), 'publicationId': str(publication['id']),
              'importPublicationId': str(imported.get('publicationId')), 'entityIds': imported['entityIds'], 'observationIds': imported.get('observationIds'),
              'floorBinding': imported.get('floorBinding', {}).get('status'), 'export': str(export.relative_to(PLATFORM)), 'title': title,
              'operations': [o['type'] for o in ops]}
    (out / 'result.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('entityIds', 'observationIds')}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['trial', 'publish'])
    p.add_argument('--config', required=True); p.add_argument('--run', required=True); p.add_argument('--scale', required=True)
    p.add_argument('--out', required=True); p.add_argument('--title')
    a = p.parse_args()
    cfg, run, scale, out = json.loads(Path(a.config).read_text()), Path(a.run).resolve(), json.loads(Path(a.scale).read_text()), Path(a.out)
    trial(cfg, run, scale, out) if a.mode == 'trial' else publish(cfg, run, scale, a.title, out)
