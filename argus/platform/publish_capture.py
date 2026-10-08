"""Import an assembled capture, apply its reference scale and export its publication.

  python -m argus.platform.publish_capture VARIANT RUN_DIR SCALE_JSON TITLE

Outputs live under PANOPTES_DATA_ROOT/publications. Platform credentials come
from explicit environment variables; management capabilities are never printed.
"""
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
import hashlib
import json
import os
import sys
if not os.environ.get('PANOPTES_DATABASE_URL'):
    raise SystemExit('PANOPTES_DATABASE_URL is not set (export it or put it in .env)')
from argus.platform.config import PlatformConfig
from argus.platform.runtime import services
from argus.platform.import_public_scene import run_import
from argus.platform import export_platform_publication as exporter
from argus import ROOT

VARIANT, RUN, SCALE_PATH, TITLE = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]), sys.argv[4]
SCENE = RUN / 'public/scene.json'
required = json.loads((ROOT / 'argus/pipeline/cells' / f"{VARIANT.split('-', 1)[0]}.json").read_text())['objects']
for name in ('result/comparisons.json', 'evidence/objects.json'):
    present = {row['object_id'] for row in json.loads((RUN / name).read_text())['objects']}
    missing = sorted(set(required) - present)
    if missing:
        raise RuntimeError(f'{name}: missing required objects: {missing}')
for oid in required:
    folder = RUN / 'generation' / oid
    output = folder / 'output.json'
    if not output.is_file():
        raise RuntimeError(f'generation: missing required object: {oid}')
    record = json.loads(output.read_text())
    if (record.get('status') != 'complete' or record.get('object_id') != oid or
            any(not (folder / (record.get('paths', {}).get(key) or '')).is_file() for key in ('mesh', 'posed_mesh'))):
        raise RuntimeError(f'generation: incomplete required object: {oid}')
PUBLICATIONS = Path(os.environ['PANOPTES_DATA_ROOT']).resolve() / 'publications'
HERE = PUBLICATIONS / VARIANT; HERE.mkdir(parents=True, exist_ok=True)
scale_doc = json.loads(SCALE_PATH.read_text())

repo, blobs = services(PlatformConfig.from_env())
repo.migrate()
imported = run_import(SCENE, repo, blobs, PUBLICATIONS / 'imports', geometry_root=RUN)
project, branch, base = imported['projectId'], imported['branchId'], imported['sceneRevisionId']
management = PUBLICATIONS / 'imports' / (hashlib.sha256(SCENE.read_bytes()).hexdigest() + '.management.json')
cap = json.loads(management.read_text())['capability']

document = repo.get_revision(base)['document']
frame = document['coordinateFrames'][0]['id']
image_ids = [c['imageId'] for c in document['cameras']]
scale = {'status': 'operator_anchored', 'nativeToMeters': scale_doc['nativeToMeters'],
         'sourceRefs': [{'kind': 'reference_object', 'object': 'emergency stop (own fit in this geometry, photos 1-3)', 'imageId': image_ids[0],
                         'specification': {'redHeadDiameterM': .04, 'yellowBodyMaxDiameterM': .08, 'heightM': .1, 'source': 'user, 2026-10-04'},
                         'method': scale_doc['method'], 'edgeRule': scale_doc.get('edgeRule'), 'limit': scale_doc['limit'], 'maxDeviation': scale_doc['maxDeviation'],
                         'passed': scale_doc['passed'], 'features': scale_doc['features']}]}
assert scale['status'] == 'operator_anchored' and scale['sourceRefs'][0]['passed'] and scale['sourceRefs'][0]['maxDeviation'] <= scale['sourceRefs'][0]['limit']
operations = [{'type': 'migrateScene', 'schemaVersion': 2}, {'type': 'setCalibration', 'coordinateFrameId': frame, 'scale': scale}]
edit = repo.commit_edits(project, cap, {'requestId': str(uuid5(NAMESPACE_URL, f'swap-20261007:{VARIANT}:edit:' + str(base))), 'branchId': branch,
                                        'baseRevisionId': base, 'operations': operations, 'label': 'v2 scene; e-stop reference scale'})
revision = edit['revision']
assert revision['document'].get('schemaVersion') == 2 and revision['document']['coordinateFrames'][0]['scale']['status'] == 'operator_anchored'
publication = repo.create_publication(project, cap, {'requestId': str(uuid5(NAMESPACE_URL, f'swap-20261007:{VARIANT}:publication:' + str(revision['id']))),
                                                     'sceneRevisionId': str(revision['id']), 'title': TITLE})
cap = None

# Export through the platform API in process, one request at a time.
from fastapi.testclient import TestClient  # noqa: E402
from argus.platform.api import create_app
from argus.platform.policy_repository import PostgresPolicyRepository
from argus.platform.policy_service import PolicyService
import io  # noqa: E402
client = TestClient(create_app(repository=repo, blobs=blobs, policy_service=PolicyService(PostgresPolicyRepository(repo), blobs)))
api = 'http://platform.invalid'
output = PUBLICATIONS / 'catalog' / str(publication['id'])


class InProcess:
    def open(self, url, timeout=None):
        response = client.get(url[len(api):])
        response.raise_for_status()
        stream = io.BytesIO(response.content); stream.__enter__ = lambda *a: stream; stream.__exit__ = lambda *a: False
        return stream


exporter.build_opener = lambda *handlers: InProcess()
if not output.exists():
    exporter.export_publication(api, str(publication['id']), output)
view = client.get(f"/api/publications/{publication['id']}/view"); view.raise_for_status()
(HERE / 'view.json').write_bytes(view.content)
result = {'variant': VARIANT, 'run': str(RUN), 'projectId': str(project), 'importRevisionId': str(base), 'revisionId': str(revision['id']),
          'publicationId': str(publication['id']), 'nativeToMeters': scale['nativeToMeters'], 'entityCount': imported.get('entityCount'),
          'observationCount': imported.get('observationCount'), 'export': str(output), 'view': str(HERE / 'view.json'), 'title': TITLE}
(HERE / 'result.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
print(json.dumps(result, ensure_ascii=False))
