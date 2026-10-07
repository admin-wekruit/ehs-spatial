"""Import a module-swapped 090 capture run (same pipeline as cell 030 / lucida-replica-01: assembled scene -> build_capture_report.py
-> RUN/public) as its own project, migrate it to the v2 scene with the e-stop reference scale, publish that revision in the LOCAL
September database and export the publication (catalog + view.json). Nothing goes to the live service or to Pages.

Operator-run from the platform checkout with the September database (port 55432) running:
  .venv/bin/python .platform/publish-swap-20261007.py VARIANT RUN_DIR SCALE_JSON TITLE
    VARIANT     pi3x-sam3d | mvs-recgen | mvs-sam3d   (output under .platform/swap-20261007/VARIANT/)
    SCALE_JSON  Pi3X world: .platform/estop-scale-20261004/plan.json (its setCalibration is reused, entity ref -> image ref);
                MVS world:  the run's own e-stop fit (nativeToMeters, limit, maxDeviation, passed, features, method, edgeRule)
Same operations as .platform/publish-cell030-20261005.py: migrateScene v2 + setCalibration. The management capability is read by
the importer and never printed.
"""
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path.cwd()))
# Platform configuration: the environment (.env of the handoff: PANOPTES_DATABASE_URL postgres:// or mongodb://, PANOPTES_BLOB_ROOT
# directory or s3://, PANOPTES_PAID_BUDGET_USD, PANOPTES_BLENDER_EXECUTABLE) wins; the original machine's .platform/identity-runtime-env.json
# fills in whatever the environment does not set.
_env_file = Path('.platform/identity-runtime-env.json')
if _env_file.exists():
    for _k, _v in json.loads(_env_file.read_text()).items():
        os.environ.setdefault(_k, _v)
if not os.environ.get('PANOPTES_DATABASE_URL'):
    raise SystemExit('PANOPTES_DATABASE_URL is not set (export it or put it in .env)')
from ehs_spatial.platform.config import PlatformConfig  # noqa: E402
from ehs_spatial.platform.runtime import services  # noqa: E402
from scripts.import_public_scene import run_import  # noqa: E402

VARIANT, RUN, SCALE_PATH, TITLE = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]), sys.argv[4]
SCENE = RUN / 'public/scene.json'
HERE = Path('.platform/swap-20261007') / VARIANT; HERE.mkdir(parents=True, exist_ok=True)
scale_doc = json.loads(SCALE_PATH.read_text())

repo, blobs = services(PlatformConfig.from_env())
repo.migrate()
imported = run_import(SCENE, repo, blobs, Path('.platform/imports'), geometry_root=RUN)
project, branch, base = imported['projectId'], imported['branchId'], imported['sceneRevisionId']
management = Path('.platform/imports') / (hashlib.sha256(SCENE.read_bytes()).hexdigest() + '.management.json')
cap = json.loads(management.read_text())['capability']

document = repo.get_revision(base)['document']
frame = document['coordinateFrames'][0]['id']
image_ids = [c['imageId'] for c in document['cameras']]
if 'operations' in scale_doc:  # the published Pi3X calibration (same world, same photos, same e-stop fit)
    cal = next(op for op in scale_doc['operations'] if op['type'] == 'setCalibration')
    scale = json.loads(json.dumps(cal['scale']))
    for ref in scale['sourceRefs']:
        ref.pop('entityId', None); ref['imageId'] = image_ids[0]; ref.setdefault('object', 'emergency stop (September fit, photos 1-3)')
else:  # the run's own e-stop fit (MVS world)
    s = scale_doc
    scale = {'status': 'operator_anchored', 'nativeToMeters': s['nativeToMeters'],
             'sourceRefs': [{'kind': 'reference_object', 'object': 'emergency stop (own fit in this geometry, photos 1-3)', 'imageId': image_ids[0],
                             'specification': {'redHeadDiameterM': .04, 'yellowBodyMaxDiameterM': .08, 'heightM': .1, 'source': 'user, 2026-10-04'},
                             'method': s['method'], 'edgeRule': s.get('edgeRule'), 'limit': s['limit'], 'maxDeviation': s['maxDeviation'],
                             'passed': s['passed'], 'features': s['features']}]}
assert scale['status'] == 'operator_anchored' and scale['sourceRefs'][0]['passed'] and scale['sourceRefs'][0]['maxDeviation'] <= scale['sourceRefs'][0]['limit']
operations = [{'type': 'migrateScene', 'schemaVersion': 2}, {'type': 'setCalibration', 'coordinateFrameId': frame, 'scale': scale}]
edit = repo.commit_edits(project, cap, {'requestId': str(uuid5(NAMESPACE_URL, f'swap-20261007:{VARIANT}:edit:' + str(base))), 'branchId': branch,
                                        'baseRevisionId': base, 'operations': operations, 'label': 'v2 scene; e-stop reference scale'})
revision = edit['revision']
assert revision['document'].get('schemaVersion') == 2 and revision['document']['coordinateFrames'][0]['scale']['status'] == 'operator_anchored'
publication = repo.create_publication(project, cap, {'requestId': str(uuid5(NAMESPACE_URL, f'swap-20261007:{VARIANT}:publication:' + str(revision['id']))),
                                                     'sceneRevisionId': str(revision['id']), 'title': TITLE})
cap = None

# Export in process (the platform's own app behind a TestClient, one request at a time, read through the unchanged
# export_platform_publication.export_publication): a uvicorn server of the full runtime precompiles every publication on this
# machine with several 1 GB database backends at once and does not answer /api/publications (scripts/workcell_rebuild_publish.py).
from fastapi.testclient import TestClient  # noqa: E402
from ehs_spatial.platform.api import create_app  # noqa: E402
from ehs_spatial.platform.policy_repository import PostgresPolicyRepository  # noqa: E402
from ehs_spatial.platform.policy_service import PolicyService  # noqa: E402
import io  # noqa: E402
client = TestClient(create_app(repository=repo, blobs=blobs, policy_service=PolicyService(PostgresPolicyRepository(repo), blobs)))
api = 'http://platform.invalid'
output = Path('.platform/publication-catalog') / str(publication['id'])
spec = importlib.util.spec_from_file_location('export_platform_publication', 'scripts/export_platform_publication.py')
exporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(exporter)


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
