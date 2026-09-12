from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import mimetypes
import os
from pathlib import Path
import tempfile
import threading
import time
from uuid import UUID

from ehs_spatial.platform.contracts import PlatformError, canonical, digest


@contextmanager
def lease(repository, job):
    stop = threading.Event()
    def maintain():
        while not stop.wait(30):
            try:
                if not repository.heartbeat_job(job['id'], job['attemptToken']):
                    return
            except Exception:
                # A disconnected worker cannot renew ownership; the repository
                # fences subsequent reservations and publication of its result.
                return
    thread = threading.Thread(target=maintain, name='job-lease', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


def export_job(repository, blobs, job):
    from ehs_spatial.platform.blender_export import prepare_export, write_glb, export_scene_revision
    revision = repository.get_revision(job['baseRevisionId'])
    document = revision['document']
    def resolve(identity):
        # A fork may retain immutable public source assets. Only assets already
        # referenced in its fixed scene are available to the exporter.
        if identity not in {a['id'] for a in document['assets']}:
            raise PlatformError('unreferenced_export_asset', 422)
        asset = repository.get_asset(identity)
        return blobs.get(asset['storageKey'], asset['sha256'], asset['sizeBytes'])
    with tempfile.TemporaryDirectory(prefix='panoptes-export-') as tmp:
        root = Path(tmp)
        validation, unplaced = {}, []
        if job['kind'] == 'export_blender':
            root = root / 'verified'
            validation = export_scene_revision(revision['id'], document, resolve, root, os.environ.get('PANOPTES_BLENDER_EXECUTABLE'))
            unplaced = validation['unplacedEntities']
        else:
            (root / 'scene.json').write_bytes(canonical(document))
            if job['kind'] == 'export_glb':
                prepared = prepare_export(revision['id'], document, resolve)
                validation = write_glb(prepared, root / 'scene.glb')
                unplaced = prepared['unplacedEntities']
        assets = []
        for path in sorted(root.iterdir()):
            if path.is_file():
                media = {'glb': 'model/gltf-binary', 'blend': 'application/x-blender', 'json': 'application/json'}.get(path.suffix[1:], 'application/octet-stream')
                blob = blobs.put(path.read_bytes(), media)
                blob['metadata'] = {'kind': 'scene_export', 'name': path.name, 'sceneRevisionId': revision['id'], 'documentSha256': digest(document)}
                assets.append(repository.register_asset(job['projectId'], blob, job['id']))
        return {'status': 'incomplete' if unplaced else 'succeeded', 'sceneRevisionId': revision['id'], 'documentSha256': digest(document), 'assets': assets, 'validation': validation, 'unplacedEntities': unplaced, 'newModelCalls': 0}


def run_job(repository, blobs, job_id, providers=None):
    repository.blobs = blobs
    try:
        job = repository.claim_job(str(UUID(job_id)))
    except PlatformError as exc:
        if exc.code == 'job_not_claimable':
            return repository.get_job(job_id)
        raise
    started = time.monotonic()
    document, result = None, {}
    with lease(repository, job):
        try:
            if job['kind'] in {'export_json', 'export_glb', 'export_blender'}:
                result = export_job(repository, blobs, job)
            elif job['kind'] in {'analyze_capture', 'generate_object', 'generate_scene', 'segment_object'}:
                from ehs_spatial.platform import reconstruction
                providers = reconstruction.providers_from_manifest(job["config"].get("providerManifest", {})) if providers is None else providers
                name = {'analyze_capture': 'run_analysis', 'generate_object': 'run_generation', 'generate_scene': 'run_generation', 'segment_object': 'run_segmentation'}[job['kind']]
                document, result = getattr(reconstruction, name)(repository, blobs, job, providers)
            else:
                raise PlatformError('unsupported_job_kind', 422, kind=job['kind'])
            status = result.get('status', 'succeeded')
        except PlatformError as exc:
            status = 'outcome_unknown' if 'outcome_unknown' in exc.code else 'failed'
            result = {'error': {'code': exc.code, 'params': exc.params}}
        except Exception as exc:
            # Only the exception class enters the public record: provider text
            # can contain private endpoints or credential-bearing request URLs.
            status = 'failed'
            result = {'error': {'code': 'worker_error', 'params': {'type': type(exc).__name__}}}
        result['timings'] = {**result.get('timings', {}), 'workerSeconds': time.monotonic() - started}
        return repository.finish_job(job['id'], job['attemptToken'], status, document=document, result=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True, type=UUID)
    args = parser.parse_args()
    from ehs_spatial.platform.runtime import services
    repository, blobs = services()
    outcome = run_job(repository, blobs, str(args.job_id))
    print(json.dumps({'jobId': outcome['id'], 'status': outcome['status'], 'headAdvanced': outcome.get('headAdvanced', False)}))


if __name__ == '__main__':
    main()
