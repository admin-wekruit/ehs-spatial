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
        validation, unplaced, export_info = {}, [], {}
        scene_mode = job.get('inputs', {}).get('sceneMode', 'models')
        if job['kind'] == 'export_blender':
            root = root / 'verified'
            validation = export_scene_revision(revision['id'], document, resolve, root, os.environ.get('PANOPTES_BLENDER_EXECUTABLE'), scene_mode=scene_mode)
            export_info = validation
            unplaced = validation['unplacedEntities']
        else:
            (root / 'scene.json').write_bytes(canonical(document))
            if job['kind'] == 'export_glb':
                prepared = prepare_export(revision['id'], document, resolve, scene_mode=scene_mode)
                validation = write_glb(prepared, root / 'scene.glb')
                unplaced = prepared['unplacedEntities']
                export_info = prepared['manifest']
                (root / 'manifest.json').write_bytes(canonical({**export_info, 'validation': {'glb': validation}}))
        assets = []
        for path in sorted(root.iterdir()):
            if path.is_file():
                media = {'glb': 'model/gltf-binary', 'blend': 'application/x-blender', 'json': 'application/json'}.get(path.suffix[1:], 'application/octet-stream')
                blob = blobs.put(path.read_bytes(), media)
                blob['metadata'] = {'kind': 'scene_export', 'name': path.name, 'sceneRevisionId': revision['id'], 'documentSha256': digest(document)}
                assets.append(repository.register_asset(job['projectId'], blob, job['id']))
        return {'status': export_info.get('status', 'succeeded'), 'sceneRevisionId': revision['id'], 'documentSha256': digest(document), 'assets': assets, 'validation': validation, 'unplacedEntities': unplaced, 'newModelCalls': 0,
                **{key: export_info[key] for key in ('sceneMode', 'exportedModelCount', 'exportedObservedRepresentationCount', 'missingModelEntities', 'modelNotRequiredEntities', 'placementPendingEntities', 'excludedRepresentations') if key in export_info}}


def run_job(repository, blobs, job_id, providers=None):
    repository.blobs = blobs
    try:
        job = repository.claim_job(str(UUID(job_id)))
    except PlatformError as exc:
        if exc.code == 'job_not_claimable':
            return repository.get_job(job_id)
        raise
    started = time.monotonic()
    document, result, continuation = None, {}, None
    with lease(repository, job):
        try:
            if job['kind'] in {'export_json', 'export_glb', 'export_blender'}:
                result = export_job(repository, blobs, job)
            elif job['kind'] == 'reassociate_scene':
                from ehs_spatial.platform.reconstruction import run_reassociation
                document, result = run_reassociation(repository, blobs, job)
            elif job['kind'] == 'validate_model':
                from ehs_spatial.platform.reconstruction import run_research_job
                result = run_research_job(repository, blobs, job)
                pipeline = job.get('config',{}).get('pipeline')
                if (pipeline and pipeline.get('phase') == 'attach' and result.get('outputAssetId') and
                        result.get('status') in ('succeeded','incomplete')):
                    continuation = {'kind':'reconstruct_scene','inputs':{
                        'phase':'attach','entityIds':pipeline['entityIds'],'processed':pipeline['processed'],
                        'entityId':pipeline['entityId'],'researchJobId':job['id']},'config':{}}
            elif job['kind'] == 'reconstruct_scene':
                from ehs_spatial.platform.reconstruction_pipeline import run_reconstruction_pipeline
                document, result, continuation = run_reconstruction_pipeline(repository, blobs, job, providers)
            elif job['kind'] == 'review_models':
                from ehs_spatial.platform.reconstruction import run_model_review
                document, result = run_model_review(repository, blobs, job, providers)
            elif job['kind'] in {'analyze_capture', 'generate_object', 'generate_scene', 'segment_object'}:
                from ehs_spatial.platform import reconstruction
                manifest = job['config'].get('providerManifest',{})
                if job['kind'] == 'analyze_capture' and manifest.get('generation',{}).get('pins',{}).get('model') == 'TRI-ML/RecGen':
                    manifest = {key:value for key,value in manifest.items() if key != 'generation'}
                providers = reconstruction.providers_from_manifest(manifest) if providers is None else providers
                name = {'analyze_capture': 'run_capture_pipeline', 'generate_object': 'run_generation', 'generate_scene': 'run_generation', 'segment_object': 'run_segmentation'}[job['kind']]
                document, result = getattr(reconstruction, name)(repository, blobs, job, providers)
                continuation = result.pop('_continuation', None)
            else:
                raise PlatformError('unsupported_job_kind', 422, kind=job['kind'])
            if document is not None:
                from ehs_spatial.platform.scene_measurements import analyze_bends, analyze_inclinations
                def measurement_asset(identity):
                    asset = repository.get_asset(identity)
                    return blobs.get(asset['storageKey'], asset['sha256'], asset['sizeBytes'])
                analysis = analyze_bends({'id': '', 'document': document}, measurement_asset, persist=True)
                result['bendAnalysis'] = {'algorithm': analysis['algorithm'],
                    'outcomes': [{k:v for k,v in row.items() if k != 'result'} for row in analysis['items']]}
                inclinations = analyze_inclinations({'id':'','document':document}, measurement_asset, persist=True)
                result['inclinationAnalysis'] = {'algorithm':inclinations['algorithm'],
                    'outcomes':[{k:v for k,v in row.items() if k!='surfaces'} for row in inclinations['items']]}
            status = result.get('status', 'succeeded')
        except PlatformError as exc:
            from ehs_spatial.platform.reconstruction import UNKNOWN_OUTCOME_CODES
            status = 'outcome_unknown' if exc.code in UNKNOWN_OUTCOME_CODES else 'failed'
            result = {'error': {'code': exc.code, 'params': exc.params}}
        except Exception as exc:
            # Only the exception class enters the public record: provider text
            # can contain private endpoints or credential-bearing request URLs.
            status = 'failed'
            result = {'error': {'code': 'worker_error', 'params': {'type': type(exc).__name__}}}
        result['timings'] = {**result.get('timings', {}), 'workerSeconds': time.monotonic() - started}
        if job['kind'] == 'validate_model':
            document = None
            result.update(scope='research_only', productReleaseStatus='not_changed', sceneRevision=None)
            pipeline, error = job.get('config',{}).get('pipeline') or {}, result.get('error') or {}
            if (status == 'failed' and pipeline.get('phase') == 'attach'
                    and error.get('code') in ('provider_failed','provider_response_invalid')
                    and error.get('params',{}).get('stage') == 'generation'):
                from ehs_spatial.platform.reconstruction_pipeline import _next, _result
                processed = [*pipeline['processed'], {'entityId':pipeline['entityId'], 'status':'failed',
                    'reason':error['code'], 'researchJobId':job['id']}]
                remaining = pipeline['entityIds'][1:]
                result = _result(pipeline['phase'], processed, remaining,
                    capture_analysis=job['config'].get('captureAnalysis'),
                    **{**result, 'generationStatus':'failed', 'status':'incomplete'})
                status, continuation = 'incomplete', _next(remaining, processed)
        options = {'continuation':continuation} if continuation is not None else {}
        try:
            return repository.finish_job(job['id'], job['attemptToken'], status, document=document, result=result, **options)
        except PlatformError as exc:
            if continuation is None:
                raise
            # The outbox transaction rolled back; keep the completed analysis
            # and exact stop reason without dispatching an unvalidated child.
            result['continuationStopped'] = exc.code
            return repository.finish_job(job['id'], job['attemptToken'], 'incomplete', document=document, result=result)


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
