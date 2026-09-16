"""Private, bounded transport for the existing pinned RecGen research runtime."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .contracts import PlatformError, canonical, digest
from .recgen import RECGEN_LICENSES, RECGEN_PINS, RECGEN_WEIGHTS_SHA256, RecGenRequest


def _save(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def invoke(payload, config, *, is_current=None):
    import modal
    from .blender_export import mesh_from_asset
    from .reconstruction import ProviderResponseError

    request = RecGenRequest.from_payload(payload)
    data = request.to_npz()
    sha = hashlib.sha256(data).hexdigest()
    # ponytail: local durable journal is sufficient for the single research worker;
    # multiple hosts must move this dispatch claim into the repository transaction.
    identity = digest({'payloadSha256':sha, 'entityId':request.entity_id, 'seed':request.seed,
        'pins':RECGEN_PINS, 'runtime':payload.get('_researchProtocol', {}).get('runtimeManifest'),
        'functionId':config['modalFunctionId']})
    root = Path(os.environ['PANOPTES_RECGEN_JOURNAL']) / identity
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / 'dispatch.json'
    manifest_path = root / 'manifest.json'
    state, record = {}, {}
    def failure():
        telemetry = {'providerRequestId':state.get('providerRequestId'), 'actualCostUsd':None,
            'workerElapsedSeconds':state.get('wallSeconds', max(0, time.time()-state.get('dispatchedAt', time.time()))),
            'gpuElapsedSeconds':record.get('gpu_function_seconds'), 'timingMethod':'provider_function_and_local_wall',
            'usage':{'views':len(request.views), 'inferenceSeconds':record.get('inference_seconds')}}
        return ProviderResponseError(telemetry, outcome='outcome_unknown')
    volume = modal.Volume.from_name(config['modalVolume'])
    if manifest_path.exists():
        state = json.loads(state_path.read_text())
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            raise failure() from None
    else:
        try:
            descriptor = os.open(state_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                state = json.loads(state_path.read_text())
            except (OSError, ValueError):
                pass
            raise failure() from None
        state = {'payloadSha256':sha, 'entityId':request.entity_id, 'remoteJobId':identity[:32],
                 'status':'preparing', 'functionId':config['modalFunctionId']}
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(canonical(state))
            stream.flush()
            os.fsync(stream.fileno())
        (root / 'input.npz').write_bytes(data)
        function = modal.Function.from_name(config['modalApp'], config['modalFunction'], environment_name='main')
        function.hydrate()
        if function.object_id != config['modalFunctionId']:
            raise PlatformError('recgen_runtime_function_mismatch', 409)
        with volume.batch_upload() as upload:
            upload.put_file(root / 'input.npz', f'/jobs/{identity[:32]}/input.npz')
        function = function.with_options(cpu=(8, 8), memory=(65536, 65536), gpu='A100-80GB',
            timeout=180, retries=0, max_containers=1, buffer_containers=0, scaledown_window=0)
        if is_current is not None and not is_current():
            raise PlatformError('stale_job_attempt', 409)
        state.update(status='dispatching', dispatchedAt=time.time())
        _save(state_path, state)
        call = function.spawn(identity[:32], object_id=request.entity_id, seed=request.seed)
        state.update(status='submitted', providerRequestId=call.object_id)
        try:
            _save(state_path, state)
            deadline = time.monotonic() + 240
            while True:
                if is_current is not None and not is_current():
                    raise PlatformError('stale_job_attempt', 409)
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise PlatformError('provider_outcome_unknown', 502)
                try:
                    manifest = call.get(timeout=min(10, remaining))
                    break
                except modal.exception.TimeoutError as exc:
                    # Poll timeout means the same invocation is still running;
                    # function execution/expiry timeouts are terminal failures.
                    if type(exc) is not modal.exception.TimeoutError:
                        raise
        except Exception:
            state['status'] = 'outcome_unknown'
            try:
                try:
                    _save(state_path, state)
                finally:
                    call.cancel(terminate_containers=True)
            finally:
                raise failure() from None
    try:
        _save(manifest_path, manifest)
        if state['status'] != 'received':
            state.update(status='received', wallSeconds=time.time()-state['dispatchedAt'])
            _save(state_path, state)
        expected_path = f'jobs/{identity[:32]}/result'
        if manifest.get('volume_path') != expected_path:
            raise PlatformError('recgen_output_path_mismatch', 409)
        files = {}
        # Read provider usage first, retaining it even if a mesh download fails.
        for name in sorted(manifest['files'], key=lambda name:name != 'output.json'):
            expected_sha = manifest['files'][name]
            if Path(name).name != name:
                raise PlatformError('recgen_output_path_mismatch', 409)
            target = root / name
            raw = target.read_bytes() if target.is_file() else b''.join(volume.read_file('/'+expected_path+'/'+name))
            if hashlib.sha256(raw).hexdigest() != expected_sha:
                raise PlatformError('recgen_output_hash_mismatch', 409)
            target.write_bytes(raw)
            files[name] = raw
            if name == 'output.json':
                decoded = json.loads(raw)
                if not isinstance(decoded, dict):
                    raise PlatformError('recgen_output_record_invalid', 409)
                record = decoded
        if 'output.json' not in files:
            raise PlatformError('recgen_output_record_missing', 409)
    except Exception:
        raise failure() from None
    result = {'providerRequestId':state['providerRequestId'], 'runtimeEvidence':record,
              'telemetry':{'gpuElapsedSeconds':record.get('gpu_function_seconds'),
                'workerElapsedSeconds':state.get('wallSeconds'), 'timingMethod':'provider_function_and_local_wall',
                'actualCostUsd':None, 'usage':{'views':len(request.views), 'inferenceSeconds':record.get('inference_seconds')}}}
    if record.get('status') != 'complete':
        return {**result, 'providerError':{'code':'recgen_inference_failed'}}
    expected = {'source_payload_sha256':sha, 'model_id':RECGEN_PINS['model'],
                'model_revision':RECGEN_PINS['modelRevision'], 'code_revision':RECGEN_PINS['codeRevision'],
                'weights_manifest_sha256':RECGEN_WEIGHTS_SHA256, 'view_count':len(request.views),
                'object_id':request.entity_id, 'seed':request.seed, 'licenses':RECGEN_LICENSES}
    if any(record.get(key) != value for key,value in expected.items()):
        return {**result, 'providerError':{'code':'recgen_runtime_provenance_mismatch'}}
    try:
        raw_mesh = mesh_from_asset(files['object.glb'], {})
        posed_mesh = mesh_from_asset(files['posed-object.glb'], {})
        object_to_camera = record['object_to_camera']
    except (PlatformError, ValueError, KeyError, TypeError):
        return {**result, 'providerError':{'code':'recgen_output_mesh_invalid'}}
    if not np.array_equal(raw_mesh.faces, posed_mesh.faces):
        return {**result, 'providerError':{'code':'recgen_pose_topology_mismatch'}}
    return {**result, 'vertices':np.asarray(raw_mesh.vertices), 'faces':np.asarray(raw_mesh.faces),
        'colors':raw_mesh.colors,
        'officialPosedVertices':np.asarray(posed_mesh.vertices), 'objectToCamera':object_to_camera,
        'pins':RECGEN_PINS}
