"""One dispatch only: uncertain provider outcomes retain the journal and reservation."""
import json
import hashlib
import sys
from types import SimpleNamespace

import pytest
from test_recgen_research import payload
from ehs_spatial.platform.contracts import PlatformError
from ehs_spatial.platform.reconstruction import ProviderResponseError
from ehs_spatial.platform.recgen_transport import invoke


@pytest.mark.parametrize('builtin_poll_timeout', [False, True])
def test_unknown_call_is_cancelled_persisted_and_never_automatically_resubmitted(tmp_path, monkeypatch, builtin_poll_timeout):
    events = []
    clock = [0.]
    class PollTimeout(Exception): pass
    monkeypatch.setattr('ehs_spatial.platform.recgen_transport.time.monotonic', lambda:clock[0])
    class Upload:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def put_file(self, *args): events.append('upload')
    class Call:
        object_id = 'fc-test'
        def get(self, timeout):
            assert 0 < timeout <= 10
            clock[0] += timeout
            raise TimeoutError() if builtin_poll_timeout else PollTimeout()
        def cancel(self, **kwargs): events.append(('cancel', kwargs))
    class Function:
        object_id = 'fu-pinned'
        def hydrate(self): pass
        def with_options(self, **kwargs):
            assert kwargs['retries'] == 0 and kwargs['timeout'] == 180
            assert kwargs['max_containers'] == 1
            return self
        def spawn(self, *args, **kwargs):
            events.append('spawn')
            return Call()
    modal = SimpleNamespace(exception=SimpleNamespace(TimeoutError=PollTimeout), Volume=SimpleNamespace(from_name=lambda *_:SimpleNamespace(batch_upload=Upload)),
        Function=SimpleNamespace(from_name=lambda *args, **kwargs:Function()))
    monkeypatch.setitem(sys.modules, 'modal', modal)
    monkeypatch.setenv('PANOPTES_RECGEN_JOURNAL', str(tmp_path))
    config = {'modalVolume':'private', 'modalApp':'research', 'modalFunction':'generate', 'modalFunctionId':'fu-pinned'}
    for _ in range(2):
        with pytest.raises(ProviderResponseError) as caught:
            invoke(payload(), config)
        assert caught.value.outcome == 'outcome_unknown'
        assert caught.value.telemetry['providerRequestId'] == 'fc-test'
        assert caught.value.telemetry['workerElapsedSeconds'] >= 0
        assert clock[0] >= 240, 'A normal SDK poll timeout must not terminate running inference'
    state = json.loads(next(tmp_path.glob('*/dispatch.json')).read_text())
    assert state['status'] == 'outcome_unknown' and state['providerRequestId'] == 'fc-test'
    assert events.count('spawn') == 1
    assert events[-1] == ('cancel', {'terminate_containers':True})

    changed = payload()
    changed['seed'] += 1
    with pytest.raises(ProviderResponseError):
        invoke(changed, config)
    assert events.count('spawn') == 2, 'Different seeds must never reuse another dispatch journal'
    assert len(list(tmp_path.glob('*/dispatch.json'))) == 2

    retry = payload()
    retry['_researchProtocol'] = {'id':'same-input-reviewed-retry', 'dispatchAttemptId':'operator-reviewed-terminal-retry-1'}
    for _ in range(2):
        with pytest.raises(ProviderResponseError):
            invoke(retry, config)
    assert events.count('spawn') == 3, 'An explicit new attempt is distinct; the same attempt never respawns'
    old_protocol = payload()
    old_protocol['_researchProtocol'] = {'id':'different-protocol-id-only'}
    with pytest.raises(ProviderResponseError):
        invoke(old_protocol, config)
    assert events.count('spawn') == 3, 'A protocol label alone must not bypass the original journal'


def test_installed_modal_pending_result_raises_builtin_timeout_without_network():
    import asyncio
    functions = pytest.importorskip('modal._functions')
    async def pending(**kwargs):
        return SimpleNamespace(outputs=[], num_unfinished_inputs=1)
    with pytest.raises(TimeoutError):
        asyncio.run(functions._Invocation.poll_function(SimpleNamespace(pop_function_call_outputs=pending), timeout=10))


@pytest.mark.parametrize('failure', ['cancelled', 'lease_lost', 'database_unavailable', 'function_timeout', 'download', 'mesh_download'])
def test_live_ownership_loss_terminates_call_and_download_failure_keeps_receipt(tmp_path, monkeypatch, failure):
    events, dispatched = [], {}
    class PollTimeout(Exception): pass
    class FunctionTimeout(PollTimeout): pass
    class Upload:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def put_file(self, *args): pass
    class Call:
        object_id = 'fc-receipt'
        def get(self, timeout):
            assert timeout <= 10
            events.append('poll')
            if failure == 'function_timeout': raise FunctionTimeout()
            if failure in ('download', 'mesh_download'):
                return {'volume_path':f"jobs/{dispatched['id']}/result", 'files':{
                    'object.glb':'a' * 64, 'output.json':hashlib.sha256(record).hexdigest()}}
            raise PollTimeout()
        def cancel(self, **kwargs): events.append(('cancel', kwargs))
    class Function:
        object_id = 'fu-pinned'
        def hydrate(self): pass
        def with_options(self, **kwargs): return self
        def spawn(self, identity, **kwargs):
            dispatched['id'] = identity
            events.append('spawn')
            return Call()
    record = json.dumps({'status':'complete', 'gpu_function_seconds':12, 'inference_seconds':8}).encode()
    def read_file(path):
        if failure == 'mesh_download' and path.endswith('/output.json'): return [record]
        raise OSError('volume disconnected')
    def is_current():
        if 'poll' not in events or failure in ('download', 'mesh_download'): return True
        if failure == 'database_unavailable': raise ConnectionError('private connection')
        return False  # heartbeat rejects cancellation, changed token or expired lease.
    monkeypatch.setitem(sys.modules, 'modal', SimpleNamespace(exception=SimpleNamespace(TimeoutError=PollTimeout),
        Volume=SimpleNamespace(from_name=lambda *_:SimpleNamespace(batch_upload=Upload, read_file=read_file)),
        Function=SimpleNamespace(from_name=lambda *a, **kw:Function())))
    monkeypatch.setenv('PANOPTES_RECGEN_JOURNAL', str(tmp_path))
    config = {'modalVolume':'private','modalApp':'research','modalFunction':'generate','modalFunctionId':'fu-pinned'}
    with pytest.raises(ProviderResponseError) as caught:
        invoke(payload(), config, is_current=is_current)
    assert caught.value.telemetry['providerRequestId'] == 'fc-receipt'
    assert caught.value.telemetry['actualCostUsd'] is None
    assert caught.value.outcome == 'outcome_unknown'
    assert events.count('spawn') == events.count('poll') == 1
    if failure in ('download', 'mesh_download'):
        assert next(tmp_path.glob('*/manifest.json')).is_file(), 'Received manifest remains recoverable'
        if failure == 'mesh_download':
            assert caught.value.telemetry['gpuElapsedSeconds'] == 12
            assert next(tmp_path.glob('*/output.json')).read_bytes() == record
    else:
        assert events[-1] == ('cancel', {'terminate_containers':True})
    state = json.loads(next(tmp_path.glob('*/dispatch.json')).read_text())
    assert state['providerRequestId'] == 'fc-receipt'
    assert state['failureType'].endswith('OSError' if failure in ('download','mesh_download') else
        'ConnectionError' if failure == 'database_unavailable' else 'FunctionTimeout' if failure == 'function_timeout' else 'PlatformError')


@pytest.mark.parametrize('attempt', [None, '', ' ', 1, 'a' * 129, '../retry'])
def test_invalid_explicit_dispatch_attempt_rejected_before_provider(attempt):
    from test_recgen_research import research_configuration
    from ehs_spatial.platform.reconstruction import _validate_research_protocol
    _, protocol = research_configuration(payload(), [])
    protocol['dispatchAttemptId'] = attempt
    with pytest.raises(PlatformError) as caught:
        _validate_research_protocol(protocol)
    assert caught.value.code == 'invalid_research_dispatch_attempt'


@pytest.mark.parametrize('reported', [True, None, 'malformed-settings'])
def test_provider_cannot_silently_ignore_explicit_no_erosion(tmp_path, monkeypatch, reported):
    from ehs_spatial.platform.contracts import digest
    from ehs_spatial.platform.recgen import RECGEN_PINS, RecGenRequest
    value = payload()
    value['maskErosionEnabled'] = False
    data = RecGenRequest.from_payload(value).to_npz()
    identity = digest({'payloadSha256':hashlib.sha256(data).hexdigest(), 'entityId':'entity',
        'seed':42, 'pins':RECGEN_PINS, 'runtime':None, 'functionId':'fu-test'})
    folder = tmp_path / identity
    folder.mkdir()
    record = json.dumps({'status':'complete','input_contract_version':'recgen-input-v2',
        'inference_settings':None if reported == 'malformed-settings' else {'mask_erosion_enabled':reported}}).encode()
    (folder / 'dispatch.json').write_text(json.dumps({'status':'received','providerRequestId':'fc-test','wallSeconds':2}))
    (folder / 'manifest.json').write_text(json.dumps({'volume_path':f'jobs/{identity[:32]}/result',
        'files':{'output.json':hashlib.sha256(record).hexdigest()}}))
    (folder / 'output.json').write_bytes(record)
    monkeypatch.setenv('PANOPTES_RECGEN_JOURNAL', str(tmp_path))
    monkeypatch.setitem(sys.modules,'modal',SimpleNamespace(Volume=SimpleNamespace(from_name=lambda *_:object())))
    result = invoke(value, {'modalVolume':'private','modalFunctionId':'fu-test'})
    assert result['providerError']['code'] == 'recgen_preprocessing_mismatch'
    assert result['providerRequestId'] == 'fc-test'


@pytest.mark.parametrize('same', [True, False])
def test_an_identical_input_already_on_the_volume_is_not_uploaded_again(tmp_path, monkeypatch, same):
    """A run that failed after its uploads leaves each job's input.npz on the volume; the next attempt found it there and
    stopped on FileExistsError (Lightning RecGen, runs 3-4). The same bytes are reused; other bytes are refused, never replaced."""
    from ehs_spatial.platform.recgen import RecGenRequest
    data = RecGenRequest.from_payload(payload()).to_npz()
    record = json.dumps({'status':'failed'}).encode()
    events = []
    class Upload:
        def __enter__(self): return self
        def __exit__(self, *args): raise FileExistsError('/jobs/x/input.npz: already exists')  # as modal's batch_upload on exit
        def put_file(self, *args): events.append('upload')
    def read_file(path):
        events.append(('read', path.rsplit('/', 1)[-1]))
        return [data if same else b'other'] if path.endswith('/input.npz') else [record]
    class Call:
        object_id = 'fc-again'
        def get(self, timeout): return {'volume_path':f"jobs/{dispatched[0]}/result", 'files':{'output.json':hashlib.sha256(record).hexdigest()}}
    dispatched = []
    class Function:
        object_id = 'fu-pinned'
        def hydrate(self): pass
        def with_options(self, **kwargs): return self
        def spawn(self, identity, **kwargs):
            dispatched.append(identity)
            return Call()
    monkeypatch.setitem(sys.modules, 'modal', SimpleNamespace(exception=SimpleNamespace(TimeoutError=TimeoutError),
        Volume=SimpleNamespace(from_name=lambda *_:SimpleNamespace(batch_upload=Upload, read_file=read_file)),
        Function=SimpleNamespace(from_name=lambda *a, **kw:Function())))
    monkeypatch.setenv('PANOPTES_RECGEN_JOURNAL', str(tmp_path))
    config = {'modalVolume':'private','modalApp':'research','modalFunction':'generate','modalFunctionId':'fu-pinned'}
    if same:
        result = invoke(payload(), config)
        assert result['providerError']['code'] == 'recgen_inference_failed' and len(dispatched) == 1
        assert events[:2] == ['upload', ('read', 'input.npz')], 'the volume copy is read back and compared, not overwritten'
    else:
        with pytest.raises(PlatformError) as caught:
            invoke(payload(), config)
        assert caught.value.code == 'recgen_input_mismatch' and dispatched == []
