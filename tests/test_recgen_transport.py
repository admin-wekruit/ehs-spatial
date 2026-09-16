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


def test_unknown_call_is_cancelled_persisted_and_never_automatically_resubmitted(tmp_path, monkeypatch):
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
            raise PollTimeout()
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


@pytest.mark.parametrize('failure', ['cancelled', 'lease_lost', 'database_unavailable', 'download', 'mesh_download'])
def test_live_ownership_loss_terminates_call_and_download_failure_keeps_receipt(tmp_path, monkeypatch, failure):
    events, dispatched = [], {}
    class PollTimeout(Exception): pass
    class Upload:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def put_file(self, *args): pass
    class Call:
        object_id = 'fc-receipt'
        def get(self, timeout):
            assert timeout <= 10
            events.append('poll')
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
