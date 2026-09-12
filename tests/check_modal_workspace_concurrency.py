"""Run: python3 tests/check_modal_workspace_concurrency.py. Stdlib only; zero providers."""
import ast
import asyncio
import os
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


async def check():
    source = Path(__file__).resolve().parents[1] / 'modal_apps/report_workspace_app.py'
    definition = next(node for node in ast.parse(source.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == 'web')
    concurrency = next(node for node in definition.decorator_list
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr == 'concurrent')
    assert next(ast.literal_eval(k.value) for k in concurrency.keywords if k.arg == 'max_inputs') == 16
    definition.decorator_list = []
    events = []
    active = 0
    fail_reconcile = False
    started, release, second_started = asyncio.Event(), asyncio.Event(), asyncio.Event()

    def reload():
        nonlocal active
        assert active == 0, 'volume reload overlapped an open report response'
        active = 1
        events.append('reload')

    def commit():
        nonlocal active
        assert active == 1
        active = 0
        events.append('commit')

    def reconcile():
        assert active == 1
        events.append('reconcile')
        if fail_reconcile:
            raise RuntimeError('reconcile failed')

    async def server(scope, receive, send):
        path = scope['path']
        events.append('server:' + path)
        if path == '/report/run/fail':
            raise RuntimeError('response failed')
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        if path in {'/report/run/surface/model.glb', '/report/run/cancel'}:
            await send({'type': 'http.response.body', 'body': b'model chunk', 'more_body': True})
        await send({'type': 'http.response.body', 'body': b'done', 'more_body': False})

    server.state = SimpleNamespace(workspace_write_auth=True)
    package, service = ModuleType('ehs_spatial'), ModuleType('ehs_spatial.serve')
    package.report_workspace = SimpleNamespace(JOBS=SimpleNamespace(shutdown=lambda **kwargs: None))
    service.app = server
    namespace = {'os': os, 'reports': SimpleNamespace(reload=reload, commit=commit),
                 'DurableUploads': lambda: SimpleNamespace(shutdown=lambda **kwargs: None), 'reconcile_finished_jobs': reconcile}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), namespace)
    with patch.dict(sys.modules, {'ehs_spatial': package, 'ehs_spatial.serve': service}), \
            patch.dict(os.environ, {'PANOPTES_WRITE_TOKEN': 'concurrency-check-only'}):
        app = namespace['web']()

    async def receive():
        return {'type': 'http.request', 'body': b''}

    async def send(message):
        if message.get('more_body'):
            started.set()
            await release.wait()

    async def request(path, method='GET'):
        await app({'type': 'http', 'path': path, 'method': method}, receive, send)

    async def second():
        second_started.set()
        await request('/api/reports/run')

    first = asyncio.create_task(request('/report/run/surface/model.glb'))
    await asyncio.wait_for(started.wait(), 1)
    queued = asyncio.create_task(second())
    await second_started.wait()
    # Login and bundled frontend assets must complete while the model stream
    # remains blocked; a second volume operation must remain behind the lock.
    for path in ['/api/session', '/published', '/published/viewer.html',
                 '/workspace-assets', '/workspace-assets/workspace.js']:
        await asyncio.wait_for(request(path, 'POST' if path == '/api/session' else 'GET'), 1)
    assert events.count('reload') == 1 and events.count('commit') == 0
    assert not queued.done() and 'server:/api/reports/run' not in events
    release.set()
    await asyncio.wait_for(asyncio.gather(first, queued), 1)
    assert [value for value in events if value in {'reload', 'commit', 'reconcile'}] == [
        'reload', 'commit', 'reload', 'reconcile', 'commit']
    assert active == 0

    # Both handler and job-reconciliation exceptions commit and release the lock.
    for path in ['/report/run/fail', '/api/reports/fail']:
        fail_reconcile = path.startswith('/api/')
        before = events.count('commit')
        try:
            await asyncio.wait_for(request(path), 1)
        except RuntimeError:
            pass
        else:
            raise AssertionError('expected request exception')
        assert events.count('commit') == before + 1 and active == 0
        fail_reconcile = False
        await asyncio.wait_for(request('/report/run/after-failure'), 1)

    # A disconnected streaming request must release the same volume boundary.
    started.clear()
    release.clear()
    cancelled = asyncio.create_task(request('/report/run/cancel'))
    await asyncio.wait_for(started.wait(), 1)
    before = events.count('commit')
    cancelled.cancel()
    try:
        await cancelled
    except asyncio.CancelledError:
        pass
    assert events.count('commit') == before + 1 and active == 0
    await asyncio.wait_for(request('/api/session/not-the-session-route'), 1)
    assert events[-1] == 'commit', 'only exact /api/session bypasses the volume'
    # to_thread keeps running after its awaiting request is cancelled. Block each
    # real synchronous I/O phase and prove its thread finishes before lock reuse.
    for phase in ['reload', 'commit']:
        io_started, io_release = threading.Event(), threading.Event()
        guard = threading.Lock()
        trace = []
        io_active = 0
        block_once = True

        def operation(kind):
            nonlocal io_active, block_once
            with guard:
                assert io_active == 0, 'cancelled request released the volume lock before its thread ended'
                io_active += 1
                trace.append('start:' + kind)
                block = block_once and kind == phase
                if block:
                    block_once = False
            try:
                if block:
                    io_started.set()
                    io_release.wait()
            finally:
                with guard:
                    trace.append('end:' + kind)
                    io_active -= 1

        namespace['reports'] = SimpleNamespace(reload=lambda: operation('reload'),
                                                commit=lambda: operation('commit'))
        with patch.dict(sys.modules, {'ehs_spatial': package, 'ehs_spatial.serve': service}), \
                patch.dict(os.environ, {'PANOPTES_WRITE_TOKEN': 'concurrency-check-only'}):
            app = namespace['web']()
        cancelled = asyncio.create_task(request('/report/run/thread-cancellation'))
        try:
            await asyncio.wait_for(asyncio.to_thread(io_started.wait), 1)
            cancelled.cancel()
            await asyncio.sleep(0)
            cancelled.cancel()  # repeated cancellation must also keep the lock
            queued_started = asyncio.Event()

            async def after_thread():
                queued_started.set()
                await request('/report/run/after-thread')

            queued = asyncio.create_task(after_thread())
            await queued_started.wait()
            await asyncio.wait_for(request('/api/session', 'POST'), 1)
            assert not cancelled.done() and not queued.done()
            assert trace.count('start:reload') == 1 and io_active == 1
            io_release.set()
            try:
                await asyncio.wait_for(cancelled, 1)
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError('request cancellation must be preserved')
            await asyncio.wait_for(queued, 1)
            second_reload = trace.index('start:reload', trace.index('start:reload') + 1)
            assert trace.index('end:' + phase) < second_reload and io_active == 0
        finally:
            io_release.set()
    print('PASS: stateless login/static concurrency, serialized volume streams, exception/cancellation commit, thread-safe reload/commit cancellation; zero providers')


if __name__ == '__main__':
    asyncio.run(check())
