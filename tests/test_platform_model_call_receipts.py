"""Fake Modal transport and real SQL: one reservation, one recoverable invocation."""
import sys
from types import SimpleNamespace

import pytest

from ehs_spatial.platform.contracts import PlatformError
from ehs_spatial.platform.reconstruction import MAP_PINS, _Stages, providers_from_manifest
from ehs_spatial.platform.storage import LocalBlobStore
from test_platform_backend import repo, project, make_job, identity


def ledger(repo):
    with repo._connect() as connection:
        return connection.execute('SELECT * FROM model_calls ORDER BY created_at').fetchall()


@pytest.mark.parametrize('stage', ['geometry', 'depth'])
@pytest.mark.parametrize('outcome', ['success', 'invalid_response', 'provider_failed', 'unknown', 'receipt_failed', 'asset_failed'])
def test_modal_receipt_precedes_get_and_survives_every_outcome(repo, tmp_path, monkeypatch, stage, outcome):
    cap, scene = project(repo)
    job = repo.claim_job(make_job(repo, cap, scene)['id'])
    pins = MAP_PINS if stage == 'geometry' else {'model': 'Ruicheng/moge-3-vitl'}
    evidence = {'pins': pins, **{gate: {'status': 'passed', 'artifactSha256': 'a' * 64}
        for gate in ('license', 'runtime', 'quality')}}
    manifest = {stage: {'provider': 'modal', 'pins': pins, 'estimatedCostUsd': .01,
        'releaseEvidence': evidence, 'modalApp': 'test-only', 'modalClass': 'Model', 'modalMethod': 'run'}}
    events = []

    class Call:
        object_id = 'fc-original'
        def get(self):
            events.append('get')
            assert ledger(repo)[0]['response']['providerRequestId'] == self.object_id
            if outcome == 'unknown':
                raise TimeoutError('provider result unavailable')
            if outcome == 'invalid_response':
                return None
            return {'pins': pins, 'telemetry': {'gpuElapsedSeconds': .2},
                **({'providerError': {'code': 'gpu_failed'}} if outcome == 'provider_failed' else {})}
        def cancel(self, **kwargs):
            events.append('cancel')

    def spawn(*args, **kwargs):
        events.append('spawn')
        assert len(ledger(repo)) == 1 and ledger(repo)[0]['status'] == 'reserved'
        return Call()

    monkeypatch.setitem(sys.modules, 'modal', SimpleNamespace(Cls=SimpleNamespace(
        from_name=lambda *a: lambda: SimpleNamespace(run=SimpleNamespace(spawn=spawn)))))
    repo.blobs = LocalBlobStore(tmp_path)
    stages = _Stages(repo, repo.blobs, job, providers_from_manifest(manifest))
    if outcome == 'receipt_failed':
        def unavailable(*args):
            raise ConnectionError('receipt transaction unavailable')
        monkeypatch.setattr(repo, 'record_model_call_dispatch', unavailable, raising=False)
    if outcome == 'asset_failed':
        monkeypatch.setattr(stages, 'put', lambda *a, **kw: (_ for _ in ()).throw(OSError('disk unavailable')))
    if outcome == 'success':
        output, _ = stages.call(stage, [], {})
        assert output['providerRequestId'] == 'fc-original'
    else:
        with pytest.raises(PlatformError):
            stages.call(stage, [], {})
    row = ledger(repo)[0]
    assert row['response']['providerRequestId'] == 'fc-original'
    assert row['status'] == {'success': 'succeeded', 'invalid_response': 'failed',
        'provider_failed': 'failed', 'unknown': 'outcome_unknown',
        'receipt_failed': 'outcome_unknown', 'asset_failed': 'outcome_unknown'}[outcome]
    if outcome == 'receipt_failed':
        assert 'get' not in events and events.count('cancel') == 1
    if outcome == 'unknown':
        assert 'cancel' not in events, 'A failed read must leave the original invocation recoverable'
    try:
        stages.call(stage, [], {})
    except PlatformError:
        pass
    assert len(ledger(repo)) == events.count('spawn') == 1
    assert ledger(repo)[0]['estimated_cost'] == row['estimated_cost']


@pytest.mark.parametrize('invalid', ['token', 'cancelled', 'expired'])
def test_dispatch_receipt_requires_current_reserved_attempt(repo, invalid):
    cap, scene = project(repo)
    job = repo.claim_job(make_job(repo, cap, scene)['id'])
    call = repo.reserve_model_call(job['id'], job['attemptToken'], 'modal', 'model', 'key', .01)
    token = job['attemptToken']
    if invalid == 'token':
        token = identity()
    elif invalid == 'cancelled':
        repo.cancel_job(job['id'], cap)
    else:
        with repo._connect() as connection:
            connection.execute("UPDATE jobs SET lease_expires_at=now()-interval '1 minute' WHERE id=%s", (job['id'],))
    with pytest.raises(PlatformError, match='stale_job_attempt'):
        repo.record_model_call_dispatch(call['id'], token, 'fc-original')
    assert not ledger(repo)[0]['response']


@pytest.mark.parametrize('terminal', ['succeeded', 'failed', 'outcome_unknown'])
def test_receipt_id_is_immutable_through_completion_and_late_result(repo, terminal):
    cap, scene = project(repo)
    job = repo.claim_job(make_job(repo, cap, scene)['id'])
    call = repo.reserve_model_call(job['id'], job['attemptToken'], 'modal', 'model', 'key', .01)
    first = repo.record_model_call_dispatch(call['id'], job['attemptToken'], 'fc-original')
    assert repo.record_model_call_dispatch(call['id'], job['attemptToken'], 'fc-original') == first
    with pytest.raises(PlatformError, match='model_dispatch_conflict'):
        repo.record_model_call_dispatch(call['id'], job['attemptToken'], 'fc-other')
    with pytest.raises(PlatformError, match='model_dispatch_conflict'):
        repo.complete_model_call(call['id'], terminal, response={'providerRequestId': 'fc-other'})
    completed = repo.complete_model_call(call['id'], terminal, response={'providerRequestId': None, 'stage': 'depth'})
    assert completed['response']['providerRequestId'] == 'fc-original'
    if terminal == 'outcome_unknown':
        late = repo.complete_model_call(call['id'], 'succeeded', response={'assetId': 'recovered-original-result'})
        assert late['status'] == 'outcome_unknown'
        assert late['response']['providerRequestId'] == 'fc-original'
        assert late['response']['lateOutcome']['response']['providerRequestId'] == 'fc-original'
        assert repo.complete_model_call(call['id'], 'succeeded', response={'assetId': 'recovered-original-result'}) == late
    with pytest.raises(PlatformError, match='model_call_already_reserved'):
        repo.reserve_model_call(job['id'], job['attemptToken'], 'modal', 'model', 'key', .01)


def test_lease_recovery_during_spawn_retains_first_late_receipt(repo):
    cap, scene = project(repo)
    job = repo.claim_job(make_job(repo, cap, scene)['id'])
    call = repo.reserve_model_call(job['id'], job['attemptToken'], 'modal', 'model', 'key', .01)
    with repo._connect() as connection:
        connection.execute("UPDATE jobs SET lease_expires_at=now()-interval '1 minute' WHERE id=%s", (job['id'],))
    repo.recover_expired_jobs()
    with pytest.raises(PlatformError, match='stale_job_attempt'):
        repo.record_model_call_dispatch(call['id'], job['attemptToken'], 'fc-original')
    result = repo.complete_model_call(call['id'], 'outcome_unknown', response={'providerRequestId': 'fc-original'})
    assert result['response']['providerRequestId'] == 'fc-original'
    assert result['status'] == repo.get_job(job['id'])['status'] == 'outcome_unknown'
    assert repo.complete_model_call(call['id'], 'outcome_unknown', response={'providerRequestId': 'fc-original'}) == result
    with pytest.raises(PlatformError, match='model_dispatch_conflict'):
        repo.complete_model_call(call['id'], 'outcome_unknown', response={'providerRequestId': 'fc-other'})
    with pytest.raises(PlatformError, match='model_call_already_reserved'):
        repo.reserve_model_call(job['id'], job['attemptToken'], 'modal', 'model', 'key', .01)
