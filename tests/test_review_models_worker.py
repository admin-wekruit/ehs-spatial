"""Review jobs use the real isolated PostgreSQL worker/CAS and fake transports."""
from copy import deepcopy

import pytest

from ehs_spatial.platform import reconstruction
from ehs_spatial.platform.correspondence import audit_correspondence
from panoptes_worker.__main__ import run_job
from test_capture_worker_chain import chain, dispatch, model_calls, start_research, successor, user_edit
from test_platform_backend import identity, repo


def existing_model_job(chain):
    _, research = start_research(chain)
    generated = dispatch(chain, research['id'])
    attached = dispatch(chain, successor(chain, generated)['id'])
    source = chain.repo.get_revision(attached['resultRevisionId'])['document']
    target = next(entity for entity in source['entities'] if entity.get('activeModelRepresentationId'))
    model = next(rep for rep in target['representations'] if rep['id'] == target['activeModelRepresentationId'])
    for field in ('qualityEvidence', 'qualityBinding'):
        model.pop(field)
    # Install this unreviewed fixture only through the normal immutable result writer.
    fixture = chain.repo.create_job(research['projectId'], chain.cap, {'requestId':identity(),
        'branchId':research['branchId'], 'baseRevisionId':attached['resultRevisionId'], 'kind':'fixture_import'})
    claimed = chain.repo.claim_job(fixture['id'])
    installed = chain.repo.finish_job(fixture['id'], claimed['attemptToken'], 'succeeded', document=source)
    job = chain.repo.create_job(research['projectId'], chain.cap, {'requestId':identity(),
        'branchId':research['branchId'], 'baseRevisionId':installed['resultRevisionId'], 'kind':'review_models',
        'inputs':{'entityIds':[target['id']]}, 'config':{'providerManifest':{'generation':{'client':'must not execute'}}}})
    return job, target, deepcopy(source)


@pytest.mark.parametrize('concurrent', [None, 'edit', 'cancel', 'stale_before_review'])
def test_review_worker_preserves_source_and_obeys_immutable_cas_and_cancellation(chain, monkeypatch, concurrent):
    job, target, source = existing_model_job(chain)
    before_calls, before_generated = len(model_calls(chain)), len(chain.generated)
    prior_review = reconstruction._model_review_invoke
    expected_head = job['baseRevisionId']
    def review(payload):
        nonlocal expected_head
        if concurrent == 'edit':
            expected_head = user_edit(chain, job['baseRevisionId'])['revision']['id']
        elif concurrent == 'cancel':
            chain.repo.cancel_job(job['id'], chain.cap)
        return prior_review(payload)
    monkeypatch.setattr(reconstruction, '_model_review_invoke', review)
    if concurrent == 'stale_before_review':
        expected_head = user_edit(chain, job['baseRevisionId'])['revision']['id']
    finished = run_job(chain.repo, chain.blobs, job['id'])
    assert len(chain.generated) == before_generated
    assert len(model_calls(chain)) == before_calls + (0 if concurrent == 'stale_before_review' else 1)
    assert chain.repo.get_revision(job['baseRevisionId'])['document'] == source
    current = chain.repo.get_project(job['projectId'])['revision']
    if concurrent == 'cancel':
        assert finished['status'] == 'cancelled' and finished['lateResultSaved']
        assert current['id'] == expected_head and finished.get('resultRevisionId') is None
    else:
        result = chain.repo.get_revision(finished['resultRevisionId'])['document']
        reviewed = next(entity for entity in result['entities'] if entity['id'] == target['id'])
        assert reviewed['activeModelRepresentationId'] == target['activeModelRepresentationId']
        assert reviewed['currentModelTransform'] == target['currentModelTransform']
        old = next(rep for rep in target['representations'] if rep['id'] == target['activeModelRepresentationId'])
        saved = next(rep for rep in reviewed['representations'] if rep['id'] == reviewed['activeModelRepresentationId'])
        assert all(saved[key] == value for key,value in old.items())
        if concurrent == 'stale_before_review':
            assert finished['result']['reviews'][0]['status'] == 'needs_information'
            assert finished['result']['newModelCalls'] == 0
        else:
            assert finished['result']['reviews'][0]['status'] == 'accepted'
            assert next(row for row in audit_correspondence(result)['rows'] if row['entityId'] == target['id'])['qualityCurrent']
        assert finished['headAdvanced'] is (concurrent is None)
        assert current['id'] == (finished['resultRevisionId'] if concurrent is None else expected_head)
    calls = len(model_calls(chain))
    assert run_job(chain.repo, chain.blobs, job['id'])['id'] == job['id']
    assert len(model_calls(chain)) == calls


def test_new_review_job_cannot_repeat_an_unknown_call_even_after_late_receipt(chain, monkeypatch):
    job, _, _ = existing_model_job(chain)
    dispatched = []
    def lost(payload):
        dispatched.append(payload)
        raise RuntimeError('Synthetic response lost after dispatch')
    monkeypatch.setattr(reconstruction, '_model_review_invoke', lost)
    first = run_job(chain.repo, chain.blobs, job['id'])
    assert first['result']['stoppedReason'] == 'provider_outcome_unknown'
    unknown = model_calls(chain)[-1]
    assert unknown['status'] == 'outcome_unknown'
    head = first['resultRevisionId']
    for late in (False, True):
        if late:
            receipt = chain.repo.complete_model_call(str(unknown['id']), 'succeeded', actual_cost=0,
                response={'providerRequestId':'synthetic-late-receipt'})
            assert receipt['status'] == 'outcome_unknown'
        next_job = chain.repo.create_job(job['projectId'], chain.cap, {'requestId':identity(),
            'branchId':job['branchId'], 'baseRevisionId':head, 'kind':'review_models', 'inputs':job['inputs']})
        outcome = run_job(chain.repo, chain.blobs, next_job['id'])
        assert outcome['result']['stoppedReason'] == 'provider_outcome_unknown'
        assert outcome['result']['newModelCalls'] == 0 and len(dispatched) == 1
        assert model_calls(chain)[-1]['id'] == unknown['id']
        head = outcome['resultRevisionId']


def test_rejected_review_cache_older_than_asset_listing_is_reused_without_dispatch(chain, monkeypatch):
    job, _, _ = existing_model_job(chain)
    dispatched = []
    def reject(payload):
        dispatched.append(payload)
        return {'review':{'status':'fail', 'reason':'Synthetic rejected shape',
            'observationIds':[view['observationId'] for view in payload['views']],
            'visibleShapeIssues':['Wrong topology'], 'nextAction':'alternate_view'}}
    monkeypatch.setattr(reconstruction, '_model_review_invoke', reject)
    first = run_job(chain.repo, chain.blobs, job['id'])
    assert first['result']['reviews'][0]['status'] == 'rejected'
    cached = first['result']['stages'][0]
    # Real immutable filler assets, inserted in this fixture's isolated schema.
    with chain.repo._connect() as connection:
        for index in range(501):
            blob = chain.blobs.put(f'newer fixture asset {index}'.encode(), 'application/octet-stream')
            chain.repo._register_asset(connection, job['projectId'], blob)
    latest = chain.repo.list_project_records(job['projectId'], 'assets')['items']
    assert len(latest) == 500 and cached['assetId'] not in {asset['id'] for asset in latest}
    calls = len(model_calls(chain))
    second = chain.repo.create_job(job['projectId'], chain.cap, {'requestId':identity(),
        'branchId':job['branchId'], 'baseRevisionId':first['resultRevisionId'], 'kind':'review_models', 'inputs':job['inputs']})
    replay = run_job(chain.repo, chain.blobs, second['id'])
    assert replay['result']['reviews'][0]['status'] == 'rejected'
    assert replay['result']['stages'][0]['status'] == 'cached'
    assert replay['result']['newModelCalls'] == 0 and len(dispatched) == 1
    assert len(model_calls(chain)) == calls
