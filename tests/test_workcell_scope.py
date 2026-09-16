"""Scope sanity check: source-photo membership, never generated-mesh distance."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ehs_spatial.platform.contracts import PlatformError
from ehs_spatial.platform.reconstruction import _review_workcell_scope, run_capture_pipeline
from ehs_spatial.platform.storage import LocalBlobStore
from test_platform_identity import source_scene
from test_platform_reconstruction import Repo, inventory_review_response, provider
from test_capture_pipeline import providers_for_new_capture


def scope_case(relation='outside', *, boundary=True, target=True, tamper=False):
    document = source_scene()
    images = [{'id': 'image-'+str(i), 'sha256': str(i)*64, 'mediaType': 'image/png',
               'bytes': b'synthetic-image', 'width': 20, 'height': 20} for i in (1, 2)]
    def call(stage, given, payload):
        assert stage == 'model_review' and payload['mode'] == 'workcell_scope'
        response = inventory_review_response(payload)
        review = response['review'];review['targetEstablished'] = target
        for entity in review['entities']:
            for view in entity['views']:
                view.update(relation=relation if entity['entityId'] == 'entity-1' else 'inside',
                            boundaryEvidence='Visible neighboring enclosure beyond target fence' if boundary else '')
        if tamper: review['inputSha256'] = 'f'*64
        return response, {'id': 'scope-evidence', 'sha256': 'e'*64}
    return document, images, SimpleNamespace(call=call)


def test_outside_preserves_sources_models_and_every_in_scope_object():
    document, images, stages = scope_case()
    before = deepcopy(document)
    result = _review_workcell_scope(document, images, stages)
    assert result['excludedEntityIds'] == ['entity-1']
    assert document['entities'][0]['sourceContext'] and not document['entities'][0]['visible']
    assert not document['entities'][1]['sourceContext']
    assert document['observations'] == before['observations']
    assert [e['representations'] for e in document['entities']] == [e['representations'] for e in before['entities']]
    decision = document['entities'][0]['workcellScopeDecision']
    assert decision['source'] == 'model_inference' and decision['sourceRefs'][0]['assetId'] == 'scope-evidence'
    assert decision['observationRefs'] == [{'observationId': 'observation-1', 'revision': 1}]


@pytest.mark.parametrize('case', ['unknown', 'no_boundary', 'no_target', 'mixed_views', 'missing_photo', 'tampered', 'manual_include'])
def test_uncertainty_conflicts_and_invalid_proofs_never_hide_objects(case):
    document, images, stages = scope_case(relation='unknown' if case == 'unknown' else 'outside',
        boundary=case != 'no_boundary', target=case != 'no_target', tamper=case == 'tampered')
    if case == 'mixed_views':
        other = deepcopy(document['observations'][1]);other['id'] = 'observation-3'
        document['observations'].append(other);document['entities'][0]['observationRefs'].append(other['id'])
        original_call = stages.call
        def mixed(*args):
            response, evidence = original_call(*args)
            response['review']['entities'][0]['views'][1]['relation'] = 'inside'
            return response, evidence
        stages.call = mixed
    if case == 'missing_photo': images = images[1:]
    if case == 'manual_include': document['entities'][0]['workcellScopeDecision'] = {'included': True, 'source': 'manual_assertion'}
    if case in ('no_boundary', 'tampered'):
        with pytest.raises(PlatformError, match='workcell_scope_review_invalid'):
            _review_workcell_scope(document, images, stages)
    else:
        result = _review_workcell_scope(document, images, stages)
        assert not result['excludedEntityIds']
    assert not document['entities'][0].get('sourceContext')


def test_new_capture_checks_scope_before_generation_and_reuses_cached_review(tmp_path):
    blobs = LocalBlobStore(tmp_path);repo = Repo(blobs)
    repo.capture['target'] = repo.document['target'] = 'scene'
    providers = providers_for_new_capture(repo)
    original = providers['model_review'].invoke
    generated = []
    def review(payload):
        result = original(payload)
        if payload.get('mode') == 'workcell_scope':
            for entity in result['review']['entities']:
                for view in entity['views']:
                    view.update(relation='outside', boundaryEvidence='Behind separate neighboring enclosure')
        return result
    providers['model_review'] = provider('model_review', review)
    providers['generation'] = provider('generation', lambda request: generated.append(request))
    document, result = run_capture_pipeline(repo, blobs, repo.job, providers)
    assert result['analysis']['workcellScope']['excludedEntityIds']
    assert not generated and result['generation']['status'] == 'not_requested'
    assert document['observations'] and all(e.get('sourceContext') for e in document['entities'])
    count = len(repo.calls)
    run_capture_pipeline(repo, blobs, repo.job, providers)
    assert len(repo.calls) == count


def test_scope_transport_compacts_ids_and_restores_exact_business_identity(monkeypatch):
    from ehs_spatial.platform.reconstruction import _workcell_scope_input, _model_review_invoke, _WorkcellScopeResponse
    from ehs_spatial.providers import gemini
    document, images, _ = scope_case()
    payload = _workcell_scope_input(document, images)
    class Adapter:
        def create_bounded_structured(self, *a, **kwargs):
            text = str(kwargs['input'])
            assert 'E0' in text and 'O0' in text and 'I0' in text
            assert 'entity-1' not in text and 'observation-1' not in text
            return SimpleNamespace(usage=None, id='fixture-call')
        def _parse(self, *args):
            return _WorkcellScopeResponse(inputSha256=payload['inputSha256'], targetEstablished=True,
                targetDescription='Fixture', entities=[{'entityId': 'E0', 'views': [
                    {'observationId': 'O0', 'relation': 'inside', 'evidence': 'fixture', 'boundaryEvidence': ''}]}]), 'fixture-call'
    monkeypatch.setattr(gemini, 'GeminiAdapter', Adapter)
    result = _model_review_invoke(payload)
    assert result['review']['entities'][0]['entityId'] == 'entity-1'
    assert result['review']['entities'][0]['views'][0]['observationId'] == 'observation-1'
