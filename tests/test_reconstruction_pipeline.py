from copy import deepcopy
import json
import sys
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from ehs_spatial.platform.contracts import PlatformError, digest
from ehs_spatial.platform.recgen import RECGEN_PINS
from ehs_spatial.platform.reconstruction import run_research_job
from ehs_spatial.platform.reconstruction_pipeline import run_reconstruction_pipeline
from test_platform_reconstruction import provider
from test_platform_research_inputs import source


@pytest.fixture
def pipeline(source):
    repo, blobs, job, entity, manifest, *_ = source
    job.update(kind='reconstruct_scene', inputs={'phase': 'prepare', 'entityIds': [entity['id']], 'processed': []})
    job['config']['providerManifest'] = manifest
    return repo, blobs, job, entity


def researched(pipeline, monkeypatch):
    repo, blobs, prepare, entity = pipeline
    _, prepared, continuation = run_reconstruction_pipeline(repo, blobs, prepare)
    research = {**prepare, 'id': str(uuid4()), 'kind': continuation['kind'], 'inputs': continuation['inputs'],
                'config': {**deepcopy(prepare['config']), **continuation['config']}, 'status': 'running'}
    calls = []
    vertices = np.array([[-1., -2 / 3, 2], [1., -2 / 3, 2], [1., 2 / 3, 2], [-1., 2 / 3, 2]])
    def invoke(value, config, *, is_current):
        calls.append(value)
        return {'vertices': vertices, 'faces': np.array([[0, 1, 2], [0, 2, 3]]),
                'colors': np.ones((4, 4)), 'objectToCamera': np.eye(4),
                'officialPosedVertices': vertices.copy(), 'pins': RECGEN_PINS,
                'telemetry': {'actualCostUsd': 0}, 'providerRequestId': 'fake-no-network'}
    monkeypatch.setitem(sys.modules, 'ehs_spatial.platform.recgen_transport', SimpleNamespace(invoke=invoke))
    repo.paid_budget = 10
    research['result'] = run_research_job(repo, blobs, research)
    research['status'] = research['result']['status']
    monkeypatch.setattr(repo, 'get_job', lambda identity: deepcopy(research) if identity == research['id'] else None, raising=False)
    attach = {**prepare, 'id': str(uuid4()), 'inputs': {**continuation['config']['pipeline'], 'researchJobId': research['id']}}
    return research, attach, calls


def test_prepare_only_registers_frozen_input_and_preserves_full_provider_config(pipeline, monkeypatch):
    repo, blobs, job, entity = pipeline
    job['config']['providerManifest']['model_review'] = {'sentinel': 'retained for later worker'}
    before = deepcopy((repo.document, job, repo.calls))
    monkeypatch.setattr(repo, 'reserve_model_call', lambda *args, **kwargs: pytest.fail('Preparation reserved a model call'))
    document, result, child = run_reconstruction_pipeline(repo, blobs, job)
    assert document is None and result['newModelCalls'] == 0
    assert child['kind'] == 'validate_model' and 'providerManifest' not in child['config']
    assert child['config']['pipeline'] == {'phase': 'attach', 'entityId': entity['id'],
        'entityIds': [entity['id']], 'processed': [], 'preparationJobId': job['id']}
    asset = repo.get_asset(child['inputs']['validationAssetId'])
    frozen = json.loads(blobs.get(asset['storageKey'], asset['sha256'], asset['sizeBytes']))
    assert asset['sha256'] == child['inputs']['validationSha256'] == digest(frozen)
    assert frozen['baseRevisionId'] == job['baseRevisionId']
    assert child['config']['researchProtocolSha256'] == digest(frozen['protocol'])
    assert (repo.document, job, repo.calls) == before


def test_prepare_retains_reference_and_missing_evidence_then_selects_next_entity(pipeline):
    repo, blobs, job, entity = pipeline
    floor = {**deepcopy(entity), 'id': str(uuid4()), 'geometryRole': 'floor', 'observationRefs': []}
    missing = {**deepcopy(entity), 'id': str(uuid4()), 'observationRefs': []}
    repo.document['entities'].extend((floor, missing))
    job['inputs']['entityIds'] = [floor['id'], missing['id'], entity['id']]
    _, result, child = run_reconstruction_pipeline(repo, blobs, job)
    assert result['processed'] == [
        {'entityId': floor['id'], 'status': 'retained', 'reason': 'generation_reference_surface'},
        {'entityId': missing['id'], 'status': 'needs_information', 'reason': 'generation_mask_required'}]
    assert child['config']['pipeline']['entityIds'] == [entity['id']]
    assert child['config']['pipeline']['processed'] == result['processed']


@pytest.mark.parametrize('review', ['pass', 'fail', None])
def test_attach_consumes_cached_generation_and_activates_only_real_quality_acceptance(pipeline, monkeypatch, review):
    repo, blobs, job, entity = pipeline
    research, attach, generated = researched(pipeline, monkeypatch)
    before = deepcopy(repo.document)
    reviewed = []
    def inspect(payload):
        reviewed.append(payload)
        return {'review': {'status': review, 'reason': 'test actual review output',
            'observationIds': [view['observationId'] for view in payload['views']],
            'visibleShapeIssues': [] if review == 'pass' else ['wrong geometry'], 'nextAction': 'none'}}
    providers = {'generation': provider('generation', lambda _: pytest.fail('Attachment invoked generation'))}
    if review is not None:
        providers['model_review'] = provider('model_review', inspect)
    document, result, continuation = run_reconstruction_pipeline(repo, blobs, attach, providers)
    assert repo.document == before and len(generated) == 1
    assert continuation is None and result['status'] == 'incomplete'
    assert result['cadValidationStatus'] == 'incomplete'
    assert result['cadPendingEntityIds'] == [entity['id']]
    assert result['correspondence']['cadValidationMethod'] == 'byte-verified-indexed-mesh-projection-v1'
    target = next(e for e in document['entities'] if e['id'] == entity['id'])
    model = next(rep for rep in target['representations'] if rep['kind'] == 'generated_mesh')
    assert model['placementState'] == 'unconfirmed' and model['shapeStatus'] == 'research_only'
    assert model['provenance']['licenseScope'] == 'noncommercial_research'
    assert model['qualityEvidence']['geometric']['status'] == 'observed_consistent'
    assert bool(target['activeModelRepresentationId']) is (review == 'pass')
    assert result['processed'][0]['status'] == {'pass': 'accepted', 'fail': 'rejected', None: 'needs_information'}[review]
    assert result['newModelCalls'] == len(reviewed)
    assert result['placementConfirmedEntityIds'] == []


def test_attach_unknown_research_outcome_never_continues(pipeline, monkeypatch):
    repo, blobs, _, _ = pipeline
    research, attach, generated = researched(pipeline, monkeypatch)
    research['status'] = 'outcome_unknown'
    document, result, child = run_reconstruction_pipeline(repo, blobs, attach)
    assert document is None and child is None and result['status'] == 'outcome_unknown'
    assert len(generated) == 1


def test_attach_initializes_only_configured_model_review_factory(pipeline, monkeypatch):
    from ehs_spatial.platform import reconstruction_pipeline
    repo, blobs, job, entity = pipeline
    review_config = {'provider': 'configured-reviewer'}
    job['config']['providerManifest']['model_review'] = review_config
    research, attach, generated = researched(pipeline, monkeypatch)
    manifests = []
    def factory(manifest):
        manifests.append(manifest)
        return {'model_review': provider('model_review', lambda value: {'review': {
            'status': 'pass', 'reason': 'reviewed matching source render',
            'observationIds': [view['observationId'] for view in value['views']],
            'visibleShapeIssues': [], 'nextAction': 'none'}})}
    monkeypatch.setattr(reconstruction_pipeline, 'providers_from_manifest', factory)
    document, result, child = run_reconstruction_pipeline(repo, blobs, attach)
    assert manifests == [{'model_review': review_config}]
    assert result['processed'][0]['status'] == 'accepted' and len(generated) == 1
    target = next(e for e in document['entities'] if e['id'] == entity['id'])
    model = next(rep for rep in target['representations'] if rep['id'] == target['activeModelRepresentationId'])
    assert {ref['observationId'] for ref in model['sourceRefs'] if 'observationId' in ref} == set(entity['observationRefs'])


@pytest.mark.parametrize('mismatch', ['branch', 'base', 'entity', 'document', 'cache'])
def test_attach_rejects_context_source_or_output_cache_tampering(pipeline, monkeypatch, mismatch):
    repo, blobs, _, entity = pipeline
    research, attach, generated = researched(pipeline, monkeypatch)
    if mismatch == 'branch': research['branchId'] = str(uuid4())
    if mismatch == 'base': research['baseRevisionId'] = str(uuid4())
    if mismatch == 'entity': attach['inputs']['entityId'] = str(uuid4())
    if mismatch == 'document': repo.document['observations'][0]['revision'] += 1
    if mismatch == 'cache':
        output = next(a for a in repo.assets if a['id'] == research['result']['outputAssetId'])
        output['metadata']['cacheKey'] = 'f' * 64
    monkeypatch.setattr(repo, 'reserve_model_call', lambda *args, **kwargs: pytest.fail('Invalid attachment invoked a model'))
    with pytest.raises(PlatformError):
        run_reconstruction_pipeline(repo, blobs, attach)
    assert len(generated) == 1


def test_attach_enqueues_next_target_without_reusing_current_input(pipeline, monkeypatch):
    repo, blobs, job, entity = pipeline
    another = {**deepcopy(entity), 'id': str(uuid4()), 'observationRefs': []}
    repo.document['entities'].append(another)
    job['inputs']['entityIds'].append(another['id'])
    research, attach, generated = researched(pipeline, monkeypatch)
    document, result, child = run_reconstruction_pipeline(repo, blobs, attach)
    assert child == {'kind': 'reconstruct_scene', 'inputs': {'phase': 'prepare', 'entityIds': [another['id']],
        'processed': result['processed']}, 'config': {}}
    assert child['inputs']['processed'][0]['entityId'] == entity['id']
    assert document is not None and len(generated) == 1


def test_unknown_review_outcome_keeps_candidate_and_stops_next_generation(pipeline, monkeypatch):
    repo, blobs, job, entity = pipeline
    another = {**deepcopy(entity), 'id': str(uuid4()), 'observationRefs': []}
    repo.document['entities'].append(another)
    job['inputs']['entityIds'].append(another['id'])
    _, attach, generated = researched(pipeline, monkeypatch)
    def unknown(_):
        raise TimeoutError('no observed terminal response')
    document, result, child = run_reconstruction_pipeline(repo, blobs, attach,
        {'model_review': provider('model_review', unknown)})
    assert document is not None and child is None and len(generated) == 1
    assert result['stoppedReason'] == 'provider_outcome_unknown'
    assert result['remainingEntityIds'] == [another['id']]
    target = next(e for e in document['entities'] if e['id'] == entity['id'])
    assert target['activeModelRepresentationId'] is None
    assert any(rep['kind'] == 'generated_mesh' for rep in target['representations'])


def test_unavailable_quality_geometry_keeps_output_reference_and_next_eligible_target(pipeline, monkeypatch):
    from ehs_spatial.platform import reconstruction_pipeline
    repo, blobs, job, entity = pipeline
    another = {**deepcopy(entity), 'id': str(uuid4()), 'observationRefs': []}
    repo.document['entities'].append(another)
    job['inputs']['entityIds'].append(another['id'])
    research, attach, generated = researched(pipeline, monkeypatch)
    def missing(*_):
        raise PlatformError('geometry_evidence_unavailable', 409)
    monkeypatch.setattr(reconstruction_pipeline, '_load_geometry', missing)
    document, result, child = run_reconstruction_pipeline(repo, blobs, attach)
    assert result['processed'][0]['status'] == 'needs_information'
    assert result['processed'][0]['reason'] == 'geometry_evidence_unavailable'
    assert result['processed'][0]['candidateRef']['assetId'] == research['result']['outputAssetId']
    assert any(asset['id'] == research['result']['outputAssetId'] for asset in document['assets'])
    assert child['inputs']['entityIds'] == [another['id']] and len(generated) == 1
