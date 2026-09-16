"""Retained-model review uses real CPU geometry and fake, ledger-backed reviewers."""
from copy import deepcopy
import io
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from PIL import Image
import pytest

from ehs_spatial.platform import reconstruction as reconstruction
from ehs_spatial.platform.contracts import PlatformError, validate_document
from ehs_spatial.platform.correspondence import audit_correspondence
from test_platform_reconstruction import provider
from test_platform_research_inputs import source


def add_model(case, entity, left=-1., right=1.):
    response = {'vertices':np.array([[left,-2/3,2.],[right,-2/3,2.],[right,2/3,2.],[left,2/3,2.]]),
        'faces':np.array([[0,1,2],[0,2,3]]), 'proposedObjectToNative':np.eye(4)}
    stages = reconstruction._Stages(case.repo, case.blobs, case.job, {})
    evidence = stages.put(response, {'kind':'fixture_model'})
    observations = [o for o in case.repo.document['observations'] if o['id'] in entity['observationRefs']]
    frame = case.repo.document['cameras'][0]['coordinateFrameId']
    return reconstruction._record_generated_representation(case.repo.document, stages, entity,
        observations, frame, response, evidence, activate=True)


@pytest.fixture
def retained(source):
    repo, blobs, job, entity, *_ = source
    case = SimpleNamespace(repo=repo, blobs=blobs, entity=entity, calls=[],
        job={**job, 'kind':'review_models', 'inputs':{'entityIds':[entity['id']]}})
    case.rep = add_model(case, entity)
    def review(payload):
        case.calls.append(deepcopy(payload))
        return {'review':{'status':'pass', 'reason':'Synthetic plane agrees with each owned source view.',
            'observationIds':[view['observationId'] for view in payload['views']],
            'visibleShapeIssues':[], 'nextAction':'none'}}
    case.providers = {'model_review':provider('model_review', review),
                      'generation':provider('generation', lambda _:pytest.fail('Review cannot generate'))}
    return case


def run(case, **kwargs):
    return reconstruction.run_model_review(case.repo, case.blobs, case.job, case.providers, **kwargs)


def without_quality(document):
    document = deepcopy(document)
    document.pop('assets')
    for entity in document['entities']:
        for rep in entity.get('representations', []):
            for key in ('qualityEvidence', 'qualityBinding', 'qualityReviewRefs'):
                rep.pop(key, None)
    return document


def test_existing_model_review_preserves_bytes_pose_sources_and_reuses_current_proof(retained):
    case = retained
    before = deepcopy(case.repo.document)
    original_asset = case.repo.get_asset(case.rep['assetId'])
    raw = case.blobs.get(original_asset['storageKey'], original_asset['sha256'], original_asset['sizeBytes'])
    document, result = run(case)
    assert result['reviews'][0]['status'] == 'accepted', result
    assert result['acceptedEntityIds'] == [case.entity['id']]
    assert result['newModelCalls'] == len(case.calls) == 1
    assert result['status'] == 'incomplete' and result['cadPendingEntityIds'], 'Shape proof cannot certify absent CAD'
    assert result['placementConfirmedEntityIds'] == []
    assert without_quality(document) == without_quality(before)
    assert case.repo.document == before
    assert case.blobs.get(original_asset['storageKey'], original_asset['sha256'], original_asset['sizeBytes']) == raw
    validate_document(document)
    assert audit_correspondence(document)['rows'][0]['qualityCurrent']
    assert {view['observationId'] for view in case.calls[0]['views']} == set(case.entity['observationRefs'])
    case.repo.document = document
    again, replay = run(case)
    assert replay['reviews'][0]['reusedCurrentQuality'] and replay['newModelCalls'] == 0
    assert len(case.calls) == 1 and again == document


@pytest.mark.parametrize('verdict', ['fail', None])
def test_failed_or_unconfigured_review_stays_explicit_and_idempotent(retained, verdict):
    case = retained
    if verdict is None:
        case.providers = {}
    else:
        def reject(payload):
            case.calls.append(payload)
            return {'review':{'status':'fail', 'reason':'Fixture shape rejection',
                'observationIds':[view['observationId'] for view in payload['views']],
                'visibleShapeIssues':['Wrong fixture topology'], 'nextAction':'alternate_view'}}
        case.providers['model_review'] = provider('model_review', reject)
    document, result = run(case)
    assert result['reviews'][0]['status'] == ('rejected' if verdict else 'needs_information')
    assert not audit_correspondence(document)['rows'][0]['qualityCurrent']
    assert without_quality(document) == without_quality(case.repo.document)
    case.repo.document = document
    again, replay = run(case)
    assert replay['newModelCalls'] == 0 and again == document
    assert len(case.calls) == (1 if verdict else 0)


@pytest.mark.parametrize('ids', [None, [], ['foreign'], ['duplicate', 'duplicate'], [1]])
def test_targets_are_preflighted_before_factory_or_artifact_writes(retained, monkeypatch, ids):
    case = retained
    monkeypatch.setattr(reconstruction, 'providers_from_manifest', lambda _:pytest.fail('Factory before preflight'))
    before = deepcopy(case.repo.assets)
    with pytest.raises(PlatformError):
        reconstruction.run_model_review(case.repo, case.blobs,
            {**case.job, 'inputs':{'entityIds':ids}}, providers=None)
    assert case.repo.assets == before and not case.calls


def test_factory_only_constructs_review_provider(retained, monkeypatch):
    case, seen = retained, []
    def factory(manifest):
        seen.append(manifest)
        return {'model_review':case.providers['model_review']}
    monkeypatch.setattr(reconstruction, 'providers_from_manifest', factory)
    job = {**case.job, 'config':{'providerManifest':{'generation':{'must':'not load'}, 'model_review':{'frozen':'review'}}}}
    _, result = reconstruction.run_model_review(case.repo, case.blobs, job)
    assert seen == [{'model_review':{'frozen':'review'}}] and result['newModelCalls'] == 1


@pytest.mark.parametrize('missing', ['model', 'stale', 'mask', 'frame'])
def test_missing_stale_and_mixed_frame_inputs_never_review(retained, missing):
    case = retained
    if missing == 'model':
        case.entity['activeModelRepresentationId'] = None
    elif missing == 'stale':
        case.rep['sourceValidity'] = 'stale'
    elif missing == 'mask':
        case.repo.document['observations'][0]['maskAssetId'] = None
    else:
        case.repo.document['cameras'][0]['coordinateFrameId'] = 'unregistered'
    document, result = run(case)
    assert result['reviews'][0]['status'] == 'needs_information'
    assert not result['acceptedEntityIds'] and result['newModelCalls'] == 0 and not case.calls
    assert without_quality(document) == without_quality(case.repo.document)


def make_family(case):
    """Parent owns whole-panel masks; child owns its separate right-half masks."""
    document = case.repo.document
    child = {**deepcopy(case.entity), 'id':str(uuid4()), 'label':'right component',
        'representations':[], 'activeModelRepresentationId':None, 'currentModelTransform':None,
        'observationRefs':[], 'measurements':{}, 'measurementEvidence':[], 'measurementSelections':{}}
    mask = np.zeros((24,32), np.uint8)
    mask[4:20,16:28] = 255
    encoded = io.BytesIO()
    Image.fromarray(mask).save(encoded, format='PNG')
    stages = reconstruction._Stages(case.repo, case.blobs, case.job, {})
    asset = stages.put(encoded.getvalue(), {'kind':'fixture_mask'}, 'image/png')
    reconstruction._include(document, asset)
    for original in list(document['observations']):
        observation = {**deepcopy(original), 'id':str(uuid4()), 'maskAssetId':asset['id'], 'maskComplete':True}
        observation.pop('maskEvidence', None)
        document['observations'].append(observation)
        child['observationRefs'].append(observation['id'])
    document['entities'].append(child)
    case.entity['representations'] = []
    case.entity['activeModelRepresentationId'] = case.entity['currentModelTransform'] = None
    case.rep = add_model(case, case.entity, right=0.)
    child_rep = add_model(case, child, left=0.)
    child.update(parentEntityId=case.entity['id'], partRelation={'source':'manual', 'baseRevisionId':case.job['baseRevisionId'],
        'reason':'Synthetic exact panel partition', 'evidenceRefs':[
            {'observationId':case.entity['observationRefs'][0], 'revision':1},
            {'observationId':child['observationRefs'][0], 'revision':1}]})
    case.job['inputs']['entityIds'].append(child['id'])
    return child, child_rep


def test_parent_review_uses_family_union_and_child_uses_only_owned_masks(retained):
    case = retained
    child, _ = make_family(case)
    document, result = run(case)
    assert [row['status'] for row in result['reviews']] == ['accepted','accepted'], result
    parent_quality = document['entities'][0]['representations'][-1]['qualityEvidence']
    assert parent_quality['geometric']['assessmentScope'] == 'parent_family'
    assert len(parent_quality['familyBinding']) == 2
    assert [set(view['observationId'] for view in call['views']) for call in case.calls] == [
        set(case.entity['observationRefs']), set(child['observationRefs'])]
    assert all(row['qualityCurrent'] for row in audit_correspondence(document)['rows'])
    assert without_quality(document) == without_quality(case.repo.document)
    case.repo.document = document
    _, again = run(case)
    assert all(row['reusedCurrentQuality'] for row in again['reviews']) and len(case.calls) == 2


def test_missing_family_member_cannot_be_replaced_with_sibling_geometry(retained):
    case = retained
    child, _ = make_family(case)
    child['activeModelRepresentationId'] = None
    _, result = run(case)
    assert all(row['status'] == 'needs_information' for row in result['reviews'])
    assert result['reviews'][0]['reason'] == 'quality_family_model_unavailable'
    assert result['newModelCalls'] == 0 and not case.calls


def test_wrong_saved_pose_is_assessed_without_refinement_or_replacement(retained, monkeypatch):
    from ehs_spatial.platform import model_quality
    case = retained
    case.entity['currentModelTransform']['position'][2] = 1.
    before = deepcopy(case.repo.document)
    monkeypatch.setattr(model_quality, 'refine_model_pose', lambda *_args, **_kwargs:pytest.fail('Review must preserve saved pose'))
    document, result = run(case)
    assert result['reviews'][0]['status'] == 'rejected'
    assert without_quality(document) == without_quality(before)


def test_agent_can_queue_only_explicit_owned_review_targets(retained, monkeypatch):
    from ehs_spatial.platform.agent_service import AgentService
    case, created = retained, []
    def create(pid, capability, body):
        created.append(body)
        return {'id':'review-job', 'status':'pending_dispatch'}
    monkeypatch.setattr(case.repo, 'create_job', create, raising=False)
    service = AgentService(case.repo, None)
    turn = {**case.job, 'request':{}, 'branchId':'branch'}
    result = service._tool('start_job', {'kind':'review_models', 'entityIds':[case.entity['id']]},
        case.repo.document, turn, 'capability', 0)
    assert result['jobId'] == 'review-job' and created[0]['kind'] == 'review_models'
    for ids in ([], ['foreign'], [case.entity['id'], case.entity['id']]):
        with pytest.raises(PlatformError):
            service._tool('start_job', {'kind':'review_models', 'entityIds':ids}, case.repo.document, turn, 'capability', 0)
    assert len(created) == 1


@pytest.mark.parametrize('changed', ['pose', 'asset_hash', 'asset_layout', 'primitive', 'relation', 'source', 'observation', 'member'])
def test_child_edit_invalidates_parent_family_proof(retained, changed):
    case = retained
    make_family(case)
    document, result = run(case)
    assert len(result['acceptedEntityIds']) == 2, result
    parent, child = [entity for entity in document['entities'] if not entity.get('sourceContext')]
    rep = child['representations'][-1]
    if changed == 'pose':
        child['currentModelTransform']['position'][0] += .1
    elif changed == 'asset_hash':
        next(a for a in document['assets'] if a['id'] == rep['assetId'])['sha256'] = '0' * 64
    elif changed == 'asset_layout':
        next(a for a in document['assets'] if a['id'] == rep['assetId'])['byteLayout']['indexCount'] = 3
    elif changed == 'primitive':
        rep['primitive'] = {'type':'cuboid', 'dimensions':[1,1,1]}
    elif changed == 'relation':
        child['partRelation']['reason'] = 'Changed reviewed relationship'
    elif changed == 'source':
        rep['sourceRefs'].append({'assetId':rep['assetId']})
    elif changed == 'observation':
        next(o for o in document['observations'] if o['id'] == child['observationRefs'][0])['maskComplete'] = False
    else:
        child['parentEntityId'] = None
        child['partRelation'] = None
    assert not next(row for row in audit_correspondence(document)['rows'] if row['entityId'] == parent['id'])['qualityCurrent']


@pytest.mark.parametrize('family', [False, True])
def test_changed_mesh_decoding_is_reassessed_instead_of_reusing_accepted_quality(retained, family):
    case = retained
    if family:
        child, _ = make_family(case)
    document, _ = run(case)
    target = next(entity for entity in document['entities'] if entity['id'] == (child['id'] if family else case.entity['id']))
    rep = target['representations'][-1]
    next(a for a in document['assets'] if a['id'] == rep['assetId'])['byteLayout']['indexCount'] = 3
    validate_document(document)
    assert not audit_correspondence(document)['rows'][0]['qualityCurrent']
    case.repo.document = document
    before = len(case.calls)
    _, result = run(case)
    assert not result['reviews'][0].get('reusedCurrentQuality')
    assert result['reviews'][0]['status'] == 'rejected' and len(case.calls) > before


@pytest.mark.parametrize('change', ['provenance', 'duplicate_owner'])
def test_parent_cannot_reuse_quality_when_child_sources_are_no_longer_current(retained, change):
    case = retained
    child, _ = make_family(case)
    document, _ = run(case)
    child = next(entity for entity in document['entities'] if entity['id'] == child['id'])
    if change == 'provenance':
        child['representations'][-1]['provenance'] = {'sourceRefs':[
            {'observationId':child['observationRefs'][0], 'revision':99}]}
    else:
        document['entities'].append({'id':'another-owner', 'representations':[],
            'observationRefs':[child['observationRefs'][0]]})
    assert not audit_correspondence(document)['rows'][0]['qualityCurrent']
    case.repo.document = document
    case.job['inputs']['entityIds'] = [case.entity['id']]
    count = len(case.calls)
    _, result = run(case)
    assert result['reviews'][0]['status'] == 'needs_information' and len(case.calls) == count


def test_unknown_review_stops_remaining_models_and_replay_cannot_repeat_dispatch(retained):
    case = retained
    make_family(case)
    def unknown(payload):
        case.calls.append(payload)
        raise RuntimeError('Synthetic response lost after dispatch')
    case.providers['model_review'] = provider('model_review', unknown)
    document, result = run(case)
    assert result['stoppedReason'] == 'provider_outcome_unknown'
    assert result['reviews'][1]['reason'] == 'review_stopped_after_unknown_outcome'
    assert len(case.calls) == result['newModelCalls'] == 1
    case.repo.document = document
    _, repeated = run(case)
    assert repeated['stoppedReason'] == 'provider_outcome_unknown' and len(case.calls) == 1


def test_capture_pipeline_reviews_retained_models_without_generation(retained, monkeypatch):
    case = retained
    monkeypatch.setattr(reconstruction, 'run_analysis', lambda *_:(deepcopy(case.repo.document), {'status':'succeeded','stages':[],'errors':[]}))
    monkeypatch.setattr(reconstruction, 'run_generation', lambda *_args, **_kwargs:pytest.fail('Existing model must not regenerate'))
    document, result = reconstruction.run_capture_pipeline(case.repo, case.blobs, case.repo.job, case.providers)
    assert result['review']['acceptedEntityIds'] == [case.entity['id']]
    assert len(case.calls) == 1 and result['qualityPendingEntityIds'] == []
    case.repo.document = document
    _, repeated = reconstruction.run_capture_pipeline(case.repo, case.blobs, case.repo.job, case.providers)
    assert repeated['review']['status'] == 'not_requested' and len(case.calls) == 1


@pytest.mark.parametrize('unknown', [False, True])
def test_direct_research_preparation_reviews_retained_models_before_next_revision(retained, unknown):
    from ehs_spatial.platform.reconstruction_pipeline import run_reconstruction_pipeline
    case = retained
    make_family(case)
    if unknown:
        def lost(_):
            raise RuntimeError('Synthetic unknown review outcome')
        case.providers['model_review'] = provider('model_review', lost)
    job = {**case.job, 'kind':'reconstruct_scene', 'inputs':{
        'phase':'prepare', 'entityIds':case.job['inputs']['entityIds'], 'processed':[]}}
    document, result, continuation = run_reconstruction_pipeline(case.repo, case.blobs, job, case.providers)
    assert without_quality(document) == without_quality(case.repo.document)
    assert result['processed'][0]['entityId'] == case.entity['id']
    if unknown:
        assert result['stoppedReason'] == 'provider_outcome_unknown' and continuation is None
    else:
        assert result['processed'][0]['status'] == 'accepted'
        assert continuation['inputs']['entityIds'] == case.job['inputs']['entityIds'][1:]
        assert continuation['inputs']['processed'] == result['processed']
