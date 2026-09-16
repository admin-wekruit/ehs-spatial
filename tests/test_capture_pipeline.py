"""New-photo entrypoint checks; providers here are synthetic, never quality proof."""
from copy import deepcopy

import numpy as np
import pytest

from ehs_spatial.platform.contracts import PlatformError, validate_document
from ehs_spatial.platform.postgres import PostgresRepository
from ehs_spatial.platform.reconstruction import _Stages, run_capture_pipeline
from ehs_spatial.platform.storage import LocalBlobStore
from test_platform_reconstruction import Repo, bundle, geometry_response, inventory_review_response, provider


def providers_for_new_capture(repo, *, verdict='pass'):
    providers = bundle(repo)
    providers['discovery'] = provider('discovery', lambda payload: {
        'items':[{'label':'new flat ceramic sample','box':[0,0,5,12]}]})
    providers['generation'] = provider('generation', lambda payload: {
        'vertices':np.array([[-1.2,-1.2,2.],[-.2,-1.2,2.],[-.2,1.2,2.],[-1.2,1.2,2.]]),
        'faces':np.array([[0,1,2],[0,2,3]]), 'proposedObjectToNative':np.eye(4)})
    providers['model_review'] = provider('model_review', lambda payload: inventory_review_response(payload) if payload.get('mode') in ('inventory', 'workcell_scope') else {'review':{
        'status':verdict, 'reason':'synthetic contract fixture',
        'observationIds':[view['observationId'] for view in payload['views']],
        'visibleShapeIssues':[] if verdict == 'pass' else ['visible structure differs'],
        'nextAction':'none' if verdict == 'pass' else 'alternate_view'}})
    return providers


def test_new_capture_runs_analysis_model_review_and_correspondence_once(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    base = deepcopy(repo.document)
    providers = providers_for_new_capture(repo)
    document,result = run_capture_pipeline(repo,blobs,repo.job,providers)
    assert result['status'] == 'incomplete' and result['generation']['status'] == 'succeeded', result
    assert repo.document == base  # Only finish_job commits the final immutable scene.
    validate_document(document)
    rows = result['correspondence']['rows']
    assert len(rows) == 1 and rows[0]['modelCurrent'] and not rows[0]['sourceErrors']
    entity = next(e for e in document['entities'] if e['id'] == rows[0]['entityId'])
    model = next(r for r in entity['representations'] if r['id'] == entity['activeModelRepresentationId'])
    assert model['shapeStatus'] == 'observed_accepted'
    assert model['placementState'] == 'unconfirmed'
    assert model['qualityEvidence']['physicalPlacementConfirmed'] is False
    assert {r['observationId'] for r in model['sourceRefs'] if 'observationId' in r} == set(entity['observationRefs'])
    # No ground reference means no CAD yet. Generation success must not report
    # this capture as a completed photo/CAD/model report.
    assert 'planProjection' not in model
    assert rows[0]['cad']['status'] == 'absent'
    assert result['cadPendingEntityIds'] == [entity['id']]
    count = len(repo.calls)
    again,replayed = run_capture_pipeline(repo,blobs,repo.job,providers)
    assert replayed['status'] == 'incomplete' and len(repo.calls) == count
    assert [e['id'] for e in again['entities']] == [e['id'] for e in document['entities']]


@pytest.mark.parametrize('retain_supported_photo', [False, True])
def test_empty_discovery_cannot_hide_unavailable_capture_geometry(tmp_path, retain_supported_photo):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = bundle(repo)
    providers['discovery'] = provider('discovery', lambda _: {'items':[]})
    def geometry(payload):
        response = geometry_response(payload['images'])
        for frame in response['frames'][:1 if retain_supported_photo else 2]:
            frame['points'] = np.broadcast_to([0., 0., 2.], frame['points'].shape).copy()
        return response
    providers['geometry'] = provider('geometry', geometry)
    def unexpected_model_call(_):
        pytest.fail('Empty discovery must not invent a model target')
    providers['generation'] = provider('generation', unexpected_model_call)
    providers['model_review'] = provider('model_review', lambda payload: inventory_review_response(payload)
        if payload.get('mode') in ('inventory', 'workcell_scope') else unexpected_model_call(payload))

    document, result = run_capture_pipeline(repo, blobs, repo.job, providers)
    assert result['status'] == result['analysis']['status'] == 'incomplete'
    missing = {image['id'] for image in repo.capture['images'][:1 if retain_supported_photo else 2]}
    assert result['errors'] == result['analysis']['errors']
    assert {error['imageId'] for error in result['errors']} == missing
    assert all(error['stage'] == 'capture_context' and error['code'] == 'observed_context_unavailable'
               for error in result['errors'])
    assert document['observations'] == [] and all(entity.get('sourceContext') for entity in document['entities'])
    retained = {ref['imageId'] for entity in document['entities'] for rep in entity['representations'] for ref in rep['sourceRefs'] if 'imageId' in ref}
    assert retained == {image['id'] for image in repo.capture['images']} - missing
    assert repo.document['entities'] == []
    validate_document(document)


def test_bad_shape_is_retained_but_not_reported_as_completed_model(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    document,result = run_capture_pipeline(repo,blobs,repo.job,providers_for_new_capture(repo,verdict='fail'))
    assert result['status'] == 'incomplete'
    assert result['generation']['qualityResults'][0]['geometric']['status'] == 'observed_consistent'
    assert result['generation']['qualityResults'][0]['status'] == 'rejected'
    assert result['correspondence']['summary']['coverageDenominator'] == 1
    assert result['correspondence']['summary']['unresolvedCount'] == 1
    entity = next(e for e in document['entities'] if not e.get('sourceContext'))
    assert not entity.get('activeModelRepresentationId')
    assert len([r for r in entity['representations'] if r['kind'] == 'generated_mesh']) == 2
    assert sum(event == ('reserve', 'generation') for event in repo.events) == 2
    assert result['generation']['shapeReadyEntityIds'] == [entity['id']]
    assert len(result['generation']['qualityResults']) == 2


@pytest.mark.parametrize('explicit_anchor', [False, True])
def test_capture_uses_reviewed_alternate_owned_photo_and_replays_both_candidates(tmp_path, explicit_anchor):
    from ehs_spatial.platform.reconstruction import run_analysis

    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = providers_for_new_capture(repo)
    source, _ = run_analysis(repo, blobs, repo.job, providers)
    entity = next(e for e in source['entities'] if not e.get('sourceContext'))
    observations = [o for o in source['observations'] if o['id'] in entity['observationRefs']]
    preferred = sorted(observations, key=lambda o:(o['geometrySupport']['validPixelCount'], o['id']), reverse=True)
    initial = preferred[-1] if explicit_anchor else preferred[0]
    job = deepcopy(repo.job)
    if explicit_anchor:
        job['inputs']['observationIds'] = [initial['id']]
    generated, reviewed = [], []
    original_generation = providers['generation'].invoke
    def generate(payload):
        generated.append((payload['entityId'], payload['imageId']))
        return original_generation(payload)
    def review(payload):
        if payload.get('mode') in ('inventory', 'workcell_scope'):
            return inventory_review_response(payload)
        reviewed.append(deepcopy(payload))
        accepted = len(generated) == 2
        return {'review':{'status':'pass' if accepted else 'fail',
            'reason':'Synthetic second source resolves the visible structure.',
            'observationIds':[view['observationId'] for view in payload['views']],
            'visibleShapeIssues':[] if accepted else ['First candidate contradicts visible structure.'],
            'nextAction':'none' if accepted else 'alternate_view'}}
    providers['generation'] = provider('generation', generate)
    providers['model_review'] = provider('model_review', review)

    document, result = run_capture_pipeline(repo, blobs, job, providers)
    generation = result['generation']
    assert generation['status'] == 'succeeded', generation
    assert generated == [(entity['id'], initial['imageId']),
        (entity['id'], next(o['imageId'] for o in observations if o['imageId'] != initial['imageId']))]
    assert generation['shapeReadyEntityIds'] == generation['acceptedEntityIds'] == [entity['id']]
    assert [q['status'] for q in generation['qualityResults']] == ['rejected', 'accepted']
    assert all({v['observationId'] for v in p['views']} == set(entity['observationRefs']) for p in reviewed)
    current = next(e for e in document['entities'] if e['id'] == entity['id'])
    models = [r for r in current['representations'] if r['kind'] == 'generated_mesh']
    assert len(models) == 2 and current['activeModelRepresentationId'] == models[1]['id']
    assert all(model['qualityEvidence']['shapeReview']['evidenceRef'] for model in models)
    assert result['correspondence']['summary']['coverageDenominator'] == 1
    assert len([e for e in document['entities'] if not e.get('sourceContext')]) == 1
    calls = len(repo.calls)
    replay, replayed = run_capture_pipeline(repo, blobs, job, providers)
    assert len(repo.calls) == calls and len(generated) == len(reviewed) == 2
    replay_entity = next(e for e in replay['entities'] if e['id'] == entity['id'])
    assert [r['id'] for r in replay_entity['representations'] if r['kind'] == 'generated_mesh'] == [r['id'] for r in models]
    assert replayed['generation']['acceptedEntityIds'] == [entity['id']]
    validate_document(document)


@pytest.mark.parametrize('action', ['none', 'correct_mask', 'invalid_review', 'unknown'])
def test_capture_does_not_try_another_photo_without_valid_alternate_view_review(tmp_path, action):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = providers_for_new_capture(repo)
    def review(payload):
        if payload.get('mode') in ('inventory', 'workcell_scope'):
            return inventory_review_response(payload)
        if action == 'unknown':
            raise TimeoutError('Unknown review outcome')
        return {'review':{'status':'fail', 'reason':'Synthetic unresolved visible structure.',
            'observationIds':['foreign'] if action == 'invalid_review' else [v['observationId'] for v in payload['views']],
            'visibleShapeIssues':['Visible structure differs.'],
            'nextAction':'alternate_view' if action == 'invalid_review' else action}}
    providers['model_review'] = provider('model_review', review)
    document, result = run_capture_pipeline(repo, blobs, repo.job, providers)
    assert result['generation']['status'] == 'incomplete'
    assert sum(event == ('reserve', 'generation') for event in repo.events) == 1
    entity = next(e for e in document['entities'] if not e.get('sourceContext'))
    assert len([r for r in entity['representations'] if r['kind'] == 'generated_mesh']) == 1
    assert not entity.get('activeModelRepresentationId')
    if action == 'unknown':
        assert result['generation']['stoppedReason'] == 'provider_outcome_unknown'


@pytest.mark.parametrize('reference_only', [False, True])
def test_complete_capture_requires_actual_model_and_reference_cad(tmp_path, reference_only):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs, size=32)
    repo.capture['target'] = repo.document['target'] = 'scene'
    providers = providers_for_new_capture(repo)
    items = [
        {'label':'flat ceramic sample', 'box':[0,0,12,32]},
        {'label':'visible floor', 'box':[16,0,32,32], 'geometryRole':'floor'}]
    providers['discovery'] = provider('discovery', lambda _: {'items':items[1:] if reference_only else items})
    if reference_only:
        repo.job['config'] = {'providerManifest':{'generation':{'pins':{'model':'TRI-ML/RecGen'}}}}
    def geometry(payload):
        y, x = np.mgrid[:32,:32]
        points = np.stack(((x-15.5)/8, (y-15.5)/8, np.full_like(x,2)), axis=-1).astype(float)
        return {'frames':[{'imageId':image['imageId'], 'points':points,
            'valid':np.ones((32,32),bool), 'K':np.array([[16,0,15.5],[0,16,15.5],[0,0,1]]),
            'cameraToWorld':np.eye(4), 'rgb':np.full((32,32,3),100,np.uint8),
            'inputToCanonical':np.eye(3)} for image in payload['images']]}
    def segmentation(payload):
        x0,y0,x1,y1 = map(int,payload['box'])
        mask = np.zeros((32,32),bool)
        mask[y0:y1,x0:x1] = True
        return {'mask':mask}
    providers['geometry'] = provider('geometry',geometry)
    providers['segmentation'] = provider('segmentation',segmentation)
    providers['generation'] = provider('generation', lambda _: {
        'vertices':np.array([[-2,-2,2],[-.5,-2,2],[-.5,2,2],[-2,2,2]]),
        'faces':np.array([[0,1,2],[0,2,3]]), 'proposedObjectToNative':np.eye(4)})
    document, result = run_capture_pipeline(repo, blobs, repo.job, providers)
    assert result['status'] == 'succeeded', result
    rows = result['correspondence']['rows']
    assert len(rows) == (1 if reference_only else 2)
    assert result['cadValidationStatus'] == 'validated'
    assert all(row['cad']['status'] == 'validated' for row in rows)
    assert {row['cad']['kind'] for row in rows} == ({'observed_reference'} if reference_only else {'model', 'observed_reference'})
    assert result['correspondence']['summary']['modelCount'] == (0 if reference_only else 1)
    assert result['correspondence']['summary']['referenceCount'] == 1
    assert not result['qualityPendingEntityIds']
    if not reference_only:
        from ehs_spatial.platform.correspondence import validate_cad_correspondence
        from ehs_spatial.platform.reconstruction_pipeline import _result
        model = next(e for e in document['entities'] if e.get('activeModelRepresentationId'))
        model['currentModelTransform']['position'][0] += 100
        # Translation preserves the CAD cache contract, but the old shape/pose
        # review must no longer complete a resumed reconstruction.
        moved = validate_cad_correspondence(document, _Stages(repo, blobs, repo.job, {}))
        resumed = _result('prepare', [], [], document=document, correspondence=moved)
        assert resumed['cadValidationStatus'] == 'validated'
        assert resumed['status'] == 'incomplete'
        assert resumed['qualityPendingEntityIds'] == [model['id']]


def test_request_cannot_supply_research_authority_or_provider_manifest():
    repo = PostgresRepository('unused')
    config = repo._job_config({'seed':7,'researchPreparation':{'authority':'forged'},
        'providerManifest':{'generation':'forged'},'researchProtocolSha256':'0'*64,
        'captureAnalysis':{'result':{'status':'succeeded'}}})
    assert config == {'seed':7,'providerManifest':{}}
    server = {'researchPreparation':{'authority':{'source':'database_admin'}}}
    repo.execution_config = server
    assert repo._job_config({'researchPreparation':'forged'}) == server


def test_research_capture_defers_preparation_until_analysis_revision_is_persisted(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = providers_for_new_capture(repo)
    providers.pop('generation')
    providers['model_review'] = provider('model_review', inventory_review_response)
    job = {**repo.job, 'config':{
        'providerManifest':{'generation':{'pins':{'model':'TRI-ML/RecGen'}}},
        'researchPreparation':{'authority':{'source':'database_admin'}}}}
    document,result = run_capture_pipeline(repo,blobs,job,providers)
    assert result['status'] == 'incomplete' and result['pipelineStatus'] == 'modeling_pending'
    successor = result.pop('_continuation')
    assert successor['kind'] == 'reconstruct_scene'
    assert successor['inputs']['phase'] == 'prepare'
    assert successor['inputs']['entityIds'] == [e['id'] for e in document['entities'] if not e.get('sourceContext')]
    assert 'baseRevisionId' not in successor  # finish_job supplies the actual new revision.
    assert successor['config']['captureAnalysis'] == {'jobId':job['id'], 'baseRevisionId':job['baseRevisionId'],
        'result':{**result['analysis'], 'review':result['review']}}
    assert all(event[1] != 'generation' for event in repo.events if event[0] == 'reserve')


def test_unknown_analysis_outcome_stops_later_paid_stages_and_model_continuation(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = providers_for_new_capture(repo)
    def unknown_depth(payload):
        raise TimeoutError('uncertain response')
    providers['depth'] = provider('depth',unknown_depth,'Ruicheng/moge-3-vitl')
    job = {**repo.job,'config':{'providerManifest':{'generation':{'pins':{'model':'TRI-ML/RecGen'}}},
        'researchPreparation':{'authority':{'source':'database_admin'}}}}
    document,result = run_capture_pipeline(repo,blobs,job,providers)
    assert document['entities'] and result['stoppedReason'] == 'provider_outcome_unknown'
    assert '_continuation' not in result
    started = [event[1] for event in repo.events if event[0] == 'reserve']
    assert started == ['discovery','model_review','discovery','model_review','geometry','depth']


@pytest.mark.parametrize('stage', ['generation', 'model_review'])
@pytest.mark.parametrize('failure', ['timeout', 'persistence'])
def test_unknown_model_outcome_stops_remaining_objects(tmp_path, monkeypatch, stage, failure):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = providers_for_new_capture(repo)
    providers['discovery'] = bundle(repo)['discovery']  # Multiple independent targets.
    if failure == 'timeout':
        def timeout(payload):
            if payload.get('mode') in ('inventory', 'workcell_scope'):
                return inventory_review_response(payload)
            raise TimeoutError('uncertain result')
        providers[stage] = provider(stage, timeout)
    else:
        original = _Stages.put
        def interrupted(self, value, metadata, *args):
            if (metadata.get('kind') == 'stage_cache' and metadata.get('stage') == stage
                    and not value.get('output', {}).get('review', {}).get('inventorySha256')):
                raise OSError('response not durably stored')
            return original(self, value, metadata, *args)
        monkeypatch.setattr(_Stages, 'put', interrupted)
    document, result = run_capture_pipeline(repo, blobs, repo.job, providers)
    assert result['status'] == 'incomplete'
    expected = 'provider_outcome_unknown' if failure == 'timeout' else 'response_persistence_failed'
    assert result['generation']['stoppedReason'] == expected
    assert sum(event == ('reserve', 'generation') for event in repo.events) == 1
    assert sum(event == ('reserve', 'model_review') for event in repo.events) == len(repo.capture['images']) + (stage == 'model_review')
    assert len([e for e in document['entities'] if not e.get('sourceContext')]) == 4


def test_foreign_cad_reference_is_rejected_before_any_provider_call(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    job = {**repo.job, 'inputs':{**repo.job['inputs'], 'referenceImageId':'foreign-photo'}}
    with pytest.raises(PlatformError, match='cad_reference_image_not_found'):
        run_capture_pipeline(repo, blobs, job, providers_for_new_capture(repo))
    assert not repo.calls


@pytest.mark.parametrize('projection_fails', [False, True])
def test_research_continuation_preserves_retained_review_outcome(tmp_path, monkeypatch, projection_fails):
    from ehs_spatial.platform import reconstruction
    from ehs_spatial.platform.reconstruction_pipeline import _result

    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = providers_for_new_capture(repo)
    document, baseline = run_capture_pipeline(repo, blobs, repo.job, providers)
    retained = next(e for e in document['entities'] if e.get('activeModelRepresentationId'))
    model = next(r for r in retained['representations'] if r['id'] == retained['activeModelRepresentationId'])
    model.pop('qualityBinding')  # A retained model needs fresh review.
    for frame in document['coordinateFrames']:
        frame['ground'] = {'normal': [0., 0., 1.]}
    stages = _Stages(repo, blobs, repo.job, {})
    _, _, images = reconstruction._capture(repo, blobs, repo.job)
    evidence = stages.put({'new': 'visible target'}, {'kind': 'discovery'})
    reconstruction._discover(document, images[0], {'items': [
        {'label': 'new separate target', 'box': [6, 0, 11, 12]}]}, evidence)
    monkeypatch.setattr(reconstruction, 'run_analysis', lambda *_: (document, deepcopy(baseline['analysis'])))
    if projection_fails:
        def invalid_projection(*args, **kwargs):
            raise PlatformError('projection_geometry_invalid', 422)
        monkeypatch.setattr(reconstruction, '_refresh_plan_projections', invalid_projection)
    job = {**repo.job, 'config': {
        'providerManifest': {'generation': {'pins': {'model': 'TRI-ML/RecGen'}}},
        'researchPreparation': {'authority': {'source': 'database_admin'}}}}

    _, result = run_capture_pipeline(repo, blobs, job, providers)
    assert result['analysis']['status'] == 'succeeded'
    assert result['review']['status'] == ('incomplete' if projection_fails else 'succeeded')
    assert result['review']['stages']  # Actual synthetic-provider retained review ran.
    if projection_fails:
        assert any(error['code'] == 'projection_geometry_invalid' for error in result['review']['errors'])
    frozen = result['_continuation']['config']['captureAnalysis']
    ancestor = frozen['result']
    assert ancestor['review'] == result['review']
    assert ancestor['errors'] == result['analysis']['errors'] + result['review']['errors']
    assert ancestor['stages'] == result['analysis']['stages'] + result['review']['stages']
    # The pending new model is accepted later. Its parent's temporary modeling
    # status must not become a permanent failure, but prior review errors must.
    final = _result('attach', [{'entityId': 'new-target', 'status': 'accepted'}], [], capture_analysis=frozen)
    assert result['status'] == 'incomplete' and result['pipelineStatus'] == 'modeling_pending'
    assert final['status'] == ('incomplete' if projection_fails else 'succeeded')
    assert final['errors'] == ancestor['errors']
    result['review']['stages'].clear()
    assert ancestor['review']['stages'] and ancestor['stages']
