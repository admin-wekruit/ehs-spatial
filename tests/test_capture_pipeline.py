"""New-photo entrypoint checks; providers here are synthetic, never quality proof."""
from copy import deepcopy

import numpy as np
import pytest

from ehs_spatial.platform.contracts import PlatformError, validate_document
from ehs_spatial.platform.postgres import PostgresRepository
from ehs_spatial.platform.reconstruction import _Stages, run_capture_pipeline
from ehs_spatial.platform.storage import LocalBlobStore
from test_platform_reconstruction import Repo, bundle, provider


def providers_for_new_capture(repo, *, verdict='pass'):
    providers = bundle(repo)
    providers['discovery'] = provider('discovery', lambda payload: {
        'items':[{'label':'new flat ceramic sample','box':[0,0,5,12]}]})
    providers['generation'] = provider('generation', lambda payload: {
        'vertices':np.array([[-1.2,-1.2,2.],[-.2,-1.2,2.],[-.2,1.2,2.],[-1.2,1.2,2.]]),
        'faces':np.array([[0,1,2],[0,2,3]]), 'proposedObjectToNative':np.eye(4)})
    providers['model_review'] = provider('model_review', lambda payload:{'review':{
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
    assert any(r['kind'] == 'generated_mesh' for r in entity['representations'])


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
        'providerManifest':{'generation':'forged'},'researchProtocolSha256':'0'*64})
    assert config == {'seed':7,'providerManifest':{}}
    server = {'researchPreparation':{'authority':{'source':'database_admin'}}}
    repo.execution_config = server
    assert repo._job_config({'researchPreparation':'forged'}) == server


def test_research_capture_defers_preparation_until_analysis_revision_is_persisted(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = providers_for_new_capture(repo)
    providers.pop('generation')
    providers.pop('model_review')
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
    assert started == ['discovery','discovery','geometry','depth']


@pytest.mark.parametrize('stage', ['generation', 'model_review'])
@pytest.mark.parametrize('failure', ['timeout', 'persistence'])
def test_unknown_model_outcome_stops_remaining_objects(tmp_path, monkeypatch, stage, failure):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = providers_for_new_capture(repo)
    providers['discovery'] = bundle(repo)['discovery']  # Multiple independent targets.
    if failure == 'timeout':
        def timeout(_):
            raise TimeoutError('uncertain result')
        providers[stage] = provider(stage, timeout)
    else:
        original = _Stages.put
        def interrupted(self, value, metadata, *args):
            if metadata.get('kind') == 'stage_cache' and metadata.get('stage') == stage:
                raise OSError('response not durably stored')
            return original(self, value, metadata, *args)
        monkeypatch.setattr(_Stages, 'put', interrupted)
    document, result = run_capture_pipeline(repo, blobs, repo.job, providers)
    assert result['status'] == 'incomplete'
    expected = 'provider_outcome_unknown' if failure == 'timeout' else 'response_persistence_failed'
    assert result['generation']['stoppedReason'] == expected
    assert sum(event == ('reserve', 'generation') for event in repo.events) == 1
    assert sum(event == ('reserve', 'model_review') for event in repo.events) == (stage == 'model_review')
    assert len([e for e in document['entities'] if not e.get('sourceContext')]) == 4


def test_foreign_cad_reference_is_rejected_before_any_provider_call(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    job = {**repo.job, 'inputs':{**repo.job['inputs'], 'referenceImageId':'foreign-photo'}}
    with pytest.raises(PlatformError, match='cad_reference_image_not_found'):
        run_capture_pipeline(repo, blobs, job, providers_for_new_capture(repo))
    assert not repo.calls
