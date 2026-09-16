"""Real DB/worker/authority chain; only synthetic model transports are substituted."""
from copy import deepcopy
from decimal import Decimal
import io
import json
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from ehs_spatial.platform import reconstruction
from ehs_spatial.platform.contracts import digest
from ehs_spatial.platform.recgen import RECGEN_PINS
from ehs_spatial.platform.spatial import transform_points
from ehs_spatial.platform.storage import LocalBlobStore
from ehs_spatial.providers.gemini import GEMINI_MODEL_ID
from panoptes_worker.__main__ import run_job
from test_platform_backend import edit_body, identity, project, repo
from test_platform_reconstruction import provider
from test_recgen_research import payload, research_configuration


@pytest.fixture
def chain(repo, tmp_path, monkeypatch):
    blobs = LocalBlobStore(tmp_path)
    repo.blobs = blobs
    manifest, protocol = research_configuration(payload(), [])
    protocol['runtimeManifest']['generation']['inputContractVersion'] = 'recgen-input-v2'
    preparation = {key: deepcopy(protocol[key]) for key in ('purpose', 'metricDefinitions', 'policyThresholds',
                                                           'split', 'callLimits', 'runtimeManifest')}
    preparation.update(authority={'source': 'database_admin', 'databaseRole': 'test-only-prior'},
                       budgetAtPreparation={'configuredBudgetUsd': '1', 'spentOrReservedUsd': '0'})
    review_pins = {'model': GEMINI_MODEL_ID, 'modelRevision': 'synthetic-test', 'adapter': 'synthetic-test'}
    manifest['model_review'] = {'provider': 'synthetic-review-transport', 'pins': review_pins, 'estimatedCostUsd': .01,
        'paid': True, 'releaseEvidence': {'pins': review_pins,
            **{gate: {'status': 'passed', 'artifactSha256': 'f' * 64} for gate in ('license', 'quality', 'runtime')}}}
    repo.execution_config = {'providerManifest': manifest, 'researchPreparation': preparation}
    cap, scene = project(repo)
    images = []
    for value in (65, 95):
        data = io.BytesIO()
        Image.fromarray(np.full((24, 32, 3), value, np.uint8)).save(data, format='PNG')
        images.append({**blobs.put(data.getvalue(), 'image/png'),
                       'metadata': {'width': 32, 'height': 24, 'pixelMapping': []}})
    capture = repo.create_capture(scene['project']['id'], cap,
        {'requestId': identity(), 'branchId': scene['branch']['id'], 'baseRevisionId': scene['revision']['id'], 'target': 'scene'}, images)
    indices = {item['id']: index for index, item in enumerate(capture['capture']['images'])}
    k = np.array([[24., 0, 15.5], [0, 24., 11.5], [0, 0, 1.]])
    def discovery(request):
        shift = indices[request['image']['imageId']]
        return {'items': [{'label': label, 'box': [left - shift, 4, right - shift, 20]}
                          for label, left, right in [('left fixture', 3, 13), ('right fixture', 19, 29)]]}
    def segmentation(request):
        x0, y0, x1, y1 = map(int, request['box'])
        mask = np.zeros((24, 32), bool)
        mask[y0:y1, x0:x1] = True
        return {'mask': mask}
    def geometry(request):
        frames = []
        y, x = np.indices((24, 32))
        for image in request['images']:
            index = indices[image['imageId']]
            camera = np.eye(4)
            camera[0, 3] = index * 2 / 24
            points = np.stack(((x - 15.5) / 12 + camera[0, 3], (y - 11.5) / 12, np.full(x.shape, 2.)), -1)
            frames.append({'imageId': image['imageId'], 'points': points, 'valid': np.ones((24, 32), bool),
                           'K': k, 'cameraToWorld': camera, 'rgb': np.full((24, 32, 3), 65 + index * 30, np.uint8),
                           'inputToCanonical': np.eye(3)})
        return {'frames': frames}
    analysis_providers = {'discovery': provider('discovery', discovery), 'geometry': provider('geometry', geometry),
        'depth': provider('depth', lambda request: {'imageSha256': request['image']['sha256']}, 'Ruicheng/moge-3-vitl'),
        'segmentation': provider('segmentation', segmentation)}
    generated, reviewed = [], []
    def generation_transport(value, _config, *, is_current):
        assert is_current()
        frozen = value['_researchProtocol']
        source = repo.get_revision(frozen['baselineRevision'])['document']
        entity = next(e for e in source['entities'] if e['id'] == value['entityId'])
        left, right = (2.5, 12.5) if entity['label'] == 'left fixture' else (18.5, 28.5)
        vertices = np.array([[(u - 15.5) / 12, (v - 11.5) / 12, 2.]
                             for u, v in [(left, 3.5), (right, 3.5), (right, 19.5), (left, 19.5)]])
        camera = np.asarray(value['views'][0]['cameraToWorld'])
        pose = np.linalg.inv(camera)
        generated.append({'entityId': entity['id'], 'baselineRevision': frozen['baselineRevision'],
                          'observationIds': [view['observationId'] for view in value['views']]})
        return {'vertices': vertices, 'faces': np.array([[0, 1, 2], [0, 2, 3]]),
                'colors': np.ones((4, 3)), 'objectToCamera': pose,
                'officialPosedVertices': transform_points(vertices, pose), 'pins': RECGEN_PINS,
                'telemetry': {'actualCostUsd': 0}, 'providerRequestId': 'synthetic-generation'}
    def review_transport(value):
        reviewed.append(deepcopy(value))
        return {'review': {'status': 'pass', 'reason': 'Synthetic visible plane matches both source cameras.',
            'observationIds': [view['observationId'] for view in value['views']],
            'visibleShapeIssues': [], 'nextAction': 'none'},
            'telemetry': {'actualCostUsd': 0}, 'providerRequestId': 'synthetic-review'}
    monkeypatch.setitem(sys.modules, 'ehs_spatial.platform.recgen_transport', SimpleNamespace(invoke=generation_transport))
    # Keep the actual configured provider factory, validation, reservation, and cache path.
    monkeypatch.setattr(reconstruction, '_model_review_invoke', review_transport)
    return SimpleNamespace(repo=repo, blobs=blobs, cap=cap, scene=scene, capture=capture,
                           providers=analysis_providers, generated=generated, reviewed=reviewed)


def dispatch(chain, job_id, *, analysis=False):
    return run_job(chain.repo, chain.blobs, job_id, providers=chain.providers if analysis else None)


def successor(chain, result):
    child_id = (result.get('result') or {}).get('continuationJobId')
    assert child_id, result
    return chain.repo.get_job(child_id)


def model_calls(chain):
    with chain.repo._connect() as connection:
        return connection.execute('SELECT * FROM model_calls ORDER BY created_at').fetchall()


def user_edit(chain, revision_id):
    revision = chain.repo.get_revision(revision_id)
    target = next(e for e in revision['document']['entities'] if not e.get('sourceContext'))
    scene = {**chain.scene, 'revision': revision}
    return chain.repo.commit_edits(chain.scene['project']['id'], chain.cap, edit_body(scene, operations=[
        {'type': 'setLabel', 'entityId': target['id'], 'label': 'User revised label'}]))


def start_research(chain):
    analyzed = dispatch(chain, chain.capture['job']['id'], analysis=True)
    assert analyzed['headAdvanced'] and analyzed['resultRevisionId'] != chain.capture['revision']['id'], analyzed
    prepared_job = successor(chain, analyzed)
    assert prepared_job['baseRevisionId'] == analyzed['resultRevisionId']
    prepared = dispatch(chain, prepared_job['id'])
    research = successor(chain, prepared)
    assert research['kind'] == 'validate_model' and research['baseRevisionId'] == analyzed['resultRevisionId']
    asset = chain.repo.get_asset(research['inputs']['validationAssetId'])
    frozen = json.loads(chain.blobs.get(asset['storageKey'], asset['sha256'], asset['sizeBytes']))
    assert frozen['baseRevisionId'] == analyzed['resultRevisionId']
    assert frozen['baseDocumentSha256'] == digest(chain.repo.get_revision(analyzed['resultRevisionId'])['document'])
    return analyzed, research


def test_capture_worker_chain_uses_persisted_baselines_and_preserves_each_model(chain):
    analyzed, research = start_research(chain)
    source = chain.repo.get_revision(analyzed['resultRevisionId'])['document']
    targets = [e for e in source['entities'] if not e.get('sourceContext')]
    assert len(targets) == 2 and all(len(e['observationRefs']) == 2 for e in targets)
    assert not chain.generated and not chain.reviewed
    first_research = dispatch(chain, research['id'])
    assert first_research['status'] == 'succeeded' and first_research['resultRevisionId'] is None, first_research
    assert first_research['result']['scope'] == 'research_only' and first_research['result']['productReleaseStatus'] == 'not_changed'
    assert chain.repo.get_project(research['projectId'])['revision']['id'] == analyzed['resultRevisionId']
    call_count = len(model_calls(chain))
    assert dispatch(chain, research['id'])['id'] == first_research['id']
    assert len(model_calls(chain)) == call_count and len(chain.generated) == 1
    attach = successor(chain, first_research)
    first_attachment = dispatch(chain, attach['id'])
    assert first_attachment['headAdvanced'], first_attachment
    first_document = chain.repo.get_revision(first_attachment['resultRevisionId'])['document']
    first_entity_id = chain.generated[0]['entityId']
    first_entity = next(e for e in first_document['entities'] if e['id'] == first_entity_id)
    assert first_entity['activeModelRepresentationId']
    first_model_id = first_entity['activeModelRepresentationId']
    next_preparation = successor(chain, first_attachment)
    assert next_preparation['baseRevisionId'] == first_attachment['resultRevisionId']
    second_prepared = dispatch(chain, next_preparation['id'])
    second_research = successor(chain, second_prepared)
    second_generated = dispatch(chain, second_research['id'])
    assert second_generated['status'] == 'succeeded', second_generated
    second_attached = dispatch(chain, successor(chain, second_generated)['id'])
    assert second_attached['headAdvanced'], second_attached
    assert not second_attached['result'].get('continuationJobId')
    final = chain.repo.get_project(research['projectId'])['revision']
    assert final['id'] == second_attached['resultRevisionId']
    assert len(chain.generated) == len(chain.reviewed) == 2
    assert chain.generated[0]['baselineRevision'] == analyzed['resultRevisionId']
    assert chain.generated[1]['baselineRevision'] == first_attachment['resultRevisionId']
    assert {call['entityId'] for call in chain.generated} == {e['id'] for e in targets}
    assert next(e for e in final['document']['entities'] if e['id'] == first_entity_id)['activeModelRepresentationId'] == first_model_id
    for entity in (e for e in final['document']['entities'] if not e.get('sourceContext')):
        model = next(r for r in entity['representations'] if r['id'] == entity['activeModelRepresentationId'])
        assert model['shapeStatus'] == 'research_only' and model['placementState'] == 'unconfirmed'
        assert model['qualityEvidence']['status'] == 'accepted'
        assert model['qualityEvidence']['physicalPlacementConfirmed'] is False
        assert model['qualityEvidence']['entityId'] == entity['id']
        assert {r['observationId'] for r in model['sourceRefs'] if 'observationId' in r} == set(entity['observationRefs'])
    assert chain.repo.get_revision(first_attachment['resultRevisionId'])['document'] == first_document


def test_user_edit_before_attachment_keeps_user_head_and_stops_next_child(chain):
    analyzed, research = start_research(chain)
    generated = dispatch(chain, research['id'])
    attach = successor(chain, generated)
    edited = user_edit(chain, analyzed['resultRevisionId'])
    calls_before = len(model_calls(chain))
    result = dispatch(chain, attach['id'])
    assert result['headAdvanced'] is False and result['resultRevisionId']
    assert result['result']['continuationStopped'] == 'branch_changed'
    assert not result['result'].get('continuationJobId')
    assert chain.repo.get_project(research['projectId'])['revision']['id'] == edited['revision']['id']
    assert len(chain.generated) == 1
    assert not chain.reviewed and len(model_calls(chain)) == calls_before
    before = len(model_calls(chain))
    dispatch(chain, attach['id'])
    assert len(model_calls(chain)) == before


def test_user_edit_after_research_enqueue_prevents_generation_reservation(chain):
    analyzed, research = start_research(chain)
    edited = user_edit(chain, analyzed['resultRevisionId'])
    calls_before = len(model_calls(chain))
    stopped = dispatch(chain, research['id'])
    assert stopped['status'] == 'failed'
    assert stopped['result']['error']['code'] == 'pipeline_base_revision_changed'
    assert not stopped['result'].get('continuationJobId')
    assert not chain.generated and not chain.reviewed and len(model_calls(chain)) == calls_before
    assert chain.repo.get_project(research['projectId'])['revision']['id'] == edited['revision']['id']


def test_standalone_fixed_revision_research_remains_artifact_only_after_user_edit(chain):
    from scripts.research.validate_sam3d import submit
    analyzed, internal_research = start_research(chain)
    asset = chain.repo.get_asset(internal_research['inputs']['validationAssetId'])
    frozen = json.loads(chain.blobs.get(asset['storageKey'], asset['sha256'], asset['sizeBytes']))
    edited = user_edit(chain, analyzed['resultRevisionId'])
    standalone = submit({'validation': frozen, 'sha256': digest(frozen)}, chain.repo, chain.blobs)
    assert standalone['kind'] == 'validate_model' and 'pipeline' not in standalone['config']
    finished = dispatch(chain, standalone['id'])
    assert finished['status'] == 'succeeded' and len(chain.generated) == 1, finished
    assert not finished['result'].get('continuationJobId') and finished['resultRevisionId'] is None
    assert chain.repo.get_project(standalone['projectId'])['revision']['id'] == edited['revision']['id']


def test_fresh_budget_rejection_preserves_analysis_and_preparation_record(chain):
    analyzed = dispatch(chain, chain.capture['job']['id'], analysis=True)
    pending = successor(chain, analyzed)
    calls_before = len(model_calls(chain))
    chain.repo.paid_budget = Decimal('0')
    stopped = dispatch(chain, pending['id'])
    assert stopped['status'] == 'incomplete', stopped
    assert stopped['result']['continuationStopped'] == 'paid_budget_not_configured'
    assert stopped['result']['validationAssetId'] and not stopped['result'].get('continuationJobId')
    assert chain.repo.get_project(pending['projectId'])['revision']['id'] == analyzed['resultRevisionId']
    assert len(model_calls(chain)) == calls_before and not chain.generated and not chain.reviewed
    assert dispatch(chain, pending['id'])['result'] == stopped['result']
    with chain.repo._connect() as connection:
        assert connection.execute("SELECT count(*) AS n FROM jobs WHERE kind='validate_model'").fetchone()['n'] == 0
