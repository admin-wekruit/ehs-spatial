"""Accepted quality belongs to one artifact, pose, and complete source view set."""
from copy import deepcopy

import pytest

from argus.platform import correspondence
from argus.platform.reconstruction import run_capture_pipeline
from argus.platform.storage import LocalBlobStore
from test_capture_pipeline import providers_for_new_capture
from test_platform_reconstruction import Repo


@pytest.fixture
def reviewed(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repository = Repo(blobs)
    document, _ = run_capture_pipeline(repository, blobs, repository.job, providers_for_new_capture(repository))
    entity = next(entity for entity in document['entities'] if entity.get('activeModelRepresentationId'))
    rep = next(rep for rep in entity['representations'] if rep['id'] == entity['activeModelRepresentationId'])
    assert rep['qualityEvidence']['status'] == 'accepted'
    rep['qualityBinding'] = correspondence.model_quality_binding(document, entity, rep)
    return document, entity, rep


def current(document):
    return next(row for row in correspondence.audit_correspondence(document)['rows'] if row['modelDeclared'])


def test_exact_review_binding_is_current_and_legacy_status_is_not(reviewed):
    document, entity, rep = reviewed
    before = deepcopy(document)
    assert current(document)['qualityCurrent'] is True
    assert document == before
    del rep['qualityBinding']
    assert current(document)['qualityStatus'] == 'accepted'
    assert current(document)['qualityCurrent'] is False


@pytest.mark.parametrize('change', ['pose', 'asset', 'mesh_hash', 'new_observation', 'observation_revision',
                                   'image_hash', 'mask_hash', 'geometry_hash', 'camera', 'quality', 'evidence_hash',
                                   'entity', 'representation', 'source_ownership', 'mask_complete'])
def test_any_reviewed_input_change_invalidates_current_quality(reviewed, change):
    document, entity, rep = reviewed
    observation = next(o for o in document['observations'] if o['id'] == entity['observationRefs'][0])
    binding = document['geometryBindings'][observation['imageId']]
    assets = {asset['id']: asset for asset in document['assets']}
    if change == 'pose':
        entity['currentModelTransform']['position'][0] += 100
    elif change == 'asset':
        rep['assetId'] = observation['maskAssetId']
    elif change == 'mesh_hash':
        assets[rep['assetId']]['sha256'] = '0' * 64
    elif change == 'new_observation':
        document['observations'].append({**deepcopy(observation), 'id': 'new-view'})
        entity['observationRefs'].append('new-view')
    elif change == 'observation_revision':
        observation['revision'] += 1
    elif change == 'image_hash':
        assets[observation['imageId']]['sha256'] = '0' * 64
    elif change == 'mask_hash':
        assets[observation['maskAssetId']]['sha256'] = '0' * 64
    elif change == 'geometry_hash':
        assets[binding['geometrySolutionId']]['sha256'] = '0' * 64
    elif change == 'camera':
        next(c for c in document['cameras'] if c['id'] == binding['cameraId'])['cameraToWorld'][0][3] += 1
    elif change == 'quality':
        rep['qualityEvidence']['geometric']['status'] = 'observed_inconsistent'
    elif change == 'evidence_hash':
        assets[rep['qualityEvidence']['evidenceRef']['assetId']]['sha256'] = '0' * 64
    elif change == 'entity':
        entity['id'] = 'different-entity'
    elif change == 'representation':
        rep['id'] = entity['activeModelRepresentationId'] = 'different-model'
    elif change == 'source_ownership':
        document['entities'].append({'id': 'other-owner', 'observationRefs': [observation['id']], 'representations': []})
    elif change == 'mask_complete':
        observation['maskComplete'] = True
    assert current(document)['qualityCurrent'] is False


def test_binding_creation_uses_candidate_pose_and_does_not_inherit_previous_active_pose(reviewed):
    document, entity, rep = reviewed
    expected = deepcopy(rep['qualityBinding'])
    entity['currentModelTransform'] = {**deepcopy(rep['transform']), 'position': [100., 0., 0.]}
    assert correspondence.model_quality_binding(document, entity, rep) == expected
    assert current(document)['qualityCurrent'] is False


def test_observation_order_does_not_change_the_reviewed_view_set(reviewed):
    document, entity, _ = reviewed
    entity['observationRefs'].reverse()
    assert current(document)['qualityCurrent'] is True


@pytest.mark.parametrize('malformed', ['accepted', ['accepted'], {'status': 'accepted', 'geometric': ['invalid']}])
def test_malformed_historical_quality_fails_closed(reviewed, malformed):
    document, _, rep = reviewed
    rep['qualityEvidence'] = malformed
    assert current(document)['qualityCurrent'] is False
