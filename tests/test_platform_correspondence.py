"""Pure metadata audit; source ownership is not geometry or byte verification."""
from copy import deepcopy

from ehs_spatial.platform.correspondence import audit_correspondence


def scene():
    pose = dict(coordinateFrameId='f', position=[0, 0, 0], quaternion=[0, 0, 0, 1], scale=[1, 1, 1])
    return {'schemaVersion': 2, 'coordinateFrames': [{'id': 'f'}],
        'cameras': [{'id': 'camera', 'imageId': 'image', 'coordinateFrameId': 'f'}],
        'geometryBindings': {'image': {'cameraId': 'camera', 'geometrySolutionId': 'geometry'}},
        'assets': [{'id': 'image', 'sha256': 'a'*64}, {'id': 'geometry', 'sha256': 'b'*64}],
        'observations': [{'id': 'observation', 'revision': 2, 'imageId': 'image'}],
        'entities': [{'id': 'object', 'observationRefs': ['observation'], 'activeModelRepresentationId': 'model',
            'currentModelTransform': pose, 'representations': [{'id': 'model', 'kind': 'primitive',
                'coordinateFrameId': 'f', 'transform': deepcopy(pose), 'placementState': 'unconfirmed',
                'primitive': {'kind': 'box', 'dimensions': [1, 1, 1]},
                'sourceRefs': [{'observationId': 'observation', 'revision': 2, 'imageId': 'image'}]}]}],
        'annotations': []}


def row(document):
    return audit_correspondence(document)['rows'][0]


def test_stale_foreign_and_missing_models_are_not_current():
    document = scene()
    assert row(document)['modelCurrent']
    document['entities'][0]['representations'][0]['sourceRefs'][0]['revision'] = 1
    assert not row(document)['modelCurrent']
    assert 'source_observation_stale' in {e['code'] for e in row(document)['sourceErrors']}
    document = scene()
    document['observations'].append({'id': 'foreign', 'revision': 2, 'imageId': 'image'})
    document['entities'][0]['representations'][0]['sourceRefs'][0]['observationId'] = 'foreign'
    assert not row(document)['modelCurrent']
    assert 'source_observation_foreign' in {e['code'] for e in row(document)['sourceErrors']}
    document['entities'][0]['activeModelRepresentationId'] = 'missing'
    assert row(document)['category'] == 'unresolved'
    assert 'active_model_missing' in row(document)['unmodeledReasons']


def test_model_reference_and_missing_coverage_keep_the_entity_denominator():
    document = scene()
    reference = deepcopy(document['entities'][0])
    reference.update(id='floor', observationRefs=['floor-observation'], activeModelRepresentationId=None)
    reference['representations'][0].update(id='surface', kind='observed_surface', assetId='surface-asset',
        sourceKind='observed_reference_surface', sourceRefs=[{'observationId': 'floor-observation', 'revision': 1, 'imageId': 'image'}],
        planProjection={'assetSha256': 'incorrect', 'transformSnapshot': {}})
    document['assets'].append({'id': 'surface-asset', 'sha256': 'c'*64})
    document['observations'].append({'id': 'floor-observation', 'revision': 1, 'imageId': 'image'})
    document['entities'].extend([reference, {'id': 'missing', 'visible': False, 'observationRefs': [], 'representations': []},
        {'id': 'context', 'sourceContext': True, 'observationRefs': [], 'representations': []}])
    before = deepcopy(document)
    result = audit_correspondence(document)
    assert document == before
    assert result['summary']['coverageDenominator'] == 3
    assert {r['entityId']: r['category'] for r in result['rows']} == {'object': 'model', 'floor': 'reference', 'missing': 'unresolved'}
    assert result['rows'][1]['cad']['status'] == 'unvalidated'
    assert result['assetBytesVerified'] is False
    assert result['rows'][1]['modelCurrent'] is False


def test_unversioned_sources_and_wrong_images_or_frames_are_explicit():
    for field, value, code in [('revision', None, 'source_observation_revision_unrecorded'),
                              ('imageId', 'other', 'source_image_mismatch')]:
        document = scene()
        document['entities'][0]['representations'][0]['sourceRefs'][0][field] = value
        assert code in {e['code'] for e in row(document)['sourceErrors']}
        assert not row(document)['modelCurrent']
    document = scene()
    document['cameras'][0]['coordinateFrameId'] = 'other'
    assert not row(document)['modelCurrent']
    assert 'source_coordinate_frame_mismatch' in {e['code'] for e in row(document)['sourceErrors']}


def test_partition_child_retains_proven_parent_source_without_owning_it():
    document = scene()
    parent = document['entities'][0]
    parent['representations'][0].update(kind='generated_mesh', assetId='source-mesh')
    document['assets'].append({'id': 'source-mesh', 'sha256': 'c'*64})
    child = deepcopy(parent)
    child.update(id='part', observationRefs=['part-observation'], parentEntityId='object', activeModelRepresentationId='part-model')
    document['observations'].append({'id': 'part-observation', 'revision': 1, 'imageId': 'image'})
    evidence = [{'observationId': 'observation', 'revision': 2}, {'observationId': 'part-observation', 'revision': 1}]
    child['partRelation'] = {'source': 'manual', 'baseRevisionId': 'revision', 'reason': 'Reviewed exact component', 'evidenceRefs': evidence}
    event = {'operation': 'partition_model_parts', 'sourceRepresentationId': 'model', 'sourceAssetId': 'source-mesh',
        'sourceAssetSha256': 'c'*64, 'manifestSha256': 'd'*64, 'evidenceRefs': evidence}
    child['lineage'] = [{**event, 'representationId': 'part-model'}]
    child['representations'][0].update(id='part-model', sourceRefs=[{'observationId': 'observation', 'revision': 2}, event],
        placementSource={'type': 'derived_source_partition', 'sourceRepresentationId': 'model', 'manifestSha256': 'd'*64})
    document['entities'].append(child)
    assert audit_correspondence(document)['rows'][1]['modelCurrent']
    del child['lineage']
    assert not audit_correspondence(document)['rows'][1]['modelCurrent']


def test_composite_is_evidence_only_and_stale_targets_invalidate_it():
    document = scene()
    document['observations'].append({'id': 'boundary', 'revision': 1, 'imageId': 'image'})
    document['entities'].append({'id': 'composite', 'observationRefs': ['boundary'], 'representations': []})
    document['annotations'].append({'id': 'mapping', 'kind': 'observation_component_mapping', 'entityId': 'composite',
        'representationType': 'composite_source_evidence', 'independentObject': False,
        'unresolvedBoundaryObservationId': 'boundary', 'unresolvedBoundaryObservationRevision': 1,
        'targets': [{'entityId': 'object', 'activeModelRepresentationId': 'model', 'observationId': 'observation', 'observationRevision': 2}]})
    result = audit_correspondence(document)
    assert result['rows'][1]['category'] == 'composite'
    assert not result['rows'][1]['modelCurrent']
    assert result['summary']['modelCount'] == 1
    document['annotations'][0]['targets'][0]['observationRevision'] = 1
    assert audit_correspondence(document)['rows'][1]['category'] == 'unresolved'


def test_existing_model_with_unproved_import_link_is_not_reported_missing():
    document = scene()
    document['assets'].append({'id': 'archive', 'sha256': 'c'*64})
    document['entities'][0]['representations'][0]['sourceRefs'] = [{'assetId': 'archive', 'sourceRecordId': 'object-record'}]
    result = audit_correspondence(document)
    assert result['rows'][0]['category'] == 'model'
    assert result['rows'][0]['modelAvailable'] and not result['rows'][0]['modelCurrent']
    assert result['rows'][0]['unmodeledReasons'] == []
    assert result['summary']['currentSourceModelCount'] == 0
    document['observations'][0]['sourceRefs'] = [{'assetId': 'archive', 'sourceRecordId': 'object-record'}]
    assert not row(document)['modelCurrent']  # Frozen record identity does not pin an observation revision.
    assert 'source_observation_revision_unrecorded' in {e['code'] for e in row(document)['sourceErrors']}


def test_model_artifact_provenance_cannot_replace_versioned_observation_inputs():
    document = scene()
    document['assets'].append({'id': 'recipe', 'sha256': 'c'*64})
    refs = document['entities'][0]['representations'][0]['sourceRefs']
    artifact = {'assetId': 'recipe', 'sha256': 'c'*64, 'sourceRecordId': 'recipe-row', 'role': 'model_artifact'}
    refs.append(artifact)
    assert row(document)['modelCurrent']
    assert row(document)['sourceErrors'] == []
    refs.pop(0)
    assert 'source_observation_link_missing' in {e['code'] for e in row(document)['sourceErrors']}
    refs.insert(0, {'observationId': 'observation', 'revision': 1, 'imageId': 'image'})
    assert 'source_observation_stale' in {e['code'] for e in row(document)['sourceErrors']}
    refs[0]['revision'] = 2
    artifact['assetId'] = 'geometry'  # An existing unrelated asset must not inherit the recipe's pinned hash.
    assert 'source_asset_hash_mismatch' in {e['code'] for e in row(document)['sourceErrors']}
    artifact['assetId'] = 'missing'
    assert 'source_asset_missing' in {e['code'] for e in row(document)['sourceErrors']}
    artifact['assetId'] = 'recipe'
    del artifact['sha256']
    assert 'source_asset_hash_unrecorded' in {e['code'] for e in row(document)['sourceErrors']}


def test_canonical_import_ref_checks_the_named_observation_and_exact_frozen_view():
    document = scene()
    document['assets'].extend([{'id': 'report', 'sha256': 'c'*64}, {'id': 'foreign-report', 'sha256': 'd'*64}])
    frozen = {'assetId': 'report', 'sourceRecordId': 'raw-id', 'sourceFrameId': 'frame-a', 'jsonPointer': '/objects/0/views/0'}
    document['observations'][0]['sourceRefs'] = [frozen]
    ref = {**frozen, 'sha256': 'c'*64, 'observationId': 'observation', 'revision': 2,
        'imageId': 'image', 'imageSha256': 'a'*64, 'cameraId': 'camera'}
    document['entities'][0]['representations'][0]['sourceRefs'] = [ref]
    assert row(document)['modelCurrent']
    for key, value, code in [('assetId', 'foreign-report', 'source_record_link_unvalidated'),
                             ('jsonPointer', '/objects/1/views/0', 'source_record_link_unvalidated'),
                             ('cameraId', 'foreign-camera', 'source_camera_mismatch'),
                             ('imageSha256', 'b'*64, 'source_asset_hash_mismatch')]:
        bad = deepcopy(document)
        bad['entities'][0]['representations'][0]['sourceRefs'][0][key] = value
        assert code in {e['code'] for e in row(bad)['sourceErrors']}
        assert not row(bad)['modelCurrent']
    document['observations'].append({'id': 'other', 'revision': 2, 'imageId': 'image', 'sourceRefs': []})
    document['entities'][0]['observationRefs'].append('other')
    ref['observationId'] = 'other'
    assert 'source_record_link_unvalidated' in {e['code'] for e in row(document)['sourceErrors']}
