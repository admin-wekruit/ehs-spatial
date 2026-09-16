"""CAD proof uses actual local mesh bytes and the product triangle projector."""
from copy import deepcopy

import numpy as np
import pytest

from ehs_spatial.platform import correspondence, reconstruction
from ehs_spatial.platform.spatial import MeshData
from ehs_spatial.platform.storage import LocalBlobStore
from test_platform_correspondence import scene
from test_platform_reconstruction import Repo


def prepared(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    stages = reconstruction._Stages(repo, blobs, repo.job, {})
    document = scene()
    document['coordinateFrames'][0]['ground'] = {'normal': [0., 0., 1.]}
    document['reportEvidence'] = {'plan': {'coordinateFrameId': 'f', 'nativeToFloor': np.eye(4).tolist()}}
    vertices, faces = [], []
    for x, y in [(x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1)] + [(5, 0)]:
        start = len(vertices)
        vertices.extend([(x, y, 0), (x+1, y, 0), (x+1, y+1, 0), (x, y+1, 0)])
        faces.extend([(start, start+1, start+2), (start, start+2, start+3)])
    # An isolated vertical triangle must remain an edge, not a bounding box.
    start = len(vertices)
    vertices.extend([(7, 0, 0), (7, 1, 0), (7, 1, 2)])
    faces.append((start, start+1, start+2))
    mesh = MeshData(np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.uint32))
    asset = reconstruction._save_mesh(stages, mesh, {})
    reconstruction._include(document, asset)
    entity = document['entities'][0]
    model = entity['representations'][0]
    model.update(kind='generated_mesh', assetId=asset['id'], primitive=None)
    observed = {**deepcopy(model), 'id': 'observed', 'kind': 'observed_surface'}
    entity['representations'].append(observed)
    reconstruction._refresh_plan_projections(document, stages)
    return document, stages


def audit(document, stages):
    return correspondence.validate_cad_correspondence(document, stages)


def test_actual_mesh_proof_preserves_holes_islands_edges_and_document(tmp_path):
    document, stages = prepared(tmp_path)
    before = deepcopy(document)
    result = audit(document, stages)
    row = result['rows'][0]
    assert row['cad']['status'] == row['cad']['model']['status'] == 'validated'
    assert row['cad']['observed'][0]['status'] == 'validated'
    proof = row['cad']['model']
    assert proof['assetSha256'] == document['entities'][0]['representations'][0]['planProjection']['assetSha256']
    assert proof['meshBytesVerified'] and proof['triangleCount'] == 19
    assert proof['polygonCount'] == 2 and proof['holeCount'] == 1 and proof['lineCount'] == 1
    assert proof['poseRelation'] == 'exact' and proof['coordinateFrameId'] == 'f'
    assert document == before and row['placementState'] == 'unconfirmed'
    assert row['qualityStatus'] == 'unreviewed'
    assert result['assetBytesVerified'] is False  # Unrelated source assets were not read.


@pytest.mark.parametrize('change, reason', [
    ('polygon', 'projection_geometry_mismatch'), ('hole', 'projection_geometry_mismatch'),
    ('line', 'projection_geometry_mismatch'), ('asset', 'projection_asset_mismatch'),
    ('pose', 'projection_pose_mismatch'), ('frame', 'projection_frame_mismatch'),
    ('plane', 'projection_plane_mismatch'), ('source', 'projection_source_mismatch'),
    ('stale_source', 'source_correspondence_invalid'), ('open_ring', 'projection_geometry_invalid'),
])
def test_corruption_cannot_validate_model_from_observed_projection(tmp_path, change, reason):
    document, stages = prepared(tmp_path)
    entity = document['entities'][0]
    rep = entity['representations'][0]
    saved = rep['planProjection']
    if change == 'polygon':
        saved['polygons'][0]['exterior'][1][0] += .25
    elif change == 'hole':
        next(p for p in saved['polygons'] if p['holes'])['holes'] = []
    elif change == 'line':
        saved['lines'] = []
    elif change == 'asset':
        saved['assetSha256'] = '0' * 64
    elif change == 'pose':
        entity['currentModelTransform']['scale'][0] = 2
    elif change == 'frame':
        saved['coordinateFrameId'] = 'foreign'
    elif change == 'plane':
        saved['nativeToPlane'][0][3] = 2
    elif change == 'source':
        saved['observationId'] = 'foreign'
    elif change == 'stale_source':
        rep['sourceRefs'][0]['revision'] = 1
    elif change == 'open_ring':
        saved['polygons'][0]['exterior'].pop()
    result = audit(document, stages)['rows'][0]['cad']
    assert result['status'] == result['model']['status'] == 'invalid'
    assert reason in result['model']['reasons']
    assert result['observed'][0]['status'] == 'validated'


def test_ring_order_and_translation_preserve_current_pose_proof(tmp_path):
    document, stages = prepared(tmp_path)
    entity = document['entities'][0]
    saved = entity['representations'][0]['planProjection']
    saved['polygons'].reverse()
    for polygon in saved['polygons']:
        polygon['exterior'].reverse()
        polygon['holes'] = [list(reversed(hole)) for hole in reversed(polygon['holes'])]
    saved['lines'] = [list(reversed(line)) for line in reversed(saved['lines'])]
    entity['currentModelTransform']['position'] = [4., -2., 1.]
    result = audit(document, stages)['rows'][0]['cad']['model']
    assert result['status'] == 'validated' and result['poseRelation'] == 'translation'
    assert result['transformSnapshot'] == entity['currentModelTransform']
    assert result['cacheTransformSnapshot'] == saved['transformSnapshot']


def test_missing_model_and_missing_floor_remain_absent(tmp_path):
    document, stages = prepared(tmp_path)
    document['entities'][0]['activeModelRepresentationId'] = None
    result = audit(document, stages)['rows'][0]['cad']
    assert result['status'] == result['model']['status'] == 'absent'
    assert result['observed'][0]['status'] == 'validated'
    document['coordinateFrames'][0]['ground'] = None
    result = audit(document, stages)['rows'][0]['cad']
    assert result['observed'][0]['status'] == 'absent'
    assert result['observed'][0]['reasons'] == ['floor_plane_missing']


def test_missing_observed_source_and_corrupt_mesh_bytes_fail_closed(tmp_path):
    document, stages = prepared(tmp_path)
    observed = document['entities'][0]['representations'][1]
    observed['planProjection'].pop('observationId')
    result = audit(document, stages)['rows'][0]['cad']
    assert result['model']['status'] == 'validated'
    assert result['observed'][0]['status'] == 'invalid'
    assert 'projection_source_mismatch' in result['observed'][0]['reasons']
    asset = stages.repo.get_asset(observed['assetId'])
    (stages.blobs.root / asset['storageKey']).write_bytes(b'corrupted')
    result = audit(document, stages)['rows'][0]['cad']
    assert result['model']['status'] == 'invalid'
    assert 'blob_integrity_error' in result['model']['reasons']


def test_reference_only_cad_validates_without_claiming_model(tmp_path):
    document, stages = prepared(tmp_path)
    entity = document['entities'][0]
    entity['activeModelRepresentationId'] = None
    entity['representations'][1]['sourceKind'] = 'observed_reference_surface'
    result = audit(document, stages)['rows'][0]
    assert result['category'] == 'reference' and result['cad']['status'] == 'validated'
    assert result['cad']['kind'] == 'observed_reference' and result['cad']['model']['status'] == 'absent'


def test_primitive_proof_pins_parameters_and_allows_unrecorded_legacy_model_cache_source(tmp_path):
    document, stages = prepared(tmp_path)
    model = document['entities'][0]['representations'][0]
    model.update(kind='primitive', assetId=None, primitive={'kind': 'box', 'dimensions': [1., 2., 3.]})
    reconstruction._refresh_plan_projections(document, stages)
    for key in ('observationId', 'observationRevision', 'imageId'):
        model['planProjection'][key] = None
    proof = audit(document, stages)['rows'][0]['cad']['model']
    assert proof['status'] == 'validated' and proof['assetId'] is None and not proof['meshBytesVerified']
    model['primitive']['dimensions'][0] = 2
    assert audit(document, stages)['rows'][0]['cad']['model']['reasons'] == ['projection_asset_mismatch']


def test_byte_verified_mesh_still_requires_valid_triangle_indices(tmp_path):
    document, stages = prepared(tmp_path)
    rep = document['entities'][0]['representations'][0]
    asset = next(asset for asset in stages.repo.assets if asset['id'] == rep['assetId'])
    declared = next(asset for asset in document['assets'] if asset['id'] == rep['assetId'])
    data = bytearray(stages.blobs.get(asset['storageKey'], asset['sha256'], asset['sizeBytes']))
    np.frombuffer(data, dtype='<u4', offset=declared['byteLayout']['indexByteOffset'])[0] = 1_000_000
    changed = stages.blobs.put(bytes(data))
    asset.update(changed)
    declared.update(changed)
    rep['planProjection']['assetSha256'] = changed['sha256']
    assert audit(document, stages)['rows'][0]['cad']['model']['reasons'] == ['invalid_mesh']


def test_scene_asset_hash_and_missing_cache_are_explicit(tmp_path):
    document, stages = prepared(tmp_path)
    model = document['entities'][0]['representations'][0]
    saved = model.pop('planProjection')
    assert audit(document, stages)['rows'][0]['cad']['model']['reasons'] == ['no_current_projection_cache']
    model['planProjection'] = saved
    next(asset for asset in document['assets'] if asset['id'] == model['assetId'])['sha256'] = '0' * 64
    assert audit(document, stages)['rows'][0]['cad']['model']['reasons'] == ['scene_asset_hash_mismatch']


def test_composite_requires_own_observed_and_every_mapped_model_cad(tmp_path):
    document, stages = prepared(tmp_path)
    observed = deepcopy(document['entities'][0]['representations'][1])
    observed.update(id='boundary-surface', sourceRefs=[{'observationId': 'boundary', 'revision': 2, 'imageId': 'image'}])
    document['observations'].append({'id': 'boundary', 'revision': 2, 'imageId': 'image'})
    document['entities'].append({'id': 'composite', 'observationRefs': ['boundary'], 'representations': [observed],
                                  'activeModelRepresentationId': None})
    document['annotations'].append({'id': 'mapping', 'kind': 'observation_component_mapping', 'entityId': 'composite',
        'representationType': 'composite_source_evidence', 'independentObject': False,
        'unresolvedBoundaryObservationId': 'boundary', 'unresolvedBoundaryObservationRevision': 2,
        'targets': [{'entityId': 'object', 'activeModelRepresentationId': 'model', 'observationId': 'observation', 'observationRevision': 2}]})
    reconstruction._refresh_plan_projections(document, stages)
    result = audit(document, stages)['rows'][1]
    assert result['category'] == 'composite' and result['cad']['kind'] == 'component_mapping'
    assert result['cad']['status'] == 'validated' and result['cad']['model']['status'] == 'absent'
    document['entities'][0]['representations'][0]['planProjection']['lines'] = []
    result = audit(document, stages)['rows'][1]['cad']
    assert result['status'] == 'invalid' and result['observed'][0]['status'] == 'validated'


def test_empty_projection_and_stale_observed_are_not_current_cad(tmp_path):
    document, stages = prepared(tmp_path)
    model, observed = document['entities'][0]['representations']
    model['qualityEvidence'] = {'status': 'accepted'}
    model['planProjection'].update(polygons=[], lines=[])
    observed['sourceValidity'] = 'stale'
    result = audit(document, stages)['rows'][0]
    assert result['qualityStatus'] == 'accepted'
    assert result['cad']['model']['reasons'] == ['projection_geometry_empty']
    assert result['cad']['observed'] == [] and result['cad']['status'] == 'invalid'


def test_observed_geometry_cannot_claim_a_foreign_camera_frame(tmp_path):
    document, stages = prepared(tmp_path)
    observed = document['entities'][0]['representations'][1]
    document['coordinateFrames'].append({'id': 'foreign', 'ground': {'normal': [0., 0., 1.]}})
    observed['coordinateFrameId'] = observed['transform']['coordinateFrameId'] = 'foreign'
    reconstruction._refresh_plan_projections(document, stages)
    result = audit(document, stages)['rows'][0]['cad']
    assert result['model']['status'] == 'validated'
    assert result['observed'][0]['reasons'] == ['projection_source_mismatch']


@pytest.mark.parametrize('field', ['cameraId', 'imageSha256', 'geometrySolutionSha256'])
def test_observed_source_pins_cannot_disagree_with_scene(field, tmp_path):
    document, stages = prepared(tmp_path)
    observed = document['entities'][0]['representations'][1]
    observed['sourceRefs'][0][field] = 'foreign'
    result = audit(document, stages)['rows'][0]['cad']
    assert result['model']['status'] == 'validated'
    assert result['observed'][0]['reasons'] == ['projection_source_mismatch']
