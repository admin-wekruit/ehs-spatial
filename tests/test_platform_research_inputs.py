from copy import deepcopy
import io
from uuid import uuid4

import numpy as np
from PIL import Image
import pytest

from argus.platform.contracts import PlatformError, canonical, digest
from argus.platform.recgen import validate_frozen_source
from argus.platform.reconstruction import _Stages, _unpacked, run_analysis
from argus.platform.research_inputs import prepare_recgen_input
from argus.platform.storage import LocalBlobStore
from test_platform_reconstruction import Repo, bundle, provider
from test_recgen_research import payload, research_configuration


@pytest.fixture
def source(tmp_path):
    blobs = LocalBlobStore(tmp_path / 'blobs')
    repo = Repo(blobs, size=(24, 32))
    providers = bundle(repo)
    providers['discovery'] = provider('discovery', lambda _: {'items': [
        {'label': 'new object', 'box': [4, 4, 28, 20]}]})
    mask = np.zeros((24, 32), bool)
    mask[4:20, 4:28] = True
    providers['segmentation'] = provider('segmentation', lambda _: {'mask': mask})
    y, x = np.indices((12, 16))
    K = np.array([[12., 0, 7.5], [0, 12., 5.5], [0, 0, 1.]])
    mapping = np.array([[.5, 0, -.25], [0, .5, -.25], [0, 0, 1.]])
    points = np.stack(((x - 7.5) / 6, (y - 5.5) / 6, np.full(x.shape, 2.)), -1)
    providers['geometry'] = provider('geometry', lambda request: {'frames': [
        {'imageId': image['imageId'], 'points': points, 'valid': np.ones((12, 16), bool),
         'K': K, 'cameraToWorld': np.eye(4), 'rgb': np.full((12, 16, 3), 100, np.uint8),
         'inputToCanonical': mapping} for image in request['images']]})
    repo.document, result = run_analysis(repo, blobs, repo.job, providers)
    assert result['status'] == 'succeeded', result
    entity = next(e for e in repo.document['entities'] if len(e.get('observationRefs', [])) == 2)
    manifest, protocol = research_configuration(payload(), [])
    protocol['runtimeManifest']['generation']['inputContractVersion'] = 'recgen-input-v2'
    preparation = {key: protocol[key] for key in ('purpose', 'metricDefinitions', 'policyThresholds',
                                                 'split', 'callLimits', 'runtimeManifest')}
    preparation.update(authority={'source': 'database_admin', 'databaseRole': 'test_admin'},
                       budgetAtPreparation={'configuredBudgetUsd': '1', 'spentOrReservedUsd': '0'})
    job = {**repo.job, 'branchId': 'reviewed-branch', 'kind': 'generate_scene',
           'inputs': {'captureId': 'browser-cannot-replace-frozen-capture'},
           'config': {'researchPreparation': preparation}}
    return repo, blobs, job, entity, manifest, mapping, K, mask


def prepare(source, **kwargs):
    repo, blobs, job, entity, manifest, *_ = source
    ids = list(reversed(entity['observationRefs']))
    return prepare_recgen_input(repo, blobs, job, entity['id'], ids, ids[0], manifest, **kwargs)


def test_original_non_square_pixels_and_shared_frame_are_frozen_without_side_effects(source, monkeypatch, tmp_path):
    repo, blobs, job, entity, manifest, mapping, K, mask = source
    before = deepcopy((repo.document, repo.assets, repo.calls, repo.events, job, manifest))
    def forbidden(*args, **kwargs):
        pytest.fail('Preparation must never write or invoke a model')
    monkeypatch.setattr(repo, 'register_asset', forbidden)
    monkeypatch.setattr(repo, 'reserve_model_call', forbidden)
    monkeypatch.setattr(blobs, 'put', forbidden)
    monkeypatch.setattr(_Stages, 'put', forbidden)
    monkeypatch.setattr(_Stages, 'call', forbidden)
    envelope = prepare(source, seed=17)
    frozen = envelope['validation']
    value = _unpacked(frozen['payload'])
    request = validate_frozen_source(value, frozen['protocol'], repo.document)
    assert envelope['sha256'] == digest(frozen)
    assert frozen['baseDocumentSha256'] == digest(repo.document)
    assert frozen['protocol']['callLimits'] == job['config']['researchPreparation']['callLimits']
    assert value['seed'] == 17 and value['maskErosionEnabled'] is True
    assert len(request.views) == 2 and len({v.coordinate_frame_id for v in request.views}) == 1
    assert [v.observation_id for v in request.views] == list(reversed(entity['observationRefs']))
    for view in request.views:
        crop = view.pixel_mapping
        assert crop['sourceShapeHW'] == [24, 32] and crop['canonicalShapeHW'] == [12, 16]
        assert crop['maskSource'] == 'original_pixels'
        np.testing.assert_array_equal(crop['sourceToCanonical'], mapping)
        left, top, right, bottom = crop['sourceCropXYXY']
        y, x = np.indices(view.mask.shape)
        source_x, source_y = x + left, y + top
        valid = (source_x >= 0) & (source_x < 32) & (source_y >= 0) & (source_y < 24)
        expected = np.zeros(view.mask.shape, bool)
        expected[valid] = mask[source_y[valid], source_x[valid]]
        np.testing.assert_array_equal(view.mask, expected)
        assert np.unique(view.rgb[valid]).tolist() in ([40], [41]), 'Original RGB, never canonical substitute'
        source_pixel = np.array([13., 9., 1.])
        crop_matrix = np.array(crop['matrix'])
        np.testing.assert_allclose(np.linalg.inv(view.K) @ crop_matrix @ source_pixel,
                                   np.linalg.inv(K) @ mapping @ source_pixel)
        assert np.all(view.depth[view.mask] == 2)
    monkeypatch.chdir(tmp_path)
    assert prepare(source, seed=17) == envelope, 'Working directory and UI capture selection cannot change frozen inputs'
    assert (repo.document, repo.assets, repo.calls, repo.events, job, manifest) == before
    frozen['protocol']['callLimits']['maxCalls'] = 99
    assert job['config']['researchPreparation']['callLimits']['maxCalls'] == 1


def test_observation_owned_by_another_entity_is_rejected(source):
    repo, blobs, job, entity, manifest, *_ = source
    oid = entity['observationRefs'][0]
    repo.document['entities'].append({'id': 'other', 'observationRefs': [oid]})
    with pytest.raises(PlatformError, match='recgen_observation_selection_invalid'):
        prepare_recgen_input(repo, blobs, job, entity['id'], [oid], oid, manifest)


def test_envelope_is_accepted_by_existing_research_job_loader(source, monkeypatch):
    from argus.platform import reconstruction
    repo, blobs, job, *_ = source
    frozen = prepare(source)['validation']
    asset = repo.register_asset(repo.pid, {**blobs.put(canonical(frozen), 'application/json'),
                                          'metadata': {'kind': 'recgen_validation_input'}})
    research_job = {**job, 'kind': 'validate_model',
                    'inputs': {'validationAssetId': asset['id'], 'validationSha256': asset['sha256']},
                    'config': {'researchProtocolSha256': digest(frozen['protocol'])}}
    def inspect(repository, storage, incoming, stage, value, images, manifest, protocol):
        assert repository is repo and storage is blobs and incoming == research_job
        assert stage == 'generation' and protocol == frozen['protocol']
        assert images == frozen['images'] and manifest == frozen['providerManifest']
        validate_frozen_source(value, protocol, repo.document)
        return {'status': 'validated_without_dispatch', 'newModelCalls': 0}
    monkeypatch.setattr(reconstruction, 'run_research_stage', inspect)
    assert reconstruction.run_research_job(repo, blobs, research_job) == {
        'status': 'validated_without_dispatch', 'newModelCalls': 0}


@pytest.mark.parametrize(('width', 'erosion'), [(6, True), (2, False)])
def test_source_crop_support_is_validated_after_original_mask_sampling(source, width, erosion):
    repo, blobs, _, entity, *_ = source
    mask = np.zeros((24, 32), bool)
    mask[4:20, 10:10 + width] = True
    stream = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format='PNG')
    asset = repo.register_asset(repo.pid, blobs.put(stream.getvalue(), 'image/png'))
    repo.document['assets'].append(asset)
    for observation in repo.document['observations']:
        if observation['id'] in entity['observationRefs']:
            observation['maskAssetId'] = asset['id']
    frozen = prepare(source, mask_erosion_enabled=erosion)['validation']
    value = _unpacked(frozen['payload'])
    assert value['maskErosionEnabled'] is frozen['protocol']['maskErosionEnabled'] is erosion
    assert all(np.count_nonzero(view['mask']) == 16 * width for view in value['views'])
    assert validate_frozen_source(value, frozen['protocol'], repo.document)


@pytest.mark.parametrize('asset_kind', ['image', 'mask', 'geometry'])
def test_revision_asset_hash_mismatch_rejected(source, asset_kind):
    repo, _, _, entity, *_ = source
    observation = next(o for o in repo.document['observations'] if o['id'] == entity['observationRefs'][0])
    identity = {'image': observation['imageId'], 'mask': observation['maskAssetId'],
                'geometry': repo.document['geometryBindings'][observation['imageId']]['geometrySolutionId']}[asset_kind]
    next(a for a in repo.document['assets'] if a['id'] == identity)['sha256'] = 'f' * 64
    with pytest.raises(PlatformError, match='research_input_hash_mismatch'):
        prepare(source)


def test_fork_inherited_assets_are_authorized_by_the_frozen_revision(source):
    repo = source[0]
    original_project = str(uuid4())
    for asset in repo.assets:
        asset['projectId'] = original_project
    frozen = prepare(source)['validation']
    assert frozen['projectId'] == repo.pid
    assert validate_frozen_source(_unpacked(frozen['payload']), frozen['protocol'], repo.document)


def test_foreign_asset_absent_from_frozen_revision_is_rejected(source):
    repo, _, _, entity, *_ = source
    observation = next(o for o in repo.document['observations'] if o['id'] == entity['observationRefs'][0])
    forged = {**repo.get_asset(observation['maskAssetId']), 'id': str(uuid4()), 'projectId': str(uuid4())}
    repo.assets.append(forged)
    observation['maskAssetId'] = forged['id']
    with pytest.raises(PlatformError, match='research_input_hash_mismatch'):
        prepare(source)


def test_canonical_mask_requires_proven_mapping_and_preserves_sampling(source):
    repo, blobs, _, entity, _, mapping, *_ = source
    observation = next(o for o in repo.document['observations'] if o['id'] == entity['observationRefs'][0])
    mask = np.zeros((12, 16), bool)
    mask[2:10, 2:14] = True
    stream = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format='PNG')
    asset = repo.register_asset(repo.pid, blobs.put(stream.getvalue(), 'image/png'))
    repo.document['assets'].append(asset)
    observation['maskAssetId'] = asset['id']
    observation.pop('maskPolygonization', None)
    observation['pixelMapping'] = [{'source': 'canonical_pixels', 'target': 'original_pixels',
                                    'matrix': np.linalg.inv(mapping).tolist()}]
    frozen = prepare(source)['validation']
    view = next(v for v in _unpacked(frozen['payload'])['views'] if v['observationId'] == observation['id'])
    assert view['pixelMapping']['maskSource'] == 'canonical_nearest_upsample'
    observation['pixelMapping'] = []
    with pytest.raises(PlatformError):
        prepare(source)


@pytest.mark.parametrize('change', ['missing_proof', 'multiple_calls', 'insufficient_cap', 'unfrozen_runtime'])
def test_server_verified_research_bounds_are_required(source, change):
    job = source[2]
    preparation = job['config']['researchPreparation']
    if change == 'missing_proof': job['config'].clear()
    if change == 'multiple_calls': preparation['callLimits']['maxCalls'] = 2
    if change == 'insufficient_cap': preparation['callLimits']['maxCostPerCallUsd'] = .001
    if change == 'unfrozen_runtime': preparation['runtimeManifest']['generation'].pop('runtimeAuditSha256')
    with pytest.raises(PlatformError):
        prepare(source)
