"""Pure RecGen multiview checks; no provider or GPU access."""
from copy import deepcopy
import io

import numpy as np
import pytest

from ehs_spatial.platform.contracts import PlatformError, digest
from ehs_spatial.platform import recgen
from ehs_spatial.platform.reconstruction import _packed
from ehs_spatial.platform.spatial import transform_points


def payload():
    views = []
    for index in range(2):
        camera = np.eye(4)
        camera[:3, 3] = [index + 3, 4, 5]
        views.append({'observationId':f'obs-{index}', 'observationRevision':2,
            'imageId':f'image-{index}', 'imageSha256':str(index + 1) * 64,
            'maskSha256':str(index + 3) * 64, 'geometrySolutionSha256':'5' * 64,
            'coordinateFrameId':'native-frame', 'rgb':np.full((8, 9, 3), 80 + index, np.uint8),
            'depth':np.full((8, 9), 2, np.float32), 'mask':np.ones((8, 9), bool),
            'K':np.array([[10., 0, 4], [0, 10, 3.5], [0, 0, 1]]), 'cameraToWorld':camera})
    return {'entityId':'entity', 'anchorObservationId':'obs-0', 'views':views, 'seed':42}


def test_npz_preserves_every_view_with_explicit_anchor_and_exact_arrays():
    value = payload()
    request = recgen.RecGenRequest.from_payload(value)
    raw = request.to_npz()
    assert raw == request.to_npz(), 'The transmitted byte digest must be reproducible'
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        assert arrays['view_count'] == 2
        for index, view in enumerate(value['views']):
            for target, source in [('rgb','rgb'), ('depth','depth'), ('camera_intrinsics','K')]:
                np.testing.assert_array_equal(arrays[f'{index}_{target}'], view[source])
            np.testing.assert_array_equal(arrays[f'{index}_mask'], view['mask'].astype(np.uint8) * 255)
    changed = deepcopy(value)
    changed['views'][1]['observationRevision'] += 1
    assert digest(_packed(changed)) != digest(_packed(value))
    for field in ('imageSha256', 'maskSha256', 'geometrySolutionSha256'):
        changed = deepcopy(value)
        changed['views'][1][field] = 'f' * 64
        assert digest(_packed(changed)) != digest(_packed(value))


@pytest.mark.parametrize('failure', ['anchor', 'duplicate', 'rgb', 'mask', 'depth', 'depth_units', 'K', 'camera', 'frame', 'hash', 'revision', 'unknown'])
def test_typed_multiview_rejects_invalid_input(failure):
    value = payload()
    view = value['views'][1]
    if failure == 'anchor': value['anchorObservationId'] = 'obs-1'
    if failure == 'duplicate': view['observationId'] = 'obs-0'
    if failure == 'rgb': view['rgb'] = view['rgb'].astype(np.float32)
    if failure == 'mask': view['mask'][:] = False
    if failure == 'depth': view['depth'][0, 0] = np.nan
    if failure == 'depth_units': view['depth'][0, 0] = 31
    if failure == 'K': view['K'][0, 0] = 0
    if failure == 'camera': view['cameraToWorld'][0, 0] = 2
    if failure == 'frame': view['coordinateFrameId'] = 'unregistered-frame'
    if failure == 'hash': view['maskSha256'] = 'mutable'
    if failure == 'revision': view['observationRevision'] = True
    if failure == 'unknown': view['untrackedField'] = 1
    with pytest.raises(PlatformError):
        recgen.RecGenRequest.from_payload(value)


def test_pose_uses_raw_mesh_and_anchor_camera_once_with_nonunit_scale():
    request = recgen.RecGenRequest.from_payload(payload())
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 1]])
    object_to_camera = np.diag([.25, .5, .75, 1.])
    object_to_camera[:3, 3] = [1, 2, 3]
    response = {'vertices':vertices, 'faces':np.array([[0, 1, 2]]),
        'officialPosedVertices':transform_points(vertices, object_to_camera),
        'objectToCamera':object_to_camera, 'pins':recgen.RECGEN_PINS}
    output = recgen.adapt_output(request, response)
    np.testing.assert_array_equal(output['vertices'], vertices)
    np.testing.assert_allclose(output['proposedObjectToNative'], request.views[0].camera_to_world @ object_to_camera)
    np.testing.assert_allclose(transform_points(output['vertices'], output['proposedObjectToNative']),
        transform_points(response['officialPosedVertices'], request.views[0].camera_to_world))
    assert output['provenance']['shapeStatus'] == 'research_only'
    assert output['provenance']['metricScaleKnown'] is False
    response['officialPosedVertices'] = vertices
    with pytest.raises(PlatformError, match='recgen_pose_mismatch'):
        recgen.adapt_output(request, response)


def test_original_crop_preserves_source_pixels_mask_detail_and_camera_rays():
    rgb = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)
    source_mask = np.zeros((12, 10), bool)
    source_mask[3:9, 2:8] = True
    source_mask[4, 3] = False
    mapping = np.array([[.5, 0, -.25], [0, .25, -.375], [0, 0, 1]])
    K = np.array([[10., 0, 2], [0, 20, 1], [0, 0, 1]])
    cropped = recgen.source_grid_crop(rgb, source_mask, np.full((3, 5), 2, np.float32), K, mapping)
    from PIL import Image
    pixel_map = cropped['pixelMapping']
    box = pixel_map['sourceCropXYXY']
    np.testing.assert_array_equal(cropped['rgb'], np.asarray(Image.fromarray(rgb).crop(box)))
    np.testing.assert_array_equal(cropped['mask'], np.asarray(Image.fromarray(source_mask).crop(box)))
    yy, xx = np.indices(cropped['depth'].shape)
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    original = pixels @ np.linalg.inv(np.asarray(pixel_map['matrix'])).T
    canonical = original @ mapping.T
    np.testing.assert_allclose(pixels @ np.linalg.inv(cropped['K']).T, canonical @ np.linalg.inv(K).T, atol=1e-6)
    assert np.all(cropped['depth'][cropped['mask']] == 2)
    changed = payload()
    # Mapping provenance participates in the frozen request hash.
    changed['views'][0]['pixelMapping'] = pixel_map
    assert digest(_packed(changed)) != digest(_packed(payload()))
    with pytest.raises(PlatformError):
        recgen.source_grid_crop(rgb, source_mask[:-1], np.ones((3, 5), np.float32), K, mapping)


def test_original_crop_request_retains_explicit_pixel_mapping():
    value = payload()
    view = value['views'][0]
    source_rgb = np.full((16, 18, 3), 91, np.uint8)
    source_mask = np.ones((16, 18), bool)
    view.update(recgen.source_grid_crop(source_rgb, source_mask, view['depth'], view['K'],
                np.array([[.5, 0, -.25], [0, .5, -.25], [0, 0, 1]])))
    request = recgen.RecGenRequest.from_payload(value)
    assert request.views[0].pixel_mapping == view['pixelMapping']
    view['pixelMapping']['matrix'][0][0] = 2
    with pytest.raises(PlatformError, match='recgen_pixel_mapping_invalid'):
        recgen.RecGenRequest.from_payload(value)


def test_build_payload_selects_only_reviewed_same_entity_views_and_derives_camera_z(tmp_path):
    from test_platform_reconstruction import Repo, bundle
    from ehs_spatial.platform.storage import LocalBlobStore
    from ehs_spatial.platform.reconstruction import run_analysis, _Stages, _capture, _load_geometry, _load_masks

    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    repo.document, _ = run_analysis(repo, blobs, repo.job, bundle(repo))
    job = {**repo.job, 'kind':'validate_model'}
    _, document, images = _capture(repo, blobs, job)
    stages = _Stages(repo, blobs, job, {})
    frames, records = _load_geometry(document, images, stages)
    masks, _ = _load_masks(document, records, stages)
    entity = next(e for e in document['entities'] if len(e.get('observationRefs', [])) == 2)
    approved = list(reversed(entity['observationRefs']))
    built = recgen.build_payload(document, entity['id'], approved, approved[0], images, frames, records, masks, repo.get_asset)
    assert [v['observationId'] for v in built['views']] == approved
    assert built['anchorObservationId'] == approved[0]
    for view in built['views']:
        np.testing.assert_array_equal(view['depth'], np.where(frames[view['imageId']].support(), 2, 0))
    with pytest.raises(PlatformError, match='recgen_observation_selection_invalid'):
        recgen.build_payload(document, entity['id'], ['unknown'], 'unknown', images, frames, records, masks, repo.get_asset)
    masks.pop(approved[-1])
    with pytest.raises(PlatformError, match='recgen_mask_required'):
        recgen.build_payload(document, entity['id'], approved, approved[0], images, frames, records, masks, repo.get_asset)


def research_configuration(value, images):
    runtime = {'pins':recgen.RECGEN_PINS, 'distribution':'recgen_inference',
        'modalFunctionId':'fu-reviewed', 'weightsManifestSha256':recgen.RECGEN_WEIGHTS_SHA256,
        'runtimeAuditSha256':'a' * 64}
    manifest = {'generation':{'provider':'modal-recgen-research', 'pins':recgen.RECGEN_PINS,
        'estimatedCostUsd':.01, 'modalApp':'lucida-private-assets', 'modalFunction':'generate_object',
        'modalFunctionId':'fu-reviewed', 'modalVolume':'panoptes-lucida-weights',
        'releaseEvidence':{'pins':recgen.RECGEN_PINS,
            'license':{'status':'passed', 'artifactSha256':'b' * 64, 'scope':'noncommercial_research'},
            'quality':{'status':'unverified'}, 'runtime':{'status':'unverified'}}}}
    protocol = {'id':'recgen-research-test', 'purpose':'runtime_validation',
        'baselineRevision':'baseline', 'entityId':value['entityId'], 'split':'research',
        'metricDefinitions':{'poseResidual':'native pose maximum vertex error'}, 'policyThresholds':{},
        'inputHashes':[image['sha256'] for image in images],
        'inputAssetHashes':[{'assetId':image['id'], 'sha256':image['sha256']} for image in images],
        'payloadSha256':digest(_packed(value)), 'providerManifestSha256':digest(manifest),
        'anchorObservationId':value['anchorObservationId'],
        'observationIds':[v['observationId'] for v in value['views']],
        'callLimits':{'maxCalls':1, 'maxCostPerCallUsd':.01, 'maxTotalCostUsd':.01},
        'runtimeManifest':{'generation':runtime}}
    return manifest, protocol


def test_recgen_factory_is_research_only_and_does_not_relax_sam3d_gates():
    from ehs_spatial.platform.reconstruction import providers_from_manifest, validate_research_manifest
    value = payload()
    images = [{'id':v['imageId'], 'sha256':v['imageSha256']} for v in value['views']]
    manifest, protocol = research_configuration(value, images)
    with pytest.raises(PlatformError, match='generation_requires_sam3d'):
        providers_from_manifest(manifest)
    validate_research_manifest(protocol, manifest)
    provider = providers_from_manifest(manifest, _research=True)['generation']
    provider.validate('generation', research_protocol=protocol)
    with pytest.raises(PlatformError):
        provider.validate('generation')
    protocol['runtimeManifest']['generation']['weightsManifestSha256'] = 'c' * 64
    with pytest.raises(PlatformError, match='research_runtime_unpinned'):
        validate_research_manifest(protocol, manifest)


@pytest.mark.parametrize('failure', ['schema', 'selection', 'source_revision', 'source_image', 'source_mask', 'source_geometry'])
def test_invalid_frozen_input_never_reaches_call_reservation(tmp_path, monkeypatch, failure):
    from test_platform_reconstruction import Repo
    from ehs_spatial.platform.storage import LocalBlobStore
    from ehs_spatial.platform.reconstruction import run_research_stage
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    value = payload()
    images = [{'id':v['imageId'], 'sha256':v['imageSha256']} for v in value['views']]
    manifest, protocol = research_configuration(value, images)
    protocol['baselineRevision'] = repo.rid
    repo.document.update(entities=[{'id':'entity', 'observationRefs':['obs-0','obs-1']}],
        observations=[{'id':v['observationId'], 'revision':v['observationRevision'], 'imageId':v['imageId'],
                       'maskAssetId':f'mask-{i}'} for i, v in enumerate(value['views'])],
        assets=[{'id':v['imageId'], 'sha256':v['imageSha256']} for v in value['views']] +
               [{'id':f'mask-{i}', 'sha256':v['maskSha256']} for i,v in enumerate(value['views'])] +
               [{'id':'geometry', 'sha256':'5' * 64}],
        geometryBindings={v['imageId']:{'geometrySolutionId':'geometry'} for v in value['views']})
    protocol['inputAssetHashes'] = [{'assetId':a['id'], 'sha256':a['sha256']} for a in repo.document['assets']]
    if failure == 'schema': value['views'][1]['mask'][:] = False
    if failure == 'selection': protocol['observationIds'] = ['obs-0']
    if failure == 'source_revision': repo.document['observations'][1]['revision'] += 1
    if failure == 'source_image': repo.document['assets'][1]['sha256'] = 'f' * 64
    if failure == 'source_mask': repo.document['assets'][3]['sha256'] = 'f' * 64
    if failure == 'source_geometry': repo.document['assets'][4]['sha256'] = 'f' * 64
    protocol['payloadSha256'] = digest(_packed(value))
    job = {**repo.job, 'kind':'validate_model', 'config':{'researchProtocolSha256':digest(protocol)}}
    monkeypatch.setattr(repo, 'reserve_model_call', lambda *a, **kw: pytest.fail('Invalid input reserved a paid call'))
    with pytest.raises(PlatformError):
        run_research_stage(repo, blobs, job, 'generation', value, images, manifest, protocol)
    assert not repo.calls


def test_research_factory_retains_received_usage_and_native_scene_assembly(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from test_platform_reconstruction import Repo, bundle
    from ehs_spatial.platform.storage import LocalBlobStore
    from ehs_spatial.platform.reconstruction import (
        run_analysis, run_research_stage, _Stages, _capture, _load_geometry, _load_masks,
        _record_generated_representation,
    )
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    repo.document, _ = run_analysis(repo, blobs, repo.job, bundle(repo))
    repo.paid_budget = 10
    _, document, images = _capture(repo, blobs, {**repo.job, 'kind':'validate_model'})
    stages = _Stages(repo, blobs, repo.job, {})
    frames, records = _load_geometry(document, images, stages)
    masks, _ = _load_masks(document, records, stages)
    entity = next(e for e in document['entities'] if len(e.get('observationRefs', [])) == 2)
    ids = entity['observationRefs']
    value = recgen.build_payload(document, entity['id'], ids, ids[0], images, frames, records, masks, repo.get_asset)
    selected_images = [next(i for i in images if i['id'] == v['imageId']) for v in value['views']]
    manifest, protocol = research_configuration(value, selected_images)
    protocol.update(baselineRevision=repo.rid, inputAssetHashes=[{'assetId':a['id'], 'sha256':a['sha256']} for a in document['assets']])
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 1]])
    pose = np.diag([.25, .5, .75, 1.])
    calls = []
    ownership_checks = []
    monkeypatch.setattr(repo, 'heartbeat_job', lambda *args:ownership_checks.append(args) or True, raising=False)
    def invoke(payload, config, *, is_current):
        assert is_current()
        calls.append(payload)
        return {'vertices':vertices, 'faces':np.array([[0, 1, 2]]), 'objectToCamera':pose,
            'colors':np.ones((3, 4)),
            'officialPosedVertices':transform_points(vertices, pose), 'pins':recgen.RECGEN_PINS,
            'telemetry':{'workerElapsedSeconds':2, 'actualCostUsd':None}, 'providerRequestId':'reviewed-call'}
    monkeypatch.setitem(sys.modules, 'ehs_spatial.platform.recgen_transport', SimpleNamespace(invoke=invoke))
    job = {**repo.job, 'kind':'validate_model', 'config':{'researchProtocolSha256':digest(protocol)}}
    before = deepcopy(repo.document)
    result = run_research_stage(repo, blobs, job, 'generation', value, selected_images, manifest, protocol)
    assert len(calls) == 1 and result['sceneRevision'] is None and repo.document == before
    assert ownership_checks == [(job['id'], job['attemptToken'])]
    assert repo.calls[-1]['response']['workerElapsedSeconds'] == 2
    assert repo.calls[-1]['response']['providerRequestId'] == 'reviewed-call'
    assert repo.calls[-1]['actual_cost'] is None
    evidence = repo.get_asset(result['outputAssetId'])
    output = stages.load(evidence)['output']
    _record_generated_representation(document, stages, entity, [next(o for o in document['observations'] if o['id'] == oid) for oid in ids],
                                     value['views'][0]['coordinateFrameId'], output, evidence)
    model = next(r for r in entity['representations'] if r['kind'] == 'generated_mesh')
    assert model['shapeStatus'] == 'research_only' and model['placementState'] == 'unconfirmed'
    assert model['provenance']['licenseScope'] == 'noncommercial_research'
    assert all({'observationId':oid, 'revision':1} in model['sourceRefs'] for oid in ids)
    mesh_asset = repo.get_asset(model['assetId'])
    assert mesh_asset['sizeBytes'] == 3 * 9 * 4 + 3 * 4, 'Native mesh format stores exactly position, normal, RGB'
    from ehs_spatial.platform.contracts import validate_document
    validate_document(document)
