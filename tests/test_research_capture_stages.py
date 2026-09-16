"""First geometry/depth validation uses owned photos and the existing research ledger."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import sys

import pytest

from ehs_spatial.platform.contracts import PlatformError, canonical, digest
from ehs_spatial.platform import reconstruction as reconstruction
from ehs_spatial.platform.storage import LocalBlobStore
from scripts.research import validate_sam3d as cli
from test_platform_backend import repo, project, identity
from test_platform_reconstruction import Repo, geometry_response, depth_response


def preparation(tmp_path, monkeypatch, stage):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    repo.paid_budget = .01
    monkeypatch.setattr(repo, '_connect', lambda: nullcontext(None), raising=False)
    monkeypatch.setattr(cli, 'admin_context', lambda *a: (
        {'source': 'database_admin'}, {'document': repo.document}))
    monkeypatch.setattr(cli, 'check_budget', lambda *a: {'configuredBudgetUsd': '.01'})
    pins = reconstruction.MAP_PINS if stage == 'geometry' else {
        'model': 'Ruicheng/moge-3-vitl', 'codeRevision': 'a' * 40,
        'modelRevision': 'b' * 40, 'adapter': reconstruction.PIPELINE_VERSION}
    # Synthetic license receipt exists only in this local test. Runtime/quality
    # remain unverified, and no manifest is deployed or real provider invoked.
    manifest = {stage: {'provider': 'modal', 'pins': pins, 'paid': True, 'estimatedCostUsd': .01,
        'modalApp': 'test-only', 'modalClass': 'Model', 'modalMethod': 'run',
        'releaseEvidence': {'pins': pins, 'license': {'status': 'passed', 'artifactSha256': 'c' * 64},
                            'runtime': {'status': 'unverified'}, 'quality': {'status': 'unverified'}}}}
    runtime = {stage: {'pins': pins, 'modalImageId': 'im-' + 'a' * 22,
                       'adapterSourceSha256': 'f' * 64,
                       'distribution': 'mapanything' if stage == 'geometry' else 'moge'}}
    protocol = {'id': 'capture-stage-fixture', 'stage': stage, 'purpose': 'runtime_validation',
        'projectId': repo.pid, 'branchId': 'test-branch', 'baselineRevision': repo.rid,
        'metricDefinitions': {'sourceGrid': 'source image and returned grid agree'},
        'policyThresholds': {}, 'split': 'test-only',
        'callLimits': {'maxCalls': 1, 'maxCostPerCallUsd': .01, 'maxTotalCostUsd': .01}}
    if stage == 'depth':
        protocol['imageIds'] = [repo.capture['images'][0]['id']]
    frozen = cli.prepare(protocol, repo, blobs, manifest, runtime)
    return repo, blobs, manifest, frozen


@pytest.mark.parametrize('stage', ['geometry', 'depth'])
def test_prepare_first_runtime_validation_without_an_entity_or_geometry(tmp_path, monkeypatch, stage):
    repo, blobs, manifest, frozen = preparation(tmp_path, monkeypatch, stage)
    assert not repo.document['entities'] and not repo.calls
    assert len(repo.assets) == 2
    assert frozen['baseDocumentSha256'] == digest(repo.document)
    assert frozen['protocol']['stage'] == stage and 'entityId' not in frozen['protocol']
    assert set(frozen['providerManifest']) == {stage}
    payloads = frozen['payload']['images'] if stage == 'geometry' else [frozen['payload']['image']]
    assert len(payloads) == (2 if stage == 'geometry' else 1)
    assert [p['imageId'] for p in payloads] == [i['id'] for i in frozen['images']]
    assert digest(frozen['payload']) == frozen['protocol']['payloadSha256']
    spec = reconstruction.providers_from_manifest(manifest, _research=True)[stage]
    spec.validate(stage, research_protocol=frozen['protocol'])
    with pytest.raises(PlatformError, match='provider_release_gate_unverified'):
        spec.validate(stage)


@pytest.mark.parametrize('stage', ['geometry', 'depth'])
@pytest.mark.parametrize('adapter_sha', [None, 'main', 'a' * 63])
def test_capture_runtime_requires_an_immutable_adapter_source_hash(tmp_path, monkeypatch, stage, adapter_sha):
    _, _, manifest, frozen = preparation(tmp_path, monkeypatch, stage)
    runtime = frozen['protocol']['runtimeManifest'][stage]
    if adapter_sha is None:
        runtime.pop('adapterSourceSha256', None)
    else:
        runtime['adapterSourceSha256'] = adapter_sha
    with pytest.raises(PlatformError, match='research_runtime_unpinned'):
        reconstruction._validate_research_runtime(frozen['protocol'], manifest[stage]['pins'], stage)


@pytest.mark.parametrize('changed', ['foreign_image', 'changed_pixels', 'two_stages', 'stage_mismatch', 'license', 'runtime_pin', 'source_hash', 'source_project', 'image_selection'])
def test_stage_validation_rejects_forged_inputs_or_gates(tmp_path, monkeypatch, changed):
    repo, blobs, manifest, frozen = preparation(tmp_path, monkeypatch, 'depth')
    frozen = deepcopy(frozen)
    if changed == 'foreign_image':
        frozen['images'][0]['id'] = 'foreign-image'
        frozen['payload']['image']['imageId'] = 'foreign-image'
    elif changed == 'changed_pixels':
        frozen['payload']['image']['dataUri'] += 'forged'
    elif changed == 'two_stages':
        frozen['providerManifest']['geometry'] = deepcopy(manifest['depth'])
    elif changed == 'stage_mismatch':
        frozen['protocol']['stage'] = 'geometry'
    elif changed == 'license':
        frozen['providerManifest']['depth']['releaseEvidence']['license']['status'] = 'unverified'
    elif changed == 'runtime_pin':
        runtime = frozen['protocol']['runtimeManifest']['depth']
        runtime['pins'] = {**runtime['pins'], 'codeRevision': 'd' * 40}
    elif changed == 'source_hash':
        repo.document['assets'][0]['sha256'] = 'f' * 64
    elif changed == 'source_project':
        repo.assets[0]['projectId'] = 'foreign-project'
    else:
        frozen['protocol']['imageIds'] = ['foreign-image']
    # Rehashing an attacker-controlled envelope must not establish source ownership.
    protocol = frozen['protocol']
    protocol['payloadSha256'] = digest(frozen['payload'])
    protocol['providerManifestSha256'] = digest(frozen['providerManifest'])
    job = {**repo.job, 'kind': 'validate_model', 'config': {'researchProtocolSha256': digest(protocol)}}
    with pytest.raises(PlatformError):
        reconstruction._validate_research_inputs(job, 'depth', frozen['payload'], frozen['images'],
            frozen['providerManifest'], protocol, repo.document, repository=repo, blobs=blobs)
        reconstruction.providers_from_manifest(frozen['providerManifest'], _research=True)['depth'].validate(
            'depth', research_protocol=protocol)
    assert repo.calls == []


@pytest.mark.parametrize('stage', ['geometry', 'depth'])
def test_frozen_stage_loader_and_dispatch_leave_scene_and_release_unchanged(tmp_path, monkeypatch, stage):
    repo, blobs, manifest, frozen = preparation(tmp_path, monkeypatch, stage)
    before = digest(repo.document)
    blob = blobs.put(canonical(frozen), 'application/json')
    blob['metadata'] = {'kind': 'stage_validation_input', 'stage': stage, 'scope': 'research_only'}
    asset = repo.register_asset(repo.pid, blob)
    spec = reconstruction.providers_from_manifest(manifest, _research=True)[stage]
    seen = []
    def invoke(payload):
        seen.append(payload)
        return geometry_response(payload['images']) if stage == 'geometry' else depth_response(payload['image'])
    monkeypatch.setattr(reconstruction, 'providers_from_manifest', lambda *a, **kw: {
        stage: replace(spec, invoke=invoke, records_dispatch=False)})
    job = {**repo.job, 'kind': 'validate_model', 'branchId': frozen['branchId'],
        'inputs': {'validationAssetId': asset['id'], 'validationSha256': asset['sha256']},
        'config': {'researchProtocolSha256': digest(frozen['protocol'])}}
    result = reconstruction.run_research_job(repo, blobs, job)
    assert result['scope'] == 'research_only' and result['sceneRevision'] is None
    assert result['productReleaseStatus'] == 'not_changed'
    assert result['stages'][0]['stage'] == stage and len(seen) == len(repo.calls) == 1
    assert seen[0]['_researchProtocol']['runtimeManifest'] == frozen['protocol']['runtimeManifest']
    assert digest(repo.document) == before
    assert manifest[stage]['releaseEvidence']['quality']['status'] == 'unverified'
    again = reconstruction.run_research_job(repo, blobs, job)
    assert again['newModelCalls'] == 0 and len(seen) == len(repo.calls) == 1


@pytest.mark.parametrize('stage', ['geometry', 'depth'])
def test_modal_stage_binds_the_reviewed_runtime_hash(tmp_path, monkeypatch, stage):
    _, _, manifest, frozen = preparation(tmp_path, monkeypatch, stage)
    protocol = frozen['protocol']
    expected = digest(protocol['runtimeManifest'][stage])
    received = []
    response = {'pins': manifest[stage]['pins'], 'runtimeManifestSha256': expected}
    def spawn(payload, **kwargs):
        received.append(kwargs)
        return SimpleNamespace(object_id='fc-test', get=lambda:response)
    monkeypatch.setitem(sys.modules, 'modal', SimpleNamespace(Cls=SimpleNamespace(
        from_name=lambda *a: lambda: SimpleNamespace(run=SimpleNamespace(spawn=spawn)))))
    provider = reconstruction.providers_from_manifest(manifest, _research=True)[stage]
    receipts = []
    provider.invoke({**frozen['payload'], '_researchProtocol': protocol}, on_dispatched=receipts.append)
    assert received == [{'expectedRuntimeManifestSha256': expected}]
    response['runtimeManifestSha256'] = 'e' * 64
    with pytest.raises(reconstruction.ProviderResponseError):
        provider.invoke({**frozen['payload'], '_researchProtocol': protocol}, on_dispatched=receipts.append)
    assert receipts == ['fc-test', 'fc-test']


@pytest.mark.parametrize('stage', ['geometry', 'depth'])
@pytest.mark.parametrize('unknown', [False, True])
def test_admin_stage_submit_worker_and_ledger_are_idempotent(repo, tmp_path, monkeypatch, stage, unknown):
    from decimal import Decimal
    from panoptes_worker.__main__ import run_job
    with monkeypatch.context() as local:
        source, blobs, manifest, template = preparation(tmp_path, local, stage)
    repo.blobs = blobs
    cap, scene = project(repo)
    pid, bid = scene['project']['id'], scene['branch']['id']
    images = [{k: asset[k] for k in ('storageKey', 'sha256', 'sizeBytes', 'mediaType')} for asset in source.assets]
    for image in images:
        image['metadata'] = {'width': 12, 'height': 12, 'pixelMapping': []}
    capture = repo.create_capture(pid, cap, {'requestId': identity(), 'branchId': bid,
        'baseRevisionId': scene['revision']['id'], 'target': 'scene'}, images)
    protocol = {k: template['protocol'][k] for k in (
        'id', 'stage', 'purpose', 'metricDefinitions', 'policyThresholds', 'split', 'callLimits')}
    protocol.update(projectId=pid, branchId=bid, baselineRevision=capture['revision']['id'])
    if stage == 'depth':
        protocol['imageIds'] = [capture['capture']['images'][0]['id']]
    frozen = cli.prepare(protocol, repo, blobs, manifest, template['protocol']['runtimeManifest'])
    prepared = {'validation': frozen, 'sha256': digest(frozen)}
    repo.paid_budget = Decimal('.009')
    with pytest.raises(PlatformError, match='paid_budget_exceeded'):
        cli.submit(prepared, repo, blobs)
    repo.paid_budget = Decimal('.01')
    job = cli.submit(prepared, repo, blobs)
    assert cli.submit(prepared, repo, blobs)['id'] == job['id']
    seen = []
    def spawn(payload, **options):
        seen.append(payload)
        def get():
            with repo._connect() as connection:
                row = connection.execute('SELECT * FROM model_calls WHERE job_id=%s', (job['id'],)).fetchone()
            assert row['status'] == 'reserved' and row['response']['providerRequestId'] == 'fc-worker'
            if unknown:
                raise TimeoutError('provider result unavailable')
            output = geometry_response(payload['images']) if stage == 'geometry' else depth_response(payload['image'])
            return {**output, 'pins': manifest[stage]['pins'],
                    'runtimeManifestSha256': options['expectedRuntimeManifestSha256']}
        return SimpleNamespace(object_id='fc-worker', get=get, cancel=lambda **kw:None)
    monkeypatch.setitem(sys.modules, 'modal', SimpleNamespace(Cls=SimpleNamespace(
        from_name=lambda *a: lambda: SimpleNamespace(run=SimpleNamespace(spawn=spawn)))))
    result = run_job(repo, blobs, job['id'])
    assert result['status'] == ('outcome_unknown' if unknown else 'succeeded')
    assert result['result']['scope'] == 'research_only'
    assert result['resultRevisionId'] is None and not result['headAdvanced']
    assert repo.get_project(pid)['branches'][0]['headRevisionId'] == capture['revision']['id']
    run_job(repo, blobs, job['id'])
    assert len(seen) == 1
    with repo._connect() as connection:
        calls = connection.execute('SELECT * FROM model_calls WHERE job_id=%s', (job['id'],)).fetchall()
    assert len(calls) == 1 and calls[0]['response']['providerRequestId'] == 'fc-worker'
    assert manifest[stage]['releaseEvidence']['runtime']['status'] == 'unverified'


@pytest.mark.parametrize('stage', ['geometry', 'depth'])
def test_matching_pins_do_not_make_empty_stage_output_valid(tmp_path, monkeypatch, stage):
    repo, blobs, manifest, frozen = preparation(tmp_path, monkeypatch, stage)
    spec = reconstruction.providers_from_manifest(manifest, _research=True)[stage]
    monkeypatch.setattr(reconstruction, 'providers_from_manifest', lambda *a, **kw: {
        stage: replace(spec, invoke=lambda payload:{'pins':spec.pins}, records_dispatch=False)})
    job = {**repo.job, 'kind':'validate_model', 'config':{
        'researchProtocolSha256':digest(frozen['protocol'])}}
    before = digest(repo.document)
    for _ in range(2):
        with pytest.raises(PlatformError):
            reconstruction.run_research_stage(repo,blobs,job,stage,frozen['payload'],
                frozen['images'],manifest,frozen['protocol'])
    assert len(repo.calls) == 1  # Retain/reject known bad output; never rebill on replay.
    assert digest(repo.document) == before


def segmentation_source(tmp_path, monkeypatch, *, jpeg=False, masked=False):
    """A persisted discovery observation, before any reconstruction or segmentation."""
    import hashlib
    import io
    from pathlib import Path
    import numpy as np
    from PIL import Image

    blobs = LocalBlobStore(tmp_path)
    repository = Repo(blobs, size=(9, 13))
    repository.paid_budget = .01
    if jpeg:
        raw = io.BytesIO()
        Image.fromarray(np.full((9, 13, 3), 73, np.uint8)).save(raw, format='JPEG')
        old = repository.capture['images'][0]['id']
        asset = repository.register_asset(repository.pid, blobs.put(raw.getvalue(), 'image/jpeg'))
        repository.capture['images'][0].update(id=asset['id'], assetId=asset['id'])
        repository.document['assets'] = [a for a in repository.document['assets'] if a['id'] != old] + [asset]
    image = repository.capture['images'][0]
    source_asset = repository.get_asset(image['assetId'])
    reconstruction._discover(repository.document, image, {'items':[
        {'label':'small visible control', 'box':[1.2, 2.3, 10.4, 7.8]}]}, source_asset)
    entity = repository.document['entities'][0]
    observation = repository.document['observations'][0]
    if masked:
        raw = io.BytesIO()
        Image.fromarray(np.ones((9, 13), np.uint8) * 255).save(raw, format='PNG')
        asset = repository.register_asset(repository.pid, blobs.put(raw.getvalue(), 'image/png'))
        repository.document['assets'].append(asset)
        observation['maskAssetId'] = asset['id']
    monkeypatch.setattr(repository, '_connect', lambda: nullcontext(None), raising=False)
    monkeypatch.setattr(cli, 'admin_context', lambda *a: (
        {'source':'database_admin'}, {'document':repository.document}))
    monkeypatch.setattr(cli, 'check_budget', lambda *a: {'configuredBudgetUsd':'.01'})
    pins = {'model':'fal-ai/sam-3-1/image-rle', 'adapter':'sam3.1-text-box-pixel-coverage-v2'}
    manifest = {'segmentation':{'provider':'fal', 'pins':pins, 'paid':True, 'estimatedCostUsd':.01,
        'releaseEvidence':{'pins':pins, 'license':{'status':'passed', 'artifactSha256':'c' * 64},
            'runtime':{'status':'unverified'}, 'quality':{'status':'unverified'}}}}
    runtime = {'segmentation':{'pins':pins, 'provider':'fal', 'endpoint':pins['model'],
        'falClientVersion':'1.0.0',
        'adapterSourceSha256':hashlib.sha256(Path(reconstruction.__file__).read_bytes()).hexdigest()}}
    protocol = {'id':'segmentation-fixture', 'stage':'segmentation', 'purpose':'runtime_validation',
        'projectId':repository.pid, 'branchId':'test-branch', 'baselineRevision':repository.rid,
        'entityId':entity['id'], 'observationId':observation['id'],
        'metricDefinitions':{'sourceGrid':'boolean mask uses the submitted original photo grid'},
        'policyThresholds':{}, 'split':'test-only',
        'callLimits':{'maxCalls':1, 'maxCostPerCallUsd':.01, 'maxTotalCostUsd':.01}}
    return repository, blobs, manifest, runtime, protocol


@pytest.mark.parametrize('masked', [False, True])
def test_segmentation_prepares_owned_observation_without_geometry(tmp_path, monkeypatch, masked):
    repo, blobs, manifest, runtime, protocol = segmentation_source(tmp_path, monkeypatch, masked=masked)
    before = digest(repo.document)
    frozen = cli.prepare(protocol, repo, blobs, manifest, runtime)
    observation = repo.document['observations'][0]
    assert not repo.document['coordinateFrames'] and not repo.document['cameras'] and not repo.calls
    assert frozen['payload']['box'] == observation['originalPixelBox'] == [1.2, 2.3, 10.4, 7.8]
    assert frozen['payload']['submittedBox'] == [1, 2, 11, 8]
    assert set(frozen['payload']) == {'image', 'box', 'submittedBox', 'prompt', 'promptSource'}
    assert frozen['payload']['prompt'] == repo.document['entities'][0]['label']
    assert frozen['payload']['promptSource'] == {'kind':'entity_label', 'entityId':protocol['entityId'], 'label':frozen['payload']['prompt']}
    assert frozen['protocol']['sourceObservation'] == {
        'entityId':protocol['entityId'], 'observationId':observation['id'],
        'promptSource':frozen['payload']['promptSource'],
        **{k:observation[k] for k in ('revision','imageId','originalPixelBox','pixelMapping','maskAssetId')}}
    expected_ids = {observation['imageId']}
    if masked:
        expected_ids.add(observation['maskAssetId'])
    assert {r['assetId'] for r in frozen['protocol']['inputAssetHashes']} == expected_ids
    assert frozen['protocol']['runtimeManifest'] == runtime
    assert digest(repo.document) == before
    provider = reconstruction.providers_from_manifest(manifest, _research=True)['segmentation']
    assert provider.records_dispatch
    provider.validate('segmentation', research_protocol=frozen['protocol'])
    with pytest.raises(PlatformError, match='provider_release_gate_unverified'):
        provider.validate('segmentation')


@pytest.mark.parametrize('bad', ['missing_entity', 'missing_observation', 'image_selection', 'seed',
                                'foreign_entity', 'foreign_observation', 'unowned_observation',
                                'ambiguous_owner', 'context_owner'])
def test_segmentation_requires_exact_owned_observation_selection(tmp_path, monkeypatch, bad):
    repo, blobs, manifest, runtime, protocol = segmentation_source(tmp_path, monkeypatch)
    if bad == 'missing_entity': protocol.pop('entityId')
    elif bad == 'missing_observation': protocol.pop('observationId')
    elif bad == 'image_selection': protocol['imageIds'] = [repo.capture['images'][0]['id']]
    elif bad == 'seed': protocol['seed'] = 0
    elif bad == 'foreign_entity': protocol['entityId'] = 'foreign-entity'
    elif bad == 'foreign_observation': protocol['observationId'] = 'foreign-observation'
    elif bad == 'ambiguous_owner': repo.document['entities'].append({**repo.document['entities'][0], 'id':'other-owner'})
    elif bad == 'context_owner': repo.document['entities'][0]['sourceContext'] = True
    else: repo.document['entities'][0]['observationRefs'] = []
    with pytest.raises(PlatformError):
        cli.prepare(protocol, repo, blobs, manifest, runtime)
    assert not repo.calls
    if bad == 'ambiguous_owner':
        with pytest.raises(PlatformError, match='observation_multiple_owners'):
            reconstruction.run_segmentation(repo, blobs,
                {**repo.job, 'inputs':{'observationId':protocol['observationId']}}, {})
        assert not repo.calls
    if bad in ('unowned_observation', 'context_owner'):
        _, result = reconstruction.run_segmentation(repo, blobs,
            {**repo.job, 'inputs':{'observationId':protocol['observationId']}}, {})
        assert result['status'] == 'incomplete'
        assert result['errors'][0]['code'] == 'segmentation_owner_unresolved'
        assert not repo.calls


@pytest.mark.parametrize('label', [None, '', '   ', 12])
def test_segmentation_missing_saved_label_stops_before_reservation(tmp_path, monkeypatch, label):
    repo, blobs, manifest, runtime, protocol = segmentation_source(tmp_path, monkeypatch)
    repo.document['entities'][0]['label'] = label
    # The observation still has a usable label; there is no alternate-source fallback.
    assert repo.document['observations'][0]['labelEvidence'][0]['label']
    with pytest.raises(PlatformError, match='segmentation_label_required'):
        cli.prepare(protocol, repo, blobs, manifest, runtime)
    document, result = reconstruction.run_segmentation(repo, blobs,
        {**repo.job, 'inputs':{'observationId':protocol['observationId']}}, {})
    assert result['status'] == 'incomplete'
    assert result['errors'][0]['code'] == 'segmentation_label_required'
    assert document['observations'][0]['maskAssetId'] is None and not repo.calls


@pytest.mark.parametrize('bad', ['box', 'submitted_box', 'payload_extra', 'snapshot_revision',
    'snapshot_mapping', 'snapshot_image', 'entity_id', 'observation_id', 'image_bytes',
    'source_revision', 'source_mapping', 'source_box', 'source_ownership',
    'source_label', 'prompt_source', 'snapshot_label'])
def test_segmentation_rejects_rehashed_forgery_before_reservation(tmp_path, monkeypatch, bad):
    repo, blobs, manifest, runtime, protocol = segmentation_source(tmp_path, monkeypatch)
    frozen = cli.prepare(protocol, repo, blobs, manifest, runtime)
    payload, protocol = frozen['payload'], frozen['protocol']
    if bad == 'box': payload['box'][0] = 0.5
    elif bad == 'submitted_box': payload['submittedBox'][0] = 0
    elif bad == 'payload_extra': payload['prompt'] = 'a different object'
    elif bad == 'snapshot_revision': protocol['sourceObservation']['revision'] += 1
    elif bad == 'snapshot_mapping': protocol['sourceObservation']['pixelMapping'] = [{'forged':True}]
    elif bad == 'snapshot_image': protocol['sourceObservation']['imageId'] = repo.capture['images'][1]['id']
    elif bad == 'entity_id': protocol['entityId'] = 'foreign-entity'
    elif bad == 'observation_id': protocol['observationId'] = 'foreign-observation'
    elif bad == 'image_bytes': payload['image']['dataUri'] += 'AAAA'
    elif bad == 'source_revision': repo.document['observations'][0]['revision'] += 1
    elif bad == 'source_mapping': repo.document['observations'][0]['pixelMapping'] = [{'forged':True}]
    elif bad == 'source_box': repo.document['observations'][0]['originalPixelBox'][0] = 0.5
    elif bad == 'source_label': repo.document['entities'][0]['label'] = 'different saved label'
    elif bad == 'prompt_source': payload['promptSource']['entityId'] = 'other-entity'
    elif bad == 'snapshot_label': protocol['sourceObservation']['promptSource']['label'] = 'different label'
    else: repo.document['entities'][0]['observationRefs'] = []
    protocol['payloadSha256'] = digest(payload)
    job = {**repo.job, 'kind':'validate_model', 'config':{'researchProtocolSha256':digest(protocol)}}
    with pytest.raises(PlatformError):
        reconstruction.run_research_stage(repo, blobs, job, 'segmentation', payload,
            frozen['images'], manifest, protocol)
    assert not repo.calls


def test_segmentation_jpeg_payload_preserves_original_bytes_and_hash(tmp_path, monkeypatch):
    import base64
    import hashlib
    repo, blobs, manifest, runtime, protocol = segmentation_source(tmp_path, monkeypatch, jpeg=True)
    frozen = cli.prepare(protocol, repo, blobs, manifest, runtime)
    image = frozen['payload']['image']
    asset = repo.get_asset(image['imageId'])
    raw = blobs.get(asset['storageKey'], asset['sha256'], asset['sizeBytes'])
    assert image['dataUri'].startswith('data:image/jpeg;base64,')
    assert base64.b64decode(image['dataUri'].split(',', 1)[1]) == raw
    assert image['sha256'] == hashlib.sha256(raw).hexdigest() == asset['sha256']
    assert (image['height'], image['width']) == (9, 13)


def test_discovery_preserves_jpeg_payload_mime_and_bytes(tmp_path, monkeypatch):
    import base64
    from ehs_spatial.providers.gemini import GeminiAdapter
    repo, blobs, manifest, runtime, protocol = segmentation_source(tmp_path, monkeypatch, jpeg=True)
    frozen = cli.prepare(protocol, repo, blobs, manifest, runtime)
    seen = []
    def create(self, *args, **kwargs):
        seen.extend(kwargs['input'])
        return SimpleNamespace(id='discovery-fixture', usage=None)
    monkeypatch.setattr(GeminiAdapter, '_create', create)
    monkeypatch.setattr(GeminiAdapter, '_parse', lambda *a: (SimpleNamespace(items=[]), 'discovery-fixture'))
    before = digest(repo.document)
    result = reconstruction._discovery_invoke({'image':frozen['payload']['image']})
    asset = repo.get_asset(frozen['images'][0]['assetId'])
    raw = blobs.get(asset['storageKey'], asset['sha256'], asset['sizeBytes'])
    assert seen[-1].mime_type == 'image/jpeg'
    assert base64.b64decode(seen[-1].data) == raw
    assert result['items'] == [] and digest(repo.document) == before and not repo.calls


def fake_sam_transport(monkeypatch, failure=None):
    import numpy as np
    import fal_client
    import fal_client.client
    import httpx
    from ehs_spatial.providers.sam3 import encode_coco_rle

    events, requests = [], []
    def post(url, **kwargs):
        events.append('post')
        requests.append((url, kwargs))
        if failure == 'post_timeout':
            raise httpx.ReadTimeout('test-only uncertain POST')
        return httpx.Response(429 if failure == 'post_429' else 200,
            request=httpx.Request('POST', url), json={'request_id':'sam-request-test',
                'response_url':'https://queue.fal.run/test/response',
                'status_url':'https://queue.fal.run/test/status',
                'cancel_url':'https://queue.fal.run/test/cancel'})
    def handle(**kwargs):
        events.append('handle')
        assert kwargs['request_id'] == 'sam-request-test'
        def get():
            events.append('get')
            assert 'receipt' in events and events.index('receipt') < events.index('get')
            if failure == 'get_timeout':
                raise httpx.ReadTimeout('test-only unavailable result')
            if failure == 'invalid_response':
                return {'rle':[], 'scores':[]}
            if failure == 'invalid_response_type':
                return ['received but not a response object']
            mask = np.zeros((9,13),bool)
            mask[2:8,1:11] = True
            return {'rle':[encode_coco_rle(mask)], 'scores':[.9]}
        return SimpleNamespace(request_id=kwargs['request_id'], get=get)
    monkeypatch.setattr(fal_client, 'SyncClient', lambda: SimpleNamespace(
        _client=nullcontext(SimpleNamespace(post=post))))
    monkeypatch.setattr(fal_client.client, 'SyncRequestHandle', handle)
    monkeypatch.setattr(fal_client, 'submit', lambda *a, **kw: pytest.fail('retrying SDK submit must not be used'))
    return events, requests


@pytest.mark.parametrize('failure', [None, 'get_timeout', 'receipt_failure', 'invalid_response',
                                   'invalid_response_type', 'post_timeout', 'post_429'])
def test_sam_one_post_receipt_fence_and_no_charge_on_replay(tmp_path, monkeypatch, failure):
    repo, blobs, manifest, runtime, protocol = segmentation_source(tmp_path, monkeypatch)
    frozen = cli.prepare(protocol, repo, blobs, manifest, runtime)
    before = digest(repo.document)
    events, requests = fake_sam_transport(monkeypatch, failure)
    def record_dispatch(call_id, attempt, request_id):
        events.append('receipt')
        assert attempt == repo.job['attemptToken']
        call = next(c for c in repo.calls if c['id'] == call_id)
        assert call['status'] == 'reserved' and request_id == 'sam-request-test'
        if failure == 'receipt_failure':
            raise PlatformError('job_attempt_stale', 409)
        call['response'] = {'providerRequestId':request_id}
    monkeypatch.setattr(repo, 'record_model_call_dispatch', record_dispatch, raising=False)
    job = {**repo.job, 'kind':'validate_model', 'config':{
        'researchProtocolSha256':digest(frozen['protocol'])}}
    def run():
        return reconstruction.run_research_stage(repo, blobs, job, 'segmentation',
            frozen['payload'], frozen['images'], manifest, frozen['protocol'])
    if failure:
        for _ in range(2):
            with pytest.raises(PlatformError):
                run()
    else:
        result = run()
        assert result['scope'] == 'research_only' and result['sceneRevision'] is None
        assert result['newModelCalls'] == 1 and result['outputValidation']
        assert run()['newModelCalls'] == 0
    assert len(requests) == len(repo.calls) == 1
    url, options = requests[0]
    assert url == 'https://queue.fal.run/fal-ai/sam-3-1/image-rle'
    assert options['follow_redirects'] is False
    assert options['json']['box_prompts'] == [{'x_min':1, 'y_min':2, 'x_max':11, 'y_max':8}]
    assert all(type(v) is int for v in options['json']['box_prompts'][0].values())
    assert options['json']['prompt'] == frozen['payload']['prompt'] == 'small visible control'
    assert frozen['payload']['box'] == [1.2, 2.3, 10.4, 7.8]
    if failure in ('post_timeout', 'post_429'):
        assert events == ['post']
    elif failure == 'receipt_failure':
        assert events == ['post', 'receipt']
    else:
        assert events == ['post', 'receipt', 'handle', 'get']
    call = repo.calls[0]
    assert call['status'] == ('succeeded' if failure is None else
        'failed' if failure in ('invalid_response', 'invalid_response_type') else 'outcome_unknown')
    if failure not in ('post_timeout', 'post_429'):
        assert call['response']['providerRequestId'] == 'sam-request-test'
    assert digest(repo.document) == before
    assert manifest['segmentation']['releaseEvidence']['runtime']['status'] == 'unverified'


@pytest.mark.parametrize('bad', ['nonboolean', 'wrong_grid'])
def test_segmentation_research_output_must_be_boolean_source_grid(tmp_path, monkeypatch, bad):
    import numpy as np
    repo, blobs, manifest, runtime, protocol = segmentation_source(tmp_path, monkeypatch)
    frozen = cli.prepare(protocol, repo, blobs, manifest, runtime)
    provider = reconstruction.providers_from_manifest(manifest, _research=True)['segmentation']
    mask = np.ones((9,13), dtype=np.uint8) if bad == 'nonboolean' else np.ones((13,9), dtype=bool)
    monkeypatch.setattr(reconstruction, 'providers_from_manifest', lambda *a, **kw: {
        'segmentation':replace(provider, invoke=lambda payload:{'mask':mask}, records_dispatch=False)})
    job = {**repo.job, 'kind':'validate_model', 'config':{
        'researchProtocolSha256':digest(frozen['protocol'])}}
    before = digest(repo.document)
    for _ in range(2):
        result = reconstruction.run_research_stage(repo, blobs, job, 'segmentation',
            frozen['payload'], frozen['images'], manifest, frozen['protocol'])
        assert result['status'] == 'incomplete' and result['outputAssetId']
        assert result['outputValidation'][0]['admissionStatus'] == 'rejected'
        assert result['errors'][0]['code'] == 'mask_image_grid_mismatch'
    assert len(repo.calls) == 1
    assert digest(repo.document) == before


def test_segmentation_runtime_binding_detects_changed_adapter_bytes(tmp_path, monkeypatch):
    from pathlib import Path
    repo, blobs, manifest, runtime, protocol = segmentation_source(tmp_path, monkeypatch)
    frozen = cli.prepare(protocol, repo, blobs, manifest, runtime)
    altered = tmp_path / 'altered_adapter.py'
    altered.write_bytes(Path(reconstruction.__file__).read_bytes() + b'\n# changed test adapter\n')
    monkeypatch.setattr(reconstruction, '__file__', str(altered))
    with pytest.raises(PlatformError, match='research_runtime_unpinned'):
        reconstruction._validate_research_runtime(frozen['protocol'], manifest['segmentation']['pins'], 'segmentation')
    assert not repo.calls


@pytest.mark.parametrize('changed', ['missing_adapter_hash', 'sdk_version', 'license'])
def test_segmentation_runtime_and_license_gate_before_call(tmp_path, monkeypatch, changed):
    repo, blobs, manifest, runtime, protocol = segmentation_source(tmp_path, monkeypatch)
    if changed == 'missing_adapter_hash':
        runtime['segmentation'].pop('adapterSourceSha256')
    elif changed == 'sdk_version':
        runtime['segmentation']['falClientVersion'] = 'future-unreviewed-version'
    else:
        manifest['segmentation']['releaseEvidence']['license']['status'] = 'unverified'
    with pytest.raises(PlatformError):
        cli.prepare(protocol, repo, blobs, manifest, runtime)
    assert not repo.calls


@pytest.mark.parametrize('unknown', [False, True])
def test_segmentation_admin_submit_real_worker_receipt_and_replay(repo, tmp_path, monkeypatch, unknown):
    from decimal import Decimal
    from panoptes_worker.__main__ import run_job
    with monkeypatch.context() as local:
        source, blobs, manifest, runtime, protocol = segmentation_source(tmp_path, local)
    repo.blobs = blobs
    cap, scene = project(repo)
    pid, bid = scene['project']['id'], scene['branch']['id']
    images = [{k:a[k] for k in ('storageKey','sha256','sizeBytes','mediaType')} for a in source.assets]
    for image in images:
        image['metadata'] = {'width':13, 'height':9, 'pixelMapping':[]}
    capture = repo.create_capture(pid, cap, {'requestId':identity(), 'branchId':bid,
        'baseRevisionId':scene['revision']['id'], 'target':'scene'}, images)
    document = deepcopy(capture['revision']['document'])
    image = capture['capture']['images'][0]
    reconstruction._discover(document, image, {'items':[
        {'label':'small visible control', 'box':[1.2,2.3,10.4,7.8]}]}, repo.get_asset(image['assetId']))
    claimed = repo.claim_job(capture['job']['id'])
    discovered = repo.finish_job(claimed['id'], claimed['attemptToken'], 'succeeded', document=document)
    baseline = discovered['resultRevisionId']
    protocol.update(projectId=pid, branchId=bid, baselineRevision=baseline,
        entityId=document['entities'][0]['id'], observationId=document['observations'][0]['id'])
    repo.paid_budget = Decimal('.01')
    frozen = cli.prepare(protocol, repo, blobs, manifest, runtime)
    prepared = {'validation':frozen, 'sha256':digest(frozen)}
    job = cli.submit(prepared, repo, blobs)
    assert cli.submit(prepared, repo, blobs)['id'] == job['id']
    before = digest(repo.get_revision(baseline)['document'])
    events, requests = fake_sam_transport(monkeypatch, 'get_timeout' if unknown else None)
    original_dispatch = repo.record_model_call_dispatch
    def record_dispatch(call_id, attempt, request_id):
        result = original_dispatch(call_id, attempt, request_id)
        with repo._connect() as connection:
            row = connection.execute('SELECT * FROM model_calls WHERE id=%s', (call_id,)).fetchone()
        assert row['status'] == 'reserved' and row['response']['providerRequestId'] == request_id
        events.append('receipt')
        return result
    monkeypatch.setattr(repo, 'record_model_call_dispatch', record_dispatch)
    result = run_job(repo, blobs, job['id'])
    assert result['status'] == ('outcome_unknown' if unknown else 'succeeded')
    assert result['result']['scope'] == 'research_only'
    assert result['resultRevisionId'] is None and not result['headAdvanced']
    assert run_job(repo, blobs, job['id'])['id'] == job['id']
    assert len(requests) == 1 and events == ['post','receipt','handle','get']
    with repo._connect() as connection:
        calls = connection.execute('SELECT * FROM model_calls WHERE job_id=%s', (job['id'],)).fetchall()
    assert len(calls) == 1 and calls[0]['response']['providerRequestId'] == 'sam-request-test'
    assert calls[0]['status'] == ('outcome_unknown' if unknown else 'succeeded')
    assert repo.get_project(pid)['branches'][0]['headRevisionId'] == baseline
    assert digest(repo.get_revision(baseline)['document']) == before
    assert not document['coordinateFrames'] and not document['observations'][0]['maskAssetId']
    assert manifest['segmentation']['releaseEvidence']['quality']['status'] == 'unverified'
