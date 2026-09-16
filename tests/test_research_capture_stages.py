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
        stage: replace(spec, invoke=invoke)})
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
    def remote(payload, **kwargs):
        received.append(kwargs)
        return response
    monkeypatch.setitem(sys.modules, 'modal', SimpleNamespace(Cls=SimpleNamespace(
        from_name=lambda *a: lambda: SimpleNamespace(run=SimpleNamespace(remote=remote)))))
    provider = reconstruction.providers_from_manifest(manifest, _research=True)[stage]
    provider.invoke({**frozen['payload'], '_researchProtocol': protocol})
    assert received == [{'expectedRuntimeManifestSha256': expected}]
    response['runtimeManifestSha256'] = 'e' * 64
    with pytest.raises(reconstruction.ProviderResponseError):
        provider.invoke({**frozen['payload'], '_researchProtocol': protocol})


@pytest.mark.parametrize('stage', ['geometry', 'depth'])
def test_admin_stage_submit_worker_and_ledger_are_idempotent(repo, tmp_path, monkeypatch, stage):
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
    spec = reconstruction.providers_from_manifest(manifest, _research=True)[stage]
    def invoke(payload):
        seen.append(payload)
        return geometry_response(payload['images']) if stage == 'geometry' else depth_response(payload['image'])
    monkeypatch.setattr(reconstruction, 'providers_from_manifest', lambda *a, **kw: {
        stage: replace(spec, invoke=invoke)})
    result = run_job(repo, blobs, job['id'])
    assert result['status'] == 'succeeded'
    assert result['result']['scope'] == 'research_only'
    assert result['resultRevisionId'] is None and not result['headAdvanced']
    assert repo.get_project(pid)['branches'][0]['headRevisionId'] == capture['revision']['id']
    run_job(repo, blobs, job['id'])
    assert len(seen) == 1
    assert manifest[stage]['releaseEvidence']['runtime']['status'] == 'unverified'


@pytest.mark.parametrize('stage', ['geometry', 'depth'])
def test_matching_pins_do_not_make_empty_stage_output_valid(tmp_path, monkeypatch, stage):
    repo, blobs, manifest, frozen = preparation(tmp_path, monkeypatch, stage)
    spec = reconstruction.providers_from_manifest(manifest, _research=True)[stage]
    monkeypatch.setattr(reconstruction, 'providers_from_manifest', lambda *a, **kw: {
        stage: replace(spec, invoke=lambda payload:{'pins':spec.pins})})
    job = {**repo.job, 'kind':'validate_model', 'config':{
        'researchProtocolSha256':digest(frozen['protocol'])}}
    before = digest(repo.document)
    for _ in range(2):
        with pytest.raises(PlatformError):
            reconstruction.run_research_stage(repo,blobs,job,stage,frozen['payload'],
                frozen['images'],manifest,frozen['protocol'])
    assert len(repo.calls) == 1  # Retain/reject known bad output; never rebill on replay.
    assert digest(repo.document) == before
