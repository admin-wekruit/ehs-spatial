"""Box-conditioned segmentation admits evidence without fabricating clipped masks."""
from copy import deepcopy

import numpy as np
import pytest

from ehs_spatial.platform import reconstruction as r
from ehs_spatial.platform.contracts import PlatformError, digest
from ehs_spatial.platform.storage import LocalBlobStore
from test_platform_reconstruction import Repo, bundle, provider


def test_exact_degenerate_and_disjoint_masks_are_rejected():
    image = {'id':'photo', 'width':10, 'height':8}
    box = [2.2, 1.3, 5.8, 4.7]
    disjoint = np.zeros((8,10),bool)
    disjoint[6:,8:] = True
    for mask, reason in [(np.ones((8,10),bool),'segmentation_full_image_for_local_box'),
                         (np.zeros((8,10),bool),'segmentation_empty'),
                         (disjoint,'segmentation_box_disjoint'),
                         (np.ones((8,10),np.uint8),'mask_image_grid_mismatch'),
                         (np.ones((10,8),bool),'mask_image_grid_mismatch')]:
        with pytest.raises(PlatformError, match=reason):
            r._segmentation_response(image, box, {'mask':mask})


@pytest.mark.parametrize('box', [[0,0,10,8], [.1,.1,9.9,7.9]])
def test_full_image_mask_allowed_for_box_covering_whole_pixel_grid(box):
    mask = np.ones((8,10),bool)
    actual, evidence = r._segmentation_response({'id':'photo','width':10,'height':8},box,{'mask':mask})
    assert np.array_equal(actual,mask)
    assert evidence['maskPixelCount'] == 80 and evidence['submittedBox'] == [0,0,10,8]


def test_large_nonconstant_mask_needs_intersection_without_ratio_or_clipping():
    mask = np.ones((8,10),bool)
    mask[0,0] = False
    original = mask.copy()
    actual, _ = r._segmentation_response({'id':'photo','width':10,'height':8},[4,3,5,4],{'mask':mask})
    assert np.array_equal(actual,original) and np.array_equal(mask,original)
    assert actual.sum() == 79  # 78 pixels outside a one-pixel source box remain intact.


def test_rejected_mask_cannot_replace_existing_observation_or_write_mask_asset():
    observation = {'id':'obs','revision':7,'originalPixelBox':[2,1,6,5],
        'maskAssetId':'existing-mask','maskEvidence':{'source':'preserved'}, 'sourceRefs':[]}
    document = {'assets':[]}
    before = deepcopy(observation)
    class NoMaskWrite:
        def put(self,*args):
            pytest.fail('rejected output must not write an observation mask')
    with pytest.raises(PlatformError,match='segmentation_full_image_for_local_box'):
        r._save_observation_mask(document,observation,{'id':'photo','width':10,'height':8},
            {'mask':np.ones((8,10),bool)}, {'id':'raw-output','sha256':'a'*64}, NoMaskWrite())
    assert observation == before and document == {'assets':[]}


def test_analysis_rejects_degenerate_mask_but_retains_raw_provider_output(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    providers = bundle(repo)
    providers['segmentation'] = provider('segmentation',lambda payload:{'mask':np.ones((12,12),bool)})
    document,result = r.run_analysis(repo,blobs,repo.job,providers)
    assert result['status'] == 'incomplete'
    assert any(e['code'] == 'segmentation_full_image_for_local_box' for e in result['errors'])
    assert all(o['maskAssetId'] is None and o['revision'] == 1 for o in document['observations'])
    saved = [a for a in document['assets'] if a.get('metadata',{}).get('stage') == 'segmentation']
    assert saved and all(a['metadata']['kind'] == 'stage_cache' for a in saved)
    assert not any(a.get('metadata',{}).get('kind') == 'observation_mask' for a in repo.assets)


def test_resegmentation_rejects_without_changing_revision_mask_or_old_surface(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    repo = Repo(blobs)
    repo.document,_ = r.run_analysis(repo,blobs,repo.job,bundle(repo))
    observation = repo.document['observations'][0]
    before_observation = deepcopy(observation)
    before_entities = digest(repo.document['entities'])
    job = {**repo.job,'kind':'segment_object','inputs':{'observationId':observation['id']}}
    providers = {'segmentation':provider('segmentation',lambda payload:{'mask':np.ones((12,12),bool)},model='new-test-provider')}
    document,result = r.run_segmentation(repo,blobs,job,providers)
    assert result['status'] == 'incomplete'
    assert result['errors'][0]['code'] == 'segmentation_full_image_for_local_box'
    assert next(o for o in document['observations'] if o['id'] == observation['id']) == before_observation
    assert digest(document['entities']) == before_entities
    stage = result['stages'][0]
    assert stage['status'] == 'succeeded' and stage['assetId'] in {a['id'] for a in document['assets']}


@pytest.mark.parametrize('bad,reason', [
    ('full','segmentation_full_image_for_local_box'), ('empty','segmentation_empty'),
    ('disjoint','segmentation_box_disjoint')])
def test_research_rejection_retains_actual_output_and_replay_never_calls_again(tmp_path,monkeypatch,bad,reason):
    from dataclasses import replace
    from test_research_capture_stages import segmentation_source
    from scripts.research import validate_sam3d as cli
    repo,blobs,manifest,runtime,protocol = segmentation_source(tmp_path,monkeypatch)
    frozen = cli.prepare(protocol,repo,blobs,manifest,runtime)
    spec = r.providers_from_manifest(manifest,_research=True)['segmentation']
    mask = np.full((9,13),bad == 'full',bool)
    if bad == 'disjoint':
        mask[0,0] = True
    monkeypatch.setattr(r,'providers_from_manifest',lambda *a,**kw:{
        'segmentation':replace(spec,invoke=lambda payload:{'mask':mask,'providerRequestId':'known-response'},records_dispatch=False)})
    job = {**repo.job,'kind':'validate_model','config':{'researchProtocolSha256':digest(frozen['protocol'])}}
    before = digest(repo.document)
    results = [r.run_research_stage(repo,blobs,job,'segmentation',frozen['payload'],
        frozen['images'],manifest,frozen['protocol']) for _ in range(2)]
    for result in results:
        assert result['status'] == 'incomplete' and result['sceneRevision'] is None
        assert result['outputValidation'][0]['admissionStatus'] == 'rejected'
        assert result['outputValidation'][0]['reason'] == reason
        assert result['errors'][0]['outputAssetId'] == result['outputAssetId']
        asset = repo.get_asset(result['outputAssetId'])
        stored = r._Stages(repo,blobs,job,{}).load(asset)
        assert np.array_equal(stored['output']['mask'],mask)
    assert results[0]['outputAssetId'] == results[1]['outputAssetId']
    assert [v['newModelCalls'] for v in results] == [1,0]
    assert len(repo.calls) == 1 and repo.calls[0]['status'] == 'succeeded'
    assert repo.calls[0]['response']['providerRequestId'] == 'known-response'
    assert digest(repo.document) == before
