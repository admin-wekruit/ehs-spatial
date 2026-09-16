"""Original-pixel segmentation crops and provider-grid admission stay reversible."""
import base64
from copy import deepcopy
import hashlib
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from ehs_spatial.platform import reconstruction as r
from ehs_spatial.platform.contracts import PlatformError
from ehs_spatial.providers.sam3 import encode_coco_rle
from test_research_capture_stages import fake_sam_transport


def roi_input(box):
    y,x = np.indices((37,83))
    rgb = np.stack((x,y,x+y),axis=-1).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream,format='PNG')
    raw = stream.getvalue()
    image = r._image_payload({'id':'source-photo','bytes':raw,'mediaType':'image/png',
        'sha256':hashlib.sha256(raw).hexdigest(),'width':83,'height':37})
    observation = {'id':'observation','imageId':'source-photo','originalPixelBox':box}
    document = {'entities':[{'id':'entity','label':'saved object label','observationRefs':['observation']}]}
    return r._segmentation_input(document,observation,image),rgb


@pytest.mark.parametrize('box,bounds', [([65.2,12.4,68.6,18.2],[56,5,77,26]),
    ([.2,.3,3.6,5.1],[0,0,11,12])])
def test_original_resolution_context_preserves_pixels_and_integer_translation(box,bounds):
    payload,rgb = roi_input(box)
    roi = payload['roi']
    left,top,right,bottom = bounds
    raw = base64.b64decode(roi['image']['dataUri'].split(',',1)[1])
    with Image.open(io.BytesIO(raw)) as image:
        assert np.array_equal(np.asarray(image),rgb[top:bottom,left:right])
        assert image.size == (right-left,bottom-top)
    assert hashlib.sha256(raw).hexdigest() == roi['image']['sha256']
    assert roi['pixelMapping']['sourceCropXYXY'] == bounds
    assert roi['pixelMapping']['sourceImageSha256'] == payload['image']['sha256']
    assert payload['box'] == box and payload['prompt'] == 'saved object label'
    assert roi['box'] == [box[0]-left,box[1]-top,box[2]-left,box[3]-top]
    assert np.array_equal(np.asarray(roi['pixelMapping']['matrix']) @ [left+2,top+3,1],[2,3,1])


def test_provider_crop_admission_precedes_exact_original_grid_mapping():
    payload,_ = roi_input([65.2,12.4,68.6,18.2])
    roi = payload['roi']
    shape = roi['pixelMapping']['targetShapeHW']
    response = {'mask':np.ones(shape,bool),'pixelMapping':deepcopy(roi['pixelMapping'])}
    with pytest.raises(PlatformError,match='segmentation_full_image_for_local_box'):
        r._segmentation_response(payload['image'],payload['box'],response)
    response['mask'] = np.zeros(shape,bool)
    response['mask'][7:10,9:12] = True
    response['mask'][0,0] = True  # Context pixels remain; the box is not used to clip output.
    mask,metrics = r._segmentation_response(payload['image'],payload['box'],response)
    expected = np.zeros((37,83),bool)
    expected[5:26,56:77] = response['mask']
    assert np.array_equal(mask,expected) and mask.sum() == response['mask'].sum()
    assert metrics['width'] == 83 and metrics['height'] == 37 and metrics['cropBoundaryTouched']
    assert metrics['providerGrid']['width'] == metrics['providerGrid']['height'] == 21
    response['pixelMapping']['matrix'][0][2] += 1
    with pytest.raises(PlatformError,match='segmentation_pixel_mapping_mismatch'):
        r._segmentation_response(payload['image'],payload['box'],response)


@pytest.mark.parametrize('empty',[False,True])
def test_sam_submits_crop_and_decodes_on_provider_grid(monkeypatch,empty):
    import fal_client.client
    payload,_ = roi_input([65.2,12.4,68.6,18.2])
    roi = payload['roi']
    candidate = np.zeros(roi['pixelMapping']['targetShapeHW'],bool)
    candidate[7:10,9:12] = True
    events,requests = fake_sam_transport(monkeypatch)
    def handle(**kwargs):
        assert events[-1] == 'receipt'
        return SimpleNamespace(request_id=kwargs['request_id'],get=lambda:{
            'rle':[] if empty else [encode_coco_rle(candidate)],'scores':[] if empty else [.9]})
    monkeypatch.setattr(fal_client.client,'SyncRequestHandle',handle)
    response = r._sam_invoke(payload,on_dispatched=lambda request_id:events.append('receipt'))
    submitted = requests[0][1]['json']
    assert len(requests) == 1 and submitted['image_url'] == roi['image']['dataUri']
    assert submitted['box_prompts'] == [{'x_min':9,'y_min':7,'x_max':13,'y_max':14}]
    assert submitted['prompt'] == payload['prompt']
    assert response['mask'].shape == (21,21) and response['pixelMapping'] == roi['pixelMapping']
    assert response['providerImageSha256'] == roi['image']['sha256']
    if empty:
        assert response['selectedCandidate'] is None and not response['mask'].any()
        with pytest.raises(PlatformError,match='segmentation_empty'):
            r._segmentation_response(payload['image'],payload['box'],response)
    else:
        assert np.array_equal(response['mask'],candidate)
        mapped,_ = r._segmentation_response(payload['image'],payload['box'],response)
        assert np.array_equal(mapped[5:26,56:77],candidate)
