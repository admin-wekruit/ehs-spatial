"""The real installed SDK over a CPU-only HTTP transport; no provider requests."""
import base64
import json
from hashlib import sha256

import httpx
from google import genai
from google.genai import interactions
from pydantic import BaseModel
import pytest

from ehs_spatial.providers.base import ProviderError
from ehs_spatial.providers.gemini import GeminiAdapter, GEMINI_MODEL_ID, _response_format


class Answer(BaseModel):
    items: list[str]


def inputs():
    return [interactions.TextContent(text='List each visible object.'),
            interactions.ImageContent(data=base64.b64encode(b'exact original image bytes').decode(), mime_type='image/jpeg')]


def client(transport):
    # Deliberately loose client defaults: the bounded method must override retries.
    return genai.Client(api_key='CPU-ONLY', http_options={
        'httpx_client': httpx.Client(transport=httpx.MockTransport(transport)),
        'retry_options': {'attempts': 5, 'initial_delay': 0.001}})


def success(request, *, reason='STOP', text='{"items":["chair"]}'):
    return httpx.Response(200, request=request, json={
        'responseId': 'provider-request-1',
        'candidates': [{'content': {'role': 'model', 'parts': [{'text': text}]}, 'finishReason': reason}],
        'usageMetadata': {'promptTokenCount': 1234, 'candidatesTokenCount': 20, 'thoughtsTokenCount': 7}})


@pytest.mark.parametrize('image_count', [1, 6])
def test_count_and_generation_receive_the_same_complete_native_request(image_count):
    sent = []
    def transport(request):
        sent.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith(':countTokens'):
            return httpx.Response(200, request=request, json={'totalTokens': 1234})
        return success(request)
    with client(transport) as sdk:
        blocks = inputs()
        blocks += [blocks[1]] * (image_count - 1)
        response = GeminiAdapter(sdk).create_bounded_structured(
            'platform.discovery', input=blocks, response_format=_response_format(Answer))
    assert [path.rsplit(':', 1)[-1] for path, _ in sent] == ['countTokens', 'generateContent']
    counted = sent[0][1]['generateContentRequest']
    assert set(sent[0][1]) == {'generateContentRequest'}
    assert counted.pop('model') == 'models/' + GEMINI_MODEL_ID
    assert counted == sent[1][1]
    parts = counted['contents'][0]['parts']
    assert len(parts) == image_count + 1
    assert all(part == parts[1] for part in parts[1:])
    assert base64.b64decode(parts[1]['inlineData']['data']) == b'exact original image bytes'
    assert parts[1]['inlineData']['mimeType'] == 'image/jpeg'
    assert parts[1]['mediaResolution'] == {'level': 'MEDIA_RESOLUTION_HIGH'}
    assert counted['generationConfig'] == {
        'responseMimeType': 'application/json', 'responseJsonSchema': Answer.model_json_schema(),
        'maxOutputTokens': 8192, 'candidateCount': 1, 'thinkingConfig': {'thinkingLevel': 'LOW'}}
    assert 'tools' not in counted and 'cachedContent' not in counted
    parsed, request_id = GeminiAdapter._parse(response, Answer, 'platform.discovery')
    assert parsed.items == ['chair'] and request_id == 'provider-request-1'
    assert response.input_token_count == 1234
    assert response.budget_evidence == {
        'model': GEMINI_MODEL_ID, 'inputTokens': 1234, 'maxInputTokens': 16384,
        'maxOutputTokens': 8192, 'imageResolution': 'high',
        'requestSha256': response.request_sha256, 'countMethod': 'models.countTokens.generateContentRequest',
        'maximumGenerationPosts': 1}
    json.dumps(response.budget_evidence, allow_nan=False)
    assert response.request_sha256 == sha256(json.dumps(counted, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert response.usage.model_dump(exclude_none=True)['thoughts_token_count'] == 7


@pytest.mark.parametrize('count', [16385, None, 0, -1])
def test_invalid_or_over_budget_input_never_generates(count):
    paths = []
    def transport(request):
        paths.append(request.url.path)
        return httpx.Response(200, request=request, json={'totalTokens': count})
    with client(transport) as sdk, pytest.raises(ProviderError):
        GeminiAdapter(sdk).create_bounded_structured('platform.discovery', input=inputs(), response_format=_response_format(Answer))
    assert len(paths) == 1 and paths[0].endswith(':countTokens')


@pytest.mark.parametrize('reason,text', [('MAX_TOKENS', '{"items":'), ('STOP', 'invalid JSON'), ('SAFETY', '')])
def test_paid_incomplete_or_invalid_output_retains_request_id_and_usage(reason, text):
    def transport(request):
        if request.url.path.endswith(':countTokens'):
            return httpx.Response(200, request=request, json={'totalTokens': 1234})
        return success(request, reason=reason, text=text)
    with client(transport) as sdk:
        response = GeminiAdapter(sdk).create_bounded_structured('platform.model_review', input=inputs(), response_format=_response_format(Answer))
    with pytest.raises(ProviderError):
        GeminiAdapter._parse(response, Answer, 'platform.model_review')
    assert response.id == 'provider-request-1'
    assert response.usage.model_dump(exclude_none=True)['prompt_token_count'] == 1234
    assert response.finish_reason == reason


@pytest.mark.parametrize('failure', [503, 429, httpx.ReadTimeout, httpx.ReadError])
def test_generation_failure_has_one_charged_post_even_with_retrying_client(failure):
    paths = []
    def transport(request):
        paths.append(request.url.path)
        if request.url.path.endswith(':countTokens'):
            return httpx.Response(200, request=request, json={'totalTokens': 1234})
        if isinstance(failure, int):
            return httpx.Response(failure, request=request, json={'error': {'code': failure, 'message': 'CPU fixture'}})
        raise failure('CPU fixture', request=request)
    with client(transport) as sdk, pytest.raises(ProviderError):
        GeminiAdapter(sdk).create_bounded_structured('platform.discovery', input=inputs(), response_format=_response_format(Answer))
    assert sum(path.endswith(':generateContent') for path in paths) == 1


@pytest.mark.parametrize('failure', [503, httpx.ReadTimeout])
def test_count_failure_never_generates_or_retries(failure):
    paths = []
    def transport(request):
        paths.append(request.url.path)
        if isinstance(failure, int):
            return httpx.Response(failure, request=request, json={'error': {'code': failure, 'message': 'CPU fixture'}})
        raise failure('CPU fixture', request=request)
    with client(transport) as sdk, pytest.raises(ProviderError):
        GeminiAdapter(sdk).create_bounded_structured('platform.discovery', input=inputs(), response_format=_response_format(Answer))
    assert len(paths) == 1 and paths[0].endswith(':countTokens')


def test_developer_api_budget_cannot_be_used_for_vertex_backend():
    sent = []
    def transport(request):
        sent.append(request)
        raise AssertionError('Must reject before network')
    with genai.Client(vertexai=True, api_key='CPU-ONLY', http_options={
        'httpx_client': httpx.Client(transport=httpx.MockTransport(transport))}) as sdk:
        with pytest.raises(ProviderError, match='bounded_developer_api_required'):
            GeminiAdapter(sdk).create_bounded_structured('platform.discovery', input=inputs(), response_format=_response_format(Answer))
    assert not sent


@pytest.mark.parametrize('bad', ['missing_evidence', 'empty_evidence', 'blank_evidence', 'missing_role'])
def test_discovery_rejects_missing_visual_evidence_or_role_with_usage_retained(monkeypatch, bad):
    from ehs_spatial.platform import reconstruction
    from ehs_spatial.providers import gemini
    item = {'label':'control', 'box_2d':[100,100,200,200],
            'evidence':'A red circular face within a yellow surround.', 'geometry_role':'unknown'}
    if bad == 'missing_evidence':
        item.pop('evidence')
    elif bad == 'missing_role':
        item.pop('geometry_role')
    else:
        item['evidence'] = '' if bad == 'empty_evidence' else ' \t\n '
    sent = []
    def transport(request):
        sent.append(request.url.path)
        if request.url.path.endswith(':countTokens'):
            return httpx.Response(200, request=request, json={'totalTokens':1234})
        return success(request, text=json.dumps({'items':[item]}))
    with client(transport) as sdk:
        adapter = GeminiAdapter(sdk)
        monkeypatch.setattr(gemini, 'GeminiAdapter', lambda:adapter)
        with pytest.raises(reconstruction.ProviderResponseError) as error:
            reconstruction._discovery_invoke({'image':{'width':100,'height':100,
                'dataUri':'data:image/jpeg;base64,'+inputs()[1].data}})
    assert error.value.outcome == 'failed'
    assert error.value.telemetry['providerRequestId'] == 'provider-request-1'
    assert error.value.telemetry['usage']['prompt_token_count'] == 1234
    assert error.value.telemetry['usage']['thoughts_token_count'] == 7
    assert [path.rsplit(':',1)[-1] for path in sent] == ['countTokens','generateContent']


@pytest.mark.parametrize('role', ['unknown','object','floor'])
def test_discovery_schema_requires_evidence_and_role_but_preserves_explicit_unknown(role):
    from ehs_spatial.platform.reconstruction import _DiscoveryResponse
    item = {'label':'surface', 'box_2d':[100,100,200,200],
            'evidence':'A continuous gray surface visible beneath the equipment.', 'geometry_role':role}
    parsed = _DiscoveryResponse.model_validate({'items':[item]})
    assert parsed.items[0].geometry_role == role and parsed.items[0].evidence == item['evidence']
    schema = _DiscoveryResponse.model_json_schema()['$defs']['_DiscoveredItem']
    assert {'evidence','geometry_role'} <= set(schema['required'])


def test_discovery_full_frame_scan_is_counted_and_sent_once_with_original_image(monkeypatch):
    from ehs_spatial.platform import reconstruction
    from ehs_spatial.providers import gemini
    sent = []
    item = {'label':'visible component', 'box_2d':[100,200,500,800],
            'evidence':'A distinct circular face attached to the larger housing.', 'geometry_role':'object'}
    def transport(request):
        sent.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith(':countTokens'):
            return httpx.Response(200, request=request, json={'totalTokens':1234})
        return success(request, text=json.dumps({'items':[item]}))
    with client(transport) as sdk:
        adapter = GeminiAdapter(sdk)
        monkeypatch.setattr(gemini, 'GeminiAdapter', lambda:adapter)
        result = reconstruction._discovery_invoke({'image':{'width':3024,'height':4032,
            'dataUri':'data:image/jpeg;base64,'+inputs()[1].data}})
    assert [path.rsplit(':',1)[-1] for path, _ in sent] == ['countTokens','generateContent']
    counted = sent[0][1]['generateContentRequest']
    counted.pop('model')
    assert counted == sent[1][1]
    parts = counted['contents'][0]['parts']
    assert len(parts) == 2
    prompt = parts[0]['text']
    for instruction in ('top to bottom and left to right', 'large structures and assemblies',
            'smaller and attached components', 'same physical instance twice',
            'original full image', 'nonblank visual evidence'):
        assert instruction in prompt
    assert base64.b64decode(parts[1]['inlineData']['data']) == b'exact original image bytes'
    assert parts[1]['inlineData']['mimeType'] == 'image/jpeg'
    assert counted['generationConfig']['responseJsonSchema'] == reconstruction._DiscoveryResponse.model_json_schema()
    assert result['items'][0]['box'] == pytest.approx([604.8,403.2,2419.2,2016])
    assert result['items'][0]['evidence'] == item['evidence']


@pytest.mark.parametrize('observation_count', [1,41])
def test_inventory_review_mode_sends_exact_source_and_inventory_through_bounded_provider(monkeypatch,observation_count):
    from copy import deepcopy
    from ehs_spatial.platform import reconstruction
    from ehs_spatial.providers import gemini
    inventory = [{'id':f'observation-{index}','revision':3,'entityId':'entity-a','imageId':'image-a',
        'pixelMapping':[{'matrix':[[1,0,0],[0,1,0],[0,0,1]]}],
        'sourceRefs':[{'assetId':'source-a','sha256':'b'*64}],
        'originalPixelBox':[10,20,30,40],'labelEvidence':[{'label':'visible housing','evidence':'Blue outline.',
            'geometryRole':'object','sourceRefs':[{'assetId':'source-a','sha256':'b'*64}]}]}
        for index in range(observation_count)]
    payload = {'mode':'inventory','inventorySha256':'a'*64,'observations':inventory,
        'image':{'width':3024,'height':4032,'dataUri':'data:image/jpeg;base64,'+inputs()[1].data}}
    original = deepcopy(payload)
    review = {'inventorySha256':'a'*64,'observationIds':[o['id'] for o in inventory],
        'reason':'A separate visible outline was omitted.',
        'additions':[{'label':'attached control','evidence':'Distinct circular face beside the housing.',
            'box_2d':[200,300,250,350],'geometry_role':'unknown'}],'unresolvedRegions':[]}
    sent = []
    def transport(request):
        sent.append((request.url.path,json.loads(request.content)))
        if request.url.path.endswith(':countTokens'):
            return httpx.Response(200,request=request,json={'totalTokens':1234})
        return success(request,text=json.dumps(review))
    with client(transport) as sdk:
        adapter = GeminiAdapter(sdk)
        monkeypatch.setattr(gemini,'GeminiAdapter',lambda:adapter)
        result = reconstruction._model_review_invoke(payload)
    assert [path.rsplit(':',1)[-1] for path,_ in sent] == ['countTokens','generateContent']
    counted = sent[0][1]['generateContentRequest']
    counted.pop('model')
    assert counted == sent[1][1]
    parts = counted['contents'][0]['parts']
    assert len(parts)==3
    compact = json.loads(parts[1]['text'])
    assert compact == {'inventorySha256':'a'*64,'imageWidth':3024,'imageHeight':4032,
        'boxConvention':'original-image pixels [xmin,ymin,xmax,ymax]',
        'observations':[{'id':o['id'],'originalPixelBox':o['originalPixelBox'],
            'labelEvidence':[{'label':'visible housing','evidence':'Blue outline.','geometryRole':'object'}]}
            for o in inventory]}
    assert payload == original
    assert not any(key in parts[1]['text'] for key in ('sourceRefs','pixelMapping','entityId','revision'))
    assert parts[2]['inlineData'] == {'mimeType':'image/jpeg','data':inputs()[1].data}
    assert counted['generationConfig']['responseJsonSchema'] == reconstruction._InventoryReviewResponse.model_json_schema()
    assert result['review'] == review and result['providerRequestId']=='provider-request-1'


@pytest.mark.parametrize('change', [
    {'max_input_tokens': 16385}, {'max_output_tokens': 8193}, {'max_output_tokens': True},
    {'input': [{'type': 'image', 'uri': 'https://example.invalid/image.jpg', 'mime_type': 'image/jpeg'}]},
    {'input': [{'type': 'text', 'text': 'x', 'tools': []}]},
])
def test_unsupported_request_or_caps_fail_before_any_http(change):
    sent = []
    def transport(request):
        sent.append(request)
        raise AssertionError('Must reject before network')
    kwargs = {'input': inputs(), 'response_format': _response_format(Answer), **change}
    with client(transport) as sdk, pytest.raises(ProviderError):
        GeminiAdapter(sdk).create_bounded_structured('platform.discovery', **kwargs)
    assert not sent


@pytest.mark.parametrize('failure', ['input_limit','count_transport','generation_transport'])
def test_bounded_stage_distinguishes_preflight_without_releasing_or_repeating_reservation(tmp_path,failure):
    from ehs_spatial.platform import reconstruction
    from ehs_spatial.platform.contracts import PlatformError,canonical
    from ehs_spatial.platform.storage import LocalBlobStore
    from test_platform_reconstruction import Repo
    sent=[]
    secret='TEST-PRIVATE-PROVIDER-ERROR-DO-NOT-PERSIST'
    def transport(request):
        sent.append(request.url.path)
        if request.url.path.endswith(':countTokens'):
            if failure=='count_transport':
                raise httpx.ReadTimeout(secret,request=request)
            return httpx.Response(200,request=request,json={'totalTokens':17575 if failure=='input_limit' else 1234})
        raise httpx.ReadTimeout(secret,request=request)
    blobs=LocalBlobStore(tmp_path)
    repo=Repo(blobs); repo.paid_budget=.3
    pins={'model':GEMINI_MODEL_ID,'adapter':'test-only-bounded'}
    # Synthetic release evidence only exercises the local HTTP mock/ledger.
    evidence={'pins':pins,**{gate:{'status':'passed','artifactSha256':'a'*64}
        for gate in ('license','runtime','quality')}}
    with client(transport) as sdk:
        adapter=GeminiAdapter(sdk)
        def invoke(payload):
            return adapter.create_bounded_structured('platform.discovery',input=inputs(),response_format=_response_format(Answer))
        spec=reconstruction.ProviderSpec('gemini',pins,invoke,.1,evidence)
        stages=reconstruction._Stages(repo,blobs,repo.job,{'discovery':spec})
        expected='provider_outcome_unknown' if failure=='generation_transport' else 'provider_preflight_failed'
        with pytest.raises(PlatformError,match=expected) as caught:
            stages.call('discovery',[],{'frozen':'input'})
        count=len(sent)
        retry_code='provider_outcome_unknown' if failure=='generation_transport' else 'provider_response_unavailable'
        with pytest.raises(PlatformError,match=retry_code):
            stages.call('discovery',[],{'frozen':'input'})
        assert len(sent)==count and len(repo.calls)==1
    call=repo.calls[0]
    assert call['estimate']==.1 and call['actual_cost'] is None
    assert secret not in canonical(call).decode()
    if failure=='generation_transport':
        assert call['status']=='outcome_unknown' and 'preflight' not in call['response']
        assert len(sent)==2 and sent[-1].endswith(':generateContent')
    else:
        assert call['status']=='failed' and len(sent)==1 and sent[0].endswith(':countTokens')
        diagnostic=call['response']['preflight']
        assert diagnostic==caught.value.params['preflight']==stages.records[0]['preflight']
        assert diagnostic['generationAttempted'] is False and len(diagnostic['requestSha256'])==64
        assert diagnostic['maxInputTokens']==16384
        assert diagnostic['code']==('bounded_input_token_limit' if failure=='input_limit' else 'bounded_count_failed')
        assert diagnostic['phase']==('input_limit' if failure=='input_limit' else 'count_tokens')
        if failure=='input_limit':
            assert diagnostic['inputTokens']==17575
        else:
            assert 'inputTokens' not in diagnostic


@pytest.mark.parametrize('count', [None,False,0,-1])
def test_invalid_token_count_has_safe_preflight_metadata_and_no_generation(count):
    from ehs_spatial.providers.gemini import BoundedPreflightError
    paths=[]
    def transport(request):
        paths.append(request.url.path)
        return httpx.Response(200,request=request,json={'totalTokens':count})
    with client(transport) as sdk,pytest.raises(BoundedPreflightError) as caught:
        GeminiAdapter(sdk).create_bounded_structured('platform.discovery',input=inputs(),response_format=_response_format(Answer))
    assert len(paths)==1 and paths[0].endswith(':countTokens')
    assert caught.value.diagnostics['code']=='bounded_token_count_invalid'
    assert caught.value.diagnostics['generationAttempted'] is False
