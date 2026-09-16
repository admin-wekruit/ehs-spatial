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
        with pytest.raises(ProviderError, match='Developer API'):
            GeminiAdapter(sdk).create_bounded_structured('platform.discovery', input=inputs(), response_format=_response_format(Answer))
    assert not sent


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
