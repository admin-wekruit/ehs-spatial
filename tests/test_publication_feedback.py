"""No-network checks for visitor feedback, isolated from frozen publications."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import asyncio
import base64
import hashlib
import json
import socket
import threading
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
import pytest

from check_publication_site import fixture as publication_fixture
from ehs_spatial.platform.agent_service import AgentResult, GeminiAgentProvider, INSTRUCTION
from ehs_spatial.platform.feedback import FEEDBACK_INSTRUCTION, FeedbackService, persistent_feedback_app
from ehs_spatial.platform.publication_site import create_app


ORIGIN = "https://report.example"
CAPABILITY = "A" * 43
OTHER_CAPABILITY = base64.urlsafe_b64encode(bytes([1]) * 32).decode().rstrip("=")
HEADERS = {"Authorization": "Feedback " + CAPABILITY, "Origin": ORIGIN}


def identity(number=None):
    return str(UUID(int=number)) if number is not None else str(uuid4())


def request(**changes):
    return {"requestId": identity(), "conversationId": identity(),
            "message": "This rail looks loose.", "language": "en", **changes}


class Provider:
    name, model, paid, max_call_cost = "test", "no-network", True, 0.25

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def respond(self, messages, context):
        self.calls.append((deepcopy(messages), deepcopy(context)))
        if self.fail:
            raise TimeoutError("response may have been generated")
        return AgentResult({"kind": "answer", "message": "Which joint is loose?"},
                           {"total_token_count": 12}, "test-provider-reference")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Feedback tests must not access the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "catalog"
    publications = []
    for offset in (0, 100):
        publication_id, project_id, revision_id, asset_id = [identity(offset + n) for n in (1, 2, 3, 4)]
        directory, bundle = publication_fixture(root, publication_id, project_id, revision_id,
            asset_id, "2026-09-13T00:00:00+00:00", "Photo bytes must stay local")
        publication = bundle["responses"]["/api/publications/" + publication_id]
        revision = publication["snapshot"]["revision"]
        document = revision["document"]
        document["assets"][0]["kind"] = "source_image"
        document["observations"] = [{"id": identity(offset + n), "imageId": asset_id,
            "originalPixelBox": [1, 2, 10, 20]} for n in (5, 6, 7)]
        document["entities"] = [
            {"id": identity(offset + 8), "label": "Selected rail", "associationState": "confirmed",
             "observationRefs": [identity(offset + 5), identity(offset + 6)], "measurements": {}},
            {"id": identity(offset + 9), "label": "Unrelated private object", "associationState": "confirmed",
             "observationRefs": [identity(offset + 7)], "measurements": {}},
        ]
        revision["documentSha256"] = hashlib.sha256(json.dumps(document).encode()).hexdigest()
        bundle["responses"]["/api/publications"]["items"][0].update(photoCount=1, objectCount=2)
        (directory / "bundle.json").write_text(json.dumps(bundle))
        publications.append(publication)
    return root, publications


def route(publication, entity_index=0):
    entity = publication["snapshot"]["revision"]["document"]["entities"][entity_index]
    return f"/api/publications/{publication['id']}/entities/{entity['id']}/feedback"


def client_for(catalog, database, provider=None, total_budget=None, call_reservation=None):
    return TestClient(create_app(catalog, allowed_origins=[ORIGIN], feedback=FeedbackService(
        database, provider=provider, total_budget=total_budget, call_reservation=call_reservation)))


def post(client, path, body, headers=HEADERS):
    response = client.post(path, json=body, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def history(client, path, body, headers=HEADERS):
    return client.get(path, params={"conversationId": body["conversationId"]}, headers=headers)


def test_disabled_feedback_persists_across_reopen_without_changing_publication(catalog, tmp_path):
    root, publications = catalog
    publication, body = publications[0], request()
    database, path = tmp_path / "feedback.sqlite", route(publication)
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with client_for(root, database) as client:
        assert history(client, path, body).json() == {"items": []}
        turn = post(client, path, body)
        assert turn["status"] == "saved" and turn["errorCode"] == "feedback_model_disabled"
        assert turn["assistantMessage"] is None
        for key in ("requestId", "conversationId", "message", "language"):
            assert turn[key] == body[key]
        assert turn["publicationId"] == publication["id"]
        assert turn["revisionId"] == publication["sceneRevisionId"]
        assert turn["entityId"] == publication["snapshot"]["revision"]["document"]["entities"][0]["id"]
        assert turn["id"] and turn["createdAt"]
    with client_for(root, database) as client:
        response = history(client, path, body)
        assert response.status_code == 200 and response.json() == {"items": [turn]}
        assert "immutable" not in response.headers.get("cache-control", "")
        assert post(client, path, body) == turn
        assert client.get("/api/publications/" + publication["id"]).json() == publication
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_capability_and_publication_entity_scope(catalog, tmp_path):
    root, publications = catalog
    path, body = route(publications[0]), request()
    with client_for(root, tmp_path / "feedback.sqlite") as client:
        post(client, path, body)
        other_headers = {"Authorization": "Feedback " + OTHER_CAPABILITY}
        for response in (history(client, path, body, other_headers),
                         client.post(path, json=body, headers=other_headers)):
            assert response.status_code == 403
            assert response.json()["error"]["code"] == "feedback_capability_forbidden"
        for wrong_path in (route(publications[0], 1), route(publications[1])):
            for response in (history(client, wrong_path, body),
                             client.post(wrong_path, json=request(conversationId=body["conversationId"]), headers=HEADERS)):
                assert response.status_code == 403
                assert response.json()["error"]["code"] == "feedback_scope_mismatch"
        unknown = path.replace("/entities/" + identity(8), "/entities/" + identity())
        assert client.post(unknown, json=request(), headers=HEADERS).status_code == 404
        # A different visitor creates an independent conversation on the same object.
        other = post(client, path, request(), headers=other_headers)
        assert history(client, path, body).json()["items"][0]["conversationId"] != other["conversationId"]


def test_paid_call_is_idempotent_and_budget_is_global_across_reopen(catalog, tmp_path):
    root, publications = catalog
    path, body, provider = route(publications[0]), request(), Provider()
    database = tmp_path / "feedback.sqlite"
    with client_for(root, database, provider, 0.25, 0.25) as client:
        turn = post(client, path, body)
        assert turn["status"] == "succeeded" and turn["assistantMessage"] == "Which joint is loose?"
        assert turn["errorCode"] is None
        assert post(client, path, body) == turn and len(provider.calls) == 1
        conflict = client.post(path, json={**body, "message": "Changed payload"}, headers=HEADERS)
        assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "idempotency_mismatch"
    with client_for(root, database, provider, 0.25, 0.25) as client:
        assert post(client, path, body) == turn
        blocked_body = request()
        blocked = post(client, route(publications[1]), blocked_body)
        assert blocked["status"] == "saved" and blocked["errorCode"] == "feedback_budget_exceeded"
        assert blocked["assistantMessage"] is None and len(provider.calls) == 1
        assert history(client, route(publications[1]), blocked_body).json() == {"items": [blocked]}


def test_unknown_provider_outcome_is_not_retried_or_refunded(catalog, tmp_path):
    root, publications = catalog
    path, body, provider = route(publications[0]), request(), Provider(fail=True)
    database = tmp_path / "feedback.sqlite"
    with client_for(root, database, provider, 0.25, 0.25) as client:
        turn = post(client, path, body)
        assert turn["status"] == "outcome_unknown" and turn["assistantMessage"] is None
        assert turn["errorCode"]
        assert post(client, path, body) == turn and len(provider.calls) == 1
    with client_for(root, database, provider, 0.25, 0.25) as client:
        assert post(client, path, body) == turn
        blocked = post(client, path, request())
        assert blocked["errorCode"] == "feedback_budget_exceeded" and len(provider.calls) == 1


def test_concurrent_service_instances_share_one_durable_budget_reservation(catalog, tmp_path):
    publication, database = catalog[1][0], tmp_path / "feedback.sqlite"
    entity_id = publication["snapshot"]["revision"]["document"]["entities"][0]["id"]
    started, release, blocked = threading.Event(), threading.Event(), threading.Event()
    barrier, bodies = threading.Barrier(2), [request(), request()]

    class BlockingProvider(Provider):
        def respond(self, messages, context):
            result = super().respond(messages, context)
            started.set()
            assert release.wait(3), "Test did not release the provider"
            return result

    provider = BlockingProvider()
    services = [FeedbackService(database, provider, 0.25, 0.25) for _ in range(2)]

    def submit(index):
        barrier.wait(3)
        turn = services[index].submit(publication, entity_id, CAPABILITY, bodies[index])
        if turn["status"] == "saved":
            blocked.set()
        return turn

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit, index) for index in range(2)]
        try:
            assert started.wait(3)
            assert blocked.wait(3), "Second service must see the reservation before the first call finishes"
            assert len(provider.calls) == 1
        finally:
            release.set()
        turns = [future.result(timeout=3) for future in futures]
    assert sorted(turn["status"] for turn in turns) == ["saved", "succeeded"]
    assert next(turn for turn in turns if turn["status"] == "saved")["errorCode"] == "feedback_budget_exceeded"
    reopened = FeedbackService(database, provider, 0.25, 0.25)
    for body, turn in zip(bodies, turns):
        assert reopened.history(publication, entity_id, CAPABILITY, body["conversationId"]) == {"items": [turn]}
        assert reopened.submit(publication, entity_id, CAPABILITY, body) == turn
    assert len(provider.calls) == 1


def test_gemini_provider_preserves_default_and_forwards_feedback_instruction_and_no_retry_options(monkeypatch):
    from google import genai

    constructions, calls = [], []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text='{"kind":"answer","message":"Saved feedback."}',
            usage_metadata=SimpleNamespace(model_dump=lambda mode: {"total_token_count": 9}), response_id="test-response")

    def client(**kwargs):
        constructions.append(kwargs)
        return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    monkeypatch.setenv("GEMINI_API_KEY", "test-only-feedback-key")
    monkeypatch.setattr(genai, "Client", client)
    default = GeminiAgentProvider(model="test-default", max_call_cost=0.25)
    options = {"timeout": 60000, "retry_options": {"attempts": 1}}
    feedback = GeminiAgentProvider(model="test-feedback", max_call_cost=0.25,
        instruction=FEEDBACK_INSTRUCTION, http_options=options)
    assert constructions == [{"api_key": "test-only-feedback-key"},
                            {"api_key": "test-only-feedback-key", "http_options": options}]
    assert constructions[1]["http_options"]["retry_options"]["attempts"] == 1
    messages, context = [{"role": "user", "content": "Loose rail"}], {"entity": {"id": identity(8)}}
    default.respond(messages, context)
    answer = feedback.respond(messages, context)
    assert calls[0]["config"]["system_instruction"] == INSTRUCTION
    assert calls[1]["config"]["system_instruction"] == FEEDBACK_INSTRUCTION
    assert calls[1]["model"] == "test-feedback"
    assert json.loads(calls[1]["contents"]) == {"context": context, "conversation": messages}
    assert "tools" not in calls[1]["config"]
    assert answer.content == {"kind": "answer", "message": "Saved feedback."}
    assert answer.usage == {"total_token_count": 9} and answer.provider_ref == "test-response"


@pytest.mark.parametrize("total,reservation", [(None, 0.25), (1, None), (0, 0.25), (1, 0)])
def test_unconfigured_budget_saves_without_provider_call(catalog, tmp_path, total, reservation):
    root, publications = catalog
    provider = Provider()
    with client_for(root, tmp_path / "feedback.sqlite", provider, total, reservation) as client:
        turn = post(client, route(publications[0]), request())
        assert turn["status"] == "saved" and turn["errorCode"] == "feedback_model_disabled"
        assert provider.calls == []


def test_model_context_contains_selected_object_evidence_only(catalog, tmp_path):
    root, publications = catalog
    publication, provider = publications[0], Provider()
    document = publication["snapshot"]["revision"]["document"]
    selected, unrelated = document["entities"]
    body = request(imageId=identity(4), observationId=identity(5))
    with client_for(root, tmp_path / "feedback.sqlite", provider, 1, 0.25) as client:
        post(client, route(publication), body)
        followup = request(conversationId=body["conversationId"], observationId=identity(6), imageId=identity(4), message="The second joint.")
        post(client, route(publication), followup)
        assert len(provider.calls) == 2
        context = json.dumps(provider.calls[0][1])
        assert selected["id"] in context and identity(5) in context
        assert unrelated["id"] not in context and unrelated["label"] not in context
        assert identity(7) not in context
        serialized_input = json.dumps(provider.calls)
        assert "Photo bytes must stay local artifact" not in serialized_input
        assert str(root) not in serialized_input and "storageKey" not in serialized_input
        assert body["message"] in json.dumps(provider.calls[1][0])
        assert "Which joint is loose?" in json.dumps(provider.calls[1][0])
        for changes in ({"observationId": identity(7), "imageId": identity(4)},
                        {"observationId": identity(5), "imageId": identity(104)}):
            denied = client.post(route(publication), json=request(**changes), headers=HEADERS)
            assert denied.status_code == 422
        assert len(provider.calls) == 2


def test_feedback_cors_and_validation_do_not_enable_publication_edits(catalog, tmp_path):
    root, publications = catalog
    publication, body = publications[0], request()
    path = route(publication)
    with client_for(root, tmp_path / "feedback.sqlite") as client:
        cors = client.options(path, headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST",
                                             "Access-Control-Request-Headers": "authorization,content-type"})
        assert cors.status_code == 200 and cors.headers["access-control-allow-origin"] == ORIGIN
        assert "authorization" in cors.headers["access-control-allow-headers"].lower()
        assert client.post(path, json=body, headers=HEADERS).headers["access-control-allow-origin"] == ORIGIN
        assert "access-control-allow-origin" not in client.get(path, params={"conversationId": body["conversationId"]},
            headers={**HEADERS, "Origin": "https://unrelated.example"}).headers
        for other_path in ("/api/publications/" + publication["id"], "/api/projects/" + publication["projectId"] + "/edits"):
            denied = client.post(other_path, json={}, headers=HEADERS)
            assert denied.status_code == 403 and denied.json()["error"]["code"] == "publication_read_only"
        assert client.options("/api/publications/" + publication["id"], headers={"Origin": ORIGIN,
            "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "authorization"}).status_code == 400
        assert client.options(path, headers={"Origin": "https://unrelated.example",
            "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization"}).status_code == 400
        for method in ("PUT", "PATCH", "DELETE"):
            assert client.request(method, path, json={}, headers=HEADERS).status_code == 403
        for authorization in (None, "Feedback short", "Bearer " + CAPABILITY, "Feedback " + "!" * 43):
            headers = {} if authorization is None else {"Authorization": authorization}
            assert client.post(path, json=request(), headers=headers).status_code == 401
        for changes in ({"requestId": "not-a-uuid"}, {"conversationId": "not-a-uuid"},
                        {"message": ""}, {"message": " "}, {"message": "a" * 8001}, {"language": "fr"},
                        {"observationId": "not-a-uuid"}, {"imageId": "not-a-uuid"}):
            assert client.post(path, json=request(**changes), headers=HEADERS).status_code == 422


def test_unmodeled_object_with_nullable_fields_accepts_feedback(catalog, tmp_path):
    publication = deepcopy(catalog[1][0])
    entity = publication["snapshot"]["revision"]["document"]["entities"][0]
    entity.update(currentModelTransform=None, measurements=None, representations=None, observationRefs=None)
    service, body = FeedbackService(tmp_path / "feedback.sqlite"), request()
    turn = service.submit(publication, entity["id"], CAPABILITY, body)
    assert turn["status"] == "saved"
    assert service.history(publication, entity["id"], CAPABILITY, body["conversationId"]) == {"items": [turn]}


@pytest.mark.parametrize("fail_on", [1, 2])
def test_checkpoint_failure_never_repeats_a_paid_call(catalog, tmp_path, fail_on):
    publication, provider, body = catalog[1][0], Provider(), request()
    database = tmp_path / "feedback.sqlite"
    entity_id = publication["snapshot"]["revision"]["document"]["entities"][0]["id"]
    checkpoints = []

    def checkpoint():
        # A second connection can see the committed record before either boundary.
        saved = FeedbackService(database).history(publication, entity_id, CAPABILITY, body["conversationId"])["items"][0]
        checkpoints.append(saved["status"])
        assert len(provider.calls) == len(checkpoints) - 1
        if len(checkpoints) == fail_on:
            raise OSError("durable volume commit failed")

    service = FeedbackService(database, provider, 0.25, 0.25, checkpoint=checkpoint)
    with pytest.raises(OSError, match="durable volume commit failed"):
        service.submit(publication, entity_id, CAPABILITY, body)
    assert checkpoints == ["outcome_unknown", "succeeded"][:fail_on]
    reopened = FeedbackService(database, provider, 0.25, 0.25)
    assert reopened.submit(publication, entity_id, CAPABILITY, body)["status"] == checkpoints[-1]
    assert len(provider.calls) == fail_on - 1


def test_persistent_feedback_serializes_writes_through_cancellation_without_blocking_metadata():
    async def exercise():
        events = []
        first_started, release_first, second_started = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Volume:
            def reload(self):
                events.append("reload")

            def commit(self):
                events.append("commit")

        async def app(scope, receive, send):
            name = scope["name"]
            events.append("start:" + name)
            if name == "first":
                first_started.set()
                await release_first.wait()
            if name == "second":
                second_started.set()
            events.append("end:" + name)

        wrapped = persistent_feedback_app(app, Volume(), lambda path: path == "/feedback")

        async def call(name, path="/feedback", method="POST"):
            await wrapped({"type": "http", "path": path, "method": method, "name": name}, None, None)

        first = asyncio.create_task(call("first"))
        await asyncio.wait_for(first_started.wait(), 1)
        first.cancel()
        second = asyncio.create_task(call("second"))
        await asyncio.wait_for(call("metadata", "/api/publications", "GET"), 1)
        await asyncio.wait_for(call("preflight", method="OPTIONS"), 1)
        await asyncio.sleep(0)
        assert not first.done() and not second_started.is_set()
        assert events == ["reload", "start:first", "start:metadata", "end:metadata", "start:preflight", "end:preflight"]
        release_first.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 1)
        await asyncio.wait_for(second, 1)
        assert events[-6:] == ["end:first", "commit", "reload", "start:second", "end:second", "commit"]

    asyncio.run(exercise())
