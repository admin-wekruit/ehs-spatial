import base64
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from ehs_spatial.contracts import (
    Assessment,
    ClimbReview,
    Criterion,
    Entity3D,
    GeometryFrame,
    GroundedAnswer,
    SceneMap,
    SpatialFact,
)
from ehs_spatial.providers.base import ProviderError


IDENTITY_4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 1.5),
    (0.0, 0.0, 0.0, 1.0),
)
INTRINSICS = ((100.0, 0.0, 1.0), (0.0, 100.0, 1.0), (0.0, 0.0, 1.0))


class FakeInteractions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses):
        self.interactions = FakeInteractions(responses)


def _scene(*, facts=True):
    entity = Entity3D(
        entity_id="pallet-1",
        label="pallet",
        observation_ids=["obs-1"],
        centroid_xyz=[1, 2, 0.4],
        footprint_xy=[[0, 0], [1, 0], [1, 1], [0, 1]],
        height_m=0.8,
        evidence_frame_ids=["frame-1", "frame-2"],
    )
    spatial_facts = (
        [
            SpatialFact(
                fact_id="fact-clearance",
                predicate="minimum_boundary_clearance",
                subject_id="pallet-1",
                object_id="fence-1",
                value=0.5,
                unit="m",
                evidence_frame_ids=["frame-1", "frame-2"],
            )
        ]
        if facts
        else []
    )
    return SceneMap(
        run_id="run-1",
        floor_plane=[0, 0, 1, 0],
        scale_source="camera_height",
        scale_factor=1,
        fence_polygon=[[0, 0], [2, 0], [2, 2]],
        entities=[entity],
        facts=spatial_facts,
        warnings=[],
    )


def _assessment(*, facts=True):
    return Assessment(
        status="FAIL" if facts else "INSUFFICIENT_EVIDENCE",
        fact_ids=["fact-clearance"] if facts else [],
        evidence_frame_ids=["frame-1", "frame-2"] if facts else [],
        approximate_distance_m=0.5 if facts else None,
    )


def _frames(tmp_path):
    formats = [("PNG", ".png"), ("JPEG", ".jpg"), ("WEBP", ".webp"), ("PNG", ".png")]
    result = []
    for index, (image_format, extension) in enumerate(formats, start=1):
        image_path = tmp_path / f"canonical-{index}{extension}"
        Image.new("RGB", (2, 2), (index, index, index)).save(
            image_path, format=image_format
        )
        result.append(
            GeometryFrame(
                frame_id=f"frame-{index}",
                canonical_image_path=str(image_path),
                pts3d_path=str(tmp_path / f"points-{index}.npy"),
                conf_path=str(tmp_path / f"conf-{index}.npy"),
                valid_mask_path=str(tmp_path / f"valid-{index}.npy"),
                camera_to_world=IDENTITY_4,
                intrinsics=INTRINSICS,
            )
        )
    return result


def _interaction(output, interaction_id="interaction-1", status="completed"):
    return SimpleNamespace(
        output_text=json.dumps(output), id=interaction_id, status=status
    )


def test_climb_review_uses_stable_model_structured_schema_and_four_raw_images(tmp_path):
    from ehs_spatial.providers.gemini import GEMINI_MODEL_ID, GeminiAdapter

    client = FakeClient(
        [
            _interaction(
                {
                    "verdict": "yes",
                    "rationale": "The object height is relevant to a climb review.",
                    "fact_ids": ["fact-clearance"],
                }
            )
        ]
    )
    frames = _frames(tmp_path)
    adapter = GeminiAdapter(client=client)

    review, cursor = adapter.review_climb(
        _scene(), _assessment(), Criterion(), frames
    )

    assert GEMINI_MODEL_ID == "gemini-3.5-flash"
    assert review.verdict == "yes"
    assert cursor == "interaction-1"
    [call] = client.interactions.calls
    assert set(call) == {
        "model",
        "input",
        "store",
        "stream",
        "background",
        "response_format",
    }
    assert call["model"] == GEMINI_MODEL_ID
    assert call["store"] is True
    assert call["stream"] is False
    assert call["background"] is False
    assert call["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": ClimbReview.model_json_schema(),
    }
    assert len(call["input"]) == 5
    assert call["input"][0].type == "text"
    assert '"minimum_clearance_m":0.6' in call["input"][0].text
    assert [item.mime_type for item in call["input"][1:]] == [
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/png",
    ]
    for block, frame in zip(call["input"][1:], frames, strict=True):
        assert block.type == "image"
        assert not block.data.startswith("data:")
        assert base64.b64decode(block.data, validate=True) == Path(
            frame.canonical_image_path
        ).read_bytes()


@pytest.mark.parametrize(
    ("payload", "interaction_id", "match"),
    [
        ("not-json", "interaction-1", "decode"),
        (
            json.dumps(
                {"verdict": "yes", "rationale": "unsupported", "fact_ids": []}
            ),
            "interaction-1",
            "grounding",
        ),
        (
            json.dumps(
                {
                    "verdict": "uncertain",
                    "rationale": "unknown",
                    "fact_ids": ["unknown-fact"],
                }
            ),
            "interaction-1",
            "grounding",
        ),
        (
            json.dumps(
                {
                    "verdict": "uncertain",
                    "rationale": "unknown",
                    "fact_ids": [],
                }
            ),
            "",
            "interaction id",
        ),
    ],
)
def test_climb_review_rejects_malformed_or_ungrounded_provider_output(
    tmp_path, payload, interaction_id, match
):
    from ehs_spatial.providers.gemini import GeminiAdapter

    client = FakeClient(
        [SimpleNamespace(output_text=payload, id=interaction_id, status="completed")]
    )

    with pytest.raises(ProviderError, match=match):
        GeminiAdapter(client=client).review_climb(
            _scene(), _assessment(), Criterion(), _frames(tmp_path)
        )


def test_climb_review_without_scene_facts_can_only_be_uncertain(tmp_path):
    from ehs_spatial.providers.gemini import GeminiAdapter

    client = FakeClient(
        [
            _interaction(
                {"verdict": "no", "rationale": "Looks safe", "fact_ids": []}
            )
        ]
    )

    with pytest.raises(ProviderError, match="grounding"):
        GeminiAdapter(client=client).review_climb(
            _scene(facts=False),
            _assessment(facts=False),
            Criterion(),
            _frames(tmp_path),
        )


def test_chat_chains_previous_interaction_and_requires_grounded_answer():
    from ehs_spatial.providers.gemini import GeminiAdapter

    client = FakeClient(
        [
            _interaction(
                {
                    "answer": "The measured clearance is approximately 0.5 m.",
                    "fact_ids": ["fact-clearance"],
                    "evidence_frame_ids": ["frame-1"],
                },
                interaction_id="interaction-2",
            )
        ]
    )

    answer, cursor = GeminiAdapter(client=client).answer(
        "How far is the pallet from the fence?",
        _scene(),
        previous_interaction_id="interaction-1",
    )

    assert answer.fact_ids == ["fact-clearance"]
    assert cursor == "interaction-2"
    [call] = client.interactions.calls
    assert set(call) == {
        "model",
        "input",
        "store",
        "stream",
        "background",
        "previous_interaction_id",
        "response_format",
    }
    assert call["previous_interaction_id"] == "interaction-1"
    assert call["response_format"]["schema"] == GroundedAnswer.model_json_schema()
    assert len(call["input"]) == 1
    assert call["input"][0].model_dump() == {
        "type": "text",
        "text": (
            "Answer only from the stored SceneMap facts. Cite exact fact_ids "
            "and their evidence_frame_ids. If the facts cannot answer the "
            "question, start answer with INSUFFICIENT_EVIDENCE: and return "
            "both ID lists empty. Question: "
            "How far is the pallet from the fence?"
        ),
    }


@pytest.mark.parametrize(
    "payload",
    [
        {
            "answer": "It is 1 m away.",
            "fact_ids": ["unknown"],
            "evidence_frame_ids": ["frame-1"],
        },
        {
            "answer": "It is safe.",
            "fact_ids": [],
            "evidence_frame_ids": [],
        },
        {
            "answer": "INSUFFICIENT_EVIDENCE: no temperature fact exists.",
            "fact_ids": ["fact-clearance"],
            "evidence_frame_ids": ["frame-1"],
        },
        {
            "answer": "The clearance is 0.5 m.",
            "fact_ids": ["fact-clearance"],
            "evidence_frame_ids": ["frame-99"],
        },
    ],
)
def test_chat_rejects_unknown_or_missing_grounding(payload):
    from ehs_spatial.providers.gemini import GeminiAdapter

    client = FakeClient([_interaction(payload, interaction_id="interaction-2")])

    with pytest.raises(ProviderError, match="grounding"):
        GeminiAdapter(client=client).answer(
            "question", _scene(), previous_interaction_id="interaction-1"
        )


def test_chat_accepts_explicit_insufficient_evidence_with_empty_ids():
    from ehs_spatial.providers.gemini import GeminiAdapter

    client = FakeClient(
        [
            _interaction(
                {
                    "answer": "INSUFFICIENT_EVIDENCE: no temperature fact exists.",
                    "fact_ids": [],
                    "evidence_frame_ids": [],
                },
                interaction_id="interaction-2",
            )
        ]
    )

    answer, _ = GeminiAdapter(client=client).answer(
        "What is the temperature?", _scene(), previous_interaction_id="interaction-1"
    )

    assert answer.fact_ids == []


def test_gemini_provider_call_errors_preserve_original_message(tmp_path):
    from ehs_spatial.providers.gemini import GeminiAdapter

    client = FakeClient([RuntimeError("quota exhausted")])

    with pytest.raises(ProviderError, match="quota exhausted") as caught:
        GeminiAdapter(client=client).review_climb(
            _scene(), _assessment(), Criterion(), _frames(tmp_path)
        )

    assert caught.value.original_message == "quota exhausted"


def test_gemini_rejects_a_non_completed_interaction(tmp_path):
    from ehs_spatial.providers.gemini import GeminiAdapter

    client = FakeClient(
        [
            _interaction(
                {
                    "verdict": "uncertain",
                    "rationale": "pending",
                    "fact_ids": [],
                },
                status="in_progress",
            )
        ]
    )

    with pytest.raises(ProviderError, match="in_progress"):
        GeminiAdapter(client=client).review_climb(
            _scene(), _assessment(), Criterion(), _frames(tmp_path)
        )


def test_default_gemini_client_uses_minimum_public_sdk_retry_setting(
    monkeypatch, tmp_path
):
    from google import genai

    from ehs_spatial.providers.gemini import GeminiAdapter

    created_with = []
    client = FakeClient(
        [
            _interaction(
                {
                    "verdict": "uncertain",
                    "rationale": "review required",
                    "fact_ids": [],
                }
            )
        ]
    )

    def fake_client(**kwargs):
        created_with.append(kwargs)
        return client

    monkeypatch.setattr(genai, "Client", fake_client)

    GeminiAdapter().review_climb(
        _scene(), _assessment(), Criterion(), _frames(tmp_path)
    )

    assert created_with == [
        {"http_options": {"retry_options": {"attempts": 1}}}
    ]
