import base64
import json
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

from ..contracts import (
    Assessment,
    ClimbReview,
    Criterion,
    GeometryFrame,
    GroundedAnswer,
    SceneMap,
)
from .base import ProviderError


GEMINI_MODEL_ID = "gemini-3.5-flash"


def _response_format(model_type: type[BaseModel]) -> dict[str, object]:
    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": model_type.model_json_schema(),
    }


def _text_block(text: str) -> object:
    from google.genai import interactions

    return interactions.TextContent(text=text)


def _image_block(image_path: str) -> object:
    from google.genai import interactions

    path = Path(image_path)
    try:
        raw = path.read_bytes()
        with Image.open(path) as image:
            mime_type = Image.MIME.get(image.format or "")
    except Exception as exc:
        raise ProviderError("gemini", "climb.image", str(exc)) from exc
    if mime_type not in {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
    }:
        raise ProviderError(
            "gemini", "climb.image", f"unsupported image MIME type: {mime_type}"
        )
    return interactions.ImageContent(
        data=base64.b64encode(raw).decode("ascii"),
        mime_type=mime_type,
    )


class GeminiAdapter:
    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def _interactions(self) -> Any:
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(
                    http_options={"retry_options": {"attempts": 1}}
                )
            except Exception as exc:
                raise ProviderError("gemini", "client", str(exc)) from exc
        return self._client.interactions

    def _create(self, operation: str, **kwargs: object) -> object:
        try:
            return self._interactions().create(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("gemini", operation, str(exc)) from exc

    @staticmethod
    def _parse(
        response: object, model_type: type[BaseModel], operation: str
    ) -> tuple[BaseModel, str]:
        status = getattr(response, "status", None)
        if status != "completed":
            raise ProviderError(
                "gemini", operation, f"interaction status is {status}"
            )
        interaction_id = getattr(response, "id", None)
        if not isinstance(interaction_id, str) or not interaction_id:
            raise ProviderError(
                "gemini", operation, "response is missing an interaction id"
            )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str):
            raise ProviderError(
                "gemini", f"{operation}.decode", "response is missing output_text"
            )
        try:
            return model_type.model_validate_json(output_text), interaction_id
        except Exception as exc:
            raise ProviderError("gemini", f"{operation}.decode", str(exc)) from exc

    def review_climb(
        self,
        scene: SceneMap,
        assessment: Assessment,
        criterion: Criterion,
        frames: list[GeometryFrame],
    ) -> tuple[ClimbReview, str]:
        if len(frames) != 4:
            raise ProviderError(
                "gemini", "climb.input", "climb review requires exactly four frames"
            )
        prompt = json.dumps(
            {
                "task": (
                    "Return a semantic climb-risk hint only. Never change the "
                    "deterministic assessment, coordinates, distances, or facts. "
                    "Use only listed fact_ids. A yes/no verdict must cite at least "
                    "one fact. If usable facts are absent, return uncertain."
                ),
                "scene_map": scene.model_dump(mode="json"),
                "assessment": assessment.model_dump(mode="json"),
                "criterion": criterion.model_dump(mode="json"),
                "image_frame_ids": [frame.frame_id for frame in frames],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        response = self._create(
            "climb.create",
            model=GEMINI_MODEL_ID,
            input=[
                _text_block(prompt),
                *[_image_block(frame.canonical_image_path) for frame in frames],
            ],
            store=True,
            stream=False,
            background=False,
            response_format=_response_format(ClimbReview),
        )
        parsed, interaction_id = self._parse(response, ClimbReview, "climb")
        review = ClimbReview.model_validate(parsed)
        allowed_fact_ids = {fact.fact_id for fact in scene.facts}
        if not set(review.fact_ids) <= allowed_fact_ids:
            raise ProviderError(
                "gemini", "climb.grounding", "response cites an unknown fact id"
            )
        if review.verdict in {"yes", "no"} and not review.fact_ids:
            raise ProviderError(
                "gemini",
                "climb.grounding",
                "a definitive climb verdict requires a fact id",
            )
        return review, interaction_id

    def answer(
        self,
        question: str,
        scene: SceneMap,
        *,
        previous_interaction_id: str,
    ) -> tuple[GroundedAnswer, str]:
        if not previous_interaction_id:
            raise ProviderError(
                "gemini", "chat.cursor", "previous interaction id is required"
            )
        response = self._create(
            "chat.create",
            model=GEMINI_MODEL_ID,
            input=[
                _text_block(
                    "Answer only from the stored SceneMap facts. Cite exact "
                    "fact_ids and their evidence_frame_ids. If the facts cannot "
                    "answer the question, start answer with "
                    "INSUFFICIENT_EVIDENCE: and return both ID lists empty. "
                    "Question: "
                    f"{question}"
                )
            ],
            store=True,
            stream=False,
            background=False,
            previous_interaction_id=previous_interaction_id,
            response_format=_response_format(GroundedAnswer),
        )
        parsed, interaction_id = self._parse(response, GroundedAnswer, "chat")
        answer = GroundedAnswer.model_validate(parsed)
        insufficient = answer.answer.startswith("INSUFFICIENT_EVIDENCE:")
        if insufficient:
            if answer.fact_ids or answer.evidence_frame_ids:
                raise ProviderError(
                    "gemini",
                    "chat.grounding",
                    "insufficient-evidence answers must have empty grounding ids",
                )
            return answer, interaction_id

        facts_by_id = {fact.fact_id: fact for fact in scene.facts}
        if not answer.fact_ids or not set(answer.fact_ids) <= facts_by_id.keys():
            raise ProviderError(
                "gemini",
                "chat.grounding",
                "factual answers require known fact ids",
            )
        allowed_evidence = {
            frame_id
            for fact_id in answer.fact_ids
            for frame_id in facts_by_id[fact_id].evidence_frame_ids
        }
        if (
            not answer.evidence_frame_ids
            or not set(answer.evidence_frame_ids) <= allowed_evidence
        ):
            raise ProviderError(
                "gemini",
                "chat.grounding",
                "factual answers require evidence from their cited facts",
            )
        return answer, interaction_id
