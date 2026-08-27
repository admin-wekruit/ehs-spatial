"""Conversational correction: describe a missed object in plain language,
the VLM locates it in the photo, SAM segments the located box, the run's
geometry measures it, and (optionally) the correction feeds back into the
run's scene + policy verdicts.

This is the spoken form of the workbench's box-refine tab — same tools,
natural-language front end. Every hop is auditable: the located box, the
SAM score, and the measurement all land in refinements.json.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from .providers.base import ProviderError
from .providers.gemini import (
    GEMINI_MODEL_ID,
    GeminiAdapter,
    _image_block,
    _response_format,
    _text_block,
)
from .refine import RefineError, refine_region


class LocatedObject(BaseModel):
    found: bool
    label_en: str = Field(
        description="short English noun phrase for the object, e.g. "
        "'safety fence', 'sloped surface', 'safety sensor'"
    )
    box_1000: tuple[int, int, int, int] = Field(
        description="tight bounding box as (xmin, ymin, xmax, ymax) in "
        "0-1000 normalized image coordinates"
    )
    rationale: str


_LOCATE_PROMPT = (
    "You are helping audit an industrial workcell photo. The reviewer says "
    "an object was missed by automatic segmentation and describes it below "
    "(possibly in Chinese). Locate that object in the photo.\n"
    "Reviewer: {instruction}\n"
    "Return found=false if you cannot see a matching object. Box must be "
    "tight around the described object only."
)


def _gemini_locator(image_path: str, instruction: str, adapter: GeminiAdapter):
    response = adapter._create(
        "agent.locate",
        model=GEMINI_MODEL_ID,
        input=[
            _text_block(_LOCATE_PROMPT.format(instruction=instruction)),
            _image_block(image_path),
        ],
        response_format=_response_format(LocatedObject),
    )
    located, _ = adapter._parse(response, LocatedObject, "agent.locate")
    return located


def agent_refine(
    run_id: str,
    instruction: str,
    *,
    runs_root: str | Path = "runs",
    apply: bool = True,
    locator=None,
    subscriber=None,
) -> dict:
    """One conversational correction turn. Raises RefineError/ProviderError
    on failure — never fabricates a measurement."""
    from PIL import Image

    run = Path(runs_root) / run_id
    image_path = next((run / "input").glob("image_*"))
    if locator is None:
        adapter = GeminiAdapter()

        def locator(path, text):  # noqa: F811 - default binding
            return _gemini_locator(path, text, adapter)

    located = locator(str(image_path), instruction)
    if not located.found:
        raise RefineError(f"VLM could not locate the object: {located.rationale}")
    with Image.open(image_path) as image:
        width, height = image.size
    x_min, y_min, x_max, y_max = located.box_1000
    box = (
        max(0, int(x_min / 1000 * width)),
        max(0, int(y_min / 1000 * height)),
        min(width, int(x_max / 1000 * width)),
        min(height, int(y_max / 1000 * height)),
    )
    result = refine_region(
        run_id,
        located.label_en,
        box,
        runs_root=runs_root,
        apply=apply,
        subscriber=subscriber,
    )
    result["instruction"] = instruction
    result["located_rationale"] = located.rationale
    return result


__all__ = ["agent_refine", "LocatedObject", "RefineError", "ProviderError"]
