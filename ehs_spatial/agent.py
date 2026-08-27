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


# What a reviewer looks for in every workcell photo — the standard sweep.
# Each item: (instruction for the locator, english label). The agent runs
# the list, skips items it cannot find (honestly), and skips boxes that
# duplicate an existing refinement.
STANDARD_SWEEP: list[tuple[str, str]] = [
    ("入口左侧的光幕立柱（黄黑色竖条传感器）", "safety sensor"),
    ("入口右侧的光幕立柱（黄黑色竖条传感器）", "safety sensor"),
    ("入口处红色的斜坡挡板/踢脚板（左侧）", "sloped surface"),
    ("入口处红色的斜坡挡板/踢脚板（右侧）", "sloped surface"),
    ("急停按钮面板（红色蘑菇头按钮）", "emergency stop button"),
    ("信号灯塔（红黄绿堆叠指示灯）", "safety light"),
    ("最靠近入口的一段安全围栏/透明护板", "safety fence"),
    ("警示/安全标识牌", "warning sign"),
]


def agent_sweep(
    run_id: str,
    items: list[tuple[str, str]] | None = None,
    *,
    runs_root: str | Path = "runs",
    apply: bool = True,
    locator=None,
    subscriber=None,
) -> list[dict]:
    """Run the standard reviewer sweep over one photo: locate each checklist
    item, refine what is found, skip what is not — every outcome reported."""
    import json as json_module

    run = Path(runs_root) / run_id
    existing_boxes: list[tuple[int, int, int, int]] = []
    log_path = run / "refinements.json"
    if log_path.exists():
        existing_boxes = [
            tuple(item["box"]) for item in json_module.loads(log_path.read_text())
        ]

    def overlaps(box) -> bool:
        x1, y1, x2, y2 = box
        area = max(1, (x2 - x1) * (y2 - y1))
        for ex1, ey1, ex2, ey2 in existing_boxes:
            iw = max(0, min(x2, ex2) - max(x1, ex1))
            ih = max(0, min(y2, ey2) - max(y1, ey1))
            inter = iw * ih
            union = area + max(1, (ex2 - ex1) * (ey2 - ey1)) - inter
            if inter / union > 0.5:
                return True
        return False

    outcomes: list[dict] = []
    for instruction, label in items or STANDARD_SWEEP:
        try:
            result = agent_refine(
                run_id,
                f"{instruction}（label: {label}）",
                runs_root=runs_root,
                apply=apply,
                locator=locator,
                subscriber=subscriber,
            )
            if overlaps(result["box"]):
                outcomes.append(
                    {"item": instruction, "status": "duplicate", "box": result["box"]}
                )
                continue
            existing_boxes.append(tuple(result["box"]))
            outcomes.append(
                {
                    "item": instruction,
                    "status": "measured",
                    "label": result["label"],
                    "sam_score": result["sam_score"],
                    "height_m": result["height_m"],
                    "camera_dist_m": result["camera_dist_m"],
                }
            )
        except RefineError as exc:
            outcomes.append(
                {"item": instruction, "status": "not_found", "reason": str(exc)}
            )
        except ProviderError as exc:
            outcomes.append(
                {"item": instruction, "status": "provider_error", "reason": str(exc)}
            )
    return outcomes
