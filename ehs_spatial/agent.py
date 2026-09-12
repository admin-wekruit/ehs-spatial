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
from .path_safety import validate_safe_path_segment
from .providers.gemini import (
    GEMINI_MODEL_ID,
    GeminiAdapter,
    _image_block,
    _response_format,
    _text_block,
)
from .refine import RefineError, input_image, refine_region


class LocatedObject(BaseModel):
    found: bool
    label_en: str = Field(
        description="short English noun phrase for the object, e.g. "
        "'safety fence', 'sloped surface', 'safety sensor'"
    )
    # Gemini's native grounding convention: [ymin, xmin, ymax, xmax],
    # 0-1000 normalized. Anything else drifts between runs.
    box_2d: tuple[int, int, int, int] = Field(
        description="tight bounding box as [ymin, xmin, ymax, xmax] in "
        "0-1000 normalized coordinates (the standard box_2d convention)"
    )
    rationale: str


_LOCATE_PROMPT = (
    "You are helping audit an industrial workcell photo. The reviewer says "
    "an object was missed by automatic segmentation and describes it below "
    "(possibly in Chinese). Locate that object in the photo.\n"
    "Reviewer: {instruction}\n"
    "Use only this photograph. Locate one visible instance; adjacent objects "
    "and objects seen through transparent surfaces remain separate. Do not "
    "infer a hidden object, physical dimensions, rectangular layout, or safety function. "
    "Return found=false if you cannot identify a single matching object. box_2d MUST be "
    "[ymin, xmin, ymax, xmax] in 0-1000 normalized coordinates, tight "
    "around the described object only."
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


class CropCheck(BaseModel):
    matches: bool
    reason: str


_VERIFY_PROMPT = (
    "This crop was auto-located for the description below. Does the crop "
    "actually show that object (fully or mostly)?\n"
    "Description: {instruction}\n"
    "Answer matches=false if the crop shows something else."
)


def _gemini_verify(crop_path: str, instruction: str, adapter: GeminiAdapter):
    response = adapter._create(
        "agent.verify",
        model=GEMINI_MODEL_ID,
        input=[
            _text_block(_VERIFY_PROMPT.format(instruction=instruction)),
            _image_block(crop_path),
        ],
        response_format=_response_format(CropCheck),
    )
    check, _ = adapter._parse(response, CropCheck, "agent.verify")
    return check


def agent_refine(
    run_id: str,
    instruction: str,
    *,
    runs_root: str | Path = "runs",
    apply: bool = True,
    locator=None,
    subscriber=None,
    verifier=None,
    frame_id: str = "frame_0001",
) -> dict:
    """Locate once in the selected photo, then persist SAM evidence.

    A supplied verifier is an explicit offline/test check; this path never
    creates a second VLM call to inspect its own crop.
    """
    from PIL import Image

    run = Path(runs_root) / run_id
    validate_safe_path_segment(run_id, "run_id")
    if not run.resolve().is_relative_to(Path(runs_root).resolve()):
        raise RefineError("run is outside runs_root")
    image_path = input_image(run, frame_id)
    if locator is None:
        adapter = GeminiAdapter()

        def locator(path, text):  # noqa: F811 - default binding
            return _gemini_locator(path, text, adapter)

    located = locator(str(image_path), instruction)
    if not located.found:
        raise RefineError(f"VLM could not locate the object: {located.rationale}")
    with Image.open(image_path) as image:
        width, height = image.size
    y_min, x_min, y_max, x_max = located.box_2d
    if min(located.box_2d) < 0 or max(located.box_2d) > 1000:
        raise RefineError(
            f"locator returned non-normalized coordinates {located.box_2d}; "
            "refusing to guess the convention"
        )
    if not (y_min < y_max and x_min < x_max):
        raise RefineError(f"degenerate located box {located.box_2d}")
    box = (
        max(0, int(x_min / 1000 * width)),
        max(0, int(y_min / 1000 * height)),
        min(width, int(x_max / 1000 * width)),
        min(height, int(y_max / 1000 * height)),
    )
    if callable(verifier):
        import tempfile

        with Image.open(image_path) as image:
            crop = image.crop(box)
            with tempfile.NamedTemporaryFile(suffix=".png") as handle:
                crop.save(handle.name)
                check = verifier(handle.name, instruction)
        if not check.matches:
            raise RefineError(
                f"located crop rejected by self-check: {check.reason}"
            )
    result = refine_region(
        run_id,
        located.label_en,
        box,
        runs_root=runs_root,
        apply=apply,
        subscriber=subscriber,
        frame_id=frame_id,
        instruction=instruction,
        located_rationale=located.rationale,
    )
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
    ("斜坡挡板下沿贴地的黄黑色传感器横杆", "safety sensor"),
    ("立柱上的圆顶指示灯", "indicator light"),
    ("立柱上的门联锁开关/把手", "door interlock switch"),
]


def agent_sweep(
    run_id: str,
    items: list[tuple[str, str]] | None = None,
    *,
    runs_root: str | Path = "runs",
    apply: bool = True,
    locator=None,
    subscriber=None,
    verifier=None,
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
                verifier=verifier,
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
                    "status": result.get("geometry_status", "measured"),
                    "label": result["label"],
                    "sam_score": result["sam_score"],
                    "height_m": result.get("height_m"),
                    "camera_dist_m": result.get("camera_dist_m"),
                    "geometry_reason": result.get("geometry_reason"),
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
