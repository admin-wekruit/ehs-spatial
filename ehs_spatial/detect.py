"""Detection layer: taxonomy-driven object detection in front of SAM.

One bulk VLM pass walks the whole safety-device taxonomy and returns a box
for every visible item (or reports it not visible); every box is verified
with a crop self-check, failed items retry through the single-item locator;
verified boxes become SAM box prompts. SAM never guesses from text here —
it always segments WITH detection context, and coverage is an auditable
checklist with an explicit missing list.

Outputs land in runs/<run>/detection/: detections.json (boxes, masks as
RLE, verdicts) and overlay.png (numbered category-colored masks).
"""

import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .agent import CropCheck, _gemini_locator, _gemini_verify
from .providers.base import ProviderError
from .providers.gemini import (
    GEMINI_MODEL_ID,
    GeminiAdapter,
    _image_block,
    _response_format,
    _text_block,
)
from .providers.sam3 import decode_coco_rle
from .taxonomy import CATEGORIES, TAXONOMY


class DetectedBox(BaseModel):
    item_id: str = Field(description="taxonomy item id, e.g. 'a1'")
    box_2d: tuple[int, int, int, int] = Field(
        description="[ymin, xmin, ymax, xmax], 0-1000 normalized (box_2d)"
    )
    note: str = ""


class TaxonomySweep(BaseModel):
    boxes: list[DetectedBox]
    not_visible: list[str] = Field(
        description="taxonomy item ids that are NOT visible in this photo"
    )


class ExtraSweep(BaseModel):
    boxes: list[DetectedBox]


_MORE_PROMPT = (
    "You are auditing an industrial robot-cell photo for machine-safety "
    "devices. The instances below are ALREADY detected. Find visible "
    "instances of checklist items NOT covered by any existing box — "
    "especially additional side-by-side guard/panel sections next to an "
    "already-boxed one, extra buttons, extra signs. Boxes MUST be box_2d "
    "[ymin, xmin, ymax, xmax] in 0-1000 normalized coordinates, tight "
    "around one device each. Return an empty list when nothing is left.\n"
    "Checklist:\n{checklist}\nAlready detected:\n{found}"
)


class SectionSplit(BaseModel):
    boxes: list[tuple[int, int, int, int]] = Field(
        description="one tight box_2d [ymin, xmin, ymax, xmax] (0-1000, "
        "within THIS crop) per individual section"
    )


_SPLIT_PROMPT = (
    "This crop shows industrial machine guarding. If it contains MULTIPLE "
    "side-by-side guard/fence/panel sections separated by vertical posts, "
    "return one tight box per individual section. If it is a single "
    "section, return exactly one box covering it. Boxes are box_2d "
    "[ymin, xmin, ymax, xmax] in 0-1000 normalized coordinates of THIS "
    "crop."
)


def _box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - inter
    )
    return inter / union if union else 0.0


_SWEEP_PROMPT = (
    "You are auditing an industrial robot-cell photo for machine-safety "
    "devices. Walk the checklist below and return ONE box per visible "
    "instance (an item marked 'multi' may return several boxes; 'pair' "
    "items are listed as separate L/R entries). Boxes MUST be box_2d "
    "[ymin, xmin, ymax, xmax] in 0-1000 normalized coordinates, tight "
    "around the device only — never include an adjacent sloped kick plate "
    "in a guard's box, and never enlarge a clear panel's box to cover "
    "things visible through it. Items you cannot see go in not_visible.\n"
    "Checklist:\n{checklist}"
)


def _bulk_sweep(image_path: str, adapter: GeminiAdapter) -> TaxonomySweep:
    checklist = "\n".join(
        f"- {t.item_id} [{t.expect}]: {t.en}" for t in TAXONOMY
    )
    response = adapter._create(
        "detect.sweep",
        model=GEMINI_MODEL_ID,
        input=[
            _text_block(_SWEEP_PROMPT.format(checklist=checklist)),
            _image_block(image_path),
        ],
        response_format=_response_format(TaxonomySweep),
    )
    sweep, _ = adapter._parse(response, TaxonomySweep, "detect.sweep")
    return sweep


def _sam_box(run: Path, image_path: Path, box, slug: str, subscriber) -> dict:
    cache_dir = run / "detection" / "sam"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{slug}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    if subscriber is None:
        from .providers.sam3 import sam_subscribe

        subscriber = sam_subscribe
    x1, y1, x2, y2 = box
    response = subscriber(
        "fal-ai/sam-3-1/image-rle",
        arguments={
            "image_url": "data:image/png;base64,"
            + base64.b64encode(image_path.read_bytes()).decode(),
            "box_prompts": [
                {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2}
            ],
            "return_multiple_masks": True,
            "include_scores": True,
            "max_masks": 3,
        },
    )
    cache.write_text(json.dumps(response) + "\n")
    return response


def detect_devices(
    run_id: str,
    *,
    runs_root: str | Path = "runs",
    adapter: GeminiAdapter | None = None,
    subscriber=None,
    verifier=None,
) -> dict:
    """Run the taxonomy detection pass over one run's photo. Returns the
    detections envelope (also written to runs/<run>/detection/)."""
    run = Path(runs_root) / run_id
    image_path = next((run / "input").glob("image_*"))
    with Image.open(image_path) as image:
        width, height = image.size
    out_dir = run / "detection"
    out_dir.mkdir(exist_ok=True)
    cache = out_dir / "detections.json"
    if cache.exists():
        return json.loads(cache.read_text())

    if adapter is None:
        adapter = GeminiAdapter()
    if verifier is None:
        def verifier(crop_path, text):  # noqa: F811 - default binding
            return _gemini_verify(crop_path, text, adapter)

    sweep = _bulk_sweep(str(image_path), adapter)
    by_id = {t.item_id: t for t in TAXONOMY}
    detections: list[dict] = []
    rejected: list[dict] = []
    for det in sweep.boxes:
        spec = by_id.get(det.item_id)
        if spec is None:
            continue
        y1, x1, y2, x2 = det.box_2d
        if max(det.box_2d) > 1000 or not (y1 < y2 and x1 < x2):
            rejected.append({"item_id": det.item_id, "reason": "bad box"})
            continue
        box = (
            max(0, int(x1 / 1000 * width)),
            max(0, int(y1 / 1000 * height)),
            min(width, int(x2 / 1000 * width)),
            min(height, int(y2 / 1000 * height)),
        )
        import tempfile

        with Image.open(image_path) as image:
            crop = image.crop(box)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                crop.save(f.name)
                check: CropCheck = verifier(f.name, spec.en)
        if not check.matches:
            rejected.append(
                {"item_id": det.item_id, "reason": f"crop check: {check.reason}"}
            )
            continue
        detections.append(
            {"item_id": det.item_id, "box": list(box), "note": det.note}
        )

    # single-item retry for checklist items with no surviving box that the
    # bulk pass did NOT explicitly mark invisible
    # the bulk pass under-claims: it marks hard items not_visible that the
    # single-item locator finds (floor sensor bar, door interlock — both
    # confirmed present by reviewers). Retry every missing non-optional
    # item; "missing" is only honest after the focused look also fails.
    seen_ids = {d["item_id"] for d in detections}
    for spec in TAXONOMY:
        if spec.item_id in seen_ids or spec.expect == "optional":
            continue
        try:
            located = _gemini_locator(str(image_path), spec.en, adapter)
        except ProviderError:
            continue
        if not located.found or max(located.box_2d) > 1000:
            continue
        y1, x1, y2, x2 = located.box_2d
        if not (y1 < y2 and x1 < x2):
            continue
        detections.append(
            {
                "item_id": spec.item_id,
                "box": [
                    max(0, int(x1 / 1000 * width)),
                    max(0, int(y1 / 1000 * height)),
                    min(width, int(x2 / 1000 * width)),
                    min(height, int(y2 / 1000 * height)),
                ],
                "note": "retry:" + located.rationale[:80],
            }
        )

    # completeness rounds: a 'multi' item with ONE hit never re-asked was
    # how three side-by-side gate sections came back as one — feed the
    # found boxes back and ask ONLY for what they do not cover, until a
    # round adds nothing
    checklist = "\n".join(
        f"- {t.item_id} [{t.expect}]: {t.en}" for t in TAXONOMY
    )
    for _ in range(2):
        found_lines = "\n".join(
            f"- {d['item_id']}: box_2d ["
            f"{int(d['box'][1] / height * 1000)}, "
            f"{int(d['box'][0] / width * 1000)}, "
            f"{int(d['box'][3] / height * 1000)}, "
            f"{int(d['box'][2] / width * 1000)}]"
            for d in detections
        )
        response = adapter._create(
            "detect.more",
            model=GEMINI_MODEL_ID,
            input=[
                _text_block(
                    _MORE_PROMPT.format(
                        checklist=checklist, found=found_lines or "(none)"
                    )
                ),
                _image_block(str(image_path)),
            ],
            response_format=_response_format(ExtraSweep),
        )
        extra, _ = adapter._parse(response, ExtraSweep, "detect.more")
        added = 0
        for det in extra.boxes:
            spec = by_id.get(det.item_id)
            if spec is None:
                continue
            y1, x1, y2, x2 = det.box_2d
            if max(det.box_2d) > 1000 or not (y1 < y2 and x1 < x2):
                continue
            box = (
                max(0, int(x1 / 1000 * width)),
                max(0, int(y1 / 1000 * height)),
                min(width, int(x2 / 1000 * width)),
                min(height, int(y2 / 1000 * height)),
            )
            if any(_box_iou(box, d["box"]) > 0.5 for d in detections):
                continue
            import tempfile

            with Image.open(image_path) as image:
                crop = image.crop(box)
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as f:
                    crop.save(f.name)
                    check = verifier(f.name, spec.en)
            if not check.matches:
                rejected.append(
                    {
                        "item_id": det.item_id,
                        "reason": f"more-round crop check: {check.reason}",
                    }
                )
                continue
            detections.append(
                {"item_id": det.item_id, "box": list(box), "note": "more"}
            )
            added += 1
        if added == 0:
            break

    # granularity: a guard box much wider than tall usually spans SEVERAL
    # side-by-side sections (one bulk box over a three-section gate) —
    # crop it and have the VLM enumerate the individual sections
    split_result: list[dict] = []
    for det in detections:
        spec = by_id[det["item_id"]]
        x1, y1, x2, y2 = det["box"]
        wide = (x2 - x1) > 1.6 * max(1, y2 - y1)
        if spec.category != "C" or spec.expect != "multi" or not wide:
            split_result.append(det)
            continue
        import tempfile

        with Image.open(image_path) as image:
            crop = image.crop((x1, y1, x2, y2))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                crop.save(f.name)
                response = adapter._create(
                    "detect.split",
                    model=GEMINI_MODEL_ID,
                    input=[
                        _text_block(_SPLIT_PROMPT),
                        _image_block(f.name),
                    ],
                    response_format=_response_format(SectionSplit),
                )
        try:
            split, _ = adapter._parse(response, SectionSplit, "detect.split")
        except Exception:
            split_result.append(det)
            continue
        good = [
            b
            for b in split.boxes
            if max(b) <= 1000 and b[0] < b[2] and b[1] < b[3]
        ]
        if len(good) < 2:
            split_result.append(det)
            continue
        for cy1, cx1, cy2, cx2 in good:
            split_result.append(
                {
                    "item_id": det["item_id"],
                    "box": [
                        x1 + int(cx1 / 1000 * (x2 - x1)),
                        y1 + int(cy1 / 1000 * (y2 - y1)),
                        x1 + int(cx2 / 1000 * (x2 - x1)),
                        y1 + int(cy2 / 1000 * (y2 - y1)),
                    ],
                    "note": "split",
                }
            )
    detections = split_result

    # one physical device, one detection: drop any box that near-duplicates
    # an earlier one, whatever checklist item claimed it (the same panel
    # answering both 'clear guard' and 'low rail' is a relabel, not a
    # second device)
    deduped: list[dict] = []
    for det in detections:
        if any(_box_iou(det["box"], d["box"]) > 0.6 for d in deduped):
            continue
        deduped.append(det)
    detections = deduped

    # detection is monotonic across rounds: a crop-verified device from a
    # previous round survives a weaker re-detect (VLM rounds vary; a bad
    # round must add nothing, never subtract)
    previous = out_dir / "detections.prev.json"
    if previous.exists():
        for old in json.loads(previous.read_text()).get("detections", []):
            if "rle" not in old:
                continue
            if any(
                _box_iou(old["box"], d["box"]) > 0.5 for d in detections
            ):
                continue
            detections.append(
                {
                    "item_id": old["item_id"],
                    "box": old["box"],
                    "note": "carried",
                }
            )
        previous.unlink()

    # SAM with detection context: every verified box becomes a box prompt
    for index, det in enumerate(detections):
        spec = by_id[det["item_id"]]
        slug = f"{det['item_id']}_{'_'.join(str(v) for v in det['box'])}"
        try:
            response = _sam_box(run, image_path, det["box"], slug, subscriber)
        except Exception as error:
            det["sam_error"] = str(error)[:120]
            continue
        rles = response.get("rle") or []
        if isinstance(rles, str):
            rles = [rles]
        if not rles:
            det["sam_error"] = "no mask"
            continue
        scores = response.get("scores") or [1.0] * len(rles)
        best = int(np.argmax(scores))
        det["rle"] = rles[best]
        det["sam_score"] = round(float(scores[best]), 3)
        det["number"] = index + 1
        det["label"] = spec.sam_label
        det["category"] = spec.category
        det["zh"] = spec.zh
        det["iso"] = spec.iso

    missing = [
        {"item_id": t.item_id, "zh": t.zh, "iso": t.iso}
        for t in TAXONOMY
        if t.item_id not in {d["item_id"] for d in detections}
        and t.expect != "optional"
    ]
    envelope = {
        "run_id": run_id,
        "image_size": [width, height],
        "detections": detections,
        "missing": missing,
        "rejected": rejected,
        "categories": {k: v[0] for k, v in CATEGORIES.items()},
    }
    cache.write_text(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n")
    render_overlay(run_id, runs_root=runs_root)
    return envelope


def render_overlay(run_id: str, *, runs_root: str | Path = "runs") -> Path:
    """Numbered, category-colored mask overlay like a machine-safety audit
    sheet. Reads detection/detections.json, writes detection/overlay.png."""
    from PIL import ImageDraw

    run = Path(runs_root) / run_id
    envelope = json.loads((run / "detection" / "detections.json").read_text())
    width, height = envelope["image_size"]
    image = np.asarray(
        Image.open(next((run / "input").glob("image_*"))).convert("RGB")
    ).copy()
    labels = []
    for det in envelope["detections"]:
        if "rle" not in det:
            continue
        colour = np.array(
            tuple(
                int(CATEGORIES[det["category"]][1][i : i + 2], 16)
                for i in (1, 3, 5)
            )
        )
        mask = decode_coco_rle(det["rle"], height=height, width=width).astype(
            bool
        )
        image[mask] = (0.5 * image[mask] + 0.5 * colour).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(xs):
            labels.append((int(xs.mean()), int(ys.min()), det["number"]))
    overlay = Image.fromarray(image)
    draw = ImageDraw.Draw(overlay)
    from PIL import ImageFont

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 56)
    except OSError:
        font = ImageFont.load_default()
    for cx, cy, number in labels:
        text = str(number)
        tw = draw.textlength(text, font=font)
        top = max(0, cy - 12)
        draw.rectangle(
            [cx - tw / 2 - 10, top, cx + tw / 2 + 10, top + 66],
            fill=(20, 20, 20),
        )
        draw.text((cx - tw / 2, top + 4), text, fill=(255, 235, 59), font=font)
    path = run / "detection" / "overlay.png"
    overlay.save(path)
    return path


__all__ = ["detect_devices", "render_overlay", "TaxonomySweep", "DetectedBox"]
