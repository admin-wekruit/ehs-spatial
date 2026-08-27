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
        import fal_client

        subscriber = fal_client.subscribe
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
