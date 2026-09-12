"""One taxonomy sweep per source photo, then native-box SAM segmentation.

Per-frame content caches make retries and appended photos incremental.
Each successful or failed instance retains its source identity. A single
VLM sweep and SAM score do not establish semantic correctness or safety.
"""

import base64
import hashlib
import tempfile
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .path_safety import validate_safe_path_segment
from .providers.gemini import (
    GEMINI_MODEL_ID,
    GeminiAdapter,
    _image_block,
    _response_format,
    _text_block,
)
from .providers.sam3 import decode_coco_rle
from .taxonomy import CATEGORIES, TAXONOMY

DETECTION_VERSION = "2026-09-09.multiframe-single-sweep-v1"


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


def _write_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def detect_devices(
    run_id: str,
    *,
    runs_root: str | Path = "runs",
    adapter: GeminiAdapter | None = None,
    subscriber=None,
    fresh: bool = False,
) -> dict:
    """One cached taxonomy sweep per input photo, followed by native-box SAM.

    Every returned instance is retained, including failed segmentation. SAM
    success is not an independent semantic verification or a safety verdict.
    """
    validate_safe_path_segment(run_id, "run_id")
    run = Path(runs_root) / run_id
    if not run.resolve().is_relative_to(Path(runs_root).resolve()):
        raise ValueError("run is outside runs_root")
    inputs = sorted((run / "input").glob("image_*"))
    if not inputs:
        raise ValueError("run has no input photographs")
    out_dir = run / "detection"
    out_dir.mkdir(exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    manifest_path = out_dir / "detections.json"
    saved = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    by_id = {t.item_id: t for t in TAXONOMY}
    all_frames, detections, missing, rejected = [], [], [], []
    sweep_calls = sam_calls = 0
    for index, image_path in enumerate(inputs, 1):
        frame_id = f"frame_{index:04d}"
        image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        with Image.open(image_path) as image:
            width, height = image.size
        frame = {"frame_id": frame_id, "source_image_sha256": image_sha,
                 "image_path": image_path.relative_to(run).as_posix(), "image_size": [width, height],
                 "detection_version": DETECTION_VERSION,
                 "overlay_path": f"detection/overlay_{frame_id}.png"}
        key = f"{frame_id}__{image_sha}__{DETECTION_VERSION}"
        cache = frames_dir / f"{key}.json"
        if cache.is_file() and not fresh:
            result = json.loads(cache.read_text())
            if any(result.get(k) != frame[k] for k in ("frame_id", "source_image_sha256", "detection_version", "image_size")):
                raise ValueError(f"detection cache identity mismatch for {frame_id}")
        elif (not fresh and index == 1 and saved and not saved.get("frames")
              and saved.get("image_size") == [width, height]):
            # Frozen first-photo reports predate source hashes. Preserve their
            # evidence explicitly as legacy; they never cover another photo.
            result = {**frame, "source_binding": "legacy_first_frame_unverified",
                      "detections": [], "missing": [], "rejected": []}
            for field in ("detections", "missing", "rejected"):
                result[field] = [{**r, "frame_id": frame_id,
                    "source_image_sha256": None, "source_binding": "legacy_first_frame_unverified",
                    "semantic_verification": "legacy_unrecorded"} for r in saved.get(field, [])]
            _write_json(cache, result)
        else:
            sweep_cache = frames_dir / f"{key}.sweep.json"
            if sweep_cache.is_file() and not fresh:
                sweep = TaxonomySweep.model_validate(json.loads(sweep_cache.read_text()))
            else:
                adapter = adapter or GeminiAdapter()
                sweep = _bulk_sweep(str(image_path), adapter)
                sweep_calls += 1
                _write_json(sweep_cache, sweep.model_dump(mode="json"))
            result = {**frame, "source_binding": "content_hash",
                      "semantic_verification": "single_sweep", "detections": [], "missing": [], "rejected": []}
            seen = set()
            for ordinal, proposed in enumerate(sweep.boxes):
                evidence = {"frame_id": frame_id, "source_image_sha256": image_sha,
                            "image_size": [width, height], "box_2d": list(proposed.box_2d),
                            "item_id": proposed.item_id, "note": proposed.note,
                            "semantic_verification": "single_sweep"}
                spec = by_id.get(proposed.item_id)
                y1, x1, y2, x2 = proposed.box_2d
                if spec is None or min(proposed.box_2d) < 0 or max(proposed.box_2d) > 1000 or not (y1 < y2 and x1 < x2):
                    result["rejected"].append({**evidence, "reason": "unknown taxonomy item" if spec is None else "bad box"})
                    continue
                box = [int(x1 * width / 1000), int(y1 * height / 1000), int(x2 * width / 1000), int(y2 * height / 1000)]
                if box[0] >= box[2] or box[1] >= box[3]:
                    result["rejected"].append({**evidence, "box": box, "reason": "box collapses at source resolution"})
                    continue
                signature = (proposed.item_id, *box)
                if signature in seen:
                    result["rejected"].append({**evidence, "box": box, "reason": "duplicate identical item and box in the same sweep"})
                    continue
                seen.add(signature)
                slug = key + "__" + hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:20]
                det = {**evidence, "box": box, "instance_id": f"{frame_id}:{proposed.item_id}:{ordinal}",
                       "label": spec.sam_label, "category": spec.category, "zh": spec.zh, "iso": spec.iso,
                       "mask_path": f"detection/sam/{slug}.json", "mask_status": "unavailable"}
                try:
                    if not (run / det["mask_path"]).exists():
                        sam_calls += 1
                    response = _sam_box(run, image_path, box, slug, subscriber)
                    rles = response.get("rle") or []
                    if isinstance(rles, str):
                        rles = [rles]
                    if not rles:
                        raise ValueError("SAM returned no mask")
                    scores = response.get("scores") or [1.0] * len(rles)
                    if len(scores) != len(rles) or not np.isfinite(scores).all():
                        raise ValueError("invalid SAM scores")
                    best = int(np.argmax(scores))
                    mask = decode_coco_rle(rles[best], height=height, width=width).astype(bool)
                    if mask.shape != (height, width) or not mask.any():
                        raise ValueError("SAM mask is empty or disagrees with source resolution")
                    det.update(rle=rles[best], sam_score=round(float(scores[best]), 3), mask_status="available",
                               mask_pixels=int(mask.sum()), mask_sha256=hashlib.sha256((run / det["mask_path"]).read_bytes()).hexdigest())
                except Exception as error:
                    det["sam_error"] = str(error)[:200]
                result["detections"].append(det)
            visible = {d["item_id"] for d in result["detections"]}
            result["missing"] = [{"item_id": t.item_id, "label": t.sam_label, "zh": t.zh, "iso": t.iso,
                                  "frame_id": frame_id, "source_image_sha256": image_sha,
                                  "reason": "not_visible_in_single_sweep" if t.item_id in sweep.not_visible else "not_reported_in_single_sweep"}
                                 for t in TAXONOMY if t.item_id not in visible and t.expect != "optional"]
            _write_json(cache, result)
        all_frames.append({**frame, "source_binding": result.get("source_binding"),
                           "cache_path": cache.relative_to(run).as_posix(),
                           "detections_count": len(result["detections"]),
                           "masked_count": sum("rle" in d for d in result["detections"])})
        detections.extend(result["detections"])
        missing.extend(result["missing"])
        rejected.extend(result["rejected"])
    for number, detection in enumerate(detections, 1):
        detection["number"] = number
    envelope = {"run_id": run_id, "detection_version": DETECTION_VERSION, "frames": all_frames,
                "detections": detections, "missing": missing, "rejected": rejected,
                "categories": {k: v[0] for k, v in CATEGORIES.items()},
                "semantic_verification": "single_sweep_or_explicit_legacy",
                "last_execution": {"sweep_calls": sweep_calls, "sam_calls": sam_calls},
                "recall": None, "recall_reason": "No instance-level ground truth is provided"}
    _write_json(manifest_path, envelope)
    render_overlay(run_id, runs_root=runs_root)
    return envelope


def render_overlay(run_id: str, *, runs_root: str | Path = "runs") -> Path:
    """Write one numbered native-resolution overlay per source photograph."""
    from PIL import ImageDraw, ImageFont

    run = Path(runs_root) / run_id
    envelope = json.loads((run / "detection/detections.json").read_text())
    outputs = []
    for frame in envelope["frames"]:
        image_path = run / frame["image_path"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != frame["source_image_sha256"]:
            raise ValueError("source photograph changed before overlay rendering")
        with Image.open(image_path) as original:
            image = np.asarray(original.convert("RGB")).copy()
        height, width = image.shape[:2]
        labels = []
        for det in envelope["detections"]:
            if det.get("frame_id") != frame["frame_id"] or "rle" not in det:
                continue
            colour = np.array(tuple(int(CATEGORIES[det["category"]][1][i:i+2], 16) for i in (1, 3, 5)))
            mask = decode_coco_rle(det["rle"], height=height, width=width).astype(bool)
            if mask.shape != (height, width):
                raise ValueError("detection mask does not match its source photograph")
            image[mask] = (0.5 * image[mask] + 0.5 * colour).astype(np.uint8)
            ys, xs = np.nonzero(mask)
            if len(xs):
                labels.append((int(xs.mean()), int(ys.min()), det["number"]))
        overlay = Image.fromarray(image)
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default(size=max(12, round(width / 55)))
        for cx, cy, number in labels:
            text = str(number)
            bounds = draw.textbbox((cx, cy), text, font=font)
            draw.rectangle(bounds, fill=(20, 20, 20))
            draw.text((cx, cy), text, fill=(255, 235, 59), font=font)
        path = run / frame["overlay_path"]
        overlay.save(path)
        outputs.append(path)
    return outputs[0]


__all__ = ["detect_devices", "render_overlay", "TaxonomySweep", "DetectedBox"]
