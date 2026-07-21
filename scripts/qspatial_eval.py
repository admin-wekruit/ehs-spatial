"""Q-Spatial-Bench (Q-Spatial++) eval: harness vs VLM on tape-measured GT.

Protocol follows the benchmark: a prediction succeeds when
max(pred/gt, gt/pred) <= 2. We additionally report median relative error.

Harness path per question: Gemini parses the two object phrases; MapAnything
reconstructs the single photo (native metric scale — no camera height is
given); fal SAM 3.1 masks each phrase; the answer is the robust minimum
3D distance between the two mask point clouds. No floor fit: many scenes
are tabletop close-ups, so this evaluates the measurement primitive
(metric mono geometry x segmentation binding), not the clearance rule.

Every provider response is disk-cached; re-scoring is free.

Usage:
  uv run --env-file .env python scripts/qspatial_eval.py            # cost preview
  uv run --env-file .env python scripts/qspatial_eval.py --live --limit 6
  uv run --env-file .env python scripts/qspatial_eval.py --live --vlm
"""

import argparse
import base64
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path("outputs/datasets/qspatial")
WORK = Path("outputs/qspatial_v1")
UNIT_M = {"centimeter": 0.01, "meter": 1.0, "millimeter": 0.001, "inch": 0.0254}


def _questions(limit: int | None) -> list[dict]:
    rows = json.loads((DATA / "questions_plus.json").read_text())
    if limit:
        seen_images: set[str] = set()
        picked = []
        for row in rows:
            if row["image"] in seen_images:
                continue
            seen_images.add(row["image"])
            picked.append(row)
            if len(picked) == limit:
                break
        return picked
    return rows


def _parse_objects(question: dict) -> Path:
    return WORK / "parse" / f"{question['qid']}.json"


def _geometry_dir(image: str) -> Path:
    return WORK / "geometry" / Path(image).stem


def _sam_cache(image: str, phrase: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
    return WORK / "sam" / f"{Path(image).stem}__{slug}.json"


def _count_uncached(questions: list[dict]) -> tuple[int, int, int]:
    parse_calls = sum(1 for q in questions if not _parse_objects(q).exists())
    map_calls = len(
        {
            q["image"]
            for q in questions
            if not (_geometry_dir(q["image"]) / "frames").is_dir()
        }
    )
    sam_calls = 0
    for q in questions:
        parse_path = _parse_objects(q)
        if parse_path.exists():
            phrases = json.loads(parse_path.read_text())
            sam_calls += sum(
                1
                for phrase in (phrases["object_a"], phrases["object_b"])
                if not _sam_cache(q["image"], phrase).exists()
            )
        else:
            sam_calls += 2
    return parse_calls, map_calls, sam_calls


def _run_parse(question: dict) -> dict:
    path = _parse_objects(question)
    if path.exists():
        return json.loads(path.read_text())
    from google import genai

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=(
            "Extract the two object phrases whose distance this question "
            "asks about. Respond with ONLY JSON "
            '{"object_a": "...", "object_b": "..."} using short noun '
            f"phrases suitable as segmentation prompts.\nQuestion: "
            f"{question['question']}"
        ),
    )
    match = re.search(r"\{.*\}", response.text or "", re.S)
    payload = json.loads(match.group(0))
    payload = {
        "object_a": str(payload["object_a"]),
        "object_b": str(payload["object_b"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return payload


def _run_geometry(image: str) -> None:
    import time

    target = _geometry_dir(image)
    if (target / "frames").is_dir():
        return
    from ehs_spatial.providers.base import ProviderError
    from ehs_spatial.providers.map_anything import MapAnythingAdapter

    # Low-credit replicate accounts are throttled to 6 predictions/min with
    # burst 1: retry with backoff instead of dying mid-batch.
    for attempt in range(6):
        try:
            MapAnythingAdapter().run([str(DATA / "images" / image)], target)
            return
        except ProviderError as exc:
            if "429" not in str(exc) and "throttled" not in str(exc):
                raise
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"MapAnything kept throttling for {image}")


def _run_sam(image: str, phrase: str) -> None:
    cache = _sam_cache(image, phrase)
    if cache.exists():
        return
    import fal_client

    from ehs_spatial.providers.sam3 import SAM3_ENDPOINT

    frame_dir = next((_geometry_dir(image) / "frames").iterdir())
    canonical = (frame_dir / "canonical.png").read_bytes()
    response = fal_client.subscribe(
        SAM3_ENDPOINT,
        arguments={
            "image_url": "data:image/png;base64,"
            + base64.b64encode(canonical).decode("ascii"),
            "prompt": phrase,
            "return_multiple_masks": True,
            "include_scores": True,
            "include_boxes": True,
            "max_masks": 3,
        },
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(response) + "\n")


def _mask_points(image: str, phrase: str) -> np.ndarray | None:
    from ehs_spatial.providers.sam3 import decode_coco_rle

    frame_dir = next((_geometry_dir(image) / "frames").iterdir())
    points = np.load(frame_dir / "pts3d.npy")
    valid = np.load(frame_dir / "valid_mask.npy").astype(bool)
    height, width = valid.shape
    response = json.loads(_sam_cache(image, phrase).read_text())
    rles = response.get("rle") or []
    if isinstance(rles, str):
        rles = [rles]
    scores = response.get("scores") or [1.0] * len(rles)
    best = None
    for index, rle in enumerate(rles):
        mask = decode_coco_rle(rle, height=height, width=width)
        if mask.sum() < 25:
            continue
        score = scores[index] if index < len(scores) else 0.0
        if best is None or score > best[0]:
            best = (score, mask)
    if best is None:
        return None
    selected = best[1].astype(bool) & valid & np.isfinite(points).all(axis=2)
    cloud = points[selected]
    if len(cloud) < 25:
        return None
    # Zero/near-zero range points are reconstruction garbage, not geometry;
    # answering from them is the false-measurement mode — drop, and abstain
    # below if too little remains.
    cloud = cloud[np.linalg.norm(cloud, axis=1) > 0.15]
    if len(cloud) < 25:
        return None
    ranges = np.linalg.norm(cloud, axis=1)
    median = np.median(ranges)
    mad = np.median(np.abs(ranges - median)) + 1e-6
    cloud = cloud[np.abs(ranges - median) < 3.0 * mad]
    return cloud if len(cloud) >= 25 else None


def _min_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a[:: max(1, len(a) // 3000)]
    b = b[:: max(1, len(b) // 3000)]
    nn = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(np.percentile(nn, 0.5))


def _ask_vlm(question: dict) -> float | None:
    cache = WORK / "vlm" / f"{question['qid']}.json"
    if cache.exists():
        return json.loads(cache.read_text()).get("value")
    from google import genai
    from google.genai import types

    client = genai.Client()
    image_bytes = (DATA / "images" / question["image"]).read_bytes()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            question["question"]
            + f" Respond with ONLY a number in {question['answer_unit']}s.",
        ],
    )
    match = re.search(r"[\d.]+", response.text or "")
    value = float(match.group(0)) if match else None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"value": value}) + "\n")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vlm", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    questions = _questions(args.limit)
    parse_calls, map_calls, sam_calls = _count_uncached(questions)
    total = parse_calls + map_calls + sam_calls
    print(
        f"{len(questions)} questions | uncached calls: {parse_calls} parse "
        f"+ {map_calls} MapAnything + ~{sam_calls} SAM"
    )
    if total and not args.live:
        print("pass --live to spend them (cached reruns are free)", file=sys.stderr)
        return 2

    for question in questions:
        question["objects"] = _run_parse(question)
    # MapAnything serially (replicate throttles low-credit accounts hard);
    # fal SAM in parallel.
    for image in sorted({q["image"] for q in questions}):
        _run_geometry(image)
    with ThreadPoolExecutor(args.workers) as pool:
        pairs = {
            (q["image"], phrase)
            for q in questions
            for phrase in (q["objects"]["object_a"], q["objects"]["object_b"])
        }
        list(pool.map(lambda pair: _run_sam(*pair), pairs))

    rows = []
    for question in questions:
        gt_m = float(question["answer_value"]) * UNIT_M[question["answer_unit"]]
        points_a = _mask_points(question["image"], question["objects"]["object_a"])
        points_b = _mask_points(question["image"], question["objects"]["object_b"])
        predicted = (
            _min_distance(points_a, points_b)
            if points_a is not None and points_b is not None
            else None
        )
        row = {
            "qid": question["qid"],
            "image": question["image"],
            "type": question["question_type"],
            "objects": [
                question["objects"]["object_a"],
                question["objects"]["object_b"],
            ],
            "gt_m": round(gt_m, 4),
            "harness_m": None if predicted is None else round(predicted, 4),
        }
        if predicted is not None and gt_m > 0:
            # A zero prediction is an answered failure, not an abstention:
            # clamp so the ratio blows up instead of dividing by zero.
            bounded = max(predicted, 1e-6)
            row["harness_ratio"] = round(max(bounded / gt_m, gt_m / bounded), 2)
        if args.vlm:
            value = _ask_vlm(question)
            vlm_m = (
                None
                if value is None
                else value * UNIT_M[question["answer_unit"]]
            )
            row["vlm_m"] = None if vlm_m is None else round(vlm_m, 4)
            if vlm_m and gt_m > 0:
                row["vlm_ratio"] = round(max(vlm_m / gt_m, gt_m / vlm_m), 2)
        rows.append(row)

    def summarize(key: str) -> dict:
        ratios = [r[key] for r in rows if r.get(key) is not None]
        answered = len(ratios)
        return {
            "answered": answered,
            "of": len(rows),
            "success_at_2x": round(
                sum(1 for value in ratios if value <= 2.0) / answered, 3
            )
            if answered
            else None,
            "median_ratio": round(float(np.median(ratios)), 2) if ratios else None,
        }

    summary = {
        "rows": rows,
        "harness": summarize("harness_ratio"),
        "vlm": summarize("vlm_ratio") if args.vlm else None,
        "protocol": "success = max(pred/gt, gt/pred) <= 2 (benchmark delta=2)",
        "scale_source": "MapAnything native metric (no camera height given)",
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in rows:
        print(row)
    print("harness:", summary["harness"])
    if args.vlm:
        print("vlm:", summary["vlm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
