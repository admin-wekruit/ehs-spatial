"""SpatialRGPT-Bench quantitative distance eval (LiDAR/RGB-D/synthetic GT).

Subset: quantitative distance_data / horizontal_distance_data /
vertical_distance_data questions with exactly two benchmark-provided masks
(376 questions; SUNRGBD, Hypersim, ARKitScenes, nuScenes, KITTI). Answers
carry mixed units in free text; parsed to metres.

Harness: MapAnything native metric mono + GT masks -> robust min 3D
distance (vertical questions: separation along the camera-up axis).
Protocol: success = relative error <= 25% (the benchmark's accuracy
threshold) plus median relative error.

Usage:
  uv run --env-file .env python scripts/spatialrgpt_eval.py --limit 10
  uv run --env-file .env python scripts/spatialrgpt_eval.py --limit 10 --live --vlm
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ehs_spatial.providers.sam3 import decode_coco_rle

DATA = Path("outputs/datasets/spatialrgpt")
WORK = Path("outputs/spatialrgpt_v1")
BASE = "https://huggingface.co/datasets/a8cheng/SpatialRGPT-Bench/resolve/main"
CATEGORIES = {
    "distance_data",
    "horizontal_distance_data",
    "vertical_distance_data",
}
UNIT_M = {
    "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0, "m": 1.0,
    "centimeter": 0.01, "centimeters": 0.01, "cm": 0.01,
    "millimeter": 0.001, "millimeters": 0.001, "mm": 0.001,
    "foot": 0.3048, "feet": 0.3048, "ft": 0.3048,
    "inch": 0.0254, "inches": 0.0254, "in": 0.0254,
}


def _parse(value):
    return value if isinstance(value, (dict, list)) else ast.literal_eval(value)


def _gt_metres(answer: str) -> float | None:
    match = re.search(
        r"([\d.]+)\s*(meters?|metres?|centimeters?|millimeters?|feet|foot|"
        r"inches|inch|cm|mm|ft|in|m)\b",
        answer.lower(),
    )
    if not match:
        return None
    return float(match.group(1)) * UNIT_M[match.group(2)]


def _questions(limit: int | None) -> list[dict]:
    rows = json.load(open(DATA / "bench_v1.json"))
    picked = []
    for row in rows:
        qa = _parse(row["qa_info"])
        if qa["type"] != "quantitative" or qa["category"] not in CATEGORIES:
            continue
        rles = _parse(row["rle"])
        if len(rles) != 2:
            continue
        conversations = _parse(row["conversations"])
        gt = _gt_metres(conversations[1]["value"])
        if gt is None or gt <= 0:
            continue
        picked.append(
            {
                "id": row["id"],
                "category": qa["category"],
                "source": _parse(row["image_info"])["dataset"],
                "file_path": row["file_path"],
                "rles": rles,
                "bboxes": _parse(row["bbox"]),
                "question": row["text_q"],
                "gt_m": gt,
            }
        )
    picked.sort(key=lambda r: r["id"])
    return picked[:limit] if limit else picked


def _image_path(file_path: str) -> Path:
    return DATA / file_path


def _fetch_image(file_path: str) -> None:
    target = _image_path(file_path)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-sSfL", "-o", str(target), f"{BASE}/{file_path}"],
        check=True,
    )


def _geometry_dir(file_path: str) -> Path:
    return WORK / "geometry" / Path(file_path).stem


def _run_geometry(file_path: str) -> None:
    target = _geometry_dir(file_path)
    if (target / "frames").is_dir():
        return
    from ehs_spatial.providers.base import ProviderError
    from ehs_spatial.providers.map_anything import MapAnythingAdapter

    for attempt in range(6):
        try:
            MapAnythingAdapter().run([str(_image_path(file_path))], target)
            return
        except ProviderError as exc:
            message = str(exc)
            if not any(code in message for code in ("429", "throttled", "500", "502", "503")):
                raise
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"MapAnything kept failing for {file_path}")


def _frame_dir(file_path: str) -> Path:
    return next((_geometry_dir(file_path) / "frames").iterdir())


def _mask_points(file_path: str, rle: dict) -> np.ndarray | None:
    from PIL import Image

    frame_dir = _frame_dir(file_path)
    points = np.load(frame_dir / "pts3d.npy")
    valid = np.load(frame_dir / "valid_mask.npy").astype(bool)
    height, width = rle["size"]
    mask = decode_coco_rle(rle["counts"], height=height, width=width)
    resized = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (valid.shape[1], valid.shape[0]), Image.NEAREST
        )
    ) > 127
    selected = resized & valid & np.isfinite(points).all(axis=2)
    cloud = points[selected]
    if len(cloud) < 25:
        return None
    cloud = cloud[np.linalg.norm(cloud, axis=1) > 0.15]
    if len(cloud) < 25:
        return None
    ranges = np.linalg.norm(cloud, axis=1)
    median = np.median(ranges)
    mad = np.median(np.abs(ranges - median)) + 1e-6
    cloud = cloud[np.abs(ranges - median) < 3.0 * mad]
    return cloud if len(cloud) >= 25 else None


def _camera_up(file_path: str) -> np.ndarray:
    camera = np.load(_frame_dir(file_path) / "camera_to_world.npy")
    return -np.asarray(camera, dtype=float)[:3, :3][:, 1]


def _min_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a[:: max(1, len(a) // 3000)]
    b = b[:: max(1, len(b) // 3000)]
    nn = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(np.percentile(nn, 0.5))


def _measure(question: dict) -> float | None:
    points = [
        _mask_points(question["file_path"], rle) for rle in question["rles"]
    ]
    if any(p is None for p in points):
        return None
    if question["category"] == "vertical_distance_data":
        up = _camera_up(question["file_path"])
        heights = [float(np.median(p @ up)) for p in points]
        return abs(heights[0] - heights[1])
    return _min_distance(points[0], points[1])


def _ask_vlm(question: dict) -> float | None:
    cache = WORK / "vlm" / f"{question['id']}.json"
    if cache.exists():
        return json.loads(cache.read_text()).get("value")
    import io

    from PIL import Image, ImageDraw
    from google import genai
    from google.genai import types

    with Image.open(_image_path(question["file_path"])) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    colors = ["red", "blue"]
    for index, box in enumerate(question["bboxes"]):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=colors[index], width=4)
    text = question["question"]
    for index in range(2):
        text = text.replace(
            f"Region [{index}]", f"the {colors[index]}-boxed region", 1
        )
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"),
            text + " Respond with ONLY a number in meters.",
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
    args = parser.parse_args(argv)

    questions = _questions(args.limit)
    files = sorted({q["file_path"] for q in questions})
    uncached = [
        f for f in files if not (_geometry_dir(f) / "frames").is_dir()
    ]
    print(
        f"{len(questions)} questions | {len(files)} images | "
        f"{len(uncached)} uncached MapAnything calls"
    )
    if uncached and not args.live:
        print("pass --live to spend them (cached reruns are free)", file=sys.stderr)
        return 2

    failures = []
    for file_path in files:
        try:
            _fetch_image(file_path)
            _run_geometry(file_path)
        except Exception as exc:
            failures.append(f"{file_path}: {exc}")
            print(f"skipping {file_path}: {exc}", file=sys.stderr)
    questions = [
        q for q in questions if (_geometry_dir(q["file_path"]) / "frames").is_dir()
    ]

    rows = []
    for question in questions:
        predicted = _measure(question)
        row = {
            "id": question["id"],
            "category": question["category"],
            "source": question["source"],
            "gt_m": round(question["gt_m"], 3),
            "harness_m": None if predicted is None else round(predicted, 3),
        }
        if predicted is not None:
            row["harness_rel"] = round(
                abs(predicted - question["gt_m"]) / question["gt_m"], 3
            )
        if args.vlm:
            value = _ask_vlm(question)
            if value is not None:
                row["vlm_m"] = round(value, 3)
                row["vlm_rel"] = round(
                    abs(value - question["gt_m"]) / question["gt_m"], 3
                )
        rows.append(row)

    def summarize(key: str, subset=None) -> dict:
        pool = [r for r in rows if subset is None or r["source"] == subset]
        errors = [r[key] for r in pool if r.get(key) is not None]
        if not errors:
            return {"answered": 0, "of": len(pool)}
        return {
            "answered": len(errors),
            "of": len(pool),
            "median_rel_err": round(float(np.median(errors)), 3),
            "success_at_25pct": round(
                sum(1 for e in errors if e <= 0.25) / len(errors), 3
            ),
        }

    summary = {
        "rows": rows,
        "harness": summarize("harness_rel"),
        "vlm": summarize("vlm_rel") if args.vlm else None,
        "harness_by_source": {
            s: summarize("harness_rel", s) for s in sorted({r["source"] for r in rows})
        },
        "image_failures": failures,
        "scale_source": "MapAnything native metric mono (no anchor available)",
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in rows[-8:]:
        print(row)
    print("harness:", summary["harness"])
    if args.vlm:
        print("vlm:", summary["vlm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
