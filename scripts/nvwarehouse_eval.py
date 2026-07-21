"""NVIDIA PhysicalAI-Spatial-Intelligence-Warehouse (val) — metric distance eval.

In-domain exam: synthetic warehouse scenes, benchmark-provided GT object
masks, answers in metres. We take the "pure measurement" subset — distance
questions with exactly two masks — so no detection or phrase parsing is
involved: this isolates MapAnything mono metric geometry + our binding.

Both distance semantics are reported (the benchmark's generator is not
documented): centroid-to-centroid and robust min point distance.

Images download individually from HF (needs HF_TOKEN with dataset access);
geometry is disk-cached; spend gated behind --live with a printed count.

Usage:
  uv run --env-file .env python scripts/nvwarehouse_eval.py --limit 12
  uv run --env-file .env python scripts/nvwarehouse_eval.py --limit 12 --live
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ehs_spatial.providers.sam3 import decode_coco_rle

DATA = Path("outputs/datasets/nvwarehouse")
WORK = Path("outputs/nvwarehouse_v1")
BASE = (
    "https://huggingface.co/datasets/nvidia/"
    "PhysicalAI-Spatial-Intelligence-Warehouse/resolve/main"
)


def _questions(limit: int | None) -> list[dict]:
    rows = json.load(open(DATA / "val.json"))
    direct = [
        r
        for r in rows
        if r["category"] == "distance" and len(r.get("rle", [])) == 2
    ]
    direct.sort(key=lambda r: (r["image"], r["id"]))
    return direct[:limit] if limit else direct


def _image_path(image: str) -> Path:
    return DATA / "images" / image


def _fetch_image(image: str) -> None:
    target = _image_path(image)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # curl instead of urllib: this environment's python lacks SSL certs.
    subprocess.run(
        [
            "curl", "-sSfL",
            "-H", f"Authorization: Bearer {os.environ['HF_TOKEN']}",
            "-o", str(target),
            f"{BASE}/val/images/{image}",
        ],
        check=True,
    )


def _geometry_dir(image: str) -> Path:
    return WORK / "geometry" / Path(image).stem


def _run_geometry(image: str) -> None:
    target = _geometry_dir(image)
    if (target / "frames").is_dir():
        return
    from ehs_spatial.providers.base import ProviderError
    from ehs_spatial.providers.map_anything import MapAnythingAdapter

    for attempt in range(6):
        try:
            MapAnythingAdapter().run([str(_image_path(image))], target)
            return
        except ProviderError as exc:
            message = str(exc)
            transient = (
                "429" in message
                or "throttled" in message
                or "502" in message
                or "503" in message
                or "500" in message
            )
            if not transient:
                raise
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"MapAnything kept failing transiently for {image}")


def _mask_points(image: str, rle: dict) -> np.ndarray | None:
    from PIL import Image

    frame_dir = next((_geometry_dir(image) / "frames").iterdir())
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


def _min_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a[:: max(1, len(a) // 3000)]
    b = b[:: max(1, len(b) // 3000)]
    nn = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float(np.percentile(nn, 0.5))


def _ask_vlm(question: dict) -> float | None:
    """SpatialRGPT-style baseline: draw the two GT regions on the image and
    ask the distance in metres."""
    cache = WORK / "vlm" / f"{question['id']}.json"
    if cache.exists():
        return json.loads(cache.read_text()).get("value")
    import io
    import re as _re

    import numpy as np
    from PIL import Image, ImageDraw
    from google import genai
    from google.genai import types

    with Image.open(_image_path(question["image"])) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    colors = ["red", "blue"]
    for index, rle in enumerate(question["rle"]):
        height, width = rle["size"]
        mask = decode_coco_rle(rle["counts"], height=height, width=width)
        ys, xs = np.nonzero(mask)
        draw.rectangle(
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            outline=colors[index],
            width=5,
        )
    text = question["conversations"][0]["value"].replace("<image>\n", "")
    for index in range(2):
        text = text.replace(
            "<mask>", f"[Region {index} in {colors[index]}]", 1
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
    match = _re.search(r"[\d.]+", response.text or "")
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
    images = sorted({q["image"] for q in questions})
    uncached = [
        image for image in images if not (_geometry_dir(image) / "frames").is_dir()
    ]
    print(
        f"{len(questions)} distance questions | {len(images)} images | "
        f"{len(uncached)} uncached MapAnything calls"
    )
    if uncached and not args.live:
        print("pass --live to spend them (cached reruns are free)", file=sys.stderr)
        return 2

    failures = []
    for image in images:
        try:
            _fetch_image(image)
            _run_geometry(image)
        except Exception as exc:  # one bad image must not kill the batch
            failures.append(f"{image}: {exc}")
            print(f"skipping {image}: {exc}", file=sys.stderr)
    questions = [
        q
        for q in questions
        if (_geometry_dir(q["image"]) / "frames").is_dir()
    ]

    rows = []
    for question in questions:
        gt = float(question["normalized_answer"])
        points = [_mask_points(question["image"], rle) for rle in question["rle"]]
        if any(p is None for p in points):
            rows.append(
                {"id": question["id"], "image": question["image"], "gt_m": gt}
            )
            continue
        centroid = float(
            np.linalg.norm(
                np.median(points[0], axis=0) - np.median(points[1], axis=0)
            )
        )
        minimum = _min_distance(points[0], points[1])
        row = {
            "id": question["id"],
            "image": question["image"],
            "gt_m": gt,
            "centroid_m": round(centroid, 3),
            "min_m": round(minimum, 3),
            "centroid_rel": round(abs(centroid - gt) / gt, 3),
            "min_rel": round(abs(minimum - gt) / gt, 3),
        }
        if args.vlm:
            value = _ask_vlm(question)
            if value is not None:
                row["vlm_m"] = round(value, 3)
                row["vlm_rel"] = round(abs(value - gt) / gt, 3)
        rows.append(row)

    def summarize(key: str) -> dict:
        errors = [r[key] for r in rows if key in r]
        if not errors:
            return {"answered": 0}
        return {
            "answered": len(errors),
            "of": len(rows),
            "median_rel_err": round(float(np.median(errors)), 3),
            "success_at_10pct": round(
                sum(1 for e in errors if e <= 0.10) / len(errors), 3
            ),
            "success_at_25pct": round(
                sum(1 for e in errors if e <= 0.25) / len(errors), 3
            ),
        }

    summary = {
        "rows": rows,
        "centroid": summarize("centroid_rel"),
        "min_dist": summarize("min_rel"),
        "vlm": summarize("vlm_rel") if args.vlm else None,
        "scale_source": "MapAnything native metric mono (synthetic warehouse)",
        "image_failures": failures,
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in rows:
        print(row)
    print("centroid:", summary["centroid"])
    print("min_dist:", summary["min_dist"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
