"""Can a second metric model supply the scale anchor automatically?

The boundary map measured the harness's one real gap: MapAnything's
unanchored mono scale is off by a tight constant (gt/pred 2.50, MAD 0.24)
on the NVIDIA warehouse benchmark, and a single global correction takes
held-out median relative error from 60% to 8.5%. That correction currently
comes from the operator (camera height). This asks whether MoGe-2 — an
independent MIT-licensed metric model — can supply it instead, per image,
with no operator input at all.

Method: for each image, MoGe-2 returns a metric point cloud in the camera
frame. MapAnything's cached cloud is transformed to the same frame. The
ratio of their median ranges is a single scalar per image; MapAnything
distances are multiplied by it and the benchmark's 105 distance questions
are re-scored.

Only the MoGe-2 call costs money and the per-image scalar is cached, so
re-scoring is free. Point clouds are streamed and discarded — 2M points
per image is not worth keeping.

Usage:
  uv run --env-file .env python scripts/moge_anchor_eval.py
  uv run --env-file .env python scripts/moge_anchor_eval.py --live --limit 12
"""

import argparse
import base64
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nvwarehouse_eval import (  # noqa: E402
    _geometry_dir,
    _image_path,
    _mask_points,
    _min_distance,
    _questions,
)

from ehs_spatial.providers.moge import MOGE_VERSION  # single source of truth
WORK = Path("outputs/nvwarehouse_v1")
CACHE = WORK / "moge_scale"


def _cache_path(image: str) -> Path:
    return CACHE / f"{Path(image).stem}.json"


def _mapanything_median_range(image: str) -> float | None:
    """Median distance from the camera over MapAnything's cached cloud."""
    frame_dir = next((_geometry_dir(image) / "frames").iterdir())
    points = np.load(frame_dir / "pts3d.npy")
    valid = np.load(frame_dir / "valid_mask.npy").astype(bool)
    camera_to_world = np.load(frame_dir / "camera_to_world.npy")
    finite = valid & np.isfinite(points).all(axis=2)
    world = points[finite]
    if not len(world):
        return None
    camera = (
        np.linalg.inv(camera_to_world) @ np.c_[world, np.ones(len(world))].T
    )[:3].T
    return float(np.median(np.linalg.norm(camera, axis=1)))


def _moge_median_range(image: str, *, live: bool) -> float | None:
    cache = _cache_path(image)
    if cache.exists():
        return json.loads(cache.read_text()).get("moge_median_range_m")
    if not live:
        return None
    import replicate
    import open3d as o3d

    payload = "data:image/png;base64," + base64.b64encode(
        _image_path(image).read_bytes()
    ).decode("ascii")
    for attempt in range(5):
        try:
            output = replicate.run(
                MOGE_VERSION, input={"image": payload, "fp16": True}, wait=False
            )
            break
        except Exception as exc:
            if not any(
                code in str(exc) for code in ("429", "throttled", "500", "502", "503")
            ):
                raise
            time.sleep(10 * (attempt + 1))
    else:
        return None

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=True) as handle:
        handle.write(output["pointcloud_ply"].read())
        handle.flush()
        cloud = np.asarray(o3d.io.read_point_cloud(handle.name).points)
    if not len(cloud):
        return None
    median = float(np.median(np.linalg.norm(cloud, axis=1)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "moge_median_range_m": median,
                "points": int(len(cloud)),
                "fov_x_deg": output.get("fov_x_deg"),
            }
        )
        + "\n"
    )
    return median


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    questions = _questions(args.limit)
    questions = [
        q for q in questions if (_geometry_dir(q["image"]) / "frames").is_dir()
    ]
    images = sorted({q["image"] for q in questions})
    uncached = [i for i in images if not _cache_path(i).exists()]
    print(f"{len(questions)} questions | {len(images)} images | "
          f"{len(uncached)} uncached MoGe-2 calls")
    if uncached and not args.live:
        print("pass --live to spend them (cached re-scoring is free)", file=sys.stderr)
        return 2

    scales: dict[str, float] = {}
    for image in images:
        moge = _moge_median_range(image, live=args.live)
        mapany = _mapanything_median_range(image)
        if moge is None or not mapany:
            continue
        scales[image] = moge / mapany

    if not scales:
        print("no scale estimates", file=sys.stderr)
        return 1
    values = np.asarray(list(scales.values()))
    print(
        f"\nMoGe/MapAnything scale: median {np.median(values):.3f} | "
        f"MAD {np.median(np.abs(values - np.median(values))):.3f} | "
        f"p10 {np.quantile(values, 0.1):.2f} | p90 {np.quantile(values, 0.9):.2f}"
    )

    rows = []
    for question in questions:
        gt = float(question["normalized_answer"])
        scale = scales.get(question["image"])
        clouds = [_mask_points(question["image"], rle) for rle in question["rle"]]
        if any(cloud is None for cloud in clouds) or scale is None:
            continue
        raw_centroid = float(
            np.linalg.norm(
                np.median(clouds[0], axis=0) - np.median(clouds[1], axis=0)
            )
        )
        raw_min = _min_distance(clouds[0], clouds[1])
        rows.append(
            {
                "id": question["id"],
                "image": question["image"],
                "gt_m": gt,
                "scale": round(scale, 3),
                "raw_centroid_rel": round(abs(raw_centroid - gt) / gt, 3),
                "moge_centroid_rel": round(
                    abs(raw_centroid * scale - gt) / gt, 3
                ),
                "moge_min_rel": round(abs(raw_min * scale - gt) / gt, 3),
                "moge_centroid_m": round(raw_centroid * scale, 3),
            }
        )

    def summarize(key: str) -> dict:
        errors = [r[key] for r in rows if r.get(key) is not None]
        if not errors:
            return {"answered": 0}
        return {
            "answered": len(errors),
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
        "scale_stats": {
            "median": round(float(np.median(values)), 3),
            "mad": round(float(np.median(np.abs(values - np.median(values)))), 3),
            "n_images": len(scales),
        },
        "raw_unanchored": summarize("raw_centroid_rel"),
        "moge_anchored_centroid": summarize("moge_centroid_rel"),
        "moge_anchored_min": summarize("moge_min_rel"),
        "reference": {
            "operator_anchor_split_half": "median 0.085 / success@25 0.906",
            "vlm_baseline": "median 0.332 / success@25 0.314",
        },
    }
    (WORK / "moge_anchor_report.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("raw (no anchor)      :", summary["raw_unanchored"])
    print("MoGe-anchored (centroid):", summary["moge_anchored_centroid"])
    print("MoGe-anchored (min dist):", summary["moge_anchored_min"])
    print("reference             :", summary["reference"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
