"""Human-in-the-loop re-segmentation: box a region, SAM segments it, the run's
own geometry measures it.

The open-vocabulary text prompts miss whole object classes (light-curtain
pillars scored 0 on every phrase tried; a box prompt scores 0.91), so a
reviewer who sees a miss draws a box instead of fighting the vocabulary.

Usage (pixel coordinates in the original image):
  uv run --env-file .env python scripts/refine_region.py \
      --run gen-01 --label "safety sensor" --box 289,228,434,705
Results land in runs/<run>/refinements.json + a tinted mask png; re-runs with
the same box+label are served from the cached SAM response ($0).
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ehs_spatial.geometry import _build_geometry
from ehs_spatial.providers.sam3 import decode_coco_rle
from ehs_spatial.viewer import _frames

MIN_POINTS = 60
MAD_CLIP = 2.5


def measure(run: Path, mask: np.ndarray, camera_height: float, scale: float | None):
    frames = _frames(run)
    frame = frames[0]
    transform = _build_geometry(
        frames, [], camera_height, scale_factor_override=scale
    ).transform
    if transform is None:
        raise SystemExit("run has no floor transform")
    points3d = np.load(frame.pts3d_path)
    valid = np.load(frame.valid_mask_path).astype(bool)
    if mask.shape != valid.shape:
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (valid.shape[1], valid.shape[0])
            )
        ) > 127
    chosen = (
        mask
        & valid
        & np.isfinite(points3d).all(axis=2)
        # MapAnything fills invalid pixels with camera-frame zeros; they all
        # transform to one point and poison any robust statistic.
        & (np.abs(points3d).sum(axis=2) > 1e-6)
    )
    if chosen.sum() < MIN_POINTS:
        raise SystemExit(f"only {int(chosen.sum())} 3D points under the mask")
    # depth bleed through/around transparent structures: a boxed object is
    # one physical thing, so keep the nearest depth cluster (p10 + 1 m)
    # rather than the background the mask leaks onto.
    depth = points3d[..., 2]
    near = np.percentile(depth[chosen], 10)
    chosen &= depth <= near + 1.0
    cloud = transform.apply(points3d[chosen])
    if len(cloud) < MIN_POINTS:
        raise SystemExit("mask collapsed after depth-band clipping")
    top = float(np.percentile(cloud[:, 2], 98))
    base = float(np.percentile(cloud[:, 2], 2))
    xy = cloud[:, :2]
    return {
        "points": int(len(cloud)),
        "height_m": round(top, 2),
        "base_m": round(base, 2),
        "extent_m": f"{np.ptp(xy[:, 0]):.2f}x{np.ptp(xy[:, 1]):.2f}",
        "centroid_xy": [round(float(v), 2) for v in xy.mean(axis=0)],
        "camera_dist_m": round(float(np.linalg.norm(xy.mean(axis=0))), 2),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--box", required=True, help="x1,y1,x2,y2 pixels")
    parser.add_argument("--camera-height", type=float, default=1.5)
    args = parser.parse_args(argv)

    run = Path("runs") / args.run
    image_path = next((run / "input").glob("image_*"))
    with Image.open(image_path) as image:
        width, height = image.size
    x1, y1, x2, y2 = (int(v) for v in args.box.split(","))

    out_dir = run / "refinements"
    out_dir.mkdir(exist_ok=True)
    slug = f"{args.label.replace(' ', '_')}_{x1}_{y1}_{x2}_{y2}"
    cache = out_dir / f"{slug}.json"
    if cache.exists():
        response = json.loads(cache.read_text())
    else:
        import fal_client

        response = fal_client.subscribe(
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
    rles = response.get("rle") or []
    if isinstance(rles, str):
        rles = [rles]
    if not rles:
        raise SystemExit("SAM returned no mask for that box")
    scores = response.get("scores") or [1.0] * len(rles)
    best = int(np.argmax(scores))
    mask = decode_coco_rle(rles[best], height=height, width=width).astype(bool)

    scene_path = run / "scene.json"
    scale = None
    if scene_path.exists():
        scale = json.loads(scene_path.read_text()).get("scale_factor")
    result = {
        "label": args.label,
        "box": [x1, y1, x2, y2],
        "sam_score": round(float(scores[best]), 3),
        "mask_pixels": int(mask.sum()),
        **measure(run, mask, args.camera_height, scale),
    }

    # tinted mask overlay for the evidence trail
    with Image.open(image_path) as image:
        overlay = np.asarray(image.convert("RGB")).copy()
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([64, 156, 255])).astype(
        np.uint8
    )
    Image.fromarray(overlay).save(out_dir / f"{slug}.png")

    log_path = run / "refinements.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    log.append(result)
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"overlay: {out_dir / f'{slug}.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
