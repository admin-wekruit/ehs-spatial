"""A/B: MoGe-2 (Replicate) vs MoGe-3 (Modal) on fence-smear cases.

Usage: uv run python scripts/moge3_ab_analysis.py
Metrics on gen-01 (has SAM fence masks):
  (a) depth-edge transition width (10%->90% of step) on 3 scanlines crossing
      fence uprights
  (b) smear fraction: fence-mask pixels with |depth - median fence depth| > 0.5m
"""

import json
import os
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np

from ehs_spatial.providers.sam3 import decode_coco_rle

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "outputs/moge3_eval"


def fence_masks(height: int, width: int) -> list[np.ndarray]:
    """SAM cache RLEs are encoded at the run's canonical frame size (518x392),
    not the source image size — decode there, then upscale nearest."""
    payload = json.loads(
        (ROOT / "runs/gen-01/inventory/sam/frame_0001__safety_fence.json").read_text()
    )
    masks = []
    for rle in payload["rle"]:
        small = decode_coco_rle(rle, height=392, width=518).astype(np.uint8)
        masks.append(
            cv2.resize(small, (width, height), cv2.INTER_NEAREST).astype(bool)
        )
    return masks


def load_moge3(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(EVAL / name / "depth.npz")
    return data["depth"], data["mask"]


def load_moge2(name: str) -> tuple[np.ndarray, np.ndarray]:
    folder = EVAL / "moge2" / name
    depth = cv2.imread(str(folder / "depth_exr__depth.exr"), cv2.IMREAD_UNCHANGED)
    if depth.ndim == 3:
        depth = depth[..., 0]
    valid = cv2.imread(str(folder / "mask_png__mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    valid &= np.isfinite(depth)
    return depth.astype(np.float32), valid


def resize_to(depth, valid, height, width):
    if depth.shape == (height, width):
        return depth, valid
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    valid = (
        cv2.resize(valid.astype(np.uint8), (width, height), cv2.INTER_NEAREST) > 0
    )
    return depth, valid


def transition_widths(depth, valid, mask, scanlines):
    """10%->90% transition width (px) of the depth step at fence-run edges."""
    widths = []
    for y in scanlines:
        row_mask = mask[y]
        edges = np.flatnonzero(np.diff(row_mask.astype(np.int8)))
        for x in edges:
            lo, hi = max(0, x - 12), min(mask.shape[1], x + 13)
            seg_d, seg_v = depth[y, lo:hi], valid[y, lo:hi]
            if seg_v.sum() < 10:
                continue
            seg = np.where(seg_v, seg_d, np.nan)
            d_lo = np.nanpercentile(seg, 5)
            d_hi = np.nanpercentile(seg, 95)
            step = d_hi - d_lo
            if step < 0.5:  # no real depth step here (fence in front of fence etc.)
                continue
            t10, t90 = d_lo + 0.1 * step, d_lo + 0.9 * step
            between = np.flatnonzero((seg > t10) & (seg < t90) & seg_v)
            widths.append(len(between))
    return np.asarray(widths, dtype=float)


def smear_fraction(depth, valid, mask):
    values = depth[mask & valid]
    if len(values) == 0:
        return float("nan"), float("nan")
    median = float(np.median(values))
    return float(np.mean(np.abs(values - median) > 0.5)), median


def per_instance_smear(depth, valid, masks):
    """Global fence median mixes real recession with smear; per SAM instance
    (one post/panel, ~constant depth) isolates actual bleed-through."""
    smeared = total = 0
    for mask in masks:
        values = depth[mask & valid]
        if len(values) < 50:
            continue
        smeared += int(np.sum(np.abs(values - np.median(values)) > 0.5))
        total += len(values)
    return smeared / total if total else float("nan")


def main() -> None:
    name = "gen-01"
    height, width = cv2.imread(str(ROOT / f"incoming/{name}.png")).shape[:2]
    masks = fence_masks(height, width)
    mask = np.logical_or.reduce(masks)

    # 3 scanlines through the fence's vertical extent
    rows = np.flatnonzero(mask.any(axis=1))
    scanlines = [int(np.percentile(rows, p)) for p in (30, 50, 70)]
    print(f"{name}: image {width}x{height}, fence px {mask.sum()}, "
          f"scanlines y={scanlines}")

    results = {}
    for label, loader in [("moge2", load_moge2), ("moge3", load_moge3)]:
        depth, valid = resize_to(*loader(name), height, width)
        tw = transition_widths(depth, valid, mask, scanlines)
        sf, med = smear_fraction(depth, valid, mask)
        results[label] = {
            "edges_measured": int(len(tw)),
            "transition_width_px_median": float(np.median(tw)) if len(tw) else None,
            "transition_width_px_mean": float(np.mean(tw)) if len(tw) else None,
            "smear_fraction_global_median": sf,
            "smear_fraction_per_instance": per_instance_smear(depth, valid, masks),
            "fence_median_depth_m": med,
        }
        print(label, json.dumps(results[label]))

    (EVAL / "ab_metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
