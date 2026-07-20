"""V3/E1 per-class recall eval for promptable segmentation on real EHS imagery.

Datasets: cylinders (bbox, COCO), loco (bbox, COCO), fence (pixel masks).
Provider: fal SAM 3.1 via the same request shape as production, one prompt per
call, with a per-class prompt ensemble. Responses are cached on disk so every
re-score is free (pay once, iterate free).

Metrics
  bbox mode: GT box is a hit if any predicted-mask bbox reaches IoU >= 0.3
             (0.5 also reported); plus per-image presence recall and
             false-positive mask count (no GT overlap at IoU 0.1).
  mask mode: union-of-predictions vs GT mask — presence (coverage >= 0.2),
             pixel IoU, and coverage (intersection / GT area).
Per class, every prompt is scored alone AND as an ensemble union.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehs_spatial.providers.sam3 import SAM3_ENDPOINT, decode_coco_rle  # noqa: E402

MAX_SIDE = 1024
COST_PER_CALL_USD = 0.01

DATASETS_ROOT = Path("outputs/datasets")

CONFIG = {
    "cylinders": {
        "mode": "bbox",
        "coco": [
            ("dataset/train/_annotations.coco.json", "dataset/train"),
            ("dataset/valid/_annotations.coco.json", "dataset/valid"),
            ("dataset/test/_annotations.coco.json", "dataset/test"),
        ],
        "classes": {"gas_cylinder": ["gas cylinder", "gas bottle", "lpg cylinder"]},
    },
    "loco": {
        "mode": "bbox",
        "coco": [("rgb/loco-all-v1.json", None)],  # None -> basename index
        "classes": {
            "pallet": ["pallet", "wooden pallet"],
            "forklift": ["forklift", "fork lift truck"],
        },
    },
    "fence": {
        "mode": "mask",
        "pairs_dir": ("IITKGP_fence/Labeled/Imgs", "IITKGP_fence/Labeled/GT"),
        "classes": {"fence": ["fence", "chain link fence", "barrier"]},
    },
}


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _mask_bbox(mask: np.ndarray) -> tuple | None:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def score_bbox_image(
    gt_boxes: list[tuple], pred_masks: list[np.ndarray]
) -> dict:
    pred_boxes = [b for b in (_mask_bbox(m) for m in pred_masks) if b is not None]
    hits30 = hits50 = 0
    for gt in gt_boxes:
        best = max((_bbox_iou(gt, p) for p in pred_boxes), default=0.0)
        hits30 += best >= 0.3
        hits50 += best >= 0.5
    false_pos = sum(
        1
        for p in pred_boxes
        if max((_bbox_iou(g, p) for g in gt_boxes), default=0.0) < 0.1
    )
    return {
        "gt": len(gt_boxes),
        "hits_iou30": hits30,
        "hits_iou50": hits50,
        "present": int(bool(gt_boxes) and bool(pred_boxes)),
        "false_positive_masks": false_pos,
    }


def score_mask_image(gt_mask: np.ndarray, pred_masks: list[np.ndarray]) -> dict:
    union = np.zeros_like(gt_mask, dtype=bool)
    for m in pred_masks:
        union |= m.astype(bool)
    gt = gt_mask.astype(bool)
    inter = float((union & gt).sum())
    gt_area = float(gt.sum())
    u_area = float((union | gt).sum())
    coverage = inter / gt_area if gt_area else 0.0
    return {
        "coverage": coverage,
        "iou": inter / u_area if u_area else 0.0,
        "present": int(coverage >= 0.2),
    }


def _load_bbox_items(root: Path, spec: dict, class_name: str) -> list[dict]:
    items = []
    for ann_rel, images_rel in spec["coco"]:
        data = json.loads((root / ann_rel).read_text(encoding="utf-8"))
        cat_ids = {
            c["id"] for c in data["categories"] if c["name"] == class_name
        }
        boxes_by_image: dict[int, list[tuple]] = {}
        for ann in data["annotations"]:
            if ann["category_id"] in cat_ids:
                x, y, w, h = ann["bbox"]
                boxes_by_image.setdefault(ann["image_id"], []).append(
                    (x, y, x + w, y + h)
                )
        if images_rel is None:
            index = {p.name: p for p in root.rglob("*.jpg")}
        for image in data["images"]:
            if image["id"] not in boxes_by_image:
                continue
            path = (
                index.get(image["file_name"].split("/")[-1])
                if images_rel is None
                else root / images_rel / image["file_name"]
            )
            if path is None or not Path(path).is_file():
                continue
            items.append(
                {
                    "id": Path(image["file_name"]).stem,
                    "path": str(path),
                    "boxes": boxes_by_image[image["id"]],
                }
            )
    return sorted(items, key=lambda item: item["id"])


def _load_mask_items(root: Path, spec: dict) -> list[dict]:
    imgs_dir = root / spec["pairs_dir"][0]
    gt_dir = root / spec["pairs_dir"][1]
    gt_by_stem = {p.stem: p for p in gt_dir.iterdir() if p.suffix.lower() == ".png"}
    items = []
    for img in sorted(imgs_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        gt = gt_by_stem.get(img.stem)
        if gt is not None:
            items.append({"id": img.stem, "path": str(img), "gt_mask": str(gt)})
    return items


def _prepare_image(path: str) -> tuple[str, float, tuple[int, int]]:
    """Return (data_uri, scale, (width, height)) with long side <= MAX_SIDE."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = min(1.0, MAX_SIDE / max(im.size))
        if scale < 1.0:
            im = im.resize(
                (round(im.width * scale), round(im.height * scale)),
                Image.Resampling.BILINEAR,
            )
        import io

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return uri, scale, im.size


def _cache_path(cache_dir: Path, item_id: str, prompt: str) -> Path:
    slug = prompt.replace(" ", "_")
    safe = "".join(ch for ch in item_id if ch.isalnum() or ch in "-_.")
    return cache_dir / f"{safe}__{slug}.json"


def _masks_from_response(response: dict, width: int, height: int) -> list[np.ndarray]:
    rles = response.get("rle") or []
    if isinstance(rles, str):
        rles = [rles]
    masks = []
    for serialized in rles:
        mask = decode_coco_rle(serialized, height=height, width=width)
        if mask.shape != (height, width):
            mask = np.asarray(
                Image.fromarray(mask * 255).resize(
                    (width, height), Image.Resampling.NEAREST
                )
            ) > 0
        masks.append(mask.astype(bool))
    return masks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(CONFIG))
    parser.add_argument("--n", type=int, default=25, help="images per class")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--out", default="outputs/recall_v1")
    args = parser.parse_args(argv)

    spec = CONFIG[args.dataset]
    root = DATASETS_ROOT / args.dataset
    out = Path(args.out)
    report: dict = {"dataset": args.dataset, "n_per_class": args.n, "seed": args.seed}
    rows = []

    for class_name, prompts in spec["classes"].items():
        if spec["mode"] == "bbox":
            items = _load_bbox_items(root, spec, class_name)
        else:
            items = _load_mask_items(root, spec)
        rng = np.random.default_rng(args.seed)
        picks = [items[i] for i in rng.permutation(len(items))[: args.n]]
        # v2: max_masks=32, the fal cap (v1 cache used the default of 3)
        cache_dir = out / "cache-v2" / args.dataset / class_name.replace(" ", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)

        planned = [
            (item, prompt)
            for item in picks
            for prompt in prompts
            if not _cache_path(cache_dir, item["id"], prompt).is_file()
        ]
        if planned and not args.live:
            print(
                f"{class_name}: {len(planned)} uncached calls "
                f"(~${len(planned) * COST_PER_CALL_USD:.2f}); rerun with --live",
                file=sys.stderr,
            )
            return 2
        if planned:
            import fal_client

            for item, prompt in planned:
                uri, _, (w, h) = _prepare_image(item["path"])
                response = fal_client.subscribe(
                    SAM3_ENDPOINT,
                    arguments={
                        "image_url": uri,
                        "prompt": prompt,
                        "return_multiple_masks": True,
                        "include_scores": True,
                        "include_boxes": True,
                        # fal default max_masks=3 caps recall at ~10% in crowded
                        # warehouse scenes (~30 GT objects per LOCO image).
                        "max_masks": 32,
                    },
                )
                payload = {"rle": response.get("rle"), "width": w, "height": h}
                _cache_path(cache_dir, item["id"], prompt).write_text(
                    json.dumps(payload), encoding="utf-8"
                )

        per_prompt: dict[str, list[dict]] = {p: [] for p in prompts}
        ensemble: list[dict] = []
        for item in picks:
            masks_by_prompt = {}
            for prompt in prompts:
                cached = json.loads(
                    _cache_path(cache_dir, item["id"], prompt).read_text()
                )
                w, h = cached["width"], cached["height"]
                masks_by_prompt[prompt] = _masks_from_response(cached, w, h)
            if spec["mode"] == "bbox":
                with Image.open(item["path"]) as im:
                    scale = min(1.0, MAX_SIDE / max(im.size))
                gt = [tuple(v * scale for v in box) for box in item["boxes"]]
                for prompt in prompts:
                    per_prompt[prompt].append(
                        score_bbox_image(gt, masks_by_prompt[prompt])
                    )
                ensemble.append(
                    score_bbox_image(
                        gt, [m for ms in masks_by_prompt.values() for m in ms]
                    )
                )
            else:
                with Image.open(item["gt_mask"]) as gt_img:
                    gt_mask = np.asarray(gt_img.convert("L").resize((w, h))) > 127
                for prompt in prompts:
                    per_prompt[prompt].append(
                        score_mask_image(gt_mask, masks_by_prompt[prompt])
                    )
                ensemble.append(
                    score_mask_image(
                        gt_mask, [m for ms in masks_by_prompt.values() for m in ms]
                    )
                )

        def _agg(scores: list[dict]) -> dict:
            if spec["mode"] == "bbox":
                gt = sum(s["gt"] for s in scores) or 1
                return {
                    "recall_iou30": round(sum(s["hits_iou30"] for s in scores) / gt, 3),
                    "recall_iou50": round(sum(s["hits_iou50"] for s in scores) / gt, 3),
                    "presence": round(
                        sum(s["present"] for s in scores) / max(1, len(scores)), 3
                    ),
                    "fp_per_image": round(
                        sum(s["false_positive_masks"] for s in scores)
                        / max(1, len(scores)),
                        2,
                    ),
                }
            return {
                "presence": round(
                    sum(s["present"] for s in scores) / max(1, len(scores)), 3
                ),
                "mean_coverage": round(
                    sum(s["coverage"] for s in scores) / max(1, len(scores)), 3
                ),
                "mean_iou": round(
                    sum(s["iou"] for s in scores) / max(1, len(scores)), 3
                ),
            }

        for prompt in prompts:
            rows.append(
                {"class": class_name, "prompt": prompt, **_agg(per_prompt[prompt])}
            )
        rows.append({"class": class_name, "prompt": "<ensemble>", **_agg(ensemble)})
        report.setdefault("classes", {})[class_name] = {
            "images": len(picks),
            "prompts": prompts,
        }

    report["rows"] = rows
    out.mkdir(parents=True, exist_ok=True)
    (out / f"report_{args.dataset}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
