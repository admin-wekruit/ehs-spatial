"""Close the perspective loop: project plan rectangles BACK into the photo
and score them against the evidence that produced them.

The dense per-pixel geometry is its own projector — no intrinsics needed:
every valid pixel already knows its floor-frame (x, y), so a floor point
maps to the image by nearest-neighbour over the floor pixels. A plan rect
whose base line lands on its structure's ground contact in the photo is
verified; one that drifts is caught numerically instead of by the owner's
eyes. Scores land in inventory/reprojection.json, the visual overlay in
inventory/reprojection.png.
"""

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .providers.sam3 import decode_coco_rle

FLOOR_BAND_M = 0.10  # |z| below this counts as floor
MATCH_RADIUS_M = 0.20  # NN acceptance radius in floor XY


def _floor_lookup(run: Path):
    """KD-tree over the floor pixels' floor-frame XY -> image pixel."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scipy.spatial import cKDTree

    from ehs_spatial.geometry import _build_geometry
    from ehs_spatial.viewer import _frames

    frames = _frames(run)
    frame = frames[0]
    scene_path = run / "scene.json"
    scale = None
    if scene_path.exists():
        scale = json.loads(scene_path.read_text()).get("scale_factor")
    transform = _build_geometry(
        frames, [], 1.5, scale_factor_override=scale
    ).transform
    points3d = np.load(frame.pts3d_path)
    valid = np.load(frame.valid_mask_path).astype(bool)
    finite = (
        valid
        & np.isfinite(points3d).all(axis=2)
        & (np.abs(points3d).sum(axis=2) > 1e-6)
    )
    floor_pts = transform.apply(points3d[finite])
    vs, us = np.nonzero(finite)
    on_floor = np.abs(floor_pts[:, 2]) < FLOOR_BAND_M
    tree = cKDTree(floor_pts[on_floor][:, :2])
    zmap = np.full(valid.shape, np.nan, np.float32)
    zmap[vs, us] = floor_pts[:, 2]
    return tree, us[on_floor], vs[on_floor], valid.shape, zmap


def _project_polyline(tree, us, vs, xy_samples):
    """Floor-frame XY samples -> image pixels (geometry-grid coords)."""
    dists, idx = tree.query(np.asarray(xy_samples, float))
    keep = dists < MATCH_RADIUS_M
    return [
        (int(us[i]), int(vs[i]))
        for i, ok in zip(idx, keep)
        if ok
    ]


def _rect_baseline(rect, samples: int = 40):
    r = np.asarray(rect, float)
    edge1, edge2 = r[1] - r[0], r[2] - r[1]
    if np.linalg.norm(edge1) < np.linalg.norm(edge2):
        a = (r[0] + r[1]) / 2
        b = (r[2] + r[3]) / 2
    else:
        a = (r[0] + r[3]) / 2
        b = (r[1] + r[2]) / 2
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return a[None, :] * (1 - t) + b[None, :] * t


def _mask_bottom_profile(
    mask: np.ndarray, zmap: np.ndarray | None = None
) -> dict[int, int]:
    """Bottom edge per column — restricted to VERIFIED ground contact
    when a height map is given: a bottom pixel sitting ~1 m off the floor
    is an occlusion boundary (second-row structure), not a contact line,
    and must neither be scored against nor corrected toward."""
    profile: dict[int, int] = {}
    vs, us = np.nonzero(mask)
    for u, v in zip(us, vs):
        if u not in profile or v > profile[u]:
            profile[u] = v
    if zmap is not None:
        profile = {
            u: v
            for u, v in profile.items()
            if np.isfinite(zmap[v, u]) and zmap[v, u] < 0.35
        }
    return profile


def verify_reprojection(run_id: str, *, runs_root: str | Path = "runs") -> dict:
    """Score every guard-line/contact-edge fence rect: reproject its base
    line into the photo, measure mean |Δv| to the mask's bottom edge
    (fraction of image height). Renders the overlay alongside."""
    run = Path(runs_root) / run_id
    inventory = json.loads(
        (run / "inventory" / "inventory.json").read_text()
    )
    tree, us, vs, (height, width), zmap = _floor_lookup(run)
    image = Image.open(next((run / "input").glob("image_*"))).convert("RGB")
    scale_x = image.size[0] / width
    scale_y = image.size[1] / height
    draw = ImageDraw.Draw(image)
    scores = []
    for obj in inventory["objects"]:
        if obj.get("footprint_method") not in ("guard-line", "contact-edge"):
            continue
        if "fence" not in obj["label"] or not obj.get("rect_snapped"):
            continue
        pixels = _project_polyline(
            tree, us, vs, _rect_baseline(obj["rect_snapped"])
        )
        if len(pixels) < 8:
            scores.append(
                {
                    "label": obj["label"],
                    "instance": obj.get("instance"),
                    "status": "unprojectable",
                }
            )
            continue
        if obj.get("refine_slug"):
            cache = run / "refinements" / f"{obj['refine_slug']}.json"
            response = json.loads(cache.read_text())
            rles = response.get("rle") or []
            if isinstance(rles, str):
                rles = [rles]
            scores_r = response.get("scores") or [1.0] * len(rles)
            raw = decode_coco_rle(
                rles[int(np.argmax(scores_r))], height=3024, width=4032
            ).astype(np.uint8)
            mask = (
                np.asarray(
                    Image.fromarray(raw * 255).resize((width, height))
                )
                > 127
            )
        else:
            slug = re.sub(r"[^a-z0-9]+", "_", obj["label"]).strip("_")
            cache = run / "inventory" / "sam" / f"frame_0001__{slug}.json"
            rles = json.loads(cache.read_text()).get("rle") or []
            if isinstance(rles, str):
                rles = [rles]
            mask = np.zeros((height, width), bool)
            for member in obj.get("merged_instances") or [obj["instance"]]:
                if member < len(rles):
                    mask |= decode_coco_rle(
                        rles[member], height=height, width=width
                    ).astype(bool)
        profile = _mask_bottom_profile(mask, zmap)
        deltas = [
            abs(v - profile[u]) for u, v in pixels if u in profile
        ]
        if len(profile) < 8 or len(deltas) < 8:
            scores.append(
                {
                    "label": obj["label"],
                    "instance": obj.get("instance"),
                    "refine_slug": obj.get("refine_slug"),
                    "method": obj["footprint_method"],
                    "status": "base-occluded",
                }
            )
            continue
        mean_dv = round(float(np.median(np.abs(deltas))) / height, 4)
        scores.append(
            {
                "label": obj["label"],
                "instance": obj.get("instance"),
                "method": obj["footprint_method"],
                "matched": len(deltas),
                "mean_dv_frac": mean_dv,
            }
        )
        for u, v in pixels:
            x, y = u * scale_x, v * scale_y
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 60, 60))
    out = {"run_id": run_id, "scores": scores}
    (run / "inventory" / "reprojection.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    )
    image.save(run / "inventory" / "reprojection.png")
    return out


__all__ = ["verify_reprojection"]
