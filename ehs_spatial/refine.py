"""Human-in-the-loop re-segmentation: box a region, SAM segments it, the
run's own geometry measures it, and the correction can feed straight back
into the run's scene + policy verdicts.

Open-vocabulary text prompts miss whole classes (light-curtain pillars
scored 0 on six phrases; a box prompt scores 0.91), so a reviewer who sees
a miss draws a box instead of fighting the vocabulary. Cached per
(label, box): re-runs are $0.
"""

import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import _build_geometry
from .providers.sam3 import decode_coco_rle
from .viewer import _frames

MIN_POINTS = 60


class RefineError(RuntimeError):
    pass


def _default_subscriber(endpoint: str, *, arguments: dict) -> dict:
    from .providers.sam3 import sam_subscribe

    return sam_subscribe(endpoint, arguments=arguments)


def measure(
    run: Path, mask: np.ndarray, camera_height: float, scale: float | None
) -> dict:
    frames = _frames(run)
    frame = frames[0]
    transform = _build_geometry(
        frames, [], camera_height, scale_factor_override=scale
    ).transform
    if transform is None:
        raise RefineError("run has no floor transform")
    points3d = np.load(frame.pts3d_path)
    valid = np.load(frame.valid_mask_path).astype(bool)
    if mask.shape != valid.shape:
        mask = (
            np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (valid.shape[1], valid.shape[0])
                )
            )
            > 127
        )
    chosen = (
        mask
        & valid
        & np.isfinite(points3d).all(axis=2)
        # MapAnything fills invalid pixels with camera-frame zeros; they all
        # transform to one point and poison any robust statistic.
        & (np.abs(points3d).sum(axis=2) > 1e-6)
    )
    if chosen.sum() < MIN_POINTS:
        raise RefineError(f"only {int(chosen.sum())} 3D points under the mask")
    # depth bleed through/around transparent structures: a boxed object is
    # one physical thing, so keep the nearest depth cluster (p10 + 1 m).
    depth = points3d[..., 2]
    near = np.percentile(depth[chosen], 10)
    chosen &= depth <= near + 1.0
    cloud = transform.apply(points3d[chosen])
    if len(cloud) < MIN_POINTS:
        raise RefineError("mask collapsed after depth-band clipping")
    top = float(np.percentile(cloud[:, 2], 98))
    base = float(np.percentile(cloud[:, 2], 2))
    xy = cloud[:, :2]
    from shapely.geometry import MultiPoint

    hull = MultiPoint([tuple(p) for p in xy]).convex_hull
    footprint = (
        [[round(float(x), 3), round(float(y), 3)] for x, y in hull.exterior.coords[:-1]]
        if hull.geom_type == "Polygon"
        else [[round(float(x), 3), round(float(y), 3)] for x, y in xy[:3]]
    )
    return {
        "points": int(len(cloud)),
        "height_m": round(top, 2),
        "base_m": round(base, 2),
        "extent_m": f"{np.ptp(xy[:, 0]):.2f}x{np.ptp(xy[:, 1]):.2f}",
        "centroid_xy": [round(float(v), 2) for v in xy.mean(axis=0)],
        "camera_dist_m": round(float(np.linalg.norm(xy.mean(axis=0))), 2),
        "footprint_xy": footprint,
    }


def refine_region(
    run_id: str,
    label: str,
    box: tuple[int, int, int, int],
    *,
    runs_root: str | Path = "runs",
    camera_height: float = 1.5,
    apply: bool = False,
    subscriber=None,
) -> dict:
    """Box-prompt SAM + floor-frame measurement for one region of a run's
    input photo. With apply=True the correction becomes a scene entity and
    the run's policies re-evaluate (idempotent per label+box)."""
    run = Path(runs_root) / run_id
    image_path = next((run / "input").glob("image_*"))
    with Image.open(image_path) as image:
        width, height = image.size
    x1, y1, x2, y2 = (int(v) for v in box)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise RefineError(f"box {box} outside image {width}x{height}")

    out_dir = run / "refinements"
    out_dir.mkdir(exist_ok=True)
    slug = f"{label.replace(' ', '_')}_{x1}_{y1}_{x2}_{y2}"
    cache = out_dir / f"{slug}.json"
    if cache.exists():
        response = json.loads(cache.read_text())
    else:
        response = (subscriber or _default_subscriber)(
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
        raise RefineError("SAM returned no mask for that box")
    scores = response.get("scores") or [1.0] * len(rles)
    best = int(np.argmax(scores))
    mask = decode_coco_rle(rles[best], height=height, width=width).astype(bool)

    scene_path = run / "scene.json"
    scale = None
    if scene_path.exists():
        scale = json.loads(scene_path.read_text()).get("scale_factor")
    result = {
        "label": label,
        "box": [x1, y1, x2, y2],
        "sam_score": round(float(scores[best]), 3),
        "mask_pixels": int(mask.sum()),
        **measure(run, mask, camera_height, scale),
    }

    with Image.open(image_path) as image:
        overlay = np.asarray(image.convert("RGB")).copy()
    overlay[mask] = (
        0.45 * overlay[mask] + 0.55 * np.array([64, 156, 255])
    ).astype(np.uint8)
    overlay_path = out_dir / f"{slug}.png"
    Image.fromarray(overlay).save(overlay_path)
    result["overlay_path"] = str(overlay_path)

    log_path = run / "refinements.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    if not any(item.get("box") == result["box"] and item.get("label") == label
               for item in log):
        log.append({k: v for k, v in result.items() if k != "overlay_path"})
        log_path.write_text(json.dumps(log, indent=2) + "\n")

    if apply:
        result["policies"] = _apply_to_scene(run, label, slug, result)
    return result


def _apply_to_scene(run: Path, label: str, slug: str, result: dict) -> list[dict]:
    from .contracts import Entity3D, PolicySpec, SceneMap
    from .policy import evaluate_policies

    scene = SceneMap.model_validate(json.loads((run / "scene.json").read_text()))
    observation_id = f"refine:{slug}"
    statuses: list[dict] = []
    if not any(observation_id in e.observation_ids for e in scene.entities):
        refine_index = (
            sum(1 for e in scene.entities if e.entity_id.startswith("refine-")) + 1
        )
        scene.entities.append(
            Entity3D(
                entity_id=f"refine-{refine_index:02d}",
                label=label,
                observation_ids=[observation_id],
                centroid_xyz=(
                    float(result["centroid_xy"][0]),
                    float(result["centroid_xy"][1]),
                    max(0.0, (result["base_m"] + result["height_m"]) / 2),
                ),
                footprint_xy=[
                    (float(x), float(y)) for x, y in result["footprint_xy"]
                ],
                height_m=max(0.01, float(result["height_m"])),
                evidence_frame_ids=["frame_0001"],
            )
        )
        (run / "scene.json").write_text(scene.model_dump_json(indent=2) + "\n")
    policies_path = run / "policies.json"
    if policies_path.exists():
        envelope = json.loads(policies_path.read_text())
        specs = [PolicySpec.model_validate(s) for s in envelope.get("specs", [])]
        results = evaluate_policies(specs, scene, capture_frame_count=1)
        envelope["results"] = [r.model_dump(mode="json") for r in results]
        policies_path.write_text(json.dumps(envelope, indent=2) + "\n")
        statuses = [
            {
                "policy_id": r.policy_id,
                "status": getattr(r.status, "value", str(r.status)),
            }
            for r in results
        ]
    return statuses
