"""Human-in-the-loop re-segmentation: box a region, SAM segments it, the
run's own geometry measures it, and the correction can feed straight back
into the run's scene + policy verdicts.

Open-vocabulary text prompts miss whole classes (light-curtain pillars
scored 0 on six phrases; a box prompt scores 0.91), so a reviewer who sees
a miss draws a box instead of fighting the vocabulary. Cached per
(label, box): re-runs are $0.
"""

import base64
import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import Observation2D
from .geometry import _build_geometry
from .path_safety import validate_safe_path_segment
from .providers.map_anything import input_mask_to_canonical
from .providers.sam3 import decode_coco_rle
from .viewer import _frames

MIN_POINTS = 60


class RefineError(RuntimeError):
    pass


def input_image(run: Path, frame_id: str = "frame_0001") -> Path:
    """Resolve the same ordered input/frame mapping used by geometry."""
    if not re.fullmatch(r"frame_[0-9]{4}", frame_id):
        raise RefineError("invalid frame_id")
    images = sorted((run / "input").glob("image_*"))
    index = int(frame_id[6:]) - 1
    if not 0 <= index < len(images):
        raise RefineError(f"input image for {frame_id} is missing")
    return images[index]


def _persist(run: Path, result: dict) -> bool:
    path = run / "refinements.json"
    records = json.loads(path.read_text()) if path.exists() else []
    record = {k: v for k, v in result.items() if k not in {"changed", "policies"}}
    for index, previous in enumerate(records):
        if previous.get("evidence_id") == result["evidence_id"]:
            if previous == record:
                return False
            records[index] = record
            break
    else:
        records.append(record)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=run, delete=False) as stream:
        json.dump(records, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)
    return True


def _default_subscriber(endpoint: str, *, arguments: dict) -> dict:
    from .providers.sam3 import sam_subscribe

    return sam_subscribe(endpoint, arguments=arguments)


def measure(
    run: Path, mask: np.ndarray, camera_height: float, scale: float | None,
    *, frame_id: str = "frame_0001",
) -> dict:
    frames = _frames(run)
    frame = next((f for f in frames if f.frame_id == frame_id), None)
    if frame is None:
        raise RefineError(f"geometry for {frame_id} is missing")
    observations_path = run / "observations.json"
    observations = [Observation2D.model_validate(item) for item in json.loads(observations_path.read_text())] if observations_path.exists() else []
    transform = _build_geometry(frames, observations, camera_height, scale_factor_override=scale).transform
    if transform is None:
        raise RefineError("run has no floor transform")
    points3d = np.load(frame.pts3d_path)
    valid = np.load(frame.valid_mask_path).astype(bool)
    mask = input_mask_to_canonical(mask, run, frame.frame_id, valid.shape)
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
    pose = np.asarray(frame.camera_to_world, dtype=float)
    depth = (points3d - pose[:3, 3]) @ pose[:3, 2]
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
        "camera_dist_m": round(float(np.linalg.norm(xy.mean(axis=0) - transform.apply(pose[None, :3, 3])[0, :2])), 2),
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
    frame_id: str = "frame_0001",
    instruction: str | None = None,
    located_rationale: str | None = None,
) -> dict:
    """Segment a selected source frame; retain evidence before measurement.

    apply=True persists the observation. Only a successful measurement can
    become a scene entity and enter the run's existing policy evaluation.
    """
    run = Path(runs_root) / run_id
    try:
        validate_safe_path_segment(run_id, "run_id")
    except ValueError as error:
        raise RefineError(str(error)) from error
    if not run.resolve().is_relative_to(Path(runs_root).resolve()):
        raise RefineError("run is outside runs_root")
    image_path = input_image(run, frame_id)
    label = label.strip()
    if not label or len(label) > 200 or any(ord(c) < 32 for c in label):
        raise RefineError("label must be a nonempty noun phrase of at most 200 characters")
    with Image.open(image_path) as image:
        width, height = image.size
    if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in box):
        raise RefineError("box must contain four integer original-image pixels")
    x1, y1, x2, y2 = (int(v) for v in box)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise RefineError(f"box {box} outside image {width}x{height}")

    out_dir = run / "refinements"
    out_dir.mkdir(exist_ok=True)
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    identity = json.dumps([frame_id, image_sha, label, [x1, y1, x2, y2]], ensure_ascii=False)
    slug = f"{frame_id}__{image_sha[:12]}__{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
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
        response = {**response, "frame_id": frame_id, "source_image_sha256": image_sha}
        cache.write_text(json.dumps(response) + "\n")
    if response.get("frame_id") != frame_id or response.get("source_image_sha256") != image_sha:
        raise RefineError("cached segmentation does not match the selected photograph")
    rles = response.get("rle") or []
    if isinstance(rles, str):
        rles = [rles]
    if not rles:
        raise RefineError("SAM returned no mask for that box")
    scores = response.get("scores") or [1.0] * len(rles)
    best = int(np.argmax(scores))
    mask = decode_coco_rle(rles[best], height=height, width=width).astype(bool)
    if mask.shape != (height, width):
        raise RefineError(f"SAM mask {mask.shape} does not match input image {(height, width)}")

    scene_path = run / "scene.json"
    scale = None
    if scene_path.exists():
        scale = json.loads(scene_path.read_text()).get("scale_factor")
    result = {
        "label": label,
        "frame_id": frame_id,
        "frame": frame_id,
        "source": "reviewer",
        "refine_slug": slug,
        "evidence_id": f"refine:{slug}",
        "source_image_sha256": image_sha,
        "mask_path": str(cache.relative_to(run)),
        "mask_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "box": [x1, y1, x2, y2],
        "sam_score": round(float(scores[best]), 3),
        "mask_pixels": int(mask.sum()),
        "geometry_status": "unmeasured",
        "geometry_reason": "measurement pending",
        "instruction": instruction,
        "located_rationale": located_rationale,
        "applied": bool(apply),
    }

    with Image.open(image_path) as image:
        overlay = np.asarray(image.convert("RGB")).copy()
    overlay[mask] = (
        0.45 * overlay[mask] + 0.55 * np.array([64, 156, 255])
    ).astype(np.uint8)
    overlay_path = out_dir / f"{slug}.png"
    Image.fromarray(overlay).save(overlay_path)
    result["overlay_path"] = str(overlay_path)

    # A 2D observation exists before geometry succeeds. Small masks are
    # still evidence and must survive every later measurement gate.
    before = (run / "refinements.json").read_bytes() if (run / "refinements.json").exists() else None
    if apply:
        _persist(run, result)
    try:
        result.update(measure(run, mask, camera_height, scale, frame_id=frame_id))
        result.update(geometry_status="measured", geometry_reason=None)
    except (RefineError, FileNotFoundError, ValueError) as error:
        result["geometry_reason"] = str(error)
    if apply:
        _persist(run, result)
    result["changed"] = bool(apply and before != (run / "refinements.json").read_bytes())
    if apply and result["geometry_status"] == "measured" and scene_path.exists():
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
                evidence_frame_ids=[result.get("frame_id", "frame_0001")],
            )
        )
        (run / "scene.json").write_text(scene.model_dump_json(indent=2) + "\n")
    policies_path = run / "policies.json"
    if policies_path.exists():
        envelope = json.loads(policies_path.read_text())
        specs = [PolicySpec.model_validate(s) for s in envelope.get("specs", [])]
        results = evaluate_policies(specs, scene, capture_frame_count=len(list((run / "input").glob("image_*"))))
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
