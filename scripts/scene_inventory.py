"""Whole-scene inventory: enumerate every object, then map the room to CAD.

Three stages over a run's cached geometry:
  1. a VLM lists every distinct object it can see, as short noun phrases
  2. SAM segments each phrase; masks are lifted into the run's floor frame
  3. wall planes are fitted from the non-floor cloud, and the result is
     written as a dimensioned floor plan (PNG) and a real DXF that opens
     in any CAD tool

Only stages 1-2 spend money, and both are disk-cached, so re-runs are free.
Nothing here feeds a verdict: this is the scene-understanding surface, not
the compliance path.

Usage:
  uv run --env-file .env python scripts/scene_inventory.py --run demo-real-factory
  uv run --env-file .env python scripts/scene_inventory.py --run demo-real-factory --live
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint, Point, Polygon

from ehs_spatial.contracts import GeometryFrame, Observation2D
from ehs_spatial.geometry import _build_geometry, _spatial_state
from ehs_spatial.providers.sam3 import (
    SAM3_ENDPOINT,
    decode_coco_rle,
    encode_coco_rle,
)

MAX_PHRASES = 26
MIN_MASK_PIXELS = 50
MIN_CLOUD_POINTS = 50
WALL_MIN_INLIERS = 400
WALL_MAX_COUNT = 6
# Beyond this range single-view depth collapses (the boundary map measured
# negative heights at ~20 m), so far entries are inventoried but kept off
# the dimensioned plan rather than drawn as if they were surveyed.
PLAN_MAX_RANGE_M = 12.0
# Structure is drawn as structure, not as furniture.
STRUCTURE_LABELS = {"floor", "ceiling", "wall", "window", "ground", "roof"}

# Painted / laid-flat classes whose correct height IS ~0; the non-positive-
# height gate must not read that as depth collapse.
FLAT_ZONE_KEYWORDS = ("marking", "line", "zone", "mat", "stripe", "tape")


def _is_flat_zone(label: str) -> bool:
    return any(keyword in label for keyword in FLAT_ZONE_KEYWORDS)


CONTACT_FAMILY = ("fence", "guard", "barrier", "rail", "curtain", "partition", "panel")


def _merge_vertical_stacks(
    masks: list[np.ndarray], image_height: int
) -> list[tuple[int, list[int]]]:
    """SAM splits one tall panel into stacked horizontal bands. Same-phrase
    masks whose x-ranges strongly overlap and that touch vertically are one
    physical structure; side-by-side sections keep disjoint x-ranges and
    stay separate. Returns (primary_index, member_indices) per group."""
    boxes: list[tuple[int, int, int, int] | None] = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        boxes.append(
            None
            if len(xs) == 0
            else (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
        )
    parent = list(range(len(masks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    gap = 0.03 * image_height
    for i in range(len(masks)):
        if boxes[i] is None:
            continue
        for j in range(i + 1, len(masks)):
            if boxes[j] is None:
                continue
            ax0, ax1, ay0, ay1 = boxes[i]
            bx0, bx1, by0, by1 = boxes[j]
            overlap = min(ax1, bx1) - max(ax0, bx0)
            narrower = min(ax1 - ax0, bx1 - bx0)
            wider = max(ax1 - ax0, bx1 - bx0)
            # both gates: a narrow mask fully inside a wide run must not
            # bridge two structures into one union-find chain
            if narrower <= 0 or overlap < 0.6 * narrower or overlap < 0.3 * wider:
                continue
            if max(ay0, by0) - min(ay1, by1) > gap:
                continue
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(masks)):
        if boxes[i] is None:
            continue
        groups.setdefault(find(i), []).append(i)
    return sorted(
        ((min(members), sorted(members)) for members in groups.values()),
        key=lambda item: item[0],
    )


def _contact_edge_rect(
    mask: np.ndarray,
    points3d: np.ndarray,
    frame,
    transform,
    theta: float | None,
) -> list | None:
    """Perspective-correct footprint for floor-standing thin structures
    (research approach 2): back-project the mask's ground-contact edge
    through the camera onto the floor plane. Per-pixel depth never enters
    the footprint — only the robust floor fit does — so the along-ray
    smear cannot tilt or stretch it. Columns whose bottom pixel is not
    actually near the ground (occluded base) are rejected by comparing
    measured depth with the ray/floor intersection."""
    intrinsics = np.asarray(frame.intrinsics, dtype=float)
    k_inv = np.linalg.inv(intrinsics)
    camera_to_world = np.asarray(frame.camera_to_world, dtype=float)
    rotation_c = camera_to_world[:3, :3]
    origin_c = camera_to_world[:3, 3]
    r_z = transform.rotation[2]
    numerator = (transform.origin - origin_c) @ r_z

    columns = np.flatnonzero(mask.any(axis=0))
    if len(columns) < 8:
        return None
    ground_points = []
    for u in columns:
        v = int(mask[:, u].nonzero()[0].max())
        ray_cam = k_inv @ np.array([u + 0.5, v + 0.5, 1.0])
        ray_world = rotation_c @ ray_cam
        denominator = ray_world @ r_z
        if abs(denominator) < 1e-9:
            continue
        t = numerator / denominator
        if t <= 0:
            continue
        measured = points3d[v, u]
        if not np.isfinite(measured).all() or np.abs(measured).sum() < 1e-6:
            continue
        z_measured = (rotation_c.T @ (measured - origin_c))[2]
        z_ground = t * ray_cam[2]
        if z_measured <= 0 or abs(z_measured - z_ground) > 0.15 * max(z_ground, 1.0):
            continue
        world = origin_c + t * ray_world
        ground_points.append(transform.apply(world[None, :])[0][:2])
    if len(ground_points) < max(8, 0.3 * len(columns)):
        return None

    grid = np.asarray(ground_points)
    centre = grid.mean(axis=0)
    _, _, vt = np.linalg.svd(grid - centre, full_matrices=False)
    direction = vt[0]
    residual = np.abs((grid - centre) @ np.array([-direction[1], direction[0]]))
    keep = residual < max(0.15, 3 * np.median(residual) + 1e-6)
    if keep.sum() >= 8:
        grid = grid[keep]
        centre = grid.mean(axis=0)
        _, _, vt = np.linalg.svd(grid - centre, full_matrices=False)
        direction = vt[0]
    phi = float(np.arctan2(direction[1], direction[0]))
    if theta is not None:
        for candidate in (theta, theta + np.pi / 2):
            if abs(((phi - candidate + np.pi / 2) % np.pi) - np.pi / 2) < np.radians(10):
                phi = float(candidate)
                break
    axis = np.array([np.cos(phi), np.sin(phi)])
    normal = np.array([-np.sin(phi), np.cos(phi)])
    along = (grid - centre) @ axis
    low, high = np.percentile(along, 2), np.percentile(along, 98)
    # fence-family structures are boards/rails: anything past ~25 cm of
    # measured "thickness" is depth smear, not the object
    thickness = float(
        np.clip(2 * np.percentile(np.abs((grid - centre) @ normal), 80), 0.05, 0.25)
    )
    corners = [
        centre + low * axis + thickness / 2 * normal,
        centre + high * axis + thickness / 2 * normal,
        centre + high * axis - thickness / 2 * normal,
        centre + low * axis - thickness / 2 * normal,
    ]
    return [[round(float(x), 3), round(float(y), 3)] for x, y in corners]


def _manhattan_theta(walls: list[dict]) -> float | None:
    """Scene principal axis from fitted wall segments: length-weighted
    circular mean over the 90°-periodic angle (closed-form, no search)."""
    if not walls:
        return None
    s = c = 0.0
    for wall in walls:
        dx = wall["end"][0] - wall["start"][0]
        dy = wall["end"][1] - wall["start"][1]
        length = float(np.hypot(dx, dy))
        if length < 0.5:
            continue
        angle = np.arctan2(dy, dx)
        s += length * np.sin(4 * angle)
        c += length * np.cos(4 * angle)
    if s == 0 and c == 0:
        return None
    return 0.25 * float(np.arctan2(s, c))


def _snap_rect(footprint: list, theta: float | None, allow_free: bool = True) -> list | None:
    """Axis-snapped robust rectangle for a footprint (research approach 1):
    express the points in the Manhattan frame, take the p2–p98 box, rotate
    back. Escape hatch: a genuinely oblique object (free rectangle >15° off
    both axes AND markedly tighter) keeps its free orientation."""
    pts = np.asarray(footprint, dtype=float)
    if theta is None or len(pts) < 3:
        return None
    rot = np.array(
        [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]]
    )
    q = pts @ rot.T
    x0, x1 = np.percentile(q[:, 0], 2), np.percentile(q[:, 0], 98)
    y0, y1 = np.percentile(q[:, 1], 2), np.percentile(q[:, 1], 98)
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    snapped = corners @ np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    ).T
    try:
        free = Polygon(footprint).minimum_rotated_rectangle
        coords = np.asarray(free.exterior.coords)[:-1]
        edge = coords[1] - coords[0]
        free_angle = float(np.arctan2(edge[1], edge[0]))
        offset = abs(
            ((free_angle - theta + np.pi / 4) % (np.pi / 2)) - np.pi / 4
        )
        snapped_area = (x1 - x0) * (y1 - y0)
        if (
            allow_free
            and offset > np.radians(15)
            and np.isfinite(free.area)
            and free.area < 0.7 * snapped_area
        ):
            return [[round(float(x), 3), round(float(y), 3)] for x, y in coords]
    except Exception:
        pass
    return [[round(float(x), 3), round(float(y), 3)] for x, y in snapped]


def _frames(run: Path) -> list[GeometryFrame]:
    out = []
    for frame_dir in sorted((run / "geometry" / "frames").iterdir()):
        if not frame_dir.is_dir():
            continue
        out.append(
            GeometryFrame(
                frame_id=frame_dir.name,
                canonical_image_path=str(frame_dir / "canonical.png"),
                pts3d_path=str(frame_dir / "pts3d.npy"),
                conf_path=str(frame_dir / "conf.npy"),
                valid_mask_path=str(frame_dir / "valid_mask.npy"),
                camera_to_world=np.load(frame_dir / "camera_to_world.npy").tolist(),
                intrinsics=np.load(frame_dir / "intrinsics.npy").tolist(),
            )
        )
    return out


def _enumerate_objects(run: Path, frame: GeometryFrame, *, live: bool):
    """Stage 1 — the VLM only supplies vocabulary, never geometry."""
    cache = run / "inventory" / "phrases.json"
    if cache.exists():
        return json.loads(cache.read_text())
    if not live:
        return None
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            types.Part.from_bytes(
                data=Path(frame.canonical_image_path).read_bytes(),
                mime_type="image/png",
            ),
            "List every distinct physical object and structure visible in "
            "this industrial photo. Respond with ONLY a JSON array of short "
            "singular noun phrases suitable as segmentation prompts (e.g. "
            '["machine", "safety fence", "worker", "floor marking"]). '
            f"At most {MAX_PHRASES} entries, most prominent first. No "
            "duplicates, no adjectives about colour or count.",
        ],
    )
    match = re.search(r"\[.*\]", response.text or "", re.S)
    phrases = [str(p).strip().lower() for p in json.loads(match.group(0))]
    phrases = list(dict.fromkeys(p for p in phrases if p))[:MAX_PHRASES]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(phrases, indent=2) + "\n")
    return phrases


def _prompt_candidates(phrase: str) -> tuple[str, ...]:
    """Ensemble prompts for a phrase: the pipeline's measured synonym lists
    when the phrase is (or maps into) the production vocabulary, else the
    bare phrase. SAM 3 misses compound noun phrases the short synonym hits
    ('safety fence' -> 0 masks, 'fence' -> hit, measured on real imagery)."""
    from ehs_spatial.providers.sam3 import LABEL_PROMPTS

    if phrase in LABEL_PROMPTS:
        return LABEL_PROMPTS[phrase]
    for label, prompts in LABEL_PROMPTS.items():
        if phrase in prompts:
            return prompts
    return (phrase,)


def _sam_call(frame: GeometryFrame, prompt: str) -> dict:
    from ehs_spatial.providers.sam3 import sam_subscribe

    return sam_subscribe(
        SAM3_ENDPOINT,
        arguments={
            "image_url": "data:image/png;base64,"
            + base64.b64encode(
                Path(frame.canonical_image_path).read_bytes()
            ).decode("ascii"),
            "prompt": prompt,
            "return_multiple_masks": True,
            "include_scores": True,
            "include_boxes": True,
            "max_masks": 12,
        },
    )


def _union_responses(
    responses: list[tuple[str, dict]], frame: GeometryFrame
) -> dict:
    """Cross-phrase instance union with IoU>0.5 dedupe (higher score wins).
    First-hit ensembles miss instances one phrase sees and another doesn't;
    the union keeps every distinct instance any phrase found."""
    valid = np.load(frame.valid_mask_path)
    height, width = valid.shape
    pool: list[tuple[float, str, str, np.ndarray]] = []
    for prompt, response in responses:
        rles = response.get("rle") or []
        if isinstance(rles, str):
            rles = [rles]
        scores = response.get("scores") or [1.0] * len(rles)
        for i, rle in enumerate(rles):
            try:
                mask = decode_coco_rle(rle, height=height, width=width).astype(bool)
            except Exception:
                continue
            if not mask.any():
                continue
            score = float(scores[i]) if i < len(scores) else 1.0
            pool.append((score, prompt, rle, mask))
    pool.sort(key=lambda item: -item[0])
    kept: list[tuple[float, str, str, np.ndarray]] = []
    for score, prompt, rle, mask in pool:
        area = mask.sum()
        duplicate = False
        for _, _, _, seen in kept:
            inter = int((mask & seen).sum())
            if inter / (area + int(seen.sum()) - inter) > 0.5:
                duplicate = True
                break
        if not duplicate:
            kept.append((score, prompt, rle, mask))
    return {
        "rle": [item[2] for item in kept],
        "scores": [round(item[0], 3) for item in kept],
        "prompts": [item[1] for item in kept],
    }


def _segment(run: Path, frame: GeometryFrame, phrase: str, *, live: bool):
    slug = re.sub(r"[^a-z0-9]+", "_", phrase).strip("_")
    cache = run / "inventory" / "sam" / f"{frame.frame_id}__{slug}.json"
    cached = json.loads(cache.read_text()) if cache.exists() else None
    prompts = _prompt_candidates(phrase)
    union_wanted = any(k in phrase for k in CONTACT_FAMILY) and len(prompts) > 1
    if union_wanted:
        # fence-family phrases take the multi-prompt union: first-hit misses
        # whole panels ("clear panel" sees what "fence" doesn't)
        if cached is not None and (cached.get("union") or not live):
            return cached if cached.get("rle") or not live else cached
        if not live:
            return None
        responses: list[tuple[str, dict]] = []
        if cached and cached.get("rle"):
            responses.append(("legacy", cached))
        for prompt in prompts:
            pslug = re.sub(r"[^a-z0-9]+", "_", prompt).strip("_")
            pcache = cache.with_name(f"{frame.frame_id}__{slug}__p_{pslug}.json")
            if pcache.exists():
                response = json.loads(pcache.read_text())
            else:
                response = _sam_call(frame, prompt)
                pcache.parent.mkdir(parents=True, exist_ok=True)
                pcache.write_text(json.dumps(response) + "\n")
            if response.get("rle"):
                responses.append((prompt, response))
        union = _union_responses(responses, frame)
        union["union"] = True
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(union) + "\n")
        return union
    if cached is not None and (cached.get("rle") or not live):
        return cached
    if not live:
        return None
    response = cached
    for prompt in prompts:
        response = _sam_call(frame, prompt)
        if response.get("rle"):
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(response) + "\n")
    return response


def _resize_map(array: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image as PILImage

    return np.stack(
        [
            np.asarray(
                PILImage.fromarray(array[..., c]).resize(
                    (width, height), PILImage.BILINEAR
                )
            )
            for c in range(array.shape[-1])
        ],
        axis=-1,
    )


def _moge3_maps(
    run: Path, frame: GeometryFrame
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-pixel camera-frame normals AND points from the MoGe-3 Modal
    deployment, resized to the geometry grid. Both come from one inference
    (same frame, same metric scale), cached per run. Returns None (and says
    why) when Modal is unreachable so the pipeline degrades gracefully."""
    valid = np.load(frame.valid_mask_path)
    height, width = valid.shape
    normals_cache = run / "geometry" / "moge3_normals.npz"
    points_cache = run / "geometry" / "moge3_points.npz"
    if not (normals_cache.exists() and points_cache.exists()):
        try:
            import modal

            handle = modal.Cls.from_name("moge3-inference", "MoGe3")
            result = handle().infer.remote(
                Path(frame.canonical_image_path).read_bytes()
            )
            normals_cache.write_bytes(result["normals_npz"])
            points_cache.write_bytes(result["points_npz"])
        except Exception as error:
            print(f"  [normals] MoGe-3 unavailable ({error}); no normal split")
            return None
    normal = _resize_map(np.load(normals_cache)["normal"], width, height)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    points = _resize_map(
        np.load(points_cache)["points"].astype(np.float32), width, height
    )
    return normal / np.maximum(norm, 1e-6), points


def _normal_split(
    mask: np.ndarray, normals: np.ndarray, points: np.ndarray | None = None
) -> np.ndarray | None:
    """Keep the mask's dominant vertical-plane normal cluster. Pixels seen
    THROUGH a clear panel carry the background's scattered normals; the
    panel frame agrees on one near-horizontal normal direction. Returns the
    filtered mask, or None when no dominant vertical plane exists."""
    sample = normals[mask]
    sample = sample[np.isfinite(sample).all(axis=1)]
    if len(sample) < 200:
        return None
    horizontal = np.abs(sample[:, 1]) < 0.5
    if horizontal.sum() < 0.2 * len(sample):
        return None
    azimuth = np.arctan2(sample[horizontal, 2], sample[horizontal, 0])
    bins = np.histogram(azimuth, bins=36, range=(-np.pi, np.pi))[0]
    dominant = (np.argmax(bins) + 0.5) / 36 * 2 * np.pi - np.pi
    full_azimuth = np.arctan2(normals[..., 2], normals[..., 0])
    angular = np.abs(((full_azimuth - dominant + np.pi) % (2 * np.pi)) - np.pi)
    kept = mask & (np.abs(normals[..., 1]) < 0.6) & (angular < np.radians(30))
    if kept.sum() < max(200, 0.15 * mask.sum()):
        return None
    if points is not None:
        # plane-consistency gate. Through clean glass BOTH depth models
        # reconstruct the background, so most mask pixels lie on it — a
        # median-anchored plane locks onto the background, not the panel.
        # Anchor on the NEAREST depth cluster instead: that is the opaque
        # metal frame. The panel's plane passes through the frame; a
        # structure receding in depth stays in its own plane, so long
        # fence runs are not truncated, while parallel objects behind the
        # glass sit offset along the normal and drop out.
        offset = points[..., 0] * np.cos(dominant) + points[..., 2] * np.sin(
            dominant
        )
        depth = points[..., 2]
        finite = np.isfinite(offset) & np.isfinite(depth)
        if (kept & finite).sum() >= 150:
            near = float(np.percentile(depth[kept & finite], 10))
            seed = kept & finite & (depth <= near + max(0.25, 0.12 * near))
            if seed.sum() >= 150:
                centre = float(np.median(offset[seed]))
                plane = kept & finite & (np.abs(offset - centre) < 0.35)
                if plane.sum() >= 150:
                    kept = plane
    return kept


def _rect_sides(rect) -> tuple[float, float]:
    r = np.asarray(rect, float)
    a = float(np.linalg.norm(r[1] - r[0]))
    b = float(np.linalg.norm(r[2] - r[1]))
    return max(a, b), min(a, b)


def _align_guard_lines(
    entries: list[dict],
    theta: float | None,
    walls_evidence: list[dict],
    run: Path,
    frame: GeometryFrame,
    transform,
) -> float | None:
    """Collinear guard-line prior: fence sections that sit side by side in
    the IMAGE form one straight physical guard line, so their plan rects
    must share one line. The line's DIRECTION comes from the contact-edge
    projection of the chain's UNION mask (ray-cast to the floor — immune
    to the depth pollution that corrupts per-section centroids and once
    flattened a receding gate into a constant-depth line); members are
    then seated along it. Falls back to a spacing-scored centroid line
    when the union has no honest contact edge."""
    points3d = np.load(frame.pts3d_path)
    valid = np.load(frame.valid_mask_path).astype(bool)
    height, width = valid.shape
    image_width = width
    finite_all = (
        valid
        & np.isfinite(points3d).all(axis=2)
        & (np.abs(points3d).sum(axis=2) > 1e-6)
    )
    floor_xyz = np.full((height, width, 3), np.nan, np.float32)
    floor_xyz[finite_all] = transform.apply(points3d[finite_all])

    def _member_mask(e: dict) -> np.ndarray | None:
        if e.get("refine_slug"):
            return _refine_mask(run, e["refine_slug"], height, width)
        slug = re.sub(r"[^a-z0-9]+", "_", e["label"]).strip("_")
        cache = run / "inventory" / "sam" / f"{frame.frame_id}__{slug}.json"
        if not cache.exists():
            return None
        rles = json.loads(cache.read_text()).get("rle") or []
        if isinstance(rles, str):
            rles = [rles]
        mask = np.zeros((height, width), bool)
        for member in e.get("merged_instances") or [e["instance"]]:
            if member >= len(rles):
                return None
            mask |= decode_coco_rle(
                rles[member], height=height, width=width
            ).astype(bool)
        return mask
    guards = [
        e
        for e in entries
        if any(k in e["label"] for k in CONTACT_FAMILY)
        and e.get("rect_snapped")
        and e.get("image_bbox")
    ]
    guards.sort(key=lambda e: (e["image_bbox"][0] + e["image_bbox"][2]) / 2)
    chains: list[list[dict]] = []
    current: list[dict] = []
    for e in guards:
        if current:
            prev = current[-1]
            px0, py0, px1, py1 = prev["image_bbox"]
            x0, y0, x1, y1 = e["image_bbox"]
            gap = x0 - px1
            y_overlap = min(py1, y1) - max(py0, y0)
            # section compatibility by IMAGE evidence: measured 3D heights
            # are noise-polluted (a 0.3 m reading on a 1.5 m section broke
            # a gate chain); side-by-side sections of one line have similar
            # pixel heights, and that survives every depth failure mode
            ih_prev = max(1, py1 - py0)
            ih_here = max(1, y1 - y0)
            if (
                gap < 0.15 * image_width
                and y_overlap > 0.3 * min(ih_prev, ih_here)
                and max(ih_prev, ih_here) / min(ih_prev, ih_here) < 1.8
            ):
                current.append(e)
                continue
            chains.append(current)
        current = [e]
    if current:
        chains.append(current)

    # pass 1: UNSNAPPED contact-edge evidence per chain — each guard line's
    # free direction, weighted by its physical length, votes on the scene
    # axis together with the walls (joint Manhattan estimate) instead of
    # merely inheriting the walls' answer.
    chain_rects: list = []
    for chain in chains:
        rect = None
        if len(chain) >= 2:
            union = None
            for e in chain:
                member_mask = _member_mask(e)
                if member_mask is None:
                    union = None
                    break
                union = (
                    member_mask if union is None else (union | member_mask)
                )
            if union is not None:
                rect = _contact_edge_rect(
                    union, points3d, frame, transform, None
                )
        chain_rects.append(rect)
    s = c = 0.0
    if theta is not None:
        # walls' aggregated vote re-enters at its established strength
        for wall in walls_evidence or []:
            dx = wall["end"][0] - wall["start"][0]
            dy = wall["end"][1] - wall["start"][1]
            length = float(np.hypot(dx, dy))
            if length < 0.5:
                continue
            angle = np.arctan2(dy, dx)
            s += length * np.sin(4 * angle)
            c += length * np.cos(4 * angle)
    for rect in chain_rects:
        if rect is None:
            continue
        r = np.asarray(rect, float)
        edge = r[1] - r[0]
        if np.linalg.norm(edge) < np.linalg.norm(r[2] - r[1]):
            edge = r[2] - r[1]
        length = float(np.linalg.norm(edge))
        angle = float(np.arctan2(edge[1], edge[0]))
        s += length * np.sin(4 * angle)
        c += length * np.cos(4 * angle)
    if s != 0 or c != 0:
        theta = 0.25 * float(np.arctan2(s, c))

    for chain_index, (chain, rect) in enumerate(zip(chains, chain_rects)):
        if len(chain) < 2:
            continue
        for e in chain:
            e["guard_chain"] = chain_index
        cents = np.array([e["centroid_xy"] for e in chain], float)
        lengths = [_rect_sides(e["rect_snapped"])[0] for e in chain]
        centre = direction = None
        rect_long = thickness = None
        union_line = None
        if len(chain) >= 2:
            union = None
            for e in chain:
                member_mask = _member_mask(e)
                if member_mask is None:
                    union = None
                    break
                union = member_mask if union is None else (union | member_mask)
            if union is not None:
                union_line = _verified_contact_line(union, floor_xyz)
        if union_line is not None:
            centre, direction, rect_long = union_line
            thickness = 0.15
            if rect is not None:
                r0 = np.asarray(rect, float)
                side_a = float(np.linalg.norm(r0[1] - r0[0]))
                side_b = float(np.linalg.norm(r0[2] - r0[1]))
                thickness = max(0.05, min(side_a, side_b, 0.25))
            axis_snapped = False
            if theta is not None:
                angle = float(np.arctan2(direction[1], direction[0]))
                for axis in (theta, theta + np.pi / 2):
                    delta = ((angle - axis + np.pi / 2) % np.pi) - np.pi / 2
                    if abs(delta) < np.radians(20):
                        snapped = angle - delta
                        direction = np.array([np.cos(snapped), np.sin(snapped)])
                        axis_snapped = True
                        break
        elif rect is not None:
            r = np.asarray(rect, float)
            edge1, edge2 = r[1] - r[0], r[2] - r[1]
            if np.linalg.norm(edge1) < np.linalg.norm(edge2):
                edge1, edge2 = edge2, edge1
            rect_long = float(np.linalg.norm(edge1))
            direction = edge1 / max(float(np.linalg.norm(edge1)), 1e-9)
            thickness = float(np.linalg.norm(edge2))
            centre = r.mean(axis=0)
            # snap the free direction onto the JOINT Manhattan axis it
            # helped estimate — orthogonality between structures becomes
            # exact, not approximate
            axis_snapped = False
            if theta is not None:
                angle = float(np.arctan2(direction[1], direction[0]))
                for axis in (theta, theta + np.pi / 2):
                    delta = ((angle - axis + np.pi / 2) % np.pi) - np.pi / 2
                    if abs(delta) < np.radians(20):
                        snapped = angle - delta
                        direction = np.array(
                            [np.cos(snapped), np.sin(snapped)]
                        )
                        axis_snapped = True
                        break
        if direction is None:
            # fallback: centroid-pair line scored by physical structure —
            # adjacent sections nearly touch, so along-line gaps must match
            # the sections' own lengths, and along-line order must match
            # the image's left-to-right order. A corrupted centroid fails
            # both and cannot win the line.
            best = None
            for i in range(len(chain)):
                for j in range(i + 1, len(chain)):
                    span = cents[j] - cents[i]
                    norm = float(np.linalg.norm(span))
                    if norm < 0.3:
                        continue
                    cand = span / norm
                    along = (cents - cents[i]) @ cand
                    if not all(
                        along[k] < along[k + 1] for k in range(len(chain) - 1)
                    ):
                        continue
                    score = sum(
                        abs(
                            (along[k + 1] - along[k])
                            - (lengths[k] + lengths[k + 1]) / 2
                        )
                        for k in range(len(chain) - 1)
                    )
                    if best is None or score < best[0]:
                        best = (score, i, cand)
            if best is None:
                continue
            _, anchor, direction = best
            centre = cents[anchor]
            angle = float(np.arctan2(direction[1], direction[0]))
            axis_snapped = False
            if theta is not None:
                for axis in (theta, theta + np.pi / 2):
                    if abs(((angle - axis + np.pi / 2) % np.pi) - np.pi / 2) < np.radians(20):
                        direction = np.array([np.cos(axis), np.sin(axis)])
                        axis_snapped = True
                        break
            thickness = float(
                np.median([_rect_sides(e["rect_snapped"])[1] for e in chain])
            )
        normal = np.array([-direction[1], direction[0]])
        # orient the line so along-position increases with image x, judged
        # by the members closest to the line (outliers get no vote)
        chain_x0 = min(e["image_bbox"][0] for e in chain)
        chain_x1 = max(e["image_bbox"][2] for e in chain)

        def image_fraction(e: dict) -> float:
            b = e["image_bbox"]
            return ((b[0] + b[2]) / 2 - chain_x0) / max(1, chain_x1 - chain_x0)

        ranked = sorted(
            chain,
            key=lambda e: abs(
                float((np.asarray(e["centroid_xy"], float) - centre) @ normal)
            ),
        )
        vote = sum(
            (image_fraction(e) - 0.5)
            * float((np.asarray(e["centroid_xy"], float) - centre) @ direction)
            for e in ranked[:2]
        )
        if vote < 0:
            direction, normal = -direction, -normal
        chain_snapshot = [
            (list(map(list, e["rect_snapped"])), list(e["centroid_xy"]))
            for e in chain
        ]
        res_before = None
        if union is not None and direction is not None:
            res_before = _reprojection_offset(
                chain, union, direction,
                np.array([-direction[1], direction[0]]),
                points3d, valid, transform, probe_only=True,
            )
        span = rect_long if rect_long is not None else sum(lengths)
        # image adjacency welds PARALLEL planes seen edge-on (a gate panel
        # in front of a mesh wall) into one chain; split by perpendicular
        # offset clusters and keep the line for the majority cluster only
        perps = [
            float((np.asarray(e["centroid_xy"], float) - centre) @ normal)
            for e in chain
        ]
        order = np.argsort(perps)
        clusters = [[int(order[0])]]
        for a, b in zip(order[:-1], order[1:]):
            if perps[int(b)] - perps[int(a)] > 0.30:
                clusters.append([])
            clusters[-1].append(int(b))
        keep_cluster = max(
            clusters, key=lambda c: (len(c), -abs(np.mean([perps[i] for i in c])))
        )
        kept_index = sorted(keep_cluster)
        # a far-off member is EITHER a different parallel structure (its
        # image box nests inside the kept sections — a mesh wall behind a
        # rail) OR the same line with a corrupted centroid (its image box
        # sits BESIDE the kept sections). Rescue the side-by-side ones —
        # image-fraction seating exists precisely for them.
        for i in range(len(chain)):
            if i in kept_index:
                continue
            x0, _, x1, _ = chain[i]["image_bbox"]
            width_i = max(1, x1 - x0)
            worst = 0.0
            for j in kept_index:
                kx0, _, kx1, _ = chain[j]["image_bbox"]
                overlap = max(0, min(x1, kx1) - max(x0, kx0))
                worst = max(worst, overlap / width_i)
            if worst < 0.5:
                kept_index.append(i)
        kept_index = sorted(kept_index)
        if len(kept_index) < 2:
            for e in chain:
                e.pop("guard_chain", None)
            continue
        dropped = [i for i in range(len(chain)) if i not in kept_index]
        for i in dropped:
            chain[i].pop("guard_chain", None)
        chain = [chain[i] for i in kept_index]
        lengths = [lengths[i] for i in kept_index]
        if rect_long is not None:
            # seat purely by each section's position in the IMAGE mapped
            # onto the ray-cast union line — monotonic by construction,
            # immune to every flavour of centroid corruption
            alongs = [(image_fraction(e) - 0.5) * span for e in chain]
        else:
            raw = []
            for e in chain:
                c = np.asarray(e["centroid_xy"], float)
                if abs(float((c - centre) @ normal)) <= 0.5:
                    raw.append(float((c - centre) @ direction))
                else:
                    raw.append((image_fraction(e) - 0.5) * span)
            # chain is image-ordered; the projections must be too
            alongs = sorted(raw)
        thickness = float(min(thickness, 0.25))
        for e, along in zip(chain, alongs):
            seat = centre + along * direction
            length = _rect_sides(e["rect_snapped"])[0]
            # the seat rect's long axis IS the line: never let the line
            # thickness exceed the section length and flip the axis
            length = max(length, thickness * 1.05)
            corners = [
                seat + length / 2 * direction + thickness / 2 * normal,
                seat + length / 2 * direction - thickness / 2 * normal,
                seat - length / 2 * direction - thickness / 2 * normal,
                seat - length / 2 * direction + thickness / 2 * normal,
            ]
            e["rect_snapped"] = [
                [round(float(x), 3), round(float(y), 3)] for x, y in corners
            ]
            e["centroid_xy"] = [round(float(seat[0]), 2), round(float(seat[1]), 2)]
            e["footprint_method"] = "guard-line"
            e["guard_axis_snapped"] = axis_snapped
        # reprojection feedback: cast the seated line back into the photo
        # and slide it along its normal until it sits on the union mask's
        # ground contact — contact-edge ray-casting inherits a few pixels
        # of mask shadow bleed, which lands the whole line ~0.5 m toward
        # the camera. The loop that verifies is the loop that corrects.
        if union is not None:
            for _pass in range(3):
                shift = _reprojection_offset(
                    chain, union, direction, normal, points3d, valid, transform
                )
                if shift is None or abs(shift) < 0.05:
                    break
                for e in chain:
                    e["rect_snapped"] = [
                        [
                            round(float(x + shift * normal[0]), 3),
                            round(float(y + shift * normal[1]), 3),
                        ]
                        for x, y in e["rect_snapped"]
                    ]
                    e["centroid_xy"] = [
                        round(float(e["centroid_xy"][0] + shift * normal[0]), 2),
                        round(float(e["centroid_xy"][1] + shift * normal[1]), 2),
                    ]
        if union is not None and direction is not None:
            res_after = _reprojection_offset(
                chain, union, direction, normal,
                points3d, valid, transform, probe_only=True,
            )
            if res_before is not None and (
                res_after is None or res_after > res_before + 1.0
            ):
                for e, (rect_old, cent_old) in zip(chain, chain_snapshot):
                    e["rect_snapped"] = rect_old
                    e["centroid_xy"] = cent_old
    # the same measure-and-correct loop, for every fence structure that is
    # NOT part of a chain: reproject its own base line and slide it onto
    # its own mask's ground contact. This is what took the first run's
    # gate from raw placement to <7% reprojection error — generalized.
    for e in entries:
        if e.get("guard_chain") is not None:
            continue
        if not any(k in e["label"] for k in CONTACT_FAMILY):
            continue
        if not e.get("rect_snapped") or not e.get("image_bbox"):
            continue
        mask = _member_mask(e)
        if mask is None:
            continue
        r = np.asarray(e["rect_snapped"], float)
        edge1, edge2 = r[1] - r[0], r[2] - r[1]
        if np.linalg.norm(edge1) < np.linalg.norm(edge2):
            edge1, edge2 = edge2, edge1
        norm1 = float(np.linalg.norm(edge1))
        if norm1 < 1e-6:
            continue
        direction = edge1 / norm1
        normal = np.array([-direction[1], direction[0]])
        res_before = _reprojection_offset(
            [e], mask, direction, normal, points3d, valid, transform,
            probe_only=True,
        )
        single_snapshot = (
            list(map(list, e["rect_snapped"])), list(e["centroid_xy"])
        )
        line = _verified_contact_line(mask, floor_xyz)
        if line is not None:
            # evidence read straight off the contact pixels — but position
            # evidence must not overrule orientation: if the fitted line
            # disagrees with the rect's established direction by >25°, take
            # only the line's perpendicular POSITION (translate, don't
            # rotate) — a contact line caught on a corner or an occluder
            # must not spin the panel
            centre_l, direction_l, extent_l = line
            gap = abs((
                (np.arctan2(direction_l[1], direction_l[0])
                 - np.arctan2(direction[1], direction[0]) + np.pi / 2)
                % np.pi) - np.pi / 2)
            if gap > np.radians(25):
                shift = float((centre_l - np.asarray(e["centroid_xy"], float)) @ normal)
                _apply_shift(e, shift, normal)
                res_after = _reprojection_offset(
                    [e], mask, direction, normal, points3d, valid,
                    transform, probe_only=True,
                )
                if res_before is not None and (
                    res_after is None or res_after > res_before + 1.0
                ):
                    e["rect_snapped"], e["centroid_xy"] = single_snapshot
                continue
            normal_l = np.array([-direction_l[1], direction_l[0]])
            length = min(norm1, extent_l)
            thickness = max(
                0.05, min(float(np.linalg.norm(edge2)), 0.25)
            )
            c_old = np.asarray(e["centroid_xy"], float)
            along = float(np.clip(
                (c_old - centre_l) @ direction_l,
                -extent_l / 2 + length / 2,
                extent_l / 2 - length / 2,
            )) if extent_l > length else 0.0
            seat = centre_l + along * direction_l
            corners = [
                seat + length / 2 * direction_l + thickness / 2 * normal_l,
                seat + length / 2 * direction_l - thickness / 2 * normal_l,
                seat - length / 2 * direction_l - thickness / 2 * normal_l,
                seat - length / 2 * direction_l + thickness / 2 * normal_l,
            ]
            e["rect_snapped"] = [
                [round(float(x), 3), round(float(y), 3)] for x, y in corners
            ]
            e["centroid_xy"] = [
                round(float(seat[0]), 2), round(float(seat[1]), 2)
            ]
            res_after = _reprojection_offset(
                [e], mask, direction_l, normal_l, points3d, valid,
                transform, probe_only=True,
            )
            if res_before is not None and (
                res_after is None or res_after > res_before + 1.0
            ):
                e["rect_snapped"], e["centroid_xy"] = single_snapshot
            continue
        for _pass in range(3):
            shift = _reprojection_offset(
                [e], mask, direction, normal, points3d, valid, transform
            )
            if shift is None or abs(shift) < 0.02:
                break
            _apply_shift(e, shift, normal)
    return theta


def _verified_contact_line(mask, floor_xyz, min_points=12):
    """Fit a guard line DIRECTLY from verified ground-contact pixels: the
    mask's bottom-edge pixels that sit within 0.35 m of the floor are 3D
    points ON the structure's base line — opaque metal, reliable depth.
    Evidence read straight off, no sliding search. Returns (centre,
    direction, along_extent) or None."""
    profile: dict[int, int] = {}
    pvs, pus = np.nonzero(mask)
    for u, v in zip(pus, pvs):
        if u not in profile or v > profile[u]:
            profile[u] = v
    pts = []
    for u, v in profile.items():
        xyz = floor_xyz[v, u]
        if np.isfinite(xyz).all() and xyz[2] < 0.35:
            pts.append(xyz[:2])
    if len(pts) < min_points:
        return None
    pts = np.asarray(pts, float)
    for _ in range(2):
        centre = np.median(pts, axis=0)
        _, _, vt = np.linalg.svd(pts - centre)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        residual = np.abs((pts - centre) @ normal)
        keep = residual < 0.4
        if keep.sum() < min_points:
            return None
        pts = pts[keep]
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre)
    direction = vt[0]
    along = (pts - centre) @ direction
    extent = float(along.max() - along.min())
    if extent < 0.5:
        return None
    return centre, direction, extent


def _apply_shift(entry: dict, shift: float, normal) -> None:
    entry["rect_snapped"] = [
        [
            round(float(x + shift * normal[0]), 3),
            round(float(y + shift * normal[1]), 3),
        ]
        for x, y in entry["rect_snapped"]
    ]
    entry["centroid_xy"] = [
        round(float(entry["centroid_xy"][0] + shift * normal[0]), 2),
        round(float(entry["centroid_xy"][1] + shift * normal[1]), 2),
    ]


def _reprojection_offset(
    chain, union_mask, direction, normal, points3d, valid, transform,
    probe_only: bool = False,
) -> float | None:
    """Signed normal offset that lands the chain's base line on the union
    mask's bottom edge when reprojected through the dense floor geometry."""
    from scipy.spatial import cKDTree

    finite = (
        valid
        & np.isfinite(points3d).all(axis=2)
        & (np.abs(points3d).sum(axis=2) > 1e-6)
    )
    floor_pts = transform.apply(points3d[finite])
    vs, us = np.nonzero(finite)
    on_floor = np.abs(floor_pts[:, 2]) < 0.10
    if on_floor.sum() < 500:
        return None
    tree = cKDTree(floor_pts[on_floor][:, :2])
    u_floor, v_floor = us[on_floor], vs[on_floor]
    zmap = np.full(valid.shape, np.nan, np.float32)
    zmap[vs, us] = floor_pts[:, 2]
    profile: dict[int, int] = {}
    pvs, pus = np.nonzero(union_mask)
    for u, v in zip(pus, pvs):
        if u not in profile or v > profile[u]:
            profile[u] = v
    # correction target = VERIFIED ground contact only: a bottom pixel a
    # metre off the floor is an occlusion boundary, not a contact line
    profile = {
        u: v
        for u, v in profile.items()
        if np.isfinite(zmap[v, u]) and zmap[v, u] < 0.35
    }
    if len(profile) < 8:
        return None
    base = np.array(
        [corner for e in chain for corner in e["rect_snapped"]], float
    )
    along = base @ direction
    centre_line = base.mean(axis=0)
    lo, hi = float(along.min()), float(along.max())
    t = np.linspace(lo, hi, 40)
    line = centre_line[None, :] + (
        (t - float(centre_line @ direction))[:, None] * direction[None, :]
    )

    def mean_signed_dv(offset: float) -> tuple[float | None, int]:
        pts = line + offset * normal[None, :]
        dists, idx = tree.query(pts)
        deltas = []
        for d, i in zip(dists, idx):
            if d > 0.20:
                continue
            u, v = int(u_floor[i]), int(v_floor[i])
            if u in profile:
                deltas.append(v - profile[u])
        if len(deltas) < 10:
            return None, len(deltas)
        return float(np.median(deltas)), len(deltas)

    if probe_only:
        dv, matched = mean_signed_dv(0.0)
        return None if dv is None else abs(dv)
    best = None
    for offset in np.linspace(-0.7, 0.7, 29):
        dv, matched = mean_signed_dv(float(offset))
        if dv is None:
            continue
        if best is None or abs(dv) < abs(best[1]):
            best = (float(offset), dv)
    if best is None or abs(best[1]) > 25:
        return None
    return best[0]


def _refine_mask(run: Path, slug: str, height: int, width: int) -> np.ndarray | None:
    """Best-score SAM mask of a box refinement, resized to the geometry
    grid — so refinements ride the SAME geometry pipeline as everything
    else instead of their own (prior-free) measurement path."""
    from PIL import Image as PILImage

    cache = run / "refinements" / f"{slug}.json"
    if not cache.exists():
        return None
    response = json.loads(cache.read_text())
    rles = response.get("rle") or []
    if isinstance(rles, str):
        rles = [rles]
    if not rles:
        return None
    scores = response.get("scores") or [1.0] * len(rles)
    # full-resolution dims come from the run's own input image — a
    # hardcoded landscape assumption garbled every portrait capture
    with PILImage.open(next((run / "input").glob("image_*"))) as full:
        full_width, full_height = full.size
    mask = decode_coco_rle(
        rles[int(np.argmax(scores))], height=full_height, width=full_width
    ).astype(np.uint8)
    return (
        np.asarray(PILImage.fromarray(mask * 255).resize((width, height)))
        > 127
    )


def _reconcile_enumeration(
    run: Path, frame: GeometryFrame, entries: list[dict],
    phrases: list[str], *, live: bool,
) -> list[dict]:
    """The guarantee the enumeration owes the user: anything the VLM says
    exists either ends up with a measured instance or is EXPLICITLY
    recorded as unresolved — never silently dropped. Text-prompt SAM has
    a long tail of zero-hits on domain-specific appearances (a branded
    parts container scored six straight zeros); the fallback localizes by
    VLM box and segments by box prompt, which bypasses the text prior."""
    covered = {e["label"] for e in entries}
    missing = [ph for ph in phrases if ph not in covered]
    unresolved: list[dict] = []
    if missing and live:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from ehs_spatial.agent import _gemini_locator
        from ehs_spatial.providers.gemini import GeminiAdapter
        from PIL import Image as PILImage

        adapter = GeminiAdapter()
        with PILImage.open(next((run / "input").glob("image_*"))) as im:
            width, height = im.size
        log_path = run / "refinements.json"
        log = json.loads(log_path.read_text()) if log_path.exists() else []
        known = {
            item["label"].replace(" ", "_")
            + "_" + "_".join(str(v) for v in item["box"])
            for item in log
        }
        added = False
        for phrase in missing:
            try:
                located = _gemini_locator(
                    str(next((run / "input").glob("image_*"))), phrase, adapter
                )
            except Exception as error:
                unresolved.append({"phrase": phrase, "reason": str(error)[:100]})
                continue
            if not located.found or max(located.box_2d) > 1000:
                unresolved.append(
                    {"phrase": phrase, "reason": located.rationale[:100]}
                )
                continue
            y1, x1, y2, x2 = located.box_2d
            if not (y1 < y2 and x1 < x2):
                unresolved.append({"phrase": phrase, "reason": "degenerate box"})
                continue
            box = [
                max(0, int(x1 / 1000 * width)),
                max(0, int(y1 / 1000 * height)),
                min(width, int(x2 / 1000 * width)),
                min(height, int(y2 / 1000 * height)),
            ]
            slug = phrase.replace(" ", "_") + "_" + "_".join(str(v) for v in box)
            if slug in known:
                continue
            destination = run / "refinements" / f"{slug}.json"
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists():
                try:
                    import base64 as _b64

                    from ehs_spatial.providers.sam3 import sam_subscribe

                    response = sam_subscribe(
                        "fal-ai/sam-3-1/image-rle",
                        arguments={
                            "image_url": "data:image/png;base64,"
                            + _b64.b64encode(
                                next((run / "input").glob("image_*")).read_bytes()
                            ).decode(),
                            "box_prompts": [
                                {
                                    "x_min": box[0], "y_min": box[1],
                                    "x_max": box[2], "y_max": box[3],
                                }
                            ],
                            "return_multiple_masks": True,
                            "include_scores": True,
                            "max_masks": 3,
                        },
                    )
                except Exception as error:
                    unresolved.append(
                        {"phrase": phrase, "reason": str(error)[:100]}
                    )
                    continue
                destination.write_text(json.dumps(response) + "\n")
            log.append(
                {
                    "label": phrase, "box": box,
                    "source": "enumeration-reconcile",
                }
            )
            known.add(slug)
            added = True
        if added:
            log_path.write_text(json.dumps(log, indent=2) + "\n")
    return unresolved


def _ingest_refinements(
    run: Path,
    frame: GeometryFrame,
    transform,
    moge_maps,
    entries: list[dict],
) -> None:
    """Fence-family box refinements become first-class geometry entries:
    normal split, nearest-cluster lift, and (downstream) contact edge,
    guard-line chaining, and reprojection feedback — one process for every
    photo, whether SAM found the object itself or a reviewer boxed it.
    Refinements duplicating an existing fence instance (IoU>0.5) are
    skipped — the inventory version already went through the pipeline."""
    log_path = run / "refinements.json"
    # the detection layer's fence-family results flow in through the same
    # door: persist them in refinement format (same SAM response schema),
    # and every downstream consumer — this ingest, the report's photo
    # pick, reprojection — inherits them with zero special cases
    det_path = run / "detection" / "detections.json"
    if det_path.exists():
        log = json.loads(log_path.read_text()) if log_path.exists() else []
        known = {
            item["label"].replace(" ", "_")
            + "_"
            + "_".join(str(v) for v in item["box"])
            for item in log
        }
        added = False
        for det in json.loads(det_path.read_text()).get("detections", []):
            if "rle" not in det:
                continue
            slug = (
                det["label"].replace(" ", "_")
                + "_"
                + "_".join(str(v) for v in det["box"])
            )
            if slug in known:
                continue
            destination = run / "refinements" / f"{slug}.json"
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists():
                source = (
                    run
                    / "detection"
                    / "sam"
                    / (
                        f"{det['item_id']}_"
                        + "_".join(str(v) for v in det["box"])
                        + ".json"
                    )
                )
                destination.write_text(
                    source.read_text()
                    if source.exists()
                    else json.dumps(
                        {
                            "rle": [det["rle"]],
                            "scores": [det.get("sam_score", 1.0)],
                        }
                    )
                    + "\n"
                )
            log.append(
                {
                    "label": det["label"],
                    "box": list(det["box"]),
                    "sam_score": det.get("sam_score"),
                    "source": "detection",
                }
            )
            known.add(slug)
            added = True
        if added:
            log_path.write_text(json.dumps(log, indent=2) + "\n")
    if not log_path.exists():
        return
    points3d = np.load(frame.pts3d_path)
    valid = np.load(frame.valid_mask_path).astype(bool)
    height, width = valid.shape
    finite = (
        valid
        & np.isfinite(points3d).all(axis=2)
        & (np.abs(points3d).sum(axis=2) > 1e-6)
    )
    existing_fence_masks = []
    for entry in entries:
        if entry.get("refine_slug"):
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", entry["label"]).strip("_")
        cache = run / "inventory" / "sam" / f"{frame.frame_id}__{slug}.json"
        if not cache.exists():
            continue
        rles = json.loads(cache.read_text()).get("rle") or []
        if isinstance(rles, str):
            rles = [rles]
        mask = np.zeros((height, width), bool)
        for member in entry.get("merged_instances") or [entry["instance"]]:
            if member < len(rles):
                mask |= decode_coco_rle(
                    rles[member], height=height, width=width
                ).astype(bool)
        existing_fence_masks.append(mask)
    # decode every refinement mask first, then carve sibling overlaps:
    # when the detector says two objects are distinct, their pixels must
    # be too — a box prompt around a container drags in the deflector
    # wing that intrudes into its box, while the wing has its own
    # detection. The smaller mask owns the intersection; the carved mask
    # is written back so display, index and measurement all agree.
    decoded_items: list[tuple[dict, str, np.ndarray]] = []
    for item in json.loads(log_path.read_text()):
        slug = (
            item["label"].replace(" ", "_")
            + "_"
            + "_".join(str(v) for v in item["box"])
        )
        mask = _refine_mask(run, slug, height, width)
        if mask is None or mask.sum() < MIN_MASK_PIXELS:
            continue
        decoded_items.append((item, slug, mask))
    decoded_items.sort(key=lambda t: int(t[2].sum()))
    # same-label high-overlap pairs are the SAME object seen twice (a
    # manual refinement and a detection of one panel): keep one, drop the
    # other outright — carving is only for DISTINCT objects
    kept_items: list[tuple[dict, str, np.ndarray]] = []
    for item, slug, mask in sorted(
        decoded_items, key=lambda t: -int(t[2].sum())
    ):
        duplicate = False
        bx = item["box"]
        area_box = max(1, (bx[2] - bx[0]) * (bx[3] - bx[1]))
        for item_k, _, _mask_k in kept_items:
            if item_k["label"] != item["label"]:
                continue
            kx = item_k["box"]
            iw = max(0, min(bx[2], kx[2]) - max(bx[0], kx[0]))
            ih = max(0, min(bx[3], kx[3]) - max(bx[1], kx[1]))
            inter = iw * ih
            union_box = (
                area_box
                + max(1, (kx[2] - kx[0]) * (kx[3] - kx[1]))
                - inter
            )
            # boxes are the processing-independent evidence: two same-label
            # boxes on one spot are one object however their masks were
            # later cleaned or filled
            if inter / union_box > 0.5:
                duplicate = True
                break
        if not duplicate:
            kept_items.append((item, slug, mask))
    decoded_items = sorted(kept_items, key=lambda t: int(t[2].sum()))
    for i, (item_i, slug_i, mask_i) in enumerate(decoded_items):
        small_area = int(mask_i.sum())
        if not small_area:
            continue
        for j in range(i + 1, len(decoded_items)):
            item_j, slug_j, mask_j = decoded_items[j]
            inter = int((mask_i & mask_j).sum())
            if inter > 0.6 * small_area:
                mask_j &= ~mask_i
                cache_j = run / "refinements" / f"{slug_j}.json"
                try:
                    response = json.loads(cache_j.read_text())
                    rles = response.get("rle") or []
                    if isinstance(rles, str):
                        rles = [rles]
                    scores = response.get("scores") or [1.0] * len(rles)
                    best = int(np.argmax(scores))
                    from PIL import Image as PILImage

                    with PILImage.open(
                        next((run / "input").glob("image_*"))
                    ) as full:
                        fw, fh = full.size
                    upscaled = (
                        np.asarray(
                            PILImage.fromarray(
                                mask_j.astype(np.uint8) * 255
                            ).resize((fw, fh))
                        )
                        > 127
                    )
                    rles[best] = encode_coco_rle(upscaled)
                    response["rle"] = rles
                    cache_j.write_text(json.dumps(response) + "\n")
                except Exception:
                    pass
    for item, slug, mask in decoded_items:
        if mask.sum() < MIN_MASK_PIXELS:
            continue
        duplicate = False
        for seen in existing_fence_masks:
            inter = int((mask & seen).sum())
            union_px = int((mask | seen).sum())
            if union_px and inter / union_px > 0.5:
                duplicate = True
                break
        if duplicate:
            continue
        is_contact = any(k in item["label"] for k in CONTACT_FAMILY)
        split = None
        if moge_maps is not None and is_contact:
            split = _normal_split(mask, *moge_maps)
        if split is not None:
            if True:
                mask = split
                # the raw box-prompt mask bleeds over whatever stands in
                # front of / behind the glass; persist the cleaned framed
                # quad so the interaction layer inherits it (same
                # treatment inventory instances get)
                cache = run / "refinements" / f"{slug}.json"
                try:
                    response = json.loads(cache.read_text())
                    rles = response.get("rle") or []
                    if isinstance(rles, str):
                        rles = [rles]
                    scores = response.get("scores") or [1.0] * len(rles)
                    best = int(np.argmax(scores))
                    rles[best] = encode_coco_rle(_hull_fill(split))
                    response["rle"] = rles
                    cache.write_text(json.dumps(response) + "\n")
                except Exception:
                    pass
        selected = mask & finite
        depth_map = points3d[..., 2]
        if selected.sum() < MIN_MASK_PIXELS:
            continue
        near = np.percentile(depth_map[selected], 10)
        selected &= depth_map <= near + max(0.25, 0.12 * near)
        cloud = _clean(transform.apply(points3d[selected]))
        if cloud is None:
            continue
        hull = MultiPoint([(x, y) for x, y in cloud[:, :2]]).convex_hull
        # a thin section seen edge-on legitimately has a tiny footprint;
        # the point-count gates already killed the noise cases
        if not isinstance(hull, Polygon) or hull.area < 0.004:
            continue
        top = float(np.quantile(cloud[:, 2], 0.95))
        ys, xs = np.nonzero(mask)
        box = hull.minimum_rotated_rectangle
        corners = list(box.exterior.coords)[:4]
        sides = sorted(
            np.hypot(
                corners[i][0] - corners[i - 1][0],
                corners[i][1] - corners[i - 1][1],
            )
            for i in range(1, 3)
        )
        entries.append(
            {
                "label": item["label"],
                "instance": None,
                "refine_slug": slug,
                "frame": frame.frame_id,
                "score": item.get("sam_score", 1.0),
                "points": int(len(cloud)),
                "height_m": round(top, 2),
                "size_m": f"{sides[1]:.2f}x{sides[0]:.2f}",
                "footprint_area_m2": round(float(hull.area), 2),
                "centroid_xy": [
                    round(float(hull.centroid.x), 2),
                    round(float(hull.centroid.y), 2),
                ],
                "camera_dist_m": round(
                    float(Point(0.0, 0.0).distance(hull)), 2
                ),
                "orientation_deg": None,
                "tilt_deg": None,
                "image_bbox": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                ],
                "footprint": [
                    (round(float(x), 3), round(float(y), 3))
                    for x, y in list(hull.exterior.coords)[:-1]
                ],
            }
        )


def _enforce_row_order(entries: list[dict], theta: float | None) -> None:
    """A guard line is a BOUNDARY: an entity whose centroid sits behind it
    cannot have footprint spilling in front of it — the spill is pixels
    seen THROUGH the guarding plus overhang projection, not floor
    occupancy. Clip such footprints at the line (the reviewer's rule:
    first the fence, then the machine's territory)."""
    from shapely.geometry import Polygon as ShapelyPolygon

    chains: dict[int, list[dict]] = {}
    for e in entries:
        if e.get("guard_chain") is not None and e.get("rect_snapped"):
            chains.setdefault(e["guard_chain"], []).append(e)
    lines = []
    for members in chains.values():
        if len(members) < 2:
            continue
        corners = np.array(
            [c for m in members for c in m["rect_snapped"]], float
        )
        centre = corners.mean(axis=0)
        _, _, vt = np.linalg.svd(corners - centre)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        # orient the normal toward the camera (origin)
        if (np.zeros(2) - centre) @ normal < 0:
            normal = -normal
        along = (corners - centre) @ direction
        lines.append((centre, direction, normal, float(along.min()), float(along.max())))
    if not lines:
        return
    for e in entries:
        if any(k in e["label"] for k in CONTACT_FAMILY):
            continue
        if _is_flat_zone(e["label"]) or not e.get("footprint"):
            continue
        c = np.asarray(e["centroid_xy"], float)
        for centre, direction, normal, lo, hi in lines:
            if (c - centre) @ normal > -0.05:
                continue  # centroid on the camera side: front-row, untouched
            fp = np.asarray(e["footprint"], float)
            along_fp = (fp - centre) @ direction
            overlap = min(hi, float(along_fp.max())) - max(lo, float(along_fp.min()))
            span = max(1e-6, float(along_fp.max() - along_fp.min()))
            if overlap < 0.3 * span:
                continue  # barely faces this line laterally
            offsets = (fp - centre) @ normal
            if float(offsets.max()) <= 0.02:
                continue  # already fully behind
            # clip: keep the half-plane behind the line (+2 cm tolerance)
            far = 100.0
            half = ShapelyPolygon(
                [
                    centre + lo * direction - far * direction + 0.02 * normal,
                    centre + hi * direction + far * direction + 0.02 * normal,
                    centre + hi * direction + far * direction - far * normal,
                    centre + lo * direction - far * direction - far * normal,
                ]
            )
            clipped = ShapelyPolygon(fp).intersection(half)
            if clipped.is_empty or clipped.geom_type != "Polygon":
                continue
            if clipped.area < 0.05:
                continue
            e["footprint"] = [
                (round(float(x), 3), round(float(y), 3))
                for x, y in list(clipped.exterior.coords)[:-1]
            ]
            e["footprint_area_m2"] = round(float(clipped.area), 2)
            e["centroid_xy"] = [
                round(float(clipped.centroid.x), 2),
                round(float(clipped.centroid.y), 2),
            ]
            snapped = _snap_rect(e["footprint"], theta, allow_free=True)
            if snapped is not None:
                e["rect_snapped"] = snapped
            e["row_clipped"] = True


# substring match — "cart" covers "material cart" / "docking cart" / the
# bare VLM enumeration phrase, so the payload loop closes regardless of
# which layer named the object
POLICY_SUBJECT_LABELS = (
    "cart", "pallet", "crate", "container", "portable work platform",
    "step ladder",
)


def _apply_payload_entities(run: Path, entries: list[dict]) -> None:
    """Ingested payload detections (the container a clearance rule
    measures) become scene entities so policies evaluate against them —
    detection closes the loop into the verdict, idempotent per slug."""
    scene_path = run / "scene.json"
    if not scene_path.exists():
        return
    payload = [
        e for e in entries
        if e.get("refine_slug")
        and any(k in e["label"] for k in POLICY_SUBJECT_LABELS)
        and e.get("footprint")
    ]
    if not payload:
        return
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ehs_spatial.contracts import Entity3D, PolicySpec, SceneMap
    from ehs_spatial.policy import evaluate_policies

    scene = SceneMap.model_validate(json.loads(scene_path.read_text()))
    changed = False
    for e in payload:
        observation_id = f"refine:{e['refine_slug']}"
        if any(observation_id in ent.observation_ids for ent in scene.entities):
            continue
        index = sum(
            1 for ent in scene.entities if ent.entity_id.startswith("refine-")
        ) + 1
        scene.entities.append(
            Entity3D(
                entity_id=f"refine-{index:02d}",
                label=e["label"],
                observation_ids=[observation_id],
                centroid_xyz=(
                    float(e["centroid_xy"][0]),
                    float(e["centroid_xy"][1]),
                    max(0.0, float(e["height_m"]) / 2),
                ),
                footprint_xy=[(float(x), float(y)) for x, y in e["footprint"]],
                height_m=max(0.01, float(e["height_m"])),
                evidence_frame_ids=[e.get("frame", "frame_0001")],
            )
        )
        changed = True
    if not changed:
        return
    scene_path.write_text(scene.model_dump_json(indent=2) + "\n")
    policies_path = run / "policies.json"
    if policies_path.exists():
        envelope = json.loads(policies_path.read_text())
        specs = [PolicySpec.model_validate(x) for x in envelope.get("specs", [])]
        results = evaluate_policies(specs, scene, capture_frame_count=1)
        envelope["results"] = [r.model_dump(mode="json") for r in results]
        policies_path.write_text(json.dumps(envelope, indent=2) + "\n")
        print("payload entities applied; policies re-evaluated:")
        for r in results:
            print("  ", r.policy_id, getattr(r.status, "value", r.status))


def _hull_fill(mask: np.ndarray) -> np.ndarray:
    """Convex hull of the mask's significant connected components,
    rasterized. Stray specks (<1% of the mask) are dropped first so they
    cannot drag the hull past the structure's frame."""
    from PIL import Image as PILImage
    from PIL import ImageDraw
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum_labels(mask, labels, range(1, count + 1))
    significant = np.isin(labels, 1 + np.flatnonzero(sizes >= 0.01 * mask.sum()))
    ys, xs = np.nonzero(significant)
    if len(xs) < 3:
        return mask
    hull = MultiPoint(list(zip(xs.tolist(), ys.tolist()))).convex_hull
    if hull.geom_type != "Polygon":
        return mask
    canvas = PILImage.new("1", (mask.shape[1], mask.shape[0]))
    ImageDraw.Draw(canvas).polygon(
        [(float(x), float(y)) for x, y in hull.exterior.coords], fill=1
    )
    return np.asarray(canvas, dtype=bool)


def _clean(points: np.ndarray) -> np.ndarray | None:
    if len(points) < MIN_CLOUD_POINTS:
        return None
    radius = np.linalg.norm(points[:, :2], axis=1)
    points = points[radius > 0.15]
    if len(points) < MIN_CLOUD_POINTS:
        return None
    radius = np.linalg.norm(points[:, :2], axis=1)
    centre = np.median(radius)
    spread = np.median(np.abs(radius - centre)) + 1e-6
    points = points[np.abs(radius - centre) < 2.5 * spread]
    return points if len(points) >= MIN_CLOUD_POINTS else None


def _fit_walls(points: np.ndarray) -> list[dict]:
    """Vertical planes above the floor, as 2D wall segments. Deterministic
    RANSAC on the horizontal projection: a wall is a line in plan view."""
    rng = np.random.default_rng(0)
    remaining = points[(points[:, 2] > 0.5) & (points[:, 2] < 6.0)]
    if len(remaining) > 40000:
        remaining = remaining[:: len(remaining) // 40000]
    walls = []
    for _ in range(WALL_MAX_COUNT):
        if len(remaining) < WALL_MIN_INLIERS:
            break
        best = None
        for _ in range(400):
            a, b = remaining[rng.choice(len(remaining), 2, replace=False)][:, :2]
            direction = b - a
            length = np.linalg.norm(direction)
            if length < 1.0:
                continue
            normal = np.array([-direction[1], direction[0]]) / length
            distance = np.abs((remaining[:, :2] - a) @ normal)
            inliers = distance < 0.12
            count = int(inliers.sum())
            if best is None or count > best[0]:
                best = (count, inliers, a, normal)
        if best is None or best[0] < WALL_MIN_INLIERS:
            break
        count, inliers, anchor, normal = best
        cloud = remaining[inliers]
        along = np.array([-normal[1], normal[0]])
        projection = (cloud[:, :2] - anchor) @ along
        low, high = np.quantile(projection, [0.02, 0.98])
        if high - low < 1.0:
            remaining = remaining[~inliers]
            continue
        walls.append(
            {
                "start": (anchor + along * low).tolist(),
                "end": (anchor + along * high).tolist(),
                "length_m": round(float(high - low), 2),
                "height_m": round(float(np.quantile(cloud[:, 2], 0.95)), 2),
                "points": count,
            }
        )
        remaining = remaining[~inliers]
    return walls


CELL_SNAP_BAND = 0.45      # boundary structure counts as "on" a side, m
CELL_MIN_SIDE = 1.5        # opposing sides of a real cell sit farther apart, m
CELL_OUTSIDE_MARGIN = 2.0  # an aisle separates cells; closer overshoot is this cell's own outer structure, m
_CELL_AXIS_TOL_DEG = 15.0
# the hazard the cell rectangle encloses — anchors which side of the
# boundary evidence is "inside"
_CELL_MACHINE_KEYWORDS = ("robot", "machine", "gantry", "press", "arm")
CELL_MAX_REACH = 7.0       # a boundary farther than this from the machine
                           # belongs to another cell, m
CELL_GAP_MAX = 2.5         # an aisle-sized jump between boundary clusters
                           # means the far one is the neighbour cell, m
# CONTACT_FAMILY matches "panel"; a control panel is equipment, not a
# guard structure
_CELL_BOUNDARY_EXCLUDE = ("control", "button", "sign")


def _cell_frame(theta: float):
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

    def to_uv(xy):
        x, y = float(xy[0]), float(xy[1])
        return (x * cos_t + y * sin_t, -x * sin_t + y * cos_t)

    def to_xy(uv):
        u, v = float(uv[0]), float(uv[1])
        return (u * cos_t - v * sin_t, u * sin_t + v * cos_t)

    return to_uv, to_xy


def _boundary_evidence(
    walls: list[dict], entries: list[dict], theta: float
) -> list[dict]:
    """Every wall plane and fence-family footprint reduced to an
    axis-aligned line vote in the Manhattan frame: which axis it runs
    along, its perpendicular offset, its along-axis extent, and a weight
    (its length — long structures know the boundary better)."""
    to_uv, _ = _cell_frame(theta)
    votes = []

    def _vote(p1, p2, source):
        u1, v1 = to_uv(p1)
        u2, v2 = to_uv(p2)
        du, dv = u2 - u1, v2 - v1
        length = float(np.hypot(du, dv))
        if length < 0.8:
            return
        angle = np.degrees(np.arctan2(dv, du)) % 180.0
        if min(angle, 180.0 - angle) < _CELL_AXIS_TOL_DEG:
            # runs along u -> evidence for a v = const side
            votes.append({
                "family": "v", "offset": (v1 + v2) / 2.0,
                "extent": (min(u1, u2), max(u1, u2)),
                "weight": length, "source": source,
            })
        elif abs(angle - 90.0) < _CELL_AXIS_TOL_DEG:
            votes.append({
                "family": "u", "offset": (u1 + u2) / 2.0,
                "extent": (min(v1, v2), max(v1, v2)),
                "weight": length, "source": source,
            })

    for wall in walls:
        _vote(wall["start"], wall["end"], "wall")
    for entry in entries:
        if not any(k in entry["label"] for k in CONTACT_FAMILY):
            continue
        if any(k in entry["label"] for k in _CELL_BOUNDARY_EXCLUDE):
            continue
        rect = entry.get("rect_snapped") or entry.get("footprint")
        if not rect or len(rect) < 3:
            continue
        ring = [tuple(p) for p in rect]
        best = max(
            zip(ring, ring[1:] + ring[:1]),
            key=lambda pair: float(np.hypot(
                pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]
            )),
        )
        _vote(best[0], best[1], "fence")
    return votes


def _cluster_side_votes(votes: list[dict]) -> list[dict]:
    """1D gap clustering of perpendicular offsets; each cluster is one
    candidate cell side with pooled support and extent."""
    if not votes:
        return []
    ordered = sorted(votes, key=lambda vote: vote["offset"])
    clusters, current = [], [ordered[0]]
    for vote in ordered[1:]:
        if vote["offset"] - current[-1]["offset"] > 0.6:
            clusters.append(current)
            current = []
        current.append(vote)
    clusters.append(current)

    # single-linkage chains: interior guards spaced along the whole cell
    # can bridge the two real walls into one cluster — a physical cell
    # side is thin, so recursively split any chain wider than 1.5 m at
    # its largest internal gap
    def _split_wide(members: list[dict]) -> list[list[dict]]:
        if len(members) < 2 or (
            members[-1]["offset"] - members[0]["offset"] <= 1.5
        ):
            return [members]
        gaps = [
            members[i + 1]["offset"] - members[i]["offset"]
            for i in range(len(members) - 1)
        ]
        cut = int(np.argmax(gaps)) + 1
        return _split_wide(members[:cut]) + _split_wide(members[cut:])

    clusters = [part for chain in clusters for part in _split_wide(chain)]
    sides = []
    for members in clusters:
        weights = np.array([m["weight"] for m in members])
        offsets = np.array([m["offset"] for m in members])
        sides.append({
            "offset": float(np.average(offsets, weights=weights)),
            "support_m": round(float(weights.sum()), 2),
            "extent": (
                min(m["extent"][0] for m in members),
                max(m["extent"][1] for m in members),
            ),
            "sources": sorted({m["source"] for m in members}),
        })
    return sides


def _pick_side_pair(
    sides: list[dict], interior: float = 0.0, trusted: bool = False
) -> tuple[dict | None, dict | None]:
    """The cell rectangle ENCLOSES the hazard. Per direction (below /
    above the interior reference) the boundary is the OUTERMOST cluster
    that is still credible: support at least half the direction's
    strongest (a lone far fence next to a 6 m wall line is the neighbour
    cell), within physical reach of the machine (a structure 10 m out is
    another cell no matter how straight), and not across an aisle-sized
    gap from the nearer boundary evidence (a strong wall 4 m beyond our
    fence line is the neighbour's wall). Inner rows become interior
    dividers, never the far boundary."""
    # A FALLBACK interior estimate that lands on or beyond the boundary
    # evidence (camera standing at the front fence; clutter seen through
    # panels dragging the median out) would empty one bucket — pull it
    # just inside the evidence span. A machine-anchored interior is
    # trusted as-is: the machine legitimately sits beyond a side that
    # simply has no far evidence yet.
    if not trusted and len(sides) >= 2:
        lo = min(s["offset"] for s in sides)
        hi = max(s["offset"] for s in sides)
        if hi - lo > 0.6:
            interior = min(max(interior, lo + 0.3), hi - 0.3)

    def _outermost(candidates: list[dict], sign: float) -> dict | None:
        candidates = [
            s for s in candidates
            if abs(s["offset"] - interior) <= CELL_MAX_REACH
        ]
        if not candidates:
            return None
        strongest = max(s["support_m"] for s in candidates)
        credible = sorted(
            (s for s in candidates if s["support_m"] >= 0.5 * strongest),
            key=lambda s: sign * s["offset"],
        )
        pick = credible[0]
        for candidate in credible[1:]:
            if (
                sign * candidate["offset"] - sign * pick["offset"]
                > CELL_GAP_MAX
            ):
                break
            pick = candidate
        return pick

    low = _outermost([s for s in sides if s["offset"] < interior], -1.0)
    high = _outermost([s for s in sides if s["offset"] >= interior], 1.0)
    if low and high and high["offset"] - low["offset"] < CELL_MIN_SIDE:
        # sliver: both clusters hug the interior — trust the stronger one
        if low["support_m"] >= high["support_m"]:
            high = None
        else:
            low = None
    return low, high


def _fit_cell_rectangle(
    walls: list[dict], entries: list[dict], theta: float | None
) -> dict | None:
    """The cell premise: walls and guard structures of one workcell form a
    closed rectangle in the Manhattan frame. Fit its sides from pooled
    boundary evidence; sides with no evidence stay open (None) rather
    than invented."""
    if theta is None:
        return None
    votes = _boundary_evidence(walls, entries, theta)
    to_uv, _ = _cell_frame(theta)
    # Interior reference: the hazard the cell encloses. Prefer the
    # machine family; fall back to non-boundary content; last resort the
    # camera origin (the operator stands at the cell).
    machine = [
        to_uv(e["centroid_xy"]) for e in entries
        if e.get("centroid_xy")
        and any(k in e["label"] for k in _CELL_MACHINE_KEYWORDS)
        # "robot safety fence" / "machine guard" are boundary, not hazard
        and not any(k in e["label"] for k in CONTACT_FAMILY)
    ]
    content = machine or [
        to_uv(e["centroid_xy"]) for e in entries
        if e.get("centroid_xy")
        and not any(k in e["label"] for k in CONTACT_FAMILY)
    ]
    interior_u = float(np.median([c[0] for c in content])) if content else 0.0
    interior_v = float(np.median([c[1] for c in content])) if content else 0.0
    u_low, u_high = _pick_side_pair(
        _cluster_side_votes([v for v in votes if v["family"] == "u"]),
        interior_u,
        trusted=bool(machine),
    )
    v_low, v_high = _pick_side_pair(
        _cluster_side_votes([v for v in votes if v["family"] == "v"]),
        interior_v,
        trusted=bool(machine),
    )
    sides = {"u_min": u_low, "u_max": u_high, "v_min": v_low, "v_max": v_high}
    # A cell side is a wall-scale structure. A short interior fence
    # segment that happens to be the only vote in its direction must not
    # become a boundary — gate on absolute support relative to the
    # strongest side this scene produced.
    strongest = max(
        (s["support_m"] for s in sides.values() if s), default=0.0
    )
    floor_support = max(2.0, 0.15 * strongest)
    sides = {
        key: side if side and side["support_m"] >= floor_support else None
        for key, side in sides.items()
    }
    u_low, u_high = sides["u_min"], sides["u_max"]
    v_low, v_high = sides["v_min"], sides["v_max"]
    if sum(1 for s in sides.values() if s) < 2:
        return None
    _, to_xy = _cell_frame(theta)
    corners = None
    if all(sides.values()):
        u0, u1 = u_low["offset"], u_high["offset"]
        v0, v1 = v_low["offset"], v_high["offset"]
        corners = [
            [round(c, 3) for c in to_xy(uv)]
            for uv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))
        ]
    return {
        "theta_deg": round(float(np.degrees(theta)), 1),
        "sides": {
            key: None if side is None else {
                "offset": round(side["offset"], 3),
                "support_m": side["support_m"],
                "extent": [round(side["extent"][0], 3),
                           round(side["extent"][1], 3)],
                "sources": side["sources"],
            }
            for key, side in sides.items()
        },
        "corners": corners,
        "size_m": None if corners is None else [
            round(u_high["offset"] - u_low["offset"], 2),
            round(v_high["offset"] - v_low["offset"], 2),
        ],
    }


def _apply_cell_rectangle(
    cell: dict, walls: list[dict], entries: list[dict], theta: float
) -> None:
    """Constrain, don't invent: walls near a fitted side are re-seated
    exactly on it (kills the RANSAC fan); fence-family entries far outside
    the rectangle are tagged outside_cell (another cell's structure, kept
    but excluded from this cell's boundary). Entries are never moved —
    guard chains carry stronger, reprojection-verified evidence."""
    to_uv, to_xy = _cell_frame(theta)
    axis_of = {"u_min": "u", "u_max": "u", "v_min": "v", "v_max": "v"}

    def _nearest_side(family: str, offset: float):
        best_key, best_d = None, None
        for key, side in cell["sides"].items():
            if side is None or axis_of[key] != family:
                continue
            d = abs(offset - side["offset"])
            if best_d is None or d < best_d:
                best_key, best_d = key, d
        return best_key, best_d

    for wall in walls:
        (u1, v1), (u2, v2) = to_uv(wall["start"]), to_uv(wall["end"])
        du, dv = u2 - u1, v2 - v1
        angle = np.degrees(np.arctan2(dv, du)) % 180.0
        if min(angle, 180.0 - angle) < _CELL_AXIS_TOL_DEG:
            family, offset = "v", (v1 + v2) / 2.0
        elif abs(angle - 90.0) < _CELL_AXIS_TOL_DEG:
            family, offset = "u", (u1 + u2) / 2.0
        else:
            continue
        key, distance = _nearest_side(family, offset)
        if key is None or distance > CELL_SNAP_BAND:
            continue
        side_offset = cell["sides"][key]["offset"]
        if family == "v":
            start = to_xy((min(u1, u2), side_offset))
            end = to_xy((max(u1, u2), side_offset))
        else:
            start = to_xy((side_offset, min(v1, v2)))
            end = to_xy((side_offset, max(v1, v2)))
        wall["start"] = [round(start[0], 3), round(start[1], 3)]
        wall["end"] = [round(end[0], 3), round(end[1], 3)]
        wall["length_m"] = round(
            float(np.hypot(end[0] - start[0], end[1] - start[1])), 2
        )
        wall["cell_side"] = key

    bounds = {
        key: None if side is None else side["offset"]
        for key, side in cell["sides"].items()
    }
    for entry in entries:
        if not any(k in entry["label"] for k in CONTACT_FAMILY):
            continue
        if any(k in entry["label"] for k in _CELL_BOUNDARY_EXCLUDE):
            continue
        cx, cy = entry.get("centroid_xy") or (None, None)
        if cx is None:
            continue
        u, v = to_uv((cx, cy))
        overshoot = max(
            bounds["u_min"] - u if bounds["u_min"] is not None else 0.0,
            u - bounds["u_max"] if bounds["u_max"] is not None else 0.0,
            bounds["v_min"] - v if bounds["v_min"] is not None else 0.0,
            v - bounds["v_max"] if bounds["v_max"] is not None else 0.0,
        )
        if overshoot > CELL_OUTSIDE_MARGIN:
            entry["outside_cell"] = True
            continue
        for family, offset in (("u", u), ("v", v)):
            key, distance = _nearest_side(family, offset)
            if key is not None and distance is not None and (
                distance <= CELL_SNAP_BAND
            ):
                entry["cell_side"] = key
                break


def _write_dxf(path: Path, walls: list[dict], objects: list[dict]) -> None:
    """Minimal but valid DXF R12: walls on WALLS, footprints on OBJECTS,
    labels on TEXT. Opens in AutoCAD/LibreCAD/QCAD. Units are metres."""
    out = ["0", "SECTION", "2", "ENTITIES"]

    def line(x1, y1, x2, y2, layer):
        out.extend(
            ["0", "LINE", "8", layer,
             "10", f"{x1:.4f}", "20", f"{y1:.4f}", "30", "0.0",
             "11", f"{x2:.4f}", "21", f"{y2:.4f}", "31", "0.0"]
        )

    for wall in walls:
        line(*wall["start"], *wall["end"], "WALLS")
    for obj in objects:
        ring = obj["footprint"] + [obj["footprint"][0]]
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            line(x1, y1, x2, y2, "OBJECTS")
        cx, cy = obj["centroid_xy"]
        out.extend(
            ["0", "TEXT", "8", "TEXT",
             "10", f"{cx:.4f}", "20", f"{cy:.4f}", "30", "0.0",
             "40", "0.18", "1", f"{obj['label']} H={obj['height_m']:.2f}"]
        )
    out.extend(["0", "ENDSEC", "0", "EOF"])
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _clip_segment(start, end, min_x, min_y, max_x, max_y):
    """Liang-Barsky clip of a wall segment to the drawing frame."""
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x1 - min_x), (dx, max_x - x1),
        (-dy, y1 - min_y), (dy, max_y - y1),
    ):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return (
        (x1 + t0 * dx, y1 + t0 * dy),
        (x1 + t1 * dx, y1 + t1 * dy),
    )


def _render_plan(
    path: Path,
    walls: list[dict],
    objects: list[dict],
    run_id: str,
    off_plan: list[dict] | None = None,
    cell: dict | None = None,
) -> None:
    """A drawing, not a scatter plot: title block, coordinate grid, oriented
    object rectangles with height/tilt labels, camera-to-subject distance,
    dimensioned clearances, contact callouts (d=0.00), hatched walls, and an
    honest exclusion strip for entries the quality gates rejected."""
    from shapely.geometry import LineString
    from shapely.ops import nearest_points as _nearest

    W, H = 1600, 1240
    plot_w, plot_h = 1180, 1120
    left, top = 40, 60
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)

    xs = [0.0] + [x for o in objects for x, _ in o["footprint"]]
    ys = [0.0] + [y for o in objects for _, y in o["footprint"]]
    min_x, max_x = min(xs) - 1.2, max(xs) + 1.2
    min_y, max_y = min(ys) - 1.2, max(ys) + 1.2
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    min_x -= (span - (max_x - min_x)) / 2
    min_y -= (span - (max_y - min_y)) / 2
    max_x, max_y = min_x + span, min_y + span
    scale = min(plot_w, plot_h) / span

    def px(point):
        return (
            left + (point[0] - min_x) * scale,
            top + plot_h - (point[1] - min_y) * scale,
        )

    # sheet border and plot frame
    draw.rectangle([12, 12, W - 12, H - 12], outline="#111111", width=3)
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline="#555555", width=1)

    for gx in np.arange(np.ceil(min_x), np.floor(max_x) + 1):
        a, b = px((gx, min_y)), px((gx, max_y))
        draw.line([a, b], fill="#e8e8e8")
        draw.text((a[0] - 8, top + plot_h + 4), f"{gx:.0f}", fill="#9a9a9a")
    for gy in np.arange(np.ceil(min_y), np.floor(max_y) + 1):
        a, b = px((min_x, gy)), px((max_x, gy))
        draw.line([a, b], fill="#e8e8e8")
        draw.text((left - 26, a[1] - 6), f"{gy:.0f}", fill="#9a9a9a")

    # fitted cell rectangle: dashed, behind everything — the premise the
    # boundary structures were constrained against
    if cell is not None:
        theta_r = np.radians(cell["theta_deg"])
        cos_t, sin_t = float(np.cos(theta_r)), float(np.sin(theta_r))

        def _uv_xy(u, v):
            return (u * cos_t - v * sin_t, u * sin_t + v * cos_t)

        def _dashed(p1, p2):
            clipped = _clip_segment(p1, p2, min_x, min_y, max_x, max_y)
            if clipped is None:
                return
            a, b = np.asarray(clipped[0], float), np.asarray(clipped[1], float)
            length = float(np.hypot(*(b - a)))
            if length < 1e-6:
                return
            unit = (b - a) / length
            pos = 0.0
            while pos < length:
                seg_end = min(pos + 0.35, length)
                draw.line(
                    [px(a + unit * pos), px(a + unit * seg_end)],
                    fill="#b8860b", width=3,
                )
                pos = seg_end + 0.25

        sides = cell["sides"]
        axis_bounds = {
            "u": (sides["u_min"], sides["u_max"]),
            "v": (sides["v_min"], sides["v_max"]),
        }
        for key, side in sides.items():
            if side is None:
                continue
            family = key[0]
            other_low, other_high = axis_bounds["v" if family == "u" else "u"]
            lo = other_low["offset"] if other_low else side["extent"][0]
            hi = other_high["offset"] if other_high else side["extent"][1]
            if family == "u":
                _dashed(_uv_xy(side["offset"], lo), _uv_xy(side["offset"], hi))
            else:
                _dashed(_uv_xy(lo, side["offset"]), _uv_xy(hi, side["offset"]))
        if cell.get("size_m"):
            corner = px(tuple(cell["corners"][3]))
            draw.text(
                (
                    min(max(corner[0], left + 4), left + plot_w - 160),
                    min(max(corner[1] - 16, top + 4), top + plot_h - 16),
                ),
                f"CELL {cell['size_m'][0]:.2f} x {cell['size_m'][1]:.2f} m",
                fill="#b8860b",
            )

    # walls: thick line plus hatch ticks on the occupied side
    for wall in walls:
        clipped = _clip_segment(wall["start"], wall["end"], min_x, min_y, max_x, max_y)
        if clipped is None:
            continue
        start, end = clipped
        draw.line([px(start), px(end)], fill="#1a1a1a", width=9)
        direction = np.asarray(end) - np.asarray(start)
        length = float(np.hypot(*direction))
        if length < 1e-6:
            continue
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        for offset in np.arange(0.12, length, 0.32):
            base = np.asarray(start) + unit * offset
            draw.line([px(base), px(base + normal * 0.18)], fill="#7a7a7a", width=2)
        mid = np.asarray(start) + unit * (length / 2) + normal * 0.35
        draw.text(px(mid), f"WALL  {wall['length_m']:.2f} m  h {wall['height_m']:.2f}",
                  fill="#1a1a1a", anchor="mm")

    def dimension(a, b, text, colour, offset=0.0):
        """Extension lines + arrowed dimension line, CAD convention."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        direction = b - a
        length = float(np.hypot(*direction))
        if length < 1e-6:
            return
        unit = direction / length
        normal = np.array([-unit[1], unit[0]]) * offset
        pa, pb = px(a + normal), px(b + normal)
        if offset:
            draw.line([px(a), pa], fill=colour, width=1)
            draw.line([px(b), pb], fill=colour, width=1)
        draw.line([pa, pb], fill=colour, width=2)
        for point, sign in ((pa, 1), (pb, -1)):
            tip = np.asarray(point)
            back = tip + sign * np.array([unit[0], -unit[1]]) * 11
            side = np.array([-unit[1], -unit[0]]) * 4.5
            draw.polygon([tuple(tip), tuple(back + side), tuple(back - side)], fill=colour)
        mid = (a + b) / 2 + normal
        draw.text(px(mid), text, fill=colour, anchor="mm")

    palette = ["#c1121f", "#1d4ed8", "#047857", "#b45309", "#6d28d9",
               "#0e7490", "#9d174d", "#4d7c0f", "#7c2d12", "#334155"]
    legend = []
    tag_boxes: list[tuple[float, float]] = []
    for index, obj in enumerate(objects):
        colour = palette[index % len(palette)]
        # another cell's structure: keep it on the sheet (it exists) but
        # visually out of this cell's story
        if obj.get("outside_cell"):
            colour = "#b9b9b9"
        if obj.get("rect_snapped"):
            ring = [tuple(p) for p in obj["rect_snapped"]]
            ring.append(ring[0])
        else:
            box = Polygon(obj["footprint"]).minimum_rotated_rectangle
            ring = list(box.exterior.coords)
        draw.polygon([px(p) for p in ring], outline=colour, width=3)
        draw.line([px(p) for p in obj["footprint"] + [obj["footprint"][0]]],
                  fill=colour, width=1)
        cx, cy = obj["centroid_xy"]
        draw.text(px((cx, cy)), f"{index + 1}", fill=colour, anchor="mm")
        # measured state next to the symbol, like the probe plot the owner
        # signed off on: label, height, tilt when the fit produced one.
        # Tags nudge downward until they stop overlapping earlier ones.
        tag = f"{obj['label']}  H {obj['height_m']:.2f} m"
        if obj.get("tilt_deg") is not None:
            tag += f", tilt {obj['tilt_deg']:.0f}°"
        if obj.get("outside_cell"):
            tag += "  [outside cell]"
        anchor_pt = list(px((cx, cy + 0.35)))
        anchor_pt[1] -= 14
        while any(
            abs(anchor_pt[0] - x) < 110 and abs(anchor_pt[1] - y) < 16
            for x, y in tag_boxes
        ):
            anchor_pt[1] += 16
        tag_boxes.append(tuple(anchor_pt))
        draw.text(tuple(anchor_pt), tag, fill=colour, anchor="mm")
        # dimension the two sides of the oriented box
        for i in (0, 1):
            p0, p1 = np.asarray(ring[i]), np.asarray(ring[i + 1])
            side = float(np.hypot(*(p1 - p0)))
            if side * scale > 46:
                dimension(p0, p1, f"{side:.2f}", colour, offset=0.14)
        legend.append((index + 1, colour, obj))

    # camera -> subject distance, the headline number a reviewer asks first.
    # Subject = the worker when present, else the nearest tall object.
    subject = next(
        (o for o in objects if o["label"] == "worker"),
        min(
            (o for o in objects if o["height_m"] > 1.0),
            key=lambda o: o["camera_dist_m"],
            default=None,
        ),
    )
    if subject is not None:
        sx, sy = subject["centroid_xy"]
        dimension(
            (0.0, 0.0),
            (sx, sy),
            f"{subject['camera_dist_m']:.2f} m",
            "#c1121f",
        )

    # contact callouts: object sitting ON a line/marking/fence reads as
    # d = 0.00, which is exactly what a zone rule wants stated out loud.
    flat_labels = ("marking", "line", "fence")
    for a in objects:
        pa = Polygon(a["footprint"])
        for b in objects:
            if a is b or not any(k in b["label"] for k in flat_labels):
                continue
            pb = Polygon(b["footprint"])
            if pa.intersects(pb) or pa.distance(pb) < 0.05:
                cx, cy = a["centroid_xy"]
                point = px((cx, cy - 0.55))
                draw.text(
                    point,
                    f"{a['label']} on {b['label']} (d=0.00)",
                    fill="#6d28d9",
                    anchor="mm",
                )

    # clearance dimensions for the closest object pairs — what a rule reads
    pairs = []
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            pa, pb = Polygon(a["footprint"]), Polygon(b["footprint"])
            gap = pa.distance(pb)
            # Sub-decimetre gaps are parts of one thing (worker / jumpsuit /
            # hard hat); dimensioning them buries the drawing in arrows.
            if pa.intersects(pb) or gap < 0.15:
                continue
            pairs.append((gap, a, b, pa, pb))
    pairs.sort(key=lambda item: item[0])
    for gap, a, b, pa, pb in pairs[:3]:
        p1, p2 = _nearest(pa, pb)
        dimension(
            (p1.x, p1.y),
            (p2.x, p2.y),
            f"{a['label']}↔{b['label']}  {gap:.2f} m",
            "#047857",
        )
    camera = px((0.0, 0.0))
    draw.ellipse([camera[0] - 7, camera[1] - 7, camera[0] + 7, camera[1] + 7],
                 fill="#c1121f")
    draw.line([(camera[0] - 14, camera[1]), (camera[0] + 14, camera[1])],
              fill="#c1121f", width=1)
    draw.line([(camera[0], camera[1] - 14), (camera[0], camera[1] + 14)],
              fill="#c1121f", width=1)
    draw.text((camera[0] + 12, camera[1] + 8), "CAM (0,0)", fill="#c1121f")

    # legend + title block on the right rail
    rail = left + plot_w + 24
    draw.text((rail, top), "OBJECT SCHEDULE", fill="#111111")
    draw.line([(rail, top + 16), (W - 24, top + 16)], fill="#111111", width=2)
    row = top + 26
    for number, colour, obj in legend:
        draw.rectangle([rail, row + 3, rail + 9, row + 12], fill=colour)
        draw.text((rail + 16, row), f"{number}. {obj['label']}", fill="#111111")
        draw.text((rail + 16, row + 14),
                  f"H {obj['height_m']:.2f}  {obj['size_m']}  d {obj['camera_dist_m']:.2f}",
                  fill="#5b5b5b")
        row += 34
    block_top = H - 150
    draw.rectangle([rail, block_top, W - 24, H - 24], outline="#111111", width=2)
    draw.line([(rail, block_top + 26), (W - 24, block_top + 26)], fill="#111111")
    draw.text((rail + 8, block_top + 6), "MEASURED FLOOR PLAN", fill="#111111")
    for offset, line in enumerate([
        f"run      {run_id}",
        f"objects  {len(objects)}   walls {len(walls)}",
        "units    metres, floor frame",
        "origin   camera position",
        "source   photogrammetry, not a survey",
    ]):
        draw.text((rail + 8, block_top + 34 + offset * 17), line, fill="#333333")
    # honest exclusion strip: what the quality gates refused, and why —
    # stated on the sheet so an empty-looking area never reads as "checked
    # and clear".
    if off_plan:
        row = block_top - 20 - 15 * min(len(off_plan), 4)
        draw.text((rail, row - 16), "EXCLUDED BY QUALITY GATE", fill="#b91c1c")
        for entry in off_plan[:4]:
            draw.text(
                (rail, row),
                f"{entry['label']} @{entry['camera_dist_m']:.0f} m — "
                f"{entry['off_plan_reason']}",
                fill="#7f1d1d",
            )
            row += 15
    bar_m = 1.0
    bar = px((min_x + 0.4, min_y + 0.35))
    draw.line([bar, (bar[0] + bar_m * scale, bar[1])], fill="#111111", width=5)
    draw.text((bar[0], bar[1] + 8), "1 m", fill="#111111")
    image.save(path)


def _write_scene(path: Path, run_id: str, entries: list[dict]) -> None:
    """A SceneMap over the whole inventory, so compiled policies can be
    evaluated against every enumerated class rather than only the production
    vocabulary."""
    from ehs_spatial.contracts import Entity3D, SceneMap

    entities = []
    for index, entry in enumerate(entries, start=1):
        if entry["height_m"] <= 0:
            if not (_is_flat_zone(entry["label"]) and entry["height_m"] > -0.20):
                continue
            entry = {**entry, "height_m": 0.0}
        entities.append(
            Entity3D(
                entity_id=f"inv-{index:03d}",
                label=entry["label"],
                observation_ids=[f"inv-obs-{index:03d}"],
                centroid_xyz=(
                    float(entry["centroid_xy"][0]),
                    float(entry["centroid_xy"][1]),
                    float(entry["height_m"]) / 2,
                ),
                footprint_xy=[(float(x), float(y)) for x, y in entry["footprint"]],
                height_m=float(entry["height_m"]),
                evidence_frame_ids=[entry["frame"]],
                # Entity3D wants [0,180); a rounded 180.0 is the same axis as 0
                orientation_deg=None
                if entry.get("orientation_deg") is None
                else float(entry["orientation_deg"]) % 180.0,
                tilt_deg=entry.get("tilt_deg"),
            )
        )
    scene = SceneMap(
        run_id=f"{run_id}-inventory",
        floor_plane=(0.0, 0.0, 1.0, 0.0),
        scale_source="camera_height",
        scale_factor=1.0,
        fence_polygon=[],
        entities=entities,
        facts=[],
        warnings=["inventory scene: exploration vocabulary, not the production path"],
    )
    path.write_text(scene.model_dump_json(indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--camera-height", type=float, default=1.5)
    args = parser.parse_args(argv)

    run = Path("runs") / args.run
    frames = _frames(run)
    observations = [
        Observation2D.model_validate(item)
        for item in json.loads((run / "observations.json").read_text())
    ]
    scale_override = None
    scene_path = run / "scene.json"
    if scene_path.exists():
        try:
            scale_override = json.loads(scene_path.read_text()).get(
                "scale_factor"
            )
        except ValueError:
            scale_override = None
    transform = _build_geometry(
        frames,
        observations,
        args.camera_height,
        scale_factor_override=scale_override,
    ).transform
    if transform is None:
        print("run has no floor transform", file=sys.stderr)
        return 1

    phrases = _enumerate_objects(run, frames[0], live=args.live)
    if phrases is None:
        print("needs 1 VLM call + up to 26 SAM calls; pass --live", file=sys.stderr)
        return 2
    print(f"VLM listed {len(phrases)} objects: {', '.join(phrases)}")

    entries: list[dict] = []
    cleaned_masks: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    for frame in frames:
        points3d = np.load(frame.pts3d_path)
        valid = np.load(frame.valid_mask_path).astype(bool)
        finite = valid & np.isfinite(points3d).all(axis=2)
        height, width = valid.shape
        moge_maps = _moge3_maps(run, frame)
        for phrase in phrases:
            response = _segment(run, frame, phrase, live=args.live)
            if response is None:
                continue
            rles = response.get("rle") or []
            if isinstance(rles, str):
                rles = [rles]
            scores = response.get("scores") or [1.0] * len(rles)
            decoded = [
                decode_coco_rle(rle, height=height, width=width).astype(bool)
                for rle in rles
            ]
            if any(k in phrase for k in CONTACT_FAMILY):
                groups = _merge_vertical_stacks(decoded, height)
            else:
                groups = [(i, [i]) for i in range(len(decoded))]
            for index, members in groups:
                mask = decoded[index]
                for member in members:
                    if member != index:
                        mask = mask | decoded[member]
                if int(mask.sum()) < MIN_MASK_PIXELS:
                    continue
                # MoGe-3 normal split: pixels seen THROUGH a clear panel
                # carry the background's normals; keep the panel's dominant
                # vertical-plane cluster, and persist the cleaned mask so
                # every downstream consumer (report photo-pick included)
                # stops highlighting what is behind the glass.
                split = None
                if moge_maps is not None and any(
                    k in phrase for k in CONTACT_FAMILY
                ):
                    split = _normal_split(mask, *moge_maps)
                if split is not None:
                    mask = split
                    # the cleaned mask is the frame skeleton; the panel FACE
                    # is the quad the frame encloses. Display = convex hull
                    # of the skeleton's significant components, so clicking
                    # the glass selects the panel and the highlight is the
                    # framed quad — not the silhouette of whatever stands
                    # behind the glass. Measurement keeps the skeleton.
                    cleaned_masks.setdefault(
                        (frame.frame_id, phrase), {}
                    )[index] = _hull_fill(split)
                # De-smear (research approach 3): flying pixels concentrate
                # on the silhouette boundary — erode 2 px unless the object
                # is thinner than the erosion; then trim the along-ray depth
                # tail to its inter-percentile core.
                from scipy import ndimage

                core = ndimage.binary_erosion(mask, iterations=2)
                if core.sum() >= 0.3 * mask.sum():
                    mask = core
                selected = mask & finite
                depth_map = points3d[..., 2]
                depths = depth_map[selected]
                if len(depths) >= 50:
                    low, high = np.percentile(depths, 15), np.percentile(depths, 85)
                    margin = 0.25 * (high - low) + 0.05
                    selected &= (depth_map >= low - margin) & (
                        depth_map <= high + margin
                    )
                cloud = _clean(transform.apply(points3d[selected]))
                if cloud is None:
                    continue
                hull = MultiPoint([(x, y) for x, y in cloud[:, :2]]).convex_hull
                if not isinstance(hull, Polygon) or hull.area < 0.01:
                    continue
                top = float(np.quantile(cloud[:, 2], 0.95))
                orientation, tilt, _ = _spatial_state(cloud, max(top, 0.0))
                box = hull.minimum_rotated_rectangle
                corners = list(box.exterior.coords)[:4]
                sides = sorted(
                    np.hypot(
                        corners[i][0] - corners[i - 1][0],
                        corners[i][1] - corners[i - 1][1],
                    )
                    for i in range(1, 3)
                )
                entries.append(
                    {
                        "label": phrase,
                        "instance": index,
                        **(
                            # a normal-split mask is written back to the
                            # cache at the primary index already unioned;
                            # members would re-add the uncleaned bands
                            {"merged_instances": members}
                            if len(members) > 1 and split is None
                            else {}
                        ),
                        **({"normal_split": True} if split is not None else {}),
                        "image_bbox": [
                            int(v)
                            for pair in (
                                (np.nonzero(mask)[1].min(), np.nonzero(mask)[0].min()),
                                (np.nonzero(mask)[1].max(), np.nonzero(mask)[0].max()),
                            )
                            for v in pair
                        ],
                        "frame": frame.frame_id,
                        "score": round(
                            float(scores[index]) if index < len(scores) else 1.0, 3
                        ),
                        "points": int(len(cloud)),
                        "height_m": round(top, 2),
                        "size_m": f"{sides[1]:.2f}x{sides[0]:.2f}",
                        "footprint_area_m2": round(float(hull.area), 2),
                        "centroid_xy": [
                            round(float(hull.centroid.x), 2),
                            round(float(hull.centroid.y), 2),
                        ],
                        "camera_dist_m": round(
                            float(Point(0.0, 0.0).distance(hull)), 2
                        ),
                        "orientation_deg": None
                        if orientation is None
                        else round(orientation, 1),
                        "tilt_deg": None if tilt is None else round(tilt, 1),
                        "footprint": [
                            (round(float(x), 3), round(float(y), 3))
                            for x, y in list(hull.exterior.coords)[:-1]
                        ],
                    }
                )

    for (frame_id, phrase), masks in cleaned_masks.items():
        slug = re.sub(r"[^a-z0-9]+", "_", phrase).strip("_")
        cache = run / "inventory" / "sam" / f"{frame_id}__{slug}.json"
        response = json.loads(cache.read_text())
        rles = response.get("rle") or []
        if isinstance(rles, str):
            rles = [rles]
        for idx, mask in masks.items():
            if idx < len(rles):
                rles[idx] = encode_coco_rle(mask)
        response["rle"] = rles
        cache.write_text(json.dumps(response) + "\n")

    unresolved = _reconcile_enumeration(
        run, frames[0], entries, phrases, live=args.live
    )
    _ingest_refinements(run, frames[0], transform, moge_maps, entries)
    # the guarantee, settled AFTER measurement: every enumerated phrase
    # either has a measured instance or an explicit unresolved record
    covered_after = {e["label"] for e in entries}
    noted = {u["phrase"] for u in unresolved}
    for phrase in phrases:
        if phrase not in covered_after and phrase not in noted:
            unresolved.append(
                {
                    "phrase": phrase,
                    "reason": "localized but failed measurement evidence "
                    "gates (too few valid 3D points / degenerate footprint)",
                }
            )
    (run / "inventory").mkdir(exist_ok=True)
    (run / "inventory" / "unresolved.json").write_text(
        json.dumps(unresolved, ensure_ascii=False, indent=2) + "\n"
    )

    # Every reliable instance goes on the sheet (the interactive report
    # plan draws instances; the static CAD sheet must match it 1:1).
    plan_entries: list[dict] = []
    for entry in entries:
        if entry["label"] in STRUCTURE_LABELS:
            continue
        if entry["camera_dist_m"] > PLAN_MAX_RANGE_M:
            entry["off_plan_reason"] = "beyond reliable single-view range"
            continue
        if entry["height_m"] <= 0.0:
            # Painted zones/markings/mats LIVE at height 0 — mono noise puts
            # them a few cm negative, which is a correct measurement of a
            # flat object, not depth collapse. Keep them, clamped to 0.
            if _is_flat_zone(entry["label"]) and entry["height_m"] > -0.20:
                entry["height_m"] = 0.0
            else:
                entry["off_plan_reason"] = "non-positive height (depth collapse)"
                continue
        plan_entries.append(entry)
    plan_objects = sorted(plan_entries, key=lambda e: -e["footprint_area_m2"])
    off_plan = [e for e in entries if e.get("off_plan_reason")]

    scene_cloud = []
    for frame in frames:
        points3d = np.load(frame.pts3d_path)
        valid = np.load(frame.valid_mask_path).astype(bool)
        finite = valid & np.isfinite(points3d).all(axis=2)
        scene_cloud.append(transform.apply(points3d[finite]))
    cloud = np.vstack(scene_cloud)
    walls = [
        w
        for w in _fit_walls(cloud)
        if min(
            float(np.hypot(*w["start"])), float(np.hypot(*w["end"]))
        )
        < PLAN_MAX_RANGE_M
    ]

    theta = _manhattan_theta(walls)
    for entry in entries:
        # Thin wall-following structures and painted zones have no honest
        # oblique reading: a "tighter" free rectangle on them IS the smear.
        oblique_ok = not (
            any(k in entry["label"] for k in CONTACT_FAMILY)
            or _is_flat_zone(entry["label"])
        )
        snapped = _snap_rect(entry["footprint"], theta, allow_free=oblique_ok)
        if snapped is not None:
            entry["rect_snapped"] = snapped

    # Contact-edge pass for floor-standing thin structures: re-decode the
    # RAW mask (the de-smear erosion moves the bottom edge) and project its
    # ground-contact edge through the camera onto the floor plane.
    for frame in frames:
        points3d = np.load(frame.pts3d_path)
        valid = np.load(frame.valid_mask_path).astype(bool)
        height, width = valid.shape
        for entry in entries:
            if entry["frame"] != frame.frame_id:
                continue
            if not any(k in entry["label"] for k in CONTACT_FAMILY):
                continue
            if entry.get("refine_slug"):
                raw_mask = _refine_mask(run, entry["refine_slug"], height, width)
                if raw_mask is None:
                    continue
            else:
                slug = re.sub(r"[^a-z0-9]+", "_", entry["label"]).strip("_")
                cache = (
                    run / "inventory" / "sam" / f"{frame.frame_id}__{slug}.json"
                )
                if not cache.exists():
                    continue
                response = json.loads(cache.read_text())
                rles = response.get("rle") or []
                if isinstance(rles, str):
                    rles = [rles]
                if entry["instance"] >= len(rles):
                    continue
                raw_mask = np.zeros((height, width), bool)
                for member in entry.get("merged_instances") or [entry["instance"]]:
                    raw_mask |= decode_coco_rle(
                        rles[member], height=height, width=width
                    ).astype(bool)
            rect = _contact_edge_rect(raw_mask, points3d, frame, transform, theta)
            if rect is not None:
                entry["rect_snapped"] = rect
                entry["footprint_method"] = "contact-edge"
                continue
            # Occluded base: no honest contact edge. The hull of a
            # see-through structure mixes the frame with background seen
            # through it; keep the nearest depth cluster instead (same
            # rule as refine.measure) when that visibly deflates the hull.
            finite = (
                valid
                & np.isfinite(points3d).all(axis=2)
                & (np.abs(points3d).sum(axis=2) > 1e-6)
            )
            selected = raw_mask & finite
            depth_map = points3d[..., 2]
            if selected.sum() < 50:
                continue
            near = np.percentile(depth_map[selected], 10)
            # machinery often hugs a guard rail from behind; the keep-band
            # must stay tighter than that gap, scaled with range
            selected &= depth_map <= near + max(0.25, 0.12 * near)
            cloud = _clean(transform.apply(points3d[selected]))
            if cloud is None:
                continue
            hull = MultiPoint([(x, y) for x, y in cloud[:, :2]]).convex_hull
            if (
                not isinstance(hull, Polygon)
                or hull.area < 0.01
                or hull.area > 0.8 * entry["footprint_area_m2"]
            ):
                continue
            top = float(np.quantile(cloud[:, 2], 0.95))
            if top > 0.0:
                entry["height_m"] = round(top, 2)
            entry["footprint"] = [
                (round(float(x), 3), round(float(y), 3))
                for x, y in list(hull.exterior.coords)[:-1]
            ]
            entry["footprint_area_m2"] = round(float(hull.area), 2)
            entry["centroid_xy"] = [
                round(float(hull.centroid.x), 2),
                round(float(hull.centroid.y), 2),
            ]
            entry["camera_dist_m"] = round(float(Point(0.0, 0.0).distance(hull)), 2)
            entry["footprint_method"] = "near-cluster"
            snapped = _snap_rect(entry["footprint"], theta, allow_free=False)
            if snapped is not None:
                entry["rect_snapped"] = snapped

    theta = _align_guard_lines(entries, theta, walls, run, frames[0], transform)
    _enforce_row_order(entries, theta)
    cell = _fit_cell_rectangle(walls, entries, theta)
    if cell is not None:
        _apply_cell_rectangle(cell, walls, entries, theta)
    _apply_payload_entities(run, entries)

    out_dir = run / "inventory"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inventory.json").write_text(
        json.dumps(
            {
                "phrases": phrases,
                "objects": entries,
                "walls": walls,
                "manhattan_theta_deg": None
                if theta is None
                else round(float(np.degrees(theta)), 1),
                "cell_rect": cell,
            },
            indent=2,
        )
        + "\n"
    )
    _render_plan(
        out_dir / "floor_plan.png", walls, plan_objects, args.run,
        off_plan=off_plan, cell=cell,
    )
    _write_dxf(out_dir / "floor_plan.dxf", walls, plan_objects)
    _write_scene(out_dir / "scene.json", args.run, entries)

    print(f"\n{len(entries)} instances | {len(plan_objects)} on the plan | "
          f"{len(off_plan)} rejected | {len(walls)} wall plane(s)")
    for obj in plan_objects:
        print(
            f"  {obj['label']:<22} H={obj['height_m']:>5.2f}m  "
            f"{obj['size_m']:>12}  d_cam={obj['camera_dist_m']:>5.2f}m  "
            f"pts={obj['points']}"
        )
    for wall in walls:
        print(f"  WALL len={wall['length_m']}m h={wall['height_m']}m pts={wall['points']}")
    for entry in off_plan[:6]:
        print(f"  [off-plan] {entry['label']}: {entry['off_plan_reason']}")
    print("\nwrote", out_dir / "floor_plan.png", "and floor_plan.dxf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
