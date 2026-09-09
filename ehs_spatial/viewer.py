"""Visual evidence artifacts: mask overlays and the interactive 3D viewer.

Every verdict ships with evidence a reviewer can check without trusting the
numbers: per-frame PNGs of the original photos with each segmented object
tinted and labelled, and a self-contained WebGL `viewer.html` (no network,
no CDN, no build step) where every object is selectable with its measured
height/size/distance/tilt on screen.

Both are derived views over the run's cached evidence (reconstructed
geometry + segmentation masks); they never call a provider.
"""

import base64
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint, Point, Polygon

from .contracts import GeometryFrame, Observation2D
from .geometry import _build_geometry, _spatial_state
from .providers.map_anything import decode_encoded_array, input_mask_to_canonical
from .providers.sam3 import decode_coco_rle

# The report loads viewer.html by URL (not inline), so the budget is a
# render budget, not an embed budget.
MAX_POINTS = 600_000
MIN_MASK_PIXELS = 50
MIN_OBJECT_POINTS = 40
# Orbit pivot: the floor point this far ahead of the camera (PRODUCT_PLAN §2.2).
LOOK_AHEAD_M = 2.5
# Labels kept as scene context, never as selectable objects.
DEFAULT_EXCLUDE = frozenset({"floor", "ceiling", "wall", "ground", "roof"})
_OVERLAY_ALPHA = 0.45
PALETTE = [
    (229, 25, 75), (60, 180, 75), (67, 99, 216), (245, 130, 49),
    (145, 30, 180), (66, 212, 244), (240, 50, 230), (191, 239, 69),
    (250, 190, 190), (0, 128, 128), (230, 190, 255), (154, 99, 36),
    (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0),
    (255, 216, 177), (0, 0, 117), (169, 169, 169), (255, 225, 25),
]


def inventory_plan_objects(inventory: dict) -> list[dict]:
    """The photo, 3D, report and CAD share this inventory-index selection set."""
    structures = DEFAULT_EXCLUDE | {"window", "factory floor", "concrete floor", "ceiling structure"}
    return [dict(obj, inv=index) for index, obj in enumerate(inventory.get("objects", []))
            if obj["label"] not in structures and not obj.get("off_plan_reason")]


def _tints(observations: list[Observation2D]) -> dict[str, tuple[int, int, int]]:
    labels = sorted({observation.label for observation in observations})
    return {
        label: PALETTE[index % len(PALETTE)]
        for index, label in enumerate(labels)
    }


def render_frame_overlays(
    frames: list[GeometryFrame],
    observations: list[Observation2D],
    out_dir: str | Path,
) -> list[Path]:
    """One PNG per frame: the original photo with every segmented mask
    tinted per label and captioned with label + score.

    The 2D half of the reviewer's evidence: it shows exactly which pixels
    each verdict was measured from. Observations without a readable local
    mask are skipped, not fatal — a run with partial masks still gets its
    photos back annotated with whatever it has.
    """
    out_dir = Path(out_dir)
    tints = _tints(observations)
    written: list[Path] = []
    for frame in frames:
        with Image.open(frame.canonical_image_path) as image:
            base = np.asarray(image.convert("RGB")).astype(np.float32)
        captions: list[tuple[int, int, str]] = []
        for observation in observations:
            if observation.frame_id != frame.frame_id or not observation.mask_path:
                continue
            try:
                with Image.open(observation.mask_path) as mask_image:
                    mask = np.asarray(mask_image).astype(bool)
            except OSError:
                continue
            if mask.shape != base.shape[:2] or not mask.any():
                continue
            tint = np.asarray(tints[observation.label], dtype=np.float32)
            base[mask] = (1.0 - _OVERLAY_ALPHA) * base[mask] + _OVERLAY_ALPHA * tint
            rows, cols = np.nonzero(mask)
            captions.append(
                (
                    int(cols.min()),
                    int(rows.min()),
                    f"{observation.label} {observation.score:.2f}",
                )
            )
        rendered = Image.fromarray(base.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(rendered)
        for x, y, text in captions:
            # Clamp inside the frame so edge-hugging masks keep readable text.
            x = max(0, min(x, rendered.width - int(draw.textlength(text)) - 2))
            anchor = (x, max(0, y - 13))
            draw.rectangle(draw.textbbox(anchor, text), fill=(0, 0, 0))
            draw.text(anchor, text, fill=(255, 255, 255))
        out_dir.mkdir(parents=True, exist_ok=True)
        destination = out_dir / f"{frame.frame_id}_overlay.png"
        rendered.save(destination, format="PNG")
        written.append(destination)
    return written


def _frames(run: Path) -> list[GeometryFrame]:
    out = []
    frames_dir = run / "geometry" / "frames"
    if not frames_dir.is_dir():
        return out
    for frame_dir in sorted(frames_dir.iterdir()):
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


def camera_anchor(
    camera_to_world,
    transform,
    frame_id: str = "",
    look_ahead_m: float = LOOK_AHEAD_M,
) -> dict:
    """One frame's camera pose in the viewer's floor frame (+Z up, z=0 on
    the floor) plus its orbit pivot.

    The pivot is the floor point `look_ahead_m` ahead of the camera along
    its horizontal heading — never the cloud centroid, which the far
    background drags away from what the photo was actually of.
    """
    c2w = np.asarray(camera_to_world, dtype=float)
    position = transform.apply(c2w[:3, 3][None])[0]
    # OpenCV columns: +X right, +Y down, +Z forward; the floor rotation is a
    # pure rotation so directions need no scale or origin shift.
    axes = transform.rotation @ c2w[:3, :3]
    right, up, forward = axes[:, 0], -axes[:, 1], axes[:, 2]
    heading = np.array([forward[0], forward[1], 0.0])
    if np.linalg.norm(heading) < 1e-6:
        # Camera pointing straight down: image-up is the only heading left.
        heading = np.array([up[0], up[1], 0.0])
    heading /= np.linalg.norm(heading) or 1.0
    pivot = np.array([position[0], position[1], 0.0]) + look_ahead_m * heading

    def rounded(vector) -> list[float]:
        return [round(float(v), 4) for v in vector]

    return {
        "frame_id": frame_id,
        "position": rounded(position),
        "forward": rounded(forward),
        "up": rounded(up),
        "right": rounded(right),
        "pivot": rounded(pivot),
    }


def _label_from_cache(path: Path) -> str:
    stem = path.stem
    return stem.split("__", 1)[1].replace("_", " ") if "__" in stem else stem


def _slug(label: str) -> str:
    """The inventory's cache-name rule (scripts/scene_inventory.py)."""
    return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")


def inventory_lookup(
    objects: list[dict],
) -> tuple[dict[tuple[str, str, int], int], dict[tuple[str, str], int]]:
    """Two maps from a mask's provenance to its inventory index (`inv`).

    by_instance[(frame, label_slug, rle_index)] for SAM-cache objects
    (rle index = `instance`, or any member of `merged_instances`);
    by_slug[(frame, slug)] for refinements (`refine_slug`, or `_slug` on a
    footprint_method == 'refine' entry).
    """
    by_instance: dict[tuple[str, str, int], int] = {}
    by_slug: dict[tuple[str, str], int] = {}
    for index, item in enumerate(objects):
        frame_id = str(item.get("frame", "frame_0001"))
        if item.get("refine_slug"):
            by_slug.setdefault((frame_id, str(item["refine_slug"])), index)
            continue
        if item.get("footprint_method") == "refine" and item.get("_slug"):
            by_slug.setdefault((frame_id, str(item["_slug"])), index)
            continue
        if item.get("instance") is None:
            continue
        slug = _slug(item.get("label", ""))
        for member in item.get("merged_instances") or [item["instance"]]:
            by_instance.setdefault((frame_id, slug, int(member)), index)
    return by_instance, by_slug


def _masks_for_frame(
    run: Path,
    frame: GeometryFrame,
    shape: tuple[int, int],
    observations: list[Observation2D],
) -> list[tuple[str, np.ndarray, int | None]]:
    """Per-INSTANCE (display label, mask, inv) for this frame, in stable
    discovery order.

    One entry per SAM RLE (instance), not one union per label — the floor
    plan draws instances, and the 3D viewer must bind one-to-one with it.
    `inv` is the object's index in inventory/inventory.json["objects"]
    (None when the mask has no inventory counterpart). Near-duplicate masks
    across sources (observation masks repeating the inventory's SAM
    instances) are dropped by IoU; the survivor adopts the duplicate's inv
    when it had none.
    """
    height, width = shape
    by_label: dict[str, list[list]] = {}  # label -> [[mask, inv], ...]

    def add(label: str, mask: np.ndarray, inv: int | None = None) -> None:
        if strict and inv is None:
            return
        if int(mask.sum()) < (1 if inv is not None else MIN_MASK_PIXELS):
            return
        peers = by_label.setdefault(label, [])
        area = mask.sum()
        for peer in peers:
            if inv is not None and peer[1] == inv:
                peer[0] |= mask  # inventory explicitly merged these SAM instances
                return
            inter = int((peer[0] & mask).sum())
            union = int(peer[0].sum()) + int(area) - inter
            if union and inter / union > 0.85 and (peer[1] is None or inv is None):
                if peer[1] is None:
                    peer[1] = inv
                return  # same physical instance seen through another source
        peers.append([mask, inv])

    inventory_path = run / "inventory" / "inventory.json"
    inventory_objects: list[dict] = []
    if inventory_path.exists():
        inventory_objects = json.loads(inventory_path.read_text())["objects"]
    by_instance, by_slug = inventory_lookup(inventory_objects)

    def derived_mask(slug: str, inv: int | None) -> np.ndarray | None:
        expected = inventory_objects[inv].get("refine_mask_sha256") if inv is not None else None
        if expected is None:
            return None
        path = run / "inventory" / "refinement_masks" / f"{frame.frame_id}__{slug}.npy"
        provenance = json.loads(path.with_suffix(".json").read_text())
        source = run / "refinements" / f"{slug}.json"
        if (provenance.get("frame_id") != frame.frame_id
                or provenance.get("coordinate_space") != "canonical"
                or provenance.get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest()
                or provenance.get("mask_sha256") != expected
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError(f"{path}: refinement mask does not match accepted inventory/source")
        mask = np.load(path, allow_pickle=False)
        if mask.shape != shape or mask.dtype != np.bool_:
            raise ValueError(f"{path}: derived refinement mask must match canonical shape {shape}")
        return mask
    # Once the inventory exists it IS the instance set (what the plan, the
    # photo pick and the report link by `inv`): SAM instances its gates
    # rejected and the pre-inventory observation layer stay out of the 3D.
    # Runs without an inventory (fresh pipeline pass) show every mask.
    strict = inventory_path.exists()

    sources = ("inventory/sam",) if strict else ("inventory/sam", "semantic_layers/sam")
    for source in sources:
        directory = run / source
        if not directory.is_dir():
            continue
        for cache in sorted(directory.glob(f"{frame.frame_id}__*.json")):
            if "__p_" in cache.stem:
                continue  # per-prompt intermediates, already unioned into the primary
            response = json.loads(cache.read_text())
            rles = response.get("rle") or []
            if isinstance(rles, str):
                rles = [rles]
            slug = cache.stem.split("__", 1)[1]
            for k, rle in enumerate(rles):
                inv = (
                    by_instance.get((frame.frame_id, slug, k))
                    if source == "inventory/sam"
                    else None
                )
                if strict and inv is None:
                    continue
                add(
                    _label_from_cache(cache),
                    decode_coco_rle(rle, height=height, width=width).astype(bool),
                    inv,
                )

    # Human box-prompt refinements (scripts/refine_region.py) are reviewer
    # ground truth for classes the text prompts miss — include them. They
    # are measured on the first frame's full-res input (refine.py), and only
    # the applied ones carry footprint_xy: the detection layer's own log
    # entries (source=detection / enumeration-reconcile) are superseded by
    # the inventory's refine_slug objects below.
    refine_log = run / "refinements.json"
    if refine_log.exists() and frame.frame_id == "frame_0001":
        for item in json.loads(refine_log.read_text()):
            if "footprint_xy" not in item:
                continue
            slug = (
                f"{item['label'].replace(' ', '_')}_"
                + "_".join(str(b) for b in item["box"])
            )
            mask = derived_mask(slug, by_slug.get((frame.frame_id, slug)))
            if mask is not None:
                add(item["label"], mask, by_slug.get((frame.frame_id, slug)))
                continue
            cache = run / "refinements" / f"{slug}.json"
            if not cache.is_file():
                continue
            response = json.loads(cache.read_text())
            rles = response.get("rle") or []
            if isinstance(rles, str):
                rles = [rles]
            scores = response.get("scores") or [1.0] * len(rles)
            if rles:
                best = int(np.argmax(scores))
                # Decode the native mask before the provider's exact image transform.
                # Box-prompt responses omit width/height — read the input.
                native_h = response.get("height")
                native_w = response.get("width")
                if not native_h or not native_w:
                    inputs = sorted((run / "input").glob("image_*"))
                    with Image.open(inputs[0]) as native:
                        native_w, native_h = native.size
                native_h, native_w = int(native_h), int(native_w)
                mask = decode_coco_rle(
                    rles[best], height=native_h, width=native_w
                ).astype(np.uint8)
                mask = input_mask_to_canonical(mask, run, frame.frame_id, shape)
                add(item["label"], mask, by_slug.get((frame.frame_id, slug)))

    # Detection-layer instances synced into the inventory (taxonomy items
    # located by the VLM and segmented by SAM box prompts) live as
    # refinements/<slug>.json without a refinements.json entry — the glass
    # guard, deflectors, e-stops. One pipeline: the viewer shows them too.
    if inventory_objects:
        inputs = sorted((run / "input").glob("image_*"))
        for inv, item in enumerate(inventory_objects):
            slug = item.get("refine_slug")
            if not slug or item.get("frame", "frame_0001") != frame.frame_id:
                continue
            mask = derived_mask(slug, inv)
            if mask is not None:
                add(str(item.get("label", slug)), mask, inv)
                continue
            cache = run / "refinements" / f"{slug}.json"
            if not cache.is_file():
                continue
            response = json.loads(cache.read_text())
            rles = response.get("rle") or []
            if isinstance(rles, str):
                rles = [rles]
            if not rles:
                continue
            scores = response.get("scores") or [1.0] * len(rles)
            best = int(np.argmax(scores))
            native_h = response.get("height")
            native_w = response.get("width")
            if not native_h or not native_w:
                with Image.open(inputs[int(frame.frame_id.rsplit("_", 1)[1]) - 1]) as native:
                    native_w, native_h = native.size
            mask = decode_coco_rle(
                rles[best], height=int(native_h), width=int(native_w)
            ).astype(np.uint8)
            mask = input_mask_to_canonical(mask, run, frame.frame_id, shape)
            add(str(item.get("label", slug)), mask, inv)

    for observation in [] if strict else observations:
        if observation.frame_id != frame.frame_id or not observation.mask_path:
            continue
        with Image.open(observation.mask_path) as image:
            add(observation.label, np.asarray(image).astype(bool))

    flat: list[tuple[str, np.ndarray, int | None]] = []
    for label, peers in by_label.items():
        for position, (mask, inv) in enumerate(peers):
            display = f"{label} #{position + 1}" if len(peers) > 1 else label
            flat.append((display, mask, inv))
    return flat


def build_viewer_html(
    run_dir: str | Path,
    *,
    frames: list[GeometryFrame] | None = None,
    observations: list[Observation2D] | None = None,
    camera_height_m: float | None = 1.5,
    scale_factor_override: float | None = None,
    exclude: frozenset[str] | set[str] = DEFAULT_EXCLUDE,
    out_path: str | Path | None = None,
    max_points: int | None = None,
) -> dict:
    """Write the run's self-contained interactive viewer HTML.

    Frames/observations default to what the run directory holds so the CLI
    can rebuild any cached run; the pipeline passes its in-memory copies.
    Raises ValueError when the run has no floor transform — the caller
    decides whether that is fatal (CLI) or a warning (pipeline).
    """
    run = Path(run_dir)
    inventory_path = run / "inventory" / "inventory.json"
    inventory_objects = json.loads(inventory_path.read_text())["objects"] if inventory_path.exists() else []
    interactive_inv = [o["inv"] for o in inventory_plan_objects({"objects":inventory_objects})]
    if frames is None:
        frames = _frames(run)
    if observations is None:
        observations = [
            Observation2D.model_validate(item)
            for item in json.loads((run / "observations.json").read_text())
        ]
    # Mirror the assessed scene's scale: without this, a run whose scale
    # came from the auto anchor (or any non-1.5 m camera) would rebuild
    # geometry under a fabricated height and could fail its MAD gate.
    if scale_factor_override is None:
        scene_path = run / "scene.json"
        if scene_path.exists():
            scale_factor_override = json.loads(
                scene_path.read_text(encoding="utf-8")
            ).get("scale_factor")
    transform = _build_geometry(
        frames,
        observations,
        camera_height_m,
        scale_factor_override=scale_factor_override,
    ).transform
    if transform is None:
        raise ValueError("run has no floor transform")
    anchors = [
        camera_anchor(frame.camera_to_world, transform, frame.frame_id)
        for frame in frames
    ]

    # One viewer object per (frame, instance), like the inventory: index in
    # `entries` + 1 is the object id. A display label repeats across frames
    # ("bollard #1" in every frame is a different bollard), so each object
    # also carries its frame.
    entries: list[tuple[str, int | None, str]] = []
    evidence = [{"inv":i, "label":o["label"], "frame":o.get("frame", "frame_0001"),
                 "mask_pixels":0, "valid_points":0, "owned_points":0, "bounded_points":0}
                for i,o in enumerate(inventory_objects)]
    chunks_xyz, chunks_rgb, chunks_id = [], [], []
    for frame in frames:
        points3d = np.load(frame.pts3d_path)
        valid = np.load(frame.valid_mask_path).astype(bool)
        finite = valid & np.isfinite(points3d).all(axis=2) & (np.abs(points3d).sum(axis=2) > 1e-6)
        provider_path = run / "geometry" / "provider" / f"{frame.frame_id}.json"
        if provider_path.is_file():
            alpha_payload = json.loads(provider_path.read_text()).get("alpha_mask")
            if alpha_payload is not None:
                alpha = decode_encoded_array(alpha_payload)
                if alpha.shape != valid.shape or not np.isin(alpha, [0, 1]).all():
                    raise ValueError(f"{provider_path}: alpha_mask must match the canonical binary grid")
                finite &= alpha.astype(bool)
        with Image.open(frame.canonical_image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        if rgb.shape[:2] != valid.shape:
            rgb = np.asarray(
                Image.fromarray(rgb).resize((valid.shape[1], valid.shape[0]))
            )
        ids = np.zeros(valid.shape, dtype=np.uint16)
        masks = _masks_for_frame(run, frame, valid.shape, observations)
        # Smaller masks paint last so a specific object wins over the big
        # container it sits inside.
        for label, mask, inv in sorted(masks, key=lambda m: -int(m[1].sum())):
            if inv is not None:
                evidence[inv]["mask_pixels"] += int(mask.sum())
                evidence[inv]["valid_points"] += int((mask & finite).sum())
            base_label = inventory_objects[inv]["label"] if inv is not None else label.rsplit(" #", 1)[0]
            if base_label in exclude or (inv is not None and inv not in interactive_inv):
                continue
            entries.append((label, inv, frame.frame_id))
            ids[mask] = len(entries)
        chunks_xyz.append(transform.apply(points3d[finite]))
        chunks_rgb.append(rgb[finite])
        chunks_id.append(ids[finite])

    xyz = np.vstack(chunks_xyz)
    rgb = np.vstack(chunks_rgb)
    ids = np.concatenate(chunks_id)
    owned_counts = np.bincount(ids, minlength=len(entries) + 1)

    keep = np.isfinite(xyz).all(axis=1) & (np.abs(xyz) < 60).all(axis=1)
    xyz, rgb, ids = xyz[keep], rgb[keep], ids[keep]
    bounded_counts = np.bincount(ids, minlength=len(entries) + 1)
    for index, (_, inv, _) in enumerate(entries, start=1):
        if inv is not None:
            evidence[inv]["owned_points"] += int(owned_counts[index])
            evidence[inv]["bounded_points"] += int(bounded_counts[index])
    point_budget = max_points or MAX_POINTS
    if len(xyz) > point_budget:
        rng = np.random.default_rng(0)
        # Prefer object points; thin the unclassified scene first, and only
        # when objects alone blow the budget (compact embeds) thin them too.
        object_index = np.flatnonzero(ids > 0)
        scene_index = np.flatnonzero(ids == 0)
        if len(object_index) > point_budget * 0.8:
            # Stratified: every object keeps at least MIN_KEEP points so
            # small objects (e-stops, fence segments) survive compact embeds
            # instead of falling under MIN_OBJECT_POINTS and vanishing.
            MIN_KEEP = 160
            per_object = [np.flatnonzero(ids == i) for i in range(1, ids.max() + 1)]
            per_object = [m for m in per_object if len(m)]
            floor_total = sum(min(len(m), MIN_KEEP) for m in per_object)
            spare = max(0, int(point_budget * 0.8) - floor_total)
            big_total = sum(max(0, len(m) - MIN_KEEP) for m in per_object) or 1
            kept = []
            for member in per_object:
                quota = min(len(member), MIN_KEEP) + int(
                    spare * max(0, len(member) - MIN_KEEP) / big_total
                )
                kept.append(
                    member
                    if len(member) <= quota
                    else rng.choice(member, quota, replace=False)
                )
            object_index = np.concatenate(kept)
        budget = max(0, point_budget - len(object_index))
        if budget < len(scene_index):
            scene_index = rng.choice(scene_index, budget, replace=False)
        order = np.sort(np.concatenate([object_index, scene_index]))
        xyz, rgb, ids = xyz[order], rgb[order], ids[order]

    objects = []
    for index, (label, inv, frame_id) in enumerate(entries, start=1):
        member = np.flatnonzero(ids == index)
        if len(member) < (1 if inv is not None else MIN_OBJECT_POINTS):
            ids[member] = 0
            continue
        # The inventory already accepted its masks; the viewer cannot reject
        # a thin observed object through a second, sampling-dependent gate.
        # Only an initial pre-inventory build derives its own measurements.
        if inv is None:
            radius = np.linalg.norm(xyz[member, :2], axis=1)
            member = member[radius > 0.15]
            radius = radius[radius > 0.15]
            if len(member) < MIN_OBJECT_POINTS:
                ids[np.flatnonzero(ids == index)] = 0
                continue
            centre = np.median(radius)
            spread = np.median(np.abs(radius - centre)) + 1e-6
            member = member[np.abs(radius - centre) < 2.5 * spread]
            if len(member) < MIN_OBJECT_POINTS:
                ids[np.flatnonzero(ids == index)] = 0
                continue
            ids[np.setdiff1d(np.flatnonzero(ids == index), member)] = 0
        cloud = xyz[member]
        hull = MultiPoint([(x, y) for x, y in cloud[:, :2]]).convex_hull
        top = float(np.quantile(cloud[:, 2], 0.95))
        orientation, tilt, _ = _spatial_state(cloud, max(top, 0.0))
        size = "-"
        if isinstance(hull, Polygon) and hull.area > 0:
            corners = list(hull.minimum_rotated_rectangle.exterior.coords)[:4]
            sides = sorted(
                float(np.hypot(corners[i][0] - corners[i - 1][0],
                               corners[i][1] - corners[i - 1][1]))
                for i in range(1, 3)
            )
            size = f"{sides[1]:.2f} x {sides[0]:.2f} m"
        objects.append(
            {
                "id": index,
                "inv": inv,  # index into inventory/inventory.json["objects"]
                "frame": frame_id,
                "label": label,
                "points": int(len(cloud)),
                "height_m": round(top, 2),
                "size": size,
                "camera_dist_m": round(
                    float(Point(0.0, 0.0).distance(hull))
                    if isinstance(hull, Polygon)
                    else float(np.min(np.linalg.norm(cloud[:, :2], axis=1))),
                    2,
                ),
                "tilt_deg": None if tilt is None else round(tilt, 1),
                "orientation_deg": None if orientation is None else round(orientation, 1),
                "color": PALETTE[(index - 1) % len(PALETTE)],
                "centroid": [round(float(v), 3) for v in np.median(cloud, axis=0)],
            }
        )
        if inv is not None:
            measured = inventory_objects[inv]
            objects[-1].update(height_m=measured["height_m"], size=f"{measured['size_m']} m",
                               camera_dist_m=measured["camera_dist_m"], tilt_deg=measured.get("tilt_deg"),
                               orientation_deg=measured.get("orientation_deg"), measurement_source="inventory")
        else:
            objects[-1]["measurement_source"] = "geometry"
    # Rendering ids are dense after filtering; only inv is an external identity.
    remap = np.zeros(len(entries) + 1, dtype=np.uint16)
    for dense_id, obj in enumerate(objects, start=1):
        remap[obj["id"]] = dense_id
        obj["id"] = dense_id
    ids = remap[ids]
    supported_inv = sorted({obj["inv"] for obj in objects if obj["inv"] is not None})
    unavailable = []
    for item in evidence:
        if item["inv"] in supported_inv or item["inv"] not in interactive_inv:
            continue
        if item["label"] in exclude:
            reason = "场景上下文，不作为可选对象"
        elif not item["mask_pixels"]:
            reason = "无同帧且来源匹配的非空分割 mask"
        elif not item["valid_points"]:
            reason = "mask 内无通过有效性、非零和 alpha 检查的观测点"
        elif not item["owned_points"]:
            reason = "重叠像素全部归属其他更具体的对象 mask"
        elif not item["bounded_points"]:
            reason = "观测点全部超出当前 ±60 m 显示范围"
        else:
            reason = "采样后无保留点"
        unavailable.append(dict(item, reason=reason))

    low = xyz.min(axis=0)
    span = float(np.max(xyz.max(axis=0) - low)) or 1.0
    quantized = np.clip(
        ((xyz - low) / span * 65535.0), 0, 65535
    ).astype("<u2")

    payload = {
        "count": int(len(xyz)),
        "origin": [float(v) for v in low],
        "span": span,
        "objects": objects,
        "supported_inv": supported_inv,
        "interactive_inv": interactive_inv,
        "unavailable": unavailable,
        "inventory_count": len(inventory_objects),
        "run": run.name,
        "xyz": base64.b64encode(quantized.tobytes()).decode("ascii"),
        "rgb": base64.b64encode(rgb.astype(np.uint8).tobytes()).decode("ascii"),
        "ids": base64.b64encode(ids.tobytes()).decode("ascii"),
    }
    html = _VIEWER_TEMPLATE.replace(
        "__PAYLOAD__", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    ).replace("__ANCHORS__", json.dumps(anchors, separators=(",", ":")))
    out = Path(out_path) if out_path is not None else run / "viewer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return {
        "path": out,
        "points": int(len(xyz)),
        "objects": objects,
        "supported_inv": supported_inv,
        "interactive_inv": interactive_inv,
        "unavailable": unavailable,
        "anchors": anchors,
    }


_VIEWER_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D scene — object highlight</title>
<style>
:root{--bg:#0f1216;--panel:#171b21;--edge:#2a3038;--ink:#e8ecf1;--ink2:#98a2ae;--accent:#4fb3de}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
  font-family:"PingFang SC","Hiragino Sans GB",-apple-system,system-ui,sans-serif}
#app{display:flex;height:100%}
#stage{flex:1;position:relative;min-width:0}
canvas{display:block;width:100%;height:100%;cursor:grab}
canvas.dragging{cursor:grabbing}
#side{width:310px;flex:none;background:var(--panel);border-left:1px solid var(--edge);
  display:flex;flex-direction:column;overflow:hidden}
#side h1{font-size:13px;margin:0;padding:14px 16px 10px;letter-spacing:.04em;
  text-transform:uppercase;color:var(--accent);border-bottom:1px solid var(--edge)}
#objectlist{overflow-y:auto;flex:1;min-height:0}
#objectlist summary{cursor:pointer;padding:8px 12px;font-size:12px;color:var(--ink2)}
#list{padding:6px}
.item{display:flex;gap:9px;align-items:flex-start;padding:8px 10px;border-radius:6px;
  cursor:pointer;border:1px solid transparent}
.item:hover{background:#1e242c}
.item:not(.on){opacity:.45}
.item.sel{background:#1d2731;border-color:var(--accent)}
.dot{width:11px;height:11px;border-radius:3px;flex:none;margin-top:3px}
.meta{font-size:11.5px;color:var(--ink2);line-height:1.45;font-variant-numeric:tabular-nums}
.name{font-size:13px;color:var(--ink)}
#foot{padding:10px 16px;border-top:1px solid var(--edge);font-size:11.5px;color:var(--ink2);line-height:1.6}
button{background:#232a33;color:var(--ink);border:1px solid var(--edge);border-radius:6px;
  padding:6px 10px;font-size:12px;cursor:pointer;font-family:inherit}
button:hover{border-color:var(--accent)}
button:focus-visible,.item:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#bar{display:flex;gap:6px;padding:10px 12px;border-bottom:1px solid var(--edge);flex-wrap:wrap}
#hud{position:absolute;left:14px;top:12px;font-size:12px;color:#c8d2dc;
  background:rgba(15,18,22,.72);padding:8px 12px;border-radius:8px;line-height:1.6;
  font-variant-numeric:tabular-nums;pointer-events:none;white-space:pre-line;max-width:calc(100% - 28px)}
@media(max-width:700px){
  #app{flex-direction:column} #stage{min-height:170px}
  #side{width:auto;max-height:42%;border-left:0;border-top:1px solid var(--edge)}
  #side h1,#foot{display:none} #bar{padding:5px 8px;gap:4px}
  button{padding:4px 7px;font-size:11px} #list{max-height:100px;overflow-y:auto}
  #objectlist summary{padding:5px 10px}
}
</style></head><body>
<div id="app">
  <div id="stage"><canvas id="c"></canvas><div id="hud"></div></div>
  <aside id="side">
    <h1 id="title">Objects</h1>
    <div id="bar">
      <button id="all">全选</button><button id="none">全不选</button>
      <button id="mode">语义色 / 照片色</button><button id="scene">场景点 开/关</button>
      <button id="reset">重置</button><span id="cams" style="display:contents"></span>
    </div>
    <details id="objectlist" open><summary>物体列表 · 点选 / 多选</summary><div id="list"></div></details>
    <div id="foot">拖动旋转 · 滚轮缩放 · 右键拖动平移<br>点击物体名或点云高亮,其余变暗 · Shift 多选 · Alt 隐藏<br>
      视角锚定在相机 1 的拍摄位姿,绕其正前方 2.5 m 地面点旋转 · 相机 N 切到该帧</div>
  </aside>
</div>
<script id="anchors" type="application/json">__ANCHORS__</script>
<script>
const DATA = __PAYLOAD__;
const ANCHORS = JSON.parse(document.getElementById('anchors').textContent);
const b64 = (s, T) => { const bin = atob(s); const b = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) b[i] = bin.charCodeAt(i);
  return new T(b.buffer); };
const q = b64(DATA.xyz, Uint16Array), rgb = b64(DATA.rgb, Uint8Array), ids = b64(DATA.ids, Uint16Array);
const N = DATA.count, S = DATA.span / 65535, O = DATA.origin;
const pos = new Float32Array(N * 3);
for (let i = 0; i < N; i++) { pos[i*3] = q[i*3]*S + O[0]; pos[i*3+1] = q[i*3+1]*S + O[1]; pos[i*3+2] = q[i*3+2]*S + O[2]; }
let cx = 0, cy = 0, cz = 0, minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
for (let i = 0; i < N; i++) { const x = pos[i*3], y = pos[i*3+1]; cx += x; cy += y; cz += pos[i*3+2];
  if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y; }
cx /= N; cy /= N; cz /= N;
let radius = 0;
for (let i = 0; i < N; i += 7) { const d = Math.hypot(pos[i*3]-cx, pos[i*3+1]-cy, pos[i*3+2]-cz); if (d > radius) radius = d; }

// Overlay lines ride in the point buffers after the cloud (one draw setup):
// a 1 m floor grid over the cloud's footprint and a frustum per camera.
const LINE_ID = DATA.objects.length + 1;  // the texel past the last object
const L = [], LC = [];
const seg = (a, b, c) => { L.push(...a, ...b); LC.push(...c, ...c); };
const GRID = [74, 84, 100], CAM = [79, 179, 222], CAM1 = [255, 216, 25];
const p0 = ANCHORS.length ? ANCHORS[0].pivot : [cx, cy, 0];
const gx0 = Math.floor(Math.max(minX, p0[0] - 30)), gx1 = Math.ceil(Math.min(maxX, p0[0] + 30));
const gy0 = Math.floor(Math.max(minY, p0[1] - 30)), gy1 = Math.ceil(Math.min(maxY, p0[1] + 30));
for (let x = gx0; x <= gx1; x++) seg([x, gy0, 0], [x, gy1, 0], GRID);
for (let y = gy0; y <= gy1; y++) seg([gx0, y, 0], [gx1, y, 0], GRID);
ANCHORS.forEach((a, i) => {
  const E = a.position, f = a.forward, r = a.right, u = a.up, c = i ? CAM : CAM1;
  const corner = (sx, sy) => [0, 1, 2].map(k => E[k] + 0.35*f[k] + sx*0.25*r[k] + sy*0.19*u[k]);
  const C = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
  C.forEach((pt, k) => { seg(E, pt, c); seg(pt, C[(k + 1) % 4], c); });
});
const NL = L.length / 3;
const posAll = new Float32Array(N*3 + L.length); posAll.set(pos); posAll.set(L, N*3);
const rgbAll = new Uint8Array(N*3 + LC.length); rgbAll.set(rgb); rgbAll.set(LC, N*3);
const idAll = new Float32Array(N + NL); idAll.set(ids); idAll.fill(LINE_ID, N);

const cv = document.getElementById('c');
// Single-sample pixels keep the id pass exact where two objects meet.
const gl = cv.getContext('webgl', {antialias:false, alpha:false});
if (!gl) { document.getElementById('hud').textContent = '无法初始化 WebGL'; throw new Error('WebGL unavailable'); }
gl.disable(gl.DITHER);
// Per-object texel: rgb = legend colour, a = 0 hidden / 170 shown / 255 selected.
const VS = `attribute vec3 p; attribute vec3 col; attribute float oid;
uniform mat4 mvp; uniform float anySel; uniform float sceneOn; uniform float psize;
uniform sampler2D vis; uniform float nObj; uniform float semantic;
varying vec3 vC; varying float vDrop; varying float vId;
void main(){
  gl_Position = mvp * vec4(p,1.0);
  vId = oid;
  vec4 meta = texture2D(vis, vec2((oid+0.5)/nObj, 0.5));
  float shown = oid < 0.5 ? sceneOn : (meta.a > 0.25 ? 1.0 : 0.0);
  vDrop = shown < 0.5 ? 1.0 : 0.0;
  float grey = dot(col, vec3(0.299,0.587,0.114));
  vec3 c = col;
  bool isLine = oid > nObj - 1.5;  // grid + camera markers keep their own colour
  bool isObj = oid > 0.5 && !isLine;
  bool isSel = isObj && meta.a > 0.9;
  // Semantic mode paints every object its legend colour and mutes the rest,
  // so a glance answers "which thing is that" without clicking.
  if (semantic > 0.5 && !isLine) { c = isObj ? meta.rgb : vec3(grey*0.42+0.30); }
  if (anySel > 0.5 && !isLine) {
    if (isSel) { c = mix(min(c*1.1+0.12, vec3(1.0)), meta.rgb, 0.55); }
    else { c = vec3(grey*0.26+0.11); }
  }
  vC = c;
  gl_PointSize = psize * (isSel ? 2.0 : 1.0);
}`;
// pick > 0.5: write the object id (hi, lo bytes) instead of colour — one
// extra draw per click, read back under the cursor.
const FS = `precision mediump float; varying vec3 vC; varying float vDrop; varying float vId;
uniform float pick;
void main(){ if (vDrop > 0.5) discard;
  if (pick > 0.5) { gl_FragColor = vec4(floor(vId/256.0)/255.0, mod(vId,256.0)/255.0, 1.0, 1.0); return; }
  gl_FragColor = vec4(vC, 1.0); }`;
function sh(t, s){ const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o);
  if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(o)); return o; }
const prog = gl.createProgram();
gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog); gl.useProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));

function buf(data, loc, size, type, norm){ const b = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  const l = gl.getAttribLocation(prog, loc); gl.enableVertexAttribArray(l);
  gl.vertexAttribPointer(l, size, type, norm, 0, 0); }
buf(posAll, 'p', 3, gl.FLOAT, false);
buf(rgbAll, 'col', 3, gl.UNSIGNED_BYTE, true);
buf(idAll, 'oid', 1, gl.FLOAT, false);

const nObj = DATA.objects.length + 2;  // 0 = scene, 1..n objects, n+1 = lines
const visData = new Uint8Array(nObj * 4).fill(255);
DATA.objects.forEach(o => { visData[o.id*4] = o.color[0];
  visData[o.id*4+1] = o.color[1]; visData[o.id*4+2] = o.color[2]; visData[o.id*4+3] = 170; });
const visTex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, visTex);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
function pushVis(){ gl.bindTexture(gl.TEXTURE_2D, visTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, nObj, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, visData); }
pushVis();
gl.uniform1i(gl.getUniformLocation(prog, 'vis'), 0);
gl.uniform1f(gl.getUniformLocation(prog, 'nObj'), nObj);

let yaw = -0.6, pitch = 0.5, dist = radius * 2.1, panX = 0, panY = 0;
let sceneOn = 1, semantic = 0;  // photo colours read best
// Selection keys are inventory indices. A fresh run without inventory uses
// local object keys, which never cross the report bridge.
const selected = new Set(), hidden = new Set();
const key = o => o.inv === null ? 'object:' + o.id : o.inv;
const byId = Object.fromEntries(DATA.objects.map(o => [o.id, o]));
const rows = new Map();
let parentOrigin = null;
try { if (window.parent !== window) parentOrigin = window.parent.origin || window.parent.location.origin; } catch (_) {}
function tellParent(type, fields){
  if (parentOrigin !== null) window.parent.postMessage({type, ...fields}, parentOrigin === 'null' ? '*' : parentOrigin);
}
function selectedInv(){ return [...selected].filter(Number.isInteger); }
function setSel(keys, notify = true){
  selected.clear();
  keys.forEach(k => { selected.add(k); hidden.delete(k); });
  DATA.objects.forEach(o => {
    const k = key(o), on = selected.has(k), el = rows.get(o.id);
    visData[o.id*4+3] = hidden.has(k) ? 0 : on ? 255 : 170;
    if (el) { el.classList.toggle('sel', on); el.classList.toggle('on', !hidden.has(k));
      el.setAttribute('aria-pressed', String(on)); }
  });
  const visibleKeys = new Set(DATA.objects.filter(o => selected.has(key(o))).map(key));
  document.getElementById('hud').textContent = selected.size
    ? `已选 ${selected.size} 个 · 当前 3D 可显示 ${visibleKeys.size} 个`
    : `${DATA.objects.length} 个物体 · ${N.toLocaleString()} 点`;
  const missing = DATA.unavailable.filter(o => selected.has(o.inv));
  if (missing.length) document.getElementById('hud').textContent += '\n' +
    missing.map(o => `${o.label} (${o.frame})：${o.reason}`).join('\n');
  pushVis(); draw();
  if (notify) tellParent('panoptes:selected', {inv: selectedInv()});
}
function clickSel(o, multi){
  const k = key(o), next = multi ? new Set(selected) : new Set();
  if (selected.has(k)) next.delete(k); else next.add(k);
  setSel([...next]);
}
function toggleHidden(o){
  const k = key(o), next = new Set(selected);
  if (hidden.has(k)) hidden.delete(k); else hidden.add(k);
  next.delete(k); setSel([...next]);
}
// Anchor the orbit rig to a frame's camera: the pivot (cx,cy,cz) is the floor
// point ahead of the phone, and yaw/pitch/dist/pan are solved so the eye sits
// exactly where the phone was, looking exactly where it looked. The pivot
// is off the view axis in general, so it rides as a pan offset.
function goTo(i){
  const a = ANCHORS[i]; if (!a) return;
  const E = a.position, f = a.forward;
  cx = a.pivot[0]; cy = a.pivot[1]; cz = a.pivot[2];
  const zx = -f[0], zy = -f[1], zz = -f[2];  // orbit z points from target to eye
  pitch = Math.asin(Math.max(-1, Math.min(1, zz))); yaw = Math.atan2(zx, zy);
  dist = Math.max(0.5, (cx-E[0])*f[0] + (cy-E[1])*f[1] + (cz-E[2])*f[2]);
  const xl = Math.hypot(zx, zy) || 1, xx = -zy/xl, xy = zx/xl;
  const ux = -zz*xy, uy = zz*xx, uz = zx*xy - zy*xx;
  const tx = E[0] + f[0]*dist - cx, ty = E[1] + f[1]*dist - cy, tz = E[2] + f[2]*dist - cz;
  panX = -(tx*xx + ty*xy); panY = -(tx*ux + ty*uy + tz*uz);
}
goTo(0);
function mat(){
  const a = cv.width / cv.height, f = 1 / Math.tan(0.5), near = 0.02, far = radius * 40;
  const P = [f/a,0,0,0, 0,f,0,0, 0,0,(far+near)/(near-far),-1, 0,0,2*far*near/(near-far),0];
  const cy_ = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const ex = cx + dist*cp*sy, ey = cy + dist*cp*cy_, ez = cz + dist*sp;
  let zx = ex-cx, zy = ey-cy, zz = ez-cz; const zl = Math.hypot(zx,zy,zz); zx/=zl; zy/=zl; zz/=zl;
  let xx = -zy, xy = zx, xz = 0; const xl = Math.hypot(xx,xy,xz) || 1; xx/=xl; xy/=xl; xz/=xl;
  const ux = zy*xz - zz*xy, uy = zz*xx - zx*xz, uz = zx*xy - zy*xx;
  const tx = cx - xx*panX - ux*panY, ty = cy - xy*panX - uy*panY, tz = cz - xz*panX - uz*panY;
  const px = tx + zx*dist, py = ty + zy*dist, pz = tz + zz*dist;
  const V = [xx,ux,zx,0, xy,uy,zy,0, xz,uz,zz,0,
             -(xx*px+xy*py+xz*pz), -(ux*px+uy*py+uz*pz), -(zx*px+zy*py+zz*pz), 1];
  const M = new Float32Array(16);
  for (let i=0;i<4;i++) for (let j=0;j<4;j++){ let s=0;
    for (let k=0;k<4;k++) s += V[i*4+k]*P[k*4+j]; M[i*4+j]=s; }
  return M;
}
function draw(pick = false){
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = cv.clientWidth * dpr | 0, h = cv.clientHeight * dpr | 0;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  gl.viewport(0,0,cv.width,cv.height);
  gl.clearColor(pick ? 0 : 0.059, pick ? 0 : 0.071, pick ? 0 : 0.086, 1); gl.enable(gl.DEPTH_TEST);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.uniformMatrix4fv(gl.getUniformLocation(prog,'mvp'), false, mat());
  gl.uniform1f(gl.getUniformLocation(prog,'pick'), pick ? 1 : 0);
  gl.uniform1f(gl.getUniformLocation(prog,'anySel'), selected.size ? 1 : 0);
  gl.uniform1f(gl.getUniformLocation(prog,'sceneOn'), sceneOn);
  gl.uniform1f(gl.getUniformLocation(prog,'semantic'), semantic);
  gl.uniform1f(gl.getUniformLocation(prog,'psize'),
    Math.min(6 * dpr, Math.max(2.2, 4.2 * dpr * (radius*2.1/dist))));
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, visTex);
  gl.drawArrays(gl.POINTS, 0, N);
  gl.drawArrays(gl.LINES, N, NL);
}
let drag = null;
cv.addEventListener('pointerdown', e => { drag = {x:e.clientX, y:e.clientY, b:e.button, moved:0};
  cv.setPointerCapture(e.pointerId); cv.classList.add('dragging'); });
cv.addEventListener('pointerup', e => {
  const d = drag; drag = null; cv.classList.remove('dragging');
  if (d && d.b === 0 && d.moved < 4) pickAt(e);  // a click, not an orbit
});
cv.addEventListener('pointermove', e => { if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
  drag.moved += Math.abs(dx) + Math.abs(dy);
  if (drag.b === 2 || e.shiftKey) { panX -= dx * dist * 0.0016; panY += dy * dist * 0.0016; }
  else { yaw -= dx * 0.007; pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.007)); }
  draw(); });
// 3D pick: id pass, read the pixel under the cursor, restore the colour pass.
function pickAt(e){
  const r = cv.getBoundingClientRect(), sx = cv.width / r.width, sy = cv.height / r.height;
  const x = (e.clientX - r.left) * sx | 0, y = cv.height - ((e.clientY - r.top) * sy | 0) - 1;
  if (x < 0 || y < 0 || x >= cv.width || y >= cv.height) return;
  draw(true);
  const px = new Uint8Array(4); gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  draw();
  const id = px[2] === 255 ? px[0] * 256 + px[1] : 0;  // blue = an unblended id texel
  const o = byId[id];
  if (!o) { if (!e.shiftKey && !e.metaKey && !e.ctrlKey) setSel([]); return; }
  if (e.altKey) toggleHidden(o); else clickSel(o, e.shiftKey || e.metaKey || e.ctrlKey);
}
cv.addEventListener('contextmenu', e => e.preventDefault());
cv.addEventListener('wheel', e => { e.preventDefault();
  dist = Math.max(radius*0.08, Math.min(radius*9, dist * Math.exp(e.deltaY * 0.0012))); draw(); },
  {passive:false});

const list = document.getElementById('list');
if (window.innerWidth <= 700) document.getElementById('objectlist').open = false;
document.getElementById('title').textContent = DATA.objects.length + ' 个物体 · ' + DATA.run;
DATA.objects.forEach(o => {
  const el = document.createElement('div');
  el.className = 'item on'; el.tabIndex = 0; el.setAttribute('role', 'button');
  el.setAttribute('data-id', String(o.id));
  if (o.inv !== null) el.setAttribute('data-inv', String(o.inv));
  const dot = document.createElement('span'); dot.className = 'dot';
  dot.style.background = `rgb(${o.color.join(',')})`; el.appendChild(dot);
  const text = document.createElement('span');
  text.textContent = `${o.label} · ${o.frame}\n高 ${o.height_m} m · ${o.size}\n距相机 ${o.camera_dist_m} m · ${o.points} 点`;
  text.style.whiteSpace = 'pre-line'; text.className = 'meta'; el.appendChild(text);
  el.addEventListener('click', e => e.altKey ? toggleHidden(o) : clickSel(o, e.shiftKey || e.metaKey || e.ctrlKey));
  el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault(); if (e.altKey) toggleHidden(o); else clickSel(o, e.shiftKey || e.metaKey || e.ctrlKey); } });
  rows.set(o.id, el);
  list.appendChild(el);
});
document.getElementById('all').onclick = () => setSel(DATA.inventory_count ? DATA.interactive_inv : DATA.objects.map(key));
document.getElementById('none').onclick = () => setSel([]);
document.getElementById('mode').onclick = () => { semantic = semantic ? 0 : 1; draw(); };
document.getElementById('scene').onclick = () => { sceneOn = sceneOn ? 0 : 1; draw(); };
document.getElementById('reset').onclick = () => { goTo(0); hidden.clear(); setSel([]); };
if (ANCHORS.length > 1) ANCHORS.forEach((a, i) => {
  const b = document.createElement('button'); b.textContent = '相机 ' + (i + 1);
  b.title = a.frame_id; b.onclick = () => { goTo(i); draw(); };
  document.getElementById('cams').appendChild(b); });
// Only the real same-origin parent may drive this viewer. Keep valid inventory
// selections without geometry so a later Shift-click does not erase them.
addEventListener('message', e => {
  const m = e.data;
  if (parentOrigin === null || e.source !== window.parent || e.origin !== parentOrigin
      || !m || m.type !== 'panoptes:select' || !Array.isArray(m.inv)) return;
  const incoming = m.inv.filter(i => Number.isInteger(i) && DATA.interactive_inv.includes(i));
  setSel(m.exclusive === false ? [...selected, ...incoming] : incoming, false);
});
addEventListener('resize', () => draw());
setSel([], false);
tellParent('panoptes:ready', {supported_inv: DATA.supported_inv, unavailable: DATA.unavailable});
</script></body></html>
"""


__all__ = ["build_viewer_html", "camera_anchor", "render_frame_overlays", "DEFAULT_EXCLUDE"]
