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
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint, Point, Polygon

from .contracts import GeometryFrame, Observation2D
from .geometry import _build_geometry, _spatial_state
from .providers.sam3 import decode_coco_rle

MAX_POINTS = 140_000
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


def _masks_for_frame(
    run: Path,
    frame: GeometryFrame,
    shape: tuple[int, int],
    observations: list[Observation2D],
) -> dict[str, np.ndarray]:
    """Per-INSTANCE masks for this frame, in stable discovery order.

    One entry per SAM RLE (instance), not one union per label — the floor
    plan draws instances, and the 3D viewer must bind one-to-one with it.
    Near-duplicate masks across sources (observation masks repeating the
    inventory's SAM instances) are dropped by IoU.
    """
    height, width = shape
    by_label: dict[str, list[np.ndarray]] = {}

    def add(label: str, mask: np.ndarray) -> None:
        if int(mask.sum()) < MIN_MASK_PIXELS:
            return
        peers = by_label.setdefault(label, [])
        area = mask.sum()
        for existing in peers:
            inter = int((existing & mask).sum())
            union = int(existing.sum()) + int(area) - inter
            if union and inter / union > 0.85:
                return  # same physical instance seen through another source
        peers.append(mask)

    for source in ("inventory/sam", "semantic_layers/sam"):
        directory = run / source
        if not directory.is_dir():
            continue
        for cache in sorted(directory.glob(f"{frame.frame_id}__*.json")):
            response = json.loads(cache.read_text())
            rles = response.get("rle") or []
            if isinstance(rles, str):
                rles = [rles]
            for rle in rles:
                add(
                    _label_from_cache(cache),
                    decode_coco_rle(rle, height=height, width=width).astype(bool),
                )

    # Human box-prompt refinements (scripts/refine_region.py) are reviewer
    # ground truth for classes the text prompts miss — include them.
    refine_log = run / "refinements.json"
    if refine_log.exists():
        for item in json.loads(refine_log.read_text()):
            slug = (
                f"{item['label'].replace(' ', '_')}_"
                + "_".join(str(b) for b in item["box"])
            )
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
                # refinements segment the full-resolution input, not the
                # canonical frame: decode at native size, then resize.
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
                if (native_h, native_w) != (height, width):
                    mask = (
                        np.asarray(
                            Image.fromarray(mask * 255).resize((width, height))
                        )
                        > 127
                    ).astype(np.uint8)
                add(item["label"], mask.astype(bool))

    # Detection-layer instances synced into the inventory (taxonomy items
    # located by the VLM and segmented by SAM box prompts) live as
    # refinements/<slug>.json without a refinements.json entry — the glass
    # guard, deflectors, e-stops. One pipeline: the viewer shows them too.
    inventory_path = run / "inventory" / "inventory.json"
    if inventory_path.exists():
        try:
            inventory_objects = json.loads(inventory_path.read_text()).get("objects", [])
        except Exception:
            inventory_objects = []
        inputs = sorted((run / "input").glob("image_*"))
        for item in inventory_objects:
            slug = item.get("refine_slug")
            if not slug or item.get("frame", "frame_0001") != frame.frame_id:
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
                with Image.open(inputs[0]) as native:
                    native_w, native_h = native.size
            mask = decode_coco_rle(
                rles[best], height=int(native_h), width=int(native_w)
            ).astype(np.uint8)
            if (int(native_h), int(native_w)) != (height, width):
                mask = (
                    np.asarray(Image.fromarray(mask * 255).resize((width, height)))
                    > 127
                ).astype(np.uint8)
            add(str(item.get("label", slug)), mask.astype(bool))

    for observation in observations:
        if observation.frame_id != frame.frame_id or not observation.mask_path:
            continue
        with Image.open(observation.mask_path) as image:
            add(observation.label, np.asarray(image).astype(bool))

    flat: list[tuple[str, np.ndarray]] = []
    for label, peers in by_label.items():
        for position, mask in enumerate(peers):
            display = f"{label} #{position + 1}" if len(peers) > 1 else label
            flat.append((display, mask))
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

    labels: list[str] = []
    chunks_xyz, chunks_rgb, chunks_id = [], [], []
    for frame in frames:
        points3d = np.load(frame.pts3d_path)
        valid = np.load(frame.valid_mask_path).astype(bool)
        finite = valid & np.isfinite(points3d).all(axis=2)
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
        for label, mask in sorted(masks, key=lambda kv: -int(kv[1].sum())):
            if label in exclude:
                continue
            if label not in labels:
                labels.append(label)
            ids[mask] = labels.index(label) + 1
        chunks_xyz.append(transform.apply(points3d[finite]))
        chunks_rgb.append(rgb[finite])
        chunks_id.append(ids[finite])

    xyz = np.vstack(chunks_xyz)
    rgb = np.vstack(chunks_rgb)
    ids = np.concatenate(chunks_id)

    keep = np.isfinite(xyz).all(axis=1) & (np.abs(xyz) < 60).all(axis=1)
    xyz, rgb, ids = xyz[keep], rgb[keep], ids[keep]
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
    for index, label in enumerate(labels, start=1):
        member = np.flatnonzero(ids == index)
        if len(member) < MIN_OBJECT_POINTS:
            ids[member] = 0
            continue
        # Depth smear puts a tail of an object's mask on the far background;
        # without trimming it every hull swells to room size and every
        # distance collapses to zero. Trim in range, then keep the dominant
        # connected blob in plan view.
        cloud = xyz[member]
        radius = np.linalg.norm(cloud[:, :2], axis=1)
        # Order matters (mirrors scene_inventory._clean): drop the degenerate
        # near-origin reconstruction pixels FIRST, then take median/MAD on
        # the survivors. Computing the median on the raw cloud let tens of
        # thousands of collapsed points set centre=0 and erase whole labels.
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
        dropped = np.setdiff1d(np.flatnonzero(ids == index), member)
        ids[dropped] = 0
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
    live_ids = {obj["id"] for obj in objects}
    ids = np.where(np.isin(ids, list(live_ids)), ids, 0).astype(np.uint16)

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
        "run": run.name,
        "xyz": base64.b64encode(quantized.tobytes()).decode("ascii"),
        "rgb": base64.b64encode(rgb.astype(np.uint8).tobytes()).decode("ascii"),
        "ids": base64.b64encode(ids.tobytes()).decode("ascii"),
    }
    html = _VIEWER_TEMPLATE.replace(
        "__PAYLOAD__", json.dumps(payload, separators=(",", ":"))
    ).replace("__ANCHORS__", json.dumps(anchors, separators=(",", ":")))
    out = Path(out_path) if out_path is not None else run / "viewer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return {
        "path": out,
        "points": int(len(xyz)),
        "objects": objects,
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
#list{overflow-y:auto;flex:1;padding:6px}
.item{display:flex;gap:9px;align-items:flex-start;padding:8px 10px;border-radius:6px;
  cursor:pointer;border:1px solid transparent}
.item:hover{background:#1e242c}
.item.on{background:#1d2731;border-color:var(--accent)}
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
  font-variant-numeric:tabular-nums;pointer-events:none}
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
    <div id="list"></div>
    <div id="foot">拖动旋转 · 滚轮缩放 · 右键拖动平移<br>点击物体名高亮,其余变暗<br>
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
const gl = cv.getContext('webgl', {antialias:true, alpha:false});
const VS = `attribute vec3 p; attribute vec3 col; attribute float oid;
uniform mat4 mvp; uniform float sel; uniform float sceneOn; uniform float psize;
uniform sampler2D vis; uniform float nObj; uniform float semantic;
varying vec3 vC; varying float vDrop;
void main(){
  gl_Position = mvp * vec4(p,1.0);
  vec4 meta = texture2D(vis, vec2((oid+0.5)/nObj, 0.5));
  float shown = oid < 0.5 ? sceneOn : meta.a;
  vDrop = shown < 0.5 ? 1.0 : 0.0;
  float grey = dot(col, vec3(0.299,0.587,0.114));
  vec3 c = col;
  bool isLine = oid > nObj - 1.5;  // grid + camera markers keep their own colour
  bool isObj = oid > 0.5 && !isLine;
  // Semantic mode paints every object its legend colour and mutes the rest,
  // so a glance answers "which thing is that" without clicking.
  if (semantic > 0.5 && !isLine) { c = isObj ? meta.rgb : vec3(grey*0.42+0.30); }
  if (sel > 0.5 && !isLine) {
    if (abs(oid - sel) < 0.5) { c = mix(min(c*1.1+0.12, vec3(1.0)), meta.rgb, 0.55); }
    else { c = vec3(grey*0.26+0.11); }
  }
  vC = c;
  gl_PointSize = psize * (abs(oid - sel) < 0.5 && sel > 0.5 ? 2.0 : 1.0);
}`;
const FS = `precision mediump float; varying vec3 vC; varying float vDrop;
void main(){ if (vDrop > 0.5) discard; gl_FragColor = vec4(vC, 1.0); }`;
function sh(t, s){ const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o);
  if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(o)); return o; }
const prog = gl.createProgram();
gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog); gl.useProgram(prog);

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
  visData[o.id*4+1] = o.color[1]; visData[o.id*4+2] = o.color[2]; visData[o.id*4+3] = 255; });
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
let sel = 0, sceneOn = 1, semantic = 0;  // photo colours read best
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
function draw(){
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = cv.clientWidth * dpr | 0, h = cv.clientHeight * dpr | 0;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  gl.viewport(0,0,cv.width,cv.height);
  gl.clearColor(0.059,0.071,0.086,1); gl.enable(gl.DEPTH_TEST);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.uniformMatrix4fv(gl.getUniformLocation(prog,'mvp'), false, mat());
  gl.uniform1f(gl.getUniformLocation(prog,'sel'), sel);
  gl.uniform1f(gl.getUniformLocation(prog,'sceneOn'), sceneOn);
  gl.uniform1f(gl.getUniformLocation(prog,'semantic'), semantic);
  gl.uniform1f(gl.getUniformLocation(prog,'psize'),
    Math.min(6 * dpr, Math.max(2.2, 4.2 * dpr * (radius*2.1/dist))));
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, visTex);
  gl.drawArrays(gl.POINTS, 0, N);
  gl.drawArrays(gl.LINES, N, NL);
}
let drag = null;
cv.addEventListener('pointerdown', e => { drag = {x:e.clientX, y:e.clientY, b:e.button};
  cv.setPointerCapture(e.pointerId); cv.classList.add('dragging'); });
cv.addEventListener('pointerup', e => { drag = null; cv.classList.remove('dragging'); });
cv.addEventListener('pointermove', e => { if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
  if (drag.b === 2 || e.shiftKey) { panX -= dx * dist * 0.0016; panY += dy * dist * 0.0016; }
  else { yaw -= dx * 0.007; pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.007)); }
  draw(); });
cv.addEventListener('contextmenu', e => e.preventDefault());
cv.addEventListener('wheel', e => { e.preventDefault();
  dist = Math.max(radius*0.08, Math.min(radius*9, dist * Math.exp(e.deltaY * 0.0012))); draw(); },
  {passive:false});

const list = document.getElementById('list');
document.getElementById('title').textContent = DATA.objects.length + ' 个物体 · ' + DATA.run;
DATA.objects.forEach(o => {
  const el = document.createElement('div');
  el.className = 'item on'; el.tabIndex = 0;
  el.innerHTML = `<span class="dot" style="background:rgb(${o.color.join(',')})"></span>
    <span><span class="name">${o.label}</span><br><span class="meta">高 ${o.height_m} m ·
    ${o.size}<br>距相机 ${o.camera_dist_m} m · ${o.points} 点${
      o.tilt_deg === null ? '' : ' · 倾角 ' + o.tilt_deg + '°'}</span></span>`;
  const toggle = () => { const on = el.classList.toggle('on');
    visData[o.id*4+3] = on ? 255 : 0; pushVis();
    if (!on && sel === o.id) sel = 0; draw(); };
  const focus = () => { sel = sel === o.id ? 0 : o.id;
    document.getElementById('hud').textContent = sel
      ? `${o.label} · 高 ${o.height_m} m · ${o.size} · 距相机 ${o.camera_dist_m} m`
      : `${DATA.objects.length} objects · ${N.toLocaleString()} points`;
    // reverse sync: tell an embedding page (the report's floor plan)
    try { if (window.parent !== window)
      window.parent.postMessage({type:'ehs-picked', label: sel ? o.label : null}, '*');
    } catch (e) {}
    draw(); };
  el.addEventListener('click', e => (e.altKey ? toggle() : focus()));
  el.addEventListener('keydown', e => { if (e.key === 'Enter') focus();
    if (e.key === ' ') { e.preventDefault(); toggle(); } });
  list.appendChild(el);
});
document.getElementById('all').onclick = () => { DATA.objects.forEach(o => visData[o.id*4+3] = 255);
  [...list.children].forEach(c => c.classList.add('on')); pushVis(); draw(); };
document.getElementById('none').onclick = () => { DATA.objects.forEach(o => visData[o.id*4+3] = 0);
  [...list.children].forEach(c => c.classList.remove('on')); sel = 0; pushVis(); draw(); };
document.getElementById('mode').onclick = () => { semantic = semantic ? 0 : 1; draw(); };
document.getElementById('scene').onclick = () => { sceneOn = sceneOn ? 0 : 1; draw(); };
document.getElementById('reset').onclick = () => { goTo(0); sel = 0; draw(); };
if (ANCHORS.length > 1) ANCHORS.forEach((a, i) => {
  const b = document.createElement('button'); b.textContent = '相机 ' + (i + 1);
  b.title = a.frame_id; b.onclick = () => { goTo(i); draw(); };
  document.getElementById('cams').appendChild(b); });
document.getElementById('hud').textContent = `${DATA.objects.length} objects · ${N.toLocaleString()} points`;
// Embedding pages (the test-set report's interactive floor plan) drive the
// same focus routine over postMessage: {type:'ehs-select', label}.
addEventListener('message', e => {
  const m = e.data;
  if (!m || m.type !== 'ehs-select') return;
  const tokens = s => String(s || '').toLowerCase().split(/[^a-z]+/).filter(t => t.length >= 4);
  const wantT = tokens(m.label);
  // Stem-prefix overlap so 'robotic arm' finds 'industrial robot arm'.
  const overlap = (a, b) => a.some(x => b.some(y => x.startsWith(y) || y.startsWith(x)));
  const o = DATA.objects.find(x => x.label === m.label)
    || DATA.objects.find(x => overlap(tokens(x.label), wantT));
  if (!o) return;
  sel = o.id;
  document.getElementById('hud').textContent =
    `${o.label} · 高 ${o.height_m} m · ${o.size} · 距相机 ${o.camera_dist_m} m`;
  draw();
});
addEventListener('resize', draw);
draw();
</script></body></html>
"""


__all__ = ["build_viewer_html", "camera_anchor", "render_frame_overlays", "DEFAULT_EXCLUDE"]
