"""Fine-grained semantic layering over a run's cached geometry.

Segments an EHS-relevant vocabulary on a completed run's canonical frames,
lifts every mask into the run's floor frame, and reports per-class 3D state
(height, tilt, footprint, distance to camera and to a reference class) plus
a colour-coded point cloud render.

Exploration tool: it reads a run's cached geometry, spends only SAM calls
(disk-cached, so re-runs are free), and never touches production vocabulary
or verdicts.

Usage:
  uv run --env-file .env python scripts/semantic_layers.py --run demo-real-factory
  uv run --env-file .env python scripts/semantic_layers.py --run demo-real-factory --live
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ehs_spatial.contracts import GeometryFrame, Observation2D
from ehs_spatial.geometry import _build_geometry, _spatial_state
from ehs_spatial.providers.sam3 import SAM3_ENDPOINT, decode_coco_rle

# EHS-relevant layers. Each entry is (canonical label, prompt synonyms tried
# in order until one hits, RGB colour). Deliberately broader than the
# production vocabulary: this is where new rule subjects get discovered.
LAYERS: list[tuple[str, tuple[str, ...], tuple[int, int, int]]] = [
    ("safety fence", ("safety fence", "fence", "guard rail"), (232, 30, 160)),
    ("person", ("person", "worker"), (255, 122, 0)),
    ("machine", ("machine", "industrial machine"), (58, 122, 196)),
    ("control panel", ("control panel", "hmi screen"), (0, 190, 190)),
    ("storage rack", ("storage rack", "shelving"), (140, 90, 220)),
    ("floor marking", ("yellow line", "floor marking"), (240, 200, 0)),
    ("pallet", ("pallet", "wooden pallet"), (170, 110, 40)),
    ("hard hat", ("hard hat", "safety helmet"), (255, 210, 40)),
    ("safety vest", ("safety vest", "high visibility vest"), (255, 70, 40)),
    ("door", ("door", "doorway"), (120, 140, 160)),
    ("ceiling light", ("ceiling light", "light fixture"), (200, 220, 240)),
    ("cable", ("cable", "hose"), (90, 90, 90)),
    ("fire extinguisher", ("fire extinguisher",), (220, 20, 20)),
    ("window", ("window",), (150, 200, 230)),
]
REFERENCE_LABEL = "safety fence"
MIN_MASK_PIXELS = 60
MIN_CLOUD_POINTS = 60


def _frames(run: Path) -> list[GeometryFrame]:
    frames = []
    for frame_dir in sorted((run / "geometry" / "frames").iterdir()):
        if not frame_dir.is_dir():
            continue
        frames.append(
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
    return frames


def _cache_path(run: Path, frame_id: str, prompt: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return run / "semantic_layers" / "sam" / f"{frame_id}__{slug}.json"


def _segment(run: Path, frame: GeometryFrame, prompt: str, *, live: bool):
    cache = _cache_path(run, frame.frame_id, prompt)
    if cache.exists():
        return json.loads(cache.read_text())
    if not live:
        return None
    import fal_client

    source = Path(frame.canonical_image_path)
    response = fal_client.subscribe(
        SAM3_ENDPOINT,
        arguments={
            "image_url": "data:image/png;base64,"
            + base64.b64encode(source.read_bytes()).decode("ascii"),
            "prompt": prompt,
            "return_multiple_masks": True,
            "include_scores": True,
            "include_boxes": True,
            "max_masks": 8,
        },
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(response) + "\n", encoding="utf-8")
    return response


def _masks(response, height: int, width: int) -> list[tuple[np.ndarray, float]]:
    rles = response.get("rle") or []
    if isinstance(rles, str):
        rles = [rles]
    scores = response.get("scores") or [1.0] * len(rles)
    out = []
    for index, rle in enumerate(rles):
        mask = decode_coco_rle(rle, height=height, width=width)
        if int(mask.sum()) < MIN_MASK_PIXELS:
            continue
        out.append(
            (mask.astype(bool), float(scores[index]) if index < len(scores) else 1.0)
        )
    return out


def _clean(points: np.ndarray) -> np.ndarray | None:
    """Drop reconstruction garbage and depth-smear tails (same discipline as
    the production binding layer, without its verdict-side gates)."""
    if len(points) < MIN_CLOUD_POINTS:
        return None
    horizontal = np.linalg.norm(points[:, :2], axis=1)
    points = points[horizontal > 0.15]
    if len(points) < MIN_CLOUD_POINTS:
        return None
    horizontal = np.linalg.norm(points[:, :2], axis=1)
    median = np.median(horizontal)
    deviation = np.median(np.abs(horizontal - median)) + 1e-6
    points = points[np.abs(horizontal - median) < 2.5 * deviation]
    return points if len(points) >= MIN_CLOUD_POINTS else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run id under runs/")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--camera-height", type=float, default=1.5)
    args = parser.parse_args(argv)

    run = Path("runs") / args.run
    frames = _frames(run)
    observations = [
        Observation2D.model_validate(item)
        for item in json.loads((run / "observations.json").read_text())
    ]
    transform = _build_geometry(frames, observations, args.camera_height).transform
    if transform is None:
        print("run has no floor transform", file=sys.stderr)
        return 1

    planned = sum(
        1
        for frame in frames
        for _, prompts, _ in LAYERS
        for prompt in prompts
        if not _cache_path(run, frame.frame_id, prompt).exists()
    )
    print(f"{len(frames)} frame(s) x {len(LAYERS)} layers | up to {planned} SAM calls")
    if planned and not args.live:
        print("pass --live to spend them (cached re-runs are free)", file=sys.stderr)
        return 2

    layer_points: dict[str, list[np.ndarray]] = {}
    layer_scores: dict[str, list[float]] = {}
    layer_prompt: dict[str, str] = {}
    layer_masks: dict[str, dict[str, np.ndarray]] = {}
    for frame in frames:
        points3d = np.load(frame.pts3d_path)
        valid = np.load(frame.valid_mask_path).astype(bool)
        finite = valid & np.isfinite(points3d).all(axis=2)
        height, width = valid.shape
        for label, prompts, _ in LAYERS:
            for prompt in prompts:
                response = _segment(run, frame, prompt, live=args.live)
                if response is None:
                    continue
                hits = _masks(response, height, width)
                if not hits:
                    continue
                kept = False
                for mask, score in hits:
                    cloud = _clean(transform.apply(points3d[mask & finite]))
                    if cloud is None:
                        continue
                    layer_points.setdefault(label, []).append(cloud)
                    layer_scores.setdefault(label, []).append(score)
                    layer_prompt.setdefault(label, prompt)
                    per_frame = layer_masks.setdefault(frame.frame_id, {})
                    per_frame[label] = per_frame.get(
                        label, np.zeros_like(mask)
                    ) | mask
                    kept = True
                if kept:
                    break

    reference = None
    if REFERENCE_LABEL in layer_points:
        from shapely.geometry import MultiPoint

        merged = np.vstack(layer_points[REFERENCE_LABEL])
        reference = MultiPoint([(x, y) for x, y in merged[:, :2]]).convex_hull

    from shapely.geometry import MultiPoint, Point

    rows = []
    for label, prompts, _ in LAYERS:
        chunks = layer_points.get(label)
        if not chunks:
            rows.append({"label": label, "instances": 0, "prompt": None})
            continue
        merged = np.vstack(chunks)
        hull = MultiPoint([(x, y) for x, y in merged[:, :2]]).convex_hull
        top = float(np.quantile(merged[:, 2], 0.95))
        orientation, tilt, overhang = _spatial_state(merged, max(top, 0.0))
        row = {
            "label": label,
            "prompt": layer_prompt[label],
            "instances": len(chunks),
            "points": int(len(merged)),
            "score": round(float(np.mean(layer_scores[label])), 3),
            "top_m": round(top, 2),
            "footprint_m2": round(float(hull.area), 2),
            "orientation_deg": None if orientation is None else round(orientation, 1),
            "tilt_deg": None if tilt is None else round(tilt, 1),
            "camera_dist_m": round(float(Point(0.0, 0.0).distance(hull)), 2),
        }
        if reference is not None and label != REFERENCE_LABEL:
            row["fence_dist_m"] = round(float(hull.distance(reference)), 2)
        rows.append(row)

    out_dir = run / "semantic_layers"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layers.json").write_text(json.dumps(rows, indent=2) + "\n")

    rendered = _render(run, frames, transform, layer_masks, out_dir)
    for row in rows:
        print(row)
    print("render:", rendered)
    return 0


def _render(
    run: Path,
    frames: list[GeometryFrame],
    transform,
    layer_masks: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
) -> bool:
    """Base scene in muted grey, every detected layer in its own colour."""
    try:
        import open3d as o3d
    except Exception:
        return False

    base_points, base_colors = [], []
    for frame in frames:
        points3d = np.load(frame.pts3d_path)
        valid = np.load(frame.valid_mask_path).astype(bool)
        finite = valid & np.isfinite(points3d).all(axis=2)
        with Image.open(frame.canonical_image_path) as image:
            rgb = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
        if rgb.shape[:2] != valid.shape:
            rgb = np.asarray(
                Image.fromarray((rgb * 255).astype(np.uint8)).resize(
                    (valid.shape[1], valid.shape[0])
                )
            ).astype(np.float32) / 255.0
        # Unsegmented scene stays a muted grey so the layers read as layers;
        # painting pixels (not appending points) avoids z-fighting duplicates.
        grey = rgb.mean(axis=2, keepdims=True)
        frame_colors = np.repeat(grey * 0.30 + 0.52, 3, axis=2)
        for label, _, tint in LAYERS:
            mask = layer_masks.get(frame.frame_id, {}).get(label)
            if mask is None:
                continue
            frame_colors[mask] = np.asarray(tint, dtype=np.float32) / 255.0
        base_points.append(transform.apply(points3d[finite]))
        base_colors.append(frame_colors[finite])

    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.vstack(base_points).astype(np.float64))
    )
    cloud.colors = o3d.utility.Vector3dVector(
        np.vstack(base_colors).astype(np.float64)
    )
    for name, front, up, zoom in (
        ("layers_perspective.png", (0.25, -0.72, 0.65), (0.0, 0.0, 1.0), 0.5),
        ("layers_topdown.png", (0.0, -0.02, 1.0), (0.0, 1.0, 0.0), 0.6),
    ):
        visualizer = o3d.visualization.Visualizer()
        if not visualizer.create_window(width=1440, height=960, visible=False):
            return False
        visualizer.add_geometry(cloud)
        options = visualizer.get_render_option()
        options.background_color = np.array([0.99, 0.99, 0.98])
        options.point_size = 2.6
        control = visualizer.get_view_control()
        control.set_front(list(front))
        control.set_lookat(cloud.get_center())
        control.set_up(list(up))
        control.set_zoom(zoom)
        visualizer.poll_events()
        visualizer.update_renderer()
        visualizer.capture_screen_image(str(out_dir / name), do_render=True)
        visualizer.destroy_window()
    return True


if __name__ == "__main__":
    raise SystemExit(main())
