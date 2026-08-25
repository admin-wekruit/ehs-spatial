"""V2 real-imagery metric eval: Redwood boardroom vs laser ground truth.

Measures inter-object base gaps in the reconstructed metric scene and compares
them with laser-scan ground truth, in two scale modes:
  - camera_height: production path (floor fit + known camera height)
  - model_native:  MapAnything's own metric scale, no height anchor

Pay once (--live), then iterate free with --reuse.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ehs_spatial.contracts import GeometryFrame, Observation2D
from ehs_spatial.geometry import (
    _fit_floor,
    _FloorTransform,
    _FrameData,
    _load_frame,
    _ransac_floor_plane,
    _reconcile_entities,
    _rotation_to_positive_z,
)


# Eval-only prompt map for the annotated pack objects. Deliberately separate
# from the production PROMPT_VOCABULARY: these classes exist to give the
# binding+clustering layer a laser-ground-truth exam, not to ship.
EVAL_PROMPTS: dict[str, tuple[str, ...]] = {
    "sofa": ("sofa", "couch"),
    "plant": ("potted plant", "plant"),
    "wicker": ("wicker basket", "basket"),
    "table": ("table",),
    "desk": ("desk", "office desk"),
    "chair": ("office chair", "chair"),
    "bin": ("trash bin", "waste bin"),
}


def _pipeline_observations(
    pack: Path,
    frames: list[GeometryFrame],
    object_names: list[str],
    *,
    live: bool,
) -> tuple[list[Observation2D] | None, int]:
    """SAM masks for the annotated objects, disk-cached per (frame, prompt).

    Returns (observations, planned_live_calls). observations is None when
    uncached calls are needed but --live was not passed. First-hit synonym
    fallback mirrors production: extra prompts fire only when the canonical
    one returns nothing."""
    import base64

    from PIL import Image as PILImage

    from ehs_spatial.providers.sam3 import SAM3_ENDPOINT, decode_coco_rle

    cache_dir = pack / "sam_cache_v1"
    mask_dir = pack / "sam_masks"
    cache_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    def cached(frame_id: str, prompt: str) -> Path:
        return cache_dir / f"{frame_id}__{prompt.replace(' ', '_')}.json"

    # Count the live calls this run would make (worst case: every fallback).
    planned = 0
    for frame in frames:
        for name in object_names:
            for prompt in EVAL_PROMPTS[name]:
                if not cached(frame.frame_id, prompt).exists():
                    planned += 1
                else:
                    break
    if planned and not live:
        return None, planned

    observations: list[Observation2D] = []
    for frame in frames:
        source = Path(frame.canonical_image_path)
        with PILImage.open(source) as image:
            width, height = image.size
        payload = None
        for name in object_names:
            hits = 0
            for prompt in EVAL_PROMPTS[name]:
                cache_path = cached(frame.frame_id, prompt)
                if cache_path.exists():
                    response = json.loads(cache_path.read_text(encoding="utf-8"))
                else:
                    import fal_client

                    if payload is None:
                        payload = (
                            "data:image/png;base64,"
                            + base64.b64encode(source.read_bytes()).decode("ascii")
                        )
                    response = fal_client.subscribe(
                        SAM3_ENDPOINT,
                        arguments={
                            "image_url": payload,
                            "prompt": prompt,
                            "return_multiple_masks": True,
                            "include_scores": True,
                            "include_boxes": True,
                            "max_masks": 4,
                        },
                    )
                    cache_path.write_text(
                        json.dumps(response) + "\n", encoding="utf-8"
                    )
                rles = response.get("rle") or []
                if isinstance(rles, str):
                    rles = [rles]
                scores = response.get("scores") or [1.0] * len(rles)
                for index, rle in enumerate(rles):
                    mask = decode_coco_rle(rle, height=height, width=width)
                    if int(mask.sum()) < 40:
                        continue
                    mask_path = (
                        mask_dir
                        / f"{frame.frame_id}__{name}__{index}.png"
                    )
                    PILImage.fromarray(
                        (mask.astype(np.uint8)) * 255
                    ).save(mask_path)
                    observations.append(
                        Observation2D(
                            observation_id=(
                                f"{name}-{frame.frame_id}-{index}"
                            ),
                            frame_id=frame.frame_id,
                            label=name,
                            instance_id=f"{name}-{index}",
                            mask_path=str(mask_path),
                            score=float(scores[index])
                            if index < len(scores)
                            else 1.0,
                            bbox=(0, 0, 1, 1),
                            source_prompt=prompt,
                        )
                    )
                    hits += 1
                if hits:
                    break
    return observations, planned


def _score_pipeline_mode(
    frames_by_id: dict,
    frame_data: dict[str, _FrameData],
    transform: _FloorTransform,
    observations: list[Observation2D],
    annotations: dict,
    frames: list[GeometryFrame],
) -> tuple[list[dict], dict]:
    from shapely.geometry import Polygon

    entities, entity_warnings = _reconcile_entities(
        frames_by_id, observations, frame_data, transform
    )
    gt_objects = _object_points(frames, frame_data, annotations, transform)
    gt_centroids = {
        name: np.median(points[:, :2], axis=0)
        for name, points in gt_objects.items()
    }

    def match(name: str):
        candidates = [e for e in entities if e.label == name]
        if not candidates or name not in gt_centroids:
            return None
        target = gt_centroids[name]
        return min(
            candidates,
            key=lambda e: float(
                np.hypot(e.centroid_xyz[0] - target[0], e.centroid_xyz[1] - target[1])
            ),
        )

    rows = []
    for pair in annotations["pairs"]:
        name_a, name_b = pair["pair_id"].split("-")
        ea, eb = match(name_a), match(name_b)
        if ea is None or eb is None:
            predicted = None
        else:
            pa, pb = Polygon(ea.footprint_xy), Polygon(eb.footprint_xy)
            predicted = 0.0 if pa.intersects(pb) else float(pa.distance(pb))
        gt = pair["gt_distance_m"]
        rows.append(
            {
                "mode": "pipeline",
                "pair": pair["pair_id"],
                "gt_m": gt,
                "predicted_m": None if predicted is None else round(predicted, 4),
                "error_cm": None
                if predicted is None
                else round(abs(predicted - gt) * 100, 1),
            }
        )
    meta = {
        "pipeline_entity_counts": dict(
            sorted(
                {
                    name: sum(1 for e in entities if e.label == name)
                    for name in {e.label for e in entities}
                }.items()
            )
        ),
        "pipeline_entity_warnings": entity_warnings,
    }
    return rows, meta


def _fit_floor_no_gate(
    frames_by_id: dict,
    frame_data: dict[str, _FrameData],
    camera_height_m: float,
) -> tuple[_FloorTransform | None, float | None]:
    """Eval-side floor fit: same math as production, but the camera-height MAD
    safety gate is reported as a metric instead of aborting the measurement."""
    points = []
    for fid in sorted(frames_by_id):
        d = frame_data[fid]
        finite = d.valid & np.isfinite(d.points).all(axis=2)
        points.append(d.points[finite])
    cloud = np.vstack(points)
    if len(cloud) > 60_000:
        cloud = cloud[:: int(np.ceil(len(cloud) / 60_000))]
    centers = np.asarray(
        [np.asarray(f.camera_to_world)[:3, 3] for f in frames_by_id.values()]
    )
    ups = [
        -np.asarray(f.camera_to_world, dtype=float)[:3, 1]
        for f in frames_by_id.values()
    ]
    up = np.mean(ups, axis=0)
    up /= np.linalg.norm(up)
    scale = None
    for _ in range(2):
        threshold = 0.03 if scale is None else 0.03 / scale
        inliers = _ransac_floor_plane(cloud, centers, up, threshold)
        if inliers is None or len(inliers) < 3:
            return None, None
        ip = cloud[inliers]
        ctr = ip.mean(axis=0)
        _, _, vt = np.linalg.svd(ip - ctr, full_matrices=False)
        normal = vt[-1] * np.sign(vt[-1] @ up)
        offset = -float(normal @ ctr)
        heights = centers @ normal + offset
        if not np.isfinite(heights).all() or np.any(heights <= 0):
            return None, None
        scale = camera_height_m / float(np.median(heights))
    scaled = heights * scale
    mad = float(np.median(np.abs(scaled - np.median(scaled))))
    return (
        _FloorTransform(
            plane=tuple(float(v) for v in (*normal, offset)),
            scale_factor=float(scale),
            rotation=_rotation_to_positive_z(normal),
            origin=-offset * normal,
        ),
        mad,
    )


MOGE_VERSION = (
    "jasonod888/moge2:"
    "daa7a9329b3d6bb513f3a20def9451b6e3cf234afe2bb60ba38f96ecd33c81f7"
)


def _moge_scale(pack: Path, frames: list[GeometryFrame], *, live: bool):
    """Scale from a second metric model instead of the operator's tape.

    Per frame: ratio of MoGe-2's median camera-frame range to MapAnything's
    over the same view. The pack scale is the median across frames. Cached
    per frame, so re-scoring is free."""
    import base64
    import tempfile
    import time

    cache_dir = pack / "moge_scale"
    ratios = []
    for frame in frames:
        cache = cache_dir / f"{frame.frame_id}.json"
        if cache.exists():
            ratios.append(json.loads(cache.read_text())["ratio"])
            continue
        if not live:
            return None, len(frames) - len(ratios)

        data = _load_frame(frame)
        finite = data.valid & np.isfinite(data.points).all(axis=2)
        world = data.points[finite]
        camera_to_world = np.asarray(frame.camera_to_world, dtype=float)
        camera = (
            np.linalg.inv(camera_to_world) @ np.c_[world, np.ones(len(world))].T
        )[:3].T
        native = float(np.median(np.linalg.norm(camera, axis=1)))
        if not native:
            continue

        import open3d as o3d
        import replicate

        payload = "data:image/png;base64," + base64.b64encode(
            Path(frame.canonical_image_path).read_bytes()
        ).decode("ascii")
        for attempt in range(5):
            try:
                output = replicate.run(
                    MOGE_VERSION, input={"image": payload, "fp16": True}, wait=False
                )
                break
            except Exception as exc:
                if not any(
                    code in str(exc)
                    for code in ("429", "throttled", "500", "502", "503")
                ):
                    raise
                time.sleep(10 * (attempt + 1))
        else:
            continue
        with tempfile.NamedTemporaryFile(suffix=".ply") as handle:
            handle.write(output["pointcloud_ply"].read())
            handle.flush()
            cloud = np.asarray(o3d.io.read_point_cloud(handle.name).points)
        if not len(cloud):
            continue
        ratio = float(np.median(np.linalg.norm(cloud, axis=1))) / native
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"ratio": ratio, "native_m": native}) + "\n")
        ratios.append(ratio)
    if not ratios:
        return None, 0
    return float(np.median(ratios)), 0


def _frames_from_disk(geometry_dir: Path) -> list[GeometryFrame]:
    frames = []
    for frame_dir in sorted((geometry_dir / "frames").iterdir()):
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


def _object_points(
    frames: list[GeometryFrame],
    frame_data: dict[str, _FrameData],
    annotations: dict,
    transform: _FloorTransform,
) -> dict[str, np.ndarray]:
    source_w = annotations["intrinsics"]["width"]
    source_h = annotations["intrinsics"]["height"]
    frame_ids = {name.split(".")[0]: frame for name, frame in
                 zip(annotations["frames"], frames)}
    pooled: dict[str, list[np.ndarray]] = {}
    for stem, boxes in annotations["object_boxes_px"].items():
        frame = frame_ids[stem]
        data = frame_data[frame.frame_id]
        height, width = data.shape
        sx, sy = width / source_w, height / source_h
        finite = data.valid & np.isfinite(data.points).all(axis=2)
        for name, (x1, y1, x2, y2) in boxes.items():
            box = np.zeros(data.shape, dtype=bool)
            box[int(y1 * sy) : int(y2 * sy), int(x1 * sx) : int(x2 * sx)] = True
            points = data.points[box & finite]
            if len(points):
                pooled.setdefault(name, []).append(transform.apply(points))
    return {
        name: np.vstack(chunks) for name, chunks in pooled.items() if chunks
    }


def _base_gap(a: np.ndarray, b: np.ndarray) -> float | None:
    band_a = a[(a[:, 2] > 0.03) & (a[:, 2] < 0.45)][:, :2]
    band_b = b[(b[:, 2] > 0.03) & (b[:, 2] < 0.45)][:, :2]
    if len(band_a) < 50 or len(band_b) < 50:
        return None
    band_a = band_a[:: max(1, len(band_a) // 4000)]
    band_b = band_b[:: max(1, len(band_b) // 4000)]
    nn = np.sqrt(((band_a[:, None, :] - band_b[None, :, :]) ** 2).sum(-1).min(axis=1))
    return float(np.percentile(nn, 0.2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", default="outputs/redwood_v2")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--moge-anchor",
        action="store_true",
        help="add a mode scaled by MoGe-2 instead of the known camera height",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="also score distances through SAM masks + entity clustering "
        "(the production binding path) instead of annotation boxes only",
    )
    args = parser.parse_args(argv)

    pack = Path(args.pack).resolve()
    annotations = json.loads((pack / "annotations.json").read_text(encoding="utf-8"))
    geometry_dir = pack / "geometry"

    if args.reuse and (geometry_dir / "frames").is_dir():
        frames = _frames_from_disk(geometry_dir)
    elif args.live:
        from ehs_spatial.providers.map_anything import MapAnythingAdapter

        images = [str(pack / "input" / name) for name in annotations["frames"]]
        frames, _ = MapAnythingAdapter().run(images, geometry_dir)
    else:
        print("pass --live to spend one MapAnything call, or --reuse", file=sys.stderr)
        return 2

    frames_by_id = {frame.frame_id: frame for frame in frames}
    frame_data = {fid: _load_frame(frame) for fid, frame in frames_by_id.items()}
    camera_height = float(annotations["camera_height_m"])
    report: dict = {"camera_height_m": camera_height}

    per_frame = annotations.get("camera_height_per_frame")
    if per_frame:
        # Per-view height anchoring: rescale each frame's depth along the rays
        # from its own camera so that its independently fitted floor sits at
        # the frame's known camera height. Attacks per-view scale drift that a
        # single global factor cannot fix.
        stems = [name.split(".")[0] for name in annotations["frames"]]
        stem_by_fid = dict(zip(sorted(frames_by_id), stems))
        anchored = {}
        for fid, data in frame_data.items():
            frame = frames_by_id[fid]
            camera = np.asarray(frame.camera_to_world, dtype=float)
            center = camera[:3, 3]
            up = -camera[:3, :3][:, 1]
            finite = data.valid & np.isfinite(data.points).all(axis=2)
            pts = data.points[finite][::7]
            inl = _ransac_floor_plane(pts, center[None, :], up, 0.03)
            if inl is None or len(inl) < 50:
                anchored[fid] = data
                continue
            ip = pts[inl]
            ctr = ip.mean(axis=0)
            _, _, vt = np.linalg.svd(ip - ctr, full_matrices=False)
            n = vt[-1] * np.sign(vt[-1] @ up)
            raw_height = float(center @ n - n @ ctr)
            true_height = float(per_frame[stem_by_fid[fid]])
            if raw_height <= 0:
                anchored[fid] = data
                continue
            s = true_height / raw_height
            anchored[fid] = (s, data)
            print(f"{fid}: raw_h={raw_height:.3f} true={true_height:.3f} s={s:.3f}")
        scales = np.asarray([s for s, _ in anchored.values()])
        median_scale = float(np.median(scales))
        rescaled_data = {}
        for fid, value in anchored.items():
            s, data = value if isinstance(value, tuple) else (median_scale, value)
            # A per-frame fit that disagrees wildly with the group caught
            # furniture, not floor: fall back to the group median scale.
            if not (0.67 <= s / median_scale <= 1.5):
                s = median_scale
            center = np.asarray(frames_by_id[fid].camera_to_world)[:3, 3]
            rescaled_data[fid] = _FrameData(
                points=center + (data.points - center) * s,
                valid=data.valid,
                shape=data.shape,
            )
        anchored = rescaled_data
        anchored_transform, anchored_mad = _fit_floor_no_gate(
            frames_by_id, anchored, camera_height
        )
        anchored_warnings = [f"anchored_mad={anchored_mad}"]
        report["anchored_mad_m"] = anchored_mad
    else:
        anchored, anchored_transform, anchored_warnings = None, None, ["no per-frame heights"]
    transform, warnings = _fit_floor(frames_by_id, frame_data, camera_height)
    report["warnings"] = warnings
    if transform is None:
        transform, joint_mad = _fit_floor_no_gate(
            frames_by_id, frame_data, camera_height
        )
        report["production_gate_rejected"] = warnings
        report["joint_mad_m"] = joint_mad
    if transform is None:
        report["passed"] = False
        (pack / "v2_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("floor fit failed:", warnings)
        return 1

    report["scale_factor"] = transform.scale_factor
    native = _FloorTransform(
        plane=transform.plane,
        scale_factor=1.0,
        rotation=transform.rotation,
        origin=transform.origin,
    )
    modes = [
        ("camera_height", transform, frame_data),
        ("model_native", native, frame_data),
    ]
    if args.moge_anchor:
        moge_ratio, pending = _moge_scale(pack, frames, live=args.live)
        if moge_ratio is None:
            print(
                f"moge anchor needs {pending} MoGe-2 call(s); pass --live",
                file=sys.stderr,
            )
            return 2
        report["moge_scale_factor"] = moge_ratio
        report["camera_height_scale_factor"] = transform.scale_factor
        print(
            f"moge scale {moge_ratio:.4f} vs camera-height scale "
            f"{transform.scale_factor:.4f} "
            f"(ratio {moge_ratio / transform.scale_factor:.3f})"
        )
        modes.append(
            (
                "moge_anchor",
                _FloorTransform(
                    plane=transform.plane,
                    scale_factor=moge_ratio,
                    rotation=transform.rotation,
                    origin=transform.origin,
                ),
                frame_data,
            )
        )
    if anchored_transform is not None:
        modes.append(("per_frame_anchor", anchored_transform, anchored))
    else:
        report["per_frame_anchor_warnings"] = anchored_warnings
    results = []
    for mode, tf, mode_data in modes:
        objects = _object_points(frames, mode_data, annotations, tf)
        for pair in annotations["pairs"]:
            name_a, name_b = pair["pair_id"].split("-")
            gap = (
                _base_gap(objects[name_a], objects[name_b])
                if name_a in objects and name_b in objects
                else None
            )
            gt = pair["gt_distance_m"]
            results.append(
                {
                    "mode": mode,
                    "pair": pair["pair_id"],
                    "gt_m": gt,
                    "predicted_m": None if gap is None else round(gap, 4),
                    "error_cm": None if gap is None else round(abs(gap - gt) * 100, 1),
                }
            )
    if args.pipeline:
        object_names = sorted(
            {
                name
                for boxes in annotations["object_boxes_px"].values()
                for name in boxes
            }
        )
        observations, planned = _pipeline_observations(
            pack, frames, object_names, live=args.live
        )
        if observations is None:
            print(
                f"pipeline mode needs {planned} uncached SAM call(s); "
                "pass --live to spend them (cached reruns are free)",
                file=sys.stderr,
            )
            return 2
        pipeline_rows, pipeline_meta = _score_pipeline_mode(
            frames_by_id, frame_data, transform, observations, annotations, frames
        )
        results.extend(pipeline_rows)
        report.update(pipeline_meta)

    report["results"] = results
    mode_names = list(dict.fromkeys(r["mode"] for r in results))
    for mode in mode_names:
        errors = [r["error_cm"] for r in results if r["mode"] == mode and r["error_cm"] is not None]
        report[f"{mode}_mae_cm"] = round(float(np.mean(errors)), 1) if errors else None
        report[f"{mode}_max_cm"] = round(float(np.max(errors)), 1) if errors else None
    (pack / "v2_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in results:
        print(row)
    print("scale_factor:", round(transform.scale_factor, 4))
    print("MAE cm:", " | ".join(f"{mode}={report[f'{mode}_mae_cm']}" for mode in mode_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
