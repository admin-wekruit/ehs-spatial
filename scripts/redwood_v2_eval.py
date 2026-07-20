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

from ehs_spatial.contracts import GeometryFrame
from ehs_spatial.geometry import (
    _fit_floor,
    _FloorTransform,
    _FrameData,
    _load_frame,
    _ransac_floor_plane,
    _rotation_to_positive_z,
)


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
    report["results"] = results
    for mode, _, _ in modes:
        errors = [r["error_cm"] for r in results if r["mode"] == mode and r["error_cm"] is not None]
        report[f"{mode}_mae_cm"] = round(float(np.mean(errors)), 1) if errors else None
        report[f"{mode}_max_cm"] = round(float(np.max(errors)), 1) if errors else None
    (pack / "v2_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in results:
        print(row)
    print("scale_factor:", round(transform.scale_factor, 4))
    print("MAE cm:", " | ".join(f"{mode}={report[f'{mode}_mae_cm']}" for mode, _, _ in modes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
