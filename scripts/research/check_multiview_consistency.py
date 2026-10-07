"""Read-only pinhole/depth diagnostics for the four cached BOR1 views; no inference.

Run from panoptes-serving:
  .venv/bin/python scripts/research/check_multiview_consistency.py --self-test
  .venv/bin/python scripts/research/check_multiview_consistency.py

The integer (column, row) grid is tested against the stored K and OpenCV c2w.
Depth agreement measures internal consistency, not accuracy against ground truth.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RUN = "user-bor1-02"
THRESHOLDS = (0.01, 0.02, 0.05, 0.10)
OCCLUSION_REL = 0.03
EDGE_REL = 0.08
FRAME_IDS = tuple(f"frame_{i:04d}" for i in range(1, 5))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def stats(values):
    values = np.asarray(values)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .90)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def camera_points(points, c2w):
    # OpenCV c2w maps column vectors by R @ p + t; numpy points are rows here.
    return (points - c2w[:3, 3]) @ c2w[:3, :3]


def frame_arrays(name, points, mask, c2w, K):
    points, c2w, K = (np.asarray(x, dtype=np.float64) for x in (points, c2w, K))
    mask = np.asarray(mask)
    if points.ndim != 3 or points.shape[-1] != 3 or mask.shape != points.shape[:2] or mask.dtype != bool:
        raise ValueError(f"{name}: expected HxWx3 points and aligned boolean mask")
    if c2w.shape != (4, 4) or K.shape != (3, 3) or not np.isfinite(c2w).all() or not np.isfinite(K).all():
        raise ValueError(f"{name}: invalid camera matrices")
    R = c2w[:3, :3]
    if not (np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-5, rtol=0)
            and np.allclose(R.T @ R, np.eye(3), atol=1e-4, rtol=0)
            and abs(np.linalg.det(R) - 1) < 1e-4):
        raise ValueError(f"{name}: c2w is not a proper rigid transform")
    if not np.allclose(K[2], [0, 0, 1], atol=1e-6, rtol=0) or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError(f"{name}: invalid pinhole intrinsics")
    finite = np.isfinite(points).all(axis=-1)
    with np.errstate(invalid="ignore"):
        depth = camera_points(points, c2w)[..., 2]
    nonzero = np.any(points != 0, axis=-1)
    usable = mask & finite & nonzero & (depth > 0)
    return {"name": name, "points": points, "mask": mask, "c2w": c2w, "K": K,
            "depth": depth, "usable": usable, "finite": finite,
            "analysis_region": np.ones(mask.shape, dtype=bool)}


def load_frame(directory):
    files = {key: directory / f"{key}.npy" for key in ("pts3d", "valid_mask", "camera_to_world", "intrinsics")}
    arrays = {key: np.load(path, allow_pickle=False) for key, path in files.items()}
    frame = frame_arrays(directory.name, arrays["pts3d"], arrays["valid_mask"],
                         arrays["camera_to_world"], arrays["intrinsics"])
    image = directory / "canonical.png"
    with Image.open(image) as source:
        if source.size != frame["mask"].shape[::-1]:
            raise ValueError(f"{directory}: image and point grid differ")
    frame["sha256"] = {key: digest(path) for key, path in {**files, "canonical.png": image}.items()}
    return frame


def project(points, frame):
    camera = camera_points(points, frame["c2w"])
    homogeneous = camera @ frame["K"].T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = homogeneous[:, :2] / homogeneous[:, 2:3]
    return uv, camera[:, 2]


def self_reprojection(frame):
    height, width = frame["mask"].shape
    y, x = np.nonzero(frame["usable"])
    uv, _ = project(frame["points"][y, x], frame)
    error = np.linalg.norm(uv - np.column_stack((x, y)), axis=1)
    report = {"frame": frame["name"], "image_pixels": height * width,
              "analysis_region_pixels": int(frame["analysis_region"].sum()),
              "native_mask_pixels": int(frame["mask"].sum()),
              "native_mask_coverage": float(frame["mask"].mean()),
              "native_mask_finite_pixels": int((frame["mask"] & frame["finite"]).sum()),
              "native_mask_zero_filled_pixels": int((frame["mask"] & np.all(frame["points"] == 0, axis=-1)).sum()),
              "usable_positive_depth_pixels": int(y.size), "usable_coverage": float(frame["usable"].mean()),
              "usable_fraction_of_analysis_region": fraction(y.size, frame["analysis_region"].sum()),
              "pixel_error": stats(error),
              "within_pixels": {str(t): fraction((error <= t).sum(), y.size) for t in (.5, 1., 2.)},
              "projected_in_image_fraction": fraction(((uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
                  & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)).sum(), y.size),
              "source_sha256": frame.get("sha256")}
    return report, error


def error_report(relative, absolute, source_pixels):
    behind = relative > OCCLUSION_REL
    visible = ~behind
    return {"absolute_relative_depth_error": stats(np.abs(relative)),
            "signed_relative_depth_error": stats(relative),
            "absolute_depth_error_native_units": stats(absolute),
            "occluded_or_behind_target_conflict_pixels": int(behind.sum()),
            "occluded_or_behind_target_conflict_fraction": fraction(behind.sum(), relative.size),
            "in_front_of_target_conflict_pixels": int((relative < -OCCLUSION_REL).sum()),
            "in_front_of_target_conflict_fraction": fraction((relative < -OCCLUSION_REL).sum(), relative.size),
            "visibility_filtered_pixels": int(visible.sum()),
            "visibility_filtered_source_coverage": fraction(visible.sum(), source_pixels),
            "visibility_filtered_absolute_relative_depth_error": stats(np.abs(relative[visible])),
            "support": {str(t): {"pixels": int((np.abs(relative) <= t).sum()),
                "fraction_of_overlap": fraction((np.abs(relative) <= t).sum(), relative.size),
                "fraction_of_source_usable": fraction((np.abs(relative) <= t).sum(), source_pixels)} for t in THRESHOLDS}}


def cross_view(source, target):
    height, width = target["mask"].shape
    source_indices = np.flatnonzero(source["usable"])
    uv, z = project(source["points"].reshape(-1, 3)[source_indices], target)
    positive = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
    inside = positive & (uv[:, 0] >= 0) & (uv[:, 0] < width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < height - 1)
    report = {"source": source["name"], "target": target["name"],
              "state_pair": "same_robot_state" if ((source["name"] in FRAME_IDS[:2]) == (target["name"] in FRAME_IDS[:2])) else "cross_robot_state",
              "source_image_pixels": int(source["mask"].size),
              "source_analysis_region_pixels": int(source["analysis_region"].sum()),
              "source_usable_pixels": int(source_indices.size),
              "target_positive_depth_pixels": int(positive.sum()),
              "projected_in_bilinear_bounds_pixels": int(inside.sum())}
    uv, z, indices = uv[inside], z[inside], source_indices[inside]
    x, y = np.floor(uv).astype(int).T
    # ponytail: four native-valid neighbours, no depth hole filling or correspondence search.
    corner_depth = np.stack([target["depth"][y, x], target["depth"][y, x + 1],
                             target["depth"][y + 1, x], target["depth"][y + 1, x + 1]], axis=1)
    valid = (target["usable"][y, x] & target["usable"][y, x + 1]
             & target["usable"][y + 1, x] & target["usable"][y + 1, x + 1])
    corners, uv, x, y, z, indices = (a[valid] for a in (corner_depth, uv, x, y, z, indices))
    dx, dy = uv[:, 0] - x, uv[:, 1] - y
    weights = np.column_stack(((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy))
    sampled = (corners * weights).sum(axis=1)
    relative = (z - sampled) / sampled
    absolute = np.abs(z - sampled)
    smooth = (corners.max(axis=1) - corners.min(axis=1)) / corners.min(axis=1) <= EDGE_REL
    report.update({"valid_overlap_pixels": int(valid.sum()),
                   "valid_overlap_fraction_of_source_usable": fraction(valid.sum(), source_indices.size),
                   "valid_overlap_fraction_of_source_image": fraction(valid.sum(), source["mask"].size),
                   "valid_overlap_fraction_of_source_analysis_region": fraction(valid.sum(), source["analysis_region"].sum()),
                   "all_valid_overlap": error_report(relative, absolute, source_indices.size),
                   "target_depth_edge_rejected_pixels": int((~smooth).sum()),
                   "smooth_overlap_pixels": int(smooth.sum()),
                   "smooth_overlap_fraction_of_source_usable": fraction(smooth.sum(), source_indices.size),
                   "smooth_overlap": error_report(relative[smooth], absolute[smooth], source_indices.size)})
    supported_indices = indices[smooth & (np.abs(relative) <= .02)]
    pools = {"all_relative": relative, "all_absolute": absolute,
             "smooth_relative": relative[smooth], "smooth_absolute": absolute[smooth]}
    return report, pools, supported_indices


def pooled_cross(items):
    source_pixels = sum(item[0]["source_usable_pixels"] for item in items)
    return {"directed_pairs": len(items), "source_usable_pair_pixels": source_pixels,
            "valid_overlap_fraction_of_source_usable": fraction(sum(item[0]["valid_overlap_pixels"] for item in items), source_pixels),
            **{name: error_report(np.concatenate([item[1][f"{prefix}_relative"] for item in items]),
                                 np.concatenate([item[1][f"{prefix}_absolute"] for item in items]), source_pixels)
               for name, prefix in (("all_valid_overlap", "all"), ("smooth_overlap", "smooth"))}}


def diagnose(frames):
    self_results = [self_reprojection(frame) for frame in frames]
    pairs, support = [], []
    for source in frames:
        counts = np.zeros(source["mask"].size, dtype=np.uint8)
        for target in frames:
            if target is source:
                continue
            pair = cross_view(source, target)
            counts[pair[2]] += 1
            pairs.append(pair)
        support.append({"frame": source["name"], "image_pixels": int(counts.size),
                        "analysis_region_pixels": int(source["analysis_region"].sum()),
                        "usable_pixels": int(source["usable"].sum()),
                        "two_percent_depth_support": {str(n): {"pixels": int((counts >= n).sum()),
                            "fraction_of_image": float((counts >= n).mean()),
                            "fraction_of_analysis_region": fraction((counts >= n).sum(), source["analysis_region"].sum()),
                            "fraction_of_usable": fraction((counts >= n).sum(), source["usable"].sum())} for n in (1, 2, 3)}})
    groups = {"all": pairs,
              "frames_1_2": [p for p in pairs if p[0]["source"] in FRAME_IDS[:2] and p[0]["target"] in FRAME_IDS[:2]],
              "frames_3_4": [p for p in pairs if p[0]["source"] in FRAME_IDS[2:] and p[0]["target"] in FRAME_IDS[2:]],
              "cross_robot_state": [p for p in pairs if p[0]["state_pair"] == "cross_robot_state"]}
    return {"self_reprojection": [r[0] for r in self_results],
            "pooled_self_pixel_error": stats(np.concatenate([r[1] for r in self_results])),
            "directed_pairs": [p[0] for p in pairs],
            "pair_groups": {name: pooled_cross(group) for name, group in groups.items() if group},
            "per_frame_multiview_support": support,
            "pooled_multiview_support": {str(n): {"pixels": sum(s["two_percent_depth_support"][str(n)]["pixels"] for s in support),
                "fraction_of_image": fraction(sum(s["two_percent_depth_support"][str(n)]["pixels"] for s in support), sum(s["image_pixels"] for s in support)),
                "fraction_of_analysis_region": fraction(sum(s["two_percent_depth_support"][str(n)]["pixels"] for s in support), sum(s["analysis_region_pixels"] for s in support)),
                "fraction_of_usable": fraction(sum(s["two_percent_depth_support"][str(n)]["pixels"] for s in support), sum(s["usable_pixels"] for s in support))} for n in (1, 2, 3)}}


def restrict(frames, masks):
    return [{**frame, "analysis_region": frame["analysis_region"] & mask,
             "usable": frame["usable"] & mask} for frame, mask in zip(frames, masks, strict=True)]


def region_diagnostics(frames, selection, geometry_root, candidate_name):
    masks = [np.zeros(frame["mask"].shape, dtype=bool) for frame in frames]
    x0, y0, x1, y1 = selection["content_rect"]
    for mask in masks:
        mask[y0:y1, x0:x1] = True
    content = restrict(frames, masks)
    result = diagnose(content)
    result["full_canonical_including_padding"] = diagnose(frames)
    for frame, mask in zip(frames, masks, strict=True):
        for x0, y0, x1, y1 in selection["exclude_rects"][frame["name"]]:
            mask[y0:y1, x0:x1] = False
    result["manual_dynamic_exclusion_diagnostic"] = {
        "scope": "Frozen image rectangles only, same masks for every candidate. Frame 1 retains its robot/load; frames 2-4 exclude manually marked dynamic regions. Only 3<->4 has both dynamic regions excluded; these rectangles are not instance ground truth. No confidence or 3D crop threshold applied.",
        **diagnose(restrict(frames, masks))}
    if candidate_name == "pi3x-01":
        if Path(selection["source_geometry"]).resolve() != geometry_root.resolve():
            raise ValueError("Frozen fusion selection's world-space crop belongs to a different cache")
        confidence_hashes = {}
        low, high = np.asarray(selection["crop_box_camera_min"]), np.asarray(selection["crop_box_camera_max"])
        reference = next(frame for frame in frames if frame["name"] == selection["reference_frame"])
        for frame, mask in zip(frames, masks, strict=True):
            path = geometry_root / "frames" / frame["name"] / "conf.npy"
            conf = np.load(path, allow_pickle=False)
            if conf.shape != mask.shape:
                raise ValueError("Confidence shape differs from image grid")
            confidence_hashes[frame["name"]] = digest(path)
            in_reference = camera_points(frame["points"], reference["c2w"])
            mask &= np.isfinite(conf) & (conf >= selection["minimum_confidence"])
            mask &= ((in_reference >= low) & (in_reference <= high)).all(axis=-1)
        result["frozen_fusion_selection_diagnostic"] = {
            "scope": "Pi3X only: exact frozen content/rectangle/confidence/reference-camera crop. Frame 1 still contains the chosen dynamic state. Smaller coverage must be read with error, not mistaken for a model improvement.",
            "confidence_sha256": confidence_hashes, **diagnose(restrict(frames, masks))}
    return result


def self_test():
    theta = .37
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    K = np.array([[32., 0, 15.5], [0, 32., 15.5], [0, 0, 1.]])
    y, x = np.indices((32, 32))
    local = np.stack(((x - 15.5) / 32, (y - 15.5) / 32, np.ones_like(x)), axis=-1) * 4
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = R, [1., -.2, 2.]
    other = c2w.copy()
    other[:3, 3] += R @ np.array([.25, 0, 0])
    mask = np.ones((32, 32), dtype=bool)
    a = frame_arrays(FRAME_IDS[0], local @ R.T + c2w[:3, 3], mask, c2w, K)
    b = frame_arrays(FRAME_IDS[1], local @ R.T + other[:3, 3], mask, other, K)
    assert self_reprojection(a)[0]["pixel_error"]["max"] < 1e-12
    assert self_reprojection(b)[0]["pixel_error"]["max"] < 1e-12
    pair, _, supported = cross_view(a, b)
    assert pair["all_valid_overlap"]["absolute_relative_depth_error"]["max"] < 1e-12
    assert 0 < supported.size < mask.size  # translated camera has incomplete overlap
    partial_mask = mask.copy()
    partial_mask[8:16, 8:16] = False
    partial = frame_arrays(FRAME_IDS[1], b["points"], partial_mask, other, K)
    assert cross_view(a, partial)[0]["valid_overlap_pixels"] < pair["valid_overlap_pixels"]
    restricted = restrict([a], [partial_mask])[0]
    assert restricted["mask"].all() and restricted["analysis_region"].sum() == partial_mask.sum()
    assert restricted["usable"].sum() == partial_mask.sum()
    empty = frame_arrays(FRAME_IDS[1], b["points"], ~mask, other, K)
    assert cross_view(a, empty)[0]["all_valid_overlap"]["absolute_relative_depth_error"]["count"] == 0
    zero_filled = a["points"].copy()
    zero_filled[0, 0] = 0
    sentinel = frame_arrays(FRAME_IDS[0], zero_filled, mask, c2w, K)
    assert sentinel["mask"][0, 0] and not sentinel["usable"][0, 0]
    nearer = frame_arrays(FRAME_IDS[1], (local * .5) @ R.T + other[:3, 3], mask, other, K)
    behind, _, support = cross_view(a, nearer)
    assert behind["all_valid_overlap"]["occluded_or_behind_target_conflict_fraction"] == 1
    assert behind["all_valid_overlap"]["visibility_filtered_pixels"] == 0 and support.size == 0
    ahead, _, support = cross_view(nearer, a)
    assert ahead["all_valid_overlap"]["in_front_of_target_conflict_fraction"] == 1 and support.size == 0
    assert ahead["all_valid_overlap"]["visibility_filtered_absolute_relative_depth_error"]["median"] > .49
    print("Self-test passed: rotated/translated c2w, pixel identity, overlap/mask coverage, behind and in-front conflicts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/candidate-evaluation/workcell-reconstruction-01/consistency.json")
    parser.add_argument("--selection", type=Path, default=ROOT / "outputs/candidate-evaluation/workcell-reconstruction-01/selection.json")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    roots = {"baseline": ROOT / "runs" / RUN / "geometry",
             **{candidate: ROOT / "outputs/candidate-evaluation" / candidate / "runs" / RUN / "geometry"
                for candidate in ("mapanything-apache-01", "pi3x-01")}}
    selection = json.loads(args.selection.read_text())
    if selection["content_rect"] != [63, 0, 455, 518]:
        raise ValueError("Expected verified BOR1 content rectangle [63,0,455,518]")
    result = {"run_id": RUN, "created_at": datetime.now(timezone.utc).isoformat(),
              "script_sha256": digest(Path(__file__)),
              "selection": {"path": str(args.selection.resolve()), "sha256": digest(args.selection), "content": selection},
              "scope": "Primary results exclude verified white padding (retain x=63..454) but include changing robot/load state: frames 1/2 and 3/4 are separate states. Raw padded, manually excluded and Pi3X fusion-selection diagnostics are separate. Internal consistency is not ground-truth accuracy or proof of metric scale.",
              "convention": "Stored world pts3d; OpenCV camera +x right/+y down/+z forward. c2w columns: p_world=R@p_camera+t; row-vector inverse=(p_world-t)@R. K projects onto integer 0-based column,row pixels.",
              "method": {"sampling": "Every native-mask, finite, nonzero, positive-depth source pixel; all-zero vectors are wrapper-invalid sentinels even when mask says true. Bilinear target camera depth requires all four native-valid nonzero positive-depth corners; no pose/K refit, search, alignment, hole filling or rescaling.",
                         "support_denominator": "All smooth valid overlap before any asymmetric occlusion filtering; coverage also measured against all source usable/image pixels.",
                         "edge_relative_limit": EDGE_REL, "edge_definition": "(max-min)/min of four target depths",
                         "occlusion_relative_limit": OCCLUSION_REL,
                         "occlusion_caveat": "Source projected >3% behind target is possibly occluded OR inconsistent. Report raw error and fraction before exclusion. In-front conflicts remain in visibility-filtered errors. Dynamic motion also contributes.",
                         "relative_error": "(projected source camera-z - sampled target camera-z) / sampled target camera-z",
                         "absolute_depth_units": "Each cache's native units; no independent metric calibration. Relative errors compare without scale alignment.",
                         "support_relative_thresholds": THRESHOLDS}, "candidates": {}}
    canonical = None
    for name, directory in roots.items():
        print(f"Checking {name}: {directory}", flush=True)
        frames = [load_frame(directory / "frames" / frame_id) for frame_id in FRAME_IDS]
        hashes = [frame["sha256"]["canonical.png"] for frame in frames]
        if canonical is not None and hashes != canonical:
            raise ValueError("Canonical input images differ between candidates")
        canonical = hashes
        candidate = {"geometry_root": str(directory), **region_diagnostics(frames, selection, directory, name)}
        manifest = directory / "candidate_manifest.json"
        if manifest.exists():
            candidate["candidate_manifest"] = {"path": str(manifest), "sha256": digest(manifest), "content": json.loads(manifest.read_text())}
        result["candidates"][name] = candidate
        print(f"  self p95={candidate['pooled_self_pixel_error']['p95']:.4f}px; >=1 support/image={candidate['pooled_multiview_support']['1']['fraction_of_image']:.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
