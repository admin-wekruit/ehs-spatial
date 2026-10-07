"""Read-only run metrics; no inference, ground-truth guesses, or pass verdicts.

  uv run python scripts/candidate_metrics.py --run runs/real-clean-01 \
      --output outputs/real-clean-01-metrics.json

Geometry formulas and eligibility match tests/test_geometry_invariants.py.
The legacy reprojection field mean_dv_frac is a within-object median, not
a mean. Neither that self-consistency score nor MoGe ratios are metric GT.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np


CEILINGS = {"real-clean-01": .10, "real-clean-02": .25, "real-clean-03": .10}
CONTACT_FAMILY = ("fence", "guard", "barrier", "rail", "curtain", "partition", "panel")


def _read(path, expected):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, expected):
        raise ValueError(f"{path}: expected {expected.__name__}")
    return value


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(
        value, bool
    ) and math.isfinite(value) else None


def _maximum(values):
    observed = [v for v in values if v is not None]
    return max(observed) if observed else None


def _key(obj):
    return {"frame_id": obj.get("frame_id", obj.get("frame")),
            "label": obj.get("label"), "instance": obj.get("instance"),
            "refine_slug": obj.get("refine_slug")}


def _rectangle(obj):
    if not obj.get("rect_snapped"):
        return None
    rect = np.asarray(obj["rect_snapped"], float)
    if rect.shape != (4, 2) or not np.isfinite(rect).all():
        raise ValueError(f"invalid rectangle for {_key(obj)}")
    return rect


def _angle(rect):
    edge1, edge2 = rect[1] - rect[0], rect[2] - rect[1]
    if np.linalg.norm(edge1) < np.linalg.norm(edge2):
        edge1 = edge2
    return float(np.degrees(np.arctan2(edge1[1], edge1[0]))) % 180.


def _area(points):
    if points is None or len(points) < 3:
        return None
    xy = np.asarray(points, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("invalid footprint polygon")
    return float(abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1))
                     - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))) / 2)


def _geometry(objects, theta):
    groups = {}
    thin = []
    for obj in objects or []:
        if obj.get("footprint_method") == "guard-line" and obj.get("guard_chain") is not None:
            groups.setdefault(obj["guard_chain"], []).append(obj)
        if obj.get("footprint_method") in ("guard-line", "contact-edge") and "fence" in obj["label"]:
            rect = _rectangle(obj)
            thin.append({"key": _key(obj), "short_side_m": None if rect is None else min(
                float(np.linalg.norm(rect[1] - rect[0])),
                float(np.linalg.norm(rect[2] - rect[1])),
            )})
    chains = []
    # ponytail: group by guard_chain alone, exactly as the existing tests do;
    # this does not establish cross-view instance identity.
    for chain_id, members in groups.items():
        if len(members) < 2:
            continue
        rects = [_rectangle(obj) for obj in members]
        row = {"guard_chain": chain_id, "members": [_key(obj) for obj in members],
               "collinearity_residual_m": None, "parallel_spread_deg": None,
               "manhattan_off_axis_deg": None}
        if all(rect is not None for rect in rects):
            angles = [_angle(rect) for rect in rects]
            # Keep max-minus-min (including the 0/180 seam) to match the test.
            row["parallel_spread_deg"] = max(angles) - min(angles)
            if all(obj.get("centroid_xy") is not None for obj in members):
                cents = np.asarray([obj["centroid_xy"] for obj in members], float)
                if cents.shape != (len(members), 2) or not np.isfinite(cents).all():
                    raise ValueError(f"invalid centroids in chain {chain_id}")
                radians = np.radians(angles[0])
                offsets = (cents - cents.mean(axis=0)) @ [-np.sin(radians), np.cos(radians)]
                row["collinearity_residual_m"] = float(np.abs(offsets).max())
            if theta is not None and members[0].get("guard_axis_snapped"):
                row["manhattan_off_axis_deg"] = min(
                    abs((angles[0] - (theta % 180) + 90) % 180 - 90),
                    abs((angles[0] - ((theta + 90) % 180) + 90) % 180 - 90),
                )
        chains.append(row)
    return {
        "collinearity_max_residual_m": _maximum(r["collinearity_residual_m"] for r in chains),
        "parallel_max_spread_deg": _maximum(r["parallel_spread_deg"] for r in chains),
        "manhattan_max_off_axis_deg": _maximum(r["manhattan_off_axis_deg"] for r in chains),
        "thin_max_short_side_m": _maximum(r["short_side_m"] for r in thin),
        "eligible_chain_count": len(chains) if objects is not None else None,
        "thin_structure_count": len(thin) if objects is not None else None,
        "chains": chains if objects is not None else None,
        "thin_structures": thin if objects is not None else None,
    }


def _reprojection(payload, objects):
    scores = payload.get("scores") if payload is not None else None
    if scores is not None and not isinstance(scores, list):
        raise ValueError("reprojection scores must be a list")
    eligible = [obj for obj in objects or [] if
                obj.get("footprint_method") in ("guard-line", "contact-edge")
                and "fence" in obj["label"] and obj.get("rect_snapped")]

    def matches(score, obj):
        key = _key(obj)
        return all(key[k] == score[k] for k in key if k in score) and (
            "frame" not in score or score["frame"] == key["frame_id"]
        )

    # verify_reprojection emits exactly one row per eligible inventory object
    # in inventory order. Use this only when the entire sequence agrees.
    sequence_matches = scores is not None and len(scores) == len(eligible) and all(
        matches(score, obj) for score, obj in zip(scores, eligible)
    )
    rows, values, statuses = [], [], Counter()
    for index, score in enumerate(scores or []):
        value = _number(score.get("mean_dv_frac"))
        status = score.get("status") or ("scored" if value is not None else "unscored")
        statuses[status] += 1
        if value is not None:
            values.append(value)
        candidates = [obj for obj in eligible if matches(score, obj)]
        obj = eligible[index] if sequence_matches else candidates[0] if len(candidates) == 1 else None
        rows.append({**score, "mean_dv_frac": value,
                     "key": _key(obj if obj is not None else score),
                     "identity_source": "inventory_sequence" if sequence_matches else
                     "unique_inventory_match" if obj is not None else "source_only",
                     "candidate_entity_keys": [_key(o) for o in candidates] if obj is None else [],
                     "status": status})
    return {
        "worst_frac": max(values) if values else None,
        "median_frac": float(np.median(values)) if values else None,
        "scored_count": len(values) if scores is not None else None,
        "total_count": len(scores) if scores is not None else None,
        "status_counts": dict(statuses) if scores is not None else None,
        "scores": rows if scores is not None else None,
        "scope": "Legacy first-frame floor lookup and visible fence contact; not independent GT.",
    }


def _scale(run, scene):
    frames = []
    for cache in sorted((run / "geometry/moge").glob("*.json")):
        moge = _number(_read(cache, dict).get("moge_median_range_m"))
        directory = run / "geometry/frames" / cache.stem
        paths = [directory / name for name in ("pts3d.npy", "valid_mask.npy", "camera_to_world.npy")]
        native = None
        if all(path.exists() for path in paths):
            points, valid, pose = [np.load(path, allow_pickle=False) for path in paths]
            world = points[valid.astype(bool) & np.isfinite(points).all(axis=2)]
            # Same range and 100-point eligibility as moge._mapanything_median_range.
            if len(world) >= 100:
                camera = (np.linalg.inv(np.asarray(pose, dtype=float))
                          @ np.c_[world, np.ones(len(world))].T)[:3].T
                native = _number(float(np.median(np.linalg.norm(camera, axis=1))))
                if native is not None and native <= 0:
                    native = None
        frames.append({"frame_id": cache.stem, "moge_median_range_m": moge,
                       "native_median_range_m": native,
                       "ratio": moge / native if moge is not None and moge > 0 and native is not None else None})
    ratios = [frame["ratio"] for frame in frames]
    return {
        "source": scene.get("scale_source"), "factor": _number(scene.get("scale_factor")),
        "confidence": _number(scene.get("scale_confidence")),
        "ratios": ratios or None, "per_frame": frames or None,
        "ground_truth_error": None,
        "interpretation": "MoGe/native ratios and stored confidence measure consistency, not metric accuracy.",
    }


def collect_metrics(run: Path) -> dict:
    run = Path(run).resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    inventory = _read(run / "inventory/inventory.json", dict)
    objects = inventory.get("objects") if inventory is not None else None
    if objects is not None and not isinstance(objects, list):
        raise ValueError("inventory objects must be a list")
    phrases = _read(run / "inventory/phrases.json", list)
    unresolved = _read(run / "inventory/unresolved.json", list)
    scene = _read(run / "scene.json", dict) or {}
    cell = inventory.get("cell_rect") if inventory is not None else None
    sides = cell.get("sides") if cell is not None else None
    dropped = None
    if phrases is not None and objects is not None and unresolved is not None:
        covered = {obj["label"] for obj in objects} | {item["phrase"] for item in unresolved}
        dropped = [phrase for phrase in phrases if phrase not in covered]
    footprints = []
    for obj in objects or []:
        if any(term in obj["label"] for term in CONTACT_FAMILY):
            footprints.append({"key": _key(obj), "footprint_method": obj.get("footprint_method"),
                               "recorded_area_m2": _number(obj.get("footprint_area_m2")),
                               "polygon_area_m2": _area(obj.get("footprint")),
                               "snapped_area_m2": _area(obj.get("rect_snapped"))})
    return {
        "run_id": run.name, "run_path": str(run),
        "canonical_test_run": run.name in CEILINGS,
        "existing_test_thresholds": {
            "collinearity_m_lt": .30, "parallel_deg_lt": 2., "manhattan_deg_lt": 6.,
            "thin_m_lte": .35, "reprojection_frac_lte": CEILINGS[run.name],
        } if run.name in CEILINGS else None,
        "geometry": _geometry(objects, _number((inventory or {}).get("manhattan_theta_deg"))),
        "reprojection": _reprojection(_read(run / "inventory/reprojection.json", dict), objects),
        "inventory": {
            "object_count": len(objects) if objects is not None else None,
            "entity_keys": [_key(obj) for obj in objects] if objects is not None else None,
            "outside_cell_count": sum(bool(obj.get("outside_cell")) for obj in objects) if objects is not None else None,
            "cell_rect": cell, "cell_side_count": sum(v is not None for v in sides.values()) if sides is not None else None,
            "phrase_count": len(phrases) if phrases is not None else None,
            "unresolved_count": len(unresolved) if unresolved is not None else None,
            "unresolved": unresolved, "silently_dropped_phrases": dropped,
        },
        "fence_footprints": footprints if objects is not None else None,
        "scale": _scale(run, scene),
        "ground_truth": None, "latency_s": None, "cost_usd": None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError("metrics output must be outside the run directory")
    result = collect_metrics(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
