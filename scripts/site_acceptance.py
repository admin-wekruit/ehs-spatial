"""Operator site-acceptance protocol: tape measure vs the production pipeline.

When the owner walks a real site with a tape measure, this script answers one
question: does the PRODUCT's measurement path reproduce the tape? It runs the
production code, not an eval-tuned variant — the production SAM3Adapter with
the production LABEL_PROMPTS synonym fallback, the production floor fit and
entity clustering (`_build_geometry`), and the production scale chain
(MoGe auto anchor first, operator camera height as fallback, model-native
last, exactly as `EHSAssessmentPipeline._resolve_scale` orders them). An
eval-style direct path would score the models, not the product; acceptance
is about what the operator will actually see.

Distances use the production clearance semantic: min distance between entity
footprint polygons in the floor frame (0 when they touch). Pass bands are
the measured error budgets imported from ehs_spatial.rules — +/-0.20 m for
multiview captures, +/-0.35 m for a single photo.

Truth file format (written by whoever holds the tape):

  {
    "camera_height_m": 1.5,          // or null if not measured
    "pairs": [
      {"name": "pallet-to-fence", "measured_m": 1.23,
       "subject_label": "pallet", "object_label": "safety fence"}
    ]
  }

Labels must come from the production PROMPT_VOCABULARY — acceptance tests
the shipping vocabulary, not ad-hoc prompts.

House pattern: geometry, SAM, and MoGe responses are disk-cached in the
pack; the first run prints the paid-call count and exits 2 without --live;
cached re-scores are free.

  uv run --env-file .env python scripts/site_acceptance.py \
      --photos captures/site1 --truth captures/site1/truth.json --live
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ehs_spatial.rules import ERROR_BUDGET_MONO_M, ERROR_BUDGET_MULTIVIEW_M

PHOTO_SUFFIXES = (".png", ".jpg", ".jpeg")


def load_truth(path: Path) -> dict:
    from ehs_spatial.providers.sam3 import PROMPT_VOCABULARY

    truth = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(truth.get("pairs"), list) or not truth["pairs"]:
        raise ValueError("truth file needs a non-empty 'pairs' list")
    allowed = set(PROMPT_VOCABULARY)
    for pair in truth["pairs"]:
        for key in ("name", "measured_m", "subject_label", "object_label"):
            if key not in pair:
                raise ValueError(f"pair is missing '{key}': {pair}")
        if float(pair["measured_m"]) <= 0:
            raise ValueError(f"measured_m must be positive: {pair}")
        for label in (pair["subject_label"], pair["object_label"]):
            if label not in allowed:
                raise ValueError(
                    f"label {label!r} is not in the production vocabulary "
                    f"{sorted(allowed)}"
                )
    return truth


def pair_distance_m(
    entities: list, subject_label: str, object_label: str
) -> tuple[float | None, list[str]]:
    """Production clearance semantic: min footprint-polygon distance over all
    candidate entity combinations; 0 when footprints touch."""
    from shapely.geometry import Polygon

    warnings: list[str] = []
    subjects = [e for e in entities if e.label == subject_label]
    objects = [e for e in entities if e.label == object_label]
    if subject_label == object_label:
        if len(subjects) < 2:
            return None, [f"fewer than two {subject_label!r} entities found"]
    elif not subjects or not objects:
        missing = subject_label if not subjects else object_label
        return None, [f"no {missing!r} entity was reconstructed"]
    if len(subjects) > 1 or len(objects) > 1:
        warnings.append(
            f"multiple candidates ({len(subjects)} {subject_label!r}, "
            f"{len(objects)} {object_label!r}): reporting the closest pair"
        )
    best = None
    for subject in subjects:
        for target in objects:
            if subject.entity_id == target.entity_id:
                continue
            pa = Polygon(subject.footprint_xy)
            pb = Polygon(target.footprint_xy)
            distance = 0.0 if pa.intersects(pb) else float(pa.distance(pb))
            best = distance if best is None else min(best, distance)
    return best, warnings


def score_pairs(
    truth: dict,
    predictions: dict[str, float | None],
    *,
    tier: str,
    scale_source: str,
    warnings: list[str] | None = None,
) -> dict:
    """Pure scoring: tape-measured pairs vs predicted distances."""
    band = ERROR_BUDGET_MULTIVIEW_M if tier == "multiview" else ERROR_BUDGET_MONO_M
    rows = []
    for pair in truth["pairs"]:
        measured = float(pair["measured_m"])
        predicted = predictions.get(pair["name"])
        row = {
            "name": pair["name"],
            "subject_label": pair["subject_label"],
            "object_label": pair["object_label"],
            "measured_m": round(measured, 3),
            "predicted_m": None if predicted is None else round(predicted, 3),
            "abs_err_m": None,
            "rel_err": None,
            "passed": False,
        }
        if predicted is not None:
            row["abs_err_m"] = round(abs(predicted - measured), 3)
            row["rel_err"] = round(abs(predicted - measured) / measured, 3)
            row["passed"] = row["abs_err_m"] <= band
        rows.append(row)
    return {
        "tier": tier,
        "pass_band_m": band,
        "scale_source": scale_source,
        "pairs": rows,
        "n_pass": sum(1 for r in rows if r["passed"]),
        "n_total": len(rows),
        "passed": bool(rows) and all(r["passed"] for r in rows),
        "warnings": warnings or [],
    }


def print_table(scorecard: dict) -> None:
    print(
        f"tier={scorecard['tier']}  pass band=+/-{scorecard['pass_band_m']} m"
        f"  scale source={scorecard['scale_source']}"
    )
    header = f"{'pair':<24}{'measured':>10}{'predicted':>11}{'abs err':>9}{'rel':>7}  verdict"
    print(header)
    print("-" * len(header))
    for row in scorecard["pairs"]:
        predicted = "-" if row["predicted_m"] is None else f"{row['predicted_m']:.3f}"
        abs_err = "-" if row["abs_err_m"] is None else f"{row['abs_err_m']:.3f}"
        rel = "-" if row["rel_err"] is None else f"{row['rel_err']:.0%}"
        verdict = "PASS" if row["passed"] else "FAIL"
        print(
            f"{row['name']:<24}{row['measured_m']:>10.3f}{predicted:>11}"
            f"{abs_err:>9}{rel:>7}  {verdict}"
        )
    print(
        f"{scorecard['n_pass']}/{scorecard['n_total']} pairs inside the band"
        f" -> {'ACCEPTED' if scorecard['passed'] else 'NOT ACCEPTED'}"
    )
    for warning in scorecard["warnings"]:
        print("warning:", warning)


def _frames_from_disk(geometry_dir: Path) -> list:
    from ehs_spatial.contracts import GeometryFrame

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


def _segment_all(
    pack: Path, frames: list, labels: list[str], *, live: bool
) -> tuple[list | None, int]:
    """Production SAM3Adapter with a disk-caching subscriber.

    Counts worst-case uncached calls first (every synonym fallback), then
    runs the exact production first-hit loop. Returns (observations, planned)
    with observations None when spend is needed but --live is absent."""
    from ehs_spatial.providers.sam3 import LABEL_PROMPTS, SAM3Adapter

    cache_dir = pack / "sam_cache_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(frame_id: str, prompt: str) -> Path:
        return cache_dir / f"{frame_id}__{prompt.replace(' ', '_')}.json"

    planned = 0
    for frame in frames:
        for label in labels:
            for prompt in LABEL_PROMPTS[label]:
                if not cache_path(frame.frame_id, prompt).exists():
                    planned += 1
                else:
                    break
    if planned and not live:
        return None, planned

    observations = []
    for frame in frames:
        for label in labels:
            for prompt in LABEL_PROMPTS[label]:
                path = cache_path(frame.frame_id, prompt)

                def subscriber(endpoint, *, arguments, _path=path):
                    if _path.exists():
                        return json.loads(_path.read_text(encoding="utf-8"))
                    import fal_client

                    response = fal_client.subscribe(endpoint, arguments=arguments)
                    _path.write_text(json.dumps(response) + "\n", encoding="utf-8")
                    return response

                found = SAM3Adapter(subscriber=subscriber).segment(
                    frame.canonical_image_path,
                    prompt=prompt,
                    label=label,
                    frame_id=frame.frame_id,
                    output_dir=pack / "masks" / frame.frame_id,
                )
                if found:
                    observations.extend(found)
                    break
    return observations, 0


def _resolve_scale(
    frames: list, geometry_dir: Path, camera_height_m: float | None
) -> dict:
    """The production scale chain (pipeline._resolve_scale, 'auto'
    preference): MoGe anchor first, camera height fallback, model native."""
    from ehs_spatial.providers.moge import MoGeAnchorAdapter

    try:
        anchor = MoGeAnchorAdapter().anchor_scale(frames, geometry_dir)
    except Exception:
        anchor = None
    if anchor is not None:
        return {
            "override": anchor.scale,
            "source": "moge_anchor",
            "confidence": anchor.confidence,
            "warnings": [],
        }
    if camera_height_m is not None:
        return {
            "override": None,
            "source": "camera_height",
            "confidence": 0.9,
            "warnings": [
                "auto scale anchor unavailable; fell back to the "
                "operator-supplied camera height"
            ],
        }
    return {
        "override": 1.0,
        "source": "model_native",
        "confidence": 0.2,
        "warnings": [
            "no scale anchor available; distances use the model's native "
            "scale and may be off by a large factor"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--pack", type=Path, default=None)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)

    truth = load_truth(args.truth)
    photos = sorted(
        p for p in args.photos.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES
    )
    if not 1 <= len(photos) <= 4:
        print(f"need 1-4 photos in {args.photos}, found {len(photos)}", file=sys.stderr)
        return 1
    pack = args.pack or Path("outputs/site_acceptance") / args.photos.resolve().name
    pack.mkdir(parents=True, exist_ok=True)
    geometry_dir = pack / "geometry"
    tier = "multiview" if len(photos) >= 2 else "mono"
    camera_height = truth.get("camera_height_m")

    if (geometry_dir / "frames").is_dir():
        frames = _frames_from_disk(geometry_dir)
    elif args.live:
        from ehs_spatial.providers.map_anything import MapAnythingAdapter

        frames, _ = MapAnythingAdapter().run([str(p) for p in photos], geometry_dir)
    else:
        print(
            f"needs 1 MapAnything call ({len(photos)} photo(s)), up to "
            f"{len(photos)} MoGe call(s), plus SAM; pass --live",
            file=sys.stderr,
        )
        return 2

    labels = sorted(
        {p["subject_label"] for p in truth["pairs"]}
        | {p["object_label"] for p in truth["pairs"]}
    )
    observations, planned = _segment_all(pack, frames, labels, live=args.live)
    if observations is None:
        print(
            f"needs up to {planned} uncached SAM call(s); pass --live "
            "(cached reruns are free)",
            file=sys.stderr,
        )
        return 2

    scale = _resolve_scale(frames, geometry_dir, camera_height)
    from ehs_spatial.geometry import _build_geometry

    geometry = _build_geometry(
        frames,
        observations,
        camera_height,
        scale_factor_override=scale["override"],
    )
    warnings = [*scale["warnings"], *geometry.warnings]
    if geometry.transform is None:
        scorecard = score_pairs(
            truth, {}, tier=tier, scale_source=scale["source"], warnings=warnings
        )
        scorecard["error"] = "floor fit failed; no measurements possible"
    else:
        predictions: dict[str, float | None] = {}
        for pair in truth["pairs"]:
            predicted, pair_warnings = pair_distance_m(
                geometry.entities, pair["subject_label"], pair["object_label"]
            )
            predictions[pair["name"]] = predicted
            warnings.extend(f"{pair['name']}: {w}" for w in pair_warnings)
        scorecard = score_pairs(
            truth,
            predictions,
            tier=tier,
            scale_source=scale["source"],
            warnings=warnings,
        )
        scorecard["scale_factor"] = geometry.transform.scale_factor
        scorecard["scale_confidence"] = scale["confidence"]
        scorecard["entity_counts"] = {
            label: sum(1 for e in geometry.entities if e.label == label)
            for label in labels
        }
    scorecard["photos"] = [p.name for p in photos]
    (pack / "site_report.json").write_text(
        json.dumps(scorecard, indent=2) + "\n"
    )
    print_table(scorecard)
    print("report:", (pack / "site_report.json").resolve())
    return 0 if scorecard["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
