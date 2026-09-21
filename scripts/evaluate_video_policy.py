"""Video object map -> metric facts -> the platform's own rule engine. The only place native units become metres.

Only confirmed entities (seen in >= 3 views) get facts. A footprint is the convex hull of what was seen, so it can
only be too small: gaps are upper bounds and a PASS means "no obstruction was observed", never "the way is clear".
Needs the real zen rule engine (run with a venv that has zen-engine); nothing is stubbed.

  python scripts/evaluate_video_policy.py --object-map DIR --scale metric-scale.json --output NEW_DIR
  python scripts/evaluate_video_policy.py --self-check
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEPTH_RELATIVE = .05  # ponytail: about twice the median depth error measured on the test clips; replace with a calibration study before trusting a band
CONFIRMED = 3
# OSHA 29 CFR 1910.36(g)(2): an exit access must be at least 28 inches wide at all points.
RULE = {"kind": "geometry", "spec": {"predicate": "min_separation", "subject_labels": ["desk"], "object_labels": ["cabinet"], "threshold": .711, "unit": "m"}}


def facts(entities, metres_per_native, scale_relative_uncertainty, subjects, objects):
    """Footprint and height in metres with one uncertainty per entity: depth noise at its range plus its share of the scale doubt on the nearest gap."""
    polygons = {e["entityId"]: Polygon(np.array(e["footprintPlanNative"]) * metres_per_native) for e in entities}
    result = {}
    for e in entities:
        others = [polygons[o["entityId"]] for o in entities if o is not e and {e["label"], o["label"]} == {subjects, objects}]
        gap = min((polygons[e["entityId"]].distance(other) for other in others), default=0.)
        uncertainty = float(np.hypot(DEPTH_RELATIVE * e["rangeNative"] * metres_per_native, scale_relative_uncertainty * gap / 2))
        result[e["entityId"]] = {"footprint": [list(xy) for xy in polygons[e["entityId"]].exterior.coords[:-1]], "uncertaintyM": uncertainty,
                                 "heightM": (e["heightNative"] - min(0., e["baseNative"])) * metres_per_native, "nearestGapM": float(gap)}
    return result


def scene_document(entities, measured, scale, frame_id="droid_final_native_world"):
    assets, observations, records = {}, [], []
    for e in entities:
        refs = []
        for oid, box in e["observationBoxes"].items():
            image = f"frame-{oid.split(':')[1]}"
            assets[image] = {"id": image}
            observations.append({"id": oid, "imageId": image, "originalPixelBox": box})
            refs.append(oid)
        m = measured[e["entityId"]]
        common = {"unit": "m", "uncertaintyM": m["uncertaintyM"], "source": "observed_measurement", "sourceRefs": refs}
        records.append({"id": e["entityId"], "label": e["label"], "associationState": "confirmed", "observationRefs": refs,
                        "measurements": {"footprint": {"value": m["footprint"], "coordinateFrameId": frame_id, **common}, "height": {"value": m["heightM"], **common}}})
    return {"schemaVersion": 1, "target": "scene", "cameras": [], "assets": list(assets.values()), "entities": records, "observations": observations,
            "coordinateFrames": [{"id": frame_id, "convention": "opencv", "scale": {**scale, "sourceRefs": [observations[0]["id"]]}}],
            "annotations": [{"id": "demo-applicability", "kind": "policy_applicability", "policyId": "video-demo", "value": "applicable",
                             "sourceRefs": [observations[0]["id"]], "note": "asserted for this pipeline demonstration, not by an EHS reviewer"}]}


def evaluate(scene):
    import zen  # noqa: F401  the real decision engine, never a stand-in
    from ehs_spatial.platform import policy_engine
    from ehs_spatial.platform.contracts import validate_document
    validate_document(scene)
    return policy_engine.evaluate_document(scene, "video-demo", {"jdm": policy_engine._jdm(RULE)}, "reconstruction")


def build(args):
    object_map, scale = json.loads((args.object_map / "object-map.json").read_text()), json.loads(args.scale.read_text())
    subjects, objects = RULE["spec"]["subject_labels"][0], RULE["spec"]["object_labels"][0]
    # a name a VLM gave to a class-agnostic segment is an interpretation nobody reviewed: it may describe a scene, not decide a rule
    named_by_vlm = [e["entityId"] for e in object_map["entities"] if e.get("labelSource") and e["label"] in (subjects, objects)]
    entities = [e for e in object_map["entities"] if len(e["observations"]) >= CONFIRMED and e["label"] in (subjects, objects) and not e.get("labelSource")]
    metres = scale["metres_per_native_unit"]
    doubt = abs(scale["height_anchor_vs_model_estimate"]) if scale.get("height_anchor_vs_model_estimate") is not None else 0.
    measured = facts(entities, metres, doubt, subjects, objects)
    scene = scene_document(entities, measured, {"status": "operator_anchored", "nativeToMeters": metres,
        "anchor": {"kind": "stated_carry_height", "metres": 1.6, "measured": False, "model_estimate_metres_per_native": scale.get("model_estimated_metres_per_native_unit"),
                   "sources_disagree_over_10pct": scale.get("scale_sources_disagree_over_10pct")}})
    findings = evaluate(scene)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "scene-document.json").write_text(json.dumps(scene, indent=1))
    report = {"rule": {"citation": "OSHA 29 CFR 1910.36(g)(2), exit access width >= 28 in (0.711 m)", "check": RULE,
                       "applicability": "asserted for the demonstration; the aisle between a desk and a cabinet is not a designated exit access"},
              "scale": scene["coordinateFrames"][0]["scale"], "scale_relative_uncertainty_used": doubt,
              "entities": {k: {kk: vv for kk, vv in v.items() if kk != "footprint"} for k, v in measured.items()},
              "excluded_vlm_named": named_by_vlm,
              "excluded_candidates": [e["entityId"] for e in object_map["entities"] if len(e["observations"]) < CONFIRMED and e["label"] in (subjects, objects)],
              "findings": findings,
              "reading": "Footprints are hulls of the seen sides, so every gap is an upper bound; FAIL and NEEDS_REVIEW are informative, PASS only says no obstruction was observed."}
    (args.output / "policy-findings.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    for e in entities:
        xy = np.array(measured[e["entityId"]]["footprint"])
        ax.fill(*xy.T, alpha=.35, color="tab:brown" if e["label"] == subjects else "tab:gray")
        ax.text(*xy.mean(0), f"{e['entityId']}\n±{measured[e['entityId']]['uncertaintyM']:.2f} m", ha="center", fontsize=8)
    ax.set_aspect("equal"); ax.set_xlabel("m"); ax.set_title("confirmed desks and cabinets on the floor plan, metres (1.6 m carry-height anchor)")
    fig.savefig(args.output / "plan-metres.png", dpi=80, bbox_inches="tight")
    for f in findings:
        comparison = f.get("comparison", {})
        print(f["entityId"], f["machineResult"], "missing:", f["missingEvidence"], "| measured:", [round(v["measured"], 3) for v in comparison.get("violations", [])] or comparison.get("measured"))


def self_check():
    """A 0.5 m and a 1.5 m aisle around the 0.711 m rule, through the real engine; candidates never reach it."""
    square = lambda x0, x1: [[x0, 0.], [x1, 0.], [x1, 1.], [x0, 1.]]
    def entity(identity, label, x0, x1):
        return {"entityId": identity, "label": label, "observations": ["a", "b", "c"], "footprintPlanNative": square(x0, x1), "heightNative": 1., "baseNative": 0.,
                "rangeNative": .2, "observationBoxes": {f"{label}:{n}:0": [0, 0, 10, 10] for n in (1, 2, 3)}}
    for gap, expected in ((.5, "FAIL"), (1.5, "PASS"), (.72, "NEEDS_REVIEW")):
        pair = [entity("desk-0", "desk", 0, 1), entity("cabinet-0", "cabinet", 1 + gap, 2 + gap)]
        measured = facts(pair, 1., 0., "desk", "cabinet")
        assert abs(measured["desk-0"]["nearestGapM"] - gap) < 1e-9 and abs(measured["desk-0"]["uncertaintyM"] - .01) < 1e-9
        finding = evaluate(scene_document(pair, measured, {"status": "operator_anchored", "nativeToMeters": 1.}))[0]
        assert finding["machineResult"] == expected and not finding["missingEvidence"], (gap, finding)
    doubtful = facts([entity("desk-0", "desk", 0, 1), entity("cabinet-0", "cabinet", 2, 3)], 2., .2, "desk", "cabinet")
    assert abs(doubtful["desk-0"]["nearestGapM"] - 2.) < 1e-9 and doubtful["desk-0"]["uncertaintyM"] > .2, "scale doubt must widen the band with the gap"
    print("video policy check passed: FAIL / PASS / NEEDS_REVIEW around 0.711 m through the real engine; scale doubt widens the band")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--object-map", type=Path)
    parser.add_argument("--scale", type=Path)
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else build(a)
