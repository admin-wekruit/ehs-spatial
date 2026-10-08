"""Run directly: pins the metric-evidence contract of policy_engine.evaluate_document's geometry branch.

Synthetic in-memory scene only; no models, GPU, DB or network. Sections marked
FINDING pin what the engine does today so a change is noticed, not endorsed.
Every scene is a schemaVersion 1 document that contracts.validate_document accepts. A v2 document
additionally needs entity.measurementEvidence/measurementSelections for footprint and height
(identity.snapshot_measurements writes them); that envelope is not exercised here.
"""
import sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import zen  # noqa: F401
    STUB = False
except ImportError:
    # ponytail: zen-engine is a declared dependency but absent from some local venvs. The JDM only
    # carries the check spec through, so a pass-through stands in; with zen installed the real JDM runs.
    sys.modules["zen"] = types.ModuleType("zen")
    STUB = True
from argus.platform import policy_engine as pe
from argus.platform.contracts import PlatformError, validate_document

if STUB:
    pe.execute_jdm = lambda jdm, applicability, labels: {"applicability": applicability, "check": jdm}

ANCHORED = {"id": "f", "convention": "opencv", "scale": {"status": "operator_anchored", "nativeToMeters": 1.0, "sourceRefs": ["obs"]}}  # repository.setCalibration demands scale.sourceRefs; the engine never re-checks them


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def ent(identity, label, poly=None, height=None, **footprint):
    """The fields the policy engine reads (plus associationState for the document validator); **footprint overrides one."""
    m = {}
    if poly is not None:
        m["footprint"] = {"value": poly, "unit": "m", "uncertaintyM": 0.0, "source": "observed_measurement",
                          "sourceRefs": ["obs"], "coordinateFrameId": "f", **footprint}
    if height is not None:
        m["height"] = height if isinstance(height, dict) else {"value": height, "unit": "m", "uncertaintyM": 0.0, "source": "observed_measurement", "sourceRefs": ["obs"]}
    return {"id": identity, "label": label, "associationState": "confirmed", "measurements": m}


def run(predicate, entities, threshold, frames=(ANCHORED,), context="reconstruction", objects=("zone",), stored=True, applicable=("obs",)):
    unit = {"not_inside": "m2", "max_tilt": "deg"}.get(predicate, "m")
    check = {"kind": "geometry", "spec": {"predicate": predicate, "subject_labels": ["item"], "object_labels": list(objects), "threshold": threshold, "unit": unit}}
    scene = {"schemaVersion": 1, "target": "scene", "cameras": [], "assets": [{"id": "img"}], "entities": entities,
             "observations": [{"id": "obs", "imageId": "img", "originalPixelBox": [0, 0, 10, 10]}], "coordinateFrames": list(frames),
             "annotations": [{"id": "a", "kind": "policy_applicability", "policyId": "p", "value": "applicable", "sourceRefs": list(applicable)}]}
    if stored:  # False only for non-finite numbers, which contracts.canonical refuses to store
        validate_document(scene)
    findings = pe.evaluate_document(scene, "p", {"jdm": check if STUB else pe._jdm(check)}, context)
    assert len(findings) == 1 and findings[0]["entityId"] == ("item" if applicable else None), findings
    return findings[0]


def verdict(*args, **kwargs):
    f = run(*args, **kwargs)
    assert f["missingEvidence"] == [] and "comparison" in f, f  # proves evaluate_policy actually ran
    return f["machineResult"]


def missing(*args, **kwargs):
    f = run(*args, **kwargs)
    assert f["machineResult"] == "INSUFFICIENT_EVIDENCE" and "comparison" not in f, f
    return f["missingEvidence"]


ZONE = ent("zone", "zone", box(0, 0, 4, 4))
at = lambda x, **kw: ent("item", "item", box(x, 0, x + 1, 1), **kw)  # 1 m square whose left edge is x; zone's right edge is 4

# 1+3. Every predicate: decisive verdicts with metric facts in an operator_anchored frame.
assert verdict("not_inside", [at(6), ZONE], 0) == "PASS"
assert verdict("not_inside", [at(3.5), ZONE], 0) == "FAIL"  # OVERLAPPING: half the item is in the zone
violation = run("not_inside", [at(3.5), ZONE], 0)["comparison"]["violations"][0]
assert (violation["measured"], violation["threshold"]) == (0.5, 0.0)  # intersection area; threshold is hard-coded 0 m2
for predicate in ("min_separation", "keep_clear"):
    assert verdict(predicate, [at(6), ZONE], 1.0) == "PASS"  # gap 2.0
    assert verdict(predicate, [at(4.5), ZONE], 1.0) == "FAIL"  # gap 0.5
    assert verdict(predicate, [at(5, uncertaintyM=0.1), ZONE], 1.0) == "NEEDS_REVIEW"  # gap == threshold, band 0.1+0
assert verdict("max_separation", [at(4.5), ZONE], 1.0) == "PASS"
assert verdict("max_separation", [at(7), ZONE], 1.0) == "FAIL"
assert verdict("max_separation", [at(5, uncertaintyM=0.1), ZONE], 1.0) == "NEEDS_REVIEW"
height = lambda value, u=0.0: {"value": value, "unit": "m", "uncertaintyM": u, "source": "observed_measurement", "sourceRefs": ["obs"]}
for predicate, low, high in (("max_height", "PASS", "FAIL"), ("min_height", "FAIL", "PASS")):
    assert verdict(predicate, [at(0, height=1.0)], 2.0, objects=()) == low
    assert verdict(predicate, [at(0, height=3.0)], 2.0, objects=()) == high
    assert verdict(predicate, [at(0, height=height(2.05, 0.1))], 2.0, objects=()) == "NEEDS_REVIEW"
# The error band is target uncertainty + the largest object uncertainty.
assert verdict("min_separation", [at(5.15, uncertaintyM=0.1), ent("zone", "zone", box(0, 0, 4, 4), uncertaintyM=0.1)], 1.0) == "NEEDS_REVIEW"
assert verdict("min_separation", [at(5.15, uncertaintyM=0.1), ZONE], 1.0) == "PASS"
far = ent("zone2", "zone", box(20, 0, 24, 4), uncertaintyM=0.1)  # largest, not sum: band 0.2 < 0.25 margin; a summed 0.3 would be NEEDS_REVIEW
assert verdict("min_separation", [at(5.25, uncertaintyM=0.1), ent("zone", "zone", box(0, 0, 4, 4), uncertaintyM=0.1), far], 1.0) == "PASS"
# sourceRefs: id strings or dicts of assetId/imageId/observationId/annotationId, every id resolvable in assets/observations/cameras/annotations.
assert verdict("not_inside", [at(6, sourceRefs=[{"observationId": "obs", "imageId": "img"}, "a"]), ZONE], 0) == "PASS"
assert missing("not_inside", [at(6, sourceRefs=[{"observationId": "obs", "assetId": "nope"}]), ZONE], 0) == ["metric_footprint:item"]
assert missing("not_inside", [at(6, sourceRefs=[{}]), ZONE], 0) == ["metric_footprint:item"]
ok = run("not_inside", [at(6), ZONE], 0)
assert ok["facts"] == [at(6)["measurements"]["footprint"], ZONE["measurements"]["footprint"]]  # facts = the footprints used, target first
# Planning context swaps the accepted measurement source.
assert verdict("not_inside", [at(6, source="planned_geometry"), ent("zone", "zone", box(0, 0, 4, 4), source="manual_assertion")], 0, context="planning") == "PASS"
assert missing("not_inside", [at(6), ZONE], 0, context="planning") == ["metric_footprint:item", "metric_footprint:zone"]
assert verdict("not_inside", [at(6, source="manual_assertion"), ZONE], 0) == "PASS"
# Only the target and object-labelled entities need metric facts; sourceContext entities are never targets (run() insists on one finding).
assert verdict("not_inside", [at(6), ZONE, ent("bystander", "bystander"), {**at(3.5), "id": "context", "sourceContext": "reference"}], 0) == "PASS"

# Scene-level gate: without an evidenced applicability annotation the geometry branch is never entered.
gate = run("not_inside", [at(6), ZONE], 0, applicable=())
assert (gate["machineResult"], gate["missingEvidence"]) == (None, ["applicability_confirmation"])

# 2. Every missing-evidence code, exact, for subject and object alike.
for bad in ({"scale": {"status": "model_estimated", "nativeToMeters": 1.0}}, {"scale": {"status": "uncalibrated"}}):
    assert missing("not_inside", [at(6), ZONE], 0, frames=[{**ANCHORED, **bad}]) == ["metric_calibration:item", "metric_calibration:zone"]
assert missing("not_inside", [at(6, coordinateFrameId="gone"), ZONE], 0) == ["metric_calibration:item"]
assert missing("not_inside", [ent("item", "item"), ZONE], 0) == ["metric_footprint:item"]
assert missing("not_inside", [at(6), ent("zone", "zone")], 0) == ["metric_footprint:zone"]
for override in ({"unit": "native"}, {"uncertaintyM": None}, {"sourceRefs": ["unknown-observation"]}, {"sourceRefs": []}, {"source": "planned_geometry"}, {"source": "model_estimate"}):
    assert missing("not_inside", [at(6, **override), ZONE], 0) == ["metric_footprint:item"], override
assert missing("max_height", [at(0)], 2.0, objects=()) == ["metric_height:item"]
# FINDING: an invalid height uncertaintyM is the soft code metric_height, while the same value on a footprint raises 422 (below).
for override in ({"unit": "cm"}, {"uncertaintyM": None}, {"sourceRefs": []}, {"value": -1.0}, {"source": "model_estimate"}, {"value": "1"}, {"value": True},
                 {"value": float("inf")}, {"uncertaintyM": -1}, {"uncertaintyM": "0.1"}, {"uncertaintyM": True}, {"uncertaintyM": float("nan")}):
    finite = all(v == v and v != float("inf") for v in override.values())
    assert missing("max_height", [at(0, height={**height(1.0), **override})], 2.0, objects=(), stored=finite) == ["metric_height:item"], override
bad = at(0)
bad["measurements"]["height"] = "tall"
assert missing("max_height", [bad], 2.0, objects=()) == ["metric_height:item"]
other = {**ANCHORED, "id": "g"}
assert missing("not_inside", [at(6), ent("zone", "zone", box(0, 0, 4, 4), coordinateFrameId="g")], 0, frames=[ANCHORED, other]) == ["registered_coordinate_frame"]
assert missing("not_inside", [ent("item", "item", [[0, 0, 0]]), ZONE], 0) == ["invalid_geometry:item"]
assert missing("not_inside", [at(6, height=height(-1.0)), ZONE], 0) == ["invalid_geometry:item"]  # height is validated even where it is not used
assert run("not_inside", [ent("item", "item", [[0, 0, 0]]), ZONE], 0)["facts"][0]["value"] == [[0, 0, 0]]  # FINDING: the invalid footprint is still recorded as a fact
for value in (-1, "0.1", True, float("nan")):
    try:
        run("not_inside", [at(6, uncertaintyM=value), ZONE], 0, stored=value == value)  # NaN is the only one a stored document cannot hold
        raise AssertionError(value)
    except PlatformError as error:
        assert error.code == "measurement_uncertainty_invalid" and error.status == 422

# 4. NESTED and stacked footprints. FINDING: footprints are 2D, so a monitor on a desk or a chair
# under a desk is indistinguishable from an intrusion; height is never consulted by area/gap predicates.
DESK = ent("zone", "zone", box(0, 0, 2, 1), height=0.75)
monitor = ent("item", "item", box(0.5, 0.2, 1.1, 0.4), height=0.4)
nested = run("not_inside", [monitor, DESK], 0)
assert nested["machineResult"] == "FAIL" and abs(nested["comparison"]["violations"][0]["measured"] - 0.12) < 1e-9  # whole monitor area
for predicate in ("min_separation", "keep_clear"):
    chair = run(predicate, [ent("item", "item", box(0.5, 0.2, 1.1, 0.8), uncertaintyM=5.0), DESK], 0.5)
    # FINDING: intersection forces FAIL even when the uncertainty band (5 m) dwarfs the threshold.
    assert chair["machineResult"] == "FAIL" and chair["comparison"]["violations"][0]["measured"] == 0.0
assert verdict("max_separation", [monitor, DESK], 1.0) == "PASS"  # nested counts as gap 0
# FINDING: not_inside has no error band. 1 mm clear with 0.5 m uncertainty is a decisive PASS, and
# edge contact (zero overlap area) passes not_inside while failing min_separation.
assert verdict("not_inside", [at(4.001, uncertaintyM=0.5), ZONE], 0) == "PASS"
assert verdict("not_inside", [at(3.999, uncertaintyM=0.5), ZONE], 0) == "FAIL"  # and 1 mm inside is a decisive FAIL
assert verdict("not_inside", [at(4), ZONE], 0) == "PASS" and verdict("min_separation", [at(4), ZONE], 1.0) == "FAIL"
# FINDING: for height predicates the footprint uncertainty is discarded, only height uncertainty bands.
assert verdict("max_height", [at(0, height=1.9, uncertaintyM=5.0)], 2.0, objects=()) == "PASS"
# FINDING: frame.scale.nativeToMeters is never applied or cross-checked; unit == "m" is taken on trust.
# Gap 0.5 m is FAIL as written; applying the factor would make it 500 m and PASS.
assert verdict("min_separation", [at(4.5), ZONE], 1.0, frames=[{**ANCHORED, "scale": {"status": "operator_anchored", "nativeToMeters": 1000.0, "sourceRefs": ["obs"]}}]) == "FAIL"
# FINDING: these end INSUFFICIENT_EVIDENCE with an EMPTY missingEvidence list (reason only in comparison.warnings).
for entities, predicate, threshold in (([at(6)], "not_inside", 0),  # no zone entity in the scene at all
                                       ([ent("item", "item", []), ZONE], "not_inside", 0),  # footprint without a polygon
                                       ([at(0)], "max_tilt", 5.0)):  # platform never supplies tilt, so max_tilt cannot decide
    f = run(predicate, entities, threshold, objects=() if predicate == "max_tilt" else ("zone",))
    assert f["machineResult"] == "INSUFFICIENT_EVIDENCE" and f["missingEvidence"] == [] and f["comparison"]["warnings"], f
# FINDING: contracts allow entity.measurements = None; the engine crashes instead of reporting metric_footprint.
try:
    run("not_inside", [{**ent("item", "item"), "measurements": None}, ZONE], 0)
    raise AssertionError("measurements=None no longer crashes; pin the new behaviour")
except AttributeError:
    pass

print("PASS platform policy geometry contract" + (" (zen stubbed)" if STUB else ""))
