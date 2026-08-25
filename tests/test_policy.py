from ehs_spatial.contracts import Entity3D, SceneMap
from ehs_spatial.policy import (
    Predicate,
    PolicySpec,
    evaluate_policies,
    evaluate_policy,
)


def _entity(entity_id, label, footprint, frames=("f1", "f2"), height=1.0, tilt=None):
    return Entity3D(
        entity_id=entity_id,
        label=label,
        observation_ids=[f"obs-{entity_id}"],
        centroid_xyz=(0.0, 0.0, height / 2),
        footprint_xy=footprint,
        height_m=height,
        evidence_frame_ids=list(frames),
        tilt_deg=tilt,
    )


def _square(x, y, size=0.4):
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]


def _scene(entities):
    return SceneMap(
        run_id="policy-test",
        floor_plane=(0.0, 0.0, 1.0, 0.0),
        scale_source="camera_height",
        scale_factor=1.0,
        fence_polygon=[],
        entities=entities,
        facts=[],
        warnings=[],
    )


def _spec(**kwargs):
    base = dict(
        policy_id="p1",
        source_text="test",
        predicate=Predicate.MIN_SEPARATION,
        subject_labels=["pallet"],
        object_labels=["safety fence"],
        threshold=0.6,
    )
    base.update(kwargs)
    return PolicySpec(**base)


def test_min_separation_fails_below_threshold_and_passes_above():
    fence = _entity("fence", "safety fence", _square(0, 0, 2.0))
    near = _entity("near", "pallet", _square(2.3, 0.5))
    far = _entity("far", "pallet", _square(3.0, 0.5))

    failing = evaluate_policy(_spec(), _scene([fence, near]))
    passing = evaluate_policy(_spec(), _scene([fence, far]))

    assert failing.status.value == "FAIL"
    assert failing.violations[0].subject_id == "near"
    assert abs(failing.violations[0].measured - 0.3) < 1e-6
    assert passing.status.value == "PASS"
    assert not passing.violations
    # Every evaluation leaves a measurement behind, pass or fail.
    assert failing.facts and passing.facts
    assert failing.evidence_frame_ids == ["f1", "f2"]


def test_boundary_is_strict_less_than():
    fence = _entity("fence", "safety fence", _square(0, 0, 2.0))
    exact = _entity("exact", "pallet", _square(2.6, 0.5))

    result = evaluate_policy(_spec(), _scene([fence, exact]))

    # 0.60 m gap against a 0.60 m minimum is compliant, matching the
    # clearance rule's documented convention.
    assert result.status.value == "PASS"


def test_missing_subject_or_object_abstains_with_a_reason():
    fence = _entity("fence", "safety fence", _square(0, 0, 2.0))

    no_subject = evaluate_policy(_spec(), _scene([fence]))
    no_object = evaluate_policy(
        _spec(), _scene([_entity("p", "pallet", _square(3, 3))])
    )

    assert no_subject.status.value == "INSUFFICIENT_EVIDENCE"
    assert "subject labels" in no_subject.warnings[0]
    assert no_object.status.value == "INSUFFICIENT_EVIDENCE"
    assert "object labels" in no_object.warnings[0]
    assert not no_subject.violations and not no_object.violations


def test_evidence_gate_rejects_single_frame_entity_on_a_four_view_capture():
    fence = _entity("fence", "safety fence", _square(0, 0, 2.0))
    thin = _entity("thin", "pallet", _square(2.3, 0.5), frames=("f1",))

    result = evaluate_policy(_spec(), _scene([fence, thin]))

    assert result.status.value == "INSUFFICIENT_EVIDENCE"


def test_evidence_gate_scales_down_for_single_photo_capture():
    fence = _entity("fence", "safety fence", _square(0, 0, 2.0), frames=("f1",))
    thin = _entity("thin", "pallet", _square(2.3, 0.5), frames=("f1",))

    result = evaluate_policy(
        _spec(), _scene([fence, thin]), capture_frame_count=1
    )

    assert result.status.value == "FAIL"


def test_max_separation_flags_the_far_subject():
    panel = _entity("panel", "control panel", _square(0, 0, 0.5))
    near = _entity("near", "fire extinguisher", _square(1.0, 0))
    far = _entity("far", "fire extinguisher", _square(9.0, 0))
    spec = _spec(
        predicate=Predicate.MAX_SEPARATION,
        subject_labels=["fire extinguisher"],
        object_labels=["control panel"],
        threshold=5.0,
    )

    result = evaluate_policy(spec, _scene([panel, near, far]))

    assert result.status.value == "FAIL"
    assert [v.subject_id for v in result.violations] == ["far"]


def test_not_inside_flags_overlap_only():
    zone = _entity("zone", "hazard zone", _square(0, 0, 2.0))
    inside = _entity("inside", "person", _square(0.5, 0.5))
    outside = _entity("outside", "person", _square(3.0, 0.5))
    spec = _spec(
        predicate=Predicate.NOT_INSIDE,
        subject_labels=["person"],
        object_labels=["hazard zone"],
        threshold=0.01,
    )

    result = evaluate_policy(spec, _scene([zone, inside, outside]))

    assert result.status.value == "FAIL"
    assert [v.subject_id for v in result.violations] == ["inside"]


def test_self_predicates_need_no_object_label():
    tall = _entity("tall", "pallet stack", _square(0, 0), height=3.2)
    short = _entity("short", "pallet stack", _square(2, 0), height=1.1)
    spec = _spec(
        predicate=Predicate.MAX_HEIGHT,
        subject_labels=["pallet stack"],
        object_labels=[],
        threshold=2.5,
    )

    result = evaluate_policy(spec, _scene([tall, short]))

    assert result.status.value == "FAIL"
    assert [v.subject_id for v in result.violations] == ["tall"]


def test_max_tilt_abstains_when_no_subject_has_a_tilt():
    spec = _spec(
        predicate=Predicate.MAX_TILT,
        subject_labels=["step ladder"],
        object_labels=[],
        threshold=15.0,
    )
    untilted = _entity("l1", "step ladder", _square(0, 0))
    tilted = _entity("l2", "step ladder", _square(2, 0), tilt=22.0)

    assert evaluate_policy(spec, _scene([untilted])).status.value == (
        "INSUFFICIENT_EVIDENCE"
    )
    assert evaluate_policy(spec, _scene([tilted])).status.value == "FAIL"


def test_unsupported_policy_never_evaluates():
    spec = _spec(unsupported_reason="requires a signage check, not geometry")
    fence = _entity("fence", "safety fence", _square(0, 0, 2.0))
    near = _entity("near", "pallet", _square(2.3, 0.5))

    result = evaluate_policy(spec, _scene([fence, near]))

    assert result.status.value == "INSUFFICIENT_EVIDENCE"
    assert "not evaluable" in result.warnings[0]
    assert not result.facts


def test_evaluate_policies_keeps_order_and_independence():
    fence = _entity("fence", "safety fence", _square(0, 0, 2.0))
    near = _entity("near", "pallet", _square(2.3, 0.5))
    scene = _scene([fence, near])
    specs = [
        _spec(policy_id="a", threshold=0.6),
        _spec(policy_id="b", threshold=0.2),
    ]

    results = evaluate_policies(specs, scene)

    assert [r.policy_id for r in results] == ["a", "b"]
    assert [r.status.value for r in results] == ["FAIL", "PASS"]
