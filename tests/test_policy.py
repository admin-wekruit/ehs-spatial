import importlib.util
import json
import re
from pathlib import Path

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


def test_boundary_band_matches_the_clearance_rule():
    # Same discipline as rules.py: multiview band ±0.20 m around the 0.60 m
    # threshold — inside the band the measurement cannot honestly pick a
    # side, so an exact-threshold gap is NEEDS_REVIEW, not a razor-edge PASS.
    fence = _entity("fence", "safety fence", _square(0, 0, 2.0))
    for offset, expected in (
        (2.35, "FAIL"),          # gap 0.35 < 0.40
        (2.6, "NEEDS_REVIEW"),   # gap 0.60, mid-band
        (2.79, "NEEDS_REVIEW"),  # gap 0.79, upper band edge
        (2.85, "PASS"),          # gap 0.85 > 0.80
    ):
        result = evaluate_policy(
            _spec(), _scene([fence, _entity("p", "pallet", _square(offset, 0.5))])
        )
        assert result.status.value == expected, offset
    review = evaluate_policy(
        _spec(), _scene([fence, _entity("p", "pallet", _square(2.6, 0.5))])
    )
    assert "cannot honestly pick a side" in review.warnings[0]
    assert not review.violations


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
    # Mono band is ±0.35 m, so the violating gap must sit below 0.25 m
    # for the verdict to stay an honest FAIL.
    thin = _entity("thin", "pallet", _square(2.1, 0.5), frames=("f1",))

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


def test_min_height_banded_triplet_matches_the_inverted_band_discipline():
    # Multiview band ±0.20 m around a 1.8 m minimum: below the band fails,
    # inside it abstains, above it passes — MAX_HEIGHT's discipline inverted.
    spec = _spec(
        predicate=Predicate.MIN_HEIGHT,
        subject_labels=["safety fence"],
        object_labels=[],
        threshold=1.8,
    )
    for height, expected in (
        (1.5, "FAIL"),           # 1.5 < 1.60 band floor
        (1.7, "NEEDS_REVIEW"),   # inside 1.60..2.00
        (2.0, "NEEDS_REVIEW"),   # upper band edge
        (2.1, "PASS"),           # 2.1 > 2.00
    ):
        result = evaluate_policy(
            spec,
            _scene([_entity("f", "safety fence", _square(0, 0), height=height)]),
        )
        assert result.status.value == expected, height
        assert result.facts, height
    review = evaluate_policy(
        spec, _scene([_entity("f", "safety fence", _square(0, 0), height=1.7)])
    )
    assert "cannot honestly pick a side" in review.warnings[0]
    assert not review.violations


def test_min_height_violations_are_ordered_worst_first():
    spec = _spec(
        predicate=Predicate.MIN_HEIGHT,
        subject_labels=["safety fence"],
        object_labels=[],
        threshold=1.8,
    )
    short = _entity("short", "safety fence", _square(0, 0), height=1.4)
    shortest = _entity("shortest", "safety fence", _square(2, 0), height=0.9)

    result = evaluate_policy(spec, _scene([short, shortest]))

    assert result.status.value == "FAIL"
    assert [v.subject_id for v in result.violations] == ["shortest", "short"]
    assert result.violations[0].measured == 0.9
    assert result.violations[0].threshold == 1.8


def test_min_height_is_a_self_predicate_needing_no_object_label():
    # No object labels, no object entities: the subject measures itself,
    # and the recorded fact points back at the subject.
    spec = _spec(
        predicate=Predicate.MIN_HEIGHT,
        subject_labels=["safety fence"],
        object_labels=[],
        threshold=1.8,
    )
    tall = _entity("tall", "safety fence", _square(0, 0), height=2.4)

    result = evaluate_policy(spec, _scene([tall]))

    assert result.status.value == "PASS"
    assert result.facts[0].subject_id == "tall"
    assert result.facts[0].object_id == "tall"
    assert result.facts[0].value == 2.4


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
        _spec(policy_id="b", threshold=0.05),
    ]

    results = evaluate_policies(specs, scene)

    assert [r.policy_id for r in results] == ["a", "b"]
    # gap 0.30: below a's 0.40 band floor (FAIL), above b's 0.25 band
    # ceiling (PASS) — the two specs stay independent.
    assert [r.status.value for r in results] == ["FAIL", "PASS"]


def test_max_separation_never_matches_the_subject_to_itself():
    # Overlapping subject/object labels: the self-match at gap 0.0 must not
    # win the min and grant a false PASS ("within X of each other" rules).
    spec = _spec(
        predicate="max_separation",
        subject_labels=["fire extinguisher"],
        object_labels=["fire extinguisher"],
        threshold=15.0,
    )
    a = _entity("ext-a", "fire extinguisher", _square(0, 0))
    b = _entity("ext-b", "fire extinguisher", _square(40.0, 0))

    result = evaluate_policy(spec, _scene([a, b]))

    assert result.status.value == "FAIL"
    assert result.violations
    assert all(v.subject_id != v.object_id for v in result.violations)

    lonely = evaluate_policy(spec, _scene([a]))
    assert lonely.status.value == "INSUFFICIENT_EVIDENCE"


def test_violations_are_ordered_worst_first():
    fence = _entity("fence", "safety fence", _square(0, 0, 2.0))
    close = _entity("close", "pallet", _square(2.05, 1.5))   # gap 0.05
    closer = _entity("closer", "pallet", _square(2.01, 0.0))  # gap 0.01

    result = evaluate_policy(_spec(), _scene([fence, close, closer]))

    assert result.status.value == "FAIL"
    assert [v.subject_id for v in result.violations] == ["closer", "close"]


def test_not_inside_reports_zero_area_limit_not_the_placement_tolerance():
    spec = _spec(
        predicate="not_inside",
        subject_labels=["pallet"],
        object_labels=["safety fence"],
        threshold=0.01,
    )
    fence = _entity("fence", "safety fence", _square(0, 0, 2.0))
    inside = _entity("inside", "pallet", _square(0.5, 0.5))

    result = evaluate_policy(spec, _scene([fence, inside]))

    assert result.status.value == "FAIL"
    assert result.violations[0].threshold == 0.0
    assert result.violations[0].unit == "m2"


def test_model_native_scale_demotes_policy_verdicts_to_review():
    fence = _entity("fence", "safety fence", _square(0, 0, 2.0))
    near = _entity("near", "pallet", _square(2.1, 0.5))
    scene = _scene([fence, near]).model_copy(
        update={"scale_source": "model_native"}
    )

    [result] = evaluate_policies([_spec()], scene)

    assert result.status.value == "NEEDS_REVIEW"
    assert any("model-native" in w for w in result.warnings)


# --- OSHA 1910 compiler exam ------------------------------------------------
# Real regulation text (tests/fixtures/oshacorpus, built by
# scripts/oshacorpus.py) grades the compiler's refusal discipline. The live
# compiler is Gemini-driven and is NOT called here: these tests pin the
# deterministic layer - the corpus schema the orchestrator will replay with
# --live, and the evaluation path's handling of hand-written PolicySpec
# fixtures that represent correct compiler output.

_OSHA_FIXTURES = Path(__file__).parent / "fixtures" / "oshacorpus"
_OSHA_CORPUS = json.loads((_OSHA_FIXTURES / "corpus.json").read_text())
_OSHA_BY_CITATION = {row["citation"]: row for row in _OSHA_CORPUS}
_CITATION_RE = re.compile(r"^1910\.\d+(\([a-zA-Z0-9]{1,4}\))*$")


def _osha_row(citation):
    return _OSHA_BY_CITATION[citation]


def _refusal_spec(citation, **kwargs):
    """Hand-written stand-in for CORRECT compiler output on a refuse row:
    plausible fields, unsupported_reason set from the corpus."""
    row = _osha_row(citation)
    assert row["expected"] == "refuse"
    base = dict(
        policy_id=f"exam-{citation}",
        source_text=row["text"],
        predicate=Predicate.MIN_SEPARATION,
        subject_labels=["pallet"],
        object_labels=["safety fence"],
        threshold=1.0,
        unsupported_reason=row["refuse_reason"],
    )
    base.update(kwargs)
    return PolicySpec(**base)


def _compile_spec(citation, subject_labels, object_labels):
    """Spec built FROM the corpus row's expected columns, so the exam's
    numbers are proven to run through the deterministic evaluator."""
    row = _osha_row(citation)
    assert row["expected"] == "compile"
    return PolicySpec(
        policy_id=f"exam-{citation}",
        source_text=row["text"],
        predicate=Predicate(row["expected_predicate"]),
        subject_labels=subject_labels,
        object_labels=object_labels,
        threshold=row["expected_threshold"],
        unit=row["expected_unit"],
    )


def test_osha_corpus_schema_and_coverage():
    assert len(_OSHA_CORPUS) >= 400  # nine full sections, per-paragraph
    counts = {"compile": 0, "refuse": 0, "skip": 0}
    for row in _OSHA_CORPUS:
        assert _CITATION_RE.match(row["citation"]), row["citation"]
        assert row["text"].strip()
        assert row["expected"] in counts
        counts[row["expected"]] += 1
        if row["expected"] == "compile":
            assert row["expected_predicate"] in {p.value for p in Predicate}
            assert row["expected_threshold"] > 0
            assert row["expected_unit"] in {"m", "deg"}
        if row["expected"] == "refuse":
            assert row["refuse_reason"].strip()
    assert counts["compile"] >= 3
    assert counts["refuse"] >= 3


def test_osha_corpus_rebuilds_offline_from_cached_fixtures():
    script = Path(__file__).parents[1] / "scripts" / "oshacorpus.py"
    spec = importlib.util.spec_from_file_location("oshacorpus", script)
    oshacorpus = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oshacorpus)

    assert oshacorpus.build_corpus() == _OSHA_CORPUS


def test_osha_exam_sheet_lists_every_curated_paragraph():
    sheet = Path(__file__).parents[1] / "docs" / "policies" / "osha1910.md"
    lines = [
        line
        for line in sheet.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    curated = [row for row in _OSHA_CORPUS if row["expected"] != "skip"]

    # Exam line N is corpus curated row N: the orchestrator diffs the live
    # compiler's pNN output against the expected columns by position.
    assert len(lines) == len(curated)
    for line, row in zip(lines, curated):
        assert line.startswith(f"- [{row['citation']}]")


def test_designed_refusal_aisle_clearance_has_no_number():
    # 1910.176(a): "sufficient safe clearances" - no threshold to compile.
    row = _osha_row("1910.176(a)")
    assert row["expected"] == "refuse"
    assert (
        "sufficient safe clearances shall be allowed for aisles" in row["text"]
    )

    spec = _refusal_spec(
        "1910.176(a)",
        subject_labels=["pallet"],
        object_labels=["aisle marking"],
    )
    # A scene that would plainly violate the rule if it were ever measured.
    scene = _scene(
        [
            _entity("aisle", "aisle marking", _square(0, 0, 2.0)),
            _entity("blocker", "pallet", _square(0.5, 0.5)),
        ]
    )

    result = evaluate_policy(spec, scene)

    assert result.status.value == "INSUFFICIENT_EVIDENCE"
    assert "not evaluable" in result.warnings[0]
    assert not result.facts and not result.violations


def test_designed_refusal_voltage_table_approach_distance():
    # 1910.333(c)(3)(ii): approach distance keyed to Table S-5 - a voltage
    # lookup, conditional on a non-spatial variable the scene cannot see.
    row = _osha_row("1910.333(c)(3)(ii)")
    assert row["expected"] == "refuse"
    assert "than shown in Table S-5" in row["text"]

    spec = _refusal_spec(
        "1910.333(c)(3)(ii)",
        subject_labels=["person"],
        object_labels=["overhead power line"],
        threshold=3.05,
    )
    scene = _scene(
        [
            _entity("line", "overhead power line", _square(0, 0, 2.0)),
            _entity("worker", "person", _square(2.5, 0.5)),
        ]
    )

    result = evaluate_policy(spec, scene)

    assert result.status.value == "INSUFFICIENT_EVIDENCE"
    assert "not evaluable" in result.warnings[0]
    assert not result.facts and not result.violations


def test_designed_refusal_travel_distance_is_path_not_euclidean():
    # 1910.157(d)(2): 75-foot TRAVEL distance - walking path length, which
    # Euclidean max_separation would systematically understate.
    row = _osha_row("1910.157(d)(2)")
    assert row["expected"] == "refuse"
    assert (
        "travel distance for employees to any extinguisher is 75 feet"
        in row["text"]
    )

    spec = _refusal_spec(
        "1910.157(d)(2)",
        predicate=Predicate.MAX_SEPARATION,
        subject_labels=["fire extinguisher"],
        object_labels=["control panel"],
        threshold=22.9,
    )
    scene = _scene(
        [
            _entity("panel", "control panel", _square(0, 0)),
            _entity("ext", "fire extinguisher", _square(30.0, 0)),
        ]
    )

    result = evaluate_policy(spec, scene)

    assert result.status.value == "INSUFFICIENT_EVIDENCE"
    assert "not evaluable" in result.warnings[0]
    assert not result.facts and not result.violations


def test_osha_cylinder_combustible_separation_compiles_and_evaluates():
    # 1910.253(b)(2)(ii): cylinders at least 20 feet (6.1 m) from highly
    # combustible materials - plain floor-plan min_separation.
    row = _osha_row("1910.253(b)(2)(ii)")
    assert "at least 20 feet (6.1 m) from highly combustible" in row["text"]
    spec = _compile_spec(
        "1910.253(b)(2)(ii)", ["gas cylinder"], ["combustible material"]
    )
    assert spec.predicate is Predicate.MIN_SEPARATION
    assert spec.threshold == 6.1 and spec.unit == "m"

    pile = _entity("pile", "combustible material", _square(0, 0, 1.0))
    close = evaluate_policy(
        spec, _scene([pile, _entity("c1", "gas cylinder", _square(5.0, 0))])
    )
    clear = evaluate_policy(
        spec, _scene([pile, _entity("c2", "gas cylinder", _square(7.5, 0))])
    )

    assert close.status.value == "FAIL"
    assert abs(close.violations[0].measured - 4.0) < 1e-6
    assert close.violations[0].threshold == 6.1
    assert clear.status.value == "PASS" and not clear.violations


def test_osha_portable_generator_clearance_compiles_and_evaluates():
    # 1910.253(f)(5)(i)(B): portable generators not within 10 feet (3 m)
    # of combustible material.
    row = _osha_row("1910.253(f)(5)(i)(B)")
    assert "within 10 feet (3 m) of combustible material" in row["text"]
    spec = _compile_spec(
        "1910.253(f)(5)(i)(B)",
        ["acetylene generator"],
        ["combustible material"],
    )
    assert spec.predicate is Predicate.MIN_SEPARATION
    assert spec.threshold == 3.0 and spec.unit == "m"

    pile = _entity("pile", "combustible material", _square(0, 0, 1.0))
    close = evaluate_policy(
        spec,
        _scene([pile, _entity("g1", "acetylene generator", _square(3.0, 0))]),
    )
    clear = evaluate_policy(
        spec,
        _scene([pile, _entity("g2", "acetylene generator", _square(4.6, 0))]),
    )

    assert close.status.value == "FAIL"
    assert abs(close.violations[0].measured - 2.0) < 1e-6
    assert clear.status.value == "PASS" and not clear.violations


def test_osha_electrical_workspace_keep_clear_compiles_and_evaluates():
    # 1910.303(h)(3): minimum clear work space about over-600V equipment,
    # 914 mm (3.0 ft) - a keep-clear band around the equipment.
    row = _osha_row("1910.303(h)(3)")
    assert "914 mm (3.0 ft) wide" in row["text"]
    spec = _compile_spec("1910.303(h)(3)", ["pallet", "crate"], ["switchgear"])
    assert spec.predicate is Predicate.KEEP_CLEAR
    assert spec.threshold == 0.914 and spec.unit == "m"

    gear = _entity("gear", "switchgear", _square(0, 0, 1.0))
    blocked = evaluate_policy(
        spec, _scene([gear, _entity("p1", "pallet", _square(1.5, 0))])
    )
    clear = evaluate_policy(
        spec, _scene([gear, _entity("p2", "pallet", _square(2.2, 0))])
    )

    assert blocked.status.value == "FAIL"
    assert abs(blocked.violations[0].measured - 0.5) < 1e-6
    assert blocked.violations[0].threshold == 0.914
    assert clear.status.value == "PASS" and not clear.violations


def test_osha_mixed_batch_refuses_without_contaminating_compilables():
    # One reviewed batch mixing compilable and refused specs: refusals
    # abstain loudly, compilables still get their measurement.
    pile = _entity("pile", "combustible material", _square(0, 0, 1.0))
    cylinder = _entity("c1", "gas cylinder", _square(5.0, 0))
    scene = _scene([pile, cylinder])
    specs = [
        _compile_spec(
            "1910.253(b)(2)(ii)", ["gas cylinder"], ["combustible material"]
        ),
        _refusal_spec("1910.176(a)"),
        _refusal_spec("1910.333(c)(3)(ii)"),
        _refusal_spec("1910.157(d)(2)"),
    ]

    results = evaluate_policies(specs, scene)

    assert [r.status.value for r in results] == [
        "FAIL",
        "INSUFFICIENT_EVIDENCE",
        "INSUFFICIENT_EVIDENCE",
        "INSUFFICIENT_EVIDENCE",
    ]
    assert results[0].facts
    for refused in results[1:]:
        assert "not evaluable" in refused.warnings[0]
        assert not refused.facts
