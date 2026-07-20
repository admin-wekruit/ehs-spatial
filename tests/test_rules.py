from ehs_spatial.contracts import Criterion, Entity3D
from ehs_spatial.rules import _assess_clearance


def _entity(
    entity_id: str,
    label: str,
    footprint: list[tuple[float, float]],
    frames: list[str],
    **spatial_state: float,
) -> Entity3D:
    return Entity3D(
        entity_id=entity_id,
        label=label,
        observation_ids=[f"obs-{frame}" for frame in frames],
        centroid_xyz=(0.0, 0.0, 0.2),
        footprint_xy=footprint,
        height_m=0.2,
        evidence_frame_ids=frames,
        **spatial_state,
    )


def test_contained_movable_dominates_outside_clear_candidate():
    fence = _entity(
        "fence",
        "safety fence",
        [(0, 0), (2, 0), (2, 2), (0, 2)],
        ["frame-1", "frame-2", "frame-3"],
    )
    contained = _entity(
        "contained",
        "pallet",
        [(0.8, 0.8), (1.0, 0.8), (1.0, 1.0), (0.8, 1.0)],
        ["frame-1", "frame-2"],
    )
    outside = _entity(
        "outside",
        "crate",
        [(2.7, 0.8), (2.9, 0.8), (2.9, 1.0), (2.7, 1.0)],
        ["frame-1", "frame-2"],
    )

    result = _assess_clearance([fence, contained, outside], Criterion())

    assert result.selected_entity_id == "contained"
    assert result.assessment.status.value == "FAIL"
    assert result.assessment.approximate_distance_m == 0.0
    assert all(fact.subject_id == "contained" for fact in result.facts)


def test_discarded_fence_fragments_surface_a_warning_instead_of_vanishing():
    fence = _entity(
        "fence",
        "safety fence",
        [(0, 0), (2, 0), (2, 2), (0, 2)],
        ["frame-1", "frame-2", "frame-3"],
    )
    fragment = _entity(
        "fence-fragment",
        "safety fence",
        [(2.5, 0.0), (2.56, 0.0), (2.56, 2.0), (2.5, 2.0)],
        ["frame-1"],
    )
    movable = _entity(
        "pallet",
        "pallet",
        [(2.7, 0.8), (2.9, 0.8), (2.9, 1.0), (2.7, 1.0)],
        ["frame-1", "frame-2"],
    )

    result = _assess_clearance([fence, fragment, movable], Criterion())

    assert result.assessment.status.value == "PASS"
    assert any("fence fragment" in warning for warning in result.warnings)


def test_single_clean_fence_produces_no_fragment_warning():
    fence = _entity(
        "fence",
        "safety fence",
        [(0, 0), (2, 0), (2, 2), (0, 2)],
        ["frame-1", "frame-2", "frame-3"],
    )
    movable = _entity(
        "pallet",
        "pallet",
        [(2.7, 0.8), (2.9, 0.8), (2.9, 1.0), (2.7, 1.0)],
        ["frame-1", "frame-2"],
    )

    result = _assess_clearance([fence, movable], Criterion())

    assert result.warnings == []


_BASE_FACT_IDS = [
    "fact-inside-or-intersects",
    "fact-minimum-boundary-clearance",
    "fact-object-height",
]


def test_spatial_state_fields_emit_facts_without_changing_verdict():
    fence = _entity(
        "fence",
        "safety fence",
        [(0, 0), (2, 0), (2, 2), (0, 2)],
        ["frame-1", "frame-2", "frame-3"],
    )
    movable = _entity(
        "pallet",
        "pallet",
        [(2.7, 0.8), (2.9, 0.8), (2.9, 1.0), (2.7, 1.0)],
        ["frame-1", "frame-2"],
        orientation_deg=45.0,
        tilt_deg=15.0,
        overhang_m=0.3,
    )

    result = _assess_clearance([fence, movable], Criterion())

    facts_by_id = {fact.fact_id: fact for fact in result.facts}
    assert sorted(facts_by_id) == sorted(
        _BASE_FACT_IDS
        + ["fact-object-orientation", "fact-object-tilt", "fact-object-overhang"]
    )
    for fact_id, value, unit in (
        ("fact-object-orientation", 45.0, "deg"),
        ("fact-object-tilt", 15.0, "deg"),
        ("fact-object-overhang", 0.3, "m"),
    ):
        fact = facts_by_id[fact_id]
        assert fact.value == value
        assert fact.unit == unit
        assert fact.subject_id == "pallet"
        assert fact.object_id == "pallet"
        assert fact.evidence_frame_ids == ["frame-1", "frame-2"]
    assert result.assessment.fact_ids == [fact.fact_id for fact in result.facts]

    bare = _assess_clearance(
        [fence, _entity("pallet", "pallet", movable.footprint_xy, ["frame-1", "frame-2"])],
        Criterion(),
    )
    assert bare.assessment.status == result.assessment.status
    assert bare.assessment.approximate_distance_m == result.assessment.approximate_distance_m


def test_none_spatial_state_fields_emit_only_the_original_facts():
    fence = _entity(
        "fence",
        "safety fence",
        [(0, 0), (2, 0), (2, 2), (0, 2)],
        ["frame-1", "frame-2", "frame-3"],
    )
    movable = _entity(
        "pallet",
        "pallet",
        [(2.7, 0.8), (2.9, 0.8), (2.9, 1.0), (2.7, 1.0)],
        ["frame-1", "frame-2"],
    )

    result = _assess_clearance([fence, movable], Criterion())

    assert [fact.fact_id for fact in result.facts] == _BASE_FACT_IDS
    assert result.assessment.fact_ids == _BASE_FACT_IDS


def test_single_frame_capture_scales_evidence_gates_down():
    fence = _entity(
        "fence",
        "safety fence",
        [(0, 0), (2, 0), (2, 2), (0, 2)],
        ["frame-1"],
    )
    movable = _entity(
        "movable",
        "pallet",
        [(0.8, 0.8), (1.0, 0.8), (1.0, 1.0), (0.8, 1.0)],
        ["frame-1"],
    )

    result = _assess_clearance([fence, movable], Criterion(), capture_frame_count=1)

    assert result.assessment.status.value == "FAIL"
    assert result.selected_entity_id == "movable"


def test_default_capture_keeps_full_evidence_gates():
    fence = _entity(
        "fence",
        "safety fence",
        [(0, 0), (2, 0), (2, 2), (0, 2)],
        ["frame-1"],
    )
    movable = _entity(
        "movable",
        "pallet",
        [(0.8, 0.8), (1.0, 0.8), (1.0, 1.0), (0.8, 1.0)],
        ["frame-1"],
    )

    result = _assess_clearance([fence, movable], Criterion())

    assert result.assessment.status.value == "INSUFFICIENT_EVIDENCE"
