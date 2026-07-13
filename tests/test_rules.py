from ehs_spatial.contracts import Criterion, Entity3D
from ehs_spatial.rules import _assess_clearance


def _entity(
    entity_id: str,
    label: str,
    footprint: list[tuple[float, float]],
    frames: list[str],
) -> Entity3D:
    return Entity3D(
        entity_id=entity_id,
        label=label,
        observation_ids=[f"obs-{frame}" for frame in frames],
        centroid_xyz=(0.0, 0.0, 0.2),
        footprint_xy=footprint,
        height_m=0.2,
        evidence_frame_ids=frames,
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
