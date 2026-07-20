from dataclasses import dataclass

from shapely.geometry import Polygon

from .contracts import (
    Assessment,
    AssessmentStatus,
    Criterion,
    Entity3D,
    SpatialFact,
)


FENCE_LABEL = "safety fence"
MOVABLE_LABELS = {
    "material cart",
    "pallet",
    "crate",
    "step ladder",
    "portable work platform",
}


@dataclass(frozen=True)
class _RuleResult:
    fence_polygon: list[tuple[float, float]]
    facts: list[SpatialFact]
    assessment: Assessment
    warnings: list[str]
    selected_entity_id: str | None


def _insufficient(warning: str) -> _RuleResult:
    return _RuleResult(
        fence_polygon=[],
        facts=[],
        assessment=Assessment(
            status=AssessmentStatus.INSUFFICIENT_EVIDENCE,
            fact_ids=[],
            evidence_frame_ids=[],
            approximate_distance_m=None,
        ),
        warnings=[warning],
        selected_entity_id=None,
    )


def _assess_clearance(entities: list[Entity3D], criterion: Criterion) -> _RuleResult:
    valid_fences = []
    discarded_fence_fragments = 0
    for entity in entities:
        if entity.label != FENCE_LABEL:
            continue
        polygon = Polygon(entity.footprint_xy)
        if len(set(entity.evidence_frame_ids)) < 3 or polygon.area < 0.25:
            # A fence fragment failing the gates must never vanish silently:
            # measuring clearance against a partial hull is the false-PASS mode.
            discarded_fence_fragments += 1
            continue
        valid_fences.append((entity, polygon))
    if len(valid_fences) != 1:
        return _insufficient(
            "expected exactly one safety fence with at least 3 evidence frames and 0.25 m2 area"
        )
    fence, fence_polygon = valid_fences[0]
    fence_warnings = (
        [
            f"{discarded_fence_fragments} safety fence fragment(s) discarded by "
            "evidence gates; clearance may be measured against a partial fence"
        ]
        if discarded_fence_fragments
        else []
    )

    valid_movables = []
    for entity in entities:
        if entity.label not in MOVABLE_LABELS:
            continue
        polygon = Polygon(entity.footprint_xy)
        if (
            len(set(entity.evidence_frame_ids)) >= 2
            and polygon.area >= 0.0025
            and entity.height_m >= 0.05
        ):
            valid_movables.append((entity, polygon))
    if not valid_movables:
        result = _insufficient(
            "no movable entity has at least 2 evidence frames, 0.0025 m2 area, and 0.05 m height"
        )
        return _RuleResult(
            fence_polygon=list(fence.footprint_xy),
            facts=result.facts,
            assessment=result.assessment,
            warnings=[*fence_warnings, *result.warnings],
            selected_entity_id=None,
        )

    movable, movable_polygon = min(
        valid_movables,
        key=lambda item: (
            0.0
            if fence_polygon.intersects(item[1])
            else item[1].distance(fence_polygon.boundary),
            item[0].entity_id,
        ),
    )
    inside_or_intersects = fence_polygon.intersects(movable_polygon)
    distance = (
        0.0
        if inside_or_intersects
        else float(movable_polygon.distance(fence_polygon.boundary))
    )
    status = (
        AssessmentStatus.FAIL
        if inside_or_intersects or distance < criterion.minimum_clearance_m
        else AssessmentStatus.PASS
    )
    combined_evidence = sorted(
        set(fence.evidence_frame_ids) | set(movable.evidence_frame_ids)
    )
    facts = [
        SpatialFact(
            fact_id="fact-inside-or-intersects",
            predicate="inside_or_intersects",
            subject_id=movable.entity_id,
            object_id=fence.entity_id,
            value=float(inside_or_intersects),
            unit="boolean",
            evidence_frame_ids=combined_evidence,
        ),
        SpatialFact(
            fact_id="fact-minimum-boundary-clearance",
            predicate="minimum_boundary_clearance",
            subject_id=movable.entity_id,
            object_id=fence.entity_id,
            value=distance,
            unit="m",
            evidence_frame_ids=combined_evidence,
        ),
        SpatialFact(
            fact_id="fact-object-height",
            predicate="object_height",
            subject_id=movable.entity_id,
            object_id=movable.entity_id,
            value=movable.height_m,
            unit="m",
            evidence_frame_ids=movable.evidence_frame_ids,
        ),
    ]
    # Spatial-state facts are evidence-only: they never feed the verdict.
    for name, unit, value in (
        ("orientation", "deg", movable.orientation_deg),
        ("tilt", "deg", movable.tilt_deg),
        ("overhang", "m", movable.overhang_m),
    ):
        if value is None:
            continue
        facts.append(
            SpatialFact(
                fact_id=f"fact-object-{name}",
                predicate=f"object_{name}",
                subject_id=movable.entity_id,
                object_id=movable.entity_id,
                value=value,
                unit=unit,
                evidence_frame_ids=movable.evidence_frame_ids,
            )
        )
    return _RuleResult(
        fence_polygon=list(fence.footprint_xy),
        facts=facts,
        assessment=Assessment(
            status=status,
            fact_ids=[fact.fact_id for fact in facts],
            evidence_frame_ids=combined_evidence,
            approximate_distance_m=distance,
        ),
        warnings=fence_warnings,
        selected_entity_id=movable.entity_id,
    )
