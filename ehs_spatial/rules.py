from dataclasses import dataclass

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .contracts import (
    Assessment,
    AssessmentStatus,
    Criterion,
    Entity3D,
    SpatialFact,
)


FENCE_LABEL = "safety fence"

# Measured accuracy of the two capture tiers, from real laser-GT evals:
# multi-view real-photo MAE 14-18 cm (docs/reviews/2026-07-21-mvp-boundary-map.md),
# mono ~30-46 cm (docs/reviews/2026-08-24-moge-auto-anchor.md). A distance
# inside the budget around the threshold cannot honestly pick a side, so
# the verdict says NEEDS_REVIEW instead of flipping on noise.
ERROR_BUDGET_MULTIVIEW_M = 0.20
ERROR_BUDGET_MONO_M = 0.35
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


def _assess_clearance(
    entities: list[Entity3D],
    criterion: Criterion,
    capture_frame_count: int = 4,
) -> _RuleResult:
    # Frame-evidence gates scale down for reduced captures: a single-photo
    # run can never satisfy a 3-frame gate, so the gate becomes "every
    # captured frame". The reduced-redundancy warning is added by the caller.
    fence_frame_gate = min(3, capture_frame_count)
    movable_frame_gate = min(2, capture_frame_count)
    valid_fences = []
    discarded_fence_fragments = 0
    for entity in entities:
        if entity.label != FENCE_LABEL:
            continue
        polygon = Polygon(entity.footprint_xy)
        if len(set(entity.evidence_frame_ids)) < fence_frame_gate or polygon.area < 0.25:
            # A fence fragment failing the gates must never vanish silently:
            # measuring clearance against a partial hull is the false-PASS mode.
            discarded_fence_fragments += 1
            continue
        valid_fences.append((entity, polygon))
    if not valid_fences:
        return _insufficient(
            f"no safety fence passed the evidence gates (at least "
            f"{fence_frame_gate} evidence frame(s) and 0.25 m2 area)"
        )
    if len(valid_fences) == 1:
        fence, fence_polygon = valid_fences[0]
        fence_polygon_out = list(fence.footprint_xy)
        fence_evidence = set(fence.evidence_frame_ids)
        merge_warnings = []
    else:
        # One physical fence often reconstructs as several clusters (thin
        # rails, occlusion, single-view depth). Merging the gated fragments
        # into one hull is what clustering would have produced had they
        # connected; the largest fragment lends its entity_id to the facts.
        fence = max(
            valid_fences, key=lambda item: (item[1].area, item[0].entity_id)
        )[0]
        fence_polygon = unary_union(
            [polygon for _, polygon in valid_fences]
        ).convex_hull
        fence_polygon_out = [
            (float(x), float(y))
            for x, y in list(fence_polygon.exterior.coords)[:-1]
        ]
        fence_evidence = set().union(
            *(set(entity.evidence_frame_ids) for entity, _ in valid_fences)
        )
        merge_warnings = [
            f"{len(valid_fences)} safety fence segments merged into a single "
            "boundary hull; clearance is measured against the merged hull"
        ]
    fence_warnings = merge_warnings + (
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
            len(set(entity.evidence_frame_ids)) >= movable_frame_gate
            and polygon.area >= 0.0025
            and entity.height_m >= 0.05
        ):
            valid_movables.append((entity, polygon))
    if not valid_movables:
        result = _insufficient(
            f"no movable entity has at least {movable_frame_gate} evidence "
            "frame(s), 0.0025 m2 area, and 0.05 m height"
        )
        return _RuleResult(
            fence_polygon=fence_polygon_out,
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
    error_budget = (
        ERROR_BUDGET_MULTIVIEW_M
        if capture_frame_count >= 2
        else ERROR_BUDGET_MONO_M
    )
    threshold = criterion.minimum_clearance_m
    if inside_or_intersects or distance < threshold - error_budget:
        status = AssessmentStatus.FAIL
    elif distance > threshold + error_budget:
        status = AssessmentStatus.PASS
    else:
        status = AssessmentStatus.NEEDS_REVIEW
        fence_warnings = [
            *fence_warnings,
            f"measured clearance {distance:.2f} m is within the "
            f"±{error_budget:.2f} m error budget of the "
            f"{threshold:.2f} m threshold; the geometry cannot honestly "
            "pick a side",
        ]
    combined_evidence = sorted(
        fence_evidence | set(movable.evidence_frame_ids)
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
            fact_id="fact-clearance-error-budget",
            predicate="clearance_error_budget",
            subject_id=movable.entity_id,
            object_id=fence.entity_id,
            value=error_budget,
            unit="m",
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
        fence_polygon=fence_polygon_out,
        facts=facts,
        assessment=Assessment(
            status=status,
            fact_ids=[fact.fact_id for fact in facts],
            evidence_frame_ids=combined_evidence,
            approximate_distance_m=distance,
            distance_error_budget_m=error_budget,
        ),
        warnings=fence_warnings,
        selected_entity_id=movable.entity_id,
    )
