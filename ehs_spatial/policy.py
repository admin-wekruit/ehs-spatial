"""Written policy -> deterministic geometric predicate.

A policy arrives as prose ("movable equipment must stay at least 0.6 m clear
of machine guarding"). A language model may translate it into a PolicySpec —
a small, closed, human-reviewable structure — and nothing else. Evaluation is
pure geometry over the SceneMap: the same shapely arithmetic the clearance
rule already uses, with the same evidence gates and the same abstention path.

The split is the whole point. Translation is reviewable before it ever runs;
verdicts never depend on model weights, only on the reviewed spec plus the
measured scene.
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Models live in contracts.py (CaptureRun.policies needs PolicySpec);
# re-exported here so existing importers keep working.
from .contracts import (
    AssessmentStatus,
    Entity3D,
    PolicyResult,
    PolicySpec,
    Predicate,
    SceneMap,
    Severity,
    SpatialFact,
    Violation,
)


_SELF_PREDICATES = {Predicate.MAX_HEIGHT, Predicate.MAX_TILT}
# Same discipline as the clearance rule: a footprint too small or seen in too
# few frames is not evidence, it is noise.
MIN_EVIDENCE_FRAMES = 2
MIN_FOOTPRINT_M2 = 0.0025


def _valid(entities: list[Entity3D], labels: set[str], frame_gate: int):
    out = []
    for entity in entities:
        if entity.label not in labels:
            continue
        polygon = Polygon(entity.footprint_xy)
        if not polygon.is_valid or polygon.area < MIN_FOOTPRINT_M2:
            continue
        if len(set(entity.evidence_frame_ids)) < frame_gate:
            continue
        out.append((entity, polygon))
    return out


def _gap(a: Polygon, b: Polygon) -> float:
    return 0.0 if a.intersects(b) else float(a.distance(b))


def evaluate_policy(
    spec: PolicySpec,
    scene: SceneMap,
    *,
    capture_frame_count: int = 4,
) -> PolicyResult:
    """Deterministic evaluation. No model is consulted here, ever."""
    if spec.unsupported_reason:
        return PolicyResult(
            policy_id=spec.policy_id,
            status=AssessmentStatus.INSUFFICIENT_EVIDENCE,
            warnings=[f"policy not evaluable: {spec.unsupported_reason}"],
        )

    frame_gate = min(MIN_EVIDENCE_FRAMES, max(1, capture_frame_count))
    subjects = _valid(scene.entities, set(spec.subject_labels), frame_gate)
    if not subjects:
        return PolicyResult(
            policy_id=spec.policy_id,
            status=AssessmentStatus.INSUFFICIENT_EVIDENCE,
            warnings=[
                "no entity matching subject labels "
                f"{sorted(spec.subject_labels)} passed the evidence gates"
            ],
        )

    objects: list[tuple[Entity3D, Polygon]] = []
    if spec.predicate not in _SELF_PREDICATES:
        objects = _valid(scene.entities, set(spec.object_labels), frame_gate)
        if not objects:
            return PolicyResult(
                policy_id=spec.policy_id,
                status=AssessmentStatus.INSUFFICIENT_EVIDENCE,
                warnings=[
                    "no entity matching object labels "
                    f"{sorted(spec.object_labels)} passed the evidence gates"
                ],
            )

    violations: list[Violation] = []
    facts: list[SpatialFact] = []
    evidence: set[str] = set()

    def record(subject: Entity3D, obj: Entity3D | None, value: float) -> None:
        evidence.update(subject.evidence_frame_ids)
        if obj is not None:
            evidence.update(obj.evidence_frame_ids)
        facts.append(
            SpatialFact(
                fact_id=f"fact-{spec.policy_id}-{len(facts) + 1:02d}",
                predicate=spec.predicate.value,
                subject_id=subject.entity_id,
                object_id=obj.entity_id if obj else subject.entity_id,
                value=round(value, 4),
                unit=spec.unit,
                evidence_frame_ids=sorted(
                    set(subject.evidence_frame_ids)
                    | (set(obj.evidence_frame_ids) if obj else set())
                ),
            )
        )

    if spec.predicate is Predicate.MAX_HEIGHT:
        for subject, _ in subjects:
            record(subject, None, subject.height_m)
            if subject.height_m > spec.threshold:
                violations.append(
                    Violation(
                        subject_id=subject.entity_id,
                        measured=round(subject.height_m, 4),
                        threshold=spec.threshold,
                        unit=spec.unit,
                    )
                )
    elif spec.predicate is Predicate.MAX_TILT:
        measured_any = False
        for subject, _ in subjects:
            if subject.tilt_deg is None:
                continue
            measured_any = True
            record(subject, None, subject.tilt_deg)
            if subject.tilt_deg > spec.threshold:
                violations.append(
                    Violation(
                        subject_id=subject.entity_id,
                        measured=round(subject.tilt_deg, 4),
                        threshold=spec.threshold,
                        unit=spec.unit,
                    )
                )
        if not measured_any:
            return PolicyResult(
                policy_id=spec.policy_id,
                status=AssessmentStatus.INSUFFICIENT_EVIDENCE,
                warnings=["no subject has a measurable tilt"],
            )
    elif spec.predicate is Predicate.MAX_SEPARATION:
        # "must be within X of" — the nearest object decides, so a subject
        # far from every object is the violation.
        merged = unary_union([polygon for _, polygon in objects])
        for subject, polygon in subjects:
            nearest, gap = min(
                ((obj, _gap(polygon, obj_poly)) for obj, obj_poly in objects),
                key=lambda item: item[1],
            )
            record(subject, nearest, gap)
            if gap > spec.threshold:
                violations.append(
                    Violation(
                        subject_id=subject.entity_id,
                        object_id=nearest.entity_id,
                        measured=round(gap, 4),
                        threshold=spec.threshold,
                        unit=spec.unit,
                    )
                )
        _ = merged
    elif spec.predicate is Predicate.NOT_INSIDE:
        for subject, polygon in subjects:
            for obj, obj_poly in objects:
                overlap = polygon.intersection(obj_poly).area
                record(subject, obj, overlap)
                if overlap > 0:
                    violations.append(
                        Violation(
                            subject_id=subject.entity_id,
                            object_id=obj.entity_id,
                            measured=round(overlap, 4),
                            threshold=spec.threshold,
                            unit="m2",
                        )
                    )
    else:
        # MIN_SEPARATION and KEEP_CLEAR share the arithmetic; they differ in
        # which side owns the zone, which only changes how it reads in a
        # report, not what is measured.
        for subject, polygon in subjects:
            for obj, obj_poly in objects:
                if obj.entity_id == subject.entity_id:
                    continue
                gap = _gap(polygon, obj_poly)
                record(subject, obj, gap)
                if gap < spec.threshold:
                    violations.append(
                        Violation(
                            subject_id=subject.entity_id,
                            object_id=obj.entity_id,
                            measured=round(gap, 4),
                            threshold=spec.threshold,
                            unit=spec.unit,
                        )
                    )

    if not facts:
        return PolicyResult(
            policy_id=spec.policy_id,
            status=AssessmentStatus.INSUFFICIENT_EVIDENCE,
            warnings=["policy matched entities but produced no measurement"],
        )
    return PolicyResult(
        policy_id=spec.policy_id,
        status=AssessmentStatus.FAIL if violations else AssessmentStatus.PASS,
        violations=violations,
        facts=facts,
        evidence_frame_ids=sorted(evidence),
    )


def evaluate_policies(
    specs: list[PolicySpec],
    scene: SceneMap,
    *,
    capture_frame_count: int = 4,
) -> list[PolicyResult]:
    return [
        evaluate_policy(spec, scene, capture_frame_count=capture_frame_count)
        for spec in specs
    ]


__all__ = [
    "Predicate",
    "PolicyResult",
    "PolicySpec",
    "Severity",
    "Violation",
    "evaluate_policies",
    "evaluate_policy",
]
