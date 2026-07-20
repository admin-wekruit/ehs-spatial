from pathlib import Path

from .contracts import (
    Assessment,
    AssessmentStatus,
    Criterion,
    GeometryFrame,
    Observation2D,
    SceneMap,
)
from .geometry import _build_geometry
from .rules import _assess_clearance
from .topdown import _render_topdown


def build_scene_and_assess(
    run_id: str,
    frames: list[GeometryFrame],
    observations: list[Observation2D],
    camera_height_m: float,
    criterion: Criterion,
    topdown_path: str | Path | None = None,
) -> tuple[SceneMap, Assessment]:
    geometry = _build_geometry(frames, observations, camera_height_m)
    capture_warnings = (
        [
            f"reduced capture ({len(frames)} view(s) instead of 4): evidence "
            "redundancy and cross-view confirmation are weaker"
        ]
        if len(frames) < 4
        else []
    )
    if geometry.transform is None:
        scene = SceneMap(
            run_id=run_id,
            floor_plane=None,
            scale_source=None,
            scale_factor=None,
            fence_polygon=[],
            entities=[],
            facts=[],
            warnings=[*capture_warnings, *geometry.warnings],
        )
        assessment = Assessment(
            status=AssessmentStatus.INSUFFICIENT_EVIDENCE,
            fact_ids=[],
            evidence_frame_ids=[],
            approximate_distance_m=None,
        )
        if topdown_path is not None:
            _render_topdown(topdown_path, [], [], None, assessment)
        return scene, assessment

    rule = _assess_clearance(
        geometry.entities, criterion, capture_frame_count=len(frames)
    )
    scene = SceneMap(
        run_id=run_id,
        floor_plane=geometry.transform.plane,
        scale_source="camera_height",
        scale_factor=geometry.transform.scale_factor,
        fence_polygon=rule.fence_polygon,
        entities=geometry.entities,
        facts=rule.facts,
        warnings=[*capture_warnings, *geometry.warnings, *rule.warnings],
    )
    if topdown_path is not None:
        _render_topdown(
            topdown_path,
            geometry.entities,
            rule.fence_polygon,
            rule.selected_entity_id,
            rule.assessment,
        )
    return scene, rule.assessment


__all__ = ["build_scene_and_assess"]
