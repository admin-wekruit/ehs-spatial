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
    camera_height_m: float | None,
    criterion: Criterion,
    topdown_path: str | Path | None = None,
    plan_view_path: str | Path | None = None,
    semantic_ply_path: str | Path | None = None,
    cloud_views_paths: tuple[str | Path, str | Path] | None = None,
    scale_factor_override: float | None = None,
    scale_source: str = "camera_height",
    scale_confidence: float | None = None,
    scale_warnings: list[str] | None = None,
) -> tuple[SceneMap, Assessment]:
    geometry = _build_geometry(
        frames,
        observations,
        camera_height_m,
        scale_factor_override=scale_factor_override,
    )
    capture_warnings = list(scale_warnings or []) + (
        [
            f"reduced capture ({len(frames)} view(s) instead of 4): evidence "
            "redundancy and cross-view confirmation are weaker"
        ]
        if len(frames) < 4
        else []
    )
    # gravity sanity: in an upright photo the fitted floor normal points
    # roughly along the camera's down axis (+y). A sideways normal means
    # the capture is rotated (e.g. an unapplied EXIF orientation) and every
    # measurement downstream would be nonsense — say so loudly.
    if geometry.transform is not None:
        import numpy as _np

        a, b, c, _ = geometry.transform.plane
        normal = _np.array([a, b, c], float)
        normal /= max(float(_np.linalg.norm(normal)), 1e-9)
        if abs(normal[1]) < 0.7:
            capture_warnings.append(
                "capture orientation suspect: fitted floor normal is "
                f"{_np.degrees(_np.arccos(abs(normal[1]))):.0f}° off the "
                "camera's vertical — the photo may be rotated (check EXIF "
                "orientation); measurements are unreliable until re-captured "
                "upright"
            )
    if geometry.transform is None:
        scene = SceneMap(
            run_id=run_id,
            floor_plane=None,
            scale_source=None,
            scale_factor=None,
            scale_confidence=None,
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
        scale_source=scale_source,
        scale_factor=geometry.transform.scale_factor,
        scale_confidence=scale_confidence,
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
    if plan_view_path is not None:
        from .planview import _render_plan_view

        _render_plan_view(
            plan_view_path,
            geometry.entities,
            rule.fence_polygon,
            rule.selected_entity_id,
            rule.assessment,
        )
    if semantic_ply_path is not None:
        from .planview import _write_semantic_ply

        _write_semantic_ply(
            semantic_ply_path, frames, observations, geometry.transform
        )
    if cloud_views_paths is not None:
        from .planview import _render_cloud_views

        _render_cloud_views(
            cloud_views_paths[0],
            cloud_views_paths[1],
            frames,
            observations,
            geometry.transform,
        )
    assessment = rule.assessment
    if scale_source == "model_native" and assessment.status in (
        AssessmentStatus.PASS,
        AssessmentStatus.FAIL,
    ):
        # An unanchored gauge cannot honestly certify either side of the
        # threshold; the distance may be off by a large factor.
        assessment = assessment.model_copy(
            update={"status": AssessmentStatus.NEEDS_REVIEW}
        )
        scene = scene.model_copy(
            update={
                "warnings": [
                    *scene.warnings,
                    "scale is model-native (unanchored); verdict demoted to "
                    "review because the measurement gauge is unknown",
                ]
            }
        )
    return scene, assessment


__all__ = ["build_scene_and_assess"]
