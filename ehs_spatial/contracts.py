from enum import Enum
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator


Vector3 = tuple[float, float, float]
Point2 = tuple[float, float]
Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


class Criterion(BaseModel):
    id: str = "fence_clearance"
    subject: str = "closest movable obstruction"
    object_label: str = "safety fence"
    minimum_clearance_m: float = Field(default=0.6, gt=0)


class CaptureRun(BaseModel):
    run_id: str
    image_paths: list[str] = Field(min_length=4, max_length=4)
    camera_height_m: float = Field(default=1.5, gt=0)
    criterion: Criterion = Field(default_factory=Criterion)


class GeometryFrame(BaseModel):
    frame_id: str
    canonical_image_path: str
    pts3d_path: str
    conf_path: str
    valid_mask_path: str
    camera_to_world: Matrix4
    intrinsics: Matrix3


class Observation2D(BaseModel):
    observation_id: str
    frame_id: str
    label: str
    instance_id: str
    mask_path: str | None = None
    mask_reference: str | dict[str, Any] | None = None
    score: float = Field(ge=0, le=1)
    bbox: tuple[float, float, float, float]
    source_prompt: str

    @model_validator(mode="after")
    def require_one_mask_location(self) -> Self:
        if (self.mask_path is None) == (self.mask_reference is None):
            raise ValueError("provide exactly one of mask_path or mask_reference")
        return self


class Entity3D(BaseModel):
    entity_id: str
    label: str
    observation_ids: list[str]
    centroid_xyz: Vector3
    footprint_xy: list[Point2]
    height_m: float = Field(ge=0)
    evidence_frame_ids: list[str]


class SpatialFact(BaseModel):
    fact_id: str
    predicate: str
    subject_id: str
    object_id: str
    value: float | None = None
    unit: str | None = None
    evidence_frame_ids: list[str]


class SceneMap(BaseModel):
    run_id: str
    floor_plane: tuple[float, float, float, float]
    scale_source: str
    scale_factor: float = Field(gt=0)
    fence_polygon: list[Point2]
    entities: list[Entity3D]
    facts: list[SpatialFact]
    warnings: list[str]


class AssessmentStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ClimbReview(BaseModel):
    verdict: Literal["yes", "no", "uncertain"]
    rationale: str
    fact_ids: list[str]


class Assessment(BaseModel):
    status: AssessmentStatus
    fact_ids: list[str]
    evidence_frame_ids: list[str]
    approximate_distance_m: float | None = Field(default=None, ge=0)
    climb_review: ClimbReview | None = None


class GroundedAnswer(BaseModel):
    answer: str
    fact_ids: list[str]
    evidence_frame_ids: list[str]
