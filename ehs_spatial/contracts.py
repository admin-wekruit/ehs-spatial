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


class Predicate(str, Enum):
    """Closed vocabulary for compiled policies. A compiler that cannot
    express a policy must say so rather than approximate it, so unsupported
    rules fail loudly at compile time instead of quietly at verdict time."""

    MIN_SEPARATION = "min_separation"
    MAX_SEPARATION = "max_separation"
    KEEP_CLEAR = "keep_clear"
    NOT_INSIDE = "not_inside"
    MAX_HEIGHT = "max_height"
    MIN_HEIGHT = "min_height"
    MAX_TILT = "max_tilt"


class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    ADVISORY = "advisory"


class PolicySpec(BaseModel):
    """One measurable requirement, compiled from prose and reviewable."""

    policy_id: str
    source_text: str
    predicate: Predicate
    subject_labels: list[str] = Field(min_length=1)
    # Empty for self-referential predicates (MAX_HEIGHT, MAX_TILT).
    object_labels: list[str] = Field(default_factory=list)
    threshold: float = Field(gt=0)
    unit: Literal["m", "deg"] = "m"
    severity: Severity = Severity.MAJOR
    rationale: str = ""
    # Set by the compiler when the prose carries a requirement this
    # vocabulary cannot express; such specs are never evaluated.
    unsupported_reason: str | None = None

    def requires_labels(self) -> set[str]:
        return set(self.subject_labels) | set(self.object_labels)


class CaptureRun(BaseModel):
    run_id: str
    image_paths: list[str] = Field(min_length=1, max_length=4)
    # None = the operator supplied no height; the scale chain then relies on
    # the auto anchor and, failing that, the model's native scale.
    camera_height_m: float | None = Field(default=1.5, gt=0)
    scale_preference: Literal["auto", "camera_height"] = "auto"
    operator: str = "unknown"
    # Empty keeps the legacy Criterion demo-rule path exactly as before.
    policies: list[PolicySpec] = Field(default_factory=list)
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
    # Spatial-state primitives; each is None when the cluster is too
    # degenerate to measure. orientation_deg: undirected yaw of the XY
    # footprint's first principal axis, so range is [0, 180). tilt_deg: angle
    # between the 3D first principal axis and vertical +Z; flat/squat objects
    # with a horizontal dominant axis read ~90. overhang_m: max XY protrusion
    # of upper-band points beyond the base-band convex hull, metres.
    orientation_deg: float | None = Field(default=None, ge=0, lt=180)
    tilt_deg: float | None = Field(default=None, ge=0, le=90)
    overhang_m: float | None = Field(default=None, ge=0)


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
    floor_plane: tuple[float, float, float, float] | None
    # "moge_anchor" | "camera_height" | "model_native"
    scale_source: str | None
    scale_factor: float | None = Field(gt=0)
    scale_confidence: float | None = Field(default=None, ge=0, le=1)
    fence_polygon: list[Point2]
    entities: list[Entity3D]
    facts: list[SpatialFact]
    warnings: list[str]


class AssessmentStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    # The measured distance sits inside the capture tier's error budget
    # around the threshold: the geometry cannot honestly pick a side.
    NEEDS_REVIEW = "NEEDS_REVIEW"
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
    # The +/- band the verdict honoured, from the capture tier's measured
    # accuracy — display distances as "d ± budget", never as bare decimals.
    distance_error_budget_m: float | None = Field(default=None, ge=0)
    climb_review: ClimbReview | None = None


class Violation(BaseModel):
    subject_id: str
    object_id: str | None = None
    measured: float
    threshold: float
    unit: str


class PolicyResult(BaseModel):
    policy_id: str
    status: AssessmentStatus
    violations: list[Violation] = Field(default_factory=list)
    facts: list[SpatialFact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence_frame_ids: list[str] = Field(default_factory=list)


class ProviderManifest(BaseModel):
    mapanything_model_id: str
    sam_endpoint: str
    gemini_model: str
    moge_version: str
    code_version: str = "unknown"


class RunManifest(BaseModel):
    """Operational provenance for one run. Kept out of Assessment on
    purpose: the verdict contract feeds prompts and pack fixtures, while
    provenance changes at deployment cadence."""

    run_id: str
    created_at: str  # UTC ISO-8601
    operator: str = "unknown"
    # "video-mono" is the uncalibrated fixed-camera video tier (ehs_spatial/
    # video.py): floor plane and intrinsics recovered from MoGe-2 mono depth.
    capture_tier: Literal["mono", "multiview", "video-mono"]
    providers: ProviderManifest


class ReviewDisposition(BaseModel):
    """A human's ruling on a machine verdict. The machine verdict is never
    edited; the disposition sits beside it."""

    run_id: str
    reviewer: str
    decision: Literal["confirmed", "overridden"]
    overridden_status: AssessmentStatus | None = None
    reason: str = ""
    created_at: str  # UTC ISO-8601


class GroundedAnswer(BaseModel):
    answer: str
    fact_ids: list[str]
    evidence_frame_ids: list[str]
