"""Versioned wire contracts and validation at the platform boundary."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from decimal import Decimal
from typing import Annotated, Any, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlatformError(Exception):
    def __init__(self, code: str, http_status: int = 400, **params: Any):
        self.code, self.status, self.params = code, http_status, params
        super().__init__(code)


def canonical(value: Any) -> bytes:
    def jsonb_numbers(item):
        if isinstance(item, float) and math.isfinite(item):
            # JSONB removes a zero's sign and expands positive exponents to
            # decimal integers. Preserve 1.0 and all other existing encodings.
            if item == 0:
                return 0.0
            if "e+" in repr(item):
                return int(Decimal(str(item)))
        if isinstance(item, dict):
            return {key: jsonb_numbers(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [jsonb_numbers(value) for value in item]
        return item

    return json.dumps(jsonb_numbers(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def capability_sha(token: str) -> str:
    if not isinstance(token, str) or not re.fullmatch(r"pcap_v1_[A-Za-z0-9_-]{43}", token):
        raise PlatformError("invalid_capability", 401)
    try:
        raw = base64.urlsafe_b64decode(token[8:] + "=")
    except ValueError:
        raise PlatformError("invalid_capability", 401) from None
    if len(raw) != 32 or base64.urlsafe_b64encode(raw).decode().rstrip("=") != token[8:]:
        raise PlatformError("invalid_capability", 401)
    return hashlib.sha256(token.encode()).hexdigest()


class DTO(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CreateProject(DTO):
    requestId: UUID
    title: str = Field(default="Untitled project", min_length=1, max_length=240)
    target: Literal["scene", "standalone_object"] = "scene"


class CreateBranch(DTO):
    requestId: UUID
    sourceRevisionId: UUID
    title: str = Field(min_length=1, max_length=240)
    kind: Literal["reconstruction", "planning"] = "planning"


class EditRequest(DTO):
    requestId: UUID
    branchId: UUID
    baseRevisionId: UUID
    operations: list[dict[str, Any]] = Field(default_factory=list, max_length=500)
    undoOf: UUID | None = None
    redoOf: UUID | None = None
    label: str | None = Field(default=None, min_length=1, max_length=240)
    agentTurnId: UUID | None = None

    @model_validator(mode="after")
    def single_action(self):
        if sum((bool(self.operations) or self.label is not None, self.undoOf is not None, self.redoOf is not None)) != 1:
            raise ValueError("Exactly one of operations, undoOf, redoOf is required")
        return self


class ForkRequest(DTO):
    requestId: UUID
    sourceRevisionId: UUID
    title: str = Field(default="Fork", min_length=1, max_length=240)
    operations: list[dict[str, Any]] = Field(default_factory=list, max_length=500)


class JobRequest(DTO):
    requestId: UUID
    branchId: UUID
    baseRevisionId: UUID
    kind: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    inputs: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class PublicationRequest(DTO):
    requestId: UUID
    sceneRevisionId: UUID
    evaluationIds: list[UUID] = Field(default_factory=list, max_length=100)
    reviewIds: list[UUID] = Field(default_factory=list, max_length=100)
    title: str = Field(min_length=1, max_length=240)


class AgentRequest(DTO):
    requestId: UUID
    conversationId: UUID
    branchId: UUID
    baseRevisionId: UUID
    message: str = Field(min_length=1, max_length=16000)
    entityId: str | None = None
    identityEntityIds: tuple[str, str] | None = None
    observationId: str | None = None
    imageId: UUID | None = None
    box: list[float] | None = None
    language: Literal["zh", "en"] = "en"
    policyId: UUID | None = None
    policyRevisionId: UUID | None = None

    @model_validator(mode="after")
    def valid_box(self):
        if self.identityEntityIds is not None and (not all(self.identityEntityIds) or self.identityEntityIds[0] == self.identityEntityIds[1]):
            raise ValueError("identityEntityIds must contain two different entity IDs")
        if (self.policyId is None) != (self.policyRevisionId is None):
            raise ValueError("policyId and policyRevisionId must be supplied together")
        if self.box is not None and (len(self.box) != 4 or min(self.box) < 0 or self.box[2] <= self.box[0] or self.box[3] <= self.box[1]):
            raise ValueError("box must have four ordered nonnegative pixel coordinates")
        return self


# Responses retain the stored JSON's optional fields and extension data. They do
# not normalize dates/numeric representations or add defaults to immutable docs.
Number = int | float
Vec3 = tuple[Number, Number, Number]
Vec4 = tuple[Number, Number, Number, Number]
Timestamp = Annotated[str, Field(json_schema_extra={"format": "date-time"})]
T = TypeVar("T")


class Items(DTO, Generic[T]):
    items: list[T]


class DocumentDTO(DTO):
    """Versioned documents allow preserved, producer-specific evidence fields."""
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


class IdentifiedDocument(DocumentDTO):
    id: str


class Transform(DTO):
    coordinateFrameId: str
    position: Vec3
    quaternion: Vec4
    scale: Vec3


class Bounds(DTO):
    min: Vec3
    max: Vec3


class ScaleEvidence(DocumentDTO):
    status: Literal["uncalibrated", "model_estimated", "operator_anchored"]
    nativeToMeters: Number | None = None
    sourceRefs: list[Any] | None = None


class GroundPlane(DocumentDTO):
    normal: Vec3 | None = None
    plane: Vec4 | None = None
    offset: Number | None = None
    sourceRefs: list[Any] | None = None


class CoordinateFrame(IdentifiedDocument):
    convention: Literal["opencv"]
    scale: ScaleEvidence
    ground: GroundPlane | None = None
    source: str | None = None
    sourceRefs: list[Any] | None = None


class Camera(IdentifiedDocument):
    imageId: str
    coordinateFrameId: str
    width: int
    height: int
    K: tuple[Vec3, Vec3, Vec3]
    cameraToWorld: tuple[Vec4, Vec4, Vec4, Vec4]
    sourceRefs: list[Any] | None = None


class Observation(IdentifiedDocument):
    captureId: str | None = None
    revision: int | None = None
    imageId: str
    originalPixelBox: Vec4
    maskAssetId: str | None = None
    pixelMapping: list[dict[str, Any]] | None = None
    geometrySupport: dict[str, Any] | None = None
    sourceRefs: list[Any] | None = None


class Representation(IdentifiedDocument):
    kind: Literal["observed_surface", "point_cloud", "generated_mesh", "primitive"]
    assetId: str | None = None
    coordinateFrameId: str
    transform: Transform
    primitive: dict[str, Any] | None = None
    placementState: Literal["unconfirmed", "confirmed"]
    placementReason: str | None = None
    placementSource: dict[str, Any] | None = None
    bounds: Bounds | None = None
    sourceRefs: list[Any] | None = None
    material: dict[str, Any] | None = None


class ObservationIdentityEvidence(DTO):
    kind: Literal["observation"]
    observationId: str = Field(min_length=1)
    observationRevision: int = Field(ge=1)


class AssetIdentityEvidence(DTO):
    kind: Literal["asset"]
    assetId: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MethodIdentityEvidence(DTO):
    kind: Literal["method"]
    name: str = Field(min_length=1, max_length=240)
    version: str = Field(min_length=1, max_length=240)
    configSha256: str = Field(pattern=r"^[0-9a-f]{64}$")


IdentityEvidenceRef = Annotated[ObservationIdentityEvidence | AssetIdentityEvidence | MethodIdentityEvidence, Field(discriminator="kind")]


class IdentityDecision(DTO):
    id: str = Field(min_length=1)
    decision: Literal["same", "different", "undecided"]
    source: Literal["geometry", "manual", "source_binding"]
    baseRevisionId: str = Field(min_length=1)
    entityIds: list[str] = Field(min_length=1)
    observationGroups: list[list[str]] = Field(min_length=2)
    survivorId: str | None
    evidenceRefs: list[IdentityEvidenceRef] = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=8000)
    supersedesDecisionId: str | None
    agentTurnId: str | None = None


class MeasurementEvidence(DTO):
    id: str = Field(min_length=1)
    measurementKey: str = Field(min_length=1)
    originalMeasurement: Any
    sourceRevisionId: str = Field(min_length=1)
    sourceEntityId: str = Field(min_length=1)
    observationRefs: list[str]
    representationId: str | None
    sourceEvidenceId: str | None = None
    sourceRefs: list[Any] | None = None
    coordinateFrameId: str | None = None


class SourceIdentityEvidence(DTO):
    assetId: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceIdentityPointer(SourceIdentityEvidence):
    jsonPointer: str = Field(pattern=r"^/")
    role: Literal['raw_source_reference', 'native_mask_provenance'] | None = None


class SourceObservationRevision(DTO):
    observationId: str = Field(min_length=1)
    revision: int = Field(ge=1)


class SourceObservationEquivalence(DTO):
    kind: Literal['canonical_sam_rle'] = 'canonical_sam_rle'
    observationRefs: tuple[SourceObservationRevision, SourceObservationRevision]
    imageId: str = Field(min_length=1)
    imageSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sourceRef: SourceIdentityPointer
    canonicalMaskSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonicalShape: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]]
    evidenceRefs: list[SourceIdentityPointer] = Field(min_length=1)


class GeometryBinding(DTO):
    geometrySolutionId: str = Field(min_length=1)
    cameraId: str = Field(min_length=1)


class Entity(IdentifiedDocument):
    label: str | None = None
    observationRefs: list[str] | None = None
    associationState: Literal["association_pending", "confirmed"]
    representations: list[Representation] | None = None
    currentModelTransform: Transform | None = None
    measurements: dict[str, Any] | None = None
    visible: bool | None = None
    material: dict[str, Any] | None = None
    groupId: str | None = None
    lineage: list[Any] | None = None
    activeModelRepresentationId: str | None = None
    measurementEvidence: list[MeasurementEvidence] | None = None
    measurementSelections: dict[str, str | None] | None = None


class SceneAsset(IdentifiedDocument):
    kind: str | None = None
    sha256: str | None = None
    sizeBytes: int | None = None
    mediaType: str | None = None
    metadata: dict[str, Any] | None = None


class Annotation(IdentifiedDocument):
    kind: str | None = None


class SceneDocument(DocumentDTO):
    # Frozen v1 responses retain their exact fields; v2 is created only by migration.
    schemaVersion: Literal[1, 2]
    captureId: str | None = None
    captureIds: list[str] | None = None
    identityDecisions: list[IdentityDecision] | None = None
    sourceIdentityEvidence: list[SourceIdentityEvidence] | None = None
    geometryBindings: dict[str, GeometryBinding | None] | None = None
    target: Literal["scene", "standalone_object"]
    coordinateFrames: list[CoordinateFrame]
    cameras: list[Camera]
    observations: list[Observation]
    entities: list[Entity]
    assets: list[SceneAsset]
    annotations: list[Annotation]


class Record(DTO):
    id: str
    createdAt: Timestamp


class OwnedRecord(Record):
    projectId: str


class RequestedRecord(OwnedRecord):
    requestId: str


class Project(Record):
    title: str
    requestId: str
    defaultBranchId: str
    forkSourceRevisionId: str | None


class Branch(OwnedRecord):
    kind: Literal["reconstruction", "planning"]
    title: str
    sourceRevisionId: str | None
    headRevisionId: str
    requestId: str | None


class Revision(OwnedRecord):
    branchId: str
    parentRevisionId: str | None
    sourceRevisionId: str | None
    document: SceneDocument
    documentSha256: str
    label: str | None


class ProjectDetail(DTO):
    project: Project
    branch: Branch
    revision: Revision
    branches: list[Branch]


class Operation(DocumentDTO):
    type: str


class EditBatch(RequestedRecord):
    branchId: str
    baseRevisionId: str
    revisionId: str
    operations: list[Operation]
    inverseOperations: list[Operation]
    undoOf: str | None
    redoOf: str | None
    agentTurnId: str | None


class Commit(DTO):
    revision: Revision
    editBatch: EditBatch
    headAdvanced: bool


class CaptureImage(IdentifiedDocument):
    assetId: str
    width: int
    height: int
    originalAssetId: str | None = None
    originalWidth: int | None = None
    originalHeight: int | None = None
    originalSha256: str | None = None
    pixelMapping: list[dict[str, Any]] | None = None
    sourceReused: bool | None = None


class Capture(RequestedRecord):
    branchId: str
    baseRevisionId: str
    revisionId: str
    target: Literal["scene", "standalone_object"]
    images: list[CaptureImage]
    task: dict[str, Any]


class Job(RequestedRecord):
    branchId: str
    baseRevisionId: str
    kind: str
    inputs: dict[str, Any]
    config: dict[str, Any]
    status: Literal["pending_dispatch", "queued", "running", "succeeded", "incomplete", "failed", "outcome_unknown", "cancelled"]
    cancelRequested: bool
    attempt: int
    dispatchedAt: Timestamp | None
    result: dict[str, Any] | None
    resultRevisionId: str | None
    headAdvanced: bool
    updatedAt: Timestamp
    heartbeatAt: Timestamp | None
    leaseExpiresAt: Timestamp | None


class CaptureCreated(DTO):
    capture: Capture
    revision: Revision
    job: Job


class AssetRecord(OwnedRecord):
    jobId: str | None
    storageKey: str
    sha256: str
    sizeBytes: int
    mediaType: str
    metadata: dict[str, Any]


class Asset(AssetRecord):
    url: str


class AgentResponse(DocumentDTO):
    kind: str | None = None
    message: str | None = None
    operations: list[Operation] | None = None
    baseRevisionId: str | None = None


class AgentTurn(RequestedRecord):
    conversationId: str
    branchId: str
    baseRevisionId: str
    request: AgentRequest
    response: AgentResponse | None
    status: Literal["pending", "running", "succeeded", "failed", "outcome_unknown"]
    updatedAt: Timestamp
    jobId: str | None
    sequence: int
    appliedEditBatchId: str | None = None
    appliedPolicyRevisionId: str | None = None


class SourceClause(DocumentDTO):
    locator: str
    text: str


class PolicySourceDocument(DocumentDTO):
    schemaVersion: Literal[1]
    title: str
    publisher: str
    language: str
    jurisdiction: str
    text: str
    clauses: list[SourceClause]
    kind: str | None = None
    url: str | None = None
    contentSha256: str | None = None


class PolicySource(OwnedRecord):
    contentSha256: str
    document: PolicySourceDocument


class Policy(RequestedRecord):
    title: str
    activeRevisionId: str | None
    draftRevisionId: str | None
    activationRequests: dict[str, str]


class PolicyRevisionDocument(DTO):
    schemaVersion: Literal[1]
    sourceId: str
    jdm: dict[str, Any]
    tests: list[Any]
    limitations: list[Any]
    sourceRefs: list[Any] | None = None


class PolicyRevision(RequestedRecord):
    policyId: str
    sourceId: str
    parentRevisionId: str | None
    document: PolicyRevisionDocument
    documentSha256: str
    agentTurnId: str | None


class PolicyCreated(DTO):
    policy: Policy
    revision: PolicyRevision


class PolicyDetail(DTO):
    policy: Policy
    revisions: list[PolicyRevision]
    sources: list[PolicySource]


class PolicyTemplate(DTO):
    title: str
    source: PolicySourceDocument
    jdm: dict[str, Any]
    tests: list[Any]
    limitations: list[Any]


class PolicyTestResult(DTO):
    name: Any
    passed: bool
    actual: dict[str, Any]


class PolicyActivation(DTO):
    policyRevisionId: str
    active: bool
    tests: list[PolicyTestResult] | None = None


class Finding(IdentifiedDocument):
    entityId: str | None
    applicability: Literal["applicable", "unknown", "not_applicable"]
    machineResult: Literal["PASS", "FAIL", "NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE"] | None
    facts: list[Any]
    missingEvidence: list[str]
    policyRevisionId: str
    policyTitle: str | None = None
    sourceId: str
    sceneRevisionId: str
    comparison: dict[str, Any] | None = None


class EvaluationDocument(DTO):
    schemaVersion: Literal[1]
    sceneRevisionId: str
    policyRevisionIds: list[str]
    context: Literal["observed", "planning"]
    evaluatorVersion: str
    factSetSha256: str
    findings: list[Finding]


class Evaluation(RequestedRecord):
    sceneRevisionId: str
    context: Literal["observed", "planning"]
    document: EvaluationDocument


class ReviewDocument(DTO):
    schemaVersion: Literal[1]
    decision: Literal["confirmed", "rejected", "needs_evidence"]
    reason: str
    displayName: str
    identityVerified: Literal[False]
    evidenceRefs: list[Any]
    actor: Literal["project_capability"]
    sceneRevisionId: str


class FindingRecord(RequestedRecord):
    evaluationId: str
    findingId: str


class Review(FindingRecord):
    document: ReviewDocument


class EvidenceIdentityBinding(DTO):
    sourceSceneRevisionId: str
    sourceFindingId: str
    sourceEntityId: str
    targetEntityId: str
    observationIds: list[str]
    identityDecisionIds: list[str]


class EvidenceRequestDocument(DTO):
    schemaVersion: Literal[1]
    status: Literal["open", "fulfilled"]
    action: str | None = None
    missingEvidence: list[str] | None = None
    entityId: str | None = None
    evidenceRequestId: str | None = None
    evidenceRefs: list[Any] | None = None
    sceneRevisionId: str | None = None
    evaluationId: str | None = None
    findingId: str | None = None
    actor: Literal["project_capability"] | None = None
    identityBinding: EvidenceIdentityBinding | None = None


class EvidenceRequest(FindingRecord):
    document: EvidenceRequestDocument


class SourceTextPage(DTO):
    page: int
    text: str


class SourceText(DTO):
    pages: list[SourceTextPage]
    status: Literal["text_extracted", "source_text_required"]


class AssetManifestEntry(DTO):
    assetId: str
    sha256: str
    sizeBytes: int
    mediaType: str


class PlaygroundDefinition(DTO):
    kind: Literal["observed", "model", "planning"]
    revisionId: str


class PublicationSnapshot(DTO):
    schemaVersion: Literal[1]
    revision: Revision
    evaluations: list[Evaluation]
    reviews: list[Review]
    jobs: list[Job] = Field(default_factory=list)
    editBatches: list[EditBatch] | None = None
    branchKind: Literal["reconstruction", "planning"] | None = None
    branchTitle: str | None = None
    reconstructionRevision: Revision | None = None
    reportSchemaVersion: int | None = None
    rendererVersion: str | None = None
    assetManifest: list[AssetManifestEntry] | None = None
    playgroundDefinitions: list[PlaygroundDefinition] | None = None


class PublicationSummary(OwnedRecord):
    sceneRevisionId: str
    title: str
    previewImageAssetId: str | None
    photoCount: int = Field(ge=0)
    objectCount: int = Field(ge=0, description="Entities excluding source context")
    spatialObjectCount: int = Field(ge=0, description="Non-context entities with any representation, including unconfirmed placement")
    modelObjectCount: int = Field(ge=0, description="Non-context entities with generated mesh or primitive representations")
    observedSurfaceObjectCount: int = Field(ge=0, description="Non-context entities with observed-surface representations")


class Publication(OwnedRecord):
    sceneRevisionId: str
    title: str
    requestId: str
    evaluationIds: list[str]
    reviewIds: list[str]
    snapshot: PublicationSnapshot


class Health(DTO):
    status: Literal["ok"]
    schemaVersion: Literal[1]


class ErrorDetail(DTO):
    code: str
    params: dict[str, Any]


class ErrorResponse(DTO):
    error: ErrorDetail


def empty_document() -> dict[str, Any]:
    return {"schemaVersion": 1, "captureId": None, "target": "scene", "coordinateFrames": [],
            "cameras": [], "observations": [], "entities": [], "assets": [], "annotations": []}


def validate_transform(value: dict[str, Any], frame_ids: set[str]) -> None:
    if not isinstance(value, dict) or value.get("coordinateFrameId") not in frame_ids:
        raise PlatformError("invalid_coordinate_frame")
    for key, size in (("position", 3), ("quaternion", 4), ("scale", 3)):
        vector = value.get(key)
        if not isinstance(vector, list) or len(vector) != size or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
            raise PlatformError("invalid_transform", field=key)
    if any(x <= 0 for x in value["scale"]):
        raise PlatformError("invalid_transform", field="scale")
    if not math.isclose(sum(x*x for x in value["quaternion"]), 1, rel_tol=1e-5, abs_tol=1e-5):
        raise PlatformError("invalid_transform", field="quaternion")


def validate_document(document: dict[str, Any]) -> dict[str, Any]:
    try:
        canonical(document)
    except (ValueError, TypeError):
        raise PlatformError("invalid_document") from None
    if document.get("schemaVersion") not in (1, 2) or document.get("target") not in ("scene", "standalone_object"):
        raise PlatformError("invalid_document")
    indexes = {}
    for key in ("coordinateFrames", "cameras", "observations", "entities", "assets", "annotations"):
        items = document.get(key)
        if not isinstance(items, list) or any(not isinstance(x, dict) or not isinstance(x.get("id"), str) for x in items):
            raise PlatformError("invalid_document", field=key)
        ids = {x["id"] for x in items}
        if len(ids) != len(items):
            raise PlatformError("duplicate_identity", field=key)
        indexes[key] = ids
    for frame in document["coordinateFrames"]:
        scale = frame.get("scale", {})
        if frame.get("convention") != "opencv" or scale.get("status") not in ("uncalibrated", "model_estimated", "operator_anchored"):
            raise PlatformError("invalid_coordinate_frame")
        factor = scale.get("nativeToMeters")
        if scale["status"] == "uncalibrated" and factor is not None:
            raise PlatformError("uncalibrated_scale")
        if scale["status"] != "uncalibrated" and (type(factor) not in (int, float) or factor <= 0 or not math.isfinite(factor)):
            raise PlatformError("invalid_scale")
    for camera in document["cameras"]:
        if camera.get("coordinateFrameId") not in indexes["coordinateFrames"]:
            raise PlatformError("invalid_coordinate_frame")
        if any(type(camera.get(key)) is not int or camera[key] <= 0 for key in ("width", "height")) or camera.get("imageId") not in indexes["assets"]:
            raise PlatformError("invalid_camera")
        from .spatial import affine, camera_intrinsics
        try:
            affine(camera.get("cameraToWorld"), rigid=True)
            camera_intrinsics(camera.get("K"))
        except (ValueError, TypeError):
            raise PlatformError("invalid_camera") from None
    for observation in document["observations"]:
        box = observation.get("originalPixelBox")
        if not isinstance(box, list) or len(box) != 4 or any(type(x) not in (int, float) for x in box) or box[2] <= box[0] or box[3] <= box[1] or min(box) < 0:
            raise PlatformError("invalid_pixel_box")
        if observation.get("imageId") not in indexes["assets"]:
            raise PlatformError("observation_image_not_found")
        if observation.get("maskAssetId") is not None and observation["maskAssetId"] not in indexes["assets"]:
            raise PlatformError("observation_mask_not_found")
    representation_ids = set()
    for entity in document["entities"]:
        if entity.get("associationState") not in ("association_pending", "confirmed") or not set(entity.get("observationRefs", [])) <= indexes["observations"]:
            raise PlatformError("invalid_entity")
        if entity.get("currentModelTransform") is not None:
            validate_transform(entity["currentModelTransform"], indexes["coordinateFrames"])
        for rep in entity.get("representations", []):
            if not isinstance(rep.get("id"), str) or rep["id"] in representation_ids:
                raise PlatformError("duplicate_representation_identity")
            representation_ids.add(rep["id"])
            if rep.get("kind") not in ("observed_surface", "point_cloud", "generated_mesh", "primitive") or rep.get("placementState") not in ("unconfirmed", "confirmed"):
                raise PlatformError("invalid_representation")
            validate_transform(rep.get("transform"), indexes["coordinateFrames"])
            if rep.get("coordinateFrameId") != rep["transform"]["coordinateFrameId"]:
                raise PlatformError("representation_coordinate_frame_mismatch")
            if rep.get("assetId") is not None and rep["assetId"] not in indexes["assets"]:
                raise PlatformError("representation_asset_not_found")
            if rep["kind"] == "primitive":
                from .spatial import primitive_mesh
                if not isinstance(rep.get("primitive"), dict):
                    raise PlatformError("invalid_primitive")
                primitive_mesh(rep["primitive"])
    if document["schemaVersion"] == 2:
        from .identity import validate_identity_document
        validate_identity_document(document)
    return document
