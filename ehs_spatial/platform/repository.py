"""Shared transaction errors and deterministic edit semantics."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from .contracts import PlatformError, digest, validate_document, validate_transform


class Repository(Protocol):
    def migrate(self) -> None: ...
    def authorize(self, project_id: str, capability: str) -> dict: ...
    def create_project(self, capability: str, body: dict, *, source_revision_id: str | None = None) -> dict: ...
    def fork_project(self, source_project_id: str, capability: str, body: dict) -> dict: ...
    def get_project(self, project_id: str) -> dict: ...
    def list_projects(self) -> dict: ...
    def get_revision(self, revision_id: str) -> dict: ...
    def list_project_records(self, project_id: str, kind: str) -> dict: ...
    def create_branch(self, project_id: str, capability: str, body: dict) -> dict: ...
    def commit_edits(self, project_id: str, capability: str, body: dict) -> dict: ...
    def register_asset(self, project_id: str, metadata: dict, job_id: str | None = None) -> dict: ...
    def get_asset(self, asset_id: str) -> dict: ...
    def create_capture(self, project_id: str, capability: str, body: dict, images: list[dict]) -> dict: ...
    def create_job(self, project_id: str, capability: str, body: dict) -> dict: ...
    def get_job(self, job_id: str) -> dict: ...
    def pending_jobs(self, limit: int = 100) -> list[dict]: ...
    def mark_dispatched(self, job_id: str, executor_ref: str) -> dict: ...
    def claim_job(self, job_id: str, *, lease_seconds: int = 300) -> dict: ...
    def heartbeat_job(self, job_id: str, attempt_token: str, *, lease_seconds: int = 300) -> bool: ...
    def recover_expired_jobs(self) -> list[str]: ...
    def cancel_job(self, job_id: str, capability: str) -> dict: ...
    def finish_job(self, job_id: str, attempt_token: str, status: str, document: dict | None = None, result: dict | None = None, *, imported_capture: dict | None = None) -> dict: ...
    def reserve_model_call(self, job_id: str, attempt_token: str, provider: str, model: str, request_key: str, estimated_cost: Any, *, code_sha256: str | None = None, model_sha256: str | None = None, adapter_sha256: str | None = None, input_sha256: str | None = None, paid: bool = True) -> dict: ...
    def complete_model_call(self, call_id: str, status: str, actual_cost: Any = None, response: dict | None = None) -> dict: ...
    def create_publication(self, project_id: str, capability: str, body: dict, *, evaluations: list[dict] | None = None, reviews: list[dict] | None = None) -> dict: ...
    def list_publications(self) -> dict: ...
    def get_publication(self, publication_id: str) -> dict: ...
    def create_agent_turn(self, project_id: str, capability: str, body: dict) -> dict: ...
    def claim_agent_turn(self, turn_id: str) -> dict: ...
    def list_agent_turns(self, project_id: str, *, conversation_id: str | None = None, after_sequence: int = 0) -> dict: ...
    def finish_agent_turn(self, turn_id: str, status: str, response: dict) -> dict: ...


class PolicyRepository(Protocol):
    def authorize(self, project_id: str, capability: str) -> dict: ...
    def list_policies(self) -> dict: ...
    def get_policy(self, policy_id: str) -> dict: ...
    def create_policy(self, project_id: str, capability: str, body: dict) -> dict: ...
    def save_revision(self, project_id: str, policy_id: str, capability: str, body: dict) -> dict: ...
    def activate(self, project_id: str, policy_id: str, capability: str, body: dict) -> dict: ...
    def evaluate(self, project_id: str, capability: str, body: dict) -> dict: ...
    def review(self, project_id: str, finding_id: str, capability: str, body: dict, *, evidence_request: bool = False) -> dict: ...
    def publication_evidence(self, project_id: str, scene_revision_id: str, evaluation_ids: list[str], review_ids: list[str]) -> dict: ...
    def fulfill_evidence_request(self, project_id: str, evidence_request_id: str, capability: str, body: dict) -> dict: ...
    def list_evaluations(self, project_id: str) -> dict: ...
    def list_reviews(self, project_id: str) -> dict: ...
    def list_evidence_requests(self, project_id: str) -> dict: ...


def apply_operations(source: dict, operations: list[dict], *, base_revision_id: str | None = None) -> tuple[dict, list[dict]]:
    document = deepcopy(source)
    frames = {x["id"] for x in document["coordinateFrames"]}

    def entity(identity: str) -> dict:
        if document["schemaVersion"] == 2:
            from .identity import resolve_entity_id
            identity = resolve_entity_id(document, identity)
        found = next((x for x in document["entities"] if x["id"] == identity), None)
        if found is None:
            raise PlatformError("entity_not_found", 422, entityId=identity)
        return found

    for operation in operations:
        kind = operation.get("type")
        if kind == "migrateScene":
            from .identity import migrate_document
            if operation.get("schemaVersion", 2) != 2:
                raise PlatformError("unsupported_scene_migration", 422)
            document = migrate_document(document, base_revision_id=base_revision_id, active_model_selections=operation.get("activeModelSelections"))
        elif kind in ("recordIdentityDecision", "mergeEntities", "splitEntity", "setActiveModelRepresentation", "selectMeasurementEvidence"):
            from .identity import apply_identity_operation
            apply_identity_operation(document, operation, base_revision_id=base_revision_id)
        elif kind == 'confirmPlacement':
            item = entity(operation.get('entityId'))
            active_id = item.get('activeModelRepresentationId')
            if not active_id or operation.get('representationId') != active_id:
                raise PlatformError('active_model_selection_required', 422)
            rep = next((r for r in item['representations'] if r['id'] == active_id), None)
            if rep is None or rep['kind'] not in ('generated_mesh', 'primitive') or rep.get('sourceValidity') == 'stale':
                raise PlatformError('active_model_required', 422)
            if rep['placementState'] != 'confirmed' and rep.get('placementReason') not in ('imported_proposal', 'requires_alignment_confirmation'):
                raise PlatformError('model_placement_required', 422)
            pose = item.get('currentModelTransform') or rep['transform']
            validate_transform(pose, frames)
            rep.update(placementState='confirmed', placementSource={
                'type': 'manual_assertion', 'operation': 'confirmPlacement',
                'representationId': active_id, 'transformSha256': digest(pose)})
        elif kind in ("setTransform", "setLabel", "setVisibility", "setMaterial", "setPrimitive"):
            item = entity(operation.get("entityId"))
            if kind == "setTransform":
                value = operation.get("transform", {k: operation[k] for k in ("coordinateFrameId", "position", "quaternion", "scale") if k in operation})
                validate_transform(value, frames)
                if document["schemaVersion"] == 2 and item["activeModelRepresentationId"] is None:
                    raise PlatformError("active_model_required", 422)
                previous_transform = item.get('currentModelTransform')
                item["currentModelTransform"] = deepcopy(value)
                for representation in item.get("representations", []):
                    if representation.get("kind") in ("generated_mesh", "primitive") and (document["schemaVersion"] == 1 or representation["id"] == item["activeModelRepresentationId"]):
                        changed = value != (previous_transform or representation.get('transform'))
                        representation["transform"] = deepcopy(value)
                        representation["coordinateFrameId"] = value["coordinateFrameId"]
                        if changed:
                            representation.update(placementState='unconfirmed', placementReason='requires_alignment_confirmation')
                            representation.pop('placementSource', None)
            elif kind == "setLabel":
                value = operation.get("label")
                if not isinstance(value, str) or not value.strip() or len(value) > 500:
                    raise PlatformError("invalid_label")
                item["label"] = value
            elif kind == "setVisibility":
                if type(operation.get("visible")) is not bool:
                    raise PlatformError("invalid_visibility")
                item["visible"] = operation["visible"]
            elif kind == "setMaterial":
                if not isinstance(operation.get("material"), dict):
                    raise PlatformError("invalid_material")
                if document["schemaVersion"] == 2:
                    active = next((r for r in item["representations"] if r["id"] == item["activeModelRepresentationId"]), None)
                    if active is None:
                        raise PlatformError("active_model_required", 422)
                    active["material"] = deepcopy(operation["material"])
                else:
                    item["material"] = deepcopy(operation["material"])
            else:
                primitive = operation.get("primitive")
                if not isinstance(primitive, dict) or primitive.get("kind", primitive.get("type")) not in ("box", "cylinder"):
                    raise PlatformError("invalid_primitive")
                from .spatial import primitive_mesh
                mesh = primitive_mesh(primitive)
                representation_id = operation.get("representationId")
                if document["schemaVersion"] == 2:
                    active_id = item["activeModelRepresentationId"]
                    if representation_id is not None and representation_id != active_id:
                        raise PlatformError("active_model_selection_required", 422)
                    rep = next((r for r in item.get("representations", []) if r.get("kind") == "primitive" and r["id"] == active_id), None)
                    if rep is None and any(r.get("kind") == "primitive" for r in item.get("representations", [])):
                        raise PlatformError("active_model_selection_required", 422)
                else:
                    rep = next((r for r in item.get("representations", []) if r.get("kind") == "primitive" and (representation_id is None or r["id"] == representation_id)), None)
                if representation_id is not None and rep is None:
                    raise PlatformError("primitive_representation_not_found", 422)
                transform = operation.get("transform") or primitive.get("transform") or item.get("currentModelTransform") or (rep or {}).get("transform")
                if transform is None:
                    raise PlatformError("primitive_placement_required", 422)
                validate_transform(transform, frames)
                if rep is None:
                    rep = {"id": str(uuid5(NAMESPACE_URL, "primitive:" + item["id"])), "kind": "primitive", "assetId": None,
                           "sourceRefs": [{"observationId": identity} for identity in item.get("observationRefs", [])]}
                    item.setdefault("representations", []).append(rep)
                changed = primitive != rep.get("primitive") or transform != (item.get("currentModelTransform") or rep.get("transform"))
                rep.update(primitive=deepcopy(primitive), transform=deepcopy(transform), coordinateFrameId=transform["coordinateFrameId"],
                           bounds={"min": mesh.vertices.min(axis=0).tolist(), "max": mesh.vertices.max(axis=0).tolist()})
                if changed:
                    rep.update(placementState="unconfirmed", placementReason="requires_alignment_confirmation")
                    rep.pop("placementSource", None)
                    rep.pop("planProjection", None)
                item["currentModelTransform"] = deepcopy(transform)
                if document["schemaVersion"] == 2:
                    item["activeModelRepresentationId"] = rep["id"]
        elif kind == "addEntity":
            value = operation.get("entity")
            if not isinstance(value, dict) or any(x["id"] == value.get("id") for x in document["entities"]):
                raise PlatformError("invalid_entity")
            if document["schemaVersion"] == 2 and any(value.get("id") in decision["entityIds"] for decision in document["identityDecisions"]):
                raise PlatformError("identity_reused_entity_id", 422)
            document["entities"].append(deepcopy(value))
        elif kind == "removeEntity":
            item = entity(operation.get("entityId"))
            document["entities"].remove(item)
        elif kind == "addObservation":
            item = entity(operation.get("entityId"))
            observation = operation.get("observation")
            if not isinstance(observation, dict) or any(x["id"] == observation.get("id") for x in document["observations"]):
                raise PlatformError("invalid_observation")
            document["observations"].append(deepcopy(observation))
            item.setdefault("observationRefs", []).append(observation["id"])
        elif kind == "addCoordinateFrame":
            frame = operation.get("frame")
            if not isinstance(frame, dict) or not isinstance(frame.get("id"), str) or frame["id"] in frames or frame.get("source") != "manual_assertion" or frame.get("ground") is not None or frame.get("scale", {}).get("status") != "uncalibrated":
                raise PlatformError("invalid_manual_coordinate_frame", 422)
            document["coordinateFrames"].append(deepcopy(frame))
            frames.add(frame["id"])
        elif kind == "setCalibration":
            frame = next((x for x in document["coordinateFrames"] if x["id"] == operation.get("coordinateFrameId")), None)
            if frame is None:
                raise PlatformError("invalid_coordinate_frame")
            scale = deepcopy(operation.get("scale", {}))
            if scale.get("status") == "operator_anchored" and not scale.get("sourceRefs"):
                raise PlatformError("calibration_evidence_required")
            frame["scale"] = scale
        elif kind == "addAnnotation":
            value = operation.get("annotation")
            if not isinstance(value, dict):
                raise PlatformError("invalid_annotation")
            document["annotations"].append(deepcopy(value))
        elif kind == "removeAnnotation":
            value = next((x for x in document["annotations"] if x["id"] == operation.get("annotationId")), None)
            if value is None:
                raise PlatformError("annotation_not_found", 422)
            document["annotations"].remove(value)
        else:
            raise PlatformError("unknown_operation", 422, type=kind)
    validate_document(document)
    # ponytail: a full immutable snapshot is the exact inverse; branch CAS bounds storage
    # and prevents overwriting intervening edits. Compact inverses can replace this later.
    return document, [{"type": "restoreDocument", "document": deepcopy(source)}]
