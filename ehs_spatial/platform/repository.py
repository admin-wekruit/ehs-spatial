"""Shared transaction errors and deterministic edit semantics."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from .contracts import PlatformError, validate_document, validate_transform


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


def apply_operations(source: dict, operations: list[dict]) -> tuple[dict, list[dict]]:
    document = deepcopy(source)
    frames = {x["id"] for x in document["coordinateFrames"]}

    def entity(identity: str) -> dict:
        found = next((x for x in document["entities"] if x["id"] == identity), None)
        if found is None:
            raise PlatformError("entity_not_found", 422, entityId=identity)
        return found

    for operation in operations:
        kind = operation.get("type")
        if kind in ("setTransform", "setLabel", "setVisibility", "setMaterial", "setPrimitive"):
            item = entity(operation.get("entityId"))
            if kind == "setTransform":
                value = operation.get("transform", {k: operation[k] for k in ("coordinateFrameId", "position", "quaternion", "scale") if k in operation})
                validate_transform(value, frames)
                item["currentModelTransform"] = deepcopy(value)
                for representation in item.get("representations", []):
                    if representation.get("kind") in ("generated_mesh", "primitive"):
                        representation["transform"] = deepcopy(value)
                        representation["coordinateFrameId"] = value["coordinateFrameId"]
                        representation["placementState"] = "confirmed"
                        representation["placementSource"] = {"type": "manual_assertion", "operation": "setTransform", "schemaVersion": 1}
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
                item["material"] = deepcopy(operation["material"])
            else:
                primitive = operation.get("primitive")
                if not isinstance(primitive, dict) or primitive.get("kind", primitive.get("type")) not in ("box", "cylinder"):
                    raise PlatformError("invalid_primitive")
                from .spatial import primitive_mesh
                primitive_mesh(primitive)
                transform = operation.get("transform") or primitive.get("transform") or item.get("currentModelTransform")
                if transform is None:
                    raise PlatformError("primitive_placement_required", 422)
                validate_transform(transform, frames)
                representation_id = operation.get("representationId")
                rep = next((r for r in item.get("representations", []) if r.get("kind") == "primitive" and (representation_id is None or r["id"] == representation_id)), None)
                if representation_id is not None and rep is None:
                    raise PlatformError("primitive_representation_not_found", 422)
                if rep is None:
                    rep = {"id": str(uuid5(NAMESPACE_URL, "primitive:" + item["id"])), "kind": "primitive", "assetId": None,
                           "sourceRefs": [{"observationId": identity} for identity in item.get("observationRefs", [])]}
                    item.setdefault("representations", []).append(rep)
                frame = next(f for f in document["coordinateFrames"] if f["id"] == transform["coordinateFrameId"])
                manual = frame.get("source") == "manual_assertion"
                rep.update(primitive=deepcopy(primitive), transform=deepcopy(transform), coordinateFrameId=transform["coordinateFrameId"], placementState="confirmed" if manual else "unconfirmed", placementReason="manual_assertion" if manual else "requires_alignment_confirmation")
                if manual:
                    rep["sourceRefs"] = [{"type": "manual_assertion", "coordinateFrameId": frame["id"]}]
                item["currentModelTransform"] = deepcopy(transform)
        elif kind == "addEntity":
            value = operation.get("entity")
            if not isinstance(value, dict) or any(x["id"] == value.get("id") for x in document["entities"]):
                raise PlatformError("invalid_entity")
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
        elif kind == "mergeEntities":
            identities = operation.get("entityIds", [])
            survivor = operation.get("survivorId")
            if len(identities) < 2 or len(set(identities)) != len(identities) or survivor not in identities:
                raise PlatformError("invalid_merge")
            items = [entity(identity) for identity in identities]
            kept = entity(survivor)
            for item in items:
                if item is kept:
                    continue
                kept["observationRefs"] = list(dict.fromkeys(kept.get("observationRefs", []) + item.get("observationRefs", [])))
                kept.setdefault("representations", []).extend(deepcopy(item.get("representations", [])))
                kept.setdefault("lineage", []).append({"operation": "merge", "entityId": item["id"]})
                document["entities"].remove(item)
            kept["associationState"] = "confirmed"
        elif kind == "splitEntity":
            item = entity(operation.get("entityId"))
            groups = operation.get("groups", [])
            if len(groups) < 2 or any(not isinstance(group, dict) or not group.get("id") for group in groups):
                raise PlatformError("invalid_split")
            refs = [ref for group in groups for ref in group.get("observationRefs", [])]
            if len(refs) != len(set(refs)) or set(refs) != set(item.get("observationRefs", [])):
                raise PlatformError("invalid_split")
            document["entities"].remove(item)
            for group in groups:
                # Representations cannot be assigned to a split observation without evidence.
                document["entities"].append({"id": group["id"], "label": group.get("label", item["label"]),
                    "observationRefs": group["observationRefs"], "associationState": "confirmed",
                    "representations": [], "currentModelTransform": None, "measurements": {}, "groupId": item.get("groupId"),
                    "lineage": [{"operation": "split", "entityId": item["id"]}]})
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
