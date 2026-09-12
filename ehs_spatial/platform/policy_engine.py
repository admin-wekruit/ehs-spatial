"""Versioned policy authoring and evidence review; never asks a VLM for a verdict.

ZEN receives applicability facts only. Numerical facts are evaluated in Python,
so editing a decision table cannot create a second geometric evaluator.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from uuid import UUID, uuid4

import zen

from .contracts import PlatformError, canonical, digest

ENGINE_VERSION = "policy-evidence-v1"
APPLICABILITY = {"applicable", "unknown", "not_applicable"}
STATUSES = {"PASS", "FAIL", "NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE"}


def _require(condition, code, **params):
    if not condition:
        raise PlatformError(code, 422, **params)


def _request(body, required=()):
    _require(isinstance(body, dict) and all(key in body for key in required), "policy_request_invalid", required=list(required))
    try:
        UUID(str(body["requestId"]))
    except (ValueError, KeyError, TypeError):
        raise PlatformError("request_id_required", 422)
    try:
        for key in ("basePolicyRevisionId", "policyRevisionId", "expectedActiveRevisionId", "sceneRevisionId", "evaluationId", "findingId", "agentTurnId"):
            if body.get(key) is not None:
                UUID(str(body[key]))
        if "policyRevisionIds" in body:
            _require(isinstance(body["policyRevisionIds"], list), "evaluation_input_invalid")
            for value in body["policyRevisionIds"]:
                UUID(str(value))
    except (ValueError, TypeError):
        raise PlatformError("policy_reference_invalid", 422) from None


def _cap(header):
    if not header or not header.startswith("Capability "):
        raise PlatformError("capability_required", 401)
    return header.removeprefix("Capability ")


def validate_jdm(jdm):
    try:
        _require(isinstance(jdm, dict) and len(canonical(jdm)) <= 256_000, "policy_jdm_invalid")
    except (ValueError, TypeError):
        raise PlatformError("policy_jdm_invalid", 422) from None
    nodes, edges = jdm.get("nodes", []), jdm.get("edges", [])
    _require(isinstance(nodes, list) and isinstance(edges, list) and all(isinstance(n, dict) for n in nodes) and all(isinstance(e, dict) for e in edges), "policy_graph_invalid")
    _require(2 <= len(nodes) <= 64 and len(edges) <= 128, "policy_graph_size")
    allowed = {"inputNode", "outputNode", "expressionNode", "decisionTableNode", "switchNode"}
    _require(all(isinstance(n.get("id"), str) and n["id"] for n in nodes), "policy_graph_identity")
    ids = {n["id"] for n in nodes}
    _require(len(ids) == len(nodes) and None not in ids, "policy_graph_identity")
    _require(sum(n.get("type") == "inputNode" for n in nodes) == 1 and sum(n.get("type") == "outputNode" for n in nodes) == 1, "policy_graph_endpoints")
    for node in nodes:
        _require(node.get("type") in allowed, "policy_node_not_allowed", nodeType=node.get("type"))
    successors = {identity: [] for identity in ids}
    for edge in edges:
        _require(edge.get("sourceId") in ids and edge.get("targetId") in ids, "policy_edge_invalid")
        successors[edge["sourceId"]].append(edge["targetId"])
    visiting, seen = set(), set()
    def visit(identity):
        _require(identity not in visiting, "policy_graph_cycle")
        if identity in seen:
            return
        visiting.add(identity)
        for nxt in successors[identity]:
            visit(nxt)
        visiting.remove(identity)
        seen.add(identity)
    for identity in ids:
        visit(identity)
    try:
        return zen.ZenEngine().create_decision(json.dumps(jdm))
    except Exception:
        raise PlatformError("policy_jdm_invalid", 422)


def execute_jdm(jdm, applicability="unknown", labels=()):
    _require(isinstance(applicability, str) and applicability in APPLICABILITY, "policy_applicability_invalid")
    _require(isinstance(labels, (list, tuple)) and all(isinstance(value, str) for value in labels), "policy_labels_invalid")
    try:
        result = validate_jdm(jdm).evaluate({"applicability": applicability, "labels": list(labels)})["result"]
    except PlatformError:
        raise
    except Exception:
        raise PlatformError("policy_jdm_execution_failed", 422)
    _require(isinstance(result, dict) and isinstance(result.get("applicability"), str) and result["applicability"] in APPLICABILITY, "policy_output_invalid")
    check = result.get("check")
    _require(isinstance(check, dict), "policy_check_required")
    _require(isinstance(check.get("kind"), str) and check["kind"] in {"manual", "geometry"}, "policy_check_kind")
    if check["kind"] == "manual":
        _require(isinstance(check.get("requirement"), str) and bool(check["requirement"]), "policy_requirement_required")
    else:
        from ..policy import PolicySpec
        try:
            PolicySpec(policy_id="validation", source_text="versioned JDM", **check["spec"])
        except Exception:
            raise PlatformError("policy_geometry_spec_invalid", 422)
    return result


def test_jdm(jdm, tests):
    _require(isinstance(tests, list) and len(tests) <= 100, "policy_tests_invalid")
    results = []
    for test in tests:
        _require(isinstance(test, dict) and isinstance(test.get("input", {}), dict) and isinstance(test.get("expected", {}), dict), "policy_tests_invalid")
        expected = test.get("expected", {})
        result = execute_jdm(jdm, test.get("input", {}).get("applicability", "unknown"), test.get("input", {}).get("labels", []))
        results.append({"name": test.get("name", ""), "passed": bool(expected) and all(result.get(k) == v for k, v in expected.items()), "actual": result})
    return results


def _jdm(check):
    return {"nodes": [
        {"id": "input", "type": "inputNode", "name": "Applicability evidence", "position": {"x": 0, "y": 0}},
        {"id": "check", "type": "expressionNode", "name": "Check specification", "position": {"x": 260, "y": 0}, "content": {"expressions": [
            {"id": "applicability", "key": "applicability", "value": 'applicability ?? "unknown"'},
            {"id": "check-spec", "key": "check", "value": json.dumps(check)}], "passThrough": False}},
        {"id": "output", "type": "outputNode", "name": "Python evidence evaluator", "position": {"x": 540, "y": 0}}],
        "edges": [{"id": "in-check", "sourceId": "input", "targetId": "check"}, {"id": "check-out", "sourceId": "check", "targetId": "output"}]}


def templates():
    # These short source excerpts define the covered atomic checks. They are not
    # an assertion that photographs cover the full obligations in each section.
    definitions = [
        ("Walking-working surfaces", "1910.22(a)(3)", "Walking-working surfaces are maintained free of hazards", {"kind": "manual", "requirement": "walking_surface_hazard_review"}),
        ("Exit route occupancy", "1910.37(a)(3)", "Exit routes must be free and unobstructed.", {"kind": "geometry", "spec": {"predicate": "not_inside", "subject_labels": ["material", "equipment"], "object_labels": ["confirmed exit route"], "threshold": 0, "unit": "m2"}}),
        ("Exit door opening", "1910.36(d)(1)", "Employees must be able to open an exit route door from the inside at all times without keys, tools, or special knowledge.", {"kind": "manual", "requirement": "exit_door_function_test"}),
        ("Electrical workspace storage", "1910.303(g)(1)(ii)", "Working space required by this standard may not be used for storage.", {"kind": "geometry", "spec": {"predicate": "not_inside", "subject_labels": ["stored material", "stored equipment"], "object_labels": ["confirmed electrical workspace"], "threshold": 0, "unit": "m2"}}),
        ("Machine hazard and guarding", "1910.212(a)(1)", "One or more methods of machine guarding shall be provided to protect the operator and other employees in the machine area", {"kind": "manual", "requirement": "machine_hazard_guard_correspondence"}),
    ]
    result = []
    for title, locator, excerpt, check in definitions:
        section = locator.split("(")[0]
        source = {"schemaVersion": 1, "kind": "regulation", "title": title, "publisher": "OSHA", "url": f"https://www.osha.gov/laws-regs/regulations/standardnumber/1910/{section}", "language": "en", "jurisdiction": "US", "text": excerpt, "clauses": [{"locator": locator, "text": excerpt}]}
        result.append({"title": title, "source": source, "jdm": _jdm(check), "tests": [{"name": v, "input": {"applicability": v}, "expected": {"applicability": v}} for v in sorted(APPLICABILITY)], "limitations": ["Requires explicit applicability and target evidence; covers only the named atomic check."]})
    return result


def validate_source(source):
    _require(isinstance(source, dict) and isinstance(source.get("text"), str) and 0 < len(source["text"].strip()) <= 2_000_000, "policy_source_text_required")
    _require(all(isinstance(source.get(k), str) and source[k].strip() for k in ["title", "publisher", "language", "jurisdiction"]), "policy_source_metadata_required")
    clauses = source.get("clauses")
    _require(isinstance(clauses, list) and bool(clauses) and all(isinstance(c, dict) and isinstance(c.get("locator"), str) and c["locator"] and isinstance(c.get("text"), str) and c["text"] and c["text"] in source["text"] for c in clauses), "policy_clause_location_required")
    return {**source, "schemaVersion": 1, "contentSha256": digest(source)}


def _evidence_refs(scene, refs):
    if not isinstance(refs, list) or not refs:
        return False
    identities = {item["id"] for collection in ("assets", "observations", "cameras", "annotations") for item in scene.get(collection, [])}
    for ref in refs:
        values = [ref] if isinstance(ref, str) else [ref[key] for key in ("assetId", "imageId", "observationId", "annotationId") if key in ref] if isinstance(ref, dict) else []
        if not values or any(not isinstance(value, str) or value not in identities for value in values):
            return False
    return True


def evaluate_document(scene, policy_id, policy_document, context):
    """Keep target membership before checking geometry; missing data stays visible."""
    from ..contracts import Entity3D, SceneMap
    from ..policy import PolicySpec, evaluate_policy
    annotations = scene.get("annotations", [])
    assertion = next((a for a in reversed(annotations) if a.get("kind") == "policy_applicability" and a.get("policyId") == policy_id), None)
    applicability = assertion.get("value", "unknown") if assertion and _evidence_refs(scene, assertion.get("sourceRefs")) else "unknown"
    output = execute_jdm(policy_document["jdm"], applicability, [e["label"] for e in scene["entities"]])
    application = output["applicability"] if applicability != "unknown" else "unknown"
    if application != "applicable":
        return [{"id": str(uuid4()), "entityId": None, "applicability": application, "machineResult": None, "facts": [], "missingEvidence": [] if application == "not_applicable" else ["applicability_confirmation"]}]
    check = output["check"]
    targets = [e for e in scene["entities"] if not check.get("subjectLabels") or e["label"] in check["subjectLabels"]] if check["kind"] == "manual" else [e for e in scene["entities"] if e["label"] in check["spec"]["subject_labels"]]
    if not targets:
        return [{"id": str(uuid4()), "entityId": None, "applicability": "applicable", "machineResult": "INSUFFICIENT_EVIDENCE", "facts": [], "missingEvidence": ["target_inventory_confirmation"]}]
    findings = []
    for target in targets:
        finding = {"id": str(uuid4()), "entityId": target["id"], "applicability": "applicable", "machineResult": "INSUFFICIENT_EVIDENCE", "facts": [], "missingEvidence": []}
        if check["kind"] == "manual":
            fact = next((a for a in reversed(annotations) if a.get("kind") == "manual_evidence" and a.get("entityId") == target["id"] and a.get("requirement") == check["requirement"] and _evidence_refs(scene, a.get("sourceRefs"))), None)
            if fact and isinstance(fact.get("passed"), bool):
                finding.update(machineResult="PASS" if fact["passed"] else "FAIL", facts=[fact])
            else:
                finding["missingEvidence"].append(check["requirement"])
        else:
            spec = PolicySpec(policy_id=policy_id, source_text="versioned JDM", **check["spec"])
            candidates = [target, *[e for e in scene["entities"] if e["label"] in spec.object_labels and e["id"] != target["id"]]]
            geometry, frames, errors = [], set(), []
            for e in candidates:
                facts = e.get("measurements", {})
                footprint = facts.get("footprint")
                allowed_sources = {"planned_geometry", "manual_assertion"} if context == "planning" else {"observed_measurement", "manual_assertion"}
                if not isinstance(footprint, dict) or footprint.get("source") not in allowed_sources or not _evidence_refs(scene, footprint.get("sourceRefs")) or footprint.get("unit") != "m" or footprint.get("uncertaintyM") is None:
                    finding["missingEvidence"].append("metric_footprint:" + e["id"])
                    continue
                if type(footprint["uncertaintyM"]) not in (int, float) or not math.isfinite(footprint["uncertaintyM"]) or footprint["uncertaintyM"] < 0:
                    raise PlatformError("measurement_uncertainty_invalid", 422)
                frame = next((f for f in scene["coordinateFrames"] if f["id"] == footprint.get("coordinateFrameId")), None)
                if not frame or frame.get("scale", {}).get("status") != "operator_anchored":
                    finding["missingEvidence"].append("metric_calibration:" + e["id"])
                    continue
                frames.add(footprint["coordinateFrameId"])
                errors.append(footprint["uncertaintyM"])
                refs = e.get("observationRefs", [])
                height = facts.get("height", {})
                if spec.predicate.value in {"max_height", "min_height"}:
                    if not isinstance(height, dict) or type(height.get("value")) not in (int, float) or not math.isfinite(height["value"]) or height["value"] < 0 or height.get("unit") != "m" or height.get("source") not in allowed_sources or not _evidence_refs(scene, height.get("sourceRefs")) or type(height.get("uncertaintyM")) not in (int, float) or not math.isfinite(height["uncertaintyM"]) or height["uncertaintyM"] < 0:
                        finding["missingEvidence"].append("metric_height:" + e["id"])
                        continue
                    errors[-1] = height["uncertaintyM"]
                try:
                    geometry.append(Entity3D(entity_id=e["id"], label=e["label"], observation_ids=refs, centroid_xyz=(0, 0, 0), footprint_xy=footprint.get("value", []), height_m=height.get("value", 0) if isinstance(height, dict) else 0, evidence_frame_ids=refs or ["manual-evidence"]))
                except (ValueError, TypeError):
                    finding["missingEvidence"].append("invalid_geometry:" + e["id"])
                finding["facts"].append(footprint)
            if len(frames) > 1:
                finding["missingEvidence"].append("registered_coordinate_frame")
            if not finding["missingEvidence"]:
                model = SceneMap(run_id="policy-scene", floor_plane=None, scale_source="operator_anchored", scale_factor=1, fence_polygon=[], entities=geometry, facts=[], warnings=[])
                result = evaluate_policy(spec, model, capture_frame_count=1, error_budget_m=(errors[0] + max(errors[1:], default=0)) if errors else 0)
                finding["machineResult"] = result.status.value
                finding["comparison"] = result.model_dump(mode="json")
        findings.append(finding)
    return findings
