"""Policy business transactions on MongoDB; same semantics as PostgresPolicyRepository."""
from uuid import uuid4

from argus.platform.contracts import PlatformError, digest
from argus.platform.mongo import _s, _wire
from argus.platform.policy_engine import APPLICABILITY, ENGINE_VERSION, _evidence_refs, _request, _require, evaluate_document, test_jdm, validate_jdm, validate_source
from argus.platform.policy_repository import _identity_evidence_binding


class MongoPolicyRepository:
    def __init__(self, repository):
        self.repo = repository

    def authorize(self, project_id, capability):
        return self.repo.authorize(project_id, capability)

    def _replay(self, table, project_id, body):
        _request(body)
        assert table in {"policies", "policy_revisions", "policy_evaluations", "policy_reviews", "policy_evidence_requests"}
        return self.repo._idempotent(table, project_id, body, scope="policy:")

    def list_policies(self):
        return {"items": _wire(list(self.repo._c("policies").find().sort([("created_at", -1), ("_id", 1)]).limit(200)))}

    def get_policy(self, policy_id):
        policy = self.repo._one("policies", {"_id": _s(policy_id)})
        revisions = list(self.repo._c("policy_revisions").find({"policy_id": _s(policy_id)}).sort([("created_at", 1), ("_id", 1)]))
        sources = list(self.repo._c("policy_sources").find({"_id": {"$in": sorted({r["source_id"] for r in revisions})}}).sort("_id", 1))
        return _wire({"policy": policy, "revisions": revisions, "sources": sources})

    def create_policy(self, project_id, capability, body):
        _request(body, ("source", "jdm", "title"))
        _require(isinstance(body["title"], str) and 0 < len(body["title"].strip()) <= 240, "policy_title_invalid")
        source = validate_source(body["source"])
        validate_jdm(body["jdm"])
        self.repo._auth(project_id, capability)
        with self._replay("policies", project_id, body) as prior:
            if prior:
                revision = self.repo._one("policy_revisions", {"_id": prior["draft_revision_id"]})
                return _wire({"policy": prior, "revision": revision})
            sid, pid, rid = str(uuid4()), str(uuid4()), str(uuid4())
            self.repo._insert("policy_sources", {"_id": sid, "project_id": _s(project_id), "content_sha256": digest(source), "document": source, "created_at": self.repo._now()})
            revision = self._insert(project_id, pid, sid, body, rid=rid)
            policy = self.repo._insert("policies", {"_id": pid, "project_id": _s(project_id), "title": body["title"], "request_id": str(body["requestId"]), "request_sha256": digest(body),
                                                    "active_revision_id": None, "draft_revision_id": rid, "activation_requests": {}, "created_at": self.repo._now()})
            return _wire({"policy": policy, "revision": revision})

    def _insert(self, project_id, policy_id, source_id, body, parent=None, rid=None):
        validate_jdm(body["jdm"])
        doc = {"schemaVersion": 1, "jdm": body["jdm"], "tests": body.get("tests", []), "limitations": body.get("limitations", []), "sourceRefs": body.get("sourceRefs", []), "sourceId": str(source_id)}
        return self.repo._insert("policy_revisions", {"_id": _s(rid) or str(uuid4()), "project_id": _s(project_id), "policy_id": _s(policy_id), "source_id": _s(source_id), "parent_revision_id": _s(parent),
                                                      "request_id": str(body["requestId"]), "request_sha256": digest(body), "document": doc, "document_sha256": digest(doc),
                                                      "created_at": self.repo._now(), "agent_turn_id": _s(body.get("agentTurnId"))})

    def save_revision(self, project_id, policy_id, capability, body):
        _request(body, ("basePolicyRevisionId", "jdm"))
        self.repo._auth(project_id, capability)
        with self._replay("policy_revisions", project_id, body) as prior:
            if prior:
                return _wire(prior)
            policy = self.repo._one("policies", {"project_id": _s(project_id), "_id": _s(policy_id)})
            if policy["draft_revision_id"] != body["basePolicyRevisionId"]:
                raise PlatformError("policy_revision_conflict", 409, currentRevisionId=policy["draft_revision_id"])
            base = self.repo._one("policy_revisions", {"_id": _s(body["basePolicyRevisionId"]), "policy_id": _s(policy_id)})
            if body.get("agentTurnId"):
                turn = self.repo._one("agent_turns", {"project_id": _s(project_id), "_id": _s(body["agentTurnId"])}, code="agent_turn_not_found")
                proposal = turn["response"] or {}
                _require(turn["status"] == "succeeded" and proposal.get("proposalType") == "policy" and str(proposal.get("policyId")) == str(policy_id) and proposal.get("basePolicyRevisionId") == body["basePolicyRevisionId"] and all(proposal.get(key, [] if key != "jdm" else None) == body.get(key, [] if key != "jdm" else None) for key in ("jdm", "tests", "limitations", "sourceRefs")), "agent_policy_proposal_mismatch")
            revision = self._insert(project_id, policy_id, base["source_id"], body, parent=base["_id"])
            moved = self.repo._c("policies").update_one({"_id": policy["_id"], "draft_revision_id": base["_id"]}, {"$set": {"draft_revision_id": revision["_id"]}})
            if moved.matched_count == 0:  # a concurrent draft won the compare-and-set
                self.repo._c("policy_revisions").delete_one({"_id": revision["_id"]})
                current = self.repo._one("policies", {"_id": policy["_id"]})["draft_revision_id"]
                raise PlatformError("policy_revision_conflict", 409, currentRevisionId=current)
            return _wire(revision)

    def activate(self, project_id, policy_id, capability, body):
        _request(body, ("policyRevisionId",))
        self.repo._auth(project_id, capability)
        with self.repo._lock("policy:" + _s(policy_id)):
            policy = self.repo._one("policies", {"project_id": _s(project_id), "_id": _s(policy_id)})
            history = policy["activation_requests"]
            if body["requestId"] in history:
                if history[body["requestId"]] != digest(body):
                    raise PlatformError("idempotency_mismatch", 409)
                return {"policyRevisionId": body["policyRevisionId"], "active": policy["active_revision_id"] == body["policyRevisionId"]}
            current = policy["active_revision_id"]
            if current != body.get("expectedActiveRevisionId"):
                raise PlatformError("policy_activation_conflict", 409, currentRevisionId=current)
            revision = self.repo._one("policy_revisions", {"project_id": _s(project_id), "policy_id": _s(policy_id), "_id": _s(body["policyRevisionId"])})
            doc = revision["document"]
            source = self.repo._one("policy_sources", {"_id": revision["source_id"]})["document"]
            validate_source(source)
            tests = test_jdm(doc["jdm"], doc["tests"])
            _require(tests and all(t["passed"] for t in tests), "policy_tests_must_pass")
            _require({t.get("input", {}).get("applicability") for t in doc["tests"]} >= APPLICABILITY, "policy_applicability_tests_required")
            history[body["requestId"]] = digest(body)
            self.repo._c("policies").update_one({"_id": policy["_id"]}, {"$set": {"active_revision_id": revision["_id"], "activation_requests": history}})
            return {"policyRevisionId": revision["_id"], "active": True, "tests": tests}

    def evaluate(self, project_id, capability, body):
        _request(body, ("sceneRevisionId", "policyRevisionIds", "context"))
        _require(isinstance(body["policyRevisionIds"], list) and 1 <= len(body["policyRevisionIds"]) <= 100, "evaluation_input_invalid")
        _require(isinstance(body.get("context"), str) and body["context"] in {"observed", "planning"} and body.get("policyRevisionIds"), "evaluation_input_invalid")
        self.repo._auth(project_id, capability)
        with self._replay("policy_evaluations", project_id, body) as prior:
            if prior:
                return _wire(prior)
            revision = self.repo._revision(project_id, body["sceneRevisionId"])
            branch = self.repo._one("scene_branches", {"_id": revision["branch_id"]})
            _require((branch["kind"] == "planning") == (body["context"] == "planning"), "evaluation_context_mismatch")
            findings, policy_refs = [], []
            for rid in body["policyRevisionIds"]:
                policy = self.repo._one("policy_revisions", {"project_id": _s(project_id), "_id": _s(rid)})
                parent = self.repo._one("policies", {"_id": policy["policy_id"]})
                _require(parent["active_revision_id"] == policy["_id"], "policy_revision_not_active")
                items = evaluate_document(revision["document"], policy["policy_id"], policy["document"], body["context"])
                for finding in items:
                    finding.update(policyRevisionId=rid, policyTitle=parent["title"], sourceId=policy["source_id"], sceneRevisionId=body["sceneRevisionId"])
                findings.extend(items)
                policy_refs.append(rid)
            doc = {"schemaVersion": 1, "sceneRevisionId": body["sceneRevisionId"], "policyRevisionIds": policy_refs, "context": body["context"], "evaluatorVersion": ENGINE_VERSION,
                   "factSetSha256": digest({"entities": revision["document"]["entities"], "annotations": revision["document"].get("annotations", [])}), "findings": findings}
            row = self.repo._insert("policy_evaluations", {"_id": str(uuid4()), "project_id": _s(project_id), "scene_revision_id": revision["_id"], "request_id": str(body["requestId"]),
                                                           "request_sha256": digest(body), "context": body["context"], "document": doc, "created_at": self.repo._now()})
            return _wire(row)

    def _finding(self, project_id, evaluation_id, finding_id):
        evaluation = self.repo._one("policy_evaluations", {"project_id": _s(project_id), "_id": _s(evaluation_id)})
        finding = next((f for f in evaluation["document"]["findings"] if f["id"] == str(finding_id)), None)
        _require(finding is not None, "evaluation_finding_mismatch")
        _require(evaluation["document"]["sceneRevisionId"] == evaluation["scene_revision_id"] and finding.get("sceneRevisionId") == evaluation["scene_revision_id"] and finding.get("policyRevisionId") in evaluation["document"]["policyRevisionIds"], "evaluation_finding_mismatch")
        return evaluation, finding

    def review(self, project_id, finding_id, capability, body, *, evidence_request=False):
        _request(body, ("evaluationId",))
        table = "policy_evidence_requests" if evidence_request else "policy_reviews"
        self.repo._auth(project_id, capability)
        with self._replay(table, project_id, body) as prior:
            if prior:
                return _wire(prior)
            evaluation, finding = self._finding(project_id, body["evaluationId"], finding_id)
            if evidence_request:
                _require(isinstance(body.get("action"), str) and bool(body["action"].strip()), "evidence_action_required")
                doc = {"schemaVersion": 1, "action": body["action"], "status": "open", "missingEvidence": finding["missingEvidence"], "entityId": finding["entityId"]}
            else:
                _require(isinstance(body.get("decision"), str) and body["decision"] in {"confirmed", "rejected", "needs_evidence"}, "review_decision_invalid")
                _require(isinstance(body.get("reason"), str) and body["reason"].strip() and isinstance(body.get("displayName"), str) and body["displayName"].strip(), "review_reason_required")
                refs = body.get("evidenceRefs", [])
                _require(isinstance(refs, list), "review_evidence_invalid")
                if refs:
                    scene = self.repo._revision(project_id, evaluation["scene_revision_id"])["document"]
                    _require(_evidence_refs(scene, refs), "review_evidence_invalid")
                doc = {"schemaVersion": 1, "decision": body["decision"], "reason": body["reason"], "displayName": body["displayName"], "identityVerified": False, "evidenceRefs": body.get("evidenceRefs", []), "actor": "project_capability", "sceneRevisionId": evaluation["scene_revision_id"]}
            return _wire(self.repo._insert(table, {"_id": str(uuid4()), "project_id": _s(project_id), "evaluation_id": evaluation["_id"], "finding_id": _s(finding_id), "request_id": str(body["requestId"]),
                                                   "request_sha256": digest(body), "document": doc, "created_at": self.repo._now()}))

    def publication_evidence(self, project_id, scene_revision_id, evaluation_ids, review_ids):
        evaluations, reviews = [], []
        for identity in evaluation_ids:
            row = self.repo._one("policy_evaluations", {"project_id": _s(project_id), "_id": _s(identity), "scene_revision_id": _s(scene_revision_id)}, code="publication_evaluation_mismatch")
            evaluations.append(_wire(row))
        for identity in review_ids:
            row = self.repo._one("policy_reviews", {"project_id": _s(project_id), "_id": _s(identity)})
            _require(row["evaluation_id"] in evaluation_ids, "publication_review_mismatch")
            self._finding(project_id, row["evaluation_id"], row["finding_id"])
            reviews.append(_wire(row))
        return {"evaluations": evaluations, "reviews": reviews}

    def fulfill_evidence_request(self, project_id, evidence_request_id, capability, body):
        _request(body, ("evaluationId", "findingId", "evidenceRefs"))
        body = {**body, "evidenceRequestId": str(evidence_request_id)}
        self.repo._auth(project_id, capability)
        requests = self.repo._c("policy_evidence_requests")
        with self._replay("policy_evidence_requests", project_id, body) as prior:
            if prior:
                return _wire(prior)
            with self.repo._lock("evidence:" + str(evidence_request_id)):  # one fulfilment per request (FOR UPDATE)
                original = self.repo._one("policy_evidence_requests", {"project_id": _s(project_id), "_id": str(evidence_request_id)}, code="evidence_request_not_found")
                _require(original["document"]["status"] == "open", "evidence_request_not_open")
                already = requests.find_one({"project_id": _s(project_id), "document.evidenceRequestId": str(evidence_request_id)}, {"_id": 1})
                if already:
                    raise PlatformError("evidence_request_already_fulfilled", 409, fulfillmentId=already["_id"])
                original_evaluation, original_finding = self._finding(project_id, original["evaluation_id"], original["finding_id"])
                evaluation, finding = self._finding(project_id, body["evaluationId"], body["findingId"])
                _require(evaluation["_id"] != original["evaluation_id"], "new_evaluation_required")
                _require(evaluation["context"] == original_evaluation["context"] and finding.get("policyRevisionId") == original_finding.get("policyRevisionId"), "evidence_finding_mismatch")
                revision = self.repo._revision(project_id, evaluation["scene_revision_id"])
                source = self.repo._revision(project_id, original_evaluation["scene_revision_id"])
                scene = revision["document"]
                _require(isinstance(body["evidenceRefs"], list) and bool(body["evidenceRefs"]) and _evidence_refs(scene, body["evidenceRefs"]), "review_evidence_invalid")
                ancestry = self.repo._ancestry(project_id, revision, stop_at=source["_id"])
                ancestry_ids = set(ancestry) if revision["branch_id"] == source["branch_id"] else set()
                binding = _identity_evidence_binding(source["document"], scene, original_finding, finding, body["evidenceRefs"], source_revision_id=source["_id"], ancestry_ids=ancestry_ids)
                doc = {"schemaVersion": 1, "status": "fulfilled", "evidenceRequestId": str(evidence_request_id), "evidenceRefs": body["evidenceRefs"], "sceneRevisionId": evaluation["scene_revision_id"], "evaluationId": evaluation["_id"], "findingId": body["findingId"], "actor": "project_capability"}
                if binding is not None:
                    doc["identityBinding"] = binding
                return _wire(self.repo._insert("policy_evidence_requests", {"_id": str(uuid4()), "project_id": _s(project_id), "evaluation_id": evaluation["_id"], "finding_id": _s(body["findingId"]),
                                                                            "request_id": str(body["requestId"]), "request_sha256": digest(body), "document": doc, "created_at": self.repo._now()}))

    def _listing(self, table, project_id, limit):
        self.repo._one("projects", {"_id": _s(project_id)}, code="project_not_found")
        return {"items": _wire(list(self.repo._c(table).find({"project_id": _s(project_id)}).sort([("created_at", -1), ("_id", 1)]).limit(limit)))}

    def list_evaluations(self, project_id):
        return self._listing("policy_evaluations", project_id, 200)

    def list_reviews(self, project_id):
        return self._listing("policy_reviews", project_id, 500)

    def list_evidence_requests(self, project_id):
        return self._listing("policy_evidence_requests", project_id, 500)
