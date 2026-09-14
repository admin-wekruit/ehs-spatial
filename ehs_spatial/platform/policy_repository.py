"""Policy business transactions; PostgreSQL details stay behind this boundary."""
from uuid import uuid4

from psycopg.types.json import Jsonb

from .contracts import PlatformError, digest
from .identity import resolve_entity_id
from .postgres import _wire
from .policy_engine import (
    APPLICABILITY, ENGINE_VERSION, _evidence_refs, _request, _require,
    evaluate_document, test_jdm, validate_jdm, validate_source,
)


def _observation_scope(scene, value):
    """Read explicit evidence references; a shared photo is not object identity."""
    observations = {o["id"] for o in scene["observations"]}
    annotations = {a["id"]: a for a in scene.get("annotations", [])}
    found, visited = set(), set()

    def visit(item):
        if isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            if item in observations:
                found.add(item)
            elif item in annotations and item not in visited:
                visited.add(item)
                visit(annotations[item].get("sourceRefs", []))
        elif isinstance(item, dict):
            if item.get("observationId") in observations:
                found.add(item["observationId"])
            for key in ("sourceRefs", "evidenceRefs", "annotationId"):
                if key in item:
                    visit(item[key])

    visit(value)
    return found


def _identity_evidence_binding(source, scene, original, finding, refs, *, source_revision_id, ancestry_ids):
    old_id, new_id = original.get("entityId"), finding.get("entityId")
    old = next((e for e in source["entities"] if e["id"] == old_id), None)
    new = next((e for e in scene["entities"] if e["id"] == new_id), None)
    old_refs = set((old or {}).get("observationRefs", []))
    new_refs = set((new or {}).get("observationRefs", []))
    if old_id == new_id and old_refs == new_refs:
        return None
    _require(old is not None and new is not None and scene.get("schemaVersion") == 2 and source_revision_id in ancestry_ids, "evidence_identity_scope_mismatch")
    original_facts = _observation_scope(source, original.get("facts", []))
    scope = original_facts.intersection(old_refs) or old_refs
    _require(bool(scope), "evidence_identity_scope_mismatch")
    try:
        targets = {resolve_entity_id(scene, old_id, observation_id=oid) for oid in scope}
    except PlatformError:
        raise PlatformError("evidence_identity_scope_mismatch", 422) from None
    _require(targets == {new_id}, "evidence_identity_scope_mismatch")
    current_facts = _observation_scope(scene, finding.get("facts", []))
    _require(_observation_scope(scene, refs) == scope and current_facts.intersection(new_refs) == scope and current_facts - new_refs == original_facts - old_refs, "evidence_identity_scope_mismatch")
    decisions = [d for d in scene["identityDecisions"] if d["baseRevisionId"] in ancestry_ids
                 and (d["decision"] == "same" or (d["decision"] == "different" and len(d["entityIds"]) == 1))
                 and scope.intersection(oid for group in d["observationGroups"] for oid in group)]
    _require(any(old_id in d["entityIds"] and scope <= {oid for group in d["observationGroups"] for oid in group} for d in decisions), "evidence_identity_scope_mismatch")
    return {"sourceSceneRevisionId": source_revision_id, "sourceFindingId": original["id"], "sourceEntityId": old_id,
            "targetEntityId": new_id, "observationIds": sorted(scope), "identityDecisionIds": [d["id"] for d in decisions]}


class PostgresPolicyRepository:
    def __init__(self, repository):
        self.repo = repository

    def authorize(self, project_id, capability):
        return self.repo.authorize(project_id, capability)

    def _replay(self, c, table, project_id, body):
        _request(body)
        assert table in {"policies", "policy_revisions", "policy_evaluations", "policy_reviews", "policy_evidence_requests"}
        c.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (str(project_id) + ":policy:" + str(body["requestId"]),))
        prior = c.execute(f"SELECT * FROM {table} WHERE project_id=%s AND request_id=%s", (project_id, body["requestId"])).fetchone()
        if prior and prior["request_sha256"].strip() != digest(body):
            raise PlatformError("idempotency_mismatch", 409)
        return prior

    def list_policies(self):
        with self.repo._connect() as c:
            return {"items": _wire(c.execute("SELECT * FROM policies ORDER BY created_at DESC LIMIT 200").fetchall())}

    def get_policy(self, policy_id):
        with self.repo._connect() as c:
            policy = self.repo._one(c, "SELECT * FROM policies WHERE id=%s", (policy_id,))
            revisions = c.execute("SELECT * FROM policy_revisions WHERE policy_id=%s ORDER BY created_at,id", (policy_id,)).fetchall()
            sources = c.execute("SELECT DISTINCT s.* FROM policy_sources s JOIN policy_revisions r ON r.source_id=s.id WHERE r.policy_id=%s", (policy_id,)).fetchall()
            return _wire({"policy": policy, "revisions": revisions, "sources": sources})

    def create_policy(self, project_id, capability, body):
        _request(body, ("source", "jdm", "title"))
        _require(isinstance(body["title"], str) and 0 < len(body["title"].strip()) <= 240, "policy_title_invalid")
        source = validate_source(body["source"])
        validate_jdm(body["jdm"])
        with self.repo._connect() as c:
            self.repo._auth(c, project_id, capability)
            prior = self._replay(c, "policies", project_id, body)
            if prior:
                revision = self.repo._one(c, "SELECT * FROM policy_revisions WHERE id=%s", (prior["draft_revision_id"],))
                return _wire({"policy": prior, "revision": revision})
            sid, pid, rid = uuid4(), uuid4(), uuid4()
            c.execute("INSERT INTO policy_sources(id,project_id,content_sha256,document) VALUES(%s,%s,%s,%s)", (sid, project_id, digest(source), Jsonb(source)))
            policy = c.execute("INSERT INTO policies(id,project_id,title,request_id,request_sha256,draft_revision_id) VALUES(%s,%s,%s,%s,%s,%s) RETURNING *", (pid, project_id, body["title"], body["requestId"], digest(body), rid)).fetchone()
            revision = self._insert(c, project_id, pid, sid, body, rid=rid)
            return _wire({"policy": policy, "revision": revision})

    def _insert(self, c, project_id, policy_id, source_id, body, parent=None, rid=None):
        validate_jdm(body["jdm"])
        doc = {"schemaVersion": 1, "jdm": body["jdm"], "tests": body.get("tests", []), "limitations": body.get("limitations", []), "sourceRefs": body.get("sourceRefs", []), "sourceId": str(source_id)}
        return c.execute("INSERT INTO policy_revisions(id,project_id,policy_id,source_id,parent_revision_id,request_id,request_sha256,document,document_sha256,agent_turn_id) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *", (rid or uuid4(), project_id, policy_id, source_id, parent, body["requestId"], digest(body), Jsonb(doc), digest(doc), body.get("agentTurnId"))).fetchone()

    def save_revision(self, project_id, policy_id, capability, body):
        _request(body, ("basePolicyRevisionId", "jdm"))
        with self.repo._connect() as c:
            self.repo._auth(c, project_id, capability)
            prior = self._replay(c, "policy_revisions", project_id, body)
            if prior:
                return _wire(prior)
            policy = self.repo._one(c, "SELECT * FROM policies WHERE project_id=%s AND id=%s FOR UPDATE", (project_id, policy_id))
            if str(policy["draft_revision_id"]) != body["basePolicyRevisionId"]:
                raise PlatformError("policy_revision_conflict", 409, currentRevisionId=str(policy["draft_revision_id"]))
            base = self.repo._one(c, "SELECT * FROM policy_revisions WHERE id=%s AND policy_id=%s", (body["basePolicyRevisionId"], policy_id))
            if body.get("agentTurnId"):
                turn = self.repo._one(c, "SELECT * FROM agent_turns WHERE project_id=%s AND id=%s", (project_id, body["agentTurnId"]), code="agent_turn_not_found")
                proposal = turn["response"] or {}
                _require(turn["status"] == "succeeded" and proposal.get("proposalType") == "policy" and str(proposal.get("policyId")) == str(policy_id) and proposal.get("basePolicyRevisionId") == body["basePolicyRevisionId"] and all(proposal.get(key, [] if key != "jdm" else None) == body.get(key, [] if key != "jdm" else None) for key in ("jdm", "tests", "limitations", "sourceRefs")), "agent_policy_proposal_mismatch")
            revision = self._insert(c, project_id, policy_id, base["source_id"], body, parent=base["id"])
            c.execute("UPDATE policies SET draft_revision_id=%s WHERE id=%s", (revision["id"], policy_id))
            return _wire(revision)

    def activate(self, project_id, policy_id, capability, body):
        _request(body, ("policyRevisionId",))
        with self.repo._connect() as c:
            self.repo._auth(c, project_id, capability)
            policy = self.repo._one(c, "SELECT * FROM policies WHERE project_id=%s AND id=%s FOR UPDATE", (project_id, policy_id))
            history = policy["activation_requests"]
            if body["requestId"] in history:
                if history[body["requestId"]] != digest(body):
                    raise PlatformError("idempotency_mismatch", 409)
                return {"policyRevisionId": body["policyRevisionId"], "active": str(policy["active_revision_id"]) == body["policyRevisionId"]}
            current = str(policy["active_revision_id"]) if policy["active_revision_id"] else None
            if current != body.get("expectedActiveRevisionId"):
                raise PlatformError("policy_activation_conflict", 409, currentRevisionId=current)
            revision = self.repo._one(c, "SELECT * FROM policy_revisions WHERE project_id=%s AND policy_id=%s AND id=%s", (project_id, policy_id, body["policyRevisionId"]))
            doc = revision["document"]
            source = self.repo._one(c, "SELECT document FROM policy_sources WHERE id=%s", (revision["source_id"],))["document"]
            validate_source(source)
            tests = test_jdm(doc["jdm"], doc["tests"])
            _require(tests and all(t["passed"] for t in tests), "policy_tests_must_pass")
            _require({t.get("input", {}).get("applicability") for t in doc["tests"]} >= APPLICABILITY, "policy_applicability_tests_required")
            history[body["requestId"]] = digest(body)
            c.execute("UPDATE policies SET active_revision_id=%s,activation_requests=%s WHERE id=%s", (revision["id"], Jsonb(history), policy_id))
            return {"policyRevisionId": str(revision["id"]), "active": True, "tests": tests}

    def evaluate(self, project_id, capability, body):
        _request(body, ("sceneRevisionId", "policyRevisionIds", "context"))
        _require(isinstance(body["policyRevisionIds"], list) and 1 <= len(body["policyRevisionIds"]) <= 100, "evaluation_input_invalid")
        _require(isinstance(body.get("context"), str) and body["context"] in {"observed", "planning"} and body.get("policyRevisionIds"), "evaluation_input_invalid")
        with self.repo._connect() as c:
            self.repo._auth(c, project_id, capability)
            prior = self._replay(c, "policy_evaluations", project_id, body)
            if prior:
                return _wire(prior)
            revision = self.repo._revision(c, project_id, body["sceneRevisionId"])
            branch = self.repo._one(c, "SELECT kind FROM scene_branches WHERE id=%s", (revision["branch_id"],))
            _require((branch["kind"] == "planning") == (body["context"] == "planning"), "evaluation_context_mismatch")
            findings, policy_refs = [], []
            for rid in body["policyRevisionIds"]:
                policy = self.repo._one(c, "SELECT r.*,p.active_revision_id,p.title AS policy_title FROM policy_revisions r JOIN policies p ON p.id=r.policy_id WHERE r.project_id=%s AND r.id=%s", (project_id, rid))
                _require(policy["active_revision_id"] == policy["id"], "policy_revision_not_active")
                items = evaluate_document(revision["document"], str(policy["policy_id"]), policy["document"], body["context"])
                for finding in items:
                    finding.update(policyRevisionId=rid, policyTitle=policy["policy_title"], sourceId=str(policy["source_id"]), sceneRevisionId=body["sceneRevisionId"])
                findings.extend(items)
                policy_refs.append(rid)
            doc = {"schemaVersion": 1, "sceneRevisionId": body["sceneRevisionId"], "policyRevisionIds": policy_refs, "context": body["context"], "evaluatorVersion": ENGINE_VERSION, "factSetSha256": digest({"entities": revision["document"]["entities"], "annotations": revision["document"].get("annotations", [])}), "findings": findings}
            row = c.execute("INSERT INTO policy_evaluations(id,project_id,scene_revision_id,request_id,request_sha256,context,document) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING *", (uuid4(), project_id, revision["id"], body["requestId"], digest(body), body["context"], Jsonb(doc))).fetchone()
            return _wire(row)

    def _finding(self, c, project_id, evaluation_id, finding_id):
        evaluation = self.repo._one(c, "SELECT * FROM policy_evaluations WHERE project_id=%s AND id=%s", (project_id, evaluation_id))
        finding = next((f for f in evaluation["document"]["findings"] if f["id"] == str(finding_id)), None)
        _require(finding is not None, "evaluation_finding_mismatch")
        _require(evaluation["document"]["sceneRevisionId"] == str(evaluation["scene_revision_id"]) and finding.get("sceneRevisionId") == str(evaluation["scene_revision_id"]) and finding.get("policyRevisionId") in evaluation["document"]["policyRevisionIds"], "evaluation_finding_mismatch")
        return evaluation, finding

    def review(self, project_id, finding_id, capability, body, *, evidence_request=False):
        _request(body, ("evaluationId",))
        table = "policy_evidence_requests" if evidence_request else "policy_reviews"
        with self.repo._connect() as c:
            self.repo._auth(c, project_id, capability)
            prior = self._replay(c, table, project_id, body)
            if prior:
                return _wire(prior)
            evaluation, finding = self._finding(c, project_id, body["evaluationId"], finding_id)
            if evidence_request:
                _require(isinstance(body.get("action"), str) and bool(body["action"].strip()), "evidence_action_required")
                doc = {"schemaVersion": 1, "action": body["action"], "status": "open", "missingEvidence": finding["missingEvidence"], "entityId": finding["entityId"]}
            else:
                _require(isinstance(body.get("decision"), str) and body["decision"] in {"confirmed", "rejected", "needs_evidence"}, "review_decision_invalid")
                _require(isinstance(body.get("reason"), str) and body["reason"].strip() and isinstance(body.get("displayName"), str) and body["displayName"].strip(), "review_reason_required")
                refs = body.get("evidenceRefs", [])
                _require(isinstance(refs, list), "review_evidence_invalid")
                if refs:
                    scene = self.repo._revision(c, project_id, evaluation["scene_revision_id"])["document"]
                    _require(_evidence_refs(scene, refs), "review_evidence_invalid")
                doc = {"schemaVersion": 1, "decision": body["decision"], "reason": body["reason"], "displayName": body["displayName"], "identityVerified": False, "evidenceRefs": body.get("evidenceRefs", []), "actor": "project_capability", "sceneRevisionId": str(evaluation["scene_revision_id"])}
            return _wire(c.execute(f"INSERT INTO {table}(id,project_id,evaluation_id,finding_id,request_id,request_sha256,document) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING *", (uuid4(), project_id, evaluation["id"], finding_id, body["requestId"], digest(body), Jsonb(doc))).fetchone())

    def publication_evidence(self, project_id, scene_revision_id, evaluation_ids, review_ids):
        with self.repo._connect() as c:
            evaluations, reviews = [], []
            for identity in evaluation_ids:
                row = self.repo._one(c, "SELECT * FROM policy_evaluations WHERE project_id=%s AND id=%s AND scene_revision_id=%s", (project_id, identity, scene_revision_id), code="publication_evaluation_mismatch")
                evaluations.append(_wire(row))
            for identity in review_ids:
                row = self.repo._one(c, "SELECT * FROM policy_reviews WHERE project_id=%s AND id=%s", (project_id, identity))
                _require(str(row["evaluation_id"]) in evaluation_ids, "publication_review_mismatch")
                self._finding(c, project_id, row["evaluation_id"], row["finding_id"])
                reviews.append(_wire(row))
            return {"evaluations": evaluations, "reviews": reviews}

    def fulfill_evidence_request(self, project_id, evidence_request_id, capability, body):
        _request(body, ("evaluationId", "findingId", "evidenceRefs"))
        body = {**body, "evidenceRequestId": str(evidence_request_id)}
        with self.repo._connect() as c:
            self.repo._auth(c, project_id, capability)
            prior = self._replay(c, "policy_evidence_requests", project_id, body)
            if prior:
                return _wire(prior)
            original = self.repo._one(c, "SELECT * FROM policy_evidence_requests WHERE project_id=%s AND id=%s FOR UPDATE", (project_id, evidence_request_id), code="evidence_request_not_found")
            _require(original["document"]["status"] == "open", "evidence_request_not_open")
            already = c.execute("SELECT id FROM policy_evidence_requests WHERE project_id=%s AND document->>'evidenceRequestId'=%s", (project_id, str(evidence_request_id))).fetchone()
            if already:
                raise PlatformError("evidence_request_already_fulfilled", 409, fulfillmentId=str(already["id"]))
            original_evaluation, original_finding = self._finding(c, project_id, original["evaluation_id"], original["finding_id"])
            evaluation, finding = self._finding(c, project_id, body["evaluationId"], body["findingId"])
            _require(str(evaluation["id"]) != str(original["evaluation_id"]), "new_evaluation_required")
            _require(evaluation["context"] == original_evaluation["context"] and finding.get("policyRevisionId") == original_finding.get("policyRevisionId"), "evidence_finding_mismatch")
            revision = self.repo._revision(c, project_id, evaluation["scene_revision_id"])
            source = self.repo._revision(c, project_id, original_evaluation["scene_revision_id"])
            scene = revision["document"]
            _require(isinstance(body["evidenceRefs"], list) and bool(body["evidenceRefs"]) and _evidence_refs(scene, body["evidenceRefs"]), "review_evidence_invalid")
            ancestry = c.execute("""WITH RECURSIVE ancestry AS (
                SELECT id,parent_revision_id FROM scene_revisions WHERE project_id=%s AND id=%s
                UNION ALL SELECT r.id,r.parent_revision_id FROM scene_revisions r JOIN ancestry a ON r.id=a.parent_revision_id
                WHERE r.project_id=%s AND a.id<>%s)
                SELECT id FROM ancestry""", (project_id, revision["id"], project_id, source["id"])).fetchall()
            ancestry_ids = {str(row["id"]) for row in ancestry} if revision["branch_id"] == source["branch_id"] else set()
            binding = _identity_evidence_binding(source["document"], scene, original_finding, finding, body["evidenceRefs"], source_revision_id=str(source["id"]), ancestry_ids=ancestry_ids)
            doc = {"schemaVersion": 1, "status": "fulfilled", "evidenceRequestId": str(evidence_request_id), "evidenceRefs": body["evidenceRefs"], "sceneRevisionId": str(evaluation["scene_revision_id"]), "evaluationId": str(evaluation["id"]), "findingId": body["findingId"], "actor": "project_capability"}
            if binding is not None:
                doc["identityBinding"] = binding
            return _wire(c.execute("INSERT INTO policy_evidence_requests(id,project_id,evaluation_id,finding_id,request_id,request_sha256,document) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING *", (uuid4(), project_id, evaluation["id"], body["findingId"], body["requestId"], digest(body), Jsonb(doc))).fetchone())

    def list_evaluations(self, project_id):
        with self.repo._connect() as c:
            self.repo._one(c, "SELECT id FROM projects WHERE id=%s", (project_id,), code="project_not_found")
            return {"items": _wire(c.execute("SELECT * FROM policy_evaluations WHERE project_id=%s ORDER BY created_at DESC,id LIMIT 200", (project_id,)).fetchall())}

    def list_reviews(self, project_id):
        with self.repo._connect() as c:
            self.repo._one(c, "SELECT id FROM projects WHERE id=%s", (project_id,), code="project_not_found")
            return {"items": _wire(c.execute("SELECT * FROM policy_reviews WHERE project_id=%s ORDER BY created_at DESC,id LIMIT 500", (project_id,)).fetchall())}

    def list_evidence_requests(self, project_id):
        with self.repo._connect() as c:
            self.repo._one(c, "SELECT id FROM projects WHERE id=%s", (project_id,), code="project_not_found")
            return {"items": _wire(c.execute("SELECT * FROM policy_evidence_requests WHERE project_id=%s ORDER BY created_at DESC,id LIMIT 500", (project_id,)).fetchall())}
