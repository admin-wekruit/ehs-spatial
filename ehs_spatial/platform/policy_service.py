"""Policy HTTP/domain entry points backed by an explicit business repository."""
from io import BytesIO
from uuid import UUID
from fastapi import Header, UploadFile, File

from .contracts import (PlatformError, EvidenceRequest, Evaluation, Items, Policy,
                        PolicyActivation, PolicyCreated, PolicyDetail, PolicyRevision,
                        PolicyTemplate, PolicyTestResult, Review, SourceText)
from .repository import PolicyRepository
from .policy_engine import (
    _cap, _require, templates, test_jdm, validate_jdm, execute_jdm,
    evaluate_document, validate_source,
)


class PolicyService:
    def __init__(self, repository: PolicyRepository, blobs=None):
        self.repository, self.blobs = repository, blobs

    def list_policies(self):
        return self.repository.list_policies()

    def get_policy(self, policy_id):
        return self.repository.get_policy(policy_id)

    def create_policy(self, project_id, capability, body):
        return self.repository.create_policy(project_id, capability, body)

    def save_revision(self, project_id, policy_id, capability, body):
        return self.repository.save_revision(project_id, policy_id, capability, body)

    def activate(self, project_id, policy_id, capability, body):
        return self.repository.activate(project_id, policy_id, capability, body)

    def evaluate(self, project_id, capability, body):
        return self.repository.evaluate(project_id, capability, body)

    def review(self, project_id, finding_id, capability, body, *, evidence_request=False):
        return self.repository.review(project_id, finding_id, capability, body, evidence_request=evidence_request)

    def publication_evidence(self, project_id, scene_revision_id, evaluation_ids, review_ids):
        return self.repository.publication_evidence(project_id, scene_revision_id, evaluation_ids, review_ids)

    def fulfill_evidence_request(self, project_id, evidence_request_id, capability, body):
        return self.repository.fulfill_evidence_request(project_id, evidence_request_id, capability, body)

    def register_routes(self, app):
        @app.get("/api/policy-templates", response_model=Items[PolicyTemplate], response_model_exclude_unset=True)
        def policy_templates(): return {"items": templates()}
        @app.get("/api/policies", response_model=Items[Policy], response_model_exclude_unset=True)
        def policies(): return self.list_policies()
        @app.get("/api/policies/{policy_id}", response_model=PolicyDetail, response_model_exclude_unset=True)
        def policy(policy_id: UUID): return self.get_policy(policy_id)
        @app.post("/api/policies/test", response_model=Items[PolicyTestResult], response_model_exclude_unset=True)
        def test(body: dict): return {"items": test_jdm(body.get("jdm"), body.get("tests", []))}
        @app.post("/api/projects/{project_id}/policy-source-text", response_model=SourceText, response_model_exclude_unset=True)
        async def source_text(project_id: UUID, file: UploadFile = File(...), authorization: str | None = Header(None)):
            self.repository.authorize(project_id, _cap(authorization))
            data = await file.read(32 * 1024 * 1024 + 1)
            _require(len(data) <= 32 * 1024 * 1024, "upload_too_large")
            from pypdf import PdfReader
            try:
                pages = [{"page": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(PdfReader(BytesIO(data)).pages)]
            except Exception:
                raise PlatformError("policy_pdf_invalid", 422)
            return {"pages": pages, "status": "text_extracted" if any(p["text"].strip() for p in pages) else "source_text_required"}
        @app.post("/api/projects/{project_id}/policies", response_model=PolicyCreated, response_model_exclude_unset=True)
        def create(project_id: UUID, body: dict, authorization: str | None = Header(None)): return self.create_policy(project_id, _cap(authorization), body)
        @app.post("/api/projects/{project_id}/policies/{policy_id}/revisions", response_model=PolicyRevision, response_model_exclude_unset=True)
        def save(project_id: UUID, policy_id: UUID, body: dict, authorization: str | None = Header(None)): return self.save_revision(project_id, policy_id, _cap(authorization), body)
        @app.post("/api/projects/{project_id}/policies/{policy_id}/activate", response_model=PolicyActivation, response_model_exclude_unset=True)
        def activate(project_id: UUID, policy_id: UUID, body: dict, authorization: str | None = Header(None)): return self.activate(project_id, policy_id, _cap(authorization), body)
        @app.post("/api/projects/{project_id}/evaluations", response_model=Evaluation, response_model_exclude_unset=True)
        def evaluate(project_id: UUID, body: dict, authorization: str | None = Header(None)): return self.evaluate(project_id, _cap(authorization), body)
        @app.get("/api/projects/{project_id}/evaluations", response_model=Items[Evaluation], response_model_exclude_unset=True)
        def evaluations(project_id: UUID):
            return self.repository.list_evaluations(project_id)
        @app.post("/api/projects/{project_id}/findings/{finding_id}/reviews", response_model=Review, response_model_exclude_unset=True)
        def review(project_id: UUID, finding_id: UUID, body: dict, authorization: str | None = Header(None)): return self.review(project_id, finding_id, _cap(authorization), body)
        @app.post("/api/projects/{project_id}/findings/{finding_id}/evidence-requests", response_model=EvidenceRequest, response_model_exclude_unset=True)
        def evidence(project_id: UUID, finding_id: UUID, body: dict, authorization: str | None = Header(None)): return self.review(project_id, finding_id, _cap(authorization), body, evidence_request=True)
        @app.get("/api/projects/{project_id}/reviews", response_model=Items[Review], response_model_exclude_unset=True)
        def reviews(project_id: UUID):
            return self.repository.list_reviews(project_id)
        @app.get("/api/projects/{project_id}/evidence-requests", response_model=Items[EvidenceRequest], response_model_exclude_unset=True)
        def evidence_requests(project_id: UUID):
            return self.repository.list_evidence_requests(project_id)
        @app.post("/api/projects/{project_id}/evidence-requests/{evidence_request_id}/fulfill", response_model=EvidenceRequest, response_model_exclude_unset=True)
        def fulfill(project_id: UUID, evidence_request_id: UUID, body: dict, authorization: str | None = Header(None)): return self.fulfill_evidence_request(project_id, evidence_request_id, _cap(authorization), body)
