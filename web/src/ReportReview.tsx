import { useEffect, useRef, useState } from "react";
import { id, request } from "./api";
import { useI18n } from "./i18n";
import { ErrorNotice } from "./App";
import { SafetyEvidence } from "./SafetyEvidence";
import { EvidenceLedger } from "./EvidenceLedger";
import { reportReviewMessages } from "./report-review-messages";
import type { Commit, Evaluation, EvidenceRequest, Policy, PolicyDetail, PolicyRevision, PolicyTemplate, ProjectDetail, Publication, Review } from "./types";
import "./report-review.css";

export type ReportReviewProps = {
  detail: ProjectDetail;
  publication?: Publication;
  reviewMode: boolean;
  canWrite: boolean;
  onSelect: (entityId: string) => void;
  onSaved: (commit: Commit) => void;
  onError?: (error: unknown) => void;
  onSummary?: (summary: AssessmentSummary) => void;
  onEvaluations?: (records: { revisionId: string; evaluations: Evaluation[] | null }) => void;
};

export type AssessmentSummary = {
  revisionId: string;
  state: "loading" | "unavailable" | "unassessed" | "assessed";
  evaluationCount: number;
  attentionCount: number;
};
export function findingResult(finding: Evaluation["document"]["findings"][number]) {
  return finding.applicability === "unknown" ? "APPLICABILITY_UNKNOWN" : finding.applicability === "not_applicable" ? "NOT_APPLICABLE" : finding.machineResult || "INSUFFICIENT_EVIDENCE";
}
export function assessmentSummary(revisionId: string, evaluations: Evaluation[]): AssessmentSummary {
  const exact = exactReviewEvidence(revisionId, evaluations, []).evaluations;
  return { revisionId, state: exact.length ? "assessed" : "unassessed", evaluationCount: exact.length,
    attentionCount: exact.flatMap(e => e.document.findings).filter(f => !["PASS", "NOT_APPLICABLE"].includes(findingResult(f))).length };
}

// Findings and reviews belong to an evaluation, never merely to a matching label.
export function exactReviewEvidence(revisionId: string, evaluations: Evaluation[], reviews: Review[]) {
  const exact = evaluations.filter(e => e.sceneRevisionId === revisionId && e.document.sceneRevisionId === revisionId).map(e => ({ ...e, document: { ...e.document, findings: e.document.findings.filter(f => f.sceneRevisionId === revisionId) } }));
  return {
    evaluations: exact,
    reviews: reviews.filter(r => r.document.sceneRevisionId === revisionId && exact.some(e => e.id === r.evaluationId && e.document.findings.some(f => f.id === r.findingId))),
  };
}

const fieldKeys: Record<string, string> = { source: "rrFieldSource", value: "rrFieldValue", unit: "rrFieldUnit", uncertaintyM: "rrFieldUncertainty", requirement: "rrFieldRequirement", passed: "rrFieldPassed", sourceRefs: "rrFieldRefs", evidenceRefs: "rrFieldRefs", annotationId: "rrFieldAnnotation", assetId: "rrFieldAsset", coordinateFrameId: "rrFieldFrame" };
export const missingKeys: Record<string, string> = { applicability_confirmation: "rrApplicabilityMissing", target_inventory_confirmation: "rrInventoryMissing", metric_footprint: "rrFootprintMissing", metric_calibration: "rrCalibrationMissing", metric_height: "rrHeightMissing", invalid_geometry: "rrGeometryMissing", registered_coordinate_frame: "rrRegistrationMissing" };

function EvidenceValue({ value, t }: { value: unknown; t: (key: string) => string }) {
  if (value === null || value === undefined) return <span className="rr-muted">{t("rrEvidenceMissing")}</span>;
  if (typeof value === "boolean") return <>{t(value ? "rrTrue" : "rrFalse")}</>;
  if (typeof value !== "object") return <>{String(value)}</>;
  if (Array.isArray(value)) return value.length ? <ul className="rr-value-list">{value.map((v, i) => <li key={i}><EvidenceValue value={v} t={t} /></li>)}</ul> : <span>—</span>;
  return <dl className="rr-facts">{Object.entries(value).map(([key, v]) => <div key={key}><dt>{t(fieldKeys[key] || key)}</dt><dd><EvidenceValue value={v} t={t} /></dd></div>)}</dl>;
}

function sourceURL(value: unknown) {
  try { const url = new URL(String(value)); return ["http:", "https:"].includes(url.protocol) ? url.href : null; } catch { return null; }
}

export function ReportReview({ detail, publication, reviewMode, canWrite, onSelect, onSaved, onError, onSummary, onEvaluations }: ReportReviewProps) {
  const { language, t: globalT } = useI18n();
  const t = (key: string) => reportReviewMessages[key]?.[language === "zh" ? 0 : 1] || globalT(key);
  const revision = publication?.snapshot.revision || detail.revision;
  const projectId = publication?.projectId || detail.project.id;
  const scope = `${projectId}:${revision.id}:${publication?.id || "live"}`;
  const current = useRef(scope); current.current = scope;
  const writable = reviewMode && canWrite && !publication && detail.revision.id === revision.id;
  const [records, setRecords] = useState<{ scope: string; evaluations: Evaluation[]; reviews: Review[]; evidence: EvidenceRequest[] }>();
  const [policyRecords, setPolicyRecords] = useState<{ projectId: string; policies: Policy[]; details: PolicyDetail[] }>();
  const [templates, setTemplates] = useState<PolicyTemplate[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [reload, setReload] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>();
  const [status, setStatus] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [reasons, setReasons] = useState<Record<string, string>>({});
  const [decisions, setDecisions] = useState<Record<string, Review["document"]["decision"]>>({});
  const [templateIndex, setTemplateIndex] = useState("");
  const [draft, setDraft] = useState<{ policy: Policy; revision: PolicyRevision }>();
  const [testResults, setTestResults] = useState<{ name: unknown; passed: boolean }[]>([]);
  const pending = useRef(new Map<string, string>());
  const reportError = (e: unknown) => { if (current.current === scope) { setError(e); onError?.(e); } };

  useEffect(() => {
    setError(undefined); setStatus(""); setReasons({}); setDecisions({}); setBusy(false);
  }, [scope]);

  useEffect(() => {
    if (publication) return;
    const abort = new AbortController();
    Promise.all([
      request<{ items: Evaluation[] }>(`/api/projects/${projectId}/evaluations`, { signal: abort.signal }),
      request<{ items: Review[] }>(`/api/projects/${projectId}/reviews`, { signal: abort.signal }),
      reviewMode ? request<{ items: EvidenceRequest[] }>(`/api/projects/${projectId}/evidence-requests`, { signal: abort.signal }) : Promise.resolve({ items: [] }),
    ]).then(([e, r, evidence]) => {
      if (!abort.signal.aborted && current.current === scope) setRecords({ scope, evaluations: e.items, reviews: r.items, evidence: evidence.items });
    }).catch(e => { if (!abort.signal.aborted) reportError(e); });
    return () => abort.abort();
  }, [scope, reload, reviewMode]);

  useEffect(() => {
    const abort = new AbortController();
    request<{ items: Policy[] }>("/api/policies", { signal: abort.signal }).then(async result => {
      const policies = result.items.filter(p => p.projectId === projectId);
      const details = await Promise.all(policies.map(p => request<PolicyDetail>(`/api/policies/${p.id}`, { signal: abort.signal })));
      if (!abort.signal.aborted) setPolicyRecords({ projectId, policies, details });
    }).catch(e => { if (!abort.signal.aborted) reportError(e); });
    if (writable) request<{ items: PolicyTemplate[] }>("/api/policy-templates", { signal: abort.signal }).then(r => { if (!abort.signal.aborted) setTemplates(r.items); }).catch(e => { if (!abort.signal.aborted) reportError(e); });
    return () => abort.abort();
  }, [projectId, reload, writable]);

  useEffect(() => { setSelected([]); setTemplateIndex(""); setDraft(undefined); setTestResults([]); }, [projectId]);

  const sourceRecords = publication ? publication.snapshot : records?.scope === scope ? records : { evaluations: [], reviews: [], evidence: [] };
  const { evaluations, reviews } = exactReviewEvidence(revision.id, sourceRecords.evaluations, sourceRecords.reviews);
  useEffect(() => {
    onEvaluations?.({ revisionId: revision.id, evaluations: !publication && records?.scope !== scope ? null : sourceRecords.evaluations });
    onSummary?.(!publication && records?.scope !== scope
      ? { revisionId: revision.id, state: error ? "unavailable" : "loading", evaluationCount: 0, attentionCount: 0 }
      : assessmentSummary(revision.id, sourceRecords.evaluations));
  }, [scope, publication, records, error, onSummary, onEvaluations]);
  const policies = policyRecords?.projectId === projectId ? policyRecords.policies : [];
  const policyDetails = policyRecords?.projectId === projectId ? policyRecords.details : [];
  const activePolicies = policies.filter(p => p.activeRevisionId);
  const activeIds = selected.filter(value => activePolicies.some(p => p.activeRevisionId === value));
  const evidence = records?.scope === scope ? records.evidence : [];
  const requests = evidence.filter(r => evaluations.some(e => e.id === r.evaluationId));
  const requestIds = new Set(requests.map(r => r.id));
  const ledger = evidence.filter(r => requestIds.has(r.id) || requestIds.has(String(r.document.evidenceRequestId)));
  const template = templateIndex !== "" ? templates[Number(templateIndex)] : undefined;
  const findings = evaluations.flatMap(e => e.document.findings);
  const counts = findings.reduce<Record<string, number>>((result, f) => { const key = f.applicability === "unknown" ? "APPLICABILITY_UNKNOWN" : f.applicability === "not_applicable" ? "NOT_APPLICABLE" : f.machineResult || "INSUFFICIENT_EVIDENCE"; result[key] = (result[key] || 0) + 1; return result; }, {});

  async function post<T>(route: string, body: Record<string, unknown>): Promise<T> {
    const key = JSON.stringify([route, body]);
    if (!pending.current.has(key)) pending.current.set(key, id());
    const result = await request<T>(route, { method: "POST", projectId, body: { ...body, requestId: pending.current.get(key) } });
    pending.current.delete(key);
    return result;
  }
  async function act(operation: () => Promise<unknown>, message: string) {
    if (!writable || busy) return;
    setBusy(true); setError(undefined); setStatus("");
    try { await operation(); if (current.current === scope) { setStatus(message); setReload(v => v + 1); } }
    catch (e) { reportError(e); }
    finally { if (current.current === scope) setBusy(false); }
  }
  const when = (value: string) => new Date(value).toLocaleString(language === "zh" ? "zh-CN" : "en-US");
  const objectButton = (entityId: string) => {
    const entity = revision.document.entities.find(e => e.id === entityId);
    return entity ? <button type="button" className="rr-object" onClick={() => onSelect(entityId)}>{entity.label} <span aria-hidden="true">↗</span><span className="rr-sr-only"> · {t("rrObject")}</span></button> : <span className="rr-muted">{t("rrObjectMissing")}</span>;
  };

  return <section className="report-review" aria-label={t("rrTitle")}>
    <header className="rr-heading"><h2>{t("rrTitle")}</h2><span className="rr-muted" title={revision.id}>{t("rrVersion")} · {revision.id.slice(0, 8)}</span></header>
    {publication && <p className="rr-muted">{t("rrFrozen")}</p>}
    <ErrorNotice error={error} />
    {status && <p className="rr-status" role="status">{t(status)}</p>}
    {!!findings.length && <ul className="rr-counts" aria-label={t("rrAssessment")}>{Object.entries(counts).map(([key, count]) => <li key={key}><strong>{count}</strong> {t(key)}</li>)}</ul>}
    {!publication && records?.scope !== scope && !error ? <p role="status">{t("loading")}</p> : (publication || records?.scope === scope) && !evaluations.length && <div className="rr-empty"><strong>{t("rrEmpty")}</strong><p>{t(writable ? "rrEmptyHint" : "rrReadOnly")}</p></div>}

    <div className="rr-evaluations">{evaluations.map(evaluation => <section className="rr-evaluation" key={evaluation.id}>
      <header className="rr-evaluation-heading"><h3>{t("rrAssessment")} · {when(evaluation.createdAt)}</h3><span>{t(evaluation.context)}</span></header>
      {!evaluation.document.findings.length && <p className="rr-muted">{t("rrNoFindings")}</p>}
      {evaluation.document.findings.slice().sort((a, b) => Number(["PASS", "NOT_APPLICABLE"].includes(findingResult(a))) - Number(["PASS", "NOT_APPLICABLE"].includes(findingResult(b)))).map(finding => {
        const key = `${evaluation.id}:${finding.id}`;
        const policy = policyDetails.find(p => p.revisions.some(r => r.id === finding.policyRevisionId));
        const policyRevision = policy?.revisions.find(r => r.id === finding.policyRevisionId);
        const source = policyRevision?.sourceId === finding.sourceId ? policy?.sources.find(s => s.id === finding.sourceId) : undefined;
        const result = findingResult(finding);
        const defaultReason = finding.applicability === "unknown" ? "rrUnknownReason" : finding.applicability === "not_applicable" ? "rrNotApplicableReason" : result === "PASS" ? "rrPassReason" : result === "FAIL" ? "rrFailReason" : result === "NEEDS_REVIEW" ? "rrReviewReason" : "rrInsufficientReason";
        const reason = typeof finding.reason === "string" ? finding.reason : typeof finding.summary === "string" ? finding.summary : t(defaultReason);
        const findingReviews = reviews.filter(r => r.evaluationId === evaluation.id && r.findingId === finding.id);
        const url = sourceURL(source?.document.url);
        return <article className="rr-finding" key={key}>
          <header className="rr-finding-heading"><h4>{finding.entityId ? objectButton(finding.entityId) : t("rrWorkcellContext")}</h4><span className={`rr-badge rr-${result.toLowerCase()}`}>{t(result)}</span></header>
          <p className="rr-check-title">{finding.policyTitle || t("rrSavedCheck")}</p>
          <p className="rr-outcome-copy">{t(defaultReason)}</p>
          {!!finding.missingEvidence.length && <div className="rr-missing"><h5>{t("rrMissing")}</h5><ul>{finding.missingEvidence.map((missing, i) => { const [kind, entityId] = missing.split(":"); return <li key={i}>{t(missingKeys[kind] || missing)} {entityId && objectButton(entityId)}</li>; })}</ul></div>}
          <details className="rr-reason"><summary>{t("rrReason")}</summary><p>{reason}</p></details>
          <details className="rr-source"><summary>{t("rrSource")}</summary>{source ? <>
            <p><strong>{source.document.title}</strong><br />{[source.document.publisher, source.document.jurisdiction].filter(Boolean).join(" · ")}</p>
            {url && <a href={url} target="_blank" rel="noreferrer">{t("rrFieldSource")} ↗</a>}
            {(source.document.clauses.length ? source.document.clauses : [{ locator: "", text: source.document.text }]).map((clause, i) => <blockquote key={i}><strong>{clause.locator}</strong><p>{clause.text}</p></blockquote>)}
            {!!policyRevision?.document.limitations.length && <><h5>{t("rrLimitations")}</h5><EvidenceValue value={policyRevision.document.limitations} t={t} /></>}
          </> : <p className="rr-muted">{t("rrSourceMissing")}</p>}<small className="rr-muted">{t("rrAssessment")} · {evaluation.id}<br />{t("rrSavedCheck")} · {finding.policyRevisionId}</small></details>
          <details><summary>{t("rrFacts")} · {finding.facts.length}</summary>{finding.facts.length ? <EvidenceValue value={finding.facts} t={t} /> : <p className="rr-muted">{t("rrNoFacts")}</p>}{finding.comparison && <><h5>{t("rrComparison")}</h5><EvidenceValue value={finding.comparison} t={t} /></>}</details>
          <details><summary>{t("rrReviews")} · {findingReviews.length}</summary>{findingReviews.length ? <ol className="rr-reviews">{findingReviews.map(review => <li key={review.id}><strong>{review.document.displayName} · {t("review_" + review.document.decision)}</strong><time dateTime={review.createdAt}>{when(review.createdAt)}</time><p>{review.document.reason}</p>{!!review.document.evidenceRefs.length && <EvidenceValue value={review.document.evidenceRefs} t={t} />}</li>)}</ol> : <p className="rr-muted">{t("rrNoReviews")}</p>}</details>
          {writable && <details className="rr-review-form"><summary>{t("rrWriteReview")}</summary><form onSubmit={event => { event.preventDefault(); if (reviewer.trim() && reasons[key]?.trim()) void act(() => post(`/api/projects/${projectId}/findings/${finding.id}/reviews`, { evaluationId: evaluation.id, decision: decisions[key] || "needs_evidence", reason: reasons[key].trim(), displayName: reviewer.trim(), evidenceRefs: [] }), "rrReviewSaved"); }}>
            <label>{t("reviewName")}<input required value={reviewer} onChange={event => setReviewer(event.target.value)} /></label>
            <label>{t("rrDecision")}<select value={decisions[key] || "needs_evidence"} onChange={event => setDecisions(old => ({ ...old, [key]: event.target.value as Review["document"]["decision"] }))}>{["confirmed", "rejected", "needs_evidence"].map(decision => <option key={decision} value={decision}>{t("review_" + decision)}</option>)}</select></label>
            <label className="rr-full">{t("reviewReason")}<textarea required value={reasons[key] || ""} onChange={event => setReasons(old => ({ ...old, [key]: event.target.value }))} /></label>
            <div className="rr-actions rr-full"><button className="primary" disabled={busy || !reviewer.trim() || !reasons[key]?.trim()}>{t("rrSaveReview")}</button><button type="button" disabled={busy || !reasons[key]?.trim()} onClick={() => void act(() => post(`/api/projects/${projectId}/findings/${finding.id}/evidence-requests`, { evaluationId: evaluation.id, action: reasons[key].trim() }), "rrRequestSaved")}>{t("requestEvidence")}</button></div>
          </form></details>}
        </article>;
      })}
    </section>)}</div>

    {writable && <section className="rr-tools"><h3>{t("rrPolicyPicker")}</h3>
      {activePolicies.length ? <div className="rr-policies">{activePolicies.map(policy => <label key={policy.id}><input type="checkbox" checked={activeIds.includes(policy.activeRevisionId!)} onChange={event => setSelected(old => event.target.checked ? [...old, policy.activeRevisionId!] : old.filter(v => v !== policy.activeRevisionId))} /><span>{policy.title}<small>{policy.activeRevisionId?.slice(0, 8)}</small></span></label>)}</div> : <p className="rr-muted">{t("rrNoActivePolicies")}</p>}
      <button className="primary" disabled={busy || !activeIds.length} onClick={() => void act(() => post(`/api/projects/${projectId}/evaluations`, { sceneRevisionId: revision.id, policyRevisionIds: activeIds, context: detail.branch.kind === "planning" ? "planning" : "observed" }), "rrEvaluated")}>{t(busy ? "loading" : "evaluate")}</button>
      {detail.revision.id === detail.branch.headRevisionId ? <SafetyEvidence key={scope} detail={detail} policies={policies} canWrite={writable && !busy} onSaved={commit => { if (current.current === scope) onSaved(commit); }} /> : <p className="rr-muted">{t("rrHistoricalEvidence")}</p>}
      <EvidenceLedger key={scope + reload} detail={detail} items={ledger} evaluations={evaluations} canWrite={writable && !busy} onChanged={() => setReload(v => v + 1)} />
      <details className="rr-template"><summary>{t("rrTemplate")}</summary>
        <label>{t("rrChooseTemplate")}<select value={templateIndex} onChange={event => { setTemplateIndex(event.target.value); setDraft(undefined); setTestResults([]); }} disabled={busy}><option value="">—</option>{templates.map((value, i) => <option value={i} key={i}>{value.title}</option>)}</select></label>
        {!templates.length && <p>{t("rrNoTemplates")}</p>}
        {template && <><p>{template.source.title} · {template.source.publisher}</p><details><summary>{t("rrSource")}</summary>{template.source.clauses.map((clause, i) => <blockquote key={i}><strong>{clause.locator}</strong><p>{clause.text}</p></blockquote>)}<EvidenceValue value={template.limitations} t={t} /></details>
          {!draft ? <button disabled={busy} onClick={() => void act(async () => { const saved = await post<{ policy: Policy; revision: PolicyRevision }>(`/api/projects/${projectId}/policies`, { title: template.title, source: template.source, jdm: template.jdm, tests: template.tests, limitations: template.limitations }); if (current.current === scope) setDraft(saved); }, "rrDraftSaved")}>{t("rrCreateDraft")}</button> : <>
            <p>{draft.policy.title} · {draft.revision.id.slice(0, 8)}</p><p className="rr-muted">{t("rrTestsHint")}</p>
            <div className="rr-actions"><button disabled={busy} onClick={() => void act(async () => { const result = await post<{ items: { name: unknown; passed: boolean }[] }>("/api/policies/test", { jdm: draft.revision.document.jdm, tests: draft.revision.document.tests }); if (current.current === scope) setTestResults(result.items); }, "rrTests")}>{t("rrTestDraft")}</button>
              <button disabled={busy || !testResults.length || !testResults.every(result => result.passed) || draft.policy.activeRevisionId === draft.revision.id} onClick={() => void act(async () => { await post(`/api/projects/${projectId}/policies/${draft.policy.id}/activate`, { policyRevisionId: draft.revision.id, expectedActiveRevisionId: draft.policy.activeRevisionId }); if (current.current === scope) setDraft({ ...draft, policy: { ...draft.policy, activeRevisionId: draft.revision.id } }); }, "rrActivated")}>{t("rrActivate")}</button></div>
            {!!testResults.length && <ul>{testResults.map((result, i) => <li key={i}>{String(result.name)} · {t(result.passed ? "PASS" : "FAIL")}</li>)}</ul>}
          </>}
        </>}
      </details>
    </section>}
  </section>;
}

export default ReportReview;
