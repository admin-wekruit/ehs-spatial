import { useI18n } from "./i18n";
import { currentEntityId, jsonObject } from "./core";
import { EvidenceValue } from "./ReportEvidence";
import { exactReviewEvidence, findingResult, missingKeys } from "./ReportReview";
import type { Evaluation, Publication, Revision } from "./types";
import "./report-object-findings.css";

const records = (value: unknown): Record<string, unknown>[] => Array.isArray(value) ? value.map(jsonObject).filter((item): item is Record<string, unknown> => !!item) : [];

// Match only IDs or explicit inventory references within the same source run.
// A rule's overall historical FAIL is never inferred from object names or categories.
export function historicalObjectFindings(revision: Revision, entityId: string) {
  const historical = jsonObject(jsonObject(revision.document.reportEvidence)?.historical);
  const findings = records(historical?.findings);
  if (!historical || typeof historical.runId !== "string" || !historical.runId || !revision.document.entities.some(entity => entity.id === entityId)) return { runId: "", total: findings.length, linked: [] };
  const sameRun = (record: Record<string, unknown>) =>
    (record.runId === undefined || record.runId === historical.runId) &&
    (record.sourceRunId === undefined || record.sourceRunId === historical.runId);
  const inventory = new Set(records(historical.inventory).filter(item => sameRun(item) && Array.isArray(item.entityIds) && item.entityIds.length === 1 && item.entityIds[0] === entityId && Number.isInteger(item.inventoryIndex)).map(item => item.inventoryIndex));
  const matches = (record: Record<string, unknown>) => sameRun(record) &&
    ([record.entityId, record.subjectId, record.objectId].some(id => typeof id === "string" && currentEntityId(revision.document, id) === entityId) ||
      Array.isArray(record.entityIds) && record.entityIds.some(id => typeof id === "string" && currentEntityId(revision.document, id) === entityId) ||
      Number.isInteger(record.inventoryIndex) && inventory.has(record.inventoryIndex));
  return { runId: typeof historical.runId === "string" ? historical.runId : "", total: findings.length,
    linked: findings.flatMap(finding => {
      if (!sameRun(finding)) return [];
      const facts = records(finding.facts).filter(matches), violations = records(finding.violations).filter(matches);
      return matches(finding) || facts.length || violations.length ? [{ finding, facts, violations }] : [];
    }) };
}

export function ReportObjectFindings({ revision, publication, entityId, evaluations, loading = false, readOnly = false, onReview }: {
  revision: Revision;
  publication?: Publication;
  entityId: string;
  evaluations?: Evaluation[];
  loading?: boolean;
  readOnly?: boolean;
  onReview: () => void;
}) {
  const { t } = useI18n();
  const scene = publication?.snapshot.revision || revision;
  const scopeMatches = scene.id === revision.id && scene.projectId === revision.projectId && (!publication || publication.projectId === scene.projectId);
  const source = publication ? publication.snapshot.evaluations : evaluations;
  const exact = scopeMatches ? exactReviewEvidence(scene.id, (source || []).filter(evaluation => evaluation.projectId === scene.projectId), []).evaluations : [];
  const rows = exact.flatMap(evaluation => evaluation.document.findings.filter(finding => finding.entityId === entityId).map(finding => ({ evaluation, finding })));
  const historical = historicalObjectFindings(scene, entityId);
  const entity = scene.document.entities.find(value => value.id === entityId);
  if (!scopeMatches || !entity) return <section className="report-object-findings"><h3>{t("objectFindings.title")}</h3><p>{t("objectFindings.unavailable")}</p></section>;
  return <section className="report-object-findings" aria-label={t("objectFindings.title")}>
    <h3>{t("objectFindings.title")}</h3>
    {!source ? <p role="status">{t(loading ? "loading" : "objectFindings.unavailable")}</p> : !exact.length ? <p>{t("rrEmpty")}</p> : !rows.length ? <p>{t("objectFindings.noObjectFinding")}</p> : null}
    {rows.map(({ evaluation, finding }) => {
      const result = findingResult(finding);
      const reasonKey = result === "APPLICABILITY_UNKNOWN" ? "rrUnknownReason" : result === "NOT_APPLICABLE" ? "rrNotApplicableReason" : result === "PASS" ? "rrPassReason" : result === "FAIL" ? "rrFailReason" : result === "NEEDS_REVIEW" ? "rrReviewReason" : "rrInsufficientReason";
      const reason = typeof finding.reason === "string" ? finding.reason : typeof finding.summary === "string" ? finding.summary : t(reasonKey);
      return <article key={evaluation.id + ":" + finding.id}>
        <div className="rof-result"><strong>{finding.policyTitle || t("rrSavedCheck")}</strong><span className={"rr-badge rr-" + result.toLowerCase()}>{t(result)}</span></div>
        <p>{t(reasonKey)}</p>
        {!!finding.missingEvidence.length && <div><h4>{t("rrMissing")}</h4><ul>{finding.missingEvidence.map((missing, i) => {
          const [kind, relatedId] = missing.split(":");
          return <li key={i}>{t(missingKeys[kind] || kind)}{relatedId && <> · {scene.document.entities.find(value => value.id === relatedId)?.label || t("rrObjectMissing")}</>}</li>;
        })}</ul></div>}
        <details><summary>{t("objectFindings.details")}</summary><p>{reason}</p><EvidenceValue value={finding.facts} />{finding.comparison && <EvidenceValue value={finding.comparison} />}<small>{t("rrAssessment")} · {evaluation.id}<br />{t("rrSavedCheck")} · {finding.policyRevisionId}</small></details>
      </article>;
    })}
    <button type="button" onClick={onReview}>{t(readOnly ? "objectFindings.details" : "objectFindings.review")} ↗</button>
    {!!historical.total && <details className="rof-history"><summary>{t("objectFindings.historical")} · {historical.linked.length} {t("objectFindings.historicalCount")} {historical.total} {t("objectFindings.historicalTotal")}</summary>
      {!historical.linked.length && <p>{t("objectFindings.noHistoricalLink")}</p>}
      {historical.linked.map(({ finding, facts, violations }, index) => <article key={String(finding.id || index)}>
        <strong>{String(finding.title || finding.id || "")}</strong><p>{t("objectFindings.historicalScope")}</p>
        <small>{historical.runId} · {t(typeof finding.status === "string" ? finding.status : "unknown")}</small>
        <details><summary>{t("objectFindings.details")}</summary>{typeof finding.summary === "string" && <p>{finding.summary}</p>}<h4>{t("objectFindings.historicalFacts")}</h4><EvidenceValue value={facts} /><h4>{t("objectFindings.historicalViolations")}</h4><EvidenceValue value={violations} /></details>
      </article>)}
      <button type="button" onClick={onReview}>{t("objectFindings.allHistorical")} ↗</button>
    </details>}
  </section>;
}
