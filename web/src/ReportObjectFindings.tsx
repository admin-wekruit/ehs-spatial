import { useI18n } from "./i18n";
import { currentEntityId, jsonObject } from "./core";
import { EvidenceValue } from "./ReportEvidence";
import { exactReviewEvidence, findingResult, missingKeys } from "./ReportReview";
import { reportReviewMessages } from "./report-review-messages";
import type { Evaluation, Publication, Revision } from "./types";
import "./report-object-findings.css";

const messages: Record<string, [string, string]> = {
  title: ["此对象的判定", "Findings for this object"],
  unavailable: ["尚未取得本版本评估记录。", "Assessment records for this revision are unavailable."],
  noObjectFinding: ["本版本已有评估，但没有明确指向此对象的判定。", "This revision has assessments, but no finding explicitly targets this object."],
  review: ["补充证据 / 复核判定", "Add evidence / review findings"],
  details: ["详细事实与判定依据", "Facts and assessment details"],
  historical: ["相关历史资料", "Related historical evidence"],
  historicalCount: ["条明确关联 /", "explicitly linked /"],
  historicalTotal: ["条历史检查", "historical checks"],
  historicalScope: ["历史规则的整体结果，不是此对象在当前版本的判定。", "The historical rule’s overall result, not a finding for this object in the current revision."],
  noHistoricalLink: ["尚无证据将这些历史判定关联到此对象；同名或同类不构成关联。", "No evidence links these historical findings to this object; matching names or categories do not establish a link."],
  allHistorical: ["查看全部历史规则与理由", "View all historical rules and reasoning"],
  historicalFacts: ["与此对象关联的历史事实", "Historical facts linked to this object"],
  historicalViolations: ["与此对象关联的历史违反记录", "Historical violation records linked to this object"],
};
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
  const { language, t: globalT } = useI18n();
  const t = (key: string) => (messages[key] || reportReviewMessages[key])?.[language === "zh" ? 0 : 1] || globalT(key);
  const scene = publication?.snapshot.revision || revision;
  const scopeMatches = scene.id === revision.id && scene.projectId === revision.projectId && (!publication || publication.projectId === scene.projectId);
  const source = publication ? publication.snapshot.evaluations : evaluations;
  const exact = scopeMatches ? exactReviewEvidence(scene.id, (source || []).filter(evaluation => evaluation.projectId === scene.projectId), []).evaluations : [];
  const rows = exact.flatMap(evaluation => evaluation.document.findings.filter(finding => finding.entityId === entityId).map(finding => ({ evaluation, finding })));
  const historical = historicalObjectFindings(scene, entityId);
  const entity = scene.document.entities.find(value => value.id === entityId);
  if (!scopeMatches || !entity) return <section className="report-object-findings"><h3>{t("title")}</h3><p>{t("unavailable")}</p></section>;
  return <section className="report-object-findings" aria-label={t("title")}>
    <h3>{t("title")}</h3>
    {!source ? <p role="status">{t(loading ? "loading" : "unavailable")}</p> : !exact.length ? <p>{t("rrEmpty")}</p> : !rows.length ? <p>{t("noObjectFinding")}</p> : null}
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
        <details><summary>{t("details")}</summary><p>{reason}</p><EvidenceValue value={finding.facts} />{finding.comparison && <EvidenceValue value={finding.comparison} />}<small>{t("rrAssessment")} · {evaluation.id}<br />{t("rrSavedCheck")} · {finding.policyRevisionId}</small></details>
      </article>;
    })}
    <button type="button" onClick={onReview}>{t(readOnly ? "details" : "review")} ↗</button>
    {!!historical.total && <details className="rof-history"><summary>{t("historical")} · {historical.linked.length} {t("historicalCount")} {historical.total} {t("historicalTotal")}</summary>
      {!historical.linked.length && <p>{t("noHistoricalLink")}</p>}
      {historical.linked.map(({ finding, facts, violations }, index) => <article key={String(finding.id || index)}>
        <strong>{String(finding.title || finding.id || "")}</strong><p>{t("historicalScope")}</p>
        <small>{historical.runId} · {t(typeof finding.status === "string" ? finding.status : "unknown")}</small>
        <details><summary>{t("details")}</summary>{typeof finding.summary === "string" && <p>{finding.summary}</p>}<h4>{t("historicalFacts")}</h4><EvidenceValue value={facts} /><h4>{t("historicalViolations")}</h4><EvidenceValue value={violations} /></details>
      </article>)}
      <button type="button" onClick={onReview}>{t("allHistorical")} ↗</button>
    </details>}
  </section>;
}
