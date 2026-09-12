import { useEffect, useState } from "react";
import { id, request } from "./api";
import { useI18n } from "./i18n";
import { ErrorNotice } from "./App";
import type { Commit, Operation, ProjectDetail } from "./types";
export function SafetyEvidence({
  detail,
  policies,
  canWrite,
  onSaved,
}: {
  detail: ProjectDetail;
  policies: any[];
  canWrite: boolean;
  onSaved: (result: Commit) => void;
}) {
  const { t } = useI18n(),
    [policyId, setPolicyId] = useState(""),
    [policy, setPolicy] = useState<any>(),
    [applicability, setApplicability] = useState("unknown"),
    [entityId, setEntityId] = useState(""),
    [result, setResult] = useState(""),
    [note, setNote] = useState(""),
    [name, setName] = useState(""),
    [error, setError] = useState<unknown>(),
    [busy, setBusy] = useState(false);
  useEffect(() => {
    setPolicy(undefined);
    setResult("");
    setApplicability("unknown");
    if (policyId)
      request<any>("/api/policies/" + policyId)
        .then(setPolicy)
        .catch(setError);
  }, [policyId]);
  const historical = detail.revision.id !== detail.branch.headRevisionId;
  const revision = policy?.revisions.find(
    (r: any) => r.id === policy.policy.activeRevisionId,
  );
  let check: any;
  try {
    const expr = revision?.document.jdm.nodes
      .find((n: any) => n.type === "expressionNode")
      ?.content.expressions.find((x: any) => x.key === "check");
    if (expr) check = JSON.parse(expr.value);
  } catch {}
  async function save(e: React.FormEvent) {
    e.preventDefault();
    if (!canWrite || historical || !policyId || !name.trim() || !note.trim())
      return;
    setBusy(true);
    setError(undefined);
    try {
      const recordId = id(),
        operations: Operation[] = [
          {
            type: "addAnnotation",
            annotation: {
              id: recordId,
              kind: "manual_assertion",
              text: note.trim(),
              displayName: name.trim(),
              identityVerified: false,
              createdAt: new Date().toISOString(),
              source: "manual_assertion",
            },
          },
        ];
      operations.push({
        type: "addAnnotation",
        annotation: {
          id: id(),
          kind: "policy_applicability",
          policyId,
          value: applicability,
          sourceRefs: [{ annotationId: recordId }],
        },
      });
      if (check?.kind === "manual" && entityId && result)
        operations.push({
          type: "addAnnotation",
          annotation: {
            id: id(),
            kind: "manual_evidence",
            entityId,
            requirement: check.requirement,
            passed: result === "pass",
            sourceRefs: [{ annotationId: recordId }],
          },
        });
      const saved = await request<Commit>(
        `/api/projects/${detail.project.id}/edits`,
        {
          method: "POST",
          projectId: detail.project.id,
          body: {
            requestId: id(),
            branchId: detail.branch.id,
            baseRevisionId: detail.revision.id,
            operations,
          },
        },
      );
      onSaved(saved);
      setNote("");
      setResult("");
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }
  return (
    <details className="manual-evidence">
      <summary>{t("addSafetyEvidence")}</summary>
      <p className="subtle">{t("manualEvidenceNote")}</p>
      {historical && (
        <p className="notice">
          {t("historicalEvidence")}{" "}
          <a
            href={`#/projects/${detail.project.id}/safety?revision=${detail.branch.headRevisionId}`}
          >
            {t("openLatest")} ↗
          </a>
        </p>
      )}
      <form onSubmit={save}>
        <label className="field-label">
          {t("ruleVersion")}
          <select
            required
            value={policyId}
            onChange={(e) => setPolicyId(e.target.value)}
          >
            <option value="">—</option>
            {policies
              .filter(
                (p) => p.projectId === detail.project.id && p.activeRevisionId,
              )
              .map((p) => (
                <option key={p.id} value={p.id}>
                  {p.title}
                </option>
              ))}
          </select>
        </label>
        <label className="field-label">
          {t("applicability")}
          <select
            value={applicability}
            onChange={(e) => setApplicability(e.target.value)}
          >
            {["unknown", "applicable", "not_applicable"].map((k) => (
              <option key={k} value={k}>
                {t(k)}
              </option>
            ))}
          </select>
        </label>
        {check?.kind === "manual" && (
          <>
            <p className="subtle">{check.requirement}</p>
            <label className="field-label">
              {t("objects")}
              <select
                value={entityId}
                onChange={(e) => setEntityId(e.target.value)}
              >
                <option value="">—</option>
                {detail.revision.document.entities.map((e) => (
                  <option value={e.id} key={e.id}>
                    {e.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field-label">
              {t("manualResult")}
              <select
                value={result}
                onChange={(e) => setResult(e.target.value)}
              >
                <option value="">{t("unknown")}</option>
                <option value="pass">{t("PASS")}</option>
                <option value="fail">{t("FAIL")}</option>
              </select>
            </label>
          </>
        )}
        <label className="field-label">
          {t("reviewName")}
          <input
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <label className="field-label">
          {t("evidenceRecord")}
          <textarea
            required
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
        </label>
        <ErrorNotice error={error} />
        <button
          className="primary"
          disabled={busy || !canWrite || historical || !policyId}
        >
          {t("saveEvidence")}
        </button>
      </form>
    </details>
  );
}
