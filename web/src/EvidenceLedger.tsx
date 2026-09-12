import { useState } from "react";
import { id, request } from "./api";
import { useI18n } from "./i18n";
import { ErrorNotice } from "./App";
import type { ProjectDetail } from "./types";
export function EvidenceLedger({
  detail,
  items,
  evaluations,
  canWrite,
  onChanged,
}: {
  detail: ProjectDetail;
  items: any[];
  evaluations: any[];
  canWrite: boolean;
  onChanged: () => void;
}) {
  const { t } = useI18n(),
    [selected, setSelected] = useState<Record<string, string>>({}),
    [reference, setReference] = useState<Record<string, string>>({}),
    [error, setError] = useState<unknown>();
  const originalFinding = (item: any) =>
    evaluations
      .find((evaluation) => evaluation.id === item.evaluationId)
      ?.document.findings.find((finding: any) => finding.id === item.findingId);
  async function fulfill(item: any) {
    try {
      const target = evaluations.find((e) => e.id === selected[item.id]);
      const finding = target?.document.findings.find(
        (f: any) =>
          f.entityId === item.document.entityId &&
          f.policyRevisionId === originalFinding(item)?.policyRevisionId,
      );
      if (!target || !finding || !reference[item.id]) return;
      await request(
        `/api/projects/${detail.project.id}/evidence-requests/${item.id}/fulfill`,
        {
          method: "POST",
          projectId: detail.project.id,
          body: {
            requestId: id(),
            evaluationId: target.id,
            findingId: finding.id,
            evidenceRefs: [{ annotationId: reference[item.id] }],
          },
        },
      );
      onChanged();
    } catch (e) {
      setError(e);
    }
  }
  return (
    <section className="evidence-ledger">
      <h2>{t("evidenceRequests")}</h2>
      <ErrorNotice error={error} />
      {items
        .filter((r) => r.document.status === "open")
        .map((item) => {
          const completed = items.find(
            (r) =>
              r.document.evidenceRequestId === item.id &&
              r.document.status === "fulfilled",
          );
          return (
            <details key={item.id}>
              <summary>
                {item.document.action} ·{" "}
                {t(completed ? "fulfilled" : "openRequest")}
              </summary>
              <p>{item.document.missingEvidence?.join(" · ")}</p>
              {completed ? (
                <p>
                  {t("version")}{" "}
                  {completed.document.sceneRevisionId.slice(0, 8)}
                </p>
              ) : (
                <>
                  <label className="field-label">
                    {t("newEvaluation")}
                    <select
                      value={selected[item.id] || ""}
                      onChange={(e) =>
                        setSelected((old) => ({
                          ...old,
                          [item.id]: e.target.value,
                        }))
                      }
                    >
                      <option value="">—</option>
                      {evaluations
                        .filter(
                          (e) =>
                            e.id !== item.evaluationId &&
                            e.sceneRevisionId === detail.revision.id &&
                            e.document.findings.some(
                              (f: any) =>
                                f.entityId === item.document.entityId &&
                                f.policyRevisionId ===
                                  originalFinding(item)?.policyRevisionId,
                            ),
                        )
                        .map((e) => (
                          <option value={e.id} key={e.id}>
                            {e.id.slice(0, 8)}
                          </option>
                        ))}
                    </select>
                  </label>
                  <label className="field-label">
                    {t("evidenceRecord")}
                    <select
                      value={reference[item.id] || ""}
                      onChange={(e) =>
                        setReference((old) => ({
                          ...old,
                          [item.id]: e.target.value,
                        }))
                      }
                    >
                      <option value="">—</option>
                      {detail.revision.document.annotations
                        .filter((a) => a.kind === "manual_assertion")
                        .map((a) => (
                          <option key={a.id} value={a.id}>
                            {typeof a.text === "string"
                              ? a.text.slice(0, 120)
                              : a.id}
                          </option>
                        ))}
                    </select>
                  </label>
                  <button
                    disabled={
                      !canWrite || !selected[item.id] || !reference[item.id]
                    }
                    onClick={() => fulfill(item)}
                  >
                    {t("fulfillRequest")}
                  </button>
                </>
              )}
            </details>
          );
        })}
      {!items.length && <p className="subtle">{t("noEvidenceRequests")}</p>}
    </section>
  );
}
