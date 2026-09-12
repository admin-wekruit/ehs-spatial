import { useEffect, useState } from "react";
import { DecisionGraph, type DecisionGraphType } from "@gorules/jdm-editor";
import "@gorules/jdm-editor/dist/style.css";
import { EvidenceLedger } from "./EvidenceLedger";
import { AgentPanel } from "./AgentPanel";
import { SafetyEvidence } from "./SafetyEvidence";
import { id, owner, ownedIds, request } from "./api";
import { useI18n } from "./i18n";
import { ErrorNotice } from "./App";
import type { Project, ProjectDetail } from "./types";
const emptyGraph: DecisionGraphType = { nodes: [], edges: [] };
export default function PolicyPage({
  projectId,
  revisionId,
  policyId,
}: {
  projectId?: string;
  revisionId?: string;
  policyId?: string;
}) {
  const { t, language } = useI18n(),
    [policies, setPolicies] = useState<any[]>([]),
    [projects, setProjects] = useState<Project[]>([]),
    [templates, setTemplates] = useState<any[]>([]),
    [owned, setOwned] = useState<Set<string>>(new Set()),
    [error, setError] = useState<unknown>(),
    [editing, setEditing] = useState(!!policyId),
    [detail, setDetail] = useState<any>(),
    [title, setTitle] = useState(""),
    [pid, setPid] = useState(projectId || ""),
    [source, setSource] = useState<any>({
      title: "",
      publisher: "",
      url: "",
      language: "en",
      jurisdiction: "US federal",
      text: "",
      clauses: [],
    }),
    [jdm, setJdm] = useState<DecisionGraphType>(emptyGraph),
    [tests, setTests] = useState("[]"),
    [limitations, setLimitations] = useState(""),
    [saving, setSaving] = useState(false),
    [status, setStatus] = useState(""),
    [selectedPolicies, setSelectedPolicies] = useState<string[]>([]),
    [assessments, setAssessments] = useState<any[]>([]),
    [reviews, setReviews] = useState<any[]>([]),
    [evidenceRequests, setEvidenceRequests] = useState<any[]>([]),
    [projectDetail, setProjectDetail] = useState<ProjectDetail>(),
    [reviewReason, setReviewReason] = useState<Record<string, string>>({}),
    [reviewName, setReviewName] = useState(""),
    [testResults, setTestResults] = useState<any[]>([]);
  async function reload() {
    try {
      const [p, projects] = await Promise.all([
        request<{ items: any[] }>("/api/policies"),
        request<{ items: Project[] }>("/api/projects"),
      ]);
      setPolicies(p.items);
      setProjects(projects.items);
      setOwned(await ownedIds());
      if (projectId) {
        const [d, e, r, evidence] = await Promise.all([
          request<ProjectDetail>("/api/projects/" + projectId),
          request<{ items: any[] }>(`/api/projects/${projectId}/evaluations`),
          request<{ items: any[] }>(`/api/projects/${projectId}/reviews`),
          request<{ items: any[] }>(
            `/api/projects/${projectId}/evidence-requests`,
          ),
        ]);
        if (revisionId) {
          const revision = await request<any>("/api/revisions/" + revisionId);
          d.revision = revision;
          d.branch =
            d.branches.find((b) => b.id === revision.branchId) || d.branch;
        }
        setProjectDetail(d);
        setAssessments(e.items);
        setReviews(r.items);
        setEvidenceRequests(evidence.items);
      }
    } catch (e) {
      setError(e);
    }
  }
  useEffect(() => {
    void reload();
    request<{ items: any[] }>("/api/policy-templates")
      .then((r) => setTemplates(r.items))
      .catch(setError);
  }, [projectId, revisionId]);
  useEffect(() => {
    if (pid && !projectId)
      request<ProjectDetail>("/api/projects/" + pid)
        .then(setProjectDetail)
        .catch(setError);
  }, [pid, projectId]);
  useEffect(() => {
    if (policyId && policyId !== "new")
      request<any>("/api/policies/" + policyId)
        .then((value) => {
          setDetail(value);
          setEditing(true);
          setTitle(value.policy.title);
          setPid(value.policy.projectId);
          const latest = value.revisions.at(-1) || value.revisions[0];
          if (latest) {
            setJdm(latest.document.jdm || emptyGraph);
            setTests(JSON.stringify(latest.document.tests || [], null, 2));
            setLimitations(
              typeof latest.document.limitations === "string"
                ? latest.document.limitations
                : (latest.document.limitations || []).join("\n"),
            );
          }
          if (value.sources[0]) setSource(value.sources[0].document);
        })
        .catch(setError);
  }, [policyId]);
  async function save(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(undefined);
    try {
      const body = {
        requestId: id(),
        jdm,
        tests: JSON.parse(tests),
        limitations: limitations
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean),
      };
      const result = detail
        ? await request<any>(
            `/api/projects/${pid}/policies/${detail.policy.id}/revisions`,
            {
              method: "POST",
              projectId: pid,
              body: {
                ...body,
                basePolicyRevisionId: detail.revisions.at(-1)?.id,
              },
            },
          )
        : await request<any>(`/api/projects/${pid}/policies`, {
            method: "POST",
            projectId: pid,
            body: {
              ...body,
              title,
              source: {
                ...source,
                clauses: source.clauses?.length
                  ? source.clauses
                  : [{ locator: "source", text: source.text }],
              },
            },
          });
      const policy = result.policy || detail.policy;
      const updated = await request<any>("/api/policies/" + policy.id);
      setDetail(updated);
      setStatus("saved");
      location.hash = "/policies/" + policy.id + "/edit";
    } catch (e) {
      setError(e);
    } finally {
      setSaving(false);
    }
  }
  async function activate() {
    if (!detail) return;
    setError(undefined);
    try {
      const revision = detail.revisions.at(-1);
      await request(
        `/api/projects/${pid}/policies/${detail.policy.id}/activate`,
        {
          method: "POST",
          projectId: pid,
          body: {
            requestId: id(),
            policyRevisionId: revision.id,
            expectedActiveRevisionId: detail.policy.activeRevisionId || null,
          },
        },
      );
      setStatus("publishedSnapshot");
      await reload();
      setDetail(await request("/api/policies/" + detail.policy.id));
    } catch (e) {
      setError(e);
    }
  }
  async function evaluate() {
    if (!projectId || !projectDetail) return;
    setSaving(true);
    setError(undefined);
    try {
      if (!selectedPolicies.length) throw new Error("policyMissing");
      await request(`/api/projects/${projectId}/evaluations`, {
        method: "POST",
        projectId,
        body: {
          requestId: id(),
          sceneRevisionId: revisionId || projectDetail.revision.id,
          policyRevisionIds: selectedPolicies,
          context:
            projectDetail.branch.kind === "planning" ? "planning" : "observed",
        },
      });
      await reload();
    } catch (e) {
      setError(e);
    } finally {
      setSaving(false);
    }
  }
  async function review(evaluation: any, finding: any, decision: string) {
    if (!projectId) return;
    try {
      await request(
        `/api/projects/${projectId}/findings/${finding.id}/reviews`,
        {
          method: "POST",
          projectId,
          body: {
            requestId: id(),
            evaluationId: evaluation.id,
            decision,
            reason: reviewReason[finding.id] || "",
            displayName: reviewName.trim(),
            evidenceRefs: [],
          },
        },
      );
      await reload();
    } catch (e) {
      setError(e);
    }
  }
  async function runTests() {
    try {
      setError(undefined);
      const result = await request<{ items: any[] }>("/api/policies/test", {
        method: "POST",
        body: { jdm, tests: JSON.parse(tests) },
      });
      setTestResults(result.items);
    } catch (e) {
      setError(e);
    }
  }
  async function importPDF(file: File) {
    try {
      const body = new FormData();
      body.set("file", file);
      const result = await request<{
        pages: { page: number; text: string }[];
        status: string;
      }>(`/api/projects/${pid}/policy-source-text`, {
        method: "POST",
        projectId: pid,
        body,
      });
      if (result.status === "source_text_required")
        throw new Error("policy_source_text_required");
      setSource((old: any) => ({
        ...old,
        text: result.pages.map((p) => p.text).join("\n\n"),
        clauses: result.pages
          .filter((p) => p.text.trim())
          .map((p) => ({ locator: "page " + p.page, text: p.text })),
      }));
    } catch (e) {
      setError(e);
    }
  }
  function useTemplate(template: any) {
    setTitle(template.title || "");
    setSource(template.source || source);
    setJdm(template.jdm || emptyGraph);
    setTests(JSON.stringify(template.tests || [], null, 2));
    setLimitations((template.limitations || []).join("\n"));
    setEditing(true);
  }
  if (projectId)
    return (
      <section className="page-width safety-page">
        <div className="title-row">
          <div>
            <p className="eyebrow">SCENE EVIDENCE × POLICY</p>
            <h1>{t("safety")}</h1>
          </div>
          <a className="button" href={`#/projects/${projectId}/workbench`}>
            {t("workbench")} ↗
          </a>
        </div>
        <p className="subtle">
          {projectDetail?.project.title} · {t("version")}{" "}
          {(revisionId || projectDetail?.revision.id)?.slice(0, 8)}
        </p>
        <ErrorNotice error={error} />
        <div className="safety-layout">
          <aside className="policy-picker">
            <h2>{t("ruleVersion")}</h2>
            {policies
              .filter((p) => p.activeRevisionId && p.projectId === projectId)
              .map((p) => (
                <label key={p.id}>
                  <input
                    type="checkbox"
                    checked={selectedPolicies.includes(p.activeRevisionId)}
                    onChange={(e) =>
                      setSelectedPolicies((values) =>
                        e.target.checked
                          ? [...values, p.activeRevisionId]
                          : values.filter((v) => v !== p.activeRevisionId),
                      )
                    }
                  />
                  <span>
                    <strong>{p.title}</strong>
                    <small>{p.activeRevisionId.slice(0, 8)}</small>
                  </span>
                </label>
              ))}
            <button
              className="primary wide"
              disabled={saving || !owned.has(projectId)}
              onClick={evaluate}
            >
              {t(saving ? "loading" : "evaluate")}
            </button>
            <a href="#/policies">{t("policies")} ↗</a>
            {projectDetail && (
              <>
                <SafetyEvidence
                  detail={projectDetail}
                  policies={policies}
                  canWrite={owned.has(projectId)}
                  onSaved={(result) => {
                    setProjectDetail({
                      ...projectDetail,
                      revision: result.revision,
                      branch: {
                        ...projectDetail.branch,
                        headRevisionId: result.revision.id,
                      },
                    });
                    location.hash = `/projects/${projectId}/safety?revision=${result.revision.id}`;
                  }}
                />
                <details>
                  <summary>{t("agent")}</summary>
                  <AgentPanel
                    projectId={projectId}
                    revision={projectDetail.revision}
                    branch={projectDetail.branch}
                    entityId={null}
                    observationId={null}
                    box={null}
                    canWrite={owned.has(projectId)}
                    onApply={async (operations, context) => {
                      const result = await request<any>(
                        `/api/projects/${projectId}/edits`,
                        {
                          method: "POST",
                          projectId,
                          body: {
                            requestId: id(),
                            branchId: projectDetail.branch.id,
                            baseRevisionId: projectDetail.revision.id,
                            operations,
                            ...context,
                          },
                        },
                      );
                      setProjectDetail({
                        ...projectDetail,
                        revision: result.revision,
                        branch: {
                          ...projectDetail.branch,
                          headRevisionId: result.revision.id,
                        },
                      });
                      return result;
                    }}
                  />
                </details>
              </>
            )}
          </aside>
          <div className="assessment-list">
            {projectDetail && (
              <EvidenceLedger
                detail={projectDetail}
                items={evidenceRequests}
                evaluations={assessments}
                canWrite={owned.has(projectId)}
                onChanged={reload}
              />
            )}
            {assessments.length ? (
              assessments.map((evaluation) => (
                <section key={evaluation.id} className="evaluation">
                  <div className="list-toolbar">
                    <h2>{evaluation.title || evaluation.id.slice(0, 8)}</h2>
                    <span>{evaluation.sceneRevisionId?.slice(0, 8)}</span>
                  </div>
                  {(evaluation.document.findings || []).map((finding: any) => (
                    <article className="finding" key={finding.id}>
                      <div className="finding-heading">
                        <h3>
                          {finding.policyTitle || finding.policyRevisionId}
                        </h3>
                        <span
                          className={
                            "badge " +
                            (finding.machineResult || "").toLowerCase()
                          }
                        >
                          {t(
                            finding.applicability === "not_applicable"
                              ? "NOT_APPLICABLE"
                              : finding.applicability === "unknown"
                                ? "APPLICABILITY_UNKNOWN"
                                : finding.machineResult ||
                                  "INSUFFICIENT_EVIDENCE",
                          )}
                        </span>
                      </div>
                      <p>{finding.summary || finding.reason}</p>
                      {(finding.entityId ? [finding.entityId] : []).map(
                        (entityId: string) => (
                          <a
                            key={entityId}
                            href={`#/projects/${projectId}/workbench?revision=${evaluation.sceneRevisionId}&object=${entityId}`}
                          >
                            {t("openWorkbench")} ↗
                          </a>
                        ),
                      )}
                      <details>
                        <summary>{t("evidence")}</summary>
                        <pre>
                          {JSON.stringify(
                            finding.evidence || finding.facts || finding,
                            null,
                            2,
                          )}
                        </pre>
                      </details>
                      <details>
                        <summary>{t("review")}</summary>
                        <label className="field-label">
                          {t("reviewName")}
                          <input
                            required
                            value={reviewName}
                            onChange={(e) => setReviewName(e.target.value)}
                          />
                        </label>
                        <label className="field-label">
                          {t("reviewReason")}
                          <textarea
                            required
                            value={reviewReason[finding.id] || ""}
                            onChange={(e) =>
                              setReviewReason((old) => ({
                                ...old,
                                [finding.id]: e.target.value,
                              }))
                            }
                          />
                        </label>
                        {reviews
                          .filter((r) => r.findingId === finding.id)
                          .map((r) => (
                            <p key={r.id}>
                              <strong>{r.document.displayName}</strong> ·{" "}
                              {t("review_" + r.document.decision)}
                              <br />
                              {r.document.reason}
                            </p>
                          ))}
                        <button
                          disabled={
                            !owned.has(projectId) ||
                            !reviewReason[finding.id]?.trim()
                          }
                          onClick={async () => {
                            try {
                              await request(
                                `/api/projects/${projectId}/findings/${finding.id}/evidence-requests`,
                                {
                                  method: "POST",
                                  projectId,
                                  body: {
                                    requestId: id(),
                                    evaluationId: evaluation.id,
                                    action: reviewReason[finding.id].trim(),
                                  },
                                },
                              );
                              await reload();
                            } catch (e) {
                              setError(e);
                            }
                          }}
                        >
                          {t("requestEvidence")}
                        </button>
                        {["confirmed", "rejected", "needs_evidence"].map(
                          (decision) => (
                            <button
                              key={decision}
                              disabled={
                                !reviewName.trim() ||
                                !reviewReason[finding.id]?.trim() ||
                                !owned.has(projectId)
                              }
                              onClick={() =>
                                review(evaluation, finding, decision)
                              }
                            >
                              {t("review_" + decision)}
                            </button>
                          ),
                        )}
                      </details>
                    </article>
                  ))}
                </section>
              ))
            ) : (
              <div className="empty-state">{t("noAssessments")}</div>
            )}
          </div>
        </div>
      </section>
    );
  return (
    <section className={"policy-page " + (editing ? "" : "page-width")}>
      <div className="page-width">
        <p className="eyebrow">RULES / EVIDENCE / REVIEW</p>
        <div className="title-row">
          <h1>{t("policyLibrary")}</h1>
          <button
            className="primary"
            onClick={() => {
              setEditing(true);
              setDetail(undefined);
              location.hash = "/policies/new/edit";
            }}
          >
            {t("newPolicy")} ＋
          </button>
        </div>
        <p className="lede">{t("policyIntro")}</p>
        <ErrorNotice error={error} />
      </div>
      {!editing ? (
        <>
          <div className="policy-list page-width">
            {policies.map((p) => (
              <a key={p.id} href={`#/policies/${p.id}/edit`}>
                <h2>{p.title}</h2>
                <span className="badge">
                  {t(p.activeRevisionId ? "publishedSnapshot" : "saveDraft")}
                </span>
                <span>↗</span>
              </a>
            ))}
            {!policies.length && (
              <p className="empty-copy">{t("noPolicies")}</p>
            )}
          </div>
          <div className="page-width template-list">
            {templates.map((template) => (
              <button
                key={template.id || template.title}
                onClick={() => useTemplate(template)}
              >
                <strong>{template.title}</strong>
                <span>{template.source?.publisher} ↗</span>
              </button>
            ))}
          </div>
        </>
      ) : (
        <form onSubmit={save}>
          <div className="policy-toolbar page-width">
            <label>
              {t("projects")}
              <select
                required
                value={pid}
                onChange={(e) => setPid(e.target.value)}
                disabled={!!detail}
              >
                <option value="">—</option>
                {projects
                  .filter((p) => owned.has(p.id) || p.id === pid)
                  .map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.title}
                    </option>
                  ))}
              </select>
            </label>
            <label>
              {t("policyTitle")}
              <input
                required
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                disabled={!!detail}
              />
            </label>
            <span role="status">{t(status)}</span>
            <button className="primary" disabled={saving || !pid}>
              {t("saveDraft")}
            </button>
            {detail && (
              <button type="button" onClick={activate}>
                {t("publishPolicy")}
              </button>
            )}
          </div>
          <div className="policy-editor-layout">
            <aside className="source-editor">
              <h2>{t("source")}</h2>
              {!detail && (
                <label className="field-label">
                  {t("importPDF")}
                  <input
                    type="file"
                    accept="application/pdf"
                    disabled={!pid}
                    onChange={(e) => {
                      if (e.target.files?.[0])
                        void importPDF(e.target.files[0]);
                    }}
                  />
                </label>
              )}
              {(["title", "publisher", "url", "jurisdiction"] as const).map(
                (key) => (
                  <label className="field-label" key={key}>
                    {t(
                      key === "url"
                        ? "sourceURL"
                        : key === "jurisdiction"
                          ? "jurisdiction"
                          : key === "title"
                            ? "title"
                            : "source",
                    )}
                    <input
                      value={source[key] || ""}
                      onChange={(e) =>
                        setSource((old: any) => ({
                          ...old,
                          [key]: e.target.value,
                        }))
                      }
                      readOnly={!!detail}
                    />
                  </label>
                ),
              )}
              <label className="field-label">
                {t("sourceText")}
                <textarea
                  rows={14}
                  value={source.text || ""}
                  onChange={(e) =>
                    setSource((old: any) => ({ ...old, text: e.target.value }))
                  }
                  readOnly={!!detail}
                />
              </label>
            </aside>
            <section className="jdm-panel">
              <h2>{t("decisionTable")}</h2>
              <div className="jdm-editor">
                <DecisionGraph
                  value={jdm}
                  onChange={(value) => setJdm(value)}
                  disabled={saving}
                />
              </div>
            </section>
            <aside className="rule-tests">
              <h2>{language === "zh" ? "规则测试" : "Rule tests"}</h2>
              <button type="button" onClick={runTests}>
                {t("runTests")}
              </button>
              {testResults.map((r, i) => (
                <p key={i} className={"badge " + (r.passed ? "pass" : "fail")}>
                  {r.name} · {t(r.passed ? "PASS" : "FAIL")}
                </p>
              ))}
              <label className="field-label">
                JSON
                <textarea
                  rows={16}
                  value={tests}
                  onChange={(e) => setTests(e.target.value)}
                  spellCheck={false}
                />
              </label>
              <label className="field-label">
                {language === "zh" ? "适用限制" : "Limitations"}
                <textarea
                  rows={5}
                  value={limitations}
                  onChange={(e) => setLimitations(e.target.value)}
                />
              </label>
            </aside>
          </div>
          {detail && projectDetail && (
            <section className="page-width policy-agent">
              <h2>{t("agent")}</h2>
              <AgentPanel
                projectId={pid}
                revision={projectDetail.revision}
                branch={projectDetail.branch}
                entityId={null}
                observationId={null}
                box={null}
                canWrite={owned.has(pid)}
                policyId={detail.policy.id}
                policyRevisionId={
                  detail.policy.draftRevisionId || detail.revisions.at(-1)?.id
                }
                onApply={() => {}}
                onPolicyApply={async (draft) => {
                  const result = await request<any>(
                    `/api/projects/${pid}/policies/${detail.policy.id}/revisions`,
                    {
                      method: "POST",
                      projectId: pid,
                      body: {
                        requestId: id(),
                        basePolicyRevisionId: draft.basePolicyRevisionId,
                        jdm: draft.jdm,
                        tests: draft.tests,
                        limitations: draft.limitations,
                        sourceRefs: draft.sourceRefs,
                        agentTurnId: draft.agentTurnId,
                      },
                    },
                  );
                  setDetail(
                    await request<any>("/api/policies/" + detail.policy.id),
                  );
                  setJdm(result.document.jdm);
                  setTests(JSON.stringify(result.document.tests, null, 2));
                  return result;
                }}
              />
            </section>
          )}
        </form>
      )}
    </section>
  );
}
