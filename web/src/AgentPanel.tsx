import { createElement, useEffect, useRef, useState } from "react";
import { ApiError, id, request } from "./api";
import { useI18n } from "./i18n";
import { ErrorNotice } from "./App";
import type { AgentRequest, Branch, Operation, Revision } from "./types";
type Proposal = {
  operations: Operation[];
  baseRevisionId: string;
  agentTurnId: string;
  policyDraft?: any;
};
export function AgentPanel({
  projectId,
  revision,
  branch,
  entityId,
  observationId,
  imageId,
  box,
  canWrite,
  onApply,
  policyId,
  policyRevisionId,
  onPolicyApply,
}: {
  projectId: string;
  revision: Revision;
  branch: Pick<Branch, "id">;
  entityId: string | null;
  observationId: string | null;
  imageId?: string | null;
  policyId?: string;
  policyRevisionId?: string;
  onPolicyApply?: (draft: any) => Promise<unknown>;
  box: number[] | null;
  canWrite: boolean;
  onApply: (
    ops: Operation[],
    context?: { baseRevisionId: string; agentTurnId?: string },
  ) => unknown;
}) {
  const { t, language } = useI18n(),
    chat = useRef<any>(null),
    context = useRef<any>(null),
    alive = useRef(true),
    [proposal, setProposal] = useState<Proposal | null>(null),
    [error, setError] = useState<unknown>(),
    [ready, setReady] = useState(false),
    [working, setWorking] = useState(false);
  context.current = {
    revision,
    branch,
    entityId,
    observationId,
    imageId,
    box,
    canWrite,
    language,
    policyId,
    policyRevisionId,
  };
  const conversation = useRef(id());
  useEffect(() => {
    alive.current = true;
    import("deep-chat")
      .then(() => customElements.whenDefined("deep-chat"))
      .then(() => {
        if (alive.current) setReady(true);
      })
      .catch(setError);
    return () => {
      alive.current = false;
    };
  }, []);
  useEffect(() => {
    if (!ready || !chat.current) return;
    request<{ items: any[] }>(`/api/projects/${projectId}/agent-turns`)
      .then((result) => {
        if (!alive.current) return;
        const turns = result.items.filter((turn) =>
          policyId
            ? turn.request.policyId === policyId
            : !turn.request.policyId,
        );
        chat.current.history = turns.flatMap((turn) => [
          { role: "user", text: turn.request.message },
          ...(turn.response?.message
            ? [{ role: "ai", text: turn.response.message }]
            : []),
        ]);
        const latest = turns.at(-1);
        if (latest) conversation.current = latest.conversationId;
        if (
          !latest?.appliedEditBatchId &&
          !latest?.appliedPolicyRevisionId &&
          latest?.response?.kind === "proposal"
        )
          setProposal({
            operations: latest.response.operations || [],
            baseRevisionId: latest.baseRevisionId,
            agentTurnId: latest.id,
            policyDraft:
              latest.response.proposalType === "policy"
                ? latest.response
                : undefined,
          });
      })
      .catch(setError);
  }, [ready, projectId, policyId]);
  useEffect(() => {
    if (!ready || !chat.current) return;
    chat.current.textInput = { placeholder: { text: t("ask") } };
    chat.current.messageStyles = {
      default: {
        shared: {
          bubble: {
            backgroundColor: "#ecefe8",
            color: "#273b32",
            borderRadius: "8px",
          },
        },
        user: { bubble: { backgroundColor: "#29493b", color: "#f9f8f2" } },
      },
    };
    chat.current.connect = {
      handler: async (body: any, signals: any) => {
        setError(undefined);
        setWorking(true);
        try {
          const ctx = context.current;
          if (!ctx.canWrite)
            throw new ApiError(403, "owner_capability_required");
          const message = body.messages
            .filter((m: any) => m.role === "user")
            .at(-1)?.text;
          const input: AgentRequest = {
            requestId: id(),
            conversationId: conversation.current,
            branchId: ctx.branch.id,
            baseRevisionId: ctx.revision.id,
            message,
            entityId: ctx.entityId,
            observationId: ctx.observationId,
            box: ctx.box,
            language: ctx.language,
            ...(ctx.imageId ? { imageId: ctx.imageId } : {}),
            ...(ctx.policyId && ctx.policyRevisionId
              ? {
                  policyId: ctx.policyId,
                  policyRevisionId: ctx.policyRevisionId,
                }
              : {}),
          };
          let turn = await request<any>(
            `/api/projects/${projectId}/agent-turns`,
            { method: "POST", projectId, body: input },
          );
          while (["pending", "running"].includes(turn.status)) {
            await new Promise((resolve) => setTimeout(resolve, 1500));
            if (!alive.current) return;
            const list = await request<{ items: any[] }>(
              `/api/projects/${projectId}/agent-turns`,
            );
            const updated = list.items.find((item) => item.id === turn.id);
            if (updated) turn = updated;
          }
          if (!alive.current) return;
          const response = turn.response;
          if (response?.kind === "proposal")
            setProposal({
              operations: response.operations || [],
              baseRevisionId: turn.baseRevisionId,
              agentTurnId: turn.id,
              policyDraft:
                response.proposalType === "policy" ? response : undefined,
            });
          else setProposal(null);
          if (!response?.message)
            throw new Error(response?.code || "agent_response_missing");
          await signals.onResponse({ text: response.message });
        } catch (e) {
          if (alive.current) {
            setError(e);
            await signals.onResponse({ error: t("error") });
          }
        } finally {
          if (alive.current) setWorking(false);
        }
      },
    };
  }, [ready, projectId, language]);
  async function apply() {
    if (!proposal) return;
    setError(undefined);
    if (proposal.policyDraft && onPolicyApply) {
      try {
        const saved = await onPolicyApply({
          ...proposal.policyDraft,
          agentTurnId: proposal.agentTurnId,
        });
        if (saved) setProposal(null);
      } catch (e) {
        setError(e);
      }
      return;
    }
    if (proposal.baseRevisionId !== revision.id) {
      setError(new ApiError(409, "revision_conflict"));
      return;
    }
    const result = await onApply(proposal.operations, {
      baseRevisionId: proposal.baseRevisionId,
      agentTurnId: proposal.agentTurnId,
    });
    if (result) setProposal(null);
  }
  return (
    <div className="agent-panel">
      <p className="subtle">{t("agentCopy")}</p>
      <div className="context-chips">
        <span>{revision.id.slice(0, 8)}</span>
        {entityId && (
          <span>
            {revision.document.entities.find((e) => e.id === entityId)?.label}
          </span>
        )}
        {box && <span>{t("selectedBox")}</span>}
      </div>
      {createElement("deep-chat", { ref: chat, className: "deep-chat" })}
      {working && <p role="status">{t("running")}</p>}
      <ErrorNotice error={error} />
      {proposal && (
        <div className="proposal">
          <h3>{t("proposal")}</h3>
          {proposal.policyDraft && (
            <pre>{JSON.stringify(proposal.policyDraft, null, 2)}</pre>
          )}
          <ul>
            {proposal.operations.map((op, i) => (
              <li key={i}>
                <strong>
                  {revision.document.entities.find((e) => e.id === op.entityId)
                    ?.label || String(op.type)}
                </strong>
                <span> · {String(op.type)}</span>
                <pre>{JSON.stringify(op, null, 2)}</pre>
              </li>
            ))}
          </ul>
          <button className="primary" onClick={apply}>
            {t("apply")}
          </button>
        </div>
      )}
    </div>
  );
}
