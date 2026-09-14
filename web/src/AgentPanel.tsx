import { createElement, useEffect, useRef, useState } from "react";
import { ApiError, feedbackSession, id, request, type FeedbackInput, type IdentitySuggestion } from "./api";
import { useI18n } from "./i18n";
import { observationsFor } from "./core";
import { ErrorNotice } from "./App";
import type { AgentRequest, AgentTurn, Branch, Operation, Revision } from "./types";
type Proposal = {
  operations: Operation[];
  baseRevisionId: string;
  agentTurnId: string;
  policyDraft?: any;
};
type AgentPanelProps = {
  feedbackPublicationId?: string;
  identitySuggestion?: IdentitySuggestion;
  identityEntityIds?: [string, string];
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
};
type FeedbackTurn = {
  id: string; requestId: string; conversationId: string; publicationId: string; revisionId: string; entityId: string;
  message: string; language: "zh" | "en"; status: "saved" | "succeeded" | "failed" | "outcome_unknown";
  identitySuggestion?: IdentitySuggestion | null; assistantMessage: string | null; errorCode: string | null; createdAt: string;
};
function feedbackReply(turn: FeedbackTurn, t: (key: string) => string) {
  return turn.assistantMessage || t(turn.identitySuggestion && turn.status === "saved" ? "identitySuggestionSaved" : turn.status === "outcome_unknown" ? "feedbackOutcomeUnknown" :
    turn.status === "failed" ? "feedbackReplyFailed" : turn.errorCode === "feedback_budget_exceeded" ? "feedbackBudgetExceeded" : "feedbackSavedNoAgent");
}
function feedbackEvidence({ revision, entityId, imageId, observationId }: Pick<AgentPanelProps, "revision" | "entityId" | "imageId" | "observationId">) {
  const entity = revision.document.entities.find((item) => item.id === entityId);
  const observations = entity ? observationsFor(revision.document, entity) : [];
  const observation = observations.find((item) => item.id === observationId && (!imageId || item.imageId === imageId));
  const image = observations.some((item) => item.imageId === imageId) ? imageId : observation?.imageId;
  return { ...(image ? { imageId: image } : {}), ...(observation ? { observationId: observation.id } : {}) };
}

function conversationScope(projectId: string, branchId: string, revisionId: string, entityId?: string | null, policyId?: string | null, identityEntityIds?: [string,string] | null) {
  return JSON.stringify(policyId ? ["policy", projectId, policyId] : ["scene", projectId, branchId, revisionId, entityId || null, identityEntityIds ? [...identityEntityIds].sort() : null]);
}

function turnScope(turn: AgentTurn) {
  return conversationScope(turn.projectId, turn.branchId, turn.baseRevisionId, turn.request.entityId, turn.request.policyId, turn.request.identityEntityIds);
}

async function agentHistory(projectId: string, signal: AbortSignal) {
  const turns: AgentTurn[] = [];
  let after = 0;
  while (true) {
    const page = await request<{ items: AgentTurn[] }>(`/api/projects/${projectId}/agent-turns?afterSequence=${after}`, { signal });
    turns.push(...page.items);
    if (page.items.length < 500) return turns;
    const next = page.items.at(-1)!.sequence;
    if (next <= after) throw new Error("invalid_agent_history_sequence");
    after = next;
  }
}

export function AgentPanel(props: AgentPanelProps) {
  const scope = props.feedbackPublicationId ? JSON.stringify(["feedback", props.feedbackPublicationId, props.entityId]) :
    conversationScope(props.projectId, props.branch.id, props.revision.id, props.entityId, props.policyId, props.identityEntityIds);
  return <AgentConversation key={scope} {...props} />;
}

function AgentConversation({
  projectId, revision, branch, entityId, observationId, imageId, box,
  canWrite, onApply, policyId, policyRevisionId, onPolicyApply, feedbackPublicationId, identitySuggestion, identityEntityIds,
}: AgentPanelProps) {
  const { t, language } = useI18n(),
    chat = useRef<any>(null),
    context = useRef<any>(null),
    alive = useRef(true),
    identitySubmission = useRef<IdentitySuggestion | null>(null),
    [proposal, setProposal] = useState<Proposal | null>(null),
    [error, setError] = useState<unknown>(),
    [ready, setReady] = useState(false),
    [historyReady, setHistoryReady] = useState(false),
    [feedbackRetry, setFeedbackRetry] = useState<FeedbackInput | null>(null),
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
    historyReady,
    feedbackPublicationId,
    identityEntityIds,
  };
  const conversation = useRef<string>(id());
  const feedbackRoute = feedbackPublicationId && entityId ? `/api/publications/${feedbackPublicationId}/entities/${encodeURIComponent(entityId)}/feedback` : null;
  async function publicHistory(signal?: AbortSignal) {
    if (!feedbackPublicationId || !entityId || !feedbackRoute) return;
    const session = await feedbackSession(feedbackPublicationId, entityId);
    const result = await request<{ items: FeedbackTurn[] }>(`${feedbackRoute}?conversationId=${session.conversationId}`, { feedbackCapability: session.capability, signal });
    if (!alive.current) return;
    conversation.current = session.conversationId;
    const pending = session.pending && !result.items.some((turn) => turn.requestId === session.pending!.requestId) ? session.pending : null;
    if (session.pending && !pending) await feedbackSession(feedbackPublicationId, entityId, null);
    if (!alive.current) return;
    chat.current.history = [...result.items.flatMap((turn) => [
      { role: "user", text: turn.message }, { role: "ai", text: feedbackReply(turn, t) },
    ]), ...(pending ? [{ role: "user", text: pending.message }] : [])];
    setFeedbackRetry(pending);
  }
  async function loadPublicHistory(signal?: AbortSignal) {
    setWorking(true); setError(undefined);
    try {
      await publicHistory(signal);
      if (alive.current) setHistoryReady(true);
    } catch (error) {
      if (error instanceof TypeError) console.error("[public-feedback history]", error.name, error.message);
      if (alive.current && !(error instanceof Error && error.name === "AbortError")) setError(error);
    } finally { if (alive.current) setWorking(false); }
  }
  async function sendFeedback(input: FeedbackInput, signals?: { onResponse: (response: { text: string }) => Promise<unknown> }) {
    if (!feedbackPublicationId || !entityId || !feedbackRoute) return;
    setWorking(true); setError(undefined);
    try {
      const session = await feedbackSession(feedbackPublicationId, entityId, input);
      const turn = await request<FeedbackTurn>(feedbackRoute, { method: "POST", body: input, feedbackCapability: session.capability });
      await feedbackSession(feedbackPublicationId, entityId, null);
      if (!alive.current) return;
      setFeedbackRetry(null);
      if (signals) await signals.onResponse({ text: feedbackReply(turn, t) });
      else await publicHistory();
    } catch (error) {
      if (error instanceof TypeError) console.error("[public-feedback send]", error.name, error.message);
      if (error instanceof ApiError && error.status < 500) await feedbackSession(feedbackPublicationId, entityId, null);
      if (alive.current) {
        setFeedbackRetry(error instanceof ApiError && error.status < 500 ? null : input);
        setError(error);
        if (signals) await signals.onResponse({ text: t(error instanceof ApiError && error.status < 500 ? "feedbackSendRejected" : "feedbackSendUnconfirmed") });
      }
    } finally { if (alive.current) setWorking(false); }
  }
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
    const abort = new AbortController();
    if (feedbackPublicationId) {
      void loadPublicHistory(abort.signal);
      return () => abort.abort();
    }
    const scope = conversationScope(projectId, branch.id, revision.id, entityId, policyId, identityEntityIds);
    agentHistory(projectId, abort.signal)
      .then((history) => {
        if (!alive.current) return;
        const turns = history.filter((turn) => turnScope(turn) === scope);
        chat.current.history = turns.flatMap((turn) => [
          { role: "user", text: turn.request.message },
          ...(turn.response?.message
            ? [{ role: "ai", text: turn.response.message }]
            : []),
        ]);
        const latest = turns.at(-1);
        // Old UI versions mixed objects in a conversation. Preserve matching
        // history, but never send that mixed transcript back to the model.
        if (latest && !history.some((turn) => turn.conversationId === latest.conversationId && turnScope(turn) !== scope))
          conversation.current = latest.conversationId;
        setProposal(null);
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
        setHistoryReady(true);
      })
      .catch((error) => { if (alive.current && error.name !== "AbortError") setError(error); });
    return () => abort.abort();
  }, [ready, projectId, policyId, feedbackPublicationId]);
  useEffect(() => {
    if (!ready || !chat.current) return;
    chat.current.textInput = { disabled: !historyReady || !!feedbackRetry || (feedbackPublicationId ? !entityId : !canWrite), placeholder: { text: t(feedbackPublicationId ? "feedbackAsk" : "ask") } };
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
          if (!ctx.historyReady) throw new Error(t("loading"));
          const message = body.messages
            .filter((m: any) => m.role === "user")
            .at(-1)?.text;
          if (ctx.feedbackPublicationId) {
            if (!ctx.entityId || typeof message !== "string" || !message.trim() || message.length > 8000) throw new Error(t("feedbackMessageInvalid"));
            await sendFeedback({ requestId: id(), conversationId: conversation.current, message: message.trim(), language: ctx.language,
              ...feedbackEvidence(ctx),
              ...(identitySubmission.current ? { identitySuggestion: identitySubmission.current } : {}),
            }, signals);
            identitySubmission.current = null;
            return;
          }
          if (!ctx.canWrite) throw new ApiError(403, "owner_capability_required");
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
            ...(ctx.identityEntityIds ? { identityEntityIds: ctx.identityEntityIds } : {}),
            ...(ctx.policyId && ctx.policyRevisionId
              ? {
                  policyId: ctx.policyId,
                  policyRevisionId: ctx.policyRevisionId,
                }
              : {}),
          };
          let turn = await request<AgentTurn>(
            `/api/projects/${projectId}/agent-turns`,
            { method: "POST", projectId, body: input },
          );
          while (["pending", "running"].includes(turn.status)) {
            await new Promise((resolve) => setTimeout(resolve, 1500));
            if (!alive.current) return;
            const list = await request<{ items: AgentTurn[] }>(
              `/api/projects/${projectId}/agent-turns?conversationId=${turn.conversationId}&afterSequence=${turn.sequence - 1}`,
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
            throw new Error(typeof response?.code === "string" ? response.code : "agent_response_missing");
          await signals.onResponse({ text: response.message });
        } catch (e) {
          if (alive.current) {
            setError(e);
            await signals.onResponse({ error: t("error") });
          }
        } finally {
          identitySubmission.current = null;
          if (alive.current) setWorking(false);
        }
      },
    };
  }, [ready, historyReady, canWrite, feedbackRetry, feedbackPublicationId, projectId, language]);
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
      <p className="subtle">{t(feedbackPublicationId ? "feedbackCopy" : "agentCopy")}</p>
      <div className="context-chips">
        <span>{revision.id.slice(0, 8)}</span>
        {entityId && (
          <span>
            {revision.document.entities.find((e) => e.id === entityId)?.label}
          </span>
        )}
        {box && <span>{t("selectedBox")}</span>}
        {feedbackPublicationId && !feedbackEvidence({ revision, entityId, imageId, observationId }).imageId && <span>{t("feedbackNoPhotoEvidence")}</span>}
      </div>
      {feedbackPublicationId && identitySuggestion && identitySuggestion.entityIds[0] === entityId && <div className="identity-preview"><p>{t("identitySuggestionContext")}</p><p>{identitySuggestion.reason}</p><button disabled={!ready || !historyReady || working || !!feedbackRetry} onClick={() => {
        identitySubmission.current = identitySuggestion;
        chat.current.submitUserMessage({text: identitySuggestion.reason});
      }}>{t("identitySuggestionSubmit")}</button></div>}
      {createElement("deep-chat", { ref: chat, className: "deep-chat", auxiliaryStyle: "#container { height: 100%; width: 100%; }" })}
      {working && <p role="status">{t("running")}</p>}
      <ErrorNotice error={error} />
      {feedbackPublicationId && !historyReady && !!error && <button disabled={working} onClick={() => loadPublicHistory()}>{t("retry")}</button>}
      {feedbackRetry && <div className="feedback-retry"><p>{t("feedbackSendUnconfirmed")}</p><button disabled={working} onClick={() => sendFeedback(feedbackRetry)}>{t("feedbackRetrySameRequest")}</button></div>}
      {proposal && !feedbackPublicationId && (
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
