// Run: node --experimental-strip-types web/checks/agent-panel.mjs
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import React from "react";
import ts from "typescript";
import { observationsFor } from "../src/core.ts";

const require = createRequire(import.meta.url);
let active, instance, language = "en", serial = 0, postHandler, feedbackHistoryHandler;
const requests = [], sessions = new Map(), feedbackTurns = [];
const turn = (sequence, entityId, extra = {}) => ({ id: `turn-${sequence}`, projectId: "project", branchId: "branch", baseRevisionId: "revision",
  conversationId: `conversation-${entityId}`, sequence, status: "succeeded", request: { entityId, message: `question-${sequence}` },
  response: { kind: "answer", message: `answer-${sequence}` }, ...extra });
const history = [...Array.from({ length: 500 }, (_, n) => turn(n + 1, "b")), turn(501, "a"), turn(502, "a"),
  turn(503, "a", { branchId: "alternative", conversationId: "other-branch" }), turn(504, "a", { baseRevisionId: "other-version", conversationId: "other-version" }),
  turn(505, null, { request: { policyId: "policy", message: "old-policy" }, baseRevisionId: "old-version" }),
  turn(506, null, { request: { policyId: "policy", message: "new-policy" } })];
const hooks = {
  createElement: React.createElement,
  useRef(value) { const i = active.cursor++; return active.slots[i] ||= { current: value }; },
  useState(value) { const owner = active, i = owner.cursor++; if (!Object.hasOwn(owner.slots, i)) owner.slots[i] = value;
    return [owner.slots[i], next => { owner.slots[i] = typeof next === "function" ? next(owner.slots[i]) : next; }]; },
  useEffect(effect, deps) { const owner = active, i = owner.cursor++, previous = owner.slots[i];
    if (!previous || deps.some((value, n) => !Object.is(value, previous.deps[n]))) owner.effects.push(() => {
      previous?.cleanup?.(); owner.slots[i] = { deps, cleanup: effect() };
    }); },
};
class ApiError extends Error { constructor(status, message) { super(message); this.status = status; } }
const api = { ApiError, id: () => `fresh-${++serial}`,
  async feedbackSession(pub, entity, pending) { const key = `${pub}:${entity}`;
    if (!sessions.has(key)) sessions.set(key, { capability: "c".repeat(43), conversationId: `feedback-${entity}` });
    const session = sessions.get(key); if (pending !== undefined) session.pending = pending || undefined; return session; },
  async request(url, options = {}) {
    requests.push({ url, options });
    if (options.method === "POST") return postHandler(url, options);
    if (url.includes("/feedback")) {
      if (feedbackHistoryHandler) return feedbackHistoryHandler(url, options);
      return { items: feedbackTurns.filter(row => url.includes(encodeURIComponent(row.entityId))) };
    }
    const params = new URLSearchParams(url.split("?")[1]);
    return { items: history.filter(row => row.sequence > Number(params.get("afterSequence") || 0) &&
      (!params.has("conversationId") || row.conversationId === params.get("conversationId"))).slice(0, 500) };
  },
};
globalThis.customElements = { whenDefined: async () => {} };
const source = (await readFile(new URL("../src/AgentPanel.tsx", import.meta.url), "utf8")).replace('import("deep-chat")', "Promise.resolve()");
const code = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText;
const module = { exports: {} };
new Function("require", "module", "exports", code)(name => name === "react" ? hooks : name === "./api" ? api :
  name === "./core" ? { observationsFor } : name === "./i18n" ? { useI18n: () => ({ t: key => key, language }) } : name === "./App" ? { ErrorNotice: () => null } : require(name), module, module.exports);
const { AgentPanel } = module.exports;
const props = { projectId: "project", revision: { id: "revision", document: { entities: [{ id: "a", label: "A" }, { id: "b", label: "B" }] } },
  branch: { id: "branch" }, entityId: "a", observationId: "obs-a", imageId: "photo-a", box: null, canWrite: true,
  onApply: () => { throw Error("This check must never apply an operation"); } };
function nodes(node) { return React.isValidElement(node) ? [node, ...React.Children.toArray(node.props.children).flatMap(nodes)] : []; }
function unmount() { if (instance) for (const slot of instance.slots) slot?.cleanup?.(); instance = undefined; }
async function render(nextProps) {
  const child = AgentPanel(nextProps);
  if (!instance || instance.key !== child.key) { unmount(); instance = { key: child.key, slots: [], chat: {}, effects: [], cursor: 0 }; }
  for (let n = 0; n < 16; n++) {
    active = instance; active.cursor = 0; active.tree = child.type(child.props);
    nodes(active.tree).find(node => node.type === "deep-chat").props.ref.current = active.chat;
    for (const effect of active.effects.splice(0)) effect();
    await Promise.resolve();
  }
  return instance;
}
let a = await render(props);
assert.deepEqual(a.chat.history.map(row => row.text), ["question-501", "answer-501", "question-502", "answer-502"], "Object, branch and revision history is isolated");
assert.ok(requests.some(({ url }) => url.endsWith("afterSequence=500")), "History beyond the backend's 500-row page is retained");
const aKey = a.key;
language = "zh"; await render({ ...props, imageId: "photo-b" });
assert.equal(instance.key, aKey, "Changing language or source photograph does not discard the object conversation");
let resolveA, oldResponses = 0;
postHandler = (_url, { body }) => { assert.equal(body.conversationId, "conversation-a"); return new Promise(resolve => { resolveA = resolve; }); };
const pending = a.chat.connect.handler({ messages: [{ role: "user", text: "A pending" }] }, { onResponse: async () => { oldResponses++; } });
await Promise.resolve();
const b = await render({ ...props, entityId: "b" });
resolveA(turn(507, "a", { response: { kind: "proposal", message: "Only A", operations: [{ type: "setLabel", entityId: "a", label: "edited" }] } }));
await pending; await render({ ...props, entityId: "b" });
assert.equal(oldResponses, 0, "An unmounted A request cannot send its reply into the active B chat");
assert.equal(nodes(b.tree).some(node => node.props.className === "proposal"), false, "A proposal never leaks to B");
assert.ok(b.chat.history.every(row => !row.text.includes("501") && row.text !== "Only A"));
history.push(turn(508, "b", { conversationId: "legacy-mixed" }), turn(509, "a", { conversationId: "legacy-mixed" }));
a = await render(props);
assert.ok(a.chat.history.some(row => row.text === "question-509"), "Matching old history remains readable");
postHandler = (_url, { body }) => { assert.notEqual(body.conversationId, "legacy-mixed", "A mixed legacy transcript cannot be replayed to the model"); return turn(510, "a"); };
await a.chat.connect.handler({ messages: [{ role: "user", text: "continue A" }] }, { onResponse: async () => {} });
const policy = await render({ ...props, entityId: null, policyId: "policy", policyRevisionId: "policy-version" });
assert.deepEqual(policy.chat.history.map(row => row.text), ["old-policy", "answer-505", "new-policy", "answer-506"], "Policy history keeps its prior cross-scene-version behavior");

const publicProps = { ...props, entityId: "object/one", feedbackPublicationId: "publication", canWrite: false,
  revision: { id: "revision", document: { entities: [{ id: "object/one", label: "One", observationRefs: ["obs-a", "obs-b"] }, { id: "object/two", label: "Unphotographed model", observationRefs: [] }],
    observations: [{ id: "obs-a", imageId: "photo-a" }, { id: "obs-b", imageId: "photo-b" }, { id: "foreign-obs", imageId: "foreign-photo" }] } } };
const failedHistoryRequests = [];
feedbackHistoryHandler = (url, options) => {
  failedHistoryRequests.push({ url, options });
  if (failedHistoryRequests.length === 1) throw new TypeError("temporary preflight failure");
  return { items: [] };
};
let publicChat = await render(publicProps);
assert.equal(publicChat.chat.textInput.disabled, true, "A failed history load must not allow messages in an unknown conversation");
assert.equal(failedHistoryRequests.length, 1, "A history failure does not automatically retry");
const savedSession = { ...sessions.get("publication:object/one") };
const beforeHistoryRetry = requests.length;
await nodes(publicChat.tree).find(node => node.type === "button" && node.props.children === "retry").props.onClick();
publicChat = await render(publicProps);
assert.equal(failedHistoryRequests.length, 2, "An explicit Retry action reloads feedback history");
assert.equal(failedHistoryRequests[0].url, failedHistoryRequests[1].url, "History recovery preserves publication, entity and conversation");
assert.deepEqual(sessions.get("publication:object/one"), savedSession, "History recovery never resets the browser capability or conversation");
assert.ok(requests.slice(beforeHistoryRetry).every(({ options }) => !options.method || options.method === "GET"), "History recovery can only read; it cannot send feedback or run a model");
assert.equal(failedHistoryRequests[1].options.feedbackCapability, savedSession.capability);
assert.equal(nodes(publicChat.tree).some(node => node.type === "button" && node.props.children === "retry"), false);
feedbackHistoryHandler = undefined;
assert.equal(publicChat.chat.textInput.disabled, false, "Public feedback does not require project ownership");
const publicKey = publicChat.key, publicConversation = sessions.get("publication:object/one").conversationId;
publicChat = await render({ ...publicProps, imageId: "photo-b", observationId: null });
assert.equal(publicChat.key, publicKey, "The same public object keeps its conversation when switching photos");
assert.equal(sessions.get("publication:object/one").conversationId, publicConversation);
const publicPosts = [];
postHandler = async (url, options) => {
  assert.ok(url.includes("/entities/object%2Fone/feedback"));
  assert.equal(options.projectId, undefined); assert.equal(options.capability, undefined);
  assert.equal(options.feedbackCapability, "c".repeat(43)); publicPosts.push(options.body);
  assert.equal(options.body.imageId, "photo-b");
  assert.equal(options.body.observationId, undefined, "A changed source photo never inherits a stale observation");
  if (publicPosts.length === 1) throw new TypeError("network interrupted after send");
  const row = { ...options.body, id: "feedback-turn", publicationId: "publication", revisionId: "revision", entityId: "object/one",
    status: "saved", assistantMessage: null, errorCode: "feedback_model_disabled", createdAt: "2026-09-13" };
  feedbackTurns.push(row); return row;
};
await publicChat.chat.connect.handler({ messages: [{ role: "user", text: "Button is missing" }] }, { onResponse: async () => {} });
unmount(); publicChat = await render(publicProps);
assert.ok(publicChat.chat.history.some(row => row.text === "Button is missing"), "Unconfirmed feedback is restored after closing the panel");
await nodes(publicChat.tree).find(node => node.type === "button" && node.props.children === "feedbackRetrySameRequest").props.onClick();
await render(publicProps);
assert.equal(publicPosts[0].requestId, publicPosts[1].requestId, "Network retry preserves the persisted request identity");
assert.equal(publicPosts[0].conversationId, publicPosts[1].conversationId);
assert.ok(publicChat.chat.history.some(row => row.text === "feedbackSavedNoAgent"), "Disabled models are reported honestly rather than as an Agent reply");
assert.equal(nodes(publicChat.tree).some(node => node.props.className === "proposal"), false, "Public feedback never exposes Apply");
assert.equal(nodes(publicChat.tree).some(node => node.props.className === "feedback-retry"), false);
assert.equal(sessions.get("publication:object/one").pending, undefined);
const other = await render({ ...publicProps, entityId: "object/two" });
assert.deepEqual(other.chat.history, [], "Public object feedback history is independently scoped");
assert.notEqual(sessions.get("publication:object/one").conversationId, sessions.get("publication:object/two").conversationId);
let unknownPosts = 0, unknownReply;
assert.ok(nodes(other.tree).some(node => node.props.children === "feedbackNoPhotoEvidence"), "An unphotographed model explicitly shows no attached photo evidence");
postHandler = async (_url, { body }) => {
  assert.equal(body.imageId, undefined, "A zero-observation model cannot inherit the current photo");
  assert.equal(body.observationId, undefined, "A zero-observation model cannot inherit another object's observation");
  unknownPosts++; return { ...body, status: "outcome_unknown", assistantMessage: null, errorCode: "provider_outcome_unknown" };
};
await other.chat.connect.handler({ messages: [{ role: "user", text: "Check this" }] }, { onResponse: async ({ text }) => { unknownReply = text; } });
await render({ ...publicProps, entityId: "object/two" });
assert.equal(unknownReply, "feedbackOutcomeUnknown");
assert.equal(unknownPosts, 1, "An unknown model outcome never triggers an automatic repeat");
assert.equal(nodes(other.tree).some(node => node.props.className === "feedback-retry"), false);
unmount();

// Structured suggestions require the explicit shared-submit button; ordinary chat stays private.
const suggestion={decision:"same",entityIds:["object/one","object/two"],observationGroups:[["obs-a","obs-b"],[]],reason:"Shared identity evidence",shareForReview:true};
let sharedPayload;
publicChat=await render({...publicProps,identitySuggestion:suggestion});
publicChat.chat.submitUserMessage=({text})=>publicChat.chat.connect.handler({messages:[{role:"user",text}]},{onResponse:async()=>{}});
postHandler=async(_url,{body})=>{sharedPayload=body;return {...body,id:"shared",entityId:"object/one",status:"saved",assistantMessage:null,errorCode:null};};
await nodes(publicChat.tree).find(node=>node.type==="button"&&node.props.children==="identitySuggestionSubmit").props.onClick();
await new Promise(resolve=>setTimeout(resolve,0));
assert.deepEqual(sharedPayload.identitySuggestion,suggestion,"Explicitly shared structure is carried by the persisted feedback transport");
await publicChat.chat.connect.handler({messages:[{role:"user",text:"Private follow-up"}]},{onResponse:async()=>{}});
assert.equal(sharedPayload.identitySuggestion,undefined,"The following private message cannot automatically share another identity suggestion");
assert.equal(nodes(publicChat.tree).some(node=>node.props.className==="proposal"),false);

// Exercise the actual fetch boundary; feedback authentication never uses the
// project-owner path or puts its private capability in a URL.
const realApi = await import("../src/api.ts");
globalThis.window = { dispatchEvent() {} };
const capability = "s".repeat(43), feedbackUrl = "/api/publications/pub/entities/object/feedback";
globalThis.fetch = async (url, options) => {
  assert.equal(url, feedbackUrl); assert.equal(options.headers.get("Authorization"), `Feedback ${capability}`);
  assert.equal(options.credentials, "omit"); return new Response(JSON.stringify({ items: [] }), { status: 200 });
};
await realApi.request(feedbackUrl, { feedbackCapability: capability });
await realApi.request(feedbackUrl, { method: "POST", feedbackCapability: capability, body: { message: "test" } });
await assert.rejects(realApi.request(feedbackUrl, { feedbackCapability: capability, projectId: "project" }), /mixed_capability_context/);
console.log("Agent panel: exact object/branch/version history, pagination, A→B pending isolation, mixed-transcript prevention, policy continuity, public feedback, explicit GET-only history recovery, persisted same-request retry and no Apply passed.");
