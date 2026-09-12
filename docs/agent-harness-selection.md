# Report agent harness decision

Verified against upstream documentation on 2026-09-09.

The current report needs a grounded question action and an auditable add-object action. Reuse `agent_hub.agent_turn` and the existing SAM/refinement tool. Installing a general coding agent would not fix missing frame identity, dropped masks, or disconnected report rebuilds.

| Candidate | Open source / hosting | Models and tools | Fit for this product |
| --- | --- | --- | --- |
| Existing Python hub | Already in this service; no new dependency | Explicit domain actions; selected box + label makes zero locator calls, text description makes one locator call, followed by SAM | Chosen for this change. The report controls action, source frame, and persistence; no shell or arbitrary code tool is added. |
| OpenCode | MIT; `opencode serve` exposes a self-hosted HTTP/OpenAPI server | Many providers including DeepSeek and local models; custom tools/MCP and granular permissions | Useful for a future coding workspace, but its project/filesystem/agent-session runtime is unnecessary for this two-action report. [License](https://github.com/anomalyco/opencode/blob/dev/LICENSE), [server](https://opencode.ai/docs/server/), [providers](https://opencode.ai/docs/providers/), [permissions](https://opencode.ai/docs/permissions/). |
| LangChain / Deep Agents | Deep Agents is MIT and runs as an application library on LangGraph | Tool-calling frontier, open-weight, and local models; custom tools, persistence, planning, filesystem, subagents | If multi-step domain workflows become necessary, start with the lighter `create_agent` and explicit tools. Upstream itself recommends that when the bundled Deep Agents middleware is unnecessary. Model/tool-call limit middleware can enforce per-run limits. [Repository and comparison](https://github.com/langchain-ai/deepagents), [customization](https://docs.langchain.com/oss/python/deepagents/customization), [call-limit middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in). |
| Claude Agent SDK | The Python wrapper has an MIT license, bundles the Claude Code CLI, and its README states SDK use is governed by Anthropic Commercial Terms except separately licensed components | Claude Code runtime; in-process custom MCP tools, hooks, `max_turns`. `allowed_tools` auto-approves named tools; it does **not** remove other tools | Not selected for an open-model, low-cost domain backend. It is not a standalone provider-neutral open-source runtime merely because the wrapper is MIT. [Official repository, tools and terms](https://github.com/anthropics/claude-agent-sdk-python). |

DeepSeek is a model/provider choice; Deep Agents is an agent harness. They are different layers. Supporting a model in a harness does not establish that the model can ground image boxes accurately; that requires evaluation on held-out photos.

## Typed report contract

`agent_turn(..., action="add_object", frame_id="frame_0002", box=[x1,y1,x2,y2], label="switch", language="en")` uses original source-image pixels. With no box, the message is located once in the specified frame. A supplied box requires a label and bypasses the locator entirely. `action="ask"` bypasses legacy keyword mutation routing.

The tool returns `evidence_id`, `frame_id`, `box`, `label`, `geometry_status`, `geometry_reason`, `applied`, `changed`, and an overlay. A small or unmeasurable mask is retained as 2D evidence without inventing a metric entity. A completed 3D measurement uses the selected frame and the run's scale assumptions; it does not certify a device's safety function or compliance.

Accepted records are written to `refinements.json` before measurement, with a cache identity derived from source frame, original image SHA-256, label, and box. Records bind the original SAM cache hash, source-image hash, instruction, and locator rationale. `chat.jsonl` records action, language, selection, and returned evidence identity. A repeated identical request reuses its SAM cache; changing the source image changes its cache identity. Preview calls do not add accepted refinement records.

The report service must finish the chain after a changed action: rebuild inventory on the selected frames from persisted masks, rebuild `object-evidence.json`, then regenerate viewer/report assets. Retained 2D evidence remains visible even when no metric geometry can be drawn. Rebuilding these artifacts must not invoke enumeration or another locator as part of the chat turn.

## Checks

`pytest tests/test_agent.py tests/test_agent_hub.py tests/test_refine.py tests/test_inventory_unmeasured.py tests/test_report_linked.py tests/test_report.py tests/test_viewer.py tests/test_object_evidence.py -q`

The regression covers a 49-pixel object in photo 2 surviving into the generic object registry; no boxed-action VLM call; one text locator call using photo 2; missing 3D geometry retaining source evidence; original-image hash invalidating a SAM cache; selected-frame measurements; and hashed refinement IDs being consumed only by their own frame. Existing 200-pixel footprint rejection and report/viewer checks remain in the suite.
