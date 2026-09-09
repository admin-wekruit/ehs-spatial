"""The product server: one FastAPI app that serves the standalone report
page and mounts the Gradio workbench under it.

    uvicorn ehs_spatial.serve:app --host 127.0.0.1 --port 7860

Routes
  /                      Gradio workbench (submit / report list / video)
  /report/<run_id>       THE report — the full interactive HTML with the
                         review form and the agent chat embedded, one page,
                         nothing nested in an iframe
  /report/<run_id>/file  the same HTML as a download (no live panels)
  POST /api/review       {run_id, reviewer, decision, overridden_status, reason}
  POST /api/agent        {run_id, message, apply} -> {intent, reply, changed}
  GET  /api/chat/<run_id> chat history for the page's timeline
"""

import html
import json
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .agent_hub import agent_turn, chat_history
from .app import (
    _review_markdown,
    analysis_state,
    build_app,
    resume_interrupted_chains,
    save_disposition,
)
from .contracts import GroundedAnswer
from .pipeline import EHSAssessmentPipeline
from .providers.base import ProviderError

RUNS = Path("runs")


class ReviewIn(BaseModel):
    run_id: str
    reviewer: str
    decision: str = "confirmed"
    overridden_status: str | None = None
    reason: str = ""


class AgentIn(BaseModel):
    run_id: str
    message: str
    apply: bool = True


def _run_dir(run_id: str) -> Path:
    if not run_id or "/" in run_id or run_id.startswith("."):
        raise HTTPException(400, "bad run id")
    run_dir = RUNS / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"run {run_id} not found")
    return run_dir


def _ensure_report(run_id: str) -> Path:
    """Serve the freshest report we can build; a stale/missing inventory
    is upgraded in the background by the same rule the tab uses."""
    from .app import _chain_running, _start_deep_report_chain

    run_dir = _run_dir(run_id)
    state = analysis_state(run_dir)
    if state != "current" and not _chain_running(run_dir):
        _start_deep_report_chain(run_id)
    if (run_dir / "inventory" / "inventory.json").exists():
        from .interactive_report import build_interactive_run_report

        return build_interactive_run_report(run_id)
    from .report import build_run_report

    return build_run_report(run_id)


_HUB_PANEL = """
<section class="hub" id="hub" data-run="__RUN__">
<style>
.hub{max-width:1020px;margin:0 auto;padding:0 22px 60px;font:15px/1.6 'Noto Sans SC','IBM Plex Sans',sans-serif}
.hub h3{margin:28px 0 10px;font-size:18px}
.hub .card{border:1px solid var(--line,#DAE0DE);border-radius:6px;padding:14px 16px;background:var(--card,#FCFDFC);margin:10px 0}
.hub label{display:block;font-size:12.5px;color:var(--muted,#5A676F);margin:8px 0 2px}
.hub input,.hub select,.hub textarea{width:100%;padding:7px 9px;border:1px solid var(--line,#DAE0DE);border-radius:4px;background:var(--surface,#fff);color:inherit;font:inherit}
.hub button{margin-top:10px;padding:8px 16px;border:0;border-radius:4px;background:var(--accent,#0E7490);color:#fff;font:inherit;cursor:pointer}
.hub .row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.hub .log{max-height:340px;overflow:auto;border:1px solid var(--line,#DAE0DE);border-radius:4px;padding:10px;background:var(--surface,#fff)}
.hub .msg{margin:6px 0;padding:6px 10px;border-radius:4px;white-space:pre-wrap}
.hub .msg.u{background:var(--insuff-bg,#E9EDEF)}.hub .msg.a{background:var(--pass-bg,#E4F3EA)}
.hub .status{font-size:12.5px;color:var(--muted,#5A676F);margin-top:6px}
.hub .hint{font-size:12px;color:var(--muted,#5A676F)}
</style>
<h3>Review 审核 — 对本 run 签字</h3>
<div class="card">
  <div id="review-now" class="status">__REVIEW__</div>
  <div class="row">
    <div><label>审核员</label><input id="rv-reviewer" placeholder="姓名"></div>
    <div><label>决定</label><select id="rv-decision"><option value="confirmed">confirmed 确认</option><option value="overridden">overridden 推翻</option></select></div>
    <div><label>推翻后的状态</label><select id="rv-status"><option value="">—</option><option>PASS</option><option>FAIL</option><option>INSUFFICIENT_EVIDENCE</option></select></div>
  </div>
  <label>理由（推翻必填）</label><textarea id="rv-reason" rows="2"></textarea>
  <button id="rv-save">保存审核</button><span id="rv-msg" class="status"></span>
</div>
<h3>Agent 对话 — 追问 / 补测 / 纠错 / 调整策略（全部写入 run，进报告）</h3>
<div class="card">
  <div id="ag-log" class="log"></div>
  <label>说一句</label>
  <input id="ag-say" placeholder="例：围栏离机器人多远？ / 右边黄色柱子帮我量 / 这块不是围栏，是导向挡板 / p01 改成 0.8m">
  <label><input type="checkbox" id="ag-apply" checked style="width:auto"> 回灌判定（改动会重评 policy）</label>
  <button id="ag-send">发送</button><span id="ag-msg" class="status"></span>
  <p class="hint">改动后刷新本页即为更新后的报告（判定、实体、附录同步）。</p>
</div>
<script>
(function(){
  const run = document.getElementById('hub').dataset.run;
  const log = document.getElementById('ag-log');
  function add(role, text){ const d=document.createElement('div'); d.className='msg '+(role==='user'?'u':'a'); d.textContent=text; log.appendChild(d); log.scrollTop=log.scrollHeight; }
  fetch('/api/chat/'+run).then(r=>r.json()).then(ms=>ms.forEach(m=>add(m.role, m.content))).catch(()=>{});
  document.getElementById('ag-send').onclick = async () => {
    const say = document.getElementById('ag-say'); const msg = say.value.trim(); if(!msg) return;
    add('user', msg); say.value=''; document.getElementById('ag-msg').textContent='处理中…';
    try {
      const r = await fetch('/api/agent',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_id:run,message:msg,apply:document.getElementById('ag-apply').checked})});
      const j = await r.json(); if(!r.ok) throw new Error(j.detail||r.status);
      add('assistant', '['+j.intent+'] '+j.reply); document.getElementById('ag-msg').textContent = j.changed ? '已改动 run — 刷新页面看更新后的报告' : '';
    } catch(e){ document.getElementById('ag-msg').textContent='失败：'+e.message; }
  };
  document.getElementById('rv-save').onclick = async () => {
    const body = {run_id:run, reviewer:document.getElementById('rv-reviewer').value, decision:document.getElementById('rv-decision').value, overridden_status:document.getElementById('rv-status').value||null, reason:document.getElementById('rv-reason').value};
    document.getElementById('rv-msg').textContent='保存中…';
    try {
      const r = await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      const j = await r.json(); if(!r.ok) throw new Error(j.detail||r.status);
      document.getElementById('review-now').textContent = j.summary; document.getElementById('rv-msg').textContent='已保存 — 刷新页面后报告头部同步';
    } catch(e){ document.getElementById('rv-msg').textContent='失败：'+e.message; }
  };
})();
</script>
</section>
"""


def create_server() -> FastAPI:
    import os

    service = EHSAssessmentPipeline()
    demo = build_app(service)
    api = FastAPI(title="Panoptes")
    # tests import this module too; they must not kick provider-spending
    # chains for whatever runs happen to be on disk
    if not os.environ.get("PANOPTES_NO_RESUME"):
        resume_interrupted_chains()

    @api.get("/report/{run_id}", response_class=HTMLResponse)
    def report_page(run_id: str) -> HTMLResponse:
        path = _ensure_report(run_id)
        page = path.read_text(encoding="utf-8")
        panel = _HUB_PANEL.replace("__RUN__", html.escape(run_id)).replace(
            "__REVIEW__", html.escape(_review_markdown(RUNS / run_id).replace("**", ""))
        )
        # one product: the report page carries the app's navigation
        nav = (
            '<nav style="position:sticky;top:0;z-index:40;display:flex;gap:18px;'
            'align-items:center;padding:10px 22px;background:var(--surface,#fff);'
            'border-bottom:1px solid var(--line,#DAE0DE);font:14px/1.4 \'Noto Sans SC\','
            '\'IBM Plex Sans\',sans-serif">'
            '<a href="/" style="font-weight:600;text-decoration:none">← 工作台 · 提交新分析</a>'
            '<a href="/" style="text-decoration:none">历史</a>'
            f'<span style="margin-left:auto;font-family:monospace;font-size:12px">{html.escape(run_id)}</span>'
            f'<a href="/report/{html.escape(run_id)}/file" style="text-decoration:none">下载 HTML</a>'
            '<a href="#hub" style="text-decoration:none">审核 / Agent ↓</a>'
            "</nav>"
        )
        head_end = page.find("</style>")
        if head_end != -1:
            head_end += len("</style>")
            page = page[:head_end] + nav + page[head_end:]
        else:
            page = nav + page
        # a single-run page shows its report open, not behind a summary row
        page = page.replace('<details class="case"', '<details class="case" open', 1)
        panel = panel.replace("_未审核_", "未审核")
        return HTMLResponse(page + panel)

    @api.get("/report/{run_id}/file")
    def report_file(run_id: str) -> FileResponse:
        path = _ensure_report(run_id)
        return FileResponse(str(path), filename=f"panoptes-{run_id}.html")

    @api.get("/api/chat/{run_id}")
    def chat(run_id: str) -> JSONResponse:
        return JSONResponse(chat_history(_run_dir(run_id)))

    @api.post("/api/review")
    def review(body: ReviewIn) -> JSONResponse:
        _run_dir(body.run_id)
        try:
            save_disposition(
                service, body.run_id, body.reviewer, body.decision,
                body.overridden_status, body.reason,
            )
        except gr.Error as exc:
            raise HTTPException(400, str(exc)) from exc
        summary = _review_markdown(RUNS / body.run_id).replace("**", "")
        return JSONResponse({"summary": summary})

    @api.post("/api/agent")
    def agent(body: AgentIn) -> JSONResponse:
        _run_dir(body.run_id)
        from .agent import agent_refine
        from .refine import RefineError

        def _answer(question: str) -> str:
            answer = GroundedAnswer.model_validate(
                service.answer_question(body.run_id, question)
            )
            facts = ", ".join(answer.fact_ids) or "none"
            return f"{answer.answer}\n\nFact IDs: {facts}"

        def _refine(rid: str, instruction: str, apply: bool) -> dict:
            try:
                return agent_refine(
                    rid, instruction, runs_root=service.store.root, apply=bool(apply)
                )
            except (RefineError, ProviderError) as exc:
                return {"message": f"补测失败：{exc}"}

        try:
            # the pipeline already logs grounded QA as chat_turn; logging
            # it again as agent_turn showed every question twice
            out = agent_turn(
                body.run_id, body.message.strip(), apply=body.apply,
                answer_fn=_answer, refine_fn=_refine, log_ask=False,
            )
        except ProviderError as exc:
            raise HTTPException(502, f"{exc.provider} {exc.operation} failed") from exc
        if out["changed"] and (RUNS / body.run_id / "inventory" / "inventory.json").exists():
            # the page the operator refreshes must already show the change
            from .interactive_report import build_interactive_run_report

            try:
                build_interactive_run_report(body.run_id)
            except Exception:
                pass
        return JSONResponse({k: out[k] for k in ("intent", "reply", "changed")})

    return gr.mount_gradio_app(api, demo, path="/")


app = create_server()
