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

import base64
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, RedirectResponse
from pydantic import BaseModel, Field

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
from .interactive_report import strip_report_chat

RUNS = Path("runs")


class ReviewIn(BaseModel):
    run_id: str
    reviewer: str
    decision: str = "confirmed"
    overridden_status: str | None = None
    reason: str = ""


class AgentIn(BaseModel):
    run_id: str
    message: str = Field(min_length=1, max_length=8000)
    apply: bool = True
    action: Literal["ask", "add_object"] | None = None
    frame_id: str | None = None
    box: tuple[int, int, int, int] | None = None
    label: str | None = Field(default=None, max_length=160)
    language: Literal["zh", "en"] = "zh"


def _run_dir(run_id: str) -> Path:
    if not run_id or "/" in run_id or run_id.startswith("."):
        raise HTTPException(400, "bad run id")
    run_dir = RUNS / run_id
    if not run_dir.exists():
        raise HTTPException(404, f"run {run_id} not found")
    return run_dir


def _ensure_report(run_id: str) -> Path:
    """Render saved evidence; opening a public report never launches inference."""
    run_dir = _run_dir(run_id)
    _check_surface_inventory(run_dir)
    saved = run_dir / 'report.html'
    if saved.is_file():
        return saved
    if (run_dir / "inventory" / "inventory.json").exists():
        from .interactive_report import build_interactive_run_report

        return build_interactive_run_report(run_id)
    from .report import build_run_report

    return build_run_report(run_id)


def _check_surface_inventory(run_dir: Path, viewer_html: str | None = None) -> None:
    manifest = run_dir / "surface" / "surface.json"
    if not manifest.exists():
        return
    inventory = run_dir / "inventory" / "inventory.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not inventory.is_file() or hashlib.sha256(inventory.read_bytes()).hexdigest() != data.get("inventory_sha256"):
        raise HTTPException(409, "物体清单已更新，内部模型需要重新建立对象对应后才能显示。")
    viewer = run_dir / "viewer.html"
    if viewer_html is None:
        viewer_html = viewer.read_text(encoding="utf-8") if viewer.is_file() else ""
    marker = re.search(r'<script id="surface-data"[^>]*>(.*?)</script>',
                       viewer_html, re.S)
    embedded = json.loads(marker.group(1)) if marker else None
    keys = ("inventory_sha256", "asset_sha256", "face_map_sha256", "source_cameras_sha256",
            "face_count", "supported_inv")
    if embedded is None or any(embedded.get(key) != data.get(key) for key in keys):
        raise HTTPException(409, "内部模型已更新，请重建 3D 视图后打开报告。")


def _embed_report_surface(page: str, run_dir: Path) -> str:
    """Keep the downloadable report independent of the running product server."""
    iframe = re.search(r'(<iframe[^>]*class="v3d"[^>]*?)srcdoc="([^"]*)"', page)
    if iframe is None:
        raise HTTPException(409, "rebuild the 3D viewer before downloading its surface")
    viewer = html.unescape(iframe.group(2))
    _check_surface_inventory(run_dir, viewer)
    marker = re.search(r'(<script id="surface-data"[^>]*>)(.*?)(</script>)', viewer, re.S)
    if marker is None or json.loads(marker.group(2)) is None:
        raise HTTPException(409, "rebuild the 3D viewer before downloading its surface")
    data = json.loads(marker.group(2))
    for key, filename, mime in (("asset_url", "surface.glb", "model/gltf-binary"),
                                ("face_map_url", "face-inv.bin", "application/octet-stream")):
        path = run_dir / "surface" / filename
        if not path.is_file():
            raise HTTPException(409, "surface export is incomplete")
        data[key] = "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    encoded = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    viewer = viewer[:marker.start(2)] + encoded + viewer[marker.end(2):]
    return page[:iframe.start(2)] + html.escape(viewer, quote=True) + page[iframe.end(2):]


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
  <a id="ag-workspace" href="/reports/__RUN__">打开报告工作区 · Agent</a>
  <p class="hint">在来源照片上选定区域，通过 Agent 添加物体；空间范围和量测随报告一起更新。</p>
</div>
<script>
(function(){
  const run = document.getElementById('hub').dataset.run;
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
    from .workspace_access import register_access
    register_access(api)
    from .report_workspace import register_workspace
    published = os.environ.get("PANOPTES_PUBLISHED_ROOT")
    register_workspace(api, service, published_root=Path(published) if published else None)

    @api.get("/")
    def library_home():
        return RedirectResponse("/reports")
    # tests import this module too; they must not kick provider-spending
    # chains for whatever runs happen to be on disk
    if not os.environ.get("PANOPTES_NO_RESUME"):
        resume_interrupted_chains()

    @api.exception_handler(HTTPException)
    async def _html_errors(request, exc: HTTPException):
        # a person clicking a report link gets a page with a way back,
        # not a JSON blob; API callers still get JSON
        if request.url.path.startswith("/report/"):
            body = (
                "<!doctype html><meta charset='utf-8'><title>Panoptes</title>"
                "<div style='font:15px/1.6 -apple-system,Noto Sans SC,sans-serif;"
                "max-width:640px;margin:60px auto;padding:0 20px'>"
                f"<h2>{exc.status_code} · {html.escape(str(exc.detail))}</h2>"
                "<p>链接可能被截断或带了多余字符。到 <a href='/'>工作台 → 报告 tab</a>"
                " 的历史列表里点对应的 run 进入报告页。</p></div>"
            )
            return HTMLResponse(body, status_code=exc.status_code)
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    @api.get("/report/{run_id}", response_class=HTMLResponse)
    def report_page(run_id: str, embedded: bool = False) -> HTMLResponse:
        # tolerate a trailing punctuation/markdown tail from copy-paste
        run_id = run_id.strip().rstrip("*）)。，,.;:")
        path = _ensure_report(run_id)
        page = strip_report_chat(path.read_text(encoding="utf-8"))
        panel = _HUB_PANEL.replace("__RUN__", html.escape(run_id)).replace(
            "__REVIEW__", html.escape(_review_markdown(RUNS / run_id).replace("**", ""))
        )
        # one product: the report page carries the app's navigation
        nav = (
            '<nav style="position:sticky;top:0;z-index:40;display:flex;flex-wrap:wrap;gap:10px 18px;'
            'align-items:center;padding:10px 22px;background:var(--surface,#fff);'
            'border-bottom:1px solid var(--line,#DAE0DE);font:14px/1.4 \'Noto Sans SC\','
            '\'IBM Plex Sans\',sans-serif">'
            '<a href="/reports" style="font-weight:600;text-decoration:none">← 报告历史</a>'
            f'<a href="/reports/{html.escape(run_id)}" style="text-decoration:none">报告工作区 · Agent</a>'
            f'<span style="margin-left:auto;font-family:monospace;font-size:12px">{html.escape(run_id)}</span>'
            f'<a href="/report/{html.escape(run_id)}/file" style="text-decoration:none">下载 HTML</a>'
            '<a href="#hub" style="text-decoration:none">审核 / Agent ↓</a>'
            "</nav>"
        )
        if embedded:
            nav = ""
            panel = ""
        head_end = page.find("</style>")
        if head_end != -1:
            head_end += len("</style>")
            page = page[:head_end] + nav + page[head_end:]
        else:
            page = nav + page
        # a single-run page shows its report open, not behind a summary row
        page = page.replace('<details class="case"', '<details class="case" open', 1)
        # served page: load the 3D viewer by URL (full point budget, page
        # stays light); the downloadable file keeps the inline srcdoc copy
        viewer = RUNS / run_id / "viewer.html"
        if viewer.exists():
            revision = hashlib.sha256(viewer.read_bytes()).hexdigest()[:16]
            page = re.sub(
                r'(<iframe[^>]*class="v3d"[^>]*?)srcdoc="[^"]*"',
                lambda m: m.group(1) + f'src="/report/{run_id}/viewer?v={revision}"',
                page,
                count=1,
            )
        panel = panel.replace("_未审核_", "未审核")
        page += '<script src="/workspace-assets/i18n-catalog.js"></script><script src="/workspace-assets/i18n.js"></script>'
        if embedded:
            page += """<script>
addEventListener('message',function(event){
  if(event.source!==parent||event.origin!==location.origin)return;
  var message=event.data;
  if(!message||message.type!=='panoptes:report-selection'||!Array.isArray(message.inv))return;
  document.querySelectorAll('.linked').forEach(function(linked){
    if(linked.bus)linked.bus.set(message.inv,{source:'workspace'});
    var picker=linked.querySelector('.photo-frame');
    if(picker&&Array.from(picker.options).some(function(o){return o.value===message.frame_id;})){
      picker.value=message.frame_id;picker.dispatchEvent(new Event('change'));
    }
  });
});</script>"""
        return HTMLResponse(page + panel)

    @api.get("/report/{run_id}/viewer")
    def report_viewer(run_id: str) -> FileResponse:
        run_dir = _run_dir(run_id.strip().rstrip("*）)。，,.;:"))
        _check_surface_inventory(run_dir)
        viewer = run_dir / "viewer.html"
        if not viewer.exists():
            raise HTTPException(404, "viewer not built yet")
        return FileResponse(str(viewer), media_type="text/html", headers={"Cache-Control": "no-cache"})

    @api.get("/report/{run_id}/surface/{asset}")
    def report_surface(run_id: str, asset: str) -> FileResponse:
        media = {"surface.json": "application/json", "surface.glb": "model/gltf-binary",
                 "face-inv.bin": "application/octet-stream"}
        if asset not in media:
            raise HTTPException(404, "unknown surface asset")
        path = _run_dir(run_id) / "surface" / asset
        if not path.is_file():
            raise HTTPException(404, "surface asset not built yet")
        return FileResponse(str(path), media_type=media[asset], headers={"Cache-Control": "no-cache"})

    @api.get("/report/{run_id}/file")
    def report_file(run_id: str) -> Response:
        path = _ensure_report(run_id)
        run_dir = _run_dir(run_id)
        page = strip_report_chat(path.read_text(encoding="utf-8"))
        if (run_dir / "surface" / "surface.json").exists():
            page = _embed_report_surface(page, run_dir)
        return HTMLResponse(page, headers={"Content-Disposition":
            "attachment; filename*=UTF-8''" + quote(f"panoptes-{run_id}.html", safe="")})

    @api.get("/api/chat/{run_id}")
    def chat(run_id: str) -> JSONResponse:
        return JSONResponse(chat_history(_run_dir(run_id)))

    @api.post("/api/review")
    def review(body: ReviewIn) -> JSONResponse:
        writable_run(body.run_id)
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
        import fcntl
        run = writable_run(body.run_id)
        with (run / ".edit.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return agent_locked(body)

    def writable_run(run_id: str) -> Path:
        run = _run_dir(run_id)
        from .artifacts import _read_json_dict
        if _read_json_dict(run / "workspace.json").get("state") in {"queued", "analyzing"}:
            raise HTTPException(409, "This report is still processing; edit it after analysis completes")
        if any(_read_json_dict(p).get('state') in {'queued', 'running'} for p in (RUNS / '.generations' / run_id).glob('*/status.json')):
            raise HTTPException(409, "This report has an object generation in progress")
        return run

    def agent_locked(body: AgentIn) -> JSONResponse:
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
                runs_root=RUNS,
                answer_fn=_answer, refine_fn=_refine, log_ask=False,
                action=body.action, frame_id=body.frame_id, box=body.box,
                label=body.label, language=body.language,
            )
        except ProviderError as exc:
            raise HTTPException(502, f"{exc.provider} {exc.operation} failed") from exc
        except (RefineError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        if out["changed"] or (out.get("applied") and out.get("evidence_id")):
            # the page the operator refreshes must already show the change
            # A saved mask can survive a failed refresh; a repeated add repairs
            # the report using that cached evidence without another model call.
            try:
                from .report_refresh import refresh_report_after_correction
                result = refresh_report_after_correction(RUNS / body.run_id)
                out["refresh"] = result
                if not result["updated"]:
                    out["refresh_error"] = result["error"]
            except Exception as exc:
                import logging
                logging.getLogger(__name__).exception("Correction saved but report refresh failed")
                out["refresh_error"] = type(exc).__name__
        # Match persisted source identity, never a label or a guessed inventory index.
        if out.get("applied") and out.get("evidence_id"):
            evidence_path = RUNS / body.run_id / "object-evidence.json"
            evidence = json.loads(evidence_path.read_text()) if evidence_path.is_file() else {}
            out["candidate_ids"] = [
                candidate["id"] for candidate in evidence.get("candidates", [])
                if any(ref.get("evidence_id") == out["evidence_id"]
                       for ref in candidate.get("source_refs", []))
            ]
        return JSONResponse(out)

    return gr.mount_gradio_app(api, demo, path="/workbench")


app = create_server()
