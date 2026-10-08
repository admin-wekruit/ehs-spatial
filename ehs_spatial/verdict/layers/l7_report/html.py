"""L7 html@1: VerdictSet -> verdicts.html, one self-contained page (inline CSS + JS, nothing fetched; opens from file:// and from a
static host) that is also the labeling UI: provenance header, the verdict queue, a focus panel per row where a reviewer records the
gold status / reason / confidence / re-measure flag, a declared-inputs form for the scene, export / import of a `verdict-labels/1`
JSON (README.md). State lives in the page and in localStorage under the run id. No clocks: the same run renders byte-identical.
Patterns (docs/research/labeling-ui-survey-2026-10-08.md): one row at a time with its evidence (Argilla focus view, Prodigy),
number-key statuses (Prodigy, Potato), queue + progress + skip (Label Studio), label + reason + confidence (Braintrust, Encord review),
single HTML file offline (VGG VIA)."""
from __future__ import annotations

import json
from collections import Counter
from html import escape as esc
from pathlib import Path
from typing import Any

from ehs_spatial.verdict.contracts import Scene, VerdictSet
from ehs_spatial.verdict.plugins import register

STATUSES = ("PASS", "FAIL", "NEEDS_MEASUREMENT", "NEEDS_INPUT", "CANNOT_DETERMINE", "NOT_APPLICABLE")   # gold statuses = hotkeys 1-6
DECLARED = ("stop_time_ms", "resolution_mm", "risk_level", "restricted_space", "reach_radius_mm", "body_part", "payload_kg", "zone_depth_mm")
HOTKEYS = (("1 … 6", "gold status, in button order"), ("n / p", "next / previous row"), ("r", "toggle re-measure"), ("u", "toggle unsure"),
           ("Enter", "save + next"))

PAGE = """<!doctype html>
<html lang=en>
<meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fff;--fg:#1b1b1b;--mut:#666;--line:#d8d8d8;--panel:#f4f4f4;--acc:#0a66c2;--pass:#2e7d32;--fail:#c62828;--warn:#b26a00;--info:#4a4a9f;--na:#777}
@media(prefers-color-scheme:dark){:root{--bg:#141618;--fg:#e6e6e6;--mut:#9a9a9a;--line:#383838;--panel:#1e2124;--acc:#6cb4ff}}
*{box-sizing:border-box}
body{margin:0;padding:12px 16px;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px}h2{font-size:15px;margin:0 0 8px}
code,kbd{font:12px/1 ui-monospace,SFMono-Regular,Menlo,monospace}kbd{border:1px solid var(--line);border-radius:3px;padding:1px 4px;background:var(--panel);color:var(--fg)}
.prov,.mut{color:var(--mut)}.prov{margin:0 0 8px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0}
progress{width:160px;height:12px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);grid-template-areas:"queue focus";gap:16px}
#queue{grid-area:queue;list-style:none;margin:0;padding:0;max-height:80vh;overflow:auto;border:1px solid var(--line);border-radius:6px}
#queue li{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:6px 8px;border-bottom:1px solid var(--line);cursor:pointer}
#queue li.cur{background:var(--panel);box-shadow:inset 3px 0 var(--acc)}
#queue .n{color:var(--mut);min-width:2em}.who{flex:1 1 10em}
.focus{grid-area:focus;background:var(--panel);border-radius:6px;padding:12px;position:sticky;top:8px;align-self:start}
@media(max-width:900px){.grid{grid-template-columns:1fr;grid-template-areas:"focus" "queue"}.focus{position:static}#queue{max-height:none}}
.chip{border-radius:10px;padding:1px 8px;font-size:12px;color:#fff;background:var(--na);white-space:nowrap}
.chip.gold{outline:2px solid var(--fg)}
.PASS{background:var(--pass)}.FAIL{background:var(--fail)}.NEEDS_MEASUREMENT{background:var(--warn)}.NEEDS_INPUT,.CANNOT_DETERMINE{background:var(--info)}.NOT_APPLICABLE{background:var(--na)}
table{border-collapse:collapse;width:100%;margin:6px 0}td,th{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left;vertical-align:top;font-size:13px}
ul{margin:4px 0;padding-left:18px}
button{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);cursor:pointer}
#status{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}#status button{color:#fff}#status button.on{outline:3px solid var(--fg);outline-offset:1px}
label{display:block;margin:6px 0}input,textarea,select{font:inherit;width:100%;padding:5px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--fg)}
input[type=checkbox],input[type=file]{width:auto}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:end}.row label{flex:1 1 12em}.row label.ck{flex:0 0 auto}
.form{display:grid;grid-template-columns:repeat(auto-fill,minmax(14em,1fr));gap:8px;margin:8px 0}
#json{height:10em;font:12px/1.3 ui-monospace,monospace}
</style>
<body>
__HEAD__
<div class=bar><progress id=progress max=__N__ value=0></progress> <b id=count></b> labelled ·
 <select id=filter style="width:auto">__FILTER__</select> <span class=mut>__LEGEND__</span></div>
<div class=grid>
<ol id=queue></ol>
<div class=focus>
 <div id=detail></div>
 <div id=status>__BUTTONS__</div>
 <div class=row><label>reason <input id=reason placeholder="why, in a few words"></label>
  <label class=ck><input type=checkbox id=unsure> unsure <kbd>u</kbd></label><label class=ck><input type=checkbox id=remeasure> re-measure <kbd>r</kbd></label></div>
 <div class=bar><button id=prev>◀ prev <kbd>p</kbd></button><button id=next>next <kbd>n</kbd> ▶</button><button id=savenext>save + next <kbd>Enter</kbd></button></div>
</div></div>
<h2 style="margin-top:16px">Scene: declared inputs</h2>
<div class=form><label>reviewer <input id=reviewer placeholder="initials"></label>__FORM__</div>
<div class=bar><button id=export>Export labels JSON</button><button id=copy>Copy to clipboard</button>
 <label style="width:auto;margin:0">Import <input type=file id=import accept=".json,application/json"></label><span id=msg class=mut></span></div>
<textarea id=json readonly placeholder="Export / Copy puts the labels JSON here (fallback: select all + copy by hand)"></textarea>
<script>
const D = __DATA__;
const STATUSES = __STATUSES__;
const KEY = "verdict-labels/" + D.run_id;
const key = r => r.rule_id + "@" + r.rule_version + "|" + r.subjects.join(",");
const $ = id => document.getElementById(id);
const h = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[c]);
const fmt = x => x == null ? "—" : String(x);
const name = id => (D.objects[id] || {}).label || id;
const who = r => r.labels.length ? r.labels.join(" ↔ ") : r.subjects.map(name).join(" ");
const meas = r => r.measured == null ? "—" : fmt(r.measured) + " ± " + fmt(r.u) + " " + (r.unit || "");
let S = {reviewer: "", declared_inputs: {}, labels: {}};
try { Object.assign(S, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
S.declared_inputs = Object.assign({}, D.declared_inputs, S.declared_inputs);
let cur = 0, filter = "all";
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch (e) {} };
const lab = r => S.labels[key(r)] || (S.labels[key(r)] = {status: "", reason: "", confidence: "sure", remeasure: false});
const st = r => (S.labels[key(r)] || {}).status || "";
const labelled = () => D.rows.filter(st).length;
const visible = () => D.rows.map((r, i) => i).filter(i => filter === "all" || (filter === "unlabelled" ? !st(D.rows[i]) : D.rows[i].status === filter));

function renderQueue() {
  $("queue").innerHTML = visible().map(i => {
    const r = D.rows[i], L = S.labels[key(r)] || {};
    return `<li class="${i === cur ? "cur" : ""}" data-i="${i}"><span class=n>${i + 1}</span><b>${h(r.rule_id)}@${h(r.rule_version)}</b><span class=who>${h(who(r))}</span>` +
      `<span class="chip ${r.status}">${r.status}</span><span class=mut>${h(meas(r))}</span>` +
      (L.status ? `<span class="chip gold ${L.status}" title="gold">${L.status}</span>` : "") +
      (L.remeasure ? "<span class=chip>re-measure</span>" : "") + (L.confidence === "unsure" ? "<span class=chip>unsure</span>" : "") + "</li>";
  }).join("");
  $("progress").value = labelled(); $("count").textContent = labelled() + " / " + D.rows.length;
  const li = $("queue").querySelector(".cur"); if (li) li.scrollIntoView({block: "nearest"});
}
function renderFocus() {
  if (!D.rows.length) return;
  const r = D.rows[cur], L = lab(r);
  const subjects = r.subjects.map((id, k) => { const o = D.objects[id] || {};
    return `<li><b>${h(r.labels[k] || o.label || id)}</b> <span class=mut>${h(o.cls || "")} ${o.size_m ? o.size_m.map(x => x.toFixed(2)).join(" × ") + " m" : ""} ${h(o.confidence || "")}</span><br><code>${h(id)}</code></li>`; }).join("");
  const facts = (r.evidence.facts || []).map(f => `<tr><td>${h(f.pred)}</td><td>${h(f.args.map(name).join(", "))}</td><td>${fmt(f.value)} ± ${fmt(f.u)} ${h(f.unit || "")}</td><td>${h((f.flags || []).join(" "))}</td></tr>`).join("");
  const extra = Object.entries(r.evidence).filter(([k]) => k !== "facts" && k !== "views").map(([k, v]) => `<tr><td>${h(k)}</td><td colspan=3>${h(JSON.stringify(v))}</td></tr>`).join("");
  $("detail").innerHTML = `<h2>${cur + 1} / ${D.rows.length} · ${h(r.rule_id)}@${h(r.rule_version)} <span class="chip ${r.status}">${r.status}</span></h2><ul>${subjects}</ul>` +
    `<table><tr><th>measured ± u</th><th>threshold</th><th>margin</th><th>needs</th></tr><tr><td>${h(meas(r))}</td><td>${fmt(r.threshold)}</td><td>${fmt(r.margin)}</td><td>${h(r.unknown_inputs.join(", ")) || "—"}</td></tr></table>` +
    (facts || extra ? `<table><tr><th>evidence</th><th>args</th><th>value</th><th>flags</th></tr>${facts}${extra}</table>` : "") +
    (r.notes.length ? `<ul>${r.notes.map(n => `<li class=mut>${h(n)}</li>`).join("")}</ul>` : "") +
    (r.evidence.views ? `<p class=mut>views ${h(r.evidence.views.join(", "))}</p>` : "");
  for (const b of $("status").children) b.classList.toggle("on", b.dataset.s === L.status);
  $("reason").value = L.reason; $("unsure").checked = L.confidence === "unsure"; $("remeasure").checked = !!L.remeasure;
}
const render = () => { renderQueue(); renderFocus(); };
function move(d) { const v = visible(); if (!v.length) return; const k = v.indexOf(cur); cur = k < 0 ? v[0] : v[Math.min(v.length - 1, Math.max(0, k + d))]; render(); }
function set(field, value) { lab(D.rows[cur])[field] = value; save(); render(); }
function fillForms() {
  $("reviewer").value = S.reviewer || "";
  for (const el of document.querySelectorAll("[data-f]")) el.value = S.declared_inputs[el.dataset.f] ?? "";
}
function exportJson() {
  const labels = D.rows.filter(st).map(r => { const L = S.labels[key(r)];
    return {rule_id: r.rule_id, rule_version: r.rule_version, subjects: r.subjects, labels: r.labels, status: L.status, reason: L.reason || "", confidence: L.confidence || "sure", remeasure: !!L.remeasure}; });
  const declared = {};
  for (const [k, v] of Object.entries(S.declared_inputs)) { const s = String(v).trim(); if (s) declared[k] = isNaN(s) ? s : Number(s); }
  return {schema: "verdict-labels/1", run_id: D.run_id, scene_id: D.scene_id, benchmark: D.benchmark, reviewer: S.reviewer || "", rule_pack: D.rule_pack, labels, declared_inputs: declared};
}
const show = () => { $("json").value = JSON.stringify(exportJson(), null, 1); return $("json").value; };

$("queue").onclick = e => { const li = e.target.closest("li"); if (li) { cur = +li.dataset.i; render(); } };
$("status").onclick = e => { const b = e.target.closest("button"); if (b) set("status", b.dataset.s); };
$("reason").oninput = e => { lab(D.rows[cur]).reason = e.target.value; save(); };
$("unsure").onchange = e => set("confidence", e.target.checked ? "unsure" : "sure");
$("remeasure").onchange = e => set("remeasure", e.target.checked);
$("prev").onclick = () => move(-1); $("next").onclick = () => move(1); $("savenext").onclick = () => { save(); move(1); };
$("filter").onchange = e => { filter = e.target.value; move(0); };
$("reviewer").oninput = e => { S.reviewer = e.target.value; save(); };
for (const el of document.querySelectorAll("[data-f]")) el.oninput = () => { S.declared_inputs[el.dataset.f] = el.value; save(); };
$("export").onclick = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([show()], {type: "application/json"}));
  a.download = `labels-${D.scene_id}-${D.run_id}.json`; a.click(); $("msg").textContent = "downloaded " + a.download; };
$("copy").onclick = () => { show(); $("json").select();
  const fallback = () => { $("msg").textContent = document.execCommand("copy") ? "copied" : "copy failed: select the text below and copy it yourself"; };
  navigator.clipboard ? navigator.clipboard.writeText($("json").value).then(() => { $("msg").textContent = "copied"; }, fallback) : fallback(); };
$("import").onchange = e => { const f = e.target.files[0]; if (!f) return;
  f.text().then(t => { const j = JSON.parse(t); if (j.schema !== "verdict-labels/1") throw new Error("schema is " + j.schema + ", not verdict-labels/1");
    const known = new Set(D.rows.map(key)); let n = 0;
    for (const l of j.labels || []) { const k = l.rule_id + "@" + l.rule_version + "|" + l.subjects.join(",");
      if (known.has(k)) { n++; S.labels[k] = {status: l.status, reason: l.reason || "", confidence: l.confidence || "sure", remeasure: !!l.remeasure}; } }
    Object.assign(S.declared_inputs, j.declared_inputs || {}); if (j.reviewer) S.reviewer = j.reviewer;
    save(); fillForms(); render();
    $("msg").textContent = `imported ${n} of ${(j.labels || []).length} labels from ${f.name}` + (j.run_id !== D.run_id ? ` (run ${j.run_id})` : "");
  }).catch(err => { $("msg").textContent = "import failed: " + err.message; }); };
document.addEventListener("keydown", e => {
  const t = e.target, typing = t.tagName === "TEXTAREA" || t.tagName === "SELECT" || (t.tagName === "INPUT" && t.type !== "checkbox");
  if (e.ctrlKey || e.metaKey || e.altKey || (typing && !(e.key === "Enter" && t.id === "reason"))) return;
  const n = "123456".indexOf(e.key);
  if (n >= 0) set("status", STATUSES[n]);
  else if (e.key === "n") move(1); else if (e.key === "p") move(-1);
  else if (e.key === "r") set("remeasure", !lab(D.rows[cur]).remeasure);
  else if (e.key === "u") set("confidence", lab(D.rows[cur]).confidence === "unsure" ? "sure" : "unsure");
  else if (e.key === "Enter") { save(); move(1); }
  else return;
  e.preventDefault();
});
fillForms(); render();
</script>
"""


def payload(vs: VerdictSet, scene: Scene) -> dict:
    p = vs.provenance
    return {"run_id": p.run_id, "benchmark": p.benchmark, "plugins": p.plugins, "scene_id": vs.scene_id, "rule_pack": vs.rule_pack,
            "decision_rule": vs.decision_rule, "rows": [v.model_dump(exclude={"provenance"}) for v in vs.verdicts],
            "objects": {o.id: {"label": o.label, "cls": o.cls, "size_m": o.size_m, "confidence": o.confidence} for o in scene.objects},
            "declared_inputs": scene.declared_inputs}


def head(vs: VerdictSet) -> str:
    p, counts = vs.provenance, Counter(v.status for v in vs.verdicts)
    return (f"<h1>{esc(vs.scene_id)}: {len(vs.verdicts)} verdicts</h1>\n<p class=prov>run <code>{esc(p.run_id)}</code> · benchmark <code>{esc(p.benchmark)}</code>"
            f" · rule pack <code>{esc(vs.rule_pack)}</code> · decision <code>{esc(vs.decision_rule)}</code><br>plugins: "
            + " · ".join(f"{esc(layer)} <code>{esc(tag)}</code>" for layer, tag in sorted(p.plugins.items()))
            + "<br>counts: " + " · ".join(f"{s} {counts[s]}" for s in STATUSES[:5]) + "</p>")


def form(fields: list[str]) -> str:
    return "\n".join(f'<label>{esc(f)}<textarea data-f="{esc(f)}" rows=2 placeholder="x,y; x,y; … (plan view, m)"></textarea></label>'
                     if f == "restricted_space" else f'<label>{esc(f)}<input data-f="{esc(f)}"></label>' for f in fields)


def render(vs: VerdictSet, scene: Scene) -> str:
    fields = list(DECLARED) + sorted(set(scene.declared_inputs) - set(DECLARED))
    data = json.dumps(payload(vs, scene), sort_keys=True, ensure_ascii=False).replace("</", "<\\/")
    return (PAGE.replace("__TITLE__", esc(f"{vs.scene_id} · {vs.provenance.run_id}")).replace("__HEAD__", head(vs)).replace("__N__", str(len(vs.verdicts)))
            .replace("__FILTER__", "<option value=all>all rows</option><option value=unlabelled>unlabelled</option>" + "".join(f"<option>{s}</option>" for s in STATUSES[:5]))
            .replace("__LEGEND__", " · ".join(f"<kbd>{k}</kbd> {what}" for k, what in HOTKEYS))
            .replace("__BUTTONS__", "".join(f'<button data-s="{s}" class="{s}"><kbd>{i}</kbd> {s}</button>' for i, s in enumerate(STATUSES, 1)))
            .replace("__FORM__", form(fields)).replace("__STATUSES__", json.dumps(STATUSES)).replace("__DATA__", data))


@register("L7", "html", "1")
class Html:
    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        out = Path(inputs["workdir"]) / "verdicts.html"
        out.write_text(render(inputs["verdicts"], inputs["scene"]))
        return {"report": out}
