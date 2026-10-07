// r5b (visits): the `visits` layer (fast_report/visits.py) in the live report's side panel. Pick visit A (the site map's objects,
// drawn where they stood as ghosts) and / or visit B (this video's objects, coloured by what changed); the changes list with
// each change's +-u and its evidence frames from both visits.
import { useMemo, useState } from "react";
import { assetURL, VISIT_COLOR, type Patch } from "./live-report";

type Tr = (a: string, b: string) => string;
const CHANGE = ["moved", "new", "missing", "changed_height", "changed_angle"];
const LABEL: Record<string, [string, string]> = {
  moved: ["移动了", "moved"], new: ["新出现", "new"], missing: ["不见了", "missing"], changed_height: ["高度变了", "changed height"],
  changed_angle: ["角度变了", "changed angle"], static: ["没变", "static"], delineated_otherwise: ["在，分割不同", "there, delineated otherwise"],
  not_observed_in_b: ["这次没看到那里", "not observed in B"], not_observed_in_a: ["上次没看到那里", "not observed in A"], not_compared: ["未比较", "not compared"],
};
const rgb = (s: string) => `rgb(${(VISIT_COLOR[s] || [.6, .6, .6]).map(v => Math.round(v * 255)).join(",")})`;
const f = (v: unknown, d = 2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—";

export function VisitsTabLabel({ patch, tr }: { patch?: Patch; tr: Tr }) {
  const n = CHANGE.reduce((a, k) => a + (patch?.data?.counts?.[k] || 0), 0);
  return <>{tr("到访", "Visits")} ({n})</>;
}

export default function VisitsPanel({ patch, show, setShow, selected, onSelect, onSeek, tr }: {
  patch: Patch; show: { a: boolean; b: boolean }; setShow: (s: { a: boolean; b: boolean }) => void; selected: string | null;
  onSelect: (id: string) => void; onSeek: (t: number) => void; tr: Tr;
}) {
  const d = patch.data, [filter, setFilter] = useState<string>("changes"), [open, setOpen] = useState<number | null>(null);
  const rows = useMemo(() => (d.objects || []).map((o: any, i: number) => ({ ...o, i }))
    .filter((o: any) => filter === "changes" ? CHANGE.includes(o.status) : filter === "all" || o.status === filter), [d, filter]);
  const reg = d.registration || [], va = d.visits?.[0], vb = d.visits?.[1];
  return <div className="visits-panel">
    <section className="mvp-block">
      <h4>{tr("同一现场的两次到访", "Two visits of one site")}</h4>
      <div className="visits-pick" role="group" aria-label={tr("显示哪次到访", "which visit to show")}>
        <button aria-pressed={show.a} onClick={() => setShow({ ...show, a: !show.a })} title={va?.report}>
          A · {tr("现场图（第一次）", "site map (first visit)")}</button>
        <button aria-pressed={show.b} onClick={() => setShow({ ...show, b: !show.b })} title={vb?.report}>
          B · {tr("这次（这个视频）", "this visit (this video)")}</button>
      </div>
      <p><small>{tr("A 的对象画在它们当时的位置（淡色幽灵），B 的对象按变化上色。", "A's objects are drawn where they stood (faint ghosts); B's objects are coloured by what changed.")}</small></p>
      {reg.map((r: any) => <p key={r.b_shot} className="visits-reg">
        {tr("镜头", "Shot")} {r.b_shot + 1} → A {tr("镜头", "shot")} {r.a_shot != null ? r.a_shot + 1 : "—"}:{" "}
        {r.accepted ? <>{tr("已配准", "registered")} ± {f(r.u_m * 100, 0)} cm · {tr("尺度", "scale")} {f(r.floor_transform?.s, 3)} · {tr("转角", "yaw")} {f(r.floor_transform?.yaw_deg, 1)}°
          · {tr("地面倾角", "floors")} {f(r.floor?.tilt_deg, 1)}° <small>(DA3 any-view, register_cut_shot {tr("门限", "gate")})</small></>
          : <>{tr("拒绝", "refused")}: <small>{r.refused_because}</small></>}
      </p>)}
    </section>
    <section className="mvp-block visits-counts">
      {[["changes", tr("所有变化", "all changes")], ...Object.keys(d.counts || {}).map(k => [k, `${tr(...(LABEL[k] || [k, k]))}`]), ["all", tr("全部", "all")]].map(([k, label]) =>
        <button key={k} aria-pressed={filter === k} onClick={() => setFilter(k)} style={{ borderColor: rgb(k) }}>
          {label}{d.counts?.[k] !== undefined ? ` ${d.counts[k]}` : ""}</button>)}
    </section>
    <ul className="visits-list">
      {rows.map((o: any) => {
        const tile = o.evidence?.tile && patch.blobs?.[o.evidence.tile], id = o.b || ("visit-a:" + o.a);
        return <li key={o.i} aria-current={selected === id || selected === o.b} className="visits-row">
          <button onClick={() => { onSelect(id); setOpen(open === o.i ? null : o.i); if (o.evidence?.b?.t != null) onSeek(o.evidence.b.t); }}>
            <span className="visits-dot" style={{ background: rgb(o.status) }} />
            <b>{tr(...(LABEL[o.status] || [o.status, o.status]))}</b> {o.name_a || o.name_b || o.a || o.b}
            {o.name_a && o.name_b && o.name_a !== o.name_b ? <small> → {o.name_b}</small> : null}
          </button>
          <Delta o={o} tr={tr} />
          {open === o.i && <div className="visits-evidence">
            {tile ? <img src={assetURL("sha256:" + tile.sha256)!} alt={tr("证据：左 A，右 B", "evidence: A left, B right")} /> : null}
            <p><small>A {o.a || "—"}{o.evidence?.a?.t != null ? ` @ ${f(o.evidence.a.t, 1)} s` : ""} · B {o.b || "—"}{o.evidence?.b?.t != null ? ` @ ${f(o.evidence.b.t, 1)} s` : ""}
              {o.place_in_b ? ` · ${tr("B 里那个位置", "its place in B")}: ${o.place_in_b.state} (${o.place_in_b.free_views ?? 0} ${tr("个看穿视角", "see-through views")})` : ""}
              {o.place_in_a ? ` · ${tr("A 里那个位置", "its place in A")}: ${o.place_in_a.state} (${o.place_in_a.free_views ?? 0} ${tr("个看穿视角", "see-through views")})` : ""}
              {o.carried?.name ? ` · ${tr("名字沿用 A", "name carried from A")}: ${o.carried.name.value}` : ""}</small></p>
          </div>}
        </li>;
      })}
      {!rows.length && <li className="mvp-empty">{tr("这一类没有对象", "nothing in this class")}</li>}
    </ul>
    <p className="mvp-block"><small>{d.rules?.not_observed}. {d.rules?.scale}.</small></p>
  </div>;
}

function Delta({ o, tr }: { o: any; tr: Tr }) {
  const dl = o.delta || {}, p = dl.position_xy_m, t = dl.top_above_floor, b = dl.base_above_floor;
  const ang = ["principal_axis_tilt_deg", "planar_slope_deg"].filter(k => dl[k]);
  if (!p && !t) return null;
  return <p className="visits-delta"><small>
    {p && <>{tr("位移", "shift")} {f(p.norm)} ± {f(p.u)} m</>}
    {t && <> · {tr("顶部", "top")} {t.value >= 0 ? "+" : ""}{f(t.value)} ± {f(t.u)} m</>}
    {b && <> · {tr("底部", "base")} {b.value >= 0 ? "+" : ""}{f(b.value)} ± {f(b.u)} m</>}
    {ang.map(k => <span key={k}> · {k === "planar_slope_deg" ? tr("坡度", "slope") : tr("倾斜", "tilt")} {dl[k].value >= 0 ? "+" : ""}{f(dl[k].value, 1)} ± {f(dl[k].u, 1)}°</span>)}
    {o.appearance_cos != null && <> · {tr("外观", "appearance")} cos {f(o.appearance_cos)}</>}
  </small></p>;
}
