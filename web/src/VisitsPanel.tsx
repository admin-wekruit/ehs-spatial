// r5b (visits): the `visits` layer (fast_report/visits.py) in the live report's side panel. Pick visit A (the site map's objects,
// drawn where they stood as ghosts) and / or visit B (this video's objects, coloured by what changed); the changes list with
// each change's +-u and its evidence frames from both visits.
import { useMemo, useState } from "react";
import { assetURL, VISIT_COLOR, type Patch } from "./live-report";

type Tr = (id: string, params?: Record<string, string | number>) => string;
const CHANGE = ["moved", "new", "missing", "changed_height", "changed_angle"];
const LABEL: Record<string, string> = {
  moved: "visits.label.moved", new: "visits.label.new", missing: "visits.label.missing", changed_height: "visits.label.changed_height",
  changed_angle: "visits.label.changed_angle", static: "visits.label.static", delineated_otherwise: "visits.label.delineated_otherwise",
  not_observed_in_b: "visits.label.not_observed_in_b", not_observed_in_a: "visits.label.not_observed_in_a", not_compared: "visits.label.not_compared",
};
const rgb = (s: string) => `rgb(${(VISIT_COLOR[s] || [.6, .6, .6]).map(v => Math.round(v * 255)).join(",")})`;
const f = (v: unknown, d = 2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—";

export function VisitsTabLabel({ patch, tr }: { patch?: Patch; tr: Tr }) {
  const n = CHANGE.reduce((a, k) => a + (patch?.data?.counts?.[k] || 0), 0);
  return <>{tr("VisitsPaneltsx.182")} ({n})</>;
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
      <h4>{tr("VisitsPaneltsx.183")}</h4>
      <div className="visits-pick" role="group" aria-label={tr("VisitsPaneltsx.184")}>
        <button aria-pressed={show.a} onClick={() => setShow({ ...show, a: !show.a })} title={va?.report}>
          A · {tr("VisitsPaneltsx.185")}</button>
        <button aria-pressed={show.b} onClick={() => setShow({ ...show, b: !show.b })} title={vb?.report}>
          B · {tr("VisitsPaneltsx.186")}</button>
      </div>
      <p><small>{tr("VisitsPaneltsx.187")}</small></p>
      {reg.map((r: any) => <p key={r.b_shot} className="visits-reg">
        {tr("VisitsPaneltsx.188")} {r.b_shot + 1} → A {tr("VisitsPaneltsx.189")} {r.a_shot != null ? r.a_shot + 1 : "—"}:{" "}
        {r.accepted ? <>{tr("VisitsPaneltsx.190")} ± {f(r.u_m * 100, 0)} cm · {tr("VisitsPaneltsx.191")} {f(r.floor_transform?.s, 3)} · {tr("VisitsPaneltsx.192")} {f(r.floor_transform?.yaw_deg, 1)}°
          · {tr("VisitsPaneltsx.193")} {f(r.floor?.tilt_deg, 1)}° <small>{tr("visits.registrationMethod", { basis: tr("VisitsPaneltsx.194") })}</small></>
          : <>{tr("VisitsPaneltsx.195")}: <small>{r.refused_because}</small></>}
      </p>)}
    </section>
    <section className="mvp-block visits-counts">
      {[["changes", tr("VisitsPaneltsx.196")], ...Object.keys(d.counts || {}).map(k => [k, `${tr(LABEL[k] || k)}`]), ["all", tr("LiveReporttsx.136")]].map(([k, label]) =>
        <button key={k} aria-pressed={filter === k} onClick={() => setFilter(k)} style={{ borderColor: rgb(k) }}>
          {label}{d.counts?.[k] !== undefined ? ` ${d.counts[k]}` : ""}</button>)}
    </section>
    <ul className="visits-list">
      {rows.map((o: any) => {
        const tile = o.evidence?.tile && patch.blobs?.[o.evidence.tile], id = o.b || ("visit-a:" + o.a);
        return <li key={o.i} aria-current={selected === id || selected === o.b} className="visits-row">
          <button onClick={() => { onSelect(id); setOpen(open === o.i ? null : o.i); if (o.evidence?.b?.t != null) onSeek(o.evidence.b.t); }}>
            <span className="visits-dot" style={{ background: rgb(o.status) }} />
            <b>{tr(LABEL[o.status] || o.status)}</b> {o.name_a || o.name_b || o.a || o.b}
            {o.name_a && o.name_b && o.name_a !== o.name_b ? <small> → {o.name_b}</small> : null}
          </button>
          <Delta o={o} tr={tr} />
          {open === o.i && <div className="visits-evidence">
            {tile ? <img src={assetURL("sha256:" + tile.sha256)!} alt={tr("VisitsPaneltsx.197")} /> : null}
            <p><small>A {o.a || "—"}{o.evidence?.a?.t != null ? ` @ ${f(o.evidence.a.t, 1)} s` : ""} · B {o.b || "—"}{o.evidence?.b?.t != null ? ` @ ${f(o.evidence.b.t, 1)} s` : ""}
              {o.place_in_b ? ` · ${tr("VisitsPaneltsx.198")}: ${o.place_in_b.state} (${o.place_in_b.free_views ?? 0} ${tr("VisitsPaneltsx.199")})` : ""}
              {o.place_in_a ? ` · ${tr("VisitsPaneltsx.200")}: ${o.place_in_a.state} (${o.place_in_a.free_views ?? 0} ${tr("VisitsPaneltsx.199")})` : ""}
              {o.carried?.name ? ` · ${tr("VisitsPaneltsx.201")}: ${o.carried.name.value}` : ""}</small></p>
          </div>}
        </li>;
      })}
      {!rows.length && <li className="mvp-empty">{tr("VisitsPaneltsx.202")}</li>}
    </ul>
    <p className="mvp-block"><small>{d.rules?.not_observed}. {d.rules?.scale}.</small></p>
  </div>;
}

function Delta({ o, tr }: { o: any; tr: Tr }) {
  const dl = o.delta || {}, p = dl.position_xy_m, t = dl.top_above_floor, b = dl.base_above_floor;
  const ang = ["principal_axis_tilt_deg", "planar_slope_deg"].filter(k => dl[k]);
  if (!p && !t) return null;
  return <p className="visits-delta"><small>
    {p && <>{tr("VisitsPaneltsx.203")} {f(p.norm)} ± {f(p.u)} m</>}
    {t && <> · {tr("VisitsPaneltsx.204")} {t.value >= 0 ? "+" : ""}{f(t.value)} ± {f(t.u)} m</>}
    {b && <> · {tr("VisitsPaneltsx.205")} {b.value >= 0 ? "+" : ""}{f(b.value)} ± {f(b.u)} m</>}
    {ang.map(k => <span key={k}> · {k === "planar_slope_deg" ? tr("VisitsPaneltsx.206") : tr("VisitsPaneltsx.207")} {dl[k].value >= 0 ? "+" : ""}{f(dl[k].value, 1)} ± {f(dl[k].u, 1)}°</span>)}
    {o.appearance_cos != null && <> · {tr("VisitsPaneltsx.208")} cos {f(o.appearance_cos)}</>}
  </small></p>;
}
