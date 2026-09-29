import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "./i18n";
import { mountSceneViewer, type SceneViewer } from "./viewer/native-viewer";
import { splatAnnotation } from "./viewer/splat-layer";
import { VideoMemory, VideoView, videoClock, type VideoPick } from "./VideoView";
import { cameraPath, cameraView, currentCameras } from "./core";
import { assetURL, clickClock, entityInfo, frameIndexAt, gunzip, latest, liveDocument, pickAt, pickMask, poll, readPick, SEVERITY, unknownRegion, worstVerdict,
  type Info, type Patch, type Pick, type Poll } from "./live-report";
import "./report-scene.css";

const OVER = 72;  // GB: 90% of an A100-80GB
const fmt = (v: unknown, digits = 1) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
type Tr = (a: string, b: string) => string;
const blobURL = (patch: Patch | undefined, role: string) => patch?.blobs?.[role] ? assetURL("sha256:" + patch.blobs[role].sha256) : null;
const seek = (t: unknown) => { if (typeof t === "number" && Number.isFinite(t)) window.dispatchEvent(new CustomEvent("panoptes:seek", { detail: t + 1e-3 })); };
const quantile = (xs: number[], q: number) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))] : undefined; };
type Clicked = VideoPick & { id: string | null; miss: ReturnType<typeof unknownRegion> | null };

/** #/live/<reportId>: a fast report as its layers arrive (fast_report/layers.py serves them on /fast). */
export default function LiveReport({ reportId }: { reportId: string }) {
  const { language } = useI18n(), zh = language === "zh", tr = (a: string, b: string) => zh ? a : b;
  const [state, setState] = useState<Poll>({ patches: [], written: {}, served: {}, run: null }), [error, setError] = useState<string>();
  useEffect(() => {  // every 500 ms: the new patches, and every written/served time so far
    let live = true, after = 0, timer = 0;
    const tick = async () => {
      try {
        const next = await poll(reportId, after);
        if (!live) return;
        if (next.patches.length) after = next.patches[next.patches.length - 1].seq;
        setState(s => next.patches.length || JSON.stringify([s.written, s.served, s.run]) !== JSON.stringify([next.written, next.served, next.run])
          ? { ...next, patches: [...s.patches, ...next.patches] } : s);
        setError(undefined);
      } catch { if (live) setError(tr("端点无响应，重试中", "The endpoint does not answer; retrying")); }
      if (live) timer = window.setTimeout(tick, 500);
    };
    void tick();
    return () => { live = false; clearTimeout(timer); };
  }, [reportId]);
  const layers = useMemo(() => latest(state.patches), [state.patches]);
  const docKey = Object.values(layers).filter(p => p.layer !== "timing").map(p => p.seq).sort((a, b) => a - b).join(",");
  const document = useMemo(() => liveDocument(reportId, layers), [reportId, docKey]);
  const docRef = useRef(document); docRef.current = document;
  // For the headless checks: an asset fetched twice was reloaded; ready = when each asset was first on the GPU (unix s); clicks = ms
  // from pointer-down to the card in the DOM; pickDecodeMs = pick + depth fetched, inflated and indexed.
  const stats = useRef({ uploads: {} as Record<string, number>, ready: {} as Record<string, number>, scenes: 0, errors: [] as string[],
    clicks: [] as number[], pickDecodeMs: null as number | null, pickSteps: null as Record<string, number> | null, pick: null as Pick | null,
    pickDecodes: [] as { seq: number; ms: number; fetch: number }[] }).current;
  const [pick, setPick] = useState<Pick | null>(null), [cardsLayer, setCardsLayer] = useState<any>(null), [clickMs, setClickMs] = useState<number[]>([]);
  useEffect(() => {  // pick + depth: fetched and inflated once per version
    const p = layers.pick;
    if (!p) return;
    let live = true;
    (async () => {
      const t0 = performance.now(), raw = await Promise.all(["pick", "depth"].map(role => p.blobs[role] ? fetch(blobURL(p, role)!, { priority: "high" } as RequestInit).then(r => r.arrayBuffer()) : null));
      const t1 = performance.now(), [runs, depth] = await Promise.all(raw.map(b => b && gunzip(b))), t2 = performance.now();
      if (!live) return;
      stats.pick = readPick(p.data, runs!, depth);
      const t3 = performance.now();
      stats.pickDecodeMs = t3 - t0; stats.pickSteps = { fetch: t1 - t0, inflate: t2 - t1, index: t3 - t2 }; setPick(stats.pick);
      stats.pickDecodes.push({ seq: p.seq, ms: t3 - t0, fetch: t1 - t0 });  // every version (v1 at load, v2 after densify)
    })().catch((e: Error) => { if (live) setError("pick: " + e.message); });
    return () => { live = false; };
  }, [layers.pick?.seq]);
  useEffect(() => {  // cards inline, or blob `cards` over 1 MB
    const p = layers.object_cards;
    if (!p) return;
    let live = true;
    (Array.isArray(p.data.cards) ? Promise.resolve(p.data.cards) : fetch(blobURL(p, "cards")!).then(r => r.json()))  // A: cards "blob" over 1 MB
      .then(cards => { if (live) setCardsLayer({ ...p.data, cards }); }).catch((e: Error) => { if (live) setError("cards: " + e.message); });
    return () => { live = false; };
  }, [layers.object_cards?.seq]);
  const judgements = layers.judgements?.data;
  const infos = useMemo(() => entityInfo(cardsLayer, judgements), [cardsLayer, judgements]);
  const resolve = async (id: string) => {  // blobs from the endpoint; a small inline layer (outlines under 1 MB) as an object URL
    const url = assetURL(id), inline = (docRef.current.assets.find(a => a.id === id) as any)?.inline;
    if (url) return url;
    if (inline) return URL.createObjectURL(new Blob([JSON.stringify(inline)], { type: "application/json" }));
    throw Error("unknown_asset");
  };

  const host = useRef<HTMLDivElement>(null), viewer = useRef<SceneViewer | undefined>(undefined), opened = useRef(false), follow = useRef(false);
  const [selected, setSelected] = useState<string | null>(null), [frame, setFrame] = useState<string | null>(null), [following, setFollowing] = useState(false);
  const [clicked, setClicked] = useState<Clicked | null>(null), [tab, setTab] = useState<"card" | "objects" | "memory">("card");
  const choose = (id: string | null) => { setSelected(id); setClicked(null); if (id) setTab("card"); };  // from 3D, the list, a card link
  const onPick = (p: VideoPick) => {  // a video click: the pick layer decides; before it lands, the smallest outline under the point
    const hit = pick ? pickAt(pick, p.t, p.x, p.y) : null, id = hit ? hit.id : p.under[0] ?? null;
    const miss = !id && pick ? unknownRegion(pick, layers.cameras?.data, cardsLayer, p.t, p.x, p.y) : null;
    setSelected(id); setClicked({ ...p, id, miss, under: p.under.filter(u => u !== id) }); setTab("card");
  };
  useLayoutEffect(() => {  // the card is in the DOM now: pointer-down -> here is the click latency
    if (!clickClock.t0) return;
    stats.clicks.push(performance.now() - clickClock.t0); clickClock.t0 = 0;
  }, [clicked]);
  useEffect(() => { if (stats.clicks.length !== clickMs.length) setClickMs([...stats.clicks]); }, [clicked]);
  const highlight = useMemo(() => pick && selected ? (t: number) => pickMask(pick, frameIndexAt(pick.data.frames, t), selected) : undefined, [pick, selected]);
  const [view, setView] = useState({ observed_surface: true, point_cloud: false, splats: true, labels: true, primitive: true });
  const [load, setLoad] = useState({ loaded: 0, total: 0 });
  useEffect(() => {
    (window as any).__live = stats;
    const v = mountSceneViewer(host.current!, {
      locale: language, layers: { lighting: true, allBounds: view.primitive, editable: false, ...view },
      resolveAsset: resolve,
      onEvent: (e) => {
        if (e.type === "selectionIntent") choose(e.entityId || null);
        else if (e.type === "userNavigation") { follow.current = false; setFollowing(false); }
        else if (e.type === "loadProgress" && e.phase === "gpu_upload") stats.uploads[e.assetId] = (stats.uploads[e.assetId] || 0) + 1;
        else if (e.type === "loadProgress" && e.phase === "assets") {
          setLoad({ loaded: e.loaded, total: e.total });
          for (const s of e.states) if (s.state === "ready" && s.assetId && !stats.ready[s.assetId]) stats.ready[s.assetId] = Date.now() / 1000;
        }
        else if (e.type === "loadError") { stats.errors.push(e.code); setError(e.code); }
      },
    });
    viewer.current = v;
    return () => { viewer.current = undefined; v.dispose(); };
  }, []);
  useEffect(() => {
    const v = viewer.current;
    if (!v) return;
    stats.scenes++;
    v.setScene({ id: docKey, document }).catch((e: Error) => setError(e.message));
    // The first cameras open a free view of the longest shot, never the (image-less) camera view.
    if (!opened.current && document.cameras.length) { opened.current = true; v.setCamera({ mode: "free" }); setFrame(currentCameras(document)[0].coordinateFrameId); }
  }, [document]);
  useEffect(() => { viewer.current?.setLayers({ ...view, allBounds: view.primitive }); }, [view]);  // boxes: fill and outline together
  useEffect(() => { viewer.current?.setSelection({ entityId: selected }); }, [selected]);
  const splat = splatAnnotation(document);
  useEffect(() => {
    viewer.current?.setSplats(splat ? { key: splat.assetId, url: assetURL(splat.assetId)!, count: splat.count, coordinateFrameId: splat.coordinateFrameId } : null);
  }, [splat?.assetId]);
  // The report's one clock is the video: the camera path marks it, and "follow" stands the view where the camera stood.
  useEffect(() => {
    const place = (time: number) => {
      viewer.current?.setLayers({ time });
      if (!follow.current || !frame) return;
      const path = cameraPath(docRef.current, frame).filter(p => p.time !== null);
      if (!path.length) return;
      const near = path.reduce((a, b) => Math.abs(b.time! - time) < Math.abs(a.time! - time) ? b : a);
      const camera = docRef.current.cameras.find(c => c.id === near.cameraId);
      if (camera) viewer.current?.setCamera(cameraView(camera));
    };
    place(videoClock.time);
    const onTime = (event: Event) => place((event as CustomEvent<number>).detail);
    window.addEventListener("panoptes:video-time", onTime);
    return () => window.removeEventListener("panoptes:video-time", onTime);
  }, [frame, following]);

  const entity = document.entities.find(e => e.id === selected) as any;
  const labels = [...new Set(Object.values(layers).flatMap(p => p.labels || []))];
  const shots = document.coordinateFrames.map(f => ({ id: f.id, cameras: document.cameras.filter(c => c.coordinateFrameId === f.id) }));
  const run = state.run || layers.timing?.data.run;
  const fixture = ["pick", "object_cards", "judgements"].map(k => layers[k]?.data?.fixture).find(Boolean);
  const duration = layers.video ? layers.video.data.frames / layers.video.data.fps : 0;
  const names = (id: string) => infos.get(id)?.card?.identity?.name || (document.entities.find(e => e.id === id) as any)?.label || id;
  return <section className="live-report">
    <header className="live-report-head">
      <strong>{tr("快速报告", "Fast report")} · {reportId}</strong>
      <span>{Object.values(layers).filter(p => p.layer !== "timing").map(p => `${p.layer}${p.version > 1 ? " v" + p.version : ""}`).join(" · ") || tr("等待第一层…", "waiting for the first layer…")}</span>
      {error && <em role="status">{error}</em>}
      {fixture && <mark className="live-report-fixture" title={fixture}>{tr("夹具数据：点选层、卡片、判断是查看器测试数据，不是测量", "Fixture: pick, cards and judgements are viewer test data, not measurements")}</mark>}
    </header>
    <div className="live-report-grid">
      <div className="live-report-main">
        <div className="live-report-video">
          <VideoView document={document} selectedId={selected} onSelect={choose} resolveAsset={resolve} onPick={onPick} highlight={highlight}
            marker={clicked?.miss ? { x: clicked.x, y: clicked.y } : null} />
          {!layers.video && <p>{tr("视频还没到", "The video has not arrived yet")}</p>}
        </div>
        <div className="live-report-3d">
          <div className="live-report-tools">
            {shots.map(s => <button key={s.id} aria-pressed={frame === s.id} onClick={() => { setFrame(s.id); follow.current = false; setFollowing(false);
              viewer.current?.setCamera({ mode: "free", cameraId: s.cameras[0]?.id }); }}>{tr("镜头", "Shot")} {Number(s.id.slice(5)) + 1} ({s.cameras.length})</button>)}
            {(["observed_surface", "point_cloud", "primitive", "labels", ...(splat ? ["splats"] : [])] as const).map(k => <label key={k}>
              <input type="checkbox" checked={(view as any)[k]} onChange={e => setView(v => ({ ...v, [k]: e.target.checked }))} />
              {({ observed_surface: tr("房间网格", "room mesh"), point_cloud: tr("点云", "points"), primitive: tr("物体框", "boxes"), labels: tr("名字", "names"), splats: tr("泼溅（相机附近）", "splats (near the path)") } as any)[k]}</label>)}
            {!!shots.length && <label><input type="checkbox" checked={following} onChange={e => { follow.current = e.target.checked; setFollowing(e.target.checked); }} />{tr("跟随视频相机", "follow the video camera")}</label>}
            <small>{load.total ? `${load.loaded}/${load.total}` : ""}</small>
          </div>
          <div ref={host} className="live-report-viewer" />
          {!!labels.length && <ul className="live-report-labels">{labels.map(l => <li key={l}>{l}</li>)}</ul>}
        </div>
      </div>
      <div className="live-report-side">
        <nav className="mvp-tabs" role="tablist">
          {([["card", tr("卡片", "Card")], ["objects", `${tr("对象", "Objects")} (${cardsLayer?.cards?.length ?? 0})`], ["memory", tr("视频记忆", "Video memory")]] as const).map(([k, label]) =>
            <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}>{label}</button>)}
        </nav>
        <div className="mvp-panel">
          {tab === "card" && (clicked?.miss ? <UnknownCard r={clicked.miss} under={clicked.under} names={names} onSelect={choose} tr={tr} />
            : <Card id={selected} info={infos.get(selected || "")} entity={entity} under={clicked?.id === selected ? clicked?.under || [] : []} names={names}
                judgementsPatch={layers.judgements} cardsPatch={layers.object_cards} duration={duration} onSelect={choose} tr={tr} />)}
          {tab === "objects" && <ObjectList cards={cardsLayer?.cards || []} infos={infos} selected={selected} onSelect={choose} tr={tr} />}
          {tab === "memory" && <VideoMemory document={document} onSelect={choose} />}
        </div>
      </div>
    </div>
    <Timing patches={state.patches} written={state.written} served={state.served} run={run} clicks={clickMs} pickMs={stats.pickDecodeMs} pickSteps={stats.pickSteps} tr={tr} />
  </section>;
}

const Chip = ({ v, tr }: { v: string | null | undefined; tr: Tr }) =>
  <span className="mvp-chip" data-v={v || "none"}>{v ? v.replace("_", " ") : tr("无检查", "no checks")}</span>;
const Tag = ({ children }: { children: React.ReactNode }) => <small className="mvp-tag">{children}</small>;

/** value ± u unit, with its bound, scale tag and note; or a not-observed / not-measurable status with its reason. */
function Quantity({ q, tr }: { q: any; tr: Tr }) {
  if (!q || q.value === undefined || q.value === null) return <span className="mvp-status">{q?.status || tr("没有值", "no value")}{q?.reason ? ` (${q.reason})` : ""}</span>;
  const digits = q.unit === "deg" || q.unit === "°" ? 1 : 2, v = Array.isArray(q.value) ? `(${q.value.map((x: number) => fmt(x, digits)).join(", ")})` : fmt(q.value, digits);
  const st = q.bound || q.status, scale = String(q.scale || "");  // A: status "at least" | "at most" | "needs review" with a value
  return <span>{st === "at least" ? "≥ " : st === "at most" ? "≤ " : ""}{v} ± {fmt(q.u, digits)} {q.unit === "deg" ? "°" : q.unit}
    {scale.startsWith("estimated") ? <Tag>{tr("估计尺度", "estimated")}</Tag> : scale ? <Tag>{tr("与尺度无关", "scale-free")}</Tag> : null}
    {st === "needs review" && <Tag>{tr("待复核", "needs review")}</Tag>}
    {q.path && <small> · {tr("路径", "path")}: {q.path}</small>}
    {q.reason && st !== "needs review" && <small> · {q.reason}</small>}
    {q.note && <small className="mvp-note"> · {q.note}</small>}</span>;
}

const PHYSICAL: [string, string, string][] = [["top_above_floor", "顶部离地", "top above floor"], ["base_above_floor", "底部离地", "base above floor"],
  ["height", "高", "height"], ["width", "宽", "width"], ["depth", "深", "depth"], ["footprint_m2", "占地", "footprint"],
  ["nearest_walked_path", "离走过的路径", "nearest walked path"], ["position_xy", "位置（地面坐标 x, y）", "position (floor frame x, y)"],
  ["principal_axis_tilt_deg", "主轴倾斜", "principal axis tilt"], ["planar_slope_deg", "平面坡度", "planar slope"]];
const SIZE_FIELDS = new Set(["top_above_floor", "base_above_floor", "height", "width", "depth", "footprint_m2", "position_xy", "nearest_walked_path"]);

function Card({ id, info, entity, under, names, judgementsPatch, cardsPatch, duration, onSelect, tr }: {
  id: string | null; info?: Info; entity: any; under: string[]; names: (id: string) => string; judgementsPatch?: Patch; cardsPatch?: Patch;
  duration: number; onSelect: (id: string) => void; tr: Tr;
}) {
  if (!id) return <p className="mvp-empty">{tr("点视频里的任何东西：它是什么、它的物理信息、它的安全判断。", "Click anything in the video: what it is, its physical info, its safety judgement.")}</p>;
  const card = info?.card;
  if (!card) return <><InfoCard entity={entity} tr={tr} />{!!info?.rows.length && <Judgements info={info} patch={judgementsPatch} tr={tr} />}<Under under={under} names={names} onSelect={onSelect} tr={tr} /></>;
  const idn = card.identity || {}, ph = card.physical || {}, bad = ph.size_check?.status === "implausible";
  const quantities = Object.values(ph).filter((q: any) => q && typeof q === "object" && "u" in q) as any[];
  return <article className="mvp-card" data-kind={card.kind}>
    <header className="mvp-block mvp-identity">
      <h3>{card.kind === "person" ? `${tr("人", "Person")} ${card.id.replace("person:", "")}` : idn.name} <Chip v={info!.verdict} tr={tr} /></h3>
      <p>{idn.confidence == null ? tr("没有置信度", "no confidence") : `${Math.round(idn.confidence * 100)}%`}{" "}
        <Tag>{idn.calibrated ? tr("已校准", "calibrated") : tr("未校准", "uncalibrated")}</Tag> · {tr("由", "decided by")} {idn.decided_by || "—"} · <Tag>{idn.label || tr("推断", "inferred")}</Tag></p>
      {(idn.status || idn.note) && <p><small>{[idn.status, idn.note].filter(Boolean).join(" · ")}</small></p>}
      {!!idn.alternatives?.length && <p><small>{tr("其他", "Alternatives")}: {idn.alternatives.map(([w, p]: [string, number]) => `${w} ${fmt(p, 2)}`).join(" · ")}</small></p>}
      {!!idn.candidates_struck?.length && <p><small>{tr("被尺寸/位置否掉", "Struck by size or placement")}: {idn.candidates_struck.map(([w, why]: [string, string]) => `${w} (${why})`).join(" · ")}</small></p>}
    </header>
    <section className="mvp-block"><h4>{tr("类别", "Kind")}</h4>
      <p>{card.class?.category || "other"} · {card.class?.mobility || "—"} <small>({card.class?.mobility_source || "—"}{card.class?.reason ? `: ${card.class.reason}` : ""})</small></p></section>
    {card.kind === "person" ? <PersonFacts card={card} names={names} onSelect={onSelect} tr={tr} /> : <section className="mvp-block">
      <h4>{tr("物理信息", "Physical")} <small>{tr("地面坐标，米为估计尺度（地面 + 假设 1.6 m 相机高）", "floor frame; metres at estimated scale (floor plane + assumed 1.6 m camera height)")}</small></h4>
      <p><small>{tr("证据级别", "Evidence")}: {ph.level || quantities[0]?.level || "2d only"}{ph.reason ? ` (${ph.reason})` : ""} · {card.views?.n ?? 0} {tr("个视角", "views")} · {Math.max(0, ...quantities.map(q => q.n_subsets || 0))} {tr("组视角子集", "view subsets")}
        {card.views?.distance_m && <> · {fmt(card.views.distance_m[0])}–{fmt(card.views.distance_m[1])} m {tr("远", "away")}</>} · {tr("方位角跨度", "azimuth spread")} {fmt(card.views?.azimuth_spread_deg, 0)}°</small></p>
      {bad && <p className="mvp-warn">{ph.size_check.reason}</p>}
      {ph.fragmented_support && <p className="mvp-warn">{tr("支撑点分散：尺寸待复核", "Fragmented support: the sizes need review")}</p>}
      <table className="mvp-physical"><tbody>
        {PHYSICAL.filter(([k]) => ph[k]).map(([k, zh, en]) => <tr key={k} data-implausible={(bad || ph.fragmented_support) && SIZE_FIELDS.has(k) || undefined}><th>{tr(zh, en)}</th><td><Quantity q={ph[k]} tr={tr} /></td></tr>)}
        {ph.primitive && <tr><th>{tr("参数化形状", "primitive")}</th><td>{ph.primitive.kind}: {ph.primitive.accepted ? tr("通过留出检验（显示用，不替代观测值）", "passed the held-out gate (beside the observed values, never replacing them)") : tr("未采用", "not accepted")}
          {ph.primitive.reason && <small> ({ph.primitive.reason})</small>}</td></tr>}
        {ph.walkway && <tr><th>{tr("通道", "walkway")}</th><td><span className="mvp-status">{ph.walkway.status}</span></td></tr>}
        {ph.size_check && <tr><th>{tr("尺寸检查", "size check")}</th><td>{ph.size_check.status}{ph.size_check.class_range_m && <small> ({ph.size_check.class || tr("其他词", "other word")}: {ph.size_check.class_range_m.join("–")} m{ph.size_check.measured_m != null ? `, measured ${fmt(ph.size_check.measured_m, 2)} m` : ""})</small>}</td></tr>}
      </tbody></table>
      <details className="mvp-parts"><summary>{tr("± 是怎么来的", "How each ± is made")}</summary>
        <table><tbody>{PHYSICAL.filter(([k]) => ph[k]?.parts).map(([k, zh, en]) => <tr key={k}><th>{tr(zh, en)}</th>
          <td>{Object.entries(ph[k].parts).map(([p, v]) => `${p} ${fmt(v, 3)}`).join(" · ")}{ph[k].subsets?.length ? ` · ${tr("子集", "subsets")} ${ph[k].subsets.map((v: number) => fmt(v, 2)).join(" / ")}` : ""}</td></tr>)}</tbody></table>
        <p><small>{tr("u = √(各项平方和) × k；k 由验证校准，未校准时为 1", "u = k × √(sum of squared parts); k comes from D's calibration, 1 until then")}</small></p>
      </details>
    </section>}
    <Time card={card} duration={duration} patch={cardsPatch} tr={tr} />
    <Judgements info={info!} patch={judgementsPatch} tr={tr} />
    <Under under={under} names={names} onSelect={onSelect} tr={tr} />
  </article>;
}

/** R1-R3 rows per keyframe (the people layer, as A copies them) -> one line per rule: the worst verdict, how many rows, the reasons. */
const byRule = (rows: any[]) => Object.values(rows.reduce((m: Record<string, any>, r: any) => {
  const x = m[r.rule] ||= { rule: r.rule, verdicts: [], reasons: new Set<string>(), n: 0 };
  x.verdicts.push(r.verdict); x.n += r.rows ?? 1; if (r.reason) x.reasons.add(r.reason);
  return m;
}, {})) as { rule: string; verdicts: string[]; reasons: Set<string>; n: number }[];

function PersonFacts({ card, names, onSelect, tr }: { card: any; names: (id: string) => string; onSelect: (id: string) => void; tr: Tr }) {
  const path = card.physical?.path_length ?? card.path_length_m, ppe = card.ppe;
  return <section className="mvp-block"><h4>{tr("轨迹", "Track")}</h4>
    {card.note && <p><small>{card.note}</small></p>}
    <table className="mvp-physical"><tbody>
      {path && <tr><th>{tr("路径长度", "path length")}</th><td><Quantity q={path} tr={tr} /></td></tr>}
      <tr><th>{tr("检测次数", "detections")}</th><td>{card.detections ?? card.time?.detections ?? "—"}</td></tr>
      {byRule(card.rules || []).map(r => <tr key={r.rule}><th>{r.rule}</th><td><Chip v={worstVerdict(r.verdicts)} tr={tr} /> <small>{r.n} {tr("行", "rows")}{r.reasons.size ? ` · ${[...r.reasons].join("; ")}` : ""}</small></td></tr>)}
      <tr><th>PPE</th><td><span className="mvp-status">{ppe?.status || tr("没问", "not asked")}{ppe?.reason ? ` (${ppe.reason})` : ""}</span></td></tr>
    </tbody></table>
    {!!card.nearest_objects?.length && <p>{tr("最近的物体", "Nearest objects")}: {card.nearest_objects.map((x: any[]) => {
      const [id, d] = [x[0], x[x.length - 1]];  // A: [id, d]; older fixtures: [id, name, d]
      return <button key={id} className="mvp-link" onClick={() => onSelect(id)}>{names(id)} {fmt(d, 2)} m</button>; })}</p>}
  </section>;
}

function Time({ card, duration, patch, tr }: { card: any; duration: number; patch?: Patch; tr: Tr }) {
  const t = card.time || {}, intervals: number[][] = t.intervals || (t.first_seen_s != null ? [[t.first_seen_s, t.last_seen_s ?? t.first_seen_s]] : []);
  const span = duration || Math.max(1, ...intervals.map(i => i[1]));
  const thumb = (e: any, label: string) => e && <button className="mvp-thumb" onClick={() => seek(e.t)}>
    {blobURL(patch, e.image) ? <img src={blobURL(patch, e.image)!} alt={label} /> : null}<span>{label} {fmt(e.t)} s</span></button>;
  return <section className="mvp-block"><h4>{tr("时间", "Time")}</h4>
    <p>{tr("首次", "first seen")} {fmt(t.first_seen_s)} s · {tr("最后", "last seen")} {fmt(t.last_seen_s)} s{t.detected_keyframes && <> · {t.detected_keyframes.length} {tr("个检测关键帧", "detected keyframes")}</>}</p>
    <div className="mvp-bar" title={tr("点一下跳到那一刻", "click to seek")} onClick={e => { const b = e.currentTarget.getBoundingClientRect(); seek((e.clientX - b.left) / b.width * span); }}>
      {intervals.map(([a, b], i) => <span key={i} style={{ left: `${a / span * 100}%`, width: `${Math.max((b - a) / span * 100, .6)}%` }} />)}
    </div>
    <p>{tr("状态", "State")}: <strong>{t.state || "—"}</strong>{t.state === "last seen at t" && t.t != null && <> {fmt(t.t)} s</>}
      {(t.last_seen_reason || t.reason) && <small> ({t.last_seen_reason || t.reason})</small>}</p>
    {t.note && <p><small>{t.note}</small></p>}
    {(t.state === "moved" || t.state === "disappeared") && t.evidence && <div className="mvp-evidence">{thumb(t.evidence.before, tr("之前", "before"))}{thumb(t.evidence.after, tr("之后", "after"))}</div>}
  </section>;
}

function Judgements({ info, patch, tr }: { info: Info; patch?: Patch; tr: Tr }) {
  if (!patch) return <section className="mvp-block"><h4>{tr("安全判断", "Safety judgement")}</h4><p><small>{tr("判断层还没到", "The judgements layer has not arrived yet")}</small></p></section>;
  return <section className="mvp-block"><h4>{tr("安全判断", "Safety judgement")}</h4>
    {!info.rows.length && <p><small>{tr("没有适用于这类对象的检查", "No check applies to this kind of object")}</small></p>}
    <ul className="mvp-judgements">{info.rows.map(r => <li key={r.id} data-v={r.verdict}>
      <p><Chip v={r.verdict} tr={tr} /> <strong>{r.title}</strong> <small>{r.check}{r.severity ? ` · ${r.severity}` : ""}</small></p>
      {r.geometry && <p>{r.geometry.quantity}{r.geometry.value != null && <>: {fmt(r.geometry.value, 2)}{r.geometry.u != null && ` ± ${fmt(r.geometry.u, 2)}`} {r.geometry.unit}</>}
        {r.geometry.threshold != null && <> {tr("对", "vs")} {r.geometry.threshold} {r.geometry.unit} ({r.geometry.direction === "max" ? tr("上限", "max") : tr("下限", "min")})</>} → {r.geometry.result}
        {r.geometry.before_forced && <small> ({tr("原为", "was")} {r.geometry.before_forced})</small>}
        {String(r.geometry.scale || "").startsWith("estimated") && <Tag>{tr("估计尺度", "estimated")}</Tag>}</p>}
      {r.vlm && <div className="mvp-vlm"><small>{tr("图像问答", "Picture")} · {r.vlm.decider}</small>
        {(r.vlm.questions ? Object.entries(r.vlm.questions).map(([question, v]: [string, any]) => ({ question, ...v })) : [r.vlm, ...(r.vlm.also || [])]).map((qq: any, j: number) => <div key={j}><small><strong>{qq.question}</strong>{qq.text ? ` "${qq.text}"` : ""}{qq.calibration ? ` · ${tr("校准", "calibration")} ${qq.calibration}` : ""}</small>
          {(qq.per_view || []).map((v: any, i: number) => <div key={i}><small>{tr("关键帧", "keyframe")} {v.keys?.join(", ")}: {v.probs ? (qq.options || r.vlm.options).map((o: string, k: number) => `${o} ${fmt(v.probs[k], 2)}`).join(" · ") : tr("未回答", "unanswered")}
            {" "}· {tr("字母概率和", "letter mass")} {fmt(v.mass, 2)}{v.p_hazard_raw != null && ` · p(hazard) ${fmt(v.p_hazard_raw, 2)}`}
            {typeof v.calibrated === "number" ? ` → ${tr("校准后", "calibrated")} ${fmt(v.calibrated, 2)}` : ` (${tr("未校准", "uncalibrated")})`}</small></div>)}</div>)}
        <small>{tr("回答", "Answer")}: <strong>{r.vlm.answer}</strong></small></div>}
      {!!r.reasons?.length && <ul className="mvp-reasons">{r.reasons.map((x: string, i: number) => <li key={i}>{x}</li>)}</ul>}
      {!!r.evidence?.length && <div className="mvp-evidence">{r.evidence.map((e: any, i: number) => <button key={i} className="mvp-thumb" onClick={() => seek(e.t)} title={tr("跳到这个关键帧", "seek to this keyframe")}>
        {blobURL(patch, e.image) ? <img src={blobURL(patch, e.image)!} alt={`${r.check} ${e.key}`} /> : null}<span>{fmt(e.t)} s</span></button>)}</div>}
      {r.rule_source && <p><small>{r.rule_source}</small></p>}
    </li>)}</ul>
  </section>;
}

function Under({ under, names, onSelect, tr }: { under: string[]; names: (id: string) => string; onSelect: (id: string) => void; tr: Tr }) {
  return under.length ? <section className="mvp-block"><h4>{tr("这个点下还有", "Also under this point")}</h4>
    <p>{under.map(id => <button key={id} className="mvp-link" onClick={() => onSelect(id)}>{names(id)}</button>)}</p></section> : null;
}

function UnknownCard({ r, under, names, onSelect, tr }: { r: NonNullable<Clicked["miss"]>; under: string[]; names: (id: string) => string; onSelect: (id: string) => void; tr: Tr }) {
  return <article className="mvp-card mvp-unknown">
    <header className="mvp-block"><h3>{tr("未知区域", "Unknown region")}</h3><p><small>{tr("不是检测到的对象：不对它是什么做任何断言", "Not a detected object: nothing is claimed about what it is")}</small></p></header>
    {r.status !== "depth" ? <section className="mvp-block"><p>{tr("这里没有三维点", "No 3D point here")}</p></section> : <section className="mvp-block">
      <table className="mvp-physical"><tbody>
        <tr><th>{tr("离相机", "distance from the camera")}</th><td><Quantity q={{ ...r.distance, unit: "m", scale: "estimated" }} tr={tr} /></td></tr>
        <tr><th>{tr("离地高度", "height above the floor")}</th><td><Quantity q={{ ...r.height, unit: "m", scale: "estimated" }} tr={tr} /></td></tr>
        <tr><th>{tr("表面", "surface")}</th><td>{r.surface.kind}{r.surface.kind === "horizontal surface" ? ` ${tr("高", "at")} ${fmt(r.height.value, 2)} m` : ""}
          {r.surface.angle_deg != null && <small> · {tr("法线离竖直", "normal from up")} {fmt(r.surface.angle_deg, 0)}°</small>}</td></tr>
        <tr><th>{tr("最近的对象", "nearest entity")}</th><td>{r.nearest ? <button className="mvp-link" onClick={() => onSelect(r.nearest.id)}>{r.nearest.name} · {fmt(r.nearest.d, 2)} m</button> : "—"}</td></tr>
      </tbody></table>
      <p><small>{tr("深度按距离 5%，加地面残差和 20% 尺度项", "u: 5% of distance for depth (height: times the ray's vertical share) + floor residual, with the 20% scale term")} · {tr("关键帧", "keyframe")} {r.frame}</small></p>
    </section>}
    <Under under={under} names={names} onSelect={onSelect} tr={tr} />
  </article>;
}

function ObjectList({ cards, infos, selected, onSelect, tr }: { cards: any[]; infos: Map<string, Info>; selected: string | null; onSelect: (id: string) => void; tr: Tr }) {
  const [verdict, setVerdict] = useState("all"), [kind, setKind] = useState("all"), [query, setQuery] = useState("");
  const kindOf = (c: any) => c.kind === "person" ? "person" : c.class?.category || "other", v = (c: any) => infos.get(c.id)?.verdict ?? "none";
  const rank = (c: any) => { const i = SEVERITY.indexOf(v(c) as any); return i < 0 ? SEVERITY.length : i; };
  const counts = cards.reduce((n: Record<string, number>, c) => ({ ...n, [v(c)]: (n[v(c)] || 0) + 1 }), {});
  const shown = cards.filter(c => (verdict === "all" || v(c) === verdict) && (kind === "all" || kindOf(c) === kind) && (!query || (c.identity?.name || c.id).toLowerCase().includes(query.toLowerCase())))
    .sort((a, b) => rank(a) - rank(b) || (a.time?.first_seen_s ?? 1e9) - (b.time?.first_seen_s ?? 1e9));
  if (!cards.length) return <p className="mvp-empty">{tr("对象卡片还没到", "The object cards have not arrived yet")}</p>;
  return <div className="mvp-list">
    <div className="mvp-filters" role="group" aria-label={tr("按判断筛选", "Filter by verdict")}>
      {["all", ...SEVERITY, "none"].map(k => <button key={k} aria-pressed={verdict === k} data-v={k} onClick={() => setVerdict(k)}>
        {k === "all" ? tr("全部", "all") : k === "none" ? tr("无检查", "no checks") : k.replace("_", " ")} {k === "all" ? cards.length : counts[k] || 0}</button>)}
    </div>
    <div className="mvp-filters">
      <select value={kind} onChange={e => setKind(e.target.value)} aria-label={tr("类别", "Kind")}>
        <option value="all">{tr("所有类别", "every kind")}</option>{[...new Set(cards.map(kindOf))].sort().map(k => <option key={k} value={k}>{k}</option>)}</select>
      <input type="search" value={query} onChange={e => setQuery(e.target.value)} placeholder={tr("按名字找", "Search names")} aria-label={tr("按名字找", "Search names")} />
      <small>{shown.length}</small>
    </div>
    <ol>{shown.map(c => <li key={c.id}><button aria-current={c.id === selected || undefined} onClick={() => { onSelect(c.id); if (c.kind === "person") seek(c.time?.first_seen_s); }}>
      <Chip v={infos.get(c.id)?.verdict} tr={tr} /><strong>{c.kind === "person" ? `${tr("人", "person")} ${c.id.slice(7)}` : c.identity?.name}</strong>
      <small>{kindOf(c)} · {fmt(c.time?.first_seen_s)} s{c.physical?.size_check?.status === "implausible" ? ` · ${tr("尺寸不合理", "implausible size")}` : ""}</small></button></li>)}</ol>
  </div>;
}

function InfoCard({ entity, tr }: { entity: any; tr: (a: string, b: string) => string }) {
  const f = entity?.fast;
  if (!f) return <aside className="live-report-card"><p>{tr("在三维或视频里点选一个物体或一个人", "Pick an object or a person in 3D or the video")}</p></aside>;
  if (f.kind === "object") {
    const size = [0, 1, 2].map(k => f.box_max_m[k] - f.box_min_m[k]), others = Object.entries(f.votes || {}).filter(([w]) => w !== f.word).sort((a: any, b: any) => b[1] - a[1]);
    return <aside className="live-report-card">
      <h3>{f.word} <small>{tr("检测词，未核", "detected word, not verified")}</small></h3>
      {!!others.length && <p>{tr("其他候选", "Other words")}: {others.map(([w, v]) => `${w} (${fmt(v)})`).join(", ")}</p>}
      <p>{tr("看到的帧数", "Frames seen")}: {f.frames} · {tr("镜头", "shot")} {f.shot + 1}</p>
      <p>{tr("框", "Box")}: {size.map(v => fmt(v, 2)).join(" × ")} m <small>{tr("估计（尺度来自地面 + 假设 1.6 m 相机高）", "estimated (scale from the floor and an assumed 1.6 m camera height)")}</small></p>
      <p>{tr("模型", "Model")}: {f.model ? <>{tr("有", "yes")} · <small>{tr("生成的显示层，不用于测量", "a generated display layer, never used to measure")}</small></> : tr("无", "none")}</p>
    </aside>;
  }
  if (f.kind === "person") return <aside className="live-report-card">
    <h3>{tr("人", "Person")} {f.id}</h3>
    <p>{fmt(f.t0)}–{fmt(f.t1)} s · {f.detections} {tr("次检测", "detections")}</p>
    {f.rules?.length ? <ul>{f.rules.map((r: any, i: number) => <li key={i}>{r.rule}: {r.verdict}</li>)}</ul>
      : <p>{tr("没有规则行", "No rule rows")}</p>}
    <p><small>{tr("尺度未测：涉及尺度的规则一律 NEEDS_REVIEW", "Scale is not measured: every rule that uses it is NEEDS_REVIEW")}</small></p>
  </aside>;
  return <aside className="live-report-card"><h3>{entity.label}</h3><p>{f.triangles} {tr("三角形", "triangles")} · {f.points} {tr("点", "points")}</p></aside>;
}

/** Spec section 7's targets for the click layers, on written times: [base layer (its v1), seconds after it]. */
const TARGETS: Record<string, Record<number, [string, number]>> = {
  pick: { 1: ["outlines", 1], 2: ["objects", 40] }, object_cards: { 1: ["objects", 10], 2: ["objects", 60], 3: ["objects", 40] }, judgements: { 1: ["objects", 12], 2: ["objects", 60] } };

function Timing({ patches, written, served, run, clicks, pickMs, pickSteps, tr }: { patches: Patch[]; written: Record<string, number>; served: Record<string, number>; run: any;
  clicks: number[]; pickMs: number | null; pickSteps: Record<string, number> | null; tr: Tr }) {
  const rows = patches.filter(p => p.layer !== "timing"), peak = (v: unknown, key: number) => <td key={key} data-over={typeof v === "number" && v > OVER || undefined}>{fmt(v)}</td>;
  const boot = run?.boot, bootS = boot?.ready_s ?? boot?.total_s, call = run?.client?.first_call;
  const at = (p?: Patch) => p && (written[p.seq] ?? p.sent_s);
  const target = (p: Patch) => {
    const rule = TARGETS[p.layer]?.[p.version], base = rule && at(patches.find(q => q.layer === rule[0]));
    if (base === undefined || !rule) return <td />;
    const limit = base + rule[1], ok = (at(p) ?? Infinity) <= limit;
    return <td data-miss={!ok || undefined} title={`${rule[0]} + ${rule[1]} s`}>≤ {fmt(limit)} {ok ? "✓" : "✗"}</td>;
  };
  return <section className="live-report-timing" aria-label={tr("计时", "Timing")}>
    <div>
      <h3>{tr("图层（从 MP4 进容器算起，秒）", "Layers (s from the MP4 in the container)")}
        {call !== undefined && <mark className="mvp-call">{call ? tr("首次调用", "first call") : tr("热调用", "warm call")}</mark>}</h3>
      <p className="mvp-clicks"><small>{tr("点击到卡片", "Click → card")}: {clicks.length ? `p50 ${fmt(quantile(clicks, .5))} ms · p95 ${fmt(quantile(clicks, .95))} ms (n ${clicks.length}, ${tr("目标 < 100 ms", "target < 100 ms")})` : tr("还没点过", "no clicks yet")}
        {pickMs != null && <> · {tr("点选层解码", "pick decode")} {fmt(pickMs, 0)} ms{pickSteps && ` (${Object.entries(pickSteps).map(([k, v]) => `${k} ${fmt(v, 0)}`).join(" · ")})`}, {tr("目标 < 300 ms", "target < 300 ms")}</>}</small></p>
      <table><thead><tr><th>{tr("层", "layer")}</th><th>{tr("发出", "sent")}</th><th>{tr("写完", "written")}</th><th>{tr("本机取到", "served")}</th><th>MB</th><th>{tr("目标", "target")}</th></tr></thead>
        <tbody>{rows.map(p => <tr key={p.seq}><td>{p.layer}{p.version > 1 ? ` v${p.version}` : ""}</td><td>{fmt(p.sent_s)}</td><td>{fmt(written[p.seq])}</td>
          <td title={tr("两个时钟：本机取到的 unix 时间减去容器的 t0", "two clocks: this machine's fetch time minus the container's t0")}>{fmt(served[p.seq] - p.t0_unix)}</td>
          <td>{fmt(Object.values(p.blobs).reduce((n, b) => n + b.bytes, 0) / 1e6)}</td>{target(p)}</tr>)}
          {bootS !== undefined && <tr className="live-report-boot"><td>{tr("冷启动（不计入）", "cold start (not counted)")}</td><td colSpan={5}>{fmt(bootS)} s</td></tr>}
        </tbody></table>
    </div>
    <div>
      <h3>{tr("阶段和显存峰值（GB，>72 标红）", "Stages and memory peaks (GB, >72 in red)")}</h3>
      {run?.stages ? <table><thead><tr><th>{tr("阶段", "stage")}</th><th>{tr("在哪", "where")}</th><th>{tr("起止", "start–end")}</th><th>GPU0</th><th>GPU1</th></tr></thead>
        <tbody>{run.stages.map((s: any, i: number) => <tr key={i}><td>{s.stage}</td><td>{s.where}</td><td>{fmt(s.start_s)}–{fmt(s.end_s)}</td>{peak(s.peak_gb?.[0], 0)}{peak(s.peak_gb?.[1], 1)}</tr>)}
          {(run.gpu_peak || []).length > 0 && <tr><td>{tr("整卡峰值", "device peak")}</td><td colSpan={2}>{run.fixture ? tr("夹具：E9 的整次峰值", "fixture: E9's run peaks") : ""}</td>
            {[0, 1].map(g => peak(run.gpu_peak.find((x: any) => x.gpu === g)?.peak_gb, g))}</tr>}
        </tbody></table> : <p>{tr("还没有计时", "No timing yet")}</p>}
      {!!run?.flags?.length && <ul className="live-report-flags">{run.flags.map((f: string) => <li key={f}>{f}</li>)}</ul>}
      {run?.fixture && <p><small>{run.fixture}</small></p>}
    </div>
  </section>;
}
