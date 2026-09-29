import { useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "./i18n";
import { mountSceneViewer, type SceneViewer } from "./viewer/native-viewer";
import { splatAnnotation } from "./viewer/splat-layer";
import { VideoMemory, VideoView, videoClock } from "./VideoView";
import { cameraPath, cameraView, currentCameras } from "./core";
import { assetURL, latest, liveDocument, poll, type Patch, type Poll } from "./live-report";
import "./report-scene.css";

const OVER = 72;  // GB: 90% of an A100-80GB
const fmt = (v: unknown, digits = 1) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";

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
  const resolve = async (id: string) => {  // blobs from the endpoint; a small inline layer (outlines under 1 MB) as an object URL
    const url = assetURL(id), inline = (docRef.current.assets.find(a => a.id === id) as any)?.inline;
    if (url) return url;
    if (inline) return URL.createObjectURL(new Blob([JSON.stringify(inline)], { type: "application/json" }));
    throw Error("unknown_asset");
  };

  const host = useRef<HTMLDivElement>(null), viewer = useRef<SceneViewer | undefined>(undefined), opened = useRef(false), follow = useRef(false);
  const [selected, setSelected] = useState<string | null>(null), [frame, setFrame] = useState<string | null>(null), [following, setFollowing] = useState(false);
  const [view, setView] = useState({ observed_surface: true, point_cloud: false, splats: true, labels: true, primitive: true });
  const [load, setLoad] = useState({ loaded: 0, total: 0 });
  useEffect(() => {
    // For the headless check: an asset fetched twice was reloaded; ready = when each asset was first on the GPU (unix s).
    const stats = { uploads: {} as Record<string, number>, ready: {} as Record<string, number>, scenes: 0, errors: [] as string[] };
    (window as any).__live = stats;
    const v = mountSceneViewer(host.current!, {
      locale: language, layers: { lighting: true, allBounds: view.primitive, editable: false, ...view },
      resolveAsset: resolve,
      onEvent: (e) => {
        if (e.type === "selectionIntent") setSelected(e.entityId || null);
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
    (window as any).__live.scenes++;
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
  return <section className="live-report">
    <header className="live-report-head">
      <strong>{tr("快速报告", "Fast report")} · {reportId}</strong>
      <span>{Object.values(layers).filter(p => p.layer !== "timing").map(p => `${p.layer}${p.version > 1 ? " v" + p.version : ""}`).join(" · ") || tr("等待第一层…", "waiting for the first layer…")}</span>
      {error && <em role="status">{error}</em>}
    </header>
    <div className="live-report-grid">
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
      <div className="live-report-side">
        <div className="live-report-video">
          <VideoView document={document} selectedId={selected} onSelect={setSelected} resolveAsset={resolve} />
          {!layers.video && <p>{tr("视频还没到", "The video has not arrived yet")}</p>}
        </div>
        <InfoCard entity={entity} tr={tr} />
        <VideoMemory document={document} onSelect={setSelected} />
      </div>
    </div>
    <Timing patches={state.patches} written={state.written} served={state.served} run={run} tr={tr} />
  </section>;
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

function Timing({ patches, written, served, run, tr }: { patches: Patch[]; written: Record<string, number>; served: Record<string, number>; run: any; tr: (a: string, b: string) => string }) {
  const rows = patches.filter(p => p.layer !== "timing"), peak = (v: unknown, key: number) => <td key={key} data-over={typeof v === "number" && v > OVER || undefined}>{fmt(v)}</td>;
  const boot = run?.boot, bootS = boot?.ready_s ?? boot?.total_s;
  return <section className="live-report-timing" aria-label={tr("计时", "Timing")}>
    <div>
      <h3>{tr("图层（从 MP4 进容器算起，秒）", "Layers (s from the MP4 in the container)")}</h3>
      <table><thead><tr><th>{tr("层", "layer")}</th><th>{tr("发出", "sent")}</th><th>{tr("写完", "written")}</th><th>{tr("本机取到", "served")}</th><th>MB</th></tr></thead>
        <tbody>{rows.map(p => <tr key={p.seq}><td>{p.layer}{p.version > 1 ? ` v${p.version}` : ""}</td><td>{fmt(p.sent_s)}</td><td>{fmt(written[p.seq])}</td>
          <td title={tr("两个时钟：本机取到的 unix 时间减去容器的 t0", "two clocks: this machine's fetch time minus the container's t0")}>{fmt(served[p.seq] - p.t0_unix)}</td>
          <td>{fmt(Object.values(p.blobs).reduce((n, b) => n + b.bytes, 0) / 1e6)}</td></tr>)}
          {bootS !== undefined && <tr className="live-report-boot"><td>{tr("冷启动（不计入）", "cold start (not counted)")}</td><td colSpan={4}>{fmt(bootS)} s</td></tr>}
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
