import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "./i18n";
import { mountSceneViewer, type SceneViewer } from "./viewer/native-viewer";
import { splatAnnotation } from "./viewer/splat-layer";
import { VideoMemory, VideoView, videoClock, type VideoPick } from "./VideoView";
import { cameraPath, cameraView, currentCameras } from "./core";
import { assetURL, chunkOrder, clickClock, emptyPick, entityInfo, fillChunk, gunzip, latest, liveDocument, onDemand, pickAt, pickChunks, pickIndexAt, pickMask, poll, readPick,
  runsMask, SEVERITY, stateAt, STATE_COLOR, unknownRegion, visitView, worstVerdict, type PickChunk,
  type Info, type Patch, type Pick, type Poll } from "./live-report";
import VisitsPanel, { VisitsTabLabel } from "./VisitsPanel";
import "./report-scene.css";

const OVER = 72;  // GB: 90% of an A100-80GB
const inflateCache = new Map<string, Promise<ArrayBuffer>>();  // pick / depth chunks by sha256, inflated once per page
const fmt = (v: unknown, digits = 1) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";
type Tr = (id: string, params?: Record<string, string | number>) => string;
const blobURL = (patch: Patch | undefined, role: string) => patch?.blobs?.[role] ? assetURL("sha256:" + patch.blobs[role].sha256) : null;
const seek = (t: unknown) => { if (typeof t === "number" && Number.isFinite(t)) window.dispatchEvent(new CustomEvent("panoptes:seek", { detail: t + 1e-3 })); };
const quantile = (xs: number[], q: number) => { const s = [...xs].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))] : undefined; };
type Clicked = VideoPick & { id: string | null; miss: ReturnType<typeof unknownRegion> | null };
/** r5b: the report's video time, at most 5 updates a second (the cards' time lines and the objects list follow the scrubber). */
function useVideoTime() {
  const [t, setT] = useState(videoClock.time);
  useEffect(() => {
    let last = 0;
    const on = (e: Event) => { const now = performance.now(); if (now - last > 200) { last = now; setT((e as CustomEvent<number>).detail); } };
    window.addEventListener("panoptes:video-time", on);
    return () => window.removeEventListener("panoptes:video-time", on);
  }, []);
  return t;
}
const STATE_CSS = (st: string) => { const c = STATE_COLOR[st]; return c ? `rgb(${c.map(v => Math.round(v * 255)).join(",")})` : st === "not observed" || st === "not seen yet" ? "#8a8f98" : "#5fb3a3"; };

/** #/live/<reportId>: a fast report as its layers arrive (fast_report/layers.py serves them on /fast). */
export default function LiveReport({ reportId }: { reportId: string }) {
  const { language, t: tr } = useI18n();
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
      } catch { if (live) setError(tr("LiveReporttsx.001")); }
      if (live) timer = window.setTimeout(tick, 500);
    };
    void tick();
    return () => { live = false; clearTimeout(timer); };
  }, [reportId]);
  const layers = useMemo(() => latest(state.patches), [state.patches]);
  const docKey = Object.values(layers).filter(p => p.layer !== "timing").map(p => p.seq).sort((a, b) => a - b).join(",");
  const [cardsLayer, setCardsLayer] = useState<any>(null);  // declared here: the document draws each card's display model (r4)
  const [od, setOd] = useState<any>(null), odKey = useRef(0);  // mvp3: the on-demand card of the last click on no entity
  const [visitShow, setVisitShow] = useState({ a: false, b: true });  // r5b: which visits the 3D pane shows (the visits panel's pick)
  const visit = useMemo(() => visitView(layers.visits?.data, visitShow.a, visitShow.b), [layers.visits?.seq, visitShow]);
  const document = useMemo(() => liveDocument(reportId, layers, cardsLayer?.cards, od?.status === "card" ? [od] : [], cardsLayer?.shots, visit),
    [reportId, docKey, cardsLayer, od?.id, visit]);  // r5b: the on-demand card's surface; the cards' shots (time reps); the visit pick
  const docRef = useRef(document); docRef.current = document;
  const sceneId = docKey + (cardsLayer ? ":cards" + cardsLayer.version : "") + (visit ? `:visits${+visitShow.a}${+visitShow.b}` : "");
  // For the headless checks: an asset fetched twice was reloaded; ready = when each asset was first on the GPU (unix s); clicks = ms
  // from pointer-down to the card in the DOM; pickDecodeMs = pick + depth fetched, inflated and indexed.
  const stats = useRef({ uploads: {} as Record<string, number>, ready: {} as Record<string, number>, scenes: 0, errors: [] as string[],
    clicks: [] as number[], pickDecodeMs: null as number | null, pickSteps: null as Record<string, number> | null, pick: null as Pick | null,
    pickDecodes: [] as { seq: number; ms: number; fetch: number; all_ms?: number; chunks?: number; reused?: number }[] }).current;
  const [pick, setPick] = useState<Pick | null>(null), [clickMs, setClickMs] = useState<number[]>([]);
  const older = useRef<Pick | null>(null);  // the previous version, for frames whose chunk of the new one is not in yet
  useEffect(() => {  // pick + depth per version, chunk by chunk (mvp2): the chunk at the video's time first, so a click works at once
    const p = layers.pick;
    if (!p) return;
    let live = true, first = true, done = 0, reused = 0;
    const t0 = performance.now(), chunks = pickChunks(p.data, p.blobs);
    const legacy = !p.data.chunks?.length, next = legacy ? null : emptyPick(p.data);
    const inflated = (role: string) => {  // by content: an unchanged chunk (depth, v1 -> v2) is fetched once per page
      const sha = p.blobs[role].sha256;
      if (inflateCache.has(sha)) reused++;
      else inflateCache.set(sha, fetch(blobURL(p, role)!, { priority: "high" } as RequestInit).then(r => r.arrayBuffer()).then(gunzip)
        .catch((e: Error) => { inflateCache.delete(sha); throw e; }));
      return inflateCache.get(sha)!;
    };
    const one = async (c: PickChunk) => {
      const tf = performance.now(), [runs, depth] = await Promise.all([inflated(c.pick), c.depth ? inflated(c.depth) : null]);
      if (!live) return;
      const t2 = performance.now(), got = legacy ? readPick(p.data, runs, depth) : (fillChunk(next!, c, runs, depth), next!);
      done++;
      if (first) {  // this version serves clicks from here; frames still pending fall back to the older one
        first = false;
        const t3 = performance.now();
        older.current = stats.pick; stats.pick = got; setPick(got);
        stats.pickDecodeMs = t3 - t0; stats.pickSteps = { fetch: t2 - tf, wait: tf - t0, inflate_index: t3 - t2 };
        stats.pickDecodes.push({ seq: p.seq, ms: t3 - t0, fetch: t2 - tf, chunks: chunks.length, at: t0, first: p.blobs[c.pick].sha256.slice(0, 12) } as any);
      }
      if (done === chunks.length) Object.assign(stats.pickDecodes[stats.pickDecodes.length - 1], { all_ms: performance.now() - t0, reused });
    };
    (async () => {
      const queue = legacy ? chunks : chunkOrder(next!, chunks, videoClock.time);
      await one(queue[0]);  // the chunk the viewer is looking at, alone
      const rest = queue.slice(1);
      await Promise.all([0, 1].map(async () => { for (let c = rest.shift(); c && live; c = rest.shift()) await one(c); }));
    })().catch((e: Error) => { stats.errors.push("pick: " + e.message); if (live) setError("pick: " + e.message); });
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
    const glb = (docRef.current.assets.find(a => a.id === id) as any)?.inlineGlb;  // r5b: an on-demand card's surface (base64 GLB)
    if (glb) return URL.createObjectURL(new Blob([Uint8Array.from(atob(glb), c => c.charCodeAt(0))], { type: "model/gltf-binary" }));
    if (inline) return URL.createObjectURL(new Blob([JSON.stringify(inline)], { type: "application/json" }));
    if (id.startsWith("frame:")) {  // r5b integrate (overlay): a keyframe's photo, cut from the report's own video at its time
      const t = (docRef.current.assets.find(a => a.id === id) as any)?.metadata?.videoTimestamp;
      if (Number.isFinite(t)) return frameURL(t);
    }
    throw Error("unknown_asset");
  };
  const grabber = useRef<{ v?: HTMLVideoElement; busy: Promise<unknown> }>({ busy: Promise.resolve() });
  const frameURL = (t: number) => {  // one hidden video element, one seek at a time -> a JPEG object URL of the frame at t
    const g = grabber.current, src = assetURL((docRef.current.annotations.find((a: any) => a.kind === "video_replay") as any)?.videoAssetId || "");
    const run = async () => {
      if (!src) throw Error("no_video");
      if (!g.v || g.v.src !== new URL(src, location.href).href) {
        const v = window.document.createElement("video");
        v.muted = true; v.preload = "auto"; v.src = src;
        await new Promise((ok, fail) => { v.addEventListener("loadeddata", ok, { once: true }); v.addEventListener("error", fail, { once: true }); });
        g.v = v;
      }
      const v = g.v;
      v.currentTime = t + 1e-3;
      await new Promise(ok => v.addEventListener("seeked", ok, { once: true }));
      const c = window.document.createElement("canvas");
      c.width = v.videoWidth; c.height = v.videoHeight; c.getContext("2d")!.drawImage(v, 0, 0);
      return URL.createObjectURL(await new Promise<Blob>(ok => c.toBlob(b => ok(b!), "image/jpeg", .9)));
    };
    const job = g.busy.then(run, run);
    g.busy = job.catch(() => undefined);
    return job;
  };

  const host = useRef<HTMLDivElement>(null), viewer = useRef<SceneViewer | undefined>(undefined), opened = useRef(false), follow = useRef(false);
  const [selected, setSelected] = useState<string | null>(null), [frame, setFrame] = useState<string | null>(null), [following, setFollowing] = useState(false);
  const [clicked, setClicked] = useState<Clicked | null>(null), [tab, setTab] = useState<"card" | "objects" | "memory" | "visits">("card");
  const choose = (id: string | null) => { odKey.current++; setOd(null); setSelected(id); setClicked(null); if (id) setTab("card"); };  // from 3D, the list, a card link
  const onPick = (p: VideoPick) => {  // a video click: the pick layer decides; before it lands, the smallest outline under the point
    let use = pick, hit = pick ? pickAt(pick, p.t, p.x, p.y) : null;
    if (hit?.pending && older.current) { use = older.current; hit = pickAt(use, p.t, p.x, p.y); }  // its chunk is on the way
    const id = hit && !hit.pending ? hit.id : p.under[0] ?? null;
    const miss = !id && use && !hit?.pending ? unknownRegion(use, layers.cameras?.data, cardsLayer, p.t, p.x, p.y) : null;
    setSelected(id); setClicked({ ...p, id, miss, under: p.under.filter(u => u !== id) }); setTab("card");
    const k = ++odKey.current, index = miss && use ? pickIndexAt(use.data.frames, p.t) : -1, t0 = performance.now();
    setOd(index >= 0 ? { pending: true } : null);
    if (index >= 0) onDemand(reportId, index, p.x, p.y).then(card => {  // mvp3 D4 (b): the report container segments, lifts and names it
      if (odKey.current !== k) return;
      setOd({ ...card, ms: performance.now() - t0 });
      if (card.status === "entity") { setSelected(card.entity); setClicked(c => c && { ...c, id: card.entity, miss: null }); }  // the point was at its edge
    }).catch((e: Error) => { if (odKey.current === k) setOd({ error: e.message }); });
  };
  useEffect(() => {  // mvp3: the report container stays up while this viewer is open (FastReport.alive; it scales down 60 s after the last)
    const beat = () => { if (window.document.visibilityState === "visible") fetch("/fast/alive", { cache: "no-store" }).catch(() => undefined); };
    beat();
    const timer = window.setInterval(beat, 30000);
    return () => clearInterval(timer);
  }, []);
  useLayoutEffect(() => {  // the card is in the DOM now: pointer-down -> here is the click latency
    if (!clickClock.t0) return;
    stats.clicks.push(performance.now() - clickClock.t0); clickClock.t0 = 0;
  }, [clicked]);
  useEffect(() => { if (stats.clicks.length !== clickMs.length) setClickMs([...stats.clicks]); }, [clicked]);
  const odMask = useMemo(() => od?.status === "card" && od.mask ? runsMask(od.mask) : null, [od]);
  const highlight = useMemo(() => pick && (selected || odMask) ? (t: number) => {
    const i = pickIndexAt(pick.data.frames, t), o = older.current;
    if (!selected) return i === od.pick_index ? odMask : null;  // the on-demand mask, on its own keyframe
    return pickMask(pick, i, selected) ?? (o && i >= 0 && !pick.ready[i] ? pickMask(o, pickIndexAt(o.data.frames, t), selected) : null);
  } : undefined, [pick, selected, odMask]);
  // r5b integrate: no boxes and no always-on names (the hovered or selected object's name only); 'overlay': the 3D pane shows the video's
  // keyframe with every model drawn from that keyframe's camera (the room's own surface hidden), following the video's time
  const [view, setView] = useState({ observed_surface: true, point_cloud: false, splats: true, labels: true, primitive: true, overlay: false });
  const layersOf = (v: typeof view) => ({ ...v, allBounds: false, showBounds: false, opacity: .85,
    observed_surface: v.observed_surface && !v.overlay, point_cloud: v.point_cloud && !v.overlay });
  const [load, setLoad] = useState({ loaded: 0, total: 0 });
  useEffect(() => {
    (window as any).__live = stats;
    const v = mountSceneViewer(host.current!, {
      locale: language, layers: { lighting: true, editable: false, ...layersOf(view) },
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
  useEffect(() => { viewer.current?.setLocale(language); }, [language]);
  useEffect(() => {
    const v = viewer.current;
    if (!v) return;
    stats.scenes++;
    v.setScene({ id: sceneId, document }).catch((e: Error) => setError(e.message));
    // The first cameras open a free view of the longest shot, never the (image-less) camera view.
    if (!opened.current && document.cameras.length) { opened.current = true; v.setCamera({ mode: "free" }); setFrame(currentCameras(document)[0].coordinateFrameId); }
  }, [document]);
  useEffect(() => { viewer.current?.setLayers(layersOf(view)); }, [view]);
  useEffect(() => { viewer.current?.setSelection({ entityId: selected }); }, [selected]);
  const [inset, setInset] = useState<string | null>(null);
  useEffect(() => {  // r4 (models): the selected object's model alone in a corner of the 3D pane (the viewer's own preview capture), its shot opened
    setInset(null);
    const e = docRef.current.entities.find(x => x.id === selected) as any;
    if (!selected || !(e?.fast?.display_model || e?.fast?.model)) return;
    const frameId = "shot-" + e.fast.shot;
    if (frame !== frameId) {
      setFrame(frameId); follow.current = false; setFollowing(false);
      viewer.current?.setCamera({ mode: "free", cameraId: currentCameras(docRef.current).find(c => c.coordinateFrameId === frameId)?.id });
    }
    let live = true, tries = 0;
    const grab = () => {
      if (!live) return;
      const png = viewer.current?.capturePreview(selected, "free", sceneId, frameId);
      if (png) setInset(png); else if (++tries < 40) window.setTimeout(grab, 250);  // until its model is on the GPU
    };
    grab();
    return () => { live = false; };
  }, [selected, sceneId]);
  const splat = splatAnnotation(document);
  useEffect(() => {
    viewer.current?.setSplats(splat ? { key: splat.assetId, url: assetURL(splat.assetId)!, count: splat.count, coordinateFrameId: splat.coordinateFrameId } : null);
  }, [splat?.assetId]);
  // The report's one clock is the video: the camera path marks it, and "follow" stands the view where the camera stood.
  const lastPhoto = useRef<string | null>(null);
  useEffect(() => {
    const place = (time: number) => {
      viewer.current?.setLayers({ time });
      if (view.overlay) {  // r5b integrate: the keyframe nearest the video's time, in any shot, as the photo under the models
        const cams = docRef.current.cameras as any[], at = (c: any) => (docRef.current.assets.find(a => a.id === c.imageId) as any)?.metadata?.videoTimestamp ?? 1e9;
        const near = cams.reduce((a: any, b: any) => !a || Math.abs(at(b) - time) < Math.abs(at(a) - time) ? b : a, null);
        if (near && near.id !== lastPhoto.current) { lastPhoto.current = near.id; setFrame(near.coordinateFrameId); viewer.current?.setCamera({ mode: "photo", cameraId: near.id }); }
        return;
      }
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
  }, [frame, following, view.overlay]);
  useEffect(() => { if (!view.overlay && lastPhoto.current) { lastPhoto.current = null; viewer.current?.setCamera({ mode: "free" }); } }, [view.overlay]);

  const entity = document.entities.find(e => e.id === selected) as any;
  const labels = [...new Set(Object.values(layers).flatMap(p => p.labels || []))];
  const shots = document.coordinateFrames.map(f => ({ id: f.id, cameras: document.cameras.filter(c => c.coordinateFrameId === f.id) }));
  const run = state.run || layers.timing?.data.run;
  const fixture = ["pick", "object_cards", "judgements"].map(k => layers[k]?.data?.fixture).find(Boolean);
  const duration = layers.video ? layers.video.data.frames / layers.video.data.fps : 0;
  const names = (id: string) => infos.get(id)?.card?.identity?.name || (document.entities.find(e => e.id === id) as any)?.label || id;
  return <section className="live-report">
    <header className="live-report-head">
      <strong>{tr("LiveReporttsx.002")} · {reportId}</strong>
      <span>{Object.values(layers).filter(p => p.layer !== "timing").map(p => `${p.layer}${p.version > 1 ? " v" + p.version : ""}`).join(" · ") || tr("LiveReporttsx.003")}</span>
      {error && <em role="status">{error}</em>}
      {fixture && <mark className="live-report-fixture" title={fixture}>{tr("LiveReporttsx.004")}</mark>}
    </header>
    <div className="live-report-grid">
      <div className="live-report-main">
        <div className="live-report-video">
          <VideoView document={document} selectedId={selected} onSelect={choose} resolveAsset={resolve} onPick={onPick} highlight={highlight}
            marker={clicked?.miss ? { x: clicked.x, y: clicked.y } : null} />
          {!layers.video && <p>{tr("LiveReporttsx.005")}</p>}
        </div>
        <div className="live-report-3d">
          <div className="live-report-tools">
            {shots.map(s => <button key={s.id} aria-pressed={frame === s.id} onClick={() => { setFrame(s.id); follow.current = false; setFollowing(false);
              viewer.current?.setCamera({ mode: "free", cameraId: s.cameras[0]?.id }); }}>{tr("LiveReporttsx.006")} {Number(s.id.slice(5)) + 1} ({s.cameras.length})</button>)}
            {(["overlay", "observed_surface", "point_cloud", "primitive", "labels", ...(splat ? ["splats"] : [])] as const).map(k => <label key={k}>
              <input type="checkbox" checked={(view as any)[k]} onChange={e => setView(v => ({ ...v, [k]: e.target.checked }))} />
              {({ overlay: tr("LiveReporttsx.007"), observed_surface: tr("LiveReporttsx.008"), point_cloud: tr("LiveReporttsx.009"),
                 primitive: tr("LiveReporttsx.010"), labels: tr("LiveReporttsx.011"), splats: tr("LiveReporttsx.012") } as any)[k]}</label>)}
            {!!shots.length && <label><input type="checkbox" checked={following} onChange={e => { follow.current = e.target.checked; setFollowing(e.target.checked); }} />{tr("LiveReporttsx.013")}</label>}
            <small>{load.total ? `${load.loaded}/${load.total}` : ""}</small>
            <small className="live-report-legend" title={tr("LiveReporttsx.014")}>
              {(["appeared", "moved", "moved away", "disappeared"] as const).map(k => <span key={k} style={{ color: STATE_CSS(k), marginLeft: 6 }}>■ {k}</span>)}</small>
          </div>
          <div ref={host} className="live-report-viewer" />
          {inset && <figure className="live-report-model-inset"><img src={inset} alt={tr("LiveReporttsx.015")} />
            <figcaption>{tr("LiveReporttsx.016")}</figcaption></figure>}
          {!!labels.length && <ul className="live-report-labels">{labels.map(l => <li key={l}>{l}</li>)}</ul>}
        </div>
      </div>
      <div className="live-report-side">
        <nav className="mvp-tabs" role="tablist">
          {([["card", tr("LiveReporttsx.017")], ["objects", `${tr("objects")} (${cardsLayer?.cards?.length ?? 0})`], ["memory", tr("LiveReporttsx.018")],
            ...(layers.visits ? [["visits", <VisitsTabLabel key="v" patch={layers.visits} tr={tr} />]] : [])] as [typeof tab, React.ReactNode][]).map(([k, label]) =>
            <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}>{label}</button>)}
        </nav>
        <div className="mvp-panel">
          {tab === "card" && (clicked?.miss ? <UnknownCard r={clicked.miss} od={od} under={clicked.under} names={names} onSelect={choose} tr={tr} />
            : <Card id={selected} info={infos.get(selected || "")} entity={entity} under={clicked?.id === selected ? clicked?.under || [] : []} names={names}
                judgementsPatch={layers.judgements} cardsPatch={layers.object_cards} duration={duration} onSelect={choose} tr={tr}
                sam={samFor(layers.models, cardsLayer?.aliases, selected)} surface={(layers.surfaces?.data.surfaces || []).find((r: any) => r.object === selected)} />)}
          {tab === "objects" && <ObjectList cards={cardsLayer?.cards || []} infos={infos} selected={selected} onSelect={choose} tr={tr} />}
          {tab === "memory" && <VideoMemory document={document} onSelect={choose} />}
          {tab === "visits" && layers.visits && <VisitsPanel patch={layers.visits} show={visitShow} setShow={setVisitShow} selected={selected}
            onSelect={id => { setSelected(id); setClicked(null); }} onSeek={seek} tr={tr} />}
        </div>
      </div>
    </div>
    <Timing patches={state.patches} written={state.written} served={state.served} run={run} clicks={clickMs} pickMs={stats.pickDecodeMs} pickSteps={stats.pickSteps} tr={tr} />
  </section>;
}

const Chip = ({ v, tr }: { v: string | null | undefined; tr: Tr }) =>
  <span className="mvp-chip" data-v={v || "none"}>{v ? v.replace("_", " ") : tr("LiveReporttsx.019")}</span>;
const Tag = ({ children }: { children: React.ReactNode }) => <small className="mvp-tag">{children}</small>;

/** value ± u unit, with its bound, scale tag and note; or a not-observed / not-measurable status with its reason. */
function Quantity({ q, tr }: { q: any; tr: Tr }) {
  if (!q || q.value === undefined || q.value === null) return <span className="mvp-status">{q?.status || tr("LiveReporttsx.020")}{q?.reason ? ` (${q.reason})` : ""}
    {q?.visible && <> · {tr("LiveReporttsx.021")} <Quantity q={q.visible} tr={tr} /></>}</span>;  // r4: a one-side depth's lower bound
  const digits = q.unit === "deg" || q.unit === "°" ? 1 : 2, v = Array.isArray(q.value) ? `(${q.value.map((x: number) => fmt(x, digits)).join(", ")})` : fmt(q.value, digits);
  const st = q.bound || q.status, scale = String(q.scale || "");  // A: status "at least" | "at most" | "needs review" with a value
  // mvp2/integrate: a bound whose u is 0 (an unresolved size: the end of its interval) shows no '± 0.00'
  return <span>{st === "at least" ? "≥ " : st === "at most" ? "≤ " : ""}{v}{q.bound && !q.u ? "" : <> ± {fmt(q.u, digits)}</>} {q.unit === "deg" ? "°" : q.unit}
    {scale.startsWith("estimated") ? <Tag>{tr("LiveReporttsx.022")}</Tag> : scale ? <Tag>{tr("LiveReporttsx.023")}</Tag> : null}
    {st === "needs review" && <Tag>{tr("LiveReporttsx.024")}</Tag>}
    {q.path && <small> · {tr("LiveReporttsx.025")}: {q.path}</small>}
    {q.reason && st !== "needs review" && <small> · {q.reason}</small>}
    {q.note && <small className="mvp-note"> · {q.note}</small>}</span>;
}

const PHYSICAL: [string, string][] = [["top_above_floor", "LiveReporttsx.text001"], ["base_above_floor", "live.physical.base_above_floor"],
  ["height", "LiveReporttsx.text002"], ["width", "live.physical.width"], ["depth", "live.physical.depth"], ["visible_length", "live.physical.visible_length"], ["footprint_m2", "live.physical.footprint_m2"],
  ["nearest_walked_path", "live.physical.nearest_walked_path"], ["position_xy", "live.physical.position_xy"],
  ["principal_axis_tilt_deg", "live.physical.principal_axis_tilt_deg"], ["planar_slope_deg", "live.physical.planar_slope_deg"]];
const SIZE_FIELDS = new Set(["top_above_floor", "base_above_floor", "height", "width", "depth", "visible_length", "footprint_m2", "position_xy", "nearest_walked_path"]);

/** r5b (models): the generator's word on a card: its accepted model (or a look-alike's copy), its 'tried' row, and the router's reason
 *  (models.data.routes: why the card has, or has not, a generated model). Merged-away ids follow the aliases. */
function samFor(models: Patch | undefined, aliases: Record<string, string> | undefined, id: string | null) {
  if (!models || !id) return null;
  const mine = (o: string) => o === id || aliases?.[o] === id, got = (models.data.models || []).find((m: any) => mine(m.object));
  const generator = models.data.generator || "SAM 3D s1cfg12", route = models.data.routes?.[id];
  return got ? { accepted: true, iou: got.gate?.silhouette_iou, reuse_of: got.reuse_of, check: got.gate?.check, final: models.data.final, generator, route }
    : { ...(models.data.tried || []).find((t: any) => mine(t.object)), final: models.data.final, generator, route };
}

const KIND: Record<string, string> = { box: "live.kind.box", cylinder: "live.kind.cylinder", plane: "live.kind.plane", "open frame": "live.kind.open frame",
  "observed surface": "live.kind.observed surface" };

/** r5b (models): the card's display model in a few lines: which tier is drawn (a generated mesh, a checked primitive, or the observed
 *  surface), why, and what the video saw of it. A person has none. */
function ModelLine({ model, sam, surface, tr }: { model: any; sam?: any; surface?: any; tr: Tr }) {
  if (!model) return null;
  const pct = (v: number) => `${Math.round(v * 100)}%`, kind = (k: string) => (KIND[k] ? tr(KIND[k]) : k);
  const shown = sam?.accepted ? "generated" : model.tier === "primitive" ? "primitive" : model.tier === 0 ? "observed" : "none";
  return <section className="mvp-block mvp-model" data-tier={shown}><h4>{tr("measure.model")} <Tag>{tr("LiveReporttsx.026")}</Tag></h4>
    {shown === "generated" && <p>{tr("LiveReporttsx.027")} · {sam.generator}
      {sam.reuse_of ? <> · {tr("LiveReporttsx.028")} {sam.reuse_of} <small>({sam.check})</small></> : <> · {tr("LiveReporttsx.029")} {fmt(sam.iou, 2)}</>}
      <br /><small>{tr("LiveReporttsx.030")}</small></p>}
    {shown === "primitive" && <p>{kind(model.kind)} · {model.chosen_by}{" · "}{tr("LiveReporttsx.031")} {pct(model.seen_share ?? 0)}
      <small> ({tr("LiveReporttsx.032")})</small>{model.depth && <><br /><small>{model.depth}</small></>}</p>}
    {shown === "observed" && <p>{tr("live.kind.observed surface")}{surface?.triangles ? ` · ${surface.triangles} ${tr("LiveReporttsx.033")}` : ` · ${tr("LiveReporttsx.034")}`}
      {" · "}{tr("LiveReporttsx.035")}
      <br /><small>{model.chosen_by}</small>{model.depth && <><br /><small>{model.depth}</small></>}</p>}
    {shown === "none" && <p><small>{model.reason}</small></p>}
    {sam?.route && !sam.accepted && <p><small>{tr("LiveReporttsx.036")}: {sam.route[0] === "generated" ? (sam.reasons ? `${tr("LiveReporttsx.037")}: ${sam.reasons.slice(0, 2).join("; ")}`
      : sam.why ? `${tr("LiveReporttsx.038")}: ${sam.why}` : sam.final ? tr("LiveReporttsx.039") : tr("LiveReporttsx.040")) : sam.route[1]}</small></p>}
  </section>;
}

/** r5 (models): the observed points' planar parts (fast_report.surface.planar_parts): each part's angle to the floor and the angle
 *  between touching parts, measured on what the video saw (never on a generated model). r5b: parts are lettered (A, B, ...). */
function SurfaceParts({ sp, tr }: { sp: any; tr: Tr }) {
  if (!sp.parts) return <tr><th>{tr("LiveReporttsx.041")}</th><td><span className="mvp-status">{sp.status}</span> <small>{sp.reason}</small></td></tr>;
  const name = (i: number) => sp.parts[i]?.name || String.fromCharCode(65 + i);
  return <>{sp.parts.map((p: any, i: number) => <tr key={"p" + i} data-part={name(i)}><th>{tr("LiveReporttsx.text003", {p0: name(i)})}</th>
    <td><Quantity q={p.tilt_deg} tr={tr} /> <small>· {fmt(p.area_m2, 2)} m² · {Math.round(p.share * 100)}% {tr("LiveReporttsx.042")}</small></td></tr>)}
    {(sp.bends || []).map((b: any, i: number) => <tr key={"b" + i}><th>{tr("LiveReporttsx.text004", {p0: name(b.parts[0]), p1: name(b.parts[1])})}</th>
      <td><Quantity q={b.angle_deg} tr={tr} /></td></tr>)}</>;
}

function Card({ id, info, entity, under, names, judgementsPatch, cardsPatch, duration, onSelect, tr, sam, surface }: {
  id: string | null; info?: Info; entity: any; under: string[]; names: (id: string) => string; judgementsPatch?: Patch; cardsPatch?: Patch;
  duration: number; onSelect: (id: string) => void; tr: Tr; sam?: any; surface?: any;
}) {
  if (!id) return <p className="mvp-empty">{tr("LiveReporttsx.043")}</p>;
  const card = info?.card;
  if (!card) return <><InfoCard entity={entity} tr={tr} />{!!info?.rows.length && <Judgements info={info} patch={judgementsPatch} tr={tr} />}<Under under={under} names={names} onSelect={onSelect} tr={tr} /></>;
  const idn = card.identity || {}, ph = card.physical || {}, bad = ph.size_check?.status === "implausible";
  const quantities = Object.values(ph).filter((q: any) => q && typeof q === "object" && "u" in q) as any[];
  return <article className="mvp-card" data-kind={card.kind}>
    <header className="mvp-block mvp-identity">
      <h3>{card.kind === "person" ? `${tr("LiveReporttsx.044")} ${card.id.replace("person:", "")}` : idn.name} <Chip v={info!.verdict} tr={tr} /></h3>
      <p>{idn.confidence == null ? tr("LiveReporttsx.045") : `${Math.round(idn.confidence * 100)}%`}{" "}
        <Tag>{idn.calibrated ? tr("LiveReporttsx.046") : tr("LiveReporttsx.047")}</Tag> · {tr("LiveReporttsx.048")} {idn.decided_by || "—"} · <Tag>{idn.label || tr("LiveReporttsx.049")}</Tag></p>
      {(idn.status || idn.note) && <p><small>{[idn.status, idn.note].filter(Boolean).join(" · ")}</small></p>}
      {!!idn.alternatives?.length && <p><small>{tr("LiveReporttsx.050")}: {idn.alternatives.map(([w, p]: [string, number]) => `${w} ${fmt(p, 2)}`).join(" · ")}</small></p>}
      {!!idn.candidates_struck?.length && <p><small>{tr("LiveReporttsx.051")}: {idn.candidates_struck.map(([w, why]: [string, string]) => `${w} (${why})`).join(" · ")}</small></p>}
    </header>
    <section className="mvp-block"><h4>{tr("LiveReporttsx.052")}</h4>
      <p>{card.class?.category || "other"} · {card.class?.mobility || "—"} <small>({card.class?.mobility_source || "—"}{card.class?.reason ? `: ${card.class.reason}` : ""})</small></p></section>
    {card.kind === "person" ? <PersonFacts card={card} names={names} onSelect={onSelect} tr={tr} /> : <section className="mvp-block">
      <h4>{tr("LiveReporttsx.053")} <small>{tr("LiveReporttsx.054")}</small></h4>
      <p><small>{tr("LiveReporttsx.055")}: {ph.level || quantities[0]?.level || "2d only"}{ph.reason ? ` (${ph.reason})` : ""} · {card.views?.n ?? 0} {tr("LiveReporttsx.056")} · {Math.max(0, ...quantities.map(q => q.n_subsets || 0))} {tr("LiveReporttsx.057")}
        {card.views?.distance_m && <> · {fmt(card.views.distance_m[0])}–{fmt(card.views.distance_m[1])} m {tr("LiveReporttsx.058")}</>} · {tr("LiveReporttsx.059")} {fmt(card.views?.azimuth_spread_deg, 0)}°</small></p>
      {bad && <p className="mvp-warn">{ph.size_check.reason}</p>}
      {ph.fragmented_support && <p className="mvp-warn">{tr("LiveReporttsx.060")}</p>}
      <table className="mvp-physical"><tbody>
        {PHYSICAL.filter(([k]) => ph[k]).map(([k, label]) => <tr key={k} data-implausible={(bad || ph.fragmented_support) && SIZE_FIELDS.has(k) || undefined}><th>{tr(label)}</th><td><Quantity q={ph[k]} tr={tr} /></td></tr>)}
        {ph.primitive && <tr><th>{tr("LiveReporttsx.061")}</th><td>{ph.primitive.kind}: {ph.primitive.accepted ? tr("LiveReporttsx.062") : tr("LiveReporttsx.063")}
          {ph.primitive.reason && <small> ({ph.primitive.reason})</small>}</td></tr>}
        {(ph.surface_parts || surface?.parts) && <SurfaceParts sp={ph.surface_parts || surface.parts} tr={tr} />}
        {ph.walkway && <tr><th>{tr("LiveReporttsx.064")}</th><td><span className="mvp-status">{ph.walkway.status}</span></td></tr>}
        {ph.size_check && <tr><th>{tr("LiveReporttsx.065")}</th><td>{ph.size_check.status}{ph.size_check.class_range_m && <small> ({ph.size_check.class || tr("LiveReporttsx.066")}: {ph.size_check.class_range_m.join("–")} m{tr("LiveReporttsx.067")}{ph.size_check.measured_m != null ? `, measured ${fmt(ph.size_check.measured_m, 2)}${ph.size_check.measured_u_m != null ? ` ± ${fmt(ph.size_check.measured_u_m, 2)}` : ""} m` : ""})</small>}</td></tr>}
      </tbody></table>
      <details className="mvp-parts"><summary>{tr("LiveReporttsx.068")}</summary>
        <table><tbody>{PHYSICAL.filter(([k]) => ph[k]?.parts).map(([k, label]) => <tr key={k}><th>{tr(label)}</th>
          <td>{Object.entries(ph[k].parts).map(([p, v]) => `${p} ${fmt(v, 3)}`).join(" · ")}{ph[k].subsets?.length ? ` · ${tr("LiveReporttsx.069")} ${ph[k].subsets.map((v: number) => fmt(v, 2)).join(" / ")}` : ""}</td></tr>)}</tbody></table>
        <p><small>{tr("LiveReporttsx.070")}</small></p>
      </details>
    </section>}
    {(card.kind === "object" || card.model) && <ModelLine model={card.model} sam={sam} surface={surface} tr={tr} />}
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

const PERSON: [string, string][] = [["position_xy", "live.person.position_xy"],
  ["top_above_floor", "live.person.top_above_floor"], ["stature", "live.person.stature"], ["foot_height", "live.person.foot_height"],
  ["moved", "live.person.moved"]];

function PersonFacts({ card, names, onSelect, tr }: { card: any; names: (id: string) => string; onSelect: (id: string) => void; tr: Tr }) {
  const path = card.physical?.path_length ?? card.path_length_m, ppe = card.ppe, sup = card.identity?.support;
  return <section className="mvp-block"><h4>{tr("LiveReporttsx.071")}</h4>
    {card.note && <p><small>{card.note}</small></p>}
    {card.identity?.note && <p className={card.identity?.name?.startsWith("person?") ? "mvp-warn" : undefined}><small>{card.identity.note}</small></p>}
    <table className="mvp-physical"><tbody>
      {path && <tr><th>{tr("LiveReporttsx.072")}</th><td><Quantity q={path} tr={tr} /></td></tr>}
      {PERSON.filter(([k]) => card.physical?.[k]).map(([k, label]) => <tr key={k}><th>{tr(label)}</th><td><Quantity q={card.physical[k]} tr={tr} /></td></tr>)}
      {sup && <tr><th>{tr("LiveReporttsx.073")}</th><td>{sup.status} · {tr("LiveReporttsx.074")} <Quantity q={{ ...sup.bottom_above_floor, unit: "m" }} tr={tr} /></td></tr>}
      <tr><th>{tr("LiveReporttsx.075")}</th><td>{card.detections ?? card.time?.detections ?? "—"}</td></tr>
      {byRule(card.rules || []).map(r => <tr key={r.rule}><th>{r.rule}</th><td><Chip v={worstVerdict(r.verdicts)} tr={tr} /> <small>{r.n} {tr("LiveReporttsx.076")}{r.reasons.size ? ` · ${[...r.reasons].join("; ")}` : ""}</small></td></tr>)}
      <tr><th>{tr("live.ppe")}</th><td><span className="mvp-status">{ppe?.status || tr("LiveReporttsx.077")}{ppe?.reason ? ` (${ppe.reason})` : ""}</span></td></tr>
    </tbody></table>
    {!!card.nearest_objects?.length && <p>{tr("LiveReporttsx.078")}: {card.nearest_objects.map((x: any) => {
      const id = Array.isArray(x) ? x[0] : x.id;  // mvp2: {id, distance: {value, u, ...}}; older cards: [id, d] (no u: not shown as a number)
      return <button key={id} className="mvp-link" onClick={() => onSelect(id)}>{names(id)}{!Array.isArray(x) && x.distance ? <> <Quantity q={x.distance} tr={tr} /></> : null}</button>; })}</p>}
  </section>;
}

function Time({ card, duration, patch, tr }: { card: any; duration: number; patch?: Patch; tr: Tr }) {
  const t = card.time || {}, intervals: number[][] = t.intervals || (t.first_seen_s != null ? [[t.first_seen_s, t.last_seen_s ?? t.first_seen_s]] : []);
  const span = duration || Math.max(1, ...intervals.map(i => i[1]));
  const now = useVideoTime(), at = stateAt(card, now), tl = t.timeline;
  const thumb = (e: any, label: string) => e && <button className="mvp-thumb" onClick={() => seek(e.t)}>
    {blobURL(patch, e.image) ? <img src={blobURL(patch, e.image)!} alt={label} /> : null}<span>{label} {fmt(e.t)} s</span></button>;
  const q = (v: any, d = 2) => Array.isArray(v) ? `${Array.isArray(v[0]) ? "(" + v[0].map((x: number) => fmt(x, d)).join(", ") + ")" : fmt(v[0], d)} ± ${fmt(v[1], d)}` : "—";
  return <section className="mvp-block" data-state-at={at.state}><h4>{tr("LiveReporttsx.079")}</h4>
    <p>{tr("LiveReporttsx.080")} {fmt(t.first_seen_s)} s · {tr("LiveReporttsx.081")} {fmt(t.last_seen_s)} s{t.detected_keyframes && <> · {t.detected_keyframes.length} {tr("LiveReporttsx.082")}</>}</p>
    <div className="mvp-bar" title={tr("LiveReporttsx.083")} onClick={e => { const b = e.currentTarget.getBoundingClientRect(); seek((e.clientX - b.left) / b.width * span); }}>
      {(tl?.intervals || []).length ? tl.intervals.map((iv: any, i: number) => <span key={i} data-state={iv.state} title={`${iv.state} ${fmt(iv.t[0])}–${fmt(iv.t[1])} s${iv.reason ? ": " + iv.reason : ""}`}
        style={{ left: `${iv.t[0] / span * 100}%`, width: `${Math.max((iv.t[1] - iv.t[0]) / span * 100, .6)}%`, background: STATE_CSS(iv.state === "moved" && String(iv.reason || "").startsWith("moved to") ? "moved away" : iv.state),
          opacity: iv.state === "not observed" ? .45 : 1 }} />)
        : intervals.map(([a, b], i) => <span key={i} style={{ left: `${a / span * 100}%`, width: `${Math.max((b - a) / span * 100, .6)}%` }} />)}
      <i className="mvp-bar-now" style={{ left: `${Math.min(now / span, 1) * 100}%`, position: "absolute", top: 0, bottom: 0, width: 2, background: "#fff" }} />
    </div>
    <p>{tr("LiveReporttsx.084")} {fmt(now)} s: <strong>{at.state}</strong>{at.reason && <small> ({at.reason})</small>}
      {at.interval?.v && <small> · {tr("LiveReporttsx.085")} {q(at.interval.v.position_xy)} m · {tr("live.physical.height")} {q(at.interval.v.height)} m</small>}</p>
    <p>{tr("LiveReporttsx.086")}: <strong>{t.state || "—"}</strong>{t.state === "last seen at t" && t.t != null && <> {fmt(t.t)} s</>}
      {(t.last_seen_reason || t.reason) && <small> ({t.last_seen_reason || t.reason})</small>}
      {t.moved && <small> · {t.moved.to ? tr("LiveReporttsx.087") + " " + t.moved.to : tr("LiveReporttsx.088") + " " + t.moved.from}</small>}</p>
    {t.note && <p><small>{t.note}</small></p>}
    {(t.state === "moved" || t.state === "disappeared") && t.evidence && <div className="mvp-evidence">{thumb(t.evidence.before, tr("LiveReporttsx.089"))}{thumb(t.evidence.after, tr("LiveReporttsx.090"))}</div>}
    {tl && <details className="mvp-parts" open={!!tl.changes?.length}><summary>{tr("LiveReporttsx.091")} · {tl.intervals.length} {tr("LiveReporttsx.092")} · {tl.windows.length} {tr("LiveReporttsx.093")}</summary>
      <table><thead><tr><th>{tr("LiveReporttsx.094")}</th><th>{tr("LiveReporttsx.095")}</th><th>{tr("LiveReporttsx.096")}</th><th>{tr("LiveReporttsx.097")}</th><th>{tr("LiveReporttsx.098")}</th><th>{tr("live.physical.height")}</th><th>{tr("live.physical.width")}</th></tr></thead>
        <tbody>{tl.intervals.map((iv: any, i: number) => <tr key={i} data-state={iv.state} onClick={() => seek(iv.t[0])}>
          <td>{fmt(iv.t[0])}–{fmt(iv.t[1])} s</td><td>{iv.state}{iv.reason ? <small> · {iv.reason}</small> : null}</td>
          <td>{q(iv.v?.position_xy)}</td><td>{q(iv.v?.top_above_floor)}</td><td>{q(iv.v?.base_above_floor)}</td><td>{q(iv.v?.height)}</td><td>{q(iv.v?.width)}</td></tr>)}</tbody></table>
      {!!tl.changes?.length && <ul>{tl.changes.map((c: any, i: number) => <li key={i}><button className="mvp-link" onClick={() => seek(c.t_after)}>{c.kind}</button>
        {" "}{fmt(c.t_before)} → {fmt(c.t_after)} s · {tr("LiveReporttsx.099")} {c.before_key} / {c.after_key}{c.new_place_key != null ? ` / ${c.new_place_key}` : ""}
        {c.distance_m != null && <> · {fmt(c.distance_m, 2)} ± {fmt(c.distance_u_m, 2)} m</>}{c.to && <> · → {c.to}</>}{c.from && <> · ← {c.from}</>}</li>)}</ul>}
      <p><small>{tl.rule}</small></p>
      <details><summary>{tr("LiveReporttsx.100")}</summary><table><tbody>{tl.windows.map((w: any) => <tr key={w.w}>
        <td>{fmt(w.t[0])}–{fmt(w.t[1])} s</td><td>{w.state}{w.reason ? <small> · {w.reason}</small> : null}</td><td>{q(w.v?.position_xy)}</td>
        <td>{w.d ? Object.entries(w.d).filter(([, d]: any) => d[3]).map(([f, d]: any) => `${f} Δ${fmt(d[0], 2)} > u ${fmt(d[1], 2)}/${fmt(d[2], 2)}`).join("; ") || tr("LiveReporttsx.101") : ""}</td></tr>)}</tbody></table></details>
    </details>}
  </section>;
}

function Judgements({ info, patch, tr }: { info: Info; patch?: Patch; tr: Tr }) {
  if (!patch) return <section className="mvp-block"><h4>{tr("LiveReporttsx.102")}</h4><p><small>{tr("LiveReporttsx.103")}</small></p></section>;
  return <section className="mvp-block"><h4>{tr("LiveReporttsx.102")}</h4>
    {!info.rows.length && <p><small>{tr("LiveReporttsx.104")}</small></p>}
    <ul className="mvp-judgements">{info.rows.map(r => <li key={r.id} data-v={r.verdict}>
      <p><Chip v={r.verdict} tr={tr} /> <strong>{r.title}</strong> <small>{r.check}{r.severity ? ` · ${r.severity}` : ""}</small></p>
      {r.geometry && <p>{r.geometry.quantity}{r.geometry.value != null && <>: {fmt(r.geometry.value, 2)}{r.geometry.u != null && ` ± ${fmt(r.geometry.u, 2)}`} {r.geometry.unit}</>}
        {r.geometry.threshold != null && <> {tr("LiveReporttsx.105")} {r.geometry.threshold} {r.geometry.unit} ({r.geometry.direction === "max" ? tr("LiveReporttsx.106") : tr("LiveReporttsx.107")})</>} → {r.geometry.result}
        {r.geometry.before_forced && <small> ({tr("LiveReporttsx.108")} {r.geometry.before_forced})</small>}
        {String(r.geometry.scale || "").startsWith("estimated") && <Tag>{tr("LiveReporttsx.022")}</Tag>}</p>}
      {r.vlm && <div className="mvp-vlm"><small>{tr("LiveReporttsx.109")} · {r.vlm.decider}</small>
        {(r.vlm.questions ? Object.entries(r.vlm.questions).map(([question, v]: [string, any]) => ({ question, ...v })) : [r.vlm, ...(r.vlm.also || [])]).map((qq: any, j: number) => <div key={j}><small><strong>{qq.question}</strong>{qq.text ? ` "${qq.text}"` : ""}{qq.calibration ? ` · ${tr("LiveReporttsx.110")} ${qq.calibration}` : ""}</small>
          {(qq.per_view || []).map((v: any, i: number) => <div key={i}><small>{tr("LiveReporttsx.111")} {v.keys?.join(", ")}: {v.probs ? (qq.options || r.vlm.options).map((o: string, k: number) => `${o} ${fmt(v.probs[k], 2)}`).join(" · ") : tr("LiveReporttsx.112")}
            {" "}· {tr("LiveReporttsx.113")} {fmt(v.mass, 2)}{v.p_hazard_raw != null && ` · ${tr("live.hazardProbability")} ${fmt(v.p_hazard_raw, 2)}`}
            {typeof v.calibrated === "number" ? ` → ${tr("LiveReporttsx.114")} ${fmt(v.calibrated, 2)}` : ` (${tr("LiveReporttsx.047")})`}</small></div>)}</div>)}
        {r.vlm.p_yes != null && <div><small>{tr("live.yesProbability")} {fmt(r.vlm.p_yes, 2)}{r.vlm.cut ? ` · ${tr("LiveReporttsx.115")} ${tr("live.thresholds", { hazard: r.vlm.cut.hazard ?? "—", clear: r.vlm.cut.clear ?? "—", veto: r.vlm.cut.veto ?? "—" })}${r.vlm.calibrated ? "" : ` (${tr("LiveReporttsx.047")})`}` : ""}{r.vlm.why ? ` · "${r.vlm.why}"` : ""}</small></div>}
        <small>{tr("LiveReporttsx.116")}: <strong>{r.vlm.answer}</strong></small></div>}
      {!!r.reasons?.length && <ul className="mvp-reasons">{r.reasons.map((x: string, i: number) => <li key={i}>{x}</li>)}</ul>}
      {!!r.evidence?.length && <div className="mvp-evidence">{r.evidence.map((e: any, i: number) => <button key={i} className="mvp-thumb" onClick={() => seek(e.t)} title={tr("LiveReporttsx.117")}>
        {blobURL(patch, e.image) ? <img src={blobURL(patch, e.image)!} alt={`${r.check} ${e.key}`} /> : null}<span>{fmt(e.t)} s</span></button>)}</div>}
      {r.rule_source && <p><small>{r.rule_source}</small></p>}
    </li>)}</ul>
  </section>;
}

function Under({ under, names, onSelect, tr }: { under: string[]; names: (id: string) => string; onSelect: (id: string) => void; tr: Tr }) {
  return under.length ? <section className="mvp-block"><h4>{tr("LiveReporttsx.118")}</h4>
    <p>{under.map(id => <button key={id} className="mvp-link" onClick={() => onSelect(id)}>{names(id)}</button>)}</p></section> : null;
}

/** mvp3 D4 (b): what the report container made of a click on no entity: an ad-hoc card (one view, no checks), a surface, or why none. */
function OnDemand({ od, tr }: { od: any; tr: Tr }) {
  if (!od) return null;
  if (od.pending) return <section className="mvp-block"><p><small>{tr("LiveReporttsx.119")}</small></p></section>;
  if (od.error) return <section className="mvp-block"><p><small>{tr("LiveReporttsx.120")}: {od.error}</small></p></section>;
  const idn = od.identity || {}, ph = od.physical || {}, took = <small>{tr("LiveReporttsx.121")} {fmt(od.ms, 0)} ms · {tr("LiveReporttsx.111")} {od.frame}</small>;
  if (od.status !== "card") return <section className="mvp-block"><p>{tr("LiveReporttsx.122")} <Tag>{tr("LiveReporttsx.123")}</Tag>: {od.surface || idn.covers || idn.namer?.name || "—"} · {tr("LiveReporttsx.124")}{od.surface_reason ? <small> ({od.surface_reason})</small> : null}</p><p>{took}</p></section>;
  return <section className="mvp-block mvp-ondemand">
    <h3>{idn.name} <Tag>{tr("LiveReporttsx.123")}</Tag> <Chip v={null} tr={tr} /></h3>
    <p>{idn.confidence == null ? tr("LiveReporttsx.045") : `${Math.round(idn.confidence * 100)}%`} <Tag>{tr("LiveReporttsx.047")}</Tag> · {tr("LiveReporttsx.048")} {idn.decided_by}{idn.status ? ` · ${idn.status}` : ""}</p>
    <p>{od.class?.category || "other"} · {od.class?.mobility || "—"}</p>
    <table className="mvp-physical"><tbody>
      {[...PHYSICAL, ["distance_from_camera", "LiveReporttsx.128"] as [string, string]].filter(([k]) => ph[k]).map(([k, label]) =>
        <tr key={k}><th>{tr(label)}</th><td><Quantity q={ph[k]} tr={tr} /></td></tr>)}
      {ph.size_check && <tr><th>{tr("LiveReporttsx.065")}</th><td>{ph.size_check.status}{ph.size_check.reason ? <small> ({ph.size_check.reason})</small> : null}</td></tr>}
    </tbody></table>
    <p><small>{od.note}</small></p>
    <ModelLine model={od.model} surface={od.model?.triangles ? { triangles: od.model.triangles } : undefined} tr={tr} />
    <p>{took}</p>
  </section>;
}

function UnknownCard({ r, od, under, names, onSelect, tr }: { r: NonNullable<Clicked["miss"]>; od: any; under: string[]; names: (id: string) => string; onSelect: (id: string) => void; tr: Tr }) {
  return <article className="mvp-card mvp-unknown">
    {od?.status === "card" ? <OnDemand od={od} tr={tr} /> : <>
      <header className="mvp-block"><h3>{tr("LiveReporttsx.125")}</h3><p><small>{tr("LiveReporttsx.126")}</small></p></header>
      <OnDemand od={od} tr={tr} /></>}
    {od?.status === "card" ? null : r.status !== "depth" ? <section className="mvp-block"><p>{tr("LiveReporttsx.127")}</p></section> : <section className="mvp-block">
      <table className="mvp-physical"><tbody>
        <tr><th>{tr("LiveReporttsx.128")}</th><td><Quantity q={{ ...r.distance, unit: "m", scale: "estimated" }} tr={tr} /></td></tr>
        <tr><th>{tr("LiveReporttsx.129")}</th><td><Quantity q={{ ...r.height, unit: "m", scale: "estimated" }} tr={tr} /></td></tr>
        <tr><th>{tr("LiveReporttsx.130")}</th><td>{r.surface.kind} <small>({tr("LiveReporttsx.131")})</small></td></tr>
        <tr><th>{tr("LiveReporttsx.132")}</th><td>{r.nearest ? <button className="mvp-link" onClick={() => onSelect(r.nearest.id)}>{r.nearest.name} · <Quantity q={{ ...r.nearest.distance, unit: "m", scale: "estimated" }} tr={tr} /></button> : "—"}</td></tr>
      </tbody></table>
      <p><small>{tr("LiveReporttsx.133")} · {tr("LiveReporttsx.111")} {r.frame}</small></p>
    </section>}
    <Under under={under} names={names} onSelect={onSelect} tr={tr} />
  </article>;
}

function ObjectList({ cards, infos, selected, onSelect, tr }: { cards: any[]; infos: Map<string, Info>; selected: string | null; onSelect: (id: string) => void; tr: Tr }) {
  const [verdict, setVerdict] = useState("all"), [kind, setKind] = useState("all"), [query, setQuery] = useState(""), [changed, setChanged] = useState(false);
  const now = useVideoTime();  // r5b: every card's state at the video's time
  const kindOf = (c: any) => c.kind === "person" ? "person" : c.class?.category || "other", v = (c: any) => infos.get(c.id)?.verdict ?? "none";
  const rank = (c: any) => { const i = SEVERITY.indexOf(v(c) as any); return i < 0 ? SEVERITY.length : i; };
  const counts = cards.reduce((n: Record<string, number>, c) => ({ ...n, [v(c)]: (n[v(c)] || 0) + 1 }), {});
  const nChanged = cards.filter(c => c.time?.timeline?.changes?.length).length;
  const shown = cards.filter(c => (verdict === "all" || v(c) === verdict) && (kind === "all" || kindOf(c) === kind) && (!query || (c.identity?.name || c.id).toLowerCase().includes(query.toLowerCase()))
    && (!changed || c.time?.timeline?.changes?.length))
    .sort((a, b) => rank(a) - rank(b) || (a.time?.first_seen_s ?? 1e9) - (b.time?.first_seen_s ?? 1e9));
  if (!cards.length) return <p className="mvp-empty">{tr("LiveReporttsx.134")}</p>;
  return <div className="mvp-list">
    <div className="mvp-filters" role="group" aria-label={tr("LiveReporttsx.135")}>
      {["all", ...SEVERITY, "none"].map(k => <button key={k} aria-pressed={verdict === k} data-v={k} onClick={() => setVerdict(k)}>
        {k === "all" ? tr("LiveReporttsx.136") : k === "none" ? tr("LiveReporttsx.019") : k.replace("_", " ")} {k === "all" ? cards.length : counts[k] || 0}</button>)}
    </div>
    <div className="mvp-filters">
      <select value={kind} onChange={e => setKind(e.target.value)} aria-label={tr("LiveReporttsx.052")}>
        <option value="all">{tr("LiveReporttsx.137")}</option>{[...new Set(cards.map(kindOf))].sort().map(k => <option key={k} value={k}>{k}</option>)}</select>
      <input type="search" value={query} onChange={e => setQuery(e.target.value)} placeholder={tr("LiveReporttsx.138")} aria-label={tr("LiveReporttsx.138")} />
      <label><input type="checkbox" checked={changed} onChange={e => setChanged(e.target.checked)} />{tr("LiveReporttsx.139")} {nChanged}</label>
      <small>{shown.length} · {tr("LiveReporttsx.140")} {fmt(now)} s</small>
    </div>
    <ol>{shown.map(c => <li key={c.id}><button data-id={c.id} aria-current={c.id === selected || undefined} onClick={() => { onSelect(c.id); if (c.kind === "person") seek(c.time?.first_seen_s); }}>
      <Chip v={infos.get(c.id)?.verdict} tr={tr} /><strong>{c.kind === "person" ? `${tr("LiveReporttsx.141")} ${c.id.slice(7)}` : c.identity?.name}</strong>
      <small>{kindOf(c)} · {fmt(c.time?.first_seen_s)} s{c.physical?.size_check?.status === "implausible" ? ` · ${tr("LiveReporttsx.142")}` : ""}
        {" · "}<span className="mvp-state" data-state={stateAt(c, now).state} style={{ color: STATE_CSS(stateAt(c, now).state) }}>{stateAt(c, now).state}</span></small></button></li>)}</ol>
  </div>;
}

function InfoCard({ entity, tr }: { entity: any; tr: Tr }) {
  const f = entity?.fast;
  if (!f) return <aside className="live-report-card"><p>{tr("LiveReporttsx.143")}</p></aside>;
  if (f.kind === "object") {
    const size = [0, 1, 2].map(k => f.box_max_m[k] - f.box_min_m[k]), others = Object.entries(f.votes || {}).filter(([w]) => w !== f.word).sort((a: any, b: any) => b[1] - a[1]);
    return <aside className="live-report-card">
      <h3>{f.word} <small>{tr("LiveReporttsx.144")}</small></h3>
      {!!others.length && <p>{tr("LiveReporttsx.145")}: {others.map(([w, v]) => `${w} (${fmt(v)})`).join(", ")}</p>}
      <p>{tr("LiveReporttsx.146")}: {f.frames} · {tr("LiveReporttsx.147")} {f.shot + 1}</p>
      <p>{tr("LiveReporttsx.148")}: {size.map(v => fmt(v, 2)).join(" × ")} m <small>{tr("LiveReporttsx.149")}</small></p>
      <p>{tr("measure.model")}: {f.model ? <>{tr("LiveReporttsx.150")} · <small>{tr("LiveReporttsx.151")}</small></> : tr("LiveReporttsx.152")}</p>
    </aside>;
  }
  if (f.kind === "person") return <aside className="live-report-card">
    <h3>{tr("LiveReporttsx.153")} {f.id}</h3>
    <p>{fmt(f.t0)}–{fmt(f.t1)} s · {f.detections} {tr("LiveReporttsx.154")}</p>
    {f.rules?.length ? <ul>{f.rules.map((r: any, i: number) => <li key={i}>{r.rule}: {r.verdict}</li>)}</ul>
      : <p>{tr("LiveReporttsx.155")}</p>}
    <p><small>{tr("LiveReporttsx.156")}</small></p>
  </aside>;
  return <aside className="live-report-card"><h3>{entity.label}</h3><p>{f.triangles} {tr("LiveReporttsx.157")} · {f.points} {tr("LiveReporttsx.158")}</p></aside>;
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
  return <section className="live-report-timing" aria-label={tr("LiveReporttsx.159")}>
    <div>
      <h3>{tr("LiveReporttsx.160")}
        {call !== undefined && <mark className="mvp-call">{call ? tr("LiveReporttsx.161") : tr("LiveReporttsx.162")}</mark>}</h3>
      <p className="mvp-clicks"><small>{tr("LiveReporttsx.163")}: {clicks.length ? `p50 ${fmt(quantile(clicks, .5))} ms · p95 ${fmt(quantile(clicks, .95))} ms (n ${clicks.length}, ${tr("LiveReporttsx.164")})` : tr("LiveReporttsx.165")}
        {pickMs != null && <> · {tr("LiveReporttsx.166")} {fmt(pickMs, 0)} ms{pickSteps && ` (${Object.entries(pickSteps).map(([k, v]) => `${k} ${fmt(v, 0)}`).join(" · ")})`}, {tr("LiveReporttsx.167")}</>}</small></p>
      <table><thead><tr><th>{tr("LiveReporttsx.168")}</th><th>{tr("LiveReporttsx.169")}</th><th>{tr("LiveReporttsx.170")}</th><th>{tr("LiveReporttsx.171")}</th><th>MB</th><th>{tr("LiveReporttsx.172")}</th></tr></thead>
        <tbody>{rows.map(p => <tr key={p.seq}><td>{p.layer}{p.version > 1 ? ` v${p.version}` : ""}</td><td>{fmt(p.sent_s)}</td><td>{fmt(written[p.seq])}</td>
          <td title={tr("LiveReporttsx.173")}>{fmt(served[p.seq] - p.t0_unix)}</td>
          <td>{fmt(Object.values(p.blobs).reduce((n, b) => n + b.bytes, 0) / 1e6)}</td>{target(p)}</tr>)}
          {bootS !== undefined && <tr className="live-report-boot"><td>{tr("LiveReporttsx.174")}</td><td colSpan={5}>{fmt(bootS)} s</td></tr>}
        </tbody></table>
    </div>
    <div>
      <h3>{tr("LiveReporttsx.175")}</h3>
      {run?.stages ? <table><thead><tr><th>{tr("LiveReporttsx.176")}</th><th>{tr("LiveReporttsx.177")}</th><th>{tr("LiveReporttsx.178")}</th><th>GPU0</th><th>GPU1</th></tr></thead>
        <tbody>{run.stages.map((s: any, i: number) => <tr key={i}><td>{s.stage}</td><td>{s.where}</td><td>{fmt(s.start_s)}–{fmt(s.end_s)}</td>{peak(s.peak_gb?.[0], 0)}{peak(s.peak_gb?.[1], 1)}</tr>)}
          {(run.gpu_peak || []).length > 0 && <tr><td>{tr("LiveReporttsx.179")}</td><td colSpan={2}>{run.fixture ? tr("LiveReporttsx.180") : ""}</td>
            {[0, 1].map(g => peak(run.gpu_peak.find((x: any) => x.gpu === g)?.peak_gb, g))}</tr>}
        </tbody></table> : <p>{tr("LiveReporttsx.181")}</p>}
      {!!run?.flags?.length && <ul className="live-report-flags">{run.flags.map((f: string) => <li key={f}>{f}</li>)}</ul>}
      {run?.fixture && <p><small>{run.fixture}</small></p>}
    </div>
  </section>;
}
