import { useEffect, useMemo, useRef, useState } from "react";
import { resolveAsset as apiAsset } from "./api";
import { useI18n } from "./i18n";
import type { SceneDocument } from "./types";
import { clickClock, pointInPolygon } from "./live-report";

/** Per-frame outlines of scene entities in source-video pixels, written by the video importer. */
/** source: "segmented" (a mask on this frame) or "projected" (carried from 3D; the fast report's in-between frames), drawn dashed. */
type Outline = { entityId: string | null; label: string; polygons: number[][][]; source?: string };
type Frame = { timeSec: number; endTimeSec: number; sourceFrame: number; objects: Outline[] };
type Analysis = { width: number; height: number; frames: Frame[] };
/** The same frames without the 4:3 crop the reconstruction used; rasterInVideo is where the outline raster sits in it (x, y, w, h). */
type FullFrame = { videoAssetId: string; width: number; height: number; rasterInVideo: number[] };
type Replay = { videoAssetId: string; analysisAssetId?: string; fullFrame?: FullFrame };
type Resolve = (id: string) => Promise<string>;

/** The report's one clock: the source video's time, read by the 3D views to show moving objects at that moment. */
export const videoClock = { time: 0 };
const announce = (time: number) => { videoClock.time = time; window.dispatchEvent(new CustomEvent("panoptes:video-time", { detail: time })); };

export function videoReplay(document: SceneDocument): Replay | null {
  const found = (document.annotations || []).find((a) => a.kind === "video_replay") as Partial<Replay> | undefined;
  // A live report shows its video before the outlines exist.
  if (!found || typeof found.videoAssetId !== "string" || found.analysisAssetId !== undefined && typeof found.analysisAssetId !== "string") return null;
  const full = found.fullFrame;
  const usable = full && typeof full.videoAssetId === "string" && full.width > 0 && full.height > 0 && full.rasterInVideo?.length === 4 && full.rasterInVideo.every(Number.isFinite);
  return { videoAssetId: found.videoAssetId, analysisAssetId: found.analysisAssetId, fullFrame: usable ? full : undefined };
}

function frameAt(frames: Frame[], time: number) {
  let low = 0, high = frames.length - 1;
  while (low < high) { const middle = (low + high + 1) >> 1; if (frames[middle].timeSec <= time) low = middle; else high = middle - 1; }
  return frames[low];
}

/** A click on the video: time, source pixel, and the entities whose outline polygons on that frame contain the point. */
export type VideoPick = { t: number; x: number; y: number; under: string[] };
/** The selected entity's pixels on the current pick frame (1 = it), drawn over the video. */
export type Highlight = { w: number; h: number; mask: Uint8Array } | null;

/** The source video as a view of the report: every frame's entities can be picked, and a pick elsewhere finds its frame.
 *  onPick (the live report's pick layer) replaces polygon clicks: the whole stage is clickable, the video pauses. */
export function VideoView({ document, selectedId, onSelect, resolveAsset = apiAsset, onPick, highlight, marker }: {
  document: SceneDocument; selectedId: string | null; onSelect: (id: string) => void; resolveAsset?: Resolve;
  onPick?: (pick: VideoPick) => void; highlight?: (time: number) => Highlight; marker?: { x: number; y: number } | null;
}) {
  const { language } = useI18n(), replay = videoReplay(document), full = replay?.fullFrame, shownVideo = full?.videoAssetId ?? replay?.videoAssetId,
    [source, setSource] = useState<string>(), [analysis, setAnalysis] = useState<Analysis>(),
    [time, setTime] = useState(0), [playing, setPlaying] = useState(false), [error, setError] = useState(false),
    video = useRef<HTMLVideoElement>(null), stage = useRef<HTMLDivElement>(null), [fill, setFill] = useState(false), [length, setLength] = useState(0);
  // A pane a little wider than the video is filled edge to edge, trimming at most a sixth of its height in all;
  // the outlines are trimmed the same way. A much wider or a narrower pane shows the whole frame.
  useEffect(() => {
    const node = stage.current, element = video.current;
    if (!node || !element) return;
    const decide = () => { const ratio = (node.clientWidth / Math.max(node.clientHeight, 1)) / ((element.videoWidth || 16) / (element.videoHeight || 9)); setFill(ratio >= 1 && ratio <= 1.2); };
    const observer = new ResizeObserver(decide);
    observer.observe(node); element.addEventListener("loadedmetadata", decide); decide();
    return () => { observer.disconnect(); element.removeEventListener("loadedmetadata", decide); };
  }, [source]);
  useEffect(() => {
    if (!replay) return;
    let live = true, blobUrl: string | undefined;
    setError(false);
    // A blob keeps seeking exact whether or not the asset endpoint answers range requests.
    resolveAsset(shownVideo!).then((url) => fetch(url)).then((r) => r.blob())
      .then((blob) => { if (!live) return; blobUrl = URL.createObjectURL(blob); setSource(blobUrl); })
      .catch(() => { if (live) setError(true); });
    return () => { live = false; if (blobUrl) URL.revokeObjectURL(blobUrl); };
  }, [shownVideo]);
  useEffect(() => {  // outlines arriving later (a live report) never reload the video
    if (!replay?.analysisAssetId) return;
    let live = true;
    resolveAsset(replay.analysisAssetId).then((url) => fetch(url)).then((r) => r.json())
      .then((data) => { if (live) setAnalysis(data); }).catch(() => { if (live) setError(true); });
    return () => { live = false; };
  }, [replay?.analysisAssetId]);
  useEffect(() => {
    const element = video.current;
    if (!element || !playing) return;
    let handle = 0;
    const tick = () => { setTime(element.currentTime); handle = requestAnimationFrame(tick); };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [playing]);
  const frame = useMemo(() => analysis?.frames.length ? frameAt(analysis.frames, time) : undefined, [analysis, time]);
  const canvas = useRef<HTMLCanvasElement>(null), lit = useMemo(() => highlight?.(time) ?? null, [highlight, frame]);  // once per keyframe
  useEffect(() => {  // the pick map's own pixels of the selection: what a click resolves, people included
    const node = canvas.current;
    if (!node) return;
    if (!lit) { node.width = node.height = 1; return; }
    node.width = lit.w; node.height = lit.h;
    const g = node.getContext("2d")!, image = g.createImageData(lit.w, lit.h);
    for (let i = 0; i < lit.mask.length; i++) if (lit.mask[i]) image.data.set([86, 214, 140, 110], 4 * i);
    g.putImageData(image, 0, 0);
  }, [lit]);
  const toSource = (clientX: number, clientY: number) => {  // client point -> source pixel, through the same fit as the video
    const element = video.current!, box = element.getBoundingClientRect(), vw = element.videoWidth || 1280, vh = element.videoHeight || 720;
    const scale = (fill ? Math.max : Math.min)(box.width / vw, box.height / vh);
    let x = (clientX - box.left - (box.width - vw * scale) / 2) / scale, y = (clientY - box.top - (box.height - vh * scale) / 2) / scale;
    if (full && analysis) { x = (x - full.rasterInVideo[0]) * analysis.width / full.rasterInVideo[2]; y = (y - full.rasterInVideo[1]) * analysis.height / full.rasterInVideo[3]; }
    return [x, y];
  };
  const pick = (event: React.PointerEvent) => {
    const element = video.current;
    if (!onPick || !element || event.button !== 0) return;
    clickClock.t0 = performance.now();
    element.pause();
    const t = element.currentTime, [x, y] = toSource(event.clientX, event.clientY), at = analysis?.frames.length ? frameAt(analysis.frames, t) : undefined;
    const under = [...new Set((at?.objects || []).filter((o) => o.entityId && o.polygons.some((p) => p.length > 2 && pointInPolygon([x, y], p))).map((o) => o.entityId!))];
    setTime(t);
    onPick({ t, x, y, under });
  };
  useEffect(() => { announce(time); }, [frame]);  // once per sampled frame, not per animation frame
  useEffect(() => {  // a time picked elsewhere in the report (an event in the video memory) moves the video there
    const seek = (event: Event) => { const element = video.current, at = (event as CustomEvent<number>).detail; if (!element || !Number.isFinite(at)) return; element.pause(); element.currentTime = at; setTime(at); };
    window.addEventListener("panoptes:seek", seek);
    return () => window.removeEventListener("panoptes:seek", seek);
  }, []);
  useEffect(() => {  // picked in 3D, CAD or the list: show the nearest moment the video sees it
    const element = video.current;
    if (!element || !analysis || !selectedId || frame?.objects.some((o) => o.entityId === selectedId)) return;
    const seen = analysis.frames.filter((f) => f.objects.some((o) => o.entityId === selectedId));
    if (!seen.length) return;
    const nearest = seen.reduce((a, b) => Math.abs(b.timeSec - time) < Math.abs(a.timeSec - time) ? b : a);
    element.pause(); element.currentTime = nearest.timeSec + 1e-3; setTime(nearest.timeSec + 1e-3);
  }, [selectedId, analysis]);
  if (!replay) return null;
  if (error) return <div className="report-scene-plan-empty" role="status">{language === "zh" ? "视频或逐帧轮廓读取失败" : "The video or its per-frame outlines could not be read"}</div>;
  const duration = analysis?.frames.length ? analysis.frames[analysis.frames.length - 1].endTimeSec : length;
  return <div className="report-video">
    <div className="report-video-stage" ref={stage} onPointerDown={onPick ? pick : undefined} data-pickable={onPick ? true : undefined} data-has-selection={selectedId ? true : undefined}>
      <video ref={video} src={source} muted playsInline preload="auto" style={fill ? { objectFit: "cover" } : undefined} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
        onLoadedMetadata={(e) => setLength(e.currentTarget.duration)}
        onSeeked={(e) => setTime(e.currentTarget.currentTime)} aria-label={language === "zh" ? "来源视频" : "Source video"} />
      {highlight && <canvas ref={canvas} className="report-video-highlight" style={fill ? { objectFit: "cover" } : undefined} />}
      {analysis && frame && <svg viewBox={full ? `0 0 ${full.width} ${full.height}` : `0 0 ${analysis.width} ${analysis.height}`} preserveAspectRatio={fill ? "xMidYMid slice" : "xMidYMid meet"}>
        {/* Uncropped video: the outlines stay on the reconstruction's raster, placed where the crop sits; the dashed frame marks it. */}
        {full && <rect className="report-video-raster" x={full.rasterInVideo[0]} y={full.rasterInVideo[1]} width={full.rasterInVideo[2]} height={full.rasterInVideo[3]} />}
        <g transform={full ? `translate(${full.rasterInVideo[0]} ${full.rasterInVideo[1]}) scale(${full.rasterInVideo[2] / analysis.width} ${full.rasterInVideo[3] / analysis.height})` : undefined}>
        {frame.objects.flatMap((object, n) => object.polygons.map((polygon, k) =>
          <polygon key={`${n}-${k}`} points={polygon.map((p) => p.join(",")).join(" ")} data-selected={object.entityId === selectedId || undefined} data-source={object.source}
            data-pickable={object.entityId && !onPick ? true : undefined} onClick={onPick ? undefined : () => object.entityId && onSelect(object.entityId)}>
            <title>{object.label}</title></polygon>))}
        {marker && <g className="report-video-marker"><circle cx={marker.x} cy={marker.y} r={14} /><path d={`M${marker.x - 22} ${marker.y}h44M${marker.x} ${marker.y - 22}v44`} /></g>}
        </g>
      </svg>}
    </div>
    <footer className="report-video-controls">
      <button onClick={() => { const element = video.current; if (element) void (element.paused ? element.play() : element.pause()); }}>
        {playing ? (language === "zh" ? "暂停" : "Pause") : (language === "zh" ? "播放" : "Play")}</button>
      <input type="range" min={0} max={duration} step={0.01} value={Math.min(time, duration)} aria-label={language === "zh" ? "视频时间" : "Video time"}
        onChange={(e) => { const value = Number(e.target.value); if (video.current) video.current.currentTime = value; setTime(value); }} />
      <span>{time.toFixed(1)} s · {language === "zh" ? "源帧" : "frame"} {frame?.sourceFrame ?? "—"} · {frame?.objects.filter((o) => o.entityId).length ?? 0} {language === "zh" ? "个可点选对象" : "pickable"}</span>
    </footer>
  </div>;
}

type Comparison = { videoAssetId: string; note?: string };
export function comparisonVideo(document: SceneDocument): Comparison | null {
  const found = (document.annotations || []).find((a) => a.kind === "static_dynamic_comparison") as Partial<Comparison> | undefined;
  return found && typeof found.videoAssetId === "string" ? found as Comparison : null;
}

/** The clip split into its static and dynamic layers, rendered from the clip's own camera, shown as it was rendered. */
export function ComparisonVideo({ document }: { document: SceneDocument }) {
  const { language } = useI18n(), comparison = comparisonVideo(document), [source, setSource] = useState<string>(), [error, setError] = useState(false);
  useEffect(() => {
    if (!comparison) return;
    let live = true, blobUrl: string | undefined;
    apiAsset(comparison.videoAssetId).then((url) => fetch(url)).then((r) => r.blob())
      .then((blob) => { if (!live) return; blobUrl = URL.createObjectURL(blob); setSource(blobUrl); }).catch(() => { if (live) setError(true); });
    return () => { live = false; if (blobUrl) URL.revokeObjectURL(blobUrl); };
  }, [comparison?.videoAssetId]);
  if (!comparison) return null;
  const zh = language === "zh";
  return <section className="report-scene-comparison" aria-label={zh ? "动静分离对比" : "Static / dynamic comparison"}>
    <header><h3>{zh ? "动静分离 · 原图 ｜ 静态+动态 ｜ 只看静态 ｜ 只看动态" : "Static / dynamic · photo | both | static only | dynamic only"}</h3>
      {comparison.note && <p>{comparison.note}</p>}</header>
    {error ? <p role="status">{zh ? "对比视频读取失败" : "The comparison video could not be read"}</p>
      : <video src={source} controls muted playsInline preload="auto" aria-label={zh ? "动静分离对比视频" : "Static / dynamic comparison video"} />}
  </section>;
}

type MemoryEvent = { t0?: number; t1?: number; actor?: string | null; action?: string; near?: string[]; ppe?: Record<string, string>; safety_note?: string | null; entityId?: string | null };
type MemoryWindow = { t0: number; t1: number; caption?: string | null; events?: MemoryEvent[] };

/** The video memory: what an open video model wrote for each window, with times that move the video and actors that select the mover. */
export function VideoMemory({ document, onSelect }: { document: SceneDocument; onSelect: (id: string) => void }) {
  const { language } = useI18n(), zh = language === "zh";
  const memory = (document.annotations || []).find((a) => a.kind === "video_events") as { model?: string; windows?: MemoryWindow[]; note?: string } | undefined;
  if (!memory?.windows?.length) return null;
  const seek = (at?: number) => { if (Number.isFinite(at)) window.dispatchEvent(new CustomEvent("panoptes:seek", { detail: at })); };
  const ppe = (value?: Record<string, string>) => value ? Object.entries(value).filter(([, v]) => v && v !== "unknown").map(([k, v]) => `${k === "helmet" ? (zh ? "安全帽" : "helmet") : (zh ? "反光衣" : "vest")}：${v === "yes" ? (zh ? "有" : "yes") : v === "no" ? (zh ? "无" : "no") : v}`).join(" · ") : "";
  return <section className="report-memory" aria-label={zh ? "视频记忆" : "Video memory"}>
    <header><h3>{zh ? "视频记忆 · 每段发生了什么" : "Video memory · what happens in each window"}</h3>
      <p>{zh ? `由 ${memory.model} 按时间窗口写的描述，是证据不是结论；点时间跳到视频，点对象在三维里选中。` : `Written per window by ${memory.model}; evidence, not a verdict.`}</p></header>
    <ol>{memory.windows.map((w) => <li key={w.t0}>
      <button className="report-memory-time" onClick={() => seek(w.t0)}>{w.t0.toFixed(1)}–{w.t1.toFixed(1)} s</button>
      <p>{w.caption}</p>
      {!!w.events?.length && <ul>{w.events.map((e, n) => <li key={n}>
        <button className="report-memory-time" onClick={() => seek(e.t0)}>{(e.t0 ?? w.t0).toFixed(1)} s</button>
        {e.entityId ? <button className="report-memory-actor" onClick={() => onSelect(e.entityId!)}>{e.actor}</button> : <span className="report-memory-actor">{e.actor || (zh ? "未标记" : "unlabelled")}</span>}
        <span>{e.action}</span>{ppe(e.ppe) && <small>{ppe(e.ppe)}</small>}{e.safety_note && <em>{e.safety_note}</em>}
      </li>)}</ul>}
    </li>)}</ol>
  </section>;
}
