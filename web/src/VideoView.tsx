import { useEffect, useMemo, useRef, useState } from "react";
import { resolveAsset } from "./api";
import { useI18n } from "./i18n";
import type { SceneDocument } from "./types";

/** Per-frame outlines of scene entities in source-video pixels, written by the video importer. */
type Outline = { entityId: string | null; label: string; polygons: number[][][] };
type Frame = { timeSec: number; endTimeSec: number; sourceFrame: number; objects: Outline[] };
type Analysis = { width: number; height: number; frames: Frame[] };
type Replay = { videoAssetId: string; analysisAssetId: string };

/** The report's one clock: the source video's time, read by the 3D views to show moving objects at that moment. */
export const videoClock = { time: 0 };
const announce = (time: number) => { videoClock.time = time; window.dispatchEvent(new CustomEvent("panoptes:video-time", { detail: time })); };

export function videoReplay(document: SceneDocument): Replay | null {
  const found = (document.annotations || []).find((a) => a.kind === "video_replay") as Partial<Replay> | undefined;
  return found && typeof found.videoAssetId === "string" && typeof found.analysisAssetId === "string" ? found as Replay : null;
}

function frameAt(frames: Frame[], time: number) {
  let low = 0, high = frames.length - 1;
  while (low < high) { const middle = (low + high + 1) >> 1; if (frames[middle].timeSec <= time) low = middle; else high = middle - 1; }
  return frames[low];
}

/** The source video as a view of the report: every frame's entities can be picked, and a pick elsewhere finds its frame. */
export function VideoView({ document, selectedId, onSelect }: {
  document: SceneDocument; selectedId: string | null; onSelect: (id: string) => void;
}) {
  const { language } = useI18n(), replay = videoReplay(document),
    [source, setSource] = useState<string>(), [analysis, setAnalysis] = useState<Analysis>(),
    [time, setTime] = useState(0), [playing, setPlaying] = useState(false), [error, setError] = useState(false),
    video = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (!replay) return;
    let live = true, blobUrl: string | undefined;
    setError(false);
    // A blob keeps seeking exact whether or not the asset endpoint answers range requests.
    Promise.all([resolveAsset(replay.videoAssetId).then((url) => fetch(url)).then((r) => r.blob()),
      resolveAsset(replay.analysisAssetId).then((url) => fetch(url)).then((r) => r.json())])
      .then(([blob, data]) => { if (!live) return; blobUrl = URL.createObjectURL(blob); setSource(blobUrl); setAnalysis(data); })
      .catch(() => { if (live) setError(true); });
    return () => { live = false; if (blobUrl) URL.revokeObjectURL(blobUrl); };
  }, [replay?.videoAssetId, replay?.analysisAssetId]);
  useEffect(() => {
    const element = video.current;
    if (!element || !playing) return;
    let handle = 0;
    const tick = () => { setTime(element.currentTime); handle = requestAnimationFrame(tick); };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [playing]);
  const frame = useMemo(() => analysis?.frames.length ? frameAt(analysis.frames, time) : undefined, [analysis, time]);
  useEffect(() => { announce(time); }, [frame]);  // once per sampled frame, not per animation frame
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
  const duration = analysis?.frames.length ? analysis.frames[analysis.frames.length - 1].endTimeSec : 0;
  return <div className="report-video">
    <div className="report-video-stage">
      <video ref={video} src={source} muted playsInline preload="auto" onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
        onSeeked={(e) => setTime(e.currentTarget.currentTime)} aria-label={language === "zh" ? "来源视频" : "Source video"} />
      {analysis && frame && <svg viewBox={`0 0 ${analysis.width} ${analysis.height}`} preserveAspectRatio="xMidYMid meet">
        {frame.objects.flatMap((object, n) => object.polygons.map((polygon, k) =>
          <polygon key={`${n}-${k}`} points={polygon.map((p) => p.join(",")).join(" ")} data-selected={object.entityId === selectedId || undefined}
            data-pickable={object.entityId ? true : undefined} onClick={() => object.entityId && onSelect(object.entityId)}>
            <title>{object.label}</title></polygon>))}
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
    resolveAsset(comparison.videoAssetId).then((url) => fetch(url)).then((r) => r.blob())
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
