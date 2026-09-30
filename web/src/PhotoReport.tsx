import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { ReportScene } from "./ReportScene";
import { ObjectFacts } from "./WorkcellReport";
import { SceneResources } from "./SceneResources";
import type { BendAnalysis, InclinationAnalysis } from "./SpatialMeasurements";
import { I18nProvider } from "./i18n";
import { activeModel } from "./core";
import { transformMatrix } from "./viewer/native-math";
import type { Revision, Selection, Representation } from "./types";
import "./styles.css";
import "./workcell-report.css";
import "./photo-report.css";

type CatalogObject = { id: string; label: string; representation: string; notes: string[]; observations: { photo: number }[]; measurements: Record<string, any>; visibleHeightNative?: number; visibleHeightRangeNative?: number[]; visibleHeightByPhoto?: Record<string, number> };
type PhotoReportData = { bendAnalysis?: BendAnalysis; inclinationAnalysis?: InclinationAnalysis; revision: Revision; assetURLs: Record<string, string>; objects: CatalogObject[]; geometry: { anchor: { nativeHeight: number; nativeWidth: number; assumedHeightM: number; assumedWidthM: number; assumptions?: string[] }; floor: { status: string } }; timing: { oneShotSeconds?: number }; nativeToMetersDefault: number };
function PhotoReport({ data }: { data: PhotoReportData }) {
  const [imageId, setImageId] = useState("photo-4"), [entityId, setEntityId] = useState<string | null>("emergency-button"), [observationId, setObservationId] = useState<string | null>(null);
  const [height, setHeight] = useState(data.geometry.anchor.assumedHeightM * 100), [width, setWidth] = useState(data.geometry.anchor.assumedWidthM * 100), [axis, setAxis] = useState("height"), [exporting, setExporting] = useState(false), [exportError, setExportError] = useState("");
  const anchor = data.geometry.anchor, valid = Number.isFinite(height) && Number.isFinite(width) && height > 0 && width > 0;
  const scaleH = height / 100 / anchor.nativeHeight, scaleW = width / 100 / anchor.nativeWidth;
  const nativeToMeters = valid ? axis === "height" ? scaleH : scaleW : data.nativeToMetersDefault;
  const mismatch = valid && Math.abs(scaleH - scaleW) / Math.min(scaleH, scaleW) > .25;
  const photo = imageId.replace("photo-", "");
  const revision = useMemo(() => ({ ...data.revision, document: { ...data.revision.document,
    coordinateFrames: data.revision.document.coordinateFrames.map(frame => ({ ...frame, scale: { ...frame.scale, nativeToMeters } })),
    entities: data.revision.document.entities.map(entity => {
      const variant = (entity.modelVariants as Record<string, Representation> | undefined)?.[photo];
      return variant ? { ...entity, representations: [variant], activeModelRepresentationId: variant.id, currentModelTransform: variant.transform } : entity;
    }),
  } }), [data, nativeToMeters, photo]);
  const resources = useMemo(() => ({ analysisAvailable: false, bendAnalysis: data.bendAnalysis, inclinationAnalysis: data.inclinationAnalysis, resolveAsset: async (id: string) => {
    const file = data.assetURLs[id]; if (!file) throw Error(`Missing asset: ${id}`);
    return new URL(file, window.location.href).href;
  } }), [data]);
  const camera = revision.document.cameras.find(item => item.imageId === imageId);
  const selection: Selection = { projectId: revision.projectId, revisionId: revision.id, entityId, observationId, cameraId: camera?.id || null };
  const selected = revision.document.entities.find(entity => entity.id === entityId), record = data.objects.find(item => item.id === entityId);
  const displayValue = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? `${(value * nativeToMeters).toFixed(3)} m` : "未知";
  const visibleHeight = selected?.observedExtentAvailable !== true ? undefined : (record?.id === "robot" ? record.visibleHeightByPhoto?.[photo] : record?.visibleHeightNative);
  const inclination = data.inclinationAnalysis?.revisionId === revision.id ? data.inclinationAnalysis.items.find(row => row.entityId === entityId) : undefined;
  const bend = data.bendAnalysis?.revisionId === revision.id ? data.bendAnalysis.items.find(row => row.entityId === entityId) : undefined;
  const physicalHeight = record?.measurements.height?.valueNative, clearance = record?.measurements.groundClearance;
  async function downloadModel() {
    setExporting(true); setExportError("");
    try {
      const base = new URL("viewer-assets/", window.location.href).href;
      const THREE = await import(/* @vite-ignore */ `${base}three.module.js`);
      const { GLTFLoader } = await import(/* @vite-ignore */ `${base}addons/loaders/GLTFLoader.js`);
      const { GLTFExporter } = await import(/* @vite-ignore */ `${base}addons/exporters/GLTFExporter.js`);
      const scene = new THREE.Scene(), group = new THREE.Group(), loader = new GLTFLoader();
      scene.add(group); group.name = "Workcell — metres, Y up"; group.scale.setScalar(nativeToMeters); group.rotation.x = -Math.PI / 2;
      for (const entity of revision.document.entities.filter(entity => entity.visible !== false && !entity.sourceContext)) {
        const rep = activeModel(entity); if (!rep?.assetId) continue;
        const model = (await loader.loadAsync(await resources.resolveAsset(rep.assetId))).scene;
        const placed = new THREE.Group(); placed.name = entity.label || entity.id;
        placed.matrix.fromArray(transformMatrix(entity.currentModelTransform || rep.transform)); placed.matrixAutoUpdate = false;
        placed.userData = { entityId: entity.id, sourceRepresentation: rep.id, physicalDimensionsUnknown: entity.physicalDimensionsUnknown === true };
        placed.add(model); group.add(placed);
      }
      scene.userData = { units: "metres", upAxis: "Y", sourcePhoto: Number(photo), nativeToMeters, scaleHypothesis: { heightCm: height, widthCm: width, chosenAxis: axis }, assumptions: anchor.assumptions };
      scene.updateMatrixWorld(true);
      const result = await new GLTFExporter().parseAsync(scene, { binary: true });
      const url = URL.createObjectURL(new Blob([result], { type: "model/gltf-binary" }));
      const link = document.createElement("a"); link.href = url; link.download = `workcell-photo-${photo}-${axis}-${axis === "height" ? height : width}cm.glb`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { setExportError(error instanceof Error ? error.message : String(error)); }
    finally { setExporting(false); }
  }
  return <SceneResources.Provider value={resources}><main className="photo-report">
    <header className="photo-report-header"><a className="photo-report-brand" href="#overview">PANOPTES <span>WORKCELL REPORT</span></a><nav><a href="#overview">概览</a><a href="#scene">对象与场景</a><a href="#sources">来源与假设</a></nav></header>
    <section className="photo-report-overview" id="overview"><div><p className="photo-report-eyebrow">四张照片 · 对象级空间重建</p><h1>工作单元空间报告</h1><p>选取对象查看可见高度、离地间距与证据。拖动分界线，在同一相机下核对照片和模型。</p></div><dl><div><dt>照片</dt><dd>{revision.document.cameras.length}</dd></div><div><dt>对象</dt><dd>{data.objects.length}</dd></div><div><dt>本次计算</dt><dd>{data.timing.oneShotSeconds?.toFixed(0) ?? "—"}<small>秒</small></dd></div></dl></section>
    <section className="photo-report-scale" aria-label="标尺尺寸"><div><h2>急停按钮整体尺寸</h2><p>红帽 + 黄体 + 灰底。尺寸为输入假设；所有距离与导出模型按同一比例换算。</p></div><label>整体高度 (cm)<input aria-label="按钮整体高度厘米" type="number" min=".01" step="1" value={Number.isFinite(height) ? height : ""} onChange={event => setHeight(event.target.valueAsNumber)} /></label><label>整体宽度 (cm)<input aria-label="按钮整体宽度厘米" type="number" min=".01" step="1" value={Number.isFinite(width) ? width : ""} onChange={event => setWidth(event.target.valueAsNumber)} /></label><label>采用标尺<select aria-label="标尺轴" value={axis} onChange={event => setAxis(event.target.value)}><option value="height">整体高度</option><option value="width">整体宽度</option></select></label><button disabled={exporting || !valid} onClick={downloadModel}>{exporting ? "正在导出…" : "下载当前模型 GLB"}</button>
      <p className="photo-report-scale-result" data-native-to-meters={nativeToMeters}>当前统一比例：1 native = {nativeToMeters.toFixed(5)} m · 换算整体高度 {(anchor.nativeHeight * nativeToMeters * 100).toFixed(1)} cm / 宽度 {(anchor.nativeWidth * nativeToMeters * 100).toFixed(1)} cm</p>
      {!valid && <p role="alert">请输入大于零的有效尺寸。</p>}{mismatch && <p role="status" className="photo-report-warning">高度与宽度推得的比例相差超过 25%。当前仅采用{axis === "height" ? "高度" : "宽度"}统一缩放，请复核整体尺寸假设。</p>}{exportError && <p role="alert">模型导出失败：{exportError}</p>}
    </section>
    <div id="scene"><ReportScene matchedComparison revision={revision} selection={selection} onSelect={(id, obs) => { setEntityId(id); setObservationId(obs || null); }} imageId={imageId} cameraId={camera?.id || null} onCamera={(id) => { setImageId(id); setObservationId(null); }} onClearSelection={() => setEntityId(null)} inspector={(surface) => selected ? <>
      <section className="photo-report-object-evidence" data-selected-object={selected.id}><h3>{selected.label}</h3><dl><div><dt>{physicalHeight != null ? "整体高度（输入假设）" : "可见高度估计（按标尺换算）"}</dt><dd data-height-native={physicalHeight ?? visibleHeight ?? "unknown"}>{displayValue(physicalHeight ?? visibleHeight)}</dd></div><div><dt>离地间距（条件估计）</dt><dd>{displayValue(clearance?.valueNative)}</dd></div></dl>{selected.observedExtentAvailable === true && record?.visibleHeightRangeNative && <p>跨照片可见高度范围：{record.visibleHeightRangeNative.map(displayValue).join(" – ")}</p>}<p>来源照片：{[...new Set(record?.observations.map(item => item.photo))].join(" / ") || "无"}</p><p>{record?.representation}</p>{clearance?.source && <p>间距依据：{clearance.source}</p>}{physicalHeight == null && <p>可见高度不代表完整物体的物理尺寸。单视图或遮挡部分保留未知。</p>}</section>
      {inclination && <section className="photo-report-angles" data-inclination-entity={selected.id}><h3>板面角度 · 模型估计</h3>
        {surface && <div key={surface.surfaceId} data-inclination-surface={surface.surfaceId}><h4>局部面 {surface.surfaceId}</h4><p>与地面夹角 <strong>{surface.inclinationDeg.toFixed(1)}°</strong>（90° 为垂直）</p><p>偏离垂直 <strong>{surface.deviationFromVerticalDeg.toFixed(1)}°</strong></p><p>{({ non_vertical: "非竖直（模型估计）", vertical: "竖直范围内（模型估计）", direction_unverified: "方向未确认" } as Record<string, string>)[surface.classification] || surface.classification} · 角度离散 {surface.angularSpreadDeg.toFixed(1)}°</p><p>{surface.result.quality.angularErrorDeg == null ? "地面方向误差未记录；相对竖直方向的分类待确认。" : `工程角度误差估计 ${surface.result.quality.angularErrorDeg.toFixed(1)}°，不代表现场标定精度。`}</p></div>}
        {!inclination.surfaces.length && <p>角度不可用：{inclination.reason || inclination.status}</p>}
        {inclination.surfaces.length > 0 && <p>已保存 {inclination.surfaces.length} 个局部拟合面，{surface ? "当前显示所选的 1 个面" : "当前尚未选择局部面"}。从上方“倾斜平面”切换；勾选“全部已测平面”可查看完整列表，所选面的参考线同步显示在 3D 中。</p>}
      </section>}
      {bend && <section className="photo-report-angles" data-bend-entity={selected.id}><h3>板件折弯</h3>{bend.status === "measured" && bend.result ? <p>折弯内角 <strong>{bend.result.value.toFixed(1)}°</strong>（模型估计，摊平为 180°）</p> : <p>折弯角度不可用：{bend.reason || bend.status}</p>}</section>}
      <ObjectFacts entity={selected} document={revision.document} />
      <details className="report-source-details" open><summary>建模来源与假设</summary>{record?.notes.map((note, index) => <p key={index}>{note}</p>)}</details>
    </> : <p>从左侧列表、照片或模型中选择对象。</p>} /></div>
    <section id="sources" className="photo-report-sources"><h2>来源与假设</h2><p>照片和相机保持同一重建坐标系；切换照片时，机器人采用该照片对应姿态。高度和间距依赖按钮整体尺寸与拟合地面，未完成现场标定。</p><p>{data.geometry.floor.status}</p>{anchor.assumptions?.map((note, i) => <p key={i}>{note}</p>)}<a href="scene-report.json" download>下载结构化报告 JSON</a></section>
  </main></SceneResources.Provider>;
}
function LoadPhotoReport() {
  const [data, setData] = useState<PhotoReportData>(), [error, setError] = useState("");
  useEffect(() => { fetch("scene-report.json").then(response => { if (!response.ok) throw Error(`HTTP ${response.status}`); return response.json(); }).then(value => { if (!value.revision?.document || !value.assetURLs || !Array.isArray(value.objects) || !(value.geometry?.anchor?.nativeHeight > 0) || !(value.geometry?.anchor?.nativeWidth > 0)) throw Error("Invalid report contract"); setData(value); }).catch(error => setError(error.message)); }, []);
  return error ? <main className="photo-report"><h1>报告加载失败</h1><p role="alert">{error}</p></main> : data ? <PhotoReport data={data} /> : <main className="photo-report" role="status">正在加载空间报告…</main>;
}
createRoot(document.getElementById("root")!).render(<I18nProvider><LoadPhotoReport /></I18nProvider>);
