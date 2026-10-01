import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { ReportScene } from "./ReportScene";
import { ObjectFacts } from "./WorkcellReport";
import { SceneResources } from "./SceneResources";
import type { BendAnalysis, InclinationAnalysis, SceneMeasurement } from "./SpatialMeasurements";
import { I18nProvider } from "./i18n";
import { activeModel } from "./core";
import { transformMatrix } from "./viewer/native-math";
import type { Revision, Selection, Representation } from "./types";
import "./styles.css";
import "./workcell-report.css";
import "./photo-report.css";

type GroundSample = { valueNative: number | null; pointNative: number[]; footNative: number[]; reason?: string; source?: string; sourcePhotos?: number[]; rangeNative?: number[] };
type CatalogObject = { id: string; label: string; representation: string; notes: string[]; observations: { photo: number }[]; measurements: Record<string, any>; visibleHeightNative?: number; visibleHeightRangeNative?: number[]; visibleHeightByPhoto?: Record<string, number>; groundDistance?: { byPhoto: Record<string, GroundSample>; feature?: GroundSample | null; rangeNative?: number[]; source: string; reason?: string } };
type Calibration = { primaryAxis: string; reference: { scope: string; scopeStatus: string; features: { wholeComponentHeightM: number; mainBodyDiameterM: number; redActuatorDiameterM: number } }; observedEnvelope: { widthM: number }; renderingAssumptions: string[] };
type Comparison = { objectId: string; label: string; method: string; estimateNative: number | null; rangeNative: number[] | null; byPhoto: Record<string, { valueNative: number | null }>; sourcePhotos: number[]; byPhotoMethod?: string; groundTruthM: number; source: string; limitation: string };
type PhotoReportData = { experiment?: { title: string; summary: string; reportURL: string; timingLabel: string }; bendAnalysis?: BendAnalysis; inclinationAnalysis?: InclinationAnalysis; revision: Revision; assetURLs: Record<string, string>; objects: CatalogObject[]; geometry: { calibration?: Calibration; anchor: { nativeHeight: number; nativeWidth: number; assumedHeightM: number; assumedWidthM: number; assumptions?: string[] }; floor: { status: string } }; measurementEvaluation?: { comparisons: Comparison[]; groundTruthUsedForCalibration: boolean }; timing: { oneShotSeconds?: number }; nativeToMetersDefault: number };
function PhotoReport({ data }: { data: PhotoReportData }) {
  const [imageId, setImageId] = useState("photo-4"), [entityId, setEntityId] = useState<string | null>(data.experiment ? "v-guard-left" : "emergency-button"), [observationId, setObservationId] = useState<string | null>(null);
  const [height, setHeight] = useState(data.geometry.anchor.assumedHeightM * 100), [width, setWidth] = useState(data.geometry.anchor.assumedWidthM * 100), [axis, setAxis] = useState("height"), [exporting, setExporting] = useState(false), [exportError, setExportError] = useState("");
  const [showGroundDistance, setShowGroundDistance] = useState(false);
  const anchor = data.geometry.anchor, calibration = data.geometry.calibration, reference = calibration?.reference.features;
  const valid = Number.isFinite(height) && height > 0 && (!!calibration || (Number.isFinite(width) && width > 0));
  const scaleH = height / 100 / anchor.nativeHeight, scaleW = width / 100 / anchor.nativeWidth;
  const nativeToMeters = valid ? calibration || axis === "height" ? scaleH : scaleW : data.nativeToMetersDefault;
  const mismatch = !calibration && valid && Math.abs(scaleH - scaleW) / Math.min(scaleH, scaleW) > .25;
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
    const url = new URL(file, window.location.href); url.searchParams.set("revision", data.revision.documentSha256); return url.href;
  } }), [data]);
  const camera = revision.document.cameras.find(item => item.imageId === imageId);
  const selection: Selection = { projectId: revision.projectId, revisionId: revision.id, entityId, observationId, cameraId: camera?.id || null };
  const selected = revision.document.entities.find(entity => entity.id === entityId), record = data.objects.find(item => item.id === entityId);
  const displayValue = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? `${(value * nativeToMeters).toFixed(3)} m` : "未知";
  const centimeters = (value: number | null | undefined) => value != null && Number.isFinite(value) ? `${(value * 100).toFixed(1)} cm` : "未知";
  const visibleHeight = selected?.observedExtentAvailable !== true ? undefined : (record?.id === "robot" ? record.visibleHeightByPhoto?.[photo] : record?.visibleHeightNative);
  const inclination = data.inclinationAnalysis?.revisionId === revision.id ? data.inclinationAnalysis.items.find(row => row.entityId === entityId) : undefined;
  const bend = data.bendAnalysis?.revisionId === revision.id ? data.bendAnalysis.items.find(row => row.entityId === entityId) : undefined;
  const physicalHeight = record?.measurements.height?.valueNative, distance = record?.groundDistance;
  const ground = distance?.feature ?? distance?.byPhoto[photo], groundRange = distance?.feature?.rangeNative ?? distance?.rangeNative;
  const selectedModel = selected && activeModel(selected);
  const groundAnnotation: SceneMeasurement | null = showGroundDistance && valid && selected && selectedModel && ground?.valueNative != null ? {
    revisionId: revision.id, coordinateFrameId: selectedModel.coordinateFrameId, kind: "ground_distance", source: "source_photo_support",
    value: ground.valueNative, unit: "native", displayLabel: `${displayValue(ground.valueNative)} · 条件估计`, method: "visible-support-to-saved-floor",
    references: [{ entityId: selected.id, representationId: selectedModel.id, assetId: selectedModel.assetId || null, assetSha256: null, placementState: null, qualityStatus: null }],
    lines: [{ points: [ground.pointNative, ground.footNative], color: "#27d3d0" }],
    labelPoint: ground.pointNative.map((v, i) => (v + ground.footNative[i]) / 2), quality: {},
  } : null;
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
      scene.userData = { units: "metres", upAxis: "Y", sourcePhoto: Number(photo), nativeToMeters, scaleHypothesis: { heightCm: height, widthCm: calibration ? null : width, chosenAxis: axis }, suppliedReference: calibration?.reference, assumptions: anchor.assumptions };
      scene.updateMatrixWorld(true);
      const result = await new GLTFExporter().parseAsync(scene, { binary: true });
      const url = URL.createObjectURL(new Blob([result], { type: "model/gltf-binary" }));
      const link = document.createElement("a"); link.href = url; link.download = `workcell-photo-${photo}-${axis}-${axis === "height" ? height : width}cm.glb`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { setExportError(error instanceof Error ? error.message : String(error)); }
    finally { setExporting(false); }
  }
  return <SceneResources.Provider value={resources}><main className="photo-report">
    <header className="photo-report-header"><a className="photo-report-brand" href="#overview">PANOPTES <span>WORKCELL REPORT</span></a><nav><a href="#overview">概览</a><a href="#scene">对象与场景</a><a href="#sources">来源与假设</a></nav></header>
    <section className="photo-report-overview" id="overview"><div><p className="photo-report-eyebrow">四张照片 · 对象级空间重建</p><h1>{reference ? "工作单元测量报告" : "工作单元空间报告"}</h1><p>选取对象查看可见高度、离地间距与证据。拖动分界线，在同一相机下核对照片和模型。</p></div><dl><div><dt>照片</dt><dd>{revision.document.cameras.length}</dd></div><div><dt>对象</dt><dd>{data.objects.length}</dd></div><div><dt>{reference ? "基准整体高度" : data.experiment?.timingLabel || "本次计算"}</dt><dd>{reference ? (reference.wholeComponentHeightM * 100).toFixed(1) : data.timing.oneShotSeconds?.toFixed(1) ?? "—"}<small>{reference ? "cm" : "秒"}</small></dd></div></dl></section>
    {data.experiment && <section className="photo-report-sources" aria-label="已有模型实验结论">{reference ? <details><summary>已有护板模型与角度实验</summary><h2>{data.experiment.title}</h2><p>{data.experiment.summary}</p><a href={data.experiment.reportURL}>查看全部实验、照片证据与测量限制</a></details> : <><h2>{data.experiment.title}</h2><p>{data.experiment.summary}</p><a href={data.experiment.reportURL}>查看全部实验、照片证据与测量限制</a></>}</section>}
    <section className="photo-report-scale" aria-label="标尺尺寸">
      <div><h2>{reference ? "实测急停按钮标尺" : "急停按钮整体尺寸"}</h2><p>{reference ? "整体高度决定统一尺度；红帽与主体直径分别约束标尺模型。围栏、光幕实测值只用于独立对照。" : "红帽 + 黄体 + 灰底。尺寸为输入假设；所有距离与导出模型按同一比例换算。"}</p></div>
      {reference && <dl className="photo-report-reference" data-measured-reference><div><dt>红色触发按钮直径</dt><dd>{centimeters(reference.redActuatorDiameterM)}</dd></div><div><dt>圆形主体最大直径</dt><dd>{centimeters(reference.mainBodyDiameterM)}</dd></div><div><dt>提供的高度</dt><dd>{centimeters(reference.wholeComponentHeightM)}</dd></div></dl>}
      <label>{reference ? "换算用整体高度 (cm)" : "整体高度 (cm)"}<input aria-label="按钮整体高度厘米" type="number" min=".01" step=".1" value={Number.isFinite(height) ? height : ""} onChange={event => setHeight(event.target.valueAsNumber)} /></label>
      {!reference && <><label>整体宽度 (cm)<input aria-label="按钮整体宽度厘米" type="number" min=".01" step="1" value={Number.isFinite(width) ? width : ""} onChange={event => setWidth(event.target.valueAsNumber)} /></label><label>采用标尺<select aria-label="标尺轴" value={axis} onChange={event => setAxis(event.target.value)}><option value="height">整体高度</option><option value="width">整体宽度</option></select></label></>}
      <button disabled={exporting || !valid} onClick={downloadModel}>{exporting ? "正在导出…" : "下载当前模型 GLB"}</button>
      <p className="photo-report-scale-result" data-native-to-meters={nativeToMeters}>当前统一比例：1 native = {nativeToMeters.toFixed(5)} m · 换算整体高度 {(anchor.nativeHeight * nativeToMeters * 100).toFixed(1)} cm{!reference && <> / 宽度 {(anchor.nativeWidth * nativeToMeters * 100).toFixed(1)} cm</>}</p>
      {reference && <p className="photo-report-scale-result">尺寸由用户现场测量提供。{calibration?.reference.scopeStatus === "pending_confirmation" ? `提供的 ${centimeters(reference.wholeComponentHeightM)} 暂对应红帽＋黄体＋灰底的总高度（不含安装支架），该部位对应仍待确认。` : `提供的 ${centimeters(reference.wholeComponentHeightM)} 对应已确认的整体高度。`}修改换算高度后，距离、模型和下表估计同步缩放；独立实测值保持不变。</p>}
      {reference && valid && Math.abs(height / 100 - reference.wholeComponentHeightM) > 1e-8 && <p className="photo-report-scale-result photo-report-warning" role="status">当前是修改标尺后的试算；现场提供的标准高度为 {centimeters(reference.wholeComponentHeightM)}。</p>}
      {reference && <details className="photo-report-scale-result"><summary>照片提取的标尺外形与已知尺寸</summary><p>按提供的高度换算，旧流程的整套可见外轮廓宽约 {centimeters(calibration!.observedEnvelope.widthM)}。它不是单独提取的主体圆盘直径，不能拿来直接除以主体直径 {centimeters(reference.mainBodyDiameterM)} 定尺度。标尺模型已采用提供的两个直径；照片提取、投影方向和地面估计误差仍保留在测量结果中。</p></details>}
      {!valid && <p role="alert">请输入大于零的有效尺寸。</p>}{mismatch && <p role="status" className="photo-report-warning">高度与宽度推得的比例相差超过 25%。当前仅采用{axis === "height" ? "高度" : "宽度"}统一缩放，请复核整体尺寸假设。</p>}{exportError && <p role="alert">模型导出失败：{exportError}</p>}
    </section>
    {data.measurementEvaluation && <section className="photo-report-evaluation" aria-label="离地距离实测对照">
      <h2>离地距离：流程估计与现场实测</h2><p>围栏采用已识别下横杆，光幕外壳采用各来源照片下缘估计的中位数。现场实测离地值未参与尺度、相机或地面的拟合；负误差表示估计偏低。范围反映照片间差异，不是精度保证。</p>
      <div className="photo-report-table-scroll"><table><thead><tr><th scope="col">对象 / 测量部位</th><th scope="col">流程估计</th><th scope="col">现场实测</th><th scope="col">误差（估计－实测）</th><th scope="col">来源范围</th></tr></thead><tbody>{data.measurementEvaluation.comparisons.map(row => {
        const estimate = row.estimateNative == null || !valid ? null : row.estimateNative * nativeToMeters;
        const error = estimate == null ? null : estimate - row.groundTruthM;
        return <tr key={row.objectId} data-measurement-comparison={row.objectId}><th scope="row"><button onClick={() => { setEntityId(row.objectId); setObservationId(null); setShowGroundDistance(true); document.getElementById("scene")?.scrollIntoView({ behavior: "smooth", block: "start" }); }}>{row.label}</button><small>{row.method === "recognized lower rail feature" ? "已识别横杆" : "多照片中位数"} · 照片 {row.sourcePhotos.join(" / ")}</small></th><td data-comparison-estimate>{centimeters(estimate)}</td><td data-comparison-truth>{centimeters(row.groundTruthM)}</td><td data-comparison-error>{error == null ? "未知" : `${error > 0 ? "+" : ""}${centimeters(error)} (${(error / row.groundTruthM * 100).toFixed(1)}%)`}</td><td>{valid && row.rangeNative ? row.rangeNative.map(x => centimeters(x * nativeToMeters)).join(" – ") : "未知"}</td></tr>;
      })}</tbody></table></div>
      <details><summary>逐照片估计与对应部位说明</summary>{data.measurementEvaluation.comparisons.map(row => <article key={row.objectId}><h3>{row.label}</h3><p>{row.source}</p><p>现场量尺端点与当前识别特征的精确对应尚未独立确认。{row.method === "recognized lower rail feature" && "下面逐照片数值是整片可见区域的下界，不是表格中已识别下横杆的逐次量测；遮挡会使两者不同。"}</p><p>{Object.entries(row.byPhoto).map(([p, sample]) => `照片 ${p}：${centimeters(sample.valueNative == null || !valid ? null : sample.valueNative * nativeToMeters)}`).join("；")}</p></article>)}</details>
      <p><a href="measurement-evaluation.json" download>下载全部物体的基准估计 JSON（{centimeters(reference?.wholeComponentHeightM)} 标尺）</a> · <a href="measurements.json" download>下载现场提供的尺寸</a></p>
    </section>}
    <div id="scene"><ReportScene matchedComparison measurementOverride={groundAnnotation} revision={revision} selection={selection} onSelect={(id, obs) => { setEntityId(id); setObservationId(obs || null); setShowGroundDistance(false); }} imageId={imageId} cameraId={camera?.id || null} onCamera={(id) => { setImageId(id); setObservationId(null); }} onClearSelection={() => { setEntityId(null); setShowGroundDistance(false); }} inspector={(surface) => selected ? <>
      <section className="photo-report-object-evidence" data-selected-object={selected.id}><h3>{selected.label}</h3>{bend && <section className="photo-report-angles photo-report-bend" data-bend-entity={selected.id}><h4>本块护板 · 两板面折弯内角</h4>{bend.status === "measured" && bend.result ? <><output>{bend.result.value.toFixed(1)}°</output><p>模型估计 · 摊平为 180°，直角折弯为 90°。</p><p>橙色 / 蓝色：本块板的两个拟合板面；紫色：交线；绿色：折弯内角。</p></> : <p>折弯角度不可用：{bend.reason || bend.status}</p>}</section>}<dl><div><dt>{physicalHeight != null ? reference ? "整体高度（实测标尺换算）" : "整体高度（输入假设）" : "可见高度估计（按标尺换算）"}</dt><dd data-height-native={physicalHeight ?? visibleHeight ?? "unknown"}>{displayValue(physicalHeight ?? visibleHeight)}</dd></div><div><dt>{distance?.feature ? "已识别下横杆离地（条件估计）" : `可见下缘离地（照片 ${photo}）`}</dt><dd data-ground-distance-native={ground?.valueNative ?? "unknown"}>{displayValue(ground?.valueNative)}</dd></div></dl>{selected.observedExtentAvailable === true && record?.visibleHeightRangeNative && <p>跨照片可见高度范围：{record.visibleHeightRangeNative.map(displayValue).join(" – ")}</p>}<p>来源照片：{[...new Set(record?.observations.map(item => item.photo))].join(" / ") || "无"}</p><p>{record?.representation}</p>{ground?.valueNative != null && <button type="button" aria-pressed={showGroundDistance} onClick={() => setShowGroundDistance(value => !value)}>{showGroundDistance ? "隐藏离地测量线" : "显示离地测量线"}</button>}{groundRange && <p data-ground-distance-range>跨照片下缘离地范围：{groundRange.map(displayValue).join(" – ")}；反映可见区域差异，不是精度保证。</p>}<p>间距依据：{ground?.source || distance?.source || "缺少来源数据。"}</p>{ground?.valueNative == null && <p>{ground?.reason || distance?.reason || "当前照片没有可用下缘；请切换到来源照片。"}</p>}{physicalHeight == null && <p>可见高度不代表完整物体的物理尺寸。单视图或遮挡部分保留未知。</p>}</section>
      {inclination && <section className="photo-report-angles" data-inclination-entity={selected.id}><h3>板面角度 · 模型估计</h3>
        {surface && <div key={surface.surfaceId} data-inclination-surface={surface.surfaceId}><h4>局部面 {surface.surfaceId}</h4><p>与地面夹角 <strong>{surface.inclinationDeg.toFixed(1)}°</strong>（90° 为垂直）</p><p>偏离垂直 <strong>{surface.deviationFromVerticalDeg.toFixed(1)}°</strong></p><p>{({ non_vertical: "非竖直（模型估计）", vertical: "竖直范围内（模型估计）", direction_unverified: "方向未确认" } as Record<string, string>)[surface.classification] || surface.classification} · 角度离散 {surface.angularSpreadDeg.toFixed(1)}°</p><p>{surface.result.quality.angularErrorDeg == null ? "地面方向误差未记录；相对竖直方向的分类待确认。" : `工程角度误差估计 ${surface.result.quality.angularErrorDeg.toFixed(1)}°，不代表现场标定精度。`}</p></div>}
        {!inclination.surfaces.length && <p>角度不可用：{inclination.reason || inclination.status}</p>}
        {inclination.surfaces.length > 0 && <p>已保存 {inclination.surfaces.length} 个局部拟合面，{surface ? "当前显示所选的 1 个面" : "当前尚未选择局部面"}。从上方“倾斜平面”切换；勾选“全部已测平面”可查看完整列表，所选面的参考线同步显示在 3D 中。</p>}
      </section>}
      <ObjectFacts entity={selected} document={revision.document} />
      <details className="report-source-details" open><summary>建模来源与假设</summary>{record?.notes.map((note, index) => <p key={index}>{note}</p>)}</details>
    </> : <p>从左侧列表、照片或模型中选择对象。</p>} /></div>
    <section id="sources" className="photo-report-sources"><h2>来源与假设</h2><p>照片和相机保持同一重建坐标系；切换照片时，机器人采用该照片对应姿态。高度和间距依赖按钮整体尺寸与拟合地面；已知标尺不能消除相机、物体边界和地面误差。</p><p>{data.geometry.floor.status}</p>{anchor.assumptions?.map((note, i) => <p key={i}>{note}</p>)}<a href="scene-report.json" download>下载结构化报告 JSON</a></section>
  </main></SceneResources.Provider>;
}
function LoadPhotoReport() {
  const [data, setData] = useState<PhotoReportData>(), [error, setError] = useState("");
  useEffect(() => { fetch("scene-report.json", { cache: "no-cache" }).then(response => { if (!response.ok) throw Error(`HTTP ${response.status}`); return response.json(); }).then(value => { if (!value.revision?.document || !value.assetURLs || !Array.isArray(value.objects) || !(value.geometry?.anchor?.nativeHeight > 0) || !(value.geometry?.anchor?.nativeWidth > 0)) throw Error("Invalid report contract"); setData(value); }).catch(error => setError(error.message)); }, []);
  return error ? <main className="photo-report"><h1>报告加载失败</h1><p role="alert">{error}</p></main> : data ? <PhotoReport data={data} /> : <main className="photo-report" role="status">正在加载空间报告…</main>;
}
createRoot(document.getElementById("root")!).render(<I18nProvider><LoadPhotoReport /></I18nProvider>);
