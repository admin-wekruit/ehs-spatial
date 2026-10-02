import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { ReportScene } from "./ReportScene";
import { ObjectFacts } from "./WorkcellReport";
import { SceneResources } from "./SceneResources";
import { PhotoSemanticExperiment, PhotoSemanticObject, type SemanticExperiment } from "./PhotoSemanticExperiment";
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
type Calibration = { primaryAxis: string; nativeToMeters: number | null; reference: { scope: string; scopeStatus: string; features: { wholeComponentHeightM: number; mainBodyDiameterM: number; redActuatorDiameterM: number } }; observedEnvelope?: { widthM: number | null }; renderingAssumptions?: string[] };
type Comparison = { objectId: string; label: string; method: string; estimateNative: number | null; rangeNative: number[] | null; byPhoto: Record<string, { valueNative: number | null }>; sourcePhotos: number[]; byPhotoMethod?: string; groundTruthM: number; source: string; limitation: string };
type EndpointEstimation = { status: "conditional_unvalidated"; photo: number; method: string; scale: { mPerNative: number; source: string }; endpoints: { objectId: string; label: string; pointNative: number[]; footNative: number[]; heightNative: number; estimateCm: number; rangeCm: number[] }[]; difference: { valueNative: number; valueCm: number; rangeCm: number[]; description: string } };
type PhotoReportData = { semanticExperiment?: SemanticExperiment; modelMeasurementScale: { nativeToMeters: number | null; rangeNativeToMeters?: number[] | null; status: string; source: string }; measurementUpdate?: { kind: string }; endpointEstimation?: EndpointEstimation; metrology?: { summary: string; reportURL: string }; experiment?: { title: string; summary: string; reportURL: string; timingLabel: string }; bendAnalysis?: BendAnalysis; inclinationAnalysis?: InclinationAnalysis; revision: Revision; assetURLs: Record<string, string>; objects: CatalogObject[]; geometry: { calibration?: Calibration; anchor: { nativeHeight: number | null; nativeWidth: number | null; assumedHeightM: number; assumedWidthM: number; mPerNative: number | null; referenceFit?: { status: string; mPerNative: number | null; candidateMPerNative?: number | null; reason?: string; diagnostics?: unknown }; assumptions?: string[] }; floor: { status: string } }; measurementEvaluation?: { comparisons: Comparison[]; groundTruthUsedForCalibration: boolean }; timing: { oneShotSeconds?: number }; nativeToMetersDefault: number | null };
export function PhotoReport({ data }: { data: PhotoReportData }) {
  useEffect(() => { const section = window.location.hash.slice(1); if (["overview", "scene", "semantics", "sources"].includes(section)) document.getElementById(section)?.scrollIntoView(); }, []);
  const entry = new URL(window.location.href).searchParams;
  const entryImage = `photo-${entry.get("photo")}`, entryObject = entry.get("object"), entryView = entry.get("view");
  const defaultEndpoints = data.endpointEstimation?.status === "conditional_unvalidated" && !["photo", "object", "view", "measurement"].some(key => entry.has(key));
  const initialView = ["photo", "point_cloud", "model", "compare"].includes(entryView || "") ? entryView as "photo" | "point_cloud" | "model" | "compare" : defaultEndpoints ? "model" : undefined;
  const [imageId, setImageId] = useState(data.revision.document.cameras.some(camera => camera.imageId === entryImage) ? entryImage : "photo-4"), [entityId, setEntityId] = useState<string | null>(entryObject && data.objects.some(item => item.id === entryObject) ? entryObject : data.endpointEstimation ? "post-box-1" : data.experiment ? "v-guard-left" : "emergency-button"), [observationId, setObservationId] = useState<string | null>(null);
  const [height, setHeight] = useState((data.geometry.calibration?.reference.features.wholeComponentHeightM ?? data.geometry.anchor.assumedHeightM) * 100), [exporting, setExporting] = useState(false), [exportError, setExportError] = useState("");
  const [showGroundDistance, setShowGroundDistance] = useState(false);
  const [showEndpointComparison, setShowEndpointComparison] = useState(entry.get("measurement") === "endpoints" || defaultEndpoints);
  const anchor = data.geometry.anchor, calibration = data.geometry.calibration, reference = calibration?.reference.features;
  const referenceHeight = reference?.wholeComponentHeightM ?? anchor.assumedHeightM;
  const valid = Number.isFinite(height) && height > 0 && Number.isFinite(referenceHeight) && referenceHeight > 0;
  const referenceRatio = valid ? height / 100 / referenceHeight : null;
  const acceptedScale = anchor.referenceFit?.status === "available" && anchor.referenceFit.mPerNative === anchor.mPerNative && anchor.mPerNative != null && Number.isFinite(anchor.mPerNative) && anchor.mPerNative > 0 ? anchor.mPerNative : null;
  // ponytail: all three dimensions change together; changing their ratios requires a new oneshot fit.
  const nativeToMeters = acceptedScale != null && referenceRatio != null && Number.isFinite(acceptedScale * referenceRatio) && acceptedScale * referenceRatio > 0 ? acceptedScale * referenceRatio : null;
  const modelBase = data.modelMeasurementScale;
  const modelScale = {
    nativeToMeters: referenceRatio != null && modelBase.nativeToMeters != null && Number.isFinite(modelBase.nativeToMeters * referenceRatio) && modelBase.nativeToMeters * referenceRatio > 0 ? modelBase.nativeToMeters * referenceRatio : null,
    status: modelBase.status,
    source: referenceRatio === 1 ? modelBase.source : `三尺寸同比试算 × ${referenceRatio ?? "未知"}；基准依据：${modelBase.source}`,
    rangeNativeToMeters: referenceRatio == null ? null : modelBase.rangeNativeToMeters?.map(value => value * referenceRatio) ?? null,
    uniformReferenceRatio: referenceRatio,
    currentReferenceDimensionsM: reference && referenceRatio != null ? Object.fromEntries(Object.entries(reference).map(([key, value]) => [key, value * referenceRatio])) : null,
    baseModelMeasurementScale: modelBase,
  };
  const modelCentimeters = (native: number) => modelScale.nativeToMeters == null ? `${native.toFixed(4)} native` : `${(native * modelScale.nativeToMeters * 100).toFixed(2)} cm`;
  const photo = imageId.replace("photo-", "");
  const revision = useMemo(() => ({ ...data.revision, document: { ...data.revision.document,
    coordinateFrames: data.revision.document.coordinateFrames.map(frame => ({ ...frame, scale: { ...frame.scale, status: nativeToMeters == null ? "uncalibrated" as const : "model_estimated" as const, nativeToMeters } })),
    entities: data.revision.document.entities.map(entity => {
      const variant = (entity.modelVariants as Record<string, Representation> | undefined)?.[photo];
      return variant ? { ...entity, representations: [variant, ...(entity.representations || []).filter(rep => !["generated_mesh", "primitive"].includes(rep.kind))], activeModelRepresentationId: variant.id, currentModelTransform: variant.transform } : entity;
    }),
  } }), [data, nativeToMeters, photo]);
  const resources = useMemo(() => ({ analysisAvailable: false, bendAnalysis: data.bendAnalysis, inclinationAnalysis: data.inclinationAnalysis, resolveAsset: async (id: string) => {
    const file = data.assetURLs[id]; if (!file) throw Error(`Missing asset: ${id}`);
    const url = new URL(file, window.location.href); url.searchParams.set("revision", data.revision.documentSha256); return url.href;
  } }), [data]);
  const camera = revision.document.cameras.find(item => item.imageId === imageId);
  const selection: Selection = { projectId: revision.projectId, revisionId: revision.id, entityId, observationId, cameraId: camera?.id || null };
  const selected = revision.document.entities.find(entity => entity.id === entityId), record = data.objects.find(item => item.id === entityId);
  const selectSemanticObject = (id: string, sourcePhoto?: number, sourceObservationId?: string) => {
    const target = revision.document.entities.find(item => item.id === id);
    if (!target || !data.objects.some(item => item.id === id)) return;
    const sourceImage = sourcePhoto == null ? null : `photo-${sourcePhoto}`;
    const nextImage = sourceImage && revision.document.cameras.some(item => item.imageId === sourceImage) ? sourceImage : imageId;
    setImageId(nextImage);
    const observations = revision.document.observations.filter(item => target.observationRefs?.includes(item.id) && item.imageId === nextImage);
    const sourceObservation = observations.find(item => item.id === sourceObservationId) ?? observations[0];
    setEntityId(id); setObservationId(sourceObservation?.id ?? null); setShowGroundDistance(false); setShowEndpointComparison(false);
    document.getElementById("scene")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const displayValue = (value: unknown) => nativeToMeters != null && typeof value === "number" && Number.isFinite(value) ? `${(value * nativeToMeters).toFixed(3)} m` : "未知";
  const centimeters = (value: number | null | undefined) => value != null && Number.isFinite(value) ? `${(value * 100).toFixed(1)} cm` : "未知";
  const visibleHeight = selected?.observedExtentAvailable !== true ? undefined : (record?.id === "robot" ? record.visibleHeightByPhoto?.[photo] : record?.visibleHeightNative);
  const inclination = data.inclinationAnalysis?.revisionId === revision.id ? data.inclinationAnalysis.items.find(row => row.entityId === entityId) : undefined;
  const bend = data.bendAnalysis?.revisionId === revision.id ? data.bendAnalysis.items.find(row => row.entityId === entityId) : undefined;
  const physicalHeight = record?.measurements.height?.valueNative, distance = record?.groundDistance;
  const ground = distance?.feature, groundRange = ground?.rangeNative;
  const sourceEdgeAvailable = ground?.valueNative != null && Number.isFinite(ground.valueNative) && ground.valueNative >= 0 && ground.pointNative?.length === 3 && ground.footNative?.length === 3 && [...ground.pointNative, ...ground.footNative].every(Number.isFinite);
  const groundValue = sourceEdgeAvailable ? ground.valueNative : null;
  const selectedModel = selected && activeModel(selected);
  const groundAnnotation: SceneMeasurement | null = showGroundDistance && sourceEdgeAvailable && selected && selectedModel && ground?.valueNative != null ? {
    revisionId: revision.id, coordinateFrameId: selectedModel.coordinateFrameId, kind: "ground_distance", source: "source_photo_support",
    value: ground.valueNative, unit: "native", displayLabel: nativeToMeters == null ? "源边缘到地面 · 尺度未知" : `${displayValue(ground.valueNative)} · 条件估计`, method: "physical-source-edge-to-ground",
    references: [{ entityId: selected.id, representationId: selectedModel.id, assetId: selectedModel.assetId || null, assetSha256: null, placementState: null, qualityStatus: null }],
    lines: [{ points: [ground.pointNative, ground.footNative], color: "#27d3d0" }],
    labelPoint: ground.pointNative.map((v, i) => (v + ground.footNative[i]) / 2), quality: {},
  } : null;
  const endpointEstimate = data.endpointEstimation;
  const endpointRange = (values: number[]) => {
    const factor = modelScale.nativeToMeters;
    if (factor == null || !endpointEstimate) return "未知";
    return values.map(value => (value / endpointEstimate.scale.mPerNative * factor).toFixed(1)).join(" – ") + " cm";
  };
  const curtainEndpoint = endpointEstimate?.endpoints.find(point => point.objectId === "post-box-1"), fenceEndpoint = endpointEstimate?.endpoints.find(point => point.objectId === "fence-0");
  const endpoints = curtainEndpoint && fenceEndpoint ? [curtainEndpoint, fenceEndpoint] : null;
  const endpointTruth = data.measurementEvaluation?.comparisons.filter(row => ["fence-0", "post-box-1"].includes(row.objectId));
  const endpointAnnotation: SceneMeasurement | null = showEndpointComparison && endpointEstimate?.status === "conditional_unvalidated" && imageId === `photo-${endpointEstimate.photo}` && endpoints?.some(point => point.objectId === selected?.id) && selectedModel?.coordinateFrameId === "workcell-floor" ? {
    revisionId: revision.id, coordinateFrameId: selectedModel.coordinateFrameId, kind: "ground_distance", source: "source_photo_support", method: "conditional-endpoint-comparison",
    value: endpointEstimate.difference.valueNative, unit: "native", displayLabel: `底边离地高差（光幕 − 围栏）：${modelCentimeters(endpointEstimate.difference.valueNative)} · 条件模型估计，未验证`,
    references: [{ entityId: selected!.id, representationId: selectedModel.id, assetId: selectedModel.assetId || null, assetSha256: null, placementState: null, qualityStatus: null }],
    // ponytail: the saved report frame is floor Z-up; the orange segment is a vertical difference, never the distance between objects.
    lines: [...endpoints.map((point, index) => ({ points: [point.pointNative, point.footNative], color: index ? "#b087ff" : "#27d3d0" })), { points: [endpoints[0].pointNative, [endpoints[1].pointNative[0], endpoints[1].pointNative[1], endpoints[0].pointNative[2]]], color: "#edbe38" }, { points: [[endpoints[1].pointNative[0], endpoints[1].pointNative[1], endpoints[0].pointNative[2]], endpoints[1].pointNative], color: "#ff8e45" }],
    labelPoint: [endpoints[1].pointNative[0], endpoints[1].pointNative[1], (endpoints[0].pointNative[2] + endpoints[1].pointNative[2]) / 2], quality: {},
  } : null;
  const endpointCard = endpointEstimate?.status === "conditional_unvalidated" && endpoints && <section className="photo-report-object-evidence" aria-label="围栏与光幕底边离地估计">
    <h3>围栏与光幕：底边离地</h3><p>条件模型估计，未验证 · 模型底边到同一估计地面</p>
    <dl>{[...endpoints].reverse().map(point => <div key={point.objectId}><dt>{point.objectId === "fence-0" ? "围栏下横杆底边离地" : "光幕壳体底边离地"}</dt><dd data-endpoint-estimate={point.objectId}>{modelCentimeters(point.heightNative)}</dd></div>)}<div><dt>底边离地高差（光幕 − 围栏）</dt><dd data-endpoint-difference>{modelCentimeters(endpointEstimate.difference.valueNative)}</dd></div></dl>
    <button type="button" aria-pressed={showEndpointComparison} onClick={() => { setShowEndpointComparison(value => !value); setImageId(`photo-${endpointEstimate.photo}`); setEntityId("post-box-1"); setObservationId(null); }}>{showEndpointComparison ? "隐藏底边离地测量线" : "显示底边离地测量线"}</button>
  </section>;
  async function downloadModel() {
    setExporting(true); setExportError("");
    try {
      const base = new URL("viewer-assets/", window.location.href).href;
      const THREE = await import(/* @vite-ignore */ `${base}three.module.js`);
      const { GLTFLoader } = await import(/* @vite-ignore */ `${base}addons/loaders/GLTFLoader.js`);
      const { GLTFExporter } = await import(/* @vite-ignore */ `${base}addons/exporters/GLTFExporter.js`);
      const scene = new THREE.Scene(), group = new THREE.Group(), loader = new GLTFLoader();
      scene.add(group); group.name = modelScale.nativeToMeters == null ? "Workcell — native units, Y up" : "Workcell — metres, Y up";
      if (modelScale.nativeToMeters != null) group.scale.setScalar(modelScale.nativeToMeters);
      group.rotation.x = -Math.PI / 2;
      for (const entity of revision.document.entities.filter(entity => entity.visible !== false && !entity.sourceContext)) {
        const rep = activeModel(entity); if (!rep?.assetId || rep.sourceValidity === "stale") continue;
        const model = (await loader.loadAsync(await resources.resolveAsset(rep.assetId))).scene;
        const placed = new THREE.Group(); placed.name = entity.label || entity.id;
        placed.matrix.fromArray(transformMatrix(entity.currentModelTransform || rep.transform)); placed.matrixAutoUpdate = false;
        placed.userData = { entityId: entity.id, sourceRepresentation: rep.id, physicalDimensionsUnknown: entity.physicalDimensionsUnknown === true };
        placed.add(model); group.add(placed);
      }
      scene.userData = { units: modelScale.nativeToMeters == null ? "native" : "metres", upAxis: "Y", ground: { normal: [0, 1, 0], offset: 0 }, sourcePhoto: Number(photo), nativeToMeters: modelScale.nativeToMeters, scaleStatus: modelScale.nativeToMeters == null ? "uncalibrated" : modelScale.status, modelMeasurementScale: modelScale, groundTruth: false, uniformReferenceRatio: referenceRatio, suppliedReference: calibration?.reference, referenceFitStatus: anchor.referenceFit?.status, assumptions: anchor.assumptions };
      scene.updateMatrixWorld(true);
      const result = await new GLTFExporter().parseAsync(scene, { binary: true });
      const url = URL.createObjectURL(new Blob([result], { type: "model/gltf-binary" }));
      const link = document.createElement("a"); link.href = url; link.download = `workcell-photo-${photo}-${modelScale.nativeToMeters == null ? "native-unscaled" : `${height}cm-${modelScale.status}`}.glb`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { setExportError(error instanceof Error ? error.message : String(error)); }
    finally { setExporting(false); }
  }
  return <SceneResources.Provider value={resources}><main className="photo-report">
    <header className="photo-report-header"><a className="photo-report-brand" href="#overview">PANOPTES <span>WORKCELL REPORT</span></a><nav><a href="#overview">概览</a><a href="#scene">对象与场景</a>{data.semanticExperiment && <a href="#semantics">语义实验</a>}<a href="#sources">来源与假设</a></nav></header>
    <section className="photo-report-overview" id="overview"><div><p className="photo-report-eyebrow">四张照片 · 对象级空间重建</p><h1>{reference ? "工作单元测量报告" : "工作单元空间报告"}</h1><p>选取对象查看可见高度、离地间距与证据。拖动分界线，在同一相机下核对照片和模型。</p></div><dl><div><dt>照片</dt><dd>{revision.document.cameras.length}</dd></div><div><dt>对象</dt><dd>{data.objects.length}</dd></div><div><dt>{reference ? "基准整体高度" : data.experiment?.timingLabel || "本次计算"}</dt><dd>{reference ? (reference.wholeComponentHeightM * 100).toFixed(1) : data.timing.oneShotSeconds?.toFixed(1) ?? "—"}<small>{reference ? "cm" : "秒"}</small></dd></div>{reference && <div><dt>{data.measurementUpdate?.kind === "saved-geometry-replay" ? "原始完整流程" : "本次端到端"}</dt><dd>{data.timing.oneShotSeconds?.toFixed(1) ?? "—"}<small>秒</small></dd></div>}</dl></section>
    {endpointEstimate?.status === "conditional_unvalidated" && endpoints && <section className="photo-report-evaluation photo-report-endpoints" aria-label="底边离地条件模型估计（未验证）">
      {endpointCard}<p><a className="photo-report-endpoint-link" href={`?photo=${endpointEstimate.photo}&object=post-box-1&view=model&measurement=endpoints#scene`}>查看照片 {endpointEstimate.photo} · 围栏与光幕底边离地模型</a></p>
      {endpointTruth?.length ? <p>现场提供的对照值：{endpointTruth.map(row => `${row.objectId === "fence-0" ? "围栏底边离地" : "光幕壳体底边离地"} ${(row.groundTruthM * 100).toFixed(2)} cm`).join("；")}。数值在开发中已知，未输入本组估计；不是盲测精度验证，实物端点对应仍待确认。</p> : null}
      <details><summary>查看条件估计的来源与范围</summary><p>当前 GLB 模型的局部底面到同一估计地面；数值使用本次按钮条件比例，不代表源图物理底端已验证。修改标尺会同步更新这里、卡尺和下载模型；原生几何不变。</p>
      <p>{endpoints.map(point => `${point.label}敏感范围 ${endpointRange(point.rangeCm)}`).join("；")}；高差敏感范围 {endpointRange(endpointEstimate.difference.rangeCm)}。</p>
      <p>{endpointEstimate.difference.description} 正值表示光幕底端更高；范围只反映本次选点与比例敏感性，不是精度保证。</p><p>比例：1 native = {modelScale.nativeToMeters?.toFixed(5) ?? "未知"} m。{modelScale.source}</p><p>方法：{endpointEstimate.method}</p>
      <p>青色、紫色竖线：两个模型底端到地面。橙色竖线：沿地面法向的高低差；黄色线仅连接同一高度。</p></details>
    </section>}
    {data.metrology && <section className="photo-report-sources" aria-label="按钮标尺测量实验"><h2>按钮标尺 · 小于 3 cm 的测量实验</h2><p>{data.metrology.summary}</p><a href={data.metrology.reportURL}>查看本次结果、原图证据与失败原因</a></section>}
    {data.experiment && <section className="photo-report-sources" aria-label="已有模型实验结论">{reference ? <details><summary>已有护板模型与角度实验</summary><h2>{data.experiment.title}</h2><p>{data.experiment.summary}</p><a href={data.experiment.reportURL}>查看全部实验、照片证据与测量限制</a></details> : <><h2>{data.experiment.title}</h2><p>{data.experiment.summary}</p><a href={data.experiment.reportURL}>查看全部实验、照片证据与测量限制</a></>}</section>}
    <section className="photo-report-scale" aria-label="标尺尺寸">
      <div><h2>统一模型标尺 · 急停按钮</h2><p>卡尺、底边比较、地面网格和下载模型共用以下比例。三尺寸联合标定状态与模型条件估计分别记录。</p></div>
      {reference && <dl className="photo-report-reference" data-measured-reference><div><dt>当前红帽直径</dt><dd>{centimeters(referenceRatio == null ? null : reference.redActuatorDiameterM * referenceRatio)}</dd></div><div><dt>当前主体最大直径</dt><dd>{centimeters(referenceRatio == null ? null : reference.mainBodyDiameterM * referenceRatio)}</dd></div><div><dt>当前整体高度</dt><dd>{centimeters(valid ? height / 100 : null)}</dd></div></dl>}
      <label>同比试算 · 整体高度 (cm)<input aria-label="按钮整体高度厘米" type="number" min=".01" step=".1" value={Number.isFinite(height) ? height : ""} onChange={event => setHeight(event.target.valueAsNumber)} /></label>
      <button disabled={exporting} onClick={downloadModel}>{exporting ? "正在导出…" : modelScale.nativeToMeters == null ? "下载原生模型 GLB（未标定）" : modelScale.status === "conditional_unvalidated" ? "下载条件标尺模型 GLB" : "下载当前模型 GLB"}</button>
      <p className="photo-report-scale-result" data-model-native-to-meters={modelScale.nativeToMeters ?? "unknown"}>{modelScale.nativeToMeters == null ? "模型卡尺：原生单位，尺度未知。" : `模型卡尺统一比例：1 native = ${modelScale.nativeToMeters.toFixed(5)} m · ${modelScale.status === "conditional_unvalidated" ? "条件估计，未验证" : "三维参考拟合"}`}</p><p>{modelScale.source}</p>
      <p className="photo-report-scale-result" data-native-to-meters={nativeToMeters ?? "unknown"}>{nativeToMeters == null ? "现场物理尺度未验证；模型卡尺的条件厘米值不代表已验证的物理尺寸。" : `当前统一比例：1 native = ${nativeToMeters.toFixed(5)} m · 三维标尺条件估计`}</p>
      <p className="photo-report-scale-result">修改高度会让红帽直径、主体直径同比变化。单独改变三者的比例需要重新生成报告；这里的试算不会重做相机或几何拟合。</p>
      {reference && <p className="photo-report-scale-result">现场提供：整体高度 {centimeters(reference.wholeComponentHeightM)}、主体直径 {centimeters(reference.mainBodyDiameterM)}、红帽直径 {centimeters(reference.redActuatorDiameterM)}。{calibration?.reference.scopeStatus === "pending_confirmation" ? "整体高度所对应的部位仍待确认。" : "整体高度所对应的部位已确认。"}</p>}
      {referenceRatio != null && Math.abs(referenceRatio - 1) > 1e-8 && <p className="photo-report-scale-result photo-report-warning" role="status">当前为三尺寸同比试算；现场实测对照值保持不变。</p>}
      {acceptedScale == null && <p className="photo-report-scale-result photo-report-warning" role="status">三尺寸联合标尺未通过验证；同比试算不会改变这个验证状态。{anchor.referenceFit?.reason}</p>}
      {!valid && <p role="alert">请输入大于零的有效尺寸。</p>}{exportError && <p role="alert">模型导出失败：{exportError}</p>}
    </section>
    {data.measurementEvaluation && <section className="photo-report-evaluation" aria-label="离地距离实测对照">
      <h2>离地距离：流程估计与现场实测</h2><p>估计只采用多视角支持的同一物理下缘到地面的距离；缺少支持时保留未知。现场实测离地值未参与拟合；负误差表示估计偏低，图像几何敏感范围不是精度保证。</p>
      <div className="photo-report-table-scroll"><table><thead><tr><th scope="col">对象 / 测量部位</th><th scope="col">流程估计</th><th scope="col">现场实测</th><th scope="col">误差（估计－实测）</th><th scope="col">来源范围</th></tr></thead><tbody>{data.measurementEvaluation.comparisons.map(row => {
        const estimate = row.estimateNative == null || nativeToMeters == null ? null : row.estimateNative * nativeToMeters;
        const error = estimate == null ? null : estimate - row.groundTruthM;
        return <tr key={row.objectId} data-measurement-comparison={row.objectId}><th scope="row"><button onClick={() => { setEntityId(row.objectId); setObservationId(null); setShowGroundDistance(true); document.getElementById("scene")?.scrollIntoView({ behavior: "smooth", block: "start" }); }}>{row.label}</button><small>{row.estimateNative == null ? "来源支持不足" : "物理下缘"} · 照片 {row.sourcePhotos.join(" / ")}</small></th><td data-comparison-estimate>{centimeters(estimate)}</td><td data-comparison-truth>{centimeters(row.groundTruthM)}</td><td data-comparison-error>{error == null ? "未知" : `${error > 0 ? "+" : ""}${centimeters(error)} (${(error / row.groundTruthM * 100).toFixed(1)}%)`}</td><td>{nativeToMeters != null && row.rangeNative ? row.rangeNative.map(x => centimeters(x * nativeToMeters)).join(" – ") : "未知"}</td></tr>;
      })}</tbody></table></div>
      <details><summary>来源与对应部位说明</summary>{data.measurementEvaluation.comparisons.map(row => <article key={row.objectId}><h3>{row.label}</h3><p>{row.source}</p><p>{row.limitation}</p><p>来源照片：{row.sourcePhotos.join(" / ") || "无可用多视角支持"}</p></article>)}</details>
      <p><a href="measurement-evaluation.json" download>下载全部物体的基准估计 JSON（{centimeters(reference?.wholeComponentHeightM)} 标尺）</a> · <a href="measurements.json" download>下载现场提供的尺寸</a></p>
    </section>}
    {data.semanticExperiment && <PhotoSemanticExperiment data={data.semanticExperiment} selectedEntityId={entityId} onSelect={selectSemanticObject} />}
    <div id="scene"><nav className="photo-report-focus" aria-label="重点检查对象"><strong>重点检查</strong>{[["post-box-1", "右侧光幕底部"], ["fence-0", "围栏下沿"], ["floor", "地面"]].filter(([id]) => data.objects.some(item => item.id === id)).map(([id, label]) => <button key={id} type="button" aria-pressed={entityId === id} onClick={() => { setEntityId(id); setImageId("photo-4"); setObservationId(null); setShowGroundDistance(true); setShowEndpointComparison(id === "post-box-1" || id === "fence-0"); }}>{label}</button>)}<small>先选对象，再切换照片、原始点云和模型核对。</small></nav><ReportScene matchedComparison measurementScale={modelScale} initialView={initialView} measurementOverride={endpointAnnotation || groundAnnotation} revision={revision} selection={selection} onSelect={(id, obs) => { setEntityId(id); setObservationId(obs || null); setShowGroundDistance(false); }} imageId={imageId} cameraId={camera?.id || null} onCamera={(id) => { setImageId(id); setObservationId(null); }} onClearSelection={() => { setEntityId(null); setShowGroundDistance(false); }} inspector={(surface) => selected ? <>
      {data.semanticExperiment && <PhotoSemanticObject data={data.semanticExperiment} entityId={selected.id} onSelect={selectSemanticObject} />}
      {["post-box-1", "fence-0"].includes(selected.id) && endpointCard}<section className="photo-report-object-evidence" data-selected-object={selected.id}><h3>{selected.label}</h3>{bend && <section className="photo-report-angles photo-report-bend" data-bend-entity={selected.id}><h4>本块护板 · 两板面折弯内角</h4>{bend.status === "measured" && bend.result ? <><output>{bend.result.value.toFixed(1)}°</output><p>模型估计 · 摊平为 180°，直角折弯为 90°。</p><p>橙色 / 蓝色：本块板的两个拟合板面；紫色：交线；绿色：折弯内角。</p></> : <p>折弯角度不可用：{bend.reason || bend.status}</p>}</section>}<dl><div><dt>{record?.id === "emergency-button" && reference ? "输入整体高度（标准尺寸）" : physicalHeight != null ? "整体高度（按标尺换算）" : "可见高度估计（按标尺换算）"}</dt><dd data-height-native={physicalHeight ?? visibleHeight ?? "unknown"}>{record?.id === "emergency-button" && reference ? centimeters(valid ? height / 100 : null) : displayValue(physicalHeight ?? visibleHeight)}</dd></div><div><dt>物理下缘离地（条件估计）</dt><dd data-ground-distance-native={groundValue ?? "unknown"}>{displayValue(groundValue)}</dd></div></dl>{selected.observedExtentAvailable === true && record?.visibleHeightRangeNative && <p>跨照片可见高度范围：{record.visibleHeightRangeNative.map(displayValue).join(" – ")}</p>}<p>来源照片：{[...new Set(record?.observations.map(item => item.photo))].join(" / ") || "无"}</p><p>{record?.representation}</p>{sourceEdgeAvailable && <button type="button" aria-pressed={showGroundDistance} onClick={() => setShowGroundDistance(value => !value)}>{showGroundDistance ? "隐藏离地测量线" : "显示离地测量线"}</button>}{groundRange && <p data-ground-distance-range>图像几何敏感范围：{groundRange.map(displayValue).join(" – ")}；不包含全部相机和地面系统误差。</p>}<p>间距依据：{ground?.source || distance?.source || "缺少来源数据。"}</p>{!sourceEdgeAvailable && <p>{ground?.reason || distance?.reason || "缺少稳定的多视角物理下缘支持，离地距离保持未知。"}</p>}{physicalHeight == null && <p>可见高度不代表完整物体的物理尺寸。单视图或遮挡部分保留未知。</p>}</section>
      {inclination && <section className="photo-report-angles" data-inclination-entity={selected.id}><h3>板面角度 · 模型估计</h3>
        {surface && <div key={surface.surfaceId} data-inclination-surface={surface.surfaceId}><h4>局部面 {surface.surfaceId}</h4><p>与地面夹角 <strong>{surface.inclinationDeg.toFixed(1)}°</strong>（90° 为垂直）</p><p>偏离垂直 <strong>{surface.deviationFromVerticalDeg.toFixed(1)}°</strong></p><p>{({ non_vertical: "非竖直（模型估计）", vertical: "竖直范围内（模型估计）", direction_unverified: "方向未确认" } as Record<string, string>)[surface.classification] || surface.classification} · 角度离散 {surface.angularSpreadDeg.toFixed(1)}°</p><p>{surface.result.quality.angularErrorDeg == null ? "地面方向误差未记录；相对竖直方向的分类待确认。" : `工程角度误差估计 ${surface.result.quality.angularErrorDeg.toFixed(1)}°，不代表现场标定精度。`}</p></div>}
        {!inclination.surfaces.length && <p>角度不可用：{inclination.reason || inclination.status}</p>}
        {inclination.surfaces.length > 0 && <p>已保存 {inclination.surfaces.length} 个局部拟合面，{surface ? "当前显示所选的 1 个面" : "当前尚未选择局部面"}。从上方“倾斜平面”切换；勾选“全部已测平面”可查看完整列表，所选面的参考线同步显示在 3D 中。</p>}
      </section>}
      {nativeToMeters == null && <p>以下模型位置使用原生坐标单位，尚未换算为米。</p>}
      <ObjectFacts entity={selected} document={revision.document} />
      <details className="report-source-details" open><summary>建模来源与假设</summary>{record?.notes.map((note, index) => <p key={index}>{note}</p>)}</details>
    </> : <p>从左侧列表、照片或模型中选择对象。</p>} /></div>
    <section id="sources" className="photo-report-sources"><h2>来源与假设</h2><p>照片和相机保持同一重建坐标系；切换照片时，机器人采用该照片对应姿态。物理米制测量需要受支持的三维按钮标尺、物理边缘与地面。当前标尺拟合：{anchor.referenceFit?.status === "available" ? "条件支持" : "未通过验证"}；已知标尺不能消除相机、物体边界和地面误差。</p><p>{data.geometry.floor.status}</p>{anchor.assumptions?.map((note, i) => <p key={i}>{note}</p>)}<a href="scene-report.json" download>下载结构化报告 JSON</a></section>
  </main></SceneResources.Provider>;
}
function LoadPhotoReport() {
  const [data, setData] = useState<PhotoReportData>(), [error, setError] = useState("");
  useEffect(() => { fetch("scene-report.json", { cache: "no-cache" }).then(response => { if (!response.ok) throw Error(`HTTP ${response.status}`); return response.json(); }).then(value => { if (!value.revision?.document || !value.assetURLs || !Array.isArray(value.objects) || !value.geometry?.anchor || !(value.geometry.anchor.mPerNative == null || (Number.isFinite(value.geometry.anchor.mPerNative) && value.geometry.anchor.mPerNative > 0))) throw Error("Invalid report contract"); setData(value); }).catch(error => setError(error.message)); }, []);
  return error ? <main className="photo-report"><h1>报告加载失败</h1><p role="alert">{error}</p></main> : data ? <PhotoReport data={data} /> : <main className="photo-report" role="status">正在加载空间报告…</main>;
}
createRoot(document.getElementById("root")!).render(<I18nProvider><LoadPhotoReport /></I18nProvider>);
