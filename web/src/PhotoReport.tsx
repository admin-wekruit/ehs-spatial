import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { ReportScene } from "./ReportScene";
import { ObjectFacts } from "./WorkcellReport";
import { SceneResources } from "./SceneResources";
import { PhotoSemanticExperiment, PhotoSemanticObject, type SemanticExperiment, type SpatialFact } from "./PhotoSemanticExperiment";
import type { BendAnalysis, InclinationAnalysis, SceneMeasurement } from "./SpatialMeasurements";
import { I18nProvider } from "./i18n";
import { activeModel } from "./core";
import { transformMatrix } from "./viewer/native-math";
import type { Revision, Selection, Representation, Entity } from "./types";
import "./styles.css";
import "./workcell-report.css";
import "./photo-report.css";

type GroundSample = { valueNative: number | null; pointNative: number[]; footNative: number[]; reason?: string; source?: string; sourcePhotos?: number[]; rangeNative?: number[] };
type CatalogObject = { id: string; label: string; representation: string; notes: string[]; observations: { photo: number }[]; measurements: Record<string, any>; physicalBottom?: { geometryScope: string }; modelTerminal?: { candidateStatus: string; acceptedForPhysicalUse: boolean }; visibleHeightNative?: number; visibleHeightRangeNative?: number[]; visibleHeightByPhoto?: Record<string, number>; groundDistance?: { byPhoto: Record<string, GroundSample>; feature?: GroundSample | null; rangeNative?: number[]; source: string; reason?: string } };
type Calibration = { primaryAxis: string; nativeToMeters: number | null; reference: { scope: string; scopeStatus: string; features: { wholeComponentHeightM: number; mainBodyDiameterM: number; redActuatorDiameterM: number } }; observedEnvelope?: { widthM: number | null }; renderingAssumptions?: string[] };
type Comparison = { objectId: string; label: string; method: string; estimateNative: number | null; rangeNative: number[] | null; byPhoto: Record<string, { valueNative: number | null }>; sourcePhotos: number[]; byPhotoMethod?: string; groundTruthM: number; source: string; limitation: string };
/** One model terminal measured on this revision's displayed representation, in native units. */
export type Endpoint = { id: string; objectId: string; label: string; side: "left" | "right" | null; measurementScope: string; pointNative: number[]; footNative: number[]; heightNative: number; representationId: string; assetId: string; assetSha256: string; modelFile: string; modelSha256: string; pairedEndpointId?: string; provenance: string };
type EndpointDifference = { id: string; label: string; minuendId: string; subtrahendId: string; valueNative: number; description: string };
type EndpointEstimation = { status: "conditional_unvalidated"; sidePhoto: number; method: string; endpoints: Endpoint[]; differences: EndpointDifference[] };
export type RevisionChoice = { id: string; label: string; branchId: string; status: string; documentSha256: string; parentRevisionId: string | null; url: string };
type Lineage = { role?: string; label?: string; parentRevisionId?: string | null; candidateEvidence?: { url?: string | null; acceptedForPhysicalUse?: boolean } };
export type PhotoReportData = { semanticExperiment?: SemanticExperiment; semanticBinding?: { status: string; reason: string; action?: string }; modelMeasurementScale: { nativeToMeters: number | null; rangeNativeToMeters?: number[] | null; status: string; source: string }; measurementUpdate?: { kind: string; revisionBuildSeconds?: number; sourceRevisionId?: string | null }; endpointEstimation?: EndpointEstimation; revisionChoices?: RevisionChoice[]; lineage?: Lineage; metrology?: { summary: string; reportURL: string }; experiment?: { title: string; summary: string; reportURL: string; timingLabel: string }; bendAnalysis?: BendAnalysis; inclinationAnalysis?: InclinationAnalysis; revision: Revision; assetURLs: Record<string, string>; objects: CatalogObject[]; geometry: { calibration?: Calibration; anchor: { nativeHeight: number | null; nativeWidth: number | null; assumedHeightM: number; assumedWidthM: number; mPerNative: number | null; referenceFit?: { status: string; mPerNative: number | null; candidateMPerNative?: number | null; reason?: string; diagnostics?: unknown }; assumptions?: string[] }; floor: { status: string } }; measurementEvaluation?: { comparisons: Comparison[]; groundTruthUsedForCalibration: boolean }; timing: { oneShotSeconds?: number }; nativeToMetersDefault: number | null };
type View = "photo" | "point_cloud" | "model" | "compare";
const scopeText: Record<string, string> = { model_bottom_face_center: "模型底面中心", visible_face_lower_terminal: "可见面下沿（整个外壳最低点未确认）", model_lower_rail_near_curtain: "光幕旁的围栏下横梁底面" };
const sideOrder = ["right", "left", null] as const;
const sideTitle = { right: "右侧（照片 4 视角）", left: "左侧（照片 4 视角）" } as Record<string, string>;
/** The endpoint is read only while its exact measured representation and asset are the displayed ones. */
export function endpointBound(row: Endpoint, entities: Entity[], assets: { id: string; sha256?: string | null }[]) {
  const entity = entities.find(item => item.id === row.objectId), rep = entity && activeModel(entity);
  return !!rep && rep.id === row.representationId && rep.assetId === row.assetId && assets.some(asset => asset.id === row.assetId && asset.sha256 === row.assetSha256);
}
export function PhotoReport({ data, base = typeof window === "undefined" ? "" : window.location.href, onRevision }: { data: PhotoReportData; base?: string; onRevision?: (choice: RevisionChoice, keep: { object: string | null; photo: string }) => void }) {
  useEffect(() => { const section = window.location.hash.slice(1); if (["overview", "scene", "semantics", "sources"].includes(section)) document.getElementById(section)?.scrollIntoView(); }, []);
  const entry = new URL(window.location.href).searchParams, resolve = (path: string) => new URL(path, base).href;
  const endpointEstimate = data.endpointEstimation?.status === "conditional_unvalidated" ? data.endpointEstimation : undefined;
  const endpointRows = endpointEstimate?.endpoints ?? [], differences = endpointEstimate?.differences ?? [];
  const firstEndpoint = endpointRows.find(row => row.side === "right") ?? endpointRows[0];
  const entryImage = `photo-${entry.get("photo")}`, entryObject = entry.get("object"), entryView = entry.get("view");
  const defaultEndpoints = !!endpointEstimate && !["photo", "object", "view", "measurement"].some(key => entry.has(key));
  const initialView = ["photo", "point_cloud", "model", "compare"].includes(entryView || "") ? entryView as View : defaultEndpoints ? "model" : undefined;
  const [imageId, setImageId] = useState(data.revision.document.cameras.some(camera => camera.imageId === entryImage) ? entryImage : `photo-${endpointEstimate?.sidePhoto ?? 4}`), [entityId, setEntityId] = useState<string | null>(entryObject && data.objects.some(item => item.id === entryObject) ? entryObject : firstEndpoint ? firstEndpoint.objectId : data.experiment ? "v-guard-left" : "emergency-button"), [observationId, setObservationId] = useState<string | null>(null);
  const [height, setHeight] = useState((data.geometry.calibration?.reference.features.wholeComponentHeightM ?? data.geometry.anchor.assumedHeightM) * 100), [exporting, setExporting] = useState(false), [exportError, setExportError] = useState("");
  const [showGroundDistance, setShowGroundDistance] = useState(false);
  const [showEndpointComparison, setShowEndpointComparison] = useState(entry.get("measurement") === "endpoints" || defaultEndpoints);
  const [viewRequest, setViewRequest] = useState<{ view: View; nonce: number } | null>(null);
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
  const modelRange = (native: number) => modelScale.rangeNativeToMeters ? modelScale.rangeNativeToMeters.map(value => (native * value * 100).toFixed(1)).sort((a, b) => +a - +b).join(" – ") + " cm" : "未知";
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
    const url = new URL(file, base); url.searchParams.set("revision", data.revision.documentSha256); return url.href;
  } }), [data, base]);
  const camera = revision.document.cameras.find(item => item.imageId === imageId);
  const selection: Selection = { projectId: revision.projectId, revisionId: revision.id, entityId, observationId, cameraId: camera?.id || null };
  const selected = revision.document.entities.find(entity => entity.id === entityId), record = data.objects.find(item => item.id === entityId);
  const bound = (row: Endpoint) => endpointBound(row, revision.document.entities, revision.document.assets ?? []);
  const endpointsOf = (id: string | null | undefined) => endpointRows.filter(row => row.objectId === id);
  const partnerOf = (row: Endpoint) => endpointRows.find(other => other.id === row.pairedEndpointId) ?? endpointRows.find(other => other.pairedEndpointId === row.id);
  const physicalFeature = (id: string | null) => { const ground = data.objects.find(item => item.id === id)?.groundDistance?.feature; return ground?.valueNative != null && Number.isFinite(ground.valueNative) && ground.valueNative >= 0 && ground.pointNative?.length === 3 && ground.footNative?.length === 3 && [...ground.pointNative, ...ground.footNative].every(Number.isFinite) ? ground : null; };
  const showMeasurementOf = (id: string) => {
    const measured = endpointsOf(id).some(bound);
    setShowEndpointComparison(measured); setShowGroundDistance(!measured && !!physicalFeature(id));
    if (measured || physicalFeature(id)) setViewRequest(request => ({ view: "model", nonce: (request?.nonce ?? 0) + 1 }));
  };
  const selectSemanticObject = (id: string, sourcePhoto?: number, sourceObservationId?: string) => {
    const target = revision.document.entities.find(item => item.id === id);
    if (!target || !data.objects.some(item => item.id === id)) return;
    const sourceImage = sourcePhoto == null ? null : `photo-${sourcePhoto}`;
    const nextImage = sourceImage && revision.document.cameras.some(item => item.imageId === sourceImage) ? sourceImage : imageId;
    setImageId(nextImage);
    const observations = revision.document.observations.filter(item => target.observationRefs?.includes(item.id) && item.imageId === nextImage);
    const sourceObservation = observations.find(item => item.id === sourceObservationId) ?? observations[0];
    // The selected object's own measurement stays available in this revision; nothing is copied from another one.
    setEntityId(id); setObservationId(sourceObservation?.id ?? null); showMeasurementOf(id);
    document.getElementById("scene")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const displayValue = (value: unknown) => nativeToMeters != null && typeof value === "number" && Number.isFinite(value) ? `${(value * nativeToMeters).toFixed(3)} m` : "未知";
  const centimeters = (value: number | null | undefined) => value != null && Number.isFinite(value) ? `${(value * 100).toFixed(1)} cm` : "未知";
  const visibleHeight = selected?.observedExtentAvailable !== true ? undefined : (record?.id === "robot" ? record.visibleHeightByPhoto?.[photo] : record?.visibleHeightNative);
  const inclination = data.inclinationAnalysis?.revisionId === revision.id ? data.inclinationAnalysis.items.find(row => row.entityId === entityId) : undefined;
  const bend = data.bendAnalysis?.revisionId === revision.id ? data.bendAnalysis.items.find(row => row.entityId === entityId) : undefined;
  const physicalHeight = record?.measurements.height?.valueNative, distance = record?.groundDistance;
  const ground = physicalFeature(entityId), groundRange = distance?.feature?.rangeNative;
  const sourceEdgeAvailable = !!ground;
  const groundValue = ground ? ground.valueNative : null;
  const groundDisplay = record?.physicalBottom && groundValue != null ? modelCentimeters(groundValue) : displayValue(groundValue);
  const selectedModel = selected && activeModel(selected);
  const groundAnnotation: SceneMeasurement | null = showGroundDistance && ground && selected && selectedModel && ground.valueNative != null ? {
    revisionId: revision.id, documentSha256: revision.documentSha256, coordinateFrameId: selectedModel.coordinateFrameId, kind: "ground_distance", source: "source_photo_support",
    value: ground.valueNative, unit: "native", displayLabel: record?.physicalBottom ? `${groundDisplay} · 条件模型估计` : nativeToMeters == null ? "源边缘到地面 · 尺度未知" : `${displayValue(ground.valueNative)} · 条件估计`, method: "physical-source-edge-to-ground",
    references: [{ entityId: selected.id, representationId: selectedModel.id, assetId: selectedModel.assetId || null, assetSha256: null, placementState: null, qualityStatus: null }],
    lines: [{ points: [ground.pointNative, ground.footNative], color: "#27d3d0" }],
    labelPoint: ground.pointNative.map((v, i) => (v + ground.footNative[i]) / 2), quality: {},
  } : null;
  const selectedEndpoint = endpointsOf(selected?.id).find(bound);
  const partner = selectedEndpoint && partnerOf(selectedEndpoint);
  const curtain = selectedEndpoint?.pairedEndpointId ? selectedEndpoint : partner, rail = curtain === selectedEndpoint ? partner : selectedEndpoint;
  const pairDifference = curtain && rail && differences.find(row => row.minuendId === curtain.id && row.subtrahendId === rail.id);
  const endpointAnnotation: SceneMeasurement | null = showEndpointComparison && selectedEndpoint && selected && selectedModel?.coordinateFrameId === "workcell-floor" ? {
    revisionId: revision.id, documentSha256: revision.documentSha256, coordinateFrameId: selectedModel.coordinateFrameId, kind: "ground_distance", source: "model_endpoint", method: "conditional-endpoint-comparison",
    value: pairDifference && curtain && rail && bound(curtain) && bound(rail) ? pairDifference.valueNative : selectedEndpoint.heightNative, unit: "native",
    displayLabel: pairDifference && curtain && rail && bound(curtain) && bound(rail) ? `${pairDifference.label}：${modelCentimeters(pairDifference.valueNative)} · 条件模型估计，未验证` : `${selectedEndpoint.label}离地 ${modelCentimeters(selectedEndpoint.heightNative)} · 条件模型估计，未验证`,
    references: [{ entityId: selected.id, representationId: selectedModel.id, assetId: selectedModel.assetId || null, assetSha256: selectedEndpoint.assetSha256, placementState: null, qualityStatus: null }],
    // ponytail: the saved report frame is floor Z-up; the orange segment is a vertical difference, never the distance between objects.
    lines: pairDifference && curtain && rail && bound(curtain) && bound(rail) ? [{ points: [curtain.pointNative, curtain.footNative], color: "#27d3d0" }, { points: [rail.pointNative, rail.footNative], color: "#b087ff" }, { points: [curtain.pointNative, [rail.pointNative[0], rail.pointNative[1], curtain.pointNative[2]]], color: "#edbe38" }, { points: [[rail.pointNative[0], rail.pointNative[1], curtain.pointNative[2]], rail.pointNative], color: "#ff8e45" }] : [{ points: [selectedEndpoint.pointNative, selectedEndpoint.footNative], color: "#27d3d0" }],
    labelPoint: pairDifference && curtain && rail ? [rail.pointNative[0], rail.pointNative[1], (curtain.pointNative[2] + rail.pointNative[2]) / 2] : selectedEndpoint.pointNative.map((v, i) => (v + selectedEndpoint.footNative[i]) / 2), quality: {},
  } : null;
  const endpointValue = (row: Endpoint) => bound(row) ? modelCentimeters(row.heightNative) : "未知（显示模型与测量模型不同）";
  const endpointTruth = data.measurementEvaluation?.comparisons.filter(row => endpointRows.some(point => point.objectId === row.objectId));
  const candidate = data.lineage?.role === "candidate", choices = data.revisionChoices ?? [];
  const endpointCard = endpointEstimate && endpointRows.length > 0 && <section className="photo-report-object-evidence" aria-label="光幕与围栏底边离地" data-endpoint-revision={revision.id}>
    <h3>光幕与围栏：底边离地</h3><p>{data.revision.label ?? "当前版本"} · 条件模型估计，未验证 · 当前显示模型的测点到同一估计地面</p>
    {sideOrder.map(side => { const rows = endpointRows.filter(row => row.side === side); if (!rows.length) return null; const pairs = differences.filter(row => rows.some(point => point.id === row.minuendId) && rows.some(point => point.id === row.subtrahendId));
      return <div key={side ?? "unsided"} className="photo-report-endpoint-side"><h4>{side ? sideTitle[side] : "未定左右"}</h4><dl>{rows.map(point => <div key={point.id}><dt>{point.label}离地</dt><dd data-endpoint-estimate={point.id} data-endpoint-object={point.objectId}>{endpointValue(point)}</dd><small>{scopeText[point.measurementScope] ?? point.measurementScope}</small></div>)}{pairs.map(row => <div key={row.id}><dt>{row.label}</dt><dd data-endpoint-difference={row.id}>{modelCentimeters(row.valueNative)}</dd></div>)}</dl></div>; })}
    {differences.filter(row => row.id.endsWith("-left-minus-right")).map(row => <p key={row.id} className="photo-report-endpoint-lr"><strong>{row.label}</strong> <span data-endpoint-difference={row.id}>{modelCentimeters(row.valueNative)}</span><small>{row.description}</small></p>)}
    <button type="button" aria-pressed={showEndpointComparison} onClick={() => { const target = selectedEndpoint ?? firstEndpoint; setShowEndpointComparison(value => !value); if (target && !selectedEndpoint) { setEntityId(target.objectId); setObservationId(null); } setViewRequest(request => ({ view: "model", nonce: (request?.nonce ?? 0) + 1 })); }}>{showEndpointComparison ? "隐藏底边离地测量线" : "显示底边离地测量线"}</button>
  </section>;
  const spatialFacts = (id: string): SpatialFact[] => {
    const facts: SpatialFact[] = [];
    for (const row of endpointsOf(id)) facts.push({ label: `${row.label}离地`, value: endpointValue(row), testId: row.id,
      status: `条件模型估计，未验证 · ${scopeText[row.measurementScope] ?? row.measurementScope}`, source: `${data.revision.label ?? revision.id} · ${row.modelFile} ${row.modelSha256.slice(0, 8)}` });
    for (const row of differences.filter(row => [row.minuendId, row.subtrahendId].some(point => endpointRows.find(item => item.id === point)?.objectId === id)))
      facts.push({ label: row.label, value: modelCentimeters(row.valueNative), testId: row.id, status: row.description });
    const feature = physicalFeature(id), item = data.objects.find(row => row.id === id);
    if (feature?.valueNative != null) facts.push({ label: "多视角源边缘到地面", value: item?.physicalBottom ? modelCentimeters(feature.valueNative) : displayValue(feature.valueNative), status: "条件估计", source: `照片 ${(feature.sourcePhotos ?? []).join(" / ")}` });
    const angle = data.bendAnalysis?.revisionId === revision.id ? data.bendAnalysis.items.find(row => row.entityId === id) : undefined;
    if (angle?.status === "measured" && angle.result) facts.push({ label: "两板面折弯内角", value: `${angle.result.value.toFixed(1)}°`, status: "模型估计，实物角度未唯一确定" });
    if (facts.length) facts.push({ label: "模型比例", value: modelScale.nativeToMeters == null ? "未知（原生单位）" : `1 native = ${modelScale.nativeToMeters.toFixed(5)} m`, status: modelScale.status === "conditional_unvalidated" ? "条件估计，未验证" : modelScale.status, source: modelScale.source });
    return facts;
  };
  async function downloadModel() {
    setExporting(true); setExportError("");
    try {
      const assetBase = new URL("viewer-assets/", window.location.href).href;
      const THREE = await import(/* @vite-ignore */ `${assetBase}three.module.js`);
      const { GLTFLoader } = await import(/* @vite-ignore */ `${assetBase}addons/loaders/GLTFLoader.js`);
      const { GLTFExporter } = await import(/* @vite-ignore */ `${assetBase}addons/exporters/GLTFExporter.js`);
      const scene = new THREE.Scene(), group = new THREE.Group(), loader = new GLTFLoader();
      scene.add(group); group.name = modelScale.nativeToMeters == null ? "Workcell — native units, Y up" : "Workcell — metres, Y up";
      if (modelScale.nativeToMeters != null) group.scale.setScalar(modelScale.nativeToMeters);
      group.rotation.x = -Math.PI / 2;
      for (const entity of revision.document.entities.filter(entity => entity.visible !== false && !entity.sourceContext)) {
        const rep = activeModel(entity); if (!rep?.assetId || rep.sourceValidity === "stale") continue;
        const model = (await loader.loadAsync(await resources.resolveAsset(rep.assetId))).scene;
        const placed = new THREE.Group(); placed.name = entity.label || entity.id;
        placed.matrix.fromArray(transformMatrix(entity.currentModelTransform || rep.transform)); placed.matrixAutoUpdate = false;
        placed.userData = { entityId: entity.id, sourceRepresentation: rep.id, assetSha256: revision.document.assets?.find(asset => asset.id === rep.assetId)?.sha256 ?? null, physicalDimensionsUnknown: entity.physicalDimensionsUnknown === true };
        placed.add(model); group.add(placed);
      }
      scene.userData = { units: modelScale.nativeToMeters == null ? "native" : "metres", upAxis: "Y", ground: { normal: [0, 1, 0], offset: 0 }, sourcePhoto: Number(photo), revisionId: revision.id, documentSha256: revision.documentSha256, revisionLabel: data.revision.label, revisionRole: data.lineage?.role ?? "main", nativeToMeters: modelScale.nativeToMeters, scaleStatus: modelScale.nativeToMeters == null ? "uncalibrated" : modelScale.status, modelMeasurementScale: modelScale, groundTruth: false, uniformReferenceRatio: referenceRatio, suppliedReference: calibration?.reference, referenceFitStatus: anchor.referenceFit?.status, assumptions: anchor.assumptions };
      scene.updateMatrixWorld(true);
      const result = await new GLTFExporter().parseAsync(scene, { binary: true });
      const url = URL.createObjectURL(new Blob([result], { type: "model/gltf-binary" }));
      const link = document.createElement("a"); link.href = url; link.download = `workcell-${revision.id}-photo-${photo}-${modelScale.nativeToMeters == null ? "native-unscaled" : `${height}cm-${modelScale.status}`}.glb`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { setExportError(error instanceof Error ? error.message : String(error)); }
    finally { setExporting(false); }
  }
  const focus = [...new Map(endpointRows.map(row => [row.objectId, row.label])).entries(), ...(data.objects.some(item => item.id === "floor") ? [["floor", "地面"]] : [])];
  return <SceneResources.Provider value={resources}><main className="photo-report" data-revision-id={revision.id} data-document-sha256={revision.documentSha256}>
    <header className="photo-report-header"><a className="photo-report-brand" href="#overview">PANOPTES <span>WORKCELL REPORT</span></a><nav><a href="#overview">概览</a><a href="#scene">对象与场景</a>{data.semanticExperiment && <a href="#semantics">语义实验</a>}<a href="#sources">来源与假设</a></nav></header>
    <section className="photo-report-overview" id="overview"><div><p className="photo-report-eyebrow">四张照片 · 对象级空间重建</p><h1>{reference ? "工作单元测量报告" : "工作单元空间报告"}</h1><p>选取对象查看可见高度、离地间距与证据。拖动分界线，在同一相机下核对照片和模型。</p></div><dl><div><dt>照片</dt><dd>{revision.document.cameras.length}</dd></div><div><dt>对象</dt><dd>{data.objects.length}</dd></div><div><dt>{reference ? "基准整体高度" : data.experiment?.timingLabel || "本次计算"}</dt><dd>{reference ? (reference.wholeComponentHeightM * 100).toFixed(1) : data.timing.oneShotSeconds?.toFixed(1) ?? "—"}<small>{reference ? "cm" : "秒"}</small></dd></div>{reference && <div><dt>{data.measurementUpdate?.kind === "saved-geometry-replay" ? "原始完整流程" : "本次端到端"}</dt><dd>{data.timing.oneShotSeconds?.toFixed(1) ?? "—"}<small>秒</small></dd></div>}</dl></section>
    {choices.length > 0 && <section className={`photo-report-revision${candidate ? " photo-report-warning" : ""}`} aria-label="模型版本">
      <label>模型版本<select data-revision-select value={revision.id} onChange={event => { const choice = choices.find(item => item.id === event.target.value); if (choice && onRevision) onRevision(choice, { object: entityId, photo }); }}>{choices.map(choice => <option key={choice.id} value={choice.id}>{choice.label} · {choice.status === "main" ? "主模型" : "候选，未通过严格门槛"}</option>)}</select></label>
      <p>当前：<strong>{data.revision.label ?? revision.id}</strong> · revision <code>{revision.id}</code> · 文档 <code>{revision.documentSha256.slice(0, 12)}</code>。{candidate ? "候选模型只用于核对，未通过跨图严格门槛，不替换主模型。" : "主模型。"}模型、离地测点、卡尺、语义空间证据和下载都只读取这个版本。{data.lineage?.candidateEvidence?.url && <> <a href={new URL(data.lineage.candidateEvidence.url, window.location.href).href}>候选原图证据</a></>}</p>
    </section>}
    {endpointCard && <section className="photo-report-evaluation photo-report-endpoints" aria-label="底边离地条件模型估计（未验证）">
      {endpointCard}<p><a className="photo-report-endpoint-link" href={`?${choices.length ? `version=${revision.id}&` : ""}photo=${endpointEstimate!.sidePhoto}&object=${firstEndpoint!.objectId}&view=model&measurement=endpoints#scene`}>查看照片 {endpointEstimate!.sidePhoto} · 光幕与围栏底边离地模型</a></p>
      {endpointTruth?.length ? <p>现场提供的对照值：{endpointTruth.map(row => `${row.label} ${(row.groundTruthM * 100).toFixed(2)} cm`).join("；")}。数值在开发中已知，未输入本组估计；不是盲测精度验证，实物端点对应仍待确认。</p> : null}
      <details><summary>查看条件估计的来源与范围</summary><p>当前 GLB 模型的指定测点到同一估计地面；数值使用本次按钮条件比例，不代表源图物理底端已验证。修改标尺会同步更新这里、卡尺、语义空间证据和下载模型；原生几何不变。</p>
      <p>{endpointRows.map(point => `${point.label}比例敏感范围 ${bound(point) ? modelRange(point.heightNative) : "未知"}`).join("；")}。范围只反映标尺表面深度第 5/95 百分位，不是精度保证。</p>
      <p>比例：1 native = {modelScale.nativeToMeters?.toFixed(5) ?? "未知"} m。{modelScale.source}</p><p>方法：{endpointEstimate!.method}</p>
      <p>测点模型：{endpointRows.map(point => `${point.label} ← ${point.modelFile} ${point.modelSha256.slice(0, 8)}`).join("；")}。</p>
      <p>青色、紫色竖线：光幕测点与旁边围栏测点到地面。橙色竖线：沿地面法向的高低差；黄色线仅连接同一高度。</p>
      <p><a href={resolve("model-endpoint-estimate.json")} download>下载本版本端点测量 JSON</a></p></details>
    </section>}
    {data.metrology && <section className="photo-report-sources" aria-label="按钮标尺测量实验"><h2>按钮标尺 · 小于 3 cm 的测量实验</h2><p>{data.metrology.summary}</p><a href={data.metrology.reportURL}>查看本次结果、原图证据与失败原因</a></section>}
    {data.experiment && <section className="photo-report-sources" aria-label="已有模型实验结论">{reference ? <details><summary>{data.experiment.title}</summary><p>{data.experiment.summary}</p><a href={data.experiment.reportURL}>查看全部实验、照片证据与测量限制</a></details> : <><h2>{data.experiment.title}</h2><p>{data.experiment.summary}</p><a href={data.experiment.reportURL}>查看全部实验、照片证据与测量限制</a></>}</section>}
    <section className="photo-report-scale" aria-label="标尺尺寸">
      <div><h2>统一模型标尺 · 急停按钮</h2><p>卡尺、底边比较、语义空间证据、地面网格和下载模型共用以下比例。三尺寸联合标定状态与模型条件估计分别记录。</p></div>
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
      <p><a href={resolve("measurement-evaluation.json")} download>下载全部物体的基准估计 JSON（{centimeters(reference?.wholeComponentHeightM)} 标尺）</a> · <a href={resolve("measurements.json")} download>下载现场提供的尺寸</a></p>
    </section>}
    {data.semanticExperiment && <PhotoSemanticExperiment data={data.semanticExperiment} selectedEntityId={entityId} onSelect={selectSemanticObject} />}
    {!data.semanticExperiment && data.semanticBinding && <section id="semantics" className="photo-semantic-experiment photo-report-warning" data-semantic-binding-status={data.semanticBinding.status}><h2>语义实验未绑定到本版本</h2><p>已有语义结果的输入与本版本不一致，不复用旧结果：{data.semanticBinding.reason}</p><p>{data.semanticBinding.action}</p></section>}
    <div id="scene"><nav className="photo-report-focus" aria-label="重点检查对象"><strong>重点检查</strong>{focus.map(([id, label]) => <button key={id} type="button" aria-pressed={entityId === id} onClick={() => { setEntityId(id); setImageId(`photo-${endpointEstimate?.sidePhoto ?? 4}`); setObservationId(null); showMeasurementOf(id); }}>{label}</button>)}<small>先选对象，再切换照片、原始点云和模型核对。</small></nav><ReportScene matchedComparison measurementScale={modelScale} initialView={initialView} viewRequest={viewRequest} measurementOverride={endpointAnnotation || groundAnnotation} revision={revision} selection={selection} onSelect={(id, obs) => { setEntityId(id); setObservationId(obs || null); setShowGroundDistance(false); }} imageId={imageId} cameraId={camera?.id || null} onCamera={(id) => { setImageId(id); setObservationId(null); }} onClearSelection={() => { setEntityId(null); setShowGroundDistance(false); }} inspector={(surface) => selected ? <>
      {data.semanticExperiment && <PhotoSemanticObject data={data.semanticExperiment} entityId={selected.id} onSelect={selectSemanticObject} facts={spatialFacts(selected.id)} resolve={resolve} onMeasure={endpointsOf(selected.id).some(bound) || physicalFeature(selected.id) ? () => showMeasurementOf(selected.id) : undefined} measureLabel="在 3D 模型中显示该对象的离地测量线" />}
      {selectedEndpoint && endpointCard}<section className="photo-report-object-evidence" data-selected-object={selected.id}><h3>{selected.label}</h3>{record?.modelTerminal && <p className="photo-report-warning" role="status">本版本的这个对象使用候选模型（{record.modelTerminal.candidateStatus}），未通过跨图严格门槛，不作为物理验收。</p>}{bend && <section className="photo-report-angles photo-report-bend" data-bend-entity={selected.id}><h4>本块护板 · 两板面折弯内角</h4>{bend.status === "measured" && bend.result ? <><output>{bend.result.value.toFixed(1)}°</output><p>模型估计 · 摊平为 180°，直角折弯为 90°。</p><p>橙色 / 蓝色：本块板的两个拟合板面；紫色：交线；绿色：折弯内角。</p></> : <p>折弯角度不可用：{bend.reason || bend.status}</p>}</section>}<dl><div><dt>{record?.id === "emergency-button" && reference ? "输入整体高度（标准尺寸）" : physicalHeight != null ? "整体高度（按标尺换算）" : "可见高度估计（按标尺换算）"}</dt><dd data-height-native={physicalHeight ?? visibleHeight ?? "unknown"}>{record?.id === "emergency-button" && reference ? centimeters(valid ? height / 100 : null) : displayValue(physicalHeight ?? visibleHeight)}</dd></div><div><dt>{record?.physicalBottom ? "模型下沿离地（条件估计）" : "物理下缘离地（多视角源边缘，条件估计）"}</dt><dd data-ground-distance-native={groundValue ?? "unknown"}>{groundDisplay}</dd></div>{endpointsOf(selected.id).map(row => <div key={row.id}><dt>{row.label}离地（本版本模型测点）</dt><dd data-selected-endpoint={row.id}>{endpointValue(row)}</dd></div>)}</dl>{selected.observedExtentAvailable === true && record?.visibleHeightRangeNative && <p>跨照片可见高度范围：{record.visibleHeightRangeNative.map(displayValue).join(" – ")}</p>}<p>来源照片：{[...new Set(record?.observations.map(item => item.photo))].join(" / ") || "无"}</p><p>{record?.representation}</p>{sourceEdgeAvailable && <button type="button" aria-pressed={showGroundDistance} onClick={() => { setShowEndpointComparison(false); setShowGroundDistance(value => !value); }}>{showGroundDistance ? "隐藏离地测量线" : "显示离地测量线"}</button>}{groundRange && <p data-ground-distance-range>图像几何敏感范围：{groundRange.map(displayValue).join(" – ")}；不包含全部相机和地面系统误差。</p>}<p>间距依据：{ground?.source || distance?.source || "缺少来源数据。"}</p>{!sourceEdgeAvailable && <p>{distance?.reason || "缺少稳定的多视角物理下缘支持，离地距离保持未知。"}</p>}{physicalHeight == null && <p>可见高度不代表完整物体的物理尺寸。单视图或遮挡部分保留未知。</p>}</section>
      {inclination && <section className="photo-report-angles" data-inclination-entity={selected.id}><h3>板面角度 · 模型估计</h3>
        {surface && <div key={surface.surfaceId} data-inclination-surface={surface.surfaceId}><h4>局部面 {surface.surfaceId}</h4><p>与地面夹角 <strong>{surface.inclinationDeg.toFixed(1)}°</strong>（90° 为垂直）</p><p>偏离垂直 <strong>{surface.deviationFromVerticalDeg.toFixed(1)}°</strong></p><p>{({ non_vertical: "非竖直（模型估计）", vertical: "竖直范围内（模型估计）", direction_unverified: "方向未确认" } as Record<string, string>)[surface.classification] || surface.classification} · 角度离散 {surface.angularSpreadDeg.toFixed(1)}°</p><p>{surface.result.quality.angularErrorDeg == null ? "地面方向误差未记录；相对竖直方向的分类待确认。" : `工程角度误差估计 ${surface.result.quality.angularErrorDeg.toFixed(1)}°，不代表现场标定精度。`}</p></div>}
        {!inclination.surfaces.length && <p>角度不可用：{inclination.reason || inclination.status}</p>}
        {inclination.surfaces.length > 0 && <p>已保存 {inclination.surfaces.length} 个局部拟合面，{surface ? "当前显示所选的 1 个面" : "当前尚未选择局部面"}。从上方“倾斜平面”切换；勾选“全部已测平面”可查看完整列表，所选面的参考线同步显示在 3D 中。</p>}
      </section>}
      {nativeToMeters == null && <p>以下模型位置使用原生坐标单位，尚未换算为米。</p>}
      <ObjectFacts entity={selected} document={revision.document} />
      <details className="report-source-details" open><summary>建模来源与假设</summary>{record?.notes.map((note, index) => <p key={index}>{note}</p>)}</details>
    </> : <p>从左侧列表、照片或模型中选择对象。</p>} /></div>
    <section id="sources" className="photo-report-sources"><h2>来源与假设</h2><p>照片和相机保持同一重建坐标系；切换照片时，机器人采用该照片对应姿态。物理米制测量需要受支持的三维按钮标尺、物理边缘与地面。当前标尺拟合：{anchor.referenceFit?.status === "available" ? "条件支持" : "未通过验证"}；已知标尺不能消除相机、物体边界和地面误差。</p><p>{data.geometry.floor.status}</p>{anchor.assumptions?.map((note, i) => <p key={i}>{note}</p>)}{data.measurementUpdate?.kind === "saved-geometry-replay" && <p>本版本由冻结的照片推理结果重放构建（未重复模型推理）：{data.measurementUpdate.sourceRevisionId}；构建 {data.measurementUpdate.revisionBuildSeconds?.toFixed(1) ?? "—"} 秒。上方完整流程耗时为原始推理运行的记录。</p>}<a href={resolve("scene-report.json")} download>下载本版本结构化报告 JSON</a></section>
  </main></SceneResources.Provider>;
}
function validReport(value: any): value is PhotoReportData {
  return !!value?.revision?.document && !!value.assetURLs && Array.isArray(value.objects) && !!value.geometry?.anchor && (value.geometry.anchor.mPerNative == null || (Number.isFinite(value.geometry.anchor.mPerNative) && value.geometry.anchor.mPerNative > 0));
}
function LoadPhotoReport() {
  const [loaded, setLoaded] = useState<{ data: PhotoReportData; base: string }>(), [error, setError] = useState("");
  const load = async (url: string) => {
    const response = await fetch(url, { cache: "no-cache" });
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    const value = await response.json();
    if (!validReport(value)) throw Error("Invalid report contract");
    return { data: value, base: new URL(url, window.location.href).href };
  };
  useEffect(() => {
    const requested = new URL(window.location.href).searchParams.get("version");
    load(new URL("scene-report.json", window.location.href).href).then(async main => {
      const choice = requested && requested !== main.data.revision.id ? main.data.revisionChoices?.find(item => item.id === requested) : undefined;
      setLoaded(choice ? await load(new URL(choice.url, main.base).href) : main);
    }).catch(error => setError(error.message));
  }, []);
  const switchRevision = (choice: RevisionChoice, keep: { object: string | null; photo: string }) => {
    if (!loaded) return;
    const url = new URL(window.location.href);
    url.searchParams.set("version", choice.id); url.searchParams.set("photo", keep.photo); url.searchParams.set("view", "model");
    if (keep.object) url.searchParams.set("object", keep.object); else url.searchParams.delete("object");
    if (keep.object && loaded.data.endpointEstimation?.endpoints.some(row => row.objectId === keep.object)) url.searchParams.set("measurement", "endpoints"); else url.searchParams.delete("measurement");
    window.history.replaceState(null, "", url);
    load(new URL(choice.url, loaded.base).href).then(setLoaded).catch(error => setError(error.message));
  };
  return error ? <main className="photo-report"><h1>报告加载失败</h1><p role="alert">{error}</p></main> : loaded ? <PhotoReport key={loaded.data.revision.id} data={loaded.data} base={loaded.base} onRevision={switchRevision} /> : <main className="photo-report" role="status">正在加载空间报告…</main>;
}
if (typeof document !== "undefined" && document.getElementById("root")) createRoot(document.getElementById("root")!).render(<I18nProvider><LoadPhotoReport /></I18nProvider>);
