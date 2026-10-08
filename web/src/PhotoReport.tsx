import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { ReportScene } from "./ReportScene";
import { ObjectFacts } from "./WorkcellReport";
import { SceneResources } from "./SceneResources";
import { PhotoSemanticExperiment, PhotoSemanticObject, type SemanticExperiment, type SpatialFact } from "./PhotoSemanticExperiment";
import { answerSpatialQuery } from "./spatial-query";
import type { BendAnalysis, InclinationAnalysis, SceneMeasurement } from "./SpatialMeasurements";
import { I18nProvider, LanguageSwitch, useI18n } from "./i18n";
import { activeModel } from "./core";
import { transformMatrix } from "./viewer/native-math";
import type { Revision, Selection, Representation, Entity } from "./types";
import "./styles.css";
import "./workcell-report.css";
import "./photo-report.css";

type GroundSample = { valueNative: number | null; pointNative: number[]; footNative: number[]; reason?: string; source?: string; sourcePhotos?: number[]; rangeNative?: number[] };
/** One multi-view model of an object: the photos that generated it, the photos its Sim(3) refine used, and its mask IoU in every photo. */
type MultiViewModel = { generationViews: number[]; refineViews: number[]; maskIoUByPhoto: Record<string, number> };
type ModelBox = { lengthNative: number; widthNative: number; heightNative: number; bottomNative: number; topNative: number; cornersNative: number[][] };
type CatalogObject = { id: string; label: string; kind?: string; representation: string; modelBoxFloor?: ModelBox; notes: string[]; observations: { photo: number }[]; measurements: Record<string, any>; multiViewModel?: MultiViewModel; physicalBottom?: { geometryScope: string }; modelTerminal?: { candidateStatus: string; acceptedForPhysicalUse: boolean }; visibleHeightNative?: number; visibleHeightRangeNative?: number[]; visibleHeightByPhoto?: Record<string, number>; groundDistance?: { byPhoto: Record<string, GroundSample>; feature?: GroundSample | null; rangeNative?: number[]; source: string; reason?: string } };
type Calibration = { primaryAxis: string; nativeToMeters: number | null; reference: { scope: string; scopeStatus: string; features: { wholeComponentHeightM: number; mainBodyDiameterM: number; redActuatorDiameterM: number } }; observedEnvelope?: { widthM: number | null }; renderingAssumptions?: string[] };
type Comparison = { objectId: string; label: string; method: string; estimateNative: number | null; rangeNative: number[] | null; byPhoto: Record<string, { valueNative: number | null }>; sourcePhotos: number[]; byPhotoMethod?: string; groundTruthM: number; source: string; limitation: string };
/** One model terminal measured on this revision's displayed representation, in native units. */
export type Endpoint = { id: string; objectId: string; label: string; side: "left" | "right" | null; measurementScope: string; railPart?: string; pointNative: number[]; footNative: number[]; heightNative: number; representationId: string; assetId: string; assetSha256: string; modelFile: string; modelSha256: string; pairedEndpointId?: string | null; pairingStatus?: string; provenance: string };
type EndpointDifference = { id: string; label: string; minuendId: string; subtrahendId: string; valueNative: number; description: string };
type EndpointExclusion = { id: string; minuendId: string; subtrahendId: string; reason: string };
type EndpointEstimation = { status: "conditional_unvalidated"; sidePhoto: number; method: string; endpoints: Endpoint[]; differences: EndpointDifference[]; excludedComparisons?: EndpointExclusion[] };
export type RevisionChoice = { id: string; label: string; branchId: string; status: string; documentSha256: string; parentRevisionId: string | null; url: string };
type Lineage = { role?: string; label?: string; parentRevisionId?: string | null; candidateEvidence?: { url?: string | null; acceptedForPhysicalUse?: boolean } };
/** The run's photo set: every photo shows one scene; the reference photo fixes left/right, posts and the button scale. */
type Capture = { photoCount: number; referencePhoto: number; sources?: { photo: number; name: string }[] | null };
export type PhotoReportData = { capture?: Capture; policyEvidence?: PolicyEvidence; semanticExperiment?: SemanticExperiment; semanticBinding?: { status: string; reason: string; action?: string }; modelMeasurementScale: { nativeToMeters: number | null; rangeNativeToMeters?: number[] | null; status: string; source: string }; measurementUpdate?: { kind: string; revisionBuildSeconds?: number; sourceRevisionId?: string | null }; endpointEstimation?: EndpointEstimation; revisionChoices?: RevisionChoice[]; lineage?: Lineage; metrology?: { summary: string; reportURL: string }; experiment?: { title: string; summary: string; reportURL: string; timingLabel: string }; bendAnalysis?: BendAnalysis; inclinationAnalysis?: InclinationAnalysis; revision: Revision; assetURLs: Record<string, string>; objects: CatalogObject[]; geometry: { calibration?: Calibration; anchor: { nativeHeight: number | null; nativeWidth: number | null; assumedHeightM: number; assumedWidthM: number; mPerNative: number | null; referenceFit?: { status: string; mPerNative: number | null; candidateMPerNative?: number | null; reason?: string; diagnostics?: unknown }; assumptions?: string[] }; floor: { status: string } }; measurementEvaluation?: { comparisons: Comparison[]; groundTruthUsedForCalibration: boolean; absentTargets?: { objectId: string; feature: string; reason: string }[] }; timing: { oneShotSeconds?: number }; nativeToMetersDefault: number | null };
type View = "photo" | "point_cloud" | "model" | "compare";
type PolicyEvidence = { engine: string; revisionId: string; documentSha256: string; ruleSet: { file: string; sha256: string; status: string }; conclusion: string;
  items: { policyId: string; sourceText: string; predicate: string; threshold: number; unit: string; compileStatus: string; refusal?: string; applicability: string; machineResult: null; spec: { file: string; sha256: string };
    subjects: { entityId: string; label: string; identitySource: string }[]; missingEvidence: string[]; evidenceStillNeeded?: { id: string; status: string; detail: string }[] }[] };
const evidenceName: Record<string, string> = { reviewer_applicability_confirmation: "PhotoReporttsx.text028", operator_anchored_metric_scale: "PhotoReporttsx.text029", subject_full_height: "PhotoReporttsx.text030", reference_region: "PhotoReporttsx.text031" };
/** The endpoint is read only while its exact measured representation and asset are the displayed ones. */
export function endpointBound(row: Endpoint, entities: Entity[], assets: { id: string; sha256?: string | null }[]) {
  const entity = entities.find(item => item.id === row.objectId), rep = entity && activeModel(entity);
  return !!rep && rep.id === row.representationId && rep.assetId === row.assetId && assets.some(asset => asset.id === row.assetId && asset.sha256 === row.assetSha256);
}
export function PhotoReport({ data, base = typeof window === "undefined" ? "" : window.location.href, onRevision }: { data: PhotoReportData; base?: string; onRevision?: (choice: RevisionChoice, keep: { object: string | null; photo: string }) => void }) {
  const { t, language } = useI18n();
  const scopeText: Record<string, string> = { model_bottom_face_center: "PhotoReporttsx.text032", visible_face_lower_terminal: "PhotoReporttsx.text033", model_lower_rail_near_curtain: "PhotoReporttsx.text034" };
  // A lower-envelope rail point is the 1% height of detected member ends on its plane, not a rail edge (rail-identity evidence).
  const scopeOf = (row: Endpoint) => row.railPart === "lower_envelope_hypothesis" ? t("PhotoReporttsx.text035") : t(scopeText[row.measurementScope] ?? row.measurementScope);
  const sideOrder = ["right", "left", null] as const;
  const sideTitle = (side: string, photo: number) => t("PhotoReporttsx.text036", {p0: side === "left" ? t("PhotoReporttsx.text002") : t("PhotoReporttsx.text003"), p1: photo});

  useEffect(() => { const section = window.location.hash.slice(1); if (["overview", "ask", "scene", "semantics", "policy", "sources"].includes(section)) document.getElementById(section)?.scrollIntoView(); }, []);
  const entry = new URL(window.location.href).searchParams, resolve = (path: string) => new URL(path, base).href;
  const endpointEstimate = data.endpointEstimation?.status === "conditional_unvalidated" ? data.endpointEstimation : undefined;
  const endpointRows = endpointEstimate?.endpoints ?? [], differences = endpointEstimate?.differences ?? [];
  const firstEndpoint = endpointRows.find(row => row.side === "right") ?? endpointRows[0];
  const entryImage = `photo-${entry.get("photo")}`, entryObject = entry.get("object"), entryView = entry.get("view");
  const defaultEndpoints = !!endpointEstimate && !["photo", "object", "view", "measurement"].some(key => entry.has(key));
  const initialView = ["photo", "point_cloud", "model", "compare"].includes(entryView || "") ? entryView as View : defaultEndpoints ? "model" : undefined;
  // Legacy runs without a capture record were four photos with photo 4 the reviewer's view: the last photo.
  const referencePhoto = endpointEstimate?.sidePhoto ?? data.capture?.referencePhoto ?? data.revision.document.cameras.length;
  const [imageId, setImageId] = useState(data.revision.document.cameras.some(camera => camera.imageId === entryImage) ? entryImage : `photo-${referencePhoto}`), [entityId, setEntityId] = useState<string | null>(entryObject && data.objects.some(item => item.id === entryObject) ? entryObject : firstEndpoint ? firstEndpoint.objectId : data.experiment ? "v-guard-left" : "emergency-button"), [observationId, setObservationId] = useState<string | null>(null);
  const [height, setHeight] = useState((data.geometry.calibration?.reference.features.wholeComponentHeightM ?? data.geometry.anchor.assumedHeightM) * 100), [exporting, setExporting] = useState(false), [exportError, setExportError] = useState("");
  const [showGroundDistance, setShowGroundDistance] = useState(false);
  const [showEndpointComparison, setShowEndpointComparison] = useState(entry.get("measurement") === "endpoints" || defaultEndpoints);
  const [viewRequest, setViewRequest] = useState<{ view: View; nonce: number } | null>(null);
  const [question, setQuestion] = useState(entry.get("q") ?? ""), [asked, setAsked] = useState(entry.get("q") ?? "");
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
    source: referenceRatio === 1 ? modelBase.source : t("PhotoReporttsx.text037", {p0: referenceRatio ?? t("unknown"), p1: modelBase.source}),
    rangeNativeToMeters: referenceRatio == null ? null : modelBase.rangeNativeToMeters?.map(value => value * referenceRatio) ?? null,
    uniformReferenceRatio: referenceRatio,
    currentReferenceDimensionsM: reference && referenceRatio != null ? Object.fromEntries(Object.entries(reference).map(([key, value]) => [key, value * referenceRatio])) : null,
    baseModelMeasurementScale: modelBase,
  };
  const modelCentimeters = (native: number) => modelScale.nativeToMeters == null ? `${native.toFixed(4)} native` : `${(native * modelScale.nativeToMeters * 100).toFixed(2)} cm`;
  const modelRange = (native: number) => modelScale.rangeNativeToMeters ? modelScale.rangeNativeToMeters.map(value => (native * value * 100).toFixed(1)).sort((a, b) => +a - +b).join(" – ") + " cm" : t("unknown");
  const photo = imageId.replace("photo-", "");
  // One scene: every entity keeps its single active model whatever photo or object is selected. Per-photo robot
  // poses stay in the data (modelVariants) but never replace the scene's model.
  const revision = useMemo(() => ({ ...data.revision, document: { ...data.revision.document,
    coordinateFrames: data.revision.document.coordinateFrames.map(frame => ({ ...frame, scale: { ...frame.scale, status: nativeToMeters == null ? "uncalibrated" as const : "model_estimated" as const, nativeToMeters } })),
  } }), [data, nativeToMeters]);
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
  const displayValue = (value: unknown) => nativeToMeters != null && typeof value === "number" && Number.isFinite(value) ? `${(value * nativeToMeters).toFixed(3)} m` : t("unknown");
  const centimeters = (value: number | null | undefined) => value != null && Number.isFinite(value) ? `${(value * 100).toFixed(1)} cm` : t("unknown");
  // A legacy per-photo robot pose has a per-photo height; one multi-view model has one visible height.
  const visibleHeight = selected?.observedExtentAvailable !== true ? undefined : (Object.keys(selected.modelVariants ?? {}).length ? record?.visibleHeightByPhoto?.[photo] : record?.visibleHeightNative);
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
    value: ground.valueNative, unit: "native", displayLabel: record?.physicalBottom ? t("PhotoReporttsx.text038", {p0: groundDisplay}) : nativeToMeters == null ? t("PhotoReporttsx.text039") : t("PhotoReporttsx.text040", {p0: displayValue(ground.valueNative)}), method: "physical-source-edge-to-ground",
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
    displayLabel: pairDifference && curtain && rail && bound(curtain) && bound(rail) ? t("PhotoReporttsx.text041", {p0: pairDifference.label, p1: modelCentimeters(pairDifference.valueNative)}) : t("PhotoReporttsx.text042", {p0: selectedEndpoint.label, p1: modelCentimeters(selectedEndpoint.heightNative)}),
    references: [{ entityId: selected.id, representationId: selectedModel.id, assetId: selectedModel.assetId || null, assetSha256: selectedEndpoint.assetSha256, placementState: null, qualityStatus: null }],
    // ponytail: the saved report frame is floor Z-up; the orange segment is a vertical difference, never the distance between objects.
    lines: pairDifference && curtain && rail && bound(curtain) && bound(rail) ? [{ points: [curtain.pointNative, curtain.footNative], color: "#27d3d0" }, { points: [rail.pointNative, rail.footNative], color: "#b087ff" }, { points: [curtain.pointNative, [rail.pointNative[0], rail.pointNative[1], curtain.pointNative[2]]], color: "#edbe38" }, { points: [[rail.pointNative[0], rail.pointNative[1], curtain.pointNative[2]], rail.pointNative], color: "#ff8e45" }] : [{ points: [selectedEndpoint.pointNative, selectedEndpoint.footNative], color: "#27d3d0" }],
    labelPoint: pairDifference && curtain && rail ? [rail.pointNative[0], rail.pointNative[1], (curtain.pointNative[2] + rail.pointNative[2]) / 2] : selectedEndpoint.pointNative.map((v, i) => (v + selectedEndpoint.footNative[i]) / 2), quality: {},
  } : null;
  // Blender-style readout of the selected model: oriented box edges, length / width / height and its clearance to the floor.
  const box = record?.modelBoxFloor;
  const boxAnnotation: SceneMeasurement | null = box && selected && selectedModel?.coordinateFrameId === "workcell-floor" ? (() => {
    const c = box.cornersNative, mid = (a: number[], b: number[]) => a.map((v, i) => (v + b[i]) / 2);
    const first = Math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]), second = Math.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1]);
    const [long, short] = first >= second ? [[0, 1], [1, 2]] : [[1, 2], [0, 1]];
    const bottom = [(c[0][0] + c[2][0]) / 2, (c[0][1] + c[2][1]) / 2, box.bottomNative], foot = [bottom[0], bottom[1], 0];
    const outward = (p: number[]) => [p[0] + .6 * (p[0] - bottom[0]), p[1] + .6 * (p[1] - bottom[1]), p[2]];
    const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    return { revisionId: revision.id, documentSha256: revision.documentSha256, coordinateFrameId: "workcell-floor", kind: "model_box", source: "model_box_floor",
      method: "oriented footprint rectangle x vertical range", value: box.heightNative, unit: "native",
      displayLabel: modelScale.nativeToMeters == null ? t("PhotoReporttsx.text043") : t("PhotoReporttsx.text044"),
      references: [{ entityId: selected.id, representationId: selectedModel.id, assetId: selectedModel.assetId || null, assetSha256: null, placementState: null, qualityStatus: null }],
      lines: [...edges.map(([a, b]) => ({ points: [c[a], c[b]], color: "#ffb000" })), ...(box.bottomNative > 1e-6 ? [{ points: [bottom, foot], color: "#27d3d0" }] : [])],
      labelPoint: c[6],
      // Labels sit outside the box (pushed away from its footprint centre) so they do not pile up on one another.
      labels: [{ point: outward(mid(c[long[0]], c[long[1]])), text: t("PhotoReporttsx.text045", {p0: modelCentimeters(box.lengthNative)}) }, { point: outward(mid(c[short[0]], c[short[1]])), text: t("PhotoReporttsx.text046", {p0: modelCentimeters(box.widthNative)}) },
        { point: outward(c[3].map((v, i) => v + .75 * (c[7][i] - v))), text: t("PhotoReporttsx.text047", {p0: modelCentimeters(box.heightNative)}) }, ...(box.bottomNative > 1e-6 ? [{ point: mid(bottom, foot), text: t("PhotoReporttsx.text048", {p0: modelCentimeters(box.bottomNative)}) }] : [])],
      quality: {} };
  })() : null;
  const primaryAnnotation = endpointAnnotation || groundAnnotation;
  const sceneAnnotation: SceneMeasurement | null = primaryAnnotation && boxAnnotation ? { ...primaryAnnotation,
    lines: [...primaryAnnotation.lines, ...boxAnnotation.lines.filter(line => line.color !== "#27d3d0")],
    labels: [...(primaryAnnotation.labels ?? []), { point: boxAnnotation.labelPoint, text: boxAnnotation.displayLabel ?? "" }, ...(boxAnnotation.labels ?? []).filter(label => !label.text.startsWith(t("ReportScenetsx.213")))] } : primaryAnnotation || boxAnnotation;
  const endpointValue = (row: Endpoint) => bound(row) ? modelCentimeters(row.heightNative) : t("PhotoReporttsx.text049");
  // A difference is read only while both of its endpoints are on the displayed representations.
  const differenceBound = (row: EndpointDifference) => [row.minuendId, row.subtrahendId].every(id => { const point = endpointRows.find(item => item.id === id); return !!point && bound(point); });
  const differenceValue = (row: EndpointDifference) => differenceBound(row) ? modelCentimeters(row.valueNative) : t("PhotoReporttsx.text049");
  const endpointTruth = data.measurementEvaluation?.comparisons.filter(row => endpointRows.some(point => point.objectId === row.objectId));
  const candidate = data.lineage?.role === "candidate", choices = data.revisionChoices ?? [];
  const endpointCard = endpointEstimate && endpointRows.length > 0 && <section className="photo-report-object-evidence" aria-label={t("PhotoReporttsx.text050")} data-endpoint-revision={revision.id}>
    <h3>{t("PhotoReporttsx.text051")}</h3><p>{data.revision.label ?? t("PhotoReporttsx.text052")} {t("PhotoReporttsx.text053")}</p>
    {sideOrder.map(side => { const rows = endpointRows.filter(row => row.side === side); if (!rows.length) return null; const pairs = differences.filter(row => rows.some(point => point.id === row.minuendId) && rows.some(point => point.id === row.subtrahendId));
      return <div key={side ?? "unsided"} className="photo-report-endpoint-side"><h4>{side ? sideTitle(side, endpointEstimate.sidePhoto) : t("PhotoReporttsx.text054")}</h4><dl>{rows.map(point => <div key={point.id}><dt>{point.label}{t("ReportScenetsx.213")}</dt><dd data-endpoint-estimate={point.id} data-endpoint-object={point.objectId}>{endpointValue(point)}</dd><small>{scopeOf(point)}{point.pairingStatus === "no_adjacent_lower_rail" ? t("PhotoReporttsx.text055") : ""}</small></div>)}{pairs.map(row => <div key={row.id}><dt>{row.label}</dt><dd data-endpoint-difference={row.id}>{differenceValue(row)}</dd></div>)}</dl></div>; })}
    {differences.filter(row => row.id.endsWith("-left-minus-right")).map(row => <p key={row.id} className="photo-report-endpoint-lr"><strong>{row.label}</strong> <span data-endpoint-difference={row.id}>{differenceValue(row)}</span><small>{row.description}</small></p>)}
    {(endpointEstimate.excludedComparisons ?? []).map(row => <p key={row.id} className="photo-report-endpoint-lr" data-endpoint-excluded={row.id}><strong>{t("PhotoReporttsx.text056")}</strong> <small>{row.reason}</small></p>)}
    <button type="button" aria-pressed={showEndpointComparison} onClick={() => { const target = selectedEndpoint ?? firstEndpoint; setShowEndpointComparison(value => !value); if (target && !selectedEndpoint) { setEntityId(target.objectId); setObservationId(null); } setViewRequest(request => ({ view: "model", nonce: (request?.nonce ?? 0) + 1 })); }}>{showEndpointComparison ? t("PhotoReporttsx.text057") : t("PhotoReporttsx.text058")}</button>
  </section>;
  const spatialFacts = (id: string): SpatialFact[] => {
    const facts: SpatialFact[] = [];
    const versionLabel = data.revision.label ?? revision.id;
    const photosOf = [...new Set(data.objects.find(row => row.id === id)?.observations.map(row => row.photo) ?? [])];
    const observed = revision.document.entities.find(entity => entity.id === id)?.observationRefs?.length ?? 0;
    // Context facts are read from the loaded revision too; nothing is carried over from the experiment's run.
    const context: SpatialFact[] = [
      { label: t("PhotoReporttsx.text059"), value: t("PhotoReporttsx.text060", {p0: observed, p1: photosOf.join(" / ") || t("VideoViewtsx.267")}), status: "source_linked", source: versionLabel },
      { label: t("PhotoReporttsx.text061"), value: t("PhotoReporttsx.text062"), status: data.geometry.floor.status, source: versionLabel },
      { label: t("PhotoReporttsx.text063"), value: acceptedScale != null ? t("PhotoReporttsx.text064") : modelScale.nativeToMeters == null ? t("PhotoReporttsx.text065") : t("PhotoReporttsx.text066"), status: modelScale.status, source: "modelMeasurementScale" }];
    for (const row of endpointsOf(id)) facts.push({ label: t("PhotoReporttsx.text067", {p0: row.label}), value: endpointValue(row), testId: row.id,
      status: t("PhotoReporttsx.text068", {p0: scopeOf(row)}), source: `${data.revision.label ?? revision.id} · ${row.modelFile} ${row.modelSha256.slice(0, 8)}` });
    for (const row of differences.filter(row => [row.minuendId, row.subtrahendId].some(point => endpointRows.find(item => item.id === point)?.objectId === id)))
      facts.push({ label: row.label, value: differenceValue(row), testId: row.id, status: row.description });
    const feature = physicalFeature(id), item = data.objects.find(row => row.id === id);
    if (feature?.valueNative != null) facts.push({ label: t("PhotoReporttsx.text069"), value: item?.physicalBottom ? modelCentimeters(feature.valueNative) : displayValue(feature.valueNative), status: t("PhotoReporttsx.text070"), source: t("PhotoReporttsx.text071", {p0: (feature.sourcePhotos ?? []).join(" / ")}) });
    const angle = data.bendAnalysis?.revisionId === revision.id ? data.bendAnalysis.items.find(row => row.entityId === id) : undefined;
    if (angle?.status === "measured" && angle.result) facts.push({ label: t("PhotoReporttsx.text072"), value: `${angle.result.value.toFixed(1)}°`, status: t("PhotoReporttsx.text073") });
    if (facts.length) facts.push({ label: t("PhotoReporttsx.text074"), value: modelScale.nativeToMeters == null ? t("PhotoReporttsx.text075") : `1 native = ${modelScale.nativeToMeters.toFixed(5)} m`, status: modelScale.status === "conditional_unvalidated" ? t("PhotoReporttsx.text076") : modelScale.status, source: modelScale.source });
    return [...facts, ...context];
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
  const focus = [...new Map(endpointRows.map(row => [row.objectId, row.label])).entries(), ...(data.objects.some(item => item.id === "floor") ? [["floor", t("PhotoReporttsx.text077")]] : [])];
  // Recomputed on every render: an answer always reads the loaded revision and the current scale.
  const answer = asked.trim() ? answerSpatialQuery(asked, { language, objects: data.objects, endpoints: endpointRows, differences, excluded: endpointEstimate?.excludedComparisons, bends: data.bendAnalysis?.revisionId === revision.id ? data.bendAnalysis.items : [],
    bound, format: modelCentimeters, revisionLabel: data.revision.label ?? revision.id, sidePhoto: referencePhoto }) : null;
  const askCard = <section className="photo-report-ask" id="ask" aria-label={t("PhotoReporttsx.text078")}>
    <h2>{t("PhotoReporttsx.text079")}</h2>
    <form onSubmit={event => { event.preventDefault(); setAsked(question); }}><label>{t("PhotoReporttsx.text080")}<input data-spatial-question value={question} placeholder={t("PhotoReporttsx.text081")} onChange={event => setQuestion(event.target.value)} /></label><button type="submit">{t("LiveReporttsx.116")}</button></form>
    <p className="photo-semantic-note">{t("PhotoReporttsx.text082")}</p>
    {answer && <div className="photo-report-answer" data-answer-status={answer.status} role="status"><p data-spatial-answer>{answer.text}</p><small>{answer.interpretation}</small>
      {answer.objects.length > 0 && <div>{answer.objects.map(id => <button type="button" key={id} onClick={() => selectSemanticObject(id)}>{data.objects.find(item => item.id === id)?.label ?? id}</button>)}</div>}</div>}
  </section>;
  return <SceneResources.Provider value={resources}><main className="photo-report" data-revision-id={revision.id} data-document-sha256={revision.documentSha256}>
    <header className="photo-report-header"><a className="photo-report-brand" href="#overview">PANOPTES <span>{t("photo.workcellReport")}</span></a><nav><LanguageSwitch /><a href="#overview">{t("PhotoReporttsx.text083")}</a><a href="#ask">{t("PhotoReporttsx.text078")}</a><a href="#scene">{t("PhotoReporttsx.text084")}</a>{data.semanticExperiment && <a href="#semantics">{t("PhotoReporttsx.text085")}</a>}{data.policyEvidence && <a href="#policy">{t("PhotoReporttsx.text086")}</a>}<a href="#sources">{t("PhotoReporttsx.text087")}</a></nav></header>
    <section className="photo-report-overview" id="overview"><div><p className="photo-report-eyebrow">{revision.document.cameras.length} {t("PhotoReporttsx.text088")}{data.capture ? t("PhotoReporttsx.text089") : ""} {t("PhotoReporttsx.text090")}</p><h1>{t("PhotoReporttsx.text091")}</h1><p className="photo-report-scale-badge" data-scale-badge={modelScale.nativeToMeters == null ? "native" : "conditional"}>{modelScale.nativeToMeters == null ? t("PhotoReporttsx.text092") : t("PhotoReporttsx.text093")}</p><p>{t("PhotoReporttsx.text094")}</p>{data.capture?.sources && <p data-capture-sources>{data.capture.sources.map(row => t("PhotoReporttsx.text095", {p0: row.photo, p1: row.name})).join(" · ")}{t("PhotoReporttsx.text096")}{data.capture.referencePhoto}{t("PhotoReporttsx.text097")}</p>}</div><dl><div><dt>{t("photos")}</dt><dd>{revision.document.cameras.length}</dd></div><div><dt>{t("objects")}</dt><dd>{data.objects.length}</dd></div><div><dt>{reference ? t("PhotoReporttsx.text098") : data.experiment?.timingLabel || t("PhotoReporttsx.text099")}</dt><dd>{reference ? (reference.wholeComponentHeightM * 100).toFixed(1) : data.timing.oneShotSeconds?.toFixed(1) ?? "—"}<small>{reference ? "cm" : t("seconds")}</small></dd></div>{reference && <div><dt>{data.measurementUpdate?.kind === "saved-geometry-replay" ? t("PhotoReporttsx.text100") : t("PhotoReporttsx.text101")}</dt><dd>{data.timing.oneShotSeconds?.toFixed(1) ?? "—"}<small>{t("seconds")}</small></dd></div>}</dl></section>
    {choices.length > 0 && <section className={`photo-report-revision${candidate ? " photo-report-warning" : ""}`} aria-label={t("PhotoReporttsx.text102")}>
      <label>{t("PhotoReporttsx.text102")}<select data-revision-select value={revision.id} onChange={event => { const choice = choices.find(item => item.id === event.target.value); if (choice && onRevision) onRevision(choice, { object: entityId, photo }); }}>{choices.map(choice => <option key={choice.id} value={choice.id}>{choice.label} · {choice.status === "main" ? t("PhotoReporttsx.text103") : t("PhotoReporttsx.text104")}</option>)}</select></label>
      <p>{t("PhotoReporttsx.text105")}<strong>{data.revision.label ?? revision.id}</strong> · {t("photo.revision")} <code>{revision.id}</code> {t("PhotoReporttsx.text106")}<code>{revision.documentSha256.slice(0, 12)}</code>。{candidate ? t("PhotoReporttsx.text107") : t("PhotoReporttsx.text108")}{t("PhotoReporttsx.text109")}{data.lineage?.candidateEvidence?.url && <> <a href={new URL(data.lineage.candidateEvidence.url, window.location.href).href}>{t("PhotoReporttsx.text110")}</a></>}</p>
    </section>}
    {endpointCard && <section className="photo-report-evaluation photo-report-endpoints" aria-label={t("PhotoReporttsx.text111")}>
      {endpointCard}<p><a className="photo-report-endpoint-link" href={`?${choices.length ? `version=${revision.id}&` : ""}photo=${endpointEstimate!.sidePhoto}&object=${firstEndpoint!.objectId}&view=model&measurement=endpoints#scene`}>{t("PhotoReporttsx.text112")}{endpointEstimate!.sidePhoto} {t("PhotoReporttsx.text113")}</a></p>
      {endpointTruth?.length ? <div className="photo-report-table-scroll"><table data-endpoint-check><caption>{t("PhotoReporttsx.text114")}</caption><thead><tr><th scope="col">{t("PhotoReporttsx.text115")}</th><th scope="col">{t("PhotoReporttsx.text116")}</th><th scope="col">{t("PhotoReporttsx.text117")}</th><th scope="col">{t("PhotoReporttsx.text118")}</th><th scope="col">{t("PhotoReporttsx.text119")}</th></tr></thead><tbody>{endpointTruth.flatMap(truth => { const points = endpointRows.filter(point => point.objectId === truth.objectId); return points.map(point => {
        // One check value names one place on the object; with several measured points it cannot be assigned to any one of them.
        const estimate = points.length === 1 && bound(point) && modelScale.nativeToMeters != null ? point.heightNative * modelScale.nativeToMeters : null, error = estimate == null ? null : estimate - truth.groundTruthM;
        return <tr key={point.id} data-endpoint-check-row={point.id}><th scope="row">{point.label}</th><td>{scopeOf(point)}</td><td>{bound(point) ? modelCentimeters(point.heightNative) : t("unknown")}</td><td>{(truth.groundTruthM * 100).toFixed(2)} cm</td><td data-endpoint-check-error>{point.railPart === "lower_envelope_hypothesis" ? t("PhotoReporttsx.text120") : points.length > 1 ? t("PhotoReporttsx.text121") : error == null ? t("unknown") : `${error > 0 ? "+" : ""}${(error * 100).toFixed(2)} cm`}</td></tr>;
      }); })}</tbody></table><p>{t("PhotoReporttsx.text122")}{(() => { const missing = endpointRows.filter(point => !endpointTruth.some(truth => truth.objectId === point.objectId)); return missing.length ? t("PhotoReporttsx.text123", {p0: missing.map(point => point.label).join("、")}) : ""; })()}{t("PhotoReporttsx.text124")}</p></div> : null}
      <details><summary>{t("PhotoReporttsx.text125")}</summary><p>{t("PhotoReporttsx.text126")}</p>
      <p>{endpointRows.map(point => t("PhotoReporttsx.text127", {p0: point.label, p1: bound(point) ? modelRange(point.heightNative) : t("unknown")})).join("；")}{t("PhotoReporttsx.text128")}</p>
      <p>{t("PhotoReporttsx.text129")}{modelScale.nativeToMeters?.toFixed(5) ?? t("unknown")} m。{modelScale.source}</p><p>{t("PhotoReporttsx.text130")}{endpointEstimate!.method}</p>
      <p>{t("PhotoReporttsx.text131")}{endpointRows.map(point => `${point.label} ← ${point.modelFile} ${point.modelSha256.slice(0, 8)}`).join("；")}。</p>
      <p>{t("PhotoReporttsx.text132")}</p>
      <p><a href={resolve("model-endpoint-estimate.json")} download>{t("PhotoReporttsx.text133")}</a></p></details>
    </section>}
    {data.metrology && <section className="photo-report-sources" aria-label={t("PhotoReporttsx.text134")}><h2>{t("PhotoReporttsx.text135")}</h2><p>{data.metrology.summary}</p><a href={data.metrology.reportURL}>{t("PhotoReporttsx.text136")}</a></section>}
    {data.experiment && <section className="photo-report-sources" aria-label={t("PhotoReporttsx.text137")}>{reference ? <details><summary>{data.experiment.title}</summary><p>{data.experiment.summary}</p><a href={data.experiment.reportURL}>{t("PhotoReporttsx.text138")}</a></details> : <><h2>{data.experiment.title}</h2><p>{data.experiment.summary}</p><a href={data.experiment.reportURL}>{t("PhotoReporttsx.text138")}</a></>}</section>}
    <section className="photo-report-scale" aria-label={t("PhotoReporttsx.text139")}>
      <div><h2>{t("PhotoReporttsx.text140")}</h2><p>{t("PhotoReporttsx.text141")}</p></div>
      {reference && <dl className="photo-report-reference" data-measured-reference><div><dt>{t("PhotoReporttsx.text142")}</dt><dd>{centimeters(referenceRatio == null ? null : reference.redActuatorDiameterM * referenceRatio)}</dd></div><div><dt>{t("PhotoReporttsx.text143")}</dt><dd>{centimeters(referenceRatio == null ? null : reference.mainBodyDiameterM * referenceRatio)}</dd></div><div><dt>{t("PhotoReporttsx.text144")}</dt><dd>{centimeters(valid ? height / 100 : null)}</dd></div></dl>}
      <label>{t("PhotoReporttsx.text145")}<input aria-label={t("PhotoReporttsx.text146")} type="number" min=".01" step=".1" value={Number.isFinite(height) ? height : ""} onChange={event => setHeight(event.target.valueAsNumber)} /></label>
      <button disabled={exporting} onClick={downloadModel}>{exporting ? t("PhotoReporttsx.text147") : modelScale.nativeToMeters == null ? t("PhotoReporttsx.text148") : modelScale.status === "conditional_unvalidated" ? t("PhotoReporttsx.text149") : t("PhotoReporttsx.text150")}</button>
      <p className="photo-report-scale-result" data-model-native-to-meters={modelScale.nativeToMeters ?? "unknown"}>{modelScale.nativeToMeters == null ? t("PhotoReporttsx.text151") : t("PhotoReporttsx.text152", {p0: modelScale.nativeToMeters.toFixed(5), p1: modelScale.status === "conditional_unvalidated" ? t("PhotoReporttsx.text076") : t("PhotoReporttsx.text004")})}</p><p>{modelScale.source}</p>
      <p className="photo-report-scale-result" data-native-to-meters={nativeToMeters ?? "unknown"}>{nativeToMeters == null ? t("PhotoReporttsx.text153") : t("PhotoReporttsx.text154", {p0: nativeToMeters.toFixed(5)})}</p>
      <p className="photo-report-scale-result">{t("PhotoReporttsx.text155")}</p>
      {reference && <p className="photo-report-scale-result">{t("PhotoReporttsx.text156")}{centimeters(reference.wholeComponentHeightM)}{t("PhotoReporttsx.text157")}{centimeters(reference.mainBodyDiameterM)}{t("PhotoReporttsx.text158")}{centimeters(reference.redActuatorDiameterM)}。{calibration?.reference.scopeStatus === "pending_confirmation" ? t("PhotoReporttsx.text159") : t("PhotoReporttsx.text160")}</p>}
      {referenceRatio != null && Math.abs(referenceRatio - 1) > 1e-8 && <p className="photo-report-scale-result photo-report-warning" role="status">{t("PhotoReporttsx.text161")}</p>}
      {acceptedScale == null && <p className="photo-report-scale-result photo-report-warning" role="status">{t("PhotoReporttsx.text162")}{anchor.referenceFit?.reason}</p>}
      {!valid && <p role="alert">{t("PhotoReporttsx.text163")}</p>}{exportError && <p role="alert">{t("PhotoReporttsx.text164")}{exportError}</p>}
    </section>
    {data.measurementEvaluation?.comparisons.length ? <section className="photo-report-evaluation" aria-label={t("PhotoReporttsx.text165")}>
      <h2>{t("PhotoReporttsx.text166")}</h2><p>{t("PhotoReporttsx.text167")}{endpointCard ? t("PhotoReporttsx.text168") : ""}{t("PhotoReporttsx.text169")}</p>
      <div className="photo-report-table-scroll"><table><thead><tr><th scope="col">{t("PhotoReporttsx.text170")}</th><th scope="col">{t("PhotoReporttsx.text171")}</th><th scope="col">{t("PhotoReporttsx.text172")}</th><th scope="col">{t("PhotoReporttsx.text173")}</th><th scope="col">{t("PhotoReporttsx.text174")}</th></tr></thead><tbody>{data.measurementEvaluation.comparisons.map(row => {
        const estimate = row.estimateNative == null || nativeToMeters == null ? null : row.estimateNative * nativeToMeters;
        const error = estimate == null ? null : estimate - row.groundTruthM;
        return <tr key={row.objectId} data-measurement-comparison={row.objectId}><th scope="row"><button onClick={() => { setEntityId(row.objectId); setObservationId(null); setShowGroundDistance(true); document.getElementById("scene")?.scrollIntoView({ behavior: "smooth", block: "start" }); }}>{row.label}</button><small>{row.estimateNative == null ? t("PhotoReporttsx.text175") : t("PhotoReporttsx.text176")} {t("PhotoReporttsx.text177")}{row.sourcePhotos.join(" / ")}</small></th><td data-comparison-estimate>{centimeters(estimate)}</td><td data-comparison-truth>{centimeters(row.groundTruthM)}</td><td data-comparison-error>{error == null ? t("unknown") : `${error > 0 ? "+" : ""}${centimeters(error)} (${(error / row.groundTruthM * 100).toFixed(1)}%)`}</td><td>{nativeToMeters != null && row.rangeNative ? row.rangeNative.map(x => centimeters(x * nativeToMeters)).join(" – ") : t("unknown")}</td></tr>;
      })}</tbody></table></div>
      <details><summary>{t("PhotoReporttsx.text178")}</summary>{data.measurementEvaluation.comparisons.map(row => <article key={row.objectId}><h3>{row.label}</h3><p>{row.source}</p><p>{row.limitation}</p><p>{t("PhotoReporttsx.text179")}{row.sourcePhotos.join(" / ") || t("PhotoReporttsx.text180")}</p></article>)}</details>
      <p><a href={resolve("measurement-evaluation.json")} download>{t("PhotoReporttsx.text181")}{centimeters(reference?.wholeComponentHeightM)} {t("PhotoReporttsx.text182")}</a> · <a href={resolve("measurements.json")} download>{t("PhotoReporttsx.text183")}</a></p>
    </section> : data.measurementEvaluation?.absentTargets?.length ? <section className="photo-report-evaluation" aria-label={t("PhotoReporttsx.text165")} data-absent-targets>
      <h2>{t("PhotoReporttsx.text184")}</h2><p>{data.measurementEvaluation.absentTargets.map(row => `${row.objectId}（${row.feature}）：${row.reason}`).join("；")}</p>
    </section> : null}
    {askCard}
    {data.semanticExperiment && <PhotoSemanticExperiment data={data.semanticExperiment} selectedEntityId={entityId} onSelect={selectSemanticObject} />}
    {data.policyEvidence && data.policyEvidence.revisionId === revision.id && <section id="policy" className="photo-report-sources photo-report-policy" aria-label={t("PhotoReporttsx.text185")} data-policy-revision={data.policyEvidence.revisionId}>
      <h2>{t("PhotoReporttsx.text186")}</h2><p><strong>{t("PhotoReporttsx.text187")}</strong>{t("PhotoReporttsx.text188")}{data.policyEvidence.engine}{t("PhotoReporttsx.text189")}{data.policyEvidence.ruleSet.file}（{data.policyEvidence.ruleSet.sha256.slice(0, 12)}{t("PhotoReporttsx.text190")}</p>
      {data.policyEvidence.items.map(item => <article key={item.policyId} data-policy-id={item.policyId}><h3>{item.sourceText}</h3>
        <p>{t("PhotoReporttsx.text191")}{item.policyId} · {item.compileStatus === "refused" ? t("PhotoReporttsx.text192") : `${item.predicate} ${item.threshold} ${item.unit}`} {t("PhotoReporttsx.text193")}{item.applicability === "unknown" ? t("PhotoReporttsx.text194") : item.applicability} {t("PhotoReporttsx.text195")}</p>
        {item.refusal && <p>{t("PhotoReporttsx.text196")}{item.refusal}</p>}
        {item.subjects.length > 0 && <p>{t("PhotoReporttsx.text197")}{item.subjects.map(subject => <button type="button" key={subject.entityId} onClick={() => selectSemanticObject(subject.entityId)}>{subject.label}</button>)}<small>{t("PhotoReporttsx.text198")}</small></p>}
        {item.evidenceStillNeeded && <ul>{item.evidenceStillNeeded.map(need => <li key={need.id} data-evidence-status={need.status}>{t(evidenceName[need.id] ?? need.id)}：{need.status === "available" ? t("PhotoReporttsx.text199") : t("PhotoReporttsx.text200")} · {need.detail}</li>)}</ul>}
      </article>)}
      <p>{t("PhotoReporttsx.text201")}</p>
    </section>}
    {!data.semanticExperiment && data.semanticBinding && <section id="semantics" className="photo-semantic-experiment photo-report-warning" data-semantic-binding-status={data.semanticBinding.status}><h2>{t("PhotoReporttsx.text202")}</h2><p>{t("PhotoReporttsx.text203")}{data.semanticBinding.reason}</p><p>{data.semanticBinding.action}</p></section>}
    <div id="scene"><nav className="photo-report-focus" aria-label={t("PhotoReporttsx.text204")}><strong>{t("PhotoReporttsx.text205")}</strong>{focus.map(([id, label]) => <button key={id} type="button" aria-pressed={entityId === id} onClick={() => { setEntityId(id); setImageId(`photo-${referencePhoto}`); setObservationId(null); showMeasurementOf(id); }}>{label}</button>)}<small>{t("PhotoReporttsx.text206")}</small></nav><ReportScene matchedComparison measurementScale={modelScale} initialView={initialView} viewRequest={viewRequest} measurementOverride={sceneAnnotation} revision={revision} selection={selection} onSelect={(id, obs) => { setEntityId(id); setObservationId(obs || null); setShowGroundDistance(false); }} imageId={imageId} cameraId={camera?.id || null} onCamera={(id) => { setImageId(id); setObservationId(null); }} onClearSelection={() => { setEntityId(null); setShowGroundDistance(false); }} inspector={(surface) => selected ? <>
      {data.semanticExperiment && <PhotoSemanticObject data={data.semanticExperiment} entityId={selected.id} onSelect={selectSemanticObject} facts={spatialFacts(selected.id)} resolve={resolve} onMeasure={endpointsOf(selected.id).some(bound) || physicalFeature(selected.id) ? () => showMeasurementOf(selected.id) : undefined} measureLabel={t("PhotoReporttsx.text207")} />}
      {selectedEndpoint && endpointCard}<section className="photo-report-object-evidence" data-selected-object={selected.id}><h3>{selected.label}</h3>{box && <dl className="photo-report-model-box" data-model-box={selected.id}><div><dt>{t("PhotoReporttsx.text208")}{modelScale.nativeToMeters == null ? t("measure.unit.native") : t("PhotoReporttsx.text070")}）</dt><dd data-model-box-size>{modelCentimeters(box.lengthNative)} × {modelCentimeters(box.widthNative)} × {modelCentimeters(box.heightNative)}</dd></div><div><dt>{t("PhotoReporttsx.text209")}</dt><dd data-model-box-bottom>{modelCentimeters(box.bottomNative)}</dd></div><div><dt>{t("PhotoReporttsx.text210")}</dt><dd data-model-box-top>{modelCentimeters(box.topNative)}</dd></div></dl>}{record?.modelTerminal && <p className="photo-report-warning" role="status">{t("PhotoReporttsx.text211")}{record.modelTerminal.candidateStatus}{t("PhotoReporttsx.text212")}</p>}{bend && <section className="photo-report-angles photo-report-bend" data-bend-entity={selected.id}><h4>{t("PhotoReporttsx.text213")}</h4>{bend.status === "measured" && bend.result ? <><output>{bend.result.value.toFixed(1)}°</output><p>{t("PhotoReporttsx.text214")}</p><p>{t("PhotoReporttsx.text215")}</p></> : <p>{t("PhotoReporttsx.text216")}{bend.reason || bend.status}</p>}</section>}<dl><div><dt>{record?.id === "emergency-button" && reference ? t("PhotoReporttsx.text217") : physicalHeight != null ? t("PhotoReporttsx.text218") : t("PhotoReporttsx.text219")}</dt><dd data-height-native={physicalHeight ?? visibleHeight ?? "unknown"}>{record?.id === "emergency-button" && reference ? centimeters(valid ? height / 100 : null) : displayValue(physicalHeight ?? visibleHeight)}</dd></div><div><dt>{record?.physicalBottom ? t("PhotoReporttsx.text220") : t("PhotoReporttsx.text221")}</dt><dd data-ground-distance-native={groundValue ?? "unknown"}>{groundDisplay}</dd></div>{endpointsOf(selected.id).map(row => <div key={row.id}><dt>{row.label}{t("PhotoReporttsx.text222")}</dt><dd data-selected-endpoint={row.id}>{endpointValue(row)}</dd></div>)}</dl>{selected.observedExtentAvailable === true && record?.visibleHeightRangeNative && <p>{t("PhotoReporttsx.text223")}{record.visibleHeightRangeNative.map(displayValue).join(" – ")}</p>}<p>{t("PhotoReporttsx.text179")}{[...new Set(record?.observations.map(item => item.photo))].join(" / ") || t("VideoViewtsx.267")}</p><p>{record?.representation}</p>{Object.keys(selected.modelVariants ?? {}).length > 0 && <p data-fixed-scene-model>{t("PhotoReporttsx.text224")}{referencePhoto} {t("PhotoReporttsx.text225")}</p>}{record?.multiViewModel && <p data-multi-view-model>{t("PhotoReporttsx.text226")}{record.multiViewModel.generationViews.join(" / ")} {t("PhotoReporttsx.text227")}{record.multiViewModel.refineViews.join(" / ")} {t("PhotoReporttsx.text228")}{Object.entries(record.multiViewModel.maskIoUByPhoto).map(([photo, iou]) => t("PhotoReporttsx.text229", {p0: photo, p1: iou.toFixed(2)})).join("，")}{t("PhotoReporttsx.text230")}</p>}{sourceEdgeAvailable && <button type="button" aria-pressed={showGroundDistance} onClick={() => { setShowEndpointComparison(false); setShowGroundDistance(value => !value); }}>{showGroundDistance ? t("PhotoReporttsx.text231") : t("PhotoReporttsx.text232")}</button>}{groundRange && <p data-ground-distance-range>{t("PhotoReporttsx.text233")}{groundRange.map(displayValue).join(" – ")}{t("PhotoReporttsx.text234")}</p>}<p>{t("PhotoReporttsx.text235")}{ground?.source || distance?.source || t("PhotoReporttsx.text236")}</p>{!sourceEdgeAvailable && <p>{distance?.reason || t("PhotoReporttsx.text237")}</p>}{physicalHeight == null && <p>{t("PhotoReporttsx.text238")}</p>}</section>
      {inclination && <section className="photo-report-angles" data-inclination-entity={selected.id}><h3>{t("PhotoReporttsx.text239")}</h3>
        {surface && <div key={surface.surfaceId} data-inclination-surface={surface.surfaceId}><h4>{t("PhotoReporttsx.text240")}{surface.surfaceId}</h4><p>{t("PhotoReporttsx.text241")}<strong>{surface.inclinationDeg.toFixed(1)}°</strong>{t("PhotoReporttsx.text242")}</p><p>{t("PhotoReporttsx.text243")}<strong>{surface.deviationFromVerticalDeg.toFixed(1)}°</strong></p><p>{({ non_vertical: t("PhotoReporttsx.text244"), vertical: t("PhotoReporttsx.text245"), direction_unverified: t("PhotoReporttsx.text246") } as Record<string, string>)[surface.classification] || surface.classification} {t("PhotoReporttsx.text247")}{surface.angularSpreadDeg.toFixed(1)}°</p><p>{surface.result.quality.angularErrorDeg == null ? t("PhotoReporttsx.text248") : t("PhotoReporttsx.text249", {p0: surface.result.quality.angularErrorDeg.toFixed(1)})}</p></div>}
        {!inclination.surfaces.length && <p>{t("PhotoReporttsx.text250")}{inclination.reason || inclination.status}</p>}
        {inclination.surfaces.length > 0 && <p>{t("saved")}{inclination.surfaces.length} {t("PhotoReporttsx.text251")}{surface ? t("PhotoReporttsx.text252") : t("PhotoReporttsx.text253")}{t("PhotoReporttsx.text254")}</p>}
      </section>}
      {nativeToMeters == null && <p>{t("PhotoReporttsx.text255")}</p>}
      <ObjectFacts entity={selected} document={revision.document} />
      <details className="report-source-details" open><summary>{t("PhotoReporttsx.text256")}</summary>{record?.notes.map((note, index) => <p key={index}>{note}</p>)}</details>
    </> : <p>{t("PhotoReporttsx.text257")}</p>} /></div>
    <section id="sources" className="photo-report-sources"><h2>{t("PhotoReporttsx.text087")}</h2><p>{data.capture ? t("PhotoReporttsx.text258") : ""}{t("PhotoReporttsx.text259")}{anchor.referenceFit?.status === "available" ? t("PhotoReporttsx.text260") : t("PhotoReporttsx.text261")}{t("PhotoReporttsx.text262")}</p><p>{data.geometry.floor.status}</p>{anchor.assumptions?.map((note, i) => <p key={i}>{note}</p>)}{data.measurementUpdate?.kind === "saved-geometry-replay" && <p>{t("PhotoReporttsx.text263")}{data.measurementUpdate.sourceRevisionId}{t("PhotoReporttsx.text264")}{data.measurementUpdate.revisionBuildSeconds?.toFixed(1) ?? "—"} {t("PhotoReporttsx.text265")}</p>}<a href={resolve("scene-report.json")} download>{t("PhotoReporttsx.text266")}</a></section>
  </main></SceneResources.Provider>;
}
function validReport(value: any): value is PhotoReportData {
  return !!value?.revision?.document && !!value.assetURLs && Array.isArray(value.objects) && !!value.geometry?.anchor && (value.geometry.anchor.mPerNative == null || (Number.isFinite(value.geometry.anchor.mPerNative) && value.geometry.anchor.mPerNative > 0));
}
function LoadPhotoReport() {
  const { t } = useI18n();
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
  return error ? <main className="photo-report"><h1>{t("PhotoReporttsx.text267")}</h1><p role="alert">{error}</p></main> : loaded ? <PhotoReport key={loaded.data.revision.id} data={loaded.data} base={loaded.base} onRevision={switchRevision} /> : <main className="photo-report" role="status">{t("PhotoReporttsx.text268")}</main>;
}
if (typeof document !== "undefined" && document.getElementById("root")) createRoot(document.getElementById("root")!).render(<I18nProvider><LoadPhotoReport /></I18nProvider>);
