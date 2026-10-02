import type { SurfacePick } from "./viewer/native-math";
import { useEffect, useId, useRef, useState, type ReactNode, type PointerEvent } from "react";
import { SpatialView } from "./App";
import { PhotoView } from "./PhotoView";
import { ComparisonVideo, VideoMemory, VideoView, videoReplay } from "./VideoView";
import { request } from "./api";
import { useSceneResources } from "./SceneResources";
import { SpatialMeasurements, type BendAnalysis, type InclinationAnalysis, type InclinationSurface, type SceneMeasurement, type MeasureRegion } from "./SpatialMeasurements";
import { CadView } from "./CadView";
import { splatAnnotation } from "./viewer/splat-layer";
import { useI18n } from "./i18n";
import { cameraForImage, activeModel, compositeModelEvidence, modelPreviewEntities, modelPreviewGeometry, modelPreviewSignature, observationsFor, entityGeometryForLayer, jsonObject, planShapes, cadReferenceImage, scenePlanOptions, sourceDimensions, sourceScale, type GeometryOptions, type PlanOptions } from "./core";
import { entityEvidenceStatus, identityCounts, isReferenceSurface } from "./scene-semantics";
import {
  add,
  cameraMatrix,
  projected,
  scale,
  sourceCamera,
  transformMatrix,
  unit,
  type Vec,
} from "./viewer/native-math";
import type { Camera, Entity, Revision, Selection, SceneDocument, RepresentationLoadState } from "./types";
import "./report-scene.css";

type Pane = "photo" | "spatial" | "cad" | "plan";
type Layer = "model" | "observed_surface" | "point_cloud";
const paneNames: Record<Pane, string> = {
  photo: "scenePhoto",
  spatial: "scene3D",
  cad: "sceneCAD",
  plan: "sceneSelectedModel",
};
const paneOrder: Pane[] = ["photo", "spatial", "cad", "plan"];
const colors = ["#e86b58", "#39ad7c", "#458ce0"];
const noEdit = () => {};

function sceneAvailability(document: SceneDocument, geometryOptions: PlanOptions, spatialLayer = geometryOptions.layer) {
  const frames = new Set(document.coordinateFrames.map((frame) => frame.id));
  const assets = new Set(document.assets.map((asset) => asset.id));
  const representations = document.entities.flatMap((entity) =>
    (entity.representations || []).filter((representation) =>
      frames.has(representation.coordinateFrameId) &&
      representation.transform.coordinateFrameId === representation.coordinateFrameId &&
      (representation.kind === "primitive" ? !!representation.primitive : !!representation.assetId && assets.has(representation.assetId))));
  const hasObserved = representations.some((representation) =>
    ["observed_surface", "point_cloud"].includes(representation.kind) && representation.placementState === "confirmed");
  const hasModel = representations.some((representation) => ["generated_mesh", "primitive"].includes(representation.kind));
  const hasGround = document.coordinateFrames.some((frame) => {
    if (frame.id !== geometryOptions.frameId) return false;
    const normal = frame.ground?.normal;
    return Array.isArray(normal) && normal.length === 3 && normal.every(Number.isFinite) && Math.hypot(...normal) > 1e-8;
  });
  return {
    spatialTitle: spatialLayer === "model" ? "sceneModelScene" : hasModel && !hasObserved ? "sceneObjectModels" : "scene3D",
    planEmpty: planShapes(document, geometryOptions).length ? null : geometryOptions.scope === "scene" && !geometryOptions.frameId ? "sceneNoPlanFrame" : hasGround ? "sceneNoPlanProjection" : "sceneNoPlanGround",
  };
}

// Uses the same camera projection and pixel-centre convention as the WebGL view.
function photoOverlay(document: SceneDocument, entity: Entity, camera: Camera, layer: Layer, observationId?: string | null) {
  if (isReferenceSurface(document, entity)) return null;
  const geometry = entityGeometryForLayer(entity, { layer, frameId: camera.coordinateFrameId, showCandidates: true, imageId: camera.imageId, observationId, observations: document.observations });
  if (
    !geometry ||
    geometry.transform.coordinateFrameId !== camera.coordinateFrameId
  )
    return null;
  const { corners, transform } = geometry;
  const center = [0, 1, 2].map(
    (k) => corners.reduce((total, p) => total + p[k], 0) / corners.length,
  );
  const cameraPosition = camera.cameraToWorld.slice(0, 3).map((row) => row[3]);
  const radius = Math.max(
    ...corners.map((p) =>
      Math.hypot(...p.map((value, k) => value - cameraPosition[k])),
    ),
    1e-4,
  );
  const vp = cameraMatrix(
    sourceCamera(camera, radius, center),
    camera.width / camera.height,
    radius,
  );
  const project = (p: Vec) => projected(vp, p, camera.width, camera.height);
  const matrix = transformMatrix(
      geometry.axisSpace === "native" ? undefined : transform,
    ),
    origin = project(center);
  const diagonal = Math.hypot(
    ...[0, 1, 2].map(
      (k) =>
        Math.max(...corners.map((p) => p[k])) -
        Math.min(...corners.map((p) => p[k])),
    ),
  );
  const axes = origin
    ? [0, 1, 2].flatMap((k) => {
        const direction = unit([
          matrix[k * 4],
          matrix[k * 4 + 1],
          matrix[k * 4 + 2],
        ]);
        // Axis length is a viewing aid, not a measurement or a scale anchor.
        const length = Math.max(diagonal * 0.24, radius * 0.012),
          end = project(add(center, scale(direction, length)));
        return end
          ? [{ start: origin, end, label: "XYZ"[k], color: colors[k] }]
          : [];
      })
    : [];
  return { corners: corners.map(project), axes, axisSpace: geometry.axisSpace };
}

function PhotoAxes({
  revision,
  camera,
  layer,
  selectedId,
  observationId,
  allBounds,
}: {
  revision: Revision;
  camera: Camera;
  layer: Layer;
  selectedId: string | null;
  observationId?: string | null;
  allBounds: boolean;
}) {
  return (
    <svg
      className="report-scene-photo-axes"
      viewBox={`0 0 ${camera.width} ${camera.height}`}
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
    >
      {revision.document.entities
        .filter(
          (e) =>
            e.visible !== false &&
            !e.sourceContext &&
            (allBounds || e.id === selectedId),
        )
        .map((entity) => {
          const overlay = photoOverlay(revision.document, entity, camera, layer, entity.id === selectedId ? observationId : undefined);
          if (!overlay) return null;
          const selected = entity.id === selectedId;
          return (
            <g key={entity.id}>
              {overlay.corners.flatMap((a, i) =>
                [0, 1, 2].flatMap((k) => {
                  const b = overlay.corners[i | (1 << k)];
                  return a && b && !(i & (1 << k)) ? (
                    <line
                      key={`${i}:${k}`}
                      x1={a[0]}
                      y1={a[1]}
                      x2={b[0]}
                      y2={b[1]}
                      stroke={selected ? "#e9b367" : "#8eafa4"}
                      strokeWidth={selected ? 2 : 1}
                      vectorEffect="non-scaling-stroke"
                    />
                  ) : (
                    []
                  );
                }),
              )}
              {selected &&
                overlay.axes.map((axis) => (
                  <g key={axis.label}>
                    <line
                      x1={axis.start[0]}
                      y1={axis.start[1]}
                      x2={axis.end[0]}
                      y2={axis.end[1]}
                      stroke={axis.color}
                      strokeWidth={3}
                      vectorEffect="non-scaling-stroke"
                    />
                    <text
                      x={axis.end[0] + camera.width * 0.01}
                      y={axis.end[1] - camera.width * 0.01}
                      fill={axis.color}
                      fontSize={camera.width * 0.035}
                      fontWeight={700}
                      stroke="#172a23"
                      strokeWidth={camera.width * 0.0025}
                      paintOrder="stroke"
                    >
                      {axis.label}
                    </text>
                  </g>
                ))}
            </g>
          );
        })}
    </svg>
  );
}

export function Extent({ entity, document }: { entity: Entity; document: SceneDocument }) {
  const { t } = useI18n(), d = sourceDimensions(entity), scale = sourceScale(document, entity),
    groundDimensions = [d.widthNative, d.depthNative, d.groundHeight],
    values = isReferenceSurface(document, entity) ? [d.widthNative, d.depthNative] :
      groundDimensions.every(Number.isFinite) ? groundDimensions : [d.extentX, d.extentY, d.extentZ];
  if (entity.physicalDimensionsUnknown === true) return <span>物理尺寸未知</span>;
  return values.some(Number.isFinite) ? <span className="report-numeric">
    {values.map((value) => Number.isFinite(value) ? (value! * (scale?.nativeToMeters || 1)).toFixed(2) : "—").join(" × ")}
    <small>{scale?.nativeToMeters ? "m" : t("uncalibrated")}</small>
  </span> : <span>—</span>;
}

export function ReportScene({
  revision, selection, onSelect, imageId, cameraId, onCamera,
  draw = false, onBox, onOpenSourceCad, inspector, objectListRequest = 0, onFeedback, onClearSelection, newerReport, matchedComparison = false, measurementOverride, initialView,
}: {
  matchedComparison?: boolean;
  initialView?: "photo" | "point_cloud" | "model" | "compare";
  measurementOverride?: SceneMeasurement | null;
  revision: Revision;
  selection: Selection;
  onSelect: (entityId: string, observationId?: string) => void;
  imageId: string | null;
  cameraId: string | null;
  onCamera: (imageId: string, cameraId: string | null) => void;
  draw?: boolean;
  onBox?: (box: number[] | null) => void;
  onOpenSourceCad?: () => void;
  inspector?: ReactNode | ((surface: InclinationSurface | undefined) => ReactNode);
  newerReport?: { href: string; title: string };
  objectListRequest?: number;
  onFeedback?: (entityId: string) => void;
  onClearSelection?: () => void;
}) {
  const { t, language } = useI18n(), container = useRef<HTMLElement>(null),
    objectList = useRef<HTMLDivElement>(null), objectSearch = useRef<HTMLInputElement>(null), currentPreviewKey = useRef(""), panePrefix = useId();
  const [layer, setLayer] = useState<Layer>(() => initialView === "point_cloud" ? "point_cloud" : revision.document.entities.some(entity =>
    !entity.sourceContext && activeModel(entity)?.sourceValidity !== "stale" && activeModel(entity)) ? "model" : "observed_surface"),
    [cadLayer, setCadLayer] = useState<"model" | "observed_surface">("model"),
    [allBounds, setAllBounds] = useState(false), [showPath, setShowPath] = useState(true),
    [focused, setFocused] = useState<Pane | null>(initialView === "photo" ? "photo" : initialView === "model" || initialView === "point_cloud" ? "spatial" : null),
    [mobileSection, setMobileSection] = useState("views"),
    [search, setSearch] = useState(""),
    [fullscreenError, setFullscreenError] = useState(false),
    [isFullscreen, setIsFullscreen] = useState(false),
    [previewMode, setPreviewMode] = useState<"free" | "front" | "side" | "top">("free"),
    [modelPreview, setModelPreview] = useState<{ key: string; image: string } | null>(null),
    [modelLoads, setModelLoads] = useState<{ revisionId: string; states: RepresentationLoadState[] } | null>(null),
    [expandedEntities, setExpandedEntities] = useState<Set<string>>(() => new Set());
  const [comparing, setComparing] = useState(matchedComparison && (!initialView || initialView === "compare")), [wipe, setWipe] = useState(50);
  const { analysisAvailable, bendAnalysis: savedBendAnalysis, inclinationAnalysis: savedInclinationAnalysis } = useSceneResources();
  const document = revision.document,
    selected = document.entities.find((entity) => entity.id === selection.entityId);
  const camera = cameraForImage(document, imageId);
  const [cloudWithModels, setCloudWithModels] = useState(!matchedComparison);
  const hasSplats = !!splatAnnotation(document), [splats, setSplats] = useState(true);
  const hasMotion = document.entities.some(entity => (entity as any).motion === "dynamic"), [part, setPart] = useState<"all" | "static" | "dynamic">("all");
  const hasSkeleton = document.entities.some(entity => (entity.representations || []).some(rep => (rep as any).sourceKind === "moving_object_skeleton")), [skeleton, setSkeleton] = useState(false);
  const hasVideo = !!videoReplay(document), [videoMode, setVideoMode] = useState(true), showVideo = hasVideo && videoMode && !draw;  // drawing a missed object needs the still keyframe
  const geometryOptions: GeometryOptions = { layer, frameId: camera?.coordinateFrameId || (!imageId ? document.coordinateFrames[0]?.id : "") || "", showCandidates: true, imageId, observations: document.observations };
  const [rawMeasurement, setMeasurement] = useState<SceneMeasurement | null>(null), [measureRegion, setMeasureRegion] = useState<MeasureRegion | null>(null), [drawingRegion, setDrawingRegion] = useState(false);
  const [remoteBendAnalysis, setBendAnalysis] = useState<BendAnalysis | null>(null), [bendError, setBendError] = useState(false), [showAngles, setShowAngles] = useState(true);
  useEffect(() => {
    if (!analysisAvailable) return;
    const controller = new AbortController(); setBendAnalysis(null); setBendError(false);
    request<BendAnalysis>(`/api/revisions/${revision.id}/bend-analysis-v1`, { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) setBendAnalysis(value); })
      .catch(() => { if (!controller.signal.aborted) setBendError(true); });
    return () => controller.abort();
  }, [revision.id]);
  const [remoteInclinationAnalysis,setInclinationAnalysis]=useState<InclinationAnalysis|null>(null), [inclinationError,setInclinationError]=useState(false), [surfaceKey,setSurfaceKey]=useState(""), [allPlanes,setAllPlanes]=useState(false);
  useEffect(()=>{
    if (!analysisAvailable) return;
    const controller=new AbortController();setInclinationAnalysis(null);setInclinationError(false);setSurfaceKey("");
    request<InclinationAnalysis>(`/api/revisions/${revision.id}/inclination-analysis-v1`,{signal:controller.signal})
      .then(value=>{if(!controller.signal.aborted)setInclinationAnalysis(value);})
      .catch(()=>{if(!controller.signal.aborted)setInclinationError(true);});
    return ()=>controller.abort();
  },[revision.id]);
  const bendAnalysis = analysisAvailable ? remoteBendAnalysis : savedBendAnalysis, inclinationAnalysis = analysisAvailable ? remoteInclinationAnalysis : savedInclinationAnalysis;
  const inclinationRows=inclinationAnalysis?.revisionId===revision.id?inclinationAnalysis.items:[];
  const incompleteInclinations=inclinationRows.filter(row=>row.status==="failed" || row.status==="partial").length;
  const allSurfaces=inclinationRows.flatMap(row=>row.surfaces.map(surface=>({entityId:row.entityId,surface,key:row.entityId+":"+surface.surfaceId})));
  const visibleSurfaces=allSurfaces.filter(row=>allPlanes || row.surface.classification==="non_vertical" || (row.surface.classification==="direction_unverified" && row.surface.deviationFromVerticalDeg>Math.max(row.surface.angularSpreadDeg,0.000001)));
  const activeSurface=allSurfaces.find(row=>row.key===surfaceKey && row.entityId===selected?.id)?.surface;
  useEffect(()=>{ if(activeSurface)setMeasurement(activeSurface.result); },[activeSurface]);
  const bendRows = bendAnalysis?.revisionId === revision.id ? bendAnalysis.items : [];
  const incompleteBends = bendRows.filter(row => row.status === "failed").length;
  const detectedBends = bendRows.filter(row => row.status === "measured" && row.result);
  const savedBend = bendRows.find(row => row.entityId === selected?.id);
  useEffect(() => {
    if (!analysisAvailable && !activeSurface) setMeasurement(savedBend?.status === "measured" ? savedBend.result || null : null);
  }, [analysisAvailable, selected?.id, savedBend, activeSurface]);
  const activeMeasurement = measurementOverride ?? rawMeasurement;
  const measurement = (showAngles || activeMeasurement?.unit !== "deg") && activeMeasurement?.revisionId === revision.id && activeMeasurement.references.some(ref => ref.entityId === selected?.id) ? activeMeasurement : null;
  const [measurePoints,setMeasurePoints]=useState<SurfacePick[]>([]),[pickingPoints,setPickingPoints]=useState(false),[pointError,setPointError]=useState(false),[pointCount,setPointCount]=useState<2|3>(2);
  useEffect(() => { setMeasureRegion(null); setDrawingRegion(false); setMeasurePoints([]); setPickingPoints(false); }, [revision.id, selection.entityId]);
  useEffect(()=>{if(layer!=="model"){setMeasurePoints([]);setPickingPoints(false);}},[layer]);
  function startPointPicking(start:boolean, count:2|3=3) { setPointCount(count); setPickingPoints(start); setMeasurePoints([]); setMeasurement(null); setPointError(false); if(start){setDrawingRegion(false);setLayer("model");setFocused("spatial");setMobileSection("views");} }
  function pickMeasurementPoint(hit:SurfacePick|null) {
    if(!pickingPoints)return;
    if(!hit||measurePoints.length>0&&hit.coordinateFrameId!==measurePoints[0].coordinateFrameId){setPointError(true);return;}
    setPointError(false);const next=[...measurePoints,hit].slice(0,pointCount);setMeasurePoints(next);
    if(next.length===pointCount){setPickingPoints(false);setMobileSection("inspector");}
  }
  const planOptions = scenePlanOptions(document, cadLayer, true, imageId);
  const availability = sceneAvailability(document, planOptions, layer);
  const referenceImageId = selected ? cadReferenceImage(document, selected) : null;
  const images = [...new Set([
    ...document.assets.filter((a) => a.kind === "source_image").map((a) => a.id),
    ...document.cameras.map((c) => c.imageId),
  ])].map((id) => ({ imageId: id, cameraId: cameraForImage(document, id)?.id || null }));
  const hasPointCloud = document.entities.some((e) => (e.representations || []).some((r) => r.kind === "point_cloud"));
  const hasRepresentation = document.entities.some((e) => layer === "model"
    ? !!entityGeometryForLayer(e, geometryOptions) : (e.representations || []).some(r => r.kind === layer));
  const hasCandidates = layer === "model" && document.entities.some((e) => (e.representations || []).some((r) =>
    r.id === e.activeModelRepresentationId && r.placementState === "unconfirmed" && ["imported_proposal", "requires_alignment_confirmation"].includes(r.placementReason || "")));
  const selectedOverlay = selected && camera ? photoOverlay(document, selected, camera, layer, selection.observationId) : null;
  const objects = document.entities.filter((entity) => !entity.sourceContext);
  const excludedObjects = document.entities.filter(entity => entity.sourceContext && jsonObject(entity.workcellScopeDecision)?.included === false);
  const visibleObjects = objects.filter(entity => entity.visible !== false);
  const modelObjects = visibleObjects.filter(entity => activeModel(entity) && entityGeometryForLayer(entity, { ...geometryOptions, layer: "model" }));
  const referenceSurfaces = visibleObjects.filter(entity => entityGeometryForLayer(entity, { ...geometryOptions, layer: "model" })?.geometryKind === "observed");
  const compositePreviews = visibleObjects.filter(entity => !modelObjects.includes(entity) && !referenceSurfaces.includes(entity) &&
    compositeModelEvidence(document, entity.id) && modelPreviewGeometry(document, entity.id, { ...geometryOptions, layer: "model" }));
  const missingModels = visibleObjects.length - modelObjects.length - referenceSurfaces.length - compositePreviews.length;
  const modelCandidates = modelObjects.filter(entity => activeModel(entity)?.placementState !== "confirmed").length;
  const modelFrameMismatch = (entity: Entity) => {
    const rep = activeModel(entity);
    return !!rep && rep.sourceValidity !== "stale" && (rep.coordinateFrameId !== geometryOptions.frameId ||
      (entity.currentModelTransform || rep.transform).coordinateFrameId !== geometryOptions.frameId);
  };
  const modelStates = modelLoads?.revisionId === revision.id ? modelLoads.states : [];
  const modelState = (entity: Entity) => {
    const rep = activeModel(entity);
    return modelStates.find(state => state.entityId === entity.id && state.representationId === rep?.id && state.assetId === (rep?.assetId || null));
  };
  const readyModels = modelObjects.filter(entity => modelState(entity)?.state === "ready"),
    failedModels = modelObjects.filter(entity => modelState(entity)?.state === "error"),
    pendingModels = modelObjects.length - readyModels.length - failedModels.length;
  const selectedComposite = selected && layer === "model" ? compositeModelEvidence(document, selected.id) : null;
  const selectedFamily = selected ? modelPreviewEntities(document, selected.id) : [];
  const selectedModels = selectedFamily.filter(entity => activeModel(entity) && entityGeometryForLayer(entity, { ...geometryOptions, layer: "model" }));
  const selectedModel = selected && modelPreviewGeometry(document, selected.id, { ...geometryOptions, layer: "model" });
  const selectedSource = selected && layer !== "model" ? entityGeometryForLayer(selected, { ...geometryOptions, observationId: selection.observationId }) : null;
  const previewGeometry = layer === "model" ? selectedModel : selectedSource?.representationIds.length && selectedSource.geometryKind === (layer === "observed_surface" ? "observed" : "point_cloud") ? selectedSource : null;
  const selectedReference = layer === "model" && previewGeometry?.geometryKind === "observed";
  const sourceReps = selected?.representations?.filter(rep => previewGeometry?.representationIds.includes(rep.id)) || [];
  const selectedModelFailed = layer === "model" && !selectedReference ? selectedModels.some(entity => modelState(entity)?.state === "error") : sourceReps.some(rep => modelStates.some(state => state.entityId === selected?.id && state.representationId === rep.id && state.assetId === rep.assetId && state.state === "error"));
  const selectedModelWrongFrame = layer === "model" && selectedFamily.some(modelFrameMismatch);
  const selectedModelStale = layer === "model" && selectedFamily.some(entity => activeModel(entity)?.sourceValidity === "stale");
  const previewKey = JSON.stringify([revision.id, selected?.id, layer, imageId, selection.observationId, layer === "model" ? selected && modelPreviewSignature(document, selected.id) : sourceReps.map(rep => [rep, document.assets.find(asset => asset.id === rep.assetId)]), geometryOptions.frameId, previewMode, measurement]);
  currentPreviewKey.current = previewKey;
  const previewRequest = previewMode !== "free" && selected && previewGeometry ? { entityId: selected.id, frameId: geometryOptions.frameId, layer, imageId, observationId: selection.observationId, mode: previewMode, requestKey: previewKey } : undefined;
  const previewImage = modelPreview?.key === previewKey ? modelPreview.image : null;
  const viewNames = { ...paneNames, plan: selectedComposite ? "sceneRelatedModels" : selectedReference ? "sceneSelectedSurface" : layer === "model" ? paneNames.plan : layer === "observed_surface" ? "sceneSelectedSurface" : "sceneSelectedPoints" };
  const counts = identityCounts(document);
  const objectNumbers = new Map(objects.map((entity, index) => [entity.id, String(index + 1).padStart(2, "0")]));
  const query = search.trim().toLowerCase();
  const numberedObject = objects.find((entity) => objectNumbers.get(entity.id) === query || String(Number(objectNumbers.get(entity.id))) === query);
  const filtered = numberedObject ? [numberedObject] : objects.filter((e) => `${e.label || ""} ${e.id}`.toLowerCase().includes(query));
  const children = new Map<string, Entity[]>();
  for (const entity of objects) if (entity.parentEntityId) children.set(entity.parentEntityId, [...children.get(entity.parentEntityId) || [], entity]);
  const objectRows: { entity: Entity; depth: number }[] = [], listed = new Set<string>();
  function appendRows(entity: Entity, depth: number) {
    if (listed.has(entity.id)) return;
    listed.add(entity.id); objectRows.push({ entity, depth });
    if (expandedEntities.has(entity.id)) for (const child of children.get(entity.id) || []) appendRows(child, depth + 1);
  }
  if (query) filtered.forEach(entity => objectRows.push({ entity, depth: 0 }));
  else objects.filter(entity => !entity.parentEntityId || !objects.some(parent => parent.id === entity.parentEntityId)).forEach(entity => appendRows(entity, 0));
  const photoIds = new Map(objects.map((entity) => [entity.id, new Set(observationsFor(document, entity).map((o) => o.imageId))]));
  const reportObjects = jsonObject(document.reportEvidence)?.objects;
  if (Array.isArray(reportObjects)) for (const item of reportObjects) {
    const record = jsonObject(item), views = record?.views;
    if (typeof record?.entityId !== "string" || !Array.isArray(views)) continue;
    for (const view of views) {
      const source = jsonObject(view);
      if (typeof source?.imageId === "string" && images.some((image) => image.imageId === source.imageId))
        photoIds.get(record.entityId)?.add(source.imageId);
    }
  }
  useEffect(() => { if (layer === "point_cloud" && !hasPointCloud) setLayer("model"); }, [hasPointCloud, layer]);
  useEffect(() => { if (draw) { setFocused("photo"); setMobileSection("views"); } }, [draw]);
  useEffect(() => {
    if (!objectListRequest) return;
    setSearch(""); setMobileSection("objects");
    const frame = window.requestAnimationFrame(() => objectSearch.current?.focus({ preventScroll: true }));
    return () => window.cancelAnimationFrame(frame);
  }, [objectListRequest]);
  useEffect(() => {
    const changed = () => {
      const active = window.document.fullscreenElement === container.current;
      setIsFullscreen(active);
      if (!active) setFocused(null);
    };
    window.document.addEventListener("fullscreenchange", changed);
    return () => window.document.removeEventListener("fullscreenchange", changed);
  }, []);
  useEffect(() => {
    const ancestors = new Set<string>();let parentId = selected?.parentEntityId;
    while (parentId && !ancestors.has(parentId)) { ancestors.add(parentId);parentId = document.entities.find(entity => entity.id === parentId)?.parentEntityId; }
    setExpandedEntities(current => [...ancestors].every(id => current.has(id)) ? current : new Set([...current, ...ancestors]));
  }, [selection.entityId, document]);
  useEffect(() => {
    const list = objectList.current, active = list?.querySelector<HTMLElement>('[aria-pressed="true"]');
    if (!list || !active) return;
    const row = active.getBoundingClientRect(), viewport = list.getBoundingClientRect();
    // Scroll only the object rail; selecting a scene object must not move the report.
    if (row.top < viewport.top) list.scrollTop -= viewport.top - row.top;
    else if (row.bottom > viewport.bottom) list.scrollTop += row.bottom - viewport.bottom;
  }, [selection.entityId, search, isFullscreen, mobileSection, expandedEntities]);
  function moveComparison(event: PointerEvent<HTMLDivElement>) {
    const bounds = event.currentTarget.parentElement!.getBoundingClientRect();
    setWipe(Math.round(Math.max(0, Math.min(100, (event.clientX - bounds.left) / bounds.width * 100))));
  }
  function chooseView(pane: Pane | null) { setComparing(false); setFocused(pane); setMobileSection("views"); }
  async function fullscreen() {
    setFullscreenError(false);
    try {
      if (window.document.fullscreenElement === container.current) await window.document.exitFullscreen();
      else if (container.current?.requestFullscreen) await container.current.requestFullscreen();
      else throw new Error("fullscreen_unavailable");
    } catch { setFullscreenError(true); }
  }
  async function openSourceCad() {
    setFullscreenError(false);
    try {
      if (window.document.fullscreenElement === container.current) await window.document.exitFullscreen();
      onOpenSourceCad?.();
    } catch { setFullscreenError(true); }
  }
  function selectEntity(entityId: string, observationId?: string) {
    setSurfaceKey("");
    onSelect(entityId, observationId);
    setMobileSection("views");
    const entity = document.entities.find((e) => e.id === entityId);
    if (!entity) return;
    const observations = observationsFor(document, entity), observation = observations.find((o) => o.id === observationId) ||
      observations.find((o) => o.imageId === imageId) || observations[0];
    if (observation && observation.imageId !== imageId)
      onCamera(observation.imageId, cameraForImage(document, observation.imageId)?.id || null);
  }
  function selectPlanEntity(entityId: string) {
    const entity = document.entities.find(entity => entity.id === entityId);
    const reference = entity && cadReferenceImage(document, entity);
    const observation = entity && observationsFor(document, entity).find(observation => observation.imageId === reference);
    selectEntity(entityId, observation?.id);
  }
  return (
    <section ref={container} className="report-scene" data-view={comparing ? "compare" : focused || "quad"}
      data-mobile-section={mobileSection} data-mobile-pane={focused || "photo"}
      onKeyDown={(event) => {
        if (event.key === "Escape" && !event.defaultPrevented) { setFocused(null); setFullscreenError(false); }
      }}>
      <header className="report-scene-toolbar">
        <div className="report-scene-intro"><strong>{t("sceneWorkspace")}</strong><span>{t("sceneLinked")}</span></div>
        <div className="report-scene-controls">
          <label><span>{t("sceneLayers")}</span><select aria-label={t("sceneLayers")} disabled={comparing} value={layer} onChange={(e) => setLayer(e.target.value as Layer)}>
            <option value="observed_surface">{t("sceneObserved")}</option><option value="model">{t("sceneModel")}</option>
            <option value="point_cloud" disabled={!hasPointCloud}>{t(hasPointCloud ? "scenePoints" : "sceneNoPoints")}</option>
          </select></label>
          {hasSplats && <label className="report-scene-check"><input type="checkbox" checked={splats} onChange={(e) => setSplats(e.target.checked)} />{t("sceneSplats")}</label>}
          {layer === "model" && hasPointCloud && <label className="report-scene-check"><input type="checkbox" checked={cloudWithModels} onChange={(e) => setCloudWithModels(e.target.checked)} />{language === "zh" ? "模型叠加点云" : "Models with point cloud"}</label>}
          {hasMotion && <label><span>{language === "zh" ? "动静" : "Motion"}</span><select aria-label={language === "zh" ? "静态与动态" : "Static and dynamic"} value={part} onChange={(e) => setPart(e.target.value as typeof part)}>
            <option value="all">{language === "zh" ? "静态+动态" : "Static + dynamic"}</option><option value="static">{language === "zh" ? "只看静态" : "Static only"}</option><option value="dynamic">{language === "zh" ? "只看动态" : "Dynamic only"}</option>
          </select></label>}
          {hasSkeleton && part !== "static" && <label className="report-scene-check"><input type="checkbox" checked={skeleton} onChange={(e) => setSkeleton(e.target.checked)} />{language === "zh" ? "骨架（代替人物表面）" : "Skeleton (instead of surfaces)"}</label>}
          <label className="report-scene-check"><input type="checkbox" checked={allBounds} onChange={(e) => setAllBounds(e.target.checked)} />{t("sceneShowBorders")}</label>
          <label className="report-scene-check"><input type="checkbox" checked={showPath} onChange={(e) => setShowPath(e.target.checked)} />{t("sceneShowCameraPath")}</label>
          {selected && onClearSelection && <button onClick={onClearSelection}>{t("sceneClearSelection")}</button>}
          <button className="report-scene-fullscreen" onClick={fullscreen} aria-pressed={isFullscreen} aria-label={t(isFullscreen ? "sceneExitFullscreen" : "sceneFullscreen")}>⛶ <span>{t(isFullscreen ? "sceneExitFullscreen" : "sceneFullscreen")}</span></button>
        </div>
      </header>
      {(analysisAvailable || bendAnalysis) && <div className="report-bend-analysis" aria-label={language === "zh" ? "已保存的折弯分析" : "Saved bend analysis"}>
        <span>{language === "zh" ? bendError ? "折弯分析加载失败，请刷新重试" : !bendAnalysis ? "读取已保存的折弯分析…" : `已计算 ${detectedBends.length} 处折弯角度${incompleteBends ? ` · ${incompleteBends} 个对象计算未完成` : ""}` : bendError ? "Could not load bend analysis; refresh to retry" : !bendAnalysis ? "Loading saved bend analysis…" : `${detectedBends.length} bend angles calculated${incompleteBends ? ` · ${incompleteBends} objects incomplete` : ""}`}</span>
        <label>{language === "zh" ? "已识别折弯" : "Detected bend"}<select value={detectedBends.some(row => row.entityId === selected?.id) ? selected!.id : ""} onChange={e => { if(e.target.value) { selectEntity(e.target.value); setLayer("model"); if (!analysisAvailable) { setMeasurement(detectedBends.find(row => row.entityId === e.target.value)?.result || null); setShowAngles(true); chooseView("plan"); } } }}>
          <option value="">{language === "zh" ? "选择有折弯结果的对象" : "Choose an object with a detected bend"}</option>
          {detectedBends.map(row => <option key={row.entityId} value={row.entityId}>{document.entities.find(entity => entity.id === row.entityId)?.label || row.entityId} · {row.result!.value.toFixed(1)}°</option>)}
        </select></label>
        <label className="report-scene-check"><input type="checkbox" checked={showAngles} onChange={e => setShowAngles(e.target.checked)} />{language === "zh" ? "显示角度标注" : "Show angle annotations"}</label>
      </div>}
      {(analysisAvailable || inclinationAnalysis) && <div className="report-bend-analysis" aria-label={language === "zh" ? "已保存的平面倾角" : "Saved plane inclinations"}>
        <span>{inclinationError ? (language === "zh" ? "平面分析加载失败，请刷新重试" : "Plane analysis could not load") : !inclinationAnalysis ? (language === "zh" ? "读取已保存的平面倾角…" : "Loading saved planes…") : language === "zh" ? `已计算 ${allSurfaces.length} 个平面倾角（${inclinationRows.filter(row=>row.surfaces.length>0).length} 个对象）${incompleteInclinations ? ` · ${incompleteInclinations} 个对象计算未完成` : ""}` : `${allSurfaces.length} plane inclinations calculated (${inclinationRows.filter(row=>row.surfaces.length>0).length} objects)${incompleteInclinations ? ` · ${incompleteInclinations} objects incomplete` : ""}`}</span>
        <label>{language === "zh" ? "倾斜平面（含待确认估计）" : "Inclined surface (includes unverified estimates)"}<select value={surfaceKey} onChange={e=>{const row=allSurfaces.find(r=>r.key===e.target.value);if(row){selectEntity(row.entityId);setSurfaceKey(row.key);setLayer("model");setPreviewMode("free");if(!analysisAvailable){setShowAngles(true);chooseView("spatial");}}else{setSurfaceKey("");setMeasurement(null);}}}>
          <option value="">{language === "zh" ? "选择局部面与地面倾角" : "Select a local surface"}</option>
          {visibleSurfaces.map(row=><option key={row.key} value={row.key}>{document.entities.find(e=>e.id===row.entityId)?.label || row.entityId} · {language === "zh" ? "面" : "surface"} {row.surface.surfaceId} · {row.surface.inclinationDeg.toFixed(1)}°{row.surface.classification==="direction_unverified" ? (language === "zh" ? "（估计）" : " (estimate)") : ""}</option>)}
        </select></label>
        <label className="report-scene-check"><input type="checkbox" checked={allPlanes} onChange={e=>{setAllPlanes(e.target.checked);setSurfaceKey("");setMeasurement(null);}} />{language === "zh" ? "全部已测平面" : "All measured planes"}</label>
        <span>{language === "zh" ? "默认显示非竖直平面估计。与地面倾角：水平 0°，竖直 90°；偏离竖直 = 90° − 倾角。" : "Showing estimated nonvertical planes. Ground inclination: horizontal 0°, vertical 90°; deviation from vertical = 90° − inclination."}</span>
      </div>}
      {newerReport && <div className="report-scene-history-notice" role="status"><span>{t("sceneHistoricalReport")}</span><a href={newerReport.href} title={newerReport.title}>{t("sceneLatestReport")} ↗</a></div>}
      {fullscreenError && <p className="report-scene-notice" role="status">{t("sceneFullscreenUnavailable")}</p>}
      <nav className="report-scene-section-tabs" aria-label={t("sceneWorkspace")}>
        {(["objects", "views", "inspector"] as const).map((section) => <button key={section}
          aria-pressed={mobileSection === section} aria-controls={`${panePrefix}-${section}`} onClick={() => setMobileSection(section)}>
          {t(section === "objects" ? "sceneObjects" : section === "views" ? "sceneViews" : "sceneInspector")}
          {section === "objects" && <span>{objects.length}</span>}
        </button>)}
      </nav>
      <div className="report-scene-shell">
        <aside className="report-scene-object-rail" id={`${panePrefix}-objects`} aria-label={t("sceneObjects")}>
          <header><h3>{t("identityRecords")}</h3><span className="report-scene-count">{objects.length}</span></header>
          <div className="report-scene-object-search"><input ref={objectSearch} type="search" aria-label={t("sceneSearch")} placeholder={t("sceneSearch")} value={search} onChange={(e) => setSearch(e.target.value)} /></div>
          <div className="report-scene-current-object" aria-live="polite">
            <span>{t("sceneSelected")}</span><strong>{selected?.label || selected?.id || t("sceneNoSelection")}</strong>
            {selected && <small>{selected.id.slice(0, 8)}</small>}
          </div>
          <div ref={objectList} className="report-scene-object-list">
            {objectRows.map(({ entity, depth }) => {
              const indices = images.flatMap((image, index) => photoIds.get(entity.id)?.has(image.imageId) ? [index + 1] : []);
              const evidence = entityEvidenceStatus(document, entity), observations = observationsFor(document, entity),
                representations = entity.representations || [],
                candidate = representations.some((representation) => representation.id === entity.activeModelRepresentationId && representation.sourceValidity !== "stale" && representation.placementState === "unconfirmed"),
                composite = compositeModelEvidence(document, entity.id),
                family = modelPreviewEntities(document, entity.id),
                familyModels = family.filter(member => activeModel(member) && entityGeometryForLayer(member, { ...geometryOptions, layer: "model" })),
                modelStatus = referenceSurfaces.includes(entity) ? "observed_reference_surface" : family.some(member => activeModel(member)?.sourceValidity === "stale") ? "identityModelStale" : family.some(modelFrameMismatch) ? "sceneModelWrongFrame" : !familyModels.length ? "reportMissingGeometry" : familyModels.some(member => modelState(member)?.state === "error") ? "sceneModelLoadFailed" : familyModels.every(member => modelState(member)?.state === "ready") ? composite ? "sceneCompositeEvidence" : "sceneModelLoaded" : "sceneModelLoading";
              return <div key={entity.id} className="report-scene-object-row" data-entity-id={entity.id} data-parent-entity-id={entity.parentEntityId || undefined} style={{ marginLeft: depth * 12 }}><button key={entity.id} aria-pressed={entity.id === selection.entityId} onClick={() => selectEntity(entity.id)}>
                <strong><b className="report-scene-object-number">{objectNumbers.get(entity.id)}</b>{entity.label || entity.id}</strong>
                <span className="report-scene-object-source">{indices.length ? `${t("scenePhotoNumber")} ${indices.join(" / ")}` : t("sceneNoPhotoLink")}<small>{entity.id.slice(0, 8)}</small></span>
                <span className="report-scene-object-evidence">{t(evidence.photoKey)}{observations.length > 0 && ` · ${observations.length} ${t("observations")}`}</span>
                {evidence.identityKey && <span className="report-scene-object-identity"><span>{t("entityIdentity")}</span>{t(evidence.identityKey)}</span>}
                <span className="report-scene-object-model" data-model-state={modelStatus}><span>{t("model")}</span>{t(modelStatus)}{composite && modelStatus !== "sceneCompositeEvidence" && <> · {t("sceneCompositeEvidence")}</>}{candidate && <em>{t("sceneCandidate")}</em>}</span>
                <span className="report-scene-object-extent"><span>{t("reportObservedExtent")}</span><Extent entity={entity} document={document} /></span>
              </button>{children.has(entity.id) && <button aria-expanded={expandedEntities.has(entity.id)} aria-label={`${t("sceneModelParts")} · ${entity.label || entity.id}`} onClick={() => setExpandedEntities(current => { const next = new Set(current);if (next.has(entity.id)) next.delete(entity.id);else next.add(entity.id);return next; })}>{expandedEntities.has(entity.id) ? "▾" : "▸"} {children.get(entity.id)!.length} {t("sceneModelParts")}</button>}{onFeedback && <button className="report-object-feedback" aria-label={`${t("sceneFeedback")} · ${entity.label || entity.id}`} onClick={() => { selectEntity(entity.id); onFeedback(entity.id); setMobileSection("inspector"); }}>{t("sceneFeedback")} ↗</button>}</div>;
            })}
            {!filtered.length && <p className="report-scene-list-empty">{t(objects.length ? "sceneNoMatches" : "sceneNoObjects")}</p>}
          </div>
          <footer><span>{filtered.length} / {counts.records} · {t("identityRecords")}</span><small>{counts.linkedGroups} {t("identityGroups")} · {counts.pending} {t("identityPending")} · {counts.observations} {t("observations")}</small>
            <details><summary>{t("sceneRecordNote")}</summary><p>{t("reportAssociationHint")}</p></details>
          </footer>
        </aside>
        <div className="report-scene-center" id={`${panePrefix}-views`}>
          {matchedComparison && <nav className="report-scene-primary-views" aria-label="照片、点云与模型">
            <button aria-pressed={!comparing && focused === "photo"} onClick={() => chooseView("photo")}><strong>原始照片</strong><small>点击物体查看对应证据</small></button>
            <button aria-pressed={comparing} onClick={() => { setComparing(true); setLayer("model"); setMobileSection("views"); }}><strong>照片 / 模型对比</strong><small>拖动分界线核对 · 相机固定</small></button>
            <button disabled={!hasPointCloud} aria-pressed={!comparing && focused === "spatial" && layer === "point_cloud"} onClick={() => { setLayer("point_cloud"); chooseView("spatial"); }}><strong>原始点云</strong><small>{hasPointCloud ? "所选照片深度点 · 可旋转" : "本报告未附原始点云"}</small></button>
            <button aria-pressed={!comparing && focused === "spatial" && layer === "model"} onClick={() => { setLayer("model"); chooseView("spatial"); }}><strong>可旋转 3D 模型</strong><small>拖动旋转 · 滚轮缩放</small></button>
          </nav>}
          {matchedComparison && !comparing && focused === "spatial" && <nav className="report-scene-photo-switch" aria-label="三维证据来源照片">{images.map((image, i) => <button key={image.imageId} aria-pressed={imageId === image.imageId} onClick={() => onCamera(image.imageId, image.cameraId)}>照片 {i + 1}</button>)}</nav>}
          {matchedComparison && !comparing && focused === "spatial" && layer === "point_cloud" && <p className="report-scene-notice" data-point-cloud-image-id={imageId}>当前只显示照片 {images.findIndex(image => image.imageId === imageId) + 1} 的深度估计点，没有补成实体表面。拖动旋转、滚轮缩放；可从左侧列表或点云选择物体。未通过标尺验证的距离仍不能读作厘米。</p>}
          <nav className="report-scene-view-switch" aria-label={t("sceneViews")}>
            <button className="report-scene-quad" aria-pressed={!focused && !comparing} onClick={() => chooseView(null)}>{t("sceneQuad")}</button>
            {paneOrder.map((pane) => <button key={pane} aria-pressed={!comparing && focused === pane} aria-controls={`${panePrefix}-${pane}`} onClick={() => chooseView(pane)}>{t(viewNames[pane])}</button>)}
          </nav>
          <nav className="report-scene-view-switch report-scene-mobile-views" aria-label={t("sceneViews")}>
            {paneOrder.map((pane) => <button key={pane} aria-pressed={(focused || "photo") === pane} aria-controls={`${panePrefix}-${pane}`} onClick={() => chooseView(pane)}>{t(viewNames[pane])}</button>)}
          </nav>
          {layer === "model" && <p className="report-scene-notice" data-geometry-coverage={modelObjects.length + referenceSurfaces.length + compositePreviews.length} data-model-coverage={modelObjects.length} data-model-loaded={readyModels.length} data-reference-surfaces={referenceSurfaces.length} data-composite-previews={compositePreviews.length} data-missing-models={missingModels}><strong>{modelObjects.length + referenceSurfaces.length + compositePreviews.length} / {visibleObjects.length} {t("sceneGeometryCoverage")}</strong> · {modelObjects.length} {t("sceneIndependentModelCount")}{referenceSurfaces.length > 0 && <> · {referenceSurfaces.length} {t("sceneReferenceSurfaceCount")}</>}{compositePreviews.length > 0 && <> · {compositePreviews.length} {t("sceneCompositePreviewCount")}</>} · {missingModels} {t("sceneMissingModelCount")} · {readyModels.length} / {modelObjects.length} {t("sceneModelLoaded")}{pendingModels > 0 && <> · {pendingModels} {t("sceneModelLoading")}</>}{failedModels.length > 0 && <> · {failedModels.length} {t("sceneModelLoadFailed")}</>}{modelCandidates > 0 && <> · {modelCandidates} {t("sceneModelCandidateCount")}</>} · {t("sceneModelCoverageMeaning")}{excludedObjects.length > 0 && <> · {t("sceneExcludedObjects")}: {excludedObjects.length}</>}</p>}
          {hasCandidates && <p className="report-scene-notice">{t("sceneCandidateNotice")}</p>}
          {comparing && <div className="report-matched-comparison">
            <div className="report-comparison-stage" data-camera-id={camera?.id}>
              <PhotoView document={document} imageId={imageId} selectedId={selection.entityId} onSelect={selectEntity} showBounds={allBounds} />
              <div className="report-comparison-model" style={{ clipPath: `inset(0 0 0 ${wipe}%)` }}>
                <SpatialView revision={revision} selection={selection} onSelect={selectEntity} onCommit={noEdit} mode="photo" cameraId={camera?.id || null} showSourcePhoto={false}
                  onAssetStates={(revisionId, states) => setModelLoads({ revisionId, states })}
                  layers={{ measurement, modelOnly: true, generated_mesh: true, primitive: true, point_cloud: cloudWithModels && hasPointCloud, showCandidates: true, editable: false, opacity: 1, imageId, observations: document.observations, showBounds: allBounds, cameraPath: false }} />
              </div>
              <div className="report-comparison-divider" style={{ left: `${wipe}%` }} role="slider" tabIndex={0} aria-label="拖动照片模型分界线" aria-valuemin={0} aria-valuemax={100} aria-valuenow={wipe}
                onPointerDown={event => { event.preventDefault(); event.currentTarget.focus({ preventScroll: true }); event.currentTarget.setPointerCapture(event.pointerId); moveComparison(event); }}
                onPointerMove={event => { if (event.currentTarget.hasPointerCapture(event.pointerId)) moveComparison(event); }}
                onPointerUp={event => { if (event.currentTarget.hasPointerCapture(event.pointerId)) { moveComparison(event); event.currentTarget.releasePointerCapture(event.pointerId); } }}
                onPointerCancel={event => { if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}
                onKeyDown={event => { const key = event.key; if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(key)) { event.preventDefault(); setWipe(value => key === "Home" ? 0 : key === "End" ? 100 : Math.max(0, Math.min(100, value + (["ArrowRight", "ArrowUp"].includes(key) ? 1 : -1)))); } }}><b aria-hidden="true">↔</b></div>
              <span className="report-comparison-photo-label">原始照片</span><span className="report-comparison-model-label">同相机模型</span>
            </div>
            <label className="report-comparison-control"><span>模型</span><input aria-label="照片与模型对比位置" type="range" min="0" max="100" value={wipe} onChange={event => setWipe(Number(event.target.value))} /><span>照片</span><output>{wipe}%</output></label>
            <footer className="report-scene-photo-switch">{images.map((image, i) => <button key={image.imageId} aria-pressed={imageId === image.imageId} onClick={() => onCamera(image.imageId, image.cameraId)}>照片 {i + 1}</button>)}</footer>
          </div>}
          {!comparing && <div className="report-scene-grid">
            {paneOrder.map((pane, index) => (
              <section className="report-scene-pane" id={`${panePrefix}-${pane}`} data-pane={pane} key={pane} aria-label={t(viewNames[pane])}>
                <header><h3><span>{String(index + 1).padStart(2, "0")}</span>{pane === "photo" && showVideo ? (language === "zh" ? "来源视频 · 每帧可点选对象" : "Source video · pick objects in any frame") : t(pane === "spatial" ? availability.spatialTitle : viewNames[pane])}</h3>
                  {pane === "cad" && <select className="report-scene-cad-layer" aria-label={t("sceneCadProjectionSource")} value={cadLayer} onChange={event => { setDrawingRegion(false); setCadLayer(event.target.value as "model" | "observed_surface"); }}>
                    <option value="observed_surface">{t("sceneCadObservedProjection")}</option><option value="model">{t("sceneCadModelProjection")}</option>
                  </select>}
                  {pane === "cad" && onOpenSourceCad && <button className="report-scene-source-cad" onClick={openSourceCad}>{t("sceneSourceCad")} ↗</button>}
                  <button className="report-scene-expand" aria-label={`${t(focused === pane ? "sceneQuad" : "sceneSingleView")} · ${t(viewNames[pane])}`}
                    title={t(focused === pane ? "sceneQuad" : "sceneSingleView")} onClick={() => chooseView(focused === pane ? null : pane)}>{focused === pane ? "⊞" : "↗"}</button>
                </header>
                <div className="report-scene-pane-body">
                  {pane === "photo" && showVideo && <VideoView document={document} selectedId={selection.entityId} onSelect={selectEntity} />}
                  {pane === "photo" && !showVideo && <><PhotoView document={document} imageId={imageId} selectedId={selection.entityId} onSelect={selectEntity} draw={draw} onBox={onBox} showBounds={allBounds} />
                    {camera && allBounds && <PhotoAxes revision={revision} camera={camera} layer={layer} selectedId={selection.entityId} observationId={selection.observationId} allBounds={allBounds} />}</>}
                  {pane === "spatial" && <>{pickingPoints && <div className="cad-measure-guide" role="status">{language === "zh" ? `第 ${measurePoints.length+1}/${pointCount} 点：${(pointCount === 2 ? ["斜边上端点（角的顶点）","同一斜边的下端点"] : ["第一条边上的点","两条边相交的顶点","第二条边上的点"])[measurePoints.length]}。点击模型表面，拖动可旋转。` : `Point ${measurePoints.length+1}/${pointCount}: ${(pointCount === 2 ? ["upper edge endpoint (vertex)","lower endpoint of the same edge"] : ["first edge","shared vertex","second edge"])[measurePoints.length]}. Click the model surface; drag to rotate.`}{pointError && <strong>{language === "zh" ? " 未点到模型表面，请重选。" : " No model surface hit. Try again."}</strong>} <button onClick={()=>startPointPicking(false)}>{language === "zh" ? "取消取点" : "Cancel picking"}</button></div>}<SpatialView revision={revision} selection={selection} onSelect={selectEntity} onCommit={noEdit} mode="free" cameraId={camera?.id || null}
                    modelPreview={previewRequest} onModelPreview={(key, image) => { if (key === currentPreviewKey.current) setModelPreview({ key, image }); }}
                    onAssetStates={(revisionId, states) => setModelLoads({ revisionId, states })} onMeasurementPoint={pickMeasurementPoint}
                    layers={{ groundDatum: matchedComparison, pickingPoints, measurePoints, measureVertexIndex: pointCount === 2 ? 0 : 1, measurement: layer === "model" ? measurement : null, modelOnly: layer === "model", observed_surface: layer === "observed_surface", generated_mesh: layer === "model", primitive: layer === "model", point_cloud: layer === "point_cloud" || layer === "model" && cloudWithModels && hasPointCloud, allBounds, showBounds: allBounds, cameraPath: showPath, showCandidates: true, editable: false, opacity: 1, imageId, observationEntityId: selected?.id, observationId: selection.observationId, observations: document.observations, part, skeleton, splats: hasSplats && splats }} />
                    {!hasRepresentation && <div className="report-scene-stage-note">{t("sceneNoRepresentation")}</div>}</>}
                  {pane === "cad" && availability.planEmpty && <div className="report-scene-plan-empty" role="status"><strong>{t("scenePlanUnavailable")}</strong><p>{t(availability.planEmpty)}</p><small>{t("sceneSelectionRetained")}</small></div>}
                  {pane === "cad" && !availability.planEmpty && <CadView key={revision.id + cadLayer} document={document} selectedId={selection.entityId} onSelect={selectPlanEntity} geometryOptions={planOptions} measurement={cadLayer === "model" ? measurement : null} region={cadLayer === "model" ? measureRegion : null} drawingRegion={drawingRegion} onRegion={region => { setMeasureRegion(region); setDrawingRegion(false); setMobileSection("inspector"); }} showPath={showPath} />}
                  {pane === "plan" && <div className="report-model-preview" data-model-entity={selected?.id || ""} data-preview-layer={layer}>
                    {previewGeometry && selected ? <>
                      <nav aria-label={t(layer === "model" ? "sceneModelOrientation" : "sceneEvidenceOrientation")}>{(["free", "front", "side", "top"] as const).map(mode => <button key={mode} aria-pressed={previewMode === mode} onClick={() => setPreviewMode(mode)}>{t(mode)}</button>)}{previewMode === "free" && <small>{language === "zh" ? "拖动旋转 · Shift / 右键拖动平移 · 滚轮缩放" : "Drag to rotate · Shift / right-drag to pan · Scroll to zoom"}</small>}</nav>
                      <div className="report-model-image">{previewMode === "free" ? <SpatialView
                        key={`${revision.id}:${selected.id}:${layer}:${geometryOptions.frameId}:${selection.observationId || ""}`}
                        revision={revision} selection={{ ...selection, entityId: null }} onSelect={noEdit} onCommit={noEdit} mode="free" cameraId={camera?.id || null}
                        layers={{ studio: true, entityIds: layer === "model" ? selectedFamily.map(entity => entity.id) : [selected.id], representationIds: previewGeometry.representationIds,
                          axisEntityId: selectedComposite ? undefined : selected.id, showAxes: !selectedComposite, showBounds: false, editable: false, showCandidates: true,
                          modelOnly: layer === "model", generated_mesh: layer === "model", primitive: layer === "model", observed_surface: layer === "observed_surface", point_cloud: layer === "point_cloud",
                          observationEntityId: selected.id, observationId: selection.observationId, imageId, observations: document.observations,
                          measurement: layer === "model" && measurement?.references.every(ref => selectedFamily.some(entity => entity.id === ref.entityId)) ? measurement : null }} /> : previewImage ? <img src={previewImage} alt={`${selected.label || selected.id} · ${t(viewNames.plan)} · ${t(previewMode)}`} /> : <p role="status">{t(selectedModelStale ? "identityModelStale" : selectedModelWrongFrame ? "sceneModelWrongFrame" : selectedModelFailed ? layer === "model" ? "sceneModelLoadFailed" : "sceneEvidenceLoadFailed" : layer === "model" ? "loadingModel" : "sceneEvidenceLoading")}</p>}</div>
                      <footer><strong>{selected.label || selected.id}</strong><span>{selectedComposite ? <>{t("sceneCompositeEvidence")} · {selectedComposite.targets.map(entity => `${objectNumbers.get(entity.id)} ${entity.label || entity.id}`).join(" + ")} · {t("sceneCompositeEvidenceMeaning")}</> : selectedReference ? <>{t("observed_reference_surface")} · {t("sceneObservedCoverage")}</> : layer === "model" ? t(selectedModels.every(entity => activeModel(entity)?.placementState === "confirmed") ? "identityPlacementConfirmed" : "identityPlacementUnconfirmed") : <>{t(layer)} · {t("scenePhotoNumber")} {images.findIndex(image => image.imageId === imageId) + 1}{selection.observationId && <> · {t("sceneObservation")} {selection.observationId.slice(0, 8)}</>} · {t("sceneObservedCoverage")}</>}</span><button onClick={() => chooseView("spatial")}>{t(layer === "model" ? "sceneOpenModelScene" : "sceneOpenEvidenceScene")} ↗</button></footer>
                    </> : <div className="report-scene-plan-empty" role="status"><strong>{t(selectedModelStale ? "identityModelStale" : selectedModelWrongFrame ? "sceneModelWrongFrame" : selected ? layer === "model" ? "sceneObjectModelMissing" : "sceneEvidenceMissing" : layer === "model" ? "sceneSelectModel" : "sceneSelectEvidence")}</strong><p>{t(selectedModelWrongFrame ? "sceneModelWrongFrameMeaning" : selected ? layer === "model" ? "sceneObjectModelMissingMeaning" : "sceneEvidenceMissingMeaning" : "sceneSelectModelMeaning")}</p></div>}
                  </div>}
                </div>
                {pane === "photo" && hasVideo && <footer className="report-scene-photo-switch">
                  <button aria-pressed={showVideo} onClick={() => setVideoMode(true)}>{language === "zh" ? "视频（逐帧可点选）" : "Video (pick in any frame)"}</button>
                  <button aria-pressed={!showVideo} onClick={() => setVideoMode(false)}>{language === "zh" ? "关键帧证据" : "Keyframe evidence"}</button></footer>}
                {pane === "photo" && !showVideo && <footer className="report-scene-photo-switch">{images.map((image, i) => <button key={image.imageId} aria-pressed={imageId === image.imageId} onClick={() => onCamera(image.imageId, image.cameraId)}>{t("scenePhotoNumber")} {i + 1}</button>)}{draw && <span>{t("sceneDrawActive")}</span>}</footer>}
              </section>
            ))}
          </div>}
          <VideoMemory document={document} onSelect={selectEntity} />
          <ComparisonVideo document={document} />
          <p className="report-scene-selection-note"><span className="report-scene-reference-note">{t(cadLayer === "model" ? "sceneCadModelSource" : "sceneCadSource")} {cadLayer === "observed_surface" && <> · {selected ? <>{t("sceneCadReference")}: {referenceImageId ? `${t("scenePhotoNumber")} ${images.findIndex(image => image.imageId === referenceImageId) + 1}` : t("sceneCadReferenceMissing")} · {t("sceneViewedPhoto")}: {imageId ? images.findIndex(image => image.imageId === imageId) + 1 : "—"}</> : t("sceneCadFixedState")}</>}</span>{selected ? isReferenceSurface(document, selected) ? t("sceneReferenceSurface") : selectedOverlay ? t(selectedOverlay.axisSpace === "native" ? "sceneNativeAxis" : "sceneSourceAxis") : !entityGeometryForLayer(selected, geometryOptions) ? t("sceneNoGeometrySelection") : t("sceneNoPhotoAxes") : t("sceneReadOnly")}</p>
        </div>
        <aside className="report-scene-inspector" id={`${panePrefix}-inspector`} aria-label={t("sceneInspector")}>
          <header><h3>{t("sceneInspector")}</h3>{selected && <span>{selected.id.slice(0, 8)}</span>}</header>
          <div className="report-scene-inspector-content">{selected && (selected as any).motionSummary && <section className="report-motion-summary">
            <h4>{language === "zh" ? "怎么走动的" : "How it moved"}</h4><p>{(selected as any).motionSummary.text}</p>
            <small>{language === "zh" ? "依据：每个采样视图可见表面的中心投到地面；只用于回答“去了哪、多快、停在哪”，不是逐关节测量。" : "From the visible surface centre per sampled view, on the floor plane."}</small></section>}{analysisAvailable && selected && <SpatialMeasurements key={revision.id + selected.id} revision={revision} selectedId={selected.id} savedBend={savedBend} savedSurface={activeSurface} inclinationOutcome={inclinationRows.find(r=>r.entityId===selected.id)} onClearSurface={()=>setSurfaceKey("")} points={measurePoints} pickingPoints={pickingPoints} onPickPoints={startPointPicking} region={measureRegion} drawing={drawingRegion} onResult={setMeasurement} onDraw={() => { setDrawingRegion(!drawingRegion); if (!drawingRegion) { setCadLayer("model"); setFocused("cad"); setMobileSection("views"); } }} />}{(typeof inspector === "function" ? inspector(activeSurface) : inspector) ?? <p className="report-scene-inspector-empty">{t("sceneReadOnly")}</p>}</div>
        </aside>
      </div>
    </section>
  );
}
