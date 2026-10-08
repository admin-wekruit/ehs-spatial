import type { MeasurementScale, SurfacePick } from "./viewer/native-math";
import { useEffect, useId, useRef, useState, type ReactNode, type PointerEvent } from "react";
import { SpatialView } from "./App";
import { PhotoView } from "./PhotoView";
import { ComparisonVideo, VideoMemory, VideoView, videoReplay } from "./VideoView";
import { request } from "./api";
import { useSceneResources } from "./SceneResources";
import { SpatialMeasurements, type BendAnalysis, type InclinationAnalysis, type InclinationSurface, type SceneMeasurement, type MeasureRegion } from "./SpatialMeasurements";
import { CadView } from "./CadView";
import { splatAnnotation } from "./viewer/splat-layer";
import { useI18n, type Language } from "./i18n";
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
import { boxDimNames, boxFaceNames, boxGeometry, boxLevels, renderMessage, validBox, type BoxDimName, type BoxFaceName, type BoxLevel, type LayerBox, type MeasurementLayer } from "./measurement-layer";
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

// Box faces are hatched in their confidence colour (the weaker the evidence, the denser the hatch); a highlighted box (the layer's
// low-confidence flag: retake photos) is outlined in pink (faint when not selected), others in teal (selected) or grey.
const levelHatch: Record<BoxLevel, [string, number]> = { high: ["rgba(92,201,140,.22)", 2], medium: ["rgba(240,180,60,.4)", 4], low: ["rgba(245,130,47,.45)", 5], unverified: ["rgba(235,70,70,.5)", 7] };
const boxEdge = { highlight: "#ff3fa4", faintHighlight: "rgba(255,63,164,.45)", selected: "#14a38b", other: "#8aa39a", floor: "rgba(255,255,255,.85)" };
const levelNames: Record<BoxLevel, string> = { high: "ReportScenetsx.209", medium: "box.level.medium", low: "box.level.low", unverified: "box.level.unverified" };
const faceNames: Record<BoxFaceName, string> = { front: "box.face.front", back: "box.face.back", left: "box.face.left", right: "box.face.right", top: "box.face.top", bottom: "box.face.bottom" };
const dimNames: Record<BoxDimName, string> = { L: "box.dim.L", W: "box.dim.W", H: "ReportScenetsx.210", bottom: "box.dim.bottom" };
const statusNames: Record<string, string> = { seen: "box.status.seen", occluded: "box.status.occluded", out_of_frame: "box.status.out_of_frame", not_facing: "box.status.not_facing" };
const cm = (metres: number) => { const v = metres * 100; return Math.abs(v) < 100 ? v.toFixed(1) : v.toFixed(0); };

type BoxFrame = { nativeToMeters: number; ground?: { normal: number[]; offset: number } | null };
/** The viewer draws a line or label carrying `facing` only while that face turns towards its camera (no depth test in the overlay),
 * and leaves every `layerBox` one out of model-preview captures. */
type Facing = { at: number[]; normal: number[] };
type BoxLine = SceneMeasurement["lines"][number] & { layerBox: true; facing?: Facing };
type BoxLabel = { point: number[]; text: string; layerBox: true; facing?: Facing };
/** One layer box in the viewer's measurement overlay. The selected box: its outline, its faces hatched in their confidence colours
 * and named at their centres (front-facing ones only), L × W × H and clearance over the unified floor. Any other shown box: the outline. */
function boxOverlay(box: LayerBox, frame: BoxFrame, selected: boolean, title: string, t: (id: string) => string, out: { lines: BoxLine[]; labels: BoxLabel[] }) {
  const g = boxGeometry(box, frame.nativeToMeters, frame.ground);
  if (!g) return;
  const edge = box.highlight ? selected ? boxEdge.highlight : boxEdge.faintHighlight : selected ? boxEdge.selected : boxEdge.other;
  for (let i = 0; i < 8; i++) for (let k = 0; k < 3; k++) if (!(i & 1 << k)) out.lines.push({ points: [g.corners[i], g.corners[i | 1 << k]], color: edge, layerBox: true });
  if (!selected) return;
  // one label for size and clearance at the bottom (by the floor tick), so the top is left to the top face's name
  const lifted = !!g.floor && !box.floorContact;
  if (lifted) out.lines.push({ points: [g.floor!, g.bottomCenter], color: boxEdge.floor, layerBox: true });
  out.labels.push({ point: lifted ? g.floor!.map((v, k) => (v + g.bottomCenter[k]) / 2) : g.bottomCenter, layerBox: true,
    text: `${title ? `${title} · ` : ""}${t("ReportScenetsx.211")} ${[box.dims.L, box.dims.W, box.dims.H].map(d => cm(d.valueM)).join("×")} cm · ${box.floorContact ? (t("ReportScenetsx.212")) : `${t("ReportScenetsx.213")} ${cm(box.dims.bottom.valueM)} cm`}` });
  for (const name of boxFaceNames) {
    const level = box.faces[name]?.confidence ?? "unverified", [color, count] = levelHatch[level], [a, b, c, d] = g.faces[name].map(i => g.corners[i]);
    const facing = { at: [0, 1, 2].map(k => (a[k] + b[k] + c[k] + d[k]) / 4), normal: box.faceNormals[name] };
    const at = (x: number, y: number) => [0, 1, 2].map(k => a[k] + x * (b[k] - a[k]) + y * (d[k] - a[k]));
    for (let i = 1; i <= count; i++) { const t = 2 * i / (count + 1); out.lines.push({ points: t <= 1 ? [at(t, 0), at(0, t)] : [at(1, t - 1), at(t - 1, 1)], color, layerBox: true, facing }); }
    out.labels.push({ point: facing.at, text: `${t(faceNames[name])}·${t(levelNames[level])}`, layerBox: true, facing });
  }
}

/** The shown measurement with the layer boxes added to its overlay (or the boxes alone); unchanged when there is nothing to add.
 * Shown: the selected box in full, highlighted boxes outlined, and with showAll every box outlined: then the selected box's label
 * carries its name and the viewer names only the hovered object (labels 'hover'), so names never pile up. */
function withBoxes(measurement: SceneMeasurement | null, layer: MeasurementLayer | null, boxes: [string, LayerBox][], frame: BoxFrame | null, selected: Entity | null | undefined, showAll: boolean, language: Language, t: (id: string) => string): SceneMeasurement | null {
  if (!layer || !frame || !boxes.length || measurement && measurement.coordinateFrameId !== layer.coordinateFrameId) return measurement;
  const out = { lines: [] as BoxLine[], labels: [] as BoxLabel[] };
  for (const [id, box] of boxes) if (showAll || box.highlight || id === selected?.id) boxOverlay(box, frame, id === selected?.id, showAll ? id === selected?.id && selected.label || renderMessage(language, box.label) : "", t, out);
  if (!out.lines.length) return measurement;
  if (measurement) return { ...measurement, lines: [...measurement.lines, ...out.lines], labels: [...(measurement.labels || []), ...out.labels] };
  const [first, ...labels] = out.labels;  // the first is the size label: never a face's, so the viewer always shows it
  return { revisionId: layer.revisionId, kind: "layer_boxes", coordinateFrameId: layer.coordinateFrameId, source: "measurement_layer", value: 0, unit: "native", method: "layer-boxes",
    references: [], lines: out.lines, labelPoint: first?.point ?? out.lines[0].points[0], displayLabel: first?.text ?? "", labels, quality: {} };
}

/** Length / width (depth) / height / clearance with σ and confidence, why the object is highlighted, which photos see each face and what to retake. */
export function LayerBoxPanel({ box, entityId }: { box: LayerBox; entityId: string }) {
  const { language, t } = useI18n();
  const reasons = (box.highlightReasons || []).map(message => renderMessage(language, message)), snapNote = box.snapNote && renderMessage(language, box.snapNote);
  const needs = boxFaceNames.flatMap(name => { const face = box.faces[name], need = face?.need && renderMessage(language, face.need); return need ? [[name, need] as const] : []; });
  return <section className="report-box-panel" data-layer-box={entityId} data-confidence={box.confidence} data-highlight={box.highlight || undefined}>
    <h4>{t("ReportScenetsx.214")}<span data-confidence={box.confidence}>{t("ReportScenetsx.215")}{t(levelNames[box.confidence])}</span></h4>
    {box.highlight && <p className="report-box-recapture">{t("ReportScenetsx.216")}</p>}
    {reasons.length > 0 && <ul className="report-box-reasons">{reasons.map((reason, i) => <li key={i}>{reason}</li>)}</ul>}
    <table className="report-box-dims"><tbody>{boxDimNames.map(k => { const d = box.dims[k]; return <tr key={k} data-dim={k} data-confidence={d.confidence}>
      <th>{t(dimNames[k])}</th><td className="report-numeric">{cm(d.valueM)} ± {d.sigmaCm === null ? "—" : d.sigmaCm.toFixed(1)} cm{k === "bottom" && box.floorContact ? (t("ReportScenetsx.217")) : ""}</td><td>{t(levelNames[d.confidence])}</td></tr>; })}</tbody></table>
    {snapNote && <p className="report-box-snap">{snapNote}</p>}
    <small className="report-numeric">{t("ReportScenetsx.218")} {cm(box.topM)} cm</small>
    <table className="report-box-faces"><thead><tr><th>{t("ReportScenetsx.219")}</th><th>{t("photos")}</th><th>{t("ReportScenetsx.220")}</th><th>{t("ReportScenetsx.221")}</th></tr></thead>
      <tbody>{boxFaceNames.map(name => { const face = box.faces[name], level = face?.confidence ?? "unverified"; return <tr key={name} data-confidence={level}>
        <th>{t(faceNames[name])}</th><td className="report-numeric">{face?.photos.length ? face.photos.join(", ") : "—"}</td>
        <td>{face ? statusNames[face.status] ? t(statusNames[face.status]) : face.status : "—"}</td><td>{t(levelNames[level])}</td></tr>; })}</tbody></table>
    {needs.length > 0 && <div className="report-box-need"><strong>{t("ReportScenetsx.222")}</strong><ul>{needs.map(([name, need]) => <li key={name}><b>{t(faceNames[name])}</b> {need}</li>)}</ul></div>}
    {box.method && <small>{box.method}</small>}
  </section>;
}

export function Extent({ entity, document }: { entity: Entity; document: SceneDocument }) {
  const { t } = useI18n(), d = sourceDimensions(entity), scale = sourceScale(document, entity),
    groundDimensions = [d.widthNative, d.depthNative, d.groundHeight],
    values = isReferenceSurface(document, entity) ? [d.widthNative, d.depthNative] :
      groundDimensions.every(Number.isFinite) ? groundDimensions : [d.extentX, d.extentY, d.extentZ];
  if (entity.physicalDimensionsUnknown === true) return <span>{t("ReportScenetsx.text005")}</span>;
  return values.some(Number.isFinite) ? <span className="report-numeric">
    {values.map((value) => Number.isFinite(value) ? (value! * (scale?.nativeToMeters || 1)).toFixed(2) : "—").join(" × ")}
    <small>{scale?.nativeToMeters ? "m" : t("uncalibrated")}</small>
  </span> : <span>—</span>;
}

export function ReportScene({
  revision, selection, onSelect, imageId, cameraId, onCamera,
  draw = false, onBox, onOpenSourceCad, inspector, objectListRequest = 0, onFeedback, onClearSelection, newerReport, variantNotice, matchedComparison = false, measurementOverride, measurementScale, initialView, viewRequest, boxLayer = null,
}: {
  /** The report's measurement layer: its boxes are drawn in the 3D scene (the selected one in full, low-confidence ones outlined). */
  boxLayer?: MeasurementLayer | null;
  matchedComparison?: boolean;
  initialView?: "photo" | "point_cloud" | "model" | "compare";
  /** A later explicit request (e.g. from a semantic result) to show one view; nonce makes repeats distinct. */
  viewRequest?: { view: "photo" | "point_cloud" | "model" | "compare"; nonce: number } | null;
  measurementOverride?: SceneMeasurement | null;
  measurementScale?: MeasurementScale;
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
  /** A comparison measurement layer is shown instead of the report's own; link back to the original. */
  variantNotice?: { label: string; href: string };
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
  const { analysisAvailable, bendAnalysis: savedBendAnalysis, inclinationAnalysis: savedInclinationAnalysis, layerBends, layerConfidence } = useSceneResources();
  const document = revision.document,
    selected = document.entities.find((entity) => entity.id === selection.entityId);
  const camera = cameraForImage(document, imageId);
  const [cloudWithModels, setCloudWithModels] = useState(!matchedComparison);
  const hasSplats = !!splatAnnotation(document), [splats, setSplats] = useState(true);
  const [allBoxes, setAllBoxes] = useState(false);
  // The one unified floor is the frame's ground (a layer's own ground is already applied to it): a box whose up axis is not that
  // floor's normal, or with a malformed field, is left out.
  const boxFrame: BoxFrame | null = boxLayer?.revisionId === revision.id ? {
    nativeToMeters: boxLayer.scale.nativeToMeters,
    ground: document.coordinateFrames.find(frame => frame.id === boxLayer.coordinateFrameId)?.ground as BoxFrame["ground"],
  } : null;
  const boxes = boxFrame ? Object.entries(boxLayer!.boxes || {}).filter((entry): entry is [string, LayerBox] => validBox(entry[1]) && !!boxGeometry(entry[1], boxFrame.nativeToMeters, boxFrame.ground)) : [];
  const boxById = new Map(boxes), highlightCount = boxes.filter(([, box]) => box.highlight).length;
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
  // A saved analysis measured one model: once an entity shows another model (e.g. a measurement layer's), its old results no longer apply.
  const measuresShownModel = (result?: SceneMeasurement) => !result || result.references.every(ref => activeModel(document.entities.find(entity => entity.id === ref.entityId) || {} as Entity)?.id === ref.representationId);
  const inclinationRows=(inclinationAnalysis?.revisionId===revision.id?inclinationAnalysis.items:[]).map(row=>({...row,surfaces:row.surfaces.filter(surface=>measuresShownModel(surface.result))}));
  const incompleteInclinations=inclinationRows.filter(row=>row.status==="failed" || row.status==="partial").length;
  const allSurfaces=inclinationRows.flatMap(row=>row.surfaces.map(surface=>({entityId:row.entityId,surface,key:row.entityId+":"+surface.surfaceId})));
  const visibleSurfaces=allSurfaces.filter(row=>allPlanes || row.surface.classification==="non_vertical" || (row.surface.classification==="direction_unverified" && row.surface.deviationFromVerticalDeg>Math.max(row.surface.angularSpreadDeg,0.000001)));
  const activeSurface=allSurfaces.find(row=>row.key===surfaceKey && row.entityId===selected?.id)?.surface;
  useEffect(()=>{ if(activeSurface)setMeasurement(activeSurface.result); },[activeSurface]);
  const bendRows = [...(bendAnalysis?.revisionId === revision.id ? bendAnalysis.items : []).filter(row => measuresShownModel(row.result) && !layerBends?.some(layerRow => layerRow.entityId === row.entityId)),
    ...(layerBends || []).filter(row => row.result?.revisionId === revision.id && measuresShownModel(row.result))];
  const incompleteBends = bendRows.filter(row => row.status === "failed").length;
  const detectedBends = bendRows.filter(row => row.status === "measured" && row.result);
  const savedBend = bendRows.find(row => row.entityId === selected?.id);
  useEffect(() => {
    if (!analysisAvailable && !activeSurface) setMeasurement(savedBend?.status === "measured" ? savedBend.result || null : null);
  }, [analysisAvailable, selected?.id, savedBend, activeSurface]);
  const [localMeasurement, setLocalMeasurement] = useState<SceneMeasurement | null>(null), [localMeasurementActive, setLocalMeasurementActive] = useState(false);
  const activeMeasurement = localMeasurementActive ? localMeasurement : measurementOverride ?? rawMeasurement;
  const measurement = (showAngles || activeMeasurement?.unit !== "deg") && activeMeasurement?.revisionId === revision.id && (localMeasurementActive || activeMeasurement.references.some(ref => ref.entityId === selected?.id)) ? activeMeasurement : null;
  const [measurePoints,setMeasurePoints]=useState<SurfacePick[]>([]),[pickingPoints,setPickingPoints]=useState(false),[pointError,setPointError]=useState(false),[pointCount,setPointCount]=useState<1|2|3>(2);
  const measurementGeometryKey = JSON.stringify([revision.id, imageId, geometryOptions.frameId, document.entities.map(entity => [entity.id, entity.visible, entity.currentModelTransform, activeModel(entity)]), document.assets, document.coordinateFrames.map(frame => [frame.id, frame.ground])]);
  useEffect(() => { setMeasureRegion(null); setDrawingRegion(false); setMeasurePoints([]); setPickingPoints(false); setLocalMeasurement(null); setLocalMeasurementActive(false); setPointError(false); }, [measurementGeometryKey, selection.entityId]);
  useEffect(()=>{if(layer!=="model"){setMeasurePoints([]);setPickingPoints(false);setLocalMeasurement(null);setLocalMeasurementActive(false);}},[layer]);
  function startPointPicking(start:boolean, count:1|2|3=3) { setPointCount(count); setPickingPoints(start); setMeasurePoints([]); setLocalMeasurement(null); setLocalMeasurementActive(start); setPointError(false); if(start){setDrawingRegion(false);setLayer("model");chooseView("spatial");} }
  function pickMeasurementPoint(hit:SurfacePick|null) {
    if(!pickingPoints)return;
    const entity = hit && document.entities.find(entity => entity.id === hit.entityId), rep = entity && activeModel(entity);
    if(!hit || hit.point.length!==3 || !hit.point.every(Number.isFinite) || hit.coordinateFrameId!==geometryOptions.frameId || !rep || rep.id!==hit.representationId || rep.sourceValidity==="stale" || entity?.visible===false || rep.coordinateFrameId!==hit.coordinateFrameId || (entity?.currentModelTransform||rep.transform).coordinateFrameId!==hit.coordinateFrameId || measurePoints.length>0&&hit.coordinateFrameId!==measurePoints[0].coordinateFrameId){setPointError(true);return;}
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
  useEffect(() => {
    if (!viewRequest) return;
    if (viewRequest.view === "compare") { setComparing(true); setLayer("model"); setMobileSection("views"); }
    else if (viewRequest.view === "photo") chooseView("photo");
    else { setLayer(viewRequest.view); chooseView("spatial"); }
  }, [viewRequest?.nonce]);
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
          {layer === "model" && hasPointCloud && <label className="report-scene-check"><input type="checkbox" checked={cloudWithModels} onChange={(e) => setCloudWithModels(e.target.checked)} />{t("ReportScenetsx.223")}</label>}
          {hasMotion && <label><span>{t("ReportScenetsx.224")}</span><select aria-label={t("ReportScenetsx.225")} value={part} onChange={(e) => setPart(e.target.value as typeof part)}>
            <option value="all">{t("ReportScenetsx.226")}</option><option value="static">{t("ReportScenetsx.227")}</option><option value="dynamic">{t("ReportScenetsx.228")}</option>
          </select></label>}
          {hasSkeleton && part !== "static" && <label className="report-scene-check"><input type="checkbox" checked={skeleton} onChange={(e) => setSkeleton(e.target.checked)} />{t("ReportScenetsx.229")}</label>}
          <label className="report-scene-check"><input type="checkbox" checked={allBounds} onChange={(e) => setAllBounds(e.target.checked)} />{t("sceneShowBorders")}</label>
          {boxes.length > 0 && <label className="report-scene-check"><input type="checkbox" checked={allBoxes} onChange={(e) => setAllBoxes(e.target.checked)} />{t("ReportScenetsx.230")}</label>}
          <label className="report-scene-check"><input type="checkbox" checked={showPath} onChange={(e) => setShowPath(e.target.checked)} />{t("sceneShowCameraPath")}</label>
          {selected && onClearSelection && <button onClick={onClearSelection}>{t("sceneClearSelection")}</button>}
          <button className="report-scene-fullscreen" onClick={fullscreen} aria-pressed={isFullscreen} aria-label={t(isFullscreen ? "sceneExitFullscreen" : "sceneFullscreen")}>⛶ <span>{t(isFullscreen ? "sceneExitFullscreen" : "sceneFullscreen")}</span></button>
        </div>
      </header>
      {(analysisAvailable || bendAnalysis) && <div className="report-bend-analysis" aria-label={t("ReportScenetsx.231")}>
        <span>{bendError ? t("ReportScenetsx.text006") : !bendAnalysis ? t("ReportScenetsx.text007") : t("ReportScenetsx.text008", {p0: detectedBends.length, p1: incompleteBends ? t("ReportScenetsx.text001", {p0: incompleteBends}) : ""})}</span>
        <label>{t("ReportScenetsx.232")}<select value={detectedBends.some(row => row.entityId === selected?.id) ? selected!.id : ""} onChange={e => { startPointPicking(false); if(e.target.value) { selectEntity(e.target.value); setLayer("model"); if (!analysisAvailable) { setMeasurement(detectedBends.find(row => row.entityId === e.target.value)?.result || null); setShowAngles(true); chooseView("plan"); } } }}>
          <option value="">{t("ReportScenetsx.233")}</option>
          {detectedBends.map(row => <option key={row.entityId} value={row.entityId}>{document.entities.find(entity => entity.id === row.entityId)?.label || row.entityId} · {row.result!.value.toFixed(1)}°</option>)}
        </select></label>
        <label className="report-scene-check"><input type="checkbox" checked={showAngles} onChange={e => setShowAngles(e.target.checked)} />{t("ReportScenetsx.234")}</label>
      </div>}
      {(analysisAvailable || inclinationAnalysis) && <div className="report-bend-analysis" aria-label={t("ReportScenetsx.235")}>
        <span>{inclinationError ? (t("ReportScenetsx.236")) : !inclinationAnalysis ? (t("ReportScenetsx.237")) : t("ReportScenetsx.text009", {p0: allSurfaces.length, p1: inclinationRows.filter(row=>row.surfaces.length>0).length, p2: incompleteInclinations ? t("ReportScenetsx.text001", {p0: incompleteInclinations}) : ""})}</span>
        <label>{t("ReportScenetsx.238")}<select value={surfaceKey} onChange={e=>{startPointPicking(false);const row=allSurfaces.find(r=>r.key===e.target.value);if(row){selectEntity(row.entityId);setSurfaceKey(row.key);setLayer("model");setPreviewMode("free");if(!analysisAvailable){setShowAngles(true);chooseView("spatial");}}else{setSurfaceKey("");setMeasurement(null);}}}>
          <option value="">{t("ReportScenetsx.239")}</option>
          {visibleSurfaces.map(row=><option key={row.key} value={row.key}>{document.entities.find(e=>e.id===row.entityId)?.label || row.entityId} · {t("ReportScenetsx.240")} {row.surface.surfaceId} · {row.surface.inclinationDeg.toFixed(1)}°{row.surface.classification==="direction_unverified" ? (t("ReportScenetsx.241")) : ""}</option>)}
        </select></label>
        <label className="report-scene-check"><input type="checkbox" checked={allPlanes} onChange={e=>{setAllPlanes(e.target.checked);setSurfaceKey("");setMeasurement(null);}} />{t("ReportScenetsx.242")}</label>
        <span>{t("ReportScenetsx.243")}</span>
      </div>}
      {variantNotice && <div className="report-scene-history-notice" role="status" data-layer-variant><span>{variantNotice.label}</span><a href={variantNotice.href}>{t("sceneOriginalReport")} ↗</a></div>}
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
              const recapture = !!boxById.get(entity.id)?.highlight, confidence = layerConfidence?.[entity.id],
                confidenceMissing = confidence?.missing?.map(message => renderMessage(language, message));
              return <div key={entity.id} className="report-scene-object-row" data-entity-id={entity.id} data-parent-entity-id={entity.parentEntityId || undefined} data-box-recapture={recapture || undefined} style={{ marginLeft: depth * 12 }}><button key={entity.id} aria-pressed={entity.id === selection.entityId} onClick={() => selectEntity(entity.id)}>
                <strong><b className="report-scene-object-number">{objectNumbers.get(entity.id)}</b>{entity.label || entity.id}</strong>
                <span className="report-scene-object-source">{indices.length ? `${t("scenePhotoNumber")} ${indices.join(" / ")}` : t("sceneNoPhotoLink")}<small>{entity.id.slice(0, 8)}</small></span>
                <span className="report-scene-object-evidence">{t(evidence.photoKey)}{observations.length > 0 && ` · ${observations.length} ${t("observations")}`}</span>
                {evidence.identityKey && <span className="report-scene-object-identity"><span>{t("entityIdentity")}</span>{t(evidence.identityKey)}</span>}
                <span className="report-scene-object-model" data-model-state={modelStatus}><span>{t("model")}</span>{t(modelStatus)}{composite && modelStatus !== "sceneCompositeEvidence" && <> · {t("sceneCompositeEvidence")}</>}{candidate && <em>{t("sceneCandidate")}</em>}</span>
                {confidence && <span className="report-scene-object-confidence" data-confidence={confidence.level} title={(confidence.reasons?.map(message => renderMessage(language, message)) || []).join(t("ReportScenetsx.244"))}><span>{t("ReportScenetsx.221")}</span><b>{renderMessage(language, confidence.label)}</b>{confidenceMissing?.[0] && <small>{confidenceMissing[0]}</small>}</span>}
                <span className="report-scene-object-extent"><span>{t("reportObservedExtent")}</span><Extent entity={entity} document={document} /></span>
                {recapture && <span className="report-scene-object-recapture">{t("ReportScenetsx.216")}</span>}
              </button>{children.has(entity.id) && <button aria-expanded={expandedEntities.has(entity.id)} aria-label={`${t("sceneModelParts")} · ${entity.label || entity.id}`} onClick={() => setExpandedEntities(current => { const next = new Set(current);if (next.has(entity.id)) next.delete(entity.id);else next.add(entity.id);return next; })}>{expandedEntities.has(entity.id) ? "▾" : "▸"} {children.get(entity.id)!.length} {t("sceneModelParts")}</button>}{onFeedback && <button className="report-object-feedback" aria-label={`${t("sceneFeedback")} · ${entity.label || entity.id}`} onClick={() => { selectEntity(entity.id); onFeedback(entity.id); setMobileSection("inspector"); }}>{t("sceneFeedback")} ↗</button>}</div>;
            })}
            {!filtered.length && <p className="report-scene-list-empty">{t(objects.length ? "sceneNoMatches" : "sceneNoObjects")}</p>}
          </div>
          <footer><span>{filtered.length} / {counts.records} · {t("identityRecords")}</span><small>{counts.linkedGroups} {t("identityGroups")} · {counts.pending} {t("identityPending")} · {counts.observations} {t("observations")}</small>
            <details><summary>{t("sceneRecordNote")}</summary><p>{t("reportAssociationHint")}</p></details>
          </footer>
        </aside>
        <div className="report-scene-center" id={`${panePrefix}-views`}>
          {matchedComparison && <nav className="report-scene-primary-views" aria-label={t("ReportScenetsx.text010")}>
            <button aria-pressed={!comparing && focused === "photo"} onClick={() => chooseView("photo")}><strong>{t("ReportScenetsx.text011")}</strong><small>{t("ReportScenetsx.text012")}</small></button>
            <button aria-pressed={comparing} onClick={() => { setComparing(true); setLayer("model"); setMobileSection("views"); }}><strong>{t("ReportScenetsx.text013")}</strong><small>{t("ReportScenetsx.text014")}</small></button>
            <button disabled={!hasPointCloud} aria-pressed={!comparing && focused === "spatial" && layer === "point_cloud"} onClick={() => { setLayer("point_cloud"); chooseView("spatial"); }}><strong>{t("ReportScenetsx.text015")}</strong><small>{hasPointCloud ? t("ReportScenetsx.text016") : t("ReportScenetsx.text017")}</small></button>
            <button aria-pressed={!comparing && focused === "spatial" && layer === "model"} onClick={() => { setLayer("model"); chooseView("spatial"); }}><strong>{t("ReportScenetsx.text018")}</strong><small>{t("ReportScenetsx.text019")}</small></button>
          </nav>}
          {matchedComparison && !comparing && focused === "spatial" && <nav className="report-scene-photo-switch" aria-label={t("ReportScenetsx.text020")}>{images.map((image, i) => <button key={image.imageId} aria-pressed={imageId === image.imageId} onClick={() => onCamera(image.imageId, image.cameraId)}>{t("photos")}{i + 1}</button>)}</nav>}
          {matchedComparison && !comparing && focused === "spatial" && layer === "point_cloud" && <p className="report-scene-notice" data-point-cloud-image-id={imageId}>{t("ReportScenetsx.text021")}{images.findIndex(image => image.imageId === imageId) + 1} {t("ReportScenetsx.text022")}</p>}
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
              <div className="report-comparison-divider" style={{ left: `${wipe}%` }} role="slider" tabIndex={0} aria-label={t("ReportScenetsx.text023")} aria-valuemin={0} aria-valuemax={100} aria-valuenow={wipe}
                onPointerDown={event => { event.preventDefault(); event.currentTarget.focus({ preventScroll: true }); event.currentTarget.setPointerCapture(event.pointerId); moveComparison(event); }}
                onPointerMove={event => { if (event.currentTarget.hasPointerCapture(event.pointerId)) moveComparison(event); }}
                onPointerUp={event => { if (event.currentTarget.hasPointerCapture(event.pointerId)) { moveComparison(event); event.currentTarget.releasePointerCapture(event.pointerId); } }}
                onPointerCancel={event => { if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}
                onKeyDown={event => { const key = event.key; if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(key)) { event.preventDefault(); setWipe(value => key === "Home" ? 0 : key === "End" ? 100 : Math.max(0, Math.min(100, value + (["ArrowRight", "ArrowUp"].includes(key) ? 1 : -1)))); } }}><b aria-hidden="true">↔</b></div>
              <span className="report-comparison-photo-label">{t("ReportScenetsx.text011")}</span><span className="report-comparison-model-label">{t("ReportScenetsx.text024")}</span>
            </div>
            <label className="report-comparison-control"><span>{t("measure.model")}</span><input aria-label={t("ReportScenetsx.text025")} type="range" min="0" max="100" value={wipe} onChange={event => setWipe(Number(event.target.value))} /><span>{t("photos")}</span><output>{wipe}%</output></label>
            <footer className="report-scene-photo-switch">{images.map((image, i) => <button key={image.imageId} aria-pressed={imageId === image.imageId} onClick={() => onCamera(image.imageId, image.cameraId)}>{t("photos")}{i + 1}</button>)}</footer>
          </div>}
          {!comparing && <div className="report-scene-grid">
            {paneOrder.map((pane, index) => (
              <section className="report-scene-pane" id={`${panePrefix}-${pane}`} data-pane={pane} key={pane} aria-label={t(viewNames[pane])}>
                <header><h3><span>{String(index + 1).padStart(2, "0")}</span>{pane === "photo" && showVideo ? (t("ReportScenetsx.245")) : t(pane === "spatial" ? availability.spatialTitle : viewNames[pane])}</h3>
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
                  {pane === "spatial" && <>{pickingPoints && <div className="cad-measure-guide" role="status">{t("ReportScenetsx.text026", {p0: measurePoints.length+1, p1: pointCount})}{pointError && <strong>{t("ReportScenetsx.246")}</strong>} <button onClick={()=>startPointPicking(false)}>{t("measure.cancelPicking")}</button></div>}<SpatialView revision={revision} selection={selection} onSelect={selectEntity} onCommit={noEdit} mode="free" cameraId={camera?.id || null}
                    modelPreview={previewRequest} onModelPreview={(key, image) => { if (key === currentPreviewKey.current) setModelPreview({ key, image }); }}
                    onAssetStates={(revisionId, states) => setModelLoads({ revisionId, states })} onMeasurementPoint={pickMeasurementPoint}
                    layers={{ groundDatum: matchedComparison || localMeasurementActive, measurementScale, pickingPoints, measurePoints, measureVertexIndex: pointCount === 2 ? 0 : 1, measurement: withBoxes(layer === "model" || layer === "point_cloud" && measurement?.method === "conditional-endpoint-comparison" ? measurement : null, boxLayer, boxes, boxFrame, selected, allBoxes, language, t), labels: allBoxes && !allBounds && "hover", modelOnly: layer === "model", observed_surface: layer === "observed_surface", generated_mesh: layer === "model", primitive: layer === "model", point_cloud: layer === "point_cloud" || layer === "model" && cloudWithModels && hasPointCloud, allBounds, showBounds: allBounds, cameraPath: showPath, showCandidates: true, editable: false, opacity: 1, imageId, observationEntityId: selected?.id, observationId: selection.observationId, observations: document.observations, part, skeleton, splats: hasSplats && splats }} />
                    {!hasRepresentation && <div className="report-scene-stage-note">{t("sceneNoRepresentation")}</div>}
                    {boxes.length > 0 && <div className="report-box-legend" role="note" data-highlight-count={highlightCount}><span>{t("ReportScenetsx.247")}</span>
                      {boxLevels.map(level => <i key={level} data-confidence={level}>{t(levelNames[level])}</i>)}
                      <b data-active={highlightCount > 0 || undefined}>{t("ReportScenetsx.248")} · {highlightCount}</b></div>}</>}
                  {pane === "cad" && availability.planEmpty && <div className="report-scene-plan-empty" role="status"><strong>{t("scenePlanUnavailable")}</strong><p>{t(availability.planEmpty)}</p><small>{t("sceneSelectionRetained")}</small></div>}
                  {pane === "cad" && !availability.planEmpty && <CadView key={revision.id + cadLayer} document={document} selectedId={selection.entityId} onSelect={selectPlanEntity} geometryOptions={planOptions} measurement={cadLayer === "model" ? measurement : null} region={cadLayer === "model" ? measureRegion : null} drawingRegion={drawingRegion} onRegion={region => { setMeasureRegion(region); setDrawingRegion(false); setMobileSection("inspector"); }} showPath={showPath} />}
                  {pane === "plan" && <div className="report-model-preview" data-model-entity={selected?.id || ""} data-preview-layer={layer}>
                    {previewGeometry && selected ? <>
                      <nav aria-label={t(layer === "model" ? "sceneModelOrientation" : "sceneEvidenceOrientation")}>{(["free", "front", "side", "top"] as const).map(mode => <button key={mode} aria-pressed={previewMode === mode} onClick={() => setPreviewMode(mode)}>{t(mode)}</button>)}{previewMode === "free" && <small>{t("ReportScenetsx.249")}</small>}</nav>
                      <div className="report-model-image">{previewMode === "free" ? <SpatialView
                        key={`${revision.id}:${selected.id}:${layer}:${geometryOptions.frameId}:${selection.observationId || ""}`}
                        revision={revision} selection={{ ...selection, entityId: null }} onSelect={noEdit} onCommit={noEdit} mode="free" cameraId={camera?.id || null}
                        layers={{ studio: true, entityIds: layer === "model" ? selectedFamily.map(entity => entity.id) : [selected.id], representationIds: previewGeometry.representationIds,
                          axisEntityId: selectedComposite ? undefined : selected.id, showAxes: !selectedComposite, showBounds: false, editable: false, showCandidates: true,
                          modelOnly: layer === "model", generated_mesh: layer === "model", primitive: layer === "model", observed_surface: layer === "observed_surface", point_cloud: layer === "point_cloud",
                          observationEntityId: selected.id, observationId: selection.observationId, imageId, observations: document.observations,
                          measurement: withBoxes(layer === "model" && measurement?.references.every(ref => selectedFamily.some(entity => entity.id === ref.entityId)) ? measurement : null, boxLayer, boxes.filter(([id]) => id === selected.id), boxFrame, selected, false, language, t) }} /> : previewImage ? <img src={previewImage} alt={`${selected.label || selected.id} · ${t(viewNames.plan)} · ${t(previewMode)}`} /> : <p role="status">{t(selectedModelStale ? "identityModelStale" : selectedModelWrongFrame ? "sceneModelWrongFrame" : selectedModelFailed ? layer === "model" ? "sceneModelLoadFailed" : "sceneEvidenceLoadFailed" : layer === "model" ? "loadingModel" : "sceneEvidenceLoading")}</p>}</div>
                      <footer><strong>{selected.label || selected.id}</strong><span>{selectedComposite ? <>{t("sceneCompositeEvidence")} · {selectedComposite.targets.map(entity => `${objectNumbers.get(entity.id)} ${entity.label || entity.id}`).join(" + ")} · {t("sceneCompositeEvidenceMeaning")}</> : selectedReference ? <>{t("observed_reference_surface")} · {t("sceneObservedCoverage")}</> : layer === "model" ? t(selectedModels.every(entity => activeModel(entity)?.placementState === "confirmed") ? "identityPlacementConfirmed" : "identityPlacementUnconfirmed") : <>{t(layer)} · {t("scenePhotoNumber")} {images.findIndex(image => image.imageId === imageId) + 1}{selection.observationId && <> · {t("sceneObservation")} {selection.observationId.slice(0, 8)}</>} · {t("sceneObservedCoverage")}</>}</span><button onClick={() => chooseView("spatial")}>{t(layer === "model" ? "sceneOpenModelScene" : "sceneOpenEvidenceScene")} ↗</button></footer>
                    </> : <div className="report-scene-plan-empty" role="status"><strong>{t(selectedModelStale ? "identityModelStale" : selectedModelWrongFrame ? "sceneModelWrongFrame" : selected ? layer === "model" ? "sceneObjectModelMissing" : "sceneEvidenceMissing" : layer === "model" ? "sceneSelectModel" : "sceneSelectEvidence")}</strong><p>{t(selectedModelWrongFrame ? "sceneModelWrongFrameMeaning" : selected ? layer === "model" ? "sceneObjectModelMissingMeaning" : "sceneEvidenceMissingMeaning" : "sceneSelectModelMeaning")}</p></div>}
                  </div>}
                </div>
                {pane === "photo" && hasVideo && <footer className="report-scene-photo-switch">
                  <button aria-pressed={showVideo} onClick={() => setVideoMode(true)}>{t("ReportScenetsx.250")}</button>
                  <button aria-pressed={!showVideo} onClick={() => setVideoMode(false)}>{t("ReportScenetsx.251")}</button></footer>}
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
            <h4>{t("ReportScenetsx.252")}</h4><p>{(selected as any).motionSummary.text}</p>
            <small>{t("ReportScenetsx.253")}</small></section>}{selected && boxById.has(selected.id) && <LayerBoxPanel box={boxById.get(selected.id)!} entityId={selected.id} />}{selected && <SpatialMeasurements key={revision.id + selected.id} revision={revision} selectedId={selected.id} geometryKey={measurementGeometryKey} analysisAvailable={analysisAvailable} measurementScale={measurementScale} onLocalResult={setLocalMeasurement} savedBend={savedBend} savedSurface={activeSurface} inclinationOutcome={inclinationRows.find(r=>r.entityId===selected.id)} onClearSurface={()=>setSurfaceKey("")} points={measurePoints} pickingPoints={pickingPoints} onPickPoints={startPointPicking} region={measureRegion} drawing={drawingRegion} onResult={setMeasurement} onDraw={() => { setDrawingRegion(!drawingRegion); if (!drawingRegion) { setCadLayer("model"); setFocused("cad"); setMobileSection("views"); } }} />}{(typeof inspector === "function" ? inspector(activeSurface) : inspector) ?? <p className="report-scene-inspector-empty">{t("sceneReadOnly")}</p>}</div>
        </aside>
      </div>
    </section>
  );
}
