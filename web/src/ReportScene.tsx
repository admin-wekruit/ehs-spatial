import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { SpatialView, PlanView } from "./App";
import { PhotoView } from "./PhotoView";
import { CadView } from "./CadView";
import { useI18n } from "./i18n";
import { observationsFor, entityGeometryForLayer, jsonObject, planShapes, type GeometryOptions } from "./core";
import { isReferenceSurface } from "./scene-semantics";
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
import type { Camera, Entity, Revision, Selection, SceneDocument } from "./types";
import "./report-scene.css";

type Pane = "photo" | "spatial" | "cad" | "plan";
type Layer = "model" | "observed_surface" | "point_cloud";
const paneNames: Record<Pane, string> = {
  photo: "scenePhoto",
  spatial: "scene3D",
  cad: "sceneCAD",
  plan: "scenePlan",
};
const paneOrder: Pane[] = ["photo", "spatial", "cad", "plan"];
const colors = ["#e86b58", "#39ad7c", "#458ce0"];
const noEdit = () => {};

function sceneAvailability(document: SceneDocument, geometryOptions: GeometryOptions) {
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
    spatialTitle: hasModel && !hasObserved ? "sceneObjectModels" : "scene3D",
    planEmpty: planShapes(document, geometryOptions).length ? null : hasGround ? "sceneNoPlanProjection" : "sceneNoPlanGround",
  };
}

// Uses the same camera projection and pixel-centre convention as the WebGL view.
function photoOverlay(document: SceneDocument, entity: Entity, camera: Camera, layer: Layer) {
  if (isReferenceSurface(document, entity)) return null;
  const geometry = entityGeometryForLayer(entity, { layer, frameId: camera.coordinateFrameId, showCandidates: true });
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
  allBounds,
}: {
  revision: Revision;
  camera: Camera;
  layer: Layer;
  selectedId: string | null;
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
          const overlay = photoOverlay(revision.document, entity, camera, layer);
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

export function ReportScene({
  revision, selection, onSelect, imageId, cameraId, onCamera,
  draw = false, onBox, onOpenSourceCad, inspector,
}: {
  revision: Revision;
  selection: Selection;
  onSelect: (entityId: string, observationId?: string) => void;
  imageId: string | null;
  cameraId: string | null;
  onCamera: (imageId: string, cameraId: string | null) => void;
  draw?: boolean;
  onBox?: (box: number[] | null) => void;
  onOpenSourceCad?: () => void;
  inspector?: ReactNode;
}) {
  const { t } = useI18n(), container = useRef<HTMLElement>(null),
    objectList = useRef<HTMLDivElement>(null), panePrefix = useId();
  const [layer, setLayer] = useState<Layer>("model"),
    [allBounds, setAllBounds] = useState(false),
    [focused, setFocused] = useState<Pane | null>(null),
    [mobileSection, setMobileSection] = useState("views"),
    [search, setSearch] = useState(""),
    [fullscreenError, setFullscreenError] = useState(false),
    [isFullscreen, setIsFullscreen] = useState(false);
  const document = revision.document,
    selected = document.entities.find((entity) => entity.id === selection.entityId);
  const camera = document.cameras.find((c) => c.id === cameraId && c.imageId === imageId) ||
    document.cameras.find((c) => c.imageId === imageId);
  const geometryOptions: GeometryOptions = { layer, frameId: camera?.coordinateFrameId || document.coordinateFrames[0]?.id || "", showCandidates: true };
  const availability = sceneAvailability(document, geometryOptions);
  const images = [...new Set([
    ...document.assets.filter((a) => a.kind === "source_image").map((a) => a.id),
    ...document.cameras.map((c) => c.imageId),
  ])].map((id) => ({ imageId: id, cameraId: document.cameras.find((c) => c.imageId === id)?.id || null }));
  const hasPointCloud = document.entities.some((e) => (e.representations || []).some((r) => r.kind === "point_cloud"));
  const hasRepresentation = document.entities.some((e) => (e.representations || []).some((r) => layer === "model"
    ? ["generated_mesh", "primitive", "observed_surface"].includes(r.kind) : r.kind === layer));
  const hasCandidates = layer === "model" && document.entities.some((e) => (e.representations || []).some((r) =>
    r.placementState === "unconfirmed" && ["imported_proposal", "requires_alignment_confirmation"].includes(r.placementReason || "")));
  const selectedOverlay = selected && camera ? photoOverlay(document, selected, camera, layer) : null;
  const objects = document.entities.filter((entity) => !entity.sourceContext);
  const objectNumbers = new Map(objects.map((entity, index) => [entity.id, String(index + 1).padStart(2, "0")]));
  const query = search.trim().toLowerCase();
  const numberedObject = objects.find((entity) => objectNumbers.get(entity.id) === query || String(Number(objectNumbers.get(entity.id))) === query);
  const filtered = numberedObject ? [numberedObject] : objects.filter((e) => `${e.label || ""} ${e.id}`.toLowerCase().includes(query));
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
    const changed = () => {
      const active = window.document.fullscreenElement === container.current;
      setIsFullscreen(active);
      if (!active) setFocused(null);
    };
    window.document.addEventListener("fullscreenchange", changed);
    return () => window.document.removeEventListener("fullscreenchange", changed);
  }, []);
  useEffect(() => {
    const list = objectList.current, active = list?.querySelector<HTMLElement>('[aria-pressed="true"]');
    if (!list || !active) return;
    const row = active.getBoundingClientRect(), viewport = list.getBoundingClientRect();
    // Scroll only the object rail; selecting a scene object must not move the report.
    if (row.top < viewport.top) list.scrollTop -= viewport.top - row.top;
    else if (row.bottom > viewport.bottom) list.scrollTop += row.bottom - viewport.bottom;
  }, [selection.entityId, search, isFullscreen, mobileSection]);
  function chooseView(pane: Pane | null) { setFocused(pane); setMobileSection("views"); }
  async function fullscreen() {
    setFullscreenError(false);
    try {
      if (window.document.fullscreenElement === container.current) await window.document.exitFullscreen();
      else if (container.current?.requestFullscreen) await container.current.requestFullscreen();
      else throw new Error("fullscreen_unavailable");
    } catch { setFullscreenError(true); }
  }
  function selectEntity(entityId: string, observationId?: string) {
    onSelect(entityId, observationId);
    setMobileSection("views");
    const entity = document.entities.find((e) => e.id === entityId);
    if (!entity) return;
    const observations = observationsFor(document, entity), observation = observations.find((o) => o.id === observationId) ||
      observations.find((o) => o.imageId === imageId) || observations[0];
    if (observation && observation.imageId !== imageId)
      onCamera(observation.imageId, document.cameras.find((c) => c.imageId === observation.imageId)?.id || null);
  }
  return (
    <section ref={container} className="report-scene" data-view={focused || "quad"}
      data-mobile-section={mobileSection} data-mobile-pane={focused || "photo"}
      onKeyDown={(event) => {
        if (event.key === "Escape" && !event.defaultPrevented) { setFocused(null); setFullscreenError(false); }
      }}>
      <header className="report-scene-toolbar">
        <div className="report-scene-intro"><strong>{t("sceneWorkspace")}</strong><span>{t("sceneLinked")}</span></div>
        <div className="report-scene-controls">
          <label><span>{t("sceneLayers")}</span><select aria-label={t("sceneLayers")} value={layer} onChange={(e) => setLayer(e.target.value as Layer)}>
            <option value="model">{t("sceneModel")}</option><option value="observed_surface">{t("sceneObserved")}</option>
            <option value="point_cloud" disabled={!hasPointCloud}>{t(hasPointCloud ? "scenePoints" : "sceneNoPoints")}</option>
          </select></label>
          <label className="report-scene-check"><input type="checkbox" checked={allBounds} onChange={(e) => setAllBounds(e.target.checked)} />{t("sceneAllBounds")}</label>
          <button className="report-scene-fullscreen" onClick={fullscreen} aria-pressed={isFullscreen} aria-label={t(isFullscreen ? "sceneExitFullscreen" : "sceneFullscreen")}>⛶ <span>{t(isFullscreen ? "sceneExitFullscreen" : "sceneFullscreen")}</span></button>
        </div>
      </header>
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
          <header><h3>{t("sceneObjects")}</h3><span className="report-scene-count">{objects.length}</span></header>
          <div className="report-scene-object-search"><input type="search" aria-label={t("sceneSearch")} placeholder={t("sceneSearch")} value={search} onChange={(e) => setSearch(e.target.value)} /></div>
          <div className="report-scene-current-object" aria-live="polite">
            <span>{t("sceneSelected")}</span><strong>{selected?.label || selected?.id || t("sceneNoSelection")}</strong>
            {selected && <small>{selected.id.slice(0, 8)}</small>}
          </div>
          <div ref={objectList} className="report-scene-object-list">
            {filtered.map((entity) => {
              const indices = images.flatMap((image, index) => photoIds.get(entity.id)?.has(image.imageId) ? [index + 1] : []);
              return <button key={entity.id} aria-pressed={entity.id === selection.entityId} onClick={() => selectEntity(entity.id)}>
                <strong><b className="report-scene-object-number">{objectNumbers.get(entity.id)}</b>{entity.label || entity.id}</strong>
                <span>{indices.length ? `${t("scenePhotoNumber")} ${indices.join(" / ")}` : t("sceneNoPhotoLink")}<small>{entity.id.slice(0, 8)}</small></span>
                {!(entity.representations || []).length ? <em>{t(entityGeometryForLayer(entity, geometryOptions) ? "sceneBoundsOnly" : "sceneNoGeometry")}</em> :
                  (entity.representations || []).some((r) => r.placementState === "unconfirmed") && <em>{t("sceneCandidate")}</em>}
              </button>;
            })}
            {!filtered.length && <p className="report-scene-list-empty">{t(objects.length ? "sceneNoMatches" : "sceneNoObjects")}</p>}
          </div>
          <footer>{filtered.length} / {objects.length} · {t("sceneObjects")}</footer>
        </aside>
        <div className="report-scene-center" id={`${panePrefix}-views`}>
          <nav className="report-scene-view-switch" aria-label={t("sceneViews")}>
            <button className="report-scene-quad" aria-pressed={!focused} onClick={() => chooseView(null)}>{t("sceneQuad")}</button>
            {paneOrder.map((pane) => <button key={pane} aria-pressed={focused === pane} aria-controls={`${panePrefix}-${pane}`} onClick={() => chooseView(pane)}>{t(paneNames[pane])}</button>)}
          </nav>
          <nav className="report-scene-view-switch report-scene-mobile-views" aria-label={t("sceneViews")}>
            {paneOrder.map((pane) => <button key={pane} aria-pressed={(focused || "photo") === pane} aria-controls={`${panePrefix}-${pane}`} onClick={() => chooseView(pane)}>{t(paneNames[pane])}</button>)}
          </nav>
          {hasCandidates && <p className="report-scene-notice">{t("sceneCandidateNotice")}</p>}
          <div className="report-scene-grid">
            {paneOrder.map((pane, index) => (
              <section className="report-scene-pane" id={`${panePrefix}-${pane}`} data-pane={pane} key={pane} aria-label={t(paneNames[pane])}>
                <header><h3><span>{String(index + 1).padStart(2, "0")}</span>{t(pane === "spatial" ? availability.spatialTitle : paneNames[pane])}</h3>
                  {pane === "cad" && onOpenSourceCad && <button className="report-scene-source-cad" onClick={onOpenSourceCad}>{t("sceneSourceCad")} ↗</button>}
                  <button className="report-scene-expand" aria-label={`${t(focused === pane ? "sceneQuad" : "sceneSingleView")} · ${t(paneNames[pane])}`}
                    title={t(focused === pane ? "sceneQuad" : "sceneSingleView")} onClick={() => chooseView(focused === pane ? null : pane)}>{focused === pane ? "⊞" : "↗"}</button>
                </header>
                <div className="report-scene-pane-body">
                  {pane === "photo" && <><PhotoView document={document} imageId={imageId} selectedId={selection.entityId} onSelect={selectEntity} draw={draw} onBox={onBox} />
                    {camera && <PhotoAxes revision={revision} camera={camera} layer={layer} selectedId={selection.entityId} allBounds={allBounds} />}</>}
                  {pane === "spatial" && <><SpatialView revision={revision} selection={selection} onSelect={selectEntity} onCommit={noEdit} mode="free" cameraId={cameraId}
                    layers={{ observed_surface: layer !== "point_cloud", generated_mesh: layer === "model", primitive: layer === "model", point_cloud: layer === "point_cloud", allBounds, showCandidates: true, editable: false, opacity: 1 }} />
                    {!hasRepresentation && <div className="report-scene-stage-note">{t("sceneNoRepresentation")}</div>}</>}
                  {(pane === "plan" || pane === "cad") && availability.planEmpty && <div className="report-scene-plan-empty" role="status"><strong>{t("scenePlanUnavailable")}</strong><p>{t(availability.planEmpty)}</p><small>{t("sceneSelectionRetained")}</small></div>}
                  {pane === "cad" && !availability.planEmpty && <CadView key={revision.id} document={document} selectedId={selection.entityId} onSelect={selectEntity} geometryOptions={geometryOptions} />}
                  {pane === "plan" && !availability.planEmpty && <PlanView document={document} selectedId={selection.entityId} onSelect={selectEntity} interactive geometryOptions={geometryOptions} />}
                </div>
                {pane === "photo" && <footer className="report-scene-photo-switch">{images.map((image, i) => <button key={image.imageId} aria-pressed={imageId === image.imageId} onClick={() => onCamera(image.imageId, image.cameraId)}>{t("scenePhotoNumber")} {i + 1}</button>)}{draw && <span>{t("sceneDrawActive")}</span>}</footer>}
              </section>
            ))}
          </div>
          <p className="report-scene-selection-note">{selected ? isReferenceSurface(document, selected) ? t("sceneReferenceSurface") : selectedOverlay ? t(selectedOverlay.axisSpace === "native" ? "sceneNativeAxis" : "sceneSourceAxis") : !entityGeometryForLayer(selected, geometryOptions) ? t("sceneNoGeometrySelection") : t("sceneNoPhotoAxes") : t("sceneReadOnly")}</p>
        </div>
        <aside className="report-scene-inspector" id={`${panePrefix}-inspector`} aria-label={t("sceneInspector")}>
          <header><h3>{t("sceneInspector")}</h3>{selected && <span>{selected.id.slice(0, 8)}</span>}</header>
          <div className="report-scene-inspector-content">{inspector ?? <p className="report-scene-inspector-empty">{t("sceneReadOnly")}</p>}</div>
        </aside>
      </div>
    </section>
  );
}
