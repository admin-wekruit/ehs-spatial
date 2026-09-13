import { useEffect, useId, useRef, useState } from "react";
import { SpatialView, PlanView } from "./App";
import { PhotoView } from "./PhotoView";
import { useI18n } from "./i18n";
import { observationsFor, modelGeometry, jsonObject, planShapes } from "./core";
import { isReferenceSurface } from "./scene-semantics";
import {
  add,
  boundsCorners,
  cameraMatrix,
  point,
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

function sceneAvailability(document: SceneDocument) {
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
    const normal = frame.ground?.normal;
    return Array.isArray(normal) && normal.length === 3 && normal.every(Number.isFinite) && Math.hypot(...normal) > 1e-8;
  });
  return {
    spatialTitle: hasModel && !hasObserved ? "sceneObjectModels" : "scene3D",
    planEmpty: planShapes(document).length ? null : hasGround ? "sceneNoPlanProjection" : "sceneNoPlanGround",
  };
}

function entityGeometry(entity: Entity, layer: Layer, frameId?: string) {
  if (entity.sourceContext || entity.visible === false) return null;
  const representations = (entity.representations || []).filter(
    (representation) =>
      (!frameId || representation.coordinateFrameId === frameId) &&
      (representation.placementState === "confirmed" ||
        ["imported_proposal", "requires_alignment_confirmation"].includes(
          representation.placementReason || "",
        )),
  );
  const representation =
    representations.find((r) =>
      layer === "model"
        ? r.kind === "generated_mesh" || r.kind === "primitive"
        : r.kind === layer,
    ) ||
    representations.find((r) => r.kind === "observed_surface") ||
    representations.find(
      (r) => r.kind === "generated_mesh" || r.kind === "primitive",
    );
  if (!representation) {
    const corners = jsonObject(entity.measurements?.basis)?.cornersNative;
    const coordinateFrameId = entity.measurements?.coordinateFrameId;
    if (
      typeof coordinateFrameId !== "string" ||
      (frameId && coordinateFrameId !== frameId) ||
      !Array.isArray(corners) ||
      corners.length !== 8 ||
      !corners.every(
        (p) =>
          Array.isArray(p) &&
          p.length === 3 &&
          p.every((v) => typeof v === "number" && Number.isFinite(v)),
      )
    )
      return null;
    return {
      corners: corners as Vec[],
      transform: {
        coordinateFrameId,
        position: [0, 0, 0],
        quaternion: [0, 0, 0, 1],
        scale: [1, 1, 1],
      },
      axisSpace: "native",
    };
  }
  const modeled = ["generated_mesh", "primitive"].includes(representation.kind);
  const transform = modeled
    ? entity.currentModelTransform || representation.transform
    : representation.transform;
  const model = modeled
    ? modelGeometry({ ...entity, representations: [representation] })
    : null;
  if (model) return { corners: model.corners, transform, axisSpace: "local" };
  if (!representation.bounds) return null;
  const matrix = transformMatrix(transform);
  return {
    corners: boundsCorners(representation.bounds).map((p) => point(matrix, p)),
    transform,
    axisSpace: modeled ? "local" : "native",
  };
}

// Uses the same camera projection and pixel-centre convention as the WebGL view.
function photoOverlay(document: SceneDocument, entity: Entity, camera: Camera, layer: Layer) {
  if (isReferenceSurface(document, entity)) return null;
  const geometry = entityGeometry(entity, layer, camera.coordinateFrameId);
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
  revision,
  selection,
  onSelect,
  imageId,
  cameraId,
  onCamera,
  draw = false,
  onBox,
  onOpenSourceCad,
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
}) {
  const { t } = useI18n(),
    container = useRef<HTMLElement>(null),
    panePrefix = useId();
  const [layer, setLayer] = useState<Layer>("model"),
    [allBounds, setAllBounds] = useState(false);
  const [mobilePane, setMobilePane] = useState<Pane>("photo"),
    [focused, setFocused] = useState<Pane | null>(null);
  const [search, setSearch] = useState(""),
    [fullscreenError, setFullscreenError] = useState(false);
  const document = revision.document;
  const availability = sceneAvailability(document);
  const selected = document.entities.find(
    (entity) => entity.id === selection.entityId,
  );
  const camera =
    document.cameras.find((c) => c.id === cameraId && c.imageId === imageId) ||
    document.cameras.find((c) => c.imageId === imageId);
  const images = [
    ...new Set([
      ...document.assets
        .filter((a) => a.kind === "source_image")
        .map((a) => a.id),
      ...document.cameras.map((c) => c.imageId),
    ]),
  ].map((id) => ({
    imageId: id,
    cameraId: document.cameras.find((c) => c.imageId === id)?.id || null,
  }));
  const hasPointCloud = document.entities.some((e) =>
    (e.representations || []).some((r) => r.kind === "point_cloud"),
  );
  const hasRepresentation = document.entities.some((e) =>
    (e.representations || []).some((r) =>
      layer === "model"
        ? ["generated_mesh", "primitive", "observed_surface"].includes(r.kind)
        : r.kind === layer,
    ),
  );
  const hasCandidates =
    layer === "model" &&
    document.entities.some((e) =>
      (e.representations || []).some(
        (r) =>
          r.placementState === "unconfirmed" &&
          ["imported_proposal", "requires_alignment_confirmation"].includes(
            r.placementReason || "",
          ),
      ),
    );
  const selectedOverlay =
    selected && camera ? photoOverlay(document, selected, camera, layer) : null;
  const objects = document.entities.filter((entity) => !entity.sourceContext);
  const filtered = objects.filter((e) =>
    `${e.label || ""} ${e.id}`.toLowerCase().includes(search.toLowerCase()),
  );
  useEffect(() => {
    if (layer === "point_cloud" && !hasPointCloud) setLayer("model");
  }, [hasPointCloud, layer]);
  useEffect(() => {
    if (draw) {
      setMobilePane("photo");
      setFocused(null);
    }
  }, [draw]);
  useEffect(() => {
    const changed = () => {
      if (!window.document.fullscreenElement) setFocused(null);
    };
    window.document.addEventListener("fullscreenchange", changed);
    return () =>
      window.document.removeEventListener("fullscreenchange", changed);
  }, []);
  async function fullscreen(pane: Pane) {
    setFocused(pane);
    setMobilePane(pane);
    setFullscreenError(false);
    try {
      if (!container.current?.requestFullscreen)
        throw new Error("fullscreen_unavailable");
      await container.current.requestFullscreen();
    } catch {
      setFullscreenError(true);
    }
  }
  async function closeFullscreen() {
    setFocused(null);
    setFullscreenError(false);
    if (window.document.fullscreenElement)
      await window.document.exitFullscreen();
  }
  function selectEntity(entityId: string, observationId?: string) {
    onSelect(entityId, observationId);
    const entity = document.entities.find((e) => e.id === entityId);
    if (!entity) return;
    const observations = observationsFor(document, entity);
    const observation =
      observations.find((o) => o.id === observationId) ||
      observations.find((o) => o.imageId === imageId) ||
      observations[0];
    if (observation && observation.imageId !== imageId) {
      const nextCamera = document.cameras.find(
        (c) => c.imageId === observation.imageId,
      );
      onCamera(observation.imageId, nextCamera?.id || null);
    }
  }
  return (
    <section
      ref={container}
      className={`report-scene ${focused ? "report-scene-focused" : ""}`}
      data-mobile-pane={mobilePane}
      data-focused-pane={focused || ""}
    >
      <div className="report-scene-toolbar">
        <div className="report-scene-intro">
          <strong>{t("sceneLinked")}</strong>
          <span>
            {selected
              ? `${t("sceneSelected")} · ${selected.label || selected.id}`
              : t("sceneReadOnly")}
          </span>
        </div>
        <div className="report-scene-controls">
          <label>
            <span>{t("sceneLayers")}</span>
            <select
              aria-label={t("sceneLayers")}
              value={layer}
              onChange={(e) => setLayer(e.target.value as Layer)}
            >
              <option value="model">{t("sceneModel")}</option>
              <option value="observed_surface">{t("sceneObserved")}</option>
              <option value="point_cloud" disabled={!hasPointCloud}>
                {t(hasPointCloud ? "scenePoints" : "sceneNoPoints")}
              </option>
            </select>
          </label>
          <label className="report-scene-check">
            <input
              type="checkbox"
              checked={allBounds}
              onChange={(e) => setAllBounds(e.target.checked)}
            />
            {t("sceneAllBounds")}
          </label>
          {focused && (
            <button onClick={closeFullscreen}>
              {t("sceneCloseFullscreen")} ×
            </button>
          )}
        </div>
      </div>
      {fullscreenError && (
        <p className="report-scene-notice" role="status">
          {t("sceneFullscreenUnavailable")}
        </p>
      )}
      {hasCandidates && (
        <p className="report-scene-notice">{t("sceneCandidateNotice")}</p>
      )}
      <div
        className="report-scene-mobile-tabs"
        role="tablist"
        aria-label={t("sceneLinked")}
      >
        {paneOrder.map((pane, index) => (
          <button
            key={pane}
            role="tab"
            tabIndex={mobilePane === pane ? 0 : -1}
            onKeyDown={(event) => {
              if (
                !["ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)
              )
                return;
              event.preventDefault();
              const next =
                event.key === "Home"
                  ? 0
                  : event.key === "End"
                    ? paneOrder.length - 1
                    : (index +
                        (event.key === "ArrowRight" ? 1 : -1) +
                        paneOrder.length) %
                      paneOrder.length;
              setMobilePane(paneOrder[next]);
              if (focused) setFocused(paneOrder[next]);
              event.currentTarget.parentElement
                ?.querySelectorAll<HTMLButtonElement>("button")
                [next]?.focus();
            }}
            aria-selected={mobilePane === pane}
            aria-controls={`${panePrefix}-${pane}`}
            onClick={() => {
              setMobilePane(pane);
              if (focused) setFocused(pane);
            }}
          >
            {t(paneNames[pane])}
          </button>
        ))}
      </div>
      <div className="report-scene-grid">
        {paneOrder.map((pane, index) => (
          <section
            className="report-scene-pane"
            id={`${panePrefix}-${pane}`}
            data-pane={pane}
            key={pane}
            aria-label={t(paneNames[pane])}
          >
            <header>
              <h3>
                <span>{String(index + 1).padStart(2, "0")}</span>
                {t(pane === "spatial" ? availability.spatialTitle : paneNames[pane])}
              </h3>
              {pane === "cad" && onOpenSourceCad && <button className="report-scene-source-cad" onClick={onOpenSourceCad}>{t("sceneSourceCad")} ↗</button>}
              <button
                className="report-scene-expand"
                aria-label={`${t("sceneFullscreen")} · ${t(paneNames[pane])}`}
                title={t("sceneFullscreen")}
                onClick={() => fullscreen(pane)}
              >
                ⛶
              </button>
            </header>
            <div className="report-scene-pane-body">
              {pane === "photo" && (
                <>
                  <PhotoView
                    document={document}
                    imageId={imageId}
                    selectedId={selection.entityId}
                    onSelect={selectEntity}
                    draw={draw}
                    onBox={onBox}
                  />
                  {camera && (
                    <PhotoAxes
                      revision={revision}
                      camera={camera}
                      layer={layer}
                      selectedId={selection.entityId}
                      allBounds={allBounds}
                    />
                  )}
                </>
              )}
              {pane === "spatial" && (
                <>
                  <SpatialView
                    revision={revision}
                    selection={selection}
                    onSelect={selectEntity}
                    onCommit={noEdit}
                    mode="free"
                    cameraId={cameraId}
                    layers={{
                      observed_surface: layer !== "point_cloud",
                      generated_mesh: layer === "model",
                      primitive: layer === "model",
                      point_cloud: layer === "point_cloud",
                      allBounds,
                      showCandidates: true,
                      editable: false,
                      opacity: 1,
                    }}
                  />
                  {!hasRepresentation && (
                    <div className="report-scene-stage-note">
                      {t("sceneNoRepresentation")}
                    </div>
                  )}
                </>
              )}
              {(pane === "cad" || pane === "plan") && availability.planEmpty && (
                <div className="report-scene-plan-empty" role="status">
                  <strong>{t("scenePlanUnavailable")}</strong>
                  <p>{t(availability.planEmpty)}</p>
                  <small>{t("sceneSelectionRetained")}</small>
                </div>
              )}
              {pane === "cad" && !availability.planEmpty && (
                <PlanView
                  document={document}
                  selectedId={selection.entityId}
                  onSelect={selectEntity}
                />
              )}
              {pane === "plan" && !availability.planEmpty && (
                <PlanView
                  document={document}
                  selectedId={selection.entityId}
                  onSelect={selectEntity}
                  interactive
                />
              )}
            </div>
            {pane === "photo" && (
              <footer className="report-scene-photo-switch">
                {images.map((image, i) => (
                  <button
                    key={image.imageId}
                    aria-pressed={imageId === image.imageId}
                    onClick={() => onCamera(image.imageId, image.cameraId)}
                  >
                    {t("scenePhotoNumber")} {i + 1}
                  </button>
                ))}
                {draw && <span>{t("sceneDrawActive")}</span>}
              </footer>
            )}
          </section>
        ))}
      </div>
      {selected && (
        <p className="report-scene-selection-note">
          <strong>
            {t("sceneSelected")} · {selected.label || selected.id}
          </strong>
          <span>
            {isReferenceSurface(document, selected)
              ? t("sceneReferenceSurface")
              : selectedOverlay
              ? t(
                  selectedOverlay.axisSpace === "native"
                    ? "sceneNativeAxis"
                    : "sceneSourceAxis",
                )
              : !entityGeometry(selected, layer)
                ? t("sceneNoGeometrySelection")
                : t("sceneNoPhotoAxes")}
          </span>
        </p>
      )}
      <details className="report-scene-objects">
        <summary>
          {t("sceneObjects")} <span>{objects.length}</span>
          {selected && <small>{selected.label || selected.id}</small>}
        </summary>
        <div className="report-scene-object-content">
          <input
            type="search"
            aria-label={t("sceneSearch")}
            placeholder={t("sceneSearch")}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <div className="report-scene-object-list">
            {filtered.map((entity) => (
              <button
                key={entity.id}
                aria-pressed={entity.id === selection.entityId}
                onClick={() => selectEntity(entity.id)}
              >
                <strong>{entity.label || entity.id}</strong>
                <span>
                  {observationsFor(document, entity).length}{" "}
                  {t("sceneEvidenceCount")} · {entity.id.slice(0, 6)}
                  {!(entity.representations || []).length
                    ? ` · ${t("sceneNoGeometry")}`
                    : (entity.representations || []).some(
                          (r) => r.placementState === "unconfirmed",
                        )
                      ? ` · ${t("sceneCandidate")}`
                      : ""}
                </span>
              </button>
            ))}
          </div>
          {!filtered.length && (
            <p>{t(objects.length ? "sceneNoMatches" : "sceneNoObjects")}</p>
          )}
        </div>
      </details>
    </section>
  );
}
