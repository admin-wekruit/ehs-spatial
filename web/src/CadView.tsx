import { useEffect, useMemo, useRef, useState } from "react";
import { cameraPath, planHits, planShapes, planPolygonPath } from "./core";
import { useI18n } from "./i18n";
import { isReferenceSurface } from "./scene-semantics";
import type { SceneDocument } from "./types";
import type { SceneMeasurement, MeasureRegion } from "./SpatialMeasurements";
import "./cad-view.css";

type Shape = ReturnType<typeof planShapes>[number];
type CadCamera = { center: number[]; scale: number; turn?: boolean };
type CadSize = { width: number; height: number };

export function cadFit(shapes: Shape[], { width, height }: CadSize, horizontalPadding = Math.min(24, width * .06), extra: number[][] = [], turn?: boolean): CadCamera {
  if (!shapes.length && !extra.length) return { center: [0, 0], scale: 1 };
  // A few far footprints (floor or wall depth near the vanishing point of a long aisle) shrank everything else to a
  // line: with many objects, fit their central 94% plus the camera path instead of the extreme corners.
  const robust = shapes.length >= 12, quantile = (values: number[], q: number) => [...values].sort((a, b) => a - b)[Math.round(q * (values.length - 1))];
  const min = [0, 1].map(k => Math.min(...(robust ? [quantile(shapes.map(shape => shape.min[k]), .03)] : shapes.map(shape => shape.min[k])), ...extra.map(p => p[k])));
  const max = [0, 1].map(k => Math.max(...(robust ? [quantile(shapes.map(shape => shape.max[k]), .97)] : shapes.map(shape => shape.max[k])), ...extra.map(p => p[k])));
  const across = Math.max(1, width - Math.min(horizontalPadding, width * .4)), down = Math.max(1, height - Math.min(24, height * .06));
  const [du, dv] = [0, 1].map(k => Math.max(max[k] - min[k], 1e-6));
  // Keep all contour extrema visible with a small margin that shrinks on short panes. A long aisle along V in a wide,
  // short pane is drawn turned a quarter (V across, U up) when that shows it larger; the grid stays axis-aligned.
  const straight = Math.min(across / du, down / dv), turned = Math.min(across / dv, down / du);
  const quarter = turn ?? turned > straight * 1.15;
  return { center: min.map((value, k) => (value + max[k]) / 2), scale: quarter ? turned : straight, turn: quarter };
}

/** Plan offset from the view centre -> screen axes (right, up); a turned view shows V across and U up. */
const toScreenAxes = (d: number[], turn?: boolean) => turn ? [d[1], -d[0]] : d;
const toPlanAxes = (e: number[], turn?: boolean) => turn ? [-e[1], e[0]] : e;

export function cadScreen(p: number[], camera: CadCamera, size: CadSize) {
  const e = toScreenAxes([p[0] - camera.center[0], p[1] - camera.center[1]], camera.turn);
  return [size.width / 2 + e[0] * camera.scale, size.height / 2 - e[1] * camera.scale];
}

export function cadWorld(p: number[], camera: CadCamera, size: CadSize) {
  const d = toPlanAxes([(p[0] - size.width / 2) / camera.scale, -(p[1] - size.height / 2) / camera.scale], camera.turn);
  return [camera.center[0] + d[0], camera.center[1] + d[1]];
}

/** Screen box of a plan box, whichever way the view is turned: [left, top], [right, bottom]. */
function screenBox(min: number[], max: number[], camera: CadCamera, size: CadSize) {
  const corners = [[min[0], min[1]], [max[0], min[1]], [min[0], max[1]], [max[0], max[1]]].map(p => cadScreen(p, camera, size));
  return [[Math.min(...corners.map(c => c[0])), Math.min(...corners.map(c => c[1]))], [Math.max(...corners.map(c => c[0])), Math.max(...corners.map(c => c[1]))]];
}

/** A pan or arrow step given along the screen (right, up) as a plan offset. */
const planStep = (right: number, up: number, camera: CadCamera) => toPlanAxes([right / camera.scale, up / camera.scale], camera.turn);

export function cadStep(target: number) {
  const exponent = 10 ** Math.floor(Math.log10(Math.max(target, 1e-12)));
  return [1, 2, 5, 10].find(value => value * exponent >= target)! * exponent;
}

export function cadNumber(value: number) {
  return Number(value.toPrecision(4)).toString();
}

export function cadCallouts(shapes: Shape[], selectedId: string | null, camera: CadCamera, size: CadSize) {
  const labels: { id: string; anchor: number[]; position: number[] }[] = [];
  // ponytail: bounded O(n²) label collision checks suit the current object lists;
  // use a spatial index if scenes grow to thousands of independently labeled parts.
  const ordered = [...shapes].sort((a, b) => Number(b.entity.id === selectedId) - Number(a.entity.id === selectedId));
  for (const shape of ordered) {
    const [lo, hi] = screenBox(shape.min, shape.max, camera, size);
    if (hi[0] < 0 || hi[1] < 0 || lo[0] > size.width || lo[1] > size.height) continue;
    const anchor = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
    const candidates = [[0, 0], [24, 0], [-24, 0], [0, -24], [0, 24], [24, -24], [-24, -24], [24, 24], [-24, 24]];
    for (const [dx, dy] of candidates) {
      const position = [Math.max(22, Math.min(size.width - 22, anchor[0] + dx)), Math.max(16, Math.min(size.height - 38, anchor[1] + dy))];
      if (labels.some(label => Math.abs(label.position[0] - position[0]) < 34 && Math.abs(label.position[1] - position[1]) < 24)) continue;
      labels.push({ id: shape.entity.id, anchor, position });
      break;
    }
  }
  return labels;
}

export function CadView({ document, selectedId, onSelect, geometryOptions, measurement, region, drawingRegion, onRegion, showPath = true }: {
  showPath?: boolean;
  measurement?: SceneMeasurement | null;
  region?: MeasureRegion | null;
  drawingRegion?: boolean;
  onRegion?: (region: MeasureRegion | null) => void;
  document: SceneDocument;
  selectedId: string | null;
  onSelect: (id: string) => void;
  geometryOptions?: Parameters<typeof planShapes>[1];
}) {
  const { t: T } = useI18n(), t = (key: string) => T("cad." + key);  // this view's ids live under cad.*
  const stage = useRef<HTMLDivElement>(null), svg = useRef<SVGSVGElement>(null), picker = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<CadSize>({ width: 600, height: 400 });
  const [camera, setCamera] = useState<CadCamera | null>(null), [candidates, setCandidates] = useState<string[]>([]);
  const [regionStart, setRegionStart] = useState<number[] | null>(null), [regionCursor, setRegionCursor] = useState<number[]>([0, 0]);
  const [focusMode, setFocusMode] = useState(false);
  const pointer = useRef<{ id: number; x: number; y: number; camera: CadCamera; moved: boolean } | null>(null);
  const suppressClick = useRef(false);
  const shapes = useMemo(() => planShapes(document, geometryOptions), [document, geometryOptions?.scope, geometryOptions?.frameId, geometryOptions?.layer, geometryOptions?.imageId, geometryOptions?.showCandidates]);
  const entities = document.entities.filter(entity => !entity.sourceContext);
  const numbers = new Map(entities.map((entity, index) => [entity.id, String(index + 1).padStart(2, "0")]));
  const selected = shapes.find(shape => shape.entity.id === selectedId);
  const selectedEntity = entities.find(entity => entity.id === selectedId);
  const frame = document.coordinateFrames.find(frame => frame.id === (geometryOptions?.frameId || shapes[0]?.coordinateFrameId));
  const units = `${t("units")}${frame?.scale?.status === "uncalibrated" ? ` · ${t("uncalibrated")}` : ""}`;
  // The walk, in the same plan projection as the objects; the dot follows the report's video time.
  const path = useMemo(() => showPath && shapes[0] ? cameraPath(document, shapes[0].coordinateFrameId).map(p => ({ ...p,
    plan: shapes[0].nativeToPlane.slice(0, 2).map(row => row[3] + row[0] * p.position[0] + row[1] * p.position[1] + row[2] * p.position[2]) })) : [], [document, shapes, showPath]);
  const [videoTime, setVideoTime] = useState<number | null>(null);
  useEffect(() => { const follow = (event: Event) => setVideoTime((event as CustomEvent<number>).detail); window.addEventListener("panoptes:video-time", follow); return () => window.removeEventListener("panoptes:video-time", follow); }, []);
  const now = videoTime === null || !path.length || path[0].time === null ? null : path.reduce((best, p) => Math.abs(p.time! - videoTime) < Math.abs(best.time! - videoTime) ? p : best);
  const fit = cadFit(shapes, size, undefined, path.map(p => p.plan)), view = camera || fit;
  const callouts = cadCallouts(focusMode ? shapes.filter(shape => shape.entity.id === selectedId) : shapes, selectedId, view, size);
  const sorted = [...shapes].sort((a, b) => {
    if (a.entity.id === selectedId) return 1;
    if (b.entity.id === selectedId) return -1;
    return Number(isReferenceSurface(document, b.entity)) - Number(isReferenceSurface(document, a.entity)) ||
      (b.max[0] - b.min[0]) * (b.max[1] - b.min[1]) - (a.max[0] - a.min[0]) * (a.max[1] - a.min[1]);
  });

  useEffect(() => {
    const node = stage.current;
    if (!node) return;
    const observer = new ResizeObserver(entries => {
      const rect = entries[0]?.contentRect;
      if (rect && rect.width > 0 && rect.height > 0) setSize({ width: rect.width, height: rect.height });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  useEffect(() => { setCamera(null); setFocusMode(false); setCandidates([]); pointer.current = null; }, [document.captureId, geometryOptions?.frameId]);
  useEffect(() => { setCandidates([]); }, [selectedId, document, geometryOptions?.scope, geometryOptions?.layer, geometryOptions?.imageId]);
  useEffect(() => { if (candidates.length) picker.current?.querySelector<HTMLButtonElement>("button[data-candidate]")?.focus(); }, [candidates]);

  useEffect(() => { setRegionStart(null); if (drawingRegion) { setRegionCursor(view.center); svg.current?.focus(); } }, [drawingRegion]);
  function regionCorner(p: number[]) {
    const basis = shapes[0];
    if (!basis || !onRegion) return;
    if (!regionStart) { setRegionStart(p); setRegionCursor(p); }
    else if (Math.abs(p[0] - regionStart[0]) > 1e-8 && Math.abs(p[1] - regionStart[1]) > 1e-8) {
      onRegion({ coordinateFrameId: basis.coordinateFrameId, nativeToPlane: basis.nativeToPlane,
        points: [regionStart, [p[0], regionStart[1]], p, [regionStart[0], p[1]]] }); setRegionStart(null);
    }
  }
  const measurementLines = measurement && measurement.coordinateFrameId === frame?.id && shapes[0] ? measurement.lines.map(line => ({ ...line, screen: line.points.map(p => cadScreen(shapes[0].nativeToPlane.slice(0, 2).map(row => row[3] + row[0]*p[0] + row[1]*p[1] + row[2]*p[2]), view, size)) })) : [];
  function choose(id: string) { setCandidates([]); onSelect(id); svg.current?.focus(); }
  function zoom(factor: number) {
    setCamera({ ...view, scale: Math.max(fit.scale / 4, Math.min(fit.scale * 1000, view.scale * factor)) });
    setCandidates([]);
  }
  function focus() {
    if (!selected) return;
    setCamera(cadFit([selected], size, 160, [], view.turn));
    setFocusMode(true);
    setCandidates([]);
  }
  // Grid lines run along the screen: across the screen is U (V when turned), up the screen is V (U when turned).
  const across = view.turn ? 1 : 0, up = view.turn ? 0 : 1;
  const screenCorners = [cadWorld([0, 0], view, size), cadWorld([size.width, size.height], view, size)];
  const range = (k: number) => [Math.min(screenCorners[0][k], screenCorners[1][k]), Math.max(screenCorners[0][k], screenCorners[1][k])];
  const step = cadStep(70 / view.scale), scaleLength = cadStep(60 / view.scale);
  const gridLines = (k: number) => { const [a, b] = range(k); return Array.from({ length: Math.min(100, Math.ceil((b - a) / step) + 1) }, (_, index) => (Math.ceil(a / step) + index) * step); };
  const gridX = gridLines(across), gridY = gridLines(up);
  const at = (k: number, value: number) => { const p = [...view.center]; p[k] = value; return p; };
  const [lo, hi] = selected ? screenBox(selected.min, selected.max, view, size) : [null, null];
  const acrossName = view.turn ? "V" : "U", upName = view.turn ? "U" : "V";
  const source = (shape: Shape) => isReferenceSurface(document, shape.entity) ? "floor" : shape.projectionSource === "saved_hull" ? "hull" : shape.geometryKind === "model" ? "model" : "observed";

  return <div className="cad-view" onKeyDown={event => {
    if (event.key === "Escape" && drawingRegion) { event.preventDefault(); event.stopPropagation(); onRegion?.(null); return; }
    if (event.key === "Escape" && candidates.length) { event.preventDefault(); event.stopPropagation(); setCandidates([]); svg.current?.focus(); }
  }}>
    <div className="cad-tools">
      <span>{shapes.length} / {entities.filter(entity => entity.visible !== false).length} {t("projected")}</span>
      <div><button type="button" onClick={() => { setCamera(null); setFocusMode(false); setCandidates([]); }}>{t("fit")}</button>
        <button type="button" disabled={!selected} onClick={focus}>{t("focus")}</button>
        <button type="button" aria-label={t("zoomOut")} onClick={() => zoom(1 / 1.4)}>−</button>
        <button type="button" aria-label={t("zoomIn")} onClick={() => zoom(1.4)}>+</button></div>
    </div>
    {drawingRegion && <div className="cad-measure-guide" role="status">{t("drawGuide")}</div>}
    <div className="cad-stage" ref={stage}>
      {!shapes.length ? <p className="cad-empty" role="status">{t("empty")}</p> : <svg ref={svg} role="group" tabIndex={0} viewBox={`0 0 ${size.width} ${size.height}`} aria-label={t("title")} data-frame-id={shapes[0]?.coordinateFrameId} data-cad-shapes={shapes.length}
        onKeyDown={event => {
          if (drawingRegion) {
            if (event.key === "Enter") { event.preventDefault(); event.stopPropagation(); regionCorner(regionCursor); return; }
            const delta = { ArrowLeft: [-10, 0], ArrowRight: [10, 0], ArrowUp: [0, 10], ArrowDown: [0, -10] }[event.key];
            if (delta) { event.preventDefault(); event.stopPropagation(); const d = planStep(delta[0], delta[1], view); setRegionCursor(p => p.map((v, i) => v + d[i])); return; }
          }
          if (event.target !== event.currentTarget) return;
          if (["+", "=", "-"].includes(event.key)) { event.preventDefault(); event.stopPropagation(); zoom(event.key === "-" ? 1 / 1.4 : 1.4); }
          const delta = { ArrowLeft: [-40, 0], ArrowRight: [40, 0], ArrowUp: [0, 40], ArrowDown: [0, -40] }[event.key];
          if (delta) { event.preventDefault(); event.stopPropagation(); const d = planStep(delta[0], delta[1], view); setCamera({ ...view, center: view.center.map((value, k) => value + d[k]) }); }
        }}
        onPointerDown={event => {
          if (event.button !== 0 || drawingRegion) return;
          suppressClick.current = false;
          pointer.current = { id: event.pointerId, x: event.clientX, y: event.clientY, camera: view, moved: false };
        }}
        onPointerMove={event => {
          if (drawingRegion) {
            const matrix = svg.current?.getScreenCTM();
            if (matrix) { const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse()); setRegionCursor(cadWorld([p.x, p.y], view, size)); } return;
          }
          const start = pointer.current;
          if (!start || start.id !== event.pointerId) return;
          if (event.pointerType !== "touch" && !(event.buttons & 1)) { pointer.current = null; return; }
          const dx = event.clientX - start.x, dy = event.clientY - start.y;
          if (!start.moved && Math.hypot(dx, dy) > 4) { start.moved = true; event.currentTarget.setPointerCapture(event.pointerId); }
          if (start.moved) { setCandidates([]); const d = planStep(-dx, dy, start.camera); setCamera({ ...start.camera, center: [start.camera.center[0] + d[0], start.camera.center[1] + d[1]] }); }
        }}
        onPointerUp={event => {
          suppressClick.current = !!pointer.current?.moved;
          pointer.current = null;
          if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={() => { pointer.current = null; suppressClick.current = true; }}
        onClick={event => {
          if (suppressClick.current) { suppressClick.current = false; return; }
          const marker = (event.target as Element).closest("[data-cad-label]")?.getAttribute("data-cad-label");
          if (marker && !drawingRegion) { choose(marker); return; }
          const matrix = svg.current?.getScreenCTM();
          if (!matrix) return;
          const local = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
          const world = cadWorld([local.x, local.y], view, size), hits = planHits(shapes, world[0], world[1], 4 / view.scale);
          if (drawingRegion) { regionCorner(world); return; }
          if (hits.length === 1) choose(hits[0].entity.id);
          else setCandidates(hits.map(hit => hit.entity.id));
        }}>
        <g className="cad-grid" aria-hidden="true">
          {gridX.map(value => { const x = cadScreen(at(across, value), view, size)[0]; return <g key={`u-${value}`}><line x1={x} x2={x} y1={0} y2={size.height} /><text x={x + 3} y={size.height - 5}>{cadNumber(value)}</text></g>; })}
          {gridY.map(value => { const y = cadScreen(at(up, value), view, size)[1]; return <g key={`v-${value}`}><line x1={0} x2={size.width} y1={y} y2={y} /><text x={4} y={y - 3}>{cadNumber(value)}</text></g>; })}
        </g>
        <g className="cad-geometry" data-selection={!!selected}>
          {sorted.map(shape => <g key={shape.entity.id}
            data-cad-entity={shape.entity.id} data-cad-source={shape.projectionSource} className={`cad-object cad-${source(shape)}${shape.entity.id === selectedId ? " is-selected" : ""}`}>
            {shape.polygons.map((polygon, index) => <path key={"area-" + index} d={planPolygonPath(polygon, p => cadScreen(p, view, size))} fillRule="evenodd" />)}
            {shape.lines.map((line, index) => <path key={"line-" + index} d={"M " + line.map(p => cadScreen(p, view, size).join(",")).join(" L ")} style={{fill: "none"}} />)}
            <title>{`#${numbers.get(shape.entity.id)} ${shape.entity.label || shape.entity.id} · ${t(source(shape))}`}</title>
          </g>)}
        </g>
        {selected && lo && hi && <g className="cad-dimensions" aria-hidden="true">
          {hi[0] - lo[0] > 52 && lo[0] > 8 && hi[0] < size.width - 8 && hi[1] > 0 && hi[1] < size.height - 48 && <g>
            <path d={`M${lo[0]},${hi[1] + 6}v20 M${hi[0]},${hi[1] + 6}v20 M${lo[0]},${hi[1] + 19}H${hi[0]}`} />
            <text x={(lo[0] + hi[0]) / 2} y={hi[1] + 15} textAnchor="middle">{`Δ${acrossName} ${cadNumber(selected.max[across] - selected.min[across])}`}</text>
          </g>}
          {hi[1] - lo[1] > 52 && lo[1] > 8 && hi[1] < size.height - 8 && hi[0] > 0 && hi[0] < size.width - 58 && <g>
            <path d={`M${hi[0] + 6},${lo[1]}h20 M${hi[0] + 6},${hi[1]}h20 M${hi[0] + 19},${lo[1]}V${hi[1]}`} />
            <text transform={`translate(${hi[0] + 32},${(lo[1] + hi[1]) / 2}) rotate(-90)`} textAnchor="middle">{`Δ${upName} ${cadNumber(selected.max[up] - selected.min[up])}`}</text>
          </g>}
        </g>}
        {path.length > 1 && <g className="cad-camera-path" pointerEvents="none" aria-label={t("cameraPath")}>
          <polyline points={path.map(p => cadScreen(p.plan, view, size).join(",")).join(" ")} fill="none" stroke="#e8900c" strokeWidth={2.5} strokeLinejoin="round" />
          {[[path[0], "#2fbf71"], [path[path.length - 1], "#e5484d"]].map(([p, fill], index) => { const [x, y] = cadScreen((p as typeof path[number]).plan, view, size); return <circle key={index} cx={x} cy={y} r={5} fill={fill as string} stroke="#fff" strokeWidth={2} />; })}
          {now && (() => { const [x, y] = cadScreen(now.plan, view, size); return <circle cx={x} cy={y} r={7} fill="#e8900c" stroke="#fff" strokeWidth={2} />; })()}
        </g>}
        <g className="cad-callouts">
          {callouts.map(label => <g key={label.id} className={label.id === selectedId ? "is-selected" : ""}>
            <path d={`M${label.anchor.join(",")}L${label.position.join(",")}`} />
            <g data-cad-label={label.id} role="button" tabIndex={0} aria-label={`#${numbers.get(label.id)} ${entities.find(entity => entity.id === label.id)?.label || label.id}`} aria-pressed={label.id === selectedId}
              onKeyDown={event => { if (drawingRegion) return; if (["Enter", " "].includes(event.key)) { event.preventDefault(); event.stopPropagation(); choose(label.id); } }}>
              <rect x={label.position[0] - 14} y={label.position[1] - 10} width={28} height={20} rx={3} />
              <text x={label.position[0]} y={label.position[1] + 3.5} textAnchor="middle">{numbers.get(label.id)}</text>
            </g>
          </g>)}
        </g>
        <g className="cad-measurement" pointerEvents="none" fill="none" strokeWidth={2.5}>
          {!drawingRegion && region?.coordinateFrameId === frame?.id && region && <polygon points={region.points.map(p => cadScreen(p, view, size).join(",")).join(" ")} stroke="#168bba" fill="#168bba11" />}
          {measurementLines.map((line, i) => <polyline key={i} points={line.screen.map(p => p.join(",")).join(" ")} stroke={line.color} />)}
          {drawingRegion && (() => { const a = cadScreen(regionStart || regionCursor, view, size), b = cadScreen(regionCursor, view, size); return <g stroke="#168bba">
            <path d={`M${b[0]-10},${b[1]}h20 M${b[0]},${b[1]-10}v20`} />
            {regionStart && <rect x={Math.min(a[0], b[0])} y={Math.min(a[1], b[1])} width={Math.abs(a[0]-b[0])} height={Math.abs(a[1]-b[1])} fill="#168bba22" />}
          </g>; })()}
        </g>
        <g className="cad-scale" transform={`translate(16,${size.height - 28})`} aria-label={`${cadNumber(scaleLength)} ${units}`}>
          <rect x={-5} y={-22} width={Math.max(96, scaleLength * view.scale + 12)} height={31} />
          <path d={`M0,-4v8 M0,0H${scaleLength * view.scale} M${scaleLength * view.scale},-4v8`} />
          <text x={0} y={-9}>{cadNumber(scaleLength)} · {units}</text>
        </g>
      </svg>}
      {candidates.length > 1 && <div className="cad-hit-picker" ref={picker} role="group" aria-label={t("overlap")}>
        <header><strong>{t("overlap")} · {candidates.length}</strong><button type="button" aria-label={t("close")} onClick={() => { setCandidates([]); svg.current?.focus(); }}>×</button></header>
        <div>{candidates.map(id => <button type="button" key={id} data-candidate={id} aria-pressed={id === selectedId} onClick={() => choose(id)}><b>{numbers.get(id)}</b>{entities.find(entity => entity.id === id)?.label || id}</button>)}</div>
      </div>}
    </div>
    <div className="cad-selection" role="status">{selected ? <><strong>#{numbers.get(selected.entity.id)} {selected.entity.label || selected.entity.id}</strong><span>{t("width")} {cadNumber(selected.max[0] - selected.min[0])} · {t("depth")} {cadNumber(selected.max[1] - selected.min[1])} · {units}</span></> : <span>{t(selectedEntity ? "missing" : "noSelection")}</span>}</div>
    <footer className="cad-legend">{["observed", "model", "hull", "floor"].filter(kind => shapes.some(shape => source(shape) === kind)).map(kind => <span key={kind} className={`cad-key-${kind}`}>{t(kind)}</span>)}{path.length > 1 && <span className="cad-key-path">{t("cameraPath")}</span>}<div>{units}</div>
      <details><summary aria-label={t("bounds")}>ⓘ</summary><p>{t("bounds")} {t("axes")}<br />{t("numbers")}<br />{t("panHint")}</p></details>
    </footer>
  </div>;
}
