import { useEffect, useMemo, useRef, useState } from "react";
import { planHits, planShapes, planPolygonPath } from "./core";
import { useI18n } from "./i18n";
import { isReferenceSurface } from "./scene-semantics";
import type { SceneDocument } from "./types";
import { cadViewMessages } from "./cad-view-messages";
import "./cad-view.css";

type Shape = ReturnType<typeof planShapes>[number];
type CadCamera = { center: number[]; scale: number };
type CadSize = { width: number; height: number };

export function cadFit(shapes: Shape[], { width, height }: CadSize, horizontalPadding = Math.min(24, width * .06)): CadCamera {
  if (!shapes.length) return { center: [0, 0], scale: 1 };
  const min = [0, 1].map(k => Math.min(...shapes.map(shape => shape.min[k])));
  const max = [0, 1].map(k => Math.max(...shapes.map(shape => shape.max[k])));
  return { center: min.map((value, k) => (value + max[k]) / 2), scale: Math.min(
    Math.max(1, width - Math.min(horizontalPadding, width * .4)) / Math.max(max[0] - min[0], 1e-6),
    // Keep all contour extrema visible with a small margin that shrinks on short panes.
    Math.max(1, height - Math.min(24, height * .06)) / Math.max(max[1] - min[1], 1e-6),
  ) };
}

export function cadScreen(p: number[], camera: CadCamera, size: CadSize) {
  return [size.width / 2 + (p[0] - camera.center[0]) * camera.scale,
    size.height / 2 - (p[1] - camera.center[1]) * camera.scale];
}

export function cadWorld(p: number[], camera: CadCamera, size: CadSize) {
  return [camera.center[0] + (p[0] - size.width / 2) / camera.scale,
    camera.center[1] - (p[1] - size.height / 2) / camera.scale];
}

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
    const lo = cadScreen([shape.min[0], shape.max[1]], camera, size);
    const hi = cadScreen([shape.max[0], shape.min[1]], camera, size);
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

export function CadView({ document, selectedId, onSelect, geometryOptions }: {
  document: SceneDocument;
  selectedId: string | null;
  onSelect: (id: string) => void;
  geometryOptions?: Parameters<typeof planShapes>[1];
}) {
  const { language } = useI18n();
  const t = (key: string) => cadViewMessages[key]?.[language === "zh" ? 0 : 1] || key;
  const stage = useRef<HTMLDivElement>(null), svg = useRef<SVGSVGElement>(null), picker = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<CadSize>({ width: 600, height: 400 });
  const [camera, setCamera] = useState<CadCamera | null>(null), [candidates, setCandidates] = useState<string[]>([]);
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
  const fit = cadFit(shapes, size), view = camera || fit;
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

  function choose(id: string) { setCandidates([]); onSelect(id); svg.current?.focus(); }
  function zoom(factor: number) {
    setCamera({ ...view, scale: Math.max(fit.scale / 4, Math.min(fit.scale * 1000, view.scale * factor)) });
    setCandidates([]);
  }
  function focus() {
    if (!selected) return;
    setCamera(cadFit([selected], size, 160));
    setFocusMode(true);
    setCandidates([]);
  }
  const topLeft = cadWorld([0, 0], view, size), bottomRight = cadWorld([size.width, size.height], view, size);
  const step = cadStep(70 / view.scale), scaleLength = cadStep(60 / view.scale);
  const gridX = Array.from({ length: Math.min(100, Math.ceil((bottomRight[0] - topLeft[0]) / step) + 1) }, (_, index) => (Math.ceil(topLeft[0] / step) + index) * step);
  const gridY = Array.from({ length: Math.min(100, Math.ceil((topLeft[1] - bottomRight[1]) / step) + 1) }, (_, index) => (Math.ceil(bottomRight[1] / step) + index) * step);
  const lo = selected && cadScreen([selected.min[0], selected.max[1]], view, size);
  const hi = selected && cadScreen([selected.max[0], selected.min[1]], view, size);
  const source = (shape: Shape) => isReferenceSurface(document, shape.entity) ? "floor" : shape.projectionSource === "saved_hull" ? "hull" : shape.geometryKind === "model" ? "model" : "observed";

  return <div className="cad-view" onKeyDown={event => {
    if (event.key === "Escape" && candidates.length) { event.preventDefault(); event.stopPropagation(); setCandidates([]); svg.current?.focus(); }
  }}>
    <div className="cad-tools">
      <span>{shapes.length} / {entities.filter(entity => entity.visible !== false).length} {t("projected")}</span>
      <div><button type="button" onClick={() => { setCamera(null); setFocusMode(false); setCandidates([]); }}>{t("fit")}</button>
        <button type="button" disabled={!selected} onClick={focus}>{t("focus")}</button>
        <button type="button" aria-label={t("zoomOut")} onClick={() => zoom(1 / 1.4)}>−</button>
        <button type="button" aria-label={t("zoomIn")} onClick={() => zoom(1.4)}>＋</button></div>
    </div>
    <div className="cad-stage" ref={stage}>
      {!shapes.length ? <p className="cad-empty" role="status">{t("empty")}</p> : <svg ref={svg} role="group" tabIndex={0} viewBox={`0 0 ${size.width} ${size.height}`} aria-label={t("title")} data-frame-id={shapes[0]?.coordinateFrameId} data-cad-shapes={shapes.length}
        onKeyDown={event => {
          if (event.target !== event.currentTarget) return;
          if (["+", "=", "-"].includes(event.key)) { event.preventDefault(); event.stopPropagation(); zoom(event.key === "-" ? 1 / 1.4 : 1.4); }
          const delta = { ArrowLeft: [-40, 0], ArrowRight: [40, 0], ArrowUp: [0, 40], ArrowDown: [0, -40] }[event.key];
          if (delta) { event.preventDefault(); event.stopPropagation(); setCamera({ ...view, center: view.center.map((value, k) => value + delta[k] / view.scale) }); }
        }}
        onPointerDown={event => {
          if (event.button !== 0) return;
          suppressClick.current = false;
          pointer.current = { id: event.pointerId, x: event.clientX, y: event.clientY, camera: view, moved: false };
        }}
        onPointerMove={event => {
          const start = pointer.current;
          if (!start || start.id !== event.pointerId) return;
          if (event.pointerType !== "touch" && !(event.buttons & 1)) { pointer.current = null; return; }
          const dx = event.clientX - start.x, dy = event.clientY - start.y;
          if (!start.moved && Math.hypot(dx, dy) > 4) { start.moved = true; event.currentTarget.setPointerCapture(event.pointerId); }
          if (start.moved) { setCandidates([]); setCamera({ scale: start.camera.scale, center: [start.camera.center[0] - dx / start.camera.scale, start.camera.center[1] + dy / start.camera.scale] }); }
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
          if (marker) { choose(marker); return; }
          const matrix = svg.current?.getScreenCTM();
          if (!matrix) return;
          const local = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
          const world = cadWorld([local.x, local.y], view, size), hits = planHits(shapes, world[0], world[1], 4 / view.scale);
          if (hits.length === 1) choose(hits[0].entity.id);
          else setCandidates(hits.map(hit => hit.entity.id));
        }}>
        <g className="cad-grid" aria-hidden="true">
          {gridX.map(value => { const x = cadScreen([value, 0], view, size)[0]; return <g key={`u-${value}`}><line x1={x} x2={x} y1={0} y2={size.height} /><text x={x + 3} y={size.height - 5}>{cadNumber(value)}</text></g>; })}
          {gridY.map(value => { const y = cadScreen([0, value], view, size)[1]; return <g key={`v-${value}`}><line x1={0} x2={size.width} y1={y} y2={y} /><text x={4} y={y - 3}>{cadNumber(value)}</text></g>; })}
        </g>
        <g className="cad-geometry" data-selection={!!selected}>
          {sorted.map(shape => <g key={shape.entity.id} role="button" tabIndex={0} aria-label={`#${numbers.get(shape.entity.id)} ${shape.entity.label || shape.entity.id}`} aria-pressed={shape.entity.id === selectedId}
            data-cad-entity={shape.entity.id} data-cad-source={shape.projectionSource} className={`cad-object cad-${source(shape)}${shape.entity.id === selectedId ? " is-selected" : ""}`}
            onKeyDown={event => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); event.stopPropagation(); choose(shape.entity.id); } }}>
            {shape.polygons.map((polygon, index) => <path key={"area-" + index} d={planPolygonPath(polygon, p => cadScreen(p, view, size))} fillRule="evenodd" />)}
            {shape.lines.map((line, index) => <path key={"line-" + index} d={"M " + line.map(p => cadScreen(p, view, size).join(",")).join(" L ")} style={{fill: "none"}} />)}
            <title>{`#${numbers.get(shape.entity.id)} ${shape.entity.label || shape.entity.id} · ${t(source(shape))}`}</title>
          </g>)}
        </g>
        {selected && lo && hi && <g className="cad-dimensions" aria-hidden="true">
          {hi[0] - lo[0] > 52 && lo[0] > 8 && hi[0] < size.width - 8 && hi[1] > 0 && hi[1] < size.height - 48 && <g>
            <path d={`M${lo[0]},${hi[1] + 6}v20 M${hi[0]},${hi[1] + 6}v20 M${lo[0]},${hi[1] + 19}H${hi[0]}`} />
            <text x={(lo[0] + hi[0]) / 2} y={hi[1] + 15} textAnchor="middle">{`ΔU ${cadNumber(selected.max[0] - selected.min[0])}`}</text>
          </g>}
          {hi[1] - lo[1] > 52 && lo[1] > 8 && hi[1] < size.height - 8 && hi[0] > 0 && hi[0] < size.width - 58 && <g>
            <path d={`M${hi[0] + 6},${lo[1]}h20 M${hi[0] + 6},${hi[1]}h20 M${hi[0] + 19},${lo[1]}V${hi[1]}`} />
            <text transform={`translate(${hi[0] + 32},${(lo[1] + hi[1]) / 2}) rotate(-90)`} textAnchor="middle">{`ΔV ${cadNumber(selected.max[1] - selected.min[1])}`}</text>
          </g>}
        </g>}
        <g className="cad-callouts">
          {callouts.map(label => <g key={label.id} data-cad-label={label.id} className={label.id === selectedId ? "is-selected" : ""}>
            <path d={`M${label.anchor.join(",")}L${label.position.join(",")}`} />
            <rect x={label.position[0] - 14} y={label.position[1] - 10} width={28} height={20} rx={3} />
            <text x={label.position[0]} y={label.position[1] + 3.5} textAnchor="middle">{numbers.get(label.id)}</text>
          </g>)}
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
    <footer className="cad-legend">{["observed", "model", "hull", "floor"].filter(kind => shapes.some(shape => source(shape) === kind)).map(kind => <span key={kind} className={`cad-key-${kind}`}>{t(kind)}</span>)}<div>{units}</div>
      <details><summary aria-label={t("bounds")}>ⓘ</summary><p>{t("bounds")} {t("axes")}<br />{t("numbers")}<br />{t("panHint")}</p></details>
    </footer>
  </div>;
}
