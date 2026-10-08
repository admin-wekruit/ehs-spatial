import { useEffect, useRef, useState } from "react";
import { resolveAsset } from "./api";
import { cameraForImage, currentEntityId, observationOwner, jsonObject } from "./core";
import { useI18n } from "./i18n";
import { hasMessage } from "./translate";
import { ReportDownload } from "./WorkcellReport";
import { ErrorNotice } from "./App";
import type { SceneDocument } from "./types";
import "./report-evidence.css";

// The offline importer normalizes historical documents to this versioned report bundle.
// Historical records retain their own run and never become current EHS findings.
type SourceRef = { assetId: string; jsonPointer?: string };
type HistoricalFinding = {
  id: string;
  title: string;
  status: string;
  summary?: string;
  predicate?: string;
  threshold?: number | null;
  unit?: string;
  violations?: unknown[];
  warnings?: unknown[];
  facts?: unknown;
  sourceRefs: SourceRef[];
};
type HistoricalPolicy = {
  id: string;
  spec?: {
    policyId: string;
    rationale?: string;
    sourceText?: string;
    unsupportedReason?: string | null;
    predicate?: string;
    threshold?: number | null;
    unit?: string;
    subjectLabels?: string[];
    objectLabels?: string[];
  };
  sourceRefs?: SourceRef[];
};
type HistoricalInventory = {
  inventoryIndex: number;
  label: string;
  sourceFrameId: string;
  source?: string;
  mappingStatus: string;
  entityIds: string[];
  heightM?: number | null;
  sizeM?: number[] | null;
  score?: number | null;
};
type InterpretationItem = {
  label?: string;
  note?: string;
  category?: string;
  sourceFrameId?: string;
  targetSourceFrameId?: string;
  sourceCandidateIds?: string[];
  sourceImageId?: string | null;
  sourcePixelBox?: number[] | null;
  imageId?: string | null;
  cameraId?: string | null;
  entityIds: string[];
  observationId?: string | null;
  mappingStatus: string;
  mappingLimitation?: string;
};
type Interpretation = {
  runId: string;
  notice: string;
  items: InterpretationItem[];
  missing: unknown[];
  rejected: unknown[];
  sourceRefs: SourceRef[];
};
type ReportBundle = {
  imageInterpretations?: Interpretation[];
  schemaVersion: 1;
  sourceRunId: string;
  reconstructionRunId: string;
  sourceRefs: SourceRef[];
  mappingNotes: string[];
  historical?: {
    runId: string;
    status?: string;
    summary?: unknown;
    assessment?: unknown;
    findings: HistoricalFinding[];
    policies?: HistoricalPolicy[];
    frames?: { sourceFrameId: string; imageId: string }[];
    inventory: HistoricalInventory[];
    cad?: {
      assetId: string;
      width: number;
      height: number;
      regions: {
        inventoryIndex: number;
        entityIds: string[];
        polygon: number[][];
      }[];
    };
  };
  objects: {
    entityId: string | null;
    views?: {observationId:string|null;entityId:string|null;imageId?:string|null}[];
    sourceRecordId: string;
    metrics?: {
      beforeIou?: number | null;
      afterIou?: number | null;
      beforeDepth?: number | null;
      afterDepth?: number | null;
    };
    metricsMeaning?: string;
  }[];
  resources: {
    id: string;
    label: string;
    kind: string;
    assetId: string;
    runId: string;
    meaning: string;
    sourceRefs: SourceRef[];
  }[];
  quality?: { metricMeaning?: string; scale?: unknown; limitations?: string[] };
};

export type ReportEvidenceSelection = {
  entityId: string;
  observationId: string | null;
  imageId: string | null;
  cameraId: string | null;
};
export function interpretationEntityIds(document: SceneDocument, item: InterpretationItem): string[] {
  if (item.observationId) { const owner=observationOwner(document,item.observationId);return owner?[owner.id]:[]; }
  const observations=document.observations.filter(observation=>observation.imageId===item.imageId && observation.sourceRefs?.some(ref=>item.sourceCandidateIds?.includes(String(jsonObject(ref)?.sourceRecordId))));
  if(observations.length) return [...new Set(observations.flatMap(observation=>{const owner=observationOwner(document,observation.id);return owner?[owner.id]:[];}))];
  return [...new Set(item.entityIds.flatMap(id=>{const current=currentEntityId(document,id);return current?[current]:[];}))];
}
export function interpretationSelection(
  document: SceneDocument,
  item: InterpretationItem,
  entityId: string,
): ReportEvidenceSelection | null {
  const entity = document.entities.find((e) => e.id === entityId);
  if (!entity || !interpretationEntityIds(document,item).includes(entityId)) return null;
  const imageId =
    item.imageId && document.assets.some((a) => a.id === item.imageId)
      ? item.imageId
      : null;
  const observations = document.observations.filter(
    (o) => entity.observationRefs?.includes(o.id) && o.imageId === imageId,
  );
  const matching = observations.filter((o) =>
    o.sourceRefs?.some((ref) =>
      item.sourceCandidateIds?.includes(
        String(jsonObject(ref)?.sourceRecordId),
      ),
    ),
  );
  const observation =
    observations.find(observation => observation.id === item.observationId) || (matching.length === 1
      ? matching[0]
      : observations.length === 1
        ? observations[0]
        : undefined);
  const camera = cameraForImage(document, imageId);
  return {
    entityId,
    observationId: observation?.id || null,
    imageId,
    cameraId: camera?.id || null,
  };
}
export function orderedInterpretations(
  document: SceneDocument,
  items: Interpretation[],
  historicalRunId?: string,
) {
  const rank = (analysis: Interpretation) =>
    analysis.items.some((item) =>
      interpretationEntityIds(document,item).some(
        (id) => interpretationSelection(document, item, id)?.imageId,
      ),
    )
      ? 0
      : analysis.runId === historicalRunId
        ? 2
        : 1;
  return [...items].sort((a, b) => rank(a) - rank(b));
}
export function partitionInterpretationItems(
  document: SceneDocument,
  items: InterpretationItem[],
  currentImageId?: string | null,
) {
  const rows = items.map((item) => ({
    item,
    sourcePhoto: interpretationSourcePhoto(document, item),
    contexts: interpretationEntityIds(document,item)
      .map((id) => interpretationSelection(document, item, id))
      .filter((value): value is ReportEvidenceSelection => !!value?.imageId),
  }));
  const linked = rows.filter((row) => row.contexts.length);
  if (currentImageId)
    linked.sort(
      (a, b) =>
        Number(b.contexts.some((c) => c.imageId === currentImageId)) -
        Number(a.contexts.some((c) => c.imageId === currentImageId)),
    );
  return {
    linked,
    sourceOnly: rows.filter((row) => !row.contexts.length && row.sourcePhoto),
    unbound: rows.filter((row) => !row.contexts.length && !row.sourcePhoto),
  };
}

export function interpretationSourcePhoto(document: SceneDocument, item: InterpretationItem) {
  const asset = document.assets.find((a) => a.id === item.sourceImageId);
  const metadata = jsonObject(asset?.metadata);
  const width = Number(asset?.width ?? metadata?.width), height = Number(asset?.height ?? metadata?.height);
  if (!asset || typeof asset.mediaType !== "string" || !["image/png", "image/jpeg", "image/webp"].includes(asset.mediaType)
      || !Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0) return null;
  const raw = item.sourcePixelBox;
  const box = Array.isArray(raw) && raw.length === 4 && raw.every(Number.isFinite)
    && raw[0] >= 0 && raw[1] >= 0 && raw[2] > raw[0] && raw[3] > raw[1]
    && raw[2] <= width && raw[3] <= height ? raw : null;
  const padX = box ? Math.max(10, (box[2] - box[0]) * .2) : 0;
  const padY = box ? Math.max(10, (box[3] - box[1]) * .2) : 0;
  const left = box ? Math.max(0, box[0] - padX) : 0, top = box ? Math.max(0, box[1] - padY) : 0;
  const crop = box ? [left, top, Math.min(width, box[2] + padX) - left, Math.min(height, box[3] + padY) - top] : [0, 0, width, height];
  return { assetId: asset.id, width, height, box, crop };
}

function InterpretationPhoto({ source, label }: {
  source: NonNullable<ReturnType<typeof interpretationSourcePhoto>>;
  label: string;
}) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false), [full, setFull] = useState(false);
  const [url, setURL] = useState<string>(), [error, setError] = useState(false), [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!open) return;
    let live = true;
    setURL(undefined); setError(false);
    resolveAsset(source.assetId).then(value => { if (live) setURL(value); }).catch(() => { if (live) setError(true); });
    return () => { live = false; };
  }, [open, source.assetId, attempt]);
  return <details className="report-source-photo" data-source-image-id={source.assetId}
    onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>{t("reViewSourcePhoto")}</summary>
    {open && <figure>
      {error ? <p role="alert">{t("rePhotoLoadError")} <button type="button" onClick={() => setAttempt(value => value + 1)}>{t("reCadRetry")}</button></p>
        : !url ? <p role="status">{t("loading")}</p>
        : <svg viewBox={(full ? [0, 0, source.width, source.height] : source.crop).join(" ")}
            role="img" aria-label={`${label} · ${t("reOriginalPhoto")}`}>
            <image href={url} width={source.width} height={source.height} onError={() => setError(true)} />
            {source.box && <rect x={source.box[0]} y={source.box[1]} width={source.box[2] - source.box[0]} height={source.box[3] - source.box[1]}
              className="report-source-photo-box" vectorEffect="non-scaling-stroke" />}
          </svg>}
      <figcaption>{label} · {source.box ? t("reSourceBoxMeaning") : t("reSourceBoxMissing")}</figcaption>
      {source.box && <button type="button" onClick={() => setFull(value => !value)}>{t(full ? "rePhotoCrop" : "rePhotoFull")}</button>}
      <ReportDownload assetId={source.assetId}>{t("reOriginalPhoto")}</ReportDownload>
    </figure>}
  </details>;
}
export function exactHistoricalPolicy(
  policies: HistoricalPolicy[],
  findingId: string,
) {
  const matches = policies.filter(
    (p) => p.id === findingId && p.spec?.policyId === findingId,
  );
  return matches.length === 1 ? matches[0] : undefined;
}
export function comparisonSource(bundle: ReportBundle, sourceRecordId: string) {
  const matches = bundle.resources.filter(
    (r) =>
      r.kind === "quality" && r.label === sourceRecordId + "-comparison.json",
  );
  return matches.length === 1 ? matches[0] : undefined;
}
/** A field name of the evidence JSON: its catalog text (evidence.<key>) when it has one, else the key split into words. */
function readable(key: string, t: (id: string) => string) {
  return hasMessage("evidence." + key) ? t("evidence." + key) : key.replace(/([a-z])([A-Z])/g, "$1 $2").replaceAll("_", " ");
}
export function EvidenceValue({ value }: { value: unknown }) {
  const { language, t } = useI18n();
  if (value === null || value === undefined) return <span>{t("unknown")}</span>;
  if (typeof value === "boolean") return <span>{String(value)}</span>;
  if (typeof value === "string" || typeof value === "number")
    return <span>{String(value)}</span>;
  if (Array.isArray(value))
    return (
      <ul>
        {value.map((v, i) => (
          <li key={i}>
            <EvidenceValue value={v} />
          </li>
        ))}
      </ul>
    );
  const object = jsonObject(value);
  return object ? (
    <dl className="report-evidence-fields">
      {Object.entries(object).map(([key, v]) => (
        <div key={key}>
          <dt>{readable(key, t)}</dt>
          <dd>
            <EvidenceValue value={v} />
          </dd>
        </div>
      ))}
    </dl>
  ) : null;
}
type SourceCAD = NonNullable<NonNullable<ReportBundle["historical"]>["cad"]>;
type CadViewBox = [number, number, number, number];

export function sourceCadFor(document: SceneDocument): { cad: SourceCAD; runId: string } | null {
  const bundle = jsonObject(document.reportEvidence);
  if (bundle?.schemaVersion !== 1) return null;
  const historical = jsonObject(bundle.historical);
  const raw = jsonObject(historical?.cad);
  if (!raw || typeof historical?.runId !== "string" || !historical.runId.trim()
      || typeof raw.assetId !== "string" || !Number.isSafeInteger(raw.width) || Number(raw.width) <= 0
      || !Number.isSafeInteger(raw.height) || Number(raw.height) <= 0 || !Array.isArray(raw.regions)) return null;
  const asset = document.assets.find(item => item.id === raw.assetId);
  const assetRun = asset && (asset.sourceRunId ?? jsonObject(asset.metadata)?.sourceRunId);
  if (!asset || typeof asset.mediaType !== "string" || !(["image/png", "image/jpeg", "image/webp", "image/svg+xml"].includes(asset.mediaType)) || assetRun !== historical.runId) return null;
  const regions: SourceCAD["regions"] = [];
  for (const value of raw.regions) {
    const region = jsonObject(value);
    if (!region || !Number.isSafeInteger(region.inventoryIndex) || Number(region.inventoryIndex) < 0
        || !Array.isArray(region.entityIds) || !region.entityIds.every(id => typeof id === "string" && id.length > 0)
        || !Array.isArray(region.polygon) || region.polygon.length < 3
        || !region.polygon.every(point => Array.isArray(point) && point.length === 2 && point.every(n => typeof n === "number" && Number.isFinite(n)))) return null;
    regions.push({ inventoryIndex: Number(region.inventoryIndex), entityIds: [...new Set(region.entityIds as string[])], polygon: region.polygon.map(point => [...point]) });
  }
  return { cad: { assetId: raw.assetId, width: Number(raw.width), height: Number(raw.height), regions }, runId: historical.runId };
}

export function cadZoomView(view: CadViewBox, width: number, factor: number, anchor: [number, number]): CadViewBox {
  if (!Number.isFinite(factor) || factor <= 0) return view;
  const zoom = Math.min(16, Math.max(1, width / view[2] * factor));
  const ratio = width / zoom / view[2];
  return [anchor[0] + (view[0] - anchor[0]) * ratio, anchor[1] + (view[1] - anchor[1]) * ratio, view[2] * ratio, view[3] * ratio];
}

export function cadPanView(view: CadViewBox, dx: number, dy: number): CadViewBox {
  return [view[0] - dx, view[1] - dy, view[2], view[3]];
}

export function cadFocusView(cad: SourceCAD, polygons: number[][][]): CadViewBox {
  const points = polygons.flat();
  if (!points.length) return [0, 0, cad.width, cad.height];
  const xs = points.map(point => point[0]), ys = points.map(point => point[1]);
  const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
  const aspect = cad.width / cad.height;
  const width = Math.min(cad.width, Math.max(cad.width / 3, (right - left) * 1.7, (bottom - top) * aspect * 1.7));
  return [(left + right - width) / 2, (top + bottom - width / aspect) / 2, width, width / aspect];
}

export function cadLinkedEntities(document: SceneDocument, region: SourceCAD["regions"][number]): string[] {
  const ids = new Set(document.entities.map(entity => entity.id));
  return [...new Set(region.entityIds)].filter(id => ids.has(id));
}

function HistoricalCAD({ cad, runId, document, onSelect, embedded = false, selectedId = null }: {
  cad: SourceCAD;
  runId: string;
  document: SceneDocument;
  onSelect: (entityId: string) => void;
  embedded?: boolean;
  selectedId?: string | null;
}) {
  const { t } = useI18n();
  const [url, setURL] = useState<string>(), [error, setError] = useState<unknown>();
  const [attempt, setAttempt] = useState(0), [imageReady, setImageReady] = useState(false);
  const [imageFailed, setImageFailed] = useState(false), [choices, setChoices] = useState<string[]>([]);
  const [view, setView] = useState<CadViewBox>([0, 0, cad.width, cad.height]);
  const svgRef = useRef<SVGSVGElement>(null), suppressClick = useRef(false);
  const drag = useRef<{ pointerId: number; clientX: number; clientY: number; point: DOMPoint; matrix: DOMMatrix; view: CadViewBox; moved: boolean } | null>(null);
  const linkedRegions = cad.regions.map(region => ({ ...region, entityIds: cadLinkedEntities(document, region) })).filter(region => region.entityIds.length);
  const objectCount = new Set(cad.regions.map(region => region.inventoryIndex)).size;
  const linkedCount = new Set(linkedRegions.map(region => region.inventoryIndex)).size;
  const sourceHistorical = jsonObject(jsonObject(document.reportEvidence)?.historical);
  const coverage = jsonObject(jsonObject(sourceHistorical?.cad)?.coverage);
  const coverageRows = coverage?.method === "source_cad_identity_v1" && Array.isArray(coverage.records)
    ? coverage.records.map(jsonObject).filter((row): row is Record<string, unknown> => Boolean(row && Number.isSafeInteger(row.inventoryIndex))) : [];
  const unresolvedRows = coverageRows.filter(row => !linkedRegions.some(region => region.inventoryIndex === row.inventoryIndex));
  const selectionMapped = linkedRegions.some(region => selectedId && region.entityIds.includes(selectedId));
  const fit = () => setView([0, 0, cad.width, cad.height]);
  const zoom = (factor: number) => setView(current => cadZoomView(current, cad.width, factor, [current[0] + current[2] / 2, current[1] + current[3] / 2]));
  useEffect(() => {
    let live = true;
    setURL(undefined); setError(undefined); setImageReady(false); setImageFailed(false); setChoices([]);
    setView([0, 0, cad.width, cad.height]);
    resolveAsset(cad.assetId).then(u => { if (live) setURL(u); }).catch(e => { if (live) setError(e); });
    return () => { live = false; drag.current = null; };
  }, [cad.assetId, cad.width, cad.height, attempt]);
  useEffect(() => { setChoices([]); }, [selectedId]);
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const wheel = (event: WheelEvent) => {
      // Normal scrolling belongs to the report; modified wheel deliberately zooms the drawing.
      if (!event.ctrlKey && !event.metaKey) return;
      const matrix = svg.getScreenCTM();
      if (!matrix) return;
      event.preventDefault();
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
      setView(current => cadZoomView(current, cad.width, Math.exp(-Math.max(-100, Math.min(100, event.deltaY)) * 0.01), [point.x, point.y]));
    };
    svg.addEventListener("wheel", wheel, { passive: false });
    return () => svg.removeEventListener("wheel", wheel);
  }, [url, cad.width]);
  function choose(entityIds: string[]) {
    if (entityIds.length === 1) { setChoices([]); onSelect(entityIds[0]); }
    else setChoices(entityIds);
  }
  return (
    <figure id={embedded ? undefined : "workcell-original-cad"} className={`report-cad-evidence${embedded ? " report-cad-embedded" : ""}`}>
      {!embedded && <h3>{t("reportCadOriginal")}</h3>}
      <div className="report-cad-tools">
        <span>{t("reCadSourceDrawing")} · {runId}</span>
        <div role="group" aria-label={t("reCadNavigation")}>
          <button type="button" aria-label={t("reCadZoomOut")} onClick={() => zoom(1 / 1.5)}>−</button>
          <button type="button" onClick={fit}>{t("reCadFit")}</button>
          <button type="button" disabled={!selectionMapped} onClick={() => setView(cadFocusView(cad, linkedRegions.filter(region => selectedId && region.entityIds.includes(selectedId)).map(region => region.polygon)))}>{t("reCadFocus")}</button>
          <button type="button" aria-label={t("reCadZoomIn")} onClick={() => zoom(1.5)}>+</button>
        </div>
      </div>
      <p className="report-cad-summary">{objectCount} {t("reCadObjectRecords")} · {linkedCount} {t("reCadLinkedObjects")} · {objectCount - linkedCount} {t("reCadUnresolvedRecords")}</p>
      {!!coverageRows.length && <details className="report-quality-details">
        <summary>{t("reCadCoverageLedger")} · {unresolvedRows.length} {t("reCadUnresolvedRecords")}</summary>
        <p>{t("reCadCoverageMeaning")}</p>
        <div className="table-scroll"><table><thead><tr>
          <th>{t("reCadSourceRecord")}</th><th>{t("reportSource")}</th><th>{t("reCadLinkStatus")}</th>
        </tr></thead><tbody>{coverageRows.map(row => {
          const linked = linkedRegions.find(region => region.inventoryIndex === row.inventoryIndex);
          return <tr key={String(row.inventoryIndex)}>
            <td>#{Number(row.inventoryIndex) + 1} {typeof row.label === "string" ? row.label : ""}</td>
            <td>{typeof row.sourceFrameId === "string" ? row.sourceFrameId : "—"}</td>
            <td>{linked ? <button type="button" onClick={() => choose(linked.entityIds)}>{linked.entityIds.map(id => document.entities.find(entity => entity.id === id)?.label).filter(Boolean).join(" · ")} ↗</button>
              : t(row.reason === "no_verified_same_photo" ? "reCadNoSourcePhoto" : row.reason === "canonical_masks_differ" ? "reCadMaskChanged"
                : row.reason === "ambiguous_entity_ownership" ? "reCadAmbiguousOwnership" : row.reason === "source_segmentation_hash_mismatch" ? "reCadSourceHashMismatch" : "reCadExactProofMissing")}
              {linked && <small>{t(row.proofStatus === "verified_original_source_mask" ? "reCadSourceMaskVerified" : "reCadExplicitBindingOnly")}</small>}</td>
          </tr>;
        })}</tbody></table></div>
        {typeof sourceHistorical?.inventoryAssetId === "string" && <ReportDownload assetId={sourceHistorical.inventoryAssetId}>{t("reCadSourceInventory")}</ReportDownload>}
        {typeof sourceHistorical?.sourceCadManifestAssetId === "string" && <ReportDownload assetId={sourceHistorical.sourceCadManifestAssetId}>{t("reCadSourceManifest")}</ReportDownload>}
      </details>}
      <ErrorNotice error={error} />
      {(error || imageFailed) && <div role="alert" className="report-cad-state">{imageFailed && t("reCadLoadError")} <button onClick={() => setAttempt(value => value + 1)}>{t("reCadRetry")}</button></div>}
      {!error && !imageFailed && !imageReady && <div className="report-cad-state" role="status">{t("reCadLoading")}</div>}
      <div className="report-cad-viewport">
        {url && <svg ref={svgRef} viewBox={view.join(" ")} role="group" aria-label={t("reportCadOriginal")}
          onClickCapture={event => { if (suppressClick.current && event.detail > 0) { event.preventDefault(); event.stopPropagation(); suppressClick.current = false; } }}
          onPointerDown={event => {
            if (event.button !== 0 || !event.isPrimary) return;
            const matrix = event.currentTarget.getScreenCTM()?.inverse();
            if (!matrix) return;
            suppressClick.current = false;
            drag.current = { pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY,
              point: new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix), matrix, view, moved: false };
          }}
          onPointerMove={event => {
            const current = drag.current;
            if (!current || current.pointerId !== event.pointerId) return;
            if (!current.moved && Math.hypot(event.clientX - current.clientX, event.clientY - current.clientY) < 4) return;
            current.moved = true; suppressClick.current = true;
            if (!event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.setPointerCapture(event.pointerId);
            const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(current.matrix);
            setView(cadPanView(current.view, point.x - current.point.x, point.y - current.point.y));
          }}
          onPointerUp={event => {
            if (drag.current?.pointerId !== event.pointerId) return;
            drag.current = null;
            if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
          }}
          onPointerCancel={() => { drag.current = null; suppressClick.current = true; }}
        >
          <image href={url} width={cad.width} height={cad.height} onLoad={() => setImageReady(true)} onError={() => setImageFailed(true)} />
          {linkedRegions.map((region, index) => <polygon key={index}
            data-selected={Boolean(selectedId && region.entityIds.includes(selectedId))}
            points={region.polygon.map(point => point.join(",")).join(" ")} tabIndex={0} role="button"
            aria-label={`${t("objects")} ${region.inventoryIndex}`} aria-pressed={Boolean(selectedId && region.entityIds.includes(selectedId))}
            onClick={() => choose(region.entityIds)}
            onKeyDown={event => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); event.stopPropagation(); choose(region.entityIds); } }}>
            <title>{t("objects")} {region.inventoryIndex}</title>
          </polygon>)}
        </svg>}
      </div>
      {choices.length > 1 && <div className="report-cad-choices" role="group" aria-label={t("reCadChooseEntity")}>
        <span>{t("reCadChooseEntity")}</span>
        {choices.map(id => <button key={id} onClick={() => { setChoices([]); onSelect(id); }}>{document.entities.find(entity => entity.id === id)?.label} · {id.slice(0, 8)}</button>)}
      </div>}
      {selectedId && !selectionMapped && <p className="report-cad-selection-note" role="status">{t("reCadSelectionUnmapped")}</p>}
      <figcaption>
        {embedded ? t("reCadEmbeddedBasis") : <>{t("reCadBasis")}<br />{t("reportSource")} · {runId} · {t("reCadPixelCoordinates")} {cad.width} × {cad.height} px</>}
        <span className="report-cad-gesture">{t("reCadGesture")}</span>
      </figcaption>
      {!embedded && <ReportDownload assetId={cad.assetId}>CAD</ReportDownload>}
    </figure>
  );
}

export function ReportEvidence({
  document,
  section,
  onSelect,
  onSelectEvidence,
  currentImageId,
}: {
  document: SceneDocument;
  section: "understanding" | "safety" | "assets" | "quality";
  onSelect: (entityId: string) => void;
  onSelectEvidence?: (context: ReportEvidenceSelection) => void;
  currentImageId?: string | null;
}) {
  const { t, language } = useI18n();
  const raw = jsonObject(document.reportEvidence),
    bundle =
      raw?.schemaVersion === 1 ? (raw as unknown as ReportBundle) : undefined;
  const historical = bundle?.historical;
  if (section === "understanding") {
    const interpretations = document.annotations.filter(
      (a) =>
        a.kind === "image_interpretation" || a.kind === "scene_interpretation",
    );
    return (
      <>
        {interpretations.map((a) => (
          <div className="report-reading-copy" key={a.id}>
            <EvidenceValue value={a.text || a.summary} />
          </div>
        ))}
        {orderedInterpretations(
          document,
          bundle?.imageInterpretations || [],
          historical?.runId,
        ).map((analysis, index) => {
          const { linked, sourceOnly, unbound } = partitionInterpretationItems(
            document,
            analysis.items,
            currentImageId,
          );
          const rows = (items: typeof linked) => (
            <div className="report-detection-notes">
              {items.map(({ item, contexts, sourcePhoto }, i) => {
                const label =
                  item.label;
                return (
                  <div className="report-detection-note" key={i}>
                    <strong>
                      {contexts.length
                        ? contexts.map((context) => (
                            <button
                              key={context.entityId}
                              onClick={() =>
                                onSelectEvidence
                                  ? onSelectEvidence(context)
                                  : onSelect(context.entityId)
                              }
                            >
                              {contexts.length === 1
                                ? label
                                : document.entities.find(
                                    (e) => e.id === context.entityId,
                                  )?.label || label}{" "}
                              ↗
                            </button>
                          ))
                        : label}
                    </strong>
                    <div>
                      {item.note && <p>{item.note}</p>}
                      {currentImageId &&
                        contexts.some(
                          (context) => context.imageId === currentImageId,
                        ) && <small>{t("reportPhotoSelection")}</small>}
                      <details className="report-evidence-origin">
                        <summary>{t("rePhotoSource")}</summary>
                        <small>
                          {t("reOriginalPhoto")} ·{" "}
                          {item.sourceFrameId || t("reportSourceFrameUnknown")}
                          {item.targetSourceFrameId && (
                            <>
                              {" "}
                              → {t("reScenePhoto")} · {item.targetSourceFrameId}
                            </>
                          )}{" "}
                          ·{" "}
                          {t(contexts.length ? "reCurrentObjectLocated" : sourcePhoto ? "reSourcePhotoAvailable" : "rePhotoUnbound")}
                          {contexts.length > 0 && item.mappingLimitation && (
                            <>
                              <br />
                              {item.mappingLimitation}
                            </>
                          )}
                        </small>
                      </details>
                      {sourcePhoto && <InterpretationPhoto key={sourcePhoto.assetId} source={sourcePhoto} label={label || t("reSourceRecord")} />}
                    </div>
                  </div>
                );
              })}
            </div>
          );
          return (
            <details
              className="report-source-details"
              key={analysis.runId + ":" + index}
            >
              <summary>
                {analysis.runId === historical?.runId
                  ? <>{t("reHistoricalArchive")} · {analysis.items.length} {t("reSourceRecords")}</>
                  : <>{t("reportUnderstanding")} · {linked.length} {t("reLinkedRecords")} / {analysis.items.length} {t("reSourceRecords")}</>}
              </summary>
              <p className="report-evidence-note">
                {analysis.runId === historical?.runId
                  ? t("reHistoricalArchiveMeaning")
                  : t("reportDetectionMeaning")}
              </p>
              <p className="report-interpretation-counts">
                <span>{linked.length} · {t("reCurrentObjectLocated")}</span>
                <span>{sourceOnly.length} · {t("reSourceOnlyCount")}</span>
                <span>{unbound.length} · {t("rePhotoUnbound")}</span>
              </p>
              {!!linked.length && rows(linked)}
              {!!sourceOnly.length && (
                <details className="report-source-details report-source-only-records">
                  <summary>
                    {t("reSourceOnlyRecords")} · {sourceOnly.length}
                  </summary>
                  <p className="report-evidence-note">
                    {t("reSourceOnlyMeaning")}
                  </p>
                  {rows(sourceOnly)}
                </details>
              )}
              {!!unbound.length && (
                <details className="report-source-details report-unbound-records">
                  <summary>{t("reUnboundRecords")} · {unbound.length}</summary>
                  <p className="report-evidence-note">{t("reUnboundMeaning")}</p>
                  {rows(unbound)}
                </details>
              )}
              {!!analysis.missing.length && (
                <details className="report-source-details">
                  <summary>
                    {t("reportMissingCandidates")} · {analysis.missing.length}
                  </summary>
                  <EvidenceValue value={analysis.missing} />
                </details>
              )}
              {!!analysis.rejected.length && (
                <details className="report-source-details">
                  <summary>
                    {t("reportRejectedCandidates")} · {analysis.rejected.length}
                  </summary>
                  <EvidenceValue value={analysis.rejected} />
                </details>
              )}
              <details className="report-evidence-origin">
                <summary>{t("reportTechnical")}</summary>
                <p className="report-evidence-note">
                  {t("reportSource")} · {analysis.runId}
                </p>
                {analysis.sourceRefs.map((ref, i) => (
                  <ReportDownload assetId={ref.assetId} key={i}>
                    {t("reportOriginalEvidence")}
                  </ReportDownload>
                ))}
              </details>
            </details>
          );
        })}
        {!interpretations.length && !bundle?.imageInterpretations?.length && (
          <p className="report-evidence-note">{t("reportNoInterpretation")}</p>
        )}
        {!!historical?.inventory?.length && (
          <details className="report-source-details">
            <summary>
              {t("reportOriginalEvidence")} · {historical.inventory.length}{" "}
              {t("objects")}
            </summary>
            <p className="report-evidence-note">{t("reportHistoricalHint")}</p>
            <div className="report-table-scroll">
              <table className="report-inventory">
                <thead>
                  <tr>
                    <th>{t("objects")}</th>
                    <th>{t("photos")}</th>
                    <th>{t("sourceEvidence")}</th>
                    <th>{t("groundHeight")}</th>
                  </tr>
                </thead>
                <tbody>
                  {historical.inventory.map((item, i) => (
                    <tr key={i}>
                      <td>
                        {item.label}
                        {item.entityIds.filter(id=>document.entities.some(entity=>entity.id===id)).map(id=><button key={id} onClick={()=>onSelect(id)}>{document.entities.find(entity=>entity.id===id)?.label || id.slice(0,8)} ↗</button>)}
                      </td>
                      <td>{item.sourceFrameId}</td>
                      <td>
                        {item.source}
                        <small>{item.mappingStatus}</small>
                      </td>
                      <td>
                        {typeof item.heightM === "number"
                          ? item.heightM.toFixed(3) + " m"
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}
      </>
    );
  }
  if (section === "safety")
    return historical?.findings?.length ? (
      <div className="report-historical">
        <h3>{t("reportHistoricalSafety")}</h3>
        <p className="report-evidence-note">{t("reportHistoricalHint")}</p>
        <p className="report-historical-count">{historical.findings.length} {t("reHistoricalChecks")}</p>
        <details className="report-source-details">
          <summary>{t("reportOriginalAssessment")}</summary>
          <p className="report-kicker">{historical.runId}</p>
          <p className="report-evidence-note">{t("rePolicySourceNote")}</p>
          <EvidenceValue value={historical.summary} />
          <EvidenceValue value={historical.assessment} />
        </details>
        {historical.findings.map((finding, i) => {
          const policy = exactHistoricalPolicy(
              historical.policies || [],
              finding.id,
            ),
            spec = policy?.spec;
          return (
            <details
              className="report-historical-finding"
              key={finding.id || i}
            >
              <summary>
                <span className="report-historical-scope">
                  <small>{t("rePolicyScope")}</small>
                  {spec?.subjectLabels?.length || spec?.objectLabels?.length
                    ? [spec.subjectLabels?.join(", "), spec.objectLabels?.join(", ")].filter(Boolean).join(" → ")
                    : finding.title || finding.id}
                </span>
                <span className={"badge " + finding.status?.toLowerCase()}>
                  {t(finding.status || "unknown")}
                </span>
              </summary>
              {finding.summary && <p>{finding.summary}</p>}
              <details className="report-rule-basis">
                <summary>{t("rePolicyBasis")}</summary>
                <h4>{finding.title || finding.id}</h4>
                {spec ? (
                  <dl className="report-evidence-fields">
                    {spec.rationale && (
                      <div>
                        <dt>{t("rePolicyReason")}</dt>
                        <dd>{spec.rationale}</dd>
                      </div>
                    )}
                    {spec.sourceText && (
                      <div>
                        <dt>{t("rePolicyText")}</dt>
                        <dd>{spec.sourceText}</dd>
                      </div>
                    )}
                    {typeof spec.threshold === "number" &&
                      Number.isFinite(spec.threshold) && (
                        <div>
                          <dt>{t("rePolicyThreshold")}</dt>
                          <dd>
                            {spec.threshold} {spec.unit}
                          </dd>
                        </div>
                      )}
                    {spec.subjectLabels?.length || spec.objectLabels?.length ? (
                      <div>
                        <dt>{t("rePolicyScope")}</dt>
                        <dd>
                          {spec.subjectLabels?.join(", ")}
                          {spec.objectLabels?.length
                            ? " → " + spec.objectLabels.join(", ")
                            : ""}
                        </dd>
                      </div>
                    ) : null}
                    {spec.unsupportedReason && (
                      <div>
                        <dt>{t("rePolicyLimit")}</dt>
                        <dd>{spec.unsupportedReason}</dd>
                      </div>
                    )}
                  </dl>
                ) : (
                  <p className="report-evidence-note">{t("rePolicyMissing")}</p>
                )}
                {finding.predicate && (
                  <p>
                    {t("reEvaluatedCheck")} · {finding.predicate}
                    {typeof finding.threshold === "number"
                      ? " · " + finding.threshold + " " + (finding.unit || "")
                      : ""}
                  </p>
                )}
                <details className="report-source-details">
                  <summary>{t("evidence")}</summary>
                  <EvidenceValue value={finding.facts} />
                </details>
                {!!finding.violations?.length && (
                  <details className="report-source-details">
                    <summary>
                      {t("reportOriginalEvidence")} ·{" "}
                      {finding.violations.length}
                    </summary>
                    <EvidenceValue value={{ violations: finding.violations }} />
                  </details>
                )}
                {!!finding.warnings?.length && (
                  <details className="report-source-details">
                    <summary>
                      {readable("warnings", t)} ·{" "}
                      {finding.warnings.length}
                    </summary>
                    <EvidenceValue value={finding.warnings} />
                  </details>
                )}
              </details>
              <details className="report-source-details">
                <summary>{t("reportTechnical")}</summary>
                <div className="report-downloads">
                {policy?.sourceRefs?.map((ref, j) => (
                  <ReportDownload key={"policy:" + j} assetId={ref.assetId}>
                    {t("rePolicyText")}
                  </ReportDownload>
                ))}
                {finding.sourceRefs?.map((ref, j) => (
                  <ReportDownload key={j} assetId={ref.assetId}>
                    {t("reportTechnical")}
                    {ref.jsonPointer ? " · " + ref.jsonPointer : ""}
                  </ReportDownload>
                ))}
                </div>
              </details>
            </details>
          );
        })}
      </div>
    ) : null;
  if (section === "assets")
    return (
      <>
        {historical?.cad && (
          <HistoricalCAD cad={historical.cad} runId={historical.runId} document={document} onSelect={onSelect} />
        )}
        {!!historical?.frames?.length && (
          <details className="report-source-details">
            <summary>
              {t("reHistoricalPhotos")} · {historical.frames.length}
            </summary>
            <p className="report-evidence-note">
              {t("reportHistoricalHint")} · {historical.runId}
            </p>
            <div className="report-downloads">
              {historical.frames.map((frame) => (
                <ReportDownload key={frame.imageId} assetId={frame.imageId}>
                  {frame.sourceFrameId}
                </ReportDownload>
              ))}
            </div>
          </details>
        )}
        {!!bundle?.resources?.length && <details className="report-source-details">
          <summary>{t("reSourceFiles")} · {bundle.resources.length}</summary>
          <div className="report-resource-list">
          {bundle?.resources?.map((resource) => (
            <div className="report-resource-row" key={resource.id}>
              <strong>{resource.label}</strong>
              <p>{resource.meaning}</p>
              <small>
                {t("reportHistorical")} · {resource.runId}
              </small>
              <br />
              <ReportDownload assetId={resource.assetId}>
                {resource.label}
              </ReportDownload>
            </div>
          ))}
          </div>
        </details>}
      </>
    );
  const comparison =
    bundle?.objects?.filter(
      (o) =>
        o.metrics &&
        [
          o.metrics.beforeIou,
          o.metrics.afterIou,
          o.metrics.beforeDepth,
          o.metrics.afterDepth,
        ].some((v) => typeof v === "number"),
    ) || [];
  const fmt = (v: number | null | undefined) =>
    typeof v === "number" && Number.isFinite(v) ? v.toFixed(4) : "—";
  return (
    <details className="report-historical report-quality-details">
      <summary>{t("reportQuality")}</summary>
      {comparison.length ? (
        <>
          <h4>{t("reExperiment")}</h4>
          <p className="report-evidence-note">{t("reExperimentMeaning")}</p>
          <p className="report-evidence-note">
            {bundle?.quality?.metricMeaning}
          </p>
          <div className="report-table-scroll">
            <table className="report-facts-table">
              <thead>
                <tr>
                  <th>{t("objects")}</th>
                  <th>{t("reMetricSource")}</th>
                  <th>IoU · {t("before")}</th>
                  <th>IoU · {t("after")}</th>
                  <th>Δ IoU</th>
                  <th>{t("metric.depth")} · {t("before")}</th>
                  <th>{t("metric.depth")} · {t("after")}</th>
                </tr>
              </thead>
              <tbody>
                {comparison.map((o, i) => {
                  const source = comparisonSource(bundle!, o.sourceRecordId),
                    entity = document.entities.find((e) => e.id === o.entityId);
                  return (
                    <tr key={i}>
                      <td>
                        {entity ? (
                          <button onClick={() => onSelect(entity.id)}>
                            {entity.label || o.sourceRecordId} ↗
                          </button>
                        ) : (
                          o.sourceRecordId
                        )}
                        <small>{t("reSourceRecord")} · {o.sourceRecordId}</small>
                        {o.views?.map((view, index) => { const owner = view.observationId ? observationOwner(document, view.observationId) : null; return owner ? <button key={view.observationId || index} onClick={() => onSelect(owner.id)}>{t("photos")} {index + 1} · {owner.label || owner.id.slice(0,8)}</button> : null; })}
                      </td>
                      <td>
                        <strong>{source?.runId || t("reRunUnknown")}</strong>
                        <p className="report-evidence-note">
                          {o.metricsMeaning || t("reMetricMeaningUnknown")}
                        </p>
                        {source && (
                          <ReportDownload assetId={source.assetId}>
                            {t("reportOriginalEvidence")}
                          </ReportDownload>
                        )}
                      </td>
                      <td>{fmt(o.metrics?.beforeIou)}</td>
                      <td>{fmt(o.metrics?.afterIou)}</td>
                      <td>
                        {typeof o.metrics?.beforeIou === "number" &&
                        typeof o.metrics?.afterIou === "number"
                          ? fmt(o.metrics.afterIou - o.metrics.beforeIou)
                          : "—"}
                      </td>
                      <td>{fmt(o.metrics?.beforeDepth)}</td>
                      <td>{fmt(o.metrics?.afterDepth)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <p className="report-evidence-note">{t("reportEmptyMetrics")}</p>
      )}
      {bundle?.quality?.limitations?.length ? (
        <ul className="report-evidence-note">
          {bundle.quality.limitations.map((text, i) => (
            <li key={i}>{text}</li>
          ))}
        </ul>
      ) : null}
      {bundle && (
        <details className="report-source-details">
          <summary>{t("reportTechnical")}</summary>
          <p>
            {t("reSceneRun")} · {bundle.reconstructionRunId}
          </p>
          <p>
            {t("reportSource")} · {bundle.sourceRunId}
          </p>
          <ul>
            {bundle.mappingNotes?.map((note, i) => (
              <li key={i}>{note}</li>
            ))}
          </ul>
          <div className="report-downloads">
            {bundle.sourceRefs?.map((ref, i) => (
              <ReportDownload assetId={ref.assetId} key={i}>
                {t("reportOriginalEvidence")}
              </ReportDownload>
            ))}
          </div>
        </details>
      )}
    </details>
  );
}

export { HistoricalCAD as OriginalCadEvidence };
