import { useEffect, useState } from "react";
import { resolveAsset } from "./api";
import { jsonObject } from "./core";
import { useI18n } from "./i18n";
import { ReportDownload } from "./WorkcellReport";
import { ErrorNotice } from "./App";
import { reportEvidenceMessages } from "./report-evidence-messages";
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
  labelZh?: string;
  note?: string;
  category?: string;
  sourceFrameId?: string;
  targetSourceFrameId?: string;
  sourceCandidateIds?: string[];
  imageId?: string | null;
  cameraId?: string | null;
  entityIds: string[];
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
    entityId: string;
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
export function interpretationSelection(
  document: SceneDocument,
  item: InterpretationItem,
  entityId: string,
): ReportEvidenceSelection | null {
  const entity = document.entities.find((e) => e.id === entityId);
  if (!entity || !item.entityIds.includes(entityId)) return null;
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
    matching.length === 1
      ? matching[0]
      : observations.length === 1
        ? observations[0]
        : undefined;
  const cameras = document.cameras.filter((c) => c.imageId === imageId);
  const camera =
    cameras.find((c) => c.id === item.cameraId) ||
    (cameras.length === 1 ? cameras[0] : undefined);
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
      item.entityIds.some(
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
    contexts: item.entityIds
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
  return { linked, unassociated: rows.filter((row) => !row.contexts.length) };
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
const labels: Record<string, [string, string]> = {
  visibleObjects: ["可见对象", "Visible objects"],
  location: ["位置", "Location"],
  reason: ["理由", "Reason"],
  reasons: ["理由", "Reasons"],
  summary: ["总结", "Summary"],
  description: ["说明", "Description"],
  assessment: ["分析", "Assessment"],
  limitations: ["局限", "Limitations"],
  scale: ["尺度依据", "Scale basis"],
  confidence: ["置信分数", "Confidence score"],
  threshold: ["阈值", "Threshold"],
  value: ["数值", "Value"],
  unit: ["单位", "Unit"],
  predicate: ["规则谓词", "Predicate"],
  evidence: ["证据", "Evidence"],
  source: ["来源", "Source"],
  sourceRunId: ["来源 run", "Source run"],
  reconstructionRunId: ["重建 run", "Reconstruction run"],
  meaning: ["含义", "Meaning"],
  missingEvidence: ["缺少依据", "Missing evidence"],
  warnings: ["注意事项", "Warnings"],
  violations: ["原始发现", "Original findings"],
  facts: ["输入事实", "Input facts"],
  measurement: ["测量", "Measurement"],
  distance: ["距离", "Distance"],
  status: ["状态", "Status"],
  scope: ["覆盖范围", "Scope"],
  sceneSummary: ["场景理解", "Scene interpretation"],
  overallAssessment: ["总体分析", "Overall assessment"],
  observations: ["照片观察", "Photo observations"],
};
function readable(key: string, language: string) {
  return (
    labels[key]?.[language === "zh" ? 0 : 1] ||
    key.replace(/([a-z])([A-Z])/g, "$1 $2").replaceAll("_", " ")
  );
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
          <dt>{readable(key, language)}</dt>
          <dd>
            <EvidenceValue value={v} />
          </dd>
        </div>
      ))}
    </dl>
  ) : null;
}
function HistoricalCAD({
  cad,
  onSelect,
}: {
  cad: NonNullable<NonNullable<ReportBundle["historical"]>["cad"]>;
  onSelect: (entityId: string) => void;
}) {
  const { t } = useI18n(),
    [url, setURL] = useState<string>(),
    [error, setError] = useState<unknown>();
  useEffect(() => {
    let live = true;
    setURL(undefined);
    setError(undefined);
    resolveAsset(cad.assetId)
      .then((u) => {
        if (live) setURL(u);
      })
      .catch((e) => {
        if (live) setError(e);
      });
    return () => {
      live = false;
    };
  }, [cad.assetId]);
  return (
    <figure className="report-cad-evidence">
      <h3>{t("reportCadOriginal")}</h3>
      <ErrorNotice error={error} />
      {url && (
        <svg
          viewBox={`0 0 ${cad.width} ${cad.height}`}
          aria-label={t("reportCadOriginal")}
        >
          <image href={url} width={cad.width} height={cad.height} />
          {cad.regions
            .filter((r) => r.entityIds.length)
            .map((r, i) => (
              <polygon
                key={i}
                points={r.polygon.map((p) => p.join(",")).join(" ")}
                tabIndex={0}
                role="button"
                aria-label={`${t("objects")} ${r.inventoryIndex}`}
                onClick={() => onSelect(r.entityIds[0])}
                onKeyDown={(e) => {
                  if (["Enter", " "].includes(e.key)) {
                    e.preventDefault();
                    onSelect(r.entityIds[0]);
                  }
                }}
              >
                <title>
                  {t("objects")} {r.inventoryIndex}
                </title>
              </polygon>
            ))}
        </svg>
      )}
      <figcaption>{t("reportCoordinateNote")}</figcaption>
      <ReportDownload assetId={cad.assetId}>CAD</ReportDownload>
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
  const { t: globalT, language } = useI18n();
  const t = (key: string) =>
    reportEvidenceMessages[key]?.[language === "zh" ? 0 : 1] || globalT(key);
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
          const { linked, unassociated } = partitionInterpretationItems(
            document,
            analysis.items,
            currentImageId,
          );
          const rows = (items: typeof linked) => (
            <div className="report-detection-notes">
              {items.map(({ item, contexts }, i) => {
                const label =
                  language === "zh" && item.labelZh ? item.labelZh : item.label;
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
                          {contexts.length
                            ? t("sourceEvidence")
                            : t("reportUnassociated")}
                          {item.mappingLimitation && (
                            <>
                              <br />
                              {item.mappingLimitation}
                            </>
                          )}
                        </small>
                      </details>
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
              open={index === 0}
            >
              <summary>
                {analysis.runId === historical?.runId
                  ? t("reportHistorical")
                  : t("reportUnderstanding")}{" "}
                · {linked.length} {t("reLinkedRecords")} /{" "}
                {analysis.items.length} {t("reSourceRecords")}
              </summary>
              <p className="report-evidence-note">
                {analysis.runId === historical?.runId
                  ? t("reportHistoricalHint")
                  : t("reportDetectionMeaning")}
              </p>
              {!!linked.length && rows(linked)}
              {!!unassociated.length && (
                <details className="report-source-details report-unassociated-records">
                  <summary>
                    {t("reUnassociatedRecords")} · {unassociated.length}
                  </summary>
                  <p className="report-evidence-note">
                    {t("reUnassociatedMeaning")}
                  </p>
                  {rows(unassociated)}
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
                        {item.entityIds.length ? (
                          <button onClick={() => onSelect(item.entityIds[0])}>
                            {item.label} ↗
                          </button>
                        ) : (
                          item.label
                        )}
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
        <p className="report-kicker">{historical.runId}</p>
        <p className="report-evidence-note">{t("rePolicySourceNote")}</p>
        <details className="report-source-details">
          <summary>{t("reportOriginalAssessment")}</summary>
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
                <span>{finding.title || finding.id}</span>
                <span className={"badge " + finding.status?.toLowerCase()}>
                  {t(finding.status || "unknown")}
                </span>
              </summary>
              {finding.summary && <p>{finding.summary}</p>}
              <div className="report-rule-basis">
                <h4>{t("rePolicyBasis")}</h4>
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
                      {readable("warnings", language)} ·{" "}
                      {finding.warnings.length}
                    </summary>
                    <EvidenceValue value={finding.warnings} />
                  </details>
                )}
              </div>
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
          );
        })}
      </div>
    ) : null;
  if (section === "assets")
    return (
      <>
        {historical?.cad && (
          <HistoricalCAD cad={historical.cad} onSelect={onSelect} />
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
    <div className="report-historical">
      <h3>{t("reportQuality")}</h3>
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
                  <th>Depth · {t("before")}</th>
                  <th>Depth · {t("after")}</th>
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
                        <small>
                          {t("reSourceRecord")} · {o.sourceRecordId}
                        </small>
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
    </div>
  );
}
