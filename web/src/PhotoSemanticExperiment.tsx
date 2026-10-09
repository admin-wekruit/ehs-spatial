import { useI18n } from "./i18n";
import { useState } from "react";

type Candidate = { id: string; label: string; phrase?: string; score: number };
type SemanticResult = { encoder: string; variant: string; top3?: Candidate[]; viewCount?: number; supportObservationIds?: string[] };
type SourceRef = { photo: number; observationId?: string; experimentObservationId?: string; cropPath?: string };
type SemanticObject = {
  entityId: string; label: string; sourceRefs?: SourceRef[]; results?: SemanticResult[];
  policyContext?: { applicability?: string; machineResult?: unknown; candidateTopics?: string[]; missingEvidence?: string[] };
};
/** A spatial fact of the currently loaded revision, formatted by the page with its current scale. */
export type SpatialFact = { label: string; value: string; status?: string; source?: string; testId?: string };
type Resolve = (path: string) => string;
export type SemanticExperiment = {
  binding?: { revisionId: string; documentSha256: string; experimentRevisionId: string; reuse: string; verified?: string[]; maxCropResampleDifference?: number };
  sourceRevisionId?: string;
  status?: string; method?: unknown; protocol?: unknown; timing?: { run?: { containerSeconds?: number; callSeconds?: number; estimateUsd?: number; callWindowEstimateUsd?: number; allAttemptsCallWindowEstimateUsd?: number; actualBilledUsd?: number | null }; [key: string]: unknown };
  summary?: { encoder: string; variant: string; objectCount?: number; referenceAgreement?: { correct: number; total: number }; crossView?: { eligible: number; matched: number; referenceCorrect: number; precision: number | null; coverage: number | null } }[];
  objects?: SemanticObject[];
  queries?: { id: string; label: string; phrase?: string; results?: { encoder: string; entityId: string; label: string; score: number }[] }[];
};
type SelectObject = (entityId: string, photo?: number, observationId?: string) => void;
const variantLabels: Record<string, string> = { single_view: "PhotoSemanticExperimenttsx.text269", multiview: "PhotoSemanticExperimenttsx.text270", spatial_semantic: "PhotoSemanticExperimenttsx.text271" };
const evidenceLabels: Record<string, string> = {
  applicability_confirmation: "PhotoSemanticExperimenttsx.text272", verified_metric_scale: "PhotoSemanticExperimenttsx.text273", hazard_relationship: "PhotoSemanticExperimenttsx.text274",
  policy_source_and_version: "PhotoSemanticExperimenttsx.text275", functional_confirmation: "PhotoSemanticExperimenttsx.text276", stopping_performance: "PhotoSemanticExperimenttsx.text277",
  equipment_and_operation_context: "PhotoSemanticExperimenttsx.text278", endpoint_correspondence: "PhotoSemanticExperimenttsx.text279",
  functional_identity: "PhotoSemanticExperimenttsx.text280", equipment_operation_context: "PhotoSemanticExperimenttsx.text281",
  stopping_performance_and_detection_zone: "PhotoSemanticExperimenttsx.text282",
};
const scoreLabel = (score: number, t: (id: string) => string) => Number.isFinite(score) ? score.toFixed(3) : t("unknown");

export function PhotoSemanticExperiment({ data, selectedEntityId, onSelect }: { data: SemanticExperiment; selectedEntityId: string | null; onSelect: SelectObject }) {
  const { t } = useI18n();
  const queries = data.queries ?? [], [queryId, setQueryId] = useState(queries[0]?.id ?? ""), [encoder, setEncoder] = useState("");
  const query = queries.find(item => item.id === queryId) ?? queries[0];
  const encoders = [...new Set(query?.results?.map(item => item.encoder) ?? [])], currentEncoder = encoders.includes(encoder) ? encoder : encoders[0];
  const results = query?.results?.filter(item => item.encoder === currentEncoder) ?? [];
  const run = data.timing?.run;
  return <section id="semantics" className="photo-semantic-experiment" aria-label={t("PhotoSemanticExperimenttsx.text283")}>
    <div className="photo-semantic-heading"><div><p className="photo-report-eyebrow">{t("photo.semanticFlow")}</p><h2>{t("PhotoSemanticExperimenttsx.text284")}</h2></div><span>{t("PhotoSemanticExperimenttsx.text285")}</span></div>
    <p>{t("PhotoSemanticExperimenttsx.text286")}</p>
    {data.binding && <p className="photo-semantic-note" data-semantic-binding={data.binding.revisionId}>{t("PhotoSemanticExperimenttsx.text287")}{data.binding.experimentRevisionId} {t("PhotoSemanticExperimenttsx.text288")}{data.binding.revisionId} {t("PhotoSemanticExperimenttsx.text289")}</p>}
    <p><a href="https://hovsg.github.io/" target="_blank" rel="noreferrer">HOV-SG</a>{t("PhotoSemanticExperimenttsx.text290")}</p>
    {run && <p className="photo-semantic-run"><strong>{t("PhotoSemanticExperimenttsx.text291")}</strong>{Number.isFinite(run.containerSeconds) && <span>{t("PhotoSemanticExperimenttsx.text292")}{run.containerSeconds!.toFixed(1)} {t("seconds")}</span>}{Number.isFinite(run.callSeconds) && <span>{t("PhotoSemanticExperimenttsx.text293")}{run.callSeconds!.toFixed(1)} {t("seconds")}</span>}{Number.isFinite(run.estimateUsd) && <span>{t("PhotoSemanticExperimenttsx.text294")}{run.estimateUsd!.toFixed(3)}</span>}{Number.isFinite(run.allAttemptsCallWindowEstimateUsd) && <span>{t("PhotoSemanticExperimenttsx.text295")}{run.allAttemptsCallWindowEstimateUsd!.toFixed(2)}</span>}<small>{t("PhotoSemanticExperimenttsx.text296")}{Number.isFinite(run.actualBilledUsd) ? ` $${run.actualBilledUsd!.toFixed(3)}` : t("PhotoSemanticExperimenttsx.text297")}. </small></p>}
    <div className="photo-semantic-queries" aria-label={t("PhotoSemanticExperimenttsx.text298")}>{queries.map(item => <button type="button" key={item.id} aria-pressed={query?.id === item.id} onClick={() => setQueryId(item.id)}>{item.label}</button>)}</div>
    {!queries.length && <p>{t("PhotoSemanticExperimenttsx.text299")}</p>}
    {query && <div className="photo-semantic-query-results">
      <div className="photo-semantic-query-title"><p>{query.phrase || query.label}</p>{encoders.length > 0 && <label>{t("PhotoSemanticExperimenttsx.text300")}<select value={currentEncoder} onChange={event => setEncoder(event.target.value)}>{encoders.map(item => <option key={item}>{item}</option>)}</select></label>}</div>
      <p className="photo-semantic-note">{t("PhotoSemanticExperimenttsx.text301")}</p>
      <ol>{results.map((item, index) => {
        const { t } = useI18n();
  const object = data.objects?.find(row => row.entityId === item.entityId), source = object?.sourceRefs?.[0];
        return <li key={`${item.encoder}:${item.entityId}:${index}`}><button type="button" aria-pressed={selectedEntityId === item.entityId} onClick={() => onSelect(item.entityId, source?.photo, source?.observationId)}><span>{index + 1}. {item.label}<small>{item.entityId}</small></span><span>{scoreLabel(item.score, t)}<small>{t("PhotoSemanticExperimenttsx.text302")}</small></span></button></li>;
      })}</ol>
      {!results.length && <p>{t("PhotoSemanticExperimenttsx.text303")}</p>}
    </div>}
    {!!data.summary?.length && <details className="photo-semantic-summary"><summary>{t("PhotoSemanticExperimenttsx.text304")}</summary><p>{t("PhotoSemanticExperimenttsx.text305")}</p><ul>{data.summary.map(row => <li key={`${row.encoder}:${row.variant}`}><strong>{row.encoder} · {t(variantLabels[row.variant] ?? row.variant)}</strong><span>{t("PhotoSemanticExperimenttsx.text306")}{row.referenceAgreement ? `${row.referenceAgreement.correct} / ${row.referenceAgreement.total}` : t("PhotoSemanticExperimenttsx.text307")}{row.crossView ? t("PhotoSemanticExperimenttsx.text308", {p0: row.crossView.matched, p1: row.crossView.eligible, p2: row.crossView.referenceCorrect}) : ""}</span></li>)}</ul></details>}
  </section>;
}

export function PhotoSemanticObject({ data, entityId, onSelect, facts, resolve = path => path, onMeasure, measureLabel }: { data: SemanticExperiment; entityId: string; onSelect: SelectObject; facts: SpatialFact[]; resolve?: Resolve; onMeasure?: () => void; measureLabel?: string }) {
  const { t } = useI18n();
  const object = data.objects?.find(item => item.entityId === entityId), [encoder, setEncoder] = useState("");
  if (!object) return null;
  const encoders = [...new Set(object.results?.map(item => item.encoder) ?? [])], currentEncoder = encoders.includes(encoder) ? encoder : encoders[0];
  const policy = object.policyContext;
  const supportById = new Map<string, { object: SemanticObject; source: SourceRef }>();
  for (const sourceObject of data.objects ?? []) for (const source of sourceObject.sourceRefs ?? []) {
    for (const id of [source.experimentObservationId, source.observationId]) if (id) supportById.set(id, { object: sourceObject, source });
  }
  return <section className="photo-semantic-object" data-semantic-entity={entityId} aria-label={t("PhotoSemanticExperimenttsx.text309")}>
    <h3>{t("PhotoSemanticExperimenttsx.text310")}</h3><p>{t("PhotoSemanticExperimenttsx.text311")}{object.label}{t("PhotoSemanticExperimenttsx.text312")}</p>
    {encoders.length > 0 && <label>{t("PhotoSemanticExperimenttsx.text313")}<select value={currentEncoder} onChange={event => setEncoder(event.target.value)}>{encoders.map(item => <option key={item}>{item}</option>)}</select></label>}
    <div className="photo-semantic-variants">{(object.results ?? []).filter(item => item.encoder === currentEncoder).map(result => <article key={result.variant}>
      <h4>{t(variantLabels[result.variant] ?? result.variant)}<small>{result.viewCount ?? "—"} {t("LiveReporttsx.056")}</small></h4>
      {result.top3?.[0] ? <p><strong>{result.top3[0].label}</strong><span>{scoreLabel(result.top3[0].score, t)}</span></p> : <p>{t("PhotoSemanticExperimenttsx.text314")}</p>}
      {(!!result.top3?.length || !!result.supportObservationIds?.length) && <details className="photo-semantic-sources"><summary>{t("PhotoSemanticExperimenttsx.text315")}</summary><ol>{result.top3?.map(item => <li key={item.id}><span>{item.label}</span><span>{scoreLabel(item.score, t)}</span></li>)}</ol>
        <p>{t("PhotoSemanticExperimenttsx.text316")}</p>
        <div>{result.supportObservationIds?.map(id => {
          const support = supportById.get(id);
          if (!support) return <p key={id} data-support-observation={id}>{t("sceneObservation")}{id}{t("PhotoSemanticExperimenttsx.text317")}</p>;
          const { object: sourceObject, source } = support;
          return <button type="button" key={id} data-support-observation={id} data-support-entity={sourceObject.entityId} onClick={() => onSelect(sourceObject.entityId, source.photo, source.observationId)}>{source.cropPath && <img loading="lazy" src={resolve(source.cropPath)} alt={t("PhotoSemanticExperimenttsx.text318", {p0: sourceObject.label, p1: source.photo, p2: id})} />}<span>{sourceObject.label} {t("PhotoReporttsx.text177")}{source.photo}</span><small>{id}</small>{sourceObject.entityId !== entityId && <span>{t("PhotoSemanticExperimenttsx.text319")}</span>}</button>;
        })}</div>
        {!result.supportObservationIds?.length && <p>{t("PhotoSemanticExperimenttsx.text320")}</p>}
      </details>}
    </article>)}</div>
    <p className="photo-semantic-note">{t("PhotoSemanticExperimenttsx.text321")}</p>
    {!!object.sourceRefs?.length && <details className="photo-semantic-sources"><summary>{t("PhotoSemanticExperimenttsx.text322", {p0: object.sourceRefs.length})}</summary><div>{object.sourceRefs.map((source, index) => <button type="button" key={`${source.photo}:${source.observationId ?? index}`} onClick={() => onSelect(entityId, source.photo, source.observationId)}>{source.cropPath && <img loading="lazy" src={resolve(source.cropPath)} alt={t("PhotoSemanticExperimenttsx.text323", {p0: object.label, p1: source.photo})} />}<span>{t("photos")}{source.photo}</span></button>)}</div></details>}
    <section className="photo-semantic-policy"><h4>{t("PhotoSemanticExperimenttsx.text324")}</h4><p>{t("PhotoSemanticExperimenttsx.text325")}</p>
      {!!policy?.candidateTopics?.length && <><h5>{t("PhotoSemanticExperimenttsx.text326")}</h5><ul>{policy.candidateTopics.map(topic => <li key={topic}>{topic}</li>)}</ul></>}
      <details open data-semantic-facts={entityId}><summary>{t("PhotoSemanticExperimenttsx.text327")}</summary><p>{t("PhotoSemanticExperimenttsx.text328")}</p>
        {facts.length ? <dl>{facts.map((fact, index) => <div key={index}><dt>{fact.label}</dt><dd data-semantic-fact={fact.testId}>{fact.value}</dd>{fact.status && <small>{fact.status}</small>}{fact.source && <small>{t("PhotoSemanticExperimenttsx.text329")}{fact.source}</small>}</div>)}</dl> : <p>{t("PhotoSemanticExperimenttsx.text330")}</p>}
        {onMeasure && <button type="button" onClick={onMeasure}>{measureLabel ?? t("PhotoSemanticExperimenttsx.text331")}</button>}</details>
      <h5>{t("PhotoSemanticExperimenttsx.text332")}</h5><ul>{(policy?.missingEvidence?.length ? policy.missingEvidence : ["applicability_confirmation", "policy_source_and_version"]).map(item => <li key={item}>{t(evidenceLabels[item] ?? item)}</li>)}</ul>
    </section>
  </section>;
}
