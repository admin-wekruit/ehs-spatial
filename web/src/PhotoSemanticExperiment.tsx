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
const variantLabels: Record<string, string> = { single_view: "单张裁剪", multiview: "多视角语义", spatial_semantic: "多视角 + 空间约束" };
const evidenceLabels: Record<string, string> = {
  applicability_confirmation: "确认该要求是否适用", verified_metric_scale: "验证真实尺度", hazard_relationship: "确认对象与危险源的关系",
  policy_source_and_version: "指定政策来源和版本", functional_confirmation: "确认设备功能", stopping_performance: "制停性能数据",
  equipment_and_operation_context: "设备与作业背景", endpoint_correspondence: "确认实物测量端点",
  functional_identity: "确认保护装置的实际功能", equipment_operation_context: "设备类型与运行方式",
  stopping_performance_and_detection_zone: "制停性能、检测区域及验证资料",
};
const scoreLabel = (score: number) => Number.isFinite(score) ? score.toFixed(3) : "未知";

export function PhotoSemanticExperiment({ data, selectedEntityId, onSelect }: { data: SemanticExperiment; selectedEntityId: string | null; onSelect: SelectObject }) {
  const queries = data.queries ?? [], [queryId, setQueryId] = useState(queries[0]?.id ?? ""), [encoder, setEncoder] = useState("");
  const query = queries.find(item => item.id === queryId) ?? queries[0];
  const encoders = [...new Set(query?.results?.map(item => item.encoder) ?? [])], currentEncoder = encoders.includes(encoder) ? encoder : encoders[0];
  const results = query?.results?.filter(item => item.encoder === currentEncoder) ?? [];
  const run = data.timing?.run;
  return <section id="semantics" className="photo-semantic-experiment" aria-label="空间到语义实验">
    <div className="photo-semantic-heading"><div><p className="photo-report-eyebrow">SPATIAL → SEMANTICS → POLICY EVIDENCE</p><h2>用语义找到三维对象</h2></div><span>实验结果 · 共用现有模型</span></div>
    <p>选择一句预设查询，再点结果，在场景中查看对应物体及其在当前模型版本中的测量。这里只展示已计算的查询，不执行在线搜索。</p>
    {data.binding && <p className="photo-semantic-note" data-semantic-binding={data.binding.revisionId}>语义结果在 {data.binding.experimentRevisionId} 上计算；当前版本 {data.binding.revisionId} 的照片、分割多边形、深度点与地面变换已逐项核对一致后复用。空间数值只读取当前版本的测量。</p>}
    <p><a href="https://hovsg.github.io/" target="_blank" rel="noreferrer">HOV-SG</a>：借鉴其特征融合与空间关联，非原版复现。</p>
    {run && <p className="photo-semantic-run"><strong>增量语义实验（复用已有重建）</strong>{Number.isFinite(run.containerSeconds) && <span>容器内计算 {run.containerSeconds!.toFixed(1)} 秒</span>}{Number.isFinite(run.callSeconds) && <span>调用总耗时 {run.callSeconds!.toFixed(1)} 秒</span>}{Number.isFinite(run.estimateUsd) && <span>函数执行费用估算 ${run.estimateUsd!.toFixed(3)}</span>}{Number.isFinite(run.allAttemptsCallWindowEstimateUsd) && <span>含一次失败尝试，调用窗口资源粗估 ${run.allAttemptsCallWindowEstimateUsd!.toFixed(2)}</span>}<small>函数费用不含启动与收尾；调用窗口粗估含等待，均非账单。本次不包含照片重建或模型生成；实际账单{Number.isFinite(run.actualBilledUsd) ? ` $${run.actualBilledUsd!.toFixed(3)}` : "尚未核对"}。</small></p>}
    <div className="photo-semantic-queries" aria-label="预设语义查询">{queries.map(item => <button type="button" key={item.id} aria-pressed={query?.id === item.id} onClick={() => setQueryId(item.id)}>{item.label}</button>)}</div>
    {!queries.length && <p>本次报告没有保存查询结果。</p>}
    {query && <div className="photo-semantic-query-results">
      <div className="photo-semantic-query-title"><p>{query.phrase || query.label}</p>{encoders.length > 0 && <label>编码器<select value={currentEncoder} onChange={event => setEncoder(event.target.value)}>{encoders.map(item => <option key={item}>{item}</option>)}</select></label>}</div>
      <p className="photo-semantic-note">余弦相似度用于本组排序，不是识别概率。点击结果，右侧会显示三个方案及其原图证据。</p>
      <ol>{results.map((item, index) => {
        const object = data.objects?.find(row => row.entityId === item.entityId), source = object?.sourceRefs?.[0];
        return <li key={`${item.encoder}:${item.entityId}:${index}`}><button type="button" aria-pressed={selectedEntityId === item.entityId} onClick={() => onSelect(item.entityId, source?.photo, source?.observationId)}><span>{index + 1}. {item.label}<small>{item.entityId}</small></span><span>{scoreLabel(item.score)}<small>余弦相似度</small></span></button></li>;
      })}</ol>
      {!results.length && <p>该查询没有可显示的结果。</p>}
    </div>}
    {!!data.summary?.length && <details className="photo-semantic-summary"><summary>三个方案的实验对照</summary><p>下列“目录一致”只与已有对象目录比较，目录并非独立标注真值，不能作为识别准确率。跨照片项比较最近邻配对；空间方案先筛选重合候选，不使用合并簇的语义门槛，因此不代表整簇正确率，也不验证物理尺寸。</p><ul>{data.summary.map(row => <li key={`${row.encoder}:${row.variant}`}><strong>{row.encoder} · {variantLabels[row.variant] ?? row.variant}</strong><span>目录一致 {row.referenceAgreement ? `${row.referenceAgreement.correct} / ${row.referenceAgreement.total}` : "未记录"}{row.crossView ? `；跨照片最近邻 ${row.crossView.matched} / ${row.crossView.eligible}，其中目录身份一致 ${row.crossView.referenceCorrect}` : ""}</span></li>)}</ul></details>}
  </section>;
}

export function PhotoSemanticObject({ data, entityId, onSelect, facts, resolve = path => path, onMeasure, measureLabel }: { data: SemanticExperiment; entityId: string; onSelect: SelectObject; facts: SpatialFact[]; resolve?: Resolve; onMeasure?: () => void; measureLabel?: string }) {
  const object = data.objects?.find(item => item.entityId === entityId), [encoder, setEncoder] = useState("");
  if (!object) return null;
  const encoders = [...new Set(object.results?.map(item => item.encoder) ?? [])], currentEncoder = encoders.includes(encoder) ? encoder : encoders[0];
  const policy = object.policyContext;
  const supportById = new Map<string, { object: SemanticObject; source: SourceRef }>();
  for (const sourceObject of data.objects ?? []) for (const source of sourceObject.sourceRefs ?? []) {
    for (const id of [source.experimentObservationId, source.observationId]) if (id) supportById.set(id, { object: sourceObject, source });
  }
  return <section className="photo-semantic-object" data-semantic-entity={entityId} aria-label="当前对象的语义证据">
    <h3>语义候选与来源</h3><p>对象：{object.label}。候选来自图像与三维关联，名称仍待核对。</p>
    {encoders.length > 0 && <label>对比编码器<select value={currentEncoder} onChange={event => setEncoder(event.target.value)}>{encoders.map(item => <option key={item}>{item}</option>)}</select></label>}
    <div className="photo-semantic-variants">{(object.results ?? []).filter(item => item.encoder === currentEncoder).map(result => <article key={result.variant}>
      <h4>{variantLabels[result.variant] ?? result.variant}<small>{result.viewCount ?? "—"} 个视角</small></h4>
      {result.top3?.[0] ? <p><strong>{result.top3[0].label}</strong><span>{scoreLabel(result.top3[0].score)}</span></p> : <p>没有候选</p>}
      {(!!result.top3?.length || !!result.supportObservationIds?.length) && <details className="photo-semantic-sources"><summary>前三名与实际支持观测</summary><ol>{result.top3?.map(item => <li key={item.id}><span>{item.label}</span><span>{scoreLabel(item.score)}</span></li>)}</ol>
        <p>以下裁剪实际参与这个方案。来自不同目录对象的观测也保留，供检查跨对象关联。</p>
        <div>{result.supportObservationIds?.map(id => {
          const support = supportById.get(id);
          if (!support) return <p key={id} data-support-observation={id}>观测 {id}：缺少裁剪来源，无法核对。</p>;
          const { object: sourceObject, source } = support;
          return <button type="button" key={id} data-support-observation={id} data-support-entity={sourceObject.entityId} onClick={() => onSelect(sourceObject.entityId, source.photo, source.observationId)}>{source.cropPath && <img loading="lazy" src={resolve(source.cropPath)} alt={`${sourceObject.label}，照片 ${source.photo}，方案支持观测 ${id}`} />}<span>{sourceObject.label} · 照片 {source.photo}</span><small>{id}</small>{sourceObject.entityId !== entityId && <span>来自其他目录对象 · 核对关联</span>}</button>;
        })}</div>
        {!result.supportObservationIds?.length && <p>未记录支持观测。</p>}
      </details>}
    </article>)}</div>
    <p className="photo-semantic-note">分数为余弦相似度，不是概率；不同编码器的分数不能直接比较。</p>
    {!!object.sourceRefs?.length && <details className="photo-semantic-sources"><summary>原目录对象的裁剪（{object.sourceRefs.length}）</summary><div>{object.sourceRefs.map((source, index) => <button type="button" key={`${source.photo}:${source.observationId ?? index}`} onClick={() => onSelect(entityId, source.photo, source.observationId)}>{source.cropPath && <img loading="lazy" src={resolve(source.cropPath)} alt={`${object.label}，照片 ${source.photo} 的分割裁剪`} />}<span>照片 {source.photo}</span></button>)}</div></details>}
    <section className="photo-semantic-policy"><h4>接到 EHS 检查还需要什么</h4><p>适用性待确认 · 尚无安全判定</p>
      {!!policy?.candidateTopics?.length && <><h5>候选检查主题</h5><ul>{policy.candidateTopics.map(topic => <li key={topic}>{topic}</li>)}</ul></>}
      <details open data-semantic-facts={entityId}><summary>可用的空间证据（当前模型版本）</summary><p>以下数值实时读取当前版本的模型测量和标尺；切换版本或修改标尺后同步变化。</p>
        {facts.length ? <dl>{facts.map((fact, index) => <div key={index}><dt>{fact.label}</dt><dd data-semantic-fact={fact.testId}>{fact.value}</dd>{fact.status && <small>{fact.status}</small>}{fact.source && <small>来源：{fact.source}</small>}</div>)}</dl> : <p>本版本没有这个对象的结构化空间测量；缺少的事实不会用文字或语言模型补全。</p>}
        {onMeasure && <button type="button" onClick={onMeasure}>{measureLabel ?? "在 3D 中查看该对象的测量"}</button>}</details>
      <h5>待补证据</h5><ul>{(policy?.missingEvidence?.length ? policy.missingEvidence : ["applicability_confirmation", "policy_source_and_version"]).map(item => <li key={item}>{evidenceLabels[item] ?? item}</li>)}</ul>
    </section>
  </section>;
}
