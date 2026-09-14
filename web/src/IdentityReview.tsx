import { useEffect, useState } from 'react';
import { ApiError, id, request, type IdentitySuggestion } from './api';
import { activeModel, currentEntityId, jsonObject, observationsFor } from './core';
import { PhotoView } from './PhotoView';
import { ErrorNotice } from './App';
import { useI18n } from './i18n';
import { entityEvidenceStatus } from './scene-semantics';
import { identityDecisionKeys, measurementLabelKeys } from './identity-messages';
import type { Entity, Operation, Revision, SceneDocument } from './types';

const rows = (value: unknown): Record<string, unknown>[] => Array.isArray(value) ? value.map(jsonObject).filter((row): row is Record<string, unknown> => !!row) : [];
export function identityEvidence(document: SceneDocument, entities: Entity[]) {
  const observations = [...new Set(entities.flatMap(entity => entity.observationRefs || []))];
  const evidence: Record<string, unknown>[] = document.observations.filter(observation => observations.includes(observation.id)).map(observation =>
    ({kind:'observation', observationId:observation.id, observationRevision:observation.revision}));
  for (const assetId of new Set(entities.flatMap(entity => (entity.representations || []).flatMap(rep => rep.assetId ? [rep.assetId] : [])))) {
    const asset = document.assets.find(asset => asset.id === assetId), sha256 = asset?.sha256 || jsonObject(asset?.metadata)?.sha256;
    if (typeof sha256 === 'string') evidence.push({kind:'asset',assetId,sha256});
  }
  return evidence;
}
export function identityOperations(revision: Revision, entities: [Entity, Entity], decision: IdentitySuggestion['decision'], reason: string, survivorId: string, activeModelRepresentationId: string | null, decisionId: string, supersedesDecisionId: string | null): Operation[] {
  if (!reason.trim() || entities[0].id === entities[1].id) throw Error('invalid_identity_decision');
  const entityIds = entities.map(entity => entity.id), observationGroups = entities.map(entity => entity.observationRefs || []);
  const evidenceRefs = identityEvidence(revision.document, entities);
  if (!evidenceRefs.length) throw Error('identityEvidenceMissing');
  return [{type:'recordIdentityDecision',decision:{id:decisionId,decision,source:'manual',baseRevisionId:revision.id,entityIds,observationGroups,
    survivorId:decision === 'same' ? survivorId : null,evidenceRefs,reason:reason.trim(),supersedesDecisionId}},
    ...(decision === 'same' ? [{type:'mergeEntities',entityIds,survivorId,decisionId,activeModelRepresentationId}] : [])];
}
export function IdentityReview({revision, entityId, canWrite, publicationId, onSelect, onApply, onSuggest, onAgent}: {
  revision: Revision; entityId: string; canWrite: boolean; publicationId?: string;
  onSelect: (entityId: string, observationId?: string) => void;
  onApply: (operations: Operation[], context: {baseRevisionId:string}) => Promise<unknown>;
  onSuggest: (suggestion: IdentitySuggestion) => void;
  onAgent: (entityIds: [string,string]) => void;
}) {
  const {t} = useI18n(), document = revision.document;
  const [anchorId,setAnchor] = useState(entityId), [otherId,setOther] = useState(''), [search,setSearch] = useState(''),
    [decision,setDecision] = useState<IdentitySuggestion['decision']>('undecided'), [reason,setReason] = useState(''),
    [survivorId,setSurvivor] = useState(entityId), [modelId,setModel] = useState<string|null>(null), [supersedes,setSupersedes] = useState<string|null>(null),
    [preview,setPreview] = useState<{revisionId:string;operations:Operation[];suggestion:IdentitySuggestion}|null>(null), [error,setError] = useState<unknown>(), [busy,setBusy] = useState(false),
    [sharedError,setSharedError] = useState<unknown>(), [sharedRefresh,setSharedRefresh] = useState(0),
    [shared,setShared] = useState<{suggestionId:string;revisionId:string;identitySuggestion:IdentitySuggestion}[]>([]);
  useEffect(() => { if (entityId !== anchorId && entityId !== otherId) { setAnchor(entityId);setOther('');setPreview(null);setReason('');setSurvivor(entityId); } },[entityId,anchorId,otherId]);
  useEffect(() => { setPreview(null); },[revision.id]);
  useEffect(() => { let live=true;setSharedError(undefined);if(publicationId) request<{items:typeof shared}>(`/api/publications/${publicationId}/identity-suggestions`).then(result=>{if(live)setShared(result.items);}).catch(error=>{if(live)setSharedError(error);});return()=>{live=false;}; },[publicationId,sharedRefresh]);
  const first=document.entities.find(entity=>entity.id===anchorId), second=document.entities.find(entity=>entity.id===otherId);
  if(!first) return null;
  const pair: [Entity,Entity] | null = second ? [first,second] : null;
  const models=(pair || [first]).flatMap(entity=>(entity.representations || []).filter(rep=>['generated_mesh','primitive'].includes(rep.kind)).map(rep=>({entity,rep})));
  const candidates = new Set(rows(jsonObject(first.associationEvidence)?.candidates).flatMap(link=>Array.isArray(link.observationIds)?link.observationIds:[]));
  const others=document.entities.filter(entity=>!entity.sourceContext && entity.id!==first.id && (entity.id===otherId || `${entity.label||''} ${entity.id}`.toLowerCase().includes(search.toLowerCase())));
  const choose=(value:string)=>{const other=document.entities.find(entity=>entity.id===value);setOther(value);setPreview(null);setSurvivor(first.id);setSupersedes(null);
    const available=[first,...(other?[other]:[])].flatMap(entity=>(entity.representations||[]).filter(rep=>['generated_mesh','primitive'].includes(rep.kind)));
    setModel(activeModel(first)?.id || (available.length===1?available[0].id:null));};
  const decisions=rows(document.identityDecisions).filter(row=>{const targets=Array.isArray(row.entityIds)?row.entityIds.flatMap(id=>{const target=typeof id==='string'?currentEntityId(document,id):null;return target?[target]:[];}):[];return targets.length>0&&targets.includes(first.id)&&targets.every(id=>id===first.id||id===second?.id);});
  async function apply() { if(!preview||busy)return;setBusy(true);setError(undefined);try{if(preview.revisionId!==revision.id)throw new ApiError(409,'revision_conflict');const result=await onApply(preview.operations,{baseRevisionId:preview.revisionId});if(result===false)throw Error("identityStale");setPreview(null);}catch(error){setError(error);}finally{setBusy(false);} }
  function prepare() { if(!pair)return;setError(undefined);try { setPreview({revisionId:revision.id,
    operations:publicationId ? [] : identityOperations(revision,pair,decision,reason,survivorId,modelId,id(),supersedes),
    suggestion:{decision,entityIds:[first!.id,second!.id],observationGroups:pair.map(entity=>entity.observationRefs||[]),reason:reason.trim(),shareForReview:true}});
  }catch(error){setError(error);} }
  return <details className="identity-review"><summary>{t('identityTitle')}</summary>
    <label className="field-label">{t('identityOther')}<input type="search" value={search} onChange={event=>setSearch(event.target.value)} placeholder={t('sceneSearch')}/>
      <select value={otherId} onChange={event=>choose(event.target.value)}><option value="">—</option>{others.map(entity=><option key={entity.id} value={entity.id}>{entity.label||entity.id} · {entity.id.slice(0,8)}{entity.observationRefs?.some(id=>candidates.has(id))?' · '+t('identityCandidates'):''}</option>)}</select></label>
    {pair && <>
      <div className="identity-photos">{pair.map(entity=><IdentityPhoto key={entity.id} document={document} entity={entity} onSelect={onSelect}/>)}</div>
      <p className="subtle">{t(entityEvidenceStatus(document, first).identityKey || 'entityAssociationNotChecked')}</p>
      <div className="identity-decisions">{(['same','different','undecided'] as const).map(value=><button key={value} aria-pressed={decision===value} onClick={()=>{setDecision(value);setPreview(null);}}>{t({same:'identitySame',different:'identityDifferent',undecided:'identityUndecided'}[value])}</button>)}</div>
      <label className="field-label">{t('identityReason')}<textarea value={reason} onChange={event=>{setReason(event.target.value);setPreview(null);}}/></label>
      {decision==='same' && <><label className="field-label">{t('identityKeep')}<select value={survivorId} onChange={event=>{setSurvivor(event.target.value);setPreview(null);}}>{pair.map(entity=><option key={entity.id} value={entity.id}>{entity.label} · {entity.id.slice(0,8)}</option>)}</select></label>
        <label className="field-label">{t('identityModel')}<select value={modelId||''} onChange={event=>{setModel(event.target.value||null);setPreview(null);}}><option value="">{t('identityNoModel')}</option>{models.map(({entity,rep})=><option key={rep.id} value={rep.id}>{entity.label} · {rep.id.slice(0,8)} · {t(rep.kind)}</option>)}</select></label></>}
      {!!decisions.length && <label className="field-label">{t('identitySupersedes')}<select value={supersedes||''} onChange={event=>{setSupersedes(event.target.value||null);setPreview(null);}}><option value="">{t('identityNoSupersedes')}</option>{decisions.map(row=><option key={String(row.id)} value={String(row.id)}>{t(identityDecisionKeys[String(row.decision)] || 'identityDecisionUnknown')} · {String(row.reason)}</option>)}</select></label>}
      <button disabled={!reason.trim()||busy} onClick={prepare}>{t('identityPreview')}</button>
      {!publicationId&&canWrite&&<button onClick={()=>onAgent([first.id,second!.id])}>{t('agent')}</button>}
      {preview && <div className="identity-preview"><strong>{t('identityPreview')} · {t(identityDecisionKeys[preview.suggestion.decision])}</strong><p>{pair.map(entity=>`${entity.label} (${entity.observationRefs?.length||0})`).join(' + ')}</p>{decision==='same'&&<><p>{t('identityKeep')}: {survivorId.slice(0,8)}</p><p>{t('identityModel')}: {modelId?.slice(0,8)||t('identityNoModel')}</p></>}<p>{reason}</p><p>{t(decision==='same'?'identityPreviewNote':decision==='different'?'identityDifferentPreviewNote':'identityUndecidedPreviewNote')}</p>
        {publicationId?<><p>{t('identityPublicNote')}</p><button onClick={()=>onSuggest(preview.suggestion)}>{t('identitySuggest')}</button></>:<button disabled={!canWrite||busy} onClick={apply}>{t('identityApply')}</button>}</div>}
    </>}
    <ErrorNotice error={error}/><ErrorNotice error={sharedError}/>{!!sharedError&&<button onClick={()=>setSharedRefresh(value=>value+1)}>{t("retry")}</button>}
    {!!shared.length&&<details><summary>{t('identityShared')}</summary>{shared.filter(row=>row.identitySuggestion.entityIds.includes(first.id)).map(row=><div key={row.suggestionId}><p>{row.identitySuggestion.reason}</p><small>{row.revisionId.slice(0,8)}</small><button onClick={()=>{const other=row.identitySuggestion.entityIds.find(id=>id!==first.id);if(other){choose(other);setDecision(row.identitySuggestion.decision);setReason(row.identitySuggestion.reason);}}}>{t('identityCopyReview')}</button></div>)}</details>}
    {canWrite && (first.observationRefs?.length||0)>1 && <IdentitySplit revision={revision} entity={first} onApply={onApply}/>}
  </details>;
}
function IdentityPhoto({document,entity,onSelect}:{document:SceneDocument;entity:Entity;onSelect:(id:string,observationId?:string)=>void}) {
  const {t}=useI18n(), observations=observationsFor(document,entity), [observationId,setObservation]=useState(observations[0]?.id||'');
  const observation=observations.find(value=>value.id===observationId);
  return <div><strong>{entity.label||entity.id}</strong><small>{entity.id.slice(0,8)}</small>{observation?<><select aria-label={t('photos')} value={observationId} onChange={event=>setObservation(event.target.value)}>{observations.map((value,i)=><option key={value.id} value={value.id}>{t('photos')} {document.assets.filter(asset=>asset.kind==='source_image').findIndex(asset=>asset.id===value.imageId)+1 || '—'} · {value.id.slice(0,6)}</option>)}</select><div className="identity-photo"><PhotoView document={{...document,entities:[entity]}} imageId={observation.imageId} selectedId={entity.id} onSelect={onSelect}/></div><button onClick={()=>onSelect(entity.id,observation.id)}>{t('identityView')}</button></>:<p>{t('identityNoPhoto')}</p>}</div>;
}
function IdentitySplit({revision,entity,onApply}:{revision:Revision;entity:Entity;onApply:(operations:Operation[],context:{baseRevisionId:string})=>Promise<unknown>}) {
  const {t}=useI18n(), [assign,setAssign]=useState<Record<string,string>>({}),[reason,setReason]=useState(''),[supersedes,setSupersedes]=useState<string|null>(null),[preview,setPreview]=useState<{operations:Operation[];revisionId:string}|null>(null),[error,setError]=useState<unknown>(),[busy,setBusy]=useState(false);
  useEffect(()=>{setPreview(null);},[revision.id,entity.id]);
  const decisions=(revision.document.identityDecisions||[]).filter(decision=>decision.entityIds.length>0&&decision.entityIds.every(id=>currentEntityId(revision.document,id)===entity.id));
  const observations=observationsFor(revision.document,entity), reps=entity.representations||[], measurements=rows(entity.measurementEvidence);
  const sources=[...observations.map(row=>({id:row.id,title:t('photos')+' · '+row.id.slice(0,8),kind:'observation'})),...reps.map(row=>({id:row.id,title:t(row.kind)+' · '+row.id.slice(0,8),kind:'representation'})),...measurements.map(row=>({id:String(row.id),title:t(measurementLabelKeys[String(row.measurementKey)] || 'identityMeasurementOther'),kind:'measurement'}))];
  function prepare(){try{const decisionId=id(),groups=['a','b'].map(key=>({id:id(),observationRefs:observations.filter(row=>assign[row.id]===key).map(row=>row.id),representationIds:reps.filter(row=>assign[row.id]===key).map(row=>row.id),measurementEvidenceIds:measurements.filter(row=>assign[String(row.id)]===key).map(row=>row.id),activeModelRepresentationId:reps.find(row=>row.id===entity.activeModelRepresentationId&&assign[row.id]===key)?.id||null}));
    if(groups.some(group=>!group.observationRefs.length)||sources.some(source=>!assign[source.id]))throw Error('identityAssign');
    setPreview({revisionId:revision.id,operations:[{type:'recordIdentityDecision',decision:{id:decisionId,decision:'different',source:'manual',baseRevisionId:revision.id,entityIds:[entity.id],observationGroups:groups.map(group=>group.observationRefs),survivorId:null,evidenceRefs:identityEvidence(revision.document,[entity]),reason,supersedesDecisionId:supersedes}},{type:'splitEntity',entityId:entity.id,decisionId,groups,retainedMeasurementEvidenceIds:measurements.filter(row=>assign[String(row.id)]==='retain').map(row=>row.id)}]});
  }catch(error){setError(error);}}
  return <details className="identity-split"><summary>{t('identitySplit')}</summary><p>{t('identitySplitNote')}</p>{sources.map(source=><label key={source.id}>{source.title}<select value={assign[source.id]||''} onChange={event=>{setAssign({...assign,[source.id]:event.target.value});setPreview(null);}}><option value="">—</option><option value="a">A</option><option value="b">B</option>{source.kind==='measurement'&&<option value="retain">{t('identityRetainMeasurement')}</option>}</select></label>)}{!!decisions.length&&<label className="field-label">{t('identitySupersedes')}<select value={supersedes||''} onChange={event=>{setSupersedes(event.target.value||null);setPreview(null);}}><option value="">{t('identityNoSupersedes')}</option>{decisions.map(decision=><option key={decision.id} value={decision.id}>{t(identityDecisionKeys[decision.decision] || 'identityDecisionUnknown')} · {decision.reason}</option>)}</select></label>}<label className="field-label">{t('identityReason')}<textarea value={reason} onChange={event=>{setReason(event.target.value);setPreview(null);}}/></label><button disabled={!reason.trim()} onClick={prepare}>{t('identitySplitPreview')}</button>{preview&&<><p>{t('identitySplitNote')}</p><button disabled={busy} onClick={async()=>{setBusy(true);try{if(preview.revisionId!==revision.id)throw new ApiError(409,"revision_conflict");const result=await onApply(preview.operations,{baseRevisionId:preview.revisionId});if(result===false)throw Error("identityStale");setPreview(null);}catch(error){setError(error);}finally{setBusy(false);}}}>{t('identityApply')}</button></>}<ErrorNotice error={error}/></details>;
}
