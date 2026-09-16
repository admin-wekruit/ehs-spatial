import { activeModel, isPartitionSource, jsonObject } from './core';
import { useI18n } from './i18n';
import { measurementLabelKeys } from './identity-messages';
import { modelLabelKey } from './scene-semantics';
import type { Entity, Operation } from './types';

export function ModelEvidence({entity,onCommit,onReview,disabled=false}:{entity:Entity;onCommit?:(operations:Operation[])=>unknown;onReview?:()=>unknown;disabled?:boolean}) {
  const {t}=useI18n(), models=(entity.representations||[]).filter(rep=>['generated_mesh','primitive'].includes(rep.kind)), active=activeModel(entity), selections=jsonObject(entity.measurementSelections)||{};
  const quality=jsonObject(active?.qualityEvidence), geometric=jsonObject(quality?.geometric), shape=jsonObject(quality?.shapeReview);
  const qualityKey=quality?.status==='accepted'?'modelReviewAccepted':quality?.status==='rejected'?'modelReviewRejected':quality?'modelReviewNeedsInformation':'modelReviewNotRun';
  const records=(Array.isArray(entity.measurementEvidence)?entity.measurementEvidence:[]).map(jsonObject).filter((row):row is Record<string,unknown>=>!!row);
  const keys=[...new Set(records.map(row=>String(row.measurementKey)))];
  return <>{active&&<section className="measurement-block" aria-label={t('modelReviewTitle')}>
    <strong>{t('modelReviewTitle')} · {t(qualityKey)}</strong>
    {geometric?.assessmentScope==='parent_family'&&<p>{t('modelReviewFamilyScope')}</p>}
    {quality&&<details><summary>{t('evidence')}</summary><p>{t('modelReviewScope')}</p>
      <dl className="measurement-list"><div><dt>{t('modelReviewGeometry')}</dt><dd>{t(geometric?.status==='observed_consistent'?'modelReviewGeometryConsistent':geometric?.status==='observed_inconsistent'?'modelReviewGeometryInconsistent':'modelReviewNeedsInformation')}</dd></div>
      <div><dt>{t('modelReviewShape')}</dt><dd>{t(shape?.status==='pass'?'modelReviewAccepted':shape?.status==='fail'?'modelReviewRejected':'modelReviewNeedsInformation')}</dd></div></dl>
      {typeof shape?.reason==='string'&&<p>{shape.reason==='shape_review_not_configured'?t('modelReviewNotConfigured'):shape.reason}</p>}
      {Array.isArray(shape?.visibleShapeIssues)&&shape.visibleShapeIssues.length>0&&<ul>{shape.visibleShapeIssues.filter((issue):issue is string=>typeof issue==='string').map((issue,index)=><li key={index}>{issue}</li>)}</ul>}
    </details>}
    {onReview&&<button className="wide" disabled={disabled} onClick={onReview}>{t('review_models')}</button>}
  </section>}{active&&modelLabelKey(active)!==active.kind&&<p className="evidence-note"><strong>{t(modelLabelKey(active))}</strong>{active.coverage==='observed_visible_surface_only'&&<><br/>{t('identityVisibleSurfaceOnly')}</>}{active.coverage==='inferred_planar_visible_region'&&<><br/>{t('identityInferredPlaneOnly')}</>}<br/>{t(active.placementState==='confirmed'?'identityPlacementConfirmed':active.placementState==='unconfirmed'?'identityPlacementUnconfirmed':'identityPlacementUnknown')}</p>}<details className="report-source-details"><summary>{t('identityMeasurementSources')}</summary>
    {active?.sourceValidity==='stale'&&<p className="evidence-note">{t('identityModelStaleNote')}</p>}
    {!!models.length&&<><label className="field-label">{t('identityCurrentModel')}<select disabled={!onCommit||disabled} value={active?.id||''} onChange={event=>onCommit?.([{type:'setActiveModelRepresentation',entityId:entity.id,representationId:event.target.value||null}])}>
      <option value="">{t('identityNoModel')}</option>{models.filter(rep=>!isPartitionSource(entity,rep.id)).map(rep=><option key={rep.id} value={rep.id}>{t(modelLabelKey(rep))} · {rep.id.slice(0,8)}{rep.sourceValidity==='stale'?' · '+t('identityModelStale'):''}</option>)}</select></label>
      <details><summary>{t('identityModelAlternatives')} · {models.length}</summary>{models.map(rep=><div key={rep.id}><strong>{rep.id.slice(0,8)} · {rep.id===active?.id?t('identityCurrentModel'):t('identitySourceModel')}</strong><p>{rep.coordinateFrameId} · {rep.transform.position.map(value=>value.toFixed(3)).join(' / ')}</p><small>{rep.sourceValidity==='stale'&&<>{t('identitySourceStale')} · </>}{t(rep.placementState==='confirmed'?'identityPlacementConfirmed':rep.placementState==='unconfirmed'?'identityPlacementUnconfirmed':'identityPlacementUnknown')} · {rep.assetId||t('primitive')}</small>{modelLabelKey(rep)!==rep.kind&&<details><summary>{t(modelLabelKey(rep))}</summary><pre>{JSON.stringify({sourceDerivation:rep.sourceDerivation,sourceRefs:rep.sourceRefs},null,2)}</pre></details>}</div>)}</details></>}
    {keys.map(key=><div key={key}><label className="field-label">{t(measurementLabelKeys[key] || 'identityMeasurementOther')}<select disabled={!onCommit||disabled} value={typeof selections[key]==='string'?String(selections[key]):''} onChange={event=>onCommit?.([{type:'selectMeasurementEvidence',entityId:entity.id,measurementKey:key,measurementEvidenceId:event.target.value||null}])}>
      <option value="">{t('identityNoMeasurement')}</option>{records.filter(row=>row.measurementKey===key).map(row=><option key={String(row.id)} value={String(row.id)}>{String(row.sourceEntityId).slice(0,8)} · {String(row.sourceRevisionId).slice(0,8)}</option>)}</select></label>
      {typeof selections[key]!=='string'&&<p className="subtle">{t('identityMeasurementConflict')}</p>}<details><summary>{t('identitySourceMeasurement')}</summary>{records.filter(row=>row.measurementKey===key).map(row=><div key={String(row.id)}><small>{key} · {String(row.sourceEntityId)} · {String(row.sourceRevisionId)}</small><pre>{JSON.stringify(row.originalMeasurement,null,2)}</pre></div>)}</details></div>)}
  </details></>;
}
