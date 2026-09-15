import { activeModel, isPartitionSource, jsonObject } from './core';
import { useI18n } from './i18n';
import { measurementLabelKeys } from './identity-messages';
import type { Entity, Operation } from './types';

export function ModelEvidence({entity,onCommit,disabled=false}:{entity:Entity;onCommit?:(operations:Operation[])=>unknown;disabled?:boolean}) {
  const {t}=useI18n(), models=(entity.representations||[]).filter(rep=>['generated_mesh','primitive'].includes(rep.kind)), active=activeModel(entity), selections=jsonObject(entity.measurementSelections)||{};
  const records=(Array.isArray(entity.measurementEvidence)?entity.measurementEvidence:[]).map(jsonObject).filter((row):row is Record<string,unknown>=>!!row);
  const keys=[...new Set(records.map(row=>String(row.measurementKey)))];
  return <details className="report-source-details"><summary>{t('identityMeasurementSources')}</summary>
    {active?.sourceValidity==='stale'&&<p className="evidence-note">{t('identityModelStaleNote')}</p>}
    {!!models.length&&<><label className="field-label">{t('identityCurrentModel')}<select disabled={!onCommit||disabled} value={active?.id||''} onChange={event=>onCommit?.([{type:'setActiveModelRepresentation',entityId:entity.id,representationId:event.target.value||null}])}>
      <option value="">{t('identityNoModel')}</option>{models.filter(rep=>!isPartitionSource(entity,rep.id)).map(rep=><option key={rep.id} value={rep.id}>{t(rep.kind)} · {rep.id.slice(0,8)}{rep.sourceValidity==='stale'?' · '+t('identityModelStale'):''}</option>)}</select></label>
      <details><summary>{t('identityModelAlternatives')} · {models.length}</summary>{models.map(rep=><div key={rep.id}><strong>{rep.id.slice(0,8)} · {rep.id===active?.id?t('identityCurrentModel'):t('identitySourceModel')}</strong><p>{rep.coordinateFrameId} · {rep.transform.position.map(value=>value.toFixed(3)).join(' / ')}</p><small>{rep.sourceValidity==='stale'&&<>{t('identitySourceStale')} · </>}{t(rep.placementState==='confirmed'?'identityPlacementConfirmed':rep.placementState==='unconfirmed'?'identityPlacementUnconfirmed':'identityPlacementUnknown')} · {rep.assetId||t('primitive')}</small></div>)}</details></>}
    {keys.map(key=><div key={key}><label className="field-label">{t(measurementLabelKeys[key] || 'identityMeasurementOther')}<select disabled={!onCommit||disabled} value={typeof selections[key]==='string'?String(selections[key]):''} onChange={event=>onCommit?.([{type:'selectMeasurementEvidence',entityId:entity.id,measurementKey:key,measurementEvidenceId:event.target.value||null}])}>
      <option value="">{t('identityNoMeasurement')}</option>{records.filter(row=>row.measurementKey===key).map(row=><option key={String(row.id)} value={String(row.id)}>{String(row.sourceEntityId).slice(0,8)} · {String(row.sourceRevisionId).slice(0,8)}</option>)}</select></label>
      {typeof selections[key]!=='string'&&<p className="subtle">{t('identityMeasurementConflict')}</p>}<details><summary>{t('identitySourceMeasurement')}</summary>{records.filter(row=>row.measurementKey===key).map(row=><div key={String(row.id)}><small>{key} · {String(row.sourceEntityId)} · {String(row.sourceRevisionId)}</small><pre>{JSON.stringify(row.originalMeasurement,null,2)}</pre></div>)}</details></div>)}
  </details>;
}
