// Run: node tests/report-evidence-check.mjs
// Optional read-only real snapshot check: PANOPTES_TEST_PUBLICATION_URL=http://.../api/publications/... node tests/report-evidence-check.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import ts from 'typescript';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
const require=createRequire(import.meta.url),root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),cache=new Map();
let language='en';
function load(filename) {
  if(cache.has(filename))return cache.get(filename).exports;
  const module={exports:{}};cache.set(filename,module);
  const code=ts.transpileModule(fs.readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX}}).outputText;
  function dependency(name) {
    if(name.endsWith('.css'))return {};
    if(name==='./App')return {ErrorNotice:()=>null};
    if(name==='./WorkcellReport')return {ReportDownload:({assetId,children})=>React.createElement('button',{'data-asset':assetId},children)};
    if(name==='./api')return {resolveAsset:()=>{throw Error('unexpected request');}};
    if(name==='./i18n')return {useI18n:()=>({language,t:key=>key})};
    if(!name.startsWith('.'))return require(name);
    const resolved=path.resolve(path.dirname(filename),name);
    return load([resolved,resolved+'.ts',resolved+'.tsx'].find(p=>fs.existsSync(p)));
  }
  new Function('require','module','exports',code)(dependency,module,module.exports);
  return module.exports;
}
const {ReportEvidence,interpretationSelection,orderedInterpretations,partitionInterpretationItems,exactHistoricalPolicy,comparisonSource}=load(path.join(root,'src/ReportEvidence.tsx'));
const item={label:'Button',labelZh:'按钮',note:'Bound button note',entityIds:['button'],imageId:'photo-b',cameraId:'camera-b',sourceFrameId:'source-3',targetSourceFrameId:'target-1',mappingStatus:'verified',sourceCandidateIds:['candidate']};
const legacy={runId:'old-run',items:[{...item,label:'Unassociated old note',entityIds:[],imageId:null}],missing:[],rejected:[],sourceRefs:[]};
const current={runId:'capture-run',items:[item],missing:[],rejected:[],sourceRefs:[]};
const policy={id:'policy-exact',spec:{policyId:'policy-exact',rationale:'Exact policy rationale',sourceText:'Exact source clause',threshold:0,unit:'m',unsupportedReason:'No temporal evidence'},sourceRefs:[{assetId:'policy-source'}]};
const doc={entities:[{id:'button',label:'Current parametric button',observationRefs:['observation-a','observation-b']}],assets:[{id:'photo-a'},{id:'photo-b'}],cameras:[{id:'camera-a',imageId:'photo-a'},{id:'camera-b',imageId:'photo-b'}],observations:[{id:'observation-a',imageId:'photo-a',sourceRefs:[]},{id:'observation-b',imageId:'photo-b',sourceRefs:[{sourceRecordId:'candidate'}]}],annotations:[],reportEvidence:{schemaVersion:1,sourceRunId:'old-run',reconstructionRunId:'assembled-run',sourceRefs:[],mappingNotes:[],imageInterpretations:[legacy,current],historical:{runId:'old-run',summary:{detail:'Saved summary'},assessment:{},inventory:[],frames:[{sourceFrameId:'old-frame',imageId:'old-photo'}],policies:[{id:'unrelated',spec:{policyId:'unrelated',rationale:'Wrong rationale'}},policy],findings:[{id:'policy-exact',title:'Saved finding',status:'NEEDS_REVIEW',sourceRefs:[]}]},objects:[{entityId:'button',sourceRecordId:'source-button',metrics:{beforeIou:0.4,afterIou:0.6,beforeDepth:0.03,afterDepth:0.02},metricsMeaning:'Original shape experiment; not current parametric button'}],resources:[{id:'metric-id',kind:'quality',label:'source-button-comparison.json',assetId:'metrics',runId:'experiment-run',sourceRefs:[]}],quality:{metricMeaning:'Input consistency only',limitations:[]}}};
const before=JSON.stringify(doc);
assert.deepEqual(interpretationSelection(doc,item,'button'),{entityId:'button',observationId:'observation-b',imageId:'photo-b',cameraId:'camera-b'});
assert.equal(interpretationSelection(doc,item,'unknown'),null);
assert.deepEqual(interpretationSelection(doc,{...item,imageId:'foreign-photo'},'button'),{entityId:'button',observationId:null,imageId:null,cameraId:null});
const ambiguous=structuredClone(doc);ambiguous.entities[0].observationRefs.push('second-b');ambiguous.observations.push({id:'second-b',imageId:'photo-b',sourceRefs:[]});
assert.equal(interpretationSelection(ambiguous,{...item,sourceCandidateIds:[]},'button').observationId,null);
assert.equal(interpretationSelection(ambiguous,item,'button').observationId,'observation-b');
assert.deepEqual(orderedInterpretations(doc,[legacy,current],'old-run').map(a=>a.runId),['capture-run','old-run']);
const sourceOnly={...item,label:'Source only',note:'Unassociated only note',entityIds:[],imageId:null};
const mixedItems=[sourceOnly,{...item,label:'Other photograph',note:'Other photograph note',imageId:'photo-a'},item];
const grouped=partitionInterpretationItems(doc,mixedItems,'photo-b');
assert.deepEqual(grouped.linked.map(row=>row.item.label),['Button','Other photograph']);
assert.deepEqual(grouped.unassociated.map(row=>row.item.label),['Source only']);
assert.equal(grouped.linked.length+grouped.unassociated.length,mixedItems.length);
assert.equal(mixedItems[0],sourceOnly,'grouping must not reorder source evidence');
assert.equal(exactHistoricalPolicy([policy],'policy-exact'),policy);
assert.equal(exactHistoricalPolicy([{...policy,spec:{...policy.spec,policyId:'wrong'}}],'policy-exact'),undefined);
assert.equal(exactHistoricalPolicy([policy,policy],'policy-exact'),undefined);
assert.equal(comparisonSource(doc.reportEvidence,'source-button').runId,'experiment-run');
assert.equal(comparisonSource(doc.reportEvidence,'different-candidate'),undefined);
function render(section,document=doc){return renderToStaticMarkup(React.createElement(ReportEvidence,{document,section,onSelect:()=>{}}));}
const understanding=render('understanding'),safety=render('safety'),quality=render('quality');
assert.doesNotMatch(understanding,/<details[^>]* open=""/,'long source interpretation lists start collapsed');
assert.match(safety,/<details class="report-rule-basis"><summary>/,'complete original rules remain available without dominating the initial reading');
assert.match(quality,/<details class="report-historical report-quality-details"><summary>/,'technical experiment metrics start collapsed');
assert.match(render('assets'),/<details class="report-source-details"><summary>Original models, analysis &amp; evidence files/);
assert.ok(understanding.indexOf('Bound button note')<understanding.indexOf('Unassociated old note'));
assert.match(understanding,/source-3.*target-1/);
const mixedDoc=structuredClone(doc);mixedDoc.reportEvidence.imageInterpretations=[{...current,items:mixedItems}];
const mixedHtml=renderToStaticMarkup(React.createElement(ReportEvidence,{document:mixedDoc,section:'understanding',currentImageId:'photo-b',onSelect:()=>{}}));
assert.match(mixedHtml,/2 linked \/ 3 source records/);
assert.ok(mixedHtml.indexOf('Bound button note')<mixedHtml.indexOf('Other photograph note'));
assert.ok(mixedHtml.indexOf('Other photograph note')<mixedHtml.indexOf('Unassociated only note'));
assert.match(mixedHtml,/<details class="report-source-details report-unassociated-records"><summary>/,'unassociated records remain present in a closed details element');
assert.match(mixedHtml,/<details class="report-evidence-origin"><summary>Photograph and association evidence/);
assert.match(safety,/Exact policy rationale/);assert.match(safety,/Exact source clause/);assert.match(safety,/No temporal evidence/);assert.match(safety,/Policy threshold<\/dt><dd>0 m/);assert.doesNotMatch(safety,/Wrong rationale/);
assert.match(quality,/experiment-run/);assert.match(quality,/Original shape experiment; not current parametric button/);assert.match(quality,/Assembled scene run.*assembled-run/);assert.match(quality,/0\.4000.*0\.6000/);
assert.match(render('assets'),/data-asset="old-photo"/);
const cadDoc=structuredClone(doc);
cadDoc.reportEvidence.historical.cad={assetId:'historical-cad',width:1600,height:1240,regions:[{inventoryIndex:1,entityIds:['button'],polygon:[]},{inventoryIndex:1,entityIds:['button'],polygon:[]},{inventoryIndex:2,entityIds:['missing-entity'],polygon:[]},{inventoryIndex:3,entityIds:[],polygon:[]}]};
const cadHtml=render('assets',cadDoc);
assert.match(cadHtml,/id="workcell-original-cad"/);
assert.match(cadHtml,/3 original CAD object records · 1 linked to the current scene/,'count source object records once and exclude stale scene associations');
assert.match(cadHtml,/old-run · Image-region coordinates 1600 × 1240 px/);
assert.match(cadHtml,/not the current model plan or a verified site survey/);
assert.match(cadHtml,/data-asset="historical-cad"/,'the original CAD download remains available');
assert.equal(JSON.stringify(doc),before,'rendering/sorting must not mutate the fixed scene snapshot');
language='zh';assert.match(render('safety'),/规则理由/);assert.match(render('quality'),/原始实验候选对比/);
if(process.env.PANOPTES_TEST_PUBLICATION_URL){
  const response=await fetch(process.env.PANOPTES_TEST_PUBLICATION_URL);assert.equal(response.status,200);
  const publication=await response.json(),actual=publication.snapshot.revision.document,bundle=actual.reportEvidence;
  const sorted=orderedInterpretations(actual,bundle.imageInterpretations,bundle.historical.runId);
  assert.ok(sorted[0].items.some(i=>i.entityIds.some(id=>interpretationSelection(actual,i,id)?.imageId)));
  const groups=partitionInterpretationItems(actual,sorted[0].items);
  assert.equal(groups.linked.length+groups.unassociated.length,sorted[0].items.length);
  assert.match(render('understanding',actual),new RegExp(`${groups.linked.length} 条已关联 / ${sorted[0].items.length} 条来源记录`));
  for(const analysis of sorted)for(const detection of analysis.items)for(const id of detection.entityIds){
    const selected=interpretationSelection(actual,detection,id);assert.ok(selected);
    if(selected.observationId)assert.equal(actual.observations.find(o=>o.id===selected.observationId).imageId,selected.imageId);
  }
  const actualSafety=render('safety',actual),actualQuality=render('quality',actual);
  for(const finding of bundle.historical.findings){const exact=exactHistoricalPolicy(bundle.historical.policies,finding.id);assert.ok(exact);if(exact.spec.rationale)assert.ok(actualSafety.includes(exact.spec.rationale.replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll("'",'&#x27;').replaceAll('<','&lt;').replaceAll('>','&gt;')));}
  for(const object of bundle.objects.filter(o=>o.metrics&&['beforeIou','afterIou','beforeDepth','afterDepth'].some(key=>typeof o.metrics[key]==='number'))){assert.ok(comparisonSource(bundle,object.sourceRecordId));if(object.metricsMeaning)assert.ok(actualQuality.includes(object.metricsMeaning));}
  console.log(`Read-only snapshot ${publication.id}: ${groups.linked.length} linked / ${sorted[0].items.length} source records; ${groups.unassociated.length} retained in closed details; ${bundle.historical.policies.length} exact historical policy specs.`);
}
console.log('Report evidence checks passed: linked selection, ambiguous observations, exact policy identity, original metric source, historical photos, immutable sorting, zh/en.');
