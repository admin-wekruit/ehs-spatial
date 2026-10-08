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
let language='en',testHooks;
let resolveTestAsset=()=>{throw Error('unexpected request');};
function load(filename) {
  if(cache.has(filename))return cache.get(filename).exports;
  const module={exports:{}};cache.set(filename,module);
  if (filename.endsWith('.json')) { module.exports = JSON.parse(fs.readFileSync(filename, 'utf8')); return module.exports; }
  const code=ts.transpileModule(fs.readFileSync(filename,'utf8'),{fileName:filename,compilerOptions:{esModuleInterop:true,module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX}}).outputText;
  function dependency(name) {
    if(name.endsWith('.css'))return {};
    if(name==='./App')return {ErrorNotice:()=>null};
    if(name==='./WorkcellReport')return {ReportDownload:({assetId,children})=>React.createElement('button',{'data-asset':assetId},children)};
    if(name==='./api')return {resolveAsset:id=>resolveTestAsset(id)};
    if(name==='react')return new Proxy(React,{get:(target,key)=>testHooks?.[key]||target[key]});
    if(name==='./i18n') { const messages = load(path.join(root, 'src/translate.ts')); return {...messages, useI18n:()=>({language,t:(key,params)=>messages.translate(language,key,params)})}; }
    if(!name.startsWith('.'))return require(name);
    const resolved=path.resolve(path.dirname(filename),name);
    return load([resolved,resolved+'.ts',resolved+'.tsx'].find(p=>fs.existsSync(p)));
  }
  new Function('require','module','exports',code)(dependency,module,module.exports);
  return module.exports;
}
const {ReportEvidence,interpretationSelection,interpretationSourcePhoto,orderedInterpretations,partitionInterpretationItems,exactHistoricalPolicy,comparisonSource,sourceCadFor,cadZoomView,cadPanView,cadFocusView,cadLinkedEntities,OriginalCadEvidence}=load(path.join(root,'src/ReportEvidence.tsx'));
const item={label:'Button',note:'Bound button note',entityIds:['button'],imageId:'photo-b',sourceImageId:'original-photo',sourcePixelBox:[20,40,60,100],cameraId:'camera-b',sourceFrameId:'source-3',targetSourceFrameId:'target-1',mappingStatus:'verified',sourceCandidateIds:['candidate']};
const legacy={runId:'old-run',items:[{...item,label:'Unassociated old note',entityIds:[],imageId:null,sourceImageId:null}],missing:[],rejected:[],sourceRefs:[]};
const current={runId:'capture-run',items:[item],missing:[],rejected:[],sourceRefs:[]};
const policy={id:'policy-exact',spec:{policyId:'policy-exact',rationale:'Exact policy rationale',sourceText:'Exact source clause',threshold:0,unit:'m',unsupportedReason:'No temporal evidence'},sourceRefs:[{assetId:'policy-source'}]};
const doc={geometryBindings:{"photo-a":{cameraId:"camera-a",geometrySolutionId:"solution"},"photo-b":{cameraId:"camera-b",geometrySolutionId:"solution"}},entities:[{id:'button',label:'Current parametric button',observationRefs:['observation-a','observation-b']}],assets:[{id:'photo-a'},{id:'photo-b'}],cameras:[{id:'camera-a',imageId:'photo-a'},{id:'camera-b',imageId:'photo-b'}],observations:[{id:'observation-a',imageId:'photo-a',sourceRefs:[]},{id:'observation-b',imageId:'photo-b',sourceRefs:[{sourceRecordId:'candidate'}]}],annotations:[],reportEvidence:{schemaVersion:1,sourceRunId:'old-run',reconstructionRunId:'assembled-run',sourceRefs:[],mappingNotes:[],imageInterpretations:[legacy,current],historical:{runId:'old-run',summary:{detail:'Saved summary'},assessment:{},inventory:[],frames:[{sourceFrameId:'old-frame',imageId:'old-photo'}],policies:[{id:'unrelated',spec:{policyId:'unrelated',rationale:'Wrong rationale'}},policy],findings:[{id:'policy-exact',title:'Saved finding',status:'NEEDS_REVIEW',sourceRefs:[]}]},objects:[{entityId:'button',sourceRecordId:'source-button',metrics:{beforeIou:0.4,afterIou:0.6,beforeDepth:0.03,afterDepth:0.02},metricsMeaning:'Original shape experiment; not current parametric button'}],resources:[{id:'metric-id',kind:'quality',label:'source-button-comparison.json',assetId:'metrics',runId:'experiment-run',sourceRefs:[]}],quality:{metricMeaning:'Input consistency only',limitations:[]}}};
doc.assets.push({id:'original-photo',mediaType:'image/jpeg',metadata:{width:200,height:150}});
const before=JSON.stringify(doc);
assert.deepEqual(interpretationSelection(doc,item,'button'),{entityId:'button',observationId:'observation-b',imageId:'photo-b',cameraId:'camera-b'});
assert.equal(interpretationSelection(doc,item,'unknown'),null);
assert.deepEqual(interpretationSelection(doc,{...item,imageId:'foreign-photo'},'button'),{entityId:'button',observationId:null,imageId:null,cameraId:null});
const ambiguous=structuredClone(doc);ambiguous.entities[0].observationRefs.push('second-b');ambiguous.observations.push({id:'second-b',imageId:'photo-b',sourceRefs:[]});
assert.equal(interpretationSelection(ambiguous,{...item,sourceCandidateIds:[]},'button').observationId,null);
assert.equal(interpretationSelection(ambiguous,item,'button').observationId,'observation-b');
assert.deepEqual(orderedInterpretations(doc,[legacy,current],'old-run').map(a=>a.runId),['capture-run','old-run']);
const sourceOnly={...item,label:'Source only',note:'Unassociated only note',entityIds:[],imageId:null};
const unbound={...sourceOnly,label:'Photo unbound',note:'Unbound photograph note',sourceImageId:null};
const mixedItems=[sourceOnly,unbound,{...item,label:'Other photograph',note:'Other photograph note',imageId:'photo-a'},item];
const grouped=partitionInterpretationItems(doc,mixedItems,'photo-b');
assert.deepEqual(grouped.linked.map(row=>row.item.label),['Button','Other photograph']);
assert.deepEqual(grouped.sourceOnly.map(row=>row.item.label),['Source only']);
assert.deepEqual(grouped.unbound.map(row=>row.item.label),['Photo unbound']);
assert.equal(grouped.linked.length+grouped.sourceOnly.length+grouped.unbound.length,mixedItems.length);
assert.equal(mixedItems[0],sourceOnly,'grouping must not reorder source evidence');
assert.equal(interpretationSelection(doc,sourceOnly,'button'),null,'a source photo never fabricates current-object selection');
assert.deepEqual(interpretationSourcePhoto(doc,sourceOnly),{assetId:'original-photo',width:200,height:150,box:[20,40,60,100],crop:[10,28,60,84]});
assert.equal(interpretationSourcePhoto(doc,{...sourceOnly,sourceImageId:'unknown-photo'}),null,'unknown asset IDs must not resolve through labels or current images');
for(const sourcePixelBox of [[0,0,201,100],[10,0,5,10],[NaN,0,10,10],[-1,0,10,10]])assert.equal(interpretationSourcePhoto(doc,{...sourceOnly,sourcePixelBox}).box,null,'invalid boxes must not crop a different source region');
assert.deepEqual(interpretationSourcePhoto(doc,{...sourceOnly,sourcePixelBox:[0,0,200,150]}).crop,[0,0,200,150],'crop context stays inside the source photograph');
const invalidPhoto=structuredClone(doc);invalidPhoto.assets.at(-1).mediaType='text/html';assert.equal(interpretationSourcePhoto(invalidPhoto,sourceOnly),null);
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
assert.match(mixedHtml,/2 linked to current objects \/ 4 source records/);
assert.ok(mixedHtml.indexOf('Bound button note')<mixedHtml.indexOf('Other photograph note'));
assert.ok(mixedHtml.indexOf('Other photograph note')<mixedHtml.indexOf('Unassociated only note'));
assert.match(mixedHtml,/<details class="report-source-details report-source-only-records"><summary>Source photo available; current-scene match unconfirmed · 1/);
assert.match(mixedHtml,/<details class="report-source-details report-unbound-records"><summary>Source records without a bound photograph · 1/);
assert.equal((mixedHtml.match(/data-source-image-id="original-photo"/g)||[]).length,3,'both linked and source-only records expose their preserved original photograph');
assert.match(understanding,/Historical detection archive · 1 source records/);
assert.match(understanding,/Archive counts do not add to the current object inventory or represent newly missed objects/);
// Exercise the real lazy preview handlers with the same small hook driver as photo-draw-check.
function findElement(node,predicate){
  if(!React.isValidElement(node))return null;
  if(predicate(node))return node;
  for(const child of React.Children.toArray(node.props.children)){const found=findElement(child,predicate);if(found)return found;}
  return null;
}
const preview=findElement(ReportEvidence({document:mixedDoc,section:'understanding',onSelect:()=>assert.fail('source previews must not select current objects')}),node=>node.type.name==='InterpretationPhoto');
const slots=[],effects=[],requestedAssets=[];let cursor=0;
testHooks={
  useState(initial){const index=cursor++;if(!Object.hasOwn(slots,index))slots[index]=initial;return [slots[index],value=>{slots[index]=typeof value==='function'?value(slots[index]):value;}];},
  useEffect(effect,deps){const index=cursor++,previous=slots[index];if(!previous||deps.some((value,i)=>!Object.is(value,previous.deps[i])))effects.push(()=>{previous?.cleanup?.();slots[index]={deps,cleanup:effect()};});},
};
resolveTestAsset=async id=>{requestedAssets.push(id);return 'https://example.invalid/'+id+'.jpg';};
function renderPreview(){cursor=0;const tree=preview.type(preview.props);for(const effect of effects.splice(0))effect();return tree;}
let previewTree=renderPreview();assert.equal(requestedAssets.length,0,'collapsed records do not fetch dozens of original photographs');
previewTree.props.onToggle({currentTarget:{open:true}});renderPreview();await Promise.resolve();previewTree=renderPreview();
assert.deepEqual(requestedAssets,['original-photo']);
assert.equal(findElement(previewTree,node=>node.type==='image').props.href,'https://example.invalid/original-photo.jpg');
assert.equal(findElement(previewTree,node=>node.type==='svg').props.viewBox,'10 28 60 84');
const sourceRect=findElement(previewTree,node=>node.type==='rect').props;
assert.deepEqual([sourceRect.x,sourceRect.y,sourceRect.width,sourceRect.height],[20,40,40,60]);
findElement(previewTree,node=>node.type==='button'&&node.props.children==='Full source photograph').props.onClick();previewTree=renderPreview();
assert.equal(findElement(previewTree,node=>node.type==='svg').props.viewBox,'0 0 200 150','full-image mode preserves the original pixel grid');
findElement(previewTree,node=>node.type==='image').props.onError();previewTree=renderPreview();
assert.ok(findElement(previewTree,node=>node.props.role==='alert'),'image failures remain explicit and retryable');
findElement(previewTree,node=>node.type==='button'&&node.props.children==='Retry').props.onClick();renderPreview();await Promise.resolve();renderPreview();
assert.equal(requestedAssets.length,2,'retry resolves a fresh asset URL');
testHooks=undefined;
assert.match(mixedHtml,/<details class="report-evidence-origin"><summary>Photograph and association evidence/);
assert.match(safety,/Exact policy rationale/);assert.match(safety,/Exact source clause/);assert.match(safety,/No temporal evidence/);assert.match(safety,/Policy threshold<\/dt><dd>0 m/);assert.doesNotMatch(safety,/Wrong rationale/);
assert.match(quality,/experiment-run/);assert.match(quality,/Original shape experiment; not current parametric button/);assert.match(quality,/Assembled scene run.*assembled-run/);assert.match(quality,/0\.4000.*0\.6000/);
assert.match(render('assets'),/data-asset="old-photo"/);
const cadDoc=structuredClone(doc);
const ring=[[0,0],[100,0],[100,100],[0,100]];
cadDoc.assets.push({id:'historical-cad',mediaType:'image/png',metadata:{sourceRunId:'old-run'}});
cadDoc.reportEvidence.historical.cad={assetId:'historical-cad',width:1600,height:1240,regions:[{inventoryIndex:1,entityIds:['button'],polygon:ring},{inventoryIndex:1,entityIds:['button'],polygon:ring},{inventoryIndex:2,entityIds:['missing-entity'],polygon:ring},{inventoryIndex:3,entityIds:[],polygon:ring}]};
const validatedCad=sourceCadFor(cadDoc);
assert.equal(validatedCad.runId,'old-run');
assert.equal(validatedCad.cad.regions.length,4);
assert.equal(sourceCadFor(doc),null);
for(const change of [d=>d.reportEvidence.schemaVersion=2,d=>d.reportEvidence.historical.runId='wrong-source',d=>d.reportEvidence.historical.cad.width=0,d=>d.reportEvidence.historical.cad.regions[0].polygon=[[1,Infinity]],d=>d.reportEvidence.historical.cad.regions[0].entityIds=[17],d=>d.assets.find(a=>a.id==='historical-cad').mediaType='text/html']){
  const invalid=structuredClone(cadDoc);change(invalid);assert.equal(sourceCadFor(invalid),null,'invalid source geometry/identity must not enter the source CAD pane');
}
for(const mediaType of ['image/png','image/jpeg','image/webp','image/svg+xml']){const imageDoc=structuredClone(cadDoc);imageDoc.assets.at(-1).mediaType=mediaType;assert.ok(sourceCadFor(imageDoc));}
assert.deepEqual(cadLinkedEntities(cadDoc,{entityIds:['button','button','foreign']}),['button']);
const multiple=structuredClone(cadDoc);multiple.entities.push({id:'second'});
assert.deepEqual(cadLinkedEntities(multiple,{entityIds:['button','second','foreign']}),['button','second'],'ambiguous source association must not silently choose its first entity');
assert.deepEqual(cadZoomView([0,0,1600,1240],1600,2,[400,300]),[200,150,800,620]);
assert.equal(cadZoomView([0,0,1600,1240],1600,1000,[0,0])[2],100,'zoom has a bounded 16x limit');
assert.equal(cadZoomView([0,0,1600,1240],1600,.001,[0,0])[2],1600,'zooming out preserves full-sheet scale');
assert.deepEqual(cadPanView([200,150,800,620],25,-10),[175,160,800,620]);
const focused=cadFocusView(validatedCad.cad,[ring]);assert.ok(focused[2]<1600&&focused[2]>=1600/3,'automatic focus retains surrounding context and limits raster enlargement to 3x');assert.equal(focused[0]+focused[2]/2,50);assert.equal(focused[1]+focused[3]/2,50);
assert.deepEqual(cadFocusView(validatedCad.cad,[]),[0,0,1600,1240]);
const embedded=renderToStaticMarkup(React.createElement(OriginalCadEvidence,{...validatedCad,document:cadDoc,onSelect:()=>{},embedded:true,selectedId:'foreign'}));
assert.match(embedded,/report-cad-embedded/);assert.match(embedded,/not linked to this source CAD/);assert.doesNotMatch(embedded,/id="workcell-original-cad"/,'embedded and archive views have distinct page identity');
assert.match(embedded,/<button type="button" disabled="">Focus selection/);
const linkedEmbed=renderToStaticMarkup(React.createElement(OriginalCadEvidence,{...validatedCad,document:cadDoc,onSelect:()=>{},embedded:true,selectedId:'button'}));
assert.doesNotMatch(linkedEmbed,/not linked to this source CAD/);assert.match(linkedEmbed,/<button type="button">Focus selection/);
const cadHtml=render('assets',cadDoc);
assert.match(cadHtml,/id="workcell-original-cad"/);
assert.match(cadHtml,/3 original CAD object records · 1 linked to the current scene/,'count source object records once and exclude stale scene associations');
assert.match(cadHtml,/old-run · Image-region coordinates 1600 × 1240 px/);
assert.match(cadHtml,/not the current model plan or a verified site survey/);
assert.match(cadHtml,/data-asset="historical-cad"/,'the original CAD download remains available');
cadDoc.reportEvidence.historical.inventoryAssetId='source-inventory';
cadDoc.reportEvidence.historical.sourceCadManifestAssetId='source-manifest';
cadDoc.reportEvidence.historical.cad.coverage={method:'source_cad_identity_v1',linkedRecordCount:99,records:[
  {inventoryIndex:1,label:'Exact source button',reason:'exact_source_mask',sourceFrameId:'frame-a',proofStatus:'verified_original_source_mask'},
  {inventoryIndex:2,label:'Source rail',reason:'no_verified_same_photo',sourceFrameId:'frame-b'},
  {inventoryIndex:3,label:'Source marker',reason:'canonical_masks_differ',sourceFrameId:'frame-c'}]};
const coverageHtml=render('assets',cadDoc);
assert.match(coverageHtml,/3 original CAD object records · 1 linked to the current scene · 2 source records to reconcile/,'coverage counts current valid bindings instead of trusting stale summary totals');
assert.match(coverageHtml,/Source CAD coverage ledger · 2 source records to reconcile/);
assert.match(coverageHtml,/Source photo has no verified correspondence to a current photo/);
assert.match(coverageHtml,/The current segmentation changed and needs reconciliation/);
assert.match(coverageHtml,/Source coordinates are not registered and do not constrain current model positions or dimensions/);
assert.match(coverageHtml,/data-asset="source-inventory"/);
assert.match(coverageHtml,/data-asset="source-manifest"/);
assert.match(coverageHtml,/Original segmentation evidence verified/);
assert.equal(JSON.stringify(doc),before,'rendering/sorting must not mutate the fixed scene snapshot');
language='zh';assert.match(render('safety'),/\u89c4\u5219\u7406\u7531/);assert.match(render('quality'),/\u539f\u59cb\u5b9e\u9a8c\u5019\u9009\u5bf9\u6bd4/);
if(process.env.PANOPTES_TEST_PUBLICATION_URL){
  const response=await fetch(process.env.PANOPTES_TEST_PUBLICATION_URL);assert.equal(response.status,200);
  const publication=await response.json(),actual=publication.snapshot.revision.document,bundle=actual.reportEvidence;
  const liveCad=sourceCadFor(actual);assert.ok(liveCad);assert.equal(liveCad.cad.assetId,bundle.historical.cad.assetId);assert.equal(liveCad.runId,bundle.historical.runId);
  assert.equal(liveCad.cad.regions.length,bundle.historical.cad.regions.length,'all original source CAD region records are preserved');
  const sorted=orderedInterpretations(actual,bundle.imageInterpretations,bundle.historical.runId);
  assert.ok(sorted[0].items.some(i=>i.entityIds.some(id=>interpretationSelection(actual,i,id)?.imageId)));
  const groups=partitionInterpretationItems(actual,sorted[0].items);
  assert.equal(groups.linked.length+groups.sourceOnly.length+groups.unbound.length,sorted[0].items.length);
  assert.match(render('understanding',actual),new RegExp(`${groups.linked.length} \u6761\u5df2\u5b9a\u4f4d\u5f53\u524d\u5bf9\u8c61 / ${sorted[0].items.length} \u6761\u6765\u6e90\u8bb0\u5f55`));
  for(const analysis of sorted)for(const detection of analysis.items)for(const id of detection.entityIds){
    const selected=interpretationSelection(actual,detection,id);assert.ok(selected);
    if(selected.observationId)assert.equal(actual.observations.find(o=>o.id===selected.observationId).imageId,selected.imageId);
  }
  const actualSafety=render('safety',actual),actualQuality=render('quality',actual);
  for(const finding of bundle.historical.findings){const exact=exactHistoricalPolicy(bundle.historical.policies,finding.id);assert.ok(exact);if(exact.spec.rationale)assert.ok(actualSafety.includes(exact.spec.rationale.replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll("'",'&#x27;').replaceAll('<','&lt;').replaceAll('>','&gt;')));}
  for(const object of bundle.objects.filter(o=>o.metrics&&['beforeIou','afterIou','beforeDepth','afterDepth'].some(key=>typeof o.metrics[key]==='number'))){assert.ok(comparisonSource(bundle,object.sourceRecordId));if(object.metricsMeaning)assert.ok(actualQuality.includes(object.metricsMeaning));}
  console.log(`Read-only snapshot ${publication.id}: ${groups.linked.length} linked / ${sorted[0].items.length} source records; ${groups.sourceOnly.length} source photos available; ${groups.unbound.length} photograph bindings missing; ${bundle.historical.policies.length} exact historical policy specs.`);
}
console.log('Report evidence checks passed: three association states, lazy source crops and full-image switch/retry, unchanged current selection, historical archive, exact policy/metric identity, immutable sorting, en/zh/nl.');

const splitDoc=structuredClone(doc);
splitDoc.entities=[{id:'child-a',label:'Child A',observationRefs:['observation-a']},{id:'child-b',label:'Child B',observationRefs:['observation-b']}];
const splitItem={...item,entityIds:['retired-parent']};
assert.equal(interpretationSelection(splitDoc,splitItem,'child-b').observationId,'observation-b','Stable photo observation ownership survives a source-record split');
assert.equal(interpretationSelection(splitDoc,splitItem,'child-a'),null,'A photo-specific interpretation cannot broadcast to the other split child');
splitDoc.reportEvidence.objects=[{entityId:null,sourceRecordId:'source-button',metrics:{beforeIou:0.2345},views:[{observationId:'observation-a',entityId:'child-a'},{observationId:'observation-b',entityId:'child-b'}]}];
const splitQuality=render('quality',splitDoc);
assert.match(splitQuality,/Child A/);assert.match(splitQuality,/Child B/);
assert.equal((splitQuality.match(/0\.2345/g)||[]).length,1,'Source-record quality metrics remain one historical row, not duplicated as child measurements');

// The same saved evidence renders in each locale, then returns to English.
for (const locale of ['en', 'zh', 'nl', 'en']) {
  language = locale;
  const html = render('safety');
  const {translate} = load(path.join(root, 'src/translate.ts'));
  assert.ok(html.includes(translate(locale, 'reportHistoricalSafety').replaceAll('&', '&amp;')), `localized heading in ${locale}`);
  if (locale !== 'zh') assert.doesNotMatch(html, /[\u3400-\u9fff]/);
}
