// Run: node tests/report-context-check.mjs
// Drive the actual report handlers/effects; URL replacement must not navigate or refetch.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import ts from 'typescript';
import React from 'react';

const require=createRequire(import.meta.url), root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const slots=[], effects=[];let cursor=0,tree;
const hooks={
  useState(initial){const i=cursor++;if(!Object.hasOwn(slots,i))slots[i]=typeof initial==='function'?initial():initial;return [slots[i],v=>{slots[i]=typeof v==='function'?v(slots[i]):v;}];},
  useRef(initial){return slots[cursor++]??={current:initial};},
  useEffect(effect,deps){const i=cursor++,previous=slots[i];if(!previous||deps.some((v,j)=>!Object.is(v,previous.deps[j])))effects.push(()=>{previous?.cleanup?.();slots[i]={deps,cleanup:effect()};});},
};
const ReportScene=()=>null,AgentPanel=()=>null,empty=()=>null;
const entities=['a','b'].map(id=>({id,label:id,representations:[],measurements:{},observationRefs:['observation-'+id]}));
const doc={entities,observations:entities.map(e=>({id:e.observationRefs[0],imageId:'image-'+e.id})),
  cameras:entities.map(e=>({id:'camera-'+e.id,imageId:'image-'+e.id,width:100,height:100})),
  assets:entities.map(e=>({id:'image-'+e.id,kind:'source_image',width:100,height:100})),coordinateFrames:[],annotations:[]};
entities[1].observationRefs.push('b-in-a-1','b-in-a-2');
doc.observations.push({id:'b-in-a-1',imageId:'image-a'},{id:'b-in-a-2',imageId:'image-a'});
const revision={id:'revision',projectId:'project',branchId:'branch',createdAt:'2026-09-12',document:doc};
const branch={id:'branch',headRevisionId:'revision',kind:'reconstruction'};
const detail={project:{id:'project',title:'Report'},revision,branch,branches:[branch]};
const edits=[
  {id:'current-edit',baseRevisionId:'previous',revisionId:'revision',operations:[{type:'setLabel'}],inverseOperations:[],createdAt:'2026-09-12'},
  {id:'future-edit',baseRevisionId:'revision',revisionId:'future',operations:[{type:'setVisibility'}],inverseOperations:[],createdAt:'2026-09-13'},
];
const requests=[],navigations=[];
let publicPublicationId='';
const publication={id:'publication',projectId:'project',title:'Frozen report',createdAt:revision.createdAt,sceneRevisionId:revision.id,
  snapshot:{revision,editBatches:[edits[0]],evaluations:[],reviews:[],jobs:[{id:'export',kind:'export_blender',status:'succeeded',baseRevisionId:revision.id,createdAt:revision.createdAt,result:{assets:[{id:'blend',name:'workcell.blend'}]}}]}};
let hash='#/projects/project/report?revision=revision&object=a&observation=observation-a&image=image-a&box=1,2,30,40&review=1&agent=1';
globalThis.location={get hash(){return hash;},set hash(value){navigations.push(value);hash=value;}};
globalThis.window={history:{replaceState(_state,_title,url){assert.ok(url.startsWith('#/'));hash=url;}},scrollTo(){}};
const code=ts.transpileModule(fs.readFileSync(path.join(root,'src/WorkcellReport.tsx'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX}}).outputText;
const core=await import('../src/core.ts');
const semantics=await import('../src/scene-semantics.ts');
const module={exports:{}};
new Function('require','module','exports',code)(name=>{
  if(name==='react')return hooks;
  if(name.endsWith('.css'))return {};
  if(name==='./i18n')return {useI18n:()=>({t:key=>key})};
  if(name==='./core')return core;
  if(name==='./scene-semantics')return semantics;
  if(name==='./App')return {ErrorNotice:empty};
  if(name==='./ReportScene')return {ReportScene};
  if(name==='./AgentPanel')return {AgentPanel};
  if(name==='./ReportEvidence')return {ReportEvidence:empty};
  if(name==='./ReportReview')return {ReportReview:empty};
  if(name==='./ReportObjectFindings')return {ReportObjectFindings:empty};
  if(name==='./api')return {get PUBLICATION_ID(){return publicPublicationId;},owner:async()=> {assert.equal(publicPublicationId,'','public viewing must not depend on local management keys');return 'local-capability';},request:async(url,options)=>{
    assert.equal(options?.method,undefined,'selection must not send mutations/model calls');requests.push(url);
    return url==='/api/publications/publication'?publication:url==='/api/projects/project'?detail:url==='/api/revisions/revision'?revision:{items:url.endsWith('/edits')?edits:publicPublicationId?[publication]:[]};
  }};
  return require(name);
},module,module.exports);
const {WorkcellReport}=module.exports;
async function render(){for(let i=0;i<8;i++){cursor=0;tree=WorkcellReport(publicPublicationId?{publicationId:publicPublicationId}:{projectId:'project',requestedRevision:'revision'});for(const effect of effects.splice(0))effect();await Promise.resolve();}}
function find(predicate,node=tree){if(!React.isValidElement(node))return null;if(predicate(node))return node;for(const child of React.Children.toArray([node.props.children,node.props.inspector])){const hit=find(predicate,child);if(hit)return hit;}return null;}
const params=()=>new URLSearchParams(hash.split('?')[1]);
await render();
const workbench=find(n=>n.type==='a'&&n.props.className==='button primary').props.href;
assert.equal(workbench.split('?')[0],'#/projects/project/workbench');
assert.deepEqual(Object.fromEntries(new URLSearchParams(workbench.split('?')[1])),{revision:'revision',object:'a',observation:'observation-a',image:'image-a'},'the model workbench keeps the exact revision and photo/object context');
assert.ok(find(n=>n.type==='a'&&n.props.href==='#/policies'),'rule sources remain accessible within the report');
find(n=>typeof n.props.onSummary==='function').props.onSummary({revisionId:'revision',state:'unassessed',evaluationCount:0,attentionCount:0});await render();
assert.ok(find(n=>n.type==='h2'&&n.props.children==='reportAssessment_unassessed'),'saved assessment status leads the report');
assert.ok(find(n=>n.type===AgentPanel),'an explicit agent context opens the feedback panel');
find(n=>n.type==='button'&&n.props.children==='sceneBackDetails').props.onClick();await render();
const facts=find(n=>typeof n.type==='function'&&n.type.name==='ObjectFacts');
const floorFacts=facts.type({...facts.props,entity:{...entities[0],geometryRole:'floor'}});
assert.equal(find(n=>n.type==='dt'&&n.props.children==='reportOrientation',floorFacts),null,'reference surfaces do not show equipment tilt');
assert.equal(find(n=>n.type==='dt'&&React.Children.toArray(n.props.children).includes('reportCurrentModel'),floorFacts),null,'reference surfaces do not show equipment model height');
find(n=>n.type===ReportScene).props.onFeedback('a');await render();
assert.ok(find(n=>n.type===AgentPanel), String(find(n=>n.type===empty)?.props.error));
assert.deepEqual(find(n=>n.type===AgentPanel).props.box,[1,2,30,40],'copy/deep-link box survives first load');
assert.ok(find(n=>n.type==='strong'&&n.props.children==='setLabel'),'the edit producing this exact revision is shown');
assert.equal(find(n=>n.type==='strong'&&n.props.children==='setVisibility'),null,'revision A cannot show a future A→B edit');
const initialRequests=requests.slice();
globalThis.document={getElementById(){return {scrollIntoView(){}};}};
const inventoryRequest=find(n=>n.type===ReportScene).props.objectListRequest;
find(n=>n.type==='button'&&n.props.className==='report-open-objects').props.onClick();await render();
assert.equal(find(n=>n.type===ReportScene).props.objectListRequest,inventoryRequest+1,'the understanding section opens the single workspace inventory');
assert.equal(find(n=>n.type===ReportScene).props.selection.entityId,'a','inventory navigation keeps object context');
assert.equal(find(n=>n.type==='table'&&n.props.className==='report-inventory'),null,'no duplicate object table below the scene');
find(n=>n.type===ReportScene).props.onSelect('b','observation-b');await render();
assert.equal(params().get('object'),'b');assert.equal(params().get('image'),'image-b');
assert.equal(params().get('observation'),'observation-b');assert.equal(params().has('box'),false);
find(n=>n.type===ReportScene).props.onCamera('image-a','camera-a');await render();
assert.equal(params().get('image'),'image-a');assert.equal(params().has('observation'),false);
find(n=>n.type===ReportScene).props.onBox([4,5,40,50]);await render();
assert.equal(params().get('box'),'4,5,40,50');assert.equal(params().get('review'),'1');assert.equal(params().get('agent'),'1');
find(n=>n.type==='button'&&React.Children.toArray(n.props.children).includes('readReport')).props.onClick();await render();
assert.equal(params().has('review'),false);assert.equal(params().has('agent'),false);
assert.equal(params().get('revision'),'revision','context changes preserve fixed revision');
assert.deepEqual(navigations,[],'replaceState must not trigger a route change/remount');
assert.deepEqual(requests,initialRequests,'context changes must not reload report data or start jobs');
// Reopen the exact final URL with fresh component state and verify restored context.
for(const slot of slots)slot?.cleanup?.();slots.length=0;effects.length=0;
await render();
const reopened=find(n=>n.type===ReportScene).props;
assert.equal(reopened.selection.entityId,'b');assert.equal(reopened.selection.observationId,null);assert.equal(reopened.imageId,'image-a');
console.log('Report context passed: copy box, entity/camera selection, cleared observation/box, review/agent flags, fixed revision, refresh, no navigation/model requests, no future edits.');

// The deployment flag keeps the same report interactions while removing unsupported writes.
for(const slot of slots)slot?.cleanup?.();slots.length=0;effects.length=0;
publicPublicationId='publication';requests.length=0;
hash='#/reports/publication?object=a&image=image-a&review=1&agent=1&box=1,2,30,40';
globalThis.document={getElementById(){return {scrollIntoView(){}};}};
await render();
assert.equal(find(n=>n.type===AgentPanel),null);
assert.equal(find(n=>n.type==='a'&&/projects|policies|workbench/.test(n.props.href)),null);
for(const label of ['copy','reportReviewAction','reportMissingObject','reportDispute','reviewReport','reportStartAssessment','reportExportBlender','reportPublish'])
  assert.equal(find(n=>n.type==='button'&&React.Children.toArray(n.props.children).includes(label)),null,label+' is unavailable in public review');
assert.equal(params().has('review')||params().has('agent')||params().has('box'),false);
assert.equal(find(n=>n.type===ReportScene).props.onBox,undefined);
assert.ok(find(n=>n.type==='span'&&n.props.children==='reportReadOnly'));
assert.ok(find(n=>n.props.assetId==='blend'),'frozen Blender export remains downloadable');
assert.ok(find(n=>n.props.assetId==='image-a'),'source photo remains downloadable');
find(n=>n.type===ReportScene).props.onFeedback('a');await render();
const feedback=find(n=>n.type===AgentPanel);
assert.equal(feedback.props.feedbackPublicationId,'publication','public feedback is scoped to the immutable report');
assert.equal(feedback.props.entityId,'a');
assert.equal(feedback.props.imageId,'image-a');
assert.equal(feedback.props.canWrite,false,'feedback never grants project edit permission');
assert.equal(find(n=>typeof n.type==='function'&&n.type.name==='ObjectFacts'),null,'the feedback panel replaces details instead of appearing below them');
find(n=>n.type===ReportScene).props.onClearSelection();await render();
assert.equal(find(n=>n.type===ReportScene).props.selection.entityId,null);
assert.equal(params().has('object')||params().has('observation'),false,'clear selection removes saved object context');
assert.equal(find(n=>n.type===AgentPanel),null);
assert.ok(find(n=>n.type==='a'&&n.props.href==='#/reports/publication'),'published history stays on the site');
assert.ok(find(n=>n.props.publication===publication&&typeof n.props.onSummary==='function'),'saved EHS data remains rendered');
find(n=>n.type===ReportScene).props.onSelect('a','observation-a');await render();
const objectFindings=find(n=>n.props.entityId==='a'&&typeof n.props.onReview==='function');
assert.equal(objectFindings.props.readOnly,true);objectFindings.props.onReview();await render();
assert.equal(params().has('review'),false,'viewing EHS details must not enable review editing');
const publicRequests=requests.slice();
find(n=>n.type===ReportScene).props.onSelect('b','observation-b');await render();
assert.equal(params().get('object'),'b');assert.equal(params().get('image'),'image-b');
assert.deepEqual(requests,publicRequests,'public object selection does not fetch or mutate data');
assert.deepEqual(requests,['/api/publications/publication','/api/projects/project','/api/publications']);

// Execute the actual route hook so direct write URLs cannot render an editing page.
for(const slot of slots)slot?.cleanup?.();slots.length=0;effects.length=0;
const appSource=fs.readFileSync(path.join(root,'src/App.tsx'),'utf8');
const parsed=ts.createSourceFile('App.tsx',appSource,ts.ScriptTarget.ES2022,true,ts.ScriptKind.TSX);
const routeSource=parsed.statements.find(node=>ts.isFunctionDeclaration(node)&&node.name?.text==='useRoute').getText(parsed);
const routeCode=ts.transpileModule(routeSource,{compilerOptions:{module:ts.ModuleKind.None,target:ts.ScriptTarget.ES2022}}).outputText;
window.addEventListener=()=>{};window.removeEventListener=()=>{};
const useRoute=new Function('PUBLICATION_ID','useState','useEffect','path',routeCode+'; return useRoute;')('publication',hooks.useState,hooks.useEffect,value=>'#'+value);
for(const [input,expected] of [['','/reports/publication'],['#/reports','/reports'],['#/reports/older?object=a','/reports/older'],['#/projects/new','/reports/publication'],['#/projects/project/workbench','/reports/publication'],['#/policies','/reports/publication']]){
  for(const slot of slots)slot?.cleanup?.();slots.length=0;effects.length=0;cursor=0;hash=input;
  const route=useRoute();for(const effect of effects.splice(0))effect();
  assert.equal(route.pathname,expected);assert.ok(hash.startsWith('#/reports'));
}
console.log('Public report passed: default and restricted routes, view-only controls, frozen EHS/history/Blender/source downloads, restored object/photo selection, no management keys or mutations.');
