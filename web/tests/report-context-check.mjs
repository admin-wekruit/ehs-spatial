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
  if(name==='./api')return {owner:async()=> 'local-capability',request:async(url,options)=>{
    assert.equal(options?.method,undefined,'selection must not send mutations/model calls');requests.push(url);
    return url==='/api/projects/project'?detail:url==='/api/revisions/revision'?revision:{items:url.endsWith('/edits')?edits:[]};
  }};
  return require(name);
},module,module.exports);
const {WorkcellReport}=module.exports;
async function render(){for(let i=0;i<8;i++){cursor=0;tree=WorkcellReport({projectId:'project',requestedRevision:'revision'});for(const effect of effects.splice(0))effect();await Promise.resolve();}}
function find(predicate,node=tree){if(!React.isValidElement(node))return null;if(predicate(node))return node;for(const child of React.Children.toArray(node.props.children)){const hit=find(predicate,child);if(hit)return hit;}return null;}
const params=()=>new URLSearchParams(hash.split('?')[1]);
await render();
const workbench=find(n=>n.type==='a'&&n.props.className==='button primary').props.href;
assert.equal(workbench.split('?')[0],'#/projects/project/workbench');
assert.deepEqual(Object.fromEntries(new URLSearchParams(workbench.split('?')[1])),{revision:'revision',object:'a',observation:'observation-a',image:'image-a'},'the model workbench keeps the exact revision and photo/object context');
assert.ok(find(n=>n.type==='a'&&n.props.href==='#/policies'),'rule sources remain accessible within the report');
find(n=>typeof n.props.onSummary==='function').props.onSummary({revisionId:'revision',state:'unassessed',evaluationCount:0,attentionCount:0});await render();
assert.ok(find(n=>n.type==='h2'&&n.props.children==='reportAssessment_unassessed'),'saved assessment status leads the report');
const facts=find(n=>typeof n.type==='function'&&n.type.name==='ObjectFacts');
const floorFacts=facts.type({...facts.props,entity:{...entities[0],geometryRole:'floor'}});
assert.equal(find(n=>n.type==='dt'&&n.props.children==='reportOrientation',floorFacts),null,'reference surfaces do not show equipment tilt');
assert.equal(find(n=>n.type==='dt'&&React.Children.toArray(n.props.children).includes('reportCurrentModel'),floorFacts),null,'reference surfaces do not show equipment model height');
assert.ok(find(n=>n.type===AgentPanel), String(find(n=>n.type===empty)?.props.error));
assert.deepEqual(find(n=>n.type===AgentPanel).props.box,[1,2,30,40],'copy/deep-link box survives first load');
assert.ok(find(n=>n.type==='strong'&&n.props.children==='setLabel'),'the edit producing this exact revision is shown');
assert.equal(find(n=>n.type==='strong'&&n.props.children==='setVisibility'),null,'revision A cannot show a future A→B edit');
const initialRequests=requests.slice();
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
