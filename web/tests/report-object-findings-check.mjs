// Run from web/: node tests/report-object-findings-check.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import ts from 'typescript';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
const require=createRequire(import.meta.url), root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'), cache=new Map();
let language='en';
function load(filename){
  if(cache.has(filename))return cache.get(filename).exports;
  const module={exports:{}};cache.set(filename,module);
  if (filename.endsWith('.json')) { module.exports = JSON.parse(fs.readFileSync(filename, 'utf8')); return module.exports; }
  const code=ts.transpileModule(fs.readFileSync(filename,'utf8'),{fileName:filename,compilerOptions:{esModuleInterop:true,module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX}}).outputText;
  function dependency(name){
    if(name.endsWith('.css'))return {};
    if(name==='./i18n') { const messages = load(path.join(root, 'src/translate.ts')); return {...messages, useI18n:()=>({language,t:(key,params)=>messages.translate(language,key,params)})}; }
    if(name==='./api')return {request(){throw Error('sidebar must not issue an API/model request');}};
    if(name==='./App')return {ErrorNotice:()=>null};
    if(name==='./WorkcellReport')return {ReportDownload:()=>null};
    if(!name.startsWith('.'))return require(name);
    const resolved=path.resolve(path.dirname(filename),name);
    return load([resolved,resolved+'.ts',resolved+'.tsx'].find(p=>fs.existsSync(p)));
  }
  new Function('require','module','exports',code)(dependency,module,module.exports);return module.exports;
}
const {ReportObjectFindings,historicalObjectFindings}=load(path.join(root,'src/ReportObjectFindings.tsx'));
const revision={id:'current',projectId:'p',document:{entities:[{id:'a',label:'Shared label'},{id:'b',label:'Shared label'}],observations:[],coordinateFrames:[],reportEvidence:{historical:{runId:'old-run',inventory:[{inventoryIndex:7,entityIds:['a']}],findings:[
  {id:'unbound',title:'Unbound aggregate FAIL',status:'FAIL',facts:[{subjectId:'legacy-a'}]},
  {id:'inventory',title:'Explicit inventory evidence',status:'FAIL',facts:[{inventoryIndex:7,value:0.2}]},
  {id:'direct-b',title:'B historical only',status:'FAIL',violations:[{entityId:'b'}]},
  {id:'wrong-run',title:'Foreign run inventory',runId:'foreign',status:'FAIL',facts:[{inventoryIndex:7}]},
]}}}};
const finding=(entityId,title,sceneRevisionId='current')=>({id:title,entityId,policyTitle:title,sceneRevisionId,policyRevisionId:'policy',sourceId:'source',applicability:'applicable',machineResult:'FAIL',reason:title+' exact reasoning',facts:[{value:0.2,unit:'m'}],missingEvidence:['metric_calibration']});
const evaluation=(id,sceneRevisionId,findings,projectId='p')=>({id,projectId,sceneRevisionId,document:{sceneRevisionId,findings}});
const evaluations=[evaluation('current-e','current',[finding('a','A current'),finding('b','B current'),finding('a','Old finding','old')]),evaluation('old-e','old',[finding('a','Old assessment','old')]),evaluation('foreign-project','current',[finding('a','Foreign project')],'other')];
const snapshot={revision,evaluations,reviews:[]};
const publication={id:'publication',projectId:'p',snapshot};
const before=JSON.stringify({revision,publication});
const render=(entityId='a',extra={})=>renderToStaticMarkup(React.createElement(ReportObjectFindings,{revision,publication,entityId,onReview(){},...extra}));
const a=render();
assert.doesNotMatch(render('a',{readOnly:true}),/Add evidence \/ review findings/);
assert.match(render('a',{readOnly:true}),/<button[^>]*>Facts and assessment details/);
assert.match(a,/A current exact reasoning/);assert.doesNotMatch(a,/B current|Old finding|Old assessment|Foreign project|B historical only|Unbound aggregate FAIL|Foreign run inventory/);
assert.match(a,/1 explicitly linked \/ 4 historical checks/);assert.match(a,/not a finding for this object in the current revision/);
assert.match(a,/<details><summary>Facts and assessment details/);assert.match(a,/Evidence-based scale calibration/);
const b=render('b');assert.match(b,/B current exact reasoning/);assert.doesNotMatch(b,/A current|Explicit inventory evidence/);
assert.deepEqual(historicalObjectFindings(revision,'a').linked.map(row=>row.finding.id),['inventory']);
assert.equal(historicalObjectFindings(revision,'missing').linked.length,0);
assert.match(render('a',{publication:{...publication,snapshot:{...snapshot,evaluations:[]}}}),/has not been assessed/,'published empty evaluations cannot inherit live results');
assert.doesNotMatch(render('a',{publication:{...publication,snapshot:{...snapshot,evaluations:[]}},evaluations}),/A current/);
assert.match(render('a',{publication:undefined,evaluations:[]}),/has not been assessed/);
assert.match(render('a',{publication:undefined}),/records for this revision are unavailable/,'unknown loading state is not an unassessed claim');
assert.match(render('a',{revision:{...revision,id:'different'}}),/records for this revision are unavailable/,'mismatched publication revision must not be claimed');
assert.doesNotMatch(render('a',{revision:{...revision,id:'different'}}),/A current/);
const unknown={...publication,snapshot:{...snapshot,evaluations:[evaluation('unknown','current',[{...finding('a','Unknown applicability'),machineResult:'PASS',applicability:'unknown'}])]}};
assert.match(render('a',{publication:unknown}),/rr-applicability_unknown/);assert.doesNotMatch(render('a',{publication:unknown}),/rr-pass/);
assert.equal(JSON.stringify({revision,publication}),before,'selection and filtering leave the saved snapshot unchanged');
language='zh';assert.match(render(),/\u6b64\u5bf9\u8c61\u7684\u5224\u5b9a|\u4ecd\u9700\u54ea\u4e9b\u8bc1\u636e/);
if(process.env.PANOPTES_TEST_PUBLICATION_URL){
  const response=await fetch(process.env.PANOPTES_TEST_PUBLICATION_URL);assert.equal(response.status,200);const actual=await response.json();
  const historical=actual.snapshot.revision.document.reportEvidence.historical;
  const linkedInventory=historical.inventory.find(row=>row.entityIds.length);
  const result=historicalObjectFindings(actual.snapshot.revision,linkedInventory.entityIds[0]);
  assert.equal(result.total,9);assert.equal(result.linked.length,0,'CAD inventory association alone does not identify the historical safety entity IDs');
  console.log(`Real snapshot ${actual.id}: 9 historical rules, no invented object-finding links.`);
}
console.log('Object findings checks passed: A/B and project/revision isolation, fixed snapshot priority, unknown applicability, explicit same-run inventory links, no name matching, no requests, immutable evidence, en/zh/nl.');

const splitRevision=structuredClone(revision);
splitRevision.document.entities=[{id:'a',observationRefs:['oa']},{id:'b',observationRefs:['ob']}];
splitRevision.document.observations=[{id:'oa'},{id:'ob'}];
splitRevision.document.identityDecisions=[{id:'split',decision:'different',entityIds:['retired-parent'],observationGroups:[['oa'],['ob']]}];
splitRevision.document.reportEvidence.historical.inventory=[{inventoryIndex:7,entityIds:['a','b']}];
splitRevision.document.reportEvidence.historical.findings=[{id:'aggregate',facts:[{inventoryIndex:7},{entityId:'retired-parent'}]}];
for(const child of ['a','b']) assert.equal(historicalObjectFindings(splitRevision,child).linked.length,0,'An unscoped historical source conclusion cannot be attached to both split children');

// The same saved evidence renders in each locale, then returns to English.
for (const locale of ['en', 'zh', 'nl', 'en']) {
  language = locale;
  const html = render();
  const {translate} = load(path.join(root, 'src/translate.ts'));
  assert.ok(html.includes(translate(locale, 'objectFindings.title')), `localized heading in ${locale}`);
  if (locale !== 'zh') assert.doesNotMatch(html, /[\u3400-\u9fff]/);
}
