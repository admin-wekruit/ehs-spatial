// One runnable check for frozen report evidence and the component's read-only output.
// Run: node tests/report-review-check.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import ts from 'typescript';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
const require = createRequire(import.meta.url), root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
let language = 'en';
const cache = new Map();
function load(filename) {
  if (cache.has(filename)) return cache.get(filename).exports;
  const module = {exports:{}}; cache.set(filename, module);
  const code = ts.transpileModule(fs.readFileSync(filename,'utf8'), {compilerOptions:{module:ts.ModuleKind.CommonJS, target:ts.ScriptTarget.ES2022, jsx:ts.JsxEmit.ReactJSX}}).outputText;
  function dependency(name) {
    if (name.endsWith('.css')) return {};
    if (name === './App') return {ErrorNotice:()=>null};
    if (name === './api') return {request:()=>{throw Error('unexpected server request during rendering');},id:()=> 'request-id'};
    if (name === './i18n') return {useI18n:()=>({language,t:key=>key})};
    if (!name.startsWith('.')) return require(name);
    const resolved=path.resolve(path.dirname(filename),name);
    return load([resolved,resolved+'.ts',resolved+'.tsx'].find(p=>fs.existsSync(p)));
  }
  new Function('require','module','exports',code)(dependency,module,module.exports);
  return module.exports;
}
const {ReportReview,exactReviewEvidence,assessmentSummary}=load(path.join(root,'src/ReportReview.tsx'));
const finding=(id,revision='old')=>({id,sceneRevisionId:revision,entityId:'bollard',policyRevisionId:'policy-old',sourceId:'source-old',policyTitle:'Saved policy title',applicability:'applicable',machineResult:'FAIL',facts:[{source:'observed_measurement',value:0.4,unit:'m'}],missingEvidence:['metric_footprint:bollard']});
const evaluation=(id,revision,findings)=>({id,projectId:'p',sceneRevisionId:revision,createdAt:'2026-09-12T12:00:00Z',context:'observed',document:{sceneRevisionId:revision,findings}});
const review=(id,evaluationId,findingId,revision='old')=>({id,evaluationId,findingId,createdAt:'2026-09-12T12:00:00Z',document:{sceneRevisionId:revision,displayName:id,reason:id+' reason',decision:'rejected',evidenceRefs:[]}});
const evaluations=[evaluation('e-old','old',[finding('f-old'),finding('foreign-finding','new'),{...finding('unknown'),applicability:'unknown',machineResult:'PASS'}]),evaluation('e-new','new',[finding('f-new','new')])];
const reviews=[review('exact-review','e-old','f-old'),review('wrong-evaluation','e-new','f-old'),review('wrong-finding','e-old','missing'),review('wrong-revision','e-old','f-old','new')];
const before=JSON.stringify({evaluations,reviews});
const exact=exactReviewEvidence('old',evaluations,reviews);
assert.deepEqual(assessmentSummary('unassessed',evaluations),{revisionId:'unassessed',state:'unassessed',evaluationCount:0,attentionCount:0},'past and future findings are never current compliance');
assert.deepEqual(assessmentSummary('old',evaluations),{revisionId:'old',state:'assessed',evaluationCount:1,attentionCount:2},'unknown applicability remains attention even when the machine field says PASS');
assert.equal(assessmentSummary('old',[evaluation('mixed','old',[{...finding('pass'),machineResult:'PASS'},{...finding('skip'),applicability:'not_applicable'},finding('fail')])]).attentionCount,1);
assert.deepEqual(exact.evaluations.map(e=>e.id),['e-old']);
assert.deepEqual(exact.evaluations[0].document.findings.map(f=>f.id),['f-old','unknown']);
assert.deepEqual(exact.reviews.map(r=>r.id),['exact-review']);
assert.equal(JSON.stringify({evaluations,reviews}),before,'source snapshot must not be mutated');
const doc={entities:[{id:'bollard',label:'Target bollard'}],assets:[],cameras:[],annotations:[]};
const publication={id:'publication',projectId:'p',snapshot:{revision:{id:'old',document:doc},evaluations,reviews}};
const detail={project:{id:'p'},branch:{id:'branch',kind:'reconstruction',headRevisionId:'new'},revision:{id:'new',document:{...doc,entities:[{id:'wrong',label:'Wrong revision object'}]}}};
function render(p=publication){return renderToStaticMarkup(React.createElement(ReportReview,{detail,publication:p,reviewMode:true,canWrite:true,onSelect:()=>{},onSaved:()=>{}}));}
const html=render();
assert.match(html,/saved publication snapshot/);
assert.match(html,/Target bollard/); assert.doesNotMatch(html,/Wrong revision object/);
assert.ok(html.indexOf('Target bollard')<html.indexOf('Saved policy title'),'object context precedes the rule title');
assert.match(html,/<details class="rr-reason"><summary>/,'long reasons are retained in a closed disclosure');
assert.match(html,/<details class="rr-source"><summary>/,'original clauses and technical IDs are available on demand');
assert.match(html,/exact-review reason/); assert.doesNotMatch(html,/wrong-(evaluation|finding|revision) reason/);
assert.doesNotMatch(html,/e-new|foreign-finding/);
assert.match(html,/applicability is unconfirmed/);
assert.doesNotMatch(html,/rr-pass[\s"]|Save attributed review|Assess this revision|Save policy draft/,'a published snapshot must not expose write controls');
assert.match(render({...publication,snapshot:{...publication.snapshot,evaluations:[],reviews:[]}}),/has not been assessed/);
const historyOnly={...publication,snapshot:{...publication.snapshot,revision:{id:'old',document:{...doc,reportEvidence:{historical:{findings:[{status:'FAIL'}]}}}},evaluations:[],reviews:[]}};
assert.match(render(historyOnly),/has not been assessed/);assert.doesNotMatch(render(historyOnly),/rr-fail/,'historical failures must not become current assessment results');
language='zh'; assert.match(render(),/判定与理由复核/); assert.match(render(),/规则适用性确认|米制占地范围/);
console.log('Report review checks passed: exact revision/finding/review binding, immutable snapshot, entity context, missing evidence, read-only controls, zh/en.');
