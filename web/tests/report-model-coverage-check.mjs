// Run: node --experimental-strip-types tests/report-model-coverage-check.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createRequire} from 'node:module';
import ts from 'typescript';
import React from 'react';
import * as core from '../src/core.ts';
import * as semantics from '../src/scene-semantics.ts';
import * as math from '../src/viewer/native-math.ts';
import * as measurement from '../src/measurement-layer.ts';
const require=createRequire(import.meta.url), slots=[]; let cursor=0;
const hooks={useId:()=> 'scene',useEffect(){},useRef:initial=>({current:initial}),
  useState(initial){const i=cursor++;if(!Object.hasOwn(slots,i))slots[i]=typeof initial==='function'?initial():initial;return [slots[i],v=>{slots[i]=typeof v==='function'?v(slots[i]):v;}];}};
const SpatialView=()=>null, empty=()=>null, module={exports:{}};
const code=ts.transpileModule(fs.readFileSync(new URL('../src/ReportScene.tsx',import.meta.url),'utf8'),
  {compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX}}).outputText;
new Function('require','module','exports',code)(name=>{
  if(name==='react')return hooks;
  if(name.endsWith('.css'))return {};
  if(name==='./i18n')return {useI18n:()=>({t:key=>key})};
  if(name==='./core')return core;
  if(name==='./measurement-layer')return measurement;
  if(name==='./SceneResources')return {useSceneResources:()=>({analysisAvailable:true})};
  if(name==='./scene-semantics')return semantics;
  if(name==='./viewer/native-math')return math;
  if(name==='./App')return {SpatialView};
  if(name==='./PhotoView')return {PhotoView:empty};
  if(name==='./CadView')return {CadView:empty};
  if(name==='./VideoView')return {VideoView:empty,VideoMemory:empty,ComparisonVideo:empty,videoReplay:()=>null};
  if(name==='./SpatialMeasurements')return {SpatialMeasurements:empty};
  if(name==='./viewer/splat-layer')return {splatAnnotation:()=>null};
  if(name==='./api')return {}; // only effects fetch, and effects never run here
  return require(name);
},module,module.exports);
const transform={coordinateFrameId:'frame',position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]};
const entity=id=>({id,label:id,observationRefs:[id+'-obs'],associationState:'confirmed',representations:[],measurements:{}});
const models=Array.from({length:24},(_,i)=>({...entity('model-'+i),activeModelRepresentationId:'mesh-'+i,
  representations:[{id:'mesh-'+i,kind:'generated_mesh',assetId:'asset-'+i,coordinateFrameId:'frame',transform,
    bounds:{min:[0,0,0],max:[1,1,1]},placementState:i?'confirmed':'unconfirmed',placementReason:i?null:'imported_proposal'}]}));
const floor={...entity('floor'),representations:[{id:'floor-mesh',kind:'observed_surface',sourceKind:'observed_reference_surface',
  assetId:'floor-asset',coordinateFrameId:'frame',transform,bounds:{min:[0,0,0],max:[2,2,0]},placementState:'confirmed',
  sourceRefs:[{observationId:'floor-obs',revision:1,imageId:'photo'}]}]};
const entities=[...models,floor,entity('composite'),entity('missing-1'),entity('missing-2'),{...entity('hidden'),visible:false}];
const document={schemaVersion:2,geometryBindings:{},geometrySolutions:[],identityDecisions:[],cameras:[],
  coordinateFrames:[{id:'frame',scale:{status:'uncalibrated',nativeToMeters:null}}],entities,
  assets:entities.flatMap(e=>e.representations.map(r=>({id:r.assetId}))),
  observations:entities.map(e=>({id:e.id+'-obs',revision:1,imageId:'photo',originalPixelBox:[0,0,10,10]})),
  annotations:[{id:'mapping',kind:'observation_component_mapping',entityId:'composite',representationType:'composite_source_evidence',independentObject:false,
    unresolvedBoundaryObservationId:'composite-obs',unresolvedBoundaryObservationRevision:1,
    targets:[{entityId:'model-0',observationId:'model-0-obs',observationRevision:1,activeModelRepresentationId:'mesh-0'}]}]};
function find(node,predicate){if(!React.isValidElement(node))return null;if(predicate(node))return node;for(const child of React.Children.toArray(node.props.children)){const hit=find(child,predicate);if(hit)return hit;}return null;}
function text(node){return React.isValidElement(node)?React.Children.toArray(node.props.children).map(text).join(''):String(node??'');}
function render(){cursor=0;return module.exports.ReportScene({revision:{id:'revision',document},selection:{entityId:null},imageId:null,cameraId:null,onSelect(){},onCamera(){}});}
function coverage(tree){return find(tree,n=>Object.hasOwn(n.props,'data-model-coverage'));}
let tree=render(), notice=coverage(tree);
assert.deepEqual(['data-model-coverage','data-reference-surfaces','data-composite-previews','data-missing-models'].map(k=>notice.props[k]),[24,1,1,2]);
assert.match(text(notice),/26 \/ 28 sceneGeometryCoverage · 24 sceneIndependentModelCount/);
assert.match(text(notice),/24 sceneModelLoading/);assert.match(text(notice),/1 sceneModelCandidateCount/);
find(tree,n=>n.type===SpatialView).props.onAssetStates('revision',models.slice(0,2).map((e,i)=>({entityId:e.id,representationId:e.activeModelRepresentationId,assetId:'asset-'+i,state:i?'error':'ready'})));
notice=coverage(render());assert.equal(notice.props['data-model-loaded'],1);assert.equal(notice.props['data-model-coverage'],24);
assert.match(text(notice),/22 sceneModelLoading/);assert.match(text(notice),/1 sceneModelLoadFailed/);
models[0].visible=false;notice=coverage(render());
assert.deepEqual(['data-model-coverage','data-reference-surfaces','data-composite-previews','data-missing-models'].map(k=>notice.props[k]),[23,1,0,3],
  'a composite with no visible component geometry is missing, and hidden entities leave the denominator');
console.log('Report coverage: mutually exclusive 24 models + 1 reference + 1 composite + 2 missing; loading, errors, candidates and visibility retained.');
