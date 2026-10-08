// Run: node --experimental-strip-types tests/cad-selection-check.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createRequire} from 'node:module';
import ts from 'typescript';
import React from 'react';
import * as core from '../src/core.ts';
import * as semantics from '../src/scene-semantics.ts';
import {translate} from '../src/translate.ts';
const require=createRequire(import.meta.url),slots=[];let cursor=0;
const hooks={useEffect(){},useMemo:fn=>fn(),useRef(initial){const i=cursor++;return slots[i]??={current:initial};},
  useState(initial){const i=cursor++;if(!Object.hasOwn(slots,i))slots[i]=typeof initial==='function'?initial():initial;return [slots[i],v=>{slots[i]=typeof v==='function'?v(slots[i]):v;}];}};
const module={exports:{}};
const code=ts.transpileModule(fs.readFileSync(new URL('../src/CadView.tsx',import.meta.url),'utf8'),
  {compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX}}).outputText;
new Function('require','module','exports',code)(name=>{
  if(name==='react')return hooks;if(name.endsWith('.css'))return {};
  if(name==='./core')return core;if(name==='./scene-semantics')return semantics;
  if(name==='./i18n')return {useI18n:()=>({language:'en',t:(key,params)=>translate('en',key,params)})};
  return require(name);
},module,module.exports);
const transform={coordinateFrameId:'frame',position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]},plane=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const ring=(lo,hi)=>[[lo,lo],[hi,lo],[hi,hi],[lo,hi],[lo,lo]];
const entities=['open panel','inner part'].map((id,i)=>({id,label:id,observationRefs:[],representations:[{id,kind:'generated_mesh',assetId:id,
  coordinateFrameId:'frame',transform,placementState:'confirmed',bounds:{min:[i?4.5:0,i?4.5:0,0],max:[i?5.5:10,i?5.5:10,1]},
  planProjection:{methodVersion:'indexed-mesh-triangle-union-v1',coordinateFrameId:'frame',assetId:id,assetSha256:id,
    transformSnapshot:transform,groundNormalSnapshot:[0,0,1],nativeToPlane:plane,
    polygons:[{exterior:ring(i?4.5:0,i?5.5:10),holes:i?[]:[ring(4,6)]}],lines:[]}}],activeModelRepresentationId:id}));
const document={captureId:'capture',entities,observations:[],annotations:[],assets:entities.map(e=>({id:e.id,sha256:e.id})),
  coordinateFrames:[{id:'frame',scale:{status:'uncalibrated'},ground:{normal:[0,0,1]}}],reportEvidence:{plan:{coordinateFrameId:'frame',nativeToFloor:plane}}};
const selections=[];function render(){cursor=0;return module.exports.CadView({document,selectedId:null,onSelect:id=>selections.push(id),geometryOptions:{layer:'model',frameId:'frame'}});}
function find(node,predicate){if(!React.isValidElement(node))return null;if(predicate(node))return node;for(const child of React.Children.toArray(node.props.children)){const found=find(child,predicate);if(found)return found;}return null;}
let tree=render();const svg=find(tree,n=>n.type==='svg'),button=find(tree,n=>n.props.role==='button'&&n.props['aria-label']==='#01 open panel');
assert.ok(button);assert.equal(button.props['data-cad-label'],'open panel','The named CAD button must be the numbered label, not an open contour bounding box');
assert.ok(find(button,n=>n.type==='rect'));assert.equal(find(button,n=>n.type==='path'),null,'Leader lines must stay outside the button hit box');
assert.deepEqual(core.planHits(core.planShapes(document,{layer:'model',frameId:'frame'}),5,5,0).map(s=>s.entity.id),['inner part'],'The open panel bounding-box centre belongs to another object');
const surface={focus(){},getScreenCTM:()=>({inverse:()=>({})}),hasPointerCapture:()=>false,setPointerCapture(){},releasePointerCapture(){}};svg.props.ref.current=surface;
const target={closest:selector=>selector==='[data-cad-label]'?{getAttribute:()=>button.props['data-cad-label']}:null};
const event={button:0,pointerId:1,clientX:300,clientY:200,pointerType:'mouse',buttons:1,currentTarget:surface,target};
svg.props.onPointerDown(event);svg.props.onPointerUp(event);svg.props.onClick(event);assert.deepEqual(selections,['open panel'],'One label click selects its identity once and bypasses spatial hit testing');
button.props.onKeyDown({key:'Enter',preventDefault(){},stopPropagation(){}});assert.deepEqual(selections,['open panel','open panel']);
svg.props.onPointerDown(event);svg.props.onPointerMove({...event,clientX:320});svg.props.onPointerUp(event);svg.props.onClick(event);assert.equal(selections.length,2,'Dragging across a label never activates it');
globalThis.DOMPoint=function(x,y){return {matrixTransform:()=>({x,y})};};
svg.props.onPointerDown(event);svg.props.onPointerUp(event);svg.props.onClick({...event,target:{closest:()=>null}});assert.equal(selections.at(-1),'inner part','Ordinary canvas clicks still hit the actual geometry');
console.log('CAD selection: named label hit box, hole/overlap identity, single activation, keyboard and drag suppression passed.');
