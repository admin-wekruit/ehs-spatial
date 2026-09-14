// Run: node tests/photo-draw-check.mjs
// Execute the real PhotoView handlers with persistent hooks and batched rerenders.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import ts from 'typescript';
import React from 'react';

const require=createRequire(import.meta.url),root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const slots=[],effects=[],cache=new Map();let cursor=0;
const hooks={
  useState(initial){
    const index=cursor++;
    if(!Object.hasOwn(slots,index))slots[index]=typeof initial==='function'?initial():initial;
    return [slots[index],value=>{slots[index]=typeof value==='function'?value(slots[index]):value;}];
  },
  useRef(initial){const index=cursor++;return slots[index]??=( {current:initial} );},
  useEffect(effect,deps){
    const index=cursor++,previous=slots[index];
    if(!previous||deps.some((value,i)=>!Object.is(value,previous.deps[i])))effects.push(()=>{previous?.cleanup?.();slots[index]={deps,cleanup:effect()};});
  },
};
function load(filename){
  if(cache.has(filename))return cache.get(filename).exports;
  const module={exports:{}};cache.set(filename,module);
  const code=ts.transpileModule(fs.readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX}}).outputText;
  function dependency(name){
    if(name==='react')return hooks;
    if(name==='./api')return {resolveAsset:async()=> 'https://example.invalid/photo.png'};
    if(name==='./i18n')return {useI18n:()=>({t:key=>key})};
    if(!name.startsWith('.'))return require(name);
    const resolved=path.resolve(path.dirname(filename),name);
    return load([resolved,resolved+'.ts',resolved+'.tsx'].find(p=>fs.existsSync(p)));
  }
  new Function('require','module','exports',code)(dependency,module,module.exports);
  return module.exports;
}
const {PhotoView}=load(path.join(root,'src/PhotoView.tsx'));
const document={cameras:[{imageId:'photo',width:800,height:600}],entities:[{id:'button',label:'Button',observationRefs:['button-photo']}],observations:[{id:'button-photo',imageId:'photo',originalPixelBox:[0,0,800,600]}]};
let draw=true,hostBox=null,showBounds;const selections=[],boxes=[],captured=[];
const surface={getBoundingClientRect:()=>({left:100,top:50,width:400,height:300}),setPointerCapture:id=>captured.push(id)};
function findSVG(node){
  if(!React.isValidElement(node))return null;
  if(node.type==='svg')return node;
  for(const child of React.Children.toArray(node.props.children)){const found=findSVG(child);if(found)return found;}
  return null;
}
function render(){
  cursor=0;
  const tree=PhotoView({document,imageId:'photo',selectedId:null,draw,showBounds,onSelect:(...selection)=>{selections.push(selection);hostBox=null;},onBox:box=>{boxes.push(box);hostBox=box;draw=false;}});
  for(const effect of effects.splice(0))effect();
  const svg=findSVG(tree);if(svg)svg.props.ref.current=surface;
  if(svg)assert.equal(tree.props.className.includes('show-bounds'),showBounds === true);
  return svg;
}
assert.equal(render(),null,'the source asset resolves before pointer interaction');
await Promise.resolve();
let svg=render();assert.ok(svg);
const pointer=(clientX,clientY)=>({clientX,clientY,pointerId:7});
svg.props.onPointerDown(pointer(140,90));
svg=render();
svg.props.onPointerMove(pointer(200,150));
// The final move is not committed to React state before pointer-up. Its coordinates win.
svg.props.onPointerUp(pointer(260,210));
assert.deepEqual(captured,[7]);
assert.deepEqual(boxes,[[80,80,320,320]],'onBox must use pointer-up coordinates, not stale box state');
assert.equal(draw,false,'the host exits drawing mode when accepting the box');
svg=render();
svg.props.onClick(pointer(260,210));
assert.equal(selections.length,0,'the click belonging to the completed draw must not select an object');
assert.deepEqual(hostBox,[80,80,320,320],'a host selection callback must not clear the accepted box');
// The gesture guard consumes one click only; ordinary pointer selection still works.
svg.props.onPointerDown(pointer(260,210));
svg.props.onPointerUp(pointer(260,210));
svg.props.onClick(pointer(260,210));
assert.deepEqual(selections,[['button','button-photo']]);
assert.equal(boxes.length,1,'an ordinary click must not emit a second drawing');
showBounds=true;render();
showBounds=false;render();
console.log('Photo draw regression passed: pointer-up coordinates, host draw=false rerender, suppressed synthetic click, and subsequent ordinary selection.');
