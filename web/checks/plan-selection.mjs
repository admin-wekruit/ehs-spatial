// Run: node --experimental-strip-types web/checks/plan-selection.mjs
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import ts from 'typescript';
import {planHits,planPolygonPath} from '../src/core.ts';

const shapes=[
  {entity:{id:'fence-a',label:'Safety fence'},min:[0,0],max:[10,10],polygon:[[0,0],[10,0],[10,10],[0,10]]},
  {entity:{id:'fence-b',label:'Right front safety fence panel'},min:[4,4],max:[6,6],polygon:[[4,4],[6,4],[6,6],[4,6]]},
  {entity:{id:'triangle',label:'Other footprint'},min:[7,7],max:[9,9],polygon:[[7,7],[9,7],[7,9]]},
].map(({polygon,...shape})=>({...shape,polygons:[{exterior:polygon,holes:[]}],lines:[]}));
const before=structuredClone(shapes);
assert.deepEqual(planHits(shapes,5,5).map(h=>h.entity.id),['fence-b','fence-a']);
assert.deepEqual(planHits(shapes,8.8,8.8).map(h=>h.entity.id),['fence-a'],'An enclosing bbox must not turn empty triangle area into a hit');
assert.deepEqual(planHits(shapes,4,5).map(h=>h.entity.id),['fence-b','fence-a'],'Shared boundaries remain available in the explicit picker');
assert.deepEqual(planHits(shapes,11,5),[]);
assert.deepEqual(shapes,before,'Hit testing never reorders source scene records');

// Run the actual PlanView event handlers with deterministic React hooks and a
// native DOMPoint-equivalent affine fixture; no WebGL or browser mock scene.
const source=await readFile(new URL('../src/App.tsx',import.meta.url),'utf8'),parsed=ts.createSourceFile('App.tsx',source,ts.ScriptTarget.ES2022,true,ts.ScriptKind.TSX);
const declaration=parsed.statements.find(n=>ts.isFunctionDeclaration(n)&&n.name?.text==='PlanView');
assert.ok(declaration);
const code=ts.transpileModule(declaration.getText(parsed).replace('export function','function'),{compilerOptions:{target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.React,module:ts.ModuleKind.None}}).outputText;
const hooks=[];let cursor=0,effects=[],dirty=false,selected=null,doc={entities:shapes.map(s=>s.entity)},calls=[];
const useState=initial=>{const i=cursor++;if(!(i in hooks))hooks[i]=initial;return[hooks[i],next=>{const value=typeof next==='function'?next(hooks[i]):next;dirty ||= !Object.is(hooks[i],value);hooks[i]=value;}];};
const useRef=()=>{const i=cursor++;return hooks[i]||= {current:null};};
const useEffect=(fn,deps)=>{const i=cursor++,old=hooks[i];if(!old||deps.some((d,j)=>!Object.is(d,old[j])))effects.push(fn);hooks[i]=deps;};
const React={createElement:(type,props,...children)=>({type,props:props||{},children:children.flat(Infinity).filter(Boolean)})};
const context=vm.createContext({React,useState,useRef,useEffect,useI18n:()=>({t:key=>key}),planShapes:()=>shapes,planHits,planPolygonPath,
  DOMPoint:class{constructor(x,y){this.x=x;this.y=y;}matrixTransform(m){return{x:m.a*this.x+m.c*this.y+m.e,y:m.b*this.x+m.d*this.y+m.f};}}});
vm.runInContext(code,context);
const nodes=root=>[root,...root.children.filter(c=>typeof c==='object').flatMap(nodes)];
function render(){let tree;for(let pass=0;pass<5;pass++){cursor=0;effects=[];dirty=false;tree=context.PlanView({document:doc,selectedId:selected,onSelect:id=>{selected=id;calls.push(id);},interactive:true});for(const fn of effects)fn();if(!dirty)break;}return tree;}
function svgFixture(tree){const all=nodes(tree),svg=all.find(n=>n.type==='svg'),drawing=all.find(n=>n.type==='g'&&n.props.ref),zoom=Number(drawing.props.transform.match(/scale\(([^)]+)\)/)[1]);
  // CSS scale 2 + SVG centering + local zoom, with a translated container.
  const a=2*zoom,e=17+2*(300-160*zoom),f=39+2*(200-160*zoom);
  drawing.props.ref.current={getScreenCTM:()=>({inverse:()=>({a:1/a,b:0,c:0,d:1/a,e:-e/a,f:-f/a})})};svg.props.ref.current={focus(){}};
  return{svg,event:(x,y)=>({clientX:e+a*x*32,clientY:f+a*(10-y)*32,detail:1,target:{closest:()=>null}})};
}
let tree=render(),fixture=svgFixture(tree);
fixture.svg.props.onClick(fixture.event(5,5));
assert.deepEqual(calls,[],'Ambiguous mouse selection must not silently change the current entity');
tree=render();let picker=nodes(tree).find(n=>n.props.className==='plan-hit-picker');
assert.ok(picker);assert.equal(nodes(picker).filter(n=>n.props['data-candidate']).length,2);
nodes(picker).find(n=>n.props['data-candidate']==='fence-b').props.onClick();
assert.deepEqual(calls,['fence-b']);tree=render();assert.equal(nodes(tree).some(n=>n.props.className==='plan-hit-picker'),false);
fixture=svgFixture(tree);fixture.svg.props.onClick(fixture.event(1,1));assert.equal(selected,'fence-a','A single polygon hit selects immediately');
tree=render();nodes(tree).find(n=>n.type==='button'&&n.children.includes('＋')).props.onClick();tree=render();fixture=svgFixture(tree);
fixture.svg.props.onClick(fixture.event(5,5));tree=render();assert.ok(nodes(tree).some(n=>n.props.className==='plan-hit-picker'),'Zoomed/translated screen coordinates resolve the same overlapping shapes');
tree.props.onKeyDown({key:'Escape',preventDefault(){},stopPropagation(){}});tree=render();assert.equal(nodes(tree).some(n=>n.props.className==='plan-hit-picker'),false);
fixture=svgFixture(tree);fixture.svg.props.onClick(fixture.event(5,5));tree=render();doc={...doc};tree=render();assert.equal(nodes(tree).some(n=>n.props.className==='plan-hit-picker'),false,'A new revision document clears stale candidates');
fixture=svgFixture(tree);const event=fixture.event(5,5),count=calls.length;fixture.svg.props.onPointerDown({...event,button:0});fixture.svg.props.onPointerMove({...event,clientX:event.clientX+10});fixture.svg.props.onClick(event);assert.equal(calls.length,count);tree=render();assert.equal(nodes(tree).some(n=>n.props.className==='plan-hit-picker'),false,'Dragging is not a click');
nodes(tree).find(n=>n.props['data-plan-entity']==='fence-b').props.onKeyDown({key:'Enter',preventDefault(){},stopPropagation(){}});assert.equal(selected,'fence-b','A keyboard-focused named shape selects that entity explicitly');
console.log('plan selection: exact polygons, overlap choices, unchanged selection, zoom/translation, Esc/revision cleanup, drag and keyboard checks passed');
