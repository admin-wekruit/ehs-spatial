// Run the actual generated viewer script; only DOM/WebGL are substituted.
// node tests/test_viewer_js.mjs runs/user-bor1-02/viewer.html
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
const withSurface = process.argv.includes('--surface');
const failSurface = process.argv.includes('--surface-error');
const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.slice(html.lastIndexOf('<script>') + 8, html.lastIndexOf('</script>'));
const listeners = new Map(), sent = [], uniforms = new Map();
let pickId = 0;
const gl = new Proxy({
  getShaderParameter: () => true, getProgramParameter: () => true,
  uniform1f: (name, value) => uniforms.set(name, value), getUniformLocation: (_, name) => name,
  uniformMatrix4fv: (name,_transpose,value) => uniforms.set(name,Array.from(value)),
  readPixels: (_x, _y, _w, _h, _f, _t, p) => p.set([pickId >> 8, pickId & 255, 255, 255]),
}, {get: (o, k) => k in o ? o[k] : /^[A-Z_0-9]+$/.test(k) ? 1 : () => ({})});
class Element {
  constructor(){ this.style = {}; this.dataset = {}; this.children = []; this.events = {}; this.attrs = {}; this.className = ''; }
  get classList(){ const el = this; return {
    contains: c => el.className.split(' ').includes(c),
    toggle(c, force){ const set = new Set(el.className.split(' '));
      const on = force ?? !set.has(c); if (on) set.add(c); else set.delete(c);
      el.className = [...set].join(' '); return on; },
    add(c){ this.toggle(c, true); }, remove(c){ this.toggle(c, false); },
  }; }
  appendChild(el){ this.children.push(el); }
  setAttribute(k, v){ this.attrs[k] = v; }
  addEventListener(k, fn){ this.events[k] = fn; }
  getContext(){ return gl; }
  getBoundingClientRect(){ return {left:0, top:0, width:478, height:320}; }
  setPointerCapture(){}
  get clientWidth(){ return 478; } get clientHeight(){ return 320; }
}
const elements = new Map();
const get = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
get('anchors').textContent = html.match(/<script id="anchors"[^>]*>([\s\S]*?)<\/script>/)[1];
get('surface-data').textContent = withSurface ? html.match(/<script id="surface-data"[^>]*>([\s\S]*?)<\/script>/)[1] : 'null';
get('surface-c').parentElement = get('stage');
get('selection-reference').parentElement = get('stage');
const parent = {origin:'https://panoptes.local', location:{origin:'null'}, postMessage:(message, target) =>
  sent.push({message:JSON.parse(JSON.stringify(message)),target})};
const context = vm.createContext({console, atob, Uint8Array, Uint16Array, Uint32Array, Float32Array, TextDecoder, Blob, URL,
  Image:class {set src(_value){this.onload();}},
  fetch:async url => {if(failSurface)return {ok:false,status:409};
    const b=fs.readFileSync(path.join(path.dirname(process.argv[2]),'surface',path.basename(url)));
    return {ok:true,arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)};},
  document:{getElementById:get, createElement:() => new Element(), documentElement:{lang:'zh-CN'}}, parent,
  location:{origin:'null'}, innerWidth:478, devicePixelRatio:1,
  addEventListener:(name, fn) => listeners.set(name, fn),
});
context.window = context;
vm.runInContext(script, context, {timeout:30000});
const read = expression => JSON.parse(JSON.stringify(vm.runInContext(expression, context)));
if(withSurface){
  const sourceFrames=read('SURFACE.cameras.frames');
  if(!failSurface)listeners.get('message')({source:parent,origin:parent.origin,data:{type:'panoptes:frame',frame_id:sourceFrames[1].id}});
  for(let i=0;i<30 && !read('surfaceView !== null');i++)await new Promise(resolve=>setImmediate(resolve));
  if(failSurface){
    assert(read('surfaceMode'),'load errors cannot silently change the requested mode');
    assert.equal(read('surfaceView'),null);assert(get('hud').textContent.includes('内部模型加载失败'));
    get('point-mode').onclick();assert.equal(read('surfaceMode'),false,'measurement mode remains an explicit user choice');
    console.log(JSON.stringify({status:'passed',checks:'visible asset error without automatic fallback'}));process.exit(0);
  }
  assert(read('surfaceView !== null'),read('surfaceStatus'));
  assert(read('surfaceMode'),'available GLB is the default 3D mode');
  assert.equal(read('surfaceView.faceCount'),read('SURFACE.face_count'));
  assert.equal(read('requestedFrameId'),sourceFrames[1].id,'a frame request before mesh loading must be replayed');
  const expected=read('Array.from(surfaceMatrix(surfaceCamera(SURFACE.cameras.frames[1],Math.hypot(...SURFACE.cameras.bounds.max.map((v,i)=>v-SURFACE.cameras.bounds.min[i]))/2),392/518,Math.hypot(...SURFACE.cameras.bounds.max.map((v,i)=>v-SURFACE.cameras.bounds.min[i]))/2))');
  assert.deepEqual(uniforms.get('mvp'),expected,'the actual uploaded mesh camera matrix honors the early source-frame request');
  get('point-mode').onclick();
}
const objects = read('DATA.objects'), supported = read('DATA.supported_inv');
assert(supported.length >= 2, 'this check needs two distinct inventory objects');
const [a,b] = supported, objectA = objects.find(o => o.inv === a), objectB = objects.find(o => o.inv === b);
const state = () => read('selectedInv()');
const send = (inv, extra = {}, origin = parent.origin, source = parent) =>
  listeners.get('message')({source,origin,data:{type:'panoptes:select',inv,exclusive:true,...extra}});
assert.deepEqual(sent.map(e => e.message), [{type:'panoptes:ready',supported_inv:supported,unavailable:read('DATA.unavailable')}]);
assert.equal(sent[0].target, parent.origin);
assert.equal(get('objectlist').open, false, 'narrow iframe starts with its list collapsed');
const before = sent.length;
send([a,b]); assert.deepEqual(state(),[a,b]); assert.equal(sent.length,before,'incoming selection must not echo');
const selectedGeometryCount=objects.filter(o=>[a,b].includes(o.inv)).length;
assert.equal(get('selection-reference').dataset.objectCount,String(selectedGeometryCount));
assert.equal(get('selection-reference').dataset.referenceFrame,'estimated-floor');
assert.equal((get('selection-reference').innerHTML.match(/data-axis=/g)||[]).length,selectedGeometryCount*3);
assert(get('selection-reference-info').textContent.includes('实测倾角未知'));
send([],{},'https://foreign.local'); send([],{},parent.origin,{});
assert.deepEqual(state(),[a,b],'source and origin are both checked');
send([null,String(a),-1,0.5,read('DATA.inventory_count'),a]); assert.deepEqual(state(),[a]);
for (const o of objects.filter(o => o.inv === a)) assert.equal(read(`visData[${o.id}*4+3]`),255);
const row = o => get('list').children[objects.indexOf(o)];
row(objectB).events.click({shiftKey:true}); assert.deepEqual(state(),[a,b]);
row(objectA).events.click({shiftKey:true}); assert.deepEqual(state(),[b]);
row(objectB).events.keydown({key:'Enter',preventDefault(){}}); assert.deepEqual(state(),[]);
row(objectA).events.click({}); row(objectA).events.click({altKey:true});
assert.deepEqual(state(),[]); assert.equal(read(`visData[${objectA.id}*4+3]`),0);
get('all').onclick(); assert.deepEqual([...state()].sort((x,y)=>x-y),read('DATA.interactive_inv'));
for (const o of objects) assert.equal(read(`visData[${o.id}*4+3]`),o.inv === null ? 170 : 255);
get('none').onclick(); assert.deepEqual(state(),[]);
assert.equal(get('selection-reference').innerHTML,'');assert.equal(get('selection-reference-info').hidden,true);
assert.equal(read(`visData[${objectA.id}*4+3]`),170,'clear selection keeps geometry visible');
const unavailable = read('DATA.interactive_inv').find(i => !supported.includes(i));
if (unavailable !== undefined) { send([unavailable]); row(objectA).events.click({shiftKey:true});
  assert.deepEqual(state(),[unavailable,a],'a Shift-click preserves selected inventory lacking geometry'); }
send([]); pickId = objectB.id;
const canvas = get('c'), e = {clientX:120,clientY:120,button:0,pointerId:1};
canvas.events.pointerdown(e); canvas.events.pointerup(e); assert.deepEqual(state(),[b]);
assert.deepEqual(sent.at(-1).message,{type:'panoptes:selected',inv:[b]});
pickId = 0; canvas.events.pointerdown(e); canvas.events.pointerup(e); assert.deepEqual(state(),[]);
send([a]); canvas.events.pointerdown(e); canvas.events.pointermove({...e,clientX:140}); canvas.events.pointerup(e);
assert.deepEqual(state(),[a],'orbiting is not a click');
listeners.get('resize')({type:'resize'}); assert.equal(uniforms.get('pick'),0,'resize restores colour, not the id pass');
assert.deepEqual(objects.map(o=>o.id),objects.map((_,i)=>i+1));
assert(read('ids.reduce((highest,id) => Math.max(highest,id),0)') <= objects.length);
if(withSurface){
  const mapped=read('SURFACE.supported_inv'), missing=read('DATA.interactive_inv').find(i=>!mapped.includes(i));
  get('surface-mode').onclick();send([missing]);
  assert(get('hud').textContent.includes('内部模型未覆盖该对象的机位/关联'));
  const surfaceCanvas=get('surface-c');pickId=mapped[0]+1;
  surfaceCanvas.events.pointerdown(e);surfaceCanvas.events.pointerup({...e,shiftKey:true});
  assert.deepEqual(state(),[missing,mapped[0]],'surface picks use inv+1 and retain unassociated selections');
  assert.deepEqual(sent.at(-1).message,{type:'panoptes:selected',inv:[missing,mapped[0]]});
  assert.equal(get('selection-reference').dataset.referenceFrame,'surface-native');
  assert.equal(get('selection-reference').dataset.objectCount,'1');
  assert.equal(get('selection-reference-info').dataset.height,undefined);
  assert(get('selection-reference-info').textContent.includes('地面高度与实测倾角未知'));
  pickId=0;surfaceCanvas.events.pointerdown(e);surfaceCanvas.events.pointerup(e);
  assert.deepEqual(state(),[]);assert(get('hud').textContent.includes('该表面尚未建立对象对应，暂不能联动'));
  send([missing,mapped[0]]);
  get('point-mode').onclick();assert.deepEqual(state(),[missing,mapped[0]],'mode switch preserves common inventory state');
  get('surface-mode').onclick();get('surface-cams').children[1].onclick();
  assert.deepEqual(sent.at(-1).message,{type:'panoptes:frame',frame_id:read('SURFACE.cameras.frames[1].id')});
  const count=sent.length;
  listeners.get('message')({source:parent,origin:parent.origin,data:{type:'panoptes:frame',frame_id:'frame_0001'}});
  assert(get('hud').textContent.includes('内部模型仅覆盖'));assert(read('surfaceMode'),'uncovered frames never change mode');
  assert.equal(sent.length,count,'parent frame messages never echo');
  listeners.get('message')({source:{},origin:parent.origin,data:{type:'panoptes:frame',frame_id:'frame_0004'}});
  assert.equal(read('requestedFrameId'),'frame_0001','frame commands also check source');
  listeners.get('message')({source:parent,origin:'https://foreign.local',data:{type:'panoptes:frame',frame_id:'frame_0004'}});
  assert.equal(read('requestedFrameId'),'frame_0001','frame commands also check origin');
  get('none').onclick();send([mapped[0]]);
  assert.equal(read('requestedFrameId'),objects.find(o=>o.inv===mapped[0]).frame,'new associated selections use their actual source frame');
  assert(get('hud').textContent.includes('尺度未标定'));
  get('none').onclick();get('all').onclick();assert.deepEqual(state(),read('DATA.interactive_inv'));
  get('none').onclick();
}
console.log(JSON.stringify({status:'passed',objects:objects.length,supported_inv:supported,
  surface:withSurface, checks:'ready, parent/source security, multi-selection, group selection, all/clear, list, keyboard, pick, drag, resize, surface mode/selection'}));
