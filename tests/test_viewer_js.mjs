// Run the actual generated viewer script; only DOM/WebGL are substituted.
// node tests/test_viewer_js.mjs runs/user-bor1-02/viewer.html
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.slice(html.lastIndexOf('<script>') + 8, html.lastIndexOf('</script>'));
const listeners = new Map(), sent = [], uniforms = new Map();
let pickId = 0;
const gl = new Proxy({
  getShaderParameter: () => true, getProgramParameter: () => true,
  uniform1f: (name, value) => uniforms.set(name, value), getUniformLocation: (_, name) => name,
  readPixels: (_x, _y, _w, _h, _f, _t, p) => p.set([pickId >> 8, pickId & 255, 255, 255]),
}, {get: (o, k) => k in o ? o[k] : /^[A-Z_0-9]+$/.test(k) ? 1 : () => ({})});
class Element {
  constructor(){ this.style = {}; this.children = []; this.events = {}; this.attrs = {}; this.className = ''; }
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
const parent = {origin:'https://panoptes.local', location:{origin:'null'}, postMessage:(message, target) =>
  sent.push({message:JSON.parse(JSON.stringify(message)),target})};
const context = vm.createContext({console, atob, Uint8Array, Uint16Array, Float32Array,
  document:{getElementById:get, createElement:() => new Element()}, parent,
  location:{origin:'null'}, innerWidth:478, devicePixelRatio:1,
  addEventListener:(name, fn) => listeners.set(name, fn),
});
context.window = context;
vm.runInContext(script, context, {timeout:30000});
const read = expression => JSON.parse(JSON.stringify(vm.runInContext(expression, context)));
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
console.log(JSON.stringify({status:'passed',objects:objects.length,supported_inv:supported,
  checks:'ready, parent/source security, multi-selection, group selection, all/clear, list, keyboard, pick, drag, resize'}));
