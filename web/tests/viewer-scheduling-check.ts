import assert from 'node:assert/strict';
import {framePaint,sceneRepresentationTasks} from '../src/viewer/native-viewer.ts';
import type {SceneDocument} from '../src/types.ts';
let draws=0,seq=0;const frames=new Map<number,()=>void>();
const paint=framePaint(()=>draws++,cb=>{frames.set(++seq,cb);return seq;},id=>{frames.delete(id);});
for(let i=0;i<500;i++)paint.request();
assert.equal(frames.size,1,'mouse-event bursts must schedule only one paint');
const cb=[...frames.values()][0];frames.clear();cb();assert.equal(draws,1);
paint.request();paint.cancel();assert.equal(frames.size,0,'dispose/capture must cancel pending paint');
paint.request();assert.equal(frames.size,1,'painting resumes after a synchronous capture');
const transform={coordinateFrameId:'f',position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]};
const rep=(id:string,kind:string,imageId='photo')=>({id,kind,assetId:id,coordinateFrameId:'f',transform,placementState:'confirmed',sourceRefs:[{imageId,legacyObjectId:id}]});
const document={entities:[{id:'object',observationRefs:['observation'],activeModelRepresentationId:'model',representations:[rep('model','generated_mesh'),rep('observed','observed_surface'),rep('other-photo','observed_surface','other'),rep('cloud','point_cloud')]},{id:'excluded',visible:false,activeModelRepresentationId:'excluded-model',representations:[rep('excluded-model','generated_mesh')]}],observations:[{id:'observation',imageId:'photo',revision:1}]} as unknown as SceneDocument;
const model={modelOnly:true,generated_mesh:true,primitive:true,point_cloud:false,observed_surface:false,imageId:'photo',showCandidates:true};
assert.deepEqual(sceneRepresentationTasks(document,'f',model).map(({r})=>r.id),['model'],'model view must not decode hidden point clouds, source meshes or excluded objects');
// Context geometry has no object mask linkage; it is still required for evidence views.
document.entities[0].sourceContext=true;
const evidence={...model,modelOnly:false,generated_mesh:false,primitive:false,observed_surface:true};
assert.deepEqual(sceneRepresentationTasks(document,'f',evidence).map(({r})=>r.id),['observed']);
const cloud={...evidence,observed_surface:false,point_cloud:true};
assert.deepEqual(sceneRepresentationTasks(document,'f',cloud).map(({r})=>r.id),['cloud']);
assert.deepEqual(sceneRepresentationTasks(document,'different-frame',model),[]);
console.log('viewer scheduling and layer loading checks passed');

// The isolated viewer must load only its selected representations, including
// linked parts, without decoding the rest of the scene a second time.
document.entities[0].sourceContext=false;
document.entities.push({id:'neighbor',activeModelRepresentationId:'neighbor-model',representations:[rep('neighbor-model','generated_mesh')]} as any);
assert.deepEqual(sceneRepresentationTasks(document,'f',{...model,entityIds:['object'],representationIds:['model']}).map(({r})=>r.id),['model']);
assert.deepEqual(sceneRepresentationTasks(document,'f',{...model,entityIds:['object','neighbor'],representationIds:['model','neighbor-model']}).map(({r})=>r.id),['model','neighbor-model']);
assert.deepEqual(sceneRepresentationTasks(document,'f',{...evidence,entityIds:['object'],representationIds:['observed']}).map(({r})=>r.id),['observed']);
assert.deepEqual(sceneRepresentationTasks(document,'f',{...model,entityIds:['object'],representationIds:['missing']}).map(({r})=>r.id),[]);
console.log('PASS: isolated model, assembly and observed-surface loading scopes');

// Moving objects: every timed surface stays loaded whatever the time or the static/dynamic switch (no reload while
// the video plays); each is drawn only inside its own time range; 'dynamic' hides the static scene, 'static' the movers.
import {representationPass} from '../src/viewer/native-viewer.ts';
const timed=(id:string,t0:number,t1:number)=>({...rep(id,'observed_surface'),timeRange:[t0,t1]});
const mover={id:'mover',motion:'dynamic',representations:[timed('at-0',0,.1),timed('at-1',.1,.2)]} as any;
const motion={entities:[document.entities[0],mover],observations:document.observations} as unknown as SceneDocument;
for(const extra of [{},{time:.15},{time:.15,part:'static'},{time:.15,part:'dynamic'}])
  assert.deepEqual(sceneRepresentationTasks(motion,'f',{...model,...extra}).filter(({e})=>e.id==='mover').map(({r})=>r.id),['at-0','at-1'],'timed surfaces are loaded once, not per moment');
const drawn=(extra:any)=>[...document.entities[0].representations!,...mover.representations].filter((r:any)=>representationPass(r.timeRange?mover:document.entities[0],r,'f',{...model,...extra}).visible).map((r:any)=>r.id);
assert.deepEqual(drawn({time:.15}),['model','at-1'],'the static scene and the mover of this moment');
assert.deepEqual(drawn({time:.15,part:'static'}),['model'],'static only');
assert.deepEqual(drawn({time:.15,part:'dynamic'}),['at-1'],'dynamic only');
assert.deepEqual(drawn({}),['model'],'no video time: no mover');
console.log('moving-object time and static/dynamic checks passed');

// A live report grows while it is open: a representation whose entity, id, kind, asset and primitive are unchanged keeps its
// GPU buffers across setScene (representationKey), anything else is reloaded.
import {representationKey} from '../src/viewer/native-viewer.ts';
const box={id:'box',kind:'primitive',primitive:{kind:'box',dimensions:[1,1,1]},transform};
assert.equal(representationKey('o',box),representationKey('o',{...box,transform:{...transform,position:[1,2,3]}}),'a move is drawn, not reloaded');
assert.notEqual(representationKey('o',box),representationKey('o',{...box,primitive:{kind:'box',dimensions:[1,2,1]}}),'a new box shape reloads');
assert.notEqual(representationKey('o',rep('m','generated_mesh')),representationKey('o',{...rep('m','generated_mesh'),assetId:'other'}),'a new asset reloads');
assert.notEqual(representationKey('o',box),representationKey('p',box),'another entity');
console.log('live-report representation reuse checks passed');
