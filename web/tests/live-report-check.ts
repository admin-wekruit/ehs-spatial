// The live report's document, built from a recorded patch stream (default: the E9 run 003 fixture), is one the viewer loads:
// every layer's representations are scheduled and every asset decodes. Run: node --experimental-strip-types tests/live-report-check.ts [ROOT REPORT]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {latest,liveDocument,frameOf,type Patch} from '../src/live-report.ts';
import {sceneRepresentationTasks,readPacked,readGLB} from '../src/viewer/native-viewer.ts';
import {splatAnnotation} from '../src/viewer/splat-layer.ts';
import {currentCameras,cameraPath} from '../src/core.ts';

const [root='/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/fb-c-fixture-001',report='fb-fixture-me340-e9-003']=process.argv.slice(2);
const folder=path.join(root,'reports',report,'patches');
const patches:Patch[]=fs.readdirSync(folder).sort().map(n=>JSON.parse(fs.readFileSync(path.join(folder,n),'utf8')));
const blob=(id:string)=>{const b=fs.readFileSync(path.join(root,'blobs/sha256',id.slice(7)));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
const layers={observed_surface:true,point_cloud:true,generated_mesh:true,primitive:true};

// Before any geometry: a video-only document is still legal and schedules nothing.
const early=liveDocument(report,latest(patches.filter(p=>p.layer==='video')));
assert.equal(early.cameras.length,0);assert.deepEqual(sceneRepresentationTasks(early,null,layers),[]);
assert.ok(early.annotations.some(a=>a.kind==='video_replay'),'the video shows before any 3D layer');

const doc=liveDocument(report,latest(patches)),all=latest(patches);
const shots=all.cameras.data.shots,walk=[...shots].sort((a:any,b:any)=>b.keys.length-a.keys.length)[0],frame=frameOf(walk.index);
assert.equal(currentCameras(doc).length,shots.reduce((n:number,s:any)=>n+s.keys.length,0),'every keyframe camera is current');
assert.equal(currentCameras(doc)[0].coordinateFrameId,frame,'the viewer opens on the longest shot');
const walked=cameraPath(doc,frame);
assert.equal(walked.length,walk.keys.length);assert.ok(walked.every(p=>p.time!==null),'the camera path carries video time');

const tasks=sceneRepresentationTasks(doc,frame,layers),ids=new Set(tasks.map(({r})=>r.id));
const objects=all.objects.data.objects.filter((o:any)=>o.shot===walk.index),modeled=new Set(all.models.data.models.map((m:any)=>m.object));
assert.ok(ids.has('room-mesh:'+walk.index)&&ids.has('room-points:'+walk.index),'room mesh and points');
for(const o of objects)assert.ok(ids.has(modeled.has(o.id)?'model:'+o.id:'box:'+o.id),'object '+o.id);
for(const t of all.people.data.tracks.filter((t:any)=>t.shot===walk.index))assert.ok(ids.has('track:'+t.id),'track '+t.id);
assert.ok(tasks.every(({r})=>r.coordinateFrameId===frame),'only the open shot is loaded');

let triangles=0,points=0;
for(const {r} of tasks){
  if(r.kind==='primitive'){assert.ok((r.primitive as any).dimensions.every((v:number)=>v>0));continue;}
  const meta=doc.assets.find(a=>a.id===r.assetId) as any;
  const meshes=meta.format==='panoptes-mesh-v1'?readPacked(blob(r.assetId!),meta):readGLB(blob(r.assetId!));
  assert.ok(meshes.length&&meshes.every(m=>m.vertices.length&&m.indices.length),r.id);
  for(const m of meshes){if(m.mode===4)triangles+=m.indices.length/3;else points+=m.indices.length;}
}
const splat=splatAnnotation(doc)!;
assert.ok(splat&&splat.coordinateFrameId===frame,'splat in the walk frame');
assert.equal(splat.count*32,blob(splat.assetId).byteLength);
const replay=doc.annotations.find(a=>a.kind==='video_replay') as any,analysis=JSON.parse(Buffer.from(blob(replay.analysisAssetId)).toString());
assert.ok(analysis.frames.length&&analysis.frames.every((f:any)=>f.objects.every((o:any)=>['segmented','projected'].includes(o.source))),'outlines say how they were made');
assert.ok((doc.annotations.find(a=>a.kind==='video_events') as any).windows.length);
console.log(`live report check passed: ${patches.length} patches, ${tasks.length} representations in ${frame} (${triangles} triangles, ${points} points), ${objects.length} objects, ${modeled.size} models, ${splat.count} splats`);
