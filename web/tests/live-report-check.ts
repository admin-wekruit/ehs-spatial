// The live report's document, built from a recorded patch stream (default: the E9 run 003 fixture), is one the viewer loads:
// every layer's representations are scheduled and every asset decodes. First, the click MVP's pick layer on a synthetic blob (RLE
// decode, ties, misses, unprojection, card merge); with --mvp ROOT REPORT, a recorded pick layer against its own outlines.
// Run: node --experimental-strip-types tests/live-report-check.ts [ROOT REPORT] [--mvp ROOT REPORT]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import {latest,liveDocument,frameOf,type Patch} from '../src/live-report.ts';
import {readPick,gunzip,pickAt,pickMask,frameIndexAt,pickIndexAt,pickChunks,emptyPick,fillChunk,chunkOrder,unknownRegion,entityInfo,worstVerdict,pointInPolygon} from '../src/live-report.ts';

// ---------------- click MVP: a synthetic pick layer (CLICK-MVP-SPEC 3.3)
const rle=(m:Uint16Array)=>{const o:number[]=[];let v=m[0],n=0;for(const x of m){if(x===v&&n<65535){n++;continue;}o.push(v,n);v=x;n=1;}o.push(v,n);return o;};
const entities=[null,'obj-0-big','obj-0-small','person:0-p'];
const f0=new Uint16Array(64*36),rect=(v:number,x0:number,x1:number,y0:number,y1:number)=>{for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++)f0[y*64+x]=v;};
rect(1,0,40,0,30);rect(2,10,20,10,20);rect(3,15,25,5,35);  // painted in tie order: the smaller mask over the larger, the person last
const f1=new Uint16Array(640*360);f1[f1.length-1]=1;        // 230399 zeros: split into 65535-long runs
const pairs=[rle(f0),rle(f1)];
assert.deepEqual(pairs[1].filter((_,i)=>i%2).slice(0,4),[65535,65535,65535,230399-3*65535],'long runs split');
const frames=[{t:0,t_end:1,frame:0,shot:0,key:0,w:64,h:36,source:'segmented',offset:0,pairs:pairs[0].length/2},
  {t:1,t_end:2,frame:30,shot:0,key:1,w:640,h:360,source:'projected',offset:pairs[0].length/2,pairs:pairs[1].length/2}];
// depth: floor 1.6 m below an identity camera (opencv, y down), a wall 5 m ahead; K on the 504x280 grid
const K=[[252,0,252],[0,252,140],[0,0,1]],I4=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],depth=new Uint16Array(2*126*70);
for(let f=0;f<2;f++)for(let r=0;r<70;r++)for(let c=0;c<126;c++){const dy=((r+.5)*4-140)/252;depth[f*8820+r*126+c]=Math.round(1000*(dy>0?Math.min(1.6/dy,5):5));}
const data:any={format:'panoptes-pick-v1',source_wh:[1280,720],entities,frames,depth:{w:126,h:70,unit:'mm',scale:'estimated'}};
const pick=readPick(data,await gunzip(zlib.gzipSync(Buffer.from(Uint16Array.from(pairs.flat()).buffer))),await gunzip(zlib.gzipSync(Buffer.from(depth.buffer))));
const at=(t:number,col:number,row:number,w=64)=>pickAt(pick,t,(col+.5)*1280/w,(row+.5)*720/(w===64?36:360)).id;
assert.equal(at(0,5,5),'obj-0-big');assert.equal(at(0,12,12),'obj-0-small','the smaller mask wins');assert.equal(at(0,17,15),'person:0-p','a person beats objects');
assert.equal(at(0,50,5),null,'a miss');assert.equal(at(.49,5,5),'obj-0-big');assert.equal(at(-.2,5,5),'obj-0-big','just before the first keyframe: the first');
assert.equal(at(-3,5,5),null,'long before the first keyframe: no map');assert.equal(at(.51,55,55,640),null,'past half the gap: the next keyframe (frame 1 has only its last pixel)');
assert.equal(at(1,0,0,640),null);assert.equal(at(1.5,639,359,640),'obj-0-big','the last pixel after split runs');
assert.equal(pickMask(pick,0,'obj-0-small')!.mask.reduce((a,b)=>a+b,0),50,'mask: 10x10 minus the person over it');
assert.deepEqual([-1,.99,1,5].map(t=>frameIndexAt(frames,t)),[0,0,1,1]);
// the nearest keyframe within a shot, never across a cut (shot 0: t 0 and .4, map until .6; cut; shot 1 starts at .8, next map 1.0)
const fr2:any=[{t:0,t_end:.4,shot:0},{t:.4,t_end:.6,shot:0},{t:.8,t_end:1,shot:1},{t:1,t_end:1.2,shot:1}];
assert.deepEqual([-.3,-.1,.1,.25,.55,.6,.62,.7,.85,.95,1.25].map(t=>pickIndexAt(fr2,t)),[-1,0,0,1,1,2,2,2,2,3,-1]);
assert.equal(pickIndexAt([{t:0,t_end:.4,shot:0},{t:.4,t_end:.8,shot:1}] as any,.3),0,'the nearer map is in the next shot: never across the cut');
// chunks (mvp2): one blob per frame range; a frame is pending until its chunk is in, then reads as the one-blob layer
const chunked:any={...data,chunks:[{frames:[0,1],blob:'pick-0',depth_blob:'depth-0'},{frames:[1,2],blob:'pick-1',depth_blob:'depth-1'}]};
assert.deepEqual(pickChunks(chunked,{}).map(c=>[c.lo,c.hi,c.pick,c.depth]),[[0,1,'pick-0','depth-0'],[1,2,'pick-1','depth-1']]);
assert.deepEqual(pickChunks(data,{depth:{}}),[{lo:0,hi:2,pick:'pick',depth:'depth'}],'a one-blob layer is one chunk');
const cp=emptyPick(chunked),u16=(a:number[])=>Uint16Array.from(a).buffer as ArrayBuffer;
assert.deepEqual(chunkOrder(cp,pickChunks(chunked,{}),1.3).map(c=>c.lo),[1,0],'the chunk at the video time first');
assert.ok(pickAt(cp,0,110,110).pending&&pickMask(cp,0,'obj-0-big')===null,'pending before its chunk');
fillChunk(cp,pickChunks(chunked,{})[1],u16(pairs[1]),depth.buffer.slice(8820*2,2*8820*2));
assert.equal(pickAt(cp,1.5,1279,719).id,'obj-0-big');assert.ok(pickAt(cp,.2,110,110).pending,'frame 0 still pending');
assert.throws(()=>fillChunk(cp,pickChunks(chunked,{})[0],u16(pairs[0].slice(2)),null),/pick_layout/,'runs that do not add up to w*h');
fillChunk(cp,pickChunks(chunked,{})[0],u16(pairs[0]),depth.buffer.slice(0,8820*2));
for(const [t,c,r] of [[0,5,5],[0,12,12],[0,17,15],[0,50,5]])assert.equal(pickAt(cp,t,(c+.5)*20,(r+.5)*20).id,at(t,c,r),`chunked == one blob at ${c},${r}`);
assert.equal((unknownRegion(cp,{shots:[]},null,0,640,300) as any).status,'no 3D point here');
const byBytes=readPick({...data,frames:frames.map(f=>({...f,offset:f.offset*4}))},pick.runs.buffer as ArrayBuffer,depth.buffer);
assert.equal(byBytes.unit,4);assert.equal(pickAt(byBytes,0,(17.5)*20,15.5*20).id,'person:0-p','byte offsets read the same');
const cameras={shots:[{index:0,keys:[0,30],times:[0,1],K:[K,K],c2w:[I4,I4],wh:[504,280],source_wh:[1280,720],floor:{normal:[0,1,0],point_m:[0,1.6,0]}}]};
const cardsLayer={shots:[{index:0,u_floor_m:.02}],cards:[{id:'obj-0-big',kind:'object',shot:0,identity:{name:'pallet'},physical:{merged_from:['obj-0-7'],size_check:{status:'plausible'},
  footprint_xy:[[4.45,-.5],[5.45,-.5],[5.45,.5],[4.45,.5]]}},{id:'person:0-p',kind:'person',shot:0}]};
const floor=unknownRegion(pick,cameras,cardsLayer,0,640,60.5*720/70) as any;  // the floor 3.95 m ahead
assert.equal(floor.surface.kind,'floor');assert.ok(Math.abs(floor.height.value)<.02&&Math.abs(floor.distance.value-Math.hypot(1.6,3.95))<.02,JSON.stringify(floor.distance));
assert.ok(floor.height.u>=.02&&floor.distance.u>=.2*floor.distance.value,'u carries floor residual and scale');
assert.equal(floor.nearest.id,'obj-0-big');assert.ok(Math.abs(floor.nearest.d-.5)<.03,'nearest footprint 0.5 m ahead');
const wall=unknownRegion(pick,cameras,cardsLayer,0,640,10.5*720/70) as any;
assert.equal(wall.surface.kind,'vertical surface');assert.ok(Math.abs(wall.height.value-(1.6+98/252*5))<.02,String(wall.height.value));
assert.equal((unknownRegion({...pick,depth:new Uint16Array(depth.length)},cameras,cardsLayer,0,640,300) as any).status,'no 3D point here');
const judgements={rows:[{subject:'obj-0-big',verdict:'PASS'},{subject:'obj-0-big',verdict:'NO_DATA'},{subject:'person:0-p',verdict:'FAIL'}],by_object:{}};
const infos=entityInfo(cardsLayer,judgements);
assert.equal(infos.get('obj-0-7')!.card.id,'obj-0-big','a merged id opens the merged card');
assert.equal(entityInfo({...cardsLayer,aliases:{'obj-0-8':'obj-0-big'}},judgements).get('obj-0-8')!.card.id,'obj-0-big','the layer\'s aliases too');
assert.equal(infos.get('obj-0-big')!.verdict,'NO_DATA','one PASS and a NO_DATA is not a PASS');assert.equal(infos.get('person:0-p')!.verdict,'FAIL');
assert.equal(entityInfo(cardsLayer,{...judgements,by_object:{'obj-0-big':'NEEDS_REVIEW'}}).get('obj-0-big')!.verdict,'NEEDS_REVIEW','the layer\'s by_object wins');
assert.equal(worstVerdict(['PASS','NEEDS_REVIEW','FAIL']),'FAIL');assert.equal(worstVerdict([]),null);
console.log('pick layer check passed (synthetic: RLE, ties, misses, unprojection, card merge)');

// ---------------- --mvp ROOT REPORT: a recorded pick layer against the outlines it was painted from (edge pixels aside)
const args=process.argv.slice(2),mvp=args.indexOf('--mvp');
if(mvp>=0){
  const [mroot,mreport]=args.slice(mvp+1,mvp+3),dir=path.join(mroot,'reports',mreport,'patches');
  const L=latest(fs.readdirSync(dir).sort().map(n=>JSON.parse(fs.readFileSync(path.join(dir,n),'utf8'))));
  const b=(r:any)=>fs.readFileSync(path.join(mroot,'blobs/sha256',r.sha256));
  let p:any;
  if(L.pick.data.chunks){p=emptyPick(L.pick.data);for(const c of pickChunks(L.pick.data,L.pick.blobs))fillChunk(p,c,await gunzip(b(L.pick.blobs[c.pick])),c.depth?await gunzip(b(L.pick.blobs[c.depth])):null);}
  else p=readPick(L.pick.data,await gunzip(b(L.pick.blobs.pick)),L.pick.blobs.depth?await gunzip(b(L.pick.blobs.depth)):null);
  const analysis=JSON.parse(b(L.outlines.blobs.analysis).toString());let seed=7,agree=0,n=0;const rnd=()=>(seed=(seed*16807)%2147483647)/2147483647;
  for(let i=0;i<4000;i++){
    const f=analysis.frames[Math.floor(rnd()*analysis.frames.length)],x=rnd()*1280,y=rnd()*720,id=pickAt(p,f.timeSec+.25*(f.endTimeSec-f.timeSec),x,y).id;  // nearer its own keyframe than the next
    if(id?.startsWith('person:'))continue;
    const inside=f.objects.filter((o:any)=>o.polygons.some((q:number[][])=>q.length>2&&pointInPolygon([x,y],q))).map((o:any)=>o.entityId);
    n++;if(id?inside.includes(id):!inside.length)agree++;
  }
  assert.ok(agree/n>.95,`pick vs outlines: ${agree}/${n}`);
  console.log(`recorded pick layer check passed: ${mreport}, ${p.data.frames.length} frames, pick vs outline polygons ${(100*agree/n).toFixed(1)}% of ${n} points`);
}
import {sceneRepresentationTasks,readPacked,readGLB} from '../src/viewer/native-viewer.ts';
import {splatAnnotation} from '../src/viewer/splat-layer.ts';
import {currentCameras,cameraPath} from '../src/core.ts';

const [root='/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/fb-c-fixture-001',report='fb-fixture-me340-e9-003']=args.slice(0,mvp<0?2:mvp);
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
