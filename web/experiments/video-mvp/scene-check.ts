import assert from 'node:assert/strict';
import {validateScene,lineTransform,pointsGLB,replayDocument,applyFrame,replayLayers,bodySample,interpolateBody} from './scene.ts';
import {readGLB,sceneRepresentationTasks} from '../../src/viewer/native-viewer.ts';
import {point,transformMatrix} from '../../src/viewer/native-math.ts';

// Isolated assertions only; this scene is never written into a delivery manifest.
const c2w=[[1,0,0,3],[0,1,0,4],[0,0,1,5],[0,0,0,1]];
const scene={schema:'phase2-replay-scene-v1',coordinate_frame:'test',units:'meters',source_video_sha256:'a'.repeat(64),method:'test',limitations:[],points:[['p',1,2,3]],frames:[{sourceFrame:0,timeSec:0,endTimeSec:.1,c2w,objects:[{entityId:'a',keypoints3d:[[1,2,3],[1,2,4],null],bones:[[0,1],[1,2]],centroid:[1,2,3.5]}]}]};
const sample={video:{durationSec:1,sha256:'a'.repeat(64)}};
const base='http://localhost/scene.json';
assert.equal(validateScene(scene,sample,base),scene);
assert.throws(()=>validateScene({...scene,source_video_sha256:'b'.repeat(64)},sample,base),/SHA256/);
assert.throws(()=>validateScene(scene,{video:{durationSec:1}},base),/SHA256/);
assert.throws(()=>validateScene({...scene,units:'unknown'},sample,base),/单位/);
assert.throws(()=>validateScene({...scene,frames:[scene.frames[0],scene.frames[0]]},sample,base),/时间/);
const sourceObject={entityId:'obs-one',label:'chair',displayName:'chair 1',meshUrl:'chair.glb',provenanceUrl:'source.json',representation:'single_frame_observed_surface',identityScope:'independent_observation',source:{sourceFrame:0,timeSec:0,endTimeSec:.1,width:640,height:480,maskUrl:'mask.png',imageUrl:'source.png',bbox:[1,1,10,10]}};
const withObject={...scene,staticObjects:[sourceObject]},sizedSample={video:{...sample.video,width:640,height:480}};
assert.equal(validateScene(withObject,sizedSample,base),withObject);
const completedObject={...sourceObject,generatedModel:{sourceFrame:0,meshUrl:'generated.glb',sha256:'f'.repeat(64),provenanceUrl:'validation.json',status:'source_consistent_model_estimate'}};
const completedScene={...scene,staticObjects:[completedObject]};
assert.equal(validateScene(completedScene,sizedSample,base),completedScene);
assert.throws(()=>validateScene({...scene,staticObjects:[{...completedObject,generatedModel:{...completedObject.generatedModel,sourceFrame:2}}]},sizedSample,base),/来源一致性/);
assert.throws(()=>validateScene({...scene,staticObjects:[{...completedObject,generatedModel:{...completedObject.generatedModel,status:'failed'}}]},sizedSample,base),/来源一致性/);
assert.equal(replayDocument(completedScene,{'obs-one':'chair.glb'},()=> '#65e2be').document.entities.find(e=>e.id==='obs-one').representations[0].streamed,true);
assert.throws(()=>validateScene({...withObject,staticObjects:[{...sourceObject,source:{...sourceObject.source,timeSec:.01}}]},sizedSample,base),/来源/);
const heldSample={...withObject,frames:[{...scene.frames[0],endTimeSec:.3}]};
assert.equal(validateScene(heldSample,sizedSample,base),heldSample); // Mask keeps its original .1s interval in a held 3D sample.
assert.throws(()=>validateScene({...withObject,staticObjects:[{...sourceObject,source:{...sourceObject.source,endTimeSec:.2}}]},sizedSample,base),/来源/);
const objectDocument=replayDocument(withObject,{'obs-one':'chair.glb'},()=> '#65e2be').document;
const objectEntity=objectDocument.entities.find(e=>e.id==='obs-one');
assert.equal(objectEntity.sourceContext,undefined);assert.equal(objectEntity.representations[0].kind,'generated_mesh');
assert.deepEqual(objectEntity.representations[0].transform.position,[0,0,0]);
assert.deepEqual(sceneRepresentationTasks(objectDocument,'test',{entityIds:['obs-one']}).map(t=>t.e.id),['obs-one']);
const layerScene={...withObject,meshUrl:'mesh.glb'};
const layerDocument=replayDocument(layerScene,{'obs-one':'chair.glb',mesh:'mesh.glb',points:'points.glb'},()=> '#65e2be').document;
const layerIds=(value:any)=>sceneRepresentationTasks(layerDocument,'test',value).map(task=>task.e.id);
assert.ok(!layerIds(replayLayers(layerScene)).includes('points'));
assert.ok(layerIds(replayLayers(layerScene)).includes('mesh'));
assert.ok(layerIds(replayLayers(layerScene)).includes('obs-one'));
assert.ok(layerIds(replayLayers(layerScene,true)).includes('points'));
assert.ok(layerIds(replayLayers(withObject)).includes('points')); // No room mesh: cloud is the default.
assert.deepEqual(layerIds(replayLayers(layerScene,true,true)),['obs-one']);
assert.ok(layerIds(replayLayers(layerScene,true,false)).includes('points'));
const glb=pointsGLB(scene.points),mesh=readGLB(glb.buffer)[0];
assert.equal(mesh.mode,0);assert.deepEqual(Array.from(mesh.vertices.slice(0,3)),[1,2,3]);
const close=(a:number[],b:number[])=>a.forEach((v,k)=>assert.ok(Math.abs(v-b[k])<1e-6));
for(const endpoint of [[0,0,2],[0,0,-2],[1,2,3]]){const t=lineTransform([0,0,0],endpoint,.1,'test')!;close(point(transformMatrix(t),[0,0,-.5]),[0,0,0]);close(point(transformMatrix(t),[0,0,.5]),endpoint);}
assert.equal(lineTransform([1,2,3],[1,2,3],.1,'test'),null);
const {document,dynamic}=replayDocument(scene,{},()=> '#65e2be');
const staticSignature=()=>JSON.stringify(document.entities.map(e=>[e.id,e.activeModelRepresentationId,e.representations.map(r=>[r.id,r.assetId,r.kind,r.primitive])]));
const before=staticSignature();
applyFrame(scene.frames[0],dynamic,'test',.1,null);
assert.equal(dynamic.get('a:bone:0:1').representations[0].material.baseColorFactor[3],1);
assert.equal(dynamic.get('a:bone:1:2').representations[0].material.baseColorFactor[3],0);
const link=dynamic.get('a:bone:0:1');
assert.equal(link.representations[0].primitive.kind,'cylinder');
assert.equal(link.representations[0].primitive.parameters.segments,8);
close(point(transformMatrix(link.currentModelTransform),[0,0,-.5]),[1,2,3]);
close(point(transformMatrix(link.currentModelTransform),[0,0,.5]),[1,2,4]);
assert.ok(link.currentModelTransform.scale[0]<=.02200001);
const camera=dynamic.get('camera:0').currentModelTransform;
close(point(transformMatrix(camera),[0,0,-.5]),[3,4,5]);
applyFrame(null,dynamic,'test',.1,null);
assert.ok([...dynamic.values()].every(e=>e.representations[0].material.baseColorFactor[3]===0));
assert.equal(staticSignature(),before);
console.log('PASS: scene provenance, time intervals, exact point XYZ, bone transforms, missing joints, clear gaps and stable GPU asset signature');

const surfaced=structuredClone(scene);surfaced.frames[0].objects[0].surface={sourceFrame:0,meshUrl:'human.glb',sha256:'c'.repeat(64),representation:'visible_rgbd_surface'};
const monocularSurface=structuredClone(surfaced);monocularSurface.units='uncalibrated_monocular';
assert.throws(()=>validateScene(monocularSurface,sample,base),/深度来源/);
monocularSurface.frames[0].objects[0].surface.representation='visible_monocular_surface';
assert.equal(validateScene(monocularSurface,sample,base),monocularSurface);
assert.equal(validateScene(surfaced,sample,base),surfaced);
surfaced.frames[0].objects[0].surface.sourceFrame=1;
assert.throws(()=>validateScene(surfaced,sample,base),/当前源帧/);
const timeline={bodyInterpolation:[{sourceFrame:1,entityId:'a',status:'accepted_model_estimate',keyframes:[0,2]}],frames:[0,1,2].map(i=>({sourceFrame:i,timeSec:i*.1,objects:[{entityId:'a'}]})),bodyKeyframes:[0,2].map(i=>({sourceFrame:i,timeSec:i*.1,entityId:'a',topology_sha256:'d'.repeat(64),vertices:1}))};
assert.equal(bodySample(timeline,timeline.frames[1],'a').t,.5);
assert.equal(bodySample({...timeline,frames:[timeline.frames[0],timeline.frames[2]]},timeline.frames[1],'a'),null);
const gap=structuredClone(timeline);gap.frames[1].objects=[];assert.equal(bodySample(gap,timeline.frames[1],'a'),null);
const wrongTopology=structuredClone(timeline);wrongTopology.bodyKeyframes[1].topology_sha256='e'.repeat(64);assert.equal(bodySample(wrongTopology,timeline.frames[1],'a'),null);
const nextMesh={...mesh,vertices:mesh.vertices.slice()};nextMesh.vertices[0]+=2;
assert.equal(interpolateBody(mesh,nextMesh,.5).vertices[0],mesh.vertices[0]+1);
assert.throws(()=>interpolateBody(mesh,{...nextMesh,indices:new Uint32Array([2])},.5),/拓扑/);
console.log('PASS: human source-frame binding, missing-observation rejection and same-topology short-gap interpolation');

assert.equal(bodySample({...timeline,bodyInterpolation:[]},timeline.frames[1],'a'),null);
assert.equal(bodySample({...timeline,bodyInterpolation:[{sourceFrame:1,entityId:'a',status:'rejected_alignment'}]},timeline.frames[1],'a'),null);
console.log('PASS: unreviewed or rejected in-between body geometry never renders');
assert.equal(bodySample({...timeline,bodyInterpolation:[{...timeline.bodyInterpolation[0],keyframes:[0,3]}]},timeline.frames[1],'a'),null);
const coloredBody={sourceFrame:0,timeSec:0,entityId:'a',representation:'inferred_anatomical_mesh',status:'accepted_model_estimate',meshUrl:'body.glb',mesh_sha256:'c'.repeat(64),topology_sha256:'d'.repeat(64),vertices:3,sourceColor:{sourceFrame:0,entityId:'a',mesh_sha256:'c'.repeat(64),geometry_changed:false,colored_vertices:2,vertices:3}};
const coloredScene={...scene,bodyKeyframes:[coloredBody],bodyInterpolation:[]};
assert.equal(validateScene(coloredScene,sample,base),coloredScene);
for(const change of [{sourceFrame:1},{entityId:'b'},{mesh_sha256:'f'.repeat(64)},{geometry_changed:true},{colored_vertices:4}]){
  assert.throws(()=>validateScene({...coloredScene,bodyKeyframes:[{...coloredBody,sourceColor:{...coloredBody.sourceColor,...change}}]},sample,base),/人体颜色/);
}
console.log('PASS: body appearance cannot change source frame, identity, geometry or mesh hash');
