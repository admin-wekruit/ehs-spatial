// Run: node --experimental-strip-types web/experiments/room-preview/check.ts [native-cloud/manifest.json]
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {roomScene} from './room.ts';
import {sceneRepresentationTasks,readGLB} from '../../src/viewer/native-viewer.ts';
const file=process.argv[2];
const frame=(id:string)=>({frame_id:id,canonical_rgb:`${id}.png`,display:{glb:`${id}.glb`},timestamp:1,canonical_size_wh:[336,252],K:[[100,0,167],[0,100,125],[0,0,1]],camera_to_world:[[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]]});
const manifest=file?JSON.parse(readFileSync(file,'utf8')):{version:1,run_id:'test',metric_scale_known:false,display:{glb:'points.glb'},frames:[frame('one'),frame('two')]};
const base=file?pathToFileURL(file).href:'https://example.test/run/native-cloud/manifest.json';
const {scene,urls}=roomScene(manifest,base),world=scene.coordinateFrames[0].id;
assert.equal(scene.entities[0].sourceContext,true);
assert.equal(scene.coordinateFrames[0].scale.status,'uncalibrated');
assert.equal(scene.entities[0].activeModelRepresentationId,undefined);
for(const id of ['cloud','mesh',...manifest.frames.map((f:any)=>`cloud-${f.frame_id}`)]){
  const tasks=sceneRepresentationTasks(scene,world,{showCandidates:true,representationIds:[id]});
  assert.equal(tasks.length,1);assert.equal(tasks[0].r.id,id);
  assert.equal(tasks[0].r.placementState,'unconfirmed');
  assert.equal(sceneRepresentationTasks(scene,world,{showCandidates:false,representationIds:[id]}).length,0);
  if(file){const bytes=readFileSync(fileURLToPath(urls[id]));const meshes=readGLB(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
    assert.ok(meshes.length);assert.ok(meshes.every(m=>m.mode===(id==='mesh'?4:0)));
    if(id==='cloud')assert.equal(meshes.reduce((n,m)=>n+m.indices.length,0),manifest.display.point_count);
  }
}
assert.deepEqual(scene.cameras[0].K,manifest.frames[0].K);
assert.deepEqual(scene.cameras[0].cameraToWorld,manifest.frames[0].camera_to_world);
const bad=structuredClone(manifest);bad.frames[0].K=[];assert.throws(()=>roomScene(bad,base),/invalid_native_frame/);
console.log('Room preview: native points, mesh, per-frame selection, camera domain, and uncertainty checks passed.');
