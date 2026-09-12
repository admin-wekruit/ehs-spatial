// node tests/test_viewer_surface.mjs [runs/user-bor1-02/surface]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
const code=fs.readFileSync(new URL('../ehs_spatial/surface_viewer.js',import.meta.url),'utf8');
const ctx=vm.createContext({TextDecoder,Uint32Array,Uint8Array,Float32Array,DataView});vm.runInContext(code,ctx);
const read=expression=>vm.runInContext(expression,ctx);
ctx.testPositions=new Float32Array([1,2,3, 4,5,6, -2,0,8, 100,100,100]);
ctx.testIds=new Float32Array([1,1,3,0]);
const boxes=read('surfaceObjectBounds(testPositions,testIds)');
assert.deepEqual(JSON.parse(JSON.stringify([...boxes])),[[0,{min:[1,2,3],max:[4,5,6]}],[2,{min:[-2,0,8],max:[-2,0,8]}]]);
assert.deepEqual(Array.from(ctx.testPositions),[1,2,3,4,5,6,-2,0,8,100,100,100],'bounds do not transform native surface geometry');
function project(M,p){const q=[0,1,2,3].map(i=>M[i]*p[0]+M[4+i]*p[1]+M[8+i]*p[2]+M[12+i]);return q.map(v=>v/q[3]);}
const camera={width:392,height:518,K:[[380,0,187],[0,381,250],[0,0,1]],
  camera_to_world:[[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]]};
ctx.frame=camera;
const M=read('surfaceMatrix(surfaceCamera(frame,3),392/518,3)');
// OpenCV pixel coordinate including the half-pixel centre must match the GL viewport.
for(const [x,y,z] of [[.2,.3,2],[-.8,-.5,3],[0,0,1]]){
  const projected=project(M,[1+x,2+y,3+z]);
  const px=(projected[0]+1)*392/2-.5,py=(1-projected[1])*518/2-.5;
  assert(Math.abs(px-(380*x/z+187))<.001);assert(Math.abs(py-(381*y/z+250))<.001);
}
if(process.argv[2]){
  const dir=process.argv[2], manifest=JSON.parse(fs.readFileSync(path.join(dir,'surface.json')));
  const array=name=>{const b=fs.readFileSync(path.join(dir,name));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
  ctx.glb=array('surface.glb');ctx.mapping=array('face-inv.bin');ctx.manifest=manifest;
  const mesh=read('readSurfaceGLB(glb,mapping,manifest)');
  assert.equal(mesh.count,manifest.face_count*3);assert.deepEqual([...new Set(mesh.vertexIds)].filter(Boolean).map(i=>i-1).sort((a,b)=>a-b),manifest.supported_inv);
  assert.equal(Buffer.from(mesh.image).subarray(0,8).toString('hex'),'89504e470d0a1a0a');
  // Raw glTF UVs cover source tiles 3/4, without a second V flip.
  assert(mesh.uv.every((v,i)=>i%2===0 || v>=.5));
  for(const frame of manifest.cameras.frames){
    ctx.frame=frame;const matrix=read('surfaceMatrix(surfaceCamera(frame,3),frame.width/frame.height,3)');
    const C=frame.camera_to_world,K=frame.K;
    for(const p of [[0,0,2],[.2,-.3,3]]){
      const world=C.slice(0,3).map(r=>r[0]*p[0]+r[1]*p[1]+r[2]*p[2]+r[3]);
      const q=project(matrix,world),pixel=[(q[0]+1)*frame.width/2-.5,(1-q[1])*frame.height/2-.5];
      assert(Math.abs(pixel[0]-(K[0][0]*p[0]/p[2]+K[0][2]))<.001);
      assert(Math.abs(pixel[1]-(K[1][1]*p[1]/p[2]+K[1][2]))<.001);
    }
  }
  ctx.bad=ctx.mapping.slice(4);assert.throws(()=>read('readSurfaceGLB(glb,bad,manifest)'),/面数量/);
  ctx.bad=ctx.mapping.slice(0);new Uint32Array(ctx.bad)[0]=999;assert.throws(()=>read('readSurfaceGLB(glb,bad,manifest)'),/未验证/);
  console.log(JSON.stringify({status:'passed',faces:mesh.count/3,supported_inv:manifest.supported_inv,checks:'actual GLB, face IDs, UV, native cameras, truncated/stale maps'}));
}else console.log(JSON.stringify({status:'passed',checks:'OpenCV native source pixel-centre projection'}));
