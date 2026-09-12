import assert from 'node:assert/strict';
import {transformMatrix,point,sourceCamera,cameraMatrix,projected,fitCamera,boundsCorners} from '../src/viewer/native-math.ts';
import {readPacked,readGLB} from '../src/viewer/native-viewer.ts';

// Non-square image, skew, rotated camera and pixel centres must agree with K.
const C=[[0,0,1,3],[0,1,0,1],[-1,0,0,2],[0,0,0,1]],K=[[610,17,233.2],[0,602,381.9],[0,0,1]],width=640,height=960;
const frame={cameraToWorld:C,K,width,height};
const camera=sourceCamera(frame,10,[5,1,2]),m=cameraMatrix(camera,width/height,10);
for(const p of [[.3,.8,4],[-.5,-.3,2],[1,1,8]]){
  const world=C.slice(0,3).map(r=>r[0]*p[0]+r[1]*p[1]+r[2]*p[2]+r[3]);
  const pixel=projected(m,world,width,height);
  const expected=[(K[0][0]*p[0]+K[0][1]*p[1])/p[2]+K[0][2]+.5,K[1][1]*p[1]/p[2]+K[1][2]+.5];
  assert.ok(pixel.every((x,i)=>Math.abs(x-expected[i])<.001),`${pixel} != ${expected}`);
}
assert.deepEqual(point(transformMatrix({position:[1,2,3],quaternion:[0,0,Math.SQRT1_2,Math.SQRT1_2],scale:[2,3,4]}),[1,0,0]).map(x=>Math.round(x)),[1,4,3]);
assert.throws(()=>transformMatrix({position:[0,0,0],quaternion:[0,0,0,0],scale:[1,1,1]}));
const buffer=new ArrayBuffer(3*36+12),vertices=new Float32Array(buffer,0,27),indices=new Uint32Array(buffer,108,3);vertices.set([0,0,1,0,0,1,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,0,0,1,0,0,1]);indices.set([0,1,2]);
const meta={byteLayout:{stride:9,vertexCount:3,indexByteOffset:108,indexCount:3,indexType:'uint32'}};
const meshes=readPacked(buffer,meta);assert.equal(meshes[0].vertices.length,33);assert.deepEqual([...meshes[0].indices],[0,1,2]);
indices[2]=3;assert.throws(()=>readPacked(buffer,meta));assert.throws(()=>readGLB(new ArrayBuffer(24)));
for(const aspect of [.4,1,2.5])for(const ortho of [false,true]){
  const corners=boundsCorners({min:[-3,-1,0],max:[4,2,6]}),c=fitCamera(corners,[1,-2,1],[0,0,1],aspect,ortho),matrix=cameraMatrix(c,aspect,10);
  for(const p of corners){const xy=projected(matrix,p,1000*aspect,1000);assert.ok(xy&&xy[0]>0&&xy[0]<1000*aspect&&xy[1]>0&&xy[1]<1000,'camera fit clips scene');}
}
console.log('renderer camera/grid/TRS/packed-asset checks passed');
