import assert from 'node:assert/strict';
import {transformMatrix,point,sourceCamera,cameraMatrix,projected,fitCamera,boundsCorners} from '../src/viewer/native-math.ts';
import {readPacked,readGLB,representationPass,selectionGeometry} from '../src/viewer/native-viewer.ts';

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
const pose={coordinateFrameId:'native',position:[2,3,4],quaternion:[0,0,0,1],scale:[1,1,1]},rep={id:'observed',kind:'observed_surface',coordinateFrameId:'native',transform:pose,placementState:'confirmed'},e={id:'button',representations:[rep]},layers={point_cloud:true,observed_surface:false,generated_mesh:false,primitive:false,showCandidates:true,editable:false};
const loaded=[{representation:rep,mesh:meshes[0]}],geometry=selectionGeometry(e,'native',layers,loaded);
assert.equal(geometry.corners.length,8,'Selecting an object in the tree/photo retains its bounds while its mesh layer is off');
assert.deepEqual(geometry.transform,pose,'XYZ uses the verified observed transform');
assert.equal(geometry.axisSpace,'native','Observed mesh storage rotation does not establish object structural axes');
assert.equal(geometry.editable,false,'Observation axes cannot mutate measured evidence');
assert.deepEqual(representationPass(e,rep,'native',layers),{available:true,visible:false,pick:true,selectable:true},'Cloud picking uses the exact observed triangles, not a synthetic box');
const modelRep={...rep,id:'model',kind:'generated_mesh'},modelEntity={...e,representations:[modelRep],currentModelTransform:pose};
assert.equal(representationPass(modelEntity,modelRep,'native',layers).pick,false,'Invisible generated geometry never draws into the pick/depth pass');
assert.equal(selectionGeometry(modelEntity,'native',layers,[{...loaded[0],representation:modelRep}]).corners.length,8,'Model-only entities retain bounds on external selection, without pretending hidden models are clickable');
const modelPose={...pose,position:[50,50,50]},both={...e,representations:[modelRep,rep],currentModelTransform:modelPose};
assert.deepEqual(selectionGeometry(both,'native',layers,[...loaded,{...loaded[0],representation:modelRep}]).transform,pose,'Cloud selection bounds use observed geometry, not a hidden model pose');
assert.equal(representationPass({...modelEntity,currentModelTransform:{...pose,coordinateFrameId:'other'}},modelRep,'native',{...layers,generated_mesh:true}).pick,false,'A model pose in another frame cannot be picked or drawn in this scene');
const context={...e,sourceContext:true},cloudRep={...rep,kind:'point_cloud'};
assert.deepEqual(representationPass(context,cloudRep,'native',layers),{available:true,visible:true,pick:true,selectable:false},'Visible context contributes real occlusion with ID zero, never a business entity ID');
assert.equal(selectionGeometry(context,'native',layers,loaded).corners.length,0,'All-object bounds exclude the giant background context');
assert.equal(representationPass(e,{...rep,coordinateFrameId:'unregistered'},'native',layers).pick,false);
assert.equal(representationPass({...e,visible:false},rep,'native',layers).pick,false);
assert.equal(representationPass(e,{...rep,placementState:'unconfirmed',placementReason:'no_depth'},'native',layers).pick,false);
const measured={id:'measured',representations:[],measurements:{coordinateFrameId:'native',basis:{cornersNative:boundsCorners({min:[1,2,3],max:[2,4,6]})}}};
assert.equal(selectionGeometry(measured,'native',layers,[]).corners.length,8,'Verified observed corners remain visible without a mesh');
assert.equal(selectionGeometry(measured,'native',layers,[]).transform,undefined,'No local structure pose is invented for a measured native range');
assert.equal(selectionGeometry(measured,'other',layers,[]).corners.length,0);
assert.equal(selectionGeometry({id:'unknown',measurements:{dimensionsNative:[1,2,3]}},'native',layers,[]).corners.length,0,'Dimensions alone never create a located box');
console.log('renderer camera/grid/TRS/assets, layer-independent selection, exact observed pick policy and context exclusion checks passed');
