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
const meshes=readPacked(buffer,meta);assert.equal(meshes[0].vertices.length,36);assert.deepEqual([...meshes[0].indices],[0,1,2]);
indices[2]=3;assert.throws(()=>readPacked(buffer,meta));assert.throws(()=>readGLB(new ArrayBuffer(24)));
// The fourth stored vertex is not used by any drawn triangle. Keep its data,
// but never let it enlarge rendered bounds; non-indexed points do use it.
const positions=new Float32Array([0,0,1,1,0,1,0,1,1,1000,2000,3000]),used=new Uint32Array([0,1,2]);
const indexedPacked=new ArrayBuffer(4*36+12),packedVertices=new Float32Array(indexedPacked,0,36);
for(let i=0;i<4;i++)packedVertices.set([...positions.slice(i*3,i*3+3),0,0,1,1,1,1],i*9);
new Uint32Array(indexedPacked,144,3).set(used);
const boundedPacked=readPacked(indexedPacked,{byteLayout:{stride:9,vertexCount:4,indexByteOffset:144,indexCount:3,indexType:'uint32'}})[0];
assert.deepEqual(boundedPacked.bounds,{min:[0,0,1],max:[1,1,1]},'Packed mesh bounds use only vertices referenced by rendered indices');
assert.equal(boundedPacked.vertices.length,48,'Computing rendered bounds does not discard original vertex data');
function indexedGLB(mode,indexed=true,positionsOverride=positions){
 const positions=positionsOverride;
 const binary=new Uint8Array(positions.byteLength+used.byteLength);binary.set(new Uint8Array(positions.buffer));binary.set(new Uint8Array(used.buffer),positions.byteLength);
 const spec={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0}],meshes:[{primitives:[{attributes:{POSITION:0},mode,...(indexed?{indices:1}:{})}]}],buffers:[{byteLength:binary.length}],bufferViews:[{buffer:0,byteOffset:0,byteLength:positions.byteLength},{buffer:0,byteOffset:positions.byteLength,byteLength:used.byteLength}],accessors:[{bufferView:0,componentType:5126,count:4,type:'VEC3'},{bufferView:1,componentType:5125,count:3,type:'SCALAR'}]};
 const raw=new TextEncoder().encode(JSON.stringify(spec)),jsonSize=Math.ceil(raw.length/4)*4,output=new ArrayBuffer(28+jsonSize+binary.length),view=new DataView(output),bytes=new Uint8Array(output);
 [0x46546c67,2,output.byteLength,jsonSize,0x4e4f534a].forEach((n,i)=>view.setUint32(i*4,n,true));bytes.fill(32,20,20+jsonSize);bytes.set(raw,20);view.setUint32(20+jsonSize,binary.length,true);view.setUint32(24+jsonSize,0x004e4942,true);bytes.set(binary,28+jsonSize);return output;
}
for(const mode of [0,4])assert.deepEqual(readGLB(indexedGLB(mode))[0].bounds,boundedPacked.bounds,'GLB triangle/point index subsets use their actual drawn vertices');
assert.deepEqual(readGLB(indexedGLB(0,false))[0].bounds,{min:[0,0,1],max:[1000,2000,3000]},'A non-indexed point cloud retains all actually drawn points');
for(const scale of [.001,1,1000]){
  const mesh=readGLB(indexedGLB(4,true,positions.map(x=>x*scale)))[0];
  for(let i=0;i<3;i++)assert.deepEqual([...mesh.vertices.slice(i*12+3,i*12+6)],[0,0,1],'Computed GLB normals are unit directions at every mesh scale');
  assert.deepEqual([...mesh.vertices.slice(39,42)],[0,0,0],'Unreferenced vertices retain finite zero normals');
}
const degenerate=readGLB(indexedGLB(4,true,new Float32Array(12)))[0];
assert.ok(degenerate.vertices.every(Number.isFinite),'Degenerate faces never create NaN normals');
for(let i=0;i<4;i++)assert.deepEqual([...degenerate.vertices.slice(i*12+3,i*12+6)],[0,0,0]);
for(const aspect of [.4,1,2.5])for(const ortho of [false,true]){
  const corners=boundsCorners({min:[-3,-1,0],max:[4,2,6]}),c=fitCamera(corners,[1,-2,1],[0,0,1],aspect,ortho),matrix=cameraMatrix(c,aspect,10);
  for(const p of corners){const xy=projected(matrix,p,1000*aspect,1000);assert.ok(xy&&xy[0]>0&&xy[0]<1000*aspect&&xy[1]>0&&xy[1]<1000,'camera fit clips scene');}
}
const observations=[{id:'obs',imageId:'photo'}], scene={observations,coordinateFrames:[]};
const pose={coordinateFrameId:'native',position:[2,3,4],quaternion:[0,0,0,1],scale:[1,1,1]},rep={id:'observed',sourceRefs:[{observationId:'obs'}],kind:'observed_surface',coordinateFrameId:'native',transform:pose,placementState:'confirmed',assetId:'mesh',bounds:meshes[0].bounds},e={id:'button',observationRefs:['obs'],representations:[rep]},layers={imageId:'photo',observations,point_cloud:true,observed_surface:false,generated_mesh:false,primitive:false,showCandidates:true,editable:false};
const geometry=selectionGeometry(scene,e,'native',layers);
assert.equal(geometry.corners.length,8,'Selecting an object in the tree/photo retains its bounds while its mesh layer is off');
assert.deepEqual(geometry.transform,pose,'XYZ uses the verified observed transform');
assert.equal(geometry.axisSpace,'native','Observed mesh storage rotation does not establish object structural axes');
assert.equal(geometry.editable,false,'Observation axes cannot mutate measured evidence');
assert.deepEqual(representationPass(e,rep,'native',layers),{available:true,visible:false,pick:true,selectable:true},'Cloud picking uses the exact observed triangles, not a synthetic box');
const modelRep={...rep,id:'model',kind:'generated_mesh'},modelEntity={...e,activeModelRepresentationId:'model',representations:[modelRep],currentModelTransform:pose};
assert.equal(representationPass(modelEntity,modelRep,'native',layers).pick,false,'Invisible generated geometry never draws into the pick/depth pass');
assert.equal(selectionGeometry(scene,modelEntity,'native',layers).corners.length,0,'A cloud view cannot borrow a model-only location');
const modelPose={...pose,position:[50,50,50]},both={...e,activeModelRepresentationId:'model',representations:[modelRep,rep],currentModelTransform:modelPose};
assert.deepEqual(selectionGeometry(scene,both,'native',layers).transform,pose,'Cloud selection bounds use observed geometry, not a hidden model pose');
assert.equal(representationPass({...modelEntity,currentModelTransform:{...pose,coordinateFrameId:'other'}},modelRep,'native',{...layers,generated_mesh:true}).pick,false,'A model pose in another frame cannot be picked or drawn in this scene');
const context={...e,sourceContext:true},cloudRep={...rep,kind:'point_cloud'};
assert.deepEqual(representationPass(context,cloudRep,'native',layers),{available:true,visible:true,pick:true,selectable:false},'Visible context contributes real occlusion with ID zero, never a business entity ID');
for(const kind of ['observed_surface','point_cloud']){
  const scoped={...rep,kind,sourceRefs:[{imageId:'photo'}]},enabled={...layers,[kind]:true};
  assert.equal(representationPass(context,scoped,'native',enabled).visible,true,'The current photo context remains visible');
  assert.deepEqual(representationPass(context,scoped,'native',{...enabled,imageId:'other-photo'}),{available:false,visible:false,pick:false,selectable:false},'Photo-bound context cannot draw or occlude another photo in the same native frame');
  assert.equal(representationPass(context,scoped,'native',{...enabled,imageId:null}).visible,false,'An explicitly photo-bound context requires the bound photo');
}
assert.equal(representationPass(context,{...cloudRep,sourceRefs:[{assetId:'verified-whole-cloud'}]},'native',{...layers,imageId:'other-photo'}).visible,true,'An unscoped whole-scene cloud is not invented into a per-photo source');
assert.equal(selectionGeometry(scene,context,'native',layers).corners.length,0,'All-object bounds exclude the giant background context');
assert.equal(representationPass(e,{...rep,coordinateFrameId:'unregistered'},'native',layers).pick,false);
assert.equal(representationPass({...e,visible:false},rep,'native',layers).pick,false);
assert.equal(representationPass(e,{...rep,placementState:'unconfirmed',placementReason:'no_depth'},'native',layers).pick,false);
const measured={id:'measured',measurementSelections:{basis:'basis'},measurementEvidence:[{id:'basis',observationRefs:['obs']}],representations:[],measurements:{coordinateFrameId:'native',basis:{cornersNative:boundsCorners({min:[1,2,3],max:[2,4,6]})}}};
assert.equal(selectionGeometry(scene,measured,'native',layers).corners.length,8,'Verified observed corners remain visible without a mesh');
assert.deepEqual(selectionGeometry(scene,measured,'native',layers).transform.position,[0,0,0],'A measured native range uses the native frame, not an invented structural pose');
assert.equal(selectionGeometry(scene,measured,'other',layers).corners.length,0);
assert.equal(selectionGeometry(scene,{id:'unknown',measurements:{dimensionsNative:[1,2,3]}},'native',layers).corners.length,0,'Dimensions alone never create a located box');
const floor={...e,id:'reference',geometryRole:'floor'},floorBefore=structuredClone(floor);
for(const display of [layers,{...layers,observed_surface:true,allBounds:true,editable:true}]){
  const selection=selectionGeometry(scene,floor,'native',display);
  assert.equal(selection.corners.length,0,'Reference surfaces have no equipment volume box in selected/all-bounds modes');
  assert.equal(selection.editable,false,'Reference surfaces never expose a transform gizmo');
  assert.equal(representationPass(floor,rep,'native',display).pick,true,'Reference surface triangles remain selectable in source/3D/cloud views');
}
assert.equal(representationPass(floor,rep,'native',{...layers,observed_surface:true}).visible,true,'Suppressing equipment overlays must not hide the floor mesh');
assert.deepEqual(floor,floorBefore,'Presentation must not rewrite measured geometry or ground evidence');
console.log('renderer camera/grid/TRS/assets, shared layer selection, exact observed pick policy and context exclusion checks passed');
