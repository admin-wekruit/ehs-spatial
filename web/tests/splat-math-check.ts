// Run: npx tsx tests/splat-math-check.ts
import assert from 'node:assert/strict';
import {splatAnnotation,splatRecord,splatCovariance,projectedCovariance,sortSplats,SPLAT_BYTES} from '../src/viewer/splat-layer.ts';
import {cameraMatrices,cameraMatrix,projected,sourceCamera,lookAround,nearCameras,ceilingCut,dot,type Vec} from '../src/viewer/native-math.ts';
const close=(a:number,b:number,tolerance:number,message:string)=>assert.ok(Math.abs(a-b)<=tolerance,`${message}: ${a} vs ${b}`);

// splat32 records: f32 position, f32 linear scale, u8 sRGB + opacity, u8 quaternion w,x,y,z = round(q*128+128).
function encode(records:{position:Vec;scale:Vec;rgba:number[];q:Vec}[]){
  const view=new DataView(new ArrayBuffer(records.length*SPLAT_BYTES));
  records.forEach(({position,scale,rgba,q},i)=>{const o=i*SPLAT_BYTES;[...position,...scale].forEach((v,k)=>view.setFloat32(o+4*k,v,true));
    rgba.forEach((v,k)=>view.setUint8(o+24+k,v));q.forEach((v,k)=>view.setUint8(o+28+k,Math.max(0,Math.min(255,Math.round(v*128+128)))));});
  return view;
}
const quarterTurnZ=[Math.SQRT1_2,0,0,Math.SQRT1_2];
const view=encode([{position:[1.5,-2,3.25],scale:[.1,.2,.05],rgba:[200,100,50,128],q:quarterTurnZ},{position:[0,0,0],scale:[1,1,1],rgba:[0,0,0,255],q:[1,0,0,0]}]);
const first=splatRecord(view,0);
assert.deepEqual(first.position,[1.5,-2,3.25]);
first.scale.forEach((v,k)=>close(v,[.1,.2,.05][k],1e-7,'scale'));
assert.deepEqual(first.color,[200,100,50]);close(first.opacity,128/255,1e-9,'opacity');
first.quaternion.forEach((v,k)=>close(v,quarterTurnZ[k],1e-9,'8-bit quaternion decodes and normalises'));
assert.deepEqual(splatRecord(view,1).quaternion.map(v=>Math.round(v*1e6)/1e6),[1,0,0,0],'w = 1 saturates at 255 and still decodes to identity');
console.log('PASS: splat32 record decoding (position, linear scale, sRGB + opacity, 8-bit quaternion w,x,y,z)');

// Σ = R S² Rᵀ: a quarter turn about z turns the local y axis (scale 0.2) onto world -x and local x onto world y.
const sigma=splatCovariance(first.scale,first.quaternion);
[[.04,0,0],[0,.01,0],[0,0,.0025]].forEach((row,i)=>row.forEach((v,j)=>close(sigma[i][j],v,1e-7,`sigma[${i}][${j}]`)));
const tilted=splatCovariance([.3,.1,.02],[.9,.2,-.3,.25].map((v,_,q)=>v/Math.hypot(...q)));
[0,1,2].forEach(i=>[0,1,2].forEach(j=>close(tilted[i][j],tilted[j][i],1e-12,'symmetric')));
close(tilted[0][0]+tilted[1][1]+tilted[2][2],.09+.01+.0004,1e-12,'rotation keeps the trace (sum of squared scales)');

// 2D covariance: an isotropic Gaussian of sigma 0.1 straight ahead at depth 5, 60 degree vertical fov on 800x600 pixels,
// is f*sigma/depth pixels wide on both axes, plus the 0.3 px² low-pass dilation.
const free={eye:[0,-5,0],target:[0,0,0],up:[0,0,1],fov:Math.PI/3};
const {view:V,projection:P}=cameraMatrices(free,800/600,1),focal=300/Math.tan(Math.PI/6);
const iso=projectedCovariance([0,0,0],[[.01,0,0],[0,.01,0],[0,0,.01]],V,P,800,600)!;
close(iso[0],(focal*.1/5)**2+.3,1e-6,'xx');close(iso[2],(focal*.1/5)**2+.3,1e-6,'yy');close(iso[1],0,1e-9,'xy');
assert.equal(projectedCovariance([0,-6,0],[[.01,0,0],[0,.01,0],[0,0,.01]],V,P,800,600),null,'behind the camera is culled');
assert.equal(projectedCovariance([40,0,0],[[.01,0,0],[0,.01,0],[0,0,.01]],V,P,800,600),null,'far outside the view is culled');
assert.ok(projectedCovariance([0,500,0],[[.01,0,0],[0,.01,0],[0,0,.01]],V,P,800,600),'beyond the far plane is still drawn (splats write no depth)');

// Off axis and anisotropic, for the free camera and for an exact source-photo camera (skew, off-centre principal
// point, non-square pixels): J Σ Jᵀ must equal the covariance pushed through the actual mesh projection (native-math
// `projected`), differentiated numerically. `projected` is y down; the shader works y up, which flips xy only.
function numeric(camera:any,center:Vec,cov:number[][],w:number,h:number){
  const m=cameraMatrix(camera,w/h,1),e=1e-4,J=[0,1,2].map(k=>{const a=[...center],b=[...center];a[k]-=e;b[k]+=e;const pa=projected(m,a,w,h)!,pb=projected(m,b,w,h)!;return [(pb[0]-pa[0])/(2*e),-(pb[1]-pa[1])/(2*e)];});
  const c=(r:number,s:number)=>[0,1,2].reduce((sum,i)=>sum+[0,1,2].reduce((t,j)=>t+J[i][r]*cov[i][j]*J[j][s],0),0);
  return [c(0,0)+.3,c(0,1),c(1,1)+.3];
}
const frame={cameraToWorld:[[0,0,1,3],[0,1,0,1],[-1,0,0,2],[0,0,0,1]],K:[[610,17,233.2],[0,602,381.9],[0,0,1]],width:640,height:960};
const exact=sourceCamera(frame,10,[5,1,2]);
const ortho={eye:[0,-5,0],target:[0,0,0],up:[0,0,1],orthographic:true,orthoHeight:3};
for(const [camera,center,w,h] of [[free,[.8,.5,-.4],800,600],[exact,[6.5,1.4,2.3],640,960],[ortho,[.7,1.2,-.3],800,600]] as [any,Vec,number,number][]){
  const {view,projection}=cameraMatrices(camera,w/h,1),shader=projectedCovariance(center,tilted,view,projection,w,h)!,expected=numeric(camera,center,tilted,w,h);
  shader.forEach((v,k)=>close(v,expected[k],Math.abs(expected[k])*2e-3+1e-3,(camera.exact?'exact':camera.orthographic?'orthographic':'free')+' camera 2D covariance['+k+']'));
}
console.log('PASS: 3D covariance and EWA 2D covariance (low-pass dilation, culling; free, exact K and orthographic cameras)');

// Back to front by depth in front of the camera, a permutation, within one 16-bit key step (relative, as the key is
// logarithmic); with a stride, the same for every stride-th splat (the even subset drawn while the camera moves).
// Two background splats kilometres away must not coarsen the order of the room (a linear 16-bit key would, to 0.15).
const count=20000,positions=new Float32Array(count*3).map(()=>Math.random()*20-10),z=[.3,-.5,.81],z3=-30;
positions.set([4000,-6000,9000,-3000,5000,-8000],0);
const depth=(i:number)=>-(z[0]*positions[3*i]+z[1]*positions[3*i+1]+z[2]*positions[3*i+2]+z3);
for(const stride of [1,3]){
  const order=new Uint32Array(count),n=sortSplats(positions,count,stride,z[0],z[1],z[2],z3,order,new Float32Array(count),new Uint16Array(count),new Uint32Array(65536)),sorted=order.subarray(0,n);
  assert.equal(n,Math.ceil(count/stride));assert.equal(new Set(sorted).size,n,'every chosen splat exactly once');assert.ok(sorted.every(i=>i%stride===0&&i<count),'only every stride-th');
  for(let i=1;i<n;i++)assert.ok(depth(sorted[i])<=depth(sorted[i-1])*(1+1e-3),'descending depth = far to near, to 0.1% even with far outliers');
}
console.log('PASS: 16-bit counting sort back to front (logarithmic key), whole set and even subset');

const annotation={id:'a',kind:'gaussian_splats',assetId:'asset',coordinateFrameId:'droid_final_native_world',format:'splat32',count:1500000,bounds:{min:[0,0,0],max:[1,1,1]}};
assert.deepEqual(splatAnnotation({annotations:[{id:'v',kind:'video_replay'},annotation]}),{assetId:'asset',count:1500000,coordinateFrameId:'droid_final_native_world',refinedCamerasAssetId:null});
assert.equal(splatAnnotation({annotations:[{...annotation,refinedCamerasAssetId:'cams'}]})!.refinedCamerasAssetId,'cams','the trainer\'s refined cameras, when the report carries them');
assert.equal(splatAnnotation({annotations:[{...annotation,refinedCamerasAssetId:7}]})!.refinedCamerasAssetId,null,'a malformed camera asset id is ignored, not the splats');
for(const broken of [{format:'ply'},{count:0},{count:1.5},{assetId:null},{coordinateFrameId:undefined}])
  assert.equal(splatAnnotation({annotations:[{...annotation,...broken}]}),null,JSON.stringify(broken));
assert.equal(splatAnnotation({annotations:[]}),null);assert.equal(splatAnnotation({}),null);
console.log('PASS: gaussian_splats annotation contract');

// On the walked path a drag turns the view about the eye: drag right looks left, drag down looks up; the eye stays put.
const look={eye:[1,2,3],target:[1,2,2],up:[0,1,0]};lookAround(look,100,0,Math.PI/2/100);
assert.deepEqual(look.eye,[1,2,3]);assert.ok(Math.abs(look.target[0]-0)<1e-9&&Math.abs(look.target[2]-3)<1e-9,'a quarter turn right-drag faces -x (left)');
const tilt={eye:[0,0,0],target:[0,0,-1],up:[0,1,0]};lookAround(tilt,0,50,.004);
assert.ok(tilt.target[1]>0&&Math.abs(dot(tilt.up,[tilt.target[0],tilt.target[1],tilt.target[2]]))<1e-9,'drag down looks up; up stays square to the view');
const walked=[{eye:[0,0,0],forward:[0,0,1]},{eye:[1,0,0],forward:[0,0,1]}];
assert.ok(nearCameras(walked,[1.05,0,0],null,.1)&&!nearCameras(walked,[.5,0,0],null,.1),'near only within the limit');
assert.ok(nearCameras(walked,[0,0,0],[Math.SQRT1_2*.99,0,Math.SQRT1_2],.1)&&!nearCameras(walked,[0,0,0],[0,0,-1],.1),'facing: within 45 degrees of a camera that stood there');
console.log('PASS: on-path look-around and the near-the-path test that keeps splats');

// Dollhouse: from above the walked cameras (median 1.6 above the floor) everything over 2.4 is cut; at camera height nothing is.
const eyes=[[0,0,1.5],[1,0,1.6],[2,0,1.7]];
assert.deepEqual(ceilingCut([0,0,1],0,eyes,[0,0,5]).map(x=>+x.toFixed(9)),[0,0,1,-2.4]);
assert.deepEqual(ceilingCut([0,0,-2],0,eyes,[0,0,5]).map(x=>+x.toFixed(9)),[0,0,1,-2.4],'a downward or unnormalised floor normal is flipped and scaled');
assert.deepEqual(ceilingCut([0,0,1],0,eyes,[0,0,1.6]),[0,0,0,0]);assert.deepEqual(ceilingCut([0,0,1],0,[],[0,0,5]),[0,0,0,0]);
console.log('PASS: dollhouse ceiling cut from above the walked cameras');
