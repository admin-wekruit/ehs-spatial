// Page driven by tests/splat-render-check.mjs: the production viewer with a synthetic report (a room wall that is
// scene context and one object model) and splat32 files served by the check; ?scene=points is a report of point grids.
import {mountSceneViewer} from '../src/viewer/native-viewer.ts';
import {cameraMatrix,cameraMatrices,projected,sourceCamera,type Camera,type Vec} from '../src/viewer/native-math.ts';
import {projectedCovariance,splatCovariance} from '../src/viewer/splat-layer.ts';

const params=new URLSearchParams(location.search),host=document.getElementById('scene')!;
host.style.width=(params.get('w')||'640')+'px';host.style.height=(params.get('h')||'480')+'px';
const frame=params.get('frame')||'f',identity={coordinateFrameId:frame,position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]};
function quad(id:string,corners:Vec[],color:Vec){
  const buffer=new ArrayBuffer(4*36+6*4),vertices=new Float32Array(buffer,0,36);
  corners.forEach((p,i)=>vertices.set([...p,0,-1,0,...color],i*9));new Uint32Array(buffer,144,6).set([0,1,2,0,2,3]);
  return {id,url:URL.createObjectURL(new Blob([buffer])),format:'panoptes-mesh-v1',byteLayout:{stride:9,vertexCount:4,indexByteOffset:144,indexCount:6,indexType:'uint32'}};
}
const rep=(id:string,kind:string,assetId:string,bounds:{min:Vec;max:Vec})=>({id,kind,assetId,coordinateFrameId:frame,transform:identity,placementState:'confirmed',bounds});
const kind=params.get('scene')||'splats',synthetic=kind==='splats'||kind==='objects';
// A GLB of coloured points (mode 0, POSITION + COLOR_0), as the dense colour point maps are.
function points(id:string,positions:Vec[],color:Vec){
  const n=positions.length,bin=new Float32Array(n*6);positions.forEach((q,i)=>{bin.set(q,i*3);bin.set(color,n*3+i*3);});
  const spec={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0}],meshes:[{primitives:[{attributes:{POSITION:0,COLOR_0:1},mode:0}]}],buffers:[{byteLength:bin.byteLength}],
    bufferViews:[{buffer:0,byteOffset:0,byteLength:n*12},{buffer:0,byteOffset:n*12,byteLength:n*12}],accessors:[{bufferView:0,componentType:5126,count:n,type:'VEC3'},{bufferView:1,componentType:5126,count:n,type:'VEC3'}]};
  const json=new TextEncoder().encode(JSON.stringify(spec)),size=Math.ceil(json.length/4)*4,out=new ArrayBuffer(28+size+bin.byteLength),view=new DataView(out),bytes=new Uint8Array(out);
  [0x46546c67,2,out.byteLength,size,0x4e4f534a].forEach((v,i)=>view.setUint32(i*4,v,true));bytes.fill(32,20,20+size);bytes.set(json,20);
  view.setUint32(20+size,bin.byteLength,true);view.setUint32(24+size,0x004e4942,true);bytes.set(new Uint8Array(bin.buffer),28+size);
  return {id,url:URL.createObjectURL(new Blob([out])),format:'glb'};
}
// Point grids 4 units in front of the camera at y=0, one point every 0.04 units: A declares a world size of 0.07 (points
// overlap), B declares none (today's fixed size). Context points of size 0.06: C far away, D close to the camera.
const cell=.04,grid=(x0:number)=>Array.from({length:21*21},(_,i)=>[x0+(i%21)*cell,0,-.4+Math.floor(i/21)*cell]);
const pointsScene={captureId:'points-check',cameras:[{id:'cam',imageId:'img',coordinateFrameId:frame,width:640,height:480,K:[[415.7,0,319.5],[0,415.7,239.5],[0,0,1]],cameraToWorld:[[1,0,0,0],[0,0,1,-4],[0,-1,0,0],[0,0,0,1]]}],
  geometryBindings:{img:{cameraId:'cam'}},observations:[{id:'obsA',imageId:'img',revision:1},{id:'obsB',imageId:'img',revision:1}],annotations:[],
  coordinateFrames:[{id:frame,convention:'opencv',ground:{normal:[0,0,1]},scale:{status:'uncalibrated'}}],
  assets:[points('gridA',grid(-1),[1,0,0]),points('gridB',grid(.2),[0,1,0]),points('far',[[0,56,12]],[1,1,1]),points('near',[[0,-3.5,-.15]],[0,0,1])],
  entities:[
    {id:'gridA',label:'grid A',observationRefs:['obsA'],representations:[{...rep('gridA-points','point_cloud','gridA',{min:[-1,0,-.4],max:[-.2,0,.4]}),pointSizeNative:.07,sourceRefs:[{imageId:'img',observationId:'obsA'}]}]},
    {id:'gridB',label:'grid B',observationRefs:['obsB'],representations:[{...rep('gridB-points','point_cloud','gridB',{min:[.2,0,-.4],max:[1,0,.4]}),sourceRefs:[{imageId:'img',observationId:'obsB'}]}]},
    {id:'room',label:'room',sourceContext:true,observationRefs:[],representations:[{...rep('far-points','point_cloud','far',{min:[0,56,12],max:[0,56,12]}),pointSizeNative:.06},{...rep('near-points','point_cloud','near',{min:[0,-3.5,-.15],max:[0,-3.5,-.15]}),pointSizeNative:.06}]},
  ]};
// A grey wall at y=2 (the room surface: sourceContext) behind three blobs at y=1.5, and the crate model: a small quad
// at y=1.8, behind the red blob, whose recorded bounds also cover the wall around it. Two more models: a magenta one
// behind the wall (y=2.3), which the wall must hide, and a yellow poster lying exactly on the wall (y=2).
const scene:any=kind==='points'?pointsScene:{captureId:'splat-check',cameras:[],observations:[],annotations:[],
  coordinateFrames:[{id:frame,convention:'opencv',ground:{normal:[0,0,1]},scale:{status:'uncalibrated'}}],
  assets:synthetic?[quad('wall',[[-3,2,-1],[3,2,-1],[3,2,3],[-3,2,3]],[.55,.55,.55]),quad('crate',[[-1.05,1.8,.95],[-.95,1.8,.95],[-.95,1.8,1.05],[-1.05,1.8,1.05]],[.95,.95,.95]),
    quad('hidden',[[1.6,2.3,1.6],[1.9,2.3,1.6],[1.9,2.3,1.9],[1.6,2.3,1.9]],[1,0,1]),quad('poster',[[-2.4,2,1.9],[-2.1,2,1.9],[-2.1,2,2.2],[-2.4,2,2.2]],[1,1,0])]:[],
  entities:synthetic?[
    {id:'room',label:'room',sourceContext:true,observationRefs:[],representations:[rep('room-surface','observed_surface','wall',{min:[-3,2,-1],max:[3,2,3]})]},
    {id:'crate',label:'crate',observationRefs:[],activeModelRepresentationId:'crate-model',currentModelTransform:identity,representations:[rep('crate-model','generated_mesh','crate',{min:[-1.3,1.75,.7],max:[-.7,2.05,1.3]})]},
    ...['hidden','poster'].map(id=>({id,label:id,observationRefs:[],activeModelRepresentationId:id+'-model',currentModelTransform:identity,representations:[rep(id+'-model','generated_mesh',id,id==='hidden'?{min:[1.6,2.3,1.6],max:[1.9,2.3,1.9]}:{min:[-2.4,2,1.9],max:[-2.1,2,2.2]})]})),
  ]:[]};

// ?scene=objects adds what a report draws per object: an orange observed surface (mug) seen in photo img, and a cyan
// moving object's surface (walker, shown from video time 0 to 100), with the camera that binds img to the frame.
if(kind==='objects'){
  Object.assign(scene,{cameras:[{id:'cam',imageId:'img',coordinateFrameId:frame,width:640,height:480,K:[[415.7,0,319.5],[0,415.7,239.5],[0,0,1]],cameraToWorld:[[1,0,0,0],[0,0,1,-4],[0,-1,0,1],[0,0,0,1]]}],
    geometryBindings:{img:{cameraId:'cam'}},observations:[{id:'obsM',imageId:'img',revision:1},{id:'obsW',imageId:'img',revision:1}]});
  scene.assets.push(quad('mug',[[1.9,1.6,.2],[2.3,1.6,.2],[2.3,1.6,.6],[1.9,1.6,.6]],[1,.5,0]),quad('walker',[[-2.3,1.6,.2],[-1.9,1.6,.2],[-1.9,1.6,.6],[-2.3,1.6,.6]],[0,1,1]));
  scene.entities.push({id:'mug',label:'mug',observationRefs:['obsM'],representations:[{...rep('mug-surface','observed_surface','mug',{min:[1.9,1.6,.2],max:[2.3,1.6,.6]}),sourceRefs:[{imageId:'img',observationId:'obsM'}]}]},
    {id:'walker',label:'walker',motion:'dynamic',observationRefs:['obsW'],representations:[{...rep('walker-at-0','observed_surface','walker',{min:[-2.3,1.6,.2],max:[-1.9,1.6,.6]}),timeRange:[0,100],sourceRefs:[{imageId:'img',observationId:'obsW'}]}]});
}

// GPU time of each splat render (the viewer's only instanced draw), when the check asks for it.
const gpuMs:number[]=[],pending:WebGLQuery[]=[];let timed:WebGL2RenderingContext|null=null;
function pollGpu(){
  const gl=timed,ext=gl?.getExtension('EXT_disjoint_timer_query_webgl2');if(!gl||!ext)return;
  while(pending.length&&gl.getQueryParameter(pending[0],gl.QUERY_RESULT_AVAILABLE)){const q=pending.shift()!;if(!gl.getParameter(ext.GPU_DISJOINT_EXT))gpuMs.push(gl.getQueryParameter(q,gl.QUERY_RESULT)/1e6);gl.deleteQuery(q);}
}
if(params.has('timer')){
  const instanced=WebGL2RenderingContext.prototype.drawArraysInstanced;
  WebGL2RenderingContext.prototype.drawArraysInstanced=function(mode:number,first:number,count:number,instances:number){
    const ext=this.getExtension('EXT_disjoint_timer_query_webgl2');if(!ext)return instanced.call(this,mode,first,count,instances);
    timed=this;pollGpu();const q=this.createQuery()!;this.beginQuery(ext.TIME_ELAPSED_EXT,q);instanced.call(this,mode,first,count,instances);this.endQuery(ext.TIME_ELAPSED_EXT);pending.push(q);
  };
}

// ?leak=1 counts live GL objects, to show a failed splat layer frees what it made.
const live:Record<string,number>={};
if(params.has('leak'))for(const kind of ['Program','Shader','Buffer','Texture','VertexArray','Framebuffer']){
  const proto=WebGL2RenderingContext.prototype as any,create=proto['create'+kind],remove=proto['delete'+kind];live[kind]=0;
  proto['create'+kind]=function(...args:any[]){const made=create.apply(this,args);if(made)live[kind]++;return made;};
  proto['delete'+kind]=function(object:any){if(object)live[kind]--;return remove.call(this,object);};
}
const events:any[]=[];
const viewer=mountSceneViewer(host,{resolveAsset:async id=>scene.assets.find((a:any)=>a.id===id).url,onEvent:event=>events.push(event),showSourcePhoto:params.has('photo'),
  layers:kind==='points'?{observed_surface:false,generated_mesh:false,primitive:false,point_cloud:true,modelOnly:false,cameraPath:false,showCandidates:true,imageId:'img',observations:scene.observations}
    :{observed_surface:true,generated_mesh:true,primitive:true,point_cloud:false,modelOnly:false,cameraPath:false,showCandidates:true,splats:true,
      ...(kind==='objects'?{imageId:'img',observations:scene.observations,time:1}:{})}});
let camera:Camera={eye:[0,-4,1],target:[0,0,1],up:[0,0,1],fov:Math.PI/3};
const canvas=()=>host.querySelector('canvas')!,sleep=(ms:number)=>new Promise(r=>setTimeout(r,ms));
const frames=async(n=2)=>{for(let i=0;i<n;i++)await new Promise(r=>requestAnimationFrame(r));};
function at(p:Vec){
  const rect=canvas().getBoundingClientRect(),q=projected(cameraMatrix(camera,rect.width/rect.height,1),p,rect.width,rect.height);
  return q&&{x:q[0],y:q[1],clientX:rect.left+q[0],clientY:rect.top+q[1]};
}
// The canvas as the page shows it, pixels at the projected points (device pixels) and a hash of every pixel.
function snapshot(points:Record<string,Vec>={},css:Record<string,number[]>={}){
  const c=canvas(),copy=document.createElement('canvas');copy.width=c.width;copy.height=c.height;
  const context=copy.getContext('2d',{willReadFrequently:true})!;context.drawImage(c,0,0);
  const image=context.getImageData(0,0,c.width,c.height),s=c.width/c.getBoundingClientRect().width,samples:Record<string,any>={};
  const read=(x:number,y:number)=>{const i=(Math.floor(y*s)*c.width+Math.floor(x*s))*4;return Array.from(image.data.slice(i,i+4));};
  for(const [name,p] of Object.entries(points)){const q=at(p);samples[name]=q&&{...q,rgba:read(q.x,q.y)};}
  for(const [name,[x,y]] of Object.entries(css))samples[name]={x,y,rgba:read(x,y)};
  let hash=2166136261;for(let i=0;i<image.data.length;i++)hash=Math.imul(hash^image.data[i],16777619);
  return {width:c.width,height:c.height,cssWidth:c.getBoundingClientRect().width,hash:hash>>>0,samples};
}
// Where one splat's projected 3-sigma ellipse should reach, by the JS mirror of the shader (checked in splat-math-check):
// points `sigmas` standard deviations out along its major and minor axes, in CSS pixels.
function ellipse(center:Vec,scale:Vec,q:Vec,sigmas:number){
  const c=canvas(),rect=c.getBoundingClientRect(),dpr=c.width/rect.width,{view,projection}=cameraMatrices(camera,rect.width/rect.height,1);
  const [xx,xy,yy]=projectedCovariance(center,splatCovariance(scale,q),view,projection,c.width,c.height)!,mid=(xx+yy)/2,r=Math.hypot((xx-yy)/2,xy),l1=mid+r,n=Math.hypot(xy,l1-xx);
  const e=n>0?[xy/n,(l1-xx)/n]:[1,0],o=at(center)!,d=sigmas*Math.sqrt(l1)/dpr;  // covariance y is up, the screen's is down
  return {major:[o.x+e[0]*d,o.y-e[1]*d],majorOther:[o.x-e[0]*d,o.y+e[1]*d],minor:[o.x-e[1]*d,o.y-e[0]*d],minorOther:[o.x+e[1]*d,o.y+e[0]*d],sigmaMajor:Math.sqrt(l1)/dpr,sigmaMinor:Math.sqrt(mid-r)/dpr};
}
// Where to look in the points report, in CSS pixels: between A's and B's points, on B's points, the far point, and the
// near point with offsets in device pixels (its 16 px clamp is a device-pixel radius of 8).
function pointProbes(){
  const dpr=canvas().width/canvas().getBoundingClientRect().width,css:Record<string,number[]>={},near=at([0,-3.5,-.15])!,far=at([0,56,12])!;
  for(const [i,j] of [[3,4],[10,10],[16,7]]){
    const a=at([-1+(i+.5)*cell,0,-.4+(j+.5)*cell])!,b=at([.2+(i+.5)*cell,0,-.4+(j+.5)*cell])!,bp=at([.2+i*cell,0,-.4+j*cell])!;
    css[`aBetween${i}`]=[a.x,a.y];css[`bBetween${i}`]=[b.x,b.y];css[`bPoint${i}`]=[bp.x,bp.y];
  }
  for(const [name,dx,dy] of [['nearCentre',0,0],['nearInside',6.5,0],['nearInsideDown',0,6.5],['nearOutside',10,0],['nearDiagonal',6.5,6.5]] as [string,number,number][])css[name]=[near.x+dx/dpr,near.y+dy/dpr];
  for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)css[`far${dx}${dy}`]=[far.x+dx/dpr,far.y+dy/dpr];
  return {css,dpr};
}
async function settle(){await sleep(250);await frames();}  // one resort (at most every 50 ms) and a repaint
async function loadSplats(url:string,count:number,key=url,probe?:Vec){
  viewer.setSplats({key,url,count,coordinateFrameId:frame});
  const started=performance.now(),log:any[]=[];let partial:any=null;
  for(;;){
    const stats=viewer.splatStats();if(!stats)throw Error('no_splat_layer');
    const failure=events.find(e=>e.type==='loadError');if(failure)throw Error(failure.code);
    log.push({ms:Math.round(performance.now()-started),loaded:stats.loaded,drawn:stats.drawn});
    if(probe&&!partial&&stats.drawn>0&&stats.loaded<stats.total){await frames();const now=viewer.splatStats()!;partial={...snapshot({probe}).samples.probe,drawn:now.drawn,loaded:now.loaded,total:now.total};}
    if(stats.total&&stats.loaded===stats.total&&stats.drawn===stats.total)break;
    if(performance.now()-started>120000)throw Error('splat_load_timeout '+JSON.stringify(stats));
    await frames(1);
  }
  await settle();return {ms:performance.now()-started,log,partial,stats:viewer.splatStats()};
}
async function setCamera(value:any){camera=value.cameraToWorld?sourceCamera({id:'exact',...value},10,[0,1.5,1]):value;viewer.setCamera(camera);await settle();}
// Where K itself puts a world point, in canvas CSS pixels (pixel centres at +0.5, as the viewer's exact camera).
function kPixel(p:Vec){
  const f=(camera as any).frame,C=f.cameraToWorld,K=f.K,d=[0,1,2].map(k=>p[k]-C[k][3]),X=[0,1,2].map(j=>C[0][j]*d[0]+C[1][j]*d[1]+C[2][j]*d[2]),s=canvas().getBoundingClientRect().width/f.width;
  return [((K[0][0]*X[0]+K[0][1]*X[1])/X[2]+K[0][2]+.5)*s,(K[1][1]*X[1]/X[2]+K[1][2]+.5)*s];
}
// Turn the camera about its target one step per frame; returns frame intervals, worker sort times, GPU splat times while
// moving, and the GPU time of the full render once the camera rests.
async function orbit(steps:number){
  const c=camera as any,[tx,ty]=c.target,dx=c.eye[0]-tx,dy=c.eye[1]-ty,intervals:number[]=[],sorts:number[]=[];let last=performance.now(),lastSort=NaN;gpuMs.length=0;
  for(let i=1;i<=steps;i++){
    const a=i*2*Math.PI/steps;camera={...c,eye:[tx+dx*Math.cos(a)-dy*Math.sin(a),ty+dx*Math.sin(a)+dy*Math.cos(a),c.eye[2]]};viewer.setCamera(camera);
    await frames(1);const now=performance.now();intervals.push(now-last);last=now;
    const s=viewer.splatStats()!.sortMs;if(s!==lastSort){sorts.push(s);lastSort=s;}
  }
  pollGpu();const moving=gpuMs.slice();await settle();await sleep(300);await frames(3);pollGpu();
  return {intervals,sorts,gpuMs:moving,restMs:gpuMs.slice(moving.length),drawn:viewer.splatStats()!.drawn};
}
// The report's follow-video mode hands the viewer stored video cameras and returns to them after the user has dragged:
// the viewer must keep its own copy, so navigation never rewrites the caller's camera object.
let held:any=null,heldJSON='';
async function holdCamera(){held={eye:[0,-4,1],target:[0,0,1],up:[0,0,1],fov:Math.PI/3};heldJSON=JSON.stringify(held);camera=held;viewer.setCamera(held);await settle();return snapshot().hash;}
async function returnToHeld(){await frames();const untouched=JSON.stringify(held)===heldJSON,dragged=snapshot().hash;camera=held;viewer.setCamera(held);await settle();return {untouched,dragged,hash:snapshot().hash};}
// Failure paths: a stale GL error from other code, and a layer whose worker cannot start (it must free its GL objects).
function staleError(){const gl=canvas().getContext('webgl2')!;gl.bindTexture(0x1234,null);}  // INVALID_ENUM, left unread
async function leakCheck(url:string,count:number){
  viewer.setSplats(null);await frames();const before={...live},Real=window.Worker;
  (window as any).Worker=class{constructor(){throw Error('worker_blocked');}};
  try{viewer.setSplats({key:'leak',url,count,coordinateFrameId:frame});}finally{(window as any).Worker=Real;}
  await frames();return {before,after:{...live},failed:events.some(e=>e.type==='loadError'&&e.code==='splat_load_failed')};
}
const lastSelection=()=>{const all=events.filter(e=>e.type==='selectionIntent');return {count:all.length,entityId:all.at(-1)?.entityId};};
Object.assign(window,{check:{mount:async()=>{await viewer.setScene({id:'r1',document:scene});await setCamera(camera);},loadSplats,setCamera,
  setLayers:async(value:any)=>{viewer.setLayers(value);await settle();},snapshot,at,kPixel,ellipse,pointProbes,holdCamera,returnToHeld,staleError,leakCheck,orbit,
  startSplats:(url:string,count:number,key:string)=>viewer.setSplats({key,url,count,coordinateFrameId:frame}),
  select:async(id:string|null)=>{viewer.setSelection({entityId:id});await settle();},lastSelection,stats:()=>viewer.splatStats(),
  errors:()=>events.filter(e=>e.type==='loadError')}});
