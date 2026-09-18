import {mountSceneViewer,readGLB,type Mesh} from '../../src/viewer/native-viewer.ts';
import {add,scale,unit,cross,fitCamera,boundsCorners,point,sourceCamera} from '../../src/viewer/native-math.ts';
import {assetUrl,frameAt} from './timeline.mjs';

type XYZ = number[];
const xyz = (p:any):p is XYZ => Array.isArray(p) && p.length===3 && p.every(Number.isFinite);
const require = (condition:any,message:string) => {if(!condition)throw Error(message);};

export function validateScene(data:any,sample:any,base:string){
  require(data?.schema==='phase2-replay-scene-v1','空间结果 schema 不正确');
  require(typeof data.coordinate_frame==='string'&&data.coordinate_frame.length,'空间结果缺少坐标系');
  require(['meters','uncalibrated_monocular'].includes(data.units),'空间结果必须声明米或未标定单目单位');
  require(/^[a-f0-9]{64}$/.test(data.source_video_sha256),'空间结果缺少来源视频 SHA256');
  require(sample.video.sha256===data.source_video_sha256,'空间结果与当前视频的来源 SHA256 不同或视频来源缺失');
  require(typeof data.method==='string'&&Array.isArray(data.limitations),'空间结果缺少方法与限制');
  require(Array.isArray(data.points)&&data.points.length<=300000,'空间显示点需限制在 300000 以内');
  const ids=new Set();
  for(const p of data.points){require(Array.isArray(p)&&p.length===4&&['string','number'].includes(typeof p[0])&&xyz(p.slice(1))&&!ids.has(p[0]),'空间点必须有唯一 ID 与有限 XYZ');ids.add(p[0]);}
  if(data.meshUrl)assetUrl(data.meshUrl,base);
  if(data.pointCloudUrl){assetUrl(data.pointCloudUrl,base);require(Number.isInteger(data.pointCloudCount)&&data.pointCloudCount>0&&data.pointCloudCount<=300000,'密集点云数量无效');}
  require(Array.isArray(data.frames),'空间结果缺少 frames');
  let previousEnd=0;
  for(const frame of data.frames){
    require(Number.isInteger(frame.sourceFrame)&&frame.sourceFrame>=0,'空间源帧号无效');
    require(Number.isFinite(frame.timeSec)&&Number.isFinite(frame.endTimeSec)&&frame.timeSec>=previousEnd&&frame.endTimeSec>frame.timeSec&&frame.endTimeSec<=sample.video.durationSec+.1,'空间时间区间重叠或超出原视频');
    previousEnd=frame.endTimeSec;
    require(Array.isArray(frame.c2w)&&frame.c2w.length===4&&frame.c2w.every((r:any)=>Array.isArray(r)&&r.length===4&&r.every(Number.isFinite)),'空间相机必须是 4×4 row-major c2w');
    require(frame.c2w[3].every((v:number,k:number)=>Math.abs(v-(k===3?1:0))<1e-5),'空间相机齐次矩阵无效');
    require(Array.isArray(frame.objects),'空间帧缺少对象列表');
    const objects=new Set();
    for(const object of frame.objects){
      require(typeof object.entityId==='string'&&object.entityId.length&&!objects.has(object.entityId),'同帧空间对象 ID 必须唯一');objects.add(object.entityId);
      require(Array.isArray(object.keypoints3d)&&object.keypoints3d.every((p:any)=>p===null||xyz(p)),'3D 关节需为 XYZ 或 null');
      require(Array.isArray(object.bones)&&object.bones.every((edge:any)=>Array.isArray(edge)&&edge.length===2&&edge.every((i:any)=>Number.isInteger(i)&&i>=0&&i<object.keypoints3d.length)),'3D 骨连接索引无效');
      require(object.centroid==null||xyz(object.centroid),'3D 中心点需为 XYZ 或 null');
      if(object.surface){const s=object.surface;assetUrl(s.meshUrl,base);require(s.sourceFrame===frame.sourceFrame&&s.representation===(data.units==='meters'?'visible_rgbd_surface':'visible_monocular_surface')&&/^[a-f0-9]{64}$/.test(s.sha256),'人物表面与当前源帧或深度来源不符');}
      if(object.world_motion!==undefined)require(['insufficient_evidence','below_resolution','observed_displacement'].includes(object.world_motion),'空间运动状态无效');
      if(object.motionEstimate){const m=object.motionEstimate;require(['below_resolution','observed_displacement'].includes(object.world_motion)&&m.status==='model_estimate_not_ground_truth'&&['hips','shoulders'].includes(m.anchor)&&Number.isFinite(m.elapsedSeconds)&&m.elapsedSeconds>0&&Number.isFinite(m.displacementM)&&m.displacementM>=0&&Array.isArray(m.sourceFrames)&&m.sourceFrames.length===2&&m.sourceFrames.every(Number.isInteger)&&m.sourceFrames[0]>=0&&m.sourceFrames[1]>m.sourceFrames[0]&&m.sourceFrames[1]<=frame.sourceFrame,'空间运动估计缺少有效来源');}

    }
  }
  if(data.bodyKeyframes?.length){
    require(Array.isArray(data.bodyInterpolation),'人体插值缺少逐帧检查');
    for(const r of data.bodyInterpolation)require(data.frames.some((f:any)=>f.sourceFrame===r.sourceFrame&&f.objects.some((o:any)=>o.entityId===r.entityId))&&['accepted_model_estimate','rejected_alignment'].includes(r.status),'人体插值检查与源帧不符');
  }
  const objects=new Set();
  for(const object of data.staticObjects||[]){
    require(typeof object.entityId==='string'&&object.entityId.startsWith('obs-')&&!objects.has(object.entityId),'静态表面观测 ID 必须唯一');objects.add(object.entityId);
    require(object.representation==='single_frame_observed_surface'&&object.identityScope==='independent_observation','对象模型必须声明独立可见表面观测');
    require(typeof object.label==='string'&&typeof object.displayName==='string','对象模型缺少名称');
    for(const url of [object.meshUrl,object.provenanceUrl,object.source?.maskUrl,object.source?.imageUrl])assetUrl(url,base);
    const source=object.source,frame=data.frames.find((f:any)=>f.sourceFrame===source.sourceFrame);
    require(frame&&source.timeSec===frame.timeSec&&source.endTimeSec>source.timeSec&&source.endTimeSec<=frame.endTimeSec,'对象来源与空间帧时间不一致');
    require(source.width===sample.video.width&&source.height===sample.video.height,'对象掩码与原视频画幅不一致');
    require(Array.isArray(source.bbox)&&source.bbox.length===4&&source.bbox.every(Number.isFinite)&&source.bbox[2]>source.bbox[0]&&source.bbox[3]>source.bbox[1],'对象掩码来源框无效');
    if(object.semanticReview)require(['clear','partial','incorrect_prompt'].includes(object.semanticReview.status)&&typeof object.semanticReview.description==='string','对象模型核验格式无效');
    if(object.generatedModel){const m=object.generatedModel;assetUrl(m.meshUrl,base);assetUrl(m.provenanceUrl,base);require(m.sourceFrame===source.sourceFrame&&m.status==='source_consistent_model_estimate'&&/^[a-f0-9]{64}$/.test(m.sha256),'生成模型缺少来源一致性检查');}
  }
  for(const body of data.bodyKeyframes||[]){
    const frame=data.frames.find((f:any)=>f.sourceFrame===body.sourceFrame);
    require(frame&&frame.timeSec===body.timeSec&&frame.objects.some((o:any)=>o.entityId===body.entityId),'人体关键帧没有对应的视频对象');
    require(body.representation==='inferred_anatomical_mesh'&&body.status==='accepted_model_estimate'&&/^[a-f0-9]{64}$/.test(body.mesh_sha256)&&/^[a-f0-9]{64}$/.test(body.topology_sha256),'人体网格来源无效');assetUrl(body.meshUrl,base);
    if(body.sourceColor){const c=body.sourceColor;require(c.sourceFrame===body.sourceFrame&&c.entityId===body.entityId&&c.mesh_sha256===body.mesh_sha256&&c.geometry_changed===false&&Number.isInteger(c.colored_vertices)&&c.colored_vertices>=0&&c.colored_vertices<=body.vertices&&c.vertices===body.vertices,'人体颜色与源帧或网格不符');}
  }
  return data;
}

export function bodySample(scene:any,frame:any,id:string){
  if(!frame?.objects.some((o:any)=>o.entityId===id))return null;
  const keys=(scene.bodyKeyframes||[]).filter((b:any)=>b.entityId===id).sort((a:any,b:any)=>a.timeSec-b.timeSec);
  const exact=keys.find((b:any)=>b.sourceFrame===frame.sourceFrame);if(exact)return {a:exact,b:exact,t:0};
  const review=scene.bodyInterpolation?.find((r:any)=>r.entityId===id&&r.sourceFrame===frame.sourceFrame&&r.status==='accepted_model_estimate');if(!review)return null;
  const a=keys.filter((b:any)=>b.timeSec<frame.timeSec).at(-1),b=keys.find((b:any)=>b.timeSec>frame.timeSec);
  if(!a||!b||review.keyframes?.[0]!==a.sourceFrame||review.keyframes?.[1]!==b.sourceFrame||b.timeSec-a.timeSec>.38||a.topology_sha256!==b.topology_sha256||a.vertices!==b.vertices)return null;
  const between=scene.frames.filter((f:any)=>f.sourceFrame>=a.sourceFrame&&f.sourceFrame<=b.sourceFrame);
  if(between.length!==b.sourceFrame-a.sourceFrame+1||between.some((f:any)=>!f.objects.some((o:any)=>o.entityId===id)))return null;
  return {a,b,t:(frame.timeSec-a.timeSec)/(b.timeSec-a.timeSec)};
}

export function interpolateBody(a:Mesh,b:Mesh,t:number):Mesh{
  require(Number.isFinite(t)&&t>=0&&t<=1&&a.vertices.length===b.vertices.length&&a.indices.length===b.indices.length&&a.indices.every((v,k)=>v===b.indices[k]),'人体插值网格拓扑不同');
  if(t===0)return a;
  const vertices=a.vertices.slice(),bounds={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};
  for(let i=0;i<vertices.length;i+=12)for(let k=0;k<6;k++){
    const value=a.vertices[i+k]*(1-t)+b.vertices[i+k]*t;vertices[i+k]=value;
    if(k<3){bounds.min[k]=Math.min(bounds.min[k],value);bounds.max[k]=Math.max(bounds.max[k],value);}
  }
  return {...a,vertices,bounds};
}

export function lineTransform(a:XYZ,b:XYZ,radius:number,coordinateFrameId:string){
  const delta=b.map((v,k)=>v-a[k]),length=Math.hypot(...delta);
  if(!(length>1e-8))return null;
  const [x,y,z]=delta.map(v=>v/length),q=z<-.999999?[1,0,0,0]:[-y,x,0,1+z],norm=Math.hypot(...q);
  return {coordinateFrameId,position:a.map((v,k)=>(v+b[k])/2),quaternion:q.map(v=>v/norm),scale:[radius,radius,length]};
}

// POINTS GLB is an in-memory adapter for the existing GLB reader; source XYZ is unchanged.
export function pointsGLB(points:any[]){
  const positions=new Float32Array(points.flatMap(p=>p.slice(1))),min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
  for(const p of points)for(let k=0;k<3;k++){min[k]=Math.min(min[k],p[k+1]);max[k]=Math.max(max[k],p[k+1]);}
  const doc={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{mesh:0}],meshes:[{primitives:[{attributes:{POSITION:0},mode:0,material:0}]}],materials:[{pbrMetallicRoughness:{baseColorFactor:[.48,.64,.65,1]}}],buffers:[{byteLength:positions.byteLength}],bufferViews:[{buffer:0,byteLength:positions.byteLength}],accessors:[{bufferView:0,componentType:5126,count:points.length,type:'VEC3',min,max}]};
  const json=new TextEncoder().encode(JSON.stringify(doc)),size=(json.length+3)&~3,bytes=new Uint8Array(28+size+positions.byteLength),view=new DataView(bytes.buffer);
  view.setUint32(0,0x46546c67,true);view.setUint32(4,2,true);view.setUint32(8,bytes.length,true);view.setUint32(12,size,true);view.setUint32(16,0x4e4f534a,true);bytes.fill(32,20,20+size);bytes.set(json,20);view.setUint32(20+size,positions.byteLength,true);view.setUint32(24+size,0x004e4942,true);bytes.set(new Uint8Array(positions.buffer),28+size);
  return bytes;
}

export function replayDocument(scene:any,urls:Record<string,string>,color:(id:string)=>string){
  const frameId=scene.coordinate_frame,identity={coordinateFrameId:frameId,position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]};
  const entities:any[]=[],assets=Object.keys(urls).map(id=>({id}));
  for(const id of ['points','mesh'])if(urls[id])entities.push({id,sourceContext:true,representations:[{id,assetId:id,kind:id==='points'?'point_cloud':'observed_surface',coordinateFrameId:frameId,transform:identity,placementState:'confirmed'}]});
  // Native observed_surface is source-photo-only; CPU triangulated models use its selectable mesh path.
  for(const object of scene.staticObjects||[])entities.push({id:object.entityId,activeModelRepresentationId:object.entityId,currentModelTransform:identity,representations:[{id:object.entityId,assetId:object.entityId,kind:'generated_mesh',coordinateFrameId:frameId,transform:identity,placementState:'confirmed',...(object.generatedModel?{streamed:true}: {})}]});
  const dynamic=new Map<string,any>(),owner=new Map<string,string>();
  const addPrimitive=(id:string,tint:string,entityId?:string,kind='cylinder')=>{
    const rgb=tint.match(/[a-f\d]{2}/gi)!.map(v=>parseInt(v,16)/255);
    const representation={id,kind:'primitive',coordinateFrameId:frameId,transform:identity,placementState:'confirmed',primitive:kind==='cylinder'?{kind,parameters:{radius:1,height:1,segments:8}}:{kind,parameters:{dimensions:[1,1,1]}},material:{baseColorFactor:[...rgb,0],alphaMode:'BLEND'}};
    const entity={id,sourceContext:!entityId,currentModelTransform:identity,activeModelRepresentationId:id,representations:[representation]};
    dynamic.set(id,entity);entities.push(entity);if(entityId)owner.set(id,entityId);
  };
  for(let i=0;i<8;i++)addPrimitive(`camera:${i}`,'#ffc56d');
  for(const frame of scene.frames)for(const object of frame.objects){
    for(const [a,b]of object.bones){const id=`${object.entityId}:bone:${a}:${b}`;if(!dynamic.has(id))addPrimitive(id,color(object.entityId),object.entityId);}
    const id=`${object.entityId}:centroid`;if(!dynamic.has(id))addPrimitive(id,color(object.entityId),object.entityId,'box');
    const surfaceId=`${object.entityId}:surface`;if((object.surface||(scene.bodyKeyframes||[]).some((b:any)=>b.entityId===object.entityId))&&!dynamic.has(surfaceId)){addPrimitive(surfaceId,'#ffffff',object.entityId,'box');dynamic.get(surfaceId).representations[0].streamed=true;dynamic.get(surfaceId).representations[0].material.alphaMode='MASK';dynamic.get(surfaceId).representations[0].material.alphaCutoff=.5;}
  }
  return {document:{schemaVersion:2,target:'scene',captureId:scene.source_video_sha256,coordinateFrames:[{id:frameId,convention:'opencv',scale:{status:scene.units==='meters'?'calibrated':'uncalibrated'}}],cameras:[],observations:[],annotations:[],assets,entities},dynamic,owner};
}

export function applyFrame(frame:any,dynamic:Map<string,any>,frameId:string,markerScale:number,selected:string|null){
  // A fixed primitive pool keeps the large mesh uploaded. Alpha zero discards in both native render and picking passes.
  for(const entity of dynamic.values())entity.representations[0].material.baseColorFactor[3]=0;
  const show=(id:string,transform:any,entityId?:string)=>{const entity=dynamic.get(id);if(!entity||!transform)return;entity.currentModelTransform=transform;entity.representations[0].material.baseColorFactor[3]=selected&&entityId&&entityId!==selected ? .35 : 1;};
  if(!frame)return;
  const c=frame.c2w,world=(p:XYZ)=>[0,1,2].map(k=>c[k][3]+p.reduce((sum,v,j)=>sum+c[k][j]*v,0)),origin=world([0,0,0]);
  const corners=[[-.6,-.4,1],[.6,-.4,1],[.6,.4,1],[-.6,.4,1]].map(p=>world(scale(p,markerScale)));
  for(let i=0;i<4;i++){show(`camera:${i}`,lineTransform(origin,corners[i],markerScale*.02,frameId));show(`camera:${i+4}`,lineTransform(corners[i],corners[(i+1)%4],markerScale*.02,frameId));}
  for(const object of frame.objects){
    // ponytail: closed 8-sided links visualize observed joints only; radius is illustrative, not body shape or measured girth.
    for(const [a,b]of object.bones){const pa=object.keypoints3d[a],pb=object.keypoints3d[b];if(pa&&pb){const length=Math.hypot(...pb.map((v:number,k:number)=>v-pa[k]));show(`${object.entityId}:bone:${a}:${b}`,lineTransform(pa,pb,Math.min(markerScale*.22,length*.12),frameId),object.entityId);}}
    if(object.centroid)show(`${object.entityId}:centroid`,{coordinateFrameId:frameId,position:object.centroid,quaternion:[0,0,0,1],scale:Array(3).fill(markerScale*.18)},object.entityId);
  }
}

export function replayLayers(scene:any,pointCloud=!scene.meshUrl,objectView=false){
  return {point_cloud:pointCloud&&!objectView,observed_surface:!pointCloud,entityIds:objectView?(scene.staticObjects||[]).map((object:any)=>object.entityId):undefined};
}

export async function mountReplay(container:HTMLElement,scene:any,base:string,options:any){
  const urls:Record<string,string>={},owned:string[]=[],observed:XYZ[]=[],objectBounds:XYZ[]=[];
  const objectMeshes=new Map<string,Mesh>();
  try {
  const staticIds=new Set<string>((scene.staticObjects||[]).map((o:any)=>o.entityId));
  for(const object of scene.staticObjects||[]){
    const response=await fetch(assetUrl(object.meshUrl,base),{signal:options.signal});if(!response.ok)throw Error(`对象表面读取失败（HTTP ${response.status}）`);
    const bytes=await response.arrayBuffer();
    const meshes=readGLB(bytes);
    for(const mesh of meshes)objectBounds.push(...boundsCorners(mesh.bounds).map(p=>point(mesh.matrix,p)));
    if(object.generatedModel){require(meshes.length===1&&!meshes[0].texture,'对象切换需要单个彩色网格');objectMeshes.set(object.entityId,meshes[0]);}
    urls[object.entityId]=URL.createObjectURL(new Blob([bytes],{type:'model/gltf-binary'}));owned.push(urls[object.entityId]);
  }
  if(scene.meshUrl&&scene.pointCloudUrl)urls.mesh=assetUrl(scene.meshUrl,base);
  else if(scene.meshUrl){
    const response=await fetch(assetUrl(scene.meshUrl,base),{signal:options.signal});if(!response.ok)throw Error(`空间网格读取失败（HTTP ${response.status}）`);
    const bytes=await response.arrayBuffer();
    for(const mesh of readGLB(bytes))observed.push(...boundsCorners(mesh.bounds).map(p=>point(mesh.matrix,p)));
    // Retain one fetched GLB for the native viewer; video frames change transforms only.
    urls.mesh=URL.createObjectURL(new Blob([bytes],{type:'model/gltf-binary'}));owned.push(urls.mesh);
  }
  if(scene.pointCloudUrl)urls.points=assetUrl(scene.pointCloudUrl,base);
  else if(scene.points.length){const bytes=pointsGLB(scene.points);urls.points=URL.createObjectURL(new Blob([bytes],{type:'model/gltf-binary'}));owned.push(urls.points);}
  const {document,dynamic,owner}=replayDocument(scene,urls,options.color);
  const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity],cameraPoints:XYZ[]=[];
  const include=(p:XYZ)=>{for(let k=0;k<3;k++){min[k]=Math.min(min[k],p[k]);max[k]=Math.max(max[k],p[k]);}};
  scene.points.forEach((p:any)=>include(p.slice(1)));scene.frames.forEach((f:any)=>{const p=f.c2w.slice(0,3).map((r:any)=>r[3]);include(p);cameraPoints.push(p);f.objects.forEach((o:any)=>o.keypoints3d.filter(Boolean).forEach((p:XYZ)=>{include(p);observed.push(p);}));});
  const extent=Math.max(Math.hypot(...max.map((v,k)=>v-min[k])),.01);
  const cameraExtent=cameraPoints.length?Math.hypot(...[0,1,2].map(k=>Math.max(...cameraPoints.map(p=>p[k]))-Math.min(...cameraPoints.map(p=>p[k])))):extent;
  const markerScale=scene.units==='meters'?.16:Math.max(cameraExtent,.01)*.04;
  // Navigation frames actual observations; the all-points view keeps every original distant point accessible.
  if(!observed.length)observed.push(...cameraPoints);
  let current:any=undefined,selected:string|null=null,disposed=false,objectView=false,fromSource=!!scene.humanSurfaces,bodyModels=false,pointCloud=!!scene.pointCloudUrl||!scene.meshUrl;
  const surfaceCache=new Map<string,Promise<ReturnType<typeof readGLB>>>();
  const loadSurface=(surface:any)=>{
    if(!surfaceCache.has(surface.meshUrl)){
      const promise=fetch(assetUrl(surface.meshUrl,base),{signal:options.signal}).then(async response=>{
        if(!response.ok)throw Error(`模型读取失败（HTTP ${response.status}）`);
        const bytes=await response.arrayBuffer(),hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(v=>v.toString(16).padStart(2,'0')).join('');
        if(hash!==(surface.sha256||surface.mesh_sha256))throw Error('模型校验失败');
        const meshes=readGLB(bytes);if(meshes.length!==1||meshes[0].texture)throw Error('模型应为单个彩色网格');return meshes;
      });
      surfaceCache.set(surface.meshUrl,promise);promise.catch(()=>{});
      // ponytail: retain only a short playback window, never every frame's mesh.
      while(surfaceCache.size>32)surfaceCache.delete(surfaceCache.keys().next().value!);
    }
    return surfaceCache.get(surface.meshUrl)!;
  };
  const rendered=new Set<string>();
  const announce=()=>options.frame(current,{mode:bodyModels?'body':'surface',rendered:rendered.size,total:current?.objects.length||0});
  const hideLinks=()=>{if(scene.humanSurfaces)for(const[id,e]of dynamic)if(!id.startsWith('camera:')&&!id.endsWith(':surface'))e.representations[0].material.baseColorFactor[3]=0;};
  const displaySurfaces=(frame:any)=>{
    if(!frame)return;
    const index=scene.frames.indexOf(frame),generation=bodyModels;
    if(!bodyModels)for(const future of scene.frames.slice(index,index+8))for(const o of future.objects)if(o.surface)void loadSurface(o.surface);
    for(const object of frame.objects){
      const body=bodyModels?bodySample(scene,frame,object.entityId):null;
      const task=body?Promise.all([loadSurface(body.a),loadSurface(body.b)]).then(([a,b])=>interpolateBody(a[0],b[0],body.t)):
        !bodyModels&&object.surface?loadSurface(object.surface).then(m=>m[0]):null;
      if(!task)continue;
      void task.then(mesh=>{
        if(disposed||current!==frame||generation!==bodyModels)return;
        const id=`${object.entityId}:surface`,entity=dynamic.get(id);viewer.setStreamMesh(id,mesh);
        entity.representations[0].material.lighting=bodyModels;entity.representations[0].material.baseColorFactor=[...Array(3).fill(selected&&selected!==object.entityId ? .65 : 1),1];
        for(const [key,e]of dynamic)if(key.startsWith(object.entityId+':')&&key!==id)e.representations[0].material.baseColorFactor[3]=0;
        rendered.add(object.entityId);void viewer.setScene(document);announce();
      }).catch(error=>{if(!disposed&&current===frame&&error.name!=='AbortError')options.error(error.message);});
    }
  };
  const viewer=mountSceneViewer(container,{showSourcePhoto:false,resolveAsset:async id=>urls[id],layers:{showCandidates:true,editable:false,showBounds:false,lighting:false,opacity:1,pointSize:1.6,...replayLayers(scene,pointCloud)},onEvent:event=>{if(event.type==='selectionIntent')options.select(owner.get(event.entityId)||(staticIds.has(event.entityId)?event.entityId:null));if(event.type==='loadError')options.error(`空间资产加载失败：${event.code}`);if(event.type==='contextLost')options.error('三维图形上下文已丢失，请重新选择此样本。');}});
  const dispose=()=>{disposed=true;viewer.dispose();owned.forEach(url=>URL.revokeObjectURL(url));options.signal?.removeEventListener('abort',dispose);};
  options.signal?.addEventListener('abort',dispose,{once:true});
  const sourceView=()=>{fromSource=true;for(const [id,e]of dynamic)if(id.startsWith('camera:'))e.representations[0].material.baseColorFactor[3]=0;void viewer.setScene(document);const frame=current||scene.frames[0];if(!frame)return;viewer.setCamera({...sourceCamera({cameraToWorld:frame.c2w},extent/2,min.map((v,k)=>(v+max[k])/2)),exact:false,mode:'free'});};
  const fit=(all=false)=>{fromSource=false;if(!Number.isFinite(extent))return;const c=scene.frames[0]?.c2w,up=c?unit(c.slice(0,3).map((r:any)=>-r[1])):[0,0,1],back=c?unit(c.slice(0,3).map((r:any)=>-r[2])):[0,-1,0];viewer.setCamera({...fitCamera(objectView&&objectBounds.length?objectBounds:!all&&observed.length?observed:boundsCorners({min,max}),unit(add(add(back,scale(cross(up,back),.35)),scale(up,.25))),up,Math.max(container.clientWidth,1)/Math.max(container.clientHeight,1)),mode:'free'});};
  await viewer.setScene(document);scene.humanSurfaces?sourceView():fit();
  return {
    setTime(time:number,selection:string|null=null){if(disposed)return;const frame=frameAt(scene.frames,time);if(frame===current&&selection===selected)return;current=frame;selected=selection;rendered.clear();applyFrame(frame,dynamic,scene.coordinate_frame,markerScale,selected);hideLinks();if(fromSource)for(const [id,e]of dynamic)if(id.startsWith('camera:'))e.representations[0].material.baseColorFactor[3]=0;viewer.setSelection({entityId:selected?(staticIds.has(selected)?selected:`${selected}:surface`):null});void viewer.setScene(document);displaySurfaces(frame);announce();},
    fit,sourceView,
    setBodyModels(value:boolean){bodyModels=value;rendered.clear();applyFrame(current,dynamic,scene.coordinate_frame,markerScale,selected);hideLinks();if(fromSource)for(const[id,e]of dynamic)if(id.startsWith('camera:'))e.representations[0].material.baseColorFactor[3]=0;void viewer.setScene(document);displaySurfaces(current);announce();},
    setPointCloud(value:boolean){pointCloud=value;viewer.setLayers(replayLayers(scene,pointCloud,objectView));},
    setObjectView(value:boolean){objectView=value;viewer.setLayers(replayLayers(scene,pointCloud,objectView));fit();},
    async setObjectModel(id:string,value:boolean){
      const object=scene.staticObjects?.find((o:any)=>o.entityId===id);
      require(object?.generatedModel&&objectMeshes.has(id),'此对象没有通过检查的生成模型');
      const mesh=value?(await loadSurface(object.generatedModel))[0]:objectMeshes.get(id)!;
      if(disposed)return;
      viewer.setStreamMesh(id,mesh);
    },
    dispose,
  };
  } catch(error) {owned.forEach(url=>URL.revokeObjectURL(url));throw error;}
}
