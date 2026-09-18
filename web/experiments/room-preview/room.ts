import {mountSceneViewer} from '../../src/viewer/native-viewer.ts';
import {add,scale,unit,cross,fitCamera,boundsCorners} from '../../src/viewer/native-math.ts';
import type {SceneDocument,Transform,Representation} from '../../src/types.ts';

// This is an experiment adapter to the existing report viewer, not a second renderer.
export function roomScene(manifest:any,base:string){
  if(manifest.version!==1||!manifest.frames?.length||manifest.metric_scale_known!==false)throw Error('invalid_native_cloud_manifest');
  const matrix=(m:any,n:number)=>Array.isArray(m)&&m.length===n&&m.every((r:any)=>Array.isArray(r)&&r.length===n&&r.every(Number.isFinite));
  const frameId='native-predicted-world';
  const transform:Transform={coordinateFrameId:frameId,position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]};
  const urls:Record<string,string>={mesh:new URL('../room.glb',base).href,cloud:new URL(manifest.display.glb,base).href};
  const rep=(id:string,kind:Representation['kind']):Representation=>({id,kind,assetId:id,coordinateFrameId:frameId,transform,placementState:'unconfirmed',placementReason:'requires_alignment_confirmation'});
  const representations=[rep('cloud','point_cloud'),rep('mesh','observed_surface')];
  const cameras=manifest.frames.map((f:any)=>{
    if(!f.frame_id||urls[f.frame_id]||!f.display?.glb||!f.canonical_rgb||!Number.isFinite(f.timestamp)||!matrix(f.K,3)||!matrix(f.camera_to_world,4)||f.canonical_size_wh?.length!==2||!f.canonical_size_wh.every((n:number)=>Number.isInteger(n)&&n>0))throw Error('invalid_native_frame');
    urls[f.frame_id]=new URL(f.canonical_rgb,base).href;
    const id=`cloud-${f.frame_id}`;urls[id]=new URL(f.display.glb,base).href;representations.push(rep(id,'point_cloud'));
    return {id:f.frame_id,imageId:f.frame_id,coordinateFrameId:frameId,width:f.canonical_size_wh[0],height:f.canonical_size_wh[1],K:f.K,cameraToWorld:f.camera_to_world};
  });
  const scene:SceneDocument={schemaVersion:2,target:'scene',captureId:manifest.run_id,coordinateFrames:[{id:frameId,convention:'opencv',scale:{status:'uncalibrated'}}],cameras,
    geometryBindings:Object.fromEntries(cameras.map((c:any)=>[c.imageId,{cameraId:c.id,geometrySolutionId:manifest.run_id}])),
    observations:[],annotations:[],assets:Object.keys(urls).map(id=>({id})),
    // Context denotes unsegmented scene evidence; this is not one physical object.
    entities:[{id:'room-evidence',label:'未分割房间观测',sourceContext:true,associationState:'association_pending',representations}]};
  return {scene,urls};
}

async function main(){
  const el=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
  const controls=Array.from(document.querySelectorAll<HTMLButtonElement|HTMLSelectElement>('button,select'));
  controls.forEach(control=>control.disabled=true);
  const base=new URL('../native-cloud/manifest.json',location.href).href;
  const response=await fetch(base);if(!response.ok)throw Error('无法读取点云清单，请先运行 export_room_pointcloud.py');
  const manifest=await response.json(),{scene,urls}=roomScene(manifest,base);
  const stage=el('stage'),frames=el<HTMLSelectElement>('frame'),scope=el<HTMLSelectElement>('scope'),view=el<HTMLSelectElement>('view'),status=el('status');
  const count=(n:number)=>n.toLocaleString('zh-CN');
  let mode='cloud',ready=false;
  const viewer=mountSceneViewer(stage,{showSourcePhoto:false,resolveAsset:async id=>{if(!urls[id])throw Error('unknown_asset');return urls[id];},layers:{showCandidates:true,editable:false,showBounds:false,point_cloud:true,observed_surface:false,generated_mesh:false,primitive:false,lighting:false,opacity:1,representationIds:['cloud']},onEvent:event=>{
    if(event.type==='loadError'){status.textContent=`几何加载失败：${event.code}`;status.setAttribute('role','alert');}
    if(event.type==='loadProgress'&&event.phase==='assets'&&!event.failed){status.setAttribute('role','status');status.textContent=event.pending?'正在读取保存的几何…':mode==='cloud'?`${count(event.vertices)} 个显示点 · 原生预测坐标 · 未做运动分离`:`${count(event.triangles)} 个三角面 · 可见表面网格，尚未拆分物体`;}
  }});
  for(const f of manifest.frames){const option=document.createElement('option');option.value=f.frame_id;option.textContent=`${String(f.frame_index+1).padStart(2,'0')} / ${manifest.frames.length}`;frames.append(option);}
  el('summary').textContent=`${manifest.frames.length} 个关键帧 · 保存 ${count(manifest.full_cloud.point_count)} 个有效原生点 · 浏览器显示 ${count(manifest.display.point_count)} 个采样点`;
  el<HTMLAnchorElement>('ply').href=new URL(manifest.full_cloud.path,base).href;
  function freeView(){
    const c=scene.cameras[0].cameraToWorld,up=unit(c.slice(0,3).map(r=>-r[1])),back=unit(c.slice(0,3).map(r=>-r[2]));
    // Camera-up is for navigation only; no measured ground plane is invented.
    viewer.setCamera({...fitCamera(boundsCorners(manifest.bounds),unit(add(add(back,scale(cross(up,back),.25)),scale(up,.2))),up,stage.clientWidth/stage.clientHeight),mode:'free'});
  }
  function camera(){view.value==='source'?viewer.setCamera(frames.value):freeView();}
  function selection(){
    const f=manifest.frames.find((f:any)=>f.frame_id===frames.value);
    el<HTMLImageElement>('photo').src=urls[f.frame_id];
    el('time').textContent=`${(f.timestamp-manifest.frames[0].timestamp).toFixed(2)} 秒`;
    el('source-count').textContent=`本帧 ${count(f.valid_points)} 个有效点 · 仅当前帧模式显示 ${count(f.display_points)} 点`;
    if(ready){viewer.setLayers({representationIds:[mode==='mesh'?'mesh':scope.value==='all'?'cloud':`cloud-${f.frame_id}`]});if(view.value==='source')camera();}
  }
  function show(next:string){mode=next;el('cloud').setAttribute('aria-pressed',String(mode==='cloud'));el('mesh').setAttribute('aria-pressed',String(mode==='mesh'));el('scope-label').hidden=mode==='mesh';viewer.setLayers({point_cloud:mode==='cloud',observed_surface:mode==='mesh',representationIds:[mode==='mesh'?'mesh':scope.value==='all'?'cloud':`cloud-${frames.value}`]});}
  frames.onchange=selection;scope.onchange=selection;view.onchange=camera;
  el('cloud').onclick=()=>show('cloud');el('mesh').onclick=()=>show('mesh');el('fit').onclick=()=>{view.value='free';freeView();};
  selection();await viewer.setScene(scene);ready=true;freeView();controls.forEach(control=>control.disabled=false);
  window.addEventListener('pagehide',()=>viewer.dispose(),{once:true});
}

if(typeof document!=='undefined')main().catch(error=>{const status=document.getElementById('status')!;status.textContent=String(error.message||error);status.setAttribute('role','alert');});
