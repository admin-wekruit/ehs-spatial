// Run in a browser after bundling with Vite; exercises the actual WebGL viewer.
import {mountSceneViewer} from '../src/viewer/native-viewer.ts';
import {pointsGLB} from '../experiments/video-mvp/scene.ts';

const stage=document.createElement('div'),result=document.createElement('pre');
stage.style.cssText='width:640px;height:480px';document.body.append(stage,result);
const transform={coordinateFrameId:'f',position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]};
const cloud=URL.createObjectURL(new Blob([pointsGLB([['near',0,0,0],['far',20,20,20]])]));
const doc={captureId:'camera-regression',cameras:[],coordinateFrames:[{id:'f'}],observations:[],assets:[{id:'cloud'}],entities:[
  {id:'box',activeModelRepresentationId:'box',currentModelTransform:transform,representations:[{id:'box',kind:'primitive',coordinateFrameId:'f',placementState:'confirmed',transform,primitive:{kind:'box',parameters:{dimensions:[2,1,3]}}}]},
  {id:'cloud',sourceContext:true,representations:[{id:'cloud',assetId:'cloud',kind:'point_cloud',coordinateFrameId:'f',placementState:'confirmed',transform}]},
]};
let ready:()=>void=()=>{};
const viewer=mountSceneViewer(stage,{resolveAsset:async()=>cloud,layers:{point_cloud:false,showBounds:false,opacity:1},onEvent:e=>{if(e.type==='renderReady')ready();}});
const painted=()=>new Promise<void>(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>resolve())));
const snapshot=()=>stage.querySelector('canvas')!.toDataURL();
async function roundtrip(){
  await painted();const before=snapshot();
  for(const showCloud of [true,false]){
    const loaded=new Promise<void>(resolve=>{ready=resolve;});
    viewer.setLayers({point_cloud:showCloud,primitive:!showCloud});await loaded;await painted();
  }
  if(snapshot()!==before)throw Error('Layer loading overwrote the chosen camera');
}
try{
  await viewer.setScene(doc);
  viewer.setCamera({eye:[4,-6,3],target:[0,0,0],up:[0,0,1],mode:'free'});
  await roundtrip();
  viewer.setCamera('front');await painted();
  stage.querySelector('canvas')!.dispatchEvent(new WheelEvent('wheel',{deltaY:100,cancelable:true}));
  await roundtrip();
  result.textContent='PASS: explicit camera and user zoom survive point-cloud / model layer reloads';
}catch(error){result.textContent='FAIL: '+error;throw error;}
finally{URL.revokeObjectURL(cloud);}
