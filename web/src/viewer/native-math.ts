// Extracted from the existing Panoptes native viewer; source camera pixels use
// pixel centres and OpenCV c2w, while WebGL uses a -Z viewing direction.
export type Vec = number[];
export const add = (a: Vec,b: Vec): Vec => a.map((v,i)=>v+b[i]);
export const scale = (a: Vec,s: number): Vec => a.map(v=>v*s);
export const dot = (a: Vec,b: Vec): number => a.reduce((s,v,i)=>s+v*b[i],0);
export const cross = (a: Vec,b: Vec): Vec => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export function unit(a: Vec): Vec { const n=Math.hypot(...a); if(!(n>1e-12))throw Error('invalid_direction');return scale(a,1/n); }
export const identity = (): number[] => [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
export function matmul(a: ArrayLike<number>,b: ArrayLike<number>): Float32Array {
  const m=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)m[c*4+r]+=a[k*4+r]*b[c*4+k];return m;
}
export const point=(m: ArrayLike<number>,p: Vec): Vec=>[0,1,2].map(i=>m[i]*p[0]+m[4+i]*p[1]+m[8+i]*p[2]+m[12+i]);
export type Transform={coordinateFrameId?:string;position:Vec;quaternion:Vec;scale:Vec};
export function transformMatrix(t?: Transform|null): number[] {
  if(!t)return identity();
  if(![t.position,t.quaternion,t.scale].every(a=>Array.isArray(a)&&a.every(Number.isFinite))||t.position.length!==3||t.quaternion.length!==4||t.scale.length!==3||t.scale.some(v=>v<=0))throw Error('invalid_transform');
  const norm=Math.hypot(...t.quaternion);if(Math.abs(norm-1)>1e-4)throw Error('invalid_quaternion');
  const [x,y,z,w]=t.quaternion;const m=[1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w),0,2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w),0,2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y),0,...t.position,1];
  for(let c=0;c<3;c++)for(let r=0;r<3;r++)m[c*4+r]*=t.scale[c];return m;
}
export const rotate=(v: Vec,a: Vec,t: number): Vec=>add(add(scale(v,Math.cos(t)),scale(cross(a,v),Math.sin(t))),scale(a,dot(a,v)*(1-Math.cos(t))));
export type Camera={eye:Vec;target:Vec;up:Vec;exact?:boolean;frame?:any;orthographic?:boolean;orthoHeight?:number;fov?:number};
export function sourceCamera(frame:any,radius:number,center:Vec):Camera {
  const C=frame.cameraToWorld;if(!C?.flat().every(Number.isFinite)||C.length!==4)throw Error('invalid_camera');
  const eye=C.slice(0,3).map((r:Vec)=>r[3]),forward=C.slice(0,3).map((r:Vec)=>r[2]);
  return {eye,target:add(eye,scale(forward,Math.max(radius*.25,dot(add(center,scale(eye,-1)),forward)))),up:C.slice(0,3).map((r:Vec)=>-r[1]),frame,exact:true};
}
export function cameraMatrix(camera:Camera,aspect:number,radius:number):Float32Array {
  const {eye,target,up,frame,exact}=camera,near=Math.max(radius*.0001,1e-6),far=Math.max(radius*100,near*1000);let P:number[];
  if(exact){const K=frame.K,w=frame.width,h=frame.height;P=[2*K[0][0]/w,0,0,0,-2*K[0][1]/w,2*K[1][1]/h,0,0,1-2*(K[0][2]+.5)/w,2*(K[1][2]+.5)/h-1,-(far+near)/(far-near),-1,0,0,-2*far*near/(far-near),0];}
  else if(camera.orthographic){const height=camera.orthoHeight!,width=height*aspect;P=[2/width,0,0,0,0,2/height,0,0,0,0,-2/(far-near),0,0,0,-(far+near)/(far-near),1];}
  else{const f=1/Math.tan((camera.fov||Math.PI/3)/2);P=[f/aspect,0,0,0,0,f,0,0,0,0,-(far+near)/(far-near),-1,0,0,-2*far*near/(far-near),0];}
  const z=unit(add(eye,scale(target,-1))),x=unit(cross(up,z)),y=cross(z,x),V=[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1];return matmul(P,V);
}
export const boundsCorners=(b:{min:Vec;max:Vec}):Vec[]=>Array.from({length:8},(_,n)=>[0,1,2].map(k=>(n>>k)&1?b.max[k]:b.min[k]));
export function projected(m:ArrayLike<number>,p:Vec,w:number,h:number):Vec|null {
  const q=[0,1,2,3].map(i=>m[i]*p[0]+m[i+4]*p[1]+m[i+8]*p[2]+m[i+12]);if(q[3]<=0||Math.abs(q[2]/q[3])>1)return null;return[(q[0]/q[3]+1)*w/2,(1-q[1]/q[3])*h/2];
}

export function fitCamera(points:Vec[],back:Vec,up:Vec,aspect:number,orthographic=false):Camera {
  if(!points.length||!(aspect>0))throw Error('invalid_camera_fit');
  const min=[0,1,2].map(k=>Math.min(...points.map(p=>p[k]))),max=[0,1,2].map(k=>Math.max(...points.map(p=>p[k]))),target=min.map((n,k)=>(n+max[k])/2);
  const z=unit(back),x=unit(cross(up,z)),y=cross(z,x),fov=Math.PI/3,tanY=Math.tan(fov/2),tanX=tanY*aspect;
  const local=points.map(p=>{const d=add(p,scale(target,-1));return [dot(d,x),dot(d,y),dot(d,z)];});
  const radius=Math.max(Math.hypot(...max.map((n,k)=>n-min[k]))/2,1e-4);
  const distance=Math.max(radius,...local.map(p=>p[2]+Math.max(Math.abs(p[0])/tanX,Math.abs(p[1])/tanY)/.85));
  const orthoHeight=Math.max(radius*.01,...local.map(p=>2*Math.max(Math.abs(p[1]),Math.abs(p[0])/aspect)/.85));
  return {eye:add(target,scale(z,distance)),target,up:y,exact:false,orthographic,orthoHeight,fov};
}
