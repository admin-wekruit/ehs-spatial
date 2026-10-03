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
export function cameraMatrix(camera:Camera,aspect:number,radius:number):Float32Array {const {projection,view}=cameraMatrices(camera,aspect,radius);return matmul(projection,view);}
// Column-major projection and view (world to OpenGL camera space, -Z forward) apart, for renderers that need camera space.
export function cameraMatrices(camera:Camera,aspect:number,radius:number):{projection:number[];view:number[]} {
  const {eye,target,up,frame,exact}=camera,near=Math.max(radius*.0001,1e-6),far=Math.max(radius*100,near*1000);let P:number[];
  if(exact){const K=frame.K,w=frame.width,h=frame.height;P=[2*K[0][0]/w,0,0,0,-2*K[0][1]/w,2*K[1][1]/h,0,0,1-2*(K[0][2]+.5)/w,2*(K[1][2]+.5)/h-1,-(far+near)/(far-near),-1,0,0,-2*far*near/(far-near),0];}
  else if(camera.orthographic){const height=camera.orthoHeight!,width=height*aspect;P=[2/width,0,0,0,0,2/height,0,0,0,0,-2/(far-near),0,0,0,-(far+near)/(far-near),1];}
  else{const f=1/Math.tan((camera.fov||Math.PI/3)/2);P=[f/aspect,0,0,0,0,f,0,0,0,0,-(far+near)/(far-near),-1,0,0,-2*far*near/(far-near),0];}
  const z=unit(add(eye,scale(target,-1))),x=unit(cross(up,z)),y=cross(z,x),V=[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1];return {projection:P,view:V};
}
// A drag while standing where the video camera stood: the view turns about the eye (the picture follows the cursor: drag right
// looks left, drag down looks up) and the eye stays on the walked path.
export function lookAround(camera:Camera,dx:number,dy:number,speed=.004){
  let f=add(camera.target,scale(camera.eye,-1));f=rotate(f,unit(camera.up),dx*speed);const r=unit(cross(f,camera.up));f=rotate(f,r,dy*speed);
  camera.up=rotate(camera.up,r,dy*speed);camera.target=add(camera.eye,f);
}
// Whether an eye is within `limit` of one of the cameras and, given a view direction, of one that looked within 45 degrees of it.
export function nearCameras(cameras:{eye:Vec;forward:Vec}[],eye:Vec,forward:Vec|null,limit:number){
  return cameras.some(c=>Math.hypot(c.eye[0]-eye[0],c.eye[1]-eye[1],c.eye[2]-eye[2])<=limit&&(!forward||dot(c.forward,forward)>=Math.SQRT1_2));
}
// Seen from above the walked cameras, everything higher than 1.5 times their median height above the floor is cut (a
// dollhouse view): the plane [n, d] with n.x + d > 0 above the cut, or zeros (no cut) from the cameras' height or below.
export function ceilingCut(normal:Vec,offset:number,eyes:Vec[],eye:Vec):number[]{
  const length=Math.hypot(...normal);if(!eyes.length||!(length>1e-8)||!normal.every(Number.isFinite)||!Number.isFinite(offset))return [0,0,0,0];
  const heights=eyes.map(e=>(dot(normal,e)+offset)/length).sort((a,b)=>a-b),median=heights[heights.length>>1],sign=median<0?-1:1;
  const n=scale(normal,sign/length),d=offset*sign/length,cut=1.5*Math.abs(median);
  return dot(n,eye)+d>cut?[...n,d-cut]:[0,0,0,0];
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

export type SurfacePick = { point: Vec; entityId: string; representationId: string; coordinateFrameId: string };
export type MeasurementScale = { nativeToMeters: number | null; status: string; source: string };
export type GroundMeasurementKind = 'point_ground' | 'point_distance' | 'region_ground';
export type GroundPointMeasurement = {
  kind: GroundMeasurementKind; pointsNative: Vec[]; ground: { normal: Vec; offset: number };
  heightsNative: number[]; feetNative: Vec[]; minHeightNative: number; maxHeightNative: number;
  distanceNative?: number; heightDifferenceNative?: number; inclinationDeg?: number;
};
export function measureGroundPoints(kind: GroundMeasurementKind, points: Vec[], ground?: { normal?: Vec | null; offset?: number | null } | null): GroundPointMeasurement {
  const count = { point_ground: 1, point_distance: 2, region_ground: 3 }[kind];
  if (!count || !Array.isArray(points) || points.length !== count || points.some(p => !Array.isArray(p) || p.length !== 3 || !p.every(Number.isFinite))) throw Error('measurement_points_invalid');
  const n = ground?.normal, d = ground?.offset;
  if (!Array.isArray(n) || n.length !== 3 || !n.every(Number.isFinite) || !Number.isFinite(d)) throw Error('measurement_ground_missing');
  const length = Math.hypot(...n);
  if (!Number.isFinite(length) || !(length > 0)) throw Error('measurement_ground_missing');
  const normal = n.map(value => value / length), offset = d! / length;
  const heightsNative = points.map(p => dot(normal, p) + offset), feetNative = points.map((p, i) => add(p, scale(normal, -heightsNative[i])));
  // Height is linear on this sampled triangle, so its extrema occur at the vertices.
  const result: GroundPointMeasurement = { kind, pointsNative: points.map(p => [...p]), ground: { normal, offset }, heightsNative, feetNative, minHeightNative: Math.min(...heightsNative), maxHeightNative: Math.max(...heightsNative) };
  if (count > 1) {
    const edge = add(points[1], scale(points[0], -1)), distance = Math.hypot(...edge);
    if (distance < 1e-8) throw Error('measurement_points_coincident');
    if (kind === 'point_distance') { result.distanceNative = distance; result.heightDifferenceNative = dot(edge, normal); }
    else {
      const other = add(points[2], scale(points[0], -1)), otherLength = Math.hypot(...other), endDistance = Math.hypot(...add(points[2], scale(points[1], -1)));
      if (Math.min(otherLength, endDistance) < 1e-8) throw Error('measurement_points_coincident');
      const regionNormal = cross(scale(edge, 1 / distance), scale(other, 1 / otherLength));
      if (Math.hypot(...regionNormal) < 1e-8) throw Error('measurement_points_collinear');
      result.inclinationDeg = Math.acos(Math.max(0, Math.min(1, Math.abs(dot(unit(regionNormal), normal))))) * 180 / Math.PI;
    }
  }
  if (![...heightsNative, ...feetNative.flat(), result.distanceNative ?? 0, result.heightDifferenceNative ?? 0, result.inclinationDeg ?? 0].every(Number.isFinite)) throw Error('measurement_points_invalid');
  return result;
}
export function measurementLength(value: number, measurementScale?: MeasurementScale) {
  if (!Number.isFinite(value)) throw Error('measurement_points_invalid');
  const factor = measurementScale?.nativeToMeters;
  if (factor != null && (!Number.isFinite(factor) || factor <= 0)) throw Error('measurement_scale_invalid');
  const converted = factor == null ? value : value * factor * 100;
  if (!Number.isFinite(converted)) throw Error('measurement_scale_invalid');
  return { value: converted, unit: factor == null ? 'native' : 'cm' };
}
export function groundMeasurementLabel(result: GroundPointMeasurement, measurementScale?: MeasurementScale) {
  const length = (value: number) => { const display = measurementLength(value, measurementScale); return `${Number(display.unit === 'cm' ? display.value.toFixed(2) : display.value.toPrecision(4))} ${display.unit}`; };
  return result.kind === 'point_ground' ? `h ${length(result.heightsNative[0])}` : result.kind === 'point_distance' ? `d ${length(result.distanceNative!)} · Δh ${length(result.heightDifferenceNative!)}` : `h ${length(result.minHeightNative)} … ${length(result.maxHeightNative)} · ${result.inclinationDeg!.toFixed(1)}°`;
}
export function threePointAngle(points: Vec[]) {
  if(points.length!==3||points.some(p=>p.length!==3||!p.every(Number.isFinite)))throw Error('measurement_points_invalid');
  const [a,b,c]=points,u=add(a,scale(b,-1)),v=add(c,scale(b,-1)),length=Math.min(Math.hypot(...u),Math.hypot(...v));
  if(length<1e-8)throw Error('measurement_points_coincident');
  const first=unit(u),second=unit(v),cosine=Math.max(-1,Math.min(1,dot(first,second))),angle=Math.acos(cosine);
  let tangent=add(second,scale(first,-cosine));
  if(Math.hypot(...tangent)<1e-8)tangent=cross(first,Math.abs(first[0])<.8?[1,0,0]:[0,1,0]);
  tangent=unit(tangent);
  const arc=Array.from({length:33},(_,i)=>add(b,scale(add(scale(first,Math.cos(angle*i/32)),scale(tangent,Math.sin(angle*i/32))),length*.3)));
  return {value:angle*180/Math.PI,arc,labelPoint:arc[16]};
}

// Nearest hit distance along a ray without per-triangle allocation: fast enough for one click on a room mesh of
// millions of triangles (vertex stride 12 floats, position first). Infinity when nothing is hit.
// ponytail: linear scan per click; a BVH if room meshes grow past ~10M triangles.
export function rayMeshDistance(vertices: Float32Array,indices: Uint32Array,origin: Vec,direction: Vec) {
  const [ox,oy,oz]=origin,[dx,dy,dz]=direction;let nearest=Infinity;
  for(let i=0;i<indices.length;i+=3){
    const a=indices[i]*12,b=indices[i+1]*12,c=indices[i+2]*12,ax=vertices[a],ay=vertices[a+1],az=vertices[a+2];
    const e1x=vertices[b]-ax,e1y=vertices[b+1]-ay,e1z=vertices[b+2]-az,e2x=vertices[c]-ax,e2y=vertices[c+1]-ay,e2z=vertices[c+2]-az;
    const px=dy*e2z-dz*e2y,py=dz*e2x-dx*e2z,pz=dx*e2y-dy*e2x,det=e1x*px+e1y*py+e1z*pz;
    if(det>-1e-12&&det<1e-12)continue;
    const inv=1/det,tx=ox-ax,ty=oy-ay,tz=oz-az,u=(tx*px+ty*py+tz*pz)*inv;if(u<0||u>1)continue;
    const qx=ty*e1z-tz*e1y,qy=tz*e1x-tx*e1z,qz=tx*e1y-ty*e1x,v=(dx*qx+dy*qy+dz*qz)*inv;if(v<0||u+v>1)continue;
    const t=(e2x*qx+e2y*qy+e2z*qz)*inv;if(t>0&&t<nearest)nearest=t;
  }
  return nearest;
}

// ponytail: linear triangle traversal only on a measurement click, after GPU
// object picking. Add a per-mesh BVH if individual selectable meshes grow larger.
export function rayMeshPoint(vertices: Float32Array,indices: Uint32Array,origin: Vec,direction: Vec) {
  let nearest=Infinity,hit:Vec|null=null;
  for(let i=0;i<indices.length;i+=3){
    const a=Array.from(vertices.subarray(indices[i]*12,indices[i]*12+3)),b=Array.from(vertices.subarray(indices[i+1]*12,indices[i+1]*12+3)),c=Array.from(vertices.subarray(indices[i+2]*12,indices[i+2]*12+3));
    const e1=add(b,scale(a,-1)),e2=add(c,scale(a,-1)),p=cross(direction,e2),det=dot(e1,p);
    if(Math.abs(det)<=1e-12*Math.hypot(...e1)*Math.hypot(...e2)*Math.hypot(...direction))continue;
    const t=add(origin,scale(a,-1)),u=dot(t,p)/det;if(u<0||u>1)continue;
    const q=cross(t,e1),v=dot(direction,q)/det;if(v<0||u+v>1)continue;
    const distance=dot(e2,q)/det;if(distance<0||distance>=nearest)continue;
    nearest=distance;hit=add(origin,scale(direction,distance));
  }
  return hit?{point:hit,distance:nearest}:null;
}

export function verticalEdgeAngle(points:Vec[],groundNormal:Vec) {
  if(points.length!==2||points.some(p=>p.length!==3||!p.every(Number.isFinite)))throw Error('measurement_points_invalid');
  if(!Array.isArray(groundNormal)||groundNormal.length!==3||!groundNormal.every(Number.isFinite)||Math.hypot(...groundNormal)<1e-8)throw Error('measurement_ground_missing');
  const [start,end]=points,edge=add(end,scale(start,-1)),normal=unit(groundNormal);
  const verticalEnd=add(start,scale(normal,Math.hypot(...edge)*(dot(edge,normal)>0?1:-1)));
  return {...threePointAngle([verticalEnd,start,end]),verticalEnd};
}
