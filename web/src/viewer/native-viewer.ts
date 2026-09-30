import {add,scale,dot,cross,unit,identity,matmul,point,rotate,transformMatrix,sourceCamera,cameraMatrix,cameraMatrices,boundsCorners,projected,fitCamera,rayMeshPoint,rayMeshDistance,lookAround,nearCameras,ceilingCut,type SurfacePick,type Camera,type Vec,type Transform} from './native-math.ts';
import {isReferenceSurface} from '../scene-semantics.ts';
import {activeModel,compositeModelEvidence,modelPreviewEntities,modelPreviewGeometry,isCurrentReferenceSurface,modelFamilyGeometry,modelFamilyTransforms,entityGeometryForLayer,representationAvailable,representationInPhoto,cameraForImage,currentCameras,cameraPath,type GeometryLayer} from '../core.ts';
import type {SceneDocument,RepresentationLoadState} from '../types';
import {createSplatLayer,type SplatSource} from './splat-layer.ts';

export type Mesh={vertices:Float32Array;indices:Uint32Array;mode:number;matrix:ArrayLike<number>;texture?:Blob;material?:{baseColorFactor:number[];alphaMode:'OPAQUE'|'MASK'|'BLEND';alphaCutoff:number};bounds:{min:Vec;max:Vec};name?:string};
type GPU={mesh:Mesh;vertex:WebGLBuffer;index:WebGLBuffer;texture:WebGLTexture;entityId:string;representation:any};
export type ViewerEvent={type:string;[key:string]:any};
export type ViewerOptions={resolveAsset:(id:string)=>Promise<string|{url:string}>;locale?:string;onEvent?:(event:ViewerEvent)=>void;layers?:Record<string,any>;showSourcePhoto?:boolean};
export type SceneViewer=ReturnType<typeof mountSceneViewer>;

export function representationPass(entity:any,representation:any,frameId:string|null,layers:any) {
  // A moving object's surface belongs to one moment: shown while the report's video time is inside it, in any layer,
  // unless the static/dynamic switch says static only. Everything else is the static scene.
  // layers.loading asks which assets to hold, not what to draw: time and the switch never change the loaded set,
  // so playing the video or flipping the switch never reloads the scene.
  if(representation?.timeRange){
    // Bones and the surface they sit on are shown one at a time: a bone on the surface is hidden by it.
    const skeleton=representation.sourceKind==='moving_object_skeleton';
    const on=!!entity&&(!layers.entityIds||layers.entityIds.includes(entity.id))&&(layers.loading||layers.part!=='static'&&skeleton===!!layers.skeleton&&Number.isFinite(layers.time)&&layers.time>=representation.timeRange[0]&&layers.time<representation.timeRange[1]);
    return {available:on,visible:on,pick:on&&!layers.loading,selectable:on&&!layers.loading};
  }
  if(layers.part==='dynamic'&&!layers.loading)return {available:false,visible:false,pick:false,selectable:false};
  const reference=!!entity&&layers.modelOnly&&isCurrentReferenceSurface(entity,representation,layers.observations);
  const available=!!entity&&(!layers.entityId||entity.id===layers.entityId)&&(!layers.entityIds||layers.entityIds.includes(entity.id))&&
    (representation.sourceKind!=='observed_reference_surface'||reference)&&
    // A video report shows its models inside the point cloud they were built from: with point_cloud switched on, the model view keeps the cloud.
    (!layers.modelOnly||layers.point_cloud===true&&representation.kind==='point_cloud'||!entity.sourceContext&&(['generated_mesh','primitive'].includes(representation.kind)||reference))&&
    (!layers.representationIds||layers.representationIds.includes(representation.id))&&
    representationAvailable(entity,representation,frameId,!!layers.showCandidates)&&(reference||representationInPhoto(entity,representation,layers.imageId,layers.observations,entity.id===layers.observationEntityId?layers.observationId:undefined));
  const visible=available&&(reference||layers[representation.kind]!==false);
  const cloudOnly=layers.point_cloud!==false&&['observed_surface','generated_mesh','primitive'].every(kind=>layers[kind]===false);
  // Pick actual observed triangles in a cloud view, never invisible generated
  // geometry or a bounding-box proxy. Visible context may occlude, but has ID 0.
  return {available,visible,pick:visible||available&&cloudOnly&&!entity.sourceContext&&representation.kind==='observed_surface'&&representation.placementState==='confirmed',selectable:available&&!entity.sourceContext};
}

// One repaint per browser frame; picking and preview capture explicitly flush.
export function framePaint(paint:()=>void,request:(cb:()=>void)=>number,cancel:(id:number)=>void){
  let pending:number|null=null;
  return {request(){if(pending===null)pending=request(()=>{pending=null;paint();});},cancel(){if(pending!==null)cancel(pending);pending=null;}};
}

export function sceneRepresentationTasks(document:SceneDocument,frameId:string|null,layers:any){
  const state={...layers,observations:document.observations,loading:true};
  return document.entities.flatMap(e=>(e.representations||[]).filter(r=>{
    const pass=representationPass(e,r,frameId,state);
    return (!['generated_mesh','primitive'].includes(r.kind)||r.id===e.activeModelRepresentationId)&&(pass.visible||pass.pick);
  }).map(r=>({e,r})));
}
const taskKey=(tasks:ReturnType<typeof sceneRepresentationTasks>)=>JSON.stringify(tasks.map(({e,r})=>[e.id,r.id,r.assetId]));
/** What a loaded representation's GPU buffers were made from; the same key in the next document reuses them. */
export const representationKey=(entityId:string,r:any)=>JSON.stringify([entityId,r.id,r.kind,r.assetId??null,r.primitive??null]);

export function selectionGeometry(document:SceneDocument,entity:any,frameId:string|null,layers:any,preview?:Transform) {
  if(!entity||entity.sourceContext||entity.visible===false||!frameId||isReferenceSurface(document,entity)||(entity.representations||[]).some((rep:any)=>isCurrentReferenceSurface(entity,rep,document.observations)))return {corners:[] as Vec[],transform:undefined,axisSpace:'native',editable:false};
  const cloudOnly=layers.point_cloud!==false&&['observed_surface','generated_mesh','primitive'].every(kind=>layers[kind]===false);
  const layer:GeometryLayer=cloudOnly?'point_cloud':layers.generated_mesh!==false||layers.primitive!==false?'model':'observed_surface';
  const subject=preview?{...entity,currentModelTransform:preview}:entity;
  const options={layer,frameId,showCandidates:!!layers.showCandidates,imageId:layers.imageId,observations:document.observations,observationId:entity.id===layers.observationEntityId?layers.observationId:undefined};
  const geometry=layer==='model'?modelFamilyGeometry(preview?{...document,entities:document.entities.map(e=>e.id===subject.id?subject:e)}:document,subject.id,options):entityGeometryForLayer(subject,options);
  return geometry?{...geometry,editable:geometry.geometryKind==='model'&&layers.editable!==false}:{corners:[] as Vec[],transform:undefined,axisSpace:'native',editable:false};
}

export function readPacked(buffer:ArrayBuffer,metadata:any):Mesh[] {
  const layout=metadata.byteLayout||metadata.metadata?.byteLayout;
  if(!layout||layout.stride!==9||layout.indexType!=='uint32')throw Error('unsupported_packed_mesh');
  const {byteOffset=0,vertexCount,indexByteOffset,indexCount}=layout;
  if(![byteOffset,vertexCount,indexByteOffset,indexCount].every(n=>Number.isSafeInteger(n)&&n>=0)||byteOffset%4||indexByteOffset%4||byteOffset+vertexCount*36>indexByteOffset||indexByteOffset+indexCount*4>buffer.byteLength||indexCount%3)throw Error('invalid_packed_mesh');
  const source=new Float32Array(buffer,byteOffset,vertexCount*9),indices=new Uint32Array(buffer,indexByteOffset,indexCount).slice(),vertices=new Float32Array(vertexCount*12),bounds={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};
  if(source.some(n=>!Number.isFinite(n))||indices.some(n=>n>=vertexCount))throw Error('invalid_packed_mesh_data');
  for(let i=0;i<vertexCount;i++){vertices.set(source.subarray(i*9,i*9+9),i*12);vertices[i*12+11]=1;}
  for(const i of indices)for(let k=0;k<3;k++){bounds.min[k]=Math.min(bounds.min[k],source[i*9+k]);bounds.max[k]=Math.max(bounds.max[k],source[i*9+k]);}
  return [{vertices,indices,mode:4,matrix:identity(),bounds}];
}

// Deliberately supports the uncompressed GLB mesh/point subset our workers emit.
// Skinning, morph targets and compression need a validated importer, never silent omission.
export function readGLB(buffer:ArrayBuffer):Mesh[] {
  const d=new DataView(buffer);if(buffer.byteLength<20||d.getUint32(0,true)!==0x46546c67||d.getUint32(4,true)!==2||d.getUint32(8,true)!==buffer.byteLength)throw Error('invalid_glb');
  let doc:any,binOffset=0,binLength=0;
  for(let o=12;o<buffer.byteLength;){if(o+8>buffer.byteLength)throw Error('invalid_glb_chunk');const n=d.getUint32(o,true),t=d.getUint32(o+4,true);o+=8;if(o+n>buffer.byteLength)throw Error('invalid_glb_chunk');if(t===0x4e4f534a)doc=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,o,n)));else if(t===0x004e4942){binOffset=o;binLength=n;}o+=n;}
  if(!doc||!binOffset||doc.buffers?.length!==1||doc.buffers[0].uri||doc.buffers[0].byteLength>binLength||(doc.extensionsRequired||[]).some((e:string)=>e!=='KHR_materials_unlit'))throw Error('unsupported_glb');
  function view(id:number){const v=doc.bufferViews?.[id];if(!v||v.buffer!==0||(v.byteOffset||0)<0||v.byteLength<0||(v.byteOffset||0)+v.byteLength>doc.buffers[0].byteLength)throw Error('invalid_glb_view');return v;}
  function accessor(id:number):{data:number[];width:number;count:number}{
    const a=doc.accessors?.[id],sizes:any={SCALAR:1,VEC2:2,VEC3:3,VEC4:4},bytes:any={5120:1,5121:1,5122:2,5123:2,5125:4,5126:4};
    if(!a||a.sparse||!sizes[a.type]||!bytes[a.componentType]||!Number.isSafeInteger(a.count)||a.count<0||a.count>20000000)throw Error('invalid_glb_accessor');
    const v=view(a.bufferView),width=sizes[a.type],b=bytes[a.componentType],stride=v.byteStride||width*b,offset=a.byteOffset||0;
    if(offset<0||stride<width*b||offset+(a.count?((a.count-1)*stride+width*b):0)>v.byteLength)throw Error('invalid_glb_accessor_range');
    const result:number[]=[];for(let n=0;n<a.count;n++)for(let k=0;k<width;k++){const i=binOffset+(v.byteOffset||0)+offset+n*stride+k*b;let x:number;
      switch(a.componentType){case 5120:x=d.getInt8(i);break;case 5121:x=d.getUint8(i);break;case 5122:x=d.getInt16(i,true);break;case 5123:x=d.getUint16(i,true);break;case 5125:x=d.getUint32(i,true);break;default:x=d.getFloat32(i,true);}
      if(a.normalized){if(a.componentType===5121)x/=255;else if(a.componentType===5123)x/=65535;else if(a.componentType===5120)x=Math.max(-1,x/127);else if(a.componentType===5122)x=Math.max(-1,x/32767);else throw Error('invalid_normalized_accessor');}if(!Number.isFinite(x))throw Error('nonfinite_glb');result.push(x);}
    return {data:result,width,count:a.count};
  }
  const result:Mesh[]=[],visiting=new Set<number>();
  function walk(id:number,parent:ArrayLike<number>){
    if(visiting.has(id))throw Error('glb_node_cycle');visiting.add(id);const node=doc.nodes?.[id];if(!node||node.skin!==undefined)throw Error('unsupported_glb_node');
    const local=node.matrix||transformMatrix({position:node.translation||[0,0,0],quaternion:node.rotation||[0,0,0,1],scale:node.scale||[1,1,1]});if(local.length!==16||!local.every(Number.isFinite))throw Error('invalid_glb_transform');const matrix=matmul(parent,local);
    if(node.mesh!==undefined)for(const p of doc.meshes[node.mesh]?.primitives||[]){
      if(![0,4].includes(p.mode??4)||p.targets||p.extensions)throw Error('unsupported_glb_primitive');
      const pos=accessor(p.attributes.POSITION);if(pos.width!==3)throw Error('invalid_glb_position');const normal=p.attributes.NORMAL===undefined?null:accessor(p.attributes.NORMAL),color=p.attributes.COLOR_0===undefined?null:accessor(p.attributes.COLOR_0),uv=p.attributes.TEXCOORD_0===undefined?null:accessor(p.attributes.TEXCOORD_0);
      if([normal,color,uv].some(a=>a&&a.count!==pos.count))throw Error('glb_attribute_count');
      const ix=p.indices===undefined?{data:Array.from({length:pos.count},(_,i)=>i)}:accessor(p.indices),indices=new Uint32Array(ix.data);
      if(ix.data.some(i=>!Number.isInteger(i)||i<0||i>=pos.count)||(p.mode??4)===4&&indices.length%3)throw Error('invalid_glb_indices');
      const material=doc.materials?.[p.material],pbr=material?.pbrMetallicRoughness,factor=pbr?.baseColorFactor||[1,1,1,1],alphaMode=material?.alphaMode||'OPAQUE',alphaCutoff=material?.alphaCutoff??.5,data=new Float32Array(pos.count*12),bounds={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};
      if(!['OPAQUE','MASK','BLEND'].includes(alphaMode)||!Number.isFinite(alphaCutoff)||alphaCutoff<0||!Array.isArray(factor)||factor.length!==4||factor.some((n:number)=>!Number.isFinite(n)||n<0||n>1)||color&&![3,4].includes(color.width))throw Error('invalid_glb_material');
      for(let i=0;i<pos.count;i++){for(let k=0;k<3;k++){data[i*12+k]=pos.data[i*3+k];data[i*12+3+k]=normal?.data[i*3+k]??0;data[i*12+6+k]=color?.data[i*color.width+k]??1;}data[i*12+9]=uv?.data[i*2]||0;data[i*12+10]=uv?.data[i*2+1]||0;data[i*12+11]=color?.width===4?color.data[i*4+3]:1;}
      for(const i of indices)for(let k=0;k<3;k++){bounds.min[k]=Math.min(bounds.min[k],pos.data[i*3+k]);bounds.max[k]=Math.max(bounds.max[k],pos.data[i*3+k]);}
      if(!normal&&(p.mode??4)===4){
        for(let i=0;i<indices.length;i+=3){const ids=[indices[i],indices[i+1],indices[i+2]],ps=ids.map(n=>pos.data.slice(n*3,n*3+3)),n=cross(add(ps[1],scale(ps[0],-1)),add(ps[2],scale(ps[0],-1)));for(const id of ids)for(let k=0;k<3;k++)data[id*12+3+k]+=n[k];}
        // Area-weighted directions must be unit length, independent of mesh scale.
        for(let i=0;i<pos.count;i++){const offset=i*12+3,length=Math.hypot(data[offset],data[offset+1],data[offset+2]);if(length>0)for(let k=0;k<3;k++)data[offset+k]/=length;}
      }
      let texture:Blob|undefined;const textureIndex=pbr?.baseColorTexture?.index;
      if(textureIndex!==undefined){if(!uv||pbr.baseColorTexture.texCoord||pbr.baseColorTexture.extensions)throw Error('unsupported_texture_coordinates');const img=doc.images?.[doc.textures?.[textureIndex]?.source];if(!img||img.uri||!['image/png','image/jpeg'].includes(img.mimeType))throw Error('unsupported_glb_texture');const v=view(img.bufferView);texture=new Blob([buffer.slice(binOffset+(v.byteOffset||0),binOffset+(v.byteOffset||0)+v.byteLength)],{type:img.mimeType});}
      result.push({vertices:data,indices,mode:p.mode??4,matrix,texture,material:{baseColorFactor:factor,alphaMode,alphaCutoff},bounds,name:node.name});
    }
    for(const child of node.children||[])walk(child,matrix);visiting.delete(id);
  }
  const scene=doc.scenes?.[doc.scene??0];if(!scene)throw Error('missing_glb_scene');for(const id of scene.nodes||[])walk(id,identity());return result;
}

export function primitive(spec:any):Mesh {
  const p=spec.parameters||spec,type=spec.kind||spec.primitiveType||spec.type;const vertices:number[]=[],indices:number[]=[];
  if(type==='box'){const dims=p.dimensions;if(!Array.isArray(dims)||dims.length!==3)throw Error('invalid_primitive');if(!dims.every(v=>Number.isFinite(v)&&v>0))throw Error('invalid_primitive');for(const c of boundsCorners({min:dims.map(v=>-v/2),max:dims.map(v=>v/2)}))vertices.push(...c,0,0,1,1,1,1,0,0,1);indices.push(0,2,1,1,2,3,4,5,6,5,7,6,0,1,4,1,5,4,2,6,3,3,6,7,0,4,2,2,4,6,1,3,5,3,7,5);}
  else if(type==='cylinder'){const r=p.radius,h=p.height,n=p.segments??64;if(!(Number.isFinite(r)&&r>0&&Number.isFinite(h)&&h>0&&Number.isInteger(n)&&n>=8&&n<=256))throw Error('invalid_primitive');for(let z=0;z<2;z++)for(let i=0;i<n;i++){const a=i*2*Math.PI/n;vertices.push(r*Math.cos(a),r*Math.sin(a),(z-.5)*h,Math.cos(a),Math.sin(a),0,1,1,1,0,0,1);}vertices.push(0,0,-h/2,0,0,-1,1,1,1,0,0,1,0,0,h/2,0,0,1,1,1,1,0,0,1);for(let i=0;i<n;i++){const j=(i+1)%n;indices.push(i,j,n+j,i,n+j,n+i,2*n,j,i,2*n+1,n+i,n+j);}}
  // r4 (models): a card's display model: boxes (one, or an open frame's parts) and cylinders whose unseen faces are fainter
  else if(type==='faces'){const ok=(a:any,n:number)=>Array.isArray(a)&&a.length===n&&a.every((v:any)=>Number.isFinite(v));const parts=p.parts;if(!Array.isArray(parts)||!parts.length||parts.length>64)throw Error('invalid_primitive');
    for(const q of parts){if(!ok(q.center,3)||!ok(q.size,3)||!q.size.every((v:number)=>v>0)||!ok(q.alpha,6))throw Error('invalid_primitive');for(let f=0;f<6;f++){const k=f>>1,s=f&1?1:-1,[a,b]=[0,1,2].filter(i=>i!==k),base=vertices.length/12;
      for(const [x,y] of [[-1,-1],[1,-1],[1,1],[-1,1]]){const v=[...q.center],n=[0,0,0];v[k]+=s*q.size[k]/2;v[a]+=x*q.size[a]/2;v[b]+=y*q.size[b]/2;n[k]=s;vertices.push(...v,...n,1,1,1,0,0,q.alpha[f]);}indices.push(base,base+1,base+2,base,base+2,base+3);}}}
  else if(type==='sectors'){const r=p.radius,h=p.height,arc=p.arc;if(!(Number.isFinite(r)&&r>0&&Number.isFinite(h)&&h>0&&typeof arc==='string'&&arc.length>=8&&arc.length<=256&&Array.isArray(p.alpha)&&Array.isArray(p.capAlpha)))throw Error('invalid_primitive');const n=arc.length;
    for(let i=0;i<n;i++){const al=arc[i]==='1'?p.alpha[0]:p.alpha[1],base=vertices.length/12;for(const [j,z] of [[i,-1],[i+1,-1],[i+1,1],[i,1]]){const t=j*2*Math.PI/n;vertices.push(r*Math.cos(t),r*Math.sin(t),z*h/2,Math.cos(t),Math.sin(t),0,1,1,1,0,0,al);}indices.push(base,base+1,base+2,base,base+2,base+3);
      for(const [z,al2] of [[-1,p.capAlpha[0]],[1,p.capAlpha[1]]]){const b2=vertices.length/12;for(const j of [i,i+1])vertices.push(r*Math.cos(j*2*Math.PI/n),r*Math.sin(j*2*Math.PI/n),z*h/2,0,0,z,1,1,1,0,0,al2);vertices.push(0,0,z*h/2,0,0,z,1,1,1,0,0,al2);indices.push(b2,b2+1,b2+2);}}}
  else throw Error('unsupported_primitive');
  const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];for(let i=0;i<vertices.length;i+=12)for(let k=0;k<3;k++){min[k]=Math.min(min[k],vertices[i+k]);max[k]=Math.max(max[k],vertices[i+k]);}
  return {vertices:new Float32Array(vertices),indices:new Uint32Array(indices),mode:4,matrix:identity(),bounds:{min,max}};
}

export function mountSceneViewer(container:HTMLElement,options:ViewerOptions){
  const stage=document.createElement('div');stage.className='native-stage';Object.assign(stage.style,{position:'relative',width:'100%',height:'100%',minHeight:'260px',display:'flex',alignItems:'center',justifyContent:'center',overflow:'hidden',background:'#111b21'});
  const photo=document.createElement('img');photo.alt='';Object.assign(photo.style,{position:'absolute',objectFit:'contain',pointerEvents:'none'});photo.hidden=true;
  const canvas=document.createElement('canvas');canvas.setAttribute('aria-label',options.locale==='en'?'Interactive scene':'交互场景');canvas.tabIndex=0;Object.assign(canvas.style,{position:'relative',touchAction:'none'});
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');Object.assign(svg.style,{position:'absolute',inset:'0',width:'100%',height:'100%',pointerEvents:'none'});stage.append(photo,canvas,svg);container.append(stage);
  const gl=canvas.getContext('webgl2',{alpha:true,antialias:true,preserveDrawingBuffer:true});if(!gl){stage.remove();throw Error('webgl_unavailable');}
  let disposed=false,epoch=0,photoEpoch=0,doc:any={entities:[],cameras:[],coordinateFrames:[]},revisionId='',selection:any={},camera:Camera|null=null,radius=1,center:Vec=[0,0,0],frameId:string|null=null,gpu:GPU[]=[],stale=new Set<GPU>(),abort=new AbortController(),preview=new Map<string,Transform>(),drag:any=null,axisDrag:any=null;
  let sceneAssetsSignature='',loadedLayerKey='',viewMode='free',backgroundOnly=false;let pickCursor:Vec|null=null;let navigationVersion=0,autoFit=true;let photoAbort=new AbortController(),photoObjectURL:string|null=null;
  let captureSize:{w:number;h:number;cw:number;ch:number}|null=null;let pathCache:{doc:any;frameId:string|null;path:ReturnType<typeof cameraPath>}={doc:null,frameId:null,path:[]};const streamedMeshes=new Map<string,Mesh>();const loadedRepresentations=new Set<string>(),assetStates=new Map<string,RepresentationLoadState>();
  let layers:any={observed_surface:true,generated_mesh:true,primitive:true,point_cloud:true,allBounds:false,opacity:.65,lighting:true,showCandidates:false,...options.layers};const cleanups:(()=>void)[]=[];
  let splat:ReturnType<typeof createSplatLayer>|null=null,splatSource:SplatSource|null=null;
  const emit=(type:string,payload:any={})=>{if(!disposed)options.onEvent?.({type,...payload});};
  const program=gl.createProgram()!;const shaders:WebGLShader[]=[];
  for(const [type,code]of [[gl.VERTEX_SHADER,`#version 300 es\nin vec3 p;in vec3 n;in vec3 c;in vec2 uv;in float a;uniform mat4 vp;uniform mat4 model;uniform float pointSize;uniform float pointWorld;uniform float pointScale;out float pointPixels;out vec3 color;out vec3 normal;out vec2 tex;out float vertexAlpha;out vec3 world;void main(){vec4 w=model*vec4(p,1.);world=w.xyz;gl_Position=vp*w;gl_PointSize=pointPixels=pointWorld>0.?clamp(pointWorld*pointScale/gl_Position.w,1.,16.):pointSize;color=c;normal=transpose(inverse(mat3(model)))*n;tex=uv;vertexAlpha=a;}`],[gl.FRAGMENT_SHADER,`#version 300 es\nprecision highp float;in vec3 color;in vec3 normal;in vec2 tex;in float vertexAlpha;uniform sampler2D image;uniform vec4 baseColorFactor;uniform float alphaMode;uniform float alphaCutoff;uniform vec3 pickColor;uniform vec3 tint;uniform float selected;uniform float pick;uniform float opacity;uniform float lighting;uniform float pointWorld;uniform vec4 ceiling;in float pointPixels;in vec3 world;out vec4 outColor;void main(){if(dot(ceiling.xyz,ceiling.xyz)>0.&&dot(ceiling.xyz,world)+ceiling.w>0.)discard;if(pointWorld>0.&&length(gl_PointCoord-.5)>.5+.5/pointPixels)discard;vec4 sampleColor=texture(image,tex);float alpha=vertexAlpha*sampleColor.a*baseColorFactor.a;if(alphaMode<.5)alpha=1.;else if(alphaMode<1.5){if(alpha<alphaCutoff)discard;alpha=1.;}else if(alpha<=0.)discard;if(pick>.5){outColor=vec4(pickColor,1.);return;}vec3 c=color*sampleColor.rgb*baseColorFactor.rgb*tint;float n=length(normal);if(lighting>1.5){c=pow(clamp(c,0.,1.),vec3(.72));if(n>.01){vec3 direction=normal/n;float key=abs(dot(direction,normalize(vec3(.3,.8,.6))));float fill=abs(dot(direction,normalize(vec3(-.7,.4,-.5))));c=c*(.78+.15*key+.07*fill)+vec3(.025*pow(key,16.));}}else if(lighting>.5&&n>.01)c*=.5+.5*abs(dot(normal/n,normalize(vec3(.3,.8,.6))));if(selected>.5)c=mix(c,vec3(.28,.86,.75),.4);outColor=vec4(c,alpha*opacity);}`]]as[number,string][]){const s=gl.createShader(type)!;gl.shaderSource(s,code);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s)||'shader_error');gl.attachShader(program,s);shaders.push(s);}
  gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error('shader_link_error');gl.useProgram(program);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);
  const u=Object.fromEntries(['vp','model','pointSize','pointWorld','pointScale','ceiling','image','pickColor','selected','pick','opacity','lighting','tint','baseColorFactor','alphaMode','alphaCutoff'].map(n=>[n,gl.getUniformLocation(program,n)]));
  const attrs=['p','n','c','uv','a'].map(n=>gl.getAttribLocation(program,n));for(const a of attrs)gl.enableVertexAttribArray(a);gl.uniform1i(u.image,0);
  function releaseMesh(g:GPU){gl!.deleteBuffer(g.vertex);gl!.deleteBuffer(g.index);gl!.deleteTexture(g.texture);}
  function release(){gpu.forEach(releaseMesh);gpu=[];stale.clear();loadedRepresentations.clear();assetStates.clear();}
  // The previous content of a representation, drawn until its new asset is loaded (or failed), then released.
  function replaced(entityId:string,representationId:string){const old=gpu.filter(g=>stale.has(g)&&g.entityId===entityId&&g.representation.id===representationId);old.forEach(g=>{releaseMesh(g);stale.delete(g);});if(old.length)gpu=gpu.filter(g=>!old.includes(g));}
  // by id, rebuilt when the document changes (mvp2: a find() per call made fittingPoints O(n^2): 1.3 s of main thread at 1082
  // objects when densify's layers landed, and every click and pick decode waited behind it); the first entity of an id wins, as find()
  let byId=new Map<string,[any,number]>(),byIdOf:any=null;
  function indexed(id:string){if(byIdOf!==doc){byId=new Map();doc.entities.forEach((e:any,i:number)=>{if(!byId.has(e.id))byId.set(e.id,[e,i]);});byIdOf=doc;}return byId.get(id);}
  function entity(id:string){return indexed(id)?.[0];}
  // One fit + draw per frame while representations stream in (mvp2: per representation, dimensions() over every object made
  // 1082 primitive boxes an O(n^2) 0.8 s task, the pick layer's decode and the user's clicks waiting behind it)
  let refit=0;
  function refitSoon(){if(refit)return;const run=()=>{refit=0;if(!disposed){dimensions();draw();}};refit=typeof requestAnimationFrame==='function'?requestAnimationFrame(run):setTimeout(run,0) as any;}
  function model(g:GPU){const e=entity(g.entityId),t=g.representation.kind==='observed_surface'||g.representation.kind==='point_cloud'?g.representation.transform:preview.get(g.entityId)||e?.currentModelTransform||g.representation.transform;return matmul(transformMatrix(t),g.mesh.matrix);}
  // A point cloud drawn at its world size (round, 1-16 device px) when it declares its cell size in native units, on the
  // representation or its asset's metadata: no gaps up close, no blobs far away. Without it, today's fixed-size squares.
  function pointWorld(g:GPU){const asset=doc.assets?.find((a:any)=>a.id===g.representation.assetId),size=g.representation.pointSizeNative??asset?.pointSizeNative??asset?.metadata?.pointSizeNative;return Number.isFinite(size)&&size>0?size:0;}
  // Splats fitted to one walk are photo-real only near where its camera stood, looking roughly its way; farther off they
  // streak, so there the scene shows its points, meshes and models, and on the path a drag turns the view, not the eye.
  // Near = within 0.4 m (3% of the scene radius without a scale) of a report camera (on ME340 they stand 2-7 cm apart), so
  // a video-following eye between them stays near. With no cameras there is no path to leave. ponytail: fixed 0.4 m and 45
  // degrees, a linear scan of the report's cameras; tune per report if needed.
  let pathCameras:{doc:any;frameId:string|null;list:{eye:Vec;forward:Vec}[]}={doc:null,frameId:null,list:[]};
  function splatsActive(){return !!splat&&!splat.failed&&layers.splats===true&&!layers.studio&&splatSource?.coordinateFrameId===frameId;}
  // A length in metres in native units, or a share of the scene radius when the report has no scale.
  function nativeFor(metres:number,share:number){const perNative=Number(doc.coordinateFrames?.find((f:any)=>f.id===frameId)?.scale?.nativeToMeters);return perNative>0?metres/perNative:share*radius;}
  function walked(){
    if(pathCameras.doc!==doc||pathCameras.frameId!==frameId)pathCameras={doc,frameId,list:currentCameras(doc).filter((c:any)=>c.coordinateFrameId===frameId&&Array.isArray(c.cameraToWorld)).map((c:any)=>({eye:[0,1,2].map(k=>Number(c.cameraToWorld[k][3])),forward:[0,1,2].map(k=>Number(c.cameraToWorld[k][2]))}))};
    return pathCameras.list;
  }
  function nearPath(facing:boolean){
    if(!camera)return false;const list=walked();
    return !list.length||nearCameras(list,camera.eye,facing?unit(add(camera.target,scale(camera.eye,-1))):null,nativeFor(.4,.03));
  }
  // From above, the ceiling would hide the room: cut away what was seen above it (ceilingCut), never a model, the selected
  // object, a preview or the studio view.
  function ceiling(){
    const ground=doc.coordinateFrames?.find((f:any)=>f.id===frameId)?.ground;
    return !camera||camera.exact||!Array.isArray(ground?.normal)?[0,0,0,0]:ceilingCut(ground.normal,Number(ground.offset),walked().map(c=>c.eye),camera.eye);
  }
  function visible(g:GPU){return representationPass(entity(g.entityId),g.representation,frameId,layers).visible;}
  function corners(id?:string){return gpu.filter(g=>(!id||g.entityId===id)&&visible(g)).flatMap(g=>boundsCorners(g.mesh.bounds).map(p=>point(model(g),p)));}
  function selectedGeometry(id:string){const scene=preview.size?{...doc,entities:doc.entities.map((e:any)=>preview.has(e.id)?{...e,currentModelTransform:preview.get(e.id)}:e)}:doc;return selectionGeometry(scene,entity(id),frameId,layers,preview.get(id));}
  function previewTransform(id:string,value:Transform){for(const {entity,transform}of modelFamilyTransforms(doc,id,value as import('../types').Transform))preview.set(entity.id,transform);}
  function selectedAxes(id:string){
    const geometry=selectedGeometry(id),ps=geometry.corners;if(!ps.length)return null;
    const origin=[0,1,2].map(k=>ps.reduce((sum:number,p:Vec)=>sum+p[k],0)/ps.length),m=geometry.axisSpace==='native'?identity():transformMatrix(geometry.transform),length=radius*(layers.studio?.48:.16);
    return {geometry,origin,length,axes:[0,1,2].map(k=>{const direction=unit([m[k*4],m[k*4+1],m[k*4+2]]);return {direction,end:add(origin,scale(direction,length)),label:'XYZ'[k],color:(layers.studio?['#c23f3b','#247545','#2861af']:['#ff6b6b','#65de96','#6bb3ff'])[k]};})};
  }
  function fittingPoints(){const objects=gpu.filter(g=>visible(g)&&!entity(g.entityId)?.sourceContext);return objects.length?objects.flatMap(g=>boundsCorners(g.mesh.bounds).map(p=>point(model(g),p))):corners();}
  function dimensions(){const ps=fittingPoints();if(!ps.length)return;const min=[0,1,2].map(k=>Math.min(...ps.map(p=>p[k]))),max=[0,1,2].map(k=>Math.max(...ps.map(p=>p[k])));center=min.map((v,k)=>(v+max[k])/2);radius=Math.max(Math.hypot(...max.map((v,k)=>v-min[k]))/2,1e-4);}
  function viewSize(){if(captureSize)return captureSize;const w=stage.clientWidth,h=stage.clientHeight,ratio=camera?.exact?camera.frame.width/camera.frame.height:w/h,cw=Math.min(w,h*ratio),ch=cw/ratio;return {w,h,cw,ch};}
  const repaint=framePaint(()=>drawNow(),cb=>requestAnimationFrame(cb),id=>cancelAnimationFrame(id));
  function draw(pick=false,captureCanvas?:HTMLCanvasElement){
    if(pick||captureCanvas){repaint.cancel();drawNow(pick,captureCanvas);}else repaint.request();
  }
  function drawNow(pick=false,captureCanvas?:HTMLCanvasElement){
    if(disposed||!camera||gl!.isContextLost())return;const {w,h,cw,ch}=viewSize();if(!(cw>0&&ch>0))return;const dpr=Math.min(devicePixelRatio||1,2);canvas.style.width=photo.style.width=cw+'px';canvas.style.height=photo.style.height=ch+'px';const pw=Math.max(1,Math.round(cw*dpr)),ph=Math.max(1,Math.round(ch*dpr));if(canvas.width!==pw||canvas.height!==ph){canvas.width=pw;canvas.height=ph;}
    const background=layers.studio?[237/255,240/255,238/255]:[17/255,27/255,33/255];
    gl!.viewport(0,0,pw,ph);gl!.clearColor(pick||camera.exact?0:background[0],pick||camera.exact?0:background[1],pick||camera.exact?0:background[2],camera.exact&&!pick?0:1);gl!.depthMask(true);gl!.clear(gl!.COLOR_BUFFER_BIT|gl!.DEPTH_BUFFER_BIT);
    const {projection,view}=cameraMatrices(camera,cw/ch,radius),vp=matmul(projection,view);
    // Photo-real splats (never over the source photo) are laid over the photo-coloured observed surfaces: the room's, and
    // every object's but the selected one's (moving objects keep theirs). Where the splats are thin those surfaces show
    // through, so a view near the path has no holes; they write depth, so what lies behind them stays hidden, and take
    // picks. Models, the selection and moving objects are drawn after the splats, on top.
    const splatsOn=splatsActive()&&!(camera.exact&&options.showSourcePhoto!==false)&&nearPath(true)&&!pick&&!captureCanvas;
    const under=(g:GPU)=>splatsOn&&g.mesh.mode===4&&g.representation.kind==='observed_surface'&&!g.representation.timeRange&&(!!entity(g.entityId)?.sourceContext||!isSelected(g));
    let splatted=false;const splatsNow=()=>{if(splatsOn&&!splatted){splatted=true;splat!.draw(view,projection,pw,ph);gl!.useProgram(program);}};
    gl!.useProgram(program);gl!.uniformMatrix4fv(u.vp,false,vp);gl!.uniform1f(u.pointSize,layers.pointSize?layers.pointSize*dpr:2);gl!.uniform1f(u.pointScale,projection[5]*ph/2);const cut=captureCanvas||layers.studio?null:ceiling();gl!.uniform1f(u.pick,pick?1:0);gl!.uniform1f(u.opacity,camera.exact&&!pick?layers.opacity:1);gl!.uniform1f(u.lighting,layers.studio?2:layers.lighting?1:0);
    // A selected observed surface can share its exact depth with scene context.
    // Draw it last with equal-depth acceptance; nearer geometry still occludes it.
    // ponytail: same-grid observed subsets use triangle count only as an
    // equal-depth interaction priority, not as evidence of object identity.
    // Unequal sampling would need source-mask area; nearer depth always wins.
    const pickRank=(g:GPU)=>entity(g.entityId)?.sourceContext?0:g.mesh.mode===4&&g.representation.kind==='observed_surface'&&g.representation.placementState==='confirmed'?2:1;
    const selectedIds=new Set(selection.entityId?modelPreviewEntities(doc,selection.entityId).map(e=>e.id):[]),isSelected=(g:GPU)=>['generated_mesh','primitive'].includes(g.representation.kind)?selectedIds.has(g.entityId):g.entityId===selection.entityId;
    const materialFor=(g:GPU)=>{const m={...g.mesh.material,...(['generated_mesh','primitive'].includes(g.representation.kind)?g.representation.material:{})};return m.selectedFactor&&(isSelected(g)||layers.studio)?{...m,baseColorFactor:m.selectedFactor}:m;};  // r4: a model shows in full when its object is selected (and in its preview)
    const blended=(g:GPU)=>materialFor(g).alphaMode==='BLEND';
    const depth=(g:GPU)=>dot(add(point(model(g),g.mesh.bounds.min.map((n,k)=>(n+g.mesh.bounds.max[k])/2)),scale(camera!.eye,-1)),unit(add(camera!.target,scale(camera!.eye,-1))));
    // ponytail: primitive-depth sorting covers separate sheets; intersecting translucent geometry needs per-triangle sorting or order-independent transparency.
    const drawing=pick?gpu.slice().sort((a,b)=>pickRank(a)-pickRank(b)||(pickRank(a)===2?b.mesh.indices.length-a.mesh.indices.length:0)||a.entityId.localeCompare(b.entityId)||a.representation.id.localeCompare(b.representation.id)):gpu.filter(under).concat(gpu.filter(g=>!under(g)&&!blended(g)&&!isSelected(g)),gpu.filter(g=>!under(g)&&!blended(g)&&isSelected(g)),gpu.filter(g=>!under(g)&&blended(g)).sort((a,b)=>depth(b)-depth(a)||Number(isSelected(a))-Number(isSelected(b))||a.entityId.localeCompare(b.entityId)));
    for(const g of drawing){const beneath=under(g);if(!beneath)splatsNow();const pass=representationPass(entity(g.entityId),g.representation,frameId,layers),selected=pass.selectable&&isSelected(g);if(!(pick?pass.pick:pass.visible))continue;const material=materialFor(g),blend=material.alphaMode==='BLEND';if(beneath){gl!.enable(gl!.POLYGON_OFFSET_FILL);gl!.polygonOffset(1,1);}else gl!.disable(gl!.POLYGON_OFFSET_FILL);gl!.depthMask(pick||!blend);if(!pick&&(blend||camera.exact)){gl!.enable(gl!.BLEND);gl!.blendFuncSeparate(gl!.SRC_ALPHA,gl!.ONE_MINUS_SRC_ALPHA,gl!.ONE,gl!.ONE_MINUS_SRC_ALPHA);}else gl!.disable(gl!.BLEND);gl!.depthFunc(pick||selected?gl!.LEQUAL:gl!.LESS);const id=pass.selectable?(indexed(g.entityId)?.[1]??-1)+1:0;gl!.bindBuffer(gl!.ARRAY_BUFFER,g.vertex);gl!.bindBuffer(gl!.ELEMENT_ARRAY_BUFFER,g.index);attrs.forEach((a,k)=>gl!.vertexAttribPointer(a,k===4?1:k===3?2:3,gl!.FLOAT,false,48,k===4?44:k*12));gl!.activeTexture(gl!.TEXTURE0);gl!.bindTexture(gl!.TEXTURE_2D,g.texture);gl!.uniformMatrix4fv(u.model,false,model(g));gl!.uniform1f(u.pointWorld,g.mesh.mode===0?pointWorld(g):0);gl!.uniform4fv(u.ceiling,cut&&(g.representation.kind==='observed_surface'||g.representation.kind==='point_cloud')&&!isSelected(g)?cut:[0,0,0,0]);gl!.uniform1f(u.lighting,layers.studio?2:layers.lighting||g.representation.material?.lighting?1:0);gl!.uniform1f(u.selected,selected?1:0);gl!.uniform3f(u.pickColor,(id&255)/255,((id>>8)&255)/255,((id>>16)&255)/255);gl!.uniform4fv(u.baseColorFactor,material.baseColorFactor||[1,1,1,1]);gl!.uniform1f(u.alphaMode,blend?2:material.alphaMode==='MASK'?1:0);gl!.uniform1f(u.alphaCutoff,material.alphaCutoff??.5);const mat=material.color;gl!.uniform3fv(u.tint,Array.isArray(mat)&&mat.length>=3?mat.slice(0,3):[1,1,1]);gl!.drawElements(g.mesh.mode===0?gl!.POINTS:gl!.TRIANGLES,g.mesh.indices.length,gl!.UNSIGNED_INT,0);}
    splatsNow();gl!.depthMask(true);gl!.disable(gl!.POLYGON_OFFSET_FILL);
    if(pick)return;const overlay=captureCanvas?.getContext('2d');if(overlay&&captureCanvas){captureCanvas.width=canvas.width;captureCanvas.height=canvas.height;overlay.drawImage(canvas,0,0);overlay.scale(canvas.width/w,canvas.height/h);}svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.replaceChildren();const project=(p:Vec)=>{const q=projected(vp,p,cw,ch);return q?add(q,[(w-cw)/2,(h-ch)/2]):null;};
    const line=(a:Vec|null,b:Vec|null,color:string,width=1.5)=>{if(!a||!b)return null;const el=document.createElementNS(svg.namespaceURI,'line');for(const[k,v]of Object.entries({x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:color,'stroke-width':width}))el.setAttribute(k,String(v));svg.append(el);if(overlay){overlay.beginPath();overlay.moveTo(a[0],a[1]);overlay.lineTo(b[0],b[1]);overlay.strokeStyle='#ffffff';overlay.lineWidth=width+2;overlay.stroke();overlay.strokeStyle=color;overlay.lineWidth=width;overlay.stroke();}return el;};
    const measurement=layers.measurement;
    if(measurement?.revisionId===revisionId&&measurement.coordinateFrameId===frameId){
      for(const path of measurement.lines)for(let i=1;i<path.points.length;i++)line(project(path.points[i-1]),project(path.points[i]),path.color,2.5);
      const p=project(measurement.labelPoint);if(p){const text=document.createElementNS(svg.namespaceURI,'text');text.textContent=Number(measurement.value.toPrecision(4))+(measurement.unit==='deg'?'°':'');for(const[k,v]of Object.entries({x:p[0]+8,y:p[1]-8,fill:'#fff',stroke:'#182a31','stroke-width':3,'paint-order':'stroke','font-size':17,'font-weight':700}))text.setAttribute(k,String(v));svg.append(text);if(overlay){overlay.font='700 20px sans-serif';overlay.lineWidth=4;overlay.strokeStyle='#fff';overlay.strokeText(text.textContent,p[0]+8,p[1]-8);overlay.fillStyle='#182a31';overlay.fillText(text.textContent,p[0]+8,p[1]-8);}}
    }
    if(!captureCanvas)for(const [i,pick] of (layers.measurePoints||[]).entries()){
      if(pick.coordinateFrameId!==frameId)continue;const p=project(pick.point);if(!p)continue;
      const circle=document.createElementNS(svg.namespaceURI,'circle');for(const[k,v]of Object.entries({cx:p[0],cy:p[1],r:5,fill:i===(layers.measureVertexIndex??1)?'#edbe38':'#e36b23',stroke:'#fff','stroke-width':2}))circle.setAttribute(k,String(v));svg.append(circle);
      const label=document.createElementNS(svg.namespaceURI,'text');label.textContent=String(i+1);for(const[k,v]of Object.entries({x:p[0]+7,y:p[1]+16,fill:'#fff',stroke:'#182a31','stroke-width':3,'paint-order':'stroke','font-size':15}))label.setAttribute(k,String(v));svg.append(label);
      if(i)line(project(layers.measurePoints[i-1].point),p,'#edbe38',2);
    }
    if(!captureCanvas&&layers.pickingPoints&&pickCursor){const x=(w-cw)/2+pickCursor[0]*cw,y=(h-ch)/2+pickCursor[1]*ch;line([x-8,y],[x+8,y],'#fff',2);line([x,y-8],[x,y+8],'#fff',2);}
    // Where the camera walked, start green, end red, and where it is at the report's video time.
    if(!captureCanvas&&!layers.studio&&layers.cameraPath!==false){
      if(pathCache.doc!==doc||pathCache.frameId!==frameId)pathCache={doc,frameId,path:cameraPath(doc,frameId)};
      // Path points next to the eye (standing on the path) would sweep across the whole view: left out.
      const gap=nativeFor(.6,.05),path=pathCache.path,screen=path.map(p=>Math.hypot(...add(p.position as Vec,scale(camera!.eye,-1)))<gap?null:project(p.position as Vec));
      const marker=(p:Vec|null,fill:string,r:number)=>{if(!p)return;const c=document.createElementNS(svg.namespaceURI,'circle');for(const[k,v]of Object.entries({cx:p[0],cy:p[1],r,fill,stroke:'#fff','stroke-width':2}))c.setAttribute(k,String(v));svg.append(c);};
      if(path.length>1){
        for(let i=1;i<screen.length;i++)line(screen[i-1],screen[i],'#ffb020',2.5);
        marker(screen[0],'#2fbf71',5);marker(screen[screen.length-1],'#e5484d',5);
        if(Number.isFinite(layers.time)&&path[0].time!==null){let now=0;for(let i=1;i<path.length;i++)if(Math.abs(path[i].time!-layers.time)<Math.abs(path[now].time!-layers.time))now=i;marker(screen[now],'#ffb020',7);}
      }
    }
    if(layers.showBounds===false&&!layers.showAxes)return;
    const ids=layers.showBounds===false?[]:layers.allBounds?doc.entities.filter((e:any)=>!e.sourceContext).map((e:any)=>e.id):[selection.entityId];for(const id of ids){if(!id)continue;const ps=selectedGeometry(id).corners;if(!ps.length)continue;const box=ps.map(project);for(let i=0;i<8;i++)for(let k=0;k<3;k++)if(!(i&(1<<k)))line(box[i],box[i|(1<<k)],id===selection.entityId?'#7ae6cf':'#607e89');
      // Names over the boxes (layers.labels): the selected one always, others once their box is 40 px wide on screen.
      // ponytail: a fixed width, no collision layout; a label list is the next step if dense scenes need every name.
      const on=box.filter(Boolean) as Vec[],xs=on.map(p=>p[0]);if(layers.labels&&on.length===8&&(id===selection.entityId||Math.max(...xs)-Math.min(...xs)>=40)){const el=document.createElementNS(svg.namespaceURI,'text');el.textContent=entity(id)?.label||'';for(const[k,v]of Object.entries({x:Math.min(...xs),y:Math.min(...on.map(p=>p[1]))-4,fill:id===selection.entityId?'#7ae6cf':'#e4ece7',stroke:'#111b21','stroke-width':3,'paint-order':'stroke','font-size':12}))el.setAttribute(k,String(v));svg.append(el);}}
    const axisId=layers.axisEntityId||selection.entityId,e=entity(axisId),axes=selectedAxes(axisId);if(e&&axes){const {geometry,origin,length}=axes,t=geometry.transform;
      for(let k=0;k<3;k++){const {direction,end,color}=axes.axes[k],a=project(origin),b=project(end),labelSize=overlay?24:14,labelOffset=overlay?8:5;line(a,b,color,overlay?5:3);if(!a||!b)continue;
        const label=document.createElementNS(svg.namespaceURI,'text');label.textContent='XYZ'[k];for(const[key,value]of Object.entries({x:b[0]+labelOffset,y:b[1]-labelOffset,fill:color,'font-size':labelSize,'font-weight':700}))label.setAttribute(key,String(value));svg.append(label);
        if(overlay){overlay.font=`700 ${labelSize}px sans-serif`;overlay.lineWidth=5;overlay.strokeStyle='#ffffff';overlay.strokeText('XYZ'[k],b[0]+labelOffset,b[1]-labelOffset);overlay.fillStyle=color;overlay.fillText('XYZ'[k],b[0]+labelOffset,b[1]-labelOffset);}
        if(t&&geometry.editable){const handle=line(a,b,'transparent',18)! as SVGElement;handle.style.pointerEvents='stroke';handle.style.cursor='move';handle.setAttribute('role','button');handle.setAttribute('aria-label',`Move ${'XYZ'[k]}`);handle.setAttribute('tabindex','0');handle.addEventListener('pointerdown',(ev:any)=>{ev.preventDefault();ev.stopPropagation();axisDrag={id:e.id,t:structuredClone(t),direction,a,b,length,x:ev.clientX,y:ev.clientY,changed:false};stage.setPointerCapture(ev.pointerId);});handle.addEventListener('keydown',(ev:any)=>{if(['ArrowLeft','ArrowDown','ArrowRight','ArrowUp'].includes(ev.key)){ev.preventDefault();const sign=['ArrowLeft','ArrowDown'].includes(ev.key)?-1:1,nt={...t,position:add(t.position,scale(direction,sign*radius*.01))};emit('transformCommitIntent',{operations:[{type:'setTransform',entityId:e.id,...nt}]});}});}
      }
    }
  }
  async function url(id:string){const r=await options.resolveAsset(id);return typeof r==='string'?r:r.url;}
  // r5b: a shot's observed surfaces come as one GLB (a node per object): fetched and parsed once, each object's representation takes its node
  const glbNodes=new Map<string,Promise<Mesh[]>>();
  function nodeMeshes(id:string){if(!glbNodes.has(id))glbNodes.set(id,url(id).then(src=>fetch(src)).then(res=>{if(!res.ok)throw Error('asset_download_failed');return res.arrayBuffer();}).then(readGLB).catch(e=>{glbNodes.delete(id);throw e;}));return glbNodes.get(id)!;}
  async function setPhoto(){const n=++photoEpoch;photoAbort.abort();photoAbort=new AbortController();photo.hidden=true;if(photoObjectURL){URL.revokeObjectURL(photoObjectURL);photoObjectURL=null;}if(!camera?.exact||options.showSourcePhoto===false)return;const f=camera.frame;let objectURL:string|null=null;try{const src=await url(f.imageId);if(disposed||n!==photoEpoch)return;const response=await fetch(src,{signal:photoAbort.signal});if(!response.ok)throw Error('photo_load_failed');objectURL=URL.createObjectURL(await response.blob());const img=new Image();img.src=objectURL;await img.decode();if(disposed||n!==photoEpoch){URL.revokeObjectURL(objectURL);return;}photoObjectURL=objectURL;photo.src=objectURL;photo.hidden=false;emit('renderReady',{phase:'photo',cameraId:f.id});}catch(error:any){if(objectURL&&objectURL!==photoObjectURL)URL.revokeObjectURL(objectURL);if(n===photoEpoch&&error.name!=='AbortError')emit('loadError',{code:'photo_load_failed',assetId:f.imageId});}}
  function setCamera(value:any){
    if(disposed)return;viewMode=typeof value==='string'?value:value?.mode||'free';const f=currentCameras(doc).find((c:any)=>c.id===(typeof value==='string'?value:value?.cameraId));if(f){frameId=f.coordinateFrameId;layers.imageId=f.imageId;layers.observations=doc.observations;}if(f&&(typeof value==='string'||value?.mode==='photo')){camera=sourceCamera(f,radius,center);setPhoto();draw();return;}
    dimensions();const mode=typeof value==='string'?value:value?.mode||'free';autoFit=!value?.eye;if(value?.eye){navigationVersion++;camera={...value};photo.hidden=true;draw();return;}  // a copy: navigation reassigns eye/target/up and must not rewrite the caller's camera
    camera=fittedCamera(mode);photoEpoch++;photoAbort.abort();photo.hidden=true;draw();
  }
  function fittedCamera(mode:string,sheet?:{subject:any;rep:any}){
    if(!sheet&&layers.entityIds?.length===1){const tasks=sceneRepresentationTasks(doc,frameId,layers);if(tasks.length===1)sheet={subject:tasks[0].e,rep:tasks[0].r};}
    const frame=doc.coordinateFrames.find((f:any)=>f.id===frameId),up=unit(frame?.ground?.normal||[0,0,1]),reference=currentCameras(doc).find((c:any)=>c.coordinateFrameId===frameId),rawFront=reference?reference.cameraToWorld.slice(0,3).map((r:Vec)=>-r[2]):[0,-1,0];let planar=add(rawFront,scale(up,-dot(rawFront,up)));if(Math.hypot(...planar)<1e-6){const axis=Math.abs(up[0])<.8?[1,0,0]:[0,1,0];planar=add(axis,scale(up,-dot(axis,up)));}const front=unit(planar),right=unit(cross(up,front)),ps=fittingPoints();
    let back=mode==='top'?up:mode==='side'?right:mode==='front'?front:unit(add(add(front,scale(right,.45)),scale(up,.55))),vup=mode==='top'?scale(front,-1):up;
    const plane=sheet?.rep.sourceDerivation?.planarModeling?.plane;
    if(mode==='free'&&sheet?.rep.sourceKind==='inferred_planar_surface_from_observed_depth'&&Array.isArray(plane)&&plane.length===4&&plane.every(Number.isFinite)&&Math.hypot(...plane.slice(0,3))>1e-8){
      const t=preview.get(sheet.subject.id)||sheet.subject.currentModelTransform||sheet.rep.transform;
      // Plane normals follow inverse-transpose TRS; part poses are already absolute.
      let normal=unit(point(transformMatrix({...t,position:[0,0,0],scale:[1,1,1]}),plane.slice(0,3).map((n:number,k:number)=>n/t.scale[k])));
      if(dot(normal,rawFront)<0)normal=scale(normal,-1);
      let vertical=add(up,scale(normal,-dot(up,normal)));
      if(Math.hypot(...vertical)<1e-6){const axis=Math.abs(normal[2])<.8?[0,0,1]:[0,1,0];vertical=add(axis,scale(normal,-dot(axis,normal)));}
      vup=unit(vertical);back=unit(add(add(normal,scale(unit(cross(vup,normal)),.25)),scale(vup,.15)));
    }
    if(layers.axisEntityId){const axes=selectedAxes(layers.axisEntityId);if(axes)ps.push(axes.origin,...axes.axes.map(axis=>axis.end));}
    return fitCamera(ps.length?ps:boundsCorners({min:[-1,-1,-1],max:[1,1,1]}),back,vup,Math.max(captureSize?.w||stage.clientWidth,1)/Math.max(captureSize?.h||stage.clientHeight,1),['top','front','side'].includes(mode));
  }
  function capturePreview(entityId:string,mode:'free'|'front'|'side'|'top',expectedRevisionId:string,expectedFrameId?:string,source:{layer:GeometryLayer;imageId?:string|null;observationId?:string|null}={layer:'model'}){
    if(disposed||revisionId!==expectedRevisionId||expectedFrameId!==undefined&&expectedFrameId!==frameId||gl!.isContextLost())return null;
    const modeled=source.layer==='model',subject=entity(entityId),composite=modeled&&compositeModelEvidence(doc,entityId),family=modeled?modelPreviewEntities(doc,entityId):subject?[subject]:[];
    if(composite&&(!frameId||!modelPreviewGeometry(doc,entityId,{...source,frameId,showCandidates:true,observations:doc.observations})))return null;
    const models=family.flatMap(subject=>{const geometry=frameId?entityGeometryForLayer(subject,{...source,frameId,showCandidates:true,observations:doc.observations}):null;const reps=(subject.representations||[]).filter((rep:any)=>(modeled||rep.kind===source.layer)&&geometry?.representationIds.includes(rep.id));return subject.visible!==false?reps.map((rep:any)=>({subject,rep})):[];});
    if(!models.length||subject?.visible===false||models.some(({subject,rep})=>!representationAvailable(subject,rep,frameId,true)||!loadedRepresentations.has(subject.id+'/'+rep.id)))return null;
    const saved={camera,layers,selection,radius,center,width:canvas.width,height:canvas.height,canvasStyle:canvas.style.cssText,photoStyle:photo.style.cssText};
    try{
      // ponytail: one synchronous capture reuses the scene GPU buffers; restore
      // before yielding so object previews cannot change scene navigation.
      captureSize={w:640,h:640,cw:640,ch:640};
      // Studio shading lifts display shadows only; mesh colors/materials remain unchanged.
      layers={...layers,measurement:modeled&&layers.measurement?.references.every((ref:any)=>family.some(e=>e.id===ref.entityId))?layers.measurement:null,studio:true,modelOnly:modeled,entityId:undefined,entityIds:family.map(e=>e.id),representationIds:models.map(({rep})=>rep.id),observationEntityId:entityId,observationId:source.observationId,observations:doc.observations,imageId:source.imageId??layers.imageId,axisEntityId:composite?undefined:entityId,showAxes:!composite,observed_surface:source.layer==='observed_surface',point_cloud:source.layer==='point_cloud',generated_mesh:modeled,primitive:modeled,showCandidates:true,showBounds:false,editable:false};
      selection={};dimensions();camera=fittedCamera(mode,models.length===1?models[0]:undefined);const captureCanvas=document.createElement('canvas');if(!captureCanvas.getContext('2d'))throw Error('canvas_2d_unavailable');draw(false,captureCanvas);return captureCanvas.toDataURL('image/png');
    }finally{
      camera=saved.camera;layers=saved.layers;selection=saved.selection;radius=saved.radius;center=saved.center;captureSize=null;
      canvas.width=saved.width;canvas.height=saved.height;canvas.style.cssText=saved.canvasStyle;photo.style.cssText=saved.photoStyle;draw();
    }
  }
  async function upload(mesh:Mesh,entityId:string,representation:any,n:number){
    if(disposed||n!==epoch)return null;
    if(!mesh.indices.length||!mesh.vertices.length)throw Error('empty_mesh');
    const vertex=gl!.createBuffer(),index=gl!.createBuffer(),texture=gl!.createTexture();
    if(!vertex||!index||!texture){gl!.deleteBuffer(vertex);gl!.deleteBuffer(index);gl!.deleteTexture(texture);throw Error('gpu_allocation_failed');}
    const g={mesh,vertex,index,texture,entityId,representation};
    try{
      gl!.bindBuffer(gl!.ARRAY_BUFFER,vertex);gl!.bufferData(gl!.ARRAY_BUFFER,mesh.vertices,gl!.STATIC_DRAW);gl!.bindBuffer(gl!.ELEMENT_ARRAY_BUFFER,index);gl!.bufferData(gl!.ELEMENT_ARRAY_BUFFER,mesh.indices,gl!.STATIC_DRAW);gl!.bindTexture(gl!.TEXTURE_2D,texture);gl!.texParameteri(gl!.TEXTURE_2D,gl!.TEXTURE_MIN_FILTER,gl!.LINEAR);gl!.texParameteri(gl!.TEXTURE_2D,gl!.TEXTURE_MAG_FILTER,gl!.LINEAR);gl!.texParameteri(gl!.TEXTURE_2D,gl!.TEXTURE_WRAP_S,gl!.CLAMP_TO_EDGE);gl!.texParameteri(gl!.TEXTURE_2D,gl!.TEXTURE_WRAP_T,gl!.CLAMP_TO_EDGE);gl!.texImage2D(gl!.TEXTURE_2D,0,gl!.RGBA,1,1,0,gl!.RGBA,gl!.UNSIGNED_BYTE,new Uint8Array([255,255,255,255]));
      if(gl!.getError()!==gl!.NO_ERROR)throw Error('gpu_upload_failed');
      if(mesh.texture){const bitmap=await createImageBitmap(mesh.texture,{imageOrientation:'none',premultiplyAlpha:'none'});try{if(!disposed&&n===epoch){gl!.bindTexture(gl!.TEXTURE_2D,texture);gl!.pixelStorei(gl!.UNPACK_FLIP_Y_WEBGL,false);gl!.texImage2D(gl!.TEXTURE_2D,0,gl!.RGBA,gl!.RGBA,gl!.UNSIGNED_BYTE,bitmap);if(gl!.getError()!==gl!.NO_ERROR)throw Error('gpu_texture_upload_failed');}}finally{bitmap.close();}}
      if(disposed||n!==epoch){releaseMesh(g);return null;}
      if(gl!.isContextLost())throw Error('gpu_context_lost');
      return g;
    }catch(error){releaseMesh(g);throw error;}
  }
  // Temporal RGB-D surfaces replace only their own buffers; static room assets stay resident.
  function setStreamMesh(entityId:string,mesh:Mesh){
    const e=entity(entityId),r=e?.representations?.find((r:any)=>r.streamed);
    if(disposed||!r||mesh.texture||!mesh.vertices.length||!mesh.indices.length||mesh.mode!==4||mesh.indices.length%3||mesh.vertices.length%12||mesh.indices.some(i=>i>=mesh.vertices.length/12)||mesh.vertices.some(v=>!Number.isFinite(v)))throw Error('invalid_stream_mesh');
    streamedMeshes.set(entityId,mesh);
    for(const g of gpu.filter(g=>g.entityId===entityId&&g.representation.id===r.id)){
      gl!.bindBuffer(gl!.ARRAY_BUFFER,g.vertex);gl!.bufferData(gl!.ARRAY_BUFFER,mesh.vertices,gl!.DYNAMIC_DRAW);
      gl!.bindBuffer(gl!.ELEMENT_ARRAY_BUFFER,g.index);gl!.bufferData(gl!.ELEMENT_ARRAY_BUFFER,mesh.indices,gl!.DYNAMIC_DRAW);g.mesh=mesh;
    }
    draw();
  }
  function loadProgress(){
    const states=[...assetStates.values()],ready=states.filter(state=>state.state==='ready');
    emit('loadProgress',{phase:'assets',revisionId,loaded:ready.length,total:states.length,message:backgroundOnly?'loadingMoving':undefined,
      failed:states.filter(state=>state.state==='error').length,pending:states.filter(state=>state.state==='loading').length,
      vertices:ready.reduce((sum,state)=>sum+state.vertexCount,0),triangles:ready.reduce((sum,state)=>sum+state.triangleCount,0),states});
  }
  async function setScene(revision:any){
    const next=revision.document||revision;if(!next||!Array.isArray(next.entities)||!Array.isArray(next.cameras))throw Error('invalid_scene_document');const nextFrame=layers.imageId?cameraForImage(next,layers.imageId)?.coordinateFrameId||null:currentCameras(next).find((c:any)=>c.id===selection.cameraId)?.coordinateFrameId||currentCameras(next)[0]?.coordinateFrameId||next.coordinateFrames?.[0]?.id||null;const tasks=sceneRepresentationTasks(next,nextFrame,layers),layerKey=taskKey(tasks);const signature=JSON.stringify([layerKey,next.entities.map((e:any)=>[e.id,e.activeModelRepresentationId,(e.representations||[]).map((r:any)=>[r.id,r.assetId,r.kind,r.primitive,r.placementState,r.placementReason,r.sourceValidity])]),next.assets]);
    if(signature===sceneAssetsSignature&&doc.captureId===next.captureId){doc=next;layers.observations=doc.observations;if(layers.imageId)frameId=cameraForImage(doc,layers.imageId)?.coordinateFrameId||null;revisionId=revision.id||'';preview.clear();for(const g of gpu){const r=entity(g.entityId)?.representations?.find((r:any)=>r.id===g.representation.id);if(r)g.representation=r;}dimensions();draw();loadProgress();return;}
    if(doc.captureId!==next.captureId)streamedMeshes.clear();
    for(const id of [...glbNodes.keys()])if(!(next.assets||[]).some((a:any)=>a.id===id))glbNodes.delete(id);  // r5b: a replaced shot GLB
    sceneAssetsSignature=signature;loadedLayerKey=layerKey;const n=++epoch;abort.abort();abort=new AbortController();preview.clear();
    // A new layer (a live report grows while it is open) keeps what is already on the GPU: an unchanged (entity,
    // representation, asset) is neither fetched nor uploaded again. A representation whose asset changed (a quick room,
    // then the full one) stays drawn until its new asset is up; everything else is released as before.
    const keys=new Set(tasks.map(({e,r})=>representationKey(e.id,r))),kept=gpu.filter(g=>keys.has(representationKey(g.entityId,g.representation))&&loadedRepresentations.has(g.entityId+'/'+g.representation.id)),keptKeys=new Set(kept.map(g=>g.entityId+'/'+g.representation.id));
    const replacing=new Set(tasks.map(({e,r})=>e.id+'/'+r.id));for(const g of gpu)if(!kept.includes(g)&&replacing.has(g.entityId+'/'+g.representation.id))stale.add(g);else stale.delete(g);
    gpu.filter(g=>!kept.includes(g)&&!stale.has(g)).forEach(releaseMesh);gpu=gpu.filter(g=>kept.includes(g)||stale.has(g));for(const key of [...loadedRepresentations])if(!keptKeys.has(key))loadedRepresentations.delete(key);for(const key of [...assetStates.keys()])if(!keptKeys.has(key))assetStates.delete(key);
    doc=next;layers.observations=doc.observations;revisionId=revision.id||'';frameId=nextFrame;for(const g of gpu){const r=entity(g.entityId)?.representations?.find((r:any)=>r.id===g.representation.id);if(r)g.representation=r;}
    tasks.splice(0,tasks.length,...tasks.filter(({e,r})=>!keptKeys.has(e.id+'/'+r.id)));
    if(!camera){if(currentCameras(doc)[0])setCamera(currentCameras(doc)[0].id);else setCamera('free');}else if(camera.exact&&currentCameras(doc).some((c:any)=>c.id===camera!.frame.id))setCamera(camera.frame.id);
    emit('loadProgress',{phase:'metadata',loaded:0,total:doc.entities.length});
    const fitVersion=navigationVersion,initialRadius=radius;
    // Static scene first, framed and pickable as soon as it is in; a video's many small time-stamped surfaces follow.
    // Waiting for all of them (433 for a 30 s clip, ~4 requests/s from the publication site) kept the view unframed for minutes.
    tasks.sort((a,b)=>Number(!!a.r.timeRange)-Number(!!b.r.timeRange));
    let staticLeft=tasks.filter(({r})=>!r.timeRange).length,fitted=false;backgroundOnly=false;
    const fit=()=>{if(fitted||n!==epoch||disposed)return;fitted=true;dimensions();if(autoFit&&!camera?.exact&&navigationVersion===fitVersion&&initialRadius!==radius)setCamera(viewMode);draw();};
    for(const {e,r}of tasks)assetStates.set(e.id+'/'+r.id,{entityId:e.id,representationId:r.id,assetId:r.assetId||null,state:'loading',vertexCount:0,triangleCount:0,errorCode:null});
    loadProgress();
    // Two downloads at a time bounds decode memory on phones. Late data may draw,
    // but never writes the host's selection or camera.
    const workers:Promise<void>[]=[];
    async function load(){while(tasks.length&&!disposed&&n===epoch){
      const {e,r}=tasks.shift()!,key=e.id+'/'+r.id,staged:GPU[]=[];
      try{
        let meshes:Mesh[];
        if(r.kind==='primitive')meshes=[primitive(r.primitive)];
        else if(r.node){if(!r.assetId)throw Error('missing_mesh_asset');meshes=(await nodeMeshes(r.assetId)).filter(m=>m.name===r.node);if(n!==epoch||disposed)return;}  // r5b: one GLB a shot, a node per object
        else{if(!r.assetId)throw Error('missing_mesh_asset');const src=await url(r.assetId);if(n!==epoch||disposed)return;const res=await fetch(src,{signal:abort.signal});if(!res.ok)throw Error('asset_download_failed');const bytes=await res.arrayBuffer();if(n!==epoch||disposed)return;emit('loadProgress',{phase:'gpu_upload',entityId:e.id,representationId:r.id,assetId:r.assetId,bytes:bytes.byteLength});const meta=doc.assets.find((a:any)=>a.id===r.assetId);meshes=(meta?.format||meta?.metadata?.format)==='panoptes-mesh-v1'?readPacked(bytes,meta):readGLB(bytes);}
        if(!meshes.length)throw Error('empty_mesh');
        for(const source of meshes){const mesh=r.streamed&&streamedMeshes.has(e.id)?streamedMeshes.get(e.id)!:source;const uploaded=await upload(mesh,e.id,r,n);if(uploaded)staged.push(uploaded);}
        if(n!==epoch||disposed){staged.forEach(releaseMesh);return;}
        const current=entity(e.id)?.representations?.find((rep:any)=>rep.id===r.id);
        if(!current)throw Error('representation_not_found');
        for(const g of staged)g.representation=current;
        gpu.push(...staged);loadedRepresentations.add(key);replaced(e.id,r.id);
        assetStates.set(key,{entityId:e.id,representationId:r.id,assetId:r.assetId||null,state:'ready',errorCode:null,vertexCount:meshes.reduce((sum,mesh)=>sum+mesh.vertices.length/12,0),triangleCount:meshes.reduce((sum,mesh)=>sum+(mesh.mode===4?mesh.indices.length/3:0),0)});
        refitSoon();
      }catch(error:any){gpu=gpu.filter(g=>!staged.includes(g));loadedRepresentations.delete(key);staged.forEach(releaseMesh);if(n===epoch)replaced(e.id,r.id);if(n===epoch&&!disposed&&error.name!=='AbortError'){assetStates.set(key,{entityId:e.id,representationId:r.id,assetId:r.assetId||null,state:'error',vertexCount:0,triangleCount:0,errorCode:error.message});emit('loadError',{code:error.message,entityId:e.id,representationId:r.id,assetId:r.assetId});}}
      if(!r.timeRange&&--staticLeft===0&&n===epoch&&!disposed){fit();backgroundOnly=tasks.length>0;if(backgroundOnly)workers.push(load(),load(),load(),load());}
      if(n===epoch&&!disposed)loadProgress();
    }}
    // Moving surfaces are tens of KB each: four more requests at a time once the static scene is in.
    workers.push(load(),load());for(let i=0;i<workers.length;i++)await workers[i];
    if(n===epoch&&!disposed){backgroundOnly=false;fit();dimensions();draw();loadProgress();emit('renderReady',{phase:'scene',revisionId,canvasReady:!gl!.isContextLost()});}
  }
  function listen(target:EventTarget,name:string,fn:any,opts?:any){target.addEventListener(name,fn,opts);cleanups.push(()=>target.removeEventListener(name,fn,opts));}
  listen(canvas,'pointerdown',(e:PointerEvent)=>{drag={x:e.clientX,y:e.clientY,b:e.button,moved:0,look:splatsActive()&&nearPath(false)};canvas.setPointerCapture(e.pointerId);});
  listen(stage,'pointermove',(e:PointerEvent)=>{if(axisDrag){const d=axisDrag,dx=d.b[0]-d.a[0],dy=d.b[1]-d.a[1],den=dx*dx+dy*dy;if(den<4)return;const amount=((e.clientX-d.x)*dx+(e.clientY-d.y)*dy)/den*d.length,t={...d.t,position:add(d.t.position,scale(d.direction,amount))};d.changed=true;previewTransform(d.id,t);emit('transformPreview',{operations:[{type:'setTransform',entityId:d.id,...t}]});draw();return;}
    if(!drag||!camera)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;drag.moved+=Math.abs(dx)+Math.abs(dy);if(camera.exact||drag.moved<4)return;if(!drag.navigated){drag.navigated=true;emit('userNavigation');}navigationVersion++;autoFit=false;let offset=add(camera.eye,scale(camera.target,-1)),right=unit(cross(camera.up,offset));if(drag.b===2||e.shiftKey){const mv=add(scale(right,-dx*radius*.003),scale(camera.up,dy*radius*.003));camera.eye=add(camera.eye,mv);camera.target=add(camera.target,mv);}else if(drag.look)lookAround(camera,dx,dy);else{offset=rotate(offset,unit(camera.up),-dx*.006);right=unit(cross(camera.up,offset));offset=rotate(offset,right,-dy*.006);camera.up=rotate(camera.up,right,-dy*.006);camera.eye=add(camera.target,offset);}draw();});
  function pickAt(x:number,y:number){
    if(!camera)return;const rect=canvas.getBoundingClientRect();if(x<0||y<0||x>=rect.width||y>=rect.height)return;
    draw(true);const px=new Uint8Array(4);gl!.readPixels(Math.floor(x*canvas.width/rect.width),canvas.height-1-Math.floor(y*canvas.height/rect.height),1,1,gl!.RGBA,gl!.UNSIGNED_BYTE,px);draw();
    const i=px[0]+(px[1]<<8)+(px[2]<<16)-1,id=doc.entities[i]?.id||(layers.pickingPoints?undefined:objectUnder(x,y,rect));
    if(layers.pickingPoints){
      const inverse=new DOMMatrix(Array.from(cameraMatrix(camera,rect.width/rect.height,radius))).inverse();
      const unproject=(z:number)=>{const p=inverse.transformPoint(new DOMPoint(2*x/rect.width-1,1-2*y/rect.height,z,1));return [p.x/p.w,p.y/p.w,p.z/p.w];};
      const origin=unproject(-1),end=unproject(1),direction=unit(add(end,scale(origin,-1)));let nearest=Infinity,hit:SurfacePick|null=null;
      for(const g of gpu.filter(g=>g.entityId===id&&visible(g)&&g.mesh.mode===4&&['generated_mesh','primitive'].includes(g.representation.kind))){
        const matrix=model(g),local=new DOMMatrix(Array.from(matrix)).inverse(),a=local.transformPoint(new DOMPoint(...origin)),b=local.transformPoint(new DOMPoint(...add(origin,direction)));
        const found=rayMeshPoint(g.mesh.vertices,g.mesh.indices,[a.x,a.y,a.z],[b.x-a.x,b.y-a.y,b.z-a.z]);
        if(found&&found.distance<nearest){nearest=found.distance;hit={point:point(matrix,found.point),entityId:g.entityId,representationId:g.representation.id,coordinateFrameId:frameId!};}
      }
      emit('measurementPoint',{hit});return;
    }
    emit('selectionIntent',{entityId:id||null,cameraId:camera.exact?camera.frame.id:null,originalPixel:camera.exact?[x*camera.frame.width/rect.width-.5,y*camera.frame.height/rect.height-.5]:null});
  }
  // The room surface is context (pick ID 0) and an object's own surfaces are loaded only for the photo in view, so in a
  // free view most objects drawn inside the room mesh were not clickable. A click on the room surface now selects the
  // object whose surface is there: the room hit point, inside the smallest box of that object's fused (else observed)
  // surfaces. Boxes, not shapes: an object's box can also hold a little of its neighbour; the smallest box wins.
  function objectUnder(x:number,y:number,rect:DOMRect){
    const inverse=new DOMMatrix(Array.from(cameraMatrix(camera!,rect.width/rect.height,radius))).inverse();
    const unproject=(z:number):Vec=>{const p=inverse.transformPoint(new DOMPoint(2*x/rect.width-1,1-2*y/rect.height,z,1));return [p.x/p.w,p.y/p.w,p.z/p.w];};
    const origin=unproject(-1),direction=unit(add(unproject(1),scale(origin,-1)));let nearest=Infinity,hit:Vec|null=null;
    for(const g of gpu.filter(g=>entity(g.entityId)?.sourceContext&&visible(g)&&g.mesh.mode===4)){
      const m=model(g),local=new DOMMatrix(Array.from(m)).inverse(),a=local.transformPoint(new DOMPoint(...origin)),b=local.transformPoint(new DOMPoint(...add(origin,direction)));
      const start:Vec=[a.x,a.y,a.z],step:Vec=[b.x-a.x,b.y-a.y,b.z-a.z],t=rayMeshDistance(g.mesh.vertices,g.mesh.indices,start,step);
      if(t<Infinity){const p=point(m,add(start,scale(step,t))),d=Math.hypot(...add(p,scale(origin,-1)));if(d<nearest){nearest=d;hit=p;}}
    }
    if(!hit)return undefined;
    const margin=radius*.01;let best:string|undefined,volume=Infinity;
    for(const e of doc.entities){
      if(e.sourceContext||e.visible===false||layers.entityIds&&!layers.entityIds.includes(e.id))continue;
      const reps=(e.representations||[]).filter((r:any)=>!r.timeRange&&r.bounds&&r.coordinateFrameId===frameId),fused=reps.filter((r:any)=>r.sourceKind==='observed_reference_surface');
      for(const r of fused.length?fused:reps){
        const m=transformMatrix(r.transform||{position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]}),cs=boundsCorners(r.bounds).map(p=>point(m,p));
        const lo=[0,1,2].map(k=>Math.min(...cs.map(p=>p[k]))-margin),hi=[0,1,2].map(k=>Math.max(...cs.map(p=>p[k]))+margin);
        if(hit.every((v,k)=>v>=lo[k]&&v<=hi[k])){const size=(hi[0]-lo[0])*(hi[1]-lo[1])*(hi[2]-lo[2]);if(size<volume){volume=size;best=e.id;}}
      }
    }
    return best;
  }
  listen(stage,'pointerup',(e:PointerEvent)=>{if(axisDrag){const d=axisDrag;axisDrag=null;if(d.changed)emit('transformCommitIntent',{operations:[{type:'setTransform',entityId:d.id,...preview.get(d.id)}]});return;}const prev=drag;drag=null;if(!prev||prev.b!==0||prev.moved>=4)return;const rect=canvas.getBoundingClientRect();pickAt(e.clientX-rect.left,e.clientY-rect.top);});
  listen(canvas,'keydown',(e:KeyboardEvent)=>{if(!layers.pickingPoints||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Enter'].includes(e.key))return;e.preventDefault();const {cw,ch}=viewSize();pickCursor ||= [.5,.5];if(e.key==='Enter'){pickAt(pickCursor[0]*cw,pickCursor[1]*ch);return;}const step=e.shiftKey?10:1;pickCursor=[Math.max(0,Math.min(1-1/cw,pickCursor[0]+(e.key==='ArrowRight'?step:e.key==='ArrowLeft'?-step:0)/cw)),Math.max(0,Math.min(1-1/ch,pickCursor[1]+(e.key==='ArrowDown'?step:e.key==='ArrowUp'?-step:0)/ch))];draw();});
  listen(stage,'pointercancel',()=>{if(axisDrag)preview.clear();axisDrag=null;drag=null;emit('transformPreview',{operations:[]});draw();});listen(canvas,'contextmenu',(e:Event)=>e.preventDefault());
  listen(canvas,'wheel',(e:WheelEvent)=>{if(!camera||camera.exact)return;e.preventDefault();emit('userNavigation');navigationVersion++;autoFit=false;const offset=add(camera.eye,scale(camera.target,-1)),factor=Math.exp(e.deltaY*.0012);if(camera.orthographic)camera.orthoHeight=Math.max(radius*.02,Math.min(radius*50,camera.orthoHeight!*factor));else camera.eye=add(camera.target,scale(offset,Math.max(radius*.02,Math.min(radius*50,Math.hypot(...offset)*factor))/Math.hypot(...offset)));draw();},{passive:false});
  listen(canvas,'webglcontextlost',(e:Event)=>{e.preventDefault();loadedRepresentations.clear();for(const[key,state]of assetStates)if(state.state==='ready')assetStates.set(key,{...state,state:'error',vertexCount:0,triangleCount:0,errorCode:'gpu_context_lost'});loadProgress();emit('contextLost');});listen(canvas,'webglcontextrestored',()=>emit('loadError',{code:'viewer_remount_required'}));const observer=new ResizeObserver(()=>draw());observer.observe(stage);
  // One splat set per report (its annotation); the same key again is a no-op, so toggling the layer never reloads it.
  function setSplats(source:SplatSource|null){
    if(disposed||(source?.key??null)===(splatSource?.key??null))return;
    splat?.dispose();splat=null;splatSource=source;
    if(source)try{const layer=splat=createSplatLayer(gl!,()=>draw());layer.load(source.url,source.count).catch((error:any)=>{if(!disposed&&splat===layer&&error.name!=='AbortError')emit('loadError',{code:'splat_load_failed'});});}
    catch{splat=null;emit('loadError',{code:'splat_load_failed'});}
    draw();
  }
  return {setScene,setStreamMesh,capturePreview,setSplats,splatStats:()=>splat?.stats()||null,setSelection(value:any){selection={...value};draw();},setCamera,setLayers(value:any){layers={...layers,...value};if(!layers.pickingPoints)pickCursor=null;if(layers.imageId)frameId=cameraForImage(doc,layers.imageId)?.coordinateFrameId||null;if(sceneAssetsSignature&&taskKey(sceneRepresentationTasks(doc,frameId,layers))!==loadedLayerKey)void setScene({id:revisionId,document:doc}).catch(error=>emit('loadError',{code:error.message}));dimensions();draw();},previewOperations(ops:any[]){for(const op of ops)if(op.type==='setTransform')previewTransform(op.entityId,op.transform||op);draw();},clearPreview(){preview.clear();draw();},resize(){draw();},dispose(){if(disposed)return;disposed=true;splat?.dispose();streamedMeshes.clear();repaint.cancel();epoch++;photoEpoch++;abort.abort();photoAbort.abort();if(photoObjectURL)URL.revokeObjectURL(photoObjectURL);observer.disconnect();cleanups.forEach(fn=>fn());release();shaders.forEach(s=>gl.deleteShader(s));gl.deleteProgram(program);gl.getExtension('WEBGL_lose_context')?.loseContext();photo.removeAttribute('src');stage.remove();}};
}
