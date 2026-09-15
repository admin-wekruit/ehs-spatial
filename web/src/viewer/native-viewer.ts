import {add,scale,dot,cross,unit,identity,matmul,point,rotate,transformMatrix,sourceCamera,cameraMatrix,boundsCorners,projected,fitCamera,type Camera,type Vec,type Transform} from './native-math.ts';
import {isReferenceSurface} from '../scene-semantics.ts';
import {entityGeometryForLayer,representationAvailable,representationInPhoto,cameraForImage,currentCameras,type GeometryLayer} from '../core.ts';
import type {SceneDocument} from '../types';

type Mesh={vertices:Float32Array;indices:Uint32Array;mode:number;matrix:ArrayLike<number>;texture?:Blob;bounds:{min:Vec;max:Vec}};
type GPU={mesh:Mesh;vertex:WebGLBuffer;index:WebGLBuffer;texture:WebGLTexture;entityId:string;representation:any};
export type ViewerEvent={type:string;[key:string]:any};
export type ViewerOptions={resolveAsset:(id:string)=>Promise<string|{url:string}>;locale?:string;onEvent?:(event:ViewerEvent)=>void};
export type SceneViewer=ReturnType<typeof mountSceneViewer>;

export function representationPass(entity:any,representation:any,frameId:string|null,layers:any) {
  const available=!!entity&&(!layers.entityId||entity.id===layers.entityId)&&
    (!layers.modelOnly||!entity.sourceContext&&['generated_mesh','primitive'].includes(representation.kind))&&
    representationAvailable(entity,representation,frameId,!!layers.showCandidates)&&representationInPhoto(entity,representation,layers.imageId,layers.observations);
  const visible=available&&layers[representation.kind]!==false;
  const cloudOnly=layers.point_cloud!==false&&['observed_surface','generated_mesh','primitive'].every(kind=>layers[kind]===false);
  // Pick actual observed triangles in a cloud view, never invisible generated
  // geometry or a bounding-box proxy. Visible context may occlude, but has ID 0.
  return {available,visible,pick:visible||available&&cloudOnly&&!entity.sourceContext&&representation.kind==='observed_surface'&&representation.placementState==='confirmed',selectable:available&&!entity.sourceContext};
}

export function selectionGeometry(document:SceneDocument,entity:any,frameId:string|null,layers:any,preview?:Transform) {
  if(!entity||entity.sourceContext||entity.visible===false||!frameId||isReferenceSurface(document,entity))return {corners:[] as Vec[],transform:undefined,axisSpace:'native',editable:false};
  const cloudOnly=layers.point_cloud!==false&&['observed_surface','generated_mesh','primitive'].every(kind=>layers[kind]===false);
  const layer:GeometryLayer=cloudOnly?'point_cloud':layers.generated_mesh!==false||layers.primitive!==false?'model':'observed_surface';
  const subject=preview?{...entity,currentModelTransform:preview}:entity;
  const geometry=entityGeometryForLayer(subject,{layer,frameId,showCandidates:!!layers.showCandidates,imageId:layers.imageId,observations:document.observations});
  return geometry?{...geometry,editable:geometry.geometryKind==='model'&&layers.editable!==false}:{corners:[] as Vec[],transform:undefined,axisSpace:'native',editable:false};
}

export function readPacked(buffer:ArrayBuffer,metadata:any):Mesh[] {
  const layout=metadata.byteLayout||metadata.metadata?.byteLayout;
  if(!layout||layout.stride!==9||layout.indexType!=='uint32')throw Error('unsupported_packed_mesh');
  const {byteOffset=0,vertexCount,indexByteOffset,indexCount}=layout;
  if(![byteOffset,vertexCount,indexByteOffset,indexCount].every(n=>Number.isSafeInteger(n)&&n>=0)||byteOffset%4||indexByteOffset%4||byteOffset+vertexCount*36>indexByteOffset||indexByteOffset+indexCount*4>buffer.byteLength||indexCount%3)throw Error('invalid_packed_mesh');
  const source=new Float32Array(buffer,byteOffset,vertexCount*9),indices=new Uint32Array(buffer,indexByteOffset,indexCount).slice(),vertices=new Float32Array(vertexCount*11),bounds={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};
  if(source.some(n=>!Number.isFinite(n))||indices.some(n=>n>=vertexCount))throw Error('invalid_packed_mesh_data');
  for(let i=0;i<vertexCount;i++)vertices.set(source.subarray(i*9,i*9+9),i*11);
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
      const material=doc.materials?.[p.material],pbr=material?.pbrMetallicRoughness,factor=pbr?.baseColorFactor||[1,1,1,1],data=new Float32Array(pos.count*11),bounds={min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]};
      for(let i=0;i<pos.count;i++){for(let k=0;k<3;k++){data[i*11+k]=pos.data[i*3+k];data[i*11+3+k]=normal?.data[i*3+k]??0;data[i*11+6+k]=(color?.data[i*color.width+k]??1)*factor[k];}data[i*11+9]=uv?.data[i*2]||0;data[i*11+10]=uv?.data[i*2+1]||0;}
      for(const i of indices)for(let k=0;k<3;k++){bounds.min[k]=Math.min(bounds.min[k],pos.data[i*3+k]);bounds.max[k]=Math.max(bounds.max[k],pos.data[i*3+k]);}
      if(!normal&&(p.mode??4)===4)for(let i=0;i<indices.length;i+=3){const ids=[indices[i],indices[i+1],indices[i+2]],ps=ids.map(n=>pos.data.slice(n*3,n*3+3)),n=cross(add(ps[1],scale(ps[0],-1)),add(ps[2],scale(ps[0],-1)));for(const id of ids)for(let k=0;k<3;k++)data[id*11+3+k]+=n[k];}
      let texture:Blob|undefined;const textureIndex=pbr?.baseColorTexture?.index;
      if(textureIndex!==undefined){if(!uv||pbr.baseColorTexture.texCoord||pbr.baseColorTexture.extensions)throw Error('unsupported_texture_coordinates');const img=doc.images?.[doc.textures?.[textureIndex]?.source];if(!img||img.uri||!['image/png','image/jpeg'].includes(img.mimeType))throw Error('unsupported_glb_texture');const v=view(img.bufferView);texture=new Blob([buffer.slice(binOffset+(v.byteOffset||0),binOffset+(v.byteOffset||0)+v.byteLength)],{type:img.mimeType});}
      result.push({vertices:data,indices,mode:p.mode??4,matrix,texture,bounds});
    }
    for(const child of node.children||[])walk(child,matrix);visiting.delete(id);
  }
  const scene=doc.scenes?.[doc.scene??0];if(!scene)throw Error('missing_glb_scene');for(const id of scene.nodes||[])walk(id,identity());return result;
}

function primitive(spec:any):Mesh {
  const p=spec.parameters||spec,type=spec.kind||spec.primitiveType||spec.type;const vertices:number[]=[],indices:number[]=[];
  if(type==='box'){const dims=p.dimensions;if(!Array.isArray(dims)||dims.length!==3)throw Error('invalid_primitive');if(!dims.every(v=>Number.isFinite(v)&&v>0))throw Error('invalid_primitive');for(const c of boundsCorners({min:dims.map(v=>-v/2),max:dims.map(v=>v/2)}))vertices.push(...c,0,0,1,1,1,1,0,0);indices.push(0,2,1,1,2,3,4,5,6,5,7,6,0,1,4,1,5,4,2,6,3,3,6,7,0,4,2,2,4,6,1,3,5,3,7,5);}
  else if(type==='cylinder'){const r=p.radius,h=p.height,n=p.segments??64;if(!(Number.isFinite(r)&&r>0&&Number.isFinite(h)&&h>0&&Number.isInteger(n)&&n>=8&&n<=256))throw Error('invalid_primitive');for(let z=0;z<2;z++)for(let i=0;i<n;i++){const a=i*2*Math.PI/n;vertices.push(r*Math.cos(a),r*Math.sin(a),(z-.5)*h,Math.cos(a),Math.sin(a),0,1,1,1,0,0);}vertices.push(0,0,-h/2,0,0,-1,1,1,1,0,0,0,0,h/2,0,0,1,1,1,1,0,0);for(let i=0;i<n;i++){const j=(i+1)%n;indices.push(i,j,n+j,i,n+j,n+i,2*n,j,i,2*n+1,n+i,n+j);}}
  else throw Error('unsupported_primitive');
  const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];for(let i=0;i<vertices.length;i+=11)for(let k=0;k<3;k++){min[k]=Math.min(min[k],vertices[i+k]);max[k]=Math.max(max[k],vertices[i+k]);}
  return {vertices:new Float32Array(vertices),indices:new Uint32Array(indices),mode:4,matrix:identity(),bounds:{min,max}};
}

export function mountSceneViewer(container:HTMLElement,options:ViewerOptions){
  const stage=document.createElement('div');stage.className='native-stage';Object.assign(stage.style,{position:'relative',width:'100%',height:'100%',minHeight:'260px',display:'flex',alignItems:'center',justifyContent:'center',overflow:'hidden',background:'#111b21'});
  const photo=document.createElement('img');photo.alt='';Object.assign(photo.style,{position:'absolute',objectFit:'contain',pointerEvents:'none'});photo.hidden=true;
  const canvas=document.createElement('canvas');canvas.setAttribute('aria-label',options.locale==='en'?'Interactive scene':'交互场景');canvas.tabIndex=0;Object.assign(canvas.style,{position:'relative',touchAction:'none'});
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');Object.assign(svg.style,{position:'absolute',inset:'0',width:'100%',height:'100%',pointerEvents:'none'});stage.append(photo,canvas,svg);container.append(stage);
  const gl=canvas.getContext('webgl2',{alpha:true,antialias:true,preserveDrawingBuffer:true});if(!gl){stage.remove();throw Error('webgl_unavailable');}
  let disposed=false,epoch=0,photoEpoch=0,doc:any={entities:[],cameras:[],coordinateFrames:[]},revisionId='',selection:any={},camera:Camera|null=null,radius=1,center:Vec=[0,0,0],frameId:string|null=null,gpu:GPU[]=[],abort=new AbortController(),preview=new Map<string,Transform>(),drag:any=null,axisDrag:any=null;
  let sceneAssetsSignature='',viewMode='free';let navigationVersion=0;let photoAbort=new AbortController(),photoObjectURL:string|null=null;
  let captureSize:{w:number;h:number;cw:number;ch:number}|null=null;const loadedRepresentations=new Set<string>();
  let layers:any={observed_surface:true,generated_mesh:true,primitive:true,point_cloud:true,allBounds:false,opacity:.65,lighting:true,showCandidates:false};const cleanups:(()=>void)[]=[];
  const emit=(type:string,payload:any={})=>{if(!disposed)options.onEvent?.({type,...payload});};
  const program=gl.createProgram()!;const shaders:WebGLShader[]=[];
  for(const [type,code]of [[gl.VERTEX_SHADER,`#version 300 es\nin vec3 p;in vec3 n;in vec3 c;in vec2 uv;uniform mat4 vp;uniform mat4 model;out vec3 color;out vec3 normal;out vec2 tex;void main(){gl_Position=vp*model*vec4(p,1.);gl_PointSize=2.;color=c;normal=transpose(inverse(mat3(model)))*n;tex=uv;}`],[gl.FRAGMENT_SHADER,`#version 300 es\nprecision highp float;in vec3 color;in vec3 normal;in vec2 tex;uniform sampler2D image;uniform vec3 pickColor;uniform vec3 tint;uniform float selected;uniform float pick;uniform float opacity;uniform float lighting;out vec4 outColor;void main(){if(pick>.5){outColor=vec4(pickColor,1.);return;}vec3 c=color*texture(image,tex).rgb*tint;float n=length(normal);if(lighting>1.5){c=pow(clamp(c,0.,1.),vec3(.72));if(n>.01){vec3 direction=normal/n;float key=abs(dot(direction,normalize(vec3(.3,.8,.6))));float fill=abs(dot(direction,normalize(vec3(-.7,.4,-.5))));c=c*(.78+.15*key+.07*fill)+vec3(.025*pow(key,16.));}}else if(lighting>.5&&n>.01)c*=.5+.5*abs(dot(normal/n,normalize(vec3(.3,.8,.6))));if(selected>.5)c=mix(c,vec3(.28,.86,.75),.4);outColor=vec4(c,opacity);}`]]as[number,string][]){const s=gl.createShader(type)!;gl.shaderSource(s,code);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s)||'shader_error');gl.attachShader(program,s);shaders.push(s);}
  gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error('shader_link_error');gl.useProgram(program);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);
  const u=Object.fromEntries(['vp','model','image','pickColor','selected','pick','opacity','lighting','tint'].map(n=>[n,gl.getUniformLocation(program,n)]));
  const attrs=['p','n','c','uv'].map(n=>gl.getAttribLocation(program,n));for(const a of attrs)gl.enableVertexAttribArray(a);gl.uniform1i(u.image,0);
  function release(){for(const g of gpu){gl!.deleteBuffer(g.vertex);gl!.deleteBuffer(g.index);gl!.deleteTexture(g.texture);}gpu=[];loadedRepresentations.clear();}
  function entity(id:string){return doc.entities.find((e:any)=>e.id===id);}
  function model(g:GPU){const e=entity(g.entityId),t=g.representation.kind==='observed_surface'||g.representation.kind==='point_cloud'?g.representation.transform:preview.get(g.entityId)||e?.currentModelTransform||g.representation.transform;return matmul(transformMatrix(t),g.mesh.matrix);}
  function visible(g:GPU){return representationPass(entity(g.entityId),g.representation,frameId,layers).visible;}
  function corners(id?:string){return gpu.filter(g=>(!id||g.entityId===id)&&visible(g)).flatMap(g=>boundsCorners(g.mesh.bounds).map(p=>point(model(g),p)));}
  function selectedGeometry(id:string){return selectionGeometry(doc,entity(id),frameId,layers,preview.get(id));}
  function selectedAxes(id:string){
    const geometry=selectedGeometry(id),ps=geometry.corners;if(!ps.length)return null;
    const origin=[0,1,2].map(k=>ps.reduce((sum:number,p:Vec)=>sum+p[k],0)/ps.length),m=geometry.axisSpace==='native'?identity():transformMatrix(geometry.transform),length=radius*(layers.studio?.48:.16);
    return {geometry,origin,length,axes:[0,1,2].map(k=>{const direction=unit([m[k*4],m[k*4+1],m[k*4+2]]);return {direction,end:add(origin,scale(direction,length)),label:'XYZ'[k],color:(layers.studio?['#c23f3b','#247545','#2861af']:['#ff6b6b','#65de96','#6bb3ff'])[k]};})};
  }
  function fittingPoints(){const objects=gpu.filter(g=>visible(g)&&!entity(g.entityId)?.sourceContext);return objects.length?objects.flatMap(g=>boundsCorners(g.mesh.bounds).map(p=>point(model(g),p))):corners();}
  function dimensions(){const ps=fittingPoints();if(!ps.length)return;const min=[0,1,2].map(k=>Math.min(...ps.map(p=>p[k]))),max=[0,1,2].map(k=>Math.max(...ps.map(p=>p[k])));center=min.map((v,k)=>(v+max[k])/2);radius=Math.max(Math.hypot(...max.map((v,k)=>v-min[k]))/2,1e-4);}
  function viewSize(){if(captureSize)return captureSize;const w=stage.clientWidth,h=stage.clientHeight,ratio=camera?.exact?camera.frame.width/camera.frame.height:w/h,cw=Math.min(w,h*ratio),ch=cw/ratio;return {w,h,cw,ch};}
  function draw(pick=false,captureCanvas?:HTMLCanvasElement){
    if(disposed||!camera||gl!.isContextLost())return;const {w,h,cw,ch}=viewSize();if(!(cw>0&&ch>0))return;const dpr=Math.min(devicePixelRatio||1,2);canvas.style.width=photo.style.width=cw+'px';canvas.style.height=photo.style.height=ch+'px';const pw=Math.max(1,Math.round(cw*dpr)),ph=Math.max(1,Math.round(ch*dpr));if(canvas.width!==pw||canvas.height!==ph){canvas.width=pw;canvas.height=ph;}
    const background=layers.studio?[237/255,240/255,238/255]:[17/255,27/255,33/255];
    gl!.viewport(0,0,pw,ph);gl!.useProgram(program);gl!.clearColor(pick?0:background[0],pick?0:background[1],pick?0:background[2],camera.exact&&!pick?0:1);gl!.clear(gl!.COLOR_BUFFER_BIT|gl!.DEPTH_BUFFER_BIT);if(camera.exact&&!pick){gl!.enable(gl!.BLEND);gl!.blendFunc(gl!.SRC_ALPHA,gl!.ONE_MINUS_SRC_ALPHA);}else gl!.disable(gl!.BLEND);
    const vp=cameraMatrix(camera,cw/ch,radius);gl!.uniformMatrix4fv(u.vp,false,vp);gl!.uniform1f(u.pick,pick?1:0);gl!.uniform1f(u.opacity,camera.exact&&!pick?layers.opacity:1);gl!.uniform1f(u.lighting,layers.studio?2:layers.lighting?1:0);
    // A selected observed surface can share its exact depth with scene context.
    // Draw it last with equal-depth acceptance; nearer geometry still occludes it.
    // ponytail: same-grid observed subsets use triangle count only as an
    // equal-depth interaction priority, not as evidence of object identity.
    // Unequal sampling would need source-mask area; nearer depth always wins.
    const pickRank=(g:GPU)=>entity(g.entityId)?.sourceContext?0:g.mesh.mode===4&&g.representation.kind==='observed_surface'&&g.representation.placementState==='confirmed'?2:1;
    const drawing=pick?gpu.slice().sort((a,b)=>pickRank(a)-pickRank(b)||(pickRank(a)===2?b.mesh.indices.length-a.mesh.indices.length:0)||a.entityId.localeCompare(b.entityId)||a.representation.id.localeCompare(b.representation.id)):gpu.filter(g=>g.entityId!==selection.entityId).concat(gpu.filter(g=>g.entityId===selection.entityId));
    for(const g of drawing){const pass=representationPass(entity(g.entityId),g.representation,frameId,layers),selected=pass.selectable&&g.entityId===selection.entityId;if(!(pick?pass.pick:pass.visible))continue;gl!.depthFunc(pick||selected?gl!.LEQUAL:gl!.LESS);const id=pass.selectable?doc.entities.findIndex((e:any)=>e.id===g.entityId)+1:0;gl!.bindBuffer(gl!.ARRAY_BUFFER,g.vertex);gl!.bindBuffer(gl!.ELEMENT_ARRAY_BUFFER,g.index);attrs.forEach((a,k)=>gl!.vertexAttribPointer(a,k===3?2:3,gl!.FLOAT,false,44,k*12));gl!.activeTexture(gl!.TEXTURE0);gl!.bindTexture(gl!.TEXTURE_2D,g.texture);gl!.uniformMatrix4fv(u.model,false,model(g));gl!.uniform1f(u.selected,selected?1:0);gl!.uniform3f(u.pickColor,(id&255)/255,((id>>8)&255)/255,((id>>16)&255)/255);const mat=['generated_mesh','primitive'].includes(g.representation.kind)?g.representation.material?.color:undefined;gl!.uniform3fv(u.tint,Array.isArray(mat)&&mat.length>=3?mat.slice(0,3):[1,1,1]);gl!.drawElements(g.mesh.mode===0?gl!.POINTS:gl!.TRIANGLES,g.mesh.indices.length,gl!.UNSIGNED_INT,0);}
    if(pick)return;const overlay=captureCanvas?.getContext('2d');if(overlay&&captureCanvas){captureCanvas.width=canvas.width;captureCanvas.height=canvas.height;overlay.drawImage(canvas,0,0);overlay.scale(canvas.width/w,canvas.height/h);}svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.replaceChildren();const project=(p:Vec)=>{const q=projected(vp,p,cw,ch);return q?add(q,[(w-cw)/2,(h-ch)/2]):null;};
    const line=(a:Vec|null,b:Vec|null,color:string,width=1.5)=>{if(!a||!b)return null;const el=document.createElementNS(svg.namespaceURI,'line');for(const[k,v]of Object.entries({x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:color,'stroke-width':width}))el.setAttribute(k,String(v));svg.append(el);if(overlay){overlay.beginPath();overlay.moveTo(a[0],a[1]);overlay.lineTo(b[0],b[1]);overlay.strokeStyle='#ffffff';overlay.lineWidth=width+2;overlay.stroke();overlay.strokeStyle=color;overlay.lineWidth=width;overlay.stroke();}return el;};
    if(layers.showBounds===false&&!layers.showAxes)return;
    const ids=layers.showBounds===false?[]:layers.allBounds?doc.entities.filter((e:any)=>!e.sourceContext).map((e:any)=>e.id):[selection.entityId];for(const id of ids){if(!id)continue;const ps=selectedGeometry(id).corners;if(!ps.length)continue;const box=ps.map(project);for(let i=0;i<8;i++)for(let k=0;k<3;k++)if(!(i&(1<<k)))line(box[i],box[i|(1<<k)],id===selection.entityId?'#7ae6cf':'#607e89');}
    const axisId=layers.axisEntityId||selection.entityId,e=entity(axisId),axes=selectedAxes(axisId);if(e&&axes){const {geometry,origin,length}=axes,t=geometry.transform;
      for(let k=0;k<3;k++){const {direction,end,color}=axes.axes[k],a=project(origin),b=project(end),labelSize=overlay?24:14,labelOffset=overlay?8:5;line(a,b,color,overlay?5:3);if(!a||!b)continue;
        const label=document.createElementNS(svg.namespaceURI,'text');label.textContent='XYZ'[k];for(const[key,value]of Object.entries({x:b[0]+labelOffset,y:b[1]-labelOffset,fill:color,'font-size':labelSize,'font-weight':700}))label.setAttribute(key,String(value));svg.append(label);
        if(overlay){overlay.font=`700 ${labelSize}px sans-serif`;overlay.lineWidth=5;overlay.strokeStyle='#ffffff';overlay.strokeText('XYZ'[k],b[0]+labelOffset,b[1]-labelOffset);overlay.fillStyle=color;overlay.fillText('XYZ'[k],b[0]+labelOffset,b[1]-labelOffset);}
        if(t&&geometry.editable){const handle=line(a,b,'transparent',18)! as SVGElement;handle.style.pointerEvents='stroke';handle.style.cursor='move';handle.setAttribute('role','button');handle.setAttribute('aria-label',`Move ${'XYZ'[k]}`);handle.setAttribute('tabindex','0');handle.addEventListener('pointerdown',(ev:any)=>{ev.preventDefault();ev.stopPropagation();axisDrag={id:e.id,t:structuredClone(t),direction,a,b,length,x:ev.clientX,y:ev.clientY,changed:false};stage.setPointerCapture(ev.pointerId);});handle.addEventListener('keydown',(ev:any)=>{if(['ArrowLeft','ArrowDown','ArrowRight','ArrowUp'].includes(ev.key)){ev.preventDefault();const sign=['ArrowLeft','ArrowDown'].includes(ev.key)?-1:1,nt={...t,position:add(t.position,scale(direction,sign*radius*.01))};emit('transformCommitIntent',{operations:[{type:'setTransform',entityId:e.id,...nt}]});}});}
      }
    }
  }
  async function url(id:string){const r=await options.resolveAsset(id);return typeof r==='string'?r:r.url;}
  async function setPhoto(){const n=++photoEpoch;photoAbort.abort();photoAbort=new AbortController();photo.hidden=true;if(photoObjectURL){URL.revokeObjectURL(photoObjectURL);photoObjectURL=null;}if(!camera?.exact)return;const f=camera.frame;let objectURL:string|null=null;try{const src=await url(f.imageId);if(disposed||n!==photoEpoch)return;const response=await fetch(src,{signal:photoAbort.signal});if(!response.ok)throw Error('photo_load_failed');objectURL=URL.createObjectURL(await response.blob());const img=new Image();img.src=objectURL;await img.decode();if(disposed||n!==photoEpoch){URL.revokeObjectURL(objectURL);return;}photoObjectURL=objectURL;photo.src=objectURL;photo.hidden=false;emit('renderReady',{phase:'photo',cameraId:f.id});}catch(error:any){if(objectURL&&objectURL!==photoObjectURL)URL.revokeObjectURL(objectURL);if(n===photoEpoch&&error.name!=='AbortError')emit('loadError',{code:'photo_load_failed',assetId:f.imageId});}}
  function setCamera(value:any){
    if(disposed)return;viewMode=typeof value==='string'?value:value?.mode||'free';const f=currentCameras(doc).find((c:any)=>c.id===(typeof value==='string'?value:value?.cameraId));if(f){frameId=f.coordinateFrameId;layers.imageId=f.imageId;layers.observations=doc.observations;}if(f&&(typeof value==='string'||value?.mode==='photo')){camera=sourceCamera(f,radius,center);setPhoto();draw();return;}
    dimensions();const mode=typeof value==='string'?value:value?.mode||'free';if(value?.eye){navigationVersion++;camera=value;photo.hidden=true;draw();return;}
    camera=fittedCamera(mode);photoEpoch++;photoAbort.abort();photo.hidden=true;draw();
  }
  function fittedCamera(mode:string){
    const frame=doc.coordinateFrames.find((f:any)=>f.id===frameId),up=unit(frame?.ground?.normal||[0,0,1]),reference=currentCameras(doc).find((c:any)=>c.coordinateFrameId===frameId),rawFront=reference?reference.cameraToWorld.slice(0,3).map((r:Vec)=>-r[2]):[0,-1,0];let planar=add(rawFront,scale(up,-dot(rawFront,up)));if(Math.hypot(...planar)<1e-6){const axis=Math.abs(up[0])<.8?[1,0,0]:[0,1,0];planar=add(axis,scale(up,-dot(axis,up)));}const front=unit(planar),right=unit(cross(up,front)),back=mode==='top'?up:mode==='side'?right:mode==='front'?front:unit(add(add(front,scale(right,.45)),scale(up,.55))),vup=mode==='top'?scale(front,-1):up,ps=fittingPoints();
    if(layers.axisEntityId){const axes=selectedAxes(layers.axisEntityId);if(axes)ps.push(axes.origin,...axes.axes.map(axis=>axis.end));}
    return fitCamera(ps.length?ps:boundsCorners({min:[-1,-1,-1],max:[1,1,1]}),back,vup,Math.max(captureSize?.w||stage.clientWidth,1)/Math.max(captureSize?.h||stage.clientHeight,1),['top','front','side'].includes(mode));
  }
  function captureModel(entityId:string,mode:'free'|'front'|'side'|'top',expectedRevisionId:string,expectedFrameId?:string){
    if(disposed||revisionId!==expectedRevisionId||expectedFrameId!==undefined&&expectedFrameId!==frameId||gl!.isContextLost())return null;
    const subject=entity(entityId),rep=subject?.representations?.find((r:any)=>r.id===subject.activeModelRepresentationId);
    if(!rep||!loadedRepresentations.has(entityId+'/'+rep.id)||!representationAvailable(subject,rep,frameId,true)||!['generated_mesh','primitive'].includes(rep.kind))return null;
    const saved={camera,layers,selection,radius,center,width:canvas.width,height:canvas.height,canvasStyle:canvas.style.cssText,photoStyle:photo.style.cssText};
    try{
      // ponytail: one synchronous capture reuses the scene GPU buffers; restore
      // before yielding so object previews cannot change scene navigation.
      captureSize={w:640,h:640,cw:640,ch:640};
      // Studio shading lifts display shadows only; mesh colors/materials remain unchanged.
      layers={...layers,studio:true,modelOnly:true,entityId,axisEntityId:entityId,showAxes:true,observed_surface:false,point_cloud:false,generated_mesh:true,primitive:true,showCandidates:true,showBounds:false,editable:false};
      selection={};dimensions();camera=fittedCamera(mode);const captureCanvas=document.createElement('canvas');if(!captureCanvas.getContext('2d'))throw Error('canvas_2d_unavailable');draw(false,captureCanvas);return captureCanvas.toDataURL('image/png');
    }finally{
      camera=saved.camera;layers=saved.layers;selection=saved.selection;radius=saved.radius;center=saved.center;captureSize=null;
      canvas.width=saved.width;canvas.height=saved.height;canvas.style.cssText=saved.canvasStyle;photo.style.cssText=saved.photoStyle;draw();
    }
  }
  async function upload(mesh:Mesh,entityId:string,representation:any,n:number){
    if(disposed||n!==epoch)return;const vertex=gl!.createBuffer()!,index=gl!.createBuffer()!,texture=gl!.createTexture()!;gl!.bindBuffer(gl!.ARRAY_BUFFER,vertex);gl!.bufferData(gl!.ARRAY_BUFFER,mesh.vertices,gl!.STATIC_DRAW);gl!.bindBuffer(gl!.ELEMENT_ARRAY_BUFFER,index);gl!.bufferData(gl!.ELEMENT_ARRAY_BUFFER,mesh.indices,gl!.STATIC_DRAW);gl!.bindTexture(gl!.TEXTURE_2D,texture);gl!.texParameteri(gl!.TEXTURE_2D,gl!.TEXTURE_MIN_FILTER,gl!.LINEAR);gl!.texParameteri(gl!.TEXTURE_2D,gl!.TEXTURE_MAG_FILTER,gl!.LINEAR);gl!.texParameteri(gl!.TEXTURE_2D,gl!.TEXTURE_WRAP_S,gl!.CLAMP_TO_EDGE);gl!.texParameteri(gl!.TEXTURE_2D,gl!.TEXTURE_WRAP_T,gl!.CLAMP_TO_EDGE);gl!.texImage2D(gl!.TEXTURE_2D,0,gl!.RGBA,1,1,0,gl!.RGBA,gl!.UNSIGNED_BYTE,new Uint8Array([255,255,255,255]));
    const g={mesh,vertex,index,texture,entityId,representation};gpu.push(g);if(mesh.texture){const bitmap=await createImageBitmap(mesh.texture,{imageOrientation:'none',premultiplyAlpha:'none'});if(!disposed&&n===epoch){gl!.bindTexture(gl!.TEXTURE_2D,texture);gl!.pixelStorei(gl!.UNPACK_FLIP_Y_WEBGL,false);gl!.texImage2D(gl!.TEXTURE_2D,0,gl!.RGBA,gl!.RGBA,gl!.UNSIGNED_BYTE,bitmap);}bitmap.close();}if(n===epoch){dimensions();draw();}
  }
  async function setScene(revision:any){
    const next=revision.document||revision;if(!next||!Array.isArray(next.entities)||!Array.isArray(next.cameras))throw Error('invalid_scene_document');const signature=JSON.stringify(next.entities.map((e:any)=>[e.id,(e.representations||[]).map((r:any)=>[r.id,r.assetId,r.kind,r.primitive,r.placementState,r.placementReason,r.sourceValidity])]));
    if(signature===sceneAssetsSignature&&doc.captureId===next.captureId){doc=next;layers.observations=doc.observations;if(layers.imageId)frameId=cameraForImage(doc,layers.imageId)?.coordinateFrameId||null;revisionId=revision.id||'';preview.clear();for(const g of gpu){const r=entity(g.entityId)?.representations?.find((r:any)=>r.id===g.representation.id);if(r)g.representation=r;}dimensions();draw();return;}
    sceneAssetsSignature=signature;const n=++epoch;abort.abort();abort=new AbortController();release();preview.clear();doc=next;layers.observations=doc.observations;revisionId=revision.id||'';frameId=currentCameras(doc).find((c:any)=>c.id===selection.cameraId)?.coordinateFrameId||currentCameras(doc)[0]?.coordinateFrameId||(!doc.cameras.length?doc.coordinateFrames?.[0]?.id:null)||null;
    if(!camera){if(currentCameras(doc)[0])setCamera(currentCameras(doc)[0].id);else setCamera('free');}else if(camera.exact&&currentCameras(doc).some((c:any)=>c.id===camera!.frame.id))setCamera(camera.frame.id);
    emit('loadProgress',{phase:'metadata',loaded:0,total:doc.entities.length});
    const fitVersion=navigationVersion,initialRadius=radius;const tasks=doc.entities.flatMap((e:any)=>(e.representations||[]).filter((r:any)=>r.sourceValidity!=='stale'&&(r.placementState==='confirmed'||['requires_alignment_confirmation','imported_proposal'].includes(r.placementReason))).map((r:any)=>({e,r})));let finished=0;
    // Two downloads at a time bounds decode memory on phones. Late data may draw,
    // but never writes the host's selection or camera.
    async function load(){while(tasks.length&&!disposed&&n===epoch){const {e,r}=tasks.shift()!;try{let meshes:Mesh[];if(r.kind==='primitive')meshes=[primitive(r.primitive)];else{const src=await url(r.assetId);if(n!==epoch||disposed)return;const res=await fetch(src,{signal:abort.signal});if(!res.ok)throw Error('asset_download_failed');const bytes=await res.arrayBuffer();emit('loadProgress',{phase:'gpu_upload',assetId:r.assetId,bytes:bytes.byteLength});const meta=doc.assets.find((a:any)=>a.id===r.assetId);meshes=(meta?.format||meta?.metadata?.format)==='panoptes-mesh-v1'?readPacked(bytes,meta):readGLB(bytes);}for(const mesh of meshes)await upload(mesh,e.id,r,n);if(n===epoch&&!disposed)loadedRepresentations.add(e.id+'/'+r.id);}catch(error:any){if(n===epoch&&error.name!=='AbortError')emit('loadError',{code:error.message,entityId:e.id,assetId:r.assetId});}if(n===epoch){finished++;emit('loadProgress',{phase:'assets',loaded:finished,total:finished+tasks.length});}}}
    await Promise.all([load(),load()]);if(n===epoch&&!disposed){dimensions();if(!camera?.exact&&navigationVersion===fitVersion&&initialRadius!==radius)setCamera(viewMode);draw();emit('renderReady',{phase:'scene',revisionId});}
  }
  function listen(target:EventTarget,name:string,fn:any,opts?:any){target.addEventListener(name,fn,opts);cleanups.push(()=>target.removeEventListener(name,fn,opts));}
  listen(canvas,'pointerdown',(e:PointerEvent)=>{drag={x:e.clientX,y:e.clientY,b:e.button,moved:0};canvas.setPointerCapture(e.pointerId);});
  listen(stage,'pointermove',(e:PointerEvent)=>{if(axisDrag){const d=axisDrag,dx=d.b[0]-d.a[0],dy=d.b[1]-d.a[1],den=dx*dx+dy*dy;if(den<4)return;const amount=((e.clientX-d.x)*dx+(e.clientY-d.y)*dy)/den*d.length,t={...d.t,position:add(d.t.position,scale(d.direction,amount))};d.changed=true;preview.set(d.id,t);emit('transformPreview',{operations:[{type:'setTransform',entityId:d.id,...t}]});draw();return;}
    if(!drag||!camera)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;drag.moved+=Math.abs(dx)+Math.abs(dy);if(camera.exact||drag.moved<4)return;navigationVersion++;let offset=add(camera.eye,scale(camera.target,-1)),right=unit(cross(camera.up,offset));if(drag.b===2||e.shiftKey){const mv=add(scale(right,-dx*radius*.003),scale(camera.up,dy*radius*.003));camera.eye=add(camera.eye,mv);camera.target=add(camera.target,mv);}else{offset=rotate(offset,unit(camera.up),-dx*.006);right=unit(cross(camera.up,offset));offset=rotate(offset,right,-dy*.006);camera.up=rotate(camera.up,right,-dy*.006);camera.eye=add(camera.target,offset);}draw();});
  listen(stage,'pointerup',(e:PointerEvent)=>{if(axisDrag){const d=axisDrag;axisDrag=null;if(d.changed)emit('transformCommitIntent',{operations:[{type:'setTransform',entityId:d.id,...preview.get(d.id)}]});return;}const prev=drag;drag=null;if(!prev||prev.b!==0||prev.moved>=4||!camera)return;const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;if(x<0||y<0||x>=rect.width||y>=rect.height)return;draw(true);const px=new Uint8Array(4);gl!.readPixels(Math.floor(x*canvas.width/rect.width),canvas.height-1-Math.floor(y*canvas.height/rect.height),1,1,gl!.RGBA,gl!.UNSIGNED_BYTE,px);draw();const i=px[0]+(px[1]<<8)+(px[2]<<16)-1;emit('selectionIntent',{entityId:doc.entities[i]?.id||null,cameraId:camera.exact?camera.frame.id:null,originalPixel:camera.exact?[x*camera.frame.width/rect.width-.5,y*camera.frame.height/rect.height-.5]:null});});
  listen(stage,'pointercancel',()=>{if(axisDrag)preview.delete(axisDrag.id);axisDrag=null;drag=null;emit('transformPreview',{operations:[]});draw();});listen(canvas,'contextmenu',(e:Event)=>e.preventDefault());
  listen(canvas,'wheel',(e:WheelEvent)=>{if(!camera||camera.exact)return;e.preventDefault();navigationVersion++;const offset=add(camera.eye,scale(camera.target,-1)),factor=Math.exp(e.deltaY*.0012);if(camera.orthographic)camera.orthoHeight=Math.max(radius*.02,Math.min(radius*50,camera.orthoHeight!*factor));else camera.eye=add(camera.target,scale(offset,Math.max(radius*.02,Math.min(radius*50,Math.hypot(...offset)*factor))/Math.hypot(...offset)));draw();},{passive:false});
  listen(canvas,'webglcontextlost',(e:Event)=>{e.preventDefault();emit('contextLost');});listen(canvas,'webglcontextrestored',()=>emit('loadError',{code:'viewer_remount_required'}));const observer=new ResizeObserver(()=>draw());observer.observe(stage);
  return {setScene,captureModel,setSelection(value:any){selection={...value};draw();},setCamera,setLayers(value:any){layers={...layers,...value};if(layers.imageId)frameId=cameraForImage(doc,layers.imageId)?.coordinateFrameId||null;dimensions();draw();},previewOperations(ops:any[]){for(const op of ops)if(op.type==='setTransform')preview.set(op.entityId,op);draw();},clearPreview(){preview.clear();draw();},resize(){draw();},dispose(){if(disposed)return;disposed=true;epoch++;photoEpoch++;abort.abort();photoAbort.abort();if(photoObjectURL)URL.revokeObjectURL(photoObjectURL);observer.disconnect();cleanups.forEach(fn=>fn());release();shaders.forEach(s=>gl.deleteShader(s));gl.deleteProgram(program);gl.getExtension('WEBGL_lose_context')?.loseContext();photo.removeAttribute('src');stage.remove();}};
}
