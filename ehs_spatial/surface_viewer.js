// ponytail: this reader supports our validated, face-local, unlit GLB export only.
// A different GLB schema must be rejected or deliberately supported by the importer.
function readSurfaceGLB(buffer, faceBuffer, manifest) {
  const fail = message => { throw new Error('内部模型：' + message); };
  const dv = new DataView(buffer);
  if (buffer.byteLength < 20 || dv.getUint32(0, true) !== 0x46546c67 || dv.getUint32(4, true) !== 2
      || dv.getUint32(8, true) !== buffer.byteLength) fail('无效 GLB 2 文件');
  let doc, bin;
  for (let offset = 12; offset < buffer.byteLength;) {
    if (offset + 8 > buffer.byteLength) fail('截断的 GLB chunk');
    const size = dv.getUint32(offset, true), type = dv.getUint32(offset + 4, true); offset += 8;
    if (offset + size > buffer.byteLength) fail('截断的 GLB 数据');
    if (type === 0x4e4f534a && !doc) doc = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, offset, size)));
    else if (type === 0x004e4942 && !bin) bin = {offset, size};
    else fail('不支持的 GLB chunk');
    offset += size;
  }
  if (!doc || !bin || doc.asset?.version !== '2.0' || doc.extensionsRequired?.length
      || doc.buffers?.length !== 1 || doc.buffers[0].uri || doc.buffers[0].byteLength > bin.size
      || doc.nodes?.length !== 1 || doc.nodes[0].mesh !== 0 || doc.nodes[0].matrix
      || doc.nodes[0].translation || doc.nodes[0].rotation || doc.nodes[0].scale || doc.nodes[0].children
      || doc.meshes?.length !== 1 || doc.meshes[0].primitives?.length !== 1) fail('不支持的场景或坐标变换');
  const primitive = doc.meshes[0].primitives[0];
  if ((primitive.mode ?? 4) !== 4 || primitive.extensions || primitive.targets) fail('需要普通三角面');
  function view(index) {
    const v = doc.bufferViews?.[index];
    if (!v || v.buffer !== 0 || v.byteStride || (v.byteOffset || 0) < 0 || v.byteLength < 0
        || (v.byteOffset || 0) + v.byteLength > doc.buffers[0].byteLength) fail('无效 buffer view');
    return {offset:bin.offset + (v.byteOffset || 0), size:v.byteLength};
  }
  function accessor(index, component, type, width, Type) {
    const a = doc.accessors?.[index];
    if (!a || a.componentType !== component || a.type !== type || a.sparse || a.normalized
        || !Number.isInteger(a.count) || a.count < 1) fail('不支持的三角面 accessor');
    const v = view(a.bufferView), offset = a.byteOffset || 0, bytes = a.count * width * Type.BYTES_PER_ELEMENT;
    if (offset < 0 || offset + bytes > v.size || (v.offset + offset) % Type.BYTES_PER_ELEMENT) fail('无效 accessor 长度');
    return new Type(buffer, v.offset + offset, a.count * width);
  }
  const indices = accessor(primitive.indices, 5125, 'SCALAR', 1, Uint32Array);
  const positions = accessor(primitive.attributes.POSITION, 5126, 'VEC3', 3, Float32Array);
  const uv = accessor(primitive.attributes.TEXCOORD_0, 5126, 'VEC2', 2, Float32Array);
  if (indices.length % 3 || positions.length !== indices.length * 3 || uv.length !== indices.length * 2
      || indices.length / 3 !== manifest.face_count || faceBuffer.byteLength !== manifest.face_count * 4) fail('面数量不匹配');
  for (let i = 0; i < indices.length; i++) if (indices[i] !== i) fail('需要逐面独立且按顺序排列的顶点');
  if (!positions.every(Number.isFinite) || !uv.every(v => Number.isFinite(v) && v >= 0 && v <= 1)) fail('无效位置或 UV');
  const faceIds = new Uint32Array(faceBuffer), supported = new Set(manifest.supported_inv), found = new Set();
  const vertexIds = new Float32Array(indices.length);
  faceIds.forEach((id, face) => {
    if (id && (!supported.has(id - 1) || id > 65535)) fail('面指向未验证的 inv');
    if (id) found.add(id - 1);
    vertexIds.fill(id, face * 3, face * 3 + 3);
  });
  if (found.size !== supported.size) fail('supported_inv 与面关联不一致');
  const material = doc.materials?.[primitive.material], pbr = material?.pbrMetallicRoughness;
  if (!pbr || (pbr.baseColorFactor || [1,1,1,1]).some(v => v !== 1)
      || (material.alphaMode || 'OPAQUE') !== 'OPAQUE' || pbr.baseColorTexture?.texCoord
      || pbr.baseColorTexture?.extensions) fail('需要白色调制的原图纹理材质');
  const texture = doc.textures?.[pbr.baseColorTexture?.index], img = doc.images?.[texture?.source];
  if (!img || img.uri || img.mimeType !== 'image/png') fail('需要内嵌 PNG 原图纹理');
  const imageView = view(img.bufferView);
  return {positions, uv, vertexIds, count:indices.length,
    image:new Uint8Array(buffer, imageView.offset, imageView.size)};
}

const svAdd = (a,b) => a.map((v,i) => v + b[i]);
const svScale = (a,s) => a.map(v => v*s);
const svDot = (a,b) => a.reduce((s,v,i) => s + v*b[i], 0);
const svCross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const svUnit = a => svScale(a,1/Math.hypot(...a));
function surfaceCamera(frame, radius) {
  const C = frame.camera_to_world, K = frame.K;
  if (!Array.isArray(C) || C.length !== 4 || C.some(r => r.length !== 4 || !r.every(Number.isFinite))
      || !Array.isArray(K) || K.length !== 3 || K.some(r => r.length !== 3 || !r.every(Number.isFinite))
      || !(K[0][0] > 0 && K[1][1] > 0 && frame.width > 0 && frame.height > 0)) throw new Error('无效源相机');
  const eye = C.slice(0,3).map(r => r[3]), forward = C.slice(0,3).map(r => r[2]);
  return {eye, target:svAdd(eye, svScale(forward,radius)), up:C.slice(0,3).map(r => -r[1]), frame, exact:true};
}
function surfaceMatrix(camera, aspect, radius) {
  const {eye,target,up,frame,exact} = camera, near = radius*0.001, far = radius*60;
  let l,r,t,b;
  if (exact) {
    const K = frame.K;
    l = -(K[0][2]+.5)/K[0][0]*near; r = (frame.width-K[0][2]-.5)/K[0][0]*near;
    t = (K[1][2]+.5)/K[1][1]*near; b = -(frame.height-K[1][2]-.5)/K[1][1]*near;
  } else { t = near*frame.height/(2*frame.K[1][1]); b = -t; r = t*aspect; l = -r; }
  const P = [2*near/(r-l),0,0,0, 0,2*near/(t-b),0,0,
    (r+l)/(r-l),(t+b)/(t-b),-(far+near)/(far-near),-1, 0,0,-2*far*near/(far-near),0];
  const z = svUnit(svAdd(eye,svScale(target,-1))), x = svUnit(svCross(up,z)), y = svCross(z,x);
  const V = [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
    -svDot(x,eye),-svDot(y,eye),-svDot(z,eye),1];
  const M = new Float32Array(16);
  for (let i=0;i<4;i++) for (let j=0;j<4;j++) for (let k=0;k<4;k++) M[i*4+j] += V[i*4+k]*P[k*4+j];
  return M;
}

// Both renderers use the actual displayed geometry in their own coordinate frame.
// Native surface coordinates have no verified registration to the measurement floor.
function surfaceObjectBounds(positions, vertexIds) {
  const bounds=new Map();
  for(let i=0;i<vertexIds.length;i++){
    const inv=vertexIds[i]-1;if(inv<0)continue;
    if(!bounds.has(inv))bounds.set(inv,{min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]});
    const box=bounds.get(inv);for(let k=0;k<3;k++){const v=positions[i*3+k];box.min[k]=Math.min(box.min[k],v);box.max[k]=Math.max(box.max[k],v);}
  }
  return bounds;
}
function drawSelectionReference({canvas,matrix,objects,floor=false,scaleSource=null}) {
  const svg=document.getElementById('selection-reference'),info=document.getElementById('selection-reference-info'),host=svg.parentElement;
  const width=host.clientWidth,height=host.clientHeight;if(!width||!height)return;
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.dataset.referenceFrame=floor?'estimated-floor':'surface-native';svg.dataset.objectCount=String(objects.length);
  info.hidden=!objects.length;delete info.dataset.height;
  const stage=host.getBoundingClientRect(),rect=canvas.getBoundingClientRect();
  const project=p=>{const q=[0,1,2,3].map(r=>matrix[r]*p[0]+matrix[4+r]*p[1]+matrix[8+r]*p[2]+matrix[12+r]);if(q[3]<=0||q[2]/q[3]<-1||q[2]/q[3]>1)return null;return[rect.left-stage.left+(q[0]/q[3]+1)*rect.width/2,rect.top-stage.top+(1-q[1]/q[3])*rect.height/2];};
  const line=(a,b,color,attrs='')=>a&&b?`<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color}" ${attrs}/>`:'';
  let markup='';for(const object of objects){const b=object.bounds,center=b.min.map((v,k)=>(v+b.max[k])/2),corners=Array.from({length:8},(_,i)=>project(b.min.map((v,k)=>i&(1<<k)?b.max[k]:v))),origin=project(center),length=Math.max(Math.hypot(...b.max.map((v,k)=>v-b.min[k]))*.28,.01);
    const key=String(object.key).replace(/[&<>"']/g,'');markup+=`<g data-selection-key="${key}">`;
    for(let i=0;i<8;i++)for(let k=0;k<3;k++)if(!(i&(1<<k)))markup+=line(corners[i],corners[i|(1<<k)],'#9de8dd','stroke-width="1.3" data-bbox-edge=""');
    for(let k=0;k<3;k++){const end=project(center.map((v,j)=>v+(j===k?length:0))),color=['#ff6464','#61de86','#65a7ff'][k],label='XYZ'[k];if(!origin||!end)continue;const dx=end[0]-origin[0],dy=end[1]-origin[1],d=Math.hypot(dx,dy)||1,nx=dx/d,ny=dy/d;
      markup+=`<g data-axis="${label}">${line(origin,end,color,'stroke-width="3"')}<polygon fill="${color}" points="${end[0]},${end[1]} ${end[0]-nx*9-ny*4},${end[1]-ny*9+nx*4} ${end[0]-nx*9+ny*4},${end[1]-ny*9-nx*4}"/><text x="${end[0]+nx*11+2}" y="${end[1]+ny*11+4}" fill="${color}" font-size="13" font-weight="700" stroke="#10171e" stroke-width="3" paint-order="stroke">${label}</text></g>`;
    }
    if(floor)markup+=line(project([center[0],center[1],0]),project([center[0],center[1],b.max[2]]),'#d6d9e3','stroke-width="1.5" stroke-dasharray="4 3" data-height-line=""');
    markup+='</g>';
  }
  svg.innerHTML=markup;
  const en=document.documentElement.lang.startsWith('en'),units=scaleSource==='camera_height'?(en?'m · camera-height scale':'m · 相机高度尺度'):scaleSource==='moge_anchor'?(en?'m · model-estimated scale':'m · 模型估计尺度'):(en?'scene units · uncalibrated':'场景单位 · 未标定');
  let message='';if(objects.length){
    if(floor&&objects.length===1){const b=objects[0].bounds,span=b.max[2]-b.min[2];info.dataset.height=String(span);message=(en?'Displayed-point height span ':'显示点高度跨度 ')+span.toFixed(3)+' '+units+'\n'+(en?'Z range to estimated floor: ':'相对估计地面的 Z 范围：')+b.min[2].toFixed(3)+' → '+b.max[2].toFixed(3)+'. ';}
    else message=(en?objects.length+' selected geometry bounds. ':objects.length+' 个选中几何包围框。');
    message+=floor?(en?'Floor-reference XYZ; object upright and measured tilt unknown.':'地面参考 XYZ；物体竖直轴与实测倾角未知。'):(en?'Native scene XYZ; floor height and measured tilt unknown. Uncalibrated units.':'原生场景 XYZ；地面高度与实测倾角未知。单位未标定。');
  }
  if(info.textContent!==message)info.textContent=message;
}

async function createSurfaceViewer({canvas, manifest, inventoryCount, onPick}) {
  async function bytes(url) { const response = await fetch(url); if (!response.ok) throw new Error('模型资源加载失败：'+response.status); return response.arrayBuffer(); }
  const [glb, mapping] = await Promise.all([bytes(manifest.asset_url), bytes(manifest.face_map_url)]);
  const mesh = readSurfaceGLB(glb, mapping, manifest);
  const objectBounds=surfaceObjectBounds(mesh.positions,mesh.vertexIds);
  const gl = canvas.getContext('webgl', {antialias:false, alpha:false});
  if (!gl) throw new Error('无法初始化内部模型 WebGL');
  gl.disable(gl.DITHER); gl.disable(gl.CULL_FACE); gl.enable(gl.DEPTH_TEST);
  const vs = `attribute vec3 p; attribute vec2 uv; attribute float inv;
    uniform mat4 mvp; varying vec2 vUV; varying float vInv;
    void main(){gl_Position=mvp*vec4(p,1.0);vUV=uv;vInv=inv;}`;
  const fs = `precision highp float; varying vec2 vUV; varying float vInv;
    uniform sampler2D photo; uniform sampler2D state; uniform float n; uniform float anySel; uniform float pick;
    void main(){vec4 s=texture2D(state,vec2((vInv+.5)/n,.5)); if(s.a<.1)discard;
      if(pick>.5){gl_FragColor=vec4(floor(vInv/256.0)/255.0,mod(vInv,256.0)/255.0,1.0,1.0);return;}
      vec3 c=texture2D(photo,vUV).rgb;
      if(anySel>.5){if(s.a>.9)c=mix(c,vec3(.22,.79,1.),.5);else c*=.38;}
      gl_FragColor=vec4(c,1.0);}`;
  const prog = gl.createProgram();
  for (const [type,code] of [[gl.VERTEX_SHADER,vs],[gl.FRAGMENT_SHADER,fs]]) {
    const shader = gl.createShader(type); gl.shaderSource(shader,code); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(prog,shader);
  }
  gl.linkProgram(prog); if (!gl.getProgramParameter(prog,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  for (const [name,data,size] of [['p',mesh.positions,3],['uv',mesh.uv,2],['inv',mesh.vertexIds,1]]) {
    gl.bindBuffer(gl.ARRAY_BUFFER,gl.createBuffer()); gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog,name); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,size,gl.FLOAT,false,0,0);
  }
  const uniform = name => gl.getUniformLocation(prog,name);
  function texture(unit, filter) {
    gl.activeTexture(gl.TEXTURE0+unit); const value = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,value);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,filter); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,filter); return value;
  }
  texture(0,gl.LINEAR);
  const img = new Image(), url = URL.createObjectURL(new Blob([mesh.image],{type:'image/png'}));
  try { await new Promise((resolve,reject) => { img.onload=resolve; img.onerror=()=>reject(new Error('原图纹理无法解码')); img.src=url; });
    // glTF UV v=0 is the source image's top row. Upload DOM pixels without a Y flip.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,img);
  } finally { URL.revokeObjectURL(url); }
  gl.uniform1i(uniform('photo'),0);
  const stateTexture = texture(1,gl.NEAREST), state = new Uint8Array((inventoryCount+1)*4).fill(170);
  gl.uniform1i(uniform('state'),1); gl.uniform1f(uniform('n'),inventoryCount+1);
  const bounds = manifest.cameras.bounds;
  const radius = Math.hypot(...bounds.max.map((v,i) => v-bounds.min[i]))/2;
  if (!(radius>0 && Number.isFinite(radius))) throw new Error('无效内部模型边界');
  let camera, drag, active = true, hasSelection = false, selectedBounds=[];
  function draw(pick=false) {
    if (!active || !camera) return;
    const parent = canvas.parentElement, w = parent.clientWidth, h = parent.clientHeight, ratio = camera.frame.width/camera.frame.height;
    const cw = camera.exact ? Math.min(w,h*ratio) : w, ch = camera.exact ? cw/ratio : h;
    canvas.style.width=cw+'px'; canvas.style.height=ch+'px';
    const dpr=Math.min(devicePixelRatio||1,2), pw=Math.max(1,Math.round(cw*dpr)), ph=Math.max(1,Math.round(ch*dpr));
    if (canvas.width!==pw || canvas.height!==ph) {canvas.width=pw;canvas.height=ph;}
    gl.viewport(0,0,pw,ph); gl.clearColor(.059,.071,.086,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.uniformMatrix4fv(uniform('mvp'),false,surfaceMatrix(camera,pw/ph,radius));
    gl.uniform1f(uniform('pick'),pick?1:0); gl.uniform1f(uniform('anySel'),hasSelection?1:0);
    gl.drawArrays(gl.TRIANGLES,0,mesh.count);
    if(!pick)drawSelectionReference({canvas,matrix:surfaceMatrix(camera,pw/ph,radius),objects:selectedBounds});
  }
  function setSelection(selected, hidden=[]) {
    const visible = new Set(manifest.supported_inv); hasSelection=selected.some(i=>visible.has(i)); state.fill(170);
    hidden.forEach(i=>{if(Number.isInteger(i))state[(i+1)*4+3]=0;});
    selected.forEach(i=>{if(visible.has(i))state[(i+1)*4+3]=255;});
    selectedBounds=selected.filter(i=>objectBounds.has(i)&&!hidden.includes(i)).map(i=>({key:i,bounds:objectBounds.get(i)}));
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D,stateTexture);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,inventoryCount+1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,state); draw();
  }
  function goTo(index) {camera=surfaceCamera(manifest.cameras.frames[index],radius);draw();}
  const rotate = (v,axis,angle) => svAdd(svAdd(svScale(v,Math.cos(angle)),svScale(svCross(axis,v),Math.sin(angle))),svScale(axis,svDot(axis,v)*(1-Math.cos(angle))));
  canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,b:e.button,moved:0};canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging');});
  canvas.addEventListener('pointermove',e=>{
    if(!drag)return; const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;drag.moved+=Math.abs(dx)+Math.abs(dy);
    if(!dx&&!dy)return;camera.exact=false;
    let offset=svAdd(camera.eye,svScale(camera.target,-1));const right=svUnit(svCross(camera.up,offset));
    if(drag.b===2 || e.shiftKey){const scale=Math.hypot(...offset)*.0016;const move=svAdd(svScale(right,-dx*scale),svScale(camera.up,dy*scale));camera.eye=svAdd(camera.eye,move);camera.target=svAdd(camera.target,move);}
    else {offset=rotate(offset,svUnit(camera.up),-dx*.006);const axis=svUnit(svCross(camera.up,offset));offset=rotate(offset,axis,-dy*.006);camera.up=rotate(camera.up,axis,-dy*.006);camera.eye=svAdd(camera.target,offset);}draw();
  });
  canvas.addEventListener('pointerup',e=>{
    const previous=drag;drag=null;canvas.classList.remove('dragging');if(!previous||previous.b!==0||previous.moved>=4)return;
    const r=canvas.getBoundingClientRect(),x=Math.floor((e.clientX-r.left)*canvas.width/r.width),y=canvas.height-1-Math.floor((e.clientY-r.top)*canvas.height/r.height);
    if(x<0||y<0||x>=canvas.width||y>=canvas.height)return;
    draw(true);const px=new Uint8Array(4);gl.readPixels(x,y,1,1,gl.RGBA,gl.UNSIGNED_BYTE,px);draw();
    const id=px[2]===255?px[0]*256+px[1]:0;onPick(id?id-1:null,e,px[2]===255);
  });
  canvas.addEventListener('pointercancel',()=>{drag=null;canvas.classList.remove('dragging');});
  canvas.addEventListener('contextmenu',e=>e.preventDefault());
  canvas.addEventListener('wheel',e=>{e.preventDefault();camera.exact=false;const offset=svAdd(camera.eye,svScale(camera.target,-1)),d=Math.hypot(...offset);
    camera.eye=svAdd(camera.target,svScale(offset,Math.max(radius*.02,Math.min(radius*12,d*Math.exp(e.deltaY*.0012)))/d));draw();},{passive:false});
  setSelection([]);goTo(0);
  return {setSelection,goTo,draw,setActive(value){active=value;draw();},faceCount:mesh.count/3};
}
