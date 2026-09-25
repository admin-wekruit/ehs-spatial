// Photo-real appearance layer: 3D Gaussian splats (3DGS) drawn in the scene viewer's own WebGL2 context, under its
// meshes. Appearance only: never picked, measured or put in an object preview.
//
// splat32 file, 32 bytes per Gaussian, little-endian: f32 x,y,z (world, the report's frame); f32 sx,sy,sz (linear
// scales); u8 r,g,b,a (sRGB, a = opacity*255); u8 qw,qx,qy,qz = round(q*128+128). Most important first, so every
// prefix of the file is a coarser scene. The records go to the GPU as they arrive, byte for byte (two RGBA32UI texels
// each); the vertex shader builds each covariance, so streaming converts nothing on the main thread.
import type {Vec} from './native-math.ts';

export const SPLAT_BYTES = 32;
const ROW = 1024, BATCH = 16 * ROW, MIN_SORT_MS = 50;  // splats per texture row; records per upload; resorts at most ~20/s
// While the camera moves, an even subset (every k-th record) of about MOTION_SPLATS_PER_PIXEL per canvas pixel is drawn:
// the cost is blending depth per pixel, which grows several-fold once a small pane holds millions of splats. A prefix of
// the file would not do: its first records are the largest splats, not a thinner copy of the scene. Everything is drawn
// again once the camera has rested for SETTLE_MS.
const MOTION_SPLATS_PER_PIXEL = 4, MIN_MOTION_SPLATS = 65536, SETTLE_MS = 150;
export type SplatSource = {key: string; url: string; count: number; coordinateFrameId: string};

/** The report's gaussian_splats annotation when it follows the splat32 contract, else null; refinedCamerasAssetId is
 *  the trainer's cameras (JSON {fps, frames, c2w}), where the splats are sharpest, when the report carries them. */
export function splatAnnotation(document: {annotations?: unknown[] | null}) {
  const a = (document.annotations || []).find((a: any) => a?.kind === 'gaussian_splats') as any;
  return a && a.format === 'splat32' && typeof a.assetId === 'string' && typeof a.coordinateFrameId === 'string' && Number.isSafeInteger(a.count) && a.count > 0
    ? {assetId: a.assetId as string, count: a.count as number, coordinateFrameId: a.coordinateFrameId as string,
      refinedCamerasAssetId: typeof a.refinedCamerasAssetId === 'string' ? a.refinedCamerasAssetId as string : null} : null;
}

/** Record i as the vertex shader reads it; quaternion w,x,y,z normalised. */
export function splatRecord(view: DataView, i: number) {
  const o = i * SPLAT_BYTES, f = (k: number) => view.getFloat32(o + 4 * k, true), b = (k: number) => view.getUint8(o + 24 + k);
  const q = [4, 5, 6, 7].map(k => (b(k) - 128) / 128), n = Math.hypot(...q) || 1;
  return {position: [f(0), f(1), f(2)], scale: [f(3), f(4), f(5)], color: [b(0), b(1), b(2)], opacity: b(3) / 255, quaternion: q.map(v => v / n)};
}

/** Σ = R S Sᵀ Rᵀ as in 3DGS, from linear scales and a unit quaternion w,x,y,z. */
export function splatCovariance(s: Vec, [w, x, y, z]: Vec): number[][] {
  const R = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
  return R.map(a => R.map(b => a[0] * b[0] * s[0] * s[0] + a[1] * b[1] * s[1] * s[1] + a[2] * b[2] * s[2] * s[2]));
}

/** The vertex shader's EWA step in JS (kept for checks): 2D covariance [xx, xy, yy] in device pixels, y up as in NDC,
 *  of a Gaussian at world `center` with 3D covariance `cov`, for the viewer's column-major view and projection on a
 *  w x h viewport, with 3DGS's 0.3 px² low-pass dilation. Null where the shader culls the splat. */
export function projectedCovariance(center: Vec, cov: number[][], view: ArrayLike<number>, projection: ArrayLike<number>, w: number, h: number) {
  const t = [0, 1, 2].map(r => view[r] * center[0] + view[4 + r] * center[1] + view[8 + r] * center[2] + view[12 + r]);
  const clip = [0, 1, 2, 3].map(r => projection[r] * t[0] + projection[4 + r] * t[1] + projection[8 + r] * t[2] + projection[12 + r]);
  if (!(clip[3] > 0) || clip[2] < -clip[3] || Math.abs(clip[0]) > 1.3 * clip[3] || Math.abs(clip[1]) > 1.3 * clip[3]) return null;
  // Rows of d(pixel)/d(world): the projection's Jacobian at t (NDC clamped to ±1.3 as 3DGS does), then the view rotation.
  const T = [0, 1].map(r => {
    const ndc = Math.max(-1.3, Math.min(1.3, clip[r] / clip[3])), half = (r ? h : w) / 2;
    const j = [0, 1, 2].map(k => half * (projection[4 * k + r] - ndc * projection[4 * k + 3]) / clip[3]);
    return [0, 1, 2].map(k => j[0] * view[4 * k] + j[1] * view[4 * k + 1] + j[2] * view[4 * k + 2]);
  });
  const quad = (a: Vec, b: Vec) => [0, 1, 2].reduce((sum, i) => sum + a[i] * (cov[i][0] * b[0] + cov[i][1] * b[1] + cov[i][2] * b[2]), 0);
  return [quad(T[0], T[0]) + .3, quad(T[0], T[1]), quad(T[1], T[1]) + .3];
}

/** Back-to-front order of every `stride`-th of the first `count` splats by their depth in front of the camera
 *  (z0..z3 = row 2 of the view matrix; an OpenGL camera looks down -z), by a 16-bit counting sort; returns how many it
 *  ordered. The key is the depth's float32 bit pattern, which grows like its logarithm: a few background splats
 *  kilometres away (trained scenes have them) cannot coarsen the order of the room, as a linear key would.
 *  Self-contained on purpose: its source text is also the worker's. */
export function sortSplats(positions: Float32Array, count: number, stride: number, z0: number, z1: number, z2: number, z3: number, order: Uint32Array, depth: Float32Array, key: Uint16Array, counts: Uint32Array) {
  const bits = new Uint32Array(depth.buffer, depth.byteOffset, depth.length);
  let lo = 4294967295, hi = 0, n = 0;
  for (let i = 0; i < count; i += stride, n++) {
    const d = -(z0 * positions[3 * i] + z1 * positions[3 * i + 1] + z2 * positions[3 * i + 2] + z3);
    depth[n] = d > 1e-6 ? d : 1e-6;  // behind the camera: culled by the shader anyway
    const b = bits[n]; if (b < lo) lo = b; if (b > hi) hi = b;
  }
  const s = hi > lo ? 65535 / (hi - lo) : 0;
  counts.fill(0);
  for (let j = 0; j < n; j++) { const k = 65535 - ((bits[j] - lo) * s | 0); key[j] = k; counts[k]++; }  // far first
  for (let k = 1; k < 65536; k++) counts[k] += counts[k - 1];
  for (let j = n - 1; j >= 0; j--) order[--counts[key[j]]] = j * stride;
  return n;
}

// The worker's whole program, sent as source text with the sort as its argument: it keeps the positions and sorts into
// the one order buffer the main thread lends it, so a resort allocates nothing.
function sortWorker(sort: typeof sortSplats) {
  const scope = self as any, counts = new Uint32Array(65536);
  let positions = new Float32Array(0), depth = new Float32Array(0), key = new Uint16Array(0), loaded = 0;
  scope.onmessage = (event: MessageEvent) => {
    const m = event.data;
    if (m.capacity) { positions = new Float32Array(m.capacity * 3); depth = new Float32Array(m.capacity); key = new Uint16Array(m.capacity); }
    else if (m.records) {
      const f = new Float32Array(m.records);
      for (let i = 0; i < m.count; i++) { const p = 3 * (m.start + i); positions[p] = f[8 * i]; positions[p + 1] = f[8 * i + 1]; positions[p + 2] = f[8 * i + 2]; }
      loaded = Math.max(loaded, m.start + m.count);
    } else {
      const started = performance.now(), total = Math.min(loaded, m.order.length);
      const n = sort(positions, total, Math.max(1, Math.ceil(total / m.limit)), m.z0, m.z1, m.z2, m.z3, m.order, depth, key, counts);
      scope.postMessage({order: m.order, count: n, ms: performance.now() - started}, [m.order.buffer]);
    }
  };
}

const VERTEX = `#version 300 es
precision highp float;precision highp int;precision highp usampler2D;
uniform usampler2D splats;uniform mat4 view;uniform mat4 projection;uniform vec2 viewport;
in vec2 corner;in uint splat;out vec4 color;out vec2 offset;
void main(){
  gl_Position=vec4(0.,0.,2.,1.);  // outside the clip volume: culled unless replaced below
  ivec2 at=ivec2(int(splat&1023u)<<1,int(splat>>10));
  uvec4 a=texelFetch(splats,at,0),b=texelFetch(splats,at+ivec2(1,0),0);
  vec4 clip=projection*(view*vec4(uintBitsToFloat(a.xyz),1.));
  if(clip.w<=0.||clip.z<-clip.w||any(greaterThan(abs(clip.xy),vec2(1.3*clip.w))))return;  // behind, too near or off screen; never too far
  color=vec4(b.z&255u,(b.z>>8)&255u,(b.z>>16)&255u,b.z>>24)/255.;
  if(color.a<1./255.)return;
  vec4 q=vec4(b.w&255u,(b.w>>8)&255u,(b.w>>16)&255u,b.w>>24)-128.;q=dot(q,q)>0.?normalize(q):vec4(1,0,0,0);
  float w=q.x,x=q.y,y=q.z,z=q.w;
  mat3 M=mat3(1.-2.*(y*y+z*z),2.*(x*y+w*z),2.*(x*z-w*y),2.*(x*y-w*z),1.-2.*(x*x+z*z),2.*(y*z+w*x),2.*(x*z+w*y),2.*(y*z-w*x),1.-2.*(x*x+y*y))
    *mat3(uintBitsToFloat(a.w),0,0,0,uintBitsToFloat(b.x),0,0,0,uintBitsToFloat(b.y));
  mat3 sigma=M*transpose(M),W=transpose(mat3(view));
  // EWA: rows of d(pixel)/d(world), the projection's Jacobian (NDC clamped to 1.3) times the view rotation.
  vec2 ndc=clamp(clip.xy/clip.w,-1.3,1.3);vec3 rw=vec3(projection[0][3],projection[1][3],projection[2][3]);
  vec3 j0=W*(.5*viewport.x*(vec3(projection[0][0],projection[1][0],projection[2][0])-ndc.x*rw)/clip.w);
  vec3 j1=W*(.5*viewport.y*(vec3(projection[0][1],projection[1][1],projection[2][1])-ndc.y*rw)/clip.w);
  float xx=dot(j0,sigma*j0)+.3,xy=dot(j0,sigma*j1),yy=dot(j1,sigma*j1)+.3;  // + 0.3 px² low-pass, as 3DGS
  float mid=.5*(xx+yy),r=length(vec2(.5*(xx-yy),xy)),l1=mid+r,l2=mid-r;
  if(l2<=0.)return;
  vec2 e=vec2(xy,l1-xx);e=dot(e,e)>0.?normalize(e):vec2(1,0);
  // The quad is the 3-sigma ellipse, or smaller where a faint splat already falls under 1/255: same pixels, fewer fragments.
  float k=min(3.,sqrt(2.*log(255.*color.a)));
  vec2 major=min(k*sqrt(l1),2048.)*e,minor=min(k*sqrt(l2),2048.)*vec2(-e.y,e.x);
  offset=k*corner;  // in standard deviations
  gl_Position=vec4(clip.xy/clip.w+(corner.x*major+corner.y*minor)*2./viewport,0.,1.);
}`;
const FRAGMENT = `#version 300 es
precision highp float;in vec4 color;in vec2 offset;out vec4 fragColor;
void main(){  // no discard: it costs ~40% on tile-based GPUs, and a zero-alpha fragment leaves the pixel as it was
  float d2=dot(offset,offset),alpha=d2>9.?0.:min(.99,color.a*exp(-.5*d2));fragColor=alpha<1./255.?vec4(0):vec4(color.rgb*alpha,alpha);}`;
// The splat image is kept in a texture and laid over the frame as is until the camera, the canvas or the order changes:
// the report repaints on every video tick, and a resting camera then costs one full-screen copy.
const COMPOSITE = [`#version 300 es
void main(){gl_Position=vec4(vec2(gl_VertexID&1,gl_VertexID>>1)*4.-1.,0.,1.);}`, `#version 300 es
precision highp float;uniform sampler2D image;out vec4 fragColor;void main(){fragColor=texelFetch(image,ivec2(gl_FragCoord.xy),0);}`];
function link(gl: WebGL2RenderingContext, vertex: string, fragment: string) {
  const program = gl.createProgram()!, shaders = ([[gl.VERTEX_SHADER, vertex], [gl.FRAGMENT_SHADER, fragment]] as [number, string][]).map(([type, code]) => {
    const s = gl.createShader(type)!; gl.shaderSource(s, code); gl.compileShader(s); gl.attachShader(program, s); return s;
  });
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = shaders.map(s => gl.getShaderInfoLog(s)).join(' ') + gl.getProgramInfoLog(program);
    shaders.forEach(s => gl.deleteShader(s)); gl.deleteProgram(program); throw Error('splat_shader_failed: ' + log);
  }
  return {program, shaders};
}

// Other code's GL error flags are not ours: read them out before our own allocations, so a check after them sees only ours.
const clearErrors = (gl: WebGL2RenderingContext) => { for (let i = 0; i < 16 && gl.getError() !== gl.NO_ERROR; i++); };

/** Splats in `gl`, drawn by draw() with the viewer's matrices; `changed` asks the viewer for a repaint. */
export function createSplatLayer(gl: WebGL2RenderingContext, changed: () => void) {
  const programs: {program: WebGLProgram; shaders: WebGLShader[]}[] = [], vao = gl.createVertexArray(), empty = gl.createVertexArray();
  const quad = gl.createBuffer(), orders = gl.createBuffer(), texture = gl.createTexture(), image = gl.createTexture(), framebuffer = gl.createFramebuffer();
  let worker: Worker, workerURL = '';
  function release() {  // every GL object and the worker; also when construction fails halfway
    gl.deleteTexture(texture); gl.deleteTexture(image); gl.deleteFramebuffer(framebuffer); gl.deleteBuffer(quad); gl.deleteBuffer(orders); gl.deleteVertexArray(vao); gl.deleteVertexArray(empty);
    for (const p of programs) { p.shaders.forEach(s => gl.deleteShader(s)); gl.deleteProgram(p.program); }
    worker?.terminate(); if (workerURL) URL.revokeObjectURL(workerURL);
  }
  try {
    programs.push(link(gl, VERTEX, FRAGMENT), link(gl, COMPOSITE[0], COMPOSITE[1]));
    workerURL = URL.createObjectURL(new Blob([`(${sortWorker})(${sortSplats})`], {type: 'text/javascript'})); worker = new Worker(workerURL);
  } catch (error) { release(); throw error; }
  const [{program}, composite] = programs;
  const u = (name: string) => gl.getUniformLocation(program, name), uView = u('view'), uProjection = u('projection'), uViewport = u('viewport');
  gl.useProgram(program); gl.uniform1i(u('splats'), 1); gl.useProgram(composite.program); gl.uniform1i(gl.getUniformLocation(composite.program, 'image'), 1);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, image);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); gl.activeTexture(gl.TEXTURE0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, image, 0); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, quad); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const corner = gl.getAttribLocation(program, 'corner'), splat = gl.getAttribLocation(program, 'splat');
  gl.enableVertexAttribArray(corner); gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, orders); gl.enableVertexAttribArray(splat); gl.vertexAttribIPointer(splat, 1, gl.UNSIGNED_INT, 0, 0); gl.vertexAttribDivisor(splat, 1);
  gl.bindVertexArray(null);
  const abort = new AbortController();
  // `order` is here unless the worker has it: one sort in flight, never a second buffer.
  let capacity = 0, loaded = 0, drawn = 0, sortedFor = -1, s0 = NaN, s1 = NaN, s2 = NaN, s3 = NaN, lastSort = -Infinity, timer = 0, sortMs = NaN, failed = false, disposed = false, order: Uint32Array | null = null;
  // What the cached image shows; `version` counts order uploads.
  const cacheView = new Float32Array(16), cacheProjection = new Float32Array(16);
  let cacheWidth = 0, cacheHeight = 0, cacheVersion = -1, version = 0, movedAt = -Infinity, idleTimer = 0;
  worker.onmessage = ({data}) => {
    if (disposed) return;
    order = data.order as Uint32Array; sortMs = data.ms;
    // A fresh store each time (orphaning), so the upload never waits for frames still drawing from the previous order.
    gl.bindBuffer(gl.ARRAY_BUFFER, orders); gl.bufferData(gl.ARRAY_BUFFER, order, gl.DYNAMIC_DRAW, 0, data.count); drawn = data.count; version++; changed();
  };
  worker.onerror = () => { failed = true; changed(); };
  // Records [start, start+n) into their texels: whole rows at once, partial rows as spans.
  function upload(words: Uint32Array, start: number, n: number) {
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texture);
    for (let i = start, src = 0; i < start + n;) {
      const col = i % ROW, rows = col ? 0 : Math.floor((start + n - i) / ROW), len = rows ? rows * ROW : Math.min(ROW - col, start + n - i);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 2 * col, Math.floor(i / ROW), rows ? 2 * ROW : 2 * len, rows || 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, words, src);
      i += len; src += 8 * len;
    }
    gl.activeTexture(gl.TEXTURE0);
  }
  async function load(url: string, count: number) {
    try {
      capacity = Math.min(count, ROW * gl.getParameter(gl.MAX_TEXTURE_SIZE)); order = new Uint32Array(capacity); worker.postMessage({capacity});
      clearErrors(gl); gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32UI, 2 * ROW, Math.ceil(capacity / ROW)); gl.activeTexture(gl.TEXTURE0);
      if (gl.getError() !== gl.NO_ERROR) throw Error('splat_gpu_allocation_failed');
      const response = await fetch(url, {signal: abort.signal});
      if (!response.ok || !response.body) throw Error('splat_download_failed');
      const reader = response.body.getReader(), total = capacity * SPLAT_BYTES;
      let batch = new Uint8Array(BATCH * SPLAT_BYTES), filled = 0, received = 0;
      const flush = () => {  // whole records only: a torn last record is dropped
        const n = Math.floor(filled / SPLAT_BYTES); if (!n || disposed) return;
        upload(new Uint32Array(batch.buffer, 0, 8 * n), loaded, n);
        worker.postMessage({start: loaded, count: n, records: batch.buffer}, [batch.buffer]);
        loaded += n; batch = new Uint8Array(BATCH * SPLAT_BYTES); filled = 0; changed();
      };
      try {
        while (received < total) {
          const {done, value} = await reader.read();
          if (done || disposed) break;
          for (let at = 0; at < value.length && received < total;) {
            const take = Math.min(value.length - at, batch.length - filled, total - received);
            batch.set(value.subarray(at, at + take), filled); filled += take; at += take; received += take;
            if (filled === batch.length) flush();
          }
        }
        if (received >= total) void reader.cancel().catch(() => {});  // records past the declared count are not ours
      } finally { flush(); }  // a broken download keeps the coarser scene it already has
      if (!disposed && !loaded) throw Error('splat_file_empty');
    } catch (error) { if (!disposed && !loaded) { failed = true; changed(); } throw error; }
  }
  const same = (a: Float32List, b: Float32Array) => { for (let i = 0; i < 16; i++) if (Math.fround(a[i]) !== b[i]) return false; return true; };
  const idle = () => { idleTimer = 0; const wait = movedAt + SETTLE_MS - performance.now(); if (wait > 0) idleTimer = window.setTimeout(idle, wait); else changed(); };
  /** Splats under whatever the viewer draws next, in the bound framebuffer and viewport (width x height device pixels). */
  function draw(view: Float32List, projection: Float32List, width: number, height: number) {
    if (failed || disposed) return;
    const now = performance.now(), still = width === cacheWidth && height === cacheHeight && same(view, cacheView) && same(projection, cacheProjection);
    if (!still) { movedAt = now; if (!idleTimer) idleTimer = window.setTimeout(idle, SETTLE_MS); }
    const limit = now - movedAt < SETTLE_MS ? Math.min(loaded, Math.max(MIN_MOTION_SPLATS, Math.round(width * height * MOTION_SPLATS_PER_PIXEL))) : loaded;
    // Resort when the camera turns or moves, or the drawn set changes; one sort at a time, at most every MIN_SORT_MS.
    const z0 = view[2], z1 = view[6], z2 = view[10], z3 = view[14];
    if (order && limit && (limit !== sortedFor || Math.abs(z0 - s0) + Math.abs(z1 - s1) + Math.abs(z2 - s2) > 1e-4 || z3 !== s3)) {
      const wait = lastSort + MIN_SORT_MS - now;
      if (wait > 0) { if (!timer) timer = window.setTimeout(() => { timer = 0; changed(); }, wait); }
      else { lastSort = now; sortedFor = limit; s0 = z0; s1 = z1; s2 = z2; s3 = z3; worker.postMessage({z0, z1, z2, z3, limit, order}, [order.buffer]); order = null; }
    }
    if (!drawn) return;
    gl.disable(gl.DEPTH_TEST); gl.depthMask(false); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);  // premultiplied, back to front
    gl.activeTexture(gl.TEXTURE1);
    if (!still || cacheVersion !== version) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      if (width !== cacheWidth || height !== cacheHeight) { gl.bindTexture(gl.TEXTURE_2D, image); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null); }
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program); gl.bindVertexArray(vao); gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.uniformMatrix4fv(uView, false, view); gl.uniformMatrix4fv(uProjection, false, projection); gl.uniform2f(uViewport, width, height);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, drawn);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      cacheView.set(view); cacheProjection.set(projection); cacheWidth = width; cacheHeight = height; cacheVersion = version;
    }
    // Over the viewer's cleared frame; no depth, so everything drawn after lies on top.
    gl.useProgram(composite.program); gl.bindVertexArray(empty); gl.bindTexture(gl.TEXTURE_2D, image); gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null); gl.activeTexture(gl.TEXTURE0); gl.enable(gl.DEPTH_TEST); gl.depthMask(true); gl.disable(gl.BLEND);
  }
  function dispose() {
    if (disposed) return; disposed = true; abort.abort(); clearTimeout(timer); clearTimeout(idleTimer); release();
  }
  return {load, draw, dispose, get failed() { return failed; }, get drawn() { return drawn; }, stats: () => ({loaded, total: capacity, drawn, sortMs})};
}
