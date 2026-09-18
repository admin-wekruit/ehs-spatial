import {assetUrl, validateManifest, validateAnalysis, frameAt, presentedTime, playbackButton, motionText, visibleJoint, trackObservations, formatTime} from './timeline.mjs';

const $ = (id) => document.getElementById(id);
const video = $('video');
const svg = $('overlay');
let fullscreenUnavailable=false;
const state = {manifest:null, manifestUrl:'', sample:null, analysis:null, analysisUrl:'', frame:undefined, selected:null, generation:0, controller:null, observationLimit:80, dimensionalError:false, analysisError:null, replay:null,scene:null,sceneUrl:'',mediaTime:0};
const palette = ['#65e2be','#fac268','#8dcaff','#e9a8ec','#f39c89','#a8d779'];
function color(id) { let n = 0; for (const c of id) n = (n * 31 + c.charCodeAt(0)) | 0; return palette[Math.abs(n) % palette.length]; }
function displayId(object) { return object.displayName || (object.nativeTrackId === undefined ? object.entityId : `轨迹 ${object.nativeTrackId}`); }
function node(tag, className, content) { const n = document.createElement(tag); if (className) n.className = className; if (content !== undefined) n.textContent = content; return n; }
function shape(tag, attrs, content) { const n = document.createElementNS('http://www.w3.org/2000/svg',tag); for (const [key,value] of Object.entries(attrs)) n.setAttribute(key,String(value)); if (content !== undefined) n.textContent = content; return n; }
function showError(id, message) { $(id).textContent = message || ''; $(id).hidden = !message; }
async function readJson(url, signal) { const response = await fetch(url,{signal}); if (!response.ok) throw new Error(`资源加载失败（HTTP ${response.status}）`); return response.json(); }
function syncReplay(time=video.currentTime) {state.replay?.setTime(state.dimensionalError ? NaN : time,state.selected);}
function reviewStatus(value){return {clear:'类别清楚',partial:'局部可见',incorrect_prompt:'类别不符'}[value]||'未核验';}
function staticObservation(id) {
  const model=state.scene?.staticObjects?.find(o=>o.entityId===id);
  return model ? {model,frame:model.source,object:{entityId:model.entityId,label:model.label+' · 可见表面',displayName:model.displayName,bbox:model.source.bbox,maskUrl:assetUrl(model.source.maskUrl,state.sceneUrl)}} : null;
}

function drawMotion() {
  const detail=$('selected-motion');if(!detail)return;
  const frame=state.dimensionalError ? null : frameAt(state.scene?.frames || [],state.mediaTime);
  detail.textContent=motionText(frame?.objects.find(object=>object.entityId===state.selected));
}

function selectEntity(id) {
  const source=staticObservation(id);
  if(source&&state.selected!==id)seekFrame(source.frame);
  state.selected = state.selected === id ? null : id;
  state.observationLimit = 80;
  drawFrame(state.frame);
  drawObservations();
  syncReplay();
}

function drawSamples() {
  $('samples').replaceChildren();
  $('sample-count').textContent = state.manifest.samples.length;
  for (const sample of state.manifest.samples) {
    const button = node('button','sample-option');
    button.setAttribute('aria-current',String(state.sample?.id === sample.id));
    button.append(node('strong','',sample.title),node('span','',`${formatTime(sample.video.durationSec).slice(0,5)} · ${sample.video.width} × ${sample.video.height}`),node('span','',sample.analysis ? '含分析产物' : sample.scene ? '含空间地图' : '尚未运行分析'));
    button.onclick = () => loadSample(sample);
    $('samples').append(button);
  }
  if (!state.manifest.samples.length) $('samples').append(node('p','empty-copy','尚未接入公开视频样本。'));
}

async function loadSample(sample) {
  const generation = ++state.generation;
  state.controller?.abort();
  state.controller = new AbortController();
  state.replay?.dispose(); state.replay = null;
  $('scene-stage').replaceChildren();
  $('scene-panel').hidden = !sample.scene;
  $('scene-details').replaceChildren();
  $('scene-objects').replaceChildren();
  $('scene-body').hidden=true;$('scene-body').setAttribute('aria-pressed','false');$('scene-body').textContent='人体网格估计';$('scene-fit').disabled = true;$('scene-source').disabled = true;
  $('scene-all').disabled = true;
  $('scene-points').disabled = true;
  $('scene-points').setAttribute('aria-pressed','false');
  showError('scene-error',null);
  video.pause();
  Object.assign(state,{sample,analysis:null,frame:undefined,selected:null,dimensionalError:false,analysisError:null,observationLimit:80,scene:null,sceneUrl:'',mediaTime:0});
  drawSamples();
  svg.replaceChildren();
  svg.setAttribute('viewBox',`0 0 ${sample.video.width} ${sample.video.height}`);
  $('sample-title').textContent = sample.title;
  $('description').textContent = sample.description || '';
  $('duration').textContent = `${formatTime(sample.video.durationSec).slice(0,5)} · ${sample.video.width} × ${sample.video.height}`;
  $('source').replaceChildren();
  const source = node('a','',sample.source.label + ' ↗');
  source.href = assetUrl(sample.source.url,state.manifestUrl); source.target = '_blank'; source.rel = 'noopener noreferrer';
  $('source').append(source);
  if (sample.source.kind) $('source').append(document.createTextNode(` · ${sample.source.kind}`));
  if (sample.source.license) $('source').append(document.createTextNode(` · ${sample.source.license}`));
  $('verification').replaceChildren(...sample.canVerify.map((item) => node('li','',item)));
  $('media-empty').hidden = true;
  $('media-stage').hidden = false;
  $('scrubber').value = $('scene-scrubber').value = 0;
  $('scrubber').max = sample.video.durationSec;
  $('scrubber').disabled = true;
  $('clock').textContent = formatTime(0);
  $('frame-time').textContent = '';
  $('analysis-details').replaceChildren();
  showError('video-error',null);
  $('geometry').hidden = !sample.geometryPreviewUrl;
  if (sample.geometryPreviewUrl) $('geometry-link').href = assetUrl(sample.geometryPreviewUrl,state.manifestUrl);
  else $('geometry-link').removeAttribute('href');
  video.width = sample.video.width; video.height = sample.video.height;
  video.src = assetUrl(sample.video.url,state.manifestUrl);
  video.load();
  const pageUrl = new URL(location.href); pageUrl.searchParams.set('sample',sample.id); history.replaceState(null,'',pageUrl);
  drawFrame(null);
  drawObservations();
  if (sample.scene) void loadScene(sample,generation);
  if (!sample.analysis) { updateStatus(sample.scene ? '此视频提供空间地图；对象轨迹、骨架与动态模型尚无输出。' : '尚未运行分析。可以播放原视频；对象轨迹、骨架与动态模型尚无输出。'); return; }
  updateStatus('正在加载此样本的分析结果，原视频可以先播放。');
  try {
    const analysisUrl = assetUrl(sample.analysis.url,state.manifestUrl);
    const data = await readJson(analysisUrl,state.controller.signal);
    if (generation !== state.generation) return;
    state.analysis = validateAnalysis(data,sample,analysisUrl);
    state.analysisUrl = analysisUrl;
    $('analysis-details').replaceChildren();
    if (data.method) $('analysis-details').append(node('p','',`分析来源：${data.method}`));
    if (sample.analysisStatus) $('analysis-details').append(node('p','',sample.analysisStatus));
    if (Array.isArray(data.limitations)) {
      const list = node('ul');
      for (const limit of data.limitations) list.append(node('li','',String(limit)));
      $('analysis-details').append(list);
    }
    state.frame = undefined;
    if (state.dimensionalError) updateStatus('视频实际画幅与分析输入不一致，已停止叠加。原视频可继续播放。',true);
    else updateTime(video.currentTime,true);
  } catch (error) {
    if (generation !== state.generation || error.name === 'AbortError') return;
    state.analysisError = error.message;
    updateStatus(`分析结果未显示：${error.message}。原视频仍可播放。`,true);
    drawFrame(null);
  }
}

async function loadScene(sample,generation) {
  $('scene-status').textContent = '正在读取此视频的空间结果…';
  try {
    const sceneUrl = assetUrl(sample.scene.url,state.manifestUrl);
    const [data,module] = await Promise.all([readJson(sceneUrl,state.controller.signal),import('./scene.js')]);
    if (generation !== state.generation) return;
    const scene = module.validateScene(data,sample,sceneUrl);
    state.scene=scene;state.sceneUrl=sceneUrl;
    $('scene-units').textContent = scene.units === 'meters' ? '米 · ' + scene.coordinate_frame : '单目单位未标定 · ' + scene.coordinate_frame;
    $('scene-details').append(node('p','',scene.method));
    for (const limitation of scene.limitations) $('scene-details').append(node('p','',String(limitation)));
    const replay = await module.mountReplay($('scene-stage'),scene,sceneUrl,{
      color,select:selectEntity,signal:state.controller.signal,
      error:message=>{if(generation===state.generation)showError('scene-error',message);},
      frame:(frame,geometry)=>{if(generation===state.generation)$('scene-status').textContent=frame ? `源帧 ${frame.sourceFrame} · ${geometry?.mode==='body'?'人体网格估计':'彩色观测表面'} ${geometry?.rendered||0} / ${frame.objects.length}${geometry?.mode==='body'?' · 仅显示通过对齐检查的时段；中间帧为模型插值':''}` : '此时刻无空间观测；仅保留已加载的静态场景。';},
    });
    if (generation !== state.generation) {replay.dispose(); return;}
    state.replay = replay;
    if(scene.staticObjects?.length){
      const isolate=node('button','','仅看对象表面');isolate.setAttribute('aria-pressed','false');isolate.onclick=()=>{const enabled=isolate.getAttribute('aria-pressed')!=='true';isolate.setAttribute('aria-pressed',String(enabled));isolate.textContent=enabled?'返回完整场景':'仅看对象表面';if(enabled)isolate.dataset.pointCloud=$('scene-points').getAttribute('aria-pressed');$('scene-points').setAttribute('aria-pressed',enabled?'false':isolate.dataset.pointCloud);$('scene-all').disabled=enabled||!(scene.pointCloudCount||scene.points.length);$('scene-points').disabled=enabled||!(scene.pointCloudCount||scene.points.length);replay.setObjectView(enabled);};
      $('scene-objects').append(node('h3','',`对象观测表面 · ${scene.staticObjects.length}`),isolate,node('p','muted','点击模型或名称回到来源帧与掩码。保留原始预测类别；照片核验是模型解释，不是真值。未验证其他时刻的位置。'));
      for(const object of scene.staticObjects){const button=node('button','',`${object.displayName} · 帧 ${object.source.sourceFrame}${object.semanticReview?' · '+reviewStatus(object.semanticReview.status):''}`);button.title=object.semanticReview?.description||'';button.dataset.entityId=object.entityId;button.setAttribute('aria-pressed','false');button.onclick=()=>selectEntity(object.entityId);$('scene-objects').append(button);}
    }
    $('scene-body').hidden=!scene.bodyKeyframes?.length;$('scene-fit').disabled = false;$('scene-source').disabled = false;
    $('scene-all').disabled = !(scene.pointCloudCount||scene.points.length);
    $('scene-points').disabled = !(scene.pointCloudCount||scene.points.length);
    $('scene-points').setAttribute('aria-pressed',String(!!scene.pointCloudUrl||!scene.meshUrl&&!!scene.points.length));
    syncReplay();
    drawFrame(state.frame);
  } catch (error) {
    if (generation !== state.generation || error.name === 'AbortError') return;
    $('scene-status').textContent = `空间结果未显示：${error.message}`;
  }
}

function updateStatus(message, error = false) { $('analysis-status').textContent = message; $('analysis-status').classList.toggle('error',error); }

function drawFrame(frame) {
  drawMotion();
  svg.replaceChildren();
  $('objects').replaceChildren();
  const source=staticObservation(state.selected),inSource=source&&state.mediaTime>=source.frame.timeSec&&state.mediaTime<source.frame.endTimeSec;
  const objects = state.dimensionalError ? [] : [...(frame?.objects || []),...(inSource?[source.object]:[])];
  $('object-count').textContent = (frame||inSource) && !state.dimensionalError ? objects.length : '—';
  for(const button of $('scene-objects').querySelectorAll('[data-entity-id]'))button.setAttribute('aria-pressed',String(button.dataset.entityId===state.selected));
  if (!objects.length) {
    const label = state.dimensionalError ? '画幅不一致，已停止叠加。' : !state.sample ? '选择视频后查看实际分析产物。' : !state.analysis ? '此视频尚无可用分析产物。' : frame ? '此区间已分析，未检测到对象。' : '此时刻没有观测，不沿用前一帧的位置。';
    $('objects').append(node('p','empty-copy',label));
  }
  for (const object of objects) {
    const selected = object.entityId === state.selected;
    const tint = color(object.entityId);
    const button = node('button','object-option');
    button.setAttribute('aria-pressed',String(selected));
    button.title = object.entityId;
    const dot = node('span','dot'); dot.style.background = tint;
    const copy = node('span');
    const kinds = [object.maskUrl || object.polygons?.length ? 'mask' : object.bbox ? '检测框' : null,object.keypoints?.some((p) => visibleJoint(p,state.sample.video.width,state.sample.video.height)) ? '二维骨架' : null].filter(Boolean);
    copy.append(node('strong','',object.label),node('small','',`${displayId(object)}${kinds.length ? ' · ' + kinds.join(' / ') : ''}`));
    button.append(dot,copy); button.onclick = () => selectEntity(object.entityId); $('objects').append(button);

    const group = shape('g',{'data-entity-id':object.entityId});
    const width = state.sample.video.width, height = state.sample.video.height;
    const lineWidth = Math.max(1.5,width/400);
    const clickable = (element) => { element.classList.add('selectable'); if (selected) element.classList.add('selected'); element.addEventListener('click',(event) => {event.stopPropagation(); selectEntity(object.entityId);}); return element; };
    if ($('show-masks').checked) {
      if (object.maskUrl) group.append(shape('image',{href:assetUrl(object.maskUrl,state.analysisUrl),x:0,y:0,width,height,opacity:selected ? .7 : .38,'pointer-events':'none'}));
      for (const polygon of object.polygons || []) group.append(clickable(shape('polygon',{points:polygon.map((p)=>p.join(',')).join(' '),fill:tint,'fill-opacity':selected ? .32 : .15,stroke:tint,'stroke-width':selected ? lineWidth*2 : lineWidth})));
      if (object.bbox) { const [x1,y1,x2,y2] = object.bbox; group.append(clickable(shape('rect',{x:x1,y:y1,width:x2-x1,height:y2-y1,fill:tint,'fill-opacity':.025,stroke:tint,'stroke-width':lineWidth,'stroke-dasharray':object.polygons?.length ? '3 3' : 'none'}))); }
    }
    if ($('show-skeleton').checked && object.keypoints) {
      for (const [a,b] of object.bones || []) {
        const pa = object.keypoints[a], pb = object.keypoints[b];
        if (visibleJoint(pa,width,height) && visibleJoint(pb,width,height)) group.append(shape('line',{x1:pa[0],y1:pa[1],x2:pb[0],y2:pb[1],stroke:tint,'stroke-width':lineWidth*1.7,'stroke-linecap':'round'}));
      }
      for (const p of object.keypoints) if (visibleJoint(p,width,height)) group.append(clickable(shape('circle',{cx:p[0],cy:p[1],r:lineWidth*1.8,fill:tint,stroke:'#192b22','stroke-width':lineWidth/2})));
    }
    if ($('show-masks').checked || $('show-skeleton').checked) {
      const anchor = object.bbox?.slice(0,2) || object.polygons?.[0]?.[0] || object.keypoints?.find((p)=>visibleJoint(p,width,height));
      if (anchor) {
        const fontSize = Math.max(11,width/62);
        const annotation = shape('text',{x:Math.min(width-10,Math.max(4,anchor[0])),y:Math.max(fontSize+4,anchor[1]-5),fill:tint,'font-size':fontSize},object.displayName || (object.nativeTrackId === undefined ? `${object.label} · ${object.entityId}` : displayId(object)));
        annotation.append(shape('title',{},object.entityId));
        group.append(clickable(annotation));
      }
    }
    svg.append(group);
  }
  for (const row of $('observations').querySelectorAll('[data-time]')) row.setAttribute('aria-current',String(Number(row.dataset.time) === frame?.timeSec));
}

function drawObservations() {
  $('observations').replaceChildren();
  $('clear-selection').hidden = !state.selected;
  const source=staticObservation(state.selected);
  if (!state.selected || (!state.analysis&&!source)) { $('selected-label').textContent = '点击画面或对象，查看它出现的时刻。'; return; }
  const observations = source ? [source] : trackObservations(state.analysis.frames,state.selected);
  const label = observations[0]?.object.label || state.selected;
  $('selected-label').textContent = `${label} · ${state.selected} · ${observations.length} 条来源观测`;
  if(!source){const detail=node('span','source-description');detail.id='selected-motion';$('selected-label').append(detail);drawMotion();}
  if(source){
    $('selected-label').append(node('span','source-description',`空间记忆：仅源帧 ${source.frame.sourceFrame}（${formatTime(source.frame.timeSec)}）的可见表面；其他时刻的位置与状态未知。`));
    const review=source.model.semanticReview;
    if(review)$('selected-label').append(node('span','source-description',`照片模型核验：${reviewStatus(review.status)} · ${review.description}（模型解释，非真值）`));
    for(const [text,url]of [['查看原始帧',source.model.source.imageUrl],['查看模型来源',source.model.provenanceUrl]]){const link=node('a','source-link',text);link.href=assetUrl(url,state.sceneUrl);link.target='_blank';link.rel='noopener noreferrer';$('selected-label').append(link);}
  }
  for(const candidate of state.analysis?.identityCandidates || []) {
    if(![candidate.fromEntityId,candidate.toEntityId].includes(state.selected))continue;
    const group=node('div','reentry-candidate');
    group.append(node('p','',`可能再次出现：轨迹 ${candidate.fromTrackId} ↔ 轨迹 ${candidate.toTrackId}（未确认）`));
    for(const [label,key] of [['先前来源','referenceFrames'],['再次出现来源','evidenceFrames']]) {
      const row=node('div','');row.append(node('span','',label+'：'));
      for(const index of candidate[key]){const frame=state.analysis.frames.find(f=>f.sourceFrame===index),button=node('button','',`帧 ${index}`);button.onclick=()=>seekFrame(frame);row.append(button);}
      group.append(row);
    }
    group.append(node('small','','仅为外观特征候选，身份没有合并。'));$('observations').append(group);
  }
  for (const {frame,object} of observations.slice(0,state.observationLimit)) {
    const row = node('button','observation-row'); row.dataset.time = frame.timeSec;
    row.setAttribute('aria-current',String(state.frame?.timeSec === frame.timeSec));
    const time = node('time','',formatTime(frame.timeSec));
    const info = [frame.sourceFrame === undefined ? null : `帧 ${frame.sourceFrame}`, object.confidence == null ? null : `${Math.round(object.confidence*100)}%`].filter(Boolean);
    row.append(time,node('span','',info.join(' · '))); row.onclick = () => seekFrame(frame); $('observations').append(row);
  }
  if (observations.length > state.observationLimit) {
    const more = node('button','more-observations',`展开后续观测（还有 ${observations.length-state.observationLimit} 条）`);
    more.onclick = () => {state.observationLimit += 80; drawObservations();}; $('observations').append(more);
  }
}

function updateTime(time, force = false) {
  state.mediaTime=time;
  $('clock').textContent = formatTime(time);
  $('scrubber').value = $('scene-scrubber').value = time;
  syncReplay(time);
  if (!state.analysis || state.dimensionalError) {drawFrame(null);return;}
  const frame = frameAt(state.analysis.frames,time);
  if (frame !== state.frame || force) {
    state.frame = frame;
    drawFrame(frame);
    $('frame-time').textContent = frame ? `观测 ${formatTime(frame.timeSec)} — ${formatTime(frame.endTimeSec)}` : '此时刻无观测';
    updateStatus(frame ? `当前区间有 ${frame.objects.length} 条时序对象观测。点击对象可查看来源时刻。` : '此时刻没有观测。没有补画位置、遮挡轨迹或骨架。');
  }
}

function seek(time) { if (video.readyState >= 1) { video.currentTime = Math.max(0,Math.min(video.duration,time)); $('clock').textContent = formatTime(time); } }
// Seek inside the observed exposure, avoiding decoder rounding onto the previous frame at its boundary.
function seekFrame(frame) {video.pause();seek((frame.timeSec+frame.endTimeSec)/2);}
function syncPlaybackControls() {
  const button=playbackButton(video);
  $('play-toggle').disabled=button.disabled;$('play-toggle').textContent=button.label;$('scene-play').disabled=button.disabled;$('scene-play').textContent=button.label;
  $('video-fullscreen').disabled=button.disabled||fullscreenUnavailable||!document.fullscreenEnabled||typeof video.requestFullscreen!=='function';
}
$('play-toggle').onclick=async()=>{
  const generation=state.generation;
  if(!video.paused){video.pause();return;}
  try {showError('video-error',null);await video.play();}
  catch(error){if(generation===state.generation&&error.name!=='AbortError')showError('video-error',`播放失败：${error.message}`);}
  syncPlaybackControls();
};
$('video-fullscreen').onclick=async()=>{
  try {await video.requestFullscreen();}
  catch(error){fullscreenUnavailable=true;syncPlaybackControls();showError('video-error','当前浏览器未允许视频全屏；页面内播放仍可使用。');}
};
// Native controls are safe only in video-only fullscreen, where the SVG is absent.
document.addEventListener('fullscreenchange',()=>{video.controls=document.fullscreenElement===video;});
for(const event of ['loadedmetadata','play','pause','ended','seeked','emptied','error'])video.addEventListener(event,syncPlaybackControls);
$('scrubber').addEventListener('input',(event)=>seek(Number(event.target.value)));
$('clear-selection').onclick = () => {state.selected = null; drawFrame(state.frame); drawObservations(); syncReplay();};
$('scene-play').onclick=()=>$('play-toggle').click();
$('scene-scrubber').addEventListener('input',event=>seek(Number(event.target.value)));
$('scene-body').onclick=()=>{const enabled=$('scene-body').getAttribute('aria-pressed')!=='true';$('scene-body').setAttribute('aria-pressed',String(enabled));$('scene-body').textContent=enabled?'返回彩色实测表面':'人体网格估计';state.replay?.setBodyModels(enabled);};
$('scene-source').onclick=()=>state.replay?.sourceView();
$('scene-expand').onclick=()=>{const panel=$('scene-panel');if(document.fullscreenElement)document.exitFullscreen();else panel.requestFullscreen().catch(error=>showError('scene-error',error.message));};
$('scene-fit').onclick = () => state.replay?.fit();
$('scene-points').onclick = () => {const enabled=$('scene-points').getAttribute('aria-pressed')!=='true';$('scene-points').setAttribute('aria-pressed',String(enabled));state.replay?.setPointCloud(enabled);};
$('scene-all').onclick = () => {$('scene-points').setAttribute('aria-pressed','true');state.replay?.setPointCloud(true);state.replay?.fit(true);};
window.addEventListener('pagehide',()=>state.replay?.dispose(),{once:true});
for (const id of ['show-masks','show-skeleton']) $(id).onchange = () => drawFrame(state.frame);
video.addEventListener('loadedmetadata',()=>{
  if (!state.sample) return;
  $('scrubber').max = $('scene-scrubber').max = video.duration;
  $('scrubber').disabled = $('scene-scrubber').disabled = !Number.isFinite(video.duration);
  $('duration').textContent = `${formatTime(video.duration).slice(0,5)} · ${video.videoWidth} × ${video.videoHeight}`;
  state.dimensionalError = video.videoWidth !== state.sample.video.width || video.videoHeight !== state.sample.video.height;
  syncReplay();
  if (state.dimensionalError) {svg.replaceChildren(); updateStatus(`视频实际尺寸为 ${video.videoWidth} × ${video.videoHeight}，与分析输入不一致，已停止叠加。原视频可继续播放。`,true); drawFrame(null);}
});
video.addEventListener('error',()=>showError('video-error','原视频暂时无法加载，请检查该样本的视频地址与本地服务。分析结果不会替代原视频。'));
video.addEventListener('seeked',()=>updateTime(video.currentTime,true));
video.addEventListener('timeupdate',()=>{if (video.paused || !video.requestVideoFrameCallback) updateTime(video.currentTime);});
video.addEventListener('ended',()=>updateTime(video.currentTime,true));
if (video.requestVideoFrameCallback) {
  const presented = (_now,metadata) => {updateTime(presentedTime(state.analysis?.frames || state.scene?.frames || [],metadata.mediaTime)); video.requestVideoFrameCallback(presented);};
  video.requestVideoFrameCallback(presented);
}

try {
  const requested = new URLSearchParams(location.search).get('manifest') || './manifest.json';
  state.manifestUrl = assetUrl(requested,location.href);
  state.manifest = validateManifest(await readJson(state.manifestUrl),state.manifestUrl);
  if (state.manifest.title) {$('page-title').textContent = state.manifest.title; document.title = `${state.manifest.title} · Panoptes`;}
  drawSamples();
  const requestedId = new URLSearchParams(location.search).get('sample');
  const first = state.manifest.samples.find((s)=>s.id === requestedId) || state.manifest.samples[0];
  if (first) await loadSample(first);
} catch (error) {showError('page-error',`样本目录未加载：${error.message}`);}
