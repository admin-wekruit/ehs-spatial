const finite = (n) => typeof n === 'number' && Number.isFinite(n);
const text = (s) => typeof s === 'string' && s.trim().length > 0;
const assert = (condition, message) => { if (!condition) throw new Error(message); };

export function assetUrl(value, base) {
  assert(text(value), '资源地址为空');
  const url = new URL(value, base);
  assert(['http:', 'https:'].includes(url.protocol), '资源地址必须使用 HTTP 或 HTTPS');
  return url.href;
}

export function validateManifest(data, base) {
  assert(data?.version === 1 && Array.isArray(data.samples), '样本目录格式不正确（需要 version: 1 与 samples）');
  const ids = new Set();
  for (const sample of data.samples) {
    assert(text(sample.id) && !ids.has(sample.id) && text(sample.title), '样本 ID 必须唯一，且需要名称');
    ids.add(sample.id);
    assert(Number.isInteger(sample.video?.width) && sample.video.width > 0 && Number.isInteger(sample.video?.height) && sample.video.height > 0, `${sample.title} 缺少原视频尺寸`);
    assert(finite(sample.video.durationSec) && sample.video.durationSec > 0, `${sample.title} 缺少原视频时长`);
    assetUrl(sample.video.url, base);
    assert(text(sample.source?.label) && text(sample.source?.url), `${sample.title} 缺少来源`);
    assetUrl(sample.source.url, base);
    assert(Array.isArray(sample.canVerify) && sample.canVerify.every(text), `${sample.title} 缺少验证范围`);
    if (sample.analysis) assetUrl(sample.analysis.url, base);
    if (sample.geometryPreviewUrl) assetUrl(sample.geometryPreviewUrl, base);
  }
  return data;
}

export function validateAnalysis(data, sample, base) {
  assert(data?.version === 1 && data.coordinateSpace === 'source_pixels', '分析结果必须声明原视频像素坐标（source_pixels）');
  assert(data.width === sample.video.width && data.height === sample.video.height, '分析画幅与原视频不一致，已停止叠加');
  assert(Array.isArray(data.frames), '分析结果缺少 frames');
  let previousEnd = 0;
  for (const [index, frame] of data.frames.entries()) {
    assert(finite(frame.timeSec) && finite(frame.endTimeSec) && frame.timeSec >= 0 && frame.endTimeSec > frame.timeSec, `第 ${index + 1} 条观测的时间区间无效`);
    assert(frame.timeSec >= previousEnd, '分析时间区间重叠或未排序');
    assert(frame.endTimeSec <= sample.video.durationSec + 0.1, '分析时间超出原视频时长');
    previousEnd = frame.endTimeSec;
    assert(Array.isArray(frame.objects), `第 ${index + 1} 条观测缺少对象列表`);
    const ids = new Set();
    for (const object of frame.objects) {
      assert(text(object.entityId) && !ids.has(object.entityId) && text(object.label), '同一时刻的对象需要唯一 ID 与标签');
      ids.add(object.entityId);
      if (object.bbox) assert(object.bbox.length === 4 && object.bbox.every(finite) && object.bbox[2] > object.bbox[0] && object.bbox[3] > object.bbox[1], '对象框需要 [x1,y1,x2,y2]');
      if (object.polygons) assert(Array.isArray(object.polygons) && object.polygons.every((polygon) => Array.isArray(polygon) && polygon.length >= 3 && polygon.every((p) => Array.isArray(p) && p.length === 2 && p.every(finite))), '对象轮廓需要原视频像素点');
      if (object.maskUrl) assetUrl(object.maskUrl, base);
      if (object.keypoints) assert(Array.isArray(object.keypoints) && object.keypoints.every((p) => p === null || (Array.isArray(p) && p.length === 3 && p.every(finite) && p[2] >= 0 && p[2] <= 1)), '关节点需要 [x,y,confidence]，缺失关节使用 null');
      if (object.bones) assert(Array.isArray(object.keypoints) && Array.isArray(object.bones) && object.bones.every((edge) => Array.isArray(edge) && edge.length === 2 && edge.every((i) => Number.isInteger(i) && i >= 0 && i < object.keypoints.length)), '骨连接的关节索引无效');
    }
  }
  return data;
}

// ponytail: frames are sorted, non-overlapping source-time intervals; no interpolation across gaps.
export function frameAt(frames, timeSec) {
  if (!finite(timeSec)) return null;
  let low = 0, high = frames.length - 1, candidate = -1;
  while (low <= high) {
    const mid = (low + high) >>> 1;
    if (frames[mid].timeSec <= timeSec) { candidate = mid; low = mid + 1; }
    else high = mid - 1;
  }
  const frame = frames[candidate];
  return frame && timeSec < frame.endTimeSec ? frame : null;
}

export function visibleJoint(point, width, height) {
  return Array.isArray(point) && point[2] >= 0.3 && point[0] >= 0 && point[0] <= width && point[1] >= 0 && point[1] <= height;
}

export function formatTime(timeSec) {
  const total = Math.max(0, Math.round((finite(timeSec) ? timeSec : 0) * 1000));
  return `${String(Math.floor(total / 60000)).padStart(2, '0')}:${String(Math.floor(total / 1000) % 60).padStart(2, '0')}.${String(total % 1000).padStart(3, '0')}`;
}

export function trackObservations(frames, entityId) {
  return frames.flatMap((frame) => frame.objects.filter((object) => object.entityId === entityId).map((object) => ({frame, object})));
}
