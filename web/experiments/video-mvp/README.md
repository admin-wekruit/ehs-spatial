# 视频回放与分析预览

Goal: 原视频始终可播放；只显示同一媒体时间上的实际观测，不填补漏帧、遮挡或缺失分析。

Architecture: 静态 HTML 与原生 video controls；SVG 在原视频像素域绘制 mask/ID/二维骨架。样本目录先加载，分析仅在选择样本后加载。3D 按需打开已有房间预览，不实现第二个渲染器。

Implementation plan:
- [x] 固定 manifest 与分帧观测输入，分离原视频时间和浏览器时间。
- [x] 实现样本选择、原视频回放、叠加选择、可跳转观测列表和按需几何预览。
- [x] 检查时间边界、坐标、缺数据和输入拒绝；静态运行文件复制到实际产物目录。

## 运行

不需要构建或新增依赖，静态服务即可。可通过 `?manifest=实际manifest地址` 指定数据；相对地址基于页面 URL。

```sh
/Users/adam/Desktop/panoptes-public/panoptes-serving/.venv/bin/python /Users/adam/.codex/worktrees/panoptes-phase2-video/web/experiments/video-mvp/serve.py /Users/adam/Desktop/panoptes-public/research-notes/phase2 --port 8799
node /Users/adam/.codex/worktrees/panoptes-phase2-video/web/experiments/video-mvp/check.mjs
```

服务复用已有 Starlette/uvicorn；`StaticFiles` 原生支持 Range。不要用 Python `http.server` 代替：实测 Chrome 即使已完整缓冲视频，也会出现 `seekable=[0,0]`，拖动重回零秒。

已有 8799 服务时不要重复启动。把本目录的 index.html、app.js、timeline.mjs、style.css 和生产 manifest.json 放进服务根目录下的 `video-mvp/` 即可；实际数据由主流程提供。页面本身不代表视频分析 MVP 已完成。

## 数据合同 v1

所有 URL 相对声明它的 JSON 文件解析，接受 http(s) 与同源静态文件。维度必须为原视频画幅，坐标原点在左上。

```json
{
  "version": 1,
  "title": "视频与空间",
  "samples": [{
    "id": "sample-id",
    "title": "样本名称",
    "description": "样本内容",
    "video": {"url": "clip.mp4", "width": 640, "height": 480, "durationSec": 10},
    "source": {"label": "来源名称", "url": "https://example.org/source", "license": "来源许可"},
    "canVerify": ["此样本能够检验的能力"],
    "geometryPreviewUrl": "../runs/room/preview/index.html",
    "analysis": {"url": "analysis.json"},
    "analysisStatus": "实际运行说明"
  }]
}
```

`analysis`、`geometryPreviewUrl` 可省略。没有分析时仍加载视频，并明确显示尚未运行。以下仅为字段说明，不作为演示数据发行：

```json
{
  "version": 1,
  "coordinateSpace": "source_pixels",
  "width": 640,
  "height": 480,
  "method": "实际执行的模型/版本",
  "limitations": ["验证边界"],
  "frames": [{
    "timeSec": 0,
    "endTimeSec": 0.04,
    "sourceFrame": 0,
    "objects": [{
      "entityId": "track-1",
      "label": "人",
      "bbox": [10, 20, 100, 200],
      "maskUrl": "masks/frame-0-track-1.png",
      "polygons": [[[10,20],[100,20],[100,200],[10,200]]],
      "keypoints": [[25,30,0.9],[40,50,0.8]],
      "bones": [[0,1]],
      "confidence": 0.9
    }]
  }]
}
```

- `timeSec/endTimeSec` 是原视频秒的半开区间 `[start,end)`，按时间排序，不得重叠。按实际采样间隔/有效范围填写，不要把稀疏帧延长为已观测运动。
- `objects: []` 表示该区间已分析但未检测到对象；帧记录之外是没有观测，不外推。
- `entityId` 由分析提供，UI 不创造跨帧/跨视角关联。标签应区分轨迹 ID 与已确认物理身份。
- bbox、polygon、keypoints 全部使用原视频像素坐标。maskUrl 是同尺寸透明 PNG；可只提供 bbox。关键点置信度低于 0.3 不画，骨连接不会跨过缺失关节。
- 每种叠加均可省略。缺少模型结果时不创建骨架、mask、3D 盒子或轨迹。
- 二维骨架不等于三维人体/房间定位。现有几何预览独立打开，不宣称已与视频时间同步。

## 检查记录

- `node check.mjs`：半开时间边界、观测空档、空目录/空分析、原画幅匹配、唯一 ID、无效骨连接与低置信关节。
- 实际 `/video-mvp/index.html` 的 walking 视频为 640×480 / 28.927429 秒，Chrome 可播、可跳至 8 秒；room 视频可跳至 12 秒。切换样本后，只有 room 保留其几何预览链接。
- 8799 的 Range 请求返回 206 / 正确 Content-Range；此前 SimpleHTTP 返回 200，Chrome 的 seekable 为 `[0,0]`，是拖动回零的原因。
- 独立 Chrome 中临时拦截的检查数据验证 SVG 中心点映射、鼠标对象选择、来源观测跳转、空档清除、叠加开关和 390px 宽度。合成观测没有写入生产 manifest/分析文件，不是模型推理证据。
- 真实两视频的人员分析尚未接入；浏览器检查只证明播放与结果显示交互，不代表完整视频分析或建模 MVP 已完成。
