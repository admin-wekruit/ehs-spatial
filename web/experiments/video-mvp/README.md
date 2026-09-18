# 视频回放与分析预览

Goal: 原视频始终可播放；只显示同一媒体时间上的实际观测，不填补漏帧、遮挡或缺失分析。

Architecture: 静态 HTML 与原生 video controls；SVG 在原视频像素域绘制 mask/ID/二维骨架。同页空间回放复用现有 `native-viewer.ts`、GLB reader 和坐标变换；静态点云/网格只加载一次，每帧更新相机与关节连线的变换。样本和场景按需加载，切换时释放旧 WebGL 与 Blob 资源。

Implementation plan:
- [x] 固定 manifest 与分帧观测输入，分离原视频时间和浏览器时间。
- [x] 实现样本选择、原视频回放、叠加选择、可跳转观测列表和按需几何预览。
- [x] 检查时间边界、坐标、缺数据和输入拒绝；静态运行文件复制到实际产物目录。
- [x] 同页视频时间驱动真实静态场景、相机和人体表面关节，支持自由拖转与同 ID 选择。

## 运行

不新增依赖；空间模块需用已安装的 Vite 构建一次，其余文件为静态文件。可通过 `?manifest=实际manifest地址` 指定数据；相对地址基于页面 URL。

在仓库根目录构建到真实产物目录（不改已有 manifest）：

```sh
node --input-type=module - <<'JS'
import {build} from '/Users/adam/Desktop/Tesla/panoptes-platform/web/node_modules/vite/dist/node/index.js';
import {copyFile} from 'node:fs/promises';
const out='/Users/adam/Desktop/panoptes-public/research-notes/phase2/video-mvp';
await build({configFile:false,root:process.cwd(),build:{outDir:out,emptyOutDir:false,lib:{entry:'web/experiments/video-mvp/scene.ts',formats:['es'],fileName:()=> 'scene.js'}}});
for(const file of ['index.html','app.js','timeline.mjs','style.css'])await copyFile('web/experiments/video-mvp/'+file,out+'/'+file);
JS
```

```sh
/Users/adam/Desktop/panoptes-public/panoptes-serving/.venv/bin/python /Users/adam/.codex/worktrees/panoptes-phase2-video/web/experiments/video-mvp/serve.py /Users/adam/Desktop/panoptes-public/research-notes/phase2 --port 8799
node /Users/adam/.codex/worktrees/panoptes-phase2-video/web/experiments/video-mvp/check.mjs
node /Users/adam/.codex/worktrees/panoptes-phase2-video/web/experiments/video-mvp/scene-check.ts
```

服务复用已有 Starlette/uvicorn；`StaticFiles` 原生支持 Range。不要用 Python `http.server` 代替：实测 Chrome 即使已完整缓冲视频，也会出现 `seekable=[0,0]`，拖动重回零秒。

已有 8799 服务时不要重复启动。实际 manifest、视频与分析数据由主流程提供；构建不生成演示观测。页面本身不代表算法质量已通过。

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
    "scene": {"url": "scene.json"},
    "analysisStatus": "实际运行说明"
  }]
}
```

`analysis`、`scene`、`geometryPreviewUrl` 可省略。没有对象分析时仍加载视频和已有空间场景。以下仅为字段说明，不作为演示数据发行：

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
- 二维骨架不等于三维人体/房间定位。只有独立的 `scene` 合同提供真实空间坐标时，同页才显示空间回放。

空间合同固定为 `phase2-replay-scene-v1`，包含 `coordinate_frame`、`units`（`meters` 或 `uncalibrated_monocular`）、`source_video_sha256`、`method`、`limitations`、可选 `meshUrl`、原坐标点 `points: [[id,x,y,z],...]`，以及 `frames: [{sourceFrame,timeSec,endTimeSec,c2w,objects}]`。`c2w` 是嵌套 4×4 row-major 矩阵；每个对象提供 `entityId`、`keypoints3d`（XYZ 或 null）、`bones` 和可选 `centroid`。视频 SHA 与场景来源需相同。

空间帧使用同样的半开时间区间。空档隐藏相机与全部人体；只有 centroid 时只显示中心标记，不补关节。静态场景保留；固定动态图元通过 alpha 0 同时退出绘制和拾取，避免重载网格。“观察区”仅调整浏览相机到真实 mesh/关节范围，或无表面时的真实相机轨迹；“全部点云”适配全部原点。两者都不改变、筛除或重新标定数据。

## 检查记录

人物采用已有八边封闭圆柱 primitive（每段32个三角面）构成关节点粗模，光照显示截面。圆柱端点严格位于输入3D关节；半径为便于观察的显示尺寸，并按短骨段长度限制，不是测量体型。躯干只保留有证据的关节连接，不填头壳、体表或隐藏肢体。2.324秒真实源帧69有两人共29个有效段；4.007秒一人只剩上身10段，另一人关节全空，粗模随数据缺失消失。没有额外模型训练、权重或几何推断系统。

- `node check.mjs`：半开时间边界、观测空档、空目录/空分析、原画幅匹配、唯一 ID、无效骨连接与低置信关节。
- 实际 `/video-mvp/index.html` 的 walking 视频为 640×480 / 28.927429 秒，Chrome 可播、可跳至 8 秒；room 视频可跳至 12 秒。切换样本后，只有 room 保留其几何预览链接。
- 8799 的 Range 请求返回 206 / 正确 Content-Range；此前 SimpleHTTP 返回 200，Chrome 的 seekable 为 `[0,0]`，是拖动回零的原因。
- 独立 Chrome 中临时拦截的检查数据验证 SVG 中心点映射、鼠标对象选择、来源观测跳转、空档清除、叠加开关和 390px 宽度。合成观测没有写入生产 manifest/分析文件，不是模型推理证据。
- 真实 walking 前175帧已接入 SAM3 图片种子 → SAM2.1 原生视频记忆 → RTMPose 二维骨架，以及传感器 RGB-D 空间对照。4.007秒对应源帧119，静态网格与轨迹0的3D骨架可见；轨迹1该帧只有中心点，关节缺失保持缺失。
- 独立浏览器实际点击可见3D骨架会选择对应轨迹；拖转后推进到6.05秒，旧骨架消失且旧位置不可拾取，观察相机不被时间更新重置。切换 room 释放 walking 场景，34.970秒对应源帧1049，显示未标定单位；19.969秒无相机观测，只有静态点云。
- `scene-check.ts` 检查来源 SHA、时间和单位拒绝、原 XYZ 的 POINTS GLB、任意/反向骨连接端点、缺关节与空档清除及固定 GPU 资产签名。390px 窄屏空间视图堆叠可用；实际浏览器无渲染错误。
- 以上验证播放器与真实结果接通，不证明全片身份稳定、SLAM 质量、完整人体或普通 RGB 已有传感器深度精度。


## 单帧对象表面与照片核验

`scene.staticObjects` 可提供独立观测：`entityId`（obs-前缀）、原 SAM `label`、`displayName`、`meshUrl`、`provenanceUrl`、`representation: single_frame_observed_surface`、`identityScope: independent_observation`、`source: {sourceFrame,timeSec,endTimeSec,width,height,bbox,maskUrl,imageUrl}`。可选 `semanticReview: {status,description,category}`，状态为 clear / partial / incorrect_prompt。maskUrl 为源画幅透明叠加图。来源半开区间必须与相机记录完全一致。

这是单帧空间记忆；其他时刻位置与状态未知。地图保留模型，但源掩码仅在来源帧区间显示。点击真实网格或名称会暂停并跳到源帧区间中点，避免浏览器将边界 seek 解码为前一帧。“仅看对象表面”通过原生图层筛选去掉可能遮挡它的 TSDF；模型源坐标未改变。CPU 三角化 GLB 使用原生 `generated_mesh` 的可选模型通路；其产品语义仍是可见表面，不是补全形体。原生 `observed_surface` 通路限定源照片，不能直接用于地图选择。

`scripts/build_video_object_models.py` 按相邻有效像素三角化 SAM 掩码内的传感器深度，保留源像素映射与颜色。最大边长5厘米，不封孔洞、轮廓或背面，不跨视角合并 ID。`--semantic-review` 复用已有照片解释，验证完整观测 ID 集、source.png 与 overlay.png 的哈希；相机更新时重新导出几何，视觉证据一致才复用，并记录原核验场景与新几何场景哈希。

`analysis.identityCandidates` 是未确认的重现候选，格式 `{fromEntityId,toEntityId,fromTrackId,toTrackId,status:'unconfirmed_reentry_candidate',cosine,evidenceFrames,referenceFrames}`。from 是新轨迹，evidenceFrames 属于它；to 是先前轨迹，referenceFrames 属于它。界面只显示未确认关联和两组可跳转的来源帧，不修改 ID，不将余弦相似度称为概率。

真实 frame120 的6份模型共74664三角面，GLB约1.54MB；照片核验有2份类别清楚、3份局部可见、1份类别不符。这是6份分割观测，不是6个已确认唯一物理对象。Chrome 已实际点击右桌三角表面，核对到原视频4.058秒、帧120、同对象mask和照片描述；纯检查断言模型确实进入可见/可拾取任务，避免仅有对象列表而无实际模型。


呈现时钟检查：真实第690帧的原PTS为23.236235056秒，Chrome呈现回调实际量化为23.236秒。仅 `requestVideoFrameCallback` 入口允许在0.5毫秒范围内找到唯一已有源帧起点时还原原PTS；不调整一般 `frameAt` 半开区间、不延长空档、不改变用户拖动时间。检查覆盖首帧、末端之外、间隙、多个近邻拒绝与真实第690帧。实际浏览器修复后同一视频帧正确显示该帧原ID。

最终002分析的真实浏览器检查：候选3↔1的帧210和候选6↔0的帧540均跳到准确源帧，保留未确认文字和两个ID；同一页新相机空间场景可见。六份对象独立视图缩放拖转后，点击椅子真实网格定位到chair-0、第120帧、原mask和照片描述；显示器对应同一源帧。控制台无警告或错误。旧001候选检查产物仅保留在 `runs/sam3-periodic-sam2-walking-001/reentry-ui-check/`，不接入交付清单。

选中人物的现有详情还显示当前空间对象的 `world_motion` 与 `motionEstimate`：参考关节中点、位移米数、实际时长及来源帧。`insufficient_evidence` 明示无估计，`below_resolution` 明示不能判定静止；数值均标为模型估计。时间更新或缺失三维观测会刷新该文本，不沿用先前位移。最终模型目录为 `runs/rgbd-walking-full-object-surfaces-002/`，承接连续观测校正后的 full-replay-002。

有空间网格时默认显示网格、对象和人物，原始点云由“点云”显式开关控制；无网格的 room 默认显示点云。“全部点云”会打开图层并适配全部原坐标点的范围。仅对象模式暂时屏蔽并禁用点云控制，退出后恢复原选择；不删点、不重新标定，也不随视频帧重新加载大模型。
