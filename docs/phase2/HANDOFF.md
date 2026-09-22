# Phase 2 视频空间 MVP — session handoff

更新时间：2026-09-19。本文件是新 session 的首读入口；实验代码基线 `a776fe3b4a8dd92fe5fe349fe0c301aeec159b44`。本文件所在提交额外保存交接文档，不改变模型或数据。机器可读快照在 [HANDOFF.json](../../.planning/HANDOFF.json)。

**完整计划与落地愿景见 [PLAN.md](PLAN.md)**（S0–S12 里程碑、验收、待用户拍板事项）；十个只读代理的原始梳理/路线/挑错保存在 `ART/runs/plan-workflow-017/agent-results.json`。

## 先明确当前结论

**MVP 未完成，完整房间点云也没有做好。** 用户最后指出 ORB 的点云仍很差；上一轮已确认，不能把“有文件、通过来源检查、点数增加、页面能转动”当作交付。最新 DROID 分辨率翻倍实验仍有孔洞、散点及破碎表面，未替换主页面。LingBot 房间版本同样未通过几何检查。

这次用户要求把已有工作 push 并交给另一个 session。不要重新开始研究、重复运行已经失败的对照，也不要把本 handoff 当成质量完成报告。下一 session 继续推进可用空间重建；不只改标签或换到另一份结果。

## 用户目标、授权与边界

- 连续第一视角视频 → 同一坐标中的场景记忆与重建 → 随时间 replay 相机、人、车和其他移动对象；支持 A 点到 B 点后对同一对象的持续关联。
- 最终需要可检查的点云、模型、对象与照片/视频来源对应，以及已有照片 VLM/SAM/CAD/几何测量/报告能力。不能只交点云或孤立帧模型。
- 人体骨架、完整人体估计和动态物体状态分开记录；当前短轨迹 ID 不等于持久物理身份。无观测时清空动态状态，不能延长旧观测。
- 不训练新基础模型；优先复用预训练算法和已有工程。不要为当前数据硬编码对象，不用臆造几何、GT 建图、降低检查门限或删除难对象来制造完成率。
- 当前累计预算授权 **100 美元**；最新已核算与预留上限 **60.36 美元**（含本轮给定相机深度实验预留2美元，资源时长估算见账本 posed_depth_room_scope），账单未对账，不能称为实际已花。剩余未分配预留空间 **39.64 美元**。包含失败尝试、验证和服务开销；不是每次实验 100 美元。
- 用户要求保护本地 RAM/磁盘，减少重复哈希校验，保留已有 SAM3 等代码/原始结果。使用同源缓存与分阶段检查，不能取消输入/输出的来源合同。
- 旧 goal 工具仍为 paused，文字留着过时的 40 美元；不要由它覆盖后续 100 美元授权，也不要把整体目标标成完成。当前交接不新建自动任务或后台收费循环。
- 用户给出的 AGENTS 约束：追根因、复用共享实现、最小改动；不加兼容补丁/兜底/未经要求的架构；非平凡逻辑留下一个可运行检查；全链路验证。

## 正确的源码与数据位置

| 用途 | 位置 |
| --- | --- |
| 当前源码 worktree | `/Users/adam/.codex/worktrees/panoptes-phase2-video` |
| 分支 | `codex/phase2-video` |
| Git 远端 | `https://github.com/admin-wekruit/ehs-spatial.git`，private |
| 远端默认分支 | `feature/ehs-spatial-mvp`；本次只推 Phase 2 分支，不合并、不部署生产 |
| Phase 1 原工作树 | `/Users/adam/Desktop/Tesla/panoptes-platform`，保留其独立改动 |
| ART：实验数据和发布产物 | `/Users/adam/Desktop/panoptes-public/research-notes/phase2` |
| Python / Modal | `/Users/adam/Desktop/panoptes-public/panoptes-serving/.venv/bin/python`、同目录 `modal` |
| 已安装 Vite | `/Users/adam/Desktop/Tesla/panoptes-platform/web/node_modules/vite/dist/node/index.js` |
| 真实样本清单 | `ART/video-mvp/manifest.json`（六个样本） |
| 来源/交付索引 | `ART/video-mvp/delivery.json`（182 个产物索引，数据代码基线 a776fe3） |
| 账本 | `ART/video-mvp/cost-ledger.json` |

**目录陷阱：`/Users/adam/Desktop/panoptes-public` 本身不是本项目 Git checkout。** 在其中执行 git 会向上发现另一个 Desktop 仓库，remote 是无关项目，并呈现大量无关改动。所有 git 操作明确使用上述 Phase 2 worktree；不要 stage、清理或 push 数据目录父级的东西。

Git 保存代码、文档及小型交接快照，不保存视频、权重和多 GB 原生数组。**同一 Mac 的新 session 可直接接手；仅 clone 到另一台机器并不包含所有实验数据。** 跨机器需要另行迁移 ART 中被索引的源文件/结果或从已有 Modal 卷收集；先恢复数据，不能用空 manifest 或重新付费推理冒充恢复。不要把凭据、模型权重或整个 ART 加进 Git。

## 服务与资源状态（交接时核实）

- 2026-09-19 检查：8799 没有监听，预览服务已停止，需要按下方命令启动；旧 URL 本身不是服务还活着的证据。
- 本地可用磁盘约 **42.22 GiB**。上一轮曾只有 3.9 GiB，此数已变；新任务开始前重新检查，不能沿用旧告警或假设无限空间。
- 最近 DROID GPU 与 CPU 任务上一轮均正常停止；交接时 Modal 列表没有活动的 `panoptes-droid-room-once` / `panoptes-lingbot-room-once` / `sam3-video-access` 应用。没有待等待的推理或待接管 subagent。
- 不停止/修改同账号其他已部署服务。新的收费任务使用唯一 run ID、明确时间上限、retries=0 与 attempt claim；不确定返回先查远端已有结果。
- SAM3.1 HF gated 权重访问：用户已经提交申请，最后实际检查仍 403。批准状态可能已变，下一次需要时做一次 read-only 检查。不是所有 Modal 登录都坏了：DROID/LingBot 已在同账号成功执行。不要再要求用户重复提交同一申请。

## 五分钟恢复

先从正确目录确认源码和文件。不要因旧 session 的声明省略现状核实。

```sh
cd /Users/adam/.codex/worktrees/panoptes-phase2-video
git status --short --branch
git log -3 --oneline
git remote -v
df -h /Users/adam/Desktop/panoptes-public
lsof -nP -iTCP:8799 -sTCP:LISTEN
```

8799 未占用时，在一个可保留的终端启动已有 Range 服务（不运行模型）：

```sh
/Users/adam/Desktop/panoptes-public/panoptes-serving/.venv/bin/python \
  /Users/adam/.codex/worktrees/panoptes-phase2-video/web/experiments/video-mvp/serve.py \
  /Users/adam/Desktop/panoptes-public/research-notes/phase2 --port 8799
```

这是 Starlette/uvicorn 的本地静态服务；不要换成 `python -m http.server`，此前会令浏览器视频 seekable 为零。已有构建位于 ART/video-mvp；不需要重建就能查看。只有前端源码改变后才按 [前端 README](../../web/experiments/video-mvp/README.md) 构建，必须 `emptyOutDir:false`，保留真实 manifest。仓库 `web/experiments/video-mvp/manifest.json` 只是空模板。

恢复后查看：

- [六个样本](http://127.0.0.1:8799/video-mvp/index.html?sample=walking)
- [LingBot 房间主结果，未通过](http://127.0.0.1:8799/video-mvp/index.html?sample=room-rgb)
- [DROID 新旧分辨率对照，均未通过完整房间](http://127.0.0.1:8799/video-mvp/index.html?manifest=/runs/droid-fr1-room-resolution2-004-preview2/comparison-manifest.json&sample=droid-640)
- [LingBot 四种配置的同源对照](http://127.0.0.1:8799/video-mvp/index.html?manifest=/runs/lingbot-room-official-cache-003-launch3-preview/comparison-manifest.json&sample=room-windowed)

核实原视频可播/seek、真实选中样本、点云与表面切换/旋转、来源时刻。用户截图曾显示 ORB 而 ambient URL 显示 room-rgb；**已经验证 sample 路由切换正常，没有证据认定路由 bug，也不要把质量差归因于用户选错页**。

## 当前六个样本：真实内容与缺口

所有相对路径从 ART 起算；机器可读快照保留实际 manifest 的六份完整记录。

| ID | 数据 / 当前 scene | 已完成与未完成 |
| --- | --- | --- |
| `walking` | TUM fr3/walking_xyz，28.927 秒 / 859 帧；`runs/lingbot-all-analyzed-replay-002/scene.json` | RGB LingBot，117736 点、259441 面、29 张源图纹理；389 份人物观测表面、205/412 次通过对齐的完整人体估计；6 份照片对象观测和1个椅子生成模型。仍有孔洞、人体缺帧，无通用持久身份 |
| `cars` | MEVA，14.667 秒 / 440 帧；`runs/lingbot-cars-002-objects/scene.json` | 5 辆车的二维短期追踪，147 张RGB几何、91544点、147份较大SUV可见表面；另4辆像素不足。固定相机，不是移动相机的完整车身/米制轨迹验收 |
| `room-rgb` | TUM fr1/room，45.405 秒 / 1362 帧；`runs/lingbot-room-windowed-005-preview/scene.json` | 681 帧原生 LingBot 窗口模式，9/19 跨视角检查通过，未达75%门限；仍有重影/厚表面/缺失。SAM3+SAM2 及RTMPose分析已做，不代表地图已好 |
| `room-droid` | 同一 fr1 RGB；`runs/droid-fr1-room-supported-display-003/scene.json` | 1362 filler 相机时刻、177原生几何关键帧、236213彩色显示点、20124网格面/423分量。显示来源正确，完整房间失败 |
| `room` | 同一 fr1 RGB；`runs/orb-fr1-session-b-001-replay/scene.json` | 11650稀疏特征点；续接会话只输出约30–45秒455个相机时刻。地图包含旧关键帧，不能说全部点只来自最后15秒。另两张地图未合并，没有稠密深度/网格 |
| `walking-rgbd` | 传感器深度对照；`runs/rgbd-walking-quality-replay-003/scene.json` | 94387静态点、59684面、670/1177次完整人体估计；控制组用RGB-D，不是普通手机RGB能力证明。仍有表面/人体缺口 |

对象/人物缓存不要丢：walking分析 `runs/sam3-periodic-sam2-walking-002/pose-preview/analysis.json`；cars `runs/car-seeded-001/preview/analysis.json`；room `runs/room-sam3-sam2-001/preview/analysis.json`。当前是 SAM3 图片周期发现 + SAM2.1 传播，不能标成 SAM3.1 已部署。fal 16帧测试只证明接口可调用，其掩码质量/实例身份未验收。

## 已做过的几何实验：不要重复付费

### ORB

原 `orb-fr1-full-002` 分成3张没有合并边的地图。当前 session-b 的 map3 有275关键帧/11650点/2条合并边，还剩另两图。`orb-fr1-revisit-full-001` 曾通过重复播放同一录像合并成一张图，但这是人工两遍输入且全局误差仍大，不可充当原45秒完整重建。诊断：`runs/droid-fr1-room-resolution2-004/orb-map-diagnostic.json`。

### DROID

固定源码 `2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99`；预训练权重 SHA `46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526`。

1. `droid-fr1-room-001`：官方轨迹显著优于碎片ORB，但低分辨率深度/表面仍不完整。全帧相机含官方 motion-only filler，1362输出不表示1362帧跟踪成功。
2. `droid-fr1-room-final-upsample-002`：已修复低显存后端先上采样、再BA导致的高分辨率旧状态。用最后匹配的学习mask重新上采样最终BA深度；GPU重算误差0、最终位姿与低分辨率深度不变。不要再提同一修复。模型320×240、离线采样160×120，177关键帧；117.15秒/峰值3.90GB。轨迹Sim3 ATE约0.040824米只用于评估，尺度没写回模型。
3. `droid-fr1-room-supported-display-003`：不再直接显示未筛选无色点；点云/融合使用同一跨视角支持并保留原RGB，236213显示点逐项通过XYZ/RGB检查。全量3398400原点保留；944849支持点（27.80%）。只是显示链修复，房间仍未通过。
4. **最新 `droid-fr1-room-resolution2-004`**：同1362帧/权重，只把推理放大到640×480，K和裁剪同步缩放；376关键帧，GPU 217.52秒/峰值19.68GB，导出211.21秒。28876800原点中8009271支持（27.74%），296640显示点，67107面/1702分量，最大分量23253面。浏览器总览/源视角/放大表面仍未通过；分辨率增大没有解决完整性。**没有替换主样本**。

最新薄元数据：`runs/droid-fr1-room-resolution2-004/`。小预览：`runs/droid-fr1-room-resolution2-004-preview2/`，内含 `review.json`、`metrics.json`、`collection.json`、对照manifest及GLB。全量数组/PLY/支持NPZ留在云端，未下载本机；只收集28,227,863字节。`metrics.scene_sha256` 对应云端原格式scene；压缩JSON的SHA不同，两份均在 `review.json` 中明确，不是几何被改。

云端卷 `panoptes-droid-room-001`，挂载 `/artifact`：

- `/artifact/runs/droid-fr1-room-resolution2-004/result/`：`prediction.npz`、原生状态、终态上采样证据、帧账本。
- 同级 `/export/`：全量PLY/支持NPZ/原格式scene；`/preview/`：可收集的小预览。
- GPU app `ap-dGCOI4p1COz7nRJJxW9M1n`，call `fc-01M2VAJK02JXH24RCT0VGGFMA6`；CPU app `ap-00qN3LpoNSsA5vhcQEQWsJ`，已完成。
- 首次CPU app `ap-OB0rgrp9SgECtUgVZaqtyz` 因导入卷上库后才reload失败；代码已改为 **volume.reload在导入`/artifact/site`之前**。没有重新跑GPU。
- attempt Dict `panoptes-droid-room-attempts` 防重复收费。不要重用已claim的run ID。

现有恢复工具：`modal_apps/droid_cloud_review.py` 调原导出器、独立显示检查并使用有32MiB/剩余2GiB限制的 collector；已完成的run直接读 `/preview` 收集，不再调用 review 重建已有 `/export`。`droid_room.py collect` 会收集原生大文件，不适合为了网页预览使用。

### LingBot

固定源码 `849e690bb086103637e44b1e91878d9d43a8bf0c`；权重revision `204754b72bb24f561f8d7e7e1e4e4cd9e809adf9`；卷 `panoptes-lingbot-map`，attempt Dict `panoptes-lingbot-map-attempts`。推理与导出入口 `modal_apps/lingbot_room.py`、`lingbot_cloud_compare.py`；原生预测留卷中，preview collector只收集小结果。

同681帧房间对照：

| 配置/预览目录 | 结果 |
| --- | --- |
| `lingbot-room-original-cache-review`，缓存间隔1 | 5/19检查通过 |
| `lingbot-room-official-cache-003-launch3-preview`，官方缓存间隔3 | 8/19；推理113.41秒 |
| `lingbot-room-rectified-004-preview`，RGB及mask同步去畸变 | 7/13；可检查帧对覆盖反而下降，未升级主结果 |
| `lingbot-room-windowed-005-preview`，官方64帧窗/16帧重叠 | 9/19；95.72秒，仍低于75%门限；当前主结果只作未通过诊断 |

不要把更密帧、官方间隔或windowed模式作为尚未试过的建议。原生点云有明显误差；光流链式跟踪也试过，不能单归因于帧对相隔太远。GT只用于事后轨迹诊断，不用于修正模型。去畸变和主点预测偏差是已查因素，不是已经证明的唯一根因。

### 给定相机的稠密深度（2026-09-19 第二个session，新增）

入口 `modal_apps/mono_room.py`（`infer` / `fuse` / `evaluate` / `self-check`）。全部复用 `droid-fr1-room-final-upsample-002` 的177关键帧相机、K和官方去畸变/裁剪（640×480），以及003的 `depth-support.npz`。没有GT/传感器深度进入 `infer`/`fuse`；`evaluate` 只把TUM传感器深度作事后独立评估（每种方法一个全局尺度）。

| 运行 | 方法 | 同一DROID显示规则的跨视角支持 | 传感器评估：相对误差中位数 / 10%以内 / 覆盖 | 结论 |
| --- | --- | --- | --- | --- |
| 基线 | DROID光流深度 | 27.8% | 全像素4.5% / 74% / 100%；仅支持像素2.6% / 92% / 32% | 低纹理墙面/地面无支持，孔洞 |
| `mono-anchored-room-full-007` | MoGe-3单帧、已知FOV、每帧一个尺度（只用DROID支持像素拟合，残差中位2.3%） | 21.4% | 4.6% / 82% / 100% | **拒绝**：稠密但不比DROID准，融合后出现平行重影层。不要再试单帧深度+尺度/仿射对齐（仿射只到3.7%） |
| `da3-posed-room-full-009` | DA3-GIANT-1.1，给定相机/K，177帧一次前向（A100-80GB，117秒） | **39.1%**（2%相对容差62.9%） | **3.1% / 88% / 100%** | 首个连贯房间：窗墙、桌面、显示器、柜子、侧墙连续且位置正确；**未验收**：地面中部与移动人物附近仍有孔洞，0.015体素网格2749个分量 |

DA3合同已验证，不同于此前MapAnything探针：返回相机与输入一致（断言1e-4）、返回K等于缩放后的输入K、相对DROID深度的拟合尺度0.995（离散1.4%）。59帧峰值显存14GB。**DA3-GIANT权重为CC BY-NC 4.0，只能做研究对照**；Apache-2.0的DA3-BASE（`da3-base-posed-room-010`，同输入，73秒）：同规则支持34.9%、传感器评估3.6% / 82% / 100%、尺度0.976（离散2.6%）——低于GIANT、高于DROID，是可商用的退路，尚未目检。

查看：[DA3与DROID同相机对照](http://127.0.0.1:8799/video-mvp/index.html?manifest=/runs/da3-posed-room-full-009/comparison-manifest.json&sample=da3-posed)；源视角网格光线投射证据 `runs/da3-posed-room-full-009/source-view-review.jpg`；结论 `review.json`。没有替换主样本。

`scripts/build_droid_replay.py::depth_support` 新增可选 `tolerance_fraction`（默认仍是固定的.005原生规则，自检通过）；2%相对容差只用于新深度源的融合筛选，同规则数字另行报告，不是降低原验收门限。

**009为什么还有孔洞（实测拆分，177关键帧、2%容差）**：保留62.9%；18.0%因为6个时间相邻关键帧里不到2个看得到该点（规则从不查其他时刻的重访视图）；18.2%看到了但深度相差>2%（集中在远处，基线太少）；0.9%被远深度先验剔除。

**`da3-posed-room-midframes-012`（当前最好，未验收、未替换主样本）**：①任何看到该视图≥10%的关键帧都可投票（仍需≥2个视图2%内一致）；②相邻关键帧的中点帧用官方filler相机加入，352视图一次DA3前向（A100-80GB）；③已有人物掩码（源帧750–1349，34个视图）融合前剔除；④显示点云每2cm原生体素保留一个真实源像素。结果：支持率90.7%；传感器评估（仅评估）2.6%中位 / 90%在10%内 / 覆盖100%；源视角命中 kf60 87→95%、kf110 61→81%、kf135 94→98%。规则没有放水的证据：新纳入像素3.4% / 86%，仍被拒像素6.7% / 66%。`da3-posed-room-allviews-011` 是只换投票规则的CPU对照（84.2%）。

查看：[352视图 / 177关键帧 / DROID 三方对照](http://127.0.0.1:8799/video-mvp/index.html?manifest=/runs/da3-posed-room-midframes-012/comparison-manifest.json&sample=da3-352)。

**漂浮碎片的来源与处理（013–015，均为CPU、复用012深度）**：实测深度边缘像素（相邻跳变>3%，占2.1%）中位误差12%、26%严重错误——深度网络固有，且被我们把378×504深度双线性放大到640×480加重；DA3最低10%置信度像素含全部严重错误的77%，而我们此前存了置信度没用。`014-edge-carve`＝边缘剔除＋直接复用 `filter_video_static_surfaces.depth_evidence` 与其原验收规则（≥3视图支持、穿透≤max(2,15%)）：分量6418→2167，支持率88.7%，代价是细结构边缘被侵蚀、kf110命中81→72%。置信度过滤两档都**拒绝**：官方默认40%（013）和10%（015）都会删掉深度正确的窗/墙/柜门。顺带修了共享规则的根因bug：`depth_support` 远深度先验用 `mean`，任一无效像素会让整帧作废，现为 `nanmean`。对照页 manifest：`/runs/da3-posed-room-014-edge-carve/comparison-manifest.json`（sample=da3-352-clean）。

**米制尺度与地面一致性（`mono_room.py metric`，`runs/sam3-room-floor-016`＝SAM3 “floor” 12关键帧）**：用户给的锚点是“相机离地约1.6 m”。复用 `ehs_spatial.geometry._ransac_floor_plane` 与 `estimate_native_ground` 的视图规则（≥200点、跨视图法向≤5°；3个视图因SAM3把小块非地面标成floor被剔除）。结果写在 `da3-posed-room-014-edge-carve/metric-scale.json`：①1.6 m假设给出1.656 m/原生单位，而评估用真值尺度是1.334（+24%）——这段视频操作者实际持机约1.29 m；②MoGe-3无真值估出的尺度1.332，与真值差0.2%（仅此一段视频）；③**各视图地面不共面**：法向一致的视图之间高度差4–16 cm原生单位。判别实验（同相机、同掩码，仅把深度换成传感器深度作诊断）下各视图地面一致到0–5 cm，所以根因是DA3在低纹理地面上的深度偏差而不是DROID相机漂移；相邻视图带同向偏差，2%跨视图投票抓不到。对EHS高度/距离类测量这是必须解决的误差源。

**2026-09-20 更正（上面第③点的结论有误，保留原文以便追溯）**：用联合最大一致平面重算，DA3地面跨视图是共面的——93.7%的地面像素在±2%深度容差内，各视图高度差0–8 cm原生单位且主要来自小块远处视图。此前“4–16 cm不共面”是拿单个视图自己的平面外推的假象（单视图法向偏2°×2–3 m力臂）；`geometry._ransac_floor_plane` 专挑“最低的受支持平面”，对已分割的地面点不适用，`metric` 改用 `consensus_plane`（有自检）。因此**不需要**在全部关键帧买floor掩码，也不需要把地面压平。
真正存在的是绝对尺度参照之间3–9%的互相不一致（仅评估数据）：这几个视图里DA3与DROID三角化深度都比传感器深度短3.5%（非地面）/5.9%（地面）；传感器地面推出的相机离地1.446 m又比动捕真值1.491 m短3%；DROID轨迹分段Sim3尺度漂移只有±4%，解释不了。地面特有的偏差约2.4%（1.4 m处约3 cm）。这段视频实际持机中位1.49 m（p10–p90 1.32–1.61），不是此前写的1.29 m。
**用户决定（2026-09-20）**：尺度锚点用持机高度1.6 m；必须能本地部署；最终目标包含移动物体。1.6 m在本片段给出1.594 m/原生单位，比MoGe估计1.332高20%，`scale_sources_disagree_over_10pct` 为真。详见 [PLAN.md](PLAN.md) 第0节。

**通用视频运行器与移动对象回放（2026-09-20，S8+S2b）**：`modal_apps/droid_room.py --clip`（`CLIPS` 表；fr1-room 路径不变，其他片段的RGB归档放卷内 `clips/<名字>/`）；`mono_room.py` 跟随DROID运行记录的片段，缺省自己按固定规则重算支持掩码（与003存档逐位相同）、`--every N` 均匀取帧、`--video` 自己写回放帧、`dynamic` 把每个实体掩码用同视图深度抬成带时间的可见表面、`evaluate --dynamic-masks` 分开评估移动/静态像素。
walking片段（fr3/walking_xyz，两人走动，管线内只用RGB）：DROID 859帧相机对真值RMSE 1.7 cm（不需要人物掩码；首次018因我的改动把 `image_size` 参数注释掉而在24秒后失败，019成功）。相机几乎不动→只有24个关键帧，所以DA3取每3帧共303视图。`runs/da3-posed-walking-020`：静态像素中位误差1.35%/90.5%在10%内（DROID光流深度11%）；移动实体像素按静态场景尺度评估4.8%、偏近3.7%；静态地图融合前剔除266个视图的人物像素；431份人物表面；人物三维位置误差（仅评估，370次观测，距离中位1.6 m）中位8 cm、p90 23 cm、最大1.2 m。场景帧按既有walking样本的约定=抽样视图、每帧持续到下一视图（≤0.101 s）。查看：`/video-mvp/index.html?manifest=/runs/da3-posed-walking-020/comparison-manifest.json&sample=walking-da3`，点“当前视角”。未验收；旧样本里的骨架/完整人体尚未接到新几何。
**规则契约测试（S1）**：`tests/check_platform_policy_geometry.py`（实现+两名对抗审查按变异测试核对+修复一轮；serving venv没有zen-engine时自动打桩，Tesla venv走真实引擎，两边都过）。它钉住了一个未来“米制事实生产者”必须满足的字段契约，并记录了规则引擎现状：`not_inside` 没有不确定度带（离区域1 mm、不确定度0.5 m也判通过，两条现成OSHA模板都用它）；占地嵌套（显示器在桌上/椅子在桌下）在净距类规则下一律未通过；`nativeToMeters` 从不被读取；`measurements` 为None会崩溃；高度类规则用高度的不确定度覆盖了占地的不确定度；`max_tilt` 经平台路径永远证据不足。这些只记录、未改引擎。

**许可闸门（S4）与可商用组合**：Apache的DA3-BASE按同一配方：房间 `da3-base-room-352-021` 静态3.2%/82.5%（GIANT 2.6%/90%）；walking `da3-base-walking-022` 静态2.9%/85.9%，但**移动的人14.8%、偏近13%、只有40%在10%内——BASE的多视图深度对移动物体不可用**（GIANT自身4.8%/偏近3.7%/83.6%）。退路已验证：`moge-walking-024`（MoGe-3单帧，303视图，L4）逐视图用静态像素的稳健比值对齐到地图尺度后，人物6.5%/偏远4%/70.5%；对GIANT地图反而不如它自己的深度（7.3%）。所以：研究用=GIANT全包；可商用=BASE静态地图＋`dynamic --entity-depth-run` 用MoGe处理移动像素。对照页 `/runs/da3-posed-walking-020/comparison-manifest.json` 有三个样本（walking-da3 / walking-base-moge / 旧LingBot）。MoGe-3权重许可未核实。
**对象级地图（S6首版，未验收）**：`scripts/build_video_object_map.py`（有自检）。掩码 `sam3-room-objects-023`＝24关键帧×6词（monitor/desk/chair/cabinet/laptop/keyboard），144次fal调用、152实例。先试平台自带关联器跑到不动点（`--method clique`，`room-object-map-025`）：欠合并，显示器38观测→17组。再按ConceptGraphs可运行配方借算法（`--method overlap`，`room-object-map-026`）：每实体维护世界点集、按“掩码点落在实体点集附近的比例”≥0.5加入、同标签才可比、每帧每实体一个掩码、重叠>0.7合并、≥3次观测才算确认；邻近距离沿用平台关联器的3%相对深度容差。结果44实体/15确认：显示器9（5确认）、桌子9（3）、键盘9（4）、笔记本3（2）、椅子9（1）、柜子5（0）。`plan-view.png` 布局自洽；相邻拼接的桌子并成一个实体（desk-014含20个观测/16帧）。没有独立的人工清单，所以“恰好出现一次/零错并”尚未验证；实体还没进查看器和报告。

**第一条从视频到规则引擎的判定（S7首版，演示性质）**：`scripts/evaluate_video_policy.py`（自检经真实zen引擎在0.711 m两侧得到 FAIL/PASS/NEEDS_REVIEW；必须用带zen-engine的venv，如 `/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python`，不打桩）。它是唯一把原生单位换成米的地方：只给≥3个视图确认的实体写事实；占地=已见侧面的凸包（只会偏小→间距是上界，PASS只表示“没观测到阻挡”）；每个实体的不确定度=深度噪声×距离 ⊕ 尺度分歧×最近间距/2；场景文档按S1契约构造并过 `validate_document`。为让柜子达到确认，在柜子出现的700–1000帧补买了12帧cabinet掩码（`sam3-room-objects-023/cabinet-c`），对象地图重建为 `room-object-map-027`（36视图，带地面平面上的占地/高度/像素框）。
`room-policy-029`：规则取 OSHA 1910.36(g)(2) 通道净宽≥0.711 m（适用性是为演示而断言的，桌子与柜子之间并非指定的疏散通道）；尺度按用户决定用1.6 m持机高度（1.594 m/原生），与模型估计相差19.7%并计入不确定度。结果：desk-016 PASS（最近柜子1.98 m）；desk-015 间距0.85±0.26 m 与 desk-021 间距0.56±0.24 m 均为 NEEDS_REVIEW（引擎原话：无法诚实地选边）；10个只见过1–2次的候选实体被排除。桌高0.8–0.9 m、柜高2.3–2.4 m，与“1.6 m锚点偏大约两成”一致。`room-policy-028` 已标记作废（尺度文件早于分歧字段，尺度不确定度误为0）。引擎限制新发现：主体与对象不能用同一标签（目标会与自己比较，间距0→未通过），所以“桌子↔桌子”过道无法表达。

**视频场景进入现有报告（S5首版，2026-09-20，仅本机）**：`scripts/import_video_scene.py`（需平台venv `/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python`，有psycopg/zen/modal）。零模型调用：去畸变关键帧（复用 `droid_room.prepare_image`）当“照片”＋DROID关键帧相机＋房间网格（sourceContext观测表面，未验收）＋地面与尺度记录＋对象地图实体及其掩码观测（≥3视图=confirmed，其余association_pending且不带测量）＋每个实体“掩码最大视图”的可见表面（`build_video_object_map.py` 导出 `surfaces/*.npz`，复用 `observed_surface`）。走平台原有流程：create_project→register_asset→import_scene作业→`commit_edits migrateScene`（现行阅读器只开schema v2，v1会被分流到本地不存在的 readers/v1）→create_publication；全部通过平台校验。
本机基础设施：Postgres 16数据目录 `ART/.platform/pgdata`，只监听127.0.0.1:54329（启动需 `LC_ALL=en_US.UTF-8`，否则macOS上报 “postmaster became multithreaded”）；`PANOPTES_DATABASE_URL=postgresql://panoptes@127.0.0.1:54329/panoptes_video`，`PANOPTES_BLOB_ROOT=ART/.platform/blobs`；API+阅读器：`uvicorn ehs_spatial.platform.runtime:application --factory --host 127.0.0.1 --port 8792`，`PANOPTES_WEB_ROOT` 指向Phase 1工作树已有的 `web/dist`（只读使用）。导入结果清单（含项目capability，权限600）在 `ART/.platform/imports/`。
当前报告：`http://127.0.0.1:8792/app.html#/reports/9f42a1c7-d4bb-4a18-8b92-84b61711d0fb`（对象地图 `room-object-map-030`，与027分组完全相同）：33张关键帧、154个掩码观测、46个实体/17个已确认、234个资产；选中实体会跳到其关键帧并显示“已绑定多张原图·多视角身份已关联”；“场景3D”显示当前关键帧实体的观测表面。未做：对象列表仍写“暂无三维资产”（另一条判定）、CAD平面投影为空（需 `planProjection`）、房间网格未显示、安全评估未接入（`room-policy-029` 的判定还没写成平台evaluation）、时间轴/移动对象不在报告里。

**用户反馈与骨架（2026-09-20）**：用户看过后：房间“新的更好”（以 `da3-posed-room-014-edge-carve` 为基线）；移动的人“出现效果可以，但需要骨架”；LingBot旧样本里的“物体选择”留到对象进查看器那一步；本地Postgres可以先用（用户有Supabase连接 “Wekruit-pa”，本机查不到其配置，迁移前须确认是否生产库）。
骨架：`mono_room.py dynamic` 复用 `build_replay_scene.surface_joints`，把缓存的RTMPose二维关节（源像素→`rectified_pixels` 去畸变/缩放/裁剪）抬到人物可见表面；walking 4207/6303个可信关节抬成三维（其余在掩码外、轮廓边缘或深度不一致，留空）。查看器原本**有意**在有人物表面时隐藏骨架线（`scene.ts::hideLinks`，骨架线贴在表面后会被挡住），旧LingBot样本也一样看不到三维骨架；现加“骨架”开关（`setSkeleton`，与人物表面二选一），并把标记尺寸改为按骨骼中位长度取比例（相机几乎不动时原来只有几毫米粗）。查看器已按README重建到 `ART/video-mvp`；检查 `scene-check.ts`、`check.mjs` 通过。
主样本页 `ART/video-mvp/manifest.json` 已把 `walking-da3`、`room-da3` 放在最前（备份 `manifest-before-da3-promotion.json`），旧六个样本保留；`delivery.json` 的哈希索引尚未更新。

**用户对产品形态的明确要求（2026-09-20）**：要的是一种**视频报告（视频工位）**，与照片报告并列：以点云/三维房间为主工作区，在里面具备照片报告已有的功能（选物体、看证据帧、测量、CAD平面图、EHS判定）；对象地图要按照片报告已有的CAD做法呈现；骨架“能结合就行，后面再说”；对象地图“大概没问题”。
据此 `import_video_scene.py` 已改为：每个观测一个观测表面（`build_video_object_map.py` 导出 `surfaces/ENTITY--label-frame-instance.npz`，154个），每个表面用平台自己的 `reconstruction._plan_projection` 生成平面投影（地面法向来自拟合地面；精确三角并集，不取凸包不补洞）；房间作为sourceContext实体同时带 `observed_surface` 网格和 `point_cloud`（`supported-keyframe-points.glb`，阅读器原生支持GLB点集），放置状态confirmed（同相机下位置精确；“未验收”指完整性，写在标签和来源说明里）。
当前视频报告：`http://127.0.0.1:8792/app.html#/reports/0ec6b467-c009-42af-8d81-b119aabc2a1a`（对象地图 `room-object-map-031`，344个资产）：空间表示下拉里“点云”已可用；CAD面板“照片观测投影”模式当前关键帧 9/45 条记录有投影、53个轮廓；证据联动正常。对象列表“当前模型：暂无三维资产”是事实（尚无生成/参数化对象模型，S10）。仍缺：安全评估栏（`room-policy-029` 的判定还不是平台evaluation）、观测范围/米制尺寸显示、对象模型、时间轴与移动的人、界面文案仍写“照片”。

**视频报告的四视角要求与参数化模型（2026-09-20）**：用户明确视频报告要四个联动视角——视频帧（每一帧可点选内容，显示对应对象及其特征）、点云、Blender/模型复现（与点云同坐标、对称对应）、CAD；并且“需要模型”。
已做：`import_video_scene.py::extent_box`——对每个≥3视图确认的实体，在拟合地面上取占地的最小外接矩形＋观测到的底/顶高度，生成贴地朝向的 box 原语（不延伸到地面、不补未见面），按 `repository.py` 写原语的同一组字段入库（`placementState=unconfirmed`、`placementReason=requires_alignment_confirmation`、`currentModelTransform`，迁移后自动成为active model），并用 `_plan_projection` 生成模型平面投影。自检：17个实体的占地与高度范围全部落在各自盒内（最大越界0）。报告 `http://127.0.0.1:8792/app.html#/reports/7acca2f0-c91f-487f-ae56-c97a36b4da3b`：CAD“当前模型投影”17/45条有投影；“暂无三维资产”45→28。
Blender：本机没有Blender可执行文件，`blender_export.export_scene_revision` 会报 `blender_worker_unavailable`；用同一模块的 `prepare_export`+`write_glb` 导出了模型场景 `runs/video-report-models-032/scene.glb`（17对象、33源相机、校验passed；可在Blender里直接导入glTF）。`models-over-pointcloud.png` 显示盒子与点云俯视对齐。已知问题：`chair-006` 盒子偏大（1.44×0.74原生单位，疑似相邻椅子被并或掩码外溢）；桌面岛被分成两个相互重叠的桌子实体；盒子只是观测范围的替身，不是对象形状。
尚未做：每一帧（而非33个关键帧）的点选——需要把实体掩码用SAM2.1在关键帧之间传播（人物已有同样流程），以及报告前端的视频视角；生成式对象模型（RecGen，非商用许可、按次付费）；`.blend` 文件（需要带Blender的worker）。

**生成式对象模型进视频报告（2026-09-20）**：用户用照片工位报告“场景3D”的截图说明他要的“模型”就是那种——每个物体是生成的真实形状模型＋带纹理的观测表面，摆在重建坐标里。`scripts/build_video_entity_model.py` 复用照片流程的同一条通道（`recgen.source_grid_crop`→`recgen_transport.invoke`→`adapt_output`→`build_lingbot_object_model.evaluate` 的源视角一致性闸门：轮廓IoU≥0.65、深度中位≤4%、p95≤10%、≥300支持像素），证据换成实体最大掩码的去畸变关键帧＋DROID相机＋该视图的DA3深度；每实体一次有日志的调用，有validation的不再请求，单个失败只记录不中断。`runs/video-entity-models-033`：16次调用，9个通过（monitor-036/038/039/040、laptop-033/034、keyboard-025/028、cabinet-005），6个被闸门拒绝（cabinet-001 p95 11.3%、chair-006 IoU 0.646、desk-021 IoU 0.33、keyboard-024/027、monitor-037），desk-015（808×808裁剪）客户端240秒超时→“结果未知、永不重提”，desk-016跳过（大面积支撑面不适合单物体生成）。RecGen权重为非商用许可。
导入器：生成模型按“锚定观测”绑定到实体（不按实体编号，避免重编号后张冠李戴），通过的作为 `generated_mesh`（放置待确认，带 `sourceConsistency` 和平面投影），未通过的退回盒子。对象地图 `room-object-map-035`：掩码根 `sam3-room-masks-034`（对象掩码＋地面掩码的符号链接），新增 floor 标签（8观测→7个未确认实体），观测表面改为全分辨率（140 MB）。报告 `http://127.0.0.1:8792/app.html#/reports/f3c9e86a-5150-4de2-a322-d57bbbf46bfb`（53实体、376资产）。平台导出器产出 `runs/video-report-models-036/scene-models.glb`（17对象，passed）和 `scene-observed.glb`（163对象，passed）；`scene-preview.jpg` 是房间网格＋9个生成模型的离线渲染。`.blend` 仍需带Blender的worker。

**用户反馈（2026-09-20）后的修正**：①报告模型层“全是白的”——白盒子替身不可接受：`import_video_scene.py` 不再放 `extent_box` 盒子（函数保留未调用处已删除调用），没有通过检查的生成模型的已确认实体改为把最大的那块带纹理观测表面标为 `sourceKind=observed_reference_surface`（阅读器模型层会显示，与照片报告里围栏/地面同一做法；只标一块，避免同一物体多视图叠层）。报告 `http://127.0.0.1:8792/app.html#/reports/395d9f05-0109-489b-9148-8d342671b5f5`。②本机有Blender：`/Users/adam/Desktop/panoptes-public/.tools/blender-4.5.9/Blender.app/Contents/MacOS/Blender`；用平台 `blender_export.export_scene_revision` 导出到 `runs/video-report-blender-037/{models,observed}/scene.blend|scene.glb`（均 passed；298 MB / 675 MB，注意磁盘）。平台导出只有 models 与 observed 两种场景，没有二者合一的模式。③用户同意换更清晰的视频（TUM 640×480且有运动模糊）；DROID/深度管线目前假定640×480输入栅格，换高清源需先把 `prepare_image` 的缩放/裁剪与K换算泛化。④“视频＋模型＋点云要结合”：`scripts/attach_entities_to_replay.py` 按回放查看器已有的 `staticObjects`/`generatedModel` 契约，把对象地图里≥3视图确认的17个实体（9个带通过检查的生成模型）接进回放场景 `runs/room-replay-objects-038/scene.json`（掩码与帧用原始源像素文件，叠加与视频严格对齐），主样本 `room-da3` 已指向它：同一页面里有视频/时间轴、点云/网格、可点选对象（回到源帧）、“查看生成模型”切换、“仅看对象表面”。

**每一帧可点选（2026-09-20）**：`scripts/project_entities_to_frames.py`（有自检）不做逐帧分割，而是把对象地图里≥3视图确认的实体的三维点（各观测表面的顶点，抽样800）用每一帧的官方filler相机投回原视频像素（`cv2.projectPoints` 含镜头畸变），对融合房间网格做射线遮挡判断（近于点距离5%以上的命中视为遮挡），可见点的凸包作为可点选轮廓（analysis的 `polygons`，entityId=`obs-<实体>` 与回放场景的静态对象同名，所以视频↔三维↔生成模型联动）。零模型调用、6秒：1362帧中1158帧有对象、7516个轮廓（中位4个/帧）；相机看向没有已确认实体的区域的204帧如实为空。一致性（非独立精度）：在117个有SAM3掩码的观测上，投影轮廓与该帧掩码凸包的IoU中位0.84、85%≥0.5（desk 0.66：实体是整片桌面岛而单帧掩码只盖一部分）。已并入已有的人物逐帧分割。`runs/room-frame-selection-039/analysis.json`，主样本 `room-da3` 已指向它。现有行为：点选静态对象会跳到它的源帧。

**网格为什么糊、照片贴图与高清多视角片段（2026-09-20）**：用户反馈“找更清晰的视频”。先查原因：融合网格的颜色挂在体素角点上（房间3 cm、ARKit 2 cm），视频再清晰网格也一样糊——瓶颈是表示方式，不是输入。`scripts/texture_fused_mesh.py`（有自检，零模型调用）：几何不变，每个三角面取“三个顶点都无遮挡看到它（对网格自身做射线判断，3%容差，朝向余弦>0.2）”的源照片；不是逐面取最优（那样是迷彩拼块，试过），而是贪心覆盖——每次让“能以不低于该面最优视图一半质量（朝向/距离）看到最多未认领面”的照片整片认领，接缝少、用图少；每张照片按融合颜色做一个全局曝光增益（对局部色调差无效，试过，保留但作用有限）；没有照片完整看到的面保留融合颜色；贴图总像素有预算（默认60 MP）。结果：房间 `runs/room-textured-043`（479722面几何不变，352视图里用50张，87.6%面贴图，GLB 13 MB，对照图 `comparison.jpg`：书封面“OpenCV”可读），联动页 `room-da3` 已指向它（清单备份 `video-mvp/manifest-before-texture-043.json`）；报告导入器加 `--shell-glb`，新报告 `http://127.0.0.1:8792/app.html#/reports/fe4be333-277c-4bc7-b0f7-1806461bcbf4`（平台自己的GLB读取器读出同样479722面；**报告页里的显示未目检**——当时浏览器面板被隐藏无法截图；同一个 native viewer 在联动页里已目检通过）。
高清片段：本机只有 ARKitScenes upsampling 子集（29个场景，每个8–47张1920×1440静态照片、约2秒一张、多数竖持所以画面横躺；家居场景），**不是连续视频**。`scripts/prepare_arkit_clip.py` 把一个场景包成“自带相机的片段”（`run.json` 的 `clip_definition`：`raster=resize`、`metric_cameras=true`、逐帧内参；参考深度与标注只复制到 `evaluation_only/`），`mono_room.py` 据此跳过DROID支持/尺度锚定（scale=1）、逐视图K、米制场景（`units: meters`，查看器契约的拼写）。场景42445448：`runs/arkit-42445448-cameras-040` → DA3-GIANT一次前向 `runs/da3-posed-arkit-041`（A100-80GB，52秒，预留0.25美元，上限68.61）→ 对激光参考深度（仅评估）**不拟合任何尺度**相对误差中位3.5%（3.9 cm，偏近2.4%），全局一个系数后95%像素在10%内——设备米制相机下不需要1.6 m举机高度假设；跨视角支持只有54%（视图稀疏，相邻0.38 m），孔洞多；贴图 `runs/arkit-textured-042`（21张照片，97.8%面）；联动页样例 `arkit-da3`（每秒一帧的幻灯片 `data/arkit-42445448/preview.mp4`）。fr1 房间评估回归不变（0.025864720344543457）。**没做**：这个场景的分割/对象地图/生成模型（横躺画面对SAM3/RecGen不利，且要花钱；`import_video_scene.rectify/run` 与 `project_entities_to_frames` 的640×480/单一K假设也还没改）。结论：要“更清晰的视频报告”，先用照片贴图（已对现有房间生效）；真正的高清连续视频仍需用户自拍或批准下载一个公开数据集（下载需用户逐项批准）。

**“模型场景和点云差很多”“有物体没检测出来”（2026-09-20，用户提问后的实测）**：两件事同一个根因——模型层只装得下“被检测到并被≥3视图确认”的东西，而分割只跑了手写的6个词＋floor，每词24个关键帧（1362帧里有掩码的源帧共40个）。实测：通过检查的9个生成模型到该实体全部观测点的中位距离0.5–2.8 cm、中心偏差2–6 cm（cabinet-005 26 cm、keyboard-035 12 cm 除外）——**不是摆放问题**；差在覆盖：17个确认实体里只有9个有模型，桌/椅/柜（desk-015 生成超时不可重发，desk-021、chair-006、cabinet-001 未过源视角一致性）只显示某一个视角的残片（桌面到处是洞），房间本身按阅读器规则不进模型层（`core.ts::entityGeometryForLayer` 对 sourceContext 返回 null；参考面必须绑定该实体的观测）。**撤回一个错误假设**：我曾以为掩码溢出污染了实体尺寸（一次用原始 min/max 算出“显示器观测范围1.9 m”），实测多视图一致率98–100%，对象地图原有的2–98百分位范围本来就对；新增的 `consensus()`（≥2视图同格或邻格才算）只让 cabinet-005、desk-021 略缩小，保留但不是主因。
已改（提交 8f7a46d 及之后）：①`scripts/discover_video_vocabulary.py`——沿用 `scene_inventory.py` 的分工，让现有受限VLM看10个关键帧只给词表（`runs/room-vocabulary-044`，一次调用）：已有词之外给出 teddy bear / person / telephone / book / cable / mouse / houseplant；cable 太细碎、person 已有专门流程，未跑；另加 window、wall。`runs/sam3-room-objects-045`（7词×24帧＝168次，0失败，实例：book 25、mouse 15、telephone 5、teddy-bear 3、houseplant 2、window 16、wall 40；预留2美元，上限70.61）。②对象地图 `runs/room-object-map-047`（掩码根 `sam3-room-masks-046`）：70实体、27确认（原17；新增4本书、1鼠标、1电话、2窗、floor、wall）；`STUFF=(floor, wall, ceiling)` 每类合成一个实体（背景没有实例、视图之间不需要重叠——原来 floor 7个实体0确认）；每个确认实体存 `surfaces/<实体>--cells.npz`。③`import_video_scene.py::fused_part`：没有通过检查的生成模型的确认实体，参考面改为“该实体的格子在照片贴图融合网格里的裁块”（多视图合一、带贴图，sourceRefs 绑定该实体全部观测），不再是单视角最大残片。④`project_entities_to_frames.py` 跳过 STUFF（墙/地面的凸包会盖住整帧、吞掉所有点击）。产物：`runs/room-replay-objects-048`、`runs/room-frame-selection-049`（25实体，1157/1362帧有轮廓，10038个）、`runs/room-textured-050`（联动页 `room-da3` 已指向；清单备份 `manifest-before-vocabulary-047.json`）、`runs/room-policy-051`（同一演示规则，实体重编号后重算：2 PASS、1 NEEDS_REVIEW）、报告 `http://127.0.0.1:8792/app.html#/reports/d0457f3e-1cea-4110-942a-d25778aa3c5f`（71实体/27确认/259观测/588资产，导入耗时9分钟；**报告页模型层未目检**，离线渲染对照见本轮发给用户的图）。
仍未解决：模型层仍是“已确认实体的并集”，没被任何词覆盖的区域是空的（ConceptGraphs 用不分类别的“分割一切”＋每块配语义，所以处处有归属；我们是“先给词再分割”，要么继续扩词表，要么加一遍不分类别的分割——后者是结构性的下一步）；泰迪熊3个视图各拍到一部分（图像边缘），三维重叠<50%没合并——纯几何关联对“每次只见一部分”的物体不够，需更密的分割帧或外观特征（ConceptGraphs 的CLIP项）；桌/椅/柜仍无完整生成模型（换锚定视图重新请求要花钱，未做）；新实体（书/鼠标/电话/窗）没有生成模型。

**“要全部东西”：不给词也分割＋起名（ConceptGraphs 的顺序），报告与联动页合并（2026-09-20）**：用户明确工厂用例要“所有物体都复现”，并问报告和联动页能否合一、怎么和 ConceptGraphs 结合。做法：①`modal_apps/sam2_everything.py`（有自检）——SAM 2.1 全自动分割（`facebook/sam2.1-hiera-large`，Apache-2.0，可本地部署；SAM3/3.1 是“给概念词找实例”，没有无提示的分割一切模式，且本地权重 HF gated），输出与 `discover_video_keyframes.py` 同目录结构、标签 `object`；掩码按面积从小到大互斥化，`--known` 时去掉已被提示词掩码覆盖≥50%的和“墙/地面本身”。`runs/sam2-room-everything-052`：24帧679个掩码→留384个，L4 74秒。②对象地图 `runs/room-object-map-054`（掩码根 `sam3-room-masks-053`）：250实体、67确认（27提示词＋40全自动）。③`scripts/name_video_entities.py`——VLM 看“最大视图的裁图＋轮廓”给每个确认的 object 实体起名（4次受限请求，`runs/room-entity-names-055`：35个命名如 stapler / tape dispenser / calculator / milk carton / kleenex box / external hard drive / whiteboard / desk lamp，5个判为墙/地面/天花板碎片→`unnamed surface`）；名字只是属性：`evaluate_video_policy.py` 把 VLM 命名的实体排除在规则主体之外（否则一个被叫作 cabinet 的碎片会把 desk-021 判成 FAIL；`runs/room-policy-059`）。④联动页 `room-da3` → `runs/room-textured-058` ＋ `runs/room-frame-selection-057`（65实体，1286/1362帧有轮廓，23796个，中位14个/帧；清单备份 `manifest-before-everything-055.json`）。⑤**报告与联动页合并**：`web/src/VideoView.tsx`＋`ReportScene.tsx`——报告“照片”面板多一个“视频”模式（默认）：源视频＋每帧实体轮廓，点轮廓＝在三维/CAD/列表里选中该实体，在别处选中则视频跳到最近的可见帧；导入器 `--video --analysis` 把视频和瘦身后的逐帧轮廓（实体id换成报告UUID）作为资产挂在 `video_replay` 注解上。阅读器从工作树构建（`web/node_modules` 是指向 Tesla 检出的符号链接，已写进 `.git/info/exclude`），本机 API 用 `PANOPTES_WEB_ROOT=<worktree>/web/dist` 重启（后台任务 bd8rn54wa）。
**诚实的覆盖率与真正的瓶颈**：融合房间表面按面积只有31.7%属于某个已确认实体（只用提示词时28.3%；不算墙/地面是20.2%）。原因不是算法：24个分割帧里，房间表面只有66.4%被至少1帧看到、**16.5%被≥3帧看到**，而确认需要≥3视图。所以已启动 `runs/sam2-room-everything-060`：对全部352个深度视图做全自动分割（不带 `--known`，分块48帧/请求）。为此对象地图已改成能吃几千个掩码：逐视图数据按需重建（不再常驻内存）、实体点云超过2万点按格子稀疏化、合并改成不重启的多遍扫描、`absorb()` 把“其实就是某个提示词实体”的全自动组并入该实体（要求双向：≥50%的点贴着它，且覆盖它≥25%——否则桌上的纸会被桌子吞掉；24帧回归：12个并入，7个VLM曾起名为 monitor/keyboard/telephone 的重复实体消失，确认数67→60）、全自动视图每实体只留4块全分辨率表面；导入器每实体最多留8个全自动证据视图、不导入未确认的全自动碎片（数量记在 provenance 的 `leftOutOfReport`）。**352帧的地图、起名和报告尚未跑**。

**352帧全自动分割的结果（2026-09-20）**：`runs/sam2-room-everything-060`（352个深度视图，10831个掩码、留9435个，L4 858秒）→ 掩码根 `sam3-room-masks-061` → `runs/room-object-map-062`：517实体、**297确认**（提示词标签50＋全自动247；泰迪熊、盆栽这次确认了），56个全自动组并入提示词实体；**融合房间表面按面积90.4%属于某个已确认实体**（24帧时31.7%，只用提示词28.3%）。代价：构建14分钟、**峰值内存10.4 GB**（STEP网格的布尔掩码常驻；更长的视频前要改成压缩存储）、输出787 MB。起名 `runs/room-entity-names-063`（25次受限VLM请求；124 clear / 53 partial / 3 uncertain / 67 not_an_object；93种名字：cardboard box、paper、poster、power strip、binder、cup、webcam、stapler、broom、highlighter…；名字有错，例如两个贯穿全片的静态实体被叫作 hand）。走过的人：`scripts/drop_dynamic_entities.py` 按几何规则（实体≥半数有移动掩码的视图里，掩码≥50%落在移动人物掩码上）剔除7个实体（person×2、man's leg、heel×2、foot、1个碎片——与VLM名字互相印证）→ `runs/room-entity-names-064`；建图器新增 `--dynamic-masks` 从源头不收这些掩码（下次重建用）。下游：`runs/room-replay-objects-065`（291个可点选对象、9个生成模型；对象表面超过8000三角面的按自身尺寸/80的格子稀疏化，254 MB→52 MB）、`runs/room-frame-selection-066`（289实体，**1362帧全部有轮廓**，46659个，中位35个/帧）、`runs/room-textured-067`、`runs/room-policy-068`（9张desk实体：7 PASS、2 NEEDS_REVIEW——桌岛被拆成多个desk实体的老问题更明显了）。联动页 `room-da3` 已指向 067＋066（清单备份 `manifest-before-dense-064.json`），实测加载无报错。导入器修了一个只在稠密视图出现的错：非关键帧视图要用 DROID 的 filler 相机（原来只查关键帧表，KeyError）。预算上限73.11。模型层三阶段离线渲染对照已发给用户（17实体单视角残片 → 27实体多视图裁块 → 297实体）。

**推断地面（2026-09-20，用户：“模型3D还原得全部还原，就算没看到，地板应该很好还原”）**：这改变了模型层的口径（观测层仍只放看到的）。`scripts/infer_room_floor.py`（有自检）：已验证的地面平面铺满“观测几何在平面图上的凸包”（先试的最小外接矩形铺到了房间外 59.8 m²，`runs/room-inferred-floor-074` 标 SUPERSEDED），`runs/room-inferred-floor-077`：49.6 m²，其中16.3%被看到；颜色取看到的地面中位色，放在平面下方1 cm；只作为 floor 实体的模型（联动页 `generatedModel`＋如实说明的 note；报告里 `generated_mesh`，`modelBasis` 写明 inferred），不进观测网格。联动页 `runs/room-replay-objects-075`→`runs/room-textured-076`（实测可切换、无报错）。墙还没这样补。
**地板为什么缺、平地板融合（2026-09-20，用户截图提问）**：逐视图深度其实在地面平面上看到了16.3 m²（≥1视图）、13.0（≥2）、10.5（≥3），融合网格里贴地5 cm内的水平面只有1.1 m²。原因：跨视图支持和刻除都是**沿视线**比深度，地面是掠射角，法向的小误差沿视线被放大→被判不一致丢掉。`mono_room.py fuse --floor-plane metric-scale.json`（`floor_support`，有自检）：落在已验证的共识地面平面1.5倍容差内、且该5 cm地面格被≥2个视图看到的像素，保留并取“视线与平面交点”的深度（这样各视图融成一张平的地面而不是起伏的薄片），这些顶点不再受沿视线的刻除。第一版只“保留不改深度、≥3视图”（`runs/da3-posed-room-069-floor`）几乎无效（1.1→…仍起伏、被刻除），保留为对照；第二版 `runs/da3-posed-room-070-flat-floor`：贴地5 cm内水平面 1.1→4.7 m²（17 cm内 7.9→11.8 m²），三角面479722→507843；`metric-scale.json` 是从014复制的（同一平面）。**相机从没看过的地面（房间中间大片）仍然是空的**，俯视对照图已给用户看过，如实说明。下游已切到新几何：`runs/room-replay-objects-071`、`runs/room-frame-selection-072`、`runs/room-textured-073`（49张照片、84.7%面贴图），联动页 `room-da3` 指向 073＋072（清单备份 `manifest-before-flat-floor-070.json`）；带视频的稠密报告正在用新几何导入（旧几何那次导入跑了约50分钟被我停掉；导入器改为每个全自动实体只从其融合多视图表面画一次平面轮廓）。
**SAM3 / SAM3.1 权限（2026-09-20 只读检查）**：用户截图显示其HF账号已获批 facebook/sam3 与 sam3.1，但 Modal 密钥 `huggingface` 里的令牌属于账号 **AdminWeKruit**（令牌名 “Get Dataset”，fineGrained，全局权限为空），两个仓库仍返回403 GatedRepo。需要用户自己把“获批账号的、带受限仓库读取权限的令牌”换进该 Modal 密钥（不要经过聊天）；换好后跑 `modal run modal_apps/sam3_video.py::access` 复查。
**内存10 GB的真正原因（子代理查出，我原先的归因是错的）**：不是掩码，是 `mono_room.load`——循环里每次 `data["keyframe_final_fullres_depth"][key][::2, ::2]` 都让 NpzFile 把54 MB的DROID深度栈重新解码一遍，每个关键帧的跨步视图各钉住一份：177份、9.6 GB，所有调用 `load` 的命令（fuse 9.4 GB、metric、dynamic、evaluate）和建图器都中招。子代理因为只被允许改建图器，用 `mock.patch(np.load)` 绕过（稠密输入峰值10.4→2.75 GB，517实体/1663个npz逐位一致）；我改在根上修（`load` 里每个数组只解码一次，提交后撤掉绕行补丁）：24帧建图 9.2→1.6 GB 且实体逐字段一致，房间评估 1.35 GB、abs-rel 中位不变。还可再省（未做，够用）：深度按需读（约433 MB）、掩码压缩存储（约650 MB）。

**当前可看的视频报告（2026-09-20）**：`http://127.0.0.1:8792/app.html#/reports/164cf7fa-a8fb-443c-9856-625ff19442e6`——352帧全物体＋视频视图＋平地板几何＋推断地面：315实体、291确认、2111条观测、329张关键帧、4030资产；导入10.5分钟、峰值2.3 GB（每个全自动实体只画一次平面轮廓之后）。在1500×1000视口下实测（读页面状态，浏览器面板被隐藏无法截图）：第一格标题“来源视频 · 每帧可点选对象”，视频481×218可播放，0 s有30个、20.3 s有55个可点选轮廓；点“binder”轮廓→报告里该实体被选中；floor 实体“当前模型：模型已加载、位置尚未确认”（即推断地面）。此前的报告：`06c157ad…`（67实体、带视频、旧地板）、`d0457f3e…`、`fe4be333…`（无视频）。中途停掉的几次导入在本机测试库里留下了没有发布的孤儿项目，无害。

**报告三维面板卡死与修复（2026-09-20，用户：“空间资产前端卡死了”）**：稠密报告 `164cf7fa…` 的模型层打不开。根因在导入器：每个实体的融合贴图裁块把它碰到的每张源照片**整张**嵌进自己的GLB（中位8张、约3 MB/实体）→ 253个融合表面共454 MB、约2900张640×480贴图（约4.7 GB显存）。修复 `import_video_scene.py::fused_part`＋`atlas()`：只裁出三角面用到的像素块，按货架法拼成每实体一张图集（≤2048边长，有重叠/完整性检查）→ 同样253个表面共23 MB。新报告 **`http://127.0.0.1:8792/app.html#/reports/abad19d1-2392-49f4-86c4-7274535d271c`**（315实体/291确认/2111观测/4030资产，库内1619 MB，其中1341 MB是逐观测的全分辨率观测表面，只在“照片重建证据”层按关键帧加载）。实测（隐藏的后台标签页，浏览器会降速）：模型层约255个资产、约117 MB，238秒载完，“正在载入空间资产”消失，JS堆峰值约530 MB，无失败请求，没有卡死；**没能目检画面**（面板隐藏无法截图）。剩余最重的是10个生成模型（每个9–18 MB、约37万三角面，合计约100 MB）——下一步给报告用的副本减面。测量教训：同一标签页只改 `#` 后面的地址不会重新加载，旧报告的下载会混进统计；资源计时缓冲区默认250条，要先 `setResourceTimingBufferSize`。
**磁盘（2026-09-20）**：本轮从21 GiB降到16 GiB：报告资产库 `.platform/blobs` 2.6 GB（7次导入＋4次被中途停掉的导入留下的无发布项目；库按内容去重，逻辑大小远大于磁盘占用）、本轮编号040–079的运行产物2.2 GB（最大 `room-object-map-062` 787 MB，现用）、`data/arkit-42445448` 0.2 GB、pgdata 0.4 GB。已向用户列出可清理项（需用户同意才删）：清空并重建本机报告库只留最终报告（约2 GB）、`runs/video-report-blender-037`（1.3 GB）、`room-object-map-047/054`（430 MB）、`data/rgbd_dataset_freiburg1_room.tgz`（746 MB，已解压）、早期LingBot对照实验（约3.3 GB，按规则失败实验保留，由用户定）。

**用户确认模型层能刷出来了；清晰度的来源；清理（2026-09-20）**：用户截图显示报告 `abad19d1…` 的模型层已渲染（生成的笔记本/显示器、推断地面、实体贴图裁块）。清晰度两个来源，量过：①贴图＝视频本身：640×480，中位观察距离1.67 m时一个视频像素盖2.9 mm（近处1.8、远处5.0），再加手持运动模糊；同视角1920宽是1.0 mm、3840宽0.5 mm——这部分只有换更清晰的视频能改善（ARKit高清样例上书脊可读已验证）。②几何与裁切＝我们的方法：融合体素15 mm原生（约24 mm），实体裁块按“观察距离3%的格子”（中位约50 mm）切→边缘锯齿、物体边界处破碎；可改为按掩码逐面判归属（未做）。生成模型清晰是因为它们是生成的。
清理：用户说“失败实验可以删掉”。按规则我不做永久删除，改为移到废纸篓 `~/.Trash/panoptes-phase2-retired-2026-09-20/`（含README，可拖回）：`lingbot-room-rgb-001/002`、`lingbot-cars-001`、`lingbot-walking-rgb-001/002`、`lingbot-walking-replay-001..005`、`lingbot-walking-body-color-001`、`orb-fr1-full-001`、`orb-rgbd-walking-unmasked-001`、`droid-fr3-walking-018`，共5.2 GB；逐个确认过不被 `video-mvp/manifest.json` 任何样例（含沿 scene.json 相对路径和符号链接追到的目录）引用，`lingbot-walking-rgb-003`、`droid-fr1-room-001` 仍被引用所以保留；移走后 walking / cars / room-rgb 三个旧样例实测加载无404。空间要用户清空废纸篓才释放。
其他场景：已启动 fr3/walking_xyz 的全物体流程（`runs/sam2-walking-everything-079`，303个深度视图；预留1美元，上限74.11）；该片段相机几乎不动、几乎看不到地面，没有尺度锚→先只做联动页（对象地图＋起名＋可点选），不做报告。

**第二、第三个场景（2026-09-20）**：用户要求“找新的数据视频”并明确授权下载。调研结论（都未下载，除下面一条）：ARKitScenes raw（真实、1920×1440连续、米制位姿＋LiDAR＋激光扫描＋3D框，非商业，公开直链）；`staerrobotics/warehouses`（HF，Isaac Sim 合成仓库漫游，1080p双目，深度/位姿/实例分割/物体清单OBB真值，CC NC-SA，需登录接受条款，整库123 GB——**最适合量“复现了多少物体”**，等HF令牌修好）；Tanks and Temples（4K真实长视频，仅非商业，Google Drive）；Hilti/ConSLAM（工地，分辨率低或格式重）；Pexels库存（4K但每条11–24秒且多为固定机位，不适合）。
**已下载**（用户授权）：ARKitScenes raw 47333932（Validation，sky_direction=Up）：`wide.zip` 664.6 MB、`lowres_depth.zip` 238.2 MB、`confidence.zip`、`wide_intrinsics.zip`、`lowres_wide.traj`，共0.92 GB，在 `data/arkit-raw-47333932/download/`。`prepare_arkit_clip.py --raw`：位姿文件不说明变换方向→用LiDAR深度在视图间搬运互相印证（world-to-camera 8.2% vs 35%）；位姿按每帧时间戳在相邻样本间插值（最近样本最多差50 ms；只取最近样本的第一版保留在 `runs/arkit-47333932-cameras-080-nearest-pose` 和 `data/.../clip-nearest-pose`）；LiDAR只留置信度2的像素、仅评估。257帧、跨度124秒（帧间隔不均匀，预览按10 fps连续播放）、相机路径22.7 m。`runs/da3-posed-arkit-raw-081`：DA3-GIANT 257视图一次前向137秒；**不拟合尺度**对LiDAR相对误差中位3.7%（3.7 cm，偏远2.7%），一个全局系数后2.95%、94.7%在10%内；`mono_room.py metric` 对米制相机改走 `device_floor`（不用掩码：沿相机平均向下方向找最低密集层，两遍一致平面）：一张平面、内点81%、相机高1.50 m；融合58秒、峰值2.3 GB、支持91.8%（中位45视图）、396304面；照片贴图 `runs/arkit-raw-textured-085`（47张、91.7%面、1280×960，对照图 comparison.jpg：拼花地板/书脊/相框清晰）。联动页样例 `arkit-raw-da3`（实测加载无报错）。全自动分割 `runs/sam2-arkit-raw-everything-084` 进行中（`--work-width 1280`：大帧缩小上传、掩码放回源像素；碎片阈值随帧面积缩放）；之后：对象地图→起名→逐帧点选→报告（米制，无需1.6 m假设）。预留1.5美元，上限75.61。
fr3/walking_xyz：`runs/sam2-walking-everything-079`（303视图，14342掩码）→ `runs/walking-object-map-083`（`--dynamic-masks` 挡掉1084个落在走动人物上的掩码；150实体、105确认；建图9分钟、峰值1.85 GB——`load()` 修复后的稠密规模）→ 起名 `runs/walking-entity-names-086`（59 clear / 31 partial / 15 not_an_object：日光灯×9、通风管×8、吊灯、金属管、椅子、隔断板、窗…）→ `runs/walking-replay-objects-087`、`runs/walking-frame-selection-088`（104实体，303个采样帧全部有轮廓，22068个，中位73个/帧，合并了原有人物分析）、`runs/walking-textured-089`（相机几乎不动：只有11张照片、60%面贴图）。联动页 `walking-da3` 已指向（清单备份 `manifest-before-walking-089.json`），实测加载无报错、0秒处2个人物表面＋75个物体观测。该片段没有地面/尺度锚→不做报告。

**“一个地方”、过期报告、3D模型（2026-09-21，用户：“模型呢？有很多stale的report删掉；视频+点云+3d模型要全部在一个地方”）**：
①一个地方＝报告页。阅读器（工作树 `web/`，`VideoView.tsx`＋`ReportScene.tsx`＋`viewer/native-viewer.ts`）：第一格来源视频（每帧可点选），三维面板的模型层新增“模型叠加点云”（默认开；`representationPass` 在 modelOnly 且 point_cloud===true 时放行点云表示），CAD 在第三格。联动页 8799 只作内部调试，不再当交付物。
②过期报告：平台报告不可变、无删除/归档接口，且我不做永久删除→新建干净库 `panoptes_reports`＋资产库 `.platform/blobs-current`，报告服务（后台任务 b3jm6mijz）指向它，库里只有两份：房间 `http://127.0.0.1:8792/app.html#/reports/d6aacfc7-ad4b-41bd-93dd-56eb6681295a`、高清房间 `http://127.0.0.1:8792/app.html#/reports/464270c9-9d46-4f44-b88a-c5115673e90b`。旧库 `panoptes_video`（约0.4 GB）与 `.platform/blobs`（2.6 GB）原样保留，删除命令已交给用户（`/opt/homebrew/Cellar/postgresql@16/16.15/bin/dropdb -h 127.0.0.1 -p 54329 -U panoptes panoptes_video`、`rm -rf .platform/blobs`）。磁盘现剩12 GiB（废纸篓里5.2 GB、旧库3 GB待用户清）。
③3D模型（高清场景）。第一版按名字：VLM名字→SAM3带词分割→取含实体点最多的实例；10个里只过1个（调试图 `prompted-debug.jpg`：名字常与实体三维位置对不上，如 container 的点在书上），且第10个物体（plastic bag）那次SAM3调用从未提交（事件日志为空）导致批次中断，保留在 `runs/arkit-raw-entity-models-095`（首个椅子试验在 093：全自动分割把椅子拆成木架＋坐垫，且该帧椅背被截断，IoU 0.28）。**现行做法纯几何**（`build_video_entity_model.py`）：`framed_view` 用实体三维范围挑“八个角点都在画面内、≥60%无遮挡、投影最大”的帧；`sam2_everything.box_masks` 用实体三维点在该帧的投影框提示 SAM 2.1（一次GPU调用处理全部），掩码须含≥50%实体点且不碰画面边缘；读入时闭运算补针孔；然后每实体一次生成器调用＋源视角一致性检查。`runs/arkit-raw-entity-models-099`：60候选→18个有完整掩码（18个无拍全的帧、10个掩码碰边、14个掩码与实体点不符）→**14个通过**（poster、plastic bag、calendar、planter、plastic container、basket、seat cushion、photograph、cup、chair×2（部件级实体）、bag、shopping bag、cardboard box），4个未过（folder 0.62、books 0.29、bag 0.49、white plastic bag 0.51）。注意：seat cushion 被生成成带腿小凳——源视角轮廓一致，其余是生成器的推测。费用18次≈3.4美元（预留5，上限80.61）。`light_model()`：展示副本减到4万三角面（约75万面/15 MB→0.8 MB，最大偏差0.25 mm），校验过的原文件不动；联动页与导入器都用它。VLM判为地面/墙/天花板碎片的实体改为背景标签（`name_video_entities.py --names` 可不再付费重套名字→`runs/arkit-raw-entity-names-098`），推断地面只挂在观测最多的那块 floor 上。
高清场景产物：`runs/arkit-raw-inferred-floor-096`（13.0 m²，32%被看到）、`runs/arkit-raw-replay-objects-100`（198对象、15个带模型含地面）、`runs/arkit-raw-frame-selection-101`（174实体、253/257帧有轮廓、5222个）、`runs/arkit-raw-textured-102`；报告导入8分钟、峰值3.4 GB：199实体/198确认/1413观测/252关键帧/2576资产。实测（1500×1000视口，读页面状态；面板隐藏无法截图）：视频1920×1440可播放；12 s处30个可点选轮廓，点 plastic bag→选中；列表选 seat cushion→视频内高亮；模型层＋点云载完：199个资产请求、65 MB、“模型已加载”16处、JS堆117 MB，无失败请求。

**“模型是为了证明理解了物体？这个 tray 明显没理解清楚”（2026-09-21，用户截图：第四格 tray 只剩几片碎面）**：
①量出来的事实：对象地图里 tray（object-224）是清楚的——9个视图、27909个三维点、99.8%跨视图一致、33×45×4.6 cm；碎的是**展示**：第四格/模型层显示的是“实体格子从融合网格里切出来的一块”，而融合网格在 tray 处几乎是空的（切片面积只有它最大单视图表面的8%）。逐阶段查（草稿脚本，未入库）：跨视图支持保留了 tray 97%的像素；**删掉它的是自由空间刻除**（97%顶点因“被看穿”删除：17个视图同意、24个视图判为看穿）。原因：同一个 tray，1.0–1.3 m 外的视图（206–218）给的DA3深度比 0.55–0.7 m 的近视图（229–245，彼此一致到±1–3 cm）**系统性远5–7 cm**（约5%，刚过4%容差），于是远视图“看穿”了近视图建出的表面。把容差放到TSDF截断带（6 cm）**不解决**（仍删92%，已试，假设被否）——这是深度模型随距离的偏差，不是噪声。对LiDAR（仅评估）：tray 处DA3偏近5–8%，比全片中位3.7%差，LiDAR在薄亮塑料上自身也不稳，不作结论。门（object-129/148）是另一种：暗、远（2.5–3 m）、0%跨视图支持，融合拒收是对的。
②影响面：174个有格子的非背景确认实体里，融合切片面积 < 最大单视图表面一半的有29个（<¼的18个：门×6、tray、床垫、塑料袋、鞋套、雨伞、干花…），中位数是1.34（多数实体融合切片比任何单视图都完整）。
③修法（展示层，`import_video_scene.py`，`FUSED_SHARE=.5`）：融合切片不到最佳单视图表面一半时，改用该单视图表面作为实体的观测参考面（全自动实体的平面轮廓也改从它取）；导入来源注记里记 `entities_shown_by_fused_cut / entities_shown_by_best_single_view`。**没有修的根因**：DA3跨距离的深度偏差（候选：每视图对多视图共识做仿射校正后再融合；未做）。规则与距离一直只用实体的跨视图一致点（`measuredOn`），生成模型从不进测量——所以模型层是“展示/佐证”，不是理解的前提；理解的证据＝视频轮廓↔实体自己的三维点↔名字↔尺寸，四者互相能点到。RecGen 生成模型建议停止追加（每个0.19美元，背面是猜的）。
④动静掩码调研（子代理，带出处；费用均为估计）：**VGGT-Ω 存在**（CVPR 2026，arXiv 2605.15195，`facebookresearch/vggt-omega`）但**不输出动态掩码**，且 FAIR 非商用许可；**Easi3R** 免训练、从 DUSt3R/MonST3R 注意力图取动态掩码，但代码与权重均 CC BY-NC-SA、65帧要23–33 GB、0.31 FPS；**VGGT4D** 同思路用VGGT注意力（MIT代码；默认权重非商用，`VGGT-1B-Commercial` 可商用但与该方法是否兼容未验证）；RoMo（Apache-2.0）＝光流＋对极残差＋SAM2特征。DAVIS-16 上：裸残差阈值40.9 < 注意力掩码57.7–62.1，但**残差＋SAM2 精修 64.3–77.3 ≥ Easi3R＋SAM2 的70.7**。结论：不引入第二套几何模型；用已有位姿做**对极残差**（SEA-RAFT 光流，BSD-3；对极残差不依赖深度，正好避开上面①的深度偏差）→连通块→现成的 `sam2_everything.box_masks` 框提示→SAM2 全片传播；估计每段0.05–0.10美元、L4 上3–6分钟。已知失效：慢速物体（拉大帧间隔）、停下的物体（动过即记为动态并传播）、沿对极线方向运动、反光/阴影。现状提醒：融合的跨视图支持＋刻除已经在**隐式**丢掉动过的东西，缺的是“什么动了”的显式实例掩码。walking 片段有SAM3人物掩码可当参照量IoU，是现成的验证场。未实现，等用户点头。
⑤清理：当前清单＋两份报告＋符号链接的依赖闭包算过（58个运行目录被引用）。被后一版完全取代、无人引用的中间产物列在 `ART/retire-superseded-2026-09-21.sh`（约3.4 GB：`video-report-blender-037` 1.3 GB、room-object-map-047/054、旧 replay/frame-selection/textured 各版、arkit-raw-entity-models-093/094/095、`clip-nearest-pose` 715 MB（图像与 `clip/` 逐字节相同）、fr1 tgz 747 MB）＋旧资产库 2.6 GB，脚本只是移到废纸篓的带日期目录。**我执行移动时被权限策略拒绝，未绕过，脚本交给用户自己跑**；旧库 `panoptes_video` 的 dropdb 与清空废纸篓同样由用户执行。早期 LingBot/ORB 对照仍被旧对照页引用，未列入。

**同日后续（用户：“3.1 我要你直接做”“3d模型呢？动静呢？清理了吗？”）**：
①SAM3.1：`modal run modal_apps/sam3_video.py::access` 仍是 403（facebook/sam3 与 sam3.1 都是 GatedRepo：Modal 密钥里的令牌所属账号不在授权名单）。用户又在聊天里贴了令牌；我不经手任何密钥（硬规则，用户授权也不行），改为在用户的终端面板里开好输入提示（`read -s … modal secret create huggingface HF_TOKEN="$T" --force`，标签页 “HF token -> Modal secret”），用户自己粘贴一次即可；贴过的令牌应作废重建（建议 Read 类型；细粒度令牌要勾选“可访问的公开受限仓库的读取”）。通过后：先重跑 access，再用 SAM3.1 视频跟踪做下面②的传播，并与 SAM2.1 同框对比。
②动静掩码第一版 `modal_apps/motion_masks.py`（有自检）：**不用深度**的对极线段残差——静止点在第二帧只能落在“无穷远投影→最近可能深度投影”这条线段上，光流终点离线段的像素距离就是相机解释不了的运动（纯旋转时线段退化成点，也成立）；RAFT-large（torchvision 预训练，BSD-3）双向光流、前后向一致性过滤；残差>max(2 px, 中位+5 MAD) 的连通块（≥0.3%画面）→ `sam2_everything.box_masks` 框提示，掩码里≥30%在动才收。`runs/walking-motion-masks-103`（fr3/walking，86对、间隔8帧、两次L4调用各约50秒，预留0.5美元，上限81.11）：对照SAM3人物掩码（只作度量，不是输入）**像素精度97.8%、召回39%、逐帧IoU中位0.30（四分位0.11–0.52）**；无人的9帧里“在动”只占画面0.5%。对照图 `sheet.jpg`：残差热图很干净（静止背景全黑）；漏掉的三类与预期一致——站着的人只有手脚在动→框提示只给出手臂/腿（部件级掩码）；慢速的人残差低于阈值；坐着不动的人此刻确实没动。结论：线索可用、几乎无误报，**缺的是时间传播**（动过即为动态，整段跟踪）：取“掩码大部分在动且面积大”的帧作种子（如 frame 600 整个人 IoU≈0.8）→ SAM2.1 视频传播（现有 `scripts/run_seeded_video.py` 是本地 mps、吃SAM3种子格式，需适配）或 SAM3.1 视频（`sam3_video.py::run_native` 的 add_prompt）。注意：019 的DROID位姿本身是在有人走动的视频上估的。
③报告：新建库 `panoptes_reports_0921`（资产库仍是 `.platform/blobs-current`，内容寻址、不重复占盘），只有两份：高清房间 `http://127.0.0.1:8792/app.html#/reports/924da280-4356-42ce-91e3-3db55b1dec5d`（199实体；145个用融合切片、34个改用最佳单视图；tray 的参考面现为 45×55 cm 的单视图表面，带平面轮廓——**数据层核对，浏览器面板隐藏无法目检**）、房间 `http://127.0.0.1:8792/app.html#/reports/c57f46a0-7f2a-4062-be0a-ac78d0663c39`（315实体/291确认，这次带轻量展示模型；导入11.6分钟、峰值2.3 GB）。报告服务后台任务 bfr9lyjnv 指向新库。`panoptes_reports`（上一版，含被取代的高清报告 464270c9 与一次中止的导入）和 `panoptes_video` 都不再服务，dropdb 由用户执行。
④清理：移到废纸篓被权限策略拒绝后，改为**项目内挪位**（不删除、可逆）：`ART/_retired-2026-09-21/`（6.4 GB：runs 25个目录、`clip-nearest-pose`、fr1 tgz、旧库资产库）。移动后无断链，联动页清单 200。释放磁盘需用户自己 `rm -rf` 该目录并清空废纸篓里的 `panoptes-phase2-retired-2026-09-20`（5 GB）。磁盘现剩14 GiB。

**自部署 SAM 3.1 打通，动静掩码闭环（2026-09-21 晚，用户自己在终端把令牌放进 Modal 密钥；我不经手密钥）**：
①`sam3_video.py::access`：facebook/sam3 与 facebook/sam3.1 均 302/accessible。权重 `sam3.1_multiplex.pt` 已进 Modal 卷 `sam3-hf-cache`（首次下载约4分钟，之后模型加载<1分钟）。
②固定提交 `660a5e9` 的上游缺陷：`Sam3BasePredictor.start_session` 总是把 `offload_state_to_cpu` 传给 `init_state`，而多路复用模型的 `init_state` 不接受（推理前就 TypeError）——`run_native` 此前因403从未真正跑过，所以没暴露。修在共用处：`sam3_video.start_session(predictor, path)`（照上游写法登记会话，去掉该参数），`run_native` 与新脚本都用它；`tests/test_sam3_video_contract.py` 3项通过。另一处接口事实：只有实例点击、没有文字提示的会话，`propagate_in_video` 必须显式给 `start_frame_index`，否则报 “No prompts are received on any frames”。
③`modal_apps/sam3_motion_tracks.py`（有自检）：`motion_masks.py` 的“在动”区域作种子（连通块≥2%画面，距离变换取3个内部点击点，面积大的先）→ SAM 3.1 实例跟踪（points＋obj_id，即其 SAM2 式提示）双向传播；已被现有轨迹覆盖≥50%的种子跳过，所以一个人不会变成十个对象；同一次调用可另开会话跑文字提示作对照；每阶段各自报错、互不拖累。
④结果（fr3/walking 源帧500–800，640×480，A100-40GB；对照＝原有人物掩码（fal 的 SAM3 周期检测＋SAM2.1 传播），只作度量）：
- `runs/walking-sam31-motion-tracks-104`：失败（②的缺陷，推理前），保留。
- `runs/walking-sam31-motion-tracks-105`：**文字 “person”**：300帧49秒，287帧有掩码，IoU中位0.975、召回98.0%、精度99.3%——自部署 3.1 一趟就复现了原来 fal-SAM3＋SAM2.1 两段式的人物掩码；运动种子阶段因③的 start_frame 问题报错。
- `runs/walking-sam31-motion-tracks-106`：**不用任何文字，只用运动种子**：34个候选种子→3个对象（31个被识别为已在跟踪），295/300帧有掩码，**IoU中位0.975、召回98.2%、精度99.3%**（单帧对运动线索＋SAM2.1框：召回39%、IoU中位0.30）。对照图 `sheet.jpg`：坐着的人因为在片段里动过，被整段跟踪——“动过即动态”。
- 费用：三次A100调用共约536秒≈0.3美元（预留1.5，上限82.61）。
⑤没有验证的：这段视频里会动的只有人，所以“类无关”只在方法上成立（全程没用词），**叉车/推车/门这类非人物体还没测过**；种子来自 019 的DROID位姿，而该位姿本身是在有人走动的视频上估的；窗口是300帧，整段859帧与更长视频的显存/时间未测；3.1 与 2.1 在同一批框上的分割质量对比还没做。
⑥接下来可替换的环节：人物/词表对象的 fal-SAM3 调用→自部署 3.1（许可条款待逐条核）；`--dynamic-masks` 的来源从“person 提示”换成运动轨迹，融合与对象地图直接受益。

**动态掩码来源切换、fal 退出（2026-09-21 深夜，用户 “ok” 后按既定顺序做）**：
①整段 walking（859帧）运动轨迹：两个重叠窗口 `runs/walking-sam31-motion-tracks-107`（0–480）、`108`（380–859），各一次A100调用（504 s / 382 s）。107 暴露了预料中的失效：第290帧图像左缘一个假运动块（`motion_masks` 单帧误报）被当种子，SAM 3.1 就老老实实跟踪了它下面的地板 479 帧（精度 62%）。**轨迹级检查**（`MOVING_SHARE=.25`，有自检）：一条轨迹在运动地图采样帧上的“在动像素占比”中位数——人 0.79–0.99、那块地板 0.005——低于 25% 不算动态。`run()` 现在遇到已有 `tracks.npz` 就只重算不再付费；107 重算后精度 62%→99.5%、IoU 0.575→0.969，其他窗口不变。上限：一个在片段里大半时间静止的物体会被判为静态——它那段时间本来就是静态几何；逐帧动/静状态是升级路径。
②`scripts/assemble_dynamic_masks.py`（有自检）：窗口并集→逆栅格化回源像素→`SOURCEINDEX-0.png`，即 `mono_room --dynamic-masks` 和 `build_video_object_map --dynamic-masks` 一直在读的布局。`runs/walking-motion-dynamic-masks-109`：708帧有掩码；对照原人物掩码（750帧）IoU中位0.974、召回98.8%、精度98.3%、无人帧零误报。
③用它重新融合 `runs/da3-posed-walking-110-motion-masks`（mono/支持文件符号链接自020，同参数，CPU 3分钟、峰值2.0 GB）：剔除252视图（020：266）、248697面（020：248625）；传感器深度评估静态像素中位1.35%/90.6%在10%内、移动像素4.85%——与020逐项相同。**动态像素的来源从此不需要任何文字提示**。
④自部署 SAM3 图像服务 `modal_apps/sam3_app.py`（transformers `facebook/sam3`，L4，早已写好但因403从未跑通）：修了一个 bfloat16→numpy 的崩溃；“basket” 在高清第216帧一次命中（分数0.918，掩码正对藤篮，目检）。`discover_video_keyframes.py --provider self-hosted`：同一目录布局、同样的 `provider-output.json` 形状（fal 路径保留为默认），`runs/sam3-selfhosted-arkit-raw-basket-111` 两帧验证通过。下一次新场景的人物/词表分割可以整段不走 fal。
⑤关于“SAM 3.1 对 SAM 2.1 框提示谁更好”：读源码后这个对比不成立——3.1 的点击/实例跟踪路径（`add_sam2_new_points`）本身就是内置的 SAM2 式跟踪器，文字/框在 3.1 里是“视觉示例”提示（找所有相似实例），语义不同。所以：实例掩码两者同源；3.1 多出来的是文字与示例提示和多目标复用。
⑥费用：SAM 3.1 五次调用约0.85美元（预留1.5）；SAM3 图像服务预留0.2；上限 **82.81/100**。本地提交，未推送。

**对照（CPU）**：`runs/da3-posed-walking-112-no-masks`＝同参数但不给任何动态掩码（250108面）。从真实相机位姿光线投射的对照图 `runs/da3-posed-walking-110-motion-masks/before-after.jpg`：第30帧右侧站立者的残影被融进墙面，110（运动轨迹掩码）里消失；第600/750帧两者几乎相同——跨视图支持与刻除本来就在隐式删掉走动的人，显式掩码补的是“站得久的人”。上一会话结束时两个本地服务一起退出，已重启（8792 报告、8799 联动页）。

下一步（用户已确认的顺序）：G1尺度→G2对象级地图（ConceptGraphs式：关键帧SAM3分割→用带位姿深度抬进地图→三维重叠+外观合并为实体；复用 `platform/identity.py`、`spatial.py::associate_observations`）→G3对象建模（选帧、A07模型—源帧一致性、尺寸对点云范围）→G4接入现有报告→G5时间规则→G7地图持久化；G6真实工厂素材暂缓。地面偏差的候选修法：对已分割为地面的像素做跨视图单平面约束（现有 `planar_surfaces.py`），只用于语义上确认为平面的区域，不补造未见区域。

仍未解决：相机从未看到的区域为空（不补造，用户已确认可接受）；外围漂浮碎片（6418分量，最大分量占88%）；源帧750之前没有人物掩码；DA3-GIANT许可。下一步：用户目检签收后再决定是否替换 `room-droid` 主样本；用DA3-BASE重跑352视图确认可商用路线；补750帧之前的人物掩码；再接回对象/骨架。不要重跑007/009/010/012。

### 其他失败对照

- `droid-mapanything-probe-001`：8个真实DROID相机/K作为MapAnything条件，35.18秒返回，但模型重预测K/相机，焦距与重投影明显偏离。已拒绝当固定相机深度融合；不要未经合同验证强行拼接。
- RGB-D 2cm体素试验曾增加面数/碎片，误差尾部变差，保留4cm默认。更密或删碎片不是质量证明。
- 原生/显示检查证明数据来源和转换正确，不证明预测深度真实准确；约28%的支持率也不是准确率。

## 可直接复用的代码与检查

统一算法图：[docs/algorithms/README.md](../algorithms/README.md)；照片A系列与视频V01–V18均保留。主技术路线：[TECHNICAL-ROADMAP.md](TECHNICAL-ROADMAP.md)；实跑证据：[VIDEO-MVP.md](VIDEO-MVP.md)（包含历史段，较早的“待实现”不能覆盖页首和后续真实结果）。

| 环节 | 入口 |
| --- | --- |
| 同源抽帧/来源/时间 | `reconstruct_room_rgb.py`、`video_motion.py` 等现有 helpers |
| 周期发现/传播 | `discover_video_keyframes.py`、`run_discovered_video.py`、`run_seeded_video.py` |
| 2D骨架/重入候选 | `build_video_pose_preview.py`、`person_reid.py`、`link_person_tracklets.py` |
| 相机/持久地图 | `phase2_camera_build.py`、`phase2_camera_run.py`、`phase2_camera_export.py` |
| 房间原生几何 | `build_droid_replay.py`、`build_lingbot_replay.py`、`export_lingbot_native_preview.py` |
| 人体/对象建模与颜色 | `build_video_body_models.py`、`attach_lingbot_objects.py`、`color_video_bodies.py`、`build_lingbot_object_model.py` |
| 照片语义复用 | `build_video_object_models.py`、`review_video_object_semantics.py`、既有 `platform/recgen.py` |
| 浏览器 | `web/experiments/video-mvp/{app.js,scene.ts,timeline.mjs}`、共享 `web/src/viewer/native-viewer.ts` |

脚本表路径默认 `scripts/`。不要把六份单帧静态对象观测称作六个跨视角融合实体。不要把人体颜色修正说成人体几何覆盖已修好。

最近代码提交：`a776fe3` 分辨率受控对照/云端导出；`04a31d2` LingBot官方模式；`190f78b` DROID支持彩色点显示；`850d03d` 同进程未变文件哈希复用；`37de0c8` 磁盘预留。旧照片阶段归档见 [PHASE1-ARCHIVE-20260917.md](../platform/PHASE1-ARCHIVE-20260917.md)，不要重置原工作树。

上一轮已通过：DROID分辨率/标定检查、compact collector资源与内容检查、DROID原生后端patch自检、DROID过滤/变换导出自检、云端每个显示点XYZ/RGB来源检查；浏览器新旧云/网格目检仍失败。快速检查命令（无GPU提交）：

```sh
cd /Users/adam/.codex/worktrees/panoptes-phase2-video
phase2_python=/Users/adam/Desktop/panoptes-public/panoptes-serving/.venv/bin/python
"$phase2_python" tests/check_droid_resolution.py
"$phase2_python" tests/check_lingbot_compact_collection.py
"$phase2_python" modal_apps/droid_room.py self-check
"$phase2_python" scripts/build_droid_replay.py --self-check
node web/experiments/video-mvp/check.mjs
node web/experiments/video-mvp/scene-check.ts
```

真实 `tests/check_droid_display.py SCENE` 需要完整同源NPZ/PLY/支持文件；004已在云端跑过。本机只有compact preview时不能直接运行后又因文件缺少重复下载多GB。新代码改动后按影响范围跑检查，避免重算全部旧hash。

## 下一 session 的具体工作顺序

1. 按上方恢复命令建立正确checkout与预览，读取真实manifest、账本和004 review。先目检用户实际指出的差点云；不要一上来重跑模型。
2. **当前优先项：可用、连续的房间几何。** 排查固定可靠相机条件下的稠密多视图深度/融合路径，先做覆盖同一真实房间区域的小实验。这是待验证方向，不是已接通的模块。MapAnything相机条件探针已失败，不能直接重用其错位输出。
3. 为新的假设明确输入、相机/K/深度域、静动态处理及失败标准；先检查已有实现/官方接口，再选最短实现。不再同时盲跑多个分辨率/门限组合。
4. 新方法要与已有DROID/LingBot同区域对照，比较源视角一致性、连续墙面/桌面/家具覆盖、重影/漂浮/孔洞；完整几何与点云实际旋转检查。通过来源测试、轨迹ATE或点数增加不能替代目检。普通RGB的GT/传感器深度始终只允许用于独立评估。
5. 地图改善后，在同一相机/时间/来源坐标中接回已有mask、骨架、人体/车与对象模型；禁止挪用RGB-D的米制XYZ。持续身份与地图保存/重定位仍要独立验证。
6. 每次实验新目录、失败保留、预算登记、GPU有上限；回放只拿compact产物。本地生成完整archive/多份权重之前检查空间。仅对新增/变更产物更新delivery hash。
7. 任意用户新视频上传、普通手机完整房间、跨时间实体记忆、全片人体连续覆盖、车辆完整模型以及CAD/EHS时间状态服务均未验收；不要因这一轮解决某个局部问题顺带宣布它们完成。

没有等待用户回答的问题，也没有必须重新批准的既有预算。HF gated访问仅阻挡相应SAM3.1路线，不应让房间几何工作空等。新的session可以继续上述工作，但本次交接本身没有启动收费任务。

## 新 session 可直接使用的指令

> 接手 Panoptes Phase 2。先阅读 `/Users/adam/.codex/worktrees/panoptes-phase2-video/docs/phase2/HANDOFF.md` 和 `.planning/HANDOFF.json`，在 `codex/phase2-video` 继续。房间点云仍未达标；不要重复已记录的失败实验或只修展示。保留全部已有SAM/VLM/模型能力，在累计100美元授权内继续推进可靠的RGB场景几何，再接回动态对象、骨架和模型。先恢复现有预览核实实际状态，优先复用缓存与现有算法，不训练新模型，不占满本地磁盘。
