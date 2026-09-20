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
