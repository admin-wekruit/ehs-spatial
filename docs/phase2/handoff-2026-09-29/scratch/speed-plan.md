# 30 秒视频 → 2–3 分钟出报告：两块 A100 的方案

**标注：** [P] 是我们自己实测的（Panoptes），[S] 是来源实测的，都注明硬件；[C] 是厂商或作者的说法；[E] 是我的估计。
**计时：** t=0 表示视频已经传进 GPU 容器。上传 50 MB 大约 8–20 s [E]，这段时间可以边传边解码。
**前提：** 两块 A100-80GB 放在同一个常驻 Modal 容器里。
**范围：** 这次只读了代码和资料，没有跑 GPU。代码行号都相对于 `/Users/adam/.codex/worktrees/panoptes-phase2-video`。

## 0. 结论
- 能做到，但靠的不是给现有流程提速，而是另建一条快速层：一个常驻的双卡容器，处理约 150 个关键帧，只用前馈模型，不训练，物体不逐个重试。现有流程保留，改成并行后作为精修层，在后台往同一份报告里补图层。
- 预计时间（都是 [E]）：
  - 约 40 s：相机轨迹、房间网格、点云、人的轨迹。
  - 约 60 s：带名字的物体。
  - 约 180 s：5–12 个完整模型；快速泼溅在 170–220 s 之间到。
  - 冷启动再多 60–120 s。
- 慢的主要原因在编排，不在模型本身。
- LingBot 不放进 3 分钟的路径里，挪到后台，大约 3–6 分钟后补上 [E]。
- RecGen 退出快速层。

## 1. 为什么慢
| 原因 | 证据 | 代价 |
|---|---|---|
| 执行器串行 | `store.py:619-620` 规定付费阶段一次只跑一个；`stages.py:508` 把所有 modal/cloud 阶段都算作付费，所以 GPU 阶段从来不会同时跑。整张图还按决策分轮执行（`store.py:748-758`），每一轮都是一道屏障 | 2–3 h 基本就是各阶段时间相加 [P] |
| 每个阶段只有 1 个容器 | `max_containers=1`：`sam2_everything.py:47`、`mono_room.py:98`、`sam3_motion_tracks.py:38`、`sam3d_research.py:77`、`lingbot_room.py:166`、`splat_train.py:471`、`droid_room.py:96`、`video_events.py:40` | SAM 2 按 48 帧一块调 `.map`，实际还是一块接一块；SAM 3D 的 4 个 worker 在 1 张卡上排队 |
| 每次都冷启动 | `min_containers=0`、`scaledown_window=2`（`droid_room.py:97`、`lingbot_room.py:167`、`splat_train.py:471`） | DA3 posed 处理 155 个视角墙钟 72 s [P]，其中前向计算约 10–15 s [E] |
| 帧太多 | SAM 2 "segment everything" 每帧 32×32 个提示点（`sam2_everything.py:27`），共约 290 帧；SAM 3.1 人跟踪没传 `--stride`，默认逐帧 30 fps（`stages.py:643`、`sam3_motion_tracks.py:308`） | SAM 2 12 min，人 14 min [P] |
| 重 CPU 阶段跑在 Mac 上 | 没写 `compute=` 的阶段默认在本机跑：fuse（Open3D TSDF）、texture、fill（runner 自估 1800 s）、object_map、outlines、import。每次调用 Modal 前还要在 Mac 上把帧编成 PNG 再上传 | object_map 12 min，import 13 min [P] |
| 物体一个一个生成 | SAM 3D 每段 146–157 次调用，每次 GPU 时间 10–13 s [P]；每个物体最多换 4 个视角再加 1 个种子，依次尝试（`complete_video_objects.py:66`）；没有开蒸馏快速模式（`sam3d_research.py:112`）；RecGen 被强制单并发（`stages.py:43`） | SAM 3D 38 min，RecGen 1–3 h [P] |
| 泼溅靠训练 | 60k 步、上限 2.5M、带位姿精修，跑在 H100 上（`stages.py:724`） | 20–25 min [P] |
| 命名是顺序的 | 调 Gemini，每个请求 10 个物体，一个接一个（`stages.py:692`） | 6 min [P] |

**你问的三个问题：**
- **SAM 2 不在本地跑。** 它已经在 Modal 的 L4 上。慢是因为只有 1 个容器、逐帧处理、每帧 1024 个提示点，约 2.5 s/帧 [P]。真正在 Mac 上跑的是融合、物体图、贴图、补洞和导入。
- **RecGen 是什么。** 它是 TRI 的 "Reconstruction by Generation"，基于 TRELLIS：输入 RGB-D、掩码和内参，输出带纹理的网格和 6-DoF 位姿。
  - 在我们流程里，它是 SAM 3D 失败之后的第二轮生成器。
  - 许可是非商用，commercial 档已经把它排除了（`profiles.py`）。
  - 不用 RecGen 时，三份交付报告的模型数从 22/67/40 降到 14/62/33 [P, `docs/phase2/STREAMING-PLAN.md` §11]。这两组数都包含人工批准的盒子。
- **完整模型其实在用。** research 档依次跑 sam3d → recgen → box → merge，结果作为 `generated_mesh` 导入（`import_video_scene.py:384-405`）。只要有一个物体带模型，查看器默认就显示"模型"图层（`ReportScene.tsx:234`）。
  - 看不到很多模型的原因：每段只有 12–41 个 SAM 3D 模型通过源视角闸门 [P]；盒子必须经人工复核批准才显示（D13）。
  - 线上页面我没有打开核对。

## 2. 别人怎么做
- 大家都分两层：先给一个快速结果，完整版稍后再到。
  - Scaniverse 在手机上约 1 分钟出结果 [C]。
  - Polycam 和 Luma 在云端要 10–45 分钟 [C]。
  - OpenSpace 平均 15 分钟 [C]。
  - Matterport 要 30 分钟到 48 小时 [C]。
- 调研的产品里，没有一家把逐物体生成模型或训练式泼溅放在快速路径上。
- NVIDIA VSS 的做法是把视频切块、分发到多卡并行，再合并结果。1 分钟视频在 1×H100 上 12.1 s，2×2 布局 5.70 s [S, H100]。
- 物体地图方面的论文大约每秒取 3 个关键帧，用检测器加 SAM。其中最快的 Open-YOLO 3D 每个场景 21.8 s [S, A100-40GB]。

## 3. 快速流水线（热启动）
**部署方式：**
- 一个常驻的 Modal 类，`gpu="A100-80GB:2"`，`min_containers=1`。
- 每个模型族一个常驻进程，各用自己的 venv。GPU 0 叫 A，GPU 1 叫 B。
- 帧只在容器内存里传，不回 Mac。

| t (s) | GPU A | GPU B | CPU |
|---|---|---|---|
| 0–4 | – | – | 解码 900 帧；检测切点；选 150 个几何关键帧（每 6 帧一个，在前后 2 帧内挑最清晰的）和 60 个物体关键帧 |
| 4–22 | DA3-GIANT any-view，150 帧，全 16:9 画幅 504×280：输出位姿、内参、深度、置信度 [E 10–18 s；DA3 论文里 32 张图 37.6 FPS [S, A100]] | 4–26：SAM 3 找 {person, floor}，150 帧 [E；由 L4 上 2 类 323 ms/帧 [P] 推算] | – |
| 22–34 | 拿到 B 的人掩码后：去掉人，做 GPU TSDF（3 cm）和稠密点 [E 5–12 s] | 26–48：SAM 3 用约 20 个 EHS 词，处理物体关键帧 1–30 [E 0.3–0.7 s/帧] | 用地面加 1.6 m 相机高度定尺度（现有规则）；人的脚点投到地面，在 3D 里连成轨迹 |
| 34–50 | SAM 3 同一词表，处理物体关键帧 31–60 | （续） | – |
| 50–58 | 把掩码抬到 3D，按体素重叠合并物体，拟合贴地的包围框 [E 3–8 s] | 48–85：Qwen3-VL-8B（vLLM）一次批处理 3 个 12 s 事件窗口 [E] | 名字先用检测词；Gemini 请求全部并发发出做精修 [E 10–20 s] |
| 58–175 | SAM 3D 快速模式，2 个进程 [E 有效 3–5 s/次] | 85–170：gsplat 快速泼溅 [E 60–85 s] | 每个模型出来就过现有的源视角闸门 |

**各时间点能看到什么**（导入在容器里做，每次只加新图层）：
- **约 40 s**：相机轨迹、顶点色房间网格、点云、人（5 fps）、尺度标为"待核"。
- **约 60 s**：加上物体，以框加名字的形式显示。
- **约 120 s**：加上事件、精修后的名字、前 2–5 个完整模型。
- **约 180 s**：完整模型共 5–12 个；泼溅在 170–220 s 之间到。

**后台继续补：**
- 其余物体的 SAM 3D：5–10 min [E]
- SAM 3.1 的 30 fps 人轨迹：1–2 min [E]
- 贴图和补洞：5–15 min [E]
- LingBot：3–6 min [E]
- 完整泼溅：20–25 min [P]
- MoGe 镜头闸门
- RecGen 只在 research 档跑

后台不替换位姿，所以一份报告始终只有一个坐标系。

**冷启动：** 容器启动、加载约 35–40 GB 权重、CUDA 预热，一共多 60–120 s [E]。

**备选方案 G2：** 如果实验 E1 说明 DA3 的位姿不行，就在 B 上 4–20 s 跑现有的 DROID（450 帧；依据是 DROID 在 RTX 3090 上 40 FPS [S]），A 在 20–35 s 跑 DA3 posed。之后各步整体推迟 15–25 s，仍能在 180 s 内完成 [E]。

## 4. 比现在的完整流程丢了什么
| 项 | 现在 | 3 分钟版 | 差别 |
|---|---|---|---|
| 相机 | DROID 标定加全局 BA | DA3 前馈，没有 BA | TUM 数据集上，无标定的前馈方法（DA3-Streaming）ATE 0.087 m，标定过的 DROID 是 0.038 m [S]。可能出现双层墙和贴图接缝 [E] |
| 房间 | TSDF 加 60 MP 贴图加补洞 | 3 cm TSDF，顶点色 | 墙面发糊，洞没补，要等后台 |
| 物体 | 约 290 帧 segment-everything | 60 帧，约 20 个词 | 词表以外的物体、小物体、只出现在几帧里的物体会漏；管子和线缆这类细长物体用框表达不好 [E] |
| 人 | 30 fps 跟踪 | 5 fps 检测加 3D 关联 | 轨迹更粗，两人交错时可能换号。不过我们的规则按 2 s 基线算速度，1 Hz 和 5 Hz 得出的结论一致 [P, M0] |
| 模型 | 每段 12–41 个 [P] | 5–12 个 [E] | 其余几分钟后到 |
| 泼溅 | 60k 步，2.5M | 5k 步，0.5M | 公开数据上 7k 步比 30k 步低约 1.7 dB（27.21 对 28.95）[S, TITAN RTX] |
| 尺度 | 地面规则加镜头闸门 | 只有地面规则 | 镜头闸门跑完之前，报告不写米制结论 |
| 画幅 | ME340 上只用中间 960×720（`lingbot_dense_map.py:36`） | 全画幅 1280×720 | 快速版反而宽 33% |

## 5. 不用 LingBot，差在哪
LingBot 在今天的报告里既不提供相机轨迹，也不提供尺度：它的相机用 Sim3 对齐到 DROID，尺度来自地面规则。所以去掉它，漂移和尺度都不受影响。它提供的是下面三样：
1. **稠密点层。** 约 390–450 帧（步长 = ceil(帧数/450)，`decide.py:35`），全画幅 518 px，格子从 7.5 mm 起，最多 400 万点。
   - 替代：用 DA3 的 150 帧深度，套用同一套"每个表面只留一层点"的构建方法。
   - 代价：视角少了约 3 倍。只在快速横扫时拍到的表面、细杆、物体边缘，点会变少 [E]。具体少多少，要靠 E1 用留出视角来量。
2. **独立的交叉检验。** dense_gate 要求至少 45% 的点落在融合网格 25 cm 以内（`decide.py:309-311`）。换成 DA3 自己的点去检验 DA3 融合出来的网格，就成了自己验自己，这项检验失效。
3. **推测地面。** 地面必须经过稠密图验证才会导入（`decide.py:332-340`）。没有 LingBot，地面层就一直空着，直到后台 LingBot 跑完。

建议：LingBot 放到后台。A100 上 450 帧大约 63 s [E，按 7.1 fps [P] 推算]，但它的构建步骤要移植到 GPU 上，现在在 CPU 上要 9–27 min [P]。它的许可还没有书面确认，商用前不能用。

## 6. 完整模型：前 3 分钟有几个
- **用哪个生成器：** SAM 3D Objects。
  - 打开 `use_stage1_distillation`，stage1 跑 4 步，stage2 约 12 步，只出网格。
  - 每张卡跑 2 个进程；80 GB 能否同时放下两个，由 E4 来测。
  - 输入传裁剪块，不传整帧点图。
- **算账：** A 卡在 58–175 s 之间约有 117 s 可用，能做 23–39 次调用 [E]。
  - 挑 12–20 个 EHS 优先的物体，每个物体 2 个视角并行跑，用现有闸门留下更好的那个。
  - 预计通过 5–12 个 [E]。这是按今天每个物体 23–52% 的通过率 [P] 推算的。
- 其他物体先显示为观测范围的框，并标明"这不是模型"。
- 不上快速路径的：RecGen、SF3D、TripoSR（后两个在杂乱场景里质量差）。

## 7. 许可（面向以后商用）
| 模型 | 许可 | 能否商用 |
|---|---|---|
| DA3-GIANT-1.1 | CC BY-NC 4.0 | 不能，只用于演示。商用可换 DA3-BASE（Apache-2.0，位姿头这种用法还要再核）、MapAnything-apache（Apache-2.0）或 VGGT-1B-Commercial（需要申请） |
| SAM 3 / 3.1、SAM 3D Objects | SAM License | 能，但禁止军事、核工业和 ITAR 用途。EHS 客户里如果有核电，要注意 |
| SAM 2.1、Qwen3-VL-8B、gsplat、vLLM、Open3D | Apache-2.0（前四个）；Open3D 为 MIT | 能 |
| Gemini | 云 API 条款 | 按 U6 的决定允许，但属于云依赖 |
| DROID-SLAM | 代码 BSD-3，权重待核 | 待核 |
| LingBot-Map | 没有书面许可 | 书面确认之前不能 |
| RecGen | TRI 非商用 | 不能 |
| FastGS、DashGaussian、EDGS | 非商用 | 只借思路，在 gsplat 上自己实现 |
| 检测器备选 OmDet-Turbo | Apache-2.0 | 能。YOLO-World 和 YOLOE 是 GPL/AGPL，不用 |

## 8. 最少的验证实验
这些都需要你批准花 GPU。统一用 ME340，在同一个热容器里跑。

| # | 测什么 | 通过标准 | A100 分钟 [E] |
|---|---|---|---|
| E1 | DA3-GIANT 和 DA3-BASE 的 any-view 模式：耗时、显存、相对 DROID 的 ATE、留出视角的深度误差 | 前向不超过 20 s；ATE 不超过 DROID 的 1.5 倍 | 10 |
| E2 | SAM 3 按批处理的速度，以及物体召回 | 2 类不超过 0.15 s/帧，20 词不超过 0.7 s/帧；能召回今天已命名物体的 80% 以上 | 10 |
| E3 | 5 fps 人轨迹对比今天的轨迹；SAM 3.1 用 stride 3 热跑的速度 | 路径差不超过 0.3 m，规则结论一致 | 10 |
| E4 | SAM 3D 快速模式，30 个物体 | 通过率下降不超过 5 个百分点；每次不超过 5 s | 15 |
| E5 | gsplat 跑 5k 和 7k 步 | 不超过 85 s，并记下 PSNR | 10 |
| E6 | 用 vLLM 跑事件 | 不超过 40 s，结果和现有 `events.json` 一致 | 5 |
| E7 | 冷启动 | 记下秒数 | 5 |

合计约 65 A100 分钟 [E]。另外还有一项不用 GPU：在云 CPU 上剖析 import 那 13 分钟到底花在哪。

## 9. 构建顺序
1. 先跑 E1–E7 和 import 剖析。E1 不通过就改用 G2。
2. 搭双卡常驻容器：解码 → DA3 → TSDF 和点云 → 在容器里导入（目标不超过 10 s）。做到这一步，就有 40 s 版的报告。
3. 接上 SAM 3：生成去人掩码、确定尺度、把物体抬到 3D。先直接复用 `build_video_object_map.py`，喂检测器的掩码；观测数量会减少约 15 倍 [E]。如果它仍超过 15 s，再移植到 torch。
4. 接上 5 fps 的人轨迹、vLLM 事件和并发的 Gemini 命名。
5. 接上 SAM 3D 快速模式和闸门，按时间点给报告发补丁（沿用 `STREAMING-PLAN.md` 里的补丁设计）。
6. 加快速泼溅。
7. 改精修层：去掉"一次只跑一个付费阶段"的限制，`max_containers` 至少 4，容器常驻，SAM 3.1 加 `--stride 3`，Gemini 并发。

**读过的文件：**
- `/Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/report_runner/{stages.py,store.py,profiles.py,decide.py}`
- `/Users/adam/.codex/worktrees/panoptes-phase2-video/modal_apps/{sam2_everything.py,sam3d_research.py,sam3_motion_tracks.py,mono_room.py,lingbot_room.py,droid_room.py,splat_train.py,video_events.py}`
- `/Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/{lingbot_dense_map.py,build_video_object_map.py,import_video_scene.py,complete_video_objects.py}`
- `/Users/adam/.codex/worktrees/panoptes-phase2-video/web/src/ReportScene.tsx`
- `/Users/adam/.codex/worktrees/panoptes-phase2-video/docs/phase2/STREAMING-PLAN.md`