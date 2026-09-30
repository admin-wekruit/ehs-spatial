# 视频对象判定与判断复用 · 设计备忘（目标：工位报告级）

快照时间：2026-09-28 22:35 CDT。全程只读，没有改任何仓库，也没有打开 `*/.platform/imports/*`。
部分实验目录还在写入（X6-010、X11-002、X9-001），这些文件的行号以这个快照为准。

路径缩写：
- PS = /Users/adam/Desktop/panoptes-public/panoptes-serving（照片工位产品）
- VW = /Users/adam/.codex/worktrees/panoptes-phase2-video（视频主线）
- FB = /Users/adam/.codex/worktrees/panoptes-phase2-video-fb-a-core（快速路径）
- X6W = /Users/adam/.codex/worktrees/panoptes-phase2-video-fx-x6-windows-time
- X7W = /Users/adam/.codex/worktrees/panoptes-phase2-video-fx-x7-object-models
- CYL = /Users/adam/.codex/worktrees/panoptes-phase2-video-m0-cylinder
- WP = /Users/adam/Desktop/panoptes-public/panoptes-workcell-pages
- RN = /Users/adam/Desktop/panoptes-public/research-notes/phase2/runs

三个视频：ME340 165-195 s、Sam's Club 337-367 s、Walmart 190-220 s（RN/fx-x6-windows-time-010/results.json:2）。

---

## 0. 先说结论

1. **目标就定在工位报告级。** 这一级不要求毫米。工位的模型评估网格只有 288 px 高（PS/scripts/research/assemble_lucida_scene.py:678；PS/scripts/research/fit_blender_posts.py:193）。文档也写明 Pi3X 深度撑不起毫米级（PS/docs/research/lucida-three-photo-experiment-2026-09-09.md:27）。视频的 DA3 网格是 504x280（FB/fast_report/core.py:21），是同一个量级。所以分辨率够用。
2. **工位的逐对象测量函数已经在视频仓库里，但视频链路从来没调用过它**（VW/ehs_spatial/measurements.py:52-118）。在 VW 的 scripts/、modal_apps/ 以及 FB 的 fast_report/ 里都搜不到调用；唯一的调用方是照片路径的 VW/ehs_spatial/object_evidence.py:18,232-240。这是最大的缺口，也是最便宜的一个：它是纯 numpy，在 CPU 上跑。
3. **对象判定现在的主要问题是"碎"，不是"漏"。** 已交付的对象里，ME340 覆盖到 20 个，其中 13 个被切成了碎块，每个中位 3 块。Walmart 覆盖 25 个，16 个是碎的（RN/fx-x6-windows-time-010/results.json:589-595,761-767）。
4. **视频上的倾角现在还不能下结论。** X6 把竖直的 CNC 门读成了 6.8°/7.2°。同一个货架，两种几何给出 12.9° 和 1.4°（RN/fx-x6-windows-time-010/results.json:4668）。工位定的倾角/坡度门槛，视频上一条都还没用。
5. **时间上：状态机已经有了，误报是 0，但几乎看不到变化。** 在 8 个没植入变化的运行上，变化声明是 0 条。植入的 20 个事件只找到 1 个（RN/fx-x6-windows-time-010/results.json:2591,2608-2613）。卡住的是检测：被移动的箱子没有作为一个独立对象被抬升到 3D。
6. **可变形物体（线缆、软管）没有任何模型。** 所有跟踪器都假设刚体，尺寸变化不超过 2 倍（X6W/fast_report/timeline.py:43）。
7. **精修（X4/X11）现在全部停在 NEEDS_REVIEW。** X4 的 15 个点里 15 个没过多视角 1.5 cm 这一关（RN/fx-x4-refine-006/results.json:47365,47981）；X11 的每一个变体也都是 NEEDS_REVIEW（RN/fx-x11-local-ba-002/me340-165.json:333,1139,2001,2883,3471,4371,5339,6331）。工位级用不到这个精度。所以把精修降为"按需补测"，不放进默认路径。
8. **下一步**：在三个视频上产出一张"工位级对象表"，每一行的字段和工位报告一一对应。要做的事是重跑一次 X6 并导出点云，其余都是 CPU 工作。在这之前不碰速度。

---

## 1. 视频上怎么判定一个对象

### 1.1 从像素到"已确认对象"的步骤

每一步都写明：做什么、复用哪段代码、要改什么、哪个实验接在这里。

**第 1 步：选帧，切内容窗口**
- 做什么：先取 5 fps 关键帧，规则是每 6 帧取最清晰的一帧（FB/fast_report/core.py:20）。当前每 3 个关键帧才做一次对象分割（FB/fast_report/segment.py:14）。
- 要改：把对象分割改成每个 5 fps 关键帧都做。X1 的结论是：三段视频的 0.5 m 召回均值从 0.550 升到 0.649，细长物体的召回从 0.2 升到 0.5（RN/fx-x1-fps-004/results.json:8178）。
- 窗口用 X6 的规则：ORB 共视阈值 0.40，每个窗口至少 8 个、至多 40 个新关键帧。三段视频分别得到 10/11/17 个窗口，中位长 1.7-2.0 s（RN/fx-x6-windows-time-010/results.json:4689）。
- 为什么要切窗口：窗口内可以当静态场景处理，照片那套"静态、多视角"的假设在窗口内成立。

**第 2 步：地面和尺度**
- 复用：照片的地面拟合，在所有候选平面里取最低的平面，法向与"上"的夹角不超过 20°，内点容差 3 cm，随机种子固定（PS/ehs_spatial/geometry.py:101-286）。VW 已经部分复用了它（VW/ehs_spatial/platform/spatial.py:568-596）。
- 要改：三套地面定义要统一成一套。照片用的是"最低平面加 MAD 检查"。快速路径用 SAM 的 floor 掩码做截尾最小二乘，并假设相机高 1.6 m（FB/fast_report/core.py:22-24,167-186）。Blender 那一版用的是 SAM floor 再与人工 ROI 取交集（PS/docs/research/blender-workcell-experiment-2026-09-09.md:9,18）。统一后，每个注册好的镜头拟合一次地面，拟合前先去掉人。
- 尺度状态直接套用 measurement_scale（VW/ehs_spatial/measurements.py:34-49）。视频的尺度来源是"地面加上假设的 1.6 m 机高"，应记为 `estimated`。现有代码会把它记成 `uncalibrated`（VW/ehs_spatial/measurements.py:47-48），所以要补一个来源分支。

**第 3 步：词表（这是什么东西）**
- 复用"分工"这条原则：VLM 只负责给出词表，实例交给 SAM，范围交给几何（PS/scripts/scene_inventory.py:282-313，第 283 行写明了这一点）。视频已经有对应版本：在 8 个关键帧上最多给出 24 个词，并注明每个词出现在哪些帧（VW/scripts/discover_video_vocabulary.py:1-5,29-34,48-55）。
- 要改：每个窗口重跑一次词表，这样后面才出现的物体也能进入词表。词表用 EHS 清单作种子（PS/ehs_spatial/taxonomy.py:14-64），另外加上人、叉车/AGV、线缆/软管这几类。

**第 4 步：2D 实例**
- 复用：SAM 3 文本提示加同义词组合。原因是"safety fence"这个词检出 0 个，而"barrier"的召回是 0.97（PS/ehs_spatial/providers/sam3.py:31-34）。围栏一类取所有同义词结果的并集，掩码 IoU>0.5 视为重复（PS/scripts/scene_inventory.py:316-435）。快速路径在每帧内还做一次泛洪去重（FB/fast_report/segment.py:246-277）。
- 类别无关的发现（X2/X10 AMG）只用作"覆盖检查"，不直接产出对象。原因是 X8 按人眼标注了 ME340 的 AMG 切片：32 个里只有 8 个是完整物体（A），17 个是部件（B），7 个是表面或背景（C）（RN/fx-x8-jev-001/sets/c-labels.json:270-275）。
- 覆盖检查可以复用平台的清单复核：每个已有观测必须恰好被点名一次，漏掉的补进来，看不清的区域登记为 unresolvedRegions（VW/ehs_spatial/platform/reconstruction.py:1124-1201）。

**第 5 步：抬升到 3D（逐关键帧、逐窗口）**
- 复用照片的去拖影三步（PS/scripts/scene_inventory.py:2644-2658,1735-1746）：
  - 掩码腐蚀 2 px，前提是腐蚀后至少还剩 30%；
  - 只保留深度在 p15-p85 之间、再放宽 0.25×IQR+0.05 的点；
  - 按径向 MAD 去离群点，阈值 2.5。
- 这一步一定要放在测量前面。测量函数本身用的是 min/max 包围盒，一个飞点就能把宽度从 0.1 m 撑到 2.5 m（见组件表中 measurements 一行的草稿验证）。
- 与人重叠 ≥50% 的掩码直接丢掉（FB/fast_report/segment.py:17,313-317；VW/scripts/build_video_object_map.py:194-197）。

**第 6 步：同一窗口内，哪些是同一个物体**
- 复用 3D 重叠累加器：新掩码有 ≥0.5 的点落在某个实体附近，就并入它；两个实体之间超过 0.7 就合并（VW/scripts/build_video_object_map.py:51-55,89-124）。
- 有争议的配对交给双向深度-掩码关联器处理。条件是：支持点 ≥32、包含率 ≥0.65、深度一致率 ≥0.65、领先第二名的差距 ≥0.15（VW/ehs_spatial/platform/spatial.py:253-259,335-357）。
- 快速路径是 5 cm 体素连通（FB/fast_report/segment.py:304-349），它就是"碎"的来源。要补一轮合并（上面 MERGE 0.7 那一步），再补上部件规则（见第 9 步）。

**第 7 步：跨窗口，是不是同一个物体**
- 复用 X6：做一对一的 Hungarian 匹配，条件是质心距离在 3σ 以内，或者稳健框 IoU≥0.2（X6W/fast_report/timeline.py:36-37）。
- σ 的算法是 sqrt(0.04² + (0.05·z)²)（X6W/fast_report/timeline.py:47-48）。
- 静态物体的身份不加外观门槛。加了反而更碎：q90 外观门槛会让对象数多出 29%，并产生 249 对拆分（RN/fx-x6-windows-time-010/results.json:4691）。

**第 8 步：长得一样的东西**
- 规则：身份只由位置决定，名字和外观都不能决定。X8 的人工审核发现了两类情况。一类是同款物品，比如车间里的几个 ProtoTRAK 吊挂控制盒、同款纸巾包，人眼也分不清。另一类是交付地图本身就把不同的东西连到了一起：Walmart 的跨对象正样本 25 对里有 11 对被翻成负样本（RN/fx-x8-jev-001/sets/a-audit.json:4-7）。
- 复用：一旦有"不同"决定，就不能再自动合并（VW/ehs_spatial/platform/spatial.py:313-316,338；VW/ehs_spatial/platform/identity.py:817-854）。
- X8 的裁决器只处理有歧义的配对，而且只能作为证据，不能直接作为结论。

**第 9 步：部件还是整体**
- 复用这几条规则：
  - 父子关系必须有观测证据（VW/ehs_spatial/platform/identity.py:84-106）；
  - 不能凭名字相近或框重叠就推断部件关系（VW/ehs_spatial/platform/agent_service.py:63-64）；
  - 某实体有 ≥0.5 的点落在一个已接受的模型上，就算那个模型的部件（VW/scripts/complete_video_objects.py:61,1112-1124）；
  - 围栏一类在竖直方向叠在一起的色带合并成一件（PS/scripts/scene_inventory.py:68-118）。
- 测试集用 X8 的集合 c（A 完整物体 / B 部件 / C 表面 / D 多物体，定义见 RN/fx-x8-jev-001/sets/c-labels.json:4）。

**第 10 步：名字**
- 名字只是属性，从不移动几何（VW/scripts/name_video_entities.py:1-8）。快速路径的名字是 SAM 3 的投票词，标注为"未验证"（FB/fast_report/core.py:517）。
- X8 集合 b 的审核记录了一批该改名的条目，比如"tool holder""control panel"（RN/fx-x8-jev-001/sets/b-audit.json:4-5）。名字错了会带偏两件事：X7 选哪种基本体（X7W/fast_report/x7.py:42-46），以及策略里的主体选择。

**第 11 步：范围（多大、多高、朝哪）**
- 复用：每个对象、每个窗口调用一次 measure_observed_points（VW/ehs_spatial/measurements.py:52-118）。
- 要改：宽和深改用最小旋转矩形来量（PS/scripts/scene_inventory.py:2666-2674）。现在的基底只跟地面对齐，水平朝向是任意的：一块 1.0×0.1 m 的板转 45° 后会读成 0.78×0.78（见组件表草稿验证）。

**第 12 步：确认**
- 满足以下全部条件才算确认：
  - 在一个窗口内至少 3 个视角（VW/scripts/build_video_object_map.py:51）；
  - 视角之间要有足够差异，建模时至少相隔 15°（X7W/fast_report/x7.py:61-72）；
  - 测量可用，即有 ≥8 个点（VW/ehs_spatial/measurements.py:72）。
- 连续视频帧能轻松凑够"帧数"，所以不能只数帧，要看视角差。照片那边的门槛数的是帧数（PS/ehs_spatial/rules.py:65-66；PS/ehs_spatial/policy.py:77）。

**第 13 步：未解决记录**
- 照片的"不静默丢弃"原则：每个词最后要么被测到，要么进 unresolved.json，并写明原因（PS/scripts/scene_inventory.py:2735-2755）。平台 agent 也写了这条（VW/ehs_spatial/platform/agent_service.py:52）。
- 要改：检查的单位从"词"扩成（实体，窗口）。现在照片侧还有静默丢失的情况：急停检测在 PS/runs/user-bor1-02/refinements.json:251 里有，但 inventory 和 unresolved 里都查不到。
- X6 已经给出"未观测"的几种原因：出视野、被遮挡、占着但没检出、观测太少、太小（X6W/fast_report/timeline.py:12-28）。

### 1.2 能看多细：分三层说

| 层 | 分辨率（每像素对应的实物尺寸，按距离缩放） | 依据 |
|---|---|---|
| 源帧 1280×720 | 约 1.7 mm × 距离(m) | 由 X11 记录的像素足迹换算：DA3 网格 1 px 在 4.86 m 处为 21.34 mm（RN/fx-x11-local-ba-001/me340-165.json:267-270）；DA3 1 px 等于 6.53 个源像素的面积（X7W/fast_report/x7.py:484-486） |
| DA3 网格 504×280 | 约 4.4 mm × 距离 | 同上；1.56 m 处为 6.84 mm（RN/fx-x11-local-ba-001/me340-165.json:1574-1577） |
| 快速 3D 抬升（步长 2） | 约 8.8 mm × 距离；每个掩码至少 16 px，所以边长下限约 3.5 cm × 距离（3 m 处约 10 cm，5 m 处约 18 cm）；体素 5 cm | FB/fast_report/segment.py:17,313-317 |
| X6 状态判断 | ≥25 px 且对角线 ≥0.1 m | X6W/fast_report/timeline.py:38-39 |
| 3D 位置误差 | 位姿约 4 cm，深度约 5%；实测：各视角质心离散的中位值约 3 cm | X6W/fast_report/timeline.py:36；RN/fx-x1-fps-004/results.json:6663 |
| 精修点（X4） | 点间距 4.4 mm，但多视角一致性的中位值 4.6 cm | RN/fx-x4-refine-006/results.json:47385,47391,47981 |

由此得出三档"看得见"：
- **L0 2D 可见**：有掩码就行。近处能看到厘米级甚至毫米级的东西，比如线缆、标牌、按钮。
- **L1 3D 可测**：能抬升，≥8 个点，朝向测量需要 ≥20 个拟合点（VW/ehs_spatial/measurements.py:72,87）。实际下限约为 3.5 cm × 距离，前提是快速路径用步长 2 抬升。测量时应在全网格（步长 1）上重新取点，下限可降到约 1.8 cm × 距离。
- **L2 细测**：只在点名的点上做。现在做不到"确认"。X4 的结论是：粗相机的 ATE 约 4 cm，加上 DA3 本身 2-5% 的深度误差，一起定出了这个下限（RN/fx-x4-refine-006/results.json:47981）。

**工位级需要的是 L0 加 L1。** 工位的对象是柱、围栏、料车、机器人、光幕，尺寸在 0.1-2 m。这些对象在 5 m 内都落在 L1 里。

每个对象都要自报一句"能看多细"：取它所有视角里最近的距离，乘上 3.5 cm（或 1.8 cm），就是 3D 下限。比下限还小的对象只记为 L0，写进未解决记录，原因写"此距离下不足以做 3D 测量"，并附上最近一次看到它的关键帧。

**补测路径（refine）**：
- 照片：评审人画框，然后 SAM，然后测量（PS/ehs_spatial/refine.py:22-223）。代理版本是 Gemini 定位，再截图自检，再测量，约 9 s（PS/ehs_spatial/agent.py:26-277）。视频主线已经支持指定帧（VW/ehs_spatial/refine.py:35-43,72-138；VW/ehs_spatial/agent.py:97-106）。
- 视频上这样改：
  1. 评审人在一个关键帧上画框；
  2. SAM 3.1 把框传播到整个窗口；
  3. 把传播得到的掩码点并起来，调用 measure_observed_points；
  4. 如果框指向一个已有实体，就给它加一条观测，不新建实体。
- X4/X11 那种 504 px 裁剪、4-6 视角的精修（约 2.3 s/点；RN/fx-x4-refine-006/results.json:47978-47979）只作为可选的显示补丁，不产出事实。

### 1.3 X6-X11 各接在哪一步

| 实验 | 接在哪一步 | 目前的结果 |
|---|---|---|
| X6 内容窗口与对象时间 | 第 1、7、13 步和第 3 节 | 有结果。窗口规则已定（results.json:4689）；关联只看位置（:4691）；变化规则误报 0、召回 1/20（:4692）；倾角不可用（:4668）；带区间的事实有 4 类（X6W/modal_apps/x6_windows_app.py:321,339,357,390） |
| X7 逐对象模型 | 第 2 节的基本体和显示模型 | 有结果。held-out 门槛是 IoU≥0.65、深度中位≤0.04、p95≤0.10（RN/fx-x7-object-models-001/results.json:6）。参数盒 58/87 被接受，RecGen 45/87。货架所有方法都是 0/4（:3282-3445）。结论是先用参数盒，隔板用参数平面（:6010-） |
| X8 Jev-Omni 裁决器 | 第 8、9、10 步 | 只建好了标注集 a/b/c，还没有打分结果（RN/fx-x8-jev-001/sets/） |
| X9 同物复用 | 第 6、7 步的加速与传播 | 目录刚建，是空的（RN/fx-x9-reuse-001）。脚本只做 SAM 3 跟踪器和 DINOv2 的成本探针（未提交的 modal_apps/x9_probe.py:1-5） |
| X10 类别无关发现 | 第 4 步的覆盖检查 | 还没有运行目录。可以参考的前身是 X2 和 X8 集合 c：AMG 大多给出部件 |
| X11 精修点局部 BA | 1.2 的 L2 补测 | ME340 上所有变体都是 NEEDS_REVIEW。BA 把托盘的多视角一致性从 7.8 cm 降到 3.7 cm，但仍然没过 1.5 cm，而且和粗图不再一致（RN/fx-x11-local-ba-002/me340-165.json:3382,4064）。线缆加了 BA 反而更差 |

---

## 2. 已有的判断，以及怎么扩到视频

### 2.1 每条事实统一带证据级别

先定义证据级别，下面的表会引用它：
- **E0 仅 2D**：只有掩码。
- **E1 support-points-v1**：measure_observed_points 返回 available（VW/ehs_spatial/measurements.py:82）。
- **E2 coarse-layout-position-v1**：模型或基本体过了粗档检查，即 5% 对角线容差内覆盖 ≥0.9、深度内点 ≥0.8（VW/ehs_spatial/platform/model_quality.py:29-30）。
- **E3 observed-model-quality-v1 加上 held-out 视角**：IoU≥0.65、精度≥0.7、深度 p50≤0.05、p95≤0.15（VW/ehs_spatial/platform/model_quality.py:22-28），并且过了 X7 的 held-out 关（X7W/fast_report/x7.py:483-503）。

每条事实都带这些字段：对象、属性、值、不确定度、`interval_s`、窗口、视角数、证据帧、级别、尺度状态。X6 的事实格式已经有其中大半（X6W/modal_apps/x6_windows_app.py:321-323）。还缺三个字段：级别、尺度状态、视角数。

### 2.2 判断清单

| 判断 | 照片端的输入与阈值 | 扩到视频 | 最低证据级别 |
|---|---|---|---|
| 高/宽/深，观测范围 | 需要 ≥8 个点和有效地面；在地面基底下取 min/max 包围盒，给出 8 个角点（VW/ehs_spatial/measurements.py:72-83）；工位端另外要求 conf≥0.1、深度>0，并断言测出的范围恰好包住支持点（WP/build-blender-ranges.py:97-107,118） | 每个对象、每个窗口算一次，窗口间取中位值和离散度；去拖影放在测量前；宽深用最小旋转矩形 | E1 |
| 主轴倾角 | 取中心 95% 的点，≥20 个；λ1/λ2≥1.5 才输出；它的含义只是"可见支撑的主轴"（VW/ehs_spatial/measurements.py:85-104） | 同上，按窗口出时间序列，可用来发现物体倾斜或转动。**不要**用 PS 的 _spatial_state 倾角做策略判断（PS/ehs_spatial/geometry.py:342-350）：直立的料车会被读成 90°，MAX_TILT 就会误判 FAIL（PS/ehs_spatial/policy.py:176-192） | E1，只作证据 |
| 平面坡度 | λ2/λ1≥0.05、λ3/λ2≤0.02、p95 残差/中轴跨度≤0.05（VW/ehs_spatial/measurements.py:105-117） | 窗口内合并多视角的点再算，更容易过门槛。先在已知竖直面上校准：ME340 的门、控制面板；货架立柱。X6 的主面倾角有约 7° 偏差（RN/fx-x6-windows-time-010/results.json:4668）；X7 的平面拟合在控制面板上给出 0.077±0.163°（RN/fx-x7-object-models-001/results.json:3446-3466） | E1 |
| 物理倾角（可以作判定） | 策略要求 tilt_reference == 'physical_axis'（VW/ehs_spatial/policy.py:184；VW/ehs_spatial/contracts.py:144），但目前没有任何生产代码会设置它 | 只从过了 held-out 的基本体取轴，比如 X7 的平面或柱（X7W/fast_report/x7.py:249-316）。要加一条角度误差带：X4 的数据说明点云算出的置信区间偏小，同一个点两次运行之间细测倾角最多差 8.6°（RN/fx-x4-refine-006/results.json:47981）。平台的倾角分析已经带 angularErrorDeg，并分类为"误差内竖直 / 非竖直"（VW/ehs_spatial/platform/planar_surfaces.py:9,110），但它作用在模型网格上，不在观测点上 | E3 |
| 参数化基本体 | 64 边直立圆柱：用种子点做圆拟合，再在 144 px 网格上渲染对比，优化 5 个参数；替换规则是平均 IoU 不降、平均深度 p50 不升（PS/scripts/research/fit_blender_posts.py:20-35,57-83,141-156） | 视频现有的是：重力对齐盒（VW/scripts/complete_video_objects.py:698-741）；证明了圆度的直立圆柱，在未合并的分支上（CYL/scripts/complete_video_objects.py:715-760）；X7 的盒、板、柱。都缺"渲染对比优化参数"这一步，也缺同一个替换规则 | E2/E3 |
| 显示模型的来源一致性 | 每个视角算 IoU、边界误差/图高、相对深度 p50/p95；loss = 1-IoU + 2×边界误差 + p50（PS/scripts/research/assemble_lucida_scene.py:155-182）；粗档检查（VW/ehs_spatial/platform/model_quality.py:29-30） | X7 已经原样复制了 score_view 和 refine（X7W/fast_report/x7.py:124-186），并加了 held-out。还缺：边界误差和粗档的容差覆盖率。主线视频只输出 IoU 和深度（VW/scripts/build_lingbot_object_model.py:69-89）。显示模型只用来展示，不当测量（RN/fx-x7-object-models-001/results.json:9） | E2 |
| 围栏间距（clearance） | 围栏需 ≥min(3, 帧数) 帧、面积 ≥0.25 m²；可移动物需 ≥min(2, 帧数) 帧、面积 ≥0.0025 m²、高 ≥0.05 m；误差带多视角 0.20 m、单目 0.35 m；默认阈值 0.6 m（PS/ehs_spatial/rules.py:22-23,65-177） | 每个窗口内，只在同时在场的实体之间计算。门槛里传视角数，不传帧数。误差带换成每条事实自己的不确定度（VW/ehs_spatial/policy.py:68,112；VW/ehs_spatial/platform/policy_engine.py:201-231）。尺度是 estimated 时，结论降为 NEEDS_REVIEW（VW/ehs_spatial/video.py:945-950） | E1 |
| 策略谓词（最小/最大间距、禁入、高度、倾角） | VW 的强化版：缺目标给 INSUFFICIENT；米制谓词需要可信尺度（VW/ehs_spatial/policy.py:63-330） | 每个窗口在场景快照上评估一次，再按时间汇总（见 3.4） | E1；倾角需 E3 |
| 间距矩阵 | 多边形两两距离（PS/ehs_spatial/interactive_report.py:154-157） | 每个窗口、只在同时在场的实体之间算。X6 的"通道净宽"事实是视频上的同类（X6W/modal_apps/x6_windows_app.py:362-393），它只受已检出物体约束 | E1 |
| 可攀爬 | Gemini 给语义提示，必须引用事实 ID，界面上只标"REVIEW only"（PS/ehs_spatial/providers/gemini.py:133-209） | 每个窗口在关键帧上调用一次（一次最多 1-4 帧，PS/ehs_spatial/providers/gemini.py:140-143）。确定性的判法是看水平构件：坡度接近 0，再看各层高度和竖向间距。**这一条没有实现** | E1，仍只作复核 |
| 薄结构落脚线 | 用接地边做射线投射，15% 容差，≥max(8, 30% 列) 列；法向拆分；竖向堆叠合并（PS/scripts/scene_inventory.py:68-118,147-201,486-530） | 每个关键帧做一次，再把验证过的接地点在多帧之间合起来拟合一条线 | E1 |
| 护栏链与重投影校正 | 链接条件：间隙<15% 图宽、高度比<1.8；滑动校正 ±0.7 m；残差不能变差（PS/scripts/scene_inventory.py:540-1146；PS/ehs_spatial/reproject.py:22-198） | 每帧建链再在 3D 里融合。用一部分帧拟合，留出另一部分帧检验，这是照片做不到的 | E1 |
| 墙与工位矩形 | 墙：RANSAC 取 400 个假设、内点 0.12 m、长 ≥1 m；边：支持 ≥最强边的 50%、距离 ≤7 m、跳跃 ≤2.5 m（PS/scripts/scene_inventory.py:1749-1793,1796-2143） | 每张注册好的静态地图跑一次。这些米制阈值只在尺度可信时才有意义 | E1 + 尺度 |
| 表面语义（黄黑条纹） | 目前只写在 VLM 提示里：水平是台面，斜面是导向板，竖直是围栏（PS/ehs_spatial/agent.py:50）；清单里的类型见 PS/ehs_spatial/taxonomy.py:52-53,63 | 改成确定性规则：平面坡度加离地高度。坡度分箱的具体数值**还没定义** | E1 |
| 悬伸 | 上段点对下段外壳的最大外伸（PS/ehs_spatial/geometry.py:352-359） | 每个窗口算一次，跟踪时间序列 | E1 |
| 空位证据 | 盒子是否悬空、是否吞了别的物体、源面是否有支撑（VW/scripts/box_free_space.py:42-73）；X6 的空位检验（X6W/fast_report/timeline.py:16-21） | 推广到所有已接受的模型，并把"看见是空的"作为间距的正面证据 | E1 |
| 人员规则 R1-R3 | 禁区、距离 2.0 m、速度 1.5 m/s、误差带 0.35 m、NO_DATA 纪律（VW/ehs_spatial/video.py:58-85,1058-1116） | 原样保留 | — |

---

## 3. 跨时间

### 3.1 对象在每个窗口的状态（直接复用 X6）

状态有六种：首次出现、静止、移动、消失、出现、未观测（带原因）（X6W/fast_report/timeline.py:12-28）。
每个变化都要带变化前、变化后的关键帧作为证据（X6W/fast_report/timeline.py:26）。

### 3.2 按可移动性分四类

| 类 | 例子 | 允许的事实 | 已有 | 缺 |
|---|---|---|---|---|
| 固定 | 围栏、墙、柱、机床、货架 | 每张地图出一整套工位事实，之后每个窗口复核一次（看离散度） | X6 静态状态 | 类别只靠名字判断 |
| 可移动刚体 | 料车、托盘、箱子、梯子 | 每个静止区间一套事实；移动后开新区间 | X6 的移动/消失规则 | 实测召回 1/20 |
| 可变形 | 线缆、软管、链条、帘子 | 每个窗口只给观测范围；不配基本体，不给倾角；"形变"不算"移动" | 无 | 全部 |
| 行为体 | 人、叉车/AGV、正在动的机械臂 | 每个采样时刻给位置和速度，套用 R1-R3 | FB/ehs_spatial/live_people.py:34,146-162；VW/modal_apps/sam3_motion_tracks.py:31-33；VW/scripts/motion_facts.py:12-56 | 行为体和静态物体不共享身份：料车被推动后就变成一个新 id |

分类怎么定：
- 先用名字给一个先验（VW/scripts/report_runner/decide.py:428-471 已经有"静态实体上的人"过滤）。
- 再用观测到的行为纠正：
  - 跨窗口移动过的，归为可移动；
  - 位置没变、形状变了超过 2 倍的，是可变形候选（今天这种情况会被 X6 误读成"消失加新出现"，X6W/fast_report/timeline.py:43）；
  - 出现在运动轨迹里的，归为行为体。
- 机械臂这类"底座固定、部件在动"的对象，用部件关系拆开处理（VW/ehs_spatial/platform/identity.py:84-106）。

### 3.3 什么时候能说"移动了/消失了"，什么时候只能说"被挡住了"

直接复用 X6 的规则（X6W/fast_report/timeline.py:36-44；RN/fx-x6-windows-time-010/results.json:4692）：
- **消失**：要"看穿了它原来的位置"。具体是：物体点的投影位置处，窗口深度（取 5×5 最小值）比物体点远出 >20%z + 2σ；这样的点在 ≥2 个关键帧里、占被判点的 ≥60%；点离相机 ≤5 m，落在视野内侧 90%；人周围外扩 2 px 不判；这个物体此前至少在 3 个关键帧上被看到过。
- **移动**：原位置是空的，另一处出现一个未匹配的实例，外观余弦 ≥ 本视频负样本对的 99 分位（约 0.98），尺寸差在 2 倍以内，距离超过 3σ。
- **出现**：必须有更早的窗口看到过这个位置是空的。否则只算"首次出现"。
- **被挡住 / 占着但没检出 / 出视野**：都记为"未观测"，写明原因，不能算作变化。
- **跨拼接比较**：两个窗口之间的 Sim3 尺度变化 Σ|log s| ≤0.1 才允许比较。

实测结果：
- 8 个没植入变化的运行，变化声明 0 条（RN/fx-x6-windows-time-010/results.json:2591）。
- 真实变化 0 个。原因是走查视频一路向前，没有哪个位置被看到两次（:2605-2606）。
- 植入的 20 个事件找到 1 个。漏掉的 19 个都是因为箱子没被抬升成独立对象：SAM 3 其实分割出来了，IoU 有 0.95-0.99，但每 3 个关键帧才分割一次，而且靠近货架时箱子被并进了货架（:2608-2613）。
- 所以要修的是检测密度和部件规则，**不是变化规则**。

### 3.4 事实怎么带时间区间，物体变了怎么重算

规则如下：
1. 一条事实属于一个（对象，静止区间）。
2. 出现移动、出现、消失时，开一个新区间，重算测量值。
3. 在区间内，每个窗口出一个值。区间的值取中位值，不确定度 = hypot(窗口间离散度, σ(距离))，和 X6 的算法一样（X6W/modal_apps/x6_windows_app.py:321-323）。
4. 如果新窗口的值偏离区间中位值超过 3σ，而状态并不是"移动"，就标记为"原地变化"，结论给 NEEDS_REVIEW。原因可能是形变，也可能是测量出错。
5. "出视野"不结束区间，但要记下最后观测时间（lastObservedAt）。这个字段目前只写在计划里（VW/docs/phase2/STREAMING-PLAN.md:117-118），代码里只有 TSDF 块级别的记录。
6. 策略按区间评估：
   - 任一有覆盖的区间 FAIL，就是 FAIL；
   - 只有覆盖时长达标、并且所有区间都 PASS，才算 PASS。
   - 现在 worst_verdict 把 NO_DATA 排在 PASS 下面（VW/ehs_spatial/video.py:83-85,1177-1181），结果是一帧 PASS 加 59 帧 NO_DATA 汇总成 PASS。这个必须先修。

### 3.5 已有的和缺的

- **已有**：
  - 窗口内静态身份（VW/scripts/build_video_object_map.py）
  - 跨窗口身份和状态（X6）
  - 运动物体的时序表面和运动事实（VW/modal_apps/mono_room.py:581-641；VW/scripts/motion_facts.py:1-56）
  - 人员规则和 NO_DATA 纪律（VW/ehs_spatial/video.py:953-1116）
- **缺**：
  - 可变形物体的模型
  - 静态物体和运动物体共用一个身份
  - 实体级的 lastObservedAt
  - 离开视野后的重识别：ReID 目前只提出候选边（VW/scripts/link_person_tracklets.py:1,16-20）
  - 合同里没有时间字段：照片的 Entity3D/SpatialFact 不带时间（PS/ehs_spatial/contracts.py:112-138），DYNAMIC-SCENE 里的字段还没实现（VW/docs/phase2/DYNAMIC-SCENE.md:78）
  - 类别由行为来判定

---

## 4. 验收标准：工位级逐字段对照

| # | 字段 | 工位的定义与阈值 | 视频今天（快速路径 + 已跑的实验） | 差距 |
|---|---|---|---|---|
| 1 | 身份 | 用确切的证据对象 ID，必须是同一张照片里的掩码（WP/build-blender-ranges.py:86-91）；靠视觉对应确认，不允许只凭标签合并（PS/outputs/candidate-evaluation/lucida-replica-01/evidence/objects.json:16-17） | 3D 重叠或 X6 的位置关联；ME340 覆盖到的 20 个里有 13 个是碎的 | 要补合并、部件规则，并用 X8 集合 a/c 审核 |
| 2 | 支持点 | 掩码 ∧ 有效 ∧ conf≥0.1 ∧ 深度>0（WP/build-blender-ranges.py:97-99,118） | 掩码 ∧ 深度>0 ∧ 非人；没有 conf 门槛（FB/fast_report/segment.py:313-317） | 补 DA3 的置信度门槛，补去拖影 |
| 3 | 观测范围和高宽深 | 点数 ≥8；地面基底下的 min/max；8 个角点（VW/ehs_spatial/measurements.py:72-83）；覆盖度记为 `observed_partial`（:56） | 世界轴对齐的 5 cm 体素盒（FB/fast_report/core.py:497-503），或原生轴的 2-98 分位盒（VW/scripts/build_video_object_map.py:249） | 调用 measure_observed_points，宽深改成按物体朝向量 |
| 4 | 主轴倾角 | ≥20 个点，λ1/λ2≥1.5（VW/ehs_spatial/measurements.py:85-104） | X6 的"主面倾角"只要求 >45°，没有其他门槛（X6W/modal_apps/x6_windows_app.py:325-333） | 换成工位的门槛 |
| 5 | 平面坡度 | 0.05 / 0.02 / 0.05 三个门槛（VW/ehs_spatial/measurements.py:113-117） | 没有；X6 竖直面偏 7° | 补上，并用已知竖直面校准 |
| 6 | 尺度状态 | uncalibrated / estimated / operator_anchored（VW/ehs_spatial/measurements.py:34-49）；工位场景本身就是 uncalibrated（WP/README.md:13） | 地面 + 假设的 1.6 m，记为 estimated（FB/fast_report/core.py:22-24）；Sam's Club 比参考大 1.12-1.17 倍（RN/fx-x6-windows-time-010/results.json:647-700） | 角度事实不依赖尺度，可以直接用；米制事实只能进复核 |
| 7 | 参数化基本体 | 圆柱加替换规则（PS/scripts/research/fit_blender_posts.py:20-35）；左柱 IoU 从 85.11% 升到 89.12%，右柱深度变差，所以没换（PS/docs/research/blender-workcell-experiment-2026-09-09.md:51-52） | X7 参数盒 58/87、平面 2/4、货架 0/4；圆柱在未合并的分支上 | 合并圆柱分支；加渲染对比优化和替换规则 |
| 8 | 显示模型一致性 | 每个视角：IoU、边界误差/图高、深度 p50/p95（PS/scripts/research/assemble_lucida_scene.py:155-182）；这些只衡量与输入视角的一致性（WP/README.md:17） | X7 有 held-out 关：IoU≥0.65、深度中位≤0.04、p95≤0.10。这一点比工位还强 | 补边界误差和粗档容差覆盖率 |
| 9 | 发现（间距、可攀爬） | 间距误差带 0.20/0.35 m；可攀爬只作复核 | 只有一条演示规则（书桌到柜子）；runner 的 import 阶段没有传 `--policy`（VW/scripts/report_runner/stages.py:864-873）；可攀爬没有 | 按窗口接入策略引擎 |
| 10 | 证据不足的记录 | 3 条几何不足的记录保留在列表里并写明原因（WP/README.md:11） | X6 有"未观测"原因；实体级的未解决清单没有 | 按（实体，窗口）写未解决清单 |
| 11 | 回到照片看 | 选中对象后，把网格、包围盒、坐标轴投回源照片（WP/README.md:11） | 3D 实体可以投到每一帧（VW/scripts/project_entities_to_frames.py:1-6,23） | 投影里加上地面对齐的角点和轴 |

---

## 5. 缺口清单和最小的下一步

### 5.1 缺口（按优先级）

1. 视频上没有调用工位的测量函数（第 4 节 #3、#4、#5）。
2. 对象太碎，部件和整体没分开（#1）。
3. 倾角有偏差，也没有用已知竖直面校准过（#4、#5）。
4. 变化检测的召回被检测密度卡住（3.3）。
5. 事实上没有证据级别和尺度状态字段；worst_verdict 会让 NO_DATA 藏在 PASS 后面（3.4）。
6. 基本体没有替换规则；圆柱在未合并的分支上（#7）。
7. 显示模型缺边界误差和粗档检查（#8）。
8. 可变形物体没有任何表示（3.2）。
9. 三套地面定义不一致（第 2 步）。
10. 策略没有接进视频报告（#9）。

### 5.2 在三个视频上验证 (1)-(3)，都放在速度之前

下面的"通过标准"是建议值，还没有验证过。跑完第一轮后再定。

**V1 对象表（验证第 1 节）**
- 做法：
  - 重跑 X6 的 (b) 几何。每个 5 fps 关键帧都分割（X1 的建议），并导出每个窗口、每个实例的点、掩码和相机。
  - 在 CPU 上补一轮合并（MERGE 0.7），加上部件规则。
  - 与交付地图和 X8 集合 a/c 对照。
- 输出：每行一个对象，字段就是第 4 节的 #1-#6 加上 #10。
- 建议的通过标准：
  - 覆盖到的交付对象里，碎块比例不超过 20%。今天 ME340 是 13/20，Walmart 是 16/25。
  - 在 X8 集合 a 里排除"看不清"的配对后，同/不同判对 ≥90%。
  - 已确认对象里，A 类（完整物体）≥80%，按集合 c 的定义。
- 成本：GPU 只在重跑 X6 时用到。按 X6 实测，一个视频全部事实 17-29 s，每张 A100 峰值 28-49 GiB（RN/fx-x6-windows-time-010/results.json:580,623,666,709,820 起；时间见 :4690）。其余都在 CPU 上。

**V2 工位级测量（验证第 2 节）**
- 做法：
  - 对 V1 的每个对象、每个窗口调用 measure_observed_points：先去拖影，宽深用最小旋转矩形。
  - 在已知竖直面上校准平面坡度：ME340 的 CNC 门、控制面板；Sam's Club 和 Walmart 的货架立柱。
- 建议的通过标准：
  - 已知竖直面的坡度在 90±3° 以内，窗口间离散度 ≤2°。
  - 同一静止区间内，高宽深在窗口间的离散度 ≤max(5 cm, 10%)。
  - Sam's Club 的堆箱顶高要和交付地图对上：现在两种几何给出 4.81/5.62 m，交付值是 4.11 m（RN/fx-x6-windows-time-010/results.json:4663-4667）。
- 如果竖直面过不了：倾角事实全部降为 E0 证据，不进入判断。

**V3 模型和基本体（验证 E2/E3）**
- 做法：
  - 对 X7 已接受的模型，加跑粗档检查（assess_model，coarse=True，VW/ehs_spatial/platform/model_quality.py:205-237）和边界误差。
  - 按 fit_blender_posts 的规则比较"参数盒或平面"和"生成模型"，决定用哪个。
- 通过标准：出一张和工位 metrics.html 同格式的逐视角对照表，外加 held-out 那一列。

**V4 时间（验证第 3 节）**
- 做法：
  - 在植入视频上把对象分割提到每个 5 fps 关键帧。
  - 给部件规则加一条："靠近货架的箱子不能并进货架"。
- 建议的通过标准：
  - 植入 20 个找到 ≥10 个；
  - 8 个未植入运行的误报仍为 0；
  - 所有事实都带区间、级别和尺度状态。
- 同时修掉 worst_verdict 里 NO_DATA 的排序问题。

**V5 发现**
- 做法：每个窗口，在同时在场的实体之间，经 VW 的策略引擎计算间距。
- 预期：尺度是 estimated 时，所有米制结论都应该是 NEEDS_REVIEW。只有角度谓词（需要 E3 的物理轴）可以直接给出 PASS 或 FAIL。
- 注意：平台引擎现在对角度谓词也要求米制足迹和 operator_anchored 尺度（VW/ehs_spatial/platform/policy_engine.py:204-212），而且构建 Entity3D 时不带 tilt_deg（:223）。要先把角度谓词从尺度门槛里放出来。

**V6 可变形物体**
- 做法：ME340 的线缆（X6 的 g85 和 g148；RN/fx-x6-windows-time-010/results.json:3056-3082）只报每个窗口的观测范围。
- 检查：线缆上不能出现"移动/消失"的声明，也不能输出倾角。

**明确不做的事**：在 V1-V6 通过之前，不做 X4/X11 精修的推广，不做速度优化。工位级用不到毫米级细节（PS/docs/research/lucida-three-photo-experiment-2026-09-09.md:27）。


---

## 代码核对员更正（以此为准）

I opened the memo's citations across PS, VW, FB, X6W, X7W, CYL, WP and RN (read-only). Most line numbers exist and say roughly what the memo claims. Five claims that drive the memo's conclusions are wrong or overstated:

- **Misses matter, not only fragmentation.** Sam's Club covers only 14-25 of 67 delivered objects, and Walmart 14-25 of 40, depending on geometry (a)/(b).
- **Densifying does not unlock planted changes.** X6 already ran every-keyframe planted runs and found 1 of 10 events.
- **X6 main-face tilt is gated,** though loosely, not ungated as the memo says.
- **X6 does not turn in-place deformation into disappear + appear.** Identity is position-only, so deformation is recorded silently as 'static'. The 2x size limit applies only to 'moved'.
- **The 58/87 box acceptance is X7's sep5 sensitivity arm.** The spec arm (sep15) is 5/9.

**Components that already exist but the memo proposes to build or import:**
- Oriented box extents: complete_video_objects box_mesh, and X7's yaw search.
- Multi-view spill voting: build_video_object_map consensus().
- Video scale classification: video.contract_scale.
- Per-entity metric uncertainty through the real engine: evaluate_video_policy.
- A mainline model-vs-box replacement rule: merge_object_models.
- Entity-level last_seen: X6 timelines.
- A deformable 're-detect, never reuse' class list: untracked X9 code.

**Policy results.** With an assumed camera height, the platform engine gives INSUFFICIENT_EVIDENCE, not the NEEDS_REVIEW the memo expects. Meanwhile ehs_spatial/policy.py accepts scale_source 'camera_height' as qualified, so an assumed 1.6 m could still yield PASS/FAIL there.

**The workcell bar itself.** It measures one reference frame per object on a 518x518 Pi3X grid. The 288 px is only the model-comparison grid. The post-replacement gate was post hoc, and the published scene uses both cylinders anyway.

**Photo assumptions the memo reuses without flagging:**
- The radial-MAD cleaner assumes one camera at the origin.
- scene_inventory, reproject and PS refine are wired to frames[0] / frame_0001.
- The cell rectangle, the ±10° axis snap, the agent's 'rectangular cell' premise and the fence convex-hull merge all assume one fixed rectangular cell.
- The taxonomy is a robot-cell LEFT/RIGHT checklist.
- _fit_floor needs a measured lens height plus a camera-height MAD gate.
- coarse_open_frame needs exactly two views, X7 caps at 4 generation views, and climb review allows 1-4 frames.

**Label provenance.** The X8 truth sets are agent-labelled, not human-labelled, so they need a human spot-check before being used as V1 pass criteria.

- **说法：**Object determination fails mainly by fragmentation, not by missing objects (ME340 20 covered/13 in pieces; Walmart 25/16).
  **问题：**The figures are correct, but they come from geometry (a) and leave out Sam's Club. Sam's Club covers only 14 of 67 delivered objects (a) and 25 of 67 (b): RN/fx-x6-windows-time-010/results.json:675-680 and 718-723. Walmart covers 25 of 40 (a) and 14 of 40 (b): :761-766 and :804-809. X1 position recall at 5 fps is only 0.487 on Sam's Club: RN/fx-x1-fps-004/results.json:8178. Only ME340 is mostly a fragmentation problem.
  **更正：**Say that both fragmentation and misses matter. ME340 (20/22 covered) is fragmentation-dominated. Sam's Club (≤25/67) and Walmart are recall-limited. Give numbers for both geometries (a) and (b).
  位置：§0 item 3 (对象判定主要是碎不是漏)
- **说法：**19 of 20 planted misses happen because objects are segmented only every 3rd keyframe, so segmenting every 5 fps keyframe should find ≥10 of 20.
  **问题：**X6 already ran every-keyframe planted runs (planted-every1) on Sam's Club a/b and Walmart a/b: results.json:2682, 2798, 2894, 2982. Together they found 1 of 10 events; the only hit is the Sam's Club b disappearance. Denser segmentation alone barely helps. The results also say the box 'joins the shelf's instance' near shelves (:2610-2612).
  **更正：**Treat the every-keyframe result as existing evidence. Make the part/merge rule (box vs shelf) the primary fix in V4 and density secondary. Recalibrate the ≥10/20 target, or justify it against the 1/10 every1 baseline.
  位置：§3.3 实测结果 and §5.2 V4
- **说法：**X6's main-face tilt only requires the angle to be >45°, with no other gate (X6W/modal_apps/x6_windows_app.py:325-333).
  **问题：**The code has two more gates: ≥30 points (:327) and smallest/middle eigenvalue ≤0.2 (:330). It also uses all points, not the central 95%. The workcell gates are middle/largest ≥0.05, smallest/middle ≤0.02 and p95 residual/span ≤0.05 (VW/ehs_spatial/measurements.py:113-117).
  **更正：**Rewrite the row as: X6 = ≥30 points, λ3/λ2 ≤0.2, angle >45°. That is 10x looser on planarity than the workcell, with no residual gate.
  位置：§4 #4 (主轴倾角 row)
- **说法：**All trackers assume rigid bodies with size change ≤2x, so a cable that deforms in place is misread by X6 as 'disappeared + new appearance' (X6W/fast_report/timeline.py:43).
  **问题：**SIZE_RATIO (:43) is used only when matching a 'moved' object (:192). Identity is decided by position alone: centroid within 3σ or box IoU ≥0.2, with no size check (:149). A matched instance is marked 'static' (:159). A cable that deforms in place therefore stays 'static' silently; it is not split into disappear + appear.
  **更正：**State the real failure: shape change is invisible to X6 because deformation is recorded as 'static'. The needed addition is a per-window shape/extent-change check, not relaxing a size limit.
  位置：§0 item 6, §3.2 可变形 row
- **说法：**lastObservedAt exists only in the plan (VW/docs/phase2/STREAMING-PLAN.md:117-118); code has only TSDF-block records.
  **问题：**X6's Tracker.timelines() already outputs per-entity first_seen_s, last_seen_s and state intervals: X6W/fast_report/timeline.py:247-266. The results include examples, e.g. cable g85 last_seen 29.46 (RN/fx-x6-windows-time-010/results.json:3080-3085).
  **更正：**Mark entity-level last-seen as existing in X6. The remaining gap is carrying it into the VW/platform contract.
  位置：§3.4 item 5, §3.5 缺
- **说法：**Parametric box accepted 58/87, RecGen 45/87, planes 2/4.
  **问题：**These are the sep5 sensitivity arm (views ≥5° apart). The spec arm sep15, which the memo's own step 12 requires, gives param box 5/9 and RecGen 3/9, and partitions 2/2: RN/fx-x7-object-models-001/results.json:6016-6060, 6083-6100. The X7 post cylinder was 'self-checked only: no post among the attempted objects' (:6158-6160).
  **更正：**Report sep15 as the primary numbers and sep5 as sensitivity. Note that no parametric cylinder has been tested on video data (neither X7 nor the unmerged m0-cylinder branch).
  位置：§1.3 X7 row, §4 #7
- **说法：**Measure width/depth with PS's minimum rotated rectangle (PS/scripts/scene_inventory.py:2666-2674).
  **问题：**Video already has two better versions. First, a gravity-aligned box with minAreaRect yaw and robust 1st/99th percentiles: VW/scripts/complete_video_objects.py:698-715. Second, X7 found minAreaRect yaw wrong when only two faces are visible and replaced it with a least-median surface-distance yaw search: X7W/fast_report/x7.py:292-301. Percentile extents also address the flying-point min/max problem.
  **更正：**Reuse X7's box_at yaw with 1/99 percentiles as the extent method. Keep measure_observed_points for scale/orientation gating. Drop the PS minimum_rotated_rectangle proposal.
  位置：§1.1 step 11, §5.2 V2
- **说法：**Reuse the photo radial-MAD 2.5 outlier removal (PS/scripts/scene_inventory.py:1735-1746).
  **问题：**_clean computes radius as horizontal distance from the floor-frame origin (points[:, :2] norm) and drops points within 0.15 m of it. That is a single-camera-at-origin assumption: scene_inventory measures per photo against frames[0]'s transform (:2572-2582). In a video map the origin is not the viewing camera.
  **更正：**Compute radial MAD against each keyframe's own camera centre. Or use video's multi-view spill vote instead: VW/scripts/build_video_object_map.py:70-81 and 235-244.
  位置：§1.1 step 5 (去拖影 third step)
- **说法：**Reusing the 3D overlap accumulator's MERGE 0.7 pass will fix fragmentation, with disputed pairs sent to the bidirectional associator.
  **问题：**accumulate() runs separately for each SAM label (VW/scripts/build_video_object_map.py:215-227; docstring :4-6 'same-label entity'). It cannot merge pieces that carry different words, which is the typical part fragmentation. The same file's docstring says the clique associator 'under-merges video' (:3). The associator also requires ≤3% relative depth agreement (VW/ehs_spatial/platform/spatial.py:257), tighter than DA3's ~5% (X6W/fast_report/timeline.py:36).
  **更正：**Specify a cross-label geometric merge, or reuse absorb()/PART=.25 (build_video_object_map.py:52,128-). State that the associator's 3% tolerance must be widened for DA3 before it can arbitrate video pairs.
  位置：§1.1 step 6, §5.2 V1 (补一轮合并 MERGE 0.7)
- **说法：**With estimated scale, all metric conclusions from the VW policy engine should be NEEDS_REVIEW, and metric predicates require trusted scale (VW/ehs_spatial/policy.py:63-330).
  **问题：**The platform engine returns INSUFFICIENT_EVIDENCE with missingEvidence 'metric_calibration:<id>' when scale is not operator_anchored (VW/ehs_spatial/platform/policy_engine.py:209-212). evaluate_video_policy's self-check asserts this for model_estimated scale (VW/scripts/evaluate_video_policy.py:119-120). Separately, ehs_spatial/policy.py:303 counts scale_source 'camera_height' or 'moge_anchor' as qualified. A SceneMap built with the assumed 1.6 m camera height would therefore get PASS/FAIL, not NEEDS_REVIEW.
  **更正：**Expect INSUFFICIENT_EVIDENCE through the platform engine. Flag policy.py:303 as a loophole: an assumed camera height passes as qualified scale.
  位置：§5.2 V5 预期, §2.2 clearance row
- **说法：**measurement_scale needs a new branch so that 'floor + assumed 1.6 m' becomes estimated.
  **问题：**The video path already classifies this. contract_scale maps assumed_camera_height to platform status model_estimated, device_metric to operator_anchored, and a failed lens gate to uncalibrated: VW/ehs_spatial/video.py:915-942. The two vocabularies differ: 'estimated' vs 'model_estimated'.
  **更正：**Reuse contract_scale's record as the video scale state and map it into measurement_scale's vocabulary. Do not fork a second classifier.
  位置：§1.1 step 2 (尺度状态要补一个来源分支)
- **说法：**The workcell bar is per-object range from its matching masks, point maps and cameras. The 288 px eval grid shows video's 504x280 is on par.
  **问题：**build-blender-ranges measures ONE reference frame per object and asserts exactly one matching view: WP/build-blender-ranges.py:86-89, 97-99. It does not union all three photos. The workcell measurement grid is Pi3X 518x518 (PS/outputs/candidate-evaluation/lucida-replica-01/manifest.json frames.canonical_shape_hw); 288 is only the model-comparison eval grid (PS/scripts/research/fit_blender_posts.py:162,193). The parity conclusion still holds.
  **更正：**Define the video equivalent as one measurement per keyframe mask, then aggregate across windows. Unioning a window's points into a min/max box adds pose/depth scatter; X1's view-centroid spread is 2.5-5.6 cm (RN/fx-x1-fps-004/results.json:6663-7307), not '约 3 cm'. Cite 518x518 for the resolution comparison.
  位置：§0 item 1, §4 #3
- **说法：**The workcell has a replacement rule (mean IoU not lower, depth p50 not higher); the right post was not replaced; video lacks any replacement rule.
  **问题：**First, fit_blender_posts labels the gate 'Specified after the initial candidate scores… exploratory, not predeclared', with promotes_source_scene False (PS/scripts/research/fit_blender_posts.py:20-23). Second, the published workcell scene replaces BOTH posts with cylinders (WP/README.md:9; blender-ranges-scene.json right_post → model-assets/blender-right_post.bin.gz). Third, the video mainline has its own rule: learned model > box > nothing, boxes off unless eye-reviewed, lower held-out residual wins (VW/scripts/merge_object_models.py docstring).
  **更正：**Call the workcell gate exploratory. Note that the published scene did not enforce it. Reconcile V3 with merge_object_models, whose preference is the opposite of X7's 'param box first'.
  位置：§2.2 参数化基本体 row, §4 #7
- **说法：**Extend box_free_space to all accepted models and use 'seen empty' as positive clearance evidence.
  **问题：**The mainline records that box_free_space's per-pixel free-space test 'is below its acceptance (plan section 5: boxes stay off until it passes)'. merge_object_models records its result but it 'admits and drops nothing' (VW/scripts/merge_object_models.py docstring items 3-4).
  **更正：**Mark the free-space test as unvalidated. Keep it as E0 evidence until it passes its own acceptance.
  位置：§2.2 空位证据 row
- **说法：**X8 labelled the AMG slices 'by human eye' and X8's 'manual audit' found the identity problems; X8 sets a/c serve as truth for V1.
  **问题：**The labels are agent-made, not human: 'agent-labelled: every pair looked at on sheets' (RN/fx-x8-jev-001/sets/a-audit.json:3) and 'agent-labelled (by looking at sheets/c-*.jpg)' (c-labels.json:3). Sets d.json and e.json were also written at 22:30/22:32, before the snapshot, so 'only a/b/c' is wrong. gemini-d.json appeared at 22:46, after the snapshot.
  **更正：**Call them agent labels and require a human spot-check before using them as V1 pass criteria. List sets a-e.
  位置：§1.1 steps 4 and 8, §5.2 V1 通过标准
- **说法：**X9's directory is empty, its script is only a SAM 3 tracker/DINOv2 cost probe, and deformable objects have no representation at all.
  **问题：**RN/fx-x9-reuse-001/stage1-me340.json was written at 22:34, before the 22:35 snapshot. The worktree has untracked modal_apps/x9_reuse.py and fast_report/x9_ladder.py. They implement a reuse ladder in which deformable/moving classes are re-detected on every frame and never reused: DEFORMABLE_HEADS {cable, cord, hose, strap, wire, rope, chain, …} at x9_reuse.py:40-51, and the x9_ladder.py header step 6. stage2 currently fails an assertion.
  **更正：**Correct the X9 status. Note that a name-based deformable class list exists (re-detect every frame, no geometry model). It lumps carts and forklifts in with cables, so the four mobility classes still need separating.
  位置：§1.3 X9 row, §0 item 6
- **说法：**Part rules reuse identity.py:84-106 (evidence-backed parent) plus the ≥0.5 on-accepted-model rule.
  **问题：**set_part_relation stores only source 'manual' (VW/ehs_spatial/platform/identity.py:101), so an automatic part rule cannot write through it. The PART_SHARE rule works only against accepted generated models and lifts points from one crop (the first payload view): VW/scripts/complete_video_objects.py:1112-1124.
  **更正：**Say that automatic part relations need a new 'observed' source in the platform contract. PART_SHARE is not a general part/whole rule for objects without models.
  位置：§1.1 step 9
- **说法：**Seed per-window vocabulary with the EHS list (PS/ehs_spatial/taxonomy.py:14-64).
  **问题：**The taxonomy is a robot-cell checklist: 'what a reviewer expects to find in a robot-cell photo' (:1-5), with LEFT/RIGHT entrance light curtains and kick plates (:30-31, :52-53) and 'the main payload in the cell' (:63). It assumes one fixed rectangular cell, which does not fit Sam's Club or Walmart aisles.
  **更正：**Use only the class names, not the per-item expected counts or LEFT/RIGHT slots. Add a retail/warehouse seed list.
  位置：§1.1 step 3 (EHS 清单作种子)
- **说法：**These photo judgements can be run once per registered static map or per keyframe.
  **问题：**Several of them hard-code a fixed rectangle cell or a single photo. The cell rectangle is Manhattan-framed and anchored on robot/machine keywords with a 7 m reach (PS/scripts/scene_inventory.py:1796-1809). Contact-edge footprints snap to the cell axis within 10° (:181-185). The agent prompt states 'every cell is a rectangular enclosure' (PS/ehs_spatial/agent.py:49). Clearance merges all fence fragments into one convex hull (PS/ehs_spatial/rules.py:94-99). reproject uses frames[0] (PS/ehs_spatial/reproject.py:37). Guard-line alignment, vocabulary and refinements use frames[0] (scene_inventory.py:2582, 2738, 2891). PS refine hard-codes frame_0001 and capture_frame_count=1 (PS/ehs_spatial/refine.py:205-213).
  **更正：**Flag each as needing de-rectangling or per-keyframe camera handling. On aisle videos, do not run the cell rectangle or the fence convex-hull merge; merge fences per connected run.
  位置：§2.2 墙与工位矩形, 薄结构落脚线, 护栏链 rows; §1.2 refine
- **说法：**PS floor fit (lowest plane, 20°, 3 cm, fixed seed) is reusable per registered shot.
  **问题：**_fit_floor needs an operator camera height or a MoGe override to set scale. It then rejects the fit if the scaled camera-height MAD exceeds 0.25 m (PS/ehs_spatial/geometry.py:256-275). scene_inventory defaults to --camera-height 1.5 and writes scale_source 'camera_height' (PS/scripts/scene_inventory.py:2554, 2540). These are measured-lens-height assumptions. VW's platform floor has per-view cross-view consistency gates better suited to windows (VW/ehs_spatial/platform/spatial.py:584-597).
  **更正：**Reuse only _ransac_floor_plane, as VW already does, plus the platform's cross-view gates. Do not reuse _fit_floor's scale step or its MAD gate on a walking camera.
  位置：§1.1 step 2 (复用照片的地面拟合)
- **说法：**X7's plane fit on the control panel (0.077±0.163°) is calibration evidence for slope.
  **问题：**X7's partition tilt uses a trimmed SVD plane with none of the workcell planarity or residual gates (X7W/fast_report/x7.py:195-203, 277-291). Its ± is a bootstrap spread, which X4 found understates the real spread (up to 8.6° between runs; RN/fx-x4-refine-006/results.json:47981). The X7 held-out gate also passes a constant confidence of 2.0 into evaluate (x7.py:489), which bypasses evaluate's confidence ≥1.5 check (VW/scripts/build_lingbot_object_model.py:79).
  **更正：**Treat the X7 number as unguarded. Re-run it through measure_observed_points' slope gates before counting it as calibration. Note the confidence bypass in §4 #2 as well.
  位置：§2.2 物理倾角 / 平面坡度 rows (X7 control panel 0.077±0.163°)
- **说法：**Segment every 5 fps keyframe (X1) while holding the V1 fragmentation target at ≤20%.
  **问题：**X1 reports that more frames produce 'many more objects … plus fragments of already-found objects': ME340 263 → 653 at 5 fps (RN/fx-x1-fps-004/results.json:8178). The thin-object recall gain 0.2 → 0.5 is ME340 only, over 10 objects.
  **更正：**Expect fragmentation to rise under densification. Pair the density change with the merge rule in the same V1 run, and qualify the thin-object figure as ME340-only.
  位置：§1.1 step 1, §5.2 V1
- **说法：**Each window has at least 8 and at most 40 new keyframes.
  **问题：**Windows also close at a cut or collapse regardless of count. The minimum span is 0.27 s on ME340 and 0.28 s on Walmart, which has 3 collapse closes (RN/fx-x6-windows-time-010/results.json:63 and 1_window_rule.per_video).
  **更正：**Say that at least 8 applies to content closes only; cut and collapse windows can be shorter.
  位置：§1.1 step 1 窗口规则
- **说法：**MIN_PIXELS=16 gives a side-length floor of about 3.5 cm × distance.
  **问题：**MIN_PIXELS is an area gate: 16 pixels on the stride-2 grid, which is 64 DA3 pixels (FB/fast_report/segment.py:17,316). A thin but long object, such as a 0.1 m light curtain at 5 m, passes.
  **更正：**Express the floor as an area, about 64 DA3 px ≈ (3.5 cm × z)² equivalent, and note that thin, long objects pass.
  位置：§1.2 table, 快速 3D 抬升 row
- **说法：**Coverage checking reuses the platform inventory review.
  **问题：**_review_inventory is one VLM model_review call per image (VW/ehs_spatial/platform/reconstruction.py:1138-1142). It is not a geometric or class-agnostic check. Per keyframe that means one VLM call per frame.
  **更正：**Describe it as a per-keyframe VLM completeness review and budget calls for it. Keep AMG/X10 as the geometric coverage signal.
  位置：§1.1 step 4 (覆盖检查复用平台清单复核)
- **说法：**X6 peak is 28-49 GiB per A100.
  **问题：**Across X6-010 runs the peak ranges from 27.7 to 50.8 GiB (RN/fx-x6-windows-time-010/results.json 2_timing_and_workers). A 5 fps object pass in a-core's layout is estimated at about 55 GiB on GPU0 (RN/fx-x1-fps-004/results.json:8178).
  **更正：**Use 28-51 GiB and budget about 55 GiB on GPU0 when segmenting every 5 fps keyframe.
  位置：§5.2 V1 成本

### 漏掉的可复用组件
- VW/scripts/build_video_object_map.py:70-81,235-244 consensus(): keeps only points that at least two views put in the same or an adjacent cell. It is a video-native spill filter and can replace the photo radial-MAD step. With --floor it already outputs footprintPlanNative/heightNative (p98)/baseNative (p2) (:250-257).
- VW/scripts/complete_video_objects.py:698-715 box_mesh (gravity box, minAreaRect yaw, 1/99 percentiles) and :68-70 FIT_GATE: held-out agreeing views judge fit and coverage, with ≥3 agreeing views and ICP rotation ≤15°. This is the mainline held-out model gate. X7W/fast_report/x7.py:292-301 has a yaw search that is better than minAreaRect.
- VW/scripts/merge_object_models.py: the mainline replacement rule (learned > box > nothing; boxes need an eye review; lower held-out residual wins). It must be reconciled with X7's 'param first' and with the workcell cylinder gate.
- VW/scripts/evaluate_video_policy.py: already takes confirmed entities (≥3 views) through the real platform engine, with per-entity uncertainty = depth noise at range + scale doubt (:21-40, 58-63, 112-132). It is the 'per-fact uncertainty band' the memo proposes; it is only not wired into the runner.
- VW/ehs_spatial/video.py:915-942 contract_scale: already maps assumed carry height to model_estimated, device poses to operator_anchored, and a failed lens gate to uncalibrated.
- VW/ehs_spatial/platform/model_quality.py:372 refine_model_pose (bounded pose refine with coarse=True), already chained with assess_model in VW/ehs_spatial/platform/reconstruction.py:1717-1723. It is the platform's own render-and-compare refine alongside X7's copy.
- VW/ehs_spatial/platform/coarse_model.py:80-140 coarse_open_frame: an open-frame primitive with rung count and bar fraction, for shelves, racks and ladders, where X7 accepted 0/4 shelves and says to 'fit uprights/beams'. It requires exactly two views (:16-31), a photo-set assumption.
- VW/ehs_spatial/platform/scene_measurements.py:17 fitted_plane, :80 fitted_bend, :173 surface_distance, :223 measure_scene (angle/inclination/bend/distance/occupancy on posed model surfaces). This is the platform's follow-up measurement path; its source is 'model_inference', so display-level only.
- VW/scripts/stitch_track_windows.py: same moving object keeps one identity across windows via mean mask IoU on overlapping frames + Hungarian. Relevant to actors and deformable objects, where 3D centroid association is weak.
- VW/scripts/filter_video_static_surfaces.py (free-space contradiction; occlusion is not a contradiction) and VW/scripts/drop_dynamic_entities.py (geometry-decided removal of moving entities).
- VW/scripts/report_runner/decide.py:435-472 static_filter: an entity whose views overlap person masks by ≥30% in most views moves to the dynamic layer. This is where a pushed cart splits from its static identity.
- X6W/fast_report/timeline.py:247-266 timelines(): entity-level first_seen_s/last_seen_s and state intervals already exist.
- X9 (untracked) X9W/fast_report/x9_ladder.py and modal_apps/x9_reuse.py:40-51: DEFORMABLE_HEADS always re-detected, plus a reuse ladder with depth-checked prediction and DINOv2 check. stage1-me340.json already exists.
- X8 set d (RN/fx-x8-jev-001/sets/d.json) includes q1 'cable or hose lying across the floor where people walk', with Gemini answers in gemini-d.json. It is a 2D judgement path for deformable trip hazards.
- VW/ehs_spatial/platform/spatial.py:540-600 platform floor with per-view cross-view angle and offset gates. It fits per-window multi-view floors better than PS _fit_floor.
- WP/blender-ranges-scene.json shows the workcell bar itself printing cart principal_axis_tilt 89.6° and guard 89.2° as 'available'. This confirms the memo's upright-cart-reads-~90° point inside the target bar; principal-axis tilt there is a shape descriptor, not a tilt judgement.
- RN/fx-x11-local-ba-003/me340-165.json appeared at 22:38, after the snapshot; its results are not reflected in the memo.
