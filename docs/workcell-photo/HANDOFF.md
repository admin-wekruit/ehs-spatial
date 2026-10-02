# 四照片工作单元：公开源码接手说明

**2026-10-02 物理下沿修正进行中：** [根因、修改与实验记录](PHYSICAL-BOTTOMS-2026-10-02.md)。此前统一的是模型卡尺与地面，尚未完成真实下沿写回主模型：光幕仍用照片 4 点云的 2% 高度分位数；左围栏曾漏入物理底边链；围栏仍沿用旧地面方向。正在用原图同部位证据替换这些来源。有限线段重合和低像素误差不能单独确认前后表面身份，候选模型不得据此自动提升为物理测量。

**2026-10-02 空间到语义实验已跑通：** [方法、实际结果、费用与复现](SEMANTIC-MATCH-2026-10-02.md)。同一 52 对象/97 观察上，双 A100 并行运行 PE-Core 和 SigLIP2，比较单图、多图及空间关联。六个固定查询可以找回对应目录对象，但光幕/围栏的自动命名仍有误判；不替换目录身份、模型或物理测量。公开报告 `#semantics` 可点击查询、查看真实支持裁剪及 EHS 缺失证据；未执行安全规则判定。

**2026-10-02 统一地面与模型卡尺已实现：** [实现、验证与使用说明](GROUND-CALIPER-2026-10-02.md)。现有 52 对象报告支持一点离地、两点距离/高差、三点采样范围和 JSON 导出；最终地面缓存、显示测量与 GLB 导出采用一致坐标/模型比例。本轮复用保存模型，未进行新 GPU 推理，未改善或验证物理重建精度。

**2026-10-02 高度入口修正：** 主报告无指定对象时默认照片 4 的右光幕，打开可旋转模型和底边离地对比。实验页原 `#models` 也先定位围栏/光幕高度入口；护板外观比较保留在后面的 `#appearance-models`。沿用既有模型条件估计（围栏 18.55 cm、右光幕 24.71 cm、高差 +6.16 cm），没有新推理或精度改善结论。

**云端证据存储：** 已使用现有 Supabase 项目 `WeKruit-VALET`（`unistzvhgvgjyzotwzxr`）的公开 `downloads` bucket，独立前缀 `panoptes-workcell-2026-10-02/`。首个 [围栏与光幕证据包](https://unistzvhgvgjyzotwzxr.supabase.co/storage/v1/object/public/downloads/panoptes-workcell-2026-10-02/workcell-height-evidence.zip) 已上传并匿名下载验证：1,482,481 bytes，SHA-256 `b5b1df1b00bf3cd4d6eab2299752e841040643cf9c13a7f015fd678928913421`。包含当前已公开的模型、照片 4、结构化报告及文件哈希；未迁移其他项目数据。公开 bucket 只用于已公开的报告产物，原始研究缓存和凭据不在此包。

**最新工程实验：** [Real2sim 对照结果](REAL2SIM-EXPERIMENT-2026-10-01.md)。相机均衡未提高留出精度；独立 COLMAP 分为两组。护板提供原图纹理 GLB 对照；光幕没有受支持的新面。未替换主场景几何或接受米制尺度。

更新：2026-10-01。范围为四张 BOR1 原图的对象级重建、模型内空间估计、可交互照片报告，以及护板角度实验。本文按当前代码和保存产物核对；代码、报告资产、原始照片、模型权重分别交接。

**模型高低直接估计：** [同一模型底面、同一地面的比较](MODEL-ENDPOINT-ESTIMATE-2026-10-01.md)。当前显示模型中右侧光幕确实更高；照片 4 按钮主体 8.5 cm 条件比例下，模型底端约 24.71 cm、邻近围栏下沿 18.55 cm，高差 +6.16 cm。这是模型内条件估计，不是源图物理端点已验证；旧负高差混用了深度边缘与模型底面。未新增 GPU 推理。

**原始点云检查：** [光幕下端、围栏与地面](POINTCLOUD-INSPECTION-2026-10-01.md)。报告现在可检查每张照片保存的推断点云，并与模型对照。右光幕显示包围盒与物理底边来自不同计算路径；查看器可用不等于测量正确。该检查复用 run-b，无新增 GPU 推理，原始运行时间仍为 366.93 s。

**最新主流程修正：** [三维标尺、实体底边与完整 oneshot](ONESHOT-CORRECTION-2026-10-01.md)。四原图重新生成 52 个可点击模型，端到端 366.93 s；A4 已接回主链，模型内底面/显示地面/垂线统一，但没有验证底面对应真实物理下沿。三尺寸标尺未通过跨照片验证，主报告不再用旧包围高度补回米制尺度；**尚未验证 sub-3 cm**。旧 17.13 / 14.36 / 15.71 cm 已退出当前主卡片。新模型下载为 native 单位；下文旧成绩与实测参考版本为历史记录。

**最新 RGB 深度实验：** [原图轮廓、三尺寸联合约束与518/1036分辨率对照](RGB-DEPTH-EXPERIMENT-2026-10-01.md)。两档深度与100/400次有界对照已完成，未达到3cm；新518档围栏条件估计16.44cm，两侧光幕未知，1036档没有受支持的围栏模型。用户已确认青色修正轮廓；联合候选模型仍未通过留出验证。

**此前测量实验：** [现有四图的按钮标尺 / 3 cm 实验](SUB3-EXPERIMENT-2026-10-01.md)。未要求新增实测相机参数；源观测对照、实现问题修正、原图叠线与支出单独保存。在线入口为 [测量实验页](https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/metrology.html)，页面可返回完整 52 对象 3D。实验页的候选不能自动当作主场景已经更新的准确模型；具体结果、是否达标以实验文档为准。

## 照片 workcell 核心 TODO：已完成 1 项，剩余 3 项（2026-10-02）

这四项是当前照片测量主线（第 1 项已完成模型卡尺实现），原先三视频物体层验收没有因此完成。已有端到端入口和并行执行继续复用。

1. [x] **通用模型测量层。** 已实现表面点取点、一点离地、两点距离与有符号高差、三点采样三角形高度范围、标尺同比试算及 JSON/GLB 一致导出。最终地面更新会刷新旧高度/垂足缓存。 复用现有 `objects[].groundDistance` 和统一 Z-up 地面，将当前指定光幕/围栏的模型面诊断推广到有 3D 几何的对象：明确测量面/边/点，统一坐标系、地面与尺度；倾斜面报告范围或明确测点；卡片、3D 测量线与导出同源。当前米制尺度未通过验证，条件模型估计不得提升为已验证物理尺寸。
2. [ ] **真实底边与左右一致性。** 用多视角同一物理底边重建光幕和围栏，替换光幕单图低分位包络、左右围栏不等价下沿。检查地面更新后的既有几何一致性；把用户提供的左右同高作为明确的先验/约束，并验证原图投影。不能把施加同高约束后得到的相等当作独立精度证据。
3. [ ] **绝对尺度与地面标定。** 继续验证按钮整体高度 10 cm、主体直径 8.5 cm、红帽直径 4 cm 对相机/尺度/地面的联合约束。当前三尺寸拟合仍未通过，accepted scale 为 null；8.5 cm 条件比例与4 cm交叉检查不等于完成联合标定。
4. [ ] **修正后的完整 oneshot 重复验证。** 将修正接回现有 oneshot，临时 Modal 2×A100 并行跑完整照片→模型→测量→报告；固定物理测量部位对比前后，分别报告已知样本、未参与调参的检验、左右一致性、精度、完整延迟和支出。保留新的模型、原图对照与可复现证据。当前尚未证明换一组照片稳定小于 3 cm。

### 最新左右检查：已经计算，尚未修正模型

[完整原始结果及文件哈希](LEFT-RIGHT-HEIGHT-2026-10-02.json)。以照片 4 面向工作单元的左右为准，全部读取当前 GLB 底面、同一保存地面、同一 0.6844702474894 m/native 条件比例；没有新 GPU 调用，也没有使用同高或20/24 cm目标进行拟合。

- 右（按钮侧）光幕 `post-box-1`：24.71 cm；左光幕 `post-box-2`：28.44 cm；左 − 右为 +3.73 cm。
- 紧邻各光幕的右围栏 `fence-0 / section-0-continued-3`：18.55 cm；左围栏 `fence-1 / section-1-continued-45`：24.34 cm；左 − 右为 +5.79 cm。
- 两个光幕的显示底边独立来自照片 4 分割点云的 2% 高度分位数。右围栏下沿为已有横杆的推断延伸；左围栏为可见结构低位包络假设，证据口径不等价。
- 已修正 `apply_source_clearances` 最终地面更新后的高度/垂足缓存。现有 fence 网格仍保留原始重建，待第 2 项按同一物理底边改善；不能把全部左右差值归因于缓存。
- 原图→3D 的底层模型未因本次读数/展示更新而改善。改变模型测点和条件尺度可能让读数接近现场值，不能据此宣称重建准确率提高。

## 1. 从哪里接手

| 内容 | 位置 / 状态 |
|---|---|
| 源代码 | https://github.com/admin-wekruit/ehs-spatial ，公开仓库，分支 `codex/workcell-photo-speed` |
| 当前工作目录 | `/Users/adam/.codex/worktrees/panoptes-workcell-photo-speed` |
| 固定交付版本 | 当前源码 release/tag `workcell-measured-reference-2026-10-01`；原始复现数据仍取 `workcell-photo-handoff-2026-09-30` |
| 静态报告资产仓库 | https://github.com/admin-wekruit/panoptes-workcell-report ，分支 `main` |
| 报告网址 | https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/ |
| 已保存的 A4 报告 | `/Users/adam/Desktop/panoptes-public/research-notes/workcell-guard-shared-report-2026-09-30/` |
| 当前实测标尺报告 | `/Users/adam/Desktop/panoptes-public/research-notes/workcell-measured-reference-2026-10-01-final/`；公开网站同网址更新 |
| 此前测量实验报告 | `/Users/adam/Desktop/panoptes-public/research-notes/workcell-sub3-report-2026-10-01-final/`；复用前一行完整模型并附加测量实验 |
| 最新 RGB 深度实验报告 | `/Users/adam/Desktop/panoptes-public/research-notes/workcell-rgb-depth-report-2026-10-01-b/`；包含两档深度、候选按钮模型、负结果及完整旧报告入口；实验结论未达3cm |

```bash
git clone --branch codex/workcell-photo-speed https://github.com/admin-wekruit/ehs-spatial.git
git clone https://github.com/admin-wekruit/panoptes-workcell-report.git
```

源码与 release 已公开，可匿名访问。原图和复现资料位于公开 release 附件；源 Git 不包含大数据、模型权重、凭据、`node_modules` 或推理虚拟环境。下载前用第 6 节命令核对固定 release、目标提交和附件；上方稳定网址不单独证明所看的版本。

## 2. 历史结果与明确限制（本轮主流程修正前）

- 完整原图运行实测 **334.60 秒**：一个临时 `2 × A100-80GB` 容器，容器内 180.32 秒，含冷启动的远程调用 317.28 秒。保存于 `workcell-three-boards-complete-2026-09-30-a/one-shot.json`、`modal-timing.json`。这是这一版四照片完整链路的历史基线；未对单 A100 测速，也未把本轮 CPU 修改算成新全流程成绩。
- 当前报告含 52 个对象记录、97 个来源观察、136 个关联网格节点。这些是产物统计，不是经现场核验的物体识别准确率。
- 左右护板的 A4 结构模型共用 **104.2702°** 参数，来自用户提供的“同规格同角”先验及照片轮廓拟合。缓存输入上的拟合/导出为 **44.04 秒**；这不含 MapAnything、SAM、OWLv2 和 RecGen 重跑。
- **真实折弯角不可唯一确定**：104.27°、120°、150°、174° 的离散候选均通过本次条件照片贴合门槛。这些采样值不是置信区间。`measurementAngleDeg` 仍为 `null`。中间护板仍无受支持的实物折弯角，不参与左右共享参数。
- 用户于 2026-10-01 提供现场尺寸：红色触发按钮直径 **4 cm**、圆形主体最大直径 **8.5 cm**、高度 **10 cm**，并确认整体高度正确，记录于 `docs/workcell-photo/measurements-2026-10-01.json`。高度对应红帽＋黄体＋灰底（不含支架）。已发布主场景仍使用高度确定统一尺度；两个直径用于对应参考部件建模。先前 A–D 实验没有将三个尺寸一起约束相机，不能将该实验称为已完成联合标定。
- A4 通过的是相对 A1 的轮廓拟合门槛。左板平均 IoU 0.66678 → 0.72784、右板 0.64792 → 0.67494；左板照片 4 从 0.76244 降为 0.72519，不能宣称所有视图均改善。
- `objects[].groundDistance` 保留各照片可见下缘到拟合地面的距离；至少两个有效来源视角，单视图保持未知。10 cm 高度标尺下，识别到的围栏下横杆约 **17.13 cm**；两侧光幕外壳下缘的多视角中位数约 **14.36 / 15.71 cm**。独立实测为围栏 **20 cm**、光幕 **24 cm**，分别偏低约 **2.87 / 9.64 / 8.29 cm**。实测离地距离仅用于评估，没有进入尺度、相机或地面拟合；照片间范围不是精度保证。
- 本轮实测标尺换算、参考部件建模与误差报告只使用保存结果，没有新增 GPU 推理调用，历史 334.60 秒完整运行成绩保持原定义。
- 这份静态报告没有自动完成 EHS 规则判定。浏览器可加载、照片对齐、模型内量距一致、真实物理精度分别验证；不能以一个代替另一个。


### 2026-10-01 实测标尺与独立评估

当前测量配置是 `docs/workcell-photo/measurements-2026-10-01.json`。公开网站、`geometry.calibration`、`measurement-evaluation.json` 和米制 GLB 使用同一尺度。`--measurements` 在付费运行前校验，报告重封装也接受同一参数。参考模型采用已知整体高度和两个不同部件的直径；部件之间的高度分配仍为建模假设。模型修改仅涉及参考按钮，其他对象的 native 几何与相机、地面保持原样。

不要把旧 `anchor.nativeWidth` 当成圆盘直径：按 10 cm 高度换算，旧整套轮廓宽约 12.52 cm，原算法没有单独可靠提取主圆盘或红帽直径。当前两个直径用于已知参考模型及照片投影诊断，尚未用于独立求尺度。卡片、3D 垂线和表格会随用户改动的换算高度同步变化；现场独立实测值不变。下载的评估 JSON 为提供的 10 cm 基准结果。

离地估计保留误差。围栏取现有已识别横杆；光幕对象为黄色外壳的观测下缘，使用全部有效照片的中位数，不能挑最接近 24 cm 的单张照片。被遮挡的底端、地面误差及未确认的现场量尺端点仍可能使比较失配。评估检查会改变实测目标，确认所有估计和模型保持不变。

2026-09-30 的原始复现包及其静态页面保留旧版本。先下载该包获得原图/相机/分割等输入，再 checkout 当前源码、按下文带 `--measurements` 的命令生成当前报告。不要把包中旧页面当作已应用新尺寸的结果。

## 3. 实际链路与代码边界

```text
4 张原始 JPEG + 急停组件 10 / 8.5 / 4 cm 三个命名尺寸
  scripts/workcell_photo_oneshot.py
    └─ modal_apps/workcell_photo_all.py::reconstruct
         GPU0: MapAnything → 相机/内参/点图 → OWLv2 检测框
         GPU1: SAM 3 文本分割 → 接收 OWLv2 框 → 车体分割
         CPU: 地面/围栏/标尺拟合 → floor-reference.json
         GPU0: RecGen 机器人各照片姿态 + 多视图模型
         GPU1: RecGen 车体 + 护板多视图模型
         CPU: 护板保形对齐、对象目录 → A4 / RGB 物理端边并行 → 统一报告
    └─ 本地证据图、统一尺度 GLB、静态 page/
         web/src/PhotoReport.tsx → 既有 ReportScene / NativeViewer

本轮原图 oneshot 已直接复用 A4 与结构模型整合；来源和模型角仍按先验记录。
```

| 模块 | 输入 → 输出 / 团队接入点 |
|---|---|
| `scripts/workcell_photo_oneshot.py::run` | 四个 `Path`、新输出目录、明确的测量配置或宽/高假设、固定 Three.js 资源目录 → 完整运行目录与 `page/`；新目录保护保留历史证据 |
| `modal_apps/workcell_photo_all.py::reconstruct` | 四个图片 bytes、文本词表、宽/高 → gzip tar bytes 与容器计时；实际 Modal 作业边界 |
| `scripts/workcell_map_worker.py` | 原图 → `frame_0001..0004.json.gz` 与 `photo-1..4.png`；保存 canonical raster、K、C2W、点图、置信度、有效 mask、原图到 canonical 变换 |
| `scripts/workcell_sam_worker.py` | `words.json`、原图、后到的 `cart-boxes.json` → `sam3.json`、`cart-masks.json`；RLE 为 Fortran 顺序 |
| `scripts/workcell_recgen_worker.py` | JSON job plan + `{robot,cart,guard}-input.npz` → 每组模型 NPZ；一个进程只加载一次 RecGen；验证形状、有限值和输出冲突 |
| `scripts/workcell_photo_geometry.py::build` | 帧、分割、四张原图、尺寸假设 → `floor-reference.json`、`geometry.json`、拟合围栏/地面 GLB、原图证据 |
| `scripts/workcell_photo_objects.py::build` | 保存几何/分割/模型 → `objects.json`、三块护板及额外观测表面；实体身份与来源观察 |
| `scripts/workcell_photo_report.py::build` | geometry + objects + GLB → `scene-report.json`、`entity-*.glb`；复用 `ehs_spatial.platform.contracts` 的 `Revision` 和 document 验证 |
| `fast_report/x7.py`、`fast_report/recgen_fast.py` | RecGen 固定代码/模型、推理参数、投影拟合；保形对齐为旋转 + 平移 + 统一缩放 |
| `scripts/workcell_guard_{controls,joint,dense,silhouette}.py` | A1 / COLMAP / LIMAP / A2-A3 / LoFTR / A4 实验；保留真实失败与不支持结果 |
| `scripts/workcell_guard_experiment_report.py::build` | baseline + controls + joint + viewer_assets + 可选 structural → 完整静态报告及 `experiments.html` |
| `scripts/workcell_photo_metrology.py::build` | 冻结帧/分割 + 原 JPEG + 标尺规格 + 可选相机 → A–D 测量实验、留出照片诊断、原图叠线；不读取检查真值 |
| `scripts/workcell_metrology_report.py::build` | 已发布完整报告 + 多轮冻结实验 + 检查真值 → 附带 `metrology.html` 的完整报告；保留失败结果与支出 |
| `web/src/PhotoReport.tsx` | 从相对 URL 加载 `scene-report.json`；对象选择、照片/模型对照、模型角注释、统一尺度和当前照片姿态 GLB 下载 |

公司已有平台可直接使用 `scene-report.json` 中的 `revision.document`、`assetURLs`、`objects`、`geometry`、`bendAnalysis`。`assetURLs` 将资产 ID 映射到相对文件；document 记录 SHA256。`SceneResources.resolveAsset` 是前端资产 URL 接口；资产服务迁移后须保留哈希与引用对应关系。此静态 UI 不需要平台 API，浏览器检查会断言没有 `/api/` 请求。

离地数据契约位于 `scene-report.json` 的 `objects[].groundDistance`：`byPhoto["1".."4"]` 保存 `valueNative`、`signedHeightNative`、`pointNative`、`footNative`、失败 `reason`；对象级 `rangeNative` 和 `sourcePhotos` 保存有效范围和来源；`feature` 可保存多视角识别到的围栏下横杆。端点已经变换到报告地面 Z-up 坐标，脚点位于 Z=0，长度保持 native 单位。负高度不截断为零，会保留 signedHeight 并将显示值设为 null。界面按当前照片选择样本，优先展示受支持的围栏 feature；用既有测量 renderer 绘线，乘同一 `nativeToMeters`，修改按钮标尺时数值与测量线统一缩放。

相机和几何原始坐标来自同一个 MapAnything 世界。报告变换到地面 Z-up 的 native 坐标，`coordinateFrames[].scale.nativeToMeters` 统一换算；GLB 导出为米制 Y-up。不要再次缩放相机或实体。机器人随照片切换对应姿态；下载按钮输出当前照片、当前选择标尺下的模型。离线 `workcell-metric.glb` 默认采用照片 4 的模型变体。

## 4. 外部资产和模型依赖

原始输入当前位于：

```text
/Users/adam/Desktop/panoptes-public/panoptes-serving/runs/user-bor1-02/input/image_01.jpg
/Users/adam/Desktop/panoptes-public/panoptes-serving/runs/user-bor1-02/input/image_02.jpg
/Users/adam/Desktop/panoptes-public/panoptes-serving/runs/user-bor1-02/input/image_03.jpg
/Users/adam/Desktop/panoptes-public/panoptes-serving/runs/user-bor1-02/input/image_04.jpg
```

本次实际读取文件并核对的 SHA256（顺序不可交换）：

```text
879ec826854e9f53c8b6e222b2d16d3e12b179ec60f4bf9c072eb5a4eccf38ac  image_01.jpg
6cd81ae4ae783c37d0a1129e39acb0d7064c46d9b51524130406642f8c7a3bea  image_02.jpg
0a8d4b8884867bf9f1836193db2fae0350a99e9e6b242eb9e01e20fda7c59438  image_03.jpg
e73931b6cdf68775ddafbc949c392c0ea2908f4dec671a4935e4c09361ae66ec  image_04.jpg
```

原图通过公开 release 的复现包 `inputs/image_01..04.jpg` 交接。Pages 提供的 `photo-*.png` 是 canonical 图片，不等价于原始 JPEG；原图不在 oneshot 返回的 tar 中，也不加入公共 Pages。不能仅 clone Pages 就声称可以重跑原图全部步骤。

| 依赖 | 当前代码 / 缓存约定 |
|---|---|
| MapAnything | `facebook/map-anything`；源码 `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`；Modal volume `mapanything-hf-cache` → `/v/map`，`HF_HOME=/v/map/huggingface` |
| SAM 3 | `facebook/sam3`，revision `3c879f39826c281e95690f02c7821c4de09afae7`；volume `sam3-hf-cache` → `/v/sam3`；使用 `/v/sam3/huggingface/hub` |
| OWLv2 | `google/owlv2-large-patch14-ensemble`；volume `panoptes-fb-models` → `/v/models`，缓存 `/v/models/hf` |
| RecGen | 源码 `TRI-ML/recgen@fe3c9315b439c50ada8b60c12b469d739fd722db`；模型 `TRI-ML/RecGen@bc0df7de2e43314830039a35a720731d4c4fac65`；`recgen_base.multiview_stereo` |
| RecGen 权重清单 | volume `panoptes-lucida-weights` → `/cache`；必须有 `recgen-weights-manifest.json`、清单引用的 snapshot 文件和 DINO 权重，路径在容器内可解析 |
| DINOv2 | 源码 `7764ea0f912e53c92e82eb78a2a1631e92725fc8` → `/opt/dinov2`；`dinov2_vitl14_reg` 权重由上述清单给定 |
| LoFTR 实验 | Kornia 0.8.2；官方镜像的 `loftr_indoor_ds_new.ckpt`；SHA256 `be9ff88b323ec27889114719f668ae41aff7034b56a4c4acbd46b8b180b87ed3`；`workcell_guard_dense.py` 下载后核验；不在普通 oneshot 内 |
| Three.js | 固定 `three@0.178.0` 的七个文件，代码验证 SHA256；可取复现包 `current-report/page/viewer-assets/` 或下方 npm 包提取命令；整个目录随 page 交付 |

模型代码与权重有各自许可。当前 RecGen 按仓库已记录的非商用/研究许可限制，只交接为内部研究链路；迁入公司服务器不自动取得商业授权。SAM 3 获取权重需要模型访问权限和相应条款许可。其他模型与镜像依赖也需保留各自来源/许可；本 handoff 没有赋予新使用权限。权重和凭据不提交 Git。

普通全流程设置 `HF_HUB_OFFLINE=1`，依赖上述缓存已经准备好。`modal setup` 只配置 Modal 身份，不会补齐模型权重。此代码没有完整的新账号缓存引导脚本；`x7.load_recgen` 严格读取已有清单。MapAnything、OWLv2 的代码调用未显式固定模型 revision，迁移时要保留实际缓存快照；仅固定 Python/Git 版本不足以字节级复现。

## 5. 本地环境、查看与复现

控制端项目声明 Python `>=3.12,<3.13`，锁文件 `uv.lock`；前端锁文件 `web/package-lock.json`。GPU 镜像中的 Python 3.11 主环境、3.11 MapAnything 和 3.10 RecGen 是分开的，不能合并成一个随意升级的 venv。

```bash
cd ehs-spatial
uv sync --extra photo --extra dev
cd web
npm ci
node node_modules/typescript/bin/tsc --noEmit
node node_modules/vite/bin/vite.js build --config vite.photo.config.ts
cd ..
```

`photo` extra 固定本地 `opencv-python-headless==5.0.0.93`、`scipy==1.18.0`、`trimesh==5.1.0`，均已进入 `uv.lock`。运行时通过当前解释器的 `python -m modal` 启动 Modal。UI 构建和七个固定 Three.js 资源先按 SHA256 校验并冻结到当前运行的 `report-ui/`，随后才开始推理，不依赖 Adam 本机路径或另一个网站目录。

优先复用复现包的 `current-report/page/viewer-assets/`。需要独立获取同一组字节时，在源仓库根目录执行：

```bash
mkdir -p /company/workcell/three-package /company/workcell/viewer-assets/addons/controls \
  /company/workcell/viewer-assets/addons/loaders /company/workcell/viewer-assets/addons/utils \
  /company/workcell/viewer-assets/addons/exporters
npm pack three@0.178.0 --pack-destination /company/workcell/three-package
tar -xzf /company/workcell/three-package/three-0.178.0.tgz -C /company/workcell/three-package
cp /company/workcell/three-package/package/LICENSE /company/workcell/viewer-assets/LICENSE
cp /company/workcell/three-package/package/build/three.core.js /company/workcell/viewer-assets/
cp /company/workcell/three-package/package/build/three.module.js /company/workcell/viewer-assets/
cp /company/workcell/three-package/package/examples/jsm/controls/OrbitControls.js /company/workcell/viewer-assets/addons/controls/
cp /company/workcell/three-package/package/examples/jsm/loaders/GLTFLoader.js /company/workcell/viewer-assets/addons/loaders/
cp /company/workcell/three-package/package/examples/jsm/utils/BufferGeometryUtils.js /company/workcell/viewer-assets/addons/utils/
cp /company/workcell/three-package/package/examples/jsm/exporters/GLTFExporter.js /company/workcell/viewer-assets/addons/exporters/
```

只查看已经交付的静态报告不需要 Python 模型依赖、Modal 或 GPU：

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory ../panoptes-workcell-report
```

浏览器打开 `http://127.0.0.1:8765/workcell-photo-direct/`。需要 HTTP 服务来加载 JSON/GLB，不能以 `file://` 打开代替验证。

完整运行入口与参数如下；输出必须是不存在的新目录。执行会启动付费 Modal 作业，权重缓存须先交接完毕：

```bash
PYTHONPATH=.:scripts .venv/bin/python scripts/workcell_photo_oneshot.py \
  --images /company/workcell/inputs/image_01.jpg /company/workcell/inputs/image_02.jpg \
           /company/workcell/inputs/image_03.jpg /company/workcell/inputs/image_04.jpg \
  --measurements docs/workcell-photo/measurements-2026-10-01.json \
  --viewer-assets /company/workcell/current-report/page/viewer-assets \
  --out /company/workcell/runs/raw-001
```

`--button-diameter-m` 是现有 CLI 的历史名称，代码实际将它作为整个组件宽度。`--viewer-assets` 为必填参数；使用独立提取资源时，改为 `/company/workcell/viewer-assets`。直接 `modal run modal_apps/workcell_photo_all.py` 只输出云端阶段的档案，不等价于已完成本地 page 封装。

现有 Modal 配置：应用 `workcell-photo-one-shot`，`gpu="A100-80GB:2"`，16 CPU、80 GiB 主存、timeout 1800 秒、retries 0、min_containers 0。使用临时 `modal run`；无常驻推理服务。查看计时应同时保留 `stage-timing.json`、`models-timing.json`、`modal-timing.json`、`one-shot.json` 和 `spend-ledger.json`。费用表是已记录资源单价的估计，`actualBilledUsd` 为 null；不当作账单。

### 保存数据上的 A4 实验与封装

以下变量都指团队收到的不可变运行目录；`ARTIFACTS` 当前本机对应 `/Users/adam/Desktop/panoptes-public/research-notes`。`INPUTS` 是上一节四张原图目录。A4 需要 baseline 的帧/分割/模型和 A1 初始化，不能从几个最终 GLB 反推这些输入。

```bash
export ARTIFACTS=/company/workcell/research-notes
export INPUTS=/company/workcell/inputs
export BASELINE="$ARTIFACTS/workcell-three-boards-complete-2026-09-30-a"
export CONTROLS="$ARTIFACTS/workcell-guard-controls-2026-09-30-a"
export JOINT="$ARTIFACTS/workcell-guard-joint-2026-09-30-b"
export DENSE="$ARTIFACTS/workcell-guard-dense-2026-09-30-a"
export REPLAY="$ARTIFACTS/workcell-guard-dense-replay-2026-09-30-a"
export STRUCTURAL="$ARTIFACTS/workcell-guard-silhouette-2026-09-30-b"
```

重新运行 A4 的实际 CLI（临时双 A100 作业；不是完整重建）：

```bash
PYTHONPATH=.:scripts .venv/bin/modal run modal_apps/workcell_guard_experiments.py \
  --baseline "$BASELINE" --out "$ARTIFACTS/a4-new" --mode silhouette \
  --alignment "$CONTROLS/A1-similarity" \
  --sources "$INPUTS/image_01.jpg,$INPUTS/image_02.jpg,$INPUTS/image_03.jpg,$INPUTS/image_04.jpg"
```

已保存 A4 的报告重新封装不调用 GPU：

```bash
PYTHONPATH=.:scripts .venv/bin/python scripts/workcell_guard_experiment_report.py \
  --baseline "$BASELINE" --controls "$CONTROLS" --joint "$JOINT" \
  --previous "$ARTIFACTS/workcell-guard-joint-2026-09-30-a" \
  --previous "$DENSE" --previous "$ARTIFACTS/workcell-guard-silhouette-2026-09-30-a" \
  --extra "$REPLAY" --structural "$STRUCTURAL" \
  --measurements docs/workcell-photo/measurements-2026-10-01.json \
  --viewer-assets /company/workcell/current-report/page/viewer-assets \
  --out "$ARTIFACTS/report-new"
```

该命令使用当前代码重新生成离地字段和 UI，复现本轮 A4 + 离地距离报告；只消费保存数据，先完成前端 build。封装共用 oneshot 的 UI/资源冻结逻辑。A1 保形对齐不会改变生成网格原有折弯角；A2/A3、COLMAP/LIMAP/LoFTR 产物属于实验对照，不能在实验中仅更新相机后把旧模型标成新测量。最初 controls 调用中的 LIMAP 失败已在日志保留，后续 joint-b 才有成功的原生 LIMAP 结果；全场两条稳定线、护板限定区域零条稳定线。

## 6. 公开 release 复现包

| 交付包 | 最小内容与能力 |
|---|---|
| 源代码 | 本分支及锁文件、检查脚本、本 handoff；不含模型权重和数据 |
| 浏览器报告 | 完整 `page/`：HTML、hashed `assets/`、`viewer-assets/`、JSON、照片、全部引用 GLB、实验页面及 `experiment-data/`；可静态托管、交互、下载当前 GLB |
| 中间结果复现包 | baseline 根文件与 controls / joint / dense / replay / structural / 历史实验目录，含 frame gzip、NPZ、SAM、模型、geometry、objects、manifest、timing、结构化运行结果；排除原始日志、既有压缩包及重复 page/report-ui；可检查或重算 CPU/A4 流程 |
| 原图包 | 上述四张 JPEG 与 SHA256；全流程和原图证据重算必需 |
| 权重包 | 四个 Modal volume 中实际用到的缓存和 RecGen/DINO manifest 引用；通过公司模型存储交接，遵循各自许可 |

原始复现包位于公开源仓库的固定 release `workcell-photo-handoff-2026-09-30`。发布流程使用按允许清单构造的 `workcell-photo-reproduction-2026-09-30.tar.gz` 和 `SHA256SUMS`；包含原图与上述实验资料，不含模型权重、凭据、原始日志、已有 archive 或重复历史 page/report-ui。包内 `MANIFEST.json` 保存逐文件相对路径和 SHA256，`current-report/validation.json` 保存最终报告验证记录。下列 `gh release view` 用于核对固定版本及远端附件，再下载并验证校验和。使用 `gh` 的下载示例需先配置 CLI；也可直接从公开 release 网页下载，无需仓库成员权限。

```bash
gh release view workcell-photo-handoff-2026-09-30 --repo admin-wekruit/ehs-spatial \
  --json tagName,targetCommitish,isDraft,url,assets
mkdir -p /company/workcell/download
gh release download workcell-photo-handoff-2026-09-30 --repo admin-wekruit/ehs-spatial \
  --pattern workcell-photo-reproduction-2026-09-30.tar.gz --pattern SHA256SUMS \
  --dir /company/workcell/download
cd /company/workcell/download
shasum -a 256 -c SHA256SUMS
tar -xzf workcell-photo-reproduction-2026-09-30.tar.gz -C /company/workcell
```

解压后的目录协议：

```text
/company/workcell/
  MANIFEST.json
  inputs/image_01.jpg ... image_04.jpg
  research-notes/workcell-three-boards-complete-2026-09-30-a/
  research-notes/workcell-guard-controls-2026-09-30-a/
  research-notes/workcell-guard-joint-2026-09-30-{a,b}/
  research-notes/workcell-guard-dense-2026-09-30-a/
  research-notes/workcell-guard-dense-replay-2026-09-30-a/
  research-notes/workcell-guard-silhouette-2026-09-30-{a,b}/
  current-report/page/
  current-report/validation.json
```

固定源码后重放上述 CPU 报告命令：

```bash
cd /company/ehs-spatial
git fetch origin tag workcell-measured-reference-2026-10-01
git switch --detach workcell-measured-reference-2026-10-01
git rev-parse HEAD
```

当前 GitHub Pages 的 canonical 照片、模型和展示证据足够看报告；它没有 frame gzip、NPZ、完整 SAM 原响应或四张原始 JPEG。这些由公开 release 附件传递，权重另行通过公司模型存储交接。下载包后也可直接运行 `python3 -m http.server 8765 --bind 127.0.0.1 --directory /company/workcell/current-report/page`，打开 `http://127.0.0.1:8765/`。`current-report/page/` 是静态交付目录；需要 Python 几何/相机检查时，先执行第 5 节保存数据上的报告封装，生成带帧文件的 `$ARTIFACTS/report-new`，再运行第 7 节检查。

要重现历史实验输入，用各目录 `input-manifest.json` 核对文件 SHA256，`implementation-manifest.json` 核对相应代码。历史清单会记录 Adam 的绝对路径，但身份应按哈希而非路径判断。`recgen-plan-*.json` 中的云容器临时路径也不是可直接在新机器运行的脚本；全流程重新创建这些计划。

## 7. 检查与验收

可运行的最小检查都已在仓库，按变更涉及范围执行。保存的历史检查结果不代表新机器已通过。

```bash
PYTHONPATH=.:scripts .venv/bin/python scripts/workcell_photo_oneshot.py --self-check
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_recgen_worker.py
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_photo_portability.py
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_ground_distance.py
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_photo_calibration.py
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_metric_export.py
PYTHONPATH=.:scripts .venv/bin/python scripts/workcell_photo_geometry_check.py "$BASELINE"
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_photo_objects.py "$BASELINE"
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_photo_report.py "$ARTIFACTS/report-new" --require-lights
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_vguard.py "$ARTIFACTS/report-new"
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_guard_joint.py
PYTHONPATH=.:scripts .venv/bin/python scripts/workcell_guard_silhouette.py --self-check
```

浏览器检查 `web/checks/photo-report-check.mjs` 使用 Playwright。它不在当前 `web/package.json` 中；接手者需提供自己已安装 Playwright 和 Chromium 的 QA 环境。可选 `PLAYWRIGHT_FROM` 指向该环境的 package.json；省略时按检查脚本自身路径解析依赖。可选 `VIEWER_ASSETS` 默认取待检查报告根目录的 `page/viewer-assets`；完整封装后不需要设置。

```bash
cd web
PLAYWRIGHT_FROM=/company/qa/package.json \
node checks/photo-report-check.mjs "$ARTIFACTS/report-new" "$ARTIFACTS/browser-check-new"
npm run check
```

检查覆盖对象列表/详情、照片与模型点选、鼠标/触摸/键盘拖动分界、四照片相机与机器人姿态切换、真实模型旋转、地面坐标轴、各护板模型折角标注、离地样本与测量线、未知值、20/40 cm 比例与导出、移动端、零 API 调用/零 runtime error。JSON 必须重新验证缓存，GLB URL 带 document revision，避免新 UI 混用旧模型。发布后还需对实际网站重新检查；本地测试通过不自动说明 CDN 内容已更新。公司 Linux 浏览器/GPU 环境需要重新运行同一检查，不能沿用 Mac 的通过记录。

几何验收仍需照片/模型对应关系和独立真实测量。报告数值、下载模型读回、可见底边来源、拟合地面、统一比例应形成同一条可追溯链；对被遮挡或来源不足的维度保持未知。

## 8. SSH 到公司 A100 时，CUDA 怎么工作

SSH 只建立远程 shell。你在远程服务器启动 Python 后，**远程** PyTorch 进程通过服务器的 NVIDIA 驱动和 CUDA runtime 在服务器 A100 上执行；Mac 不需要 NVIDIA GPU，也不会通过 SSH“传递本机 CUDA”。模型权重和图片可以留在服务器/公司存储，只传命令、日志和必要报告产物。

先在远程目标运行环境做一次真实检查：

```bash
ssh user@company-a100
nvidia-smi
CUDA_VISIBLE_DEVICES=0 /opt/mapanything/bin/python -c \
  'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.ones((32,32),device="cuda"); print((x@x).sum().item())'
```

`/opt/mapanything/bin/python` 是当前镜像中的约定路径；公司原生 venv 必须换成实际解释器。`CUDA_VISIBLE_DEVICES=1` 时，进程内 `cuda:0` 指映射后的第二张物理卡。当前 `_start` 正是用这一机制把 MapAnything/SAM 和后续 RecGen 两个进程分配到不同卡；CPU 几何进程使用空值禁止访问 GPU。PyTorch wheel 的 CUDA 版本、服务器驱动以及 xformers/spconv 等二进制扩展必须相容；以实际 import 和 GPU 张量运算验证。

可采用带 NVIDIA Container Toolkit 的 GPU 容器承载这些环境；容器通过 `--gpus` 获得主机 GPU，主机仍需 NVIDIA 驱动。现有仓库提供 Modal Image 的 build 配方，**没有已交付可直接运行的公司 Docker 镜像/Dockerfile，也没有 `--backend ssh` 或完整原生 GPU oneshot 入口**。

公司自托管所需的明确改动边界是 `workcell_photo_all.reconstruct` 外的执行与存储层：将 Modal 镜像/volume/远程调用改接现有平台作业环境；保留 worker 调用顺序、目录文件协议、进程 GPU 分配、输出与失败语义；按镜像配方提供三个隔离 Python 环境和固定缓存路径。现有几何、对象/报告构建函数和前端资产契约可直接复用。仅通过 SSH 运行当前 `workcell_photo_oneshot.py` 仍会调用 Modal，不会自动使用公司 A100。

若服务器只有一张 A100，当前硬编码双卡并行排程必须改为正确的单卡顺序和模型生命周期管理，并重新测量显存峰值/总耗时。不能把 `CUDA_VISIBLE_DEVICES=0` 一行当成双卡流水线已经完成单卡移植，也不能承诺 334.6 秒。目标输入和输出契约保持一致，单/双卡是调度及验证工作。

## 9. 接手时第一步

先确认公开 release 的版本，下载并验哈希，checkout 当前源码的固定 tag，再运行静态报告与不收费的结构/浏览器检查。重跑模型前接收权重清单。接公司平台时，以原图 SHA256 → 帧坐标/相机 → mask/对象身份 → 模型/观测几何 → 条件量距 → report JSON → 浏览器/导出 GLB 为验收顺序。完整新环境重跑、单 A100 迁移测速和物理精度验证尚未由本 handoff 代为完成。
