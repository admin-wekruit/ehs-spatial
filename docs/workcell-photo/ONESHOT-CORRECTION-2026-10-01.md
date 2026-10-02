# 四张原图 oneshot：三维标尺、实体底边与完整模型修正

日期：2026-10-01（America/Chicago）。源码目录：`/Users/adam/.codex/worktrees/panoptes-workcell-photo-speed`，分支 `codex/workcell-photo-speed`。本文描述本轮实现与交接边界。

**运行完成，物理标定未通过。** 完整原图输出：`/Users/adam/Desktop/panoptes-public/research-notes/workcell-oneshot-correct-2026-10-01-b`。52 个对象模型、97 条照片证据；端到端 **366.93 s**，容器 **313.42 s**，含冷启动的调用 **350.81 s**。容器窗口估算 **$0.556**，调用窗口估算 **$0.623**，实际账单未知。新主报告没有受支持的米制比例，不能声称达到 sub-3 cm。公开报告同原 `workcell-photo-direct/` 地址更新。

首次 run-a 在末尾相机来源检查失败：原始像素变换 A 的 `0` 与数组转 JSON 后的 `0.0` 数值相同，但哈希不同。四张原图已复现并修复统一序列化；混合整数/浮点回归通过。该失败调用 **300.66 s**，调用窗口估算 **$0.5337**，未返回容器计时；已计入 ledger。失败时保留已完成模型的归档处理也已补齐。run-b 从四张原图重新执行成功。

两次调用窗口估算合计约 **$1.16**，包含可能未计费的调度时间，不是账单。`implementation-manifest.json` 保存各次执行源文件哈希；本轮最后的清理超时保护、检查、文档与界面耗时显示不改变 run-b 的推理结果。

## 1. 原始问题与修正

### 标尺必须是同一个三维物体

旧主链把整套按钮在一个估计前平面上的投影包围高度，直接对应到用户给定的 10 cm 轴向高度。红帽、黄色圆体和灰底具有深度；不同高度处的投影极值不一定来自同一条轴线。因此，即使用户给定的整体高度正确，“投影包围高度”仍不等于“实体轴向高度”。旧链也没有将两个直径共同约束这个三维模型。

现在 `workcell_photo_geometry.build(..., reference=...)` 调用共享的 `workcell_button_bundle.fit_reference_shape`，用完整透视投影同时约束：

- 红色触发按钮直径：4 cm。
- 黄色主体最大圆直径：8.5 cm。
- 红帽＋黄体＋灰底整体轴向高度：10 cm，不含安装支架。

本轮主链**固定原始 MapAnything 相机 K、pose 和原图至 canonical 的 A**，拟合按钮位置、轴向、共同尺度及部件形状参数；没有把另一套优化相机生成的按钮放进原相机世界。相机、原图轮廓哈希随 `anchor.referenceFit` 保存，`apply_measurements` 会核对相机来源。

三个尺寸是用户输入约束，不是独立重测的结果。各部件高度、黄色体截面变化、灰底宽深及外形仍是拟合假设。保留原有收敛、投影、可辨识性、逐照片留出检查及尺度一致性门槛；只有 `referenceFit.status == available` 才能输出 `mPerNative`。未通过的候选模型可保留为 native 形状假设，不能产生厘米测量。

### 点云最小值不能充当物理底端

旧光幕路径对整张 SAM 掩码内有效深度支持取最低值，再汇总各照片。这会把深度尾部、遮挡边界或被混入的表面当作实体底端；增加一个异常支持点即可改变距离。

本轮审计同时排除了一个容易混淆的解释：被检查的历史包围盒使用与地面相同的 up 轴，按同一支持集重算时，包围盒最低高度与实际支持最低高度一致；没有证据把低估归因于“旋转 AABB 合成了更低的虚拟角点”。历史 Photo 2 的虚拟包围盒中心确实有较大水平漂移，但漂移本身不能证明实例关联错误。

新主链调用 `workcell_photo_metrology.apply_source_clearances(root, sources)`：

1. 冻结的实例掩码与旧几何只关联照片区域。
2. 原始 RGB 寻找可见刚性下缘；围栏复用源照片横杆边线，黄色外壳复用 RGB 端边检测器。
3. 不同照片的可见边缘共同三角化，保留实际可见片段和未见间隙；原有 3 raw-pixel 重投影门槛不变。
4. 拟合共同局部地面，再计算实体源边缘到该平面的垂线。
5. 只有 `status == conditional` 且 `heightNative` 有值的结果可进入主卡片。单照片、遮挡或不一致支持保持 unknown。

旧最低值只保留在 `physicalClearances.observedEnvelopeBaselines` 供审计，不用于物理离地距离回填。没有用分位数替代实体底端，也没有把独立实测的 20/24 cm 送入估计。

### 原图端边漏检的一个实现原因

旧检测器要求一条连续长侧线支撑壳体轴线和末端宽度。纸张、反光和遮挡会将同一侧边分成多个 LSD 碎段，使照片 1/3 等视图无法提供第二个端边观察。

现在将同一黄色壳体旁、相互共线的可见侧边碎段组合。轴向支持仍要求累计可见长度达到原来的 4 倍粗略壳宽；共线误差仍不超过 1.5 raw px；未见间隙不计入长度。末端 face width 仍需附近实际可见的侧边片段。遮挡、颜色延续、端点接触、可见跨度和多视图门槛继续生效。合成碎段回归先失败、修正后通过；真实图能否补足多视角支持，要看本轮输出。

## 2. 一个地面、一个尺度、一条主链

```text
四张原始 JPEG + 三个已知按钮尺寸
  ├─ GPU 0：MapAnything → 深度、K、pose、像素变换 A；随后 OWLv2
  └─ GPU 1：SAM 3 → 物体分割与源掩码
       ↓
原生地面/围栏 + 三维按钮标尺拟合（CPU；与后续模型生成重叠）
       ↓
两个 GPU 并行 RecGen 物体模型 → 分片/对象目录
       ├─ 源实体底边 + 局部地面
       └─ A4 左右护板共享角模型拟合
       ↓ 两项完成后整合
已接受尺度/unknown → 全部对象报告 → GLB + 照片/点击/可旋转 3D 页面
```

- 调度仍为一次临时 `modal run`、一台 `2×A100-80GB` 容器；不常驻部署。RecGen 仅供内部使用。GPU、重几何求解在 Modal 执行，Mac 负责发起和封装。
- 三维标尺是唯一米制尺度来源。`apply_measurements` 仅核对并传播已接受的三维拟合；不再重新用 `10 cm / 旧包围高度` 覆盖它。
- `physicalClearances.ground.normal/offset` 是距离计算用的 native 平面。支持通过后同步 `geometry.floor`、`floor-reference.json`，并将已有 `floor-fitted.glb` 顶点投影到同一平面，保留 floor 节点身份和原有外形边界。报告只做一次 native→Z-up 刚体变换；下载场景用相应 Y-up 变换。
- A4 在 `a4/` 内拟合，只读取固定相机、SAM、guard masks 和独立保存的 `a1/` 初始化模型；与物理测量不共享写入文件。两个 future 完成后才调用 `structural_models` 整合模型和对象目录。
- A4 左右共角由用户“同规格同角”的先验施加；`measurementAngleDeg` 仍为 `null`，不能当成独立测得的真实角度。中间护板不参加左右共享参数拟合。源贴合 gate 未通过时不推广该候选。
- 不固定对象总数为 52 或 53；以本轮 `objects.json`、`scene-report.json` 的实际对象与模型为准。
- 修复 Modal 主进程只加 `/repo` 的导入问题：主进程现在同时加 `/repo/scripts`，避免在 RecGen 完成后因裸模块导入而失败。子进程继续使用显式 `PYTHONPATH` 与镜像内已有 `TORCH_HOME=/opt/torch-hub` 缓存。

前端 `PhotoReport.tsx` 与主报告共享 accepted scale。未标定时数值显示 unknown，仍可查看和下载 native GLB。已标定时用户调整整体参考尺寸，会把三个已知尺寸按相同比例调整，并统一作用于场景和显示量；这是比例假设的调整，不是新的相机/轮廓拟合。未知尺度不会因为修改输入框而自动变成有效尺度。只有实体源边缘结果能显示物理离地垂线。

## 3. 最短完整复现

先按 [HANDOFF.md](HANDOFF.md) 准备 Python/Modal 环境、已缓存权重、四张原始 JPEG 和 Three.js 0.178.0 资源。原始 JPEG 的顺序和哈希见该文档；Pages 的 canonical `photo-*.png` 不可替代。以下命令从源码仓库根目录执行，输出目录必须不存在。

```bash
(cd web && npx tsc --noEmit && npm exec vite build -- --config vite.photo.config.ts)

PYTHONPATH=.:scripts .venv/bin/python scripts/workcell_photo_oneshot.py \
  --images /company/workcell/inputs/image_01.jpg /company/workcell/inputs/image_02.jpg \
           /company/workcell/inputs/image_03.jpg /company/workcell/inputs/image_04.jpg \
  --measurements docs/workcell-photo/measurements-2026-10-01.json \
  --viewer-assets /company/workcell/current-report/page/viewer-assets \
  --out /company/workcell/runs/oneshot-correct-new
```

本机已存在的原图目录是 `/Users/adam/Desktop/panoptes-public/panoptes-serving/runs/user-bor1-02/input/`。本机解释器可使用 `/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python`。不要重用正在执行的 `workcell-oneshot-correct-2026-10-01-a` 输出目录。

入口在付费计算前冻结已构建 UI。它自己调用临时 Modal 作业；直接调用 `modal_apps/workcell_photo_all.py` 只取得云端档案，未完成最终页面封装。只有 `reference` 会传入云端标尺求解；测量配置中的 `evaluation` 留在下游用于误差对照。

运行完成后用 HTTP 查看：

```bash
python3 -m http.server 8765 --bind 127.0.0.1 \
  --directory /company/workcell/runs/oneshot-correct-new/page
```

浏览器访问 `http://127.0.0.1:8765/`。公开交付仍更新 [workcell-photo-direct/](https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/)；发布前须检查本次页面与模型，不能用旧在线页面证明新作业已完成。

## 4. 源码与输出契约

| 文件 / 入口 | 本轮职责 |
|---|---|
| `scripts/workcell_button_bundle.py::fit_reference_shape` | 固定相机下的完整三维参考物拟合、源轮廓与独立留出验证 |
| `scripts/workcell_photo_geometry.py::build` | 构建原生几何，调用同一个三维标尺求解器，保存 `anchor.referenceFit` |
| `scripts/workcell_metrology_models.py::reference_meshes` | 主报告和独立候选导出共用红/黄/灰参考物模型构造 |
| `scripts/workcell_photo_metrology.py::source_physical_clearances` | 返回 native 的 ground、对象源端点、来源照片、状态和诊断；不需要米制尺度 |
| `scripts/workcell_photo_metrology.py::apply_source_clearances` | 对象目录建好后的主链桥接；同步物理下缘结果与显示地面 |
| `scripts/workcell_photo_calibration.py::apply_measurements` | 传播被接受的尺度、核对尺寸与相机来源、保留 unknown |
| `scripts/workcell_photo_report.py::_ground_distance` | 按对象 ID 取实体端边，检查端点与显示地面一致，不读取旧最小值作回填 |
| `modal_apps/workcell_photo_all.py::reconstruct` | 单次两卡调度、并行物理/A4 拟合、模型整合和云端报告 |
| `scripts/workcell_photo_oneshot.py::run` | 原图入口、冻结 UI、调用与支出记录、评估、GLB 与 page 封装 |
| `web/src/PhotoReport.tsx` | 唯一尺度、物理下缘卡片/垂线、照片比较、旋转场景与 GLB 下载 |

完成时应核对这些输出：

- `geometry.json`：`anchor.referenceFit`、`floor`、`physicalClearances`。
- `physical-clearances.json`：`objects[].status/pointNative/footNative/heightNative/sourcePhotos`、`ground` 及 `diagnostics.objectEdges`。`observedEnvelopeBaselines` 不是新的物理测量。
- `raw-image-features-{1..4}.jpg`：原图 RGB 边缘、实际地面支持和可用垂线证据；不能只看 3D 网格判断端点正确。
- `structural-result.json`、`structural-*.jpg`、`guard-*-initializer.glb`：A4 推广条件、源贴合证据及初始化模型。
- `objects.json`、`scene-report.json`、`entity-*.glb`：本轮完整对象与可点击模型。
- `workcell-metric.glb` 或 `workcell-native.glb`：按接受状态导出，未标定文件不得标成米制。
- `measurement-evaluation.json`：冻结估计之后才与独立实测作差。围栏 20 cm、光幕 24 cm 均为评估值。
- `modal-call.json`、`modal-timing.json`、`stage-timing.json`、`one-shot.json`、`spend-ledger.json`：分别区分远程调用、容器、阶段和端到端时间；估算费用不等于账单。

## 5. 检查与验收

最小源码检查命令：

```bash
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_button_bundle.py
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_photo_metrology.py
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_photo_calibration.py
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_metrology_models.py
PYTHONPATH=.:scripts .venv/bin/python scripts/workcell_photo_oneshot.py --self-check
(cd web && node checks/photo-scale-check.mjs)
(cd web && npx tsc --noEmit && npm exec vite build -- --config vite.photo.config.ts)
git diff --check
```

检查覆盖：曲面投影包围高度与轴向高度的区别、三尺寸同一模型、留出与尺度门槛、原图像素变换、真实可见边缘片段、遮挡/未见间隙、ground 与模型一致、旧最低值不可回填、无尺度状态、相机来源绑定，以及独立实测不能改变模型或估计。metrology 碎侧线和地面导出回归已通过；最终整合检查、运行与浏览器证据由主会话补齐。合成检查通过不能证明现场达到 sub-3 cm。

本轮真实结果：

| 验收项 | 本次结果 |
|---|---|
| 完整四原图作业 | run-b exit 0；52 个模型；`one-shot.json` / `run-result.json` |
| 三维标尺 | 求解收敛、局部尺度可辨识；源按钮贴合与整图留出验证失败，`mPerNative=null` |
| 三图拟合最大偏差 | 照片 2/3/4：14.33 / 16.37 / 24.22 raw px |
| 留出照片最大偏差 | 照片 2/3/4：120.87 / 44.64 / 34.57 raw px；不能认定物理厘米精度 |
| 围栏底边 | 照片 3/4 支持，native 高度 0.2714604；地面显示与垂线一致；缺少受支持尺度，米值未知 |
| 右光幕 | 多视图候选没有附近受支持地面；其中边线跨未见间隙，不能宣称实体端点已正确恢复 |
| 左光幕 | 没有稳定的多视图共同端边；未知 |
| 完整交互 | 本地 52/52 模型加载；对比分界线实际拖动 50→64%；点击围栏、旋转与缩放、原生测量线通过 |
| GLB | `workcell-native.glb`，明确原生单位；不存在本次米制 GLB |
| A4 左右模型 | 源贴合门槛通过；共用模型角 62.015°，均值 IoU 左 0.6642→0.7538、右 0.5994→0.6796；仍非现场实测角 |
| sub-3 cm | **未达到已验证标准**，unknown 不计作达标样本 |

新固定相机候选尺度为 0.7403554 m/native，只保留在诊断，未投入主模型米制输出。其失败表明现有相机、轮廓和参数化按钮形状还不能同时解释各照片；本轮证据不足以把剩余误差只归因于相机或只归因于形状。下一步应分别检查跨视角相机与可测按钮外形，不能简单补一个比例，不能用 20/24 cm 真值改解。

仍未由代码修正直接解决的物理不确定性：原始 RGB 相机/深度误差、未确认的混合/遮挡端边、局部支撑面是否确为混凝土地面、灰底及不可见几何的模型偏差。图像残差、像素扰动范围和地面散差不能覆盖全部系统误差。应根据本轮真实源证据报告可估计项与失败项，再决定下一步；不能放宽门槛或把数值调向评估真值。
