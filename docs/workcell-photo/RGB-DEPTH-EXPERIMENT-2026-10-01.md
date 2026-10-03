# RGB 深度、按钮联合约束与离地量测实验

日期：2026-10-01。输入仍为原来的四张 BOR1 JPEG。此文记录真实失败、对照边界、模型产物和复现方法；不是新的厘米精度承诺。

**结论：本轮完成了两档深度推理与100/400次优化对照，仍未达到3cm目标。** 缓存分支增加优化次数没有改善留出表现；1036档未生成受支持的围栏模型；两侧光幕底边仍无可靠跨图对应。本轮报告交付目录为 `workcell-rgb-depth-report-2026-10-01-b`；首轮 `workcell-rgb-depth-report-2026-10-01-a` 保留作为历史产物。

## 1. 问题与不可混用的证据

- 按钮整体高 **0.10 m**、黄色主体最大直径 **0.085 m**、红色触发部件直径 **0.04 m** 是用户提供的输入约束。
- 围栏离地 **0.20 m**、两侧光幕离地 **0.24 m** 是独立评估真值。远程求解 payload 只发送 `reference`，不发送 `evaluation`；`_reference` 再做字段白名单。修改检查真值不能改变相机、尺度、模型或任何估计。
- 用户已经确认对比图里的**青色修正轮廓是好的**。这项确认对应图像轮廓；灰底是否包含安装唇边、隐藏面的形状、地面身份和真实厘米精度仍需单独验证。
- 四图联合 MapAnything 重建已经存在。本轮考察原图轮廓、三尺寸联合相机优化，以及两档网络输入分辨率，不能描述为此前只做了单图深度。
- 新深度分支有各自相机、尺度和世界坐标。只复用原图分割与对象身份；不把旧 native 点、旧包围盒或旧模型平移到新世界后当作新结果。

## 2. 实际执行记录

所有目录以下列根目录为基准：

`/Users/adam/Desktop/panoptes-public/research-notes/`

| 已保存运行 | 真实状态 | 可得结论 |
|---|---|---|
| `workcell-depth-resolution-2026-10-01-a` | 两档深度推理完成 | 同一四图、模型快照与 seed；得到518档和1036档各自的新深度/相机。不能据此判定量测精度改善。 |
| `workcell-depth-metrology-2026-10-01-a` | 入口失败，`ValueError` | Volume 符号链接路径中，未解析的根路径与已解析候选路径被错误比较。已统一路径解析，并增加符号链接回归检查；没有量测成绩。 |
| `workcell-depth-metrology-2026-10-01-b` | 入口失败，`ModuleNotFoundError` | 远程父进程直接导入 `scripts`，此前只给子进程设置了 `PYTHONPATH`。已修正远程入口路径；没有量测成绩。 |
| `workcell-depth-metrology-2026-10-01-c` | 部分分支完成，整次标记失败 | 518档完成几何及量测链；1036档没有找到受图像支持的成对围栏边，未伪造围栏。缓存联合拟合未通过验证。 |
| `workcell-depth-resolution-2026-10-01-b` | 两个worker在模型加载阶段失败 | 并行worker竞争同一个默认torch hub动态下载目录，出现缺失 `hubconf.py` / `dinov2.hub.cell_dino`。未完成推理，没有新的准确度结果。 |
| `workcell-depth-resolution-2026-10-01-c` | 两档深度推理完成 | 两个启动器使用镜像已有的 `TORCH_HOME=/opt/torch-hub`；已保存原图空间梯度归一化后的518/1036两档。 |
| `workcell-depth-metrology-2026-10-01-d` | 部分分支完成，整次标记失败 | 518档及缓存分支完成；400次仍未收敛且留出不通过。1036档仍无成对围栏边。 |

深度-b的worker加载错误与部署环境有关，不能当作1036档精度差的证据。该次容器阶段12.083 s，调用窗口268.815 s，两者不应混为推理耗时。

失败目录、输入 manifest 和支出 ledger 保留，不从报告历史中删除。子进程返回0只表示执行结束；`joint-reference.json` 的 `unsupported` 仍是失败候选。

### 2.1 第一轮深度耗时与相机异常

读取 `workcell-depth-resolution-2026-10-01-a/run.json`：

| 指标 | 518档：392×518 | 1036档：784×1036 |
|---|---:|---:|
| 四图神经推理 | 0.850 s | 1.168 s |
| 模型加载 | 30.223 s | 29.765 s |
| 序列化 | 4.180 s | 17.818 s |
| worker 内耗时 | 37.420 s | 51.077 s |
| 峰值已分配显存 | 8.382 GiB | 13.436 GiB |

两档各占一张 A100，并行执行；容器总耗时65.037 s，调用窗口245.006 s。这些数字是**深度实验阶段**，不含所有对象生成、RecGen和完整报告产出，不能替代历史334.60 s完整工作单元基线。

1036档换算回原图后的 `fx/fy` 为约 **0.800–0.818**，518档约 **0.998–1.002**。更高分辨率输出存在明显相机内参差异；像素更多不等价于几何更可靠。

1036档下游在 `freshGeometry` 失败：`No image-supported paired fence edges; cannot fabricate a fence model`。没有有效围栏结果可用于宣称误差变小。

### 2.2 第一轮缓存联合拟合：不能当作已标定

来源：`workcell-depth-metrology-2026-10-01-c/cached-joint-button/joint-reference.json`。

- 全量与三个留出拟合均达到 `max_nfev=100`，并报告尚未收敛。
- 全量 cost **2128.153→177.311**；场景点重投影P95 **3.581 px**，但按钮训练图最大误差仍为 **7.174 / 4.519 / 4.866 px**。
- 留出Photo2/3/4的误差为 **108.446 / 43.537 / 31.694 px**；明显超过4px门槛。
- 候选尺度 `0.6545407334 m/native` 仅用于预览；`mPerNative=null`，不能被量测路线拿作有效标尺。
- 未测灰底主轴/短轴被拟合为约 **17.65 / 6.16 cm**。这暴露了未观测形状在补偿，不能称为真实尺寸。
- 180条选中轨迹中169条仅跨两张照片；Photo2与Photo3/4共同轨迹仅5/10条，Photo3–4有109条。相机约束分布不均匀。
- 缓存联合构建耗时56.292 s；外层该子进程记录58.126 s。不能把“达到迭代上限”单独当作所有误差的原因，也不能在未收敛时宣称已证实不可修复。

## 3. 轮廓与联合优化的具体修正

### 原图轮廓

`_color_observation` 使用HSV关联颜色区域，再在小ROI上进行确定性的RGB GrabCut，保留旧HSV轮廓、初始RGB轮廓和被排除顶点。原图黄色宽度Photo2/3/4从135/98/111px变为121/96/101px；红色修正后为56/48/48px。用户确认青色轮廓表现可接受。

不能删除被线遮挡的极值点后，把剩余凸包的新内缩弦作为真实边界。`_support_validity` 使用删除前RGB hull与排除顶点定位受影响的方向，**只取消对应方向的约束**；不做全局腐蚀，也不把原本被拒绝的顶点恢复成真值。

保存观测中，Photo4黄色向右支撑，即零基 **index16**，被排除；Photo2/3不变。有效掩码同时作用于优化残差、可辨识性及全量/留出误差。每个颜色至少保留8/16个分布充分的方向；诊断保留排除方向及其误差。

### 三尺寸联合模型

- 红色有限圆柱、黄色锥台、灰色椭圆底缘模型共同参与相机和场景轨迹优化。灰底不是方盒；红帽实际圆润形状与灰底安装唇边仍是模型/端点假设。
- 内部高度分配、轴向、灰底宽深和朝向由拟合决定，不读取展示模型的固定比例。
- 消去场景点并边缘化形状未知量后，分别判断相机/尺度是否可辨识。灰底某个未知参数不可辨识，不再自动否定所有相机；真正的相机—尺度退化仍拒绝。
- 每个留出折的按钮初始位置、尺度与尺度边界只取训练照片；不继承全图按钮解或全图标尺位置。留出图只保留非按钮场景轨迹。
- 局部线性尺度sigma不是物理精度置信区间：轮廓支撑方向相关，形状偏差与真实端点歧义未被该数值覆盖。

## 4. 第二轮的对照边界

第二轮同时处理两个不同问题，必须按分支解释：

1. **缓存几何分支**：继续使用冻结的旧深度、旧相机/轨迹和相同原图轮廓；只将联合优化上限由100提高到400。它隔离“是否只是未收敛”的问题。验收阈值、真值隔离、形状模型不变。
2. **新518/1036分支**：深度梯度通过实际像素仿射变换换回原图空间，并按原518档像素足迹归一化；围栏边深度采样窗口也保持同一原图范围。提高分辨率后不再同时改变梯度阈值和邻域的物理图像范围。这些分支还包含新的推理结果及400次上限，不能把它们相对第一轮的差异只归因于优化次数。上游 `mask_edges` 仍依赖输出栅格；这里比较的是分辨率处理链，而非剥离所有后处理影响的纯神经网络对照。

### 第二轮完成结果

`workcell-depth-resolution-2026-10-01-c/run.json` 已确认两档推理完成：

| 指标 | 518档 | 1036档 |
|---|---:|---:|
| 四图神经推理 | 0.736 s | 0.810 s |
| 模型加载 | 24.030 s | 24.267 s |
| 序列化 | 3.520 s | 15.203 s |
| worker 内耗时 | 29.859 s | 42.162 s |
| 峰值已分配显存 | 8.382 GiB | 13.436 GiB |

两档并行，容器总耗时 **51.867 s**，调用窗口 **204.713 s**。模型与图像解码/序列化耗时仍与神经推理单独报告；这些不是完整对象报告的端到端时延。

`workcell-depth-metrology-2026-10-01-d` 量测结果：

- **新518档路线C**（原图实体底边＋局部地面，沿用整体高度标尺）：围栏 **16.441 cm**，相对20cm实测偏低 **3.559 cm**。仍是条件估计，相机与地面语义系统误差未被界定。
- **冻结缓存路线C**：围栏 **16.225 cm**，偏低 **3.775 cm**。两者差约0.216cm；这不是三尺寸联合标定的成功结果，也没有达到3cm。
- 两侧光幕均为 `unsupported`、`heightM=null`：没有稳定的同一RGB底边跨图重投影支持，不能以旧mask极值数值填补。
- **缓存100→400次对照**：全量cost **177.311→176.250**，仅降低约0.60%；留出误差由 **108.446/43.537/31.694px** 变为 **108.638/43.643/31.820px**，没有改善。全量及三个留出均再次达到400次上限，仍未报告收敛。可以说这次增加次数没有帮助，不能说优化已经收敛，也不能证明所有可能的几何模型都无解。
- **新518档联合拟合**同样不支持：全量cost176.300，留出 **108.659/43.651/31.823px**，全部达到400次上限。
- **1036档**有42,397个围栏输入点，拟合得到2个平面，留下6个边候选，但 **0对有效成对边**。因此没有围栏网格和离地结果；输入点数更多并未产生可测结构。

量测-d的应用记录为380.070s；外层函数计时 **383.505s**，调用窗口 **395.826s**。这是包含多个对照分支和重复联合优化的实验总耗时，**不是部署后的oneshot单次时延**。其中缓存联合子进程216.518s，新518分支136.230s。

本轮到此停止付费重跑。下一步应针对相机约束分布、部件形状/端点身份及围栏边配对开展新的有证据实验，不能继续单纯增加迭代次数或把候选尺度升级为测量尺度。

## 5. 模型和报告产物

`export_reference_candidate` 从保存的联合拟合参数导出三个独立命名部件：`red-actuator`、`yellow-body`、`gray-housing`。输出 `reference-candidate.glb` 与 `reference-candidate.json`。

- GLB保留对应联合相机的native世界坐标，单独旋转预览；不与使用不同相机的fresh floor/fence叠放。
- sidecar的 `metricScaleMPerNative` 只在联合验证通过时有值；不支持时为null，`candidateScaleForPreviewOnly` 单独标示。
- `knownDimensions` 是输入约束；灰底、红帽等拟合形状不会被界面标成已实测。
- 缺少拟合几何时只导出说明，不补造模型。
- fresh几何已经生成的 `floor-fitted.glb`、`fence-fitted.glb` 直接复用；新分支未生成的网格不会从旧世界复制补齐。
- 52对象完整旧报告保留。量测实验中的可浏览模型不自动证明原主场景已经完成全部精度升级。

最终报告交付目录：`/Users/adam/Desktop/panoptes-public/research-notes/workcell-rgb-depth-report-2026-10-01-b/`。公开报告入口保持 [workcell-photo-direct/metrology.html](https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/metrology.html)；部署提交与浏览器验收由发布记录确认，目录名本身不代表线上已更新。

## 6. 复现命令

先按[工作单元交接](HANDOFF.md)准备原始四图、冻结baseline、相机control与Modal模型缓存。以下重计算只通过临时 `modal run` 在 `2×A100-80GB` 执行；没有常驻deploy。输出目录必须不存在。

```bash
PHOTO_ROOT=/company/workcell/inputs
PHOTO_SOURCES="$PHOTO_ROOT/image_01.jpg,$PHOTO_ROOT/image_02.jpg,$PHOTO_ROOT/image_03.jpg,$PHOTO_ROOT/image_04.jpg"
RUN_ROOT=/company/workcell/runs

PYTHONPATH=.:scripts .venv/bin/python -m modal run modal_apps/workcell_depth_resolution.py \
  --sources "$PHOTO_SOURCES" --out "$RUN_ROOT/depth-normalized"

PYTHONPATH=.:scripts .venv/bin/python -m modal run modal_apps/workcell_guard_experiments.py \
  --mode metrology-depth \
  --baseline /company/workcell/baseline \
  --control /company/workcell/square-pixel-control \
  --sources "$PHOTO_SOURCES" \
  --measurements docs/workcell-photo/measurements-2026-10-01.json \
  --depth-run "$RUN_ROOT/depth-normalized/run.json" \
  --joint-max-nfev 400 \
  --out "$RUN_ROOT/metrology-normalized"

PYTHONPATH=.:scripts .venv/bin/python scripts/workcell_metrology_models.py \
  "$RUN_ROOT/metrology-normalized/cached-joint-button/joint-reference.json"
```

`--depth-run` 读取上一步保存的Volume结果路径。无需把所有稠密帧下载到Mac；量测重建留在云端。模型CLI可同时接收多个已有 `joint-reference.json`，每份结果旁创建新的 `reference-model/`。新的分支未生成该JSON时不应伪造输入。

CPU小规模回归检查：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_button_bundle.py
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_photo_metrology.py
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_depth_resolution.py
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_depth_metrology.py
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_metrology_models.py
```

每次付费作业都保存 `input-manifest.json` 与 `spend-ledger.json`。`actualBilledUsd` 仍为null时，只能称资源费率估算；调用窗口包含排队/启动，不能称为真实账单。失败调用也计入历史，不以成功分支耗时替代总耗时。


## 7. 七次调用的支出记录

下表来自上述各运行目录的 `spend-ledger.json`，包括失败调用。金额是按已记录资源费率乘调用窗口得到的估算；七条 `actualBilledUsd` 均为null，没有实际账单金额。

| 调用 | 调用窗口 | 窗口估算USD |
|---|---:|---:|
| depth-resolution-a | 245.006 s | 0.434935 |
| depth-metrology-a | 123.594 s | 0.219404 |
| depth-metrology-b | 7.992 s | 0.014188 |
| depth-metrology-c | 249.816 s | 0.443473 |
| depth-resolution-b | 268.815 s | 0.477200 |
| depth-resolution-c | 204.713 s | 0.363407 |
| depth-metrology-d | 395.826 s | 0.702670 |
| 合计 | 1495.762 s | **2.655277** |

合计时间是七次调用窗口的加和，不是用户等待时间或单个报告生成时间。排队、冷启动与失败重跑均包含其中，构建时间按ledger原说明处理。不能把2.655277美元称为真实扣费。
