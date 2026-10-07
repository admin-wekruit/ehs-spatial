# 090 模块替换：同一流程，只换两个模块（2026-10-07）

原则：工程流程不动。只在两个模块接口处换输入，其余每一步调用原脚本、原参数。

## 原流程（lucida-replica-01 → 发布 4b58dbd2 → 测量层）

| 步 | 内容 | 入口 | 依赖模块 |
|---|---|---|---|
| S1 | 冻结照片，518 规范网格 | `panoptes-serving/scripts/research/prepare_capture_evidence.py freeze` | 无 |
| **S2** | **几何**：每张照片的点图 / 置信度 / 有效掩码 / 内参 / 位姿 → `geometry/frames/<f>/*.npy`；物体点云 `evidence/objects/<id>/<f>/points.npy`；地面 `evidence/floor.json` | Pi3X：`modal_apps/pi3x_geometry.py` ／ MVS：`scripts/onprem/run_stage.py geometry`（导出 `bbab-export-090-mvs-scipyba`，同一文件布局，契约检查通过） | **几何模块** |
| S3 | 掩码（SAM 3）→ `evidence/objects/` | 原样 | 无 |
| **S4** | **补全**：每个物体网格 + 位姿 → `generation/<id>/{object.ply, posed-object.ply, output.json}` | RecGen：`generate_lucida_assets.py` ／ SAM 3D：`generate_sam3d_assets.py --completion sam3d`（同一输出文件；每张照片各生成一候选，按全部照片的轮廓 + 深度一致性择优，择优在模块内部） | **补全模块** |
| S5 | 组装：9 自由度，全部照片的轮廓 + 深度 → `result/comparisons.json`（逐照片 IoU、深度残差、位姿） | `modal_apps/assemble_scene.py --run RUN`（= `assemble_lucida_scene.assemble`）。**2026-10-07 组装 v2**：`refine()` 加地面接触罚项 `floor_penalty`（最低点 0.5 百分位低于地面就罚，/ 物体高度，权重 2；只罚下沉）。五个版本各在 `swap-runs/v2/<版本>` 副本上重跑（`run_v2_chain.sh`），表里"组装v2"列 | 无 |
| S6 | 公开场景 + 报告文档 → `RUN/public` | `build_capture_report.py --run RUN --label` | 无 |
| S7 | 平台导入 + 急停尺度 + 导出发布视图 | `.platform/publish-cell030-20261005.py` 模式：`run_import(geometry_root=RUN)` → migrateScene v2 → setCalibration → export | 无 |
| S8 | 形状校验（三照片光度一致性） | `modal_apps/workcell_shape_check.py` | 无 |
| S9 | 通用照片检查：地面接触、直线、平面立体、投影补掩码、跨视角身份、MoGe 第二意见 | `modal_apps/workcell_view_checks.py --checks floor,lines,plane_stereo,transfer,clearance` | 无 |
| S10 | 防护板三视图平面重定深 + 折角 | `sept-guard-shape-2026-10-05/code/{sweep,compare,facets,redepth,retexture,build_layer}.py`；通用版 `scripts/workcell_checks/plane_facets.py` | 无 |
| S11 | 光幕罩壳下沿、围栏下横杆（现场 24 / 20） | `workcell-lower-edge-2026-10-05/runle.py` → `scripts/workcell_checks/lower_edge.py` | 无 |
| S12 | 盒子 / 面 / 高亮 / 平滑 | `workcell_view_checks.py --checks box_faces` → `workcell-boxes-2026-10-06/{build,integrate,finalize}.py`；`scripts/workcell_smooth_models.py` | 无 |
| S13 | 置信度 + 流程断点事实 | `cell030-sept-pipeline-2026-10-05/{confidence.py, refresh.py}` | 无 |
| S14 | 明显错误门 G1–G8 | `workcell_view_checks.py --checks obvious_errors` | 无 |

## 三个版本

| 版本 | S2 几何 | S4 补全 | 已有输出 |
|---|---|---|---|
| 只换补全 | Pi3X（lucida-replica-01） | SAM 3D | 候选 + 组装：`scratchpad/checks/completionAB-090/`，择优：`completion-ab-090-2026-10-06/results.json` |
| 只换几何（内部对照，RecGen 非商用） | MVS | RecGen | **要跑**：`generate_lucida_assets.py --root swap-runs/mvs-recgen`（A100） |
| 两个都换 | MVS | SAM 3D | 候选 + 组装：`scratchpad/mvs090/completion2/`（在等价的 MVS 导出上生成） |
| 发布 | 报告服务 = 平台 `modal_apps/publication_site.py`（导出目录编译后部署，`publish_site.sh`：三份已发布 + 列出的版本）；测量层 `build_swap_layer.py` → Pages `measurement-layer/<id>.json`；云端浏览器检查 `modal_apps/web_fixture_check.py` | | | |
| 030 | `FILL_CELL=030 fill_geometry.py` → `backbone_ab_modal.py analyse/export mvs-fill 030` → `completion_ab.py --stage sam3d`（AB_CELL=030，两个几何）→ `run_030_chain.sh`（三个版本：组装 v2 → 报告 → 本地发布 `030-v2-<版本>` → `STAGES_CELL=030 run_stages.py` → 层）→ `tables_030.py` | | | |
| 两个都换 + 几何模块第 2 版 | MVS + 掩码内补洞（`fill_geometry.py`：同一 MVS 点图，物体掩码内的空洞用 MoGe-3 补，契约不变；`checks/bbab-export-090-mvs-fill`） | SAM 3D（在补洞几何上重新生成：`swap-runs/mvs-fill-ab`） | `run_fill_chain.sh`：S4 择优 → `swap-runs/mvs-fill-sam3d` → S5–S14 原样 |

对比分两层交付：
1. **S2–S5 层**（JSON 直接对比，不需要发布）：逐物体逐照片轮廓 IoU、深度残差、边界误差、组装迭代与损失、模型尺寸与最低点、掩码内点覆盖率、几何层的现场值（罩壳 24 / 横杆 20 / 急停 4 / 8）、地面残差、相机高度。
2. **S6–S14 层**（需要本地平台导入，不发布）：形状校验、各项通用检查、盒子 / 置信度 / 事实 / 折角 / 下沿 / G1–G8。
