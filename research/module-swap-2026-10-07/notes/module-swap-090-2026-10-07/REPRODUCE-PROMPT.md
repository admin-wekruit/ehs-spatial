# 复现 prompt：工位照片 → 可商用三维报告（MVS + 补洞 + SAM 3D + 组装 v2），内部 GPU 环境

> 把这一整份交给内部环境里的工程师或 agent。它在自己的 GPU 机器上重跑 090 / 030 两个工位的完整流程，
> 结果要和 2026-10-07 已发布的报告对上（验收数字在最后）。没有 Modal、没有外网也能跑（离线包见 §3）。
> 只有一处需要你给的东西：Hugging Face token（`HF_TOKEN`，只在拉权重那一步用，不进代码、不进日志）。

## 0. 你要做什么

同一条工程流程（S1–S14，`module-swap-090/RUNBOOK.md`），三处模块换成可商用版本：

| 模块 | 原来 | 现在 | 许可 |
|---|---|---|---|
| S2 几何 | Pi3X（CC BY-NC） | MVS：DA3-BASE 起始 → RoMa v1 outdoor 稠密匹配 → numpy LM 光束法平差（`ba_scipy`，无 GPL）→ 两视图三角化（CERT 0.05）；然后 **掩码内补洞**：MoGe-3 单目深度按每张照片全局比例 + 每个物体局部偏移场对齐（`fill_geometry.py`） | Apache-2.0 / MIT / BSD-3 |
| S4 补全 | RecGen（TRI 非商用） | SAM 3D Objects，每张有掩码的照片各生成一个候选，按全部照片一致性择优（`completion_ab.py --stage sam3d` + `compare.py`） | SAM License（禁核、军工、ITAR；要向客户确认） |
| S5 组装 | 轮廓 + 边界 + 深度 | 同上 + **地面接触罚项**（`assemble_lucida_scene.floor_penalty`：最低点 0.5 百分位低于地面就罚，/ 物体高度，权重 2，只罚下沉） | 自有 |

其余各步（冻结、掩码、报告、平台导入、形状校验、地面、直线、平面立体、投影补掩码、跨视角、下沿、盒子、G1–G8）原样。

## 1. 代码（两个仓库，各一个分支）

| 仓库 | 分支 | 内容 |
|---|---|---|
| `admin-wekruit/panoptes-serving` | `feat/module-swap-2026-10-07` | `scripts/research/assemble_lucida_scene.py`（组装 v2），`scripts/research/build_capture_report.py`，`scripts/research/generate_sam3d_assets.py`（SAM 3D 在进程内，无 Modal），`research/module-swap-2026-10-07/`（本次全部脚本、表、说明：`module-swap-090/`，`geometry-backbone-ab/`（MVS 路线 + 公平评判器），`geometry-licence-ab-fair/`，`licence-clean-stack/licences.md`（许可清单与权重 sha256），`completion-licence-ab/completion_ab.py`，`completion-ab-090/compare.py`，`platform/publish-swap-20261007.py`） |
| `admin-wekruit/ehs-spatial` | `codex/workcell-photo-speed` | 平台 + 检查模块 + 离线套件：`scripts/onprem/`（`run_stage.py`、`fetch_weights*.py`、`airgap.sh`、`modal_stub/`），`scripts/workcell_checks/*.py`（floor / lines / plane_stereo / transfer / clearance / lower_edge / box_faces / obvious_errors / plane_facets），`scripts/workcell_shape_check.py`，`modal_apps/{sam3d_research.py, assemble_scene.py（在 serving）, workcell_layer_trial.py, geometry_clean_ab.py, bundle_adjust.py, publication_site.py}`，`docker/*.Dockerfile`，`web/`（查看器） |

脚本里的绝对路径（`/private/tmp/claude-501/.../scratchpad`、`/Users/adam/...`）是这台 Mac 的；每个脚本顶部的 `SP / RN / SERV / PLAT / WT` 常量改成你的目录即可，别的不用动。

## 2. 数据（不在 git 里，要拷过去）

| 目录 | 是什么 |
|---|---|
| `panoptes-serving/outputs/candidate-evaluation/lucida-replica-01/` | 090：3 张照片、冻结的 518 规范帧与 alpha、SAM 3 掩码（`evidence/objects/<物体>/<帧>/{mask.png, canonical_mask.npy}`）、地面掩码、RecGen 生成结果（原版对照用）、`manifest.json` |
| `panoptes-serving/outputs/candidate-evaluation/bor1-030-01/` | 030：同上，2 张照片 |
| `scratchpad/checks/da3fair-data/` | 公平评判器的固定输入：急停照片与种子、现场值、地面掩码、旧报告视图（`fair_ab_modal.build_data` 生成的那份） |
| `scratchpad/checks/clean-gpu/{090,030}/` | 可选：已算好的 RoMa 匹配 / 稠密 warp（`roma-*.npz`、`dense-*.npz`）和 MoGe-3 深度（`moge-frame_*.npz`）。不拷就在你的 GPU 上重算（§4 S2） |
| `panoptes-platform/.platform/{estop-scale-20261004/plan.json, cell030-20261005/estop-scale-v4.json}` 与 `scratchpad/mvs090/estop-scale2.json`、`scratchpad/mvs030/estop-scale.json` | 急停尺度文件（Pi3X 世界用已发布标定；MVS 世界用公平评判器自己的急停拟合） |
| `research-notes/workcell-lower-edge-2026-10-05/090-part-masks.json`、`scratchpad/sept/new-view.json`、`scratchpad/checks/cd84-view.json` | S11 下沿的点击部件掩码（只有 090）与已发布报告的视图（实体 ↔ 物体对应） |

现场值（只做验收，**永远不是输入**）：090 右罩壳下沿 24 cm、右围栏下横杆 20 cm；030 左右罩壳下沿各 24 cm；急停红钮 4 cm、黄体 8 cm、高 10 cm。

## 3. 环境与权重

- GPU：SAM 3D Objects 用了 A100-80GB（每个候选 ~10 s，权重 12 GB）；RoMa / DA3-BASE / MoGe-3 在 L4（24 GB）够。CPU 步骤 8 核 16 GB。
- 镜像：`ehs-spatial/docker/*.Dockerfile`（`panoptes-workcell-cpu`、`panoptes-workcell-gpu`、`sam3d`、`geometry` 等，依赖版本都 hash 固定）。离线：`scripts/onprem/airgap.sh save/load/wheelhouse`。
- 权重（只拉一次，之后 `HF_HUB_OFFLINE=1`）：
  ```bash
  # 在 ehs-spatial 检出里；HF_TOKEN 只在这两条命令的进程里，不写文件
  export HF_TOKEN=<你的新 token>            # 先在 HF 网页接受 facebook/sam-3d-objects 和 facebook/sam3 的许可
  python scripts/onprem/fetch_weights.py --cache /weights            # SAM 3（掩码只有在新照片上才需要；本次复现用冻结掩码，可跳）
  python scripts/onprem/fetch_weights_sam3d.py --cache /weights      # SAM 3D Objects @2e73555 + DINOv2 ViT-L/14 reg4
  python scripts/onprem/fetch_weights_geometry.py --cache /weights   # RoMa outdoor + DINOv2（sha256 见 licences.md）
  python scripts/onprem/fetch_weights_da3.py --cache /weights        # DA3-BASE @f4a6c9b
  # MoGe-3：Ruicheng/moge-3-vitl@184008f（公开，不需要 token）
  python scripts/onprem/fetch_weights_sam3d.py --cache /weights --verify
  unset HF_TOKEN
  ```
- 不要用：Pi3X、RecGen、DA3-LARGE、pycolmap 轮子（含 GPL 代码）。清单：`licence-clean-stack/licences.md`。

## 4. 跑（按步；每步的输出目录就是下一步的输入）

所有 "modal run X.py --flags" 在内部环境都换成 `python scripts/onprem/run_stage.py --weights /weights X.py --flags`：
Modal 装饰器被桩替换，函数体在本进程跑，不联网。

**S1 冻结 + S3 掩码**：直接用拷过来的运行目录（已冻结、掩码已有）。新照片才需要 `prepare_capture_evidence.py freeze` 和 SAM 3。

**S2 几何（GPU 一次，之后 CPU）**
1. RoMa 匹配 + DA3-BASE 起始深度 + MoGe-3 深度 → `checks/clean-gpu/<cell>/`：`geometry-backbone-ab/prod_route_modal.py` 的 route 阶段（函数体 = `ehs-spatial/modal_apps/geometry_clean_ab.py` 的 `roma_model` / `refine` / `mvs`）。
2. `python geometry-backbone-ab/mvs_route.py da3-base`（CPU，numpy）→ `checks/bbab-geom/<cell>-mvs-da3-base-padded/geometry`。
3. 公平评判器 + 导出：`python geometry-backbone-ab/backbone_ab_modal.py analyse <cell>-mvs-da3-base-padded`，`... export mvs-da3-base <cell>` → `checks/bbab-export-<cell>-mvs-scipyba`（名字按脚本里的常量）。
4. 补洞：`FILL_CELL=<cell> python module-swap-090/fill_geometry.py` → `checks/bbab-geom/<cell>-mvs-fill-padded`；再 `backbone_ab_modal.py analyse <cell>-mvs-fill-padded`、`export mvs-fill <cell>` → `checks/bbab-export-<cell>-mvs-fill`。
5. 门：`python module-swap-090/field_values_fill.py`：现场值与不补洞的 MVS 行相同（090 24.0 / 21.0；030 23.6 / 24.6），急停尺度逐位相同。不同 = 补洞碰了不该碰的像素，停。

**S4 补全（GPU）**
```bash
cd completion-licence-ab
AB_CELL=<cell> AB_RUN=checks/bbab-export-<cell>-mvs-fill AB_OUT=swap-runs/<cell>/mvs-fill-ab AB_FRAME=all \
  python scripts/onprem/run_stage.py --weights /weights completion_ab.py --stage sam3d          # 每张有掩码的照片一个候选
AB_CELL=<cell> AB_RUN=... AB_OUT=... python .../run_stage.py completion_ab.py --stage assemble --variants sam3d,sam3d-frame_0001,...   # 只列有内容的目录
cd ../completion-ab-090
CMP_NOTES=cmp-<cell>-mvs-fill CMP_SCALE=<该几何的急停尺度 json> AB_CELL=<cell> AB_RUN=... AB_OUT=... python compare.py <物体 id...>   # 择优 + 逐物体对比图
python module-swap-090/swap_generation.py <变体>      # 选中的候选落成 generation/<物体>/{object.ply, posed-object.ply, output.json}
python module-swap-090/pin_run.py swap-runs/<变体>    # 平台导入要的 sha256 pin + point_cloud.glb
```
（SAM 3D 的输入 = 照片 + 掩码 + 本几何点图；`generate_sam3d_assets.py --completion sam3d` 是单候选的生产入口，等价。）

**S5 组装 v2（CPU）**：`python scripts/onprem/run_stage.py panoptes-serving/modal_apps/assemble_scene.py --run swap-runs/<变体>` → `result/comparisons.json`（每个物体 `refinement.final.floor.lowest_native` 应 ≥ −0.02 原生单位；`floor_contact.weight` = 2）。

**S6 报告**：`python scripts/research/build_capture_report.py --run swap-runs/<变体> --label "<标题>"` → `public/scene.json`。

**S7 平台导入 + 导出**：平台数据库（Postgres；`.platform/identity-runtime-env.json` 指向它）跑着：`python platform/publish-swap-20261007.py <变体> swap-runs/<变体> <急停尺度 json> "<标题>"` → `.platform/publication-catalog/<id>` + `view.json`。

**S8–S14 检查**：`python module-swap-090/serve_export.py <变体>`，然后 `STAGES_CELL=<cell> python module-swap-090/run_stages.py <变体>`（内部用 `workcell_layer_trial.py` 跑原检查模块；on-prem 同样走 `run_stage.py`）。

**层 + 表**：`python module-swap-090/build_swap_layer.py <变体>` → 查看器的 `measurement-layer/<id>.json`；`compare_json.py` / `compare_layers.py`（090）、`tables_030.py`（030）。

**发布**：`module-swap-090/publish_site.sh <变体...>`（把目录编译成静态站：`scripts/prepare_publication_site.py` + `modal_apps/publication_site.py`；内部环境用自己的静态服务器放 `publication-http/`，查看器 `web/` 指向它）。

## 5. 验收（和 2026-10-07 的数字对）

| 项 | 090 | 030 |
|---|---|---|
| 几何层现场值（MVS + 补洞） | 右罩壳 24.0，右横杆 21.0；4 值 MAE 0.50 cm | 左 23.6，右 24.6 |
| 急停尺度（m / 原生） | 3.2371372 | 3.5616493 |
| 掩码内点图覆盖率（补洞前 → 后，9 / 8 物体均值） | 0.55 → 0.85 | 见 `fill-record.json` |
| 组装 v2 后最低点 < −2 cm 的物体 | 0 | 0 |
| G1–G8 明显错误物体（可商用组合） | 0 | 1（防护板，已知） |
| 轮廓 IoU 均值（可商用组合） | 0.823 | 0.762 |
| 浮点差异容忍 | 现场值 ±0.1 cm；IoU ±0.01；SAM 3D 种子 42 固定，仍可能有 ±0.02 的候选差异（则择优结果可能换照片，记下来） | 同 |

已知问题，不是复现失败：防护板（折板）SAM 3D 两个工位都做不好 → 测量盒或 S10 三视图平面重定深；报告级 G1 地面覆盖 75 %（平台侧地面面片）；030 右围栏 / 右防撞柱只有 1 张照片有掩码，悬空 8–26 cm。

## 6. 不要做的事

- 不改公平评判器、组装、检查模块的代码来"对上数字"。对不上就把差异记下来。
- 现场值不进任何输入。
- HF token 不写进仓库、镜像、日志；拉完权重 `unset`。
- Pi3X / RecGen 只作内部对照，不进交付。
