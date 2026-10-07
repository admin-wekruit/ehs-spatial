# 复现 prompt（直接 clone，所有模型本机 GPU 复现，不用 Modal）

> 把这份原样交给内部环境的工程师或 agent。仓库的 `main` 分支已经包含代码、冻结的输入数据、权重校验和与运行脚本；
> 只需要：一台有 GPU（≥ 24 GB；SAM 3D Objects 用过 A100-80GB）的 Linux 机器、Python 3.11/3.12、Postgres 16 或 Mongo（`PANOPTES_DATABASE_URL`，见 `docs/STORAGE.md`）、一个 Hugging Face token。
> 流程和已发布的报告完全一样，唯一不同是 GPU 在你那边（`scripts/onprem/run_stage.py` 让每个 Modal 应用在本进程跑）。

## 1. clone（一个仓库，main 分支）

```bash
git clone -b main https://github.com/admin-wekruit/ehs-spatial.git   # 平台 + 检查模块 + 离线套件 + 模型服务 + 流程 + 数据，全在这一个仓库（-b main：默认分支还没切）
```
运行环境就是 `docker/` 里的镜像（依赖全部 hash 固定）：`workcell-cpu.Dockerfile`（组装、报告、检查）、`workcell-gpu.Dockerfile`、`sam3d.Dockerfile`（SAM 3D Objects @f91db41，torch 2.4 cu121）、`geometry.Dockerfile` + `geometry-requirements.txt`（RoMa / DA3 / MoGe-3，torch 2.5.1）。
两条路：`docker build -f docker/<x>.Dockerfile`（上下文按 `scripts/onprem/stage_context.sh` 准备，离线用 `scripts/onprem/airgap.sh`），或者在一个 Python 3.11 venv 里 `pip install -r docker/geometry-requirements.txt` 再按 `sam3d.Dockerfile` 的 pip 行装 SAM 3D 的依赖。流程脚本用的就是这些环境（`uv sync --extra dev --extra photo` 装 CPU 侧依赖）。
`research/module-swap-2026-10-07/` 里有：`notes/`（全部脚本，和发布时逐字节相同，只是机器路径改成环境变量）、`data/`（冻结输入：两个工位的照片、518 规范帧、SAM 3 掩码、地面掩码、公平评判器的固定输入、急停尺度文件、RoMa / DA3-BASE / MoGe-3 的 GPU 中间结果、MVS 导出；共 0.5 GB）、`env.sh`、`run_all.sh`、`platform/publish-swap-20261007.py`。

## 2. 环境变量（一次）

```bash
cd ehs-spatial
export PANOPTES_WORKCELL=$PWD                       # 平台、检查模块、服务、流程都在这一个检出里
export PG=/usr/lib/postgresql/16/bin               # pg_ctl / pg_isready 的目录
source research/module-swap-2026-10-07/env.sh      # 其余变量从这里来：SWAP_SCRATCH=.../data，MODAL_RUN=本机 on-prem runner
```
`env.sh` 的默认就是"本机 GPU、不用 Modal"：`MODAL_RUN="$PY $PANOPTES_WORKCELL/scripts/onprem/run_stage.py --weights $WEIGHTS"`。
每个 `modal run APP.py --flags` 都经它变成本进程调用：Modal 装饰器被 `scripts/onprem/modal_stub` 替换，函数体在本机跑，不联网（`run_stage.py --self-test` 可自检）。

平台数据库（S7 导入 / 导出要）：建一个空 Postgres 16 库，写 `ehs-spatial/.platform/identity-runtime-env.json`，四个键：`PANOPTES_DATABASE_URL`（你的库）、`PANOPTES_BLOB_ROOT`（任一可写目录）、`PANOPTES_BLENDER_EXECUTABLE`（可填空串，本流程不用）、`PANOPTES_PAID_BUDGET_USD`（`0`）。`.platform/` 已在 `.gitignore`。

## 3. 权重（一次；只有这一步需要 token）

```bash
cd $PANOPTES_WORKCELL
export HF_TOKEN=<你的 token>     # 先在 huggingface.co 接受 facebook/sam-3d-objects 的许可（SAM License：禁核、军工、ITAR）
python scripts/onprem/fetch_weights_sam3d.py --cache $WEIGHTS       # SAM 3D Objects @2e73555（12.1 GB，mesh-only 文件）+ DINOv2 ViT-L/14 reg4
python scripts/onprem/fetch_weights_geometry.py --cache $WEIGHTS    # RoMa v1 outdoor + DINOv2 + DA3-BASE @f4a6c9b + MoGe-3 @184008f（都公开；sha256 校验，
                                                                    #   清单在 notes/licence-clean-stack-2026-10-06/licences.md）
python scripts/onprem/fetch_weights_sam3d.py --cache $WEIGHTS --verify
python scripts/onprem/fetch_weights_geometry.py --cache $WEIGHTS --verify
unset HF_TOKEN
```
不要拉、不要用：Pi3X、RecGen、DA3-LARGE、pycolmap 轮子。token 不写进任何文件。

## 4. 跑

```bash
cd ehs-spatial/research/module-swap-2026-10-07
./run_all.sh 090      # 约 40 分钟（SAM 3D 25 个候选 ≈ 5 分钟 GPU，其余 CPU）
./run_all.sh 030      # 约 25 分钟
```
每步幂等，中断后重跑接着来。`run_all.sh` 原样保留作参考；同样的步骤、脚本和跳过规则也有 CLI（`ehs_spatial/cli.py`，`uv sync` 后在 PATH 上）：
`panoptes run --cell 090 [--from S4a] [--only S2a] [--dry-run]`（`--dry-run` 只列步骤、完成状态和命令），`panoptes status --cell 090`；
每跑一步追加 `data/swap-runs/<cell>/ledger.json`（输入 sha256、起止时间、后端、模型 `model_info`、主机）。设 `SAM3D_BACKEND` / `GEOMETRY_MVS_BACKEND`=`http`
（`env.template`，`docs/BACKENDS-v1.md`）时 S4a / S2a 走 GPU 模型服务而不是本机 run_stage；不设则与 `run_all.sh` 完全一致。
它做的事（和 `notes/module-swap-090-2026-10-07/RUNBOOK.md` 的 S1–S14 一一对应）：

| 步 | 做什么 | GPU |
|---|---|---|
| S2a | RoMa 匹配 + DA3-BASE 起始 + MoGe-3 深度（`geometry_clean_ab.py --stage infer/refine/dense-infer`）。**中间结果已在 `data/checks/clean-gpu`、`clean-geom`、`da3fair-geom` 里，默认跳过**；想从照片完全重算就删掉这三个目录再跑 | 是（跳过时否） |
| S2b | numpy LM 光束法平差 + 两视图三角化（`mvs_route.py da3-base`，CERT 0.05） | 否 |
| S2c | 公平评判器（地面、急停尺度、现场值；`backbone_ab_modal.py analyse`）+ 导出运行目录 | 否 |
| S2d | MoGe-3 掩码内补洞（`fill_geometry.py`）→ 再评判 → 导出 → **门：现场值与不补洞一致** | 否 |
| S4 | SAM 3D Objects 每张有掩码的照片一个候选（`completion_ab.py --stage sam3d`，种子 42）→ 候选组装 → 全照片一致性择优（`compare.py`）→ `generation/` 契约（`swap_generation.py`）→ 导入 pin（`pin_run.py`） | 是 |
| S5–S6 | 组装 v2（地面接触罚项，`assemble_lucida_scene.py`）→ 报告（`build_capture_report.py`） | 否 |
| S7 | 平台导入 + 导出发布目录（`publish-swap-20261007.py`） | 否 |
| S8–S14 | 原检查模块：形状校验、地面、直线、平面立体、投影补掩码、跨视角、下沿、盒子、G1–G8、三视图平面（`run_stages.py`） | 否 |
| 层 / 表 | `build_swap_layer.py`（查看器的测量层）；`compare_json.py` / `compare_layers.py`（090）、`tables_030.py`（030） | 否 |

看报告：`publish_site.sh` 把导出目录编译成静态站（`scripts/prepare_publication_site.py`）；内部用任意静态服务器放 `publication-http/`，查看器 `ehs-spatial/web`（已构建的在 Pages 仓库 `admin-wekruit/panoptes-workcell-report` 的 `workcell-photo-direct/report/`）指向它，测量层放 `measurement-layer/<发布 id>.json`。

## 5. 验收（和 2026-10-07 发布的数字对）

| 项 | 090 | 030 |
|---|---|---|
| 补洞记录（`data/checks/bbab-geom/<cell>-mvs-fill-padded/fill-record.json`） | 25 条，补 48722 像素，与仓库里的逐字节相同 | `fill-record.json` |
| 几何层现场值（MVS + 补洞） | 右罩壳 24.0，右横杆 21.0；4 值 MAE 0.50 cm，最大 0.99 | 左 23.6，右 24.6 |
| 急停尺度（m / 原生） | 3.2371372 | 3.5616493 |
| 组装 v2 后最低点 < −2 cm 的物体 | 0 | 0 |
| G1–G8 明显错误物体 | 0 | 1（防护板，已知） |
| 轮廓 IoU 均值 | 0.823 | 0.762 |
| 容忍 | 现场值 ±0.1 cm；IoU ±0.01；SAM 3D 固定种子仍可能有 ±0.02 的候选差异，择优可能换照片——记下来，不算失败 | 同 |

已知问题，不是复现失败：防护板（折板）SAM 3D 两个工位都做不好；报告级 G1 地面覆盖 75 %（平台侧地面面片）；030 右围栏 / 右防撞柱只有 1 张照片有掩码，悬空 8–26 cm。

## 6. 规则

- 不改评判器、组装、检查模块的代码来"对上数字"；对不上就记差异。
- 现场值（24 / 20 / 24 / 24 cm，急停 4 / 8 / 10 cm）只做验收，不进任何输入。
- Pi3X / RecGen 不进交付（仓库里也没有它们的输出；原版的数字在 `notes/module-swap-090-2026-10-07/{table.md, layers-table.md}`）。
- 已发布的报告（4b58dbd2、cd84d3fb、25686138）不动。
