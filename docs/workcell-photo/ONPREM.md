# 工位照片流程 on-prem / Workcell photo pipeline on-prem (2026-10-05)

目标：在客户自己的 Linux GPU 服务器上跑完整流程，不用 Modal、不调 SaaS API、运行时不联网。
Goal: run the pipeline on a customer's own Linux GPU server: no Modal, no SaaS API, no internet at run time.

证据 / evidence: `research-notes/workcell-onprem-2026-10-05/` (results.json, README)；加固 / hardening：`research-notes/workcell-onprem-hardening-2026-10-05/`；
收尾（modal 存根、权重 overlay、torch.hub、预算闸、覆盖率规则…）/ final round: `research-notes/workcell-onprem-final-2026-10-05/`。

## 结论 / Bottom line

- **已证明 / proven**
  - `scripts/onprem/run_stage.py` 在本进程里运行任何 Modal app 的 local entrypoint：`import modal` 是 `scripts/onprem/modal_stub/modal.py`（存根，约 200 行），`Function.remote/map/spawn` → 原函数体，`@app.cls` 的方法先跑 `@enter`。**镜像里没有 modal 客户端**（五个镜像都不装，也就没有它拉进来的 10 个未固定依赖：anyio、cbor2、rich、synchronicity、toml、watchfiles…）；12 个 on-prem 阶段 app 在真 modal 被拦的情况下全部能导入。Runs any Modal app's entrypoint in-process against a stub `modal`; no image carries the modal client or its unpinned dependencies.
  - 030 地面检查 / 030 floor check：Mac 本地离线（网络被拦）与 `030-results.json` 最大差 8e-17 m；CPU 镜像（Modal 上 `from_dockerfile` 构建、`block_network=True`）里最大差 0.0 m，8 个物体状态全一致。Mac offline: max diff 8e-17 m; inside the CPU image with the network blocked: 0.0 m, all 8 statuses equal.
  - CPU 镜像里全部自检通过（run_stage、fetch_weights、orthonormalise_cameras、airgap.sh、clearance B、layer build、shape core、floor/lines/plane_stereo/transfer、MoGe 检查 CPU 部分、组装 venv 导入）。两个 venv 的 pip freeze = 之前的减去 modal 1.5.4 和它的 10 个依赖，其余逐项相同。All CPU self-tests pass in the image; the freeze is the previous one minus the modal client and its 10 dependencies.
  - 方法 B 间隙（`scripts/workcell_clearance_b.py`，从 research note 原样搬入，sha256 相同）在 CPU 镜像里断网跑 030：`results.json` 除耗时外与原 Modal 运行**完全相同**，5 张叠图逐字节相同。Clearance B in the CPU image, offline: identical to the Modal run except the seconds; overlays byte-equal.
  - RecGen 镜像构建成功，离线导入 RecGen/xformers/spconv 成功，pin 与 Modal 镜像相同。
  - RecGen job 副本（`/cache/jobs/<id>/input.npz`，含照片数据）现在落在 `run_stage.py --weights` 给 `/cache` 建的**运行私有 overlay**（TMPDIR 下），不进权重目录，阶段结束即删：A100 断网实测权重目录前后都只有 `recgen-weights-manifest.json`、`weights`，没有任何 `input.npz`，`/cache` 链接与 overlay 都已删除。Modal GPU 预算闸在 on-prem（`PANOPTES_ONPREM=1`）关闭：预先填满 3600 s 的 `gpu-budget.json` 下照样生成，调用记 `onprem: true`。
  - SAM 3D（`docker/sam3d.Dockerfile`）：**容器有网**时，Python 层 socket/DNS 审计在生成 030 left_post 全程 **0 次外连**（13 个事件全是本地子进程：nvidia-smi、ldconfig、gcc、git version…）；对照：不经 `run_stage.py` 的同一 `torch.hub.load` 会先查 github.com 并连它的 443 端口（两次运行：140.82.112.4、20.26.156.215）。位姿与之前的 on-prem 输出逐位相同，网格 Hausdorff 0.0015 原生单位。
  - `scripts/onprem/orthonormalise_cameras.py` 对 030 照片 1 的 bf16 原始位姿复现已发布的修正**逐位相同**（最大改动 1.79e-6，偏差 3.9e-6 → 6.0e-8）。
  - **RecGen 真实生成（加固后）**：`fetch_weights.py --models recgen` 在镜像默认 `HF_HUB_OFFLINE=1` 下自己联网下载，写出的 `recgen-weights-manifest.json`
    与报告用的逐字节相同（sha256 `78f89f80…`）；之后 A100、网络被拦，`run_stage.py --weights` 跑原驱动脚本生成 030 left_post（seed 42）。
    用原 `input.npz` 原样重放：位姿矩阵**逐位相同**，网格 108462 对 108467 顶点，posed 网格 Hausdorff 1.9e-3、Chamfer 1.1e-5（原生单位，×1.1166 ≈ 2.1 mm / 0.01 mm）；
    同一输入在镜像里跑两次也差这么多（108475 对 108474，Hausdorff 1.8e-3）——RecGen 的网格解码本身不是逐位确定的。驱动在镜像里重算的输入只有深度差 float32 末位（≤ 1.8e-5），
    位姿差 2.8e-4，Chamfer 8.1e-5（0.09 mm）。One real RecGen generation on-prem: pose bit-identical for the original input; the mesh matches within RecGen's own run-to-run noise (~2 mm Hausdorff, 0.01 mm Chamfer).
  - 加固 / hardening：构建上下文 `stage_context.sh` + `.dockerignore`（约 8 MB，原 serving 检出 24 GB）；基础镜像按 digest 固定（三个镜像都已按新 Dockerfile 在 Modal 上重建并通过：
    CPU 地面检查差 0.0 m，GPU 镜像 Pi3X 仍逐位相同、MoGe-3 / SAM 3 结果不变）；`airgap.sh save/load/wheelhouse`；经 `run_stage.py` 的花费账记为 `on-prem in-process`、无 USD；
    `floor_masks.py` 地面实例选择；`build_capture_report.py --pages-root`。
  - GPU 镜像（Modal 上 `from_dockerfile` 构建；一次 A100、`block_network=True`、权重来自 `fetch_weights.py` 填好的 volume）：
    Pi3X 重跑 030 两张照片，相机位姿、内参、点图、置信度、有效掩码**逐位相同**（最大差 0），点云 GLB sha256 相同；
    MoGe-3 检查合并尺度 1.28177 对原 1.28179，16/16 行状态不变，残差最大差 3.3e-4（原来 L4，现在 A100）；
    SAM 3 文本提示 "floor" 与产品运行（fal SAM 3.1）地面掩码 IoU 0.936 / 0.971。
    GPU image, one A100 with the network blocked: Pi3X bit-identical to the published 030 geometry; MoGe-3 check within
    3.3e-4 (pooled scale 1.7e-5); SAM 3 "floor" masks IoU 0.94 / 0.97 against the SaaS masks.
  - 但地面平面对掩码敏感：用 SAM 3 掩码重拟合，平面倾 1.38°，在观测到的地面上中位差 0.3 cm（p95 3.1 cm），相机处 4–5 cm；
    门口物体底部变化 −0.1…+4.8 cm（光幕立柱 +1.1 / +1.0，防撞柱 +0.4 / −0.1，围栏 +4.0，料车 +4.4），机器人 +11.6 cm。
    现有报告的地面掩码已冻结在 `evidence/floor-masks/`，复现报告不需要 SaaS；只有新采集才需要 SAM 3 分地面。
    But the floor fit is sensitive: refitting on the SAM 3 masks tilts the plane 1.38 deg (median 0.3 cm on the observed
    floor, 4–5 cm at the cameras; gate-object bottoms move -0.1…+4.8 cm). The published reports keep their frozen floor masks.
- **未就绪 / not yet**: 见最后两节 / see the last two sections (没有在 Linux 主机上真正 `docker build` / `docker run` 过、`docker save/load` 没执行；平台与发布服务没有容器化、报告网页托管、地面拟合的采样不稳定、许可 / no plain docker build or run on a Linux host yet, docker save/load not executed; platform and publication service not containerised, web hosting, floor-fit sampling instability, licences).

## 镜像 / Images

都从 `scripts/onprem/stage_context.sh SRC SERVING` 的上下文构建（本仓库 docker、scripts、modal_apps、configs + serving 的 scripts、modal_apps、ehs_spatial；约 8 MB）。基础镜像按 digest 固定，pip 全部 `==` 且 `--no-deps`，git 按 commit；**都不含 modal 客户端**。Every image builds from the `stage_context.sh` context; bases by digest, pip `==` with `--no-deps`, git by commit; no modal client.

| 镜像 image | Dockerfile | 内容 contents | 阶段 stages | 证明 proof (Modal `from_dockerfile`) |
|---|---|---|---|---|
| `panoptes-workcell-cpu` | `docker/workcell-cpu.Dockerfile` | `/opt/checks`（Py 3.11.10）、`/opt/assemble`（Py 3.12.6 + shapely）；`/check/shape_core.py`、`/check/workcell_checks`、`/check/clearance_b.py`；`/workcell/configs` | 检查、形状检查、间隙 B、急停尺度、测量层生成、组装、证据、地面掩码选择、相机正交化 | 断网：全部自检、地面检查差 0.0 m、间隙 B 相同 |
| `panoptes-workcell-gpu` | `docker/workcell-gpu.Dockerfile` | `/opt/pi3x`（torch 2.5.1 cu124）、`/opt/sam3`（torch 2.14 cu13）、`/opt/moge`（torch 2.13 cu13）、Pi3 代码 `/vendor/pi3` | Pi3X、SAM 3（物体/地面/急停/部位/掩码迁移）、MoGe-3 | 断网 A100：Pi3X 逐位相同、MoGe-3 16/16、SAM 3 地面 IoU 0.936 / 0.971 |
| `panoptes-recgen` | `docker/recgen.Dockerfile` | `/opt/recgen-py`（Py 3.10.13，torch 2.4.0 cu121）、RecGen `fe3c931`、DINOv2 `7764ea0` | RecGen 物体模型 | 断网 A100：030 left_post 生成，网格在 RecGen 自身运行间差异内 |
| `panoptes-workcell-da3` | `docker/da3.Dockerfile` | `/opt/da3`（torch 2.5.1 cu124）、DA3 代码 `/vendor/depth-anything-3` @ `3d835ec` | DA3 联合几何（Pi3X 的替代） | 断网 A100：与 A/B 的 DA3-LARGE-1.1 几何逐位相同（`geometry-licence-ab-fair-2026-10-05`）；本轮未改、未重跑 |
| `panoptes-sam3d` | `docker/sam3d.Dockerfile` | `/opt/sam3d`（Py 3.11.10，torch 2.5.1 cu121，pytorch3d / 扩展编译 sm_80 + sm_90）、SAM 3D `f91db411` + mesh-only patch、DINOv2 代码作 torch.hub 缓存 | SAM 3D 物体补全（RecGen 的许可干净替代） | 有网 A100 + socket 审计：0 次外连；位姿逐位相同 |

## 阶段表 / Stages

| # | 阶段 stage | 代码 code | 今天 today | on-prem | 外部依赖 → 替换 external → replacement | 模型 pin / licence | 证明 proof |
|---|---|---|---|---|---|---|---|
| 1 | 冻结照片 + 证据 capture freeze, evidence | serving `scripts/research/prepare_capture_evidence.py` | Mac CPU | CPU 镜像 `/opt/assemble`（加了 shapely 2.1.2：`floor` 子命令经 `ehs_spatial.geometry` 需要它） | 无 none | — | 未跑 not run；`ehs_spatial.geometry` 的导入在上一轮镜像里通过（同样的 pin；`floor_masks.py` 现在不再导入它） |
| 2 | 物体掩码 object masks | `scripts/workcell_sam_worker.py`（030：一键流程 `workcell_photo_all.py` 中 A100；车的框来自 OWLv2 + MapAnything 栅格）；090：SAM2.1 本机 MPS + 产品运行缓存掩码 | Modal A100 / Mac | GPU 镜像 `/opt/sam3`（文本提示）；框提示用 `workcell_mask_transfer.py` | fal SAM 3.1 + Gemini（产品运行）→ 本地 SAM 3 | `facebook/sam3@3c879f3`，SAM License（自定义，商用前须审） | 地面提示已证 floor prompt proven；物体未重跑 objects not re-run |
| 3 | 地面掩码 floor masks（两份报告） | 产品运行 `runs/user-bor1-02/inventory/sam/*__floor.json`（已冻结为 `evidence/floor-masks/`） | fal SAM 3.1 + Gemini（SaaS） | `/opt/sam3`：`words.json = ["floor"]`，再用 `scripts/onprem/floor_masks.py` 选实例（分数 ≥ 0.5 且 ≥ 60 % 抬升点在**最大一致集** RANSAC 地面 2×阈值内）→ `prepare_capture_evidence.py floor --spec` | SaaS → 本地 SAM 3 | 同上 | **IoU 0.936 / 0.971**；030：保留 0.93/0.97，去掉 0.57（近地面 12 %）和 0.45；**090 留出检验**：原"最低平面"选择规则倾 4.3°、把三个真地面实例全去掉（失败、无输出）；改为最大一致集后保留 0.95/0.94/0.92（99 % 近地面）、去掉 0.52（高出地面 22 cm），平面相对已发布倾 0.21°，相机高差 ≤ 0.8 cm（阈值未改；改平面规则是看了 090 之后做的，只有这两个工位验证过） |
| 4 | Pi3X 联合几何 joint geometry | serving `modal_apps/pi3x_geometry.py` + `scripts/candidate_pi3x_backend.py` | 030：Modal A100；090：Mac MPS fp32 | GPU 镜像 `/opt/pi3x`，代码 `/vendor/pi3` | HF 下载 → 权重缓存 | `yyfz233/Pi3X@bb1deea`，sha256 `69972d6e…`；代码 `9fa3ddb`（BSD 式）；**权重 CC-BY-NC-4.0 非商用** | **030 逐位相同 bit-identical** |
| 4b | 相机正交化 camera orthonormalisation（平台导入要求 \|RᵀR − I\| ≤ 1e-6） | `scripts/onprem/orthonormalise_cameras.py RUN/geometry`（通用：任何主干、任何 run） | 手工（030 照片 1） | CPU 镜像 `/opt/checks` | 无 | — | 自检；030 照片 1 bf16 原始位姿 → 与已发布修正**逐位相同**（改动 1.79e-6）；原文件保留为 `camera_to_world.<keep-as>.npy`，记录写 `camera-orthonormalisation.json`，改动 > 1e-4 拒绝。090、030 现有几何都 ≤ 1e-6，不需要改 |
| 4' | DA3 联合几何（Pi3X 的替代）DA3 joint geometry | serving `scripts/candidate_geometry_backend.py`（DA3Runner） | Modal A100（A/B） | `docker/da3.Dockerfile` `/opt/da3`；权重 `scripts/onprem/fetch_weights_da3.py` | HF → 权重目录 | `depth-anything/DA3-LARGE-1.1`（**许可有争议**）/ `DA3-BASE`（Apache-2.0）；代码 `3d835ec` Apache-2.0 | 断网 A100 与 A/B 几何逐位相同（`geometry-licence-ab-fair-2026-10-05`） |
| 5 | RecGen 物体模型 | serving `modal_apps/lucida_assets.py` + `scripts/research/generate_lucida_assets.py` | Modal A100（`block_network`） | `docker/recgen.Dockerfile`，`run_stage.py --weights`（`/cache` → 运行私有 overlay，job 副本不进权重目录、结束即删）；`PANOPTES_ONPREM=1` 关掉 Modal GPU 预算闸 | Modal Volume → 本地目录 | `TRI-ML/RecGen@bc0df7d`，代码 `fe3c931`，DINOv2 `7764ea0`；**代码 TRI 非商用，权重 CC-BY-NC-4.0** | **030 left_post 已离线生成**（两轮）：网格在 RecGen 自身运行间差异内（Hausdorff ≈ 2 mm）；本轮位姿与前两轮差 8.2e-4（同一输入，见 research note） |
| 5' | SAM 3D 物体补全（RecGen 的替代）SAM 3D Objects | `modal_apps/sam3d_research.py` `SAM3DObjects`（驱动脚本经 `run_stage.py`） | Modal A100（A/B） | `docker/sam3d.Dockerfile` `/opt/sam3d`；权重 `scripts/onprem/fetch_weights_sam3d.py`；torch.hub → 镜像内 DINOv2 代码（`run_stage.py` 强制 `source='local'`） | HF（gated）→ 权重目录 | `facebook/sam-3d-objects@2e73555`，SAM License；代码 `f91db411` | 有网 A100，socket 审计 0 次外连；left_post 位姿逐位相同 |
| 6 | 场景组装 assembly | serving `modal_apps/assemble_scene.py` | Modal CPU | CPU 镜像 `/opt/assemble` | 无 | — | 导入 imports only |
| 7 | 公开场景与报告文档 public scene + report doc | serving `scripts/research/build_capture_report.py` | Mac | CPU 镜像 `/opt/assemble`，`--pages-root DIR` 或 `PANOPTES_PAGES_ROOT`（放 pack-model.py、build-unified-data.py 两个文件；默认仍是本机 Pages 仓库） | 写死路径 → 参数 | — | 030 用 `--pages-root`（两文件副本）在副本上重建：`public/` 与已发布的逐字节相同 |
| 8 | 平台导入/发布/导出 platform import, publish, export | panoptes-platform FastAPI `ehs_spatial.platform.runtime` + PostgreSQL + 本地 blob | Mac | 同一应用部署在客户服务器（`docs/platform/OPERATIONS.md` "Run locally"：`PANOPTES_BLOB_BACKEND=local`、`PANOPTES_EXECUTOR_BACKEND=local`） | 无（本地 blob、本地执行器） | — | 未跑（本任务不启动 Postgres） |
| 9 | 发布服务 publication service | `modal_apps/publication_site.py` → `ehs_spatial.platform.publication_site:create_app` | Modal 部署（公开 URL） | 同一 FastAPI 应用用 uvicorn 在本地跑 | Modal → uvicorn | — | 未跑 |
| 10 | 测量层 + 报告网页 measurement layer + web | `web/`（Vite，`VITE_API_ORIGIN`、`VITE_PUBLICATION_ID`）；测量层 JSON | GitHub Pages | 静态目录（nginx 或平台 `PANOPTES_WEB_ROOT`） | GitHub Pages → 本地静态；`web/src/styles.css` 的 Google Fonts 离线时退回系统字体 | — | 未跑 |
| 11 | 检查 checks: floor, lines, plane_stereo, transfer | `modal_apps/workcell_view_checks.py` + `scripts/workcell_checks/*` | Modal CPU | CPU 镜像 `/opt/checks` | 读 GitHub Pages 测量层 + Modal 发布服务 → 本地平台 URL，或 `run_stage.py --offline` 包 | — | **floor 已证（相等）**；四个模块自检 |
| 12 | 形状检查 shape check | `modal_apps/workcell_shape_check.py` | Modal CPU | `/opt/checks`（scipy 1.14.1，Modal 解析为 1.17.1） | 同 11 | — | 自检（含新的覆盖率规则：1 附近 NaN、或 1 两侧有限点没到 ±0.05、或有限点 < 60 % → inconclusive）；四份已存结果重算见 research note |
| 13 | 掩码迁移 mask transfer | `modal_apps/workcell_mask_transfer.py` | Modal L4 | `/opt/sam3` | Modal Volume → 权重缓存 | SAM 3 同上 | 同一 venv/权重，未单独跑 |
| 14 | MoGe-3 第二意见 | `modal_apps/workcell_moge_check.py` | Modal L4 | `/opt/moge` | HF → 权重缓存 | `Ruicheng/moge-3-vitl@184008f`，MIT；代码 MoGe `74fbce0`、FlexGEMM `b2fadb2`（MIT） | **030 尺度差 1.7e-5，残差差 ≤ 3.3e-4（16/16 行）** |
| 15 | 浏览器检查 browser check | `modal_apps/workcell_browser_check.py` | Modal CPU，克隆 GitHub Pages 仓库 | 未就绪 | GitHub + mcr Playwright 镜像 | — | 否 |
| 16 | 急停尺度 e-stop scale | `modal_apps/workcell_estop_mask.py`（SAM 3 文本提示，1008 px 瓦片，候选掩码）→ `scripts/workcell_estop_scale.py`（只用掩码；种子窗口以红钮宽度为单位） | Modal L4 + Mac CPU | GPU 镜像 `/opt/sam3`（掩码）+ CPU 镜像 `/opt/checks`（尺度） | 无 | SAM 3 同上；唯一特殊参考，尺度不变 | **GPU 镜像离线（L4，网络拦截）：75 个候选掩码与 `modal run` 逐字节相同、分数差 0，由它们求的尺度 090 1.286551 / 030 1.117679 与之相同**；CPU 镜像离线 +0.038 % / +0.097 %（相对发布值 1.28601 / 1.11662）。research-notes `workcell-onprem-fixes-2026-10-05/` |
| 17 | 间隙 B（罩壳下沿、围栏下横梁）clearance B | `modal_apps/workcell_clearance_b.py` → `scripts/workcell_clearance_b.py`（原 research note 代码原样搬入） | Modal CPU | CPU 镜像 `/opt/checks`（`/check/clearance_b.py`） | 同 11 | — | **断网 030：除耗时外与 Modal 运行相同，叠图逐字节相同** |

## 硬件 / Hardware

- GPU：1 张 NVIDIA **sm_80–sm_90**（Ampere A100 / A10 / RTX A6000，Ada L4 / L40S / RTX 6000 Ada，Hopper H100）：bf16 autocast 要 ≥ sm_80；SAM 3D 的 pytorch3d 等扩展编译了 sm_80 + sm_90（sm_86/89 跑 sm_80 代码）；RecGen 镜像 `TORCH_CUDA_ARCH_LIST=8.0`（只影响现场 JIT 编译的扩展）。
  **实测只在 A100（sm_80，全部 GPU 阶段）和 L4（sm_89，SAM 3 急停掩码）上**；sm_86、sm_90 没测过。One sm_80-sm_90 GPU; tested on A100 (all GPU stages) and L4 (SAM 3) only.
- 显存峰值：SAM 3D 17.2 GiB（torch 预留）、RecGen 12.2 GB、DA3-LARGE-1.1 3.7 GiB、SAM 3 2.2 GiB；建议 ≥ 24 GB（证明用 A100-80GB）。离线 GPU 证明里 Pi3X 26–35 s、MoGe-3 检查 22–35 s、SAM 3 13–23 s、RecGen 一个物体 48 s、SAM 3D 一个物体 83–106 s（都含加载）。
- 驱动：`/opt/sam3`、`/opt/moge` 是 cu13 torch 轮子，须 ≥ 580；`/opt/pi3x`、`/opt/da3`（cu124）、RecGen、SAM 3D（cu121）≥ 525；+ NVIDIA Container Toolkit。Driver ≥ 580 for the cu13 venvs.
- CPU：8 核、32 GB 内存（检查在 Modal 上用 8 CPU / 16 GiB）。
- 权重 weights（`fetch_weights*.py` 写的 manifest 里的字节数 / bytes from the manifests）：

  | 模型 model | GB | 用于 used by |
  |---|---|---|
  | Pi3X `yyfz233/Pi3X@bb1deea` | 5.44 | 联合几何 |
  | SAM 3 `facebook/sam3@3c879f3` | 3.45 | 物体 / 地面 / 急停 / 部位掩码 |
  | MoGe-3 `Ruicheng/moge-3-vitl@184008f` | 1.48 | 第二意见深度 |
  | RecGen `TRI-ML/RecGen@bc0df7d` + DINOv2 ViT-L/14 reg4 | 6.54（5.32 + 1.22） | 物体模型 |
  | SAM 3D Objects `facebook/sam-3d-objects@2e73555`（只取 mesh 的 11 个文件）+ DINOv2 | 13.33（12.11 + 1.22） | 物体补全（替代 RecGen） |
  | DA3-LARGE-1.1 / DA3-BASE | 1.64 / 0.54 | 联合几何（替代 Pi3X） |
  | 现行一套 current set（Pi3X + SAM 3 + MoGe-3 + RecGen） | **16.9** | |
  | 全部 all | **32.4** | |
- 镜像大小没有量过（这里没有 Docker daemon）；估计 GPU 约 20 GB、RecGen 约 15 GB，SAM 3D（CUDA devel 基础 + 全套研究环境）预计更大。Image sizes were not measured.

## 许可 / Licences

技术上能离线跑 ≠ 可以商用。不是法律意见。On-prem ≠ licensed for commercial use. Not legal advice.

| 模型 model | 作用 role | 权重 weights | 代码 code | 商用 commercial |
|---|---|---|---|---|
| Pi3X | 联合几何（现行） | **CC-BY-NC-4.0**（HF 卡确认） | yyfz/Pi3 `9fa3ddb`，BSD 式 | **否**，须作者许可 |
| RecGen | 物体模型（现行） | **CC-BY-NC-4.0** | **Toyota Research Institute 非商用**（TRI-NC）；TRELLIS 部分 MIT | **否** |
| DA3-LARGE-1.1 | 联合几何（候选） | **有争议**：HF 模型卡 apache-2.0，官方 GitHub README 模型表 CC BY-NC 4.0（含我们 pin 的 `3d835ec`；前代 DA3-LARGE 卡也是 cc-by-nc-4.0） | Apache-2.0 | 按非商用对待，直到作者书面确认 |
| DA3-BASE | 联合几何（候选） | Apache-2.0（卡与 README 一致） | Apache-2.0 | 是；但不够准（4 个现场值 MAE 4.5 cm，Pi3X 1.4 cm） |
| SAM 3D Objects | 物体补全（RecGen 的替代） | **SAM License**（Meta 自定义：禁 ITAR / 军事 / 核 / 间谍用途；权重须随许可证分发） | SAM License | 有条件可以，须审许可 |
| MoGe-3 | 第二意见深度 | MIT | MIT（MoGe `74fbce0`、FlexGEMM `b2fadb2`） | 是 |
| SAM 3 | 分割 | **SAM License**（Meta 自定义） | SAM License（transformers 5.17.0 Apache-2.0） | 有条件可以，须审许可 |
| DINOv2 | RecGen / SAM 3D 的编码器 | Apache-2.0 | Apache-2.0 `7764ea0` | 是 |

- 现行流程有两个非商用（Pi3X、RecGen）。许可干净的组合目前是 SAM 3D 替 RecGen（形状检查打平，`completion-licence-ab-fair-2026-10-05`）；Pi3X 还没有既干净又够准的替代（DA3-LARGE-1.1 许可有争议，DA3-BASE 不够准）。
  The current pipeline has two non-commercial models. SAM 3D can replace RecGen; nothing clean and accurate enough replaces Pi3X yet.
- SAM 3D 镜像去掉了 GPL / AGPL / 非商用的 Python 包（bpy、pymeshfix、smplx、igraph、plyfile、point-cloud-utils、pdoc3、gsplat）。YOLOE（AGPL）只在一键流程镜像里，不在任何 on-prem 镜像。

## 命令 / Commands

```sh
# 构建上下文 / build context: SRC/workcell = ehs-spatial (codex/workcell-photo-speed), SRC/serving = panoptes-serving, code only
# (~8 MB, 435 files) + SRC/.dockerignore (whitelist; a raw serving checkout is 24 GB of outputs/ and a 1.6 GB .venv)
scripts/onprem/stage_context.sh SRC /path/to/panoptes-serving
docker build -f SRC/workcell/docker/workcell-cpu.Dockerfile -t panoptes-workcell-cpu SRC     # 构建时需联网 / internet at build time only
docker build -f SRC/workcell/docker/workcell-gpu.Dockerfile -t panoptes-workcell-gpu SRC     # 基础镜像按 digest 固定 / bases pinned by digest
docker build -f SRC/workcell/docker/recgen.Dockerfile       -t panoptes-recgen SRC
docker build -f SRC/workcell/docker/da3.Dockerfile          -t panoptes-workcell-da3 SRC   # 可选：Pi3X 的替代 / optional
docker build -f SRC/workcell/docker/sam3d.Dockerfile        -t panoptes-sam3d SRC          # 可选：RecGen 的替代 / optional

# 权重一次性下载（唯一联网的运行步骤；SAM 3 需先在 HF 接受许可并 export HF_TOKEN，或 --source 已有 HF 缓存）。
# fetch_weights.py 自己把 HF_HUB_OFFLINE 设为 0（镜像默认 1）；recgen = 快照 + DINOv2 + recgen-weights-manifest.json（/cache 路径）
docker run --rm -e HF_TOKEN -v /srv/panoptes-weights:/weights panoptes-workcell-gpu \
  /opt/pi3x/bin/python scripts/onprem/fetch_weights.py --cache /weights --models pi3x,sam3,moge3,recgen
docker run --rm --network none -v /srv/panoptes-weights:/weights panoptes-workcell-gpu \
  /opt/pi3x/bin/python scripts/onprem/fetch_weights.py --cache /weights --verify          # 离线重算 sha256（含 DINOv2 与 recgen 清单）

# 离线服务器 / air-gapped server：在联网机器上构建 + 下载权重，带走两个目录（镜像包、权重目录）
scripts/onprem/airgap.sh save /media/panoptes-images        # docker save | gzip -> *.tar.gz + SHA256SUMS（默认三个镜像；DA3 / SAM 3D 要列名字）
scripts/onprem/airgap.sh load /media/panoptes-images        # 服务器上 / on the server: sha256sum -c --strict, then docker load ONLY the tarballs SHA256SUMS lists
scripts/onprem/airgap.sh wheelhouse /media/panoptes-wheels  # 可选 / optional: each venv's pins as wheels; restore a venv with
scripts/onprem/airgap.sh install-wheels /media/panoptes-wheels/panoptes-workcell-cpu/checks /opt/checks/bin/python   # pip --no-index

# 之后全部 --network none / everything below runs with --network none
W="-v /srv/panoptes-weights:/weights"; D="-v /srv/panoptes-runs:/runs"
# Pi3X（与 modal run modal_apps/pi3x_geometry.py 同参数 / same flags as modal run）
docker run --rm --gpus all --network none $W $D panoptes-workcell-gpu \
  /opt/pi3x/bin/python scripts/onprem/run_stage.py --weights /weights /serving/modal_apps/pi3x_geometry.py --run /runs/RUN
# SAM 3 文本提示（RUN_SAM 内放 words.json、source-N.jpg、photo-N.png、cart-boxes.json）
docker run --rm --gpus all --network none $W $D panoptes-workcell-gpu \
  /opt/sam3/bin/python scripts/onprem/run_stage.py --weights /weights scripts/workcell_sam_worker.py /runs/RUN_SAM
# 新采集的地面掩码：选实例 → 冻结 → 拟合地面 / floor masks of a new capture: select instances, freeze, fit the floor
docker run --rm --network none $D panoptes-workcell-cpu /opt/assemble/bin/python scripts/onprem/floor_masks.py \
  --run /runs/RUN --sam3 /runs/RUN_SAM/sam3.json --out /runs/RUN/evidence/floor-masks-sam3
docker run --rm --network none $D panoptes-workcell-cpu /opt/assemble/bin/python /serving/scripts/research/prepare_capture_evidence.py \
  floor --output /runs/RUN --spec /runs/RUN/evidence/floor-masks-sam3/floor-spec.json
# RecGen（--weights 给 /cache 一个运行私有 overlay：读的是 /weights/recgen，job 副本 /cache/jobs/<id>（含照片数据）写在 TMPDIR 下，
# 阶段结束即删，不进权重目录）。先跑一次环境检查（驱动要求 RUN/generation/environment-recgen/output.json 为 complete）。
# 不要用 -v /srv/panoptes-weights/recgen:/cache：那样 job 副本会写进权重目录。on-prem 不受 Modal GPU 预算闸限制（PANOPTES_ONPREM=1）。
docker run --rm --gpus all --network none $W $D panoptes-recgen python /workcell/scripts/onprem/run_stage.py --weights /weights \
  /serving/modal_apps/lucida_assets.py --mode check --output-dir /runs/RUN/generation/environment-recgen
docker run --rm --gpus all --network none $W $D panoptes-recgen python /workcell/scripts/onprem/run_stage.py --weights /weights \
  /serving/scripts/research/generate_lucida_assets.py --root /runs/RUN --object-ids robot,cart
# 组装 / assembly
docker run --rm --network none $D panoptes-workcell-cpu \
  /opt/assemble/bin/python scripts/onprem/run_stage.py /serving/modal_apps/assemble_scene.py --run /runs/RUN
# 检查（API 指向本地平台或发布服务；或用 --offline 冻结包）/ checks against the local platform, or a frozen bundle
docker run --rm --network none $D panoptes-workcell-cpu /opt/checks/bin/python scripts/onprem/run_stage.py \
  modal_apps/workcell_view_checks.py --checks floor,lines --view /runs/view.json --photos-dir /runs/RUN/input \
  --photo ID1=image_01.jpg,ID2=image_02.jpg --layer-url http://PLATFORM/…/measurement-layer/ID.json --api http://PLATFORM --out /runs/checks
# MoGe-3
docker run --rm --gpus all --network none $W $D panoptes-workcell-gpu /opt/moge/bin/python scripts/onprem/run_stage.py \
  --weights /weights modal_apps/workcell_moge_check.py --view … --run-dir /runs/RUN --out /runs/moge
# 急停尺度 / e-stop scale: SAM 3 候选掩码（GPU，1008 px 瓦片）→ 只用掩码求尺度（CPU）/ SAM 3 candidate masks, then the scale from masks only
docker run --rm --gpus all --network none $W $D panoptes-workcell-gpu /opt/sam3/bin/python scripts/onprem/run_stage.py \
  --weights /weights modal_apps/workcell_estop_mask.py --photos-dir /runs/RUN/input --photo image_01.jpg,image_02.jpg --out /runs/estop-masks
docker run --rm --network none $D panoptes-workcell-cpu /opt/checks/bin/python scripts/workcell_estop_scale.py \
  --run /runs/RUN --mask frame_0001=/runs/estop-masks/image_01,frame_0002=/runs/estop-masks/image_02 --out /runs/estop-scale.json
# 相机正交化（Pi3X 在 CUDA bf16 下位姿偏 ~4e-6，平台导入要 ≤ 1e-6）/ camera orthonormalisation before the platform import
docker run --rm --network none $D panoptes-workcell-cpu /opt/checks/bin/python scripts/onprem/orthonormalise_cameras.py /runs/RUN/geometry --keep-as pi3x-bf16
# 间隙 B（罩壳下沿、围栏下横梁）/ clearance B
docker run --rm --network none $D panoptes-workcell-cpu /opt/checks/bin/python scripts/onprem/run_stage.py \
  modal_apps/workcell_clearance_b.py --view /runs/view.json --photos-dir /runs/RUN/input --photo ID1=image_01.jpg,... \
  --layer-url http://PLATFORM/…/measurement-layer/ID.json --api http://PLATFORM --run-dir /runs/RUN --targets ID=post,ID=panel --out /runs/clearb
# DA3（可选）/ DA3, optional: weights once with fetch_weights_da3.py --cache /weights (online), then
docker run --rm --gpus all --network none $W $D panoptes-workcell-da3 /opt/da3/bin/python /serving/scripts/candidate_geometry_backend.py \
  --vendor-dir /vendor/depth-anything-3 --model-dir /weights/local/da3-large-1.1 --model-id depth-anything/DA3-LARGE-1.1 --device cuda \
  --inputs /runs/RUN/evidence/canonical/frame_0001.png /runs/RUN/evidence/canonical/frame_0002.png --output /runs/RUN/geometry
# SAM 3D（可选）/ SAM 3D, optional: weights once with fetch_weights_sam3d.py --cache /weights (gated: HF_TOKEN or --source), then a
# driver that calls modal_apps/sam3d_research.SAM3DObjects().run.remote(rgb, mask, pointmap, seed) (see onprem_image_proof.SAM3D_DRIVER)
docker run --rm --gpus all --network none $W $D panoptes-sam3d python /workcell/scripts/onprem/run_stage.py --weights /weights DRIVER.py IN OUT
```

冻结包 / frozen bundle：`run_stage.py --record DIR …`（联网机器上跑一次，保存该阶段读到的每个 URL，sha256 寻址），`--offline DIR …` 回放、拒绝其它 URL 和所有非回环 socket/DNS，并设 `HF_HUB_OFFLINE=1`。

## 离线模式 / Offline mode

- 运行时：`HF_HUB_OFFLINE=1`（镜像默认）；`--weights` 把缓存链接到各阶段原本读取的路径（`/tmp/pi3x`、`/v/sam3/huggingface/hub`、`HF_HOME`、`/cache`），所以函数体一行未改。
- 进程内保护只覆盖 Python socket/urllib；子进程和 C 层客户端要靠容器 `--network none`（证明里用 Modal `block_network=True`）。The in-process guard covers Python sockets/urllib only; use `--network none` for the container.
- **没有 modal 客户端**：`run_stage.py` 把 `scripts/onprem/modal_stub` 放到 `sys.path` 和 `PYTHONPATH` 最前，`import modal` 得到存根（App / function / cls + enter / method / local_entrypoint / Volume / Image / Secret；部署对象 `Function.from_name`、`Cls.from_name` 直接拒绝；其它属性在导入时就报 AttributeError）。镜像 freeze = Modal 镜像的 freeze（`/opt/assemble` 另加 shapely），不再多 modal 1.5.4 和它的 10 个未固定依赖。
  No modal client: `import modal` is the stub; the images' freeze is the Modal images' (plus shapely in /opt/assemble).
- `--weights`：每个模型的容器路径（`/cache`、`/tmp/pi3x`、`/v/sam3/huggingface/hub`、`/opt/torch-hub/hub/checkpoints`）链到 TMPDIR 下的运行私有 overlay，里面只有 manifest 里登记过的权重条目（逐条 symlink）；阶段在这些路径下新写的东西（RecGen 的 `/cache/jobs/<id>/input.npz`、HF 的锁文件…）留在 overlay，结束时（含 `docker stop` 的 SIGTERM）删掉；另一个还在跑（或崩溃留下）的 overlay 不会被接管，会报错并提示删除。
- torch.hub：`torch.hub.load(repo, ..., source='github')` 一律改成 `source='local'`，指向 torch.hub 自己目录下 vendored 的 `<owner>_<repo>_<ref>`（没有就报错）；除 `file://` 外的 torch.hub 下载一律拒绝。所以即使容器有网，也不会去 github.com 探默认分支（SAM 3D 实测两次：有网时 0 次外连；不经 run_stage 时同一调用连 github.com:443）。
- `PANOPTES_ONPREM=1`：阶段据此关掉只对 Modal 有意义的规则（RecGen 的 GPU 预算闸）。
- `_loopback` 按解析后的地址判断（127/8、::1、IPv4 映射、0.0.0.0、精确的 `localhost`），`localhost.evil…`、`127.example.com` 都算外网。
- 花费账：各阶段为 Modal 写的 `*spend-ledger.json`，经 `run_stage.py` 运行时改记为 `mode: "on-prem in-process"`、本机硬件（GPU 名、cgroup 的 CPU 配额与内存上限，即 `docker --cpus / --memory`；Modal 沙箱里报的是 24 CPU、上百 GiB，不是预留值）、所有 `*Usd` 为 null、
  去掉 `rateSource`，原硬件写到 `modalHardwareProfile`（拦截 `Path.write_text`，阶段文件一行未改）；阶段自己打印到 stdout 的那行仍是 Modal 价目估计。
  Spend ledgers written through `run_stage.py` become `on-prem in-process` with no USD (stage files unchanged); the stage's stdout line is not rewritten.

## 证明命令 / Proof commands (Modal is only the test bench)

```sh
# (a) Mac, CPU, no Modal credentials: record once, replay offline
python scripts/onprem/run_stage.py --record BUNDLE modal_apps/workcell_view_checks.py --checks floor --view VIEW --photos-dir DIR --photo MAP --layer-url URL --api API --out A1
python scripts/onprem/run_stage.py --offline BUNDLE modal_apps/workcell_view_checks.py --checks floor ... --out A2
# (b) CPU image, network blocked          (c) weights, then GPU image on one A100, network blocked          (RecGen image build)
ONPREM_PROOF=cpu   modal run modal_apps/onprem_image_proof.py --view VIEW --photos-dir DIR --photo MAP --layer-url URL --api API --bundle BUNDLE --reference 030-results.json \
                   [--run-dir RUN --targets ID=post,ID=panel,.. --clearance-reference CLEARB/results.json] --out B
ONPREM_PROOF=fetch modal run modal_apps/onprem_image_proof.py --models pi3x,moge3,sam3 --out F
ONPREM_PROOF=gpu   modal run modal_apps/onprem_image_proof.py --view VIEW --photos-dir DIR --photo MAP --layer-url URL --api API --bundle BUNDLE --run-dir RUN --moge-reference MOGE.json [--steps sam3] --out C
ONPREM_PROOF=recgen modal run modal_apps/onprem_image_proof.py --out R
# (d) RecGen 一次真实生成 / one real RecGen generation: fetch recgen into the volume (network on), then A100 with the network blocked
ONPREM_PROOF=recgen-generate modal run modal_apps/onprem_image_proof.py --run-dir RUN030 --object-id left_post --seed 42 [--no-fetch] [--replay] --out G
# (f) SAM 3D，容器有网 + Python socket 审计 / SAM 3D with the network ON and a socket audit (torch.hub without / with run_stage, generation)
ONPREM_PROOF=sam3d modal run modal_apps/onprem_image_proof.py --inputs NPZ_DIR --reference EARLIER_OUT --out H
# (e) 急停 SAM 3 掩码在 GPU 镜像里离线（一张 L4）/ e-stop SAM 3 masks in the GPU image, one L4, network blocked; compared with a `modal run` output
ONPREM_PROOF=estop modal run modal_apps/onprem_image_proof.py --photos-dir DIR --photo image_01.jpg,... --reference MODAL_ESTOP_OUT --out E
```
所有镜像都从 `stage_context.sh` 的上下文构建（与客户相同）；证明脚本（`onprem_image_proof.py`、`onprem_tools_proof.py`）需要 `PANOPTES_SERVING=serving 检出目录`（没有本机默认路径）。
Modal 是测试台：它给 `from_dockerfile` 加一层自己的 Python 包（`/usr/local`），运行时把自己的客户端 `/pkg` 放进 `PYTHONPATH`；证明脚本在每个阶段的环境里去掉 `/pkg`，"无 modal 客户端"的检查才有意义。
Every proof image is built from the `stage_context.sh` context; the proof scripts need `PANOPTES_SERVING`. Modal adds its own layer and puts its client (/pkg) on PYTHONPATH at run time; the proof removes /pkg from every stage's environment.

## 已证明 / 未证明 / What is proven, what is not

已证明（Modal `from_dockerfile` 构建、`block_network=True`，除 SAM 3D 审计外）/ proven (Modal from_dockerfile, network blocked unless noted):
- CPU、GPU、RecGen、SAM 3D 四个镜像按本轮 Dockerfile 构建成功；任何 venv 里都没有 modal 客户端（DA3 镜像从来没有，本轮未改、未重建）。
- 存根：12 个 on-prem 阶段 app 在本机真 modal 被拦时都能导入；镜像里经存根跑通的阶段：地面检查、间隙 B、Pi3X、MoGe-3 检查、SAM 3 worker、RecGen 环境检查 + 生成、SAM 3D 生成。
- 数值：地面检查差 0.0 m；间隙 B 除耗时外相同、叠图逐字节相同；Pi3X 逐位相同；MoGe-3 16/16 不变；SAM 3 地面 IoU 0.936 / 0.971 不变；SAM 3D 位姿逐位相同、网格 Hausdorff 0.0015 原生单位；RecGen 网格在自身运行间差异内（位姿差 8.2e-4，见下）。
- RecGen job 副本不进权重目录且被删；预算闸 on-prem 关闭。SAM 3D 有网时 Python 层 0 次外连。
- 本机（Mac）：`run_stage.py`、`orthonormalise_cameras.py`、`floor_masks.py`、`airgap.sh`、`fetch_weights*.py`、形状检查、间隙 B 自检；地面检查经存根离线 8.3e-17 m；间隙 B 经存根离线与 Modal 差 ≤ 4.5e-5 m（Mac 的 cv2 5.0 / numpy 2.5，不是镜像的 pin）。

未证明 / not proven:
- **没有在 Linux 主机上用普通 `docker build` / `docker run --network none` 构建或运行过任何镜像**：只有 Modal 的构建器（它多加一层 Python 包）和 Modal 沙箱（gVisor）。
- **`docker save` / `airgap.sh load` 没有执行过**（没有 Docker daemon）；只用假 docker 测过 load 的逻辑（只加载 SHA256SUMS 列出的、校验通过的包）。wheelhouse 来回只在本机 venv 测过。
- `--network none` 本身：用 Modal `block_network` 代替。除 SAM 3D 外，"有网也不外连"没有逐阶段审计过；审计只看 Python 层（socket / DNS / 子进程），C 层客户端看不到。
- GPU 只测了 A100（sm_80）和 L4（sm_89，仅 SAM 3）；sm_86、sm_90（H100）、客户驱动版本没测。显存 24 GB 卡没测。
- cgroup 硬件串：只用合成的 cgroup 文件测过（v1、v2、父级配额）；真实 `docker --cpus` 下没看过。
- RecGen：同一输入，本轮位姿与前两轮 on-prem 差 8.2e-4（前两轮彼此相同），网格 Chamfer 7e-5（0.08 mm）；原因没分离（GPU 非确定性或环境里少了 Modal 的 /pkg）。
- `prepare_capture_evidence.py floor`（serving）仍用"最低平面"规则重拟合，`floor_masks.py` 的最大一致集平面只决定选哪些实例。
- 平台、发布服务、网页托管、浏览器检查：见下一节，没动。

## 尚未 on-prem 及修法 / Not yet on-prem, and the fix

1. 地面掩码（两份报告）和 090 部分物体掩码来自产品运行（Gemini 命名 + fal SAM 3.1）。已冻结的报告不受影响；新采集用 `/opt/sam3` 的 SAM 3 文本提示（已离线跑通，IoU 0.94 / 0.97），但地面拟合会倾 1.4°、物体底部变几厘米：新流程应把地面掩码作为证据冻结并复核，或让拟合只用相机下方的地面区域。已加 `scripts/onprem/floor_masks.py`（实例选择规则 + 自检，输出 `prepare_capture_evidence.py floor` 的 spec 和逐实例记录）；它的选择平面已从"最低平面"改为最大一致集（090 留出检验里原规则失败，见阶段表第 3 行）。030 实测（原规则）：选后平面相对已发布地面仍倾 1.03°，
   但**同一份已发布地面点、同一代码**只改子采样起点或删 7 个点，平面就在 0.05–0.89° 之间跳（右围栏最低点 10.8–15.9 cm，右光幕 17.2–19.2 cm）：倾角主要来自
   `_ransac_floor_plane`"最低的足够支撑平面"规则 + 子采样的不稳定，不是掩码。已发布地面点对选后平面 96 % 在阈值内，对自己的已发布平面只有 75 %。
   修法（未做，属 serving `prepare_capture_evidence.py` / `ehs_spatial`）：拟合不子采样或对候选平面做稳健投票，并报告平面的采样区间；物体附近用局部地面（方法 B 的 W 扫描）复核。`workcell_sam_worker.py` 需要一个不等 OWLv2 车框的模式（现在用空 `cart-boxes.json` + `photo-N.png` 绕过）。
2. 030 车掩码用到 OWLv2 框（在 MapAnything 栅格上），一键流程镜像（`fast_report_app.build_image`：vLLM、SAM 3D、RAM++、YOLOE(AGPL) 等，未锁版本）没有 Dockerfile。修法：直接在原图上用 SAM 3 文本/框提示（mask transfer 路径），或把 OWLv2（transformers，Apache-2.0）加进 `/opt/sam3`。090 的 SAM2.1 同理改用 SAM 3 或加 venv。
3. ~~`build_capture_report.py` 写死 Pages 仓库绝对路径~~ 已改为 `--pages-root` / `PANOPTES_PAGES_ROOT`（默认不变）。剩下：交付时把 Pages 仓库的 `pack-model.py`、`build-unified-data.py` 两个文件随代码放进一个目录（或镜像）并传 `--pages-root`。
4. 平台（FastAPI + PostgreSQL + 本地 blob）和发布服务没有容器，本任务也没跑（不启动 Postgres）。修法：按 `uv.lock` 写平台 Dockerfile + 官方 Postgres 容器；发布服务同一 FastAPI 用 uvicorn。
5. 报告网页和测量层在 GitHub Pages；`styles.css` 引 Google Fonts。修法：`npm ci && VITE_API_ORIGIN=… VITE_PUBLICATION_ID=… npm run build` 后本地静态托管；字体随包或删去该 import。浏览器检查改为本地 Playwright 打开本地站点。
6. 许可：见"许可 / Licences"一节。Pi3X、RecGen 非商用；DA3-LARGE-1.1 有争议；SAM 3、SAM 3D 是 Meta 自定义许可。技术上已能离线跑，但商用部署前必须取得许可或换模型。Technically on-prem ≠ licensed for commercial use.
7. 090 的 Pi3X 几何是在 Mac MPS fp32（torch 2.14）上算的，CUDA 镜像（torch 2.5.1 bf16）重算会有小差异；030 是 CUDA，可逐位对照。
8. Python 构建不同：Docker 官方 CPython 3.11.10 / uv 的 python-build-standalone，对 Modal 的 python-build-standalone，版本号相同。
9. 可复现性的边界 / reproducibility limits：基础镜像按 digest 固定、pip 全部 `==` 固定、git 按 commit 固定；apt 包与 uv 安装脚本只按版本、未按 digest。
   所以离线交付的可复现单元是 `airgap.sh save` 出来的镜像包（附 SHA256SUMS），不是在客户处重新 `docker build`。wheelhouse 只覆盖各 venv 的 pip 包。
   Bases pinned by digest, pip by `==`, git by commit; apt packages are not. The reproducible unit for an air-gapped site is the saved image tarball.
10. 急停尺度与分辨率：种子（`seed_from_mask`）已改为以红钮宽度为单位，030 急停截图缩放 0.5–3 倍都找得到；但 `fit()`（从 `estop_cylinder.py` 原样复制）有固定像素窗口（±8 px 轴搜索、强度台阶 σ ≤ 3 px），同一急停放大 2–3 倍尺度 +1.5…+1.9 %，缩小 0.5 倍 −0.35 %（相对 1 倍截图）。
    急停在照片里明显大于现在（红钮 > ~60 px）时需要先改 `fit()` 并重新验证；现有两份报告（红钮 47–58 px）不受影响。
    E-stop scale vs resolution: the seed is resolution-independent now, but fit() keeps fixed pixel windows: +1.5…+1.9 % at 2–3x, -0.35 % at 0.5x.
