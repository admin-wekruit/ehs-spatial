# 工位照片流程 on-prem / Workcell photo pipeline on-prem (2026-10-05)

目标：在客户自己的 Linux GPU 服务器上跑完整流程，不用 Modal、不调 SaaS API、运行时不联网。
Goal: run the pipeline on a customer's own Linux GPU server: no Modal, no SaaS API, no internet at run time.

证据 / evidence: `research-notes/workcell-onprem-2026-10-05/` (results.json, README).

## 结论 / Bottom line

- **已证明 / proven**
  - `scripts/onprem/run_stage.py` 在本进程里运行任何 Modal app 的 local entrypoint：`Function.remote/map/spawn` → modal 自带的 `Function.local()`（即原函数体），不需要 Modal 账号、token、配置（全部 `MODAL_*` 去掉、`MODAL_CONFIG_PATH` 指向不存在的文件也能跑）。Runs any Modal app's entrypoint in-process through modal's own `Function.local()`; no account, token or config.
  - 030 地面检查 / 030 floor check：Mac 本地离线（网络被拦）与 `030-results.json` 最大差 8e-17 m；CPU 镜像（Modal 上 `from_dockerfile` 构建、`block_network=True`）里最大差 0.0 m，8 个物体状态全一致。Mac offline: max diff 8e-17 m; inside the CPU image with the network blocked: 0.0 m, all 8 statuses equal.
  - CPU 镜像里全部自检通过（run_stage、fetch_weights、shape core、floor/lines/plane_stereo/transfer、MoGe 检查 CPU 部分、组装 venv 导入）。镜像 pip freeze 与 Modal 镜像逐项相同，只多 modal 客户端及其依赖。All CPU self-tests pass in the image; its pip freeze equals the Modal image's, plus the modal client.
  - RecGen 镜像构建成功，离线导入 RecGen/xformers/spconv 成功，pin 与 Modal 镜像相同（未在 GPU 上生成物体）。RecGen image builds and imports offline with the Modal pins (no object generated).
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
- **未就绪 / not yet**: 见最后一节 / see the last section (平台与发布服务没有容器化、报告网页托管、build_capture_report 的绝对路径、两个非商用许可 + SAM 3 自定义许可 / platform and publication service not containerised, web hosting, an absolute path, two non-commercial licences + SAM 3's custom licence).

## 阶段表 / Stages

| # | 阶段 stage | 代码 code | 今天 today | on-prem | 外部依赖 → 替换 external → replacement | 模型 pin / licence | 证明 proof |
|---|---|---|---|---|---|---|---|
| 1 | 冻结照片 + 证据 capture freeze, evidence | serving `scripts/research/prepare_capture_evidence.py` | Mac CPU | CPU 镜像 `/opt/assemble` | 无 none | — | 未跑 not run |
| 2 | 物体掩码 object masks | `scripts/workcell_sam_worker.py`（030：一键流程 `workcell_photo_all.py` 中 A100；车的框来自 OWLv2 + MapAnything 栅格）；090：SAM2.1 本机 MPS + 产品运行缓存掩码 | Modal A100 / Mac | GPU 镜像 `/opt/sam3`（文本提示）；框提示用 `workcell_mask_transfer.py` | fal SAM 3.1 + Gemini（产品运行）→ 本地 SAM 3 | `facebook/sam3@3c879f3`，SAM License（自定义，商用前须审） | 地面提示已证 floor prompt proven；物体未重跑 objects not re-run |
| 3 | 地面掩码 floor masks（两份报告） | 产品运行 `runs/user-bor1-02/inventory/sam/*__floor.json`（已冻结为 `evidence/floor-masks/`） | fal SAM 3.1 + Gemini（SaaS） | `/opt/sam3`：`words.json = ["floor"]` | SaaS → 本地 SAM 3 | 同上 | **IoU 0.936 / 0.971；平面倾 1.38°** |
| 4 | Pi3X 联合几何 joint geometry | serving `modal_apps/pi3x_geometry.py` + `scripts/candidate_pi3x_backend.py` | 030：Modal A100；090：Mac MPS fp32 | GPU 镜像 `/opt/pi3x`，代码 `/vendor/pi3` | HF 下载 → 权重缓存 | `yyfz233/Pi3X@bb1deea`，sha256 `69972d6e…`；代码 `9fa3ddb`（BSD 式）；**权重 CC-BY-NC-4.0 非商用** | **030 逐位相同 bit-identical** |
| 5 | RecGen 物体模型 | serving `modal_apps/lucida_assets.py` + `scripts/research/generate_lucida_assets.py` | Modal A100（`block_network`） | `docker/recgen.Dockerfile`，权重挂到 `/cache` | Modal Volume → 本地目录 | `TRI-ML/RecGen@bc0df7d`，代码 `fe3c931`，DINOv2 `7764ea0`；**代码 TRI 非商用，权重 CC-BY-NC-4.0** | 构建 + 离线导入 build + imports |
| 6 | 场景组装 assembly | serving `modal_apps/assemble_scene.py` | Modal CPU | CPU 镜像 `/opt/assemble` | 无 | — | 导入 imports only |
| 7 | 公开场景与报告文档 public scene + report doc | serving `scripts/research/build_capture_report.py` | Mac | 未就绪 | 写死 `/Users/adam/.../panoptes-workcell-pages`（pack-model.py、build-unified-data.py） | — | 否 no |
| 8 | 平台导入/发布/导出 platform import, publish, export | panoptes-platform FastAPI `ehs_spatial.platform.runtime` + PostgreSQL + 本地 blob | Mac | 同一应用部署在客户服务器（`docs/platform/OPERATIONS.md` "Run locally"：`PANOPTES_BLOB_BACKEND=local`、`PANOPTES_EXECUTOR_BACKEND=local`） | 无（本地 blob、本地执行器） | — | 未跑（本任务不启动 Postgres） |
| 9 | 发布服务 publication service | `modal_apps/publication_site.py` → `ehs_spatial.platform.publication_site:create_app` | Modal 部署（公开 URL） | 同一 FastAPI 应用用 uvicorn 在本地跑 | Modal → uvicorn | — | 未跑 |
| 10 | 测量层 + 报告网页 measurement layer + web | `web/`（Vite，`VITE_API_ORIGIN`、`VITE_PUBLICATION_ID`）；测量层 JSON | GitHub Pages | 静态目录（nginx 或平台 `PANOPTES_WEB_ROOT`） | GitHub Pages → 本地静态；`web/src/styles.css` 的 Google Fonts 离线时退回系统字体 | — | 未跑 |
| 11 | 检查 checks: floor, lines, plane_stereo, transfer | `modal_apps/workcell_view_checks.py` + `scripts/workcell_checks/*` | Modal CPU | CPU 镜像 `/opt/checks` | 读 GitHub Pages 测量层 + Modal 发布服务 → 本地平台 URL，或 `run_stage.py --offline` 包 | — | **floor 已证（相等）**；四个模块自检 |
| 12 | 形状检查 shape check | `modal_apps/workcell_shape_check.py` | Modal CPU | `/opt/checks`（scipy 1.14.1，Modal 解析为 1.17.1） | 同 11 | — | 自检 self-test |
| 13 | 掩码迁移 mask transfer | `modal_apps/workcell_mask_transfer.py` | Modal L4 | `/opt/sam3` | Modal Volume → 权重缓存 | SAM 3 同上 | 同一 venv/权重，未单独跑 |
| 14 | MoGe-3 第二意见 | `modal_apps/workcell_moge_check.py` | Modal L4 | `/opt/moge` | HF → 权重缓存 | `Ruicheng/moge-3-vitl@184008f`，MIT；代码 MoGe `74fbce0`、FlexGEMM `b2fadb2`（MIT） | **030 尺度差 1.7e-5，残差差 ≤ 3.3e-4（16/16 行）** |
| 15 | 浏览器检查 browser check | `modal_apps/workcell_browser_check.py` | Modal CPU，克隆 GitHub Pages 仓库 | 未就绪 | GitHub + mcr Playwright 镜像 | — | 否 |
| 16 | 急停尺度 e-stop scale | research-notes `cell030-sept-pipeline-2026-10-05/estop*.py`（numpy/cv2） | Mac CPU | `/opt/checks` 或 `/opt/assemble` 可跑 | 无 | 唯一特殊参考，尺度不变 | 未跑 |

## 硬件 / Hardware

- GPU：1 张 NVIDIA Ampere 或更新（bf16），建议 ≥ 24 GB；实测峰值 RecGen 12.2 GB（030 运行记录）、SAM 3 2.2 GiB（两张 12 MP 照片）；证明用 A100-80GB。离线 GPU 证明里 Pi3X 26 s、MoGe-3 检查 22 s、SAM 3 13 s（含加载）。驱动须支持 CUDA 13（≥ 580：`/opt/sam3`、`/opt/moge` 是 cu13 torch 轮子；`/opt/pi3x` cu124、RecGen cu121 在同一驱动上可跑）+ NVIDIA Container Toolkit。One Ampere+ GPU, ≥ 24 GB suggested; driver ≥ 580 (CUDA 13 wheels).
- CPU：8 核、32 GB 内存（检查在 Modal 上用 8 CPU / 16 GiB）。
- 磁盘：权重约 17.3 GB（Pi3X 5.44、SAM 3 3.44、MoGe-3 1.48、RecGen 5.32 + DINOv2 1.22）；GPU 镜像三套 torch，估计约 20 GB；RecGen 镜像（CUDA devel 基础）估计约 15 GB。Weights ≈ 17.3 GB; images estimated ≈ 20 GB (GPU) and ≈ 15 GB (RecGen).

## 命令 / Commands

```sh
# 源码并排 / sources side by side: SRC/workcell = ehs-spatial (codex/workcell-photo-speed), SRC/serving = panoptes-serving
docker build -f SRC/workcell/docker/workcell-cpu.Dockerfile -t panoptes-workcell-cpu SRC     # 构建时需联网 / internet at build time only
docker build -f SRC/workcell/docker/workcell-gpu.Dockerfile -t panoptes-workcell-gpu SRC
docker build -f SRC/workcell/docker/recgen.Dockerfile       -t panoptes-recgen SRC

# 权重一次性下载（唯一联网的运行步骤；SAM 3 需先在 HF 接受许可并 export HF_TOKEN，或 --source 已有 HF 缓存）
docker run --rm -e HF_TOKEN -v /srv/panoptes-weights:/weights panoptes-workcell-gpu \
  /opt/pi3x/bin/python scripts/onprem/fetch_weights.py --cache /weights --models pi3x,sam3,moge3,recgen
docker run --rm --network none -v /srv/panoptes-weights:/weights panoptes-workcell-gpu \
  /opt/pi3x/bin/python scripts/onprem/fetch_weights.py --cache /weights --verify          # 离线重算 sha256

# 之后全部 --network none / everything below runs with --network none
W="-v /srv/panoptes-weights:/weights"; D="-v /srv/panoptes-runs:/runs"
# Pi3X（与 modal run modal_apps/pi3x_geometry.py 同参数 / same flags as modal run）
docker run --rm --gpus all --network none $W $D panoptes-workcell-gpu \
  /opt/pi3x/bin/python scripts/onprem/run_stage.py --weights /weights /serving/modal_apps/pi3x_geometry.py --run /runs/RUN
# SAM 3 文本提示（RUN_SAM 内放 words.json、source-N.jpg、photo-N.png、cart-boxes.json）
docker run --rm --gpus all --network none $W $D panoptes-workcell-gpu \
  /opt/sam3/bin/python scripts/onprem/run_stage.py --weights /weights scripts/workcell_sam_worker.py /runs/RUN_SAM
# RecGen（权重目录的 recgen/ 挂到 /cache；先跑一次环境检查，驱动脚本要求 RUN/generation/environment-recgen/output.json 为 complete）
docker run --rm --gpus all --network none -v /srv/panoptes-weights/recgen:/cache $D panoptes-recgen python /workcell/scripts/onprem/run_stage.py \
  /serving/modal_apps/lucida_assets.py --mode check --output-dir /runs/RUN/generation/environment-recgen
docker run --rm --gpus all --network none -v /srv/panoptes-weights/recgen:/cache $D panoptes-recgen python /workcell/scripts/onprem/run_stage.py \
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
```

冻结包 / frozen bundle：`run_stage.py --record DIR …`（联网机器上跑一次，保存该阶段读到的每个 URL，sha256 寻址），`--offline DIR …` 回放、拒绝其它 URL 和所有非回环 socket/DNS，并设 `HF_HUB_OFFLINE=1`。

## 离线模式 / Offline mode

- 运行时：`HF_HUB_OFFLINE=1`（镜像默认）；`--weights` 把缓存链接到各阶段原本读取的路径（`/tmp/pi3x`、`/v/sam3/huggingface/hub`、`HF_HOME`、`/cache`），所以函数体一行未改。
- 进程内保护只覆盖 Python socket/urllib；子进程和 C 层客户端要靠容器 `--network none`（证明里用 Modal `block_network=True`）。The in-process guard covers Python sockets/urllib only; use `--network none` for the container.
- `modal` pip 包只作库用（装饰器、`Function.local()`），不连 Modal；镜像 freeze 与 Modal 镜像相同，只多 modal 客户端依赖。

## 证明命令 / Proof commands (Modal is only the test bench)

```sh
# (a) Mac, CPU, no Modal credentials: record once, replay offline
python scripts/onprem/run_stage.py --record BUNDLE modal_apps/workcell_view_checks.py --checks floor --view VIEW --photos-dir DIR --photo MAP --layer-url URL --api API --out A1
python scripts/onprem/run_stage.py --offline BUNDLE modal_apps/workcell_view_checks.py --checks floor ... --out A2
# (b) CPU image, network blocked          (c) weights, then GPU image on one A100, network blocked          (RecGen image build)
ONPREM_PROOF=cpu   modal run modal_apps/onprem_image_proof.py --view VIEW --photos-dir DIR --photo MAP --layer-url URL --api API --bundle BUNDLE --reference 030-results.json --out B
ONPREM_PROOF=fetch modal run modal_apps/onprem_image_proof.py --models pi3x,moge3,sam3 --out F
ONPREM_PROOF=gpu   modal run modal_apps/onprem_image_proof.py --view VIEW --photos-dir DIR --photo MAP --layer-url URL --api API --bundle BUNDLE --run-dir RUN --moge-reference MOGE.json --out C
ONPREM_PROOF=recgen modal run modal_apps/onprem_image_proof.py --out R
```

## 尚未 on-prem 及修法 / Not yet on-prem, and the fix

1. 地面掩码（两份报告）和 090 部分物体掩码来自产品运行（Gemini 命名 + fal SAM 3.1）。已冻结的报告不受影响；新采集用 `/opt/sam3` 的 SAM 3 文本提示（已离线跑通，IoU 0.94 / 0.97），但地面拟合会倾 1.4°、物体底部变几厘米：新流程应把地面掩码作为证据冻结并复核，或让拟合只用相机下方的地面区域。`workcell_sam_worker.py` 需要一个不等 OWLv2 车框的模式（现在用空 `cart-boxes.json` + `photo-N.png` 绕过）。
2. 030 车掩码用到 OWLv2 框（在 MapAnything 栅格上），一键流程镜像（`fast_report_app.build_image`：vLLM、SAM 3D、RAM++、YOLOE(AGPL) 等，未锁版本）没有 Dockerfile。修法：直接在原图上用 SAM 3 文本/框提示（mask transfer 路径），或把 OWLv2（transformers，Apache-2.0）加进 `/opt/sam3`。090 的 SAM2.1 同理改用 SAM 3 或加 venv。
3. `build_capture_report.py` 写死 Pages 仓库绝对路径。修法：改成环境变量，并把 `pack-model.py`、`build-unified-data.py` 放进仓库/镜像。
4. 平台（FastAPI + PostgreSQL + 本地 blob）和发布服务没有容器，本任务也没跑（不启动 Postgres）。修法：按 `uv.lock` 写平台 Dockerfile + 官方 Postgres 容器；发布服务同一 FastAPI 用 uvicorn。
5. 报告网页和测量层在 GitHub Pages；`styles.css` 引 Google Fonts。修法：`npm ci && VITE_API_ORIGIN=… VITE_PUBLICATION_ID=… npm run build` 后本地静态托管；字体随包或删去该 import。浏览器检查改为本地 Playwright 打开本地站点。
6. 许可：Pi3X 权重 CC-BY-NC-4.0、RecGen 代码与权重非商用、SAM 3 自定义许可。技术上已能离线跑，但商用部署前必须取得许可或换模型。Licences: Pi3X weights and RecGen are non-commercial; SAM 3 has a custom licence. Technically on-prem ≠ licensed for commercial use.
7. 090 的 Pi3X 几何是在 Mac MPS fp32（torch 2.14）上算的，CUDA 镜像（torch 2.5.1 bf16）重算会有小差异；030 是 CUDA，可逐位对照。
8. Python 构建不同：Docker 官方 CPython 3.11.10 / uv 的 python-build-standalone，对 Modal 的 python-build-standalone，版本号相同。
