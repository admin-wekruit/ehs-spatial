# Panoptes 企业版工程计划（2026-10-07）

目标一句话：**clone 两个仓库 + 一份 `.env`（三组凭证）= 跑通**。GPU 在他们那边（两张 A100），存储是他们的 Mongo + S3 兼容对象存储，
算法一行不改。以后每个新站点 = 复制 `.env`；每个新模型 = 一个服务 + 一个 provider，流程不动。

## 0. 验收标准

| # | 验收 | 怎么验 |
|---|---|---|
| A1 | 新机器 `git clone` × 2 → 填 `.env` → `make up`（GPU 机）/ `panoptes run --cell 090`（流程机）→ 报告可看 | 交接当天在他们的 jump + A100 上走一遍 |
| A2 | 090 / 030 的数字与已发布版一致（现场值 ±0.1 cm，IoU ±0.01；SAM 3D 候选允许 ±0.02） | `research/module-swap-2026-10-07/REPRODUCE-PROMPT.md` §5 |
| A3 | 两张 A100 同时用：090 与 030 并行，或一个工位的候选分到两卡 | `panoptes run --cell 090 & --cell 030`，`/healthz` 看两卡都在跑 |
| A4 | 换存储不改代码：`PANOPTES_DATABASE_URL` postgres ↔ mongodb，`PANOPTES_BLOB_ROOT` 目录 ↔ `s3://` | 同一套契约测试两种后端全过 |
| A5 | 换模型位置不改代码：`*_BACKEND=http`（他们的 GPU）/ `local`（本进程）/ `modal`（我们） | 三种后端各跑一次 S4 |
| A6 | 没有第二处配置、没有机器路径、没有手工步骤：`env.template` 列全，CI 绿 | 代码审查 + CI |

## 1. 现状（能复用的，不重做）

| 已有 | 在哪 | 状态 |
|---|---|---|
| 模型服务 + 合同：SAM 3 `:8801`、MapAnything `:8802`、MoGe-3 `:8803`，FastAPI，一次 JSON POST | `serving/{sam3,mapanything,moge}_service.py`，`docs/BACKENDS.md`，`serving/HANDOFF.md` | **已在他们 Azure A100 上跑**（jump `10.21.72.251` socat 8080/8090 → `172.184.217.34`） |
| 客户端切换：`SAM3_BACKEND / GEOMETRY_BACKEND / MOGE_BACKEND = http \| modal \| fal \| replicate` + `*_HTTP_URL` | `ehs_spatial/providers/{sam3,map_anything,moge}.py` | 在用 |
| 平台数据层接口：`Repository`（32 方法）、`PolicyRepository`（12）、`BlobStore`（7） | `ehs-spatial/ehs_spatial/platform/{repository,storage}.py` | 只有 Postgres + 本地目录实现 |
| 离线 runner：任何 Modal 应用在本进程跑，权重本地，不联网 | `ehs-spatial/scripts/onprem/run_stage.py`、`fetch_weights*.py`、`airgap.sh`、`docker/*.Dockerfile` | 自检通过 |
| 可商用流程（MVS + MoGe-3 补洞、SAM 3D、组装 v2）+ 冻结数据 + 复现 prompt | `research/module-swap-2026-10-07/`（`env.sh`、`run_all.sh`） | 已验证；**但 S2a / S4 还直接调 Modal 应用**，这是本计划要收口的 |
| 平台作业表（租约 claim / heartbeat / recover） | `Repository.create_job / claim_job / …` | 有，没用来调度 GPU |

## 2. 目标架构（四层，每层一个替换点）

```
 .env ─┬─ 模型服务层  GPU VM ×2：每卡一个 compose 栈  serving/registry.yaml
       │     /v1/sam3  /v1/geometry-mvs  /v1/sam3d  /v1/moge  (+ /healthz /v1/info /v1/jobs)
       │                      ▲  X-API-Key；输入内容寻址（sha256）；长任务异步
       ├─ 流程层      panoptes CLI（CPU，任意机器；流程 = 现 run_all.sh 的 Python 化）
       │     providers/*.py  ← *_BACKEND = http | local | modal   MODEL_SERVICE_URLS 多卡
       ├─ 存储层      Repository: postgres:// | mongodb://     BlobStore: path | s3://
       │     工件布局  s3://<bucket>/<prefix>/{runs,publications,layers,weights}/…（内容寻址）
       └─ 发布层      publication_site 静态编译 → S3 静态托管 或 nginx 容器；测量层同桶
```

### 2.1 模型服务层（GPU）
- 一个模型 = 一个 FastAPI 服务 = 一个容器（镜像来自 `ehs-spatial/docker/*.Dockerfile`，依赖 hash 固定，权重从卷读、启动校验 sha256）。
- 合同统一（在 `docs/BACKENDS.md` 上加版本号，旧路径保留）：
  - `POST /v1/<model>`：JSON；大输入用 base64 或 `s3://` 引用（服务端有桶凭证时直接读）；输入带 `sha256`，命中缓存直接返回。
  - `GET /healthz`：`{ok, gpu, vram_free, queue_depth, model_revision}` —— 多卡调度就靠它。
  - `GET /v1/info`：`{model_id, revision, weights_sha256, licence, code_sha, seed_policy}` —— 每个响应也带这四个字段，报告里能溯源。
  - 长任务：`POST /v1/jobs/<model>` → `job_id`；`GET /v1/jobs/{id}`；SAM 3D 25 个候选、几何一次 RoMa 全对，都走这条。
  - 鉴权 `X-API-Key`（服务端 `.env`），错误码表，客户端统一超时 / 重试 / 退避（放在 `providers/base.py`）。
- 新增两个服务：
  - `geometry-mvs`（`:8804`）：输入冻结帧（518 规范帧 + alpha）→ 输出 RoMa 匹配 + 稠密 warp + DA3-BASE 起始几何 + MoGe-3 深度（即今天 `checks/clean-gpu` + `da3fair-geom` 的内容）。BA、三角化、补洞仍在流程侧 CPU，不动。
  - `sam3d`（`:8805`）：输入照片 + 掩码 + 本几何点图（+ 种子）→ 网格 + 位姿；对应 `modal_apps/sam3d_research.SAM3DObjects`，函数体原样。
- `serving/registry.yaml`：模型名、代码 revision、权重 revision + sha256、许可、显存、端口、所在卡。`make up` 读它起容器。

### 2.2 流程层（CPU）
- `panoptes` CLI（`ehs_spatial/cli.py`）：`panoptes run --cell 090 [--from S4] [--gpu-urls …]`，就是 `run_all.sh` 的 Python 化，步骤幂等、每步写 `ledger.json`（输入 sha、模型 revision、耗时、所在卡）。
- `providers/geometry_mvs.py`、`providers/sam3d.py`（新）+ 现有三个：每个 provider 三种后端：`http`（他们的服务）、`local`（`run_stage` 机制，本进程）、`modal`（我们）。**流程代码只认 provider**，`completion_ab.py` / `prod_route_modal.py` 里直接 `modal run` 的地方全部改走 provider。
- 多卡：`MODEL_SERVICE_URLS=http://gpu-a:8805,http://gpu-b:8805`，provider 按 `/healthz` 的 `queue_depth` 选最空的；工位级并行直接 `panoptes run --cell 090 & --cell 030`。第二阶段：`panoptes worker` 每卡一个，从平台作业表（已有租约语义）领任务——那是真正的队列，fork 多站点时用。

### 2.3 存储层
- `PANOPTES_DATABASE_URL`：`postgres://` → 现有；`mongodb://` → `MongoRepository` + `MongoPolicyRepository`（同一 Protocol）。集合名前缀 `PANOPTES_MONGO_PREFIX`（默认 `panoptes_`）：`panoptes_projects / scene_branches / scene_revisions / edit_batches / assets / captures / jobs / model_calls / publications / agent_turns / policy_*`。
  - 要点：`commit_edits` 的 requestId 幂等 + 基修订校验 → 单文档原子 `findOneAndUpdate`；作业租约 `claim / heartbeat / recover` → 带条件的原子更新；发布 requestId 幂等 → 唯一索引。不依赖多文档事务（单机 Mongo 也能跑）。
- `PANOPTES_BLOB_ROOT`：目录 → 现有 `LocalBlobStore`；`s3://bucket/prefix` → `S3BlobStore`（boto3，`AWS_ENDPOINT_URL` 指向他们的 S3 兼容端点，键 = sha256，`url()` 给 presigned 或静态站路径，分段上传）。
- 工件也进桶：`runs/<cell>/<run_id>/…`（运行目录）、`publications/<id>/`（发布目录 = 静态站源）、`layers/<id>.json`、`weights/<model>/<revision>/`（权重镜像，sha256 清单）。`panoptes artifacts push/pull` 两条命令，本地目录与桶同布局。
- 契约测试一套，两种后端都跑（CI 用容器化 Postgres + Mongo + MinIO）。

### 2.4 发布 / 查看层
- `scripts/prepare_publication_site.py` 的输出直接放桶（S3 静态托管）或 on-prem 栈的 nginx 容器（`ehs-spatial/containers/onprem/`，已有）。查看器（`web/`）构建产物同桶。

## 3. 接口守则（future-proof 的四条）

1. 版本在路径（`/v1/`）；破坏性改动开 `/v2/` 并行，旧的至少保留一个版本周期。
2. 每个响应带 `model_id / model_revision / weights_sha256 / code_sha`；流程把它们写进 `ledger.json` 和报告的 provenance。
3. 输入内容寻址，输出可缓存、可重放；种子与确定性标志是输入的一部分。
4. 只有一个配置面：`.env`（模板 `env.template` 每个键有注释和默认值）。代码里没有机器路径、没有 Modal 专有调用、没有"手工在盒子上做的事"。

## 4. 工作包与顺序

| WP | 内容 | 产出 | 估时 |
|---|---|---|---|
| WP1 存储后端 | `MongoRepository`、`MongoPolicyRepository`、`S3BlobStore`、URL 自动选择、集合前缀；契约测试双后端；`panoptes artifacts push/pull` | A4 | 2 d |
| WP2 providers 收口 | `providers/{geometry_mvs,sam3d}.py`（http / local / modal）；S2a、S4 改走 provider；`run_stage` 作为 `local` 后端；`ledger.json` | A5 | 2 d |
| WP3 新模型服务 | `serving/{geometry_mvs,sam3d}_service.py`，`/healthz /v1/info /v1/jobs`、API key、`registry.yaml`、每卡 compose、权重卷 + 校验、冒烟 curl；`docs/BACKENDS.md` v1 | A1（GPU 侧） | 2 d |
| WP4 CLI + 多卡 | `panoptes run`（run_all.sh Python 化，幂等）、`MODEL_SERVICE_URLS` 调度、`panoptes worker`（作业表租约） | A3 | 1.5 d |
| WP5 发布层 | S3 静态托管 / nginx；测量层；查看器指向可配置 | A1（查看） | 1 d |
| WP6 交接包 + CI | `HANDOFF.md` 重写（clone → `.env` → `make up` → 验收）；`env.template`；CI：契约测试、lint、docker build、合同冒烟（fake backend）；版本 `v1.0.0` | A6 | 1 d |

总计约 9.5 个工作日。里程碑：**M1**（WP1+2，第 4 天）本机用容器化 Mongo + MinIO 跑通 090；**M2**（WP3+4，第 8 天）他们两张 A100 上 090 + 030 并行跑通；**M3**（WP5+6，第 10 天）交接验收 A1–A6。

## 5. 他们要提供的（只有这些）

| 项 | 说明 |
|---|---|
| 两张 A100 的 VM（或一台两卡） | docker + NVIDIA runtime；`make up` 起服务；出站只需拉镜像 / 权重一次（或用我们给的 `airgap.sh` 包） |
| jump 的转发 | 现有 socat 再开 2 个端口（8804 / 8805），或换 nginx + TLS（建议，附 compose） |
| Mongo | 连接串 + 库名（`PANOPTES_DATABASE_URL=mongodb://…/<db>`）；允许建索引 |
| S3 兼容 | endpoint、bucket、access key / secret（`PANOPTES_BLOB_ROOT=s3://bucket/prefix`，`AWS_ENDPOINT_URL`，`AWS_ACCESS_KEY_ID/SECRET`） |
| HF token | 一次，拉 SAM 3D Objects / SAM 3（gated）；或用我们的权重镜像 tar（sha256 清单） |
| 许可确认 | SAM License：站点不涉核、军工、ITAR |

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| SAM 3D 在不同 GPU 上非确定（±0.02 IoU，择优可能换照片） | 种子固定；验收用容忍范围；响应带 GPU 型号 |
| Mongo 无副本集 → 无多文档事务 | 设计上只用单文档原子操作 + requestId 幂等（§2.3） |
| S3 兼容实现差异（分段上传、presigned、一致性） | 契约测试用 MinIO + 他们的端点各跑一遍；`S3BlobStore` 只用基础 API |
| jump socat 单点、明文 | nginx + TLS + API key；socat 保留做回滚 |
| 权重 gated / 外网受限 | `fetch_weights*.py` 一次 + `airgap.sh` 离线包 + sha256 校验 |
| 两张卡显存：SAM 3D ~20 GB，geometry-mvs（RoMa + DA3 + MoGe）~12 GB | 每卡一栈：卡 A = sam3d + sam3，卡 B = geometry-mvs + moge + mapanything；`registry.yaml` 写死，`/healthz` 报 vram |

## 7. 不做

不改算法与参数；不训练；Pi3X / RecGen 不进交付；不上 Kubernetes（两台机，compose 够）；反馈助手（sqlite）不动。
