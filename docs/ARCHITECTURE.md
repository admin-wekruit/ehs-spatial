# Panoptes 架构（企业版 1.0.0-rc1）

四层，每层一个替换点，全部由仓库根的一份 `.env`（模板 `env.template`）决定。算法代码不认机器、不认 Modal、不认存储实现。

```
 .env（env.template）
  │
  ├─ 模型服务层  GPU 机 ×2：每卡一个 compose 栈            deploy/compose.gpu-a.yml  deploy/compose.gpu-b.yml  serving/registry.yaml
  │     卡 A  sam3d :8805 (serving/sam3d_service.py)   sam3 :8801 (serving/sam3_service.py)
  │     卡 B  geometry-mvs :8804 (serving/geometry_mvs_service.py)   moge :8803   mapanything :8802
  │     合同  v1 = /healthz /v1/info /v1/<model> /v1/<model>/jobs /v1/jobs/{id}（docs/BACKENDS-v1.md）；v0 = 一次 POST（docs/BACKENDS.md）
  │     鉴权  X-API-Key = PANOPTES_SERVICE_API_KEY；输入内容寻址 input_sha256；长任务异步
  │                       ▲  jump VM 10.21.72.251：socat 单元 或 nginx+TLS，一端口/一前缀对一个服务      deploy/jumpbox/
  ├─ 流程层      panoptes CLI（CPU，任意机器）                 ehs_spatial/cli.py   = research/module-swap-2026-10-07/run_all.sh 的 Python 化
  │     providers/{sam3d,geometry_mvs,sam3,map_anything,moge}.py  ← *_BACKEND = http | local | modal（| fal | replicate）
  │     providers/service_client.py：*_HTTP_URLS 多卡（按 /healthz queue_depth 选卡）、超时、重试、ledger
  │     每步幂等，写 ledger.json（input_sha256、model_info、耗时、所在卡）
  ├─ 存储层      ehs-spatial ehs_spatial/platform/            Repository: PANOPTES_DATABASE_URL = postgresql:// | mongodb://
  │                                                           BlobStore:  PANOPTES_BLOB_ROOT   = /目录 | s3://bucket/prefix
  │     工件布局 <root>/{runs/<cell>/<run_id>, publications/<id>, layers/<id>.json, weights/<model>/<rev>}   scripts/panoptes_artifacts.py push/pull
  └─ 发布层      catalog/<id> → prepare_publication_site.py → 发布服务 + nginx（查看器 + 测量层）或 S3 静态站      deploy/publications/
                 PANOPTES_PUBLIC_BASE_URL/app.html#/reports/<id>
```

## 替换点

| 层 | 换什么 | 怎么换（只动 `.env`） | 代码在哪 |
|---|---|---|---|
| 模型服务 | 模型跑在哪张卡、哪台机、哪条代理 | `SAM3D_HTTP_URLS` / `GEOMETRY_MVS_HTTP_URLS`（逗号多卡）、`SAM3_HTTP_URL` / `GEOMETRY_HTTP_URL` / `MOGE_HTTP_URL`；jump 的转发表 `deploy/README.md` | `serving/*_service.py`，镜像 `ehs-spatial/docker/*.Dockerfile`，`serving/registry.yaml` |
| 模型服务 | 新模型 | 一个 FastAPI 服务（复制 `serving/sam3d_service.py` 的骨架，函数体 = 对应 Modal 应用）+ 一个 provider + `registry.yaml` 一行 + compose 一个 service | `serving/`、`ehs_spatial/providers/` |
| 流程 | 模型在服务 / 本进程 / Modal | `SAM3D_BACKEND`、`GEOMETRY_MVS_BACKEND` = `http` / `local`（`ehs-spatial/scripts/onprem/run_stage.py` 的 stub 在本进程跑同一函数体）/ `modal` | `ehs_spatial/providers/{sam3d,geometry_mvs}.py`，`providers/service_client.py` |
| 流程 | 从某步重跑 | `panoptes run --cell 090 --from S4`；每步看自己的产物存在就跳过，`input_sha256` 命中服务端缓存直接拿结果 | `ehs_spatial/cli.py` |
| 存储 | Postgres ↔ Mongo | `PANOPTES_DATABASE_URL` 的 scheme；`PANOPTES_MONGO_PREFIX` 集合前缀 | `ehs-spatial/ehs_spatial/platform/{config,repository}.py`（`Repository` 32 方法、`PolicyRepository` 12 方法同一 Protocol） |
| 存储 | 目录 ↔ S3 兼容 | `PANOPTES_BLOB_ROOT` 的 scheme；`AWS_ENDPOINT_URL`、`AWS_ACCESS_KEY_ID/SECRET` | `ehs-spatial/ehs_spatial/platform/storage.py`（`BlobStore` 7 方法：`LocalBlobStore` / `S3BlobStore`） |
| 发布 | 报告站在哪 | `PANOPTES_PUBLIC_BASE_URL`、`PANOPTES_PUBLICATION_{CATALOG,HTTP}`、`PANOPTES_WEB_ROOT`、`PANOPTES_PAGES` | `deploy/publications/`，`ehs-spatial/containers/onprem/serve_publications.py` |
| 全部 | 新站点 | 复制 `.env`，改 jump 地址和三组凭证 | — |

## 一次 SAM 3D 调用（S4a，`SAM3D_BACKEND=http`）

```
 panoptes run --cell 090                     流程机（CPU）
   │ S4a：每张有掩码的照片一个候选
   ▼
 providers/sam3d.py::generate(rgb, mask, pointmap, seed=42)
   │ 组请求体 {image_b64, mask_b64, pointmap_npz_b64, seed}，input_sha256 = sha256(规范化输入字节)
   ▼
 providers/service_client.py
   │ 1. 对 SAM3D_HTTP_URLS 的每个根 GET <root>/healthz（X-API-Key），选 queue_depth 最小、ok:true 的根
   │ 2. POST <root>/v1/sam3d/jobs → 202 {job_id, input_sha256, cached}        （cached:true = 服务端缓存命中，直接到 5）
   │ 3. GET  <root>/v1/jobs/{job_id} 轮询，退避；status queued|running|done|error
   │    连接错误 / 429 busy / 503 → 重试 PANOPTES_SERVICE_RETRIES 次；model_error 不重试；总时限 PANOPTES_SERVICE_TIMEOUT_S
   ▼
 jump VM 10.21.72.251
   │ socat :8085 → 卡 A :8805（或 nginx /sam3d/ → gpu-a:8805），X-API-Key 原样透传
   ▼
 serving/sam3d_service.py（容器：ehs-spatial/docker/sam3d.Dockerfile；权重卷只读，启动 sha256 校验，HF_HUB_OFFLINE=1）
   │ 校验 key（401）→ 队列深度 ≥ PANOPTES_SERVICE_MAX_QUEUE 则 429 → 查 input_sha256 缓存
   │ 未命中：modal_apps/sam3d_research.SAM3DObjects.run(rgb, mask, pointmap, seed) 原函数体，cuDNN deterministic
   │ 结果：mesh_npz（vertices/faces/colors/object_to_camera_p3d）
   │   小 → 响应里 result_b64；设了 PANOPTES_RESULT_BUCKET → 上传 s3://…/<input_sha256>.tar.gz，响应 result_uri
   │ 每个响应带 model_info = /v1/info（model_id、model_revision、weights_sha256、code_sha、seed_policy）+ gpu + seconds
   │ 日志一行 JSON {job_id, input_sha256, seconds, gpu, status}（无载荷、无 key）
   ▼
 流程机
   │ 4. result_uri → 用 AWS_* 凭证从桶拉；result_b64 → 直接解
   │ 5. 写 gen/<variant>/<object>.npz（和研究 A/B 完全相同的落盘格式，后面的候选组装 / 择优 / 组装 v2 不知道它来自哪）
   │ 6. ledger.json 追加 {step:"S4a", object, input_sha256, root, job_id, cached, seconds, gpu, model_info}
   ▼
 S7 平台导入：Repository.create_job / model_calls 记一行（postgres 或 mongo），工件进 BlobStore（目录或 s3://）；
 报告的 provenance 面板显示 model_info —— 每个数字都能追到权重 sha256 和代码 sha。
```

`SAM3D_BACKEND=local` 时 2–4 步变成本进程调用 `run_stage.py` 的 stub（同一函数体，本机 GPU）；`modal` 时变成 `modal.Cls.from_name(...)`。
ledger 的字段一样。

## future-proof 守则（计划 §3 + 合同 v1 规则）

1. **版本在路径**：`/v1/`；破坏性改动开 `/v2/` 并行，`/v1/` 至少保留一个版本周期。v0 的一次 POST 路径（`/sam3`、`/geometry`、`/moge`）照旧。
2. **每个响应可溯源**：`model_id / model_revision / weights_sha256 / code_sha` 随响应返回，流程写进 `ledger.json` 和报告 provenance。
3. **输入内容寻址，输出可缓存可重放**：种子、确定性标志是输入的一部分，进 `input_sha256`；同一输入第二次调用不算第二次。
4. **只有一个配置面**：`.env`（`env.template` 每个键一行注释 + 默认）。代码里没有机器路径、没有 Modal 专有调用、没有"手工在盒子上做的事"；
   `scripts/check_env.py` 对未登记的键报警，CI 跑 `check_env.py env.template`。
5. **服务无状态、不联网**：只读权重卷、不存 runs、不碰业务数据；日志无载荷无密钥。
6. **契约测试一套，多后端都跑**：`tests/contract/` 对 fake backend（CI）和真服务（`PANOPTES_CONTRACT_URL`）同一组请求；存储层对
   Postgres + 目录 和 Mongo + S3 兼容同一组契约测试。
7. **流程只认 provider**：没有任何 stage `import modal`；换模型位置 = 改 `*_BACKEND`，换模型 = 加 provider，流程不动。
8. **不改算法来"对数字"**：验收用容忍范围（`research/module-swap-2026-10-07/REPRODUCE-PROMPT.md` §5），对不上记差异。
