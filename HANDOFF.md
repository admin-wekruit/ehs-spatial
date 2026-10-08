# Panoptes 企业版交接（1.0.0-rc1，2026-10-07）

> 系统现状（架构、服务、结果、什么不在交付里）以 `docs/STATE.md` 为准；本文件只讲怎么跑。

一句话：**clone 一个仓库 + 填一份 `.env` + GPU 机 `make up` + `panoptes run --cell 090` = 报告可看**。
GPU 在你们的两张 A100 上，存储是你们的 Mongo + S3 兼容对象存储，算法与参数和 2026-10-07 已发布的报告一字不改
（`research/module-swap-2026-10-07/REPRODUCE-PROMPT.md`）。以后每个新站点 = 复制 `.env`；每个新模型 = 一个服务 + 一个 provider。

计划与架构：`docs/ENTERPRISE-PLAN-2026-10-07.md`、`docs/ARCHITECTURE.md`；服务合同：`docs/BACKENDS-v1.md`（v1）、`docs/BACKENDS.md`（v0）；
部署文件：`deploy/`（`deploy/README.md`）；变更：`CHANGELOG.md`。

## 0. 四步

| 步 | 在哪台机器 | 做什么 |
|---|---|---|
| 1 | 所有机器 | `git clone -b main https://github.com/admin-wekruit/ehs-spatial.git`（平台、检查、服务、流程、数据都在这一个仓库；`-b main`：默认分支还没切） |
| 2 | 流程机 + 两台 GPU 机 | `cd ehs-spatial && make env` → 编辑 `.env`（§2：三组凭证 + jump 地址）→ `make check-env` 全绿；**同一份 `.env` 复制到每台机器** |
| 3 | GPU 机 | 权重一次（§3）→ 卡 A `make up GPU=a`、卡 B `make up GPU=b` → `make smoke`（每个服务 `/healthz` + 一次真调用） |
| 4 | 流程机 | `make run CELL=090`（约 40 分钟），`make run CELL=030`（约 25 分钟）→ 结束时打印的地址 `<PANOPTES_PUBLIC_BASE_URL>/app.html#/reports/<publicationId>` 打开报告 |

流程机只要 CPU、Python 3.12 + `uv`（`uv sync --frozen`）；GPU 机只要 docker + NVIDIA Container Toolkit（驱动 ≥ 580，支持 CUDA 13）。
jump VM 要加三个转发（§4）。报告站起法在 `deploy/publications/README.md`。

## 1. 你们提供的（只有这些）

| 项 | 具体要什么 | 进 `.env` 的键 |
|---|---|---|
| 两张 A100 | 两台 VM 或一台两卡；docker + NVIDIA runtime；出站只需一次（拉镜像、拉权重；或用我们给的 `airgap.sh` 离线包 + 权重 tar） | — |
| jump VM 转发 | 现有 socat 再开 3 个端口（8085、8084、8081），或换 nginx + TLS（§4） | `SAM3D_HTTP_URLS`、`GEOMETRY_MVS_HTTP_URLS`、`SAM3_HTTP_URL`（+ 已有的 `GEOMETRY_HTTP_URL`、`MOGE_HTTP_URL`） |
| 服务密钥 | 你们生成一个：`openssl rand -hex 32`；GPU 机和流程机的 `.env` 用同一个值 | `PANOPTES_SERVICE_API_KEY` |
| Mongo | 连接串 + 库名；该用户能建集合和索引（单机、无副本集也行：只用单文档原子操作） | `PANOPTES_DATABASE_URL=mongodb://user:pass@host:27017/panoptes`，可选 `PANOPTES_MONGO_PREFIX` |
| S3 兼容对象存储 | endpoint URL、bucket（已建或允许建）、access key / secret（list / get / put / multipart 权限）；是否 path-style；私有 CA 的话给 CA 文件 | `PANOPTES_BLOB_ROOT=s3://bucket/prefix`、`AWS_ENDPOINT_URL`、`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`（可选 `PANOPTES_RESULT_BUCKET`） |
| Hugging Face token | 一个已在 huggingface.co 接受 `facebook/sam-3d-objects` 和 `facebook/sam3` 许可（SAM License）的账号，**只在拉权重的那一次用**；或直接用我们的权重 tar（带 sha256 清单），不需要 token | 不进 `.env`（`check_env` 会拒绝） |
| 许可确认 | SAM License：站点不涉核、军工、ITAR | — |
| TLS（可选） | jump 用 nginx 方案时的证书对，或接受自签 | `SSL_CERT_FILE`、`REQUESTS_CA_BUNDLE` |

## 2. `.env`：唯一的配置面

`env.template` 列全了系统读的每一个键（分 7 节，每键一行注释：是什么 / 默认 / 谁读）；`make env` 复制成 `.env`。要填的只有：

```
# 1 GPU 服务（两台 GPU 机）
PANOPTES_SERVICE_API_KEY=<openssl rand -hex 32>
WEIGHTS=/srv/panoptes/weights
# 2 客户端：jump 地址（socat 形式；nginx 形式见 deploy/README.md）
SAM3D_HTTP_URLS=http://10.21.72.251:8085
GEOMETRY_MVS_HTTP_URLS=http://10.21.72.251:8084
SAM3_HTTP_URL=http://10.21.72.251:8081/sam3
GEOMETRY_HTTP_URL=http://10.21.72.251:8080/geometry
MOGE_HTTP_URL=http://10.21.72.251:8090/moge
# 3 存储
PANOPTES_DATABASE_URL=mongodb://user:pass@mongo.internal:27017/panoptes
PANOPTES_BLOB_ROOT=s3://panoptes/prod
AWS_ENDPOINT_URL=https://s3.internal
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
# 4 报告站
PANOPTES_PUBLIC_BASE_URL=http://reports.internal:8080
# 6 路径（这一个 checkout；权重目录在第 1 节）
PANOPTES_SERVING=/srv/panoptes/ehs-spatial
PANOPTES_WORKCELL=/srv/panoptes/ehs-spatial
```

`make check-env`（= `scripts/check_env.py .env --strict`）：必填键齐、`*_BACKEND` 合法、URL 能解析、`.env` 里没有 `HF_TOKEN`、没有占位符、路径存在、
没有模板之外的键。`make check-env-live` 再对每个 URL 打 `/healthz`。第 6 节其余键（`SWAP_SCRATCH`、`PANOPTES_RUNS`、`PY`、`MODAL_RUN`…）
由 `research/module-swap-2026-10-07/env.sh` 推导，不用填。

## 3. GPU 机：权重一次 + 起服务

```sh
# 权重（联网机器一次；之后永远离线）—— 只有这一步要 token
export HF_TOKEN=<token>
cd ehs-spatial
python scripts/onprem/fetch_weights_sam3d.py    --cache $WEIGHTS      # SAM 3D Objects @2e73555（12.1 GB，mesh-only）+ DINOv2
python scripts/onprem/fetch_weights_geometry.py --cache $WEIGHTS      # RoMa outdoor + DA3-BASE @f4a6c9b + MoGe-3 @184008f（公开）
python scripts/onprem/fetch_weights_sam3d.py --cache $WEIGHTS --verify && python scripts/onprem/fetch_weights_geometry.py --cache $WEIGHTS --verify
unset HF_TOKEN                                                        # 或：解我们的权重 tar 到 $WEIGHTS，跑两条 --verify
# 镜像：ehs-spatial/docker/{sam3d,geometry,workcell-gpu}.Dockerfile（联网 docker build，或 scripts/onprem/airgap.sh load）
# 起服务
cd ehs-spatial
make up GPU=a      # 卡 A：sam3d :8805 + sam3 :8801           (deploy/compose.gpu-a.yml)
make up GPU=b      # 卡 B：geometry-mvs :8804 + moge :8803 + mapanything :8802   (deploy/compose.gpu-b.yml)
make smoke         # serving/smoke_v1.sh：每个 /healthz ok:true + /v1/info + 一次真调用；PANOPTES_FAKE_MODEL=1 可在没权重时先验链路
make logs SERVICE=sam3d
```

### 两张 A100 的布局（计划 §6；每卡一栈，`serving/registry.yaml` 写死，`/healthz` 报 `vram_free_mb`）

| 卡 | compose | 服务 | 端口 | 显存 |
|---|---|---|---|---|
| A | `deploy/compose.gpu-a.yml` | sam3d（SAM 3D Objects，25 候选/工位走 `/v1/sam3d/jobs`） | 8805 | ~20 GB |
| A | | sam3（SAM 3 掩码，v0） | 8801 | v0 三服务常驻合计 ~20 GB bf16（`docs/archive/serving-HANDOFF-v0-services.md` 历史实测），sam3 是其中一份 |
| B | `deploy/compose.gpu-b.yml` | geometry-mvs（RoMa + DA3-BASE + MoGe-3 深度） | 8804 | ~12 GB |
| B | | moge（MoGe-3，v0） | 8803 | 同上 |
| B | | mapanything（v0） | 8802 | 同上 |

两卡都在 40 GB 以内，80 GB 卡有余量。090 和 030 并行时两卡同时忙；想把一个工位的 25 个候选分到两卡，卡 B 再起一个 sam3d
（compose 可选 profile），`SAM3D_HTTP_URLS` 写两个根。

## 4. jump VM（`10.21.72.251`）：你们加的

应用只指向 jump，不指向 GPU 机 IP。现有：socat `8080 → 172.184.217.34:8802`（mapanything）、`8090 → :8803`（moge）。再加三个（`deploy/README.md` 有完整端口表与两种方案）：

| jump 端口 | → GPU 卡:端口 | 服务 | `.env` 里贴的行 |
|---|---|---|---|
| 8085 | A:8805 | sam3d | `SAM3D_HTTP_URLS=http://10.21.72.251:8085` |
| 8084 | B:8804 | geometry-mvs | `GEOMETRY_MVS_HTTP_URLS=http://10.21.72.251:8084` |
| 8081 | A:8801 | sam3 | `SAM3_HTTP_URL=http://10.21.72.251:8081/sam3` |

(a) socat：`deploy/jumpbox/panoptes-forward@.service` + `deploy/jumpbox/forwards/<服务>.env`，和现有单元同形
（`systemctl enable --now panoptes-forward@sam3d panoptes-forward@geometry-mvs panoptes-forward@sam3`）。
(b) 建议：nginx + TLS，`deploy/jumpbox/compose.nginx.yml` + `nginx.conf`，一个服务一个 `location`，`X-API-Key` 透传；`.env` 改成
`SAM3D_HTTP_URLS=https://10.21.72.251:8443/sam3d` 这种前缀形式，私有 CA 时加 `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`。socat 保留做回滚。

探活：`curl -s -H "X-API-Key: $PANOPTES_SERVICE_API_KEY" http://10.21.72.251:8085/healthz` → `{"ok":true,"model":"sam3d","gpu":"NVIDIA A100…","queue_depth":0}`；
或一次性 `make check-env-live`。

## 5. 验收 A1–A6（计划 §0；交接当天在你们的 jump + A100 上走一遍）

| # | 验收 | 证明它的命令 / 看什么 |
|---|---|---|
| A1 | 新机器 clone → `.env` → `make up` → `panoptes run` → 报告可看 | §0 四步原样跑；`make check-env` 全绿；`make smoke` 5/5；`make run CELL=090` 打印 `DONE 090` 和报告地址；浏览器打开该地址看到报告和测量层（置信度 chips） |
| A2 | 090 / 030 的数字与已发布版一致 | `REPRODUCE-PROMPT.md` §5 的表：`data/checks/bbab-geom/<cell>-mvs-fill-padded/fill-record.json` 与仓库里逐字节相同（`diff`）；几何层现场值 090 右罩壳 24.0 / 右横杆 21.0（4 值 MAE 0.50 cm），030 左 23.6 / 右 24.6；急停尺度 3.2371372 / 3.5616493；组装 v2 后最低点 < −2 cm 的物体 0 / 0；G1–G8 明显错误 0 / 1（防护板，已知）；轮廓 IoU 0.823 / 0.762。容忍：现场值 ±0.1 cm，IoU ±0.01，SAM 3D 候选 ±0.02（择优换照片不算失败，记下来） |
| A3 | 两张 A100 同时用 | `make run CELL=090 & make run CELL=030 & wait`；跑的时候 `curl …:8085/healthz` 和 `…:8084/healthz` 都报 `running:1`（或 `panoptes status`）；两份报告都出来 |
| A4 | 换存储不改代码 | 在 ehs-spatial：`PANOPTES_DATABASE_URL=mongodb://… PANOPTES_BLOB_ROOT=s3://… uv run pytest tests/ -q -k "repository or blob or platform"` 全过，再用 `postgresql://` + 目录跑同一组全过；`make run CELL=090 ARGS="--from S7"` 在 Mongo + S3 下导入导出成功 |
| A5 | 换模型位置不改代码 | S4 跑三次：`SAM3D_BACKEND=http`（你们的卡）、`SAM3D_BACKEND=local`（流程机本进程，需本机 GPU）、`SAM3D_BACKEND=modal`（我们的 Modal，需 `modal token`），每次 `make run CELL=090 ARGS="--from S4"`；三份 `ledger.json` 的 `model_info.weights_sha256` 相同，IoU 差 ≤ 0.02 |
| A6 | 没有第二处配置、没有机器路径、没有手工步骤 | `make check-env`；`grep -rnE "10\.21\.|172\.184|/Users/|/private/tmp" ehs_spatial serving scripts deploy` 只命中文档/注释；`.github/workflows/ci.yml`：lint、shell / YAML / compose / Dockerfile 语法全绿；`pytest tests/`（1500+）只允许 `CHANGELOG.md` 列出的 6 个 `main` 上已有的平台版本钉测试失败（`versions.json` / delivered profile，与交付链路无关） |

## 6. 排障

| 现象 | 原因 / 处理 |
|---|---|
| `/healthz` 连不上 | jump 转发没起（`systemctl status 'panoptes-forward@*'` / nginx 容器日志）或 GPU 机容器没起（`make logs SERVICE=…`）；先在 GPU 机本机 `curl localhost:8805/healthz` 分清是哪一段 |
| `/healthz` 返回 `ok:false` / 503 | 模型没加载：看 `make logs` 的启动行——权重卷路径（`WEIGHTS`）、sha256 校验、显存不够（`nvidia-smi`） |
| 401 `unauthorized` | 两边 `PANOPTES_SERVICE_API_KEY` 不一致（GPU 机 `.env` vs 流程机 `.env`），或代理把 `X-API-Key` 头去掉了（`curl -v` 对比直连与经 jump） |
| 429 `busy` | 该卡队列满（`PANOPTES_SERVICE_MAX_QUEUE`，默认 4）；客户端默认退避重试 3 次。长期如此：调大，或在 `*_HTTP_URLS` 多加一张卡的根 |
| job 一直 `running` / 超时 | `GET /v1/jobs/{id}` 不变：`make logs SERVICE=sam3d` + `nvidia-smi` 看是否真在算；超时阈值 `PANOPTES_SERVICE_TIMEOUT_S`（默认 1800 s）。中断后 `make run CELL=090 ARGS="--from S4"` 接着来：步骤幂等，`input_sha256` 命中服务端缓存直接返回 |
| 启动报 weights sha256 mismatch | 服务拒绝启动并点名文件：`python scripts/onprem/fetch_weights_*.py --cache $WEIGHTS --verify` 找出坏文件，重拉那一个；不要改清单。换权重 revision 必须同时改 `serving/registry.yaml` |
| `check_env` 报 `not in env.template` | 有人加了第二处配置：把键登记进 `env.template`（一行注释）或删掉 |
| 数字对不上 | 不改评判器、组装、检查模块来"对上"；按 A2 的容忍记差异。SAM 3D 在不同 GPU 上有 ±0.02 IoU 的非确定性，响应里带 GPU 型号 |

## 7. 回滚（一行）

任何一个服务出问题，`.env` 里对应的后端改回去，重跑即可，零代码改动：

```
SAM3D_BACKEND=modal          # 或 local（流程机本进程跑同一函数体，需本机 GPU）
GEOMETRY_MVS_BACKEND=modal
SAM3_BACKEND=fal   GEOMETRY_BACKEND=replicate   MOGE_BACKEND=modal      # v0 三个的云端默认
```

`modal` 需要流程机有我们 workspace 的 `modal token`；`fal` / `replicate` 需要 `FAL_KEY` / `REPLICATE_API_TOKEN`。
jump 从 nginx 回到 socat：把 5 个 URL 改回端口形式。

## 8. 边界

- GPU 机只做无状态推理：不存 runs、不接公网、不碰业务数据；日志无载荷无密钥。
- 不改算法与参数；不训练；Pi3X / RecGen 不进交付；不上 Kubernetes；反馈助手（sqlite）不动；VLM（Gemini）不在 `panoptes run` 的链路里。
- 工厂照片只在私有仓库（`research/module-swap-2026-10-07/data/` 的冻结输入）；`.env`、`.platform/`、权重永远不进 git。

---

## History

2026-08-30 的交接总纲（四图 MVP、tarball、公开 serving 仓库）已移到 `docs/archive/HANDOFF-history-2026-08-30.md`，仅供备查；`panoptes-serving` 仓库已冻结。
