# Model services on the GPU boxes

One model = one FastAPI service = one container; one compose stack per card (`serving/registry.yaml` says which services sit on
which card). Contract: `docs/BACKENDS-v1.md`. Config: `.env` only (`env.template` section 1, the GPU services keys).

```sh
git clone -b main https://github.com/admin-wekruit/ehs-spatial.git && cd ehs-spatial   # one repository: services, function bodies, docker recipes, pipeline
make env                                                     # .env from env.template: PANOPTES_SERVICE_API_KEY, WEIGHTS (section 1)
python3 scripts/onprem/fetch_weights_sam3d.py    --cache $WEIGHTS   # card A (HF_TOKEN; gated facebook/sam-3d-objects)
python3 scripts/onprem/fetch_weights_geometry.py --cache $WEIGHTS   # card B (da3-base, roma_outdoor, roma_dinov2, moge3)
make images GPU=a && make up GPU=a && make smoke GPU=a       # sam3d :8805 + sam3 :8801
make images GPU=b && make up GPU=b && make smoke GPU=b       # geometry-mvs :8804 + moge :8803 + mapanything :8802
```

- `make context` assembles the build context (`deploy/context.sh`): ehs-spatial's `stage_context.sh` layout (`SRC/workcell`,
  `SRC/serving`, its `.dockerignore` whitelist) plus this repo's `serving/` and `deploy/`.
- `deploy/Dockerfile.sam3d` and `deploy/Dockerfile.geometry` build FROM the ehs-spatial images (`docker/sam3d.Dockerfile`,
  `docker/{da3,geometry,workcell-gpu}.Dockerfile`), so every pin is the audited one and lives in one place; `make images` builds
  the bases first. `deploy/Dockerfile.serving` is the three `docs/BACKENDS.md` services as `docs/archive/serving-HANDOFF-v0-services.md` installs them.
- Air-gapped boxes: build on a connected machine, then ehs-spatial `scripts/onprem/airgap.sh save / load`.
- Verify: `make test` (fake backends, no GPU) and `serving/smoke_v1.sh URL` (a running service; the real model when deployed).
  `PANOPTES_CONTRACT_URL=http://<gpu-a>:8805 pytest tests/contract/live.py` runs the contract tests against the live service.


---

# deploy/ — GPU 机、jump VM、报告站

| 目录 / 文件 | 跑在哪 | 用途 |
|---|---|---|
| `compose.gpu-a.yml`、`compose.gpu-b.yml`，根目录 `Makefile` 的 `up GPU=a\|b` / `down` / `smoke` / `logs`（WP3） | 两台 GPU 机，或一台两卡 | 每卡一个 compose 栈，读仓库根 `.env` 第 1 节：卡 A = `sam3d :8805` + `sam3 :8801`；卡 B = `geometry-mvs :8804` + `moge :8803` + `mapanything :8802`。端口、权重 revision、显存由 `serving/registry.yaml` 定 |
| `jumpbox/` | jump VM `10.21.72.251` | 把 jump 的端口转到 GPU 机端口：(a) socat systemd 单元（客户现有的形式）或 (b) nginx + TLS（建议） |
| `publications/` | 任一台看报告的人能访问的机器 | 报告站：nginx 容器（查看器 + 测量层 + `/api/` → 发布服务）或 S3 静态托管，见 `publications/README.md` |

应用永远指向 jump，不指向 GPU 机 IP（今天是 `172.184.217.34`）：换 GPU 机只改 jump 的转发，`.env` 不动。

## jump VM：端口表（两种方案都按这张表）

| 服务 | 合同 | GPU 卡:端口 | (a) socat jump 端口 | (b) nginx 前缀 | `.env` 里的行（socat 形式） |
|---|---|---|---|---|---|
| sam3d | v1 `docs/BACKENDS-v1.md` | A:8805 | **8085**（新） | `/sam3d/` | `SAM3D_HTTP_URLS=http://10.21.72.251:8085` |
| geometry-mvs | v1 | B:8804 | **8084**（新） | `/geometry-mvs/` | `GEOMETRY_MVS_HTTP_URLS=http://10.21.72.251:8084` |
| sam3 | v0 `docs/BACKENDS.md` | A:8801 | **8081**（新） | `/sam3/` | `SAM3_HTTP_URL=http://10.21.72.251:8081/sam3` |
| mapanything | v0 | B:8802 | 8080（已有） | `/mapanything/` | `GEOMETRY_HTTP_URL=http://10.21.72.251:8080/geometry` |
| moge | v0 | B:8803 | 8090（已有） | `/moge/` | `MOGE_HTTP_URL=http://10.21.72.251:8090/moge` |

sam3 今天和 mapanything / moge 同在一台 Azure A100 上；两卡布局把它挪到卡 A（和 sam3d 一起），jump 新开 8081。
一个工位的 SAM 3D 候选要分到两张卡时，卡 B 也起一个 sam3d（WP3 compose 的可选 profile），jump 再开一个端口（例 8095 → B:8815），
`.env` 写成 `SAM3D_HTTP_URLS=http://10.21.72.251:8085,http://10.21.72.251:8095`，provider 按 `/healthz` 的 `queue_depth` 选最空的卡。

### (a) socat systemd 单元 — `jumpbox/panoptes-forward@.service` + `jumpbox/forwards/<服务>.env`

和现有 8080 → 8802、8090 → 8803 的单元同一形式：一个端口一个实例，`%i` = 服务名 = `/etc/panoptes-forward/<服务>.env`（`LISTEN_PORT`、`TARGET`）。

```sh
cd deploy/jumpbox
sudo cp panoptes-forward@.service /etc/systemd/system/
sudo mkdir -p /etc/panoptes-forward && sudo cp forwards/*.env /etc/panoptes-forward/   # 卡 B 在另一台 VM：只改对应文件的 TARGET 主机
sudo systemctl daemon-reload
sudo systemctl enable --now panoptes-forward@sam3d panoptes-forward@geometry-mvs panoptes-forward@sam3   # 8080 / 8090 的旧单元保留
systemctl status 'panoptes-forward@*' --no-pager
```

探活（在任何能到 jump 的机器上；v1 服务的 `/healthz` 也要 key）：

```sh
curl -s -H "X-API-Key: $PANOPTES_SERVICE_API_KEY" http://10.21.72.251:8085/healthz   # {"ok":true,"model":"sam3d","gpu":"NVIDIA A100...","queue_depth":0,...}
curl -s -H "X-API-Key: $PANOPTES_SERVICE_API_KEY" http://10.21.72.251:8084/healthz   # geometry-mvs
curl -s http://10.21.72.251:8081/healthz                                             # sam3（v0：无 key）；8080、8090 同
make check-env-live                                                                  # 一次探完 .env 里每个 URL
```

### (b) nginx + TLS（建议）— `jumpbox/compose.nginx.yml` + `jumpbox/nginx.conf`

一个 `server`（8443，TLS），每个服务一个 `location`，前缀之后的 URI 原样交给服务根；`X-API-Key` 原样透传（nginx 默认转发全部客户端头，服务端校验）。

```sh
cd deploy/jumpbox
cp jumpbox.env.example .env                      # GPU_A_IP / GPU_B_IP（一台两卡：同一个 IP）
mkdir -p certs && openssl req -x509 -newkey rsa:2048 -nodes -days 825 -subj /CN=10.21.72.251 \
  -addext subjectAltName=IP:10.21.72.251 -keyout certs/jump.key -out certs/jump.crt   # 或客户 CA 签的证书对
docker compose -f compose.nginx.yml up -d
curl -s --cacert certs/jump.crt https://10.21.72.251:8443/healthz                                                   # jump 本身
curl -s --cacert certs/jump.crt -H "X-API-Key: $PANOPTES_SERVICE_API_KEY" https://10.21.72.251:8443/sam3d/healthz   # 穿到卡 A
```

`.env` 改成前缀形式（根 = `https://10.21.72.251:8443/<前缀>`）：

```
SAM3D_HTTP_URLS=https://10.21.72.251:8443/sam3d
GEOMETRY_MVS_HTTP_URLS=https://10.21.72.251:8443/geometry-mvs
SAM3_HTTP_URL=https://10.21.72.251:8443/sam3/sam3
GEOMETRY_HTTP_URL=https://10.21.72.251:8443/mapanything/geometry
MOGE_HTTP_URL=https://10.21.72.251:8443/moge/moge
SSL_CERT_FILE=/etc/panoptes/jump.crt          # 自签或私有 CA 时；公有 CA 不用
REQUESTS_CA_BUNDLE=/etc/panoptes/jump.crt
```

要求：provider 把路径**拼**在根后面（`root + "/v1/sam3d/jobs"`），不能用 `urljoin` 丢掉前缀——`docs/BACKENDS-v1.md` 的 "service roots" 按这个理解。
回滚：`.env` 的五行改回 socat 端口即可，socat 单元一直在。

## 排障入口

| 现象 | 看哪 |
|---|---|
| `connection refused` | jump：`systemctl status panoptes-forward@sam3d` / `docker compose -f compose.nginx.yml logs`；GPU 机：`make logs SERVICE=sam3d` |
| `/healthz` 返回 `ok:false` / 503 | 模型没加载：权重卷（`WEIGHTS`）、sha256 校验、显存；`make logs` |
| 401 | 两边 `PANOPTES_SERVICE_API_KEY` 不一致，或代理把头去掉了（`curl -v` 看请求头） |
| 429 | 队列满：`PANOPTES_SERVICE_MAX_QUEUE`，或多加一张卡的根到 `*_HTTP_URLS` |
