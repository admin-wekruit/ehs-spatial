# GPU 机器部署 Handoff — Panoptes 模型服务

把下面整段作为任务交给 GPU 机器上的执行者（Claude 会话或运维）。

---

## 任务

在这台 GPU 机器上部署三个模型推理服务（SAM 3 / MapAnything / MoGe-3），
每个是一个独立 HTTP endpoint。合同（请求/响应 JSON、RLE 编码、端口建议）
以本包内 `docs/BACKENDS.md` 为准，**schema 一个字段都不能改** ——
客户端行为由对方仓库的 tests/test_backends.py 钉死。

## 你拿到的包

`panoptes-serving.tar.gz`，解开后：

- `serving/sam3_service.py` — FastAPI，端口 8801，POST /sam3
- `serving/mapanything_service.py` — FastAPI，端口 8802，POST /geometry
- `serving/moge_service.py` — FastAPI，端口 8803，POST /moge
- `serving/requirements.txt`
- `docs/BACKENDS.md` — 冻结合同（唯一权威）
- `modal_apps/*.py` — 同一套推理体的云上参考实现（已在生产验证），
  行为有疑问以它们为准

## 步骤

1. 环境：Python 3.11+，先装匹配本机 CUDA 的 torch/torchvision，再
   `pip install -r serving/requirements.txt`。
2. HF 权重访问：
   - `export HF_TOKEN=<这台机器自己的 token>`
   - **facebook/sam3 是 gated 模型**：该 token 的账号必须先在
     https://huggingface.co/facebook/sam3 点过 "Agree and access"，
     否则 401。facebook/map-anything 和 Ruicheng/moge-3-vitl 不 gated。
3. 起服务（各自一个进程，建议 systemd/supervisor 常驻）：
   ```
   uvicorn sam3_service:app        --host 0.0.0.0 --port 8801
   uvicorn mapanything_service:app --host 0.0.0.0 --port 8802
   uvicorn moge_service:app        --host 0.0.0.0 --port 8803
   ```
   首次启动会下载权重（SAM3 ~3GB、MapAnything ~2GB、MoGe ~1.5GB），
   之后走 HF 缓存。
4. 冒烟验收（每个都必须过）：
   ```
   curl -s localhost:8801/healthz
   curl -s localhost:8802/healthz
   curl -s localhost:8803/healthz
   # 真图冒烟：任选一张 jpg
   B64=$(base64 -w0 test.jpg 2>/dev/null || base64 -i test.jpg)
   curl -s -X POST localhost:8801/sam3 -H 'Content-Type: application/json' \
     -d "{\"image_b64\":\"$B64\",\"prompt\":\"fence\"}" | head -c 300
   curl -s -X POST localhost:8803/moge -H 'Content-Type: application/json' \
     -d "{\"image_b64\":\"$B64\"}" | head -c 200
   curl -s -X POST localhost:8802/geometry -H 'Content-Type: application/json' \
     -d "{\"inputs\":[\"data:image/jpeg;base64,$B64\"]}" | head -c 300
   ```
   验收标准：sam3 返回非空 `rle`+`scores`；moge 返回 `ply_b64`+`intrinsics`；
   geometry 返回 `data`（每图一项）+`point_cloud`。
5. 显存：三个服务合计约 20GB bf16。单卡 24GB 可以同时常驻；更小的卡
   就按需起停或加 `CUDA_VISIBLE_DEVICES` 分卡。
6. 回报三个 URL（内网地址 + 端口 + 路径）。

## 我们这边切换（部署完成后，workbench 机器上）

```
SAM3_BACKEND=http      SAM3_HTTP_URL=http://<gpu>:8801/sam3
GEOMETRY_BACKEND=http  GEOMETRY_HTTP_URL=http://<gpu>:8802/geometry
MOGE_BACKEND=http      MOGE_HTTP_URL=http://<gpu>:8803/moge
```

写进 .env、重启 app，零代码改动。任何一个服务出问题，把对应
`*_BACKEND` 改回 `fal` / `replicate` / `modal` 即回滚。

## 边界

- 这台 GPU 机器只做无状态推理：不存 runs、不接公网、不碰业务数据。
- VLM（Gemini）不在此次范围；将来内部大模型走 OpenAI 兼容 adapter 另接。
