# Panoptes 交接总纲（Master Handoff）

一句话：拍 1–4 张工位照片 → 三维重建 + 分割测距 → policy 合规判定 →
交互式报告 + 中文对话补测 agent。Gradio app 即前端。

## 所有地址

| 资源 | 地址 | 权限 |
|---|---|---|
| 主仓库（全部代码 + 历史） | https://github.com/admin-wekruit/ehs-spatial 分支 `feature/ehs-spatial-mvp` | **私有** |
| Handoff Release | https://github.com/admin-wekruit/ehs-spatial/releases/tag/handoff-2026-08-30 | **私有** |
| ├ `panoptes-all-in-one.tar.gz` (802MB) | 全部代码 + git bundle + policy specs + 4 个真实 run | 私有 |
| ├ `panoptes-full.tar.gz` (93MB) | 同上但不含 runs | 私有 |
| └ `panoptes-serving.tar.gz` (27KB) | 仅 GPU 模型服务 | 私有 |
| GPU serving 公开仓库 | https://github.com/admin-wekruit/panoptes-serving | **公开** |
| Modal 云托管（当前在用） | workspace `wekruit-livekit-agents`：`sam3-inference` / `moge3-inference` / `mapanything-inference` | Modal 账号 |
| 本机文件副本 | `~/Desktop/panoptes-{all-in-one,full,serving}.tar.gz` | 本机 |

## 场景 A：GPU 机器只部模型服务（推荐拓扑）

给那台机器的 session 贴这段：

> `git clone https://github.com/admin-wekruit/panoptes-serving.git`，
> 按 `serving/HANDOFF.md` 逐步执行：部署三个 FastAPI 服务
> （SAM3 :8801 / MapAnything :8802 / MoGe-3 :8803）。合同以 BACKENDS.md
> 为准，schema 一个字段不能改。facebook/sam3 是 gated 模型——本机
> HF_TOKEN 的账号需先在 https://huggingface.co/facebook/sam3 点同意。
> 首启会下载权重（合计 ~7GB），三服务常驻约 20GB 显存（bf16）。
> 跑完 HANDOFF.md 的 6 条冒烟 curl（3×healthz + 3×真图），全过后回报：
> SAM3_HTTP_URL / GEOMETRY_HTTP_URL / MOGE_HTTP_URL 三个内网地址。

拿到三个 URL 后，在 workbench 机器 `.env` 加：

```
SAM3_BACKEND=http      SAM3_HTTP_URL=http://<gpu>:8801/sam3
GEOMETRY_BACKEND=http  GEOMETRY_HTTP_URL=http://<gpu>:8802/geometry
MOGE_BACKEND=http      MOGE_HTTP_URL=http://<gpu>:8803/moge
```

重启 app 即切换；出问题改回 `fal`/`replicate`/`modal` 一行回滚。

## 场景 B：整套系统搬去新机器

新机器上（需要先 `gh auth login` 本账号，或用只读 PAT）：

```bash
gh release download handoff-2026-08-30 -R admin-wekruit/ehs-spatial -p panoptes-all-in-one.tar.gz
tar xzf panoptes-all-in-one.tar.gz && cd ehs-spatial
uv sync
cp .env.example .env   # 填 GEMINI_API_KEY + 所选后端 key
uv run pytest tests/ -q          # 验收: 414 passed
uv run --env-file .env python -c \
  "from ehs_spatial.app import build_app; build_app().launch(server_name='0.0.0.0', server_port=7860)"
```

完整说明在包内 `HANDOFF_FULL.md`（运行拓扑、设计不变量、已知开放项）。
git 历史：`git clone repo.bundle panoptes`。

## 密钥与安全

- 任何包和公开仓库里都**没有**真实密钥（已验证）；`.env.example` 列了需要什么
- 需要各自准备：GEMINI_API_KEY（VLM）；云模式加 FAL_KEY / REPLICATE_API_TOKEN；
  自托管模式只要 HF_TOKEN（sam3 要点 gated 许可）
- 工厂照片（incoming/、docs/demos/、runs/）只存在于私有仓库和私有 Release —— 别转公开
- SSH 密码、PAT 由你亲手传递，不经任何 agent

## 实测性能基线（验收参照）

- 快速判定：单图 55s / 4 图 242s（replicate 冷启动占 1–2 分钟方差；切自托管后消除）
- 完整交互报告：判定后自动后台生成，+5–8 分钟
- 对话问答 11s / 补测 agent 9s
- 测试基线：414 passed（几何不变量 + 后端路由 + cell-rectangle 全钉死）
