# Panoptes 全量 Handoff

自进化工厂视觉理解 harness：拍 1-4 张工位照片 → 三维重建 + 分割测距 →
policy 判定 → 交互式合规报告 + 对话补测 agent。本文档 = 在新机器上把
整套系统跑起来的完整说明。

## 包内容

- `panoptes-full.tar.gz` 解开即完整仓库工作树（含 git 历史 bundle）：
  - `ehs_spatial/` — 产品本体：`app.py` 是前端（Gradio：Workbench 提交 /
    Video / 补测 Refine / 报告 tab）+ pipeline + providers + 报告生成
  - `scripts/` — 深链与工具（detect_devices、scene_inventory、
    verify_reprojection、policy_compile…）
  - `serving/` — 内部 GPU 三个模型服务 + 部署 HANDOFF.md
  - `modal_apps/` — 同一套推理的 Modal 云托管版（已部署可用）
  - `docs/BACKENDS.md` — 模型后端冻结合同
  - `tests/` — 414 项回归（含几何不变量）
  - `outputs/policies/compiled/` — 已编译的 policy specs（UI 只加载这些）
  - `repo.bundle` — 完整 git 历史（`git clone repo.bundle panoptes`）
  - `.env.example` — 所需环境变量清单（自己填 key，**没有任何真实密钥**）
- `panoptes-runs-sample.tar.gz`（另一个包，~770MB）— 4 个真实 run
  （3 个 canonical 测试集 + 1 个 BOR1 工位），几何不变量测试和报告 demo
  依赖它们；不需要可以不解。

## 起服务（app 机器）

```bash
tar xzf panoptes-full.tar.gz && cd ehs-spatial
uv sync                       # Python 3.12；没有 uv: pip install uv
cp .env.example .env          # 填 GEMINI_API_KEY + 所选后端的 key
# 可选：解 runs 样例包到 ./runs
uv run pytest tests/ -q       # 期望: 414 passed（缺 runs 样例会 skip 少量）
uv run --env-file .env python -c \
  "from ehs_spatial.app import build_app; build_app().launch(server_name='0.0.0.0', server_port=7860)"
```

打开 http://<机器>:7860。

## 模型后端三种形态（随时 env 切换，零代码）

| 形态 | 配置 | 说明 |
|---|---|---|
| 云 API（默认） | 填 FAL_KEY / REPLICATE_API_TOKEN | 开箱即用 |
| Modal 自托管 | `SAM3_BACKEND=modal` 等 + Modal 账号 | 三个 app 已写好在 modal_apps/ |
| 内部 GPU | `*_BACKEND=http` + 三个 URL | 按 serving/HANDOFF.md 在 GPU 机器部署 |

VLM（Gemini）是第四个座位：adapter 缝在 `ehs_spatial/providers/gemini.py`，
将来换任何 OpenAI 兼容内部端点在此替换。

## 运行拓扑

```
操作员浏览器 ──> app 机器 (Gradio 7860)
                  ├─ pipeline: 方向判定→几何→分割→标定→policy→climb review
                  ├─ 深链(自动后台): 装置检测→inventory 精修→交互报告
                  ├─ runs/ 全部数据落这台
                  └─ 模型调用 ──> 云 API / Modal / 内部 GPU (env 决定)
```

时延（实测）：快速判定单图 ~55s / 4 图 ~4 分钟；完整交互报告再 +5-8 分钟
（自动生成，报告 tab 有进度横幅）。

## 关键设计不变量（改动前必读）

- 报告 tab 的时间倒序列表 = 全部历史（History tab 已按产品决策移除）
- 深链顺序固定：detect → inventory（inventory 的检测同步读 detections.json）
- 任何 VLM 枚举到的物体必须有实例或 unresolved 记录，禁止静默丢弃
- 工位 premise：cell 是矩形（cell-rectangle 约束）；条纹面按几何判语义
- 所有几何行为由 tests/test_geometry_invariants.py + test_cell_rect.py 钉死；
  改几何必须全绿才能合
- 渲染禁用 open3d Visualizer（macOS 线程死锁史）；点云渲染是 numpy 泼溅

## 已知开放项

- 跨视角实例融合未做（同一物体两帧两个实例）
- real-clean-02 回投分数 0.25 上限待压到 0.10（需第二视角）
- 检测层反转为实例主来源、VLM 关系层、YOLO 蒸馏在路线图上
