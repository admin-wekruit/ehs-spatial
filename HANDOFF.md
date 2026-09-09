# Panoptes 交接总纲（Master Handoff）— 2026-09-09

一句话：拍 1–4 张工位照片 → 三维重建 + 分割测距 → policy 合规判定 →
一个联动的交互报告页（点一个物体，照片/3D/CAD/平面图四处同时高亮）→
审核签字 + Agent 对话，全在同一页。

## 地址

| 资源 | 地址 | 权限 |
|---|---|---|
| 主仓库（全部代码 + 历史） | https://github.com/admin-wekruit/ehs-spatial 分支 `feature/ehs-spatial-mvp` | **私有** |
| 最新提交 | `5dc4aac` — 2×2 联动选择 | — |
| 公开镜像（无照片、无 git 历史，随时可 clone） | https://github.com/admin-wekruit/panoptes-serving | **公开**，最新 `e2f292e` |
| 本机运行 | `http://127.0.0.1:7860`（工作台）/ `http://127.0.0.1:7860/report/<run_id>`（报告页） | 本机 |

## 起服务

```bash
cd ~/Desktop/Tesla/ehs-spatial
uv run --env-file .env uvicorn ehs_spatial.serve:app --host 127.0.0.1 --port 7860
```

**⚠️ 重启前必查**：`for f in runs/*/deep_report.status; do cat "$f"; done` ——
若有 `detect`/`inventory`/`report` 状态说明有深链在跑，等它完或接受中断
（链路是可恢复的，重启后 `resume_interrupted_chains()` 会自动续跑，但会多等
几分钟）。

## 这次做完的事（今天新增：2×2 联动）

用户原话："我们点一个物体四个要交互，所以可以是2x2？的report"。

共享键 = `inv`（inventory.json["objects"] 里的序号）。四个面板：

```
[ 原图点选 (canvas)        |  交互 3D (相机锚定 viewer) ]
[ CAD 平面图 (可点多边形)   |  交互平面 SVG + 距离矩阵   ]
```

点任意一处 → 一条 JS 选择总线广播 `inv` → 其余三处同步高亮。3D 面板通过
`postMessage`（`panoptes:select` 进 / `panoptes:selected` 出）跨 iframe 联动。

**实测证据**（独立 subagent + 我自己直接像素级复核，不是靠猜）：点 CAD
多边形后，原图 canvas 真实重绘了 5410 个像素；同时 3D viewer 的
`selectedInv()` 和平面 SVG 的 `aria-pressed` 都同步变成同一个 `inv`。

顺带修的两件事：
- **平面视图删除**：报告里那个"小蓝块+相机在原点"的图是 pipeline 内部
  中间产物（纯文本 SAM 抓到的围栏碎片凸包），不该出现在产品报告里，已删。
- **3D 颗粒度**：之前报告内嵌 3D 被压到 14 万点；现在报告页通过
  `/report/<id>/viewer` 按 URL 加载完整 viewer（60 万点上限），下载的
  自包含 HTML 仍保留内嵌版本。

## 上一轮做完的事（报告即产品）

- 报告 tab = 历史列表（时间倒序，含判定/审核状态），点一行跳到独立的
  `/report/<run_id>` 页面——不再是 iframe 套娃。
- 报告页顶部固定导航：返回工作台 / 历史 / 下载 HTML / 跳到审核+Agent。
- 14 段完整报告：header(审核+版本戳) · 判定 · VLM 枚举与去向 · 检测清单
  (含拒绝理由) · 联动 2×2(原图/3D/CAD/平面) · 实体测量(含尺度来源) ·
  回投验证 · 证据图集 · 补测记录 · Agent 对话 · Review 记录 · 附录。
- Review 面板：保存写 `review.json`，报告头部和历史行同步。
- Agent 面板：一个对话框，五个动词（追问/补测/纠错/描述缺漏/调整策略），
  确定性路由，全部写 `chat.jsonl`，纠错和调整策略会重评 policy 并把
  原值存进 `policies.original.json`。
- 分析版本戳（`ANALYSIS_VERSION`）：任何 run 打开时若分析规则已升级，
  自动后台重算，不会让两代分析结果混在产品里。

## 模型后端（随时切换）

`docs/BACKENDS.md` 是唯一权威合同。三个开关：`SAM3_BACKEND` /
`GEOMETRY_BACKEND` / `MOGE_BACKEND`，各自 `fal|modal|http` 三选一（MoGe 默认
modal）。GPU serving 参考实现在 `serving/`（FastAPI，端口 8801/8802/8803）。

## 关键设计不变量（改动前必读）

- 报告 tab 时间倒序列表 = 全部历史（无独立 History tab）
- 深链顺序固定：detect → inventory（inventory 的检测同步读 detections.json）
- VLM 枚举必有交代：实例或 `unresolved.json` 里的原因，禁止静默丢弃
- cell-rectangle 约束：工位是矩形先验，几何行为由
  `tests/test_geometry_invariants.py` + `tests/test_cell_rect.py` 钉死
- 渲染禁用 open3d Visualizer（macOS 主线程死锁史）；点云渲染是 numpy 泼溅
- 2×2 联动的共享键是 `inv`，任何新面板要联动就必须携带这个字段

## 验收基线

- 测试：467 passed, 2 skipped
- 单图判定 ~55s；4 图 ~4 分钟；完整报告链路再 +5–8 分钟（自动后台跑）
- 对话问答 ~11s；补测 agent ~9s

## 已知开放项

- 跨视角实例融合未做（同一物体两帧两个 inv）
- real-clean-02 回投分数上限 0.25（需要第二视角/更干净的接地边）
- 相机高度（camera_height_m）没持久化进 run 目录，报告附录显示"—"
- VLM 关系层、检测层反转为实例主来源、YOLO 蒸馏 —— 见 `docs/RESEARCH_BRIEF.md`

## 密钥与安全

- 公开镜像和所有交接包里都**没有**真实密钥（已验证）；`.env.example` 列了
  需要什么
- 工厂照片（`incoming/`、`docs/demos/`、`runs/`）只在私有仓库，同步公开
  镜像时用 `git archive` + `--exclude` 排除，别手动改这个规则
