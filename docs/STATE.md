# 当前状态（唯一权威；2026-10-08）

这份文件说"系统现在是什么"。和它冲突的其它文档以它为准。怎么跑看 `HANDOFF.md`；怎么复现数字看 `research/module-swap-2026-10-07/REPRODUCE-PROMPT.md`；
阶段 / 里程碑看 `docs/MILESTONES.md`；历史文档在 `docs/archive/`（只读，不是现状）；研究文档在 `docs/research/`（提案和实验，不是已交付系统）。

## 1. 仓库与版本

| 项 | 值 |
|---|---|
| 唯一仓库 | https://github.com/admin-wekruit/ehs-spatial ，分支 `main`（默认分支仍是 `feature/ehs-spatial-mvp`，所以 clone 要 `-b main`） |
| 版本 | `VERSION` 1.0.0-rc1（企业版交接候选，2026-10-07）；`CHANGELOG.md` |
| 被冻结的旧仓库 | `panoptes-serving`（README 指向本仓库；不要再 clone） |
| Python | 3.12，`uv sync --frozen`（`uv.lock`）；控制台命令 `panoptes run|status` |
| 测试 | `PANOPTES_FAKE_MODEL=1 PANOPTES_WORKCELL=$PWD pytest tests` → 1530 通过、6 个已知失败（`tests/test_report_runner_reproduce.py` delivered-profile ×4、`tests/test_report_runner_stages.py` versions.json + splat_train 默认值：平台侧过期的版本钉，与交付链路无关） |

## 2. 交付的流程（照片 → 报告），只有这一条

| 步 | 做什么 | 实现 | 跑在哪 |
|---|---|---|---|
| S1 | 冻结照片，518 规范网格 | `scripts/research/prepare_capture_evidence.py` | CPU |
| S2 几何 | **geometry-mvs**：DA3-BASE 起始 → RoMa v1 outdoor 匹配 → numpy LM 光束法平差 → 两视角三角化；MoGe-3 掩码内补洞（v2 规则） | `modal_apps/geometry_clean_ab.py`、`bundle_adjust.py`、`moge3_app.py`；服务 `serving/geometry_mvs_service.py` | GPU 卡 B :8804 |
| S3 掩码 | SAM 3（冻结） | 服务 `serving/sam3_service.py` | GPU 卡 A :8801 |
| S4 补全 | **SAM 3D Objects**，每张照片出候选，统一择优 | `completion_ab.py --stage sam3d`；服务 `serving/sam3d_service.py` | GPU 卡 A :8805 |
| S5 组装 | 9-DOF 轮廓 + 边界 + 深度，**v2 加地面接触罚项** | `scripts/research/assemble_lucida_scene.py`（`floor_penalty`） | CPU |
| S6–S9 | 报告、平台导入 / 发布、检查（shape / floor / lines / plane_stereo / clearance / lower_edge / box_faces / 明显错误 G1–G8） | `build_capture_report.py`、`platform/publish-swap-20261007.py`、`workcell_layer_trial.py` | CPU |
| S10–S14 | 对比表、测量层、报告站 | `compare_json.py`、`build_swap_layer.py`、`publish_site.sh` | CPU |

**不在交付里**：Pi3X、RecGen（只在对比表里出现）、DA3-LARGE、pycolmap；训练；Kubernetes。
**Gemini（VLM）**：平台的 Gradio 工作台、报告聊天、补测 agent、方向判定用它；`panoptes run` 不调用；on-prem 不设 `GEMINI_API_KEY`（`env.template` 第 57 行）。

## 3. 服务与契约

| 服务 | 端口 | 契约 | 卡 |
|---|---|---|---|
| sam3d（SAM 3D Objects @2e73555） | 8805 | v1（`docs/BACKENDS-v1.md`：/healthz、/v1/info、/v1/sam3d、/v1/sam3d/jobs、X-API-Key、幂等 input_sha256） | A |
| geometry-mvs（DA3-BASE @f4a6c9b + RoMa @77f8d68 + MoGe-3 @184008f） | 8804 | v1 | B |
| sam3 / mapanything / moge（v0） | 8801 / 8802 / 8803 | v0（`docs/BACKENDS.md`） | A / B / B |

客户端：`SAM3D_BACKEND|GEOMETRY_MVS_BACKEND = http|local|modal`，`*_HTTP_URLS` 逗号列表（按 queue_depth 最小），`PANOPTES_SERVICE_API_KEY/TIMEOUT_S/RETRIES`。
存储按 URL 选：`PANOPTES_DATABASE_URL` postgres:// 或 mongodb://（`PANOPTES_MONGO_PREFIX`），`PANOPTES_BLOB_ROOT` 目录或 s3://（`AWS_ENDPOINT_URL`）；`docs/STORAGE.md`。
唯一配置面：`.env`（`env.template` 列全；`make check-env`）。拓扑：流程机 → jump 10.21.72.251（socat 8085/8084/8081）→ A100 VM。

已证明：真 A100 上 sam3d 服务经 http provider 端到端（`docs/E2E-2026-10-07.md`，left_post 93036 vs 93034 顶点，10.6 s GPU）。
客户侧未关闭：默认分支切 main、两张 A100 上 docker 构建 + `make up` + `make smoke`、真实 Mongo / S3、geometry-mvs 真 GPU 跑、验收 A1–A6（`HANDOFF.md` §5）。

## 4. 已发布的结果（2026-10-07，报告站 `panoptes-publications-estop`）

| 工位 | 推荐版本（MVS + 补洞 + SAM 3D + 组装 v2） | 明显错误 G1–G8 | 轮廓 IoU | 现场值 |
|---|---|---|---|---|
| 090 | `a9a6e0a0…` | 0（原版 Pi3X + RecGen 为 8） | 0.823 | 罩壳下沿 24.0 cm、横杆 21.0 cm（4 值 MAE 0.50 cm） |
| 030 | `fafdeb6b…` | 1（防护板，已知） | 0.762 | 左 23.6、右 24.6 cm |

全部版本、链接、结论：`research/module-swap-2026-10-07/REVIEW-LINKS-2026-10-06.md`。测量层在 Pages 仓库 `admin-wekruit/panoptes-workcell-report`。
冻结输入与中间结果：`research/module-swap-2026-10-07/data/`（0.5 GB，在 git 里）。

## 5. 报告里有什么、没有什么

有：对象盒（尺寸、离地高、每维 σ 部分有、置信度、每面被哪些照片看到）、事实、流水线面板、现场值对比。
**没有机器判定**：平台的政策引擎（`ehs_spatial/platform/policy_engine.py`，7 个 2.5D 谓词 + ZEN 适用性决策表 + LLM 离线编译）按设计弃权——没有审核员适用性断言、尺度状态不是 operator_anchored、只有下沿没有整高、没有危险区模型。桥接脚本 `scripts/workcell_policy_evidence.py` 只出"每条规则还缺什么"的台账。

## 6. 正在做的（研究，未交付）

判定层（Phase 4）：调研 + 试跑完成，**没有进产品**。入口 `docs/research/README.md`。结论：架构 = 薄 3D 场景图（几何算出的米制关系，Spark-DSG 兼容）→ clingo 确定性判定 + 护带 + 五态；
规范侧用"抽取 + 对齐 + 检索 + 验证过的检查合成"代替逐句模板（用户否决了逐句模板法）。等用户提供 safety specs 和模型 key 后开工；先不开工。

## 7. 给复现 / 接手 session 的硬规则

- 只信本文件 + `HANDOFF.md` + `REPRODUCE-PROMPT.md`；`docs/archive/` 是历史，`docs/research/` 是提案；`research/module-swap-2026-10-07/notes/` 是冻结的实验记录（措辞是当时的，路径已参数化）。
- 几何是 geometry-mvs，不是 Pi3X；补全是 SAM 3D，不是 RecGen。
- `.env` 是唯一配置；HF token 只在拉权重时用，不进任何文件；`.platform/` 不读、不提交。
- 不改算法与参数；数字对不上按 `REPRODUCE-PROMPT.md` §5 的容忍记录差异，不"调到对上"。
