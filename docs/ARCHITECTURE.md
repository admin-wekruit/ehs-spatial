# Panoptes 架构地图与状态（跨仓库）

更新规则：这是活文档。任何 session 改了任何一个仓库的状态，结束前**重新测量**并覆盖 §5 的状态表（日期 + 命令 + 结果），不要凭记忆改。§7 列出每一行怎么测。跨 session 的副本在 Claude 记忆 `panoptes-architecture-map`。

最后测量：2026-10-11（只读、离线、`PANOPTES_FAKE_MODEL=1`，没有 GPU / LLM / 数据库）。

## 1. 仓库与角色（远端都在 GitHub `admin-wekruit`）

| 仓库 / 本地路径 | 分支 @ HEAD | 角色 |
|---|---|---|
| ehs-spatial — `/Users/adam/Desktop/Tesla/panoptes-platform`（worktree） | `main` = `verdict-lab` @ f881ddc；`archive/research-2026-10` @ 2fd7e0a；`wip/viewer-i18n-2026-10-08` @ 3c9b1fc（不要合并） | **清理前**的 Modal 源码 + 第一版判定实验台 `ehs_spatial/verdict`（clingo@2，95 个测试）。清理步骤 1/2/3/5 已做，4/6/8/9/10/11 没做。交付角色已被 argus 取代。 |
| argus — `/Users/adam/Desktop/panoptes-public/phase5-delivery` | `codex/phase5-delivery` @ 87796eb（已推，工作树干净） | **清理后的交付**（Codex session 做的）：包 `argus/{pipeline,checks,platform,providers}` + `web/ tests/ scripts/ docs/`；`panoptes run --cell 030|090`；locales en/zh/nl 各 1783 键。独立复核：第一轮 @ 2d8a419（可作 Modal 源码交付），第二轮冻结 @ dd80ce5（**没做完**，子代理因额度 429 中止）。dd80ce5 之后的 39fa940 → 922c8ce → 87796eb（边缘一致 + 自动部件）**未经复核**，用户 2026-10-10 暂停。 |
| panoptes-verdict-lab — `/Users/adam/Desktop/panoptes-public/phase4-verdict-lab` | `codex/verdict-lab` @ c26d07a | **当前**判定研究：包 `verdict_lab`，命令 `panoptes-verdict`，契约 verdict/2；插件 measurement-layer@3 / scene-json@2 / relations@5 / handwritten@5 / function-library@3 / code-synthesis@3 / redundant-translation@3 / clingo@5 / python@6 / html@1。取代 `ehs_spatial/verdict`（五处双引擎分歧在这里修了）。 |
| phase5-portability | `codex/phase5-portability` @ 4c7b8e5 | 保留源码 == argus 跟踪文件（复核时 0 差异）。只作历史。 |
| phase5-inputs | — | 冻结输入镜像（原图、mask、几何、尺度）。只读，不得覆盖。 |
| phase5-research-reference | — | 清理前源码，含旧的照片→SAM 链（`ehs_spatial/pipeline.py::_segment_unit` → `providers/sam3.py::SAM3Adapter.segment` → `sam_subscribe`，`SAM3_BACKEND=fal|modal|http` → `modal_apps/sam3_app.py::Sam3.segment`）。没有接进 argus。 |
| phase5-artifacts | — | 复核证据；`automatic-parts-2026-10-10/HANDOFF-LEADER-PIPELINE-2026-10-10.md`（流水线交接，已答复）；`cloud-review-2026-10-09/` 私人站打包脚本。 |

站点：原报告应用 `https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html`（schema-1 层；`4b58dbd2… / cd84d3fb…` 是 Pi3X+RecGen 原版，不是目标；推荐 MVS 层 = 090 `a9a6e0a0…` / 030 `fafdeb6b…`）；私人比较站 `https://panoptes-verdict-review-20261009.windy-crab-9030.chatgpt.site/workspace`（只展示保存结果：12 runs / 236 verdicts / 0 个 validated box）。

## 2. 交付链路（argus `argus/pipeline/cli.py` 的 `ITEMS`；离线 `--dry-run` 已验证 S2a→S8 全部打印）

S2a `providers.geometry_mvs.run`（DA3-BASE + RoMa + MoGe-3，Modal GPU）→ S2b BA + 两视角三角化 → S2c evaluator analyse/export → S2d MoGe-3 mask 内 fill + 字段门（MAE ≤ 1.56 cm、max ≤ 3 cm）→ S4a SAM 3D Objects（Modal，`completion.py`）→ S4b assemble + `select_sam3d` → S4c `swap_generation` + `pin_run` → S5 `assemble_scene`（9-DOF，地面接触 v2）→ S6 `build_capture_report` → S7 `platform.publish_capture` → S8 `serve_export` / `run_stages` / `build_swap_layer`（schema-2 测量层；只序列化 S7 的 revision，不重算）。

要点：
- 照片 → 整物体 2D mask **不是**链路里的阶段：链路从冻结的 `PANOPTES_RUNS/<run>/evidence/objects.json` 起（030 = 2026-10-04 研究 run 的 SAM3，backend 未记录；090 = SAM2.1 本地 MPS 提示 19 条 + 6 条旧 canonical mask）。`panoptes run` 内部唯一的 SAM 是 SAM 3D Objects。
- 87796eb 起 S7 没有 provider manifest 就拒绝运行：钉死 Gemini discovery + FAL `sam-3-1/image-rle` + Gemini model review（`publish_capture.py:53-56`，`reconstruction.py:2627-2646`）。不是用户需求；2026-10-10 裁定：撤硬前置，分割走 `SAM3_BACKEND` 路由（自托管 Modal 为默认），discovery 不用 LLM。
- `MEASUREMENT_ROLES` 的 coverage=`visible_interval`，而 `geometry_engine.py:196` 要求 `full_target` → 当前代码下自动围栏 / 光幕量值永远 `value=None, flags=['untrusted']`。
- 手工 `partMasks` 已从 `cells/*.json` 删掉，但 `lower_edge.py` / `box_faces.py` / `shape_core.py` / `geometry_engine.py` 的代码路径仍接受 `opts.partMasks`。
- Modal 接缝 = `argus/` 下 9 个 `import modal` 文件（completion、geometry_clean_ab、field_evaluator、sam3d_modal、moge3_modal、publication_modal、recgen_transport、reconstruction、executor）。客户只换这些。

## 3. 判定层：在流水线之外，是设计

报告只有测量、盒子、事实，**没有机器合规判定**（两个仓库的 STATE.md 都这么写）。流水线内的残留：`argus/pipeline/policy.py`（PolicySpec → 几何谓词，误差带内弃权），只被 `platform/policy_engine.py` 调用。判定实验台单独消费 schema-2 层：L1 scene → L2 relations → L3 monitor → L4 clause graph → L5 rule pack → L6 clingo/python 五态（护带 k=2）→ L7 报告。

真实工位现状：`zones=[]`、`coverage=null`、`declared_inputs={}` → 基线 030 = 9/1/1/6/1，090 = 10/2/2/7/1（PASS/FAIL/NEEDS_MEASUREMENT/NEEDS_INPUT/CANNOT_DETERMINE），金标是**试跑输出**（provisional）；function-library 编译 9 条、绑定 0 条（缺 `perimeter_of` / `covers_opening` 区域谓词）→ 全部 NEEDS_INPUT；至今 0 次真实 LLM 调用（Mac 上没有 Claude 凭证）。计划阶段（`docs/research/verdict-layer-plan-2026-10-08.md`）：A 完成，B 一半，C 代码在 / 没真跑，D、E 未开始。

## 4. 文件管理（客户审查的 11 步）

在 argus 里**已完成**：一个包布局、研究出交付路径、数据出库（`data-manifest.json`，无 >10 MB 文件）、三语目录、门全过（见 §5）。在 ehs-spatial 里**没做**（4/6/8/9/10/11），也不必再做：交付角色已经在 argus。

## 5. 状态表（2026-10-11 测量）

| 项 | 结果 | 结论 |
|---|---|---|
| ehs-spatial `tests/verdict` | 95 passed | 老实验台仍绿 |
| ehs-spatial 全套（基线 23 个已知失败） | 23 failed / 1608 passed / 235 skipped（632 s） | = 基线，无回归 |
| ehs-spatial 门 | CJK 代码 0；`/Users/`、`/private/tmp` 的 .py 54（交付路径 48）；>10 MB 1；Makefile 1；env 模板 1；deploy 目录 2（`deploy/`、`docker/`） | 未清理 |
| argus 门 @ 87796eb | CJK 代码 0；机器路径 0；>10 MB 0；Makefile 1；`env.template` 1；`CLAUDE.md`（`AGENTS.md` 软链接）；locales 1783/1783/1783 | 全过 |
| argus 测试 @ 87796eb（干净 venv，无 DB） | 688 passed / 194 skipped（82 缺 `PANOPTES_TEST_DATABASE_URL`、15 缺 Mongo、6 Blender、2 S3、1 torch） | 记录的 831/51 需要真实 PostgreSQL + Mongo |
| argus `panoptes run --cell 030 --dry-run` | S2a→S8 全部打印，S7 = `argus.platform.publish_capture` | 链路结构完整；S7 真跑被 provider manifest 挡住 |
| panoptes-verdict-lab 测试 @ c26d07a | 211 passed / 1 skipped | 绿 |
| 自动部件真实 030/090 结果 | `phase5-artifacts/automatic-parts-2026-10-10/{030,090}/results.json` 不存在 | 未执行 |

## 6. MVP 结论（2026-10-11）

- **测量 MVP**：以 Modal 源码交付，**可以**，版本 = dd80ce5（复核过的；当时 747/51），带已知保留项：严格 schema-2 查看器不能直接替换 schema-1 的原报告站；镜像缺 S8 中间产物。87796eb **不能**交付：S7 被付费 provider 挡住、自动量值全部 untrusted、030/090 真实自动运行从未执行。
- **判定层**：不是 MVP，不在产品里；在有区域 / coverage / 声明输入和真实金标之前只是研究。
- **文件管理**：argus 已完成；ehs-spatial 没做也不必做。

## 7. 怎么复测（每行一条命令，离线）

```bash
# ehs-spatial（在 worktree 根）
PANOPTES_FAKE_MODEL=1 PANOPTES_WORKCELL=$PWD .venv/bin/python -m pytest tests/verdict -q -p no:cacheprovider
PANOPTES_FAKE_MODEL=1 PANOPTES_WORKCELL=$PWD .venv/bin/python -m pytest tests -q -p no:cacheprovider
# 门（任一仓库根）
git ls-files | grep -E '\.(py|ts|tsx|js|mjs|sh|ya?ml|json)$' | grep -v -E '^(docs/|research/|web/src/locales/|data/)' | xargs grep -l -P '[\x{4e00}-\x{9fff}\x{3000}-\x{303f}\x{ff00}-\x{ffef}]' | wc -l
git ls-files '*.py' | xargs grep -l -E '/Users/|/private/tmp' | wc -l
git ls-files -z | xargs -0 stat -f '%z %N' | awk '$1>10485760'
# argus（临时克隆，不在交付目录里跑）
git clone /Users/adam/Desktop/panoptes-public/phase5-delivery /private/tmp/status-<date>/argus
PANOPTES_DATA_ROOT=/private/tmp/status-<date>/data PANOPTES_FAKE_MODEL=1 HF_HUB_OFFLINE=1 python -m argus.pipeline.cli run --cell 030 --dry-run
PANOPTES_FAKE_MODEL=1 HF_HUB_OFFLINE=1 EHS_LIVE_BENCHMARK=0 python -m pytest tests -q -rs -p no:cacheprovider
# panoptes-verdict-lab（临时克隆）
PANOPTES_FAKE_MODEL=1 python -m pytest tests -q -p no:cacheprovider
```

## 8. 待用户决定（建议）

1. 宣布 argus 为唯一源头；ehs-spatial `main` 冻结为归档；判定工作只在 panoptes-verdict-lab 继续。
2. 自动部件继续之前先撤 S7 的 Gemini/FAL 硬前置（§2）。
3. 判定层下一步仍是 2026-10-08 定的顺序：L2 区域谓词 → 凭证到位跑 L5 三行 → html 面板标第一轮金标。
