# Session handoff — 2026-10-08（晚）：判定层第二轮 + 交付回去（Phase 5）第一天，全部暂停

读的顺序不变：`README.md` → `docs/STATE.md` → `HANDOFF.md` → `research/module-swap-2026-10-07/REPRODUCE-PROMPT.md`。本文只说"现在在哪、今天动了什么、怎么接着做"。
算法和参数一行没改（CLAUDE.md 规则）；已发布的 090 / 030 测量层仍是回归标准。

## 0. 分支与提交（全部已推到 `origin`）

| 分支 | 位置 | 内容 |
|---|---|---|
| `main` = `verdict-lab` | `3116dff` 之后的 docs 提交（本文） | 今天的一切已完成工作；`main` 和 `verdict-lab` 同一提交 |
| `archive/research-2026-10` | `2fd7e0a` | 归档前的快照（审查前的完整仓库），Phase 5 第 6 步删研究目录时从这里找历史 |
| `wip/viewer-i18n-2026-10-08` | 1 个提交 | 查看器三语迁移做到一半（能 build；9 个文件还有中文）。**不要合并**，按 §3 第 8 步接着做 |

本机工作树干净。`runs/`（实验台输出，不入库）里有 `runs/lab-v1`（记分卡 v1 的 12 次运行）和 `runs/wp4`。

## 1. 今天做了什么

### Phase 4 判定层实验台（`ehs_spatial/verdict/`，提交 `616f8e5`）

- 五个决定已定（`docs/research/verdict-layer-plan-2026-10-08.md` §6）：包围规则没有 coverage → CANNOT_DETERMINE（`clingo@2` / `python@2` / `handwritten@2`，基准 v0 计数不变）；L5 第一轮三种合成方案；LLM = Claude Haiku（`ehs_spatial/verdict/llm.py`，官方 SDK，响应缓存是包里唯一的哈希，`PANOPTES_FAKE_MODEL=1` 只回放缓存）；金标在报告里标；规范输入 = "Safety concept tool"式自然语言 + 条款引用。
- 新插件：L4 `llm-extract@0`（文本 → 条款候选 → 词表 / 数字落地 / 引用 / 表 id 核查 → 与手抽文件的 diff；`panoptes verdict extract`）；L5 `function-library@0` / `code-synthesis@0` / `redundant-translation@0`（共享 `l5_rules/asp.py` 渲染；合并图 38 条：12 编译 / 9 需输入 / 17 拒绝）；L7 `html@1`（自包含标注报告：金标 1–6、理由、把握、复测、声明输入、导出 / 导入 `verdict-labels/1`；`panoptes verdict labels merge|declared`）；`relations@2`（光幕 / 扫描仪的 `vertical` / `horizontal`；TS 9.1.7 不再误判竖光幕）。
- EHS safety subagent：`.claude/agents/ehs-safety-spec.md`（接一份新规范文本的流程：存样本 → 抽取 → 读 flags/diff → 提词表 → 只改被标记的条款 → 重跑 → 报告）。
- 记分卡 v1：`docs/research/verdict-lab-scorecard-v1-2026-10-08.md`（六配置 × 两工位）。`tests/verdict` 95 通过。标注 UI 调研：`docs/research/labeling-ui-survey-2026-10-08.md`。
- 对实验台本身的 ponytail-review：`docs/research/verdict-lab-ponytail-review-2026-10-08.md`（可删 −930 行，逐条标了 todo / policy）。

### Phase 5 交付回去（客户审查，提交 `2fd7e0a`、`0073e5b`、`3116dff`）

- 审查原文 + 我们的 11 条决定：`docs/REVIEW-ARGUS-2026-10-08.md`（**这个仓库就是交付给他们的那份**，按意见改好再交；他们的 26 个提交由他们在新版本上重放，我们直接做了点名的修复）。
- 已做（审查表第 1–3、5 行）：一个 `Makefile`（`Makefile.handoff` 并入）、一个 `env.template`（`.env.example` 删）、`AGENTS.md` → `CLAUDE.md` 软链接、`.impeccable.md` 删；`ehs_spatial/pipeline.py` 覆盖图 / 查看器失败直接抛错；`run_stages.py` 有检查缺结果就退出 1；`run_all.sh` `pipefail`；三处固定 `/tmp` 路径改 `mkdtemp`；交付路径上 25 个文件的 `/Users/...` 回退全部改成必需的 env 键；S8 的三个基线文件随代码发（57 KB）；`deploy/Dockerfile.serving` 的三处修复（22.04 CUDA 标签、`python3-dev`、先升级 pip）。
- `CLAUDE.md` 加了交付回去的硬规则（代码只用英文、不留机器路径回退、失败非零退出、测试在干净 Linux 上过）。

## 2. 判定层：接着做什么（优先级顺序）

1. **让 TS 条款真的绑定到对象**：记分卡 v1 里 function-library 的 9 条编译规则在 090 / 030 上一条都没绑定，因为 L2 不算 `perimeter_of(F, Z)` 和 `covers_opening(L, C)`。做法：报告标注面板的 `restricted_space`（多边形文本）→ `panoptes verdict labels declared` → `scene-json` 的 `declared:` 把它变成 `Scene.zones`（kind restricted_space / hazard_zone）→ `relations@3` 算：围栏脚印到区域边界 ≤ 0.5 m = `perimeter_of`；光幕脚印跨在区域开口上 = `covers_opening`。合成测试已证明加上这两个谓词后 4.4 / Table2-min-height / 9.1.2 与手写包逐对象一致。
2. **第一次真跑 Haiku**（本机没有凭证；需要 `.env` 里 `ANTHROPIC_API_KEY=` 或 `brew install anthropics/tap/ant && ant auth login`）：
   ```
   .venv/bin/python -m ehs_spatial.cli verdict extract --spec-text ehs_spatial/verdict/spec/samples/manual-loading-station-concept-2026-10-08.md --out runs/llm-extract/ts/clauses.json
   PANOPTES_WORKCELL=$PWD .venv/bin/python -m ehs_spatial.cli verdict matrix --benchmark ehs_spatial/verdict/benchmark/v0 --configs ehs_spatial/verdict/configs/{llm-extract-ts,codegen-ts,redundant-ts}.yaml --runs-dir runs/lab-v2 --jobs 1 --run-id lab-v2
   ```
   预计 11 / 24 / 48 次调用，之后全部命中 `runs/llm-cache`。结果进记分卡 v2，和手抽条款、function-library 行对比（编译率、拒绝原因、diff）。
3. **金标**：用 `runs/lab-v1/lab-v1-html-report-090/L7/verdicts.html`（或重跑 `html-report.yaml`）标一轮，导出 → `labels merge` → `benchmark/v1/gold.json`（v0 冻结，金标仍是试跑的 provisional）。
4. 实验台瘦身：按 `verdict-lab-ponytail-review-2026-10-08.md` 的 todo 项做（机械项约 −480 行）；两引擎并存不动（等价测试是实验台的设计）。
5. 判定层不在交付路径上：Phase 5 第 6 步归档时随研究走，除非客户要 Phase 4。

## 3. 交付回去：剩下的步骤（`docs/REVIEW-ARGUS-2026-10-08.md` §2 的表；这里只写怎么接）

| 步 | 状态 | 怎么接 |
|---|---|---|
| 4 测试 0 失败 + Linux 可移植 | **未开始**（agent 被停在读代码阶段，没有改动） | 23 个失败 id 在 `CHANGELOG.md` "Test baseline note"；逐个找根因（c9da31a 改了 `box_faces.py` / `lower_edge.py` / `workcell_photo_oneshot.py` / README）；macOS 专用：`scripts/report_runner/store.py:187 st_birthtime`、`publish.py` `cp -c`；`tests/test_viewer*.py` 缺 `node` 时跳过；`tests/test_cli.py` 用数据临时副本；`test_app.py` README 链接测试；`test_cell_rect.py` 容差找原因。另：`scripts/report_runner/publish.py:30`、`adopt.py:36` 的机器路径回退 |
| 6 一个包 + 归档 | **未开始**；清单 agent 被停，没产出 | 先出 `docs/DELIVERY-PATH-<date>.md`：从 `ehs_spatial/cli.py`（S2a–S8 的子进程脚本列表在 `research/module-swap-2026-10-07/run_all.sh`）+ `run_stages.py` 调的检查 + `serving/` + `deploy/` + `ehs_spatial/platform|providers` + `containers/onprem` 做 import 闭包；然后 `argus/pipeline|checks|platform|providers`，每工位一个配置文件（代替 `cli.py CELLS` / `swap_generation.py VARIANTS` / `build_swap_layer.py cmp_dir` / `run_stages.py:23` 四处硬编码和 090 无前缀 / 030 有前缀）；一个 `ROOT`；其余进 `archive/research-2026-10`。回归：搬完后 `panoptes run --cell 090/030` 的测量层和已发布的逐字段一致（`compare_json.py` 的 090 路径今天还按无前缀布局读，见 WP-6 的说明） |
| 7 去 Modal 仿真 | 未开始 | 随第 6 步：步骤进包时去掉 `@app.function`，GPU 只走 `providers/` HTTP，然后删 `scripts/onprem/run_stage.py` + `modal_stub` |
| 8 i18n | **做到一半**，在 `wip/viewer-i18n-2026-10-08` | 已有：`web/src/locales/{en,zh}.json` 945 键、`nl.json` 空（回退 en）、`i18n.tsx` 的 `t()` / `useLanguage()`、5 个 `[zh, en]` 元组文件已迁。还差：9 个文件的内联中文（LiveReport、VisitsPanel、PhotoReport、PhotoSemanticExperiment、ReportScene、spatial-query 等）、数据侧 `labels/labelsEn` 之类成对字段的单一选择函数、目录 / CJK 扫描测试 `tests/test_viewer_i18n.py`、三选一语言开关。之后：流水线输出改消息码 + 参数（`build_swap_layer.py:75-111`、`box_faces.py:79,862`），22 个 py 文件去中文，删 `ehs_spatial/static/i18n-catalog.js`（随工作台） |
| 9 数据出库 | 未开始 | 冻结输入（0.5 GB `data/`、`incoming/`）改 sha 清单 + `fetch_inputs.py`（同 `fetch_weights*.py`），放客户 S3 兼容存储；交付回去用新仓库（干净历史），本仓库归档 |
| 10 服务 | 未开始 | MoGe 只留 `serving/moge_service.py`；v0 服务去留问客户（采集工具是否保留） |
| 11 CI 验收门 | 未开始 | ubuntu 任务：`make check-env` + `make test` + CJK / `/Users/` / `import modal` / 10 MB 扫描 |

## 4. 需要用户的

- Claude 凭证（§2 第 2 条）；荷兰语目录由客户填还是先机器翻译后校；v0 服务是否保留；客户在干净 Linux + A100 上跑一次验收门。

## 5. 常用命令

```
PANOPTES_FAKE_MODEL=1 PANOPTES_WORKCELL=$PWD .venv/bin/python -m pytest tests/verdict -q            # 95 passed
PANOPTES_FAKE_MODEL=1 PANOPTES_WORKCELL=$PWD .venv/bin/python -m pytest tests -q --ignore=tests/verdict   # 23 known failures (step 4)
PANOPTES_FAKE_MODEL=1 PANOPTES_WORKCELL=$PWD .venv/bin/python -m ehs_spatial.cli verdict matrix --benchmark ehs_spatial/verdict/benchmark/v0 --configs ehs_spatial/verdict/configs/{baseline,python-engine,stpl-k1,clause-kg-ts,funclib-ts,html-report}.yaml --runs-dir runs/lab-v1 --jobs 2 --run-id lab-v1
cd web && npm run build
```
