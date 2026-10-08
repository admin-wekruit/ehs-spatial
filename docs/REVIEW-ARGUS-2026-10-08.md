# Argus review of this repository (2026-10-08) and our response

The customer's engineers cloned `ehs-spatial` as `digital-experience/argus`, carried it through their first on-prem runs, and reviewed it
with `/ponytail-review` + `/ponytail-debt` (over-engineering and deferred-shortcut passes). Section 1 is their review, transcribed from
the screenshots they shared (their wording; "we" = the Argus side). Section 2 is our response: decision per finding, size, order.
Section 3 is our own `ponytail:` debt ledger (the other half of what they ran).

## 1. Their review (verbatim as far as the screenshots allow)

> 看完了。直接说结论：交付流程只用到全部代码里大约三成，其余大量研究代码、实验代码和重复的部署文件都堆在同一个仓库里。真正在跑的生产流程，散落在 10 个按日期命名的实验目录里，靠互相改 sys.path 串起来。中文直接写在代码和输出数据里，现在的界面只支持中英两种语言，加荷兰语要逐条改。

### Argus / ehs-spatial review for the next handoff

Baseline: ehs-spatial main `c9da31a` plus our on-prem fixes (`digital-experience/argus` main). 2,270 tracked files, 704 Python files.
The delivery path (`panoptes run`, S8 checks, `serving/`, report site, on-prem runner) reaches about **135 Python files (~53k lines)**.
About **127k Python lines are not on it**.

#### Requirements for the handoff back

1. **English only in code**: identifiers, comments, docstrings, log and exception messages, JSON keys. Chinese and Dutch only in locale catalogs.
2. **Language switch en / zh / nl** in the viewer. Default `en`; fall back to `en` when a key is missing.
3. **One clean layout** (target below). Research and experiments move to an archive branch or a separate repo.
4. **Our on-prem fixes merged upstream**: the 26 commits on `argus` main after `c9da31a`.
5. **Acceptance gates** (bottom of this review) pass on a clean Linux clone.

#### Target layout

```
argus/
  argus/              one Python package (today: ehs_spatial + the delivery parts of scripts/, notes, ...)
    pipeline/         S1–S14 as importable modules, plain functions, one cell config per cell
    checks/           today scripts/workcell_checks
    platform/         publication API (today ehs_spatial/platform)
    providers/        HTTP clients for the GPU services
  services/           one FastAPI service per model (today serving/) + their Dockerfiles
  web/                viewer; web/src/locales/{en,zh,nl}.json
  deploy/             compose per GPU card, jump forwards / systemd, report site
  tests/
  README.md  docs/STATE.md  HANDOFF.md  env.template  Makefile  pyproject.toml
  (not in git)        frozen inputs fetched by sha manifest, like fetch_weights*.py
```

#### Findings

**Structure**

- `research/module-swap-2026-10-07/notes/*-2026-10-0x/`: yagni: the production pipeline lives in 10 dated experiment folders and wires itself with `sys.path.insert` / `spec_from_file_location` chains (`field_values_fill.py:16-17` → `licence-clean-stack-2026-10-06/geometry/compile.py:14` → `geometry-licence-ab-fair-2026-10-05/compile.py:15`). Move the used steps into `argus/pipeline/`; keep the notes as frozen records outside the code path.
- `scripts/`: delete: 309 files, 43 on the delivery path, ~80k lines off it. Move them to the archive.
- `modal_apps/`: delete: 83 files, 15 on the path, ~17.7k lines off it (droid, lingbot, r4/r5, x13, probes, benches). Move them to the archive.
- `ehs_spatial/`: delete: 81 files, 21 on the path, ~18.6k lines off it (Gradio workbench, agent, policy engine, video). Move them out with the workbench.
- `fast_report/`: yagni: the delivery path imports benchmark modules (`x7`, `r5_bench.align_free`) from a 33-file research package. Move the few functions it uses into `argus/pipeline/`.
- `containers/onprem/`, `deploy/`, `docker/`: shrink: three deployment kits. Keep one `deploy/`.
- `Makefile` + `Makefile.handoff` (included by the first): shrink: two Makefiles. Keep one.
- `.env.example` + `env.template`: shrink: two env templates. Keep `env.template`.
- `CLAUDE.md` / `AGENTS.md`: delete: the two files are byte-identical. Keep one.
- Root `app.py`, `.impeccable.md`: delete: workbench launcher and tool config at the repo root.
- `docs/`: delete: 191 markdown files, 129 of them in `docs/archive/`. Move the archive out; keep README, STATE and HANDOFF.

**Modal emulation on the delivery path**

- `scripts/onprem/run_stage.py` (743 lines) + `scripts/onprem/modal_stub/modal.py` (261 lines): yagni: they emulate Modal so `@app.function` bodies run in-process. We patched them 8 times to get one run through (image mounts, `/tmp` scratch, `App.include`, `enable_output`, `__main__` sub-commands, a cross-process lock). Make the steps plain functions or CLIs; GPU work already goes through `providers/` over HTTP. Delete both files after that.
- `completion_ab.py:251,259`, `fair_ab_modal.py:86-100`, `backbone_ab_modal.py:91`: stdlib: fixed `/tmp/w`, `/tmp/runs`, `/tmp/in`, `/tmp/out` assume a fresh container per call and collide when two cells run on one machine. Use `tempfile.mkdtemp()`.
- `backbone_ab_modal.classic`, `fair_ab_modal.infer`, `completion_ab.trellis_meshes`: delete: GPU functions in the delivery modules that `panoptes run` never calls.

**Configuration**

- 90 code files: delete: env fallbacks to `/Users/adam/...` or `/private/tmp/claude-501/...` (for example `completion_ab.py:33-34`). Read the required env and fail fast.
- `ehs_spatial/cli.py:34 CELLS`, `swap_generation.py VARIANTS`, `build_swap_layer.py cmp_dir`, `run_stages.py:23`: shrink: per-cell names are hardcoded four times, and 090 uses unprefixed paths while 030 is prefixed. Three of our fixes were caused by this. Use one config file per cell, read by every step.
- `PANOPTES_SERVING`, `PANOPTES_WORKCELL`, `PANOPTES_PLATFORM`: shrink: all three point at the same checkout. Use one `ROOT`.

**Failure semantics**

- `run_stages.py`: delete: it printed MISSING and exited 0 when all 10 checks failed, which produced a report without boxes, lower edges or G1–G8. We changed it to fail; please keep that.
- `ehs_spatial/pipeline.py:214,291`: delete: warn and continue on failed overlays or viewer build. Fail instead.
- `compare_json.py` (S8, cell 090): delete from the delivery path, or ship its baselines (`completion-ab-090-2026-10-06/results.json`). Today it fails on every clean run.

**Data in git**

- `research/.../data/` (0.5 GB) and `incoming/` (factory photos): native: frozen inputs are committed. `sept/new-view.json` (20.6 MB) breaks the GitHub Enterprise 10 MB limit; we store it gzipped. Move frozen inputs to the blob store or a release asset with a sha manifest; `fetch_weights*.py` already does this for weights.

**Services**

- MoGe exists four times: `serving/moge_service.py`, `modal_apps/moge3_app.py`, inside geometry-mvs, and `providers/moge.py` (MoGe-2 via Replicate). shrink: keep one implementation.
- `serving/{sam3,mapanything,moge}_service.py` (v0): yagni: `panoptes run` never calls them; only the capture tool and the workbench do. Keep them only if the capture tool stays.
- `deploy/Dockerfile.serving`: broken as shipped: the `ubuntu24.04` CUDA tag does not exist, pip asserts in `get_topological_weights`, and the missing `python3-dev` makes every MoGe inference return 500. We fixed all three; please merge.

**Tests** (11 failures on Linux, 2 of them the known version pins)

- `scripts/report_runner/` (`cp -c`) and `store.py:187` (`st_birthtime`): macOS-only code paths. Make them portable or delete them.
- `tests/test_viewer.py`, `tests/test_viewer_surface.py`: they need `node`, which is not declared anywhere. Declare it or skip when it is missing.
- `tests/test_cli.py` (dry run / status): they assume the repo data has never been run. Use a temp copy of the data.
- `tests/test_app.py::test_readme_links_each_provider_credential_source`: fails since the README rewrite. Fix or delete.
- `tests/test_cell_rect.py::test_interior_guard_chain_does_not_bridge_walls`: −1.601 vs −2.0 ± 0.3 on Linux. Fix or loosen with a reason.

**Language / i18n**

- `web/src/i18n.tsx:6`: `type Language = "zh" | "en"`; messages are positional `[zh, en]` tuples spread over 8 files; the default is `zh` (:420-423). Use per-locale catalogs `web/src/locales/{en,zh,nl}.json` keyed by message id, default `en`, falling back to `en`. No new dependency.
- 15 components have inline Chinese: `PhotoReport.tsx` (2,798 CJK chars), `SpatialMeasurements.tsx` (1,369), `LiveReport.tsx` (1,058), `PhotoSemanticExperiment.tsx` (842), `ReportScene.tsx` (623), `spatial-query.ts` (588). Move the strings to the catalogs.
- The pipeline bakes prose into data as paired fields: `labels`/`labelsEn` (`build_swap_layer.py:75-79`); `need`/`needEn`, `highlightReasonsEn`, `snapNoteEn` (`box_faces.py:79,862`); stage panel text in Chinese only (`build_swap_layer.py:87-111`). Emit message codes plus parameters, for example `{"code": "face.need_photos", "photos": [1, 2]}`, and let the viewer translate. Adding Dutch should then be one catalog file with no pipeline change.
- 22 delivery-path Python files contain Chinese (~6k chars): `capture_plan.py`, `compare_json.py`, `box_faces.py`, `build_swap_layer.py`, `assemble_lucida_scene.py`, `obvious_errors.py`, `tables_030.py`, and others. Comments, logs and errors in English; user-facing text through codes.
- `ehs_spatial/static/i18n-catalog.js` (7.4k CJK chars): workbench-only. Delete it with the workbench.

#### Acceptance gates for the handoff back

- Clean Linux clone → `make check-env` and `make test` with **0 failures** (no macOS-only paths, no data-state assumptions).
- `panoptes run --cell 090` and `--cell 030` from a clean clone, on our two A100s over HTTP. The measurement layers have the same fields as the published ones (today 117/117, boxes 9 and 8). No step exits 0 on a failure.
- Outside the locale files there is no CJK in code: `rg -c '[一-鿿]' -g '*.py' -g '*.ts' -g '*.tsx'` must return only `web/src/locales/*`.
- The viewer switches between en / zh / nl, including the box, face and pipeline panel texts.
- No `import modal` and no `/Users/` path on the delivery path; no file over 10 MB in git.
- One Makefile, one env template, one `deploy/` directory, one agent instructions file.

net: about **−128k lines possible** (≈127k Python lines off the delivery path, plus about 1k lines of Modal emulation), before the pipeline itself is consolidated.

(Attached by the reviewer, not yet received here: `argus-audit.py` (+36), `argus-closure.py` (+75).)

## 2. Our response（决定、顺序、大小）

范围（2026-10-08 晚定）：**这个仓库是 Modal 版，是源头**；客户的内部 app 克隆它，只把 Modal 的 GPU 调用换成 A100 经 jumpbox，其余不改。所以审查里关于
Linux / Docker / on-prem 运行器的部分（Linux 上的测试、macOS 专用路径、Dockerfile、Modal 仿真、干净 Linux 克隆上的验收门）是**他们克隆后的适配**，不是我们的工作，
也不由我们验证；我们只做让自己的仓库干净、让『克隆 + 换 GPU 调用』变简单的部分：英文化、三语目录、一个干净布局、去重、研究出交付路径、数据出库。

总体：接受我们范围内的部分。这份审查说的就是 `docs/STATE.md` 和 `HANDOFF.md` 早已承认的事（一个仓库里堆着研究、实验和交付三层）。交付回去的形态按他们的目标布局做，
分六步，先小后大；**算法和参数一行不改**（CLAUDE.md 规则），回归标准就是他们的验收门：117/117 字段、090 九个盒 / 030 八个盒逐字节一致。

| # | 他们的要求 / 发现 | 决定 | 大小 | 何时 |
|---|---|---|---|---|
| 0 | 把审查存进仓库 | 本文件；`docs/MILESTONES.md` 开 Phase 5 | 小 | 已做 2026-10-08 |
| 1 | 两个 Makefile / 两个 env 模板 / CLAUDE.md = AGENTS.md / 根目录 `.impeccable.md` | 合成一个 Makefile；删 `.env.example`（引用改到 `env.template`）；`AGENTS.md` 变成指向 `CLAUDE.md` 的软链接；删 `.impeccable.md`。根目录 `app.py` 随工作台一起走（第 4 步） | 小 | 今天（WP-6） |
| 2 | 失败语义：`run_stages.py` 退出码、`pipeline.py:214,291` 警告继续、`compare_json.py` 每次都失败 | 失败就非零退出；覆盖图 / 查看器构建失败直接 fail；S8 基线随代码发（小）或移出交付路径（大）——按大小定 | 小 | 今天（WP-6） |
| 3 | 固定 `/tmp/w` 等路径；`/Users/adam` 回退（交付路径上） | `tempfile.mkdtemp()`；交付路径上的回退全部改成读 env.template 的键、缺了就报错；归档脚本不动（随第 4 步归档） | 小–中 | 今天（WP-6） |
| 4 | Linux 上的测试失败：macOS 专用路径、`node` 未声明、数据状态假设、README 链接测试、`test_cell_rect` 容差；以及我们 Mac 上 c9da31a 之后的 23 个失败 | **我们的部分**：Mac 上 23 个失败修到 0、`node` 缺失时跳过、`test_app` README 链接测试。**他们的部分**：Linux 可移植性（macOS 专用路径、Linux 上的容差），克隆后自行适配，我们不验证 | 中 | 下一个 session（我们的部分） |
| 5 | 他们在 `argus` main 上 c9da31a 之后的 26 个提交（含 `Dockerfile.serving` 三处修复、`run_stages.py` 退出码） | **他们的**：on-prem 适配留在他们的克隆里，我们不合并也不验证。`run_stages.py` 非零退出我们顺手做了（通用代码质量）；`Dockerfile.serving` 的三处改动按他们的描述照抄了一份，未验证（on-prem 镜像是他们的） | — | 已做（不验证） |
| 6 | 一个干净布局：`argus/` 包（pipeline / checks / platform / providers）、`services/`、`web/`、`deploy/`、`tests/`；研究和实验进归档分支或独立仓库 | 接受目标布局，包名 `argus`；交付路径上的 10 个日期目录里的步骤搬进 `argus/pipeline/`（S1–S14 为可导入函数 + 每工位一个配置文件 = 同时解决"工位名硬编码四次"和"三个 ROOT 变量"）；`research/`、`scripts/`（非交付）、`modal_apps/`、`fast_report/` 研究部分、工作台 / agent / 策略引擎 / 视频进 `archive/research-2026-10` 分支；`docs/archive` 同去。`ehs_spatial` 保留一个发布周期的导入兼容壳 | 大（2–3 天） | 第 4 步之后 |
| 7 | Modal 仿真（`run_stage.py` + `modal_stub`）：步骤改成普通函数 / CLI 后删除；删掉 `panoptes run` 不调用的 GPU 函数 | **不做**：我们的流水线就是 Modal；`scripts/onprem/run_stage.py` + `modal_stub` 是 on-prem 套件，归他们维护或删除。我们第 6 步搬包时保持 Modal 调用，GPU 调用点集中在 `providers/`（他们换这一处） | — | 他们的 |
| 8 | i18n：`en / zh / nl` 目录，默认 `en`；15 个组件的内联中文进目录；流水线输出改消息码 + 参数；22 个交付路径 py 文件去中文 | 接受。顺序：(a) `web/src/locales/{en,zh,nl}.json` + `Language` 三值 + 回退；(b) 组件字符串迁入；(c) 流水线输出改 `{"code", ...}`（查看器翻译；measurement layer 的 `labels/labelsEn` 等成对字段在一个发布周期内并存）；(d) py 文件注释 / 日志 / 异常改英文。荷兰语目录由他们填或先机器翻译后他们校 | 大（2–3 天） | 第 6 步之后（避免搬家时改两遍） |
| 9 | 数据不进 git：0.5 GB 冻结输入 + `incoming/`；单文件 ≤ 10 MB | 冻结输入改为 sha 清单 + `fetch_inputs.py`（和 `fetch_weights*.py` 同一机制），数据放他们的 S3 兼容存储；历史不重写——交付回去的是新仓库（干净历史），本仓库保留为归档 | 中 | 与 6 同期 |
| 10 | 服务：MoGe 四份、v0 服务是否保留、`Dockerfile.serving` 修复 | MoGe 只留 `serving/moge_service.py` 一份（geometry-mvs 内部调用它）；v0 服务（sam3 / mapanything / moge）去留取决于采集工具是否保留——**问他们**；Dockerfile 修复随第 5 步合并 | 中 | 5 之后 |
| 11 | 验收门在干净 Linux 克隆上通过 | **他们的**（Linux 验收是克隆后的事）。我们自己的门：Mac 上 `make test` 0 失败、代码无 CJK（目录除外）、无机器路径回退、无 > 10 MB 文件，放进我们的 CI | 小 | 第 4 步后 |

对审查的两点补充（不是反对）：

- "117/117 字段、盒子 9 / 8 逐字节一致"作为搬家的回归标准：搬家前先把两个工位的 measurement layer 冻结成基线文件进 CI，`panoptes run` 的输出和它逐字段比（`scripts/compare_json.py` 的用途正是这个——所以它的基线要随代码发，而不是删）。
- 判定层实验台（`ehs_spatial/verdict/`，2026-10-08 起）不在交付路径上，和研究一起归档，**除非**他们要 Phase 4 的成果；它已经是独立包、只依赖契约，搬家成本低。

## 3. Our `ponytail:` debt ledger（`/ponytail-debt` 在本仓库的结果，2026-10-08）

命令：`grep -rnE '(#|//) ?ponytail:' --include='*.py' --include='*.ts' --include='*.tsx' --include='*.js' --include='*.sh' .`（跳过 `.venv`、`node_modules`、`runs/`）。

| 区域 | 标记数 | 代表性条目（文件:行，简化了什么。ceiling / upgrade） |
|---|---|---|
| `ehs_spatial/` 交付路径 + 平台 | 28 | `providers/sam3.py:293` 固定 5 次重试线性等待 → 需要时改可配置；`platform/mongo.py:127` 租约锁代替行锁 → LOCK_SECONDS 限定崩溃后的等待；`platform/feedback.py:49` 单写者串行 → 多 worker 时移到共享 SQL；`cli.py:72` CPU-only 流程机无权重镜像；`verdict/lab/runner.py:89` 台账靠 O_APPEND → 多机时换队列 |
| `scripts/`（交付 + 研究） | 60 | `report_runner/store.py:713` 无指纹行仍可重发；`workcell_layer_build.py:459` 资产改名中途失败可能半替换 → 原子改名；`onprem/run_stage.py:367` 只支持字面 REPO/path；`workcell_checks/*` 的采样 / 阈值上限（如 `transfer.py:53` ≤ 2M 射线） |
| `modal_apps/` | 22 | `moge3_app.py:19` 用 PyPI 默认 torch 轮子；`sam3_video.py:179,280` 单段单类不重采样；`splat_train.py:388,646` 训练下采样 / ME340 选帧 |
| `fast_report/` | 8 | `core.py:1977,2242` 失败的模型任务 300 s 后放行；`cascade.py:47` YOLOE 未重训新增类别 |
| `web/` | 15 | `viewer/native-math.ts:132` 点击时线性扫描 → 网格 >~10M 三角形时上 BVH；`api.ts:228` 只去重进行中的查找（签名 URL 会过期）；`CadView.tsx:64` O(n²) 标签碰撞 |
| `research/`、`containers/`、`tests/` | 6 | `containers/onprem/serve_publications.py:59` 反馈助手模型在 on-prem 关闭（SaaS 调用）；`tests/check_platform_policy_geometry.py:16` zen-engine 可能缺席 |

合计 **139 个标记**；逐条看过的交付路径（`ehs_spatial/`、`scripts/onprem`、`scripts/workcell_checks`、`serving/`、`web/src`）里 **没有缺 upgrade 触发条件的（no-trigger = 0）**；研究目录里 3 条只写了上限没写触发（`clearance_b.py:378` 的 3 px 网格、`workcell_gantry.py:567` 的邻近判定、`name_video_entities.py:32` 的固定名单）——随第 6 步归档，不再追。
