# Session handoff — 2026-10-08（晚）：给一个零上下文的 session

目的：一个没有我们任何 idea 的 session 读完本文，知道（1）这个项目是什么、现在交付了什么；（2）我们做了哪些研究、每份研究的结论；（3）正在建的判定层架构和实验台；（4）客户审查后文件 / 代码结构怎么清理；（5）UI 往哪里提升；（6）每样东西在哪个分支、哪个文件；（7）怎么接着做。
全部工作 2026-10-08 晚**暂停**，工作树干净，一切已推到 `origin`。

读的顺序（先读这四份，再读本文余下的部分）：`README.md` → `docs/STATE.md`（系统今天是什么）→ `HANDOFF.md`（怎么跑）→ `research/module-swap-2026-10-07/REPRODUCE-PROMPT.md`（怎么复现发布的数字）。硬规则在 `CLAUDE.md`（几何是 geometry-mvs 不是 Pi3X；补全是 SAM 3D 不是 RecGen；`.env` 唯一配置；不动算法参数；不动已发布的 `nativeToMeters`；不读 `.platform/`；不部署）。

---

## 1. 项目一句话 + 现状

Panoptes：工位（机器人单元）照片 → 米制 3D 重建（RoMa + DA3-BASE + numpy BA + MoGe-3 补洞 = **geometry-mvs**；SAM 3 掩码；SAM 3D Objects 补全；急停参照物定尺度）→ 测量层（盒、面、离地缝、距离）→ 报告站。
已交付（Phase 3，`1.0.0-rc1`，2026-10-07）：客户一个 `git clone` → `.env` → `make up`（两张 A100 上的 GPU 服务）→ `panoptes run --cell 090 | 030`，报告站。发布的两个工位：090（`4b58dbd2…`）、030（`cd84d3fb…`），这两个测量层是所有改动的回归标准（117/117 字段、盒子 9 / 8）。
报告里今天**没有机器合规判定**（策略引擎按设计弃权）——判定层就是 Phase 4 要补的东西。

---

## 2. 研究：问了什么、结论是什么、在哪（全在 `docs/research/`，索引 `docs/research/README.md`）

| 问题（用户当时问的） | 结论 | 文档 |
|---|---|---|
| 我们现在的判定是怎么出的？有 harness 吗？规则语言用什么（OWL / SWRL / SPARQL / STL / Datalog…）？哪些安全规范数字能从照片判？ | 审计：三套扁平 JSON、7 个 2.5D 谓词、判定确定性、LLM 只翻译、090/030 报告引擎按设计弃权、harness 窄。调研后建议：显式关系层的类型化场景图（不是 RDF 优先）+ **clingo 作唯一判定内核** + 不确定度进判定（ILAC-G8 护带）+ 先解决让引擎弃权的四件事（尺度状态、整高 σ、危险区、适用性）。附录 A / B 是调研原文和每个数字的出处链接 | `verdict-layer-rules-2026-10-07.md`（+ `survey-A`、`survey-B`）|
| 别的领域（BIM 合规、法律、临床、自动驾驶、功能安全、计量…）怎么做"表示 → 规则 → 判定"？场景图真的帮助理解吗？预训练 / 后训练能不能替代？ | 九个领域的共同模式：封闭词表、显式"未知"状态、规则对齐条款并带测试、符号执行器 + ML 感知但 ML 永不当裁判、证据链、版本化规则包、带不确定度的阈值、人工适用性签字。场景图是**接口和证据底座**，不是"理解"；我们的边是算出来的米制量，所以它是"带不确定度的可查询测量台账"。训练路线：不替代，判定不经权重。给我们的 12 条教训（计量式判定、缺失是值、规则包是产品、发布编译率、声明输入保持声明、感知层单独规约、适用性签字、证据链、FAIL 精确率优先、图保持薄、未知-不安全 backlog、形式化会发现规范 bug） | `verdict-patterns-lessons-2026-10-07.md`（+ `survey-C`、`survey-D`）|
| 别人做过的类似系统：做了什么、效果、失败、怎么学、链接 | 30 个系统分五组（规范 → 结构化场景 → 判定；场景图 + LLM；带验证的自动形式化；检索；已部署系统）。学习顺序：验证循环先于一切（GuardEn、ARc）→ 对齐单独做（VLTL-Bench、GinSign）→ 关系算出来暴露成 API（RieMind、FuncMapper、TUM）→ 条款图带表格（ACCORD、BifrostRAG）→ 先检索再编译（DriveReg、Lawful-AD）→ 片段级审阅 + 置信度分流（nl2spec、Han）→ 判定只认引擎 | `verdict-prior-work-links-2026-10-08.md` |
| 这些系统的仓库、许可证、能不能商用、怎么和我们的组件结合 | 许可证干净能直接用：Spark-DSG、clingo、AEC3PO schema、CODE-ACCORD + accord-nlp、TUM 函数生成循环、nl2spec、GinSign、Logic-LM 模式、RTAMT / Reelay / MoonLight、pySHACL / Nemo / ZEN。效果最好的（GuardEn、RieMind、SGR-BIM、P4IR）没代码，只抄方法和消融数字。避开 NC（CompAgent、ConstructionSite-10K）和 GPL（heracles、SceneFlowLang） | `verdict-prior-work-repos-2026-10-08.md`（+ `survey-F`）|
| 规范是自然语言，怎么和规则匹配？（用户否决了"逐句整理进固定模板"） | 推荐七级：① 文档 → 结构化条款图（表变成带类型的查表函数、定义变别名、例外变 defeater）；② 概念对齐表（工程师审表不审条文）；③ 按工位检索适用条款；④ 对有类型的场景 API 合成检查（LLM 写短检查 + 子翻译）；⑤ 人看之前自动验证（冗余翻译差分执行、性质测试、签名验证、蜕变测试）；⑥ 字面化审阅屏（置信度分流）；⑦ 确定性执行 + 证据。度量：编译率 > 80 %、对齐覆盖率、每条审阅分钟、与工程师一致率、误拒率、检索召回、冗余不一致率。丢：逐句模板、词表缺口即拒绝、预先编译全部 | `verdict-spec-to-check-2026-10-08.md`（+ `survey-E`）|
| 怎么评测？靠 LLM 吗？视频太多怎么办？怎么泛化到很多工位和很多规范？ | 冻结基准（工位, 照片集, 规范包）+ 固定输出 schema + 金标（人工五态 + 现场测量 + 适用性，双人标注 α ≥ 0.8）+ 感知缓存；每个版本整套跑一次出记分卡；按层记分（编译率 / 对齐 / 检索 / 感知 σ 校准 / 5×5 判定混淆矩阵 / 解释）；泛化靠留出工位、留出规范；验证别人的结论靠消融不靠复现论文；视频 = 关键帧 + 同一判定层 | `verdict-evaluation-protocol-2026-10-08.md` |
| 验证层怎么分层、怎么和重建解耦？2026 的 4D 工作（DAAAM、WorldSGG、Hydra、Khronos、ChronoGraph）放哪？ | 七层四契约（见 §3）；4D 系统只是 C1 Scene 的生产者；时间轴靠 `t` 字段进来不加层；每层失败隔离 | `verdict-layer-architecture-2026-10-08.md`（+ `.png`）|
| 像科研机构一样比较多种方案；可插拔、可并行、报告里看得出谁产的；少用哈希、不要过度防御 | 实验台：每层一个插件注册表 + YAML 配置一次只换一层 + 按 run id 复用 + 台账 + 记分卡（每行插件签名、与上一行的"差在哪一层"）；五个阶段 A 契约与实验台 → B 让引擎能判 → C 规范侧多方案 → D 金标 + 消融 → E 泛化与视频。§6 的五个决定已定（见 §3） | `verdict-layer-plan-2026-10-08.md` |
| 实验台的结果 | 记分卡 v0（基线 / python 引擎等价 / stpl + k=1）、v1（引擎 v2、clause-kg、function-library、html 报告，六配置 × 两工位） | `verdict-lab-scorecard-v0-2026-10-08.md`、`verdict-lab-scorecard-v1-2026-10-08.md` |
| 做得好的数据标注 UI 长什么样？我们的报告里怎么加人工输入？ | 14 个工具（Label Studio、Argilla、Prodigy、doccano、CVAT、Labelbox、Encord、Scale、Roboflow、LangSmith、Braintrust、Potato、VIA、UDT），抄 8 个模式：一次一条上下文齐全、键盘优先数字选状态、队列 + 进度 + 过滤、标签 + 理由 + 置信度、跳过 / 标记是状态不是删除、标注 vs 审核并排、多人盲审 + 冲突外露、文件就是接口单文件离线 | `labeling-ui-survey-2026-10-08.md` |
| 实验台本身过度工程了吗 | ponytail-review：可删 −930 行（机械 −480），逐条 todo / policy；两引擎并存保留（等价测试是设计） | `verdict-lab-ponytail-review-2026-10-08.md` |
| 客户对整个仓库的审查 | 见 §5 | `docs/REVIEW-ARGUS-2026-10-08.md` |

试跑代码（2026-10-07，clingo 规则包 v0 对 090 / 030）：`research/verdict-layer-trial-2026-10-07/`（README 说明哪些已被取代）。

---

## 3. 架构与实验台（正在建的东西，代码在 `ehs_spatial/verdict/`）

**七层五契约**（`verdict-layer-architecture-2026-10-08.md`；代码 `contracts.py`）：
L1 感知 / 重建 → **C1 Scene**（米制盒、σ、置信度、可见视角、区域、coverage、声明输入）→ L2 薄场景图 = 算出来的米制关系 → **C2 Facts**（谓词 + 值 + U + 平面占用网格）→ L3 感知规约（跨视角持久、尺寸一致、地面接触 → `untrusted`）→ L4 规范侧（条款图 + 对齐 + 检索，对 **C3 Signature** = 封闭词表）→ L5 检查合成 + 验证 → **C4 RulePack**（每条规则：引擎中立的 `spec` + ASP + 状态 compiled / needs_input / vocabulary_gap / refused）→ L6 引擎 → **C5 Verdict**（五态 PASS / FAIL / NEEDS_MEASUREMENT / NEEDS_INPUT / CANNOT_DETERMINE，护带 k=2，证据 + 来源）→ L7 报告。

**实验台规则**（`ehs_spatial/verdict/README.md`）：插件只 import 契约；`@register(layer, name, version)`，`name@version` 进每条判定的来源和记分卡；确定性（跑两次字节相同）；缺失是值；按 run id 复用；先等价再归因；毫米；每个插件目录有 README；不过度防御。用法 `ehs_spatial/verdict/lab/README.md`：`panoptes verdict run | matrix | scorecard | plugins | extract | labels`。

**今天有的插件**：L1 `scene-json@1`（+ 未接线的 coverage / sigma 工具）· L2 `relations@2` · L3 `none@1` / `stpl@1` · L4 `none@1` / `clause-kg@0`（手抽条款：ISO 20 条 + TS-0011963 Rev 10 18 条）/ `llm-extract@0`（Claude 从文本抽）· L5 `handwritten@2`（对照组）/ `function-library@0` / `code-synthesis@0` / `redundant-translation@0` · L6 `clingo@2` / `python@2`（等价）· L7 `markdown@1` / `html@1`（带标注面板）。基准 `benchmark/v0`（090 / 030 的试跑场景，金标 provisional）。

**已定的五个决定**（`verdict-layer-plan` §6）：包围规则无 coverage → CANNOT_DETERMINE；L5 第一轮三方案；LLM = Claude Haiku（`llm.py`，响应缓存是包里唯一哈希，`PANOPTES_FAKE_MODEL=1` 只回放）；金标在报告里标（`html@1` → `labels merge` → 新版本 `gold.json`）；规范输入 = "Safety concept tool"式自然语言 + 条款引用（样本 `ehs_spatial/verdict/spec/samples/`）。

**EHS safety subagent**：`.claude/agents/ehs-safety-spec.md`——接一份新规范文本：存样本 → `panoptes verdict extract` → 读 flags / diff → 提词表增补 → 只改被标记条款 → 重跑 → 报告。

**记分卡 v1 说了什么**：基线等五组数字与 v0 一致；`function-library@0` 在合并图 38 条上 12 编译 / 9 需输入 / 17 拒绝，但编译的 9 条在 090 / 030 上**一条没绑定**（L2 不算 `perimeter_of` / `covers_opening`）；22 / 18 条 NEEDS_INPUT 缺的正是标注面板要收的声明输入；LLM 三行没跑（本机无凭证）。

**判定层下一步（按序）**：① L2 从声明的限制空间算 `perimeter_of` / `covers_opening`（`labels declared` → `scene-json declared:` → `Scene.zones` → `relations@3`）；② 第一次真跑 Haiku（命令见 §8）→ 记分卡 v2；③ 用 html 报告标一轮金标 → `benchmark/v1`；④ 按 ponytail-review 的 todo 瘦身；⑤ 判定层不在交付路径上，Phase 5 归档时随研究走，除非客户要 Phase 4。

---

## 4. 分支与提交（都在 `origin`）

| 分支 | 提交 | 内容 |
|---|---|---|
| `main` = `verdict-lab` | 本文所在提交（`3116dff` 之后） | 今天完成的一切：判定层第二轮（`616f8e5`）、审查 + 响应（`2fd7e0a`、`0073e5b`）、清理第一天（`3116dff`）|
| `archive/research-2026-10` | `2fd7e0a` | 清理前的完整快照；Phase 5 第 6 步删研究目录后到这里找历史 |
| `wip/viewer-i18n-2026-10-08` | `3c9b1fc` | 查看器三语迁移做到一半（能 build；9 个文件还有中文）。**不要合并**，按 §6 接着做 |

不入库：`runs/`（实验台输出；`runs/lab-v1` = 记分卡 v1 的 12 次运行、`runs/wp4` = html 报告样例）。Claude Code 记忆（跨 session）：`~/.claude/projects/-Users-adam-Desktop-Tesla-ehs-spatial/memory/`（`verdict-lab-decisions.md`、`phase5-handoff-back.md`）。

---

## 5. 文件 / 代码结构清理（Phase 5，客户审查）

客户（仓库 `digital-experience/argus` = 我们交付的这份）用 `/ponytail-review` + `/ponytail-debt` 审了：交付路径只占全部代码三成（~135 个 py / 53k 行 在路径上，127k 行不在），生产流水线散在 10 个日期目录里靠 `sys.path` 串联，中文写在代码和输出数据里。**要求**：代码只用英文（中 / 荷兰语只在 `web/src/locales/*.json`）；查看器 en / zh / nl 切换默认 en；一个干净布局（`argus/` 包：pipeline / checks / platform / providers；`services/`、`web/`、`deploy/`、`tests/`）；研究进归档分支或独立仓库；验收门在干净 Linux 克隆上过（`make test` 0 失败、`panoptes run` 两工位结果字段一致、代码无 CJK、交付路径无 `import modal` / `/Users/`、无 > 10 MB 文件、一个 Makefile / env 模板 / deploy / agent 文件）。原文和我们的 11 条决定：`docs/REVIEW-ARGUS-2026-10-08.md`。

| 步 | 状态 | 怎么接 |
|---|---|---|
| 1 去重 | **已做**（`3116dff`） | 一个 `Makefile`（`Makefile.handoff` 并入）、`.env.example` 删、`AGENTS.md` → `CLAUDE.md` 软链接、`.impeccable.md` 删 |
| 2 失败语义 | **已做** | `ehs_spatial/pipeline.py` 失败抛错；`run_stages.py` 缺结果退出 1；`run_all.sh` `pipefail`；S8 的 `cmp-*/results.json` 基线随码发 |
| 3 机器路径 / 固定 `/tmp` | **已做**（交付路径） | 25 个文件改成必需 env 键；`mkdtemp`；研究目录的回退随第 6 步归档 |
| 5 他们点名的修复 | **已做** | `deploy/Dockerfile.serving`：22.04 CUDA 标签、`python3-dev`、先升级 pip |
| 4 测试 0 失败 + Linux 可移植 | 未开始 | 23 个失败 id 在 `CHANGELOG.md` "Test baseline note"（c9da31a 之后出现；它改了 `box_faces.py` / `lower_edge.py` / `workcell_photo_oneshot.py` / README）；macOS 专用 `store.py:187 st_birthtime`、`publish.py` `cp -c`；`tests/test_viewer*.py` 缺 `node` 时跳过；`tests/test_cli.py` 用数据临时副本；`test_app.py` README 链接测试；`test_cell_rect.py` 容差先找原因；`scripts/report_runner/publish.py:30`、`adopt.py:36` 的机器路径 |
| 6 一个包 + 归档 | 未开始 | 先出交付路径清单（import 闭包：`ehs_spatial/cli.py` 的子进程脚本列表在 `research/module-swap-2026-10-07/run_all.sh`；`run_stages.py` 调的检查；`serving/`、`deploy/`、`ehs_spatial/platform|providers`、`containers/onprem`），写成 `docs/DELIVERY-PATH-<date>.md`；然后 `argus/pipeline|checks|platform|providers`，每工位一个配置文件（代替 `cli.py CELLS` / `swap_generation.py VARIANTS` / `build_swap_layer.py cmp_dir` / `run_stages.py:23` 四处硬编码和 090 无前缀 / 030 有前缀）、一个 `ROOT`；其余删除（历史在 `archive/research-2026-10`）；`ehs_spatial` 留一个发布周期的导入兼容壳。回归：`panoptes run --cell 090/030` 的测量层和已发布的逐字段一致 |
| 7 去 Modal 仿真 | 未开始 | 随第 6 步：步骤进包时去掉 `@app.function`，GPU 只走 `providers/` HTTP，然后删 `scripts/onprem/run_stage.py` + `modal_stub/` |
| 8 i18n | **做到一半**（`wip/viewer-i18n-2026-10-08`） | 见 §6 |
| 9 数据出库 | 未开始 | 冻结输入（0.5 GB `data/`、`incoming/`）改 sha 清单 + `fetch_inputs.py`（同 `fetch_weights*.py`），放客户 S3 兼容存储；交付回去用新仓库（干净历史） |
| 10 服务 | 未开始 | MoGe 四份只留 `serving/moge_service.py`；v0 服务（sam3 / mapanything / moge）去留问客户（采集工具是否保留） |
| 11 CI 验收门 | 未开始 | ubuntu 任务：`make check-env` + `make test` + CJK / `/Users/` / `import modal` / 10 MB 扫描 |

`CLAUDE.md` 已加硬规则：代码只用英文、不留机器路径回退、失败非零退出、测试在干净 Linux 上过；**算法参数照旧不改**。

---

## 6. UI 提升

**（a）判定报告里的人工输入**（已做，`ehs_spatial/verdict/layers/l7_report/html.py`，`html@1`，配置 `configs/html-report.yaml`）：一个自包含 `verdicts.html`（无 CDN、无服务器、Pages 可放）：来源头 → 队列 + 焦点面板（规则@版本、对象 + 类别 + 尺寸、测量 ± u、阈值、裕度、缺的输入、证据事实、视角）→ 金标按钮 `1–6`（五态 + NOT_APPLICABLE）、理由、`u` 不确定、`r` 复测、`n/p` 上下、`Enter` 存并下一条；进度条、过滤；每工位的声明输入表单（stop_time_ms、resolution_mm、risk_level、restricted_space、reach_radius_mm、body_part、payload_kg、zone_depth_mm）和审核人；导出 / 导入 `verdict-labels/1` JSON；`panoptes verdict labels merge`（一致进 verdicts、分歧进 conflicts，写新版本 `gold.json`，基准目录冻结）、`labels declared`（→ `scene-json declared:`）。设计依据：`labeling-ui-survey-2026-10-08.md` 的 8 个模式。样例：`runs/lab-v1/lab-v1-html-report-090/L7/verdicts.html`。

**（b）报告站查看器三语**（客户要求；做到一半在 `wip/viewer-i18n-2026-10-08`）：已有 `web/src/locales/{en,zh}.json`（945 键）、`nl.json`（空 = 回退 en）、`i18n.tsx` 的 `t()` / `useLanguage()`（默认 en、localStorage 记住）、5 个 `[zh, en]` 元组文件已迁；build 通过。还差：9 个文件的内联中文（LiveReport、VisitsPanel、PhotoReport、PhotoSemanticExperiment、ReportScene、spatial-query 等）；数据侧 `labels/labelsEn`、`need/needEn`、`highlightReasonsEn`、`snapNoteEn` 的单一选择函数；三选一语言开关；`tests/test_viewer_i18n.py`（目录键一致 + CJK 扫描）。之后流水线输出改消息码 + 参数（`build_swap_layer.py:75-111`、`box_faces.py:79,862`），22 个 py 文件去中文，删 `ehs_spatial/static/i18n-catalog.js`（随工作台）。荷兰语目录由客户填或先机器翻译后校。

**（c）客户对 UI 的要求**：查看器 en / zh / nl 切换包括盒、面、流水线面板文本；无其他 UI 要求。

---

## 7. 需要用户的

Claude 凭证（`.env` 的 `ANTHROPIC_API_KEY=` 或 `brew install anthropics/tap/ant && ant auth login`）；荷兰语目录谁填；v0 服务是否保留；客户在干净 Linux + A100 上跑一次验收门；是否要 Phase 4 成果进交付（否则判定层随研究归档）。

---

## 8. 常用命令

```
PANOPTES_FAKE_MODEL=1 PANOPTES_WORKCELL=$PWD .venv/bin/python -m pytest tests/verdict -q                 # 95 passed
PANOPTES_FAKE_MODEL=1 PANOPTES_WORKCELL=$PWD .venv/bin/python -m pytest tests -q --ignore=tests/verdict  # 23 known failures (Phase 5 step 4)
PANOPTES_FAKE_MODEL=1 PANOPTES_WORKCELL=$PWD .venv/bin/python -m ehs_spatial.cli verdict matrix --benchmark ehs_spatial/verdict/benchmark/v0 --configs ehs_spatial/verdict/configs/{baseline,python-engine,stpl-k1,clause-kg-ts,funclib-ts,html-report}.yaml --runs-dir runs/lab-v1 --jobs 2 --run-id lab-v1
# first live Haiku runs (needs credentials; ~11 / 24 / 48 calls, then cached under runs/llm-cache):
.venv/bin/python -m ehs_spatial.cli verdict extract --spec-text ehs_spatial/verdict/spec/samples/manual-loading-station-concept-2026-10-08.md --out runs/llm-extract/ts/clauses.json
PANOPTES_WORKCELL=$PWD .venv/bin/python -m ehs_spatial.cli verdict matrix --benchmark ehs_spatial/verdict/benchmark/v0 --configs ehs_spatial/verdict/configs/{llm-extract-ts,codegen-ts,redundant-ts}.yaml --runs-dir runs/lab-v2 --jobs 1 --run-id lab-v2
cd web && npm run build                                                                                   # viewer
```
