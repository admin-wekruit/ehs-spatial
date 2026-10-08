# 数据标注 / 审核 UI 调研（2026-10-08）：判定报告里的人工输入该长什么样

背景：我们的报告是 Pages / nginx 上的静态文件，没有服务器、没有 SaaS、Mac 内存有限；每次让人把文件发来发去不可行。要收的人工输入：每条判定行的
gold 状态（PASS / FAIL / NEEDS_MEASUREMENT / NEEDS_INPUT / CANNOT_DETERMINE / NOT_APPLICABLE）+ 一句理由 + 置信度（sure / unsure）+ 重测标记；每个场景的
declared inputs（stop_time_ms、resolution_mm、risk_level、restricted_space、reach_radius_mm、body_part、payload_kg、zone_depth_mm）+ 审核人。
这些进 `benchmark/<v>/gold.json` 和 `Scene.declared_inputs`。本文看了 14 个工具，只记"做得好、我们能抄"的部分；每条带可点的链接；没有核到原文的标
**未核实**。落地：`ehs_spatial/verdict/layers/l7_report/html.py`（`html@1`，一个自包含的 verdicts.html）和 `lab/labels.py`（`labels merge | declared`）。

## 一、逐个看

| 产品 | 链接 | 做得好的地方 | 对我们的启发 | 许可证 / 是否开源 |
|---|---|---|---|---|
| **Label Studio** | [labeling 流程](https://labelstud.io/guide/labeling) · [skip](https://labelstud.io/guide/skip) · [默认键位 keymap.json](https://github.com/HumanSignal/label-studio/blob/develop/web/libs/editor/src/core/settings/keymap.json) · [Choice.hotkey](https://labelstud.io/tags/choice) · [导出格式](https://labelstud.io/guide/export) | "Label All Tasks" 进入标注流，Submit 即跳下一条；Skip 只在流里有，跳过的任务落到 Data Manager 的 Cancelled 列，别人会接着看到；默认键位 `ctrl+enter` 提交、`ctrl+space` 跳过、`ctrl+z` 撤销；每个 `<Choice>` 可配 `hotkey`；导出 JSON 每条标注带 `completed_by`、`result`、`was_cancelled`、`ground_truth`、`lead_time` | 流式一条接一条 + 跳过不等于删除；选项直接绑数字键；导出 schema 里要有"谁标的、是不是跳过" | Apache-2.0（[GitHub](https://github.com/HumanSignal/label-studio)）；审核 / 一致性是企业版 |
| **Label Studio Enterprise（审核 / 一致性）** | [agreement](https://docs.humansignal.com/guide/stats) · [review](https://docs.humansignal.com/guide/quality) | 一致性按控件分别算，Consensus（多少人选了众数）或 Pairwise（两两平均）；分类题用 Exact Match；审核动作三个：Accept / Fix & Accept / Reject（审核页内容来自搜索摘要，**未逐字核实**） | 我们的"一致性"只需要 exact match：同键不同状态 = 冲突；审核页就是机器判定和人工 gold 并排 | 商用，闭源 |
| **Argilla** | [annotate](https://docs.argilla.io/latest/how_to_guides/annotate/) · [distribution](https://docs.argilla.io/latest/how_to_guides/distribution/) | Focus view"一条接一条线性标注"vs Bulk view 列表快扫；状态四个：Pending / Draft / Submitted / Discarded；`Enter` 提交、`Backspace` 丢弃、`Ctrl+S` 存草稿、`1 2 3` 选标签、←/→ 翻页；进度条 + 自己的四种计数；可组合的过滤器；可展开的标注指南；`min_submitted` 决定一条记录几个人提交才算完成 | 默认 focus 视图 + 队列列表并存；Enter = 提交并下一条；进度要显示"已标 / 总数" | Apache-2.0（[GitHub](https://github.com/argilla-io/argilla)） |
| **Prodigy** | [web app 键位](https://prodi.gy/docs/api-web-app) · [docs](https://prodi.gy/docs) | `a` 接受、`x` 拒绝、`space` 忽略、`backspace` 撤销、`0–9` 选项（1–9 → 第 1–9 项，0 → 第 10 项）、`f` 标记、`h` 帮助；侧栏保留最近 10 条决定且可改；进度小部件 | 一只手的键位：数字选状态、单字母翻页 / 标记；最近决定要能回头改（我们：队列里点回去） | 付费商用，"不是免费也不是开源" |
| **doccano** | [tutorial](https://doccano.github.io/doccano/tutorial/) | 建标签时就填 shortcut key、前景色、背景色 | 快捷键是标签定义的一部分，不是事后加的 | MIT（[GitHub](https://github.com/doccano/doccano)） |
| **CVAT** | [Manual QA & Review](https://docs.cvat.ai/docs/qa-analytics/manual-qa/) · [GitHub](https://github.com/cvat-ai/cvat) | 任务 Stage 设成 Validation + 指定审核人即进审核；Review mode 只露出 Issue 工具、藏掉其它；右键"Quick issue: incorrect position / incorrect attribute"一键开常见问题；修正阶段逐个 Resolve；README 列了 consensus、Ground Truth、Honeypot | 审核模式 = 少而固定的动作；"常见理由一键填"值得抄（我们：理由自由文本，键位固定） | MIT |
| **Labelbox** | [键位](https://docs.labelbox.com/docs/keyboard-shortcuts) · [consensus](https://docs.labelbox.com/docs/consensus) | `E` 提交、`Q` 跳过、`Tab` / `Shift+Tab` 下一 / 上一项、`Cmd/Ctrl+/` 显示键位表；consensus 按 `% coverage` 和 `# labels` 抽样多标，分数 0–1，可按分数过滤 | 键位表要在页面上随时可见；多人标只抽一部分也行 | 商用 SaaS，闭源（许可证文本**未核实**） |
| **Encord** | [review](https://docs.encord.com/platform-documentation/Annotate/annotate-label-editor/annotate-label-editor-review) · [settings & shortcuts](https://docs.encord.com/platform-documentation/Annotate/annotate-label-editor/annotate-label-editor-settings-shortcuts) | 审核 `n` 通过、`b` 拒绝；拒绝必须填理由（评论），拒绝记录可解决不可删；Single label review 用 ↑/↓ 逐个看；`Ctrl+Shift+K` 列出全部键位（搜索摘要，**未逐字核实**） | 拒绝 / FAIL 必须带理由；逐条模式隐藏其它内容 | 商用 SaaS，闭源 |
| **Scale Rapid / Nucleus** | [calibration batch](https://api-reference.scale.com/docs/launch-a-batch) · [Review and fix bad annotations](https://nucleus.scale.com/docs/review-and-fix-bad-annotations) | 先发小批"calibration batch"校准指令和分类法，Calibration Score ≥ 80 % 再上生产；Nucleus 里按查询过滤 → 逐条看 → 👍 / 👎 或 `y` / `n` → `metadata.review_status = rejected` 查回被拒的 → 打包重标 | 先用 2 个场景校准标注说明再放量；被标"重测"的行要能一条查询捞出来（我们：labels JSON 的 remeasure） | 商用 SaaS，闭源 |
| **Roboflow Annotate** | [键位](https://docs.roboflow.com/annotate/use-roboflow-annotate/keyboard-shortcuts) | 审核模式 `a` 通过 `r` 拒绝；←/→ 上 / 下一张；`Enter` 确认高亮的类别；`n` 标 Null | 单字母审核键 + 方向键翻页 | 商用 SaaS，闭源 |
| **LangSmith annotation queues** | [docs](https://docs.langchain.com/langsmith/annotation-queues) | 队列级 Instructions 固定显示在每条的侧栏；feedback key 带说明；Reservation length 锁定条目、过期回队列；"Number of reviewers per run"决定几个人 Done 才出队，审核人互相看不到反馈；选项旁显示快捷键，成对比较用 `A` / `B` / `E`，最后一项 `Enter` = Done；审完 Add to Dataset | 指南和键位贴在条目旁边；盲审（互不可见）；审完的东西直接进数据集 ↔ 我们的 labels → gold.json | 商用 SaaS，闭源 |
| **Braintrust human review** | [human review](https://www.braintrust.dev/docs/guides/human-review) · [manage review work](https://www.braintrust.dev/docs/annotate/human-review/manage-review-work) | 评分三种：categorical / continuous / free-form，可直接写进 `expected`；Blind reviews 提交前看不到别人的分；Require all scores 全填才算完成；视图 Awaiting review / Assigned to me，看板 Backlog / Pending / Complete；`r` 进入键盘导航的 review mode（搜索摘要，**未核实**） | "标签 + 自由文本理由"是标准配置；没填完不算完成（我们：未选状态 = 未标，进度不计） | 商用 SaaS，闭源 |
| **Potato** | [GitHub](https://github.com/davidjurgens/potato) · [productivity](https://potato-annotation.readthedocs.io/en/latest/productivity/) · [EMNLP 2022 论文](https://aclanthology.org/2022.emnlp-demos.33/) | `sequential_key_binding: True` 把第 1…10 个选项绑到 `1…0`，或每个标签自定 `key_value`；关键词动态高亮（TSV 文件，正则）；选项 tooltip（纯文本或 HTML）；全部 YAML 配置；注意力检查、训练阶段、Prolific / MTurk | 顺序数字键是最便宜的提速；论文称键位 + 高亮 + tooltip 带来提速（数字来自搜索摘要，**未核实**） | GPL-3.0-or-later |
| **VGG Image Annotator (VIA)** | [主页](https://www.robots.ox.ac.uk/~vgg/software/via/) | "以单个 HTML 文件分发，在浏览器里离线运行"；标注 CSV / JSON 导入导出走顶部菜单 | **和我们一样的部署形态**：一个 HTML、无服务器、导入 / 导出文件就是协作方式 | BSD-2 |
| **Universal Data Tool** | [GitHub](https://github.com/UniversalDataTool/universal-data-tool) | 自带 `.udt.json` / `.udt.csv` 文件标准，网页版和桌面版，CSV / JSON 上传下载 | 文件格式本身是接口；给它起名、加 `schema` 字段 | MIT |

## 二、我们要抄的 8 个模式

1. **一次一条、上下文齐全**（Argilla focus view、Prodigy 单条流、LangSmith 侧栏指南）。焦点面板显示规则@版本、对象名 + id + 类别 + 尺寸、测量值 ± u、阈值、裕度、缺的输入、证据事实表（谓词、参数、值 ± u、flags）、备注、视图 id。人不用翻别的文件。
2. **键盘优先、数字选状态**（Prodigy `0–9` / Potato `sequential_key_binding` / Label Studio `Choice.hotkey` / Labelbox `E` `Q` / Roboflow `a` `r` / Encord `n` `b` / Nucleus `y` `n`）。我们：`1–6` 六个 gold 状态、`n` / `p` 上下条、`r` 重测、`u` 不确定、`Enter` 保存并下一条；键位表常驻页面顶部（Labelbox 的 `Cmd+/` 思路，但不用按）。
3. **队列 + 进度 + 过滤**（Label Studio 标注流、Argilla 进度条和"自己的计数"、Prodigy 进度小部件）。左列是全部判定行的队列（机器状态、测量、gold 芯片、标记），右列是焦点；`<progress>` 显示已标 / 总数；按机器状态或"未标"过滤——"未标"过滤器就是 Label Studio 的跳过队列。
4. **标签 + 理由 + 置信度**（Braintrust categorical + free-form、Encord 拒绝必填理由、LangSmith feedback key 带说明、Argilla rating / text 题）。每行一个状态、一句理由、sure / unsure；理由输入框里按 `Enter` 直接保存并跳下一条。
5. **跳过和标记是状态，不是删除**（Label Studio Skip → Cancelled 列、Prodigy `space` 忽略 / `f` 标记、Roboflow `n` Null）。我们：`NOT_APPLICABLE` 是正式 gold 状态；`re-measure` 标记单独导出，和状态正交；没选状态的行就是"还没标"。
6. **标注 vs 审核两种视角并排**（CVAT review mode 只露审核工具、Label Studio Enterprise Accept / Fix & Accept / Reject、Encord single label review）。页面同时显示机器判定芯片和人工 gold 芯片，审核就是对比两者；记分卡（`lab/scorecard.py`）再把一致率算出来。
7. **多人一致性 = 盲审 + 冲突外露**（LangSmith 审核人互不可见、Braintrust blind reviews、Argilla `min_submitted`、Labelbox consensus 分数、Label Studio exact match）。每个审核人导出自己的 labels JSON（天然盲审）；`labels merge` 对同一 (scene, rule, subjects) 键：一致 → 一行 gold，审核人并列；不一致 → 两条都进 `conflicts`，这个键不进 `verdicts`。没有 κ 系数、没有矩阵，先 exact match。
8. **文件就是接口、单文件离线**（VIA 单 HTML 离线、UDT `.udt.json`、Label Studio 导出带 `completed_by` / `was_cancelled` / `ground_truth`、Prodigy 最近决定可改）。verdicts.html 自包含（内联 CSS / JS，无 CDN、无字体、无 fetch），file:// 和 Pages 都能开；状态存内存 + localStorage（按 run id）；导出 = 下载 `labels-<scene_id>-<run_id>.json` + 复制到剪贴板 + 文本框兜底；导入 = 选文件回填。schema 名 `verdict-labels/1`，见 `layers/l7_report/README.md`。

## 三、我们不做的

- **多用户服务器、预留、分配、角色、登录**（LangSmith reservation、Braintrust assign、Argilla 分发、Label Studio Enterprise 角色）：没有服务器；一人一文件，合并时处理冲突。
- **图像 / 像素标注工具**（CVAT、Labelbox、Encord、Roboflow 的框、多边形、mask）：我们标的是判定行，不是像素；照片证据只列 view id。
- **一致性统计面板和 κ 系数**（Label Studio stats、Labelbox consensus 分数）：两三个审核人、几十行，`conflicts` 列表够用。
- **主动学习、预标注、关键词高亮**（Potato active learning、Label Studio ML backend）：机器判定本身就是"预标注"，已经并排显示。
- **众包和校准批次**（Potato Prolific / MTurk、Scale calibration batch）：审核人是自己人；"先用 090 / 030 两个场景校准标注说明"这一条留着当流程，不做功能。
- **草稿 / 已提交 / 已丢弃四态**（Argilla）：一行只有"没标 / 标了"，标了的可以改；没有提交动作，导出就是提交。
