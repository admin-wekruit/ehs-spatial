# docs/research：研究文档索引（提案和实验，不是已交付的系统）

规则：这里的文件描述调研、试跑和提案。**和 `docs/STATE.md` / `HANDOFF.md` 冲突时，以那两份为准。** 带 `survey-*` 后缀的是调研 agent 的原文（英文，带链接，"unverified" 是 agent 自己的标记）。

## 判定层（Phase 4，2026-10-07 → 08）——进行中，未交付

| 文件 | 内容 | 状态 |
|---|---|---|
| `verdict-layer-plan-2026-10-08.md` | 计划：严格 / 解耦 / 可比较；实验台（插件注册表、台账、记分卡，一次只换一层）；五个阶段 | **当前** |
| `verdict-lab-scorecard-v0-2026-10-08.md` | 实验台第一张记分卡：基线 / python 引擎等价 / stpl + k=1，090 与 030 | **当前**；代码在 `ehs_spatial/verdict/`（`README.md` 是 lab 规则，`lab/README.md` 是用法） |
| `verdict-layer-architecture-2026-10-08.md`（+ `.png`） | 七层、四个契约的分层与解耦设计；DAAAM / WorldSGG / ChronoGraph 放在哪 | **当前设计** |
| `verdict-spec-to-check-2026-10-08.md`（+ `survey-E`） | 从自然语言规范到可执行检查：对逐句模板法的批评、替代路线对比、推荐架构、先做什么 | **当前设计**（取代了逐句模板法） |
| `verdict-evaluation-protocol-2026-10-08.md` | 评测与泛化：冻结基准 + 固定输出、分层记分、留出工位 / 规范、消融代替复现论文 | 当前设计 |
| `verdict-prior-work-links-2026-10-08.md` | 别人做过的类似系统：做了什么 / 效果 / 失败 / 怎么学 / 链接（30 个） | 参考 |
| `verdict-prior-work-repos-2026-10-08.md`（+ `survey-F`） | 同上系统的仓库、许可证、可否商用、怎么和我们组件结合（逐个核过） | 参考 |
| `verdict-patterns-lessons-2026-10-07.md`（+ `survey-C`、`survey-D`） | 九个领域怎么做"表示 → 规则 → 判定"；场景图帮不帮助理解；预训练 / 后训练能不能替代 | 参考 |
| `verdict-layer-rules-2026-10-07.md`（+ `survey-A`、`survey-B`） | 现状审计（判定不经 LLM；三套扁平 JSON；报告无机器判定）、规则语言调研、照片可判的标准数字、建议、harness、匹配示例 | 参考；其中 D2 / D6 的"逐句模板"部分已被 spec-to-check 文档取代 |

试跑代码与结果：`research/verdict-layer-trial-2026-10-07/`（README 说明哪些已被取代）。

## 以前的研究（2026-07 → 09）

已移到 `docs/archive/research/`（架构与基准综述、数据源、RecGen 质量、视频空间记忆等）。它们描述当时的实验，不是现状。
