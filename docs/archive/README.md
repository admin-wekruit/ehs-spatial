# docs/archive：历史文档（只读，不是现状）

2026-10-08 整理：这里的每份文件都是写它那天的状态，之后被取代或完成了。**不要按它们操作，也不要把它们当作系统现状。** 现状只看 `docs/STATE.md`，操作只看 `HANDOFF.md`。
文件内部的相对链接仍指向它们原来的位置（例如 `docs/reviews/...`），没有改写。

| 目录 / 文件 | 原来是什么 | 为什么归档 |
|---|---|---|
| `HANDOFF_FULL-2026-08-30.md`、`HANDOFF-history-2026-08-30.md` | 8 月的四图 MVP 交接（tarball、Release、公开 serving 仓库、Gradio 工作台） | 被 2026-10-07 的企业版交接（根目录 `HANDOFF.md`）取代；panoptes-serving 仓库已冻结 |
| `serving-HANDOFF-v0-services.md` | v0 三个模型服务（sam3 / mapanything / moge）的 GPU 机部署说明 | 并入 `HANDOFF.md` §3 和 `deploy/`；v0 契约仍在 `docs/BACKENDS.md` |
| `PRODUCT_PLAN.md`、`RESEARCH_BRIEF.md`、`reconciliation-handoff-vs-mvp.md`、`agent-harness-selection.md`、`detection-multiframe-check.md`、`object-generation-workflow.md` | 9 月产品计划、研究简报、早期设计选择 | 阶段过去；现状在 `docs/MILESTONES.md` |
| `platform/` | 9 月平台实现 / QA / 验收记录（身份、关联、CAD、报告加载…） | 已完成的工作记录；平台运维现行文档是 `docs/platform/OPERATIONS.md` |
| `workcell-photo/` | 9 月底 → 10 月 6 日工位照片流程的研发记录（精度计划、下沿、地面卡尺、one-shot、on-prem 草案、拍摄协议、两个工位说明） | 被 `research/module-swap-2026-10-07/`（模块替换、冻结数据、REPRODUCE-PROMPT）和 `HANDOFF.md` 取代；`CAPTURE-PROTOCOL.md`、`LOWER-EDGE.md` 仍是有用的背景 |
| `phase2/` | 视频空间 MVP（固定相机、人员、动态场景、流式计划）的设计与交接 | 视频阶段暂停；判定层设计里时间轴的处理见 `docs/research/verdict-layer-architecture-2026-10-08.md` §4；`INFERENCE-POLICY.md` 的"只用观察到的几何"原则仍有效 |
| `algorithms/` | 9 月 18 日的"算法与分析流程总表" | 流程已被模块替换版取代（`docs/STATE.md` §2） |
| `reviews/`、`demos/`、`superpowers/` | 7–8 月的评审、POC 演示、计划 / 规格 | 历史；`reviews/2026-08-25-osha-compiler-exam.md` 仍被代码注释和判定层调研引用 |
| `research/` | 7–9 月的架构 / 基准综述、数据源扫描、RecGen 质量、视频空间记忆 | 历史；现行研究在 `docs/research/` |
