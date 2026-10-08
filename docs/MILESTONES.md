# Panoptes 阶段与里程碑（标记日期 2026-10-07）

这份文件是"我们现在在哪"的唯一标记；每次阶段变化改这里，和内部 / 客户同步时引用这里。细节各自链接。

## 当前位置

**Phase 3（企业版交接 1.0.0-rc1）已交付，等客户侧验收；Phase 4（判定层）已开工：调研 + 试跑完成，等安全规范（safety specs）输入。**

| 阶段 | 时间 | 状态 | 里程碑 / 证据 |
|---|---|---|---|
| Phase 0 MVP：照片 → SceneMap → 政策编译器 + 7 谓词几何判定 + 报告 | 2026-08 | 完成（已被后续取代，保留） | 交接 2026-08-30（`HANDOFF.md` History 节）；OSHA 编译器考试 `docs/archive/reviews/2026-08-25-osha-compiler-exam.md` |
| Phase 1 工位照片流程基线（Sept）：四视角 WorkcellReport、RecGen、对象流水线视图、measurement layer | 2026-09 | 完成 | 发布 090 `4b58dbd2…`、030 `cd84d3fb…`、`25686138…`；Pages `admin-wekruit/panoptes-workcell-report` |
| Phase 2 同一流程换模块：SAM 3D 替 RecGen；MVS + MoGe-3 补洞替 Pi3X；组装 v2 地面接触 | 2026-10-01 → 10-07 | 完成，结论已给 | 090 五版 + 030 三版上线；推荐组合 090 明显错误 0（原 8）、IoU 0.823、现场值 MAE 0.50 cm；030 明显错误 1；链接 `research/module-swap-2026-10-07/REVIEW-LINKS-2026-10-06.md` |
| Phase 3 企业版交接：一个仓库、GPU 模型服务 v1 契约、provider http/local/modal、`panoptes run` CLI、Mongo / S3、HANDOFF、CI | 2026-10-07 | 交付；等客户验收 | ehs-spatial `main` 4ce0880；`HANDOFF.md`、`CHANGELOG.md` 1.0.0-rc1、`docs/E2E-2026-10-07.md`（真 A100 上 sam3d 服务经 http provider 端到端）；panoptes-serving 冻结（dfae8fa） |
| Phase 4 判定层（verdict layer）：安全规范 → 规则 → 带不确定度的判定 | 2026-10-07 开工 | 进行中 | 调研 `docs/research/verdict-layer-rules-2026-10-07.md`（+ 附录 A / B）；试跑 `research/verdict-layer-trial-2026-10-07/`（clingo 规则包 v0 对 090 / 030：15 + 12 条判定） |

## Phase 3 客户侧未关闭项（我们等）

1. 把 ehs-spatial 默认分支切到 `main`（文档都写了 `-b main`，不阻塞）。
2. 两张 A100 上 docker 构建 + `make up GPU=a|b` + `make smoke`。
3. 真实 Mongo / S3 过一遍（契约测试用 mongomock / moto 过了）。
4. geometry-mvs 服务真 GPU 跑一次（中间结果已随 `data/` 给出）。
5. 验收 A1–A6（`HANDOFF.md` §5）。

## Phase 4 已定原则与下一步

- **判定层与重建解耦**（2026-10-07 用户要求）：判定层只读米制场景契约（`research/verdict-layer-trial-2026-10-07/scene.py`），不读网格 / run 目录 / 流程；adapter 隔离。
- **判定不经模型权重**：LLM 只把规范编译成可审阅的规则（封闭词表 + 拒绝），判定由确定性引擎（clingo）按护带决策规则出 PASS / FAIL / NEEDS_MEASUREMENT / CANNOT_DETERMINE。
- 下一步顺序（调研 D9，2026-10-08 按 `docs/research/verdict-spec-to-check-2026-10-08.md` 修订）：① 等用户提供 safety specs → **条款图抽取（表 / 定义 / 例外）+ 对齐表 + 按场景检索 + 对有类型场景 API 合成检查 + 自动验证 + 字面化审阅**（不再逐句整理进固定模板）→ 规则包 v1；② 补让引擎弃权的四个缺口（尺度状态、整高 σ、危险区、适用性）；③ 关系层 + clingo 内核迁移现有 7 谓词；④ 拓扑规则；⑤ 金标集 + 蜕变测试。
- 试跑结论：围栏高、光幕最低光束、机器人到立柱 / 光幕间隙今天就能判；离地缝判不了的原因是 low 置信度盒没有 σ（重建层的事）；包围判不了是因为照片没覆盖全（要观察范围）。
- **架构立场（2026-10-07 晚，跨领域调研 C + 训练路线调研 D，`docs/research/verdict-patterns-lessons-2026-10-07.md`）**：
  基础模型产"带不确定度的符号" → 符号规则判定 → 解释 = 证据链。场景图保持薄，只放几何算出来的米制关系（学出来的关系谓词在所有基准上都是最差的数）；
  不训练"照片 → 判定"；后训练只给符号生产者（受管类别检测适配、类别链接器）和规则编译器；尺度标定是流程不是模型。视频阶段同一骨架：训练给动态符号生产者，判定是每帧场景图上的时序逻辑。
- **采纳的教训（12 条，按优先级）**：① 计量式判定（值 ± U、决策规则、四态陈述）；② 缺失是值（三值传播，缺边 → NEEDS_INPUT / CANNOT_DETERMINE，永不 PASS）；
  ③ 规则包是产品（条款对齐、RASE 标注、按标准版本带日期、带测试、专家签字）；④ 发布编译率（编译 / 拒绝 / 需人工修）；⑤ 声明输入保持声明（T、接近速度、PLr、参照物尺寸）；
  ⑥ 感知层单独规约（跨视角持久性、尺寸一致、地面接触，规则前检查）；⑦ 适用性人工签字；⑧ 每条判定带可申诉的证据链；⑨ FAIL 精确率优先于召回；
  ⑩ 图薄且可查询；⑪ 词表外情况记 SOTIF 式 backlog；⑫ 形式化发现的规范 bug 回给 spec 作者。
- **规范 → 检查的路线（2026-10-08，用户否决逐句模板法后调研 E 的结论）**：用抽取代替整理；词表是对齐的签名不是拒绝的门；每条检查在人看之前先过冗余翻译差分 + 性质测试 + 蜕变；判定仍是 clingo 确定性 + 护带 + 证据链。先在 ISO 13857 + 13855 的 20 条条款上和旧编译器对比（编译率、每条审阅时间、与工程师一致率）。
- **评测立场**：没有现成的"从照片判机械安全"benchmark（两份调研都没找到），判定层的 benchmark 自己建；别人的 benchmark 只评部件。分层评，每层有自己的真值，LLM 永不当判定的评委（细节：研究主文 F 节）。

## 内部同步时引用

- 代码与数据：https://github.com/admin-wekruit/ehs-spatial（`main`）
- 交接：`HANDOFF.md`；复现：`research/module-swap-2026-10-07/REPRODUCE-PROMPT.md`
- 报告链接与结论：`research/module-swap-2026-10-07/REVIEW-LINKS-2026-10-06.md`
- 判定层：`docs/research/verdict-layer-rules-2026-10-07.md`、`research/verdict-layer-trial-2026-10-07/README.md`
