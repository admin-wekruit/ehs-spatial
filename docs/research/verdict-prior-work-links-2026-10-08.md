# 别人做过的类似研究：效果、链接、我们怎么学（2026-10-08）

按和我们相似度排序。数字是各系统自己报告的；来源见链接（全文摘录在附录 A–E）。

## 第一组：规范 → 结构化场景 → 判定（最像我们）

| 系统 | 效果 | 失败点 | 我们怎么学 | 链接 |
|---|---|---|---|---|
| **GuardEn**（EMNLP 2026） | SafetyVisionBench 3.4 万图 F1 78.7（基线 68.9），建筑类 86.4；消融：去关系落地 −30.1、去验证循环 −27.4、去同义词重落地 −16.5 | 覆盖受规则完整度限制；表格 / 米制阈值弱 | 编译要带验证循环；词表缺口重落地不拒绝；关系用几何算 | https://arxiv.org/abs/2609.18328 |
| **I-SNACC**（ITcon 2023） | IBC 第 10 章 P 95.2 / R 100；第一轮 51.7，靠人工逻辑精修到 100；每案例规则开发 24 → 13 min | 换章节要人工适配 | 高准确率 = 抽取 + 人精修闭环；每条预算十几分钟 | https://itcon.org/papers/2023_01-ITcon-Wu.pdf · https://par.nsf.gov/biblio/10420120 |
| **ACCORD / AEC3PO + RASE**（EU 2022–25） | RASE 标到表格单元格；D2.2 走人工 + 自动混合；CODE-ACCORD 语料 862 句（只收自足句） | 需要上下文的句子被排除 | RASE 四问在章节 + 表格层面问；表格单独建模 | https://accordproject.eu/wp-content/uploads/2024/02/ACCORD_D2.2_BCO_Ontology_and_Rules_Format.pdf · https://arxiv.org/abs/2403.02231 · https://cumincades.scix.net/pdfs/w78-2011-Paper-45.pdf · https://github.com/Accord-Project/aec3po |
| **LLM-FuncMapper**（清华 2023） | 66 个原子函数；top-5 检索 81.55 %；"几乎 100 % 可计算条文可表达" | 口径自定；长尾 | 关系库 → 有类型的函数 API，模型组合调用 | https://arxiv.org/abs/2308.08728 |
| **TUM Fuchs / Borrmann**（EC3 2026） | CodeAct agent 生成可复用检查函数，验证器迭代精修 | 需辅助构造的几何仍难；成功率未报 | 生成的检查进库复用；验证器在循环里 | https://ec-3.org/?p=15476 |
| **SGR-BIM**（HKUST 2026） | 679 条专家核验消防查询 84.3 %（+8.6） | — | "法规图 ↔ 场景图"对齐层独立建、独立评 | https://arxiv.org/abs/2606.12065 |
| **Han et al.**（Buildings 2026） | 翻译准确 95.8 %、可执行 98.3 %；专家工时 1,620 → 168 h | — | 置信度分流审阅；度量每条审阅时间 | https://www.mdpi.com/2075-5309/16/4/719 |
| **Zhang & El-Gohary 概念对齐**（2023） | IFC ↔ 法规概念 transformer 对齐约 80 %（未核） | — | 对齐是可学习、可缓存的一级 | https://experts.illinois.edu/en/publications/transformer-based-approach-for-automated-context-aware-ifc-regula/ |
| **Solibri**（商用） | 50+ 参数模板；复杂规则填模板 ≥ 30 min；用户抱怨陈旧结果；Zou 2023：采用率低的技术原因含"统一的条文解释方式" | 模板天花板 | 条文解释当产品做，带版本和 QC | https://help.solibri.com/hc/en-us/articles/1500005009042-Understanding-Checking · https://society.solibri.com/post/13642 · https://par.nsf.gov/servlets/purl/10347911 · https://ascelibrary.org/doi/abs/10.1061/JMENEA.MEENG-5051 |

## 第二组：场景图 + LLM 推理（显式几何胜过端到端）

| 系统 | 效果 | 我们怎么学 | 链接 |
|---|---|---|---|
| **RieMind**（2026-03） | LLM + 3D 场景图 + 几何工具不微调：VSI-Bench 静态 GPT-4.1 89.5 %（基础 35.3 %），超微调 SOTA 73.6 % | 关系库暴露成工具做证据收集；判定仍由引擎出 | https://arxiv.org/abs/2603.15386 |
| **MIT Structured Interfaces**（2025-10） | Cypher 工具 77 % vs 塞图 33 %；token 2,395 vs 582,202 | 图要可查询，不塞给模型 | https://arxiv.org/abs/2510.16643 |
| **3DSSG**（CVPR 2020 / 2025-11 边中心模型） | 关系 R@50 ≈ 0.29–0.31；三元组 R@50 0.40 | 关系算出来，不学 | https://arxiv.org/abs/2004.03967 · https://arxiv.org/abs/2511.15288 |
| **GQA 场景图消融** | 真值图 94.6 % vs 生成图 52.9 % | 同上 | https://ar5iv.labs.arxiv.org/html/2101.05479 |
| **CodeAct**（ICML 2024） | 代码动作比 JSON 动作成功率高最多 20 % | 固定 JSON 模板是被打败的那一方 | https://proceedings.mlr.press/v235/wang24h.html |

## 第三组：带验证的自动形式化

| 系统 | 效果 | 我们怎么学 | 链接 |
|---|---|---|---|
| **ARc / AWS Automated Reasoning checks** | k 份冗余翻译 + SMT 等价 → > 99 % 可靠；TRANSLATION_AMBIGUOUS 状态；精修 diff + 测试影响，人批准 | 整理先过冗余翻译 + 差分执行；歧义是正式状态 | https://arxiv.org/abs/2511.09008 · https://aws.amazon.com/about-aws/whats-new/2025/08/automated-reasoning-checks-amazon-bedrock-guardrails · https://aws.amazon.com/blogs/machine-learning/automated-reasoning-policy-refinement-in-amazon-bedrock |
| **nl2spec**（CAV 2023） | 子翻译交互：44 % → 86.1 %，1.4 轮 | 审阅单位 = 片段 ↔ 调用对照表 | https://arxiv.org/abs/2303.04864 |
| **VLTL-Bench**（2025-12） | 翻译 78–99 %，落地 26–72 % | 失败在落地；对齐单独做 | https://arxiv.org/abs/2507.00877 |
| **GinSign**（2025-12） | 对签名做结构化分类，落地等价 95.5 % | 签名是对齐目标 | https://arxiv.org/abs/2512.16770 |
| **P4IR**（2026-06） | SFT + GRPO 产规则 IR，胜 Opus 4.5 / GPT-5.2 零样本 | 数据够再微调编译器 | https://arxiv.org/abs/2606.22402 |
| **PropTest**（EMNLP 2024） | 先写性质测试再执行：+6–8 分 | 每条检查配自动性质测试 | https://aclanthology.org/2024.findings-emnlp.483/ |
| **Know Your Limits**（2026-06） | scope laundering、漏隐含约束 | 判定只认引擎输出 | https://arxiv.org/abs/2606.16118 |
| **Logic-LM**（2023） | LLM → 求解器 +39.2；可执行率是 KPI | 发布编译率 | https://arxiv.org/abs/2305.12295 |
| **LogiSafetyGen**（2026-01） | 签名验证器拒绝未知谓词；73.9 % 通过作者审 | 签名验证器保留 | https://arxiv.org/html/2601.08196v1 |
| **MONIR 四值合规推理**（2026-06） | 四值判定的理论 | 五状态有理论依据 | https://arxiv.org/abs/2606.04619 |

## 第四组：检索

| 系统 | 效果 | 我们怎么学 | 链接 |
|---|---|---|---|
| **DriveReg** | 562 条规则，场景 → 条款检索 100 %（BM25 60 %） | 先检索再编译 | https://arxiv.org/abs/2410.04759 |
| **Lawful-AD**（2026-04） | 锚在场景分类的检索 +29.1 % | 检索锚在工位分类 | https://arxiv.org/abs/2604.24562 |
| **BifrostRAG**（2025-07） | 实体图 + 文档结构图，OSHA 多跳 F1 87.3 | 条款图带文档结构 | https://arxiv.org/abs/2507.13625 |

## 第五组：已部署系统与基线

| 系统 | 效果 | 我们怎么学 | 链接 |
|---|---|---|---|
| **MonitorVLM-v2**（2026-08） | VLM 只预测有限规则 ID，熵高交人；井下 4 个月，确认违规 2.78× 人工 | 输出空间有限可枚举；不确定交人 | https://arxiv.org/abs/2608.00975 |
| **ConstructionSite-10K**（Molmo 评测） | 直判 F1 61–73 | 直判当基线 | https://arxiv.org/abs/2511.15720 · https://www.cambridge.org/core/services/aop-cambridge-core/content/view/4F9F8B39B34FD6F2B201C9947CDF42E8 |
| **HomeSafeBench** | 最好 VLM F1 34.7 vs 人 98.0 | 同上 | https://arxiv.org/abs/2509.23690 |
| **CompAgent**（2025） | agent + 工具 76 % F1（UnsafeBench） | 工具收证据可以，判定不行 | https://arxiv.org/abs/2511.00171 |
| **STPL / Scene Flow Specs / Abstract Scene Graphs**（AD） | 感知输出自己的规约；纯场景图写不了时序性质 | 规则前先跑感知规约；视频实体持久 | https://arxiv.org/abs/2206.14372 · https://doi.org/10.1145/3729382 · https://arxiv.org/abs/2511.14430 · https://par.nsf.gov/biblio/10596359 |
| **ILAC-G8 / ISO 14253-1 / JCGM 106 / UKAS LAB 48** | 护带 + 四态陈述 | 已采用；报告印决策规则和 U | https://ilac.org/?p=122723 · https://www.evs.ee/en/iso-14253-1-2017 · https://www.bipm.org/documents/20126/2071204/JCGM_106_2012_E.pdf · https://ukas.com/wp-content/uploads/2023/05/LAB-48-Decision-rules-and-statements-of-conformity.pdf |
| **空间后训练基线**（VSI-Bench、SpatialRGPT） | 绝对距离接近随机；±25 % 内 35–59 % | 训练买不到厘米级 | https://arxiv.org/abs/2412.14171 · https://arxiv.org/abs/2406.01584 |

## 学习顺序
1. 验证循环先于一切（GuardEn、ARc）。2. 落地 / 对齐单独做（VLTL-Bench、GinSign）。3. 关系算出来并暴露成 API / 工具（GuardEn、RieMind、FuncMapper、TUM）。
4. 条款图带表格和结构（ACCORD、BifrostRAG）。5. 先检索再编译（DriveReg、Lawful-AD）。6. 片段级审阅 + 置信度分流，度量每条分钟数（nl2spec、Han、I-SNACC）。7. 判定只认引擎（MonitorVLM-v2、Know Your Limits）。
