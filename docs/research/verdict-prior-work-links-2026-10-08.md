# 别人做过的类似研究：做了什么、效果、失败、我们怎么学、链接（2026-10-08）

按和我们相似度排序。数字是各系统自己报告的；全文摘录在附录 A–E。仓库 / 许可证 / 可否商用 / 怎么和我们的组件结合 → 见同目录
`verdict-prior-work-repos-2026-10-08.md`（查证中，查到即补）。

## 第一组：规范 → 结构化场景 → 判定（最像我们）

| 系统 | 做了什么 | 效果 | 失败点 | 我们怎么学 | 链接 |
|---|---|---|---|---|---|
| **GuardEn**（EMNLP 2026） | 安全政策分解成原子命题树（object-attribute / object-relation-object），分解带验证循环；每张图用 SAM 3 + SigLIP 2 + Depth Anything 3 + OCR 建场景图，关系用预定义几何规则算；检测置信度 < 0.8 就换同义词重落地并缓存 | SafetyVisionBench 3.4 万图（FDA 食品法规、NYC 建筑法规、Meta 标准、UK CAP）：F1 78.7（最强基线 68.9）；建筑类 86.4 vs CompAgent 80.6。消融：去关系落地 −30.1、去验证循环 −27.4、去同义词落地 −16.5、去场景落地执行 −12.8 | 覆盖受规则完整度限制；只能演绎；表格 / 米制阈值弱；没有不确定度 | 编译要带验证循环，不是单次；词表缺口要重落地，不是拒绝；关系用几何算——这正是我们场景图的做法，是 F1 的主要来源 | https://arxiv.org/abs/2609.18328 |
| **I-SNACC**（ITcon 2023，Purdue） | IBC 2015 第 10 章：模式规则把句子转成 Prolog Horn 子句；IFC → "不变签名" → 逻辑事实；带人工逻辑规则精修模块 | 两个真实项目 P 95.2 / R 100；60 个手造不合规案例；第一轮 51.7，精修后 100；每案例规则开发 24.23 → 13.22 min | 换章节要"人工适配"；模式规则不泛化 | 高准确率来自"抽取 + 人精修"闭环，不是全自动；每条预算十几分钟人工精修，精修记录当资产 | https://itcon.org/papers/2023_01-ITcon-Wu.pdf · https://par.nsf.gov/biblio/10420120 |
| **ACCORD / AEC3PO + RASE**（EU Horizon 2022–25） | 条文按 Requirement / Applicability / Selection / Exception 标注（细到表格单元格），编成 AEC3PO 本体（含 Definition / Table / Cell / Row / Column 模块）+ SHACL；D2.2 明确走人工 + 自动混合 | CODE-ACCORD 语料 862 句、4,297 实体、4,329 关系、12 位标注者——只收"自足"的句子 | 需要上下文的句子被排除；RASE 四标签的理论基础至今在找；自动翻译准确率"未证明" | RASE 的四问（选谁 / 何时 / 要求 / 例外）值得问，但在章节 + 表格层面问；表格、定义、例外要单独建模 | https://accordproject.eu/wp-content/uploads/2024/02/ACCORD_D2.2_BCO_Ontology_and_Rules_Format.pdf · https://arxiv.org/abs/2403.02231 · https://cumincades.scix.net/pdfs/w78-2011-Paper-45.pdf · https://github.com/Accord-Project/aec3po |
| **LLM-FuncMapper**（清华 2023 / 2026） | 从建筑规范蒸馏 66 个原子函数；LLM 检索相关函数并组合成可执行代码 | top-5 函数检索 81.55 %（比微调 BERT 高 19 %）；"几乎 100 % 可计算条文可表达"（自定口径） | "可计算"由自己的分类器定义；长尾 | 我们的关系库就是"原子函数库"的雏形；下一步是有类型的 API 让模型组合调用，不是加模板 | https://arxiv.org/abs/2308.08728 |
| **TUM Fuchs / Hellin / Borrmann**（EC3 2026） | CodeAct 式 agent 为 3 类（高层语义）规范生成**可复用**检查函数，用 BIM 数据做强验证器迭代精修 | 函数跨场景复用；成功率摘要未给（未核） | "需要辅助构造的几何推理仍难" | 生成的检查函数进库复用；验证器在生成循环里 | https://ec-3.org/?p=15476 |
| **SGR-BIM**（HKUST 2026） | 跨模态知识图谱对齐用户意图、法规语义、BIM 几何 | 679 条专家核验消防查询 84.3 %，比"单 agent + 增强工具"高 8.6 | — | "法规图 ↔ 场景图"的对齐层是独立一级，单独建、单独评 | https://arxiv.org/abs/2606.12065 |
| **Han et al.**（Buildings 2026） | BERT 抽取 + 文法结构校验 + 置信度分流给专家审 | 翻译准确 95.8 %、可执行 98.3 %；专家工时 1,620 h → 168 h（−90 %）；法规变更处理 −94 % | — | 置信度分流审阅是把人工压下来的唯一有数据的办法；度量"每条审阅时间" | https://www.mdpi.com/2075-5309/16/4/719 |
| **Zhang & El-Gohary 概念对齐**（2023） | transformer 把 IFC 概念对齐到法规概念 | 约 80 %（未核） | — | 对齐是可学习、可缓存的一级 | https://experts.illinois.edu/en/publications/transformer-based-approach-for-automated-context-aware-ifc-regula/ |
| **Solibri + Zou 2023**（商用 / 采用率研究） | 50+ 参数化规则模板，用户填参数不写逻辑；严重度按偏离量自动算；Zou 访谈 20 位专家 8 国 | 复杂规则填模板 ≥ 30 min（Zhang & El-Gohary）；采用率低的技术原因含"统一的条文解释方式"；用户抱怨陈旧 / 错误结果 | 模板天花板（Solihin–Eastman 1–2 类）；超出要写 Java | 条文解释当产品做，带版本和 QC；模板不是终点 | https://help.solibri.com/hc/en-us/articles/1500005009042-Understanding-Checking · https://society.solibri.com/post/13642 · https://par.nsf.gov/servlets/purl/10347911 · https://ascelibrary.org/doi/abs/10.1061/JMENEA.MEENG-5051 |

## 第二组：场景图 + LLM 推理（显式几何胜过端到端）

| 系统 | 做了什么 | 效果 | 失败点 | 我们怎么学 | 链接 |
|---|---|---|---|---|---|
| **RieMind**（2026-03） | LLM 接持久 3D 场景图 + 四类工具（记忆摘要、场景查询含最近邻、几何：尺寸 / 体积 / 距离 / 坐标变换），不微调 | VSI-Bench 静态 4,185 题：GPT-4.1 89.5 %、GPT-4o 85.2 %（基础 VLM 35.3 %）、Qwen2.5-VL-7B 64.1 %；超微调 SOTA 73.6 %；每题 1–4 次工具调用 | 场景图来自真值标注；没有"不知道"状态；小模型 5–6 步链失败 | 关系库暴露成工具做证据收集；判定仍由引擎出 | https://arxiv.org/abs/2603.15386 |
| **MIT Structured Interfaces**（2025-10） | 对比三种让 LLM 用场景图的方式：Cypher 查询工具 / Python 工具 / 把图塞进上下文 | 大图问答 Cypher 77 % vs 塞图 33 % vs Python 32 %；token 2,395 vs 582,202 | 只度量任务成功率 | 图要可查询，不塞给模型 | https://arxiv.org/abs/2510.16643 |
| **3DSSG**（CVPR 2020；2025-11 边中心模型） | 1,553 个 3RScan 场景、160 类、26 谓词的学习式场景图 | 预测标签下关系 R@50 ≈ 0.29–0.31；三元组 R@50 0.40；hanging on / attached to / supported by 系统性混淆 | 学出来的谓词不可靠 | 关系算出来，不学 | https://arxiv.org/abs/2004.03967 · https://arxiv.org/abs/2511.15288 |
| **GQA 场景图消融** | 真值场景图 vs 生成场景图做 VQA | 94.6 % vs 52.9 %；生成图的边和真值只重叠 1–2 处 | 同上 | 同上 | https://ar5iv.labs.arxiv.org/html/2101.05479 |
| **CodeAct**（ICML 2024） | 把"代码"当 agent 的动作空间，对比 JSON / 文本动作 | M3ToolEval 82 任务上最多高 20 % | — | 固定 JSON 模板是被打败的那一方 | https://proceedings.mlr.press/v235/wang24h.html |

## 第三组：带验证的自动形式化

| 系统 | 做了什么 | 效果 | 失败点 | 我们怎么学 | 链接 |
|---|---|---|---|---|---|
| **ARc / AWS Automated Reasoning checks**（2025–26，商用） | k 份冗余 LLM 翻译 → SMT 等价比较得置信度；政策带测试（VALID / INVALID / SATISFIABLE / IMPOSSIBLE / TRANSLATION_AMBIGUOUS）；精修建议带测试影响 diff，人必须批准；规则链接到来源语句 | > 99 % 可靠、近零误报；Bedrock Guardrails GA | 要先有领域模型（变量、类型） | 整理先过冗余翻译 + 差分执行；歧义是正式状态；审阅屏显示 diff 和测试影响 | https://arxiv.org/abs/2511.09008 · https://aws.amazon.com/blogs/machine-learning/automated-reasoning-policy-refinement-in-amazon-bedrock |
| **nl2spec**（CAV 2023） | NL → LTL，UI 显示（条文片段 → 子公式）对照表让人逐条改 | 36 条专家规约：44.4 % → 86.1 %，平均 1.4 轮 | — | 审阅单位 = 片段 ↔ 调用对照表 | https://arxiv.org/abs/2303.04864 |
| **VLTL-Bench**（2025-12） | 分开度量"提升式翻译"和"落地到系统谓词" | 翻译 78–99 %；落地后 NL2TL 54–60 %、nl2spec 30–35 %、NL2LTL 26–38 %、Lang2LTL 38–72 % | 落地是瓶颈 | 对齐 / 落地单独做、单独评 | https://arxiv.org/abs/2507.00877 |
| **GinSign**（2025-12） | 把落地到"系统签名"（预定义原子命题 + 类型常量）当结构化分类 | 落地逻辑等价 95.5 %（1.4× SOTA） | 假设签名存在 | 签名是对齐目标，不是拒绝的门 | https://arxiv.org/abs/2512.16770 |
| **P4IR**（2026-06） | SFT + GRPO 训练模型产建筑规范规则的代码骨架 IR | 树编辑距离 −23.8 %、Levenshtein −38.6 % vs SFT；胜 Claude Opus / Sonnet 4.5、GPT-5.2、Qwen-3-Max 零样本 | 奖励是结构不是功能正确 | 数据够了再微调编译器；奖励加测试通过 | https://arxiv.org/abs/2606.22402 |
| **PropTest**（EMNLP 2024） | LLM 先为生成的程序写性质测试（类型、语法、语义）再执行 | GQA 46.1 %（+6.0）、RefCOCO+ 59.5 %（+8.1），Llama3-8B | — | 每条生成的检查配自动性质测试 | https://aclanthology.org/2024.findings-emnlp.483/ |
| **Know Your Limits**（2026-06） | ContractNLI → Z3 自动形式化的失败分析 | 记录 scope laundering（不跑求解器就报答案）、隐含约束盲区、结构化提示下仍写错 Z3 | — | 判定只认引擎输出 | https://arxiv.org/abs/2606.16118 |
| **Logic-LM**（2023） | LLM → 形式程序 → 求解器，求解器错误回喂 | 五个逻辑数据集 +39.2（标准提示）/ +18.4（CoT）；可执行率按数据集 33–100 % | 真实文本编译率上限 | 发布编译率当 KPI | https://arxiv.org/abs/2305.12295 |
| **LogiSafetyGen**（2026-01） | 政策 → LTLf（两个模板），签名验证器拒绝引用不存在 API 谓词的公式；fuzzer 造满足规约的轨迹 | 73.9 % 生成公式通过作者审 | 模板受限 | 签名验证器保留 | https://arxiv.org/html/2601.08196v1 |
| **MONIR**（2026-06） | 四值合规推理理论 | — | — | 五个判定状态有理论依据 | https://arxiv.org/abs/2606.04619 |

## 第四组：检索

| 系统 | 做了什么 | 效果 | 失败点 | 我们怎么学 | 链接 |
|---|---|---|---|---|---|
| **DriveReg** | VLM 写场景描述 + 查询 → 段落检索（ada-002 + FAISS）→ 句子精排（MiniLM）；562 条规则（波士顿 / LA / 新加坡） | 20 场景检索 100 %（BM25 60 %）；检索让 GPT-4o 合规 91 → 93 % | 规模小 | 先检索再编译 | https://arxiv.org/abs/2410.04759 |
| **Lawful-AD**（2026-04） | 法规检索锚在场景分类（5,897 场景） | 法规-场景匹配 +29.1 %，强制性要求 +36.9 % | — | 检索锚在工位分类 | https://arxiv.org/abs/2604.24562 |
| **BifrostRAG**（2025-07） | 实体图 + 文档结构导航图做 OSHA 多跳问答 | F1 87.3 vs 向量 RAG 75.0 vs Neo4j 图 RAG 56.5 | — | 条款图带文档结构（章节、表、交叉引用） | https://arxiv.org/abs/2507.13625 |

## 第五组：已部署系统与基线

| 系统 | 做了什么 | 效果 | 失败点 | 我们怎么学 | 链接 |
|---|---|---|---|---|---|
| **MonitorVLM-v2**（2026-08） | 把巡检变成对有限规则 ID 集合的单 token 预测；SymPO 对比训练；熵高交人 | 井下 10 路 4 个月：确认违规 2.78× 人工；不做测量声明 | 只分类，不测量 | 模型输出空间有限可枚举；不确定交人 | https://arxiv.org/abs/2608.00975 |
| **ConstructionSite-10K**（Molmo 评测） | VLM 直判工地规则 | Molmo-7B 单提示 F1 61.5，10 提示集成 67.2；空间关系类规则最差 | 漏报、编造、不出数字 | 直判当基线 | https://arxiv.org/abs/2511.15720 · https://www.cambridge.org/core/services/aop-cambridge-core/content/view/4F9F8B39B34FD6F2B201C9947CDF42E8 |
| **HomeSafeBench** | 1,000 个人工核验的家居危险任务 | 最好 VLM F1 34.7 vs 人 98.0；微调 4B 45.3；精确率远高于召回 = 系统性漏报 | — | 同上 | https://arxiv.org/abs/2509.23690 |
| **CompAgent**（2025） | 规划 agent 选工具（检测、人脸、NSFW、描述）+ 验证 agent 融合图像 / 工具 / 政策文本 | UnsafeBench F1 76 %（+10） | 无不确定度、无成本数据 | 工具收证据可以，判定不行 | https://arxiv.org/abs/2511.00171 |
| **STPL / Scene Flow Specs / Abstract Scene Graphs / SGSM**（AD 2022–25） | 给感知输出本身写规约（对象持久、分类一致、空间关系）；发现纯场景图写不了时序性质，加持久实体 | Scene Flow Specs 编码 96 % 的 AD 规约 vs 之前 76 % | — | 规则前先跑感知规约；视频实体身份跨帧持久 | https://arxiv.org/abs/2206.14372 · https://doi.org/10.1145/3729382 · https://arxiv.org/abs/2511.14430 · https://par.nsf.gov/biblio/10596359 |
| **ILAC-G8 / ISO 14253-1 / JCGM 106 / UKAS LAB 48** | 计量合格判定：护带、四态陈述、声明决策规则 | 行业标准 | — | 已采用；报告印决策规则和 U | https://ilac.org/?p=122723 · https://www.evs.ee/en/iso-14253-1-2017 · https://www.bipm.org/documents/20126/2071204/JCGM_106_2012_E.pdf · https://ukas.com/wp-content/uploads/2023/05/LAB-48-Decision-rules-and-statements-of-conformity.pdf |
| **VSI-Bench / SpatialRGPT**（空间后训练基线） | 视频 / 图像空间推理评测与后训练 | 绝对距离接近随机（Gemini-1.5 Pro 31 %）；±25 % 内 35–59 % | 捷径、非厘米级 | 训练买不到厘米级 | https://arxiv.org/abs/2412.14171 · https://arxiv.org/abs/2406.01584 |

## 学习顺序
1. 验证循环先于一切（GuardEn、ARc）。2. 落地 / 对齐单独做（VLTL-Bench、GinSign）。3. 关系算出来并暴露成 API / 工具（GuardEn、RieMind、FuncMapper、TUM）。
4. 条款图带表格和结构（ACCORD、BifrostRAG）。5. 先检索再编译（DriveReg、Lawful-AD）。6. 片段级审阅 + 置信度分流，度量每条分钟数（nl2spec、Han、I-SNACC）。7. 判定只认引擎（MonitorVLM-v2、Know Your Limits）。
