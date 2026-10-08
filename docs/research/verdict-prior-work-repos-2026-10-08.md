# 别人的系统：开源仓库、许可证、可否商用、怎么和我们的组件结合（2026-10-08 查证）

每一格都对照仓库页面 / GitHub API（stars、最后 push、LICENSE 文件）/ PyPI / HF / 论文页核过；"未找到" = 摘要页、全文、HF papers、GitHub 搜索都没有。
全文（含每条核验方式）在附录 F `verdict-prior-work-repos-2026-10-08-survey-F.md`。和 `verdict-prior-work-links-2026-10-08.md`（做了什么 / 效果 / 失败 / 怎么学）配套。

## 1. 真正能拿来用的（有代码、许可证允许）

| 系统 | 仓库 | 许可证 | 发布了什么 | 活跃度 / 安装 | 商用 | 怎么和我们结合 |
|---|---|---|---|---|---|---|
| **Spark-DSG / Hydra / Clio / Khronos**（MIT-SPARK） | https://github.com/MIT-SPARK/Spark-DSG · https://github.com/MIT-SPARK/Hydra · https://github.com/MIT-SPARK/Khronos | Spark-DSG BSD-2；Hydra BSD-2；Clio BSD-2；Khronos BSD-3 | C++ 库 + Python 绑定 + ROS 包 | Spark-DSG ~90★ / 2026-10-07 / `pip install spark-dsg` 1.1.3；Hydra ~1,176★ / 2026-10-08 | 允许 | **场景图容器**（已接：`export_spark_dsg.py`）；JSON 存 / 读；节点属性 + 层间边产 clingo 事实。Hydra / Khronos 只在要自己从 RGB-D 建图（视频阶段）时用 |
| **clingo**（potassco） | https://github.com/potassco/clingo | MIT | 引擎 | ~843★ / 2026-10-08 / `pip install clingo` 5.8.2 | 允许 | **判定内核**（已接）；弱约束 / 多模型可表达不确定度 |
| **AEC3PO 本体**（ACCORD） | https://github.com/Accord-Project/aec3po | 仓库无 LICENSE 文件；TTL 头声明 CC BY 4.0 → 署名可用，建议书面确认 | OWL/TTL 模块：document、statement、rase_statement、table、check_method、evidence、compliance_verification_report… | ~9★ / 2025-04-22 / rdflib 加载 | 署名允许（确认后） | **条款图的 schema**：复用 Document / DocumentSubdivision / Table / DefinitionStatement / CheckStatement 和 RASE 四类 Statement；SHACLCheckMethod → pySHACL，CompositeCheckMethod → 一组 clingo 规则 |
| **CODE-ACCORD 语料 + accord-nlp** | https://github.com/Accord-Project/CODE-ACCORD（Zenodo 10.5281/zenodo.10210022）· https://github.com/Accord-Project/accord-nlp | 语料 Zenodo CC BY 4.0；accord-nlp Apache-2.0；RegulationTransformationTool Apache-2.0；RuleFormalizationTool / RaseLLM **无许可证** | 862 句、4,297 实体、4,329 关系；NER + RE notebook；Java 条文 → RASE 转换器 | accord-nlp ~12★ / 2026-05-29 | 语料 / accord-nlp / RTT 允许；RFT / RaseLLM 不行 | 用语料 + accord-nlp **冷启动条款抽取器**（实体 / 关系）；RTT 做条文 → RASE 结构的参考实现 |
| **TUM ACC-function-generation** | https://github.com/stefan-1992/ACC-function-generation | MIT | CodeAct 式 agent 写 `check_*.py`、评测报告、12 个 IFC 模型 | ~4★ / 2026-05-05 / Python 3.12 `uv sync`（要 GLM 4.7 / OpenRouter key） | 允许 | **检查函数生成循环的模板**：把它的 IFC + 验证器换成我们的场景图 JSON + 合成场景差分，就是推荐架构的第 4–5 级 |
| **nl2spec** | https://github.com/realChrisHahn2/nl2spec | MIT | Flask UI + CLI，提示文件（含 STL） | ~66★ / 2024-02-13 / clone | 允许 | **子翻译审阅 UI 的参考**（片段 ↔ 子公式对照表）；时序条款直接可用（STL 提示） |
| **GinSign**（+ VLTL-Bench） | https://github.com/Dubascudes/GinSign · https://github.com/Dubascudes/VLTL-Bench | GinSign MIT；VLTL-Bench 仓库无许可证 | BERT 落地器、训练 / 评测脚本；基准生成器 | GinSign 0★ / 2026-08-30 / `pip install -e .` | GinSign 允许；VLTL 不清楚 | **对齐 / 落地层**：签名 = 我们的类别 / 区域 / 边；把条款词落到签名当结构化分类 |
| **Logic-LM** | https://github.com/teacherpeterpan/Logic-LLM | MIT | 代码、提示、数据加载 | ~413★ / 2024-06-13 | 允许（求解器各自许可证） | **求解器错误回喂的自精修循环**：把 clingo 的 parse / grounding 错误喂回编译器 |
| **CodeAct** | https://github.com/xingyaoww/code-act | 代码 MIT；Mistral 版模型 Apache-2.0；Llama-2 版受 Llama 2 许可；数据 Apache-2.0 | 代码、2 个模型、7k 多轮数据 | ~1,709★ / 2024-05-23 | 允许（Llama-2 版有条件） | "代码当动作"的 agent 框架参考；不必引入，模式照抄 |
| **RTAMT / Reelay / MoonLight** | https://github.com/nickovic/rtamt · https://github.com/doganulus/reelay · https://github.com/MoonLightSuite/moonlight | BSD-3；MPL-2.0；Apache-2.0 | 监控库 | rtamt ~79★ / 2026-09-19 / `pip install rtamt`；reelay `pip install reelay`；moonlight Java + pip 包装 | 都允许 | **视频阶段**：每帧判定裕量信号 → STL 监控；MoonLight 的 STREL 是唯一原生在空间图上推理的 |
| **pySHACL / Nemo / SWI-Prolog + janus / Soufflé / ZEN** | https://github.com/RDFLib/pySHACL · https://github.com/knowsys/nemo · https://github.com/SWI-Prolog/swipl-devel · https://github.com/souffle-lang/souffle · https://github.com/gorules/zen | Apache-2.0；Apache-2.0 OR MIT；BSD-2；UPL-1.0；MIT | 引擎 | pySHACL 0.40.1 pip；Nemo `pip install nemo-python`；janus-swi 1.5.3；Soufflé 无 pip；ZEN `pip install zen-engine` 2.1.2 | 都允许 | pySHACL 查条款图自身的结构（RDF 时）；Nemo 是 clingo 的 Datalog 替代；ZEN 已是依赖，留给阈值表 / 适用性决策表，不做空间逻辑 |
| **PerceMon / SGSM / SceneFlowLang**（AD 感知监控） | https://github.com/CPS-VIDA/PerceMon · https://github.com/less-lab-uva/SGSM · https://github.com/less-lab-uva/SceneFlowLang | PerceMon BSD-3；SGSM MIT 但依赖 MONA（GPL-2）和 LTLf2DFA（LGPL-3）；SceneFlowLang GPL-3 | C++ 在线监控器；LTLf-over-scene-graph 监控 + 9 条性质 | 小项目，2025–26 有提交 | PerceMon 允许；SGSM 有 GPL 依赖；SceneFlow copyleft | **"感知规约"的参考设计**（对象持久、分类一致）；抄思路不抄代码 |
| **heracles / heracles_agents**（MIT Structured Interfaces） | https://github.com/GoldenZephyr/heracles · https://github.com/GoldenZephyr/heracles_agents | **GPL-3.0** | spark_dsg → Neo4j 加载器 + Cypher 查询工具 + agent 框架 | ~10★ / 2026-04-13 / Docker demo | 允许但 copyleft | 场景图 → 查询层的参考；若分发服务则重写不 vendor |

## 2. 不能商用或不能直接用

| 系统 | 原因 |
|---|---|
| **CompAgent**（AWS） | https://github.com/amazon-science/compagent 是 **CC BY-NC 4.0**，不能商用 |
| **ConstructionSite-10K** 数据 | HF 数据集 **CC BY-NC 4.0** 且 gated；实现仓库无许可证。只能当内部基准参考，不能进产品 |
| **AWS Automated Reasoning checks**（ARc） | 托管服务，无开源组件；按量计费 **$0.17 / 1,000 text units / policy**，文档 ≤ 5 MB / 50,000 字符，仅英文；不吃场景图。可当"规范侧"的对照服务，不是组件 |
| **Solibri** | Java、专有；自定义规则只在 Advanced / Premium 档；无 headless pip 路径。只能当 IFC 真值检查器（TUM 的用法） |
| 无 LICENSE 文件 = 保留所有权利 | ACCORD RuleFormalizationTool、RaseLLM；VLTL-Bench 仓库；ConstructionSite 实现仓库；DriveReg 仓库（空）；monitorvlm_v1；Awesome-DCIST-T4 |
| **MonitorVLM-v2** | 论文指向的 ms-swift fork 里没有权重，数据保密；权重许可证未声明 → 不可验证 |

## 3. 只有论文、没有代码（要从方法描述重实现）

GuardEn + SafetyVisionBench（无下载）、I-SNACC、LLM-FuncMapper、SGR-BIM（只有一个无许可的非官方复现）、Han et al. 2026（数据链接 404）、Zhang & El-Gohary 对齐（只有数据，无许可证说明）、RieMind、P4IR、PropTest、LogiSafetyGen、DriveReg / Lawful-AD / BifrostRAG、Abstract Scene Graphs。

## 4. 对我们的结论

- 能直接进流水线且许可证干净的只有：**Spark-DSG、clingo、AEC3PO schema、CODE-ACCORD + accord-nlp、TUM 函数生成循环、nl2spec、GinSign、Logic-LM 模式、RTAMT / Reelay / MoonLight、pySHACL / Nemo / ZEN**。这些刚好覆盖推荐架构的每一级：条款图 schema（AEC3PO）→ 抽取冷启动（CODE-ACCORD）→ 对齐（GinSign）→ 检查生成 + 验证循环（TUM + Logic-LM）→ 审阅 UI（nl2spec）→ 引擎（clingo）→ 视频（RTAMT / MoonLight）。
- 效果最好的几个（GuardEn、RieMind、SGR-BIM、P4IR）都没代码，只能抄方法；它们的价值是**消融数字**告诉我们什么重要，不是可用组件。
- 两个 NC（CompAgent、ConstructionSite-10K）和三个 GPL（heracles、SceneFlowLang、SGSM 的依赖）要避开或只做内部参考。
