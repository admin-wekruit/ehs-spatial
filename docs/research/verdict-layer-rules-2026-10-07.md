# 报告的判定层（verdict layer）：现状审计 + 规则语言调研（2026-10-07）

问题：要把安全规范（对物体的要求等）接进报告的判定层。现在是纯 JSON 还是有 3D 图？拓扑规则 / 形状规则 / 语义规则各是什么状态？
OWL / SWRL / SPARQL / SHACL / STL 这些能不能用？现在有没有 harness，还是直接丢给 LLM？

先说结论（细节在后面各节）：

1. **判定不是丢给 LLM 的。** 现有判定层是"LLM 只做一次离线翻译（prose → 封闭谓词 JSON，可审阅、可拒绝），判定由确定性几何代码出"，
   并且平台引擎的 docstring 明文写着 never asks a VLM for a verdict。但它的规则语言只有 **7 个 2.5D 几何谓词**，覆盖"形状规则"的一小部分，
   拓扑规则只有 `not_inside`（平面多边形重叠），语义规则只有"标签字符串属于某集合"。
2. **表示是扁平 JSON，没有场景图。** 三套 JSON 并存（平台 scene document、研究用 SceneMap、10 月的 measurement layer），
   对象之间的关系（相邻、包围、之间、开口、可达）都不显式存在，每条规则临场用 shapely 算一次两两距离。trimesh 的 `scene.graph` 只是变换树。
3. **harness 有，但窄。** 63 个单元测试 + 一个合成场景基准（eval_pack：围栏 / 机器人 / 梯子按设计距离摆放）+ 一次 13 条 OSHA 条款的
   编译器考试。覆盖的只有"间距 / 高度"这一族规则；没有真实场景的金标判定集，没有拓扑 / 语义规则的测试，没有蜕变测试，
   不确定度只是一条固定宽度的带。
4. **今天的 090 / 030 报告没有任何机器判定。** `scripts/workcell_policy_evidence.py` 把 10 月的报告接到引擎上时，引擎按设计弃权：
   适用性没有审核员断言、尺度状态是 `accepted_3d_reference` 而引擎只认 `operator_anchored`、只量了下沿没有整高、机器人工作区没有建模。
   它输出的是"每条规则还缺什么证据"的台账，标题就叫 never a safety verdict。

---

## A. 现状审计（按代码，2026-10-07 的 ehs-spatial `main`）

### A1. 三种 JSON 表示，没有图

| 表示 | 在哪 | 一个对象长什么样 | 对象间关系 |
|---|---|---|---|
| 平台 scene document（schemaVersion 1/2） | `ehs_spatial/platform/contracts.py`、`policy_engine.evaluate_document` 读的 `scene` | `entities[]`：`label`、`measurements.footprint`（米制 2D 多边形 + `uncertaintyM` + `sourceRefs` + `source`∈{observed_measurement, planned_geometry, manual_assertion}）、`measurements.height`、`observationRefs`；`coordinateFrames[].scale.status`；`annotations[]`（`policy_applicability`、`manual_evidence`） | 无；引擎临场把 footprint 喂给 shapely |
| 研究用 SceneMap（8 月） | `ehs_spatial/contracts.py: SceneMap / Entity3D / SpatialFact` | `Entity3D`：`footprint_xy`、`height_m`、`centroid_xyz`、`orientation_deg`、`tilt_deg`、`overhang_m`、`evidence_frame_ids` | `SpatialFact(predicate, subject_id, object_id, value, unit)`——判定时产出的事实，不是先验的图 |
| measurement layer（10 月工位报告） | `web/src/measurement-layer.ts`，Pages 上的 `measurement-layer/<id>.json` | `boxes[id]`：地面上的有向盒（`axes`、`faceNormals`、`sizeM` L/W/H 各带 `sigmaCm` 和置信度、`bottomM`/`topM`、`floorContact`、每个面被哪些照片看到、`highlightReasons`）；`facts[id]`、`confidence[id]`、`pipelines[id]`、`scale`、`ground`、`models` | 无 |

也就是：10 月的报告有最丰富的几何（盒、面、离地距离、每维不确定度），但它和判定层之间只有 `workcell_policy_evidence.py` 这座桥，
而桥把盒退化成 footprint 多边形 + 高度喂给 8 月的 7 谓词引擎。

### A2. 规则语言：7 个谓词，全是 2.5D

`ehs_spatial/contracts.py: Predicate`（封闭词表，编译器表达不了必须拒绝）：

| 谓词 | 含义 | 规则类 | 评价方式（`ehs_spatial/policy.py`） |
|---|---|---|---|
| `min_separation` / `max_separation` | 主体与客体 footprint 的最小间隙 ≥ / ≤ 阈值 | 形状（数值） | shapely `distance`，误差带内 → NEEDS_REVIEW |
| `keep_clear` | 同上的"保持净空"写法 | 形状 | 同上 |
| `not_inside` | 主体 footprint 与客体区域重叠面积 = 0 | 拓扑（仅平面重叠） | shapely 相交面积 |
| `max_height` / `min_height` | 主体整高 ≤ / ≥ 阈值 | 形状 | `height_m`，带 |
| `max_tilt` | 主体倾角 ≤ 阈值（只认 `tilt_reference='physical_axis'`） | 形状 | `tilt_deg` |

`PolicySpec` 还带 `subject_labels` / `object_labels`（字符串标签集合 = 唯一的"语义"）、`threshold`、`unit`∈{m, m2, deg}、`severity`、`unsupported_reason`。
没有：包围 / 开口 / 之间 / 可达 / 视线这类拓扑；没有 3D 关系（只有 footprint + 高度）；没有属性语义（光幕分辨率、急停颜色、联锁有无）；没有时间。

### A3. 判定怎么出：确定性几何，LLM 只翻译

- **编译**（`scripts/policy_compile.py`）：prose 一行一条 → gemini-3.5-flash → `PolicySpec` JSON，按内容/词表/模型缓存，离线，可 diff；
  表达不了的必须填 `unsupported_reason`，永不近似。编译产物是安全工程师签字的对象。
- **判定**（`ehs_spatial/policy.py: evaluate_policy`）：PASS / FAIL / NEEDS_REVIEW / INSUFFICIENT_EVIDENCE。证据门：主体至少 2 帧看见
  （`MIN_EVIDENCE_FRAMES`）；误差带：多视角 0.20 m、单目 0.35 m（`rules.py`，来自激光 GT 评测的 MAE），平台路径改为各实体 `uncertaintyM` 之和；
  落在阈值 ± 带内不选边，报 NEEDS_REVIEW。
- **平台引擎**（`ehs_spatial/platform/policy_engine.py`，`ENGINE_VERSION = policy-evidence-v1`）：政策是版本化文档（表 `policy_sources / policies /
  policy_revisions / policy_evaluations / policy_reviews / policy_evidence_requests`）。每条政策 = 来源条文 + 一个 GoRules ZEN 决策图（JDM，依赖 `zen-engine`）。
  **ZEN 只收适用性事实和标签列表**，输出 `check`：`manual`（要人工 `manual_evidence` 注释，带 `passed`）或 `geometry`（上面那 7 谓词之一）。
  数值判断全在 Python，"编辑决策表不可能造出第二个几何评价器"。适用性只能由审核员带证据断言（`policy_applicability` 注释），否则引擎弃权。
  模板 5 条 OSHA 1910（3 条 manual、2 条 `not_inside`）。
- **VLM 的位置**：只在"为什么不合格"的解释里（`docs/archive/demos/2026-08-25-poc-policy-vlm.md`：Gemini 引用 fact id 解释，数字本地渲染），
  以及方向判定 / 补测定位等感知环节。判定本身从不经过模型权重。

### A4. 今天的 090 / 030 报告：引擎按设计弃权

`scripts/workcell_policy_evidence.py`（"never a safety verdict"）把一份报告 revision 喂给引擎，对 `docs/policies/compiled_v2/` 的 3 条规则
（围栏 ≥ 1.8 m 高；急停离机器人工作区 ≤ 3 m；工位开口 2 m 内要有安全传感器，第三条编译为诚实拒绝）输出台账，每条列"还缺什么"：

- `reviewer_applicability_confirmation`：没有审核员断言这条规则适用于这个工位；
- `operator_anchored_metric_scale`：报告尺度状态是 `accepted_3d_reference`，引擎只认 `operator_anchored`；
- `subject_full_height`：这个 revision 只量了下沿（lower_edge），没有顶高事实；
- `reference_region`：机器人工作区包络没有建模，一个静态机器人模型不是危险区。

结论行是固定的："No safety conclusion … photos cannot establish stopping performance, interlocks or detection-zone validity."

### A5. 现有 harness（不是没有，是窄）

| 层 | 在哪 | 覆盖什么 | 没覆盖什么 |
|---|---|---|---|
| 单元测试 | `tests/test_policy.py`(32) `test_rules.py`(11) `test_rule_semantics.py`(7) `test_policy_compile.py`(3) `tests/check_platform_policy_geometry.py` | 7 谓词的边界三元组（阈值 ±2 cm，手搭 `Entity3D`）、误差带、证据门、编译缓存键、平台几何分支的证据契约 | 拓扑 / 语义规则（不存在）、真实场景 |
| 合成场景基准 | `ehs_spatial/eval_pack.py`（13 测试 + 1 live） | 程序生成围栏 / 机器人 / 梯子 / 平台，按设计距离摆、正常 / 遮挡相机，跑全流程比 `expected_status` 和 `expected_distance_m` | 只有间距一族；不含光幕 / 急停 / 开口 |
| 编译器考试 | `docs/archive/reviews/2026-08-25-osha-compiler-exam.md`，`docs/policies/osha1910.md` + `tests/fixtures/oshacorpus/corpus.json` | 13 条 29 CFR 原文：该拒的 10/10 拒了，0 条错编译，该编的 1/3 编对、2 条过度拒绝 | 只跑过一次（live），没有进 CI |
| 推断政策 | `docs/archive/phase2/INFERENCE-POLICY.md` | 视频报告哪些几何可以参与规则（只有"观察到"的；结构推断的最多 需复核） | — |

LLM-as-judge 不存在于判定路径，所以也没有对应的评测；缺的是：真实工位的逐规则金标判定集、拓扑 / 语义规则的测试、蜕变测试
（整体平移 / 旋转不改判定，挪 1 cm 过阈值必翻），以及把每维 `sigmaCm` 真正传播到判定（现在是固定带）。

---

## B. 规则语言 / 表示方法调研（浓缩版；全文和每条链接在附录 A `verdict-layer-rules-2026-10-07-survey-A-languages.md`，由调研 agent 产出，链接未逐一复核）

前提（所有方案共用）：**几何在 Python 里算，不在逻辑里算。** 相邻 / 包围 / 离地间隙 / 两两距离 / 之间 / 视线这些关系用 shapely（纯 2D，GEOS 谓词丢 Z）、
trimesh（3D 包含、布尔）、Open3D（有向盒）从盒和网格算一次，作为事实喂给规则引擎；规则语言只需要比较、算术、否定、递归、聚合。

| 方案 | 是什么 | 对我们的判断 |
|---|---|---|
| **RDF + OWL 2 + SWRL + SPARQL** | 描述逻辑，开放世界假设；SWRL 加 Horn 规则和算术内建；推理机 HermiT 停更、Pellet 续作 Openllet（Java）、ELK 只支持 EL；Python 侧 owlready2 0.51 要 Java | **不用作规则引擎。** 开放世界下"没量到间隙"只是未知，永远不是违规；"没有一段边界未被覆盖"这种包围判断表达不了；fail 要靠否定，OWL 没有，最后还是写 SPARQL。最多当交换词表 |
| **SHACL（Core / SHACL-SPARQL / 规则）** | W3C 推荐（2017），闭世界校验，输出结构化 validation report（focusNode、path、value、message、三级 severity）；pySHACL 0.40.1 纯 Python；SHACL 1.2 仍是 Working Draft（2026-08 Core WD、Rules WD） | **标准化方案里最合适的**，特别是客户要 RDF 交付时；算术走 SPARQL `BIND`，聚合走 `GROUP BY`；拓扑递归只有属性路径，`adjFree+` 这种边要先物化；一个 50 物体的场景要 IRI/Turtle 的管道成本。不要依赖 1.2 独有特性 |
| **空间演算 RCC-8 / DE-9IM / GeoSPARQL 1.1** | 定性拓扑关系的标准词表；GeoSPARQL 暴露 `geof:rcc8*`、`geof:sf*`、`geof:relate`，但**没有 3D 语义**，三元组库基本把 3D 压扁 | 词表可借（命名关系），存储不用；平面围栏能判，离地间隙和高度看不见。SparQ / GQR / CLP(QS) 这类定性求解器是在"关系未知"时做组合推理，我们有米制几何，关系是算出来的，跳过 |
| **ASP / clingo 5.8**（MIT，pip） | 回答集编程：失败即否定、递归、聚合（#count/#sum/#min/#max）、整数算术，稳定的 Python API；clingo-dl / clingcon / clingo-lpx 1.3 加差分 / 整数线性 / 有理线性约束；解释：xclingo、xASP2，或让判定原子自带证据 | **推荐做唯一判定内核。** 闭世界，包围 / 可达靠递归天然表达，三值判定（fail / needs_meas / pass）就是否定即失败的副产品；单位用整数毫米、毫秒，基本用不到 lpx。四类示例规则见下 |
| Datalog（Soufflé、Nemo）、Prolog（SWI + janus-swi） | 同等表达力；Soufflé 有 proof tree（`explain`）但是外部二进制；pyDatalog 已死；Prolog 浮点原生，解释要自己写元解释器 | clingo 的替代；团队熟 Prolog 也行 |
| 生产规则 / 策略引擎（OPA/Rego、CEL、JSON-Logic、Drools） | 表达式和策略语言：算术、聚合、trace 都有，**没有递归**；durable_rules 已死；cel-expr-python 2026-03 开源 | 只适合"阈值表当数据"的数值规则（a、b 类）；拓扑（c、d 类）做不了 |
| **时序 / 信号逻辑 LTL / MTL / STL**（STL = Signal Temporal Logic，不是网格格式） | 对实值信号的时序规约，鲁棒度语义 = 带符号的裕量（如 ρ = S 实测 − S 要求）；工具 RTAMT（Python，离线 / 在线，ROS 桥）、MoonLight（STREL：带图上的 reach/escape/somewhere/everywhere）、Reelay、Breach / S-TaLiRo（MATLAB）、PSY-TaLiRo；自动驾驶里"场景图 + 时序逻辑"是活跃方向（STPL、Scene Flow Specifications、Abstract Scene Graphs） | **视频阶段的路径。** STL 没有对象量词：每帧跑静态判定，把布尔 / 裕量信号喂给 RTAMT，例如 `always[0,T](gap_fence3 <= 180)`、`always(door_open -> eventually[0,2] robot_stopped)`；鲁棒度就是证据裕量 |
| **3D 场景图**（Armeni 2019、Hydra / Hydra-Multi / Clio 的 Spark-DSG、ConceptGraphs、HOV-SG、Open3DSG；时序的 Khronos、Aion） | 分层图：places / rooms / objects / agents，节点带包围盒、位置、语义，类型化边；Spark-DSG 有 pip 绑定、BSD-2、JSON 序列化；查询实践是 NetworkX 进程内或 Cypher/Neo4j（2025 MIT 研究：图数据库查询比把图塞进 LLM 上下文可扩展得多） | 用其"对象 + 显式关系边"的形式，不必引入其库；合规用途极少（PPE 场景图比对、SafeSceneReason 的"安全场景图 + 声明式规则"），没有人做过米制的防护 / 光幕规则——这是我们的空位 |
| **BIM/AEC 规范自动检查**（最接近的已解决类比） | Eastman 2009 四阶段：规则解释 → 模型准备 → 规则执行 → 报告；RASE 把法规文本标成 Requirement / Applicability / Selection / Exception；LegalRuleML（OASIS，重）；Solibri（闭源模板）；语义网分支用 SHACL over ifcOWL/BOT；EU ACCORD 项目（2022–2025）用 RASE + SHACL；buildingSMART IDS 1.0 做属性约束（无跨对象算术、无拓扑） | 我们的流水线就是 Eastman 四阶段；RASE 的四操作符值得直接用在条文标注上；IDS 对应"语义规则"那一层 |
| **神经符号模式：LLM 编译、引擎判定** | Logic-LM、LLM→ASP、LLASP、nl2spec（子公式回映自然语言让人逐条审）、NL2LTL、NL2TL、NL2SHACL-Bench（LLM 生成的 SHACL 语法总对、复杂逻辑 / 路径语义常错）、PolicyKG；失败模式：优先级错、单位 / 常数错、丢异常（RASE 的 E）、编造词表外谓词 | 和我们 8 月的设计一致（封闭词表 + 拒绝）。可加的验证：round-trip（形式→NL→形式等价检查，统计上把等价率从 45–61 % 提到 83–85 %）、让 LLM 为每条规则生成一正一负的合成场景并真跑、对手写参考规则做差分测试 |

四类规则在 clingo 里长什么样（事实由 Python 几何层产出，单位毫米 / 毫秒）：

```prolog
% (a) 光幕安全距离 S >= K*T + C（ISO 13855；S<=500 时 K=2000 的分支另写一条）
req_s(LC,S) :- light_curtain(LC), res_mm(LC,D), stop_ms(LC,T), D>=14, D<=40, S = 1600*T/1000 + 8*(D-14).
fail(lc_dist,LC,have(H),need(N)) :- req_s(LC,N), dist_hazard_mm(LC,H), H < N.
pass(lc_dist,LC) :- req_s(LC,N), dist_hazard_mm(LC,H), H >= N.
needs_meas(lc_dist,LC) :- light_curtain(LC), not dist_hazard_mm(LC,_).
% (b) 围栏离地间隙；ISO 13857 的开口-距离表作为事实 req_sr(Emin,Emax,Sr)
fail(fence_gap,F,gap(G)) :- fence(F), floor_gap_mm(F,G), G > 180.
needs_meas(fence_gap,F) :- fence(F), floor_gap_mm(F,_), conf(F,C), C < 70.
fail(reach,O,have(S),need(R)) :- opening(O,E), hazard_dist_mm(O,S), req_sr(Lo,Hi,R), E>Lo, E<=Hi, S<R.
% (c) 包围：平面占用格；外部格经未被防护物覆盖的格能走到危险区 = 未包围
reach(C) :- outside(C).  reach(C2) :- reach(C1), adj(C1,C2), not blocked(C2).
fail(enclosure,Z,via(C)) :- hazard_zone(Z), cell_of(Z,C), reach(C).
pass(enclosure,Z) :- hazard_zone(Z), not fail(enclosure,Z,_).
% (d) 急停可达
reachable(P,E) :- operator_pos(P), estop(E), dist_mm(P,E,D), D<=600, height_mm(E,H), H>=600, H<=1700, not occluded(P,E).
fail(estop,P) :- operator_pos(P), not reachable(P,_).  pass(estop,P,E) :- reachable(P,E).
```

对照表（调研 agent 的评分，附录 A 原表）：

| 方案 | 拓扑 (c,d) | 数值 (a,b) | 语义 | 解释 / 证据 | Python 成熟度 | 对 Panoptes |
|---|---|---|---|---|---|---|
| 扁平 JSON + 临场 Python | 每条规则重算 | 好（浮点） | 好 | 手搭 | 原生 | 现状；规则多了不可维护 |
| 类型化场景图（NetworkX）+ Python 检查 | 好（图算法） | 好 | 好 | 手搭 | 原生 | 好；但没有声明式规则 |
| RDF + OWL 2 + SWRL | 差（开放世界、无失败即否定） | 弱 | 强 | 仅 Java 有 justification | owlready2 要 Java | 避免 |
| RDF + SHACL / SHACL-SPARQL（pySHACL 0.40） | 中（路径；边要物化） | 好（BIND） | 好 | 极好（validation report） | 好，纯 Python | 导出 / 报告层很强 |
| GeoSPARQL 1.1 | 只有 2D RCC-8 / DE-9IM | — | — | 查询结果 | 库少、无 3D | 内部只用 shapely |
| **ASP / clingo 5.8（+lpx）** | 极好 | 整数；有理数用 lpx | 好 | 好（证据原子、xclingo） | 极好（pip，MIT） | **推荐内核** |
| Datalog Soufflé / Nemo | 极好 | 好 | 好 | proof tree | 外部二进制 / API 年轻 | clingo 的替代 |
| Prolog + janus-swi | 极好 | 极好 | 好 | 要元解释器 | 好 | 团队熟则可 |
| OPA / CEL / JSON-Logic | 弱（无递归） | 好 | 好 | OPA trace | 子进程 / 新库 | 只做阈值表 |
| STL（RTAMT / MoonLight） | 仅 STREL | 极好（鲁棒度裕量） | — | 鲁棒度值 | RTAMT pip | 视频路径 |
| 3DSG + Cypher / Neo4j | 好 | 好 | 好 | 查询结果 | Neo4j 驱动 | 以后多场景记忆 |

---

## C. 安全规范：哪些能从照片场景判、数字是多少（浓缩版；全文、每个数字的出处链接和"unverified"标记在附录 B `verdict-layer-rules-2026-10-07-survey-B-specs-harness.md`）

版本提醒（附录 B）：ISO 13855 在 2024-11 出了第 3 版（公式改写成 S = K·T + DDS + Z），下面的数字是厂商仍在印的 2010 版；ISO 13857 在修订中（ISO/CD 13857）；
ISO 10218-1/-2:2025 已取代 2011 版（TS 15066 并入、必需安全功能 >20 项、加网络安全和机器人分类）。ISO 正文付费，调研的数字来自 SICK / Pilz / Omron / Reer / Troax / ABB 等厂商复述，
附录 B 把不能确认的条款号和数值标了 unverified——**进规则包前要对照购买的正文核一遍**。

| # | 要求 | 标准 / 条款 | 公式 / 阈值 | 照片场景里看得到 | 必须人 / 数据表给 | 规则类 |
|---|---|---|---|---|---|---|
| 1 | 光幕 / ESPE 最小距离（垂直接近） | ISO 13855（2010 版数字） | S = K·T + C；S ≤ 500 时 K = 2000 mm/s，否则 1600；分辨率 d ≤ 40 mm 时 C = 8(d−14)，d > 40 时 C = 850；S 下限 100（K=2000）/ 500（K=1600） | 光幕到危险边的水平距离、朝向、高度 / 范围 | **T**（机器停止时间 + ESPE 响应 + 接口）、**d**（有时在标签上）、哪条边是危险 | 几何（缺 T 则 needs_measurement） |
| 2 | 水平 / 平行场（扫描仪、地面平行光幕） | ISO 13855（2010） | C = 1200 − 0.4·H 且 ≥ 850；H ≤ 1000；H > 300 要评估钻入 | 场离地高度 H、到危险距离 | T、d、扫描仪配置的场大小 | 几何 |
| 3 | 多光束门禁 | ISO 13855 | 4 束 300/600/900/1200；3 束 300/700/1100；2 束 400/900；1 束 750 mm；C = 850 → S = 1600·T + 850 | 光束数、各束离地高度（±1–2 cm 对 100 mm 台阶足够） | T | 几何 + 语义（束数） |
| 4 | 越过围栏够到 | ISO 13857:2019 表 2（高风险；表 1 低风险） | 查表(危险高 a, 结构高 b) → 最小水平距离 c。例：b=1400：a=2000→1100，a=1000→1000；b=1800：a=1400→800；b=2000：a=2600→500；b=2200：a=2600→400；b=2500：a=2600→100；b=2700→0。结构 < 1400 不应单独使用 | 围栏顶高 b、危险区高 a（机器人 / 工具最高点近似）、水平距离 c | 用哪张表（风险评估）、真正的危险区范围（机器人伸展，不是照片里的姿态） | 几何 |
| 5 | 穿过开口（网孔 / 槽） | ISO 13857:2019 表 4 | 开口 e → 最小距离 sr（槽/方/圆）：≤4：2/2/2；4–6：10/5/5；6–8：20/15/5；8–10：80/25/20；10–12：100/80/80；12–20：120；20–30：850/120/120；30–40：850/200/120；40–120：850。ABB 用法：40×40 网 → 200 mm | 围栏到危险的距离；网孔尺寸只有分辨得出时（1–2 cm 精度常常分不出） | 网孔节距（目录） | 几何 + 语义 |
| 6 | 绕过边缘（防护边与相邻结构的缝） | ISO 13857 表 3 | 肩限 ≥ 850；肘 ≥ 550；腕 ≥ 230；指节 ≥ 130 mm | 防护边与相邻结构的缝、到危险距离 | 身体受限情形的选择 | 拓扑 + 几何 |
| 7 | 围栏离地间隙 / 下肢开口 | ISO 13857:2019 4.4、表 7 | 地面缝 ≤ 180 mm（槽）/ ≤ 240（方 / 圆），否则整身可入；表 7：槽 35–60 → 180，60–80 → 650，80–95 → 1100，95–180 → 1100 | 围栏底边离地高度（我们已有的 floor-contact / bottomM）、到危险距离 | — | 几何 |
| 8 | 防护装置构造 | ISO 14120:2015（5.3.9 固定防护只能用工具拆、5.18 攀爬、附录 C 115 J 冲击） | 14120 本身无高度数字，高度来自 13857 | 防护存在、连续（包围拓扑）、立柱、明显踏脚点 | 紧固件、冲击等级 | 拓扑 + 语义 |
| 9 | 周界防护 vs 机器人受限空间 | ISO 10218-2（2011 5.4.3；2025 版定义 safeguarded / maximum / restricted space） | 围栏到受限空间外边界的距离要满足 13857；网状围栏不可当限位装置 | 围栏位置、机器人底座位置 | 受限空间（控制器软限位 / SafeMove 配置）、机器人臂展（数据表） | 几何（缺配置） |
| 10 | 机器人与固定结构的夹困间隙 | ISO 10218-2:2011 / ANSI-RIA R15.06-2012（**条款号 unverified**） | ≥ 500 mm（20 in）；旧版 18 in | 机器人盒 ↔ 固定物盒最小距离 | 操作 / 受限空间（随程序） | 几何 + 拓扑 |
| 11 | 最小挤压间隙 | ISO 13854:2017 表 1 | 身体 500、头 300、腿 180（预览里已核）；脚 120、脚趾 50、臂 120、手 / 腕 / 拳 100、手指 25（二手来源，手指值 unverified）；只管挤压，不管撞击 / 剪切；多部位可入取最大 | 运动部件盒与固定物的缝 | 哪个身体部位能进入 | 几何 |
| 12 | 急停装置设计 | ISO 13850:2015 4.3；IEC 60204-1；OSHA 1910.144(a)(1)(iii) 红色 | 红蘑菇头、黄底、自锁、手动复位；最低 PL c | 颜色、形状、存在 | 自锁 / 复位、停止类别、接线、PL | 语义 |
| 13 | 急停位置 / 可达 | IEC 60204-1 10.7（每个操作站 + 必要处，readily accessible）；10.1.2 手动操作件离地 ≥ 0.6 m；ISO 13850 4.1.1.1 | 高度 ≥ 0.6 m；上限没有单一条款（常说的 0.6–1.7 m **追不到一条条款，unverified**；NEC 404.8(A) 2.0 m 是开关不是急停） | 急停离地高度、每个控制面板 / HMI 可达范围内有一个、接近路径无阻挡 | 哪些位置是操作站 | 语义 + 拓扑 + 几何 |
| 14 | 地面标线 / 通道 | 29 CFR 1910.176(a)、1910.22、1910.144(a)(3)（黄 = 注意）、ANSI Z535.1；1910.36(g)(2) 出口通道 ≥ 28 in；NFPA 101 36 in（既有）/ 44 in（新建）；叉车通道比最宽设备 / 载荷宽 ≥ 3 ft | 标线有无 / 颜色、盒之间的通道宽度、通道内障碍 | 是否出口通道、车宽 | 语义 + 几何 |
| 15 | AGV / AMR 路径（略） | ISO 3691-4 | 路径两侧到固定结构 0.5 m、高 2.1 m 包络；ESPE 场边缘 180 mm 内检测 | 路径（地面标线）↔ 结构距离 | 路径定义 | 几何 |
| 16 | 照片判不了的 | ISO/TS 15066（力 / 压强表）、ISO 13849-1（PL）、ANSI B11.19、IEC 62046、ISO 11161 | — | 只有装置存在与否 | 全部 | — |

**永远不在照片里的输入**：停止时间 T、ESPE 分辨率 d 和响应时间、机器人受限空间 / 软限位、停止类别、PL、紧固件、自锁、联锁接线、运行模式、低 / 高风险分类。
设计后果：13855 一族规则在没有 T、d 时一律出 `needs_measurement`；13857 / 13854 一族纯几何就能出 pass / fail。

**专业上判定怎么写**（附录 B PART 2）：条款 → 适用性（ISO 12100 危险识别决定）→ 测量量 + 方法 → 阈值（表 / 公式）→ 证据 → 判定。
判定词表参考 TTCN-3 的 `none < pass < inconc < fail < error` 格（聚合 = 子判定取最大）；计量学的 ILAC-G8:09/2019 非二元判定（pass / conditional pass / conditional fail / fail，护带）
和 ISO/IEC 17025 7.8.6（必须声明决策规则）。审核员记录：条款、测量值、仪器 / 方法、不确定度、照片引用、签字的验证表；ISO 10218-2:2025 要求集成商按计算验证停止距离。

**现成工具**：没有一个工具拿**拍下来的**布局去对 13857 / 13854（SICK Safety Designer 在测得轮廓 / 导入图上配场；Pilz Safety Distance Calculator 按 T、d、PLr 算 13855；
Pilz PAScal / 西门子 TIA Safety Evaluation / 罗克韦尔 Safety Automation Builder 算 PL/SIL、在导入图上画区；ABB RobotStudio 仿真制动距离定 SafeMove 区）。
3D / 数字孪生做法主要是专利（Veo 的 3D 感知功能安全孪生、扫描定安全传感器位置、"检查机器人安全区的方法"）；学术上有激光测距 + 数字孪生验证安全距离（2022）、
在线距离跟踪（2023）。建筑 BIM 是成熟类比（Solibri 坠落防护规则、Scan-to-BIM 防火分区检查、TUM Scene2Compliance）。**没有找到从照片 / 点云出 ISO 13857 / 13854 判定的同行评审论文**（缺席本身 unverified）。

---

## D. 建议：怎么做判定层

### D1. 表示：显式关系层的类型化场景图，不是 RDF 优先，也不再是扁平 JSON

保留今天的对象列表（measurement layer 的盒 / 面 / 每维 σ / 置信度 / 可见照片），**加一层由几何算出来的显式关系**，存成 NetworkX 图 + 可 JSON 序列化：

- 节点：object（现有盒 + 属性槽：class、subclass、attributes 如光幕分辨率 d、急停颜色，来源标明 observed / datasheet / reviewer）、
  zone（危险区、操作位、通道；今天没有，见 D3）、cell（平面占用格，给包围 / 可达用）、camera。
- 边（全部带数值和不确定度、带产出它的方法和照片）：`distance_mm`、`floor_gap_mm`、`adjacent`、`contains`、`between`、`line_of_sight`、`covers(cell)`、`seen_in(camera)`。
- 关系命名借 BOT（containsZone / adjacentZone / hasElement）和 IndoorGML（cell + connectivity）的叫法，不引入它们的库。
- 需要 RDF 交付时，同一张图 50 行导出到 rdflib，SHACL 报告作为证据格式。

这解决的是 A1 的问题：三套 JSON 各算各的、关系临场重算、规则多了不可维护。

### D2. 规则引擎：clingo 做唯一判定内核；现有 7 谓词降级为"阈值表"规则族

- 一个模块从场景图产事实（整数毫米 / 毫秒），每个规则族一个 `.lp`，一个函数把回答集原子映射成 `Verdict` 记录。
- 现有 `PolicySpec`（min/max_separation、keep_clear、not_inside、min/max_height、max_tilt）**不丢**：它们就是"阈值当数据"的规则族，
  编译成 clingo 事实 `rule(id, predicate, subject_labels, object_labels, threshold_mm)` + 一条通用规则；8 月的边界三元组测试原样迁移。
- 适用性仍走审核员断言（ZEN 决策图不必动）；LLM 仍只做 prose → 规则的离线编译，封闭词表 + 拒绝机制保留，词表从 7 个谓词扩到
  "事实谓词 + 规则模板"（见 C 节的规则包），并加 B 节说的三种编译验证（round-trip、一正一负合成场景、对手写规则差分）。
- 三值判定由失败即否定自然得出：`fail`、`needs_meas`（事实缺失或置信度 / σ 过阈值）、`pass`；优先级 fail > needs_meas > pass 一条规则。
- 解释：判定原子自带 `have(...)/need(...)/via(...)`，报告直接渲染；需要推导树再上 xclingo。

### D3. 先解决今天让引擎弃权的四件事（A4），否则换什么语言都没有判定

| 缺口 | 现状 | 要做的 |
|---|---|---|
| 尺度状态 | 报告是 `accepted_3d_reference`（急停参照物定尺度，±% 已算出），引擎只认 `operator_anchored` | 引擎接受"参照物定尺度 + uncertaintyRelative"，把尺度不确定度并入判定裕量（D4） |
| 整高 | 只量了下沿 / 底边 | measurement layer 的盒已经有 `topM`、`sizeM[H]` 和 σ；桥接时直接用盒，不再退化成 footprint + height |
| 危险区 | 机器人工作区没有建模；"一个静态机器人模型不是危险区" | 新节点类型 zone：机器人工作区 = 机器人包络 + 工艺给的伸展半径（数据表输入，不从照片推）；通道 / 操作位 = 地面标线或审核员画 |
| 适用性 | 没有审核员断言 | 保留；但引擎在"适用性未知"时也应产出 **条件判定**（"若适用：fail，差 120 mm"），台账里不再只有"缺什么" |

### D4. 不确定度进判定（替换今天固定的 0.20 / 0.35 m 带）

每条数值规则比较的是 `measured ± u` 对 `threshold`：u 来自盒的 `sigmaCm`（每维）+ 尺度 `uncertaintyRelative × 距离` + 面被几张照片看到的置信度。
决策规则按计量学的护带（guard band）写：`measured − k·u ≥ threshold` → pass；`measured + k·u < threshold` → fail；中间 → needs_meas，
k 按客户要的置信水平定（C 节附录 B 给 ISO 14253-1 / ILAC-G8 的做法）。这正是 `AssessmentStatus.NEEDS_REVIEW` 今天想表达但用固定带近似的东西。

### D5. 视频阶段

每帧跑同一套静态判定，把布尔 / 裕量信号喂给 RTAMT 的 STL：`always[0,T](gap_fence3 <= 180)`、`always(door_open -> eventually[0,2] robot_stopped)`；
鲁棒度 = 证据裕量。需要图上的时空算子（reach / escape）再看 MoonLight 的 STREL。

### D6. 第一版规则包：12 条照片可判的要求（附录 B 的清单，数字见 C 节表）

1. 围栏顶高 ≥ 1400 mm（13857 表 2 注）——几何
2. 越过够到查表（13857 表 2）：a = 机器人最高点，b = 围栏高，c = 实测——几何
3. 防护离地缝 ≤ 180 mm（13857 4.4）——几何；**我们今天的 bottomM / floorContact 直接能判**
4. 包围闭合：周界防护绕机器人成闭环，除联锁门 / ESPE 外没有 > 180 mm 槽 / 240 mm 的开口（13857 表 7）——拓扑
5. 网孔距离：已知网孔节距时围栏 ≥ sr(e)（13857 表 4；40×40 → 200 mm）——几何 + 语义
6. 防护边缘绕过：开放边到危险 ≥ 850 mm（13857 表 3）——拓扑 + 几何
7. 机器人包络与固定物的挤压缝 ≥ 500 身体 / 300 头 / 180 腿 / 120 脚 / 100 手 / 25 指（13854 表 1；低行二手来源）——几何
8. 机器人 ↔ 结构夹困间隙 ≥ 500 mm（10218-2 / R15.06，条款待核）——几何
9. 包围的每个开口处有 ESPE；输出 S_required = K·T + C，T、d 作未知量 → 未提供则 needs_measurement（13855）——拓扑 → 几何
10. 多光束高度 ∈ {300/600/900/1200, 300/700/1100, 400/900} ± 50 mm（13855）——几何
11. 急停：红蘑菇黄底；每个控制面板 / HMI 可达范围内 ≥ 1 个；离地 ≥ 0.6 m（60204-1 10.1.2）；无阻挡；1910.144——语义 + 拓扑
12. 通道：有标线、无阻挡、≥ 711 mm（出口）或 ≥ 914 mm（NFPA 既有），黄 / 黄黑标线——语义 + 几何
（可选 13–14：AGV 路径 0.5 m × 2.1 m（3691-4）；防护上无攀爬点（14120 5.18）。）

### D7. 判定对象（Verdict）字段（附录 B 的 schema，和 D4 的护带规则配套）

```
Verdict {
  rule_id, rule_version, standard, edition, clause, table_row
  rule_class: TOPOLOGY | GEOMETRY | SEMANTIC
  applicability: {applies, reason, subject_ids[], hazard_ids[]}
  measurand: {name, value, unit, method, U_k2, sources: [object_ids, 面 / 盒边]}
  threshold: {op, value, unit, derived_from: {inputs: {T?, d?, risk_level?}, formula}}
  unknown_inputs: [{name, needed_for, source_hint: datasheet | config | human}]
  decision_rule: guard_band_U_k2 | simple_acceptance
  verdict: PASS | FAIL | NEEDS_MEASUREMENT | NOT_APPLICABLE | CANNOT_DETERMINE
  margin: value − threshold（带符号）, margin_sigma
  confidence: {object_identity, measurement, overall（校准过）}
  evidence: [{object_id, class, class_conf, obb, floor_contact, view_ids, image_crops}]
  explain; severity（ISO 13849-1 风险图 S/F/P 提示，人给）
  provenance: {scene_id, reconstruction_version, rule_pack_hash, timestamp}
}
```
和今天 `evaluate_document` 产出的 finding（`entityId / applicability / machineResult / facts / missingEvidence / comparison`）是同一件事的扩展：
`missingEvidence` → `unknown_inputs`，`comparison` → `measurand + threshold + margin`，新增 `rule_class`、`decision_rule`、`confidence`、`provenance`。

### D8. harness 计划（在 A5 的基础上补；附录 B PART 3）

1. **规则包 v1 冻结**：上面 12 条，每条规则文件带条款、链接、阈值表、单位、不确定度等级（±2 cm 下可判 / 不可判）。
   13857 表 2（100 mm 台阶）和 180 mm 地面缝是可判的；表 4 的 2–25 mm 开口和 13854 的 25–100 mm 指 / 手缝不可判，默认 needs_measurement，除非是目录件。
2. **合成场景单元测试**：参数化生成围栏 / 机器人 / 急停 / 通道，在每个数值阈值的 ±ε、±2σ、±5σ 处摆（1399/1400/1401，179/180/181，199/200/201，499/500/501，光束 300/600/900/1200 vs 350/650）。
   这是 `eval_pack.py` 的直接扩展（它今天只有间距一族）。
3. **性质测试（Hypothesis）/ 蜕变关系**：整体 SE(3) 变换 → 判定和测量值不变；参照物同步缩放 → 不变；防护跨阈值移动 > 2σ → 判定单调翻转（pass → fail，不许 pass → inconc → pass）；
   删除不在任何证据里的对象 → 不变；复制对象 → 不变；不确定度增大 → 只能向 inconclusive 移动，不许 pass↔fail 翻；T 或 d 收紧 → S_required 单调不减。（机械安全检查器的蜕变测试没有文献，这是提案。）
4. **金标集**：按 (场景, 对象, 规则) 标 {pass, fail, inconclusive, n/a} + 测量值 + 证据对象 id，双人标注 + 仲裁，Krippendorff α ≥ 0.80（暂定 ≥ 0.667）；
   每条规则 ≥ 40 个实例、≥ 10 个 fail、≥ 5 个 inconclusive，30–60 个场景覆盖 12 条规则；用 datasheet 记录数据集。
5. **指标门槛**：每条规则在金标上 fail 召回 ≥ 0.95、fail 精确率 ≥ 0.80；inconclusive 率只报告，> 50 % 才算问题；置信度画校准图（可靠性图）。
6. **不确定度**：D4 的护带规则，U 来自尺度参照物误差模型 + 每维 σ；用 split conformal 在金标集上校准，使"90 % 置信 pass"经验上就是 90 %。
7. **LLM 编译器评测**：对手写规则算实体 / 关系 F1（参考：GPT-3.5 few-shot 把建筑规范编成 LegalRuleML 只有 53–71 % F1）、round-trip 等价、两次独立编译的差分测试，全部跑在合成网格上；判定路径里没有 LLM。
   若用 LLM 做解释的评审，只能换序双打分 + 10 % 人抽检（位置偏差、冗长偏差、自我偏好都有文献）。
8. **CI 回归**：规则包语义版本（rule_id@version + 条款 + 版本年）；每个场景的金标判定快照，任何翻转即失败，explain 文本 diff 只告警；每晚全场景蜕变扫描；证据哈希防止重渲染偷偷改证据。
9. **版本跟踪**：规则文件先钉在 ISO 13855:2010 的数字上，开 ticket 等拿到 13855:2024（S = K·T + DDS + Z）和 ISO/CD 13857 正文后重推；13854 低行和 10218-2 的 500 mm 标"待对购买正文核验"。
10. **人在环字段**：T、d、受限空间、风险等级（低 / 高表）、操作站位置；UI 在这些没填之前，13855 一族只能停在 NEEDS_MEASUREMENT。
11. **报告格式**照审核员习惯：条款 → 测量值 ± U → 阈值 → 判定 → 证据裁图；聚合用 TTCN-3 式的最大格（fail > inconclusive > pass）。

### D9. 顺序建议

1. 先做 D3 的四个缺口 + D4 护带（引擎能对 090 / 030 出条件判定）；2. 关系层 + clingo 内核，先迁 7 谓词和规则 3 / 1 / 7 / 8（纯几何，今天的盒就够）；
3. 规则 4 / 6 / 9 / 11（拓扑：占用格 + 可达）；4. 金标集和蜕变套件；5. 规则 2 / 5 / 10 / 12（要人给输入或分辨率不够的）；6. SHACL 导出（有客户要时）；7. 视频阶段 RTAMT。


---

## E. 从别的领域学到的（2026-10-07 晚；案例、模式、失败和证据在 `verdict-patterns-lessons-2026-10-07.md` 及其附录 C / D）

| # | 教训 | 落到本文哪里 |
|---|---|---|
| 1 | 计量式判定：值 ± U、决策规则、护带、ILAC-G8 四态陈述 | D4、D7（报告上印决策规则和 U） |
| 2 | 缺失是值：三值传播；缺对象 / 缺边 → NEEDS_INPUT / CANNOT_DETERMINE，永不 PASS | D2（clingo 失败即否定）、D3 |
| 3 | 规则包是产品：条款对齐、RASE 适用性 / 例外、按标准版本带日期、带测试、专家签字；约一个专家日 / 页 | C、D6、D8 |
| 4 | 发布编译率（编译 / 拒绝 / 需人工修），拒绝桶是功能 | D8 第 7 条 |
| 5 | 声明输入保持声明（T、接近速度、PLr、参照物尺寸），不从照片推 | C "永远不在照片里的输入" |
| 6 | 感知层单独规约（STPL 式：跨视角持久性、尺寸一致、地面接触），规则前检查 | A5 的 workcell_layer_trial 检查要形式化 |
| 7 | 适用性人工签字（IDS / ISO 12100 式） | D3 |
| 8 | 每条判定的证据链足以申诉（照片裁图、盒、值、U、条款、规则版本） | D7 |
| 9 | FAIL 精确率优先于召回；宁可 CANNOT_DETERMINE 不要投机的 FAIL | D8 第 5 条 |
| 10 | 图薄且可查询，只放米制关系，别让图变成"理解" | D1 |
| 11 | 词表外情况记 SOTIF 式 backlog | D2（编译器拒绝的条文 + 无法归类的对象） |
| 12 | 形式化会发现规范 bug，回给 spec 作者 | D8 第 10 条 |

训练路线（附录 D 结论）：训练买到检测 / 定位 / 指点 / 计数 / 定性关系，买不到厘米级米制和审计链；成熟团队收敛到"基础模型产符号 + U → 符号规则 → 证据链"；
我们只后训练符号生产者和规则编译器，永不训练 照片 → 判定；视频阶段训练给动态符号生产者，判定是每帧场景图上的时序逻辑。

## F. 评测：有没有 benchmark、怎么评、LLM 在哪、视频怎么办

### F1. 现成的 benchmark 能评什么

| 类 | 有什么 | 能评我们的哪一层 |
|---|---|---|
| 空间理解（不是合规） | VSI-Bench（视频 → 空间问答，人 79 %）、SpatialRGPT-Bench、ERQA、Real-3DQA / ReVSI（修掉捷径的新版）、3DSSG（关系）、ScanQA / SQA3D（已被证明可作弊） | 只评"符号生产者"的部分能力，评不了判定 |
| 安全 / 合规 | SafetyVisionBench（GuardEn）、ConstructionSite-10K（规则级标签，空间关系类最难）、HomeSafeBench（家居危险，SOTA F1 34.7）、PPE 数据集 | 都是工地 / 家居；**从照片判 ISO 13857 / 13854 的工业工位 benchmark 不存在** |
| 工厂多视角数据（8 月已核验，`2026-08-24-factory-imagery-and-policy-sources.md`） | NVIDIA PhysicalAI-SmartSpaces MTMC_Tracking_2026（28 个真实仓库场景、标定多相机，CC-BY-4.0）、LOCO（CC0）、NVIDIA SDG-Warehouse（合成，5–10 同步视角，EHS 事故场景，商用可） | 视频阶段的感知 / 跟踪评测底料；判定真值要自己加 |

结论：判定层的 benchmark 自己建（D8 的金标集 + 合成网格 + 蜕变），别人的只评部件。

### F2. 分层评，每层自己的真值，LLM 永不当判定的评委

| 层 | 真值来源 | 指标 | LLM 的角色 |
|---|---|---|---|
| 感知（对象类别、盒、σ） | 现场卷尺 / 激光（我们已有 090 / 030 的现场值：罩壳 24.0、横杆 21.0 cm 等）、合成场景精确真值、多视角一致性 | 每对象尺寸 / 位置误差；**σ 校准**（真误差落在 ±2σ 内的比例，目标 ~95 %）；类别精确率 / 召回 | 无；最多做类别提议，人确认 |
| 关系（几何算出） | 从真值盒重算；合成场景 | 边值误差、U 覆盖率 | 无 |
| 规则编译 | 手写参考规则 + 条文考试集（13 条 OSHA 已有；specs 来后扩） | 编译率 / 拒绝率 / 错编率；round-trip 等价；两次独立编译差分 | **被评对象**；另一个 LLM 只做回译辅助人审 |
| 判定 | 金标集：(场景, 对象, 规则) 双人标 + 仲裁，α ≥ 0.8，每规则 ≥ 40 例 ≥ 10 个 fail；合成阈值网格 ±ε / ±2σ / ±5σ；蜕变关系（刚体变换不变、删无关对象不变、U 增大只能向 inconclusive 移、跨阈值单调翻转） | FAIL 召回 ≥ 0.95、精确率 ≥ 0.80；inconclusive 率；置信度校准图 | 无 |
| 解释 | 审核员任务：一分钟内追到照片 + 数字；系统"缺什么"清单与安全工程师清单的一致率；改一个输入（补 σ / 补照片）判定按预期变化 | 任务完成率、用时 | 解释质量初筛（换序双打分，人抽 10 %） |
| 整链 | 真实工位 + 现场值 + 人工判定 | 端到端 FAIL P/R、每条判定的证据可追率 | 无 |

LLM 能做的评测工作：生成合成场景 / 规范变体、把规则回译成自然语言给人审、起草事件标签供人确认、解释初筛。不能做的：出判定、当最终评委、替代现场真值。

### F3. 视频：不标每帧

1 小时 30 fps = 108k 帧，逐帧标不可能也没必要：
1. **标事件不标帧**：真值 = 事件区间（人进入危险区 t1–t2、门开、机器人越界、速度超限），每小时几十条；STL 规则的输出也是事件 / 鲁棒度曲线 → 事件级 P / R + 时间偏差（±0.5 s 内算命中）。
2. **合成视频给逐帧真值**：Isaac Sim / SDG-Warehouse 带精确位姿、速度、区域；真视频只做分布迁移检验。
3. **无标签性质**：时序一致性当感知监控（对象持久、尺寸不变、地面接触——STPL 式）；蜕变（倒放 / 平移 / 裁剪不改事件）；这些不要人标。
4. **主动抽帧**：只把护带内 / 规则边界附近 / 高不确定度的片段给人标（MonitorVLM-v2 式熵分流）；每小时约 200 帧量级。
5. **现成回放集**：平台 phase2 的固定相机视频 R1 / R2 / R3 逐帧判定和 ME340 / Walmart / Sam's Club 的 run journals 可直接当回放基线；"留出一半视角"的评估法也适用。
6. 规模：事件标注 ~50 条 / 小时，抽帧 ~200 帧 / 小时，合成 10 小时真值免费。

## G. 规范怎么"对上"场景图：匹配 = 变量绑定（图：`research/verdict-layer-trial-2026-10-07/out/090/matching.png`）

规范侧不是第二张场景，是**规则模板**：每条规则 = 带变量的模式，四个部分（RASE）各落到场景图词表的一类东西上——
Selection 落到节点类别，Applicability 落到关系（边）或区域节点，Requirement 落到一种边 + 阈值 + 方向（护带），Exception 落到节点属性。
场景图是实例。匹配就是模式匹配（数据库查询 / Datalog 的 grounding）：引擎枚举所有满足 Selection + Applicability 的变量绑定，对每个绑定取 Requirement 指定的那条边的值 ± U 做护带比较，一个绑定出一条判定。
"不同的规则怎么匹配" = 不同的模式，共用同一张图、同一套词表；引擎一次把所有规则的绑定都算出来。

090 的例子：规则 A（离地缝）的 Selection 绑定到 3 个节点（left fence、right fence、safety guard）→ 3 条 floor_gap 边 → −2 ± 100 PASS、84 ± 100 NEEDS_MEASUREMENT、422 ± 101 FAIL；
规则 B（≥ 500 mm）绑定到 7 对（robot × 固定物）→ 7 条 min_distance_3d 边；规则 C（包围）绑定到 1 个区域 → 占用格可达 → CANNOT_DETERMINE。
Applicability 里今天没有的条件（F 是危险区周界的一部分；X 在机器人运动包络附近）就是 zone 节点和受限空间输入要补的地方；补上之前，判定是"若适用"的条件判定。

specs 到了之后的对应流程：每条条款 → 四个部分 → 查词表：Selection 的类别有没有、Applicability 的关系有没有、Requirement 的边有没有、Exception 的属性有没有；
都有 → 编译成规则；缺一个 → 要么加边 / 属性（表示层工作），要么标 needs_input（人 / 数据表给），要么拒绝（照片永远表达不了）。
