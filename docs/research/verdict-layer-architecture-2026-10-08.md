# 判定层（verification / verdict layer）的分层与解耦设计（2026-10-08）

问题：有了 3D 场景图、知识图谱、数字孪生、规则引擎、GraphRAG 这些概念，又有 2026 年的 4D 工作（DAAAM、ChronoGraph、WorldSGG），我们的判定层到底分几层、每层之间靠什么解耦？

一句话：**七层、四个契约。每一层只认上一层的契约，不认上一层的实现。** 重建换成 Hydra / DAAAM、规范从 ISO 换成客户站点规则、照片换成视频，都只换一层，其余不动。
图：`verdict-layer-architecture-2026-10-08.png`。

## 1. 七层

| 层 | 做什么 | 今天的实现 | 以后可换成 | 产出（契约） |
|---|---|---|---|---|
| **L1 感知 / 重建** | 从传感器得到带不确定度的对象 | 照片：SAM 3 → MVS + MoGe-3 → SAM 3D → 组装 v2（急停参照定尺度） | 视频：Hydra / **DAAAM**（BSD-3，RGB-D + 位姿 → 分层 4D 动态场景图，建在 Hydra 上）；**WorldSGG**（单目视频 → 世界系逐时刻 OBB，含被遮挡物体的持久性）；Khronos | **契约 C1 场景契约**：对象 id（跨帧稳定）、类别 + 置信度、有向盒、离地 / 顶高、每维 σ、看到它的视角、`t`；地面；**coverage**（观察到的空间）；区域（人画或控制器给） |
| **L2 薄场景图** | 从 C1 算关系，落盘，可查询 | `scene_graph.py`：3D 最近距离、水平距离、离地缝、z 重叠、above、越过三元组、视线、占用格；导出 Spark-DSG | 关系库扩展（perimeter_of、covers_opening、aisle_width、reachable_from、observed）；Cypher 查询接口（heracles 思路，GPL → 自写） | **契约 C2 事实 schema**：类型化谓词（名、单位、值、U、来源、`t`）；节点 / 边 / 层的 JSON（Spark-DSG 兼容） |
| **L3 感知规约监控** | 检查场景图自身是否可信，再放进判定 | 雏形：workcell_layer_trial 的 shape / floor / lines / plane_stereo / obvious_errors 检查 | STPL / SGSM 式规约：跨帧对象持久、尺寸一致、地面接触、coverage 阈值 | 场景质量事实（进 C2）；不合格的对象 / 帧标 `untrusted`，判定层对其只能出 CANNOT_DETERMINE |
| **L4 规范侧（独立于场景）** | 把规范变成可检索、可对齐的结构 | 8 月编译器（7 谓词）→ 被取代 | 条款图（AEC3PO schema：Document / Table / Definition / RASE Statement；表变查表函数、定义变别名、例外变 defeater、版本 id）+ **对齐表**（条款词 ↔ 签名，GinSign 式，带置信度）+ 检索索引（DriveReg / Lawful-AD 式，锚在工位分类） | **契约 C3 签名**：类别 / 区域 / 边 / 属性 / 声明输入的注册表（`vocabulary.json` 的角色，但是对齐目标不是拒绝门）；条款图 JSON-LD |
| **L5 检查合成与验证** | 把检索到的条款变成经过验证的检查 | — | 对 C2 的有类型 API 合成检查（TUM 函数生成循环）；冗余翻译差分执行、性质测试（PropTest）、签名验证、蜕变；字面化审阅（nl2spec 对照表）；置信度分流 | **契约 C4 规则包**：版本化（rule_id@version + 条款 id + 标准版本）、ASP 规则 + 事实化阈值 + 测试场景 + 审阅记录 |
| **L6 判定引擎** | 把 C2 事实和 C4 规则绑定，出判定 | clingo，护带，五态 | 视频：逐帧 L6 → 裕量信号 → STL / STREL 监控（RTAMT / MoonLight）→ 事件级判定 | **契约 C5 判定对象**：rule@version、subjects、status、measured ± U、threshold、margin、unknown_inputs、evidence refs、provenance |
| **L7 证据与报告** | 渲染、追溯、反馈 | 表、平面图、立体图、匹配图 | 审核员屏：条款 → 值 ± U → 阈值 → 判定 → 照片裁图；coverage / gaps 报告反馈给 L1（补拍）和 L4（词表缺口） | 报告、`coverage.json`、`gaps.json` |

横切：**签名注册表**（C3，L2 和 L4 共用的唯一词表）、**版本三元组**（场景 schema v / 规则包 v / 引擎 v，每条判定都带）、**评测 harness**（冻结基准 + 固定输出，`verdict-evaluation-protocol-2026-10-08.md`）。

## 2. 四个契约 = 四个解耦点

| 契约 | 两侧 | 规则 | 换掉一侧时另一侧动不动 |
|---|---|---|---|
| C1 场景契约 | L1 ↔ L2 | 任何重建 / 建图器都要有一个 adapter 产 C1；L2 永不 import 重建代码（`scene.py` 已是这个边界） | 照片流程 → DAAAM / Hydra / WorldSGG：只写 adapter；L2–L7 不动 |
| C2 事实 schema | L2 ↔ L6 | 引擎只读事实，永不算几何；几何只在 L2（`relations.py` / 关系库）算 | 加一种关系 = 关系库 + 签名注册表各加一行；引擎不动 |
| C3 签名 | L2 ↔ L4 | 规范侧只认签名，不认场景实例；对齐是分类，缺口是待审行 | 换规范包（ISO → 客户规则）：L4 重跑抽取 + 对齐；L1–L3、L6 不动 |
| C4 规则包 | L5 ↔ L6 | 规则包是版本化工件，带测试和审阅记录；引擎不接受没有版本的规则 | 规则改版：新规则包版本 + 金标快照重跑；其余不动 |

这对应截图里的"泛化三解耦"：**表示解耦** = C1 + C2（谁产场景图都行，关系算法一套）；**规则解耦** = C3 + C4（规范侧与引擎只靠签名和规则包相连）；
**检索 / 生成解耦** = L4 检索 + L5 合成与 L6 执行分离（生成的检查先验证再进引擎，LLM 永不在 L6 里）。

## 3. 2026 的 4D 工作放在哪一层

| 工作 | 是什么（查证） | 放哪 | 注意 |
|---|---|---|---|
| **DAAAM**（MIT SPARK，CVPR 2026；https://github.com/MIT-SPARK/DAAAM ，BSD-3，~530★） | 建在 Hydra 上的实时分层 4D 动态场景图：SAM 分割 + BotSort 跟踪 + VLM 描述；输入 RGB-D + 位姿；ROS 2 接口另仓库 | **L1 的视频产者**：写一个 DAAAM → C1 的 adapter（节点 → 对象 + 盒 + id + t；places → coverage） | 它的语义来自 VLM 描述，只取几何 / 位姿 / 身份进 C1，类别当"提议 + 置信度"走对齐表；不取它的学习式关系（3DSSG 教训） |
| **WorldSGG / ActionGenome4D**（2026-03；https://github.com/rohithpeddi/WorldSGG ，许可证页面未显示，数据集邮件申请、checkpoint 待发） | 单目视频 → 世界系逐时刻 OBB + 时空关系，含被遮挡物体（掩码自编码补全） | **L1 的"物体持久性"来源**：被遮挡时的盒进 C1 并标 `believed, not observed` | 判定层对 believed 对象只能出 CANNOT_DETERMINE 或条件判定；这正是 coverage 字段存在的理由 |
| **ChronoGraph**（arXiv 2609.39665，2026-09） | "功能 4D 场景图"：动作作用于 affordance 部件 → 语义 / 几何状态变化；用于交互理解与规划，训练 VLM；代码 / 数据未声明；不涉及安全合规 | **暂不接**。将来做"工艺 / 语义规则"（门开 → 机器人停）时，它的"状态变化边"可作 C2 的事件边来源 | 它是规划表示，不是判定表示 |
| Hydra++ / Khronos / Clio | 建图与长期状态 | L1 产者（同 DAAAM） | — |

共同点：它们都是 **C1 的产者**，一个都不进 L2–L7。截图里"建议路径 Hydra++ → DAAAM → ChronoGraph"是建图路线，对我们等于"换 L1 的 adapter"。

## 4. 时间轴怎么进来（不加层）

- C1 的对象带 `t` 和跨帧稳定 id（由 L1 的跟踪给，L2 不猜）；C2 的每条边带 `t`；静态层（围栏、立柱、光幕）缓存，动态层（人、车、机器人姿态）逐帧更新关联的边。
- L3 加时序规约：对象持久、尺寸不变、跳变检测。
- L6 逐帧跑同一套规则，输出裕量信号；时序规则（"门开后 2 s 内机器人停"）在 STL / STREL 上写（RTAMT / MoonLight），输出事件 + 鲁棒度。
- L7 的证据从"一张裁图"变成"一个事件区间 + 关键帧"。评测标事件不标帧。

## 5. 每层的失败隔离（这才是解耦的意义）

| 坏在哪 | 表现 | 不会波及 |
|---|---|---|
| L1 重建错（尺寸偏、漏物体） | L3 标 untrusted / coverage 低 → L6 出 CANNOT_DETERMINE，附"缺什么" | 规则、引擎、报告格式 |
| L2 关系算错 | 关系库单元测试 + 蜕变测试抓；边带来源可追 | 规范侧 |
| L4 规范抽取 / 对齐错 | 对齐表审阅行；检索召回评测 | 场景侧 |
| L5 检查合成错 | 冗余翻译不一致、测试场景失败 → 不进规则包 | 引擎、已有规则 |
| L6 引擎 bug | 金标快照 diff 翻转 → CI 失败 | 其余层 |
| 规范改版 | 新规则包版本，旧判定仍可复现（版本三元组） | 场景侧 |

## 6. 现状对照（2026-10-08）

已有：C1（`scene.py`）、L2（`scene_graph.py` + Spark-DSG 导出）、L6（clingo + 护带 + 五态）、L7 的图和表、L3 雏形、评测协议。
缺：C1 的 coverage 和 zone、low 置信度盒的 σ（L1）；L3 的形式化；L4 全部（条款图、对齐表、检索）；L5 全部；C4 的版本化规则包；L7 的审阅屏。
顺序见 `verdict-spec-to-check-2026-10-08.md` §3（先 ISO 13857 + 13855 二十条做 L4–L5 端到端，和旧编译器比）。
