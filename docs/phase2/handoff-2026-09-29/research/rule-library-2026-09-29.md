# 走查视频的判定规则库（车间 / 仓库 / 卖场）

日期：2026-09-29 · 对象：mvp3/judge（`fast_report/judge.py`、`fast_report/cards.py`）· 状态：研究稿，未实现

问题（用户原话）：覆盖太少，为什么这么少？

标记约定：
- **【法】** 美国联邦法规原文：29 CFR 1910 / 1926、28 CFR 36，以及 ADA 2010 Standards。
- **【规范】** 消防规范：NFPA、IFC。地方主管机关（AHJ）采纳后才有法律效力。
- **【标准】** 行业共识标准：ANSI。
- **【站点】** 站点政策的默认值，不是法律。站点可以改。
- **核实：一手** 原文逐字读过：eCFR API 2026-08-01 快照，或 access-board.gov 全文。
- **核实：二手** 只读到二手摘录：厂商博客、UpCodes、大学 EHS 资料。标准本身收费或拿不到。
- **〔估〕** 我们自己的设计或假设，没有外部来源。
- **〔未核〕** 没有核实到原文。

数字来源：
- 本轮的三段视频：`research-notes/phase2/runs/mvp3-judge-{me340,samsclub-a2,walmart}-001`，每段取最后一次调用的 object_cards 和 judgements。
- round 2 的结果：`runs/mvp2-results/summary.md`。
- 离线投影：用现有卡片字段重算新规则，不是新代码的运行结果。

---

## 0. 结论：为什么覆盖这么少

覆盖少有四个原因。第一个原因最大，但单独修它不够。

**1. 规则类型太少。大多数类别根本没有规则。**
- judge 只有 J1-J9，其中常出结论的只有 J1、J2、J4、J5、J9 和人员规则。
- `applicable()` 只按 STACK / GOODS / DEFORMABLE / LADDER 这几个名字表派规则，另外按位置给 J5 和 J9。
- 结果：ME340 的 255 个对象里，工具 36、材料 36、储物 29、电气 12、无类 27 全部没有任何一行规则。这五类合计 140 个，占 55%。

**2. 规则适用时，测量的 u 盖不住阈值。**
- **J1（2.5 m）**：Sam's Club 有 369 行，其中 239 行是"straddles"（区间跨阈值）。u 的大头是 25% 的尺度项：2.2 m 处约 ±0.55 m。
  - 离线重算：尺度项降到 10% 时，J1 单看数值能判 PASS 的从 116 行升到 218 行（共 366 行）。这个数还没算"单视角集 / 尺寸不合理"这些降级闸门。
- **位置 u 太大**：货物和散件的位置 u（已去掉尺度项）中位数是 Sam's 0.77 m、Walmart 0.46 m、ME340 0.36 m。凡是"离某物多远"而阈值在 0.1-0.5 m 的规则，用卡片位置都判不了。
- **深度基本看不到**：看到 depth 的卡只有 ME340 46/255、Sam's 4/622、Walmart 31/723。所以 J2 和堆高/底宽比都缺一边。

**3. 名字映射有错类。** 有些错类会把规则用到错的对象上。
- ME340 有 8 张 door 卡，其中 7 张是机器或柜子的门：cabinet door、cnc machine door、lathe cabinet door、locker door、cnc mill door 等。
- 其他错类：`electrical enclosure` → machine，`ceiling panel` → control panel，Walmart 的 `display rack panel` → control panel，`sign holder` → tool holder。
- 仍然无类的卡：ME340 27、Sam's 11、Walmart 16。

**4. 三段视频里没有确认到任何安全设备。**
- 没有一个灭火器、出口标志、急停、洗眼器、梯子或叉车被确认。它们只以候选词出现并被闸门拒绝。例如 `fire extinguisher` 出现在一个油壶和一袋狗粮的检测词里。
- 所以即使有消防和出口规则，这三段视频上也不会产生行。

**本文怎么补：**
- **按对象族写规则。** 每个常见族至少有一条能从几何、位置和类别判出结果的规则，不依赖 VLM。见 §3 和 §4 的族矩阵。
- **优先用不怕尺度误差的量。**
  - 比值：堆高/底宽。
  - 角度：梯子、立柱。
  - 同一融合点云里两面的相对差：面到面的突出量、面前净空。
  - 大余量规则：头顶净高、储物顶到天花。
- **每条规则写明**：来源条款、法律还是站点默认、适用的类/族、要哪些字段、阈值、什么情况给 NEEDS_REVIEW 或 NO_DATA。
- **先修的其实是字段，不是规则条数。**
  - 只用现有卡片字段离线投影：ME340 能出结论的对象从 17 升到 92，主要是 H1 的"已离地存放"70 个，外加 O1 的 5 个灯和管。
  - 卖场两段几乎不涨：Sam's 103 → 110，Walmart 63 → 64。
  - 卖场要涨，需要 §2 的三个新字段：储物顶到天花的距离、面到面突出量、堆的两边尺寸，外加尺度锚。

---

## 1. 统一约定（所有规则共用）

### 1.1 判定

判定沿用 `judge.py` 的写法：`video.banded_verdict`、`forced()`、`fact_doubts()`，mvp3 的 `combine()` 也不变。

**上限规则**（值不得超过阈值）：
- PASS：v + u < T_pass
- FAIL：v − u > T_fail

**下限规则**（值不得低于阈值）：
- PASS：v − u > T_pass
- FAIL：v + u < T_fail

**两个阈值怎么来。** 有些条件在视频里看不到，比如电压、建造年份、喷淋头类型、是不是公共通道。这时：
- T_pass 取最严的可能阈值；
- T_fail 取最宽的可能阈值；
- 值落在两者之间，给 NEEDS_REVIEW，并写出缺的是哪一项站点输入。

站点档案一旦填了这项，两个阈值合成一个。这样不会因为假设而产生假 PASS，也不会产生假 FAIL。

**强制降为 NEEDS_REVIEW 的情况**（沿用现有逻辑）：
- 只有一个视角子集；
- 尺寸对这个类不合理，或点云支撑是碎的（`doubts()`）；
- 边界值在错的一侧：'at least' 的值不能对上限判 PASS；'at most' 的值不能对下限判 PASS。

**`plain(f)`** 表示同时满足：有 value、没有 status、n_subsets ≥ 2、所属卡片 `usable()`。

**两物之间的间距**，u 取 `sqrt(u_rel(a)² + u_rel(b)² + (SCALE_REL·gap)²)`（`judge.u_rel`）。两物的共同尺度误差不计两次。

**NO_DATA**：需要的字段缺失或测不到。

**NOT_RESOLVABLE**（新 reason，归在 NO_DATA 下）：u 本身比容差还大，PASS 不可能；值又没到 FAIL。例：立柱垂直度容差 0.24°，而角度 u ≥ 1°。这种情况不要进复核队列。

**N/A 行（新增）**：规则存在，但按位置或类别不适用，例如"离走道 3 m，不做 J5"。
- 现在 `geometry()` 在这种情况返回 None，结果看不见，用户就看到"no check applies"。
- 改成出一行"不适用：原因"。覆盖率分三档统计：有结论、复核或缺数据、不适用。"没有任何规则"应当接近 0。

### 1.2 VLM 的位置

- 看图只能做三件事：让 PASS 降为复核、让 FAIL 降为复核（与几何意见不一致时）、给复核行排优先级。它永远不能单独产生 PASS 或 FAIL，这与 mvp3 的 `combine()` 一致。
- 只能看图回答的问题（§5）只交给本地级联：
  - 先用命名时已经算好的 SigLIP 2 物体向量，配一对属性文本做零样本判断。向量现成，只多算文本向量和点积，几乎不花钱。
  - 拿不准的按向量聚类，每类挑一张裁剪图，交给本地 vLLM 上的 Qwen3-VL（与 `cascade.py` 第 5 步同一机制）。
  - 不对每个物体调用云端 VLM。
- 只能看图判的规则不算"可判"，不计入 §4 的覆盖目标。

### 1.3 站点档案（新，每份报告一份）

每项都有保守默认。默认值不会让 PASS 变宽。

| 键 | 取值 | 默认 | 影响的规则 |
|---|---|---|---|
| `site_type` | workshop / warehouse / retail | 从视频判断，拿不准时为 unknown | W1、P1、CT1 |
| `accessible_route` | 顾客通道是否算 ADA 无障碍通道 | retail 时为 yes〔站点〕 | W1 |
| `elec_voltage_to_ground` | ≤150 / 151-600 / unknown | unknown | E1 |
| `elec_condition` | A / B / C / unknown | unknown | E1 |
| `elec_built_after_1981` | yes / unknown | unknown | E1 |
| `sprinkler` | none / standard spray / ESFR 或 CMSA / unknown | unknown | S1 |
| `deflector_allowance_m` | 喷淋头低于管和顶棚、却没有重建出来的余量 | 0.30〔估，未核〕 | S1 |
| `exit_access` | 被走过的通道都算出口通道 | yes（与现在的 J5 一致） | W1、O1、X1 |
| `forklift_aisles` | 画面里有叉车，或站点声明有 | 看到叉车即为 yes | W1 |

### 1.4 尺度

- 现在所有米制值的 u 都含 25% 的尺度项：`cards.SCALE_REL`，来自假定 1.6 m 相机高度。
- 规则库本身解决不了尺度问题。规则只能做两件事：
  1. 优先用不受尺度影响的量：比值、角度、同一融合点云里的相对量；
  2. 优先用余量大的阈值。
- 不能把"假定相机高"当成可信尺度。见代码核对员更正里 `policy.py:303` 的漏洞：`camera_height` 被当作合格尺度。视频路径应沿用 `video.contract_scale` → model_estimated。
- 尺度锚（〔估〕，要单独验证）是降低尺度项的唯一办法：
  - 站点给一次实测参照，比如卷尺量过的门高或托盘；这就是 operator_anchored。
  - 或者用标准尺寸物体。例：GMA 托盘 48×40 in（1.219×1.016 m，行业常用尺寸，本文未核）。
  - Sam's Club 上 20 块托盘的 visible_length 中位数是 1.22 m（u 0.50 m），与 48 in 一致。这只是一个中位数，不是标定。要逐个镜头验证它们是否一致。

---

## 2. 要新增的测量字段

下面这些字段都从现有数组算，没有新模型调用：
- 房间 TSDF 点：`shot['room_floor']`；
- J5 的占用网格：`judge.aisle()`，5 cm 网格，含 seen / occupied / owner；
- `face_overhang` 的"面对相机路径的面"坐标系。

| 字段 | 定义 | u | 状态语义 | 用于 |
|---|---|---|---|---|
| `clear_under` | 物体 footprint 下的地面格里，"地面被看到"且在 0.1 m 到 base−0.1 m 之间房间点少于 OCC_MIN_PTS 的格所占比例 | 比例，没有 u；要求至少 20 格 | 地面没看到时 NO_DATA | O1 |
| `over_aisle` | aisle() 的横截线上，这个物体的 footprint 落在两侧停止点之间（空的那段）的样本数 | — | 0 表示不在走道上方 | O1、P1 |
| `ceiling_gap` | 两个值：`gap_low` = 顶面以上最低的上方表面（footprint 外扩 0.3 m 的柱内，高于 top+0.3 的房间点取 p10）减 top；`gap_deck` = 同一柱内最高的致密水平层减 top | sqrt(u_rel(top)² + (DEPTH_REL·到天花的距离)² + 体素² + (SCALE_REL·gap)²) | 上方点少于 20 时 NO_DATA | S1 |
| `front_clearance` | 从面板、灭火器或门的前面（房间点在其 footprint 内沿法向取 p10，做法同 face_overhang）沿外法向扫描。扫描带宽 max(物体宽, 0.762 m)，高度 0.1-1.98 m。距离取到第一个被占的格（TSDF 点或可用卡片的 footprint）；遇到长于 0.3 m 的未见段时停下，结果为下界 | sqrt(2·(GRID/2)² + (DEPTH_REL·Δrange)² + 体素² + (SCALE_REL·d)²)。面和障碍都来自同一个融合点云，不用卡片位置 | 'at least'（未见）；只看到侧面时 NO_DATA | E1、F2、X1、EW1 |
| `protrusion_m` | 物品的前面（朝向走道的点取 p10）相对其宿主（货架、层板、挂板、墙）前面的外伸，做法同 face_overhang，按横向分 3 段 | 段间离散 + √2·体素 + SCALE_REL·值 | 宿主面少于 60 点时 NO_DATA | P1 |
| `stack_base` | stacks() 聚合出的堆：root 的两条边 width 和 depth、堆高 H = top(top 卡) − base(root) | 各自的 u | 任一边只看到一侧时只有下界 | S2 |
| `zone_hits` | 在规则定义的禁放区（多边形，高 0.1-1.8 m）内：可用卡片 footprint 侵入的深度，以及区内被占的 TSDF 格 | 侵入深度的 u 用 u_gap | 区内被看到的格少于 80% 时 NO_DATA | E1、X1、F2、H2 |
| `route_m` | 沿走过路径（相机路径和通过闸门的人员路径，并成一张图）的路程：上界 = 沿路径走的长度；下界 = 直线距离 | 路径位置 u | — | F3、EW1 |
| `low_clutter` | 区内离地 0.1-0.4 m 的被占格，且不属于任何固定类卡片 | — | — | H2 |

后续字段（P3，需要 SAM 3 部件提示或 OCR，本期不做）：
- `forks_base`：叉车货叉离地高度；
- `letter_height`：出口标志字高，由 OCR 框和深度换算；
- `rail_above_surface`：护栏顶到其所在平台面的高度；
- `marked_lane`：两条平行地标线围成的通道多边形。

---

## 3. 规则

每条规则的标题注明三件事：新增、保留还是修订；优先级 P1-P3；可判程度（可判 / 只能判 FAIL / 只能判 PASS / 只能看图）。

"适用"一栏写 `cards.TAXONOMY` 的族和类。类名以 §6 修正后的映射为准。

### 3.A 通道、出口、头顶

#### W1 通道与出口通道净宽（修订 J5）〔修订 · P1 · 可判〕
- **来源**：
  - 29 CFR 1910.36(g)(2)：出口通道各处至少 28 in（0.711 m）。【法】核实：一手
  - 1910.36(g)(4)：伸入出口路线的物体不得让宽度低于最小值。【法】核实：一手
  - 1910.37(a)(3)：出口路线必须畅通；不得在出口路线内放置材料或设备，临时也不行。【法】核实：一手
  - 1910.176(a)：使用搬运机械的通道要留足安全间隙、保持畅通。没有给数值。【法】核实：一手
  - 卖场：ADA 2010 §403.5.1，走行面净宽至少 36 in（915 mm）；可以缩到 32 in（815 mm），但每段不超过 24 in（610 mm）长。维护义务见 28 CFR 36.211(a)。【法·ADA】核实：一手
  - 叉车通道：OSHA 1972 年的解释信建议"比最大车辆宽 3 ft，至少 4 ft"。这封信已撤回，不可执法。【站点】核实：二手（JJ Keller）
- **适用**：所有非 agent、非 deformable 的卡，只要在走过的路径旁（现在 J5 的位置闸门）。这涵盖全部族。
- **字段**：aisle() 扫描出的净宽 w ± u、'bounded' / 'pairs' / 下界标记，全部现有。
- **阈值**：
  - workshop 和 warehouse：T = 0.711。
  - retail 且 accessible_route=yes：T_pass = 0.915。T_fail 有两种情况：窄段连续 ≥ 0.61 m 时为 0.915；任意长度时为 0.815。
  - forklift_aisles：T_pass = max(1.22, 最宽车宽 + 0.914)〔站点〕。只有站点采用这个值时才用于 FAIL。
- **判定**：沿用 g_j5。
  - PASS：w − u > T_pass。下界值也可以 PASS。
  - FAIL：w + u < T_fail，两侧都停在观测到的表面，并且本物在 ≥2 个相邻样本上界定窄处。
  - 其余情况给 NEEDS_REVIEW。
  - 没有房间点或没有路径时 NO_DATA。
- **看图问题**：无。现有的 q4 只用来排优先级。
- **三段视频**：mvp3 已有 12/64/68 行（round 2 是 14/82/70）。卖场上的 retail 阈值会让一部分 PASS 变为 NEEDS_REVIEW。这是更正确的结果。

#### O1 头顶净高（悬挂物、伸出物）〔新 · P1 · 可判〕
- **来源**：
  - 1910.36(g)(1)：出口路线天花至少 7 ft 6 in（2.3 m）；天花上的伸出物不得低于 6 ft 8 in（原文写 2.0 m）。【法】核实：一手
  - 1910.25(b)(2)：楼梯踏步上方净高至少 6 ft 8 in（203 cm）。【法】核实：一手
  - ADA §307.4：通行路径上方净高至少 80 in（2030 mm）；门闭门器和门挡例外，可到 78 in（1980 mm）。【法·ADA】核实：一手
- **适用**：满足 clear_under ≥ 0.8 且 over_aisle ≥ 2 的任何非 agent 卡。
  - 典型的类：light fixture、fan（electrical 族），pipe、duct、cable tray，悬挂的 cable 和 hose（linear 族），sign、safety sign、exit sign（signage 族），跨通道的 rack beam、conveyor，门楣和卷帘门底边（building 族）。
  - clear_under < 0.8 表示下面有支撑，是装在货架或墙上的东西。这时 O1 给 N/A，交给 P1。
- **字段**：base_above_floor（现有）；clear_under、over_aisle（新）。
- **阈值**：T_pass = 2.03，T_fail = 1.98（取 ADA 例外值和 OSHA 米制写法中较宽的那个）。
- **判定**：
  - PASS：base − u > 2.03。
  - FAIL：base + u < 1.98，且 base − u > 0.685，且 clear_under ≥ 0.8，且 over_aisle ≥ 2。base 低于 0.685 m 的算地面障碍，由 W1/J9 管。
  - 其余给 NEEDS_REVIEW；base 测不到时 NO_DATA。
- **看图问题**：无。
- **三段视频**：余量大，能判。
  - ME340 的灯 base 3.14-3.64 ± 0.82-0.97 m，线管 3.4-3.6 ± 0.9 m；投影有 5 个 PASS。
  - Walmart 的吊牌 base 3.9 ± 1.1 m。
  - 反例：Walmart 的 obj-1-482 'store sign'，base 1.45 ± 0.44 m，离路径 0.36 m。没有 clear_under 就会被错判 FAIL；它其实装在货架上。这个例子说明 clear_under 是必需的。

#### X1 出口门、门口禁放区〔新 · P2 · 可判〕
- **来源**：
  - 1910.37(a)(3)：出口路线内不得放置材料或设备。【法】核实：一手
  - 1910.36(g)(2)：门口通道宽度。【法】
  - 禁放区深度 0.9 m：〔站点〕，〔估〕，没有法定数值。
- **适用**：building 族的 door 类，只算修正后的建筑门：door、roll up door、overhead door、doorway、exit door。
  - "出口门"的判定：3 m 内、门上方有已确认的 exit sign 卡（位置规则），或者站点声明。
  - 不是出口门的，只出 minor 级复核。
- **字段**：门洞线取门 footprint 的长边；zone_hits（两侧各 0.9 m 深、宽等于门洞宽的矩形）；front_clearance（新）。
- **判定**：
  - PASS：两侧区内被看到的格 ≥ 80%，并且把区收缩 u 后，区内没有占用格、也没有卡片。
  - FAIL：是出口门，且某张可用卡片（非 agent、非墙柱、base < 1.8 m）侵入区内的深度 − u > 0。
  - 不是出口门但满足同样条件：给 NEEDS_REVIEW（门口通行）。
  - 门只看到侧面、法向不明时 NO_DATA。
- **看图问题**：无。门能不能打开属于 1910.36(d)，是功能测试，视频做不了，不列入。

#### X2 出口门净宽 ≥ 28 in〔新 · P2 · 可判〕
- **来源**：1910.36(g)(2)。【法】核实：一手
- **适用**：出口门（判定方式同 X1）。
- **字段**：门 footprint 长边，或 visible_length（'at least'）。
- **阈值与判定**：
  - 门扇宽和净宽之差：留 0.05 m〔估〕。
  - PASS：w − 0.05 − u > 0.711。'at least' 的值也可以 PASS。
  - FAIL：w + u < 0.711，且是出口门，且 w 不是 'at least'。
  - 其余给 NEEDS_REVIEW。
- **三段视频**：ME340 修正映射后只剩 1 扇卷帘门。

#### X3 出口标志字高 ≥ 6 in〔新 · P3 · 只能判 FAIL〕
- **来源**：1910.37(b)(7)："Exit"字母不小于 6 in（15.2 cm），笔画不小于 3/4 in。【法】核实：一手
- **适用**：exit sign 类。
- **判定**：
  - FAIL：标志牌总高 + u < 0.152。牌子本身都放不下 6 in 的字。
  - PASS 需要 letter_height（P3 字段）；在此之前给 NO_DATA，reason 写"字高未测"。
- **看图问题**：V7，标志是否点亮（1910.37(b)(6)）。只作复核提示。

#### W2 物体侵入地标通道〔新 · P3 · 可判〕
- **来源**：
  - 1910.176(a)：永久通道要适当标示。【法】
  - starter 政策"货架不得侵入标示通道"，以及"托盘距地标线 ≥ 0.5 m"。【站点】
- **适用**：有 marked_lane（P3 字段）的镜头。三段视频里都没有地标线卡，所以这里暂时覆盖 0。
- **判定**：
  - FAIL：footprint 侵入 lane 的深度 − u_gap > 0。
  - PASS：footprint 到 lane 的距离 − u_gap > 站点距离（默认 0）。

### 3.B 地面与整洁

#### J9 地面上的矮物、在走过的路径旁（绊倒）〔保留〕
- **来源**：
  - 1910.22(a)(3)：走行面上不得有突出物等危险。【法】核实：一手
  - 1910.176(c)：存储区不得堆积会绊倒人的材料。【法】核实：一手
- 判定逻辑不变。mvp3 规定：只看图的话，最多给 NEEDS_REVIEW，并标"likely hazard"。

#### J4 线缆或软管横过走过的路径〔保留〕
- **来源**：1910.22(a)(3)，1910.176(c)。【法】
- 判定逻辑不变。
- **补充**：电线穿过门洞属于 1910.305(g)(1)(iv)(C)【法，核实：一手】。
  - 条件：cable 类的 footprint 与建筑门的门洞线相交，且相交长度 − u > 0 → FAIL。
  - 列为 P3。位置 u 0.4 m 左右，大多会是 NEEDS_REVIEW。

#### H1 散放物件落地（整洁）〔新 · P1 · 可判〕
- **来源**：
  - 1910.22(a)(1)：场所、通道、库房保持清洁有序。【法】核实：一手
  - 1910.22(a)(3)。【法】
  - 1910.176(c)。【法】
  - 这些条款都没有数值。"落地"和"矮"的界线沿用 judge 现有的 FLOOR_BASE_M 0.10 m 和 TRIP_TOP_M 0.40 m。【站点】〔估〕
- **适用**：散件类。
  - tool 族：hand tool、power tool、tool tray。
  - material 族：metal part、metal sheet、wooden board。
  - misc 族：rag、paper、wrap。
  - goods 族里的小件：bottle、can、cup。
  - 不包括：tool box、bucket、trash can、drum（本来就放在地上）；卖场陈列的 merchandise、box、bag（交给 W1/J9）。
- **字段**：base_above_floor、top_above_floor、nearest_walked_path（均为现有）。
- **判定**：
  - PASS，说明写"离地存放"：base − u > 0.30。
  - FAIL：base + u ≤ 0.10，且 top + u ≤ 0.40，且卡片 plain，且在走过路径 1 m 内或在 H2 的机器区内。
  - 同样落地、但远离路径和机器区：给 NEEDS_REVIEW（minor）。
  - 介于两者之间：NEEDS_REVIEW。
  - base 缺失：NO_DATA。
- **报告口径**：H1 的 PASS 是便宜的证据，只说明东西不在地上。统计覆盖率时要和"有危险含义的检查"分开计。
- **三段视频**（离线投影，只用现有字段）：ME340 70 PASS、5 NEEDS_REVIEW；Sam's 7 PASS、5 NEEDS_REVIEW；Walmart 0 行。

#### H2 机器操作区地面整洁〔新 · P1 · 可判〕
- **来源**：
  - 1910.22(a)(1)(3)。【法】核实：一手
  - 1910.212(a)(1) 的"机器区域"（machine area）概念。【法】
  - 区半径 0.9 m：【站点】〔估〕。
- **适用**：machine 族全部类，外加 workbench。区域 = footprint 外扩 0.9 m，再减去其他固定类卡片的 footprint。
- **字段**：zone_hits、low_clutter（新）；散件卡片取 H1 的"落地"条件。
- **判定**：
  - PASS：区内被看到的格 ≥ 80%，没有 low_clutter 簇（≥ 0.05 m²），也没有落地的散件卡。说明写："机器 0.9 m 内没有 ≥0.10 m 高的地面物（区域看到 x%）"。
  - FAIL：区内有 H1 判 FAIL 的散件卡，即 H1 落地条件成立且到机器 footprint 的距离 + u ≤ 0.9。
  - NEEDS_REVIEW：有 low_clutter 簇但说不出是什么；或者区域被看到的比例 < 80%。
  - 没有房间点：NO_DATA。
- **看图问题**：无。
- **三段视频**：ME340 的 38 台机器今天只有 5 行，H2 能让它们全部出行。

#### SP1 地面溢液〔新 · P2 · 只能看图〕
- **来源**：1910.22(a)(2)：地面尽量保持干燥。(a)(3)：泄漏和溢液。【法】核实：一手
- **说明**：薄液膜没有几何高度，只能看图（V5）。落在走过路径上的，给 NEEDS_REVIEW 并提高优先级；其余给 NO_DATA。

### 3.C 储存与堆放

#### J1 堆高 ≤ 2.5 m〔保留，重新标来源〕
- **来源**：2.5 m 来自 starter.md 的"Stacked pallets must not exceed 2.5 m"。这份文件明确写着 illustrative, not certified。所以 **J1 是【站点】规则，没有法律数值**。1910.176(b) 只要求"限制高度以保持稳定"。
- 判定不变。
- 它判不出来的主因是尺度项（§0 第 2 条）。有法律或规范依据的堆放规则是 S1 和 S2。

#### J2 堆稳定（外伸 ≤ 0.10 m、倾斜 ≤ 5°）〔保留，【站点】〕
- 判定不变。
- Sam's Club 本轮有 68 行 NEEDS_REVIEW、35 行 NO_DATA，全部判不出。原因是只看到一面，另一侧不能证明没有外伸。
- S2 给出不依赖面的稳定证据。

#### S1 储物顶到天花、喷淋头的净空〔新 · P1 · 可判〕
- **来源**：
  - IFC 2021 §315.3.1：无喷淋区，储物离天花 ≥ 2 ft（610 mm）；有喷淋区，储物离喷淋头溅水盘 ≥ 18 in（457 mm）。靠墙储物例外。【规范】核实：二手（UpCodes 摘录和搜索摘要；ICC 原页 403）
  - NFPA 13：标准喷头 18 in；ESFR 和 CMSA 36 in（914 mm）。【规范】核实：二手（UpCodes）
  - 29 CFR 1910.159(c)(10)：喷头与下方材料最小竖直净空 18 in（45.7 cm）。但 (a)(1) 和 (b) 规定它只适用于为满足 OSHA 要求而装的系统。大多数商店和仓库的这条要求来自消防规范。【法，适用面窄】核实：一手
- **适用**：每一列储存的最高物。包括：
  - J1 的堆顶（goods 族）；
  - 货架和层架最高层上的货（storage 族：shelf、rack、display rack、cabinet、refrigerator 顶上的东西）；
  - 储物家具本身的顶。
- **字段**：top_above_floor（现有）；ceiling_gap 的 gap_low 和 gap_deck（新）。
- **阈值**：
  - T_fail = 0.457：所有情况里最宽的一个。按 IFC，无喷淋是 0.61，比 0.457 严。
  - T_pass 取决于 sprinkler：unknown 或 ESFR 时 0.914；standard spray 时 0.457；none 时 0.61。
- **判定**：
  - PASS：gap_low − u − deflector_allowance > T_pass。量到"最低的上方表面"，再扣掉没有重建出来的喷头。
  - FAIL：gap_deck + u < 0.457。喷头都在顶棚下面，所以到顶棚的距离小于 0.457 时，到喷头一定也小于 0.457，这个 FAIL 是可靠的。
  - 物体离建筑墙面 < 0.3 m 时，FAIL 降为 NEEDS_REVIEW（IFC 的靠墙例外可能适用）。
  - 其余给 NEEDS_REVIEW。
  - 顶以上重建出的点少于 20 个：NO_DATA，reason 写"上方没有重建"。
- **看图问题**：无。能认出喷头类的话（§6 建议新增 `sprinkler`），可以直接量到溅水盘。
- **可行性**：用较早的 DA3 锚定点云检查过，这不是 mvp3 的 TSDF。
  - Sam's Club（run 281）的点一直到 8.2 m（p99.9），7.5-8 m 有一层，是屋顶结构。
  - ME340（176-fused）3.5-4.0 m 有一层。
  - Walmart（175-fused）的点到约 3 m 就没了（p99.9 3.7 m）。Walmart 的 S1 会是 NO_DATA，除非镜头往上拍。

#### S2 堆高与最小底边之比 ≤ 3（与尺度无关）〔新 · P2 · 两边都看到才可判；只看到一边时只能判 FAIL〕
- **来源**：
  - 南非 General Safety Regulations 8(4)(b)（OHS Act 85 of 1993）：规则形状、侧面垂直的堆，总高不超过底面较短边的 3 倍。【法（南非）；在美国作【站点】默认】核实：二手（UCT 转载的条文）
  - 奥克兰大学的资料说摩擦好时可以放宽到 4:1。【站点】〔未核，没打开原文〕
  - 在美国，这是 1910.176(b)"限高以保持稳定"的一种具体解读。
- **适用**：J1 的地面堆，以及梁上的货位（用 stack_base）；goods 族。
- **字段**：stack_base 给出的 H、B = min(root 的 width, depth)。比值的相对 u 为 sqrt((u_H/H)² + (u_B/B)²)，只用几何项，尺度项在比值里抵消。
- **判定**：
  - PASS：两条边都 plain，且 r·(1+u_r) < 3。
  - FAIL：r·(1−u_r) > 3，并且同时满足：
    - H plain 或 'at least'；
    - 用来算 B 的那条边 plain 或 'at most'；
    - root 卡可用；
    - u_B/B ≤ 0.5。
    - 注：只看到一边时 B ≤ 看到的那条边，所以算出的比值是下界，FAIL 仍然成立。
  - 只看到一边且下界没超过 3：NO_DATA，reason 写"只看到一边：比值只能判 FAIL"。
- **三段视频**：
  - Sam's Club 地面堆的 h/w 约 0.9-1.8，远低于 3。但 depth 没看到，所以现在全是 NO_DATA（投影 16 行）。有了第二侧视角或顶面深度下界就能 PASS。
  - 投影中出现过一个 FAIL 陷阱：obj-1-354 'pallet of toilet paper'，w 0.25 ± 0.20 m。这是碎片，不是整堆，u_B/B = 0.8。u_B/B ≤ 0.5 这个闸门就是为挡住它设的。

#### R1 货架立柱垂直度〔新 · P2 · 只能判 FAIL，平时给 NOT_RESOLVABLE〕
- **来源**：ANSI MH16.1-2023 §4.10（据 Damotech 摘录）。受载立柱的倾斜率和弯曲率上限都是高度的 1/240，约 0.24°，1/2 in 每 10 ft。超出的要卸载并纠正，受损部分隔离停用（§4.4）。【标准】核实：二手
- **适用**：storage 族的 rack 类，包括 rack upright、upright frame 等部件名。
- **字段**：principal_axis_tilt_deg 或立柱面的 planar_slope（现有，已过 plumb 闸门）。
- **判定**：
  - FAIL：tilt − u > 0.24°。
  - PASS 不可能，因为 u ≥ UP_MIN_DEG = 1°。其余给 NO_DATA/NOT_RESOLVABLE。
  - 大角度（例如 6 ± 2°）的立柱是真的损坏，能被抓到。
- **看图问题**：V8，立柱有没有明显凹陷或弯折。只作复核提示。

#### P1 伸入通行路径的突出物〔新 · P1 字段 / P2 规则 · 可判（需 protrusion_m）〕
- **来源**：
  - ADA §307.2：前缘离地 27-80 in（685-2030 mm）的物体，水平伸入通行路径不得超过 4 in（100 mm）。§307.3：立柱上的物体最多外伸 12 in（305 mm）。适用于公共通行路径（卖场属 Title III 公共场所）。【法·ADA】核实：一手
  - 员工区：1910.22(a)(3) 禁止"突出物"，但没有数值。4 in 作为【站点】默认。
  - 员工工作区按 ADA §203.9 大部分可以豁免。【法·ADA】核实：一手
- **适用**：
  - 货架、货位、挂板上的 goods 和 tool 族物件，宿主是 storage 族；
  - 墙上的 sign、control panel、tool holder；
  - 条件：面向走道，即 over_aisle 或 nearest_walked_path − u ≤ 1 m。
- **字段**：protrusion_m（新）、base_above_floor、top_above_floor。
- **判定**：
  - PASS：protrusion + u ≤ 0.10。
  - FAIL：protrusion − u > 0.10，且 base − u > 0.685，且 base + u < 2.03，且面向走道。
  - 其余给 NEEDS_REVIEW。
  - 宿主面测不到：NO_DATA。
- **为什么非要新字段**：用卡片位置算，u_rel 中位数 0.36-0.77 m，是容差的 4-8 倍。离线投影里 P1 在三段视频全部是 NEEDS_REVIEW（33/158/274 行），PASS 为 0。**P1 不得用卡片位置上线**，否则会淹没复核队列。

### 3.D 电气

#### E1 电气设备前的工作空间（深度、宽度、净高，不得堆物）〔新 · P1 · 可判〕
- **来源**：29 CFR 1910.303(g)(1)，适用于对地 600 V 及以下、可能带电检修的设备。【法】核实：一手
  - (i)(A) + Table S-1 给出最小深度：
    - 0-150 V：三种条件都是 0.9 m（3 ft）；
    - 151-600 V：条件 A 0.9 m，条件 B 1.0 m，条件 C 1.2 m；
    - 注 1：1981-04-16 以前的装置可以是 0.7 m（2.5 ft）。
  - (i)(B)：宽度取 max(设备宽, 762 mm)，并且门要能开到 90°。
  - (i)(C)：从地面一直净空到 (vi) 规定的高度。上下方同属电气装置的设备可以伸出 ≤ 153 mm。
  - (vi)：净高 1.91 m（2007-08-13 以前）或 1.98 m；设备更高时取设备高。
  - (ii)：工作空间不得用于储物。
- **适用**：§6 新拆出的 electrical 族 `electrical panel` 类。
  - 包括：electrical panel、breaker panel、panelboard、switchboard、fuse box、disconnect switch、motor control center、electrical enclosure、electrical cabinet、control cabinet、junction box、electrical box。junction box 只出 minor 级。
  - 不包括操作用的 control panel（DRO、readout、pendant、console）；也不包括被错映射的 ceiling panel 和 display rack panel。
- **字段**：front_clearance（新）。扫描带宽 max(设备宽, 0.762)，高度 0.1-1.98 m。紧贴设备上下方的 conduit/pipe 卡，在外伸 ≤ 0.153 m 时不算障碍。
- **阈值**：
  - T_fail：elec_built_after_1981=yes 时 0.9，否则 0.7。
  - T_pass：voltage ≤150 或 condition A 时 0.9；否则 1.2。
- **判定**：
  - PASS：d − u > T_pass（下界值也可以）。
  - FAIL：d + u < T_fail，挡住的是观测到的表面，并且该表面属于一张可用卡片，或是被 ≥2 个视角子集看到的 TSDF 簇。挡的是 storage、goods、handling、furniture 族时，理由同时引用 (g)(1)(ii)。
  - 其余给 NEEDS_REVIEW，写明缺的是电压、条件还是建造年份。
  - 只看到侧面、没有前面：NO_DATA。
- **净高**：同一次扫描中，区内上方有伸出物且 base + u < 1.91 → FAIL（引用 (vi)）；base − u > 1.98 → 不构成问题。
- **看图问题**：无。
- **三段视频**：ME340 有 electrical enclosure 1 个，控制柜可能再有几个；卖场 0 个。这条规则主要面向车间。

#### E2 线盒、插座盖板〔新 · P3 · 只能看图〕
- **来源**：1910.305(b)(2)(i)：接线盒、分线盒要有盖；完工后每个出线盒都要有盖、面板或灯座罩。【法】核实：一手
- **适用**：electrical outlet、switch、junction box。
- **判定**：只能看图（V6）。给 NEEDS_REVIEW（有提示时）或 NO_DATA。

### 3.E 消防与应急设备

三段视频里这一族是 0 个对象。规则照写，覆盖要等检测到这些设备以后。

#### F1 灭火器安装高度〔新 · P2 · 可判〕
- **来源**：
  - NFPA 10 §6.1.3.8：
    - ≤ 40 lb 的灭火器，顶部离地 ≤ 5 ft（1.53 m）；
    - > 40 lb 的（推车式除外），顶部离地 ≤ 3.5 ft（1.07 m）；
    - 底部离地 ≥ 4 in（102 mm）。
    - 【规范】核实：二手（Fire Engineering，引 2013 版；UpCodes 的 NFPA 1 2021 摘要）
  - 29 CFR 1910.157(c)(1)：要安装、定位、标识，便于员工取用。没有数值。【法】核实：一手
- **适用**：fire extinguisher 类，且通过了 hazard_gate。height > 1.0 m 且落地的，按推车式处理，给 N/A。
- **字段**：top_above_floor、base_above_floor、height（均为现有）。
- **阈值**：
  - 顶部：T_fail = 1.53（任何重量都不能超过）。
  - T_pass = 1.07。如果站点允许"height + u ≤ 0.65 m 视为 ≤ 40 lb"，T_pass 取 1.53。这条是〔估，未核〕的尺寸推重量。
  - 底部：下限 0.102。
- **判定**：
  - 顶部 PASS：top + u < T_pass。
  - 顶部 FAIL：top − u > 1.53。
  - 底部 FAIL：base + u < 0.102（放在地上）。
  - 底部 PASS：base − u > 0.102。
  - 两部分按 worst 合并，同 J2 的合并方式。
- **看图问题**：无。V3（挂架还是柜里）不需要，底部高度已经能判。
- **尺度**：顶部 1.2-1.5 m 时 u 约 0.3-0.4，会常常跨阈值。这是尺度锚最直接的受益者之一。

#### F2 灭火器前方净空〔新 · P2 · 可判（站点阈值）〕
- **来源**：
  - NFPA 10：不得被遮挡。没有面积或距离。
  - 36 in（0.914 m）是多数 AHJ 借用 NFPA 70 §110.26 的惯例，不是条文。【站点】核实：二手（JJ Keller、Healthcare Facilities Today）
  - 1910.157(c)(1)"readily accessible"。【法】
- **适用**：fire extinguisher；同样用于 eyewash station、first aid kit、fire alarm。
- **字段**：front_clearance（新）。
- **判定**：
  - PASS：d − u > 0.914。
  - FAIL：d + u < 0.914，且挡物是可用的 movable、storage、goods 类卡片，不是墙或柱。
  - 其余给 NEEDS_REVIEW；没有前面时 NO_DATA。

#### F3 到灭火器的行走距离〔新 · P2 · 只能判 PASS〕
- **来源**：
  - 1910.157(d)(2)：A 类火灾，行走距离 ≤ 75 ft（22.9 m）。【法】核实：一手
  - (d)(4)：B 类危险区到灭火器 ≤ 50 ft（15.2 m）。【法】核实：一手
  - NFPA 10 §6.2.1：沿通道量，不是直线。【规范】核实：二手
- **适用**：
  - 区域行：每个镜头一行，按走过路径的采样点统计。
  - 物件行：B 类危险物，即 container 里的 jerry can / gas can、drum、flammables cabinet。
- **字段**：route_m（新）。沿走过路径的路程是真实最短路程的上界。
- **判定**：
  - PASS：route_up + u ≤ 22.9（B 类用 15.2）。
  - 视频看不到的地方可能有灭火器，所以"没看到"不能 FAIL。直线距离 − u > 阈值的路段给 NEEDS_REVIEW，写"沿 x m 路径未见灭火器"。
- **三段视频**：ME340 的区域行会是 NEEDS_REVIEW："走过 N m 没有确认到灭火器"。这本身有信息量：要么现场缺，要么检测漏了。

#### EW1 洗眼器〔新 · P3 · 部分可判〕
- **来源**：
  - 1910.151(c)：接触腐蚀性物质时，要在工作区内提供冲洗设施。没有数值。【法】核实：一手
  - ANSI Z358.1-2014：
    - 离危险源 10 s 内，约 55 ft（16.8 m）；
    - 出水峰高离站立面 33-53 in（0.84-1.35 m）；
    - 离墙或最近障碍 ≥ 6 in。
    - 【标准】核实：二手（大学 EHS 资料）
- **判定**：
  - 路程：route_up + u ≤ 16.8 → PASS。危险源位置要站点给，否则 N/A。
  - 前方净空：同 F2。
  - 出水高度：视频测不到，给 NO_DATA。

#### FA1 手动报警按钮〔新 · P3 · 可判但多为复核〕
- **来源**：NFPA 72 §17.14.8：操作部件离地 42-48 in（1.07-1.22 m）；每个出口门洞 60 in（1.52 m）以内要有一个。【规范】〔未核：只有搜索摘要，NFPA 博客正文取不到〕
- **判定**：
  - 高度：只在明显越界时 FAIL，即 top + u < 1.07 或 base − u > 1.22。区间窄，大多是 NEEDS_REVIEW。
  - 位置：到最近出口门的距离 + u ≤ 1.52 → PASS。出口门附近没看到报警按钮，不能 FAIL。

### 3.F 机器、工具、防护

#### M1 机器防护是否存在〔新 · P2 · 只能看图〕
- **来源**：1910.212(a)(1)：要提供机器防护，保护操作者和机器区域内其他人员。(a)(3)(ii)：加工点要有防护。(a)(3)(iv) 点名了铣床、电锯、压力机等。【法】核实：一手
- **适用**：machine 族的 milling machine、lathe、drill press、saw、grinder、press、cnc machine、machine。
- **判定**：只能看图（V1）。"疑似无防护"给 NEEDS_REVIEW 并提高优先级；其余给 NO_DATA，reason 写"防护无法从视频证明"。永远不 PASS。

#### M2 砂轮机托架间隙 ≤ 1/8 in、挡板间隙 ≤ 1/4 in〔新 · P3 · NOT_RESOLVABLE〕
- **来源**：1910.215(a)(4)：托架间隙最大 1/8 in。(b)(9)：上方挡板间隙 ≤ 1/4 in。【法】核实：一手
- **判定**：毫米级的量低于视频分辨率，永远给 NO_DATA/NOT_RESOLVABLE。V2 只回答"托架和挡板在不在"，作复核提示。

#### J6 可移动物离机器防护 ≥ 0.6 m〔保留，【站点】starter s01〕
- 判定不变。只在防护 1.5 m 以内时适用。

#### G1 防护围栏高度 ≥ 1.8 m〔新 · P2 · 可判（站点）〕
- **来源**：工位策略 `docs/policies/compiled_v2/p01.json`（min_height 1.8 m）。【站点】
  - ISO 13857 规定了围栏高度与到危险区距离的关系，本文没有核实具体数值。〔未核〕
- **适用**：guarding 族的 guard 和 fence，且离某台机器 ≤ 1.5 m。在卖场，fence 通常不是机器防护，给 N/A。
- **判定**：
  - PASS：height − u > 1.8。
  - FAIL：height + u < 1.8，且顶部没有被画面切掉，且 plain。
  - 其余给 NEEDS_REVIEW。

### 3.G 登高设备

三段视频里都是 0 个。

#### L1 梯子角度（修订 J7）〔修订 · P3 · 可判〕
- **来源**：
  - 29 CFR 1926.1053(b)(5)(i)（建筑业）：非自立梯的放置角度，使顶部支点到梯脚的水平距离约为工作长度的 1/4，即离竖直约 14.5°（75.5°）。【法（建筑业）】核实：一手
  - 通用工业的 1910.23 没有角度条款，只有 (c)(4)（稳定、水平面，或固定住）和 (c)(2)（人字梯撑杆锁住）。【法】核实：一手
- **现有 J7 的问题**：J7 规定"倾斜 ≤ 10°"，来自 starter 政策。这样一把正确靠放的延伸梯（14.5°）会被 FAIL。
- **修订后的判定**：
  - 靠放梯（梯顶离房间表面 ≤ 0.2 m）：PASS 当 |θ − 14.5°| + u ≤ 5°〔站点容差，估〕；FAIL 当 |θ − 14.5°| − u > 5°。
  - 人字梯：几何上不判，只看图（V4：是否完全打开、撑杆是否锁住）。

#### L2 通道或门口里的梯子〔新 · P3 · 只能判 PASS〕
- **来源**：1910.23(c)(7)：放在通道、门口、车道上的梯子，要固定住，或用锥桶、警示带隔开。【法】核实：一手
- **判定**：
  - 梯子 footprint 在走道或门口区内，且 1.5 m 内〔站点〕有 safety cone 或 barrier 卡 → PASS。
  - 否则给 NEEDS_REVIEW。"固定住"看不到，不能 FAIL。

#### L3 梯子侧轨高出上层平台 ≥ 3 ft〔新 · P3 · 可判〕
- **来源**：1910.23(c)(11)。【法】核实：一手
- **判定**：梯顶减平台面，这是同一视角下的相对量。PASS 当 −u 后仍 ≥ 0.9 m；FAIL 当 +u 后仍 < 0.9 m。

#### ST1-ST4 楼梯〔新 · P3 · 可判〕
全部来源都是【法】，核实：一手。
- **ST1** 宽度 ≥ 22 in（0.56 m），1910.25(c)(4)。
- **ST2** 踏步前缘上方净高 ≥ 6 ft 8 in，1910.25(b)(2)。用 O1 的算法。
- **ST3** 扶手高 30-38 in（0.76-0.97 m），从踏步前缘量，1910.29(f)(1)(i)。
- **ST4** 楼梯角度 30-50°，1910.25(c)(1)。前缘连线的坡度与尺度无关，可以判。
- 另外，≥3 个踏步且 ≥4 个踢面就要有扶手，1910.28(b)(11)(ii)。没看到扶手不能 FAIL，只能给 NEEDS_REVIEW。

#### G2 高处边缘护栏高度〔新 · P3 · 只能判 FAIL〕
- **来源**：
  - 1910.28(b)(1)(i)：离下层 ≥ 4 ft（1.2 m）的开放边缘要防坠。【法】核实：一手
  - 1910.29(b)(1)：护栏顶 42 ± 3 in；允许高于 45 in。【法】核实：一手
- **适用**：railing 或 guard，且位于 work platform 或 mezzanine 的边缘，平台高差 ≥ 1.2 m。卖场的推车栏杆等给 N/A。
- **判定**：
  - FAIL：rail_above_surface + u < 0.99（39 in）。
  - PASS：rail_above_surface − u ≥ 0.99。
  - 边缘没有护栏卡：给 NEEDS_REVIEW，不能 FAIL。

### 3.H 搬运设备

三段视频里 0 台叉车；pallet jack 的候选都被闸门拒绝了。

#### FL1 无人看管的叉车，货叉要落到底〔新 · P3 · 可判〕
- **来源**：1910.178(m)(5)(i)(ii)。操作者离车 25 ft（7.6 m）以上，或离车且看不见车，就算无人看管；这时货叉要完全放下、控制置空挡、断电、制动。【法】核实：一手
- **判定**：
  - FAIL：叉车静止的时段里，7.6 m 内的区域被看到 ≥ 80% 且没有人员轨迹，并且 forks_base − u > 0.10。
  - PASS：forks_base + u ≤ 0.10。
  - 其余给 NEEDS_REVIEW。

#### FL2 人员在升起的货叉或货物下方〔新 · P3 · 可判〕
- **来源**：1910.178(m)(2)。【法】核实：一手
- **判定**：forks_base − u > 0.3 时，人员 footprint 与货叉区相交 → FAIL。

#### FL3 人车间距〔保留 J8 的 R1-R3，【站点】〕
- 1910.178 没有固定的人车间距。沿用 `ehs_spatial.video` 的禁区、2.0 m 距离、1.5 m/s 速度规则，不变。

- **cart、pallet jack、hand truck** 停放：由 W1、J9、X1、E1、F2 的禁放区按位置判，不另设规则。

### 3.I 人员（保留）
- J3a（1910.22）、J3b（只作复核）、J8（R1-R3）不变。

### 3.J 卖场专项

#### CT1 收银台高度〔新 · P3 · 可判〕
- **来源**：ADA §904.4.1：至少有一段 ≥ 36 in 长、≤ 36 in（915 mm）高的台面。【法·ADA】核实：一手
- **适用**：checkout counter（misc 族）。
- **判定**：需要台面高度剖面。
  - PASS：存在一段 ≥ 0.915 m 长、top + u ≤ 0.915 的台面。
  - FAIL：整个台面 top − u > 0.915，且台面全长都看到了。
  - 其余给 NEEDS_REVIEW。

---

## 4. 族到规则的矩阵，以及覆盖目标

"今天"指 mvp3-judge-*-001 的数据：对象数 / 有行的 / 有结论的。只统计非人员卡。

| 族 | ME340 | Sam's Club | Walmart | 今天的规则 | 可判的新规则（优先级） | 只能看图 |
|---|---|---|---|---|---|---|
| goods | 19 / 5 / 5 | 572 / 372 / 96 | 616 / 78 / 44 | J1、J2、J5、J9 | S1（P1）、S2（P2）、P1（P2）、W1 零售阈值（P1） | — |
| storage | 29 / 0 / 0 | 21 / 3 / 3 | 53 / 16 / 14 | J5 | S1 固定件顶（P1）、R1 只判 FAIL（P2）、W1 | V8 |
| machine | 38 / 5 / 3 | — | — | J5、J9 | **H2**（P1）、E1 电气柜（P1） | V1、V2 |
| tool | 36 / 0 / 0 | 8 / 1 / 1 | — | — | **H1**（P1）、J9 | — |
| material | 36 / 0 / 0 | — | — | — | **H1**（P1）、J9 | — |
| linear | 33 / 15 / 3 | 2 / 0 / 0 | — | J4 | **O1**（P1，管、线槽、风管、悬挂线） | V10 |
| electrical | 12 / 0 / 0 | — | 1 / 0 / 0 | — | **E1**（P1）、**O1** 灯和风扇（P1） | V6 |
| building | 12 / 5 / 4 | 2 / 0 / 0 | — | J5 | X1、X2（P2，先修门的映射）；柱由 W1 管；地漏由 J9 管 | — |
| furniture | 8 / 2 / 1 | — | — | J5、J9、J3b | W1、J9、H2（workbench） | — |
| signage | — | 4 / 0 / 0 | 32 / 2 / 2 | — | **O1** 悬挂牌（P1）、X3 只判 FAIL（P3）；label 和 price tag 豁免 | V7 |
| guarding | 1 / 0 / 0 | — | — | J6 | G1（P2，站点） | — |
| misc | 2 / 1 / 1 | 1 / 1 / 0 | — | J5 | H1（rag、paper、wrap）、CT1（P3） | — |
| electronics | 1 / 0 / 0 | — | 1 / 0 / 0 | — | 豁免（自身没有危险），参与 W1 | — |
| safety / access / handling / hazard | 0 | 0 | 0 | J7 | F1-F3、EW1、FA1、L1-L3、ST、G2、FL1-FL2 | V3-V5、V9 |
| 无类 | 27 / 0 / 0 | 11 / 3 / 3 | 16 / 4 / 3 | — | 先修映射（§6） | — |

**覆盖目标**：
- 三段视频里出现的每个常见族，至少有一条标粗的 P1 规则能从几何、位置、类别判出结果。表中已经做到。
- 豁免类要显式列出理由，出 N/A 行，不再显示"no check applies"：
  - label、price tag、shelf talker：是固定件的信息部件；
  - window、wall panel、vent：属于结构，只作为通道边界；
  - monitor、computer、keyboard、printer、phone、clock、book、cup、whiteboard、curtain。

**只用现有字段的离线投影**（规则 H1、O1 代理版、P1 卡片位置版、S2）：

| 视频 | 有行的对象：今天 → 投影 | 有结论的对象：今天 → 投影 | 新增的行 |
|---|---|---|---|
| ME340 | 33 → 115 | 17 → 92 | H1 70 PASS / 5 NEEDS_REVIEW；O1 5 PASS；P1 33 NEEDS_REVIEW |
| Sam's Club | 380 → 454 | 103 → 110 | H1 7 PASS / 5 NEEDS_REVIEW；P1 158 NEEDS_REVIEW；S2 1 FAIL（就是碎片 obj-1-354；加上 u_B/B ≤ 0.5 闸门后为 0）/ 1 NEEDS_REVIEW / 16 NO_DATA |
| Walmart | 100 → 376 | 63 → 64 | P1 274 NEEDS_REVIEW；O1 1 NEEDS_REVIEW / 1 FAIL（代理版的假 FAIL，即 obj-1-482） |

怎么读这张表：
- 车间靠补规则就能涨。
- 卖场有结论的数几乎不动。要涨得靠 S1 的 ceiling_gap、P1 的 protrusion_m、S2 的第二边，以及 J1 的尺度锚。
- P1 和 O1 不能用代理字段上线。

---

## 5. 只能看图的问题（交给本地级联，从不单独定案）

每个问题都是对象裁剪图上的一对 SigLIP 2 文本提示，拿不准的聚类后交给本地 Qwen3-VL。上线前要有带标签的集合做校准，按 X8 set d 的做法，并且要人抽检。

| ID | 问题 | 服务的规则 | 为什么几何判不了 |
|---|---|---|---|
| V1 | 机器 [1] 的加工点（主轴、刀具、刃口）有没有防护罩或挡板？ | M1 | 防护是否"起作用"是语义问题 |
| V2 | 砂轮机有托架和上方挡板吗？ | M2 | 间隙是毫米级 |
| V4 | 人字梯完全打开、撑杆锁住了吗？ | L1 | 撑杆太细 |
| V5 | 地上有液体、油渍或湿滑区吗？ | SP1 | 液膜没有高度 |
| V6 | 插座、开关、线盒缺盖板或盖板破损吗？ | E2 | 盖板是薄件 |
| V7 | 出口标志是否点亮、可见？ | X3 | 亮度和状态 |
| V8 | 货架立柱或横梁有明显凹陷、弯折吗？ | R1 | 局部变形小于 u |
| V9 | 叉车货叉抬离地面了吗？ | FL1（在 forks_base 之前） | 没有部件分割 |
| V10 | 软线是否钉在或贴在墙、天花上，当固定线路用？ | 1910.305(g)(1)(iv)(A)(D)【法，核实：一手】 | 固定方式看不出 |

现有 calibration.json 里，q2、q3、q6、q7 的正样本都是 0（没有校准）。它们和 V1-V10 一样，只能用来排优先级。

---

## 6. 名字映射要先修（`cards.TAXONOMY` / `canonical()`）

以下都在 mvp3 最新代码上用 `canonical()` 跑过。

1. **"<物件> door"要归到物件本身**：cabinet door → cabinet，cnc machine door → cnc machine，lathe cabinet door → lathe，locker door → locker，cnc mill door → milling machine。
   - 做法：在 canonical() 里，把 `door` 前面有类名的情况当作 PART_WORDS 处理。
   - 只有这些保留为建筑门：door、roll up door、overhead door、doorway、exit door、fire door。
   - 这一步是 X1 和 X2 的前提。
2. **删掉 control panel 的裸同义词 `panel`**：
   - ceiling panel → not an object（或新增 ceiling tile）；
   - display rack panel → display rack（部件）。
3. **新类 `electrical panel`**（electrical 族），收电气类名字：electrical panel、breaker panel、panelboard、switchboard、fuse box、disconnect switch、motor control center、electrical enclosure、electrical cabinet、control cabinet、junction box、electrical box、switch box。
   - control panel 只保留操作件：control station、console、DRO、readout、pendant、controller。
   - machine 的同义词 `enclosure` 只保留 machine enclosure。
4. **`holder` 太贪**：sign holder → sign，shelf ticket channel → label。
5. **ME340 仍然无类的名字**：
   - tool bits、machinist jack → hand tool；
   - steadylast → machine（steady rest 已有）；
   - beaker → container；
   - metal ring、spacer、nozzle assembly → metal part；
   - rotary tool set → power tool；
   - toolholder stand → tool holder；
   - milling machine spindle head → milling machine：把 `head` 加进 PART_WORDS。
6. **卖场仍然无类的名字**：
   - divider bar → shelf；
   - strip merchandiser → display rack；
   - display box logo、picture of a person → not an object。
7. **规则需要的新类**：
   - `sprinkler`（sprinkler head、fire sprinkler），safety 族，用于 S1 直接量到溅水盘；
   - `electrical panel`（见第 3 条）；
   - 叉车部件 `forklift forks`（P3）。

---

## 7. 和照片工位策略的关系（复用什么）

- `panoptes-serving/ehs_spatial/policy.py` 现有七个谓词：min_separation、max_separation、keep_clear、not_inside、max_height、min_height、max_tilt。这是一个封闭词表，编译器（`scripts/policy_compile.py`）遇到表达不了的规则必须拒绝。
- 本库的规则按原则都能写成可审阅的 PolicySpec，但要补下面几个谓词。每个都是对 §2 某个字段的阈值比较：

| 新谓词 | 对应规则 |
|---|---|
| MIN_BASE_HEIGHT | O1，F1 的底部 |
| MAX_TOP_HEIGHT | F1 |
| MIN_FRONT_CLEARANCE（带方向） | E1、F2、X1 |
| MIN_GAP_ABOVE | S1 |
| MAX_RATIO | S2 |
| MAX_ROUTE_DISTANCE | F3、EW1 |
| MAX_PROTRUSION | P1 |

- 已有谓词能直接表达的：
  - E1 的"不得堆物"和 X1 的禁放区是 not_inside / keep_clear 对区域多边形；
  - G1 是 min_height；
  - J6 是 min_separation。
- 平台的 `ehs_spatial/platform/policy_engine.py` 模板已经收了 1910.22(a)(3)、1910.37(a)(3)、1910.36(d)(1)、1910.303(g)(1)(ii)、1910.212(a)(1)。本库在同样的条款上给出了视频里可以测的版本。
- 注意：平台引擎在尺度为 model_estimated 时返回 INSUFFICIENT_EVIDENCE。视频 judge 则按 u 判。两者口径不同，报告里要写明是哪条路径出的结论。
- `docs/policies/osha1910.md` 的 OSHA 考卷列出的缺口，在这里的对应：
  - 1910.36(g)(1) 净高 → O1；
  - 1910.159(c)(10) 竖直净空 → S1；
  - 1910.303(g)(1)(i)(B) 与设备相关的宽度 → E1 的扫描带宽；
  - 1910.36(g)(2) 通行宽度 → W1，已由 aisle() 实现。

---

## 8. 实施顺序（以三段视频上"有结论的对象数"排）

1. **§6 映射修正**。不改判定，只让规则落到对的对象上。
2. **H1**，只用现有字段；**N/A 行**和三档覆盖统计。
3. **O1**：从 aisle() 网格加 clear_under 和 over_aisle。
4. **H2**：zone_hits 和 low_clutter，同一网格。
5. **E1、F2**：front_clearance。这个扫描复用 aisle() 的侧向扫描。
6. **S1**：ceiling_gap。先在 Sam's Club 和 ME340 上验证天花层；Walmart 预期是 NO_DATA。
7. **W1 站点阈值**（retail 0.915）和**站点档案**。
8. P2：P1 的 protrusion_m、S2 的第二边或顶面深度下界、X1/X2、R1、F1/F3。
9. **尺度锚的验证**。与规则无关，但它决定 J1、F1、G1 能否出结论：先比较 Sam's Club 每个镜头的托盘尺寸是否一致。
10. P3：楼梯、护栏、叉车、洗眼器、报警按钮、收银台、W2、X3。

**验收**（每一步都要做）：
- 在 audit 表上看，新规则的 PASS/FAIL 有没有被 agent 标签推翻。标签要人抽检，见代码核对员更正。
- **没有假 PASS 是第一条。**
- 看复核队列增长多少。P1 用卡片位置时会加 400 多行复核，这正是不上线它的原因。

---

## 9. 来源（带核实状态）

**美国联邦法规原文。** 用 eCFR API 2026-08-01 快照逐字核对（`https://www.ecfr.gov/api/versioner/v1/full/2026-08-01/title-29.xml?part=1910&section=…`）。其中 1910.36、1910.37、1910.157、1910.159、1910.176、1910.212、1910.303 与 `panoptes-serving/tests/fixtures/oshacorpus/*.xml` 一致。
- 1910.22 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.22 ：(a)(1)(2)(3)
- 1910.23 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.23 ：(c)(2)(4)(7)(11)；没有梯子角度条款
- 1910.25 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.25 ：(b)(2)、(c)(1)(4)
- 1910.28 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.28 ：(b)(1)(i)、(b)(11)
- 1910.29 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.29 ：(b)(1)、(f)(1)
- 1910.36 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.36 ：(g)(1)(2)(4)
- 1910.37 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.37 ：(a)(3)、(b)(2)(6)(7)
- 1910.151 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.151 ：(c)
- 1910.157 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.157 ：(c)(1)、(d)(2)(4)
- 1910.159 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.159 ：(a)(1)、(b)、(c)(10)
- 1910.176 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.176 ：(a)(b)(c)
- 1910.178 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.178 ：(m)(2)(5)
- 1910.212 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.212 ：(a)(1)(3)、(b)
- 1910.215 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.215 ：(a)(4)、(b)(9)
- 1910.303 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.303 ：(g)(1)(i)-(vi)、Table S-1
- 1910.305 https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.305 ：(b)(2)(i)、(g)(1)(iv)
- 1926.1053 https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.1053 ：(b)(5)(i)
- 28 CFR 36.211 https://www.ecfr.gov/current/title-28/section-36.211

**ADA 2010 Standards。** 从 https://www.access-board.gov/ada/ 全文逐字核对：§203.9、§307.2、§307.3、§307.4、§403.5.1、§904.4.1。

**二手核实：**
- NFPA 10 安装高度：https://www.fireengineering.com/fire-safety/fire-extinguisher-installation/ ；https://up.codes/s/installation-height
- 灭火器前 36 in 是 AHJ 惯例：https://jjkellercompliancenetwork.com/regsense/how-much-open-space-must-be-kept-around-a-fire-extinguisher-so-it-is-not-considered-blocked-is-it-36 ；https://www.healthcarefacilitiestoday.com/posts/Fire-Safety-QA-Where-To-Place-Fire-Extinguishers--25454
- IFC 315.3.1：https://up.codes/s/ceiling-clearance ；https://www.pcfd.org/wp-content/uploads/2015/07/Inspection-Rational-38.pdf 。ICC 原页 https://codes.iccsafe.org/s/IFC2021P1/chapter-3-general-requirements/IFC2021P1-Pt02-Ch03-Sec315.3 返回 403。
- NFPA 13 ESFR 36 in：https://up.codes/s/clearance-from-deflector-to-storage
- ANSI MH16.1-2023：https://www.damotech.com/blog/ansi-mh-16.1-excerpts
- ANSI Z358.1-2014：https://ehs.tcu.edu/chemical/docs/ansi-eyewash-fact-sheet.pdf
- 南非 GSR 8：https://vula.uct.ac.za/access/content/group/c7716642-ada7-42a0-acee-0434c085df90/Electronic%20Resources/CD3/sow/Legislation/OHSA/gsr_8_stacking_of_articles.htm
- 通道宽度（1972 年解释信已撤回）：https://www.jjkellersafety.com/resources/articles/2023/does-osha-specify-a-minimum-aisle-width

**未核：**
- NFPA 72 §17.14.8：只有搜索摘要，https://www.nfpa.org/news-blogs-and-articles/blogs/2023/02/06/fire-alarm-pull-station-installation-height 正文取不到。
- 奥克兰大学的 4:1：https://cdn.auckland.ac.nz/assets/science/for/current-students/HR/health-safety-wellness/documents/SafestackingandStorage.pdf ，没有打开。
- ISO 13857 的数值。
- GMA 托盘尺寸。
- 喷淋头余量 0.30 m、门扇与净宽之差 0.05 m、H2 区半径 0.9 m、L1 容差 5°。这四个是〔估〕。
