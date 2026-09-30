# 粗处处 + 细按需：别人怎么做，我们用什么规则定

日期：2026-09-28

要回答的问题有四个：
- 别人是用算法做，还是用工程优化？
- 能不能靠工程规则来判定？
- "沿物体扩"时，怎么确定是这个物体？往哪里扩？
- 细长的东西是不是不该当 static？

标记约定：
- **[名字]**：来源，链接见文末第 5 节。文中数字都来自该来源的论文或官方仓库。
- **〔估〕**：我们自己的估计或设计，没有论文实测。
- **〔未核〕**：来源里提到，但没有核实到一手资料。

---

## 0. 先说结论

1. **别人的做法分两层。**
   - 感知层用模型：分割、深度、特征。
   - 判定层用人手设的规则：阈值、类别表、投票。
   - 这四件事都靠规则判定：哪里要细、是哪个物体、往哪扩、是搬走了还是被挡住。我们查过的实时系统全部如此。
2. **连最"算法"的方法，最后也落在人设的阈值上。** Clio 用信息瓶颈聚类，但仍然要设 α=0.23 和一个停止阈值 δ̄ [Clio]。
3. **我们的路子是主流。** 更准确的叫法是"工程规则判定"：用可测量的量加阈值做决定。它不只是"工程优化"（那是指让东西跑得更快）。这对 EHS 是好事：每个判断都能解释、能调、能测。
4. **"是这个物体"和"往哪扩"，都能用规则判。**
   - **是这个物体：** SAM 3 的实例 track 升到 3D 后，空间、语义、时间三票同时通过。
   - **往哪扩：**
     - 沿这个实例的 mask，加 2 倍体素的边距。
     - 沿表面：看法向和曲率。
     - 细长物：沿骨架走。
     - 时间上：只在残差超标或射线冲突时重新打开。
   - **停止边界：** mask 边 ∧ 深度跳变 ∧ 凹边。
5. **细长且可变形的东西不能当 static，你说得对。** 线缆、软管、绑带、链条、绳都属于这一类。文献里它们都按动态处理，每帧或每个时间窗重新估计形状 [TrackDLO][CDCPD2][Khronos]。
   - 但"细长"本身不是判据。护栏、固定管道、脚手架管也细长，却是刚性、固定的，可以当 static。只是它们要用线段来表示，3 cm 体素表示不了 [LIMAP]〔估〕。
   - 真正的判据是：会不会变形，会不会被挪动。
6. **行业 EHS 产品不做 3D。** 它们的做法是 2D 摄像头、手画区域、阈值加停留时间 [Protex][Intenseye][Visionify][Voxel]。需要真实距离时，它们加硬件：双目、雷达、可穿戴设备 [Intenseye-Sentinel][Blaxtair][Everguard]。
7. **最大的风险不在算法，在许可证。** 我们粗层用的 DA3-GIANT，权重是 CC BY-NC 4.0，不能商用 [DA3]。

---

## 1. 别人怎么做（按问题）

### 1.1 哪里要细？

每种判据都能测，都有阈值：

| 判据 | 谁在用 | 具体规则 |
|---|---|---|
| 类别表 | Panoptic Multi-TSDFs [PMT] | 复杂小物 2–4 cm，大物 5–10 cm，free space 30 cm |
| 类别表 + 几何复杂度 | MAP-ADAPT [MAPA] | 类别决定档位（8 / 4 / 1 cm）。曲率变化 CC=λ3/(λ1+λ2+λ3) > 0.05 升中档，> 0.1 升细档。合并回粗档要求标签置信 ≥ 0.95，且两个判据都同意 |
| 任务相关度 | Clio [Clio] | 用 CLIP 余弦衡量任务文本与物体的相关度，低于 α=0.23 算背景。合并到"再合并就会丢掉超过 δ̄ 的任务信息"为止 |
| SDF 方差 | MrHash [MrHash] | 从最细层开始积分，每块用 Welford 在线算方差，方差低的块合并到粗层 |
| 像素足迹（上限） | supereight [SE2] | 体素 ≈ 反投影像素大小 z/f，不再往下细 |
| 误差界 | wavemap [WM] | 误差界 ≥ 阈值才下降一层，已饱和的体素跳过 |
| 重建残差 | RTG-SLAM [RTG]、SplaTAM [SplaTAM] | RTG：只在 T>0.5、深度误差 >0.1 m 或颜色误差 >0.1 的像素新增，稳定的冻结。SplaTAM：轮廓 <0.5，或深度误差 >50 倍中位误差 |
| 图像密度 | WireSegHR [Wire] | 先在粗图上找线缆像素，只对密度超阈值的 patch 跑全分辨率，提速 2.3 倍 |

要点：
- "细够了"只有两个硬标准。一是像素足迹：比它更细，就是编出来的细节。二是误差或方差：再细也不会改变结果。
- 其余判据（类别表、任务相关度）回答的是"需要多细"，都由人来定。

### 1.2 是哪个物体？

通用做法是三票都要通过。

**① 空间票：新 mask 升到 3D 后，要和已有物体重叠。**
- PMT：渲染出的 mask IoU ≥ 0.1，且同类 [PMT]。
- PanopticFusion：IoU > 0.25 [PanFusion]。
- Fusion++：检测面积被覆盖 > 0.2 [F++]。
- ConceptGraphs：2.5 cm 内有最近邻的点所占比例，加上 CLIP 分，总分 > 1.1 [CG]。
- Voxblox++：≥ 20 个点落进同一个全局段的体素 [VB++]。
- SAM3D：重叠 > 0.5 × 较小的 mask。适合"一个是另一个的一部分"的情况 [SAM3D]。

**② 语义票：同类标签，或嵌入余弦过线。**
- Clio：≥ 0.7，同时 3D IoU ≥ 0.4 [Clio]。
- OVIR-3D：≥ 0.75，同时 IoU ≥ 0.25 [OVIR]。

**③ 时间 / 多视角票。**
- 连续跟踪 3 帧才建物体，连续 5 帧没见到就冻结 [PMT]。Khronos 要求 ≥ 15 次观测 [Khronos]。
- 多视角一致率 ≥ 0.9 才合并 [MaskClustering]。
- 或者有 > 5 个第三视角支持 [OnlineAnySeg]。支持的定义：c 与 a、b 各自的重叠 > 0.8，反向重叠 > 0.1。

我们已有的 SAM 3 视频 track ID，就是时间票的 2D 版本。Gaussian Grouping 用 DEVA 做的是同一件事 [GG]。

**两个挨着的箱子怎么分开：**
- **多视角：** 只要足够多的视角把它们分成两个 mask，就不合并。如果一个 mask 在超过 20% 的帧里同时覆盖多个已知物体，判为欠分割，丢掉 [MaskClustering]。
- **凹缝：** LCCP 在凹边断开，凹性容差 β=10° [LCCP]。
- **深度跳变：** 有断层就分开 [VB++]。
- **运动：** 其中一个动了就分开 [Khronos][PMT]。
- **例外：** 完全齐平、共面的堆叠箱，纯几何分不开 [LCCP]。只能靠跨视角的 mask 缝，或者整体当"一堆"处理。EHS 关心的多半也是堆高和占道〔估〕。

**两个坑：**
- 按类别做 Euclidean 聚类，会把挨着的同类物体粘成一个。Hydra 就是这么做的 [Hydra]，所以要加 mask 或凸性切分。
- 唯一把合并规则学出来的是 ESAM。它要用 ScanNet 训练，换了数据集效果明显下降 [ESAM][OnlineAnySeg]。

### 1.3 往哪扩？

**沿物体**
- 只在该实例 mask 点的 2ν 范围内分配细块（ν 是体素大小），不往未知区域自由生长 [PMT]。
- Fusion++ 的做法 [F++]：
  - 体积的中心和大小取点云的 p10–p90，再放大 1.5 倍。
  - 新检测覆盖到更多区域时，扩大体积。上限 128³ 体素、3 m。
- 需要"扩大一圈"时，用 CL-Splats 的办法：把变化区域聚类，套包围球（半径 × 1.1），只在球内优化 [CLS]。

**沿表面**
- 从已确认的体素或超点出发，做受限 region growing。一个邻居要同时满足四条才接纳：
  - **mask 支持：** 投影到能看到它的帧里，落在该物体的 mask 内。Open3DIS 用 IoU > 0.9，且特征余弦 > 0.9 [O3DIS]。
  - **没有深度跳变：** Voxblox++ 看相邻顶点之间的 3D 距离 [VB++]。
  - **不是凹边：** β=10° [LCCP]。
  - **法向夹角小：** PCL 教程用 3° [PCL]。
- 触发进一步细化：曲率变化 > 0.05 / 0.1 [MAPA]，或 SDF 方差高 [MrHash]。
- 平面和圆柱用局部 RANSAC 拟合，拟合出的形状决定往哪延伸。地面、货架板用平面，管道、立柱、扶手用圆柱 [RANSAC][Open3D]。
- mask 边缘"渗"出物体：OVO 对深度做低通，屏蔽边缘像素 [OVO]。

**沿细长物**
1. SAM 3 出 cable/hose/strap 的 mask，skeletonize 成中心线 [skimage]。
2. 交叉处选累计弯曲能量最小的配对，也就是直行的那根赢。纯几何，结果确定 [mBEST]。
3. 断开的片段接不接，看"端点距离 + 方向 + 曲率"的加权代价，低于阈值才接。论文明确说只看曲率会误连远处刚好方向对齐的段 [Keipour]。
4. 往哪扩：从每个端点沿切向开一个 ROI，在 ROI 里做高分辨率 crop，重跑分割，或者看 Frangi/Sato 的 ridge 响应。响应延续就继续扩〔估，组合自 [Wire][skimage]〕。
5. 停：碰到附着物 mask（插座、设备、卷盘、人手）、出画，或连续 k 步没有响应〔估〕。
- 刚性、直的细物（护栏、管道）可以做成 3D 线段 [LIMAP]，但 LIMAP 是离线的。

**沿时间**
- 已细化的区域，残差连续 N 帧低于阈值就冻结。只有新证据不一致时才重新打开：深度或颜色残差超标，或射线冲突 [RTG]。
- 被动视频不能让相机"去看"，所以"下一个最佳视角"换成"下一个最佳帧"。挑哪些帧做 crop，可以按覆盖（贪心选覆盖最多未覆盖体素的帧）[H2]，或按信息增益 [FisherRF]。

### 1.4 变化：搬走了还是被挡住？

核心是分清两件事：
- **有证据表明不在**（evidence of absence）。
- **只是没看到**（absence of evidence）。

**Khronos 的射线规则 [Khronos]：**
- 用背景射线检查物体的表面点：
  - 射线穿过物体所在位置，打到更远处：看穿了，算"不在"的证据。
  - 落在顶点 30 cm 以内：还在。
  - 测到的比物体更近：被挡，不算证据。
- 5 s 窗口内 ≥ 60% 的射线说"不在"，才判定不在。
- 出现或消失的时间，取"最后一次空"和"第一次在"的中点。
- 限制：
  - 搬动不做关联，会显示成"删一个 + 增一个"。
  - 物体后面必须有表面，开阔处效果差。
  - 精度只有 31.3%（GT 位姿）和 21.3%（Kimera 位姿）。

**Panoptic Multi-TSDFs [PMT]：**
- 把旧子图的表面点放到当前子图里查 SDF：
  - |sdf| < ν：一致。
  - sdf < −ν：相交。
  - sdf > ν：与 free space 冲突。
- 加权计数 > 20 个点或 > 2%，判为 matching 或 conflicting。
- 每个子图有三种状态：persistent、absent、unobserved。unobserved 的不删除。

**Dynablox [Dynablox]：**
- 一个体素和它所有邻居都被看到、并且连续 5 帧都空，才算"高置信 free"。
- 落进这种体素的点就是动态种子，向邻近的低置信点生长。小于 20 个体素的簇丢掉。
- 不需要类别。在笔记本 CPU 上 17.2 FPS。
- 失效情况：比位姿漂移还慢的移动检测不到 [nvblox-docs]；细而稀疏的物体也会漏掉 [Dynablox]。

**搬动还是新增：**
- SceneDiff 先做可见性门：在另一次拍摄里超过 50% 不可见的区域，不下"移除"的结论。然后用 DINOv3 外观余弦 > 0.7 做一对一匹配，匹配上的判 moved，匹配不上的判 added 或 removed [SceneDiff]。
- POCD 给每个物体一个 Beta 计数，外加类别可动性表，例如机器人记 0、货架记 1 [POCD]。

**还在不在（信念）：**
- Persistence Filter：按类别设存活先验（hazard rate），加上漏检率 P_M 和虚警率 P_F，做递推贝叶斯更新，每次常数时间 [PersF]。
- Perpetua 在此基础上还能处理"重新出现" [Perpetua]。

**提醒：变化检测的精度都不高。** SceneDiff 的物体 AP 只有 22.8% 和 10.6% [SceneDiff]，Khronos 的精度 21–31% [Khronos]。所以变化结论先当候选，再交给 VLM 或人确认。

### 1.5 行业 EHS 产品怎么做

**通用模式：**
- 输入是现有的 2D CCTV。
- 在每个摄像头画面上画多边形区域（走道、车道、出口），设阈值（距离、速度、停留时间），再加排除区 [Protex][Intenseye][Visionify]。
- Intenseye 说每个场景只需设 3–4 个参数 [Intenseye]。

**具体案例：**
- **吊装下方站人：** Intenseye 的专利用一个学出来的回归，把 bbox 映射到地面投影点，人用脚点表示，距离低于阈值就报警。专利明确说不用深度，也不做 2D-3D 对应 [Intenseye-patent]。
- **机器旁的危险行为：** SAS 的专利在多边形区域内看像素差，判断机器在不在动，再结合 2D 人体姿态 [SAS]。

**真 3D 只出现在专用硬件里：**
- Intenseye Sentinel：双目相机生成度量"危险平面"，用于机器防护 [Intenseye-Sentinel]。
- Blaxtair：车载双目 [Blaxtair]，部分参数〔未核〕。
- Everguard：雷达加定位可穿戴设备 [Everguard]。

**公开资料里没有的：**
- 没有厂商公开堆高的测量方法或精度 [viAct]。
- 没有厂商公开做线缆 3D [Intenseye][Protex]。

**精度宣称都没有独立验证：**
- Intenseye：95% 是报警精度的目标，靠现场调参达到，没有召回率 [Intenseye-FAQ]。
- Voxel：宣称"95%+"，同时说用"5 billion hours"训练，约合 57 万年视频，不可信 [Voxel]。
- 2025 年一篇综述看了 10 个商用系统，都没有定量验证 [OSH-review]。

**对我们的意义〔估〕：** 用规则做判定是行业常规。我们的差异在两点：一是度量 3D（相机高度给出尺度），支持移动相机；二是用 1 倍速回放，按事实类型实测 precision 和 recall。

---

## 2. 哪些是算法，哪些是工程规则

| 环节 | 算法（感知 + 数学） | 工程规则（我们定阈值） |
|---|---|---|
| 分割 / 候选 | SAM 3、CLIP / VLM | 词表、类别表 |
| 深度 / 位姿 | DA3、PromptDA | 对齐到粗深度时用哪片置信区 |
| 融合 | TSDF 积分、RANSAC 拟合 | 体素大小、分配边距 2ν |
| 同一物体 | — | IoU、重叠率、余弦、帧数阈值 |
| 往哪扩 | 骨架、法向、曲率、SDF 方差的计算 | 接纳条件、停止条件 |
| 变化 | 射线投射 / z-buffer | 30 cm 带、60%、5 s |
| 细长物跟踪 | CPD / GMM-EM 配准、Motion Coherence | 总长不变、拉伸 ≤ 1.1、锚点固定 |
| 还在不在 | 贝叶斯递推 | 按类别的存活先验 |

**可以直接采用的判据。** 下面都是起始值，全部要在我们自己的素材上重新标定：

| # | 用途 | 判据 | 起始值 | 来源 |
|---|---|---|---|---|
| 1 | 建物体 | 连续跟踪帧数 | ≥ 3 帧 | [PMT] |
| 2 | 冻结物体 | 连续未检测帧数 | 5 帧 | [PMT] |
| 3 | 2D→3D 关联 | 渲染 mask IoU + 同类 | ≥ 0.1 / > 0.25 | [PMT] / [PanFusion] |
| 4 | 语义一致 | 嵌入余弦 | ≥ 0.7 / ≥ 0.75 | [Clio] / [OVIR] |
| 5 | 合并两个实例 | 第三视角支持数，或相似度和 | > 5，或 > 2.3 | [OnlineAnySeg] |
| 6 | 欠分割 | mask 跨越多个物体的帧占比 | > 0.2 则丢弃 | [MaskClustering] |
| 7 | 部分包含 | 重叠 / 较小 mask | > 0.5 | [SAM3D] |
| 8 | 细块分配范围 | 到实例 mask 点的距离 | ≤ 2ν | [PMT] |
| 9 | 类别分辨率 | 小物 / 大物 / free | 2–4 / 5–10 / 30 cm | [PMT] |
| 10 | 表面细化触发 | 曲率变化 CC | > 0.05 中档；> 0.1 细档 | [MAPA] |
| 11 | 合并回粗档 | 标签置信 | ≥ 0.95，且两个判据同意 | [MAPA] |
| 12 | 表面生长 | 法向夹角 | 3°（DA3 噪声下要放大〔估〕） | [PCL] |
| 13 | 凹边切分 | 凹性容差 β | 10° | [LCCP] |
| 14 | 细度下限 | 体素 ≥ 像素足迹 z/f_eff | — | [SE2] |
| 15 | 细化稳定 | 深度残差 0.1 m、颜色残差 0.1 | 连续 N 帧低于阈值则冻结，超过则重开 | [RTG] |
| 16 | 缺失几何 | 轮廓，或深度误差 | < 0.5，或 > 50 倍中位误差 | [SplaTAM] |
| 17 | 不在 vs 被挡 | 深度带；投票 | 30 cm；5 s 窗口内 ≥ 60% | [Khronos] |
| 18 | 子图变化 | 冲突点数 | > 20 点，或 > 2% | [PMT] |
| 19 | 无类别动态检测 | 自身和邻居都 free；簇大小 | 连续 5 帧；≥ 20 体素 | [Dynablox] |
| 20 | 搬动 | 外观余弦；可见性门 | > 0.7；另一侧超过 50% 不可见则不判 | [SceneDiff] |
| 21 | 线缆交叉 | 累计弯曲能量 | 取最小 | [mBEST] |
| 22 | 线缆接段 | 距离 + 方向 + 曲率的加权代价 | 低于阈值 | [Keipour] |
| 23 | 线缆拉伸 | 边长 / 静止长度 | ≤ 1.1 | [CDCPD2] |
| 24 | 物体存在 | Beta 期望 E[p] | < 0.1 则删除 | [F++] |

---

## 3. 推荐设计（我们的管线）

### 3.1 "细"怎么定义

**fine_voxel = max(事实容差 / 2, 像素足迹 z / f_eff)**〔估：这是我们组合的规则，其中像素足迹来自 [SE2]〕

- **事实容差**来自 EHS 规则本身，例如"线缆离地 < 几 cm"、"通道宽 ≥ X"〔估〕。
- **像素足迹**的例子〔估，假设画面宽 1920 px、f≈1000 px〕：
  - DA3 输入长边 504 px 时，f_eff ≈ 262 px，5 m 处 1 px ≈ 1.9 cm。这意味着在 5 m 处做 1 cm 体素，细节是编出来的。
  - 同理，现有的 3 cm 粗体素，在约 8 m 以外也已经没有像素支撑。
  - 从画面里切一块 480 px 宽的 crop，放到 504 px 重跑，f_eff ≈ 1050 px，5 m 处约 5 mm/px。
- **停止条件**，满足任一条就停〔估〕：
  - 体素 ≤ 容差 / 2；
  - 体素 ≤ 像素足迹；
  - 残差和 SDF 方差连续 N 帧低于阈值。
- **起始档位**（借 [PMT][MAPA] 的档位思路）：
  - 粗：3 cm，就是现有的。
  - 中：1–2 cm。
  - 细：0.5–1 cm，只在 crop 深度能支撑时使用。
- **代码复用：**
  - supereight2（BSD-3）只借规则。
  - MAP-ADAPT（MIT）只借规则。它在 CPU 上 139–189 ms/帧，太慢，要在 GPU 上重写 [MAPA]。
  - MrHash 只借思路。代码含 GPL-3 和 Inria 非商用部分 [MrHash]。

### 3.2 物体范围规则

- **物体的定义：** SAM 3 track 升到 3D，通过三票（见 1.2 节）。
- **细子图：** 每个与事实相关的实例单独一个子图，只在 mask 点 2ν 以内分配块，体素大小按类别表 [PMT]。
- **怎么判"与事实相关"：** 先用 A〔估〕。
  - A. 类别表：词表里每个类别标上"细 / 中 / 粗"和可动性 [PMT][MAPA]。
  - B. 任务相关度：EHS 事实文本与实例的相关度 > α，Clio 用 0.23 [Clio]。Clio 每帧 0.22–0.31 s（RTX 3090），只能降频跑。
- **边界：** mask 边 ∧ 深度跳变 ∧ 凹边 [VB++][LCCP][OVO]。
- **合并与拆分：**
  - SAM 3 的 track ID 作硬链接。
  - 再加体素哈希重叠率和第三视角支持 [OnlineAnySeg]，以及欠分割过滤 [MaskClustering]。
  - 后台定期清理：删掉很少被确认的点，按包含关系合并，用 DBSCAN 拆开不连通的部分 [OVIR]。
- **代码复用与延迟：**
  - PMT 代码是 BSD-3，可商用，但它是 ROS1 + CPU，要在我们的 GPU TSDF 上按原设计重写。源实测：640×480、笔记本 CPU 上 5.1–6.5 FPS，不含分割 [PMT]。重写后的目标：关联加分配每关键帧 ≤ 20 ms〔估，需实测〕。
  - OnlineAnySeg 和 MaskClustering 没有许可证文件，只能按论文重写。源实测：OnlineAnySeg 在 RTX 4090 上 15 FPS，是否含 CropFormer 时间〔未核〕。
  - nvblox（Apache-2.0）可以直接做粗层和动态层。源实测：5 cm 体素、3090 Ti 上 TSDF 0.4 ms/帧 [nvblox]。

**例子：一个托盘怎样被确定、被扩展〔估〕**
1. 第 1–3 帧，SAM 3 给出 pallet track #12。它的 mask 升到 3D 后，和已有物体都不重叠，于是第 3 帧建新物体 O7。
2. pallet 在类别表里是"中档，可移动刚体"，于是在 mask 点的 2ν 以内分配 1–2 cm 的细块。
3. 第 40 帧从侧面看到更多，mask 变大。新点和 O7 重叠，而且是同一个 track，于是扩到新点的 2ν 以内。
4. 旁边的地面有深度跳变和凹边，不扩过去。
5. 第 200 帧有人走到托盘前面。把 O7 的点投影到画面里，测到的深度更近，说明被挡了。O7 标为 unobserved，不删除。
6. 第 400 帧那个位置又能看见了，但测到的深度比 O7 远（看穿了），而且 5 s 内 ≥ 60% 的点都是这样，于是 O7 成为 absent 候选。如果别处出现一个外观余弦 > 0.7 的新托盘，就成为 moved 候选。两种候选都交给 VLM 确认。

### 3.3 扩展规则

| 方向 | 规则 | 停止条件 | 借鉴 | 代码与许可 | 延迟（源实测 / 我们估计） |
|---|---|---|---|---|---|
| 沿物体 | 实例 mask 点 2ν 以内；新检测覆盖更多就扩大 | mask 边、深度跳变 | [PMT][F++] | PMT 是 BSD-3，需重写；F++ 没有代码 | 源：PMT 5–6.5 FPS（CPU）/ 〔估〕≤ 20 ms 每关键帧 |
| 沿表面 | 受限 region growing（mask 支持、无深度跳变、非凹、法向一致）；CC > 0.05 / 0.1 或 SDF 方差高才细化；平面和圆柱用 RANSAC | 凹边、深度边、mask 边 | [PCL][LCCP][MAPA][MrHash][Open3D] | PCL（BSD）、Open3D（MIT）可直接用；CGAL 是 GPL，不用 | 源：LCCP 470 ms/帧（CPU，95% 花在超体素）/ 〔估〕只在 ROI 内跑 GPU 版，< 30 ms |
| 沿细长物 | 骨架 → 交叉处最小弯曲能量 → 端点切向 ROI + ridge 响应 → 接段代价 | 附着物 mask、出画、k 步无响应 | [mBEST][Keipour][Wire][skimage] | mBEST 是 GPL-3，自己重写（算法很小）；scikit-image（BSD-3）直接用 | 源：mBEST 30.6–39.6 FPS（含分割网络，2080 Ti）/ 〔估〕用 SAM 3 mask 代替分割网络后，骨架加交叉 < 10 ms 每个 ROI |
| 沿时间 | 稳定就冻结；残差超阈或射线冲突就重开；挑帧按覆盖或信息增益 | 连续 N 帧稳定 | [RTG][H2][FisherRF] | RTG 和 H2 是 GPL-3，FisherRF 是 Inria 非商用，都只借思路 | 源：RTG 整个系统 17.9 FPS（4090）/ 〔估〕状态机开销可忽略 |

**线缆怎样变成 3D〔估〕**
- **地上的线缆：** 骨架像素的射线与地面平面求交，尺度由相机高度给出。这样不依赖 DA3 在细结构上的深度，后者我们还没验证。
- **悬挂的线缆：** 用两端锚点，加多帧数据，拟合 5 参数悬链线 [Catenary]。这个方法没找到代码，运行时间〔未核〕。
- **线缆不进 TSDF。** 原因有两个：3 cm 体素比线径粗〔估〕；Dynablox 对细而稀疏的物体会失效 [Dynablox]。

**细 ROI 的深度从哪里来**
- **A. DA3 crop。**
  - DA3 支持以位姿为条件的深度，并输出置信度 [DA3]。
  - 具体做法〔估〕：平移主点，按缩放比例缩放 f，喂入粗层位姿，然后在高置信的重叠区域把尺度和偏移对齐到粗深度。
  - 源实测（A100，504×336）：DA3-Small 160.5 FPS（约 6 ms 一个 crop），Base 126.5 FPS。这两个都是 Apache-2.0 [DA3]。
- **B. PromptDA。**
  - 把粗 TSDF 渲染出的深度或 DA3 深度作为提示，在全分辨率 crop 上跑 [PromptDA]。
  - 源实测（A100，768×1024）：ViT-S 80 FPS（约 12.5 ms 一个 crop），ViT-L 20.4 FPS。代码和权重都是 Apache-2.0。
  - 风险：它训练时用的提示是真实 LiDAR。
- **不用的：**
  - Depth Pro：权重仅限研究使用，0.3 s 一张 [DepthPro]。
  - PatchFusion：4K 图 1.56–7.82 s 一张 [PatchFusion]。
  - BoostingMonocularDepth：仅限学术使用 [BMD]。

### 3.4 可动性分四类，各自的时间规则

| 类 | 例子 | 表示方式 | 进静态 TSDF？ | 时间规则 | 借鉴 |
|---|---|---|---|---|---|
| 固定 | 地面、墙、柱、固定货架、护栏、固定管道 | 粗 TSDF；细长刚性物另加 3D 线段 | 进 | 很少重查；由 ever-free 冲突（Dynablox 式）触发重查；存在信念按天衰减〔估〕 | [PMT][Dynablox][LIMAP] |
| 可移动刚体 | 托盘、箱子、推车、梯子、隔离栏；停着的叉车〔估〕 | 每个实例一个子图 | 不进，用自己的子图 | 三种状态 persistent / absent / unobserved；看得见时做射线投票（5 s 内 ≥ 60%）；搬动 = 外观余弦 > 0.7，且类别和尺寸一致；存在信念按分钟到小时衰减〔估〕 | [PMT][Khronos][SceneDiff][PersF] |
| 可变形 | 线缆、软管、绑带、链条、绳 | 每个时间窗一条带时间戳的 polyline（节点链） | 永远不进 | 每个窗口重新检测；身份靠锚点、总长度和外观；看得见却没检测到，判 moved 或 removed；被挡住，判 unknown；存在信念按秒衰减〔估〕 | [TrackDLO][CDCPD2][MultiDLO][Khronos] |
| 主体 | 人、行驶中的叉车和车辆 | 轨迹，加一个会衰减的占用层 | 永远不进 | 每帧更新；离开后占用逐渐衰减 | [nvblox][Dynablox] |

- **"细长"不是分类依据。** 分类看的是会不会变形、会不会被挪动。护栏细长但固定，放"固定"类；线缆细长又会变形，放"可变形"类。
- **叉车：** 停着时按可移动刚体处理，一开动就转为主体〔估〕。

**可变形类的时间规则（细化）**
- **总长度不变：** 可见长度变短，说明被挡住了，不是线变短了，也不是被拿走了 [TrackDLO][CDCPD2]。
- **锚点固定：** 插头还插在同一个插座上，就是同一根线。这对应 CDCPD2 的 known correspondence 约束 [CDCPD2]。
- **可见性检验〔估，组合自 [Khronos][TrackDLO]〕：** 把上一个窗口的 polyline 投影到当前帧，逐段判断：
  - 这段在画面内、没被人或物体的 mask 挡住、前面也没有更近的表面，却没检测到线缆：判 moved 或 removed。
  - 锚点没变，但地面上的走向变了：判 moved。
- **被挡住的节点：** 跟随看得见的邻居一起移动（Motion Coherence）[TrackDLO]，或者保持上一次的状态，并标 unknown。
- **不做逐帧跟踪。** TrackDLO 假设帧间位移很小，二手来源给出的上限约 5 mm/帧 @30 fps〔未核一手〕。所以我们改为每个窗口重新检测，再做关联。窗长先取 1 s〔估〕。
- **EHS 事实的判定〔估，规则结构与 [Intenseye][Protex] 相同〕：** polyline 穿过走道多边形，离地 < 几 cm，持续时间超过停留阈值。事实里要引用具体的帧和时间。

**例子：线缆 C3〔估〕**
1. SAM 3 给出 cable 的 mask，做成骨架，得到两个端点。
2. 一端碰到插座 mask，记为锚点 A。另一端到了画面边缘，出画，停。
3. 骨架像素与地面求交，得到 3D polyline，长度记为 L。
4. 下一个窗口：同一个插座上有线，外观也相近，所以还是 C3。可见长度 < L，缺的那段被叉车的 mask 挡着，所以是被挡住，不是变短。
5. 如果缺的那段在画面内、没被挡、地面上却没有线，就判那段 moved。如果整根都是这样，连锚点处也没有了，就是 removed 候选。
6. 最终事实写成："C3 在 t1–t2 穿过走道 W1，离地 < 2 cm，持续 45 s"，并引用具体的帧。

### 3.5 实时队列（1 倍速回放）

以下分层和预算都是〔估〕，要在我们自己的 GPU 上实测。源数字多数是在 A100、4090 或笔记本 CPU 上测的。

**L0：每帧，硬实时**
- SAM 3 跟踪，包括人。用时我们还没测。
- DA3 粗深度和位姿（现有的，按 shot 跑）。
- 粗 TSDF 积分。源实测：nvblox 5 cm 体素 0.4 ms [nvblox]。3 cm 会慢一些〔估〕。
- ever-free 位更新（Dynablox）。源实测 17.2 FPS（CPU）；nvblox 里有 GPU 版 [Dynablox][nvblox]。

**L1：每关键帧，目标 3–5 Hz〔估〕**
- 实例升 3D，做三票关联和子图分配（PMT 重写）。
- 可见性和射线变化检测。源实测：Khronos 活动窗 45.5 ms（CPU），变化检测加调和 < 100 ms [Khronos]。
- 线缆：骨架、交叉、接段、地面求交。源实测：mBEST 30–40 FPS [mBEST]。
- 事实相关度：先用类别表，开销几乎为零。Clio 式打分要 0.22–0.31 s/帧，只能降频 [Clio]。

**L2：细化作业，有预算，可以丢或推迟**
- 优先级 = 事实相关度 × (不确定性，或刚发生变化) / 代价〔估〕。
- 作业内容：DA3 crop 或 PromptDA crop 出深度，积分进细子图，再做表面细化。
- 每帧给 L2 一个固定的 GPU 时间预算，即帧间隔减去 L0 的用时。超了就留到下一帧〔估〕。
- 同一区域只保留最新的一个作业，新帧顶替旧作业（latest-wins）〔估〕。
- 区域稳定后冻结，不再排队 [RTG]。
- 单个 crop 的源实测：DA3-Small 约 6 ms，PromptDA ViT-S 约 12.5 ms，都在 A100 上 [DA3][PromptDA]。

**L3：后台，不保证实时**
- 实例维护：OVIR 式清理，源设定每 300 帧一次 [OVIR]。
- MaskClustering 式的一致率合并。
- 悬链线拟合、LIMAP 线段。
- 用 VLM 确认变化候选。

**代码与许可证汇总**

| 处理方式 | 组件 |
|---|---|
| 直接用（可商用） | nvblox（Apache-2.0）；PromptDA 代码和权重（Apache-2.0）；DA3-BASE / SMALL / METRIC-LARGE / MONO-LARGE 权重（Apache-2.0）；Open3D（MIT）；PCL（BSD）；scikit-image（BSD-3）。可选：HQ-SAM、tapnet、gsplat（均 Apache-2.0），Perpetua（MIT） |
| 按原设计重写（代码可商用，但是 ROS / CPU 栈） | PMT、Khronos、Dynablox、supereight2（BSD-3）；Clio（BSD-2）；MAP-ADAPT、TrackDLO、MultiDLO、OVIR-3D、OVO（MIT） |
| 只能照论文重写（没有许可证或有限制） | OnlineAnySeg、MaskClustering、CDCPD2、RT-DLO、FASTDLO（没有许可证文件）；mBEST、Persistence Filter、RTG-SLAM、H2-Mapping（GPL-3）；MrHash（GPL-3 + Inria）；CGAL（GPL）；PEAC（MERL 非商用） |
| 不能用于商用 | DA3-GIANT / LARGE-1.1 / NESTED 权重（CC BY-NC 4.0）；Depth Pro 权重（仅研究）；CoTracker（CC BY-NC）；pi3 权重（CC BY-NC）；Inria 3DGS 派生代码（H3DGS、Octree-GS、FisherRF、CL-Splats 的一部分、Gaussian Grouping 子模块）；BoostingMonocularDepth |
| 注意条款 | SAM 3 用的 SAM License 允许商用，但有出口管制和军事用途限制 [SAM3-LIC] |

---

## 4. 未解决的风险

1. **许可证。** DA3-GIANT 权重是 CC BY-NC 4.0，DA3-LARGE-1.1 和 DA3NESTED 也一样 [DA3]，而粗层现在用的就是 GIANT。商用有两条路：找 ByteDance 授权，或者换成 Apache-2.0 的 BASE、SMALL、METRIC-LARGE 或 MONO-LARGE。换了模型要重测精度〔估〕。
2. **位姿和尺度不一致。**
   - DA3 按 shot 做前馈推理，不同 shot 之间尺度和位姿会有偏差。
   - 论文里以 cm 计的阈值，大多来自 RGB-D 传感器，必须按我们实测的深度残差重新标定。
   - 建议用体素相对容差（ξ=ν）[PMT]，边距随深度增大〔估〕。
   - 标不好的后果：一个物体被切成几个，或者相邻的物体被粘成一个。
3. **慢速移动测不到。** 比位姿漂移还慢的移动检测不出来 [nvblox-docs]。需要加一道位姿质量门〔估〕。
4. **变化检测精度低。** Khronos 21–31% [Khronos]，SceneDiff AP 10.6–22.8% [SceneDiff]。变化结论只能当候选。
5. **开阔场景。** 射线法要求物体后面有表面 [Khronos]。户外堆场或大空间里，可能判不了"不在"。
6. **细结构的深度。**
   - DA3 在线缆上的深度质量没有验证过。
   - 3 cm TSDF 会把线缆抹掉〔估〕。
   - SAM 3 和 HQ-SAM 在线缆上谁更好，没有验证过。
   - 地面求交的前提是线躺在地上，搭在箱子上的线不适用〔估〕。
7. **线缆规则会失效的场景：** 同色线缆成束交叉，线宽不到约 2 px，透明或反光的软管，快速拖动〔估，部分依据 [mBEST][FASTDLO][TrackDLO] 的局限说明〕。
8. **齐平的堆叠箱。** 纯几何分不开 [LCCP]。
9. **PromptDA 的提示域差异。** 它训练时的提示是真实 LiDAR，换成渲染深度或前馈深度后效果未知 [PromptDA]。
10. **GPU 争用。** 如果 L0 每帧都跑 DA3-GIANT，A100 上约 26.6 ms/帧（1/37.6 FPS）〔估，据 [DA3]〕，30 fps 下只剩约 7 ms 给其他任务。SAM 3 的用时我们还没测。L2 可能长期拿不到预算。换成 Small 或 Base，能同时缓解许可证和速度两个问题〔估〕。
11. **阈值多，而且互相耦合。** 没有我们自己的标注回放集，就调不准〔估〕。建议每条规则都有开关和日志，用 1 倍速回放按事实类型测 precision 和 recall。
12. **生成出来的遮挡部分。** SAM 3D Objects 推断的被遮挡部分是生成的，只能标"推断"，不能当"观测" [SAM3DObj]，速度和许可〔未核〕。
13. **未核实项：**
    - Fusion++ 和 PanopticFusion 是否真的没有官方代码。
    - OnlineAnySeg 的 15 FPS 是否包含 CropFormer。
    - Voxblox++ 的 τ_p 取值（论文没给）。
    - MultiDLO 和 CGGT 的实际速度。
    - 悬链线方法的代码和运行时间。
    - TrackDLO 位移上限的一手来源。
    - SAGA 子模块和 sam-3d-objects 的许可证。
    - RTG-SLAM 的光栅器是否带 Inria 许可。
    - 各厂商的延迟和精度数字。

---

## 5. 来源

- [Clio] https://arxiv.org/abs/2404.13696 · https://github.com/MIT-SPARK/Clio
- [PMT] Panoptic Multi-TSDFs：https://arxiv.org/abs/2109.10165 · https://github.com/ethz-asl/panoptic_mapping
- [Khronos] https://arxiv.org/abs/2402.13817 · https://github.com/MIT-SPARK/Khronos
- [Dynablox] https://arxiv.org/abs/2304.10049 · https://github.com/ethz-asl/dynablox
- [nvblox] https://arxiv.org/html/2311.00626 · https://github.com/nvidia-isaac/nvblox
- [nvblox-docs] https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/technical_details.html
- [MAPA] MAP-ADAPT：https://arxiv.org/abs/2406.05849 · https://github.com/GradientSpaces/MAP-ADAPT
- [MrHash] https://arxiv.org/abs/2511.21459 · https://github.com/rvp-group/mrhash
- [SE2] supereight：https://github.com/smartroboticslab/supereight2 · https://www.doc.ic.ac.uk/~sleutene/publications/Vespa_3DV19.pdf
- [WM] wavemap：https://github.com/ethz-asl/wavemap · https://arxiv.org/pdf/2306.01279
- [RTG] RTG-SLAM：https://arxiv.org/html/2404.19706v1 · https://github.com/MisEty/RTG-SLAM
- [SplaTAM] https://arxiv.org/html/2312.02126
- [H2] H2-Mapping：https://arxiv.org/html/2306.03207
- [FisherRF] https://arxiv.org/abs/2311.17874
- [DA3] https://arxiv.org/html/2511.10647 · https://github.com/ByteDance-Seed/Depth-Anything-3 · https://huggingface.co/depth-anything/DA3-GIANT-1.1
- [PromptDA] https://arxiv.org/html/2412.14015 · https://github.com/DepthAnything/PromptDA
- [DepthPro] https://huggingface.co/apple/DepthPro
- [PatchFusion] https://arxiv.org/html/2312.02284
- [BMD] https://github.com/compphoto/BoostingMonocularDepth
- [CG] ConceptGraphs：https://arxiv.org/html/2309.16650
- [Hydra] https://arxiv.org/pdf/2201.13360
- [VB++] Voxblox++：https://arxiv.org/html/1903.00268v2
- [PanFusion] PanopticFusion：https://arxiv.org/pdf/1903.01177
- [F++] Fusion++：https://arxiv.org/abs/1808.08378
- [MaskClustering] https://arxiv.org/pdf/2401.07745
- [OnlineAnySeg] https://arxiv.org/abs/2503.01309
- [ESAM] https://arxiv.org/html/2408.11811
- [SAM3D] https://arxiv.org/html/2306.03908
- [OVIR] OVIR-3D：https://arxiv.org/pdf/2311.02873
- [O3DIS] Open3DIS：https://arxiv.org/html/2312.10671
- [OVO] https://arxiv.org/abs/2411.15043
- [GG] Gaussian Grouping：https://arxiv.org/html/2312.00732
- [PCL] https://pcl.readthedocs.io/projects/tutorials/en/latest/region_growing_segmentation.html
- [LCCP] https://openaccess.thecvf.com/content_cvpr_2014/papers/Stein_Object_Partitioning_using_2014_CVPR_paper.pdf
- [RANSAC] https://doc.cgal.org/latest/Shape_detection/index.html
- [Open3D] https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
- [mBEST] https://arxiv.org/abs/2302.09444 · https://github.com/StructuresComp/mBEST
- [FASTDLO] https://github.com/lar-unibo/fastdlo
- [Keipour] https://arxiv.org/abs/2201.06775
- [TrackDLO] https://github.com/RMDLO/trackdlo · https://deformable-workshop.github.io/icra2024/spotlight/02_07_wdo_xiang_trackdlo.pdf
- [CDCPD2] https://arxiv.org/abs/2011.00627
- [MultiDLO] https://arxiv.org/abs/2310.13245
- [Wire] WireSegHR：https://arxiv.org/abs/2304.00221
- [skimage] https://scikit-image.org/docs/stable/api/skimage.filters.html
- [Catenary] https://publications.ri.cmu.edu/multi-view-reconstruction-of-wires-using-a-catenary-model-2/
- [LIMAP] https://github.com/cvg/limap
- [SceneDiff] https://arxiv.org/abs/2512.16908
- [POCD] https://arxiv.org/abs/2205.01202
- [PersF] Persistence Filter：https://github.com/david-m-rosen/Persistence-Filter
- [Perpetua] https://arxiv.org/abs/2507.18808
- [CLS] CL-Splats：https://arxiv.org/abs/2506.21117
- [Intenseye] https://www.intenseye.com/core-ai/housekeeping
- [Intenseye-FAQ] https://www.intenseye.com/resources/faq
- [Intenseye-Sentinel] https://www.intenseye.com/products/sentinel
- [Intenseye-patent] https://patents.google.com/patent/WO2024264050A2/en
- [Protex] https://www.protex.ai/safety/overview
- [Visionify] https://visionify.ai/restricted-zones
- [Voxel] https://www.voxelai.com/solutions-safety
- [Everguard] https://www.globenewswire.com/news-release/2022/09/29/2525358/0/en/Everguard-ai-Solutions-Create-Safer-Manufacturing-Workplaces-to-Attract-More-Workers.html
- [Blaxtair] https://blaxtair.com/en/solutions/blaxtair-pedestrian-detection
- [viAct] https://www.viact.ai/post/pallet-racking-safety-in-warehouse
- [SAS] https://patents.google.com/patent/US12293602B1/en
- [OSH-review] https://pmc.ncbi.nlm.nih.gov/articles/PMC12110780/
- [SAM3-LIC] https://github.com/facebookresearch/sam3/blob/main/LICENSE
- [SAM3DObj] https://github.com/facebookresearch/sam-3d-objects


---

## 检查员更正（以此为准，优先于正文）

总体判断：这份备忘录大部分数字和许可证与一手来源一致，已核对的有 PMT、Khronos、Dynablox、nvblox、MAP-ADAPT、RTG-SLAM、Fusion++、PanopticFusion、LCCP、ConceptGraphs、OVIR-3D、MaskClustering、OnlineAnySeg 的阈值、DA3 各档权重许可、PromptDA 许可与速度、SAM 3 许可、Intenseye 的专利和 FAQ。但有几处需要改。

硬错误：
- mBEST 的 FPS 抄成了 RT-DLO 的。
- Persistence Filter 的许可是 LGPL-3，不是 GPL-3。
- Clio 的阈值和耗时归因不对：0.7/0.4 是论文里 Khronos 基线的参数；0.26–0.30 s/帧是 FastSAM+CLIP+建图整条管线的耗时，不是"打分"的耗时。
- DA3 的商用替代只有 BASE 和 SMALL 能出位姿，METRIC-LARGE 和 MONO-LARGE 不能替代粗层。
- Voxel 的引用页上 95% 指的是摄像头兼容率，不是精度。
- "实时系统全靠规则"与文中自己引用的 ESAM 矛盾，另外 MoonSeg3R 也是学出来的关联。
- PromptDA 的训练提示不全是真实 LiDAR。

实时性论证最大的缺口是 SAM 3：论文说视频里只有约 5 个并发物体时才接近实时（H200），要到 30 FPS 必须上多卡，所以 L0"每帧硬实时"缺乏证据支撑。另外有几个方法被当成候选，实际是离线或非实时的，应标明：SceneDiff（每对视频 2–3 分钟）、SplaTAM（约 2.4 s/帧）、WireSegHR（0.82 s/图）、Keipour（0.54 s/帧）、MaskClustering（全局图，离线）。

遗漏的重要工作：
- 同仓库的 DA3-Streaming。
- 只用单目 RGB 的实时系统：VGGT-SLAM 2.0（已接入 SAM 3）、MASt3R-SLAM（非商用）、MoonSeg3R。
- 自适应分辨率的经典基线：Kähler 2016。
- 视频分析里"粗处处、细按需"的系统工程做法：DDS、Remix、SAHI。
- 逐物体动态 SLAM 的经典基线：MaskFusion、MID-Fusion。

还有一个输入模态问题：TrackDLO、CDCPD2、Khronos、PMT、OnlineAnySeg、ESAM 用的都是 RGB-D 或 LiDAR（Dynablox 是 LiDAR）。它们的阈值（30 cm、β=10°、ever-free）在 DA3 单目深度下必须按深度噪声重新标定，并且要加位姿门控。下面每条都给了来源。

- **问题：**mBEST 的速度写错。文中'30.6–39.6 FPS'（3.3 节表格和 L1）其实是 RT-DLO 在 Table II 里的范围（30.58–39.60）。
  **更正：**改成：mBEST 在复杂背景（C1–C3，含 DCNN 分割，640×360，i9-9900KF + RTX 2080 Ti）上是 31.9–32.2 FPS；在简单背景（S1–S3，用颜色阈值代替网络）上是 37.1–52.8 FPS。输入只有 RGB。论文说 DCNN 分割占了计算时间的大头，所以换成 SAM 3 mask 后，'骨架 + 交叉 <10 ms'这个估计仍然成立。
  来源：https://arxiv.org/html/2302.09444v5 （Table II）
- **问题：**Persistence Filter 的许可证写成了 GPL-3（3.5 节许可汇总）。
  **更正：**改为 LGPL-3.0。以动态链接方式使用可以进商用产品，但要遵守 LGPL 义务。算法本身很小，照论文重写也可以。
  来源：https://github.com/david-m-rosen/Persistence-Filter （LICENSE 文件是 GNU LESSER GENERAL PUBLIC LICENSE v3）
- **问题：**Clio 的阈值归属不对。1.2 节和判据表 #4 写'Clio：嵌入余弦 ≥ 0.7，同时 3D IoU ≥ 0.4'，但 θ_track=0.7、γ=0.4 是论文里给 Khronos 基线用的；Clio-Prim 用的是 0.9 / 0.6。
  **更正：**写成：'0.7/0.4（Clio 论文中的 Khronos 基线）；Clio-Prim 用 0.9/0.6'。起始值取两者之间，再在我们自己的素材上标定。
  来源：https://arxiv.org/html/2404.13696v4
- **问题：**Clio 的耗时归因不对。3.2B 和 3.5 L1 写'Clio 式打分要 0.22–0.31 s/帧，只能降频'。实际上 Clio-online 的 TPF 是 0.26–0.30 s（RTX 3090 + i9-12900K），这是整条管线的耗时：FastSAM + CLIP ViT-L/14 + Khronos 建图。单独跑 Khronos 基线也要 0.26–0.31 s，可见 IB 聚类和相关度打分几乎不增加开销。
  **更正：**改成：相关度打分就是'事实文本嵌入'和'实例嵌入'的余弦，开销可以忽略。真正的成本是给每个 mask 算 CLIP 或 PE 嵌入，可以只在 track 新建或外观变化时算一次。所以 B 方案可以放在 L1，不需要降频。
  来源：https://arxiv.org/html/2404.13696v4 （Table I 的 TPF 一栏，以及'We run FastSAM and CLIP'）
- **问题：**DA3 的商用替代写得不对（第 4 节风险 1、3.5 节许可表）。文中说可以换成 BASE、SMALL、METRIC-LARGE 或 MONO-LARGE，但只有 any-view 模型能估计位姿，而其中 Apache-2.0 的只有 DA3-BASE 和 DA3-SMALL。METRIC-LARGE 和 MONO-LARGE 是单目深度模型，不出位姿。另外，非 1.1 版的 DA3-GIANT 和 DA3-LARGE 也是 CC BY-NC 4.0。
  **更正：**改成：粗层要位姿加深度，商用替代只有 BASE 和 SMALL。METRIC-LARGE 和 MONO-LARGE 只能用于细 ROI 的 crop 深度。不可商用的清单补上 DA3-LARGE、DA3-GIANT（非 1.1）。
  来源：https://github.com/ByteDance-Seed/Depth-Anything-3 （model zoo 表）；https://huggingface.co/depth-anything/DA3-GIANT-1.1
- **问题：**DA3 的逐 crop 延迟是推出来的。'DA3-Small 约 6 ms 一个 crop'和风险 10 的'GIANT 26.6 ms/帧'都来自 Table 8 的 FPS（A100，504×336）。这张表同时列了 max images 一栏，应该是多视图批量吞吐，不是单次调用的延迟。
  **更正：**标为〔估〕：'Table 8 是吞吐量；单个 crop 加位姿条件的调用延迟要实测'。L2 的预算要用单次调用的实测值来定。
  来源：https://arxiv.org/html/2511.10647 （Table 8）
- **问题：**L0 写'SAM 3 跟踪，每帧，硬实时'，但没给数，和一手来源冲突。SAM 3 论文：单张图 100+ 物体 30 ms（H200）；视频里只有约 5 个并发物体时接近实时；成本随物体数线性增长，30 FPS 下 10 个物体要 2 张 H200，28 个要 4 张，64 个要 8 张。SAM 3.1 做了物体复用，一次前向最多 16 个物体，H100 上从 16 FPS 提到 32 FPS（中等物体数）。
  **更正：**L0 改成：用 SAM 3.1；限制并发 track 数；检测器降到 L1 的关键帧频率，中间帧只做传播；回放帧率可以降到 10–15 fps 再实测。按词表里的多类别加人数估算，单卡每帧硬实时没有证据支撑。SAM 3.1 的许可证〔未核〕。
  来源：https://arxiv.org/abs/2511.16719 ；https://ai.meta.com/blog/segment-anything-model-3/
- **问题：**与自己引用的内容矛盾。结论 1 说'我们查过的实时系统全部如此（规则判定）'，1.2 节又说'唯一把合并规则学出来的是 ESAM'。可是 ESAM 标题就是 'in Real Time'（FastSAM 版约 10 FPS）。还有 MoonSeg3R（CVPR 2026 Findings），用学到的 query/identity 描述子做跨帧关联。
  **更正：**改成'多数实时系统是规则判定；也有学习式关联，如 ESAM、MoonSeg3R'。ESAM 的泛化结论有争议：OnlineAnySeg 报告它在 SceneNN 上明显下降，ESAM 自己报告 ScanNet200→SceneNN 为 28.8 AP。
  来源：https://arxiv.org/html/2408.11811 ；https://arxiv.org/html/2512.15577 ；https://arxiv.org/pdf/2503.01309 （4.2 节）
- **问题：**OnlineAnySeg 的 15 FPS 条件没写清。它只在每 10 帧一个关键帧上做分割（SceneNN 上是 20 帧），每 5 个关键帧合并一次，最终结果还用了序列结束后的 OVIR-3D 后处理。输入是带位姿的 RGB-D。论文原话是'从 10 FPS 提到 15 FPS（分割过程中）'，是否包含 CropFormer 仍未说明。
  **更正：**标注：15 FPS 是摊销吞吐，分割只跑 1/10 的帧，最终精度依赖离线后处理。第 4 节第 13 条的〔未核〕改为'部分核实'。
  来源：https://arxiv.org/pdf/2503.01309 （3.5 Implementation Details）
- **问题：**SceneDiff 实际是离线方法，文中没说明。V100 上每对视频约 2–3 分钟；几何用 π³（权重 CC BY-NC 4.0），分割用 SAM。
  **更正：**1.4 节和判据表 #20 注明'离线，只借规则'：可见性门是另一次拍摄中超过 50% 像素被遮挡就不判移除，外观匹配用 DINOv3 余弦 > 0.7。DINOv3 License 允许商用，限制与 SAM 相同。
  来源：https://arxiv.org/html/2512.16908 ；https://github.com/yyfz/Pi3 ；https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md
- **问题：**Dynablox 的输入和失效条件没写。它是按 LiDAR 设计并评测的（Ouster OS0/OS1，10 Hz），在 Ryzen 4800U 笔记本 CPU 上 17 FPS；限定 20 m 量程时每帧 58.1 ms。论文写明的失效情况：极细而稀疏的物体、反光表面。nvblox 文档说：比里程计漂移还慢的移动检测不到。
  **更正：**L0 的 ever-free 层只在同一个 shot 内使用，或者在跨 shot 全局对齐之后（DA3-Streaming 或 VGGT-SLAM 2.0）再用。DA3 深度噪声下的误报率要实测，并加位姿质量门。
  来源：https://arxiv.org/html/2304.10049 ；https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/technical_details.html
- **问题：**RGB-D 方法的阈值直接当成我们的起始值，但没标输入模态。Khronos（RGB-D + OneFormer，i7-12700H 上每帧 45.5 ms）、PMT（RGB-D + 外部位姿，5–6 Hz，不含 66 ms 分割）、OnlineAnySeg、ESAM、ConceptGraphs、Fusion++（RGB-D，GTX1080Ti，4–8 Hz）、TrackDLO 和 CDCPD2（RGB-D）都是这样。
  **更正：**每个方法写明输入模态。30 cm 射线带、β=10°、3° 法向这类阈值，改为随深度和我们实测的深度残差缩放。另外补上 Khronos 的召回率：69.1%（GT 位姿）/ 62.9%（Kimera 位姿）。它是低精度、较高召回，正好支持'候选 + VLM 确认'的设计。
  来源：https://arxiv.org/html/2402.13817v2 ；https://arxiv.org/html/2109.10165v2 ；https://arxiv.org/abs/1808.08378
- **问题：**TrackDLO 位移上限标了〔未核一手〕，现在可以部分核实。一手论文写明的局限是：需要好的深度分辨率、帧间运动小、被遮挡节点的运动要能从可见节点反映出来。实验用 RealSense D435，计时不含分割。论文没有给 mm/帧 的具体数。
  **更正：**把〔未核一手〕改成引用论文第五节的定性结论，删掉 5 mm/帧 的二手数字。这也进一步支持'按窗重检测，不做逐帧跟踪'：5–10 m 外的单目深度满足不了'good depth resolution'。
  来源：http://brian.coltin.org/pub/xiang2023trackdlo.pdf （IV 节实验设置、V 节结论）
- **问题：**PromptDA 的风险表述不准。文中说'它训练时用的提示是真实 LiDAR'，其实训练提示有三种：ARKit 真 LiDAR；合成 LiDAR（GT 深度降到 192×256，再做稀疏锚点插值）；ScanNet++ 上用 Zip-NeRF 重渲染的伪 GT。论文没有测过 LiDAR 形式以外的提示。
  **更正：**风险改写成：提示是低分辨率（192×256）的度量深度。渲染出来的 TSDF 深度形式上接近，但噪声结构和尺度误差不同，需要实测。
  来源：https://arxiv.org/html/2412.14015
- **问题：**Voxel 的引用错了页。[Voxel] 那一页上唯一的 95% 是'兼容 95% 的现有 IP 摄像头'，不是检测精度。'95%+ 精度'和'五十亿小时训练'出现在别的页面和媒体稿里。
  **更正：**换成真正包含这两句话的来源，或标〔未核〕。
  来源：https://www.voxelai.com/solutions-safety ；https://www.thesafetymag.com/ca/annual-guides/voxel-advances-workplace-safety-by-turning-cameras-into-smart-risk-prevention-tools/547577
- **问题：**OSH 综述的结论说过头了。文中写'2025 年一篇综述看了 10 个商用系统，都没有定量验证'。该综述确实列了 10 个商用应用，但它只是自己没做批判性评估，没有断言这些系统缺乏定量验证。
  **更正：**改成：'该综述列了 10 个商用系统，没有提供或引用任何独立的定量评估'。
  来源：https://pmc.ncbi.nlm.nih.gov/articles/PMC12110780/
- **问题：**几个被当成规则来源的方法并不实时，文中没标。SplaTAM 在 3080 Ti 上每帧约 1.0 s 跟踪 + 1.44 s 建图；而且它的深度判据漏了一个条件，原文要求'GT 深度在渲染深度之前'且误差 > 50×MDE。WireSegHR 在 V100 上 0.82 s/图（全分辨率 1.91 s），代码仓库没有许可证文件。Keipour 每帧 0.537 s（CPU）。MaskClustering 是离线全局图。MAP-ADAPT 除了 TSDF 更新，出网格还要 1.1–1.6 s/帧。
  **更正：**给这些条目标上'离线或非实时，只借规则'。SplaTAM 的判据补上'前方'条件。WireSegHR 许可改为'没有许可证文件，照论文重写'。
  来源：https://arxiv.org/html/2312.02126 ；https://arxiv.org/pdf/2304.00221 （Table 4）；https://arxiv.org/pdf/2201.06775 ；https://arxiv.org/pdf/2401.07745 ；https://arxiv.org/html/2406.05849
- **问题：**几个〔未核〕现在可以关掉。SAM 3D Objects 的代码和权重都用 SAM License，允许商用，但有出口管制和军事用途限制。RTG-SLAM 论文说光栅化用的是自写 CUDA kernel，仓库整体是 GPL-3。
  **更正：**更新第 4 节第 12、13 条：SAM 3D Objects 改为'SAM License，可商用但有限制'；RTG-SLAM 改为'GPL-3，自写光栅器（是否带 Inria 代码仍建议查 submodule）'。
  来源：https://github.com/facebookresearch/sam-3d-objects ；https://arxiv.org/html/2404.19706v1
- **问题：**遗漏：DA3-Streaming，就在我们已经用的 DA3 仓库里。它按 chunk 滑窗处理，重叠为 chunk 的一半；在 A100 上跑 KITTI 11,373 帧达到 8.51 FPS，显存不到 12 GB（chunk=30 时 11.5–18.7 GB）。它直接针对风险 2（跨 shot 尺度和位姿不一致）和'可流式'的要求。
  **更正：**加到 3.5 节 L0/L1 的可选项和风险 2 的缓解里。注明 chunk 级延迟是秒级，1 倍速回放要实测；默认用哪个权重（关系到许可）〔未核〕。
  来源：https://github.com/ByteDance-Seed/Depth-Anything-3/tree/main/da3_streaming
- **问题：**遗漏：和我们粗层最接近的只用 RGB 的实时系统 VGGT-SLAM 2.0（2026）。它是单目 RGB 前馈 SLAM：16 帧 submap 在 3090 上 8.4 FPS，Jetson Thor 上 3.5 FPS。开放词汇检测（PE-CLIP 检索关键帧，SAM 3 出 mask，再求 3D OBB）打开后 6.3 FPS。它用 SL(4) 因子图消除 15 自由度的漂移。
  **更正：**加到第 1 节作为参照系统，并作为跨 shot 对齐的候选。许可：代码 BSD-2；VGGT 原权重不可商用，另有 VGGT-1B-Commercial 需要申请。注意它的 SAM 3 物体只在查询时生成，不做持续的实例建图和变化判定。
  来源：https://arxiv.org/html/2601.19887 ；https://github.com/MIT-SPARK/VGGT-SLAM ；https://github.com/facebookresearch/vggt
- **问题：**遗漏：只用单目 RGB 的在线 3D 实例分割 MoonSeg3R（CVPR 2026 Findings）。它用 CUT3R 做几何，CropFormer 出 mask，学习式跨帧关联（阈值 0.8 和 1.8）。A6000 上每帧共 321 ms（约 3.1 FPS），其中 CropFormer 200 ms、mask 融合 55 ms。局限：长序列上几何误差会累积。
  **更正：**加到 1.2 节。它是'学出来的关联在单目上能跑'的反例，也可以作为我们规则关联的对照基线。代码在 VICO-UoE/MoonSeg3R，许可证〔未核〕；CUT3R 的许可证〔未核〕。
  来源：https://arxiv.org/html/2512.15577
- **问题：**遗漏：单目稠密 SLAM 的常用基线 MASt3R-SLAM（CVPR 2025）。单目 RGB，4090 上 15 FPS，不需要标定也能跑。
  **更正：**作为对照基线列出，同时放进'不能商用'一栏：MASt3R 代码和权重都是 CC BY-NC-SA 4.0，另外还受训练数据许可约束。
  来源：https://arxiv.org/abs/2412.12392 ；https://github.com/naver/mast3r
- **问题：**遗漏：自适应分辨率 TSDF 的经典基线 Kähler、Prisacariu、Valentin、Murray 的 'Hierarchical Voxel Block Hashing'（RA-L 2016）。它在 GPU 哈希上做多分辨率，按局部表面曲率自动选分辨率。supereight 论文也把它作为'按曲率而非距离选细度'的代表。
  **更正：**加到 1.1 节表格，放在 MAP-ADAPT 和 MrHash 之前作为源头。代码是否公开〔未核〕。
  来源：https://ieeexplore.ieee.org/iel7/7083369/7163696/07368096.pdf ；https://www.doc.ic.ac.uk/~sleutene/publications/Vespa_3DV19.pdf
- **问题：**遗漏：视频分析系统里'粗处处 + 细按需'的工程做法，这正好回答'算法还是工程'。DDS（SIGCOMM 2020）：先传低质量视频，服务器端 DNN 反馈出相关区域，只把这些区域高质量重传，带宽最多省 59%。Remix（MobiCom 2021）：在延迟预算内，按历史物体分布生成非均匀分块计划，同延迟下精度提升 18–70%，或提速最多 5.5 倍。SAHI：切片推理，小物体 AP 提升 5–7 个点。
  **更正：**加到第 1、2 节：别人把'在哪里花算力'当成系统调度问题来做，用模型自身的反馈（置信度、历史分布）加预算规划。这可以直接用在 L2 调度器，也可以用在线缆 ROI 的 crop 重跑。许可：SAHI 是 MIT；DDS 代码没有许可证文件；Remix 代码〔未核〕。
  来源：https://people.cs.uchicago.edu/~junchenj/docs/DDS-Sigcomm20.pdf ；https://www.microsoft.com/en-us/research/wp-content/uploads/2021/08/Flexible-High-resolution-Object-Detection-on-Edge-Devices-with-Tunable-Latency.pdf ；https://arxiv.org/abs/2202.06934 ；https://github.com/obss/sahi
- **问题：**遗漏：'可移动刚体 = 自己的子图 + 自己的位姿'的经典基线。MaskFusion（ISMAR 2018）：RGB-D，每个实例单独建模，独立跟踪运动。MID-Fusion（ICRA 2019）：基于 supereight 八叉树的逐物体动态 SLAM。
  **更正：**加到 3.4 节'可移动刚体'的借鉴列。许可：MaskFusion 是 GPL-3，并含 ElasticFusion 自带的许可；Co-Fusion 是 GPL-3。只借设计。
  来源：https://arxiv.org/abs/1804.09194 ；https://github.com/martinruenz/maskfusion ；https://www.doc.ic.ac.uk/~sleutene/publications/MID_Fusion_2019.pdf
- **问题：**PMT 的速度只给了一半。5–6 Hz 是在 640×480、i7-8550U 上测的，不含 66 ms 的 V100 分割。同一篇论文在 RIO 数据集（224×172，物体更少，每 10 帧做一次变化检测）上是 21.3 FPS（2–5 cm）和 32.2 FPS（4–10 cm）。
  **更正：**两组数都写上，并补一条〔估〕：关联和分配可以在降采样后的 mask 和深度上跑，这是把重写后目标定在'≤ 20 ms 每关键帧'的依据。
  来源：https://arxiv.org/html/2109.10165v2
