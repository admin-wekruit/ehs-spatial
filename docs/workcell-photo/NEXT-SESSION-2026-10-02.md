# Workcell 接手：模型、尺度、测量与语义同源

更新：2026-10-02。这是本次交接的首读文件和完整未完成项清单；比旧文档中的“剩余三项”范围更完整。旧实验文档用于查证，不是当前状态总表。

## 1. 用户要什么

四张照片 + 可调整的已知按钮尺寸，一次自动运行，得到可选择、可旋转、有细节的 workcell 3D 模型；每个对象的名字、原图证据、尺寸、离地高度、夹角和模型对应。统一地面；语义查询能找到对象并读取同一模型的空间事实，后续连接 EHS policy。几分钟可接受，优先正确性，不做新训练。不要继续只展示点云或用一堆方块代替所有物体。

最新明确要求：**新模型产生后，尺度、地面、测量、语义卡片应自动随同一模型版本更新，不能每轮靠人另外挂接高度。** 先修这条工程链路，再继续精度和语义实验。

用户只要求本轮整理、推送 handoff；本文件提出的待办尚未执行。交接文档提交不等于算法或 oneshot 已修好。

## 2. 为什么新高度没接进去：已查明的工程原因

1. `scripts/workcell_semantic_report.py::attach()` 把当时 `endpointEstimation` 中的厘米值格式化成 `policyContext.facts[].value` 字符串。`web/src/PhotoSemanticExperiment.tsx::PhotoSemanticObject` 直接显示该字符串，未读取当前选中模型的测量。它也不随网页中的标尺试算更新。
2. 新 G 轮模型被导出到独立 `housing-boundaries/` 候选页；主报告模型更新数为 **0**。同一 `post-box-1/2` 在不同模型版本中有不同几何。只把新数字贴到旧主模型也会错。
3. `workcell_photo_report.build()` 重建报告，没有把语义作为构建步骤；`workcell_photo_oneshot._build_page()` 也未打包语义裁剪。语义现在是重建后单独运行、单独 attach 的实验。
4. 最后一次模型/地面/尺度写入后的统一收尾没有覆盖所有入口。本轮只给云端 oneshot 末尾补了 `estimate()` 端点重算；本地 `run()` 之后还会 `apply_measurements()`、`build_report()`，必须沿整条写入链审查，不能把云端的一次重算当成完整修复。

**计算可以分模块，发布的数据必须属于同一 revision。** 新网格先通过节点/对象变换落在场景坐标，随后用该场景的地面和尺度测量。生成器输出的归一化网格不能直接读作米。模型内测量正确，也不自动证明生成表面就是现场的真实物理下沿。

## 3. 当前可核对版本与结果

- 公开源码：<https://github.com/admin-wekruit/ehs-spatial>，分支 `codex/workcell-photo-speed`。本次交接前最后算法提交 **`e8586bb`**；handoff 文档提交在其后。仓库默认分支是 `feature/ehs-spatial-mvp`，不要误用默认分支。
- 本地源码：`/Users/adam/.codex/worktrees/panoptes-workcell-photo-speed`。
- 网站仓库：<https://github.com/admin-wekruit/panoptes-workcell-report>，`main`，当前产物提交 **`5731635fe7f338886f83a509b6d204ab2097653d`**。
- 本地网站：`/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages`。
- [完整 52 对象报告](https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/)；[旧模型上的语义入口](https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/?view=model#semantics)；[G 轮光幕模型与共同地面卡尺](https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/housing-boundaries/#models)。
- G 轮导出两根封闭挤出候选。**可见宽主体下沿**条件估计：右 `post-box-1` **24.5120 cm**，左 `post-box-2` **24.1081 cm**，差 **0.4039 cm**。左右以照片 4 面向工作单元为准。没有强制同高、没有用 24 cm 目标拟合。
- 底边原图来源：右照片 3/4，左照片 2/3，其余视角只补充上端/侧边。厚度与隐藏背面为假设，左侧厚度触及搜索下限；外壳阶梯最低端不等于当前可见面底边。严格跨图投影门槛仍未通过，**没有完成实体物理精度验收**。
- 四图平均 SAM 轮廓 IoU：右 0.6863→0.7249（2/4 视角改善），左 0.5879→0.7396（4/4）；不是物理 3D 精度。全量及留一照片重算范围：右 24.104–26.585 cm、左 24.108–27.223 cm，不含尺度/地面系统误差，不是置信区间。
- 主模型仍是原 52 对象。其条件读数：右光幕 24.71 cm、左光幕 28.44 cm；右围栏 18.55 cm、左围栏 24.34 cm。语义卡只挂了旧右光幕 24.71 cm，左侧未挂高度。**不得把这组与 G 候选混为同版结果。**
- 当前语义：24 个文本候选类别、6 个离线查询、52 对象、97 观察。PE-Core 查询“光幕”前两名正确；SigLIP2 排左光幕、围栏、右光幕。两编码器对两根光幕的多视角分类第一名都是“普通黄色柱”。目录中的“光幕”名称含用户确认来源，不是模型已正确分类的证据。
- 按钮已知规格：整体高 **10 cm**，主体最大直径 **8.5 cm**，红帽直径 **4 cm**；整体高度范围用户已确认，含红帽/黄体/灰底，不含安装支架。三尺寸已进入拟合实现，但没有通过验证，accepted physical scale **null**。当前模型条件比例 **0.6844702474893979 m/native**，来源照片 4 主体 8.5 cm，不能冒充三尺寸联合标定成功。
- 围栏 **20 cm**、光幕 **24 cm** 是已知开发检查值，只在冻结结果后评估；不是优化目标、候选选择条件或盲测。后续独立检验要另留未参与调参的实测点/照片。

## 4. 已完成：保留、复用，不重写

- 统一模型地面与旧高度/垂足缓存刷新；一点离地、两点距离/有符号高差、三点采样范围；原生点/模型哈希/地面/比例的 JSON 导出与 GLB 同源导出。
- 照片/模型拖动对照、可旋转 3D、对象选择、照片证据、护板两面角度标注；当前公开主报告有 52 个对象。
- 本轮修复宽主体被窄侧翼候选挤掉、端边拉歪参考长边后循环认证，以及整面提取重复调用旧三维匹配。已有反例检查。
- G 页模型读取实际 GLB 顶点 0/1 作卡尺；两侧垂足共用同一地面。真实浏览器验证过旋转、缩放、单独选择与 4 个上下文模型加载。
- 已修 G 页 `entity-floor.glb` 坐标误用：该资产已转报告坐标并居中，不能再按 native 地面旋转。改用 `floor-fitted.glb`，与冻结输入哈希一致；不是另加角度补偿。
- PE-Core/SigLIP2 已有双 A100 并行语义实验；继续复用编码器、SAM 裁剪、身份映射和报告组件。

## 5. 全部待办与验收（按顺序；未勾选即未完成）

### P0 — 一个模型版本，一套事实：下一会话先做

- [ ] **统一最终构建顺序。** 确定当前 revision 使用的模型 → 校验 GLB/节点/端边绑定 → 最终地面与尺度 → `estimate()` → `build_report()` → 语义证据绑定 → `_build_page()`/导出。追踪所有调用者，覆盖云端和本地末尾写入；接入后的模型不得被后续 `_posts()` 覆盖。不另建一套测量引擎或通用任务框架。
- [ ] **移除语义卡的重复厘米快照。** 当前空间值从选中模型所在 revision 的结构化测量和现有格式化/尺度状态派生；历史实验快照只作历史证据。包含左右光幕、围栏及其他可测对象，不硬编码光幕数值。
- [ ] **模型选择与测量绑定。** 复用实体 ID、representation、`activeModelRepresentationId`、`documentSha256`、资产 hash、节点、测量端边引用。主模型/候选可有不同状态，但选中哪个就读哪个的测量。先让候选版本自洽，不通过放宽物理 gate 强行接替主模型。
- [ ] **语义重新构建与打包。** 重建报告后语义不丢、裁剪不漏。仅图像及裁剪不变的视觉特征可复用；空间聚类依赖的几何若改变须重算。保留冻结实验 hash 校验和复用来源，不能改写旧 `sourceRevisionId` 或关掉检查蒙混过关。
- [ ] **标尺联动与 unknown。** 改标尺后模型、卡尺、语义空间值、JSON 和 GLB 采用同一换算。只改比例不重写 native 点；改地面要重算高度与垂足。`attach()` 当前 `estimateCm:.2f` 遇到 null 会报错，必须覆盖未知尺度。同比修改三尺寸是试算；改变三个规格的比例需要重新拟合。
- [ ] **语义点击全链路。** 查询→正确左右对象→对应原图/模型→该版本测量。当前 `selectSemanticObject` 会关闭端点卡尺，需让用户有明确可用的测量入口；不要跳到错误对象或使用另一个版本的数字。

P0 最小验收：在现有检查中加入“同 ID 不同 GLB hash”“只改地面”“只改尺度/尺度 null”“主模型↔候选”“语义源变了却复用旧结果”反例。旧测点必须拒绝；重算后卡片、模型测点、垂足、语义、导出逐项相同。已有 `check_workcell_bottom_models.py`、`check_workcell_photo_report_endpoints.py`、`check_workcell_semantic_match.py` 与前端卡尺检查可复用。至少跑一次实际浏览器查询到测量与下载回读，不以 build 通过代替。

### P1 — 物理模型、尺寸与语义正确性

- [ ] **光幕完整外壳及真实测量部位。** 在宽主体基础上处理阶梯、侧翼、厚度和遮挡；明确可见面下沿、整个外壳最低端、安装支架、实际检测区域的区别。不能把包围盒低分位、假设背面的最低点或未知处补全当实测端边。逐图投影、留出图、端点身份和实际 GLB 回读都要通过对应验收。
- [ ] **左右围栏等价下沿。** 左右均定位同一类下横梁的物理底面，移除右延伸横杆/左结构包络口径差。共同地面上的左右高差应有原图证据。同高是用户提供的先验，可单列对照，不能靠强制相等宣称测准。
- [ ] **三尺寸标尺、相机与局部地面。** 核对 10/8.5/4 cm 对应部件与圆轮廓透视、原图→canonical 仿射/像素中心、相机初始化及目标附近地面支持。改变相机后同步更新依赖几何，不能混用新相机和旧点图。现有相机均衡/COLMAP/分辨率实验的负结果保留，先隔离原因，不重复无效增加迭代。无需先要求用户补相机参数；先把现有四图做到并实测可达程度。
- [ ] **局部测量验收。** 继续以清楚可见的围栏/光幕局部离地误差 <3 cm 为试验目标；每项报告测量部位、误差、覆盖和失败。当前左右差 0.40 cm 不等于对真值误差/全场误差 <3 cm。新增独立点/照片后再验证泛化，不能承诺任意物体厘米精度。
- [ ] **三组折弯护板。** 每组两片独立面及夹角、对地面倾角与原图一致。左右 104.2702° 是共享结构候选；104/120/150/174° 多个候选可贴合，真实角度未唯一确定；中间仍无可靠实测角度。改纹理不算几何改进；不把共享参数当物理真值。
- [ ] **其他物体完整性。** 复用原报告的按钮、灯、机器人、小车、护板等，不因改测量缩减对象。逐个检查点击、identity、部件边界、姿态及模型原图对应；细小模糊处按用户接受范围表示。机器人跨图不同姿态不能合成静态实体或危险包络。通用卡尺不等于自动识别了所有对象的物理底面。
- [ ] **光幕名称与参考图 prelist。** 已有的是名字＋文字描述；尚无名字＋参考图的输入、编码和匹配。复用现有编码器和 SAM，先对照干净主体/光学面裁剪与现有背景混杂裁剪，测试参考图能否区分光幕和普通黄柱。参考实例、测试实例分开；保留用户确认身份与模型预测各自来源。
- [ ] **开放文本和空间查询。** 当前网页只有 6 个离线查询；配置可增加 phrase 后重算，但没有自由输入执行。接入自然语言定位及“右侧光幕离地多高/左右谁更高”时，数值和比较必须调用当前版本的结构化事实，不让语言模型重新猜测长度。具体类别、区域与测量部位有依据；缺事实明确返回缺失。
- [ ] **跨图语义身份与覆盖评估。** 检索正确不等于自动类别正确。当前跨照片最近邻由 45/97 到空间筛选 42/68，覆盖 70.1%；不能只报 61.8% 而隐藏覆盖下降，也不能当成整簇准确率。建立独立标注，对照背景、分割、单图/多图和空间融合，不把目录标签当独立真值。

### P2 — 完整自动流程、速度、发布

- [ ] **接入并跑完整 oneshot。** 在四张原 JPEG 上一个命令完成重建、模型、统一测量、语义、报告；最终选择和状态自动产出，不靠事后手工复制数字。重用现有并行，云端是同一预算内 2×A100；CPU 几何与可并行 GPU 编码安排清楚。当前没有这个整合后的完整运行成绩。
- [ ] **真实性能复测。** 区分 UI 构建、冷启动/排队、推理、后处理、打包和上传；报告完整延迟、阶段耗时、GPU 峰值、成功/失败支出。366.93 s 是旧完整运行；71.41 s 是 G 缓存重算；47.62 s 是另一个语义增量，不能直接相加当新实测 latency。
- [ ] **同版本发布与回读。** 主报告、GLB、测量、语义、照片、资源哈希一起发布；本地和线上验证选择/旋转/卡尺/标尺调整/导出、控制台错误及实际 URL。网站仅改 `workcell-photo-direct/`，不要覆盖其他报告目录。原始证据与失败记录保留，避免只发布成功截图。
- [ ] **数据可移交。** 当前源码、主报告、G 页证据已公开；最新冻结输入和完整 F/G/语义缓存主要仍在本 Mac，旧 release 不包含它们全部。需要远端机器重放时，将所需新缓存按允许清单归档到已有授权云存储、附逐文件 SHA 和无秘密扫描；不要声称旧 release 能直接重放 G。权重独立交接，不进公开仓库。

### P3 — 已提出、尚未实现，不能漏记

- [ ] **EHS policy 结合。** 对象身份/部件→空间事实→功能与危险源关系→政策来源及版本→适用性→规则执行→带证据结果。当前只有 candidate topics 和 missing evidence，`applicability=unknown`、`machineResult=null`；没有正式条款检索/阈值执行。照片不能确认光幕停机性能、接线/联锁或检测区有效性，这些需设备信息/测试证据。先修对象事实，不提前输出安全结论。
- [ ] **公司 GPU / SSH 迁移。** 用户此前询问过公司 A100 和交接。现有推理由框架调用 CUDA，SSH 是启动/传输通道；尚无经新公司环境实测的部署入口或单 A100 性能。权重缓存、依赖/环境、GPU 显存和排程需实际验收。当前生产实施仍沿用已授权 Modal 双卡，不擅自改后端或承诺单卡相同速度。
- [ ] **原三视频工作仍未验收关闭。** ME340/Sam’s Club/Walmart 的位置、模型重合、GT 地面误差、单视图尺寸不输出确定值，以及时间/多次访问验收仍属于视频主线；照片结果不替它通过。原目标：80% 以上 RecGen 对齐、重合度中位≥0.6、中心偏差≤8 cm、GT 地面位置中位≤8 cm，约267 s可接受。当前优先照片；恢复视频时读 `handoff/phase2-2026-09-29-docs` 分支的 `HANDOFF.md`、`NEXT-SESSION-GOAL.md`、`SHOW-REPORTS.md`，重新核对该分支最新数据，不用本页候选值推断视频状态。

## 6. 接手入口与文件地图

先读本文件，再按任务读：

- `HANDOFF.md`：历史输入包、环境、模型、公司交接；旧固定 tag 不是本次算法版本。
- `PHYSICAL-BOTTOMS-2026-10-02.md`：真实端边实验 A–E、外壳 A–G、失败、根因、支出。
- `GROUND-CALIPER-2026-10-02.md`：已完成的模型卡尺/坐标/导出契约。
- `SEMANTIC-MATCH-2026-10-02.md`：语义协议与负结果。
- `ONESHOT-CORRECTION-2026-10-01.md`：现有全流程与三尺寸模型。
- `RGB-DEPTH-EXPERIMENT-2026-10-01.md`、`REAL2SIM-EXPERIMENT-2026-10-01.md`、`SUB3-EXPERIMENT-2026-10-01.md`：已做过的负实验，防止重走。旧计划中的实现描述可能已过期，以现代码和运行清单为准。

代码复用位置：

- `scripts/workcell_photo_oneshot.py::run/_build_page`、`modal_apps/workcell_photo_all.py::reconstruct`：全流程和最终产物顺序。
- `scripts/workcell_bottom_models.py::apply_housing_models`：hash/节点/来源/gate 与 `physicalBottom` 绑定；不要把未通过候选标成 accepted。
- `scripts/workcell_endpoint_estimate.py::estimate`：实际网格经节点变换后的端点、共同地面、输入 hash。
- `scripts/workcell_photo_report.py::build`：端点验证、报告与统一换算。
- `scripts/workcell_semantic_report.py::attach`、`scripts/workcell_semantic_match.py`：冻结特征、对象映射、当前重复数值快照。
- `web/src/PhotoReport.tsx`、`PhotoSemanticExperiment.tsx`、`ReportScene.tsx`：选择、语义、现有测量格式化与卡尺，勿重写第二份公式。
- `scripts/workcell_post_faces.py`、`workcell_photo_metrology.py`：原图边缘/面身份与联合拟合。
- `scripts/workcell_physical_bottoms.py::report_housings`、`workcell_metrology_report.py::_preview_markup`：G 页、同地面查看器及原图证据。
- `scripts/workcell_button_bundle.py`、`workcell_photo_geometry.py`、`workcell_photo_calibration.py`：按钮参考与 accepted scale。

## 7. 数据、运行与恢复命令

本 Mac 路径：

```text
项目 Python/Modal：/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/{python,modal}
原始四 JPEG：/Users/adam/Desktop/panoptes-public/panoptes-serving/runs/user-bor1-02/input/image_01.jpg ... image_04.jpg
G 冻结输入：/Users/adam/Desktop/panoptes-public/research-notes/workcell-physical-bottoms-input-2026-10-02
G 输出：/Users/adam/Desktop/panoptes-public/research-notes/workcell-housing-2026-10-02-g
F 输出：/Users/adam/Desktop/panoptes-public/research-notes/workcell-housing-2026-10-02-f
旧完整运行：/Users/adam/Desktop/panoptes-public/research-notes/workcell-oneshot-correct-2026-10-01-b
语义成功运行：/Users/adam/Desktop/panoptes-public/research-notes/workcell-semantic-match-2026-10-02-b
语义冻结基线：/Users/adam/Desktop/panoptes-public/research-notes/workcell-ground-caliper-2026-10-02
总账：/Users/adam/Desktop/panoptes-public/research-notes/phase2/video-mvp/cost-ledger.json
G 线上截图：research-notes/workcell-housing-2026-10-02-g/published-report.jpg
```

G 发布证据、候选 GLB、拟合 JSON、输入/源码清单、费用、卡尺回读 JSON 在网站 Git 的 `workcell-photo-direct/housing-boundaries/`。`input-manifest.json` 指向的 baseline 有约204 MB文件，不等于发布目录。G 的 `updated/` 是未通过主写回的实验产物，**不得整包覆盖主报告**；失败分支会清掉部分受支持测量。

原四 JPEG/历史数据公开包：源仓库 release `workcell-photo-handoff-2026-09-30`。先 `gh release view ... --json assets,url` 核对，再下载并验证 `SHA256SUMS`。`workcell-measured-reference-2026-10-01` 是旧固定源码 tag，不含本次后续修复。最新算法至少需 `e8586bb`；实际接手 checkout 本文所在分支提交。

同一 Mac：先查 `git status`，不要覆盖已有改动。当前仅遗留未跟踪 `web/node_modules`、`web/tsconfig.tsbuildinfo`，不要提交它们。新机器可小量检出：

```sh
git clone --single-branch --branch codex/workcell-photo-speed https://github.com/admin-wekruit/ehs-spatial.git
git clone --filter=blob:none --no-checkout https://github.com/admin-wekruit/panoptes-workcell-report.git
git -C panoptes-workcell-report sparse-checkout set workcell-photo-direct
git -C panoptes-workcell-report checkout main
```

先做 P0 的免费检查；下列是当前入口示例，**不是本次已运行或已经包含语义整合的命令**：

```sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:modal_apps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# 用上面的项目 venv，或已准备好相同依赖的环境。
python scripts/check_workcell_bottom_models.py
python scripts/check_workcell_photo_report_endpoints.py
python scripts/check_workcell_semantic_match.py

# 修改、审查后，再在全新输出目录跑完整流程：
python scripts/workcell_photo_oneshot.py \
  --images "$INPUTS/image_01.jpg" "$INPUTS/image_02.jpg" "$INPUTS/image_03.jpg" "$INPUTS/image_04.jpg" \
  --measurements docs/workcell-photo/measurements-2026-10-01.json \
  --viewer-assets "$PAGES/workcell-photo-direct/real2sim/viewer-assets" \
  --out "$NEW_OUTPUT"

# 当前缓存外壳实验入口（独立实验，不代表完整 oneshot）：
python -m modal run modal_apps/workcell_guard_experiments.py \
  --mode post-shells --baseline "$FROZEN_INPUT" \
  --sources "$INPUTS/image_01.jpg,$INPUTS/image_02.jpg,$INPUTS/image_03.jpg,$INPUTS/image_04.jpg" \
  --out "$NEW_EXPERIMENT"
```

`INPUTS`、`PAGES`、`FROZEN_INPUT` 按上述真实路径设置，输出必须是不存在的新目录。不要覆盖历史证据或从主 Pages 反推完整冻结输入。项目 Modal 为1.5.4；系统全局0.74曾在分配资源前 precondition 失败，使用项目 venv。

## 8. 费用、失败、资源和交付规则

- 本次 handoff 只读/写文档，新增 GPU 运行 **0**。
- Housing A–G 函数资源价格估算合计 **$1.9727177**，包含 D 超时和 E SIGSEGV；G **71.4148 s / $0.1267756**，客户端调用83.3196 s。实际账单 null。
- 更早 physical-bottoms A–E 是另一组，总估算 **$1.762**；不要和 housing A–G 当同一轮。语义成功47.62 s，首次失败＋成功函数估算约 **$0.08032**。完整冷启动/排队窗口与函数窗口分别保留，不能当精确账单。
- 当前未核实一个新的剩余金额上限。继承用户允许有界实验、记录支出的授权；禁止无限重跑。继续沿既有 ledger 追加每次成功/失败，不为查账停下工作。
- D 卡在600 s上限；E重复诊断后SIGSEGV原因未证实。已消除重复旧求解、预先过滤小连通域并批量等价计算，F/G成功；不能说已证明某个库导致崩溃。
- builders → integrator → independent adversarial review。共享文件明确 ownership；不向运行中的 workflow agent 使用 SendMessage，不制造重复写入者。
- 重计算仅 ephemeral `modal run`，2×A100-80GB、显式超时、retry0、min0；不 deploy，不 SSH 已拒绝的部署容器。Jev-Omni如使用放独立已有服务器，不抢这两张卡。VLM最后且少用，RecGen仅内部；不训练新模型。
- 不读 `.platform/imports`，不处理/打印秘密，不把权重、env、凭据、原始日志、`node_modules` 进公开仓库。输出日志过滤 `capabilit|token|secret`。Modal volume 名称可从 `workcell_photo_all.py`核对，缓存许可按原交接。
- 当前本 Mac约 **7.0 GiB** 空闲（本次实查），低于旧8 GB资源线；这轮仅小文档。不在本机新增大副本/重建，大数据处理与归档优先云端。不要删原图/冻结证据/他人修改来腾空间。
- 已有源码和指定网站发布授权，验证后直接 push；不要再问能否发布。不要给其他会话主动发消息，本次交给用户复制 prompt 即可。
- 短中文，图与可打开的报告；不要重复“位置对不对”表格。明确本轮实际变了什么、尚未变什么、延迟和费用；修完主动报告效果。

## 9. 下一会话第一步与完成定义

第一步：复现“主模型24.71 cm/候选24.51 cm/语义快照不随当前模型或标尺变化”，沿 `run → build → attach → _build_page → PhotoReport/PhotoSemanticObject` 追踪全部写入与读取。用已有共享测量函数修一次，先让候选和主版本各自数据一致；不先开新重建实验、不另写报告、不手改高度字符串。

P0 完成后，按 P1/P2推进。整体验收：**一次原图输入生成可用模型；用户用名称找到任意有证据的对象，看到的模型、原图、测点、统一地面、尺度、语义空间回答、JSON和GLB为同一版本；真实尺寸与不确定范围有来源。** 端到端一致性与物理精度分项验收，任何未过项留在本清单，不能用 UI 能打开或左右数字接近替代。
