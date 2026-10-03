# 同一版本、同一套事实：模型 → 地面/尺度 → 测量 → 语义 → 导出

更新：2026-10-03。本轮在云端容器（4 vCPU，无 GPU）完成；新增 GPU/Modal 调用 **0**，GPU 费用 **$0**。容器的网络策略拒绝 `modal.com`、`api.modal.com`、`huggingface.co`、Supabase 与 `*.github.io`，且没有 Modal 凭据，所以凡需要 GPU 或模型权重下载的项目本轮都没有运行（见第 7 节）。

## 1. 根因（已修）

| 交接记录的问题 | 本轮处理 |
|---|---|
| `attach()` 把 `estimateCm` 写成语义卡字符串；不随模型或标尺变化；`None` 时崩溃 | 语义卡不再保存任何数值。页面从**当前加载版本**的端点表和当前标尺实时派生“可用的空间证据”；`None` 尺度显示 native |
| G 候选在独立页面，主报告模型更新数 0；同 ID 在不同版本几何不同 | 候选模型以独立 revision 安装（`install_candidate_models`），父版本为主模型；页面可切换版本，所有读数只读所选版本 |
| `build()` 不含语义；`_build_page()` 不打包语义裁剪 | 语义绑定成为 `build()` 的一步；`_build_page()` 打包被绑定实验的裁剪与清单 |
| 云端末尾才重算端点；本地 `run()` 之后还会 `apply_measurements/build_report` | 统一尾部 `workcell_photo_report.finalize`：参考尺寸 → `estimate()`（当前 GLB、当前地面）→ `build()`（校验端点哈希/地面，绑定语义）。云端 `reconstruct`、本地 `run()`（现在**总是**本地再跑一次）、post-shells 实验、历史 A4 封装都调用它 |
| 端点只测右光幕+右围栏，左侧数值只存在于文档 | `estimate()` 读取目录中所有光幕（接受的或候选的端点绑定，否则未改动的显示原语），每根光幕配对水平最近的同类下横梁成员；左右由照片 4 相机投影决定，不按对象 ID |
| （审查中新发现）目录里的可见尺寸与倾角是 objects 阶段在更早的帧和地面上测的：地面法向此后变了 **0.276°**；76 个单实例观察中，用 SAM 实例掩码 ∩ 当前帧有效掩码重算，只有约一半（40 个）的支持像素数与目录一致 | 倾角按**本版本地面**精确换算：拟合方向本身不依赖地面，30 项最大变化 0.276°，原值保留为 `objectsStageValueDeg`。可见尺寸保留 objects 阶段的分割支持，每个对象注明其测量地面与本版本地面的夹角（`visibleExtentFloor`）。曾尝试用保存的外轮廓多边形在本版本帧上重测，第二轮审查证实这是错的：外轮廓会把围栏网孔后的背景点收进来，fence-1 照片 2 多出 4,884 像素，倾角从 35.8° 变成 14.1°。该做法已撤回；按本版本帧精确重测需要复用 objects 阶段的支持规则（SAM 掩码＋平面过滤），列为待办。可见尺寸在无 accepted 尺度时页面显示“未知” |

## 2. 两个版本的读数（条件模型估计，未验证）

比例均为照片 4 主体 8.5 cm 的条件比例 0.6844702474893968 m/native（与已发布值差 1e-15，来自数值库版本），accepted physical scale 仍为 null。

| 测点 | 主模型 `workcell-main-2026-10-03` | 候选 `workcell-housing-g-2026-10-03` |
|---|---|---|
| 右光幕 | 底端（原语底面中心）**24.71 cm** | 可见面下沿 **24.51 cm** |
| 左光幕 | 底端 **28.44 cm** | 可见面下沿 **24.11 cm** |
| 右光幕旁围栏下沿 | 18.55 cm | 18.55 cm |
| 左光幕旁围栏下沿 | 24.34 cm | 24.34 cm |
| 光幕 − 旁边围栏（右 / 左） | +6.16 / +4.10 cm | +5.96 / −0.23 cm |
| 左 − 右（光幕 / 围栏） | +3.73 / +5.79 cm | −0.40 / +5.79 cm |
| 文档 SHA | `bd47ece13e68…` | `4f9bd0e5b82f…`（父：主模型） |

文档 SHA 与审查前（`8ba60ed…`/`cd1e657…`）不同，只因 30 项倾角按本版本地面换算（最大 0.276°，中位 0.21°）并增加来源字段；可见高度与端点数值逐位不变。

开发检查值（非盲测，冻结后才相减，页面以表格列出）：右围栏 20 cm，左右光幕各 24 cm；左围栏没有检查值。主模型差：右围栏 −1.45、右光幕 +0.71、左光幕 +4.44 cm；候选：右光幕 +0.51、左光幕 +0.11 cm。这些差值**不是**精度验证：测量部位与现场量尺端点的对应仍未确认，检查值也从不约束左右同高。候选来自 G 轮“封闭挤出假设”，其源面未通过严格跨图门槛（`sourceFaceGateAccepted=false`），在页面和数据中都标为候选，不会成为物理离地或替换主模型。

## 3. 数据来源：公开包即可重放

`scripts/workcell_replay_inputs.py` 从公开 release `workcell-photo-handoff-2026-09-30` 的 09-30 帧与公开报告目录重建 2026-10-02 主报告运行目录：10-01 运行与 09-30 帧是同一次网络输出（相机逐位相同），只把深度梯度掩码换成 `be93511` 的分辨率归一化规则。重放后四张照片的有效点与已发布点云**逐字节、同顺序相等**；其余文件按 G 冻结输入清单的 SHA-256 核对后复制；`depth_z` 不在 09-30 帧里，且无 workcell 消费者，记录为省略。用审查前代码对重放目录 `build()`，得到的 document 与已发布 document 只差 `entity-floor.glb` 一个资产（顶点 4.4e-16 的浮点舍入）；当前代码另把倾角换算到本版本地面并增加来源字段（见第 1 节）。

在同一重放上，G 轮光幕外壳拟合（`build_multiview` + `build_volume_candidates`）在本机 CPU **53.6 s** 重跑完成：两根光幕的拟合下沿高度逐位相同，两个封闭外形候选 GLB 字节相同；其余差异为 1e-7 量级数值库差异和纹理重采样 1 个灰阶。也就是说，G 实验现在可以只用公开数据复现。

## 4. 语义绑定（不再用整文件哈希，也不放松检查）

实验只消费：标准化照片、目录观察多边形、帧点图与有效掩码、刚体地面变换；模型 GLB、尺度、端点与表示从不进入实验。`workcell_semantic_report.bind` 对每个版本按内容核对：照片字节相同；用本版本多边形重建的掩码**逐像素**等于实验掩码；支持点数与采样点数相同；地面变换相同；帧文件相同，或带有重放清单中“有效点与原报告点云逐位相等”的证明链。审查后又加一项：实验观察使用的**绝对多边形**必须与实验自己哈希过的目录（`semantic-experiment/source-objects.json`，SHA 等于实验记录的 `objects.json`；新实验由 `prepare` 直接写入）逐点相同——归一化裁剪对平移不敏感，旧检查会接受平移后的多边形。实验文件缺失或无法解析也记录为未绑定而不是崩溃。全部通过才复用，`sourceRevisionId` 保留实验真实运行的版本（`workcell-ground-caliper-2026-10-02`）；任一不同则记录 `semanticBinding: not_bound` 和原因，页面显示需重算，绝不沿用旧结果。97 个观察全部通过；裁剪图因 OpenCV 版本插值最大差 6 个灰阶（源图与多边形完全相同，已记录）。重建后的语义块与已发布版本在结果、主题、待补证据、来源引用上逐项相同。attach 写入的 158 条 `policyContext.facts` 全部去掉：52×照片支持、52×测量地面、52×真实尺度验证，都标注为实验运行版本，另有 2 条厘米快照。页面改为从**加载版本**实时派生这些事实，每个对象的语义卡都有。

oneshot 新增 `--semantic-protocol`：重建结束并本地 `finalize` 之后，在**这个**版本上跑语义 GPU 作业，再 `finalize` 一次绑定；失败的支出写入同一 `spend-ledger.json`。本轮因网络限制未实际运行该 GPU 阶段。

## 5. 两轮独立对抗审查后的修复（均有回归检查与变异测试）

第一轮（7 项确认）：

| # | 审查发现 | 修复 |
|---|---|---|
| 1 | 语义绑定接受平移后的多边形（真实重放中右光幕照片 4 平移 (−6,−6) 像素仍被复用，裁剪差 253 灰阶） | 精确比对实验哈希过的绝对多边形；裁剪重采样差上限 32 灰阶 |
| 2 | 候选模型可装到已接受 `physicalBottom` 的对象上 | 拒绝：已有接受端点或已有候选的对象不再安装 |
| 3 | 被拒绝的构建已覆盖 `entity-*.glb`，旧报告与资产哈希错配；旧版端点文件导致 KeyError | 构建先写入临时目录，全部校验通过才替换；不再引用的网格删除；端点文件必须是 schema 2，否则明确提示重跑 `finalize` |
| 4 | real2sim 汇总页遇 null 尺度崩溃，且写死 `fence-0`/`post-box-1` | 按报告自身的成对端点与标签生成；尺度未知时显示原生单位 |
| 5 | 某个端点不在显示模型上时，差值仍显示 | 卡片、左右差、语义空间证据、空间提问都只在两个端点均绑定时读数 |
| 6 | 配对忽略 `geometryPlaneIndex`、无距离上限、同一横梁可被当成“左右”比较 | 平面身份用 `_fence_plane_index`；只配对水平距离 ≤ 光幕模型竖向尺寸 × 0.25 的横梁（真实数据 0.09/0.14 对上限约 0.65 native），否则不配对；同一横梁不做左右差 |
| 7 | 已完成运行目录含 `report-ui/`、`page/` 时，版本构建失败 | 复制运行目录时忽略页面、UI 与中断的临时目录 |

第二轮（审查第一轮修复本身，4 项确认 + 文档不实 4 处）：

| 审查发现 | 修复 |
|---|---|
| 用外轮廓多边形重测可见尺寸会收进透空背景（见第 1 节），新流程的目录值也会被覆盖 | 撤回重测；倾角精确换算到本版本地面，可见尺寸注明测量地面 |
| 语义文件可解析但字段为 null 时 `finalize` 崩溃（TypeError/AttributeError） | 记录为 `not_bound` |
| 候选文件名可与构建/导出生成的 `entity-*.glb`、`workcell-*.glb` 同名，被构建覆盖后版本自相矛盾 | 保留这两类名字，候选不得使用 |
| `finalize` 先写端点表再构建：构建被拒时端点表与旧报告不一致 | 端点表与报告、网格、评估一起暂存，全部通过才落盘；提供的参考尺寸属于输入，保留 |
| 文档：§3 “只差 entity-floor.glb”、§4 “只去掉两条快照”、“37/76”、线上 `?version=` 尚未发布 | 已更正；网址在本轮发布后才有效 |

同时处理了审查列出的三项可疑点：检查值对照表的“无检查值”说明改为由数据生成，一个检查值不再与同一对象的多个测点相减；版本构建器拒绝以候选版本作为根目录（否则候选会被标成主模型）；接受模型不得装到带未接受候选的对象上。

另修：导出改为新尺度状态时删除旧状态的 `workcell-*.glb`；安装函数核对传入目录等于磁盘 `objects.json`；候选文件不得覆盖共享或已有模型；先写模型、最后写目录；候选端点来源改写为“未接受的视觉候选”（不再称 source-supported）；浏览器检查中“逐个点选全部对象”移到“切换版本保留所选对象”核对之后（原顺序使该检查必然失败）。

## 6. 页面

- 版本切换：主模型 / G 候选。切换保留所选对象和照片；模型资产带所选文档 SHA；候选目录只放变化的 2 个光幕模型，其余文件仅在字节相同时共享。
- 光幕与围栏卡：两侧四个测点与四个差值，均为 native × 当前标尺；测点只在“显示的表示 = 测量的表示 + 资产哈希”时读取，否则显示未知。
- 语义：查询结果点击后选中对象、打开该版本的离地测量线并切到 3D 模型；对象卡“可用的空间证据”实时派生；有“在 3D 模型中显示该对象的离地测量线”入口。
- 空间提问：规则解析对象类别、左右、问题类型（离地/比较/角度/定位），数值只读本版本测点与当前标尺；缺事实明确返回缺失。示例（浏览器实测）：主模型“左右光幕谁更高？”→“左侧光幕测点更高：左 − 右 = 3.73 cm（左 28.44，右 24.71）”；候选 → “右侧光幕测点更高：左 − 右 = −0.40 cm（左 24.11，右 24.51）”。任意新名称的开放词汇检索需要文本编码器，本页不运行。
- EHS 规则证据链：每个版本把示例规则（`docs/policies/compiled_v2`，非认证）送入平台规则引擎；没有审核人确认适用性，引擎弃权（applicability unknown，machineResult null）；列出对象候选、规则哈希和仍缺的证据（适用性确认、操作员锚定米制尺度、整体高度、机器人工作区包络）。**没有安全结论。**
- 导出：卡尺 JSON 与浏览器下载 GLB 均写入 `revisionId`、`documentSha256`；离线 `workcell-conditional.glb` 元数据同样带文档 SHA。

## 7. 验证

- `scripts/check_workcell_revision_sync.py`（新）：同 ID 不同 GLB 哈希 → 旧端点被拒、被拒构建不改动任何文件、`finalize` 后重测；只改地面 → 高度与垂足全部重算、语义 `not_bound`；地面倾斜 2° → 可见尺寸与倾角按新地面重测、注入的目录旧值不显示；只改尺度和尺度 null → native 端点完全相同、旧尺度导出被删除；主模型↔候选 → 候选自洽、从不 accepted、不写物理离地、主目录逐文件不变，且不能覆盖共享文件、已接受端点或已有候选；平面身份互换、远处横梁、两根光幕共用一根横梁；改/平移多边形、照片、帧、缺失或被改的实验目录、被改或无法解析的清单 → 语义不复用；oneshot 语义阶段在本版本绑定、失败支出保留；卡片/模型顶点/垂足/倾角/语义/导出共用同一版本。地面倾斜 2° 时倾角按新地面换算、可见尺寸注明夹角；被拒的 `finalize` 不改任何派生文件；null 字段的语义文件记录为未绑定；候选不能占用生成文件名；候选目录不能作为版本根目录。变异测试：去掉以下任一守卫，检查都失败。
  - 多边形比对、清单交叉核对、接受端点守卫、相邻上限、平面身份、不同横梁条件、目录参数核对；
  - 倾角换算、TypeError/AttributeError 捕获、生成文件名保留、端点表暂存、候选根拒绝、候选上不装接受模型；
  - 先前的 4 个守卫。
- 更新：`check_workcell_photo_report_endpoints`、`check_workcell_endpoint_estimate`、`check_workcell_bottom_models`、`check_workcell_metric_export`、`web/checks/photo-scale-check.mjs`；新增 `web/tests/spatial-query-check.ts`（并入 `npm run check`）。
- 真实数据：`check_workcell_photo_report.py`（更新为排除点云上下文实体与点云表示）在两个版本上通过（52 对象、相机变换、模型边界、倾角与本版本地面一致）。28 个 workcell 检查中 24 个直接通过；`camera_pixels`（pycolmap 接口）、`photo_objects`（旧目录字段）、`vguard`（需 oneshot 中间文件 `guard-input.npz`）在交接提交 `0485e33` 上同样失败，`photo_report` 已更新并通过。
- 真实浏览器 `web/checks/photo-revision-check.mjs`（Chromium 141，软件 WebGL）：两个版本 52/52 模型加载、52 个对象逐个点选身份正确；“寻找光幕”→ 右光幕 → 语义证据 24.71 cm 与 3D 标注“右侧光幕底端减去旁边围栏下沿：6.16 cm”；标尺 10→20 cm 时语义证据与卡片同步翻倍；切到候选后同一对象读 24.51 cm，模型请求带候选文档 SHA 且光幕模型来自候选目录；下载 GLB 中光幕测点顶点回读等于报告值（主 0.24711 m / 0.28445 m，候选 0.24512 m / 0.24108 m）；卡尺 JSON 绑定候选文档；无控制台错误、无 API 请求。记录见 `evidence-2026-10-03/browser-revision-check.json`。
- 全量 `pytest`（两轮修复后）：1405 通过，6 失败；这 6 个在未改动的交接提交 `0485e33` 上同样失败（集合一致，属视频/runner 模块），不是本轮引入。

## 8. 负结果、诊断与未完成

- **光幕严格门槛为何不过：** 照片 1 残差最大（右/左最大 25.7 / 43.1 原图像素，门槛 7.7），留一照片预测在每张照片都是 16–50 像素。`scripts/workcell_camera_offset_diagnostic.py` 去掉一张照片重拟合两根光幕，再用一根光幕在该照片的边界求解相机修正并套到另一根：只转动（0.4–2.6°）或刚体修正都**不能**让另一根进入门槛（7/8 情况更差）。单张照片的相机偏差不能解释失败；更可能是跨图面身份（正面/侧翼、阶梯外壳）或矩形面模型不成立。证据：`evidence-2026-10-03/camera-offset-*.json`。下一步应逐图核对面身份，而不是再调相机或加迭代。
- **对象对照审计：** `scripts/workcell_object_audit.py` 把每个对象模型经保存相机投到每个来源视图，与分割多边形求 IoU（分诊信号，不是物理精度）。机器人、小车、护板 0.60–0.76；光幕原语与候选只覆盖分割的一部分（0.12–0.44，模型比分割的立柱窄）；`fence-0` 照片 2 的观察是 2×8 像素的碎片（IoU 0），是目录中的伪观察。证据：`evidence-2026-10-03/object-audit-*.json`。
- **因网络/凭据未运行：** 完整 oneshot 重跑与性能复测（P2）、语义 GPU 阶段、参考图 prelist 与开放词汇检索（P1）。代码路径已接好，需要在允许 Modal 的环境重放。
- **未开始或仍待研究：** 左右围栏等价下沿、三尺寸联合标尺与局部地面、折弯护板实物角度、跨图语义独立标注、公司 GPU/SSH 迁移、三视频主线。

## 9. 复现命令

```sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:modal_apps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# 1. 公开数据 → 运行目录（约 3 s；含语义实验哈希过的目录 source-objects.json）
python scripts/workcell_replay_inputs.py --release EXTRACTED_RELEASE --pages PAGES/workcell-photo-direct --out RUN
# 2. 主模型 + G 候选两个版本与一页报告（约 14 s；先 vite build 照片报告）
python scripts/workcell_photo_revisions.py --root RUN --out NEW_DIR --viewer-assets PAGES/workcell-photo-direct/viewer-assets \
  --main-id workcell-main-2026-10-03 \
  --candidate PAGES/workcell-photo-direct/housing-boundaries/evidence workcell-housing-g-2026-10-03 "G 光幕封闭外形候选" \
  --evidence-url housing-boundaries/
# 3. 检查
python scripts/check_workcell_revision_sync.py
python scripts/check_workcell_photo_report.py NEW_DIR/workcell-main-2026-10-03
(cd web && npm run check && node checks/photo-scale-check.mjs)
PLAYWRIGHT_FROM=/qa/package.json node web/checks/photo-revision-check.mjs NEW_DIR/workcell-main-2026-10-03/page OUT
# 可选：G 外壳拟合 CPU 重放、相机偏差诊断、对象审计
python scripts/workcell_camera_offset_diagnostic.py --root RUN --sources image_01.jpg image_02.jpg image_03.jpg image_04.jpg --out NEW_DIAG
python scripts/workcell_object_audit.py --root NEW_DIR/workcell-main-2026-10-03 --out audit.json
```

完整 oneshot（需 Modal 凭据与网络）：在原命令上加 `--semantic-protocol docs/workcell-photo/semantic-match-protocol.json`，即在同一版本上完成重建、测量、语义与打包。
