# 跨照片工位对象身份统一 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 同一工位的同一实物在多张照片中复用一个稳定实体；自动关联、人工核对、Agent、模型、CAD 和报告共同使用该身份，保留全部来源证据。

**Architecture:** 复用现有 Observation / Entity、几何关联、编辑事务、任务和不可变 Publication。修正共享身份操作，再让分析、修复和人工应用调用同一逻辑；在现有右侧详情内完成核对。追加照片在同一工位分支上扩展来源，不清空已有实体。

**Tech Stack:** Python / NumPy / Pydantic / PostgreSQL、现有模型 adapters、React / TypeScript、现有 WebGL 与 Deep Chat；不增加数据库、队列、ReID 服务或前端依赖。

---

## 0. 本次范围、现状与交付顺序

本文是实施计划，未表示下述修复已实现。源码工作目录为 `/Users/adam/Desktop/Tesla/panoptes-platform`，审查基线为 `6e9ac89`。相对路径均相对于该目录。

用户要求是一套通用物体身份，不是把列表中同名的行隐藏。明确以一个项目表示一个工位；跨项目不会自动关联。第一交付覆盖同一组 1–4 张照片及当前报告；第二交付覆盖在同一工位继续追加照片。两个交付分别验收。

当前冻结报告：

```text
publicationId = f4e5ca43-543d-4624-8ea0-27aa2843e6e4
revisionId    = a97267c4-d435-43ee-915f-a59f2bb44ed6
projectId     = a2b7c04d-0162-488b-b7db-37711a37ea62
```

| 已核实事实 | 影响 |
| --- | --- |
| 68 条可见对象记录、84 条照片观察；9 个实体已有来源确认的跨图身份，59 条单图记录待核对 | 当前不能把 68 宣称为独立实物总数，也不能预设最终应减少到多少 |
| 84 条观察仅 58 条有 maskAssetId；另 26 份原始 mask 存在于冻结源目录 | 先把真实 mask 和像素映射纳入资产，才能对全部已存观察运行关联 |
| 旧离线检查的 7 对几何接受结果未应用，其中存在部分／整体风险 | 几何匹配分数不能直接成为物理身份真值 |
| 新图分析调用 `associate_observations`，当前报告是导入快照 | 统一输入合同后用产品关联任务重新计算，不能只改导入器状态 |
| 自动关联直接删实体；手工合并追加所有模型；split 清空几何和测量 | 必须统一无损身份操作，随后才开放核对按钮 |
| `create_capture` 从 `empty_document()` 开始 | 追加照片会替换场景头，尚未形成工位的增量分析 |

依据：`docs/platform/ASSOCIATION-RELEASE-QA.md`、`spatial.py:135–235`、`reconstruction.py:356–403`、`repository.py:75–173`、`postgres.py:271–296`。文中旧实验数字是已保存审计结果，本次没有重新运行模型或将候选合并。

## 1. 最终用户操作

左侧每个实体一行，行内展示照片编号和证据数量；点击实体后，中间原图、3D、CAD、平面保持同一选择。切照片只切换 Observation。右侧显示该实体的数据、当前模型、不同照片证据、判定及反馈。

右侧增加“跨照片核对”：两张带框或 mask 的来源照片并排，标出照片、对象及关联理由；用户可从候选进入，也可主动选择另一条记录比较。

| 操作 | 保存行为 |
| --- | --- |
| 同一物体 | 预览合并的观察、保留 ID、当前模型及受影响引用；有编辑权者应用一次编辑批次 |
| 不同物体 | 保留两实体，保存排除关系；后续自动关联必须读取 |
| 暂无法判断 | 保存核对记录与原因，不确认相同或不同，不隐藏记录 |
| 发现原来合错 | 明确分配照片证据与资产，拆分为新版本；后续关联不自动合回 |
| 在公开报告提出建议 | 保存用户明确提交的对象对及选定反馈；不直接修改发布快照 |
| Agent 辅助判断 | 读取同一对对象的证据，生成同一编辑提案；收到提交成功后才显示已应用 |

初次打开仍不默认选择对象；选中后保留当前照片描边。不要增加另一个独立“关联工作台”，也不要再次复制对象清单。

计数示例：`68 条对象记录 · 9 组已有跨图关联 · 59 条待核对 · 84 条照片观察`。这些口径并行展示，关联组数和待决记录不能相加后称为已验证实物总数。

## 2. 数据和共享操作合同

### 2.1 一次明确的场景合同升级

新工作版本使用 scene schema v2，通过显式迁移从 v1 生成新 revision；不原地改写旧报告文档，不在每个调用方添加猜测旧字段的分支。旧 Publication 保留其固定 v1 文档与 rendererVersion；生产切换前验证旧链接仍由对应版本阅读。

v2 的新增字段为本次身份与增量需求服务，不新建平行身份表：

```ts
type IdentityEvidenceRef =
  | {kind: 'observation'; observationId: string; observationRevision: number}
  | {kind: 'asset'; assetId: string; sha256: string}
  | {kind: 'method'; name: string; version: string; configSha256: string};

type IdentityDecision = {
  id: string;
  decision: 'same' | 'different' | 'undecided';
  source: 'geometry' | 'manual' | 'source_binding';
  baseRevisionId: string;
  entityIds: string[];             // 决策发生时的身份，允许在 lineage 中已退役
  observationGroups: string[][];  // 每组是决定前的一个身份；different 禁止跨组自动合并
  survivorId: string | null;       // same 时必填
  evidenceRefs: IdentityEvidenceRef[];
  reason: string;
  supersedesDecisionId: string | null;
};

// SceneDocument v2
// captureIds: string[]；identityDecisions: IdentityDecision[]
// Observation.captureId: string
// Entity.activeModelRepresentationId: string | null
// Entity.measurementEvidence: 来源测量记录数组，结构见下文
```

上述来源引用用 Pydantic discriminated union 校验；不接收任意本地路径。source_binding 仅用于迁移已有明确来源身份，不包装为本次几何证明。Agent 提议经用户应用后，来源是 manual，并另外保留 agentTurnId。

每条 `measurementEvidence` 保存 `id / measurementKey / originalMeasurement / sourceRevisionId / sourceEntityId / observationRefs / representationId`；originalMeasurement 是完整原测量文档，沿用现有单位、方法、误差与事实来源校验。`measurements` 保留当前选用的量测并带 `measurementEvidenceId`；多来源冲突且未选定时当前值为 null，右侧仍展示各来源值。当前模型尺寸由 active representation 派生，不能冒充 observed measurement。

`captureIds` 单元素初始化不等于追加流程已实现。新分析任务固定本次 captureId、baseRevisionId 和几何工作集；不依赖列表最后一项猜任务输入。

人工“不同”关系绑定明确的观察组及决策时版本；mask 修正不自动撤销人工身份决定。冲突必须由明确 supersedes 的新决定解决。自动决定可随新证据重新评估，人工决定优先；规划分支不自动改写重建分支身份。

### 2.2 合并、拆分与当前模型

新增 `ehs_spatial/platform/identity.py`，只负责现有合并／拆分／身份约束和引用映射。`repository.apply_operations` 和 `reconstruction` 都调用它；删除自动关联内部另写的实体删除逻辑。

合并规则：

1. 校验同项目、同分支固定基础版本、实体及观察归属，拒绝与有效人工排除关系冲突的自动合并。
2. 手工操作明确 survivorId；自动扩展已确认组时保留原组 ID。两个已有确认组的合并必须有跨组证据；确定性保留规则写入决策，不随输入顺序变化。
3. 保留 observationRefs 的并集；不可修改原框、mask、相机、原始 ID 和哈希。
4. 所有原 representations 保留自身姿态及来源。替代生成模型只允许一份 active；保留 survivor 原 active，没有时仅在唯一候选情况下自动选定，多候选则明确选择。
5. 同一实物的替代模型不当作多个组件一起渲染。真正组件继续使用独立实体及现有 groupId；本次不添加机械装配或关节系统。
6. 全部原测量保留来源版本／实体／观察引用；有冲突的测量并列保存，当前显示值来自明确的证据或 active 模型，不取平均、不以 survivor 值覆盖其他来源。
7. 移动、材质、尺寸编辑仅作用于 active 模型；原始观测几何及其他来源模型不被同一绝对 transform 覆盖。
8. 新版本的当前附件按明确字段重映射；历史证据正文保持不变。
9. 在同一事务记录身份决定、逆操作、新 revision 并推进 head；失败不留下半次合并。

split 必须提供每组观察、representation 和测量来源分配。依赖来源跨越多组且无法唯一归属时，保留为旧版本来源证据并要求明确模型归属，不能悄悄丢弃或复制为多份当前实物。恢复刚合并前状态优先复用原 undo；普通 split 产生明确的排除决定。资产内容和历史版本始终可追溯。

### 2.3 引用更新边界

| 数据 | 新 revision 的处理 |
| --- | --- |
| 当前 entity 选择、工具目标、annotations、新生成任务目标 | 通过本版本 identityDecisions / lineage 解析，退役 ID 可解析到唯一实体才继续；split 多解返回需选择；已排队任务的固定输入不能改写 |
| `reportEvidence.objects[].entityId` | 更新当前附件；每个 view 的 observationId 不变 |
| `reportEvidence.historical.inventory[].entityIds` | 更新平台附件，不改 sourceRecordId、sourceFrameId 或 inventoryIndex |
| `reportEvidence.historical.cad.regions[].entityIds` | 更新平台附件；CAD polygon、坐标系、比例和来源图纸内容不变 |
| 原生 source ID、run ID、asset hash、jsonPointer、历史 violations / facts | 原样保留；禁止递归替换所有名为 entityId 的字段 |
| EHS evaluation / review | 保留旧版本结果；当前版本重新评估，旧确认不继承 |
| 补证据请求与 manual evidence | 保留原 finding 和证据范围；新 finding 只能通过明确身份决定和唯一观察归属连接，不能绕过原有作用域检查或复制到全部拆分对象 |
| 已发布报告及其访客私有聊天 | 保留原 publication/revision/entity 绑定；不迁移私有聊天权限 |

split 后同一来源 record 的多个 views 可能属于不同子实体：以稳定 observationId 分别确定 view 的当前归属。record 的单一 entityId 仅在所有 views 都唯一属于同一实体时填写，否则为 null 并显示来源记录关联多个实体。record 级测量／判定不广播到子对象；原 record 不删除，不复制其整体结论。新增检查必须覆盖两张 views 拆到两个实体后，照片仍能分别定位、整体指标未被复制。

只读访客显式提交的身份建议可进入项目复核；提交内容是选定的反馈及对象对，不默认把整段私有会话开放给项目管理者。复核仍对当前基础版本重新校验。

## 3. 通用关联计算

### 3.1 固定输入与输出

复用持久化任务，新增 `reassociate_scene` 任务类型：

```json
{
  "requestId": "<uuid>",
  "branchId": "<uuid>",
  "baseRevisionId": "<uuid>",
  "kind": "reassociate_scene",
  "inputs": {"captureIds": ["<uuid>"]},
  "config": {"associationConfigVersion": "workcell-identity-v2"}
}
```

配置由服务端校验；任务冻结观察版本、mask 和几何哈希、人工决定、算法参数。输出是关联证据、接受／待核对／拒绝原因，以及通过共享操作生成的新 revision。身份计算本身不调用 VLM、分割或生成模型；缺失阶段需要明确的独立任务，不由刷新触发。

### 3.2 修正现有算法，不另建识别路线

- 保留原图像素映射、正深度、遮挡、独立支持像素、双向投影和双方最佳匹配。
- 当前阈值 32 / 0.65 / 3% / 0.15 仅作为基线。新增有效支持占比及空间覆盖统计，防止只剩一小片深度一致区域时获得虚高包含率；新阈值通过固定实验确定并版本化。
- 区分关系：有支持的相同候选、明确冲突、不可比较。不可见不能当作不同物体，低分也不能一律等同几何冲突。
- 先以已有确认组作为身份约束再评估候选，不能要求新增一张照片重新证明已确认组中每一对历史观察。
- 新观察可通过组中可验证视角加入；必须在这些视角中唯一匹配，并且不违反组内可验证关系、同照片独立实例或人工排除关系。
- 两个尚未确认的组不因 A–B、B–C 两条边盲目传递合并。保留现有保守跨组要求，实体级扩展与新组合并分别测试。
- 同张照片的两个实例不自动合并；只有原图、mask 内容及映射完全一致的重复观察可去重。按钮与按钮盒、围栏与光幕、整体与零件不能因包含或同名合并。
- 已保存的明确部分／整体关系作为禁止自动合并的约束。缺少对象粒度依据的重叠候选进入核对；不把类别字符串当真值。
- 运动的料车或不同姿态的机械臂保留观察状态，几何不一致时不自动融合。人工或 Agent 可提出同一实体判断，但不能因此平均位置或覆盖旧姿态。
- 无 mesh、缺深度、未知类别、小 mask 都保留 Observation 和 Entity，并输出具体缺失依据。

自动关联的发布条件同时包含误合并和正确关联覆盖；不能通过把所有对象都设为待确认获得“零错误”。

## 4. 同工位追加照片

上传入口明确分为“创建工位”和“为此工位补充照片”。一次上传继续 1–4 张，补图带已有 project、branch、baseRevision；不创建空场景覆盖原 head。

每次 Capture 不可变。新任务在基础 revision 上添加观察和来源 Capture，原实体 ID、人工决定、已接纳模型及编辑保留。重传完全相同的照片按内容哈希及输入转换识别来源重复，不新建一批物体。

Capture 身份与几何解身份分开：同一张参考照片可参与多次 joint geometry。直接复用持久化几何阶段资产 ID 作为 geometrySolutionId，不新建几何数据库；每个解有自己的 frame ID，camera ID 由 geometrySolutionId 与 imageId 确定。当前工作版本明确保存 `geometryBindings[imageId] = {geometrySolutionId, cameraId}`，观察的支持／mask 栅格映射引用具体解。绑定改变只发生在新 revision；保留旧几何解全部资产和相机。显示与关联必须消费显式绑定，禁止按 imageId 找第一台相机，也禁止按 imageId 替换其他解的相机。Task 7 才引入该增量字段并与补图功能共同验收。

跨 Capture 先建立坐标关系，再进行空间身份比较。使用已有 joint geometry provider，在固定旧参考照片与新照片上计算；每个工作集最多 4 张，最多 3 张新图加 1 张旧参考。4 张新图分为有明确共同参考的工作集，分别缓存并验证，不能仅因共享项目 ID 就使用同一坐标系。

共同参考照片提供对应像素的 3D 点对。复用 `scripts/research/probe_glb_alignment.py` 中 NumPy 相似变换求解的数学实现，抽入 `spatial.py`，移除样例路径和固定单位阈值。以静态背景、非退化且覆盖充分的点估计 Sim(3)，独立保留点验证残差及相机回投；不得用未确认对象的最近质心来证明注册。

```text
p_project = s * R * p_capture + t
R_camera_project = R * R_camera_capture
t_camera_project = s * R * t_camera_capture + t
```

K 保留；相机旋转保持正交，不把含尺度的 3×3 直接塞入 cameraToWorld。原始几何、相机与尺度不覆盖，派生坐标关系保存两端 frame ID、点对来源、算法和验证指标。

注册失败或缺少重叠时，新增照片仍可阅读和选中，跨图身份保持待核对；不能报告已经合并到同一空间。该状态属于证据不足，不新增替代模型路线。用户可提供有重叠的照片或明确身份判断；手工身份判断不会自动证明坐标变换。

这部分有独立发布门槛：共同照片不保证两套重建可由单个相似变换准确对齐。验证不过，不把“同工位增量空间关联”标为完成，不宣称模型已达到稳定重建精度。

## 5. 实施任务与文件责任

### Task 1：冻结真实身份验收集、补齐来源资产

**Files:** 修改 `scripts/import_public_scene.py`、`scripts/import_report_evidence.py`；扩展 `tests/test_platform_import.py`、`tests/test_platform_report_evidence.py`；新增 `scripts/research/evaluate_object_identity.py` 和 `docs/platform/IDENTITY-ACCEPTANCE.md`。

- [ ] 保存当前 publication、revision、84 条观察及所有资产的哈希清单；记录当前 68/9/59 计数。
- [ ] 从 `ASSOCIATION-RELEASE-QA.md` 的冻结源读取 26 份实际 mask，按原始分辨率／canonical 网格及 pixelMapping 注册不可变资产；逐项验证原文件哈希，不从显示轮廓重造证据。
- [ ] 为每条观察记录几何是否存在、frame ID 和坐标来源；不同源坐标先注册验证，不能静默拼接。
- [ ] 人工对照原照片标注可判定正例、不同实物、部分／整体及未知案例。至少覆盖九个已有跨图组、重复外观的小物体、同帧重复 crop、运动对象；已存 source ID 是来源证据，不替代照片核对。
- [ ] 冻结一个未参与阈值调试的工位样例；标注没有真值的项为不可评分。
- [ ] 评测脚本支持 `--manifest <path> --output <path>`，只读取冻结资产，输出实际样本数、pair/group 指标、失败 ID 和原因；不得调用模型或写生产数据库。
- [ ] 运行下述已有检查后单独提交资产合同与验收工具。

```sh
.venv/bin/python -m pytest -q tests/test_platform_import.py tests/test_platform_report_evidence.py
```

**验收：** 84 条原观察和历史照片轮廓保留；26 份 mask 全部可按哈希取回；未知案例不计为自动正确。

### Task 2：统一身份事务和模型归属

**Files:** 新增 `ehs_spatial/platform/identity.py`；修改 `contracts.py`、`repository.py`、`postgres.py`；扩展 `tests/test_platform_backend.py`。同时更新 OpenAPI 生成 DTO。

- [ ] 写失败用例：两实体合并后观察、资产、测量来源守恒；两份不同姿态生成模型只选一份 active；移动 active 不改变来源候选；split 不丢来源；人工 different 阻止再次自动合并。
- [ ] 实现场景 v2 DTO、显式 v1→v2 新 revision 迁移和 §2 的边界校验；已有单模型选择直接迁移，多模型依实际来源确定，不按数组首项猜当前模型。
- [ ] 将 merge/split 改为唯一共享身份操作，完成命名字段的引用重映射；保留服务器原值生成逆操作的机制。
- [ ] 加入 `recordIdentityDecision` 编辑操作；same 必须与对应 merge 在同一批次提交，自动与手工决定都进入版本记录。不同／未决决策也有 requestId 幂等和 baseRevision CAS。
- [ ] 验证操作重试、撤销重做、分支冲突、跨项目凭证、同一来源多次合并及拆分后的歧义 ID 解析。
- [ ] 运行后端检查及 `npm --prefix web run schema`，仅提交本任务文件。

```sh
.venv/bin/python -m pytest -q tests/test_platform_backend.py tests/test_platform_report_evidence.py
```

**关键断言模式：**

```python
assert after_observation_ids == before_observation_ids
assert after_source_asset_hashes == before_source_asset_hashes
assert current_model_ids == {chosen_representation_id}
assert undo_document == original_document
assert original_publication_hash == stored_publication_hash
```

这些变量在现有 repository 测试的真实提交前后计算，不写只验证手工构造常量的测试。

### Task 3：通用实体级关联与重跑任务

**Files:** 修改 `spatial.py`、`reconstruction.py`、`panoptes_worker/__main__.py` 及现有任务 kind 校验调用方；扩展 `tests/test_platform_spatial.py`、`tests/test_platform_reconstruction.py`。

- [ ] 写失败用例：确认的 A/B 组新增 C，C 只在 B 可验证仍能正确扩展；可验证冲突阻止扩展；两个相同按钮不误合并；小片支持不产生虚高确认；different 决定重跑有效。
- [ ] 保留 `_direction` 的投影检查，增加原始支持、可见支持和覆盖率统计，输出明确关系类型。
- [ ] 以现有确认组和人工排除约束计算实体级候选，使用 §3 的扩展规则；沿用版本化基线参数开展固定实验。
- [ ] 删除 `_associate_and_surfaces` 的直接删实体分支，全部结果经过 Task 2 的共享操作。
- [ ] 修正后续 observed surface 写入：不得因选择一张主观察重新赋值 representations 而删除其他来源表示；新的表面按 observation/state 明确归属。
- [ ] 让初始分析、单物体补证据和 `reassociate_scene` 共用上述入口；任务只用冻结资产，缺失输入返回分阶段原因。
- [ ] 校验重启／重复派发不增加模型调用，分支 head 已变化时产物保存为独立 revision 而不覆盖。
- [ ] 运行关联和重建测试及固定清单评测，提交共享路径；候选失败清单随结果保存。

```sh
.venv/bin/python -m pytest -q tests/test_platform_spatial.py tests/test_platform_reconstruction.py
```

**验收：** 充分可见正例自动关联，人工否定不被重跑覆盖，两个同外观实物和整体／零件不误合并；保持全部原始观察。零模型调用的重跑与真正新增推理分别统计。

### Task 4：四视图、测量、CAD、Blender 共同消费当前实体

**Files:** 修改 `web/src/core.ts`、`web/src/ReportScene.tsx`、`web/src/scene-semantics.ts`、`web/src/viewer/native-viewer.ts`、`web/src/ReportEvidence.tsx`、`web/src/ReportObjectFindings.tsx`、`ehs_spatial/platform/blender_export.py`；同时检查 `policy_engine.py`、`policy_repository.py` 的证据和补证据引用。扩展 `web/checks/report-scene.mjs`、`web/tests/report-context-check.mjs` 和现有 Blender／Policy 检查。

- [ ] 搜索全部 `representations`、`currentModelTransform`、`entityId` 消费点，明确当前模型、来源观测、历史证据三种用途；不全树替换 ID。
- [ ] 所有当前模型消费点只使用 activeModelRepresentationId；当前范围、尺寸、CAD 当前投影和 Blender 导出使用同一姿态。
- [ ] 观测层保留原照片对应的状态和坐标；禁止把不同时间的机器人或料车姿态合成一个观测实体网格。
- [ ] 当前详细 CAD 的平台附件跟随稳定 ID，来源图纸轮廓不移动；当前模型投影与历史源 CAD 分别保留单位和版本。
- [ ] 新评估重新绑定适用对象；人工证据有唯一来源才继续适用；补证据请求通过明确身份决定连接新 finding，保留旧请求，拆分多解须重新核对。
- [ ] 合并后使用保留 ID，切照片只换 observation；快速 A→B 选择、延迟 mesh 和旧回调不改变当前选择。
- [ ] 导出真实合并 revision，重开 Blender 核对 active 模型、对象 ID、独立来源资产与姿态；验证原报告资产不变。

```sh
npm --prefix web run build
node web/checks/report-scene.mjs
node web/tests/report-context-check.mjs
.venv/bin/python -m pytest -q tests/test_platform_spatial.py
```

### Task 5：右侧核对、Agent 和反馈

**Files:** 修改 `web/src/WorkcellReport.tsx`、`ReportScene.tsx`、`AgentPanel.tsx`、`report-scene.css` 及其现有双语消息文件；后端修改 `agent_service.py`、`feedback.py` 和反馈 DTO；扩展 `web/checks/agent-panel.mjs`、`tests/test_publication_feedback.py`、`tests/test_platform_integration.py`。

- [ ] 右侧加入两照片对比及三种决定，沿用对象列表搜索选择第二个对象；不新增页面或重复清单。
- [ ] 展示合并前后预览：保留 ID、照片数量、当前模型选择、测量冲突、CAD 附件和 EHS 需重新评估。
- [ ] GUI 和 Agent 应用相同 edits 请求；检测 baseRevision 冲突后保留提案，让用户重新核对。
- [ ] 公网只读报告仅提交建议；访客明确选择的建议进入复核，私有完整聊天不可被管理者直接读取。
- [ ] 同时验证不同物体、未决、重复提交、无 mesh、小 mask、切语言、切对象 A→B 的会话隔离。
- [ ] 点击后真实查看原图轮廓、3D、详细 CAD、平面和右侧详情；公开快照上不得出现可直接改原场景的 Apply。

```sh
node web/checks/agent-panel.mjs
.venv/bin/python -m pytest -q tests/test_publication_feedback.py tests/test_platform_integration.py
npm --prefix web run build
```

### Task 6：重算当前报告并发布新版本

**Files:** 复用 `scripts/import_public_scene.py`、`scripts/import_report_evidence.py`、`scripts/export_platform_publication.py`、`publication_site.py`；修改 `web/src/App.tsx` 的报告阅读器版本路由，更新 `docs/platform/IDENTITY-ACCEPTANCE.md`。

- [ ] 在当前 revision 派生的新重建版本上补齐输入，用 `reassociate_scene` 计算，不按 BOR1 物体名称或 ID 写合并清单进业务代码。
- [ ] 对人工验收集检查所有自动接受项；难例通过 Task 5 的普通核对机制处理，记录决定及证据。
- [ ] 输出逐项变更：原实体→当前实体、观察数量、模型选择、测量冲突、未决原因；不预设结果数量。
- [ ] 重建当前 CAD 投影、模型导出，并对新版本执行可用的 EHS 评估；旧历史规则结论继续标为历史，不改称当前合规。
- [ ] 新建 Publication。验证旧 publication/revision/资产哈希不变，公网新报告加载真实新版本。
- [ ] 在切换前保存并托管现用 v1 前端构建，发布 manifest 固定 rendererVersion、schemaVersion 和构建哈希的映射。`App.tsx` 读取 publication 后将旧版本交给对应只读构建，保留原 URL 中的图片／对象选择；新 v2 工作版本只用 v2 合同。在 v2 部署后使用原旧报告链接检查，而非另造归档链接代替验收。两种报告导航均在本网站，历史阅读器不是第二套项目写入路径。
- [ ] 使用 computer use 在公网完成不同照片选同一物体、无模型物体、候选核对、取消选择、手机布局和直接刷新；保存版本、选择 ID、截图和失败项。

**第一交付门槛：** Tasks 1–6 全部通过，才能称“当前报告和同批新图片采用统一身份路径”。新增页面或计数变化本身不算完成。

### Task 7：同工位补图与坐标注册

**Files:** 修改 `postgres.py`、`api.py`、`contracts.py`、`reconstruction.py`、`spatial.py`、`web/src/App.tsx` 的上传调用方；扩展 `tests/test_platform_backend.py`、`tests/test_platform_spatial.py`、`tests/test_platform_reconstruction.py`。

- [ ] 写失败用例：原工位已有对象及编辑，追加照片后全部保留；相同照片重试不新增实体；新 capture 不覆盖旧 source geometry。
- [ ] 将新建工位与补图做成明确的请求语义；补图从固定基础版本开始，不再调用 empty_document 替换场景。
- [ ] 实现 §4 的固定工作集和缓存；旧照片的发现、分割与模型继续复用，只重算需要的联合几何与关联。
- [ ] 将 `_bind_geometry` 的 Capture 唯一 frame 及按 imageId 替换相机逻辑改为解级身份；`ReportScene.tsx` 和关联入口都读取固定 geometryBindings。测试同一参考图参与两个 K、姿态、尺度不同的解，旧来源不变且当前回投选择正确相机。
- [ ] 抽取 NumPy 相似变换求解，验证非单位尺度、旋转相机、退化背景、动态物体、无重叠和残差不达标案例。
- [ ] 注册通过后生成派生坐标输入 Task 3；注册失败记录原因，不制造共同坐标或确认新的独立实物数量。
- [ ] 检查并发编辑、取消、费用 reservation 与迟到结果；完成任务后只提交一个装配版本。
- [ ] 真实未参与调试样例完成“首次分析→补图→身份延续→编辑→发布→Blender 重开”全流程。

```sh
.venv/bin/python -m pytest -q tests/test_platform_backend.py tests/test_platform_spatial.py tests/test_platform_reconstruction.py
```

**第二交付门槛：** Task 7 的真实坐标注册和端到端验证通过；模型许可／质量／预算门槛沿用原平台要求，不因新增身份功能绕过。

## 6. 量化验收与并行安排

| 项目 | 通过要求 |
| --- | --- |
| 数据守恒 | 原观察 ID、源资产哈希、来源测量与原生 CAD 证据全部可追溯，零静默丢失 |
| 身份正确性 | 固定可判定负例、相同外观、父子物体测试集零误合并；按样本量报告，不声称总体零错误 |
| 正例覆盖 | 固定充分可见静态正例组须自动关联；报告 pair 和 entity-group 两类覆盖，不能全待决通过 |
| 决策稳定性 | same / different / undecided、合并／拆分及撤销重跑符合记录，输入排序不造成 ID 漂移 |
| 模型一致性 | active 模型在四视图、尺寸、CAD 当前投影、GLB 和 Blender 一致，来源模型不叠画或误移 |
| 访问与版本 | 访客不改原报告；旧评估、旧审核、旧私有会话不改归属；CAS 冲突不覆盖 |
| 新图通用性 | 同批未见工位和补图分开验收，无样例路径、名称或对象 ID 特判 |
| 耗时与成本 | 记录 mask/geometry 读取、候选计算、装配、上传、渲染时间；缓存、纯关联、新模型调用分别统计 |

分工建议：后端负责 Task 2；几何线负责 Tasks 1、3、7；前端线在 Task 2 合同固定后做 Tasks 4、5；集成负责人核对跨层引用并执行 Task 6。`contracts.py`、`identity.py`、`core.ts` 分别指定单一编辑负责人，其他人只消费合同，避免同时改同一身份逻辑。

每个任务先运行一项能暴露当前错误的检查，再修改共享入口；检查通过后单独提交，相关新失败才扩大检查范围。数据库检查不能以跳过代替验收。完成时交付代码、公开新报告、逐实体映射、真实配对验收清单、模型／CAD／Blender 一致性结果及仍未决项。

## 7. 计划自查

- [x] 同一实物多照片、当前报告、未来同批照片和同工位补图都有明确任务。
- [x] 保留物体、照片、来源 CAD、3D 资产、测量、反馈、EHS 与历史边界。
- [x] 自动、GUI、Agent 共用身份操作；已发布报告不会被直接覆盖。
- [x] 复用现有依赖和任务／事务／反馈能力，没有按名称去重、UI 隐藏或样例特判。
- [x] 同一身份不冒充完整重建、准确放置或真实米制标定；模型及注册发布门槛单独保留。
- [x] 正例自动覆盖与误合并一起验收；原始记录数量不会再冒充独立实物数量。
