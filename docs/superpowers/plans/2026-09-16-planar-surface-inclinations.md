# 自动局部平面倾角 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task in the current task. Do not spawn agents without separate authorization.

**Goal:** 工位内每个有可靠几何的对象，自动识别所有达到支持面积的局部平面，计算相对地面倾角并保存；报告可筛选和显示非竖直面。

**Architecture:** 复用当前对象/网格/地面坐标、NumPy/Open3D、发布前计算和报告读取方式。新增局部面提取，复用角度数学和三维标注；折弯保留原有语义。算法目录为统一说明入口，不建设独立服务或配置后台。

**Tech Stack:** Python、现有 NumPy/Open3D、FastAPI、React/TypeScript；无新增第三方依赖计划。

## 产品定义与范围

用户圈出的左右斜板都是候选。判断单元是对象内的“连续局部面”，不是完整对象的主平面。
- 地面倾角 θ：水平 0°，竖直 90°；同时显示偏离竖直 δ=90°−θ。
- 不垂直包括水平物体表面；地面自身作为参考排除。未知地面归属不得仅因水平就排除。
- 足够面积按局部实际支持面积判断；不要求占整个物体 10% 或总共 60%。
- 微小网格三角片不独立成为面；两片不相连的共面物体不得合并为一个面。
- 同一板的正反面应归为同一个薄板局部面，避免重复结果。
- 先计算并保存所有可靠面的角度，再按偏离竖直筛选显示；筛选不抹掉垂直面的计算结果。
- 角度、面积、残差、来源和未输出原因都保存；模型角度不称为现场实测。
- 本次不新增风险结论、不改变身份合并/工位范围逻辑、不自动重建缺失模型。
- 用户新图目前是截图证据，尚未证明存在对应已配准模型；验收中必须核对其真实源照片/几何，不能拿旧护板结果冒充这两块斜板。

## 统一记录

唯一入口：[算法与分析流程总表](../../algorithms/README.md)。本次更新 A10/A11；其他条目的未解决状态仍保留。
每次实现/参数变更/发布都更新方法版本、来源指纹、检查命令和线上验收，避免只有进度口头说明。

## 任务 1：建立可复现的局部面验收集

**Files:** Create `tests/check_planar_surfaces.py`; modify this plan and `docs/algorithms/README.md` with measured parameter evidence.

- [ ] 写最小 assert 测试，输入为程序生成的已知角度三角网格：0°、30°、60°、90°；包含同一对象三个局部面、窄长面、双层薄板、断开共面板、圆柱/噪声面。
- [ ] 增加旋转整个对象与地面后角度不变、改变三角剖分密度后面积与面数量稳定、地面缺失/无效时无倾角的断言。
- [ ] 保存真实中央/左右板与用户新图对应输入清单（asset/hash、对象、坐标、地面）；只有截图的条目标明没有三维角度真值。
- [ ] 校准最小支持面积与平面残差配置：对上述连续面、碎片和噪声输入做参数对照，选择保留窄长板且排除噪声碎片的配置，并把数值、单位、通过/失败样本写入算法目录后固定版本。未经此检查不得选一个“看起来合理”的全场景面积比例。
- [ ] Run `.venv/bin/python tests/check_planar_surfaces.py`，确认当前实现因缺多局部面入口而失败；提交该可复现用例。

预期接口在任务 2 定义，测试使用 `extract_planar_surfaces(triangles, config)`；config 必须显式包含 `minAreaNative2`、平面距离容差和法向容差，不能暗中使用每对象总面积比例。

## 任务 2：局部面提取与倾角计算

**Files:** Create `ehs_spatial/platform/planar_surfaces.py`; modify `ehs_spatial/platform/scene_measurements.py`; test `tests/check_planar_surfaces.py` and existing `tests/check_scene_measurements.py`.

- [ ] 定义 `extract_planar_surfaces(triangles, config)`，返回确定性排序的面：`surfaceId`、`normal`、`center`、`boundary`、`areaNative2`、`rmsResidualNative`、支持三角面索引及质量状态。
- [ ] 使用已安装 Open3D 的多局部平面检测提议候选；采样使用固定种子和面积权重，保存采样配置，避免网格密度改变支持权重；按真实网格支持和连通性分开区域，再以面积加权拟合。检测盒仅用于候选，不能作为实际面积或轮廓。
- [ ] 对双层薄板复用已经验证的共同法向拟合逻辑；只在实际重叠、平行且厚度受支持时合并，面积只计一个板面，保存两层来源。
- [ ] 用实际三角面投影并集形成面轮廓和面积，保留孔洞、凹形；面 ID 绑定当前几何/算法指纹，不承诺修改网格后 ID 不变。
- [ ] 复用现有倾角公式，统一坐标下对每个面计算：

```python
cosine = np.clip(abs(np.dot(unit_surface_normal, unit_ground_normal)), 0., 1.)
inclination_deg = float(np.degrees(np.arccos(cosine)))
deviation_from_vertical_deg = 90. - inclination_deg
```

- [ ] 记录平面法向拟合波动和地面拟合波动，作为工程角度误差界；不得把普通拟合残差直接当成角度或称为已标定置信区间。偏离竖直超过双方误差界总和才进入默认非竖直列表；误差无法估计时保留角度和“方向待确认”。
- [ ] Run 两个 assert 脚本，要求所有解析角度误差在 0.1° 内、同一几何重剖分不增面、反绕序/刚体变换不改角度。真实重建模型不使用这个解析精度作为现场准确度声明。
- [ ] 检索所有现有 `fitted_plane` / `fitted_bend` 调用，确保未把自身折弯和两物体夹角改成地面倾角；提交几何实现。

## 任务 3：在 process 中计算并保存

**Files:** `ehs_spatial/platform/scene_measurements.py`, `panoptes_worker/__main__.py`, `ehs_spatial/platform/publication_site.py`, `modal_apps/publication_site.py`; test `tests/check_planar_surfaces.py`.

- [ ] 增加 `analyze_inclinations(revision, load_asset, *, persist=False, cache=None)` 与 `saved_inclinations(revision)`，复用现有分析阶段调用方式，不新增任务队列。
- [ ] 指纹包含完整网格资产/hash、pose、frame、ground、检测配置和算法版本；模型/地面/参数变化使旧结果失效。
- [ ] 派生结果存 `entity.inclinationAnalysis`，不写入带身份/观测约束的 `entity.measurements`。结构：

```json
{
  "entityId": "object-id",
  "inputSha256": "sha256",
  "status": "measured",
  "surfaces": [{
    "surfaceId": "input-bound-surface-id",
    "inclinationDeg": 30.0,
    "deviationFromVerticalDeg": 60.0,
    "areaNative2": 0.5,
    "classification": "non_vertical",
    "result": {"kind": "inclination", "unit": "deg", "source": "model_inference", "value": 30.0}
  }]
}
```

`result` 在实现中使用现有测量结果完整结构（references/lines/labelPoint/quality/revisionId），不是再造标注协议。
其他 status：`unsupported`（没有稳定局部面）、`skipped`（无有效模型/地面）、`failed`（计算错误/资源上限）、`not_processed`（缺少匹配指纹的结果）。reason 保存具体阶段与条件；部分面失败保留已成功面并记录未完成，不报告“全部完成”。

- [ ] 共享 worker 场景产出后执行；公共发布准备重算冻结 revision 的独立派生文件，保留原始报告不可变。
- [ ] 公共测量服务镜像加入仓库已用的 Open3D 固定版本及其运行依赖；核对镜像可加载检测模块，不能只在本机安装环境验证。
- [ ] 增加 `GET /api/revisions/{revision_id}/inclination-analysis-v1`，只读小结果，不解析完整网格、不触发计算。
- [ ] 测试：首次计算、相同输入零资产读取、地面/姿态/配置修改失效、错误状态、原始 revision 不变、GET 不计算、worker 产物确实附带结果。
- [ ] Run `.venv/bin/python tests/check_planar_surfaces.py` 与现有 publication/report-loading 检查；提交批处理与发布集成。

## 任务 4：报告中选择局部面、显示倾角

**Files:** `web/src/ReportScene.tsx`, `web/src/SpatialMeasurements.tsx`, existing measurement styles only if required.

- [ ] 报告首次读取小型倾角结果；沿用 existing request cancellation，切换 revision 时清除旧结果。
- [ ] 在已保存分析位置提供“倾斜平面”入口，列表使用“对象名 · 面编号 · 地面倾角”，同对象多面逐条可选。
- [ ] 默认只看已确认偏离竖直的局部面；提供“全部已测平面”和现有标注开关。水平物体表面显示 0°，地面参考本身不列入。
- [ ] 选一条：选中对应对象、原图/CAD 联动、02/04 高亮该局部面的真实轮廓并显示地面倾角；不显示整个对象的包围矩形。
- [ ] 详情列出面积单位、倾角/偏离竖直、参考地面和误差依据。没有结果时说明具体原因，不再建议用户盲目换另一个对象。
- [ ] `npm --prefix web run check` 与 `npm --prefix web run build`；浏览器验证一物多面、切换对象、刷新自动读取、隐藏/显示、Free 3D 拖动后标注跟随，提交前端。

## 任务 5：全量重算、发布与真实新图验收

**Files:** `docs/algorithms/README.md`; existing prepared-publication output and deployment paths.

- [ ] 运行 `.venv/bin/python scripts/prepare_publication_site.py --catalog .platform/publication-catalog --output .platform/publication-http`，记录每个 revision/object/surface 结果与未完成原因。
- [ ] 检查当前 26 对象没有被新筛选删除；折弯仍独立显示；倾角候选数量按面计数，不拿面数冒充对象覆盖率。
- [ ] 对用户新图定位真实源图与模型输入；走常规图像处理得到局部面后，核对圈出的左右斜板各自的照片区域、对象 ID、三维支持与倾角。若只有截图/缺几何，记录缺口，不能声明该图已验证。
- [ ] 以单独的已知角度输入验证新图片路径，不依赖当前场景的对象名称/ID。测试模型几何与图片支持，生成模型显示有面不等于原图已有测量证据。
- [ ] 发布公共 API 与前端；记录 commit、部署结果、公开 route、浏览器截图。用用户同一公开入口刷新，验证新增局部面结果，不以 localhost 验证替代。
- [ ] 单独核对生产 worker 已审核镜像和 secret 配置并发布同版代码；缺少部署配置必须在算法目录保留“源代码已实现、独立 worker 未部署”，不可将公共报告重算作为新图生产链路验收。
- [ ] 更新 A10/A11 与本计划复选框，提交验收记录。

## 完成条件

- 候选来自连续局部平面，支持一个对象多个面，面积依据明确且不随网格细分重复增长。
- 工位外对象不参与、地面是有证据的参考；同坐标计算，缺地面不偷换世界 Z。
- 所有可靠面的角度在 process 保存；报告只筛选/展示；切换和刷新不会触发重建。
- 真实新图圈定的左右板完成同实体照片/三维对应，或明确列出尚缺输入，不能伪称已完成。
- 算法目录、代码版本、批量结果、生产运行版本和公开页面证据一致。
