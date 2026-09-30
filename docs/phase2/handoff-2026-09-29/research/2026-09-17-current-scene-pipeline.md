# 当前照片／空间链路审计与最小复用点

2026-09-17，只读研究；主代码 `/Users/adam/Desktop/Tesla/panoptes-platform`，分支 `codex/panoptes-platform`。未执行模型调用，未修改产品。本页核实当前源码及本地真实产物；Modal 实际部署版本需由主研究单独核实，不能由源码声明推断。

**结论：现在不是只支持单照片。报告上传和几何链路支持一组 1–4 张照片，一次联合推理得到共享坐标的相机与点图；但旧报告的可选物体仍是逐照片 mask 候选，观测表面没有跨图融合。因此已有局部多视角几何能力，尚不能声称是移动摄像头的完整空间重建。**

## 实际产品入口的架构

```text
POST /api/reports（1–4张图）
  → DurableUploads → Modal process_upload
  → run_upload → EHSAssessmentPipeline.run_assessment
      → 原图归档、旋转归一化
      → MapAnything：全部图一次 infer → 每图 K/C2W/pts3d/conf/valid + 点云
      → SAM逐图分割 → MoGe尺度锚/相机高度 → 地面、实体、规则
  → _run_deep_report_chain
      → 全图检测 → inventory → object-evidence
      → observed scene（逐帧相邻像素三角化）→ viewer / CAD / report
单独点“生成物体” → candidate_id → 单图候选RecGen → 组装、比对、上下文、打包
```

入口与任务关系见 [report_workspace.py:195](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/report_workspace.py:195)、[Modal调度:112](/Users/adam/Desktop/Tesla/panoptes-platform/modal_apps/report_workspace_app.py:112)、[process_upload:66](/Users/adam/Desktop/Tesla/panoptes-platform/modal_apps/report_workspace_app.py:66)、[run_upload:49](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/report_workspace.py:49)、[deep report:232](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/app.py:232)。该部署入口导入的是 `ehs_spatial.serve`，并未调用新 `panoptes_worker`。

## 已有／缺失／可复用

| 层 | 当前证据与边界 | 移动采集复用点 |
|---|---|---|
| 输入 | HTTP、CaptureRun、MapAnything均限制1–4张。模型层一次 `infer(views)`，不是逐图独立推理。 | 短视频抽帧可接入同一CaptureRun；仅改HTTP上限不够，Gemini review也限制4张。 |
| 来源、相机、深度 | 顺序归档为image_01…并映射frame_0001…；保存源图hash、原图到canonical仿射、K、C2W、世界pts3d、conf、valid。depth可由inv(C2W)×世界点求camera-z。 | 视频须补存原视频hash、原始frame index与时间戳对应表；后续可保持相同GeometryFrame契约。 |
| 共享空间 | 同一次联合预测的pts3d直接拼为全局点云，各帧相机保留；地面使用全图点云及相机中心拟合。 | 现有几何结果可喂给后续融合；直接拼点云尚不是表面融合或闭环优化。 |
| 单位与ground | SceneMap有floor_plane、scale_source/factor/confidence。自动MoGe锚、相机高度、model_native分开；model_native可能误差很大。观测GLB仍声明native units，未宣称实测米。 | 移动采集须统一全空间尺度与地面，不能让每个窗口单独“米制”后直接拼接。 |
| 对象身份 | 旧candidate ID基于frame+源图hash+mask；`same_frame_exact_mask`、`cross_view_verified=False`。 | 必须把跨视角观测关联到一个实体，才能避免每拍到一次就多出一个“物体”。 |
| 观测网格 | 每张图独立三角化相邻有效点，保留洞；未见处不补。mask可重叠，明确无唯一triangle ownership、无跨帧融合。 | Open3D已安装，但TSDF/融合未接入旧产品链。完整空间须有统一表面及来源／实体归属。 |
| RecGen | 旧生成按钮一次一个candidate；exporter明确`views:[view]`。底层payload能装多views，新platform也支持同一entity多个owned observations。 | 不应按每视频帧重建一次物体；先统一空间与实体，再按实体生成或建模。 |
| CAD／报告 | 旧CAD是inventory平面图及`inv`索引热区，报告／CAD／3D共享索引；观测场景另保留candidate provenance。 | 延用选择关联，但索引相同不能替代跨图物理实体确证，CAD视图也不等于参数化CAD实体。 |
| 视频 | OpenCV抽帧→SAM→ByteTrack→MoGe关键帧选出一个FloorCamera→全轨迹抬到同一地面。代码明确摄像头固定。 | 只复用解码／抽帧。FloorCamera、人体/车辆轨迹规则不能作为移动相机pose重建。 |

关键源码：

- 上限和一次请求：[CaptureRun:85](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/contracts.py:85)、[MapAnythingAdapter:313](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/providers/map_anything.py:313)、[实际模型infer:139](/Users/adam/Desktop/Tesla/panoptes-platform/modal_apps/mapanything_app.py:139)、[Gemini:248](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/providers/gemini.py:248)。
- 相机与点图持久化：[parse_frame_json:161](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/providers/map_anything.py:161)；源图hash、映射、深度：[object_evidence:86](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/object_evidence.py:86)；候选身份：[193](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/object_evidence.py:193)。
- 地面与尺度：[geometry:177](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/geometry.py:177)、[scale chain:48](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/pipeline.py:48)；观测本地单位：[assemble:590](/Users/adam/Desktop/panoptes-public/panoptes-serving/scripts/research/assemble_lucida_scene.py:590)。
- 逐帧网格：[observed_mesh:222](/Users/adam/Desktop/panoptes-public/panoptes-serving/scripts/research/assemble_lucida_scene.py:222)；明确无融合、无ownership：[416](/Users/adam/Desktop/panoptes-public/panoptes-serving/scripts/research/assemble_lucida_scene.py:416)、[522](/Users/adam/Desktop/panoptes-public/panoptes-serving/scripts/research/assemble_lucida_scene.py:522)。
- 单候选RecGen：[generate_candidate:95](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/object_generation.py:95)、[exporter:89](/Users/adam/Desktop/panoptes-public/panoptes-serving/scripts/research/export_object_evidence.py:89)、[多views payload:85](/Users/adam/Desktop/panoptes-public/panoptes-serving/scripts/research/generate_lucida_assets.py:85)；旧CAD关联：[interactive_report:218](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/interactive_report.py:218)。
- 视频抽帧：[video:113](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/video.py:113)；固定camera假设：[553](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/video.py:553)、[全轨迹同camera:1082](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/video.py:1082)。本地`.venv`元数据实际确认Open3D 0.19.0、OpenCV headless 5.0.0.93、trimesh 5.1.0；Modal旧报告镜像也显式安装OpenCV与trimesh（[部署:38](/Users/adam/Desktop/Tesla/panoptes-platform/modal_apps/report_workspace_app.py:38)）。

## 新platform代码可借用，但不能算旧产品已上线

新入口是 [platform_worker:20](/Users/adam/Desktop/Tesla/panoptes-platform/modal_apps/platform_worker.py:20) → [runtime.services:13](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/platform/runtime.py:13)（Postgres+blob）→ `panoptes_worker.run_job` → [run_capture_pipeline:2121](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/platform/reconstruction.py:2121)。部署需要已审计镜像digest和DB/storage secret；本审计不把存在代码等同于部署完成。

它已有 [append_geometry:1059](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/platform/reconstruction.py:1059)：选择一个既有背景参考帧，每组参考帧+最多3新图联合推理，以[register_reference:156](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/platform/spatial.py:156)求变换，把新帧移到旧坐标并保存配准证据。还已有几何支持的跨图关联及RecGen同实体多观测输入（[recgen:157](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/platform/recgen.py:157)）。但这不是完整移动SLAM：引用固定旧anchor，不能保证走出anchor视野后仍有效，且[reconstruction:947](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/platform/reconstruction.py:947)仍明确不跨图融合context。不要为视频入口另建数据库/队列；先决定沿哪个实际运行入口复用这些几何函数。

## A点走到B点再回来的空间记忆

本次对 `ehs_spatial/`、`panoptes_worker/`、`modal_apps/` 的Python/SQL/JSON源码检索，未找到place retrieval、relocalization、loop closure、pose graph optimization或可重新加载继续定位的SLAM地图实现。已有文件产物、图像ID、相机、坐标系、`scene_revisions` 和配准证据持久化；这是场景版本与局部配准历史，尚不是机器人“回到A点能识别并纠正一路漂移”的地图记忆。

特别注意：[register_reference:156](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/platform/spatial.py:156)要求 **source与target为同一个image_id、同一个image hash**，用同张旧图在两次几何求解中的对应像素估计Sim(3)；不是从新的A点照片识别曾经来过的A点。它保存held-out点误差、重投影误差及参考相机残差，但不做地点检索、跨时间闭环或全轨迹优化。现有 [scene_revisions插入:158](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/platform/postgres.py:158)可复用来保存优化后的新版本。

最短合同延伸建议（待实现，不是现状）：沿现有capture/asset/revision加每帧“源视频asset/hash、frame index、timestamp”；让轨迹／地图后端输出带稳定`coordinateFrameId`和尺度状态的每帧K/C2W及一个可重载地图artifact；将“相邻帧配准、重定位或闭环”的约束类型、关联关键帧ID、验证误差保存在现有配准证据体系。观测、对象、CAD继续引用这些frame/coordinate IDs。闭环更新相机后，应生成新几何版本并重算其衍生surface/测量/重投影，而非只移动显示相机。先复用现有asset与revision存储，不另造地图数据库；地点索引、闭环和优化由选定地图引擎提供。

画面切换并不能直接认定换了物理空间：应以相机连续跟踪及与已存地点的几何验证来决定是否延续地图。没有空间关联证据的片段应保留独立坐标关系待建立，不能因为场景语义相似就拼在一起。

## 和可编辑Blender部件建模结合的最小接口

可把现有scene的K/C2W、对象owned masks、观测点图、统一坐标与尺度状态作为建模约束，把外部建模结果接到现有“模型候选→源图检验”接口。可编辑性来自部件/参数，正确性来自原始证据检查，两者分别记录。

- [coarse_model:80](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/platform/coarse_model.py:80)已经用两个已注册且属于同一物体的mask估计细长框架轴、长度/宽度并生成box部件；bar thickness与rung count明确是给定假设。可复用其“证据约束参数”的模式，但它不是任意工位自动建模器。
- 新结果只需提供 `vertices/faces/colors + proposedObjectToNative + provenance`，进入已有[_assess_generation:1683](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/platform/reconstruction.py:1683)。[model_quality.assess_model:228](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/platform/model_quality.py:228)用原始K/C2W、mask、camera-z逐图射线比对轮廓/遮挡/深度，并绑定hash；[refine_model_pose:372](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/platform/model_quality.py:372)已有有界位姿/尺度修正。它证明与已观测面相符，不能证明背面补全或毫米精度。
- [blender_export.export_scene_revision:663](/Users/adam/Desktop/Tesla/panoptes-platform/ehs_spatial/platform/blender_export.py:663)已有后台Blender执行、`.blend`/GLB输出、重开与相机投影验证；[worker:38](/Users/adam/Desktop/Tesla/panoptes-platform/panoptes_worker/__main__.py:38)已有export job。复用该执行与导出通道即可，不需额外造Blender基础设施；现有script是确定性导出，接收任意生成代码执行仍需另行设计具体接口。

## 推荐的最小研究验证顺序

1. 同一静态工位短视频，复用OpenCV解码选3–4张有重叠且有位移的清晰帧，送现用照片pipeline；保留原视频帧索引与时间戳。先与现有手拍3图比较共享坐标、遮挡、物体对应关系。这只能验证视频作为局部输入，不能据此宣称整空间成功。
2. 若目标确实是走遍整个空间，关键改造点在**geometry输出到observed scene之间**：连续帧的统一位姿/尺度、跨窗口配准/漂移验证、观测表面融合、源像素到统一表面及实体的归属。可借用现有append/register_reference和Open3D；长轨迹、闭环与融合算法选择交给模型研究对比，不能直接无限循环4图并声称完成。
3. 在统一scene上按实体做可编辑部件建模，将Blender/程序化结果经已有模型质量接口投回全部源图；之后更新同一实体的CAD/3D/report。这样每个模型只对应一个有证据的物理对象，生成速度不会按视频总帧数乘上去。

## 耗时证据（历史单物体，不能外推整空间）

真实产物 `lucida-replica-01/generation/robot/output.json`：模型infer **10.55秒**（[54](/Users/adam/Desktop/panoptes-public/panoptes-serving/outputs/candidate-evaluation/lucida-replica-01/generation/robot/output.json:54)），GPU函数 **52.73秒**（[166](/Users/adam/Desktop/panoptes-public/panoptes-serving/outputs/candidate-evaluation/lucida-replica-01/generation/robot/output.json:166)），本地远程调用往返 **106.92秒**（[350](/Users/adam/Desktop/panoptes-public/panoptes-serving/outputs/candidate-evaluation/lucida-replica-01/generation/robot/output.json:350)）。cart相应 **11.15 / 54.47 / 82.69秒**。这些是两个对象的历史运行，未包括照片全链路。`630秒/物体`是保守GPU预算预留，不是推理实测。本次未取得移动整空间的端到端运行计时，因此没有给出该时长预测。
