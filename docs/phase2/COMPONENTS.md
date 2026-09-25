# 组件清单：职责、文件、交接契约、缺口

日期：2026-09-23。架构和缺口的来龙去脉见 [ARCHITECTURE.md](ARCHITECTURE.md)（缺口编号 A0–A20 指向那里；PLAN 的 G1–G7 是里程碑，不是缺口）。本文件回答：**每个组件由哪些文件实现、读什么写什么、还缺什么、能不能单独分给一个 session 做。**

状态标记：**在用** = P3 视频→报告主线；**照片** = 照片报告在用、视频不用；**库** = 主入口已废，但在用代码导入它的函数；**已取代**；**放弃**；**评估** = 评估脚本。

## 交接契约（组件之间只通过这些文件对话）

并行开发的前提：**不改这些文件的已有字段**；要加字段就加，不要改名、不要改语义。

| 文件 | 写入方 | 读取方 | 关键字段 |
|---|---|---|---|
| `clip.json` + `rgb/*.png` + `rgb.txt` | C1 `prepare_video_clip`（新） | C2 `droid_room`（自动登记） | `K`、`D`（全零）、`raster: tum`（640×480）、`calibrated: false`、`intrinsics.source`（`stated` / `moge3_median_fov`）、`source.{video_sha256, fps, first_frame, crop_xywh}` |
| `run.json` + `prediction.npz` | C1 `prepare_arkit_clip`、C2 `droid_room` | C2 `mono_room` | `poses_c2w`、`keyframe_*`、`scale: uncalibrated_monocular` |
| `metric-scale.json` | C2 `mono_room metric` | C2、C4、C7、R | `metres_per_native_unit`、`scale_status`（`device_metric` / `assumed_camera_height` / `assumed_camera_height_floor_views_disagree` / `uncalibrated`）、`camera_height_native_median`、`camera_height_native_p10_p90`、`scale_sources_disagree_over_10pct` |
| `scene.json`（phase2-replay-scene-v1） | C2 `mono_room fuse` | C4、R | `coordinate_frame`、`units`、`frames[].c2w`、`points`、`limitations` |
| `motion.json` + `NNNNN-residual.npz` | C3 `motion_masks` | C3 `sam3_motion_tracks` | `frames[].blobs`、`trusted_moving` |
| `tracks.npz` + `tracks.json` | C3 `sam3_motion_tracks`（每窗口一份） | C3 下游三个脚本 | `{motion\|text}/{frame}/{id}`，**`frame` 是跟踪器下标，不是源帧号**：源帧号 = `frames[0] + frame × stride`（A15 修好前只有 stride=1 的运行可以直接用）；`objects[stage][].kept`（motion 的 kept 表示“动过”，文字轨迹不看它）；`frames:[first,end)`、`stride` |
| `stitched.json` | C3 `stitch_track_windows`（新） | **暂无** | `windows[].folded_into_text`、`joins`、`refused`、`identities`（`w{n}:{stage}/{id}` → 身份） |
| `analysis.json`（移动层） | C3 `motion_tracks_to_analysis` | C2 `mono_room dynamic`、R、C6 | `entities`、`frames[].objects[{entityId,maskUrl,label,source}]` |
| `object-map.json` | C4 `build_video_object_map` → `name_video_entities` | C7、R、C10 | `entityId`、`observations`、`centroidNative`、`footprintPlanNative`、`heightNative`、`baseNative`、`label`、`labelSource` |
| `events.json` | C6 `video_events` | R | `windows[].events[{t0,t1,actor,action,near,ppe,safety_note}]`、`model`、`temperature` |
| `policy-findings.json` / `scene-document.json` | C7 `evaluate_video_policy` | R | `findings[].machineResult`、`missingEvidence`、`applicability` |
| 平台场景文档（Postgres `scene_revisions.document`） | R `import_video_scene` | 平台全部 | `coordinateFrames[].scale{status,nativeToMeters,anchor}`（由 `evaluate_video_policy.contract_scale` 按 `scale_status` 生成，两处共用）、`entities`、`annotations` |

## C1 采集与片段

**职责：** 把现场拍摄或外部视频变成管线能吃的片段：帧、时间戳、（可选）设备位姿、选段。

| 文件 | 状态 | 说明 |
|---|---|---|
| `scripts/prepare_arkit_clip.py` | 在用 | ARKitScenes → 带设备米制位姿的片段 |
| `scripts/prepare_video_clip.py` | 新，在用 | 任意 MP4 → 640×480 片段：中心裁 4:3、缩放；K 来自 `--fov-deg` 或 MoGe-3 五帧中位数；写 `clip.json`（标 `calibrated: false`）。Lightning 3585–3611 s 是第一个用户 |
| `scripts/scout_clip_segments.py` | 在用（无下游） | 给长视频打分选段；**选出的窗口没人切成片段** |
| `modal_apps/droid_room.py` 的 `CLIPS` | 在用 | TUM 片段写死在 `:26-28`；另外自动登记 `data/clips/*/clip.json` |

**缺口：** A1 第一版已有。剩下：另一个会话的 `motion_masks.fixed_video_frames` 补边到 640×368，与这里的裁 4:3 到 640×480 冲突，要定一套；畸变假设为零；选段结果（`scout_clip_segments`）还没接到转换器。
**可单独分出去：** 是。

## C2 几何与尺度

**职责：** 相机、深度、融合网格、地面平面、尺度（带来源与不确定度）。

| 文件 | 状态 | 说明 |
|---|---|---|
| `modal_apps/droid_room.py` | 在用（只取相机） | DROID-SLAM，640×480 |
| `modal_apps/mono_room.py` | 在用 | `infer`（DA3/MoGe-3）、`metric`、`fuse`（TSDF + 雕刻）、`dynamic`、`evaluate`；**用模块全局变量的库**，10+ 个脚本导入它 |
| `scripts/texture_fused_mesh.py` | 在用 | 照片贴图 |
| `scripts/infer_room_floor.py` | 在用（可选） | 推断地面模型；`:61` 写死的是“1.6 m”这段出处文字，数值本身不写死 |
| `modal_apps/moge3_app.py` | 照片 + 在用 | 照片尺度锚；视频里作尺度交叉检查和移动像素深度 |
| `modal_apps/mapanything_app.py` | 照片 | 视频上已否决 |
| `modal_apps/lingbot_room.py`、`scripts/phase2_camera_*.py` | 放弃 | LingBot 9/19；ORB-SLAM3 |
| `ehs_spatial/geometry.py`、`ehs_spatial/measurements.py` | 照片 | 地面拟合、持机高度尺度（置信度写死 0.9） |
| `ehs_spatial/platform/spatial.py`、`planar_surfaces.py`、`scene_measurements.py` | 平台 | 原生单位的测量，不给结论 |

**缺口：** A0 一个共用的“尺度能否判规则”函数（线上 `policy.py:303`、`scene.py:134-137`、`measurements.py:44` 与 P3 的 `contract_scale` 各说各的）；尺度不确定度（p10–p90 只记了备注，没换成误差带）；三套尺度词表（A18）；A3 按人身高估尺度（优先级最低，现行闸门下解锁不了结论）；坐标系 ID 对 ARKit 也写死成 `droid_final_native_world`（`import_video_scene.py:26`）。
**可单独分出去：** 共用尺度闸门可以（小，且修的是线上）。重建本身不建议拆。

## C3 动静分离、跟踪、身份

**职责：** 找出动的东西、全程跟踪、给稳定 ID、把动的像素从静态融合里剔除。

| 文件 | 状态 | 说明 |
|---|---|---|
| `modal_apps/motion_masks.py` | 在用（有未提交改动，属另一 session） | 对极残差 + RAFT，找相机解释不了的运动 |
| `modal_apps/sam3_motion_tracks.py` | 在用 | 运动种子驱动 SAM 3.1，每窗口一次；轨迹级 `kept` 检查；`--stride`（A15 的种子夹紧、getattr、融合掩码沿用已修）。**1 号对象只剩种子帧的 bug**：点击会话先放一个一次性对象吸收它（原因未查清，关确认的假设已被 run 139 否定，见 ARCHITECTURE §7）。另一个会话改了 `kept`：moving share 为 None 时丢弃 |
| `modal_apps/sam3_video.py` | 库 | 共享的会话启动与镜像；`run_native` 无下游。省显存开关在 `predictor.model.tracker` 上；打开会关掉记忆选择，所以只记录、默认不开 |
| `scripts/assemble_dynamic_masks.py` | 在用 | 动态掩码 → 融合 |
| `scripts/motion_tracks_to_analysis.py` | 在用 | 轨迹 → 移动层；窗口重叠处按“离窗口中心近”切开（`owner_of`） |
| `scripts/stitch_track_windows.py` | 新，未接入 | 重叠帧集合匹配。按评审修过：窗口内同一个人的重复轨迹先合并（motion 与 text 各跟一次；text 会话中途给同一人换 id，所以合并要反复做到不再变化）；两边都空的帧不算；首帧、stride 读 `tracks.json`；空窗口不崩。Lightning 三窗口（修复前的轨迹）：6 次接上（IoU 0.93–0.998），1 次因只重叠 4 帧拒绝 |
| `scripts/motion_facts.py` | 在用 | 路径、速度、停留、经过的物体（只出事实，不出结论） |
| `scripts/run_discovered_video.py`、`run_seeded_video.py` | P2 | 带一帧重叠的窗口链 ID（`match_discovery`） |
| `scripts/link_person_tracklets.py`、`person_reid.py` | P2（只提议） | OSNet 外观 + 时间，提议重入 |
| `scripts/video_motion.py` | P2 | 唯一按轨迹空洞判“证据不足”的代码 |
| `ehs_spatial/video.py` | P1 | ByteTrack + R1/R2/R3 |

**缺口：** A12 缝合还没接进流水线；A13 三套窗口边界处理，评审建议身份用 `stitch_track_windows`、`owner_of` 只管画哪个窗口的掩码、`match_discovery` 随 P2 退役；A14 站着不动的人没从静态融合里剔除；A15 stride 下游读错帧；轨迹空洞（被挡住 vs 走出画面）在 P3 里没有“证据不足”状态（`mono_room.py:617` 写死 `insufficient_evidence`）。
**可单独分出去：** 是。

## C4 物体地图与命名

**职责：** 分割 → 抬进三维 → 多视角合并成实体 → 起名。

| 文件 | 状态 | 说明 |
|---|---|---|
| `modal_apps/sam2_everything.py` | 在用 | 全自动分割，掩码互斥 |
| `modal_apps/sam3_app.py`、`scripts/discover_video_keyframes.py` | 在用 | SAM3 按词分割 |
| `scripts/discover_video_vocabulary.py` | 在用（可选） | 让 VLM 列词表；也用云端 Gemini（`:18`） |
| `scripts/build_video_object_map.py` | 在用 | ConceptGraphs 式点重叠累积；≥3 视图才确认；`--method clique` 未用 |
| `scripts/name_video_entities.py` | 在用 | VLM 起名；**用云端 Gemini，违反 on-prem（A19）** |
| `scripts/project_entities_to_frames.py` | 在用 | 实体投回每帧（可点击） |
| `scripts/build_video_entity_model.py` | 在用（可选） | RecGen 生成模型，研究许可，HANDOFF 建议停用 |
| `ehs_spatial/platform/identity.py` | 平台 | 场景内与跨版本的同一/不同决策；**不做跨次匹配** |
| `scripts/scene_inventory.py`（3045 行） | 照片 | 只看第一帧列物体、墙面、工位矩形、DXF 输出 |

**缺口：** 对象地图没有人工清单对照（A20）；命名换自部署 VLM（A19）；物体关系（A8）；少于 3 视角的候选在导入文档里是 `association_pending`，引擎却照样要它们的占地（A10）。
**可单独分出去：** 命名替换可以；关系边（支撑、附着、包含）可以，输入就是 `object-map.json`。

## C5 区域

**职责：** 通道、出口路线、禁入区等多边形（带高度范围、作者、版本），供规则使用。

**现状：没有。** CAD 视图的画框只是临时测量；策略模板引用的 `confirmed exit route` 等标签没有生产者；没有 DXF/DWG/IFC 导入。
**缺口：** A7。
**可单独分出去：** 是。最小版本：在报告的 CAD 视图里画多边形，存成带 `labelSource: operator` 的实体，走现有 `addEntity`。

## C6 视频记忆

**职责：** 按窗口让开源视频模型写事件，挂在实体 ID 和视频时间上。只出证据。

| 文件 | 状态 | 说明 |
|---|---|---|
| `modal_apps/video_events.py` | 在用 | Qwen3-VL-8B 默认；帧上画出被跟踪对象并标名；12 s 窗口 2 fps |

**缺口：** 事件只进报告注记，没有事件表（A17）；T>0 采样没有记录随机种子，事件不可复现；名册核对（期望 actor 与实际 actor 的差集）做过一次离线计算，没有进代码。
**可单独分出去：** 是。

## C7 规则

**职责：** 条文 → 谓词 → 四态结论 + 证据。

| 文件 | 状态 | 说明 |
|---|---|---|
| `ehs_spatial/policy.py` | 核心，**线上工作台在用** | 7 个谓词（枚举在 `ehs_spatial/contracts.py:29-40`）：`min_separation`、`keep_clear`、`max_separation`、`not_inside`、`max_height`、`min_height`、`max_tilt` |
| `ehs_spatial/platform/policy_engine.py`、`policy_repository.py`、`policy_service.py` | 平台 | ZEN 决策图 + 适用性 + 复核；6 张 `policy_*` 表 |
| `ehs_spatial/rules.py` | 照片（线上 `/reports`） | 写死的围栏净空；共享误差带 0.20 / 0.35 m |
| `scripts/policy_compile.py` | 工具 | 条文 → `PolicySpec`（封闭词表，必须能拒绝）；**用云端 Gemini**（`:28`） |
| `scripts/oshacorpus.py` | 评估 | OSHA 1910 考试集 |
| `scripts/evaluate_video_policy.py` | 演示 | 视频对象地图 → 米制事实 → 平台引擎；`contract_scale` 在这里。演示把 1910.36(g)(2) 编成桌↔柜 `min_separation`，而考试集要求拒绝这条（A5） |
| `docs/policies/*` | 示例 | 非认证 |

**缺口：** A4 覆盖输入；A5 自由空间/净宽算子；A6 时间类谓词；A9 引擎已知缺陷（含 `capture_frame_count=1` 关掉两视角闸门）；A10 演示结论≠平台结论；A11 例外结构与过期考题；A16 ZEN 引擎不在线上。
**可单独分出去：** 是，最适合拆：每个新谓词都是“签名 + 求值 + 自检”的独立单元。

## C8 对话与检索

**职责：** 自然语言提问 → 工具调用 → 带证据的回答。

| 文件 | 状态 | 说明 |
|---|---|---|
| `ehs_spatial/platform/agent_service.py` | 平台 | 云端 Gemini 工具循环（`runtime.py:37-51`，A19），只能提议；10 个工具（`get_entity`、`list_entities`、`test_policy`、`draft_policy`、`propose_operations`、`start_job` 等） |
| `ehs_spatial/agent.py`、`agent_hub.py` | 照片（线上） | 直接改运行目录里的文件 |

**缺口：** A17 没有检索（无时间区间、无空间邻近、无向量）；上下文靠把整份实体塞进提示词；两套 agent 并存（A18）。
**可单独分出去：** 是，但依赖事件表和轨迹表先落库。

## C9 骨架与姿态

**职责：** 从位置升级到动作。

| 文件 | 状态 | 说明 |
|---|---|---|
| `scripts/build_video_pose_preview.py` | P2 + 库 | RTMPose COCO17；其 `sha / intervals / source_spans` 被广泛导入 |
| `scripts/build_video_body_models.py` | P2 | SAM 3D Body，按传感器深度定尺度 |
| `mono_room.py dynamic` | 在用 | 关节抬到可见表面 |

**缺口：** 关节没有按时间串成序列，没有姿态量与姿态事件。注意：没有用 SMPL，许可干净。

## C10 仿真导出

**现状：有 GLB 导出，没有仿真导出。** `ehs_spatial/platform/blender_export.py` 出 GLB 和 .blend，调用方：worker 的导出任务（`panoptes_worker/__main__.py:38-60`）、界面（`App.tsx:1000-1001`、`WorkcellReport.tsx:656`）、agent 的 `start_job`（`agent_service.py:245`）。只有坐标系是 `operator_anchored` 时导出米制（`:402-404`），否则原生单位——**今天尺度改标后，假设高度的视频场景都导出原生单位**。OpenCV 坐标系、无上方向对齐、无碰撞体、无质量、无 USD。
**缺口：** A20。最小版本：Z 朝上、米制的 USD；整间房作为静态碰撞体；每个实体一个网格 + 语义标签；可移动实体用 CoACD 凸分解。**前置条件是尺度为米**。
**可单独分出去：** 是，在 C2 尺度落地之后。

## R 报告与平台

| 文件 | 状态 | 说明 |
|---|---|---|
| `scripts/import_video_scene.py` | 在用（本机） | 零模型调用导入 → 发布 |
| `ehs_spatial/platform/*`（postgres、repository、storage、api、contracts） | 平台（本机） | 16 张表；场景是一个 JSONB 文档；不可变由触发器保证 |
| `ehs_spatial/report_workspace.py`、`pipeline.py`、`scene.py`、`app.py`（工作台）、`serve.py` | 照片（线上） | P0a 上传、P0b 工作台、P1 视频页都在这一个应用里 |
| `web/src/*` | 在用 | 报告页：视频、三维、动静切换、骨架、记忆 |

**缺口：** A16 线上与平台两套并存（评审建议收敛到平台：P0 上传走 `import_scene`，冻结工作台策略选择器，0.6 m 规则变成平台策略后删 `rules._assess_clearance`）；导入记录 `.platform/imports/*.json` 含项目凭证（文件权限 600），不要读、不要外传。

## E 评估

所有评估脚本都是照片时代的（`site_acceptance`、`nvwarehouse_eval`、`qspatial_eval`、`spatialrgpt_eval`、`redwood_v2_eval`、`arkitscenes_eval`、`moge_anchor_eval`、`recall_eval`、`vlm_baseline`、`ehs_eval`、`drift_check`、`arm_poc`）。视频路径的数字来自 `droid_room.evaluate`（相机对动捕 RMSE：fr1/room 4.08 cm、walking 1.74 cm）和 `mono_room.evaluate`（深度对传感器中位误差 2.59%）。

**缺口：** A20 对象地图、动静分离、身份续接都没有进代码的评估。

## 不要删的“已取代”脚本

下面这些主入口已废，但在用代码导入它们的函数，**删了会断主线**：

| 脚本 | 被导入的函数 | 导入方 |
|---|---|---|
| `reconstruct_room_rgb.py` | `digest`、`integrate`、`new_volume`、`validate_frame` | TSDF 相关 |
| `build_replay_scene.py` | `media_spans`、`world_points`、`surface_joints` | `mono_room.py:389,581`；`build_video_object_models.py:21`、`build_droid_replay.py:19` 等 |
| `video_motion.py` | `annotate_motion`（模块级导入） | `build_replay_scene.py:18` |
| `reconstruct_tum_room.py` | `read_rows` | `build_replay_scene.py:17`、`build_droid_replay.py:21` |
| `build_video_surfaces.py` | `export_surface` | `mono_room.py:580` |
| `build_lingbot_replay.py` | `read_prediction`、`resize_mask`、`source_transform` | `build_lingbot_object_model.py:18` |
| `build_video_object_models.py` | `observed_surface` | 对象地图 `:266`、`mono_room :579` |
| `filter_video_static_surfaces.py` | `depth_evidence` | `mono_room.py:338` |
| `build_droid_replay.py` | `depth_support` | `mono_room.py:180,292` |
| `build_lingbot_object_model.py` | `evaluate`（源视角闸门） | `build_video_entity_model.py:28` |
| `attach_entities_to_replay.py` | `light_model` | `import_video_scene.py:252,310` |
| `review_video_object_semantics.py` | `REMOTE` 程序串 | 命名与词表脚本 |

## 文件结构：暂不重组

现在是平铺结构（`scripts/` 88 个、`modal_apps/` 20 个）。**建议暂不按组件挪目录**，原因：

1. 另一个 session 在 `motion_masks.py`、`sam3_motion_tracks.py`、`HANDOFF.md`、`tests/check_fixed_camera_motion.py` 上有未提交改动，挪文件必然冲突；提交时也要按会话分开；
2. 上表那些“库”依赖是跨组件的，挪之前要先把共享函数抽到一个公共模块；
3. 组件边界已经由上面的交接契约定义，目录不是并行开发的前提。

等那边的改动合入后，按“先抽公共模块（`sha`、`raycaster`、地面拟合、尺度），再按组件分目录”两步做。
