# 人体骨骼、模型、世界运动：代码审计与选型

核查日期：2026-09-17。由代码审计、姿态算法、世界运动三路独立研究汇总。**本次是源码与一手资料研究，没有安装或实跑下面的人体算法，没有性能验收。** 实施优先级仍是[房间地图与运动定义](DYNAMIC-SCENE.md)，不是立即新增一套人体系统。

## 先纠正当前能力

当前没有已接入的 2D 人体骨骼、3D 关节、人体网格或人体动画。此前“已有人的视频跟踪能力，可以复用”不准确：能复用的只是历史固定机位二维框跟踪的部分代码/接口，不能将其记为当前可用能力。

| 核查范围 | 实际证据 | 结论 |
|---|---|---|
| serving `ehs_spatial/video.py:287` | SAM 的 person 提示、ByteTrack 关联框与时间；`lift_tracks` 将框底中点投到同一地面 | 是框跟踪设计，没有人体关节；地面投影假定固定相机与脚接触地面 |
| serving 当前 `.venv` | 缺 `trackers`、`supervision`；无 `mediapipe`、`mmpose`、`smplx`、`ultralytics` | 不能说旧人物跟踪在当前环境已就绪 |
| serving `tests/test_video.py` | 显式注入假 SAM、MoGe、抽帧和 `_fake_track_fn` | 测试验证逻辑，不证明真人视频跟踪效果 |
| `docs/demos/2026-08-25-poc-video.md` | 历史记录声称 MEVA 44 帧实验，并记录三人产生六条碎片轨迹、脚被截断后的米级跳跃 | 仅核实记录存在；引用原始产物现已找不到，未复核当时结果 |
| Phase2 `rgb-mapanything-032` | RGB、点图、相机、静态 TSDF；没有 mask、track、joint、body-model 产物 | 当前房间实验未接入人物链 |
| `web/src/viewer/native-viewer.ts` 与 `blender_export.py` | GLB 读取明确拒绝 skin/morph | 现有静态模型显示不等于支持人体骨骼动画 |

`skeleton` 搜索命中中的机械臂正运动学和围栏掩码形态学不是人体骨骼。旧 Gradio `analyze_video → run_video_assessment` 也不是当前房间预览入口。

## 四件事不能混为一谈

1. **检测/分割**：人在哪些像素中。可用于限制人进入静态建图的观测。
2. **身份跟踪**：不同帧是不是同一个人。一个框 ID 不包含手脚姿态。
3. **姿态/人体模型**：关节如何弯曲、身体形状是什么。以髋部为原点的 3D 不等于房间坐标。
4. **世界运动与交互**：同一人在房间哪里、与当时的设备/车辆有什么空间关系。需要地图、相机、尺度、身份和时间共同成立。

原地挥手时人体根位置可保持静止，但手臂仍在运动；人物跟随相机走动时画面中的框可以几乎不动，但世界位置在变。因此身体根运动与关节运动分开表示。车辆车体位姿、转向/车轮等部件运动也按这一原则分开。

## 现成预训练方案

| 方案 | 能提供什么 | 对当前项目的边界与成本 |
|---|---|---|
| [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) | 每人 33 点、图像坐标、以髋部为原点的预测 3D，可选分割；当前 API 可配置多人 | 适合本机 CPU 骨架基线。视频内部跟踪不是持久人员 ID，WorldLandmarks 不是房间坐标；Python GPU 文档不支持据此宣称 M2 MPS 可用 |
| [RTMPose / RTMW](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) | 人员框内的 2D 骨架；常用 RTMPose 17 点，RTMW 133 点含手足脸 | 预训练 ONNX 路径适合验证；多人通常要先检测再逐人估计。当前没有可直接复用的房间人员框产物，首测必须包含人员检测，不能只计 pose 耗时 |
| [ViTPose / ViTPose++](https://huggingface.co/docs/transformers/model_doc/vitpose) | 人员框中的 2D 关节 | 现成 Transformers 接口；需要检测与身份关联。不是 3D 人体模型，也没有房间地图 |
| [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) | 单图人体网格及 MHR 参数；可接框、mask、相机内参，覆盖身体、手、足 | 单帧身体形状不等于连续人体运动。权重需申请；[官方 estimator](https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/sam_3d_body_estimator.py) 仍硬编码将 batch 放到 CUDA，不能因 demo 有 CPU 分支就声称原版能在 M2 跑通 |
| [WHAM](https://github.com/yohanshin/WHAM) | 视频人体姿态/形状及世界运动，集成关节和相机运动信息 | 有预训练推理；相机运动来自 DPVO/DROID 等，人体依赖 SMPL。忽略 SLAM 的 local-only 模式不是房间运动；不能用默认焦距/预测人体尺度替代本项目标定 |
| [GVHMR](https://github.com/zju3dv/GVHMR) | 利用重力视角恢复世界人体运动，减少脚滑等问题 | 有预训练权重，包含姿态/图像特征/相机估计多个阶段；2025 年默认相机入口改为 SimpleVO。输出仍需与房间地图坐标、尺度对齐，不能把另一套世界原点直接叠入地图 |
| [PromptHMR](https://github.com/yufu-wang/PromptHMR) | 图像提示人体模型及多人视频世界运动，支持移动相机，导出人体与相机动画 GLB | 最接近整条研究参考链：检测、SAM2、DROID-SLAM、Metric3D、ViTPose 等共同工作。官方环境使用 CUDA；不能把完整链性能等同于人体网络单次推理 |
| [DuoMo，2026](https://github.com/facebookresearch/DuoMo) | 相机空间与世界空间两阶段扩散，生成随时间变化的人体表面 | [官方明确不含 SLAM](https://raw.githubusercontent.com/facebookresearch/DuoMo/main/README.md)，移动相机需要外部相机位姿；输出以首帧相机为原点。支持提供预训练权重推理，官方安装为 CUDA 12.8。遮挡时生成的合理动作属于推断，不能作为已观测的交互证据 |

[TRAM](https://github.com/yufu-wang/tram) 是可读性很好的架构参照：先检测/跟踪和 masked SLAM，再恢复人体，最后放到同一时间与坐标中；官方 2025 年提示其已整合入 PromptHMR。无需同时把所有候选库接入项目。

### 运行与许可边界

无需从零训练上述算法，但检测、分割、姿态、相机估计、时间处理与渲染仍消耗计算。不能给出未经本机实测的 FPS、显存下限或视频总用时。M2 首个基线可选择 CPU 的 RTMPose ONNX 或 MediaPipe；它们还需实际安装与带人物视频验证。SAM 3D Body、PromptHMR、DuoMo 的官方完整链应先按 CUDA 环境预算，不能假定 `.to("mps")` 即可。

选型还要区别代码与权重/身体资产许可：MMPose 是 Apache-2.0，查看[例外清单](https://github.com/open-mmlab/mmpose/blob/main/LICENSES.md)及所选权重；[WHAM 代码](https://github.com/yohanshin/WHAM/blob/main/LICENSE)目前为 MIT，SMPL 资产另有条款；[GVHMR](https://github.com/zju3dv/GVHMR/blob/main/LICENSE)、[PromptHMR](https://github.com/yufu-wang/PromptHMR/blob/main/LICENSE)、DuoMo 的公开许可有非商业研究限制；[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE)采用自定义 SAM License。公开可下载不能直接等同于可以无条件进入商用报告。此处记录选型条件，本次未下载受限资产或接受条款。

## 对 Panoptes 的结论与验证顺序

**先把地图和运动定义做好，同时保留未来身体模型的输入。** 静态地图不能吸收动态物体的多帧重影；但只为移除人对建图的影响，不需要先实现精细骨骼模型。语义掩码可排除人的像素，其他物体仍需要相机补偿后的几何一致性检查。

1. 固定第一轮的共同地图、回环、保存/重载、A→B→A 和第二次视频重定位验证。相机失跟不能沿用旧位姿。
2. 实际恢复并核验检测/掩码/身份入口；在地图坐标下分别记录 `static_now / moving / unknown`，用连续观测和误差支持判断。人、车、被搬动的物体都保留独立实体。
3. 身体算法首测使用一段 5 秒连续真人视频：检测加 RTMPose-m 预训练 17 点，保存每帧原图、时间、人员观测、关节和分数；人工核对遮挡、多人交叉与错连，记录完整耗时。它只验收 2D 关节，不算世界人体运动完成。
4. 地图通过后，再比较一个预训练人体模型/世界运动方案。使用同一地图的相机位姿、尺度、原始帧与关节可见性，复用统一实体 ID；若算法自己的世界原点不同，记录并验证显式坐标关系。不能让人体链和房间链各自创造尺度后看起来叠在一起。
5. 浏览器读取保存的模型和时间状态。刚体模型主要更新位姿，人体复用身体几何并更新骨架/形变；不在报告加载时重新推理，也不逐帧生成完整车体。骨骼/时间动画支持属于待实现工作。

人体根部轨迹误差、关节重投影、骨架遮挡状态、与场景的空间关系应分别验收。任何生成的遮挡期动作单独标记；近距离、碰触、拿起、操作分别需要对应证据，不能从一个人体网格自动推出全部交互语义。

本研究没有改变当前交付状态：静态房间模型完整度与地图记忆尚未验收；人体骨架、人体动画和动态场景交互尚未实现。
