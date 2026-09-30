# 移动摄像头视频 → 真实空间重建：算法核查

核查日期：2026-09-17。范围为 MASt3R-SLAM、SLAM3R、MapAnything、Depth Anything 3 / DA3-Streaming、WorldMirror 2.0。依据为作者论文、官方代码、模型卡和许可证；未安装、未执行模型、未用工厂视频做性能或精度验证。这里的“已公开可运行”指存在实现、入口和公开权重，不代表本机已经跑通。

## 结论

这五项中，**MASt3R-SLAM 最符合“移动相机定位 + 地图 + 回环”的完整系统定义；DA3-Streaming 最值得作为长视频离线重建的对照候选**。两者默认研究配置均有非商业许可证限制，所以这是本组技术研究优先级，不是生产采用结论。纯 MapAnything 和 WorldMirror 2.0 是多视图几何模型，其本身不等于长序列 SLAM；SLAM3R 则主动绕过显式相机参数，较难直接承接相机轨迹、逐帧回投和尺度审计。主研究分工另已核查 Map-Long，因此团队首次普通RGB视频对比可优先采用 **Map-Long + DA3-Streaming**，MASt3R-SLAM保留为跟踪/回环参考；这不改变本组对五项原生能力的判断。

用户所问“从A走到B后还记得A”应分成：跨帧保持相机和几何关联、返回A时识别并纠正漂移、跨次运行载入同一张地图。**有多视图注意力或把点云保存到磁盘，均不等于这三层记忆已实现。**

| 方法 | 单次行程中的地图记忆 | 回到旧位置 | 跨次运行持久地图 |
|---|---|---|---|
| MASt3R-SLAM | 有关键帧、当前相机跟踪和局部融合 | 有检索库、回环候选几何验证、图优化与失跟重定位 | 本次未证实完整的地图保存→加载→继续定位接口；轨迹/点云导出不等同于它 |
| SLAM3R | 有固定容量、reservoir采样的历史场景帧 | 检索历史参考帧，隐式重定位；不是显式位姿图回环 | 本次未证实 |
| DA3-Streaming | 先将视频帧按chunk处理，保留跨块几何和变换 | 全序列SALAD地点检索、回环Sim(3)与图优化 | 本次未证实；已核查脚本面向一次图片目录的离线批处理 |
| 纯MapAnything | 一个推理批次内跨视图联合预测 | 原生入口未见地点检索/回环层 | 未见；外部Map-Long系统另行评价 |
| WorldMirror 2.0 worldrecon | 一个推理批次内跨视图联合预测 | 本次未见地点检索、丢失恢复或回环层 | 本次未证实；worldgen的生成记忆不能算现场地图记忆 |

以上机制证据分别来自[MASt3R-SLAM方法](https://arxiv.org/html/2412.12392v2#S3)、[SLAM3R历史帧策略](https://arxiv.org/html/2412.09401v2#S3)、[DA3-Streaming运行代码](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/da3_streaming/da3_streaming.py#L522)、[MapAnything API](https://github.com/facebookresearch/map-anything)、[WorldMirror API](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/DOCUMENTATION.md)。持久地图、定位状态和空间对象记忆应由系统层另行确认；本报告没有把未找到的能力断言为永远不可能实现。

| 方法 | 真实输入 / 一致性机制 | 实际输出 | 长序列与硬件证据 | 主要边界 |
|---|---|---|---|---|
| **MASt3R-SLAM** | MP4、RGB图片目录、实时相机；显式轨迹，关键帧跟踪、局部点图融合、检索回环、Sim(3)全局优化、重定位 | 轨迹 TXT + 彩色点云 PLY；未见原生三角网格或3DGS导出 | 论文：RTX 4090，图像长边512，约15 FPS；有TUM/7-Scenes/EuRoC/ETH3D评测 | 全局优化不细化所有像素几何；非商业代码/骨干权重约束；不保证真实米制尺度。来源：[官方入口](https://github.com/rmurai0610/MASt3R-SLAM)、[论文](https://arxiv.org/html/2412.12392v2)、[导出实现](https://github.com/rmurai0610/MASt3R-SLAM/blob/e6f4e3d474fad0e11f561482012be864ba8c3f17/mast3r_slam/evaluate.py#L47)。 |
| **SLAM3R** | 视频抽帧，在线/离线入口；I2P局部点图 + L2W映射到世界坐标；历史帧 reservoir 检索实现隐式重定位 | 场景/逐帧彩色点云 PLY；系统不显式求相机参数 | 论文整体20+ FPS，单RTX 4090D；有固定容量历史缓存，未找到完整工厂长路线回环验收 | 隐式检索并不等于显式位姿图回环优化；不直接交付受审计的轨迹；非商业代码。来源：[官方项目](https://github.com/PKU-VCL-3DV/SLAM3R)、[论文](https://arxiv.org/html/2412.09401v2)、[在线导出](https://github.com/PKU-VCL-3DV/SLAM3R/blob/f531d841ab743217a4464344119a350eb0556d17/slam3r/pipeline/recon_online_pipeline.py#L75)。 |
| **MapAnything** | 多视图图像；可输入相机标定、位姿、深度；联合预测公共坐标几何和相机 | 深度/置信度/内外参/点云；GLB支持**逐帧图像网格三角化**；COLMAP；3DGS须另行训练 | 当前官方说明内存优化可达2000视图/140GB；是批量视图能力，未见原生在线跟踪/回环系统 | GLB包含各帧表面不等于融合、封闭、去重的工厂mesh；默认权重NC，另有Apache权重。来源：[当前README](https://github.com/facebookresearch/map-anything)、[GLB导出代码](https://github.com/facebookresearch/map-anything/blob/3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9/mapanything/utils/viz.py#L181)、[商业权重](https://huggingface.co/facebook/map-anything-apache)。 |
| **DA3 / DA3-Streaming** | DA3支持任意视图及位姿条件；Streaming处理视频抽帧目录，重叠chunk + Sim(3)对齐 + SALAD回环检索 + 位姿图优化 | Streaming实际输出相机外参、内参、合并点云PLY、逐帧深度/置信度；默认链路不导出3DGS/三角网格 | 官方：KITTI三条序列11,373帧，A100，22分17秒 / 8.51 FPS；室内TUM 504×378，chunk30–120显存18.7–28.3GB | 作者明确说不是SLAM系统；默认大模型NC；“<12GB”不适用于此室内分辨率。来源：[Streaming文档](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md)、[主流程](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/da3_streaming/da3_streaming.py)。 |
| **WorldMirror 2.0** | 真实视频/图片目录；多视图前馈预测，可输入相机和深度先验；本次未见长序列回环或增量地图维护 | 深度、法线、相机、点云PLY、3DGS PLY、可选COLMAP；**worldrecon导出链路未见mesh** | 视频默认32帧，底层视频抽帧最多64；图片目录可更多。论文H20、518×378、单卡BF16：32视图15.10GB/2.11秒，128视图41.73GB/16.96秒 | 大批次视图、多GPU并行不能等同于无限视频流；Tencent自定义许可。来源：[API文档](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/DOCUMENTATION.md)、[视频输入/导出代码](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/hyworld2/worldrecon/hyworldmirror/utils/inference_utils.py)、[论文表14](https://arxiv.org/html/2604.14268v1#S8.SS2.SSS2)。 |

## 两个研究候选的取舍

1. **MASt3R-SLAM：优先验证闭合路线和几何一致性。** 它真正维护相机状态和回环，最适合检验“绕机器走一圈后是否回到同一面墙、同一台设备”。作者已说明全局优化并未优化全部几何、较大镜头畸变会降低预测质量，因此不能把论文的轨迹精度转写成工厂构件的尺寸精度。只适合作为许可允许范围内的研究对照，当前代码/权重不能直接当商业产品依赖。[论文局限](https://arxiv.org/html/2412.12392v2#S5)、[代码许可](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/LICENSE.md)、[骨干权重条款](https://github.com/naver/mast3r/blob/main/README.md#checkpoints)。
2. **DA3-Streaming：优先验证数千帧离线视频的资源与累积误差。** 可检查的处理链是：读取整个图片目录 → 分chunk预测 → 相邻chunk对齐 → 运行全序列回环检索 → 估计回环Sim(3) → 优化 → 保存全局结果。该脚本是分块省显存的离线流程，不能称为具有实时丢失恢复能力的在线SLAM。默认下载Nested-1.1研究权重；更换为Apache的小模型会改变能力，需要独立验收，不能继承原benchmark。[主流程run](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/da3_streaming/da3_streaming.py#L522)、[默认配置](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/da3_streaming/configs/base_config.yaml)、[下载脚本](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/da3_streaming/scripts/download_weights.sh#L12)。

本组暂不把其余三项排在这两个之前：SLAM3R缺少显式相机链路且受NC限制；纯MapAnything缺少长序列系统层；WorldMirror 2.0公开视频入口存在抽帧上限，未见回环且不是视频到mesh的完整实现。**MapAnything作为已有backbone仍值得保留**；本结论不评价由另一研究分工核查的Map-Long/VGGT-Long等外部系统，不能由“纯模型无回环”推断“没有可复用的长序列系统”。

## WorldMirror 2.0：需要分开的两条路径

真实重建路径为视频或图片目录 → `WorldMirrorPipeline` → 几何/相机/3DGS预测 → 保存结果。`pipeline.py`的`video_max_frames`默认32；`prepare_input()`第130行执行`min(64,max_frames)`，同时约束新、旧视频抽帧策略。图片目录分支不会施加该64帧上限，但仍一次加载为多视图推理输入，不是分块回环方案。[入口](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/hyworld2/worldrecon/pipeline.py#L395)、[抽帧逻辑](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/hyworld2/worldrecon/hyworldmirror/utils/inference_utils.py#L118)。

`save_results()`明确保存`gaussians.ply`、`points.ply`、相机、深度、法线和可选COLMAP；没有三角面导出。前者是高斯参数，后者由深度回投形成，PLY扩展名不能证明存在mesh。此处“未见mesh”限定为审阅过的公开worldrecon入口及保存函数，不外推为该团队完全没有mesh技术。[保存函数](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/hyworld2/worldrecon/hyworldmirror/utils/inference_utils.py#L673)。

另一条路径为文本/单图 → HY-Pano全景生成 → WorldNav轨迹规划 → WorldStereo合成新视图 → WorldMirror/3DGS优化。README的世界生成、mesh、可游玩场景展示不能证明真实工厂未观察区域的几何；WorldStereo合成帧也不能作为现场测量证据。两种能力在本报告中分开评价。[官方架构与worldgen入口](https://github.com/Tencent-Hunyuan/HY-World-2.0#-architecture)。

论文额外报告4张H20、SP+BF16+FSDP时128视图42.71GB/卡、5.60秒，256视图78.78GB/卡、17.52秒。这说明大批次可扩展性，不能替代长路径漂移、回环或工厂覆盖验证；其时间口径是模型推理，不是录像上传到可交付mesh的端到端时间。[论文表14](https://arxiv.org/html/2604.14268v1#S8.SS2.SSS2)。

## 许可证：代码与权重分别判断

| 项目 | 本次直接查到的条款 | 对选型的影响 |
|---|---|---|
| MASt3R-SLAM | 代码CC BY-NC-SA 4.0；MASt3R骨干也为NC-SA，checkpoint还要求遵守训练数据条款 | 不能称商业开源可直接采用；需要不同授权或不同技术路线。[SLAM许可](https://github.com/rmurai0610/MASt3R-SLAM/blob/main/LICENSE.md)、[骨干说明](https://github.com/naver/mast3r/blob/main/README.md#checkpoints) |
| SLAM3R | 仓库LICENSE为CC BY-NC-SA 4.0；两份HF模型卡未找到明确独立许可证声明 | 代码已不满足直接商业采用；权重许可也不能臆测为Apache。[LICENSE](https://github.com/PKU-VCL-3DV/SLAM3R/blob/main/LICENSE)、[I2P卡](https://huggingface.co/siyan824/slam3r_i2p)、[L2W卡](https://huggingface.co/siyan824/slam3r_l2w) |
| MapAnything | 代码Apache-2.0；默认`facebook/map-anything`为CC BY-NC 4.0；另有`facebook/map-anything-apache` | 这是本组最清晰的商业backbone选项之一，仍须确认实际使用的是哪个权重；模型版本和效果不可混用。[官方模型列表](https://github.com/facebookresearch/map-anything#models) |
| DA3 | 代码Apache-2.0；Nested/Giant/Large及-1.1权重为CC BY-NC 4.0；Base/Small、Metric-Large、Mono-Large为Apache-2.0 | Streaming默认Nested-1.1并非商业配置；Base/Small没有同等3DGS/米制组合能力，Metric单帧模型也不是多视图SLAM。[模型表](https://github.com/ByteDance-Seed/Depth-Anything-3#-model-zoo) |
| WorldMirror 2.0 | 代码和权重受Tencent HY-WORLD 2.0 Community自定义协议；排除欧盟、英国、韩国；版本发布日前一月总MAU超过100万须另获授权；禁止用其输出改进其它AI模型 | 不是Apache/MIT；“免费商用”不能省略地域、规模和用途条款。[完整原文](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/License.txt) |

以上是条款定位和摘要，不是已完成该产品具体使用方式的授权审查。第三方依赖及权重也须随实际候选确定，不能只看顶层仓库徽章。

## 实测数字如何使用

- MASt3R-SLAM约15 FPS与SLAM3R 20+ FPS来自各自论文，硬件分别为4090和4090D、场景与处理方式不完全相同；不能据此宣布后者在该工厂更快。SLAM3R某些消融表中的43/92 FPS只计对齐模块，不是完整系统速度。[MASt3R-SLAM实验](https://arxiv.org/html/2412.12392v2#S4)、[SLAM3R表4](https://arxiv.org/html/2412.09401v2#S4)。
- DA3-Streaming的8.51 FPS排除了warm-up、模型加载和PLY保存；“<12GB”对应KITTI低高比504×154、chunk30的11.5GB，不能用于室内504×378预算。官方TUM表中chunk120→30时ATE RMSE从0.087m变为0.227m，显示节省显存有精度代价；这是数据集轨迹指标，不是工厂构件误差。[官方实测表](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md#performance)。
- MapAnything的“2000视图/140GB”来自当前README能力说明；本次没有复跑其profiling脚本，也没有得到我们硬件的耗时。不可改写成消费卡可实时处理2000帧。[profiling说明](https://github.com/facebookresearch/map-anything#profiling)。

## 工厂场景仍未被这些来源证明的事项

以下是从需求和几何原理作出的验收约束，不是对任何候选已发生故障的断言：

- 视频必须有视差和重叠；原地旋转、快速模糊、极少纹理或反光表面不足以确定可靠几何。动态工人、叉车和设备运动需与静态空间区分。
- “完整空间”只能对实际观察覆盖负责。机器背面、遮挡处、天花/地面未拍摄区域不能靠外观完整的点云或生成视图宣称已重建。
- 单目模型预测米制尺度不等于校准过的实测尺度；绝对距离、门宽、通道宽度应通过独立已知尺寸/传感标定验证。真实尺度不由相机内参单独唯一确定。
- 3DGS是可视化场景表示，点云是采样几何，逐帧三角化mesh可能重叠且有孔洞；三者都不自动提供物体实例、封闭实体、CAD边面关系或可用碰撞几何。
- 真正决定能否采用的是同一段现场视频：覆盖完整度、回到起点是否重合、相机轨迹与原帧回投是否一致、与实测尺寸是否一致、导出的几何能否承接后续用途。不能用宣传视频或不同数据集最佳数字替代。

## 审阅版本

| 仓库 | 核查时main SHA | 最近commit UTC |
|---|---|---|
| rmurai0610/MASt3R-SLAM | `e6f4e3d474fad0e11f561482012be864ba8c3f17` | 2025-11-09 |
| PKU-VCL-3DV/SLAM3R | `f531d841ab743217a4464344119a350eb0556d17` | 2025-10-18 |
| facebookresearch/map-anything | `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9` | 2026-08-07 |
| ByteDance-Seed/Depth-Anything-3 | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | 2026-07-27 |
| Tencent-Hunyuan/HY-World-2.0 | `df9988efb87bfc0f4947eb3889411cf957478b06` | 2026-08-12 |

SHA由各仓库GitHub commits/main API当日读取。代码链接尽量固定于上述提交，模型卡和README的main链接后续可能更新。
