# Phase 2：公开视频到室内空间

2026-09-17 开始执行。这里的 Phase 2 指照片阶段之后的视频空间阶段，不等于实施计划中的内部“阶段 2：上传”。不进行基础模型训练或逐场景训练。

- 照片阶段：[归档及验收缺口](../platform/PHASE1-ARCHIVE-20260917.md)，归档提交 `1a3529a`（本分支 `b659ea5`）。
- 完整目标：[视频到可编辑空间](../algorithms/2026-09-17-video-full-scene-plan.md)。当前执行公开室内样本，之后换入用户家庭视频。
- 实验分支：`codex/phase2-video`；原照片源码及公开报告保持其归档版本。

## 首个公开样本

[TUM RGB-D fr1/room](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download#freiburg1_room)，CC BY 4.0，真实办公室房间环行。包含墙、家具及杂物；它是室内技术对照，不是用户家中的数据。

公开页面标称 48.90 秒；实际 RGB 文件为 1,362 帧、45.3717 秒，深度 1,360 帧。20 ms 时间匹配形成 1,352 对。更长的真值轨迹覆盖 48.8991 秒。原始 TGZ 782,381,450 字节，SHA256 `5ace47a1d2e53696bc939a84999293a04a7226958e848a609666691fd3cc38da`。

数据与输出放在 Git 以外的 `/Users/adam/Desktop/panoptes-public/research-notes/phase2/`；`data/tum_freiburg1_room.manifest.json` 保存下载来源、许可、标定、索引及逐文件校验。

## 两个明确区分的实验

| 实验 | 重建时输入 | 输出及用途 | 仍未验证 |
|---|---|---|---|
| RGB-D 连续运动对照 | 连续 RGB、传感器深度、公开相机参数；不读真值位姿来建图 | Open3D 估计轨迹、TSDF 三角网格；完成后与真值比较 | 普通 RGB 视频、持久地图重定位、对象语义 |
| RGB 预训练几何实验 | 同一录像中 8 个 RGB 关键帧；不提供深度、真值位姿或真值尺度 | 已缓存 MapAnything Apache 本地 MPS 推理、原生相机及深度、融合网格 | 连续 SLAM、跨次地图记忆、每个物体可编辑建模 |

RGB-D 对照的事先检查目标（首轮运行前记录）：有效关联帧跟踪比例至少 95%；米制刚体对齐的轨迹 ATE RMSE 不超过 0.15 米；网格有限、非空且有真实三角面。失跟必须记为失败并停止累计，不沿用旧位姿继续堆叠。目标未通过就保留失败；这些是本实验的筛查目标，不是现场精度承诺。两组都记录所有输入、尝试、时间、内存和失败状态。

RGB 稀疏实验用于验证现有模型能否生成房间可见表面，不用成功导出或三角面数量宣称完整房屋模型。传感器深度与真值轨迹只供独立评估，不输入这组推理；模型预测米制尺度另待独立检验。

## 与完整目标的连接

下一步仍按原计划验证 ORB-SLAM3 的权威轨迹、回环与地图保存/重载，再接现有深度、VLM/分割、实体累计、粗模型和报告。当前原生视频上传仍未接入生产；不会把固定 4 张照片契约直接放宽后声称视频功能完成。RGB-D 基线与稀疏 RGB 实验均不能替代跨次重定位验收。

用户输入应保留原始连续视频、拍摄顺序和原文件；从房间一角缓慢移动，扫到地面、墙角和家具侧面，再返回起点。若需要米制尺寸，另提供至少一个清晰可对应的实测长度。镜头及缩放应保持固定，不能用未知相机高度指定尺度。

本页下方保存首轮实际结果，后续尝试追加记录。

## 首轮实跑结果

本机 Apple M2 Max、32 GiB RAM；全部本地执行，无训练、无新增付费 GPU/API 调用。下列耗时包含本次模型装载和导出，不包含首次模型下载，也不代表任意视频的处理速度。

| 运行 | 实际输入/输出 | 用时 / 峰值进程 RSS | 独立检查 | 结论 |
|---|---|---|---|---|
| `rgbd-control-01` | 首帧后 macOS Open3D tensor RGBToGray 未实现 | 失败记录完整保留 | 未进入有效轨迹验收 | 环境接口失败；之后使用同库已支持的 CPU legacy API |
| `rgbd-control-02` | 1,352/1,352 帧跟踪，272 帧融合，935,114 个三角面 | 61.64 秒 / 1.24 GB | 米制 SE3 ATE RMSE **0.211 米**，超过 0.15 米目标 | 执行完成，质量验收失败；说明局部跟踪成功仍会累计漂移 |
| `rgb-mapanything-001` | RGB 8/8 关键帧，78,392 个三角面 | 35.41 秒 / 7.81 GB | 原尺度 SE3 ATE **1.842 米**；有效像素深度中位相对误差 **17.61%** | 房间布局明显失真，保留失败对照 |
| `rgb-mapanything-032` | RGB 32/32 关键帧，166,442 个三角面 | 37.66 秒 / 10.00 GB | 原尺度 SE3 ATE **0.641 米**；有效像素深度中位相对误差 **15.85%** | 布局明显改善，仍有孔洞与不准确位置，未验收为完整家庭模型 |

8/32 帧分别覆盖同一 RGB 录像的 45.37 秒。32 帧中的模型加载 23.20 秒、联合推理 6.80 秒；MPS driver 抽样峰值 8.78 GB，不与进程 RSS 相加。两组取样和有效像素集合不同，不能把百分比变化当成同一观测集合上的严格消融结果。

独立评估在推理结束后读取传感器深度/相机真值，未回写预测或网格。另做 Sim3 拟合只作尺度诊断：8 帧残差 0.934 米、32 帧 0.333 米；没有用拟合结果伪装原始输出精度。完整逐帧指标、匹配时间差和文件哈希在各运行的 `evaluation.json`；对应 `evaluate_rgb.py` 可复跑。

### 已定位的几何问题

- 原生点图与预测相机的 Z 自洽不足以证明完整几何自洽。8 帧原生 XYZ 反投影的 p95 像素残差为 6.03–9.52 像素（336×252）；拟合 pinhole K 与原生点图存在差异，TSDF 采用 K+z 时改变了部分射线。后续输入要同时记录 XYZ 像素残差。
- 更大的问题已存在于原生跨视角预测：在 TSDF 之前，其他帧的原生表面就会挡在当前观察前面。因此主要错位不是 GLB、前端矩阵或 TSDF 的 c2w 取逆写错。
- 浏览器实际检查了两组 RGB 的整体视角、来源视角对照和 RGB-D 网格。32 帧改善房间轮廓，仍不能由导出成功、三角面增加或单视角外观推断完整性。页面明确标出对应误差和未验收状态。

这支持继续原方案：**先接可靠的连续轨迹、回环和全局优化，再让几何、对象模型使用同一坐标**。本轮不通过重新训练或删除难对象来掩盖位置问题。

## 复现入口与下一份输入

代码是实验 CLI，尚未部署到公开报告上传入口。可以直接接原始视频，不需要用户手工抽帧。使用已有固定源码、权重和环境：

```bash
/Users/adam/Desktop/panoptes-public/panoptes-serving/.venv/bin/python \
  scripts/reconstruct_room_rgb.py \
  --video /absolute/path/home.mp4 --views 32 \
  --output /absolute/path/new-home-run \
  --serving-root /Users/adam/Desktop/panoptes-public/panoptes-serving \
  --vendor-dir /Users/adam/Desktop/panoptes-public/panoptes-serving/outputs/candidate-evaluation/vendor/map-anything \
  --model-dir /Users/adam/Desktop/panoptes-public/panoptes-serving/outputs/candidate-evaluation/weights/map-anything-apache
```

只检查视频解码时加 `--prepare-only`；不调用模型。视频顺序解码，保存实际选取帧号与时间戳、原视频及解码图像哈希。此入口当前是最多 32 帧联合几何实验，没有连续跟踪/地图重载能力。每次使用新的输出目录；失败与先前运行不覆盖。

原始样本回放在外部产物目录的 `source-replay.mp4`，由公开 RGB PNG 和时间戳重新编码，**不是原始相机容器**；来源记录在 `source-replay.json`。浏览器预览复用已有 `lucida_viewer.html` 及索引三角网格格式，原始点/相机/PLY/GLB 均保留；`tools/build_*_preview.py` 保存预览转换。运行 `python -m http.server 8799 --bind 127.0.0.1 --directory /Users/adam/Desktop/panoptes-public/research-notes/phase2` 后可查看：

- RGB 32 帧：`http://127.0.0.1:8799/runs/rgb-mapanything-032/preview/index.html`
- RGB-D 对照：`http://127.0.0.1:8799/rgbd-control-02/preview/index.html`

检查：原视频相关测试与 RGB-D 契约检查共 34 项通过；RGB 实验自检覆盖深度/K/坐标域、刚体相机、TSDF 取逆、真实 MP4 帧号/时间/RGB、空视频拒绝；两组网格导出回读与哈希检查通过。**代码检查通过和实验执行完成，不改变上述几何验收失败/未通过的状态。**
