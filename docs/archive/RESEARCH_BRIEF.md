# Panoptes × World Models — 研究 Brief（给研究 session）

目标：基于 Panoptes 现有系统，评估最新空间/世界模型能否**可测量地**提升
工位理解质量。产出必须是可落地的实验计划 + 可对比的数字，不是综述。

## 我们现在是什么

一条确定性的工位合规管线（代码在 `ehs_spatial/`、`scripts/`）：

```
1-4 张照片
 → 方向判定 (Gemini)
 → 多视角几何 MapAnything（度量点云 + 相机位姿）
 → 文本词表分割 SAM 3（围栏/机器人/托盘/传感器…）
 → 尺度锚定 MoGe-3（单目度量深度中值比，校正 MapAnything 绝对尺度）
 → 场景构建：地面拟合、实体 3D 足迹、Manhattan 主方向
 → 装置检测层（taxonomy A-F，VLM 出框 + 裁剪自检 + SAM box-prompt）
 → inventory 精修：多短语并集、法线分割、接触边回投、护栏链、
   cell-rectangle 约束（工位=矩形先验）、payload→policy 闭环
 → policy 判定（间距/入侵/高度…）+ climb review
 → 交互报告（点击选物、平面图、距离矩阵、3D）
```

关键设计约束（改动前必读 `HANDOFF_FULL.md`）：
- 一切几何行为由 `tests/test_geometry_invariants.py` + `tests/test_cell_rect.py`
  钉死：共线 <0.30m、平行 <2°、Manhattan <6°、回投偏差上限（01: 0.10、
  02: 0.25、03: 0.10 图高占比）。**任何新模型必须让这些数字变好或持平**。
- VLM 枚举到的物体必须有实例或 unresolved 记录，禁止静默丢弃。
- 模型后端全部接口化（`docs/BACKENDS.md`）：换模型 = 实现同 schema 的 endpoint。

## 我们的痛点（按对产品的影响排序）

1. **透明/网状结构的几何**：透明围栏、铁网在单目深度里"透"到背景，
   实体足迹被后面的东西污染。现在靠 MoGe 法线分割 + 最近深度簇门控硬修。
   → 世界模型若能给出**遮挡感知的实例级几何**（前景薄片 vs 背景），直接命中。
2. **跨视角实例融合缺失**：同一容器在两帧里是两个实体（世界坐标差 2.6m）。
   现在 inventory 层按帧独立。→ 需要统一场景表征里的实例一致性。
3. **绝对尺度**：MapAnything 与 MoGe 原始尺度差 ~1.4×，靠中值比校正。
   → 有更稳的度量先验（相机高度、已知物尺寸、地面重力）的模型可对比。
4. **回投分数瓶颈**（real-clean-02 停在 0.25）：围栏底边被遮挡时接地线不可见。
   → 能推断"看不见的接地线"的模型（先验补全）是差异化点。
5. **长尾类别召回**：text-SAM 词表召回不稳；检测层是 VLM 出框。
   → 开放词表检测/分割的新模型（或 3D-aware 检测）对比。
6. **速度**：单图 55s / 4 图 ~4min（云 API 冷启动主导）。端到端世界模型
   若一次前向给出几何+语义，可能压缩链路。

## 候选方向（研究 session 自行核实最新版本与许可，以下为起点）

- **World Labs（李飞飞团队）**：Marble / 生成式世界模型系列 —— 评估：从 1-4 张
  照片生成一致 3D 场景的能力是否可作为"几何补全先验"（尤其遮挡接地线、
  透明面后方）。关注：可导出显式几何（gaussian/mesh）否、度量尺度否、API 否。
- **OpenSpatial / 空间理解基座**：评估其空间关系推理（"A 在 B 后方 0.5m"）
  能否替代/增强我们计划中的 VLM 关系层（L2）。
- **前馈多视角重建新模型**：VGGT、π3、Depth Anything 3、MapAnything 新版本 ——
  和当前 MapAnything 在我们三个 canonical run 上直接 A/B（回投分数、
  尺度误差、透明结构足迹面积）。
- **3D-aware 开放词表分割/检测**：任何能在点云/多视角上直接给实例的模型
  （对比 text-SAM 词表 + VLM 出框两层）。
- **SpatialVLA / 具身空间 VLM**：作为 climb review 和关系推理的替代评估。

## 实验协议（必须这样对比，否则结论无效）

1. 数据：`runs/real-clean-01/02/03`（canonical）+ `runs/user-bor1-02`（BOR1 工位），
   输入图在各 run 的 `input/`。
2. 接入方式：实现 `docs/BACKENDS.md` 对应 endpoint（或 adapter），**不改管线**。
3. 指标（现有脚本全能算）：
   - `scripts/verify_reprojection.py --run X` → 回投偏差（图高占比）
   - `uv run pytest tests/test_geometry_invariants.py` → 共线/平行/Manhattan/上限
   - inventory.json 里 `cell_rect`、`outside_cell`、实体数 vs 人工清单
   - 尺度：MoGe 锚定置信 + 与已知尺寸物（托盘 1.2m、护栏高 1.1m）对比
   - 时延与成本
4. 报告格式：每个候选一张表（同一 run、同一指标、前后对比），
   附失败案例截图。没有数字的"看起来更好"不算。

## 边界

- 不改判定语义（policy 阈值、状态枚举）。
- 不引入需要训练的方案作为第一步；先零样本/前馈评估。
- 工厂照片不外传：所有实验在本地/私有算力跑，API 上传前先确认许可。

## 现成资产

- 公开源码：https://github.com/admin-wekruit/panoptes-serving
- 私有仓库（含 runs 数据、git 历史）：https://github.com/admin-wekruit/ehs-spatial
- 4 个真实 run 样例包：私有 Release `handoff-2026-08-30` → `panoptes-all-in-one.tar.gz`
