# Real2sim experiment — 2026-10-01

本轮完成现有四张照片的三组工程对照。**未证明测量精度提高；未接受新的米制尺度或相机。** 左右护板导出了带原图纹理的独立 GLB 对照，保留原顶点、面和位置；主报告 52 个对象仍可查看。光幕检测没有形成受支持的闭合可见面，未生成替代模型。

公开入口：[可旋转模型与原图诊断](https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/real2sim/real2sim.html)。主报告保留 52 对象，并提供本轮实验链接。

## 运行与交付

- 实验 A：`workcell-real2sim-2026-10-01-a`，临时 Modal 2×A100。缓存输入的实验墙钟 267.77 s；返回容器窗口 274.24 s；客户端调用 523.50 s，期间发生一次平台抢占重启。纹理和相机完成，光幕结果写入失败。
- 实验 B：`workcell-real2sim-2026-10-01-b`，只并行重跑纹理与光幕。实验墙钟 41.26 s；纹理 11.98 s、光幕 41.25 s；返回容器 45.06 s、调用 53.48 s。未重复相机拟合。
- 这些时间复用已有几何、分割和模型，不是新四图完整 oneshot。此前完整流程记录仍为 366.93 s。
- 两次返回容器窗口按资源价估算共 $0.567；含抢占、调度的完整调用窗口计价共 $1.024。前者不含已被回收的尝试，后者包含非计算等待；两者都不是账单。`actualBilledUsd=null`。原始 `spend-ledger.json` 与 `platform-events.json` 保留。
- 使用既有 `modal_apps/workcell_guard_experiments.py`：`--mode real2sim` 运行全对照，`--mode real2sim-details` 仅细节分支。始终 ephemeral `modal run`，2×A100，未部署服务。

## 外观与光幕实测

A 的多图补纹理出现条纹接缝。原因是已有相机和表面不能把多张照片的纹理精确对齐，逐像素从不同照片填空会形成错开的条纹。B 改为每片选一张质量优先且有可见支持的原图，未知区域保留原模型外观，不把更多覆盖率当成正确率。

B 左护板两片照片覆盖率 59.2% / 55.8%，右护板 69.3% / 50.4%。四片的导出读回均验证顶点、面与节点变换不变。未覆盖处仍可见旧外观接缝；本轮作为可旋转实验模型供比较，未提升为默认生成路径或物理精度改善。

光幕失败先定位到共享 `_interval_union`：合并区间时写回了 `numpy.float32`，导致嵌套诊断 JSON 无法序列化。修复为与非重叠路径一致的 Python float，未改几何值或门槛。合成真实检测 schema 先复现失败，再验证修复。B 成功输出诊断，但 105 个候选上下端不对应同一组被观测侧线，另 2 个视图缺少同图上下端组合，因此 **0 个光幕面候选**。这是当前边缘关联方法的失败；不能据此宣称照片不可能重建，也没有用 20/24 cm 检查值补造模型。

## 本地空间与云存储

用户已授权删除无用旧数据。删除了有 SHA256 对照的旧工作单元重复大文件，以及 2026-09-21 已退役的生成产物；原始照片、当前公开报告、五个冻结输入根目录顶层资产和诊断 JSON/日志保留。部分旧 `page/` 的重复大文件已去除，复现应从顶层资产重封装。清单：`research-notes/workcell-duplicate-cleanup-2026-10-01.json` 和 `research-notes/phase2/retired-generated-cleanup-2026-10-01.json`。可用空间从约 4.1 GiB 恢复至约 8.6 GiB。

Supabase Storage 可用于大文件；本轮只读检查的 `wekruit-pa` 上传实现使用 Firebase Storage，仅发现旧 Supabase 数据库线索，未找到可复用 bucket。没有读取凭据或迁移业务数据；本轮继续使用已连通的 Modal 计算路径与现有报告发布路径。

## 复现与检查

四张原 JPEG：`panoptes-serving/runs/user-bor1-02/input/image_01.jpg` 至 `image_04.jpg`。冻结 baseline：`research-notes/workcell-endpoint-estimate-2026-10-01`。

```sh
PYTHONPATH=.:scripts:modal_apps python -m modal run modal_apps/workcell_guard_experiments.py \
  --mode real2sim --baseline /path/to/workcell-endpoint-estimate-2026-10-01 \
  --sources /path/image_01.jpg,/path/image_02.jpg,/path/image_03.jpg,/path/image_04.jpg \
  --measurements docs/workcell-photo/measurements-2026-10-01.json \
  --out /path/to/new-experiment
```

轻量检查：`check_workcell_button_bundle.py`、`check_workcell_photo_texture.py`、`check_workcell_post_faces.py`、`workcell_real2sim_report.py --check`。按钮检查覆盖非平凡相机投影、像素中心、轨迹顺序、不同 gauge 与整颗按钮观测留出（该图场景轨迹保留）；纹理检查覆盖 UV、遮挡、单源选择与 GLB 不改几何；光幕检查覆盖相同端点测量、原图变换、平面与断边拒绝、真实诊断 JSON。

## Camera and reference-size controlled comparison

Artifacts: `/Users/adam/Desktop/panoptes-public/research-notes/workcell-real2sim-2026-10-01-a/real2sim/`.
These are cached-input experiments, not a fresh full photo-to-model benchmark.
Only the confirmed 10 cm whole-component height, 8.5 cm main diameter and 4 cm red diameter enter the reference fit; evaluation clearances do not enter fitting.

### Same inputs, different capped track selection

Both arms use the same four original JPEGs, the same `camera-control` input K/poses, and exactly equal `observations`, `sourceInputs`, `reference` and `gauge` values. Canonical JSON SHA256 (`sort_keys=True`, compact separators):

- Observations: `532d13acb794d3d299cd99651d845b8f6c92f93ad792d9405d09c0a8d826ac36`.
- Source inputs including input K/poses: `52712bd93d42c7cb4690ea5997b2683ac73c3c23bda35ccd6ed004841088bbf3`.
- Saved fit `sourceHash` (existing non-compact JSON convention): `343990b6ade0730c28b9760df06098ac3cef4ca529d8e47546f1f8d622c76be8` in both arms.

The original JPEG SHA256 values also match `camera-control/source-frames.json`. Relative to frozen runB, both colored hulls, the housing-bottom support and bounding boxes are unchanged. Only Photo4's rejected yellow-boundary vertices have small Lab-contrast metadata differences; their pixel coordinates and excluded support direction are unchanged.

| Metric (original JPEG pixels) | Spatial round robin | Connectivity balanced |
| --- | ---: | ---: |
| Retained tracks | 180 | 180 |
| Distinct tracks linking photos {1,2} to {3,4} | 29 | 41 |
| Full-fit maximum button error | 7.174942 | 6.954598 |
| Full-fit scene reprojection P95 | 3.587568 | 5.018663 |
| Photo2 whole-button holdout maximum | 108.436099 | 130.032168 |
| Photo3 whole-button holdout maximum | 43.529778 | 56.042383 |
| Photo4 whole-button holdout maximum | 31.698372 | 41.711873 |
| Median of the three holdout maxima | 43.529778 | 56.042383 |

There is **no demonstrated accuracy improvement**. All three held-out errors worsen (19.9%, 28.7%, 31.6%). Both full fits and every holdout reached the unchanged 100-evaluation limit without convergence; full-fit button support also fails. Balanced additionally fails scene-reprojection support. All final statuses remain `unsupported`, `mPerNative` remains null, and candidate scale values 0.65479648 / 0.65507492 are not accepted physical calibration. Pixel errors must not be relabeled as centimetre accuracy.

Balanced retains all 41 eligible tracks crossing the two photo groups. Photo2–3 and Photo2–4 still have only 5 and 10 tracks, respectively, already fully retained by the original selector. More balanced counts cannot supply missing or correct invalid correspondence evidence. `spatial_round_robin` remains the default in `_prepare`, `solve` and `build`; balanced is an explicitly selected experiment only. When no truncation is required, either arm preserves the original deterministic ordering so it does not unnecessarily perturb sparse numerical optimization. Synthetic checks cover explicit balanced selection, capped connectivity, exact uncapped ordering, gauge and raw-pixel projection.

### Independent pose initialization from the same SIFT database

The neural-pose initialized control registered four images, producing 692 points with mean reprojection error 0.432584 **control-image pixels** after BA (control resize factor 1554/4032 relative to the original JPEGs). Those four registered cameras do not constitute an independent validation of the neural poses.

Native incremental COLMAP used the same SIFT database and SIMPLE_PINHOLE intrinsics initialization, with no initial camera poses. It ended normally after 3.70 s, not at the 180 s time bound. It produced two disconnected reconstructions:

- Photos 1/2: 76 points; mean 0.462809 control-image pixels.
- Photos 3/4: 771 points; mean 0.278117 control-image pixels.

No single model registered all four images, so the result is `unsupported` and no replacement-camera files or independent joint fit were produced. The database SHA recorded by the worker is `4d0a426af9b9579976e7f4b6cc2cdb92349d28bfbc7e3e9e82e5da2887a74c70`; source-metadata SHA is `2469b1488c217b292201385324981975e00c74ccffcc0cc7990d51ac5b39514d`.

This supports insufficient reliable cross-group geometry as a current pipeline limitation, beyond track-budget allocation. It does not prove that the photographs are intrinsically unusable, that viewing angle alone is responsible, or that more iterations would solve the problem. SIFT association, pose initialization, and button shape/contour consistency are not yet isolated from each other; the nonconverged fits limit further causal claims. No gates were relaxed and no failed camera candidate replaces the current report geometry.
