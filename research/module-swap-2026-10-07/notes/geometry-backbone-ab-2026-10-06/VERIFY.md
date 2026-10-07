# 核验：MVS 路线替代 Pi3X / Verification: the MVS route as a Pi3X replacement (2026-10-06)

核验对象：本目录 `README.md`、`results.json` 及其代码，以及导出的 090 几何。四个独立核验员（数字复现 / 公平与泄漏 / 实际运行的许可 / 导出可用性）全部只读；本地计算用 `nice` + serving venv，没有跑 Modal。本文件不改 README。
Scope: this note's `README.md`, `results.json` and code, and the exported 090 geometry. Four independent read-only verifiers (numbers / fairness and leakage / licence as run / export usability), local compute only, no Modal. This file does not edit the README.

---

## 中文

### 总判定：**可以，但要先修**

- 几何本身成立：数字可复现，评判器未改，冻结输入一致，现场值没进入任何输入，几何过滤没放松，导出数组就是打分的数组，原 run 未被改动。
- 但新报告不能照搬 README 的说法，有 6 个主要问题：
  - 标题数字靠两处联动改动，不是一处。
  - "超过 Pi3X"来自路线本身，不来自换掉 Pi3X。
  - pycolmap 轮子**确实**含 GPL-2.0+ 代码。
  - 补全 A/B 的 assemble 阶段在这份导出上会报错。
  - compare 的默认世界坐标不对。
  - 4 个物体的 SAM 3D 点图大部分是空的。
- 都能修，或者在报告里写明；没有一项推翻"许可干净的权重 + 这条路线在 4 个现场值上通过门限"。

### 说法 × 状态 × 证据

状态：确认 / 有保留 / 否定 / 未确认。

| # | 说法 | 状态 | 证据 |
|---|---|---|---|
| 1 | `results.json` 可由逐 run 文件 + `compile.py` 重建 | 确认 | 重建结果只差 `sources.clean` 路径一行（NOTE 改了位置）；没有重复键。`scratchpad/verify-numbers/rerun.out` |
| 2 | 判定表、全部候选表每个数与 `results.json` 一致 | 确认 | MVS DA3 起始：3.304 / 1.513 %；24.024 / 23.611 / 24.980 / 20.994；MAE 0.5968，最大 0.994；相机高极差 1.676。焦距 375.7 / 376.2 / 372.1 · 381.2 / 374.9 |
| 3 | 本地重跑未改的 `analyse_one` 复现 | 确认 | Pi3X 已发布与 MVS 都在 ≤ 0.0084 cm 内（库版本不同）。`scratchpad/fairness-lens/run_local.py` |
| 4 | Pi3X 参考行 = 公平 A/B 的 1.4 / 2.7 | 确认，有保留 | 6 个配置差 0.0（MAE 1.44052，最大 2.67258）。但这是"Pi3X 已发布几何 × 公平评判器"。按已发布数字本身是 1.052 / 2.153（公平 README L113）。MVS 两者都赢 |
| 5 | 评判器与公平 A/B 相同 | 确认 | `analyse` / `analyse_one` 在公平打分后未改。后来只加了 `freeze`（L112-123、L303-306）。`polygon_mask` 与 16c4a03 逐字节相同 |
| 6 | 所有主干用同一批冻结补边 518 帧 | 确认 | canonical sha256 在 manifest、da3fair-data、10 份 bbab 几何、clean-geom 和导出中一致。RoMa 稠密 warp 的输入只由代码路径证明（npz 不存输入哈希）：**未确认** |
| 7 | 现场值从未作为输入或调参目标 | 确认 | grep 路线代码，没有 FIELD_CM / PIN / 物体 id。现场值只出现在打分侧 |
| 8 | 几何过滤没放松 | 确认 | FB 1 px、夹角 2°、重投影 1 px、双深度 > 0，10 份 manifest 都记录。`mvs_route.py:27` 只覆盖 CERT |
| 9 | "只改了一处"（CERT 0.5 → 0.05） | **否定** | `mvs_route.py:46-47` 还把保留像素的 `conf` 设成 1。这关掉了 harness 的 conf ≥ 0.1 规则，影响 RoMa 置信度落在 [0.05, 0.1) 的像素：090 占保留像素 6.3–7.1 %，030 占 3.2–3.7 %。用 conf = 置信度重算：24.06 / 23.61 / 25.61 / 20.99，**MAE 0.77，最大 1.61**，仍通过。`scratchpad/fairness-lens/res-*-confcert.json` |
| 10 | 0.05 在算新数之前定下 | 确认，有保留 | 时间线（UTC）：20:05:01 写入，20:07:32 打分，20:08:07 加敏感性参数。值是 RoMa 默认 `sample_thresh`，没扫。但 20:03:23 已看过 CERT 0.5 的逐物体分数。这是同 4 个现场值上的第 4 次尝试。`sample_thresh` 在 RoMa 里是采样饱和值，不是稠密 warp 的有效性门限 |
| 11 | "删像素的是置信度，不是几何" | 有保留 | 090 照片 3 ← 1 成立：中位 0.14 / 0.14 / 0.14。030L 照片 2 不成立：FB < 1 px 删掉 70 个里的 42 个（FB 中位 3.11 px），CERT 0.05 保留的 28 个正好等于 FB 允许的数。诊断输出没存进笔记 |
| 12 | 带内点数 9→28、15→47、45→67、0→46 | 确认 | surfacePoints 字段一致；MoGe 起始相同 |
| 13 | 敏感性 0.05–0.25 都通过 | 确认，有保留 | 主配置三档都过。CERT 0.25 时 090R 照片 3 只有 11 点（MIN_PTS 12），只剩照片 1；090L 为 None。最低平面地面 p95 不单调：2.69（不过）/ 2.45 / 1.22 |
| 14 | 起始无关，"各值差 ≤ 0.1 cm" | 有保留 | 只在 maxInlier / lsq / 竖直面成立：最大差 0.092（090R）/ 0.098 / 0.081。最低平面下围栏差 0.63（19.31 对 19.94），MAE 0.54 对 0.36 |
| 15 | 三种地面 × 竖直面都过，"MAE 0.4–0.7" | **否定**（范围） | 实际 0.33–0.70：MoGe 起始最低平面 × 竖直面 = 0.329（`rows[65]`）。DA3 起始 0.468–0.699。判定不变 |
| 16 | "超过 Pi3X" | 有保留 | 收益来自路线（BA-f + MVS），不来自离开 Pi3X。Pi3X + BA-f 是 0.78 / 1.75。从 Pi3X BA-f 相机跑同样的 CERT 0.05 MVS：23.97 / 23.54 / 24.91 / 20.97，MAE 0.59，最大 0.97，打平。两行都不在 bbab ROWS 里。`scratchpad/fairness-lens/res-*-pi3x-start.json` |
| 17 | CERT 0.5 重跑 = clean A/B MVS（< 1e-6 cm） | 确认 | pts3d / valid / K / c2w 逐字节相同，只有 conf 不同。最大差 4.33e-8 cm |
| 18 | 急停尺度是各 run 自己的，3.30 / 1.51 % | 确认 | 090 是 3.2371 m / 单位，最差特征照片 1 红边 −3.30 %，离门限 0.7 pp。030 的尺度支撑偏薄：204 个急停样本中 108 个有内容（DA3 为 204 / 204） |
| 19 | 门限 MAE ≤ 1.56 = 1.44 + 0.12 | 确认 | 1.44052 + 0.11967 = 1.5602。草稿曾写 1.6，打分后改成 1.56（更严，不偏向候选） |
| 20 | 无逐物体特判 | 确认 | 路线代码里没有物体 id；`gc.mvs` 对所有内容像素做同样检查 |
| 21 | 花费 ≈ $0.32 | 确认，有保留 | 评判 16 个 job $0.2082 + ALIKED $0.1131 = $0.3212。VGGT 访问"< $0.001"没有账本：**未确认** |
| 22 | 导出通过契约检查，数组 = 打分数组 | 确认 | `check_geometry` 通过；5 个数组 × 3 帧 sha 与 `bbab-geom/090-mvs-da3-base-padded` 相同；背后 0 个点；c2w det 1 |
| 23 | 重投影 0.14–0.15 px 证明精度 | 有保留 | 这是方法内置的过滤（REPROJ_PX 1.0）被再测一遍，不是独立证据 |
| 24 | 主点 / 焦距 | 有保留（信息） | 主点 259.0（继承 DA3），不是 258.5，差 0.5 px，约 0.08°，对 cm 量级可忽略。焦距 372–376 px，Pi3X 378.6–382.5 px |
| 25 | floor.json 尺度 = 门限值 | 确认 | `estopNativeToMeters` 3.23714，偏差 0.033045。地面点 35 165（= 18 111 + 17 054），Pi3X 35 487。`inlier_points` = null（Pi3X 35 430） |
| 26 | mask / rgba 是"只读符号链接" | **否定** | 链接目标是 -rw-r--r--、属主 adam。往 RUN/input 或 evidence/objects/* 写会改到 lucida-replica-01 |
| 27 | 原 run 未被改动 | 确认 | 82 个链接、21 个几何文件、objects.json、floor.json、125 个物体视图哈希都匹配。lucida 里 10 月 6 日没有写入 |
| 28 | 下游可直接跑补全 A/B | **否定**（现状） | `completion_ab.py:390-391` 打包链接时没有 dereference，`:252` 的 `extractall(filter='data')` 报 AbsoluteLinkError（本地复现）。导出没有 `generation/`，RecGen 参照臂无法组装。`compare.py:30-46` 默认 boxes 文件在 Pi3X 世界（1.28601，地面法向不同）。floor.json 键名是 `estopNativeToMeters`，直接传给 CMP_SCALE 会 KeyError。`--stage sam3d` 不受影响 |
| 29 | 覆盖率是 Pi3X 的 21–97 % | 确认，有遗漏 | 数字对。漏了围栏：left_fence 0.31 / 0.61，right_fence 0.65 / 0.79。全帧内容覆盖 57 / 49 / 63 %（Pi3X 96 / 95 / 96 %）。每视图深度样本 79–1 572（Pi3X 210–2 414） |
| 30 | SAM 3D 可用这份点图 | 有风险，**未确认** | SAM 3D 实际用的那张照片上的 mask 覆盖（括号里是最大空洞占 mask 的比例）：left_light_curtain f1 0.21（0.78），right_post f2 0.30（0.69），left_fence f2 0.31（0.52），left_post f1 0.35（0.65）。Pi3X 是 0.93–1.00。位姿和尺度能否保住要 GPU 跑才知道 |
| 31 | 护罩点图 | 有保留 | 双峰：离中位深度 > 20 % 的点占 8.9 / 23.9 / 22.3 %，Pi3X 0.2 / 0.6 / 0.0 %。哪个对：**未确认** |
| 32 | 090L 23.9（Pi3X 19.9） | 有保留 | 没有现场值；带内 22 对 63 点，支撑更薄。不能写成"改进" |
| 33 | 导出 manifest 来源信息 | 有保留 | `privacy` 仍写 Pi3X/SAM2.1。`created_at_utc` 和 `script_sha256` 是 lucida 的。地面 mask 的分割器版本没记录 |

### 必修清单（按严重度）

**主要（改标题或挡住下游）**

1. **两处联动改动**（README L19 / L40 / L142-143）。写成：CERT 0.5 → 0.05，加上保留像素 conf := 1（后者对 3–7 % 像素关掉了 conf ≥ 0.1）。在 0.60 / 0.99 旁边报 conf = 置信度的 **0.77 / 1.61**（030R 25.6 cm），两个都通过。
2. **pycolmap 含 GPL**。改掉三处"可能 / 推断"：README L29 / L50 / L157-158、`licences.md` L58 / L160；判定表许可格（README L78）和 `results.json` 的 `licence` 字符串都要改。PyPI 轮子 `pycolmap-4.2.1-cp311-cp311-manylinux_2_28_x86_64.whl`（sha256 `7627f0ba…dd90f`）的 `_core.so` 静态链接了 SuiteSparseQR 和 CHOLMOD Supernodal / MatrixOps，都是 GPL-2.0-or-later。clean A/B README L20 / L61 / L208 写的"BSD-3、许可干净"也要改。交付需要二选一：履行 GPL，或者重编不带 SuiteSparse 的版本（没试过，数字要重核）。
3. **补全 assemble 会断**。`completion_ab.py:390-391` 要么用 `tarfile.open(..., dereference=True)`，要么把 mask 实拷进导出。注意：该目录归另一个 agent，本核验没改。
4. **不要混世界坐标**。AB_RUN=export 时：CMP_BOXES=none；CMP_SCALE 文件的顶层键必须是 `nativeToMeters` = 3.2371。RecGen 臂要么重新生成，要么标"不可用"；lucida 的 `generation/` 在 Pi3X 世界。
5. **SAM 3D 风险物体**。报告里逐物体写出点图覆盖：左光幕 21 %、右立柱 30 %、左围栏 31 %、左立柱 35 %，对 Pi3X 93–100 %。这些物体的补全不能写成与 Pi3X 等价。
6. **RoMa / DINOv2 权重没锁**。`roma_outdoor.pth`（445 647 516 B，github.com/Parskatt/storage）和 `dinov2_vitl14_pretrain.pth`（1 217 586 395 B，dl.fbaipublicfiles.com）都是运行时下载，没校验哈希，实际用的字节没记录。任何 on-prem 说法之前：先镜像，记 sha256，再显式传 `weights` / `dinov2_weights`。

**次要（措辞 / 完整性）**

7. "超过 Pi3X"（README L12 / L33）改成：Pi3X 走同一路线时打平；本路线优于 Pi3X 已发布管线。加两行对照：Pi3X + BA-f 0.78 / 1.75，Pi3X 起始 CERT 0.05 MVS 0.59 / 0.97。
8. 1.4 / 2.7（README L14 / L35）标成"Pi3X 已发布几何 × 公平评判器"；已发布数字本身是 1.05 / 2.15。
9. Limits 加一行：CERT 0.05 是同 4 个现场值上的第 4 次尝试，是在看到 CERT 0.5 的逐物体不完整之后才改的。值本身是 RoMa 默认，20:05:01Z 写入，20:07:32Z 打分。
10. "各值差 ≤ 0.1 cm"（README L16 / L37）限定为 maxInlier / lsq / 竖直面；最低平面下围栏差 0.63 cm，MAE 0.54 对 0.36。
11. "MAE 0.4–0.7"（README L17 / L38）改为 0.3–0.7。
12. "置信度而非几何"（README L20 / L41）限定为 090 照片 3 ← 1；030L 有 60 % 是 FB 删的。把 `band_diag` 输出存进笔记。
13. 敏感性（README L22 / L43）注明：CERT 0.25 时 090R 只用了照片 1（照片 3 只有 11 点 < 12），各档的照片集合不同。
14. "只读符号链接"（README L125-126）改成"可写链接，指向 lucida-replica-01，下游禁止经它写入"。
15. 090L 23.9（README L18 / L39）不是改进：没有现场值，22 对 63 点。
16. 覆盖率摘要（README L133-134）补上围栏，并补护罩双峰。
17. 导出 manifest：去掉陈旧的 Pi3X `privacy` 字符串，`created_at_utc` / `script_sha256` 标成继承值。

### 许可与交付

- **权重**：都是宽松许可，无 NC，无 HF gate。
  - DA3-BASE@f4a6c9b：Apache-2.0，按 revision + sha256 锁定。
  - MoGe-3@184008f：MIT，只锁了 revision，没有 sha256。
  - DINOv2 ViT-L/14：Apache-2.0；同仓库现在也有 FAIR-NC 模型，只镜像这一个文件。
  - RoMa outdoor：MIT，**这是推断**（来自 Parskatt/storage 仓库的 MIT LICENSE）；RoMa README 的许可声明只覆盖代码。MegaDepth 训练数据来源未审计。
- **二进制**：按实际运行的样子，**不是**全宽松。
  - pycolmap 4.2.1 轮子：GPL-2.0+（SPQR、CHOLMOD Supernodal）。还静态打包了 OpenSSL、libcurl 8.21.0、OpenImageIO、METIS、TIFF 4.7.2、zlib，没带这些的声明；GCC 运行库受 RLE 例外保护，libquadmath 是 LGPL。
  - opencv-python-headless 4.10.0.84：Apache-2.0；动态带 FFmpeg（LGPL-2.1），自带第三方声明。
  - torch 2.5.1 的 NVIDIA CUDA / cuDNN 轮子在 NVIDIA EULA 下，未审计。
- **实际运行的镜像不能直接交付**：
  - 公平 `gpu_image` 含 plyfile（GPL-3.0+）、Pi3 代码，同一函数先跑了 Pi3X（CC BY-NC）。
  - `vr_image` 含 vggt（VGGT License），即使 vggt=False。
  - moge3_app 的 MoGe / torch 没锁版本。
  - 离线镜像验证跑的是 DA3-LARGE-1.1，不是 DA3-BASE。
- **romatch**：`utils.py` 里的 `estimate_pose` 系谱经过 LGPL-2.1 的 DenseMatching 到 Magic Leap 代码；不在运行路径上。vendoring 时删掉 `estimate_pose` 和 `estimate_pose_uncalibrated`。
- **on-prem 要镜像的东西**（sha256 在镜像时记录，这里没算）：
  - DA3-BASE@f4a6c9b：`config.json` 5e34115e…、`model.safetensors` e01067dc…；或 MoGe-3@184008f `model.pt`。
  - `roma_outdoor.pth`。
  - `dinov2_vitl14_pretrain.pth`。
  - 用 `--require-hashes` 锁定的轮子：pycolmap `7627f0ba…`、opencv `377d08a7…`。
  - 代码锁定：romatch@77f8d68、Depth-Anything-3@3d835ec、MoGe@74fbce0 + utils3d-moge@62f09d5 / pipeline@1c51139 / flex-gemm@b2fadb2。
  - 去掉 vggt、plyfile、Pi3。
- **pycolmap 三个选项**：(a) 按 GPL 交付并提供源码；(b) 用 vcpkg `ceres[lapack,schur]`（不带 suitesparse）重编 COLMAP / pycolmap 4.2.1；(c) 换成 BSD 的 BA 求解器。(b) 和 (c) 都没试过，换了之后数字要重核。
- 不是法律意见。

---

## English

### Verdict: **yes, with fixes first**

- The geometry holds up: the numbers reproduce, the evaluator is unchanged, the frozen inputs are identical, no field value reaches any input, no geometric filter was relaxed, the exported arrays are the scored arrays, and the original run is untouched.
- The new report must not copy the README's wording, though. There are six major problems:
  - The headline rests on two coupled changes, not one.
  - "Beats Pi3X" comes from the route, not from leaving Pi3X.
  - The pycolmap wheel **does** contain GPL-2.0+ code.
  - The completion A/B assemble stage breaks on this export.
  - compare's default world is wrong.
  - The SAM 3D point maps are mostly empty for 4 objects.
- Each one can be fixed or stated in the report. None overturns "licence-clean weights plus this route pass the gate on the 4 field values".

### Claims × status × evidence

Status: confirmed / caveat / refuted / unconfirmed.

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | `results.json` rebuilds from the per-run files with `compile.py` | confirmed | The rebuild differs in one line only, the `sources.clean` path (NOTE moved); no duplicate keys. `scratchpad/verify-numbers/rerun.out` |
| 2 | Every number in the verdict and every-candidate tables matches `results.json` | confirmed | MVS DA3 start: 3.304 / 1.513 %; 24.024 / 23.611 / 24.980 / 20.994; MAE 0.5968, max 0.994; camera range 1.676. Focals 375.7 / 376.2 / 372.1 · 381.2 / 374.9 |
| 3 | A local re-run of the unchanged `analyse_one` reproduces them | confirmed | Pi3X published and MVS both within 0.0084 cm (library versions differ). `scratchpad/fairness-lens/run_local.py` |
| 4 | The Pi3X reference = the fair A/B's 1.4 / 2.7 | confirmed, caveat | Diff 0.0 across 6 configs (MAE 1.44052, max 2.67258). But this is Pi3X's published geometry under the fair evaluator. The numbers as published are 1.052 / 2.153 (fair README L113). The MVS beats both |
| 5 | The evaluator is the same as the fair A/B's | confirmed | `analyse` / `analyse_one` untouched since the fair scoring. Only `freeze` was added (L112-123, L303-306). `polygon_mask` is byte-identical to 16c4a03 |
| 6 | Every backbone sees the same frozen padded 518 frames | confirmed | Canonical sha256 equal across manifests, da3fair-data, 10 bbab geometries, clean-geom and the export. The RoMa dense-warp inputs are proven only by the code path (the npz files store no input hash): **unconfirmed** |
| 7 | Field values are never inputs or tuning targets | confirmed | Grep of the route code finds no FIELD_CM / PIN / object ids. Field values appear on the scoring side only |
| 8 | No geometric filter relaxed | confirmed | FB 1 px, angle 2°, reprojection 1 px and both depths > 0 are recorded in all 10 manifests. `mvs_route.py:27` overrides CERT only |
| 9 | "Only one change" (CERT 0.5 → 0.05) | **refuted** | `mvs_route.py:46-47` also sets `conf` = 1 on kept pixels. That switches off the harness's conf ≥ 0.1 rule for pixels with RoMa certainty in [0.05, 0.1): 6.3–7.1 % of kept pixels in 090 and 3.2–3.7 % in 030. Rebuilt with conf = certainty: 24.06 / 23.61 / 25.61 / 20.99, **MAE 0.77, max 1.61**, still a pass. `scratchpad/fairness-lens/res-*-confcert.json` |
| 10 | 0.05 was fixed before any new number | confirmed, caveat | Timeline (UTC): written 20:05:01, scored 20:07:32, sensitivity argument added 20:08:07. The value is RoMa's default `sample_thresh` and was not swept. But the CERT 0.5 per-object scores had been read at 20:03:23. This is the 4th attempt on the same 4 field values. In RoMa, `sample_thresh` is a sampling saturation, not a validity cut on the dense warp |
| 11 | "Certainty, not geometry, removed the pixels" | caveat | Holds for 090 photo 3 ← 1 (medians 0.14 / 0.14 / 0.14). Does not hold for 030L photo 2: FB < 1 px removes 42 of 70 (FB median 3.11 px), and the 28 kept at CERT 0.05 are exactly the FB-limited count. The diagnostic output is not saved in the note |
| 12 | Band points 9→28, 15→47, 45→67, 0→46 | confirmed | The surfacePoints fields match; the MoGe start gives the same |
| 13 | Every threshold 0.05–0.25 passes | confirmed, caveat | All three pass in the primary config. At CERT 0.25, 090R photo 3 has only 11 points (MIN_PTS 12), so photo 1 alone is used, and 090L is None. Lowest-plane floor p95 is non-monotonic: 2.69 (fail) / 2.45 / 1.22 |
| 14 | Start-independent, "values within 0.1 cm" | caveat | Holds for maxInlier / lsq / vertical only: max diff 0.092 (090R) / 0.098 / 0.081. Under the lowest-plane rule the fence differs by 0.63 (19.31 vs 19.94), MAE 0.54 vs 0.36 |
| 15 | All floor rules × vertical pass, "MAE 0.4–0.7" | **refuted** (range) | Actual range 0.33–0.70: MoGe start, lowest-plane × vertical-plane = 0.329 (`rows[65]`). DA3 start 0.468–0.699. Verdict unchanged |
| 16 | "Beats Pi3X" | caveat | The gain comes from the route (BA-f + MVS), not from leaving Pi3X. Pi3X + BA-f gives 0.78 / 1.75. The same CERT 0.05 MVS from Pi3X BA-f cameras gives 23.97 / 23.54 / 24.91 / 20.97, MAE 0.59, max 0.97: a tie. Neither row is in the bbab ROWS. `scratchpad/fairness-lens/res-*-pi3x-start.json` |
| 17 | The CERT 0.5 re-run = the clean A/B MVS (< 1e-6 cm) | confirmed | pts3d / valid / K / c2w are byte-identical; only conf differs. Largest diff 4.33e-8 cm |
| 18 | Each run's own e-stop scale; 3.30 / 1.51 % | confirmed | 090 is 3.2371 m per unit; the worst feature is the photo 1 red lip at −3.30 %, a 0.7 pp margin. The 030 scale has thin support: 108 of 204 e-stop samples have content (DA3: 204 of 204) |
| 19 | Gate MAE ≤ 1.56 = 1.44 + 0.12 | confirmed | 1.44052 + 0.11967 = 1.5602. A draft said 1.6; it became 1.56 after scoring (stricter, so it does not favour the candidate) |
| 20 | No per-object special-casing | confirmed | No object ids in the route code; `gc.mvs` applies the same tests to every content pixel |
| 21 | Spend ≈ $0.32 | confirmed, caveat | 16 analysis jobs $0.2082 + ALIKED $0.1131 = $0.3212. The VGGT access "< $0.001" has no ledger: **unconfirmed** |
| 22 | The export passes the contract; arrays = scored arrays | confirmed | `check_geometry` passes. 5 arrays × 3 frames are sha-equal to `bbab-geom/090-mvs-da3-base-padded`. 0 points behind the camera; c2w det 1 |
| 23 | 0.14–0.15 px reprojection shows accuracy | caveat | It re-measures the method's own built-in filter (REPROJ_PX 1.0); it is not independent evidence |
| 24 | Principal point / focal | caveat (info) | Principal point 259.0 (inherited from DA3), not 258.5: 0.5 px, about 0.08°, negligible at cm scale. Focal 372–376 px vs Pi3X 378.6–382.5 px |
| 25 | floor.json scale = the gate value | confirmed | `estopNativeToMeters` 3.23714, deviation 0.033045. Floor points 35 165 (= 18 111 + 17 054) vs Pi3X 35 487. `inlier_points` = null (Pi3X 35 430) |
| 26 | Masks / rgba are "read-only symlinks" | **refuted** | The targets are -rw-r--r--, owned by adam. Writing to RUN/input or evidence/objects/* modifies lucida-replica-01 |
| 27 | Original run untouched | confirmed | 82 links, 21 geometry files, objects.json, floor.json and 125 object-view hashes all match. Nothing in lucida was written on Oct 6 |
| 28 | Ready for the downstream completion A/B | **refuted** (as is) | `completion_ab.py:390-391` tars the links without dereference, and `extractall(filter='data')` at `:252` raises AbsoluteLinkError (reproduced locally). The export has no `generation/`, so the RecGen reference arm cannot be assembled. The default boxes file in `compare.py:30-46` is in the Pi3X world (1.28601, different floor normal). floor.json's key is `estopNativeToMeters`, so passing it as CMP_SCALE gives a KeyError. `--stage sam3d` is unaffected |
| 29 | Coverage is 21–97 % of Pi3X | confirmed, omission | The numbers are right but the fences are missing: left_fence 0.31 / 0.61, right_fence 0.65 / 0.79. Whole-frame content coverage is 57 / 49 / 63 % (Pi3X 96 / 95 / 96 %). Depth samples per view 79–1 572 (Pi3X 210–2 414) |
| 30 | SAM 3D can use these point maps | at risk, **unconfirmed** | Mask coverage on the photo SAM 3D actually uses (in brackets: largest hole as a share of the mask): left_light_curtain f1 0.21 (0.78), right_post f2 0.30 (0.69), left_fence f2 0.31 (0.52), left_post f1 0.35 (0.65). Pi3X: 0.93–1.00. Whether pose and scale survive needs a GPU run |
| 31 | Guard point map | caveat | Bimodal: 8.9 / 23.9 / 22.3 % of points lie > 20 % from the median depth, vs Pi3X 0.2 / 0.6 / 0.0 %. Which geometry is right: **unconfirmed** |
| 32 | 090L 23.9 (Pi3X 19.9) | caveat | No field value; 22 vs 63 band points, so thinner support. Not an improvement |
| 33 | Export manifest provenance | caveat | `privacy` still mentions Pi3X/SAM2.1. `created_at_utc` and `script_sha256` are lucida's. The floor-mask segmenter version is not recorded |

### Must-fix list (by severity)

**Major (changes the headline or blocks downstream)**

1. **Two coupled changes** (README L19 / L40 / L142-143). State both: CERT 0.5 → 0.05, and conf := 1 on kept pixels (the second switches off conf ≥ 0.1 for 3–7 % of pixels). Report conf = certainty, **0.77 / 1.61** (030R 25.6 cm), next to 0.60 / 0.99. Both pass.
2. **pycolmap contains GPL code.** Replace the "may / inferred" wording in README L29 / L50 / L157-158 and `licences.md` L58 / L160. Also fix the licence cell in the verdict table (README L78) and the `licence` strings in `results.json`. The PyPI wheel `pycolmap-4.2.1-cp311-cp311-manylinux_2_28_x86_64.whl` (sha256 `7627f0ba…dd90f`) has a `_core.so` that statically links SuiteSparseQR and CHOLMOD Supernodal / MatrixOps, all GPL-2.0-or-later. The clean A/B README L20 / L61 / L208 ("BSD-3, licence-clean") needs the same fix. Shipping needs either GPL compliance or a SuiteSparse-free rebuild (untested; the numbers would need a re-check).
3. **Completion assemble breaks.** In `completion_ab.py:390-391`, either use `tarfile.open(..., dereference=True)` or put real copies of the masks in the export. That directory belongs to another agent; this verification did not touch it.
4. **Do not mix worlds.** With AB_RUN=export, set CMP_BOXES=none and use a CMP_SCALE file whose top-level key is `nativeToMeters` = 3.2371. Either regenerate the RecGen arm or label it unavailable; lucida's `generation/` is in the Pi3X world.
5. **SAM 3D at-risk objects.** List per-object point-map coverage in the report: left light curtain 21 %, right post 30 %, left fence 31 %, left post 35 %, vs Pi3X 93–100 %. Do not present those completions as equivalent to Pi3X's.
6. **RoMa / DINOv2 weights are unpinned.** `roma_outdoor.pth` (445 647 516 B, github.com/Parskatt/storage) and `dinov2_vitl14_pretrain.pth` (1 217 586 395 B, dl.fbaipublicfiles.com) were downloaded at run time with no hash check, and the bytes used were not recorded. Before any on-prem claim: mirror them, record sha256, and pass `weights` / `dinov2_weights` explicitly.

**Minor (wording / completeness)**

7. Change "beats Pi3X" (README L12 / L33) to: ties Pi3X when Pi3X gets the same route, and beats the published Pi3X pipeline. Add two control rows: Pi3X + BA-f 0.78 / 1.75, and the Pi3X-start CERT 0.05 MVS 0.59 / 0.97.
8. Label 1.4 / 2.7 (README L14 / L35) as "Pi3X published geometry under the fair evaluator". The numbers as published are 1.05 / 2.15.
9. Add one line to Limits: CERT 0.05 is the 4th attempt on the same 4 field values, and the change was made after seeing incomplete objects at CERT 0.5. The value itself is RoMa's default (written 20:05:01Z, scored 20:07:32Z).
10. Limit "values within 0.1 cm" (README L16 / L37) to maxInlier / lsq / vertical. Under the lowest-plane rule the fence differs by 0.63 cm, MAE 0.54 vs 0.36.
11. Change "MAE 0.4–0.7" (README L17 / L38) to 0.3–0.7.
12. Limit "certainty, not geometry" (README L20 / L41) to 090 photo 3 ← 1; in 030L, FB removes 60 %. Save the `band_diag` output in the note.
13. Sensitivity (README L22 / L43): note that at CERT 0.25, 090R uses photo 1 only (photo 3 has 11 points < 12), so the photo sets differ between thresholds.
14. Change "read-only symlinks" (README L125-126) to "writable links into lucida-replica-01; downstream must never write through them".
15. 090L 23.9 (README L18 / L39) is not an improvement: no field value, and 22 vs 63 points.
16. Add the fences and the bimodal guard to the coverage summary (README L133-134).
17. Export manifest: drop the stale Pi3X `privacy` string, and label `created_at_utc` / `script_sha256` as inherited.

### Licence and shipping

- **Weights**: all permissive, none NC, none HF-gated.
  - DA3-BASE@f4a6c9b: Apache-2.0, pinned by revision + sha256.
  - MoGe-3@184008f: MIT, pinned by revision only, no sha256.
  - DINOv2 ViT-L/14: Apache-2.0. The same repo now also hosts FAIR-NC models, so mirror this one file only.
  - RoMa outdoor: MIT **by inference** (from the MIT LICENSE of the Parskatt/storage repo); the RoMa README's licence line covers code only. MegaDepth training-data provenance is unaudited.
- **Binaries**: **not** all permissive as run.
  - pycolmap 4.2.1 wheel: GPL-2.0+ (SPQR, CHOLMOD Supernodal). It also statically bundles OpenSSL, libcurl 8.21.0, OpenImageIO, METIS, TIFF 4.7.2 and zlib with no notices shipped. The GCC runtimes fall under the Runtime Library Exception; libquadmath is LGPL.
  - opencv-python-headless 4.10.0.84: Apache-2.0, with dynamic FFmpeg (LGPL-2.1); its third-party notices are included.
  - torch 2.5.1's NVIDIA CUDA / cuDNN wheels are under the NVIDIA EULA, not audited.
- **The images as run are not shippable**:
  - The fair `gpu_image` holds plyfile (GPL-3.0+) and Pi3 code, and the same function ran Pi3X (CC BY-NC) first.
  - `vr_image` installs vggt (VGGT License) even with vggt=False.
  - moge3_app leaves MoGe / torch unpinned.
  - The offline image proof ran DA3-LARGE-1.1, not DA3-BASE.
- **romatch**: the lineage of `estimate_pose` in `utils.py` runs through LGPL-2.1 DenseMatching to Magic Leap code. It is off the runtime path; delete `estimate_pose` and `estimate_pose_uncalibrated` when vendoring.
- **Mirror for on-prem** (record sha256 at mirror time; not computed here):
  - DA3-BASE@f4a6c9b: `config.json` 5e34115e…, `model.safetensors` e01067dc…; or MoGe-3@184008f `model.pt`.
  - `roma_outdoor.pth`.
  - `dinov2_vitl14_pretrain.pth`.
  - Wheels pinned with `--require-hashes`: pycolmap `7627f0ba…`, opencv `377d08a7…`.
  - Code pins: romatch@77f8d68, Depth-Anything-3@3d835ec, MoGe@74fbce0 + utils3d-moge@62f09d5 / pipeline@1c51139 / flex-gemm@b2fadb2.
  - Drop vggt, plyfile and Pi3.
- **pycolmap options**: (a) ship under GPL with a source offer; (b) rebuild COLMAP / pycolmap 4.2.1 with vcpkg `ceres[lapack,schur]` (no suitesparse); (c) swap in a BSD BA solver. (b) and (c) are untested, and the numbers need a re-check after either.
- Not legal advice.

---

## 地址 / Paths to check

S = `/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad`

- 本笔记 / this note: `/Users/adam/Desktop/panoptes-public/research-notes/geometry-backbone-ab-2026-10-06/` (`README.md`, `results.json`, `mvs_route.py` L27 / L46-47, `compile.py`, `band_diag.py`, `backbones.py`, `backbone_ab_modal.py` L184 / L202)
- 公平 harness / fair harness: `/Users/adam/Desktop/panoptes-public/research-notes/geometry-licence-ab-fair-2026-10-05/` (`fair_ab_modal.py` L42-52 / L164-234, `fair_ab.py` L49-51, `README.md` L94 / L113)
- clean A/B: `/Users/adam/Desktop/panoptes-public/research-notes/licence-clean-stack-2026-10-06/geometry/README.md` (L20 / L61 / L208); `/Users/adam/Desktop/panoptes-public/research-notes/licence-clean-stack-2026-10-06/licences.md` (L58 / L160)
- 路线代码 / route code: `/Users/adam/.codex/worktrees/panoptes-workcell-photo-speed/modal_apps/geometry_clean_ab.py` (L61-68, L126-127, L172-173, L341, L358-385)
- 下游 / downstream: `/Users/adam/Desktop/panoptes-public/research-notes/completion-licence-ab-2026-10-05/completion_ab.py` (L188-192, L216-234, L252, L390-394); `/Users/adam/Desktop/panoptes-public/research-notes/completion-ab-090-2026-10-06/compare.py` (L30-46)
- 导出 / export: `S/checks/bbab-export-090-mvs-da3-base/`; 打分几何 / scored geometry `S/checks/bbab-geom/090-mvs-da3-base-padded/`
- 原 run / original run: `/Users/adam/Desktop/panoptes-public/panoptes-serving/outputs/candidate-evaluation/lucida-replica-01/`
- 核验草稿 / verifier scratch:
  - `S/verify-numbers/`: rerun.py, rerun.out, band_diag.out, results.json
  - `S/fairness-lens/`: run_local.py, mvs_variants.py, res-*-confcert.json, res-*-pi3x-start.json, author_timeline.txt
  - `S/verify-export090/`: contract.py, cover.py, integrity.py, sam3d_cov.py, tarprobe
  - The licence verifier streamed the wheel in memory and left no files.
