# 几何主干 A/B：同一输入、同一输出、同一评判 / Geometry backbone A/B on one contract (2026-10-06)

目标：给 Panoptes 工位照片管线找一个许可干净、能替换 Pi3X（CC BY-NC）的几何主干。用户原话：「我要有各种 ab test，最好的 pipeline 每个地方的 input output 都是一致的然后 evaluate on it directly」。
Goal: a licence-clean replacement for Pi3X (CC BY-NC) as the geometry backbone. Every backbone gets the same input, must emit the same output, and is scored by the same evaluator.

本笔记建在两份已有结果上，没有重做：`geometry-licence-ab-fair-2026-10-05`（公平 harness，Pi3X / DA3-L / DA3-BASE）和 `licence-clean-stack-2026-10-06/geometry`（RoMa + pycolmap BA、MoGe-3、map-anything、MVS、补洞；VGGT 403）。
This builds on two existing results without redoing them: the fair harness (Pi3X / DA3-L / DA3-BASE) and the licence-clean geometry A/B (RoMa + pycolmap BA, MoGe-3, map-anything, MVS, fill; VGGT 403).

## 更新 2026-10-06 / Update 2026-10-06

依据：`VERIFY.md`（四个只读核验员）；GPL-free BA（`ba_scipy.py`、`ba_scipy_modal.py`、`ba_scipy_results.json`，独立核验在 `scratchpad/verify-scipyba/`）；权重镜像（Modal volume `panoptes-geometry-weights`，`scripts/onprem/fetch_weights_geometry.py`）。本节之后是原记录（15:20），没有改写；标 〔U#〕 的句子以本节为准。
Sources: `VERIFY.md`, the GPL-free BA run and its independent check, the weight mirror. Everything after this section is the original record (15:20), not rewritten; sentences marked 〔U#〕 are superseded here.

### 中文

**交付判定**
- **组件层面：可以。** 路线上已经没有 GPL（SuiteSparse / CHOLMOD / SPQR / COLMAP / Ceres）或 NC 部分。
  - BA：pycolmap 换成 `ba_scipy.bundle_adjust`（numpy Levenberg-Marquardt + 相机 / 点 Schur 补，只用 numpy / scipy / OpenCV）。4 个配置（2 个起始 × conf 1 / 置信度）全过，现场值与 pycolmap 差 ≤ 0.0004 cm。见下"GPL-free BA"。
  - 权重：RoMa outdoor、DINOv2 ViT-L/14、DA3-BASE、MoGe-3 已镜像并按 sha256 锁定。全是宽松许可，无 NC；RoMa outdoor 的 MIT 是推断。见下"权重锁定"。
- **打包层面：还不行。** 代码接线、来源字符串、收敛标记、轮子哈希没做完，见"仍开放"。

**更正（原说法 → 现在的说法）**
- **U1** pycolmap 许可（结论、判定表许可格、局限）："轮子可能带 SuiteSparse GPL 部分" → **确实带**。PyPI `pycolmap-4.2.1-cp311-cp311-manylinux_2_28_x86_64.whl`（sha256 `7627f0ba…dd90f`）的 `_core.so` 静态链接 SuiteSparseQR 和 CHOLMOD Supernodal / MatrixOps，都是 GPL-2.0-or-later。原 MVS 行是用 pycolmap 算的，记录不变；交付路线改用 `ba_scipy`（通过，见 BA 表）。`results.json` 的 `licence` 字符串是原记录，没改。
- **U2** "只改了一处"（结论、方法"表面"一条）→ **两处联动**：CERT 0.5 → 0.05，加上保留像素 conf := 1（`mvs_route.py:46-47`）。后者关掉 harness 的 conf ≥ 0.1，影响置信度在 [0.05, 0.1) 的像素：090 占保留像素 6.3–7.1 %，030 占 3.2–3.7 %。conf = 置信度：24.06 / 23.61 / 25.61 / 20.99，**MAE 0.77，最大 1.61**（VERIFY 本地重建；Modal 重算 0.765 / 1.615）。两种都通过。
- **U3** "超过 Pi3X"（结论首句）→ **同一路线上与 Pi3X 打平，优于 Pi3X 已发布管线。** 收益来自路线（BA-f + MVS），不来自换掉 Pi3X。Pi3X BA-f 相机 + 同样的 CERT 0.05 MVS：23.97 / 23.54 / 24.91 / 20.97，**MAE 0.59，最大 0.97**。Pi3X + BA-f（无 MVS）：0.78 / 1.75。
- **U4** "Pi3X 已发布 1.4 / 2.7"（结论、判定表表头）→ **Pi3X 已发布几何 × 公平评判器**：1.44 / 2.67。按已发布数字本身是 1.05 / 2.15。MVS 两个都赢。
- **U5** "MAE 0.4–0.7"（结论）→ **0.3–0.7**。MoGe 起始最低平面 × 竖直面 = 0.329；DA3 起始 0.468–0.699。
- **U6** "各值差 ≤ 0.1 cm"（结论）→ 只在最大内点和 lsq 地面（及竖直面）成立：最大差 0.092 / 0.098 / 0.081 cm。最低平面规则下围栏差 0.63 cm（19.31 对 19.94），MAE 0.54 对 0.36。
- **U7** "删像素的是置信度，不是几何"（结论）→ 只对 090 照片 3 ← 1 成立。030L 照片 2：FB < 1 px 删掉 70 个里的 42 个（60 %，FB 中位 3.11 px）。诊断输出：`band_diag.out`。
- **U8** 敏感性（结论）：CERT 0.25 时 090R 照片 3 只有 11 点（< MIN_PTS 12），只用照片 1；090L 为 None。各档的照片集合不同。
- **U9** 090L 23.9（结论）：没有现场值，带内 22 对 63 点。不是改进。
- **U10** "只读符号链接"（导出）→ **可写链接**，指向 lucida-replica-01 里 -rw-r--r-- 的文件；下游禁止经它写入。新的 scipyba 导出已全部换成实拷（090 82 个、030 49 个）。
- **U11** 覆盖率（导出注意）补：left_fence 0.31 / 0.61、right_fence 0.65 / 0.79（对 Pi3X 之比）；全帧内容覆盖 57 / 49 / 63 %（Pi3X 96 / 95 / 96 %）；护罩点图双峰（离中位深度 > 20 % 的点 8.9 / 23.9 / 22.3 %，Pi3X 0.2 / 0.6 / 0.0 %；哪个对未确认）。
- 第 4 次尝试的说明已加进"局限"。

**仍开放**
1. **接线**：`modal_apps/geometry_clean_ab.py` 的 `refine` 仍用 pycolmap，`roma_outdoor()`（L127 / L173）仍不传 `weights` / `dinov2_weights`。`ba_scipy.py` 还只在本笔记里，不在 on-prem kit；`docker/geometry-requirements.txt` 已含 scipy 1.14.1 / OpenCV 4.10.0.84，但 numpy 是 1.26.4，BA 是在 numpy 2.2.6（Modal）和 2.5.1（本地复跑）上验证的。交付前：放进 kit、接上 refine、显式传镜像权重，离线端到端重跑一次并对上 BA 表。
2. **来源字符串错**：`ba_scipy.refine` 的 `rep['ba_backend']`、`ba_scipy_modal` 写进 `*-scipyba-f/candidate_manifest.json` 的 `ba`、两份 scipyba 导出 `manifest.json` 的 `exportNote.ba` 都写 `scipy.optimize.least_squares`；实际跑的是 numpy Schur LM。
3. **收敛不可见**：090 的 4 遍（2 起始 × 2 遍）都停在 MAX_ITER 500（梯度 1e-7 – 2e-7，没到 GTOL 1e-8），`usable` 写死 True。代价与 Ceres 7 位一致，所以这里结果对，但阶段要能报未收敛。"放宽到 1e-6 会更快"没测。
4. **自检缺 cheirality**：删掉"相机后方 = 0 残差"规则的突变体能过自检（8 个突变体杀掉 7 个）；只有真实数据的一致性挡住它。
5. **打包**：轮子哈希没锁（`--require-hashes`）；torch cu124 的 NVIDIA 轮子在 NVIDIA EULA 下，未审计；numpy / scipy 轮子通常带 libgfortran（GPL-3 + GCC 运行库例外）和 libquadmath（LGPL），Debian 基础镜像有 GPL 用户态，要附声明（没重扫）；vendoring romatch 时删 `estimate_pose*`；跑过的 `vr_image`（含 vggt）和公平 `gpu_image`（plyfile、Pi3）不能交付，要按新 requirements 重建。
6. **不变的风险**：SAM 3D 点图覆盖（左光幕 21 %、右立柱 30 %、左围栏 31 %、左立柱 35 %，Pi3X 93–100 %）；030 导出 right_fence 0 点、right_post 34 点（只有两张照片，与 pycolmap 路线相同）；RoMa 的 MegaDepth 训练数据来源未审计。

### English

**Shipping verdict**
- **Components: yes.** No GPL part (SuiteSparse / CHOLMOD / SPQR / COLMAP / Ceres) and no NC part is left on the route.
  - BA: pycolmap is replaced by `ba_scipy.bundle_adjust`, a numpy Levenberg-Marquardt with the camera/point Schur complement, using only numpy / scipy / OpenCV. All 4 configurations (2 starts × conf 1 / certainty) pass; field values match pycolmap to ≤ 0.0004 cm. See "GPL-free BA".
  - Weights: RoMa outdoor, DINOv2 ViT-L/14, DA3-BASE and MoGe-3 are mirrored and pinned by sha256. All permissive, none NC; RoMa outdoor's MIT is by inference. See "Weight pins".
- **Package: not yet.** Code wiring, provenance strings, the convergence flag and wheel hashes are unfinished; see "Still open".

**Corrections (old wording → current wording)**
- **U1** pycolmap licence (Verdict, the verdict-table licence cell, Limits): "the wheel may carry SuiteSparse GPL parts" → **it does**. The `_core.so` of the PyPI `pycolmap-4.2.1-cp311-cp311-manylinux_2_28_x86_64.whl` (sha256 `7627f0ba…dd90f`) statically links SuiteSparseQR and CHOLMOD Supernodal / MatrixOps, all GPL-2.0-or-later. The original MVS rows were computed with pycolmap and stay as recorded; the shipped route uses `ba_scipy` (passes, see the BA table). The `licence` strings in `results.json` are the original record and are unchanged.
- **U2** "the one change" (Verdict, the Method surface line) → **two coupled changes**: CERT 0.5 → 0.05, and conf := 1 on kept pixels (`mvs_route.py:46-47`). The second switches off the harness's conf ≥ 0.1 for pixels with certainty in [0.05, 0.1): 6.3–7.1 % of kept pixels in 090, 3.2–3.7 % in 030. With conf = certainty: 24.06 / 23.61 / 25.61 / 20.99, **MAE 0.77, max 1.61** (VERIFY local rebuild; Modal re-score 0.765 / 1.615). Both pass.
- **U3** "beats Pi3X" (Verdict headline) → **matches Pi3X on the same route, and beats the published Pi3X pipeline.** The gain comes from the route (BA-f + MVS), not from leaving Pi3X. Pi3X BA-f cameras + the same CERT 0.05 MVS: 23.97 / 23.54 / 24.91 / 20.97, **MAE 0.59, max 0.97**. Pi3X + BA-f without MVS: 0.78 / 1.75.
- **U4** "Pi3X's published 1.4 / 2.7" (Verdict, the verdict-table header) → **Pi3X published geometry under the fair evaluator**: 1.44 / 2.67. The numbers as published are 1.05 / 2.15. The MVS beats both.
- **U5** "MAE 0.4–0.7" (Verdict) → **0.3–0.7**. MoGe start, lowest-plane × vertical-plane = 0.329; DA3 start 0.468–0.699.
- **U6** "values within 0.1 cm" (Verdict) → holds for the max-inlier and lsq floors (and the vertical-plane surface) only: max diff 0.092 / 0.098 / 0.081 cm. Under the lowest-plane rule the fence differs by 0.63 cm (19.31 vs 19.94), MAE 0.54 vs 0.36.
- **U7** "certainty, not geometry, removed the pixels" (Verdict) → holds for 090 photo 3 ← 1 only. In 030L photo 2, FB < 1 px removes 42 of 70 (60 %, FB median 3.11 px). Diagnostic output: `band_diag.out`.
- **U8** Sensitivity (Verdict): at CERT 0.25, 090R photo 3 has 11 points (< MIN_PTS 12), so photo 1 alone is used, and 090L is None. The photo sets differ between thresholds.
- **U9** 090L 23.9 (Verdict): no field value, 22 vs 63 band points. Not an improvement.
- **U10** "read-only symlinks" (Export) → **writable links** to -rw-r--r-- files in lucida-replica-01; downstream must never write through them. The new scipyba exports hold real copies only (82 links in 090, 49 in 030).
- **U11** Coverage (Export caveats), add: left_fence 0.31 / 0.61, right_fence 0.65 / 0.79 (ratio to Pi3X); whole-frame content coverage 57 / 49 / 63 % (Pi3X 96 / 95 / 96 %); the guard point map is bimodal (8.9 / 23.9 / 22.3 % of points > 20 % from the median depth, Pi3X 0.2 / 0.6 / 0.0 %; which is right is unconfirmed).
- The 4th-attempt note is now in Limits.

**Still open**
1. **Wiring.** `modal_apps/geometry_clean_ab.py` `refine` still calls pycolmap, and `roma_outdoor()` (L127 / L173) still passes no `weights` / `dinov2_weights`. `ba_scipy.py` lives only in this note, not in the on-prem kit; `docker/geometry-requirements.txt` has scipy 1.14.1 / OpenCV 4.10.0.84 but numpy 1.26.4, while the BA was validated on numpy 2.2.6 (Modal) and 2.5.1 (local rerun). Before shipping: add it to the kit, wire it into refine, pass the mirrored weights explicitly, and rerun once offline end to end against the BA table.
2. **Wrong provenance strings.** `ba_scipy.refine` `rep['ba_backend']`, the `ba` field that `ba_scipy_modal` writes into `*-scipyba-f/candidate_manifest.json`, and `exportNote.ba` in both scipyba exports' `manifest.json` say `scipy.optimize.least_squares`; the solver that ran is the numpy Schur LM.
3. **Non-convergence is invisible.** All 4 passes on 090 (2 starts × 2 passes) stop at MAX_ITER 500 (gradient 1e-7 – 2e-7, GTOL 1e-8 not reached), and `usable` is hard-coded True. Costs match Ceres to 7 digits, so the result is right here, but the stage must be able to report non-convergence. The "1e-6 tolerance would be faster" claim is untested.
4. **No cheirality self-test.** The mutant without "behind camera = zero residual" passes the self-test (7 of 8 mutants killed); only the real-data parity guards it.
5. **Packaging.** Wheel hashes are not pinned (`--require-hashes`). torch cu124's NVIDIA wheels are under the NVIDIA EULA, not audited. numpy / scipy wheels normally bundle libgfortran (GPL-3 with the GCC Runtime Library Exception) and libquadmath (LGPL), and the Debian base carries GPL userland: ship the notices (not re-scanned). Delete `estimate_pose*` when vendoring romatch. The images as run (`vr_image` with vggt, the fair `gpu_image` with plyfile and Pi3) are not shippable; rebuild from the new requirements.
6. **Unchanged risks.** SAM 3D point-map coverage (left light curtain 21 %, right post 30 %, left fence 31 %, left post 35 %; Pi3X 93–100 %). The 030 export has right_fence 0 and right_post 34 points (two photos; same as the pycolmap route). RoMa's MegaDepth training-data provenance is unaudited.

## GPL-free BA（`ba_scipy`，2026-10-06）

**中文**
- 求解器：`ba_scipy.bundle_adjust`，numpy LM + 相机 / 点 Schur 补（Ceres 对这种 2–3 张照片问题用的同一方法）。`scipy.optimize.least_squares`（稀疏 Jacobian）只作自检里的参照：在 090 上 DA3-BASE 起始撞到 3600 s Modal 超时，MoGe 起始停在 nfev 1000 未收敛、删掉 2 509 点；030 约 30 s 收敛。
- 与 pycolmap 的设置相同：每张一个焦距、主点固定；每个观测 Cauchy 尺度 1.0；gauge = 照片 1 固定 + 照片 2 一个平移分量固定（COLMAP 4.2.1 `TWO_CAMS_FROM_WORLD` 的选法）；观测在相机后方计 0（同 COLMAP）；两遍，第一遍后删误差 > 2 px 的点。阶段接口与 `geometry_clean_ab.refine` 相同，MVS 原样使用。
- 镜像核查：pip freeze 只有 numpy 2.2.6、scipy 1.14.1、opencv-python-headless 4.10.0.84、pillow 11.0.0 + Modal 客户端；pycolmap 不能 import；扫了 874 个 .so + ldd / ldconfig / dpkg，没有 CHOLMOD / SuiteSparse / SPQR / Ceres。文本命中（cv2 的 "spqr"、"ceres"，brotli / libunistring / unicodedata 的 "ceres"）判为字体 / 标识符 / Unicode 文本，原始上下文没存，不可复查。准确说法：**无 SuiteSparse / CHOLMOD / SPQR / COLMAP / Ceres 代码**，不是"无 GPL 代码"（见仍开放 5）。
- 评判器未改（mtime 核查），打分数组来自新 BA（K / c2w 与 scipyba-f 输出相等，与 pycolmap 不等），现场值没进 BA 或路线。只打分了最终一次运行。
- 花费（Modal list 价）**$0.403**：评判 12 个 job $0.138；最终 LM 路线 $0.022；失败的 least_squares $0.240（其中超时 job $0.158，估算）；第一次 LM $0.0025；.so 探针 $0.0004。

**English**
- Solver: `ba_scipy.bundle_adjust`, a numpy LM with the camera/point Schur complement (the method Ceres uses for 2–3 photo problems). `scipy.optimize.least_squares` (sparse Jacobian) is only the self-test reference: on 090 the DA3-BASE start hit the 3600 s Modal timeout and the MoGe start stopped at nfev 1000 unconverged, dropping 2 509 points; 030 converged in about 30 s.
- Same setup as pycolmap: one focal per photo, principal point fixed; Cauchy scale 1.0 per observation; gauge = photo 1 fixed plus one translation component of photo 2 (COLMAP 4.2.1's `TWO_CAMS_FROM_WORLD` choice); an observation behind a camera counts as zero (as COLMAP); two passes, dropping points > 2 px after pass 1. Same stage interface as `geometry_clean_ab.refine`; the MVS consumes it unchanged.
- Image check: pip freeze has only numpy 2.2.6, scipy 1.14.1, opencv-python-headless 4.10.0.84, pillow 11.0.0 plus the Modal client; pycolmap is not importable; 874 .so files plus ldd / ldconfig / dpkg show no CHOLMOD / SuiteSparse / SPQR / Ceres. The text hits (cv2 "spqr" and "ceres"; "ceres" in brotli / libunistring / unicodedata) were judged font / identifier / Unicode text; the raw contexts were not saved, so this is not re-checkable. Accurate claim: **no SuiteSparse / CHOLMOD / SPQR / COLMAP / Ceres code**, not "no GPL code" (see Still open 5).
- Evaluator unchanged (mtime check); the scored arrays come from the new BA (K / c2w equal the scipyba-f output, not pycolmap's); no field value enters the BA or route. Only the final run was scored.
- Spend at Modal list price **$0.403**: analysis 12 jobs $0.138; final LM route $0.022; failed least_squares $0.240 (of which the timed-out job $0.158, estimated); first LM $0.0025; .so probe $0.0004.

主配置（补边帧 + 最大内点地面 + 点图近面）。误差 = 测量 − 现场值（cm）。通过 = MAE ≤ 1.56、每个误差 ≤ 3、两个急停门限 < 4 %、地面 p95 ≤ 2.5、现场值与 pycolmap 差 ≤ 0.3 cm。
Primary configuration. Error = measured − field (cm). Pass = MAE ≤ 1.56, every error ≤ 3, both e-stop gates < 4 %, floor p95 ≤ 2.5, field values within 0.3 cm of pycolmap.

| 起始 start | conf | 090R 罩壳 housing (24) | 030L (24) | 030R (24) | 090R 围栏 fence (20) | **MAE** | 最大 max | 急停 e-stop 090 / 030 | 地面 floor p95 090 / 030 | pycolmap MAE / max | 对 pycolmap 最大 \|Δ\| vs pycolmap | 通过 pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DA3-BASE | 1 | 24.02 (+0.02) | 23.61 (−0.39) | 24.98 (+0.98) | 20.99 (+0.99) | **0.60** | 0.99 | 3.30 / 1.51 % | 0.79 / 0.28 | 0.60 / 0.99 | 0.0000 | 是 yes |
| DA3-BASE | 置信度 certainty | 24.06 (+0.06) | 23.61 (−0.39) | 25.62 (+1.62) | 20.99 (+0.99) | **0.76** | 1.62 | 3.32 / 1.51 % | 0.79 / 0.27 | 0.76 / 1.62 | 0.0000 | 是 yes |
| MoGe-3 | 1 | 23.93 (−0.07) | 23.62 (−0.38) | 24.96 (+0.96) | 20.97 (+0.97) | **0.60** | 0.97 | 3.34 / 1.58 % | 0.81 / 0.28 | 0.60 / 0.97 | 0.0003 | 是 yes |
| MoGe-3 | 置信度 certainty | 23.97 (−0.03) | 23.62 (−0.38) | 25.60 (+1.60) | 20.97 (+0.97) | **0.75** | 1.60 | 3.34 / 1.58 % | 0.81 / 0.27 | 0.75 / 1.60 | 0.0004 | 是 yes |

- 相机高（ba_scipy）：DA3 起始 146.1 / 145.8 / 146.5 · 145.7 / 144.9 cm，MoGe 起始 146.3 / 145.9 / 146.8 · 145.5 / 144.7 cm。其他地面规则和竖直面：两种 BA 的 MAE 每格差 ≤ 0.003 cm。
  Camera heights (ba_scipy) as above. Other floor rules and the vertical-plane surface: MAE equal between the two BAs to ≤ 0.003 cm in every cell.
- pycolmap 的 conf = 置信度参照在同一 Modal 镜像里重建，与原打分的 MVS 逐字节相同（新镜像复现旧运行）。
  The pycolmap conf = certainty references were rebuilt in the same Modal image; the rebuilt pycolmap MVS is byte-equal to the originally scored one.

相机与 BA，ba_scipy 对 pycolmap（Δ = ba_scipy − pycolmap）/ Cameras and BA, ba_scipy vs pycolmap:

| 工位 / 起始 cell / start | 最大 Δ 旋转 rotation | 最大 Δ 相机中心 centre | Δ 焦距 focal (px) | 焦距 focals (px) | BA 代价 cost 第 1 → 2 遍 pass 1 → 2，ba_scipy / Ceres | 删点 dropped ba_scipy / Ceres | MVS 点差 point diff | 每遍 per pass |
|---|---|---|---|---|---|---|---|---|
| 090 DA3-BASE | 4.5e-6° | 1.7e-6 cm | ≤ 1.6e-5 | 375.674 / 376.241 / 372.070（两者 both） | 1134.818 → 1120.388 / 1134.818 → 1120.388 | 51 / 51（只核了个数 count only） | 中位 median 2e-5、p95 1.5e-4 cm | ≈ 117 s |
| 090 MoGe-3 | 0.0025° | 0.003 cm | +0.029 – +0.030 | 376.136 / 376.331 / 372.425 对 vs 376.106 / 376.302 / 372.396 | 1250.960 → 1239.983 / 1249.758 → 1238.775（Ceres 更低 lower） | 494 / 507（"Ceres 把 13 点推到相机后方"未核实 unverified） | 中位 0.04、p95 0.25 cm；孤立像素最大 max on isolated pixels 0.055 native ≈ 17.8 cm；valid 每帧差 1 px | ≈ 110 s |
| 030 DA3-BASE | ≈ 1e-9° | 5e-10 cm | 4e-9 | 381.225 / 374.942（两者 both） | 339.603 / 339.603 | 0 / 0 | valid 相同 equal；最大 max 4.8e-7 native ≈ 1.7e-4 cm（不是逐字节 not byte-equal） | 2 s |
| 030 MoGe-3 | ≈ 1e-9° | 8e-11 cm | 6e-10 | 380.987 / 374.083（两者 both） | 438.584 / 438.584 | 4 / 4 | valid 相同 equal；最大 max 1.9e-6 native | 2 s |

导出（lucida-replica-01 格式，全部实拷，0 个符号链接，`backbones.check_geometry` 通过）/ Exports (lucida-replica-01 layout, real files only, 0 symlinks, contract passed):

| 工位 cell | 路径 path（S = scratchpad） | 大小 size | 实拷文件 real files | valid 比例 fraction | 重投影中位 reproj median | 备注 note |
|---|---|---|---|---|---|---|
| 090 | `S/checks/bbab-export-090-mvs-scipyba/` | 66.9 MB | 84（82 个链接替换），与 lucida-replica-01 逐字节相同 byte-equal | 0.43 / 0.37 / 0.47 | 0.14–0.15 px | 物体点数 = pycolmap 导出 object point counts = pycolmap export |
| 030 | `S/checks/bbab-export-030-mvs-scipyba/` | 57.2 MB | 50（49 个链接替换），与 bor1-030-01 逐字节相同 byte-equal | 0.22 / 0.17 | 0.08–0.09 px | right_fence 0 点 points、right_post 34（两张照片 two photos） |

两份 `manifest.json` 的 `exportNote.ba` 写错求解器（仍开放 2）。Both `manifest.json` files name the wrong solver in `exportNote.ba` (Still open 2).

## 权重锁定 / Weight pins (2026-10-06)

**中文**
- 镜像到新 Modal volume `panoptes-geometry-weights`（≈ 3.69 GB），两次临时 `modal run`（CPU，retries=0，min_containers=0），边下边算 sha256。romatch@77f8d68 只下两个文件（`model_zoo/__init__.py:8` RoMa outdoor，`:14` 和 `encoders.py:33` DINOv2）；VGG 用 `pretrained=False`，没有第三个下载。
- 新 kit 文件（worktree `panoptes-workcell-photo-speed`，未提交）：`scripts/onprem/fetch_weights_geometry.py`（先下到临时文件，sha256 不符就删掉并报错；已有且正确的文件保留；`--verify` 离线可用；DA3 / MoGe 的 pin 直接取自 `fetch_weights_da3.py` / `fetch_weights.py`）；`docker/geometry-requirements.txt`（43 行 pin，无 pycolmap，romatch git@77f8d68 `--no-deps`，轮子哈希未锁）。
- 断网证明（同一镜像，network blocked）：`--verify`、`--self-test` 通过；从镜像文件显式传入构建 RoMa，strict load 成功（603 / 343 个键）；合成 20 px 平移，dx 中位 −0.1112，期望 −0.1111。
- 交叉核对：DA3-BASE、MoGe-3 的 sha256 = HF LFS id = 已有 pin。RoMa outdoor：GitHub 资产 digest 为 null，这是它的第一份哈希记录，大小与 GitHub 元数据一致。DINOv2：没有发布哈希，大小 = Content-Length。两个文件的大小都等于 VERIFY 记下的运行时字节数；原运行没存哈希，所以**不能证明**与打分时用的是同一份字节。
- 花费（list 价）≈ **$0.0084**（不含镜像构建和 volume 存储）。

**English**
- Mirrored into a new Modal volume `panoptes-geometry-weights` (≈ 3.69 GB) by two ephemeral `modal run`s (CPU, retries=0, min_containers=0), hashed while streaming. romatch@77f8d68 downloads exactly two files (`model_zoo/__init__.py:8` RoMa outdoor; `:14` and `encoders.py:33` DINOv2); VGG is built with `pretrained=False`, so there is no third download.
- New kit files (worktree `panoptes-workcell-photo-speed`, uncommitted): `scripts/onprem/fetch_weights_geometry.py` (downloads to a temp file, deletes it and fails on a sha256 mismatch; keeps a correct existing file; `--verify` works offline; DA3 / MoGe pins are read from `fetch_weights_da3.py` / `fetch_weights.py`), and `docker/geometry-requirements.txt` (43 pinned lines, no pycolmap, romatch git@77f8d68 with `--no-deps`, wheel hashes not pinned).
- Offline proof (same image, network blocked): `--verify` and `--self-test` pass; RoMa built from the mirrored files passed explicitly loads strictly (603 / 343 keys); a synthetic 20 px shift gives dx median −0.1112 against −0.1111 expected.
- Cross-checks: DA3-BASE and MoGe-3 sha256 = HF LFS id = the existing pins. RoMa outdoor: the GitHub asset digest is null, so this is the first hash on record; the size matches GitHub's metadata. DINOv2: no published hash; the size equals the Content-Length. Both sizes equal the run-time bytes VERIFY recorded; the original run stored no hashes, so identity with the scored bytes is **not proven**.
- Spend at list price ≈ **$0.0084** (image builds and volume storage excluded).

| 权重 weight | 来源 source（代码实际加载的 URL / as the code loads it） | 版本 revision | sha256 | 字节 bytes | 许可 licence | volume 路径 path |
|---|---|---|---|---|---|---|
| RoMa v1 outdoor | `https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth` | tag `roma` = `6fed4401e851d3c73ceaf15b885fea078a2701f1`；资产 asset 2023-05-25；无上游 digest | `c7a45c80d41ad788a63c641d1b686d7cb3f297f40097c6f4e75039889e5cc8ba` | 445,647,516 | MIT，推断 inferred：Parskatt/storage@6fed440 LICENSE 首行 "MIT License"（sha256 `73c934424d4489a3bf557902a07c693296484b3a3e4c3599003400ef15ae267c`，1,070 B，已镜像）；RoMa README 的 MIT 只覆盖代码 | `roma/outdoor/roma_outdoor.pth`（+ `LICENSE`） |
| DINOv2 ViT-L/14（RoMa 编码器 encoder） | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth` | 无版本 URL unversioned；Last-Modified 2023-04-13，ETag `cc03629d9da78b4829d02245498340b1-146`；无上游哈希 | `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428` | 1,217,586,395 | Apache-2.0（dinov2@7764ea0 LICENSE sha256 `600cc67cc4cb2f5ea317dcfc687ad1c74dc4bec8782bbe9db0afd83513b935b7`，已镜像；README L660；MODEL_CARD L32）。同一站点也有 FAIR NC 模型：只镜像这一个文件 | `roma/dinov2/dinov2_vitl14_pretrain.pth`（+ `LICENSE`） |
| DA3-BASE `model.safetensors` | `https://huggingface.co/depth-anything/DA3-BASE` | `f4a6c9b3c95e41c82048423d3493a81ec3fa810e` | `e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5`（= HF LFS id） | 541,518,028 | Apache-2.0（卡 card `license: apache-2.0`；官方 README@3d835ec L223 "Apache 2.0"） | `local/da3-base/model.safetensors` |
| DA3-BASE `config.json` | 同上 same repo | `f4a6c9b…` | `5e34115ebc17bd2d8d43033c5f72e9446ac8833fd61d3fa160b7e67e0bb5b7b5` | 1,205 | Apache-2.0 | `local/da3-base/config.json` |
| DA3-BASE `README.md`（模型卡 card） | 同上 same repo | `f4a6c9b…` | `5388563071b4361b11d6d811b354d454f26aed2adba634fcf4ee3775c8eda480` | 5,682 | 许可卡 licence card | `local/da3-base/README.md` |
| MoGe-3 `model.pt`（备选起始 alternate start） | `https://huggingface.co/Ruicheng/moge-3-vitl` | `184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be` | `9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925`（= HF LFS id） | 1,481,333,394 | MIT（卡 `license: mit`；MoGe@74fbce0 LICENSE "MIT License"） | `hf/hub/models--Ruicheng--moge-3-vitl/snapshots/184008f…/model.pt` |
| MoGe-3 `README.md`（模型卡 card） | 同上 same repo | `184008f…` | `fc81933791979233a76c29e3a44ed91a44f69555121d2160fdfc39219fb59fd4` | 51 | 许可卡 licence card | 同一 snapshot / same snapshot `README.md` |

volume 上另有 `/manifest.json`（fetch_weights 格式，2026-10-06T21:33:23Z）、`/provenance/probe-2026-10-06.json`、`/provenance/licences/`（12 份文本）。显式调用写在 `fetch_weights_geometry.py` 的 docstring：`roma_outdoor(weights=..., dinov2_weights=...)`。
Also on the volume: `/manifest.json`, `/provenance/probe-2026-10-06.json`, `/provenance/licences/` (12 texts). The explicit call is in the `fetch_weights_geometry.py` docstring.

## 结论（原记录）/ Verdict (original record)

**中文**
- **有了：许可干净的 MVS 路线在噪声内达到并超过 Pi3X。** 〔U3〕
  - 路线：任一干净主干作初值 → RoMa（MIT）匹配 + pycolmap（BSD-3）〔U1〕BA（相机 + 每张焦距）→ RoMa 稠密 warp 两视图三角化。表面完全不用学习得到的深度。
  - 4 个现场值：090 右罩壳 24.0（+0.0），030 左罩壳 23.6（−0.4），030 右罩壳 25.0（+1.0），围栏 21.0（+1.0）cm。**MAE 0.6 cm，最大 1.0 cm**；Pi3X 已发布是 1.4 / 2.7。〔U4〕
  - 急停门限通过（3.30 / 1.51 %），地面 p95 0.79 / 0.28 cm（Pi3X 0.37 / 0.97），五张相机高 144.9–146.5 cm，极差 1.7 cm（Pi3X 6.3）。
  - 不依赖起始主干：DA3-BASE（Apache）起始和 MoGe-3（MIT）起始给出相同结果（MAE 0.6 / 0.6，各值差 ≤ 0.1 cm）。〔U6〕
  - 三种地面规则、竖直面变体都通过（MAE 0.4–0.7）〔U5〕。只有"最低平面"规则在 DA3-BASE 起始时地面 p95 2.69 cm，略超 2.5 的门限；这个规则本来就不是主配置。
  - 被 QA 排除的 090 左罩壳（防撞柱挡住一半）也读到 23.9 cm（Pi3X 19.9）。〔U9〕
- **改了什么（只有一处，通用，不针对 030L）**：RoMa 置信度门限 0.5 → **0.05，即 RoMa 自己的 `sample_thresh`**（RoMa 把高于它的置信度直接设成 1，视为"确定"）。〔U2〕
  - 依据是改之前做的带内诊断（`band_diag.py`）：在罩壳近面带里，删掉像素的是置信度，不是几何。比如 090 照片 3 ← 照片 1：前后一致误差中位 0.14 px、重投影 0.14 px，但置信度中位 0.14 < 0.5，全部被删。〔U7〕
  - 几何检查一个没放松：前后一致 < 1 px、射线夹角 ≥ 2°、重投影 < 1 px、两边深度 > 0。
  - 敏感性：门限 0.1 → MAE 0.8，0.25 → 0.9，0.5 → 030L 只剩 9 个点（不完整，与 clean A/B 相同，差 < 1e-6 cm）。0.05–0.25 都通过，结论不靠调参。〔U8〕
  - 030L 带内点数 9 → 28，030R 15 → 47，090R 照片 1 45 → 67、照片 3 0 → 46。
- **补洞不再需要**：MVS 自己就完整了，所以没有再做"只在物体 mask 里补洞 + 地面用三角化点 RANSAC 平面"。如果下游 SAM 3D 需要更密的物体点图再加。
- **被挡住的 / blocked**：
  - VGGT-1B-Commercial：Modal secret `huggingface` 的账号能登录（whoami 正常），但下载 `config.json` / `model.safetensors` 都是 `GatedRepoError 403`。**要用户用这个 HF 账号去申请访问**（人工审批）。没有重试。
  - map-anything-apache "加我们的 K" 没有单独跑：clean A/B 的 BA-f 已经把相机（含焦距）修到与主干无关的值，map-anything 仍然 MAE 12.5（030 右罩壳 −46 cm），问题在稠密表面，不在 K；MVS 已通过，这一行不会改变结论。
  - 我自己写的经典路线（ALIKED + LightGlue + MoGe-3 深度）：518 帧上每对只有 217–572 个匹配，焦距扫描几乎是平的，MAE 4.4–4.6。被 RoMa（每对 10 000 个）取代，只作诊断。
- **许可**：MVS 路线的组件 RoMa（MIT，outdoor 权重）、pycolmap 4.2.1（BSD-3；轮子可能带 SuiteSparse GPL 部分，见 licences.md）〔U1〕、OpenCV（Apache-2.0）、DINOv2（Apache-2.0，在 RoMa 里），起始主干 DA3-BASE（Apache-2.0）或 MoGe-3（MIT）。不是法律意见。
- **下一步**：090 的几何已按 lucida-replica-01 的目录格式导出（见"导出"），可以直接跑下游补全 A/B（SAM 3D + 原装配）。

**English**
- **Yes: the licence-clean MVS route matches Pi3X within noise, and beats it on these four values.** 〔U3〕
  - Route: any clean backbone as the start → RoMa (MIT) matches + pycolmap (BSD-3) 〔U1〕 BA (cameras + one focal per photo) → two-view triangulation of RoMa's dense warp. No learned depth in the surface.
  - Field values: 090R housing 24.0 (+0.0), 030L 23.6 (−0.4), 030R 25.0 (+1.0), fence 21.0 (+1.0) cm. **MAE 0.6 cm, worst 1.0 cm**, against Pi3X's published 1.4 / 2.7. 〔U4〕
  - E-stop gate passed (3.30 / 1.51 %). Floor p95 0.79 / 0.28 cm (Pi3X 0.37 / 0.97). Five camera heights 144.9–146.5 cm, spread 1.7 cm (Pi3X 6.3).
  - Independent of the start: DA3-BASE (Apache) and MoGe-3 (MIT) starts give the same result (MAE 0.6 / 0.6; values within 0.1 cm). 〔U6〕
  - It passes under all three floor rules and the vertical-plane variant (MAE 0.4–0.7) 〔U5〕. The one exception is the lowest-plane rule with the DA3-BASE start: floor p95 2.69 cm, just over the 2.5 limit. That rule is not the primary configuration.
  - The 090L housing (half hidden by the bollard, not scored) reads 23.9 cm (Pi3X 19.9). 〔U9〕
- **The one change (generic, not a 030L special case): RoMa certainty threshold 0.5 → 0.05, RoMa's own `sample_thresh`.** RoMa sets any certainty above it to 1, i.e. treats it as certain. 〔U2〕
  - It follows a band diagnostic made before any new number was scored (`band_diag.py`). In the housing near-face bands, certainty removed the pixels, not geometry. For example, in 090 photo 3 ← photo 1 the median forward-backward error was 0.14 px and the median reprojection 0.14 px, but the median certainty was 0.14 < 0.5, so every pixel was dropped. 〔U7〕
  - No geometric check was relaxed: forward-backward < 1 px, ray angle ≥ 2°, reprojection < 1 px, both depths > 0.
  - Sensitivity: 0.1 → MAE 0.8; 0.25 → 0.9; 0.5 → 030L keeps 9 points (incomplete, the clean A/B values to within 1e-6 cm). Every threshold from 0.05 to 0.25 passes, so the result does not hinge on tuning. 〔U8〕
  - Band points: 030L 9 → 28, 030R 15 → 47, 090R photo 1 45 → 67, photo 3 0 → 46.
- **No hole-filling needed.** The MVS is complete on its own, so the "fill only object masks, floor from a RANSAC plane on triangulated points" fix was not built. Add it if the downstream SAM 3D needs denser object point maps.
- **Blocked or not run:**
  - VGGT-1B-Commercial: the account behind the Modal secret `huggingface` authenticates (whoami works), but `config.json` and `model.safetensors` both return `GatedRepoError 403`. **The user must request access with that HF account** (manual approval). Not retried.
  - map-anything-apache "with our K" was not run separately. The clean A/B's BA-f already refines its cameras, focal included, to the backbone-independent values, and map-anything still reads MAE 12.5 (030R −46 cm): its dense surface is the problem, not K. With the MVS passing, this row cannot change the verdict.
  - My own classic route (ALIKED + LightGlue + MoGe-3 depth) found only 217–572 matches per pair on the 518 frames, an almost flat focal sweep, and MAE 4.4–4.6. RoMa (10 000 per pair) supersedes it; it is kept as a diagnostic.
- **Licences** of the MVS route: RoMa (MIT, outdoor weights), pycolmap 4.2.1 (BSD-3; the wheel may carry SuiteSparse GPL parts, see licences.md) 〔U1〕, OpenCV (Apache-2.0), DINOv2 (Apache-2.0, inside RoMa); start backbone DA3-BASE (Apache-2.0) or MoGe-3 (MIT). Not legal advice.
- **Next:** cell 090's geometry is exported in the lucida-replica-01 layout (see Export), ready for the downstream completion A/B (SAM 3D + unchanged assembly).

## 几何阶段契约 / Geometry stage contract

每个主干完全相同；评判器（`fair_ab_modal.analyse` → `analyse_one`）不改。Identical for every backbone; the evaluator is unchanged.

| | 内容 / content |
|---|---|
| **输入 input** | 该 run 冻结的 canonical 帧：补边 518×518 RGB PNG，与 `<run>/evidence/canonical/frame_000N.png` 逐像素相同，按帧顺序，经 `MapAnythingAdapter` 的 runner 接口（data URI）或 `fair_ab_modal.frames_for`。可选：我们的 K（每个工位一个 3×3，canonical 网格）。单目 / 匹配模型可以只看内容区 `[:, 63:455]`（同一批像素），结果放回 518 网格。The run's frozen canonical frames (padded 518×518, pixel-identical), in frame order. Optional: our K. Monocular models and matchers may see only the content crop of the same pixels; their output goes back on the 518 grid. |
| **输出 output** | `<geometry>/frames/frame_000N/`，N = 1..帧数，即 `MapAnythingAdapter` 的文件：`canonical.png`（输入帧原样）、`pts3d.npy` H×W×3 float32 世界点（像素 (u, v) → 它的 3D 点）、`conf.npy` H×W（越大越可信；harness 只用 ≥ 0.1）、`valid_mask.npy` H×W bool、`intrinsics.npy` 3×3（canonical 网格，像素中心在整数）、`camera_to_world.npy` 4×4 OpenCV c2w（右 / 下 / 前，刚体）；外加 `candidate_manifest.json`（来源）。单位任意：米 / 原生单位由每个主干自己的急停定。The MapAnythingAdapter files above plus `candidate_manifest.json`. Units are native; each backbone's own e-stop sets metres per native unit. |
| **检查 check** | `backbones.check_arrays()` / `check_geometry()`，打分前对每份几何都跑：帧数与帧号、形状与类型、有效值有限、有效比例 > 5 %、K 合法（`K[2] = [0,0,1]`、焦距 > 0、无斜切、主点在图内）、c2w 刚体（R Rᵀ = I、det = 1，容差 1e-3）、点在相机前方、**点图在 (K, c2w) 的网格上**（有效点重投影中位 < 5 px，抓网格错位 / 裁剪偏移）、`canonical.png` 与冻结帧逐像素相同。Run on every geometry before scoring: frame count and ids, shapes and dtypes, finite valid values, valid fraction > 5 %, a proper K, a rigid c2w, points in front of the camera, the point map on the grid of (K, c2w) (median reprojection < 5 px: catches grid shifts and crop offsets), and the frozen canonical frames. |

各主干的适配器（都满足这份契约）/ adapters, all emitting this contract：
- Pi3X `scripts/candidate_pi3x_backend.py: Pi3XRunner`；DA3-L / DA3-BASE `scripts/candidate_geometry_backend.py: DA3Runner`；map-anything-apache `scripts/candidate_mapanything_backend.py: MapAnythingRunner`（panoptes-serving，经 `MapAnythingAdapter`）。
- VGGT-1B-Commercial、MoGe-3、RoMa + pycolmap BA、MVS、补洞：`modal_apps/geometry_clean_ab.py`（worktree `panoptes-workcell-photo-speed`；VGGT 获批后加 `--vggt`）。
- 本笔记：`mvs_route.py`（完成的 MVS，复用 `geometry_clean_ab.mvs`）、`backbones.py`（契约检查 + 已被取代的 ALIKED 经典路线）。

契约检查的诊断值（有效比例 / 点图重投影中位 px）：Pi3X 0.96–0.98 / 0.28–0.63；DA3 0.96–0.97 / 0.00（深度 + 相机反投影）；map-anything 0.91–0.92 / 1.3–2.0（它的点和它自己的相机不完全一致）；MVS 0.17–0.47 / 0.08–0.15。
Contract diagnostics (valid fraction / point-map reprojection median, px): Pi3X 0.96–0.98 / 0.28–0.63; DA3 0.96–0.97 / 0.00; map-anything 0.91–0.92 / 1.3–2.0 (its points disagree with its own cameras); MVS 0.17–0.47 / 0.08–0.15.

## 判定表 / Verdict table

主配置 = 补边帧 + 最大内点地面 + 点图近面（自由平面），与公平 A/B 相同。误差 = 测量 − 现场值（cm）。通过 = 两个工位急停门限都过、MAE ≤ 1.56（Pi3X 1.44 + 噪声 0.12）、每个误差 ≤ 3 cm、地面 p95 ≤ 2.5 cm。
Primary = padded frames, max-inlier floor, near face on the point map (free plane), as in the fair A/B. Error = measured − field (cm). Pass = both e-stop gates, MAE ≤ 1.56 (Pi3X 1.44 + 0.12 noise), every error ≤ 3 cm, floor p95 ≤ 2.5 cm.

| | Pi3X（已发布几何 × 公平评判器 published geometry, fair evaluator）〔U4〕 | DA3-LARGE-1.1 | DA3-BASE | VGGT-1B-Commercial | **MVS（DA3-BASE 起始 start）** | MVS（MoGe-3 起始 start） |
|---|---|---|---|---|---|---|
| 权重许可 licence | CC BY-NC 4.0 | 有争议 disputed | Apache-2.0 | VGGT License + AUP | **MIT / BSD-3 / Apache-2.0** 〔U1〕 | MIT / BSD-3 〔U1〕 |
| 急停最大偏差 e-stop max dev 090 / 030 | 2.98 / 1.53 % | 3.47 / 1.69 % | 1.94 / 1.64 % | **没跑 not run：HF 403** | 3.30 / 1.51 % | 3.35 / 1.58 % |
| 090 右罩壳 090R housing (24) | 23.4 (−0.6) | 22.6 (−1.4) | 19.3 (−4.7) | — | **24.0 (+0.0)** | 23.9 (−0.1) |
| 030 左罩壳 030L housing (24) | 22.1 (−1.9) | 23.4 (−0.6) | 16.4 (−7.6) | — | **23.6 (−0.4)** | 23.6 (−0.4) |
| 030 右罩壳 030R housing (24) | 23.4 (−0.6) | 25.7 (+1.7) | 18.8 (−5.2) | — | **25.0 (+1.0)** | 25.0 (+1.0) |
| 090 右围栏下横梁 090R fence (20)，两视图 | 22.7 (+2.7) | 19.0 (−1.0) | 19.7 (−0.3) | — | **21.0 (+1.0)** | 21.0 (+1.0) |
| **MAE（4 个）** | **1.4** | **1.2** | **4.5** | — | **0.6** | **0.6** |
| 最大误差 max abs | 2.7 | 1.7 | 7.6 | — | 1.0 | 1.0 |
| 最小二乘地面 lsq floor MAE | 1.4 | 1.1 | 4.4 | — | 0.6 | 0.6 |
| 最低平面规则 lowest-plane rule MAE | 1.8 | 1.9 | 5.8 | — | 0.5（地面 p95 2.69） | 0.4 |
| 竖直面变体 vertical-plane MAE | 1.4 | 1.2 | 4.4 | — | 0.7 | 0.7 |
| 三根同型罩壳极差 housing range | 1.3 | 3.1 | 2.9 | — | 1.4 | 1.3 |
| 地面残差 p95 floor residual 090 / 030 | 0.37 / 0.97 | 0.78 / 1.78 | 1.57 / 2.30 | — | 0.79 / 0.28 | 0.81 / 0.28 |
| 黄/红比 yellow/red (2.00) | +2.3 % | +2.5 % | +2.2 % | — | +2.5 % | +2.3 % |
| 可见高度 visible height (~8.5) 090 / 030 | 8.37 / 8.27 | 8.26 / 8.21 | 8.14 / 8.27 | — | 8.04 / 8.37 | 8.04 / 8.37 |
| 五张相机高 camera heights（假设 assumption） | 154.8 / 154.1 / 155.7 · 149.5 / 152.0 | 151.9 / 152.0 / 153.4 · 150.2 / 153.6 | 163.1 / 161.2 / 165.4 · 153.1 / 155.7 | — | 146.1 / 145.8 / 146.5 · 145.7 / 144.9 | 146.3 / 145.9 / 146.7 · 145.5 / 144.7 |
| 相机高极差 camera range | 6.3 | 3.5 | 12.4 | — | 1.7 | 2.1 |
| 090 左罩壳（不计分）090L (listed) | 19.9 | 19.1 | 12.4 | — | 23.9 | 23.8 |
| 在 Pi3X 噪声内？ within noise of Pi3X? | 参考 ref | 是 yes（许可有争议） | 否 no | 被挡 blocked | **是 yes** | **是 yes** |

### 所有候选（主配置）/ Every candidate (primary configuration)

| 候选 candidate | 来源 source | 急停 e-stop 090 / 030 | 090R | 030L | 030R | 围栏 fence | MAE | 最大 max | 地面 floor p95 | 通过 pass |
|---|---|---|---|---|---|---|---|---|---|---|
| map-anything-apache（只图像 image only） | 本笔记重算 = clean A/B | **6.20** / 1.50 % | 18.8 | 20.4 | 117.1 | 25.9 | — | — | 1.28 / 8.72 | 否（急停门限）|
| map-anything-apache + RoMa BA-f | clean A/B | 2.96 / 1.65 % | 21.5 | 22.8 | −21.6 | 20.9 | 12.5 | 45.6 | 2.54 / 5.00 | 否 |
| MoGe-3 + RoMa BA-f | clean A/B | 3.29 / 1.71 % | 21.8 | 13.8 | 20.0 | 21.5 | 4.5 | 10.2 | 0.70 / 1.48 | 否 |
| DA3-BASE + RoMa BA-f | clean A/B | 2.62 / 1.52 % | 19.8 | 15.8 | 18.6 | 19.6 | 4.6 | 8.2 | 1.27 / 2.26 | 否 |
| MVS + DA3-BASE 补洞 fill（CERT 0.5） | clean A/B | 3.35 / 1.72 % | 19.5 | 22.3 | 23.9 | 21.0 | 1.8 | 4.5 | 0.91 / **11.83** | 否 |
| MVS, CERT 0.5 | clean A/B；本笔记重跑相同 re-run identical（< 1e-6 cm） | 3.36 / 1.51 % | 24.7 | —（9 点 pts） | 25.9 | 21.0 | — | — | 0.80 / 0.25 | 否（不完整 incomplete）|
| MVS, CERT 0.25 | 本笔记 敏感性 sensitivity | 3.31 / 1.51 % | 24.5 | 23.6 | 25.7 | 21.0 | 0.9 | 1.7 | 0.79 / 0.26 | 是 yes |
| MVS, CERT 0.1 | 本笔记 敏感性 sensitivity | 3.31 / 1.51 % | 24.1 | 23.6 | 25.6 | 21.0 | 0.8 | 1.6 | 0.79 / 0.27 | 是 yes |
| **MVS, CERT 0.05（DA3-BASE 起始）** | **本笔记 this note** | 3.30 / 1.51 % | 24.0 | 23.6 | 25.0 | 21.0 | **0.6** | 1.0 | 0.79 / 0.28 | **是 yes** |
| MVS, CERT 0.05（MoGe-3 起始） | 本笔记 | 3.35 / 1.58 % | 23.9 | 23.6 | 25.0 | 21.0 | 0.6 | 1.0 | 0.81 / 0.28 | 是 yes |
| ALIKED + LightGlue，扫描焦距，MoGe-3 深度 | 本笔记 诊断 diagnostic | 2.61 / 1.64 % | 24.2 | 14.2 | 18.1 | 21.8 | 4.4 | 9.8 | 5.56 / 3.61 | 否 |
| 同上，MoGe-3 K（早期代码版本 earlier code） | 本笔记 诊断 | 2.30 / 1.59 % | 24.6 | 14.1 | 20.0 | 23.9 | 4.6 | 9.9 | 4.58 / 1.23 | 否 |

- 所有数字由同一个评判器算出；公平 A/B 和 clean A/B 的行从它们自己的逐 run 结果重新汇总，与各自 README 一致。map-anything 只图像那行本笔记又评了一遍，与 clean A/B 相同（差 < 1e-6 cm，浮点噪声）。
  Every number comes from the same evaluator. The fair and clean A/B rows are re-aggregated from their own per-run files and match their READMEs. The image-only map-anything row was re-scored here and matches the clean A/B to within 1e-6 cm (floating-point noise).
- 相机高是一致性参考（假设同一拍摄者、手机高度相近），不是真值。Camera heights are a consistency check (one photographer, similar phone height), not ground truth.
- MVS 的 BA-f 焦距 090 为 375.7 / 376.2 / 372.1，030 为 381.2 / 374.9 px（Pi3X 380.5 / 382.5 / 378.6 · 391.9 / 391.0）。MVS BA-f focals; Pi3X's in brackets.

## 导出：090 几何，lucida-replica-01 目录格式 / Export: cell 090 in the lucida-replica-01 layout

`/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/checks/bbab-export-090-mvs-da3-base/`（20 MB）

- `geometry/frames/frame_000{1,2,3}/`：`pts3d.npy`、`conf.npy`、`valid_mask.npy`、`intrinsics.npy`、`camera_to_world.npy`、`canonical.png`（冻结帧原字节）、`content_valid_mask.npy`（`prepare_capture_evidence.RULE`）；`geometry/candidate_manifest.json`。
- `evidence/objects/<物体 object>/<帧 frame>/points.npy`、`colors.npy` 用新点图重算（mask ∩ content，同 `prepare_lucida_evidence`），`objects.json` 的点数、中位质心、成对距离已更新；mask、rgba 是指向原 run 的只读符号链接。〔U10〕
  Object points and colours are re-derived from the new point maps (mask ∩ content); masks and rgba crops are read-only symlinks to the original run. 〔U10〕
- `evidence/floor.json` + `floor_points.npy` / `floor_colors.npy`：地面 = 公平 A/B 主规则（maxInlier）在冻结地面 mask 上，附该主干自己的急停尺度 `estopNativeToMeters` = 3.2371 m / 原生单位（偏差 3.30 %）。**不是**已发布的尺度。
  Floor = the fair A/B primary rule on the frozen floor masks, with this backbone's own e-stop scale (3.2371 m per native unit, 3.30 % deviation). Not a published scale.
- `input` → 原 run 的 `input/`（符号链接）；`evidence/canonical` 复制；`manifest.json` 的 geometry / evidence 记录已改写。原 run 的哈希核对过，未被改动。
- 下游自检：原封不动的 `assemble_lucida_scene.scoring_height` + `load_view` 读出全部 9 个物体、25 个视图，每个视图在 144 网格上有 79–1 572 个深度样本。
  Downstream check: the unchanged `assemble_lucida_scene.scoring_height` + `load_view` load all 9 objects and 25 views, with 79–1 572 depth samples per view on the 144 grid.
- 注意 / caveats：
  - 物体 mask 里的点数是 Pi3X 的 21–97 %（机器人、小车、护罩 77–97 %；立柱和光幕 21–69 %，例：左立柱 1 579 / 1 421 / 1 117，Pi3X 4 405 / 3 544 / 3 247；地面 35 165 对 35 487）。MVS 点图有洞（`valid_mask` = 假），SAM 3D 的点图输入接受 NaN。〔U11〕
    Object-mask coverage is 21–97 % of Pi3X's (robot, cart, guard 77–97 %; posts and light curtains 21–69 %). The point maps have holes, which SAM 3D's pointmap input accepts as NaN. 〔U11〕
  - 原生单位与 Pi3X 不同（3.24 对 1.29 m / 单位）：下游如有原生单位阈值，要按 `estopNativeToMeters` 换算。
    Native units differ from Pi3X's (3.24 vs 1.29 m per unit); rescale any native-unit threshold downstream.

## 方法 / Method

- 相机：clean A/B 的 `DA3-BASE + RoMa BA-f`（RoMa outdoor 每对 10 000 个匹配 → USAC-MAGSAC F（1 px）→ 两视图轨迹 → pycolmap 4.2.1 BA，Cauchy，每张一个焦距，主点固定）。MoGe-3 起始给出同样的相机。
  Cameras: the clean A/B's DA3-BASE + RoMa BA-f. The MoGe-3 start gives the same cameras.
- 表面：`geometry_clean_ab.mvs` 原样（RoMa 完整 warp 在每个内容区像素双向采样、中点三角化、每像素取夹角最大的伙伴），只把 `CERT` 换成 0.05；`conf` 按 RoMa 的约定，保留的像素 = 1。〔U2〕
  Surface: `geometry_clean_ab.mvs` unchanged except `CERT` = 0.05; `conf` = 1 on kept pixels, RoMa's convention. 〔U2〕
- 评判：`fair_ab_modal.analyse`（它自己的 app，临时运行），先跑契约检查。汇总：公平 A/B 的 `compile.config`，不完整行用 clean A/B 的 `safe_config`。
  Scoring: `fair_ab_modal.analyse` (its own app, ephemeral) after the contract check; aggregation by the fair `compile.config` (`safe_config` from the clean A/B for incomplete rows).
- 门限 0.05 是在看到旧 MVS 不完整之后、算任何新数之前定的，依据是带内诊断和 RoMa 的默认值；之后没有再调。敏感性三档一并报告。这是第 4 次尝试，见局限。
  The 0.05 threshold was chosen after seeing the old MVS was incomplete but before any new number, from the band diagnostic and RoMa's default; it was not changed afterwards. Three sensitivity levels are reported. It was the 4th attempt; see Limits.

## 局限 / Limits

- 只有 4 个现场值、两个工位、5 张照片。0.6 对 1.4 cm 只能说"不差于 Pi3X"，不能排名。
  Four field values, two cells, five photos: 0.6 vs 1.4 cm reads as "no worse than Pi3X", not as a ranking.
- 030 只有两张照片：每个像素只有一个伙伴，覆盖率（17–22 %）比 090（37–47 %）低。拍摄时每个物体至少两张、基线 ≥ 10° 的规则仍然重要。
  Cell 030 has two photos, so one partner per pixel and lower coverage (17–22 % vs 37–47 % in 090). Capture guidance (two photos per object, ≥ 10° baseline) still matters.
- MVS 依赖 RoMa 权重（MegaDepth 训练，数据来源未审计，见 clean A/B 许可表）。
  The MVS depends on RoMa's weights (MegaDepth-trained; data provenance not audited).
- pycolmap 轮子可能含 GPL 的 SuiteSparse 部分（licences.md 推断，未拆轮子核实）；on-prem 交付前要么履行 GPL，要么自编 Ceres 不带 SuiteSparse。〔U1〕
  The pycolmap wheel may contain GPL SuiteSparse parts (inferred in licences.md); fulfil the GPL or build Ceres without SuiteSparse before on-prem delivery. 〔U1〕
- **（2026-10-06 补）** CERT 0.05 是同 4 个现场值上的第 4 次尝试，是看到 CERT 0.5 的逐物体不完整之后才改的（CERT 0.5 逐物体分数 20:03:23Z 读过）。值本身是 RoMa 默认 `sample_thresh`，没扫，20:05:01Z 写入，20:07:32Z 打分。`sample_thresh` 在 RoMa 里是采样饱和值，不是稠密 warp 的有效性门限。
  **(added 2026-10-06)** CERT 0.05 is the 4th attempt on the same 4 field values, made after seeing incomplete objects at CERT 0.5 (its per-object scores were read at 20:03:23Z). The value is RoMa's default `sample_thresh`, not swept; written 20:05:01Z, scored 20:07:32Z. In RoMa, `sample_thresh` is a sampling saturation, not a validity cut on the dense warp.

## 文件 / Files

- `backbones.py`：契约（`check_arrays`、`check_geometry`、`finish`）+ ALIKED 经典路线（诊断）；`python backbones.py` 自检。Contract + the ALIKED diagnostic route; self-test.
- `mvs_route.py`：完成的 MVS（`python mvs_route.py da3-base [CERT ...]`）。The completed MVS.
- `backbone_ab_modal.py`：`--stage access`（VGGT 访问，只记状态）、`--stage classic`；`python backbone_ab_modal.py analyse ...`（契约检查 + 原评判器）、`export BACKBONE CELL`。
- `band_diag.py`：带内诊断（改门限的依据）；输出 `band_diag.out`（VERIFY 重跑）。The band diagnostic behind the threshold change; its output is `band_diag.out` (VERIFY re-run).
- `ba_scipy.py`：GPL-free BA（`bundle_adjust` numpy Schur LM、`bundle_adjust_lsq` 参照、`refine` 接口同 `geometry_clean_ab.refine`）；`python ba_scipy.py` 自检。`ba_scipy_modal.py`：路线 + 评判 + 导出（临时 `modal run`）。`ba_scipy_results.json`：4 个配置、相机对照、镜像核查、导出、花费。GPL-free BA, its Modal driver, and its results.
- `VERIFY.md`：独立核验。The independent verification.
- `compile.py` → `results.json`：全部行、三种地面 × 两种近面、契约诊断、VGGT 访问记录、花费。Every row, floor rule and surface variant, contract diagnostics, the VGGT access record, spend.
- 草稿 / scratch（`scratchpad/checks/`）：`bbab-geom/`（新几何；`vggt-access.json`；map-anything 只图像是指向 `geomab-*` 的链接）、`bbab-analyse/`（逐 run 结果 + 花费）、`bbab-export-090-mvs-da3-base/`、`bbab-superseded/`（ALIKED 第一版焦距扫描，未计分）。

## 花费 / Spend（Modal list 价，按函数秒，不是账单 / list rates on function seconds, not an invoice）

| 项 item | USD |
|---|---|
| 评判 analysis（8 CPU，16 个 job） | 0.21 |
| ALIKED 经典路线 classic route（L4，3 次） | 0.11 |
| VGGT 访问检查 access check（1 CPU，几秒） | < 0.001 |
| **合计 total** | **≈ 0.32** |

- 加上调用秒和镜像构建（MoGe + LightGlue 镜像、一个没用上的 A100 镜像），上限约 $0.40。没有 GPU 推理被重做；没有新建 Modal volume（只读挂了已有的 `moge3-hf-cache`）。
  With call seconds and image builds (the MoGe + LightGlue image and an A100 image that ended up unused), at most about $0.40. No GPU inference was redone; no Modal volume was created (the existing `moge3-hf-cache` was mounted).
- **2026-10-06 更新**：GPL-free BA $0.403，权重镜像 ≈ $0.0084，均为 list 价，不在上表合计里。Update: GPL-free BA $0.403, weight mirror ≈ $0.0084, list price, not in the total above.
- clean A/B 的 $0.43 和公平 A/B 的 $0.33 在各自笔记里计。The clean and fair A/Bs' spend is counted in their notes.
