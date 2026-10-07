# 许可干净的几何主干 A/B（公平规则，对 Pi3X）/ Licence-clean geometry backbone A/B vs Pi3X (2026-10-06)

评分用 `geometry-licence-ab-fair-2026-10-05` 的公平规则，代码一行没改：冻结的 canonical 帧、中性的最大一致集地面、相同的照片集、每个主干用自己的急停定尺度（`estop_cylinder.fit` + `joint_scale`，4 % 门限）、在主干自己的点图上取罩壳下沿近面、围栏用两视图。现场值只用来验证：罩壳下沿 24 cm，围栏下横梁 20 cm，急停黄/红 2.0，支架以上可见约 8.5 cm，相机高约 1.5 m（同一个拍摄者）。已发布的 nativeToMeters 和测量层都没动。
Scored with the fair harness of `geometry-licence-ab-fair-2026-10-05`, code unchanged: frozen canonical frames, neutral max-inlier floor, the same photo sets, each backbone's own e-stop scale (`estop_cylinder.fit` + `joint_scale`, 4 % gate), the housing lower edge on the near face of the backbone's own point map, and the two-view fence. The field values are used for validation only. Published nativeToMeters and layers are untouched.

## 结论 / Verdict

**中文**
- **今天没有一个许可干净的主干能达到 Pi3X 的精度。**
  - 判定标准：MAE ≤ Pi3X 已发布值 1.44 cm + 噪声 0.12 cm（Pi3X 同一权重 CUDA 对 MPS 的差）。
  - 达标的只有 Pi3X 自己（非商用）和 DA3-LARGE-1.1（许可有争议）。
- **VGGT-1B-Commercial 没跑成。** 用现有 Modal secret `huggingface` 下载，HF 返回 `GatedRepoError 403`：这个账号没有获批。
  - 要先用 secret 对应的 HF 账号在 HF 上提交申请（README 说是自动审批）。
  - 代码已经写好，获批后加 `--vggt` 就能跑。
  - 另外，它的 AUP 禁止用于"关键基础设施、交通、**重型机械的运行**"。工位安全检查算不算，要法务先看。
- **四个许可干净的候选（主配置下的 MAE / 最大误差，cm）：**
  - DA3-BASE：4.5 / 7.6；加 BA 后 4.8 / 8.2，加 BA 并精修焦距后 4.6 / 8.2。三根罩壳仍然低 4–8 cm。
  - map-anything-apache：原始结果 090 急停 6.2 %，过不了门限。加 BA 后能过，但 030 右罩壳是 +47 / −46 cm，MAE 14.2 / 12.5。
  - MoGe-3（MIT）单张 + RoMa + BA：4.3–4.7 / 10–11，030 左罩壳低约 10 cm。
- **BA 本身是有效的，也是许可干净的**（RoMa MIT + pycolmap BSD-3）。〔2026-10-06 更正：PyPI pycolmap 轮子含 GPL-2.0+ 的 SuiteSparse 部分，见许可表；交付路线改用 `ba_scipy`。〕
  - 作对照，同一套 BA 用在 Pi3X 上：MAE 1.44 → 0.9（只修相机）/ 0.8（相机 + 焦距）；围栏 +2.7 → +0.6 / +0.2 cm。
  - BA 精修焦距后，无论从哪个主干开始，焦距都收敛到同一个值：090 约 375 / 376 / 372 px，030 约 381 / 375 px。Pi3X 自己估的是 380 / 382 / 379，DA3-BASE 400–405，MapAnything 475–485，MoGe 326–371。所以 **BA 给出的相机与主干无关**。
  - 干净主干差在**稠密表面**（罩壳近面、地面），不在相机。
- **又试了两种绕法（参数都在算数之前定下，见方法）：**
  1. **MVS**：BA 相机 + RoMa 稠密匹配 + 两视图三角化，完全不用学习得到的深度。
     - 它与起始主干无关：四个起点给出相同的值。090 右罩壳 +0.7，030 右罩壳 +1.9，围栏 +1.0 cm。
     - 五张照片的相机高 145–147 cm，极差 1.2–3.4 cm（Pi3X 6.3）。地面 p95 0.25 / 0.8 cm。
     - 但 **030 左罩壳那张照片只三角化出 9 个点**（门限 12），090 右罩壳照片 3 一个点也没有。所以 4 个值缺 1 个，**不完整**。
     - 已有的 3 个值平均误差 1.2 cm；Pi3X 在同样这 3 个上是 1.3 cm。
  2. **MVS + 主干补洞**（σ = 16 px 平滑后的 MVS/主干深度比）：
     - 最好的是 DA3-BASE：MAE 1.83，最大 4.5 cm，比 Pi3X 的上限 1.56 高 0.27 cm。
     - 但 030 地面 p95 变成 11.8 cm：补出来的深度和三角化的地面不一致。MoGe 和 MapAnything 补洞后的 090 右罩壳差 −24 cm。
     - **失败。按规则在第三次尝试后停止。**
- **只在次要的竖直面变体下打平 / ties only under the secondary vertical-plane variant**（这个变体是公平 A/B 预先登记的 / pre-registered in the fair A/B）：
  - map-anything-apache + BA-f：MAE 1.41，最大 2.0；DA3-BASE 补洞：1.35，最大 2.4。Pi3X 在同一变体下是 1.35，最大 2.7。
  - 但前者主配置是 12.5（030 右罩壳的近面平面倾斜了），地面 p95 2.5 / 5.0 cm，相机高 160–173 cm；后者 030 地面 p95 11.8 cm。所以它们的点图仍然不对，不算达标。这是线索，不是结论。
  - map-anything-apache + BA-f: MAE 1.41, worst 2.0. DA3-BASE fill: 1.35, worst 2.4. Pi3X under the same variant: 1.35, worst 2.7.
  - But the first is 12.5 in the primary configuration (its 030R near-face plane tilts), with floor p95 2.5 / 5.0 cm and camera heights 160–173 cm. The second has a 030 floor p95 of 11.8 cm. Their point maps are still wrong, so neither counts as a pass. This is a lead, not a result.
- **建议**
  - 许可干净里最有希望的路线是 **"任意干净主干作初值（DA3-BASE Apache 或 MoGe-3 MIT）→ RoMa + pycolmap BA → MVS"**。主干只用来初始化，最终相机和表面与主干无关。
  - 它现在缺的是**覆盖率**，不是精度。下一步：
    - 用原始照片分辨率匹配（2880×3840，现在是 518 的 canonical 帧）；
    - 或者补拍：每根立柱至少两张、基线 ≥10°。
  - 在覆盖率补上之前：已发布报告继续用 Pi3X。商用 on-prem 需要以下之一：
    - Pi3X 作者的书面商用许可；
    - DA3-LARGE-1.1 作者书面确认 Apache；
    - VGGT-1B-Commercial 获批，并通过 AUP 审查。

**English**
- **No licence-clean backbone matches Pi3X today.**
  - Rule: MAE ≤ Pi3X published 1.44 cm + 0.12 cm noise (the same Pi3X weights, CUDA vs MPS).
  - Only Pi3X itself (non-commercial) and DA3-LARGE-1.1 (disputed licence) meet it.
- **VGGT-1B-Commercial was not run.** The existing Modal secret `huggingface` gets `GatedRepoError 403`: that account has not been approved.
  - Someone has to apply on HF with the account behind that secret (the VGGT README says approval is automatic).
  - The code path is ready: rerun with `--vggt`.
  - Its AUP also prohibits use related to the "operation of … heavy machinery". Legal should check whether workcell safety audits fall under it.
- **The four clean candidates (primary MAE / worst error, cm):**
  - DA3-BASE: 4.5 / 7.6; with BA 4.8 / 8.2, with BA + focal 4.6 / 8.2. The three housings still read 4–8 cm low.
  - map-anything-apache: raw fails the e-stop gate in 090 (6.2 %). With BA it passes the gate, but 030R housing reads +47 / −46 cm; MAE 14.2 / 12.5.
  - MoGe-3 (MIT) per photo + RoMa + BA: 4.3–4.7 / 10–11; 030L housing reads about 10 cm low.
- **The bundle adjustment works, and it is licence-clean** (RoMa MIT + pycolmap BSD-3). [Corrected 2026-10-06: the PyPI pycolmap wheel contains GPL-2.0+ SuiteSparse parts, see Licences; the shipped route uses `ba_scipy`.]
  - Control, the same BA on Pi3X: MAE 1.44 → 0.9 (cameras only) / 0.8 (cameras + focal); fence +2.7 → +0.6 / +0.2 cm.
  - With focal refinement, every starting backbone converges to the same focal: about 375 / 376 / 372 px in 090 and 381 / 375 px in 030. The backbones' own estimates were 380 / 382 / 379 (Pi3X), 400–405 (DA3-BASE), 475–485 (MapAnything) and 326–371 (MoGe). So **the BA cameras do not depend on the backbone**.
  - What the clean backbones get wrong is their **dense surface** (housing near faces, floor), not their cameras.
- **Two further workarounds (parameters fixed before any number was computed):**
  1. **MVS**: BA cameras + RoMa dense warp + two-view triangulation, with no learned depth at all.
     - It does not depend on the starting backbone: all four starts give the same values. 090R housing +0.7, 030R housing +1.9, fence +1.0 cm.
     - Five camera heights 145–147 cm, spread 1.2–3.4 cm (Pi3X 6.3). Floor p95 0.25 / 0.8 cm.
     - But **the 030L housing photo triangulates only 9 band points** (the harness needs 12), and 090R photo 3 triangulates none. One of the four values is missing, so the result is **incomplete**.
     - On the 3 values it has, the mean error is 1.2 cm; Pi3X gets 1.3 cm on the same three.
  2. **MVS + backbone hole-filling** (MVS/backbone depth ratio smoothed with σ = 16 px):
     - The best is DA3-BASE: MAE 1.83, worst 4.5 cm, 0.27 cm above Pi3X's limit of 1.56.
     - But the 030 floor p95 becomes 11.8 cm, because the fill disagrees with the triangulated floor. MoGe and MapAnything fills put 090R at −24 cm.
     - **Failed. Stopped after the third attempt, per the rules.**
- **Recommendation**
  - The most promising licence-clean route is: **any clean backbone as initial value (DA3-BASE Apache or MoGe-3 MIT) → RoMa + pycolmap BA → MVS**. The backbone only initialises; the final cameras and surface do not depend on it.
  - What it lacks now is **coverage**, not accuracy. Next steps:
    - match at the original photo resolution (2880×3840; today it is the 518 canonical frames);
    - or capture more: at least two photos per post, with a baseline of 10° or more.
  - Until coverage is fixed, keep Pi3X for the published reports. Commercial on-prem needs one of these:
    - written commercial permission for Pi3X from its authors;
    - written confirmation of Apache for DA3-LARGE-1.1 from its authors;
    - VGGT-1B-Commercial approved and cleared under its AUP.

## 判定表 / Verdict table

主配置：补边帧、最大一致集地面、点图近面（自由平面）。误差 = 测量值 − 现场值（cm）。"≥ Pi3X" 指 MAE ≤ 1.44 + 0.12。
Primary configuration: padded frames, max-inlier floor, near face on the point map (free plane). Error = measured − field (cm). "≥ Pi3X" means MAE ≤ 1.44 + 0.12.

| 候选 candidate | 许可 licence | 急停偏差 e-stop dev 090 / 030（门限 gate） | 090R 罩壳 housing (24) | 030L 罩壳 (24) | 030R 罩壳 (24) | 090R 围栏 fence (20) | MAE | 最大 worst | 罩壳极差 housing range | 相机高极差 cam range (5) | 地面 floor p95 090 / 030 | ≥ Pi3X |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pi3X 已发布 published | CC BY-NC | 2.98 / 1.53 %（过 pass） | 23.4 (−0.6) | 22.1 (−1.9) | 23.4 (−0.6) | 22.7 (+2.7) | **1.4** | 2.7 | 1.3 | 6.3 | 0.37 / 0.97 | 参考 ref |
| Pi3X CUDA 重跑 re-run | CC BY-NC | 3.02 / 1.53 % | 23.4 (−0.6) | 22.1 (−1.9) | 23.4 (−0.6) | 23.1 (+3.1) | 1.6 | 3.1 | 1.3 | 6.0 | 0.39 / 0.97 | 噪声 noise |
| *Pi3X + BA（对照 control）* | CC BY-NC | 3.40 / 1.65 % | 23.2 (−0.8) | 22.3 (−1.7) | 23.5 (−0.5) | 20.6 (+0.6) | 0.9 | 1.7 | 1.2 | 4.3 | 0.43 / 0.89 | — |
| *Pi3X + BA-f（对照 control）* | CC BY-NC | 3.34 / 1.57 % | 23.4 (−0.6) | 22.3 (−1.7) | 23.4 (−0.6) | 20.2 (+0.2) | 0.8 | 1.7 | 1.2 | 6.2 | 0.39 / 0.88 | — |
| DA3-LARGE-1.1 | **有争议 disputed** | 3.47 / 1.69 % | 22.6 (−1.4) | 23.4 (−0.6) | 25.7 (+1.7) | 19.0 (−1.0) | 1.2 | 1.7 | 3.1 | 3.5 | 0.78 / 1.78 | 是 yes |
| **VGGT-1B-Commercial** | VGGT License + AUP | — | — | — | — | — | — | — | — | — | — | **没跑 not run（HF 403）** |
| DA3-BASE | Apache-2.0 | 1.94 / 1.64 % | 19.3 (−4.7) | 16.4 (−7.6) | 18.8 (−5.2) | 19.7 (−0.3) | 4.5 | 7.6 | 2.9 | 12.4 | 1.57 / 2.30 | 否 no |
| DA3-BASE + BA | Apache-2.0 | 2.81 / 1.51 % | 18.9 (−5.1) | 15.8 (−8.2) | 18.6 (−5.4) | 20.7 (+0.7) | 4.8 | 8.2 | 3.1 | 10.1 | 1.16 / 2.21 | 否 no |
| DA3-BASE + BA-f | Apache-2.0 | 2.62 / 1.52 % | 19.8 (−4.2) | 15.8 (−8.2) | 18.6 (−5.4) | 19.6 (−0.4) | 4.6 | 8.2 | 4.0 | 17.1 | 1.27 / 2.26 | 否 no |
| map-anything-apache | Apache-2.0 | **6.20 ✗** / 1.50 % | 18.8 (−5.2) | 20.4 (−3.6) | 117.1 (+93.1) | 25.9 (+5.9) | 不计 dropped | — | — | 7.7 | 1.28 / 8.72 | 否 no |
| map-anything-apache + BA | Apache-2.0 | 3.49 / 1.59 % | 19.0 (−5.0) | 22.6 (−1.4) | 71.0 (+47.0) | 23.3 (+3.3) | 14.2 | 47.0 | 52.1 | 11.4 | 1.91 / 5.66 | 否 no |
| map-anything-apache + BA-f | Apache-2.0 | 2.96 / 1.65 % | 21.5 (−2.5) | 22.8 (−1.2) | −21.6 (−45.6) | 20.9 (+0.9) | 12.5 | 45.6 | 44.5 | 12.8 | 2.54 / 5.00 | 否 no |
| MoGe-3（只做 sim(3) 配准 only） | MIT | 3.28 / 1.87 % | 22.9 (−1.1) | 12.7 (−11.3) | 19.5 (−4.5) | 22.0 (+2.0) | 4.7 | 11.3 | 10.2 | 25.9 | 1.28 / 2.86 | 否 no |
| MoGe-3 + BA | MIT | 2.90 / 1.65 % | 23.4 (−0.6) | 13.6 (−10.4) | 20.2 (−3.8) | 22.4 (+2.4) | 4.3 | 10.4 | 9.8 | 21.3 | 3.73 / 1.73 | 否 no |
| MoGe-3 + BA-f | MIT | 3.29 / 1.71 % | 21.8 (−2.2) | 13.8 (−10.2) | 20.0 (−4.0) | 21.5 (+1.5) | 4.5 | 10.2 | 8.0 | 19.8 | 0.70 / 1.48 | 否 no |
| MVS（DA3-BASE 相机 cameras）| Apache-2.0 + MIT/BSD | 3.36 / 1.51 % | 24.7 (+0.7) | — (9 点 pts) | 25.9 (+1.9) | 21.0 (+1.0) | 3 个平均 mean of 3: 1.2 | — | — | 1.4 | 0.80 / 0.25 | **不完整 incomplete** |
| MVS（MoGe-3 相机 cameras）| MIT + MIT/BSD | 3.34 / 1.57 % | 24.6 (+0.6) | — | 25.9 (+1.9) | 21.0 (+1.0) | 1.2 (3) | — | — | 2.0 | 0.81 / 0.25 | 不完整 incomplete |
| MVS（MapAnything 相机 cameras）| Apache-2.0 + MIT/BSD | 3.32 / 1.63 % | 24.7 (+0.7) | — | 25.9 (+1.9) | 21.2 (+1.2) | 1.2 (3) | — | — | 3.4 | 0.82 / 0.26 | 不完整 incomplete |
| MVS + DA3-BASE 补洞 fill | Apache-2.0 + MIT/BSD | 3.35 / 1.72 % | 19.5 (−4.5) | 22.3 (−1.7) | 23.9 (−0.1) | 21.0 (+1.0) | 1.8 | 4.5 | 4.5 | 3.6 | 0.91 / **11.83** | 否 no |
| MVS + MoGe-3 补洞 fill | MIT + MIT/BSD | 3.34 / 1.63 % | 0.2 (−23.8) | 21.4 (−2.6) | 24.6 (+0.6) | 21.0 (+1.0) | 7.0 | 23.8 | 24.3 | 1.9 | 0.89 / 13.04 | 否 no |
| MVS + MapAnything 补洞 fill | Apache-2.0 + MIT/BSD | 3.30 / 1.72 % | 0.6 (−23.4) | 23.6 (−0.4) | 10.2 (−13.8) | 21.2 (+1.2) | 9.7 | 23.4 | 23.0 | 4.3 | 0.89 / 7.76 | 否 no |

**急停门限 / e-stop gate**
- 除原始 map-anything-apache（090 6.2 %）外，全部通过 4 % 门限。All pass the 4 % gate except raw map-anything-apache (6.2 % in 090).

**其它配置的 MAE / Other configurations' MAE**
- 竖直面变体 / vertical-plane variant：
  - Pi3X 1.35；DA3-BASE 补洞 1.35，MoGe 补洞 1.76，MapAnything + BA-f 1.41；其它干净候选 2.7–4.7。
  - Pi3X 1.35; DA3-BASE fill 1.35, MoGe fill 1.76, MapAnything + BA-f 1.41; the other clean candidates 2.7–4.7.
  - 见结论：只在这个次要变体下打平，主配置不行，地面也不对，所以不采用。
  - See the verdict: they tie only under this secondary variant; they fail the primary configuration and their floors are wrong, so they are not adopted.
- 最小二乘地面 / lsq floor：DA3-BASE 补洞 2.4；其它干净候选 4.0–14.3。DA3-BASE fill 2.4; the other clean candidates 4.0–14.3.
- 完整数字见 `results.json`。Every number is in `results.json`.

**急停的两个量区分不了主干 / The e-stop ratio and visible height cannot separate backbones**
- 黄/红比：所有候选 +2.0 至 +2.6 %，现场值 2.00。Yellow/red: +2.0 to +2.6 % for every candidate (field 2.00).
- 可见高度中位数：7.97–8.38 cm，现场值约 8.5。Median visible height: 7.97–8.38 cm (field ~8.5).

**相机高 / camera heights**
- 相机高是一致性参考，不是真值：它假设同一拍摄者举手机的高度差不多。
- MVS 和补洞系列五张都在 143–150 cm；Pi3X 是 149.5–155.7；DA3-BASE 153–171；MoGe 128–156。
- Camera heights are a consistency check, not ground truth: they assume one photographer holding the phone at a similar height.
- All five are 143–150 cm for the MVS and fill rows; 149.5–155.7 for Pi3X, 153–171 for DA3-BASE, 128–156 for MoGe.

## BA 报告 / BA report（`results.json` → `refinement`）

| 起点 init | 090 焦距 focal（前 before → BA-f） | 030 焦距 focal | BA-f 深度比 depth ratio 090 / 030 | 最终重投影 final reprojection BA-f |
|---|---|---|---|---|
| Pi3X | 380 / 382 / 379 → 375 / 376 / 371 | 392 / 391 → 383 / 377 | 1.006 / 1.007 / 1.002 · 1.007 / 0.993 | 0.17 / 0.16 px |
| DA3-BASE | 400 / 404 / 403 → 376 / 376 / 372 | 419 / 423 → 381 / 375 | 1.10 / 1.12 / 1.09 · 1.11 / 1.08 | 0.17 / 0.16 px |
| map-anything-apache | 475 / 480 / 481 → 376 / 376 / 372 | 480 / 485 → 387 / 381 | 1.01 / 1.00 / 0.99 · 0.99 / 0.97 | 0.17 / 0.16 px |
| MoGe-3 | 326 / 346 / 347 → 376 / 376 / 372 | 350 / 371 → 381 / 374 | 0.92 / 0.89 / 0.87 · 1.04 / 0.99 | 0.17 / 0.17 px |

- 匹配 / matches：RoMa 每对采样 10 000 个，USAC-MAGSAC F 内点 9 148–9 976。有主干深度的 7 700–9 300 个成为两视图轨迹。
  RoMa samples 10 000 per pair; 9 148–9 976 USAC-MAGSAC F inliers. 7 700–9 300 with backbone depth become 2-view tracks.
- 只修相机（BA）时焦距固定在主干自己的值：MoGe 090 平均重投影因此是 4.5e150（少数点发散）。BA-f 是正确的用法。
  With cameras only (BA), the focal stays at the backbone's own value. For MoGe in 090 the mean reprojection is then 4.5e150 (a few points diverge). BA-f is the right mode.
- BA-f 后深度比随深度增大（log-log 斜率 0.4–1.3）：单一尺度不能把干净主干的深度图变对，这就是罩壳还偏的原因。
  After BA-f the depth ratio grows with depth (log-log slope 0.4–1.3). One scale cannot fix a clean backbone's depth map, which is why the housings stay off.

## 方法 / Method

代码：`modal_apps/geometry_clean_ab.py`（worktree `panoptes-workcell-photo-speed`，新文件），以及本目录的 `compile.py`。评分直接 import 公平 A/B 的 `fair_ab_modal.analyse_one`、`fair_ab.py` 和 `compile.config`，代码未改。
Code: `modal_apps/geometry_clean_ab.py` (new file in worktree `panoptes-workcell-photo-speed`) and `compile.py` here. Scoring imports the fair A/B's `fair_ab_modal.analyse_one`, `fair_ab.py` and `compile.config`, unchanged.

- **输入 / input**：每个工位冻结的 canonical 帧（`fair_ab_modal.frames_for`，与 Pi3X 看到的字节相同）。
  - DA3-BASE、map-anything-apache、VGGT 用补边的 518×518 帧。
  - MoGe-3 和 RoMa 用 392×518 内容区（单目模型不适合补边），结果放回 518 网格。
  - Every cell's frozen canonical frames (`fair_ab_modal.frames_for`, the same bytes Pi3X saw).
  - DA3-BASE, map-anything-apache and VGGT get the padded 518×518 frames.
  - MoGe-3 and RoMa get the 392×518 content crop (a monocular model should not see the pad); results go back on the 518 grid.
- **几何来源 / geometry sources**：
  - DA3-BASE = 公平 A/B 的几何（`da3fair-geom`，`f4a6c9b`）。The fair A/B's geometry (`da3fair-geom`, `f4a6c9b`).
  - map-anything-apache = 第一次 A/B 的几何（`geomab-*-mapanything`，权重 `00f9c24`，代码 `3d10cf7`，fp32），输入帧逐像素相同。The first A/B's geometry (weights `00f9c24`, code `3d10cf7`, fp32); its frames are pixel-identical.
  - MoGe-3 = `Ruicheng/moge-3-vitl@184008f`，L4 上新跑。新跑的 MoGe 镜像沿用 `modal_apps/moge3_app.py` 的未 pin 镜像，代码版本没有逐个固定。Run fresh on an L4; the image is the unpinned `moge3_app.py` image, so its code revisions are not pinned individually.
  - K 用它自己的点反推（u = fx·X/Z + cx），各照片之间用 RoMa 匹配做 sim(3) RANSAC 配准到照片 1。K comes from its own points (u = fx·X/Z + cx); photos are registered to photo 1 by sim(3) RANSAC on the RoMa matches.
- **BA**（每个主干同一份代码）/ the same code for every backbone：
  1. RoMa outdoor（`romatch@77f8d68`，`use_custom_corr=False`），每对 10 000 个匹配，坐标换成像素中心在整数上。
     RoMa outdoor (`romatch@77f8d68`, `use_custom_corr=False`): 10 000 matches per pair, converted to integer pixel centres.
  2. USAC-MAGSAC F 矩阵（1 px），再按主干深度取两视图轨迹，3D 点初值取两条反投影的平均。
     USAC-MAGSAC F-matrix at 1 px, then 2-view tracks with the backbone depth; each 3D point starts at the mean of the two back-projections.
  3. pycolmap 4.2.1 BA：Cauchy 损失，gauge `TWO_CAMS_FROM_WORLD`。先跑一遍，删掉误差 > 2 px 的点，再跑一遍。
     pycolmap 4.2.1 BA with a Cauchy loss and gauge `TWO_CAMS_FROM_WORLD`. One pass, drop points with error > 2 px, a second pass.
  4. 两种变体：`ba` 只修位姿（PINHOLE 固定）；`ba-f` 每张照片一个焦距（SIMPLE_PINHOLE，主点固定）。
     Two variants: `ba` refines poses only (PINHOLE fixed); `ba-f` adds one focal per photo (SIMPLE_PINHOLE, principal point fixed).
  5. 每张照片的主干深度乘"BA 深度 / 主干深度"中位数，再经 BA 相机重投成点图。
     Each photo's backbone depth is scaled by the median ratio of BA depth to backbone depth, then re-cast through the BA camera.
  - pycolmap 的 BA 没有深度先验残差。深度先验在两处起作用：3D 点的初值，和被重新缩放的稠密表面。
    pycolmap's BA has no depth-prior residual. The depth prior enters twice: as the initial 3D points and as the dense surface that gets rescaled.
- **MVS**（第二次尝试；阈值在计算前定下 / second attempt; thresholds fixed beforehand：certainty ≥ 0.5、前后一致 forward-backward < 1 px、射线夹角 ≥ 2°、重投影 < 1 px）：
  - RoMa 的完整 warp 在每个内容区像素中心双向采样。
  - 用 BA-f 相机做中点三角化。
  - 每个像素取夹角最大的伙伴照片，置信度 = certainty。K 和 c2w 用 BA-f 的。
  - RoMa's full warp, sampled both ways at every content pixel centre.
  - Midpoint triangulation with the BA-f cameras.
  - Each pixel takes the partner with the widest ray angle; confidence = certainty; K and c2w are the BA-f ones.
- **补洞 / fill**（第三次尝试；σ 在计算前定下 / third attempt; σ fixed beforehand）：
  - 有 MVS 的像素用 MVS 深度。
  - 其它像素用主干深度 × exp(平滑后的 log(MVS/主干))：σ = 16 px 的归一化高斯卷积，离 MVS 太远处用全局中位数。
  - MVS depth where it exists.
  - Elsewhere backbone depth × exp(smoothed log(MVS/backbone)): normalised Gaussian convolution with σ = 16 px, and the global median far from any MVS pixel.
- **自检 / self-test**：`python modal_apps/geometry_clean_ab.py check`（需要 pycolmap 4.2.1 + opencv / needs pycolmap 4.2.1 + opencv）。合成场景是地面 + 墙、三台相机，检查四件事：
  - 位姿扰动约 2° 后，BA 恢复相对位姿：< 0.05°，基线方向 < 0.3°。
  - 深度缩放后三张一致，误差 < 0.5 %。
  - 焦距偏长 6 % 的照片，BA-f 收敛回原值，误差 < 1 %。
  - sim(3) 有 30 % 外点时仍然精确。MVS 用真相机时恢复真深度（< 1e-4）。补洞在 MVS 区域内精确，在 1σ 内 < 0.5 %。
  - Synthetic floor + wall seen by three cameras. With poses perturbed by about 2°, BA recovers the relative poses to < 0.05° (baseline direction < 0.3°).
  - The depth rescale makes the three photos agree within 0.5 %.
  - BA-f brings a 6 %-long focal back within 1 %.
  - sim(3) stays exact with 30 % outliers. MVS with the true cameras returns the true depth (< 1e-4). The fill is exact inside the MVS area and < 0.5 % within 1σ.

## 许可 / Licences（不是法律意见 / not legal advice）

| 组件 component | 许可 licence | 备注 note |
|---|---|---|
| RoMa（`romatch@77f8d68`）| MIT（代码；权重来自作者的 GitHub release / code; weights from the author's GitHub release） | 权重在 MegaDepth 上训练，训练数据来源未审计（同 on-prem 审计）。DINOv2 ViT-L 权重 Apache-2.0；VGG 部分在 RoMa 权重里，`pretrained=False`，没有另外下载 torchvision 的权重。用的是 outdoor 权重，不是 indoor（ScanNet 只许研究用）。Trained on MegaDepth; data provenance not audited. DINOv2 ViT-L Apache-2.0. The VGG part lives in the RoMa checkpoint (`pretrained=False`); no torchvision weights are fetched. Outdoor, not indoor (ScanNet is research-only). |
| pycolmap 4.2.1 | BSD-3-Clause（COLMAP、Ceres）；**PyPI 轮子含 GPL-2.0-or-later**（2026-10-06 核实）/ the PyPI wheel contains GPL-2.0-or-later (verified 2026-10-06) | `pycolmap-4.2.1-cp311-cp311-manylinux_2_28_x86_64.whl`（sha256 `7627f0ba…dd90f`）的 `_core.so` 静态链接 SuiteSparseQR 和 CHOLMOD Supernodal / MatrixOps。本笔记的 BA 数字是用它算的，记录不变；交付路线换成 `ba_scipy`（numpy / scipy），现场值差 ≤ 0.0004 cm，见 `geometry-backbone-ab-2026-10-06/README.md`。The wheel's `_core.so` statically links SuiteSparseQR and CHOLMOD Supernodal / MatrixOps. The BA numbers here were computed with it and stay as recorded; the shipped route uses `ba_scipy` (numpy / scipy), field values within 0.0004 cm. |
| OpenCV 4.10 headless | Apache-2.0 | — |
| MoGe-3 `moge-3-vitl@184008f` | MIT | 和 on-prem 审计一致。Same as the on-prem audit. |
| DA3-BASE `f4a6c9b` | Apache-2.0（HF 模型卡与 README 一致 / card and README agree） | — |
| map-anything-apache `00f9c24` | Apache-2.0 | 默认的 `map-anything` 是 CC BY-NC，不要用。The default `map-anything` is CC BY-NC. |
| VGGT-1B-Commercial `ebb29a5` | VGGT License（2025-07-29）+ AUP | 可商用，但需要人工审批（我们 403）。AUP 禁止军事、核、ITAR，以及"关键基础设施、交通或**重型机械的运行**"相关用途，要法务看。Commercial use allowed but gated (we got 403). The AUP excludes military, nuclear and ITAR use, and use related to the "operation of critical infrastructure, transportation technologies, or **heavy machinery**". Needs legal review. |
| `romatch` 未装的依赖 deps not installed | — | wandb / h5py / poselib / albumentations 只在训练代码里用，镜像里没装（`--no-deps`）。They are training-only and were not installed. |

## 复现 / Reproduce（Modal 只用 ephemeral 运行，retries=0，min_containers=0，没有部署任何东西 / ephemeral only, nothing deployed）

```
cd /Users/adam/.codex/worktrees/panoptes-workcell-photo-speed
modal run modal_apps/geometry_clean_ab.py --stage access          # VGGT 访问检查 → 403
modal run modal_apps/geometry_clean_ab.py --stage infer           # L4×2：RoMa 匹配 + MoGe-3（获批后加 --vggt）
modal run modal_apps/geometry_clean_ab.py --stage refine          # CPU：MoGe sim(3) + pycolmap BA，4 个起点 × {ba, ba-f}
modal run modal_apps/geometry_clean_ab.py --stage dense-infer     # L4：RoMa 稠密 warp
modal run modal_apps/geometry_clean_ab.py --stage mvs  --bases da3-base,mapanything,moge,pi3x    # 本机 local
modal run modal_apps/geometry_clean_ab.py --stage fuse --bases da3-base,mapanything,moge,pi3x    # 本机 local
modal run modal_apps/geometry_clean_ab.py --stage analyse         # CPU：公平 harness / fair harness
python research-notes/licence-clean-stack-2026-10-06/geometry/compile.py
```

- 草稿目录 / scratch：`scratchpad/checks/clean-gpu`（匹配、MoGe、稠密 warp / matches, MoGe, dense warps）、`clean-geom/CELL-NAME/geometry`（与 run 同格式 / run format）、`clean-analyse/*.json`。
- `vggt-access.json`：VGGT 访问检查的记录（只有错误类型，没有 token）。The VGGT access record (error type only, no token).

## 花费 / Spend

按 Modal 公开单价估算，不是账单。List-rate estimates, not an invoice.

| 项 item | USD |
|---|---|
| L4 匹配 + MoGe matches + MoGe | 0.019 |
| L4 稠密 warp dense warps | 0.009 |
| BA（CPU，16 个任务 jobs） | 0.019 |
| 分析 analysis（CPU，36 个任务 jobs） | 0.384 |
| **合计 total** | **≈ 0.43** |

- 镜像构建时间和 VGGT 访问检查没有计入。Image builds and the VGGT access check are not included.
- 复用的公平 A/B 结果已在该笔记里计过。The reused fair A/B runs were paid for in that note.

## 局限 / Limits

- 只有 4 个现场值、两个工位、5 张照片。0.1–0.3 cm 的差不能排名。
  Only 4 field values, two cells, five photos; a 0.1–0.3 cm difference cannot rank.
- Pi3X + BA 的 0.8–0.9 cm 不能直接用：权重仍是非商用，而且它是对照，不是新的发布几何。
  Pi3X + BA's 0.8–0.9 cm cannot be used directly: the weights are still non-commercial, and it is a control, not new published geometry.
- MVS 用的是 518 canonical 网格（立柱处 1 px ≈ 0.6–0.8 cm）和每张照片 1–2 个伙伴，覆盖率受限。立柱带里的点数是硬伤，不是精度问题。
  MVS runs on the 518 canonical grid (1 px ≈ 0.6–0.8 cm at the posts) with one or two partners per photo, which limits coverage. The point count in the post band is the blocker, not accuracy.
- MVS 和补洞的阈值是在看过前两轮干净候选的现场误差之后、算 MVS 之前定的，之后没有再调。三次尝试后按规则停止。
  The MVS and fill thresholds were set after seeing the first clean candidates' field errors but before any MVS number, and were not changed afterwards. Stopped after three attempts, per the rules.
