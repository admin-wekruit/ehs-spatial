# Adversarial verification of x4-refine: holds-with-corrections

These corrections take precedence over results.json conclusions.

The experiment's core numbers check out, but several of its conclusions need correcting. I recomputed every median and count from fx-x4-refine-006/results.json, the three per-video JSONs, the stage logs, run 005, the code at 9a98198 (the version run 006 used) and the Modal billing report. No GPU jobs were run and verification spent $0.

**What holds:**
- Every per-spot median and every sweep median matches exactly.
- 0 of 15 spots are confirmed, all 15 fail the multi-view check, and the other failed-check counts, the VLM answer counts and the trigger counts all match.
- Inputs were the same across all settings: every variant used the same source-full.mp4 clips as the harness bench.
- Cold start is excluded consistently: model warm-up happens at boot for every resolution.
- The refine GPU's whole-device memory peak, measured by the NVIDIA driver (NVML), is real: 47.95 / 48.28 / 48.3 GiB during the coarse sam3.person stage, at most 41.68 GiB during refine steps, and none over 90%.
- Looking at the images supports some of the stated detail: the coarse map invents octagon patterns from the Puffs box print (Sam's s1), and the fine map shows the shelf beam's lip (Sam's s2), the pallet's blocks (Walmart s0) and a wire-like ridge (ME340 s0).

**What needs correcting:**
1. **Spot finding is much slower than claimed.** It took 2.5 / 32.1 / 12.5 s per video, not ~2.5 s, so the refine cost after the coarse pass is 15–44 s per video, not 12–15 s. In the repeat run it was 27–70 s.
2. **Per-spot time leaves out a required step.** The metrics step feeding the confirmation verdict is not counted. With it, the median is 2.38 s, and the repeat run took 3.3–5.3 s per spot, so the 3 s budget is not supported.
3. **The tilt "repeatability" result is not about the maps.** The coarse and fine maps were identical in runs 005 and 006. The 5.8° vs 0.49° difference comes from the RANSAC plane fit reacting to point order. What it actually shows is that X4's tilt estimator is unstable, by up to 10.7° on identical points, not that the fast core's tilt is unreliable.
4. **The density gains follow from the settings.** More points and finer spacing come from the 1 cm voxel size, and the fine map covers only about half the coarse surface, with gaps on about 7 of 15 spots.
5. **No comparison with the delivered reports.** Nothing was compared with c40fbd08 / 913daf2a / 32cec650, and the merged harness was never called.
6. **The LingBot quality numbers are not a fair comparison.** They were measured after ICP moved LingBot's points by up to 2.8 m, and the fine method got a per-view scale from the coarse map that LingBot did not. LingBot's alignment was reliable on 2 of 15 spots, not 3. The finding that it is slower still stands.
7. **Several table values are wrong.** Methods disagree on tilt by up to 22.7°, not 8.5°, and LingBot's tilt interval width is 0.06–1.27°, not 0.2–0.9°.
8. **Spend is $4.22 billed, not $3.46.** Run 005 cost $1.44 and run 006 $0.75. This is still under the $8 cap.

Files: /Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/fx-x4-refine-006/results.json and /Users/adam/.codex/worktrees/panoptes-phase2-video-fx-x4-refine/modal_apps/x4_refine.py. Spot finding is at lines 726-820, the per-spot total that leaves out metrics at 1421-1423, and the ICP cap at 1218-1223.

## Problems
- [major] The spot-finding time is understated. It is reported as '~2.5 s' per video and the conclusion says 5 spots cost '12–15 s per 30 s video'. The logged refine.spots stage took 2.5 / 32.1 / 12.5 s in run 006 (5.1 / 54.3 / 23.7 s in run 005). Sam's Club has 7537 detections in 1391 clusters, and find_spots clusters them in a Python loop that scales with detections times clusters. The real refine cost after the coarse pass is 15.3 / 44.2 / 24.2 s in run 006 and 27.2 / 69.5 / 42.8 s in run 005.
  Fix: Report find_spots for each video and the per-video refine total (find spots + spots + metrics). Replace the naive clustering, for example with the fast core's lift+merge, before claiming a ~15 s overhead.
- [major] The repeatability claim is misread. Coarse 5.8° vs fine 0.49° is presented as map repeatability, and the issues say coarse tilt 'matters for any tilt fact the fast core reports'. In fact both maps are identical between runs 005 and 006: same coarse point counts (for example 193803/859880), same points in each box, same fine points and raw pixels, same per-view scales, same multi-view medians. Only the RANSAC inlier counts differ (for example 294 vs 277). The tilt differences come from the RANSAC plane fit reacting to point order on identical geometry. The fast core's own geometry is not shown to be unstable.
  Fix: Say that run 005 was a deterministic re-run and gives no evidence about reconstruction repeatability. Report instead that X4's tilt estimator moves by up to 10.7° (coarse) and 8.6° (fine) on identical points, and that its bootstrap CI misses this. Make plane selection stable (fixed point order, more iterations, or a deterministic face choice) before any tilt claim.
- [major] The per-spot time leaves out the refine.metrics stage: pairwise multi-view check, coarse agreement, plane tilt with bootstrap, and thin-object height. The confirmation verdict needs all of these. The stage takes a median 0.14 s (max 0.25) in run 006 and 0.21 s (max 0.37) in run 005. With it, per-spot time is 2.38 s median (2.11–2.96) in 006 and 3.32–5.26 s in 005. The recommended '3 s' budget was exceeded by all 14 spots in the repeat run, whose container was slower throughout (coarse pass 1.3–1.4x, VLM 2.5x).
  Fix: Include metrics in the per-spot time and budget about 5 s per spot, or show why the slower host in run 005 is not representative.
- [major] The density figures (5906 vs 832 points, 4.4 vs 14.5 mm spacing, 6.0 vs 10.7 mm footprint) follow from the settings: 1 cm vs 3 cm voxels, and a 720 px crop vs the 1280 px frame, both at 504 px. The fine/coarse point ratio per spot has a median of 4.7, where equal coverage would give about 9. So the fine map covers roughly half the coarse surface in the box (inferred). The before/after images show broken-up fine maps on ME340 s1/s3/s4, Sam's s4 and Walmart s1/s3/s4. The conclusion lists only the gains.
  Fix: Add a coverage measure (the share of the coarse box surface with a fine point within 1–2 cm). State that density comes from the voxel size and that the fine map covers less area on about half the spots.
- [major] There is no comparison with the delivered reports c40fbd08 / 913daf2a / 32cec650. fb/d-harness was merged but never called (no recall, no ATE vs DROID). 'Finer' is judged only against X4's own single-GPU coarse re-run and a per-span LingBot run. So nothing shows that the fine spots approach the detail of the delivered demo the user wants to match.
  Fix: Compare each spot's geometry with the delivered reports' LingBot maps at the same place, and run the harness metrics on the patched map.
- [major] The LingBot quality comparison is not fair. The fine method gets a per-view scale from the coarse map; LingBot gets one Sim3 scale. Without that per-view scale, DA3 scores 9.7 cm, worse than LingBot's 7.8 cm. LingBot's coarse agreement (2.9 cm), points (8507), spacing, tilt and display-gate share (0.82–1.00) were all measured after ICP corrections of up to 2.82 m / 21.2°, and ICP minimises exactly that distance. 'Alignment reliable' is 2/15, not 3/15: Sam's s0 counts only because ICP had fitness 0 with 0 points in the box. The conclusion that LingBot is slower still stands.
  Fix: Re-run the LingBot rows with the 10 cm / 3° cap (commit 4d062c1) and the same per-view scale, or drop the LingBot quality numbers and keep only the timing.
- [major] Spend is understated. The Modal billing report for today now shows $4.22 for the X4 apps: 005 is $1.4357 (claimed $0.87, which was read before the hour closed) and 006 is $0.7476 (claimed about $0.55, from a $0.00106 per second rate; the actual rate is $0.00144 per second). Still under the $8 cap.
  Fix: Report $4.22 billed.
- [minor] 'Coarse, fine and LingBot tilt disagree by up to 8.5°' is wrong. On ME340 s2 fine says 3.4° and LingBot 26.1°, a 22.7° spread. Coarse vs fine alone reaches 15.6° (ME340 s1: 16.5 vs 0.9). Box spots with tilts of 29.9–37.8° show that the 'dominant near-vertical plane' is not the box face, so tilt means nothing there.
  Fix: Correct the spread and exclude spots where the fitted plane is not the surface.
- [minor] The LingBot tilt CI range is 0.06–1.27°, not 0.2–0.9°. 'Identical frames → same' tilt is false for ME340 s4 (16.0 to 28.1°) and s3.
  Fix: Correct the table.
- [minor] The outline IoUs do not compare like with like. The fine outline comes from SAM 3 masks on the other crops, lifted through the fine depth. The coarse outline comes from the trigger cluster's union points (2 cm voxels, capped at 1.5 m, drawn with a wider radius). The re-projected IoU is scored on a median of 9 keyframes that have a SAM 3 mask (range 3–23); the '52 keyframes' only received drawn outlines, which were never checked.
  Fix: Use the same mask source for both, or label the comparison as 'object-specific points vs cluster union'. State how many keyframes each IoU was scored on.
- [minor] The coarse-agreement check (fine within 2.0 cm of coarse) is partly guaranteed: each view's depth scale is set from the coarse map around the spot, so the check passed 15/15.
  Fix: Say that this check cannot fail under the anchor method.
- [minor] LingBot's 'peak_gib' is memory allocated by PyTorch on its own GPU, not the whole-device per-stage peak from the NVIDIA driver (NVML) that the units line claims. Its 13.3 fps counts inference only, on spans of 66 frames or fewer, and cannot be compared with the 7.1 fps measured earlier on a full A100 video.
  Fix: Relabel both, or log LingBot's GPU with NVML per stage.
- [minor] The unposed method is described as giving 'no fine points', but 4 of 15 spots produced points (up to 9105; the median is 0), and several spots with 0 fused points still have multi-view samples. A fusion problem is as likely as a property of the method.
  Fix: Report 4/15 and call it unexplained.
- [minor] The conclusion states mm and cm values without calling them estimated (26.5 mm wire, 4.4 mm, 4.6 cm, 1.5 cm rule). The '~4 cm coarse camera error' given as the cause comes from ME340's earlier fast-core run and was not measured here. The note 'wide crops 1.00–1.10' conflicts with run 006's per-view scales of 0.33–1.44.
  Fix: Label all lengths as estimated, mark the cause as inferred, and update the note on wide-crop scales.
- [minor] '4–6 views' is recommended, but 6 views were never tested (14 spots had 4 views, 1 had 5). The claim that the occlusion drop 'caught a person and a box' cannot be checked in run 006, which has one occlusion drop (ME340 s4) and one 'spot not seen'.
  Fix: Recommend 4–5 views and cite the run where the person and box drops happened.

## Corrected table

| Measure (run 006, 15 spots, medians unless noted) | coarse 3 cm | fine: DA3 wide crops 504 + per-view scale + 1 cm TSDF | LingBot span |
|---|---|---|---|
| Time per spot, one A100 (s) | – | Claimed 2.29, but that leaves out the metrics step the verdict needs (median 0.14 s). With it: **2.38** (range 2.11–2.96). Repeat run 005: **3.32–5.26**; all 14 spots over 3 s | 4.06 inference only, plus 0.10–0.27 prep and 0.11–0.52 align. 13.3 fps is on spans of 66 frames or fewer, so it cannot be compared with the 7.1 fps full-video figure |
| Finding the spots, per video (s) | – | **2.5 / 32.1 / 12.5** (ME340 / Sam's / Walmart), not "~2.5". Run 005: 5.1 / 54.3 / 23.7 | – |
| Refine cost per 30 s video after the coarse pass (s) | – | **15.3 / 44.2 / 24.2** (find spots + 5 spots + metrics), not "12–15". Run 005: 27.2 / 69.5 / 42.8 | – |
| Coarse pass, 1 GPU, excluded (s) | 37.7 / 30.8 / 27.9 (run 005: 52.9 / 41.8 / 38.7) | | |
| Points in spot box | 832 | 5906. The gain comes from the 1 cm vs 3 cm voxel size. Fine/coarse ratio is a median 4.7, against about 9 for equal coverage, so fine covers roughly half the coarse surface (inferred). Fine map is broken up on about 7 of 15 tiles | 8507 raw pixels, not TSDF points; counted after an ICP correction of up to 2.8 m |
| Point spacing (mm, est.) | 14.5 | 4.4 (set by the voxel size) | 7.4 (after the uncapped ICP) |
| Pixel footprint at best view (mm, est.) | 10.7 | 6.0 (set by the crop scale) | – |
| Multi-view median abs dz (cm, est.) | – | 4.6 (4.1 with 10 cm occlusion cut). Rule is 1.5, so 0/15 pass | 7.8, with a single Sim3 scale. DA3 without the per-view scale scores 9.7, worse than LingBot |
| Agreement with coarse, NN median (cm, est.) | – | 2.0. Partly guaranteed, because each view's scale is set from the coarse map (raw DA3: 7.0) | 2.9, measured after ICP corrections of up to 2.82 m / 21.2°, so not valid |
| Held-out photometric (fine/coarse) | 1 | 0.994 (fine better 8/13) | not run |
| Held-out outline IoU vs SAM 3 | 0.41 (spot cluster points) | 0.56 (fine better 10/15). Not like-for-like: fine uses SAM 3 masks on the other crops | – |
| Re-projected IoU vs SAM 3 | 0.32 | 0.42, scored on the keyframes that have a SAM 3 mask (median 9, range 3–23), not on the 52 keyframes that got outlines | – |
| Tilt diff between runs 005 and 006 (deg) | 5.8 | 0.49 (max 8.6). **Not a repeatability measure**: both maps are identical across the runs. The difference is the RANSAC plane fit reacting to point order | changed on ME340 s4 (16.0 to 28.1) |
| Tilt CI95 width (deg) | 0.2–4.5 | 0.09–1.8 (misses estimator swings of up to 10.7°) | **0.06–1.27** (claimed 0.2–0.9) |
| Tilt spread across methods, same spot | up to **22.7°** (ME340 s2: fine 3.4, LingBot 26.1). Coarse vs fine up to 15.6° (ME340 s1). Claimed "up to 8.5°" | | |
| Confirmation | – | 0 CONFIRMED / 15 NEEDS_REVIEW; 6/15 fail only the multi-view check | alignment reliable **2/15**. The third claimed spot (Sam's s0) had ICP fitness 0 and 0 points in the box |
| GPU peak (GiB) | refine GPU, measured by the NVIDIA driver (NVML): 48.3 / 80 during coarse sam3.person; refine steps max 41.7 including vLLM; 0 over 90% | | 8.9 median / 10.3 max, as memory allocated by PyTorch only, not a whole-device NVML peak |
| Spend (USD) | billed **4.22** (005: 1.44, 006: 0.75), not 3.46. Crash loop 0.52. Under the $8 cap | | |
| Delivered reports c40fbd08 / 913daf2a / 32cec650 | **not used**: fb/d-harness was merged but no recall, ATE or spot-detail comparison with the delivered reports | | |

| Sweep, run 006, medians over 15 spots (all re-derived exactly) | fine s | multi-view cm | coarse NN cm | photometric | outline IoU | fine pts |
|---|---|---|---|---|---|---|
| wide / anchor / 504 / all views | 0.70 | 4.6 | 2.0 | 0.994 | 0.56 | 5906 |
| wide / anchor / 336 | 0.51 | 4.0 | 2.1 | 0.966 | 0.53 | 5105 |
| wide / anchor / 756 | 1.18 | 4.9 | 2.5 | 0.963 | 0.56 | 5858 |
| wide / anchor / 504 / 3 views | 0.56 | 3.9 | 2.3 | no hold-out | – | 2806 |
| tight / anchor / 504 | 0.69 | 7.0 | 2.8 | 1.006 | 0.47 | 5368 |
| wide / raw posed | 0.68 | 9.7 | 7.0 | 1.048 | 0.47 | 2582 |
| wide / anchor + per-view ICP (14 of 44 corrections applied) | 0.89 | 4.2 | 2.1 | 1.011 | 0.56 | 5657 |
| wide / DA3 unposed + Sim3 | 0.87 | 4.5 | 3.7 | 1.031 | 0.00 | median 0 (4 of 15 spots have points, up to 9105) |
