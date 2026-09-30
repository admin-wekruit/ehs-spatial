# Adversarial verification of x3-lingbot: holds-with-corrections

These corrections take precedence over results.json conclusions.

I re-derived every number from fx-x3-lingbot-001/002/004/005 and the code on fx/x3-lingbot (commit 6d19395). The arithmetic holds: stage times, written_s, core deltas (baseline = mean of ME340 runs 1 and 5, run 0 excluded; single baselines for Sam's and Walmart), fps, per-GPU whole-device peaks (50 ms mem_get_info sampling plus the worker's torch peaks), 0 flags over 90%, D8 metrics, share-within-5 cm, thickness and points per box all match the saved JSONs.

- **Clock:** time starts when the MP4 bytes are in the container, boot is excluded, and it covers decode, cuts, pre, infer, D7, points, align, pack and write. The clock is consistent between the core and the lane.
- **Inputs:** all runs used data/clips/{me340-165, samsclub-337, walmart-190}/source-full.mp4.
- **References:** runs/me340-lingbot-map-222, samsclub-a2 dense_gate/icp (samsclub-a2-lingbot-map-295-icp) and walmart dense_gate/icp (walmart-lingbot-map-268-icp) are the adopted runner decisions for publications c40fbd08, 913daf2a and 32cec650. I checked this through adopted/*.json and keys.jsonl, without opening .platform/imports.

The findings survive, with corrections:
- **Third GPU:** on a third A100 the core's times are unchanged (within noise). A gated dense layer lands at 61.3 s (ME340) and 43.6 s (Sam's).
- **Shared GPU 1:** vLLM naming about doubles there (8.0 to 16.0 s on ME340), so the core slows by 8-17 s.
- **Lane speed:** LingBot runs at about 8 fps, and the non-inference part of the lane takes about 10 s.
- **Stride:** stride 1 gains little over stride 2.
- **Fusion:** fusing LingBot into a TSDF loses agreement with the delivered layer.

The main problems:
1. **Timed and evaluated layers differ.** The headline times come from run B: first point rule (2.7M points), centres-only Sim3, no masks. The quality rows come from the untimed final-rule evaluation with person masks. As timed (no masks) the agreement is 67 / 57 / 70%, not 71 / 58 / 70%. Walmart never produced a passing timed layer.
2. **'About twice as well' is overstated.** The ratio is 1.15-2.2x. The metric runs one way only, and it is measured against a reference made by the same model.
3. **The renders show colour, not geometry.** They are same-viewpoint renders with source colours. No lettering is legible. The Sam's conduit also shows, blurred, in the DA3 TSDF. The ME340 hanging cable is missing from every layer.
4. **Scale labels are missing.** Every cm figure uses the estimated DA3 scale and is not labelled as such.

Smaller corrections:
- Run A was on an A100 PCIe card, not SXM4.
- The GPU-1 fps quoted for Walmart is from the withheld shot 0.
- ME340 had three core-alone runs, not two.
- The cut-away 'withheld' claim was never observed in a run.
- The densify cost leaves out moving full-resolution frames to the GPU.
- About $1.00 of the spend has no saved records.

I ran no GPU jobs (spend $0).

## Problems
- [major] The timed pipeline is not the pipeline whose quality is reported. The headline times were measured in run B (fx-x3-lingbot-002), which used the first point rule (par-03: 2,700,498 points, 43.2 MB layer against 421k / 6.7 MB for the final rule), the centres-only Sim3 and no person masks. Every quality row (D8 final rule, 71/58/70%, noise, points per box) was measured in the untimed evaluation (fx-x3-lingbot-005) with the final point rule, the orientation Sim3 candidate, ICP always, and person masks. Walmart has no passing in-run time at all. This is disclosed only in the issues list, and the table puts the times and the quality rows side by side as if they described one layer.
  Fix: Label the time rows 'first point rule, centres-only Sim3, no masks'. Re-time the final lane (final rule + orientation Sim3 + masks) in the 3-GPU container for all three videos: one boot plus 3 calls, about $1.5-3 at the run-B rate. Until then, give the final-rule time as an estimate and label it: lane-alone final-rule stage times plus align_s 3.9-5.9 s.
- [major] The 'LingBot s2 points (separate layer)' row (71% / 58% / 70%) and the noise and points-per-box rows use the person-masked evaluation variant (eval clouds lingbot_s2_points: 383,520 points on ME340). The timed lane has no masks. Its equivalent (lingbot_s2_points_no_people_mask) scores 67.25% / 56.86% / 70.12%, with thickness 4.82 / 3.65 / 4.12 cm. Meanwhile the 'Output points' row quotes the unmasked lane counts (421k / 858k / 1.01M), so the table mixes the two variants.
  Fix: Report the unmasked numbers as the timed lane's (67 / 57 / 70%). Show the masked variant as a separate row marked 'evaluation only, not wired in'.
- [major] 'Matches the full report's LingBot layer about twice as well as the fast DA3 TSDF' is overstated. The ratios are 2.16x on ME340, 1.16x on Sam's and 1.51x on Walmart (masked), or 2.04 / 1.15 / 1.51 as timed. The metric also runs one way only (share of our points near the reference; completeness never measured), so a conservatively filtered cloud scores high and DA3 is penalised for covering regions LingBot's filters drop. And the reference is a LingBot-Map output from the same weights at the same stride 2 (lingbot-run.json keyframe_interval 2). So the metric measures how well the delivered layer is reproduced, not geometric accuracy.
  Fix: Say '1.2x to 2.2x'. Add the reverse direction (share of the delivered layer within 5 cm of each cloud) as a completeness measure. State that the reference is the same model, so agreement is not accuracy.
- [major] The visual evidence is overstated. Checked in compare-*.jpg: each cloud is rendered from its own camera at the source frame, with per-pixel source colours. LingBot points therefore reproduce the image's texture (carton print, pegboard holes) whatever their depth accuracy, while the DA3 TSDF is a colour-averaged 3 cm mesh. So the panels mostly show colour resolution, not geometric detail. Specific claims fail: no carton lettering is legible in any LingBot panel (Sam's rows 2-3). The Sam's wall conduit is also visible, blurred, in the DA3 TSDF panel, so it is not 'visible only in LingBot'. The ME340 hanging cable (row 1, obj-1-86, frame 418) is missing from every layer, LingBot and delivered included. The ME340 vise and the shelf/box edges are sharper in LingBot, which is fair.
  Fix: Describe the renders as appearance at the capture viewpoint. To support 'LingBot keeps structure', render from a viewpoint offset by 20-30 degrees, or show depth- or normal-shaded renders. Drop 'lettering' and 'only in LingBot' for the conduit.
- [major] The cm values are quoted without the 'estimated' label: 5 cm and 25 cm agreement, floor offset, patch thickness, ICP step move. The DA3 metre frame is scaled from the floor plane and an assumed 1.6 m camera height (results.json 'policy' and the code's LABELS), which is not a measured scale. The ME340 floor offset of 4.57 cm against the 5 cm gate is inside plausible scale error.
  Fix: Label every cm value in the table and conclusion 'estimated'. Note that the gate margins (ME340 floor offset 4.6 of 5 cm) depend on that scale.
- [minor] The 2x2 densify cost is quoted as 0.03-0.09 s, but that is the kernel only. It needs the full-resolution frames on the GPU. The evaluation measured decode + upload at 2.4-11.2 s. In the lane the frames are already decoded in RAM, so the real extra cost would be the host-to-GPU upload, and that was never measured.
  Fix: Quote the kernel time plus an 'upload unmeasured' note, or time it inside the lane.
- [minor] The issue note on card type is wrong. fx-x3-lingbot-001/boot.json records 'NVIDIA A100 80GB PCIe' for run A, not SXM4. Only run B (the 3-GPU parallel run) was on SXM4. So the lane-alone runs (PCIe) and the third-GPU runs (SXM4) differ in card type, not run A and run A4.
  Fix: Correct the note: runs A, A4 and the compile test were on PCIe; run B was on SXM4.
- [minor] The GPU-1 fps range '4.3 to 5.5' includes Walmart's 4.99, which is shot 0, the one D7 withheld. Walmart's main shot 1 ran at 7.82 fps on GPU 1 because the core had finished by then. Likewise results.json parallel_lingbot_fps_main_shot gpu2 lists Walmart 7.68, which is shot 0; the main shot ran at 7.81.
  Fix: Use the main shot's fps: Walmart GPU1 7.82 and GPU2 7.81. The GPU-1 slowdown then applies only to ME340 (5.5) and Sam's (4.33).
- [minor] 'ME340 has two core-alone baselines': there are three (par-00, par-01, par-05). Run 0 is excluded as the first call after boot, which task4_summary documents. Against run 5 alone the delta is -0.6 s; against all three it is -1.9 s. Both are still noise, so the conclusion stands.
  Fix: State three baseline runs, one excluded, and give the range of deltas.
- [minor] 'The ME340 cut-away shot is correctly withheld' was not observed in any run. In run B the cut-away (path 6 cm, Sim3 scale 0.49 against 5.1, gate use None) was still written in LingBot's own frame (dense v2, 56.9 MB, 75.1 s). results.json alignment_in_lane_run_B says 'withheld', which is wrong for that run. Withholding comes only from the final code, and the final code was not run on that shot.
  Fix: Say 'fails the gate; the final code would withhold it (not run)'. Correct results.json accordingly.
- [minor] Spend of $6.62 is only partly checkable. Run A ($0.63), run B (calls $2.55 plus boot, consistent with $3.12), A4 (about $0.66) and the final evaluation (1,096 s of evaluation wall time, about $1.22) re-derive from the saved call wall times. About $1.00 of smoke, compile, debug and failed-evaluation items have no saved records.
  Fix: Record wall times for those calls, or mark that $1.00 as unverified.

## Corrected table

| Measure. Time (s) runs from video bytes in the container to the layer written. Boot is excluded. One run per setting. All cm values use the estimated DA3 scale (floor plane plus an assumed 1.6 m camera height). | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| **Third GPU** (3 x A100 SXM4 container, run B), stride 2: core objects.v2 delta against the core-alone mean (ME340 baseline = runs 1 and 5; run 0 excluded as the first call after boot) | -1.0 s (noise: -0.6 s against run 5 alone, -1.9 s against all 3 runs) | -1.6 s | -2.0 s |
| **Third GPU**, stride 2: first dense layer in the DA3 frame that passes the gate. Timed with the FIRST point rule (2.7M points, 43 MB), the centres-only Sim3 and no person masks. This is not the layer evaluated below. | **61.3 s** | **43.6 s** | no passing layer in the run: ICP step 1 was 2.60 deg, over the 2 deg cap, so it was refused. The shot was still written in LingBot's own frame at 60.6 s. The final rule was not re-timed. |
| Shared GPU 0 (the DA3 GPU), stride 2: core delta / layer / LingBot fps | -0.7 s / 74.4 s / 6.3 fps | not run | not run |
| Shared GPU 1 (the SAM 3 + vLLM GPU), stride 2: core objects.v2 delta / layer | +8.2 s / 82.9 s | +16.7 s / 65.3 s | +12.0 s / gate failed (layer written unaligned at 76.7 s) |
| LingBot fps on the main shot, shared GPU 1 | 5.5 | 4.33 | 7.82 (the 4.99 quoted is for shot 0, which was withheld) |
| Peak whole-device GPU memory when shared (sampled every 50 ms) | GPU1 67.2 GB, GPU0 67.1 GB (79% of 85.1) | GPU1 66.9 GB | GPU1 67.0 GB |
| Flags over 90% | 0 | 0 | 0 |
| LingBot alone on 1 A100 80GB **PCIe**, stride 2, final point rule. LingBot's own frame: no alignment, no masks. First layer with points / all shots | 51.2 / 63.6 s | 36.9 / 56.9 s | 56.4 / 56.4 s (shot 0 was inferred first, about 25 s, then withheld by D7) |
| LingBot alone, stride 1 | 94.0 / 123.5 s | 62.9 / 103.1 s | 102.2 / 102.2 s |
| LingBot fps on the main shot, lane alone (s2 / s1) | 8.09 / 7.96 | 7.50 / 7.90 | 7.68 / 8.08 |
| Peak memory, lane alone | 15.1-16.4 GB whole device; worker torch allocation 12.9-14.3 GB | | |
| Output points on the main shot, timed lane without masks (s2 / s1); delivered LingBot layer | 421k / 447k; 2.91M | 858k / 817k; 3.51M | 1.01M / 1.05M; 3.35M |
| D8 gate, final rule (orientation or centres Sim3, then ICP). Evaluation only, on person-masked points. Share within 25 cm / floor offset (estimated) | 0.907 / 4.6 cm | 0.911 / 2.4 cm | 0.952 / 1.8 cm |
| Share of each cloud's points within 5 cm (estimated) of the delivered LingBot layer. One direction only; completeness not measured; the reference is itself a LingBot output. DA3 TSDF 3 cm | 33% | 50% | 46% |
| same: LingBot s2 points **as timed (no person masks)** | **67%** | **57%** | **70%** |
| same: LingBot s2 points with person masks (evaluation only, not in the timed lane) | 71% | 58% | 70% |
| same: LingBot s1 points with masks | 73% | 63% | 73% |
| same: LingBot-only TSDF at 1.5 cm | 65% | 50% | 69% |
| same: DA3 + LingBot fused TSDF at 1.5 cm | 35% | 43% | 53% |
| Ratio, LingBot s2 as timed / DA3 TSDF | 2.0x | 1.15x | 1.5x |
| Noise: median patch thickness at 10 cm (estimated). Delivered / s2 masked / s2 unmasked (timed) / s2 without the one-layer pass | 4.4 / 4.6 / 4.8 / 8.5 cm | 5.0 / 3.6 / 3.7 / 9.2 cm | 3.6 / 4.0 / 4.1 / 6.6 cm |
| Median points per EHS object box: DA3 TSDF / LingBot s2 (masked) / s2 with 2x2 samples / delivered | 1714 / 1738 / 6947 / 18341 | 1230 / 3597 / 14392 / 8574 | 1135 / 6434 / 25758 / 16741 |
| 2x2 densify cost: the kernel alone. Moving full-resolution frames to the GPU was not measured in the lane; the evaluation's decode + upload took 2.4-11.2 s | 0.06 s (s2) | 0.06 s | 0.03 s |
