# Adversarial verification of x11-local-ba: holds-with-corrections

These corrections take precedence over results.json conclusions.

The numbers hold, but the headline needs corrections.

**What I re-derived**
- Every cell of the per-spot table matches results.json.
- Re-running summarise() on the saved per-video JSONs reproduces the aggregate, per-video and tilt-repeatability blocks exactly.
- X4's 15 spots come back with the same centres and exactly the same multi-view values (for example ME340 s0 2.313 cm).
- Both self-checks pass locally: `local_ba.py --self-check` and `x11_local_ba.py --self-check`.
- GPU peaks are real (nvidia-smi sampled at 20 Hz): 48.97 GiB (61%), nothing over 90%.
- Cold start is reported separately (ready at 110 s).
- Licences are correct against the primary sources: ALIKED BSD-3-Clause, LightGlue Apache-2.0 (its README says ALIKED is BSD-3), kornia Apache-2.0, threadpoolctl BSD-3-Clause, MASt3R CC BY-NC-SA 4.0, SuperPoint non-commercial research.
- The 8 commits exist on fx/x11-local-ba with the required trailer, and nothing is pushed.
- Spend is $6.18 billed across the 8 apps, under the $10 cap.

**What needs correcting**
- The single confirmation does not repeat across frame choices. Sam's s1 passes only on frames A; Sam's s7 passes only on frames B, and only just. No spot is confirmed on both frame sets.
- On frames B, X4's own method already has 2 spots within 1.5 cm, the same as BA.
- BA alone is not significant per spot on frames A (18 better, 11 worse, p = 0.27). BA+DA3 is robust (25 better, 4 worse, p = 0.0004).
- The tilt gains are an artifact of comparing different spot sets. Paired on the 14 shared spots: 1.72 / 1.74 / 1.20°, with BA+DA3 better on 7 and worse on 7 (p = 0.5).
- 'Distance matters' is really the 640x480 factory clip.
- 'BA fixed the cameras' rests on fitted residuals (the held-out resection is also a fit). The independent held-out outline IoU got worse, 0.555 -> 0.501.
- Methods were tuned in runs 004-007 on the same 31 spots used for the result, and no negatives were tested.
- Smaller errors:
  - The factory range is 5.1-29.8 cm, not 10-30 cm.
  - The spot-finder baseline was re-timed on a host about 1.5x slower than X4's own run. The speed-up against X4's recorded times is 4.7-7.9x.
  - Per-spot time leaves out the metrics stage and includes about 0.5 s of BA diagnostics.
  - The confidence-interval miscalibration is about 12-30x, not 3-7x.

**Other checks**
- My look at the 4 agent labels agrees on the substance. The one wording difference: ME340 s1 sits on a workbench, not on a machine.
- The delivered references c40fbd08 / 913daf2a / 32cec650 are not cited by X11 and could not be checked from this machine.

**Files**
- Results: /Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/fx-x11-local-ba-008/results.json and the four per-video JSONs in that folder.
- Code: /Users/adam/.codex/worktrees/panoptes-phase2-video-fx-x11-local-ba/modal_apps/{local_ba.py, x11_local_ba.py, x4_refine.py}.

## Problems
- [major] Confirmations don't repeat across frame sets. Sam's Club s1 is confirmed on frames A (BA / BA+DA3 / BA+DA3+tri) but fails on frames B with every method: held-out outline IoU 0.47-0.48 against the 0.5 threshold. Sam's Club s7 passes only on frames B with BA+DA3, and only just: mv 1.40 cm (limit 1.5), fine-to-coarse 4.93 cm (limit 5), VLM 'loose... misses parts of the box'. It fails on frames A with mv 1.96-2.41 cm. No spot is confirmed on both frame sets by any method. On frames B, X4's own anchor method already has 2 spots within 1.5 cm (ME340 s4 at 1.20 cm, Sam's s1 at 1.35 cm), the same count as BA. ME340 s4 goes from 8.1 cm on frames A to 1.2 cm on frames B with no BA at all, so frame choice moves the metric more than BA does. The claim presents s7 on frames B as extra support and reports the 0 -> 2 -> 3 -> 3 count only for frames A.
  Fix: Headline: '1/31 on frames A, 1/30 on frames B, 0 spots on both'. Report the frames-B counts within 1.5 cm (2 -> 2 -> 3). State that frame selection is a variance source at least as large as BA's gain.
- [major] The tilt repeatability medians come from different spot sets (19 / 16 / 14 spots). On the 14 spots common to all three methods, face-matched |A-B| is anchor 1.72°, BA 1.74°, BA+DA3 1.20°. BA+DA3 beats anchor on 7 spots and loses on 7 (Wilcoxon p = 0.50), and 5/5/5 are within 1°. So the claimed improvement from 2.13° to 1.74° with BA is an artifact of the subsets, and 'BA + DA3 halves the anchor spread' is not supported.
  Fix: Report paired numbers on the common spots, 1.72 / 1.74 / 1.20° (7/7, p = 0.5), and drop the 'halves' claim.
- [major] 'Distance matters' is confounded with the factory clip. 7 of the 12 spots beyond 6 m are the 640x480 factory clip at 9-19 m. The 5 non-factory spots beyond 6 m have a BA median of 2.9 cm, no worse than 3.66 cm under 3 m or 3.29 cm at 3-6 m. The quoted '2.9 / 2.1-2.2 / 9-12 cm' mixes methods: tri under 3 m, BA+DA3 or tri at 3-6 m, tri-to-BA beyond 6 m. Spots under 3 m are worse than 3-6 m for every method, which contradicts 'DA3 error grows with depth'.
  Fix: Give one method per bin, split the factory clip out, and state that these data show no distance trend on the 720p videos.
- [major] 'BA fixed the cameras / poses are no longer the limit' rests mostly on in-sample residuals.
- Reprojection 1.00 -> 0.32 px is the solve's own residual. The 'before' case also keeps the frozen DA3 depth terms, which pull the points away from pure triangulation.
- 'Held-out view reprojection 4.96 -> 0.82 px' is the residual of a 6-DOF resection fitted to those same 2D-3D pairs (local_ba.resect), not a held-out test.
- The one independent check, held-out outline IoU, got worse with BA: median 0.555 -> 0.501, failures 14 -> 15.
- Per spot on frames A, BA alone beats anchor on 18 spots and loses on 11 (Wilcoxon p = 0.27, not significant). Only BA+DA3 is robust (25/4, p = 0.0004).
- After BA alone, the systematic offset between view pairs (1.8 cm) is about as large as the spread within pairs (2.0 cm). 5 spots have cameras at the ±3°/±10 cm bound, and the pose prior pulls towards the coarse cameras.
- The decomposition cannot separate remaining pose error from DA3's low-frequency depth error.
  Fix: Label the pose and DA3 attribution as inferred. Call the resection error a fit residual. Report the held-out outline IoU and the paired significance. Say that BA alone is not significant on frames A.
- [major] There is no held-out spot set, no negatives and no ground truth.
- Runs 004-007 tuned the solver on the same 31 spots used for run 008: baseline-scale bound, outlier-view drop, start-point filter, and the choice of BA+DA3 as the method.
- No known-wrong geometry was run through the rule to show that BA or +tri does not make it pass.
- BA fits per-view depth scale and shift to agreement near the spot, which is the statistic the multi-view rule then checks. +tri bends the depth onto track depths that were themselves solved together with the DA3 depth term.
- So 'confirmed' means internally consistent, not validated.
- The confirmed count stayed at 1 across runs 006-008, so the leakage mostly affects the medians and the 'default' recommendation.
  Fix: Call the result a development-set result. Before adopting BA+DA3 as the refine default, test on new spots or clips and include hard negatives, for example deliberately shifted or tilted surfaces.
- [minor] Several small factual errors.
- 'Factory s0-s7: 10-30 cm with every method' is wrong: BA+DA3 gives 5.1, 8.1 and 9.8 cm on s5, s6 and s2 (tri: 5.4 and 8.2), for a range of 5.1-29.8 cm.
- Factory s3 has no multi-view value with any method, so the 'all 31 spots' medians are over 30.
- The refused factory s1 record carries a pose change that was never applied (156 cm median, 263 cm max).
  Fix: Say 5-30 cm and give the medians as n = 30. Flag or omit the pose change on refused solves.
- [minor] The spot-finding baseline is not X4's recorded time. X4's function re-timed in this container took 4.0 / 48.1 / 19.3 s, against 2.54 / 32.1 / 12.5 s in X4's own run 006 (about 1.5x slower host). Only the fast finder ran with one BLAS thread. Against the recorded X4 times the speed-up is 4.7x / 7.9x / 6.6x. 'Identical picks incl. triggers' is only partly checked on real videos: the runtime check compares group, word, keyframe count and centre, and triggers are compared only in the synthetic self-check.
  Fix: Label the baseline 'X4 function re-timed here'. Also time X4 with one BLAS thread, or state the difference. Compare triggers at runtime.
- [minor] Timing does not include every step, and the BA overhead is overstated. total_without_sweeps (inherited from X4) leaves out the metrics stage: the multi-view and coarse-agreement computation, median about 0.15 s per spot. It includes about 0.48 s per spot of BA diagnostics that are not part of the product (the before-solve and the consistency diagnostics, run in both BA calls). BA's real overhead over X4 is about 0.6 s per spot, not 1.05 s. The X4 method here took 3.24 s against 2.3 s in X4's own run.
  Fix: Add the metrics stage to the total, leave the diagnostics out, and note the host difference.
- [minor] The confidence-interval miscalibration factor is computed wrongly. A 95% interval width (about 3.9σ) was compared with the median |A-B|, which is about 0.95σ for two independent estimates. Calibrated intervals 0.29-0.39° wide predict a median |A-B| of about 0.07-0.10°. The observed 1.2-2.1° means the understatement is about 12-30x, not '~3-7x'. The face-matched |A-B| also selects frames B's points inside frames A's ±3 cm slab, which biases B towards A.
  Fix: Restate the factor as about 12-30x (the direction was right) and note the selection bias.
- [minor] The 'triangulation floor 0.6 cm' is a best case: it assumes the maximum baseline and 0.5 px. The 'DA3 misses the triangulated tracks by 2.1 cm' figure uses track points solved together with the DA3 depth term, not pure triangulation.
  Fix: Call it a best-case floor and describe the track depths as coming from the joint solve.
- [minor] Metric claims: cm values are marked 'estimated' only in the units block and the issues list, not in the conclusion text. Across solved spots the baseline scale ranges from 0.84 to 1.24, so the scale between trajectory and depth is itself uncertain by up to about 24%, yet the 1.5 cm threshold is applied as an absolute value.
  Fix: Mark the cm values as estimated in the conclusion and mention the scale uncertainty next to the absolute threshold.
- [minor] Spend and bookkeeping. 8 X11 apps are billed $6.18 now (run 008: $1.470), against the claimed $5.91 at read time and $6.3 estimated, all within the $10 cap. The mapping of runs 003-005 to apps does not follow app creation order: the $0.1625 app was created before 22:43, so it cannot be run 005. The delivered references c40fbd08 / 913daf2a / 32cec650 are not cited by X11, which compares only against X4 run 006. They are publications in the platform database and cannot be checked from this machine.
  Fix: Report $6.18 final and map apps to runs by creation time. For c40fbd08 / 913daf2a / 32cec650, add a note that X11 does not use or cite them.
- [minor] My audit of the agent labels on the contact sheets agrees with all 4 on the substance.
- Sam's s1: stacked Puffs lotion-tissue boxes. The VLM called it 'Pampers wipes' and still passed, because the check only tests 'box'.
- ME340 s1: not a pallet. It is black cast fixture blocks with windows, but they sit on a workbench, not 'on a machine' (partial disagreement on wording).
- ME340 s2: a horizontal ledge or band on a Haas machine front.
- ME340 s5: no wire visible. The outline sits on a light roll-up door or wall behind the machine.
The sheets are 224 px thumbnails, which is enough to judge what the object is but not fine detail.
  Fix: Change ME340 s1 to 'black fixture blocks on a workbench'. Note that the VLM misnamed the brand on the confirmed spot.

## Corrected table

Run of record fx-x11-local-ba-008. Every cell of the claimed per-spot table matches results.json. I rebuilt it from the per-spot records, and re-running summarise() on the saved <clip>.json files reproduces 'aggregate', 'videos' and 'tilt_repeatability' exactly. The per-spot table stands as claimed (15 X4 spots; one A100 with vLLM on the same card). What changes is the summary lines below. Scale is 'estimated' throughout.

**All spots on frames A** (anchor -> BA -> BA+DA3 -> BA+DA3+tri)
- Multi-view median: 7.28 -> 4.08 -> 3.84 -> 3.22 cm.
  - This is over 30 spots, not 31: factory s3 has no multi-view value with any method.
  - Without the factory clip (23 spots): 5.25 -> 3.29 -> 3.14 -> 2.75 cm.
- Spots within 1.5 cm: 0 -> 2 -> 3 -> 3. On frames B the counts are 2 -> 2 -> 3 (anchor / BA / BA+DA3).
  - X4's own method already reaches 2 on frames B: ME340 s4 at 1.20 cm and Sam's s1 at 1.35 cm.
  - ME340 s4 measures 8.1 cm on frames A and 1.2 cm on frames B with the same X4 method.
- Confirmed: A 0 -> 1 -> 1 -> 1 (Sam's s1). B 0 -> 0 -> 1 (Sam's s7, BA+DA3).
  - Sam's s1 fails on frames B with every method (held-out outline 0.47-0.48, needs 0.5).
  - Sam's s7 fails on frames A (mv 1.96-2.41 cm). Its frames-B pass is marginal: fine-to-coarse 4.93 cm against the 5 cm limit, and the VLM called the outline "loose".
  - **No spot is confirmed on both frame sets by any method.**
- Paired per spot, frames A:
  - BA vs anchor: 18 better, 11 worse, Wilcoxon p = 0.27, so BA alone is not significant.
  - BA+DA3 vs anchor: 25 better, 4 worse, p = 0.0004.
  - On frames B, BA is 17/9 (p = 0.004) and BA+DA3 is 19/8 (p = 0.046).

**Factory clip s0-s7:** 5.1-29.8 cm, not "10-30 cm with every method". BA+DA3 gives 5.1, 8.1 and 9.8 cm on s5, s6 and s2. s1 was refused at the x1.35 bound: its stored pose change (156 cm median) was not applied. s3 had no tracks and no multi-view value.

**Distance (frames A)**

| distance | spots | anchor | BA | BA+DA3 | BA+DA3+tri |
|---|---|---|---|---|---|
| under 3 m | 5 | 7.91 | 3.66 | 3.36 | 2.90 |
| 3-6 m | 13 | 4.22 | 3.29 | 2.13 | 2.23 |
| over 6 m, all | 12 | 10.79 | 12.34 | 8.65 | 8.90 |
| over 6 m, without factory | 5 | 6.33 | 2.90 | 4.23 | 3.92 |

The drop-off beyond 6 m comes from the 640x480 factory clip, not from distance.

**Tilt repeatability, face-matched |A-B|**
- On the 14 spots that all three methods share: anchor 1.72°, BA 1.74°, BA+DA3 1.20°.
- BA+DA3 is better than anchor on 7 spots and worse on 7 (p = 0.50). Within 1°: 5 / 5 / 5.
- The claimed 2.13 / 1.74 / 1.20 are medians over different spot sets (19 / 16 / 14).
- The X4 confidence interval is 0.29-0.39° wide. If it were calibrated, the median |A-B| would be about 0.07-0.10°. The observed 1.2-2.1° means it understates the spread about 12-30 times, not 3-7.

**Camera evidence**
- Reprojection 1.00 -> 0.32 px and held-out resection 4.96 -> 0.82 px are fitted residuals, not held-out tests.
- The independent held-out outline IoU got worse: 0.555 -> 0.501, with failures 14 -> 15.

**Time per spot, frames A median**
- BA 4.29 s vs anchor 3.24 s re-timed in this container. X4's own run 006 recorded 2.3 s/spot.
- The total includes about 0.48 s of BA diagnostics that are not part of the product, and leaves out the metrics stage (about 0.15 s). BA's real overhead is therefore about 0.6 s per spot, not 1.05 s.

**Spot finding, X4 function vs fast finder**

| video | X4 function in this container, s | fast, s | X4 run 006, s |
|---|---|---|---|
| ME340 | 4.00 | 0.55 | 2.54 |
| Sam's Club | 48.1 | 4.06 | 32.1 |
| Walmart | 19.3 | 1.89 | 12.5 |
| Factory | 11.9 | 1.03 | not run |

- Against the recorded X4 times the speed-up is 4.7x / 7.9x / 6.6x.
- The fast finder alone ran with one BLAS thread.
- "Same picks" was checked on group, word, keyframe count and centre only. Triggers were compared only in the synthetic self-check.

**Unchanged and confirmed**
- GPU peak 48.97 GiB (coarse pass) and 40.93 GiB (refine); nvidia-smi sampled at 20 Hz; nothing over 90%.
- Spend: $6.18 billed now across the 8 apps (run 008: $1.470), under the $10 cap.
- Licences match the primary sources.
