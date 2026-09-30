# Adversarial verification of x2-discover: holds-with-corrections

These corrections take precedence over results.json conclusions.

The recorded numbers check out, but three claims need correcting before the recommendation is used.

**What reproduces.** I re-derived every number from results.json, the per-run discover.json, scores.json and harness-2d.json, and from the code on fx/x2-discover:
- Recall 85/107/93 against a baseline of 83/96/85, the same in runs 003 and 004/005.
- Harness objects-2D 88->88, 107->116, 104->107.
- Found and expansion counts, the Qwen judge counts, the rounds, the vlm-ground token cap, the 'spill' and 'fire extinguisher' counts, and the timings per design.
- Peak GPU memory of 71.43 of 85.1 GB (83.9%), with no sample over 90%.
- Inputs match across runs: the same MP4 hash, identical vocabularies, and the correct references c40fbd08 / 913daf2a / 32cec650 from the fixtures.
- No metric claims: heights and centroids are labelled estimated.

**The major problems.**
- **The recall gain is mostly more masks, not correctly named new objects.** The new words add up to 14,600 masks per video and roughly double masks per frame. Only 8 of the 21 gained delivered objects are hit by a mask with the right name. Both of ME340's gains are 'cylinder' masks on a mirror and a plastic bin. Word-match recall (20->21, 29->57, 54->68) is the better headline.
- **The detector veto removes the one real EHS find.** The veto as coded drops the cable tag on ME340's genuine hanging power cord, because 'power cord' is a vocabulary word. That contradicts recommending the catch-all words as a cord add-on.
- **Two precision counts are from the wrong design.** The ME340 and Sam's counts in the amg16:vlm row were taken from run 002's amg16 sheets. My own count on the Sam's amg16:vlm sheet puts bands and groups at about 42%, not 35%.

**Minor problems.**
- The "81-83% printed labels" share comes from the amg-32 run and mixes letter classes.
- The 18-21 s headline leaves out the required prep and write steps; the totals are about 21-22 s, and up to 30 s in run 003.
- Run 003's 26.6 s includes loading SAM 2 inside the timed stage.
- The "faster host" explanation for run 002 cannot be checked.
- Per-stage memory peaks are high-water marks that carry over from earlier stages, so they are upper bounds only.
- Convergence after one round happens by construction, not by measurement.

I viewed the contact sheets myself: 003 ME340, Sam's and Walmart amg16:vlm; 005 Sam's amg16:vlm expansion; 002 ME340 sam3-generic; 002 Walmart amg. No GPU jobs were run.

Key files:
- /Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/fx-x2-discover-005/results.json
- /Users/adam/.codex/worktrees/panoptes-phase2-video-fx-x2-discover/fast_report/discover.py (run_design, recall_eval, warm_amg)
- /Users/adam/.codex/worktrees/panoptes-phase2-video-fx-x2-discover/modal_apps/fast_report_app.py (FastReport.discover)
- /Users/adam/.codex/worktrees/panoptes-phase2-video-fx-x2-discover/scripts/x2_discover.py and x2_results.py (own-look provenance)

## Problems
- [major] The position-recall gain mostly comes from adding many masks, not from correctly named new objects. Expansion adds 1,373 / 14,643 / 9,930 raw SAM 3 masks, against 20,145 / 11,439 / 6,093 vocabulary masks. The harness shows the effect too: masks per frame go from 195 to 409 on Sam's and from 85 to 226 on Walmart. Of the 21 delivered objects gained (found_only_by_discovery for amg16:vlm), only 8 are hit by a mask whose name matches under the harness's same_name rule: 0 of 2 on ME340 (both are 'cylinder' masks on a 'mirror' and a 'plastic bin'), 3 of 11 on Sam's ('plastic wrap' or 'case of bottles' on paper towel, toilet paper and diaper packs), and 5 of 8 on Walmart. No control with the same number of masks was run. The claims "ME340 88.3 -> 90.4%" and "the gain comes from the new words" overstate what was found.
  Fix: Report word-match recall next to position recall: 20->21, 29->57, 54->68. Say that ME340's +2 are wrongly named hits. Add a control that adds the same number of masks from other words, or report precision of the expansion masks. Do not describe the position gain as finding objects by name.
- [major] The recommended detector veto (a name in the original vocabulary gets no EHS tag) also removes the one genuine EHS find in the study. 'power cord' is an ME340 vocabulary word, so the hanging power cord that sam3-generic found is vetoed. results.json shows sam3-generic ME340 with 1 name vetoed and EHS tags {'boxes/stacks': 1}, with no cable/wire tag. This conflicts with recommending the catch-all words as an add-on for cords.
  Fix: Make the veto check whether that word's SAM 3 masks actually covered the region with IoU below the threshold (the rule as stated), rather than only whether the name is in the vocabulary. Or exempt catch-all proposals from the veto, then re-check the ME340 cord case.
- [major] The own-look precision in the amg16:vlm row for ME340 (8/11/1/0) and Sam's (28/3/0/17) was counted on run 002's amg16 sheets (zero-shot names), not on amg16:vlm; results.json has own_look null for amg16:vlm on those videos. The verifier counted the 003 amg16:vlm sheets: ME340 about 7 A / 10 B / 3 C; Sam's 25 A / 3 B-C / 20 D of 48, so bands and groups are about 42% rather than the claimed 35%. Walmart (20/12/0/4 of 36) matches.
  Fix: Label these counts as coming from the amg16 proposals of run 002. Recount them on the amg16:vlm sheets, 003/me340 and 005/samsclub-a2.
- [minor] The "in stores 81-83% of expansion instances are printed labels on packs" figure comes from the run 002 amg (32x32, zero-shot) expansion sheets, not from the recommended amg16:vlm. It is also inconsistent: Sam's B is 35/48 = 73%, and 81% only if D groups are added, while Walmart's B is 40/48 = 83%. On the verifier's look at 005 Sam's amg16:vlm expansion page 0, about half the tiles are price tags or printed labels; the rest are wrap bands and whole packs.
  Fix: Give 73% (Sam's) and 83% (Walmart) and say they are from amg-32 run 002. Count the amg16:vlm expansion sheets before claiming a share for the recommended design.
- [minor] The headline 18-21 s leaves out required steps. Prep (SAM 3 wall/ceiling/subtitle masks, which the filter needs) takes 0.73-2.2 s, and the write takes 1.40-1.84 s. The ranges given for these, 0.9-2.2 and 1.5-1.9 s, are slightly off. The measured totals are 21.9 / 21.3 / 20.7 s (004/005) and 29.9 / 25.2 / 23.9 s (003).
  Fix: Report the total, about 21-22 s warm and up to 30 s in run 003, as the added analysis time.
- [minor] Run 003's ME340 time of 26.6 s is described as a cold first AMG call with about 8 s of warm-up. Before commit 64be441 the app loaded SAM 2 only when the exact design 'amg' was in the list. Run 003 used amg16:vlm, amg16 and amg:vlm, so load_sam2 ran inside the timed propose stage (sam2_load_s = 0.0). The 19.8 s proposal time therefore includes loading the model onto 2 GPUs. The 004/005 'warmed' figures were measured with amg16:vlm as the third AMG design on the same video, after a per-video warm-up of 1.5-7.9 s that is excluded.
  Fix: Describe 003 ME340 as including the SAM 2 load, and note that the warmed figures follow two earlier AMG passes on the same video.
- [minor] The 'faster host' explanation for run 002 (amg16 proposals 7.0-8.3 s against 10.3-12.4 s) cannot be checked. amg() is identical from 5a5f9a9 to HEAD, and every run used the same A100-SXM4 model. The table also mixes timings from different runs across rows.
  Fix: Call it an unexplained run-to-run variance of about 30-50%, and compare designs only within one run.
- [minor] The per-stage GPU peaks are whole-device used memory from mem_get_info. Under the PyTorch caching allocator this only rises, so each stage shows the high-water mark of the stages before it (for example, every stage after expand.sam3 shows 65.73 GB). The maximum of 71.43 GB (83.9%) is a valid upper bound, but the per-stage attribution means nothing.
  Fix: Call these upper bounds, or reset and read torch.cuda.max_memory_reserved per stage alongside the device figure.
- [minor] "The loop always converges after one productive round" follows from the code, not from measurement. Round 1 tries to name every new cluster, and round 2 only names clusters not yet in `names`, so round 2 cannot add words.
  Fix: Say the loop is effectively one-shot by design.
- [minor] On the harness's own 56 frames, the recommended amg16:vlm gets 107/123 on Walmart. The rejected amg16 (zero-shot, 111) and amg:vlm (112) both score higher there, while the paired-keyframe metric ranks them the other way (92-95 and 94 against 93).
  Fix: State the disagreement between the two metrics when ranking the designs.

## Corrected table

Delivered named objects found on the delivered mask frames paired with our object keyframes (37 / 23 / 19 paired frames). "Position" means any of our masks has IoU >= 0.5 with the object, whatever its name. "Word" means a mask at that IoU whose name matches the delivered name under fast_report_eval.same_name. Baseline (vocabulary only): position 83/94, 96/121, 85/116; word 20, 29, 54 (ME340 / Sam's / Walmart). Hardware: 2 x A100-SXM4-80GB in every run; 004's hardware comes from core-run.json because it has no meta.json. Discovery ran after the core in the same warm container. The harness objects-2D job ran separately on 1 x A100-80GB PCIe. Scale is estimated.

| Design (best first) | Position ME340 / Sam's / Walmart | Word-match | Gains whose matching mask has the right name | Added analysis s (2 A100): design / + prep + write | Found objects (+expansion) | Own look A/B/C/D | Note |
|---|---|---|---|---|---|---|---|
| **amg16:vlm** | 85 / 107 / 93 (+2 / +11 / +8); the same in 003 and 004/005 | 21 / 57 / 68 | 0 of 2 / 3 of 11 / 5 of 8. ME340's two gains are 'cylinder' masks on a mirror and a plastic bin | 19.1 / 19.0 / 17.7 (004/005; AMG warmed by 2 earlier AMG designs on the same video plus a per-video warm-up of 1.5-7.9 s left out as cold start). Totals 21.9 / 21.3 / 20.7. Run 003: 26.6* / 21.4 / 20.4, totals 29.9* / 25.2 / 23.9. *includes loading SAM 2 inside the timed stage | 20-21 / 50 / 35-36 (+14 / 172-179 / 97-105) | ME340 and Sam's numbers are from run 002's amg16 sheets, not amg16:vlm. Verifier's look at 003 amg16:vlm: ME340 about 7/10/3/0 of 20; Sam's 25 A / 3 B-C / 20 D of 48 (D 42%); Walmart about 21/11/0/4 of 36 (matches the claim) | Harness objects-2D, new words only: 88->88/91, 107->116/132, 104->107/123. Masks per frame rise 195->409 (Sam's) and 85->226 (Walmart). On Walmart, amg16 (111) and amg:vlm (112) beat amg16:vlm (107) |
| amg16 (same proposals, SigLIP zero-shot) | 85 / 106 / 92 (002) and 95 (003) | 21 / 49 / 67 | - | 13.0 / 14.1 / 11.8 (002); 23.9 / 19.9 / 16.8 (003) | 20 / 49 / 44 | same proposals | zero-shot named 9 (amg16) to 22 (amg-32) Walmart slippers and sign letters 'spill'. Why run 002 was faster is not verified: same amg() code and same GPU model |
| amg (32x32):vlm | 86 / 108 / 94 | 23 / 77 / 63 | - | 41.0 / 45.8 / 43.9 (003) | 32 / 79 / 43 | Walmart 26/16/0/1 of 43 | +1 object per video over amg16:vlm for 1.6-3x the proposal time; up to 6 false 'fire extinguisher' per video |
| amg:part | 85 / 103 / 93 | - | - | 28.6 / 28.2 / 26.5 (002) | 16 / 20 / 42 | ME340 6/4/6/0 | -1 / -4 / -1 against amg |
| amg16/2:vlm ; amg16/3:vlm | 83 / 101 / 86 ; 84 / 100 / 85 | - | - | 9.8 / 10.7 / 9.1 ; 9.1 / 4.9 / 4.6 | 10 / 18 / 10 ; 6 / 2 / 1 | - | gain collapses |
| sam3-generic | 83 / 96 / 86 | - | - | 3.8 / 2.0 / 3.2 | 2 / 0 / 5 | 6 of 7 whole objects | Its one real hanging power cord is suppressed by the recommended detector veto: 'power cord' is an ME340 vocabulary word |
| vlm-ground | 83 / 96 / 85 | - | - | 33.4 / 33.0 / 31.5 (57.5 in 001) | 6 / 3 / 0 | ME340 1/0/2/3; Sam's 0/0/0/3 | 24000 completion tokens; all 16 answers hit the 1500-token cap |

Rounds: 1 productive round and 1 check round (<= 0.05 s). This is by construction: round 1 tries to name every new cluster and round 2 only names clusters it has not tried. GPU peak: the highest whole-device used memory over 50 ms samples is 71.43 of 85.1 GB (83.9%); no sample is over 90%. Per-stage values are a high-water mark under PyTorch's caching allocator, not the peak of each stage. Spend is about $6.2-6.3 upper estimate (004 and the harness job are estimates); billing not verified.
