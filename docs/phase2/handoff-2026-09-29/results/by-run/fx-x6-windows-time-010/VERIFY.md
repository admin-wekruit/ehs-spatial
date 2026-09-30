# Adversarial verification of x6-windows-time: holds-with-corrections

These corrections take precedence over results.json conclusions.

Most of the numbers check out, but four of the headline conclusions need correcting.

**What I confirmed from the saved outputs.** I recomputed from the per-run JSONs, the evaluation files, the sweep summaries, the raw change sweep (scratchpad/sweep008) and a re-run of x6_evaluate on three runs. These all match:
- window counts per run, map pieces, ATE and path shares
- peak memory (sampled by NVML, about 0.6 GiB above torch's reserved peak; no stage above 90% of a card)
- cold start of 42-63 s, excluded from analysis time
- object and instance counts
- the association sweep (1,743 objects; +5% at q50; +29% and 249 splits at q90; 117 -> 101 covered at 5 sigma)
- the final rule's 0 claims on 8 unplanted runs and its single planted hit
- the facts' values and intervals
- spend of about $4

The delivered-report fixtures map correctly to c40fbd08 (22 objects), 913daf2a (67) and 32cec650 (40). Every analysis step sits inside the clock except the reused vocabulary, which is disclosed. No novel views are evaluated, so there is no held-out-view leak; ATE is measured against DROID, which is independent. I looked at every false-claim sheet (10 + 5 + 1) and all the claims are false by eye. I confirmed the one planted 'disappeared' myself on the planted MP4 (box visible at frames 47-67, gone from 68, place empty at 95). The saved sheet for it is drawn on the unplanted source, so its 'before' frame shows no box. The self-checks for timeline, windows, the app and the planting script all pass. x6_evaluate.py has none.

**What needs correcting.**
1. **Speed of (b).** The case mixes one worker per GPU for (b) with two per GPU for (a), and planted with unplanted clips. Under the same setting, (b) gives first objects 1.6-2.5x sooner, but all facts only 3-19% sooner.
2. **Walmart accuracy.** The Walmart ATE of (b) (3.95 cm) is scored on only 54% of the keyframes. On those same keyframes (a) gets 4.19 cm, so (b)'s accuracy advantage holds only on Sam's Club.
3. **Zero false changes.** The rule was tuned on the same clips it is scored on, and it rarely judges anything: only 123 in-view judgements across 6 runs. In (b) the stitch-chain gate blocks most comparisons, and Walmart (b) judged no place at all. So the claim that recall is limited by detection rather than the change rule fails for (b).
4. **Planted recall.** '1 of 20' is 5 distinct events run in 4 configs. The only hit came from the every-keyframe config, which about doubles the time to all facts. The recommended config found 0 of 10, and the two Walmart events fall in the last 0.8 s of the clip.

Smaller issues:
- 'Fewer pieces than today's core' does not hold: the total pieces inside delivered boxes are equal (102/99 vs 101).
- Threshold 0.70 was never run in the container.
- Walmart gives 17 windows in the local sweep but 16 in the container.
- Container timing variance is 21-37%, not about 20%, and the code differed between those runs.
- Stitch scales swing up to 19% per link.
- The SAM 3 plant probe was run on the earlier plants, and its data is only in the scratchpad.
- The branch's final code (timeline and facts) was never run on GPU; the reported facts were recomputed on CPU after the first outputs were seen.

Files: /Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/fx-x6-windows-time-010/results.json, branch fx/x6-windows-time in /Users/adam/.codex/worktrees/panoptes-phase2-video-fx-x6-windows-time. My planted-frame check is in /private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/x6verify/planted_sams_disp.jpg. I ran no GPU jobs and spent nothing.

## Problems
- [major] The speed case for (b) compares different worker settings and inputs. '(b) first objects 6-11 s vs 15-39 s for (a)' and 'all facts 17-29 s vs 19-44 s' put (b) at one worker per GPU against (a)'s two-worker ME340 run (39.4 / 43.6 s), and mix planted and unplanted clips. Within the same container and setting, (b) reaches first objects 1.6-2.5x sooner but all facts only 3-19% sooner (run 005: 37.2 vs 43.6, 24.5-24.9 vs 30.2, 24.6 vs 26.3; run 003: 28.7-30.8 vs 31.9, 20.6 vs 23.1, 18.5 vs 20.1; run 006: 19.3 vs 20.2, 16.6 vs 18.6).
  Fix: Report (a) vs (b) only within one run and one worker setting (the corrected table). State all-facts as 3-19% faster, not 17-29 vs 19-44 s. Time ME340 (a) with one worker per GPU before quoting a range for it.
- [major] The Walmart accuracy claim for (b) (3.95 cm vs 17.8 cm for (a)) compares different keyframe sets: (b) is scored on its largest piece (33 of 61 reference keyframes), (a) on all 61. I re-ran the evaluator's align() on the same keyframes: (a) 4.19 vs (b) 3.95 cm on piece 1, 2.3 vs 1.6 cm on piece 2, 5.5 vs 5.0 cm on piece 3. (b) is only clearly more accurate on Sam's Club (14.5 vs 22.1 cm), and (a) is better on ME340 (4.09 vs 4.29 cm).
  Fix: Score both options on the same keyframes, or report (a) on (b)'s pieces. Replace 'ATE 4.3/14.5/4.0 vs 4.1/22.1/17.8' with the same-keyframe numbers and drop accuracy as a reason to prefer (b), except on Sam's Club.
- [major] The '0 false changes' result is in-sample and nearly vacuous. The free-space rule was tightened twice to remove the false claims seen on these same clips (10, then 5), and the change sweep then picked its config on the same 8 unplanted and 8 planted runs. There is no held-out clip. Under the final rule, in-view place judgements are rare: 30/28/37/12/16/0 object-windows 'occupied' across the six runs, and none free. In (b), the stitch-chain gate (CHAIN_MAX 0.1) blocks 297/405/106 object-windows, so Walmart (b) judged no place at all. The claim that recall is limited by detection 'not by the change rule' therefore fails for (b), whose chain gate disables most comparisons after 2-4 stitches.
  Fix: Report how many places each run actually judged (occupied/free) and how many were blocked by the chain gate. Call the 0 an in-sample result on few judgements. Validate on a held-out clip with a real revisit. Do not recommend (b) for change detection until the chain gate lets comparisons through.
- [major] The planted recall of '1 of 20' overstates the sample and hides the config. The 20 are 5 distinct rendered events x 4 configs. The single hit (Sam's Club (b), g8) happened only with every keyframe as an object keyframe, which takes 41.0 s to all facts against 19.3 s in the recommended config. The recommended and timed config found 0 of 10. Both Walmart events fall at 29.16-29.2 s of a 30 s clip (the last 0.8 s, about 4 keyframes, not 1.3 s), so they can hardly be detected by construction. The only saved diagnostics for 'the box never became its own object' are scratchpad files (plant_ours.json, plant_depth.json), not the notes.
  Fix: Report recall as 1 of 5 events in the every-keyframe config and 0 of 5 in the recommended config, with the time cost of the every-keyframe config. Plant events with at least 2 windows of video after the change. Save the per-event lift diagnostics in the run folder.
- [major] 'Splits are 0 and the delivered objects end up in fewer pieces than today's whole-shot lift (12-13 of 20 against 16 of 20)' misreads the metric. fragments() counts our object centroids inside each delivered box. The total pieces inside delivered boxes are equal: (a) 102, (b) 99, core 101. Only the number of boxes holding 2 or more centroids differs. The metric also counts real sub-parts (a door, a panel) as pieces. For example, in ME340 (b) one machine appears as 'door' g176 and 'machine' g143 and g181, all within about 0.8 m. 'Split candidates = 0' uses a narrow definition: same word, never together in a window, within 0.3 m.
  Fix: Report ours_inside next to in_pieces and drop the 'fewer pieces than today's core' claim. Measure splits with a label-agnostic duplicate test, or by manual review on a sample.
- [minor] Threshold 0.70 never ran in the container (every run is at 0.4 or 0.55). The claim that '0.55 and 0.70 ... changed ATE by at most 1.5 cm and all-facts time by at most 4 s' is supported only for 0.55: ATE changes of 0.04-1.55 cm and times of -3.4 to +0.9 s.
  Fix: Limit the claim to 0.55, or run 0.70.
- [minor] The local window sweep gives Walmart 17 windows (3 collapses) at 0.40, but the container runs give 16 (2 collapses), with different keyframe spans. The recommendation text still says '10/11/17 windows'. The same rule is not reproducible between the Mac and the container.
  Fix: Quote the container's 16 for the pipeline, and note or fix the local vs container difference (decode or keyframe choice).
- [minor] Container-to-container timing variance is understated. Run 003 vs run 005 differ by 21-37% (ME340 (a) 31.9 vs 43.6 s), not 'about 20%'. It is also not the same code: run 003 ran commit 278d02b and run 005 ran 10dea26.
  Fix: State a 21-37% spread and the code difference, or repeat runs on one commit.
- [minor] The branch head was never run on GPU. The deferred appearance test (3be0c3e) and the new facts function (35d6d45) came after every GPU run. The reported facts (run 009) and the final-rule results were recomputed on CPU from the saved windows. The timed runs produced different facts, for example ME340 closest person to 'workbench' 0.03 m, Sam's Club closest person to 'cable' 14.93 m, and only 2 facts on Walmart. So 'all facts' times come from older timeline and facts code, and the facts function was revised after seeing its output on these clips.
  Fix: Run the head code once per video to time the final pipeline, and say the facts were revised after inspecting the first outputs.
- [minor] The SAM 3 probe (IoU 0.95-0.99, score 0.83-0.97) ran on the earlier run-005 plants (0.6 m boxes; frames 25/47, 225/240, 229/261, 704/716, 726/736), not on the 0.8 m plants of runs 006/007 whose recall is reported. Its data exists only in the scratchpad (plant-sam3.json), not in the notes. Scores depend on the prompt: 'boxes' gives 0.87-0.96, while 'stacked boxes' gives IoU 0 on 6 of 10 frames and scores of 0.52-0.59 where it hits.
  Fix: Re-run the probe on the 006/007 plants with the pipeline's own vocabulary, and save it under the run folder.
- [minor] The planted-found contact sheet is drawn on the unplanted source MP4, so its 'before' frame shows empty floor, not the box. I checked the planted clip myself: the box is visible at frames 47/60/67, gone from frame 68, and its place is empty at frame 95. The detection is real, but the saved evidence does not show it.
  Fix: Draw planted-run sheets from the planted MP4.
- [minor] Stitch scales swing from 0.857 to 1.19 per link (up to 19%), not '8-14%'. 'All 28 stitches passed the register_cut_shot gate' says nothing about scale, because that gate only checks centre and rotation residuals. Cumulative drift reaches a product of about 0.82 on ME340 (b).
  Fix: Report the scale range and add a scale-consistency term to the stitch gate.
- [minor] Self-checks: scripts/x6_evaluate.py has none, although the claim says each script has one. The others (timeline, windows, x6_windows_app, x6_plant) run and pass. fast_report/windows.py defaults THRESHOLD to 0.55, while 0.40 is the recommendation.
  Fix: Add an assert-based self-check to x6_evaluate.py and set the default threshold to 0.40.
- [minor] The ORB vs DA3 Pearson (0.947/0.965/0.915) pools keyframe pairs at gaps of 1, 5, 10 and 20, which inflates the correlation. 'Window counts agree within 2' holds per shot, but per video Sam's Club gives 11 (ORB) vs 15 (DA3) at 0.40.
  Fix: Report agreement near the threshold (for example classification agreement at 0.40) and per-video counts.
- [minor] 'Height facts differ between options by 0.8-1.8 m': the 1.8 m on Walmart compares different objects (pallet 1.12 m vs box 2.94 m). Single-window tilt facts carry ± 0.0 (Sam's Club shelves 12.9 and 1.4), which is not an uncertainty.
  Fix: Compare only the same physical object across options, and give no ± for single-window facts (or give a model-based one).
- [minor] 'Two workers per GPU are slower because of Python's global lock' is stated as a cause but was not measured. The merge commit e2d2acd lacks the Co-Authored-By trailer.
  Fix: Call it a hypothesis, or profile it. The trailer is cosmetic.

## Corrected table

Threshold 0.40, 2 x A100-80GB, MPS on. ATE is against DROID after a Sim3 on the reference shot only. Timings are in seconds from the MP4 bytes in the container. p2 = two workers per GPU (the same container for all six rows: run 005, repeated in run 003). p1 = one worker per GPU. (a) was run at p1 only on the planted clips (run 006).

| video / geometry | windows | map pieces | ATE vs DROID | first objects: p2 run005 / p2 run003 / p1 | all facts (old facts code): p2 run005 / p2 run003 / p1 | peak GiB | global objects (per-window) | delivered boxes holding >= 2 of our centroids / covered (our centroids inside) | change claims, final rule | in-view place judgements (occupied) / blocked by stitch chain, final rule |
|---|---|---|---|---|---|---|---|---|---|---|
| ME340 (a) | 10 | 2 | 4.09 cm (97% of ref keyframes) | 39.4 / 28.8 / not run | 43.6 / 31.9 / not run | 49.1/45.6 (p2) | 232 (453) | 13 of 20 (102) | 0 | 30 / 0 |
| ME340 (b) | 10 | 3 | 4.29 cm (same 97%) | 20.4 / 15.9-17.4 / 11.3 | 37.2 / 28.7-30.8 / 29.5 | 32.4/32.1 p1; 45.3/49.0 p2 | 244 (470) | 12 of 19 (99) | 0 | 28 / 297 |
| Sam's Club (a) | 11 | 2 | 22.1 cm (100%) | 22.3 / 18.0 / 15.3 planted clip | 30.2 / 23.1 / 20.2 planted clip | 48.2/36.5 (p2) | 375 (652) | 7 of 14 (52) | 0 | 37 / 0 |
| Sam's Club (b) | 11 | 2 | 14.5 cm (100%) | 9.7-10.4 / 8.3 / 7.0 (6.2 planted) | 24.5-24.9 / 20.6 / 21.7 (19.3 planted) | 32.0/32.0 p1 | 377 (689) | 14 of 25 (73) | 0 | 12 / 405 |
| Walmart (a) | 16 | 2 | 17.8 cm on the whole ref shot; 4.19 cm on (b)'s 33 keyframes | 22.8 / 16.7 / 15.3 planted | 26.3 / 20.1 / 18.6 planted | 35.6/32.2 (p2) | 229 (422) | 16 of 25 (97) | 0 | 16 / 0 |
| Walmart (b) | 16 (local sweep: 17) | 4 | 3.95 cm on 33 of 61 ref keyframes (54%) | 14.2 / 9.0 / 7.3 planted | 24.6 / 18.5 / 16.6 planted | 30.7/29.2 (p2) | 286 (487) | 11 of 14 (44) | 0 | 0 / 106 |

Today's core on ME340 (fb-a-core-011, 167 objects): ATE 4.09 cm; 16 of 20 boxes hold >= 2 centroids, with 101 centroids inside. That is the same number of pieces inside delivered boxes as (a) at 102 and (b) at 99, so it is not 'more pieces'.

Corrected comparisons:

- **(b) vs (a), same container and same worker setting.**
  - First objects: 1.6-2.5x faster.
  - All facts: only 3-19% faster (run 005: 37.2 vs 43.6, 24.5-24.9 vs 30.2, 24.6 vs 26.3; run 003: 28.7-30.8 vs 31.9, 20.6 vs 23.1, 18.5 vs 20.1; run 006 p1 planted: 19.3 vs 20.2, 16.6 vs 18.6).
  - The claimed '6-11 vs 15-39 s' and '17-29 vs 19-44 s' compare (b) at p1 with (a) at p2, and mix planted and unplanted clips.
- **ATE, (b) vs (a), on the same keyframes:**
  - ME340: (a) 4.09 vs (b) 4.29 cm, so (a) is better.
  - Sam's Club: (b) 14.5 vs (a) 22.1 cm, so (b) is better.
  - Walmart: (b) 3.95 vs (a) 4.19 cm on piece 1; 1.6 vs 2.3 cm on piece 2; 5.0 vs 5.5 cm on piece 3. That is about equal, not 4.0 vs 17.8.
- **Change rule (final config, recomputed from the raw sweep):**
  - 0 claims on 8 unplanted runs; the only claim on 8 planted runs is g8.
  - Across the 6 main runs, only 123 object-windows were judged in view (occupied), and 0 were judged free except the planted g8.
  - In (b), the stitch-chain gate blocked 297 / 405 / 106 object-windows. Walmart (b) judged no place at all.
  - The rule was tuned on these same clips, so the 0 is in-sample and nearly vacuous.
- **Planted recall:** 1 hit, counted as 1 of 20. The 20 are 5 distinct events run in 4 configs. The hit needed every keyframe to be an object keyframe, which roughly doubles the time to all facts (41.0 s vs 19.3 s). The recommended config found 0 of 10. The 2 Walmart events fall in the last 0.8 s of the clip (frames 729-749), not the last 1.3 s.
- **Window rule:**
  - 0.55 changed ATE by 0.04-1.55 cm and all-facts time by -3.4 to +0.9 s.
  - 0.70 was never run in the container: those ATE and time claims are unmeasured.
  - ORB vs DA3 window counts agree within 2 per shot, but per video Sam's Club gives 11 (ORB) vs 15 (DA3) at 0.40.
  - Pearson 0.947 / 0.965 / 0.915 is pooled over keyframe gaps of 1-20, which inflates it.
- **Other checks:**
  - Stitch scales range 0.857-1.19 (up to 19% per link), not 8-14%.
  - Run 003 vs run 005 timings differ by 21-37%, not ~20%, and the code differed between them (278d02b vs 10dea26).
  - The spend recomputes to about $3.42 GPU from container lifetimes at the file's list prices, against $3.37 claimed; total about $4.
