# Adversarial verification of x7-object-models: holds-with-corrections

These corrections take precedence over results.json conclusions.

X7 mostly holds, with corrections. Its tables and code check out, but several interpretive claims and all three fact types fall short.

**What holds**
- results.json rebuilds exactly from the saved outputs with the branch's own evaluate step. Every value in the eligibility, acceptance, time-to-model, memory and overlap tables matches.
- Held-out views are distinct frames, never used to generate or place any model.
- t0 is the hand-off in the model container, and the volume commit is included. Model loads and cold start are kept outside the analysis time.
- GPU peaks are real nvidia-smi samples; none is over 90%.
- The delivered reports are the right ones: c40fbd08, 913daf2a and 32cec650, with 22 / 67 / 40 models and the same clip windows.
- The GLBs on the Modal volume match the acceptance counts. Both self-checks pass.
- I looked at the contact sheets: the accepted generator models plausibly match their held-out crops.

**What needs correcting**
1. **Tilt facts are unstable.** The example "control panel tilt 1.73 ± 0.12 deg" is 0.08 ± 0.16 deg for the same object in the 15 deg arm, so the ± understates the error more than tenfold.
2. **Box footprints are mostly degenerate.** Most box fits use one view and the unseen depth is inferred: 9 of 58 have a side of 3 cm or less and 21 of 58 of 5 cm or less, and they render as flat slabs.
3. **The per-type conclusions rest on a weak test.** They come from the 5 deg arm, where the held-out view is only about 6 deg away and often tiny. They also shrink sharply at the 300 px floor: Sam's Club drops from 36 / 36 / 44 to 23 / 21 / 26 accepted.
4. **Smaller issues:**
   - The machine bullet leaves out that the same object failed both generators at 5 deg.
   - The overlap "ceiling" counts objects that were never attempted. For eligible objects it is 5 / 12 / 3, not 7 / 23 / 8.
   - The Walmart 0/40 is structural: 11 of 14 accepted objects fall before frame 383, where no delivered models exist.
   - SAM 3D memory is 28.1 GiB, not 30.2 (1e9-byte units).
   - Parametric fits write no GLB.
   - Cold start from the client call is 115–153 s, not 106.5 s.
   - The spend of $2.65 is a list-price estimate.
   - The "fb/integrate mismatch" side claim appears false.

No GPU jobs were run for this check. Files: `/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/fx-x7-object-models-001/results.json` and the `models2/*.json` files next to it; code in `/Users/adam/.codex/worktrees/panoptes-phase2-video-fx-x7-object-models/fast_report/x7.py` and `/Users/adam/.codex/worktrees/panoptes-phase2-video-fx-x7-object-models/modal_apps/x7_object_models.py`.

## Problems
- [major] The example fact does not hold up. The same control panel (obj-1-140) gives a tilt of 0.077 ± 0.163 deg in the sep15 arm (the arm that follows the rule) and 1.73 ± 0.124 deg in sep5. The difference, about 1.65 deg, is more than 10 times the stated ±. The clipboard also disagrees (0.37 ± 0.04 against 0.70 ± 0.11). The ± comes from resampling points and ignores which views are used, so the headline example '1.73 ± 0.12 deg' is not reproducible. The '67 facts' are 60 unique objects, and 7 of them appear in both arms with conflicting values.
  Fix: Report tilt facts with a spread that includes the choice of views (for example, refit on each subset of the generation views, or across the two arms), or drop the ±. Replace the example, or show both arm values. Count unique objects, not records.
- [major] Box footprints are presented as measurable, estimated facts, but most are single-view fits of the visible face. 44 of 58 sep5 box fits use one view. For 9 of 58 one footprint side is 3 cm or less (the 2 cm minimum in `parametric()`), and 21 of 58 have a side of 5 cm or less (for example a carton 0.444 x 0.029 m). The contact sheets show these boxes as flat slabs. The unseen depth is inferred, not observed, which breaks the rule to keep observed and inferred apart. The gate on a view 5 deg away cannot catch this.
  Fix: Report a footprint only when two or more faces were observed, or when the generation views span enough angle. Otherwise mark the depth side as inferred or unknown. Keep the top and bottom heights, which agree within 1 cm across arms.
- [major] The per-type conclusions ('What worked') rest on the sep5 arm. There the held-out view is only a median 5.6–6.5 deg from the generation views. Most accepts are on held-out views tagged 'small' (26 of 36 RecGen, 25 of 36 SAM 3D and 32 of 44 parametric in Sam's Club sep5), with supported pixels as low as 108. The counts also depend heavily on the 103 px floor: at fb/b's 300 px on the grid, Sam's Club sep5 falls from RecGen 36 / SAM 3D 36 / parametric 44 to 23 / 21 / 26. The views are technically held out (distinct frames, 13 or more frames from any generation frame, never used for generation or placement), but they are weak out-of-sample evidence. The machine object shows how fragile single verdicts are: the same SAM 3D input (frame 561) passes with the frame-418 held-out view (IoU 0.685) and fails with frame 597 (IoU 0.366).
  Fix: State the per-type rates as conditional on a 5 deg separation and the 103 px floor, and give the 300 px and sep15 counts next to them. Treat the per-type ranking as provisional until a test uses 15 deg or more, for example across shots or across windows.
- [minor] The machine bullet reports only the sep15 pass (RecGen IoU 0.82 against SAM 3D 0.68). It leaves out that the same object (obj-1-111) failed both generators at sep5 (IoU 0.366 each).
  Fix: Give both arm outcomes, or drop the IoU comparison.
- [minor] The last overlap column ('delivered models matched by anything we attempted': 7/22, 23/67, 8/40) counts all 100 ranked objects, including ones that were never eligible and never modelled. Counting only eligible, actually attempted objects gives 5/22, 12/67 and 3/40. The issue list blames the 100-object cap, but eligibility is the binding limit: in Sam's Club every one of the 12 reachable delivered models was covered.
  Fix: Relabel the column as 'matched by any of the 100 ranked objects', add a column for eligible objects (5 / 12 / 3), and correct the stated cause.
- [minor] The Walmart 0/40 is largely structural. The delivered Walmart models appear only in frames 383–748, and 11 of the 14 accepted objects are seen only at 7.4–15.2 s (frames ≤ 381). The claim says the author 'did not look further into why'.
  Fix: Annotate the row: only 3 of the 14 accepted objects fall inside the delivered frames, and none of them matched.
- [minor] SAM 3D per-process memory is in 1e9-byte GB (sam3d.py divides by 1e9) but is reported as GiB. The claimed 'SAM 3D 30.2 GiB' is 30.17 GB, which is 28.1 GiB. The RecGen figure (22.4 GiB) is a true GiB value. The per-GPU peaks come from nvidia-smi and are real, and none is over 90%.
  Fix: Convert to GiB (28.1) or relabel the unit as GB.
- [minor] The parametric fits write no GLB: `fit_param` has no `display_glb` call, and the panoptes-x7 volume holds only the generator GLBs (5/6/7/72/0/14). The stated clock end ('gate decided and GLB written') therefore does not apply to parametric, and the 'first models at 2.5–6 s' exist only as in-memory meshes and JPEG tiles.
  Fix: Write parametric GLBs inside the clock, or state that parametric timings end at the gate decision and produce no model file.
- [minor] 'Container cold start is 106.5 s' measures from container entry to ready. From the client call it is 115.4 s. The first run took 114.5 s from entry and 153.0 s from the call, with 38.5 s before entry. Across the 3 boots, entry to ready ranged 95.2–114.5 s.
  Fix: Report the range and the time from the client call alongside it.
- [minor] Some reported numbers are not in results.json: the machine IoUs (0.82 / 0.68) and the 94 s image build. The `recommendation` block was added by hand and is not produced by aggregate(), so `--evaluate` does not reproduce it. Everything else re-derives exactly.
  Fix: Have aggregate() emit these numbers and the recommendation block.
- [minor] Run 1 is presented as a 'repeat', but it ran different code: the feat commit cdf08ca, without the light copies and with the 300 px floor. Its SAM 3D (b) count at 300 px on ME340 sep15 is 3 against 2 in the main run.
  Fix: Label run 1 as an earlier code version, not a repeat.
- [minor] The side claim 'fb/integrate still has that mismatch' looks wrong. fb/integrate's fast_report_app.py imports fast_report.instrument and calls Vram([0, 1], source=...), which matches instrument's signature.
  Fix: Remove or re-check the claim.
- [minor] The timing model holds param_s constant at 6.236 s for 20, 40 and 70 objects. Its RecGen rate comes from Sam's Club, where most objects had one view, so it predicts 89.5 s for ME340's 30 objects against 103.7 s measured, and 98 s for Walmart's 33 against 82.9 s (about ±15%). 'From the video bytes add 28–38 s' leaves out the hand-off (dump 1.9–2.3 s, load 3.1–6.6 s). It also leaves out that the core (46–52 GiB peak) and X7 (62–69 GiB peak) cannot share one 2-GPU container.
  Fix: Scale the parametric time with object count, give the error band, and add the hand-off and the second-container assumption to the end-to-end estimate.
- [minor] The spend of $2.647 is a list-price estimate (container entry to done, plus scaledown), not an invoice. It is consistent with the Modal app lifetimes (3 panoptes-x7 apps, all stopped, at 268, 312 and 605 s, plus the core app ap-Y2Gt853), but the actual bill cannot be checked here. The image build and the time before entry are excluded.
  Fix: Label the figure as an estimate.
- [minor] The saved records cannot confirm that 'almost every (c) rejection is small in every view': they store only 'no other view passes the prepare rule' (48 of 53). Separately, the pool for (c) views excludes only the held-out frame itself, so a (c) view could sit next to the held-out view. Since (c) accepted nothing, this changes no result.
  Fix: Record each view's reasons for failing; exclude views within the separation angle of the held-out view from (c).
- [minor] The self-check in x7_object_models.py covers only the mask packing. The non-trivial overlap(), aggregate() and timing_model() logic has no check. Both self-checks do pass when run.
  Fix: Add one assert-based check for overlap() on synthetic masks.

## Corrected table

All numbers were re-derived from the saved outputs. Running `x7_object_models.py --evaluate` on a scratch copy reproduces results.json exactly, apart from the `recommendation` block, which was added afterwards by hand. Every value in the eligibility, acceptance, time-to-model and overlap tables matches the saved runs. The corrections and qualifications below should be added to the claimed tables.

**Held-out separation (min pairwise angle, median; eligible objects)**
- sep15: ME340 17.8 deg, Sam's Club 16.3, Walmart 17.6.
- sep5: 5.6 / 6.05 / 6.5 deg (p10 about 5.1–5.3).
- 230 of the 277 objects not eligible at 15 deg had 2 or more good views but too little angle, so the "walking past" explanation holds.

**Accepted on the held-out view, with fb/b's 300 px floor on the grid alongside the 103 px floor (RecGen / SAM 3D b / SAM 3D c / parametric)**
| video / arm | at 103 px (as claimed) | at 300 px |
|---|---|---|
| ME340 sep15 | 3 / 2 / 0 / 2 | 3 / 2 / 0 / 2 |
| ME340 sep5 | 3 / 3 / 0 / 2 | 2 / 3 / 0 / 2 |
| Sam's Club sep15 | 3 / 4 / – / 5 | 0 / 0 / – / 0 |
| Sam's Club sep5 | 36 / 36 / 0 / 44 | 23 / 21 / 0 / 26 |
| Walmart sep15 | 0 / 0 / – / 0 | 0 / 0 / – / 0 |
| Walmart sep5 | 11 / 3 / 0 / 14 | 4 / 3 / 0 / 5 |
- In Sam's Club sep5, the held-out view is tagged 'small' for 26 of 36 RecGen, 25 of 36 SAM 3D and 32 of 44 parametric accepts. Supported pixels go as low as 108.

**Time to first / last accepted (sep5)**
- All values as claimed.
- The parametric fits write no GLB: `fit_param` writes none, and the volume holds 0 parametric GLBs. For parametric, the clock stops when the gate is decided, not when a GLB is written.
- GLB counts on the panoptes-x7 volume match the generator acceptances: 5 / 6 / 7 / 72 / 0 / 14.

**Per-object inference and load**
- RecGen medians 8.2–11.8 s, and 6.8 s for Walmart sep15 (n=1). Measured while 6 generator processes shared the 2 GPUs.
- SAM 3D medians 5.6–7.0 s, and 3.6 s for Walmart sep15 (n=1).
- Model loads: RecGen 55.4–56.5 s + 12.5–13.3 s warm-up; SAM 3D 87.3 s + 10.0–10.1 s warm-up. Both correct.
- Cold start: 106.5 s from container entry to ready. From the client call it is 115.4 s (main run) and 153.0 s (first run, which had 38.5 s before entry). Across the 3 boots, entry to ready ranged 95.2–114.5 s.

**GPU memory (Sam's Club sep5)**
- Peaks 61.55 GiB (GPU 0) and 68.56 GiB (GPU 1), sampled by nvidia-smi every 50 ms. Both are real, set within that run, and no stage went over 72 GiB.
- Largest RecGen process: 22.41 GiB.
- Largest SAM 3D process: 30.17 GB, which is 28.1 GiB. sam3d.py divides by 1e9, not 2^30.
- In later runs, peaks show up at t=0.03 s during the CPU-only x7.rank_stage. This is allocator cache left over from earlier runs, so per-stage attribution means little in those runs.

**Overlap with the delivered reports (sep5, any method)**
| video | accepted | of those, matching a delivered model | delivered covered | matched by any of the 100 ranked objects | matched by an eligible (actually attempted) object |
|---|---|---|---|---|---|
| ME340 | 4 | 2 | 2/22 | 7/22 | 5/22 |
| Sam's Club | 48 | 14 | 12/67 | 23/67 | 12/67 |
| Walmart | 14 | 0 | 0/40 | 8/40 | 3/40 |
- The binding limit is eligibility (the angle rule), not the 100-object cap.
- The delivered Walmart models appear only in frames 383–748. 11 of the 14 accepted Walmart objects are seen only at 7.4–15.2 s (frames ≤ 381), so they cannot match by construction.
- The report mapping is correct: the tests/fixtures/delivered-303 files tie c40fbd08, 913daf2a and 32cec650 to the *-303-merged runs, with 22 / 67 / 40 models and the same clip windows.

**By object type (sep5)**
- Boxes, as claimed: parametric 58/87, RecGen 45/87, SAM 3D b 37/87. Where SAM 3D actually generated (51 boxes), it accepted 37 against RecGen's 34 on the same 51. The small-object rejection explanation holds.
- Machine: one object (obj-1-111). At sep15 both passed (RecGen IoU 0.82, SAM 3D 0.685, held-out frame 418). At sep5 both failed (IoU 0.366, held-out frame 597), although the SAM 3D input was the same (frame 561).
- "Other" group: 3/25 for each generator, 4/25 for the union. It also contains a welder, cylinders, toilet paper, paper towels, wipes and a plastic bag, not only small or thin items.

**Facts**
- 67 records cover 60 unique objects; 7 objects appear in both arms.
- Control panel tilt: 0.08 ± 0.16 deg in sep15 (fit frames 617 and 504) against 1.73 ± 0.12 deg in sep5 (fit frames 617, 577, 524 and 486). The gap is more than 10 times the stated ±.
- Clipboard tilt: 0.37 ± 0.04 against 0.70 ± 0.11 deg.
- Heights above the floor agree within 1 cm across the two arms.
- 44 of the 58 box fits in sep5 use a single view. For 9 of 58 the footprint has a side of 3 cm or less (2 cm is the code's minimum), and 21 of 58 have a side of 5 cm or less. On the contact sheets these boxes render as flat slabs.

**Core**
- Merged objects 251 / 589 / 552; objects.v1 at 27.8 / 19.1 / 17.6 s; objects.v2 at 38.2 / 34.4 / 27.9 s; boot 70.57 s. All correct.
- The hand-off between containers is not counted: dump 1.9–2.3 s plus load 3.1–6.6 s.

**Spend**
- $2.647 is a list-price estimate, not an invoice: 0.564 + 0.324 + 0.55 + 1.209.
- It matches the lifetimes of the 3 stopped panoptes-x7 apps (268, 312 and 605 s) and of the core app ap-Y2Gt853.
