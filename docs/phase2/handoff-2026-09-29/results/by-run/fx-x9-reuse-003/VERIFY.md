# Adversarial verification of x9-reuse: holds-with-corrections

These corrections take precedence over results.json conclusions.

The numbers hold and the time findings hold, even a little more strongly than stated. The claims of better outlines and better IDs do not hold as stated, so the L6-n06 recommendation should not be adopted until a no-lift baseline is run.

**Reproduction.** Every number in results.json reproduces exactly. I copied fx-x9-reuse-002/-003 to /private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/runs/ and ran `x9_reuse.py --evaluate` on the copy: 0 differences. I spent $0 and ran no GPU jobs.

**What holds:**
- The table values match the summary block.
- SAM 3 time (frames × measured seconds per frame ÷ 2) and the savings against B15 (64 / 41 / 34 %).
- Frame shares, the people/moving-class pass (21–23 s), ladder loop time, Qwen cost and agreement, check AUC, occlusion counts, and spend of $7.54 (no local ledger edits).
- No stage above 90 % memory.
- The X1 comparison (.787 / .819 vs .81 / .84 on ME340) and the X8 Jev-Omni numbers (.8897 vs .9338, n = 272).
- Provenance timestamps: probe at 23:33 after 2aa9727, sweep at 23:43 after 6c1cbd8.
- The numpy self-check passes locally. The torch ladder self-check runs at the top of every container before any record is written, so it passed in all 6. That is inferred, because torch is not installed locally.
- The calibration fit itself is leave-one-video-out, with standardisation fitted on the training rows only, and the AUC code is correct.

**What needs correcting:**
- **Outlines (blocker).** B5 and B15 are lifted and the ladder is not, and each reference is almost circular for one side (.99 on its own frames). The outline win measures the lift, not reuse. A no-lift B5/B15 baseline is needed.
- **Totals exclude DA3 depth.** The L* configurations need 15 fps depth and B5 does not, so the ladder-vs-B5 gap is understated by about 17–25 s.
- **Reused share has a bug.** L6-n06 is .72 / .79 / .75, not .75 / .82 / .78; L3 is .59 / .65 / .64.
- **Check precision leaves out 10–32 % of reused predictions.** Over all of them it is .77 / .84 / .76 with the check and .36 / .63 / .49 without. The label is 'overlaps some SAM 3 mask', so there are no identity hard negatives.
- **Tuning was not held out.** Thresholds and the recommended configuration were chosen on the same three test videos.
- **Licences are missing.** DA3-GIANT-1.1 is CC BY-NC 4.0 (non-commercial); SAM 3 uses the gated SAM License; DINOv2 and Qwen3-VL are Apache-2.0.
- **Look-alike identity is a regression, not 'no gain'.** Walmart has 7.2 vs B15's 3.6 switches per 100.
- **The Walmart sweep reused a warm container** holding a leftover vLLM, so the 52 GiB peak is an artifact.

**My agreement with the agent labels, from the contact sheets:**
- Moved re-identifications: 13 of 14. Sam's #0 is uncertain; the agent counted it as the same object.
- Look-alike pairs: Walmart 12 of 12 and Sam's 12 of 12. On ME340 I agree there are no distinct identical items; the 7 same / 5 unclear split cannot be checked exactly.
- Masks missing from the lift reference: 72 of 72 show real structures. But they are repeated views of about 10 physical things, some masks leak onto neighbours, and Sam's #1 and #6 are the camera operator's own cart, which the lift should drop.
- Sam's grey decider: about 19–20 of 22, against the agent's 20 of 22.

**Code:** /Users/adam/.codex/worktrees/panoptes-phase2-video-fx-x9-reuse/fast_report/x9_ladder.py and modal_apps/x9_reuse.py (the reuse-share bug is in `evaluate()`).

## Problems
- [blocker] The outline comparison does not test reuse. B5 and B15 go through X1's lift, which drops 16 / 21 / 12 % of SAM 3's static mask area. The ladder outputs raw SAM 3 masks. Each reference is almost circular for one side: the ladder on its own text-search frames scores .99 against 'raw', and B15 on its own segmented frames scores .99 / .99 / .98 against 'lift' but only .74 / .68 / .78 against raw. So 'ladder outlines .77 vs .67' (and L3 .81 vs .64) measures lift versus no lift, not reuse. The L6-n06 recommendation ('outlines 0.77 vs 0.67') is therefore unsupported. My guess, not measured: a no-lift B15 projected to the 30 fps frames would score at or above the ladder against raw. The agent audit cannot settle this either: (1) the 72 masks were picked largest-first and show about 10 physical structures seen repeatedly (roller door, milling machines, pallet racks, shoe walls, one shelving unit); (2) some masks leak onto neighbours (door plus CNC machine on ME340); (3) Sam's #1 and #6 'shopping cart' is the camera operator's own cart at the bottom of the frame, which a 3D lift should drop. Also, by area the dropped mass is mostly shelves 49 % / stacked boxes 33 % (Sam's), display stands 85 % (Walmart) and machine / pipe / light fixture / metal part 15–20 % each (ME340). It is not 'mostly pipes, light fixtures, ducts, carts and strollers'.
  Fix: Add B5-raw and B15-raw baselines: the same stage-1 SAM 3 masks, no lift, per-frame ids, projected with E6b 'pair' to the 60 evaluation frames. The inputs are on the Volume and this is about a minute of GPU. Compare L3 against B5-raw and L6-n06 against B15-raw. Until then, report outlines as not comparable and withdraw 'keeps outlines better'. Relabel the reference audit as the 24 largest masks per video, and correct the 'mostly pipes...' sentence using the by-area ranking.
- [major] The totals (59 / 40 / 38 s and the rest) are not analysis time. They leave out the DA3 15 fps depth chunks: 50.7 / 36.0 / 33.8 GPU-s, about 25 / 18 / 17 s on 2 GPUs including the 60 evaluation frames. Every L* configuration needs this depth for the depth check, the splat prediction, the coverage test and 3D association, and B15 needs it for the lift. B5 at 5 fps does not: x1.project_to uses only the target pose. The totals also leave out decode, cuts and G5 (7.8 / 6.5 / 6.1 s). So the ladder-vs-B5 gap is understated by about 17–25 s. Also, '5 fps stays 2–4x cheaper' does not match the stated totals: they give 1.6–2.7x (L6-n06) and 2.0–2.9x (L3). Only with DA3 added does it reach about 2.3–3.8x.
  Fix: Rename the column to 'SAM 3 + ladder, 2 GPUs'. Add a full-analysis column: decode + cuts + G5 + (for B15 and L*) DA3 15 fps depth + segmentation + ladder. Restate 'L3 costs 27–37 s more than B5' as about 45–60 s including depth.
- [major] The reused share is inflated by a bug. In `evaluate()`, reused = agree + small + grey_agree is counted on every frame, including text-search frames where the output is SAM 3's fresh masks. Recomputed from frames_log: L6-n06 .72 / .79 / .75 (claimed .75 / .82 / .78); L3 .59 / .65 / .64 (claimed .68 / .73 / .72); L3-box .59 / .65 / .63.
  Fix: Count agree + small only on frames where SAM 3 did not run. Fix it in `x9_reuse.evaluate`, which feeds reuse.object_frames_reused and reused_share_of_output.
- [major] 'Reused predictions that match SAM 3 on the same frame: .91 / .93 / .94 with the check, .46 / .79 / .71 without' drops the reused predictions with IoU .2–.5. That is 16 / 10 / 19 % of them with the check and 23 / 20 / 32 % without. Over all reused predictions the shares are .77 / .84 / .76 with the check and .36 / .63 / .49 without. The truth label is IoU >= .5 with any SAM 3 mask of the frame, not the same object. So the calibration set has no hard negatives for identity (a look-alike or neighbour of similar shape counts as 'right'), and the AUC is computed on the easy >= .5 vs < .2 split only.
  Fix: Report precision over all reused predictions, with the ambiguous band shown separately. Call the label 'mask overlaps a SAM 3 mask', not 'right'. Add identity-hard negatives, for example predictions matched to a different SAM 3 mask of the same word, and report AUC on them.
- [major] The tuning is not held out. Only the logistic weights are leave-one-video-out. THETA_BAD = .3 was set from probe rows of all three videos (the code comment says so). p_hi / p_lo, theta and N, and the recommended L6-n06, were all chosen on the same three videos' test metrics. ME340 was also the development video (smoke run 001). lightning-3585 was available for outlines, timing and reuse, which need no delivered report, but was not used.
  Fix: State the selection leakage. Validate the chosen L6-n06 (and L3) on lightning-3585 or another clip against SAM 3 on the evaluation frames, with thresholds frozen.
- [major] No licences are stated anywhere in results.json or the notes. From primary sources: DA3-GIANT-1.1 is CC BY-NC 4.0 (Hugging Face card, license: cc-by-nc-4.0), which is non-commercial, and the ladder needs it densely at 15 fps. SAM 3 (facebook/sam3) uses Meta's custom 'SAM License' and is gated (manual approval). DINOv2-base is Apache-2.0. Qwen3-VL-8B-Instruct is Apache-2.0.
  Fix: Add a licences block with these entries and the source URLs. Flag that the ladder's depth dependency is non-commercial.
- [major] 'Identical boxes ... no gain on look-alike switches' understates a regression. On Walmart, the only real identical-box case (agent sheet: 9 of 12 pairs are distinct identical boot boxes), L6-n06 has 7.2 look-alike switches per 100 matched observations against B15's 3.6. On Sam's it is 6.1 against B5 4.2 and B15 5.2. The ID metrics also compare different subsets. On ME340 the entities with at least 5 matches number 67 (L6-n06) vs 42 (B5) vs 28 (B15). The box-in-mask rule (at least 30 % of the id's mask inside the box) also structurally penalises merged, lifted objects. On Sam's the ladder is slightly worse, not 'on par': 11.2 vs 8.9 / 9.9 switches and 2.04 vs 1.86 / 1.78 ids.
  Fix: Say the ladder is worse than B15 on look-alike switches on Walmart and Sam's. Report the ID metrics on the entities matched by all configurations.
- [minor] The Walmart sweep ran in a reused warm container. Its t0 was 16 s after the Sam's sweep ended, with load 1.2 s and vLLM ready in 1.2 s, against 86–90 s for the others. A leftover vLLM was on GPU 1, so GPU 1 read 43–52 GiB during the run. The reported 'highest was 52 GiB' is therefore an artifact of two vLLM servers, and Walmart's sweep boot was not a cold start. 'No stage above 90 %' still holds: stage-1 SAM 3 peaked at 42.4 GiB, DA3 at 39.3 GiB, and ME340 and Sam's GPU 1 at 26 GiB.
  Fix: Flag the container reuse. Report 42.4 GiB as the real peak, or rerun the Walmart sweep in a fresh container.
- [minor] Stage 1 deduped the people/moving-class masks together with the full vocabulary. So on frames without a text search, the ladder's re-detections of moving classes had already been cleaned against masks it would not have. The dedupe cost on those frames (0.018 / 0.013 / 0.006 s per frame) is not charged. Separately, the 2-GPU estimate assumes SAM 3 splits perfectly across GPUs while the ladder runs sequentially, and triggered text searches run on single frames although their cost was measured in 8-frame batches.
  Fix: Note both. Time a dedupe of the moving-class masks alone, and a single-frame text search.
- [minor] Commit 6c1cbd8 changed behaviour, not only speed. It subsamples each object to at most 3,000 voxels (MEM_CAP) and culls by the last centroid. The probe rows that trained the check used the old memory prediction and the sweep used the new one, so the 'memory' rows shift between training and use. Provenance describes the change as speed only.
  Fix: Describe it as a behaviour change, or regenerate the probe rows at 6c1cbd8.
- [minor] 'Reliability close' overstates it. Approximate ECE is .065 / .023 / .081. On ME340, predictions with p .4–.5 are right 28 % of the time; on Walmart, p .3–.4 is right 57 % and p .4–.5 is right 70 %. So the fixed .3 and .7 thresholds mean different error rates on different videos.
  Fix: Report ECE per video, and state that the reused and rejected error rates differ by video.
- [minor] Several smaller claims are off. Box prompts add 7.8–19.5 s, not 8–17 s, and change outlines by -0.016 to +0.006, not ±0.01; they also worsen ME340 ids (3.06 vs 2.24 per entity), which the write-up does not mention. agent-labels.json says 'the same events occur in every calibrated configuration', but the moved-event counts are 19 / 1 / 7 (L3), 6 / 4 / 6 (L6-n06) and 0 / 4 / 2 (Linf). The ME340 recall is equal to B5's (.726) with 1,013 vs 649 objects, which is object density. Sam's scale check fails (1.124). The recall columns are not marked 'estimated'.
  Fix: Correct the ranges. Remove the 'same events' sentence. Mark metres as estimated in the table. Note that ME340 recall equality comes with 56 % more objects.
- [minor] Jev-Omni was not tested in the grey-decider role, which the user asked for. The decision is carried over from X8's same-object pairs, a different question from X9's grey cases ('is the outline right?', where Qwen scored 53 % on Sam's). The delivered publication ids are also not recorded in results.json. I checked them through tests/fixtures/delivered-303 and they match c40fbd08 / 913daf2a / 32cec650.
  Fix: Label the Jev-Omni decision 'inferred from X8', or run Jev-Omni on the 24 grey cases per video. Write the reference ids into results.json.

## Corrected table

Values are ME340 / Sam's / Walmart. All base numbers reproduce exactly from the saved stage files. Rerunning `evaluate()` on a scratch copy of runs/fx-x9-reuse-002 and -003 gives 0 differences against results.json.

What changed from the claimed table:
- The "reused share" column is recomputed. The code counted "agree" decisions on text-search frames, where the output is SAM 3's own fresh masks, not reused ones.
- A new last column gives the share of all reused predictions whose IoU with SAM 3 on the same frame is at least 0.5. It includes the IoU 0.2–0.5 cases the claim left out.
- "SAM 3 frames" means frames with the full-vocabulary text search. The vision pass and the people/moving-class pass run on all 443 / 375 / 375 frames in every L* configuration.
- "Total" is not analysis time. It leaves out:
  - video decode and cuts;
  - DA3 G5 (7.8 / 6.5 / 6.1 s);
  - the DA3 15 fps depth chunks, about +25 / +18 / +17 s on 2 GPUs. B15 and every L* configuration need these chunks. B5 does not: its projection needs only target poses.

| config | full-vocab SAM3 frames | SAM3 s (2 GPUs) | total s (2 GPUs, est., excl. DA3/decode) | outline vs SAM3 raw [favours unlifted] | outline vs lift [favours B5/B15] | recall 0.5 m (est.) | ids per delivered entity [different subsets] | switches /100 matched | reused share (corrected) | reused with IoU>=.5, all reused |
|---|---|---|---|---|---|---|---|---|---|---|
| B5 | 147/125/125 | 34.4/14.9/12.4 | 37/17/14.5 | .64/.60/.57 | .79/.83/.67 | .73/.50/.73 | 3.5/1.9/2.2 | 11.8/8.9/19.0 | - | - |
| B15 | 443/375/375 | 103.6/44.6/37.4 | 107/46/38 | .67/.65/.64 | .82/.86/.73 | .71/.50/.74 | 6.7/1.8/2.1 | 36.8/9.9/17.4 | - | - |
| L3 | 149/125/125 | 50.3/29.7/26.8 | 74/45/41 | .81/.90/.66 | .67/.66/.61 | .77/.55/.76 | 2.4/2.1/2.3 | 7.8/11.8/17.5 | .59/.65/.64 (claimed .68/.73/.72) | .79/.85/.80 |
| L6-n06 | 78/70/79 | 37.5/26.5/24.8 | 59/40/38 | .77/.88/.62 | .65/.65/.57 | .73/.55/.78 | 2.2/2.0/2.3 | 6.4/11.2/18.4 | .72/.79/.75 (claimed .75/.82/.78) | .77/.84/.76 |
| L12-n06 | 50/42/54 | 32.4/24.8/23.7 | 55/38/37 | .73/.85/.54 | .62/.64/.52 | .76/.53/.78 | 2.3/1.9/2.1 | 7.1/10.9/15.3 | .76/.85/.78 (claimed .78/.86/.80) | .75/.83/.75 |
| Linf-n06 | 37/32/37 | 30.0/24.2/23.0 | 51/38/37 | .69/.79/.51 | .59/.61/.51 | .71/.55/.72 | 2.1/1.9/2.2 | 5.7/11.2/16.0 | .77/.86/.81 (claimed .77/.87/.82) | .73/.84/.76 |
| L3-box | 149/125/125 | 50.3/29.7/26.8 | 88/58/49 | .80/.88/.67 | .65/.64/.61 | .76/.55/.75 | 2.8/2.1/2.3 | 8.6/10.9/16.8 | .59/.65/.63 (claimed .68/.73/.72) | .75/.82/.75 |
| L6-n06-nocheck | 75/70/78 | 36.9/26.5/24.8 | 61/42/38 | .71/.85/.59 | .57/.61/.52 | .75/.55/.78 | 3.0/2.1/2.4 | 9.7/12.5/19.5 | .85/.84/.84 (claimed .87/.86/.86) | .36/.63/.49 |
| L6-n06-qwen | 77/70/79 | 37.3/26.5/24.8 | 399/163/217 | .78/.88/.63 | .65/.65/.57 | .74/.55/.78 | 2.4/2.1/2.3 | 6.4/11.4/17.8 | .74/.79/.77 (claimed .78/.82/.82) | .73/.83/.75 |
| B5-raw / B15-raw (same SAM 3 masks, no lift) | MISSING | | | not measured; needed to judge outlines | | | | | | |

**Check quality and look-alikes:**
- Leave-one-video-out AUC is .94 / .89 / .90. It is scored only on rows with IoU >= .5 or < .2 against any SAM 3 mask, so it measures mask overlap, not same-object identity.
- Excluding the IoU .2–.5 band, the reused share with IoU >= .5 is .91 / .93 / .94 with the check and .46 / .79 / .71 without it.
- Look-alike switches per 100 matched observations:
  - L6-n06: 0.3 / 6.1 / 7.2
  - B5: 1.1 / 4.2 / 8.6
  - B15: 0 / 5.2 / 3.6
- So on Walmart, the one real identical-box case, the ladder switches between look-alikes twice as often as B15.

**Delivered-report references:** c40fbd08 / 913daf2a / 32cec650, confirmed through tests/fixtures/delivered-303. They are not recorded in results.json.
