# Adversarial verification of x8-jev: holds-with-corrections

These corrections take precedence over results.json conclusions.

The numbers reproduce, but three conclusions need correcting.

**What I re-derived.** Running scripts/x8_metrics.py on a scratch copy of decisions.json, decisions-d-pass2.json, sets/*.json and the audit files gives a metrics.json identical to the saved one, and every table figure in results.json matches it. All three self-checks pass. Other checks that hold:
- **Card:** worst difference against the card's reference is 0.01761. The card text does contain the three inconsistencies claimed (FP32 about 50 GB, "downloads the original Gemma 4 components", fair die P(6)=0.46); the loader code contradicts the first two. The card's own sha256 for the weights matches Hugging Face.
- **Licences, from primary sources:** Jev-Omni metadata says apache-2.0. google/gemma-4-12B-it metadata says apache-2.0, and its gemma_4_license page is the Apache 2.0 text. Laya is apache-2.0; its card states near-chance zero-shot and its config ships choice:11+ temperature 0.10.
- **References:** the delivered reports are c40fbd08 / 913daf2a / 32cec650, via tests/fixtures/delivered-303.
- **Memory:** read from nvidia-smi (nvml) in GiB; peaks 25.97 and 26.07; no flag over 90%.
- **Timing:** model load and warm-up are excluded, and each stage includes decoding its inputs.
- **Spend:** five Modal apps match the spend lines (setup, smoke, main 10m40s, and set d pass 2 run twice, overlapping). Gemini used 123,218 input and 5,623 output tokens on gemini-3.5-flash. About $2.13 is plausible.
- **Leakage:** the audits of sets a, b and c were saved (22:21-22:25) before the main run (22:53) and were blind to the b and c outputs. The first-pass set d labels are unchanged in the git diff. Thresholds are 2-fold cross-fitted, and the SigLIP calibration is 5-fold cross-fitted.

**My agreement with the agent labels, from the contact sheets:**
- Set a: 36 pairs checked; all 11 flips I saw are correct and all kept pairs are correct. I would have flipped 2 of the "unsure" pairs rather than excluded them, so exclusion was the conservative choice.
- Set b: 12 of 13 renames or drops agree.
- Set c: 47 of 48 tiles agree.
- Set d: clear labels agree almost everywhere; judgement calls about 65-70%.

**What needs correcting:**
1. **Placement (major):** the "45 GiB + 25 GiB, just under 72" arithmetic uses the pre-integration core's GPU 0 peak. The integrated core runs that X8 used as its source peak at 60-66 GiB, so Jev does not fit on GPU 0 at all.
2. **Set e severity (major):** in this set severity is fully determined by the rule, so a lookup after rule routing beats Jev's severity recommendation. The facts are also agent-written against the rule texts; word overlap alone picks the rule for 49% of them. The 99% rule accuracy is an upper bound on a synthetic set.
3. **Minor:**
   - The 25.3 GiB includes SigLIP 2 and Laya; Jev alone is about 22.3 GiB.
   - "No decider reaches 90% precision" on set c is false: Qwen reaches it on 6 items.
   - The Qwen screening thresholds for set d are in-sample.
   - The q1 and q5 results each rest on one or two scenes.
   - The 0.9858 SigLIP threshold is the naming-cache threshold, not the core's merge rule.
   - The batching flips on one-image sets mix padding with a JPEG round-trip.
   - Gemini and X2's judge had different inputs.
   - The look-alike negatives match on name only and are only mildly harder.

No GPU jobs were run and no files were written outside the scratchpad.

## Problems
- [major] The placement claim ('core GPU 0 peak 45 GiB + 25 GiB = ~70 GiB, just under the 72 GiB flag') uses fb-a-core-011's pre-integration peak (45.01 GiB). The integrated fast-core runs that X8 itself uses as the source of sets a/b show GPU 0 peaks of 60.1-66.5 GiB (fb-integrate-me340-006 61.0/60.1, samsclub-002 65.7/66.5, walmart-001 62.9/63.4). Adding Jev (22-25 GiB) gives 83-91 GiB, which does not fit on an 80 GiB card.
  Fix: State that Jev-Omni does not fit on GPU 0 next to the integrated core and must run on the third A100, or be loaded only when GPU 0 is otherwise empty. Cite the fb-integrate peaks.
- [major] Set e severity is fully determined by the rule: every rule's items carry one severity, so all 96 match. A lookup table applied after rule routing (Jev 99%) would give about 99% severity. The recommendation to accept Jev's severity at p>=0.79 and send 22-25% to review is therefore worse than a trivial baseline. The facts are also agent-written against the known rule texts: 81/96 share a word longer than 4 letters with their rule, and word overlap alone picks the right rule uniquely in 47/96 (49%), which beats Laya's 41.5%. The 99% rule accuracy is an upper bound on an easy synthetic set, not evidence about facts derived from video.
  Fix: Report the rule-to-severity lookup baseline and the word-overlap baseline. Drop the severity-by-Jev recommendation, or re-test it on facts whose severity varies within a rule. Mark the e results as synthetic and an upper bound.
- [minor] 'Jev-Omni sits at 25.3 GiB / needs 25 GiB' conflates models. 25.26 GiB is torch.max_memory_reserved on GPU 0 after loading Jev + SigLIP 2 base + Laya and warming up; the results key itself is named resident_gib_gpu0_jev_siglip_laya. Jev's checkpoint is 11.96 B bf16 params, about 22.3 GiB of weights (HF safetensors metadata).
  Fix: Say Jev alone is about 22-23 GiB and 25.3 GiB is the GPU-0 trio.
- [minor] 'No decider reaches 90% precision cross-fitted' on set c, and 'no model reaches 90% precision on any subset', are false. Qwen's 4-way decider accepts 2.3% (6 items) at 100% cross-fitted, and 2.7% at 100% in-sample.
  Fix: Rephrase: no decider reaches 90% precision on more than 2.3% of items; Jev reaches 77% on its surest 17%.
- [minor] The screening figures for Qwen (p_yes 0.004 -> 63% recall at 10% false alarms; 0.011 -> 82% at 6% on clear labels) are in-sample. The thresholds were picked on the same 154/104 items and are not cross-fitted, but the conclusion presents them as operating points.
  Fix: Label them in-sample, or cross-fit them the way the 90%-precision points are.
- [minor] The per-question set d AUROCs rest on one or two scenes. 7 of the 9 q1 positives come from one TUM office room: cables under desks, 8 of 9 labelled as judgement calls, and some are arguably not 'where people walk'. All 6 q5 positives are near-duplicate NASA frames 2907-3220 (one ~10 s event). 31 items (the second pass) were added after the first metrics were seen; the first-pass labels were not changed (verified by git diff).
  Fix: Report the per-question AUROCs as one or two events each and mark the second pass as post-hoc. Within TUM alone, Jev and Qwen AUROC are both 1.0, so the result is not a pure source confound but is very thin.
- [minor] The SigLIP baseline '0.9858 threshold calls 0 of 300 same' is an apples-to-oranges comparison. 0.9858 is the cascade's naming-cache threshold, the q99 of negative cosines between object-mean embeddings (fast_report/cascade.py). It is not the core's same-object merge rule, which comes from the 3D lift, and it is applied here to single-crop pair cosines.
  Fix: Drop the row, or describe it as the naming-cache threshold.
- [minor] The batched-vs-unbatched flips are attributed to right padding. For single-image sets, though, the unbatched path goes through the card's predict(), which re-encodes the crop as a JPEG at quality 95, while the batched path passes the decoded pixels. The b and c flip counts (29/428, 45/428, 19/263) therefore mix the input path with padding. Only the two-image set a (4/300), which uses the same path in both modes, isolates batching.
  Fix: Attribute the b and c flips to batching plus the JPEG round-trip, or rerun the unbatched arm on the same PIL input.
- [minor] The baseline inputs are not identical in places. Gemini saw 10 frames and several questions per request (cross-frame context, ~1.1k image tokens per frame) and returned stated p. X2's generative judge saw 448 px crops, while X8's c deciders saw 220 px sheet tiles. Qwen was not run on the full vocabulary (57 options on ME340, beyond its 26 letters and beyond the card's supported 20 for Jev).
  Fix: Mark these comparisons as having different input budgets. Only claim 'most useful confidence' among the deciders run on the same options.
- [minor] The 'hard' negatives are only name-level look-alikes: same delivered head noun, at least 0.5 m apart, e.g. 'cardboard box' vs 'switch box'. SigLIP cos median is 0.763 vs 0.733 for other-class negatives and 0.87-0.91 for positives. Jev/Qwen score 0.88/0.95 on them against 0.96/0.96 on other-class.
  Fix: Call them name look-alikes and do not describe them as visually hard negatives.
- [minor] There is label noise from a single labeller. The agent's set c counts differ from X2's own look at the same sheets by 3-4 tiles per letter per sheet (me340 amg B 17 vs 13; walmart amg16 B 18 vs 14, D 3 vs 7). The container cold start (image start plus argument upload, about 19 s: client wall 632.4 s vs 94.35 s boot + 519.2 s analysis) is not reported separately from model load.
  Fix: Report the c label disagreement bound, and add the container start to the cold-start record.

## Corrected table

| Set (n) | Jev-Omni | Qwen3-VL-8B (letter log-probs) | Other | Note |
|---|---|---|---|---|
| a same object? audited (272 of 300) | acc 0.890, AUROC 0.95, ECE 0.054 | acc 0.934, AUROC 0.98, ECE 0.053 | SigLIP cos AUROC 0.86. The 0.9858 figure is the cascade's naming-cache threshold on object-mean embeddings, not the core's merge rule (merges come from the 3D lift), so "0 of 300 single-crop pairs reach it" says nothing about the core | When both agree (90%) they are 95.5% right. Against the raw reference: Jev 0.763, Qwen 0.817 (walmart cross-object positives: 11 of 25 flipped, 7 excluded). "Look-alike" negatives share only a name (SigLIP cos median 0.76 vs 0.73 for other-class negatives); Jev/Qwen get 0.88/0.95 on them against 0.96/0.96 on other-class, so they are only mildly harder |
| b which label? audited best view (233) | shortlist 0.519; full vocab 0.416 (up to 57 options, above the card's 20), AUROC(conf) 0.88; the p>=0.95 cut accepts 20% at 91% in-sample, and the cross-fitted operating point accepts 21% at 88% | shortlist 0.498, ECE 0.42 (not run on the full vocabulary, which exceeds its 26 letters) | SigLIP zero-shot 0.322; the core's cascade name 0.365 | "None of these" was correct for 53 items: Jev picked it 0 times, Qwen once. The vocabulary has the right word for only 78% of items |
| c object/part/bg/group (263, agent-labelled) | acc 0.662, ECE 0.12; whole-object-vs-rest AUROC 0.77 | acc 0.547, ECE 0.38; AUROC 0.70 | X2's generative Qwen judge 0.58 (it saw 448 px crops; X8 used 220 px tiles, so the inputs differ) | At 90% precision cross-fitted, no decider covers more than 2.3% of items: Qwen reaches 100% on 6 items (2.3%), Jev 77% on 17% |
| d EHS yes/no (154, 43 yes, agent-labelled) | bal acc 0.63, AUROC 0.74 (cables 0.99, platform 1.0, aisle 0.64) | bal acc 0.63, AUROC 0.87 (aisle 0.78). In-sample operating point: p_yes 0.004 gives 63% recall at 10% false alarms | Gemini 3.5 Flash stated p: bal acc 0.83, AUROC 0.85 (aisle 0.88). Each request carried 10 frames and several questions, so its budget differs | Jev and Qwen say yes to under 10% of questions. q2 and q3 have no positives. 7 of the 9 q1 positives come from one TUM office scene (8 of 9 are judgement calls). All 6 q5 positives are near-duplicate NASA frames from one ~10 s moment. 31 items were added after the first metrics |
| e fact to rule (106) / severity (96), text | 0.991 / 0.771, ECE 0.010 / 0.10 | 0.962 / 0.698, ECE 0.05 / 0.24 | Laya zero-shot 0.415 / 0.479, ECE 0.07 / 0.06. Baselines: in this set severity is fully determined by the rule (a lookup gets 96/96 given the right rule), and word overlap with the rule text alone picks the right rule uniquely for 47/96 | Facts are agent-written against the known rule texts, so this is an easy upper bound |
| Speed on one A100-80GB | image 105-140 ms unbatched (includes a JPEG round-trip), 74-131 ms/q at batch 8 (8-13 q/s); text ~0.1 s/q; video, 16 frames from the 30 s clip: 1.6-1.95 s | 38-54 ms/q with 16 in flight, 73-165 ms unbatched (first 40 per set) | Laya 30 ms/fact (2 questions) | card: 26 ms image, 504 ms video on an H200 |
| Memory / cold start | 25.3 GiB is the torch-reserved peak of Jev + SigLIP 2 base + Laya after warm-up; Jev weights alone are 22.3 GiB (11.96 B bf16). Jev loads in 27 s from the volume; container start/upload (~19 s) not reported | vLLM share 0.35 (26 GiB peak); ready in 84 s | GPU peaks 26.0 / 26.1 GiB (nvml); no flag over 90% | Placement: the integrated core's GPU 0 peaks at 60-66 GiB (fb-integrate me340-006 / samsclub-002 / walmart-001), not 45 GiB. Adding Jev gives 83-91 GiB, which exceeds 80 GiB, so Jev needs the third A100 |
