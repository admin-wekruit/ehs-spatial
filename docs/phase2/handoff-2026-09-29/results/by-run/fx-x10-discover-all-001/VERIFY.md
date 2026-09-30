# Adversarial verification of x10-discover-all: holds-with-corrections

These corrections take precedence over results.json conclusions.

The experiment holds with corrections. Every reported number re-derives exactly from the saved masks, labels and records: pool recall (45 sets), own-look recall (45 sets), mask composition (32 sets), reference counts, per-frame times, stack times, memory peaks (max 19.3 / 40.5 GB for the recommended stack; nothing over 90% in any run), boot times (88–116 s) and spend ($8.63 upper bound). Both self-checks pass. Both commits carry the trailer and nothing was pushed.

The main answer is robust. SAM 2.1 AMG finds little on its own in the tested settings: own-look 0.27 [0.22–0.32] / 0.47 / 0.60 with frame-bootstrap intervals, and 0.28 / 0.50 / 0.70 on my 4 held-out frames. The recommended union stack clearly beats the vocabulary alone:
- Own look: 0.69 → 0.85, a gain of +0.16 [0.11–0.22].
- Labels: 0.40 → 0.70.
- My held-out frames: 0.59 → 0.85, labels 0.33 → 0.80.

Corrections needed:
1. **Stack table.** The '+AMG 16x16' row mixes a ridge v1 recall with a ridge v2 time; the timed stack adds 0 own-look items. The '+AMG 32x32' row is not the recommended stack plus AMG32; the real recommended + AMG32 is 0.77 / 0.87 and was never timed.
2. **Stack timing.** Timing stops at GPU completion. It leaves out the dedupe/people/size filters and writing the result, estimated at +1.5–3 s. So 15.0–19.8 s is a lower bound, about 16.5–22.5 s.
3. **In-sample design.** Label words and ridge v2 were designed after the own look was drawn and were scored on it. This is not disclosed, although my held-out check supports the result.
4. **Label noise.** The pool reference is noisy at the item level: I clearly agree with only 22 of 35 audited W/T/K items (84% of tile letters overall). That noise produces a 0.79 pool recall for recommended + geometry, which the report does not mention and which the own look does not confirm.
5. **Smaller issues.** AMG was not tried at denser settings. The 5 fps time is a lower bound. The 20–30 s budget cannot be traced to the user request. The delivered publications were not used. Licences check out against primary sources but no URLs are given.

I ran no GPU jobs and spent $0. My scripts and held-out boxes (verifier-labelled) are in /private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/verify-x10/: rederive.py, boot.py, pool_extra.py, geo_items.py, post_time.py and heldout-look.json.

## Problems
- [major] Two rows of the stack table do not describe the stacks they are labelled as. The recall in the '+ AMG 16x16' row (0.74 / 0.84) was computed with ridge v1 (vocab+generic+owlbox+labelw+ridge+amg16). Its time (24.3–30.2 s) was measured with ridge v2. The stack that was actually timed scores 0.747 pool and 0.851 own-look, which is +0 own-look items over the recommended stack, not '+1 item'. The '+ AMG 32x32' row is catch-all + OWLv2 boxes + AMG32, with no label words and no ridge v2. Placed under the recommended row, it reads as if adding AMG lowered recall (0.84 < 0.85). The recommended stack + AMG32 scores 0.769 / 0.868 (+6 own-look items) and was never timed.
  Fix: Relabel or replace the rows with the recomputed values (see correctedTable). State that recommended + AMG32 is untimed: about 44–47 s, inferred additively. The conclusion that AMG is not worth it still holds.
- [major] The timed stack window omits steps. stack_time() stops when both GPUs finish. It does not apply the 'as used' filters that the recall numbers depend on: SAM 3 dedupe (generic masks drop from 136 to 39 per frame), people-out and size limits. It also does not copy the SAM 3 and OWLv2-box masks to the CPU or write any result. This breaks the 'inputs → result written' rule. On my Mac those filters cost about 0.11 s per frame, which is roughly +1.5–3 s per video.
  Fix: Time the full path, including filters and writing the masks and layer, inside the container. Until then, report 15.0–19.8 s as a lower bound (about 16.5–22.5 s, inferred).
- [major] The recommended stack was designed on the same data it was scored on, and this is not disclosed. label words (LABEL_WORDS) and ridge v2 (curved-line filter) were created after the own-look boxes were finished: look.json 23:25, ridge2 23:31, words 23:35. Both were then scored on the same 25 frames. I found no held-out split. Also, the own look was drawn after the agent had audited 2,835 method outlines from those same frames, so it is not fully independent of the methods.
  Fix: Disclose this. Report a held-out check. My check on 4 held-out frames (46 items, verifier-labelled; I had seen some lightning-219 tiles beforehand) supports the direction: vocabulary 0.59 → recommended 0.85, labels 0.33 → 0.80. Add frame-bootstrap intervals.
- [major] The pool reference has substantial noise at the item level. Re-labelling 4 contact sheets (192 tiles), I agree with 84% of tile letters but only 22 of the 35 W/T/K reference items clearly (4 wrong, 9 ambiguous). The agent also missed at least 2 real items: a beam price label (samsclub 275.407) and a sign (walmart 527.110), both marked P. Retail product faces are labelled inconsistently: W in Walmart (527.267, 527.232) but P under the block rule. Of the 19 items that geometry adds to the stack, about half are strips or regions of benches and floor labelled W or T. For example, me340-486-345 is the floor inside a copper tube's curve, labelled T. Tiles also show method word hints, which may bias labels toward named masks.
  Fix: Report the pool recall together with this noise level. Treat the own-look and held-out recall as the primary numbers. Have a human audit the W/T/K tiles, starting with items found by only one method.
- [minor] Geometry is dismissed only on its solo recall (6–9%). Its measured complementary gain is not mentioned: vocab+geosam = 0.669, the second-largest single addition, and recommended+geosam = 0.790 (thin 0.61 → 0.74) for about 0.16 s per frame, more than AMG32 adds. On the own look it adds 0 items; on my held-out frames it adds +1 item. The stated reason, 'floor cables sit inside the depth noise', is inferred but not marked as inferred.
  Fix: Report the stack-level pool gain and the own-look null result. Explain the gap as label noise. Mark the depth-noise sentence as inferred.
- [minor] The AMG baseline covers only 16/32 grids, at most 1 crop layer, pred_iou 0.8 and stability 0.92. Denser settings were not tried: 64 points, use_m2m, lower thresholds (Meta's dense example). Recall rises steadily with density: own look 0.27 → 0.47 → 0.60, my held-out frames 0.28 → 0.50 → 0.70. 'SAM 2.1 cannot cut out everything' is therefore stronger than the data.
  Fix: Say: 'not at affordable cost in the tested settings (32x32 with one crop layer is already 3.7 s per frame)'.
- [minor] The 5 fps stack time (41.8–53.7 s) leaves out the core state for the extra keyframes. The core computes SAM 3 features on every 5 fps keyframe but keeps them only for object keyframes (SamWork caches only frames where (x+j)%3 == 0). The stack needs them all, which costs either unmeasured memory or about +6–9 s. The '+4–6 points' 5 fps gain is per method (unweighted counts, 3.6–5.6), not for the recommended stack.
  Fix: Label 41.8–53.7 s as a lower bound (about 50–62 s if the features are recomputed). Score the stack-level 5 fps gain, or say it was not measured.
- [minor] The '20–30 s budget' and the judgement that 5 fps is 'over budget' cannot be traced to the relayed user request. The user said 44–60 s is acceptable and that minutes are fine, rather than tens of minutes.
  Fix: Cite the budget's source, or present 5 fps as an option at about 50–62 s rather than ruling it out.
- [minor] The delivered publications (c40fbd08 for ME340, 913daf2a for Sam's Club, 32cec650 for Walmart) are never used. There is no cross-check of the agent-built reference against the delivered object maps, so this check could not be done and is unverified.
  Fix: Map the delivered objects to the reference keyframes and report recall on them, or say explicitly that this was not done.
- [minor] Precision of the recommended stack is unknown: label-word and ridge v2 masks were never audited, and masks per frame for the stack are not reported. There is no hard-negative set and no calibration, although neither was claimed.
  Fix: Audit a sample of labelw and ridge2 masks. Report masks per frame and the whole/part/background shares for the recommended stack.
- [minor] Licences are correct but no sources are given. I checked each against its primary source: HF cardData for SAM 2.1, OWLv2, RAM++ weights and Qwen3-VL-8B is apache-2.0; RAM++ code is Apache 2.0; DA3-GIANT-1.1 is cc-by-nc-4.0; SAM 3 is 'other', i.e. the SAM License, which allows commercial use with trade-control, military and ITAR restrictions; YOLO-World is GPL-3.0. Separately, the own-look rule says 6 frames per video but Lightning has 7 (25 frames total). Spend ($8.63) is an upper bound computed from wall time × list price and includes an estimated $0.45; it is not reconciled with billing.
  Fix: Add licence URLs. Fix the own-look rule text. Label the spend as an estimate.

## Corrected table

Every number below was re-derived from the saved masks, audit labels and own-look boxes. I ran the branch's own scoring (scripts/x10_eval.py score / look_score) on the scratch copy (scratchpad/x10/run001) and got 0 differences from results.json. That covers pool recall (all 45 sets), own-look recall (45 sets), mask composition (32 sets), reference counts and per-frame times. Stack times, memory peaks, boot times and spend also match the raw records. Corrected or added cells are marked [C].

Single methods: every number in the claimed table is unchanged. Two notes:
- 3D geometry → SAM 2.1 [C]: on top of the recommended stack it raises pool recall from 0.739 to 0.790 (thin 0.61 to 0.74) for about 0.16 s per frame, but adds 0 own-look items. Most of that pool gain sits on doubtful labels (see problems).
- Frame-bootstrap 95% intervals on own-look recall (25 frames) [C]: vocabulary 0.69 [0.59–0.77]; AMG 16x16 0.27 [0.22–0.32]; AMG 32x32 0.47 [0.43–0.51]; AMG 32x32 with one crop layer 0.60 [0.56–0.65].

Stacks (added on top of the core's vocabulary):

| Stack | Pool recall | Own-look recall (labels) | Added s per 30 s video, 2 A100 |
|---|---|---|---|
| + catch-all words + OWLv2 boxes | 0.72 | 0.79 (0.51) | 6.0–8.0 (GPU work only) [C] |
| + ridge | 0.73 | 0.80 (0.52) | 7.9–12.2 (GPU work only) [C] |
| **+ label words + ridge v2 (recommended)** | **0.74** | **0.85 [0.79–0.90] (0.70 [0.58–0.82])** | **15.0–19.8 measured GPU work. Filtering (dedupe, people out, size limits) and writing the result are not timed; estimated +1.5–3 s, so about 16.5–22.5 s (inferred)** [C] |
| recommended + AMG 16x16 [C] | 0.75 | 0.85 (0.70), +0 items | 24.3–30.2 (measured) |
| recommended + AMG 32x32 [C] | 0.77 | 0.87 (0.71), +6 items | not measured; about 44–47 s (inferred, additive) |
| catch-all + OWLv2 boxes + AMG 32x32, without label words or ridge v2 [C: this is what the claimed '+ AMG 32x32' row measured] | 0.75 | 0.84 (0.61) | 33.9–37.8 (measured) |
| recommended + geometry → SAM 2.1 [C] | 0.79 (thin 0.74) | 0.85, +0 items | not measured |
| + label and catch-all words on tiles | 0.78 (tiny 0.70) | 0.86 | about 35–45 (estimated) |
| All methods and the listing | 1.00 by construction | 0.90 (0.73) | minutes (estimated) |
| Recommended stack on every 5 fps keyframe | — | — | 41.8–53.7 measured. Lower bound [C]: excludes SAM 3 features and person masks for the 5 fps keyframes that are not object keyframes (core state rises from 4.3–7.7 s to 12.9–15.9 s) |

Gains, frame bootstrap [C]:
- Recommended vs vocabulary: +0.16 [0.11–0.22].
- Recommended vs catch-all + boxes: +0.06 [0.03–0.11].
- Labels vs vocabulary: +0.30 [0.17–0.43].

Held-out check [C]: my own boxes (verifier-labelled) on 4 reference frames that are not in the experiment's own-look set (me340-400, samsclub-221, walmart-217, lightning-219); 46 items.

| Set | Recall (labels) |
|---|---|
| Vocabulary | 0.59 (0.33) |
| Catch-all + OWLv2 boxes | 0.76 (0.53) |
| Recommended | 0.85 (0.80) |
| Recommended + AMG 16x16 | 0.87 |
| Recommended + AMG 32x32 | 0.91 |
| AMG 16x16 / 32x32 / 32x32 + crop, alone | 0.28 / 0.50 / 0.70 |

Contact-sheet audit [C]: I re-labelled 4 sheets (me340-003, samsclub-004, walmart-012, lightning-002; 192 tiles).

| Measure | Result |
|---|---|
| Tile letters agreed | 162 / 192 (84%) |
| Clear disagreements | 7 |
| Ambiguous tiles | 23 |
| Agreement on clear-cut tiles | 96% |
| Reference items (W/T/K) with clear agreement | 22 / 35 (63%) |
| Reference items wrong | 4 |
| Reference items ambiguous | 9 |
| Items the agent missed (beam price label 275.407 and wall sign 527.110, both marked P) | 2 |
