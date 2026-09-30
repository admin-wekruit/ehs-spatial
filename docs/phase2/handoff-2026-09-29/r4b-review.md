# Round 4 (r4b/integrate 40a153e) independent review — 2026-09-29

Saved from the review agent's structured output (workflow run wf_bfb33ee8-b46).

## Verdict

PARTIAL. Not accepted as "all four acceptance items met". Four parts are met: segmentation, physical info and display-model presence on all three videos, and types on Sam's Club and Walmart. ME340 types are borderline: the family is right on 26/51 = 0.51 held-out items, and my 20-card audit found 4 wrong and 5 unclear. Two explicit conditions of the acceptance are not met. First, "VLM last, cluster medoids only": Qwen's scene vocabulary runs at 0.4–8.6 s and gates SAM 3's wave-2 words, which account for 77–90% of the objects found; there are also 2 event-caption requests per call, and every on-demand click sends its own Qwen question. Second, "hazard VLM questions off": both event-caption requests on every call ask for PPE (helmet, hi-vis vest) and a workplace-safety note, and the vocabulary prompt asks for EHS / trip-hazard types. The integration is merged, self-checks pass and the numbers re-derive. It is usable as the object layer once the two VLM items are fixed or explicitly signed off by the user.

## Corrected table

| (warm call; first call after boot in brackets) | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| **1. Types:** object cards with a family type | **95%** (98%) of 376 (370) | **100%** (100%) of 917 (923) | **98%** (98%) of 896 (906) |
| cards with a specific name [round 3 warm] | 20% (48%) [98%] | 53% (61%) [100%] | 64% (62%) [99%] |
| builder's 30-card audit: type right / close / wrong / unclear | 14 / 5 / 3 / 8 | 21 / 2 / 3 / 4 | 29 / 1 / 0 / 0 |
| reviewer's 20-card audit (seed 4242), same labels | 9 / 2 / 4 / 5 | 12 / 6 / 0 / 2 | 19 / 1 / 0 / 0 |
| held-out items: family right, warm / first / round 3 warm | **0.51** / 0.62 / 0.44 | 1.00 / 1.00 / 1.00 | 0.83 / 0.83 / 0.83 |
| held-out items: specific names right / right-or-close / named, warm [round 3 warm] | 3 / 6 / 10 [20 / 30 / 51] | 14 / 15 / 15 [24 / 26 / 28] | 13 / 13 / 15 [21 / 24 / 25] |
| **2. Segmentation:** object cards with pick-map pixels on at least 1 keyframe (corrected: the builder's 100% counted entity-list membership) | 376/376 | 912/917 (99.5%) | 895/896 (99.9%) |
| object cards with outline polygons on at least 1 keyframe | 376/376 | 908/917 (99.0%) | 892/896 (99.6%) |
| maps are per keyframe, not per frame (keyframes / 899 frames) | 147 | 125 | 125 |
| builder's 60 random clicks: right card / nothing / wrong / background right / background hit | 27 / 4 / 0 / 28 / 1 | 35 / 11 / 0 / 13 / 1 | 33 / 6 / 1 / 20 / 0 |
| reviewer's 20 random clicks (seed 5151), same labels | 11 / 0 / 0 / 8 / 1 | 15 / 0 / 0 / 5 / 0 | 12 / 2 / 0 / 6 / 0 |
| clicks on a thing that opened the right card: builder / reviewer | 0.87 / 11 of 11 | 0.76 / 15 of 15 | 0.82 / 12 of 14 |
| same clicks: right card now, nothing on round 3 warm (builder) | 6 | 18 | 18 |
| outline right / partial / wrong / unclear: builder 30 ; reviewer 20 | 17/8/1/4 ; 12/5/0/3 | 28/0/0/2 ; 15/4/0/1 | 28/0/2/0 ; 18/2/0/0 |
| delivered objects (clean set) covered, in pieces, wrong-merge cards [round 3 warm] (builder's scorer, not re-run) | 80/105, 7, 4 [53, 1, 3] | 79/91, 10, 4 [41, 5, 0] | 61/74, 2, 0 [38, 3, 0] |
| **3. Physical:** object cards with every required field; contract violations over all cards versions, both calls | **100%**; 0 | **100%**; 0 | **100%**; 0 |
| position / top / base: a value or a bound | 100% | 100% | 100% |
| width: value / 'at most' bound / other | 41% / 40% / 19% | 23% / 62% / 15% | 43% / 48% / 9% |
| depth 'not observed' (seen from one side; a visible-depth lower bound is given) | 85% | 98% | 93% |
| tilt with a value (including 'needs review') | 12% | 10% | 0% (reasons: 3 or fewer views, loose fit, plumb check) |
| physical plausible / implausible / unclear: builder 30 ; reviewer 20 | 24/1/5 ; 15/0/5 | 29/1/0 ; 14/2/4 | 28/2/0 ; 17/0/3 |
| **4. Models:** object cards with a display model (present from cards v1) | **100%** | **100%** | **100%** |
| model kinds: box / plane / cylinder / open frame (box share) | 317/33/20/6 (84%) | 818/33/42/24 (89%) | 781/55/31/29 (87%) |
| person cards with a model | 0/6 | 0/16 | 0/7 |
| SAM 3D meshes accepted / eligible / attempted | 1 / 68 / 30 | 5 / 167 / 30 | 4 / 234 / 30 |
| model plausible / implausible / unclear: builder 30 ; reviewer 20 random ; reviewer 20 stratified (every SAM 3D mesh, then cylinders, planes, open frames) | 26/3/1 ; 14/3/3 ; 12/4/4 | 19/9/2 ; 14/5/1 ; 11/7/2 | 19/11/0 ; 11/8/1 ; 12/6/2 |
| **Times** (s from MP4 bytes in the container to Volume commit), warm (first): first 3D | 18.0 (23.7) | 15.3 (15.8) | 13.1 (16.2) |
| objects v1 | 32.2 (36.5) | 24.7 (25.3) | 21.8 (25.1) |
| cards v1 (types, physical and primitive models on the first-wave cards) | 43.1 (49.4) | 30.1 (36.4) | 31.2 (38.6) |
| types, first pass | 60.3 (68.4) | 65.1 (70.9) | 60.3 (64.8) |
| cards v3 [round 3 warm] | 83.5 (89.9) [77.6] | 74.8 (81.3) [47.3] | 83.0 (89.6) [50.1] |
| types, densify pass (every card typed) [round 3 'all names'] | 97.8 (104.4) [86.3] | 87.5 (97.2) [71.7] | 93.1 (99.2) [73.6] |
| first SAM 3D mesh / models final (corrected: Volume commit; the builder's were queue times) | 176.6 / 182.8 (135.6 / 190.3) | 105.6 / 180.3 (122.2 / 191.8) | 141.9 / 182.4 (156.5 / 194.2) |
| splat preview committed (corrected) / call end | 226.7 / 230.2 (232.7 / 237.2) | 226.7 / 228.5 (225.9 / 245.6) | 226.3 / 226.7 (226.6 / 227.5) |
| cold start, not counted as analysis | 181.6 | 157.3 | 171.9 |
| **GPU** peak GiB, GPU0 / GPU1, warm (first); stages over 72 GiB | 70.0 / 58.1 (67.7 / 58.9); none | 70.2 / 65.7 (68.8 / 65.5); none | 64.0 / 61.9 (63.8 / 61.8); none |
| **VLM** requests per call, warm (first): total | 51 (70) | 35 (35) | 22 (21) |
| naming: one question per cluster medoid, after the cheap votes (54–100 s) | 48 (67) | 32 (32) | 19 (18) |
| scene vocabulary at 0.4–8.6 s: gates SAM 3's wave-2 words (objects found only through its words: 77% / 90% / 90%); the prompt asks for 'ehs_relevant' types, including trip hazards | 1 (1) | 1 (1) | 1 (1) |
| event captions at 3.7–21 s: each asks for PPE (helmet, hi-vis vest) and a 'safety_note' | 2 (2) | 2 (2) | 2 (2) |
| judge / hazard stage; Qwen per-object decider; escalations; Gemini | 0; 0; 0; 0 | 0; 0; 0; 0 | 0; 0; 0; 0 |
| on-demand click (not benched) | 1 Qwen naming question per click on no entity | same | same |
| naming bank rows at load -> after (shared Volume file; grows on every call) | 1798->1861 (first); 1861->1906 | 1906->1934; 1934->1963 | 1963->1980; 1980->1998 |
| bench spend, list-price upper bound: app lifetime (bench summary) | $1.78 ($1.77) | $2.16 ($2.15) | $1.44 ($1.41) |

GT table: the builder's numbers stand. I re-derived them from r4b-gt-001/002 accuracy/rows.json; they match within filter differences (for example ARKit 42 base is 13.5 cm on values only, 15.8 cm including 'needs review'). TUM depth is 5.5 (4.5) cm against round 4 physical's 3.7 (2.6) cm. Add this row: width 'at most' bounds that hold, at the assumed / true camera height: TUM 81/98 / 74/98, ARKit 47 29/36 / 26/36, ARKit 42 12/13 / 11/13.

## Problems

- **major** — Hazard/PPE VLM questions are still asked on every call. The 2 event-caption requests per call (stage vlm.events, 3.7–21 s) use modal_apps/video_events.PROMPT. It asks Qwen for 'ppe': {'helmet': yes|no|unknown, 'hi_vis_vest': yes|no|unknown} and a 'safety_note: anything that could matter for workplace safety'. The stored events layers carry, for example, 'helmet: no, hi_vis_vest: no' for the ME340 presenter and the Sam's Club shoppers. vlm.VOCAB_PROMPT also asks for 'ehs_relevant' types ('trip hazards', 'stored items that could fall or block a path'). The builder's table shows 'hazard+judge 0' and the summary says hazard questions are off: that counts judge stages only.
  - fix: Drop 'ppe' and 'safety_note' from the events prompt (caption and action only), or add an option that turns events off and make it the default. Drop the ehs_relevant split from VOCAB_PROMPT. Re-run one warm call per video and report the VLM prompts by purpose.
- **major** — The VLM comes first, not last. Qwen's scene vocabulary (1 request at 0.4–8.6 s, before any object exists) sets SAM 3's wave-2 words. Objects found only through those words are 832/1087 on ME340 (77%; 88% with the core words), 1533/1703 on Sam's Club (90%) and 1238/1375 on Walmart (90%). Those SAM 3 words are then the cascade's 'cheap' vote. The bench has no way to turn it off (--vocab qwen|gemini). On-demand clicks (kept from round 3) send one Qwen naming question per clicked object, not per cluster medoid. The builder mentions '1 scene-vocabulary request' but not that it drives detection.
  - fix: Seed wave 2 from the fixed 102-class taxonomy / bank classes, or from a frozen per-site list, and measure coverage and types against this bench. Otherwise get the user's explicit sign-off that the detector word list is exempt from 'VLM last'. Route on-demand naming through the cascade (bank / zero-shot first) and ask Qwen only when it is unsettled.
- **major** — ME340 is not 'most objects recognised by type'. The family is right on 26/51 = 0.51 held-out items (0.62 first call, 0.44 in round 3). My 20-card audit: 9 right, 2 close, 4 wrong, 5 unclear. Wrong examples: a machine region and a ceiling light typed 'material / part', a roller-door rail typed 'guard / barrier', a 12 cm floor object typed 'machine'. 'material / part (type only)' is a catch-all on 104/376 cards, and it is the shown type on most of the wrong held-out items (tool holders, cabinet doors, a fan, a vise). The summary's 'The four acceptance items are met' overstates this: the 95% is the share of cards that carry a family label, not the share that is correct.
  - fix: State the result per video (ME340 types not met). Add shop-floor bank rows and calibration. When the family vote is weak, show 'unidentified' (shape type) instead of the 'material / part' fallback.
- **minor** — Display models are present on 100% of object cards, but 84–89% are boxes and many are implausible. My audit, plausible / implausible / unclear: 20 random cards 14/3/3, 14/5/1, 11/8/1; 20 stratified cards 12/4/4, 11/7/2, 12/6/2. The failures are boxes that run past a single slipper, pack or bag, cylinders much wider than thin cords and water packs, and open frames that do not match the shelf. The SAM 3D meshes all looked right (5/5, 4/4, and 1/1 in the viewer; the audit tile misplaced the ME340 mesh), but only 1/5/4 are accepted per call. Person cards have no model (0/6, 0/16, 0/7).
  - fix: Fit to the mask-core points and cap at the visible extent (the builder's own plan). Cap a cylinder's radius at the mask width. Use an open frame only when uprights are seen. Say in the table that person cards have no model.
- **minor** — Times table: the 'first SAM 3D model', 'models final' and 'splat preview' rows use the queue (put) time, not the Volume commit the header states. They are 1.6–4.1 s early. Volume-commit values, warm (first): ME340 176.6/182.8 (135.6/190.3), Sam's Club 105.6/180.3 (122.2/191.8), Walmart 141.9/182.4 (156.5/194.2); splat 226.7 (232.7), 226.7 (225.9), 226.3 (226.6). The table also never says that primitive display models are on the cards from cards v1 (43.1/30.1/31.2 s).
  - fix: Use written_s (written.json / run.layers) for every row, and add a 'models on every card' time.
- **minor** — The segmentation shares are rounded up. 'Pick region 100%' is membership in the entity list. Cards with pick pixels on at least one keyframe: 376/376, 912/917, 895/896; with outline polygons: 376/376, 908/917, 892/896; none of the gaps carries a reason. Outlines are per keyframe (147/125/125 maps over 899 frames), not per frame.
  - fix: Count pixels, not list membership. Give the 6–13 cards without a region a stated reason, or drop them from the clickable set.
- **minor** — Width bounds are weak on retail. Width is only an 'at most' bound on 62% of Sam's Club cards, 48% of Walmart and 40% of ME340. Against GT those bounds fail 8–19% at the assumed camera height and 15–28% at the true height (TUM 81/98 and 74/98 hold, ARKit 47 29/36 and 26/36). On one-sided shelf goods the visible along-aisle extent is shown as 'depth >= x' while 'width <= 0.11–0.24 m' is smaller than the visible face: 2 of my 20 Sam's Club cards, and 342/572 Sam's Club width bounds sit below the card's own visible depth. The summary says nothing about this.
  - fix: Report an unresolved one-sided width as 'not observed' (or 'at least' the visible extent) instead of 'at most'. Report the bound-hold rate beside the value coverage.
- **minor** — The naming bank is mutable shared state on the Modal Volume. Every call appends its VLM answers and saves: 1531 rows as baked by r4/naming, 1798 at r4b's first call, 1998 after. It includes rows from other agents' runs, so results depend on run order and cannot be reproduced. Concurrent writers can lose rows (tmp.replace). Lookups exclude the same video, so there is no direct leakage. Related: ME340's specific-name share swings from 48% to 20% between two calls on the same video.
  - fix: Benches load a frozen bank snapshot read-only and record its hash in the run. Write back offline after review.
- **minor** — Judgement is not paused by default. The core uses opts.get('judge', True), and r4b's bench keeps --judge default 'on'; r4/integrate f49a7a9 had set the bench default to off, and r4b did not take it. The r4b benches passed judge off, so the numbers are unaffected, but a default run executes the judge rules.
  - fix: Default judge off in core.py and fast_report_bench.py.
- **minor** — The viewer's 3D pane labels objects with objects-layer words ('unidentified object', 'cnc machine', 'metal pipe') instead of the cards' types. 'unidentified object' therefore still appears even though every card is typed; it is visible in the ME340 and Walmart screenshots.
  - fix: Label the 3D boxes with the card's shown name or type (through the aliases).
- **minor** — Small inaccuracies in the report. 'The builder has not written a results page' is wrong: runs/r4-models-results/results.md has existed since 16:41 (final code, bench 003). Bench spend is $1.78/$2.16/$1.44 by app lifetime but $1.77/$2.15/$1.41 in tables.md and the bench summaries. r4/models' last commit dc54750 (sheet --sam3d, scripts only) is not merged. 'Values within ±u 0.88–0.99' is 0.88–1.00 when the angle rows are included.
  - fix: Correct the text and name the source used for spend.
- **minor** — Possible overlap with other agents' benches. modal app list shows panoptes-r5-models-bench alive 17:42:41–17:52:28 and panoptes-recgen-fast-bench 16:36:31–17:52:55, while r4b's ME340 (17:10–17:24), Sam's Club (17:25–17:42) and Walmart (17:43–17:54) benches ran. Whether those apps had GPU tasks then can no longer be checked, so 'one at a time behind an idle check' is unverified.
  - fix: Log the idle check's app/task snapshot into each bench folder.

## Summary

Everything was re-derived on the CPU only; no GPU jobs were run, and the only Modal command was `modal app list`. I parsed the six warm and first calls from the saved mirrors and call records myself, with the scratchpad scripts derive.py and sheets.py. The self-checks for the changed modules and scripts and fast_report_app --self-check pass. The branch is merged: every input branch is an ancestor of r4b HEAD 40a153e, except r4/models dc54750, which only touches scripts. It is not pushed, and every commit carries the trailer.

**What reproduces exactly:**
- object-card counts
- family-typed and specific-name shares
- physical completeness (0 contract violations over all cards versions)
- model kinds and SAM 3D 1/5/4 of 30
- GPU peaks, at most 70.2 GiB, with no stage over 72
- cold starts, recorded separately from analysis time
- first and warm call times from the MP4 in the container to the Volume commit
- VLM counts 48/32/19 (67/32/18)
- GT medians, from rows.json
- spend, $12.04 against the 18 USD cap; the three bench and two GT app lifetimes check out, and the aborted runs could not be checked

**Corrections to the table:**
- The models and splat time rows are queue times, 1.6–4.1 s early.
- The 'pick region 100%' is really 99.5% on Sam's Club (912/917 cards) and 99.9% on Walmart (895/896) by pixels. Outline polygons are on 99.0% (908/917) and 99.6% (892/896).

**My own look** (20 random cards, 20 random clicks and 20 stratified models per video; sheets in scratchpad/rev/{cards,models,clicks}):
- Clicks: 38/40 on-object clicks opened the right card, with 1 background hit.
- Outlines: mostly right.
- Physical: mostly plausible. 2 Sam's Club cards have a width bound smaller than the visible face.
- Models: present everywhere, but on random cards 3/20 are implausible on ME340, 5/20 on Sam's Club and 8/20 on Walmart. The SAM 3D meshes all looked right.
- Types: ME340 9 right, 2 close, 4 wrong, 5 unclear; retail 18/18 and 20/20 right-or-close among labelled cards.

**What the builder missed or understated:**
- Every call asks Qwen for PPE (helmet, hi-vis vest) and a safety note in its 2 event-caption requests. The vocabulary prompt asks for EHS and trip-hazard types.
- The scene vocabulary at 0.4–8.6 s supplies the words behind 77–90% of detected objects, so the VLM is first, not last.
- On-demand clicks each send a Qwen question per object.
- ME340 types are right on only about half the held-out items (0.51), so 'four items met' overstates.
- The naming bank is shared state that grew 1798 -> 1998 rows during the benches.
- The judge still defaults on in the core and the bench.
- Width 'at most' bounds, the most common retail width state, fail 8–28% against GT.

**Verdict:**
- Met: segmentation, physical info and display-model presence on all three videos; types on Sam's Club and Walmart.
- Borderline: ME340 types.
- Not met: 'VLM last, cluster medoids only' and 'hazard VLM questions off'. Both are small code changes (the events prompt, the vocabulary source), or they need the user's sign-off.

Files: /private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/rev/derived.json, the same folder's derive.py and sheets.py, and its cards/, models/ and clicks/ folders. Source evidence: fast_report/core.py:731-771 (vocabulary then events), modal_apps/video_events.py PROMPT (ppe, safety_note), fast_report/vlm.py VOCAB_PROMPT (ehs_relevant), fast_report/cascade.py Bank.save, fast_report/ondemand.py name().
