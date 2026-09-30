# r5b models: every object's display model with the right outline; angles readable

Branch `r5b/models` (worktree `panoptes-phase2-video-r5b-models`), from `r4b/integrate` 40a153e, with `r5/models` 804a4ca, `recgen/fast` 1c137cc and `route/jev` 6cc6f27 merged. Runs: `r5b-models-commercial-002` (SAM 3D profile: ME340 first call, Sam's Club, Walmart, ME340 warm), `r5b-models-internal-002` (RecGen profile: ME340 first call, Sam's Club, Walmart), `r5b-models-gt-001` (TUM fr1-room, ARKitScenes 42445448 / 47333932), plus `r5b-models-smoke-001` and `r5b-models-{commercial,internal}-001` (before two fixes; kept for their numbers). Exactly 2 x A100-80GB per bench, `min_containers=0`, `--judge off`; Jev-Omni on its own A100 (a separate Modal class). Labels by eye are agent-labelled. Times are `written_s` (MP4 bytes in the container -> Volume commit returned).

## Answer

- **Tier 0 (observed surface) for every object card, from the first cards build**: its own depth fused (TSDF), in the video's colours, from the points its views agree on; one GLB a shot (a node per card). It lands **2.5-4.7 s after cards v1** in every call (target <= 5 s).
- **Outline by eye (30 random + 20 stratified cards a video, commercial run)**, shown model vs r4b's box on the same card: ME340 **20 / 25 / 5** right / partial / wrong vs 16 / 23 / 11; Sam's Club **26 / 19 / 2** vs 26 / 17 / 4 (3 unclear each); Walmart **37 / 12 / 1** vs 17 / 31 / 2. Every card, measured (IoU of the silhouette with the SAM 3 outline at the card's best view, median): ME340 0.53 vs 0.45, Sam's Club 0.67 vs 0.65, Walmart 0.69 vs 0.59; share with IoU >= 0.7: 20 / 42 / 47 % vs 13 / 40 / 30 %.
- **Tiers** (commercial): observed surface 95 / 65 / 82 %, checked primitive 5 / 35 / 17 %, generated 0 / 0 / 0.4 % (ME340 / Sam's Club / Walmart). Internal: generated 0.3 / 0 / 1 %.
- **Generated models are right where they pass the held-out gate, and rare.** RecGen FAST on Walmart: 8 of 9 right by eye (r4b's boxes: 0 right, 7 partial, 2 wrong), 6 accepted + 4 look-alike copies; SAM 3D: 1 right, 3 partial of 4. ME340's machines, tools and benches pass neither generator (0-1 of 20-23 tried); Sam's Club routes nothing to them (boxes and shelves: primitives).
- **Tier 1 done**: internal 146 / 106 / 118 s (ME340 first call / Sam's / Walmart: all <= 150 s); commercial 199 (first) and 189 (warm) / 123 / 171 s: SAM 3D's queued generations run past the 135 s start deadline on ME340 and Walmart (issue 1).
- **GPU**: every stage <= 72 GiB in both profiles (max 70.3 / 66.4 GiB, commercial Sam's Club); the internal profile with 804a4ca's cache fix peaks at 51.8-59.0 GiB on GPU 0.
- **Angles**: every card lists its planar parts (lettered, each angle to the floor and between touching parts +- u) or why not. Parts on 47 % (ME340) and 75 % (Sam's Club) of cards; **none on Walmart**: the plumb check failed (walls 2.3-2.5 deg off vertical, limit 2; issue 2). Against ground truth with **every part counted** (no 30 deg match gate): pairs' error median 2.6 deg, p90 17.6 deg, +- u covers 0.85 of pairs; but 57 % of our parts have no GT part near them (half of those on cards where GT finds no plane at all), so +- u covers 0.37 of all our parts; on GT-tilted parts (15-75 deg) the median error is 10.7 deg and +- u covers 0.59. On the videos, of the parts read 15-75 deg that could be judged by eye, 1 per video is truly tilted (read within 2.5 / 5.2 deg of the eye's estimate); the rest are pipes, cables, rod clusters and tools with no plane (issue 3).

## Per video

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| object cards (commercial / internal) | 369 / 370 | 921 / 918 | 904 / 907 |
| tier shares, commercial: observed surface / primitive / generated | 95.4 / 4.6 / 0 % | 64.8 / 35.2 / 0 % | 82.2 / 17.4 / 0.4 % |
| primitive kinds (commercial) | box 9, plane 7, open frame 1 | box 317, open frame 4, cylinder 2, plane 1 | box 144, open frame 9, plane 4 |
| tier shares, internal | 95.1 / 4.6 / 0.3 % | 54.5 / 45.5 / 0 % | 82.9 / 16.1 / 1.0 % |
| outline by eye, shown: right / partial / wrong (50 cards) | 20 / 25 / 5 | 26 / 19 / 2 (+3 unclear) | 37 / 12 / 1 |
| outline by eye, r4b's box on the same cards | 16 / 23 / 11 | 26 / 17 / 4 (+3 unclear) | 17 / 31 / 2 |
| outline, every card: IoU median shown vs r4b | 0.53 vs 0.45 | 0.67 vs 0.65 | 0.69 vs 0.59 |
| outline, every card: IoU >= 0.7 / < 0.3, shown vs r4b | 20 / 22 % vs 13 / 27 % | 42 / 13 % vs 40 / 11 % | 47 / 7 % vs 30 / 11 % |
| outline, every card: coverage / overflow, shown vs r4b | 0.84 / 0.36 vs 0.89 / 0.48 | 0.91 / 0.25 vs 0.91 / 0.27 | 0.90 / 0.22 vs 0.95 / 0.37 |
| generated models by eye (internal RecGen / commercial SAM 3D) | 0 right, 1 partial / none | none / none | 8 right, 1 partial / 1 right, 3 partial |
| tier 1: routed groups -> tried -> accepted + copies (commercial; internal) | 24 -> 20 -> 0; 24 -> 23 -> 1 | 0; 0 | 20 -> 20 -> 4 + 1; 21 -> 21 -> 6 + 4 |
| Jev-Omni: questions (groups), round trip | 31, 3.7 s | 20, 3.2 s | 23, 3.1 s |
| tier 0 lands: cards v1 -> surfaces v1 written (commercial; internal) | 50.0 -> 52.8 s (+2.8; first call +3.2); +2.7 | 44.3 -> 49.0 s (+4.7); +3.5 | 34.9 -> 38.3 s (+3.4); +2.5 |
| tier 1 done, written s (commercial; internal; target <= 150 warm) | 189.2 warm, 198.8 first; 146.3 first | 123.0; 106.3 | 171.3; 118.0 |
| GPU peak GiB, GPU 0 / 1 (commercial; internal); stages > 72 | 65.4 / 59.9; 53.5 / 59.1; none | 70.3 / 66.4; 59.0 / 65.6; none | 63.1 / 62.1; 51.8 / 61.6; none |
| cards with planar parts / parts with an angle / bends | 174 of 369 / 197 / 9 | 692 of 921 / 857 / 72 | 0 (plumb check failed) |
| parts' u median; parts from one view set | 16.9 deg; 58 | 12.3 deg; 351 | - |
| parts read 15-75 deg -> judged by eye -> truly tilted (read vs eye) | 45 -> 27 -> 1 (67 +- 20 vs ~65) | 26 -> 22 -> 1 (55 +- 24 vs ~50) | - |

Outline by eye: right = the silhouette follows the outline (about IoU >= 0.7), partial = on the object but part of it or running past it, wrong = mostly off it or nearly nothing drawn. The measured rows use the outlines layer's polygons (SAM 3's masks) at each card's best view, half resolution. The eye sheets are `run-002/sheets-*/sheet-*.jpg` (outline | shown | r4b), labels `run-002/sheets-*/labels.json`.

Before the rim fix (run-001, same code otherwise): tier-0 surfaces covered a median 60 / 67 / 65 % of their outlines (the lift cuts depth-edge pixels and erodes 1 px; small objects lost most) and the every-card IoU was 0.45 / 0.64 / 0.62; growing each dense view's depth image by one cell took coverage to 0.84 / 0.91 / 0.90 and IoU to 0.53 / 0.67 / 0.69 at the cost of overflow (0.19 -> 0.36 on ME340).

## What the round built

- **Tier 0** (`fast_report/surface.py`, `core.tier0_job`): after cards v1 and again after cards v3, per card (process pool): the cards' main cluster cut to the points its other views agree on (`surface.consistent`: a point stays when half of the views that see it, in frame and not behind their depth, put it inside the object's outline; ME340: 60 of 237 cards lost > 10 % of their points, mostly mask bleed onto machines and walls), a TSDF of them (1-4 cm voxels, rims grown one cell), vertex colours from the frames, <= 3000 triangles; the planar parts from the same points. One GLB a shot (`surface.shot_glb`, a node per card; ME340 4.4 MB) in a `surfaces` layer; the parts go onto the next cards puts (v2 / v4; v3 is put again with its own).
- **Primitives only when a fixed-shape class fits** (`display_model.check`, `for_card`): route/jev's fixed-shape class list names the kind (a shelf with hidden uprights falls back to its box); the primitive must stay inside the object's outline in its worst best view (overflow <= 35 %, occluders excused) and cover it (>= 60 %); a cylinder is never wider than seen (radius <= 0.55 x the width across the view); an open frame needs two uprights seen. Otherwise tier 0. `display_model.r4_record` keeps r4b's choice for comparisons.
- **One observed rule** (`fast_report/observed.py`): a face, arc sector, cap, frame member or generated-mesh vertex is observed only when a camera faced it (normal within 75 deg of the ray), saw it with clear line of sight (its depth there not nearer by more than max(3 cm, 3 %)) and object points lie on it; used by the primitives' flags and by both generators' GLB alphas (`sam3d.judge`, `r5_bench.gate_mesh`); unseen parts are drawn guessed (faint).
- **Tier 1** (`fast_report/route.py`, `core.tier1_plan / models_job`): route/jev's rule (the view gate, the fixed-shape classes, Jev-Omni Q5 with P(none of the simple shapes) > 0.5), bags and soft goods never, look-alike groups (the same shown name or type, every sorted extent within x1.25), Jev asked once per group on its outlined best view, at most 60 groups a video best-seen and largest first. Planned on the first names (cards v2), generating once the facts are done (the smoke run showed generating from densify's SAM 3 end took cards v3 82 -> 125 s and GPU 0 to 74.6 GiB), no new generation after 135 s. A group's model is copied onto its look-alikes (turned about the floor normal to each footprint, scaled by the extents) only where the copy fits the member's own outline (Walmart internal: 4 of 16 kept). Internal profile: RecGen FAST (recgen/fast's setting, 804a4ca's cache fix in its worker); commercial: SAM 3D s1cfg12.
- **Jev-Omni on its own GPU** (`fast_report/jev.py`, the report app's `Jev` class, A100-80GB, scale-down 300 s): loaded and warmed at container start (the first request took 48-60 s in run-001, 3-5 s after the warm-up); the bench wakes it before each call (cold start, not analysis time).
- **Person cards** say `person: no model`. **On-demand cards** carry their one-view observed surface as an inline GLB the viewer draws.
- **Viewer** (`web/src/live-report.ts`, `LiveReport.tsx`, `viewer/native-viewer.ts`): generated (and look-alike copies) > checked primitive > the observed-surface node of its shot's GLB > the see-through box; 3D labels are the cards' shown names or types (not the objects layer's words); a shot GLB is fetched and parsed once and each card takes its node; the model line says which tier and why; parts are lettered ('part A 31 +- 3 deg to the floor', 'parts A-B 120 +- 4 deg'). Checks: `web/tests/r5-surfaces-check.ts`, `web/tests/r5b-models-browser.mjs` (ME340 smoke and Walmart internal: model lines by tier, 3D pane redraws, preview insets, 3D labels = card names, lettered parts, shot GLB nodes load).
- **Measurement** (`scripts/r5b_models.py`: sheets, tilted, stats, ious, gt, gt-r5; `scripts/r5b_render.py`).

## Angles against ground truth, every part counted

Our parts and GT parts (the same finder on the card's GT points: its pick regions lifted with GT depth and pose) paired by centre only (Hungarian, within 0.5 x size + 0.1 m, no normal gate); an unmatched part is a miss.

| | our parts | GT parts | pairs | ours unmatched (on cards with GT parts) | GT unmatched | error median / p90 | u median | +- u covers: pairs / all ours | GT-tilted pairs: error median, coverage |
|---|---|---|---|---|---|---|---|---|---|
| this round (r5b-models-gt-001), all | 713 | 412 | 308 | 405 (203) | 104 | 2.6 / 17.6 deg | 10.6 deg | 0.85 / 0.37 | 66: 10.7 deg, 0.59 |
| ARKit 42445448 | 116 | 82 | 59 | 57 (39) | 23 | 1.6 / 12.3 | 10.6 | 0.92 / 0.47 | 12: 5.7, 0.83 |
| ARKit 47333932 | 214 | 134 | 108 | 106 (73) | 26 | 4.9 / 25.2 | 12.4 | 0.78 / 0.39 | 33: 14.9, 0.55 |
| TUM fr1-room | 383 | 196 | 141 | 242 (91) | 55 | 1.8 / 13.7 | 9.2 | 0.88 / 0.32 | 21: 8.6, 0.52 |
| r5's records re-scored (r5-models-bench-001), all | 717 | 392 | 319 | 398 (194) | 73 | 3.0 / 23.0 | 10.9 | 0.85 / 0.38 | 74: 9.3, 0.69 |

For comparison with r5's 30 deg-gated number: pairs whose normals agree within 30 deg (271 of 308) err 2.1 deg median, 12.0 p90, covered 0.93. The gate hid the mismatched pairs (37 of 308 with normals > 30 deg apart, median error 35 deg) and every unmatched part.

## Tilted parts by eye (`run-002/tilted-*`)

Every part read 15-75 deg (and the parts of cards named guard, ramp, board, panel, sign ...), drawn as its plane's rectangle on the card's best view. ME340: 48 listed, 27 judged: 1 truly tilted (a lid leaning in a yellow bin: 67 +- 20 deg, ~65 by eye), 26 not (14 pipes, cables, ducts and light fixtures and 8 rod clusters and tools, none of them a plane; 4 level or vertical within u); 21 machine parts could not be told from one view. Sam's Club: 28 listed, 22 judged: 1 truly tilted (a diagonal brace: 55 +- 24 deg, ~50 by eye), 21 not (vertical package, sign and pallet faces, level tops); the two vertical signs listed by name read 88 +- 26 and 88 +- 14. Walmart: none (plumb). No machine guard at ~30 deg is its own card in these three clips; the tilted-panel case is untested on video beyond these two.

## Spend

18.74 USD upper bound (Modal list prices x each ephemeral app's lifetime, 2 x A100-80GB + 32 cores + 160 GiB for the report container and 1 x A100-80GB + 4 cores + 48 GiB for Jev-Omni over the whole app life): smoke-001 1.65, commercial-001 5.15, internal-001 4.48, gt-001 0.90, commercial-002 3.70, internal-002 2.87. The cost ledger was not edited.

## Issues

1. **SAM 3D overruns the tier-1 deadline** (commercial ME340 189-199 s, Walmart 171 s): the deadline stops new prepares, but prepared objects' generations are already queued in the two SAM 3D processes; the worker should skip a job whose deadline passed (a deadline in the message). RecGen's queue stayed inside (<= 146 s). Not changed after the final runs.
2. **Walmart has no angles**: the cards' plumb gate (walls <= 2 deg off vertical) failed on both shots (2.3 / 2.5 deg; p90 3.0 / 4.1 deg). The parts' u already carries the plumb term; a looser gate for parts (e.g. 5 deg) would give Walmart angles with a wider u. Not changed (no budget to re-run).
3. **Planar parts are over-found**: 57 % of our parts on GT sequences have no GT part near them (half on cards where GT finds no plane), and on the videos most parts read 15-75 deg are on pipes, cables, rod clusters and tools. u is wide (median 10.6 deg on GT, 12-17 deg on the videos), so a 30 deg rule is rarely decidable. 58 / 351 parts (ME340 / Sam's Club) come from one view set (flagged 'one view set').
4. **Tier 0 is often partial** (ME340 25 of 50): thin and receding objects (ceiling pipes, cables) lose their ends to the lift's depth trim, the rim growth adds overflow on small objects (ME340 overflow median 0.36), and some cards' own points span neighbours (the cards' segmentation; `surface.consistent` cuts the worst for the surface only, the card's measurements still use every point).
5. **Cards v1 is 1-1.5 s later** (the primitives' checks: `models_cpu` 23-36 s CPU a build on 24 processes). The absolute times of these runs are 6-16 s later than r4b's already before cards v1 (SAM 3 and the merge ran slower on these hosts: `merge` 7.5-9.8 s vs 3.6-5.0 s in r4b), so r4b's times are not comparable one to one.
6. **Names differ between the two profiles' runs** (Sam's Club primitives 324 vs 418): the naming cascade's shared bank on the Volume grows between runs (the review's item 7); the tier shares move with the names.
7. The on-demand card's surface is unit-tested (ondemand self-check, the viewer document check) but was not clicked live in a browser this round.
8. `web/tests/live-report-check.ts` needs `runs/fb-c-fixture-001`, which is not on this Mac (it fails before and after this round); its model assertion was updated to the tiers.
