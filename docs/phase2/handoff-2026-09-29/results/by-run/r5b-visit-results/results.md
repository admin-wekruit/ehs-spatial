# r5b / visits: the same site, visited again

Branch `r5b/visit` (from r4b/integrate 40a153e). A new visit's video is registered into the first visit's site map, its objects
are matched to the site map's, identities / names / display models carry over, and the differences are a `visits` layer on
the new visit's report that the live viewer shows in a Visits tab. Judgement stays paused: no hazard, PPE or safety question
is asked anywhere in these runs (event captions off, vocabulary prompt without its safety list; see Integration).

Numbers below come from `results.json` / `tables.md` (this folder), made by `scripts/r5b_visits_eval.py results` on the final
bench `r5b-visit-002` (13 site-map / revisit pairs through `FastReport.visit`, 3 revisits analysed with their visit step on the
analysis clock) over the analyses of `r5b-visit-001` (14 videos). Labels by eye are agent-labelled (`labels.json`).

## How a revisit is compared (fast_report/visits.py)

1. **Retrieval.** DINOv2-L (the naming encoder, GPU 1) describes every keyframe of both visits; each shot of the revisit (B)
   votes for the site map (A) shot it looks like most; 10 B keyframes (4 apart) and, for each, its 2 closest A keyframes, then
   their neighbours +-5 keyframes (up to 20 A views: a wider camera spread for the fit).
2. **Registration** (register_cut_shot's method and gate). One DA3-GIANT any-view forward on the ~30 frames (GPU 0). DA3's A cameras
   are carried onto A's own (rotation average; scale from the depth ratio against A's pick depth *or* from the camera centres;
   translation from the centres; views over 3x the median residual trimmed and the fit made again); B's own cameras then give
   the similarity B shot -> A shot the same way. Refused unless both fits pass the gate (centre residual <= 0.1 of the cameras'
   spread, at least 0.25 m; rotation residual <= 3 deg; depth-ratio spread <= 15 %) and the two floors agree (tilt <= 5 deg,
   shift <= 0.2 m). Of the two scales, the one that puts B's floor nearer A's wins, and the similarity is then snapped to the
   floors (a turn about +z, a floor-plane shift, a scale), so heights compare directly. u = both fits' centre residuals, both
   maps' own pose u, the rotation residual at the scene depth.
3. **Matching.** Every card of the two shots in A's floor frame (B's scaled). A pair needs: centres within max(3 x pose u, half
   the smaller object, 15 cm) or the smaller footprint half inside the larger; height ranges overlapping; DINOv2 cos >= 0.5
   (0.7 when two specific types differ); **the grown volumes' IoU >= 0.1** (or a part >= 80 % inside that looks alike). Cost =
   distance / gate + appearance + type + volume overlap - same name; one to one (Hungarian).
4. **See-through test** (fast_report.timeline.place, X6's rule, pose u = the registration's). A match survives only if neither
   visit sees the other's object's place empty. An unmatched object's own points (its pick masks lifted with its depth) in the
   other visit's keyframes: free -> **missing** (A's) / **new** (B's); occupied -> "there, delineated otherwise" (no claim);
   out of view, occluded, unjudged -> **not observed** (never missing or new). A missing and a new object that look alike
   (cos >= the pair's negative 99th percentile), of a compatible type and size, far apart -> **moved** (one site id).
   **Changed height** only when the part one visit has and the other lacks is seen through by the other visit;
   **changed angle** only on a strong match (look, type, place) with both angles measured. Every claim: +-u and an evidence
   JPEG (A's keyframe | B's keyframe, the object outlined where segmented, dashed where only projected).
5. **Carry-over.** Every B card gets a site id (`v0:<A card>` when matched or moved, else `v1:<B card>`); A's specific name /
   type when B has none; A's display model (and A's SAM 3D mesh reference when A has one) placed in B's frame; a missing
   object's model stays as a ghost where it stood.

## Results per revisit (GT: TUM fr1 mocap for every TUM pair; ARKit sessions aligned by ICP of their LiDAR, fitness 0.83, rmse 2.9 cm)

| B (revisit) | A (site map) | kind | shots registered | u (cm) | GT error cm (median / p90), deg | objects compared | match P (surface / box) / R | claims right / wrong / unclear | GT changes found (all / >= 30 cm) | false changes / judged | visit s (to commit) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TUM room 30-45 s | TUM room 0-30 s | TUM | 1/1 | 8 | 5.0 / 7.3, 0.99 | 98 | 0.79 / 0.79 / 0.92 | 0 / 0 / 0 | 0 of 0 | 0 / 98 | 9.0 |
| TUM desk | TUM room | TUM | 1/1 | 8 | 6.3 / 7.6, 1.23 | 83 | 0.79 / 0.82 / 0.93 | 0 / 0 / 1 | 0 of 0 | 0 / 83 | 9.6 |
| TUM desk2 | TUM room | TUM | 2/2 | 8, 7 | 8.2 / 10.0, 2.7; 7.9 / 9.6, 2.4 | 156 | 0.76 / 0.78 / 0.95 | 0 / 1 / 0 | 0 of 2 (0 of 1) | 1 / 156 | 12.8 |
| TUM xyz | TUM room | TUM | 2/2 | 8, 7 | 12.0 / 12.9, 2.0; 9.6 / 11.2, 2.2 | 86 | 0.73 / 0.78 / 1.00 | 0 / 0 / 0 | 0 of 8 (0 of 5) | 0 / 86 | 10.6 |
| TUM 360 | TUM room | TUM | 1/7 | 7 | 8.9 / 9.8, 2.0 | 76 | 0.69 / 0.81 / 0.21 | 0 / 0 / 0 | 0 of 12 (0 of 9) | 0 / 76 | 10.5 |
| TUM plant | TUM room | TUM | 1/1 | 10 | 7.0 / 8.8, 1.5 | 236 | 0.51 / 0.58 / 0.90 | 6 / 1 / 0 | 6 of 43 (4 of 28) | 1 / 236 | 13.2 |
| TUM teddy | TUM room | TUM | 1/1 | 8 | 7.1 / 9.3, 1.5 | 239 | 0.61 / 0.69 / 0.91 | 4 / 0 / 0 | 4 of 56 (1 of 29) | 0 / 239 | 14.1 |
| TUM desk2 | TUM desk | TUM | 2/2 | 7, 6 | 8.8 / 11.9, 3.0; 6.9 / 10.1, 2.6 | 140 | 0.76 / 0.80 / 0.93 | 0 / 0 / 0 | 0 of 0 | 0 / 140 | 10.0 |
| TUM desk | TUM xyz | TUM | 1/1 | 6 | 10.4 / 12.7, 1.8 | 37 | 0.81 / 0.81 / 0.86 | 0 / 0 / 0 | 0 of 4 (0 of 4) | 0 / 37 | 5.7 |
| ARKit 47333932 w1 | 47333932 w0 | static | 0/1 refused | — | — | 0 | — | — | — | — | 6.0 |
| ARKit 47333931 w0 | 47333932 w0 | static | 1/1 | 13 | 4.9 / 7.1, 1.0 | 93 | 0.76 / 0.76 / 0.88 | 0 / 0 / 0 | 0 of 2 | 0 / 93 | 7.8 |
| ARKit 47333931 w1 | 47333932 w0 | static | 1/1 | 10 | 9.3 / 10.7, 0.7 | 67 | 0.71 / 0.73 / 1.00 | 0 / 1 / 0 | 0 of 5 | 1 / 67 | 6.5 |
| Sam's Club 352-382 s | Sam's Club 337-367 s | our clip | 1/2 | 9 | no GT | 349 | no GT | 3 / 0 / 0 (by eye) | no GT | 0 / 349 | 17.4 |

Pooled over the pairs with GT:
- **Registration vs GT**: 4.9-12.0 cm (median over the revisit's keyframes) and 0.7-3.0 deg on every accepted shot; the reported
  u covers the GT error on 7 of 14 registrations, 2u on all 14. Refused: ARKit 47333932 w1 (the site map's cameras 0.106 of
  their spread off after the fit, gate 0.1) and 6 of the 360 sequence's 7 shots (cuts made them 6-49 keyframes: two are under
  the 10-keyframe minimum, one gives too few frames to pair, three fail the gate). A refused shot's objects are 'not compared', never changes.
- **Object match** (GT points of both cards, 5 cm voxels): precision 573 / 823 = 0.70 (0.51-0.81 per pair; a looser GT box
  test reads 0.58-0.82), recall 248 / 327 = 0.76 (0.86-1.00 on the pairs whose shots registered; 360 0.21, ARKit w1 0).
- **Change claims**: 17 in all; 13 right, 3 wrong, 1 unclear (GT depth decides 12, the eye 5): precision 13/16 = 0.81.
  The right ones are real rearrangements between TUM recordings (a chair moved 2.5 m, a notepad gone in two recordings, a
  monitor, a chair, a jacket, a mouse and small boxes new) and, on our clip, the filmer's own cart (it moves with the camera, so it
  really is elsewhere in the other window). Wrong: a chair called missing whose teddy-laden seat sat in the corner of the
  revisit's view, a bag on an ARKit shelf, a small box in the plant visit.
- **Heights and angles**: 55 matched pairs had a top or base differing by more than 2u, none confirmed by the see-through test
  (a partial view lowers a top without any change; the first version, r5b-visit-001's in-run layers with looser matching and
  no such test, claimed 9 and 25 height changes on the desk and later-room revisits, nearly all on mismatched neighbours); no angle was measured on both visits of a strong match, so no angle claim (and no GT for angles here).
- **Carried over**: 1,097 display models placed in the revisits' frames, 193 names and 130 types the revisit lacked, 4 ghosts.
- **False changes on the static revisits**: 0 of 98 (room later), 0 of 93 and 1 of 67 (ARKit other session), 0 of 349 claims on
  static things in our Sam's Club clip (its 3 claims are the filmer's cart); over every pair 3 wrong claims on 1,660 judged objects (0.18 %).
- **Change recall is low**: 10 of the 137 card places GT depth sees emptied / filled (5 of 83 for objects >= 30 cm). Where
  GT's changes went: 'there, delineated otherwise' (our see-through test read the place occupied), 'static' (matched to a
  look-alike at the same place), 'not observed'. Calibrated on 4,020 TUM cards against the GT oracle, the see-through test at
  X6's margins calls 12 places free (10 right) of GT's 116; relaxed margins (0.1 / 1 px / 0.5) call 22 (18 right) but raised our
  clip's claims from 3 to 10, 5 of the added ones on static shelves (by eye), so X6's rule stays. With ~5-12 cm registration and ~5 % estimated depth, a
  place reads empty only when the background is ~0.3-0.6 m behind the object: things lying on desks or against walls stay
  'occupied'.

Evidence: `evidence/*.jpg` (every claim, site map left, revisit right). Viewer: `viewer/teddy/`, `viewer/plant/` (Visits tab:
pick visit A / B, counts, a change's evidence; `visits-a-and-b.jpg` shows the site map's missing notepad as a ghost).

## Time on 2 x A100-80GB (min_containers 0; analysis time = MP4 bytes in the container -> Volume commit)

| revisit analysed with its visit step | cards final written | visits written | visits stage (descriptors / DA3 + fits / matching + see-through) | GPU peak GiB |
|---|---|---|---|---|
| TUM desk (20.4 s video, first call after boot) | 43.8 s | 56.4 s | 10.9 s (2.5 / 2.8 / 4.3) | 67.1 / 64.9 |
| ARKit 47333931 w0 (25.6 s) | 40.5 s | 57.7 s | 14.7 s (2.4 / 3.4 / 7.9) | 63.6 / 60.3 |
| Sam's Club 352-382 s (30 s) | 69.7 s | 99.0 s | 17.3 s (1.7 / 2.5 / 12.0) | 66.6 / 57.6 |

A revisit whose analysis is already on the Volume (FastReport.visit): 5.7-17.4 s from the call to the visits layer's commit
(median 10.0 s; it decodes both videos again), GPU peak 43.9 / 33.6 GiB (the resident models). Cold start (not analysis):
108 s ready in the container. No stage over 72 GiB. The comparison's CPU part (lifting unmatched objects and the see-through
tests) runs on one thread and is the largest part on Sam's Club (~900 cards a visit).

## Spend (list-price upper bound from app lifetimes)

Analyses of 14 videos (r5b-visit-001, 1,400 s): 3.16 USD; final bench (r5b-visit-002, 615 s): 1.46 USD; one-GPU dev recording of
13 pairs (visit_dev, 382 s): 0.30 USD; CPU jobs (dataset preparation, GT oracle incl. a slow first version stopped midway):
<= 2.2 USD. **Total <= 7.1 USD** (cap 12).

## Data (Modal Volume `panoptes-r5b-visit`, evaluation only; nothing on the Mac but the MP4 windows)

- TUM RGB-D fr1 room (0-30 s site map, 30-45 s revisit), desk, desk2, xyz, 360, plant, teddy (first <= 30 s each): one office,
  one motion-capture frame (their GT floors agree within 4 cm). Real differences between the recordings exist: 137 card places
  the GT oracle (GT depth + GT poses see-through test) sees emptied or filled across the pairs.
- ARKitScenes venue 467305 (metadata lookup: video ids 47333927 / 31 / 32): raw 47333932 (the round-4 GT capture, windows
  0-128 and 128-257 frames) and raw 47333931 (downloaded, prepared with the same prepare_arkit_clip rules). Two sessions of
  one venue minutes apart: static revisits; GT alignment by ICP of their LiDAR (confident) depth from 36 turns about +z.
- Our clips: data/clips/*/clip.json has no two clips of one place at different times (samsclub-337/-a/-a2, lightning-3572/3585
  and me340-165 / me340-oneshot are the same footage), so the pair is Sam's Club 337-367 s against 352-382 s, cut from the
  clip's source video: 15 s of shared footage, 15 s each side the other never saw (a 'not observed' test as much as a
  static one).

## Integration notes

Files: new `fast_report/visits.py`, `scripts/r5b_visits_eval.py`, `modal_apps/r5b_visit_data.py`, `web/src/VisitsPanel.tsx`,
`web/tests/r5b-visits-browser.mjs`; changed `fast_report/core.py` (end of analyse: the visit step; events switch; bank switch),
`fast_report/timeline.py` (place(): pose_m, rel_margin, neigh, free_share parameters, defaults unchanged), `fast_report/layers.py`
(Writer.drain), `fast_report/vlm.py` (vocabulary prompt), `modal_apps/fast_report_app.py` (FastReport.visit, entrypoints `visits`
and `visit_dev`, accuracy plans' `visit_of_site`, the `visits` milestone), `web/src/LiveReport.tsx`, `web/src/live-report.ts`
(liveDocument's optional 4th argument, visitView), `web/src/report-scene.css`.

Options (core): `visit_of` (the site map's report id; off by default) and `visit_site` (a name for `sites/<site>/visits.json`);
`events` (default True: False = no event captions, whose prompt asks about PPE and safety); `bank_save` (default True: False =
the shared naming bank on the Volume is left as it was; review finding 7). `vlm.VOCAB_PROMPT` lost its environment-health-and-
safety list (first list = the main object types, key `main`); if the objects builder removes or rewrites the VLM vocabulary
(finding 1), take theirs. Expect conflicts in core.py (vocab_then_events, the tail of analyse), vlm.py, fast_report_app.py
(near accuracy()), LiveReport.tsx (tabs, document memo) and live-report.ts (the entity loop).

## Issues

- Change recall: 7 % of GT's changes (6 % of the >= 30 cm ones). Registration precision (5-12 cm) and estimated depth decide it;
  a place-appearance check (the other visit's crop at the projected place against the object's) is the next lever, untried.
- Match precision 0.51-0.81: neighbours of similar look swap under 5-12 cm registration; worst where the revisit's camera
  height and viewpoints differ most (plant, teddy).
- Registration refusals: short shots (the 360 sequence's cuts) and one ARKit window at 0.106 against the gate's 0.1. One site
  map per site (the first visit): a revisit shot is registered to one site-map shot; site-map objects in shots no revisit shot
  registered to are 'not compared'.
- The filmer's own cart (Sam's Club) moves with the camera and reads as real changes: a within-video "carried by the camera"
  rule would drop it.
- Reported u covers the GT registration error at 1u on half the registrations (always at 2u).
- The comparison's CPU part is single-threaded (4-12 s); a thread pool over objects would cut it.
- web/tests/live-report-check.ts needs runs/fb-c-fixture-001, no longer on disk (its synthetic part passes); tsc passes.
- These runs used events off and the neutral vocabulary prompt, so their object layers are not r4b's settings.
- Local disk fell to ~4 GB during the round (other builders): after that only the three revisit mirrors (117 MB) were pulled.
