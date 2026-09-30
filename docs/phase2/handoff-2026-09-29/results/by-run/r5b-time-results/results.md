# r5b time: cross-time within a video, and honest width bounds

Branch `r5b/time` (from `r4b/integrate` 40a153e; X6's `fx/x6-windows-time` 95f8247 code reused: `fast_report/windows.py`,
`fast_report/timeline.py`, `scripts/x6_plant.py`, `scripts/x6_evaluate.py`). Worktree
`/Users/adam/.codex/worktrees/panoptes-phase2-video-r5b-time`.

Head ce62d95 (commits 27f7314, ab5d0f5, cd611a5, ce62d95). Three GPU runs, each one ephemeral container on 2 x A100-80GB
(min_containers 0, retries 0):

| run | code | what | calls |
|---|---|---|---|
| `r5b-time-gt-001` | 27f7314 | ground truth: ARKitScenes 42445448 (first call and a warm call), 47333932 (w0, w1), TUM fr1 room (w0, w1), as in r4b-gt-* | 6 |
| `r5b-time-retail-001` | ab5d0f5 | ME340, Sam's Club, Walmart: timeline on and off in one container; r5b plants (Walmart, Sam's Club) | 9 |
| `r5b-time-retail-002` | cd611a5 | the same after the look-alike test and the move-link fix, plus X6's run-006 plants | 11 |

The GT widths below are the final code replayed on gt-001's dumped inputs (the width rule did not change after ab5d0f5);
retail-002's blobs stay on the Volume (the Mac had < 8 GB free), so its cards were evaluated there
(`modal_apps/r5b_eval_app.py`, CPU). ce62d95 adds only evaluation code and drops stale 'last seen' notes on linked moves.

Every call: `judge False`, `bank_write False` (the shared label bank is read, never grown: review finding 7), `events False`
(the event captions' prompt asks for PPE and safety notes), `vocab_prompt neutral` (the scene vocabulary without its
'ehs_relevant' list). Times are seconds from the MP4 bytes in the container to the Volume commit (`written_s`, review
finding 9); cold start (102-110 s) is apart. Labels by eye are **agent-labelled**.

## Summary

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| object cards with a timeline (warm, retail-002) | 276 / 276 | 1052 / 1052 | 975 / 975 |
| content windows per shot | 1, 8 | 7, 4 | 8, 8 |
| change claims, unplanted video (retail-001 + 002 on-calls) | 0 in 4 calls | 0 in 2 calls | 0 in 2 calls |
| unseen windows whose place the test judged (occupied / free), warm | 116 / 0 | 127 / 0 | 179 / 0 |
| planted events claimed (r5b plants / X6 plants) | not placeable (cluttered) | 1 of 1 / 1 of 3 | 2 of 3 / 1 of 2 |
| false claims on the planted clips (retail-002) | - | 0 / 0 | 3 / 0 |
| width with a number or a bound (was 91 / 89 / 94 % in r4b) | 55 % | 32 % | 48 % |
| cards stage wall s, timeline on minus off (retail-001; retail-002) | +0.5, +0.4; -0.1, -1.0 | +1.7, +0.8; +2.7, -1.8 | +0.3, +0.6; -1.2, +0.4 |
| cards v3 written_s, on minus off (retail-001; retail-002) | +3.9; -17.4 * | +16.3; +14.1 * | +3.5; +1.7 |

\* confounded: the object sets differ between the on and off calls (ME340 688 vs 1073 objects in retail-002, Sam's Club 2352
vs 1958 in both runs) because the neutral vocabulary prompt returned different word lists (see Issues). Walmart's calls have the
same objects (1568 vs 1572) and are the clean pair.

- **Width.** An unresolved width is 'at most' only when both of its ends were seen bounded by free space; otherwise it is
  'at least' the extent seen in single views (when that is resolved) or 'not observed'. On the same GT cards the new
  'at least' bounds hold 122 of 124 (a) / 119 of 124 (b); round 4's 'at most' held 150 of 158 (a) / 149 of 158 (b); both ends
  were seen free on 1 unresolved width in 158 ('at most' is now rare). Value coverage is unchanged (the rule touches only
  unresolved widths). Retail width coverage falls from 89-94 % to 32-55 %: goods on shelves have neighbours at their depth,
  so their ends are never bounded.
- **Timelines.** Every object card carries its shot's content windows (ORB co-visibility 0.40) with a state per window
  (first seen / appeared / static / moved / disappeared / not observed with the place test's reason), values per window and
  per interval (value +- u over [t0, t1]), change flags only where a difference exceeds both u's, and evidence keyframes for
  every claim. Deformables are re-measured per window and get no place claims; people are placed per 5 fps keyframe.
- **Evaluation.** 0 claims on real, unplanted footage (8 retail calls, 5,239 cards) and on the static ground-truth
  scenes (6 calls, 799 cards, 1,502 unseen windows judged 'occupied', 0 'free'). Planted: 5 of 9 events claimed; 10 claims on
  planted clips, 7 true, 3 false (all three on the planted Walmart boxes: depth drift re-lifts one static box 1-2 m further
  along the aisle, and its first card's place is then seen empty), and 1 of 2 move links names the wrong source card. Real
  moves (a pushed cart, a hand-carried item): 0 of 2 (the cart is seen for about a second and its place never again; the item
  moves in a person's hand).
- **Viewer.** The card shows the timeline bar (state colours), the state at the video's time, values per interval, the
  change list with evidence keyframes and a per-window table; the objects list shows every card's state at t and filters the
  cards with a change; in 3D a changed card is drawn per interval (its model where it is, coloured boxes where it appeared,
  moved or left) and people as a marker per keyframe.
- **Spend** (list-price upper bound from the containers' lives): GT 1.01 + retail-001 2.37 + retail-002 2.76 = **6.14 USD**,
  plus two CPU containers on the Volume (evaluation, trace) about 0.05 USD.

## 1. Width bounds against ground truth

Ground truth as in `scripts/accuracy_gt.py` (the card's own pick regions lifted with GT depth and poses, the width along the
card's own footprint axes; the GT side is the one nearest the card's pooled width). (a) as delivered (assumed 1.6 m camera
height); (b) at the true camera height (x s, u without its scale part). Warm calls, cards with GT on >= 50 % of their mask.
Both rules on the **same cards** (round 4's 'at most' = measured + u is recovered from each field's `measured`), final code
replayed on the GT run's dumped inputs (`width-gt.json`; the run's own cards at 27f7314 give the same 'at most' counts and
the same values within one card, `width-gt-run-cards-27f7314.json`).

| sequence | cards | rule | values: n, within +-u (a) / (b) | 'at most': n, hold (a) / (b) | 'at least': n, hold (a) / (b) | 'not observed' / 'not measurable' |
|---|---|---|---|---|---|---|
| arkit42 | 57 | round 4 | 38, 35 / 33 | 11, 11 / 11 | 5, 5 / 5 | 3 |
| arkit42 | 57 | r5b | 38, 35 / 33 | 0 | 9, 8 / 8 | 10 |
| arkit47 | 199 | round 4 | 125, 116 / 112 | 42, 41 / 41 | 20, 20 / 17 | 12 |
| arkit47 | 199 | r5b | 125, 116 / 112 | 0 | 49, 49 / 46 | 25 |
| tum | 354 | round 4 | 212, 200 / 193 | 105, 98 / 97 | 12, 11 / 11 | 25 |
| tum | 354 | r5b | 212, 200 / 193 | 1, 1 / 1 | 66, 65 / 65 | 75 |

True camera heights: arkit42 1.23 m, arkit47 1.48-1.50 m, TUM 1.42-1.51 m (s = 0.77-0.94).

How the ends are tested (`cards.ends`): the object's points in the last 20 % of its extent at each end (above the lowest
30 % of its height: the surface it stands on continues beside its base), moved 6 px past its extreme point at their range
and spread 3 px deep, go through `timeline.place` on the object's own views with no between-window margin (one depth map):
'free' = the camera saw past them in >= 2 views. 'at most' is then max(measured + u, the distance between the two probes).
The lower bound is the median over single views of the extent seen: pooled views smear pose and depth error into width
(on the 87 unresolved GT widths it applies to, a pooled lower bound would hold 75 (a) / 71 (b), the worst at 2.9x the true
width; the single-view median holds 86 / 86).

Where the ends go (158 unresolved GT widths): an end 'occupied' 110, one end free 24, unjudged / occluded / out of view 23,
both free 1. Visual check (`ends_debug` sheets, TUM): flat things on desks see the desk beyond their
ends at their own depth, and products on shelves see their neighbours: the rule declines to bound them, which is what it
says. In r4b's own runs round 4's 'at most' held 72-92 % (TUM 81 of 98 (a)); on this round's cards it held 93-100 %: the
failure rate of 'at most' depends on the run's cards, the new rule does not claim it without the evidence.

## 2. A timeline on every object card

Content windows per shot (`core.shot_windows`: `fast_report.windows` on the keyframes' grey rasters, threshold 0.40, in the
process pool beside DA3: 0.8-3.1 s of one process per shot, off the critical path). Per card (`timeline.card_timeline`, in the cards'
process pool) and per window, from the window's own keyframes (carried ones belong to the window before):

- seen (pick-map detection or lifted points): values from that window's points alone (`cards.window_measure`: top, base,
  height, width, depth, position, each +-u from the model terms with the one-view-set k, plus a coverage term = what the
  window did not see of the largest per-window extent); 'first seen', or 'appeared' when the keyframes before its first
  sighting saw its place empty (X6's see-through rule, >= 2 views); later 'static', or 'moved' when its position differs
  from the previous window's by more than both u's (the shared-scale term left out: one shot, one scale) **and** the old
  place is seen empty;
- not seen: the place test on its last seen points over every keyframe since its last sighting: 'disappeared' when free
  (withdrawn if it is seen there again), else 'not observed' with the reason (occupied: there, not detected; occluded; out of
  view; not judged);
- a place seen free onto an object with the same detected word (the pick maps under the free pixels, >= 30 %) is not a
  change: a static object lifted twice by a drifting depth estimate;
- intervals = runs of windows with one state and no flagged change, values pooled over the run; a 'disappeared' and an
  'appeared' card of the same kind within one window, apart by more than both positions' u, are one moved object (both
  cards say so; the better-seen card's model is drawn at each place);
- claims are dropped for objects that sit in the frame's top or bottom 15 % band in >= 80 % of their views (ME340's
  burned-in caption was a 'sign' that 'appeared' and 'moved' before this guard), and for deformable and agent classes
  (re-measured per window / present and not observed).

Stored on the card as `time.timeline` = {windows, intervals, changes, rule, centre_xy} (compact: values [value, u]); the
name-free states stay in `raw.time.timeline_states` so `apply_name` can relabel for a new class. Cards bytes grow 26-31 %
(Walmart v3 9.1 -> 11.9 MB). People: `stateAt` and the viewer read the people cards' per-keyframe positions (PeopleLoop).

## 3. Evaluation

### 3a. Planted events (retail-002; claims matched in the image: the box's outline with the delivered DROID camera against the claimed card's pick region on the claim's evidence keyframe, IoU >= 0.25, kind and time agree)

| plants | video | event (s) | claimed | claim, IoU | the box got a card |
|---|---|---|---|---|---|
| r5b | Sam's Club | disappeared 2.72 | yes | obj-0-24 disappeared 2.64 -> 3.84 s, 0.98 | yes |
| r5b | Walmart | disappeared 19.56 | yes | obj-1-1360 disappeared 19.28 -> 19.60 s, 0.98 | yes |
| r5b | Walmart | moved 0.5 m at 23.92 | both halves | old place obj-1-1474 disappeared 23.52 -> 23.92 s; new place obj-1-1493 appeared 21.56 -> 24.68 s (0.99), linked to the wrong source (obj-1-1496) | yes |
| r5b | Walmart | appeared 15.80 | no | (0.5 s after the cut: its place was seen empty on 2-3 keyframes; retail-002's test saw through it onto another planted box and held it back) | yes |
| X6 | Sam's Club | disappeared 2.72 | no | | yes |
| X6 | Sam's Club | appeared 9.56 | yes | obj-0-994 appeared 7.80 -> 9.80 s, 0.52 | yes |
| X6 | Sam's Club | moved 1.2 m at 14.24 | new place only | obj-0-1054 appeared 12.36 -> 14.28 s (its old place was not claimed) | yes |
| X6 | Walmart | disappeared 29.16 | yes | obj-1-1476 disappeared 28.80 -> 29.40 s, 0.99 | yes |
| X6 | Walmart | appeared 29.20 | no | (the clip's last 0.8 s) | yes |

r5b plants (`scripts/r5b_plant.py`, X6's boxes and renderer): the box 1.2-7 m away on >= 4 keyframes on its side of the change,
its empty place 1.2-5 m away (the place test's range) on >= 3 on the other side, the change >= 3.5 s before the reference
shot's end, and a corrected free test (X6's asked the floor under the bottom corners to lie beyond them, which a textured
floor never does: ME340 could not be planted by either). Placed: Walmart 3 (moved 0.5 m, appeared, disappeared), Sam's Club
1, ME340 0 (DROID saw something in front of or inside every candidate place). Contact sheet `plants/plants-contact-sheet.jpg`.

### 3b. Real moves (agent-labelled, `labels.json`)

| video | what | frames | claimed |
|---|---|---|---|
| Walmart | a shopper pushes a cart across the view, about 2 m, 26.0-27.2 s | 650, 663, 680 | no: seen about 1.2 s (6 keyframes), its old place never seen empty again |
| Sam's Club | a shopper at the shelf handles a small orange item, 0.0-1.5 s | 0, 12, 25, 38 | no: the item moves in a person's hand (person pixels are never judged) |

ME340: no object moved (the presenter walks and gestures). The walk-throughs rarely see a change twice (X6's finding).

### 3c. False claims on static objects

| footage | calls | cards | claims | false |
|---|---|---|---|---|
| GT static scenes (TUM fr1 room, ARKit x2; revisits in arkit47) | 6 | 799 | 0 | 0 |
| retail, unplanted (retail-001 and -002 on-calls) | 8 | 5,239 | 0 | 0 |
| planted clips, retail-002 | 4 | 3,999 | 10 | 3 (Walmart r5b plants) |
| planted clips, retail-001 (before the look-alike test) | 2 | 2,084 | 8 | 4 (Walmart) |

The false ones (sheets in `claims/`, trace in `trace-walmart-planted-free-places.json`): one static planted box gets 2-5
cards along the aisle as the camera approaches (per-window positions of one box 5.2 -> 6.9 m, the moved box 7.5 -> 11.6 m;
real boxes in the unplanted Walmart drift 1-2 m too), and a card lifted from far is later seen empty because its place is
1-2 m short of the box. The look-alike test removed one of them (the camera saw through onto the same box's next card) but
three see mostly floor through their place (10-24 % box pixels). Their see-through margins (median 1.1-1.7 x the place's range)
are the same as the true claims' (1.4-2.3 x): no depth margin separates them. This is the object layer's fragmentation under
DA3's drift; the timeline's rule is right given its inputs.

### 3d. Recall / precision per kind (retail-002, all planted events and every claim)

| kind | events | claimed | claims | true | false |
|---|---|---|---|---|---|
| disappeared | 4 | 3 | 6 | 4 | 2 |
| appeared | 3 | 1 | 2 | 2 | 0 |
| moved | 2 | 1 as a move (link wrong), 1 half | 2 | 1 (the new place; wrong source) | 1 |
| all | 9 | 5 | 10 | 7 | 3 |

### 3e. Added seconds

Per cards build, the timeline adds 1.8-6.1 CPU s (v1) and 3.0-9.8 CPU s (v3), the width ends 1.1-1.9 / 1.8-3.6 CPU s, summed
over the pool's 24 processes (16 chunks): about 0.3-0.9 s of wall time; windows 0.8-3.1 s of one process per shot beside DA3; labels
0.04 s; move links 0.1-0.2 s. Measured in one container, same video (s):

| | run | cards v1 written_s on / off | cards v3 on / off | cards.v1 stage on / off | cards.v3 stage on / off |
|---|---|---|---|---|---|
| ME340 | 001 | 34.0 / 33.4 | 79.3 / 75.3 | 4.1 / 3.5 | 6.2 / 5.8 |
| ME340 | 002 | 27.7 / 32.4 * | 58.1 / 75.5 * | 2.5 / 2.6 | 4.4 / 5.4 |
| Sam's Club | 001 | 38.1 / 32.5 * | 103.4 / 87.1 * | 9.5 / 7.8 | 14.4 / 13.6 |
| Sam's Club | 002 | 35.7 / 31.2 * | 100.6 / 86.5 * | 9.1 / 6.4 | 13.9 / 15.6 |
| Walmart | 001 | 29.9 / 29.7 | 82.9 / 79.4 | 7.2 / 6.9 | 9.5 / 8.9 |
| Walmart | 002 | 28.4 / 30.2 | 80.9 / 79.3 | 6.1 / 7.3 | 9.6 / 9.1 |

\* different object sets (see the summary). Target <= +5 s: met on the cards stages (-1.8 to +2.7 s) and on Walmart's clean
pair (+1.7 / +3.5 s at cards v3); ME340 and Sam's Club's milestone differences are dominated by the vocabulary.

## 4. Viewer

`web/src/LiveReport.tsx`, `web/src/live-report.ts` (`stateAt`, `timeReps`, `floorToShot`); screenshots in `viewer/`
(`web/tests/r5b-time-shots.mjs`, on retail-001's planted Walmart report: the viewer code is the final one, those cards are the
earlier commit's). At 18.5 s obj-1-1360 reads 'first seen' in the list and on its card, at 22.0 s 'moved away (moved to
obj-1-1494)'; the list and the card agree at both times (asserted). `web/tests/live-report-check.ts` checks the timed
representations (model per interval, posed where the object moved, none before its first sighting, a red box after it
left) and `stateAt`; its recorded-fixture half needs `fb-c-fixture-001`, which is not on this disk.

## Times, GPU, spend

| | cameras | objects | cards v1 | objects v3 | cards v3 (written_s, warm, timeline on, retail-002) |
|---|---|---|---|---|---|
| ME340 | 17.4 | 27.7 | 27.7 | 52.6 | 58.1 |
| Sam's Club | 14.1 | 22.9 | 35.7 | 83.1 | 100.6 |
| Walmart | 13.0 | 20.8 | 28.4 | 68.5 | 80.9 |

Per-GPU peaks (GiB, GPU 0 / GPU 1): GT 57.6-63.1 / 52.7-60.7; ME340 62.4-66.5 / 53.4-57.2; Walmart 68.8-74.7 / 61.5-66.6;
Sam's Club 65.4-74.8 / 56.2-63.7. **Over 72 GiB on GPU 0** in 5 of 20 retail calls: Sam's Club on (72.8, 74.8), off (73.2),
planted (73.8), Walmart X6-planted (74.7), during densify's SAM 3, lift and naming (21-80 s); the timeline is CPU work and the
timeline-off call is among them. r4b's peak was 70.2: more objects from the neutral vocabulary (Sam's Club 35k densify masks).

Spend: `spend.json` (containers' lives x list price: 468 s + 1,096 s + 1,276 s).

## Issues

1. Planted Walmart: 3 false 'disappeared' (see 3c). A fix belongs where one physical object becomes several cards (the
   objects layer), or a continuation rule (a look-alike first seen within the shot's drift of a vanished place: move or
   drift, no claim), which would also withhold true small moves; not done (it would be tuned on this clip).
2. The move link names the nearest look-alike: with drift of 1-2 m a 0.5 m move is below what the geometry can separate;
   both halves are claimed right, the pairing is not.
3. The neutral vocabulary prompt (used here to keep safety framing out) gives unstable lists: 16-50 kept words, repeated
   'warehouse X' entries, tools listed in a paper-towel aisle; object counts vary 1.6x between calls of one video. The
   objects builder's vocabulary change should replace it.
4. Width coverage falls to 32-55 % on retail; 'at most' is almost never granted (1 of 158 unresolved GT widths had both ends
   free).
5. Real moves: 0 of 2 claimed; neither is seen twice.
6. GPU 0 above 72 GiB in 5 retail calls (densify, not the timeline).
7. Cross-visit timelines (the same site on another day) are not in this branch.
