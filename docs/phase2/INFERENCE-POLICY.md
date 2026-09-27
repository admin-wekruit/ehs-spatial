# What a video report may infer (2026-09-25)

Better footage is not an option: a report works from whatever video it gets, and no video sees every side of everything.
A wrong inference is worse than a blank, so the rule is:

**Infer only what a physical reason supports AND a test against the video could refute. Otherwise leave it blank and
label it unobserved.**

## Three kinds of geometry, always shown apart

| Kind | Shown | Used for measurements and EHS rules |
|---|---|---|
| Observed (what the video saw) | solid | yes, the only kind |
| Inferred from structure (floor, walls, support) | dimmed | no: a rule that depends on it is at best 需复核, never 通过 |
| Inferred by a model (complete object shapes) | unseen sides see-through | no |
| Unobserved | dark, labelled | no |

## Tiers

| Tier | What | Physical reason | Refutation test |
|---|---|---|---|
| A structure | floor under and between things seen standing on it; walls and ceiling extended only between observed edges | one verified plane; things stand on the floor | free-space check + hidden-block test |
| B objects | SAM 3D / RecGen complete shapes | the object was seen from >= 3 agreeing views | source view + held-out views (`complete_video_objects.py`) |
| B objects, box | a gravity-aligned box of the video's own points, where no generator shape passed | the object is box-shaped (cartons, packs, pallets, panels) | the same gate; the box is shaped only by the half of the views the gate does not judge on; then reviewed by eye |
| C support | a bench or machine body extended down to the floor | things do not float | free-space check + hidden-block test; translucent only |
| D never | behind walls, inside cabinets, aisles nobody walked, small items | none | none, so it stays blank |

## The tests

1. **Free-space refutation, per inferred element.** Project it into every camera that could see it. If >= 2 views saw
   clearly past it (reliable depth beyond it by 8% + 5 cm), the video proves nothing is there: delete it.
2. **The test itself is checked first.** It must stay quiet on geometry the video did see (flags <= 1% of the seen
   floor) and catch a deliberately wrong version (the floor plane raised 15 cm must be flagged in >= 50% of cells).
   Otherwise the scene gets no inference of that kind. (At 4% + 3 cm it flagged 8.9% of ME340's real floor: noise, not
   evidence.)
3. **Hide and predict, per inference kind.** Hide the seen geometry in 20% of 1 m blocks, infer again, score the hidden
   blocks: recall, and the share placed where the video proves there is none. Each knob (e.g. the gap-closing distance)
   takes the largest value whose false share is <= 5%; if none passes, that kind is off for the scene.
4. **The numbers ship with the report** (inferred-floor.json `dense.validation`): share seen, share inferred, measured
   false rate, cells deleted by refutation.

## ME340 floor (scripts/infer_room_floor.py --dense --droid-run)

- Region: floor seen, under things 1 cm to 2 m tall, gaps up to the chosen distance, and what they enclose; never open
  ground no camera saw (the first version used the map's convex outline and reached behind walls: withdrawn).
- Test noise on seen floor 0.37%; raised-plane control flagged 53.8%; hidden-block false floor 0.3% at 0.25 / 0.5 /
  1.0 m, recall 94.5 / 99.2 / 99.7%; chosen 1.0 m; 3 of 51,923 inferred cells deleted by refutation.
- 187.7 m² of floor, 57.9 m² seen (43%); 340k inferred points (2 cm), each the median colour of the 24 nearest seen
  floor points within 0.5 m, dimmed to 85%.
- Limit: where cameras only graze the floor the test is weaker (the control catches a 15 cm error in about half the
  seen floor), so under benches and in far corners the floor rests on the physical reason alone.

## Box shapes (2026-09-27, `complete_video_objects.py --generator box`)

Learned generators fail on shelf goods mostly for pose and size (ICP turned > 15 deg, or hit its scale clamp), since
one image does not fix either. The box needs no learned model: it is fitted to what the video saw and judged by the same
held-out test, so it survives only where the object really is box-shaped. The gate can still pass a thin flat thing
whose box swallows its surroundings (a coiled hose on a bench, a plug among cords), so every accepted box is also
checked by eye on its review sheet; ME340 dropped 2 of 6 that way, Sam's Club 2 of 23.

## Next, in this order

1. Walls and ceiling planes (tier A), same tests.
2. Support volumes (tier C), same tests, translucent.
3. Sam's Club and Walmart, after their per-shot rebuilds (their clips have cuts).
