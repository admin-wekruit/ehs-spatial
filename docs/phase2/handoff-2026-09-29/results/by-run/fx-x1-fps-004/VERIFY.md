# Adversarial verification of x1-fps: holds-with-corrections

These corrections take precedence over results.json conclusions.

Every number re-derives exactly. I re-ran `modal_apps/x1_fps_sweep.py --evaluate` (branch fx/x1-fps, commit 6d86969) on copies of the raw records and got results.json back with 0 differences; the claim's tables match it.

What checks out:
- **References:** the harness fixtures point to the delivered reports c40fbd08, 913daf2a and 32cec650.
- **Inputs:** every rate uses the same SAM 3 per-frame masks, the same fixed vocabulary and the same geometry world.
- **Sam's Club from run 003:** the later code changes do not touch it. The ME340 and Walmart reruns (003 vs 004) give bit-identical quality numbers, so the pipeline is deterministic.
- **Timing:** cold start is excluded the same way in every run.
- **Memory:** peaks are device-wide nvidia-smi readings per GPU. The maximum is 67.98 GiB, under the 72 GiB flag line.
- **Spend:** $4.199 matches the container records. I spent $0 and ran no GPU jobs.

The main objects conclusion does not hold as stated. The rise in recall@0.5m from 0.550 to 0.649 at 5 fps is an object-density effect:
- Cut the 5 fps objects back to the current count at random and mean recall is 0.547.
- Move every centroid 1 m in a random direction and recall still rises by 0.102 of the 0.099 gain.
- The share of our objects near a delivered object falls from 0.678 to 0.503.
- The thin-object gain (0.2 → 0.7, pool of 10) sits inside the same null.
- 'New vs current' marks objects the current rate already found as new, which inflates the contact sheets.

The 5 fps recommendation is still supported by the outline IoU gain and by the brief real objects seen in the sheets.

Geometry statements need corrections. G30c (consecutive chunks) is the best configuration on Walmart and has the best held-out absrel on all three sites. G10, not Gmotion, has the best ATE on Sam's Club. More views lower ATE on two of the three sites.

SAM 3 seconds are M x n estimates, 6-41% below the measured phase walls, and they leave out DA3 and classification-cascade costs. The people conclusions hold.

My computations are in `/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/x1verify/x1-verify.json`, with the scripts null.py, null2.py and null3.py in the same folder.

## Problems
- [major] The main objects claim, that 5 fps raises recall from 0.550 to 0.649, is mostly an effect of having more objects. The metric counts a delivered object as found if any of our centroids is within 0.5 m, and the object count goes up 2.5x. Cut the 5 fps objects back to the current rate's count at random and mean recall is 0.547, against 0.550 at the current rate. Move every centroid 1 m in a random direction and recall still rises 0.384 → 0.486, almost the whole +0.099. Meanwhile the share of our objects that lie near a delivered object falls 0.678 → 0.503 → 0.408 → 0.294. Word-matched recall barely moves: 0.158 → 0.195 mean, and on ME340 it falls 0.139 → 0.101. At 0.3 m there is a small real gain over the null (+0.06). On top of this, the lift's fixed CONFIRMED=2-frame rule is a much weaker filter at 30 fps (two frames 1/30 s apart) than at 1.7 fps, which also inflates counts at high rates.
  Fix: Report recall together with the count-matched value, a displaced-centroid null and ours_near_reference (a precision proxy) for every rate. Base the 5 fps recommendation on the outline IoU gain (0.37-0.70 → 0.65-0.81, which is real) and on contact-sheet evidence of brief real objects, not on recall@0.5m. Say that recall per object does not improve.
- [major] The thin-object gain on ME340 (0.2 → 0.7) comes from a pool of 10 objects, and the 1 m displaced-centroid null rises almost as much (0.21 → 0.64). The small-object value of 1.0 at the current rate has a null of 0.77. Neither result shows that the rate improves thin or small objects. A 30 fps loss on small objects (count-matched 0.29) points to merged components rather than detection.
  Fix: Present the small and thin splits with their pool sizes and null values, and label them 'not distinguishable from object density (pool 10 / 23)'.
- [major] The contact sheets and 'new vs current' counts mix in objects that the current rate already found. The test (under 10% of voxels inside any current object) fails when a high-rate object's voxel union is inflated. Four examples from ME340 at every frame:
- 'monitor' seen in 331 views has an 8.8 m union extent, and the current rate has a 'monitor' 1.6 m away.
- 'container' (184 views) has a current 'bottle' 0.11 m away.
- 'pipe' (89 views) has a current 'pipe' 0.37 m away.
- 'cable' (83 views) has a current 'cable' 0.25 m away.
The 5 fps sheets also repeat the same box, container and light fixture across tiles. The claim that the sheets show power tools is not supported: none appears in the ME340 sheets. 'power tool' shows up only as a word, 14 times among the every-frame new objects.
  Fix: Do not describe the sheets as 'objects that appear only at higher rates'. Before calling an object new, also require no current object of the same word within about 0.5 m. Drop 'power tools' from the list of things the sheets show.
- [minor] Two geometry statements go further than the data:
- 'Consecutive chunks break the path (16-30 cm)' holds on ME340 (16.4 vs 4.1 cm) and Sam's Club (30.3 vs 22.2), with ghosting visible in those renders. On Walmart, G30c is the best configuration: ATE 16.2 vs 18.1 cm, absrel 0.061 vs 0.079, within-10% 0.72 vs 0.57, and no visible ghosting.
- 'Depth agreement on held-out views does not improve' is contradicted by G30c, which has the best held-out absrel on all three sites (0.053/0.162/0.061 vs G5 0.056/0.171/0.079), including the 0.053 in the claim's own table.
  Fix: Say the consecutive chain fails on ME340 and Sam's Club and is best on Walmart. Say held-out absrel does not improve for G10, G30i or G30single, but does for G30c.
- [minor] 'Gmotion gave the best ATE on ME340 and Sam's Club' is wrong for Sam's Club: G10 is 19.6 cm against Gmotion's 20.7 cm. 'More DA3 views do not lower ATE' is also too strong: G10 lowers Sam's Club from 22.2 to 19.6 cm and Gmotion lowers ME340 from 4.11 to 3.06 cm, while Walmart gets worse. The effect depends on the site.
  Fix: Say that a 10 fps view budget lowers ATE on ME340 (Gmotion) and Sam's Club (G10), raises it on Walmart, and costs 2.7x the DA3 time (22 s vs 8 s on ME340).
- [minor] The SAM 3 seconds in the tables are M x n estimates that exclude dedupe and packing. They are 6-41% below the measured nested phase walls: ME340 current 11.2 vs 14.1 s measured, 5 fps 33.6 vs 38.5 s. The '+22 s / 7-9 s' increment also counts person detection and the vision backbone, which a-core already runs on every 5 fps keyframe; the vocabulary-only increment is 19.0/6.1/4.8 s. Three downstream costs are not costed:
- DA3 depth for frames above 5 fps (G10 22 s, G30i 104 s on ME340);
- the 2.5x larger object list fed to a-core's VLM classification cascade (203 VLM requests for 251 objects in fb-a-core-011);
- the in-between-frame geometry needed by person-adaptive.
No end-to-end analysis time was measured per rate.
  Fix: Report the measured phase walls next to M x n, give the a-core increment as vocabulary-only, and add a column for DA3 and cascade cost or mark the end-to-end time as not measured.
- [minor] The claim calls run-to-run noise 'unmeasured'. In fact ME340 and Walmart ran twice (runs 003 and 004), and every recall, ATE, coverage and people number came out bit-identical, so the pipeline is deterministic. The real uncertainty is pool size: 95/154/157 objects, 10 thin, 23 small, 3 small on Sam's Club, and 12 matched people points on Sam's Club. A 0.02 change is 2-3 objects.
  Fix: Replace the noise caveat with pool sizes and sampling uncertainty, and cite the identical 003/004 reruns.
- [minor] 'Every frame in one forward is feasible for these 30 s clips' was tested only on each clip's reference shot (ME340: 651 views, about 22 s), not on a full 30 s shot of about 900 views. The 68 GiB peak is already 85% of the card. Separately, the per-stage peaks results.json lists for G30i/G30c on GPU 1 are inherited from G30single's allocator cache, because the cache was not emptied after that test.
  Fix: Restate as 'feasible per shot up to 651 views (68 GiB, 85%)'. Mark the G30i/G30c GPU 1 peaks as contaminated, or empty the cache before the chunk stages in any rerun.
- [minor] The people conclusion 'going higher changes little' plays down Walmart, whose path difference improves 33% (15.2 → 10.2 cm, on 22 → 65 matched points). Attributing Sam's Club's 64 cm to its geometry is an inference, not a measurement, and it rests on 12 matched points.
  Fix: Label the Sam's Club attribution as inferred and state the matched-point counts.

## Corrected table

X1 frame-rate sweep, re-derived. I re-ran `x1_fps_sweep.py --evaluate` on copies of raw-*.json and got 0 differences from results.json. The references are the right ones (fixtures delivered-303 give c40fbd08 / 913daf2a / 32cec650). Hardware was 2 x A100-SXM4-80GB, no MPS. Cold start is excluded the same way in every run (the clock starts after the models load; boot 22-31 s is recorded separately). Metres are estimated.

OBJECTS. Main shot only; the table's object counts are all shots. "Matched" is recall@0.5m after randomly cutting the rate's objects down to the current rate's count (181 / 236 / 214; 50 draws). "Null" is recall@0.5m with every centroid moved 1 m in a random direction (20 draws). "Near" is the harness's ours_near_reference_0.5m: the share of our objects within 0.5 m of a delivered object, a precision proxy. SAM 3 is reported two ways: M x n as claimed, and the measured cumulative wall time of the nested phases.

| rate | recall@0.5m ME/SC/WM (mean) | matched (mean) | null (mean) | recall@0.3m mean (null) | near mean | word-matched mean | px-IoU ME/SC/WM | SAM 3 s M x n | SAM 3 s measured |
|---|---|---|---|---|---|---|---|---|---|
| current | 0.589/0.455/0.605 (0.550) | 0.550 | 0.384 | 0.365 (0.123) | 0.678 | 0.158 | 0.70/0.66/0.37 | 11.2/4.5/3.7 | 14.1/6.4/4.8 |
| 5 fps | 0.716/0.487/0.745 (0.649) | 0.547 | 0.486 | 0.455 (0.154) | 0.503 | 0.195 | 0.81/0.78/0.65 | 33.6/13.5/11.1 | 38.5/16.7/13.0 |
| 10 fps | 0.726/0.513/0.764 (0.668) | 0.498 | 0.479 | 0.426 (0.153) | 0.408 | 0.203 | 0.81/0.82/0.70 | 67.2/26.9/22.2 | 75.0/31.1/24.7 |
| 15 fps | 0.695/0.487/0.745 (0.642) | 0.466 | 0.481 | 0.419 (0.154) | 0.376 | 0.194 | 0.84/0.85/0.72 | 100.7/40.4/33.4 | 117.5/49.1/40.3 |
| every frame | 0.716/0.448/0.662 (0.609) | 0.361 | 0.503 | 0.390 (0.169) | 0.294 | 0.171 | reference | 196/79/65 | 211/87/69 |

- **Adaptive frame placement:** the numbers as claimed are verified (a1 0.604/0.589 mean, a2 0.663/0.655).
- **Cost of going from current to 5 fps objects in a-core:** a-core already runs person detection and the vision backbone on every 5 fps keyframe, so the extra work is the vocabulary pass only. That is about 19.0 / 6.1 / 4.8 s [M x n]. The claim says 22 / 9 / 7 s; the measured phases are 24.4 / 10.3 / 8.2 s. Above 5 fps, the in-between frames also need DA3 depth (G10 22 s, G30i 104 s on ME340), which the table leaves out.
- **ME340 by delivered type, current / 5 / 10 / 15 / 30 fps:**

| type (pool) | recall | 1 m null | count-matched to 181 objects |
|---|---|---|---|
| small (23) | 1.0 / 1.0 / 0.96 / 0.83 / 0.70 | 0.77 / 0.67 / 0.57 / 0.56 / 0.50 | 1.0 / 0.92 / 0.67 / 0.51 / 0.29 |
| thin (10) | 0.2 / 0.5 / 0.6 / 0.6 / 0.7 | 0.21 / 0.34 / 0.42 / 0.45 / 0.64 | 0.2 / 0.33 / 0.39 / 0.39 / 0.34 |

GEOMETRY. Absrel is held-out absrel on all three sites.

| config | ATE cm ME/SC/WM | coverage | absrel ME/SC/WM | DA3 s ME |
|---|---|---|---|---|
| G5 | 4.11/22.2/18.1 | 0.948/0.820/0.760 | 0.056/0.171/0.079 | 8.1 |
| G10 | 4.04/**19.6**/18.4 | 0.970/0.860/0.834 | 0.064/0.154/0.083 | 22.2 |
| Gmotion | **3.06**/20.7/25.2 | 0.971/0.826/0.832 | 0.057/0.162/0.084 | 21.5 |
| G30i | 4.34/23.5/19.4 | 0.980/0.879/0.874 | 0.059/0.176/0.089 | 104 |
| G30c | 16.4/30.3/**16.2** | 0.989/0.858/0.873 | **0.053/0.162/0.061** | 75 |
| G30single (reference shot only: 651/403/352 views) | 4.37/24.7/20.1 | 0.979/0.876/0.874 | 0.075/0.179/0.091 | 126 |

- G30single's device peak is 68.0 GiB on GPU 1. The G30i/G30c peaks listed for GPU 1 (68.0 / 44.3 / 39.6) are carried over from G30single's allocator cache; they are not those configs' own peaks.

PEOPLE: verified as claimed (7.2/64.3/15.2 → 8.3/64.8/10.9 → 8.3/64.9/10.2 cm; ID switches 0/0/1 at every rate; extra tracks 0/0/1 → 1/1/2; R3 agreement 0.82-0.99; 0 flips; person-adaptive 437/325/214 frames vs uniform 443/375/375). Caveats:
- Sam's Club's path difference rests on 12 matched points (115 extra).
- Walmart improves by 33% (15.2 → 10.2 cm).

MEMORY: the highest device peak in any run is 67.98 GiB, under the 72 GiB flag line. Peaks during SAM 3 were 38.2/32.0, 41.7/35.0 and 43.8/37.5 GiB.

SPEND: $4.199 = 0.339 + 0.459 + 1.96 + 1.441, from container seconds (container start-up before the function runs is not counted). My own spend was $0.
