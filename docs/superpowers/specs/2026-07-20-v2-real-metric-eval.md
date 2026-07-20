# V2 — Real-imagery metric accuracy eval (design)

## Claim under test

The live geometry path (MapAnything multi-view + floor-fit camera-height scaling from
`ehs_spatial/geometry.py`) recovers **metric ground-plane distances on real photos**
accurately enough for clearance rules — quantified, not assumed. V0 proved the
deterministic path on oracle data; V1 proves the provider chain runs; V2 is the first
accuracy claim on real imagery.

Scope: geometry only. No SAM, no rules, no Gemini — a measurement error at this layer
invalidates everything downstream, so it gets isolated first.

## Data

Redwood Indoor LiDAR-RGBD, scene `boardroom` (**public domain** — the only
license-clean choice for commercial-citable numbers; verified
redwood-data.org/indoor_lidar_rgbd/license.html). Artifacts used:
- `boardroom_rgbd.zip` — RGB frames (VGA-class PrimeSense; deliberately worst-case
  realistic input).
- `boardroom_lidar_resampled.zip` — laser-scanner point cloud = metric ground truth.
- `boardroom_poses.zip` — authors' estimated trajectory (camera-height derivation
  aid; treated as approximate, spread recorded).

Owner will later shoot a real workcell scene (4 angles, measured camera height, EXIF
kept, 3–5 tape-measured floor distances); same eval code consumes it unchanged.

## Protocol

1. **Frame selection:** 4 RGB frames with wide viewpoint spread over the same static
   furniture arrangement; similar camera heights (record per-frame height estimate
   and spread; reject sets with >10cm spread).
2. **Ground truth annotation** (one-time, versioned JSON):
   - Pick 3–6 point pairs between distinct floor-contact features (table legs, chair
     legs, cabinet corners) visible in the frames.
   - Measure each pair's ground-plane distance on the laser cloud (top-down
     orthographic render with known metre-per-pixel → pixel picks → world metres).
   - Annotate each point's pixel location in at least one selected frame.
   - Schema: `{camera_height_m, frames: [...], pairs: [{pair_id, description,
     frame_id, pixel_a: [u,v], pixel_b: [u,v], gt_distance_m}]}`.
3. **Run:** MapAnything (live, same adapter as production) on the 4 frames →
   `_build_geometry` floor fit + camera-height scale → for each pair, lift annotated
   pixels through the frame pointmap, project to floor plane, Euclidean distance.
4. **Report** per pair: predicted m, GT m, abs error (cm), % error; aggregate MAE /
   max; two scale modes side by side:
   - `camera_height` (production path),
   - `model_native` (MapAnything's own metric scale, no height correction) —
     research says images-only model scale is ~16%-class error; this row quantifies
     what the calibration stage buys us.

## Pass bar (provisional, owner-adjustable)

camera_height mode MAE ≤ 10 cm and max ≤ 15 cm over pairs in 0.5–3 m. Rationale: at
the 0.6 m demo rule, ±10 cm defines the INSUFFICIENT_EVIDENCE band the product must
declare; worse than that and verdicts near threshold are noise. model_native mode has
no pass bar — it is a measurement, not a gate.

## Non-goals

Segmentation/rule/Gemini evaluation (V3+), multi-scene sweeps, fine-tuning, pose GT
comparison, non-floor (height) distances — Tier-3 class, still deferred.

## Cost

1 MapAnything call per eval run. No SAM/Gemini calls.

## Results — 2026-07-20, Redwood boardroom (worst-case input tier)

Two frame sets run (~$0.2 total), all downstream analysis free via `--reuse`.
Runner: `scripts/redwood_v2_eval.py`; reports in `outputs/redwood_v2{,w}/v2_report.json`.

| Set | Input character | camera_height MAE | model_native MAE | per_frame_anchor MAE |
|---|---|---|---|---|
| close-up (017955/018406/018553/018935) | VGA, dim, single-object close-ups, equal heights ±3 cm | 27.5 cm | 24.0 cm | 27.0 cm |
| wide (017900/018000/018450/018900) | VGA, dim, wider views, heights via per-frame depth | 34.2 cm | 40.7 cm | 44.2 cm |

Findings:
1. The measurement instrument works end to end (annotation → reconstruction →
   floor fit → base-band gaps → laser-GT compare) and is reusable for any capture.
2. MapAnything on VGA dim indoor imagery: registration qualitatively coherent
   (unlike the synthetic pack), but per-view scale/pose drift ±30% (scaled camera
   heights 1.02/1.24/0.71/1.32 vs true ~1.13) → pair-gap MAE 24–44 cm at
   0.17–0.71 m GT. **Not compliance-grade on this input class.**
3. Camera-height global anchor ≈ model-native here: per-view inconsistency, not
   global scale, is the bottleneck. Per-frame anchoring failed because close-up
   frames do not contain enough floor for independent per-frame fits.
4. Production geometry gained the required "lowest adequately-supported plane"
   floor selection (tabletop dominance in real close-ups; regression-tested;
   offline oracle still 4/4).
5. Input protocol dominates outcome → owner capture protocol: 12 MP phone
   (40× the pixels of this VGA data), good lighting, 4 views at moderate
   distance with large mutual overlap, fixed measured camera height, EXIF kept.

Verdict: Redwood VGA defines the failure floor, not the product regime. The
decisive V2 datapoint is the owner's phone-grade capture through this same
runner — that is now the V2 critical path.

## Results — 2026-07-20 addendum: ETH3D `office` (good-input tier)

ETH3D high-res multi-view, scene `office` (26 MP DSLR downscaled to 2048 px for
the API; mm-grade laser GT; COLMAP-derived camera heights, spread 5.1 cm).
License CC BY-NC-SA — **internal eval only**, keep out of commercial-facing
claims. Pack `outputs/eth3d_v2/`; 1 MapAnything call.

| pair | GT | camera_height | model_native |
|---|---|---|---|
| desk-sofa | 0.737 | 0.451 (err 28.6 cm) | 0.442 (29.5) |
| sofa-chair | 1.349 | 1.479 (err 13.0 cm) | 1.498 (14.9) |
| sofa-bin | 1.667 | **1.652 (err 1.5 cm)** | 1.675 (0.7) |
| **MAE** | | **14.4 cm** | 15.0 cm |

scale_factor 0.9868 — MapAnything's own metric scale within ~1.3% of the
camera-height anchor on good input (vs 13–87% off on VGA). Input-quality tier
progression (same harness, same operator): VGA close-up 27.5 → VGA wide 34.2 →
2048-px DSLR **14.4** cm MAE, best pair 1.5 cm across 1.7 m. The
`per_frame_anchor` mode is deprecated: per-frame floor fits catch furniture in
both tiers (30.5–44.2 cm MAE).

Reading: accuracy is a monotone function of input quality with the architecture
held fixed; the desk-sofa outlier tracks cluster-definition ambiguity (tucked
chairs / dark under-desk region), not scale. Extrapolation to 12 MP bright
deliberate captures is favourable but must be measured, not claimed — owner
capture remains the decisive datapoint.
