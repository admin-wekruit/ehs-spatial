# Motion/trajectory scan — what the video layer actually needs

Date 2026-08-25. Survey + verification pass for the motion extension:
state understanding (fence moved, arm active, person in zone) and
trajectory understanding (zone entry/exit, min-distance-over-time, sweep
envelopes) over fixed-camera workcell video. Every license and access
claim below was checked against the license page, repo, or API schema —
sources inline. No API spend, no downloads in this scan.

## The finding before the table

**The state layer needs no new model.** "Fence moved from registered
position", "person in zone", "arm active" are all per-frame masks (the
SAM slot we already run) + the existing 3D lift + a diff against the
registered SceneMap. The only genuinely new capability motion requires
is **identity over time**, and on a fixed camera that is a CPU
association algorithm (Kalman + IoU), not a model. The per-frame 3D lift
is also mostly free on a fixed camera: for on-floor actors, ray-cast the
mask footprint onto the registered floor plane — deterministic, zero
model calls. Per-frame MoGe-2 is needed only for off-floor geometry
(robot-arm sweep envelopes). Verdicts stay in deterministic geometry;
the tracker is just another swappable commodity slot. The thesis holds
for motion.

## Verdict summary

| Candidate | Verdict | Why |
|---|---|---|
| ByteTrack / OC-SORT via `roboflow/trackers` (Apache-2.0) | **ADOPT now** | CPU identity-stitch over our existing per-frame fal masks; clean-room Apache re-implementations |
| SAM 3.1 video mode, self-hosted | **ADOPT (next GPU round)** | The tracking slot *is* the segmentation slot: per-frame COCO-RLE + native track IDs; same gated-weights procurement flag as the image model |
| PhysicalAI-SmartSpaces: rest of MTMC_2025 + MTMC_2026 | **ADOPT** | CC-BY-4.0, ungated HF, we already hold Warehouse_016 and have `scripts/mtmc_eval.py`; 2026 adds two real-world scenes |
| MEVA + MEVID | **ADOPT** | The only real fixed-camera surveillance video with CC-BY-4.0 video *and* annotations, open S3, KRTD camera models, 158 global IDs / 8,092 tracklets |
| fal SAM 3/3.1 video endpoints | **EVALUATE (blocked)** | $0.01 per 16 frames, but output is a rendered video — no machine-readable masks (verified against the OpenAPI schema) |
| SAM 2.1 (Apache-2.0) | EVALUATE | Clean-license fallback propagator if SAM License blocks procurement; visual prompts only, no text head |
| TAPIR / TAPNext (Apache-2.0 incl. checkpoints) | EVALUATE | Point tracking for fine arm motion — only if mask-level motion proves too coarse |
| CoTracker3 | SKIP | CC-BY-NC — not a commodity slot, whatever its benchmarks say |
| FoundationPose | SKIP | NVIDIA license: "only may be used … non-commercially"; also wants depth + a CAD mesh we don't have |
| DeepStream nvtracker (NvDCF/NvSORT/NvDeepSORT) | SKIP now | Proprietary EULA, closed-source C/GStreamer stack, NVIDIA-GPU-only; revisit only for an edge-appliance product |
| Metropolis MTMC/RTLS microservices | SKIP | K8s platform product; verdicts would come from vendor analytics — violates the harness thesis |
| Cosmos world models | SKIP | Generation/reasoning, not tracking; 2026-08-24 scan verdict stands |
| Isaac Sim / Isaac Perceptor | SKIP | AMR nav stack (cuVSLAM/nvblox); our synthetic data already comes out of Omniverse |
| MOT17/MOT20, DanceTrack, KITTI tracking, BDD100K, AIC CityFlow | SKIP | NC licenses, registration gates, or wrong domain — details below |

## 1. NVIDIA: "trajectory-related models" resolves to data, not models

The owner's framing — NVIDIA has trajectory models we should chase —
**does not survive contact**. What NVIDIA actually ships around
trajectories:

- **PhysicalAI-SmartSpaces** (HF, CC-BY-4.0, ungated —
  https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) is the
  adoptable artifact, and we already hold a slice of it
  (`outputs/datasets/mtmc`, Warehouse_016). The card lists
  MTMC_Tracking_2024 (90 scenes/212 h), MTMC_Tracking_2025 (23 scenes/
  42 h, world-coordinate 3D boxes + calibration + depth), and
  **MTMC_Tracking_2026 (28 scenes/28.7 h, including two real-world test
  scenes)** — the first real captures in the series. Caveat to check at
  download: challenge *test* scenes often ship GT-withheld for the eval
  server. NVIDIA's own scoring code (HOTA evaluator,
  `evaluate_aicity_mtmc.py`) is public in
  https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/tree/develop/libs/analytics/spatialai-data-utils/tools/evaluation.
- **AI City Challenge Track-1 "trackers"** are third-party challenge
  entries (e.g. https://github.com/Hamidreza-Hashempoor/Glance-MCMT,
  https://github.com/ZIOVISION/AIC2025_Track1_ZV; survey
  https://arxiv.org/abs/2508.13564), mixed licenses, research code.
  Worth *reading* — their method (per-camera 2D tracking + calibrated
  foot-point lift into world coordinates + cross-camera clustering) is
  exactly our architecture — but not dependencies.
- **DeepStream nvtracker** (IOU/NvSORT/NvDeepSORT/NvDCF in the unified
  NvMultiObjectTracker library —
  https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html)
  is real production tracking, and NVIDIA confirms a production path
  under the DeepStream EULA
  (https://forums.developer.nvidia.com/t/deepstream-and-nvdcf-commercial-licensing/282024)
  — but it is a proprietary, closed-source GStreamer/C stack that only
  runs on NVIDIA GPUs. Wrong shape for a Python harness whose slots must
  be swappable. Revisit only if the product becomes an on-prem edge
  appliance.
- **Metropolis multi-camera AI workflow / MTMC + RTLS microservices**
  (https://www.nvidia.com/en-us/ai-data-science/ai-workflows/multi-camera-tracking/,
  https://docs.nvidia.com/mms/text/MDX_Multi_Camera_Tracking_MS_Overview.html)
  is a Kubernetes platform product delivered via NGC. Adopting it means
  the tracking *verdict machinery* lives inside NVIDIA's analytics
  stack — the opposite of "models are commodities, verdicts are ours".
- **FoundationPose** (https://github.com/NVlabs/FoundationPose) is the
  one genuine NVIDIA trajectory *model* (6-DoF pose tracking of novel
  objects), and its LICENSE says: "The Work and any derivative works
  thereof only may be used or intended for use non-commercially"
  (https://github.com/NVlabs/FoundationPose/blob/main/LICENSE). It also
  wants an RGB-D stream and a CAD mesh or reference views of the object.
  Product-path dead on license alone.
- **Cosmos** — world foundation models for synthetic generation and
  embodied reasoning; nothing tracking-shaped. The 2026-08-24 scan's
  SKIP (Cosmos-Reason2 EVALUATE only for the no-photos-leave-building
  VLM case) stands unchanged.
- **Isaac Perceptor** (https://developer.nvidia.com/isaac/perceptor) is
  robot-mounted perception for AMRs (cuVSLAM stereo-inertial SLAM +
  nvblox reconstruction). We are the fixed camera watching the robot,
  not the robot. Skip.

## 2. Commodity trackers that fit the thesis

- **SAM 3 / 3.1 video** — the segmentation incumbent already *is* a
  video tracker: detector + tracker sharing one vision encoder, and 3.1
  adds the multiplex tracker (Meta: ~32 FPS at 128 objects vs ~16
  before — https://ai.meta.com/blog/segment-anything-model-3/).
  Self-hosted, the video API emits per-frame COCO-RLE masks with stable
  tracking IDs (worked example:
  https://samgeo.gishub.org/examples/sam3_video_masks/) — exactly the
  input our 3D lift wants. Same weights gating as the image model
  (HF gated, Meta "SAM License" — https://github.com/facebookresearch/sam3,
  https://huggingface.co/facebook/sam3); the procurement flag from the
  2026-08-24 scan covers this too, no new legal work.
- **fal's video endpoints exist but can't feed us** — see Corrections.
- **ByteTrack (MIT — https://github.com/ifzhang/ByteTrack/blob/main/LICENSE)
  and OC-SORT (MIT — https://github.com/noahcao/OC_SORT)** are
  detector-agnostic box-association algorithms: Kalman + IoU, no
  weights, CPU. **`roboflow/trackers`**
  (https://github.com/roboflow/trackers) re-implements SORT, ByteTrack,
  OC-SORT, and BoT-SORT clean-room under Apache-2.0, maintained, plugs
  into any boxes — including the bounding boxes of our fal RLE masks.
  This is the "tracking today with zero new model risk" move.
- **SAM 2.1** (Apache-2.0 code *and* checkpoints —
  https://github.com/facebookresearch/sam2/blob/main/LICENSE) does video
  mask propagation from visual prompts only (no text head). It is the
  clean-license fallback if SAM License sign-off stalls: LLMDet (Apache,
  already on the adopt list) supplies the boxes, SAM 2.1 propagates.
- **TAPIR / BootsTAPIR / TAPNext** — Apache-2.0, code and released
  checkpoints (https://github.com/google-deepmind/tapnet/blob/main/LICENSE).
  Point tracking, not object tracking: relevant only if we later need
  sub-object motion (arm joint speed, direction reversals) beyond what
  mask centroids/hulls give. Concrete gate: if mask-level sweep
  envelopes on the pilot video miss arm extents by more than the tier
  error band, run TAPNext on a 32-point grid over the arm mask.
- **CoTracker3** — technically attractive, license kills it: "The
  majority of CoTracker is licensed under CC-BY-NC"
  (https://github.com/facebookresearch/co-tracker). Internal research
  only, and TAPNext covers the same slot under Apache. SKIP.

## 3. Datasets — what adds to Warehouse_016

We hold: MTMC_Tracking_2025 Warehouse_016 (multi-camera video, 9,000
frames of world-coordinate GT, CC-BY-4.0) + `scripts/mtmc_eval.py`.
Correction already on record: that data is synthetic (Omniverse). What
adds:

| Dataset | License | Access | Adds | Verdict |
|---|---|---|---|---|
| MTMC_2025 remaining 22 scenes + MTMC_2026 | CC-BY-4.0 | HF, ungated, direct | More warehouses + hospital/retail/lab; 2026 has **two real-world scenes** (GT presence to verify) | **ADOPT** |
| MEVA (https://mevadata.org) | CC-BY-4.0 (video and annotations, stated on site) | AWS Open Data S3, no registration (`aws s3 sync s3://mevadata-public-01/... --no-sign-request` — https://registry.opendata.aws/mevadata/) | **Real** fixed multi-camera surveillance, 328 h, KRTD camera models in the GitLab data repo; >1.7M activity-linked boxes | **ADOPT** |
| MEVID (https://github.com/Kitware/mevid) | CC-BY-4.0 | Via MEVA site | Global person identities over MEVA: 158 people, 8,092 tracklets, 33 views (https://arxiv.org/abs/2211.04656) — real track GT for identity-persistence eval | **ADOPT** (with MEVA) |
| MOT17/MOT20 (https://motchallenge.net) | CC BY-NC-SA 3.0 | Direct zips | Crowded-pedestrian tracker stress; MOT20 is static-camera | Internal-only at best; not needed given MEVA |
| DanceTrack (https://github.com/DanceTrack/DanceTrack) | Annotations CC-BY-4.0 but dataset "non-commercial research purposes only" | HF | Uniform-appearance association stress | SKIP — NC data, wrong domain |
| KITTI tracking (https://www.cvlibs.net/datasets/kitti/) | CC BY-NC-SA 3.0 | Direct | Driving, ego-motion camera | SKIP |
| BDD100K (https://doc.bdd100k.com/download.html) | Custom Berkeley license; commercial rights only for BDD/BAIR members | Registration + license acceptance | Driving MOT at 5 Hz | SKIP |
| AI City CityFlow (https://www.aicitychallenge.org/ai-city-challenge-dataset-access/) | Non-commercial participation agreement, institutional-email registration | Gated | City traffic MTMC | SKIP |

Nothing public gives robot-arm or fence-motion GT. For "fence moved" and
"arm active" validation, the path is the one we already built for
distances: owner's own capture + the `scripts/site_acceptance.py`
tape-measure pattern, extended with a stopwatch (move the fence a
measured 0.5 m between clips; run the arm on a known program).

Download commands (not run in this scan; MEVA is multi-GB — pull only
selected clips):

```bash
uv run hf download nvidia/PhysicalAI-SmartSpaces --repo-type dataset \
  --include "MTMC_Tracking_2026/*" --local-dir outputs/datasets/mtmc2026
git clone https://gitlab.kitware.com/meva/meva-data-repo   # annotations + KRTD camera models
aws s3 ls s3://mevadata-public-01/ --no-sign-request        # then sync chosen drops
```

## 4. Cost model — 10-min fixed-camera video, 0.5–1 fps (300–600 frames)

Measured house facts carried in: fal SAM image calls ≈ $0.01/view/prompt
(docs/reviews/2026-07-20-independent-review.md); MapAnything/MoGe are
GPU-second billed on Replicate and effectively free at our scale ($0.12
at the 90-run mark, boundary-map doc); the 32-mask cap is fal's response
limit, not SAM's.

| Approach | Per 10-min video | Notes |
|---|---|---|
| fal `sam-3-1/image-rle` per frame, 2 moving-class prompts @ 1 fps | ~$12 (halve at 0.5 fps) | The dominant cost; static classes (fence) need ~1 frame/50 s → +$0.24 |
| fal `sam-3-1/video-rle`, one call | ~$0.38 ("$0.01 per 16 frames" — https://fal.ai/models/fal-ai/sam-3-1/video-rle; sam-3 video is $0.005/16) | 30× cheaper **but output unusable for geometry today** (see Corrections) |
| MoGe-2 per-frame lift (off-floor geometry only) | <$1 | ~$0.0013/frame measured on Replicate |
| Floor-plane ray-cast lift (on-floor actors) | $0 | Deterministic; needs the registered SceneMap we already build |
| ByteTrack/OC-SORT association | $0 | CPU, milliseconds/frame |
| Self-hosted SAM 3.1 video (24 GB GPU) | $0 marginal | 600 frames ≪ 1 min GPU at Meta's claimed multiplex throughput |

So: cloud-only motion costs **$3–12 per 10-minute video** and is
prompt-count-bound; the self-host round already planned for images
collapses that to zero and upgrades identity from stitched to native.

## 5. Recommended pilot stacks

**(a) Cloud-only, today** — no new models, one new Apache dependency:

1. Sample the video at 0.5–1 fps (moving classes) and ~1 frame/50 s
   (fence, registered-position check).
2. fal `sam-3-1/image-rle` per sampled frame with the existing adapter
   and prompt ensemble (person / robot arm / fence).
3. `roboflow/trackers` ByteTrack over the mask bounding boxes → track
   IDs. (OC-SORT as the free swap-invariance check — same boxes in,
   verdicts must not flip.)
4. 3D lift per frame: floor-plane ray-cast of mask footprints for
   people/movables; MoGe-2 anchor only for arm sweep geometry.
5. Deterministic rules over track series, all with tier error bands:
   zone entry/exit = point-in-polygon on the floor plane;
   min-distance-over-time = per-frame hull distance series (report min
   and dwell); fence-moved = current vs registered footprint offset
   beyond band; arm-active = hull motion energy over threshold; sweep
   envelope = union of lifted arm hulls over a sliding window.
6. Validate: MTMC Warehouse_016 world-coordinate GT through
   `scripts/mtmc_eval.py` (already proven for scale; extend to track
   continuity + zone events), then one MEVA/MEVID clip for real-video
   identity persistence.

**(b) Self-host GPU round** — fold into the already-approved SAM 3.1 +
MapAnything + MoGe-2 self-host: run SAM 3.1 in video mode for per-frame
COCO-RLE with native track IDs (ByteTrack demoted to cross-check),
everything else unchanged. TAPNext enters only if the mask-level sweep
envelope misses measured arm extents by more than the error band.

## Corrections forced by verification

1. **"NVIDIA has trajectory-related models" is mostly not actionable.**
   The adoptable NVIDIA trajectory asset is the *dataset* we already
   hold plus its siblings (CC-BY-4.0). The models are a proprietary
   GStreamer stack (DeepStream), a platform product (Metropolis), a
   non-commercial license (FoundationPose), or third-party challenge
   code. Nothing there beats a CPU association algorithm over our
   existing mask slot.
2. **fal's `video-rle` endpoints do not return RLE.** The OpenAPI schema
   (https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=fal-ai/sam-3-1/video-rle)
   shows outputs of exactly `video` (rendered segmented video) and
   optional `boundingbox_frames_zip` ("per-frame bounding box
   overlays"). No mask data, no coordinates JSON documented. Cheap
   ($0.38/video) but blind for geometry. Open probe, ~$0.40, not run
   under this scan's no-spend rule: check whether the bbox zip contains
   coordinate files rather than rendered overlay images; if coordinates,
   the cloud pilot gets 30× cheaper overnight.
3. **CoTracker3 is CC-BY-NC** despite its ubiquity in tracking write-ups
   — same trap shape as UniDepth v2 last scan. TAPNext (Apache,
   ungated, checkpoints included) is the licensed occupant of that slot.
4. **MOT/KITTI/DanceTrack, the default academic tracking benchmarks, are
   all NC or NC-adjacent.** The clean-licensed real-video GT that
   actually exists is MEVA/MEVID (CC-BY-4.0) — surveillance domain, not
   industrial, but real fixed cameras with camera models.
5. **No public dataset has moving-fence or robot-arm GT.** Those two
   states get validated the way distances did: owner capture + measured
   ground truth, not a benchmark download.
