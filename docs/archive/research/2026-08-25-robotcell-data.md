# Robot-arm workcell data for video POC v2 (2026-08-25)

Owner rejected MEVA/MTMC imagery: wants a factory robot-arm workcell — an industrial arm in
a cell, ideally people nearby, fences/zones visible or definable. This scan verified licenses
first, downloaded bounded samples (405 MB total, budget 2.5 GB), decode-verified everything,
and built a contact sheet: `outputs/datasets/robotcell_atlas.png`.

## Verdicts per candidate

### CHICO — Cobots and Humans in Industrial COllaboration (ECCV 2022) — SKIPPED (license unstated)
- **What it is:** KUKA LBR iiwa 14 R820 in a 500 m2 Industry-4.0 lab; 20 operators doing 7
  industrial actions next to the arm; multi-view RGB (~1M frames); 3D poses for **human and
  robot**; **226 genuine human-robot collisions**. The single best content fit for
  person-arm min-distance and collision scenarios — nothing else found comes close.
- **Access (verified in browser, anonymous, no registration):** open SharePoint folder from
  the [repo README](https://github.com/AlessioSam/CHICO-PoseForecasting) containing
  `camera_calib_parameters.json` (1.95 KB), `chico_3d_skeletons.zip` (51.8 MB),
  `chico_RGB_videos.zip` (**10.5 GB, monolithic**).
- **License:** none stated anywhere — checked both GitHub repos (no LICENSE file, GitHub API
  returns null), the arXiv paper (2208.07308; the paper itself is CC-BY but says nothing
  about dataset terms), and the SharePoint folder (no license/readme file). Per our
  license-first rule: **documented, not downloaded**.
- **Unblock path:** email the authors (Cunico / Sampieri, Univ. Verona) for terms. If cleared,
  the monolithic zip can be sliced without a full download via HTTP range requests against the
  zip central directory (SharePoint supports ranges), or the 52 MB skeletons zip grabbed whole.
- **Fit if unblocked:** arm-state ✓ (robot 3D pose), person-arm min-distance ✓✓ (both poses +
  calib), sweep envelope ✓, fence-zone ✓ (calib + defined floor zones). GT: provided 3D poses.

### DROID raw (Stanford et al.) — DOWNLOADED, recommended
- **License:** **CC-BY-4.0**, stated in the paper itself ("the full dataset under CC-BY 4.0
  license", arXiv 2403.12945) and echoed by the project site/mirrors.
- **What we took (208 MB):** one full raw episode from each of 5 labs
  (`AUTOLab`, `ILIAD`, `IPRL`, `RAIL`, `TRI`) from `gs://gresearch/robotics/droid_raw/1.0.1/`
  via plain HTTPS (no gsutil needed): 2 exterior + 1 wrist 720p60 MP4 per episode (+ stereo
  variants where present), `trajectory.h5`, episode metadata JSON. Deleted the bundled
  `trajectory_im128.h5` preview files (300 MB of 128px images we don't need).
- **Decode-verified:** all 21 MP4s open in OpenCV, full frame counts (116–360 frames @ 60fps,
  1280x720 / 2560x720 stereo), first and last frames readable.
- **Ground truth (validated, not just claimed):** `trajectory.h5` has per-frame 7-DoF
  `joint_positions`, `cartesian_position`, joint velocities/torques, gripper state, and
  **per-camera 6-DoF extrinsics per frame**. I implemented Franka Panda forward kinematics
  (standard modified-DH table, 40 lines, no URDF download needed) and compared against the
  recorded `cartesian_position`: **0.00 mm error on every frame of all 5 episodes** — the
  recorded pose is exactly FK of the joint states. Joint states + FK + extrinsics = exact
  arm geometry projectable into every camera view.
- **Fit:** arm-state detection ✓✓ (exact joint angles as labels), sweep envelope ✓✓ (FK over
  episode = true swept volume), fence-zone ✓✓ (define zones in robot base frame; extrinsics
  give exact camera-frame geometry for zone-entry evaluation), person-arm min-distance ✗
  (no people in frame — DROID scenes are unstaffed).
- **Caveat:** scenes are labs/kitchens/offices, not a factory floor. AUTOLab's episode is the
  most workcell-like (black-curtained cell around the arm). Raw MP4s are not face-blurred
  (unlike the RLDS export) — irrelevant here since no people appear, but note for future pulls.
- **Pulling more:** list episodes with
  `curl "https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix=robotics/droid_raw/1.0.1/<LAB>/success/&delimiter=/"`,
  fetch objects at `https://storage.googleapis.com/gresearch/<name>`. ~8–25 MB per episode
  without SVO/stereo.

### Berkeley FANUC Manipulation (Open X-Embodiment) — DOWNLOADED
- **License:** **CC-BY-4.0** via Open X-Embodiment distribution (google-deepmind/open_x_embodiment
  README: "All other materials are licensed under CC-BY 4.0"). The LeRobot mirror we pulled is
  tagged MIT. Either way permissive with attribution.
- **What we took (195 MB):** the complete LeRobot-v3 mirror `lerobot/berkeley_fanuc_manipulation`
  minus the wrist camera: exterior video (all **415 episodes** concatenated, 224x224 AV1 10fps,
  62,613 frames — decode-verified in OpenCV), data parquet with per-frame 8-dim
  `observation.state` (7 joints + gripper), episode index parquet with per-episode video
  timestamp ranges, task strings.
- **Ground truth:** joint states per frame → arm-state detection labels (moving/stopped, joint
  speeds). **No camera calibration** (OXE sheet: "Has Camera Calibration? No") → cannot project
  arm geometry into the image; sweep envelope only in robot frame, not verifiable in pixels.
- **Fit:** arm-state ✓ (labels), sweep envelope (robot-frame only) ~, fence-zone ✗ (no
  extrinsics), person-arm distance ✗ (no people). Big plus: a real FANUC Mate 200iD — the most
  *industrial-looking* arm of everything downloadable. Big minus: 224px thumbnails, weak for a
  demo the owner will *watch*.

### NVIDIA PhysicalAI-Robotics-Manipulation-SingleArm — DOWNLOADED (sample)
- **License:** **CC-BY-4.0** (stated on the HF dataset card; "available for commercial use").
- **What we took (2.6 MB):** 5 world-camera episodes (512x512 30fps MP4, decode-verified) +
  state parquets + meta from the `panda-open-drawer` subset.
- **What it is:** fully **synthetic** IsaacSim Franka doing cabinet/stacking tasks; per-frame
  sim state (proprioception + object poses), depth videos available.
- **Fit:** perfect GT (it's sim) for arm-state/sweep/zones, but the scene is a sterile gray sim
  room — not factory footage, and "synthetic demo" undercuts a video-POC pitch. NVIDIA branding
  is the one selling point. Checked the rest of the PhysicalAI family (42 datasets on HF):
  nothing is a robot-arm *cell* video dataset — warehouse sets (SmartSpaces/MTMC = already
  rejected), AV sets, GR00T humanoid teleop, sim manipulation. No NVIDIA factory-arm-cell asset exists today.

### InHARD — SKIPPED (unsliceable)
- **License:** CC-BY-4.0 (verified on [Zenodo 4003541](https://zenodo.org/records/4003541)).
  Real assembly station, human + UR-style arm, 3-view RGB + skeletons — good content fit.
- **Why skipped:** distribution is a **50 GB split 7z** (11 x 4.3 GB + 2.8 GB); split 7z needs
  all volumes to extract anything, so no bounded sample is possible. InHARD-DT
  (Zenodo 7644247, CC-BY-4.0) is the same story: 29 GB split zip. Documented for a future
  full-pull if the owner wants it (it's the only *permissive* people-near-arm video source found).

### HA4M — SKIPPED (no robot in scene)
Assembly-bench monitoring of a human building a gear train (Azure Kinect, 41 subjects), hosted
on ScienceDB. No robot arm in the scene — fails the core requirement regardless of license.

### Others checked, empty-handed
- **MIME** (CMU Baxter, 2018): tabletop Baxter via Google-Sites/Drive links, no license stated,
  not industrial — skip.
- **AgiBot World**: HF-gated (must accept terms) and CC-BY-NC-SA per project README → NC, skip.
- **RoboMIND**: Apache-2.0 tag but HF-gated (agreement acceptance required) → per rules
  documented, no registration performed.
- **fanuc_manipulation_v2** (`gs://gresearch/robotics/fanuc_manipulation_v2`): same Berkeley
  FANUC data, v2 RLDS export — redundant with the LeRobot mirror we took.
- **berkeley_autolab_ur5** (OXE, CC-BY): UR5 tabletop, fixed cam — viable fallback, less
  industrial-looking than the FANUC; not downloaded to stay bounded.

## What's on disk

```
outputs/datasets/robotcell/            405 MB total
  droid/                               208 MB  CC-BY-4.0
    <LAB>__<timestamp>/ x5             MP4s (720p60) + trajectory.h5 + metadata json
    frames/                            25 extracted JPGs
  fanuc_berkeley/                      195 MB  CC-BY-4.0 (OXE)
    videos/.../file-000.mp4            415 episodes, 224x224 AV1
    data/, meta/                       joint states + episode index parquets
    frames/                            10 extracted JPGs
  nvidia_singlearm/                    2.6 MB  CC-BY-4.0 (synthetic)
    panda-open-drawer/                 5 world-cam MP4s + state parquets + meta
    frames/                            6 extracted JPGs
outputs/datasets/robotcell_atlas.png   contact sheet (source | license | GT per row)
```

## Recommendation for POC v2

**Primary: DROID raw, AUTOLab episode (`AUTOLab__Tue_Nov__7_13:53:20_2023`), exterior cameras**
— a real Franka in a curtained cell, two calibrated exterior views, and the strongest GT of any
candidate:

1. **Arm-state detection:** VLM says moving/stopped/speed-class per window → score against
   joint velocities from `trajectory.h5`. Exact, per-frame, zero labeling cost.
2. **Sweep envelope:** FK over all frames = true swept volume in base frame; project through
   per-frame extrinsics into the exterior view → overlay truth vs. VLM/policy claims.
3. **Fence-zone rules:** define virtual keep-out zones in the robot base frame (policy-compiled,
   like our existing fence rules); zone-entry events are computed *exactly* from FK + extrinsics
   and compared with the vision verdicts. FK correctness is already proven (0.00 mm vs. recorded
   cartesian pose, all 5 episodes).
4. **Person-arm min-distance:** honestly **not testable** with permissive data today. Options:
   (a) judgment-only demo on this footage with a staged "person region" polygon, clearly labeled;
   (b) get CHICO cleared (one email) — it adds real people, real collisions, and both-skeleton
   GT, upgrading this axis from demo to measured. Recommend (b) in parallel with shipping (a).

Grab 3–10 more AUTOLab episodes (~15 MB each, listing pattern above) if the POC needs variety;
use the FANUC clip in the pitch deck as the "this generalizes to industrial arms" still, and keep
NVIDIA sim as an optional NVIDIA-branded backdrop only if synthetic is acceptable.
