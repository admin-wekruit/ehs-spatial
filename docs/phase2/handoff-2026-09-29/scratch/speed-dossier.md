### feedforward-recon
## Feed-forward 3D reconstruction for Panoptes: can one A100 give cameras, dense depth and a point cloud for a 30 s clip in under 30 s?

Yes, on paper, if the model is kept loaded on a warm GPU and fed about 150 keyframes (every 6th frame, 5 fps). The best-quality option is DA3-GIANT in any-view mode. It is the same model the depth stage already uses, but in this mode it also predicts the camera poses. Nothing here has been run on our clips yet, so every runtime for our case is an estimate.

**Labels:** **S** = measured by the source (hardware given). **C** = claimed, with hardware or protocol unclear, or from a secondary summary. **P** = measured by Panoptes. **E** = my estimate.

### 1. Runtime and memory for about 100–300 frames

| Model | Runtime | Peak VRAM | Estimate for 150 / 300 frames on one A100 |
|---|---|---|---|
| **VGGT-1B** | H100 with FlashAttention-3, 336×518: 100 frames 3.12 s, 200 frames 8.75 s **S** ([Table 9](https://arxiv.org/html/2503.11651)) | 21.2 GB (100 frames), 40.6 GB (200 frames) **S**. A May 2026 fix fits "2–3× more frames" in the same memory **C** ([repo](https://github.com/facebookresearch/vggt)) | 10–14 s / 35–50 s **E** (A100 taken as 1.75–2.5× slower than H100; our LingBot H100/A100 ratio is 1.75× **P**) |
| **VGGT-Ω-1B-512** | A100-80GB: 1000 frames 240.2 s, and 20–25% faster than VGGT **S/C** ([paper](https://arxiv.org/abs/2605.15195)) | A100 at 624×416, weights included: 13.4 GB (100 frames), 20.8 GB (200), 28.3 GB (300) **S** ([README](https://github.com/facebookresearch/vggt-omega)) | 6–10 s / 22–28 s **E** (assumes time grows with the square of frame count) |
| **DA3-GIANT-1.1** (any-view) | A100, 32 images at 504×336: 37.6 FPS. LARGE 78.4 FPS, BASE 126.5 FPS. GIANT handles at most 900–1000 images **S** ([paper, Table 8](https://arxiv.org/abs/2511.10647)) | The MapAnything plot shows DA3-Nested at about 23 GB (100 views) and 37 GB (200 views); plot GPU not stated **C** | 6–15 s / 25–40 s **E**. Our posed run on 155 views took **72 s wall on A100-80GB (P)**; see note below |
| **DA3-Streaming** (chunk 120, overlap 60, loop closure) | A100, KITTI, 11,373 frames: 8.51 FPS. On the same test VGGT-Long ran at 2.91 FPS and Pi-Long at 3.15 FPS **S** ([README](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md)) | 28.3 GB at 504×378 **S** | 18 s / 35 s **E** (from the measured rate) |
| **MapAnything** | About 4.5 s at 100 views and 15 s at 200 views, read from the plot; GPU not stated **C** ([profiling](https://github.com/facebookresearch/map-anything#profiling)) | 117 GB at 200 views in default mode, about 29 GB in memory-efficient mode **C** | 8–15 s / 40–60 s **E** |
| **π³ / Pi3X** | A800 on KITTI: 57.4 FPS **S** ([paper, Table 4](https://arxiv.org/abs/2507.13347)). Plot: about 6 s (100 views), 21 s (200 views) **C** | π³ about 17 GB at 200 views; Pi3X about 37 GB at 100 views **C** | 8–12 s / 35–45 s **E** |
| **Fast3R** | A100: 32 views in 0.509 s using 13.25 GiB; 251 FPS at 224 px; up to 1500 views **S** ([paper](https://arxiv.org/abs/2501.13928)) | Low | 2–5 s / 5–15 s **E** |
| **CUT3R** (streaming) | 17 FPS claimed **C**. 6.98 FPS on A800 KITTI, as measured by the π³ authors **S** | Constant state size | 9–21 s / 18–43 s **E** |
| **StreamVGGT** | Its KV cache grows with every frame; OOM on A100-80GB over long runs is reported **C** ([XStreamVGGT](https://sid.onlinelibrary.wiley.com/doi/10.1002/jsid.70086)) | Grows | not recommended |
| **MASt3R-SLAM** | 15 FPS on an RTX 4090 **S** ([paper](https://arxiv.org/abs/2412.12392)) | – | 10 s / 20 s **E** |
| **LingBot-Map** | 518×378: 20 FPS on H800, 12 FPS on RTX 4090 **S** ([paper, Table 9](https://arxiv.org/abs/2604.14141)). **Ours: 7.1 fps on A100-80GB, 12.3–12.5 fps on H100, 13 GB peak (P)** | 13.3 GB with the bounded window **S** | **21 s / 42 s E** (from our measured rate) |
| **DROID-SLAM / DPVO** | On an RTX 3090 (DPVO paper): DROID 40 FPS using 8.7 GB; DPVO 60 FPS using 4.9 GB, 120 FPS in its fast mode **S** ([DPVO](https://arxiv.org/abs/2208.04726)). DPVO gives sparse points only, no dense depth | – | DROID on all 900 frames: about 25 s for tracking plus global bundle adjustment **E** |

**About our 72 s DA3 run:** `modal_apps/mono_room.py` does all of this inside one timed call: start the app, upload PNGs from the Mac, load the 1.1B weights, run inference, download the npz. The forward pass itself is probably 10–15 s of that **E**. This overhead, not the model, is most of the gap to a 2–3 minute target.

### 2. Accuracy against SLAM and SLAM+TSDF

Higher is better for AUC and F1; lower is better for ATE.

| Benchmark | Results |
|---|---|
| **Pose, AUC@3 / AUC@30, pose-free** ([DA3 Table 2](https://arxiv.org/abs/2511.10647)) | HiRoom: **DA3-G 80.3/95.9**, π³ 67.0/94.8, VGGT 49.1/88.0, DA3-L 58.7/94.2, MapAnything 17.9/82.8, Fast3R 25.9/77.0. ScanNet++: **DA3-G 85.0/98.1**, VGGT 62.6/95.1, π³ 50.7/92.1, MapAnything 20.2/84.1. 7Scenes: all methods between 12.6 and 29.2 at AUC@3 **S** |
| **Reconstruction F1** ([DA3 Table 3](https://arxiv.org/abs/2511.10647)) | HiRoom: DA3-G 85.1, π³ 75.8, DA3-L 69.5, VGGT 56.7. ScanNet++: DA3-G 77.0, DA3-L 67.9, VGGT 66.4, π³ 63.1 **S** |
| **VGGT-Ω-1B vs DA3-G, AUC@3** (VGGT-Ω's own protocol) | 7 Scenes 29.6 vs 18.7; Sintel (moving content) 35.3 vs 16.2 **S** |
| **TUM RGB-D ATE (m), SLAM baselines** | With calibration: MASt3R-SLAM 0.030, **DROID-SLAM 0.038**. Without calibration: DROID-SLAM 0.158–0.163, MASt3R-SLAM 0.060 **S** ([MASt3R-SLAM](https://arxiv.org/abs/2412.12392), [DA3-Streaming](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md)) |
| **TUM RGB-D ATE (m), feed-forward, uncalibrated** | DA3-Streaming 0.087, Pi-Long 0.094, VGGT-Long 0.110 **S**. VGGT-SLAM 0.053, and 0.041 for version 2.0 **C** ([secondary](https://www.emergentmind.com/topics/vggt-slam-2-0)) |
| **Oxford Spires, 320 sparse frames, ATE** ([LingBot Table 2](https://arxiv.org/abs/2604.14141)) | **LingBot 5.37**, VIPE 10.52, DA3 12.87, π³ 14.03, CUT3R 18.16, DROID 21.84, VGGT 24.78, StreamVGGT 28.41, Fast3R 34.80 **S** |
| **7-Scenes surface quality** | MASt3R-SLAM Chamfer 0.066 m, better than DROID-SLAM **S** |

Where the time goes in the current chain: DROID uses calibrated intrinsics plus global bundle adjustment, which is about 2× better ATE than the best uncalibrated feed-forward result on TUM. TSDF fusion itself is not the bottleneck (seconds on a GPU). The slow parts are the 9–27 min CPU dense build and running everything in series.

### 3. Metric scale and licence

| Model | Metric scale? | Licence |
|---|---|---|
| VGGT-1B | No | Original weights non-commercial. **VGGT-1B-Commercial** allows commercial use except military; gated by an automated application ([repo](https://github.com/facebookresearch/vggt)) |
| VGGT-Long | Sim(3) chunk alignment; metric through the Map-Long variant | Follows the VGGT licence; commercial use needs the Commercial weights ([repo](https://github.com/DengKaiCQ/VGGT-Long)) |
| VGGT-Ω | Not stated | **FAIR Noncommercial** for code and weights |
| DA3 GIANT / LARGE / NESTED | Nested only (output in metres) | CC BY-NC 4.0 |
| DA3 BASE / SMALL / METRIC-LARGE | METRIC-LARGE is monocular metric | **Apache 2.0** ([repo](https://github.com/ByteDance-Seed/Depth-Anything-3)) |
| MapAnything | **Yes** | Code Apache 2.0. Weights: `map-anything` CC BY-NC; **`map-anything-apache` Apache 2.0** |
| π³ / Pi3X | Pi3X "approximate metric" | Code BSD-3; weights **CC BY-NC** |
| Fast3R | No | FAIR NC |
| CUT3R | Yes (metric pointmaps) | CC BY-NC-SA 4.0 |
| StreamVGGT | No | CC BY-NC-SA 4.0 |
| MASt3R-SLAM | Near-metric | CC BY-NC-SA 4.0, and the MASt3R checkpoint is also non-commercial |
| LingBot-Map | No: the paper normalises scale to its anchor frames | Repo says Apache-2.0, but the HF weights repo has no licence field. **Get written confirmation**, as `STREAMING-PLAN.md` already notes |
| DROID-SLAM / DPVO | No | BSD-3 / MIT |

### 4. What fits under 30 s on one A100, and what quality is lost

"Warm" means the container is already running with weights loaded.

| Option | 150 keyframes, warm: forward pass + point cloud + GPU TSDF | Under 30 s? | Licence | Quality lost vs DROID+DA3+TSDF+LingBot |
|---|---|---|---|---|
| **DA3-GIANT-1.1 any-view** (demo pick) | 6–15 s + 1–3 s + 3–8 s ≈ **12–25 s E** | Yes | NC, research demo only | **Poses:** no bundle adjustment. Expect 1.5–2× the ATE of calibrated DROID **E** (from the TUM numbers), which shows up as double walls and texture seams. **Depth:** same model family we use now; loses only pose-conditioning. **Drift:** one global pass over 30 s, so none. **Moving people:** can bias poses. |
| **VGGT-Ω-1B-512** | ≈ **10–20 s E** | Yes | NC | Same bundle-adjustment gap. Best on moving content (Sintel), so the people risk is lower |
| **VGGT-1B-Commercial** | ≈ **15–25 s E** | Yes | Commercial (gated) | Weaker than DA3-G (HiRoom AUC@3 49 vs 80) |
| **MapAnything-apache** | ≈ **12–22 s E** | Yes | Apache | Metric scale for free, but the weakest poses in DA3 Table 2 |
| **DA3-BASE** any-view | ≈ **8–15 s E** | Yes | Apache | Clearly lower pose and F1 quality (DA3-L already drops from 80 to 59 AUC@3 on HiRoom) |
| **LingBot-Map** on A100 | 21 s (150 frames) or 42 s (300 frames) **E from P** | Only at 150 frames; about 12 s on H100 | Needs confirmation | Its streaming ATE is best on long sequences. On a 30 s clip, a single global pass removes most of that edge **E** |
| **DROID + DA3 posed + TSDF** (current, done warm) | ≈ **60–120 s E**; 7 min today **P** | No | BSD / NC | Reference quality |

About dropping LingBot specifically: its dense point map is per-frame depth on every 3rd frame. DA3 any-view on 150 frames produces the same kind of product from half the frames, so thin structures may lose points. The size of that loss is unknown. It needs an A/B on ME340 against held-out views before anyone claims "no loss".

### 5. What this means for Panoptes

1. **Fast path for geometry:** decode the video on the GPU box → about 150 keyframes → DA3-GIANT any-view (poses, depth, confidence) → back-project for the point map plus GPU TSDF. Keep scale from the existing floor and camera-height rule. This replaces the DROID census, DROID camera, DA3 posed depth, LingBot and the CPU dense build: roughly 20–40 min today down to about 20–30 s warm **E**.
2. **Most of the speed gain is infrastructure, not model choice.** The job needs a warm container with weights already on the GPU and no Mac-to-cloud PNG round trip. Our 72 s DA3 wall versus about 10–15 s of compute **E** shows this.
3. **Two A100s:** GPU 0 runs geometry (about 20 s); GPU 1 runs SAM 3 detection and tracks at the same time. Optionally, a 10–20 s bundle adjustment starting from the feed-forward poses runs on GPU 0 afterwards; this is how to recover most of DROID's accuracy advantage **E**. Splitting geometry across both GPUs (two overlapping chunks plus Sim(3) alignment) only pays off above about 300 frames.
4. **Licences:** the demo can use DA3-GIANT, VGGT-Ω or π³ (all non-commercial, research only). The commercial path is VGGT-1B-Commercial, MapAnything-apache or DA3-BASE/METRIC-LARGE, plus LingBot-Map once its weights licence is confirmed in writing.
5. **One measurement settles it** (needs your approval to run a GPU job): on ME340, in one warm A100-80GB container, run DA3-GIANT any-view, VGGT-Ω and VGGT-1B-Commercial on the same 150 keyframes. Compare each one's ATE against the DROID trajectory, and its TSDF depth error on held-out views against the current chain.

**Code read:** `/Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/report_runner/stages.py`, `/Users/adam/.codex/worktrees/panoptes-phase2-video/modal_apps/mono_room.py`, `/Users/adam/.codex/worktrees/panoptes-phase2-video/modal_apps/da3_unposed.py`, `/Users/adam/.codex/worktrees/panoptes-phase2-video/modal_apps/lingbot_room.py`, `/Users/adam/.codex/worktrees/panoptes-phase2-video/docs/phase2/STREAMING-PLAN.md`

Other sources: [FastVGGT](https://arxiv.org/abs/2509.02560) (training-free, 4× faster than VGGT at 1000 frames; only matters for long inputs), [CUT3R](https://arxiv.org/abs/2501.12387), [StreamVGGT](https://github.com/wzzheng/StreamVGGT), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [LingBot-Map](https://github.com/Robbyant/lingbot-map), [LingBot-Map weights](https://huggingface.co/robbyant/lingbot-map).

### scene-understanding
## Fast open-vocabulary 3D object mapping: what published methods do, and a design for 30–60 s

The workable design is: pick about 60 keyframes, run an open-vocabulary detector plus box-prompted SAM 2.1, lift the masks with the poses and depth we already compute, and match objects by 3D voxel overlap. I estimate 15–35 s on one warm A100. The fastest published method (Open-YOLO 3D) takes about 22 s per scene. None of the published methods use SAM "segment everything" (automatic mask generation, AMG) on every 3rd frame, which is what Panoptes does today.

Labels: **[M: hw]** measured by the cited source on that hardware; **[C]** claimed by the authors or vendor, conditions unclear; **[E]** my estimate; **[P]** Panoptes' own measurement from the task brief.

### 1. How they pick keyframes

| Method | Frames processed | Views used for labelling |
|---|---|---|
| ConceptGraphs | Fixed stride: 5 in the main configs, 10 in the `ali-dev` branch | Up to 10 best views per object (by point contribution) → LLaVA-7B captions → GPT-4 summary |
| HOV-SG | Fixed stride; follow-up work reports every 10th frame | CLIP on every segment, merged through an overlap graph |
| OpenMask3D | Every 10th frame (ScanNet) | Top-5 views per 3D mask by visibility, 3 crop scales, CLIP |
| Open3DIS | Every 10th frame; its best ScanNet++ result uses all frames | Superpoint-guided hierarchical merge, point-wise CLIP |
| Open-YOLO 3D | Every 10th frame (ScanNet200), all frames (Replica) | The detector's label on every view, voted per 3D mask (MVPDist) |
| Any3DIS / SAM2Object | Every 10th frame; SAM2 tracks forward and backward from a "pivot" view per superpoint, 7-frame memory | — |
| MaskClustering | Stride-sampled frames | Top-5 masks per instance, multi-scale CLIP crops |
| OVI-MAP | Every 10th frame | Asks the vision-language model again only when a new view adds uncovered surface |
| OnlineAnySeg | Every 10th frame; merges every 5 keyframes | CLIP + FCGF similarity |
| TrackGraph | FastSAM every 24th frame; DINOv3 carries masks in between | — |
| OpenTrack3D | Stream of frames | Top-3 views by share of the object inside the camera view |

The consensus is roughly 3 keyframes per second at 30 fps, or sparser. Panoptes uses every 3rd frame plus the census keyframes, about 290 frames for a 26 s clip (`decide.py:381`). That is 3–8× denser than any of these methods.

### 2. Segmenter or detector used, with licences

| Method | 2D model | 3D proposals | Licence risk for commercial use |
|---|---|---|---|
| ConceptGraphs | SAM automatic masks, or RAM tags + Grounding DINO + SAM | None | MIT code; LLaVA and GPT-4 for captions |
| HOV-SG, SAI3D, OV-SAM3D | SAM automatic masks (OV-SAM3D adds RAM tags) | Superpoints | SAM is Apache-2.0 |
| OpenMask3D | SAM with point prompts, CLIP | Mask3D trained on ScanNet200 | The ScanNet-trained 3D network will not transfer to phone video of work sites |
| Open3DIS | Grounded-SAM (Grounding DINO + SAM) | ISBNet trained on ScanNet200 | Same 3D-network issue |
| Open-YOLO 3D | YOLO-World-X | Mask3D | YOLO-World is GPL-3.0 |
| Any3DIS, SAM2Object | SAM 2 large, video tracking | Superpoints | Apache-2.0 |
| OpenTrack3D | YOLO-World-X + SAM2-L + DINO; no 3D network needed | — | GPL-3.0 via YOLO-World |
| MaskClustering, OVI-MAP, OnlineAnySeg | CropFormer (OVI-MAP: CropFormer beat SAM2 by +5.7 AP50) | — | CropFormer licence not checked |
| ESAM-E, TrackGraph | FastSAM | — | AGPL-3.0 (ultralytics) |

Commercially usable fast options:
- **OmDet-Turbo:** Apache-2.0. Claimed 100 FPS on A100 with TensorRT [C].
- **SAM 2.1:** Apache-2.0. The large model is listed at 39.5 FPS on A100 [M: A100, SAM2 README].
- **SAM 3 / 3.1:** under the SAM License, which allows commercial use; it excludes military uses and requires trade-control compliance.

Research or demo only:
- **YOLOE:** AGPL-3.0. Has a prompt-free mode. v8-L runs at 102 FPS on T4 with TensorRT [C].
- **YOLO-World:** GPL-3.0. 52 FPS on V100 [C].
- **PanSt3R:** non-commercial licence.

### 3. Runtime per scene on one GPU

| Method | Runtime | Label |
|---|---|---|
| OpenMask3D | 554 s (ScanNet200), 547 s (Replica) | [M: A100-40GB, by the Open-YOLO 3D authors] |
| Open3DIS | 360 s (ScanNet200), 35 s (Replica) | [M: A100-40GB, same source] |
| **Open-YOLO 3D** | **21.8 s (ScanNet200), 16.6 s (Replica)** | [M: A100-40GB] |
| OpenTrack3D | 266 s + about 90 s labelling; 2D stage 0.52 s/frame. Prior best (DetailMatters) 484 s + 330 s | [M: A100] |
| ConceptGraphs | 0.40 s/frame [M: RTX 4090]; 0.16 FPS [M: RTX 2080 Ti]; about 16 min per 2,000-frame Replica scene | [M: A100, reported by TrackGraph] |
| HOV-SG | 0.03 FPS | [M: unspecified large desktop, rough]; its own paper calls construction slow |
| TrackGraph / OVI-MAP | 3.9–8.9 min / 15.1 min per Replica scene | [M: A100] |
| OVI-MAP per keyframe | CropFormer 965 ms, association 176 ms | [M: RTX 3090] |
| ESAM / ESAM-E | 80 ms/frame, plus 1,369 ms (SAM) or 20 ms (FastSAM) | [M: GPU not stated]; about 10 FPS [C] |
| OnlineAnySeg | 15 FPS | [M: RTX 4090] |
| MaskClustering-style pipeline (CropFormer) | 96 s per scene (in SceneVerse++) | [M: hardware not stated] |
| SAI3D, OV-SAM3D | Not reported | SAM automatic masks per view, so minutes [E] |
| SAM automatic masks (32×32 grid) | Mask decoder about 894 ms per image | [M: V100, TinySAM] |
| Panoptes SAM2 automatic masks | About 2.5 s/frame | [P: L4] |
| SAM-V / PanSt3R | 38–49 s / about 4.8 s per scene | [M: L40S] |
| LangSplat | About 168 s per frame amortised (35 s of that is SAM) | [M: RTX 3090, by the Online Language Splatting authors] |
| Dr. Splat | About 10 min on top of a trained splat | [C] |
| Gaussian Grouping | Trains 30K iterations alongside the splat on an A100; DEVA mask association takes 1 min | [C] |

Two things rule approaches out:
- **Splat-based methods** (Gaussian Grouping, LangSplat and similar) need an already trained splat plus minutes of extra work. They cannot fit 30–60 s.
- **SAM 3 video tracking:** Meta claims 32 fps on H100 for a medium number of objects with 3.1 multiplexing [C]. Panoptes measured 0.7–1.4 fps [P]. That 20–40× gap is worth investigating for people tracking. For static objects it doesn't matter, because poses plus depth replace 2D tracking.

### 4. Object quality (ScanNet200 open-vocabulary AP unless noted; all as reported by the papers)

| Method | AP | Notes |
|---|---|---|
| OpenTrack3D | **26.0** (Replica 23.9) | No trained 3D network |
| Any3DIS | 25.8 (its own paper) / 19.1 (OpenTrack3D's table, 2D-only setting) | The two settings differ |
| Open-YOLO 3D | 24.7 | Uses Mask3D trained on ScanNet |
| Open3DIS | 23.7 with 3D network / 18.2 2D-only | |
| OpenMask3D | 15.4 | CLIP naming |
| ESAM | 13.7 | |
| MaskClustering | 12.0 | |
| SAI3D | 9.6 | |
| OV-SAM3D | 9.0 | |
| ConceptGraphs vs HOV-SG (Replica, not AP) | F-mIoU 0.23 vs 0.386 | HOV-SG paper; ConceptGraphs node precision 71% (human-judged) |

Takeaways:
- Detector labels name objects better than CLIP crop voting: Open-YOLO 3D beats OpenMask3D by about 9 AP.
- Multi-view CLIP fusion is noisy: 13% versus a 31% upper bound (Bare Necessities, ScanNet++).
- 2D-only lifting with detector + SAM2 is now the best option without a trained 3D network.

### 5. What this means for Panoptes

**Where the 30 minutes goes today (read from the stage graph):**
- **SAM2 already runs on Modal, not on the Mac.** It uses one L4 container (`max_containers=1`) that loops serially over about 290 frames with the full 32×32 grid (1,024 prompts per frame).
- **The object map runs on the Mac CPU.** The `object_map` stage in `stages.py` has no `compute=` setting, so it defaults to local CPU; the import stage is local too.
- **Naming calls Gemini with 10 objects per request, one request at a time.**
- **The design is already ConceptGraphs** ("segment everything, lift, then name"), which is the slowest family in table 3.

**Proposed design:** one warm Modal A100 function; weights on a volume; container warmed while the upload is still running.

1. **Keyframes:** about 45–60 of the 900 frames. Take every 15–20th frame, move each to the sharpest frame within ±3, and skip a keyframe if the camera moved less than 10 cm and less than 5°.
2. **Detect and segment** in batches of 8–16 frames. OmDet-Turbo prompted with the existing 40-name list in `name_video_entities.py`, plus catch-alls ("object", "equipment", "container"), gives boxes; SAM 2.1-L with box prompts gives masks. About 0.1 s per frame, so **6–10 s** [E].
   - For a research demo, YOLOE is an alternative detector (AGPL).
   - SAM 3.1 with vocabulary prompts would work, but each prompt adds cost; [P] measured 323 ms per frame for 2 prompts on L4. Estimated 0.5–1.5 s per frame with 40 prompts on A100 [E], so only if split across both GPUs.
3. **Lift** each mask at 1/4 resolution using the DA3 depth and the DROID or LingBot poses, voxelise at 3–5 cm, all in torch on the GPU. Under 1 s [E].
4. **Associate** masks into objects by greedy voxel-overlap merge with a label vote (the existing `--method overlap`, vectorised with voxel hashing as OnlineAnySeg does), then denoise and fit a box aligned to the floor. **2–6 s** [E], against 12 min on the Mac today [P].
5. **Name:** the detector label is the name. Optional refinement: one best crop per object, sent as a single batch to Qwen3-VL on the second A100 (Apache-2.0; the existing commercial naming profile) or to Gemini with all requests at once. **5–15 s** [E], overlapping step 4.

**Total:** about **15–35 s warm** on one A100 [E], plus 20–60 s if the container starts cold [E]. The second A100 runs depth, LingBot or people tracks at the same time.

**Things to drop:**
- SAM2 automatic masks on every frame (at most, a 16×16 sweep on about 10 keyframes to catch objects outside the vocabulary).
- 2D video tracking for static objects.
- Any ScanNet-trained 3D proposal network (Mask3D, ISBNet).

**Risks:**
- Objects outside the vocabulary will be missed; the catch-all prompts and the VLM relabel step reduce this.
- Monocular depth noise will cause some wrong merges and splits.
- Thin objects (pipes, hoses, cables) are poorly captured by boxes.
- YOLO-World, YOLOE and FastSAM are GPL or AGPL, so they are for the demo only.

Sources:
- [Open-YOLO 3D](https://arxiv.org/html/2406.02548v2), [abs](https://arxiv.org/abs/2406.02548), [repo](https://github.com/aminebdj/OpenYOLO3D)
- [OpenTrack3D](https://arxiv.org/html/2512.03532)
- [ConceptGraphs](https://arxiv.org/html/2309.16650v1), [repo](https://github.com/concept-graphs/concept-graphs), [ali-dev branch](https://github.com/concept-graphs/concept-graphs/tree/ali-dev)
- [HOV-SG](https://arxiv.org/html/2403.17846v2)
- [Bare Necessities](https://arxiv.org/html/2412.01539v1)
- [BiMoSG](https://arxiv.org/html/2605.31067v1)
- [TrackGraph](https://arxiv.org/html/2609.31005)
- [OVI-MAP](https://arxiv.org/html/2603.26541)
- [ESAM](https://arxiv.org/html/2408.11811)
- [OnlineAnySeg](https://arxiv.org/html/2503.01309)
- [Any3DIS](https://arxiv.org/html/2411.16183)
- [MV3DIS](https://arxiv.org/html/2604.08916)
- [MaskClustering](https://arxiv.org/html/2401.07745)
- [SceneVerse++](https://arxiv.org/html/2604.01907v1)
- [OV-SAM3D](https://arxiv.org/html/2405.15580v2)
- [Open3DIS](https://open3dis.github.io/)
- [SAI3D](https://arxiv.org/abs/2312.11557)
- [SAM2Object](https://github.com/jihuaizhaohd/SAM2Object)
- [SAM-V](https://arxiv.org/html/2609.25490v1)
- [PanSt3R](https://github.com/naver/panst3r)
- [Gaussian Grouping](https://arxiv.org/html/2312.00732v2)
- [Online Language Splatting](https://arxiv.org/html/2503.09447v2)
- [Dr. Splat](https://drsplat.github.io/)
- [YOLOE](https://github.com/THU-MIG/yoloe)
- [YOLO-World](https://arxiv.org/abs/2401.17270), [licence](https://github.com/AILab-CVC/YOLO-World/blob/master/LICENSE)
- [OmDet-Turbo](https://arxiv.org/abs/2403.06892), [model card](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [SAM 3.1 blog](https://ai.meta.com/blog/segment-anything-model-3/), [SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md), [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- [TinySAM](https://arxiv.org/pdf/2312.13789)

Repo files read (no edits made):
- /Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/report_runner/stages.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/report_runner/decide.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/modal_apps/sam2_everything.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/build_video_object_map.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/name_video_entities.py

### object-gen
## Fast single-image-to-3D for many objects on two A100s

**Labels:** **[P]** measured in Panoptes run manifests (A100-SXM4-80GB). **[M]** measured by the source, with its hardware. **[C]** claimed by the source. **[E]** my estimate. To convert H100 figures to A100 I multiply by about 1.75. That factor comes from our LingBot measurement (7.1 fps on A100 vs 12.4 fps on H100) and is itself **[E]**.

### 0. Why SAM 3D takes 38 min today (from our manifests and code)

| Fact | Value |
|---|---|
| SAM 3D calls per 30 s clip | 146–157 **[P]** (samsclub-291, walmart-267, me340-241) |
| GPU time per call (mesh only, 25+25 steps) | median 10.0–13.2 s, p90 10.8–14.1 s **[P]**. The 19 s in the brief likely includes overhead. |
| Wall time per call | median 14–17 s **[P]**. Sum over a clip: 2567–2811 s, about 43–47 min **[P]** |
| Attempts per object (tried one after another) | 1: 55, 2: 87, 3: 30, 4: 10, 5: 20. Mean about 2.3 **[P]** |
| Objects accepted | 12–41 of 53–79 modelled per clip. 131 of 208 not accepted overall; 85 of those failed the "source-view gate (silhouette/depth)" **[P]** |
| GPU parallelism | `modal_apps/sam3d_research.py:77` sets `max_containers=1` and has no `@modal.concurrent`. The 4 client workers therefore queue on one A100. |
| Released fast mode in use? | No. The upstream `run(...)` accepts `use_stage1_distillation`, `stage1_inference_steps` and `stage2_inference_steps` (upstream `inference_pipeline_pointmap.py`). Panoptes never passes them, so it runs 25+25 steps with classifier-free guidance. |
| Batching | Upstream asserts `image.ndim == 3  # no batch dimension as of now`. Batching would need code changes. |

So about 1600–2000 GPU-seconds (27–33 min) goes through a single container.

### 1. Speed per object and batchability

| Model | Time per object | A100 time | Batchable? |
|---|---|---|---|
| **SAM 3D Objects**, as we run it | — | 10–13 s **[P]** | No (upstream code) |
| SAM 3D, upstream baseline | 31.0 s total on A800: sparse structure 4.1 s, SLaT 9.7 s, mesh decoder 13.8 s, other 3.4 s **[M, A800]** ([Fast-SAM3D](https://arxiv.org/html/2602.05293)) | A800 has A100 compute | — |
| SAM 3D, shortcut on the geometry model (released checkpoint) | 4-step is 10× and 1-step 38× faster on that stage; "sub-second shape and layout" **[C]** ([paper §C.4](https://arxiv.org/html/2511.16624)). The texture/refinement stage has no distillation, but the paper says it "performs well with fewer steps". | Full mesh about 4–7 s/call; `stage1_only` (coarse 64³ voxels plus pose) about 0.3–1 s **[E]** | Only by patching the code |
| Fast-SAM3D (training-free) | 31.0 → 11.6 s per object; 462 → 230 s per scene; F1 92.3 → 92.6 **[M, A800]** | Stacked on the shortcut: about 3–5 s **[E]** | No; needs a separate install |
| TRELLIS.2-4B | 512³: about 3 s. 1024³: about 17 s. 1536³: about 60 s. All including texture **[C, H100]** ([repo](https://github.com/microsoft/TRELLIS.2)). GLB export is slow and leaves CPU and GPU idle ([#18](https://github.com/microsoft/TRELLIS.2/issues/18)). | 512³ about 5–6 s plus export **[E]** | Not documented |
| TRELLIS v1 (image-large) | Not found | 10–20 s **[E]** | Yes, in its sparse framework |
| Hunyuan3D-2mini-Turbo + FlashVDM | Shape in under 1 s on a 4090 **[C]**. The default Hunyuan3D-2 shape takes over 30 s **[C]** ([FlashVDM](https://arxiv.org/html/2503.16302)). Texturing (Paint-Turbo) is separate. | Shape about 1 s; texture 10–20 s **[E]** | Probably (fixed-length latent) **[E]** |
| SPAR3D | 0.7 s **[C]** ([arXiv](https://arxiv.org/abs/2501.04689)) | about 1.2 s **[E]** | Yes, `run.py --batch_size` |
| SF3D | 0.5 s **[M, H100]** ([paper Table 1](https://arxiv.org/html/2408.00653)) | about 0.9 s **[E]** | Yes, `--batch_size` |
| TripoSR | 0.3 s **[M, H100, SF3D authors]**; under 0.5 s **[C, A100]** ([repo](https://github.com/VAST-AI-Research/TripoSR)) | about 0.5 s | Yes |
| InstantMesh | 32.4 s **[M, H100, SF3D authors]** | about 55 s **[E]** | — |
| LGM | 64.6 s including mesh extraction **[M, H100, SF3D authors]** | over 60 s **[E]** | — |
| RecGen (TRI) | "1.8× faster than SAM3D", 50 steps **[C]** ([arXiv](https://arxiv.org/html/2604.27106)) | 63.6 s provider-function time for one 2-view call **[P]** (`docs/research/2026-09-15-recgen-wire23-multiview-quality.md`) | No; Panoptes runs it one call at a time on purpose (`SERIAL_GENERATORS`) |

### 2. Quality on cluttered real crops with masks

| Model | Evidence | Fit for Panoptes |
|---|---|---|
| **SAM 3D Objects** | On SA-3DAO (real, occluded scenes) Meta measured F1@0.01 0.234 and Chamfer 0.040. Competitors: TRELLIS 0.148/0.090, HY3D-2.0 0.157/0.087, HY3D-2.1 0.140/0.113, TripoSG 0.153/0.084, Hi3DGen 0.163/0.094 **[M, self-reported]** ([paper Table 2](https://arxiv.org/html/2511.16624)). It also claims 5:1 human preference **[C]**. | Best. It takes our pointmap and returns a metric pose. |
| RecGen | Claims +30.1% shape, +9.1% texture and +33.9% pose over SAM3D on occluded RGB-D data **[C]** | Good in principle. Our own audits rejected #12 and #23, it is slow, and it is non-commercial. |
| Amodal3R | TRELLIS plus an occlusion-aware branch; under 10 s per object **[C]** ([arXiv](https://arxiv.org/html/2503.13439)) | Worth testing. Licence not verified. |
| TRELLIS.2 / Hunyuan3D | Built for single objects with the background removed. Their predecessors score lower on SA-3DAO (above). | Mid. Expect made-up shape around occluders **[E]**. |
| SF3D / SPAR3D / TripoSR | Trained on single-object renders. GSO F-score@0.1: SF3D 0.701, TripoSR 0.645 **[M, synthetic]** | Weakest on clutter. Would likely fail our source-view gate more often **[E]**. |

### 3. Licences

| Model | Licence | Commercial use |
|---|---|---|
| SAM 3D Objects | SAM License | Yes, no user-count cap. Prohibits military, **nuclear-industry**, weapons and espionage uses; trade controls apply ([LICENSE](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE)). The nuclear clause matters for EHS customers. Keep texture baking off, or use the `pytorch3d` rendering engine, to avoid nvdiffrast. |
| Fast-SAM3D | MIT code on top of SAM 3D weights | Yes |
| TRELLIS / TRELLIS.2 | MIT | The model is fine, but nvdiffrast/nvdiffrec are NVIDIA research/evaluation-only ([licence](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt)). The export path would need replacing. |
| Hunyuan3D-2 / mini / turbo | Tencent Hunyuan 3D 2.0 Community | Not valid in the EU, UK or South Korea; above 1M monthly users you need Tencent's approval |
| SPAR3D, SF3D | Stability Community | Free below $1M annual revenue, then an enterprise licence |
| TripoSR, LGM, InstantMesh | MIT, MIT, Apache-2.0 (code) | Yes. Check LGM/InstantMesh's multi-view base weights. |
| **RecGen** | CC-BY-NC-4.0 weights, TRI non-commercial code | **Research demo only** |

### 4. How many models can two A100s make in 60–90 s?

Assumes warm containers and inputs ready: 120–180 GPU-seconds in total. **All rows are [E].**

| Setup | Seconds per call | Models in 60–90 s |
|---|---|---|
| SAM 3D as deployed, about 2.3 attempts per object | 10–13 **[P]** | 4–8 objects |
| As deployed, 1 attempt per object, 2 containers | 10–13 | 9–18 |
| SAM 3D with shortcut (4 steps) + stage 2 at about 12 steps, mesh only | 4–7 | 17–45 |
| Above + 2 processes per GPU + torch.compile (and/or Fast-SAM3D) | about 3 effective | 40–60 |
| SAM 3D `stage1_only` with shortcut (coarse voxels plus pose) | 0.3–1 | every object (over 120) |
| TRELLIS.2 at 512³ | 5–6 plus export | 20–35 |
| Hunyuan3D-2mini-Turbo, shape only | about 1 | 120–180, untextured, no pose |
| SF3D / SPAR3D / TripoSR, batched | 0.5–1.2 | 100–300+ |
| RecGen (1 view, parallelised) | 20–60 | 3–9 |

These two A100s also have to run camera, depth, SAM and LingBot. Realistically the object stage gets about 1 GPU for about 60 s. That means **about 10–25 full SAM 3D models plus coarse models for the rest [E]**.

### 5. What this means for Panoptes

1. **The 38 min is mostly orchestration, not the model.** About 150 calls go through one container one at a time (`max_containers=1`), each object retries up to 5 times in sequence, and the released shortcut mode is off. Fixing those, before changing model, is the biggest win.
2. **Recommended model: stay on SAM 3D Objects**, for both the demo and commercial use. It has the best measured real-world accuracy, it uses our pointmap for metric pose, and its licence allows commercial use (watch the nuclear clause). Plan:
   - Turn on `use_stage1_distillation=True`, `stage1_inference_steps=4` and `stage2_inference_steps≈12`. Before adopting, A/B it on one clip against today's 18–58% acceptance rate. It must not make the gate worse.
   - Two tiers:
     - `stage1_only` coarse voxels plus pose for every object shown in the 2–3 min report.
     - Full meshes for the top 15–25 EHS objects.
     - The rest refine in the background after the report is on screen.
   - Run the attempts for one object at the same time (2 views in parallel, keep the best by the existing gate), not one after another. Set `max_containers≥2` or run 2 processes per 80 GB GPU. Keep containers warm. Send crops instead of full 1280×720 float32 pointmaps (about 11 MB per call **[E]**).
3. **RecGen** is TRI's "Reconstruction by Generation" (ECCV'26). It is built on TRELLIS-image-large and turns RGB-D + mask + camera intrinsics into a textured mesh, splat and 6-DoF pose.
   - In `stages.py` it is the second-round generator for the objects the SAM 3D plan hands over.
   - It is forced to one call at a time (`--workers 1`, `SERIAL_GENERATORS`), which is why that stage takes 1–3 h.
   - It is non-commercial, so it should come out of the fast path.
4. **Likely reason you don't see many models in the report:** in the three clips measured, only 12–41 SAM 3D models per clip passed acceptance **[P]**. Box stand-ins are shown only after an eye review approves them (`stages.py`, D13). Many objects therefore end up with no rendered model. I checked the code and manifests only, not the report viewer.
5. **Fallbacks:**
   - Hunyuan3D-2mini-Turbo is the fastest shape-only option, but its licence excludes the EU, UK and South Korea.
   - TRELLIS.2 is MIT, but it is built for single isolated objects and depends on nvdiffrast.
   - SF3D, SPAR3D and TripoSR are fast enough to cover everything, but are placeholder quality only on clutter.

Relevant files:
- /Users/adam/.codex/worktrees/panoptes-phase2-video/modal_apps/sam3d_research.py (line 77 `max_containers=1`; lines 111–113 run call with no distillation flags)
- /Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/complete_video_objects.py (lines 59–66 attempt policy)
- /Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/report_runner/stages.py (lines 40–56, 760–860)
- /Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/{samsclub-a2-object-models-291,walmart-object-models-267,me340-object-models-241}-sam3d/manifest.json

Sources:
- [SAM 3D paper](https://arxiv.org/html/2511.16624)
- [sam-3d-objects repo](https://github.com/facebookresearch/sam-3d-objects)
- [Fast-SAM3D](https://arxiv.org/html/2602.05293) / [code](https://github.com/wlfeng0509/Fast-SAM3D)
- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
- [TRELLIS.2 issue #18](https://github.com/microsoft/TRELLIS.2/issues/18)
- [FlashVDM](https://github.com/Tencent-Hunyuan/FlashVDM)
- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- [Hunyuan3D-2 licence](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE)
- [SPAR3D](https://huggingface.co/stabilityai/stable-point-aware-3d)
- [SF3D paper](https://arxiv.org/html/2408.00653)
- [TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- [RecGen paper](https://arxiv.org/html/2604.27106) / [repo](https://github.com/TRI-ML/recgen)
- [Amodal3R](https://arxiv.org/html/2503.13439)
- [MV-SAM3D](https://arxiv.org/abs/2603.11633)
- [nvdiffrast licence](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt)
- [fal SAM 3D guide](https://fal.ai/learn/devs/sam-3d-developer-guide) (claims 4–8 s per object, hardware not stated)

### products
## Fast video-to-3D and EHS analysis: how other products and papers do it, and how fast

**Labels:** **[M]** = measured by the source (hardware given where known) · **[C]** = vendor or author claim with no stated setup · **[E]** = my estimate.

### 1. Consumer and prosumer capture apps

| Product | Live, during capture | Later | Time to result | Label |
|---|---|---|---|---|
| Scaniverse (Niantic) | On-device tracking | Splat and mesh trained on the phone | "about a minute"; a reviewer says most captures finish in under 90 s | C (vendor), reviewer-reported — [dev.scaniverse.com](https://dev.scaniverse.com/news/creating-splats-which-app-to-choose), [radiancefields](https://radiancefields.com/platforms/scaniverse) |
| Polycam | LiDAR mode builds the mesh on screen as you scan; one room takes "seconds" on the device | Photo/video splats and photogrammetry are processed in the cloud | "ten or more minutes"; a third-party guide says 15–45 min | C — [CG Channel](https://www.cgchannel.com/2020/11/polycam-turns-your-ipad-pro-into-a-real-time-lidar-scanner/), [3dgsviewers](https://www.3dgsviewers.com/learn/guide/polycam) |
| Luma | Mobile app no longer makes splats | Cloud | Commonly quoted as 20–45 min; one test waited about 1 h in the queue (reported by the Scaniverse blog, a competitor) | C — [same Scaniverse post](https://dev.scaniverse.com/news/creating-splats-which-app-to-choose) |
| Matterport | Capture app aligns scans on site | Cortex cloud processing, measurements, defurnishing | About 30 min for 1–2 scans; "a few hours" for about 1,500 sq ft; 24–48 h for 200+ scans; face blurring adds about 2 min per scan | C — [Matterport support](https://support.matterport.com/hc/en-us/articles/212952928-How-long-does-a-model-take-to-process-) |
| Niantic Spatial Capture | Scaniverse on-device splats and meshes | VPS maps, geo-referenced digital twins, multi-sensor data | No time published | — [nianticspatial.com](https://www.nianticspatial.com/products/capture) |

### 2. Site and construction capture

| Product | Live | Later | Time | Label |
|---|---|---|---|---|
| Spot + Leica BLK ARC | GrandSLAM (LiDAR + visual + IMU) runs on the scanner module; BLK Live app shows a real-time preview; 420k points/s | Registration in Cyclone REGISTER 360 (desktop) or HxDR (cloud) | No time published | spec sheet — [Leica](https://shop.leica-geosystems.com/sites/default/files/2025-01/BLKARC_SpecSheet.pdf) |
| OpenSpace | 360° video walk (capture only) | Cloud "Vision Engine": places images on the plan, BIM compare | 15 min on average; a first capture can take a couple of hours; publication within 4 h | C — [FAQ](https://www.openspace.ai/faq/) |
| Reconstruct | Point-cloud processing starts while the upload is still running | Point cloud, walkthrough, as-built plans; email when ready | No time published; recommends at most 10 min of 360° video per point cloud | C — [help page](https://help.reconstructinc.com/hc/en-us/articles/360029498731-Uploading-and-Processing-360-Videos) |
| Buildots | 360° helmet camera | Cloud computer vision compared against BIM and the schedule | 24–36 h; faster turnaround on request | C — [buildots.com](https://buildots.com/product/) (from a search snippet) |

### 3. EHS and video analytics

| Product | Live | Later | Latency | Label |
|---|---|---|---|---|
| Intenseye | Edge inference on existing CCTV over RTSP, or on the Sentinel hub (Jetson Orin NX) | Cloud EHS workflows and analytics | "Subsecond"; can stop machinery in under 1 s | C — [Sentinel blog](https://www.intenseye.com/blog/introducing-sentinel-physical-ai-system-for-safety) |
| Protex AI | Edge appliance does detection, blurring and encryption | Cloud gets clips and metadata for dashboards, reports and plain-language answers | 1–2 s from detection to signal | C — [protex.ai](https://www.protex.ai/) (snippet), [edge post](https://www.protex.ai/post/edge-processing---what-it-means-for-privacy-latency-and-scalability) |
| Voxel | Runs continuously on existing cameras | Hybrid cloud, "Actions" workflow; adapts to a new site in 48 h | No number published | C — [voxelai.com](https://www.voxelai.com/) |
| NVIDIA VSS / Metropolis | Real-time CV microservice (DeepStream: RT-DETR or Grounding DINO, NvDCF tracker with SAM2 masks) | File mode: cut into 10 s chunks (20 frames each), spread chunks across GPUs in parallel, VLM caption per chunk, LLM summary, then merge the metadata | 1-min video: 12.1 s on 1×H100, 5.70 s on the 2×2 layout; 60-min video: 265 s on 1 GPU, 64.7 s on 4×4 | M (NVIDIA, H100) — [perf](https://docs.nvidia.com/vss/3.2.1/performance-lvs.html), [CV pipeline](https://docs.nvidia.com/vss/2.4.0/content/cv_pipeline.html) |

None of these EHS products builds 3D from a moving video. They use fixed cameras, zones drawn once, 2D detection, a clip per event, and do the analytics later.

### 4. Fast Gaussian splat training

| Method | Time | Hardware | Quality / Gaussians | Code licence | Label |
|---|---|---|---|---|---|
| Vanilla 3DGS, 30k steps | 20.93 min (Mip-NeRF 360) | RTX 4090 | PSNR 27.53 / 2.63M | Inria, non-commercial | M — [FastGS paper](https://arxiv.org/html/2511.04283v3) |
| gsplat default, 30k | 19.39 min | A100 | 29.00 / 3.24M | Apache-2.0 | M — [gsplat eval](https://docs.gsplat.studio/main/tests/eval.html) |
| gsplat MCMC, 1M cap, 30k | 15.42 min, 1.98 GB | A100 | 29.18 / 1M (3M cap: 27.63 min, 29.65) | Apache-2.0 | M — same |
| gsplat, 7k steps | 5m35s | TITAN RTX | 27.21 (vs 28.95 at 30k) | Apache-2.0 | M — same |
| Taming 3DGS (budget mode) | 5.36 min on Mip-NeRF 360; its own paper claims Tanks&Temples 28 → 7 min | 4090 | 27.48 / 0.68M | Built on Inria code; licence not checked | M / C — [paper](https://arxiv.org/html/2406.15643) |
| DashGaussian | 3.23 min (on Taming); 6.35 min when FastGS re-ran it | GPU with about 80 TFLOPS FP32 (an RTX 4090) | 27.73 / 2.40M | CC BY-NC-SA | M — [arXiv](https://arxiv.org/html/2503.18402v1), [repo](https://github.com/YouyuChen0207/DashGaussian) |
| FastGS | 1.93 min (Mip-NeRF 360), 1.32 min (T&T), 1.28 min (Deep Blending) | 4090 | 27.56 / 0.40M | MIT, but "adhere to" the 3DGS, Taming and Speedy-Splat licences, so research-only in practice | M — [repo](https://github.com/fastgs/FastGS) |
| EDGS (dense RoMa-matched initial points, no densification) | Reaches 3DGS quality in 15% of the training time; the full schedule takes 23–30 min | A100 | about half the splats of 3DGS | CC BY-NC-SA | C / M — [arXiv](https://arxiv.org/html/2504.13204v2) |
| SIGGRAPH Asia 2025 challenge (input: video + noisy SLAM poses + SLAM point cloud; 60 s cap, 30 s target; the clock covers reconstruction only) | Winner 55 s; 3rd place 34 s | Organizers don't state hardware; the winner's paper uses a 4090 | Winner PSNR 28.43; 3rd 27.58 | Winner's code CC BY-NC-SA | M — [challenge](https://gaplab.cuhk.edu.cn/projects/gsRaceSIGA2025/), [winner](https://arxiv.org/html/2601.19489) |
| AnySplat (feed-forward, no training loop) | 32 views 1.4 s, 64 views 4.1 s, 200 views 33 s; optional 1,000-step refinement under 2 min | GPU not stated | Matches pose-aware baselines (authors' claim) | Repo MIT; weights licence not checked | M — [paper](https://arxiv.org/html/2505.23716v1) |
| InstantSplat | "40 s" for sparse views without known poses | — | — | NVlabs; licence not checked | C — [site](https://instantsplat.github.io/) |

**GPU caveat [E]:** most "100 s" figures are from an RTX 4090. Splat rasterization runs on FP32 CUDA cores: about 19.5 TFLOPS on an A100 against about 82 on a 4090. Expect an A100 to be at least 1.5–2× slower per step.

### 5. Published speeds for the pipeline's pieces

| Model | Speed | Label |
|---|---|---|
| SAM 2.1 | hiera-large 39.5 FPS, tiny 91.2 FPS (A100, torch 2.5.1, compiled) | M — [sam2 repo](https://github.com/facebookresearch/sam2) |
| SAM 3 / 3.1 | 30 ms per image with 100+ objects on H200; in video, near real-time for about 5 objects, cost grows linearly with object count; 3.1 multiplexing goes from 16 to 32 fps on one H100, up to 16 objects per pass | C — [Meta](https://ai.meta.com/blog/segment-anything-model-3/), [arXiv](https://arxiv.org/html/2511.16719v1) |
| DA3 | About 20 FPS at 768×1024 with ViT-L on A100. **GIANT and LARGE are CC BY-NC 4.0**; BASE, SMALL, METRIC-LARGE and MONO-LARGE are Apache-2.0 | M / C — [repo](https://github.com/ByteDance-Seed/Depth-Anything-3) |
| VGGT | Commercial use only through the gated VGGT-1B-Commercial checkpoint | — [repo](https://github.com/facebookresearch/vggt) |

### 6. What latency users accept

| Category | Typical latency | Examples |
|---|---|---|
| Safety alert | Under 1–2 s, on the edge | Intenseye, Protex |
| On-device scan | About 1 min | Scaniverse |
| Cloud splat or photogrammetry | 10–60 min, plus queue | Polycam, Luma |
| Construction 360° progress | 15 min average, up to 24–36 h | OpenSpace, Buildots |
| Enterprise digital twin | Hours to 2 days | Matterport |
| Text summary of video | 1 h of video in about 1–4.5 min on 1–16 H100s | NVIDIA VSS |

Every product that takes more than a minute works asynchronously and emails when done (Matterport, OpenSpace, Reconstruct). None of them puts complete per-object 3D models on the fast path.

### What this means for Panoptes

1. **The 2–3 min target**
   - A full 3D report in 2–3 min would beat every cloud 3D product surveyed. Only Scaniverse (splat only, on the phone) and VSS (text only) are faster.
   - They all work in two tiers: a quick first look, then full quality delivered later.
   - Suggested v1 in 2–3 min: camera path, dense points and mesh, named objects (2D masks lifted to 3D boxes), people with paths, events, and a small-budget splat.
   - v2 streams in afterwards: SAM 3D complete models and the high-quality splat.
   - Take RecGen off the fast path. It runs 1–3 h serially and no surveyed product does anything like it in its fast path.
2. **Why it's slow, from reading the repo**
   - **SAM2 runs on Modal, not locally.** `modal_apps/sam2_everything.py` runs SAM 2.1 hiera-large in "everything" mode with 32×32 = 1,024 point prompts per frame. It uses one L4 (`max_containers=1`) and loops over the frames one by one. That works out to about 2.5 s per frame [M, from your run]. Meta's 39.5 FPS is for tracking a video, not everything-mode.
     - Fix: about 20–30 keyframes, spread across containers or both A100s, or switch to SAM 3 concept detection (you measured 323 ms per frame on L4) plus tracking [E: under 1 min].
   - **Splats.** `stages.py` trains gsplat MCMC for 60k steps with a 2.5M cap and pose refinement on an H100, which takes 20–25 min. The papers train 30k steps with 0.2–0.4M Gaussians in 1.3–1.9 min on a 4090, and the challenge winner hit PSNR 28.4 in 55 s from SLAM poses.
     - Fast tier: 5–10k steps, 0.3–0.5M cap, start from the DA3/TSDF points, train at half resolution [E: 60–120 s on A100].
     - Keep the current 60k / 2.5M run as the v2 splat.
     - Rebuild the ideas on gsplat (Apache-2.0). The FastGS, DashGaussian and EDGS code is non-commercial.
   - **SAM 3.1 people tracks.** The 3 windows run one after another at 0.7–1.4 fps, against Meta's claimed 32 fps on H100. Run the windows in parallel on both GPUs and sample people at 10 fps [E: 30–60 s].
   - **Borrow the VSS pattern.** Cut the video into chunks, fan them out across GPUs, then merge the metadata. The same applies to Gemini naming: sending the requests concurrently instead of one by one should bring 6 min down to about 10–30 s [E].
   - **Mac CPU stages.** The object map (12 min) and import (13 min) run on the Mac. No cloud product does its heavy processing on the uploader's machine; move these to cloud containers.
3. **Licences for later commercial use**
   - DA3-GIANT, which the posed-depth step uses, is CC BY-NC 4.0. Apache-2.0 alternatives: DA3-BASE, DA3-SMALL, DA3METRIC-LARGE.
   - SAM 2.1 weights are Apache-2.0. gsplat is Apache-2.0.
   - SAM 3 / 3.1 use a custom licence; check its terms.
   - The FastGS, DashGaussian, EDGS and challenge-winner code is non-commercial or built on Inria code, so demo only.
   - VGGT is only available commercially as VGGT-1B-Commercial.
4. **Rough critical path on 2×A100 [E]**
   - Decode and pick keyframes: 10–15 s.
   - GPU0: camera path and depth.
     - LingBot on 900 frames at your measured 7.1 fps ≈ 127 s, or about 42 s if sampled at 10 fps.
     - DA3 posed depth: 72 s for 155 views (measured).
     - Then TSDF: 15–30 s, then the fast splat: 60–120 s.
   - GPU1, in parallel: SAM 3 on keyframes (about 10 s), people tracks (30–60 s), events.
   - Object naming runs concurrently: 10–30 s.
   - Critical path totals 2.5–4 min. Getting to 2–3 min needs 10 fps sampling, with the splat as the last layer to arrive.
   - Complete models: SAM 3D takes about 19 s per call when warm [M]. 20 objects on 2 GPUs ≈ 3+ min [E], so they belong in v2.