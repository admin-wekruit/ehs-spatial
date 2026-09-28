# M3 follow-up E5b: faster splats on ME340

Question (FAST-PATH-PLAN.md, E5 follow-up): how good a gsplat preview of ME340's walk shot can we get in about 120 s and
about 180 s of warm training? Four levers: (a) seeds from dense points, (b) coarse-to-fine resolution, (c) H100 against A100,
(d) two GPUs. Plus the question from the user: can one GPU be made faster?

Everything runs on Modal as ephemeral apps (retries 0, explicit timeouts). Probe: `fast_splat.py`, next to this file. Outputs:
`research-notes/phase2/runs/m3-fu-e5b-splat-*` (`result.json`, `launch.json`, held-out real | render PNGs).

**Scoring.** Scoring matches `runs/me340-splat-232`, which delivered 30.82 dB:
- the 86 held-out walk-shot frames (every 8th frame; the cut-away 14-225 is skipped);
- 1280x720, with the presenter masks (dilated 15 px) and the caption box left out;
- the exported splat file, on pose-corrected cameras.

**Training setup.** Same loss, pose refinement, MCMC densification and elongation cap 4 as `splat_train.py`. Cap 500k
Gaussians unless stated. `--seconds` is a wall-clock budget: learning-rate decay, densification stop (at 5/6) and the
resolution schedule all follow elapsed time, so every GPU gets the same time, not the same step count.

**Labels.** M = measured here, E = estimate. Warm training time is the timed loop only. Decoding, evaluation and
container start are listed separately.

```
python modal_apps/m3_e5b/fast_splat.py --prep            # once, local CPU (7 s): inputs cache runs/m3-fu-e5b-splat-inputs/inputs.pkl
python modal_apps/m3_e5b/fast_splat.py --bench --gpu H100 --init da3 --output RUNS/m3-fu-e5b-splat-bench-h100-v3
python modal_apps/m3_e5b/fast_splat.py --gpu H100 --init da3 --voxel .02 --seconds 120 --output RUNS/m3-fu-e5b-splat-h100-da3v2-120
python modal_apps/m3_e5b/fast_splat.py --gpu H100 --gpus 2 --init da3 --voxel .02 --seconds 180 --output RUNS/m3-fu-e5b-splat-h100x2-da3v2-180
python modal_apps/m3_e5b/fast_splat.py --self-check
```

Other flags:

| Flag | What it does |
|---|---|
| `--schedule 4:.25,2:.5,1:1 --small-tiles` | coarse-to-fine |
| `--selective-adam` | Adam updates only the Gaussians the current view sees |
| `--batch 2` | two views per step on one GPU |
| `--slow` | `splat_train.py`'s own step |
| `--cap`, `--init surface\|da3\|lingbot` | Gaussian cap and seed source |

## 1. Why round 1 was slow, and what one GPU can do (M)

Time per step with the Gaussian count fixed: 604,543 Gaussians (DA3 seeds, 1 cm voxel), 1280x720, pose refinement on. Mean
of steps 60-360 (`bench-*`).

| Step variant | H100 ms/step | A100 80GB PCIe ms/step |
|---|---|---|
| `splat_train.py`'s step | 11.6-11.8 | **36.5** |
| pose without host syncs + separable SSIM (same numbers, checked every call) | 7.8 | 11.5 |
| ... + gsplat packed mode | 8.4 | 12.6 |
| ... + SelectiveAdam (updates only Gaussians the view sees) | 6.8 | 10.1 |
| ... without pose refinement | 7.1 | 10.7 |
| ... at 1/2 size, 16 px tiles / 8 px tiles | 9.2 / 7.0-7.4 | 12.4 / 9.9 |
| ... at 1/4 size, 16 px tiles / 4 px tiles | 15.5 / 7.3-7.5 | 20.2 / 9.6 |
| ... 2 views per step (per step / per view) | 12.2 / 6.1 | - |
| ... 4 views per step (per step / per view) | 22.0 / 5.5 | - |

- Round 1's step stalled the CPU twice. The pose correction's boolean-mask mean and `torch.linalg.matrix_exp` each wait for
  the GPU every step. On the A100 PCIe host that tripled the step time: 36.5 ms against 11.5 ms without the stalls.
  - `fast_splat.py` replaces them with a weighted mean and Rodrigues' formula.
  - It also blurs SSIM with two 1-D passes instead of an 11x11 window.
  - Both are checked against `splat_train.py` at the start of every call: they agree to within 2e-6 (camera matrix) and 1e-5 (SSIM).
  - This is the one-GPU speed-up: **1.5x on H100, 3.2x on A100**, with no quality trade.
- Smaller images do not make a step cheaper at 0.5-0.6M Gaussians. The time goes into per-Gaussian work, not per-pixel
  work.
  - With gsplat's 16 px tiles, 1/4 size is twice as slow.
  - With tiles scaled down to match, it is only 5% faster.
- Rendering several views in one call makes each view about 25% cheaper, but it gives fewer optimiser steps.

## 2. 120 s and 180 s previews (M)

Warm training time, one run each. A repeat of the H100 DA3 120 s run gave 28.08 against 28.10 dB, so differences under
about 0.1 dB are noise. Snapshots come from inside the same run, at a higher learning rate than a finished run of that
length. Steps count optimiser steps; each 2-GPU step is 2 views.

| Run | GPU | Seeds | Train s | Steps | Held-out PSNR dB | SSIM / LPIPS | Snapshots dB (30/60/90 s) |
|---|---|---|---|---|---|---|---|
| round 1: `splat_train.py`, 5k steps | A100 PCIe | report surfaces | 181.8 | 5,000 | 25.87 | 0.854 / 0.177 | - |
| `splat_train.py`'s step (`--slow`) | H100 | report surfaces | 120.2 | 11,325 | 27.36 | 0.879 / 0.143 | 25.60 / 26.73 / 27.08 |
| fast step | H100 | report surfaces (212k) | 120.2 | 16,525 | 28.11 | 0.891 / 0.128 | 26.38 / 27.21 / 27.75 |
| fast step | H100 | DA3 any-view, 1 cm (605k, cap 605k) | 120.0 | 13,875 | 27.98 | 0.888 / 0.135 | 25.81 / 26.76 / 27.50 |
| fast step | H100 | **DA3 any-view, 2 cm (202k)** | 120.0 | 15,175 | **28.10** (repeat 28.08) | 0.889 / 0.137 | 26.19 / 27.35 / 27.64 |
| fast step | H100 | LingBot map, 1 cm (628k, cap 628k) | 120.2 | 14,500 | 27.85 | 0.889 / 0.133 | 26.04 / 27.16 / 27.52 |
| + coarse-to-fine 1/4, 1/2, full (`4:.25,2:.5,1:1`, small tiles) | H100 | DA3 2 cm | 120.0 | 15,500 | 28.15 | 0.890 / 0.131 | 19.54 / 24.48 / 27.89 |
| + SelectiveAdam | H100 | DA3 2 cm | 120.1 | 13,875 | 27.89 | 0.886 / 0.141 | 25.94 / 27.14 / 27.52 |
| + 2 views per step, one GPU | H100 | DA3 2 cm | 120.1 | 8,425 | 27.84 | 0.887 / 0.143 | 25.91 / 27.13 / 27.51 |
| fast step | A100 PCIe | DA3 2 cm | 120.3 | 10,025 | 27.48 | 0.885 / 0.150 | 25.36 / 26.76 / 27.23 |
| **2 GPUs, data parallel** | 2x H100 | DA3 2 cm | 120.2 | 13,650 | **28.40** | 0.897 / 0.129 | 26.94 / 27.90 / 28.24 |
| 2 GPUs, data parallel | 2x A100 SXM4 | DA3 2 cm | 120.2 | 8,700 | 27.70 | 0.889 / 0.145 | 25.91 / 26.98 / 27.41 |
| fast step, 180 s | H100 | DA3 2 cm | 180.1 | 22,925 | 28.69 | 0.899 / 0.123 | 120 s: 28.21, 150 s: 28.45 |
| fast step, 180 s | A100 PCIe | DA3 2 cm | 180.2 | 14,500 | 27.99 | 0.892 / 0.139 | 120 s: 27.68, 150 s: 27.86 |
| **2 GPUs, 180 s** | 2x H100 | DA3 2 cm | 180.1 | 20,375 | **28.94** | 0.904 / 0.119 | 120 s: 28.59, 150 s: 28.87 |

**(a) Dense seeds.**
- DA3-GIANT any-view points reach the same quality as the report's own surfaces: 28.10 against 28.11 dB. The report's
  surfaces take the slow path (DROID plus fused mesh) to build; DA3's points come from one forward pass.
- Dense seeds give no head start once there are about 10k steps:
  - At 1 cm, DA3 and LingBot seeds number 0.6M. That raises the cap and slows each step, which costs 0.1-0.25 dB.
  - DA3 at 2 cm (202k seeds, grown to 500k) is the pick.
- In round 1, LingBot seeds were 3.4 dB ahead after 500 steps. At 30 s, about 3,500 steps, the lead is gone.

**(b) Coarse-to-fine.**
- 28.15 against 28.10 dB: no measurable gain.
- Small images do not make steps cheaper here (section 1). The early low-resolution phase only reshuffles which steps come
  first.

**(c) H100 against A100.**
- With the fast step, an H100 does 1.5x the steps of an A100 80GB PCIe in the same time, for +0.6-0.7 dB (120 s and 180 s).
- An A100 at 180 s (27.99 dB) still trails an H100 at 120 s (28.10 dB).
- With round 1's step, the gap was 3x, mostly host stalls.

**(d) Two GPUs.** Data parallel works like this:
- each GPU renders its own training view;
- gradients are all-reduced every step (NCCL);
- every 50 steps, rank 0's Gaussians, Adam moments and pose knots are broadcast, and the random stream is reseeded.

Why the sync is there:
- A first run without it drifted apart during densification: some Gaussians differed by 48 native units after 3,000 steps
  (`h100x2-smoke-nosync`).
- With it, the ranks match exactly: drift is 0.0 over the last 24-49 steps of each run.

What two GPUs cost and buy:
- Each step is 10-15% slower (8.8 ms against 7.9 ms on H100), but carries two views.
- At a fixed time, 2x H100 gains +0.30 dB at 120 s and +0.25 dB at 180 s.
- That equals roughly 1.3x the time on one H100: 2x H100 at 120 s (28.40) is about 1x H100 at 150 s (28.45 snapshot).
- 2x A100 SXM4 gained +0.22 dB over 1x A100 PCIe. The hosts also differ.

**The second GPU is better spent on the other stages running at the same time as the splat** (FAST-PATH-PLAN GPU A / GPU B)
than on one splat.

## 2a. Gaussian cap (M, one H100, DA3 2 cm seeds)

| Budget | 500k cap | 1M cap | 2.5M cap |
|---|---|---|---|
| 120 s | 28.10 / 28.08 (15.2k steps) | 28.19 (11.0k steps) | 27.73 (mid-run snapshot of the 1200 s run) |
| 180 s | 28.69 (22.9k steps) | 28.75 (16.3k steps) | **28.97** (11.3k steps) |

At 120 s, more Gaussians barely help. At 180 s, 2.5M wins: one H100 reaches what two H100s reach with 500k (28.94 dB).
2x H100 with a 2.5M cap was not run.

## 3. Best preview reachable

| Budget | Best measured (M) | On the plan's hardware (A100) |
|---|---|---|
| ~120 s warm | 28.40 dB, 2x H100, 500k cap; 28.19 dB, 1x H100, 1M cap | 27.70 dB, 2x A100 SXM4; 27.48 dB, 1x A100 PCIe |
| ~180 s warm | 28.97 dB, 1x H100, 2.5M cap; 28.94 dB, 2x H100, 500k cap | 27.99 dB, 1x A100 PCIe, 500k cap |

The recipe for all rows: fast step, DA3 any-view seeds at 2 cm, pose refinement, elongation cap 4, full size, time-based
schedule, no coarse-to-fine.

Compared with round 1's 25.87 dB in 182 s, that is +2.1 dB on the same A100 PCIe and +3.1 dB on one H100. It is 1.9-2.8 dB
below the delivered 30.82 dB, which continued training recovers (section 4).

## 4. Continued in the background (M)

Run: `h100-da3v2-cap2.5m-1200`. One H100, the same recipe with the delivered run's 2.5M cap, a 1200 s budget.

| Warm training s | 30 | 60 | 120 | 180 | 240 | 300 | 420 | 600 | 900 | **1200 (end of schedule)** |
|---|---|---|---|---|---|---|---|---|---|---|
| Held-out PSNR dB | 25.07 | 26.66 | 27.73 | 28.83 | 29.21 | 29.41 | 30.14 | 30.41 | 30.77 | **31.02** (SSIM 0.933, LPIPS 0.083) |

- **Yes, it reaches the delivered quality.** 31.02 dB after 1200 s: 63,750 steps, 2.5M Gaussians, exported file before
  cleanup.
  - Delivered: 30.82 dB after cleanup, 30.92 dB before.
  - That took 57,737 steps and 3027 s on an A100-SXM4 with `splat_train.py`'s step.
- **How long:** about 20 min of warm training on one H100, against about 50 min for the delivered run.
  - Container start, 25 s of decoding and scoring come on top.
  - The mid-run curve crosses 30.8 dB between 900 and 1200 s. A 900 s schedule would probably end near 30.8 dB (E): a
    finished schedule scores above the same-time snapshot of a longer one.
- A 2.5M cap trails a 500k cap at 120 s: 27.73 dB as a mid-run snapshot, against 28.10 dB for a finished 120 s run. It is
  ahead by 180 s: 28.83 against 28.69 dB. Cap runs for the previews: see section 2a.

## 5. Where the rest of the time goes (M)

| Part | H100 host | A100 host |
|---|---|---|
| Decode 899 frames plus exclusion masks (CPU, 4 cores) | 15-17 s | 15-32 s |
| Frames to GPU | 1.2-2.1 s | 1.3-2.2 s |
| Seeds to Gaussians | 0.04-0.4 s | 0.04-0.06 s |
| Scoring 86 frames plus export (not needed in production) | 5-7 s | 25-35 s |
| Container start + upload (call minus container) | 17-190 s, mostly queueing for GPUs | 20-190 s |

**Cost.** One 120 s H100 run is $0.19 billed (E: $0.25 with 60 s start). One 2x H100 180 s run is about $0.65 (E).

In the fast path these parts change:
- the decode would be shared with DA3's;
- the container would stay warm (`min_containers=1`);
- scoring would go.

**Spend.** About $7 billed (E). Summing each run's own estimate gives $7.9, plus about $1.2 for two failed 2x H100
smoke calls.
- The first hung for about 12 minutes with forked workers, before the switch to spawned workers.
- The second failed at import.
- Modal's hourly report billed a 120 s H100 run at $0.19, against $0.25 estimated.

## What others do ([C], not measured here)

- **InstantSplat** (NVIDIA, UT Austin, Stanford and others): dense MASt3R seeds plus joint pose optimisation, a few seconds for
  sparse views.
  - Here dense seeds help only in the first few hundred steps.
  - On a 600-frame walk the budget goes to steps, not to initialisation.
- **Resolution schedules** (for example DashGaussian): here they bought nothing. gsplat's step at 0.5M Gaussians is dominated
  by per-Gaussian work.
- **Multi-GPU 3DGS** (Grendel, gsplat `distributed=True`) targets scenes too big for one GPU. At 0.5-2.5M Gaussians, one GPU
  holds everything, and data parallel buys about 1.3x.
- **Stanford/Caltech HomeBody** (Sep 2026):
  - A GPT agent builds an Isaac Sim digital twin from iPhone video, a D435i, LiDAR SLAM and the robot's joint poses. Measured
    SLAM geometry constrains the dimensions.
  - It runs on one RTX 4090 laptop. The page gives no reconstruction timing and describes no photoreal splat.
  - Its speed comes from metric sensors, not from a trainer.

## Caveats

- **Validation only.**
  - Validation, not an untouched test: the settings were picked on these held-out frames, as with 232.
  - DA3's TSDF was fused from 112 keyframes that include held-out frames. The report's surfaces were built the same way.
  - The LingBot seeds exclude held-out frames.
- **Cameras.**
  - Training uses DROID cameras, so the score is comparable with 232.
  - The fast path would have DA3 any-view cameras instead. Pose refinement absorbs small errors, but that is not measured
    here: ATE is 4.3 cm against DROID.
- **Run to run.**
  - Time budgets make runs non-deterministic, and host CPUs vary.
  - The SelectiveAdam run's host did 9% fewer steps, and its decode took 24 s against about 16 s.
  - One repeat differed by 0.02 dB.
- **Hardware.** Modal assigned an A100 80GB PCIe for the one-GPU runs and A100-SXM4 for the pair. The CPU was 4 cores per GPU
  throughout; `splat_train.py` uses 2.
