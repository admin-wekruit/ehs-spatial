# M3 fast path, experiment E5: splats, events, cold start

Probes for FAST-PATH-PLAN.md section 7, row E5, on ME340. All GPU work runs on Modal as ephemeral `modal run`-style apps
(`app.run()`), retries 0, explicit timeouts; outputs go to new run folders under
`research-notes/phase2/runs/m3-exp-e5-splats-events-cold-*`. Python: `/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python`.
Pipe logs through `grep -v -i "capabilit\|token\|secret"`.

| Part | Command | Output |
|---|---|---|
| (a) gsplat 5k / 7k steps, 0.5M cap, A100-80GB | `python modal_apps/splat_train.py --output RUNS/m3-exp-e5-splats-events-cold-gsplat-5000 --steps 5000 --cap 500000 --pose --max-elongation 4 --skip 14-225 --gpu A100-80GB --max-minutes 14` (and `--steps 7000`) | `train.json`, `splats.*`, held-out PNGs |
| (b) feed-forward splat: DA3-GIANT-1.1 Gaussian head (CC BY-NC, demo only) | `python modal_apps/m3_e5/da3_splat.py --output RUNS/m3-exp-e5-splats-events-cold-da3-gs --views 100` | `result.json`, `*-heldout-*.jpg` (real, render) |
| (c) Qwen3-VL-8B events through vLLM vs transformers | `python modal_apps/m3_e5/vllm_events.py --output RUNS/m3-exp-e5-splats-events-cold-vllm-events` | `summary.json`, `events-*.json`, `raw.json` |
| (d) cold start, one container, 2x A100-80GB, four model families | `python modal_apps/m3_e5/cold_start.py --runs 2 --output RUNS/m3-exp-e5-splats-events-cold-coldstart` | `result.json` |

`splat_train.py` gained two flags: `--gpu` pins a GPU type instead of its H100-first fallback list, and `--train-scale N` trains on
1/N of the frame's width and height (held-out scoring stays full size). Also run: `--train-scale 2`, and without `--pose`. The delivered splat it is
compared against is `runs/me340-splat-232` (60k steps, 2.5M cap, pose refinement, elongation cap 4, cleanup): held-out 30.82 dB on
the same 86 frames and masks.

Self-checks (no GPU): `python modal_apps/m3_e5/da3_splat.py --self-check`, `python modal_apps/m3_e5/vllm_events.py --self-check`,
`python modal_apps/splat_train.py --self-check`.

## Results on ME340 (2026-09-28)

M = measured here, E = estimate. Held-out = the 86 walk-shot frames splat_train holds out (every 8th), presenter and caption
excluded, scored at 1280x720. Train seconds are warm GPU time inside the container (after loading); container start, upload and
evaluation are separate.

**(a) gsplat, 0.5M cap, elongation cap 4, cut-away skipped**

| Run | GPU (Modal assigned) | Train s (M) | Held-out PSNR dB (M) | vs 30.82 |
|---|---|---|---|---|
| 5k, pose refinement | A100 80GB PCIe | 181.8 | 25.87 (refined cameras), 23.70 (report cameras) | -4.95 |
| 7k, pose refinement | A100 80GB PCIe | 261.6 | 26.63 / 23.57 | -4.19 |
| 5k, no pose refinement | A100 80GB PCIe | 159.1 | 25.30 (report cameras) | -5.52 |
| 5k, pose, trained at half size (`--train-scale 2`) | A100-SXM4-80GB | 81.1 | 23.77 / 22.43 | -7.05 |
| 5k, no pose, half size | A100-SXM4-80GB | 56.3 | 23.42 | -7.40 |

Per step (M): full size 0.036 s with pose, 0.030-0.031 s without (PCIe); half size 0.016 s with pose, 0.011 s without (SXM4).

**(b) DA3-GIANT-1.1 Gaussian head (feed-forward), held-out PSNR dB at 1280x720 (M)**

| Views | Forward incl. GS head (M) | Peak GB | all Gaussians | static only | static, 8 nearest views |
|---|---|---|---|---|---|
| 100 posed (DROID) | 5.8 s | 23.2 | 15.01 | 16.51 | 14.90 |
| 100 unposed | 5.7 s | 23.4 | 14.51 | 15.54 | 12.55 |
| 100 unposed + 86 held-out as inputs (own cameras) | 14.9 s | 37.5 | 16.26 | 17.16 | 12.62 |
| 32 posed | 1.3 s | 12.0 | 16.09 | 16.70 | 15.38 |
| 32 unposed | 1.2 s | 12.2 | 14.99 | 15.32 | 13.48 |

Every view's per-pixel Gaussians land in slightly different places, so the union streaks and ghosts (see `*-static-00400.jpg`).

**(c) Qwen3-VL-8B events, 2 windows (7,678 + 11,431 prompt tokens), greedy**

| Path | GPU | Load s (M) | Generate s (M) | Agreement with runs/me340-events-197 |
|---|---|---|---|---|
| vLLM 0.11.0, both windows batched, 0.45 of the card | A100 80GB PCIe | 144.5 (compile + CUDA graphs) | 4.44 warm (11.68 first batch) | 4/4 events matched (tIoU 1.0), actor, PPE and safety note identical; caption words 0.62 / 0.93 Jaccard |
| transformers (video_events.py), one window at a time | A100-SXM4-80GB | 15.8 | 23.2 | identical text |
| reference 197 | A100-SXM4-40GB | 10.7 | 20.5 | - |
| vLLM 0.11.0, `--eager` (no compile, no CUDA graphs) | A100-SXM4-80GB | 39.0 | 4.63 warm (10.62 first batch) | 4/4 matched, fields identical; caption words 0.62 / 1.0 |

vLLM's two batches in one process did not return identical text (`batches_identical: false`): greedy vLLM is not bit-reproducible.

**(d) cold start, one container with 2x A100-SXM4-80GB, four loader processes started together (M)**

| Run | Submit to function start | Function start to all loaded / warm | DA3 warm | SAM 3 warm | SAM 3D loaded / after 1 call | Qwen3-VL (vLLM) warm |
|---|---|---|---|---|---|---|
| 1, new image | 630.6 s (queued for 2x A100-80GB, plus first image pull) | 188.0 / 188.3 s | 42.0 | 31.9 | 79.2 / 94.5 | 188.3 |
| 2 | 367.3 s (queued) | 196.4 / 196.6 s | 55.2 | 44.6 | 123.0 / 137.8 | 196.6 |
| 3, vLLM `enforce_eager` | 7.8 s (capacity free, image cached) | 87.3 / 88.6 s | 35.9 | 28.4 | 73.0 / 88.1 | 88.6 |

Seconds are from the function's start unless stated. Imports alone take 20-40 s per venv process (torch, vLLM). Modal's
"waiting to be scheduled on a GPU_A100_80GB worker" message explains the 6-10 min waits: a demo needs `min_containers=1`.
