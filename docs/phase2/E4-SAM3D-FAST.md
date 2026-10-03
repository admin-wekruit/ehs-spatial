# E4: SAM 3D Objects fast mode (FAST-PATH-PLAN section 7)

Probe: `modal_apps/e4_sam3d_fast.py`. Results: `research-notes/phase2/runs/m3-exp-e4-sam3d-1/`
(`index.json`, `inputs.json`, `bench.json`, `gate-all.json`, `summary.json`).

## What it tests

- **Objects:** the first 30 objects that `runs/me340-object-models-241-sam3d` modelled, in its priority order (EHS equipment first).
- **Gate:** the unchanged gate. `scripts/complete_video_objects.py` runs read-back only, with 241's arguments: `--generator sam3d --skip-frames 14-225 --entities <one id>`, and no `--invoke`. Only the SAM 3D journal differs between arms.

| Arm | Settings | Tries |
|---|---|---|
| baseline | 241's own journaled meshes (Mac inputs), judged again on Linux | the ones 241 made |
| current | upstream defaults: stage 1 25 steps + CFG 7, stage 2 25 steps + CFG 5 | every try |
| fast | `use_stage1_distillation`, stage 1 4 steps, stage 2 12 steps | every try |
| fast-s2d | fast, plus `use_stage2_distillation` with stage 2 at 4 steps | every try |
| fast-crop | fast, on the source-grid crop and its K instead of the whole frame | attempt 1 |

"Every try" is the runner's own order: up to 4 views, best first, then the best view with seed 43. The runner stops at the first accepted try.

GPU: one A100-80GB container (8 cores, 64 GiB). Each SAM 3D process is a subprocess that loads the model, makes one warm-up call, and waits at a barrier.
- A: 1 process runs every arm, plus a 20-call throughput block.
- B: 2 processes run the same 20 calls.
- C: 2 processes under NVIDIA MPS, same 20 calls.

## Rerun

All steps are ephemeral runs; nothing is deployed. State lives in the Modal volume `panoptes-e4-sam3d`, mounted at the phase2 path.

```
M=/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal
$M run modal_apps/e4_sam3d_fast.py::upload        # 241's inputs for the 30 objects + its SAM 3D journal (~1.2 GB, no compute)
$M run modal_apps/e4_sam3d_fast.py::inputs_step   # CPU x30, ~9 min wall: every try's SAM 3D input; maps 241's meshes to them
$M run modal_apps/e4_sam3d_fast.py::bench_step    # GPU, ~40 min (--smoke: 2 objects, ~5 min)
$M run modal_apps/e4_sam3d_fast.py::gate_step     # CPU x30: gate per object and arm (--only ID --arms a,b for one)
python modal_apps/e4_sam3d_fast.py summary        # summary.json from the local results
python modal_apps/e4_sam3d_fast.py self-check
```

## Caveats

- On Linux, FFmpeg decodes the source frames within ±3 levels of the Mac decode. The SAM 3D journal keys therefore differ from 241's.
- 241's meshes are mapped to the new tries by (observation, seed): 89 of its 92 tries mapped.
- One object (object-026) picked different views on Linux, because the sharpness scores were close.
- The current rerun and baseline differ only in those decode levels and GPU nondeterminism. Together they show the run-to-run noise of the gate.

## Results (2026-09-28)

All numbers are measured (M) unless marked E. GPU: A100-SXM4-80GB.

### Seconds per call

"Warm" is the worker's span around the upstream `run()`, excluding the input load. n = 113 tries per arm (30 for crop).

| Arm | Stage 1 | Stage 2 | Warm s/call, median (p10–p90) | Stage 1 s | Stage 2 s |
|---|---|---|---|---|---|
| current | 25 + CFG | 25 + CFG | 10.52 (10.32–11.19) | 5.25 | 4.98 |
| fast (plan) | shortcut 4 | 12 + CFG | 3.45 (3.36–3.91) | 0.61 | 2.58 |
| fast-s2d | shortcut 4 | shortcut 4 | 1.58 (1.45–1.79) | 0.63 | 0.66 |
| fast-crop | shortcut 4 | 12 + CFG | 3.33 (3.26–3.65), crop 541² vs whole 1280x720 frame | 0.62 | 2.52 |
| s1cfg12 (follow-up) | 12 + CFG | shortcut 4 | 3.49 (3.33–4.04) | 2.58 | 0.61 |

- Mesh decode: 0.13 s. The rest of a call is about 0.1 s.
- A process's first call: 14 s (the decoder's one-off set-up).
- Model load in the process: 58 s (47–50 s each when two processes load together).
- Container start to function entry: 8–36 s (image cached).

### Gate pass rate

Same 30 objects and the unchanged gate (`complete_video_objects.assess`), run on every try.

| Arm | Objects accepted, try 1 | Within 2 tries | Within all tries (runner policy) | Per try |
|---|---|---|---|---|
| baseline (241's meshes) | 4 | 6 | 8 (26.7%) | 8/89 |
| current, rerun | 3 | 5 | 7 (23.3%) | 12/113 (10.6%) |
| fast | 1 | 2 | 4 (13.3%) | 5/113 (4.4%) |
| fast-s2d | 1 | 2 | 4 (13.3%) | 5/113 (4.4%) |
| fast-crop (try 1 only) | 2 | – | – | 2/30 |
| s1cfg12 | 5 | 6 | 7 (23.3%) | 9/113 (8.0%) |

**Noise.** The same settings run twice (241 against the Modal rerun) disagree on 3 of 30 objects and on 4 of 89 paired tries. At n = 30, a difference of up to about 3 objects is within that run-to-run noise.

**The stage-1 shortcut is what fails.** It drops CFG, and the layout gets worse:
- Median silhouette IoU: 0.606 → 0.523.
- "ICP scale at clamp" rejections: 13 → 34.
- Source-view failures: 75 → 103 of 113.
- Paired per try against current: 11 tries pass only under current, 4 only under fast (McNemar p = 0.12).

**Stage-2 steps do not change a single gate decision.** fast and fast-s2d have identical outcomes on all 113 tries; the meshes differ only in detail.

The follow-up keeps stage 1 with CFG at 12 steps and uses the shortcut only in stage 2. It matches current on objects (7 = 7). Per try, 7 pass only under s1cfg12 and 10 only under current (p = 0.63).

### 1 vs 2 processes on one A100-80GB

Measured on the same 20 fast calls. Each process loads its own inputs in the timed span.

| Setup | Wall per call (effective) | Calls/min | GPU memory, peak | GPU utilisation, mean |
|---|---|---|---|---|
| 1 process | 4.18 s (3.40 s compute) | 14.3 | 34.5 GB | 54% |
| 2 processes | 2.97 s | 20.2 | 53.5 GB | 91% |
| 2 processes + MPS | 2.09 s | 28.7 | 53.5 GB | 80% |

MPS works on Modal: `nvidia-cuda-mps-control -d` starts, a server is spawned and both clients attach. Against the single-process compute rate, 2 processes are 1.14x and 2 processes with MPS are 1.63x.

### Gate CPU cost

Two-core container:
- `assess()`: median 23 s per try (p90 38 s).
- `prepare()`: median 46 s per object (p90 102 s); it loads all 312 posed views.

At 3.5 s of GPU per try, one SAM 3D process needs about 7 such containers to keep up [E]. In the fast path, the gate costs more than generation.

### Cost

$4.17 measured: list price x function seconds; the GPU part is local wall time, an upper bound.
- inputs $0.25
- smoke $0.32
- bench $2.31
- gate $0.60
- follow-up bench $0.52
- follow-up gate $0.19

Container start-up for about 90 CPU containers is not included, about +$0.1 [E].
