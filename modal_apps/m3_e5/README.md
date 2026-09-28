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

`splat_train.py` gained one flag, `--gpu`, to pin a GPU type instead of its H100-first fallback list. The delivered splat it is
compared against is `runs/me340-splat-232` (60k steps, 2.5M cap, pose refinement, elongation cap 4, cleanup): held-out 30.82 dB on
the same 86 frames and masks.

Self-checks (no GPU): `python modal_apps/m3_e5/da3_splat.py --self-check`, `python modal_apps/m3_e5/vllm_events.py --self-check`,
`python modal_apps/splat_train.py --self-check`.
