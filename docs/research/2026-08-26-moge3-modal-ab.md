# MoGe-3 on Modal + A/B vs MoGe-2 on fence-smear cases

2026-08-26. MoGe-3 (`Ruicheng/moge-3-vitl`, 370M) stood up on Modal first try;
A/B'd against our production MoGe-2 (Replicate `jasonod888/moge2`, pinned
version from `ehs_spatial/providers/moge.py`) on `incoming/gen-01.png` and
`incoming/real-clean-01.jpeg`.

## Deploy recipe (what worked, first attempt)

App: `modal_apps/moge3_app.py`, deployed as `moge3-inference`
(https://modal.com/apps/wekruit-livekit-agents/main/deployed/moge3-inference).

- Image: `debian_slim(python_version="3.11")` + `apt git build-essential`
  + `pip install torch torchvision` (default PyPI wheels — cu12x with bundled
  Triton; did NOT need the cu121 index pin) + `pip install
  git+https://github.com/microsoft/MoGe.git huggingface_hub
  opencv-python-headless trimesh`.
- The MoGe git package pulls `flex-gemm` from its pinned GitHub commit and
  builds it as a pure-Python wheel in ~seconds — **no FlexGEMM/Triton pain at
  all**. It JIT-compiles Triton kernels at runtime, which is why NVIDIA-only.
- GPU: `L4` worked; A10G fallback never needed.
- Weights: `MoGeModel.from_pretrained("Ruicheng/moge-3-vitl")` in
  `@modal.enter()`, HF cache on a `modal.Volume` (`moge3-hf-cache`,
  `HF_HOME=/cache/huggingface`).
- Gotcha: the locally installed modal CLI 0.74.0 is hard-refused by the server
  ("deprecated"); ran everything via `uv run --with modal` (client 1.5.4).

Latency / cost (L4, $0.80/hr):
| | cold | warm |
|---|---|---|
| image build (one-time) | ~4 min total, MoGe layer 41s | — |
| model load | 25.0s (HF download) | 6.1s (volume cache) |
| infer, fp16, refine_steps=3 | 10.95s @ 1448x1086 | 11.29s @ 2048x1536 |
| per-image GPU cost | ~$0.008 | ~$0.0025–0.004 |

MoGe-2 on Replicate for the same images: 10.3s (gen-01) and 70.3s
(real-clean-01 at full 4032px) predict time, ~$0.01–0.05 total for the eval.

`infer()` returns metric `points` (H,W,3), metric `depth`, per-pixel float
`normal`, validity `mask`, normalized `intrinsics` (fp32 npz + colored PLY);
FOV recovered from intrinsics. gen-01: MoGe-3 82.5° vs MoGe-2 79.6°;
real-clean-01: 74.4° vs 71.5°. Fence median depth agrees to 2mm (5.684 vs
5.686m) — metric scale is consistent between the two, so either can anchor.

## A/B on gen-01 (SAM fence masks from runs/gen-01, canonical 518x392 grid)

Script: `scripts/moge3_ab_analysis.py`, output
`outputs/moge3_eval/ab_metrics.json`. Three scanlines (y=355/502/649) crossing
the fence panels/uprights; transition width = pixels between 10% and 90% of
the local depth step (>=0.5m steps only, 25px window).

| metric (gen-01) | MoGe-2 | MoGe-3 | delta |
|---|---|---|---|
| depth-edge transition width, median (px) | 5.0 | 3.0 | **-40%** |
| depth-edge transition width, mean (px) | 5.76 | 3.41 | **-41%** |
| edges measured | 33 | 34 | — |
| smear fraction >0.5m vs global fence median | 0.746 | 0.769 | ~equal |
| smear fraction >0.5m vs per-instance median | 0.536 | 0.522 | ~equal |
| per-pixel normals | 8-bit PNG only | float32 on-grid | MoGe-3 |

Caveat on the smear fractions: gen-01's "safety_fence" masks are transparent
polycarbonate guarding panels (overlay: `outputs/moge3_eval/fence_mask_overlay.png`).
Most mask pixels legitimately see the background *through* the panel, so
deviation-from-median measures transparency, not model error — it is the same
for both models by construction and not a discriminator on this imagery. The
discriminating number is the edge transition width.

## Verdict

**Yes, measurably, but modestly.** MoGe-3 cuts depth-edge transition width at
fence boundaries by ~40% (5px -> 3px median at 1448px width) with identical
metric scale, and adds float normals for free. It does not change what the
model reports *inside* transparent fence panels — both models see through the
guarding, so panel-interior "smear" is a segmentation/materials problem
(treat SAM fence masks as occupancy, not as depth-trustworthy), not a
depth-model problem. Worth adopting where thin-structure edges matter
(uprights, light-curtain posts); not a fix for transparent-panel geometry.

Ops: self-hosted Modal L4 at ~11s/image is cost-competitive with Replicate
MoGe-2 and 6x faster on 4k imagery (Replicate ran 70s); warm-container
latency is dominated by inference itself.

## Files

- `modal_apps/moge3_app.py` — Modal app (deployed)
- `scripts/moge2_eval_fetch.py` — Replicate MoGe-2 fetch for the A/B
- `scripts/moge3_ab_analysis.py` — metrics
- `outputs/moge3_eval/{gen-01,real-clean-01}/` — MoGe-3 points/depth/normals/PLY/meta
- `outputs/moge3_eval/moge2/{gen-01,real-clean-01}/` — MoGe-2 full outputs (EXR depth etc.)
- `outputs/moge3_eval/ab_metrics.json`, `fence_mask_overlay.png`
