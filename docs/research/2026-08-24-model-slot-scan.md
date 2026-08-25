# Model-slot scan — what to add, what to skip, what Open3D is doing

Date 2026-08-24. Three parallel surveys (NVIDIA ecosystem / open-weight
per-slot candidates / Open3D + geometry-library alternatives) plus an
adversarial verification pass on the load-bearing claims. 57 candidates
assessed; the overwhelming majority are SKIP.

## Verdict summary

| Slot | Incumbent | Change | Why |
|---|---|---|---|
| Geometry (multi-view metric) | MapAnything (Replicate API) | **ADOPT self-hosted Apache weights** | Nothing surveyed beats it at multi-view metric under a commercial licence |
| Geometry (mono cross-check) | — | **ADOPT MoGe-2** (MIT) | 326M / ~60 ms per image, independent per-view metric opinion — a second read on the scale anchor |
| Segmentation | SAM 3.1 (fal API) | **ADOPT self-hosted weights** | Same model, no per-call cost — **and the 32-mask cap is fal's response limit, not the model's** |
| Detection (dense small objects) | — | **ADOPT LLMDet** (Apache, ungated) | Box pre-pass feeding box prompts into SAM 3.1 when text-prompt recall thins out |
| VLM explainer | Gemini 3.5 Flash | **ADOPT Qwen3-VL-8B-Instruct** as the V4 swap target | Apache 2.0, ungated, ~16 GB bf16 on one 24 GB card, mature serving |
| Geometry/render library | Open3D 0.19.0 | **keep the pin** | Still the head of the released line |

Everything else — Cosmos 3, Cosmos-Predict/Transfer, Nemotron Nano VL,
C-RADIO backbones, Isaac Sim/nvblox, FoundationStereo, PyTorch3D, Kaolin,
PDAL, CloudCompare, pyrender, PyVista, NKSR — SKIP, with reasons recorded
in the workflow transcript.

## Open3D: no change, and that is the finding

- **Latest tagged release is still v0.19.0 (2025-01-08)** — verified against
  GitHub tags/releases and PyPI. No v0.20 tag, no public date, 19+ months.
- The project is active on `main`; the notable unreleased change is an
  **auto-EGL headless path**, which would make our
  `Visualizer.create_window(visible=False)` render work on a display-less
  Linux server without a rebuild.
- Decision: **keep the 0.19.0 pin.** An untagged branch has no place in a
  compliance harness whose renders are audit evidence. Revisit when 0.20
  tags, or earlier if headless renders move to a Linux server *and* the
  GLFW hidden-window path fails there.
- Alternatives all SKIP: PyTorch3D (we have no gradients), Kaolin (CUDA
  only, dead on the macOS dev box), PDAL (survey-scale LAS, wrong shape),
  CloudCompare (GPL, GUI-shaped), pyrender (**unmaintained, zero commits
  in 365 days**), PyVista/VTK (lateral move, same window-server
  constraint), NKSR (non-commercial licence).

## NVIDIA: one real fit, one to remember

- **Cosmos-Reason2 (2B/8B)** — EVALUATE for the VLM-explainer slot, and only
  if workcell photos cannot leave the building. NVIDIA Open Model License is
  commercially usable but carries NVIDIA's own derivative terms; legal must
  read it rather than filing it under "open like Apache". Gemini is cheaper
  and better today, and the VLM never touches the numbers.
- **Fast-FoundationStereo** — SKIP now (needs a rectified stereo pair), but
  this is *the* model to revisit if the capture story ever becomes a fixed
  two-camera jig: 14.6M params, ~650 MB peak, commercially licensed, true
  metric depth **with no scale anchor needed**.
- **PhysicalAI-Spatial-Intelligence-Warehouse** (already adopted as a
  benchmark) has **the cleanest licence of our four benchmarks: CC BY-4.0**,
  beating SpatialRGPT-Bench (undeclared) and ETH3D (CC BY-NC-SA) for any
  external or commercial claim.
- Isaac Sim / SimReady warehouse assets: only relevant if we start
  generating synthetic EHS scenes, which the eval plan does not need.

## Corrections the verification pass forced

1. **The V4 "MapAnything → Depth Anything 3 model-swap invariance" plan does
   not hold as written**: no Apache-licensed DA3 checkpoint produces
   multi-view metric geometry. DA3METRIC-LARGE is usable only as a *mono*
   metric cross-check, not as a geometry-slot replacement. The model-swap
   test must be re-specified.
2. **UniDepth v2 is CC BY-NC 4.0** — the unlicensed HF mirror makes it look
   adoptable. Research baseline only; never in the product path.
3. **SAM 3.1's licence needs procurement sign-off** before it lands on
   customer hardware: genuinely commercial-permissive, but §8 allows
   unilateral amendment, weights are manually gated, and redistribution
   terms are copyleft-flavoured.

## Recommended sequence

1. Self-host SAM 3.1 + MapAnything Apache weights on one 24 GB GPU; re-run
   the six-layer exam suite; scores must not regress. This also removes the
   fal mask cap that bounds dense-small-object recall.
2. Add MoGe-2 as a per-view scale cross-check — cheapest possible second
   opinion on the one thing the boundary map named as the gap.
3. Swap Gemini → Qwen3-VL-8B on the explanation path only.
4. LLMDet box pre-pass only if SAM 3.1's text head still under-recalls dense
   pallets once the API cap is gone.
