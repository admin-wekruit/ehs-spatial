# Architecture landscape vs. the EHS eng harness — 2026-07-20

Five-track web research (π/VLA, spatial VLMs, 3D geometry models, segmentation,
compliance systems & eval methodology), run to answer: do mid-2026 architectures
change the harness bet, and what must harness validation measure. Every claim below
carries a source in the underlying research output; confidence flags preserved there.

## Headline

**The harness bet survives 2026 intact and strengthened.** No model family — π0.x
VLA, frontier spatial VLMs, Gemini Robotics-ER — ships calibrated metres-from-photos.
Every published method that genuinely improves VLM metric answers does it by injecting
explicit 3D reconstruction (VLM-3R, Ego3D-VLM, SpatialRGPT) — i.e. they converge on
our architecture. The durable, un-commoditized layer is exactly what we built:
deterministic metric reconstruction + auditable rule engine + eval harness.

## Track verdicts

### π0.x / VLA — irrelevant (not component, not competitor)
- π0.7 released 2026-04-16 (steerable compositional VLA): outputs robot actions, no
  metric scene output, **closed weights** (openpi still ships only π0/π0-FAST/π0.5,
  Apache). π0.6 also closed (Gemma3-4B backbone + 860M action expert).
- GR00T N1.7 (weights: NVIDIA Open Model License, not Apache), Figure Helix:
  action policies, same verdict. Watch: GR00T N2 "world action model" end-2026.
- One adjacent capability: Gemini Robotics-ER 1.6 pointing/gauge-reading — pixel-space
  (0–1000 normalized), API-only, no weights → fails on-prem-capable; optional eval
  baseline row only.

### Spatial VLMs — 2025 consensus strengthened; explanation-slot candidates found
- Q-Spatial-Bench: "success" = within 0.5×–2× of truth; GPT-4o 65% even at that
  tolerance. ViewDiag (arXiv 2606.02742): VLM distance answers are prior-driven
  collapse — consistent across viewpoints, insensitive to actual geometry. ReVSI
  (arXiv 2604.24300): earlier spatial benchmark scores partly artifact-inflated.
- Nothing open or closed does cm–dm metric measurement at compliance reliability.
- **Gemini 3.5 Flash swap shortlist (on-prem capable, license-clean):**
  Qwen3-VL-8B/-32B (Apache), Molmo 2 8B / Molmo2-O-7B (Apache; pointing = pixel
  evidence alongside fact_ids), InternVL3.5 (Apache), GLM-4.5V (MIT).
  Exclude SpatialLM1.1 (CC-BY-NC).

### Geometry slot — MapAnything Apache checkpoint stays incumbent
- Only clean-license native-metric multi-view model as of 2026-07. Use
  `facebook/map-anything-apache` for the self-host path (the better 13-dataset
  checkpoint is CC-BY-NC — do not use commercially).
- Images-only metric scale is mediocre (~16% abs-rel pointmap at 50 views; ~32%
  metric-depth abs-rel ScanNet) → **camera-height scale calibration stays primary**
  (~1–3% class); model scale is a cross-check, not the source of truth.
- Cheap accuracy win: pass phone EXIF intrinsics into MapAnything (paper shows large
  scale-error reduction). Independent cross-check: MoGe-2 (MIT, monocular metric).
- Swap candidates for V4 invariance testing: Depth Anything 3 Apache tier
  (DA3-BASE + DA3METRIC-LARGE; avoid NC Giant/Nested), VGGT-1B-Commercial
  (conditional: gated, military exclusion, up-to-scale). License-dead: MASt3R/DUSt3R,
  CUT3R, Spann3R, Pi3.

### Segmentation slot — restructure as box-then-mask; fence failure partly explainable
- SAM 3/3.1 self-hostable: gated weights, custom Meta SAM License (commercial OK, no
  MAU caps; military/ITAR bans; copyleft-style redistribution) — usable, not Apache.
- Our fence zero-detection matches documented failure modes, with evidence-backed
  general mitigations (NOT synthetic-specific hacks):
  1. fal `detection_threshold` default ~0.5; docs recommend 0.2–0.3 when text prompts
     return nothing (SAM3 presence head calibrated at 0.5).
  2. SAM3 is trained on short noun phrases — "safety fence" is off-distribution;
     ensemble "fence"/"barrier"/"railing"/"guardrail".
  3. SAHI tiled inference (MIT): +5–14 AP small/thin objects, no retraining.
  4. SAM-family thin/wiry-structure weakness is published (arXiv 2412.04243, HQ-SAM);
     fence segmentation is a standalone research problem (de-fencing literature).
- **On-prem candidates:** (1) LLMDet (Apache, best open rare-class recall) boxes →
  SAM-HQ (Apache, thin-structure specialist) masks + SAHI tiling; (2) MM-Grounding-DINO
  (Apache) + SAM 2.1/HQ-SAM; (3) self-hosted SAM 3.1 (SAM License); (4) OWLv2+SAM-HQ
  (pure Apache, built-in prompt ensembling). Hard-exclude: Grounding DINO 1.5/2,
  DINO-X, T-Rex2 (API-only), Rex-Omni (non-commercial), Ultralytics YOLO-World (AGPL).

### Compliance systems & eval methodology — architecture validated; eval is the moat
- Detection-guided consensus holds: arXiv 2604.05210 (YOLO boxes into sVLM prompts:
  hazard F1 34.5%→50.6%); MonitorVLM lineage itself detection-augmented; NVIDIA VSS 3.x
  uses deterministic analytics for events, VLM only for alert verification. No credible
  shipped end-to-end VLM compliance judge found.
- EHS vendors (Voxel/Intenseye/Protex) publish marketing numbers, zero methodology, no
  FPR, no third-party validation → publishing a real eval is differentiation.
- Eval prior art to copy: VDI/VDE 2634 artifact-based 3D metrology acceptance tests;
  tape/total-station ground truth; per-distance-band MAE + FNR; per-rule recall at
  fixed FPR; INSUFFICIENT_EVIDENCE rate as first-class metric.
- fact_id grounding = the 2026 standard stack's "deterministic citation check first";
  EU AI Act workplace obligations (Art. 26 lifetime event logging, 6-month retention)
  apply from Aug 2026 → the fact ledger is a compliance feature, package it.
- Open ground, low competition: ISO 13857 reach tables + ISO 13855 D=K(T+C) as
  versioned rules-as-code — nobody has published this.

## Proposed harness validation ladder (the focus going forward)

| Tier | Claim proved | Status |
|---|---|---|
| V0 | Deterministic geometry+rule path correct (oracle masks/pointmaps) | ✅ done (offline eval 4/4) |
| V1 | Paid provider chain works end-to-end on one case | blocked on Replicate credit |
| V2 | **Metric accuracy on real photos vs tape-measured ground truth** (per-distance-band MAE; "verdicts valid to ±X cm" statement; EXIF-intrinsics mode) | missing — highest value |
| V3 | Thin-structure (fence) recall on real imagery: threshold sweep × prompt ensemble × tiling delta × per-view recall | designed (review doc) |
| V4 | Model-swap invariance = on-prem capable proof (MapAnything→DA3; fal SAM→LLMDet+SAM-HQ; Gemini→Qwen3-VL) same eval, same rule engine | not started |
| V5 | Rule engine generalizes: rule #2 = ISO 13857/13855 rules-as-code | not started |
| V+ | End-to-end VLM baseline row (prove harness beats VLM judgment; ViewDiag-style prior-collapse probe; tolerance-band sweep ±5cm/±10cm/±25%/2×) | optional, sales-grade evidence |

## Per-site object variability (owner question 2026-07-20)

Every EHS workshop has different objects. The harness absorbs this by design — with
two concrete gaps to close:

1. **Rules key on roles, not objects.** `fence_clearance` binds "one fence-role
   entity" and "nearest movable-role entity"; the 5 demo movables are just the current
   role membership list. New site with gas cylinders/bins → edit the site's entity
   vocabulary, rule logic untouched.
2. **Segmentation slot is promptable/open-vocab** — that is exactly why SAM3-class /
   LLMDet-class models sit in that slot. Per-site prompt list should be *derived from
   the site's rule vocabulary*. **Gap A:** `PROMPT_VOCABULARY` is a hardcoded 8-label
   constant in `providers/sam3.py` — must become per-site config (schedule with V5
   rule-engine generalization).
3. **Unknown objects fail loud, not silent.** Undetected/below-evidence entities →
   INSUFFICIENT_EVIDENCE + human review, never silent PASS (asymmetric-failure
   design already in the rule gates).
4. **Long-term:** per-site LoRA fine-tune of the detector from a few hundred reviewed
   images (original handoff Phase 2 pattern) once HITL exists.
5. **Gap B (eval):** V3/V4 gain an *unseen-vocabulary generalization* dimension —
   swap the label set, re-measure recall, proving the harness is not welded to the 5
   demo objects.

## V2 sourcing decision (owner 2026-07-20)

Dual-track: public dataset first, owner will additionally shoot + tape-measure a real
scene later. Owner also approved Gemini→open-VLM swap scheduled at V4.

Dataset research verdict (sourced; licenses verified from primary pages):
- **Primary: Redwood Indoor LiDAR-RGBD** (Apartment/Bedroom/Boardroom/Lobby/Loft) —
  **public domain**, the only license clean for commercial-citable benchmarking.
  Laser-scanner GT model per scene; sample 2–8 frames from RGB video; inter-object
  distances measured on the laser PLY (CloudCompare/Open3D). Caveat: VGA-class RGB
  (worst-case realistic input); camera height via floor-plane fit, poses estimated.
- **Secondary: ETH3D indoor** (delivery_area/kicker/office/pipes) — mm-accurate Faro
  laser GT registered to COLMAP poses → exact camera height available; 24MP stills.
  **CC BY-NC-SA 4.0** — internal-benchmark use is a legal gray zone; flag before any
  result feeds a commercial claim.
- Shelf: ARKitScenes (Apple custom license, commercial arguably OK <700M-MAU clause,
  needs legal), Hypersim (CC BY-SA, synthetic stress set). Research-only, rejected for
  product-facing numbers: ScanNet++/ScanNet, Replica. No-GT rejects: RealEstate10K,
  SUN RGB-D. Details + URLs in session agent report.

Owner capture protocol (for their own scene later): 4 photos from 4 angles at
measured camera height (note it), keep EXIF intact, tape-measure 3–5 floor distances
between distinct floor-standing objects (log which pairs), avoid moving anything
between shots.

Full sourced findings: workflow output `wf_be4489cb-559` (journal in session transcript
dir); key sources include pi.website/blog/pi07, github.com/Physical-Intelligence/openpi,
arXiv 2606.02742, 2604.24300, 2509.13414 (MapAnything), 2604.05210, 2412.04243,
github.com/facebookresearch/sam3 LICENSE, NVIDIA VSS 3.2 docs, artificialintelligenceact.eu/article/26.
