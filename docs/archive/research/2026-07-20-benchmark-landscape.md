# Public benchmark landscape for the EHS spatial harness

Date 2026-07-20. Three-angle web survey (VLM metric-QA benchmarks /
metric-GT reconstruction datasets / industrial-safety datasets) with an
adversarial spot-check pass on the six load-bearing candidates. Full agent
returns archived in the session workflow transcript.

## Tier 1 — drop-in exams: photo → metric answer, scoreable head-to-head vs VLMs

| Benchmark | GT | Size | License | Verdict |
|---|---|---|---|---|
| **NVIDIA PhysicalAI-Spatial-Intelligence-Warehouse** | synthetic warehouse, per-image distance/count QA | ~19k test QA | **CC-BY-4.0** | **Adopt first: domain (warehouse), task (distance QA), license, and scale all fit.** Synthetic imagery is its one weakness |
| **Q-Spatial-Bench** (arXiv 2409.09788) | ScanNet RGB-D + **real tape-measured** (Q-Spatial++) | 271 QA (~101 tape-measured) | Apache-2.0 | Adopt second: exactly our single-photo→distance task, real photos, tiny but decision-grade — the natural harness-vs-VLM headline |
| SpatialRGPT-Bench | LiDAR/RGB-D 3D boxes | 749 quantitative QA | unstated (HF card empty) | Good pool; resolve license + region-prompt format first |
| CA-1M / CA-VQA (Apple) | FARO laser, mm-grade | 440k boxes / VQA split | CC-BY-**NC-ND** | Best GT quality; internal comparison only, nothing shippable |
| VSI-Bench | ScanNet++/ARKitScenes reconstructions | 5.1k QA | Apache-2.0 | Video input ≠ our pipeline; cite as VLM-failure context; GT noise documented (ReVSI) |
| MMSI / OmniSpatial / SpatialBench(BAAI) | MCQ / qualitative | — | mixed | Not metric — excluded |

## Tier 2 — reconstruction-layer GT (feed OUR pipeline, derive pair distances like V2)

| Dataset | GT | License | Verdict |
|---|---|---|---|
| DIODE | FARO laser ±1 mm | **MIT** | Best license; per-frame depth only (single-frame pipeline slot) |
| ARKitScenes | FARO laser + iPhone captures | Apple custom (NC language) | Imagery matches phone-photo product exactly; license blocker to resolve |
| Hypersim | synthetic exact depth + 3D boxes | usable | Free exact GT at scale; synthetic |
| ScanNet++ v2 | laser ~0.9 mm | gated ToU, NC research | Best real GT; access friction, residential scenes |
| ETH3D | laser (already in V2) | CC-BY-NC-SA | Keep, internal eval only (current status) |

## Tier 3 — industrial domain (entity/rule layers; mostly 2D, no metric)

- **Rohbau3D (MIT)**: real terrestrial-laser construction point clouds w/
  semantics — metric, industrial-adjacent, permissive; candidate for a real
  clearance-style eval.
- **ANavS LiDAR Warehouse (CC-BY-SA)**: metric cuboids incl. forklift —
  forklift-clearance GT derivable.
- **SODA**: fence + person + PPE 2D boxes; survey reports an open direct
  download now (earlier Baidu-only block — recheck).
- SH17 (manufacturing PPE/person, NC), MOCS (13-class instance seg,
  license unconfirmed), ConstructionSite10k (safety-rule-violation VQA —
  closest published analog to our rule-eval concept, CC-BY-NC), SHEL5K
  (CC-BY PPE).

## Recommended adoption order

1. **NVIDIA warehouse spatial QA** — external headline number in-domain,
   statistical power (19k), clean license.
2. **Q-Spatial-Bench (Q-Spatial++)** — real-photo tape-measure GT
   head-to-head vs VLM; replaces our n=9 in the pitch.
3. ARKitScenes or DIODE for the reconstruction layer once license/fit is
   resolved; Rohbau3D for an industrial metric scene test.

Verification notes: all Tier-1 rows confirmed to exist via fetch of their
pages; NVIDIA "WorldModel-Synthetic-Warehouse-Operations-Scenes" (18+ TiB
video training set) is NOT the eval — the sibling
"PhysicalAI-Spatial-Intelligence-Warehouse" is.
