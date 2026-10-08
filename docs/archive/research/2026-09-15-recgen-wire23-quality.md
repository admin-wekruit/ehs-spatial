# RecGen #23 follow-up quality audit — 2026-09-15

**Outcome: runtime succeeded; shape and native scale rejected.** This is an independent CPU-only follow-up for entity `3a875626-ce97-5f04-a3d0-24f207ce6243`, the wire cable tray beside the entrance post. The prior eight-object audit and its five frozen evidence hashes remain unchanged.

The frozen source intentionally contains **7,201 partial positive wire-core pixels**, with no complete-object mask. **4,712** have positive supplied depth, but the existing 5×5 erosion retains only **5** depth-supported pixels. Source widths, many junctions, hidden intervals and the complete wire assembly remain unassigned. A core-mask overlap score cannot establish full-object reconstruction.

The generated object has **6,172 vertices / 12,344 triangles**, finite preserved vertex colors and verified input/output hashes. Native composition exactly matches the anchor camera and official pose: maximum posed-GLB discrepancy is **5.959665427823779e-8** (existing tolerance `2e-5`); native TRS round-trip error is **4.484324023223962e-11**. Coordinate conversion passes, but the provider's geometry/scale does not.

| Independent check | Result |
|---|---|
| Mesh structure | One narrow rectangular wire loop; no repeated transverse rungs/grid |
| Topology | 6,172 used vertices, 18,516 unique edges, 12,344 faces; every edge has two incident faces; Euler characteristic 0, consistent with one aperture |
| Native extents | `[0.00145106, 0.00167838, 0.00108677]` in uncalibrated source units |
| Original-photo projected bounds | x `2777.746–2778.637`, y `637.724–641.408`; only about **0.89×3.68 pixels** |
| Partial source-core bounds | x `2711–2864`, y `521–3230` |
| Exact original-pixel strip | x `2680–2880`, y `490–3300`; all 7,201 labeled cores included, **zero covered core pixels** and zero raster samples of the tiny posed mesh |
| Shape review | Does not reconstruct the visible wire tray; cannot be accepted as complete or delivery quality |

Full original-photo strip panels retain actual unmasked image context, not an invented complete-object silhouette. Object-local front/back/side images inspect unchanged mesh topology at an explicitly separate display scale. The broad partial-core depth distribution and erosion support are under a separate source-input audit; this report does not assert a proven inference-internal cause or modify input masks to pass a gate.

At **2026-09-16 03:15:53 UTC**, read-only model-call accounting showed **9 successes + 1 prior outcome_unknown + 1 newly reserved correction = $11 reserved**, plus the root's $2 overhead allowance under the authorized $20 total. The earlier $0.43748322 app-hour report covers 02:00–03:00 UTC and does not include this later call. Per-call actual and full-task actual costs remain unknown; no budget reservation was released.

Private evidence is separate under `.platform/model-correspondence/model-completion/output-quality/followup-23/`:

| Evidence | SHA-256 |
|---|---|
| `summary.json` | `c412e316570fa68e7016e763620813f8b8e385caed0e7c71f544ea08f44a347a` |
| `full-source-comparison.json` | `c2d5c8673321d48b5850d63bf270782ed360758a30248829c0bf201c5c5eb5e1` |
| `topology.json` | `5c0688d375d0fb267c8bdfb57bb796a98e59de3bca704a61ddf4d0a30ce2d57d` |
| `cost-snapshot.json` | `374aee7cc93103fb9ea8dbc82b74bd9596c1a739f58afc8128aef2a4cf08a771` |
| `parent-evidence-unchanged.json` | `d527df17efc6fdc812a9320007785d100eb48e0c1c0e3725ad3bb5c7ddd44a57` |
| `audit.py` | `05dcf232d42697d2e3f689a4fbc44fc6c95aff067469ab5bda924719b7904ca0` |
| `full_source_review.py` | `de4fb10ec46c8bd9e4b8f178a66dc11b3597da340eee6f55d8d913d09f742ddd` |

Visual evidence: `object-topology.png`, `original-strip-0.png` through `original-strip-3.png`, and `object-23-view-0.png`.

Reproduce numeric evidence locally (CPU and read-only asset lookup):
```sh
PYTHONPATH=. .venv/bin/python .platform/model-correspondence/model-completion/output-quality/followup-23/audit.py 23
PYTHONPATH=. .venv/bin/python .platform/model-correspondence/model-completion/output-quality/followup-23/full_source_review.py
```
