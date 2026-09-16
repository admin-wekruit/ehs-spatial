# Corrected fence input/output follow-up — 2026-09-15

**Both corrected provider calls succeeded, but both new shapes/native placements are rejected.** This bounded CPU audit covers corrected #12 observation revision 2 and #28 observation revision 3. It does not modify source/scene data, promote a model, enqueue a call, or overwrite the previous eight-object or #23 evidence.

Inputs are explicitly `partial_visible_metal_only`, not complete-object masks. The corrected masks remove background/apertures and adjacent material; no completeness claim follows from matching these partial positives. Both retain substantial 5×5-eroded depth support (#12: 33,176 pixels; #28: 19,057), unlike #23's five surviving core pixels.

Both outputs retain verified frozen input/output hashes, finite positions/colors, valid indices and unchanged provider topology/colors. Native composition and official posed GLB checks pass, so the failed shape/placement is not a post-transform mismatch.

| Object | Vertices / triangles | Official posed GLB max residual | Native TRS round-trip max error | Visual result |
|---|---:|---:|---:|---|
| #12 | 10,708 / 21,408 | 1.19178e-7 | 8.21894e-8 | Narrow upright with sparse/disconnected crosspieces; much of the prior grid structure is absent; native placement misses the visible frame |
| #28 | 9,508 / 19,142 | 2.38356e-7 | 6.88923e-8 | Narrow ladder/chain-like strip placed diagonally across the scene, not the source's vertical fence panel |

The prior and corrected meshes were independently rendered with the **same original camera/crop**, at 650×650 with pixel-center mapping. The comparison uses the **new partial metal mask for both meshes** and excludes black crop padding from metric denominators. Original full-photo context remains visible beside both actual meshes.

| Object | Partial metal coverage, previous → corrected | Partial-metal IoU inside original photo, previous → corrected |
|---|---:|---:|
| #12 | 44.60% → **0.93%** | 0.1096 → **0.0044** |
| #28 | 93.62% → **6.43%** | 0.0569 → **0.0324** |

#28's old 93.6% partial coverage was caused by an oversized opaque board: its low precision and missing apertures made it wrong despite high coverage. The corrected thin strip is also wrong. #12's earlier panel-like candidate had ownership/alignment limitations; the corrected sparse fragment does not repair them. Neither new candidate should replace a model or be counted as delivery quality.

Accounting snapshot **2026-09-16T03:21:53.881954+00:00**: 1 outcome_unknown, 11 succeeded; **$12.0 reserved**, plus $2 overhead allowance under the $20 total authorization. These later calls are outside the earlier $0.43748322 reported app-hour. Per-call and full-task actual cost remain unknown; no reservation was released.

Private evidence is in `.platform/model-correspondence/model-completion/output-quality/followup-fences/`:

| Evidence | SHA-256 |
|---|---|
| `summary.json` | `b423c010b4fe7bf9ea469965bdbdacede3053026a9fc127d8025af59aecc1123` |
| `old-new-comparison.json` | `70114fc194e8afe9951ac880de93cdc78497f5f78b2c4e52a27dd55f8aacc873` |
| `cost-snapshot.json` | `7a2a4c1eb480fa6b2d37088a8fac5c079eeed8d1e637ec861ed164ead15ea2f4` |
| `parent-evidence-unchanged.json` | `d527df17efc6fdc812a9320007785d100eb48e0c1c0e3725ad3bb5c7ddd44a57` |
| `audit.py` | `d4ceb868c43567d7840d33920936cc703a775f859a088bd7b259029803cbe787` |
| `compare.py` | `340d5a98e6a6f4213ea44b4f85e312270f6315bec8817d34e4ce93039e7bd163` |

Visuals: `object-12-old-new.png`, `object-28-old-new.png`, and individual `object-<index>-view-0.png`. The parent eight-object evidence hashes were checked again and are unchanged.

CPU-only replay with read-only asset lookup:
```sh
PYTHONPATH=. .venv/bin/python .platform/model-correspondence/model-completion/output-quality/followup-fences/audit.py 12 28
PYTHONPATH=. .venv/bin/python .platform/model-correspondence/model-completion/output-quality/followup-fences/compare.py
```
