# Detection coverage and call-budget check

The same two source images were passed through the previous `HEAD` detector and the revised detector. The Python code actually ran, with deterministic injected VLM/SAM providers. This tests frame coverage, source coordinates, and request count; it is **not** a model accuracy or billed-cost benchmark.

| Metric, same two input images | Before | After |
| --- | ---: | ---: |
| Input photos | 2 | 2 |
| Photos processed | 1 | 2 |
| Recorded masked instances | 1 | 2 |
| Logical semantic model calls | 20 | 2 |
| Of these: bulk sweeps | 1 | 2 |
| Of these: per-object locator retries | 17 | 0 |
| Of these: crop checks / extra sweeps | 2 | 0 |
| SAM calls | 1 | 2 |
| Repeated identical run: model calls | 0 | 0 |
| Repeated identical run: SAM calls | 0 | 0 |
| Recall / precision | Unknown | Unknown |
| Billed dollars | Not measured | Not measured |

The additional SAM call covers the second source photo. The two scripted instances do not prove that the new prompt finds more real objects. The output explicitly marks semantics as `single_sweep`; a SAM mask cannot establish a switch's function or compliance.

Artifacts: `runs/detect-multiframe-check-20260909/metrics.json`, including identical source-image hashes, before/after source-code hashes, and per-operation counts. Reproduce using `.venv/bin/python runs/detect-multiframe-check-20260909/check.py`; it replaces only its own generated `before/` and `after/` fixtures.

`tests/test_detect.py` separately verifies an old first-photo cache plus a new second photo of different dimensions: only photo 2 receives one sweep and one SAM request; both overlays keep their native sizes; the generic object registry decodes the second mask on its correct source grid. Repeating is zero-call; changing photo 2's pixels creates a new hash-bound cache while photo 1 stays cached. Empty SAM output remains a localized instance with an explicit segmentation failure.

New cache keys include frame ID, source SHA-256, and detection version. Existing frozen detections without a source hash are marked `legacy_first_frame_unverified`, never silently assigned to subsequent photos or upgraded to verified semantic evidence.
