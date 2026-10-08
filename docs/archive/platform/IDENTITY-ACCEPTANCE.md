# Identity acceptance — source v2 batch, 2026-09-13

All 17 fixed targets pass persisted-data acceptance: 14 native-geometry targets succeeded, and 3 QA fixtures remain explicitly incomplete / not comparable. The shared source-provenance repair is now applied. Main base `a97267c4` becomes `b638e847` with 59 business records and all 84 original observations retained. Component head base `6770adce` becomes `6fca273f` with 81 records and all 82 observations retained.

Physical identity quality is still not fully established: the main report retains 47 independent pending observation records, and no independent human or field ground truth is available. Machine-readable IDs, hashes and coverage are in [IDENTITY-ACCEPTANCE.json](IDENTITY-ACCEPTANCE.json).

## Persisted evidence and original history

The authorized batch created independent reconstruction branches and ordinary durable reassociation jobs, using fixed original revision IDs, immutable prepared assets, and normal claim/finish CAS. All 17 heads advanced. No publication was modified by this batch, and no paid model was invoked. Two formerly null-Capture bases received new real Capture records referencing their exact original photo assets, target and source revision/manifest; no existing ambiguous Capture was guessed.

Independent acceptance then used a repeatable-read, read-only PostgreSQL transaction. All original 28 revisions, 19 publication snapshots, 369 assets, 6 branch heads, 12 Captures and 2 catalog bundles remain unchanged. The prior 17 batch documents and 4 canonical-repair successors are also retained. The acceptance did not read private feedback or capabilities and wrote no database rows or identity decisions.

Every new result passes `validate_document`, matches its stored document digest after PostgreSQL JSONB read-back, has the expected parent and branch head, and preserves source observation IDs/revisions, images, original boxes/polygons, pixel mappings, sourceRefs, existing masks, exact source cameras and original asset IDs. All 9,123 original non-null measurement values across the 17 targets survive with their original entity and source revision in measurement evidence. Actual bytes/sizes were verified for 523 referenced asset IDs, representing 442 distinct blobs. Legacy fork references to existing source-project assets are preserved.

| Fixed base | Final result | Business records | Observations | Verified source pairs | Job result |
| --- | --- | ---: | ---: | ---: | --- |
| `0e8851b9` | `d99fc805` | 68→61 | 58 | 0 | succeeded |
| `13c11c61` | `331ee47c` | 68→62 | 67 | 3 | succeeded |
| `2c7f5987` | `2e31a535` | 68→62 | 67 | 3 | succeeded |
| `45f843bb` | `3f702fb4` | 68→62 | 67 | 3 | succeeded |
| `518e7869` | `29000feb` | 68→62 | 67 | 3 | succeeded |
| `58ccf6df` | `b859de40` | 68→62 | 67 | 3 | succeeded |
| `5f397ec6` | `5ecdd5e7` | 85→81 | 82 | 0 | succeeded |
| `6770adce` | `6fca273f` | 85→81 | 82 | 0 | succeeded |
| `6a03b5ac` | `930ac2dc` | 1→1 | 1 | 0 | incomplete / not comparable |
| `89ca3522` | `fbd27668` | 68→62 | 67 | 3 | succeeded |
| `8df72130` | `8a575397` | 1→1 | 1 | 0 | incomplete / not comparable |
| `930db000` | `4b64c70f` | 85→81 | 80 | 0 | succeeded |
| `96f60839` | `ef5c38b2` | 68→62 | 67 | 3 | succeeded |
| `a97267c4` | `b638e847` | 68→59 | 84 | 6 | succeeded |
| `bdc2923f` | `79ad659d` | 85→81 | 82 | 0 | succeeded |
| `d403c6b5` | `348c140e` | 68→62 | 67 | 3 | succeeded |
| `fe366c68` | `65d94008` | 2→2 | 1 | 0 | incomplete / not comparable |

The full independent result is `.platform/identity-acceptance-source-v2-20260913-01/audit.json`; the durable execution record is `.platform/identity-source-v2-batch-audit-20260913-01.json`. Both retain all fixed-publication coverage through the frozen manifest.

## Exact source identity repair

Six main-report aliases were recovered for two historical views each of `left_post`, `right_post` and `right_fence`. Each pair has the same original SAM JSON SHA256 and RLE instance, maps to the same photograph SHA256, and loads to byte-identical canonical boolean masks. This is explicit source evidence, not a label or overlap heuristic.

The generic importer reads supported raw-source references and native mask provenance, stores the original source blobs and a proof document as immutable assets, and retains both original observation IDs and masks. Proofs carry both observation revisions, photograph hash, canonical-mask hash/shape, original instance pointer, and separately typed raw/native provenance pointers. Registration resolves actual database asset IDs before freezing proof bytes. Production reassociation checks both pointer owners, original RLE bytes and both canonical masks before the shared identity helper consumes the proof; known-equivalent observations then cease to compete as different identities. No manual identity decision was recorded.

All 30 proof pairs across the fixed target set were independently read back from registered assets and revalidated after execution. None was skipped. The main target's 6 pairs form 3 source merges that absorb 6 duplicate records while retaining their observations. Its 3 geometric merges remain supported by the prior review gallery. The converter is `public-scene-v9`; evidence is preserved for later verification rather than depending on local source paths.

## Current unresolved records and quality boundary

The main result has 12 confirmed multiview groups covering 37 observations and 47 independent pending observation records: 28 `competing_candidates` and 19 `no_supported_match`. Those 47 records are **not 47 established physical objects**. The record count changed 68→65 in the first geometry batch, then 68→59 when the complete source-v2 batch consumed the exact source aliases.

Replaying the current strongest-competitor calculation with proven source identities excluded finds that the top eligible pair for all 28 competing records still fails the configured 0.15 uniqueness margin in at least one direction. All 28 involve a remaining same-image competitor with canonical-mask IoU at least 0.80; 16 reach 0.95. No candidate-sort or wrong-second-candidate bug was found. Not being proven equivalent does not prove those records depict distinct objects.

| Pending record / tested direction | Top score | Competitor score | Margin | Mask IoU |
| --- | ---: | ---: | ---: | ---: |
| `37e1203b` sloped surface, reverse | 0.930067 | 0.899922 | 0.030145 | 0.840728 |
| `49637bd6` fence, reverse | 0.873783 | 0.871287 | 0.002496 | 0.988673 |
| `5fdd01a4` sensor, reverse | 0.974576 | 0.924670 | 0.049906 | 0.926273 |

The inspected raw sensor and guard use different source files/prompts and different masks from their generated counterparts; their provenance contains no exact alias. Their overlap cannot distinguish duplicate identity from physical part/containment relationships. Separate identity evidence or explicit review is still required; thresholds were not lowered and labels were not used to merge them.

At pair level, the current main run has 1,436 mask mismatches, 206 depth disagreements, 272 insufficient-support pairs, 96 eligible but unaccepted pairs and 6 accepted pairs. Pair counts overlap record participation and are not physical-object or ground-truth conflict counts. Every pending record and both directions of the competition replay are saved in `current-main-unresolved.json` and `current-main-competition.json` under the new acceptance directory.

## Visual review and prior hash incident

Actual original-photo and exact-pixel crop inspection previously covered 9 source groups, 5 negative comparisons, 11 offline pair variants and 4 additional actual-batch pairs. These are **Codex visual reviews, not human or field ground truth**. All reviewed same-identity proposals were visually supported; all negative comparisons supported distinct objects. Two restored component masks have only single-view evidence and remain ambiguous for cross-view identity. Comparing the complete new batch against the 15-pair actual-batch gallery found zero new geometric pair variants needing additional visual review.

Four initial component results had 21 negative-zero values each whose JSONB round trip changed their serialized hashes. The shared canonical implementation was fixed and ordinary append-only successors were verified in the previous round. This source-v2 batch includes the corrected canonical path and all 17 new digests pass. No historical document or stored digest was rewritten; the original failure audit, four successors and previous acceptance are retained, including `.platform/identity-acceptance-source-v2-20260913-01/previous-acceptance.json`.

No independent real-workcell holdout or human/field labels were available; precision and recall remain unmeasured.

## Export and public release

The two final scene revisions were exported, reopened in Blender and validated before publication. The main export contains 48 representations and three cameras; the component export contains 83 representations and four cameras. These are representation counts, not physical-object counts. Both exports retain an `incomplete` result because existing candidate placements remain unconfirmed; their Blender/GLB artifacts and validation records are available. No failed generation was presented as a validated physical placement.

The main publication is `8a61c1b8-65a0-498a-ac9a-7ed9924d6c9f` and the component publication is `02592388-82fd-494b-a478-4a833613eb98`. The public catalog contains these two results plus both previously deployed frozen reports. Catalog tests passed for four publication bundles, 483 exact responses and 454 byte-verified assets. Intermediate publications were retained locally but were not added to the deployed catalog.

The final Python suite passed 137 tests, including real PostgreSQL isolation and Blender validation where applicable. TypeScript, frontend self-checks and the production build passed. A real browser discovered an omitted photograph dependency in the CAD memo; source commit `5c0d549` fixes it and leaves a check that executes the real memo initializer. Code commit `8825095` and the CAD fix were pushed to `codex/panoptes-platform`; the public frontend was published through Pages commit `fab1163393d412e03dfbcf57ed354cbf12ff9201`.

Public API read-back confirmed both final scene IDs and the unchanged original report. Real computer-use checks covered linked source-photo selection, photo-specific CAD, public identity preview, the feedback panel, default no-selection state and the frozen v1 report route. Details are in [IDENTITY-BROWSER-QA.md](IDENTITY-BROWSER-QA.md). This release did not rerun EHS policies or paid reconstruction models; historical EHS results remain explicitly historical.
