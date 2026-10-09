# Independent-review follow-up — 2026-10-08

The independent review accepted `2d8a419` as Modal source delivery and correctly identified several execution and evidence gaps. The repairs below preserve the selected reconstruction, model revisions, scales and physical measurement algorithms. The original acceptance remains a dated record; this document describes the subsequent source state.

## Findings and repairs

| Review finding | Current action and evidence |
| --- | --- |
| Builder could publish fewer boxes than configured objects | The shared builder now checks each configured object through its entity mapping to a nonempty box before writing. Both real-cell replay fixtures failed the new missing-box check before the fix and pass afterward; incomplete inventory produces no layer. |
| Frozen published boxes could conceal a box-algorithm regression | Published-cell tests are explicitly serialization/inventory regressions. The existing `test_box_computation_matches_original_selfcheck` already executes the actual `_check()` computation on both synthetic scenes and compares their original numerical outputs. No duplicate test was added. These checks do not recompute the original field scenes. |
| Modal executable could come from another Python installation | `Ctx.modal_run` invokes `sys.executable -m modal`, including in dry runs. A foreign PATH executable cannot override the locked environment. Both workcells have a regression for this. |
| S2d required the other workcell's analysis JSON | Producer, CLI gate and layer consumer now use mandatory selected-cell input and `pipeline/field-values-<cell>-mvs-fill.json`. Single-cell data roots succeed independently. All 30 individual field/scale/floor values in the frozen analyses are exactly unchanged. Aggregate MAE now covers that cell's two scored values, with the original 1.56 cm MAE and 3 cm maximum-error limits; it is no longer labelled as four-value MAE. |
| Agent path changed Dutch to English | Frontend handlers, API DTO and Python request contracts now retain `en`, `zh` and `nl` for agent and feedback requests. A real-handler harness failed on Dutch before the fix and passes all four switches afterward. It does not call a model provider. |
| Code audit used whichever `argus` installation Python imported | The script audits the checkout containing the script itself. A separate temporary Git checkout with a foreign `PYTHONPATH` was used to verify it reports that checkout's invalid source. |
| Fullwidth punctuation and clear Dutch terminology errors | The source gate includes fullwidth characters; non-catalog source punctuation was removed or localized. Dutch safety vest and dimension labels use `hesje` and `L×B×H`. Customer terminology sign-off remains open. |
| Viewer language check only covered labels | The mounted DOM/WebGL-stub harness now loads a scene primitive and schema-2 measurement layer. It checks submitted world corners against box geometry, vertices, indices, model transforms, projected measurement lines and native ground grid across `en → zh → nl → en`; buffer uploads stay at two. This is executable geometry evidence, not browser/GPU screenshot acceptance. |
| PostgreSQL replay command missed locale | The local restart command now sets `LC_ALL=en_US.UTF-8`. The full gate used the isolated PostgreSQL 16 UTF-8 test database on port 55439. |
| CI only triggered on main/staging | Push filters now include the two actual delivery/migration branches. The local checks above do not establish a hosted Linux CI result. |

## Old published reports and input preservation

The viewer continues to require schema 2. The two existing published layers are schema 1. Their originals were copied into the mirror as **reference-only** `baselines/published/<publicationId>.json`, and their physical projections match the frozen test references. They are not active viewer layers.

The manifest now verifies **1,024 files / 520,912,075 bytes**; the original 1,022 entries retain their sizes and SHA-256 values. A complete mirror verification checked all 1,024 files without fetching any.

The mirror and searched local source/report archives do not contain the genuine S8 check outputs, assembly and generation inputs needed to regenerate those old layers. Test-created replay files are not those original outputs. No schema-1 loader branch, fabricated reconstruction evidence, old-site upgrade or deployment was introduced. The source remains suitable for a fresh customer run producing schema 2; replacing the existing site requires obtaining its real stage artifacts and rebuilding its layers first.

## Executed source checks

- Actual `make test`, locked Python 3.12 environment and isolated PostgreSQL 16 UTF-8 database: **747 passed / 51 skipped**, one existing Starlette/httpx warning, 63.06 seconds.
- Targeted runtime/CLI gate: **37 passed**; field-value gate rejection at both existing error limits is covered.
- Measurement regressions: **8 passed**, including two missing-box rejections and the existing recomputed box self-check.
- `npm run check` (six executable checks), `npm run build`, the Python viewer gate and four existing report/evidence harnesses passed. Each locale has **1,777 matching keys** and matching placeholders.
- Delivery gate, Ruff checks for edited Python and `git diff --check` passed.
- Rebuilt wheel installed offline into a fresh target, with package and model cache read-only: **16 passed**, plus eight package/import checks, both-cell console dry runs and nested English diagnostic rendering. The new wheel is 658,655 bytes, SHA-256 `4a6c368c999840293244f4ea554d185a3ab3a0fd33d86d74c2655329e419993c`; its 121 package entries match current source digest `18a8449a949e842bfcc346e012ba5f509dc5b7c97ce3acb5447737ea1fdb94a3`. Local wheel and exact commands are in sibling `phase5-artifacts/verified-review-followup-2026-10-08`; the old wheel and metadata are preserved in `phase5-artifacts/accepted-2d8a419`.

The 51 conditional skips still include real Blender, torch/SAM3D source, MongoDB and S3 integrations. No fresh GPU reconstruction, live Modal/HTTP inference, LLM evaluation, input upload, customer Linux/Docker execution or physical calibration was performed.

## Remaining review observations

The configured delivery still supports these two workcells; their retained numerical source contains cell-specific mappings, and the 090 package configuration includes the existing reviewer-drawn `partMasks` polygons. Broad configuration extraction would change the accepted source boundary and is not part of these bug fixes. Builder optional stage defaults, upstream evidence/config synchronization, Windows symlink behavior, Mongo publication policy plumbing, live integration checks, npm advisories and Dutch customer terminology remain separate observations requiring their own evidence or implementation decision. The saved npm audit was not refreshed and no dependency was upgraded.

The independent verdict repository owns its engine changes and research evidence. `clingo@5` encodes exact rational interval bounds and thresholds as small order ranks; native ASP still decides statuses independently of `python@6`. The Python engine also preserves tiny uncertainty in its interval arithmetic. This removes decimal rounding and integer overflow, and fixes the strict-`<` crossing case. Original measured values and uncertainty remain in the output; ranks are only solver inputs. Shared input resolution and interval arithmetic are not independently validated merely because statuses agree.

The lab also records benchmark paths relative to its runset directory and rejects absolute stored paths when regenerating a scorecard; scorecard rows sort by run ID before computing differences. Only the two reviewed frozen runsets are explicitly allowed by `.gitignore`, while other experiment outputs remain ignored. Old v5 evidence stays historical and is not rewritten.

Its [RESULTS.md](https://github.com/admin-wekruit/panoptes-verdict-lab/blob/codex/verdict-lab/RESULTS.md) records the current runset. Frozen trial gold is a regression reference, not human-labelled accuracy; compiled rules with no real bindings, missing floor coverage and reviewer polygons remain unresolved. No compliance conclusion follows from engine parity.
