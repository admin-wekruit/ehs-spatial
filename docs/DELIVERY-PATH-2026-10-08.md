# Delivery dependency closure — 2026-10-08

The delivered implementation is the existing DA3-BASE → RoMa/MVS → MoGe-3 masked fill → SAM 3D Objects → assembly → measurement → publication pipeline. Its Python source is now one `argus` package. Customer Linux, Docker and GPU-service adaptation remains outside this source delivery. The supported GPU seams are the existing Modal calls and HTTP contract. Numeric algorithms, thresholds, model revisions and saved publication scales are preserved.

## Size, ownership and shortest structure

The final source has **113 Python files**: root 1, pipeline 35, checks 18, providers 16, platform 43. The map records 113 existing source origins, including one extracted `camera_depth` function, plus one new package initializer. It preserves the existing platform API/domain modules rather than redesigning them. There are **8 package assets**: two workcell JSON files, the shared English message catalog, four SQL migrations and the existing SAM3D mesh-only patch. Repository tools are five Python files, including their initializer. The retained test/check inventory has 46 Python files, plus the frozen measurement fixtures and current viewer checks.

```text
argus/
  __init__.py                 ROOT: the one code location
  pipeline/                   CLI, selected geometry, completion, assembly and report
    cells/{090,030}.json       existing per-cell inputs, object/frame order and scale
  checks/                     original numerical measurement/check implementations
  providers/                  Modal source, HTTP client, codecs and pinned weight helpers
  platform/                   existing API, storage, import, publication and worker
scripts/                      input fetch and environment/delivery/regression checks
web/                          existing viewer, locales, build config, assets and checks
tests/                        retained provider/platform/measurement checks
```

`panoptes = argus.pipeline.cli:main` is the package entry point. CPU subprocesses use `python -m argus.pipeline.<stage>` or `python -m argus.checks.workcell_layer_trial`. Existing setuptools package data includes the cell JSON, English catalog, migrations and patch. There is no old package alias, dated path loader, Modal emulator or customer service implementation in the delivery tree.

`argus.ROOT` locates source and bundled cell configuration. `PANOPTES_DATA_ROOT` locates capture inputs and every pipeline/publication output; the CLI default is `./data` resolved in the calling workspace, so an installed command does not write into site-packages. `PANOPTES_RUNS` and `PANOPTES_PAGES` default to its `runs` and `measurement-layer` directories. All three paths, including explicit relative overrides, are resolved once at the CLI boundary and propagated to every child. `PANOPTES_DATABASE_URL` selects the existing PostgreSQL or MongoDB repository. S7 invokes the publisher directly; database provisioning is external.

## Actual call graph

| Stage | Source and real call | Required input / produced output |
| --- | --- | --- |
| CLI | `pipeline.cli.main` → `Ctx` → selected `ITEMS`; child failures stop the run and are recorded in the existing ledger | Cell JSON; environment; one chosen data root |
| S2a GPU | `providers.geometry_mvs.run` → HTTP or `_run_stages`; canonical imports of `field_evaluator` and `geometry_clean_ab`; the request's frames replace `field_evaluator.frames_for` | Original canonical PNG/alpha arrays → `checks/clean-gpu/<cell>`, `checks/da3fair-geom/<cell>-da3-base-padded/geometry`, `checks/clean-geom/<cell>-da3-base-ba-f/geometry` |
| S2a bodies | Fair `infer.remote` → existing `DA3Runner` → existing `MapAnythingAdapter` array serializer; clean `main(infer)` → MoGe-3 + RoMa, `main(refine)` → NumPy BA, `main(dense-infer)` → dense matches | DA3-BASE pin and unchanged 518-grid processing; original crop and RoMa settings |
| S2b | CLI subprocess `pipeline.mvs_route` → existing triangulation/fusion routines in `geometry_clean_ab` | `checks/bbab-geom/<cell>-mvs-da3-base-padded/geometry` |
| S2c | `geometry_evaluator.analyse_local` → plain CPU `field_evaluator.analyse` / `analyse_one`; then `export_run` → original capture-evidence helpers | Frozen floor/object/e-stop evidence; `checks/bbab-analyse`, `checks/bbab-export-<cell>-mvs-scipyba` |
| S2d | `fill_geometry` → original masked fit/fill; analyse/export again; `field_values_fill` → original `field_summary_safe.safe_config` and field-value gate | Both cells require explicit `*-mvs-fill-padded`; no 030 historical fallback. Output `pipeline/field-values-mvs-fill.json` and `checks/bbab-export-<cell>-mvs-fill` |
| S4a GPU | `completion.stage_sam3d` → `providers.sam3d.generate` → cached `sam3d_modal.SAM3DObjects.run.remote` or HTTP | Same selected RGB/mask/point-map inputs and seed 42; `swap-runs/<cell>/mvs-fill-ab/sam3d*` candidates; every configured object requires at least one actual NPZ |
| S4b CPU | `completion.stage_assemble` → `assemble_variants` → original `assemble_lucida_scene.assemble`; `select_sam3d` uses the same selection score, floor rules and candidates | Candidate geometry, comparisons and required evidence sheets under `pipeline/cmp-<cell>-mvs-fill` |
| S4c | `swap_generation` → original generation contract; `pin_run` → importer provenance | `swap-runs/<cell>/mvs-fill-sam3d`; no historical Pi3X/RecGen run dependency |
| S5 | `assemble_scene` → original 9-DOF placement fit with v2 floor penalty, iterations 100 and evaluation size 288 | Native generated mesh, canonical masks/cameras and one floor; result scene/binary/GLB/comparisons |
| S6 | `build_capture_report` → existing Pages `report_geometry` helpers → subprocess `pack_model` | Exact per-object packed bytes; report photo polygons, floor-frame plan, cameras and provenance |
| S7 | `platform.publish_capture` → existing `runtime.services`, `run_import`, report/geometry evidence import, migrateScene v2 + saved e-stop calibration, `export_platform_publication` | Configured database/blob store; `publications/<variant>/result.json` and `view.json`; `publications/catalog/<publicationId>` |
| S8–S14 | `serve_export` → local immutable asset layout; `run_stages` → original shape/floor/lines/stereo/transfer/clearance/lower-edge/box/obvious/facet checks | `swap-runs/<variant>-served*` and `swap-runs/<variant>-stages/<check>`; every child must succeed and write results |
| Layer and web | `build_swap_layer` consumes this cell's selection, field values, assembly and checks; existing web measurement loader/report/boxes consume schema 2 | `measurement-layer/<publicationId>.json`, required pipeline sheets, platform view and hash-addressed model assets |

The platform's existing worker remains `python -m argus.platform.worker`. SQL policy modules belong to the retained platform API; they are separate from the independent Phase 4 verdict package. Existing generic reconstruction/receipt contracts are retained where the API actually imports them. They do not add historical model options to the delivered CLI.

An error-only SAM3D record is not stage completion. The all-view producer and S4a cache guard require an NPZ for every configured object; an individual view error remains recorded when another view supplies a candidate. S4b reuses selection only when every configured object has a current decision message with a code; stale prose, missing objects or invalid JSON cause current CPU selection to run. S4c refreshes selection metadata only for the same candidate and anchor; a changed candidate fails with the required S4c–S8 recomputation scope before old geometry can be relabeled. The selector and generation contract reject missing configured objects before partial output can be published. The publication entry also validates evidence, comparisons and complete generated mesh files against the cell configuration before creating storage services; the layer builder validates the configured object inventory again.

## Hidden imports and build assets

- DA3 inference really calls `candidate_geometry_backend.DA3Runner` and `providers.map_anything.MapAnythingAdapter`. Their names are historical; they are part of the selected DA3 numerical/array closure. The adapter requires the injected runner and has no default hosted-provider call. DA3 image construction now clones only the pinned Depth-Anything-3 source. The unselected Pi3X, MapAnything vendor checkout and DA3-LARGE image branches are absent.
- `geometry_clean_ab` really imports `providers.moge3_modal` for its original image, weight volume, model and revision. That source must stay even though the unrelated `providers.moge` adapter is archived. RoMa and MoGe-3 are inputs to the delivered reconstruction, not optional comparison modules.
- Modal image definitions use `add_local_python_source('argus')`. SAM3D also mounts `prepare_sam3d_mesh_source.py` and the reviewed patch together under `/opt/prep/scripts`; the helper locates the patch beside itself. No dated notes folder or old source package is mounted.
- `camera_depth` was copied verbatim from the old asset generator into the assembly module because SAM3D input construction actually uses it. The rest of that generator is research.
- The two Pages helpers were reused from sibling `panoptes-workcell-pages/build-unified-data.py` (only `floor_transform`, `pose`, `contours`, `metrics`) and `pack-model.py`. The pose helper's existing SciPy `Rotation` import is included. There is no external Pages checkout import at runtime.
- `pipeline/locales/en.json` is the shared English catalog bundled for selection diagnostic sheets. Wheel execution reads it beside the selector, without a separate web checkout.
- The assembler's independent research `lucida_viewer.html` copy had no formal publication consumer and is archived with that viewer. The official web viewer consumes the retained scene/report/publication data. Assembly math and comparison outputs remain in place.
- Importer source hashes now read canonical package files. Code hashes necessarily change with relocation; saved physical scales and measurements do not.
- The only retained first-party `sys.path` addition is inside the existing DA3 runner for the pinned upstream vendor source in its Modal image. Dated first-party `spec_from_file_location` loaders and path aliases have been removed.

## Existing source origins

The table is an exhaustive Python map, including function extraction. Entries describe relocation, import/path adaptation and removal of unused branches; they do not assert byte-identical whole modules after cleanup.

| Original source | Canonical source | Treatment |
| --- | --- | --- |
| `ehs_spatial/__init__.py` | `argus/__init__.py` | Existing module; canonical imports/paths |
| `ehs_spatial/backends.py` | `argus/pipeline/backends.py` | Existing module; canonical imports/paths |
| `ehs_spatial/cli.py` | `argus/pipeline/cli.py` | Existing module; canonical imports/paths |
| `ehs_spatial/contracts.py` | `argus/pipeline/contracts.py` | Existing module; canonical imports/paths |
| `ehs_spatial/geometry.py` | `argus/pipeline/geometry.py` | Existing module; canonical imports/paths |
| `ehs_spatial/measurements.py` | `argus/pipeline/measurements.py` | Existing module; canonical imports/paths |
| `ehs_spatial/object_evidence.py` | `argus/pipeline/object_evidence.py` | Existing module; canonical imports/paths |
| `ehs_spatial/path_safety.py` | `argus/pipeline/path_safety.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/__init__.py` | `argus/platform/__init__.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/agent_service.py` | `argus/platform/agent_service.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/api.py` | `argus/platform/api.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/blender_export.py` | `argus/platform/blender_export.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/coarse_model.py` | `argus/platform/coarse_model.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/config.py` | `argus/platform/config.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/contracts.py` | `argus/platform/contracts.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/correspondence.py` | `argus/platform/correspondence.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/executor.py` | `argus/platform/executor.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/feedback.py` | `argus/platform/feedback.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/hosted_sam3d.py` | `argus/platform/hosted_sam3d.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/identity.py` | `argus/platform/identity.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/model_quality.py` | `argus/platform/model_quality.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/mongo.py` | `argus/platform/mongo.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/mongo_policy.py` | `argus/platform/mongo_policy.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/planar_surfaces.py` | `argus/platform/planar_surfaces.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/policy_engine.py` | `argus/platform/policy_engine.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/policy_repository.py` | `argus/platform/policy_repository.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/policy_service.py` | `argus/platform/policy_service.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/postgres.py` | `argus/platform/postgres.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/publication_site.py` | `argus/platform/publication_site.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/publication_view.py` | `argus/platform/publication_view.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/recgen.py` | `argus/platform/recgen.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/recgen_transport.py` | `argus/platform/recgen_transport.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/reconstruction.py` | `argus/platform/reconstruction.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/reconstruction_pipeline.py` | `argus/platform/reconstruction_pipeline.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/repository.py` | `argus/platform/repository.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/research_authority.py` | `argus/platform/research_authority.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/research_inputs.py` | `argus/platform/research_inputs.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/runtime.py` | `argus/platform/runtime.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/s3_storage.py` | `argus/platform/s3_storage.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/scene_measurements.py` | `argus/platform/scene_measurements.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/source_cad.py` | `argus/platform/source_cad.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/spatial.py` | `argus/platform/spatial.py` | Existing module; canonical imports/paths |
| `ehs_spatial/platform/storage.py` | `argus/platform/storage.py` | Existing module; canonical imports/paths |
| `ehs_spatial/policy.py` | `argus/pipeline/policy.py` | Existing module; canonical imports/paths |
| `ehs_spatial/providers/__init__.py` | `argus/providers/__init__.py` | Existing module; canonical imports/paths |
| `ehs_spatial/providers/base.py` | `argus/providers/base.py` | Existing module; canonical imports/paths |
| `ehs_spatial/providers/gemini.py` | `argus/providers/gemini.py` | Existing module; canonical imports/paths |
| `ehs_spatial/providers/geometry_mvs.py` | `argus/providers/geometry_mvs.py` | Existing module; canonical imports/paths |
| `ehs_spatial/providers/map_anything.py` | `argus/providers/map_anything.py` | Existing module; canonical imports/paths |
| `ehs_spatial/providers/sam3.py` | `argus/providers/sam3.py` | Existing module; canonical imports/paths |
| `ehs_spatial/providers/sam3d.py` | `argus/providers/sam3d.py` | Existing module; canonical imports/paths |
| `ehs_spatial/providers/service_client.py` | `argus/providers/service_client.py` | Existing module; canonical imports/paths |
| `ehs_spatial/rules.py` | `argus/pipeline/rules.py` | Existing module; canonical imports/paths |
| `modal_apps/assemble_scene.py` | `argus/pipeline/assemble_scene.py` | Existing module; canonical imports/paths |
| `modal_apps/bundle_adjust.py` | `argus/pipeline/bundle_adjust.py` | Existing module; canonical imports/paths |
| `modal_apps/geometry_clean_ab.py` | `argus/pipeline/geometry_clean_ab.py` | Original selected MoGe/RoMa/BA/MVS bodies |
| `modal_apps/moge3_app.py` | `argus/providers/moge3_modal.py` | Existing module; canonical imports/paths |
| `modal_apps/publication_site.py` | `argus/platform/publication_modal.py` | Existing module; canonical imports/paths |
| `modal_apps/sam3d_research.py` | `argus/providers/sam3d_modal.py` | Existing module; canonical imports/paths |
| `modal_apps/workcell_layer_trial.py` | `argus/checks/workcell_layer_trial.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/cell030-sept-pipeline-2026-10-05/estop_cylinder.py` | `argus/checks/estop_cylinder.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/completion-ab-090-2026-10-06/compare.py` | `argus/pipeline/select_sam3d.py` | Selected SAM3D score/rules + required sheets |
| `research/module-swap-2026-10-07/notes/completion-licence-ab-2026-10-05/completion_ab.py` | `argus/pipeline/completion.py` | SAM3D/candidate assembly; unused TRELLIS branch archived |
| `research/module-swap-2026-10-07/notes/geometry-backbone-ab-2026-10-06/backbone_ab_modal.py` | `argus/pipeline/geometry_evaluator.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/geometry-backbone-ab-2026-10-06/backbones.py` | `argus/pipeline/geometry_contract.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/geometry-backbone-ab-2026-10-06/mvs_route.py` | `argus/pipeline/mvs_route.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/geometry-licence-ab-2026-10-05/geometry_ab.py` | `argus/checks/geometry_measurements.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/geometry-licence-ab-fair-2026-10-05/compile.py` | `argus/pipeline/field_summary.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/geometry-licence-ab-fair-2026-10-05/fair_ab.py` | `argus/checks/field_geometry.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/geometry-licence-ab-fair-2026-10-05/fair_ab_modal.py` | `argus/pipeline/field_evaluator.py` | Original DA3-BASE inference + fair evaluator |
| `research/module-swap-2026-10-07/notes/licence-clean-stack-2026-10-06/geometry/compile.py` | `argus/pipeline/field_summary_safe.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/module-swap-090-2026-10-07/build_swap_layer.py` | `argus/pipeline/build_swap_layer.py` | Original numbers + schema-2 catalog messages |
| `research/module-swap-2026-10-07/notes/module-swap-090-2026-10-07/field_values_fill.py` | `argus/pipeline/field_values_fill.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/module-swap-090-2026-10-07/fill_geometry.py` | `argus/pipeline/fill_geometry.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/module-swap-090-2026-10-07/pin_run.py` | `argus/pipeline/pin_run.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/module-swap-090-2026-10-07/run_stages.py` | `argus/pipeline/run_stages.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/module-swap-090-2026-10-07/serve_export.py` | `argus/pipeline/serve_export.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/module-swap-090-2026-10-07/shape_all_task.py` | `argus/checks/shape_all.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/module-swap-090-2026-10-07/swap_generation.py` | `argus/pipeline/swap_generation.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/notes/workcell-clearance-b-2026-10-05/clearance_b.py` | `argus/checks/clearance_geometry.py` | Existing module; canonical imports/paths |
| `research/module-swap-2026-10-07/platform/publish-swap-20261007.py` | `argus/platform/publish_capture.py` | Existing module; canonical imports/paths |
| `scripts/candidate_geometry_backend.py` | `argus/pipeline/candidate_geometry_backend.py` | Existing module; canonical imports/paths |
| `scripts/export_platform_publication.py` | `argus/platform/export_platform_publication.py` | Existing module; canonical imports/paths |
| `scripts/import_geometry_evidence.py` | `argus/platform/import_geometry_evidence.py` | Existing module; canonical imports/paths |
| `scripts/import_public_scene.py` | `argus/platform/import_public_scene.py` | Existing module; canonical imports/paths |
| `scripts/import_report_evidence.py` | `argus/platform/import_report_evidence.py` | Existing module; canonical imports/paths |
| `scripts/onprem/fetch_weights.py` | `argus/providers/fetch_weights.py` | Existing module; canonical imports/paths |
| `scripts/onprem/fetch_weights_da3.py` | `argus/providers/fetch_weights_da3.py` | Existing module; canonical imports/paths |
| `scripts/onprem/fetch_weights_geometry.py` | `argus/providers/fetch_weights_geometry.py` | Existing module; canonical imports/paths |
| `scripts/onprem/fetch_weights_sam3d.py` | `argus/providers/fetch_weights_sam3d.py` | Existing module; canonical imports/paths |
| `scripts/prepare_publication_site.py` | `argus/platform/prepare_publication_site.py` | Existing module; canonical imports/paths |
| `scripts/prepare_sam3d_mesh_source.py` | `argus/providers/prepare_sam3d_mesh_source.py` | Existing module; canonical imports/paths |
| `scripts/reference_object_scale.py` | `argus/checks/reference_object_scale.py` | Existing module; canonical imports/paths |
| `scripts/research/assemble_lucida_scene.py` | `argus/pipeline/assemble_lucida_scene.py` | Existing module; canonical imports/paths |
| `scripts/research/build_capture_report.py` | `argus/pipeline/build_capture_report.py` | Existing module; canonical imports/paths |
| `scripts/research/generate_lucida_assets.py` | `argus/pipeline/assemble_lucida_scene.py` | Existing camera_depth function only |
| `scripts/research/prepare_capture_evidence.py` | `argus/pipeline/prepare_capture_evidence.py` | Existing module; canonical imports/paths |
| `scripts/research/prepare_lucida_evidence.py` | `argus/pipeline/prepare_lucida_evidence.py` | Existing module; canonical imports/paths |
| `scripts/workcell_checks/__init__.py` | `argus/checks/__init__.py` | Existing module; canonical imports/paths |
| `scripts/workcell_checks/box_faces.py` | `argus/checks/box_faces.py` | Original numbers + schema-2 catalog messages |
| `scripts/workcell_checks/clearance.py` | `argus/checks/clearance.py` | Existing module; canonical imports/paths |
| `scripts/workcell_checks/floor.py` | `argus/checks/floor.py` | Existing module; canonical imports/paths |
| `scripts/workcell_checks/lines.py` | `argus/checks/lines.py` | Existing module; canonical imports/paths |
| `scripts/workcell_checks/lower_edge.py` | `argus/checks/lower_edge.py` | Existing module; canonical imports/paths |
| `scripts/workcell_checks/obvious_errors.py` | `argus/checks/obvious_errors.py` | Existing module; canonical imports/paths |
| `scripts/workcell_checks/plane_facets.py` | `argus/checks/plane_facets.py` | Existing module; canonical imports/paths |
| `scripts/workcell_checks/plane_stereo.py` | `argus/checks/plane_stereo.py` | Existing module; canonical imports/paths |
| `scripts/workcell_checks/transfer.py` | `argus/checks/transfer.py` | Existing module; canonical imports/paths |
| `scripts/workcell_shape_check.py` | `argus/checks/shape_core.py` | Existing module; canonical imports/paths |
| `../panoptes-workcell-pages/pack-model.py` | `argus/pipeline/pack_model.py` | Existing module; canonical imports/paths |
| `../panoptes-workcell-pages/build-unified-data.py` | `argus/pipeline/report_geometry.py` | Four existing Pages geometry helpers |
| `panoptes_worker/__main__.py` | `argus/platform/worker.py` | Existing module; canonical imports/paths |
| `scripts/preflight_sam3d.py` | `argus/providers/preflight_sam3d.py` | Existing module; canonical imports/paths |
| New package marker | `argus/pipeline/__init__.py` | Package initializer |

## Non-Python assets, inputs and output layout

| Asset / input | Required use |
| --- | --- |
| `argus/pipeline/cells/{090,030}.json` | Existing run name, object/frame order, exact scale path, lower-edge targets, part-mask bindings and current baseline identity |
| `argus/pipeline/locales/en.json` | Shared English message catalog for required selection diagnostic sheets; bundled with the Python package |
| `argus/platform/migrations/{001_core,002_policy,003_job_leases,004_policy_agent_links}.sql` | Existing repository schema; all four travel in the wheel |
| `argus/providers/sam3d_mesh_only.patch` | Existing reviewed mesh-only upstream source build |
| `web/app.html`, package lock/Vite/TypeScript configuration, `web/src/**`, `web/src/locales/{en,zh,nl}.json`, source fonts/licenses | Existing official viewer/build/locale consumers; generated `web/dist` is built from source |
| `web/checks/fixtures/model-family.json` | Existing platform/frontend model-family contract test |
| `tests/fixtures/measurements/**` | Frozen 090/030 measurement projection, source identities and original box self-check numbers |
| `runs/lucida-replica-01/**`, `runs/bor1-030-01/**` under the configured data root | Capture manifest, raw/canonical photos, alpha/content masks, evidence/object masks and original pixel/camera provenance used by the selected route |
| `checks/da3fair-data/{090,030}/**` | Required fair evaluator input: manifest, view, clearb, floor masks, e-stop photos/features and canonical frames |
| `mvs090/estop-scale2.json`, `mvs030/estop-scale.json` | Exact saved publication scale; publisher never substitutes a new value |
| `pipeline/cmp-<cell>-mvs-fill/{results.json,*.jpg}` | Selection and required evidence sheets; produced by S4b and consumed by generation/layer builder |
| `pipeline/field-values-mvs-fill.json` | Both cells' explicit MVS and MVS+fill analysis; produced before publication layer |
| `publications/<variant>/{result.json,view.json}`, `publications/catalog/<id>`, `publications/{imports,http}` | One publication result/catalog/import/HTTP layout shared by publisher and consumers |

The preserved input source is sibling `phase5-inputs`. The complete `data-manifest.json` records **1022 files / 520,797,028 bytes** with every size and SHA-256. Its preservation set includes historical analysis artifacts; these are not all mandatory for a new selected reconstruction. The indispensable original capture/fair-evaluator/scale subset is listed explicitly as `required_input_paths` in the machine-readable migration map. Recomputed geometry, candidates, assembly, checks, sheets and publications are pipeline outputs. The original comparison inputs remain recoverable in the explicit input mirror, outside Git source.

`scripts.fetch_inputs` accepts an explicit source mirror and verifies checksums before a fetched file becomes an input. The existing supported file, HTTP and S3 mechanisms are used. There is no default customer storage location and no upload was performed. Model weights remain separate from capture inputs: original model/code revisions, checkpoint SHA records and volume paths are preserved in the Modal/fetch helpers. No weights, tokens or `.platform` data were read or copied into the source archive.

## Existing HTTP seam

`SAM3D_BACKEND` and `GEOMETRY_MVS_BACKEND` select `modal` (default) or `http`. The shared stdlib `providers.service_client` implements the existing v1 service protocol, authentication, least-queue selection, bounded retry, job polling, model-info receipt and result download. `SAM3D_HTTP_URLS`, `GEOMETRY_MVS_HTTP_URLS` and `PANOPTES_SERVICE_API_KEY` configure customer endpoints; comma-separated roots retain the existing queue-selection behavior.

- SAM3D sends `seed`, `image_b64`, `mask_b64`, `pointmap_npz_b64`. The PNG mask is binary; the point map preserves the original SAM3D coordinate convention. Existing result normalization validates vertices, faces, pose and model metadata.
- Geometry sends `cell`, unchanged `options={start: da3-base, roma: outdoor, pairs: all}` and per-frame `frame_id`, `canonical_png_b64`, `alpha_npy_b64`. The downloaded tar must contain all three canonical geometry/GPU directory outputs before downstream execution continues.

The original native-service/emulation code is preserved in the research reference. The delivery includes its existing client contract and real Modal source; it does not claim that customer Docker/GPU services are already implemented or validated.

## Archive and tests

Sibling `phase5-research-reference` preserves the original modules, scripts, full documentation and original tests before source removal. Archived directories include `ehs_spatial`, `modal_apps`, `fast_report`, `panoptes_worker`, old serving/services/on-prem/deploy/Docker source, dated research, old configs/containers/eval/experiments, ordinary historical incoming/runs data and the old app entry point/version/changelog. Legacy comparison table builders, independent viewer, capture-plan prose, unused adapters, video/workbench experiments and their tests stay there. Phase 4 verdict source/tests/lab live independently in sibling `phase4-verdict-lab`.

The unused `web/experiments` subtree and old `web/checks` scripts are also archived: all 35 removed files were copied and checksum-verified first. The delivered `web/checks` directory contains only `fixtures/model-family.json`, which the retained Python contract tests read. The unused photo-only entry and thirteen obsolete live-video/splat browser checks, plus their video-experiment camera check, are also preserved outside delivery. The final web inventory contains 80 source/config/test files. Current `web/src` and retained `web/tests` form the viewer source and checks.

The retained tests exercise the actual provider HTTP and Modal dispatch boundaries, per-cell subprocess/output paths, packed report bytes/geometry helpers, language-neutral cap messages, platform import/publication/worker/API/storage, current viewer catalogs and frozen physical measurement projections. Mixed platform test files retain the real platform assertions while their standalone workbench-helper cases are separated. Original tests remain available in the reference. Native-service emulator tests and historical workspace/video/alternative-provider tests are not part of the delivery default suite. Root-owned input-fetch and measurement regression tools/fixtures remain delivered.

Executed evidence during migration: complete retained offline suite **570 passed / 186 conditionally skipped** before the final report/CLI checks; real loopback HTTP contract **16 passed**; CLI/runtime report/cap/candidate/cache/path checks **25 passed** after the last fixes. All six retained Modal definition modules import without model calls; both SAM3D mounted files exist. Root's final full suite, live temporary-database run, producer-to-viewer locale tests and wheel check supersede these intermediate counts. Source/check imports alone do not validate a fresh GPU reconstruction.

After web archival and Modal test isolation, the combined CLI, completion, provider-contract and model-family suite passed **66 tests in one process**. The fake Modal test replaces both Python module bindings (`sys.modules` and the parent package attribute), so prior completion imports cannot select a real remote runner.

## Regression identity and remaining validation

| Cell | Current MVS+fill/SAM3D publication | Revision | Boxes | Saved nativeToMeters |
| --- | --- | --- | --- | --- |
| 090 | `a9a6e0a0-a77e-4ca0-b460-aa6f18d72698` | `86207849-d672-4df4-b453-22584f491894` | 9 | `3.2371372068568487` |
| 030 | `fafdeb6b-7122-4434-a62d-676b3ff9e50a` | `f464b25e-9e27-4966-aca1-e21472fa2962` | 8 | `3.5616493184162774` |

`docs/STATE.md` and the frozen fixtures are authoritative. Historical `4b58…` / `cd84…` Pi3X/RecGen reports are not these acceptance references. `scripts.check_measurement_regression` compares object identities, source frame, scale, floor, box geometry, uncertainty and confidence exactly; root's original synthetic box computation fixture catches numerical drift. Display prose is replaced by the single schema-2 message representation and is checked through the real viewer consumers.

See [final acceptance](ACCEPTANCE-2026-10-08.md) for the fresh repository and environment checks. No fresh GPU/LLM run, input upload, deployment or customer-machine execution was made during this migration. A fresh paid reconstruction against the frozen inputs and the customer's GPU-service/environment acceptance remain external validation. The shortest customer change is to implement the existing two GPU request/result contracts or replace their invocation in `providers.geometry_mvs` / `providers.sam3d`; local CPU calculations and publication/measurement contracts remain the existing code.
