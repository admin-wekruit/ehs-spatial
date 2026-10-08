# Single-candidate generation

`ehs_spatial.object_generation.generate_candidate` reuses the existing research exporter, RecGen driver, floor fit, assembly, comparison renderer, and lossless per-object packer. It does not call a language model, infer cross-view identity, or select additional candidates for generation.

```python
result = generate_candidate(
    source_run, exact_candidate_id, new_job_directory, research_repo,
    gpu_budget_seconds=0,  # cache-only; live call requires >= 630
    cached_generation_root=existing_generation_directory,
    environment_record=verified_environment_json,  # required for a new GPU call
    python_executable=research_python,
    viewer_repo=public_viewer_repository,
)
```

The job directory must be outside the source run and must not exist. A job has `status.json`, per-stage logs, `input-provenance.json`, and a derived `run/`. The full 2D evidence registry remains in `run/object-evidence.json`. The exporter preflights only the selected candidate; `run/evidence/objects.json` contains exactly that ID. Unselected candidates are not reported as generation failures.

Observed floor input comes from the source run's SHA-bound `scene.json` native fitted plane and the same frames' original depth, confidence, validity, cameras, and photos. It does not require a semantic `floor` label. The existing floor-fit residual gate selects actual supported canonical pixels; camera orientation, positive height, at least 200 support pixels and nondegenerate planar support are checked before triangulation. Missing or inconsistent evidence produces `blocked` with its exact reason, before GPU use. The result remains estimated observed geometry with unknown metric scale; no infinite plane, filled holes, or independently verified floor identity is introduced. The source scene SHA is part of the evidence revision, so changed floor inputs allow an explicit retry without changing object identity.

`status.state` progresses through validating, exporting, fitting_floor, cached or generating, assembling, rendering_comparisons, adding_context, packing, and complete. `blocked` denotes insufficient evidence, missing exact cache with zero budget, or missing configured environment. `failed` records a failed stage with a local log. Completed status includes `result_scene`, `result_viewer`, `result_metrics`, `result_glb`, and hashes for every result file. Native generation files and their recorded input/model/asset hashes remain available. No source run file is written.

`adding_context` reuses the generated object's exact native anchor-frame pointmap, valid pixels and colors to triangulate a read-only workcell background, excluding generated-object masks. It verifies the native record and source hashes, makes no model or pose-fitting calls, and preserves existing mesh and comparison bytes. Full original photographs are included with `original_K = inverse(input_to_canonical_pixel_centres) @ K`; canonical camera fields remain unchanged. Object links open the original photograph with projected selection axes. Free 3D shows the anchor-frame observed context without fusing other capture states or filling unseen surfaces. Context-only updates of completed jobs create a new result revision while retaining the entire prior job and its artifact hashes.

Cache replay requires exact candidate ID, payload bytes, seed, pinned RecGen code/model revisions, original source-record hash, and all generated asset hashes. A same label or inventory number does not authorize replay. The source snapshot is checked again after assembly. An independent new job is required for another attempt; failed jobs remain inspectable.

For new inference the configured environment check must already match the model pins. The existing driver reserves 630 seconds before upload/remote execution and uses its 600-second Modal function timeout plus overhead allowance. Failed or interrupted calls retain their reservation; budget and call accounting are saved in `run/generation/gpu-budget.json`. The displayed limit is an execution budget, not a currency quote. The caller owns durable scheduling and authorization; the helper cannot silently generate all candidates.

Set `LUCIDA_DEPLOYED_APP=lucida-private-assets` in a deployed CPU worker to call the existing deployed `generate_object` via `modal.Function.from_name`. The local driver retains its normal `app.run` mode. Payload preparation, the private Modal volume, inference budget, hash validation, and output retrieval use the same code in both modes.

The research Python runtime requires the already-used NumPy, SciPy, Pillow, OpenCV, trimesh, Open3D and Modal packages, plus the EHS package dependencies used by export/floor fitting. The web worker can pass this interpreter explicitly instead of importing geometry libraries into a request handler. No new dependency is added to the EHS web package by this wrapper.

A real cache replay completed on BOR1's `object_6dd85942d104987a83c677d6` with zero new GPU calls. Its model input SHA was `a97639218931f1b250cc258d072d48afd12e22f95064aabc4cb15d6d6b2646e2`, identical to the original native RecGen call. The result contained the requested object and observed floor, with per-view initial/refined metrics and lossless individual compressed meshes. This validates execution and provenance, not geometric accuracy, metric calibration, or functional safety. RecGen's existing non-commercial licensing still applies; its native output records retain the license metadata.

Small automated checks: `.venv/bin/python -m pytest tests/test_object_generation.py tests/test_report_refresh.py tests/test_workcell_surface.py -q`.
