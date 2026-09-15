# Object model scene and Blender delivery

The model view and Blender export now use the same active object representations.
Observed surfaces remain a separate photo evidence view. They are not counted as
independent models or substituted into an incomplete model scene.

## Fixed revision

- Project: `a2b7c04d-0162-488b-b7db-37711a37ea62`
- Reconstruction revision: `e211fcc8-ab63-4832-9fcc-b1604418d0e0`
- Publication: `5cf6429b-8ccd-4ab3-8153-fde364bf4b5b`
- Export job: `2f7f3f98-7b8c-403b-a38a-373b02b1dd52`
- 28 entity records and 84 photo observations retained.
- 9 independent models: 7 existing generated meshes and 2 restored parametric cylinders.
- 18 other entity records lack an independent model; the explicitly classified
  floor remains an observed reference surface and does not require a solid model.
- All 9 model placements remain unconfirmed. File validation passed; the export
  job is correctly `incomplete` because model coverage and placement remain incomplete.

The cylinders were restored from the hash-pinned imported source and
`blender/post-parameters.json`, not inferred from their labels. The converter
checks vertex agreement and complete surface-facet coverage, allowing alternate
triangle diagonals but rejecting missing or duplicate surfaces. Maximum native
vertex differences were 6.29e-8 and 4.49e-8. Their source meshes, measurement
evidence, IDs, observation links and source CAD references are retained.

The generated `scene.glb` is 68,960,120 bytes, SHA-256
`81f450fdea42aa87d05490ef274fd0cf5bc0e133982af4b1e4c3be04d27953d0`.
The generated `scene.blend` is 195,069,030 bytes, SHA-256
`a920bdd0a027df33a91c37c6f7f03adecbfaa8f52e02a8ba65da4948ed8e931f`.
Blender 4.5.9 reopened the file and verified 9 objects, source cameras, transforms,
materials and both cylinder parameter drivers. This checks file consistency,
not physical measurement accuracy.

## Editing and report behavior

The report defaults to the model scene when available. The fourth pane displays
the selected actual model with front/side/top/free camera choices, using the
existing viewer's GPU buffers. Capture restores scene navigation before yielding.
CAD remains the scene-wide floor projection. Model mode projects 9 model assets;
photo evidence mode retains all 28 observed entity projections.

Changing transform or primitive parameters revokes placement confirmation.
Only explicit `confirmPlacement` on the active model records acceptance of the
current pose. Parameter edits update bounds and invalidate the old projection.
GUI and Agent use the same revision operations. Visitor previews cannot create
persisted placement confirmation.

Browser QA found and corrected a real preview cropping defect: the image exceeded
its 233 px container with a 436.875 px intrinsic height. The image now fits the
container. Actual local UI checks covered default empty selection, linked cart
selection, enlarged model and side view, missing-model button selection, and
switching to the complete observed CAD without losing that selection.

## Persisted edit and export check

An unpublished planning branch `09c394e8-4b48-4957-af33-de7460eecc2b` was created.
Final test revision `721c9d64-44b5-49c0-8f54-ae95dff8534a` changed the left cylinder
radius to 0.061026123947167846 native units and moved the cart X by 0.125.
Job `c57ebdbf-6844-428e-8618-0569d4cfe106` exported it and independently reopened
Blender. Parameters, regenerated bounds, driver edits and GLB transforms matched;
observations and the reconstruction head stayed unchanged. Before/after evidence
is under `.platform/model-delivery-20260915/edit-check/`.

Executable validation:

```sh
PYTHONPATH=. .venv/bin/python -m pytest tests/test_platform_identity.py tests/test_platform_import.py tests/test_platform_spatial.py tests/test_agent.py tests/test_agent_hub.py tests/test_sam3d_preflight.py tests/test_sam3d_runtime.py -q
cd web
node checks/model-view.mjs
node checks/report-scene.mjs
node checks/identity-review.mjs
node checks/workspace-selection.mjs
node checks/observed-plan.mjs
node checks/scene-plan-reference.mjs
npm run check
npm run build
```

85 Python tests passed (four existing synthetic ray fixture warnings). Listed web
checks and production build passed. The exported publication catalog passed 601
exact read responses and 562 byte-verified assets across 9 immutable publications.

## Remaining model release gates

No new paid inference ran. Existing RecGen assets were preserved; their
noncommercial provenance does not establish a commercial model release.
The SAM 3D worker now disables internal depth construction before Hydra recursively
instantiates the pipeline. Runtime preflight and regression checks are available,
but the pinned GPU runtime, weight access, explicit paid budget and actual
pose/pointmap release experiments are not yet satisfied.

Static inspection of official SAM 3D revision
`f91db411c50efee93d8db7aeb323885650f6f722` found unconditional Gaussian renderer
imports containing INRIA noncommercial notices and Gaussian decoder construction.
Requesting mesh output alone does not prove the dependency/license gate passed.
The hash-pinned source audit and preflight evidence are stored under
`.platform/model-delivery-20260915/`. No alternative model route was substituted.

The public deployment is the existing read-only publication service plus scoped
feedback. It is not the full writable SaaS/GPU deployment. Local project editing,
durable versions and export were validated separately as described above.
