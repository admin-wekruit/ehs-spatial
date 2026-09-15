# Report CAD and model visibility — 2026-09-15

Publication: `5cf6429b-8ccd-4ab3-8153-fde364bf4b5b`; scene revision: `e211fcc8-ab63-4832-9fcc-b1604418d0e0`.

The report CAD accidentally reused the 3D representation selector. Selecting the model scene therefore replaced 28 saved observation contours with the 9 active model projections. No source geometry had been deleted. Report and workbench CAD now use the fixed observation reference states independently of the 3D selector. The workbench's distinct model plan still follows model edits. Unregistered historical source CAD is not merged into the current coordinate frame.

The isolated object preview reused the main scene's dark background and multiplicative lighting, obscuring dark materials. It now uses a light neutral background and preview-only shadow lift/key/fill lighting. Source vertices, textures, material values, transforms, and placement states stay unchanged; scene navigation and picking are restored after capture.

## Verification

- Executed actual report controls with the frozen publication document: 28 identical source contours across all three photographs and all three 3D display modes. Actual model coverage remains nine. Model edits change model projections and leave source CAD intact.
- Executed workbench control regression, viewer capture/draw and interaction checks; TypeScript and production build passed.
- Computer Use on the local report: all nine mode/photo combinations retained the same 28 CAD entity IDs and the selected cart. Inspected enlarged cart, white fence side view, and yellow/black guard previews. Dark chassis and colored/metal parts remain visible; mesh textures remain intact.

This is a presentation fix. It does not generate the 18 missing independent models, confirm placements, calibrate the scene, change the archived 33-record source CAD, or modify any publication snapshot.
