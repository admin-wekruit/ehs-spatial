# Workcell model delivery audit

The browser-visible model requirement is **not fully met**. The current BOR1 090 scene retains all 28 object records. It has 24 displayable current models, one observed floor reference, one composite observation that previews its reviewed existing components, and two objects whose generated shapes were rejected. These categories are mutually exclusive; the latter two must not be hidden or counted as models.

## Accepted work

Six additional RecGen research outputs were accepted for browser inspection: the emergency-stop button, both overhead signal lights, the rear-right fence panel, the small unidentified indicator and the left-fence poster. They retain fixed inputs, provider versions, original meshes and unconfirmed placement. The poster's generated lettering is not authoritative source text. Existing parametric models, measured visible surfaces and inferred transparent planes keep their own provenance; this is not a claim that all 24 are complete generated solids.

The rear panel of the left fence is now an individually selectable component of its existing model. Its 135,999 triangles and the parent's remaining 155,219 triangles form an exhaustive, disjoint partition of the original 291,218 triangles. Source vertices, colors, normals, winding and placement were retained. Photos 2 and 3 establish the component identity; they do not establish calibrated physical placement. The floor stays an observed reference without invented thickness or equipment axes. The mixed cart/guard observation previews the already-reviewed component models instead of generating a fictitious composite object.

## Remaining shape gaps

| Record | Result and next requirement |
| --- | --- |
| #12 adjacent-station fence | Corrected visible-body input still generated background red machinery as a hood and opaque panels in see-through areas. Rejected; no current model. Requires a shape result consistent with the actual fence, not another acceptance threshold change. See [body audit](2026-09-15-recgen-panel12-body-quality.md). |
| #23 wire cable tray | Corrected erosion, independently reviewed second-view mask, exact intrinsics and narrow real-image crops still generated a broad railing instead of the narrow tray. Rejected in both source views; no current model. See [two-view audit](2026-09-15-recgen-wire23-multiview-quality.md). |

Arithmetic agreement between native transforms and exported vertices is verified separately from shape and physical placement. All 24 current model placements remain unconfirmed; native scale remains uncalibrated. The fixed RecGen route is a noncommercial research preview, not a commercial model release.

## CAD and other photographs

Observed CAD covers all 28 current records. This is separate from the historical drawing's 33 source records: only nine have explicit current links, 24 remain unresolved, and the historical drawing is not geometrically registered. Neither count proves the other.

The model-CAD audit exposed 11 missing caches after imports/partitions and one stale cache after a pose correction. The shared triangle-union projector, with the frozen scene asset layout and current pose, refreshed those projections. Partition creation now invokes that same projector; floor references validate their own source observation. Actual frontend helpers now return 24 model projections plus one floor reference in the model layer, and all 28 observed projections in the observed layer. Independent recomputation from actual mesh faces produced the same projected bounds. Hash, pose, source-image, observation-revision and ownership tampering are rejected. Bounding-box hulls and artificial hexagons are not used to fill missing contours.

The other imported photographs were also scanned. The old four-photo capture mixes BOR1 030 and BOR1 090. Revision `7dc381ca-a426-4a70-86ba-ff0dc72c0130` split an incorrectly linked bollard identity across the two stations while preserving its historical observations/assets. The 090 photographs duplicated in older captures do not need new inference solely because their JPEG resolution differs. After that split, BOR1 030 has 54 entity records with 55 observations, only two old models, and unresolved within-station identity work; these are not 54 verified physical objects. It has **not** been completely modeled by this batch.

## Budget and publication evidence

The authorized cap is USD20. Fifteen model calls have USD15 accounted reservations, including failed/rejected attempts and the earlier uncertain outcome; USD2 is held for other task costs. Full actual billing remains unknown. Reservations are not reported as actual charges. No further paid retries are started for the two rejected inputs.

Final immutable publication: `336ff87d-4336-4280-8348-2120d594cb7b`, scene revision `a9ec2a62-1c48-4ae4-a6cd-6807ec8b71d3`, document SHA-256 `ad27023461364790013934614c28991c4e44231756508d315a87b548a37b7675`.

The final manifest's 405 assets (542,039,483 bytes) were verified. The full 12-publication catalog passed 694 exact source-API response comparisons and 650 unique historical-asset byte checks. Four concurrent reads passed; measured local peak RSS was 2,800,664,576 bytes. The publication server now reserves 6 GiB with four concurrent inputs. This measurement excludes the Modal runtime.

The deployed public API returned the exact frozen publication with SHA-256 `75e7ea00af9711c3ad4e941784c5157a6d13adb8e2659b630d4a32e353e67ac0`. Its initial request took 69.9 seconds; this is not a claim of fast cold loading. Publication history and existing feedback storage were preserved.

CAD selection also had an independent UI defect: assigning a button to the entire geometric group made its center hit overlapping objects. The shared control now binds its accessible button to the actual numbered label; geometric hit testing retains overlap handling. A runnable regression covers overlap/hole centers, keyboard selection and drag suppression. Computer-use checks reproduced the #28-to-#25 misselection before the fix and selected #28 correctly afterwards.

Build, TypeScript checks, reconstruction tests (40), partition regressions (4), interaction checks and the CAD label regression passed. Private reproducible outputs are under `.platform/model-correspondence/model-completion/final-delivery/`; no management capabilities are included in this document.

The frontend is live at [the fixed publication](https://admin-wekruit.github.io/panoptes-workcell-report/app.html#/reports/336ff87d-4336-4280-8348-2120d594cb7b). Pages commit `1df6753ee354b98a6eeebe65396e48a32ae220ed` passed [deployment run 35056968836](https://github.com/admin-wekruit/panoptes-workcell-report/actions/runs/35056968836), and public HTML serves `app-DTpoUx7y.js`.

Public computer-use verification observed all 24 current models loaded, all 28 observed CAD projections, and the explicit 1 reference / 1 composite / 2 missing classification. Clicking CAD label #28 selected entity `5a62a2d3-0216-512d-b384-3eadd69d3e9e`, with the corresponding orange photo contour, cyan scene highlight and independent rear-fence panel plus XYZ axes. Selecting #16 loaded its actual emergency-button mesh and source-photo selection. The page was left with no object selected. This verifies those UI interactions and asset loading, not accurate placement or completion of the two rejected models.

Known delivery limits remain: slow first report loading, all model placements unconfirmed, incomplete historical-CAD registration/linkage, two missing qualified shapes in BOR1 090, and incomplete modeling of the separate BOR1 030 scene.
