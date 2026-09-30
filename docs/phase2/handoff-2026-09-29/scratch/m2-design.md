# M2 one-shot refine runner: design

`codex/phase2-video` @ b56d5c4 · 2026-09-27 · CPU-only evidence. No GPU or Modal calls, nothing written under `runs/` or `data/`, and no `.platform/imports/*` file opened.

Tags used below: **(V)** I recomputed or checked it today from cached runs. **(R)** the value comes from a run manifest. **(E)** it is an estimate.

## 0. Summary

`scripts/run_video_report.py` turns an MP4 and a time window into a platform import with one command. It is built on three things:
- **Content-addressed stage directories.** A stage's key hashes its code version, resolved argv, the digests of what it consumes, the model pins and the versions of the decision rules it uses.
- **Decision stages.** Each hand decision becomes a small keyed CPU stage that writes a JSON value.
- **Three profiles:** `research`, `commercial`, and a frozen `delivered` profile used only to prove reproduction.

Seven findings shape the design:

1. **One rule set cannot reproduce all three delivered reports.**
   - They were built by three different processes.
   - They contain operator lists that no rule recomputes. The floor-mask frame lists are the main example: ME340 used every 75th frame, and Sam's Club and Walmart used the same hand list `0 66 135 … 747`.
   - They also contain defects the rules must fix, such as ME340's DROID bundle adjustment running over both shots.
   - So the 100%-cache-hit proof runs under a frozen **`delivered`** profile that is cache-only and never calls anything. `research` runs the M2 rules.
   - The difference between the two is an enumerated **deviation register** (§9), and a test asserts that the set of `research` cache misses equals the closure of that register exactly.
   - Consequence for the plan: "研究配置重现三组数字" (research reproduces 22/67/40) can only be proved through `delivered`. §11 U3 asks you to confirm this.
2. **Most hand decisions already have a rule that reproduces them (V).**
   - Mapping shot: pick the shot with the most keyframes from a full-clip DROID "census" run. Walmart gets 39 vs 28, Sam's Club 24 vs 8, ME340 18 vs 1/1. Path length would pick the wrong shot on Sam's Club (2.58 vs 2.21), and "the longer shot" would pick the wrong shot on Walmart.
   - SAM 2.1 frame list: census keyframes ∪ every 3rd frame. It matches all three exactly (312 / 267 / 298 frames).
   - Lens frames: `round(linspace(a,b,15))`. It matches Walmart and Sam's Club exactly.
   - Voxel: `round(0.04/m, 6)` gives 0.01399 / 0.00567 / 0.011858.
   - SAM 3.1 windows: a greedy rule with a 400-frame cap reproduces all three.
   - LingBot stride: `ceil(n/450)` gives 2 / 1 / 1.
   - LingBot confidence: the decile rule gives 1.06 (ME340) and 1.74 (Sam's Club); Walmart would get 1.01 instead of the 1.06 it copied from ME340.
   - Splat cleanup pick: "fewest Gaussians with held-out loss ≤ 0.2 dB" picks `negligible_1` on all three.
3. **The ICP acceptance caps must be per iteration (V).**
   - Replaying `lingbot_icp_refine.py` at HEAD gives **byte-identical** `dense-points.glb` for Walmart (268-icp) and Sam's Club (295-icp).
   - The largest single iteration is 0.84° / 0.041 m (Walmart) and 0.97° / 0.081 m (Sam's Club), measured as median point movement.
   - The totals, 2.26° (Walmart) and 0.67 m (Sam's Club), would both fail a 2° / 0.3 m cap applied to the whole refinement.
4. **Walmart fails the plan's lens gate (R).** Its floor plane inlier share is **0.884**, below the 0.90 in plan §5. ME340 has 0.959 and Sam's Club 0.965. See §11 U1.
5. **All three published documents carry a false limitation.** `import_video_scene.py:496` hard-codes "disagrees with the model scale estimate by about 20%". `model_estimated_metres_per_native_unit` is null for ME340, Sam's Club and Walmart.
6. **DA3 posed inference is one joint pass over all views** (`mono_room.infer_da3_remote`). ME340's 223 is a subset of 173, so its walk-only depth was still inferred together with the cut-away cameras. The runner never serves a subset of a joint stage. Each infer key gets a fresh directory, so mono_room's per-file "paid once" reuse cannot mix view sets.
7. **Two cache hazards the store has to route around:**
   - `lingbot_dense_map.upload_masks` reuses `{run_id}/dynamic-masks.tar` whatever `--masks` says.
   - `complete_video_objects` serialises generator calls (`ThreadPoolExecutor(max_workers=1)`), which blocks the ≤ 60 min target.

## 1. Command

```
python scripts/run_video_report.py --video MP4 --start S --end E --site NAME
       [--profile research|commercial|delivered] [--review DIR] [--dry-run] [--publish] [--verify]
```

- **No per-stage flags exist.** The only human input is `--review DIR`. It holds eye review data as `DIR/<site>.json`: `boxesApproved {entity: model.glb sha256}`, `boxesDropped`, and `entitiesExcluded {entity: reason}` with an `objectMapDigest`. The file's digest enters the key of every stage that reads it.
- **Environment:**
  - `PANOPTES_PAID_BUDGET_USD` is the run-level USD cap, following the existing platform convention: unset means no paid call, and the run stops before the first paid cache miss.
  - `PANOPTES_DATABASE_URL` and `PANOPTES_BLOB_ROOT` are needed by the import only.
  - The ART root is taken from `droid_room.ART`.
- **`--dry-run`:** no subprocess, no network, no writes. It prints the plan with a key, a hit/miss/unresolved/refused status, the directory, and estimated USD and seconds per stage, then the totals.
- **`--verify`:** recompute the full sha256 of every hit's outputs instead of checking size and mtime.
- **`--publish`:** after a successful import, run export → checks → prepare → deploy (§4, R38). It never runs otherwise.
- **Stdout and log echo** drop lines matching `capabilit|token|secret`. The runner never reads `.platform/imports/*`; it passes those paths and parses the import's own stdout, which already omits `capability`.

## 2. Module layout (new files only, plus the listed script fixes)

| File | Lines (E) | Part | Contents |
|---|---|---|---|
| `scripts/run_video_report.py` | 80 | A | argparse, then `report_runner.main()` |
| `scripts/report_runner/store.py` | 350 | A | `StageSpec`, key and digest functions, the `keys.jsonl` index, staging, `lock.json`, `Ledger`, executor (thread pool, timeouts), dry-run table |
| `scripts/report_runner/stages.py` | 400 | B | the M2 stage graph `graph(ctx)`, argv templates, `DEFAULTS`, `normalize()`, `versions.json` guard |
| `scripts/report_runner/decide.py` | 450 | C | decision rules D1–D21 as pure functions plus `python -m report_runner.decide NAME`, so each is an ordinary CPU stage |
| `scripts/report_runner/profiles.py` | 120 | C | `PROFILES`, the model pin and licence table, refusals |
| `scripts/report_runner/adopt.py` | 300 | D | the one-time adopt of delivered runs (§7) and the import fingerprint (read-only SQL) |
| `scripts/report_runner/publish.py` | 80 | D | export → check → prepare → deploy |
| `tests/fixtures/delivered-303/{me340,samsclub-a2,walmart}.json` | — | D | the three stage graphs from this workflow, machine-readable (§7.1) |
| `tests/test_report_runner_{store,stages,decisions,reproduce}.py` | — | A–D | one file per part; each skips when `$ART` is absent |

Runner state lives in `$ART/runs/report-runner/` (a new directory):
- `keys.jsonl` (append-only)
- `imports.jsonl`
- `adopted/<site>.json`
- `adopted/locks/<key>.json`
- `replays/`

New runs go to `$ART/runs/<site>-<stage>-<key[:10]>/`, and derived clips to `$ART/data/clips/<site>-<key[:10]>/`.

## 3. Store (Part A)

**Key.**
```
key = sha256(canonical_json({
  "stage": name, "version": int,                 # semantic version; tests/versions.json guards the deps sha
  "argv": normalize(commands), where
      a path from stage X -> {"in": role, "digest": consumed_digest(X, role)}
      a leaf file         -> {"leaf": sha256}     # MP4, review JSON, docs files
      dropped: --output, --invoke, --run-id, --reuse-build-from (compiled-build cache; run.json records its sha),
               budgets (--max-usd, --max-minutes), free-text notes and reasons (entity id lists and their order stay)
  "models": [(role, hf_id, revision | "unpinned", weights_sha256 | null)],
  "rules": {decision: "name@version"}}))
```
- **Consumed digest.** By default it is sha256 over the sorted `(role, file sha256)` pairs of the producer's declared outputs. Roles, not file names, are hashed, so an adopted `fov-check.json` and a new `lens.json` give the same digest.
- **Narrower consumed digests:**
  - DROID: the frames in `[a,b)` (sha256 of each) plus K and D.
  - LingBot prepare: the source video bytes plus the range.
  - Decisions: the decision JSON's `value` only.
- **Early cutoff.** Because downstream keys use output digests rather than producer keys, a rebuilt stage with byte-identical outputs leaves everything downstream a hit.

**Index.** Each `keys.jsonl` line holds:
- `{key, stage, site, dir}`
- `outputs {role: [relpath, size, mtime_ns, sha256]}` and `outputDigest`
- `verification` (`ran`, `replayed`, `recorded-rev`, `content` or `delivered-only`)
- `scope` (`["delivered"]` or `["delivered","research"]`)
- `lock`, `adoptedFrom`, `at`

Several keys may name one directory if their output roles are disjoint. For example, 281 holds both the infer outputs and the metric+fuse outputs.

A hit requires every output's size and mtime to match (or its sha, with `--verify`). A changed file invalidates the hit.

**Directories.**
- Each run gets a fresh directory (`exist_ok=False`).
- Staging is code: read-only inputs become **relative** symlinks, and files the tool rewrites are copied. Examples: `scene.json` for `mono_room dynamic`, and `plan.json`/`diagnose.json` for a LingBot build.
- A test asserts that every staged link resolves (ME340's `305/scene/mono` link is dangling).
- A failed directory is renamed `…-failed-<ts>` and never indexed.

**Lock (`lock.json` per directory; adopted locks go in `adopted/locks/`).**
```
{key, stage, version, depsSha256, git:{commit, dirty:false}, argv, models:[{role,id,revision,weightsSha256,licence,commercial}],
 image:{specSha256, registry, modalImageId|null}, gpu:{requested, used}, modal:{app, function},
 usd:{reserved, estimate}, wall_s, started, finished}
```
A stage whose dependency files differ from HEAD refuses to run: a result must name a commit.

**Budget and execution.**
- Before a paid stage starts, `Ledger.reserve()` sets aside its worst case: function timeout × list price × calls.
- Generators get `--max-usd = min(profile cap, remaining)`, and the splat gets `--max-minutes = min(profile, remaining / price)`.
- Prices are the ones already in code (`splat_train.USD_PER_S`, `video.MODAL_L4_USD_PER_S`); a test keeps the runner's table equal to them.
- Stages that are ready run concurrently in a pool of 4.
- Each subprocess times out at the Modal function timeout × calls + 300 s. Killing the client stops the ephemeral Modal app.
- When a stage fails, its dependents are marked `blocked`. The import runs if its required inputs exist; optional layers are left blank.

## 4. Stage graph (M2, research and commercial)

Notation:
- `[a,b)` is the primary shot. `others` are the other shots.
- All spans are canonicalised to end-exclusive `[a,b)`. Templates emit each script's own syntax: `START:END`, or `A-B` inclusive for splat `--skip`, `complete_video_objects --skip-frames` and `infer_room_floor --skip-frames`.
- Delivered directories are listed ME340 · Sam's Club · Walmart. `*` means the delivered node is a patch (a legacy structure, see §7.3).

| id | stage | template (abridged) | outputs (roles) | compute, est. | delivered |
|---|---|---|---|---|---|
| R01 | clip | `prepare_video_clip.py --video V --start S --end E --name SITE --output @new`, then `--full-video @new` | clip.json, rgb/, rgb.txt, source-rgb.mp4, source-full.{mp4,json} | L4 MoGe-3 on 5 frames, ~$0.01 | me340-165 · samsclub-337 · walmart-190 |
| R02 | cuts (D1) | `detect_shot_cuts.py --clip @R01 --output @new/segments.json` | segments | CPU 30 s | m0-integrate-cuts/<clip> |
| R03 | census | `droid_room.py execute --clip @R01 --run-id … --output @new` (whole clip) | prediction.npz (keyframes only are used) | A100-40, ~$0.03 | 171 · 157 · 168 |
| R04 | shots (D2) | decision | shots.json | CPU | new |
| R05 | overlay (D6) | decision over the clip frames | overlay.json | CPU ~60 s | new |
| R06 | lens (D4) | MoGe-3 (`moge3_app`) on `round(linspace(a,b,15))`. If the shot's FOV differs from the clip's by more than 3°, derive a clip: parent frames `[0,b)` hard-linked, MP4s written with `prepare_video_clip.playback/full_video` (as in `samsclub-337-a/made-by.py`), and the new K | lens, clip-ref | L4 $0.02 | none* · a2/shot-fov.json · 250/fov-check.json |
| R07 | camera | `droid_room.py execute --clip {R06 clip} [--frames a:b]` (omitted when `[a,b)` is the whole clip), then `collect` | prediction.npz, input-manifest.json, run.json | A100-40, ~$0.03 | 171* · 280 · 250 |
| R08 | floor masks | `discover_video_keyframes.py --video @R01/source-rgb.mp4 --frames F --prompt floor --provider self-hosted` | frame-*/ masks | L4 ~$0.02 | 174 · 159 · 170 |
| R09 | SAM 2.1 (D12) | `sam2_everything.py --video @R01/source-rgb.mp4 --frames kf(R03) ∪ range(0,N,3)` | object-a/ | L4 ~750 s, $0.17 | 184 · 160 · 181 |
| R10 | mask root | staging only: `floor-a → R08`, `object-a → R09/object-a` | links | — | 194 · 162 · 182 |
| R11 | motion cue | `motion_masks.py --droid-run @R07 [--exclude-frames others]` | motion.json, residuals | L4 ~$0.03 | 177* · 262* · 252 |
| R12 | tracks (D10) | per window: `sam3_motion_tracks.py --droid-run @R07 --motion @R11 --frames w0 w1` | tracks.{json,npz} | A100-40, $0.18 per window | 178-180* · 263,265* · 253 |
| R13 | stitch | if more than one window: `stitch_track_windows.py --tracks …` | stitched.json | CPU | 186 · 266 · — |
| R14 | analysis | `motion_tracks_to_analysis.py --droid-run @R07 --tracks … [--stitched]` | analysis.json, masks/ | CPU | 187 · 267 · 254 |
| R15 | dynamic masks | `assemble_dynamic_masks.py --droid-run @R07 --analysis @R14` | masks/ | CPU | 188 · 268 · 255 |
| R16 | depth infer | `mono_room.py infer --droid-run @R07 --da3-model P.depth --every 3 [--exclude-frames others]` | mono/, infer.json | A100-80, $0.06–0.10 | 173* · 281 · 251 |
| R17 | depth fuse (D5) | staging `mono → R16`; `metric --floor-masks @R10/floor-a --camera-height 1.6 [--exclude]`; `fuse --voxel-length-native V --support-relative .02 --support-all-views --edge-jump .03 --carve --dynamic-masks @R15/masks --video @R06clip/source-rgb.mp4 --floor-plane @out/metric-scale.json [--exclude]` | metric-scale, mesh, predicted-scene, supported-points, scene.json, fuse-metrics | CPU | 223* · 281 · 251 |
| R18 | dynamic layer | staging (`mono`, `droid-support` links; copies of metric-scale, infer, scene) then `mono_room.py dynamic --droid-run @R07 --analysis @R14` | scene.json, dynamic/*.glb | CPU | 190 · 285 · 262 |
| R19 | texture | `texture_fused_mesh.py --droid-run @R07 --fused @R17 --video @R06clip/source-full.mp4 --dynamic-masks @R15/masks [--overlay-rows]` | textured-scene.glb, texture-report | CPU | 224 · 282 · 256 |
| R20 | fill | `fill_scene_holes.py … --shell @R19 --step 2 --carve --video … [--overlay-rows]` | textured-scene.glb, fill.json | CPU (up to 29 min) | 225 · 283 · 257 |
| R21 | object map | `build_video_object_map.py --droid-run @R07 --depth-run @R17 --masks @R10 --floor @R17/metric-scale.json --dynamic-masks @R15/masks --method overlap` | object-map.json, surfaces/ | CPU | 195* · 284 · 259 |
| R22 | names | `name_video_entities.py --object-map @R21 --masks @R10 --per-request 10` (Gemini; `commercial`: Qwen3-VL) | object-map.json, names.json | cloud, ~$0.05 | 200 · 288 · 261 |
| R22b | static filter (D20) | decision: reclassify person-overlapping entities, clear person labels on static ones | object-map.json | CPU | new |
| R23 | frame outlines (D16) | `project_entities_to_frames.py --droid-run @R07 --scene @R17/scene.json --object-map @R22b --mesh @R20` | analysis.json | CPU | 201* · 289 · 263 |
| R24 | events (D15) | `video_events.py --video @R01/source-rgb.mp4 [--marks @R14/analysis.json]` | events.json | A100-40, ~$0.1 | 197 · 198 · 199 |
| R25 | LingBot infer (D7) | `lingbot_room.py prepare --video @R01/source-full.mp4 --stride ceil(n/450) --frames a:b`, then `execute` | plan.json, lingbot-run.json (+ volume) | H100/A100, $0.04–0.15 | 222* · 264* · 258 |
| R26 | LingBot diagnose | `lingbot_dense_map.py diagnose --run-id … --droid-run @R07 --clip … --masks @R15 --depth-run @R17 --overlay-rows (band ±2 or 0:0)` | diagnose.json | Modal CPU | 222 · 264* · 258 |
| R27 | LingBot build (D7) | staging (plan, diagnose copies); `build … --mesh @R17 --conf C --overlay-rows … --exclude-frames others` | dense-points.glb, points.json, attributes, remote.json | Modal CPU ~$0.05 | 222* · 286 · 258 |
| R28 | dense gate (D8) | `evaluate` into a new directory; if the gate fails, `lingbot_icp_refine.py @R27 @new @R17`, evaluate, gate again | dense.json {use, metrics, steps} | CPU ~5 min | 222 · 295-icp · 268-icp |
| R29 | inferred floor (D9) | `infer_room_floor.py --fused @R17 --dense {R28.use} --droid-run @R07 [--skip-frames]` | inferred-floor.* | CPU | 244 · 296 · 265 |
| R30 | splat | `splat_train.py --clip @R06clip --droid-run @R07 --masks @R15/masks --mesh @R17 --fill @R20 --scale @R17/metric-scale.json --captions (D6) --steps 60000 --cap 2500000 --pose --max-elongation 4 --skip others --max-minutes (budget)` | splats.splat, train/launch.json, refined-cameras | H100, $1.4 | 217/skip-…* · 287 · 260 |
| R31 | splat clean and pick (D11) | staging, `--clean` (all inputs passed explicitly), pick rule, `--pick R`, then package (link `splats-clean.splat` → `splats.splat`, copy the json/npz, write `candidates.json` with the rule) | splats.splat, splats.json, refined-cameras.* | GPU $0.09 | 232 · 294 · 266 |
| R32 | register each other shot (D3) | `register_cut_shot.py --droid-run @R07 --depth-run @R17 --clip @R01 --shot o --mesh @R17/mesh --model P.depth --invoke` | registration.json, da3-unposed.npz | A100, 1 call, ~$0.05 | 304 · — · — |
| R33 | other-shot movers (D3b) | if R32 is accepted: text-only person tracks on `o` (windows D10), analysis, then `register_cut_shot.py --registration @R32 --scene @R18 --analysis … --invoke` (no `--merge`, no `--person-tracks`) | scene/, analysis/ | A100 ×2, ~$0.3 | 305* · — · — |
| R34s | SAM 3D (D17) | `complete_video_objects.py --generator sam3d --droid-run @R07 --depth-run @R17 --object-map @R22b --masks @R10 --dynamic-masks @R15/masks --clip @R06clip --all --voxel-native V --skip-frames others [--no-captions] --exclude {review} --invoke` | models/, manifest | A100-80, $2.1–2.3 | 241 · 291 · 267 |
| R34r | RecGen (`research` only) | same inputs, `--generator recgen --entities {R34s rejects}`, journal seeded (D17) | models/, manifest | A100, $3.8–9.2 | 231* · 292 · 264 |
| R34b | box | same inputs, `--generator box --exclude {learned-accepted ∪ review}` | models/ | CPU | 302* · 301 · 300 |
| R35 | box test | `box_free_space.py --box @R34b` | box-test.json | CPU | m0-box-test-* |
| R36 | merge (D13) | `merge_object_models.py [--recgen] --sam3d --box --box-test [--review DIR/<site>.json]` | merge.json, models/ | CPU | 303-merged ×3 |
| R37 | import (D14) | `import_video_scene.py` with the resolved arguments (§5 D14); cwd `$ART` | DB + blobs + `imports.jsonl` sidecar | CPU + DB, 24–60 min | c40fbd08 · 913daf2a · 32cec650 |
| R38 | publish (`--publish` only) | `export_platform_publication.py --api http://127.0.0.1:8792 --publication P --output CAT/P`; `tests/check_publication_site.py --catalog CAT --source-api …`; `prepare_publication_site.py --catalog CAT --output HTTP`; `PANOPTES_PUBLICATION_CATALOG=CAT PANOPTES_PUBLICATION_HTTP=HTTP modal deploy modal_apps/video_publication_site.py` | catalog, deploy | — | — |

Critical path for the ≤ 60 min target (plan M2):
- The runner only adds concurrency between stages. The long stages are:
  - SAM 2.1: 12–13 min (R)
  - fill: 29 min on Sam's Club (R)
  - splat: up to 58 min (R)
  - serial generator calls: 154 SAM 3D calls on ME340 (R)
  - import: 24 min on Sam's Club, 60 min on Walmart (R)
- The target needs three changes outside the runner:
  - parallel containers in `complete_video_objects` (Part B)
  - a splat budget cap from the profile
  - a faster planar projection for box meshes in the import
- Every lock records `wall_s`, so the first full run measures the real critical path.

## 5. Decision rules and their tests

Each rule is a function `rule(**input paths) -> {"value", "evidence", "rule": "name@v"}`, run as a keyed CPU stage. "Delivered" is the value it must reproduce.

| # | Rule | Tested against (delivered) | Status |
|---|---|---|---|
| D1 cuts | `detect_shot_cuts.py`, unchanged | ME340 {14,226} → segments [0,13] [14,225] [226,898]; Sam's Club {420}; Walmart {383}; Lightning {} | passes (M0, R) |
| D2 primary shot | The shot with the most **census** keyframes (a DROID run over the whole clip; only its keyframe indices are used). Ties go to more frames. A shot needs ≥ 60 frames and ≥ 8 keyframes to be mapped, otherwise the report has no 3D layers. All other shots go to D3. | Walmart [383,749] 39 vs 28; Sam's Club [0,419] 24 vs 8; ME340 [226,898] 18 vs 1/1 | **V** |
| D3 other shots | `register_cut_shot` gate (centre ≤ 10%, rotation ≤ 3°, as in code). Accepted: the shot's cameras feed only the moving layer (R33). Refused: the shot is blank. `--exclude-frames` at import always lists every other shot. | ME340 [14,226): accepted, 0.92% / 2.38°, replayed from the cached `da3-unposed.npz` (CPU); centres within 5 cm / 2° of 304 (plan). Negative: Walmart frames against the ME340 map must be refused (1 GPU call, needs approval) | replay: build; negative: GPU |
| D3b cross-shot identity | Tracks never span a cut. People in other shots are text-prompted "person" tracks. Identities are **not** joined across shots (a blank beats a wrong join). | ME340's hand `--merge 178-text-1=178-text-0` becomes a deviation (X6) | build |
| D4 lens | MoGe-3 on `round(linspace(a,b,15))` of the shot. Keep the clip's K if \|median − clip FOV\| ≤ 3°; otherwise derive a clip (R06) with `fx = 320/tan(fov/2)`. Then gate on `plane_inlier_fraction ≥ 0.90`: if it fails, `scale_status = intrinsics_uncertain` (a new `contract_scale` status that allows no measurements). The gate only affects the import. | Walmart frames equal the rule exactly, 55.96 vs 54.06 → keep; Sam's Club frames equal the rule exactly, 55.3194 vs 60.08 → fx 610.5529 = a2's K. Gate: 0.959 / 0.965 / **0.884** | **V** frames and values; the gate fails Walmart (U1) |
| D5 voxel | `round(0.04 / metres_per_native_unit, 6)`, passed to fuse and to `--voxel-native` | 0.01399 / 0.00567 / 0.011858. ME340's model runs used 0.014 (X5). | **V** |
| D6 overlay band | Count still ORB matches where the pair's scene homography moves the point by ≥ 4 px. A band is 8-px raster rows holding ≥ 10% of those matches when there are ≥ 5 per pair. Map to video pixels (×1.5), pad 12 px, snap outward to the 8-px grid. Derived flags: texture and fill `--overlay-rows Y0:Y1`; LingBot `Y0-2:Y1+2`; splat `--captions Y0 Y1 160 1120` (x = the crop); objects keep the caption test on. No band: omit / `0:0` / `0 0 0 0` / `--no-captions`. Two or more bands: refuse those layers and leave them blank. | ME340: 11,865 such matches, 63% in raster rows 440–455 → video 660–684, which covers the text rows 661–689 → band ⊆ 640:712 and ⊇ 661:689 (delivered 648:704). Sam's Club 186 and Walmart 74 scattered matches → none. | probe **V**; exact 648:704 is not required (ME340 rebuilds anyway) |
| D7 LingBot | stride `ceil(n/450)`. conf = `round(conf_to` of the last decile in the leading run with `share_within_4pct < 0.90`, 2`)`. | stride 2/1/1; conf 1.06 (1.061) ✓, 1.74 (1.7373) ✓, Walmart **1.01** vs 1.06 (D4 in §9) | **V** |
| D8 dense display gate | Use the un-refined map if `within_25cm share (all points) ≥ 0.45` and \|floor offset\| ≤ 0.05 m. Otherwise ICP with a **per-iteration** cap: rotation ≤ 2° and median point movement ≤ 0.3 m (\|dt\| depends on the frame origin). Gate again; if it still fails, show no dense points. The floor offset is a new metric: the median signed distance of confident points within 0.3 m of the `metric-scale.json` plane over observed floor cells. | Sam's Club 286: 0.404 → ICP → 0.6155 ✓, max step 0.97° / 0.081 m. Walmart 258: 0.702 but floor −8.7 cm → ICP → 74% / 0.9 cm ✓, max step 0.84° / 0.041 m. ME340 222: 0.483 against the 189 mesh; must be re-evaluated against 223. Offset test: −8.7 ± 1 / 0.9 ± 1 / ~0 cm (from ARCHITECTURE §7 prose; the definition is fixed on Walmart and checked on Sam's Club). | ICP **V** (byte-identical replay, per-step logged); offset metric: build |
| D9 inferred floor | `--inferred-floor` only if `inferred-floor.json.kind != inferred_floor_withheld` | 244 kept; 296 and 265 withheld | R |
| D10 track windows | Greedy per shot: while more than 400 frames remain, emit `[s, s+300)` and advance 260; the last window takes the rest. Stitch when there is more than one window. | ME340 (whole clip) [0,300) [260,560) [520,899); Sam's Club [0,300) [260,420); Walmart [383,750). 400 is justified by the largest measured windows (379, 367). | **V** |
| D11 splat pick | Fewest Gaussians among cleanup rules whose held-out PSNR is ≤ 0.2 dB below `none` | `negligible_1` on all three (drops 0.100 / 0.049 / 0.137 dB) | **V** |
| D12 SAM 2.1 frames | census keyframes ∪ `range(0, N, 3)` over the whole clip. Frames outside the shot cost L4 seconds but are never views. | equals 184 / 160 / 181 (312 / 267 / 298 frames) | **V** |
| D13 merge | `merge_object_models.py` unchanged; boxes only with a review approval by sha256 | 22 / 67 / 40 with `box-review-303`; 20 / 46 / 31 without; 14 / 62 / 33 without RecGen | R (`m0-integrate-merge-replays.json`) |
| D14 import args | `--exclude-frames` = all other shots. Dense = D8. Floor = D9. Dynamic scene and analysis = R33 if a shot was registered, else R18/R14. `--request-suffix key[:12]`. `--republish` = the site's previous record path from `imports.jsonl` (path only). Title rule: `"{site} {mm:ss}–{mm:ss} (imported, not accepted)"`. | fingerprint (§8 P5) | build |
| D15 events marks | `--marks` = R14 whenever the analysis has entities | Sam's Club and Walmart had none (D) | — |
| D16 outline mesh | `--mesh` = the published fill shell (R20) | Walmart 257 ✓; Sam's Club used 282 (D); ME340 used 196 (X7) | — |
| D17 generator plan | SAM 3D `--all`, then RecGen on the SAM 3D rejects, then box excluding learned-accepted ∪ review excludes. Review `entitiesExcluded` applies to **every** generator. A missed generator stage seeds its journal from the latest run with the same (object-map, depth, masks, generator) digests, as in the 231 clone, so paid calls are never repeated. | Walmart object-156 is dropped → 39 (X11). Walmart's RecGen set changes but replays from its journal at $0 (the journal key must be verified in Part B). | build |
| D18 floor-mask frames | Operator data when adopted. Default: 12 evenly spaced frames on the every-3rd grid of the shot. | not reproducible by rule (every 75th; hand list) | O |
| D19 camera height | 1.6 m, `assumed_camera_height`, until a `measured_reference` exists | R | — |
| D20 person-overlap reclassification (plan §5) | An entity overlapping person masks by ≥ 30% in more than half its views moves to the dynamic layer. A static entity with a person label gets its label cleared. | Sam's Club: the worker leaves the static map. ME340 #32 "man" (a floor patch with 0–1% overlap) loses the label. | build (new) |
| D21 budgets | not part of any key; recorded in the lock; a partial result (cap reached) is shown as partial | — | — |

## 6. Profiles (Part C)

| | research | commercial | delivered |
|---|---|---|---|
| Rules | M2 (D1–D21) | M2 | adopted values (cache-only; any miss or refusal stops the run) |
| Depth (R16, R32) | DA3-GIANT-1.1 (CC BY-NC) | DA3-BASE (Apache-2.0); unposed camera output to be verified | adopted |
| Generators | SAM 3D, RecGen, box | SAM 3D, box (no `--recgen` at merge) | adopted |
| Dense map | LingBot-Map (code 849e690b, weights 204754b7) | **off** until Robbyant's licence is in writing | adopted |
| Naming | Gemini through `panoptes-report-workspace` (cloud, A19) | Qwen3-VL-8B (Apache-2.0), **new**, gated by an agreement test on the 150-crop set. Until it passes, names and models stay blank and the import still runs. | adopted |
| Refusals | none | licence False or unverified (DA3-GIANT, RecGen, `facebook/map-anything` non-apache, LingBot, and every "verify" row below) | any miss |

Model pins in the lock:
- **Pinned today:** DROID 2dfd39f0 (+ `weights_sha256` in run.json); SAM 3.1 daa63191 plus source commit; SAM 3D `MODEL_REVISION`; LingBot code and weights; gsplat 1.5.3; RecGen function `fu-Hh2leT3x1kprDaWpWsZ09l`.
- **Unpinned today:** DA3, MoGe-3, SAM 2.1, SAM 3 image, Qwen3-VL. Part C adds `revision=` constants for new runs. Adopted runs record `"unpinned"`; research accepts that for adopted runs only, and commercial requires pins.
- **Licences still to verify:** MoGe-3 weights, DROID weights, DA3-BASE camera head.

## 7. Adopt (Part D; one time, CPU plus read-only SQL)

`python -m report_runner.adopt tests/fixtures/delivered-303/<site>.json --video MP4`

**7.1 Fixture schema.**
```
{site, video_sha256, start, end, publication,
 nodes:[{id, stage, dirs, outputs{role: relpath}, commands:[[argv…]], basis{"--flag": "M|D|I|U"}, evidence, patch: null|"X1".., staging: bool}],
 decisions:{name: value}, deviations:[{id, class: "X|D|O", nodes, delivered, m2, reason}]}
```
Part D writes the fixtures from the three stage graphs this workflow produced. Sites are named `me340`, `samsclub-a2` and `walmart`, matching `box-review-303`.

**7.2 Steps, per site, in topological order.**
1. Hash the named `--video` file (open it, never list its directory) and compare it with `clip.json source.video_sha256`.
2. For each node:
   - Resolve the runner's argv under `delivered`, with inputs taken from already-adopted keys.
   - Normalise it (§8) and compare it with the fixture argv.
   - `M` arguments must be equal.
   - `D` and `I` arguments must be equal **and** pass the node's evidence check. Examples: the view-set recomputation (173 / 223 / 251 / 281), the fuse-metrics field mapping, the depth-ratio and tie recomputation (304 / 305), `published_surface_ray_hit = fill.coverage_after`, and the carry-height median over shot frames (Walmart 0.47430).
   - `U` arguments are allowed only for the import's `--title/--republish/--request-suffix` and ME340 events `--marks`, and are listed.
   - Staging nodes (194 / 162 / 182, 189 / 190 / 223 / 285 / 262, 232 / 294 / 266, and 286's copies) are checked by output equality: link targets resolve to the owner's directory and copies are byte-equal.
3. Hash the declared outputs (roles → sha256), write a lock to `adopted/locks/`, and append the **delivered** key with `verification` set.
4. **Research key.** Also append one when two conditions hold:
   - The node's M2 resolution equals the delivered normalised argv.
   - Its code is verified current by one of:
     - `recorded-rev`: the manifest's revision or script sha equals the versions.json entry, e.g. DROID `source_revision`, DA3 `script_sha256`, LingBot `code_revision`.
     - `replayed`: a CPU re-run at HEAD into `runs/report-runner/replays/` gives equal outputs. This applies to cuts, the ICP (done today), the merge (`--against`), register-from-journal, the inferred floor, and the frame outlines.
     - `content`: the clip, checked by re-decoding sampled frames within max abs difference 1.
5. **Decisions.** Run every D-rule whose inputs are adopted, store `decisions/<name>.json` under its research key, and compare it with the fixture's `decisions`. Differences must be listed in `deviations`.
6. **Import.**
   - Find the record by file name `video-import-<publicationId>.json`; list names only, never open it.
   - Read the published document with `SELECT document FROM scene_revisions …` (read-only, port 54329, `panoptes_video`).
   - Compute the fingerprint (§8 P5) and write `imports.jsonl {site, key, publicationId, projectId, importRecordPath, fingerprint}`.
   - Look up the latest record path of the project, by `project_id` from `publications` (SELECT), as the next `--republish` target.

**7.3 Patches (delivered-only).** These nodes have no M2 template, and their argv comes from the record, flagged as tautological:
- ME340: 173 (all views), 189, 196, 213, the 223 subset staging, 217 (seeds 189 + 213), the 222 whole-video run, 231 (clone + reassess), 242 (feeds the box excludes), 302.
- Sam's Club: the old-K chain 260 / 261 / 262–268, 264 (prepared from CLIP_A), CLIP_A / a2 made-by, 293.
- Walmart: 268-merged.

Every patch carries a deviation id.

## 8. Proof of reproduction

**Normalisation (both sides go through the same function):**
- Expand `$ART` and `$REPO`.
- Canonicalise spans to `[a,b)`.
- Fill defaults from `stages.DEFAULTS`. An AST test checks it against each script's `add_argument` defaults, including the ME340 defaults hard-coded in `splat_train`, `lingbot_dense_map` and `complete_video_objects`.
- Drop the key-excluded flags and note texts.
- Map each path to (producing node, role) by resolving symlinks to their owner, and each leaf to its sha256.
- Sort flags; keep list order.

| Check | Pass condition |
|---|---|
| P1 argv | Every non-patch, non-staging node: normalised runner argv equals the fixture argv. The test prints counts (template / patch / staging / U) and fails on any unlisted difference. |
| P2 hits | `--profile delivered --review docs/phase2/box-review-303 --dry-run` on the three MP4s (165–195, 337–367, 190–220) shows every node a hit, 0 calls, $0.00. With `--verify`, the full sha256 of every output matches. |
| P3 decisions | Each D-rule computed from cached evidence equals the delivered value (§5), or the difference is in the register (§9). |
| P4 replays | `merge --against` exits 0 for all three (R); the ICP replay is byte-identical for Sam's Club and Walmart (V); cuts equal; `register_cut_shot` from its journal equals 304; the inferred-floor kind is equal. |
| P5 import | Run `build_document` offline with the resolved R37 arguments and a stub asset store (as the recorded Sam's Club check N50 did, 1439 s). Its fingerprint must equal the published document's, computed as sha256 over: sorted asset sha256s; sorted (entityId, label, associationState, model sha); counts of cameras and observations; annotation kinds; `leftOutOfReport`; the scale record; the title. The only allowed field difference is `limitations` (X13) once the Part B fix lands. Expected: ME340 124 entities / 123 confirmed / 22 models, Sam's Club 67 models, Walmart 40. |
| P6 negatives | Point a copy of the index's DROID entry at a byte-changed `prediction.npz`: the run is refused. Change an approved sha in a review file: that box is dropped. An unknown CLI flag is rejected. The dry-run with `subprocess` and `socket` patched to raise still completes. |
| P7 research diff | `--profile research --dry-run`: the set of misses equals the closure of §9 over the graph, as asserted by a fixture per site. |

## 9. Deviation register (delivered vs M2) and research rebuild cost

Classes: **X** is a delivered defect the rule fixes. **D** means a rule exists and gives another value, while the delivered value was not wrong. **O** is operator data (adopted, and kept in `research` for known clips).

| id | site | delivered | M2 | class |
|---|---|---|---|---|
| X1 | ME340 | DROID bundle adjustment over the whole clip (keyframes 14 and 226 sit on the cuts) | camera `--frames 226:899`; [0,13] and [14,225] go to D3 | X |
| X2 | ME340 | metric over all 899 cameras; view 150 (cut-away) in the plane → 2.8592 m/unit | `--exclude-frames` → 2.7946 (−2.3%) | X |
| X3 | ME340 | walk depth = subset of a joint 312-view DA3 pass | fresh per-shot infer | X |
| X4 | ME340 | object map, names, models, splat seeds and LingBot evaluate on the all-view 189 | per-shot R17 | X |
| X5 | ME340 | model voxel 0.014 | 0.01399 | X |
| X6 | ME340 | motion pairs and track window 0 span the cut; hand `--merge` / `--person-tracks` | tracks per shot, no cross-shot join (D3b) | X |
| X7 | ME340 | outlines occlude against 196 (texture of 189) | fill shell | X |
| X8 | ME340 | LingBot on the whole video, stride 2, no `--frames` | per-shot | X |
| X9 | ME340 | lens from 5 frames, one of them cut-away (80.57°) | 15 walk frames; expected kept (walk frames 79.4–81.3°) | D |
| X10 | Sam's Club | motion cue and tracks on old-K 260 (60.08°) | on 280 (masks are image-space; early cutoff possible) | D |
| X11 | Walmart | box for object-156 (a display on the moving cart, source frame 657) was approved | review excludes apply to all generators → 39 | X |
| X12 | Walmart | plane inliers 0.884 claimed as `assumed_camera_height` | `intrinsics_uncertain` (U1) | X |
| X13 | all | false "~20% disagreement" sentence | the sentence only when a model estimate exists | X |
| D1 | Sam's Club | outline mesh = texture 282 | fill 283 | D |
| D2 | Sam's Club | LingBot on the re-encoded CLIP_A video; diagnose on 260 / 261 | parent video `--frames 0:420` on 280 / 281 | D |
| D3 | SC, WM | events without marks | marks from R14 | D |
| D4 | Walmart | LingBot conf 1.06 (copied from ME340) | 1.01 (its own deciles) | D |
| D5 | SC, WM | other shot not attempted | R32 attempt (expected refused) | D |
| D6 | all | box run excludes from an intermediate merge | learned-accepted ∪ review excludes | D |
| O1 | all | floor-mask frame lists, entity notes and order, excludes, budgets, titles | adopted for these clips | O |

Estimated research rebuild cost from recorded spend (E):
- **ME340:** full rebuild from R07, about **$9**.
- **Sam's Club:**
  - If R15 comes out byte-identical (early cutoff): about **$0.4** plus CPU (LingBot, events, registration, outlines, import).
  - Otherwise a full object rebuild, about **$8**. In that case the 21 eye-approved boxes stay off until they are reviewed again, because the review binds to mesh bytes.
- **Walmart:** about **$0.25** (events, registration, LingBot build) plus CPU: merge → 39, ICP gate, floor, and a 60-min import.
- **All three:** import again with the fixed limitation, as republished versions, local only.

## 10. Build split

| Part | Owns | Interface it provides | Tests (done when) |
|---|---|---|---|
| **A: store, executor, CLI** | `run_video_report.py`, `store.py` | `StageSpec(name, site, version, deps, commands, inputs{role: (producer, roles)}, consumes, leaves, outputs{role: relpath}, stage_from{link, copy}, compute, gpu, timeout_s, worst_usd, est_usd, est_s, budget_flags, models, env)`; `Store.key/lookup/record/execute/plan`; `Ledger.reserve/settle`; `Hit(dir, outputs, verification)` | a toy three-stage graph: stable keys; budgets and output path excluded; early cutoff; an edited output invalidates the hit; over-cap refused; dry-run makes no subprocess or socket call; dirty deps refused; staged links resolve; redaction; concurrency and failure isolation |
| **B: stage table and script fixes** | `stages.py`, `versions.json`, fixes in: `lingbot_dense_map` (masks tar keyed by digest; stale "me340-filled-213" text); `complete_video_objects` (`inputs.clip` repr bug at :1076; parallel workers); `motion_tracks_to_analysis` (text tracks keep their prompt as label); `sam3_motion_tracks` (text-only mode, verify); `import_video_scene` (X13); derived clip in `prepare_video_clip` (`--derive PARENT --frames 0:b --fov-deg F`, reusing `playback`/`full_video`) | `graph(ctx) -> list[StageSpec]` (raises `Pending(decision)` until a decision resolves); `normalize(script, argv, owner_of)`; `DEFAULTS` | AST defaults check; every template resolves on the adopted directories; `--derive` rebuilds a2 byte-equal (source-rgb 9ffd4381, source-full 8a5faee4) or the test documents why not; generator journal-key check |
| **C: decisions and profiles** | `decide.py`, `profiles.py`, overlay detector, lens stage, D8 (floor offset metric, ICP per-step log and caps in `lingbot_icp_refine.py`), D20, `contract_scale` status `intrinsics_uncertain`, HF revision pins, Qwen3-VL naming (`commercial`) | `RULES[name](**paths) -> {value, evidence, rule}`; `PROFILES[name] -> Profile(models, generators, dense_map, namer, cache_only, rules)` | every §5 row reproduces its number from cached runs; commercial profile refuses unverified licences; the naming gate is recorded |
| **D: adopt, proof, publish** | `adopt.py`, `publish.py`, fixtures, `test_report_runner_reproduce.py`, fingerprint (read-only SQL) | `adopt(fixture, video, store) -> AdoptReport`; `fingerprint(document) -> str`; `publish(publication_id)` | P1–P7 pass for the three sites |

Order:
- A, B and C start together against the interfaces above.
- D starts in parallel with fixtures, the fingerprint and the publish chain, and integrates last.
- No part renames existing contract fields (COMPONENTS rule: add fields only).

## 11. Decisions needed from you

1. **U1 (Walmart lens gate).** Walmart's plane inliers are 0.884, below 0.90. Keeping 0.90 downgrades only Walmart's scale claims to "needs review"; the geometry does not change. Lowering the threshold to fit Walmart is tuning to pass. **Recommendation:** keep 0.90.
2. **U2 (research rebuilds).** Approve the research rebuild costs: ME340 about $9; Sam's Club $0.4–8 (the higher end also means re-reviewing 21 boxes); Walmart about $0.25. The alternative is to hold on `delivered` until the new rules are needed.
3. **U3 (plan wording).** The M2 acceptance line "研究配置重现三组数字" (research reproduces 22/67/40) should read: "the delivered profile reproduces 22/67/40 from cache; research reports its own numbers".
4. **U4 (Walmart models).** 40 → 39 (drop the object-156 box).
5. **U5 (GPU approval).** Two small calls: the Walmart-into-ME340 negative control (1 DA3 call) and the ME340 walk lens estimate (1 MoGe call). Both are needed only for the research proof, not for `delivered`.
6. **U6 (commercial naming).** Commercial profile names and models stay blank until Qwen3-VL naming passes its gate. Alternatively, allow Gemini in commercial.
7. **U7 (≤ 60 min).** The target needs generator parallelism, a splat cap and a faster import. The runner alone does not reach it.

## Evidence computed today

All paths are in the scratchpad `/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/`:
- `icp/{wm,sc}/points.json` and `steps.json`: ICP replays at HEAD. `file_sha256` equals 268-icp and 295-icp; `steps.json` holds the per-iteration steps.
- `icp/icp_logged.py`: the instrumented copy of `lingbot_icp_refine.py`.
- `overlay_probe2.py`: the D6 probe.

The other checks (census keyframes, SAM 2.1 frame lists, lens frames, voxel, windows, LingBot confidence, splat pick) were computed in memory from `runs/*` manifests.