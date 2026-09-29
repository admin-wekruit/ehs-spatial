# M3 E8: import profile and cloud write path

`docs/phase2/FAST-PATH-PLAN.md` section 7, E8. Pass: find the import's bottleneck; the prototype writes one layer in 10 s or less.

Run from the repo root with the platform venv's Modal CLI (`/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal`).
Everything is an ephemeral `modal run`: nothing is deployed, and no URL is public.

| Step | Command | Writes |
|---|---|---|
| Stage inputs (once, about 1.3 GB) | `modal run experiments/m3_e8_import/probe.py::stage` | Volume `panoptes-m3-e8-import` |
| (a) Profile the import | `modal run experiments/m3_e8_import/probe.py::profile` | `runs/m3-exp-e8-import-profile-N/profile.json` |
| (b) Write-path prototype | `modal run experiments/m3_e8_import/probe.py::layer` | Volume `panoptes-m3-e8-layers`, `runs/m3-exp-e8-import-layer-N/layer.json` |

## (a) Profile

- **Inputs:** the argv that published `c40fbd08` (ME340), taken from `tests/fixtures/delivered-303/me340.json` (node `IMPORT`). `--republish` is dropped, so the import makes a new project.
- **Container:** a CPU container (4 cores, 32 GiB) copies the staged files from the Volume to local disk and links them at the Mac path, because every run records its Mac path.
- **Stores:** it starts a throwaway Postgres (Debian's version) and uses a throwaway blob root under `/tmp`. Both disappear with the container. The Mac's Postgres on 54329 is never touched.
- **Timers:**
  - wall time of the script's own functions;
  - the repository calls;
  - blob put/get;
  - for each asset kind: bytes, plus the time to put and register them.
- **Sampler:** a stdlib stack sampler checks the import every 20 ms. It attributes wall time to the script's line ranges (`BUILD`/`RUN` in `probe.py`, checked against the file) and to the leaf functions.
  - Each gap between samples is charged to the stack seen before the gap. A C call that holds the GIL (open3d) blocks the sampler, so this is the stack that was running.
  - Run 1 charged each sample a fixed share instead. That gave too little time to open3d and too much to shapely, which releases the GIL. Trust its timers, not its sections.
- **Per-call rows:** every `_plan_projection` and `light_model` call is recorded with where it happened and its triangle count.
  - For each generated model, the probe also outlines the 40k-triangle display mesh that `light_model` just made. It records the time and the IoU against the kept outline.
  - This probe work is reported separately as `probe_overhead_in_run`.

## (b) Write path

- `write_layers` writes three layers, one after another, into the Volume:
  - **room:** the fused room mesh as panoptes-mesh-v1, plus the walk's camera path;
  - **room-photo:** the photo-textured room GLB;
  - **splats:** the splat file.
- Each layer is stored as content-addressed blobs (`blobs/sha256/<hex>`, never rewritten) plus a small patch JSON (`reports/<id>/patches/<seq>-<layer>.json`). Then the Volume is committed.
- `serve_layers` is started first, like a server that is already running. It polls the Volume and serves each new patch and its blobs over HTTP on loopback. A client in the same container fetches them and checks every sha256.
- Timing is measured at each step: write, commit, visible in the other container, and HTTP fetch.

## Results (2026-09-28)

M = measured, E = estimated. Raw numbers are in `runs/m3-exp-e8-import-profile-{1,2,3}` and `runs/m3-exp-e8-import-layer-{1,2,3}`.

**(a) Import, warm compute**

- **Total:** 920, 1050 and 1186 s (M, three runs; run 3 excludes 35 s of probe overhead) on a 4-core Modal CPU container. Hosts are shared, so the spread is noise.
- **Output:** the same document as `c40fbd08`: 1994 assets, 124 entities, 225 cameras, 902 observations. That is 451 MB of blobs, a 5.2 MB document and a 22 MB database.
- **Where the time goes (run 3, M):**

The rows do not overlap and add up to the 1186 s total.

| Where | Seconds | Share | What |
|---|---|---|---|
| Outlines of the 22 generated models | 562 | 47% | Exact shapely union of every triangle of the full-resolution model (0.2–1.35 M triangles each). One union raised a GEOS error after 22 s and was redone on the 1e-6 grid; that model took 82 s. |
| `light_model` decimation, 22 generated models | 257 | 22% | open3d quadric decimation to 40k triangles, plus a distance check |
| `light_model` decimation, 433 mover surfaces | 191 | 16% | 0.44 s each |
| `register_asset` (1994 calls) | 62 | 5% | A new Postgres connection and a blob re-verify for each asset. It is most of the mask cost: 29 s for 902 masks totalling 1.4 MB. |
| Fused cuts (`fused_part` + their outlines) | 25 | 2% | |
| Masks and source images: read, rectify, PNG | 22 | 2% | |
| `finish_job` + `commit_edits` + `create_publication` + validation | 20 | 2% | |
| Blob put (write, fsync, verify) | 5 | 0.4% | |
| Everything else | 43 | 4% | Observed-surface packing, video replay, room shell, loads |

- **Fix candidate (M):** outline each generated model from its 40k-triangle display mesh. That took 35 s instead of 562 s, with IoU 0.9989–1.0000 (median 0.9997) against the full-resolution outline.
- **Staging (M), data transfer, not compute:**
  - Mac to Volume: 1.30 GB in 181 s.
  - Volume to container disk: 1030 s with serial `copytree` (about 3 files/s), 10–12 s with 32 threads.

**(b) Write path, per layer, warm container (M, three runs)**

| Layer | Bytes | Write + commit | Visible to the reader after the commit | HTTP fetch + sha256 check | Write to served |
|---|---|---|---|---|---|
| room (mesh v1 + 673-camera path) | 6.36–6.56 MB | 1.53–2.80 s | 0.39–0.50 s | 0.58–3.04 s | 2.70–5.55 s |
| room-photo (textured GLB) | 26.5 MB | 1.41–1.91 s | 0.25–2.70 s | 0.35–0.47 s | 2.27–4.58 s |
| splats | 52.0 MB | 1.35–2.16 s | 0.27–1.78 s | 0.39–0.46 s | 2.67–3.58 s |

- In runs 1–2 the camera path held only the 19 DROID keyframes. In run 3 it holds all 673 walk frames (210 KB).
- Runs 2–3 found the room-photo and splat blobs already stored, because content addressing makes each key unique, so they skipped writing them. Run 1 wrote everything fresh.
- Writing the blobs takes 0.01–0.09 s. The Volume commit (1.3–2.8 s) is the write cost.
- The patch JSON is 276–496 bytes.
- The first read from a fresh reader container is the slow fetch: 2.3–3.0 s for 6 MB.
- The writer container's cold start plus loading its inputs took 4–7 s (E, from its wall time minus its writes).
