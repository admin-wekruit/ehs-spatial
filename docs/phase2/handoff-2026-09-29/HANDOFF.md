# Panoptes phase 2 handoff: video → clickable 3D object layer (2026-09-29)

You are taking over the Panoptes phase-2 video work from a long agent session. Read this file first, then
`digest.md` in the same folder (research paths, every experiment with numbers and sources), then the round-5 results.
Everything below is either the user's own direction (quoted) or a measured result with its path. Where the state was
still moving when this was written, it says so.

---

## 1. Mission (the user's words)

Active goal, verbatim:

> 你的目标是做出来一个比较细致的，可以快速的，mvp出来，处理市场不能太久；可以explore各种新技术，做清楚research，gpu并行优化，2块A100，然后给我看结果，我们可以接受不那么完美，一开始细致程度不那么高，但是至少要能辨别出来是什么就行，复现这个demo没那么那么重要，但是要能做比方说在一个视频里，我点击视频，我们能知道这个物体的物理信息和判定是什么，就是点云如果只是为了复现没那么重要（lingbot那个demo），但是理解是什么，这些很重要；如果点云作用是这个我们就得要精准

In short: from a walk-through video, click any thing and get **what it is** (at least a type), **its segmentation**,
**its physical information** (position, top/base above floor, extents, angles, each ±u) and a **display model whose
outline is right**. The result must be fast (minutes, not hours) on **2 A100** (production has only 2–3 A100).
Priorities, in order: correct fine object determination → cross-time → speed.

## 2. Decisions the user already made (do not ask again)

The user is angry at repeated questions ("你别他妈一直问"). Decide with sensible defaults, state the decision in one
line, and report results. Only ask when truly blocked (secrets, deletions, money far beyond the ledger).

- **Judgement (safety verdicts) is paused.** Build the 3D object layer first: find objects, say what they are, and still
  delineate unidentifiable things as objects with physical properties. Acceptance for now: most objects typed; every
  object has segmentation, physical info and a model.
- **VLM last and rare.** No per-object or per-click VLM questions ("太贵"), no hazard / PPE / safety prompts, judge off by
  default. **Exception accepted on 2026-09-29: one VLM scene-vocabulary request at the start of each video**
  (seeds the detector's word list). Cluster-medoid naming after cheap votes is the only other VLM use.
- **No fine-tuning for now.** YOLO11/26 and open models are fine; fine-tune later.
- **Jev-Omni** may be used for routing decisions and runs on its **own GPU/server**, never on the 2 pipeline A100s.
- **Models: RecGen (internal, non-commercial) is THE display-model generator for every non-simple object.** Simple
  structure (walls, floors, ceilings, doors, plain panels/boards, beams, poles/pipes, plain boxes, shelf frames) gets
  simple primitives. The observed-surface mesh shows until the RecGen model lands. Commercial alternatives are out of
  scope ("我们内部用recgen可以快速，别管了"). RecGen must be fast (parallel on both GPUs).
- **The 3D view must look like a rough reconstruction of the place**: models placed where the objects are, lined up
  with the video ("这个图得对上"), no wireframe boxes, no always-on labels (the user's screenshot of r4b's 3D pane with
  overlapping boxes/labels: `before-3d-pane-r4b.png` in this folder). Needs speed + parallelism.
- **Everything cross-time**: timelines and per-interval values within a video; diffs across visits of the same site.
- **Scale**: assume a 1.6 m camera height for now (metric values labelled "estimated"); a real height input comes later.
- **Budget** can be exceeded ("可以超预算 做吧"); still log every spend in the ledger (§9).
- **The Mac is weak and on battery**: heavy compute goes to Modal; run one local workstream at a time; start the local
  viewer only when needed and stop it after. The user wants results quickly and in stages ("先做一个就行 ME340").
- The user wants a report with **latency and visible effect** (screenshots, before/after) after each round.

## 3. Hard rules (security and operations)

- Never open files under `*/.platform/imports/*`. Filter any printed log with `grep -v -i "capabilit\|token\|secret"`.
- No secrets in files or URLs; never handle an API key locally. NEVER `fly secrets set`; secrets only via ATM/Infisical.
- No Gemini unless a named Modal secret exists (create via ATM/Infisical, not by hand). Local Qwen3-VL-8B (vLLM) is used.
- Modal: **ephemeral `modal run` only; never `modal deploy`**; explicit timeouts, retries 0, `min_containers=0`.
  Remote exec into the deployed `panoptes-report-workspace` container was denied: do not do it.
- Don't list `~/Downloads` (use the exact paths in `data/clips/*/clip.json`). Don't stop local Postgres on 54329.
- Deletions need the user's approval each time. Commit/push only when asked; commit trailer
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. On GitHub (`admin-wekruit/ehs-spatial`): this handoff on
  branch `handoff/phase2-2026-09-29-docs`; code branches `r4b/integrate`, `r5b/{vocab,models,time,visit,integrate}`,
  `recgen/fast`, `route/jev`, `r5/models` (pushed by the user at ~23:10; `r5b/integrate` was at 1a8f42e then and kept
  moving). Older experiment branches (`fx/*`, `x13/*`, `r4/*`, `mvp*/*`) are local only. The auto-mode classifier
  blocks agent pushes of code branches; ask the user to run the push.
- Disk: the Mac had ~14 GB free; keep pulls small (jpg ≤ 1600 px), check `df -h /System/Volumes/Data`, stop if < 8 GB;
  large blobs stay on Modal Volumes.
- Never `SendMessage` to a running workflow agent: it spawns a duplicate writer. To change a running workflow, stop it
  and start a new script that injects finished results (see §8 on resume pitfalls).
- RecGen licence is non-commercial: internal profile only. YOLOE (ultralytics) is AGPL: internal unless licensed.
- Labels "by eye" come from agents and must be called agent-labelled; the user checks by eye too.

## 4. Where things are

| what | where |
|---|---|
| code repo (base, never modify) | `/Users/adam/.codex/worktrees/panoptes-phase2-video` (branch `codex/phase2-video`) |
| code worktrees (one per branch) | `/Users/adam/.codex/worktrees/panoptes-phase2-video-<name>` |
| research notes, results, ledger | `/Users/adam/Desktop/panoptes-public/research-notes/phase2/` (`runs/<run>/`, `video-mvp/cost-ledger.json`) |
| this handoff + saved plans/scripts | `research-notes/phase2/handoff-2026-09-29/` (`scratch/` session plans and dossiers, `workflows/` workflow scripts) |
| Python / Modal CLI | `/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python`, `.../.venv/bin/modal` |
| videos | code repo `data/clips/{me340-165, samsclub-337, walmart-190}/clip.json` (ME340 workshop, Sam's Club, Walmart; 899 frames each at 30 fps; bench site names `me340`, `samsclub-a2`, `walmart`) |
| ground truth | ARKitScenes 47333932 / 42445448 (LiDAR), TUM RGB-D fr1-room (+ fr1 desk/desk2/xyz/360/plant/teddy for visits) on Modal Volumes; runs `r4b-gt-*`, `r5b-time-gt-001`, `r5b-models-gt-001`, `r5b-visit-*` |
| main app / bench | `modal_apps/fast_report_app.py` (the pipeline in one container on 2× A100-80GB), `scripts/fast_report_bench.py` (`--sites`, `--plan first,warm`, `--judge off`, `--display on`, `--vocab`, `--self-check`; outputs `call-*.json`, `eval-*.json`, `mirror/`) |
| pipeline code | `fast_report/` (`core.py` orchestration, `cards.py`, `segment.py`, `cascade.py` naming, `vlm.py`, `vocab.py` (r5b), `ondemand.py`, `display_model.py`, `recgen_fast.py`, `timeline.py`, `windows.py`, `layers.py` patch store + HTTP server) |
| viewer | `web/src` (`LiveReport.tsx`), served by `python -m fast_report.layers serve <root> --port 8794` + `FAST_PORT=8794 npx vite --port 5185`; page `http://localhost:5185/app.html#/live/<report id>`; helper `workflows/check-viewer.sh` (edit its worktree `W` and root paths first; the root is a folder of `reports/` + `blobs/` symlinks to a run's `mirror/`). Heavy on the Mac: start only when needed |
| reports for the user (claude.ai artifacts) | round 5: https://claude.ai/artifact/86jUzcrvntqFAgXPQiUnuC · object layer / round 4 / older experiments: https://claude.ai/artifact/5Xhnj2SHg3pTmTNJLGZG9s |
| older handoff / architecture | code repo `docs/phase2/` (`ARCHITECTURE.md`, `FAST-PATH-PLAN.md`, `FAST-BUILD-SPEC.md`, `CLICK-MVP-SPEC.md`, `HANDOFF.md`, ...) |

## 5. The pipeline as it stands (one Modal container, 2× A100-80GB, MPS)

1. MP4 bytes → decode keyframes (5 fps is enough; X1 found more fps does not help static scenes) → shot cuts.
2. **DA3-GIANT any-view** depth + poses per shot → GPU TSDF at 3 cm → room mesh and points (first 3D at ~13–20 s).
3. **Word list** (r5b: RAM++ tags on keyframes by default; Qwen scene vocabulary allowed once; union being measured)
   → **SAM 3** masks per keyframe (wave 1 core words, wave 2 per-video words) + OWLv2 objectness boxes → SAM 3 tracker
   masks for coverage.
4. Lift masks to 3D, merge instances by mutual projection → object list; people via PeopleLoop (speed gate 12 m/s).
5. **Cards**: type via the cascade (SAM 3 word → second vote: DINOv2-L bank kNN / PE-Core zero-shot / YOLOE-26L → Qwen
   on cluster medoids only → family "(type only)" → "unidentified (shape)"); physical fields with ±u (k calibrated on
   ARKit/TUM; "not observed" / "at least" / "at most" states; contract checks); cards v1 at ~26–34 s.
6. **Models**: observed-surface mesh per object (14–25 ms each) at cards v1 + ~3 s; primitives for simple structure;
   RecGen FAST for non-simple objects (2.0–2.5 s/object; 2 processes per GPU); planar-part angles ±u.
7. **Time**: content windows (ORB co-visibility 0.40) → per-object timelines and per-interval values; **visits**:
   registration into the first visit's site map, object matching, diffs with evidence frames.
8. Layers written progressively to a patch store on a Modal Volume; the viewer polls and draws; on-demand clicks
   segment/name a new object in a live container.

## 6. Results so far (details and sources in `digest.md`)

**Round 4 (r4b/integrate 40a153e; review: PARTIAL)** — warm call, ME340 / Sam's Club / Walmart:
family-typed 95/100/98 % (held-out family right 0.51/1.00/0.83 → ME340 not met); clicks on real things open the right
card 0.87/0.76/0.82; physical fields 100 % complete, 0 contract violations; models 100 % but 84–89 % boxes (reviewer:
3/5/8 of 20 random implausible); cards 30–43 s, all types 88–98 s, models ~180–183 s (Volume commit); GPU ≤ 70.2 GiB.
GT: at true camera height heights 2–3 cm, tops 1.6–4.2 cm; at the assumed height tops 5.7–21 cm. Review findings:
VLM vocabulary gated 77–90 % of detections (VLM first), PPE prompts still asked, width "at most" bounds failed 8–28 %,
mutable naming bank, 3D pane labels wrong. Results: `runs/r4b-results/`.

**Round 5 builders (all finished; merged in r5b/integrate)** — see https://claude.ai/artifact/86jUzcrvntqFAgXPQiUnuC:
- *vocab* (r5b/vocab c03e686): RAM++ words, 0 VLM calls before cards (was 3); held-out family ME340 0.60 (0.51),
  Sam's 1.00, Walmart 0.79 (0.83); clean coverage ME340 73/105 (80), Sam's 79/91 (79), Walmart 58/74 (61); RAM++ + PE
  union on ME340 81/105; cards v1 34/26/26 s, all types 67/79/74 s (models off in those runs). `runs/r5b-vocab-results/`.
- *models* (r5b/models 18a9022): shown model = observed surface for most objects; by eye (50 per video)
  right/partial/wrong ME340 20/25/5, Sam's 26/19/2, Walmart 37/12/1; RecGen-generated models on Walmart 8/1/0 vs the
  same objects' r4b boxes 0/7/2 — but only 0–9 generated per video (route + view gate too strict); Tier 1 done 199 s
  (ME340 first call); angles on GT with every part counted: median 1.6°, p90 12°; **false tilts**: 26 of 27 ME340 parts
  listed at 15–75° are not tilted. `runs/r5b-models-results/`.
- *time* (r5b/time ce62d95): every card has a timeline; 0 false change claims on unplanted videos; planted events
  found 5/9, 3 false claims on one planted Walmart clip; +0–3 s; honest widths now give a number or bound on only
  55/32/48 % (was 91/89/94 %). `runs/r5b-time-results/`.
- *visit* (r5b/visit e06573a): registration vs TUM GT 5–12 cm, 1–3°; object match recall ~0.9; real changes found
  6/43 (plant) and 4/56 (teddy); 0–1 false changes per revisit; 10–17 s per revisit. `runs/r5b-visit-results/`.

**Model generators measured**: RecGen default 6.3–8.4 s/object; RecGen FAST 1.96 s (SXM4) / 2.5 s (PCIe), acceptance
9.3 vs 10.0 of 19 (within seed noise), 2 A100 → 20/40/70 objects in ~19/36/62 s; by eye on 90 objects RecGen 58/27/5,
SAM 3D 29/36/8, TRELLIS and TripoSR worse, observed surface 38/52/0 (`runs/recgen-fast-results/`,
`runs/r5-models-results/`). Jev-Omni routing (fixed-shape list → Jev Q5 raw) ~0.82–0.90 held out, no VLM (route/jev).

**ME340 on the merged code, run 1 (r5b-int-me340-001, 23:00, warm):** first 3D 18.2 s, cards 39.9 s, observed
surfaces 42.1 s, all types 98.0 s, first RecGen model 130.5 s, all RecGen models 298.7 s (too slow; r4b's 182.8 s was
mostly boxes). 275 cards: 130 RecGen, 139 observed surface, 6 primitives, 0 boxes. Overlay alignment (model rendered from
the video camera vs the object mask): observed surface IoU median 0.51 / 4.6 cm, RecGen 0.40 / 10.9 cm (30 % above
0.5) → run 2 shows a RecGen model only when it lines up (42 of 118 on run 1's data), else the observed surface.
RecGen GPU time: view selection 658 s + generation 666 s (129 models, ~5 s each) + acceptance check 519 s → cut
selection and checks first. VLM: 1 scene-vocabulary request + 50 medoid names; 0 hazard. GPU peak 69.3 GiB.
Files: `runs/r5b-int-me340-001/`, `runs/r5b-results/me340/quick-run001.json`, `runs/r5b-results/viewer/me340/`.

**ME340 on the merged code, run 2 (r5b-int-me340-002, 23:25, warm; the current state):** first 3D 15.9 s, cards 37.4 s,
observed surfaces 39.5 s, all types 89.6 s, first RecGen model 105.7 s, all RecGen models 266.5 s (target 150 s).
274 cards: 75 RecGen shown, 192 observed surface, 7 primitives, 0 boxes. RecGen attempted 149; a model is shown only
when it lines up (size 0.25-3x the card box, centre within half its diagonal + 0.1 m, best-view IoU >= 0.5): 74 shown,
75 fall back to the observed surface. Shown RecGen IoU median 0.54 / 7.0 cm; surfaces 0.46 / 6.6 cm; primitives
0.76 / 1.5 cm. RecGen GPU time: selection 328 s, generation 624 s (~4 s each), checks 252 s. Angles only from >= 2
view sets with u <= 12°. VLM: 1 scene vocabulary + 51 medoid names. GPU peak 70.2 GiB. The 3D pane has no boxes and
no always-on names; an overlay mode draws models on the video frame. Files: `runs/r5b-int-me340-002/`,
`runs/r5b-results/me340/{quick.json,alignment.json}`, `runs/r5b-results/viewer/me340/` (3d-before-r4b / 3d-after,
click cards, overlays, worst generated). Round-5 report (ME340 section): https://claude.ai/artifact/86jUzcrvntqFAgXPQiUnuC

**Round 5 integration: IN PROGRESS when this was written** (Sam's Club, Walmart, GT and one visit pair were running at 23:40). Workflow run `wf_401bb21d-ec8` (script
`workflows/r5d.js`) on branch `r5b/integrate` (merge of the four builders done at 854e3e7). It implements RecGen for all
non-simple objects (FAST, 2 processes per GPU, look-alike reuse), the clean 3D pane (models in place, no boxes/labels),
an overlay alignment metric + overlay view, and the false-tilt fix; runs ME340 first and writes
`runs/r5b-results/me340/quick.json` + `runs/r5b-results/viewer/me340/*.jpg`, then Sam's Club / Walmart / GT / one visit
pair in parallel, smaller audits, `runs/r5b-results/summary.{md,json}`, then an adversarial review (CPU only). Check
`runs/r5b-results/` and `git -C <base>-r5b-integrate log` for where it got to.

## 7. What is missing (prioritised next steps)

1. **Finish and verify round-5 integration** (above). Publish ME340 first (latency waterfall r4b vs r5 — generator
   `report/latency.py` in this folder, page source `report/template.html`; before/after 3D pane, overlay frames), then
   the other videos and the review. Update the round-5 artifact in place (pass its URL).
2. **RecGen for every non-simple object, fast.** Script ready, never run: `workflows/recgen-fast2-speedups-not-run.js`
   (training-free speed-ups: partial denoising from the observed occupancy, size-based token budget, compile/bf16/CUDA
   graphs, cross-object batching, processes per GPU; scheduler `generate_all` with look-alike reuse, click-first
   priority, progressive writes; one-sided objects with guessed backs). Note: 2 processes per GPU gave only +17 %, so
   the remaining gains must come from doing less work per object and fewer objects (reuse), not more parallel copies.
3. **Aligned reconstruction view**: models posed in world space so the overlay matches the video; per-object silhouette
   IoU / centre offset; overlay mode; render on demand (the WebGL pane made the Mac slow).
4. **Angles**: only from multi-view measurements with small u; label the rest "not measurable"; audit truly tilted parts
   (ME340 guards) — the user's example is a guard panel at ~30° to the floor.
5. **Change recall** (in-video and across visits) is low for small objects; improve matching (appearance + place).
6. **Width coverage** fell when dishonest "at most" bounds were removed; measure widths from more views.
7. **ME340 types** (workshop domain) at 0.6; add workshop classes/rows without leakage; decide the default word source
   from the Qwen / RAM++ / union comparison.
8. **Latency targets** on 2 A100 (warm): clickable cards ≤ 30 s, all typed ≤ 60 s, all models ≤ 90–120 s.
9. Later: resume judgement per interval / per visit (`research-notes/phase2/rule-library-2026-09-29.md`), real camera
   height input, deployment (only after the user asks).

## 8. Working method that worked (and pitfalls)

- **Builders → integrator → adversarial reviewer** workflows. Reviewers re-derive numbers from saved JSON and look at
  images themselves; they caught real overclaims every round (VLM-first vocabulary, PPE prompts, rounded-up shares,
  selection-biased angle stats, queue vs commit times). Keep that step.
- Report per video, warm and first call; times from MP4 bytes in the container to the Volume commit (`written_s`),
  cold start separate; GPU peak per stage (flag > 72 GiB).
- **Workflow resume pitfall**: cache keys depend on call order; removing or reordering an `agent()` call re-runs later
  agents. To skip finished work, write a new script that injects the finished results as constants
  (see `workflows/r5d.js`, which injects the four builder results).
- The session's Modal client dying stops ephemeral apps; after a restart check `modal app list` for leftovers.
- Keep the user informed in short Chinese messages; lead with results and pictures; don't end with questions.

## 9. Budget

Ledger: `research-notes/phase2/video-mvp/cost-ledger.json`. Authorization history: 250 → 300 → 500 USD, then the user
allowed exceeding it (2026-09-29). Accounted + reserved ceiling was 669.71 USD after reserving round 5 (82) and
recgen/fast2 (15, not spent). Round-5 builders spent about 38 USD (models 18.74, visit 7.10, time 6.19, vocab ~6 est.);
the integration's spend is not recorded yet. Round 4 integrate 12.04 USD. Record reserve / estimate / result / release
for every run.
