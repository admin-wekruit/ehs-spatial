## What this is

Branch `mvp/integrate` (worktree `/Users/adam/.codex/worktrees/panoptes-phase2-video-mvp-integrate`), from `fb/integrate`, with
`mvp/d-validate`, `mvp/c-viewer`, `mvp/a-cards` and `mvp/b-judge` merged in that order, then fixed in place. One command per
video runs the whole thing on 2 x A100-80GB (a first call after boot, a warm call, and a +5 s shifted-window call for
repeatability):

    cd /Users/adam/.codex/worktrees/panoptes-phase2-video-mvp-integrate
    PYTHONPATH=. /Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python scripts/fast_report_bench.py \
        --out RUNS/mvp-integrate-<site>-NNN --sites <me340|samsclub-a2|walmart> --plan first,warm,shifted --mirror-max-mb 8 --no-gpu-eval

Final runs (round 5): `runs/mvp-integrate-me340-005`, `-samsclub-004`, `-walmart-004`. Scored together (one pooled k, one
judgement table) with D's harness into `runs/mvp-results/eval` (`fast_report_eval.py --mvp ... --labels ... --no-gpu`,
reusing D1's SAM 3 click references). Viewer checks and screenshots: `runs/mvp-results/viewer/<site>/`.

## Open the viewer

    cd /Users/adam/.codex/worktrees/panoptes-phase2-video-mvp-integrate
    PY=/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python; R=/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs
    PYTHONPATH=. $PY -m fast_report.layers serve $R/mvp-integrate-me340-005/mirror --port 8793 \
        --replay $R/mvp-integrate-me340-005/mirror mvp-me340-e84efffd-1790666683 --as me340-demo --speed 4
    cd web && npx vite --host 127.0.0.1        # second terminal; proxies /fast to 8793
    open "http://127.0.0.1:5173/app.html#/live/me340-demo"

Sam's Club: `mvp-integrate-samsclub-004` / `mvp-samsclub-a2-d5e0c855-1790666685`; Walmart: `mvp-integrate-walmart-004` /
`mvp-walmart-c0761a2a-1790666653`. Click anything in the video: the card opens with what it is, its physical values
(each ± u, 'estimated' on every metre, 'not observed' / 'not measurable' with the reason) and its judgements. The mirrors keep
blobs under 8 MB only, so the full room and the splat stay on the Modal Volume; the replay leaves those two layers out.

## Main changes made while integrating (all general; no per-video case)

- Judge wired to A's cards v1 and v3 (B's stand-in removed), its rules in the process pool, VLM answers carried from the v1
  run to the v3 run.
- Identity: the decider's prompt now lists the options as letters (A's prompt listed none); both crops sent; asked for
  every object seen on >= 3 views (EHS classes first) on 336 px crops; replaces the free-text naming (spec 4.7), which
  held vLLM 34 s on Sam's Club; identity that lands after densify is merged into cards v3 as v4.
- Heights carry an `up` part (the walls' p90 plumb reading x horizontal distance): warm-vs-shifted height coverage
  72% -> 93% on ME340. Angles with a fit term over 15 deg are 'not measurable' (bulky objects read 70 ± 43 deg).
- k from repeatability written to `fast_report/calibration.json` (round 2, pooled): height 1.071, extent 1.345, position 1.024,
  angle 1.0; round 5 re-measured 1.028 / 1.0 / 1.001 / 1.0 with it applied (same clips: in-sample).
- J1 on goods off the floor uses the stack's own height (rack goods at 3.6-4.1 m had failed a floor-stack rule, 20 false
  FAILs on Sam's Club); J2 ignores principal axes at >= 45 deg; J4 shows the base height when its PASS rests on it; J5 scans
  from the nearest free path point within the path's band.
- Pick maps end at a shot cut (viewer, eval and core agree); objects v1 boxes in the process pool; the identity pass never
  raises (a raise ended round 3's runs while densify ran and faulted both GPUs for every later call).
- Eval: blob cards read; still-camera shots are not Sim3-aligned (ME340 shot 0: 2 cm path, scale 0.80); part/whole pairs
  (a 2 m stack vs one box of it) are not counted as repeats (8 / 23 / 18 pairs).

## Reading the numbers

- Everything metric is at estimated scale (floor plane + an assumed 1.6 m camera height). 'Agreement' is with the
  delivered reports, themselves model-made and at estimated scale. Sam's Club's delivered heights are known to sit
  0.7-1.5 m off (L2), which is where the MVP agrees least (top median 0.24 m).
- Identity 'agreement' is exact-name agreement with the delivered report's product-level names after a 0.5 m match
  ('stacked boxes' vs 'paper towel pack' counts as wrong). The agent audit (40 cards a video on their best view) is the
  direct measure of 'does it recognise what things are'.
- Judgements: no FAIL on any of the three videos; PASS only where value - u clears the threshold (J1, J4). J5 never
  decides: the spec's gap rule turns every PASS into NEEDS_REVIEW while the far side of a footprint is unseen (79-99% of
  cards, forward walks), and person paths from mono depth run through footprints. The 60 audited rows found no hazard
  present; the audit shows 0 false PASS but cannot measure FAIL precision (n = 0).
- Timing misses: objects v1's Volume commit queues behind events and the full room (sent at fb + ~0.5 s, written 1-2 s
  later); the first SAM 3D model on ME340 (168 s) and Sam's Club (160 s) waits for the facts (spec 7 moved it after cards v3);
  pick v2's fetch in the browser waits 0.7-2.6 s behind the densify burst (v1 at load: 89-107 ms).
- Earlier rounds: round 1 `mvp-integrate-me340-001`; round 2 `-me340-002`, `-samsclub-001`, `-walmart-001` (k fitted here);
  round 3 `-me340-003`, `-samsclub-002`, `-walmart-002` failed (the identity-pass TypeError); round 4 `-me340-004`,
  `-samsclub-003`, `-walmart-003`.
