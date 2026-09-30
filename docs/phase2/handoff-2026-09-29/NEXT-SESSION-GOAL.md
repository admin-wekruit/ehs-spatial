# Next session: goal, problems, definition of done (Panoptes phase 2, written 2026-09-29)

Read `HANDOFF.md` (decisions, rules, where things are) and skim `digest.md` (all research and experiments with sources)
before starting. This file says **what to solve next and how to know it is solved**.

## Goal

**Make the object layer correct.** In the three test videos (ME340 workshop, Sam's Club, Walmart), clicking any object
must show a model that sits on that object in the video, a position and size that are right within their stated ±u,
and a type. Speed stays as it is (the user: "267s还可以接受，问题不大，先解决正确性问题"). Every result states whether
positions are right or wrong.

Suggested `/goal` line for the new session (the user writes goals in Chinese):

> 修正物体层的正确性：ME340、Sam's Club、Walmart 三段视频里，点任何物体，它的模型要贴在视频里对应的物体上（80% 以上的 RecGen 模型对得上，重合度中位 ≥ 0.6、中心偏 ≤ 8 cm），在有真值的数据上地面位置误差中位 ≤ 8 cm，只拍到一面的尺寸不再当成数值；做完给我报告，第一张表写清每段视频的位置对不对；速度保持现在的水平（全部模型约 267 s 可以接受）。

## Starting point

- Code: branch `r5b/integrate` at `ac04f0d` (on GitHub, `github.com/admin-wekruit/ehs-spatial`, with every other
  phase-2 branch). Branch `r6/correct` (at `ac04f0d`, on GitHub too) is the starting branch; on the Mac an empty
  worktree `panoptes-phase2-video-r6-correct` is ready.
- Planned-but-not-run workflow for this goal: `workflows/r6-correctness.js`.
- Current report: https://claude.ai/artifact/86jUzcrvntqFAgXPQiUnuC (source and rendered copy in `report/`).
- Current numbers and files: `HANDOFF.md` §6, `results/by-run/r5b-results/` (quick.json, alignment.json, gt/).

## Problems (priority order, with evidence)

| # | problem | evidence (merged code, warm runs unless noted) | target |
|---|---|---|---|
| P1 | RecGen models do not line up with the video | ME340: 75 of 149 generated models fail the display rule (size 0.25–3× card box, centre within half its diagonal + 0.1 m, best-view IoU ≥ 0.5) and are hidden; Sam's Club 20 of 32; worst 40 cm–3 m off or not drawn. `runs/r5b-results/{me340,samsclub-a2}/alignment.json`, `viewer/*/modal-worst-generated.jpg` | ≥ 80 % pass the unchanged rule; root cause per failure class (scale / translation / rotation / view or crop / mask / bad generation) |
| P2 | Floor position error | GT `r5b-int-gt-001`: median 12.5 cm ARKit47 / 9.0 cm TUM at the true camera height, 23.6 / 20.0 cm at the assumed 1.6 m; p90 23–35 cm. `runs/r5b-results/gt/physical-score.json` | median ≤ 8 cm at true height; report the assumed-height number and how much the height assumption causes |
| P3 | One-view sizes stated as values | Sam's Club cart width 2.56 m (1 view set); ME340 machine base 1.06 m above floor (both flagged "needs review") | 0 cards state a single-view width/length/height as a value; class-prior checks; both examples fixed |
| P4 | Round 5 not finished | Walmart run `r5b-int-walmart-002` finished but not summarised; audits, `summary.*` and the adversarial review not done | Walmart in every table; audits; review with no open blocker/major |
| P5 | ME340 types weak | held-out family ≈ 0.60 (r5b vocab builder); merged run not measured | ≥ 0.70 without leakage; retail not lower than round 4 |
| P6 | Angles | gate (≥ 2 view sets, u ≤ 12°) keeps 182 of 764 GT part readings, error median 1.65°, p90 8.4°; truly tilted ME340 parts (guards; the user's example is a guard panel at ~30°) not audited | every truly tilted ME340 part audited by eye; GT p90 ≤ 8° |
| P7 | Width coverage | only 32–55 % of cards have a width number or bound after dishonest "at most" bounds were removed | raise coverage from more views without false bounds |
| P8 | Change detection recall | cross-visit TUM plant 6/43, teddy 4/56; planted in-video 5/9 with 3 false claims on one Walmart clip | report; improve only after P1–P4 |
| P9 | Observed-surface models are partial | ME340 by eye right / partial / wrong 20 / 25 / 5 | acceptable fallback; improves with P1 |
| P10 | Speed (deferred by the user) | all RecGen models 267 s ME340 / 162 s Sam's Club; cards 37 s | do not make it > 10 % worse |

## Definition of done

1. Warm calls on ME340, Sam's Club and Walmart: ≥ 80 % of non-simple objects show a RecGen model that passes the
   unchanged display rule; over all shown models the overlay IoU median ≥ 0.6, centre offset median ≤ 8 cm, p90 ≤ 25 cm,
   and no shown model is > 50 cm off.
2. GT (ARKitScenes 47333932 / 42445448, TUM fr1-room), every card counted (no selection): floor position median ≤ 8 cm
   at the true camera height; height and top-above-floor ≤ 3 cm; each stated value's ±u covers GT ≥ 85 %.
3. No card states a single-view-set width / length / height as a value; class-prior violations are flagged.
4. ME340 held-out family accuracy ≥ 0.70 without leakage; Sam's Club and Walmart not lower than round 4.
5. Every truly tilted ME340 part audited; GT angle error p90 ≤ 8°.
6. An independent adversarial review (CPU only, re-derives numbers, looks at images) passes with no open blocker or major.
7. The three interactive 3D report pages regenerated from the new runs with `artifact-report/` (same URLs: republish
   in place) — this is how the user looks at results now — and a report (update the round-5 artifact in place, or a new one) whose first table is "positions right or wrong" per
   video, before → after, with overlay images before/after on the same frames, and latency within 10 % of round 5.

Not in scope: speed work, safety judgement, commercial model alternatives, deployment.

## Paste-ready prompt for the new session

```
You are continuing the Panoptes phase-2 "video -> clickable 3D object layer" work. First read, in order:
docs/phase2/handoff-2026-09-29/HANDOFF.md, NEXT-SESSION-GOAL.md, then skim digest.md (branch
handoff/phase2-2026-09-29-docs of github.com/admin-wekruit/ehs-spatial; the same files are in
/Users/adam/Desktop/panoptes-public/research-notes/phase2/handoff-2026-09-29/ on the Mac).
Your goal is NEXT-SESSION-GOAL.md: make the object layer correct (models sit on the objects in the video, positions and
sizes right within ±u, one-view sizes never stated as values), with the definition of done listed there. Start from
branch r5b/integrate (worktree panoptes-phase2-video-r6-correct, branch r6/correct). Follow HANDOFF.md §2-3 exactly:
decide instead of asking, VLM last, RecGen internal only, ephemeral `modal run` on 2x A100 only, heavy compute on Modal
(the Mac is weak), no secrets, never SendMessage to workflow agents, builders -> integrator -> adversarial review, log
spend in the ledger. Report in short Chinese messages with pictures; the first table of every report says whether
positions are right or wrong per video. The user views results as claude.ai artifact pages (one per video: the 3D
follows the video camera, click an object for its card) made with docs/phase2/handoff-2026-09-29/artifact-report/
(README there); regenerate and republish them in place after every round.
```
