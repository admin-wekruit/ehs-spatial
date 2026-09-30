# mvp2/identity: results (R1, R2)

Branch `mvp2/identity` (worktree `/Users/adam/.codex/worktrees/panoptes-phase2-video-mvp2-identity`), from `mvp/integrate`.

## What was built

- R1, one source of truth: `fast_report/cards.py` keeps each card's name-free measurements (`card.raw`: the size inputs, the
  angles, the un-reviewed statuses, the fitted primitive, the kind-free time state, observed extent change) and `apply_name(card)`
  derives everything a name decides from the final name: canonical class, kind/mobility, size check and its 'needs review' marks,
  the deformable no-angle rule, the primitive (kept only when the name is box/plane-like), the agent/deformable time state, the
  struck candidates. Called when a card is built and on every identity update (idempotent; self-check). Every identity update
  republishes the newest cards version (v2, or v4 after densify's v3) and re-runs its judgements; the judge's rules read the
  canonical class (`judge.class_of`); the objects layer's labels (the viewer's 3D labels) follow and pass the hazard gate.
- R2, naming: a canonical EHS/object taxonomy (`cards.TAXONOMY`, 17 families, ~120 classes, head-noun synonyms, 'X of Y' and
  'X with Y' rules, 'not an object'); open names from Gemini (cloud) for every object seen on >= 3 keyframes or carrying an EHS
  word, started as soon as the outlines exist, two objects a sheet (the thing outlined in its surroundings | a close crop), 14 a
  request, every request at once through the bench's relay into the deployed report container (the existing
  `name_video_entities` mechanism; no key leaves that container), stragglers and errors re-sent at 15 s, 36 s deadline; a second
  pass names densify's own cards; the Qwen lettered decider only when Gemini named nothing.
- Hazard-class names (spill, ladder, guard/fence/barrier/railing, fire extinguisher, cable, hose, forklift, pallet jack, exit and
  safety signs, e-stop, eyewash, first aid, fire alarm) are shown only when the detector's words include the class or its family,
  the class's size and placement rules pass on the measured box, and a VLM named it (not an 'unclear' or p < 0.5 answer);
  otherwise the top non-hazard detector word (or 'unidentified object') is shown with the reason.
- Study harness: `scripts/identity_study.py` (items, blind sheets, Gemini variants, scoring, R1 checker, timing, this page),
  `modal_apps/identity_study_app.py` (Qwen open naming, one A100).

## Caveats

- The relay runs in the local bench process (research harness): a product needs the Gemini call server-side (key in ATM/Infisical).
- Gemini's p is a stated probability (uncalibrated), so `judge.identity_confirmed` stays false for its names: a VLM answer still
  never makes a FAIL. Names are 'inferred'.
- Labels: one agent labeller, blind to the methods' outputs, from contact sheets; 'unclear' items excluded. The dev table is
  partly in-sample (the method, the packing and the taxonomy's extra words were chosen on it). The held-out items were drawn after
  that (seed 1, none of the dev items), labelled on one run's crops and matched to the final run's cards by id or nearest box
  centre (<= 0.3 m); round 1's names by nearest box centre. n is small on the retail videos (27 / 29).
- Run-to-run spread is large on the workshop video: the same held-out ME340 items scored 0.66 / 0.75 on run 004 and 0.80 / 0.84 on
  run 005 (Gemini's answers and the chosen views change between runs).
- No spill, ladder, fire extinguisher or forklift is truly present in the three clips: for those classes the gate was only tested
  against false names (floor patterns, pallets, slippers called 'spill' / 'guard' / 'ladder' in round 1). The detector-word
  condition fails when the site's vocabulary lacks the class: Sam's Club's 21 words have no 'pallet jack', so a real one was held back.
- Judgements: J4 now runs on confirmed cables/hoses and PASSes ones whose segment is off the floor; a hanging hose whose floor run is
  a separate segment can PASS (for the judge's owner). The last judgements land at 94 / 104 / 84 s warm (round 1: 82 / 117 / 56 s):
  densify's own names are re-judged.

Labels are agent-made by looking at blind contact sheets (not ground truth). Times: s from the MP4 bytes in the container to the layer committed on the Volume (written); cold start apart. Final runs: runs/mvp2-identity-me340-005, runs/mvp2-identity-samsclub-a2-005, runs/mvp2-identity-walmart-005.

## Identity timing (first-pass names, warm call; first call after boot in brackets)

| | objects v1 | Gemini sent | identity written | identity - objects | densify's names written | last judgements | request s (median / max) | GPU peaks GiB |
|---|---|---|---|---|---|---|---|---|
| me340 | 35.398 (38.92) | 39.377 (43.761) | 61.816 (66.195) | 26.42 (27.27) | 91.117 (99.905) | 94.229 (103.225) | 12.8 / 17.3 | [65.15, 56.9] ([60.35, 56.41]) |
| samsclub-a2 | 23.866 (25.65) | 25.284 (29.67) | 59.788 (64.81) | 35.92 (39.16) | 83.007 (101.069) | 104.422 (117.257) | 13.4 / 18.6 | [61.74, 54.88] ([67.76, 54.97]) |
| walmart | 23.272 (22.615) | 23.432 (26.555) | 59.342 (61.823) | 36.07 (39.21) | 81.917 (91.465) | 83.944 (93.215) | 12.8 / 26.2 | [62.61, 56.62] ([59.89, 56.81]) |

Round 1 (runs/mvp-results, warm): identity complete - objects v1 = me340 24.0 s, samsclub-a2 61.1 s, walmart 49.3 s. Target: <= 45 s.

## R1: name -> kind, size check, checks, verdicts

Violations over every cards and judgements version of every call (a card whose class or size-check class is not the one its shown name gives; a card apply_name would change; a judgement row on a check its card's final class does not apply): me340 0, samsclub-a2 0, walmart 0. Round 1's review: 51 / 155 / 51 cards.

## Names on fresh held-out items (warm call of the final runs; round 1 on the same objects)

| | n | right | right or close | round 1, same objects: n / right / right or close |
|---|---|---|---|---|
| me340 | 61 | 0.80 | 0.84 | 53 / 0.45 / 0.68 |
| samsclub-a2 | 27 | 0.85 | 0.85 | 27 / 0.89 / 0.93 |
| walmart | 29 | 0.93 | 0.93 | 29 / 0.72 / 0.83 |
| all | 117 | 0.85 | 0.86 | 109 / 0.63 / 0.78 |

Repeat (runs mvp2-identity-me340-004, mvp2-identity-samsclub-a2-004): me340 0.66 / 0.75 (n 61), samsclub-a2 0.89 / 0.89 (n 28), all 0.73 / 0.80 (n 89)

Hazard-class names on these items: 11 shown (9 right or same family, 2 on items labelled unclear), 5 held back by the second check (1 of them were true). Items not matched between the labelled run and the final run: 2. Dev set (Gemini's answers with the gate applied offline): 17 hazard names shown, 15 right, 2 held back (both false); round 1 showed 35, 20 right ('spill' on floor patterns, bags and slippers; 'guard' / 'ladder' / 'spill' on pallets).

## Dev study (runs/mvp2-identity-study-001: 181 labelled items from round 1's runs; the method and the taxonomy were chosen on it)

| method | ME340 | Sam's Club | Walmart | all | hazard-class items |
|---|---|---|---|---|---|
| A: lettered Qwen decider (round 1, deployed) | 0.41 / 0.72 | 0.85 / 0.89 | 0.83 / 0.85 | 0.64 / 0.80 | 0.43 / 0.57 (n 35) |
| D: Gemini open name, 6 a request, context + close crop as 2 images | 0.75 / 0.81 | 0.96 / 1.00 | 0.98 / 0.98 | 0.86 / 0.90 | 0.83 / 0.89 (n 35) |
| D: Gemini open name, 2 a sheet, 28 a request | 0.74 / 0.85 | 0.98 / 0.98 | 0.98 / 0.98 | 0.86 / 0.92 | 0.94 / 1.00 (n 35) |
| D: Gemini open name, 4 context tiles a sheet | 0.63 / 0.78 | 0.96 / 0.98 | 0.94 / 0.96 | 0.80 / 0.88 | 0.74 / 0.89 (n 35) |
| D: Gemini open name, 16 context tiles a sheet | 0.71 / 0.79 | 0.89 / 0.94 | 0.81 / 0.81 | 0.79 / 0.83 | 0.57 / 0.69 (n 35) |
| B1: Qwen open name, context + close crop | 0.37 / 0.44 | 0.26 / 0.77 | 0.55 / 0.60 | 0.39 / 0.56 | 0.43 / 0.51 (n 35) |
| B2: Qwen open name, set-of-marks pair | 0.26 / 0.36 | 0.21 / 0.66 | 0.43 / 0.64 | 0.29 / 0.51 | 0.34 / 0.49 (n 35) |
| C: decider, 'none of these' or p < 0.5 -> B1 | 0.42 / 0.69 | 0.79 / 0.87 | 0.72 / 0.74 | 0.60 / 0.75 | 0.40 / 0.54 (n 35) |

## Spend

Modal, list-price upper bound over each bench container's life (queue included): $13.34 for 8 benches (mvp2-identity-me340-001 $3.34, mvp2-identity-me340-002 $2.74, mvp2-identity-me340-003 $1.34, mvp2-identity-me340-004 $1.18, mvp2-identity-me340-005 $1.42, mvp2-identity-samsclub-a2-004 $1.09, mvp2-identity-samsclub-a2-005 $1.15, mvp2-identity-walmart-005 $1.08); benches stopped early and the one-A100 naming study app are not in it. Gemini (gemini-3.5-flash through the deployed report container): 6.71 M input and 0.60 M output tokens (relayed pipeline requests incl. re-sent copies, and the study); the repo records no price for it. The cost ledger was not edited.
