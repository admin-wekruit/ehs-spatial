# Rules for any agent or session working in this repository (ehs-spatial)

Read in this order, nothing else first: `README.md` → `docs/STATE.md` (what the system is today) → `HANDOFF.md` (how to run it) →
`research/module-swap-2026-10-07/REPRODUCE-PROMPT.md` (how to reproduce the published numbers). When documents disagree, `docs/STATE.md`
and `HANDOFF.md` win.

What the other directories are:
- `docs/archive/` — history. Written on the date in the file name; superseded. Never act on it, never cite it as current.
- `docs/research/` — proposals and experiments (verdict layer, Phase 4). Not shipped. `docs/research/README.md` says which are current.
- `research/module-swap-2026-10-07/notes/` — frozen experiment records behind the published reports. Wording is from that day (it may
  say "two repositories"); paths are parametrised by `env.sh`. The valid reproduction prompt is the one at `research/module-swap-2026-10-07/REPRODUCE-PROMPT.md`.
- `research/verdict-layer-trial-2026-10-07/` — a trial; its README lists what is already superseded.

Facts that sessions have hallucinated before — do not:
- The shipped geometry is **geometry-mvs** (DA3-BASE + RoMa + numpy bundle adjustment + MoGe-3 in-mask fill). Pi3X and RecGen are
  comparison baselines only; do not fetch, run or cite them as the pipeline.
- Completion is SAM 3D Objects @2e73555. Scale comes from the e-stop reference object; never change a published `nativeToMeters`.
- Clone with `git clone -b main https://github.com/admin-wekruit/ehs-spatial.git` (default branch is still `feature/ehs-spatial-mvp`).
  `panoptes-serving` is frozen; do not clone it.
- `.env` is the only configuration surface (`env.template` lists every key; `make check-env`). `research/module-swap-2026-10-07/env.sh`
  derives the rest. Do not invent endpoints: the jump VM forwards are 8085 (sam3d), 8084 (geometry-mvs), 8081 (sam3), 8080, 8090.
- `HF_TOKEN` is used once by `scripts/onprem/fetch_weights_*.py`; never write it into any file or log. Never read or commit `.platform/`.
- Gemini is used by the Gradio workbench, report chat and the refine agent; `panoptes run` never calls it; on-prem leaves `GEMINI_API_KEY` unset.
- Reports contain measurements, boxes and facts; they contain **no machine compliance verdicts** today (the policy engine abstains by design).
- Do not change algorithms or parameters to make numbers match; record differences against the tolerances in `REPRODUCE-PROMPT.md` §5.
- Test baseline: `PANOPTES_FAKE_MODEL=1 PANOPTES_WORKCELL=$PWD pytest tests` → 6 known failures (platform version pins, listed in
  `CHANGELOG.md`); anything else failing is a regression.
- Do not deploy, do not call cloud GPUs beyond what `HANDOFF.md` describes, do not run `fly`/`modal deploy`.

Relationship with the customer (2026-10-08): THIS repository is the Modal version and the source of truth. The customer's internal app
clones it and replaces only the Modal GPU calls with their A100s behind the jumpbox; nothing else changes. Their Linux / Docker /
on-prem adaptation and its verification are theirs: not our work, never something we verify here.
Cleanup rules we apply to our own repo (from their review, `docs/REVIEW-ARGUS-2026-10-08.md`): code is English only (identifiers,
comments, docstrings, logs, exceptions, JSON keys; Chinese / Dutch only in `web/src/locales/*.json`); no machine path (`/Users/...`,
`/private/tmp/...`) as a fallback: read the `env.template` key and fail fast; a failed step exits non-zero; one clean layout, research
out of the delivery path. Algorithms and parameters still never change during the cleanup: the regression standard is the published
measurement layers (117/117 fields, boxes 9 and 8).

Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` when a Claude session authors the commit.
