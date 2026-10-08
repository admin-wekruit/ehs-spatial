# Agent rules

Read `README.md` → `docs/STATE.md` → `HANDOFF.md` → `docs/DELIVERY-PATH-2026-10-08.md`. Current state and run instructions take precedence over preserved historical documentation. `AGENTS.md` is a link to this file.

- Understand the original need and trace every caller before changing code. Prefer the existing helper, standard library or installed dependency; write the smallest complete change.
- Do not add abstractions, compatibility aliases, duplicate schemas, silent fallbacks or business behavior outside the requested scope.
- This is the Modal source. Customer Linux/Docker/on-prem adaptation is their responsibility; do not claim it was verified here.
- Geometry is DA3-BASE + RoMa + NumPy BA + MoGe-3 masked fill. Completion is SAM 3D Objects, assembly v2. Historical model comparisons belong in the research reference.
- Never change algorithms, numerical parameters or a published `nativeToMeters` during structural cleanup. Check the frozen MVS reports recorded in `docs/STATE.md`.
- Use `argus.ROOT`, explicit environment configuration and one cell file per workcell. No machine-path fallback or date-directory import chain.
- Code, comments, docstrings, logs and exception text are English. Chinese and Dutch display prose belongs in `web/src/locales/*.json`; use schema-2 code/parameter messages in data.
- Preserve input validation and errors that prevent data loss. A failed child must fail the pipeline. Nontrivial logic leaves one runnable meaningful check.
- Do not read or commit `.platform`, credentials, model weights, local runtime caches or `node_modules`. Do not deploy or start a paid GPU/model run without task authorization.
- Reports contain measurements and evidence. Do not claim machine compliance, physical calibration, observed coverage or reviewed labels without the corresponding evidence.
- Run the retained offline checks before claiming completion, and distinguish passing tests, skipped tests and external validation.
