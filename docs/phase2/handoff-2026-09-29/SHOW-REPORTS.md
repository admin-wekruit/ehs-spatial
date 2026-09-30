# How to show the user a report (read this before reporting any result) — 2026-09-30

**The user looks at results on GitHub Pages, one interactive 3D page per video:**

- Index: https://admin-wekruit.github.io/panoptes-workcell-report/video/
- ME340: https://admin-wekruit.github.io/panoptes-workcell-report/video/me340/
- Sam's Club: https://admin-wekruit.github.io/panoptes-workcell-report/video/samsclub/
- Walmart: https://admin-wekruit.github.io/panoptes-workcell-report/video/walmart/

Each page: the video on top; the 3D reconstruction below follows the video's camera while it plays (drag to orbit);
click any object for its card (type and how it was decided, position / top / base / height / width / depth with ±u or
an honest status, time seen, which model is shown: RecGen where it lines up, else a simple shape, else the observed
surface); a searchable object list. The current pages show the round-5 merged runs (2026-09-29).

The site is the repo `admin-wekruit/panoptes-workcell-report` (PUBLIC, Pages from `main` at `/`). The user chose to
publish these pages publicly **with the videos** on 2026-09-30. Other folders there belong to other work (the workcell
reports, `phase2-r6-preview/`): never touch them; only write `video/`.

## After every round (the same URLs, updated in place)

1. In `artifact-report/` (this folder's generator; copy it next to a writable scratch dir), point `VIDEOS` in
   `export.py` at the new runs' mirrors (`runs/<run>/mirror` + the warm call's report id), then:
   `.../panoptes-platform/.venv/bin/python export.py` → `python add_fov.py` → `python build.py`.
   `build.py` writes `pages/video/` (plain .glb, relative links, full HTML) and also the claude.ai artifact variant.
   Update the counts in `landing.html` if they change.
2. Check locally: `python3 -m http.server 8801` in that folder, open `pages/video/<key>/`: the 3D must follow the
   video, clicking must open cards, no failed requests.
3. Publish: `git clone --depth 1 --filter=blob:none --sparse https://github.com/admin-wekruit/panoptes-workcell-report.git`,
   `git sparse-checkout set video`, copy `pages/video/.` into `video/`, commit (message says which runs), `git push origin HEAD:main`.
   If the auto-mode classifier blocks the push, give the user the exact command to run.
4. Wait for Pages (about a minute), check each URL returns 200 (`curl -s -o /dev/null -w "%{http_code}"`), open one in
   the browser pane, then send the user the links with a short Chinese summary: what changed, whether positions are
   right or wrong (numbers), latency. Keep file sizes small (each file well under 50 MB; the three pages are ~115 MB).

The claude.ai artifact variant (private: `pack.py`, then publish `<key>/index.html` with `files.json`) is only for
something that must not be public.
