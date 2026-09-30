# Report page generator (GitHub Pages first; claude.ai artifact variant) — 2026-09-30

**Primary: GitHub Pages** https://admin-wekruit.github.io/panoptes-workcell-report/video/ — see `../SHOW-REPORTS.md` for the publish steps. The notes below describe the generator and the private claude.ai artifact variant.

The interactive Fast-report viewer (`web/src`, `python -m fast_report.layers serve`) only runs locally: a report
mirror is 400-840 MB (full-resolution room meshes, every layer version, two calls per bench) and GitHub holds code and
docs only. What the user looks at now are **self-contained claude.ai artifact pages**, one per video, made from a run's
mirror by the scripts in this folder:

- ME340: https://claude.ai/artifact/5chsXyryTRx29BgTLdT4He
- Sam's Club: https://claude.ai/artifact/PAYw5ESJLvM7fmydiYUNMz
- Walmart: https://claude.ai/artifact/PcQi51Wggr8YD9mU8CtxUH

Each page: the video on top; below it the 3D reconstruction that follows the video's camera while it plays (drag to
orbit); click any object for its card (type and how it was decided, position / top / base / height / width / depth
with +-u or an honest status, time seen, which model is shown); a searchable object list; links between the three.
Models shown: RecGen where it lines up with the video, else a simple shape (box / cylinder / plane / frame), else the
observed surface. Everything is decimated to fit (<= 15 MB a file, <= 64 MB a page).

## Steps

1. Point `VIDEOS` in `export.py` at the runs to show (mirror folder + report id; the round-5 merged warm calls now).
2. `.../panoptes-platform/.venv/bin/python export.py [me340 samsclub walmart]` - reads the latest patches of the report's
   layers (room, surfaces, models, object_cards blob, cameras, video) and writes `<video>/`: `room-<shot>.glb`
   (decimated to 160k triangles, vertex colours), `surfaces-<shot>.glb` (one node per card id), `models-<shot>-<k>.glb`
   (RecGen models placed with their transforms, deduped and decimated, chunks <= ~11 MB), `data.json` (slim cards,
   primitives, camera keyframes), `video.mp4`. The GLB reader is local (`read_glb`): trimesh drops COLOR_0 when a
   material is present. Quadric decimation stalls on open scans, so it falls back to vertex clustering.
3. `python add_fov.py` - adds each shot's vertical field of view from the cameras layer's K (the follow camera needs it).
4. `python pack.py` - wraps each .glb as `.glb.json` (artifacts do not serve .glb) and writes `<video>/files.json`.
5. `python build.py` - fills `index.html` (the page template: three.js 0.147 from jsdelivr, GLTFLoader.parse on the
   decoded JSON) into `<video>/index.html`, with the links in `links.json`, plus a `test.html` for a local check
   (`python3 -m http.server` in this folder, then open `<video>/test.html`).
6. Publish each `<video>/index.html` with the Artifact tool, `files` = the map in its `files.json` (first publish:
   icon `cube`). Put the three URLs in `links.json`, rebuild and republish the index pages only (files are kept).

Scale: metric values come from the cards (estimated with the assumed 1.6 m camera height); the 3D frames are the
shot frames (OpenCV: y down), in metres at that estimated scale.
