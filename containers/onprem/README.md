# Panoptes serving stack on-prem / 平台、发布服务与报告网站本地部署

The platform API (import, edit, publish, export), the read-only publication service and the report website, meant to run on a
customer's Linux server with no Modal, no SaaS API and no internet at run time. Internet is needed only where the images are built.
在客户自己的 Linux 服务器上运行平台 API（导入、编辑、发布、导出）、只读发布服务和报告网站；不用 Modal、不调 SaaS、运行时不联网。

**Status / 现状**: every process of this stack has run together, offline, as processes in one Linux container (the Modal proof
below), driven by the two operator CLIs, from an empty install to a report with its measurement layer. `docker-compose.yml` itself
has **not** been brought up anywhere (Modal cannot run Docker; the development Mac has too little free disk for the images); only
`docker compose config` was run. See "What is not tested".
所有进程已在一个断网的 Linux 容器里一起跑通（从空安装到带测量层的报告，用两个操作员 CLI 驱动）；`docker-compose.yml` 本身还没有在任何
机器上启动过，见“未测试”。

| Today 今天 | On-prem | Files 文件 |
|---|---|---|
| platform on the Mac (`uvicorn ehs_spatial.platform.runtime:application --factory`, PostgreSQL, local blobs) | `platform` + `postgres` services | `platform.Dockerfile`, `docker-compose.yml`, `../../.dockerignore` |
| operator scripts in `.platform/` (per report) | one CLI: import → migrateScene v2 + setCalibration → publish → export → prepare | `../../scripts/onprem_publish.py` |
| measurement layers committed to GitHub Pages next to the bundle | one CLI: rebase the published layer onto the on-prem publication, install it in the web root | `../../scripts/onprem_layer.py` |
| Modal app `panoptes-publications-estop` (`modal_apps/publication_site.py`), CORS for GitHub Pages, feedback SQLite on a Modal volume | `publications` service: the same `publication_site.create_app` under uvicorn, feedback SQLite on a local volume, feedback model off; an empty catalog serves an empty library | `serve_publications.py` |
| GitHub Pages: one report bundle per build + `measurement-layer/<publicationId>.json` | `web` service: nginx serves ONE bundle for every publication and proxies `/api/` to `publications` (same origin, no CORS), looking `publications` up per request | `nginx.conf` |

Pins: `python:3.12.13-slim-bookworm@sha256:4766d8b5…` (the platform `.venv` is 3.12.13), uv 0.10.9 by wheel hash, every Python
package from `uv.lock` (`uv sync --frozen`, hashes), `postgres:16.15-bookworm` = `postgres@sha256:efedf359…`, `nginx:1.30.5-alpine` =
`nginx@sha256:0985e772…` (digests read from Docker Hub on 2026-10-05).

## 1. Build (connected machine) / 构建（联网机器）

```sh
cd panoptes-platform      # build context = the checkout; the root .dockerignore is an allowlist (code + uv.lock), so .platform/ never enters
docker build --platform linux/amd64 -f containers/onprem/platform.Dockerfile -t panoptes-platform:onprem .
docker pull --platform linux/amd64 postgres@sha256:efedf3595f1d6f415c08568ba171029bf54052e754cc9f030e3f2412b21f3d67
docker tag postgres@sha256:efedf3595f1d6f415c08568ba171029bf54052e754cc9f030e3f2412b21f3d67 postgres:16.15-bookworm
docker pull --platform linux/amd64 nginx@sha256:0985e772fb9f729e6fa0980da05fca5d9c468e870eed43071545afa9d2e27d94
docker tag nginx@sha256:0985e772fb9f729e6fa0980da05fca5d9c468e870eed43071545afa9d2e27d94 nginx:1.30.5-alpine
IMAGES="panoptes-platform:onprem postgres:16.15-bookworm nginx:1.30.5-alpine"
docker image inspect -f '{{.Id}} {{index .RepoTags 0}}' $IMAGES > image-ids.txt
docker save $IMAGES | gzip > panoptes-serving-images.tar.gz          # saved BY TAG: the tags survive docker load
```

A pull by digest gives an image without a tag, and `docker load` may not restore RepoDigests, so the images are tagged explicitly,
saved by tag, and compose names them by tag with `pull_policy: never` (it never tries the registry). On the server:
`docker load < panoptes-serving-images.tar.gz`, then `docker image inspect -f '{{.Id}} {{index .RepoTags 0}}' $IMAGES | diff - image-ids.txt`
(same Docker image store on both machines; the containerd store reports different ids than the classic one).
Use `--platform` for the server's architecture (an arm64 Mac otherwise builds and pulls arm64).

## 2. Report bundle, built once / 报告网页包（只构建一次）

The report is the Vite app in the workcell worktree (`ehs-spatial` branch `codex/workcell-photo-speed`, `web/`). Build it into its
own directory and copy that into `data/www`; never build with `--emptyOutDir` into `data/www` itself, which would delete the
installed measurement layers (`data/www/measurement-layer/`):

```sh
cd WORKTREE/web && npm ci          # once, connected
VITE_PUBLICATION_ID=/ VITE_API_ORIGIN= node node_modules/vite/bin/vite.js build --outDir /tmp/panoptes-www --emptyOutDir
cp -R /tmp/panoptes-www/. /srv/panoptes/containers/onprem/data/www/     # app.html + assets/; measurement-layer/ is kept
```

- `VITE_API_ORIGIN=` (empty): the bundle calls `/api/…` on its own origin; nginx proxies that to the publication service. Any
  hostname works, no CORS, nothing to rebuild per publication or per host.
- `VITE_PUBLICATION_ID=/` is a coupling with the app's router (`web/src/App.tsx`, `useRoute`), not a documented setting: any non-empty
  value switches the app to read-only report mode and sends every route that is not `#/reports` or `#/reports/ID` to
  `#/reports/<value>`. With `/` that is `#/reports//`, and because the router drops empty path segments it renders the report
  library. So `app.html` lands on the library (every publication of the catalog) and each report is
  `http://HOST:8080/app.html#/reports/PUBLICATION_ID`. Any other value would land every visitor on that one publication. If the
  router changes, check the library landing again (the proof checks it).
- Fonts are in the bundle (`web/src/fonts`: DM Sans and Newsreader woff2, SIL OFL 1.1), and the feedback chat's `deep-chat` element
  is given `font-family: inherit` (`web/src/AgentPanel.tsx`), which stops it from adding its Google Fonts (Inter) stylesheet. The
  proof records every browser request: none leaves the site's origin, the feedback chat included. These web changes
  (`styles.css`, `src/fonts/`, `AgentPanel.tsx`) are uncommitted in the worktree as of 2026-10-05: a build of the branch head
  `5a0d84a` alone still requests Google Fonts.

## 3. Start / 启动

```sh
cd containers/onprem
mkdir -p secrets data/{imports,input,catalog,publication-http,www/measurement-layer}
openssl rand -base64 32 > secrets/postgres_password && chmod 600 secrets/postgres_password
sudo chown 10001 data/imports data/catalog data/publication-http data/www/measurement-layer && chmod 700 data/imports
docker compose up -d     # postgres (healthy over TCP) -> platform (migrates the schema) -> publications -> web
```

A fresh install starts with an empty catalog: the publication service serves an empty report library until the first publish.
nginx looks the `publications` service up through Docker's DNS on every request (cached 10 s), so it starts even while
`publications` is down (502 until it is up) and follows its new address within 10 s after a restart. The postgres healthcheck is `pg_isready -h 127.0.0.1`
(TCP): the image's first-start initialisation runs a temporary socket-only server, which a socket check would already report healthy.
The platform reaches PostgreSQL through the shared unix socket (initdb's socket rule is trust), so the platform holds no database
password; the password file only guards network logins inside the compose network. The platform API listens on 127.0.0.1:8792 only
(operator, SSH tunnel). Report viewers reach only nginx (8080); the publication service has no published port.
Back up with `pg_dump --format=custom` and the `blobs` volume (`docs/platform/OPERATIONS.md`, "Controlled migration"), plus the
`feedback` volume (visitor feedback) and `data/`.

## 4. Publish a capture run / 发布一次采集

Copy the run (`manifest.json`, `input/`, `geometry/`, `evidence/`, `public/`) and its scale JSON into `data/input/RUN/`. The scale
JSON comes from the e-stop CLI `scripts/workcell_estop_scale.py` (ehs-spatial worktree, workcell CPU image), or any file with the
same core (`nativeToMeters`, `limit`, `maxDeviation`, `passed`, `features`). Every feature must name the photo it was measured on as
`… photo N` (scene camera order, e.g. `030 photo 2`), and `--photo` must be exactly that set; `workcell_estop_scale.py` writes the
measured view's image id there, which the CLI cannot map to this scene's cameras, so relabel those features `photo N` first.

```sh
publish() { docker compose exec platform /app/.venv/bin/python scripts/onprem_publish.py /input/RUN/public/scene.json \
  --geometry-root /input/RUN --scale /input/RUN/estop-scale.json --photo 2 \
  --object 'emergency stop BOR1 +00030.RAT01-ES01 (photo 2)' --title 'TITLE' \
  --specification '{"redHeadDiameterM": 0.04, "yellowBodyMaxDiameterM": 0.08, "heightM": 0.1, "source": "SOURCE"}' "$@"; }
publish --dry-run                    # no database: converts the scene, validates the scale and --photo, applies the edit in memory
publish --prepare /publication-http  # import -> migrateScene v2 + setCalibration -> publish -> export -> prepare; prints the ids
docker compose restart publications
```

The CLI takes the database and blob settings from the container environment and exports through the API at 127.0.0.1:8792
(`PANOPTES_PLATFORM_API`). The project's management capability stays in `data/imports/*.management.json` (0600); the CLI reads it
internally and never prints or writes it elsewhere. Re-running is safe (idempotent import, edit, publication; existing exports kept).
The scale is accepted only if it passed, its `nativeToMeters` / `maxDeviation` recompute from its features and its features name
the `--photo` set; the rest of the file is kept as calibration provenance in the revision. Then open `http://HOST:8080/app.html`.
A report without a measurement layer shows the published revision as it is (no confidence chips or photo-measured facts).

## 5. Install a measurement layer / 安装测量层

A layer (`measurement-layer/PUBLICATION_ID.json` + its `.bin` mesh files) applies only to the revision it names. A layer published
for the same capture on another platform (e.g. GitHub Pages for the Modal publication service) must be rebased onto the on-prem
publication: publication, revision and asset ids (matched by sha256) and the revision-derived ids, and only when the on-prem
revision's document equals the layer's revision document up to those ids, the calibration provenance record and float rounding
(`scripts/onprem_layer.py` docstring). Copy the layer directory and the source publication (`GET /api/publications/SOURCE_ID` of the
service the layer was made for) into `data/input/RUN/`:

```sh
layer() { docker compose exec platform /app/.venv/bin/python scripts/onprem_layer.py --published /input/RUN/published.json \
  --layer /input/RUN/layer/measurement-layer/SOURCE_ID.json --publication NEW_ID --catalog /catalog --www /www "$@"; }
layer --dry-run     # compares and rebases, writes nothing; also checks that a tampered document is refused
layer               # writes data/www/measurement-layer/NEW_ID.json and the layer's .bin files (sha256-checked)
```

`NEW_ID` is the `publicationId` printed by `onprem_publish.py`. Nothing else needs a restart: the bundle reads the layer file on load.
A different file already installed under the same `.bin` name is refused (layer files of several publications share the directory).

## Proof / 证明 (Modal as test bench only)

`proof_modal.py` builds `platform.Dockerfile` on Modal (`from_dockerfile`, code-only context incl. the root `.dockerignore`), adds
proof-only PGDG PostgreSQL 16.15, Debian nginx (1.22, with this `nginx.conf`), a stand-in for Docker's DNS at 127.0.0.11 and
Playwright Chromium, blocks the network and runs the whole flow in one container: the healthcheck command against a socket-only and
a TCP server; the publication service on an empty catalog and nginx started while `publications` resolves to a dead address; the
empty library in the browser; `onprem_publish.py` (dry run, run, re-run); a restart of the publication service; the report without a
layer (all models loaded); `onprem_layer.py` (dry run, install, re-run); library → report with the layer (all models loaded, chips,
facts, the view and section buttons, English, one feedback message through `/api/`); the stored feedback row; every browser request of
the session must stay on `127.0.0.1:8080`; and a capability scan. Results and evidence:
`panoptes-public/research-notes/workcell-onprem-platform-2026-10-05/`.

```sh
PANOPTES_ONPREM_RUN=…/candidate-evaluation/bor1-030-01 PANOPTES_ONPREM_WEB=BUILT_WWW .venv/bin/modal run containers/onprem/proof_modal.py \
  --published PUBLICATION.json --layer LAYER_DIR/measurement-layer/cd84d3fb-7d1f-4736-8ffa-d44855e59fab.json \
  --scale .platform/cell030-20261005/estop-scale-v4.json --photo 2 --object '…' --specification '…' \
  --guard 63e3d7f0-7789-5977-aafc-26ccbf206311 --objects 8 --out NEW_DIR
```

2026-10-05, cell 030, network blocked (final of 3 runs, 29/29 checks; numbers in the research note's `results.json`):
- Healthcheck command: against a socket-only server `pg_isready -h 127.0.0.1` returns 2 (the socket check returns 0), against
  the TCP server 0.
- Fresh install: the publication service started on an empty catalog (`/api/publications` = `{"items": []}`, directly and through
  nginx) and the library rendered "no reports". nginx started while `publications` resolved to a dead address (502) and served 200
  0.4 s after the DNS answer changed, without a reload (an answer is cached up to 10 s per nginx worker: the first run saw one more
  502 right after the first 200, most likely another worker's cached answer); `/api/publications/<unknown>?x=1` got the app's own 404, so the `/api/…`
  URI reaches the app unchanged. The stand-in DNS was asked for nothing but `publications`.
- CLI: dry run 4 s (a wrong `--photo` refused); import → edit → publish → export → prepare 131 s; re-run: same ids, no edit, export
  kept. Calibrated revision: `nativeToMeters` 1.1166239729160208 (the published value), schema v2, calibration photo = published
  photo 2 (by sha256). After `restart` the library lists it; the view (= the platform API's) and all 64 assets (120 MB, sha256)
  through `/api/` in 1.7 s.
- The report with NO layer: the same 9 rows (the 8 objects + the floor), no confidence chips, 8/8 models loaded; the layer file
  request is a 404 (the only console error of the session).
- `onprem_layer.py`: the dry run wrote nothing and refused a tampered document and a layer for another revision; the document
  equals the published `c9482042` up to 15 revision-derived ids (a bijection; 15/15 recomputed in both documents; 13 reproduced
  from the new import revision id, 2 after also taking their hashed measurement value, ≤ 1.1e-16 apart), the calibration provenance
  record and 28 numbers ≤ 1.1e-16 apart; 0 content differences. Layer `a443629` installed: publication id, revision id and 3 asset
  ids changed; a re-run gives the same result.
- With the layer: library 0.3 s → report rows 2.1 s, 8 chips (guard "medium" 2.6 s), 8/8 models loaded 15 s, every fact text of the
  3 field-checked objects shown, 10 view / section buttons clicked without a page error, English (UI only; the layer's own names,
  chips and facts are Chinese data), one feedback message typed into the chat: POST 200 through `/api/`, stored once as `saved` /
  `feedback_model_disabled` (no provider), the chat says "Feedback saved. Agent replies are not currently enabled."
- Requests: all 131 browser requests of the session (4 pages, the feedback chat included) went to `127.0.0.1:8080`; no websocket,
  no failed request. Version 2's one `ERR_ABORTED` (a 7.7 MB mesh) was, by the viewer code and the v2 server log, the viewer
  cancelling its own download when the scene changed while models were still loading, then fetching it again (200); not
  reproduced in this version (research note).
- The capability is in none of 31 848 files under `/data` (except `data/imports`), `/tmp` (incl. the PostgreSQL cluster), the web
  root, the nginx logs, `/app`, `/root` and `/etc`, and not in the CLI output. The image's venv equals `uv.lock` (133 packages, 0
  version differences).

## What is not tested / 未测试

- `docker compose up` on any machine: Docker's embedded DNS itself (the proof's stand-in answers `publications` the same way at the
  same address), the shared PostgreSQL socket volume between two containers, named-volume ownership for uid 10001, the postgres
  image's own first-start initialisation with `POSTGRES_PASSWORD_FILE` (the proof checked the healthcheck command's behaviour against
  a socket-only and a TCP PGDG 16.15 server, not the image's entrypoint), `depends_on` ordering, restarts. Only `docker compose
  config` (v2.19.1) was run: the file parses and interpolates.
- The pinned `postgres:16.15-bookworm` and `nginx:1.30.5-alpine` images themselves (the proof used PGDG 16.15 packages and Debian's
  nginx 1.22 with the same `nginx.conf`), and the `docker save` / `docker load` round trip of section 1.
- A customer network that silently drops egress: the proof's network is blocked (connections fail at once). No browser request
  left the origin, so nothing would wait on a dropped connection; services were not probed for egress beyond the browser's requests.
- More than one publication in one catalog / one bundle (the library and `#/reports/ID` routing are generic; only one was served).
- `onprem_layer.py` against a layer whose revision differs in content: only the self-test's tampered copy was refused.
- In the report: the photo / point-cloud / model selector bar is not shown for this publication and the "Representation" select
  was not changed, so those paths were not exercised; mobile widths were not tested.

## Limits / 限制

- The publication service's feedback assistant (Gemini) is off on-prem; visitor feedback is stored only.
- One `publications` process is the single SQLite feedback writer (no multi-worker scaling).
- Model jobs (Modal executor, SaaS providers) are not part of this stack; the GPU stages are `docker/` in the workcell worktree.
- The platform's own management web UI (`panoptes-platform/web`) is not built into the image; the API serves it if
  `PANOPTES_WEB_ROOT` points at a build.
