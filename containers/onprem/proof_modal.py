"""Proof of the on-prem serving stack in ONE ephemeral Modal CPU container (Modal is only the test bench; nothing is deployed).

What this proves, and what it does not: the platform image (platform.Dockerfile), the operator CLIs (scripts/onprem_publish.py,
scripts/onprem_layer.py), the publication service (serve_publications.py), nginx with this nginx.conf, and the report bundle, all
as processes in one container with the network blocked. It does NOT run Docker or docker-compose.yml: PostgreSQL is PGDG 16.15 (the
packages of the pinned postgres image) and nginx is Debian's, both inside this container; nginx's resolver (Docker's embedded DNS
at 127.0.0.11 under compose) is a tiny stand-in DNS responder here that answers "publications".

In the container: network probe -> PostgreSQL (the compose healthcheck's TCP pg_isready vs a socket-only server) -> platform API
-> publication service on an EMPTY catalog -> nginx started while "publications" resolves to a dead address (502), then follows
the new address without a reload -> browser: empty library -> onprem_publish.py --dry-run / run / re-run -> restart the
publication service -> browser: the report with NO measurement layer, all models loaded -> onprem_layer.py --dry-run / install ->
browser: library -> report with the layer, all models loaded, chips, facts, every view / tab, English, one feedback message
through the /api/ proxy (stored, model off) -> (explanatory) the previous run's click sequence while the models still load -> the stored
feedback row -> every browser request of the whole session must be on 127.0.0.1:8080 -> capability scan (/data, /tmp, the web
root, the nginx logs, /app, /root, /etc).

  PANOPTES_ONPREM_RUN=RUN PANOPTES_ONPREM_WEB=BUILT_WWW modal run containers/onprem/proof_modal.py \
      --published PUBLICATION.json --layer LAYER_DIR/measurement-layer/ID.json --scale SCALE.json --photo 2 \
      --object 'REFERENCE OBJECT' --specification 'JSON' --guard ENTITY_ID --objects 8 --out NEW_DIR
PUBLICATION.json = GET /api/publications/ID of the published report (its snapshot holds the document to compare against).
BUILT_WWW = the report bundle built ONCE: VITE_PUBLICATION_ID=/ VITE_API_ORIGIN= (README section 2).
"""
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import modal

HERE = Path(__file__).resolve().parent  # containers/onprem locally; /root inside the container
PG_VERSION = '16.15-1.pgdg12+2'   # PGDG bookworm build of postgres:16.15-bookworm
PLAYWRIGHT = '1.55.0'
RATE = 4 * .0000131 + 8 * .00000222  # 4 CPU, 8 GiB list rate (USD/s); not an invoice
SECRETISH = re.compile(r'capabilit|token|secret|password|postgres://|postgresql://|authorization', re.I)
WEB = 'http://127.0.0.1:8080'
FEEDBACK_MESSAGE = 'On-prem proof: the guard looks correctly placed in photo 2.'
MODELS_LOADED = """n => { const s = [...document.querySelectorAll('.report-scene-object-model')].map(e => e.dataset.modelState);
    return s.length >= n && !s.includes('sceneModelLoading'); }"""


def stage_context() -> Path:
    """Exactly what platform.Dockerfile COPYs (the root .dockerignore allowlist), so nothing else of the checkout is uploaded."""
    ROOT = HERE.parents[1]
    ctx = Path(tempfile.mkdtemp(prefix='panoptes-onprem-platform-'))
    skip = shutil.ignore_patterns('__pycache__', '*.pyc')
    for name in ('pyproject.toml', 'uv.lock', '.dockerignore'):
        shutil.copy2(ROOT / name, ctx / name)
    for name in ('ehs_spatial', 'panoptes_worker', 'scripts'):
        shutil.copytree(ROOT / name, ctx / name, ignore=skip)
    (ctx / 'containers/onprem').mkdir(parents=True)
    for name in ('platform.Dockerfile', 'serve_publications.py'):
        shutil.copy2(HERE / name, ctx / 'containers/onprem' / name)
    return ctx


if modal.is_local():
    import atexit
    CTX = stage_context()
    atexit.register(shutil.rmtree, CTX, True)
    RUN, BUNDLE = Path(os.environ['PANOPTES_ONPREM_RUN']), Path(os.environ['PANOPTES_ONPREM_WEB'])
    image = (modal.Image.from_dockerfile(CTX / 'containers/onprem/platform.Dockerfile', context_dir=CTX)
             .run_commands(
                 'apt-get update && apt-get install -y --no-install-recommends postgresql-common ca-certificates curl gnupg nginx',
                 '/usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y',
                 f'apt-get install -y --no-install-recommends postgresql-16={PG_VERSION} && rm -rf /var/lib/apt/lists/* /etc/nginx/sites-enabled/default',
                 f'pip install playwright=={PLAYWRIGHT} && playwright install --with-deps chromium')
             .add_local_file(HERE / 'nginx.conf', '/etc/nginx/conf.d/panoptes.conf', copy=True)
             .add_local_file(RUN / 'manifest.json', '/proof/run/manifest.json'))
    for name in ('input', 'geometry', 'evidence', 'public'):  # the files the importer reads (traced), not generation/ or result/
        image = image.add_local_dir(RUN / name, f'/proof/run/{name}')
    image = image.add_local_dir(BUNDLE, '/proof/www')
else:
    image = modal.Image.debian_slim()
app = modal.App('panoptes-onprem-platform-proof')


def dns_responder(sock, answer, log):
    """Stand-in for Docker's embedded DNS: 'publications' A -> answer['ip'] (TTL 1 s), any other name NXDOMAIN."""
    import socket
    while True:
        data, peer = sock.recvfrom(512)
        try:
            i, labels = 12, []
            while data[i]:
                labels.append(data[i + 1:i + 1 + data[i]].decode())
                i += 1 + data[i]
            qtype, name = int.from_bytes(data[i + 1:i + 3], 'big'), '.'.join(labels).lower()
            log.append([name, qtype])
            known, hit = name == 'publications', name == 'publications' and qtype == 1
            header = data[:2] + (b'\x81\x80' if known else b'\x81\x83') + b'\x00\x01' + (b'\x00\x01' if hit else b'\x00\x00') + b'\x00' * 4
            rr = b'\xc0\x0c\x00\x01\x00\x01\x00\x00\x00\x01\x00\x04' + socket.inet_aton(answer['ip']) if hit else b''
            sock.sendto(header + data[12:i + 5] + rr, peer)
        except Exception:  # noqa: BLE001  a malformed query gets no answer
            pass


@app.function(image=image, cpu=4, memory=8 * 1024, timeout=2400, retries=0, min_containers=0, block_network=True)
def proof(reference: dict, published: bytes, layer: bytes, layer_name: str, layer_files: dict, scale: bytes, cli_args: dict,
          guard: str, facts: dict, objects: int):
    import hashlib
    import socket
    import sqlite3
    import subprocess
    import threading
    import urllib.error
    import urllib.request

    start = time.monotonic()
    timings, record, processes, logs, outputs = {}, {'checks': {}}, {}, {}, []
    evidence = {}
    checks = record['checks']
    mark = lambda name: timings.__setitem__(name, round(time.monotonic() - start, 2))
    env = {k: v for k, v in os.environ.items() if not k.startswith('MODAL')}
    sock, pgbin = '/tmp/pgsock', '/usr/lib/postgresql/16/bin'
    api, pubs = 'http://127.0.0.1:8792', 'http://127.0.0.1:8793'
    env.update(PANOPTES_DATABASE_URL=f'postgresql://panoptes@/panoptes?host={sock}',
               PANOPTES_BLOB_ROOT='/data/blobs', PANOPTES_EXECUTOR_BACKEND='local', PYTHONPATH='/app', HOME='/tmp',
               PANOPTES_PLATFORM_API=api)
    pub_env = {'PANOPTES_PUBLICATION_CATALOG': '/data/catalog', 'PANOPTES_PUBLICATION_HTTP': '/data/publication-http',
               'PANOPTES_PUBLICATION_ORIGINS': '', 'PANOPTES_FEEDBACK_DB': '/data/feedback/feedback.sqlite'}
    venv = '/app/.venv/bin/'
    www = Path('/usr/share/nginx/html')

    def sh(cmd, check=True, **kw):
        p = subprocess.run(cmd, capture_output=True, text=True, env=kw.pop('env', env), cwd=kw.pop('cwd', '/app'), timeout=kw.pop('timeout', 1200))
        outputs.append(p.stdout + p.stderr)
        if check and p.returncode:
            raise RuntimeError(f'{cmd[:3]} rc={p.returncode}: {p.stderr[-3000:]}')
        return p.stdout if check else p.returncode

    def serve(name, cmd, url, extra=None):
        log = open(f'/tmp/{name}.log', 'w')
        processes[name] = subprocess.Popen(cmd, cwd='/app', env={**env, **(extra or {})}, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(240):
            if processes[name].poll() is not None:
                raise RuntimeError(f'{name} exited rc={processes[name].returncode}: {Path(f"/tmp/{name}.log").read_text()[-2000:]}')
            try:
                urllib.request.urlopen(url, timeout=5).read()
                return
            except (OSError, urllib.error.HTTPError):
                time.sleep(0.5)
        raise RuntimeError(f'{name} did not start')

    def http(url, headers=None, ok=True):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=300) as response:
                return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read()
        except urllib.error.HTTPError as error:
            if not ok:
                return error.code, {k.lower(): v for k, v in error.headers.items()}, error.read()
            raise RuntimeError(f'{url.split("/api/")[-1][:80]} -> HTTP {error.code}: {error.read()[:500]!r}') from None

    def cli_json(cmd, **kw):
        return json.loads(sh(cmd, **kw).strip().splitlines()[-1])

    events, websockets, page_errors, dns_log, pw, phase = [], [], [], [], None, ['']
    b = record['browser'] = {}
    def summarize_requests():
        """The whole browser session, every page, the feedback chat included, must stay on the site's origin."""
        urls = [e[3] for e in events if e[0] == 'request']
        network = [u.removeprefix('blob:') for u in urls if not u.startswith(('data:', 'about:', 'chrome-extension:'))]
        hosts = sorted({re.sub(r'^([a-z]+://[^/]+).*', r'\1', u) for u in network})
        failed = [e for e in events if e[0] == 'failed']
        ok_later = lambda e: any(x[0] == 'response' and x[3] == e[3] and x[2] == 200 and x[1] >= e[1] for x in events)
        downloads = {}
        for e in events:
            if e[0] == 'request' and (e[3].endswith('/content') or e[3].endswith('.bin')):
                downloads.setdefault(e[4], {}).setdefault(e[3].replace(WEB, ''), 0)
                downloads[e[4]][e[3].replace(WEB, '')] += 1
        b['meshDownloadsRequestedTwice'] = {name: sorted(u for u, n in urls_.items() if n > 1) for name, urls_ in downloads.items()}
        b.update(requestCount=len(urls), requestHosts=hosts, offOrigin=[u[:160] for u in network if not u.startswith(WEB)], websockets=websockets,
                 failedRequests=[[e[5], e[1], e[2], e[3].replace(WEB, '')[:120], 'loaded 200 later' if ok_later(e) else 'not loaded later',
                                  next(([x[1], x[2]] for x in events if x[0] == 'request' and x[3] == e[3] and x[1] <= e[1]), None)] for e in failed][:30],
                 errors=page_errors[:30], fontFiles=sorted({u.rsplit('/', 1)[-1] for u in network if u.endswith('.woff2')}))
        loaded_fonts = {f[0] for f in b.get('fonts', []) if f[1] == 'loaded'}
        main_session = [e for e in failed if e[5] != 'v2SequenceWhileLoading']
        checks.update(onlyLocalRequests=bool(urls) and hosts == [WEB] and not websockets,
                      noFailedRequestsInMainSession=not main_session,
                      selfHostedFontsLoaded={'DM Sans', 'Newsreader'} <= loaded_fonts and all(f.startswith(('dm-sans', 'newsreader')) for f in b['fontFiles']))
        record['dnsQueries'] = sorted({f'{n} type {q}' for n, q in dns_log})

    try:
        # 0. the network really is blocked
        probe = {}
        for name, call in (('tcp 1.1.1.1:443', lambda: socket.create_connection(('1.1.1.1', 443), timeout=5)),
                           ('dns fonts.googleapis.com', lambda: socket.getaddrinfo('fonts.googleapis.com', 443))):
            t = time.monotonic()
            try:
                call(); probe[name] = 'REACHABLE'
            except Exception as error:  # noqa: BLE001
                probe[name] = f'blocked: {type(error).__name__} after {time.monotonic() - t:.2f} s'
        record['networkProbe'] = probe
        checks['networkBlocked'] = all(v.startswith('blocked') for v in probe.values())

        # 1. PostgreSQL 16 (fresh test cluster, trust: no password exists anywhere). The compose healthcheck is pg_isready over TCP:
        # a socket-only server (what the image's first-start init runs) must not pass it; the old socket check would have.
        sh(['install', '-d', '-o', 'postgres', '-m', '0755', '/tmp/pg', sock])
        runuser = ['runuser', '-u', 'postgres', '--']
        isready = [f'{pgbin}/pg_isready', '-U', 'panoptes', '-d', 'panoptes']
        sh(runuser + [f'{pgbin}/initdb', '-D', '/tmp/pg/data', '-U', 'panoptes', '--auth=trust', '-E', 'UTF8'], cwd='/tmp')
        pg_ctl = runuser + [f'{pgbin}/pg_ctl', '-D', '/tmp/pg/data', '-l', '/tmp/postgres.log', '-w']
        sh(pg_ctl + ['-o', f"-c listen_addresses='' -k {sock}", 'start'], cwd='/tmp')
        socket_only = {'tcpCheck': sh(isready + ['-h', '127.0.0.1'], check=False, cwd='/tmp'), 'socketCheck': sh(isready + ['-h', sock], check=False, cwd='/tmp')}
        sh(pg_ctl + ['stop'], cwd='/tmp')
        sh(pg_ctl + ['-o', f"-c listen_addresses='127.0.0.1' -k {sock}", 'start'], cwd='/tmp')
        record['healthcheck'] = {'socketOnlyServer': socket_only, 'tcpServer': {'tcpCheck': sh(isready + ['-h', '127.0.0.1'], check=False, cwd='/tmp')},
                                 'command': 'pg_isready -h 127.0.0.1 -U panoptes -d panoptes (rc 0 = healthy)'}
        checks['healthcheckTcpOnly'] = socket_only['tcpCheck'] != 0 and socket_only['socketCheck'] == 0 and record['healthcheck']['tcpServer']['tcpCheck'] == 0
        sh(runuser + [f'{pgbin}/createdb', '-h', sock, '-U', 'panoptes', 'panoptes'], cwd='/tmp')
        record['postgres'] = sh([f'{pgbin}/postgres', '--version'], cwd='/tmp').strip()
        mark('postgres')

        # 2. platform API (migrates at startup)
        serve('platform', [venv + 'uvicorn', 'ehs_spatial.platform.runtime:application', '--factory', '--host', '127.0.0.1', '--port', '8792'], api + '/api/health')
        mark('platformApi')

        # 3. fresh install: the publication service on an empty catalog, nginx while "publications" points at a dead address
        for d in ('/data/catalog', '/data/publication-http', '/data/feedback'):
            Path(d).mkdir(parents=True, exist_ok=True)
        uvicorn_pubs = [venv + 'uvicorn', 'serve_publications:app', '--factory', '--host', '127.0.0.1', '--port', '8793']
        serve('publications-empty', uvicorn_pubs, pubs + '/api/publications', pub_env)
        empty = json.loads(http(pubs + '/api/publications')[2])
        answer = {'ip': '127.0.0.2'}  # nothing listens there: a stale address
        dns = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        conf = Path('/etc/nginx/conf.d/panoptes.conf')
        try:
            dns.bind(('127.0.0.11', 53))
            record['resolver'] = 'nginx.conf unchanged: resolver 127.0.0.11 (stand-in DNS bound there)'
        except OSError as error:
            dns.bind(('127.0.0.1', 5353))
            conf.write_text(conf.read_text().replace('resolver 127.0.0.11 ', 'resolver 127.0.0.1:5353 '))
            record['resolver'] = f'127.0.0.11:53 not bindable ({error}); proof copy of nginx.conf uses resolver 127.0.0.1:5353'
        threading.Thread(target=dns_responder, args=(dns, answer, dns_log), daemon=True).start()
        shutil.rmtree(www, ignore_errors=True)
        shutil.copytree('/proof/www', www)
        sh(['nginx', '-t'])
        sh(['nginx'])
        # nginx master + workers (gVisor keeps the original argv in /proc/PID/cmdline, so count by comm)
        workers = max(1, sum(1 for c in Path('/proc').glob('[0-9]*/comm') if c.read_text().strip() == 'nginx') - 1)
        stale = [http(WEB + '/api/publications', ok=False)[0] for _ in range(2 * workers)]   # every worker caches the dead address
        answer['ip'] = '127.0.0.1'
        # each nginx worker keeps its own resolver cache (valid=10s): stable once 3 * workers requests in a row are 200
        t, first, followed, streak = time.monotonic(), None, None, 0
        while time.monotonic() - t < 40:
            streak = streak + 1 if http(WEB + '/api/publications', ok=False)[0] == 200 else 0
            first = first or (round(time.monotonic() - t, 1) if streak else None)
            if streak >= 3 * workers:
                followed = round(time.monotonic() - t, 1)
                break
            time.sleep(0.2)
        via = json.loads(http(WEB + '/api/publications')[2])
        missing = http(WEB + '/api/publications/00000000-0000-4000-8000-000000000000?x=1', ok=False)
        record['freshInstall'] = {'directPublications': empty, 'viaNginx': via, 'nginxWorkers': workers, 'nginxWhileStale': sorted(set(stale)),
                                  'secondsToFirst200': first, 'secondsUntilEveryWorkerFollowed': followed,
                                  'unknownPublicationViaNginx': [missing[0], missing[2][:120].decode(errors='replace')]}
        checks['emptyCatalogServes'] = empty == via == {'items': []}
        checks['nginxResolvesPerRequest'] = set(stale) == {502} and followed is not None and followed <= 15
        checks['apiPrefixPassedUnchanged'] = missing[0] == 404 and b'record_not_found' in missing[2]   # the app's 404, not nginx's
        mark('freshInstall')

        # 4. browser for the whole session: every request of every page is recorded (context level)
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        browser = pw.chromium.launch(args=['--ignore-gpu-blocklist', '--enable-gpu', '--enable-unsafe-swiftshader'])
        context = browser.new_context(viewport={'width': 1440, 'height': 1000}, locale='zh-CN')
        t0 = time.monotonic()
        stamp = lambda: round(time.monotonic() - t0, 2)
        context.on('request', lambda r: events.append(['request', stamp(), r.method, r.url, phase[0]]))
        context.on('response', lambda r: events.append(['response', stamp(), r.status, r.url, phase[0]]) if '/api/' in r.url or 'measurement-layer' in r.url else None)
        def failed_request(r):
            try:
                at = r.frame.page.url
            except Exception:  # noqa: BLE001  a request without a page (worker)
                at = None
            events.append(['failed', stamp(), r.failure, r.url, at, phase[0]])
        context.on('requestfailed', failed_request)

        def new_page(name):
            phase[0] = name
            page = context.new_page()
            page.on('pageerror', lambda e: page_errors.append([name, 'pageerror', str(e)[:300]]))
            page.on('console', lambda m: page_errors.append([name, 'console', m.text[:300]]) if m.type == 'error' else None)
            page.on('websocket', lambda ws: websockets.append(ws.url))
            return page

        def shoot(page, name, **kw):
            page.screenshot(path=f'/tmp/{name}', type='jpeg', quality=55, **kw)

        def wait_models(page, phase):
            t = time.monotonic()
            page.wait_for_function(MODELS_LOADED, arg=objects, timeout=600000)
            states = page.eval_on_selector_all('.report-scene-object-row', """els => els.map(e => [e.dataset.entityId,
                e.querySelector('.report-scene-object-model')?.dataset.modelState])""")
            notice = page.eval_on_selector('.report-scene-notice[data-model-coverage]', 'e => ({...e.dataset})') if page.locator('.report-scene-notice[data-model-coverage]').count() else None
            b.setdefault('models', {})[phase] = {'seconds': round(time.monotonic() - t, 1), 'states': states, 'notice': notice}
            return states, notice

        page = new_page('emptyLibrary')
        page.goto(WEB + '/app.html', wait_until='domcontentloaded', timeout=180000)
        page.locator('.report-catalog .empty-state').wait_for(state='visible', timeout=120000)
        b['emptyLibrary'] = {'url': page.url, 'text': ' '.join(page.locator('.report-catalog').inner_text().split())[:200],
                             'reportLinks': page.locator('a.catalog-report-link').count()}
        shoot(page, 'empty-library.jpg')
        page.close()
        checks['emptyLibraryRenders'] = b['emptyLibrary']['reportLinks'] == 0
        mark('browserEmptyLibrary')

        # 5. the operator CLI: dry run (no DB), the real run (+ prepare), and a re-run that must change nothing
        Path('/tmp/scale.json').write_bytes(scale)
        cli = [venv + 'python', 'scripts/onprem_publish.py', '/proof/run/public/scene.json', '--geometry-root', '/proof/run', '--scale', '/tmp/scale.json',
               '--photo', cli_args['photo'], '--object', cli_args['object'], '--specification', cli_args['specification'], '--title', reference['title'],
               '--imports-dir', '/data/imports', '--catalog', '/data/catalog']
        dry = cli_json(cli + ['--dry-run'], env={k: v for k, v in env.items() if k != 'PANOPTES_DATABASE_URL'})
        mark('cliDryRun')
        run = cli_json(cli + ['--prepare', '/data/publication-http'])
        mark('cliPublish')
        again = cli_json(cli)
        mark('cliRerun')
        record['cli'] = {'dryRun': dry, 'run': run, 'rerun': again}
        pid, head = run['publicationId'], run['revisionId']
        checks['cliDryRunPassed'] = dry['operations'] == ['migrateScene', 'setCalibration'] and dry['nativeToMeters'] == reference['nativeToMeters'] and 'wrong --photo refused' in dry['checks']
        checks['cliRerunNoOp'] = (again['publicationId'], again['revisionId'], again['operations'], again['exported']) == (pid, head, [], False)
        revision = json.loads(http(f'{api}/api/revisions/{head}')[2])
        ours_doc, frame = revision['document'], revision['document']['coordinateFrames'][0]
        new_by_sha = {asset['sha256']: asset['id'] for asset in ours_doc['assets']}
        published_photo = reference['document']['coordinateFrames'][0]['scale']['sourceRefs'][0]['imageId']
        record.update(publicationId=pid, projectId=run['projectId'], revisionId=head, branchId=revision['branchId'], schemaVersion=ours_doc.get('schemaVersion'),
                      scale={k: frame['scale'].get(k) for k in ('status', 'nativeToMeters')})
        checks['nativeToMetersSameAsPublished'] = frame['scale'].get('nativeToMeters') == reference['nativeToMeters']
        checks['calibrationPhotoSameAsPublished'] = frame['scale']['sourceRefs'][0].get('imageId') == new_by_sha.get(reference['assetSha'][published_photo])

        # 6. publish -> restart the publication service (compose: docker compose restart publications)
        old = processes.pop('publications-empty')
        old.terminate()
        old.wait(timeout=60)
        serve('publications', uvicorn_pubs, pubs + '/api/publications', pub_env)
        listed = json.loads(http(WEB + '/api/publications')[2])['items']
        checks['publishedListedAfterRestart'] = [item['id'] for item in listed] == [pid]
        mark('publicationService')

        # 7. the view and every asset through nginx's /api/ proxy
        t = time.monotonic()
        status, headers, raw = http(f'{WEB}/api/publications/{pid}/view', {'Accept-Encoding': 'identity'})
        record['view'] = {'via': WEB + '/api/ (nginx -> publications)', 'status': status, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                          'seconds': round(time.monotonic() - t, 3), 'equalsPlatformApiView': json.loads(raw) == json.loads(http(f'{api}/api/publications/{pid}/view')[2])}
        evidence['onprem-view.json'] = raw
        manifest = json.loads(http(f'{WEB}/api/publications/{pid}')[2])['snapshot']['assetManifest']
        t, total, bad = time.monotonic(), 0, []
        for entry in manifest:
            body = http(f"{WEB}/api/assets/{entry['assetId']}/content")[2]
            total += len(body)
            if hashlib.sha256(body).hexdigest() != entry['sha256'] or len(body) != entry['sizeBytes']:
                bad.append(entry['assetId'])
        record['assets'] = {'via': 'nginx', 'count': len(manifest), 'bytes': total, 'seconds': round(time.monotonic() - t, 2), 'mismatched': bad,
                            'largest': max((e['sizeBytes'], e['assetId']) for e in manifest)}
        checks['viewThroughProxy'] = status == 200 and record['view']['equalsPlatformApiView']
        checks['assetsVerified'] = not bad and len(manifest) == len(reference['assetSha'])
        mark('fetchViewAndAssets')

        # 8. the report with NO measurement layer installed
        page = new_page('reportNoLayer')
        page.goto(f'{WEB}/app.html#/reports/{pid}', wait_until='domcontentloaded', timeout=180000)
        page.wait_for_function(f"document.querySelectorAll('.report-scene-object-row').length >= {objects}", timeout=300000)
        states, notice = wait_models(page, 'noLayer')
        page.wait_for_timeout(3000)
        shoot(page, 'report-nolayer.jpg')
        layer_responses = [e[2] for e in events if e[0] == 'response' and e[3].endswith(f'/measurement-layer/{pid}.json')]
        b['noLayer'] = {'rows': len(states), 'rowIds': [s[0] for s in states], 'confidenceChips': page.locator('.report-scene-object-confidence').count(),
                        'factSections': page.locator('[data-measurement-facts]').count(), 'layerResponses': layer_responses}
        page.close()
        no_layer_states = {s[1] for s in states}
        checks['reportWithoutLayer'] = (bool(layer_responses) and set(layer_responses) == {404} and b['noLayer']['confidenceChips'] == 0 and b['noLayer']['factSections'] == 0
                                        and 'sceneModelLoadFailed' not in no_layer_states and notice is not None
                                        and notice.get('modelLoaded') == notice.get('modelCoverage'))
        mark('browserNoLayer')

        # 9. the measurement layer: onprem_layer.py --dry-run, then install into the web root
        Path('/tmp/published.json').write_bytes(published)   # as an operator copies them into data/input/RUN/
        Path('/tmp/layer-source/measurement-layer').mkdir(parents=True)
        Path(f'/tmp/layer-source/measurement-layer/{layer_name}').write_bytes(layer)
        for url, data in layer_files.items():
            Path('/tmp/layer-source', url).write_bytes(data)
        tool = [venv + 'python', 'scripts/onprem_layer.py', '--published', '/tmp/published.json', '--layer', f'/tmp/layer-source/measurement-layer/{layer_name}',
                '--publication', pid, '--catalog', '/data/catalog', '--www', str(www)]
        layer_dry = cli_json(tool + ['--dry-run'])
        wrote_on_dry_run = (www / 'measurement-layer').exists()
        layer_run = cli_json(tool)
        layer_again = cli_json(tool)
        installed = json.loads((www / 'measurement-layer' / f'{pid}.json').read_text())
        record['layerTool'] = {'dryRun': layer_dry, 'run': layer_run, 'rerunSameResult': layer_again == layer_run,
                               'installedSha256': hashlib.sha256((www / 'measurement-layer' / f'{pid}.json').read_bytes()).hexdigest()}
        comparison = layer_run['comparison']
        record['documentComparison'] = comparison
        files_ok = all(hashlib.sha256((www / a['url']).read_bytes()).hexdigest() == a['sha256'] for a in installed.get('assets', []))
        checks['layerDryRunWroteNothing'] = not wrote_on_dry_run and 'refused' in layer_dry['selfTest']
        checks['documentMatchesPublished'] = comparison['contentDifferences'] == 0 and comparison['renamingBijection'] and comparison['scaleEqual']
        checks['layerInstalled'] = (installed['publicationId'], installed['revisionId']) == (pid, head) and files_ok and record['layerTool']['rerunSameResult']
        evidence['rebased-layer.json'] = json.dumps(installed, ensure_ascii=False, indent=1).encode()
        layer_objects = set(installed['labels'])
        floors = {e['id'] for e in ours_doc['entities'] if e.get('geometryRole') == 'floor'}
        mark('layer')

        # 10. library -> report with the layer: rows, chips, all models loaded, facts, every view / tab, English, feedback
        page = new_page('reportWithLayer')
        t = time.monotonic()
        timing = b['timings'] = {}
        shot = 'library.jpg'
        try:
            page.goto(WEB + '/app.html', wait_until='domcontentloaded', timeout=180000)
            link = page.locator(f'a.catalog-report-link[href="#/reports/{pid}"]')
            link.wait_for(state='visible', timeout=120000)
            timing['library'] = round(time.monotonic() - t, 2)
            b['libraryUrl'] = page.url
            page.wait_for_timeout(1500)
            shoot(page, 'library.jpg')
            link.click()
            shot = 'report.jpg'
            page.wait_for_function(f"document.querySelectorAll('.report-scene-object-row').length >= {objects}", timeout=300000)
            timing['objectList'] = round(time.monotonic() - t, 2)
            rows = [r[0] for r in page.eval_on_selector_all('.report-scene-object-row', 'els => els.map(e => [e.dataset.entityId])')]
            b.update(reportUrl=page.url, objectRows=sum(r in layer_objects for r in rows), referenceSurfaceRows=[r for r in rows if r in floors],
                     countBadge=page.locator('.report-scene-count').first.inner_text())
            checks['objectList'] = len(rows) == len(set(rows)) and set(rows) == layer_objects | floors and b['objectRows'] == objects == len(layer_objects)
            checks['sameRowsWithoutLayer'] = sorted(rows) == sorted(b['noLayer']['rowIds'])
            chip = page.locator(f'.report-scene-object-row[data-entity-id="{guard}"] .report-scene-object-confidence')
            chip.wait_for(state='visible', timeout=120000)
            timing['guardChip'] = round(time.monotonic() - t, 2)
            chips = {row: page.locator(f'.report-scene-object-row[data-entity-id="{row}"] .report-scene-object-confidence').count() for row in layer_objects}
            b.update(guardChip={'level': chip.get_attribute('data-confidence'), 'text': chip.inner_text()}, confidenceChips=sum(chips.values()))
            checks['confidenceChips'] = chip.get_attribute('data-confidence') == installed['confidence'][guard]['level'] and all(n == 1 for n in chips.values())
            states, notice = wait_models(page, 'withLayer')
            timing['allModelsLoaded'] = round(time.monotonic() - t, 2)
            b['rowList'] = page.eval_on_selector_all('.report-scene-object-row', """els => els.map(e => [e.dataset.entityId,
                e.querySelector('strong')?.innerText.replace(/\\s+/g, ' '), e.querySelector('.report-scene-object-model')?.dataset.modelState])""")
            loaded_states = {s[1] for s in states}
            checks['allModelsLoaded'] = (notice is not None and notice.get('modelLoaded') == notice.get('modelCoverage')
                                         and not loaded_states & {'sceneModelLoading', 'sceneModelLoadFailed'})
            page.wait_for_timeout(3000)
            shoot(page, 'report.jpg')
        except Exception:
            shoot(page, shot)
            raise
        found, squash = {}, lambda s: ' '.join(s.split())
        for entity, texts in facts.items():
            page.locator(f'.report-scene-object-row[data-entity-id="{entity}"] > button').first.click()
            section = page.locator(f'[data-measurement-facts="{entity}"]').first
            section.wait_for(state='visible', timeout=120000)
            shown = squash(section.inner_text())
            found[entity] = {'facts': len(texts), 'missing': [x[:80] for x in texts if squash(x) not in shown]}
            if entity == next(iter(facts)):
                page.wait_for_timeout(2000)
                section.screenshot(path='/tmp/facts.jpg', type='jpeg', quality=60)
        b['facts'] = found
        checks['factsShown'] = all(not f['missing'] for f in found.values())
        timing['facts'] = round(time.monotonic() - t, 2)
        # every visible view / tab button, once
        errors_before, views = len(page_errors), []
        for selector in ('.report-scene-primary-views button', '.report-scene-view-switch:not(.report-scene-mobile-views) button', '.report-index button'):
            buttons = page.locator(selector)
            for i in range(buttons.count()):
                button = buttons.nth(i)
                label = squash(button.inner_text())[:40]
                if not button.is_visible() or button.is_disabled():
                    views.append([selector.split()[0], label, 'not shown / disabled'])
                    continue
                try:
                    button.click(timeout=10000)
                    page.wait_for_timeout(1500)
                    views.append([selector.split()[0], label, 'clicked'])
                except Exception as error:  # noqa: BLE001
                    views.append([selector.split()[0], label, f'click failed: {type(error).__name__}'])
        page.locator('.report-scene-view-switch:not(.report-scene-mobile-views) button.report-scene-quad').first.click(timeout=10000)
        page.wait_for_timeout(1500)
        b['views'] = views
        b['errorsDuringViews'] = page_errors[errors_before:]
        checks['viewsExercised'] = sum(v[2] == 'clicked' for v in views) >= 6 and not b['errorsDuringViews']
        timing['views'] = round(time.monotonic() - t, 2)
        # English
        page.select_option('select[aria-label="Language"]', 'en')
        page.wait_for_timeout(1500)
        chip_en = page.locator(f'.report-scene-object-row[data-entity-id="{guard}"] .report-scene-object-confidence').first.inner_text()
        b['english'] = {'guardChip': squash(chip_en), 'objectsHeading': page.locator('.report-scene-object-rail').first.get_attribute('aria-label'),
                        'htmlLang': page.evaluate('document.documentElement.lang')}
        shoot(page, 'report-en.jpg')
        # one feedback message through the /api/ proxy (stored only; the model is off on-prem)
        page.locator(f'.report-scene-object-row[data-entity-id="{guard}"] > button').first.click()
        with page.expect_response(lambda r: '/feedback?conversationId=' in r.url, timeout=120000) as history:
            page.locator('.report-inspector-agent').first.click()
        page.locator('deep-chat').first.wait_for(state='attached', timeout=120000)
        page.wait_for_timeout(2000)
        b['feedback'] = {'historyStatus': history.value.status}
        def send(action):
            with page.expect_response(lambda r: r.request.method == 'POST' and r.url.endswith(f'/entities/{guard}/feedback'), timeout=30000) as posted:
                action()
            return posted.value

        def typed():
            page.locator('deep-chat #text-input').first.click(timeout=15000)
            page.keyboard.type(FEEDBACK_MESSAGE)
            page.keyboard.press('Enter')
        try:
            response = send(typed)
            b['feedback']['sentBy'] = 'typed into deep-chat, Enter'
        except Exception as error:  # noqa: BLE001  deep-chat's own API, as the report's identity button uses
            response = send(lambda: page.evaluate('m => document.querySelector("deep-chat").submitUserMessage({text: m})', FEEDBACK_MESSAGE))
            b['feedback']['sentBy'] = f'deep-chat submitUserMessage (typing: {type(error).__name__})'
        reply = response.json()
        page.wait_for_timeout(3000)
        saved_text = page.locator('deep-chat').get_by_text('Feedback saved', exact=False).count()
        b['feedback'].update(postStatus=response.status, postUrl=response.url.replace(WEB, ''), replyStatus=reply.get('status'),
                             replyErrorCode=reply.get('errorCode'), replyLanguage=reply.get('language'), savedMessageShown=saved_text > 0)
        shoot(page, 'feedback-en.jpg')
        timing['feedback'] = round(time.monotonic() - t, 2)
        checks['english'] = 'Confidence' in chip_en and b['english']['objectsHeading'] == 'Scene objects' and saved_text > 0
        b['fonts'] = page.evaluate("""async () => { await document.fonts.ready;
            return [...document.fonts].map(f => [f.family.replace(/"/g, ''), f.status, f.unicodeRange.slice(0, 12)]) }""")
        page.close()
        # explanatory, not a check: the v2 run's one ERR_ABORTED (the 7.7 MB fence mesh) came from its sequence, which did not
        # wait for the models: facts objects clicked, then the guard and the feedback chat, while meshes were still loading
        page = new_page('v2SequenceWhileLoading')
        clicks = b['v2SequenceClicks'] = []
        try:
            page.goto(f'{WEB}/app.html#/reports/{pid}', wait_until='domcontentloaded', timeout=180000)
            page.wait_for_function(f"document.querySelectorAll('.report-scene-object-row').length >= {objects}", timeout=300000)
            for entity in list(facts) + [guard]:
                page.locator(f'.report-scene-object-row[data-entity-id="{entity}"] > button').first.click()
                clicks.append([stamp(), 'select', entity[:8]])
                if entity != guard:
                    page.locator(f'[data-measurement-facts="{entity}"]').first.wait_for(state='visible', timeout=120000)
            page.locator('.report-inspector-agent').first.click()
            clicks.append([stamp(), 'feedback chat', guard[:8]])
            b['v2SequenceStatesAtChat'] = [s[1] for s in page.eval_on_selector_all('.report-scene-object-row', '''els => els.map(e =>
                [e.dataset.entityId, e.querySelector('.report-scene-object-model')?.dataset.modelState])''')]
            wait_models(page, 'v2SequenceWhileLoading')
            clicks.append([stamp(), 'all models loaded', ''])
        except Exception as error:  # noqa: BLE001
            b['v2SequenceError'] = str(error)[:300]
        page.close()
        context.close()
        browser.close()
        mark('browser')

        # 11. the stored feedback row
        with sqlite3.connect('/data/feedback/feedback.sqlite') as db:
            turns = db.execute('SELECT t.status, t.error_code, t.body, c.publication_id, c.entity_id, t.provider FROM feedback_turns t JOIN conversations c ON c.id = t.conversation_id').fetchall()
        record['feedbackStore'] = [{'status': s, 'errorCode': e, 'message': json.loads(body)['message'], 'language': json.loads(body)['language'],
                                    'publicationId': p, 'entityId': en, 'provider': prov} for s, e, body, p, en, prov in turns]
        checks['feedbackStored'] = (b['feedback']['postStatus'] == 200 and len(turns) == 1 and turns[0][:2] == ('saved', 'feedback_model_disabled')
                                    and json.loads(turns[0][2])['message'] == FEEDBACK_MESSAGE and turns[0][3:] == (pid, guard, None))

        record['venvFreeze'] = sh(['uv', 'pip', 'freeze', '--python', '/app/.venv/bin/python']).split()
    except Exception:  # noqa: BLE001  keep what was learned; the failing step is in the traceback
        import traceback
        record['error'] = '\n'.join(line for line in traceback.format_exc().splitlines() if not SECRETISH.search(line))[-4000:]
    finally:
        try:
            summarize_requests()
            b['eventsTail'] = [e[:2] + [str(x).replace(WEB, '')[:120] for x in e[2:]] for e in events[-15:]]
            if pw:
                pw.stop()   # also closes the browser after a failure
        except Exception as error:  # noqa: BLE001
            record['summaryError'] = repr(error)[:300]
        for name, process in processes.items():
            process.terminate()
        subprocess.run(['nginx', '-s', 'quit'], capture_output=True)
        subprocess.run(['runuser', '-u', 'postgres', '--', f'{pgbin}/pg_ctl', '-D', '/tmp/pg/data', 'stop'], capture_output=True, cwd='/tmp')
        for name in ('platform', 'publications-empty', 'publications', 'postgres'):
            path = Path(f'/tmp/{name}.log')
            if path.exists():
                logs[name] = '\n'.join(line for line in path.read_text(errors='replace').splitlines() if not SECRETISH.search(line))[-2500:]
        for name in ('access.log', 'error.log'):
            path = Path('/var/log/nginx', name)
            if path.exists():
                logs['nginx-' + name] = '\n'.join(path.read_text(errors='replace').splitlines()[-12:])
    record.update(timings=timings, containerSeconds=round(time.monotonic() - start, 1), logs=logs)
    # the capability: only ever in /data/imports (the importer's 0600 file) - not in the CLI output, the record, the database
    # files, the catalog, the prepared bodies, the website, any log, the image (/app), /root or /etc
    management = next(Path('/data/imports').glob('*.management.json'), None)
    if management:
        secret = json.loads(management.read_text())['capability']
        roots = ('/data', '/tmp', '/usr/share/nginx/html', '/var/log/nginx', '/app', '/root', '/etc')
        scanned = [0]

        def holds(path):
            try:
                if path.is_file() and not path.is_symlink():
                    scanned[0] += 1
                    return secret.encode() in path.read_bytes()
            except OSError:
                pass
            return False
        hits = [str(p) for root in roots for p in Path(root).rglob('*') if not str(p).startswith('/data/imports/') and holds(p)]
        text = json.dumps(record, ensure_ascii=False)
        record = json.loads(text.replace(secret, '[redacted]'))
        record['capabilityScan'] = {'roots': [r + (' (except /data/imports)' if r == '/data' else '') for r in roots], 'filesScanned': scanned[0],
                                    'filesContaining': hits, 'inCliOutput': any(secret in o for o in outputs), 'inRecord': secret in text,
                                    'managementFileMode': oct(management.stat().st_mode & 0o777)}
        record['checks']['capabilityOnlyInImportsDir'] = not hits and not record['capabilityScan']['inCliOutput'] and secret not in text
        secret = text = None
    names = ('empty-library.jpg', 'report-nolayer.jpg', 'library.jpg', 'report.jpg', 'facts.jpg', 'report-en.jpg', 'feedback-en.jpg')
    images = {name: Path(f'/tmp/{name}').read_bytes() for name in names if Path(f'/tmp/{name}').exists()}
    return {'record': record, 'evidence': {**evidence, **images}}


@app.local_entrypoint()
def main(published: str, layer: str, scale: str, photo: str, object: str, specification: str, guard: str, out: str, objects: int = 8):
    import hashlib
    import tomllib
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    published_bytes = Path(published).read_bytes()
    publication = json.loads(published_bytes)
    snapshot = publication['snapshot']
    document = snapshot['revision']['document']
    asset_sha = {a['id']: a['sha256'] for a in document['assets']} | {e['assetId']: e['sha256'] for e in snapshot['assetManifest']}
    reference = {'publicationId': publication['id'], 'title': publication['title'], 'revisionId': snapshot['revision']['id'], 'document': document,
                 'assetSha': asset_sha, 'nativeToMeters': document['coordinateFrames'][0]['scale']['nativeToMeters']}
    layer_path = Path(layer)
    layer_json = json.loads(layer_path.read_bytes())
    files = {a['url']: (layer_path.parent.parent / a['url']).read_bytes() for a in layer_json.get('assets', [])}
    # every fact of every object that carries a field check must be shown verbatim
    facts = {entity: [f['text'] for f in items] for entity, items in layer_json['facts'].items() if any(f.get('kind') == 'check' for f in items)}
    destination.mkdir(parents=True)
    start = time.monotonic()
    result = proof.remote(reference, published_bytes, layer_path.read_bytes(), layer_path.name, files, Path(scale).read_bytes(),
                          {'photo': photo, 'object': object, 'specification': specification}, guard, facts, objects)
    call = time.monotonic() - start
    for name, data in result['evidence'].items():
        (destination / name).write_bytes(data)
    record = result['record']
    norm = lambda name: re.sub(r'[-_.]+', '-', name).lower()
    locked = {norm(p['name']): p['version'] for p in tomllib.loads((HERE.parents[1] / 'uv.lock').read_text())['package'] if 'version' in p}
    freeze = {norm(k): v for k, v in (line.split('==', 1) for line in record.pop('venvFreeze', []) if '==' in line)}
    record['venvVsLock'] = {'installed': len(freeze), 'locked': len(locked),
                            'versionMismatches': sorted(k for k, v in freeze.items() if locked.get(k) != v),
                            'lockedNotInstalled': sorted(set(locked) - set(freeze))}
    record['reference'] = {k: reference[k] for k in ('publicationId', 'revisionId', 'nativeToMeters')} | {
        'layer': str(layer_path), 'layerSha256': hashlib.sha256(layer_path.read_bytes()).hexdigest(), 'scale': str(scale),
        'scaleSha256': hashlib.sha256(Path(scale).read_bytes()).hexdigest(), 'publishedSha256': hashlib.sha256(published_bytes).hexdigest()}
    record['spend'] = {'mode': 'ephemeral modal run', 'hardware': '4 CPU, 8 GiB, no GPU, block_network', 'functionSeconds': record['containerSeconds'],
                       'callSeconds': round(call, 1), 'estimateUsd': round(RATE * record['containerSeconds'], 4),
                       'callWindowEstimateUsd': round(RATE * call, 4), 'rateSource': 'https://modal.com/pricing', 'imageBuild': 'not itemised'}
    (destination / 'record.json').write_text(json.dumps(record, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'checks': record['checks'], 'publicationId': record.get('publicationId'), 'timings': record['timings'],
                      'spend': record['spend'], 'error': record.get('error', '')[-1500:]}, ensure_ascii=False))
