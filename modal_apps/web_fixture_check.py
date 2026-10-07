"""Real-browser check of a locally built web bundle with a fixture overlay, run in the cloud (nothing renders locally).

A CPU container serves SITE (the built bundle) with EXTRA's files laid over it (e.g. a fixture measurement-layer/<id>.json)
from a static host on 127.0.0.1 inside the container, opens one report object in Chromium (software WebGL), then the
other objects through the object list, and saves the layer-box panel, legend, 3D overlay labels and screenshots (3D view,
inspector, object list), then the show-all mode and a front model-preview capture. With --expect-boxes the expectations come
from the fixture layer itself (its "boxes": highlight flags, reasons, retake hints, lifted boxes); a malformed box in it must be
skipped without a page error.
The report API only allows the Pages origin, so the browser's API requests are fetched by Playwright and answered with
that local origin added (reads only; nothing is written anywhere).

modal run modal_apps/web_fixture_check.py --site BUNDLE_DIR --extra FIXTURE_DIR --hash-path 'app.html#/reports/PUB?snapshot=1' \
  --objects ID1,ID2 [--expect-boxes] [--english] --out NEW_DIR
"""
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP
import io
import json
from pathlib import Path
import re
import tarfile
import time

import modal

PLAYWRIGHT = '1.55.0'
API = 'https://wekruit-livekit-agents--panoptes-publications-estop-web.modal.run'
RATE = 4 * .0000131 + 8 * .00000222  # 4 CPU, 8 GiB list rate (USD/s); not an invoice
HIGHLIGHT, FLOOR_TICK = '#ff3fa4', 'rgba(255,255,255,.85)'
HATCHES = ('rgba(92,201,140', 'rgba(240,180,60', 'rgba(245,130,47', 'rgba(235,70,70')
OUTLINES = (HIGHLIGHT, 'rgba(255,63,164', '#14a38b', '#8aa39a')  # pink = highlighted (opaque when selected), teal selected, grey other
BOX_COLOURS = OUTLINES + (FLOOR_TICK,) + HATCHES
FACE_NAMES = ('前面', '后面', '左面', '右面', '顶面', '底面')

app = modal.App('web-fixture-check')
image = (modal.Image.from_registry(f'mcr.microsoft.com/playwright:v{PLAYWRIGHT}-noble', add_python='3.11')
         .env({'PLAYWRIGHT_BROWSERS_PATH': '/ms-playwright'})
         .pip_install(f'playwright=={PLAYWRIGHT}')
         .run_commands('python -m playwright install chromium'))  # a no-op when the image's browsers already match


def tar_dir(path: str) -> bytes:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as bundle:
        if path:
            bundle.add(path, arcname='.')
    return data.getvalue()


@app.function(image=image, cpu=4, memory=8 * 1024, timeout=1800, retries=0, min_containers=0)
def check(site: bytes, extra: bytes, hash_path: str, objects: list, expect_boxes: bool, english: bool):
    import functools, http.server, threading
    from playwright.sync_api import sync_playwright
    started = time.monotonic()
    root, out = Path('/fixture-site'), Path('/out')
    root.mkdir(); out.mkdir()
    for blob in (site, extra):  # extra's files replace the site's
        with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as bundle:
            bundle.extractall(root, filter='data')

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 8000), functools.partial(Quiet, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = 'http://127.0.0.1:8000'
    record = {'hashPath': hash_path, 'objects': objects, 'expectBoxes': expect_boxes, 'steps': [], 'errors': [], 'apiFailures': [], 'cancelled': []}

    def api(route, request):
        cors = {'access-control-allow-origin': origin, 'access-control-allow-methods': 'GET, POST, OPTIONS',
                'access-control-allow-headers': request.headers.get('access-control-request-headers', '*'),
                'access-control-expose-headers': 'ETag, Content-Length, Content-Range'}
        if request.method == 'OPTIONS':
            return route.fulfill(status=204, headers=cors)
        try:
            response = route.fetch(timeout=240000)
        except Exception as error:  # keep the page's own error path
            record['apiFailures'].append([request.url[len(API):][:160], str(error)[:200]])
            if 'aborted' in str(error) or 'disposed' in str(error):
                record['cancelled'].append(request.url)
            return route.abort()
        headers = {k: v for k, v in response.headers.items() if k.lower() not in ('access-control-allow-origin', 'content-encoding', 'content-length', 'transfer-encoding')}
        route.fulfill(response=response, headers={**headers, **cors})

    def overlay(page):
        """Texts and box-coloured line counts of the 3D view's SVG overlay (face labels = '前面·低' style texts)."""
        found = page.evaluate("""colours => { const pane = document.querySelector('[data-pane="spatial"]'); if (!pane) return null;
          const lines = [...pane.querySelectorAll('svg line')].map(l => l.getAttribute('stroke') || '');
          const count = list => lines.filter(s => list.some(c => s.startsWith(c))).length;
          return { texts: [...pane.querySelectorAll('svg text')].map(t => t.textContent).filter(Boolean), boxLines: count(colours.box),
                   hatchLines: count(colours.hatch), highlightLines: count([colours.highlight]), floorTicks: count([colours.floor]), outlineLines: count(colours.outline), lines: lines.length }; }""",
            {'box': list(BOX_COLOURS), 'hatch': list(HATCHES), 'highlight': HIGHLIGHT, 'floor': FLOOR_TICK, 'outline': list(OUTLINES)})
        if found:
            found['faceLabels'] = [t for t in found['texts'] if t.split('·')[0] in FACE_NAMES]
        return found

    def texts(page, selector):
        return page.locator(selector).all_inner_texts()

    pub = re.search(r'#/reports/([^?&/]+)', hash_path)
    layer_file = root / 'measurement-layer' / f'{pub.group(1) if pub else ""}.json'
    boxes = json.loads(layer_file.read_text()).get('boxes', {}) if layer_file.is_file() else {}
    real = {k: b for k, b in boxes.items() if not k.startswith('fixture-malformed')}  # the fixture's one malformed box must be skipped

    def fixed(v, q):
        """JS toFixed text of v (it breaks exact ties away from zero, Python's format to even: either is accepted)."""
        return {str(Decimal(v).quantize(Decimal(q), rounding=r)) for r in (ROUND_HALF_UP, ROUND_HALF_EVEN)}

    def cm_texts(metres):
        v = metres * 100
        return fixed(v, '0.1' if abs(v) < 100 else '1')

    def step_checks(step, box):
        panel, ov = '\n'.join(step['panel']), step['overlay'] or {}
        dims = box['dims'].values()
        return {
            'panel rows 长 宽（进深） 高 离地 ± 置信度': all(w in panel for w in ('长', '宽（进深）', '高', '离地', '±', '置信度')),
            'panel values': all(cm_texts(d['valueM']) & set(re.findall(r'-?\d+(?:\.\d)?', panel)) for d in dims),
            'panel sigma or —': all(any(f'± {t} cm' in panel for t in (fixed(d['sigmaCm'], '0.1') if d['sigmaCm'] is not None else {'—'})) for d in dims),
            'panel highlight = layer flag': ('低置信度（需补拍）' in panel) == box['highlight'],
            'panel reasons': all(r in panel for r in box.get('highlightReasons') or []),
            'panel retake hints': all(f['need'] in panel for f in box['faces'].values() if f.get('need')),
            'panel snapNote': not box.get('snapNote') or box['snapNote'] in panel,
            '3D face labels 1-3 (front-facing only)': 1 <= len(ov.get('faceLabels', [])) <= 3,
            '3D face hatches': ov.get('hatchLines', 0) > 0,
            '3D size label': sum('长×宽×高' in t for t in ov.get('texts', [])) == 1,
            '3D highlight outline': not box['highlight'] or ov.get('highlightLines', 0) >= 12,
            '3D floor tick over the frame floor': box['floorContact'] or box['bottomM'] <= .005 or ov.get('floorTicks', 0) >= 1,
            '3D outlines = highlighted + selected boxes': ov.get('outlineLines') == 12 * len({k for k in real if real[k].get('highlight') is True} | {step['object']}),
        }

    with sync_playwright() as p:
        browser = p.chromium.launch(args=['--ignore-gpu-blocklist', '--enable-gpu', '--enable-unsafe-swiftshader'])
        try:
            page = browser.new_page(viewport={'width': 1600, 'height': 1200})
            page.on('pageerror', lambda error: record['errors'].append(str(error)[:300]))
            def console(message):
                # an API download the page itself cancelled (object switch, reload) is answered with an abort: not a page error
                if message.type == 'error' and not ('net::ERR_FAILED' in message.text and message.location.get('url') in record['cancelled']):
                    record['errors'].append(f"{message.text[:300]} @ {message.location.get('url', '')[:160]}")
            page.on('console', console)
            page.route(lambda url: url.startswith(API), api)
            page.route(lambda url: 'fonts.googleapis.com' in url or 'fonts.gstatic.com' in url, lambda route, _: route.fulfill(status=200, content_type='text/css', body=''))
            page.goto(f'{origin}/{hash_path}&object={objects[0]}', wait_until='domcontentloaded', timeout=120000)
            row = lambda entity: f'.report-scene-object-row[data-entity-id="{entity}"] > button:first-child'
            page.locator(f'{row(objects[0])}[aria-pressed="true"]').wait_for(state='visible', timeout=300000)
            page.locator('.report-scene-view-switch:not(.report-scene-mobile-views) button[aria-controls$="-spatial"]').click()
            page.wait_for_timeout(20000)  # models arrive after the document
            for i, entity in enumerate(objects):
                if i:
                    page.locator(row(entity)).click()
                    page.wait_for_timeout(6000)
                step = {'object': entity, 'panel': texts(page, f'[data-layer-box="{entity}"]'), 'legend': texts(page, '.report-box-legend'),
                        'legendCount': page.locator('.report-box-legend').first.get_attribute('data-highlight-count') if page.locator('.report-box-legend').count() else None,
                        'recaptureRows': page.eval_on_selector_all('.report-scene-object-row[data-box-recapture]', 'rows => rows.map(r => r.dataset.entityId)'),
                        'rows': page.eval_on_selector_all('.report-scene-object-row', 'rows => rows.map(r => r.dataset.entityId)'),
                        'facts': texts(page, f'[data-measurement-facts="{entity}"]'), 'overlay': overlay(page)}
                record['steps'].append(step)
                page.locator('[data-pane="spatial"]').screenshot(path=str(out / f'{i + 1}-spatial.png'))
                page.locator('.report-scene-inspector').screenshot(path=str(out / f'{i + 1}-inspector.png'))
                page.locator('.report-scene-object-rail').screenshot(path=str(out / f'{i + 1}-objects.png'))
            toggle = page.get_by_label('显示全部尺寸框')
            record['toggle'] = toggle.count()
            if toggle.count():
                toggle.check(); page.wait_for_timeout(3000)
                record['allBoxes'] = overlay(page)
                page.locator('[data-pane="spatial"]').screenshot(path=str(out / 'all-boxes-spatial.png'))
                # hover the screen centres of the five largest other boxes (12 edges each, in draw order) until the viewer names one
                centres = page.evaluate("""() => { const svg = [...document.querySelectorAll('[data-pane="spatial"] svg')].find(s => s.querySelector('line'));
                  const r = svg.getBoundingClientRect(), lines = [...svg.querySelectorAll('line')].filter(l => (l.getAttribute('stroke') || '').startsWith('rgba(255,63,164') || l.getAttribute('stroke') === '#8aa39a');
                  const boxes = []; for (let i = 0; i + 12 <= lines.length; i += 12) { const p = lines.slice(i, i + 12).flatMap(l => [[+l.getAttribute('x1'), +l.getAttribute('y1')], [+l.getAttribute('x2'), +l.getAttribute('y2')]]);
                    const xs = p.map(q => q[0]), ys = p.map(q => q[1]); boxes.push({ x: r.left + xs.reduce((a, b) => a + b) / xs.length, y: r.top + ys.reduce((a, b) => a + b) / ys.length, size: (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys)) }); }
                  return boxes.sort((a, b) => b.size - a.size).slice(0, 5); }""")
                before = set(record['allBoxes']['texts'])
                for centre in centres:
                    page.mouse.move(centre['x'], centre['y']); page.wait_for_timeout(1500)
                    record['allBoxesHover'] = overlay(page)
                    if set(record['allBoxesHover']['texts']) - before:
                        break
                page.locator('[data-pane="spatial"]').screenshot(path=str(out / 'all-boxes-hover-spatial.png'))
                toggle.uncheck(); page.wait_for_timeout(1000)
            page.locator('.report-scene-quad').click(); page.wait_for_timeout(6000)
            page.screenshot(path=str(out / 'quad.png'))
            # a front model-preview capture of the selected object: the layer boxes (pink when highlighted) stay out of it
            front = page.locator('.report-model-preview nav button').nth(1)
            if front.count():
                front.click()
                image = page.locator('.report-model-preview img')
                image.wait_for(state='visible', timeout=60000)
                record['capturePinkPixels'] = page.evaluate("""img => { const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
                  const g = c.getContext('2d'); g.drawImage(img, 0, 0); const d = g.getImageData(0, 0, c.width, c.height).data; let n = 0;
                  for (let i = 0; i < d.length; i += 4) if (d[i] > 235 && Math.abs(d[i + 1] - 63) < 30 && Math.abs(d[i + 2] - 164) < 30) n++; return n; }""", image.element_handle())
                page.locator('.report-model-preview').screenshot(path=str(out / 'capture-front.png'))
            if english:
                page.evaluate("localStorage.setItem('panoptes.language', 'en')")
                page.goto(f'{origin}/{hash_path}&object={objects[-1]}', wait_until='domcontentloaded', timeout=120000)
                page.reload(wait_until='domcontentloaded', timeout=120000)  # a hash-only goto keeps the loaded page and its language
                page.locator(f'{row(objects[-1])}[aria-pressed="true"]').wait_for(state='visible', timeout=300000)
                page.wait_for_timeout(8000)
                record['english'] = {'panel': texts(page, f'[data-layer-box="{objects[-1]}"]'), 'legend': texts(page, '.report-box-legend')}
                checks_en = {'english panel': any('Clearance' in t and 'Width (depth)' in t for t in record['english']['panel']),
                             'english legend': any('low confidence' in t.lower() for t in record['english']['legend'])}
                page.locator('.report-scene-inspector').screenshot(path=str(out / 'inspector-en.png'))
            checks = {}
            if expect_boxes:
                shown = set(record['steps'][-1]['rows'])
                highlighted = {k for k, b in boxes.items() if k in shown and b.get('highlight') is True}
                for i, s in enumerate(record['steps']):
                    for name, ok in step_checks(s, boxes[s['object']]).items():
                        checks[f'{i + 1} {s["object"][:8]} {name}'] = ok
                last, every = record['steps'][-1]['overlay'] or {}, record.get('allBoxes') or {}
                checks.update({
                    'object list highlight = layer flags': all(set(s['recaptureRows']) == highlighted for s in record['steps']) and bool(highlighted),
                    # every highlighted box counts (nested part rows may be collapsed); the malformed one would add 1
                    'legend count = layer flags': all(s['legendCount'] == str(sum(b.get('highlight') is True for b in real.values())) for s in record['steps']),
                    'legend text': all('低置信度（需补拍）' in ''.join(s['legend']) for s in record['steps']),
                    'malformed box skipped': not any(k.startswith('fixture-malformed') for s in record['steps'] for k in s['recaptureRows']) and not record['errors'],
                    'show-all toggle': record['toggle'] == 1,
                    'show-all outlines every box': every.get('outlineLines') == 12 * len(real),
                    'show-all: selected name in its size label, no other names': len(every.get('texts', [])) == len(last.get('texts', [])) and any(' · 长×宽×高' in t for t in every.get('texts', [])),
                    'show-all: one more name on hover': len((record.get('allBoxesHover') or {}).get('texts', [])) == len(every.get('texts', [])) + 1,
                    'capture has no layer boxes': record.get('capturePinkPixels') == 0,
                })
            else:
                checks.update({'no panel or legend': all(not s['panel'] and not s['legend'] for s in record['steps']), 'no toggle': record['toggle'] == 0,
                               'no box labels': all(s['overlay'] is not None and not any('长×宽×高' in t for t in s['overlay']['texts']) for s in record['steps'])})
            if english:
                checks.update(checks_en)
            record['checks'] = checks
            record['failedChecks'] = [name for name, ok in checks.items() if not ok]
            record['passed'] = all(checks.values()) and not record['errors']
        except Exception as error:
            record['failure'] = str(error)[:2000]
            try:
                page.screenshot(path=str(out / 'failure.png'))
            except Exception:
                pass
        finally:
            browser.close()
            server.shutdown()
    (out / 'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as bundle:
        for path in sorted(out.iterdir()):
            bundle.add(path, arcname=path.name)
    return {'passed': bool(record.get('passed')), 'archive': archive.getvalue(), 'containerSeconds': time.monotonic() - started}


@app.local_entrypoint()
def main(site: str, out: str, hash_path: str, objects: str, extra: str = '', expect_boxes: bool = False, english: bool = False):
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    destination.mkdir(parents=True)
    start = time.monotonic()
    result = check.remote(tar_dir(site), tar_dir(extra), hash_path, objects.split(','), expect_boxes, english)
    with tarfile.open(fileobj=io.BytesIO(result.pop('archive')), mode='r:gz') as bundle:
        bundle.extractall(destination, filter='data')
    ledger = {'mode': 'ephemeral modal run', 'hardware': '4 CPU, 8 GiB, no GPU', 'status': 'passed' if result['passed'] else 'failed',
              'functionSeconds': result['containerSeconds'], 'callSeconds': time.monotonic() - start,
              'estimateUsd': RATE * result['containerSeconds'], 'callWindowEstimateUsd': RATE * (time.monotonic() - start),
              'actualBilledUsd': None, 'rateSource': 'https://modal.com/pricing'}
    (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'estimateUsd': ledger['estimateUsd']}))
    if not result['passed']:
        raise SystemExit(1)
