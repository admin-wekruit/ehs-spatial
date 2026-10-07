// Run: PLAYWRIGHT_FROM=/check/package.json node checks/scene-review-check.mjs REPORT_URL OUT_DIR OBJECTS_JSON [CAP_SECONDS]
// A live four-view report, fully loaded: wait until every model reports loaded and the scene's asset status is gone
// (or the cap, then report what is still loading), record every request (size, time, failures) and console error,
// then capture the four-view, the 场景 3D pane after 适应画布 (with / without the point cloud overlay, orbited models only, point
// cloud alone) and each named object's model preview (free, front, side, top). Screenshots only, no assertions on looks.
// OBJECTS_JSON = [["robot", "<entity id>"], ...]. Review only: it changes nothing on the page's data.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const { chromium } = createRequire(process.env.PLAYWRIGHT_FROM || import.meta.url)('playwright');
const [url, outArg, objectsArg, capArg] = process.argv.slice(2);
const out = path.resolve(outArg), objects = JSON.parse(objectsArg || '[]'), cap = 1000 * Number(capArg || 180);
fs.mkdirSync(out, { recursive: true });
const record = { url, capSeconds: cap / 1000, timeline: [], requests: [], failedRequests: [], errors: [], shots: [], objects: [] };
const browser = await chromium.launch({ args: ['--ignore-gpu-blocklist', '--enable-gpu', '--enable-unsafe-swiftshader'] });
const t0 = Date.now(), secs = () => +((Date.now() - t0) / 1000).toFixed(1);
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1200 } });
  await page.route(/fonts\.(googleapis|gstatic)\.com/, route => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const started = new Map();
  page.on('pageerror', error => record.errors.push({ t: secs(), kind: 'pageerror', text: error.message }));
  page.on('console', message => { if (['error', 'warning'].includes(message.type())) record.errors.push({ t: secs(), kind: message.type(), text: message.text().slice(0, 500) }); });
  page.on('request', request => started.set(request, secs()));
  page.on('requestfailed', request => record.failedRequests.push({ url: request.url(), t0: started.get(request), t: secs(), failure: request.failure()?.errorText }));
  page.on('requestfinished', async request => {
    const response = await request.response().catch(() => null), sizes = await request.sizes().catch(() => null);
    const headers = response ? await response.allHeaders().catch(() => ({})) : {};
    record.requests.push({ url: request.url().slice(0, 300), type: request.resourceType(), status: response?.status() ?? null, t0: started.get(request), t1: secs(),
      bodyBytes: sizes?.responseBodySize ?? null, contentLength: headers['content-length'] ? +headers['content-length'] : null,
      contentType: headers['content-type'] || null, contentEncoding: headers['content-encoding'] || null, cache: headers['cache-control'] || null });
  });
  const state = () => page.evaluate(() => {
    const notice = document.querySelector('[data-model-loaded]'), stage = [...document.querySelectorAll('[data-pane="spatial"] .stage-status')].map(node => node.innerText);
    const rows = [...document.querySelectorAll('.report-scene-object-row')].map(row => ({ id: row.dataset.entityId, label: row.querySelector('strong')?.innerText,
      model: row.querySelector('[data-model-state]')?.dataset.modelState }));
    return { loaded: notice ? +notice.dataset.modelLoaded : null, coverage: notice ? +notice.dataset.modelCoverage : null, notice: notice?.innerText || null, stage, rows };
  });
  // Spatial-pane assets (models, point cloud, floor) must settle again after a layer switch.
  const settle = async (ms = 60000) => { await page.locator('[data-pane="spatial"] .stage-status').first().waitFor({ state: 'hidden', timeout: ms }).catch(() => {}); await page.waitForTimeout(2500); };
  const shot = async (locator, name) => { await locator.screenshot({ path: path.join(out, name) }); record.shots.push(name); };
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120000 });
  let now = null;
  for (;;) {
    now = await state();
    const { rows, ...brief } = now;
    record.timeline.push({ t: secs(), ...brief, notice: undefined });
    if (now.coverage > 0 && now.loaded >= now.coverage && !now.stage.length) break;
    if (Date.now() - t0 > cap) break;
    await page.waitForTimeout(2000);
  }
  record.loadedAfterSeconds = now.coverage > 0 && now.loaded >= now.coverage && !now.stage.length ? secs() : null;
  record.notice = now.notice; record.stageStatus = now.stage;
  record.stillLoading = now.rows.filter(row => /Loading|Failed|Missing/.test(row.model || ''));
  record.rows = now.rows;
  await page.waitForTimeout(3000);
  const center = page.locator('.report-scene-center').first(), spatial = page.locator('[data-pane="spatial"]').first();
  await shot(center, 'four-view.png');
  const views = page.locator('.report-scene-view-switch:not(.report-scene-mobile-views)');
  await views.getByRole('button', { name: '场景 3D', exact: true }).click();
  const fit = async () => { await spatial.locator('.spatial-fit').click(); await page.waitForTimeout(3000); };
  await settle(); await fit(); await shot(spatial, 'scene3d-fit.png');
  const overlay = page.getByLabel('模型叠加点云');
  if (await overlay.count()) { await overlay.uncheck(); await settle(); await fit(); await shot(spatial, 'scene3d-fit-models-only.png'); }
  const canvas = spatial.locator('canvas').first(), box = await canvas.boundingBox();
  // Each orbit starts from the fitted view (drags do not accumulate).
  for (const [name, dx, dy] of [['scene3d-orbit-right.png', 120, 0], ['scene3d-orbit-left.png', -120, 0], ['scene3d-orbit-low.png', 0, -90], ['scene3d-orbit-top.png', 0, 200]]) {
    await fit();
    const x = box.x + box.width / 2, y = box.y + box.height / 2;
    await page.mouse.move(x, y); await page.mouse.down(); await page.mouse.move(x + dx, y + dy, { steps: 20 }); await page.mouse.up();
    await page.waitForTimeout(2500); await shot(spatial, name);
  }
  if (await overlay.count()) { await overlay.check(); await settle(); }  // orbits above are models only
  const layerSelect = page.getByLabel('空间表示');
  if (await layerSelect.locator('option[value="point_cloud"]:not([disabled])').count()) {
    await layerSelect.selectOption('point_cloud'); await settle(); await fit(); await shot(spatial, 'scene3d-point-cloud.png');
    await layerSelect.selectOption('model'); await settle();
  }
  await fit();
  // Object previews in the single 所选对象模型 pane (larger than in the four-view).
  await views.getByRole('button', { name: '所选对象模型', exact: true }).click();
  const list = page.locator('.report-scene-object-list');
  for (let i = 0; i < 5 && await list.locator('button[aria-expanded="false"]').count(); i++) await list.locator('button[aria-expanded="false"]').first().click();
  const preview = page.locator('.report-model-preview').first();
  for (const [name, id] of objects) {
    const item = { name, id };
    record.objects.push(item);
    const row = page.locator(`[data-entity-id="${id}"] > button`).first();
    if (!await row.count()) { item.missing = 'no object row'; continue; }
    await row.scrollIntoViewIfNeeded(); await row.click();
    await page.locator(`.report-model-preview[data-model-entity="${id}"]`).waitFor({ timeout: 30000 }).catch(() => { item.missing = 'preview did not switch'; });
    item.details = (await page.locator('.report-selection-details').first().innerText().catch(() => '')).slice(0, 2500);
    item.previewFooter = (await preview.locator('footer').first().innerText().catch(() => '')).slice(0, 600);
    for (const [label, mode] of [['自由 3D', 'free'], ['正视', 'front'], ['侧视', 'side'], ['俯视', 'top']]) {
      const button = preview.getByRole('button', { name: label, exact: true });
      if (!await button.count()) { item[mode] = 'no button'; continue; }
      await button.click();
      if (mode === 'free') await preview.locator('.stage-status').first().waitFor({ state: 'hidden', timeout: 60000 }).catch(() => { item.freeStillLoading = true; });
      else await preview.locator('.report-model-image img').first().waitFor({ timeout: 30000 }).catch(() => { item[mode] = 'no capture'; });
      await page.waitForTimeout(mode === 'free' ? 3000 : 1000);
      await shot(preview, `obj-${name}-${mode}.png`);
    }
  }
  record.passed = true;
} catch (error) {
  record.failure = String(error?.stack || error).slice(0, 2000);
} finally {
  const assets = record.requests.filter(row => !/\.(js|css|woff2?)(\?|$)/.test(row.url) && row.type !== 'document');
  const byHost = {};
  for (const row of record.requests) {
    const host = new URL(row.url).host, entry = byHost[host] ||= { count: 0, bodyBytes: 0, slowestSeconds: 0, lastFinish: 0, statuses: {} };
    entry.count++; entry.bodyBytes += row.bodyBytes || 0; entry.slowestSeconds = Math.max(entry.slowestSeconds, +(row.t1 - row.t0).toFixed(1));
    entry.lastFinish = Math.max(entry.lastFinish, row.t1); entry.statuses[row.status] = (entry.statuses[row.status] || 0) + 1;
  }
  record.summary = { byHost, assetRequests: assets.length, assetBytes: assets.reduce((sum, row) => sum + (row.bodyBytes || 0), 0),
    largest: [...assets].sort((a, b) => (b.bodyBytes || 0) - (a.bodyBytes || 0)).slice(0, 12).map(row => [row.url, row.bodyBytes, +(row.t1 - row.t0).toFixed(1)]),
    slowest: [...assets].sort((a, b) => (b.t1 - b.t0) - (a.t1 - a.t0)).slice(0, 12).map(row => [row.url, row.bodyBytes, +(row.t1 - row.t0).toFixed(1)]),
    httpErrors: record.requests.filter(row => row.status >= 400).map(row => [row.url, row.status]) };
  fs.writeFileSync(path.join(out, 'record.json'), JSON.stringify(record, null, 2));
  await browser.close();
}
console.log(JSON.stringify({ passed: !!record.passed, loadedAfterSeconds: record.loadedAfterSeconds, notice: record.notice, stillLoading: record.stillLoading?.length,
  failedRequests: record.failedRequests.length, errors: record.errors.length, shots: record.shots.length, failure: record.failure }));
