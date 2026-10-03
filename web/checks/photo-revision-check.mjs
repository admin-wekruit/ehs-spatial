// Run: PLAYWRIGHT_FROM=/qa/package.json node checks/photo-revision-check.mjs PACKAGED_PAGE_DIR [OUT_DIR]
// Real browser over a packaged page with revisions/: semantic query -> object -> this revision's
// measurement, scale trial, revision switch, caliper JSON and downloaded GLB readback.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { createRequire } from 'node:module';
const { chromium } = createRequire(process.env.PLAYWRIGHT_FROM || import.meta.url)('playwright');
const root = path.resolve(process.argv[2]), out = path.resolve(process.argv[3] || '/tmp/panoptes-revision-check');
fs.mkdirSync(out, { recursive: true });
const read = file => JSON.parse(fs.readFileSync(path.join(root, file)));
const main = read('scene-report.json');
const candidateChoice = main.revisionChoices.find(row => row.status !== 'main');
const candidate = read(candidateChoice.url);
const mime = { '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html', '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.glb': 'model/gltf-binary' };
const glbRequests = [];
const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://local'), name = decodeURIComponent(url.pathname).replace(/^\//, '') || 'index.html';
  if (name.endsWith('.glb')) glbRequests.push(url);
  const file = path.resolve(root, name);
  if (!file.startsWith(root) || !fs.existsSync(file) || !fs.statSync(file).isFile()) { res.writeHead(404).end(); return; }
  res.setHeader('content-type', mime[path.extname(file)] || 'application/octet-stream'); fs.createReadStream(file).pipe(res);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${server.address().port}/`;
const cm = (report, native, factor = 1) => `${(native * report.modelMeasurementScale.nativeToMeters * factor * 100).toFixed(2)} cm`;
const browser = await chromium.launch({ args: ['--ignore-gpu-blocklist', '--enable-gpu', '--enable-unsafe-swiftshader'] });
const record = { revisions: {}, checks: [] };
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1200 }, acceptDownloads: true });
  // Web fonts are presentation only; this sandbox cannot reach the font CDN.
  await page.route(/fonts\.(googleapis|gstatic)\.com/, route => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const errors = [], api = [];
  page.on('pageerror', error => errors.push(error.message)); page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) api.push(request.url()); });
  const modelsLoaded = async report => {
    const expected = report.revision.document.entities.filter(entity => !entity.sourceContext && entity.activeModelRepresentationId).length;
    await page.waitForFunction(count => { const node = document.querySelector('[data-model-loaded]'); return node && +node.dataset.modelLoaded === count && +node.dataset.modelCoverage === count; }, expected, { timeout: 600000 });
  };
  async function endpointCard(report) {
    for (const row of report.endpointEstimation.endpoints)
      assert.equal(await page.locator(`.photo-report-endpoints [data-endpoint-estimate="${row.id}"]`).textContent(), cm(report, row.heightNative), row.id);
    for (const row of report.endpointEstimation.differences)
      assert.equal(await page.locator(`.photo-report-endpoints [data-endpoint-difference="${row.id}"]`).first().textContent(), cm(report, row.valueNative), row.id);
  }
  function glbJSON(file) { const buffer = fs.readFileSync(file); assert.equal(buffer.readUInt32LE(0), 0x46546c67); return { buffer, gltf: JSON.parse(buffer.subarray(20, 20 + buffer.readUInt32LE(12)).toString()) }; }
  function vertices({ buffer, gltf }, entityId) {
    const binOffset = 20 + buffer.readUInt32LE(12) + 8, multiply = (a, b) => Array.from({ length: 16 }, (_, i) => [0, 1, 2, 3].reduce((s, k) => s + a[(i % 4) + k * 4] * b[k + Math.floor(i / 4) * 4], 0));
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], points = [];
    function visit(id, parent, inside) {
      const node = gltf.nodes[id], matrix = multiply(parent, node.matrix || identity), take = inside || node.extras?.entityId === entityId;
      if (take && node.mesh !== undefined) for (const primitive of gltf.meshes[node.mesh].primitives) {
        const accessor = gltf.accessors[primitive.attributes.POSITION], view = gltf.bufferViews[accessor.bufferView];
        const data = new Float32Array(buffer.buffer.slice(buffer.byteOffset + binOffset + (view.byteOffset || 0) + (accessor.byteOffset || 0), buffer.byteOffset + binOffset + (view.byteOffset || 0) + (accessor.byteOffset || 0) + accessor.count * 12));
        for (let i = 0; i < accessor.count; i++) { const v = [data[3 * i], data[3 * i + 1], data[3 * i + 2]]; points.push([0, 1, 2].map(k => matrix[k] * v[0] + matrix[k + 4] * v[1] + matrix[k + 8] * v[2] + matrix[k + 12])); }
      }
      for (const child of node.children || []) visit(child, matrix, take);
    }
    for (const id of gltf.scenes[gltf.scene || 0].nodes) visit(id, identity, false);
    return points;
  }
  async function downloadGLB(report, label) {
    const pending = page.waitForEvent('download');
    await page.getByRole('button', { name: /下载.*模型 GLB/ }).click();
    const download = await pending, file = path.join(out, `${label}.glb`); await download.saveAs(file);
    const glb = glbJSON(file), extras = glb.gltf.scenes[glb.gltf.scene || 0].extras;
    assert.equal(extras.revisionId, report.revision.id); assert.equal(extras.documentSha256, report.revision.documentSha256);
    // Readback: the curtain's measured lower point is a vertex of the exported mesh at the reported height (Y-up metres).
    const scale = report.modelMeasurementScale.nativeToMeters, rows = [];
    for (const row of report.endpointEstimation.endpoints.filter(row => row.measurementScope !== 'model_lower_rail_near_curtain')) {
      const heights = vertices(glb, row.objectId).map(p => p[1]), expected = row.heightNative * scale;
      const nearest = heights.reduce((best, h) => Math.abs(h - expected) < Math.abs(best - expected) ? h : best, Infinity);
      const tolerance = row.measurementScope === 'model_bottom_face_center' ? 1e-6 + (row.bottomFaceHeightRangeNative[1] - row.bottomFaceHeightRangeNative[0]) * scale : 1e-6;
      assert.ok(Math.abs(nearest - expected) <= tolerance, `${row.id}: exported vertex ${nearest} vs ${expected}`);
      rows.push({ endpoint: row.id, exportedVertexHeightM: nearest, reportedHeightM: expected });
    }
    return { file: path.basename(file), extras: { revisionId: extras.revisionId, documentSha256: extras.documentSha256 }, readback: rows };
  }
  // 1. Main revision.
  await page.goto(base + 'index.html');
  await modelsLoaded(main);
  assert.equal(await page.locator('main.photo-report').getAttribute('data-revision-id'), main.revision.id);
  await endpointCard(main);
  await everyObject(main);
  assert.ok(glbRequests.length && glbRequests.every(url => url.searchParams.get('revision') === main.revision.documentSha256));
  // 2. Semantic query -> light curtain -> its measurement in this revision.
  const query = main.semanticExperiment.queries.find(row => /光幕/.test(row.label));
  await page.locator('.photo-semantic-queries button').filter({ hasText: query.label }).click();
  const results = page.locator('.photo-semantic-query-results li button');
  const measuredObjects = new Set(main.endpointEstimation.endpoints.map(row => row.objectId));
  let target = null;
  for (let i = 0; i < await results.count(); i++) { const id = (await results.nth(i).locator('small').first().textContent()).trim(); if (measuredObjects.has(id)) { target = id; await results.nth(i).click(); break; } }
  assert.ok(target, 'light-curtain query returns a measured object');
  await page.waitForFunction(id => document.querySelector('[data-selected-object]')?.getAttribute('data-selected-object') === id, target);
  const endpoint = main.endpointEstimation.endpoints.find(row => row.objectId === target);
  assert.equal(await page.locator(`[data-semantic-fact="${endpoint.id}"]`).textContent(), cm(main, endpoint.heightNative), 'semantic fact reads this revision');
  const pair = main.endpointEstimation.differences.find(row => row.minuendId === endpoint.id);
  const label = `${pair.label}：${cm(main, pair.valueNative)} · 条件模型估计，未验证`;
  await page.locator('[data-pane="spatial"] .native-stage svg text').filter({ hasText: label }).waitFor({ timeout: 120000 });
  record.checks.push({ step: 'semantic query -> measured object', query: query.label, object: target, fact: cm(main, endpoint.heightNative), annotation: label });
  await page.screenshot({ path: path.join(out, 'main-semantic-measurement.png'), fullPage: false });
  // 3. Scale trial changes every derived value together.
  await page.getByLabel('按钮整体高度厘米').fill('20');
  assert.equal(await page.locator(`[data-semantic-fact="${endpoint.id}"]`).textContent(), cm(main, endpoint.heightNative, 2));
  assert.equal(await page.locator(`.photo-report-endpoints [data-endpoint-estimate="${endpoint.id}"]`).textContent(), cm(main, endpoint.heightNative, 2));
  await page.getByLabel('按钮整体高度厘米').fill('10');
  assert.equal(await page.locator(`[data-semantic-fact="${endpoint.id}"]`).textContent(), cm(main, endpoint.heightNative));
  async function everyObject(report) {
    // Every catalog object stays selectable with its own identity; measurements never shrink the scene.
    for (const item of report.objects) {
      await page.locator(`[data-entity-id="${item.id}"] > button`).first().click();
      assert.equal(await page.locator('[data-selected-object]').getAttribute('data-selected-object'), item.id);
    }
    assert.equal(await page.locator('[data-policy-revision]').getAttribute('data-policy-revision'), report.revision.id, 'EHS evidence belongs to this revision');
    record.checks.push({ step: 'every object selectable', revision: report.revision.id, objects: report.objects.length });
  }
  async function ask(report) {
    const left = report.endpointEstimation.differences.find(row => row.id === 'curtain-left-minus-right');
    await page.locator('[data-spatial-question]').fill('左右光幕谁更高？'); await page.getByRole('button', { name: '回答', exact: true }).click();
    const text = await page.locator('[data-spatial-answer]').textContent();
    assert.ok(text.includes(`左 − 右 = ${cm(report, left.valueNative)}`) && text.includes(left.valueNative > 0 ? '左侧光幕测点更高' : '右侧光幕测点更高'), text);
    record.checks.push({ step: 'structured spatial question', revision: report.revision.id, question: '左右光幕谁更高？', answer: text });
  }
  await ask(main);
  record.revisions.main = { id: main.revision.id, documentSha256: main.revision.documentSha256, glb: await downloadGLB(main, 'main') };
  // 4. Switch to the candidate revision; the same object now reads the candidate model.
  glbRequests.length = 0;
  await page.locator('[data-revision-select]').selectOption(candidate.revision.id);
  await page.waitForFunction(id => document.querySelector('main.photo-report')?.getAttribute('data-revision-id') === id, candidate.revision.id);
  await modelsLoaded(candidate);
  assert.equal(new URL(page.url()).searchParams.get('version'), candidate.revision.id);
  await endpointCard(candidate);
  assert.ok(glbRequests.every(url => url.searchParams.get('revision') === candidate.revision.documentSha256), 'candidate assets carry its document hash');
  assert.ok(glbRequests.some(url => url.pathname.includes(`/revisions/${candidate.revision.id}/entity-post-box-`)), 'candidate curtain models load from the candidate revision');
  const candidateEndpoint = candidate.endpointEstimation.endpoints.find(row => row.objectId === target);
  assert.equal(await page.locator('[data-selected-object]').getAttribute('data-selected-object'), target);
  assert.equal(await page.locator(`[data-semantic-fact="${candidateEndpoint.id}"]`).textContent(), cm(candidate, candidateEndpoint.heightNative));
  assert.notEqual(cm(candidate, candidateEndpoint.heightNative), cm(main, endpoint.heightNative), 'candidate and main measure different models');
  record.checks.push({ step: 'revision switch keeps object, reads candidate model', object: target, main: cm(main, endpoint.heightNative), candidate: cm(candidate, candidateEndpoint.heightNative) });
  await page.screenshot({ path: path.join(out, 'candidate-semantic-measurement.png'), fullPage: false });
  // Only after the switch kept the selection: every candidate object stays selectable.
  await everyObject(candidate);
  await ask(candidate);
  record.revisions.candidate = { id: candidate.revision.id, documentSha256: candidate.revision.documentSha256, glb: await downloadGLB(candidate, 'candidate') };
  // 5. Generic caliper on the candidate: one surface point, JSON bound to this document. The caliper needs a selected
  // object with a model; photo-only objects (outside the modelled workcell) keep it disabled.
  const modelled = candidate.revision.document.entities.find(entity => !entity.sourceContext && entity.activeModelRepresentationId);
  await page.locator(`[data-entity-id="${modelled.id}"] > button`).first().click();
  const panel = page.locator('.report-scene-inspector .spatial-measurements').first();
  if (process.env.DEBUG_CALIPER) { fs.writeFileSync(path.join(out, 'caliper-panel.html'), await panel.innerHTML()); await page.screenshot({ path: path.join(out, 'caliper-panel.png'), fullPage: true }); }
  await panel.locator('select').first().selectOption('point_ground');
  const canvas = page.locator('[data-pane="spatial"] canvas').first();
  let picked = false;
  for (let attempt = 0; attempt < 60 && !picked; attempt++) {
    // A miss keeps picking active; only restart once a pick has completed or been cancelled.
    const start = panel.getByRole('button', { name: /在 3D 中选择 1 个点|重新取点/ });
    if (await start.count()) await start.click();
    const box = await canvas.boundingBox(), x = box.x + box.width * (0.3 + 0.4 * ((attempt * 7) % 10) / 9), y = box.y + box.height * (0.3 + 0.4 * ((attempt * 3) % 10) / 9);
    await page.mouse.click(x, y);
    picked = await panel.getByRole('button', { name: '计算并标注' }).isEnabled();
  }
  assert.ok(picked, 'a model surface point was picked');
  await panel.getByRole('button', { name: '计算并标注' }).click();
  const pending = page.waitForEvent('download'); await panel.getByRole('button', { name: '下载测量 JSON' }).click();
  const measurementFile = path.join(out, 'candidate-caliper.json'); await (await pending).saveAs(measurementFile);
  const measurement = JSON.parse(fs.readFileSync(measurementFile));
  assert.equal(measurement.revisionId, candidate.revision.id); assert.equal(measurement.documentSha256, candidate.revision.documentSha256);
  const hashes = new Set(candidate.revision.document.assets.map(asset => asset.sha256));
  assert.ok(measurement.references.length && measurement.references.every(ref => hashes.has(ref.assetSha256)));
  const [point] = measurement.caliper.pointsNative, foot = measurement.caliper.feetNative[0];
  assert.ok(Math.abs(foot[2]) < 1e-9 && Math.abs(point[2] - measurement.caliper.heightsNative[0]) < 1e-9, 'Z-up report floor');
  record.checks.push({ step: 'caliper JSON bound to candidate document', entity: measurement.references[0].entityId, heightNative: measurement.caliper.heightsNative[0], label: measurement.displayLabel });
  assert.deepEqual(errors, []); assert.deepEqual(api, []);
  record.consoleErrors = errors.length; record.apiRequests = api.length;
  fs.writeFileSync(path.join(out, 'browser-revision-check.json'), JSON.stringify(record, null, 2));
  console.log('PASS: semantic query -> measured object -> live fact/annotation; scale x2; revision switch reloads models with candidate hash; caliper JSON and GLB vertex readback bound to each document; no console errors or API calls');
} finally { await browser.close(); server.close(); }
