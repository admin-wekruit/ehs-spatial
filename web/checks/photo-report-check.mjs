// Run: node checks/photo-report-check.mjs REPORT_ASSET_DIR [SCREENSHOT_DIR]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { createRequire } from 'node:module';
const { chromium } = createRequire(process.env.PLAYWRIGHT_FROM || '/Users/adam/Desktop/ontab/package.json')('playwright');
const assets = path.resolve(process.argv[2]), out = path.resolve(process.argv[3] || '/tmp/panoptes-photo-check');
const web = path.resolve(new URL('..', import.meta.url).pathname), payload = JSON.parse(fs.readFileSync(path.join(assets, 'scene-report.json')));
const viewerAssets = process.env.VIEWER_ASSETS || '/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages/workcell-photo-direct/viewer-assets';
fs.mkdirSync(out, { recursive: true });
const mime = { '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html', '.json': 'application/json', '.png': 'image/png', '.glb': 'model/gltf-binary' };
const server = http.createServer((req, res) => {
  const name = decodeURIComponent(new URL(req.url, 'http://local').pathname).replace(/^\//, '');
  const file = name.startsWith('viewer-assets/') ? path.join(viewerAssets, name.slice(14)) : path.join(name === '' || name === 'photo.html' || name.startsWith('assets/') ? path.join(web, 'dist-photo') : assets, name || 'photo.html');
  if (!fs.existsSync(file) || !fs.statSync(file).isFile()) { res.writeHead(404).end(); return; }
  res.setHeader('content-type', mime[path.extname(file)] || 'application/octet-stream'); fs.createReadStream(file).pipe(res);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({ args: ['--use-angle=metal', '--ignore-gpu-blocklist', '--enable-gpu'] });
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1200 }, acceptDownloads: true });
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/photo.html`);
  const expectedModels = payload.revision.document.entities.filter(entity => entity.representations.length).length;
  await page.waitForFunction(expected => { const n = document.querySelector('[data-model-loaded]'); return n && +n.dataset.modelLoaded === expected && +n.dataset.modelCoverage === expected; }, expectedModels);
  assert.equal(await page.locator('.report-scene-object-row').count(), payload.objects.length);
  for (const item of payload.objects) {
    await page.locator(`[data-entity-id="${item.id}"] > button`).first().click();
    assert.equal(await page.locator('[data-selected-object]').getAttribute('data-selected-object'), item.id);
    if (!payload.revision.document.entities.find(entity => entity.id === item.id).observedExtentAvailable && item.id !== 'emergency-button') assert.equal(await page.locator('[data-height-native]').textContent(), '未知');
  }
  const slider = page.getByRole('slider', { name: '照片与模型对比位置' });
  let canvasBox;
  for (const value of ['0', '50', '100']) {
    await slider.fill(value);
    assert.equal(await page.locator('.report-comparison-model').evaluate(e => e.style.clipPath), `inset(0px 0px 0px ${value}%)`);
    const box = await page.locator('.report-comparison-model canvas').boundingBox();
    if (canvasBox) assert.deepEqual(box, canvasBox, 'wipe must not change canvas or camera viewport'); canvasBox = box;
  }
  await slider.fill('100');
  const buttonEntity = payload.revision.document.entities.find(entity => entity.id === 'emergency-button');
  const photoTarget = payload.revision.document.observations.find(o => o.imageId === 'photo-4' && buttonEntity.observationRefs.includes(o.id));
  assert.ok(photoTarget, 'button must have photo4 evidence');
  await page.locator('[data-entity-id="robot"] > button').first().click();
  await page.locator('.report-matched-comparison .report-scene-photo-switch button').filter({ hasText: '照片 4' }).click();
  {
    const c = payload.revision.document.cameras.find(c => c.imageId === 'photo-4'), box = await page.locator('.report-comparison-stage > .photo-view svg').boundingBox();
    const scale = Math.min(box.width / c.width, box.height / c.height), b = photoTarget.originalPixelBox;
    await page.mouse.click(box.x + (box.width - c.width * scale) / 2 + (b[0] + b[2]) / 2 * scale, box.y + (box.height - c.height * scale) / 2 + (b[1] + b[3]) / 2 * scale);
    assert.equal(await page.locator('[data-selected-object]').getAttribute('data-selected-object'), 'emergency-button');
  }
  await slider.fill('0');
  await page.locator('[data-entity-id="robot"] > button').first().click();
  const sourceCamera = payload.revision.document.cameras.find(c => c.imageId === 'photo-4'), c2w = sourceCamera.cameraToWorld;
  const delta = buttonEntity.currentModelTransform.position.map((n, i) => n - c2w[i][3]);
  const xyz = [0, 1, 2].map(k => delta.reduce((sum, n, i) => sum + n * c2w[i][k], 0));
  const pixel = sourceCamera.K.map(row => row.reduce((sum, n, i) => sum + n * xyz[i], 0));
  const canvas = await page.locator('.report-comparison-model canvas').boundingBox();
  await page.mouse.click(canvas.x + pixel[0] / pixel[2] / sourceCamera.width * canvas.width, canvas.y + pixel[1] / pixel[2] / sourceCamera.height * canvas.height);
  assert.equal(await page.locator('[data-selected-object]').getAttribute('data-selected-object'), 'emergency-button', 'model canvas picks button from robot selection');
  assert.equal(await page.locator('[data-height-native]').textContent(), '0.200 m');
  async function downloadAt(height) {
    await page.getByLabel('按钮整体高度厘米').fill(String(height));
    const pending = page.waitForEvent('download'); await page.getByRole('button', { name: '下载当前模型 GLB' }).click();
    const download = await pending, target = path.join(out, `${height}.glb`); await download.saveAs(target);
    const buf = fs.readFileSync(target); assert.equal(buf.readUInt32LE(0), 0x46546c67); return JSON.parse(buf.subarray(20, 20 + buf.readUInt32LE(12)).toString());
  }
  const first = await downloadAt(20), second = await downloadAt(40);
  const group = gltf => gltf.nodes.find(node => node.name === 'Workcell — metres, Y up');
  const columnLengths = node => [0, 4, 8].map(i => Math.hypot(...node.matrix.slice(i, i + 3)));
  assert.deepEqual(columnLengths(group(second)), columnLengths(group(first)).map(v => 2 * v), 'exported geometry must scale uniformly');
  function worldBounds(gltf, entityId) {
    const identity = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1], points = [];
    const multiply = (a,b) => Array.from({length:16}, (_,i) => [0,1,2,3].reduce((s,k) => s + a[(i%4)+k*4]*b[k+Math.floor(i/4)*4],0));
    function visit(id, parent, included) {
      const node = gltf.nodes[id]; assert.ok(!node.translation && !node.rotation && !node.scale, 'exporter must retain matrix transforms');
      const matrix = multiply(parent, node.matrix || identity), selected = included || !entityId || node.extras?.entityId === entityId;
      if (selected && node.mesh !== undefined) for (const primitive of gltf.meshes[node.mesh].primitives) {
        const a = gltf.accessors[primitive.attributes.POSITION];
        for(let corner=0;corner<8;corner++){const v=[0,1,2].map(k=>(corner&(1<<k)?a.max:a.min)[k]);points.push([0,1,2].map(k=>matrix[k]*v[0]+matrix[k+4]*v[1]+matrix[k+8]*v[2]+matrix[k+12]));}
      }
      for (const child of node.children || []) visit(child, matrix, selected);
    }
    for (const id of gltf.scenes[gltf.scene || 0].nodes) visit(id, identity, false);
    assert.ok(points.length); return [0,1,2].map(k => [Math.min(...points.map(p=>p[k])), Math.max(...points.map(p=>p[k]))]);
  }
  const firstBounds = worldBounds(first), secondBounds = worldBounds(second), floorBounds = worldBounds(first, 'floor');
  for(let k=0;k<3;k++) for(let j=0;j<2;j++) assert.ok(Math.abs(secondBounds[k][j] - 2*firstBounds[k][j]) < 1e-6, 'world geometry bounds double');
  assert.ok(Math.max(...floorBounds[1].map(Math.abs)) < 1e-5, 'exported floor is horizontal at Y=0');
  assert.equal(await page.locator('[data-height-native]').textContent(), '0.400 m');
  assert.equal(second.nodes.filter(node => node.extras?.entityId).length, payload.revision.document.entities.filter(entity => entity.representations.length).length, 'export all active geometry objects');
  await page.getByLabel('按钮整体高度厘米').fill('20'); await slider.fill('50');
  await page.locator('.report-matched-comparison .report-scene-photo-switch button').filter({ hasText: '照片 1' }).click();
  assert.equal(await page.locator('.report-comparison-stage').getAttribute('data-camera-id'), 'camera-1');
  await page.locator('.report-matched-comparison .report-scene-photo-switch button').filter({ hasText: '照片 4' }).click();
  await page.screenshot({ path: path.join(out, 'desktop.png'), fullPage: true });
  await page.getByRole('button', { name: '场景 3D', exact: true }).first().click();
  await page.waitForSelector('[data-pane="spatial"] canvas');
  await page.waitForFunction(expected => +document.querySelector('[data-model-loaded]').dataset.modelLoaded === expected, expectedModels);
  await page.screenshot({ path: path.join(out, 'free-3d.png'), fullPage: true });
  await page.getByLabel('标尺轴').selectOption('width');
  const widthScale = Number(await page.locator('[data-native-to-meters]').getAttribute('data-native-to-meters'));
  assert.ok(Math.abs(widthScale - .2 / payload.geometry.anchor.nativeWidth) < 1e-10);
  await page.getByLabel('按钮整体高度厘米').fill('80');
  assert.equal(await page.locator('.photo-report-warning').count(), 1);
  await page.getByLabel('按钮整体高度厘米').fill('0');
  assert.equal(await page.getByRole('button', { name: '下载当前模型 GLB' }).isDisabled(), true);
  await page.getByLabel('按钮整体高度厘米').fill('20'); await page.getByLabel('标尺轴').selectOption('height');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('.report-scene-section-tabs button').filter({ hasText: '场景对象' }).click();
  await page.locator('[data-entity-id="robot"] > button').first().click();
  await page.locator('.report-scene-section-tabs button').filter({ hasText: '对象详情' }).click();
  assert.equal(await page.locator('[data-selected-object]').isVisible(), true);
  await page.screenshot({ path: path.join(out, 'mobile.png'), fullPage: true });
  assert.deepEqual(errors, [], 'no browser runtime errors');
  console.log(JSON.stringify({ pass: true, objects: payload.objects.length, checks: ['list and inspector', 'single-view dimensions unknown', 'slider 0/50/100 stable viewport', 'photo and model selection', 'photo camera switch', '20/40 cm geometry export', 'free 3D scene', 'width scale and mismatch', 'mobile list and inspector'], screenshot: path.join(out, 'desktop.png') }));
} finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
