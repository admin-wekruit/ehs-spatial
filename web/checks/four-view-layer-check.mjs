// Run: PLAYWRIGHT_FROM=/qa/package.json node checks/four-view-layer-check.mjs URL OBJECT_ID [OUT_DIR] [EXPECTED_SIZE_TEXT] [zh|en]
// A live four-view report with a measurement layer: open one object, record its model size readout and the layer's
// photo-measured facts, and require the expected size text when one is given (e.g. the reference e-stop "0.080 × 0.080 × 0.100").
// With en the page opens in English (the viewer's saved language), as after picking English at the top right.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const { chromium } = createRequire(process.env.PLAYWRIGHT_FROM || import.meta.url)('playwright');
const [url, objectId, outArg, expected, language = 'zh'] = process.argv.slice(2);
const out = path.resolve(outArg || '/tmp/four-view-layer-check');
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ args: ['--ignore-gpu-blocklist', '--enable-gpu', '--enable-unsafe-swiftshader'] });
const record = { url, objectId, expected: expected || null, language };
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1200 } });
  await page.route(/fonts\.(googleapis|gstatic)\.com/, route => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const errors = [], layer = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('response', response => { if (response.url().includes('/measurement-layer/')) layer.push([response.url(), response.status()]); });
  if (language === 'en') await page.addInitScript(() => localStorage.setItem('panoptes.language', 'en'));
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120000 });
  const facts = page.locator(`[data-measurement-facts="${objectId}"]`);
  try {
    await facts.waitFor({ state: 'visible', timeout: 240000 });
  } catch (error) {  // keep what the page did show
    record.finalUrl = page.url(); record.errors = errors; record.layerRequests = layer;
    record.details = (await page.locator('.report-selection-details').allInnerTexts()).join('\n').slice(0, 4000);
    await page.screenshot({ path: path.join(out, 'timeout.png') });
    throw error;
  }
  record.finalUrl = page.url();
  record.layerRequests = layer;
  record.facts = await facts.innerText();
  const details = page.locator('.report-selection-details').first();
  record.details = (await details.innerText()).slice(0, 4000);
  if (expected) assert.ok(record.details.includes(expected), `size readout should include ${expected}`);
  await page.waitForTimeout(10000); // the selected model draws after its asset arrives
  await page.screenshot({ path: path.join(out, 'four-view.png') });
  await details.screenshot({ path: path.join(out, 'details.png') });
  record.bends = await page.locator('.report-bend-analysis').allInnerTexts();
  // The selected object's own model pane, in its top and front orthographic views.
  const preview = page.locator('.report-model-preview').first();
  for (const [label, file] of language === 'en' ? [['Top', 'preview-top.png'], ['Front', 'preview-front.png']] : [['俯视', 'preview-top.png'], ['正视', 'preview-front.png']]) {
    const button = preview.getByRole('button', { name: label, exact: true });
    if (await button.count()) {
      await button.click();
      await preview.getByText('正在载入空间资产').waitFor({ state: 'hidden', timeout: 120000 }).catch(() => {});
      await page.waitForTimeout(3000); await preview.screenshot({ path: path.join(out, file) });
    }
  }
  record.layerRequests = layer;  // again: the layer's meshes arrive after the facts
  record.errors = errors;
  record.passed = true;
} finally {
  fs.writeFileSync(path.join(out, 'record.json'), JSON.stringify(record, null, 2));
  await browser.close();
}
console.log(JSON.stringify({ passed: record.passed, finalUrl: record.finalUrl, facts: record.facts, errors: record.errors }));
