// Run: PLAYWRIGHT_FROM=/qa/package.json node checks/four-view-layer-check.mjs URL OBJECT_ID [OUT_DIR] [EXPECTED_SIZE_TEXT]
// A live four-view report with a measurement layer: open one object, record its model size readout and the layer's
// photo-measured facts, and require the expected size text when one is given (e.g. the reference e-stop "0.080 × 0.080 × 0.100").
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const { chromium } = createRequire(process.env.PLAYWRIGHT_FROM || import.meta.url)('playwright');
const [url, objectId, outArg, expected] = process.argv.slice(2);
const out = path.resolve(outArg || '/tmp/four-view-layer-check');
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ args: ['--ignore-gpu-blocklist', '--enable-gpu', '--enable-unsafe-swiftshader'] });
const record = { url, objectId, expected: expected || null };
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1200 } });
  await page.route(/fonts\.(googleapis|gstatic)\.com/, route => route.fulfill({ status: 200, contentType: 'text/css', body: '' }));
  const errors = [], layer = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('response', response => { if (response.url().includes('/measurement-layer/')) layer.push([response.url(), response.status()]); });
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120000 });
  const facts = page.locator(`[data-measurement-facts="${objectId}"]`);
  await facts.waitFor({ state: 'visible', timeout: 240000 });
  record.finalUrl = page.url();
  record.layerRequests = layer;
  record.facts = await facts.innerText();
  const details = page.locator('.report-selection-details').first();
  record.details = (await details.innerText()).slice(0, 4000);
  if (expected) assert.ok(record.details.includes(expected), `size readout should include ${expected}`);
  await page.waitForTimeout(10000); // the selected model draws after its asset arrives
  await page.screenshot({ path: path.join(out, 'four-view.png') });
  await details.screenshot({ path: path.join(out, 'details.png') });
  record.errors = errors;
  record.passed = true;
} finally {
  fs.writeFileSync(path.join(out, 'record.json'), JSON.stringify(record, null, 2));
  await browser.close();
}
console.log(JSON.stringify({ passed: record.passed, finalUrl: record.finalUrl, facts: record.facts, errors: record.errors }));
