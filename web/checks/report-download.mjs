import assert from 'node:assert/strict';
import { downloadAsset } from '../src/api.ts';

// A signed asset URL must never become a report-navigation target.
const requests = [], anchors = [];
globalThis.location = { origin: 'https://panoptes.example' };
globalThis.document = { createElement() {
  const anchor = { click() { assert.match(this.href, /^blob:/); anchors.push(this); } };
  return anchor;
} };
let contentStatus = 200, bytes = 'blend';
globalThis.fetch = async (url, options) => {
  requests.push({ url, options });
  if (url.endsWith('/content')) return new Response(bytes, { status: contentStatus });
  return new Response(JSON.stringify({ id: 'asset', mediaType: 'application/x-blender', sizeBytes: 5,
    metadata: {}, url: 'https://storage.example/asset?signature=ephemeral' }), {status:200});
};
await downloadAsset('asset');
assert.equal(anchors.length, 1);
assert.equal(anchors[0].download, 'asset.blend');
assert.deepEqual(requests.map(r=>r.url), ['/api/assets/asset', '/api/assets/asset/content']);
assert.equal(requests[1].options.credentials, 'omit');
assert.equal(requests[1].options.redirect, 'error');
contentStatus = 503;
await assert.rejects(downloadAsset('asset'), /asset_download_failed/);
contentStatus = 200; bytes = 'cut';
await assert.rejects(downloadAsset('asset'), /asset_size_mismatch/);
assert.equal(anchors.length, 1, 'failed or truncated download does not launch a file');
console.log('report download: local blob target, correct extension, no credentials/navigation, failure handling passed');
