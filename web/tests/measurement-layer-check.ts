// Run: node --experimental-strip-types tests/measurement-layer-check.ts
import assert from 'node:assert/strict';
import { renderMessage, validMessage, validBox, withLocalizedLabels, loadMeasurementLayer, applyMeasurementLayer, type DisplayMessage, type LayerBox, type MeasurementLayer } from '../src/measurement-layer.ts';
import { LANGUAGES, translate } from '../src/translate.ts';
import catalog from '../src/locales/en.json' with { type: 'json' };
import type { Revision } from '../src/types.ts';

const joined = { code: 'message.join', params: { first: { code: 'box.face.top' }, second: { code: 'box.level.high' } } };
for (const language of LANGUAGES) {
  assert.equal(renderMessage(language, joined), translate(language, 'message.join', { first: translate(language, 'box.face.top'), second: translate(language, 'box.level.high') }));
  assert.doesNotMatch(renderMessage(language, joined), /\{\w+\}/);
}
for (const value of ['legacy text', null, { code: '' }, { code: 'message.join', params: { first: [] } }, { code: 'message.join', params: { first: Infinity } }]) {
  assert.equal(validMessage(value), false);
  assert.throws(() => renderMessage('en', value as any), /invalid_display_message/);
}

const x = [1, 0, 0], y = [0, 1, 0], z = [0, 0, 1], dim = { valueM: .1, sigmaCm: .5, confidence: 'high' };
const box = { label: { code: 'box.face.front' }, centerNative: [0, 0, 0], axes: [x, y, z], sizeM: [.1, .1, .1], bottomM: 0, topM: .1, floorContact: true,
  faceNormals: { front: [0, -1, 0], back: y, left: [-1, 0, 0], right: x, top: z, bottom: [0, 0, -1] },
  dims: { L: dim, W: dim, H: dim, bottom: dim }, highlight: true, confidence: 'medium',
  highlightReasons: [joined], snapNote: { code: 'viewer.ground' },
  faces: { top: { photos: [], status: 'not_facing', confidence: 'unverified', need: { code: 'box.face.top' } } } } as unknown as LayerBox;
assert.ok(validBox(box));
assert.ok(validBox({ ...box, highlightReasons: undefined, snapNote: null }));
assert.ok(!validBox({ ...box, label: 'legacy text' }));
assert.ok(!validBox({ ...box, highlightReasons: ['legacy text'] }));
assert.ok(!validBox({ ...box, snapNote: 3 }));
assert.ok(!validBox({ ...box, faces: { top: { ...box.faces.top!, need: ['retake'] } } }));

const revision = { id: 'r1', document: { entities: [{ id: 'a', label: 'Imported evidence' }, { id: 'b', label: 'Unlabelled evidence' }], assets: [],
  coordinateFrames: [{ id: 'f', scale: { status: 'uncalibrated', nativeToMeters: null } }] } } as unknown as Revision;
const layer = { schemaVersion: 2, publicationId: 'p1', revisionId: 'r1', coordinateFrameId: 'f', scale: { nativeToMeters: .02, status: 'operator_anchored', source: { code: 'viewer.ground' } },
  labels: { a: { code: 'box.face.front' } }, boxes: { a: box } } as MeasurementLayer;
const before = JSON.stringify({ revision, layer });
for (const language of [...LANGUAGES, 'en'] as const) {
  const shown = withLocalizedLabels(applyMeasurementLayer(revision, layer), layer, language);
  assert.deepEqual(shown.document.entities.map(entity => entity.label), [translate(language, 'box.face.front'), 'Unlabelled evidence']);
  assert.equal(shown.document.coordinateFrames[0].scale.source, translate(language, 'viewer.ground'));
}
assert.equal(withLocalizedLabels(revision, { ...layer, revisionId: 'r0' }, 'en'), revision);
assert.equal(withLocalizedLabels(revision, null, 'en'), revision);
assert.equal(JSON.stringify({ revision, layer }), before, 'language display does not mutate geometry or messages');
const fetchBefore = globalThis.fetch;
for (const candidate of [layer, { ...layer, schemaVersion: 1 }, { ...layer, labels: { a: 'legacy text' } }]) {
  globalThis.fetch = async () => ({ ok: true, json: async () => candidate }) as Response;
  assert.equal(await loadMeasurementLayer('p1'), candidate === layer ? layer : null);
}
globalThis.fetch = fetchBefore;
console.log('PASS: schema 2 nested messages, strict message validation, localized labels/source and immutable geometry in en/zh/nl');

// Exercise every producer message with numbers and the same nested face/level/status codes it emits.
const nested: Record<string, DisplayMessage> = {
  face: { code: 'box.face.front' }, dimension: { code: 'box.dim.H' }, level: { code: 'box.level.high' }, confidence: { code: 'box.level.high' },
  part: { code: 'measurement.part.housingRight' }, verdict: { code: 'measurement.shape.ok' }, status: { code: 'measurement.floor.on_floor' },
  result: { code: 'measurement.penalty', params: { penalty: 1.25 } }, reason: { code: 'measurement.need.fixMasks' },
  other: { code: 'measurement.object.robot' }, gates: { code: 'measurement.passed' },
};
const messages = Object.entries(catalog).filter(([code]) => code.startsWith('measurement.'));
assert.ok(messages.length, 'producer messages must be present in the viewer catalog');
for (const [code, text] of messages) {
  const params = Object.fromEntries([...text.matchAll(/\{(\w+)\}/g)].map(([, name]) => [name, nested[name] || 1.25]));
  for (const language of LANGUAGES) {
    const rendered = renderMessage(language, { code, params });
    assert.doesNotMatch(rendered, /\{\w+\}/, `${language}: ${code}`);
    assert.ok(rendered && !rendered.includes(code), `${language}: ${code} must render its display message`);
    for (const value of Object.values(params)) assert.ok(rendered.includes(typeof value === 'object' ? renderMessage(language, value) : String(value)), `${language}: ${code} must substitute its parameters`);
    if (language !== 'zh') assert.doesNotMatch(rendered, /[\u3400-\u9fff]/, `${language}: ${code}`);
  }
}
console.log(`PASS: ${messages.length} producer messages × 3 languages, including nested face/confidence/status parameters`);
