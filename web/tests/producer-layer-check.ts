/** The real Python builder's serialized output must pass the actual viewer boundary and localize without changing geometry. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { applyMeasurementLayer, boxGeometry, loadMeasurementLayer, renderMessage, validBox, withLocalizedLabels, type MeasurementLayer } from '../src/measurement-layer.ts';
import type { Revision } from '../src/types.ts';

const layer = JSON.parse(readFileSync(process.argv[2], 'utf8')) as MeasurementLayer;
const frozen = JSON.stringify(layer);
globalThis.fetch = async () => ({ ok: true, json: async () => layer }) as Response;
assert.equal(await loadMeasurementLayer(layer.publicationId), layer);
const revision = {
  id: layer.revisionId,
  document: { assets: [], entities: Object.keys(layer.labels!).map(id => ({ id, label: id })),
    coordinateFrames: [{ id: layer.coordinateFrameId, scale: {}, ground: {} }] },
} as unknown as Revision;
const applied = applyMeasurementLayer(revision, layer);
assert.equal(applied.document.coordinateFrames[0].scale.nativeToMeters, layer.scale.nativeToMeters);
const names = new Map<string, Set<string>>();
for (const lang of ['en', 'zh', 'nl', 'en'] as const) {
  const localized = withLocalizedLabels(applied, layer, lang);
  for (const entity of localized.document.entities) {
    assert.equal(typeof entity.label, 'string');
    assert.ok(entity.label);
    names.set(entity.id, (names.get(entity.id) ?? new Set()).add(entity.label));
  }
  for (const box of Object.values(layer.boxes!)) {
    assert.ok(validBox(box));
    assert.ok(boxGeometry(box, layer.scale.nativeToMeters, layer.ground));
    assert.ok(renderMessage(lang, box.label).length);
    for (const face of Object.values(box.faces)) if (face?.need) assert.ok(renderMessage(lang, face.need).length);
    for (const reason of box.highlightReasons ?? []) assert.ok(renderMessage(lang, reason).length);
  }
  for (const pipeline of Object.values(layer.pipelines!)) {
    for (const stage of pipeline.stages) {
      assert.ok(renderMessage(lang, stage.label).length);
      assert.ok(renderMessage(lang, stage.text).length);
    }
  }
}
assert.ok([...names.values()].some(values => values.size === 3));
assert.equal(JSON.stringify(layer), frozen);
console.log(`PASS: actual Python builder → schema loader → boxes/pipelines → en/zh/nl/en (${Object.keys(layer.boxes!).length} boxes)`);
