// Run: node --experimental-strip-types tests/measurement-layer-check.ts
// Layer texts in the viewer's language: the "En" sibling in English mode, the Chinese field whenever it is absent.
import assert from 'node:assert/strict';
import { layerText, validBox, withEnglishLabels, type LayerBox, type MeasurementLayer } from '../src/measurement-layer.ts';
import type { Revision } from '../src/types.ts';

assert.equal(layerText('zh', '高', 'High'), '高');
assert.equal(layerText('en', '高', 'High'), 'High');
assert.equal(layerText('en', '高', undefined), '高');
assert.equal(layerText('en', '补拍顶面', null), '补拍顶面');
assert.equal(layerText('en', null, null), null);
assert.deepEqual(layerText('en', ['遮挡'], ['occluded']), ['occluded']);
assert.deepEqual(layerText('en', ['遮挡'], undefined), ['遮挡']);
assert.equal(layerText('en', undefined, undefined), undefined);

// A layer box with English siblings stays valid; a malformed English sibling leaves the box out, like a malformed Chinese one.
const x = [1, 0, 0], y = [0, 1, 0], z = [0, 0, 1], dim = { valueM: .1, sigmaCm: .5, confidence: 'high' };
const box = { label: 'e-stop', centerNative: [0, 0, 0], axes: [x, y, z], sizeM: [.1, .1, .1], bottomM: 0, topM: .1, floorContact: true,
  faceNormals: { front: [0, -1, 0], back: y, left: [-1, 0, 0], right: x, top: z, bottom: [0, 0, -1] },
  dims: { L: dim, W: dim, H: dim, bottom: dim }, highlight: true, confidence: 'medium',
  highlightReasons: ['顶面未见'], highlightReasonsEn: ['top not seen'], snapNote: '贴地', snapNoteEn: 'on the floor',
  faces: { top: { photos: [], status: 'not_facing', confidence: 'unverified', need: '补拍顶面', needEn: 'retake the top' } } } as unknown as LayerBox;
assert.ok(validBox(box));
assert.ok(validBox({ ...box, highlightReasonsEn: undefined, snapNoteEn: null }));
assert.ok(!validBox({ ...box, highlightReasonsEn: 'top not seen' }));
assert.ok(!validBox({ ...box, snapNoteEn: 3 }));
assert.ok(!validBox({ ...box, faces: { top: { ...box.faces.top!, needEn: ['retake'] } } }));

// Entity names: the load applied the Chinese ones; English mode puts labelsEn over them, only for the revision the layer names.
const revision = { id: 'r1', document: { entities: [{ id: 'a', label: '急停' }, { id: 'b', label: '护栏' }] } } as unknown as Revision;
const layer = { revisionId: 'r1', labels: { a: '急停', b: '护栏' }, labelsEn: { a: 'E-stop' } } as unknown as MeasurementLayer;
const names = (r: Revision) => r.document.entities.map(entity => entity.label);
assert.equal(withEnglishLabels(revision, layer, 'zh'), revision);
assert.deepEqual(names(withEnglishLabels(revision, layer, 'en')), ['E-stop', '护栏']);
assert.equal(withEnglishLabels(revision, { ...layer, labelsEn: undefined }, 'en'), revision);
assert.equal(withEnglishLabels(revision, { ...layer, revisionId: 'r0' }, 'en'), revision);
assert.equal(withEnglishLabels(revision, null, 'en'), revision);
console.log('measurement layer language: ok');
