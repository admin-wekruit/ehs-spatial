// Run: node checks/photo-scale-check.mjs — actual component contract check, no browser/GPU.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url), ts = require('typescript'), React = require('react');
let state = 0, height = 10, exported, filename;
class Group {
  children = []; scaleValue = 1; rotation = {}; matrix = { fromArray() {} };
  scale = { setScalar: value => { this.scaleValue = value; } };
  add(child) { this.children.push(child); }
  updateMatrixWorld() {}
}
const ReportScene = () => null, module = { exports: {} };
const source = fs.readFileSync(new URL('../src/PhotoReport.tsx', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 } }).outputText;
vm.runInNewContext(compiled, {
  exports: module.exports, module,
  require: id => {
    if (id === 'react') return { ...React, useState: initial => [state++ === 3 ? height : state === 2 ? 'fence-0' : state === 7 ? true : initial, () => {}], useMemo: fn => fn() };
    if (id === 'react/jsx-runtime') return require(id);
    if (id === 'react-dom/client') return { createRoot: () => ({ render() {} }) };
    if (id === './ReportScene') return { ReportScene };
    if (id === './SceneResources') return { SceneResources: { Provider: 'provider' } };
    if (id === './core') return { activeModel: entity => entity.representations[0] };
    if (id === './viewer/native-math') return { transformMatrix: () => Array.from({ length: 16 }, (_, i) => +(i % 5 === 0)) };
    if (id.endsWith('three.module.js')) return { Scene: Group, Group };
    if (id.endsWith('GLTFLoader.js')) return { GLTFLoader: class { async loadAsync() { return { scene: {} }; } } };
    if (id.endsWith('GLTFExporter.js')) return { GLTFExporter: class { async parseAsync(scene) { exported = scene; return new ArrayBuffer(8); } } };
    return {};
  },
  document: { getElementById: () => ({}), createElement: () => ({ click() { filename = this.download; } }) },
  window: { location: { href: 'https://example.test/report/' } }, URL, Blob, setTimeout: fn => fn(),
}, { filename: 'PhotoReport.tsx' });
const walk = node => !node || typeof node !== 'object' ? [] : [node, ...[node.props?.children].flat(Infinity).flatMap(walk)];
const feature = { valueNative: .4, pointNative: [0, 0, .4], footNative: [0, 0, 0], sourcePhotos: [1, 2], rangeNative: [.39, .41] };
const entities = Array.from({ length: 52 }, (_, i) => ({ id: i ? `object-${i}` : 'fence-0', visible: true, representations: [{ id: `rep-${i}`, assetId: `asset-${i}`, coordinateFrameId: 'world', transform: {} }] }));
const data = {
  revision: { id: 'r', projectId: 'p', documentSha256: 'hash', document: { coordinateFrames: [{ id: 'world', scale: { status: 'model_estimated', nativeToMeters: 9 } }], entities, cameras: [{ imageId: 'photo-4', id: 'camera-4' }] } },
  assetURLs: Object.fromEntries(entities.map((_, i) => [`asset-${i}`, `${i}.glb`])),
  objects: [{ id: 'fence-0', label: '围栏', measurements: {}, observations: [{ photo: 1 }, { photo: 2 }], notes: [], groundDistance: { feature, byPhoto: { '4': { ...feature, valueNative: 99 } } } }],
  geometry: { anchor: { nativeHeight: .001, nativeWidth: 999, assumedHeightM: .1, mPerNative: .5, referenceFit: { status: 'available', mPerNative: .5, candidateMPerNative: 7 } }, floor: { status: 'conditional' }, calibration: { nativeToMeters: .5, reference: { scopeStatus: 'confirmed', features: { wholeComponentHeightM: .1, mainBodyDiameterM: .085, redActuatorDiameterM: .04 } } } },
  timing: {}, nativeToMetersDefault: 123,
  measurementEvaluation: { comparisons: [{ objectId: 'fence-0', label: '围栏', estimateNative: .4, groundTruthM: .24, sourcePhotos: [1, 2], byPhoto: {}, rangeNative: [.39, .41] }] },
};
function render(payload, cm = 10) {
  state = 0; height = cm; const nodes = walk(module.exports.PhotoReport({ data: payload })), scene = nodes.find(n => n.type === ReportScene);
  assert.equal(scene.props.revision.document.entities.length, 52);
  const card = walk(scene.props.inspector(null));
  return { scene, nodes, scale: nodes.find(n => 'data-native-to-meters' in n.props).props['data-native-to-meters'], ground: card.find(n => 'data-ground-distance-native' in n.props).props.children };
}
let result = render(data);
assert.equal(result.scale, .5, 'accepted scale must override envelope/default/candidate'); assert.equal(result.ground, '0.200 m');
assert.equal(result.scene.props.measurementOverride.displayLabel, '0.200 m · 条件估计');
await result.nodes.find(n => n.type === 'button' && n.props.children === '下载当前模型 GLB').props.onClick();
assert.equal(exported.children[0].scaleValue, .5, 'metric export uses the accepted scale, never an identity fallback');
result = render(data, 20); assert.equal(result.scale, 1); assert.equal(result.ground, '0.400 m');
const dimensions = walk(result.nodes.find(n => 'data-measured-reference' in n.props)).filter(n => n.type === 'dd').map(n => n.props.children);
assert.deepEqual(dimensions, ['8.0 cm', '17.0 cm', '20.0 cm']);
assert.equal(result.nodes.find(n => 'data-comparison-truth' in n.props).props.children, '24.0 cm');
await result.nodes.find(n => n.type === 'button' && n.props.children === '下载当前模型 GLB').props.onClick();
assert.equal(exported.userData.units, 'metres'); assert.equal(exported.children[0].scaleValue, 1); assert.equal(exported.children[0].children.length, 52);
const unsupported = structuredClone(data); unsupported.geometry.anchor.mPerNative = null; unsupported.geometry.anchor.referenceFit.status = 'unsupported';
for (const cm of [10, 20, 0, NaN]) {
  result = render(unsupported, cm); assert.equal(result.scale, 'unknown'); assert.equal(result.ground, '未知');
  assert.equal(result.scene.props.revision.document.coordinateFrames[0].scale.status, 'uncalibrated');
  assert.equal(result.scene.props.revision.document.coordinateFrames[0].scale.nativeToMeters, null);
  assert.equal(result.nodes.find(n => 'data-comparison-estimate' in n.props).props.children, '未知');
}
await result.nodes.find(n => n.type === 'button' && n.props.children === '下载原生模型 GLB（未标定）').props.onClick();
assert.equal(exported.userData.units, 'native'); assert.equal(exported.userData.nativeToMeters, null); assert.equal(exported.children[0].scaleValue, 1); assert.equal(exported.children[0].children.length, 52); assert.match(filename, /native-unscaled/);
const noEdge = structuredClone(data); noEdge.objects[0].groundDistance.feature = null;
result = render(noEdge); assert.equal(result.ground, '未知'); assert.equal(result.scene.props.measurementOverride, null, 'per-photo minima cannot replace a physical edge');
assert.equal(render(data, 0).scale, 'unknown'); assert.equal(render(data, NaN).scale, 'unknown');
const stale = structuredClone(data); stale.geometry.anchor.referenceFit.status = 'unsupported'; assert.equal(render(stale).scale, 'unknown');
const inconsistent = structuredClone(data); inconsistent.geometry.anchor.referenceFit.mPerNative = .6; assert.equal(render(inconsistent).scale, 'unknown', 'fit and accepted anchor scale must agree');
const clouds = structuredClone(data), subject = clouds.revision.document.entities[1];
subject.representations[0].kind = 'generated_mesh';
subject.representations.push({ id: 'source-points-photo4', kind: 'point_cloud', assetId: 'cloud', sourceRefs: [{ imageId: 'photo-4', observationId: 'obs4' }] });
subject.modelVariants = { '4': { ...subject.representations[0], id: 'model-photo4' } };
const photoSubject = render(clouds).scene.props.revision.document.entities[1];
assert.equal(photoSubject.activeModelRepresentationId, 'model-photo4');
assert.deepEqual(Array.from(photoSubject.representations, r => r.id), ['model-photo4', 'source-points-photo4'], 'Changing model pose must retain source point evidence');
assert.equal(subject.representations[0].id, 'rep-1', 'Photo selection must not mutate saved representations');
console.log('PASS: accepted scale, proportional three dimensions, unchanged GT, 52-model metric/native exports, null scale, no bbox fallback, photo-model variants preserve source points');
