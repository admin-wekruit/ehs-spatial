// Run: node checks/photo-scale-check.mjs — actual component contract check, no browser/GPU.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url), ts = require('typescript'), React = require('react');
let state = 0, height = 10, exported, filename, endpointLines = false, selectedPhoto = 'photo-4';
const reportLocation = { href: 'https://example.test/report/' };
class Group {
  children = []; scaleValue = 1; rotation = {}; matrix = { fromArray() {} };
  scale = { setScalar: value => { this.scaleValue = value; } };
  add(child) { this.children.push(child); }
  updateMatrixWorld() {}
}
const ReportScene = () => null, SemanticObject = () => null, module = { exports: {} };
const source = fs.readFileSync(new URL('../src/PhotoReport.tsx', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 } }).outputText;
vm.runInNewContext(compiled, {
  exports: module.exports, module,
  require: id => {
    if (id === 'react') return { ...React, useEffect: () => {}, useState: initial => [state++ === 3 ? height : state === 1 ? selectedPhoto : state === 2 ? 'fence-0' : state === 7 ? true : state === 8 ? endpointLines ?? initial : initial, () => {}], useMemo: fn => fn() };
    if (id === 'react/jsx-runtime') return require(id);
    if (id === 'react-dom/client') return { createRoot: () => ({ render() {} }) };
    if (id === './ReportScene') return { ReportScene };
    if (id === './PhotoSemanticExperiment') return { PhotoSemanticExperiment: () => null, PhotoSemanticObject: SemanticObject };
    if (id === './SceneResources') return { SceneResources: { Provider: 'provider' } };
    if (id === './core') return { activeModel: entity => entity.representations[0] };
    if (id === './viewer/native-math') return { transformMatrix: () => Array.from({ length: 16 }, (_, i) => +(i % 5 === 0)) };
    if (id.endsWith('three.module.js')) return { Scene: Group, Group };
    if (id.endsWith('GLTFLoader.js')) return { GLTFLoader: class { async loadAsync() { return { scene: {} }; } } };
    if (id.endsWith('GLTFExporter.js')) return { GLTFExporter: class { async parseAsync(scene) { exported = scene; return new ArrayBuffer(8); } } };
    return {};
  },
  document: { getElementById: () => ({}), createElement: () => ({ click() { filename = this.download; } }) },
  window: { location: reportLocation }, URL, Blob, setTimeout: fn => fn(),
}, { filename: 'PhotoReport.tsx' });
const walk = node => !node || typeof node !== 'object' ? [] : [node, ...[node.props?.children].flat(Infinity).flatMap(walk)];
const feature = { valueNative: .4, pointNative: [0, 0, .4], footNative: [0, 0, 0], sourcePhotos: [1, 2], rangeNative: [.39, .41] };
const entities = Array.from({ length: 52 }, (_, i) => ({ id: i ? `object-${i}` : 'fence-0', visible: true, representations: [{ id: `rep-${i}`, assetId: `asset-${i}`, coordinateFrameId: 'world', transform: {} }] }));
const data = {
  revision: { id: 'r', projectId: 'p', documentSha256: 'hash', document: { coordinateFrames: [{ id: 'world', scale: { status: 'model_estimated', nativeToMeters: 9 } }], entities, cameras: [{ imageId: 'photo-4', id: 'camera-4' }] } },
  assetURLs: Object.fromEntries(entities.map((_, i) => [`asset-${i}`, `${i}.glb`])),
  objects: [{ id: 'fence-0', label: '围栏', measurements: {}, observations: [{ photo: 1 }, { photo: 2 }], notes: [], groundDistance: { feature, byPhoto: { '4': { ...feature, valueNative: 99 } } } }],
  geometry: { anchor: { nativeHeight: .001, nativeWidth: 999, assumedHeightM: .1, mPerNative: .5, referenceFit: { status: 'available', mPerNative: .5, candidateMPerNative: 7 } }, floor: { status: 'conditional' }, calibration: { nativeToMeters: .5, reference: { scopeStatus: 'confirmed', features: { wholeComponentHeightM: .1, mainBodyDiameterM: .085, redActuatorDiameterM: .04 } } } },
  timing: {}, nativeToMetersDefault: 123, modelMeasurementScale: { nativeToMeters: .5, status: 'accepted_3d_reference', source: 'test' },
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
const unsupported = structuredClone(data); unsupported.geometry.anchor.mPerNative = null; unsupported.geometry.anchor.referenceFit.status = 'unsupported'; unsupported.modelMeasurementScale = {nativeToMeters:null,status:'uncalibrated',source:'test'};
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
const conditional = structuredClone(unsupported);
conditional.revision.document.entities[0].representations[0].coordinateFrameId = 'workcell-floor';
conditional.revision.document.entities[1].id = 'post-box-1';
conditional.revision.document.assets = [{ id: 'asset-0', sha256: 'h0' }, { id: 'asset-1', sha256: 'h1' }];
conditional.objects.push({ id: 'post-box-1', label: '光幕', measurements: {}, observations: [{ photo: 4 }], notes: [], groundDistance: { byPhoto: {}, feature: null, source: 'fixture' } });
conditional.modelMeasurementScale = {nativeToMeters:.5,rangeNativeToMeters:[.45,.55],evidence:{conditionalMPerNative:.5},status:'conditional_unvalidated',source:'test'};
conditional.semanticExperiment = { objects: [{ entityId: 'fence-0', label: '围栏', sourceRefs: [], results: [], policyContext: { applicability: 'unknown', machineResult: null } }] };
conditional.endpointEstimation = { status: 'conditional_unvalidated', sidePhoto: 4, method: 'model endpoints', endpoints: [
  { id: 'fence-0:near:post-box-1', objectId: 'fence-0', label: '右侧光幕旁围栏下沿', side: 'right', measurementScope: 'model_lower_rail_near_curtain', pointNative: [5, 7, .3], footNative: [5, 7, 0], heightNative: .3, representationId: 'rep-0', assetId: 'asset-0', assetSha256: 'h0', modelFile: 'fence-fitted.glb', modelSha256: 'm0', provenance: 'fixture' },
  { id: 'post-box-1:terminal', objectId: 'post-box-1', label: '右侧光幕底端', side: 'right', measurementScope: 'model_bottom_face_center', pointNative: [1, 2, .4], footNative: [1, 2, 0], heightNative: .4, representationId: 'rep-1', assetId: 'asset-1', assetSha256: 'h1', modelFile: 'posts.glb', modelSha256: 'm1', pairedEndpointId: 'fence-0:near:post-box-1', provenance: 'fixture' },
], differences: [{ id: 'post-box-1:terminal-minus-rail', label: '右侧光幕底端减去旁边围栏下沿', minuendId: 'post-box-1:terminal', subtrahendId: 'fence-0:near:post-box-1', valueNative: .1, description: 'relative height' }] };
endpointLines = true;
result = render(conditional);
assert.equal(result.scale, 'unknown', 'conditional endpoint estimates cannot promote the accepted scene scale');
assert.equal(result.ground, '未知', 'endpoint estimates cannot overwrite the physical groundDistance card');
assert.equal(result.nodes.find(n => n.props['data-endpoint-difference'] === 'post-box-1:terminal-minus-rail').props.children, '5.00 cm');
assert.equal(result.nodes.find(n => n.props['data-endpoint-estimate'] === 'fence-0:near:post-box-1').props.children, '15.00 cm');
const inspector = walk(result.scene.props.inspector(null));
assert.equal(inspector.find(n => n.props?.['aria-label'] === '光幕与围栏底边离地').type, 'section', 'the estimate must remain visible inside the fullscreen inspector');
const semanticFacts = scale => walk(render(conditional, scale).scene.props.inspector(null)).find(n => n.type === SemanticObject).props.facts;
assert.equal(semanticFacts(10).find(fact => fact.testId === 'fence-0:near:post-box-1').value, '15.00 cm', 'semantic facts read the revision measurement');
assert.equal(semanticFacts(20).find(fact => fact.testId === 'fence-0:near:post-box-1').value, '30.00 cm', 'semantic facts follow the scale trial, no snapshot');
assert.equal(semanticFacts(10).find(fact => fact.testId === 'post-box-1:terminal-minus-rail').value, '5.00 cm');
const annotation = result.scene.props.measurementOverride;
assert.equal(annotation.method, 'conditional-endpoint-comparison');
assert.equal(annotation.coordinateFrameId, 'workcell-floor');
assert.equal(annotation.documentSha256, 'hash');
assert.deepEqual(Array.from(annotation.lines[3].points, point => Array.from(point)), [[5, 7, .4], [5, 7, .3]], 'high-low difference must follow floor Z; endpoint array order must not change the meaning');
assert.match(render(conditional, 20).scene.props.measurementOverride.displayLabel, /10.00 cm/, 'all model measurements track the same explicit scale');
const nullScale = structuredClone(conditional); nullScale.modelMeasurementScale = { nativeToMeters: null, status: 'uncalibrated', source: 'test' };
assert.equal(render(nullScale).nodes.find(n => n.props['data-endpoint-estimate'] === 'fence-0:near:post-box-1').props.children, '0.3000 native', 'unknown scale keeps native units');
result = render(conditional, 20);
assert.equal(result.scene.props.measurementScale.nativeToMeters, 1);
await result.nodes.find(n => n.type === 'button' && n.props.children === '下载条件标尺模型 GLB').props.onClick();
assert.equal(exported.children[0].scaleValue, 1);
assert.equal(exported.userData.scaleStatus, 'conditional_unvalidated');
assert.equal(exported.userData.groundTruth, false);
assert.equal(exported.userData.documentSha256, 'hash'); assert.equal(exported.userData.revisionId, 'r');
const scaleMetadata = JSON.parse(JSON.stringify(exported.userData.modelMeasurementScale));
assert.deepEqual(scaleMetadata.rangeNativeToMeters, [.9,1.1]);
assert.equal(scaleMetadata.uniformReferenceRatio, 2);
assert.deepEqual(scaleMetadata.currentReferenceDimensionsM, {wholeComponentHeightM:.2,mainBodyDiameterM:.17,redActuatorDiameterM:.08});
assert.deepEqual(scaleMetadata.baseModelMeasurementScale.rangeNativeToMeters, [.45,.55]);
assert.equal(scaleMetadata.baseModelMeasurementScale.evidence.conditionalMPerNative,.5);
assert.equal(scaleMetadata.evidence, undefined, 'baseline evidence must not masquerade as the current trial');
assert.match(scaleMetadata.source, /同比试算/);
assert.equal(result.scene.props.revision.document.coordinateFrames[0].scale.nativeToMeters, null, 'conditional scale never becomes accepted physical scale');
// The measured representation must be the displayed one: another photo's model variant is not read.
const variant = structuredClone(conditional); variant.revision.document.entities[0].modelVariants = { '3': { ...variant.revision.document.entities[0].representations[0], id: 'rep-0-photo-3' } };
selectedPhoto = 'photo-3';
result = render(variant);
assert.notEqual(result.scene.props.measurementOverride?.method, 'conditional-endpoint-comparison', 'an endpoint is never drawn on another representation');
assert.match(result.nodes.find(n => n.props['data-endpoint-estimate'] === 'fence-0:near:post-box-1').props.children, /未知/);
selectedPhoto = 'photo-4'; endpointLines = undefined; reportLocation.href = 'https://example.test/report/?measurement=endpoints';
assert.equal(render(conditional).scene.props.measurementOverride.method, 'conditional-endpoint-comparison', 'the shared URL enables annotations without another click');
console.log('PASS: accepted scale, 52-model exports, no bbox fallback, source points preserved, unified conditional model scale/export, revision-bound endpoints and live semantic facts');
