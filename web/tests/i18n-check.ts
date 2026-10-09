// Run: node --experimental-strip-types tests/i18n-check.ts
import assert from "node:assert/strict";
import { LANGUAGES, translate } from "../src/translate.ts";
import { mountSceneViewer } from "../src/viewer/native-viewer.ts";
import { point } from "../src/viewer/native-math.ts";
import { answerSpatialQuery } from "../src/spatial-query.ts";
import { applyMeasurementLayer, boxGeometry, renderMessage, validBox, withLocalizedLabels, type LayerBox, type MeasurementLayer } from "../src/measurement-layer.ts";
import type { Revision } from "../src/types.ts";

// The real viewer runs against a DOM/WebGL stub: capture submitted geometry, not rendered pixels.
const bound = new Map<number, object>(), buffers = new Map<object, Float32Array | Uint32Array>();
const draws: { vertices: number[]; indices: number[]; matrix: number[] }[] = [];
const constants = new Map<string | symbol, number>([["NO_ERROR", 0], ["ARRAY_BUFFER", 34962], ["ELEMENT_ARRAY_BUFFER", 34963]]);
let uploads = 0, matrix: number[] = [];
const methods: Record<string, (...args: any[]) => any> = {
  getExtension: () => null, getShaderParameter: () => true, getProgramParameter: () => true,
  isContextLost: () => false, getError: () => 0, getUniformLocation: (_: object, name: string) => name,
  getAttribLocation: () => 0,
  bindBuffer: (target: number, buffer: object) => bound.set(target, buffer),
  bufferData: (target: number, data: Float32Array | Uint32Array) => { buffers.set(bound.get(target)!, data); uploads++; },
  uniformMatrix4fv: (name: string, _: boolean, data: ArrayLike<number>) => { if (name === "model") matrix = Array.from(data); },
  drawElements: (_: number, count: number) => {
    const vertices = Array.from(buffers.get(bound.get(34962)!)!), indices = Array.from(buffers.get(bound.get(34963)!)!);
    assert.equal(count, indices.length);
    draws.push({ vertices, indices, matrix: [...matrix] });
  },
};
const gl = new Proxy({}, { get: (_, key) => {
  if (methods[String(key)]) return methods[String(key)];
  if (/^[A-Z_]+$/.test(String(key))) { if (!constants.has(key)) constants.set(key, constants.size + 1); return constants.get(key); }
  return () => ({});
} });
class Element extends EventTarget {
  name: string; style = {}; children: Element[] = []; attributes = new Map<string, string>();
  namespaceURI = "http://www.w3.org/2000/svg"; textContent = ""; clientWidth = 800; clientHeight = 600;
  constructor(name: string) { super(); this.name = name; }
  setAttribute(name: string, value: string) { this.attributes.set(name, value); }
  removeAttribute(name: string) { this.attributes.delete(name); }
  append(...children: Element[]) { this.children.push(...children); }
  replaceChildren(...children: Element[]) { this.children = children; }
  remove() {}
  getContext() { return gl; }
}
let frameId = 0;
const frames = new Map<number, () => void>();
function flush() { while (frames.size) { const pending = [...frames.values()]; frames.clear(); pending.forEach(callback => callback()); } }
Object.assign(globalThis, {
  document: { createElement: (name: string) => new Element(name), createElementNS: (_: string, name: string) => new Element(name) },
  ResizeObserver: class { observe() {} disconnect() {} },
  devicePixelRatio: 1,
  requestAnimationFrame: (callback: () => void) => { frames.set(++frameId, callback); return frameId; }, cancelAnimationFrame: (id: number) => frames.delete(id),
});
const host = new Element("host");
let assetRequests = 0;
const errors: unknown[] = [];
const viewer = mountSceneViewer(host as unknown as HTMLElement, { resolveAsset: async () => { assetRequests++; return "unused"; }, onEvent: event => { if (event.type === "loadError") errors.push(event); } });
const canvas = host.children[0].children.find(element => element.name === "canvas")!;
assert.equal(canvas.attributes.get("aria-label"), "Interactive scene", "default viewer language is English");
const ground = { normal: [0, 0, 1] as [number, number, number], offset: 0, plane: [0, 0, 1, 0] as [number, number, number, number], source: { code: "viewer.ground" } };
const dim = { valueM: 2, sigmaCm: .5, confidence: "high" as const };
const box: LayerBox = { label: { code: "box.face.front" }, centerNative: [0, 0, 2], axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], sizeM: [2, 3, 4], bottomM: 0, topM: 4, floorContact: true,
  faceNormals: { front: [0, -1, 0], back: [0, 1, 0], left: [-1, 0, 0], right: [1, 0, 0], top: [0, 0, 1], bottom: [0, 0, -1] },
  dims: { L: dim, W: { ...dim, valueM: 3 }, H: { ...dim, valueM: 4 }, bottom: { ...dim, valueM: 0 } }, faces: {}, highlight: false, confidence: "high" };
assert.ok(validBox(box));
const representation = { id: "model", kind: "primitive", coordinateFrameId: "f", primitive: { kind: "box", dimensions: box.sizeM }, placementState: "confirmed",
  transform: { coordinateFrameId: "f", position: box.centerNative, quaternion: [0, 0, 0, 1], scale: [1, 1, 1] } };
const revision = { id: "r1", document: { captureId: "capture", entities: [{ id: "box", label: "Imported box", representations: [] }], cameras: [], observations: [], assets: [], coordinateFrames: [{ id: "f" }] } } as unknown as Revision;
const layer = { schemaVersion: 2, publicationId: "p1", revisionId: "r1", coordinateFrameId: "f", scale: { nativeToMeters: 1, status: "operator_anchored", source: { code: "viewer.groundScaled" } },
  ground, labels: { box: box.label }, boxes: { box }, models: { box: { representation, note: box.label } } } as unknown as MeasurementLayer;
const applied = applyMeasurementLayer(revision, layer), geometry = boxGeometry(box, 1, ground)!;
const lines = geometry.corners.flatMap((point, i) => [0, 1, 2].filter(k => !(i & 1 << k)).map(k => ({ points: [point, geometry.corners[i | 1 << k]], color: "#fff" })));
const unchanged = JSON.stringify({ revision, layer, geometry, lines });
const labels: string[] = [], groundLabels: string[] = [], boxLabels: string[] = [];
let baseline: { vertices: number[]; indices: number[]; matrix: number[] } | undefined, grid: string | undefined, overlay: string | undefined;
for (const language of [...LANGUAGES, "en"] as const) {
  await viewer.setScene(withLocalizedLabels(applied, layer, language));
  viewer.setLayers({ groundDatum: true, measurementScale: layer.scale, showBounds: false,
    measurement: { revisionId: "r1", coordinateFrameId: "f", lines, labelPoint: geometry.bottomCenter, value: 4, unit: "m", displayLabel: renderMessage(language, box.label) } });
  viewer.setLocale(language);
  const count = draws.length;
  flush();
  assert.ok(draws.length > count, "a loaded box is submitted for rendering after each locale update");
  const submitted = draws.at(-1)!;
  assert.equal(submitted.vertices.length, 8 * 12);
  assert.equal(submitted.indices.length, 36);
  assert.ok(submitted.vertices.every(Number.isFinite));
  assert.deepEqual(Array.from({ length: 8 }, (_, i) => point(submitted.matrix, submitted.vertices.slice(i * 12, i * 12 + 3))), geometry.corners,
    "the uploaded primitive's world corners match the measurement box");
  baseline ??= submitted;
  assert.deepEqual(submitted, baseline, "actual vertex/index buffers and model matrix stay identical across locale updates");
  assert.equal(uploads, 2, "locale/label updates reuse the original vertex and index buffers");
  const groundSVG = host.children[0].children[1], svg = host.children[0].children[3];
  const projectedLines = JSON.stringify(svg.children.filter(element => element.name === "line").map(element => ["x1", "y1", "x2", "y2"].map(name => element.attributes.get(name))));
  overlay ??= projectedLines;
  assert.equal(projectedLines, overlay, "the projected measurement box and axes stay identical");
  const nativeGrid = JSON.stringify(groundSVG.children.map(element => [element.attributes.get("data-native-start"), element.attributes.get("data-native-end")]));
  assert.ok(groundSVG.children.length, "the scaled ground grid was drawn");
  grid ??= nativeGrid;
  assert.equal(nativeGrid, grid, "the ground grid's native geometry stays identical");
  const text = svg.children.filter(element => element.name === "text").map(element => element.textContent);
  assert.ok(text.includes(translate(language, "viewer.groundScaled")), "drawNow updates the visible ground text");
  assert.ok(text.includes(renderMessage(language, box.label)), "the measurement overlay uses the selected locale");
  groundLabels.push(translate(language, "viewer.groundScaled")); boxLabels.push(renderMessage(language, box.label));
  const label = canvas.attributes.get("aria-label")!;
  assert.equal(label, translate(language, "viewer.interactiveScene"));
  labels.push(label);
  const answer = answerSpatialQuery("robot height", { language, objects: [], endpoints: [], differences: [], bound: () => true, format: String, revisionLabel: "r1" });
  assert.equal(answer.status, "missing");
  assert.doesNotMatch(answer.text + answer.interpretation, /\{\w+\}/);
  if (language !== "zh") assert.doesNotMatch(answer.text + answer.interpretation, /[\u3400-\u9fff]/);
}
assert.equal(new Set(labels.slice(0, 3)).size, 3, "English, Chinese and Dutch have distinct visible labels");
assert.equal(new Set(groundLabels.slice(0, 3)).size, 3);
assert.equal(new Set(boxLabels.slice(0, 3)).size, 3);
assert.equal(JSON.stringify({ revision, layer, geometry, lines }), unchanged, "language display does not mutate the scene or measurement geometry");
assert.deepEqual(errors, []);
assert.equal(host.children.length, 1, "language updates keep the mounted viewer");
assert.equal(assetRequests, 0, "language updates do not reload assets");
viewer.dispose();
console.log("PASS: loaded primitive/measurement viewer en -> zh -> nl -> en text, identical box buffers/matrix/grid, no asset reload (DOM/WebGL stub)");
