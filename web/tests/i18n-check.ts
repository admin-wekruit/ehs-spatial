// Run: node --experimental-strip-types tests/i18n-check.ts
import assert from "node:assert/strict";
import { LANGUAGES, translate } from "../src/translate.ts";
import { mountSceneViewer } from "../src/viewer/native-viewer.ts";
import { answerSpatialQuery } from "../src/spatial-query.ts";

// A DOM/WebGL call sink checks the real mount/update path without a GPU or assets.
const gl = new Proxy({}, { get: (_, key) => key === "getExtension" ? () => null : key === "getShaderParameter" || key === "getProgramParameter" ? () => true : key === "isContextLost" ? () => false : () => ({}) });
class Element extends EventTarget {
  name: string; style = {}; children: Element[] = []; attributes = new Map<string, string>();
  constructor(name: string) { super(); this.name = name; }
  setAttribute(name: string, value: string) { this.attributes.set(name, value); }
  removeAttribute(name: string) { this.attributes.delete(name); }
  append(...children: Element[]) { this.children.push(...children); }
  remove() {}
  getContext() { return gl; }
}
Object.assign(globalThis, {
  document: { createElement: (name: string) => new Element(name), createElementNS: (_: string, name: string) => new Element(name) },
  ResizeObserver: class { observe() {} disconnect() {} },
  requestAnimationFrame: () => 1, cancelAnimationFrame: () => {},
});
const host = new Element("host");
let assetRequests = 0;
const viewer = mountSceneViewer(host as unknown as HTMLElement, { resolveAsset: async () => { assetRequests++; return "unused"; } });
const canvas = host.children[0].children.find(element => element.name === "canvas")!;
assert.equal(canvas.attributes.get("aria-label"), "Interactive scene", "default viewer language is English");
const labels: string[] = [];
for (const language of [...LANGUAGES, "en"] as const) {
  viewer.setLocale(language);
  const label = canvas.attributes.get("aria-label")!;
  assert.equal(label, translate(language, "viewer.interactiveScene"));
  labels.push(label);
  const answer = answerSpatialQuery("robot height", { language, objects: [], endpoints: [], differences: [], bound: () => true, format: String, revisionLabel: "r1" });
  assert.equal(answer.status, "missing");
  assert.doesNotMatch(answer.text + answer.interpretation, /\{\w+\}/);
  if (language !== "zh") assert.doesNotMatch(answer.text + answer.interpretation, /[\u3400-\u9fff]/);
}
assert.equal(new Set(labels.slice(0, 3)).size, 3, "English, Chinese and Dutch have distinct visible labels");
assert.equal(host.children.length, 1, "language updates keep the mounted viewer");
assert.equal(assetRequests, 0, "language updates do not reload assets");
viewer.dispose();
console.log("PASS: mounted viewer en → zh → nl → en labels and localized spatial answers");
