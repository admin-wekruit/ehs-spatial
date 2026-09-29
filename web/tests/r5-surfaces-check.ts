// r5 (models): an object's model in the scene document: SAM 3D's accepted mesh first, then the observed-surface GLB (the video's
// surface opaque + the primitive translucent), then the card's primitive, then the box.   node --experimental-strip-types tests/r5-surfaces-check.ts
import { liveDocument } from "../src/live-report.ts";

const patch = (layer: string, data: any, blobs: Record<string, any> = {}) =>
  ({ schema: "panoptes-fast-patch-v1", report: "r", seq: 1, layer, version: 1, status: "generated", labels: [], t0_unix: 0, sent_s: 1, data, blobs });
const blob = (sha: string) => ({ sha256: sha, bytes: 10, mediaType: "model/gltf-binary", format: "glb" });
const obj = (id: string) => ({ id, shot: 0, word: "box", box_min_m: [0, 0, 0], box_max_m: [1, 1, 1] });
const layers: any = {
  objects: patch("objects", { objects: ["a", "b", "c", "d"].map(obj) }),
  models: patch("models", { models: [{ object: "a", transform: { position: [1, 2, 3], quaternion: [0, 0, 0, 1], scale: [1, 1, 1] } }] }, { "model-a": blob("aa") }),
  surfaces: patch("surfaces", { surfaces: [{ object: "a", position: [0, 0, 0] }, { object: "b", position: [.5, .5, .5], observed_triangles: 10 }] },
    { "surface-a": blob("sa"), "surface-b": blob("sb") }),
};
const prim = { kind: "box", size_m: [1, 1, 1], faces: {}, position: [0, 0, 0], quaternion: [0, 0, 0, 1] };
const cards = ["a", "b", "c"].map(id => ({ id, kind: "object", model: prim }));
const doc: any = liveDocument("r", layers, cards);
const rep = (id: string) => doc.entities.find((e: any) => e.id === id).representations[0];
console.assert(rep("a").id === "model:a", "SAM 3D's accepted mesh wins");
console.assert(rep("b").id === "surface:b" && rep("b").kind === "generated_mesh" && rep("b").transform.position[0] === .5 && rep("b").assetId === "sha256:sb",
  "the observed surface GLB, about its centre");
console.assert(rep("c").id === "prim:c" && rep("d").id === "box:d", "then the primitive, then the box");
if ([rep("a").id === "model:a", rep("b").id === "surface:b", rep("c").id === "prim:c", rep("d").id === "box:d"].some(x => !x)) process.exit(1);
console.log("r5 surfaces check passed: SAM 3D mesh > observed surface > primitive > box");
