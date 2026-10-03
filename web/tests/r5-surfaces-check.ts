// r5b (models): an object's model in the scene document: its generated mesh (tier 1, or a look-alike's copy with its own pose),
// then the primitive its fixed-shape class fits, then its observed surface (tier 0: a node of the shot's one GLB), then the box;
// the 3D label is the card's shown name; an on-demand card's inline surface is drawn.   node --experimental-strip-types tests/r5-surfaces-check.ts
import { liveDocument } from "../src/live-report.ts";

const patch = (layer: string, data: any, blobs: Record<string, any> = {}) =>
  ({ schema: "panoptes-fast-patch-v1", report: "r", seq: 1, layer, version: 1, status: "generated", labels: [], t0_unix: 0, sent_s: 1, data, blobs });
const blob = (sha: string) => ({ sha256: sha, bytes: 10, mediaType: "model/gltf-binary", format: "glb" });
const obj = (id: string, extra: any = {}) => ({ id, shot: 0, word: "unidentified object", box_min_m: [0, 0, 0], box_max_m: [1, 1, 1], ...extra });
const layers: any = {
  objects: patch("objects", { objects: [..."abcdef"].map(k => obj(k)).concat([obj("g", { merged_into: "b" })]) }),
  models: patch("models", { models: [{ object: "a", transform: { position: [1, 2, 3], quaternion: [0, 0, 0, 1], scale: [1, 1, 1] }, bounds: { min: [.5, 1.5, 2.5], max: [1.5, 2.5, 3.5] } },
    { object: "f", reuse_of: "a", transform: { position: [4, 2, 3], quaternion: [0, 0, .7071, .7071], scale: [1.1, 1.1, 1.1] } }] }, { "model-a": blob("aa") }),
  surfaces: patch("surfaces", { surfaces: ["a", "b", "c"].map(o => ({ object: o, blob: "shot-0", min: [0, 0, 0], max: [1, 1, 1], triangles: 10 })) },
    { "shot-0": blob("s0") }),
};
const prim = { kind: "box", tier: "primitive", size_m: [1, 1, 1], faces: {}, position: [0, 0, 0], quaternion: [0, 0, 0, 1] };
const t0 = { kind: "observed surface", tier: 0 };
const cards = [["a", t0, "lathe"], ["b", t0, "sneaker"], ["c", prim, "cardboard box"], ["d", t0, "shelf"], ["e", { ...prim, tier: undefined }, "drum"], ["f", t0, "lathe"]]
  .map(([id, model, name]) => ({ id, kind: "object", model, identity: { name } }));
const od = { id: "ondemand:1:2:3", status: "card", shot: 0, identity: { name: "vise" }, model: { tier: 0, glb_b64: "Z2xURg==", node: "ondemand:1:2:3", bounds: { min: [0, 0, 0], max: [1, 1, 1] } } };
const doc: any = liveDocument("r", layers, cards, [od]);
const ent = (id: string) => doc.entities.find((e: any) => e.id === id);
const rep = (id: string) => ent(id).representations[0];
const checks: [boolean, string][] = [
  [rep("a").id === "model:a" && rep("a").assetId === "sha256:aa", "the generated mesh wins"],
  [rep("f").id === "model:f" && rep("f").assetId === "sha256:aa" && rep("f").transform.scale[0] === 1.1 && rep("f").transform.position[0] === 4,
    "a look-alike's copy: the rep's GLB at its own pose"],
  [rep("f").bounds?.min[0] === -.5 && rep("a").bounds?.max[2] === .5, "a copy carries its group model's local bounds (the 3D label, the preview)"],
  [rep("b").id === "surface:b" && rep("b").node === "b" && rep("b").assetId === "sha256:s0" && rep("b").transform.position[0] === 0, "tier 0: the shot GLB's node"],
  [rep("c").id === "prim:c", "a checked primitive before the surface"],
  [rep("d").id === "box:d" && rep("e").id === "box:e", "no surface and no checked primitive: the see-through box"],
  [ent("b").label === "sneaker" && ent("a").label === "lathe", "3D labels are the cards' names"],
  [!ent("g"), "a merged-away object is drawn by its card"],
  [rep("ondemand:1:2:3").node === "ondemand:1:2:3" && doc.assets.some((a: any) => a.inlineGlb === "Z2xURg=="), "the on-demand card's inline surface"],
];
for (const [ok, what] of checks) console.assert(ok, what);
if (checks.some(([ok]) => !ok)) { console.error(checks.filter(([ok]) => !ok).map(([, w]) => w)); process.exit(1); }
console.log("r5b models check passed: generated (and look-alike copies) > checked primitive > observed-surface node > box; card names; on-demand surface");
