// ponytail: Unicode escapes keep multilingual fixtures independent of catalogs while source text stays English.
// Run: node --experimental-strip-types tests/spatial-query-check.ts
import assert from "node:assert/strict";
import { answerSpatialQuery, type QueryEndpoint, type QueryInput } from "../src/spatial-query.ts";

const endpoints: QueryEndpoint[] = [
  { id: "post-box-1:terminal", objectId: "post-box-1", label: "\u53f3\u4fa7\u5149\u5e55\u5e95\u7aef", side: "right", measurementScope: "model_bottom_face_center", heightNative: .36 },
  { id: "post-box-2:terminal", objectId: "post-box-2", label: "\u5de6\u4fa7\u5149\u5e55\u5e95\u7aef", side: "left", measurementScope: "model_bottom_face_center", heightNative: .42 },
  { id: "fence-0:near:post-box-1", objectId: "fence-0", label: "\u53f3\u4fa7\u5149\u5e55\u65c1\u56f4\u680f\u4e0b\u6cbf", side: "right", measurementScope: "model_lower_rail_near_curtain", heightNative: .27 },
  { id: "fence-1:near:post-box-2", objectId: "fence-1", label: "\u5de6\u4fa7\u5149\u5e55\u65c1\u56f4\u680f\u4e0b\u6cbf", side: "left", measurementScope: "model_lower_rail_near_curtain", heightNative: .355 },
];
const differences = [
  { id: "post-box-1:terminal-minus-rail", label: "\u53f3\u4fa7\u5149\u5e55\u5e95\u7aef\u51cf\u53bb\u65c1\u8fb9\u56f4\u680f\u4e0b\u6cbf", minuendId: "post-box-1:terminal", subtrahendId: "fence-0:near:post-box-1", valueNative: .09 },
  { id: "post-box-2:terminal-minus-rail", label: "\u5de6\u4fa7\u5149\u5e55\u5e95\u7aef\u51cf\u53bb\u65c1\u8fb9\u56f4\u680f\u4e0b\u6cbf", minuendId: "post-box-2:terminal", subtrahendId: "fence-1:near:post-box-2", valueNative: .065 },
  { id: "curtain-left-minus-right", label: "\u5de6\u53f3\u5149\u5e55\u79bb\u5730\u5dee（\u5de6 − \u53f3）", minuendId: "post-box-2:terminal", subtrahendId: "post-box-1:terminal", valueNative: .06 },
  { id: "rail-left-minus-right", label: "\u5de6\u53f3\u56f4\u680f\u4e0b\u6cbf\u79bb\u5730\u5dee（\u5de6 − \u53f3）", minuendId: "fence-1:near:post-box-2", subtrahendId: "fence-0:near:post-box-1", valueNative: .085 },
];
const objects = [
  { id: "post-box-1", label: "\u9ec4\u8272\u5149\u5e55\u7acb\u67f1 1", kind: "yellow safety post" }, { id: "post-box-2", label: "\u9ec4\u8272\u5149\u5e55\u7acb\u67f1 2", kind: "yellow safety post" },
  { id: "fence-0", label: "\u5b89\u5168\u56f4\u680f 1", kind: "safety fence" }, { id: "fence-1", label: "\u5b89\u5168\u56f4\u680f 2", kind: "safety fence" },
  { id: "robot", label: "\u5de5\u4e1a\u673a\u5668\u4eba", kind: "robot" }, { id: "v-guard-left", label: "\u5de6\u4fa7\u6298\u5f2f\u62a4\u677f", kind: "folded guard board" },
  { id: "floor", label: "\u5730\u9762\u53c2\u8003\u9762", kind: "floor" },
];
let factor: number | null = .5, unbound = new Set<string>();
const input = (): QueryInput => ({ language: "zh", objects, endpoints, differences, bends: [{ entityId: "v-guard-left", status: "measured", result: { value: 104.27 } }],
  bound: row => !unbound.has(row.id), format: native => factor == null ? `${native.toFixed(4)} native` : `${(native * factor * 100).toFixed(2)} cm`, revisionLabel: "\u4e3b\u6a21\u578b" });

let answer = answerSpatialQuery("\u53f3\u4fa7\u5149\u5e55\u79bb\u5730\u591a\u9ad8？", input());
assert.equal(answer.status, "answered"); assert.deepEqual(answer.objects, ["post-box-1"]);
assert.match(answer.text, /\u53f3\u4fa7\u5149\u5e55\u5e95\u7aef\u79bb\u5730 18\.00 cm/); assert.equal(answer.facts[0].sourceId, "post-box-1:terminal");
answer = answerSpatialQuery("\u5de6\u53f3\u5149\u5e55\u8c01\u66f4\u9ad8", input());
assert.equal(answer.status, "answered"); assert.match(answer.text, /^\u5de6\u4fa7\u5149\u5e55\u6d4b\u70b9\u66f4\u9ad8：\u5de6 − \u53f3 = 3\.00 cm/);
assert.deepEqual(answer.objects.sort(), ["post-box-1", "post-box-2"]);
answer = answerSpatialQuery("\u5de6\u53f3\u56f4\u680f\u4e0b\u6cbf\u76f8\u5dee\u591a\u5c11", input());
assert.match(answer.text, /\u5de6\u4fa7\u56f4\u680f\u6d4b\u70b9\u66f4\u9ad8：\u5de6 − \u53f3 = 4\.25 cm/);
answer = answerSpatialQuery("\u53f3\u4fa7\u5149\u5e55\u548c\u56f4\u680f\u8c01\u66f4\u9ad8", input());
assert.equal(answer.facts.length, 1); assert.match(answer.text, /\u53f3\u4fa7\u5149\u5e55\u5e95\u7aef\u51cf\u53bb\u65c1\u8fb9\u56f4\u680f\u4e0b\u6cbf：4\.50 cm/);
factor = 1; assert.match(answerSpatialQuery("\u53f3\u4fa7\u5149\u5e55\u79bb\u5730\u591a\u9ad8", input()).text, /36\.00 cm/, "the current scale trial applies");
factor = null; assert.match(answerSpatialQuery("\u53f3\u4fa7\u5149\u5e55\u79bb\u5730\u591a\u9ad8", input()).text, /0\.3600 native/, "unknown scale stays native");
factor = .5;
answer = answerSpatialQuery("\u673a\u68b0\u81c2\u79bb\u5730\u591a\u9ad8", input());
assert.equal(answer.status, "missing"); assert.deepEqual(answer.objects, ["robot"]); assert.doesNotMatch(answer.text, /\d+\.\d+ cm/, "no invented length");
answer = answerSpatialQuery("\u5de6\u4fa7\u62a4\u677f\u6298\u5f2f\u89d2\u5ea6", input());
assert.equal(answer.status, "answered"); assert.match(answer.text, /104\.3°/);
answer = answerSpatialQuery("\u627e\u4e00\u4e0b\u5149\u5e55", input());
assert.equal(answer.status, "answered"); assert.deepEqual(answer.objects, ["post-box-1", "post-box-2"]);
answer = answerSpatialQuery("\u5149\u5e55\u79bb\u5730\u9762\u591a\u9ad8", input());
assert.equal(answer.facts.length, 2, "'\u79bb\u5730\u9762' does not turn the question into the floor object");
assert.equal(answerSpatialQuery("\u706d\u706b\u5668\u5728\u54ea", input()).status, "unrecognized");
unbound = new Set(["post-box-2:terminal"]);
assert.equal(answerSpatialQuery("\u5de6\u53f3\u5149\u5e55\u8c01\u66f4\u9ad8", input()).status, "missing", "an endpoint off the displayed model is never compared");
assert.match(answerSpatialQuery("\u5de6\u4fa7\u5149\u5e55\u79bb\u5730\u591a\u9ad8", input()).text, /\u672a\u77e5（\u663e\u793a\u6a21\u578b\u4e0e\u6d4b\u91cf\u6a21\u578b\u4e0d\u540c）/);
unbound = new Set(["fence-0:near:post-box-1"]);
answer = answerSpatialQuery("\u53f3\u4fa7\u5149\u5e55\u548c\u56f4\u680f\u8c01\u66f4\u9ad8", input());
assert.equal(answer.status, "missing", "a curtain-minus-rail pair with an unbound rail is never read"); assert.doesNotMatch(answer.text, /\d+\.\d+ cm/);
answer = answerSpatialQuery("\u5149\u5e55\u548c\u56f4\u680f\u8c01\u66f4\u9ad8", input());
assert.equal(answer.facts.length, 1); assert.match(answer.text, /\u5de6\u4fa7\u5149\u5e55\u5e95\u7aef\u51cf\u53bb\u65c1\u8fb9\u56f4\u680f\u4e0b\u6cbf：3\.25 cm.*\u672a\u6bd4\u8f83/);
unbound = new Set();
const shared = { ...input(), differences: differences.filter(row => row.id !== "rail-left-minus-right") };
assert.match(answerSpatialQuery("\u5de6\u53f3\u56f4\u680f\u4e0b\u6cbf\u76f8\u5dee\u591a\u5c11", shared).text, /\u6ca1\u6709\u5de6\u53f3\u4e24\u4e2a\u4e0d\u540c\u56f4\u680f\u6d4b\u70b9/, "one rail is never compared with itself");
// A lower-envelope rail hypothesis is a point of its own: described as such and never differenced.
const hypothesis = { ...input(), endpoints: endpoints.map(row => row.objectId === "fence-1" ? { ...row, label: "\u5de6\u4fa7\u5149\u5e55\u65c1\u56f4\u680f\u4e0b\u5305\u7edc\u5047\u8bbe", railPart: "lower_envelope_hypothesis" } : row),
  differences: differences.filter(row => !["post-box-2:terminal-minus-rail", "rail-left-minus-right"].includes(row.id)),
  excluded: [{ id: "post-box-2:terminal-minus-rail", minuendId: "post-box-2:terminal", subtrahendId: "fence-1:near:post-box-2", reason: "\u65c1\u8fb9\u7684\u56f4\u680f\u70b9\u662f\u4e0b\u5305\u7edc\u5047\u8bbe" },
             { id: "rail-left-minus-right", minuendId: "fence-1:near:post-box-2", subtrahendId: "fence-0:near:post-box-1", reason: "\u6d4b\u7684\u4e0d\u662f\u540c\u4e00\u90e8\u4f4d" }] };
answer = answerSpatialQuery("\u5de6\u53f3\u56f4\u680f\u4e0b\u6cbf\u76f8\u5dee\u591a\u5c11", hypothesis);
assert.equal(answer.status, "missing"); assert.match(answer.text, /\u4e0d\u6bd4\u8f83\u5de6\u53f3\u56f4\u680f：\u6d4b\u7684\u4e0d\u662f\u540c\u4e00\u90e8\u4f4d/); assert.doesNotMatch(answer.text, /\d+\.\d+ cm/);
answer = answerSpatialQuery("\u5de6\u4fa7\u5149\u5e55\u548c\u56f4\u680f\u8c01\u66f4\u9ad8", hypothesis);
assert.equal(answer.status, "missing"); assert.match(answer.text, /\u4e0d\u6bd4\u8f83：\u65c1\u8fb9\u7684\u56f4\u680f\u70b9\u662f\u4e0b\u5305\u7edc\u5047\u8bbe/);
answer = answerSpatialQuery("\u5149\u5e55\u548c\u56f4\u680f\u8c01\u66f4\u9ad8", hypothesis);
assert.equal(answer.facts.length, 1); assert.match(answer.text, /\u53f3\u4fa7\u5149\u5e55\u5e95\u7aef\u51cf\u53bb\u65c1\u8fb9\u56f4\u680f\u4e0b\u6cbf：4\.50 cm.*\u53e6\u4e00\u4fa7\u4e0d\u6bd4\u8f83：\u65c1\u8fb9\u7684\u56f4\u680f\u70b9\u662f\u4e0b\u5305\u7edc\u5047\u8bbe/);
assert.match(answerSpatialQuery("\u5de6\u4fa7\u56f4\u680f\u79bb\u5730\u591a\u9ad8", hypothesis).text, /\u56f4\u680f\u4e0b\u5305\u7edc\u5047\u8bbe，\u4e0d\u662f\u4e0b\u6a2a\u6881\u4e0b\u6cbf/);
// Left/right are read in the scene's reference photo: a missing side names that photo, not a fixed photo 4.
const unsided = { ...input(), endpoints: endpoints.map(row => ({ ...row, side: null })), sidePhoto: 2 };
assert.match(answerSpatialQuery("\u627e\u4e00\u4e0b\u5de6\u4fa7\u5149\u5e55", unsided).text, /\u5de6\u53f3\u53ea\u5bf9\u6709\u7167\u7247 2 \u6d4b\u70b9\u7684\u5bf9\u8c61\u7ed9\u51fa/);
console.log("PASS: structured spatial questions read the loaded revision's endpoints with the current scale; comparisons, missing facts, unknown scale and refused unlike comparisons stay explicit");
