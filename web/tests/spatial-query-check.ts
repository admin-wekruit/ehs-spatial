// Run: node --experimental-strip-types tests/spatial-query-check.ts
import assert from "node:assert/strict";
import { answerSpatialQuery, type QueryEndpoint, type QueryInput } from "../src/spatial-query.ts";

const endpoints: QueryEndpoint[] = [
  { id: "post-box-1:terminal", objectId: "post-box-1", label: "右侧光幕底端", side: "right", measurementScope: "model_bottom_face_center", heightNative: .36 },
  { id: "post-box-2:terminal", objectId: "post-box-2", label: "左侧光幕底端", side: "left", measurementScope: "model_bottom_face_center", heightNative: .42 },
  { id: "fence-0:near:post-box-1", objectId: "fence-0", label: "右侧光幕旁围栏下沿", side: "right", measurementScope: "model_lower_rail_near_curtain", heightNative: .27 },
  { id: "fence-1:near:post-box-2", objectId: "fence-1", label: "左侧光幕旁围栏下沿", side: "left", measurementScope: "model_lower_rail_near_curtain", heightNative: .355 },
];
const differences = [
  { id: "post-box-1:terminal-minus-rail", label: "右侧光幕底端减去旁边围栏下沿", minuendId: "post-box-1:terminal", subtrahendId: "fence-0:near:post-box-1", valueNative: .09 },
  { id: "post-box-2:terminal-minus-rail", label: "左侧光幕底端减去旁边围栏下沿", minuendId: "post-box-2:terminal", subtrahendId: "fence-1:near:post-box-2", valueNative: .065 },
  { id: "curtain-left-minus-right", label: "左右光幕离地差（左 − 右）", minuendId: "post-box-2:terminal", subtrahendId: "post-box-1:terminal", valueNative: .06 },
  { id: "rail-left-minus-right", label: "左右围栏下沿离地差（左 − 右）", minuendId: "fence-1:near:post-box-2", subtrahendId: "fence-0:near:post-box-1", valueNative: .085 },
];
const objects = [
  { id: "post-box-1", label: "黄色光幕立柱 1", kind: "yellow safety post" }, { id: "post-box-2", label: "黄色光幕立柱 2", kind: "yellow safety post" },
  { id: "fence-0", label: "安全围栏 1", kind: "safety fence" }, { id: "fence-1", label: "安全围栏 2", kind: "safety fence" },
  { id: "robot", label: "工业机器人", kind: "robot" }, { id: "v-guard-left", label: "左侧折弯护板", kind: "folded guard board" },
  { id: "floor", label: "地面参考面", kind: "floor" },
];
let factor: number | null = .5, unbound = new Set<string>();
const input = (): QueryInput => ({ objects, endpoints, differences, bends: [{ entityId: "v-guard-left", status: "measured", result: { value: 104.27 } }],
  bound: row => !unbound.has(row.id), format: native => factor == null ? `${native.toFixed(4)} native` : `${(native * factor * 100).toFixed(2)} cm`, revisionLabel: "主模型" });

let answer = answerSpatialQuery("右侧光幕离地多高？", input());
assert.equal(answer.status, "answered"); assert.deepEqual(answer.objects, ["post-box-1"]);
assert.match(answer.text, /右侧光幕底端离地 18\.00 cm/); assert.equal(answer.facts[0].sourceId, "post-box-1:terminal");
answer = answerSpatialQuery("左右光幕谁更高", input());
assert.equal(answer.status, "answered"); assert.match(answer.text, /^左侧光幕测点更高：左 − 右 = 3\.00 cm/);
assert.deepEqual(answer.objects.sort(), ["post-box-1", "post-box-2"]);
answer = answerSpatialQuery("左右围栏下沿相差多少", input());
assert.match(answer.text, /左侧围栏测点更高：左 − 右 = 4\.25 cm/);
answer = answerSpatialQuery("右侧光幕和围栏谁更高", input());
assert.equal(answer.facts.length, 1); assert.match(answer.text, /右侧光幕底端减去旁边围栏下沿：4\.50 cm/);
factor = 1; assert.match(answerSpatialQuery("右侧光幕离地多高", input()).text, /36\.00 cm/, "the current scale trial applies");
factor = null; assert.match(answerSpatialQuery("右侧光幕离地多高", input()).text, /0\.3600 native/, "unknown scale stays native");
factor = .5;
answer = answerSpatialQuery("机械臂离地多高", input());
assert.equal(answer.status, "missing"); assert.deepEqual(answer.objects, ["robot"]); assert.doesNotMatch(answer.text, /\d+\.\d+ cm/, "no invented length");
answer = answerSpatialQuery("左侧护板折弯角度", input());
assert.equal(answer.status, "answered"); assert.match(answer.text, /104\.3°/);
answer = answerSpatialQuery("找一下光幕", input());
assert.equal(answer.status, "answered"); assert.deepEqual(answer.objects, ["post-box-1", "post-box-2"]);
answer = answerSpatialQuery("光幕离地面多高", input());
assert.equal(answer.facts.length, 2, "'离地面' does not turn the question into the floor object");
assert.equal(answerSpatialQuery("灭火器在哪", input()).status, "unrecognized");
unbound = new Set(["post-box-2:terminal"]);
assert.equal(answerSpatialQuery("左右光幕谁更高", input()).status, "missing", "an endpoint off the displayed model is never compared");
assert.match(answerSpatialQuery("左侧光幕离地多高", input()).text, /未知（显示模型与测量模型不同）/);
unbound = new Set(["fence-0:near:post-box-1"]);
answer = answerSpatialQuery("右侧光幕和围栏谁更高", input());
assert.equal(answer.status, "missing", "a curtain-minus-rail pair with an unbound rail is never read"); assert.doesNotMatch(answer.text, /\d+\.\d+ cm/);
answer = answerSpatialQuery("光幕和围栏谁更高", input());
assert.equal(answer.facts.length, 1); assert.match(answer.text, /左侧光幕底端减去旁边围栏下沿：3\.25 cm.*未比较/);
unbound = new Set();
const shared = { ...input(), differences: differences.filter(row => row.id !== "rail-left-minus-right") };
assert.match(answerSpatialQuery("左右围栏下沿相差多少", shared).text, /没有左右两个不同围栏测点/, "one rail is never compared with itself");
// A lower-envelope rail hypothesis is a point of its own: described as such and never differenced.
const hypothesis = { ...input(), endpoints: endpoints.map(row => row.objectId === "fence-1" ? { ...row, label: "左侧光幕旁围栏下包络假设", railPart: "lower_envelope_hypothesis" } : row),
  differences: differences.filter(row => !["post-box-2:terminal-minus-rail", "rail-left-minus-right"].includes(row.id)),
  excluded: [{ id: "post-box-2:terminal-minus-rail", minuendId: "post-box-2:terminal", subtrahendId: "fence-1:near:post-box-2", reason: "旁边的围栏点是下包络假设" },
             { id: "rail-left-minus-right", minuendId: "fence-1:near:post-box-2", subtrahendId: "fence-0:near:post-box-1", reason: "测的不是同一部位" }] };
answer = answerSpatialQuery("左右围栏下沿相差多少", hypothesis);
assert.equal(answer.status, "missing"); assert.match(answer.text, /不比较左右围栏：测的不是同一部位/); assert.doesNotMatch(answer.text, /\d+\.\d+ cm/);
answer = answerSpatialQuery("左侧光幕和围栏谁更高", hypothesis);
assert.equal(answer.status, "missing"); assert.match(answer.text, /不比较：旁边的围栏点是下包络假设/);
answer = answerSpatialQuery("光幕和围栏谁更高", hypothesis);
assert.equal(answer.facts.length, 1); assert.match(answer.text, /右侧光幕底端减去旁边围栏下沿：4\.50 cm.*另一侧不比较：旁边的围栏点是下包络假设/);
assert.match(answerSpatialQuery("左侧围栏离地多高", hypothesis).text, /围栏下包络假设，不是下横梁下沿/);
console.log("PASS: structured spatial questions read the loaded revision's endpoints with the current scale; comparisons, missing facts, unknown scale and refused unlike comparisons stay explicit");
