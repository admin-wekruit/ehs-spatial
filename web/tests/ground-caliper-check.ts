// Run: node --experimental-strip-types tests/ground-caliper-check.ts
import assert from 'node:assert/strict';
import { add, dot, scale, groundMeasurementLabel, measureGroundPoints, measurementLength } from '../src/viewer/native-math.ts';

const close = (actual: number, expected: number) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);
// Tilted floor through [0, 0, 5], given with a nonunit normal.
const floor = { normal: [0, 6, 8], offset: -40 }, n = [0, .6, .8], origin = [0, 0, 5];
const at = (x: number, y: number, h: number) => add(add(origin, [x, .8 * y, -.6 * y]), scale(n, h));
const a = at(0, 0, 2), b = at(3, 0, 6);
const one = measureGroundPoints('point_ground', [a], floor);
close(one.heightsNative[0], 2);
one.feetNative[0].forEach((v, i) => close(v, origin[i]));
close(dot(floor.normal, one.feetNative[0]) + floor.offset, 0);
close(measureGroundPoints('point_ground', [at(0, 0, -3)], floor).heightsNative[0], -3);
const unitFloor = measureGroundPoints('point_ground', [a], { normal: n, offset: -4 });
close(one.heightsNative[0], unitFloor.heightsNative[0]);
one.feetNative[0].forEach((v, i) => close(v, unitFloor.feetNative[0][i]));

const pair = measureGroundPoints('point_distance', [a, b], floor);
close(pair.distanceNative!, 5);
close(pair.heightDifferenceNative!, 4);
close(measureGroundPoints('point_distance', [b, a], floor).heightDifferenceNative!, -4);
close(measureGroundPoints('point_distance', [a, at(7, 9, 2)], floor).heightDifferenceNative!, 0);
const region = measureGroundPoints('region_ground', [a, at(1, 0, 3), at(0, 1, 2)], floor);
close(region.minHeightNative, 2); close(region.maxHeightNative, 3); close(region.inclinationDeg!, 45);
region.feetNative.forEach(foot => close(dot(floor.normal, foot) + floor.offset, 0));
close(measureGroundPoints('region_ground', [a, at(1, 0, 2), at(0, 1, 2)], floor).inclinationDeg!, 0);

const conditional = { nativeToMeters: .1, status: 'conditional', source: 'test-reference' };
const unknown = { ...conditional, nativeToMeters: null, status: 'unknown' };
assert.deepEqual(measurementLength(2, conditional), { value: 20, unit: 'cm' });
assert.deepEqual(measurementLength(-2, unknown), { value: -2, unit: 'native' });
assert.equal(groundMeasurementLabel(pair, conditional), 'd 50 cm · Δh 40 cm');
assert.equal(groundMeasurementLabel(pair, { ...conditional, nativeToMeters: .2 }), 'd 100 cm · Δh 80 cm');
assert.equal(groundMeasurementLabel(one, unknown), 'h 2 native');
assert.equal(groundMeasurementLabel(measureGroundPoints('point_ground', [[0, 0, 1e-8]], {normal: [0, 0, 1], offset: 0}), conditional), 'h 0 cm');
assert.equal(groundMeasurementLabel(region, conditional), 'h 20 cm … 30 cm · 45.0°');
assert.deepEqual(pair.pointsNative, [a, b]); // Display conversion never mutates native evidence.

assert.throws(() => measureGroundPoints('point_distance', [a, a], floor), /coincident/);
assert.throws(() => measureGroundPoints('region_ground', [a, b, a], floor), /coincident/);
assert.throws(() => measureGroundPoints('region_ground', [a, at(1, 0, 2), at(2, 0, 2)], floor), /collinear/);
for (const point of [[NaN, 0, 0], [Infinity, 1, 2], [1, 2]]) assert.throws(() => measureGroundPoints('point_ground', [point], floor), /points_invalid/);
for (const ground of [{ normal: [0, 0, 0], offset: 1 }, { normal: [0, 0, 1], offset: NaN }, { normal: [0, 0, 1] }, { normal: [1, 2], offset: 0 }]) assert.throws(() => measureGroundPoints('point_ground', [a], ground), /ground_missing/);
for (const nativeToMeters of [0, -1, NaN, Infinity]) assert.throws(() => measurementLength(2, { ...conditional, nativeToMeters }), /scale_invalid/);
assert.throws(() => measureGroundPoints('point_distance', [a], floor), /points_invalid/);
assert.throws(() => measurementLength(Infinity, conditional), /points_invalid/);
console.log('Ground caliper checks passed: normalized floor, signed heights, triangle range, scale and invalid inputs.');
