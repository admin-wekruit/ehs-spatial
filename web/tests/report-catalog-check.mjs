// node --experimental-strip-types tests/report-catalog-check.mjs
import assert from 'node:assert/strict';
import { groupPublications } from '../src/core.ts';

const old = { id: 'old', projectId: 'workcell', createdAt: '2026-09-11T10:00:00Z', title: 'Identical title', objectCount: 10 };
const latest = { ...old, id: 'latest', createdAt: '2026-09-12T10:00:00Z', objectCount: 68 };
const other = { ...old, id: 'test', projectId: 'other-workcell', createdAt: '2026-09-12T11:00:00Z', objectCount: 2 };
const input = [other, latest, old];
const groups = groupPublications(input);
assert.equal(groups.length, 2, 'one entry per project; equal titles must not merge projects');
assert.deepEqual(groups[0], [other], 'latest ordering retains the exact project identity');
assert.deepEqual(groups[1], [latest, old], 'older publications stay attached to their project');
assert.equal(groups[1][0].objectCount, 68, 'the primary report uses its own fixed content summary');
assert.deepEqual(input, [other, latest, old], 'grouping does not mutate the API response');
const microsecondLater = { ...latest, id: 'a', createdAt: '2026-09-12T10:00:00.000001+00:00' };
const sameSecondEarlier = { ...old, id: 'z', createdAt: '2026-09-12T10:00:00+00:00' };
assert.equal(groupPublications([microsecondLater, sameSecondEarlier])[0][0].id, 'a', 'retain server microsecond ordering; JS Date loses this precision');
assert.deepEqual(groupPublications([]), []);
console.log('Report catalog: exact project grouping, newest snapshot, preserved history, independent identical titles.');
