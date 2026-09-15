// node --experimental-strip-types web/checks/publication-entry.mjs
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {currentPublicationURL, publicationReaderURL} from '../src/core.ts';
const publication = (id, projectId='workcell', branchId='reconstruction') =>
  ({id, projectId, snapshot:{revision:{branchId}}});
const old=publication('old'), current=publication('current');
const entry='https://example.com/base/app.html#/reports/old?image=photo&object=entity&observation=observation';
const destination=new URL(currentPublicationURL(old,current,entry));
assert.equal(destination.pathname,'/base/app.html');
assert.equal(destination.searchParams.get('report'),'current');
const [route,query]=destination.hash.split('?'), params=new URLSearchParams(query);
assert.equal(route,'#/reports/current');
assert.equal(params.get('image'),'photo');
assert.equal(params.get('object'),'entity');
assert.equal(params.get('observation'),'observation');
assert.equal(params.get('fromReport'),'old');
assert.equal(currentPublicationURL(old,current,entry+'&snapshot=1'),null,'Explicit historical snapshot cannot redirect');
assert.equal(currentPublicationURL(current,current,destination.href),null,'No redirect loop');
assert.equal(currentPublicationURL(old,publication('other','another-workcell'),entry),null,'Never cross projects');
assert.equal(currentPublicationURL(old,publication('planning','workcell','planning'),entry),null,'Never cross planning branches');
const archived=publicationReaderURL(1,entry+'&snapshot=1');
assert.equal(new URLSearchParams(new URL(archived).hash.split('?')[1]).get('snapshot'),'1','Pinned reader preserves explicit snapshot intent');
const report=await readFile(new URL('../src/WorkcellReport.tsx',import.meta.url),'utf8');
assert.ok(report.indexOf('currentPublicationURL(pub, candidate, location.href)')<report.indexOf('publicationReaderURL(revision.document.schemaVersion, location.href)'), 'Resolve daily report before selecting a historical reader');
assert.match(report,/if \(!live\) return;\s*if \(destination\)/,'An obsolete route cannot redirect a newly opened report');
assert.match(report,/href=\{"#\/reports\/" \+ p.id \+ "\?snapshot=1"\}/,'History deliberately opens immutable snapshots');
console.log('Publication entry: latest same-branch report, preserved selection, explicit immutable history, reader intent, and stale navigation guard passed');
