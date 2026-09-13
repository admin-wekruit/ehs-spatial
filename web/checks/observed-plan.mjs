// node --experimental-strip-types web/checks/observed-plan.mjs
import assert from 'node:assert/strict';
import {observedGeometry, planShapes} from '../src/core.ts';

const transform = {coordinateFrameId:'native',position:[10,20,30],quaternion:[0,0,Math.SQRT1_2,Math.SQRT1_2],scale:[2,3,4]};
const observed = {id:'observed',kind:'observed_surface',assetId:'mesh',coordinateFrameId:'native',transform,placementState:'confirmed',bounds:{min:[0,0,0],max:[1,2,3]}};
const entity = {id:'object',label:'Observed object',representations:[observed],currentModelTransform:{...transform,position:[900,900,900]},measurements:{}};
const document = {schemaVersion:1,entities:[entity],assets:[{id:'mesh'}],observations:[],cameras:[],annotations:[],coordinateFrames:[{id:'native',convention:'opencv',scale:{status:'uncalibrated',nativeToMeters:null},ground:{normal:[0,0,1]}}]};
const original = structuredClone(document);
const [shape] = planShapes(document);
const close = (actual, expected) => actual.forEach((value,index)=>assert.ok(Math.abs(value-expected[index])<1e-10,`${actual} != ${expected}`));
close(shape.min,[-22,4]); close(shape.max,[-20,10]);
assert.equal(shape.projectionSource,'observed_bounds');
assert.equal(shape.coordinateFrameId,'native');
assert.deepEqual(shape.representationIds,['observed']);
assert.equal(observedGeometry(entity,'native').corners.length,8);
assert.deepEqual(document,original,'Projection cannot rewrite a frozen source scene or apply model edits to observed geometry');

for (const change of [{placementState:'unconfirmed'}, {coordinateFrameId:'different'}, {transform:{...transform,coordinateFrameId:'different'}}, {assetId:null}, {bounds:{min:[2,0,0],max:[1,2,3]}}, {bounds:{min:[0,0,0],max:[Infinity,2,3]}}]) {
  assert.equal(planShapes({...document,entities:[{...entity,representations:[{...observed,...change}]}]}).length,0);
}
assert.equal(planShapes({...document,coordinateFrames:[{...document.coordinateFrames[0],ground:null}]}).length,0);
assert.equal(planShapes({...document,entities:[{...entity,sourceContext:true}]}).length,0,'Context does not become a selectable object footprint');
assert.equal(planShapes({...document,entities:[{...entity,visible:false}]}).length,0);
assert.equal(planShapes({...document,entities:[{...entity,representations:[],measurements:{dimensionsNative:{width:1,depth:2,height:3}}}]}).length,0,'Dimensions alone do not locate an object');
const cornersNative=observedGeometry(entity,'native').corners;
assert.equal(planShapes({...document,entities:[{...entity,representations:[],measurements:{coordinateFrameId:'native',basis:{cornersNative}}}]}).length,1);
assert.equal(planShapes({...document,entities:[{...entity,representations:[],measurements:{coordinateFrameId:'native',basis:{corners_native:cornersNative}}}]}).length,0,'Legacy field normalization belongs at import, never in the renderer');
const flat={...entity,representations:[{...observed,bounds:{min:[0,0,0],max:[1,2,0]}}]};
assert.equal(planShapes({...document,entities:[flat]}).length,1,'A measured planar surface remains projectable without invented thickness');

if (process.env.PANOPTES_TEST_API_BASE) {
  for (const [publicationId,expected] of [['2d77667b-5369-4d66-b5c4-d426e9833e18',68],['20962b70-3d10-4180-9d52-257deeb20ec1',81]]) {
    const response=await fetch(process.env.PANOPTES_TEST_API_BASE+'/api/publications/'+publicationId);
    assert.equal(response.status,200);
    const publication=await response.json(),doc=publication.snapshot.revision.document,before=JSON.stringify(doc),shapes=planShapes(doc);
    assert.equal(shapes.length,expected);
    assert.equal(JSON.stringify(doc),before);
    console.log(`Frozen publication ${publicationId}: ${shapes.length}/${doc.entities.filter(e=>!e.sourceContext).length} objects projectable`);
  }
}
console.log('Observed plan: real bounds/TRS, frame isolation, confirmed placement, planar surfaces, immutable input and explicit missing geometry passed');
