// node --experimental-strip-types web/checks/observed-plan.mjs
import assert from 'node:assert/strict';
import {entityGeometryForLayer as geometry, planShapes as projectPlan} from '../src/core.ts';
import {representationPass, selectionGeometry} from '../src/viewer/native-viewer.ts';

const observation={id:'observation',imageId:'photo',revision:1}, baseOptions={imageId:'photo',observations:[observation]};
const planShapes=(document,options={})=>projectPlan(document,{...baseOptions,...options});
const entityGeometryForLayer=(entity,options)=>geometry(entity,{...baseOptions,...options});
const observedGeometry=(entity,frameId)=>entityGeometryForLayer(entity,{layer:'observed_surface',frameId});
const transform = {coordinateFrameId:'native',position:[10,20,30],quaternion:[0,0,Math.SQRT1_2,Math.SQRT1_2],scale:[2,3,4]};
const observed = {id:'observed',sourceRefs:[{observationId:'observation',revision:1}],kind:'observed_surface',assetId:'mesh',coordinateFrameId:'native',transform,placementState:'confirmed',bounds:{min:[0,0,0],max:[1,2,3]}};
const entity = {observationRefs:['observation'],measurementSelections:{basis:'basis'},measurementEvidence:[{id:'basis',observationRefs:['observation']}],id:'object',label:'Observed object',representations:[observed],currentModelTransform:{...transform,position:[900,900,900]},measurements:{}};
const document = {schemaVersion:1,entities:[entity],assets:[{id:'mesh',sha256:'a'.repeat(64)}],observations:[observation],cameras:[],annotations:[],coordinateFrames:[{id:'native',convention:'opencv',scale:{status:'uncalibrated',nativeToMeters:null},ground:{normal:[0,0,1]}}]};
const defaultPlane=[[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]];
const projectionFor=(rep,plane,points)=>({methodVersion:'indexed-mesh-triangle-union-v1',coordinateFrameId:'native',assetId:rep.assetId,assetSha256:'a'.repeat(64),imageId:'photo',observationId:'observation',observationRevision:1,transformSnapshot:structuredClone(rep.transform),groundNormalSnapshot:[0,0,1],nativeToPlane:plane,polygons:[{exterior:[...points,points[0]],holes:[]}],lines:[]});
observed.planProjection=projectionFor(observed,defaultPlane,[[-22,4],[-20,4],[-20,10],[-22,10]]);
const original = structuredClone(document);
const [shape] = planShapes(document);
const close = (actual, expected) => actual.forEach((value,index)=>assert.ok(Math.abs(value-expected[index])<1e-10,`${actual} != ${expected}`));
close(shape.min,[-22,4]); close(shape.max,[-20,10]);
assert.equal(shape.projectionSource,'mesh_projection');
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
assert.equal(planShapes({...document,entities:[{...entity,representations:[],measurements:{coordinateFrameId:'native',basis:{cornersNative}}}]}).length,0);
assert.equal(planShapes({...document,entities:[{...entity,representations:[],measurements:{coordinateFrameId:'native',basis:{corners_native:cornersNative}}}]}).length,0,'Legacy field normalization belongs at import, never in the renderer');
const flat={...entity,representations:[{...observed,bounds:{min:[0,0,0],max:[1,2,0]}}]};
assert.equal(planShapes({...document,entities:[flat]}).length,1,'A measured planar surface remains projectable without invented thickness');

const modelTransform={coordinateFrameId:'native',position:[100,200,300],quaternion:[0,0,Math.SQRT1_2,Math.SQRT1_2],scale:[1,2,1]};
const model={...observed,planProjection:undefined,id:'model',kind:'generated_mesh',assetId:'model-asset',transform:modelTransform,bounds:{min:[0,0,0],max:[2,3,4]}};
const combined={...entity,activeModelRepresentationId:'model',currentModelTransform:modelTransform,representations:[model,observed]};
const projection={coordinateFrameId:'native',nativeToFloor:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]};
observed.planProjection=projectionFor(observed,projection.nativeToFloor,[[4,20],[10,20],[10,22],[4,22]]);
model.planProjection=projectionFor(model,projection.nativeToFloor,[[94,200],[100,200],[100,202],[94,202]]);
const layered={...document,entities:[combined],assets:[...document.assets,{id:'model-asset',sha256:'a'.repeat(64)}],reportEvidence:{plan:projection}};
for(const layer of ['model','observed_surface','point_cloud']) {
  const options={layer,frameId:'native',showCandidates:true};
  const shared=entityGeometryForLayer(combined,options),[plan]=planShapes(layered,options);
  const flags={...baseOptions,showCandidates:true,point_cloud:layer==='point_cloud',observed_surface:layer!=='point_cloud',generated_mesh:layer==='model',primitive:layer==='model'};
  const selected=selectionGeometry(layered,combined,'native',flags);
  assert.deepEqual(selected.corners,shared.corners,'Native 3D selection and source-photo helper choose identical transformed corners');
  assert.equal(shared.geometryKind,layer==='model'?'model':'observed');
  assert.deepEqual(plan.representationIds,shared.representationIds);
  close(plan.min,[0,1].map(k=>Math.min(...shared.corners.map(p=>p[k]))));
  close(plan.max,[0,1].map(k=>Math.max(...shared.corners.map(p=>p[k]))));
}
assert.notDeepEqual(planShapes(layered,{layer:'model'})[0].polygons,planShapes(layered,{layer:'observed_surface'})[0].polygons,'Changing layers must change the real plan geometry');
for(const change of [{sourceValidity:'stale'},{placementState:'unconfirmed',placementReason:'insufficient_observed_depth'},{transform:{...modelTransform,coordinateFrameId:'other'}}]) {
  const unavailable={...combined,currentModelTransform:null,representations:[{...model,...change}]};
  assert.equal(entityGeometryForLayer(unavailable,{layer:'model',frameId:'native'}),null);
  assert.equal(planShapes({...layered,entities:[unavailable]}).length,0,'No unplaced, stale or foreign-frame model appears in CAD');
  assert.equal(representationPass(unavailable,unavailable.representations[0],'native',{showCandidates:true,generated_mesh:true}).visible,false);
}
const modelOnly={...combined,representations:[model]};
for(const layer of ['observed_surface','point_cloud']) assert.equal(entityGeometryForLayer(modelOnly,{layer,frameId:'native'}),null,'Missing observed evidence never borrows a model');
const cloud={...observed,id:'cloud',kind:'point_cloud',bounds:{min:[0,0,0],max:[1,1,1]}};
assert.deepEqual(entityGeometryForLayer({...combined,representations:[...combined.representations,cloud]},{layer:'point_cloud',frameId:'native'}).representationIds,['cloud']);
const twoFrames={...layered,coordinateFrames:[{...document.coordinateFrames[0],id:'other'},document.coordinateFrames[0]]};
assert.equal(planShapes(twoFrames,{layer:'model',frameId:'native'}).length,1,'The caller frame wins over the first grounded frame');
assert.equal(planShapes(twoFrames,{layer:'model',frameId:'unregistered'}).length,0);
const saved={coordinateFrameId:'native',nativeToPlane:projection.nativeToFloor,points:[[999,999],[1000,999],[999,1000]],representationSnapshot:structuredClone(combined.representations),modelTransformSnapshot:structuredClone(modelTransform)};
const ambiguous={...combined,measurements:{projectedHull:saved}};
for(const layer of ['model','observed_surface','point_cloud']) assert.notEqual(planShapes({...layered,entities:[ambiguous]},{layer})[0].projectionSource,'saved_hull','A combined snapshot cannot prove which layer produced a historical hull');
const explicit={...ambiguous,representations:combined.representations.map(rep=>({...rep,planProjection:undefined})),measurements:{projectedHull:{...saved,geometryKind:'model',representationIds:['model']}}};
explicit.measurements.projectedHull.representationSnapshot=structuredClone(explicit.representations);
assert.equal(planShapes({...layered,entities:[explicit]},{layer:'model'})[0].projectionSource,'saved_hull');
assert.notEqual(planShapes({...layered,entities:[explicit]},{layer:'observed_surface'})[0]?.projectionSource,'saved_hull');
const contradictory={...modelOnly,measurements:{projectedHull:{...saved,representationSnapshot:[model],geometryKind:'observed',representationIds:['model']}}};
assert.notEqual(planShapes({...layered,entities:[contradictory]})[0]?.projectionSource,'saved_hull','Explicit incompatible provenance cannot be replaced by an inferred source');
const sparse={measurementSelections:{observedBounds:'bounds'},measurementEvidence:[{id:'bounds',observationRefs:['observation']}],id:'sparse-button',representations:[{...observed,sourceValidity:'stale'}],measurements:{coordinateFrameId:'native',observedBounds:{coordinateFrameId:'native',min:[1,2,3],max:[2,3,4],source:'observed_measurement',sourceRefs:[{observationId:'updated',revision:2}]}}};
for(const layer of ['model','observed_surface','point_cloud']) {
  const geometry=entityGeometryForLayer(sparse,{layer,frameId:'native'});
  assert.equal(geometry.geometryKind,'observed_measurement','Fresh sparse points remain locatable even when the old mesh is stale and no new faces can be formed');
  assert.equal(geometry.corners.length,8);
  assert.equal(planShapes({...document,entities:[sparse]},{layer}).length,0, 'Sparse point bounds remain selectable geometry, not an invented surface contour');
}
for(const change of [{sourceValidity:'stale'},{coordinateFrameId:'other'},{sourceRefs:[]},{source:'generated_mesh'}]) {
  const missing={...sparse,measurements:{...sparse.measurements,observedBounds:{...sparse.measurements.observedBounds,...change}}};
  assert.equal(entityGeometryForLayer(missing,{layer:'point_cloud',frameId:'native'}),null);
}
assert.equal(entityGeometryForLayer({...sparse,measurements:{}},{layer:'point_cloud',frameId:'native'}),null,'Resegmentation clears the old measurement before attempting a new surface');

console.log('Observed plan: true mesh contours, frame/layer isolation, planar surfaces, immutable input and bounds-only geometry withheld from CAD passed');
