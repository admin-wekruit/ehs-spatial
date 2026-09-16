// node --experimental-strip-types web/checks/scene-plan-reference.mjs
// Optional frozen-data scan: append <scene-document.json> <publication-catalog-directory>.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {planShapes,scenePlanOptions,cadReferenceImage,entityGeometryForLayer,activeModel,representationAvailable} from '../src/core.ts';
const frame='frame',plane=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],sha='a'.repeat(64);
const transform={coordinateFrameId:frame,position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]};
const observation=(id,imageId)=>({id,imageId,revision:1});
const observations=[observation('a-front','front'),observation('a-side','side'),observation('b-side','side')];
const surface=(observationId,x)=>{const o=observations.find(o=>o.id===observationId),id='rep-'+observationId;return{id,kind:'observed_surface',assetId:id,coordinateFrameId:frame,transform,placementState:'confirmed',sourceRefs:[{observationId,revision:1,imageId:o.imageId}],bounds:{min:[x,0,0],max:[x+2,2,1]},planProjection:{methodVersion:'indexed-mesh-triangle-union-v1',coordinateFrameId:frame,assetId:id,assetSha256:sha,imageId:o.imageId,observationId,observationRevision:1,transformSnapshot:transform,groundNormalSnapshot:[0,0,1],nativeToPlane:plane,polygons:[{exterior:[[x,0],[x+2,0],[x+2,1],[x+1,1],[x+1,2],[x,2],[x,0]],holes:[]}],lines:[]}}};
const a={id:'a',observationRefs:['a-front','a-side'],representations:[surface('a-front',0),surface('a-side',100)],cadReference:{status:'resolved',referenceImageId:'front',source:'explicit_reference_image',sourceRefs:[{observationId:'a-front',revision:1}]}},b={id:'b',observationRefs:['b-side'],representations:[surface('b-side',10)],cadReference:{status:'resolved',referenceImageId:'side',source:'single_source_image',sourceRefs:[{observationId:'b-side',revision:1}]}};
const document={schemaVersion:2,entities:[a,b],observations,assets:[...a.representations,...b.representations].map(r=>({id:r.assetId,sha256:sha})),coordinateFrames:[{id:frame,ground:{normal:[0,0,1]}}],cameras:[{id:'camera-front',imageId:'front',coordinateFrameId:frame},{id:'camera-side',imageId:'side',coordinateFrameId:frame}],geometryBindings:{front:{cameraId:'camera-front',geometrySolutionId:'solution'},side:{cameraId:'camera-side',geometrySolutionId:'solution'}},reportEvidence:{plan:{coordinateFrameId:frame,nativeToFloor:plane}}};
const before=structuredClone(document),options=scenePlanOptions(document,'observed_surface'),expected=planShapes(document,options);
assert.equal(expected.length,2,'The workcell plan retains objects absent from the viewed photo');
assert.deepEqual(expected[0].representationIds,['rep-a-front'],'The reference exposure wins over a different dynamic pose');
assert.equal(expected[0].max[0],2,'Different exposure states are not blindly unioned');
assert.equal(expected[0].polygons[0].exterior.length,7,'The source L contour is not replaced by box corners');
for(const imageId of ['front','side',null])assert.deepEqual(planShapes(document,{...options,imageId}),expected,'Changing the viewed photo never changes the persisted workcell plan');
assert.equal(cadReferenceImage(document,a),'front');assert.equal(cadReferenceImage(document,b),'side');
assert.equal(entityGeometryForLayer(a,{layer:'observed_surface',frameId:frame,imageId:'side',observations}).corners[0][0],100,'Photo and 3D keep the actual selected exposure state');
for(const change of [d=>d.entities[0].cadReference.status='unresolved',d=>d.entities[0].cadReference.referenceImageId='missing',d=>d.entities[0].cadReference.sourceRefs[0].revision=2,d=>d.entities[0].observationRefs=['a-side'],d=>d.geometryBindings.front=null]){
 const changed=structuredClone(document);change(changed);assert.equal(cadReferenceImage(changed,changed.entities[0]),null);assert.equal(planShapes(changed,scenePlanOptions(changed,'observed_surface')).length,1,'Invalid reference provenance cannot be replaced with another photo');
}
const mixed=structuredClone(document);mixed.cameras[1].coordinateFrameId='unregistered-frame';assert.equal(scenePlanOptions(mixed,'observed_surface').frameId,'','Unregistered coordinate frames cannot be silently combined');
const added=structuredClone(document);added.observations.push(observation('a-front-extra','front'));added.entities[0].observationRefs.push('a-front-extra');
assert.equal(cadReferenceImage(added,added.entities[0]),null,'An ownership change requires refreshed exact reference evidence');
added.entities[0].cadReference.sourceRefs.push({observationId:'a-front-extra',revision:1});
assert.deepEqual(planShapes(added,scenePlanOptions(added,'observed_surface')).map(({entity,...shape})=>shape),expected.map(({entity,...shape})=>shape),'Refreshing reference owners preserves the exact existing source contours without inventing a new mesh');
const historical=structuredClone(document);for(const entity of historical.entities)delete entity.cadReference;
const historicalBefore=JSON.stringify(historical),historicalOptions=scenePlanOptions(historical,'observed_surface',true,'side');
assert.equal(historicalOptions.scope,'photo','A snapshot without saved scene references retains its source-photo rendering contract');
assert.deepEqual(planShapes(historical,historicalOptions).map(shape=>shape.representationIds),[['rep-a-side'],['rep-b-side']]);
const invalidReferences=structuredClone(historical);for(const entity of invalidReferences.entities)entity.cadReference={status:'unresolved',referenceImageId:null};
assert.equal(scenePlanOptions(invalidReferences,'observed_surface',true,'side').scope,'scene','Explicit unresolved references never become historical photo selection');
assert.equal(planShapes(invalidReferences,scenePlanOptions(invalidReferences,'observed_surface',true,'side')).length,0);
invalidReferences.entities[0].cadReference=null;
assert.equal(scenePlanOptions(invalidReferences,'observed_surface',true,'side').scope,'scene','An explicitly malformed reference cannot opt into a different rendering contract');
assert.equal(planShapes(historical,scenePlanOptions(historical,'observed_surface',true,'missing')).length,0,'A historical photo must have its own verified geometry binding');
assert.equal(JSON.stringify(historical),historicalBefore);
assert.deepEqual(document,before,'Reference display never rewrites the immutable scene');
console.log('Scene CAD references: complete inventory across photo switches, distinct dynamic states, exact source provenance, honest frame failures and unchanged photo geometry passed');

if(process.argv[2]) {
 const scene=JSON.parse(await readFile(process.argv[2],'utf8')),before=JSON.stringify(scene),records=scene.entities.filter(entity=>!entity.sourceContext&&entity.visible!==false);
 const images=scene.assets.filter(asset=>asset.kind==='source_image').map(asset=>asset.id);
 for(const layer of ['observed_surface','model','point_cloud']) {
  const options=scenePlanOptions(scene,layer),expected=planShapes(scene,options);
  const count=layer==='model'?records.filter(entity=>{const model=activeModel(entity);return model&&representationAvailable(entity,model,options.frameId,true);}).length:records.length;
  assert.equal(expected.length,count,'Every eligible active model or observed record retains its own real contour');
  if(layer==='model')assert.ok(expected.every(shape=>shape.geometryKind==='model'),'Observed contours never fill missing model coverage');
  assert.ok(expected.every(shape=>shape.projectionSource==='mesh_projection'));
  for(const imageId of images)assert.deepEqual(planShapes(scene,{...options,imageId}),expected,'Every source-photo switch preserves actual full-scene contours');
  console.log(`${layer}: ${expected.length}/${records.length} entities; ${images.length} photo switches preserve exact geometry`);
 }
 assert.equal(JSON.stringify(scene),before);
}

if(process.argv[3]) {
 const cases=[['e38008a7-41e2-451a-aa10-2154962617cf',7,2],['cfb403f4-d930-4a61-80e3-ad4497db1a43',23,9],['f292baf7-4f27-4962-b7df-6e931c614413',28,9]];
 const imageId='6d69d290-2cea-410b-8c96-73b377ff5917';
 for(const [id,count,modelCount] of cases) {
  const bundle=JSON.parse(await readFile(`${process.argv[3]}/${id}/bundle.json`,'utf8'));
  const scene=bundle.responses[`/api/publications/${id}`].snapshot.revision.document,before=JSON.stringify(scene);
  const options=scenePlanOptions(scene,'observed_surface',true,imageId),shapes=planShapes(scene,options);
  assert.equal(shapes.length,count,`${id}: Preserve the frozen document's actual CAD evidence`);
  const models=planShapes(scene,scenePlanOptions(scene,'model',true,imageId));
  assert.equal(models.length,modelCount,`${id}: Model coverage counts only active models`);
  assert.ok(models.every(shape=>shape.geometryKind==='model'));
  assert.ok(shapes.some(shape=>shape.entity.id.startsWith('5c16a2fb')),'The selected mixed cart/guard record remains visible in its photo3 contour');
  if(options.scope==='scene')for(const image of scene.assets.filter(asset=>asset.kind==='source_image'))assert.deepEqual(planShapes(scene,scenePlanOptions(scene,'observed_surface',true,image.id)),shapes,'Fixed scene CAD is independent of the viewed photo');
  assert.equal(JSON.stringify(scene),before,'Historical report data remains immutable');
  console.log(`${id}: ${shapes.length}/28 observed contours; ${models.length}/28 model contours; ${options.scope} rendering contract`);
 }
}
