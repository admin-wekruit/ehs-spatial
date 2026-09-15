// node --experimental-strip-types web/checks/scene-plan-reference.mjs
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {planShapes,scenePlanOptions,cadReferenceImage,entityGeometryForLayer} from '../src/core.ts';
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
assert.deepEqual(document,before,'Reference display never rewrites the immutable scene');
console.log('Scene CAD references: complete inventory across photo switches, distinct dynamic states, exact source provenance, honest frame failures and unchanged photo geometry passed');

if(process.argv[2]) {
 const scene=JSON.parse(await readFile(process.argv[2],'utf8')),before=JSON.stringify(scene),records=scene.entities.filter(entity=>!entity.sourceContext&&entity.visible!==false);
 const images=scene.assets.filter(asset=>asset.kind==='source_image').map(asset=>asset.id);
 for(const layer of ['observed_surface','model','point_cloud']) {
  const options=scenePlanOptions(scene,layer),expected=planShapes(scene,options);
  assert.equal(expected.length,records.length,'Every record in this repaired scene must have a real reference contour');
  assert.ok(expected.every(shape=>shape.projectionSource==='mesh_projection'));
  for(const imageId of images)assert.deepEqual(planShapes(scene,{...options,imageId}),expected,'Every source-photo switch preserves actual full-scene contours');
  console.log(`${layer}: ${expected.length}/${records.length} entities; ${images.length} photo switches preserve exact geometry`);
 }
 assert.equal(JSON.stringify(scene),before);
}
