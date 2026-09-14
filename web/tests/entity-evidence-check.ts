// Run: node --experimental-strip-types tests/entity-evidence-check.ts
import assert from "node:assert/strict";
import { entityEvidenceStatus } from "../src/scene-semantics.ts";
import type { Entity, SceneDocument } from "../src/types.ts";

const document = { observations: [{id:"a",imageId:"photo-a"},{id:"b",imageId:"photo-b"}], cameras:[] } as unknown as SceneDocument;
const entity = {id:"object",observationRefs:["a"],associationState:"association_pending",representations:[]} as Entity;
assert.deepEqual(entityEvidenceStatus(document,entity),{photoKey:"entityPhotoLocated",identityKey:"entityAssociationNotChecked",photoCount:1});
assert.equal(entityEvidenceStatus({...document,observations:[document.observations[0]]},entity).identityKey,"entitySingleView");
assert.equal(entityEvidenceStatus(document,{...entity,observationRefs:["a","b"],associationState:"confirmed"}).identityKey,"entityCrossViewConfirmed");
assert.equal(entityEvidenceStatus(document,{...entity,observationRefs:["a","a"]}).photoCount,1,"count photos rather than repeated observation references");
for(const [status,key] of Object.entries({mask_missing:"entityAssociationNeedsMask",geometry_missing:"entityAssociationNeedsGeometry",insufficient_support:"entityAssociationNeedsSupport",competing_candidates:"entityAssociationAmbiguous",no_supported_match:"entityAssociationNoMatch"}))
  assert.equal(entityEvidenceStatus(document,{...entity,associationEvidence:{status}}).identityKey,key);
assert.deepEqual(entityEvidenceStatus(document,{...entity,observationRefs:[]}),{photoKey:"entityNeedsPhoto",identityKey:null,photoCount:0});
const model: Entity = {...entity,observationRefs:[],representations:[{id:"box",kind:"primitive",coordinateFrameId:"frame",placementState:"unconfirmed",
  transform:{coordinateFrameId:"frame",position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]}}]};
assert.deepEqual(entityEvidenceStatus(document,model),{photoKey:"entityModelWithoutPhoto",identityKey:null,photoCount:0});
assert.equal(entityEvidenceStatus(document,{...entity,observationRefs:["missing"]}).photoKey,"entityNeedsPhoto","a dangling reference is not photo evidence");
console.log("Entity evidence: located objects, explicit multiview, single photo, missing mesh, model-only, and distinct association reasons passed.");
