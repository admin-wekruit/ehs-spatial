// node --experimental-strip-types web/checks/identity-review.mjs
import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import * as sceneSemantics from '../src/scene-semantics.ts';
import * as identityCopy from '../src/identity-messages.ts';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import ts from 'typescript';
import * as core from '../src/core.ts';
import {representationPass} from '../src/viewer/native-viewer.ts';

const transform=x=>({coordinateFrameId:'frame',position:[x,0,0],quaternion:[0,0,0,1],scale:[1,1,1]});
const rep=(id,x)=>({id,kind:'primitive',coordinateFrameId:'frame',transform:transform(x),placementState:'confirmed',primitive:{kind:'box',dimensions:[2,2,2]}});
const models=[rep('model-a',1),rep('model-b',20)], observations=[{id:'oa',imageId:'photo-a',revision:3},{id:'ob',imageId:'photo-b',revision:4}];
const entity={id:'a',label:'Same-label',associationState:'association_pending',activeModelRepresentationId:'model-a',currentModelTransform:transform(1),representations:models,observationRefs:['oa'],measurements:{},measurementEvidence:[],measurementSelections:{}};
const other={...entity,id:'b',activeModelRepresentationId:null,currentModelTransform:null,representations:[],observationRefs:['ob']};
const document={schemaVersion:2,target:'scene',entities:[entity,other],observations,assets:[{id:'photo-a',kind:'source_image'},{id:'photo-b',kind:'source_image'}],annotations:[],identityDecisions:[],coordinateFrames:[{id:'frame',ground:{normal:[0,0,1]}}],
 cameras:[{id:'old',imageId:'photo-a',coordinateFrameId:'frame'},{id:'current',imageId:'photo-a',coordinateFrameId:'frame'}],geometryBindings:{'photo-a':{geometrySolutionId:'g',cameraId:'current'},'photo-b':null}};
assert.equal(core.cameraForImage(document,'photo-a').id,'current','The explicit solution binding wins over camera array order');
assert.equal(core.cameraForImage(document,'photo-b'),null);assert.equal(core.cameraForImage({...document,geometryBindings:{}},'photo-a'),null,'Missing binding never guesses a camera');
assert.equal(core.modelGeometry(entity).corners[0][0],0);
for(const status of [{sourceValidity:"stale"},{placementState:"unconfirmed",placementReason:"no_depth"}]) assert.equal(core.editableTransform({...entity,representations:[{...models[0],...status}]}),null,"Unknown/stale placement cannot appear as editable current pose");
assert.equal(representationPass(entity,models[1],'frame',{primitive:true}).visible,false,'Non-active alternatives never render or pick');
const moved=core.previewOperations(document,[{type:'setTransform',entityId:'a',transform:transform(5)},{type:'setMaterial',entityId:'a',material:{color:'#ff0000'}}]);
assert.deepEqual(moved.entities[0].representations[1],models[1],'Editing the current model preserves alternative source pose/material');
assert.equal(moved.entities[0].representations[0].transform.position[0],5);
assert.equal(moved.entities[0].representations[0].material.color,'#ff0000');
const certified=structuredClone(document),certificate={type:'manual_assertion',transformSha256:'recorded-pose'},originalProjection={source:'saved-contour'};
certified.entities[0].representations[0].placementSource=certificate;
certified.entities[0].representations[0].bounds={min:[-1,-1,-1],max:[1,1,1]};
certified.entities[0].representations[0].planProjection=originalProjection;
const certificateBefore=structuredClone(certified);
const noop=core.previewOperations(certified,[{type:'setTransform',entityId:'a',transform:transform(1)},{type:'setPrimitive',entityId:'a',primitive:{kind:'box',dimensions:[2,2,2]}}]);
assert.deepEqual(noop,certified,'Identical primitive and effective pose preserve bounds, saved contour and confirmation evidence');
const changedPose=core.previewOperations(certified,[{type:'setTransform',entityId:'a',transform:transform(9)}]);
assert.equal(changedPose.entities[0].representations[0].placementState,'unconfirmed');
assert.equal(changedPose.entities[0].representations[0].placementReason,'requires_alignment_confirmation');
assert.equal(changedPose.entities[0].representations[0].placementSource,undefined,'A moved preview cannot retain confirmation for the previous pose');
assert.deepEqual(changedPose.entities[0].representations[1],models[1],'Inactive alternatives retain their original placement');
const fallbackPose=structuredClone(certified);fallbackPose.entities[0].currentModelTransform=null;
assert.deepEqual(core.previewOperations(fallbackPose,[{type:'setTransform',entityId:'a',transform:transform(1)}]).entities[0].representations[0].placementSource,certificate,'The representation pose supplies a no-op comparison when no override exists');
const resized=core.previewOperations(certified,[{type:'setPrimitive',entityId:'a',primitive:{kind:'box',dimensions:[6,4,2]},transform:transform(3)}]);
const resizedRep=resized.entities[0].representations[0];
assert.deepEqual(resizedRep.bounds,{min:[-3,-2,-1],max:[3,2,1]},'Resizing replaces imported bounds with actual new primitive dimensions');
assert.equal(core.modelGeometry(resized.entities[0]).corners[0][0],0,'The new primitive pose and new bounds reach scene geometry together');
assert.equal(resizedRep.planProjection,undefined,'A changed primitive cannot retain its old CAD contour');
assert.equal(resizedRep.placementSource,undefined);assert.equal(resizedRep.placementState,'unconfirmed');
const sameSourceFrame=structuredClone(certified);sameSourceFrame.coordinateFrames[0].source='manual_assertion';
assert.equal(core.previewOperations(sameSourceFrame,[{type:'setPrimitive',entityId:'a',primitive:{kind:'cylinder',radius:2,height:5,segments:8}}]).entities[0].representations[0].placementState,'unconfirmed','A manually declared coordinate frame does not confirm a changed shape');
assert.throws(()=>core.previewOperations(certified,[{type:'setPrimitive',entityId:'a',representationId:'model-b',primitive:{kind:'box',dimensions:[3,3,3]}}]),/active_model_selection_required/);
for(const operation of [{type:'setTransform',entityId:'a',transform:{...transform(1),coordinateFrameId:'missing'}},{type:'setPrimitive',entityId:'a',primitive:{kind:'cylinder',radius:1,height:2,segments:7}}])assert.throws(()=>core.previewOperations(certified,[operation]));
assert.deepEqual(core.previewOperations(changedPose,[{type:'confirmPlacement',entityId:'a',representationId:'model-a'}]),changedPose,'Confirmation stays pending until a saved server response supplies evidence');
assert.throws(()=>core.previewOperations(changedPose,[{type:'confirmPlacement',entityId:'a',representationId:'model-b'}]),/active_model_selection_required/);
const noPlacement=structuredClone(changedPose);noPlacement.entities[0].representations[0].placementReason='insufficient_observed_depth';
assert.throws(()=>core.previewOperations(noPlacement,[{type:'confirmPlacement',entityId:'a',representationId:'model-a'}]),/model_placement_required/);
const legacy=structuredClone(certified);legacy.schemaVersion=1;delete legacy.entities[0].activeModelRepresentationId;
const legacyMoved=core.previewOperations(legacy,[{type:'setTransform',entityId:'a',transform:transform(8)}]);
assert.ok(legacyMoved.entities[0].representations.every(rep=>rep.placementState==='unconfirmed'&&rep.transform.position[0]===8),'Schema 1 updates all modeled representations consistently');
const legacyResize=core.previewOperations(legacy,[{type:'setPrimitive',entityId:'a',representationId:'model-b',primitive:{kind:'box',dimensions:[8,2,2]}}]);
assert.deepEqual(legacyResize.entities[0].representations[0],legacy.entities[0].representations[0]);assert.equal(legacyResize.entities[0].representations[1].bounds.max[0],4,'Schema 1 primitive edits respect the explicit representation');
assert.deepEqual(certified,certificateBefore,'All edit previews preserve the saved source document');
const switched=core.previewOperations(moved,[{type:'setActiveModelRepresentation',entityId:'a',representationId:'model-b'}]);
assert.equal(core.modelGeometry(switched.entities[0]).corners[0][0],19,'Changing the active model restores its own pose');
assert.equal(core.planShapes(switched).length,0,'An active primitive without a computed contour retains 3D geometry without inventing CAD');
assert.equal(core.modelGeometry(core.previewOperations(switched,[{type:'setActiveModelRepresentation',entityId:'a',representationId:null}]).entities[0]),null);
const surface=(id,observationId,x)=>({...rep(id,x),kind:'observed_surface',assetId:id,bounds:{min:[0,0,0],max:[1,1,1]},sourceRefs:[{observationId}]});
const states={...entity,observationRefs:['oa','ob'],representations:[surface('state-a','oa',1),surface('state-b','ob',20)],activeModelRepresentationId:null};
for(const [imageId,x] of [['photo-a',1],['photo-b',20]]) {
 const options={layer:'observed_surface',frameId:'frame',imageId,observations};
 assert.equal(core.entityGeometryForLayer(states,options).corners[0][0],x,'Each photo uses its own observed state, never a fused volume');
 assert.equal(core.planShapes({...document,entities:[states]},options).length,0,'Observed bounds without a mesh projection cannot become CAD');
}
assert.equal(core.entityGeometryForLayer(states,{layer:'observed_surface',frameId:'frame',observations}),null,'Unscoped source states do not invent a current pose');
const importedObservations=observations.map((observation,i)=>({...observation,sourceRefs:[{assetId:'frozen-records',sourceRecordId:`record-${i}`}]}));
const importedStates={...states,representations:states.representations.map((representation,i)=>({...representation,sourceRefs:[{assetId:'frozen-records',sourceRecordId:`record-${i}`}]}))};
for(const [imageId,x] of [['photo-a',1],['photo-b',20]]) {
 const options={layer:'observed_surface',frameId:'frame',imageId,observations:importedObservations};
 assert.equal(core.entityGeometryForLayer(importedStates,options).corners[0][0],x,'A frozen asset+record source link resolves the exact current photo');
 assert.equal(core.planShapes({...document,observations:importedObservations,entities:[importedStates]},options).length,0,'Source-linked bounds alone do not masquerade as mesh contours');
 assert.equal(core.entityGeometryForLayer(importedStates,{...options,frameId:'unrelated-frame'}),null,'A source match never authorizes a different coordinate frame');
}
for(const sourceRefs of [[{assetId:'wrong-asset',sourceRecordId:'record-0'}],[{assetId:'frozen-records',sourceRecordId:'wrong-record'}],[{assetId:'frozen-records'}],[{sourceRecordId:'record-0'}]]) {
 assert.equal(core.representationInPhoto(importedStates,{...importedStates.representations[0],sourceRefs},'photo-a',importedObservations),false,'Both frozen asset and source record must match');
}
assert.equal(core.representationInPhoto({...importedStates,observationRefs:['ob']},importedStates.representations[0],'photo-a',importedObservations),false,'Retired ownership cannot keep an observed state attached after splitting');
const samePhotoObservations=importedObservations.map(observation=>({...observation,imageId:'photo-a'}));
assert.equal(core.entityGeometryForLayer(importedStates,{layer:'observed_surface',frameId:'frame',imageId:'photo-a',observations:samePhotoObservations}).representationIds.length,2,'Multiple explicitly owned observations in the same photo remain accessible');
assert.equal(core.publicationReaderURL(1,'https://example.com/base/app.html?v=2#/reports/old?object=a'),'https://example.com/base/readers/v1/app.html?v=2#/reports/old?object=a');
assert.equal(core.publicationReaderURL(1,'https://example.com/app.html#/reports/old'),'https://example.com/readers/v1/app.html#/reports/old','Root hosting uses the same relative archive path');
assert.equal(core.publicationReaderURL(2,'https://example.com/base/app.html#/reports/new'),null);
assert.equal(core.publicationReaderURL(1,'https://example.com/base/readers/v1/app.html#/reports/old'),null,'Frozen reader routing cannot loop');

// Drive the real review component and captured Apply context; no network/model is mocked as production output.
const require=createRequire(import.meta.url),slots=[],effects=[];let cursor=0,serial=0,tree,props;
const hooks={useState(initial){const i=cursor++;if(!Object.hasOwn(slots,i))slots[i]=initial;return [slots[i],value=>{slots[i]=typeof value==='function'?value(slots[i]):value;}];},useEffect(effect,deps){const i=cursor++,old=slots[i];if(!old||deps.some((v,k)=>v!==old[k]))effects.push(()=>{slots[i]=deps;effect();});}};
const PhotoView=()=>null,ErrorNotice=()=>null;
class ApiError extends Error {constructor(status,code){super(code);this.status=status;}}
const module={exports:{}},source=await readFile(new URL('../src/IdentityReview.tsx',import.meta.url),'utf8');
const code=ts.transpileModule(source.replace('function IdentitySplit(', 'export function IdentitySplit('),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX}}).outputText;
new Function('require','module','exports',code)(name=>name==='react'?hooks:name==='./core'?core:name==='./scene-semantics'?sceneSemantics:name==='./identity-messages'?identityCopy:name==='./api'?{ApiError,id:()=>`decision-${++serial}`,request:async()=>({items:[]})}:name==='./PhotoView'?{PhotoView}:name==='./App'?{ErrorNotice}:name==='./i18n'?{useI18n:()=>({t:key=>key})}:require(name),module,module.exports);
const {IdentityReview,identityOperations}=module.exports;
const revision={id:'revision',document},applied=[],suggestions=[];
props={revision,entityId:'a',canWrite:true,onSelect(){},onApply:async(operations,context)=>{applied.push({operations,context});return true;},onSuggest:suggestion=>suggestions.push(suggestion),onAgent(){}};
const nodes=node=>React.isValidElement(node)?[node,...React.Children.toArray(node.props.children).flatMap(nodes)]:[];
function render(){for(let i=0;i<3;i++){cursor=0;tree=IdentityReview(props);for(const effect of effects.splice(0))effect();}return nodes(tree);}
function choosePair(){render().find(n=>n.type==='select').props.onChange({target:{value:'b'}});render();}
choosePair();
assert.ok(render().some(n=>n.type==='p'&&n.props.children==='entityAssociationNotChecked'),'The review renders a translated identity label, not a raw backend status');
assert.equal(render().filter(n=>typeof n.type==='function'&&n.type.name==='IdentityPhoto').length,2);
render().find(n=>n.type==='button'&&n.props.children==='identitySame').props.onClick();
render().find(n=>n.type==='textarea').props.onChange({target:{value:'The same control seen in two photographs'}});
render().find(n=>n.type==='button'&&n.props.children==='identityPreview').props.onClick();
await render().find(n=>n.type==='button'&&n.props.children==='identityApply').props.onClick();
assert.equal(applied.length,1);assert.equal(applied[0].context.baseRevisionId,'revision');
assert.deepEqual(applied[0].operations.map(op=>op.type),['recordIdentityDecision','mergeEntities'],'Same decision and merge are one CAS edit batch');
assert.deepEqual(applied[0].operations[0].decision.observationGroups,[['oa'],['ob']]);
assert.deepEqual(applied[0].operations[0].decision.evidenceRefs,[{kind:'observation',observationId:'oa',observationRevision:3},{kind:'observation',observationId:'ob',observationRevision:4}]);
for(const decision of ['different','undecided']) {
 assert.equal(identityOperations(revision,[entity,other],decision,'reason','a',null,'decision',null).length,1,'Non-merge decisions do not modify memberships');
 render().find(n=>n.type==='button'&&n.props.children===identityCopy.identityDecisionKeys[decision]).props.onClick();
 render().find(n=>n.type==='button'&&n.props.children==='identityPreview').props.onClick();
 const preview=render().find(n=>n.props.className==='identity-preview'),previewText=renderToStaticMarkup(preview);
 assert.ok(previewText.includes(decision==='different'?'identityDifferentPreviewNote':'identityUndecidedPreviewNote'));
 assert.equal(previewText.includes('identityKeep'),false,'Non-merge previews never show a surviving ID or model-after-merge field');
 assert.equal(previewText.includes('identityModel'),false);
}
props={...props,entityId:'b'};render();assert.equal(render().find(n=>n.type==='select').props.value,'b','Selecting the other photo keeps the same compared pair');
props={...props,publicationId:'pub',canWrite:false};render();
render().find(n=>n.type==='button'&&n.props.children==='identityPreview').props.onClick();
assert.equal(render().some(n=>n.type==='button'&&n.props.children==='identityApply'),false,'Public review cannot apply scene edits');
render().find(n=>n.type==='button'&&n.props.children==='identitySuggest').props.onClick();
assert.equal(applied.length,1);assert.equal(suggestions.length,1);assert.equal(suggestions[0].shareForReview,true);
assert.deepEqual(suggestions[0].entityIds,['a','b']);
const oldApply=identityOperations(revision,[entity,other],'same','reason','a',null,'id',null);
assert.equal(oldApply[0].decision.baseRevisionId,'revision');
props={...props,revision:{...revision,id:'new-revision'}};render();assert.equal(render().some(n=>n.type==='button'&&n.props.children==='identitySuggest'),false,'A changed revision invalidates the preview');
console.log('Identity review: active-only render/edit/CAD, exact camera/state binding, old reader routing, two-photo preview, atomic owner CAS and shared public suggestion without scene writes passed');

// A mistaken prior merge is split with explicit evidence partitions and an explicit superseding decision.
const merged={...entity,observationRefs:['oa','ob'],measurementEvidence:[{id:'cross',measurementKey:'basis',observationRefs:['oa','ob']}]};
const splitDocument={...document,entities:[merged],identityDecisions:[{id:'prior',decision:'same',source:'manual',entityIds:['a','b'],observationGroups:[['oa'],['ob']],reason:'prior confirmation'}]};
const splitProps={revision:{id:'split-base',document:splitDocument},entity:merged,onApply:async(operations,context)=>{applied.push({operations,context});return true;}};
slots.length=0;effects.length=0;
function renderSplit(){for(let i=0;i<3;i++){cursor=0;tree=module.exports.IdentitySplit(splitProps);for(const effect of effects.splice(0))effect();}return nodes(tree);}
for(const [index,value] of ['a','b','a','b','retain','prior'].entries()) renderSplit().filter(node=>node.type==='select')[index].props.onChange({target:{value}});
renderSplit().find(node=>node.type==='textarea').props.onChange({target:{value:'Two distinct fixed objects; the old cross-group measurement remains source evidence only'}});
renderSplit().find(node=>node.type==='button'&&node.props.children==='identitySplitPreview').props.onClick();
await renderSplit().find(node=>node.type==='button'&&node.props.children==='identityApply').props.onClick();
const split=applied.at(-1);
assert.equal(split.context.baseRevisionId,'split-base');assert.equal(split.operations[0].decision.supersedesDecisionId,'prior');
assert.deepEqual(split.operations[1].groups.map(group=>group.observationRefs),[['oa'],['ob']]);
assert.deepEqual(split.operations[1].groups.map(group=>group.representationIds),[['model-a'],['model-b']]);
assert.deepEqual(split.operations[1].groups.map(group=>group.measurementEvidenceIds),[[],[]]);
assert.deepEqual(split.operations[1].retainedMeasurementEvidenceIds,['cross']);
assert.notEqual(split.operations[1].groups[0].id,split.operations[1].groups[1].id);
console.log('Identity split: exact observation/model partitions, retained cross-group measurement, prior-decision supersession and captured revision passed');

// Manager-facing status/measurement labels use the same bilingual source dictionary.
const evidenceModule={exports:{}}, evidenceSource=await readFile(new URL('../src/ModelEvidence.tsx',import.meta.url),'utf8');
const evidenceCode=ts.transpileModule(evidenceSource,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX}}).outputText;
let language=0;
new Function('require','module','exports',evidenceCode)(name=>name==='./core'?core:name==='./scene-semantics'?sceneSemantics:name==='./identity-messages'?identityCopy:name==='./i18n'?{useI18n:()=>({t:key=>identityCopy.identityMessages[key]?.[language]||key})}:require(name),evidenceModule,evidenceModule.exports);
const evidenceEntity={...entity,representations:[{...models[0],placementState:'unconfirmed'}],measurementEvidence:[{id:'measurement',measurementKey:'projectedHull',sourceEntityId:'a',sourceRevisionId:'old',originalMeasurement:{evidence:'unchanged original record'}}]};
for(language=0;language<2;language++) {
 const markup=renderToStaticMarkup(evidenceModule.exports.ModelEvidence({entity:evidenceEntity}));
 assert.ok(markup.includes(identityCopy.identityMessages.identityPlacementUnconfirmed[language]));
 assert.ok(markup.includes(identityCopy.identityMessages.identityMeasurementHull[language]));
 assert.ok(markup.includes(identityCopy.identityMessages.identityNoMeasurement[language]));
 assert.ok(!markup.includes('>unconfirmed ·'),'Placement status is not a raw enum');
 assert.ok(!markup.includes('>projectedHull<select'),'Measurement keys are not manager-facing field labels');
 assert.ok(markup.includes('unchanged original record'),'The collapsible original measurement document is retained');
}
for(const entries of [identityCopy.identityDecisionKeys,identityCopy.measurementLabelKeys])for(const key of Object.values(entries))assert.equal(identityCopy.identityMessages[key].length,2);
console.log('Identity copy: bilingual decision/placement/measurement labels, explicit unknown choice and retained original source record passed');

const surfaceEntity={...evidenceEntity,activeModelRepresentationId:models[0].id,representations:[{...models[0],kind:'generated_mesh',primitive:null,assetId:'source-derived-mesh',sourceKind:'observed_depth_surface',coverage:'observed_visible_surface_only',sourceDerivation:{methodVersion:'native-observed-topology-original-photo-uv-v1',observationId:'source-observation',observationRevision:3},sourceRefs:[{observationId:'source-observation',revision:3}]}]};
assert.equal(sceneSemantics.entityEvidenceStatus(document,surfaceEntity).modelKey,'observed_depth_surface');
assert.equal(core.activeModel(surfaceEntity).kind,'generated_mesh','A source-derived surface uses the existing editable mesh contract');
for(language=0;language<2;language++) {
 const markup=renderToStaticMarkup(evidenceModule.exports.ModelEvidence({entity:surfaceEntity}));
 assert.ok(markup.includes(identityCopy.identityMessages.observed_depth_surface[language]));
 const coverageIndex=markup.indexOf(identityCopy.identityMessages.identityVisibleSurfaceOnly[language]);
 assert.ok(coverageIndex>=0&&coverageIndex<markup.indexOf('<details'),'Visible-only coverage is readable before opening provenance details');
 assert.ok(markup.includes('native-observed-topology-original-photo-uv-v1')&&markup.includes('source-observation'),'Stored derivation and observation evidence remain inspectable');
 assert.ok(!renderToStaticMarkup(evidenceModule.exports.ModelEvidence({entity:evidenceEntity})).includes(identityCopy.identityMessages.observed_depth_surface[language]),'Generic meshes do not acquire invented photo-derived provenance');
}
console.log('Photo-derived surface: bilingual source label, visible-only coverage and retained derivation passed');

const planarEntity={...surfaceEntity,representations:[{...surfaceEntity.representations[0],sourceKind:'inferred_planar_surface_from_observed_depth',coverage:'inferred_planar_visible_region',placementState:'unconfirmed'}]};
assert.equal(sceneSemantics.entityEvidenceStatus(document,planarEntity).modelKey,'inferred_planar_surface_from_observed_depth');
assert.equal(sceneSemantics.modelLabelKey({...planarEntity.representations[0],kind:'observed_surface'}),'observed_surface','A source observation does not acquire the active model label');
for(language=0;language<2;language++) {
 const markup=renderToStaticMarkup(evidenceModule.exports.ModelEvidence({entity:planarEntity}));
 for(const key of ['inferred_planar_surface_from_observed_depth','identityInferredPlaneOnly','identityPlacementUnconfirmed'])assert.ok(markup.indexOf(identityCopy.identityMessages[key][language])>=0&&markup.indexOf(identityCopy.identityMessages[key][language])<markup.indexOf('<details'),'Inferred shape, partial coverage and pending placement remain visible');
 assert.ok(!markup.includes(identityCopy.identityMessages.observed_depth_surface[language]),'An inferred plane is distinct from an observed depth surface');
}
console.log('Inferred plane: distinct bilingual source label, visible-region limitation and unconfirmed placement passed');

// A source update must not silently switch the chosen model or claim it is current valid geometry.
const staleEntity={...evidenceEntity,representations:[{...models[0],sourceValidity:'stale'},models[1]],activeModelRepresentationId:'model-a'};
assert.equal(sceneSemantics.entityEvidenceStatus(document,staleEntity).modelKey,'identityModelStale');
assert.equal(core.activeModel(staleEntity).id,'model-a','The old asset and explicit choice remain stored');
assert.equal(core.editableTransform(staleEntity),null);assert.equal(core.modelGeometry(staleEntity),null);
assert.equal(sceneSemantics.entityEvidenceStatus(document,{...staleEntity,activeModelRepresentationId:null}).modelKey,'identitySourceModelsOnly','An unselected alternative is source evidence, not the current model');
for(language=0;language<2;language++) {
 const markup=renderToStaticMarkup(evidenceModule.exports.ModelEvidence({entity:staleEntity}));
 assert.ok(markup.includes(identityCopy.identityMessages.identityModelStaleNote[language]));
 assert.ok(markup.includes(identityCopy.identityMessages.identitySourceStale[language]));
}
const appSource=await readFile(new URL('../src/App.tsx',import.meta.url),'utf8'),workspace=appSource.slice(appSource.indexOf('function Workspace('),appSource.indexOf('function EntityInspector('));
assert.match(workspace,/objects = document\.entities\.filter\(entity => !entity\.sourceContext\)/);
assert.match(workspace,/<span>\{objects\.length\}<\/span>/);
assert.match(workspace,/\{objects\s*\.filter/);
assert.ok(!workspace.includes('{document.entities\n            .filter'),'The workspace list and count use the same non-context collection');
console.log('Workspace evidence: background context excluded consistently, stale active model labelled and preserved without geometry/editability passed');
