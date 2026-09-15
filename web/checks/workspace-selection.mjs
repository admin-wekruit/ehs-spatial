// Run: node --experimental-strip-types web/checks/workspace-selection.mjs
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import React from 'react';
import ts from 'typescript';
import * as core from '../src/core.ts';
import {entityEvidenceStatus,isReferenceSurface} from '../src/scene-semantics.ts';

// Execute the real Workspace handlers without mounting GPU/network children.
const source=await readFile(new URL('../src/App.tsx',import.meta.url),'utf8');
const parsed=ts.createSourceFile('App.tsx',source,ts.ScriptTarget.ES2022,true,ts.ScriptKind.TSX);
const declaration=parsed.statements.find(n=>ts.isFunctionDeclaration(n)&&n.name?.text==='Workspace');
const code=ts.transpileModule('export '+declaration.getText(parsed),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX}}).outputText;
const slots=[],effects=[];let cursor=0,tree,props,serial=0;
const hooks={
 useState(initial){const i=cursor++;if(!Object.hasOwn(slots,i))slots[i]=typeof initial==='function'?initial():initial;return [slots[i],v=>{slots[i]=typeof v==='function'?v(slots[i]):v;}];},
 useRef(initial){const i=cursor++;return slots[i]??={current:initial};},
 useEffect(effect,deps){const i=cursor++,previous=slots[i];if(!previous||deps.some((v,j)=>!Object.is(v,previous[j]))){slots[i]=deps;effects.push(effect);}},
};
const children=Object.fromEntries(['PhotoView','SpatialView','CadView','PlanView','AssetImage','EntityInspector','IdentityReview','AgentPanel','PrimitiveCreator','ErrorNotice','ModelEvidence','Badge','PrimitiveFields','TransformFields'].map(name=>[name,()=>null]));
const scope={...core,...hooks,...children,entityEvidenceStatus,isReferenceSurface,useI18n:()=>({t:key=>key}),id:()=>`request-${++serial}`};
const module={exports:{}},require=createRequire(import.meta.url);
new Function('require','module','exports',...Object.keys(scope),code)(require,module,module.exports,...Object.values(scope));
const nodes=node=>React.isValidElement(node)?[node,...React.Children.toArray(node.props.children).flatMap(nodes)]:[];
function render(flush=true){cursor=0;tree=module.exports.Workspace(props);if(flush){for(let i=0;i<3;i++){for(const effect of effects.splice(0))effect();cursor=0;tree=module.exports.Workspace(props);}}return nodes(tree);}
const find=type=>render().find(n=>n.type===children[type]);
const selected=()=>find('PhotoView').props.selectedId;
const observation=()=>find('SpatialView').props.selection.observationId;
function reset(document,query=''){
 slots.length=0;effects.length=0;
 const url=new URL('http://localhost/app.html#/projects/p/workbench'+query);
 globalThis.location=url;
 globalThis.history={replaceState(_state,_title,next){url.hash=next;}};
 props={revision:{id:'revision',document},project:{id:'p'},branch:{id:'branch'},canWrite:true,onCommit:async()=>true};
 render();
}
const entity=(id,refs)=>({id,label:id,observationRefs:refs,representations:[],measurements:{}});
const base={schemaVersion:2,target:'scene',entities:[entity('a',['oa']),entity('b',['ob']),entity('c',['oc'])],observations:[{id:'oa',imageId:'photo',originalPixelBox:[0,0,5,5]},{id:'ob',imageId:'photo',originalPixelBox:[10,0,15,5]},{id:'oc',imageId:'photo',originalPixelBox:[20,0,25,5]}],assets:[{id:'photo',kind:'source_image'}],annotations:[],coordinateFrames:[],cameras:[{id:'camera',imageId:'photo'}],geometryBindings:{photo:{cameraId:'camera',geometrySolutionId:'solution'}},identityDecisions:[]};
const merged={...base,entities:[entity('a',['oa','ob']),base.entities[2]],identityDecisions:[{entityIds:['a','b'],observationGroups:[['oa'],['ob']],decision:'same',survivorId:'a'}]};
reset(merged,'?object=b&observation=ob&image=photo');
assert.equal(find('EntityInspector').props.canConfirmPlacement,true,'Only owner workspaces offer saved placement confirmation');
props={...props,canWrite:false};assert.equal(find('EntityInspector').props.canConfirmPlacement,false,'Visitor temporary edits cannot claim persistent placement confirmation');props={...props,canWrite:true};
assert.equal(selected(),'a','A retired deep-link ID resolves to its unique current survivor');
assert.equal(observation(),'ob','The exact retained observation remains selected');
assert.equal(render().filter(n=>n.props.className?.startsWith('entity-row ')).length,2,'Merged same-photo masks produce one entity row, not one row per mask');
for(const x of [2,12]){const hit=core.photoHits(merged,'photo',x,2)[0];find('PhotoView').props.onSelect(hit.entity.id,hit.observation.id);assert.equal(selected(),'a');assert.equal(observation(),hit.observation.id);}
find('PhotoView').props.onSelect('b','ob');assert.equal(selected(),'a');assert.equal(observation(),'ob','Retained old-ID selection events use the same mapping');
const transform={coordinateFrameId:'frame',position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]};
const surfaces=['oa','ob'].map((observationId,i)=>({id:'surface-'+i,assetId:'asset-'+i,kind:'observed_surface',coordinateFrameId:'frame',transform,placementState:'confirmed',bounds:{min:[i*3,0,0],max:[i*3+1,1,1]},sourceRefs:[{observationId,revision:1}],planProjection:{methodVersion:'indexed-mesh-triangle-union-v1',coordinateFrameId:'frame',assetId:'asset-'+i,assetSha256:'a'.repeat(64),imageId:'photo',observationId,observationRevision:1,transformSnapshot:transform,groundNormalSnapshot:[0,0,1],nativeToPlane:[[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]],polygons:[{exterior:[[0,i*3],[-1,i*3],[-1,i*3+1],[0,i*3+1],[0,i*3]],holes:[]}],lines:[]}}));
const projection=core.planShapes({...merged,observations:merged.observations.map(observation=>({...observation,revision:1})),assets:surfaces.map(rep=>({id:rep.assetId,sha256:'a'.repeat(64)})),coordinateFrames:[{id:'frame',ground:{normal:[0,0,1]}}],entities:[{...merged.entities[0],representations:surfaces}]},{layer:'observed_surface',frameId:'frame',imageId:'photo'});
assert.equal(projection.length,1,'CAD projects the current entity once even when it retains two same-photo surfaces');assert.deepEqual(projection[0].representationIds,['surface-0','surface-1']);
reset(merged,'?object=b&observation=oc');assert.equal(selected(),'a');assert.equal(observation(),null,'An unrelated observation cannot be attached to the survivor');
reset(merged);assert.equal(selected(),null,'No query means no default selection');
reset(base,'?object=b&observation=ob');props={...props,revision:{id:'merged',document:merged}};
assert.equal(render(false).find(n=>n.type===children.PhotoView).props.selectedId,'a','Revision changes resolve before children see a retired ID');
render();assert.equal(observation(),'ob');
const split={...merged,entities:[entity('a1',['oa']),entity('a2',['ob']),base.entities[2]],identityDecisions:[...merged.identityDecisions,{entityIds:['a'],observationGroups:[['oa'],['ob']],decision:'different'}]};
props={...props,revision:{id:'split',document:split}};render();assert.equal(selected(),null,'A split with multiple current owners is not guessed');assert.equal(observation(),null);

// A queued normalization must use/retain a newer user choice.
reset(base,'?object=b&observation=ob');props={...props,revision:{id:'merged',document:merged}};
render(false).find(n=>n.type===children.PhotoView).props.onSelect('c','oc');render();
assert.equal(selected(),'c');assert.equal(observation(),'oc');
// A late model-save completion must also not steal the user's new object selection.
reset(base,'?object=a&observation=oa');let complete;
props={...props,onCommit:()=>new Promise(resolve=>{complete=resolve;})};
render().find(n=>n.type==='button'&&n.props['aria-label']==='addModel').props.onClick();
const pending=find('PrimitiveCreator').props.onCommit([{type:'setPrimitive',entityId:'a'}]);
find('PhotoView').props.onSelect('c','oc');render();complete(true);await pending;
assert.equal(selected(),'c','A late creation response cannot override a newer selection');assert.equal(observation(),'oc');
const inspectorDeclaration=parsed.statements.find(n=>ts.isFunctionDeclaration(n)&&n.name?.text==='EntityInspector');
const inspectorCode=ts.transpileModule(inspectorDeclaration.getText(parsed).replace('function EntityInspector','export function ConfirmInspector'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX}}).outputText;
const inspectorModule={exports:{}};new Function('require','module','exports',...Object.keys(scope),inspectorCode)(require,inspectorModule,inspectorModule.exports,...Object.values(scope));
const candidate={...entity('a',['oa']),activeModelRepresentationId:'model',currentModelTransform:transform,representations:[{id:'model',kind:'primitive',coordinateFrameId:'frame',transform,primitive:{kind:'box',dimensions:[2,2,2]},placementState:'unconfirmed',placementReason:'requires_alignment_confirmation'}]},confirmations=[];
function inspectorNodes(canConfirmPlacement){slots.length=0;effects.length=0;cursor=0;return nodes(inspectorModule.exports.ConfirmInspector({entity:candidate,document:{...base,entities:[candidate],coordinateFrames:[{id:'frame',ground:{normal:[0,0,1]}}]},busy:false,canConfirmPlacement,onCommit:operations=>confirmations.push(operations)}));}
assert.equal(inspectorNodes(false).some(n=>n.type==='button'&&n.props.children==='acceptPlacement'),false,'A visitor sees the candidate state without a persistent-confirmation control');
inspectorNodes(true).find(n=>n.type==='button'&&n.props.children==='acceptPlacement').props.onClick();
assert.deepEqual(confirmations,[[{type:'confirmPlacement',entityId:'a',representationId:'model'}]],'Owner confirmation names the exact active representation instead of abusing a no-op transform');
console.log('Workspace selection: retired IDs, exact mask ownership, one merged row, default none, ambiguous split, revision remapping and late user selection passed');
