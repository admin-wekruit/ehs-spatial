// node --experimental-strip-types web/checks/renderer-selection.mjs [publication-bundle.json]
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import ts from 'typescript';
import {representationPass} from '../src/viewer/native-viewer.ts';
import {cameraForImage} from '../src/core.ts';
import {modelFamily} from '../src/scene-semantics.ts';
import {boundsCorners,point,transformMatrix} from '../src/viewer/native-math.ts';

// Execute the actual draw function against a one-pixel depth buffer. Geometry,
// selection uniforms, ordering and LESS/LEQUAL decisions all come from the reader.
const source=await readFile(new URL('../src/viewer/native-viewer.ts',import.meta.url),'utf8');
const ast=ts.createSourceFile('native-viewer.ts',source,ts.ScriptTarget.ES2022,true,ts.ScriptKind.TS);
const functions={};let signature;
const visit=node=>{if(ts.isFunctionDeclaration(node)&&['draw','fittingPoints','corners'].includes(node.name?.text))functions[node.name.text]=node.getText(ast);if(ts.isVariableDeclaration(node)&&node.name.getText(ast)==='signature')signature=node.initializer?.getText(ast);ts.forEachChild(node,visit);};visit(ast);assert.ok(functions.draw&&functions.fittingPoints&&functions.corners&&signature);
const code=ts.transpileModule(functions.draw+'\nglobalThis.render=draw;',{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
const transform={coordinateFrameId:'native',position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]};
const rep={id:'surface',assetId:'mesh',kind:'observed_surface',coordinateFrameId:'native',placementState:'confirmed',transform,sourceRefs:[{observationId:'observation'}]};
const observations=[{id:'observation',imageId:'photo'}];
const context={id:'context',sourceContext:true},marking={id:'marking',observationRefs:['observation'],activeModelRepresentationId:'model',currentModelTransform:transform},occluder={id:'occluder',observationRefs:['observation']};
const gpu=(entityId,depth,representation=rep,indexCount=3)=>({entityId,representation,vertex:{},index:{entityId,depth},texture:{},mesh:{mode:representation.kind==='point_cloud'?0:4,indices:new Uint32Array(indexCount)}});
function pixel(objects,selected='marking',override={},pick=false,exact=false){
 let depth=Infinity,depthFunc='LESS',selectedUniform=0,current,pixel=null;
 const gl={LESS:'LESS',LEQUAL:'LEQUAL',ELEMENT_ARRAY_BUFFER:'ELEMENT_ARRAY_BUFFER',COLOR_BUFFER_BIT:1,DEPTH_BUFFER_BIT:2,
  isContextLost:()=>false,viewport(){},useProgram(){},clearColor(){},enable(){},disable(){},blendFunc(){},
  clear(){depth=Infinity;pixel=null;},depthFunc(value){depthFunc=value;},bindBuffer(target,buffer){if(target==='ELEMENT_ARRAY_BUFFER')current=buffer;},
  uniformMatrix4fv(){},uniform1f(name,value){if(name==='selected')selectedUniform=value;},uniform3f(){},uniform3fv(){},activeTexture(){},bindTexture(){},
  drawElements(){if(current.depth<depth||depthFunc==='LEQUAL'&&current.depth===depth){depth=current.depth;pixel={entityId:current.entityId,selected:!!selectedUniform};}},
 };
 const doc={entities:[context,marking,occluder]},layers={observed_surface:true,generated_mesh:true,primitive:true,point_cloud:false,showBounds:false,showCandidates:false,imageId:'photo',observations,...override};
 const scope=vm.createContext({disposed:false,camera:{exact},gl,viewSize:()=>({w:100,h:100,cw:100,ch:100}),devicePixelRatio:1,
  canvas:{style:{},width:100,height:100},photo:{style:{}},program:{},cameraMatrix:()=>[],radius:1,u:{selected:'selected'},gpu:objects,doc,layers,
  entity:id=>doc.entities.find(e=>e.id===id),selection:{entityId:selected},frameId:'native',representationPass,modelFamily,attrs:[],model:()=>[],
  svg:{setAttribute(){},replaceChildren(){}},projected(){},add(){},document:{}});
 vm.runInContext(code,scope);scope.render(pick);return pixel;
}
for(const exact of [false,true])for(const objects of [[gpu('context',.5),gpu('marking',.5)],[gpu('marking',.5),gpu('context',.5)]]){
 assert.deepEqual(pixel(objects,'marking',{},false,exact),{entityId:'marking',selected:true},'A selected observed surface must tint over its coplanar context, independent of async asset order');
 assert.deepEqual(pixel(objects,'marking',{},true,exact),{entityId:'marking',selected:true},'Picking retains exact coplanar entity ownership');
}
for(const objects of [[gpu('marking',.5),gpu('occluder',.5,rep,30)],[gpu('occluder',.5,rep,30),gpu('marking',.5)]]){
 assert.deepEqual(pixel(objects,null,{},true),{entityId:'marking',selected:false},'At equal depth the smaller actual observed triangle set wins picking, independent of asset arrival order');
 assert.deepEqual(pixel(objects,'occluder'),{entityId:'occluder',selected:true},'An explicitly selected broader surface still receives its own tint without changing its identity');
}
assert.deepEqual(pixel([gpu('marking',.5),gpu('occluder',.4,rep,30)],null,{},true),{entityId:'occluder',selected:false},'Observed interaction priority cannot pick through a nearer actual surface');
const tied=[gpu('marking',.5),gpu('occluder',.5)];
assert.deepEqual(pixel(tied,null,{},true),pixel(tied.slice().reverse(),null,{},true),'Equal-size observed ties resolve by stable identity, not asynchronous GPU upload order');
assert.deepEqual(pixel([gpu('context',.5),gpu('occluder',.3),gpu('marking',.5)]),{entityId:'occluder',selected:false},'A real nearer surface still occludes the selected geometry');
assert.deepEqual(pixel([gpu('context',.5),gpu('marking',.4,{...rep,id:'model',kind:'generated_mesh'})]),{entityId:'marking',selected:true},'The active generated model retains its existing selected tint');
assert.equal(pixel([gpu('context',.5),gpu('marking',.5)],null).selected,false,'No selection never tints another surface');
assert.equal(pixel([gpu('context',.5),gpu('marking',.5)],'marking',{observed_surface:false}),null,'Disabled geometry is not redrawn as a selection overlay');
assert.deepEqual(pixel([gpu('context',.5),gpu('marking',.5,{...rep,coordinateFrameId:'other'})]),{entityId:'context',selected:false},'Selection never borrows geometry from another frame');
assert.deepEqual(pixel([gpu('context',.5),gpu('marking',.5,{...rep,sourceRefs:[{imageId:'other-photo'}]})]),{entityId:'context',selected:false},'Selection retains the exact photo-source guard');
const cloud={observed_surface:false,generated_mesh:false,primitive:false,point_cloud:true},cloudGPU=gpu('context',.5,{...rep,kind:'point_cloud'});
assert.deepEqual(pixel([cloudGPU,gpu('marking',.5)],'marking',cloud),{entityId:'context',selected:false},'Cloud color rendering does not inject hidden observed triangles or claim ownership of background samples');
assert.deepEqual(pixel([cloudGPU,gpu('marking',.5)],'marking',cloud,true),{entityId:'marking',selected:true},'Cloud picking retains its exact observed-triangle evidence without changing color visibility');
assert.deepEqual(pixel([cloudGPU,gpu('marking',.5,{...rep,kind:'point_cloud'})],'marking',cloud),{entityId:'marking',selected:true},'An actual visible entity-owned point cloud can receive its own selected tint');
assert.deepEqual(pixel([cloudGPU,gpu('marking',.5)],null,cloud),{entityId:'context',selected:false},'Unselected observed triangles stay hidden in cloud mode');
assert.deepEqual(pixel([gpu('context',.3,{...rep,kind:'point_cloud'}),gpu('marking',.5,{...rep,kind:'point_cloud'})],'marking',cloud),{entityId:'context',selected:false},'Nearer cloud samples still occlude a selected entity-owned point cloud');
assert.deepEqual(pixel([cloudGPU,gpu('marking',.5,{...rep,id:'model',kind:'generated_mesh'})],'marking',cloud),{entityId:'context',selected:false},'Cloud highlighting never enables hidden generated geometry');
const fittingCode=ts.transpileModule(functions.corners+'\n'+functions.fittingPoints+'\nglobalThis.fit=fittingPoints;\nglobalThis.signatureFor=next=>'+signature+';',{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
const fitGPU=(entityId,min,max,representation=rep)=>({...gpu(entityId,.5,representation),mesh:{bounds:{min,max}}});
const fitScope=vm.createContext({gpu:[],entity:id=>[context,marking,occluder].find(entity=>entity.id===id),selection:{entityId:null},
 visible:g=>representationPass([context,marking,occluder].find(entity=>entity.id===g.entityId),g.representation,'native',{observed_surface:true,generated_mesh:true,primitive:true,point_cloud:true,showCandidates:true,imageId:'photo',observations}).visible,
 boundsCorners,point,model:()=>transformMatrix(transform)});
vm.runInContext(fittingCode,fitScope);
const distantContext=fitGPU('context',[-100,-100,-100],[100,100,100]);
for(const kind of ['observed_surface','generated_mesh','primitive','point_cloud']){
 const object=fitGPU('marking',[1,2,3],[2,3,4],{...rep,kind,id:kind==='generated_mesh'||kind==='primitive'?'model':rep.id});
 fitScope.gpu=[distantContext,object];
 const points=fitScope.fit();
 assert.equal(points.length,8,'Visible object-owned geometry frames the scene without including distant background');
 assert.equal(Math.max(...points.map(p=>p[0])),2,'Every visible representation kind can frame its actual business object');
 const before=JSON.stringify(points);fitScope.selection.entityId='marking';
 assert.equal(JSON.stringify(fitScope.fit()),before,'Selection never changes the scene fitting scope');
 assert.equal(fitScope.gpu.length,2,'Framing keeps the full context available for rendering');
}
fitScope.gpu=[distantContext,fitGPU('marking',[1,2,3],[2,3,4],{...rep,id:'model',kind:'generated_mesh'}),fitGPU('occluder',[10,2,3],[11,3,4])];
assert.equal(Math.max(...fitScope.fit().map(p=>p[0])),11,'Model availability cannot exclude other visible observed objects from the frame');
fitScope.gpu=[distantContext,fitGPU('marking',[1,2,3],[2,3,4],{...rep,sourceRefs:[{imageId:'other-photo'}]})];
assert.equal(Math.max(...fitScope.fit().map(p=>p[0])),100,'With no visible business geometry, normal fitting uses the visible context');
const signatureDoc={entities:[{...marking,representations:[rep]}]},staleDoc=structuredClone(signatureDoc);
staleDoc.entities[0].representations[0].sourceValidity='stale';
assert.notEqual(fitScope.signatureFor(staleDoc),fitScope.signatureFor(signatureDoc),'A stale-to-active representation must reload GPU data, not take the metadata-only fast path');
const movedDoc=structuredClone(signatureDoc);movedDoc.entities[0].currentModelTransform.position=[5,6,7];
assert.equal(fitScope.signatureFor(movedDoc),fitScope.signatureFor(signatureDoc),'A pose edit alone still retains the existing GPU buffers');
if(process.argv[2]){
 const bundle=JSON.parse(await readFile(process.argv[2],'utf8'));
 const publication=bundle.responses['/api/publications/'+bundle.publicationId],doc=publication.snapshot.revision.document;
 const entity=doc.entities.find(e=>e.id==='4f584238-6ce2-5f12-97b2-4429f122b72e');assert.ok(entity);
 const photo=doc.assets.filter(a=>a.kind==='source_image')[2],camera=cameraForImage(doc,photo.id);assert.ok(camera);
 const layers={imageId:photo.id,observations:doc.observations,observed_surface:true,point_cloud:false,generated_mesh:true,primitive:true,showCandidates:true};
 const visible=entity.representations.filter(rep=>representationPass(entity,rep,camera.coordinateFrameId,layers).visible);
 assert.ok(visible.some(rep=>rep.kind==='observed_surface'),'The reported marking has actual visible geometry in photo 3');
 console.log('Publication marking: '+visible.length+' visible source representations in the bound photo-3 frame');
}
console.log('Native draw: coplanar selection, deterministic picking, real occlusion, frame/photo scope, object-based fit, retained context and stale/active GPU lifecycle passed');
