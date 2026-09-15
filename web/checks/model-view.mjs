// node --experimental-strip-types web/checks/model-view.mjs
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import ts from 'typescript';
import * as math from '../src/viewer/native-math.ts';
import * as core from '../src/core.ts';
import {modelFamily} from '../src/scene-semantics.ts';
import {representationPass,selectionGeometry} from '../src/viewer/native-viewer.ts';

const familyFixture=JSON.parse(await readFile(new URL('./fixtures/model-family.json',import.meta.url),'utf8'));
for(const test of familyFixture.cases){
 const document=structuredClone(familyFixture.document);
 for(const entity of document.entities){assert.deepEqual(modelFamily(document,entity.id).map(member=>member.id),familyFixture.families[entity.id]);Object.assign(entity,structuredClone(test.entityOverrides[entity.id]||{}));}
 const before=JSON.stringify(document);
 if(test.errorCode)assert.throws(()=>core.previewOperations(document,test.operations,familyFixture.baseRevisionId),new RegExp(test.errorCode),test.name);
 else{
  const after=core.previewOperations(document,test.operations,familyFixture.baseRevisionId);
  assert.deepEqual(after.observations,document.observations);assert.deepEqual(after.assets,document.assets);
  for(const entity of after.entities){const expected=test.expected[entity.id],matrix=math.transformMatrix(entity.currentModelTransform);
   if(expected.matrix===null)assert.equal(entity.currentModelTransform,null);else matrix.forEach((value,index)=>assert.ok(Math.abs(value-expected.matrix[index])<=1e-6+1e-5*Math.abs(expected.matrix[index]),test.name+' '+entity.id+' matrix '+index));
   assert.equal(entity.visible??null,expected.visible);assert.equal(entity.label,expected.label);
   assert.deepEqual(Object.fromEntries(entity.representations.map(rep=>[rep.id,{material:rep.material??null,placementState:rep.placementState}])),expected.representations,test.name);
   if(test.expectedRelations)assert.deepEqual({parentEntityId:entity.parentEntityId??null,partRelation:entity.partRelation??null,lineage:entity.lineage||[]},test.expectedRelations[entity.id],test.name+' relation provenance');
  }
 }
 assert.equal(JSON.stringify(document),before,'Family previews are atomic and preserve the source document');
}

const transform={coordinateFrameId:'frame',position:[0,0,0],quaternion:[0,0,0,1],scale:[1,1,1]};
const model={id:'model-a',kind:'generated_mesh',assetId:'asset-a',coordinateFrameId:'frame',transform,placementState:'unconfirmed',placementReason:'imported_proposal',bounds:{min:[-1,-1,-1],max:[1,1,1]}};
const observed={...model,id:'surface',kind:'observed_surface',placementState:'confirmed',sourceRefs:[{imageId:'photo'}]};
const a={id:'a',activeModelRepresentationId:'model-a',currentModelTransform:{...transform,position:[7,2,1]},representations:[model,{...model,id:'alternative'},observed]};
const b={id:'b',activeModelRepresentationId:'model-b',representations:[{...model,id:'model-b',assetId:'asset-b'}]};
const missing={id:'missing',representations:[observed]},context={id:'context',sourceContext:true,representations:[observed]};
const document={captureId:'capture',entities:[a,b,missing,context],coordinateFrames:[{id:'frame',ground:{normal:[0,0,1]}}],cameras:[],observations:[],assets:[]};
const strict={modelOnly:true,observed_surface:false,generated_mesh:true,primitive:true,point_cloud:false,showCandidates:true,imageId:'other-photo'};
assert.equal(core.entityGeometryForLayer(missing,{layer:'model',frameId:'frame',imageId:'photo'}),null,'Observed meshes never substitute for a missing model');
assert.ok(core.entityGeometryForLayer(missing,{layer:'observed_surface',frameId:'frame',imageId:'photo'}),'Observed evidence remains available in its own mode');
assert.deepEqual(a.representations.filter(rep=>representationPass(a,rep,'frame',strict).visible).map(rep=>rep.id),['model-a'],'Only the active candidate renders, independent of viewed photo');
assert.equal(representationPass(context,observed,'frame',strict).pick,false,'Observed context never renders or picks in the model scene');
assert.equal(representationPass(a,model,'other-frame',strict).visible,false,'A different camera frame cannot borrow a model');
assert.equal(core.entityGeometryForLayer(a,{layer:'model',frameId:'frame',imageId:'other-photo'}).corners[0][0],6,'The current model transform is authoritative');
assert.equal(model.placementState,'unconfirmed','Rendering cannot confirm placement');

const source=await readFile(new URL('../src/viewer/native-viewer.ts',import.meta.url),'utf8');
const parsed=ts.createSourceFile('native-viewer.ts',source,ts.ScriptTarget.ES2022,true,ts.ScriptKind.TS);
const wanted=['captureModel','fittedCamera','fittingPoints','corners','dimensions','model','primitive','selectedGeometry','selectedAxes'];
const declarations=[];let drawSource;const visit=node=>{if(ts.isFunctionDeclaration(node)&&node.name?.text==='draw')drawSource=node.getText(parsed);if(ts.isFunctionDeclaration(node)&&wanted.includes(node.name?.text))declarations.push(node.getText(parsed));ts.forEachChild(node,visit);};visit(parsed);
assert.equal(declarations.length,wanted.length);
const code=ts.transpileModule(declarations.join('\n')+'\nglobalThis.capture=captureModel;globalThis.primitiveMesh=primitive;globalThis.axesFor=selectedAxes;', {compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
const gpu=document.entities.flatMap(entity=>entity.representations.map(representation=>({entityId:entity.id,representation,mesh:{bounds:model.bounds,matrix:math.identity()}})));
let shots=[],failCapture=false;
const canvas={width:900,height:500,style:{cssText:'original-canvas'},toDataURL(){if(failCapture)throw Error('capture failed');shots.push({camera:structuredClone(scope.camera),studio:scope.layers.studio,axes:scope.axesFor(scope.layers.axisEntityId),size:scope.captureSize,entities:gpu.filter(scope.visible).map(g=>g.entityId),representations:gpu.filter(scope.visible).map(g=>g.representation.id)});return 'data:image/png;base64,actual-model';}};
const scope=vm.createContext({...math,...core,modelFamily,representationPass,selectionGeometry,document:{createElement(){return {getContext:()=>({}),toDataURL:()=>canvas.toDataURL()};}},disposed:false,revisionId:'revision',gl:{isContextLost:()=>false},doc:document,frameId:'frame',gpu,
 entity:id=>document.entities.find(e=>e.id===id),layers:{...strict},camera:{eye:[30,20,10],target:[1,2,3],up:[0,0,1]},selection:{entityId:'b'},radius:99,center:[4,5,6],preview:new Map(),captureSize:null,
 loadedRepresentations:new Set(['a/model-a','b/model-b']),canvas,photo:{style:{cssText:'original-photo'}},stage:{clientWidth:0,clientHeight:0},draw(){},
 visible:g=>representationPass(document.entities.find(e=>e.id===g.entityId),g.representation,'frame',scope.layers).visible,
});
vm.runInContext(code,scope);
const original={camera:scope.camera,layers:scope.layers,selection:scope.selection,radius:scope.radius,center:scope.center},before=JSON.stringify(document);
for(const mode of ['free','front','side','top']){
 assert.equal(scope.capture('a',mode,'revision'),'data:image/png;base64,actual-model');
 const shot=shots.at(-1);assert.equal(shot.studio,true,'Object capture always enables readable studio display without changing the main view lighting');assert.deepEqual(shot.entities,['a']);assert.deepEqual(shot.representations,['model-a']);
 assert.equal(shot.size.w,shot.size.h,'Model previews fit the near-square fourth pane');
 assert.deepEqual([...shot.axes.axes.map(axis=>axis.label)],['X','Y','Z']);
 const projection=math.cameraMatrix(shot.camera,1,scope.radius);for(const axis of shot.axes.axes){const pixel=math.projected(projection,axis.end,640,640);assert.ok(pixel&&pixel[0]>16&&pixel[0]<610&&pixel[1]>20&&pixel[1]<620,'Snapshot fitting keeps real axis endpoints and labels inside the frame');}
 assert.equal(!!shot.camera.orthographic,mode!=='free','Existing orthographic camera modes are reused');
 for(const key of Object.keys(original))assert.equal(scope[key],original[key],`Capture restores ${key}`);
 assert.equal(scope.captureSize,null);assert.equal(canvas.width,900);assert.equal(canvas.style.cssText,'original-canvas');
}
const originalPose=a.currentModelTransform;
a.currentModelTransform={...originalPose,quaternion:[0,0,Math.SQRT1_2,Math.SQRT1_2],scale:[2,3,4]};
for(const mode of ['free','front','side','top']){
 scope.capture('a',mode,'revision');const shot=shots.at(-1),expected=[[0,1,0],[-1,0,0],[0,0,1]];
 shot.axes.axes.forEach((axis,i)=>axis.direction.forEach((v,k)=>assert.ok(Math.abs(v-expected[i][k])<1e-6,'XYZ follows the current saved model quaternion, with non-unit scale normalized')));
 const matrix=math.cameraMatrix(shot.camera,1,scope.radius);for(const axis of shot.axes.axes){const pixel=math.projected(matrix,axis.end,640,640);assert.ok(pixel&&pixel[0]>16&&pixel[0]<610&&pixel[1]>20&&pixel[1]<620,'Rotated/non-unit model axes and labels remain inside every preview');}
}
a.currentModelTransform=originalPose;
assert.equal(scope.capture('a','free','old-revision'),null,'An old revision cannot supply a newer preview');
assert.equal(scope.capture('a','free','revision','other-frame'),null,'A pending camera-frame change cannot publish geometry from the old frame');
scope.loadedRepresentations.delete('a/model-a');assert.equal(scope.capture('a','free','revision'),null,'A partially loaded multi-mesh model is never published as a complete preview');scope.loadedRepresentations.add('a/model-a');
assert.equal(scope.capture('missing','free','revision'),null,'No invented bounds image for an unmodeled object');
failCapture=true;assert.throws(()=>scope.capture('a','top','revision'),/capture failed/);for(const key of Object.keys(original))assert.equal(scope[key],original[key],'Errors restore scene navigation too');
assert.equal(JSON.stringify(document),before,'Preview rendering leaves source state and placement confirmation unchanged');
failCapture=false;
const assembly={id:'assembly',activeModelRepresentationId:null,currentModelTransform:{...transform,position:[100,0,0]},representations:[]};
document.entities.push(assembly);a.parentEntityId=assembly.id;b.parentEntityId=assembly.id;
assert.equal(core.editableTransform(assembly,document),assembly.currentModelTransform,'An empty-residual assembly retains a transform control for its family');
assert.equal(core.editableTransform({...assembly,id:'not-an-assembly'},document),null,'A loose record with a pivot does not invent an editable model');
assert.deepEqual(modelFamily(document,'assembly').map(entity=>entity.id),['assembly','a','b']);
assert.deepEqual(modelFamily(document,'b').map(entity=>entity.id),['b'],'Selecting a part never includes its parent or sibling');
const familyGeometry=core.modelFamilyGeometry(document,'assembly',{layer:'model',frameId:'frame'});
assert.equal(Math.min(...familyGeometry.corners.map(p=>p[0])),-1);
assert.equal(Math.max(...familyGeometry.corners.map(p=>p[0])),8,'Absolute child transforms must not receive the parent pivot transform again');
assert.equal(scope.capture('assembly','front','revision'),'data:image/png;base64,actual-model','An empty-residual parent previews the actual part union');
assert.deepEqual(shots.at(-1).entities,['a','b'],'Each actual active mesh appears once in the assembly preview');
assert.equal(scope.capture('b','front','revision'),'data:image/png;base64,actual-model');
assert.deepEqual(shots.at(-1).entities,['b'],'The child preview contains only its own active mesh');
const familyKey=JSON.stringify(core.modelFamilySignature(document,'assembly'));
b.currentModelTransform={...transform,position:[-4,0,0]};
assert.notEqual(JSON.stringify(core.modelFamilySignature(document,'assembly')),familyKey,'A descendant pose change invalidates its parent snapshot');
const movedKey=JSON.stringify(core.modelFamilySignature(document,'assembly'));
b.representations[0].assetId='replacement-asset';
assert.notEqual(JSON.stringify(core.modelFamilySignature(document,'assembly')),movedKey,'A descendant asset replacement invalidates its parent snapshot');
scope.loadedRepresentations.delete('b/model-b');
assert.equal(scope.capture('assembly','front','revision'),null,'An assembly with a failed or pending part never publishes a partial preview');
delete a.parentEntityId;delete b.parentEntityId;delete b.currentModelTransform;b.representations[0].assetId='asset-b';document.entities.pop();scope.loadedRepresentations.add('b/model-b');
const cylinder=scope.primitiveMesh({kind:'cylinder',radius:2,height:3});assert.equal(cylinder.vertices.length/11,130);assert.equal(cylinder.indices.length,768,'Default 64-segment cylinder has the same cap-center topology as backend primitive_mesh');
for(const mesh of [cylinder,scope.primitiveMesh({kind:'box',dimensions:[1,2,3]})])for(let i=0;i<mesh.vertices.length;i+=11)assert.deepEqual([...mesh.vertices.slice(i+6,i+9)],[1,1,1],'Primitive vertex color is neutral, so explicit material RGB is applied exactly once and the absent-material default matches export');
assert.equal(scope.primitiveMesh({kind:'cylinder',radius:2,height:3,segments:8}).vertices.length/11,18);
for(const segments of [7,257,8.5])assert.throws(()=>scope.primitiveMesh({kind:'cylinder',radius:2,height:3,segments}),/invalid_primitive/);
assert.equal((source.match(/getContext\('webgl2'/g)||[]).length,1,'Object previews never allocate a second WebGL context');
// Exercise the actual WebGL draw setup: studio affects presentation uniforms
// only, including when the normal scene's lighting has been switched off.
const displayed={};
const displayScope=vm.createContext({disposed:false,camera:{exact:false},devicePixelRatio:1,
 modelFamily,
 gl:{isContextLost:()=>false,viewport(){},useProgram(){},clearColor(...rgba){displayed.background=rgba;},clear(){},enable(){},disable(){},uniformMatrix4fv(){},uniform1f(name,value){displayed[name]=value;}},
 viewSize:()=>({w:640,h:400,cw:640,ch:400}),canvas:{width:640,height:400,style:{}},photo:{style:{}},program:{},cameraMatrix:()=>[],radius:1,
 layers:{studio:true,lighting:false,showBounds:false},u:{lighting:'lighting'},gpu:[],selection:{},svg:{setAttribute(){},replaceChildren(){}},
});
vm.runInContext(ts.transpileModule(drawSource+';globalThis.draw=draw;', {compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText,displayScope);
displayScope.draw();assert.deepEqual(displayed.background,[237/255,240/255,238/255,1]);assert.equal(displayed.lighting,2,'Studio uses soft illumination even when the main scene lighting is off');
displayScope.layers.studio=false;displayScope.draw();assert.deepEqual(displayed.background,[17/255,27/255,33/255,1]);assert.equal(displayed.lighting,0,'The regular scene retains the user lighting choice');
displayScope.layers.studio=true;displayScope.draw(true);assert.deepEqual(displayed.background,[0,0,0,1],'Studio background cannot create phantom objects in the ID-picking buffer');


// Composite the same projected native SVG axes into the exported model image.
const capturedLabels=[],capturedLines=[],svgNodes=[];
const overlay={drawImage(image){assert.equal(image,displayScope.canvas);},scale(){},beginPath(){},moveTo(){},lineTo(x,y){capturedLines.push([x,y]);},stroke(){},strokeText(){},fillText(text){capturedLabels.push(text);}};
const captureCanvas={getContext:()=>overlay},shot=shots[0];
Object.assign(displayScope,{camera:shot.camera,cameraMatrix:math.cameraMatrix,projected:math.projected,add:math.add,
 viewSize:()=>({w:640,h:640,cw:640,ch:640}),layers:{studio:true,showBounds:false,showAxes:true,axisEntityId:'a'},
 doc:document,entity:id=>document.entities.find(e=>e.id===id),selectedAxes:()=>shot.axes,
 svg:{namespaceURI:'http://www.w3.org/2000/svg',setAttribute(){},replaceChildren(){svgNodes.length=0;},append(node){svgNodes.push(node);}},
 document:{createElementNS(_ns,tag){return {tag,attributes:{},setAttribute(k,v){this.attributes[k]=v;}};}},
});
displayScope.draw(false,captureCanvas);assert.deepEqual(capturedLabels,['X','Y','Z'],'Snapshot includes all three native axis labels instead of exporting WebGL pixels alone');
assert.equal(capturedLines.length,3,'Snapshot contains axes only, without re-enabling all object bounds');assert.equal(overlay.font,'700 24px sans-serif','Snapshot labels stay legible when fitted into a report pane');
assert.equal(captureCanvas.width,640);assert.equal(captureCanvas.height,640);
const svgLines=svgNodes.filter(node=>node.tag==='line');capturedLines.forEach(([x,y],i)=>assert.deepEqual([x,y],[Number(svgLines[i].attributes.x2),Number(svgLines[i].attributes.y2)],'Snapshot and native viewer use identical axis projection'));

// Execute SpatialView's real effect/event bridge: delayed A asset events must
// service the current B request, never replace B with the previous selection.
const app=await readFile(new URL('../src/App.tsx',import.meta.url),'utf8');
const ast=ts.createSourceFile('App.tsx',app,ts.ScriptTarget.ES2022,true,ts.ScriptKind.TSX);
const component=ast.statements.find(node=>ts.isFunctionDeclaration(node)&&node.name?.text==='SpatialView');
const executable=ts.transpileModule(component.getText(ast).replace('export function','function'),{compilerOptions:{target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.React}}).outputText;
const hooks=[];let cursor=0,effects=[],events,mounts=0,disposeCount=0;const ready=new Set(),results=[],cameras=[];
const runtime={setScene(){},setSelection(){},setLayers(){},setCamera(value){cameras.push(value);},dispose(){disposeCount++;},captureModel(id){return ready.has(id)?'data:image/png;base64,'+id:null;}};
const bridge=vm.createContext({React:{createElement:(type,props,...children)=>({type,props:props||{},children})},
 useRef(initial){const i=cursor++;return hooks[i]??={current:initial};},useState(initial){const i=cursor++;if(!(i in hooks))hooks[i]=initial;return [hooks[i],value=>{hooks[i]=typeof value==='function'?value(hooks[i]):value;}];},
 useEffect(fn,deps){const i=cursor++,old=hooks[i];if(!old||deps.some((value,j)=>!Object.is(value,old[j]))){hooks[i]=deps;effects.push(fn);}},
 useI18n:()=>({language:'en',t:key=>key}),mountSceneViewer(_host,options){mounts++;events=options.onEvent;return runtime;},resolveAsset(){},ErrorNotice(){},photoHits:core.photoHits,currentCameras:core.currentCameras,modelFamily,
});vm.runInContext(executable,bridge);
const bridgeDocument={...document,cameras:[{id:'camera',imageId:'photo-1',coordinateFrameId:'frame'},{id:'camera-2',imageId:'photo-2',coordinateFrameId:'frame'},{id:'camera-3',imageId:'photo-3',coordinateFrameId:'other'}],geometryBindings:{'photo-1':{cameraId:'camera'},'photo-2':{cameraId:'camera-2'},'photo-3':{cameraId:'camera-3'}}};
let props={revision:{id:'revision',document:bridgeDocument},selection:{entityId:'a'},onSelect(){},onCommit(){},mode:'free',cameraId:'camera',layers:strict,modelPreview:{entityId:'a',mode:'free',requestKey:'a-key'},onModelPreview:(key,image)=>results.push({key,image})};
function render(){cursor=0;effects=[];const tree=bridge.SpatialView(props);const bind=node=>{if(node?.props?.className==='native-viewer')node.props.ref.current={};for(const child of node?.children||[])if(child&&typeof child==='object')bind(child);};bind(tree);for(const effect of effects)effect();return tree;}
render();assert.equal(results.length,0);
props={...props,selection:{entityId:'b'},modelPreview:{entityId:'b',mode:'front',requestKey:'b-key'}};render();
ready.add('a');events({type:'loadProgress',phase:'assets'});assert.equal(results.length,0,'Late A loading does not satisfy B');
events({type:'loadError',entityId:'a',code:'late_a_failure'});
let bridgeTree=render();assert.equal(bridgeTree.children.find(node=>node?.type===bridge.ErrorNotice).props.error,undefined,'A late failure for A never replaces the selected B state');
events({type:'loadError',entityId:'b',code:'b_failure'});bridgeTree=render();
assert.equal(bridgeTree.children.find(node=>node?.type===bridge.ErrorNotice).props.error.message,'b_failure','A failure belongs to its selected model');
ready.add('b');events({type:'loadProgress',phase:'assets'});assert.deepEqual(results,[{key:'b-key',image:'data:image/png;base64,b'}]);
events({type:'renderReady',phase:'scene'});assert.equal(results.length,1,'Later source events do not overwrite a completed current preview');
props={...props,layers:{...strict,modelOnly:false,generated_mesh:false,primitive:false,observed_surface:true}};render();assert.equal(cameras.length,2,'Changing representation refits the existing scene viewer');
props={...props,cameraId:'camera-2'};render();assert.equal(cameras.length,2,'Selecting another source photo in the same frame preserves the free-view camera');
props={...props,cameraId:'camera-3'};render();assert.equal(cameras.length,3,'Changing the native frame refits its own scene');
props={...props,mode:'photo',cameraId:'camera'};render();assert.equal(cameras.length,4);
props={...props,cameraId:'camera-2'};render();assert.equal(cameras.length,5,'Photo mode still follows each source camera');
assert.equal(mounts,1);assert.equal(disposeCount,0,'Selection and layer changes retain the one mounted viewer');
console.log('Model view: strict active models, candidate state, current transforms, same-GPU snapshots, restored camera after success/error, hidden-pane capture, exact cylinder tessellation and late A/B preview events passed');
