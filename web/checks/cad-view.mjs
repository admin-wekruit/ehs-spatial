// Run: node --experimental-strip-types web/checks/cad-view.mjs [scene-document.json]
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import ts from "typescript";
import { planShapes } from "../src/core.ts";

const source = await readFile(new URL("../src/CadView.tsx", import.meta.url), "utf8");
const parsed = ts.createSourceFile("CadView.tsx", source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX);
const names = ["cadFit", "cadScreen", "cadWorld", "cadStep", "cadNumber", "cadCallouts"];
const declarations = parsed.statements.filter(node => ts.isFunctionDeclaration(node) && names.includes(node.name?.text));
assert.equal(declarations.length, names.length);
const code = ts.transpileModule(declarations.map(node => node.getText(parsed)).join("\n"), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText;
const context = vm.createContext({ exports: {} });
vm.runInContext(code, context);
const { cadFit, cadScreen, cadWorld, cadStep, cadCallouts } = context.exports;
const shape = (id, min, max) => ({ entity: { id }, min, max, polygons: [{exterior: [min, [max[0], min[1]], max, [min[0], max[1]]], holes: []}], lines: [] });
let shapes = [shape("small", [0, 0], [1, 2]), shape("far", [-5, 4], [-4.8, 4.2])];
if (process.argv[2]) {
  const document = JSON.parse(await readFile(process.argv[2], "utf8"));
  shapes = planShapes(document, { layer: "model", frameId: document.coordinateFrames[0].id, showCandidates: true });
  assert.ok(shapes.length > 0);
}
const original = JSON.stringify(shapes);
for (const size of [{ width: 436.5, height: 47.65625 }, { width: 360, height: 260 }, { width: 900, height: 700 }]) {
  const camera = cadFit(shapes, size);
  const extents = shapes.flatMap(shape => shape.polygons[0].exterior.map(point => cadScreen(point, camera, size)));
  const occupied = [0, 1].map(k => Math.max(...extents.map(p => p[k])) - Math.min(...extents.map(p => p[k])));
  assert.ok(occupied[0] >= size.width * .79 || occupied[1] >= size.height * .79,
    "Automatic fit must fill at least 79% of one canvas dimension, including compact panes");
  for (const shape of shapes) for (const point of shape.polygons[0].exterior) {
    const screen = cadScreen(point, camera, size);
    assert.ok(screen[0] >= 0 && screen[0] <= size.width);
    assert.ok(screen[1] >= 0 && screen[1] <= size.height);
    const roundTrip = cadWorld(screen, camera, size);
    assert.ok(roundTrip.every((value, index) => Math.abs(value - point[index]) < 1e-9));
  }
  const step = cadStep(60 / camera.scale);
  assert.ok(step * camera.scale >= 60 && step * camera.scale <= 150,
    "Native scale bar must correspond to the rendered world-space distance");
  const selectedId = shapes[0].entity.id;
  const labels = cadCallouts(shapes, selectedId, camera, size);
  assert.equal(labels[0].id, selectedId, "Selected object gets first choice of callout position");
  for (let i = 0; i < labels.length; i++) for (let j = i + 1; j < labels.length; j++) {
    assert.ok(Math.abs(labels[i].position[0] - labels[j].position[0]) >= 34 || Math.abs(labels[i].position[1] - labels[j].position[1]) >= 24,
      "Short object-number callouts must not collide");
  }
}
assert.equal(JSON.stringify(shapes), original, "Layout cannot rewrite or discard saved detailed polygons");
const focusedSize = { width: 450, height: 280 }, focusedShape = shape("selected", [0, 0], [3, 1]);
const focusedCamera = cadFit([focusedShape], focusedSize, 160);
assert.ok(cadScreen(focusedShape.max, focusedCamera, focusedSize)[0] <= focusedSize.width - 80,
  "Focus mode must reserve space for the projected-depth dimension");
const narrowSize = { width: 100, height: 40 }, narrowFocus = cadFit([focusedShape], narrowSize, 160);
assert.ok(cadScreen(focusedShape.max, narrowFocus, narrowSize)[0] - cadScreen(focusedShape.min, narrowFocus, narrowSize)[0] >= 60,
  "Explicit focus padding must also shrink on narrow panes instead of compressing the object to one pixel");
assert.deepEqual(Array.from(cadWorld([120, 70], { center: [2, -3], scale: 20 }, { width: 200, height: 100 })), [3, -4]);
assert.ok(source.includes("planPolygonPath(polygon, p => cadScreen(p, view, size))") && source.includes('fillRule="evenodd"'), "Render each true polygon with its holes directly");
assert.ok(source.includes("planHits(shapes, world[0], world[1], 4 / view.scale)"), "Overlap picking uses the same geometry as the drawing");
assert.ok(!source.includes("onWheel="), "Reading the report must not trap wheel scrolling");
assert.ok(source.includes("suppressClick.current = !!pointer.current?.moved"), "Dragging must not trigger selection");

// Execute the real memo initializer across two photographs in the same frame.
let memoExpression;
function findMemo(node) { if (ts.isVariableDeclaration(node) && node.name.getText(parsed) === "shapes") memoExpression = node.initializer.getText(parsed); ts.forEachChild(node, findMemo); }
findMemo(parsed);
const transform = x => ({coordinateFrameId:"frame",position:[x,0,0],quaternion:[0,0,0,1],scale:[1,1,1]});
const observations = [{id:"a",imageId:"photo-a"},{id:"b",imageId:"photo-b"}];
const photographed = {entities:[{id:"object",observationRefs:["a","b"],representations:observations.map((observation,i)=>({id:observation.id,kind:"observed_surface",assetId:observation.id,coordinateFrameId:"frame",transform:transform(i*10),placementState:"confirmed",bounds:{min:[0,0,0],max:[1,1,1]},sourceRefs:[{observationId:observation.id}]}))}],observations,coordinateFrames:[{id:"frame",ground:{normal:[0,0,1]}}]};
photographed.assets = observations.map(observation=>({id:observation.id,sha256:'1'.repeat(64)}));
photographed.observations.forEach(observation=>observation.revision=1);
for (const rep of photographed.entities[0].representations) {
  rep.sourceRefs[0].revision=1;
  const observation=observations.find(observation=>observation.id===rep.id),x=rep.transform.position[0];
  rep.planProjection={methodVersion:'indexed-mesh-triangle-union-v1',coordinateFrameId:'frame',assetId:rep.assetId,assetSha256:'1'.repeat(64),imageId:observation.imageId,observationId:observation.id,observationRevision:1,groundNormalSnapshot:[0,0,1],transformSnapshot:structuredClone(rep.transform),nativeToPlane:[[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]],polygons:[{exterior:[[-1,x],[0,x],[0,x+1],[-1,x+1],[-1,x]],holes:[]}],lines:[]};
}
let cache;
const memoContext = vm.createContext({document:photographed,geometryOptions:{layer:"observed_surface",frameId:"frame",imageId:"photo-a"},planShapes,
  useMemo:(make,deps)=>{if(!cache || deps.some((value,i)=>value!==cache.deps[i])) cache={deps,result:make()};return cache.result;}});
const memoCode = ts.transpileModule(`globalThis.result = ${memoExpression}`,{compilerOptions:{target:ts.ScriptTarget.ES2022}}).outputText;
vm.runInContext(memoCode,memoContext);
assert.equal(memoContext.result[0].representationIds[0],"a");
memoContext.geometryOptions={...memoContext.geometryOptions,imageId:"photo-b"};vm.runInContext(memoCode,memoContext);
assert.equal(memoContext.result[0].representationIds[0],"b","Changing only the photograph must recompute CAD geometry and its dimension labels");
memoContext.geometryOptions={...memoContext.geometryOptions,scope:"scene"};vm.runInContext(memoCode,memoContext);
assert.equal(memoContext.result.length,0,"Switching from photo evidence to a scene reference cannot reuse the old photo cache when no reference was declared");
assert.ok(source.includes("[selectedId, document, geometryOptions?.scope, geometryOptions?.layer, geometryOptions?.imageId]"),"Changing photo must dismiss stale overlap candidates");

// Execute the component's actual pointer handlers, including a release outside
// before capture starts. A later button-up hover must not resume that gesture.
const handlers = {};
function visit(node) {
  if (ts.isJsxAttribute(node) && ["onPointerDown", "onPointerMove", "onPointerUp", "onPointerCancel"].includes(node.name.text)) {
    handlers[node.name.text] = node.initializer.expression.getText(parsed);
  }
  ts.forEachChild(node, visit);
}
visit(parsed);
const pointer = { current: null }, suppressClick = { current: false }, cameras = [], captured = new Set();
const gestures = vm.createContext({ pointer, suppressClick, view: { center: [2, 3], scale: 10 }, setCamera: value => cameras.push(value), setCandidates: () => {} });
for (const [name, handler] of Object.entries(handlers)) vm.runInContext(ts.transpileModule(`globalThis.${name} = ${handler}`, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
}).outputText, gestures);
const currentTarget = { setPointerCapture: id => captured.add(id), hasPointerCapture: id => captured.has(id), releasePointerCapture: id => captured.delete(id) };
const event = (overrides = {}) => ({ pointerId: 1, pointerType: "mouse", button: 0, buttons: 1, clientX: 100, clientY: 100, currentTarget, ...overrides });
gestures.onPointerDown(event());
gestures.onPointerMove(event({ clientX: 102 }));
assert.equal(cameras.length, 0);
assert.equal(captured.size, 0, "Click targeting is preserved until a drag starts");
// The SVG receives no pointerup when a mouse releases outside before capture.
gestures.onPointerMove(event({ buttons: 0, clientX: 130, clientY: 120 }));
assert.equal(pointer.current, null, "Button-up reentry clears a pending uncaptured gesture");
assert.equal(cameras.length, 0, "Button-up reentry must not pan the camera");
gestures.onPointerDown(event());
gestures.onPointerMove(event({ clientX: 120, clientY: 110 }));
assert.deepEqual(Array.from(cameras.at(-1).center), [0, 4], "A new press still starts a normal drag");
gestures.onPointerUp(event({ buttons: 0, clientX: 120, clientY: 110 }));
assert.equal(pointer.current, null);
assert.equal(suppressClick.current, true);
assert.equal(captured.size, 0);
gestures.onPointerDown(event({ pointerType: "touch" }));
gestures.onPointerMove(event({ pointerType: "touch", buttons: 0, clientX: 120 }));
assert.equal(cameras.length, 2, "Touch contact does not depend on mouse button reporting");
gestures.onPointerCancel();
assert.equal(pointer.current, null);
console.log(`CAD drafting check passed: ${shapes.length} shapes; full-scene fit, invertible coordinates, native scale, labels and pointer lifecycle.`);
