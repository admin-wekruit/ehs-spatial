// Run: node --experimental-strip-types web/checks/report-scene.mjs
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import ts from "typescript";
import * as math from "../src/viewer/native-math.ts";
import { entityEvidenceStatus, identityCounts, isReferenceSurface, modelFamily } from "../src/scene-semantics.ts";
import { cameraForImage, activeModel, modelFamilyGeometry, modelFamilySignature, entityGeometryForLayer, observationsFor, jsonObject, planShapes, cadReferenceImage, scenePlanOptions, sourceDimensions, sourceScale, previewOperations } from "../src/core.ts";

// Exercise the actual two pure functions without importing the browser app.
const source = await readFile(
  new URL("../src/ReportScene.tsx", import.meta.url),
  "utf8",
);
const parsed = ts.createSourceFile(
  "ReportScene.tsx",
  source,
  ts.ScriptTarget.ES2022,
  true,
  ts.ScriptKind.TSX,
);
const names = ["photoOverlay", "sceneAvailability"];
const declarations = parsed.statements.filter(
  (node) => ts.isFunctionDeclaration(node) && names.includes(node.name?.text),
);
assert.equal(declarations.length, names.length);
const executable = ts.transpileModule(
  declarations.map((node) => node.getText(parsed)).join("\n"),
  {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.None,
    },
  },
).outputText;
const context = vm.createContext({
  ...math,
  entityGeometryForLayer,
  isReferenceSurface,
  jsonObject,
  planShapes,
  colors: ["red", "green", "blue"],
});
vm.runInContext(executable, context);
const { photoOverlay, sceneAvailability } = context;
const geometryOptions = { layer: "model", frameId: "f", showCandidates: true, imageId: "photo", observations:[{id:"obs",imageId:"photo"}] };
const document = { observations: [{id:"obs",imageId:"photo"}], coordinateFrames: [] };
const camera = {
  id: "camera",
  imageId: "photo",
  coordinateFrameId: "f",
  width: 640,
  height: 960,
  K: [
    [610, 17, 233.2],
    [0, 602, 381.9],
    [0, 0, 1],
  ],
  cameraToWorld: [
    [0, 0, 1, 3],
    [0, 1, 0, 1],
    [-1, 0, 0, 2],
    [0, 0, 0, 1],
  ],
};
const transform = {
  coordinateFrameId: "f",
  position: [6, 1, 2],
  quaternion: [0, 0, 0, 1],
  scale: [1, 1, 1],
};
const rep = {
  id: "mesh",
  assetId: "model-asset",
  kind: "generated_mesh",
  coordinateFrameId: "f",
  transform,
  placementState: "unconfirmed",
  placementReason: "imported_proposal",
  bounds: { min: [-0.1, -0.2, -0.3], max: [0.1, 0.2, 0.3] },
};
const entity = {
  id: "object", activeModelRepresentationId: "mesh",
  associationState: "confirmed",
  representations: [rep],
  currentModelTransform: transform,
};
const geometry = entityGeometryForLayer(entity, geometryOptions),
  overlay = photoOverlay(document, entity, camera, "model");
assert.equal(overlay.corners.length, 8);
assert.equal(overlay.axes.length, 3);
for (let i = 0; i < geometry.corners.length; i++) {
  const world = geometry.corners[i],
    local = [2 - world[2], world[1] - 1, world[0] - 3];
  const expected = [
    (610 * local[0] + 17 * local[1]) / local[2] + 233.2 + 0.5,
    (602 * local[1]) / local[2] + 381.9 + 0.5,
  ];
  assert.ok(
    overlay.corners[i].every(
      (value, axis) => Math.abs(value - expected[axis]) < 0.001,
    ),
    "Photo SVG must agree with the calibrated pixel projection",
  );
}
assert.equal(
  photoOverlay(document, { id: "no-mesh", representations: null }, camera, "model"),
  null,
);
assert.equal(
  photoOverlay(document, entity, { ...camera, coordinateFrameId: "other" }, "model"),
  null,
);
assert.equal(
  photoOverlay(document,
    { ...entity, representations: [{ ...rep, placementReason: "no_depth" }] },
    camera,
    "model",
  ),
  null,
);
assert.equal(photoOverlay(document, entity, camera, "point_cloud"), null,
  "A point-cloud display cannot borrow a generated model bounding box");
const observed = {
  ...rep,
  kind: "observed_surface", sourceRefs:[{observationId:"obs"}],
  placementState: "confirmed",
};
const observedOnly = {
  ...entity,
  observationRefs: ["obs"],
  representations: [observed],
  currentModelTransform: { ...transform, position: [100, 100, 100] },
};
assert.equal(
  JSON.stringify(entityGeometryForLayer(observedOnly, { ...geometryOptions, layer: "observed_surface" }).corners),
  JSON.stringify(geometry.corners),
  "Immutable observed geometry must not move with a model transform",
);
assert.equal(photoOverlay(document, observedOnly, camera, "point_cloud").corners.length, 8,
  "Point-cloud mode reuses the object's confirmed observed surface");
assert.equal(photoOverlay(document, observedOnly, camera, "point_cloud").axisSpace, "native");
const measuredOnly = { measurementSelections:{basis:"basis"}, measurementEvidence:[{id:"basis",observationRefs:["obs"]}], id: "measured-only", representations: [], measurements: {
  coordinateFrameId: "f", basis: { cornersNative: geometry.corners },
} };
const measuredOverlay = photoOverlay(document, measuredOnly, camera, "point_cloud");
assert.equal(measuredOverlay.corners.length, 8);
assert.equal(measuredOverlay.axes.length, 3);
assert.equal(measuredOverlay.axisSpace, "native",
  "A located native observation gets scene axes, never an invented local structure pose");
assert.equal(photoOverlay(document, measuredOnly, { ...camera, coordinateFrameId: "other" }, "model"), null);
assert.equal(photoOverlay(document, { ...measuredOnly, sourceContext: true }, camera, "model"), null);
assert.equal(photoOverlay(document, { id: "size-only", measurements: { dimensionsNative: [1, 2, 3] } }, camera, "model"), null);
const floorObservation = { id: "floor-observation", revision: 2, labelEvidence: [{ label: "floor", source: "legacy_import" }] };
const floorDocument = { observations: [floorObservation], coordinateFrames: [] };
const floor = { ...observedOnly, id: "reference", label: "任意显示名称", observationRefs: [floorObservation.id] };
assert.equal(isReferenceSurface(floorDocument, floor), true, "Frozen observation categories survive renaming");
assert.equal(isReferenceSurface(floorDocument, { ...floor, label: "a machine" }), true);
assert.equal(isReferenceSurface(document, { ...observedOnly, label: "floor" }), false, "Display names never classify a surface");
assert.equal(isReferenceSurface(document, { ...observedOnly, geometryRole: "floor" }), true);
for (const role of ["unknown", "object"]) {
  assert.equal(isReferenceSurface(floorDocument, { ...floor, geometryRole: role }), false, "Explicit role overrides frozen category");
  assert.equal(isReferenceSurface({ ...floorDocument, observations: [{ ...floorObservation, labelEvidence: [{ label: "floor", geometryRole: role }] }] }, floor), false);
}
for (const layer of ["model", "observed_surface", "point_cloud"])
  assert.equal(photoOverlay(floorDocument, floor, camera, layer), null, "Reference surfaces never get photo equipment axes/volume boxes");
assert.equal(entityGeometryForLayer({...floor,representations:[{...observed,sourceRefs:[{observationId:floorObservation.id}]}]}, { ...geometryOptions, observations:[{...floorObservation,imageId:camera.imageId}], layer: "observed_surface" }).corners.length, 8, "Observed evidence is retained independently of pose overlays");
const groundDocument = { observations: [{ ...floorObservation, labelEvidence: [] }], coordinateFrames: [{ ground: { sourceRefs: [{ observationId: floorObservation.id, revision: 2 }] } }] };
assert.equal(isReferenceSurface(groundDocument, floor), true, "An explicitly bound ground observation establishes a reference role, not a slope");
assert.equal(isReferenceSurface({ ...groundDocument, coordinateFrames: [{ ground: { sourceRefs: [{ observationId: floorObservation.id, revision: 1 }] } }] }, floor), false, "Stale ground observation revisions cannot supply a role");
assert.equal(isReferenceSurface({ ...document, coordinateFrames: [{ ground: { normal: [0, 0, 1], sourceRefs: [{ assetId: "whole-scene" }] } }] }, observedOnly), false, "A ground normal or whole-scene source asset does not classify every object");
const behind = { ...transform, position: [1, 1, 2] };
const behindOverlay = photoOverlay(document,
  { ...entity, currentModelTransform: behind },
  camera,
  "model",
);
assert.ok(behindOverlay.corners.every((corner) => corner === null));
assert.equal(behindOverlay.axes.length, 0);
const objectOnlyDocument = { schemaVersion: 1, cameras: [], assets: [{ id: "model-asset" }],
  observations: [], annotations: [], coordinateFrames: [{ id: "f", convention: "opencv", ground: null }],
  entities: [{ ...entity, representations: [{ ...rep, assetId: "model-asset" }] }, { id: "unmodeled", representations: [] }] };
const unchanged = structuredClone(objectOnlyDocument);
assert.equal(sceneAvailability(objectOnlyDocument, geometryOptions).spatialTitle, "sceneModelScene",
  "Individual model assets must not imply an observed scene reconstruction");
assert.equal(sceneAvailability(objectOnlyDocument, geometryOptions).planEmpty, "sceneNoPlanGround");
assert.deepEqual(objectOnlyDocument, unchanged, "Presentation must preserve no-geometry entities");
const sceneDocument = structuredClone(objectOnlyDocument);
sceneDocument.entities.push({ id: "context", sourceContext: true, representations: [{ ...observed, kind: "point_cloud", assetId: "cloud" }] });
sceneDocument.assets.push({ id: "cloud" });
assert.equal(sceneAvailability(sceneDocument, { ...geometryOptions, layer: "observed_surface" }).spatialTitle, "scene3D",
  "Confirmed registered observed geometry uses a neutral scene title, without asserting completeness");
sceneDocument.coordinateFrames[0].ground = { normal: [0, 0, 1] };
assert.equal(sceneAvailability(sceneDocument, geometryOptions).planEmpty, "sceneNoPlanProjection", "Model corners without a true mesh contour remain 3D evidence, not CAD");
sceneDocument.entities = [{ id: "unmodeled", representations: [] }];
assert.equal(sceneAvailability(sceneDocument, geometryOptions).planEmpty, "sceneNoPlanProjection",
  "Ground alone does not establish an object footprint");
// These are architecture invariants, separate from projection assertions.
assert.equal(
  (source.match(/<SpatialView\b/g) || []).length,
  1,
  "Exactly one WebGL host",
);
assert.match(source, /editable:\s*false/);
assert.match(source, /onCommit=\{noEdit\}/);
assert.match(source, /disabled=\{!hasPointCloud\}/);

// Run the actual workbench handlers with React hooks, without loading the app,
// WebGL or remote assets. CSS visibility preserves the keyed renderer subtree.
const component = parsed.statements.find((node) => ts.isFunctionDeclaration(node) && node.name?.text === "ReportScene");
const componentCode = ts.transpileModule(component.getText(parsed).replace("export function", "function"), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.React, module: ts.ModuleKind.None },
}).outputText;
const hooks = []; let cursor = 0, effects = [], dirty = false, fullscreenRequests = 0;
const useState = (initial) => { const i = cursor++; if (!(i in hooks)) hooks[i] = typeof initial === "function" ? initial() : initial; return [hooks[i], (next) => { const value = typeof next === "function" ? next(hooks[i]) : next; dirty ||= !Object.is(value, hooks[i]); hooks[i] = value; }]; };
const useRef = (initial) => { const i = cursor++; return hooks[i] ||= { current: initial }; };
const useEffect = (fn, deps) => { const i = cursor++, old = hooks[i]; if (!old || deps.some((value, n) => !Object.is(value, old[n]))) effects.push(fn); hooks[i] = deps; };
const React = { createElement: (type, props, ...children) => ({ type, props: props || {}, children: children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false) }) };
const SpatialView = () => {}, PlanView = () => {}, PhotoView = () => {}, PhotoAxes = () => {}, CadView = () => {};
let uiDocument = { ...objectOnlyDocument,
  assets: [{ id: "model-asset", sha256: "a".repeat(64) }, { id: "photo", kind: "source_image" }, { id: "photo-2", kind: "source_image" }],
  geometryBindings:{photo:{cameraId:"camera",geometrySolutionId:"solution"},"photo-2":{cameraId:"camera-2",geometrySolutionId:"solution"}},
  cameras: [camera, { ...camera, id: "camera-2", imageId: "photo-2" }],
  coordinateFrames: [{ id: "f", ground: { normal: [0, 0, 1] } }],
  entities: [{ ...entity, observationRefs: ["observation"], representations: [rep, { ...observed, id: "surface" }] },
    { id: "no-geometry", label: "small button", observationRefs: ["observation-2"], representations: [] },
    { id: "background", sourceContext: true, representations: [] }],
  observations: [{ id: "observation", imageId: "photo", revision: 1 }, { id: "observation-2", imageId: "photo-2", revision: 1 }],
};
uiDocument.entities[0].representations[0] = { ...rep, planProjection: {
  methodVersion: "indexed-mesh-triangle-union-v1", coordinateFrameId: "f", assetId: "model-asset", assetSha256: "a".repeat(64),
  imageId: null, transformSnapshot: structuredClone(transform), groundNormalSnapshot: [0,0,1],
  nativeToPlane: [[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]],
  polygons: [{exterior: [[-1.2,5.9],[-.8,5.9],[-.8,6.1],[-1.2,6.1],[-1.2,5.9]], holes: []}], lines: [],
} };
uiDocument.entities[0].cadReference={status:"resolved",referenceImageId:"photo",source:"explicit_reference_image",sourceRefs:[{observationId:"observation",revision:1}]};
uiDocument.entities[0].representations[1]={...observed,id:"surface",sourceRefs:[{observationId:"observation",imageId:"photo",revision:1}],planProjection:{...structuredClone(uiDocument.entities[0].representations[0].planProjection),imageId:"photo",observationId:"observation",observationRevision:1}};
const cadSwitchDocument = structuredClone(uiDocument);
let uiSelection = { entityId: "object", cameraId: "camera" }, uiImageId = "photo", calls = [], objectListRequest = 0, feedbackEnabled = false, feedbackCalls = [], sourceCadCalls = [];
const inspector = React.createElement("div", { id: "inspector-content" }, "real host inspector");
const ui = vm.createContext({ React, useState, useRef, useEffect, useId: () => "workspace-check",
  modelFamily, modelFamilyGeometry, modelFamilySignature,
  useI18n: () => ({ t: (key) => key }), cameraForImage, activeModel, identityCounts, jsonObject, observationsFor, entityGeometryForLayer, cadReferenceImage, scenePlanOptions,
  photoOverlay, sceneAvailability, isReferenceSurface, entityEvidenceStatus, sourceDimensions, sourceScale, SpatialView, PlanView, PhotoView, PhotoAxes, CadView,
  paneOrder: ["photo", "spatial", "cad", "plan"], paneNames: { photo: "scenePhoto", spatial: "scene3D", cad: "sceneCAD", plan: "scenePlan" }, noEdit: () => {},
  window: { document: { fullscreenElement: null, addEventListener() {}, removeEventListener() {} }, requestAnimationFrame(callback) { callback(); return 1; }, cancelAnimationFrame() {}, scrollTo() { throw Error("Selecting an object must not scroll the report"); } },
});
const extent = parsed.statements.find((node) => ts.isFunctionDeclaration(node) && node.name?.text === "Extent");
vm.runInContext(ts.transpileModule(extent.getText(parsed).replace("export function", "function"), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.React, module: ts.ModuleKind.None },
}).outputText, ui);
vm.runInContext(componentCode, ui);
const nodes = (root) => [root, ...root.children.filter((child) => typeof child === "object").flatMap(nodes)];
function renderWorkspace() {
  let tree;
  for (let n = 0; n < 5; n++) {
    cursor = 0; effects = []; dirty = false;
    tree = ui.ReportScene({ revision: { id: "revision", document: uiDocument }, selection: uiSelection, imageId: uiImageId,
      cameraId: uiSelection.cameraId, inspector, objectListRequest, newerReport:{href:"#/reports/latest",title:"Latest report"},
      onOpenSourceCad: () => sourceCadCalls.push("open"),
      onFeedback: feedbackEnabled ? (id) => feedbackCalls.push(id) : undefined,
      onSelect: (id) => { uiSelection = { ...uiSelection, entityId: id }; calls.push(id); },
      onCamera: (imageId, cameraId) => { uiImageId = imageId; uiSelection = { ...uiSelection, cameraId }; },
    });
    for (const effect of effects) effect();
    if (!dirty) break;
  }
  return tree;
}
let tree = renderWorkspace();
assert.equal(nodes(tree).find(node => node.type === "select").props.value, "model", "A model-only report starts on its available representation");
const rail = () => nodes(tree).find((node) => node.props.className === "report-scene-object-rail");
const switcher = () => nodes(tree).find((node) => node.props.className === "report-scene-view-switch");
const spatialHost = () => nodes(tree).filter((node) => node.type === SpatialView);
assert.equal(nodes(tree).some(node => node.type === PlanView), false, "The fourth report pane no longer duplicates CAD footprints");
assert.equal(spatialHost()[0].props.layers.modelOnly, true);
assert.equal(spatialHost()[0].props.layers.observed_surface, false, "Current models never include the observed background");
assert.equal(nodes(tree).find(node => node.props['data-model-coverage'] !== undefined).props['data-model-coverage'], 1, "Missing models are not counted from observed geometry");
assert.equal(nodes(tree).find(node => node.props['data-model-loaded'] !== undefined).props['data-model-loaded'], 0, "Declared models are not reported as loaded before the renderer confirms upload");
const selectedAssetState={entityId:'object',representationId:'mesh',assetId:'model-asset',vertexCount:0,triangleCount:0,state:'error',errorCode:'invalid_packed_mesh'};
spatialHost()[0].props.onAssetStates('revision',[selectedAssetState]);tree=renderWorkspace();
assert.ok(nodes(tree).some(node=>node.children.includes('sceneModelLoadFailed')), 'The report shows a failed model instead of an endless loading preview');
assert.equal(nodes(tree).find(node=>node.props['data-model-loaded']!==undefined).props['data-model-loaded'],0,'Failed assets never increase the ready count');
spatialHost()[0].props.onAssetStates('revision',[{...selectedAssetState,state:'ready',vertexCount:3,triangleCount:1,errorCode:null}]);tree=renderWorkspace();
assert.equal(nodes(tree).find(node=>node.props['data-model-loaded']!==undefined).props['data-model-loaded'],1,'A successful complete upload increases actual readiness');
const firstPreview = spatialHost()[0].props;
firstPreview.onModelPreview(firstPreview.modelPreview.requestKey, "data:image/png;base64,model-a");tree = renderWorkspace();
assert.ok(nodes(tree).some(node => node.type === "img" && node.props.src.endsWith("model-a")));
uiSelection = { ...uiSelection, entityId: "no-geometry" };tree = renderWorkspace();
firstPreview.onModelPreview(firstPreview.modelPreview.requestKey, "data:image/png;base64,late-model-a");tree = renderWorkspace();
assert.equal(nodes(tree).some(node => node.type === "img"), false, "A late model-A preview cannot appear for the newer missing-model selection");
uiSelection = { ...uiSelection, entityId: "object" };tree = renderWorkspace();
uiSelection = { ...uiSelection, entityId: null };tree = renderWorkspace();
assert.equal(spatialHost()[0].props.modelPreview, undefined, "No selection never starts a model preview for an arbitrary object");
assert.ok(nodes(tree).some(node => node.children.includes("sceneSelectModel")), "The empty model pane asks the user to select an object");
uiSelection = { ...uiSelection, entityId: "object" };tree = renderWorkspace();
assert.equal(tree.props["data-view"], "quad");
assert.equal(nodes(tree).find((node) => node.type === PhotoView).props.showBounds, false, "Photo borders start hidden even with an explicit selected object");
assert.equal(spatialHost()[0].props.layers.showBounds, false, "3D bounds and axes start hidden without changing the selected entity");
assert.equal(nodes(tree).some((node) => node.type === PhotoAxes), false);
const borderToggle = nodes(tree).find((node) => node.type === "input" && node.props.type === "checkbox");
borderToggle.props.onChange({ target: { checked: true } }); tree = renderWorkspace();
assert.equal(nodes(tree).find((node) => node.type === PhotoView).props.showBounds, true);
assert.equal(spatialHost()[0].props.layers.showBounds, true);
borderToggle.props.onChange({ target: { checked: false } }); tree = renderWorkspace();
assert.equal(nodes(rail()).filter((node) => node.type === "button").length, 2, "All non-context entities have permanent selectable rows, including no-geometry objects");
assert.equal(nodes(tree).some((node) => node.type === "details" && node.props.className === "report-scene-objects"), false, "Objects are not hidden in a bottom disclosure");
assert.ok(nodes(tree).includes(inspector), "The host inspector is rendered inside the right rail");
const objectRow = nodes(rail()).find((node) => node.type === "button" && node.props.key === "object");
for (const field of ["evidence", "identity", "model", "extent"])
  assert.ok(nodes(objectRow).some((node) => node.props.className === `report-scene-object-${field}`), `The left inventory retains ${field}`);
assert.ok(nodes(objectRow).some((node) => node.type === ui.Extent), "Left inventory reuses the observed extent renderer");
assert.ok(nodes(objectRow).some((node) => node.type === "em" && node.children.includes("sceneCandidate")), "Unconfirmed placement is explicitly shown");
const measured = { ...entity, measurements: { coordinateFrameId:"f", widthNative: 2, depthNative: 3, groundHeightNative: 4 } };
const nativeExtent = ui.Extent({ entity: measured, document: uiDocument });
assert.equal(nativeExtent.children[0], "2.00 × 3.00 × 4.00");
assert.ok(nodes(nativeExtent).some((node) => node.children.includes("uncalibrated")), "Native values are never shown as metres without scale");
const scaledDocument = { ...uiDocument, coordinateFrames: [{ id: "f", scale: { nativeToMeters: 0.5 } }] };
assert.equal(ui.Extent({ entity: measured, document: scaledDocument }).children[0], "1.00 × 1.50 × 2.00");
assert.equal(ui.Extent({ entity: { ...measured, geometryRole: "floor" }, document: uiDocument }).children[0], "2.00 × 3.00", "Floor extent excludes equipment height");
for (const pane of ["photo", "spatial", "cad", "plan"]) {
  nodes(switcher()).find((node) => node.type === "button" && node.props["aria-controls"] === `workspace-check-${pane}`).props.onClick();
  tree = renderWorkspace();
  assert.equal(tree.props["data-view"], pane);
  assert.equal(spatialHost().length, 1, "View switches preserve one mounted WebGL subtree");
  assert.equal(spatialHost()[0].props.cameraId, "camera");
  assert.ok(rail() && nodes(tree).includes(inspector), "Single views retain both side rails");
}
assert.equal(fullscreenRequests, 0, "Single-view controls never require fullscreen permission");
tree.props.onKeyDown({ key: "Escape", defaultPrevented: false }); tree = renderWorkspace();
assert.equal(tree.props["data-view"], "quad", "Escape restores the four-view layout");
nodes(rail()).find((node) => node.type === "button" && node.props.key === "no-geometry").props.onClick();
tree = renderWorkspace();
assert.deepEqual(calls, ["no-geometry"]);
assert.equal(uiImageId, "photo-2", "A no-geometry object still selects its source photograph");
assert.equal(spatialHost()[0].props.selection.entityId, "no-geometry", "The host selection is shared with 3D, not copied to local state");
nodes(tree).find((node) => node.type === "input" && node.props.type === "search").props.onChange({ target: { value: "does-not-exist" } });
tree = renderWorkspace();
assert.ok(nodes(rail()).some((node) => node.type === "strong" && node.children.includes("small button")), "The current selection remains visible when a search hides its row");
nodes(tree).find((node) => node.type === "select").props.onChange({ target: { value: "observed_surface" } }); tree = renderWorkspace();
assert.equal(spatialHost()[0].props.layers.generated_mesh, false);
assert.equal(nodes(tree).find(node => node.type === CadView).props.geometryOptions.layer, "observed_surface", "CAD preserves its source layer");
for (const section of ["objects", "inspector", "views"]) {
  const tab = nodes(tree).find((node) => node.type === "button" && node.props["aria-controls"] === `workspace-check-${section}`);
  tab.props.onClick(); tree = renderWorkspace();
  assert.equal(tree.props["data-mobile-section"], section);
  assert.equal(spatialHost().length, 1, "Mobile panel switches do not destroy the renderer");
}
tree.props.ref.current = { requestFullscreen: async () => { fullscreenRequests++; } };
await nodes(tree).find((node) => node.props.className === "report-scene-fullscreen").props.onClick();
assert.equal(fullscreenRequests, 1, "Only the separate fullscreen control requests browser fullscreen");
const sourceCadButton = () => nodes(tree).find(node => node.props.className === "report-scene-source-cad");
await sourceCadButton().props.onClick();
assert.deepEqual(sourceCadCalls, ["open"], "Source CAD opens directly outside fullscreen");
sourceCadCalls = [];
ui.window.document.fullscreenElement = tree.props.ref.current;
let finishExit;
ui.window.document.exitFullscreen = () => { sourceCadCalls.push("exit"); return new Promise(resolve => { finishExit = resolve; }); };
const openingSourceCad = sourceCadButton().props.onClick();
assert.deepEqual(sourceCadCalls, ["exit"], "Source CAD outside the fullscreen host must wait until exit completes");
finishExit(); await openingSourceCad;
assert.deepEqual(sourceCadCalls, ["exit", "open"]);
sourceCadCalls = [];
ui.window.document.exitFullscreen = async () => { throw Error("fullscreen_exit_failed"); };
await sourceCadButton().props.onClick(); tree = renderWorkspace();
assert.deepEqual(sourceCadCalls, [], "Failed fullscreen exit must not scroll an invisible outside section");
assert.ok(nodes(tree).some(node => node.props.className === "report-scene-notice" && node.children.includes("sceneFullscreenUnavailable")), "Fullscreen errors reuse the existing status message");
ui.window.document.fullscreenElement = null;
await sourceCadButton().props.onClick(); tree = renderWorkspace();
assert.deepEqual(sourceCadCalls, ["open"]);
assert.equal(nodes(tree).some(node => node.props.className === "report-scene-notice" && node.children.includes("sceneFullscreenUnavailable")), false, "Successful retry clears the fullscreen error");
// CAD keeps the fixed source state even when the photo has no geometry for that entity.
const referenceCad=nodes(tree).find(node=>node.type===CadView);
assert.ok(referenceCad);
assert.equal(referenceCad.props.geometryOptions.scope,"scene");
assert.equal(referenceCad.props.geometryOptions.imageId,undefined);
assert.equal(spatialHost()[0].props.layers.imageId,"photo-2","3D remains scoped to the selected source photo");
assert.equal(planShapes(uiDocument,referenceCad.props.geometryOptions).length,1);
assert.equal(nodes(tree).find(node=>node.type==="a"&&node.props.href==="#/reports/latest").children[0],"sceneLatestReport","The fullscreen workbench exposes a real latest-report link without automatic navigation");
referenceCad.props.onSelect("object");tree=renderWorkspace();
assert.equal(uiImageId,"photo","Clicking reference CAD uses its declared source photo");
nodes(tree).find(node=>node.type==="input"&&node.props.type==="search").props.onChange({target:{value:""}});tree=renderWorkspace();
nodes(rail()).find(node=>node.type==="button"&&node.props.key==="no-geometry").props.onClick();tree=renderWorkspace();
nodes(tree).find(node=>node.type==="select").props.onChange({target:{value:"model"}});tree=renderWorkspace();
// Source CAD stays evidence; it must never replace the current scene projection.
uiDocument.reportEvidence = {historical: {runId: "source-run", cad: {assetId: "full-resolution-cad", width: 1600, height: 1240, regions: []}}};
tree = renderWorkspace();
const currentCad = nodes(tree).find((node) => node.type === CadView);
assert.equal(currentCad.props.document, uiDocument);
assert.equal(currentCad.props.selectedId, "no-geometry", "Objects without a historical CAD link stay selected in current CAD");
assert.equal(currentCad.props.geometryOptions.frameId, "f");
assert.equal(currentCad.props.key, "revision");
currentCad.props.onSelect("object"); tree = renderWorkspace();
assert.equal(uiSelection.entityId, "object");
assert.equal(uiImageId, "photo", "Current CAD uses the common object and camera selection");
nodes(tree).find((node) => node.type === "input" && node.props.type === "search").props.onChange({ target: { value: "small button" } });
tree = renderWorkspace();
assert.deepEqual(nodes(rail()).filter((node) => node.props.className === "report-scene-object-number").map((node) => node.children[0]), ["02"], "Filtering does not renumber object callouts");
nodes(tree).find((node) => node.type === "input" && node.props.type === "search").props.onChange({ target: { value: "2" } });
tree = renderWorkspace();
assert.deepEqual(nodes(rail()).filter((node) => node.props.className === "report-scene-object-number").map((node) => node.children[0]), ["02"], "CAD number searches resolve the matching row");
uiDocument.coordinateFrames[0].ground = null;
tree = renderWorkspace();
assert.equal(nodes(tree).some((node) => node.type === CadView), false, "No inferred projection without a shared ground reference, even when source CAD exists");
assert.ok(nodes(tree).some((node) => node.props.className === "report-scene-plan-empty"));
uiDocument.entities.push(...Array.from({ length: 66 }, (_, index) => ({ id: `extra-${index}`, label: `record ${index}`, representations: [], observationRefs: [] })));
objectListRequest++;
tree = renderWorkspace();
assert.equal(tree.props["data-mobile-section"], "objects", "The understanding section opens the same complete inventory on mobile");
assert.equal(nodes(rail()).filter((node) => node.type === "button").length, 68, "All 68 records remain accessible; there is no first-ten truncation");
assert.equal(nodes(tree).find((node) => node.type === "input" && node.props.type === "search").props.value, "", "View all clears the inventory search");
nodes(tree).find((node) => node.type === "input" && node.props.type === "search").props.onChange({ target: { value: "record 65" } });
tree = renderWorkspace();
const lastRow = nodes(rail()).find((node) => node.type === "button" && node.props.key === "extra-65");
assert.ok(lastRow, "Search reaches the last object even without geometry");
lastRow.props.onClick(); tree = renderWorkspace();
assert.equal(spatialHost()[0].props.selection.entityId, "extra-65", "Inventory selection still drives the existing four-view selection");
feedbackEnabled = true;
objectListRequest++;
tree = renderWorkspace();
assert.equal(nodes(rail()).filter((node) => node.props.className === "report-object-feedback").length, 68, "Every object, including no-geometry objects, has feedback");
nodes(rail()).find((node) => node.props["aria-label"] === "sceneFeedback · small button").props.onClick(); tree = renderWorkspace();
assert.deepEqual(feedbackCalls, ["no-geometry"]);
assert.equal(uiSelection.entityId, "no-geometry");
assert.equal(uiImageId, "photo-2", "Feedback uses the same entity/photo selection as all views");
assert.equal(tree.props["data-mobile-section"], "inspector", "Feedback opens the right pane on mobile as well");
const css = await readFile(new URL("../src/report-scene.css", import.meta.url), "utf8");
assert.match(css, /grid-template-columns:\s*224px minmax\(0, 1fr\) 280px/);
assert.match(css, /\.report-scene-object-list\s*\{[^}]*overflow-y:\s*auto/);
assert.match(css, /\.report-scene-inspector-content\s*\{[^}]*overflow-y:\s*auto/);
assert.match(css, /\.report-scene\[data-view="quad"\]\s*\{\s*height:\s*max\(1000px/, "Desktop four-view height permits usable CAD panes on short screens");
const report = await readFile(new URL("../src/WorkcellReport.tsx", import.meta.url), "utf8");
assert.doesNotMatch(report, /className="report-inventory"|filteredObjects|setAllObjects/, "No second object inventory or search state remains below the workspace");
assert.match(report, /section="understanding"/, "Image interpretation and its source history remain in the report");
assert.match(report, /objectListRequest=\{objectListRequest\}/, "The lower section opens the workspace inventory");
uiDocument.entities.find(entity => entity.id === "background").representations = [{...observed, coverage: "observed_camera_state_only"}];
hooks.length = 0;
tree = renderWorkspace();
assert.equal(nodes(tree).find(node => node.type === "select").props.value, "model", "An available active model is the default even when complete photo evidence is attached");

// Exercise the real report controls with more observed objects than active models.
const sourceOnly = cadSwitchDocument.entities[1], sourceRep = structuredClone(cadSwitchDocument.entities[0].representations[1]);
sourceRep.id = "source-only";
sourceRep.sourceRefs = [{ observationId: "observation-2", imageId: "photo-2", revision: 1 }];
Object.assign(sourceRep.planProjection, { imageId: "photo-2", observationId: "observation-2" });
sourceOnly.representations = [sourceRep];
sourceOnly.cadReference = { status: "resolved", referenceImageId: "photo-2", sourceRefs: [{ observationId: "observation-2", revision: 1 }] };
cadSwitchDocument.entities[2].representations = [{ ...structuredClone(sourceRep), id: "context-cloud", kind: "point_cloud" }];
const cadGeometry = (shapes) => shapes.map(({ entity, ...geometry }) => ({ entityId: entity.id, ...geometry }));
async function checkCadModes(scene, label) {
  uiDocument = scene; hooks.length = 0;
  const images = scene.assets.filter(asset => asset.kind === "source_image").map(asset => asset.id);
  uiImageId = images[0]; uiSelection = { entityId: scene.entities.find(entity => !entity.sourceContext).id, cameraId: cameraForImage(scene, uiImageId)?.id };
  tree = renderWorkspace();
  const expected = cadGeometry(planShapes(scene, scenePlanOptions(scene, "observed_surface", true, uiImageId)));
  assert.ok(expected.length > planShapes(scene, scenePlanOptions(scene, "model", true, uiImageId)).length, "The regression scene contains observed-only objects");
  const before = JSON.stringify(scene);
  for (const mode of ["model", "observed_surface", "point_cloud"]) for (const imageId of images) {
    uiImageId = imageId; uiSelection = { ...uiSelection, cameraId: cameraForImage(scene, imageId)?.id };
    nodes(tree).find(node => node.type === "select").props.onChange({ target: { value: mode } }); tree = renderWorkspace();
    const cad = nodes(tree).find(node => node.type === CadView);
    assert.ok(cad, `${label}: source CAD remains available in ${mode}`);
    assert.equal(cad.props.geometryOptions.layer, "observed_surface", `${label}: the 3D selector cannot replace the CAD source layer`);
    assert.deepEqual(cadGeometry(planShapes(scene, cad.props.geometryOptions)), expected, `${label}: ${mode}/${imageId} preserves every source contour`);
    assert.equal(cad.props.selectedId, uiSelection.entityId, "CAD selection remains shared");
    assert.equal(spatialHost().length, 1);
    if (mode === "model") {
      const models = scene.entities.filter(entity => !entity.sourceContext && entityGeometryForLayer(entity, { layer: "model", frameId: cad.props.geometryOptions.frameId, showCandidates: true }));
      assert.equal(nodes(tree).find(node => node.props["data-model-coverage"] !== undefined).props["data-model-coverage"], models.length, "Restoring CAD evidence never inflates model coverage");
      assert.ok(nodes(tree).some(node => node.children.includes("sceneModelScene")), "Source CAD does not change the model scene title");
    }
  }
  assert.equal(JSON.stringify(scene), before, "View switches preserve source evidence");
  const modeled = scene.entities.find(entity => activeModel(entity)), model = activeModel(modeled), originalPose = modeled.currentModelTransform || model.transform;
  const modelOptions = scenePlanOptions(scene, "model", true, images[0]);
  const previousModels = cadGeometry(planShapes(scene, modelOptions));
  const pose = { ...originalPose, position: originalPose.position.map((value, i) => value + (i === 0 ? 1 : 0)) };
  uiDocument = previewOperations(scene, [{ type: "setTransform", entityId: modeled.id, transform: pose }]); tree = renderWorkspace();
  assert.deepEqual(cadGeometry(planShapes(uiDocument, nodes(tree).find(node => node.type === CadView).props.geometryOptions)), expected, "Model edits cannot rewrite observed CAD evidence");
  assert.notDeepEqual(cadGeometry(planShapes(uiDocument, modelOptions)), previousModels, "The distinct model plan still follows edited model placement");
  assert.equal(JSON.stringify(scene), before, "Editing a preview does not mutate the source document");
  console.log(`${label}: ${expected.length} source contours stable across ${images.length} photos × 3 display modes; model edits remain separate`);
}
await checkCadModes(cadSwitchDocument, "Report CAD regression");
if (process.argv[2]) await checkCadModes(JSON.parse(await readFile(process.argv[2], "utf8")), "Frozen report CAD");
uiDocument=structuredClone(cadSwitchDocument);hooks.length=0;uiImageId='photo';
const part=uiDocument.entities.find(entity=>entity.id==='object');
part.parentEntityId='assembly';uiDocument.entities.unshift({id:'assembly',label:'Whole assembly',representations:[],activeModelRepresentationId:null,currentModelTransform:structuredClone(transform)});
uiSelection={entityId:'assembly',cameraId:'camera'};tree=renderWorkspace();
assert.equal(nodes(tree).some(node=>node.props['data-entity-id']==='object'),false,'The assembly begins as one expandable row');
assert.equal(spatialHost()[0].props.modelPreview.entityId,'assembly','An empty-residual assembly requests a real family preview');
nodes(tree).find(node=>node.props['aria-expanded']===false).props.onClick();tree=renderWorkspace();
assert.ok(nodes(tree).some(node=>node.props['data-entity-id']==='object'&&node.props['data-parent-entity-id']==='assembly'&&node.props.style.marginLeft===12),'The real part appears below and inside its assembly');
nodes(tree).find(node=>node.props['data-entity-id']==='object').children[0].props.onClick();tree=renderWorkspace();
assert.equal(uiSelection.entityId,'object');assert.equal(spatialHost()[0].props.modelPreview.entityId,'object','Selecting the part retains the part identity across model preview and shared selection');
const partPreviewKey=spatialHost()[0].props.modelPreview.requestKey;
part.currentModelTransform={...transform,coordinateFrameId:'unregistered'};tree=renderWorkspace();
assert.equal(spatialHost()[0].props.modelPreview,undefined);
assert.ok(nodes(tree).some(node=>node.children.includes('sceneModelWrongFrame')),'Wrong-frame geometry has a distinct reason rather than a false missing-model or loaded state');
assert.ok(partPreviewKey);
console.log(
  "report scene: 68-record unified evidence inventory, observed extents/scale, no duplicate table, linked no-geometry selection, mobile navigation, photo projection, floor semantics and one WebGL passed",
);
