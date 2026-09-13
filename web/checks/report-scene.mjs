// Run: node --experimental-strip-types web/checks/report-scene.mjs
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import ts from "typescript";
import * as math from "../src/viewer/native-math.ts";
import { isReferenceSurface } from "../src/scene-semantics.ts";
import { entityGeometryForLayer, observationsFor, jsonObject, planShapes } from "../src/core.ts";

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
const geometryOptions = { layer: "model", frameId: "f", showCandidates: true };
const document = { observations: [], coordinateFrames: [] };
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
  id: "object",
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
  kind: "observed_surface",
  placementState: "confirmed",
};
const observedOnly = {
  ...entity,
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
const measuredOnly = { id: "measured-only", representations: [], measurements: {
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
assert.equal(entityGeometryForLayer(floor, { ...geometryOptions, layer: "observed_surface" }).corners.length, 8, "Observed evidence is retained independently of pose overlays");
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
assert.equal(sceneAvailability(objectOnlyDocument, geometryOptions).spatialTitle, "sceneObjectModels",
  "Individual model assets must not imply an observed scene reconstruction");
assert.equal(sceneAvailability(objectOnlyDocument, geometryOptions).planEmpty, "sceneNoPlanGround");
assert.deepEqual(objectOnlyDocument, unchanged, "Presentation must preserve no-geometry entities");
const sceneDocument = structuredClone(objectOnlyDocument);
sceneDocument.entities.push({ id: "context", sourceContext: true, representations: [{ ...observed, kind: "point_cloud", assetId: "cloud" }] });
sceneDocument.assets.push({ id: "cloud" });
assert.equal(sceneAvailability(sceneDocument, geometryOptions).spatialTitle, "scene3D",
  "Confirmed registered observed geometry uses a neutral scene title, without asserting completeness");
sceneDocument.coordinateFrames[0].ground = { normal: [0, 0, 1] };
assert.equal(sceneAvailability(sceneDocument, geometryOptions).planEmpty, null, "Real same-frame model corners remain projectable");
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
const useState = (initial) => { const i = cursor++; if (!(i in hooks)) hooks[i] = initial; return [hooks[i], (next) => { const value = typeof next === "function" ? next(hooks[i]) : next; dirty ||= !Object.is(value, hooks[i]); hooks[i] = value; }]; };
const useRef = (initial) => { const i = cursor++; return hooks[i] ||= { current: initial }; };
const useEffect = (fn, deps) => { const i = cursor++, old = hooks[i]; if (!old || deps.some((value, n) => !Object.is(value, old[n]))) effects.push(fn); hooks[i] = deps; };
const React = { createElement: (type, props, ...children) => ({ type, props: props || {}, children: children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false) }) };
const SpatialView = () => {}, PlanView = () => {}, PhotoView = () => {}, PhotoAxes = () => {}, CadView = () => {};
const uiDocument = { ...objectOnlyDocument,
  assets: [{ id: "model-asset" }, { id: "photo", kind: "source_image" }, { id: "photo-2", kind: "source_image" }],
  cameras: [camera, { ...camera, id: "camera-2", imageId: "photo-2" }],
  coordinateFrames: [{ id: "f", ground: { normal: [0, 0, 1] } }],
  entities: [{ ...entity, observationRefs: ["observation"], representations: [rep, { ...observed, id: "surface" }] },
    { id: "no-geometry", label: "small button", observationRefs: ["observation-2"], representations: [] },
    { id: "background", sourceContext: true, representations: [] }],
  observations: [{ id: "observation", imageId: "photo" }, { id: "observation-2", imageId: "photo-2" }],
};
let uiSelection = { entityId: "object", cameraId: "camera" }, uiImageId = "photo", calls = [];
const inspector = React.createElement("div", { id: "inspector-content" }, "real host inspector");
const ui = vm.createContext({ React, useState, useRef, useEffect, useId: () => "workspace-check",
  useI18n: () => ({ t: (key) => key }), jsonObject, observationsFor, entityGeometryForLayer,
  photoOverlay, sceneAvailability, isReferenceSurface, SpatialView, PlanView, PhotoView, PhotoAxes, CadView,
  paneOrder: ["photo", "spatial", "cad", "plan"], paneNames: { photo: "scenePhoto", spatial: "scene3D", cad: "sceneCAD", plan: "scenePlan" }, noEdit: () => {},
  window: { document: { fullscreenElement: null, addEventListener() {}, removeEventListener() {} }, scrollTo() { throw Error("Selecting an object must not scroll the report"); } },
});
vm.runInContext(componentCode, ui);
const nodes = (root) => [root, ...root.children.filter((child) => typeof child === "object").flatMap(nodes)];
function renderWorkspace() {
  let tree;
  for (let n = 0; n < 5; n++) {
    cursor = 0; effects = []; dirty = false;
    tree = ui.ReportScene({ revision: { id: "revision", document: uiDocument }, selection: uiSelection, imageId: uiImageId,
      cameraId: uiSelection.cameraId, inspector,
      onSelect: (id) => { uiSelection = { ...uiSelection, entityId: id }; calls.push(id); },
      onCamera: (imageId, cameraId) => { uiImageId = imageId; uiSelection = { ...uiSelection, cameraId }; },
    });
    for (const effect of effects) effect();
    if (!dirty) break;
  }
  return tree;
}
let tree = renderWorkspace();
const rail = () => nodes(tree).find((node) => node.props.className === "report-scene-object-rail");
const switcher = () => nodes(tree).find((node) => node.props.className === "report-scene-view-switch");
const spatialHost = () => nodes(tree).filter((node) => node.type === SpatialView);
assert.equal(tree.props["data-view"], "quad");
assert.equal(nodes(rail()).filter((node) => node.type === "button").length, 2, "All non-context entities have permanent selectable rows, including no-geometry objects");
assert.equal(nodes(tree).some((node) => node.type === "details" && node.props.className === "report-scene-objects"), false, "Objects are not hidden in a bottom disclosure");
assert.ok(nodes(tree).includes(inspector), "The host inspector is rendered inside the right rail");
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
assert.ok(nodes(tree).filter((node) => [PlanView, CadView].includes(node.type)).every((node) => node.props.geometryOptions.layer === "observed_surface" && node.props.geometryOptions.frameId === "f"), "Both plans use the same selected layer and frame as the photo and 3D");
for (const section of ["objects", "inspector", "views"]) {
  const tab = nodes(tree).find((node) => node.type === "button" && node.props["aria-controls"] === `workspace-check-${section}`);
  tab.props.onClick(); tree = renderWorkspace();
  assert.equal(tree.props["data-mobile-section"], section);
  assert.equal(spatialHost().length, 1, "Mobile panel switches do not destroy the renderer");
}
tree.props.ref.current = { requestFullscreen: async () => { fullscreenRequests++; } };
await nodes(tree).find((node) => node.props.className === "report-scene-fullscreen").props.onClick();
assert.equal(fullscreenRequests, 1, "Only the separate fullscreen control requests browser fullscreen");
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
const css = await readFile(new URL("../src/report-scene.css", import.meta.url), "utf8");
assert.match(css, /grid-template-columns:\s*232px minmax\(0, 1fr\) 300px/);
assert.match(css, /\.report-scene-object-list\s*\{[^}]*overflow-y:\s*auto/);
assert.match(css, /\.report-scene-inspector-content\s*\{[^}]*overflow-y:\s*auto/);
console.log(
  "report scene: shared geometry, exact photo projection, floor semantics, permanent three rails, no-geometry selection, view/Esc/mobile/fullscreen handlers and one WebGL passed",
);
