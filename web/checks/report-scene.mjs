// Run: node --experimental-strip-types web/checks/report-scene.mjs
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import ts from "typescript";
import * as math from "../src/viewer/native-math.ts";
import { isReferenceSurface } from "../src/scene-semantics.ts";
import { modelGeometry, jsonObject, planShapes } from "../src/core.ts";

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
const names = ["entityGeometry", "photoOverlay", "sceneAvailability"];
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
  modelGeometry,
  isReferenceSurface,
  jsonObject,
  planShapes,
  colors: ["red", "green", "blue"],
});
vm.runInContext(executable, context);
const { entityGeometry, photoOverlay, sceneAvailability } = context;
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
const geometry = entityGeometry(entity, "model"),
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
assert.equal(photoOverlay(document, entity, camera, "point_cloud").corners.length, 8,
  "The display layer must not erase verified object bounds on photo/tree selection");
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
  JSON.stringify(entityGeometry(observedOnly, "observed_surface").corners),
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
assert.equal(entityGeometry(floor, "observed_surface").corners.length, 8, "Observed evidence is retained independently of pose overlays");
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
assert.equal(sceneAvailability(objectOnlyDocument).spatialTitle, "sceneObjectModels",
  "Individual model assets must not imply an observed scene reconstruction");
assert.equal(sceneAvailability(objectOnlyDocument).planEmpty, "sceneNoPlanGround");
assert.deepEqual(objectOnlyDocument, unchanged, "Presentation must preserve no-geometry entities");
const sceneDocument = structuredClone(objectOnlyDocument);
sceneDocument.entities.push({ id: "context", sourceContext: true, representations: [{ ...observed, kind: "point_cloud", assetId: "cloud" }] });
sceneDocument.assets.push({ id: "cloud" });
assert.equal(sceneAvailability(sceneDocument).spatialTitle, "scene3D",
  "Confirmed registered observed geometry uses a neutral scene title, without asserting completeness");
sceneDocument.coordinateFrames[0].ground = { normal: [0, 0, 1] };
assert.equal(sceneAvailability(sceneDocument).planEmpty, null, "Real same-frame model corners remain projectable");
sceneDocument.entities = [{ id: "unmodeled", representations: [] }];
assert.equal(sceneAvailability(sceneDocument).planEmpty, "sceneNoPlanProjection",
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
console.log(
  "report scene: exact photo projection, missing geometry, coordinate boundaries, observed immutability, readonly and one WebGL passed",
);
