import assert from "node:assert/strict";
import {
  eulerQuaternion,
  editableTransform,
  jobResultSummary,
  sourceDimensions,
  observationsFor,
  modelGeometry,
  modelTilt,
  originalPixel,
  photoHits,
  planShapes,
  previewOperations,
  quaternionEuler,
  representationInPhoto,
} from "../src/core.ts";
import type { Entity, Representation, SceneDocument } from "../src/types.ts";
const transform = {
  coordinateFrameId: "f",
  position: [0, 0, 0],
  quaternion: [0, 0, 0, 1],
  scale: [1, 1, 1],
} as const;
const entity = (id: string): Entity => ({
  id,
  label: id,
  observationRefs: [id],
  associationState: "confirmed",
  representations: [],
  currentModelTransform: null,
  measurements: {},
});
const doc: SceneDocument = {
  schemaVersion: 2,
  geometrySolutions: [], geometryBindings: {}, identityDecisions: [], identitySchemaVersion: 1,
  captureId: "capture",
  target: "scene",
  assets: [],
  annotations: [],
  cameras: [],
  coordinateFrames: [
    {
      id: "f",
      convention: "opencv",
      scale: { status: "uncalibrated", nativeToMeters: null },
      ground: null,
    },
  ],
  entities: [entity("fence"), entity("button")],
  observations: [
    {
      id: "fence",
      revision: 1,
      imageId: "photo",
      originalPixelBox: [0, 0, 100, 200],
      maskAssetId: null,
    },
    {
      id: "button",
      revision: 1,
      imageId: "photo",
      originalPixelBox: [40, 40, 47, 47],
      maskAssetId: null,
    },
  ],
};
assert.equal(
  photoHits(doc, "photo", 43, 43)[0].entity.id,
  "button",
  "49-pixel object wins over enclosing fence without a mesh",
);
assert.equal(
  photoHits(doc, "photo", 43, 43).length,
  2,
  "enclosing entity remains selectable",
);
const observed: Representation = {id: "observed", kind: "observed_surface", coordinateFrameId: "f",
  transform: {coordinateFrameId: "f", position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: [1, 1, 1]}, placementState: "confirmed"};
for (const sourceRefs of [
  [{imageId: "photo", observationId: "fence", revision: 1}],
  [{imageId: "photo", observationId: "button", revision: 0}],
  [{observationId: "button", revision: 0}],
]) {
  for (const observationId of [undefined, "button"])
    assert.equal(representationInPhoto(doc.entities[1], {...observed, sourceRefs}, "photo", doc.observations, observationId), false,
      "A matching photo never bypasses observation ownership or the recorded revision");
}
assert.equal(representationInPhoto(doc.entities[1], {...observed, sourceRefs: [{imageId: "photo"}]}, "photo", doc.observations), true,
  "An explicit image-only source still requires a current owned observation in that photo");
assert.equal(representationInPhoto({...doc.entities[1], observationRefs: []}, {...observed, sourceRefs: [{imageId: "photo"}]}, "photo", doc.observations), false);
const frozenObservations = doc.observations.map(o => ({...o, sourceRefs: [{assetId: "frozen", sourceRecordId: o.id}]}));
for (const sourceRefs of [[{imageId: "photo", assetId: "wrong", sourceRecordId: "button"}], [{imageId: "photo", assetId: "frozen", sourceRecordId: "fence"}]])
  assert.equal(representationInPhoto(doc.entities[1], {...observed, sourceRefs}, "photo", frozenObservations), false,
    "An image annotation cannot override the frozen asset and record pair");
assert.equal(representationInPhoto(doc.entities[1], {...observed, sourceRefs: [{assetId: "frozen", sourceRecordId: "button"}]}, "photo", frozenObservations), true);
assert.deepEqual(
  originalPixel(
    110,
    60,
    { left: 10, top: 10, width: 200, height: 100 },
    100,
    100,
  ),
  [50, 50],
);
assert.equal(
  originalPixel(
    12,
    60,
    { left: 10, top: 10, width: 200, height: 100 },
    100,
    100,
  ),
  null,
  "letterbox is not source pixels",
);
const hidden = previewOperations(doc, [
  { type: "setVisibility", entityId: "button", visible: false },
]);
assert.equal(photoHits(hidden, "photo", 43, 43)[0].entity.id, "fence");
assert.equal(doc.entities[1].visible, undefined);
const modeled = structuredClone(doc);
modeled.entities[1].activeModelRepresentationId = "rep";
modeled.entities[1].representations = [
  {
    id: "rep",
    kind: "primitive",
    placementState: "confirmed",
    coordinateFrameId: "f",
    assetId: null,
    transform: structuredClone(transform) as any,
    primitive: { kind: "box", dimensions: [2, 3, 4] },
  },
];
assert.deepEqual(
  [
    modelGeometry(modeled.entities[1])?.width,
    modelGeometry(modeled.entities[1])?.height,
  ],
  [2, 4],
);
assert.equal(
  modelTilt(modeled, modeled.entities[1]),
  null,
  "no ground means no claimed floor tilt",
);
assert.equal(
  planShapes(modeled).length,
  0,
  "no ground means no plan pretending to be floor",
);
modeled.coordinateFrames[0].ground = { normal: [0, 0, 1] };
assert.equal(planShapes(modeled).length, 0, "Grounded model bounds without a mesh projection are not a CAD contour");
// Scene reference geometry can come from a different source exposure than the
// entity's observed-layer CAD reference, while retaining exact source revision checks.
const referencePlan = structuredClone(doc), floor = entity("floor");
const plane = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
referencePlan.coordinateFrames[0].ground = {normal: [0, 0, 1]};
referencePlan.reportEvidence = {plan: {coordinateFrameId: "f", nativeToFloor: plane}};
referencePlan.observations = [
  {id: "source", revision: 1, imageId: "source-photo", originalPixelBox: [0, 0, 10, 10]},
  {id: "cad", revision: 1, imageId: "cad-photo", originalPixelBox: [0, 0, 10, 10]},
];
referencePlan.cameras = ["source-photo", "cad-photo"].map(imageId => ({
  id: imageId, imageId, coordinateFrameId: "f", width: 10, height: 10,
  K: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], cameraToWorld: structuredClone(plane),
})) as SceneDocument["cameras"];
referencePlan.geometryBindings = Object.fromEntries(referencePlan.cameras.map(camera =>
  [camera.imageId, {geometrySolutionId: "solution", cameraId: camera.id}]));
floor.observationRefs = ["source", "cad"];
floor.cadReference = {referenceImageId: "cad-photo", source: "explicit_reference_image", status: "resolved", sourceRefs: [{observationId: "cad", revision: 1}]};
const reference = {id: "reference", kind: "observed_surface", sourceKind: "observed_reference_surface",
  assetId: "surface", coordinateFrameId: "f", transform: structuredClone(transform),
  bounds: {min: [0, 0, 0], max: [2, 2, 0]}, placementState: "confirmed",
  sourceRefs: [{observationId: "source", revision: 1, imageId: "source-photo"}],
  planProjection: {methodVersion: "indexed-mesh-triangle-union-v1", coordinateFrameId: "f",
    assetId: "surface", assetSha256: "surface-hash", imageId: "source-photo", observationId: "source", observationRevision: 1,
    transformSnapshot: structuredClone(transform), groundNormalSnapshot: [0, 0, 1], nativeToPlane: plane,
    polygons: [{exterior: [[0, 0], [2, 0], [2, 2], [0, 0]], holes: []}], lines: []}};
floor.representations = [reference] as any;
referencePlan.entities = [floor]; referencePlan.assets = [{id: "surface", sha256: "surface-hash"}] as any;
const referenceOptions = {scope: "scene", layer: "model", frameId: "f"} as const;
assert.equal(planShapes(referencePlan, referenceOptions).length, 1, "Model reference surface uses its own current source photo");
referencePlan.observations[0].revision = 2;
assert.equal(planShapes(referencePlan, referenceOptions).length, 0, "Stale reference-source revision remains rejected");
referencePlan.observations[0].revision = 1;
reference.planProjection.imageId = "wrong-photo";
assert.equal(planShapes(referencePlan, referenceOptions).length, 0, "Reference projection must still match its own source photo");
reference.planProjection.imageId = "source-photo";
const ordinaryPlan = structuredClone(referencePlan), ordinary = ordinaryPlan.entities[0].representations![0];
delete ordinary.sourceKind;
ordinary.sourceRefs = [{observationId: "source", revision: 1}, {observationId: "cad", revision: 1}];
assert.equal(planShapes(ordinaryPlan, {...referenceOptions, layer: "observed_surface"}).length, 0,
  "Ordinary observed geometry cannot project a different photo than its CAD reference");
ordinaryPlan.entities[0].cadReference = {referenceImageId: "source-photo", source: "explicit_reference_image", status: "resolved", sourceRefs: [{observationId: "source", revision: 1}]};
assert.equal(planShapes(ordinaryPlan, {...referenceOptions, layer: "observed_surface"}).length, 1,
  "Ordinary observed geometry remains visible for its exact source photo");
const edited = previewOperations(modeled, [
  {
    type: "setTransform",
    entityId: "button",
    transform: {
      ...transform,
      scale: [2, 1, 1],
      quaternion: eulerQuaternion([30, 0, 0]),
    },
  },
]);
assert.equal(modelGeometry(edited.entities[1])?.width, 4);
assert.ok(Math.abs(modelTilt(edited, edited.entities[1])! - 30) < 1e-6);
assert.deepEqual(
  edited.observations,
  modeled.observations,
  "model edits never rewrite source observations",
);
assert.equal(
  modelGeometry(modeled.entities[1])?.width,
  2,
  "preview never mutates base revision",
);
const before = JSON.stringify(modeled);
previewOperations(modeled, [
  {
    type: "setPrimitive",
    entityId: "button",
    primitive: { kind: "box", dimensions: [5, 3, 4] },
  },
]);
assert.equal(JSON.stringify(modeled), before);
const angles = [15, 22, -45] as [number, number, number];
assert.ok(
  quaternionEuler(eulerQuaternion(angles)).every(
    (n, i) => Math.abs(n - angles[i]) < 1e-6,
  ),
);
console.log(
  "PASS: photo pixel mapping, 49px object selection, immutable edits, measured/model separation, no invented floor, common-frame projection and pose math",
);

// Asset identity is durable; an access URL is not. A later view must resolve
// a fresh signed URL rather than retain one from the first download forever.
const { asset: resolveAssetMetadata } = await import("../src/api.ts");
const originalFetch = globalThis.fetch;
let lookups = 0;
globalThis.fetch = async () =>
  new Response(
    JSON.stringify({
      id: "asset-1",
      url: `https://assets.example/mesh.glb?signature=${++lookups}`,
      sha256: "sha",
      sizeBytes: 1,
      mediaType: "model/gltf-binary",
      metadata: {},
    }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  );
Object.defineProperty(globalThis, "location", {
  value: { origin: "https://studio.example" },
  configurable: true,
});
try {
  const first = resolveAssetMetadata("asset-1");
  assert.equal(resolveAssetMetadata("asset-1"), first);
  assert.match((await first).url, /signature=1$/);
  assert.match((await resolveAssetMetadata("asset-1")).url, /signature=2$/);
  assert.equal(lookups, 2);
} finally {
  globalThis.fetch = originalFetch;
  Reflect.deleteProperty(globalThis, "location");
}
console.log("asset identity and expiring access URL checks passed");

const emptyEntity: Entity = {
  id: "no-geometry",
  associationState: "association_pending",
};
assert.deepEqual(observationsFor(doc, emptyEntity), []);
assert.equal(editableTransform(emptyEntity), null);
assert.equal(modelGeometry(emptyEntity), null);
assert.equal(sourceDimensions(emptyEntity).groundHeight, undefined);
assert.equal(
  sourceDimensions({
    ...emptyEntity,
    measurements: { groundHeightNative: "2.4" },
  }).groundHeight,
  undefined,
  "extension JSON must not coerce a string into a measured height",
);
const exportResult = jobResultSummary({
  status: "incomplete",
  error: { code: "partial_export" },
  errors: [{ code: "unplaced_entity" }],
  timings: { workerSeconds: 1.25 },
  assets: [{ id: "export-asset", metadata: { name: "scene.blend" } }],
});
assert.deepEqual(exportResult, {
  errors: ["partial_export", "unplaced_entity"],
  assets: [{ id: "export-asset", name: "scene.blend" }],
  workerSeconds: 1.25,
});
assert.deepEqual(
  jobResultSummary({
    assets: [null, { id: 3 }],
    errors: "unknown",
    timings: { workerSeconds: "1" },
  }),
  { errors: [], assets: [], workerSeconds: undefined },
);
console.log(
  "generated DTO nullable geometry and real worker result checks passed",
);

// Component-linked records project the exact same parts used by the 3D preview.
const compositePlan = structuredClone(referencePlan);
const part = compositePlan.entities[0]; part.id = "part";
const model = part.representations![0];
model.kind = "generated_mesh"; model.sourceKind = undefined;
part.activeModelRepresentationId = model.id;
const group = entity("composite"); group.observationRefs = ["boundary"];
compositePlan.entities.push(group);
compositePlan.observations.push({id:"boundary", revision:1, imageId:"source-photo", originalPixelBox:[0,0,10,10]});
compositePlan.annotations = [{id:"mapping",kind:"observation_component_mapping",entityId:"composite",
  representationType:"composite_source_evidence",independentObject:false,
  unresolvedBoundaryObservationId:"boundary",unresolvedBoundaryObservationRevision:1,
  targets:[{entityId:"part",activeModelRepresentationId:model.id,observationId:"source",observationRevision:1}]}];
const linkedPlans = planShapes(compositePlan,referenceOptions);
assert.equal(linkedPlans.length,2);
assert.deepEqual(linkedPlans[1].polygons,linkedPlans[0].polygons,"Composite CAD keeps exact component outlines");
assert.deepEqual(linkedPlans[1].representationIds,[model.id]);
model.planProjection = {...model.planProjection as Record<string, unknown>, assetSha256:"stale"};
assert.equal(planShapes(compositePlan,referenceOptions).length,0,"An invalid component projection cannot produce composite CAD");
