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
} from "../src/core.ts";
import type { Entity, SceneDocument } from "../src/types.ts";
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
