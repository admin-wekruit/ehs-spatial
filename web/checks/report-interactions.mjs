// Run: node --experimental-strip-types web/checks/report-interactions.mjs
import assert from "node:assert/strict";
import {
  observationPolygons,
  insidePolygons,
  photoHits,
  originalPixel,
  sourceDimensions,
  sourceScale,
  planShapes,
} from "../src/core.ts";

const outer = [
  [0, 0],
  [99, 0],
  [99, 99],
  [0, 99],
];
const hole = [
  [20, 20],
  [80, 20],
  [80, 80],
  [20, 80],
];
const fence = {
  id: "fence-view",
  imageId: "photo",
  revision: 1,
  originalPixelBox: [0, 0, 100, 100],
  originalPixelPolygons: [outer, hole],
  polygonCoordinateConvention: "pixel_centers",
  boxConvention: "edges_xyxy_right_bottom_exclusive",
  fillRule: "evenodd",
};
const before = structuredClone(fence);
const contours = observationPolygons(fence);
assert.deepEqual(
  contours[0][0],
  [0.5, 0.5],
  "Stored pixel centres become SVG image-edge coordinates exactly once",
);
assert.deepEqual(observationPolygons(fence), contours);
assert.deepEqual(
  fence,
  before,
  "Projection must not mutate source observations",
);
for (const rings of [contours, contours.map((ring) => [...ring].reverse())]) {
  assert.equal(
    insidePolygons(rings, 10, 50),
    true,
    "Fence material is selectable",
  );
  assert.equal(
    insidePolygons(rings, 50, 50),
    false,
    "Even-odd inner ring is a hole regardless of winding",
  );
  assert.equal(
    insidePolygons(rings, 0.5, 30),
    true,
    "Outer contour boundary remains selectable",
  );
  assert.equal(
    insidePolygons(rings, 20.5, 50),
    true,
    "Hole contour boundary remains selectable",
  );
  assert.equal(
    insidePolygons(rings, 20.6, 50),
    false,
    "Point just inside the hole is not fence material",
  );
  assert.equal(insidePolygons(rings, 0.49, 30), false);
}
const button = {
  id: "button-view",
  revision: 1,
  imageId: "photo",
  originalPixelBox: [10, 40, 17, 47],
  boxConvention: "edges_xyxy_right_bottom_exclusive",
};
const document = {
  schemaVersion: 1,
  captureId: "capture",
  target: "scene",
  assets: [],
  annotations: [],
  cameras: [],
  coordinateFrames: [],
  entities: [
    {
      id: "fence",
      associationState: "association_pending",
      observationRefs: ["fence-view"],
    },
    {
      id: "button",
      associationState: "association_pending",
      observationRefs: ["button-view"],
    },
  ],
  observations: [fence, button],
};
const idsAt = (x, y) =>
  photoHits(document, "photo", x, y).map((hit) => hit.entity.id);
assert.deepEqual(
  idsAt(13, 43),
  ["button", "fence"],
  "The 49-pixel no-mesh button wins over its enclosing fence",
);
assert.deepEqual(
  idsAt(50, 50),
  [],
  "A contour hole must not become a solid bbox hit",
);
assert.deepEqual(
  idsAt(0.25, 30),
  [],
  "Pixels inside the bbox but outside its contour are not hits",
);
assert.deepEqual(idsAt(17, 43), ["fence"], "Right bbox edge is exclusive");
assert.deepEqual(idsAt(13, 47), ["fence"], "Bottom bbox edge is exclusive");
assert.deepEqual(
  idsAt(10, 40),
  ["button", "fence"],
  "Left/top bbox edges are included",
);
assert.deepEqual(photoHits(document, "another-photo", 13, 43), []);
assert.deepEqual(
  photoHits(
    {
      ...document,
      entities: document.entities.map((entity) => ({
        ...entity,
        visible: false,
      })),
    },
    "photo",
    13,
    43,
  ),
  [],
);

// A portrait image in a wide pane and a landscape image in a tall pane both
// preserve source pixels; padding must not select an image object.
assert.deepEqual(
  originalPixel(
    335,
    75,
    { left: 10, top: 20, width: 800, height: 400 },
    100,
    200,
  ),
  [12.5, 27.5],
);
assert.equal(
  originalPixel(
    300,
    75,
    { left: 10, top: 20, width: 800, height: 400 },
    100,
    200,
  ),
  null,
);
assert.deepEqual(
  originalPixel(
    135,
    445,
    { left: 10, top: 20, width: 400, height: 800 },
    200,
    100,
  ),
  [62.5, 62.5],
);
assert.equal(
  originalPixel(
    135,
    200,
    { left: 10, top: 20, width: 400, height: 800 },
    200,
    100,
  ),
  null,
);

// This canonical measurement has no generated mesh and no dimensionsNative:
// width/depth are local observations, not assumed XYZ world extents or metres.
const measured = {
  id: "measured-no-mesh",
  associationState: "confirmed",
  measurements: {
    coordinateFrameId: "native",
    widthNative: 2,
    depthNative: 3,
    groundHeightNative: 7,
    basis: {
      cornersNative: [
        [2, -4, 0.5],
        [5, -4, 0.5],
        [2, -2, 0.5],
        [5, -2, 0.5],
        [2, -4, 7.5],
        [5, -4, 7.5],
        [2, -2, 7.5],
        [5, -2, 7.5],
      ],
    },
  },
};
assert.deepEqual(sourceDimensions(measured), {
  groundHeight: 7,
  widthNative: 2,
  depthNative: 3,
  extentX: undefined,
  extentY: undefined,
  extentZ: undefined,
});
const unknown = sourceDimensions({
  id: "unknown",
  measurements: {
    widthNative: "2",
    depthNative: Infinity,
    groundHeightNative: null,
  },
});
assert.equal(unknown.widthNative, undefined);
assert.equal(unknown.depthNative, undefined);
assert.equal(unknown.groundHeight, undefined);
const measuredDocument = {
  ...document,
  entities: [measured],
  coordinateFrames: [
    {
      id: "native",
      convention: "opencv",
      scale: { status: "uncalibrated", nativeToMeters: null },
      ground: { normal: [0, 0, 2] },
    },
  ],
};
assert.deepEqual(sourceScale(measuredDocument, measured), {
  status: "uncalibrated",
  nativeToMeters: null,
});
const [shape] = planShapes(measuredDocument);
assert.equal(shape.entity.id, measured.id);
assert.deepEqual(shape.min, [2, 2]);
assert.deepEqual(shape.max, [4, 5]);
assert.equal(
  planShapes({
    ...measuredDocument,
    coordinateFrames: [
      { ...measuredDocument.coordinateFrames[0], ground: null },
    ],
  }).length,
  0,
);
assert.equal(
  planShapes({
    ...measuredDocument,
    entities: [
      {
        ...measured,
        measurements: {
          ...measured.measurements,
          coordinateFrameId: "unregistered",
        },
      },
    ],
  }).length,
  0,
);
assert.equal(
  planShapes({
    ...measuredDocument,
    entities: [
      { ...measured, measurements: { ...measured.measurements, basis: null } },
    ],
  }).length,
  0,
  "Dimensions alone cannot manufacture a located footprint",
);
assert.equal(
  planShapes({
    ...measuredDocument,
    entities: [
      {
        ...measured,
        measurements: {
          ...measured.measurements,
          basis: { cornersNative: [[1, 2, "3"]] },
        },
      },
    ],
  }).length,
  0,
);
// Frozen projected hulls are evidence for an exact representation + pose +
// plane, not permanent display geometry after a model or floor edit.
const identityPlane = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];
const placed = {
  id: "verified-footprint",
  associationState: "confirmed",
  observationRefs: [],
  currentModelTransform: {
    coordinateFrameId: "native",
    position: [0, 0, 0],
    quaternion: [0, 0, 0, 1],
    scale: [1, 1, 1],
  },
  representations: [{
    id: "mesh", kind: "generated_mesh", assetId: "mesh-asset",
    coordinateFrameId: "native", placementState: "confirmed",
    bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
    transform: {
      coordinateFrameId: "native", position: [0, 0, 0],
      quaternion: [0, 0, 0, 1], scale: [1, 1, 1],
    },
  }],
  measurements: { coordinateFrameId: "native" },
};
const triangle = [[-1, -1], [1, -1], [0, 1]];
const circle = Array.from({ length: 16 }, (_, i) => [
  Math.cos(i * Math.PI / 8), Math.sin(i * Math.PI / 8),
]);
const hullDocument = (points) => {
  const entity = structuredClone(placed);
  entity.measurements.projectedHull = {
    coordinateFrameId: "native", nativeToPlane: structuredClone(identityPlane),
    points: structuredClone(points),
    representationSnapshot: structuredClone(entity.representations),
    modelTransformSnapshot: structuredClone(entity.currentModelTransform),
  };
  return {
    ...measuredDocument, entities: [entity],
    reportEvidence: { plan: { coordinateFrameId: "native", nativeToFloor: structuredClone(identityPlane) } },
  };
};
for (const polygon of [triangle, circle]) {
  const scene = hullDocument(polygon);
  const snapshot = structuredClone(scene);
  assert.deepEqual(planShapes(scene)[0].polygon, polygon,
    "Verified triangular/circular hull must not collapse to its enclosing rectangle");
  assert.deepEqual(scene, snapshot, "Rendering never rewrites the evidence snapshot");
}
const moved = hullDocument(triangle);
moved.entities[0].currentModelTransform.position[0] = 10;
assert.deepEqual(planShapes(moved)[0].min, [9, -1]);
assert.deepEqual(planShapes(moved)[0].max, [11, 1]);
assert.equal(planShapes(moved)[0].polygon.length, 4,
  "A changed pose must derive the current model footprint, never reuse the old triangle");
const replaced = hullDocument(circle);
replaced.entities[0].representations[0].assetId = "replacement-mesh";
assert.equal(planShapes(replaced)[0].polygon.length, 4,
  "An asset change invalidates the frozen hull even when the old bounds remain");
const resized = hullDocument(triangle);
resized.entities[0].representations[0].bounds.max[0] = 3;
assert.equal(planShapes(resized)[0].max[0], 3);
assert.equal(planShapes(resized)[0].polygon.length, 4);
const rotated = hullDocument(triangle);
rotated.entities[0].currentModelTransform.quaternion = [0, 0, Math.sin(Math.PI / 8), Math.cos(Math.PI / 8)];
const diamond = planShapes(rotated)[0].polygon;
assert.equal(diamond.length, 4);
assert.ok(diamond.every(([x, y]) => Math.abs(x) < 1e-10 || Math.abs(y) < 1e-10),
  "Current rotated OBB projects to a diamond; min/max alone would invent four corner areas");
const tiltedFloor = hullDocument(triangle);
tiltedFloor.coordinateFrames = structuredClone(tiltedFloor.coordinateFrames);
tiltedFloor.coordinateFrames[0].ground.normal = [0, 1, 0];
assert.equal(planShapes(tiltedFloor)[0].polygon.length, 4,
  "A changed ground normal rejects the old plane and its saved footprint");
const otherPlaneFrame = hullDocument(triangle);
otherPlaneFrame.reportEvidence.plan.coordinateFrameId = "other-native-frame";
assert.equal(planShapes(otherPlaneFrame)[0].polygon.length, 4,
  "Identical matrices in different frames do not establish registration");
const otherHullFrame = hullDocument(triangle);
otherHullFrame.entities[0].measurements.projectedHull.coordinateFrameId = "other-native-frame";
assert.equal(planShapes(otherHullFrame)[0].polygon.length, 4);
const shiftedPlane = hullDocument(triangle);
shiftedPlane.reportEvidence.plan.nativeToFloor[0][3] = 1e-8;
assert.equal(planShapes(shiftedPlane)[0].polygon.length, 4,
  "Plane snapshots match exactly; even a small translation must not retain frozen coordinates");
assert.equal(planShapes(shiftedPlane)[0].min[0], -1 + 1e-8);
const sameChangedPlane = hullDocument(circle);
sameChangedPlane.reportEvidence.plan.nativeToFloor[0][3] = 7;
sameChangedPlane.entities[0].measurements.projectedHull.nativeToPlane[0][3] = 7;
assert.deepEqual(planShapes(sameChangedPlane)[0].polygon, circle,
  "A verified matching plane is used as recorded, with no second reprojection");
console.log(
  "report interactions: pixel-centre contours, even-odd holes/boundaries, nested small objects, letterboxing, native dimensions and exact-frame projected hull snapshot invalidation passed",
);
