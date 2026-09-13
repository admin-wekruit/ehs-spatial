import {
  boundsCorners,
  cross,
  dot,
  point,
  transformMatrix,
  unit,
} from "./viewer/native-math.ts";
import type {
  Entity,
  Job,
  Observation,
  Operation,
  PublicationSummary,
  Representation,
  SceneDocument,
  Transform,
  Vec3,
} from "./types.ts";

export function groupPublications(items: PublicationSummary[]) {
  const groups = new Map<string, PublicationSummary[]>();
  // Preserve the API's PostgreSQL timestamp ordering, including microseconds.
  // String sorting and JavaScript Date can both select the wrong snapshot here.
  for (const item of items) {
    const group = groups.get(item.projectId) || [];
    group.push(item);
    groups.set(item.projectId, group);
  }
  return [...groups.values()];
}

export const refs = (entity: Entity) => entity.observationRefs || [];
export function observationsFor(document: SceneDocument, entity: Entity) {
  const ids = new Set(refs(entity));
  return document.observations.filter((o) => ids.has(o.id));
}
export function photoHits(
  document: SceneDocument,
  imageId: string,
  x: number,
  y: number,
) {
  const hits: { entity: Entity; observation: Observation; area: number }[] = [];
  for (const entity of document.entities) {
    if (entity.visible === false) continue;
    for (const observation of observationsFor(document, entity)) {
      if (observation.imageId !== imageId) continue;
      const polygons = observationPolygons(observation);
      const points = polygons.flat();
      const b = points.length ? points.reduce((bounds, [px, py]) => [
        Math.min(bounds[0], px), Math.min(bounds[1], py),
        Math.max(bounds[2], px), Math.max(bounds[3], py),
      ], [Infinity, Infinity, -Infinity, -Infinity]) : observation.originalPixelBox;
      if (!b) continue;
      if (polygons.length ? insidePolygons(polygons, x, y)
        : x >= b[0] && x < b[2] && y >= b[1] && y < b[3])
        hits.push({ entity, observation, area: (b[2] - b[0]) * (b[3] - b[1]) });
    }
  }
  return hits.sort(
    (a, b) => a.area - b.area || a.entity.id.localeCompare(b.entity.id),
  );
}
export function observationPolygons(observation: Observation): number[][][] {
  const raw = observation.originalPixelPolygons;
  if (!Array.isArray(raw)) return [];
  // Legacy contours use pixel centres; new mask unions already use image edges.
  const offset = observation.polygonCoordinateConvention === "pixel_edges" ? 0 : 0.5;
  return raw
    .filter(
      (p): p is number[][] =>
        Array.isArray(p) &&
        p.length >= 3 &&
        p.every(
          (v) =>
            Array.isArray(v) &&
            v.length === 2 &&
            v.every((n) => typeof n === "number" && Number.isFinite(n)),
        ),
    )
    .map((p) => p.map(([x, y]) => [x + offset, y + offset]));
}
export function insidePolygons(polygons: number[][][], x: number, y: number) {
  let inside = false;
  for (const polygon of polygons)
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const [ax, ay] = polygon[j],
        [bx, by] = polygon[i];
      if (
        Math.abs((x - ax) * (by - ay) - (y - ay) * (bx - ax)) < 1e-8 &&
        x >= Math.min(ax, bx) &&
        x <= Math.max(ax, bx) &&
        y >= Math.min(ay, by) &&
        y <= Math.max(ay, by)
      )
        return true;
      if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax)
        inside = !inside;
    }
  return inside;
}
export function planHits(shapes: ReturnType<typeof planShapes>, x: number, y: number) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return [];
  const area = (polygon: number[][]) => Math.abs(polygon.reduce((sum, p, i) => {
    const q = polygon[(i + 1) % polygon.length];
    return sum + p[0] * q[1] - q[0] * p[1];
  }, 0)) / 2;
  return shapes.filter((shape) => insidePolygons([shape.polygon], x, y))
    .sort((a, b) => area(a.polygon) - area(b.polygon) || a.entity.id.localeCompare(b.entity.id));
}
export function originalPixel(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  width: number,
  height: number,
) {
  const factor = Math.min(rect.width / width, rect.height / height),
    left = rect.left + (rect.width - width * factor) / 2,
    top = rect.top + (rect.height - height * factor) / 2;
  const x = (clientX - left) / factor,
    y = (clientY - top) / factor;
  return x < 0 || y < 0 || x > width || y > height
    ? null
    : ([x, y] as [number, number]);
}
export function editableTransform(entity: Entity): Transform | null {
  const model = (entity.representations || []).find(
    (r) => r.kind === "generated_mesh" || r.kind === "primitive",
  );
  return model ? entity.currentModelTransform || model.transform : null;
}
export function eulerQuaternion([x, y, z]: Vec3): [
  number,
  number,
  number,
  number,
] {
  const [a, b, c] = [x, y, z].map((n) => (n * Math.PI) / 360),
    [cx, cy, cz] = [a, b, c].map(Math.cos),
    [sx, sy, sz] = [a, b, c].map(Math.sin);
  return [
    sx * cy * cz - cx * sy * sz,
    cx * sy * cz + sx * cy * sz,
    cx * cy * sz - sx * sy * cz,
    cx * cy * cz + sx * sy * sz,
  ];
}
export function quaternionEuler([x, y, z, w]: number[]): Vec3 {
  return [
    Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
    Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x)))),
    Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
  ].map((a) => (a * 180) / Math.PI) as Vec3;
}
export function previewOperations(
  document: SceneDocument,
  operations: Operation[],
): SceneDocument {
  const next = structuredClone(document);
  for (const op of operations) {
    const entity = next.entities.find((e) => e.id === op.entityId);
    if (op.type === "setTransform" && entity) {
      entity.currentModelTransform = structuredClone(
        (op.transform || {
          coordinateFrameId: op.coordinateFrameId,
          position: op.position,
          quaternion: op.quaternion,
          scale: op.scale,
        }) as Transform,
      );
      (entity.representations || [])
        .filter((r) => r.kind === "generated_mesh" || r.kind === "primitive")
        .forEach(
          (r) => (r.transform = structuredClone(entity.currentModelTransform!)),
        );
    } else if (op.type === "setVisibility" && entity)
      entity.visible = op.visible as boolean;
    else if (op.type === "setLabel" && entity)
      entity.label = op.label as string;
    else if (op.type === "setMaterial" && entity)
      entity.material = op.material as Record<string, unknown>;
    else if (op.type === "setPrimitive" && entity) {
      let rep = (entity.representations || []).find(
        (r) => r.kind === "primitive",
      );
      if (!rep) {
        const transform = op.transform as Transform;
        rep = {
          id: "preview:" + entity.id,
          kind: "primitive",
          assetId: null,
          coordinateFrameId: transform.coordinateFrameId,
          transform,
          placementState: "unconfirmed",
        };
        (entity.representations ??= []).push(rep);
        entity.currentModelTransform = transform;
      }
      rep.primitive = structuredClone(op.primitive as Record<string, unknown>);
    } else if (op.type === "addCoordinateFrame")
      next.coordinateFrames.push(
        structuredClone(op.frame as SceneDocument["coordinateFrames"][number]),
      );
    else if (op.type === "addEntity")
      next.entities.push(structuredClone(op.entity as Entity));
    else if (op.type === "removeEntity")
      next.entities = next.entities.filter((e) => e.id !== op.entityId);
  }
  return next;
}
export function sourceDimensions(entity: Entity) {
  const m = entity.measurements || {},
    d = m.dimensionsNative;
  return {
    groundHeight: finiteNumber(m.groundHeightNative),
    widthNative: finiteNumber(m.widthNative),
    depthNative: finiteNumber(m.depthNative),
    extentX: Array.isArray(d) ? finiteNumber(d[0]) : undefined,
    extentY: Array.isArray(d) ? finiteNumber(d[1]) : undefined,
    extentZ: Array.isArray(d) ? finiteNumber(d[2]) : undefined,
  };
}
export function sourceScale(document: SceneDocument, entity: Entity) {
  const id =
    (entity.representations || [])[0]?.coordinateFrameId ||
    entity.currentModelTransform?.coordinateFrameId ||
    entity.measurements?.coordinateFrameId;
  return document.coordinateFrames.find((f) => f.id === id)?.scale || null;
}

export function modelGeometry(entity: Entity) {
  const rep = (entity.representations || []).find(
      (r) => r.kind === "primitive" || r.kind === "generated_mesh",
    ),
    transform = editableTransform(entity);
  if (!rep || !transform) return null;
  const spec = rep.primitive;
  let dims: Vec3 | undefined;
  if (spec?.kind === "box" && isVec3(spec.dimensions)) dims = spec.dimensions;
  else if (spec?.kind === "cylinder") {
    const radius = finiteNumber(spec.radius),
      height = finiteNumber(spec.height);
    if (radius !== undefined && height !== undefined)
      dims = [radius * 2, radius * 2, height];
  } else if (rep.bounds)
    dims = [
      rep.bounds.max[0] - rep.bounds.min[0],
      rep.bounds.max[1] - rep.bounds.min[1],
      rep.bounds.max[2] - rep.bounds.min[2],
    ];
  if (!dims || !dims.every((n) => Number.isFinite(n) && n > 0)) return null;
  const bounds = rep.bounds || {
      min: dims.map((n) => -n / 2),
      max: dims.map((n) => n / 2),
    },
    matrix = transformMatrix(transform);
  return {
    width: dims[0] * transform.scale[0],
    depth: dims[1] * transform.scale[1],
    height: dims[2] * transform.scale[2],
    corners: boundsCorners(bounds).map((p) => point(matrix, p)),
    frameId: transform.coordinateFrameId,
  };
}
export function observedGeometry(entity: Entity, frameId: string) {
  const representations = (entity.representations || []).filter((rep) =>
    ["observed_surface", "point_cloud"].includes(rep.kind) &&
    rep.sourceValidity !== "stale" &&
    rep.placementState === "confirmed" && !!rep.assetId &&
    rep.coordinateFrameId === frameId && rep.transform.coordinateFrameId === frameId &&
    isVec3(rep.bounds?.min) && isVec3(rep.bounds?.max) &&
    rep.bounds.max.every((value, index) => value >= rep.bounds!.min[index]));
  const corners = representations.flatMap((rep) =>
    boundsCorners(rep.bounds!).map((p) => point(transformMatrix(rep.transform), p)));
  return corners.length && corners.every(isVec3)
    ? { corners, frameId, representationIds: representations.map((rep) => rep.id) }
    : null;
}
export type GeometryLayer = "model" | "observed_surface" | "point_cloud";
export type GeometryOptions = { layer: GeometryLayer; frameId: string; showCandidates?: boolean };

export function representationAvailable(entity: Entity, rep: Representation, frameId: string | null, showCandidates = false) {
  const modeled = ["generated_mesh", "primitive"].includes(rep.kind);
  const transform = modeled ? entity.currentModelTransform || rep.transform : rep.transform;
  return entity.visible !== false && rep.sourceValidity !== "stale" && !!frameId &&
    rep.coordinateFrameId === frameId && transform?.coordinateFrameId === frameId &&
    (rep.placementState === "confirmed" || showCandidates &&
      ["requires_alignment_confirmation", "imported_proposal"].includes(rep.placementReason || ""));
}

/** One located geometry choice for source-photo axes, 3D bounds and plan views. */
export function entityGeometryForLayer(entity: Entity, { layer, frameId, showCandidates = true }: GeometryOptions) {
  if (entity.sourceContext || entity.visible === false) return null;
  const representations = (entity.representations || []).filter((rep) =>
    representationAvailable(entity, rep, frameId, showCandidates) &&
    (rep.kind === "primitive" ? !!rep.primitive : !!rep.assetId));
  const identity: Transform = {coordinateFrameId: frameId, position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: [1, 1, 1]};
  function fromRepresentations(reps: Representation[], geometryKind: "model" | "observed" | "point_cloud") {
    const geometries = reps.flatMap((rep) => {
      const transform = geometryKind === "model" ? entity.currentModelTransform || rep.transform : rep.transform;
      const bounds = rep.bounds;
      const validBounds = isVec3(bounds?.min) && isVec3(bounds?.max) && bounds.max.every((value, k) => value >= bounds.min[k]);
      const corners = validBounds
        ? boundsCorners(bounds!).map((p) => point(transformMatrix(transform), p))
        : rep.kind === "primitive" ? modelGeometry({...entity, representations: [rep]})?.corners : null;
      return corners?.length && corners.every(isVec3) ? [{rep, transform, corners}] : [];
    });
    if (!geometries.length) return null;
    const all = geometries.flatMap((g) => g.corners);
    const corners = geometries.length === 1 ? all : boundsCorners({
      min: [0, 1, 2].map((k) => Math.min(...all.map((p) => p[k]))),
      max: [0, 1, 2].map((k) => Math.max(...all.map((p) => p[k]))),
    });
    return {corners, transform: geometries[0].transform, frameId,
      axisSpace: geometryKind === "model" ? "local" as const : "native" as const,
      geometryKind, representationIds: geometries.map((g) => g.rep.id)};
  }
  if (layer === "model") {
    const model = fromRepresentations(representations.filter((r) => ["generated_mesh", "primitive"].includes(r.kind)), "model");
    if (model) return model;
  }
  if (layer === "point_cloud") {
    const cloud = fromRepresentations(representations.filter((r) => r.kind === "point_cloud"), "point_cloud");
    if (cloud) return cloud;
  }
  const observed = fromRepresentations(representations.filter((r) => r.kind === "observed_surface"), "observed");
  if (observed) return observed;
  const measurements = entity.measurements || {};
  const basis = jsonObject(measurements.basis)?.cornersNative;
  const bounds = jsonObject(measurements.observedBounds);
  const measuredBounds = bounds?.coordinateFrameId === frameId && bounds.source === "observed_measurement" &&
    bounds.sourceValidity !== "stale" && Array.isArray(bounds.sourceRefs) && bounds.sourceRefs.length > 0 &&
    isVec3(bounds.min) && isVec3(bounds.max) && bounds.max.every((value, k) => value >= (bounds.min as Vec3)[k]);
  // A sparse observed point set can locate a range without enough neighbouring
  // pixels to form triangles. Its source-bound bounds remain usable evidence.
  const corners = Array.isArray(basis) && basis.length === 8 && basis.every(isVec3)
    ? basis : measuredBounds ? boundsCorners({min: bounds!.min as Vec3, max: bounds!.max as Vec3}) : null;
  if (measurements.coordinateFrameId === frameId && measurements.sourceValidity !== "stale" &&
      corners)
    return {corners, transform: identity, frameId, axisSpace: "native" as const,
      geometryKind: "observed_measurement" as const, representationIds: [] as string[]};
  return null;
}
export function modelTilt(document: SceneDocument, entity: Entity) {
  const transform = editableTransform(entity);
  if (!transform) return null;
  const normal = document.coordinateFrames.find(
    (f) => f.id === transform.coordinateFrameId,
  )?.ground?.normal;
  if (
    !Array.isArray(normal) ||
    normal.length !== 3 ||
    Math.hypot(...normal) < 1e-8
  )
    return null;
  const matrix = transformMatrix(transform),
    axis = unit([matrix[8], matrix[9], matrix[10]]);
  return (
    (Math.acos(Math.max(-1, Math.min(1, Math.abs(dot(axis, unit(normal)))))) *
      180) /
    Math.PI
  );
}
export function planShapes(document: SceneDocument, options: Partial<GeometryOptions> = {}) {
  const frame = document.coordinateFrames.find(
    (f) =>
      (!options.frameId || f.id === options.frameId) &&
      Array.isArray(f.ground?.normal) && f.ground.normal.every(Number.isFinite) && Math.hypot(...f.ground.normal) > 1e-8,
  );
  const normal = frame?.ground?.normal;
  if (!frame || !normal) return [];
  const n = unit(normal),
    x = unit(cross(Math.abs(n[0]) < 0.8 ? [1, 0, 0] : [0, 1, 0], n)),
    y = cross(n, x);
  const evidencePlan = jsonObject(jsonObject(document.reportEvidence)?.plan);
  const rawPlane = evidencePlan?.nativeToFloor;
  const plane =
    evidencePlan?.coordinateFrameId === frame.id &&
    Array.isArray(rawPlane) &&
    rawPlane.length === 4 &&
    rawPlane.every(
      (row) =>
        Array.isArray(row) &&
        row.length === 4 &&
        row.every((v) => typeof v === "number" && Number.isFinite(v)),
    ) &&
    Math.abs(Math.abs(dot(unit(rawPlane[2].slice(0, 3)), n)) - 1) < 1e-6
      ? (rawPlane as number[][])
      : null;
  const project = (p: number[]) =>
    plane
      ? [dot(p, plane[0]) + plane[0][3], dot(p, plane[1]) + plane[1][3]]
      : [dot(p, x), dot(p, y)];
  return document.entities
    .filter((e) => e.visible !== false && !e.sourceContext)
    .flatMap((entity) => {
      const geometry = entityGeometryForLayer(entity, {layer: options.layer || "model", frameId: frame.id, showCandidates: options.showCandidates});
      if (!geometry) return [];
      const frameId = geometry?.frameId;
      const corners = geometry?.corners;
      if (
        frameId !== frame.id ||
        !Array.isArray(corners) ||
        !corners.length ||
        !corners.every(isVec3)
      )
        return [];
      const saved = jsonObject(entity.measurements?.projectedHull);
      const snapshot = saved?.representationSnapshot;
      // A frozen hull with both model and observed sources is ambiguous. Recompute
      // from the chosen layer unless its exact representation IDs are declared.
      const sameGeometry = saved?.representationIds !== undefined || saved?.geometryKind !== undefined
        ? saved?.geometryKind === geometry.geometryKind && sameJSON(saved?.representationIds, geometry.representationIds)
        : Array.isArray(snapshot) && snapshot.length > 0 && snapshot.every((rep) =>
          rep && typeof rep === "object" && geometry.representationIds.includes(rep.id) &&
          (geometry.geometryKind === "model" ? ["generated_mesh", "primitive"].includes(rep.kind)
            : geometry.geometryKind === "point_cloud" ? rep.kind === "point_cloud" : rep.kind === "observed_surface"));
      const canReuse =
        sameGeometry &&
        saved?.coordinateFrameId === frame.id &&
        plane &&
        sameJSON(saved.nativeToPlane, plane) &&
        sameJSON(saved.representationSnapshot, entity.representations) &&
        sameJSON(saved.modelTransformSnapshot, entity.currentModelTransform);
      const useSavedHull =
        !!canReuse &&
        Array.isArray(saved.points) &&
        saved.points.length >= 3 &&
        saved.points.every(
          (p) =>
            Array.isArray(p) &&
            p.length === 2 &&
            p.every((v) => typeof v === "number" && Number.isFinite(v)),
        );
      const ps = useSavedHull
          ? (saved!.points as number[][])
          : convexHull2D(corners.map(project));
      return [
        {
          entity,
          polygon: ps,
          coordinateFrameId: frame.id,
          projectionSource: useSavedHull ? "saved_hull" as const : geometry.geometryKind === "model"
            ? "model_bounds" as const : geometry.geometryKind === "observed_measurement" ? "observed_measurement" as const : "observed_bounds" as const,
          geometryKind: geometry.geometryKind,
          representationIds: geometry.representationIds,
          min: [
            Math.min(...ps.map((p) => p[0])),
            Math.min(...ps.map((p) => p[1])),
          ],
          max: [
            Math.max(...ps.map((p) => p[0])),
            Math.max(...ps.map((p) => p[1])),
          ],
        },
      ];
    });
}

function sameJSON(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (!a || !b || typeof a !== "object" || typeof b !== "object") return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const left = Object.keys(a),
    right = Object.keys(b);
  return (
    left.length === right.length &&
    left.every(
      (k) =>
        Object.hasOwn(b, k) &&
        sameJSON(
          (a as Record<string, unknown>)[k],
          (b as Record<string, unknown>)[k],
        ),
    )
  );
}
function convexHull2D(points: number[][]) {
  const sorted = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const turn = (a: number[], b: number[], c: number[]) =>
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const half = (values: number[][]) => {
    const out: number[][] = [];
    for (const p of values) {
      while (
        out.length > 1 &&
        turn(out[out.length - 2], out[out.length - 1], p) <= 0
      )
        out.pop();
      out.push(p);
    }
    return out;
  };
  return [
    ...half(sorted).slice(0, -1),
    ...half(sorted.slice().reverse()).slice(0, -1),
  ];
}

// Extension JSON is intentionally open in OpenAPI; narrow it before rendering.
export function jsonObject(
  value: unknown,
): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}
export function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}
function isVec3(value: unknown): value is Vec3 {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every((n) => finiteNumber(n) !== undefined)
  );
}

export function jobResultSummary(result: Job["result"]) {
  const failures = [
    result?.error,
    ...(Array.isArray(result?.errors) ? result.errors : []),
  ];
  const errors = failures.flatMap((failure) => {
    const code = jsonObject(failure)?.code;
    return typeof code === "string" ? [code] : [];
  });
  const assets = (Array.isArray(result?.assets) ? result.assets : []).flatMap(
    (value) => {
      const asset = jsonObject(value);
      if (typeof asset?.id !== "string") return [];
      const name = jsonObject(asset.metadata)?.name;
      return [
        { id: asset.id, name: typeof name === "string" ? name : undefined },
      ];
    },
  );
  return {
    errors,
    assets,
    workerSeconds: finiteNumber(jsonObject(result?.timings)?.workerSeconds),
  };
}
