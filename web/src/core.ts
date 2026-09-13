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
      const b = observation.originalPixelBox;
      if (observation.imageId !== imageId || !b) continue;
      const polygons = observationPolygons(observation);
      if (
        x >= b[0] &&
        x < b[2] &&
        y >= b[1] &&
        y < b[3] &&
        (!polygons.length || insidePolygons(polygons, x, y))
      )
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
  // Source contours are pixel centres; SVG and pointer positions use image edges.
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
    .map((p) => p.map(([x, y]) => [x + 0.5, y + 0.5]));
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
export function planShapes(document: SceneDocument) {
  const frame = document.coordinateFrames.find(
    (f) =>
      Array.isArray(f.ground?.normal) && Math.hypot(...f.ground.normal) > 1e-8,
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
      const model = modelGeometry(entity);
      const observed = observedGeometry(entity, frame.id);
      const basis = jsonObject(entity.measurements?.basis)?.cornersNative;
      const measured = entity.measurements?.coordinateFrameId === frame.id &&
        Array.isArray(basis) && basis.length > 0 && basis.every(isVec3)
        ? { corners: basis, frameId: frame.id } : null;
      const geometry = model?.frameId === frame.id ? model : measured || observed;
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
      const canReuse =
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
          projectionSource: useSavedHull ? "saved_hull" as const : geometry === model
            ? "model_bounds" as const : geometry === measured ? "observed_measurement" as const : "observed_bounds" as const,
          representationIds: geometry === observed ? observed!.representationIds : [],
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
