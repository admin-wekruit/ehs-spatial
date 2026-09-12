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
  SceneDocument,
  Transform,
  Vec3,
} from "./types.ts";

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
      if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3])
        hits.push({ entity, observation, area: (b[2] - b[0]) * (b[3] - b[1]) });
    }
  }
  return hits.sort(
    (a, b) => a.area - b.area || a.entity.id.localeCompare(b.entity.id),
  );
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
    extentX: Array.isArray(d) ? finiteNumber(d[0]) : undefined,
    extentY: Array.isArray(d) ? finiteNumber(d[1]) : undefined,
    extentZ: Array.isArray(d) ? finiteNumber(d[2]) : undefined,
  };
}
export function sourceScale(document: SceneDocument, entity: Entity) {
  const id =
    (entity.representations || [])[0]?.coordinateFrameId ||
    entity.currentModelTransform?.coordinateFrameId;
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
  return document.entities
    .filter((e) => e.visible !== false)
    .flatMap((entity) => {
      const model = modelGeometry(entity);
      const frameId =
        model?.frameId ||
        (entity.representations || [])[0]?.coordinateFrameId ||
        entity.measurements?.coordinateFrameId;
      const corners =
        model?.corners || jsonObject(entity.measurements?.basis)?.cornersNative;
      if (
        frameId !== frame.id ||
        !Array.isArray(corners) ||
        !corners.length ||
        !corners.every(isVec3)
      )
        return [];
      const ps = corners.map((p) => [dot(p, x), dot(p, y)]);
      return [
        {
          entity,
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
