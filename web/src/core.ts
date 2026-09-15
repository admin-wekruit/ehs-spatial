import {
  boundsCorners,
  cross,
  dot,
  matmul,
  point,
  transformMatrix,
  unit,
} from "./viewer/native-math.ts";
import type {
  Entity,
  Job,
  Observation,
  Operation,
  Publication,
  PublicationSummary,
  Representation,
  SceneDocument,
  Transform,
  Vec3,
} from "./types.ts";
import { modelFamily } from "./scene-semantics.ts";

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
export function cameraForImage(document: SceneDocument, imageId: string | null | undefined) {
  if (!imageId) return null;
  const binding = jsonObject(jsonObject(document.geometryBindings)?.[imageId]);
  return binding && typeof binding.cameraId === "string"
    ? document.cameras.find(camera => camera.id === binding.cameraId && camera.imageId === imageId) || null : null;
}
export function publicationReaderURL(schemaVersion: number, href: string): string | null {
  if (schemaVersion === 2) return null;
  if (schemaVersion !== 1) throw Error("unsupported_scene_schema");
  const current = new URL(href);
  if (current.pathname.includes("/readers/v1/")) return null;
  const legacy = new URL("readers/v1/app.html", new URL(".", current));
  legacy.search = current.search; legacy.hash = current.hash;
  return legacy.href;
}
/** Daily report links follow the same workcell branch; explicit snapshots stay fixed. */
export function currentPublicationURL(publication: Publication, candidate: Publication, href: string): string | null {
  const url = new URL(href), params = new URLSearchParams(url.hash.split("?")[1] || "");
  if (params.get("snapshot") === "1" || publication.id === candidate.id || publication.projectId !== candidate.projectId ||
      publication.snapshot.revision.branchId !== candidate.snapshot.revision.branchId) return null;
  params.set("fromReport", publication.id);
  url.searchParams.set("report", candidate.id);
  url.hash = "/reports/" + encodeURIComponent(candidate.id) + "?" + params;
  return url.href;
}
export function currentEntityId(document: SceneDocument, entityId: string): string | null {
  if (document.entities.some(entity => entity.id === entityId)) return entityId;
  const refs = new Set<string>();
  for (const raw of Array.isArray(document.identityDecisions) ? document.identityDecisions : []) {
    const decision = jsonObject(raw), ids = decision?.entityIds, groups = decision?.observationGroups;
    if (!Array.isArray(ids) || !Array.isArray(groups) || !ids.includes(entityId)) continue;
    const selected = ids.length === groups.length ? [groups[ids.indexOf(entityId)]] : groups;
    for (const group of selected) if (Array.isArray(group)) for (const ref of group) if (typeof ref === "string") refs.add(ref);
  }
  const owners = new Set([...refs].flatMap(ref => { const owner = observationOwner(document, ref); return owner ? [owner.id] : []; }));
  return owners.size === 1 ? [...owners][0] : null;
}
export function currentCameras(document: SceneDocument) {
  return [...new Set(document.cameras.map(camera => camera.imageId))].flatMap(imageId => {
    const camera = cameraForImage(document, imageId); return camera ? [camera] : [];
  });
}
export function activeModel(entity: Entity) {
  return (entity.representations || []).find(rep => rep.id === entity.activeModelRepresentationId &&
    ["generated_mesh", "primitive"].includes(rep.kind)) || null;
}
export function isPartitionSource(entity: Entity, representationId: string) {
  return (entity.lineage || []).some(raw => {
    const event = jsonObject(raw);return event?.operation === "partition_model_parts" && event.sourceRepresentationId === representationId;
  });
}
export function modelFamilySignature(document: SceneDocument, entityId: string) {
  return modelFamily(document, entityId).map(entity => {
    const rep = activeModel(entity);
    return [entity.id, entity.parentEntityId, entity.visible, entity.activeModelRepresentationId,
      entity.currentModelTransform, rep, document.assets.find(asset => asset.id === rep?.assetId)];
  });
}
export function observationOwner(document: SceneDocument, observationId: string) {
  const owners = document.entities.filter(entity => entity.observationRefs?.includes(observationId));
  return owners.length === 1 ? owners[0] : null;
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
export function planHits(shapes: ReturnType<typeof planShapes>, x: number, y: number, lineTolerance = 0) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return [];
  const area = (polygon: number[][]) => Math.abs(polygon.reduce((sum, p, i) => {
    const q = polygon[(i + 1) % polygon.length];
    return sum + p[0] * q[1] - q[0] * p[1];
  }, 0)) / 2;
  const size = (shape: ReturnType<typeof planShapes>[number]) => shape.polygons.reduce((sum, polygon) =>
    sum + area(polygon.exterior) - polygon.holes.reduce((total, hole) => total + area(hole), 0), 0);
  const nearLine = (line: number[][]) => line.some((b, i) => {
    if (!i) return false;
    const a = line[i - 1], dx = b[0] - a[0], dy = b[1] - a[1], squared = dx * dx + dy * dy;
    const t = squared ? Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / squared)) : 0;
    return Math.hypot(x - a[0] - t * dx, y - a[1] - t * dy) <= Math.max(0, lineTolerance);
  });
  return shapes.filter((shape) => shape.polygons.some(polygon => insidePolygons([polygon.exterior, ...polygon.holes], x, y)) || shape.lines.some(nearLine))
    .sort((a, b) => size(a) - size(b) || a.entity.id.localeCompare(b.entity.id));
}
export type PlanPolygon = { exterior: number[][]; holes: number[][][] };
export function planPolygonPath(polygon: PlanPolygon, project: (p: number[]) => number[]) {
  return [polygon.exterior, ...polygon.holes].map(ring => "M " + ring.map(p => project(p).join(",")).join(" L ") + " Z").join(" ");
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
export function editableTransform(entity: Entity, document?: SceneDocument): Transform | null {
  const model = activeModel(entity);
  if (!model && entity.currentModelTransform && document?.entities.some(child => child.parentEntityId === entity.id)) return entity.currentModelTransform;
  return model && model.sourceValidity !== "stale" && (model.placementState === "confirmed" || ["requires_alignment_confirmation", "imported_proposal"].includes(model.placementReason || ""))
    ? entity.currentModelTransform || model.transform : null;
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
/** Apply a world-space assembly delta while retaining absolute part poses. */
export function modelFamilyTransforms(document: SceneDocument, entityId: string, value: Transform) {
  const family = modelFamily(document, entityId), target = family[0];
  if (!target) throw Error("entity_not_found");
  if (document.schemaVersion === 2 && !activeModel(target) && family.length === 1) throw Error("active_model_required");
  if (family.length === 1) return [{ entity: target, transform: value }];
  const old = target.currentModelTransform;
  if (!old || old.coordinateFrameId !== value.coordinateFrameId) throw Error("part_coordinate_frame_mismatch");
  const matrix = transformMatrix(old), inverse = Array(16).fill(0);inverse[15] = 1;
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) inverse[c * 4 + r] = matrix[r * 4 + c] / (old.scale[r] ** 2);
  for (let r = 0; r < 3; r++) inverse[12 + r] = -[0, 1, 2].reduce((sum, k) => sum + inverse[k * 4 + r] * old.position[k], 0);
  const delta = matmul(transformMatrix(value), inverse);
  return family.map((entity, index) => {
    if (!index) return { entity, transform: value };
    const pose = entity.currentModelTransform;
    if (!pose || pose.coordinateFrameId !== old.coordinateFrameId) throw Error("part_coordinate_frame_mismatch");
    if (sameJSON(old, value)) return { entity, transform: pose };
    const m = matmul(delta, transformMatrix(pose)), scales = [0, 1, 2].map(c => Math.hypot(m[c * 4], m[c * 4 + 1], m[c * 4 + 2]));
    if (scales.some(s => !Number.isFinite(s) || s <= 0)) throw Error("invalid_transform_scale");
    const columns = scales.map((s, c) => [0, 1, 2].map(r => m[c * 4 + r] / s));
    if (columns.some((a, i) => columns.some((b, j) => Math.abs(dot(a, b) - (i === j ? 1 : 0)) > 1e-6)) || dot(cross(columns[0], columns[1]), columns[2]) < .999999) throw Error("transform_shear_or_reflection");
    const r = (i: number, j: number) => columns[j][i], trace = r(0, 0) + r(1, 1) + r(2, 2), q = [0, 0, 0, 0];
    if (trace > 0) {
      const s = Math.sqrt(trace + 1) * 2;q[3] = s / 4;q[0] = (r(2, 1) - r(1, 2)) / s;q[1] = (r(0, 2) - r(2, 0)) / s;q[2] = (r(1, 0) - r(0, 1)) / s;
    } else {
      const i = [0, 1, 2].reduce((best, candidate) => r(candidate, candidate) > r(best, best) ? candidate : best, 0), j = (i + 1) % 3, k = (i + 2) % 3, s = Math.sqrt(1 + r(i, i) - r(j, j) - r(k, k)) * 2;
      q[i] = s / 4;q[3] = (r(k, j) - r(j, k)) / s;q[j] = (r(j, i) + r(i, j)) / s;q[k] = (r(k, i) + r(i, k)) / s;
    }
    const transform: Transform = { coordinateFrameId: pose.coordinateFrameId, position: [m[12], m[13], m[14]], quaternion: q.map(v => v / Math.hypot(...q)) as Transform["quaternion"], scale: scales as Vec3 };
    if (transformMatrix(transform).some((v, i) => Math.abs(v - m[i]) > 1e-6 + 1e-5 * Math.abs(m[i]))) throw Error("transform_roundtrip_failed");
    return { entity, transform };
  });
}
export function previewOperations(
  document: SceneDocument,
  operations: Operation[],
  baseRevisionId?: string,
): SceneDocument {
  const next = structuredClone(document);
  function transformValue(raw: unknown): Transform {
    const value = jsonObject(raw);
    if (!value || !next.coordinateFrames.some(frame => frame.id === value.coordinateFrameId)) throw Error("invalid_coordinate_frame");
    if (!isVec3(value.position) || !isVec3(value.scale) || value.scale.some(n => n <= 0) ||
      !Array.isArray(value.quaternion) || value.quaternion.length !== 4 || value.quaternion.some(n => finiteNumber(n) === undefined) ||
      Math.abs(value.quaternion.reduce((sum, n) => sum + n * n, 0) - 1) > 1e-5) throw Error("invalid_transform");
    return structuredClone(value) as Transform;
  }
  for (const op of operations) {
    const entity = next.entities.find((e) => e.id === op.entityId);
    if (op.type === "setPartRelation") {
      if (next.schemaVersion !== 2) throw Error("parts_require_scene_v2");
      const evidence = Array.isArray(op.evidenceRefs) ? op.evidenceRefs.map(jsonObject) : [];
      if ((op.parentEntityId !== null && typeof op.parentEntityId !== "string") || typeof op.reason !== "string" || !op.reason.trim() || op.reason.length > 8000 || !evidence.length ||
        evidence.some(ref => !ref || typeof ref.observationId !== "string" || !ref.observationId || !Number.isInteger(ref.revision) || Number(ref.revision) < 1)) throw Error("part_relation_evidence_required");
      if (!baseRevisionId) throw Error("part_base_revision_required");
      if (!entity) throw Error("entity_not_found");
      const parent = op.parentEntityId === null ? null : next.entities.find(item => item.id === op.parentEntityId);
      if (op.parentEntityId !== null && !parent) throw Error("part_parent_not_found");
      const refs = evidence as { observationId: string; revision: number }[], ids = new Set(refs.map(ref => ref.observationId));
      if (refs.some(ref => next.observations.find(observation => observation.id === ref.observationId)?.revision !== ref.revision)) throw Error("part_observation_revision_mismatch");
      if (!entity.observationRefs?.some(id => ids.has(id)) || parent && !parent.observationRefs?.some(id => ids.has(id))) throw Error("part_relation_evidence_scope");
      if (ids.size !== refs.length) throw Error("part_relation_evidence_invalid");
      if (parent && (!entity.currentModelTransform || !parent.currentModelTransform || entity.currentModelTransform.coordinateFrameId !== parent.currentModelTransform.coordinateFrameId)) throw Error("part_coordinate_frame_mismatch");
      const seen = new Set([entity.id]);let ancestor = parent;
      while (ancestor) {
        if (seen.has(ancestor.id)) throw Error("part_relation_cycle");seen.add(ancestor.id);
        const parentId = ancestor.parentEntityId;ancestor = parentId ? next.entities.find(item => item.id === parentId) : null;
        if (parentId && !ancestor) throw Error("part_parent_not_found");
      }
      const oldParent = next.entities.find(item => item.id === entity.parentEntityId);
      (entity.lineage ||= []).push({ operation: "setPartRelation", sourceRevisionId: baseRevisionId, parentEntityId: entity.parentEntityId || null, partRelation: structuredClone(entity.partRelation || null) });
      entity.parentEntityId = op.parentEntityId as string | null;
      entity.partRelation = { source: "manual", baseRevisionId, evidenceRefs: structuredClone(refs), reason: op.reason };
      if (oldParent && !oldParent.activeModelRepresentationId && !next.entities.some(item => item.parentEntityId === oldParent.id)) oldParent.currentModelTransform = null;
    } else if (op.type === "setTransform" && entity) {
      const value = transformValue(op.transform || {
          coordinateFrameId: op.coordinateFrameId,
          position: op.position,
          quaternion: op.quaternion,
          scale: op.scale,
        });
      for (const { entity: member, transform } of modelFamilyTransforms(next, entity.id, value)) {
        const previous = member.currentModelTransform;
        member.currentModelTransform = structuredClone(transform);
        (member.representations || [])
          .filter(r => ["generated_mesh", "primitive"].includes(r.kind) && (next.schemaVersion === 1 || r.id === member.activeModelRepresentationId))
          .forEach(r => {
            const changed = !sameJSON(transform, previous || r.transform);
            r.transform = structuredClone(transform); r.coordinateFrameId = transform.coordinateFrameId;
            if (changed) { r.placementState = "unconfirmed"; r.placementReason = "requires_alignment_confirmation"; delete r.placementSource; }
          });
      }
    } else if (op.type === "confirmPlacement" && entity) {
      if (!entity.activeModelRepresentationId || op.representationId !== entity.activeModelRepresentationId) throw Error("active_model_selection_required");
      const rep = activeModel(entity);
      if (!rep || rep.sourceValidity === "stale") throw Error("active_model_required");
      if (rep.placementState !== "confirmed" && !["imported_proposal", "requires_alignment_confirmation"].includes(rep.placementReason || "")) throw Error("model_placement_required");
      transformValue(entity.currentModelTransform || rep.transform);
      // Only the saved server response supplies manual placement confirmation.
    } else if (op.type === "setVisibility" && entity) {
      if (typeof op.visible !== "boolean") throw Error("invalid_visibility");
      modelFamily(next, entity.id).forEach(member => { member.visible = op.visible as boolean; });
    }
    else if (op.type === "setLabel" && entity)
      entity.label = op.label as string;
    else if (op.type === "setMaterial" && entity) {
      const material = jsonObject(op.material);
      if (!material) throw Error("invalid_material");
      if (next.schemaVersion === 1) entity.material = structuredClone(material);
      else {
        const models = modelFamily(next, entity.id).flatMap(member => activeModel(member) ? [activeModel(member)!] : []);
        if (!models.length) throw Error("active_model_required");
        models.forEach(rep => { rep.material = structuredClone(material); });
      }
    }
    else if (op.type === "setActiveModelRepresentation" && entity) {
      if (!("representationId" in op)) throw Error("active_model_representation_required");
      const rep = (entity.representations || []).find(rep => rep.id === op.representationId && ["generated_mesh", "primitive"].includes(rep.kind));
      if (op.representationId !== null && !rep) throw Error("active_model_representation_invalid");
      if (rep && isPartitionSource(entity, rep.id)) throw Error("part_source_model_inactive");
      const assemblyPose = next.entities.some(child => child.parentEntityId === entity.id) ? entity.currentModelTransform : null;
      entity.activeModelRepresentationId = op.representationId as string | null;
      entity.currentModelTransform = structuredClone(rep?.transform || assemblyPose || null);
    }
    else if (op.type === "selectMeasurementEvidence" && entity) {
      const key = String(op.measurementKey), selectedId = typeof op.measurementEvidenceId === "string" ? op.measurementEvidenceId : null;
      const evidence = entity.measurementEvidence?.find(item=>item.id===selectedId && item.measurementKey===key);
      if (selectedId && !evidence) throw Error("measurement_evidence_not_found");
      entity.measurementSelections = {...entity.measurementSelections,[key]:selectedId};
      entity.measurements = {...entity.measurements,[key]:evidence ? structuredClone(evidence.originalMeasurement) : null};
    }
    else if (op.type === "setPrimitive" && entity) {
      const spec = jsonObject(op.primitive), kind = spec?.kind || spec?.type;
      let bounds: { min: Vec3; max: Vec3 };
      if (kind === "box" && isVec3(spec?.dimensions) && spec.dimensions.every(n => n > 0)) {
        bounds = { min: spec.dimensions.map(n => -Math.fround(n / 2)) as Vec3, max: spec.dimensions.map(n => Math.fround(n / 2)) as Vec3 };
      } else if (kind === "cylinder" && finiteNumber(spec?.radius) !== undefined && Number(spec!.radius) > 0 &&
        finiteNumber(spec?.height) !== undefined && Number(spec!.height) > 0 && Number.isInteger(spec!.segments ?? 64) && Number(spec!.segments ?? 64) >= 8 && Number(spec!.segments ?? 64) <= 256) {
        const radius = Number(spec!.radius), height = Number(spec!.height), segments = Number(spec!.segments ?? 64),
          xs = Array.from({ length: segments }, (_, i) => Math.fround(radius * Math.cos(i * 2 * Math.PI / segments))),
          ys = Array.from({ length: segments }, (_, i) => Math.fround(radius * Math.sin(i * 2 * Math.PI / segments)));
        bounds = { min: [Math.min(...xs), Math.min(...ys), -Math.fround(height / 2)], max: [Math.max(...xs), Math.max(...ys), Math.fround(height / 2)] };
      } else throw Error("invalid_primitive");
      const models = entity.representations || [];
      if (next.schemaVersion === 2 && op.representationId != null && op.representationId !== entity.activeModelRepresentationId) throw Error("active_model_selection_required");
      let rep = models.find(r => r.kind === "primitive" && (next.schemaVersion === 2 ? r.id === entity.activeModelRepresentationId : op.representationId == null || r.id === op.representationId));
      if (next.schemaVersion === 2 && !rep && models.some(r => r.kind === "primitive")) throw Error("active_model_selection_required");
      if (op.representationId != null && !rep) throw Error("primitive_representation_not_found");
      const rawTransform = op.transform || spec?.transform || entity.currentModelTransform || rep?.transform;
      if (!rawTransform) throw Error("primitive_placement_required");
      const value = transformValue(rawTransform);
      const changed = !rep || !sameJSON(rep.primitive, spec) || !sameJSON(value, entity.currentModelTransform || rep.transform);
      if (!rep) {
        rep = {
          id: "preview:" + entity.id,
          kind: "primitive",
          assetId: null,
          coordinateFrameId: value.coordinateFrameId,
          transform: value,
          placementState: "unconfirmed",
          sourceRefs: (entity.observationRefs || []).map(observationId => ({ observationId })),
        };
        (entity.representations ??= []).push(rep);
      }
      rep.primitive = structuredClone(spec!); rep.transform = structuredClone(value); rep.coordinateFrameId = value.coordinateFrameId;
      rep.bounds = bounds; entity.currentModelTransform = value;
      if (next.schemaVersion === 2) entity.activeModelRepresentationId = rep.id;
      if (changed) {
        rep.placementState = "unconfirmed"; rep.placementReason = "requires_alignment_confirmation";
        delete rep.placementSource; delete rep.planProjection;
      }
    } else if (op.type === "addCoordinateFrame")
      next.coordinateFrames.push(
        structuredClone(op.frame as SceneDocument["coordinateFrames"][number]),
      );
    else if (op.type === "addEntity")
      next.entities.push(structuredClone(op.entity as Entity));
    else if (op.type === "removeEntity")
      next.entities = next.entities.filter((e) => e.id !== op.entityId);
  }
  // Match the server's final-batch check: a whole family may switch frames together.
  for (const entity of next.entities) if (entity.parentEntityId) {
    const parent = next.entities.find(parent => parent.id === entity.parentEntityId);
    if (!parent) throw Error("part_parent_not_found");
    if (!entity.currentModelTransform || !parent.currentModelTransform || entity.currentModelTransform.coordinateFrameId !== parent.currentModelTransform.coordinateFrameId) throw Error("part_coordinate_frame_mismatch");
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
  const id = entity.measurements?.coordinateFrameId;
  return document.coordinateFrames.find((f) => f.id === id)?.scale || null;
}

export function modelScale(document: SceneDocument, entity: Entity) {
  return document.coordinateFrames.find(frame => frame.id === editableTransform(entity, document)?.coordinateFrameId)?.scale || null;
}

export function modelGeometry(entity: Entity) {
  const rep = activeModel(entity),
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
export type GeometryLayer = "model" | "observed_surface" | "point_cloud";
export type GeometryOptions = { layer: GeometryLayer; frameId: string; showCandidates?: boolean; imageId?: string | null; observations?: Observation[] };

export function representationInPhoto(entity: Entity, rep: Representation, imageId?: string | null, observations: Observation[] = []) {
  if (!["observed_surface", "point_cloud"].includes(rep.kind)) return true;
  const explicitImages = (rep.sourceRefs || []).flatMap(ref => { const value = jsonObject(ref); return typeof value?.imageId === "string" ? [value.imageId] : []; });
  if (explicitImages.length) return !!imageId && explicitImages.includes(imageId);
  if (entity.sourceContext) return true;
  const sourceIds = (rep.sourceRefs || []).flatMap(ref => {
    const value = jsonObject(ref); return typeof value?.observationId === "string" ? [value.observationId] : [];
  });
  const photoObservations = observations.filter(observation => entity.observationRefs?.includes(observation.id) && observation.imageId === imageId);
  if (sourceIds.length) return !!imageId && photoObservations.some(observation => sourceIds.includes(observation.id));
  // Imported evidence references a record within a frozen asset. Both fields
  // identify the source; a shared label, asset alone or record ID alone cannot.
  return !!imageId && (rep.sourceRefs || []).some(raw => {
    const source = jsonObject(raw);
    return typeof source?.assetId === "string" && typeof source.sourceRecordId === "string" &&
      photoObservations.some(observation => (observation.sourceRefs || []).some(rawRef => {
        const ref = jsonObject(rawRef);
        return ref?.assetId === source.assetId && ref?.sourceRecordId === source.sourceRecordId;
      }));
  });
}

export function representationAvailable(entity: Entity, rep: Representation, frameId: string | null, showCandidates = false) {
  const modeled = ["generated_mesh", "primitive"].includes(rep.kind);
  if (modeled && rep.id !== entity.activeModelRepresentationId) return false;
  const transform = modeled ? entity.currentModelTransform || rep.transform : rep.transform;
  return entity.visible !== false && rep.sourceValidity !== "stale" && !!frameId &&
    rep.coordinateFrameId === frameId && transform?.coordinateFrameId === frameId &&
    (rep.placementState === "confirmed" || showCandidates &&
      ["requires_alignment_confirmation", "imported_proposal"].includes(rep.placementReason || ""));
}

/** One located geometry choice for source-photo axes, 3D bounds and plan views. */
export function entityGeometryForLayer(entity: Entity, { layer, frameId, showCandidates = true, imageId, observations }: GeometryOptions) {
  if (entity.sourceContext || entity.visible === false) return null;
  const representations = (entity.representations || []).filter((rep) =>
    representationAvailable(entity, rep, frameId, showCandidates) && representationInPhoto(entity, rep, imageId, observations) &&
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
    return fromRepresentations(representations.filter((r) => ["generated_mesh", "primitive"].includes(r.kind)), "model");
  }
  if (layer === "point_cloud") {
    const cloud = fromRepresentations(representations.filter((r) => r.kind === "point_cloud"), "point_cloud");
    if (cloud) return cloud;
  }
  const observed = fromRepresentations(representations.filter((r) => r.kind === "observed_surface"), "observed");
  if (observed) return observed;
  const measurements = entity.measurements || {};
  const selectedMeasurement = (key: string) => {
    const selection = jsonObject(entity.measurementSelections)?.[key];
    const evidence = (Array.isArray(entity.measurementEvidence) ? entity.measurementEvidence : []).map(jsonObject).find(row => row?.id === selection);
    const refs = evidence?.observationRefs;
    return Array.isArray(refs) && !!imageId &&
      (observations || []).some(observation => refs.includes(observation.id) && observation.imageId === imageId);
  };
  const basis = selectedMeasurement("basis") ? jsonObject(measurements.basis)?.cornersNative : null;
  const bounds = selectedMeasurement("observedBounds") ? jsonObject(measurements.observedBounds) : null;
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
export function modelFamilyGeometry(document: SceneDocument, entityId: string, options: GeometryOptions) {
  const family = modelFamily(document, entityId), target = family[0];
  if (!target || target.visible === false || target.sourceContext) return null;
  const geometries = family.flatMap(entity => {
    const geometry = entityGeometryForLayer(entity, { ...options, layer: "model" });
    return geometry ? [geometry] : [];
  });
  if (!geometries.length) return null;
  if (family.length === 1) return geometries[0];
  const points = geometries.flatMap(geometry => geometry.corners);
  // Part transforms already locate their vertices in the native scene frame.
  return { ...geometries[0], transform: target.currentModelTransform || geometries[0].transform,
    corners: boundsCorners({ min: [0, 1, 2].map(k => Math.min(...points.map(p => p[k]))),
      max: [0, 1, 2].map(k => Math.max(...points.map(p => p[k]))) }),
    representationIds: geometries.flatMap(geometry => geometry.representationIds) };
}
export function modelTilt(document: SceneDocument, entity: Entity) {
  const transform = editableTransform(entity, document);
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
export type PlanOptions = Partial<GeometryOptions> & { scope?: "photo" | "scene" };

/** The saved source exposure chooses an observed state; browsing a photo does not. */
export function cadReferenceImage(document: SceneDocument, entity: Entity): string | null {
  const reference = jsonObject(entity.cadReference), imageId = reference?.referenceImageId;
  if (reference?.status !== "resolved" || typeof imageId !== "string" || !cameraForImage(document, imageId)) return null;
  const observations = observationsFor(document, entity).filter(observation => observation.imageId === imageId);
  const sourceRefs = reference.sourceRefs;
  if (!observations.length || !Array.isArray(sourceRefs) || sourceRefs.length !== observations.length ||
      !observations.every(observation => sourceRefs.some(raw => {
        const ref = jsonObject(raw); return ref?.observationId === observation.id && ref.revision === observation.revision;
      }))) return null;
  return imageId;
}

export function planReferenceContract(document: SceneDocument): "photo" | "scene" {
  // Before saved scene references, observation sources define the photo contract.
  // An explicit invalid reference still declares scene scope; it cannot select another exposure.
  return document.entities.some(entity => Object.hasOwn(entity, "cadReference")) ? "scene" : "photo";
}

export function scenePlanOptions(document: SceneDocument, layer: GeometryLayer, showCandidates = true, imageId?: string | null): PlanOptions {
  if (planReferenceContract(document) === "photo")
    return {scope: "photo", layer, imageId, frameId: cameraForImage(document, imageId)?.coordinateFrameId || "", showCandidates};
  const frames = new Set(document.entities.filter(entity => entity.visible !== false && !entity.sourceContext).flatMap(entity => {
    const model = layer === "model" ? activeModel(entity) : null;
    if (layer === "model") return model && representationAvailable(entity, model, model.coordinateFrameId, showCandidates) ? [model.coordinateFrameId] : [];
    const camera = cameraForImage(document, cadReferenceImage(document, entity));
    return camera ? [camera.coordinateFrameId] : [];
  }));
  return {scope: "scene", layer, frameId: frames.size === 1 ? [...frames][0] : "", showCandidates};
}

export function planShapes(document: SceneDocument, options: PlanOptions = {}) {
  const frame = document.coordinateFrames.find(
    (f) =>
      (options.frameId === undefined || f.id === options.frameId) &&
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
      : [[...x, 0], [...y, 0], [...n, 0], [0, 0, 0, 1]];
  return document.entities
    .filter((e) => e.visible !== false && !e.sourceContext)
    .flatMap((entity) => {
      const imageId = options.scope === "scene" ? cadReferenceImage(document, entity) : options.imageId;
      const geometry = entityGeometryForLayer(entity, {layer: options.layer || "model", frameId: frame.id, showCandidates: options.showCandidates, imageId, observations: document.observations});
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
      const projections = (entity.representations || []).filter(rep => geometry.representationIds.includes(rep.id)).flatMap(rep => {
        const saved = jsonObject(rep.planProjection), asset = document.assets.find(asset => asset.id === rep.assetId);
        const modeled = ["generated_mesh", "primitive"].includes(rep.kind);
        const transform = modeled ? entity.currentModelTransform || rep.transform : rep.transform;
        const snapshot = jsonObject(saved?.transformSnapshot);
        const unchanged = sameJSON(snapshot, transform);
        const translated = modeled && isVec3(snapshot?.position) && sameJSON({...snapshot, position: transform.position}, transform);
        const matrix = saved?.nativeToPlane;
        const samePlane = Array.isArray(matrix) && matrix.length === 4 && matrix.every((row, i) =>
          Array.isArray(row) && row.length === 4 && row.every((value, j) => typeof value === "number" && Number.isFinite(value) &&
            Math.abs(value - plane[i][j]) <= 1e-12 * Math.max(1, Math.abs(value), Math.abs(plane[i][j]))));
        const observation = document.observations.find(observation => observation.id === saved?.observationId);
        if (saved?.methodVersion !== "indexed-mesh-triangle-union-v1" || saved.coordinateFrameId !== frame.id ||
            !sameJSON(saved.groundNormalSnapshot, normal) || !samePlane || (!unchanged && !translated) ||
            (rep.kind === "primitive" ? saved.assetId != null || saved.assetSha256 != null || !sameJSON(saved.primitiveSnapshot, rep.primitive)
              : !asset?.sha256 || saved.assetId !== rep.assetId || saved.assetSha256 !== asset.sha256) ||
            (!modeled && (saved.imageId !== imageId || !observation || observation.imageId !== saved.imageId ||
              observation.revision !== saved.observationRevision || !entity.observationRefs?.includes(observation.id) ||
              !(rep.sourceRefs || []).some(raw => { const ref = jsonObject(raw); return ref?.observationId === observation.id && ref.revision === observation.revision; })))) return [];
        const points = (value: unknown): value is number[][] => Array.isArray(value) &&
          value.every(point => Array.isArray(point) && point.length === 2 && point.every(value => typeof value === "number" && Number.isFinite(value)));
        const ring = (value: unknown): value is number[][] => points(value) && value.length >= 4 && sameJSON(value[0], value[value.length - 1]);
        if (!Array.isArray(saved.polygons) || !Array.isArray(saved.lines) ||
            !saved.polygons.every(raw => { const polygon = jsonObject(raw); return polygon && ring(polygon.exterior) && Array.isArray(polygon.holes) && polygon.holes.every(ring); }) ||
            !saved.lines.every(line => points(line) && line.length >= 2)) return [];
        const delta = translated && !unchanged ? transform.position.map((value, k) => value - (snapshot!.position as number[])[k]) : [0, 0, 0];
        const offset = [dot(delta, plane[0]), dot(delta, plane[1])], shift = (p: number[]) => [p[0] + offset[0], p[1] + offset[1]];
        return [{representationId: rep.id,
          polygons: (saved.polygons as PlanPolygon[]).map(polygon => ({exterior: polygon.exterior.map(shift), holes: polygon.holes.map(hole => hole.map(shift))})),
          lines: (saved.lines as number[][][]).map(line => line.map(shift))}];
      });
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
      const polygons: PlanPolygon[] = projections.length ? projections.flatMap(projection => projection.polygons)
        : useSavedHull ? [{exterior: saved!.points as number[][], holes: []}] : [];
      const lines = projections.flatMap(projection => projection.lines);
      const ps = [...polygons.flatMap(polygon => [polygon.exterior, ...polygon.holes].flat()), ...lines.flat()];
      if (!ps.length) return [];
      const min = [Infinity, Infinity], max = [-Infinity, -Infinity];
      for (const p of ps) for (let k = 0; k < 2; k++) { min[k] = Math.min(min[k], p[k]); max[k] = Math.max(max[k], p[k]); }
      return [
        {
          entity,
          polygons,
          lines,
          coordinateFrameId: frame.id,
          projectionSource: projections.length ? "mesh_projection" as const : "saved_hull" as const,
          geometryKind: geometry.geometryKind,
          representationIds: projections.length ? projections.map(projection => projection.representationId) : geometry.representationIds,
          min,
          max,
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
