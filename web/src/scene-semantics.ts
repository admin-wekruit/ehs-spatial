import type { Entity, SceneDocument } from "./types";

function records(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => !!item && typeof item === "object" && !Array.isArray(item))
    : [];
}

// A geometry role identifies a reference surface; it does not prove its normal,
// slope, metric scale or extent. The frozen "floor" observation category is
// also a presentation role; mutable entity names and flat bounds are not.
export function isReferenceSurface(
  document: Pick<SceneDocument, "observations" | "coordinateFrames">,
  entity: Entity | null | undefined,
): boolean {
  if (!entity) return false;
  if (entity.geometryRole === "floor") return true;
  if (entity.geometryRole === "object" || entity.geometryRole === "unknown") return false;
  const observationIds = new Set(entity.observationRefs || []);
  const observations = document.observations.filter((observation) => observationIds.has(observation.id));
  if (observations.some((observation) => records(observation.labelEvidence).some((evidence) =>
    evidence.geometryRole === "floor" || evidence.geometryRole === undefined && evidence.label === "floor"))) return true;
  return document.coordinateFrames.some((frame) => records(frame.ground?.sourceRefs).some((ref) =>
    ref.entityId === entity.id || observations.some((observation) => ref.observationId === observation.id &&
      (ref.revision === undefined || ref.revision === observation.revision))));
}
