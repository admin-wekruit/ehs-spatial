import type { Entity, Representation, SceneDocument } from "./types";

export function modelLabelKey(representation: Representation): string {
  return representation.kind === "generated_mesh" && ["observed_depth_surface", "inferred_planar_surface_from_observed_depth"].includes(String(representation.sourceKind))
    ? String(representation.sourceKind) : representation.kind;
}

/** Explicit model parts use absolute transforms in the same native frame. */
export function modelFamily(document: SceneDocument, entityId: string): Entity[] {
  const target = document.entities.find(entity => entity.id === entityId);
  if (!target) return [];
  const children = new Map<string, Entity[]>();
  for (const entity of document.entities) if (entity.parentEntityId) {
    const group = children.get(entity.parentEntityId) || []; group.push(entity); children.set(entity.parentEntityId, group);
  }
  const family: Entity[] = [], pending = [target], seen = new Set<string>();
  while (pending.length) {
    const entity = pending.pop()!;
    if (seen.has(entity.id)) throw Error("part_relation_cycle");
    seen.add(entity.id); family.push(entity); pending.push(...(children.get(entity.id) || []).slice().reverse());
  }
  return family;
}

export function identityCounts(document: SceneDocument) {
  const entities = document.entities.filter(entity => !entity.sourceContext);
  return { records: entities.length,
    linkedGroups: entities.filter(entity => entity.associationState === "confirmed" && entityEvidenceStatus(document, entity).photoCount > 1).length,
    pending: entities.filter(entity => entity.associationState !== "confirmed").length,
    observations: new Set(entities.flatMap(entity => entity.observationRefs || [])).size };
}

// Photo ownership, cross-view identity, and mesh placement are independent facts.
export function entityEvidenceStatus(document: SceneDocument, entity: Entity) {
  const refs = new Set(entity.observationRefs || []);
  const images = new Set(document.observations.filter(o => refs.has(o.id)).map(o => o.imageId));
  const sceneImages = new Set(document.observations.map(o => o.imageId));
  document.cameras.forEach(camera => sceneImages.add(camera.imageId));
  const photoKey = images.size > 1 ? "entityPhotosLinked" : images.size === 1 ? "entityPhotoLocated" :
    entity.representations?.length ? "entityModelWithoutPhoto" : "entityNeedsPhoto";
  const evidence = entity.associationEvidence as { status?: string } | undefined;
  const reasons: Record<string, string> = {
    mask_missing: "entityAssociationNeedsMask", geometry_missing: "entityAssociationNeedsGeometry",
    insufficient_support: "entityAssociationNeedsSupport", competing_candidates: "entityAssociationAmbiguous",
    no_supported_match: "entityAssociationNoMatch",
  };
  const identityKey = !images.size ? null : entity.associationState === "confirmed" ?
    images.size > 1 ? "entityCrossViewConfirmed" : "entityIdentityConfirmed" :
    sceneImages.size <= 1 ? "entitySingleView" : reasons[evidence?.status || ""] || "entityAssociationNotChecked";
  const representations = entity.representations || [],
    models = representations.filter(rep => ["generated_mesh", "primitive"].includes(rep.kind)),
    active = models.find(rep => rep.id === entity.activeModelRepresentationId),
    observed = representations.find(rep => ["observed_surface", "point_cloud"].includes(rep.kind) && rep.sourceValidity !== "stale");
  const modelKey = active?.sourceValidity === "stale" ? "identityModelStale" : (active ? modelLabelKey(active) : undefined) || observed?.kind ||
    (models.length ? "identitySourceModelsOnly" : representations.length ? "identitySourceStale" : "noGeometry");
  return { photoKey, identityKey, photoCount: images.size, modelKey };
}

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
