import type { Representation, Revision, Transform } from "./types";

/** A static measurement layer published next to a frozen report: the frame's reference-object scale, a corrected ground plane,
 * reference-object models built from their specification, and per-object facts measured from the photos. It applies to exactly
 * the revision it names; any other revision is shown unchanged. */
export type LayerFact = { label: string; text: string; kind: string; method?: string };
export type MeasurementLayer = {
  schemaVersion: 1;
  publicationId: string;
  revisionId: string;
  coordinateFrameId: string;
  scale: { nativeToMeters: number; status: "operator_anchored"; source: string };
  ground?: { normal: [number, number, number]; offset: number; plane: [number, number, number, number]; source: string };
  models?: Record<string, { representation: Representation; note: string }>;
  facts?: Record<string, LayerFact[]>;
};

export async function loadMeasurementLayer(publicationId: string): Promise<MeasurementLayer | null> {
  try {
    const response = await fetch(`./measurement-layer/${encodeURIComponent(publicationId)}.json`, { cache: "no-cache" });
    return response.ok ? (await response.json()) as MeasurementLayer : null;
  } catch {
    return null;  // ponytail: no layer published for this report; show it as frozen
  }
}

export function applyMeasurementLayer(revision: Revision, layer: MeasurementLayer | null): Revision {
  if (!layer || layer.schemaVersion !== 1 || layer.revisionId !== revision.id) return revision;
  const document = revision.document;
  const coordinateFrames = document.coordinateFrames.map(frame => frame.id !== layer.coordinateFrameId ? frame : {
    ...frame,
    scale: { ...frame.scale, status: layer.scale.status, nativeToMeters: layer.scale.nativeToMeters, source: layer.scale.source },
    ground: layer.ground ? { ...(frame.ground || {}), normal: layer.ground.normal, plane: layer.ground.plane, offset: layer.ground.offset, source: layer.ground.source } : frame.ground,
  });
  const entities = document.entities.map(entity => {
    const model = layer.models?.[entity.id];
    if (!model || model.representation.coordinateFrameId !== layer.coordinateFrameId) return entity;
    const representation = model.representation;
    return {
      ...entity,
      representations: [...(entity.representations || []).filter(rep => rep.id !== representation.id), representation],
      activeModelRepresentationId: representation.id,
      currentModelTransform: representation.transform as Transform,
    };
  });
  return { ...revision, document: { ...document, coordinateFrames, entities } };
}
