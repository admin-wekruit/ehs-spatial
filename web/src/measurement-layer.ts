import type { Representation, Revision, Transform } from "./types";
import type { BendOutcome } from "./SpatialMeasurements";

/** A static measurement layer published next to a frozen report: the frame's reference-object scale, a corrected ground plane,
 * reference-object models built from their specification, and per-object facts measured from the photos. It applies to exactly
 * the revision it names; any other revision is shown unchanged. */
export type LayerFact = { label: string; text: string; kind: string; method?: string };
export type LayerConfidence = { level: "high" | "medium" | "low" | "unverified"; label: string; missing?: string[]; reasons?: string[] };
export type MeasurementLayer = {
  schemaVersion: 1;
  publicationId: string;
  revisionId: string;
  coordinateFrameId: string;
  scale: { nativeToMeters: number; status: "operator_anchored"; source: string };
  ground?: { normal: [number, number, number]; offset: number; plane: [number, number, number, number]; source: string };
  models?: Record<string, { representation: Representation; note: string }>;
  /** Extra mesh assets published next to the page (url relative to it), registered in the document under their ids. */
  assets?: (Record<string, unknown> & { id: string; url: string })[];
  facts?: Record<string, LayerFact[]>;
  /** Per-object confidence from the generic checks, with what is missing to raise it. */
  confidence?: Record<string, LayerConfidence>;
  /** Display names for entities whose imported label is a working name (e.g. English evidence labels). */
  labels?: Record<string, string>;
  /** Fold angles measured from the photos, shown like saved bends; each references the layer model it measured. */
  bends?: BendOutcome[];
};

/** The report's resources with the layer's own files served from the website and its measured bends; everything else unchanged. */
export function withLayerAssets<T extends { resolveAsset: (id: string) => Promise<string>; layerBends?: BendOutcome[]; layerConfidence?: Record<string, LayerConfidence> }>(resources: T, layer: MeasurementLayer | null): T {
  const urls = new Map((layer?.assets || []).map(asset => [asset.id, new URL(asset.url, location.href).href]));
  if (!urls.size && !layer?.bends?.length && !layer?.confidence) return resources;
  return { ...resources, layerBends: layer?.bends, layerConfidence: layer?.confidence, resolveAsset: async (id: string) => urls.get(id) ?? resources.resolveAsset(id) };
}

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
    const label = layer.labels?.[entity.id];
    if (label) entity = { ...entity, label };
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
  const added = new Set((layer.assets || []).map(asset => asset.id));
  const assets = [...document.assets.filter(asset => !added.has(asset.id)),
    ...(layer.assets || []).map(({ url: _url, ...asset }) => asset as unknown as (typeof document.assets)[number])];
  return { ...revision, document: { ...document, coordinateFrames, entities, assets } };
}
