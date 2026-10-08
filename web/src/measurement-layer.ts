import type { Representation, Revision, Transform } from "./types";
import type { BendOutcome } from "./SpatialMeasurements";
import { translate, type Language } from "./translate.ts";

export type DisplayMessage = { code: string; params?: Record<string, string | number | DisplayMessage> };
export function validMessage(value: unknown): value is DisplayMessage {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const message = value as DisplayMessage;
  return typeof message.code === "string" && !!message.code.trim() && (message.params === undefined || !!message.params && typeof message.params === "object" && !Array.isArray(message.params) &&
    Object.values(message.params).every(param => typeof param === "string" || typeof param === "number" && Number.isFinite(param) || validMessage(param)));
}
/** Data supplies a stable code and parameters; the same catalog lookup renders nested messages in the selected language. */
export function renderMessage(language: Language, message: DisplayMessage): string {
  if (!validMessage(message)) throw Error("invalid_display_message");
  const params = Object.fromEntries(Object.entries(message.params || {}).map(([key, value]) => [key, typeof value === "object" ? renderMessage(language, value) : value]));
  return translate(language, message.code, params);
}

/** A static measurement layer published next to a frozen report: the frame's reference-object scale, a corrected ground plane,
 * reference-object models built from their specification, and per-object facts measured from the photos. It applies to exactly
 * the revision it names; any other revision is shown unchanged. Every display field is one language-neutral message. */
export type LayerFact = { label: DisplayMessage; text: DisplayMessage; kind: string; method?: string };
export type LayerConfidence = { level: "high" | "medium" | "low" | "unverified"; label: DisplayMessage; missing?: DisplayMessage[]; reasons?: DisplayMessage[] };
export type MeasurementLayer = {
  schemaVersion: 2;
  publicationId: string;
  revisionId: string;
  coordinateFrameId: string;
  /** uncertaintyRelative: half the spread of the per-feature scales (e.g. red head only vs yellow body only) over the joint scale. */
  scale: { nativeToMeters: number; status: "operator_anchored"; source: DisplayMessage; uncertaintyRelative?: number };
  ground?: { normal: [number, number, number]; offset: number; plane: [number, number, number, number]; source: DisplayMessage };
  models?: Record<string, { representation: Representation; note: DisplayMessage }>;
  /** Extra mesh assets published next to the page (url relative to it), registered in the document under their ids. */
  assets?: (Record<string, unknown> & { id: string; url: string })[];
  facts?: Record<string, LayerFact[]>;
  /** Per object: every stage from photos to the shown model and its measurements, with a sheet image (url relative to the page). */
  pipelines?: Record<string, { url?: string; caption?: DisplayMessage; stages: { label: DisplayMessage; text: DisplayMessage }[] }>;
  /** A comparison layer published next to the report's own (`<publicationId>.<id>.json`, opened with ?layer=<id>); never replaces it. */
  variant?: { id: string; label: DisplayMessage };
  /** Per-object confidence from the generic checks, with what is missing to raise it. */
  confidence?: Record<string, LayerConfidence>;
  /** Display names for entities whose imported label is a working name (e.g. English evidence labels). */
  labels?: Record<string, DisplayMessage>;
  /** Fold angles measured from the photos, shown like saved bends; each references the layer model it measured. */
  bends?: BendOutcome[];
  /** Per-object measured boxes over the one unified floor, with which photos see each face. */
  boxes?: Record<string, LayerBox>;
};

export type BoxLevel = "high" | "medium" | "low" | "unverified";
export type BoxFaceName = "front" | "back" | "left" | "right" | "top" | "bottom";
export type BoxDimName = "L" | "W" | "H" | "bottom";
export type BoxFace = { photos: number[]; status: string; confidence: BoxLevel; need: DisplayMessage | null };
export type BoxDim = { valueM: number; sigmaCm: number | null; confidence: BoxLevel };
type V3 = [number, number, number];
/** One object's measured box over the one unified floor, as the layer publishes it. axes = [l, w, u] unit native vectors (u = the
 * floor's up normal); faceNormals = each named face's outward unit normal; sizeM = [L, W, H]; bottomM / topM = heights above the
 * floor. Face names, dimensions and the highlight (shown as low confidence, retake photos) are the layer's; the viewer only draws. */
export type LayerBox = {
  label: DisplayMessage; centerNative: V3; axes: [V3, V3, V3]; faceNormals: Record<BoxFaceName, V3>; sizeM: V3;
  bottomM: number; topM: number; floorContact: boolean; snapNote?: DisplayMessage | null;
  dims: Record<BoxDimName, BoxDim>; faces: Partial<Record<BoxFaceName, BoxFace>>;
  highlight: boolean; highlightReasons?: DisplayMessage[]; confidence: BoxLevel; method?: string;
};
export const boxFaceNames: BoxFaceName[] = ["front", "back", "left", "right", "top", "bottom"];
export const boxDimNames: BoxDimName[] = ["L", "W", "H", "bottom"];
export const boxLevels: BoxLevel[] = ["high", "medium", "low", "unverified"];

const isLevel = (value: unknown): value is BoxLevel => boxLevels.includes(value as BoxLevel);
const vec = (value: unknown, n: number): value is number[] => Array.isArray(value) && value.length === n && value.every(Number.isFinite);
const isUnit = (value: unknown) => vec(value, 3) && Math.abs(Math.hypot(...value) - 1) < 1e-3;
const optText = (value: unknown) => value == null || validMessage(value);
const optTexts = (value: unknown) => value == null || Array.isArray(value) && value.every(validMessage);
const dot3 = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const along = (p: number[], d: number[], s: number): V3 => [p[0] + d[0] * s, p[1] + d[1] * s, p[2] + d[2] * s];
/** The box axis (0 l, 1 w, 2 u) a face's layer normal lies along and whether it is the positive side; null when it lies along none. */
function faceSide(box: LayerBox, name: BoxFaceName): [number, boolean] | null {
  const n = box.faceNormals[name];
  for (let k = 0; k < 3; k++) { const d = dot3(n, box.axes[k]); if (Math.abs(d) > .99) return [k, d > 0]; }
  return null;
}
/** The layer file comes from the website: a box with a malformed field is left out, never drawn half-way. */
export function validBox(box: unknown): box is LayerBox {
  const b = box as LayerBox;
  if (!b || typeof b !== "object" || !vec(b.centerNative, 3) || !Array.isArray(b.axes) || b.axes.length !== 3 || !b.axes.every(isUnit) ||
    !vec(b.sizeM, 3) || !b.sizeM.every(v => v >= 0) || !Number.isFinite(b.bottomM) || !Number.isFinite(b.topM) || b.bottomM > b.topM ||
    typeof b.floorContact !== "boolean" || typeof b.highlight !== "boolean" || !isLevel(b.confidence) ||
    !validMessage(b.label) || !optText(b.snapNote) || !optTexts(b.highlightReasons) ||
    !b.dims || typeof b.dims !== "object" || !b.faces || typeof b.faces !== "object" || !b.faceNormals || typeof b.faceNormals !== "object") return false;
  const dimsOk = boxDimNames.every(k => { const d = b.dims[k]; return !!d && Number.isFinite(d.valueM) && (d.sigmaCm === null || Number.isFinite(d.sigmaCm) && d.sigmaCm >= 0) && isLevel(d.confidence); });
  const facesOk = boxFaceNames.every(name => { const f = b.faces[name]; return f === undefined || !!f && Array.isArray(f.photos) && f.photos.every(Number.isFinite) && typeof f.status === "string" && isLevel(f.confidence) && (f.need === null || validMessage(f.need)); });
  // six unit normals, each along a box axis, on six different sides
  const sides = boxFaceNames.map(name => isUnit(b.faceNormals[name]) ? faceSide(b, name) : null);
  return dimsOk && facesOk && sides.every(Boolean) && new Set(sides.map(side => side!.join())).size === 6 && sides[4]!.join() === "2,true";  // top is +u
}

/** Corner i of the box is on the positive side of axis k (l, w, u) when bit k is set; each face lists its four corners in order,
 * picked by the face's layer normal. With a floor the box spans bottomM..topM above it along u, so the drawn box always agrees with
 * the clearance figures; null when u is not that floor's up normal (its heights belong to another floor). Without a floor: centre ± size / 2. */
export function boxGeometry(box: LayerBox, nativeToMeters: number, ground: { normal: number[]; offset: number } | null | undefined) {
  const s = nativeToMeters, c = box.centerNative, [l, w, u] = box.axes, half = box.sizeM.map(v => v / s / 2), n = ground?.normal;
  const norm = vec(n, 3) && Number.isFinite(ground!.offset) ? Math.hypot(...n) : 0;
  if (norm > 1e-9 && dot3(u, n!) / norm <= .99) return null;
  const base = norm > 1e-9 ? along(c, n!, -(dot3(n!, c) + ground!.offset) / norm / norm) : c, z = norm > 1e-9 ? [box.bottomM / s, box.topM / s] : [-half[2], half[2]];
  const corners = [0, 1, 2, 3, 4, 5, 6, 7].map(i => along(along(along(base, l, (i & 1 ? 1 : -1) * half[0]), w, (i & 2 ? 1 : -1) * half[1]), u, z[i >> 2]));
  const faces = Object.fromEntries(boxFaceNames.map(name => {
    const [k, positive] = faceSide(box, name)!, [a, b] = [0, 1, 2].filter(i => i !== k), o = positive ? 1 << k : 0;
    return [name, [o, o | 1 << a, o | 1 << a | 1 << b, o | 1 << b]];
  })) as Record<BoxFaceName, number[]>;
  return { corners, faces, bottomCenter: along(base, u, z[0]), floor: norm > 1e-9 ? base : null };
}

/** The report's resources with the layer's own files served from the website and its measured bends; everything else unchanged. */
export function withLayerAssets<T extends { resolveAsset: (id: string) => Promise<string>; layerBends?: BendOutcome[]; layerConfidence?: Record<string, LayerConfidence> }>(resources: T, layer: MeasurementLayer | null): T {
  const urls = new Map((layer?.assets || []).map(asset => [asset.id, new URL(asset.url, location.href).href]));
  if (!urls.size && !layer?.bends?.length && !layer?.confidence) return resources;
  return { ...resources, layerBends: layer?.bends, layerConfidence: layer?.confidence, resolveAsset: async (id: string) => urls.get(id) ?? resources.resolveAsset(id) };
}

/** The revision's data-side display names and scale/ground explanations in the viewer's current language. */
export function withLocalizedLabels(revision: Revision, layer: MeasurementLayer | null, language: Language): Revision {
  if (!layer || layer.schemaVersion !== 2 || layer.revisionId !== revision.id) return revision;
  const entities = revision.document.entities.map(entity => {
    const message = layer.labels?.[entity.id], label = message && renderMessage(language, message);
    return label && label !== entity.label ? { ...entity, label } : entity;
  });
  const coordinateFrames = revision.document.coordinateFrames.map(frame => frame.id !== layer.coordinateFrameId ? frame : {
    ...frame, scale: { ...frame.scale, source: renderMessage(language, layer.scale.source) },
    ground: frame.ground && layer.ground ? { ...frame.ground, source: renderMessage(language, layer.ground.source) } : frame.ground,
  });
  return { ...revision, document: { ...revision.document, entities, coordinateFrames } };
}

export async function loadMeasurementLayer(publicationId: string, variant?: string | null): Promise<MeasurementLayer | null> {
  const suffix = variant && /^[a-z0-9-]{1,32}$/.test(variant) ? `.${variant}` : "";
  try {
    const response = await fetch(`./measurement-layer/${encodeURIComponent(publicationId)}${suffix}.json`, { cache: "no-cache" });
    if (!response.ok) return null;
    const layer = await response.json() as MeasurementLayer;
    if (layer.schemaVersion !== 2) return null;
    const messages = [layer.scale?.source, ...(layer.ground ? [layer.ground.source] : []),
      ...Object.values(layer.labels || {}), ...Object.values(layer.models || {}).map(model => model.note),
      ...Object.values(layer.facts || {}).flatMap(facts => facts.flatMap(fact => [fact.label, fact.text])),
      ...Object.values(layer.confidence || {}).flatMap(confidence => [confidence.label, ...(confidence.reasons || []), ...(confidence.missing || [])]),
      ...Object.values(layer.pipelines || {}).flatMap(pipeline => [...(pipeline.caption ? [pipeline.caption] : []), ...pipeline.stages.flatMap(stage => [stage.label, stage.text])]),
      ...(layer.variant ? [layer.variant.label] : []),
      ...Object.values(layer.boxes || {}).flatMap(box => [box.label, ...(box.snapNote ? [box.snapNote] : []), ...(box.highlightReasons || []), ...Object.values(box.faces).flatMap(face => face?.need ? [face.need] : [])])];
    return messages.every(validMessage) ? layer : null;
  } catch {
    return null;  // ponytail: no layer published for this report; show it as frozen
  }
}

export function applyMeasurementLayer(revision: Revision, layer: MeasurementLayer | null): Revision {
  if (!layer || layer.schemaVersion !== 2 || layer.revisionId !== revision.id) return revision;
  const document = revision.document;
  const coordinateFrames = document.coordinateFrames.map(frame => frame.id !== layer.coordinateFrameId ? frame : {
    ...frame,
    scale: { ...frame.scale, status: layer.scale.status, nativeToMeters: layer.scale.nativeToMeters, source: renderMessage("en", layer.scale.source) },
    ground: layer.ground ? { ...(frame.ground || {}), normal: layer.ground.normal, plane: layer.ground.plane, offset: layer.ground.offset, source: renderMessage("en", layer.ground.source) } : frame.ground,
  });
  const entities = document.entities.map(entity => {
    const label = layer.labels?.[entity.id];
    if (label) entity = { ...entity, label: renderMessage("en", label) };
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
