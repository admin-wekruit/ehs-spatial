// Fast report patches (fast_report/layers.py) -> the smallest legal SceneDocument the report viewer draws. One coordinate frame
// per shot ("shot-<i>", estimated metres); every layer's newest version replaces the older ones. Check: tests/live-report-check.ts.
import type { SceneDocument } from "./types";

export type BlobRef = { sha256: string; bytes: number; mediaType?: string; format?: string; byteLayout?: unknown; pointSizeNative?: number };
export type Patch = {
  schema: string; report: string; seq: number; layer: string; version: number; status: string; labels: string[];
  t0_unix: number; queued_s?: number; sent_s: number; data: any; blobs: Record<string, BlobRef>;
};
export type Poll = { patches: Patch[]; written: Record<string, number>; served: Record<string, number>; run: any };

export const assetURL = (id: string) => id.startsWith("sha256:") ? "/fast/blobs/" + id.slice(7) : null;
export const frameOf = (shot: number) => "shot-" + shot;

export async function poll(report: string, after: number): Promise<Poll> {
  const response = await fetch(`/fast/reports/${encodeURIComponent(report)}/patches?after=${after}`, { cache: "no-store" });
  if (!response.ok) throw Error("live_poll_failed");
  return response.json();
}

/** The newest version of each layer. */
export function latest(patches: Patch[]) {
  const out: Record<string, Patch> = {};
  for (const p of patches) if (!out[p.layer] || out[p.layer].seq < p.seq) out[p.layer] = p;
  return out;
}

const identity = (frame: string) => ({ coordinateFrameId: frame, position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: [1, 1, 1] });
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export function liveDocument(report: string, layers: Record<string, Patch>): SceneDocument {
  const doc: any = { schemaVersion: 2, target: "scene", captureId: report, coordinateFrames: [], cameras: [], observations: [], entities: [],
    assets: [], annotations: [], geometryBindings: {} };
  const asset = (ref: BlobRef) => {
    const id = "sha256:" + ref.sha256;
    if (!doc.assets.some((a: any) => a.id === id))
      doc.assets.push({ id, sha256: ref.sha256, sizeBytes: ref.bytes, mediaType: ref.mediaType, format: ref.format, byteLayout: ref.byteLayout,
        pointSizeNative: ref.pointSizeNative, metadata: { format: ref.format, byteLayout: ref.byteLayout, pointSizeNative: ref.pointSizeNative } });
    return id;
  };
  const { cameras, room, people, objects, models, splat, video, outlines, events } = layers;
  // The longest shot first: the viewer opens on the first camera's frame.
  for (const s of [...(cameras?.data.shots || [])].sort((a: any, b: any) => b.keys.length - a.keys.length)) {
    const frame = frameOf(s.index), n = s.floor?.normal, offset = n ? -dot(n, s.floor.point_m) : undefined;
    doc.coordinateFrames.push({ id: frame, convention: "opencv", scale: { status: "model_estimated", nativeToMeters: 1, source: cameras.data.scale?.source },
      ground: n ? { normal: n, offset, plane: [...n, offset] } : null });
    s.keys.forEach((key: number, i: number) => {
      const imageId = "frame:" + key, id = `camera:${s.index}:${key}`;
      doc.assets.push({ id: imageId, kind: "source_image", metadata: { sourceFrame: key, videoTimestamp: s.times[i] } });
      doc.cameras.push({ id, imageId, coordinateFrameId: frame, width: s.wh[0], height: s.wh[1], K: s.K[i], cameraToWorld: s.c2w[i] });
      doc.geometryBindings[imageId] = { geometrySolutionId: report, cameraId: id };
    });
  }
  for (const s of room?.data.shots || []) {
    const frame = frameOf(s.index), mesh = room.blobs["mesh-" + s.index], points = room.blobs["points-" + s.index];
    doc.entities.push({ id: "room:" + s.index, label: "room surface (estimated)", sourceContext: true, associationState: "association_pending", visible: true,
      observationRefs: [], fast: { kind: "room", ...s }, representations: [
        mesh && { id: "room-mesh:" + s.index, kind: "observed_surface", assetId: asset(mesh), coordinateFrameId: frame, transform: identity(frame), placementState: "confirmed", sourceRefs: [] },
        points && { id: "room-points:" + s.index, kind: "point_cloud", assetId: asset(points), coordinateFrameId: frame, transform: identity(frame), placementState: "confirmed", sourceRefs: [] },
      ].filter(Boolean) });
  }
  const accepted = new Map<string, any>((models?.data.models || []).map((m: any) => [m.object, m]));
  for (const o of objects?.data.objects || []) {
    const frame = frameOf(o.shot), min = o.box_min_m, max = o.box_max_m, model = accepted.get(o.id), glb = model && models.blobs["model-" + o.id];
    // Detected words are names to check, never verified: a see-through box, the model when SAM 3D's gate took one.
    const box = { id: "box:" + o.id, kind: "primitive", primitive: { kind: "box", dimensions: [0, 1, 2].map(k => Math.max(max[k] - min[k], .01)) },
      coordinateFrameId: frame, transform: { ...identity(frame), position: [0, 1, 2].map(k => (min[k] + max[k]) / 2) }, placementState: "confirmed",
      material: { alphaMode: "BLEND", baseColorFactor: [1, 1, 1, .05], color: [.45, .9, .8] } };
    const reps: any[] = [box];
    if (glb) reps.push({ id: "model:" + o.id, kind: "generated_mesh", assetId: asset(glb), coordinateFrameId: frame,
      transform: { ...identity(frame), ...model.transform }, placementState: "confirmed", bounds: model.bounds });
    doc.entities.push({ id: o.id, label: o.word, associationState: "association_pending", visible: true, observationRefs: [],
      activeModelRepresentationId: glb ? "model:" + o.id : box.id, representations: reps, fast: { kind: "object", ...o, model: model || null } });
  }
  for (const t of people?.data.tracks || []) {
    const ref = people.blobs["track-" + t.id];
    if (!ref) continue;
    const frame = frameOf(t.shot), rules = (people.data.rules || []).filter((r: any) => r.track === undefined || r.track === t.id);
    // The viewer's model class draws it and lets it be picked; its status stays observed+estimated (fast.kind).
    doc.entities.push({ id: "person:" + t.id, label: "person " + t.id, associationState: "association_pending", visible: true, observationRefs: [],
      activeModelRepresentationId: "track:" + t.id, fast: { kind: "person", ...t, points: undefined, rules },
      representations: [{ id: "track:" + t.id, kind: "generated_mesh", sourceKind: "people_track", assetId: asset(ref), coordinateFrameId: frame,
        transform: identity(frame), placementState: "confirmed" }] });
  }
  if (splat?.blobs.splat)
    doc.annotations.push({ id: "splat", kind: "gaussian_splats", format: splat.data.format, assetId: asset(splat.blobs.splat), count: splat.data.count,
      coordinateFrameId: frameOf(splat.data.shot), splatKind: splat.data.kind });
  if (video?.blobs.video) {
    const analysis = outlines?.blobs.analysis ? asset(outlines.blobs.analysis) : outlines?.data.analysis ? "inline:outlines:" + outlines.seq : undefined;
    if (analysis?.startsWith("inline:")) doc.assets.push({ id: analysis, inline: outlines.data.analysis });
    doc.annotations.push({ id: "video", kind: "video_replay", videoAssetId: asset(video.blobs.video), analysisAssetId: analysis });
  }
  if (events) doc.annotations.push({ id: "events", kind: "video_events", model: events.data.model, windows: events.data.windows });
  return doc as SceneDocument;
}
