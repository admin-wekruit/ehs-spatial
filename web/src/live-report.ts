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
    doc.entities.push({ id: o.id, label: o.label || o.word, associationState: "association_pending", visible: true, observationRefs: [],
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

// ---------------------------------------------------------------- click MVP (docs/phase2/CLICK-MVP-SPEC.md sections 3, 4.9, 5.6)
// pick: which entity is under each pixel of each keyframe (uint16 (value, run) pairs, gzip) + nearest depth per 4x4 DA3 block.
export type PickFrame = { t: number; t_end: number; frame: number; shot: number | null; key: number | null; w: number; h: number; source: string; offset: number; pairs: number };
export type PickData = { format: string; source_wh: [number, number]; entities: (string | null)[]; frames: PickFrame[]; depth?: { w: number; h: number; unit: string; scale: string }; fixture?: string };
export type Pick = { data: PickData; runs: Uint16Array; depth: Uint16Array | null; unit: number };
export type Verdict = "FAIL" | "NEEDS_REVIEW" | "NO_DATA" | "PASS";
/** Severity order; NO_DATA ranks above PASS (the worst_verdict fix: one PASS and 59 NO_DATA is not a PASS). */
export const SEVERITY: Verdict[] = ["FAIL", "NEEDS_REVIEW", "NO_DATA", "PASS"];
export const worstVerdict = (vs: (string | null | undefined)[]) =>
  (vs.filter(Boolean) as Verdict[]).sort((a, b) => SEVERITY.indexOf(a) - SEVERITY.indexOf(b))[0] ?? null;
/** pointer-down time of the last video click (performance.now()); the page reads it once the card is in the DOM. */
export const clickClock = { t0: 0 };

export async function gunzip(bytes: ArrayBuffer | Uint8Array): Promise<ArrayBuffer> {
  return new Response(new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}

/** Frame offsets are counted in pairs (the fixture); a writer counting uint16s or bytes also reads right: the unit is the one that
 *  makes the first frames' runs add up to w*h. */
export function readPick(data: PickData, runs: ArrayBuffer, depth?: ArrayBuffer | null): Pick {
  const r = new Uint16Array(runs), total = r.length / 2;
  const adds = (unit: number) => data.frames.slice(0, 3).every(f => {
    const o = f.offset / unit;
    if (!Number.isInteger(o) || o + f.pairs > total) return false;
    let n = 0;
    for (let i = 0; i < f.pairs; i++) n += r[2 * (o + i) + 1];
    return n === f.w * f.h;
  });
  const unit = [1, 2, 4].find(adds);
  if (!unit) throw Error("pick_layout");
  return { data, runs: r, depth: depth ? new Uint16Array(depth) : null, unit };
}

/** The last frame with t <= time (the first one before it), as VideoView.frameAt. */
export function frameIndexAt(frames: { t: number }[], time: number) {
  let low = 0, high = frames.length - 1;
  while (low < high) { const middle = (low + high + 1) >> 1; if (frames[middle].t <= time) low = middle; else high = middle - 1; }
  return low;
}

/** Source pixel (x, y) at video time t -> the entity there. O(pairs of that frame); ponytail: no frame cache, a scan is < 0.1 ms. */
export function pickAt(pick: Pick, t: number, x: number, y: number) {
  const index = frameIndexAt(pick.data.frames, t), f = pick.data.frames[index], [W, H] = pick.data.source_wh;
  if (f.t_end != null && t >= f.t_end) return { index, frame: f, id: null as string | null, gap: true };  // past a cut: no map
  const col = Math.min(f.w - 1, Math.max(0, Math.floor(x * f.w / W))), row = Math.min(f.h - 1, Math.max(0, Math.floor(y * f.h / H)));
  let rest = row * f.w + col;
  const o = f.offset / pick.unit;
  for (let k = 0; k < f.pairs; k++) {
    const run = pick.runs[2 * (o + k) + 1];
    if (rest < run) { const v = pick.runs[2 * (o + k)]; return { index, frame: f, id: v ? pick.data.entities[v] ?? null : null }; }
    rest -= run;
  }
  return { index, frame: f, id: null as string | null };
}

/** One entity's pixels on one pick frame (w*h, 1 = the entity), for the highlight. */
export function pickMask(pick: Pick, index: number, id: string) {
  const f = pick.data.frames[index], value = pick.data.entities.indexOf(id), mask = new Uint8Array(f.w * f.h), o = f.offset / pick.unit;
  if (value < 1) return null;
  for (let k = 0, at = 0; k < f.pairs; k++) {
    const run = pick.runs[2 * (o + k) + 1];
    if (pick.runs[2 * (o + k)] === value) mask.fill(1, at, at + run);
    at += run;
  }
  return { w: f.w, h: f.h, mask };
}

type Cam = { K: number[][]; c2w: number[][]; wh: [number, number]; source_wh: [number, number]; floor?: { normal: number[]; point_m: number[] } };
const sub = (a: number[], b: number[]) => a.map((v, i) => v - b[i]);
const norm = (a: number[]) => Math.hypot(...a);
const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a: number[]) => { const n = norm(a) || 1; return a.map(v => v / n); };

/** A keyframe's camera from the cameras layer (K on the DA3 grid, camera-to-world in estimated metres). */
export function cameraOf(cameras: any, frame: PickFrame): Cam | null {
  const s = (cameras?.shots || []).find((x: any) => x.index === frame.shot) || (cameras?.shots || []).find((x: any) => x.keys.includes(frame.frame));
  if (!s) return null;
  const i = frame.key ?? s.keys.indexOf(frame.frame);
  return i < 0 || !s.K[i] ? null : { K: s.K[i], c2w: s.c2w[i], wh: s.wh, source_wh: s.source_wh, floor: s.floor };
}

/** Source pixel + depth along the optical axis (m) -> world point (the shot's frame, estimated metres). */
export function unproject(cam: Cam, x: number, y: number, z: number) {
  const u = x * cam.wh[0] / cam.source_wh[0], v = y * cam.wh[1] / cam.source_wh[1], K = cam.K, M = cam.c2w;
  const p = [(u - K[0][2]) / K[0][0] * z, (v - K[1][2]) / K[1][1] * z, z];
  return [0, 1, 2].map(r => M[r][0] * p[0] + M[r][1] * p[1] + M[r][2] * p[2] + M[r][3]);
}

/** The shot's floor frame (spec 4.1): the cards layer's when present, else from the cameras layer's first camera and floor. */
export function floorFrame(cameraShot: any, cardsShot?: any) {
  if (cardsShot?.floor_frame) { const f = cardsShot.floor_frame; return { origin: f.origin_m, x: f.x, z: f.z, y: cross(f.z, f.x) }; }
  const M = cameraShot.c2w[0], c = [M[0][3], M[1][3], M[2][3]], p = cameraShot.floor.point_m;
  let n = cameraShot.floor.normal;
  if (n.reduce((s: number, v: number, i: number) => s + v * (c[i] - p[i]), 0) < 0) n = n.map((v: number) => -v);
  const h = n.reduce((s: number, v: number, i: number) => s + v * (c[i] - p[i]), 0), f = [M[0][2], M[1][2], M[2][2]];
  const fn = f.reduce((s, v, i) => s + v * n[i], 0), x = unit(f.map((v, i) => v - fn * n[i]));
  return { origin: c.map((v, i) => v - h * n[i]), x, z: n, y: cross(n, x) };
}
export const toFloor = (F: ReturnType<typeof floorFrame>, P: number[]) => { const d = sub(P, F.origin); return [F.x, F.y, F.z].map(a => a[0] * d[0] + a[1] * d[1] + a[2] * d[2]); };

/** Nearest valid depth (m) within `reach` cells of the click, and its cell. */
export function depthAt(pick: Pick, index: number, x: number, y: number, reach = 2) {
  const g = pick.data.depth, d = pick.depth;
  if (!g || !d) return null;
  const [W, H] = pick.data.source_wh, c = Math.floor(x * g.w / W), r = Math.floor(y * g.h / H), base = index * g.w * g.h;
  let best: { d2: number; v: number; row: number; col: number } | null = null;
  for (let dr = -reach; dr <= reach; dr++) for (let dc = -reach; dc <= reach; dc++) {
    const row = r + dr, col = c + dc;
    if (row < 0 || col < 0 || row >= g.h || col >= g.w) continue;
    const v = d[base + row * g.w + col], d2 = dr * dr + dc * dc;
    if (v && (!best || d2 < best.d2)) best = { d2, v, row, col };
  }
  return best && { m: best.v / 1000, row: best.row, col: best.col };
}

const pointInPolygon = (p: number[], poly: number[][]) => {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++)
    if ((poly[i][1] > p[1]) !== (poly[j][1] > p[1]) && p[0] < (poly[j][0] - poly[i][0]) * (p[1] - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0]) inside = !inside;
  return inside;
};
export { pointInPolygon };
const segDist = (p: number[], a: number[], b: number[]) => {
  const ab = sub(b, a), t = Math.max(0, Math.min(1, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / ((ab[0] ** 2 + ab[1] ** 2) || 1)));
  return Math.hypot(p[0] - a[0] - t * ab[0], p[1] - a[1] - t * ab[1]);
};
/** 2D distance from a point to a footprint polygon (0 inside). */
export const footprintDistance = (p: number[], poly: number[][]) =>
  pointInPolygon(p, poly) ? 0 : Math.min(...poly.map((a, i) => segDist(p, a, poly[(i + 1) % poly.length])));

/** A click on no entity (spec 3.4): distance, height above the floor with its u, the surface type from the local normal, the
 *  nearest carded entity. u: depth 5% of distance (height: times the ray's vertical share) + floor residual, and the 20%
 *  scale term of spec 4.4 in quadrature (the scale is estimated). */
export function unknownRegion(pick: Pick, cameras: any, cardsLayer: any, t: number, x: number, y: number) {
  const index = frameIndexAt(pick.data.frames, t), frame = pick.data.frames[index];
  if (frame.t_end != null && t >= frame.t_end) return { status: "no pick map at this moment (past a shot change)" as const, t: frame.t, x, y };
  const cam = cameraOf(cameras, frame), hit = depthAt(pick, index, x, y);
  if (!cam || !hit) return { status: "no 3D point here" as const, t: frame.t, x, y };
  const camShot = cameras.shots.find((s: any) => s.keys.includes(frame.frame)), cardsShot = cardsLayer?.shots?.find((s: any) => s.index === camShot.index);
  const F = floorFrame(camShot, cardsShot), P = unproject(cam, x, y, hit.m), c = [cam.c2w[0][3], cam.c2w[1][3], cam.c2w[2][3]];
  const ray = sub(P, c), distance = norm(ray), height = toFloor(F, P)[2], vshare = Math.abs(unit(ray).reduce((s, v, i) => s + v * F.z[i], 0));
  const uFloor = cardsShot?.u_floor_m ?? 0, g = pick.data.depth!, [W, H] = pick.data.source_wh;
  const cellPoint = (row: number, col: number) => {
    const v = pick.depth![index * g.w * g.h + row * g.w + col];
    return v && row >= 0 && col >= 0 && row < g.h && col < g.w ? unproject(cam, (col + .5) * W / g.w, (row + .5) * H / g.h, v / 1000) : null;
  };
  const [l, r, u, d] = [cellPoint(hit.row, hit.col - 1), cellPoint(hit.row, hit.col + 1), cellPoint(hit.row - 1, hit.col), cellPoint(hit.row + 1, hit.col)];
  // mvp2 (R7): a class only, never a number: one keyframe's 3x3 depth cells are one view (the policy wants two agreeing
  // view sets for an angle), and the classes' 15 / 75 degree bands are wider than its depth noise near the camera
  let surface: { kind: string } = { kind: "no surface estimate (a neighbouring cell has no depth)" };
  if (l && r && u && d) {
    const n = unit(cross(sub(r, l), sub(d, u))), tilt = Math.acos(Math.min(1, Math.abs(n.reduce((s, v, i) => s + v * F.z[i], 0)))) * 180 / Math.PI;
    surface = tilt <= 15 ? { kind: height < .1 ? "floor" : "horizontal surface" } : tilt >= 75 ? { kind: "vertical surface" } : { kind: "sloped surface" };
  }
  const here = toFloor(F, P).slice(0, 2);
  const nearest: any = (cardsLayer?.cards || []).filter((k: any) => k.shot === camShot.index && k.kind === "object" && k.physical?.size_check?.status !== "implausible")
    .map((k: any) => { const fp = k.physical?.footprint_xy, poly = Array.isArray(fp) ? fp : fp?.value;  // A: {value: [[x, y], ...], unit, frame}
      return { id: k.id, name: k.identity?.name, d: poly?.length ? footprintDistance(here, poly) : Array.isArray(k.physical?.position_xy?.value) ? norm(sub(here, k.physical.position_xy.value)) : Infinity,
               uFoot: (poly?.length ? fp?.u : k.physical?.position_xy?.u) ?? 0 }; })
    .filter((k: any) => Number.isFinite(k.d)).sort((a: any, b: any) => a.d - b.d)[0] || null;
  if (nearest) {  // mvp2 (R7): the distance's u: this point's depth (horizontal share), the footprint's own u, the 20% scale term
    const hshare = Math.sqrt(Math.max(0, 1 - vshare * vshare)), ud = .05 * distance * hshare;
    nearest.distance = { value: nearest.d, u: Math.hypot(ud, nearest.uFoot, .2 * nearest.d), parts: { depth: ud, footprint: nearest.uFoot, scale: .2 * nearest.d } };
  }
  return {
    status: "depth" as const, t: frame.t, x, y, frame: frame.frame, shot: camShot.index, surface, nearest, point_floor: toFloor(F, P),
    distance: { value: distance, u: Math.hypot(.05 * distance, .2 * distance), parts: { depth: .05 * distance, scale: .2 * distance } },
    height: { value: height, u: Math.hypot(.05 * distance * vshare + uFloor, .2 * Math.abs(height)), parts: { depth: .05 * distance * vshare, floor: uFloor, scale: .2 * Math.abs(height) } },
  };
}

export type Info = { card: any | null; rows: any[]; verdict: Verdict | null };
/** Cards and judgement rows by entity id (a merged object's old ids point at it too: the layer's aliases {old: id} and each card's
 *  merged_from); the verdict is the layer's by_object, else the worst row. Entities with neither are absent. */
export function entityInfo(cardsLayer: any, judgements: any): Map<string, Info> {
  const out = new Map<string, Info>();
  const get = (id: string) => { if (!out.has(id)) out.set(id, { card: null, rows: [], verdict: null }); return out.get(id)!; };
  for (const c of cardsLayer?.cards || []) { const i = get(c.id); i.card = c; for (const old of c.physical?.merged_from || []) out.set(old, i); }
  for (const [old, id] of Object.entries(cardsLayer?.aliases || {})) if (out.has(id as string) && !out.has(old)) out.set(old, out.get(id as string)!);
  for (const r of judgements?.rows || []) get(r.subject).rows.push(r);
  for (const [id, i] of out) if (id === i.card?.id || !i.card) i.verdict = judgements?.by_object?.[id] ?? worstVerdict(i.rows.map(r => r.verdict));
  return out;
}
