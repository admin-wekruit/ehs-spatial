"""E3 scoring on the CPU (light): fast-path people against today's person layer, and movers that are not people.

Reads a probe run (modal_apps/e3_people_probe.py: detections.json, tracks-stride3.npz, probe.json). Nothing is rerun on
a GPU here.

(a) The live people loop (ehs_spatial.live_people.PeopleLoop, the rules of video.judge_frame) replays ME340 through
    scripts/replay_people_stream.py's stream (DROID poses, DA3 posed depth, the stated 1.6 m carry height) with:
      ref10   today's cached SAM 3.1 person masks (run 187), every depth frame (~10 Hz): the reference
      fast5   SAM 3 image 'person' at 5 Hz: the fast path
      sam3_10 SAM 3 at 10 Hz and cached5 today's masks at 5 Hz: detector vs cadence
      trk3    SAM 3.1 text tracks at stride 3 (10 Hz): the background tier
    Scored: each row's visible-surface centroid against today's centroid (runs/me340-cutaway-dynamic-305) of the same
    frame, on the floor; R1/R2/R3 every 0.2 s against ref10, before and after the scale gate.
(c) Today's motion-session tracks (runs 178-180) against SAM 3 'person' and mover-word masks; and a static-scene
    reprojection residual (DA3 depth + DROID poses, frames 0.2 s apart) as a mover seed, scored against today's person
    masks and motion_masks.py's trusted motion (run 177).

  python scripts/e3_people_eval.py --probe PROBE_DIR --output NEW_DIR
  python scripts/e3_people_eval.py --self-check
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "modal_apps")]
import replay_people_stream as R  # noqa: E402
from ehs_spatial.live_people import PeopleLoop, decode_frame, run  # noqa: E402
from ehs_spatial.providers.sam3 import decode_coco_rle  # noqa: E402
from ehs_spatial.video import R3_MAX_SPEED_MPS, SPEED_BAND_MPS, banded_verdict, contract_scale, track_speed  # noqa: E402

PHASE2 = R.PHASE2
ZONE = "POLYGON ((1.8 -3.9, 2.8 -3.9, 2.8 -2.9, 1.8 -2.9, 1.8 -3.9))"  # the m0 replay's test zone (m0-people-fix-me340-zone)
MOVER_WORDS = ("forklift", "pallet jack", "cart", "vehicle")
TRACK_RUNS = {"178": (0, 300), "179": (260, 560), "180": (520, 899)}
WALK_START = 228  # tracks-stride3.npz frame keys count from here
MATCH_M = 1.0  # a fast row farther than this from every person of today's layer at that frame is another detection
TAU, EDGE_JUMP, CAPTION_ROW = 0.10, 0.03, 440  # residual: relative depth excess; flying-pixel rule (mono_room); caption band
BLOB, SEED = .003, .02  # motion_masks.py blob and sam3_motion_tracks.py seed, as shares of the frame


def decode(rle):
    return decode_coco_rle(rle, height=480, width=640).astype(bool)


def sam3_detector(probe, words=("person",), called=None, dedupe=False):
    """SAM 3 masks of the probe as a live detector. dedupe: a mask at least half inside a higher-scoring one of the same
    word is the same object seen twice (SAM 3 sometimes returns the presenter twice) and is dropped."""
    frames = json.loads((probe / "detections.json").read_text())["frames"]

    def detect(frame):
        if called is not None:
            called.append(frame["frame"])
        found = frames.get(str(frame["frame"]))
        if found is None:
            return None  # never asked: a gap, not an empty scene
        out = []
        for w in words:
            kept = np.zeros((480, 640), bool)
            for n, (r, s) in sorted(enumerate(zip(found["rle"][w], found["scores"][w])), key=lambda x: -x[1][1]):
                m = decode(r)
                if dedupe and (m & kept).sum() >= 0.5 * m.sum():
                    continue
                kept |= m
                out.append({"label": w, "source": f"sam3-{w}-{n}", "score": s, "mask": m})
        return out
    return detect, {int(k) for k in frames}


def tracker_detector(probe, called=None):
    data = np.load(probe / "tracks-stride3.npz")
    frames = {}
    for key in data.files:
        if key != "report":
            frame, obj = key.split("/")
            frames.setdefault(WALK_START + int(frame), []).append((int(obj), key))

    def detect(frame):
        if called is not None:
            called.append(frame["frame"])
        return [{"label": "person", "source": f"sam31-{obj}", "score": None,
                 "mask": np.unpackbits(data[key])[:480 * 640].reshape(480, 640).astype(bool)}
                for obj, key in frames.get(frame["frame"], [])]
    return detect, set(range(WALK_START, 898, 3))


def cached(called=None):
    inner = R.cached_detector()

    def detect(frame):
        if called is not None:
            called.append(frame["frame"])
        return inner(frame)
    return detect


def replay(detector, rate, out, only=None):
    """One PeopleLoop pass over the replay stream, as replay_people_stream.replay builds it; `only` drops posed frames
    the detector was never run on (the tracker's stride grid)."""
    scale = json.loads((R.DEPTH / "metric-scale.json").read_text())
    loop = PeopleLoop(np.array(scale["plane_point_native"]) * scale["metres_per_native_unit"], scale["up_native"],
                      contract_scale(scale), detector, wkt.loads(ZONE))
    frames = (decode_frame(m) for m in R.stream(rate))
    if only is not None:
        frames = (f for f in frames if f["cameraToWorld"] is None or f["frame"] in only)
    started = time.perf_counter()
    counts = run(frames, loop, out)
    counts["wall_s"] = round(time.perf_counter() - started, 2)
    return counts


def rows_of(folder, name):
    return [json.loads(line) for line in (folder / name).read_text().splitlines()]


def offline_people():
    """frame -> [(entity, centroid metres)] of today's layer, people only (replay_people_stream.cached_detector's rule:
    an entity 'person' ever named; the subtitle track 180-motion-4 is not one)."""
    analysis = json.loads((R.MASKS / "analysis.json").read_text())
    people = {o["entityId"] for f in analysis["frames"] for o in f["objects"] if o["label"] == "person"}
    metres = json.loads((R.DEPTH / "metric-scale.json").read_text())["metres_per_native_unit"]
    found = {}
    for f in json.loads(R.OFFLINE.read_text())["frames"]:
        for o in f["objects"]:
            if o.get("centroid") and o["entityId"] in people:
                found.setdefault(f["sourceFrame"], []).append((o["entityId"], np.array(o["centroid"]) * metres))
    return found


def stats(values):
    v = np.asarray(values, float)
    return {"n": int(v.size), **({"median": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3),
                                  "max": round(float(v.max()), 3)} if v.size else {})}


def path_vs_offline(folder, frames_run):
    """Every row's visible-surface centroid against today's nearest person centroid at the same frame: floor-plane and 3D
    distance; rows with no person of today's layer within MATCH_M are extra, today's people with no row are missed.
    Also the tracks: which of today's entities each loop track followed, and how many loop tracks one entity took."""
    offline, up = offline_people(), np.asarray(json.loads((R.DEPTH / "metric-scale.json").read_text())["up_native"], float)
    up /= np.linalg.norm(up)
    floor, full, extra, used, follow, seen = [], [], 0, set(), {}, {}
    for row in rows_of(folder, "tracks.jsonl"):
        if row["label"] != "person" or row["centroidWorld"] is None:
            continue
        people = offline.get(row["frame"], [])
        if not people:
            extra += 1
            continue
        entity, ref = min(people, key=lambda p: np.linalg.norm(p[1] - row["centroidWorld"]))
        d = np.asarray(row["centroidWorld"]) - ref
        if np.linalg.norm(d) > MATCH_M:
            extra += 1
            continue
        floor.append(float(np.linalg.norm(d - (d @ up) * up)))
        full.append(float(np.linalg.norm(d)))
        used.add((row["frame"], entity))
        follow.setdefault(row["track"], []).append(entity)
        seen.setdefault(entity, []).append((row["t"], np.asarray(row["centroidWorld"])))
    # The path between samples: this replay's positions of each of today's people, linear in time between two
    # consecutive samples at most 0.45 s apart, at every frame of today's layer inside them.
    clock = [line.split() for line in (R.CLIP / "rgb.txt").read_text().splitlines() if not line.startswith("#")]
    between = []
    for entity, samples in seen.items():
        samples.sort(key=lambda s: s[0])
        for (t0, a), (t1, b) in zip(samples, samples[1:]):
            if not 0 < t1 - t0 <= 0.45:
                continue
            for f, people in offline.items():
                t = float(clock[f][0]) - float(clock[0][0])
                if t0 <= t <= t1:
                    for e, ref in people:
                        if e == entity:
                            d = a + (b - a) * (t - t0) / (t1 - t0) - ref
                            between.append(float(np.linalg.norm(d - (d @ up) * up)))
    expected = [(f, e) for f in set(frames_run) for e, _ in offline.get(f, [])]
    per_entity = {}
    for track, entities in follow.items():
        main = max(set(entities), key=entities.count)
        per_entity.setdefault(main, []).append(track)
    return {"centroid_floor_m": stats(floor), "centroid_3d_m": stats(full), "path_interpolated_floor_m": stats(between), "rows_matched": len(floor), "rows_extra": extra,
            "today_person_samples": len(expected), "today_person_samples_missed": sum(k not in used for k in expected),
            "loop_tracks_per_today_entity": {e: sorted(t) for e, t in per_entity.items()},
            "loop_track_purity": {t: round(max(map(e.count, set(e))) / len(e), 3) for t, e in follow.items()}}


def body_speeds(folder):
    """(today's entity, t to 0.2 s) -> speed of the visible-body centroid on the floor (video.track_speed, 2 s line fit).
    ME340's feet are hidden behind benches in nearly every frame, so R3 never gets a foot speed; this is the same rule
    on the body path, the diagnostic replay_people_stream.compare uses. Rows go to today's nearest person (MATCH_M)."""
    offline, samples = offline_people(), {}
    for row in rows_of(folder, "tracks.jsonl"):
        people = offline.get(row["frame"], [])
        if row["label"] != "person" or row["centroidWorld"] is None or not people:
            continue
        entity, ref = min(people, key=lambda p: np.linalg.norm(p[1] - row["centroidWorld"]))
        if np.linalg.norm(np.asarray(row["centroidWorld"]) - ref) <= MATCH_M:
            samples.setdefault(entity, []).append((row["t"], tuple(row["centroidXyM"])))
    return {(e, round(t * 5) / 5): track_speed(rows, t)[0] for e, rows in samples.items() for t, _ in rows}


def speed_agreement(ref, other):
    a, b = body_speeds(ref), body_speeds(other)
    shared = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    verdict = lambda v: banded_verdict(v, R3_MAX_SPEED_MPS, SPEED_BAND_MPS, fail_low=False)  # noqa: E731
    return {"pairs": len(shared), "abs_difference_mps": stats([abs(a[k] - b[k]) for k in shared]),
            "speed_ref_mps": stats([a[k] for k in shared]), "same_banded_verdict": sum(verdict(a[k]) == verdict(b[k]) for k in shared)}


def verdicts(folder, gated):
    """rule -> [(t, verdict)] changes; gated False reads the verdict before the scale gate."""
    out = {}
    for r in rows_of(folder, "findings.jsonl"):
        out.setdefault(r["rule"], []).append((r["t"], r["verdict"] if gated else (r["beforeScaleGate"] or r["verdict"])))
    return out


def at(timeline, t):
    value = None
    for s, v in timeline:
        if s > t + 1e-6:
            break
        value = v
    return value


def agreement(a, b, start, end, step=0.2):
    """Per rule, both gates: share of ticks where the two replays hold the same verdict, and the disagreeing spans."""
    ticks = np.arange(start, end, step)
    result = {}
    for gated in (False, True):
        va, vb = verdicts(a, gated), verdicts(b, gated)
        for rule in sorted(set(va) | set(vb)):
            pairs = [(round(float(t), 2), at(va.get(rule, []), t), at(vb.get(rule, []), t)) for t in ticks]
            differ = [p for p in pairs if p[1] != p[2]]
            spans = []
            for t, x, y in differ:
                if spans and abs(t - spans[-1][1] - step) < 1e-6 and spans[-1][2:] == [x, y]:
                    spans[-1][1] = t
                else:
                    spans.append([t, t, x, y])
            seen = lambda seq: sorted({v for _, v in seq})  # noqa: E731
            result[f"{rule}{'' if gated else '_before_gate'}"] = {
                "ticks": len(pairs), "agree_share": round(1 - len(differ) / len(pairs), 3),
                "verdicts_ref": seen(va.get(rule, [])), "verdicts_other": seen(vb.get(rule, [])), "disagree_spans": spans[:12]}
    return result


def detection_match(probe, frames):
    """SAM 3 'person' masks against today's cached person masks, frame by frame (IoU >= 0.5 is a match)."""
    detect, _ = sam3_detector(probe)
    reference = R.cached_detector()
    tp = fn = fp = 0
    ious = []
    for f in frames:
        mine = [d["mask"] for d in detect({"frame": f})]
        theirs = [d["mask"] for d in reference({"frame": f})]
        taken = set()
        for t in theirs:
            scores = [((m & t).sum() / max((m | t).sum(), 1), i) for i, m in enumerate(mine) if i not in taken]
            best = max(scores, default=(0.0, None))
            ious.append(float(best[0]))
            if best[0] >= 0.5:
                tp += 1
                taken.add(best[1])
            else:
                fn += 1
        fp += len(mine) - len(taken)
    return {"frames": len(frames), "today_masks": tp + fn, "matched": tp, "missed": fn, "extra_sam3_masks": fp,
            "best_iou": stats(ious)}


# ------------------------------------------------------------------ movers
def cameras():
    data = np.load(R.CAMERAS / "prediction.npz")
    keys = {int(i): k for k, i in enumerate(data["keyframe_source_indices"])}
    poses, keyframe = data["poses_c2w"], data["keyframe_c2w"]
    scale = json.loads((R.DEPTH / "fuse-metrics.json").read_text())["scale_native_per_mono_metre_median"]
    k = R.raster(np.zeros((480, 640, 3), np.uint8), R.calibration())[1]
    K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.]])

    def view(i):
        mono = np.load(R.DEPTH / "mono" / f"{i:05d}.npz")
        depth = np.where(mono["mask"], mono["depth"], 0).astype(np.float64) * scale  # DROID native units, as the map
        return depth, (keyframe[keys[i]] if i in keys else poses[i]).astype(np.float64)
    return K, view


def flying(depth):
    pad = np.pad(depth, 1, mode="edge")
    jump = np.max([np.abs(depth - pad[a:a + depth.shape[0], b:b + depth.shape[1]]) for a, b in [(0, 1), (2, 1), (1, 0), (1, 2)]], 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return jump / depth > EDGE_JUMP


def gone(depth_a, c2w_a, depth_b, c2w_b, K, tau=TAU):
    """Pixels of view a whose surface view b does not find where a static scene would put it: b sees farther along that
    ray (the surface left). Occlusion only makes b see nearer, so it cannot fake this; flying pixels are dropped."""
    h, w = depth_a.shape
    v, u = np.mgrid[0:h, 0:w]
    ok = (depth_a > 0) & ~flying(depth_a)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    world = (rays * depth_a[..., None]) @ c2w_a[:3, :3].T + c2w_a[:3, 3]
    local = (world - c2w_b[:3, 3]) @ c2w_b[:3, :3]
    z = local[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        ub = np.rint(local[..., 0] / z * K[0, 0] + K[0, 2])
        vb = np.rint(local[..., 1] / z * K[1, 1] + K[1, 2])
    inside = ok & (z > 0) & (ub >= 0) & (ub < w) & (vb >= 0) & (vb < h)
    seen = np.zeros((h, w))
    seen[inside] = depth_b[vb[inside].astype(int), ub[inside].astype(int)]
    return inside & (seen > 0) & (seen > z * (1 + tau))


def seeds(moved, share):
    """Connected parts of the opened residual at least `share` of the frame."""
    opened = cv2.morphologyEx(moved.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, st, _ = cv2.connectedComponentsWithStats(opened)
    keep = [n for n in range(1, count) if st[n, cv2.CC_STAT_AREA] >= share * moved.size]
    return np.isin(labels, keep) if keep else np.zeros_like(moved)


def residual_seeds(frames, gap=6):
    """frame -> (blob mask, seed mask, ms): the residual against the views `gap` frames before and after."""
    K, view = cameras()
    with_depth = {int(p.stem) for p in (R.DEPTH / "mono").glob("*.npz")}
    out = {}
    for f in frames:
        if not {f - gap, f + gap} <= with_depth:
            continue
        t = time.perf_counter()
        d, c = view(f)
        moved = np.zeros(d.shape, bool)
        for g in (f - gap, f + gap):
            dg, cg = view(g)
            moved |= gone(d, c, dg, cg, K)
        out[f] = (seeds(moved, BLOB), seeds(moved, SEED), (time.perf_counter() - t) * 1000)
    return out


def residual_scores(res):
    """Against today's person masks (dilated 15 px) and run 177's trusted motion; the caption band shows false seeds."""
    reference, grow = R.cached_detector(), np.ones((31, 31), np.uint8)
    rows = {"frames": len(res), "ms_per_frame_cpu": stats([ms for _, _, ms in res.values()])}
    for name, index in (("blobs", 0), ("seeds", 1)):
        person_frames = person_hit = in_person = total = in_caption = 0
        for f, value in res.items():
            m = value[index]
            people = [d["mask"] for d in reference({"frame": f})]
            person = cv2.dilate(np.any(people, 0).astype(np.uint8), grow) > 0 if people else np.zeros_like(m)
            total += m.sum()
            in_person += (m & person).sum()
            in_caption += m[CAPTION_ROW:].sum()
            if people:
                person_frames += 1
                person_hit += bool((m & person).sum() >= 0.05 * np.any(people, 0).sum())
        rows[name] = {"frames_with_any": int(sum(bool(v[index].any()) for v in res.values())),
                      "share_of_frame_mean": round(float(np.mean([v[index].mean() for v in res.values()])), 4),
                      "frames_with_person": person_frames, "person_frames_hit_5pct": person_hit,
                      "residual_pixels_on_person": round(float(in_person / max(total, 1)), 3),
                      "residual_pixels_in_caption_band": round(float(in_caption / max(total, 1)), 3)}
    motion = PHASE2 / "runs/me340-motion-masks-177"
    pairs = []
    for f, (blob, _, _) in res.items():
        path = motion / f"{f:05d}-residual.npz"
        if path.exists():
            with np.load(path) as data:
                trusted = data["trusted_moving"] if "trusted_moving" in data else None
            if trusted is not None and trusted.any():
                pairs.append((float((blob & trusted).sum() / trusted.sum()), float((blob & trusted).sum() / max(blob.sum(), 1))))
    rows["vs_run177_trusted_motion"] = {"frames": len(pairs), "recall": stats([p[0] for p in pairs]),
                                       "precision": stats([p[1] for p in pairs])}
    return rows


def mover_tracks(probe, res):
    """Each motion-session track of runs 178-180, at the detection frames inside its window: the share of its pixels a
    SAM 3 'person' mask, a mover-word mask or a residual seed covers (median over frames), and where it sits."""
    det = json.loads((probe / "detections.json").read_text())["frames"]
    unions = {}

    def union(f, words):
        if (f, words) not in unions:
            masks = [decode(r) for w in words for r in det[str(f)]["rle"][w]]
            unions[f, words] = np.any(masks, 0) if masks else np.zeros((480, 640), bool)
        return unions[f, words]
    out = []
    for run_id, (first, last) in TRACK_RUNS.items():
        folder = PHASE2 / f"runs/me340-sam31-tracks-{run_id}"
        kept = {o["object"]: o["kept"] for o in json.loads((folder / "tracks.json").read_text())["objects"]["motion"]}
        data = np.load(folder / "tracks.npz")
        by_object = {}
        for key in data.files:
            if key.startswith("motion/"):
                _, frame, obj = key.split("/")
                by_object.setdefault(int(obj), {})[first + int(frame)] = key
        for obj, frames in sorted(by_object.items()):
            person, words, resid, rows = [], [], [], []
            for f in sorted(set(frames) & {int(k) for k in det}):
                m = np.unpackbits(data[frames[f]])[:480 * 640].reshape(480, 640).astype(bool)
                if m.sum() < 500:
                    continue
                person.append((m & union(f, ("person",))).sum() / m.sum())
                words.append((m & union(f, MOVER_WORDS)).sum() / m.sum())
                if f in res:
                    resid.append((m & res[f][0]).sum() / m.sum())
                rows.append(np.nonzero(m)[0].mean())
            out.append({"track": f"{run_id}-motion-{obj}", "kept_today": kept.get(obj), "frames_sampled": len(person),
                        "person_cover_median": round(float(np.median(person)), 3) if person else None,
                        "mover_word_cover_median": round(float(np.median(words)), 3) if words else None,
                        "residual_cover_median": round(float(np.median(resid)), 3) if resid else None,
                        "mean_row": round(float(np.mean(rows)), 1) if rows else None})
    return out


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    probe = args.probe
    report = {"probe": str(probe), "zone_wkt": ZONE, "measured": "M: CPU scoring on this Mac; GPU numbers are in probe.json"}
    walk_end = 898 / 29.97
    runs, called_by = {}, {}
    for name, rate, make in (("ref10", 0, lambda c: (cached(c), None)), ("fast5", 5, lambda c: sam3_detector(probe, called=c)),
                             ("sam3_10", 0, lambda c: sam3_detector(probe, called=c)), ("cached5", 5, lambda c: (cached(c), None)),
                             ("trk3", 0, lambda c: tracker_detector(probe, called=c)),
                             ("fast5_dedupe", 5, lambda c: sam3_detector(probe, called=c, dedupe=True))):
        called = called_by[name] = []
        detector, only = make(called)
        out = args.output / name
        counts = replay(detector, rate, out, only if name == "trk3" else None)
        runs[name] = {"rate_hz": rate or "every depth frame", **{k: counts[k] for k in ("frames", "track_rows", "findings", "gaps", "wall_s")},
                      "gap_reasons": counts["gap_reasons"], "detector_frames": len(called),
                      "vs_today_centroids": path_vs_offline(out, called)}
        if name != "ref10":
            runs[name]["rules_vs_ref10"] = agreement(args.output / "ref10", out, 226 / 29.97, walk_end)
            runs[name]["body_speed_vs_ref10"] = speed_agreement(args.output / "ref10", out)
    report["replays"] = runs
    report["sam3_person_vs_today_masks_5hz"] = detection_match(probe, called_by["fast5"])
    res = residual_seeds(range(234, 892, 6))
    report["residual"] = residual_scores(res)
    report["mover_tracks"] = mover_tracks(probe, res)
    (args.output / "eval.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1)[:12000])


def self_check():
    """gone(): a static wall seen from two cameras flags nothing; a box that left between the views is flagged, and
    something new in front (occlusion) is not. agreement(): per-tick verdicts and spans."""
    K = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1.]])
    a, b = np.eye(4), np.eye(4)
    b[0, 3] = 0.1
    wall = np.full((480, 640), 4.0)
    assert not gone(wall, a, wall, b, K).any(), "a static wall is explained by the cameras"
    box = wall.copy()
    box[200:280, 300:340] = 2.0
    flagged = gone(box, a, wall, b, K)
    assert flagged[210:270, 310:330].all() and flagged.sum() < 80 * 40 + 200, "the box's pixels in view a are flagged"
    assert not gone(wall, a, box, b, K).any(), "a new occluder in view b is not motion of view a"
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        x, y = Path(folder, "x"), Path(folder, "y")
        for p, rows in ((x, [(0.0, "PASS"), (1.0, "NEEDS_REVIEW")]), (y, [(0.0, "PASS"), (1.4, "NEEDS_REVIEW")])):
            p.mkdir()
            (p / "findings.jsonl").write_text("".join(json.dumps({"t": t, "rule": "R3_speed", "verdict": v, "beforeScaleGate": v}) + "\n" for t, v in rows))
        got = agreement(x, y, 0.0, 2.0)["R3_speed"]
        assert got["ticks"] == 10 and got["agree_share"] == 0.8 and got["disagree_spans"] == [[1.0, 1.2, "NEEDS_REVIEW", "PASS"]], got
    print("e3 check passed: static reprojection flags only what left; verdict agreement per tick")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    a = parser.parse_args()
    self_check() if a.self_check else main(a)
