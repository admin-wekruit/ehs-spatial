"""Quality of one fast report next to the delivered report (FAST-BUILD-SPEC.md section 10): agreement, not accuracy.

References come from tests/fixtures/delivered-303/<site>.json by following its node graph from the names node (R22):
object map (R21) -> depth fuse with metric_scale (R17) -> camera (R07); mask root (R10); dynamic layer (R18) and its
motion analysis (R14); events (R24); splat package (R31). No run path is written here.

    python scripts/fast_report_eval.py RUN_DIR --site me340|samsclub-a2|walmart [--gpu]   # RUN_DIR: one mirrored report
    python scripts/fast_report_eval.py --e9 RUNS/m3-fu-e9-onegpu-003 --site me340          # the evaluator on E9's record
    python scripts/fast_report_eval.py --gaps OUT_DIR    # the review's gaps, 3 videos: fixed list vs per-video list, word match, person cut
    python scripts/fast_report_eval.py --self-check      # no GPU, no network

Layer fields read (latest version of each layer; shots are in their own frame, metres 'estimated'):
    cameras.shots[]      {index, frames: [a, b], keyframes: [source frame], c2w_m: [4x4]}
    objects.objects[]    {id, shot, word, centroid_m};  objects.vocabulary: [every word SAM 3 ran, waves merged]
    people.tracks[]      [{frame, xyz}] (xyz in the frame of the shot holding `frame`); people.rules: PeopleLoop findings rows
    outlines             VideoView analysis {frames: [{sourceFrame, objects: [{entityId, label, polygons, source}]}]}, inline
                         or blob 'analysis'; projected objects may carry polygons_before_cut (else before/after is not scored)
    events.windows[]     {t0, t1, caption, events}
    models               {models: [...], attempted: n}
    splat                {kind, holdout: {psnr, ssim, lpips}}
GPU rows (--gpu, one Modal A100 container, retries 0): SAM 3 (sam3_app's pin, bf16, 80 pairs per forward, floor 0.3, no
cap) on today's mask frames of the reference shot for the objects-2D recalls, and on sampled projected frames for outlines.
"""
import argparse
import gzip
import io
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

REPO = Path(__file__).resolve().parents[1]
# appended, not prepended: modal_apps/fast_report.py must not shadow the fast_report package for callers such as the bench
sys.path += [str(REPO / "modal_apps"), str(REPO / "modal_apps/m3_e5"), str(REPO / "modal_apps/m3_fu_e2b"), str(REPO / "scripts")]
import sam3_app as s3  # noqa: E402  SAM 3 pin, image, weights volume

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
FIXTURES = REPO / "tests/fixtures/delivered-303"
CLIPS = {"me340": "me340-165", "samsclub-a2": "samsclub-337", "walmart": "walmart-190"}  # the bench's MP4: data/clips/<clip>/source-full.mp4
FLOOR, PAIRS, IOU_FOUND, MATCH_M, MAX_FRAMES, OUTLINE_FRAMES = .3, 80, .5, 1., 56, 12
MIN_AREA_SRC = 1200  # reference masks for outlines, px at 1280x720 (E6b's 300 px at 640x480, x4)
E3_RULES = {"me340": PHASE2 / "runs/m3-exp-e3-people-eval-v3/ref10/findings.jsonl"}  # E3's replay of today's layer; no other site has one
E4_SAM3D = {"me340": (7, 30)}  # E4 s1cfg12: accepted / tried
S3_IMAGE = s3.image.add_local_python_source("sam3_app")
app = modal.App("panoptes-fb-eval")


# ---------- references ----------

def reference(site):
    fx = json.loads((FIXTURES / f"{site}.json").read_text())
    nodes = {n["id"]: n for n in fx["nodes"]}

    def first(stage, start=None, need=None):
        """BFS up the inputs from `start` (or over all nodes) to the first node of `stage` with output `need`."""
        todo = [start] if start else list(nodes)
        seen = set()
        while todo:
            n = nodes[todo.pop(0)]
            if n["id"] in seen:
                continue
            seen.add(n["id"])
            if n["stage"] == stage and (need is None or need in n["outputs"]) and n["id"] != start:
                return n
            if start:
                todo += n.get("inputs") or []
        raise KeyError(f"{site}: no {stage} node{f' above {start}' if start else ''}")
    names = first("R22")
    fuse = first("R17", names["id"], "metric_scale")
    camera, omap, masks = first("R07", fuse["id"]), first("R21", names["id"]), first("R10", names["id"])
    dynamic = first("R18")
    path = lambda node, key: PHASE2 / node["outputs"][key]  # noqa: E731
    shots = fx["decisions"]["shots"]
    a, b = shots["primary"]
    others = [range(*o) for o in shots["others"]]
    scale = json.loads(path(fuse, "metric_scale").read_text())
    frames = [f for f in range(a, b) if not any(f in o for o in others)]
    runs = np.split(frames, np.flatnonzero(np.diff(frames) > 1) + 1)
    return {"site": site, "publication": fx["publication"], "frames": frames, "segment": [int(f) for f in max(runs, key=len)],
            "others": others, "droid": path(camera, "prediction"), "mpn": scale["metres_per_native_unit"], "up": scale["up_native"],
            "names": path(names, "names"), "object_map": path(names, "object_map"), "mask_root": path(masks, "object_a"),
            "dynamic": path(dynamic, "scene"), "analysis": path(first("R14", dynamic["id"]), "analysis"),
            "events": path(first("R24"), "events"), "splat": path(first("R31"), "splats_json"),
            "nodes": {"names": names["id"], "object_map": omap["id"], "metric": fuse["id"], "camera": camera["id"], "masks": masks["id"], "dynamic": dynamic["id"]}}


def holdout_frames(site):
    """The delivered splat's held-out frames: the bench passes them as options['eval_holdout']."""
    return json.loads(reference(site)["splat"].read_text())["heldOutFrames"]


# ---------- the run ----------

def load_layers(run_dir, report=None):
    """{layer: data} from the latest version of each layer's patch under run_dir (of `report`, when several are mirrored
    there); blobs are found by their sha256 name."""
    run_dir, latest = Path(run_dir), {}
    for path in run_dir.rglob("patches/*.json"):
        p = json.loads(path.read_text())
        if p.get("schema") == "panoptes-fast-patch-v1" and report in (None, p["report"]) and (p["layer"] not in latest or (p["version"], p["seq"]) > (latest[p["layer"]]["version"], latest[p["layer"]]["seq"])):
            latest[p["layer"]] = p
    reports = {p["report"] for p in latest.values()}
    assert len(reports) == 1, f"{run_dir}: expected one report, found {sorted(reports)}"
    out = {k: dict(p["data"] or {}) for k, p in latest.items()}
    if "analysis" in (latest.get("outlines", {}).get("blobs") or {}):
        blob = next(run_dir.rglob(latest["outlines"]["blobs"]["analysis"]["sha256"])).read_bytes()
        out["outlines"] = json.loads(gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob)
    out["_report"] = reports.pop()
    return out


def layers_from_e9(r, vocabulary):
    """One E9 call record (runs[i] of m3-fu-e9-onegpu-*/2gpu.json) as the fast layers this evaluator reads."""
    import video_events
    shots = [{"index": i, "frames": s["shot"], "keyframes": s["keyframes"], "c2w_m": s["c2w_m"]} for i, s in enumerate(r["shots"])]
    objects = [{"id": f"s{i}-{k}", "shot": i, "word": o["word"], "centroid_m": o["centroid_m"]} for i, s in enumerate(r["shots"]) for k, o in enumerate(s["objects"])]
    tracks = [t for s in r["shots"] for t in s["people"]["tracks"]]
    windows = [{"t0": e["t0"], "t1": e["t1"], **(video_events.parse(e["text"]) or {"caption": None, "events": []})} for e in r["events"]]
    return {"cameras": {"shots": shots}, "objects": {"objects": objects, "vocabulary": vocabulary}, "people": {"tracks": tracks},
            "events": {"windows": windows}, "_report": f"e9-{r.get('mode')}-{r.get('index')}"}


# ---------- CPU rows ----------

def stats(values):
    v = np.asarray(values, float)
    return {"n": int(v.size), **({"median": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3),
                                  "max": round(float(v.max()), 3)} if v.size else {})}


def matched_shot(cameras, ref):
    """Our shot with the most keyframes on the reference shot's frames (frame-range conventions do not matter)."""
    frames = set(ref["frames"])
    return max(cameras["shots"], key=lambda s: sum(k in frames for k in s["keyframes"]))


def camera_rows(layers, ref):
    import m3_exp_geometry as geo  # E1's Sim3 (orientations fix the roll of a straight walk)
    shot, frames = matched_shot(layers["cameras"], ref), set(ref["frames"])
    keys = [k for k in shot["keyframes"] if k in frames]
    ours = np.array([c for k, c in zip(shot["keyframes"], shot["c2w_m"]) if k in frames], np.float64)
    target = np.load(ref["droid"])["poses_c2w"][keys].astype(np.float64)
    target[:, :3, 3] *= ref["mpn"]
    s, R, t = geo.align_sim3(ours, target)
    err = np.linalg.norm((s * (R @ ours[:, :3, 3].T)).T + t - target[:, :3, 3], axis=1)
    rot = [geo.angle_deg(tg[:3, :3].T @ R @ o[:3, :3]) for o, tg in zip(ours, target)]
    path_m = float(np.linalg.norm(np.diff(target[:, :3, 3], axis=0), axis=1).sum())
    ate = float(np.sqrt((err ** 2).mean()))
    blocking = ref["site"] == "me340"
    cam = {"shot": shot["index"], "keyframes": len(keys), "ate_m": round(ate, 4), "path_m": round(path_m, 3),
           "ate_share_of_path": round(ate / path_m, 4) if path_m else None,
           "rotation_error_deg": {"median": round(float(np.median(rot)), 3), "max": round(float(np.max(rot)), 3)},
           "criterion": "ATE <= 0.057 m (E1)" if blocking else "ATE <= 1.5% of the path (report only)",
           "pass": ate <= .057 if blocking else None, "within_1.5pct": bool(path_m and ate / path_m <= .015)}
    scale = {"ours_over_reference": round(1 / s, 4), "criterion": "0.9-1.1", "pass": .9 <= 1 / s <= 1.1,
             "note": "both assume a 1.6 m camera height; ours from the fast floor plane, the reference from its own floor rule"}
    return cam, scale, (s, R, t, shot)


def reference_objects(ref, shot_frames):
    """E1/E9's filter: named-map objects seen >= 3 times on the reference frames of our shot and never in another shot."""
    omap = json.loads(ref["object_map"].read_text())
    frames = set(ref["frames"]) & shot_frames
    keep = [e for e in omap["entities"] if e["entityId"].startswith("object-") and sum(f in frames for f in e["sourceFrames"]) >= 3
            and not any(f in o for o in ref["others"] for f in e["sourceFrames"])]
    return keep, np.array([e["centroidNative"] for e in keep]).reshape(-1, 3) * ref["mpn"]


def objects3d_row(layers, ref, align):
    s, R, t, shot = align
    frames = set(range(min(shot["keyframes"]), max(shot["keyframes"]) + 1))
    ents, ref_xyz = reference_objects(ref, frames)
    objs = [o for o in layers["objects"]["objects"] if o["shot"] == shot["index"]]
    row = {"ours": len(objs), "reference": len(ents), "criterion": "report; E9 0.53, E1 with SAM 2 0.64 (recall at 0.5 m, ME340)", "pass": None}
    if objs and len(ents):
        xyz = (s * (R @ np.array([o["centroid_m"] for o in objs], np.float64).T)).T + t
        d = np.linalg.norm(ref_xyz[:, None] - xyz[None], axis=2)
        for th in (.3, .5):
            row[f"reference_recall_{th}m"] = round(float((d.min(1) < th).mean()), 3)
            row[f"ours_near_reference_{th}m"] = round(float((d.min(0) < th).mean()), 3)
    return row


def reference_people(ref):
    """source frame -> [(entity, centroid metres)]: persons of the delivered dynamic layer (label from its motion analysis)."""
    people = {o["entityId"] for f in json.loads(ref["analysis"].read_text())["frames"] for o in f["objects"] if o["label"] == "person"}
    out = {}
    for f in json.loads(ref["dynamic"].read_text())["frames"]:
        for o in f["objects"]:
            if o.get("centroid") and o["entityId"] in people:
                out.setdefault(f["sourceFrame"], []).append((o["entityId"], np.asarray(o["centroid"], float) * ref["mpn"]))
    return out


def at_frame(samples, f, max_gap):
    """Linear interpolation of [(frame, xyz)] (sorted) at frame f, or None when f is not between samples <= max_gap apart."""
    for (a, x), (b, y) in zip(samples, samples[1:]):
        if a <= f <= b and b - a <= max_gap:
            return x + (y - x) * (f - a) / max(b - a, 1)
    return next((x for a, x in samples if a == f), None)


def rule_agreement(ref_rows, our_rows, start, end, step=.2):
    """Per rule, before and after the scale gate: share of 0.2 s ticks with the same verdict, and PASS<->FAIL flips
    (e3_people_eval.agreement's rule, on in-memory rows)."""
    def timeline(rows, gated):
        out = {}
        for r in sorted(rows, key=lambda r: r["t"]):
            out.setdefault(r["rule"], []).append((r["t"], r["verdict"] if gated else (r.get("beforeScaleGate") or r["verdict"])))
        return out

    def at(tl, t):
        v = None
        for s, x in tl:
            if s > t + 1e-6:
                break
            v = x
        return v
    ticks, result = np.arange(start, end, step), {}
    for gated in (False, True):
        a, b = timeline(ref_rows, gated), timeline(our_rows, gated)
        for rule in sorted(set(a) | set(b)):
            pairs = [(at(a.get(rule, []), t), at(b.get(rule, []), t)) for t in ticks]
            result[rule + ("" if gated else "_before_gate")] = {
                "ticks": len(pairs), "agree_share": round(sum(x == y for x, y in pairs) / max(len(pairs), 1), 3),
                "pass_fail_flips": sum({x, y} == {"PASS", "FAIL"} for x, y in pairs)}
    return result


def people_row(layers, ref, align, fps):
    s, R, t, shot = align
    lo, hi = min(shot["keyframes"]), max(shot["keyframes"])
    up = np.asarray(ref["up"], float)
    up /= np.linalg.norm(up)
    refs = reference_people(ref)
    by_entity = {}
    for f in sorted(refs):
        for e, x in refs[f]:
            by_entity.setdefault(e, []).append((f, x))
    on_floor = lambda v: float(np.linalg.norm(v - (v @ up) * up))  # noqa: E731  a foot point vs a body centroid differ in height only
    floor, extra, n, frames = [], 0, 0, set(ref["frames"])
    for track in layers["people"]["tracks"]:
        for p in track:
            if not lo <= p["frame"] <= hi or p["frame"] not in frames:
                continue
            n += 1
            x = s * (R @ np.asarray(p["xyz"], float)) + t
            near = [on_floor(y - x) for y in (at_frame(v, p["frame"], round(.45 * fps)) for v in by_entity.values()) if y is not None]
            if not near or min(near) > MATCH_M:
                extra += 1
                continue
            floor.append(min(near))
    path = stats(floor)
    row = {"our_points_in_shot": n, "matched": len(floor), "extra": extra, "path_difference_floor_m": path,
           "criterion": "median <= 0.3 m; no PASS<->FAIL flip", "note": "our xyz vs the delivered layer's visible-body centroid (interpolated "
           "to our frame, gaps <= 0.45 s), both on the floor plane; a point farther than 1 m from every person there is 'extra'"}
    rules_ok = None
    if ref["site"] in E3_RULES and layers["people"].get("rules"):
        ref_rows = [json.loads(line) for line in E3_RULES[ref["site"]].read_text().splitlines()]
        row["rules_vs_e3_ref10"] = rule_agreement(ref_rows, layers["people"]["rules"], lo / fps, hi / fps)
        rules_ok = not any(v["pass_fail_flips"] for v in row["rules_vs_e3_ref10"].values())
    else:
        row["rules_vs_e3_ref10"] = "not scored: " + ("no rules rows in the layer" if ref["site"] in E3_RULES else "only ME340 has a reference rule stream (E3 ref10, its m0 test zone)")
    row["pass"] = None if not floor else bool(path["median"] <= .3 and rules_ok is not False)
    return row


def events_row(layers, ref):
    import vllm_events  # E5: temporal matching of events, field agreement
    agreement = vllm_events.agreement(json.loads(ref["events"].read_text())["windows"], layers["events"]["windows"])
    matched = agreement[next(k for k in agreement if k.startswith("matched"))]
    return {**agreement, "criterion": "actor and PPE the same on every matched event; caption Jaccard report only",
            "pass": bool(matched) and agreement["actor_same"] == matched and agreement["ppe_same"] == matched}


def sam3d_row(layers, ref):
    m = layers["models"]
    accepted, tried = len(m.get("models") or []), m.get("attempted")
    row = {"accepted": accepted, "attempted": tried, "pass": None}
    if ref["site"] in E4_SAM3D:
        e4 = E4_SAM3D[ref["site"]]
        row["criterion"] = f"within +-3 of E4 s1cfg12 {e4[0]}/{e4[1]} on {e4[1]} tries (our objects differ from E4's)"
        row["pass"] = tried == e4[1] and abs(accepted - e4[0]) <= 3 if tried else None
    else:
        row["criterion"] = "report only (E4 measured ME340 only)"
    return row


def splat_row(layers, ref):
    got = layers["splat"].get("holdout") or {}
    held = json.loads(ref["splat"].read_text())["metrics"]["heldOut"]
    preview = layers["splat"].get("kind") == "preview"
    return {"kind": layers["splat"].get("kind"), **{k: got.get(k) for k in ("psnr", "ssim", "lpips")}, "reference": held,
            "criterion": "preview >= 27.0 dB on the delivered splat's held-out frames; full report only",
            "pass": (got["psnr"] >= 27. if preview else None) if got.get("psnr") is not None else None}


# ---------- names ----------

def singular(w):
    for tail, new in (("ves", "f"), ("ies", "y"), ("xes", "x"), ("ches", "ch"), ("shes", "sh"), ("ss", "ss"), ("s", "")):
        if len(w) >= len(tail) + 2 and w.endswith(tail):
            return w[:-len(tail)] + new
    return w


def tokens(text):
    return [singular(w) for w in text.lower().replace("-", " ").split()]


def same_name(word, name):
    """SAM 3 word vs today's free-text name: one's tokens run inside the other's (whole words), or the heads agree."""
    a, b = tokens(word), tokens(name or "")
    if not a or not b:
        return False
    inside = lambda x, y: any(y[i:i + len(x)] == x for i in range(len(y) - len(x) + 1))  # noqa: E731
    return inside(a, b) or inside(b, a) or a[-1] == b[-1]


# ---------- GPU ----------

@app.function(image=S3_IMAGE, volumes={"/cache": s3.volume}, secrets=[modal.Secret.from_name("huggingface")], timeout=1800, cpu=4,
              memory=32768, gpu="A100-80GB", retries=0, max_containers=1, scaledown_window=2)
def sam3_eval(jobs):
    """jobs: {'recall': {frames (png), labels (npz), obs [(frame index, label)], sets {name: words}},
              'outlines': {frames (png 1280x720), variants (npz: name -> (n,H,W) uint16), cut [names], words}}"""
    import types
    import torch
    import torch.nn.functional as F
    import transformers
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    enter = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False  # exact pixel counts in the IoU matmuls
    t = time.perf_counter()
    processor = Sam3Processor.from_pretrained(s3.MODEL_ID, revision=s3.REVISION)
    model = Sam3Model.from_pretrained(s3.MODEL_ID, revision=s3.REVISION, torch_dtype=torch.bfloat16).to("cuda").eval()
    report = {"gpu": torch.cuda.get_device_name(), "transformers": transformers.__version__, "load_s": round(time.perf_counter() - t, 2)}
    ip = processor.image_processor
    size = (ip.size["height"], ip.size["width"])
    mean = torch.tensor(ip.image_mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, device="cuda").view(1, 3, 1, 1)

    def pixels(batch):
        x = torch.from_numpy(np.stack([np.asarray(i) for i in batch])).cuda().permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=size, mode="bilinear", antialias=True, align_corners=False)
        return ((x - mean) / std).to(torch.bfloat16)

    def masks(images, words):
        """(frame index, word index, scores, masks) for every pair: vision once per frame, text once per list, PAIRS per forward."""
        text = processor(text=words, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            enc = model.get_text_features(input_ids=text["input_ids"], attention_mask=text["attention_mask"])
            wrapped = hasattr(enc, "pooler_output")
            feats, amask = (enc.pooler_output if wrapped else enc), text["attention_mask"]
            per, n_chunk = min(len(words), PAIRS), max(1, PAIRS // len(words))
            h, w = images[0].height, images[0].width
            for s in range(0, len(images), n_chunk):
                chunk = list(range(s, min(s + n_chunk, len(images))))
                vision = model.get_vision_features(pixel_values=pixels([images[i] for i in chunk]))
                for p0 in range(0, len(words), per):
                    sub = list(range(p0, min(p0 + per, len(words))))
                    vis = type(vision)(**{k: tuple(x.repeat_interleave(len(sub), 0) for x in v) for k, v in vision.items() if k.startswith("fpn_")})
                    tx = feats[sub].repeat(len(chunk), 1, 1)
                    out = model(vision_embeds=vis, attention_mask=amask[sub].repeat(len(chunk), 1),
                                text_embeds=types.SimpleNamespace(pooler_output=tx) if wrapped else tx)
                    results = processor.post_process_instance_segmentation(out, threshold=FLOOR, mask_threshold=.5, target_sizes=[(h, w)] * (len(chunk) * len(sub)))
                    for j, r in enumerate(results):
                        if len(r["scores"]):  # an empty result keeps the model's mask size: skip it
                            yield chunk[j // len(sub)], sub[j % len(sub)], r["scores"].float(), r["masks"].bool()

    def iou(a, b):
        """a (m,HW), b (k,HW) float -> (m,k) IoU."""
        inter = a @ b.T
        return inter / (a.sum(1)[:, None] + b.sum(1)[None] - inter).clamp(min=1)

    if "recall" in jobs:
        job = jobs["recall"]
        images = [Image.open(io.BytesIO(p)).convert("RGB") for p in job["frames"]]
        labels = torch.from_numpy(np.load(io.BytesIO(job["labels"]))["labels"].astype(np.int32)).cuda()
        rows_of = {}
        for n, (i, _) in enumerate(job["obs"]):
            rows_of.setdefault(i, []).append(n)
        observed = {i: torch.stack([(labels[i] == job["obs"][n][1]).flatten() for n in rows]).float() for i, rows in rows_of.items()}
        report["recall"] = {}
        for name, words in job["sets"].items():
            list(masks(images[:2], words))  # warm-up at this prompt count
            best, kept = torch.zeros(len(job["obs"]), len(words)), 0
            torch.cuda.synchronize()
            t = time.perf_counter()
            for i, w, _, m in masks(images, words):
                kept += len(m)
                if i in observed:
                    rows = rows_of[i]
                    best[rows, w] = torch.maximum(best[rows, w], iou(observed[i], m.flatten(1).float()).max(1).values.cpu())
            torch.cuda.synchronize()
            report["recall"][name] = {"words": words, "s_per_frame_with_scoring": round((time.perf_counter() - t) / len(images), 4),
                                      "masks_per_frame": round(kept / len(images), 1), "best_iou": best.numpy().round(4).tolist()}
    if "outlines" in jobs:
        job = jobs["outlines"]
        images = [Image.open(io.BytesIO(p)).convert("RGB") for p in job["frames"]]
        variants = dict(np.load(io.BytesIO(job["variants"])))
        words = ["person"] + job["words"]
        person, refs = {}, {}
        for i, w, _, m in masks(images, words):
            if w == 0:
                person[i] = person.get(i, torch.zeros_like(m[0])) | m.any(0)
            else:
                refs.setdefault(i, []).append(m[m.flatten(1).sum(1) >= MIN_AREA_SRC])
        rows = {}
        for name, maps in variants.items():
            for cut in ([False, True] if name in job["cut"] else [False]):
                key = name + ("+cut" if cut else "")
                rows[key] = {"rows": [], "projected_px_on_person": 0, "projected_px": 0}
                for i in range(len(images)):
                    lab = torch.from_numpy(maps[i].astype(np.int64)).cuda()
                    p = person.get(i, torch.zeros(lab.shape, dtype=torch.bool, device="cuda"))
                    if cut:
                        lab = torch.where(p, 0, lab)
                    rows[key]["projected_px"] += int((lab > 0).sum())
                    rows[key]["projected_px_on_person"] += int(((lab > 0) & p).sum())
                    ref = torch.cat(refs[i]) if refs.get(i) else None
                    ids = torch.unique(lab)
                    ids = ids[ids > 0]
                    if ref is None or not len(ref) or not len(ids):
                        continue
                    flat = ref.flatten(1).float()
                    onehot = (lab.flatten()[None] == ids[:, None]).float()
                    best = torch.cat([iou(flat[k:k + 64], onehot).max(1).values for k in range(0, len(flat), 64)])
                    on_person = (flat @ p.flatten().float()) / flat.sum(1)
                    rows[key]["rows"] += [[round(float(b), 4), int(a), bool(o < .5)] for b, a, o in zip(best, flat.sum(1), on_person)]
        report["outlines"] = {"rows": rows, "reference_masks": sum(len(torch.cat(v)) for v in refs.values()),
                              "person_px_share": round(float(np.mean([float(v.float().mean()) for v in person.values()])) if person else 0., 4)}
    report["function_wall_s"], report["enter_unix"] = round(time.time() - enter, 2), enter
    return json.dumps(report)


def mask_grid(ref):
    """Today's mask frames in the reference shot's longest segment, its first (cut) frame left out."""
    segment = set(ref["segment"][1:])
    return sorted(f for f in (int(p.name[6:]) for p in ref["mask_root"].glob("frame-*")) if f in segment)


def spread(grid, n):
    """n frames evenly over the grid (by index)."""
    n = min(n, len(grid))
    return [grid[int((i + .5) * len(grid) / n)] for i in range(n)]


def recall_inputs(ref, frames=None):
    """Today's mask frames of the reference shot (evenly thinned to <= MAX_FRAMES), their label maps (round 1's rule: instance
    i -> label i+1, painted in glob order), and the named observations on them (clear or partial names, object masks)."""
    import cv2
    root = ref["mask_root"]
    frames = frames or spread(mask_grid(ref), MAX_FRAMES)
    labels = np.zeros((len(frames), 480, 640), np.uint16)
    for n, f in enumerate(frames):
        for path in (root / f"frame-{f:05d}").glob("instance-*-mask.png"):
            labels[n][cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0] = int(path.stem.split("-")[1]) + 1
    names = json.loads(ref["names"].read_text())
    entities = {e["entityId"]: e for e in json.loads(ref["object_map"].read_text())["entities"]}
    obs = []
    for entity, spec in names.items():
        if spec["status"] in ("clear", "partial") and entity in entities:
            for kind, f, i in (o.split(":") for o in entities[entity]["observations"]):
                if kind == "object" and int(f) in frames:
                    obs.append((entity, frames.index(int(f)), int(i) + 1, spec["category"]))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, labels=labels)
    pngs = [(root / f"frame-{f:05d}" / f"frame-{f}.png").read_bytes() for f in frames]
    return frames, pngs, buffer.getvalue(), obs


def recall_row(result, obs):
    """Position recall (any word's mask, IoU >= 0.5: E2's rule) and word-match recall (the mask's word matches today's name)."""
    best = np.array(result["best_iou"]).reshape(len(obs), -1)
    words = result["words"]
    match = np.array([[same_name(w, o[3]) for w in words] for o in obs]).reshape(len(obs), -1)
    pool = {o[0] for o in obs}
    pos = {o[0] for o, b in zip(obs, best) if b.max(initial=0) >= IOU_FOUND}
    word = {o[0] for o, b, m in zip(obs, best, match) if (b * m).max(initial=0) >= IOU_FOUND}
    return {"words": len(words), "pool": len(pool), "position_found": len(pos), "position_recall": round(len(pos) / max(len(pool), 1), 3),
            "word_match_found": len(word), "word_match_recall": round(len(word) / max(len(pool), 1), 3),
            "masks_per_frame": result["masks_per_frame"], "s_per_frame_with_scoring": result["s_per_frame_with_scoring"]}


def fixed_list():
    """The control: the 40 words E9's resident core ran (E2: 20 EHS + 20 generic) plus the 8 EHS core words, 42 unique.
    The 20 generic words were picked after seeing ME340's misses in round 1, so on ME340 this control is favoured."""
    import m3_fu_e9_onegpu as e9
    import vocab_probe as vp
    return vp.union(e9.EHS, e9.GENERIC, vp.CORE)


def rasterize(frames_json, want, size=(720, 1280), key="polygons"):
    """VideoView analysis frames -> (n,H,W) uint16 maps of the objects' polygons (object k of a frame -> k+1)."""
    import cv2
    by = {f.get("sourceFrame", f.get("frame")): f for f in frames_json}
    out = np.zeros((len(want), *size), np.uint16)
    for n, f in enumerate(want):
        for k, o in enumerate(by[f]["objects"]):
            polys = [np.round(np.asarray(p, float)).astype(np.int32) for p in (o.get(key) or []) if len(p) >= 3]
            if polys:
                cv2.fillPoly(out[n], polys, k + 1)
    return out


def decode(mp4, frames):
    """Source frames of an MP4 as PNG bytes (cv2, sequential read)."""
    import cv2
    cap, out, i, want = cv2.VideoCapture(str(mp4)), {}, 0, set(frames)
    while len(out) < len(want):
        ok, image = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = cv2.imencode(".png", image)[1].tobytes()
        i += 1
    cap.release()
    return [out[f] for f in frames]


def outline_job(analysis, frames_in_shot, words, clip):
    """Sampled projected frames of an outlines analysis -> the GPU job: as delivered, and before the person cut when the
    layer keeps it; E6b-style layers without a cut get the cut applied on the GPU ('+cut')."""
    projected = sorted(f.get("sourceFrame", f.get("frame")) for f in analysis["frames"]
                       if f.get("source") == "projected" or any(o.get("source") == "projected" for o in f["objects"]))
    projected = [f for f in projected if f in frames_in_shot]
    want = spread(projected, OUTLINE_FRAMES)
    variants = {"delivered": rasterize(analysis["frames"], want)}
    has_before = any("polygons_before_cut" in o for f in analysis["frames"] for o in f["objects"])
    if has_before:
        variants["before_cut"] = rasterize(analysis["frames"], want, key="polygons_before_cut")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **variants)
    return want, {"frames": decode(PHASE2 / "data/clips" / clip / "source-full.mp4", want), "variants": buffer.getvalue(),
                  "cut": [] if has_before else ["delivered"], "words": words}


def outline_rows(result):
    out = {}
    for name, r in result["rows"].items():
        rows = np.array(r["rows"], float).reshape(-1, 3)
        for part, keep in (("static", rows[:, 2] > 0), ("all", np.ones(len(rows), bool))):
            iou, area = rows[keep, 0], rows[keep, 1]
            out.setdefault(name, {})[part] = {"n": int(keep.sum()), "mean": round(float(iou.mean()), 4) if keep.any() else None,
                                              "area_weighted": round(float((iou * area).sum() / area.sum()), 4) if keep.any() else None}
        out[name]["projected_px_on_person_share"] = round(r["projected_px_on_person"] / max(r["projected_px"], 1), 4)
    return out


# ---------- the whole table ----------

def evaluate(layers, site, gpu=False, fps=None):
    ref = reference(site)
    fps = fps or json.loads((PHASE2 / "data/clips" / CLIPS[site] / "source-full.json").read_text())["fps"]
    rows, skipped = {}, {}
    rows["cameras"], rows["scale"], align = camera_rows(layers, ref)
    for name, fn, need in (("objects_3d", lambda: objects3d_row(layers, ref, align), "objects"),
                           ("people", lambda: people_row(layers, ref, align, fps), "people"),
                           ("events", lambda: events_row(layers, ref), "events"),
                           ("sam3d", lambda: sam3d_row(layers, ref), "models"),
                           ("splat", lambda: splat_row(layers, ref), "splat")):
        if need in layers:
            rows[name] = fn()
        else:
            skipped[name] = f"no {need} layer"
    if gpu and "objects" in layers:
        frames, pngs, labels, obs = recall_inputs(ref)
        vocabulary = layers["objects"]["vocabulary"]
        jobs = {"recall": {"frames": pngs, "labels": labels, "obs": [(o[1], o[2]) for o in obs], "sets": {"run": vocabulary, "fixed42": fixed_list()}}}
        shot = align[3]
        if "outlines" in layers:
            want, jobs["outlines"] = outline_job(layers["outlines"], set(range(min(shot["keyframes"]), max(shot["keyframes"]) + 1)), vocabulary, CLIPS[site])
        with modal.enable_output(), app.run():
            result = json.loads(sam3_eval.remote(jobs))
        run, fixed = recall_row(result["recall"]["run"], obs), recall_row(result["recall"]["fixed42"], obs)
        rows["objects_2d"] = {**run, "frames": len(frames), "criterion": "position recall >= 80%; word-match recall report only",
                              "pass": run["position_recall"] >= .8, "gpu": result["gpu"]}
        rows["vocab_control"] = {"per_video": run, "fixed42": fixed, "criterion": "the per-video list stays the default only if it wins on word-match recall",
                                 "pass": run["word_match_recall"] > fixed["word_match_recall"]}
        if "outlines" in result:
            rows["outlines"] = {"frames": want, **outline_rows(result["outlines"]), "criterion": "report only, no accuracy claim", "pass": None}
        else:
            skipped["outlines"] = "no outlines layer"
    elif not gpu:
        skipped.update({k: "needs --gpu" for k in ("objects_2d", "vocab_control", "outlines")})
    return {"schema": "panoptes-fast-eval-v1", "site": site, "report": layers.get("_report"), "reference_publication": ref["publication"],
            "reference_nodes": ref["nodes"], "rows": rows, "not_scored": skipped,
            "verdict": {k: v.get("pass") for k, v in rows.items() if isinstance(v, dict)}}


# ---------- the review's gaps, measured before the pipeline exists ----------

def gaps(out):
    """Three videos: per-video Qwen3-VL list (v1 prompt, 5 frames, first 50 + core) vs the fixed 42-word list, on the same
    frames, position and word-match recall; E2b's ME340 list again (reproduction); the person cut on E6b's 'pair' outlines."""
    import vocab_probe as vp
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    refs = {site: reference(site) for site in CLIPS}
    # ME340 on E2b's own 56 frames and 5 VLM frames (a reproduction); the others by the same rule over their shot
    inputs = {site: recall_inputs(r, vp.KEYSETS["2.5fps"] if site == "me340" else None) for site, r in refs.items()}
    vlm_frames = {site: vp.VLM_FRAMES[5] if site == "me340" else spread(mask_grid(refs[site]), 5) for site in CLIPS}
    requests = {f"{site}-qwen-v1-5": (vp.prompt(5, "v1"), [(refs[site]["mask_root"] / f"frame-{f:05d}" / f"frame-{f}.png").read_bytes() for f in vlm_frames[site]])
                for site in CLIPS}
    started = time.time()
    with modal.enable_output(), vp.app.run():
        qwen = vp.qwen_vocab.remote(requests)
    qwen["client_wall_s"], qwen["vlm_frames"] = round(time.time() - started, 1), vlm_frames
    (out / "vlm.json").write_text(json.dumps(qwen, ensure_ascii=False, indent=1))
    e2b = json.loads((PHASE2 / "runs/m3-fu-e2b-vocab-1/vlm.json").read_text())["vocab"]["qwen-v1-5"]
    fixed = fixed_list()
    jobs, sets = {}, {}
    for site in CLIPS:
        lists = vp.vocabulary(qwen["answers"][f"{site}-qwen-v1-5"][-1]["text"])  # the second (warm) answer, as E2b read it
        sets[site] = {"qwen-v1-5+core": vp.union(vp.union(*lists)[:vp.MAX_TYPES], vp.CORE), "fixed42": fixed}
        if site == "me340":
            sets[site]["e2b-qwen-v1-5+core"] = vp.union(vp.union(e2b["ehs_relevant"], e2b["other"])[:vp.MAX_TYPES], vp.CORE)
    e6b = json.loads(gzip.decompress((PHASE2 / "runs/m3-fu-e6b-outlines-002/da3-outlines-step5.json.gz").read_bytes()))["pair"]
    walk = set(range(226, 899))
    results = {}
    started = time.time()
    with modal.enable_output(), app.run():
        calls = {}
        for site, (frames, pngs, labels, obs) in inputs.items():
            job = {"recall": {"frames": pngs, "labels": labels, "obs": [(o[1], o[2]) for o in obs], "sets": sets[site]}}
            if site == "me340":
                want, job["outlines"] = outline_job(e6b, walk, sets[site]["e2b-qwen-v1-5+core"], CLIPS[site])
            calls[site] = sam3_eval.spawn(job)
        for site, call in calls.items():
            results[site] = json.loads(call.get())
    (out / "sam3.json").write_text(json.dumps(results))
    summary = {"frames": {site: v[0] for site, v in inputs.items()}, "client_wall_s": round(time.time() - started, 1), "recall": {}, "gpu": {}}
    for site, r in results.items():
        obs = inputs[site][3]
        summary["recall"][site] = {name: recall_row(v, obs) for name, v in r["recall"].items()}
        summary["gpu"][site] = {"gpu": r["gpu"], "load_s": r["load_s"], "function_wall_s": r["function_wall_s"]}
        if "outlines" in r:
            summary["outlines_me340_e6b_pair_2fps"] = {"frames": want, **outline_rows(r["outlines"]), "reference_masks": r["outlines"]["reference_masks"]}
        summary["recall"][site]["per_video_wins_word_match"] = summary["recall"][site]["qwen-v1-5+core"]["word_match_recall"] > summary["recall"][site]["fixed42"]["word_match_recall"]
    gpu_s = sum(r["function_wall_s"] for r in results.values()) + qwen["container_s"]
    summary["usd_estimate"] = round(gpu_s * (.000694 + 4 * .0000131 + 32 * .00000222), 3)
    summary["words"] = {site: {k: v for k, v in s.items() if k != "fixed42"} for site, s in sets.items()} | {"fixed42": fixed}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: summary[k] for k in ("recall", "outlines_me340_e6b_pair_2fps", "usd_estimate") if k in summary}, indent=1))


def self_check():
    assert same_name("fire extinguisher", "red fire extinguisher") and same_name("conduit", "electrical conduit")
    assert same_name("shelves", "metal shelf") and same_name("boxes", "cardboard box") and same_name("glass", "glass door")
    assert same_name("tool rack", "tool holder rack"), "heads agree"
    assert not same_name("cart", "cartridge box") and not same_name("glass", "gas cylinder") and not same_name("x", "")
    obs = [("a", 0, 1, "drill press"), ("a", 1, 1, "drill press"), ("b", 0, 2, "cable"), ("c", 1, 3, "metal plate")]
    r = recall_row({"words": ["machine", "cable", "drill press"], "masks_per_frame": 1, "s_per_frame_with_scoring": 1,
                    "best_iou": [[.9, 0, .2], [0, 0, .6], [0, .4, 0], [.7, 0, 0]]}, obs)
    assert (r["pool"], r["position_found"], r["word_match_found"]) == (3, 2, 1), r  # a by 'drill press' on frame 1; c only by 'machine'
    assert rule_agreement([{"t": 0, "rule": "R1", "verdict": "NEEDS_REVIEW", "beforeScaleGate": "FAIL"}, {"t": .5, "rule": "R1", "verdict": "PASS"}],
                          [{"t": 0, "rule": "R1", "verdict": "NEEDS_REVIEW", "beforeScaleGate": "PASS"}], 0, 1) == {
        "R1_before_gate": {"ticks": 5, "agree_share": .4, "pass_fail_flips": 3},
        "R1": {"ticks": 5, "agree_share": .6, "pass_fail_flips": 0}}
    samples = [(0, np.zeros(3)), (6, np.array([6., 0, 0])), (30, np.zeros(3))]
    assert np.allclose(at_frame(samples, 3, 13), [3, 0, 0]) and at_frame(samples, 20, 13) is None and at_frame(samples, 30, 13) is not None
    maps = rasterize([{"sourceFrame": 7, "objects": [{"polygons": [[[0, 0], [9, 0], [9, 9], [0, 9]]]}, {"polygons": [[[5, 5], [9, 5], [9, 9]]]}]}], [7], (20, 20))
    assert maps[0, 2, 2] == 1 and maps[0, 8, 8] == 2 and maps[0, 15, 15] == 0
    for site in CLIPS:  # the fixture graph resolves to existing reference files
        ref = reference(site)
        for key in ("droid", "names", "object_map", "mask_root", "dynamic", "analysis", "events", "splat"):
            assert ref[key].exists(), (site, key, ref[key])
    assert reference("me340")["nodes"]["camera"] == "S03K" and reference("samsclub-a2")["nodes"]["camera"] == "N30", "lens-K camera, not the old one"
    assert len(holdout_frames("me340")) == 86, "E5b's held-out frames of run 232"
    print("fast_report_eval self-check passed: names, recalls, rule ticks, interpolation, rasterising, fixture graph")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", type=Path)
    p.add_argument("--site", choices=sorted(CLIPS))
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--e9", type=Path, help="evaluate every call of an E9 run folder instead of a mirrored report")
    p.add_argument("--gaps", type=Path, help="measure the review's gaps into this new folder")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        self_check()
    elif a.gaps:
        gaps(a.gaps)
    elif a.e9:
        import m3_fu_e9_onegpu as e9
        report = {}
        for path in sorted(a.e9.glob("*gpu*.json")):
            for r in json.loads(path.read_text()).get("runs", []):
                if "shots" in r:
                    report[f"{path.stem}-{r['index']}-{r['mode']}"] = evaluate(layers_from_e9(r, e9.TEXTS[2:]), a.site, a.gpu)
        print(json.dumps({k: v["rows"] for k, v in report.items()}, indent=1))
    else:
        result = evaluate(load_layers(a.run_dir), a.site, a.gpu)
        (a.run_dir / f"eval-{a.site}.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
