"""Quality of one fast report next to the delivered report (FAST-BUILD-SPEC.md section 10): agreement, not accuracy.

References come from tests/fixtures/delivered-303/<site>.json by following its node graph from the names node (R22):
object map (R21) -> depth fuse with metric_scale (R17) -> camera (R07); mask root (R10); dynamic layer (R18) and its
motion analysis (R14); events (R24); splat package (R31). No run path is written here.

    python scripts/fast_report_eval.py RUN_DIR --site me340|samsclub-a2|walmart [--gpu]   # RUN_DIR: one mirrored report
    python scripts/fast_report_eval.py --e9 RUNS/m3-fu-e9-onegpu-003 --site me340          # the evaluator on E9's record
    python scripts/fast_report_eval.py --gaps OUT_DIR    # the review's gaps, 3 videos: fixed list vs per-video list, word match, person cut
    python scripts/fast_report_eval.py --self-check      # no GPU, no network

Click MVP (CLICK-MVP-SPEC section 8; section 'click MVP' below): clicks against SAM 3 references on X1's 60 evaluation
frames (the references run once per video on one Modal A100 and are kept as OUT/refs-<site>.npz), identity and physical
info against the delivered report, repeatability between calls, judgements from agent labels, latency against spec 7,
the spec 10 acceptance table. Default runs: fb/integrate's warm calls (D1's baseline).

    python scripts/fast_report_eval.py --mvp OUT [--runs site=RUN_DIR:REPORT,...] [--repeat site=RUN_DIR:REPORT:OFFSET:NAME,...]
        [--labels OUT] [--no-gpu] [--write-calibration]
    python scripts/mvp_sheets.py clicks|boxes OUT --site SITE [--run RUN_DIR:REPORT]   # agent audit sheets + label templates

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
sys.path += [str(REPO), str(REPO / "modal_apps"), str(REPO / "modal_apps/m3_e5"), str(REPO / "modal_apps/m3_fu_e2b"), str(REPO / "scripts")]
import sam3_app as s3  # noqa: E402  SAM 3 pin, image, weights volume

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
FIXTURES = REPO / "tests/fixtures/delivered-303"
CLIPS = {"me340": "me340-165", "samsclub-a2": "samsclub-337", "walmart": "walmart-190"}  # the bench's MP4: data/clips/<clip>/source-full.mp4
FLOOR, PAIRS, IOU_FOUND, MATCH_M, MAX_FRAMES, OUTLINE_FRAMES = .3, 80, .5, 1., 56, 12
PERSON_REF_SCORE = .4  # click references: person at the core's person rule (segment.PERSON_SCORE)
MIN_AREA_SRC = 1200  # reference masks for outlines, px at 1280x720 (E6b's 300 px at 640x480, x4)
E3_RULES = {"me340": PHASE2 / "runs/m3-exp-e3-people-eval-v3/ref10/findings.jsonl"}  # E3's replay of today's layer; no other site has one
E4_SAM3D = {"me340": (7, 30)}  # E4 s1cfg12: accepted / tried
S3_IMAGE = s3.image.add_local_python_source("sam3_app", "fast_report")  # fast_report: segment.dedupe for the click references
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
    if (out.get("object_cards") or {}).get("cards") == "blob":  # A's cards over 1 MB go out of line as the list itself
        out["object_cards"]["cards"] = json.loads(next(run_dir.rglob(latest["object_cards"]["blobs"]["cards"]["sha256"])).read_bytes())
    out["_report"] = reports.pop()
    for s in (out.get("cameras") or {}).get("shots", []):  # the viewer's names (fast_report.layers docstring) -> the ones read here
        s.setdefault("keyframes", s.get("keys"))
        s.setdefault("c2w_m", s.get("c2w"))
    if "people" in out:
        out["people"]["tracks"] = [t["points"] if isinstance(t, dict) else t for t in out["people"].get("tracks", [])]
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
    if "refs" in jobs:  # the click references (CLICK-MVP-SPEC 8.1): the run's words >= 0.3 deduped as the core does, person >= 0.4
        import base64
        from fast_report import segment
        job, t = jobs["refs"], time.perf_counter()
        images = [Image.open(io.BytesIO(p)).convert("RGB") for p in job["frames"]]
        words = ["person"] + job["words"]
        packed, meta = [], []
        for i, image in enumerate(images):  # one frame at a time: a frame's masks alone can pass 10 GB
            parts = []
            for _, w, sc, m in masks([image], words):
                keep = sc >= (PERSON_REF_SCORE if w == 0 else FLOOR)
                parts.append((torch.full((int(keep.sum()),), w), sc[keep].cpu(), m[keep]))
            if not parts:
                continue
            word, score, m = (torch.cat(x) for x in zip(*parts))
            person = m[(word == 0).to(m.device)].any(0) if (word == 0).any() else torch.zeros(m.shape[1:], dtype=torch.bool, device=m.device)
            obj = torch.nonzero(word > 0).flatten()
            kept, _ = segment.dedupe(torch.zeros(len(obj), dtype=torch.long), word[obj], score[obj], m[obj]) if len(obj) else (np.zeros(0, int), [])
            for k in torch.nonzero(word == 0).flatten().tolist() + obj[torch.from_numpy(np.asarray(kept, np.int64))].tolist():
                area = int(m[k].sum())
                meta.append([i, int(word[k]) - 1, round(float(score[k]), 4), area, round(float((m[k] & person).sum()) / max(area, 1), 4)])
                packed.append(np.packbits(m[k].cpu().numpy().reshape(-1)))
        buf = io.BytesIO()
        np.savez_compressed(buf, masks=np.stack(packed) if packed else np.zeros((0, 1), np.uint8), meta=np.array(meta, float).reshape(-1, 5),
                            shape=np.array(images[0].size[::-1]))
        report["refs"] = {"npz_b64": base64.b64encode(buf.getvalue()).decode(), "masks": len(meta), "s": round(time.perf_counter() - t, 2),
                          "meta_columns": ["frame index", "word index (-1 person)", "score", "area px", "share on a person"]}
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


# ---------- click MVP (CLICK-MVP-SPEC section 8): what D adds ----------
# Layers read (spec 3.3, 4.9, 5.6): 'pick' (data + blobs pick/depth), 'object_cards' (inline or blob 'cards'), 'judgements'
# (evidence blobs ev-*). Their versions are the patch versions: pick v1/v2, cards v1-v3, judgements v1-v3.

X1_RUN = PHASE2 / "runs/fx-x1-fps-004"  # X1's 60 evaluation frames per video (outlines.eval_frames)
FB_RUNS = {"me340": ("fb-integrate-me340-006", "fb-me340-e84efffd-1790649860"),  # fb/integrate's warm calls (runs/fb-results)
           "samsclub-a2": ("fb-integrate-samsclub-002", "fb-samsclub-a2-d5e0c855-1790650595"),
           "walmart": ("fb-integrate-walmart-001", "fb-walmart-c0761a2a-1790651216")}
REF_MIN_PX, REF_ON_PERSON, CLICK_ERODE, BG_MARGIN, CLICKS_PER_REF, BG_PER_FRAME = 1200, .5, 3, 5, 2, 5
PICK_IOU, PICK_COVER, PERSON_IOU, MATCH_DELIVERED_M, LONG_M = .3, .5, .5, .5, 3.
MIN_PATH_M = .5  # repeatability: a shot whose aligned cameras moved less cannot be Sim3-aligned (ME340 shot 0: 2 cm, scale 0.80)
# ponytail: spec 4.2's seed size table, used for today's boxes (the baseline) and for 'large class' in the L1 acceptance. The
# MVP's cards carry their own size_check (A's cards.CLASS_SIZE); the rows read that when it is there.
CLASS_SIZE = [  # (names, longest side lo, hi in m, large class)
    (("marker", "eraser", "wrench", "screwdriver", "hammer", "bolt", "drill bit", "knob", "switch", "battery", "bottle", "can", "cup",
      "glove", "tape"), .01, .6, False),
    (("monitor", "display screen", "tv"), .2, 1.6, False), (("exit sign",), .1, .6, False), (("sign", "label"), .05, 2., False),
    (("fire extinguisher",), .3, 1., False), (("box", "carton", "package", "crate", "tote", "bin", "bag"), .05, 1.5, False),
    (("pallet",), .8, 1.4, False), (("stacked boxes", "pallet of goods"), .3, 3.5, False),
    (("cart", "trolley", "shopping cart", "pallet jack"), .5, 2.2, False), (("chair",), .3, 1.3, False), (("stool",), .3, .8, False),
    (("table", "desk", "workbench"), .5, 3.5, False), (("cabinet", "locker"), .3, 2.5, False), (("door",), .6, 3., False),
    (("ladder",), .5, 6., True), (("machine", "lathe", "mill", "cnc machine", "forklift"), .5, 6., True),
    (("shelf", "rack", "shelving", "display rack", "conveyor", "duct", "pipe"), .3, 30., True),
    (("cable", "hose", "cord", "wire"), .1, 30., True), (("spill",), .05, 5., False), (("ceiling light", "light fixture"), .1, 2.5, False)]
VERTICAL_PLANES = ("door", "control panel", "controller panel")  # 8.3 known verticals: planar slope must include 90 deg
VERTICAL_AXES = ("shelf post", "rack support", "shelf support", "vertical support", "upright")  # principal-axis tilt must include 0


def size_class(name):
    """Head-noun match (the name ends with the entry's words; longest entry wins) -> (class, lo, hi, large)."""
    t = tokens(name or "")
    best = None
    for names, lo, hi, large in CLASS_SIZE:
        for n in names:
            w = tokens(n)
            if t[-len(w):] == w and (best is None or len(w) > best[0]):
                best = (len(w), names[0], lo, hi, large)
    return best[1:] if best else ("other", 0., 6., False)


# ---------- layers and blobs ----------

def blob_bytes(run_dir, sha):
    path = Path(run_dir) / "blobs/sha256" / sha
    raw = (path if path.exists() else next(Path(run_dir).rglob(sha))).read_bytes()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def patch_versions(run_dir, report, layer):
    """Every patch of `layer` for `report`, oldest version first."""
    out = [p for p in (json.loads(f.read_text()) for f in Path(run_dir).rglob("patches/*.json"))
           if p.get("schema") == "panoptes-fast-patch-v1" and p["report"] == report and p["layer"] == layer]
    return sorted(out, key=lambda p: (p["version"], p["seq"]))


def patch_data(run_dir, patch, blob=None):
    """A patch's data, or its named JSON blob when the data went out of line (cards over 1 MB, outlines)."""
    if blob and blob in (patch.get("blobs") or {}):
        got = json.loads(blob_bytes(run_dir, patch["blobs"][blob]["sha256"]))
        # A's cards blob is the card list itself (the layer's data says "cards": "blob"): the layer with the list in place
        return {**(patch.get("data") or {}), blob: got} if isinstance(got, list) else got
    return patch.get("data") or {}


# ---------- pick (spec 3.1-3.3) ----------

def pick_encode(maps, frames, entities, depth=None):
    """(n,h,w) uint16 id maps + frame records -> (data, pick blob gzip bytes[, depth blob]): panoptes-pick-v1."""
    pairs, recs = [], []
    for m, f in zip(maps, frames):
        flat = np.asarray(m, np.uint16).ravel()
        starts = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
        runs = np.diff(np.r_[starts, flat.size])
        vals = np.repeat(flat[starts], (runs + 65534) // 65535)  # runs over 65535 are split
        lens = np.concatenate([[65535] * (r // 65535) + ([r % 65535] if r % 65535 else []) for r in runs]).astype(np.uint16)
        recs.append({**f, "h": m.shape[0], "w": m.shape[1], "offset": sum(len(p) for p in pairs), "pairs": len(vals)})
        pairs.append(np.stack([vals, lens], 1).astype("<u2"))
    data = {"format": "panoptes-pick-v1", "source_wh": [1280, 720], "entities": entities, "frames": recs}
    return data, gzip.compress(np.concatenate(pairs).tobytes() if pairs else b"", 6)


class Pick:
    """A decoded pick layer: frames by time, one id map decoded on demand (the last one cached, as the viewer does)."""

    def __init__(self, data, blob):
        self.data, self.frames = data, sorted(data["frames"], key=lambda f: f["t"])
        self.pairs = np.frombuffer(blob, "<u2").reshape(-1, 2)
        self.times = np.array([f["t"] for f in self.frames])
        self.sw, self.sh = data.get("source_wh", [1280, 720])
        self._cache = (None, None)
        n = 0
        for f in data["frames"]:  # offsets are in pairs; accept a layer that counts them in bytes
            assert f["offset"] in (n, 4 * n), f"pick frame offset {f['offset']} is neither pairs ({n}) nor bytes"
            f["_start"], n = n, n + f["pairs"]

    def map(self, i):
        if self._cache[0] != i:
            f = self.frames[i]
            p = self.pairs[f["_start"]:f["_start"] + f["pairs"]]
            m = np.repeat(p[:, 0], p[:, 1].astype(np.int64))
            assert m.size == f["h"] * f["w"], (m.size, f["h"], f["w"])
            self._cache = (i, m.reshape(f["h"], f["w"]))
        return self._cache[1]

    def frame_at(self, t):
        """The last pick frame with t_key <= t (VideoView.frameAt), or -1 before the first."""
        return int(np.searchsorted(self.times, t + 1e-6, side="right")) - 1

    def at(self, t, x, y):
        """(entity id or None, frame index, code) for a click at video time t, source pixel (x, y)."""
        i = self.frame_at(t)
        if i < 0 or (self.frames[i].get("t_end") is not None and t >= self.frames[i]["t_end"]):  # past a cut: no map (as pickAt)
            return None, -1, 0
        f = self.frames[i]
        code = int(self.map(i)[min(int(y * f["h"] / self.sh), f["h"] - 1), min(int(x * f["w"] / self.sw), f["w"] - 1)])
        return self.data["entities"][code] if code else None, i, code


def pick_from_outlines(outlines, size=(360, 640)):
    """Pick v1's stand-in for a run without a pick layer (D1's baseline): each outlines keyframe's polygons painted at
    640x360, largest first so the smaller wins (spec 3.2). Today's outlines hold no people, so neither does this."""
    import cv2
    W = outlines.get("width", 1280)
    entities, code, maps, frames = [None], {}, [], []
    for f in sorted(outlines["frames"], key=lambda f: f["timeSec"]):
        m, objs = np.zeros(size, np.uint16), []
        for o in f["objects"]:
            polys = [np.round(np.asarray(p, float) * size[1] / W).astype(np.int32) for p in o.get("polygons") or [] if len(p) >= 3]
            if polys:
                objs.append((sum(abs(cv2.contourArea(p)) for p in polys), o["entityId"], polys))
        for _, eid, polys in sorted(objs, key=lambda x: -x[0]):
            if eid not in code:
                code[eid] = len(entities)
                entities.append(eid)
            cv2.fillPoly(m, polys, code[eid])
        maps.append(m)
        frames.append({"t": f["timeSec"], "t_end": f.get("endTimeSec"), "frame": f["sourceFrame"], "source": f.get("source", "segmented")})
    data, blob = pick_encode(maps, frames, entities)
    data["note"] = "D1 baseline: rasterised from today's outlines layer (no people, no depth)"
    return data, blob


def run_picks(run_dir, report, layers):
    """[(name, Pick)] for every pick version in the run, or today's outlines as 'pick v1 (from outlines)'."""
    out = [(f"pick v{p['version']}", Pick(p["data"], blob_bytes(run_dir, p["blobs"]["pick"]["sha256"])))
           for p in patch_versions(run_dir, report, "pick")]
    if not out and "outlines" in layers:
        data, blob = pick_from_outlines(layers["outlines"])
        out = [("pick v1 (from outlines)", Pick(data, gzip.decompress(blob)))]
    return out


# ---------- click references and clicks (spec 8.1) ----------

def eval_frames(site):
    return json.loads((X1_RUN / f"raw-{site}.json").read_text())["outlines"]["eval_frames"]


def load_refs(path):
    """refs-<site>.npz -> {frames, words, masks (n, H, W) bool, frame, word (-1 person), score, area, on_person}."""
    z = np.load(path, allow_pickle=False)
    H, W = (int(v) for v in z["shape"])
    meta = z["meta"]
    return {"frames": z["frames"].tolist(), "words": z["words"].tolist(), "H": H, "W": W, "packed": z["masks"],
            "frame": meta[:, 0].astype(int), "word": meta[:, 1].astype(int), "score": meta[:, 2], "area": meta[:, 3], "on_person": meta[:, 4]}


def ref_mask(refs, k):
    """Reference k as an (H, W) bool mask (stored bit-packed: ~1500 full-size masks a video would hold 1.4 GB)."""
    return np.unpackbits(refs["packed"][k], count=refs["H"] * refs["W"]).reshape(refs["H"], refs["W"]).astype(bool)


def compute_refs(jobs, out):
    """{site: (frames, words, clip)} -> out/refs-<site>.npz, one Modal container (SAM 3 bf16, retries 0)."""
    import base64
    with modal.enable_output(), app.run():
        for site, (frames, words, clip) in jobs.items():
            r = json.loads(sam3_eval.remote({"refs": {"frames": decode(PHASE2 / "data/clips" / clip / "source-full.mp4", frames), "words": words}}))
            z = dict(np.load(io.BytesIO(base64.b64decode(r["refs"]["npz_b64"]))))
            np.savez_compressed(out / f"refs-{site}.npz", frames=np.array(frames), words=np.array(words), **z)
            (out / f"refs-{site}.json").write_text(json.dumps({k: v for k, v in r.items() if k != "refs"} | {"refs": {k: v for k, v in r["refs"].items() if k != "npz_b64"}}, indent=1))
            print(site, "refs:", r["refs"]["masks"], "masks in", r["refs"]["s"], "s", flush=True)


def make_clicks(refs, seed=0):
    """Seed-0 clicks: 2 uniform pixels in each static object reference's 3 px-eroded interior (>= 1200 px, < 50% on a person),
    2 in each person reference (same area rule), 5 background pixels per frame >= 5 px outside every reference mask."""
    import cv2
    rng = np.random.default_rng(seed)
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CLICK_ERODE + 1,) * 2)
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BG_MARGIN + 1,) * 2)
    clicks = []
    for fi, frame in enumerate(refs["frames"]):
        idx = np.flatnonzero(refs["frame"] == fi)
        for k in idx:
            person = refs["word"][k] < 0
            if refs["area"][k] < REF_MIN_PX or (not person and refs["on_person"][k] >= REF_ON_PERSON):
                continue
            ys, xs = np.nonzero(cv2.erode(ref_mask(refs, k).astype(np.uint8), erode))
            for j in rng.choice(len(ys), min(CLICKS_PER_REF, len(ys)), replace=False) if len(ys) else []:
                clicks.append({"frame": frame, "x": int(xs[j]), "y": int(ys[j]), "kind": "person" if person else "object", "ref": int(k)})
        union = np.zeros((refs["H"], refs["W"]), bool)
        for k in idx:
            union |= ref_mask(refs, k)
        taken = cv2.dilate(union.astype(np.uint8), grow)
        ys, xs = np.nonzero(taken == 0)
        for j in rng.choice(len(ys), min(BG_PER_FRAME, len(ys)), replace=False) if len(ys) else []:
            clicks.append({"frame": frame, "x": int(xs[j]), "y": int(ys[j]), "kind": "background", "ref": -1})
    return clicks


def click_row(pick, refs, clicks, fps, drop_frames=()):
    """Clicks resolved through the pick layer (Pick.at: spec 3.1's rule) and scored against the SAM 3 references: hit, unknown,
    correct (the picked entity's region on that pick frame has IoU >= 0.3 with the reference or covers >= 50% of it; people:
    a person entity at IoU >= 0.5), wrong entity, background false hits. -> (row, per-click records)."""
    import cv2
    drop, recs = set(drop_frames), []
    for c in clicks:
        if c["frame"] in drop:
            continue
        ent, i, code = pick.at(c["frame"] / fps, c["x"], c["y"])
        r = {**c, "entity": ent, "pick_frame": pick.frames[i]["frame"] if i >= 0 else None, "source": pick.frames[i].get("source") if i >= 0 else None,
             "iou": None, "cover": None, "correct": None}
        if c["kind"] != "background" and ent is not None:
            f = pick.frames[i]
            region = pick.map(i) == code
            ref = cv2.resize(ref_mask(refs, c["ref"]).astype(np.float32), (f["w"], f["h"]), interpolation=cv2.INTER_AREA) >= .5
            inter = float((region & ref).sum())
            r["iou"], r["cover"] = round(inter / max(float((region | ref).sum()), 1), 4), round(inter / max(float(ref.sum()), 1), 4)
            is_person = ent.startswith("person")
            r["correct"] = (is_person and r["iou"] >= PERSON_IOU) if c["kind"] == "person" else (not is_person and (r["iou"] >= PICK_IOU or r["cover"] >= PICK_COVER))
        recs.append(r)

    def rates(rs):
        n = len(rs)
        if not n:
            return {"n": 0}
        hit = [r for r in rs if r["entity"] is not None]
        return {"n": n, "hit_rate": round(len(hit) / n, 4), "unknown_rate": round(1 - len(hit) / n, 4),
                "correct_rate": round(sum(bool(r["correct"]) for r in rs) / n, 4),
                "wrong_entity_rate": round(sum(r["correct"] is False for r in hit) / n, 4),
                "correct_by_cover_only": sum(bool(r["correct"]) and r["iou"] < PICK_IOU for r in hit),
                # D1's audit: the cover branch calls a whole workbench right for a board lying on it; the IoU branch alone
                # agreed with the agent on 'same object' 0.915 vs 0.709 (165 labelled clicks, runs/mvp-d1-baseline-001)
                "correct_iou_only_rate": round(sum(r["correct"] is not None and r["iou"] >= (PERSON_IOU if r["kind"] == "person" else PICK_IOU)
                                                   and (r["kind"] == "person") == r["entity"].startswith("person") for r in hit) / n, 4)}
    obj = [r for r in recs if r["kind"] == "object"]
    bg = [r for r in recs if r["kind"] == "background"]
    row = {"frames": len({c["frame"] for c in clicks} - drop), "dropped_frames_on_keyframes": sorted(drop & {c["frame"] for c in clicks}),
           "object": rates(obj), "person": rates([r for r in recs if r["kind"] == "person"]),
           "object_by_source": {s: rates([r for r in obj if r["source"] == s]) for s in sorted({r["source"] for r in obj if r["source"]})},
           "background": {"n": len(bg), "false_hit_rate": round(sum(r["entity"] is not None for r in bg) / max(len(bg), 1), 4)},
           "criterion": "reported for pick v1 and v2; densify stays only if v2's correct rate beats v1's on all three videos; background false hits <= 10%"}
    row["pass"] = row["background"]["false_hit_rate"] <= .1 if bg else None
    return row, recs


# ---------- our objects: cards when present, else today's boxes (the baseline) ----------

def shot_floor(shot):
    """(point, unit up towards the cameras) of a cameras-layer shot's floor, or None."""
    f = shot.get("floor") or {}
    if "normal" not in f:
        return None
    n, p = np.asarray(f["normal"], float), np.asarray(f["point_m"], float)
    n /= np.linalg.norm(n)
    c = np.asarray(shot["c2w_m"][0], float)[:3, 3]
    return p, (n if (c - p) @ n > 0 else -n)


def plane_basis(up):
    a = np.cross(up, [1., 0, 0] if abs(up[0]) < .9 else [0, 1., 0])
    a /= np.linalg.norm(a)
    return a, np.cross(up, a)


def fact(x, k=1., angle=False):
    """A card value -> (value, u, u without its scale part) or None when not observed / not measurable / only a bound.
    u = k * sqrt(sum parts^2) (spec 4.4), so the scale-free u is sqrt(u^2 - (k * parts.scale)^2)."""
    if not isinstance(x, dict) or x.get("status") in ("not observed", "not measurable") or x.get("bound") or "least" in str(x.get("status", "")):
        return None
    v, u = x.get("value"), x.get("u")
    if v is None or u is None:
        return None
    if angle:
        return v, float(u), float(u)
    parts = x.get("parts") or {}
    scale = k * float(parts["scale"]) if "scale" in parts else k * .2 * float(np.linalg.norm(v))
    return v, float(u), float(np.sqrt(max(float(u) ** 2 - scale ** 2, 0.)))


def floor_to_shot(ff, xy):
    o, x, z = (np.asarray(ff[k], float) for k in ("origin_m", "x", "z"))
    return o + xy[0] * x + xy[1] * np.cross(z, x)


def ours_objects(layers):
    """Our objects as the rows read them: id, shot, centroid (shot frame), name, route, p, plausible, longest side,
    top/base/height (value, u, u_noscale) or None, sides [(v, u, u_ns)] long first (both footprint sides observed) or None,
    position (xy floor frame, u, u_ns), slope/tilt (deg). Cards when the run has them; else today's boxes, with no u."""
    import cv2
    objs = (layers.get("objects") or {}).get("objects") or []
    shots = {s["index"]: s for s in layers["cameras"]["shots"]}
    cards = layers.get("object_cards")
    out = []
    if cards:
        k = (cards.get("calibration") or {}).get("k") or {}
        cent = {o["id"]: o["centroid_m"] for o in objs}
        ffs = {s["index"]: s.get("floor_frame") for s in cards.get("shots", [])}
        for c in cards["cards"]:
            if c.get("kind", "object") != "object":
                continue
            ph, idn = c.get("physical") or {}, c.get("identity") or {}
            box, ff = ph.get("box") or {}, ffs.get(c["shot"])
            centroid = cent.get(c["id"]) or (floor_to_shot(ff, box["center_m"][:2]) + np.asarray(ff["z"]) * box["center_m"][2] if ff and box else None)
            w, d = fact(ph.get("width"), k.get("extent", 1)), fact(ph.get("depth"), k.get("extent", 1))
            out.append({"id": c["id"], "shot": c["shot"], "centroid": centroid, "name": idn.get("name"), "route": idn.get("decided_by"),
                        "p": idn.get("confidence"), "calibrated": idn.get("calibrated"),
                        "plausible": (ph.get("size_check") or {}).get("status", "plausible") == "plausible",
                        "longest": max(box["size_m"]) if box.get("size_m") else None,
                        "top": fact(ph.get("top_above_floor"), k.get("height", 1)), "base": fact(ph.get("base_above_floor"), k.get("height", 1)),
                        "height": fact(ph.get("height"), k.get("extent", 1)), "sides": sorted([w, d], key=lambda f: -f[0]) if w and d else None,
                        "position": fact(ph.get("position_xy"), k.get("position", 1)), "floor_frame": ff,
                        "slope": fact(ph.get("planar_slope_deg"), angle=True), "tilt": fact(ph.get("principal_axis_tilt_deg"), angle=True)})
        return out
    for o in objs:
        lo, hi = np.asarray(o["box_min_m"], float), np.asarray(o["box_max_m"], float)
        corners = np.array([[a[0], b[1], c[2]] for a in (lo, hi) for b in (lo, hi) for c in (lo, hi)])
        floor = shot_floor(shots[o["shot"]])
        top = base = sides = None
        if floor:
            p, up = floor
            h = (corners - p) @ up
            base, top = (float(h.min()), None, None), (float(h.max()), None, None)
            a, b = plane_basis(up)
            (sw, sh) = cv2.minAreaRect(np.stack([(corners - p) @ a, (corners - p) @ b], 1).astype(np.float32))[1]
            sides = [(float(max(sw, sh)), None, None), (float(min(sw, sh)), None, None)]
        votes, cas = o.get("votes") or {}, o.get("cascade") or {}
        word = o.get("word") or o.get("label")
        named = cas.get("label") and cas.get("source")  # today's names: the cascade's when it gave one (fast_report_app.cascade_label)
        out.append({"id": o["id"], "shot": o["shot"], "centroid": o["centroid_m"], "name": cas["label"] if named else word, "word": word,
                    "route": f"cascade {cas['source'].split(' ')[0]}" if named else "sam3 vote",
                    "p": None if named else round(votes[word] / sum(votes.values()), 4) if word in votes and sum(votes.values()) else None,
                    "calibrated": False, "plausible": None, "longest": float((hi - lo).max()), "top": top, "base": base,
                    "height": None if top is None else (top[0] - base[0], None, None), "sides": sides, "position": None, "floor_frame": None,
                    "slope": None, "tilt": None})
    for o in out:  # today's boxes: plausibility from the seed table on SAM 3's word (longest side only; no placement rules)
        _, lo, hi, _ = size_class(o["word"])
        o["plausible"] = lo <= o["longest"] <= hi
    return out


# ---------- matching to the delivered report ----------

def match_delivered(ours, ref, align, max_m=MATCH_DELIVERED_M):
    """1:1 Hungarian on centroid distance <= max_m after the camera Sim3; delivered objects seen >= 3 times on the reference
    frames of our shot, never in another shot, status 'clear'. -> ([(ours, delivered entity, metres)], n delivered)."""
    from scipy.optimize import linear_sum_assignment
    s, R, t, shot = align
    names = json.loads(ref["names"].read_text())
    ents, xyz = reference_objects(ref, set(range(min(shot["keyframes"]), max(shot["keyframes"]) + 1)))
    keep = [i for i, e in enumerate(ents) if (names.get(e["entityId"]) or {}).get("status") == "clear"]
    ents, xyz = [ents[i] for i in keep], xyz[keep]
    mine = [o for o in ours if o["shot"] == shot["index"] and o["centroid"] is not None]
    if not mine or not ents:
        return [], len(ents)
    X = (s * (R @ np.array([o["centroid"] for o in mine], float).T)).T + t
    D = np.linalg.norm(xyz[:, None] - X[None], axis=2)
    r, c = linear_sum_assignment(np.where(D <= max_m, D, 1e6))
    return [(mine[j], ents[i], float(D[i, j])) for i, j in zip(r, c) if D[i, j] <= max_m], len(ents)


def identity_row(ours, ref, align, video):
    """Our name vs the delivered name on matched objects (same_name), by route; -> (row, calibration rows)."""
    names = json.loads(ref["names"].read_text())
    pairs, n_ref = match_delivered(ours, ref, align)
    rows = [{"video": video, "id": o["id"], "delivered": e["entityId"], "ours": o["name"], "theirs": names[e["entityId"]]["category"],
             "route": o["route"], "p": o["p"], "correct": int(same_name(o["name"] or "", names[e["entityId"]]["category"]))} for o, e, _ in pairs]
    by = {}
    for r in rows:
        by.setdefault(r["route"], []).append(r["correct"])
    word = [same_name(o["word"], names[e["entityId"]]["category"]) for o, e, _ in pairs if o.get("word")]
    return {"delivered_clear": n_ref, "matched": len(rows), "agree": sum(r["correct"] for r in rows),
            "sam3_word_agreement": round(float(np.mean(word)), 4) if word else None,
            "agreement": round(float(np.mean([r["correct"] for r in rows])), 4) if rows else None,
            "by_route": {k: {"n": len(v), "agreement": round(float(np.mean(v)), 4)} for k, v in sorted(by.items())},
            "rule": "1:1 Hungarian <= 0.5 m after the camera Sim3; fast_report_eval.same_name(ours, delivered category)",
            "note": "agreement with the delivered report (itself model-named), not ground truth"}, rows


def delivered_values(e, mpn):
    import cv2
    w, h = cv2.minAreaRect((np.asarray(e["footprintPlanNative"], float) * mpn).astype(np.float32))[1]
    return {"top": e["heightNative"] * mpn, "base": e["baseNative"] * mpn, "sides": sorted([w, h], reverse=True)}


def dist(values):
    v = np.asarray(values, float)
    return {"n": int(v.size), **({"median": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3)} if v.size else {})}


def physical_row(ours, ref, align):
    """Spec 8.3 against the delivered report: top/base above floor and footprint sides on matched objects, in the delivered
    metric frame (ours x the Sim3 scale, so geometry not scale); coverage = share with |delta| <= u (scale part removed);
    box inflation (L1); implausible sizes per class; known verticals."""
    s = align[0]
    pairs, n_ref = match_delivered(ours, ref, align)
    got = {q: [] for q in ("top", "base", "long_side", "short_side")}
    not_compared = {"top": 0, "sides": 0}
    for o, e, _ in pairs:
        dv = delivered_values(e, ref["mpn"])
        for q in ("top", "base"):
            if o[q]:
                got[q].append((abs(s * o[q][0] - dv[q]), None if o[q][2] is None else s * o[q][2]))
            else:
                not_compared["top"] += q == "top"
        if o["sides"]:
            for q, mine, theirs in zip(("long_side", "short_side"), o["sides"], dv["sides"]):
                got[q].append((abs(s * mine[0] - theirs), None if mine[2] is None else s * mine[2]))
        else:
            not_compared["sides"] += 1
    agreement = {}
    for q, xs in got.items():
        d = [x[0] for x in xs]
        withu = [x for x in xs if x[1] is not None]
        agreement[q] = {**dist(d), "coverage": round(float(np.mean([a <= b for a, b in withu])), 4) if withu else None, "with_u": len(withu)}
    row = {"matched": len(pairs), "delivered_clear": n_ref, "sim3_scale_ours_to_delivered": round(s, 4), "agreement": agreement,
           "not_compared": {"top (not observed)": not_compared["top"], "footprint (a side not observed)": not_compared["sides"]},
           "inflation": inflation(ours), "known_verticals": known_verticals(pairs, ref),
           "note": "delivered = heightNative / baseNative / min-area rectangle of footprintPlanNative x metres_per_native (p98 / p2 / hull): "
                   "the same definitions as the cards; agreement with the delivered report, itself at estimated scale"}
    return row


def inflation(ours):
    """L1: the longest-side distribution, the share over 3 m, the same among shown (plausible) boxes of non-large classes,
    and the implausible sizes per class."""
    cls = lambda o: size_class(o.get("word") or o["name"])  # noqa: E731  today's boxes were checked on SAM 3's word
    longest = [o["longest"] for o in ours if o["longest"] is not None]
    shown = [o for o in ours if o["plausible"] and o["longest"] is not None and not cls(o)[3]]
    per = {}
    for o in ours:
        if not o["plausible"]:
            per[cls(o)[0]] = per.get(cls(o)[0], 0) + 1
    over = lambda xs: round(float(np.mean(np.asarray(xs) > LONG_M)), 4) if len(xs) else None  # noqa: E731
    return {"objects": len(longest), "over_3m_share": over(longest), **{k: v for k, v in dist(longest).items() if k != "n"},
            "max": round(max(longest), 3) if longest else None, "shown_non_large": len(shown),
            "shown_non_large_over_3m_share": over([o["longest"] for o in shown]),
            "implausible": sum(per.values()), "implausible_by_class": dict(sorted(per.items(), key=lambda kv: -kv[1])),
            "criterion": "<= 5% of shown (plausible) boxes of non-large classes over 3 m; no implausible size shown as a fact"}


def known_verticals(pairs, ref):
    """Matched objects the delivered report names door / control panel (planar slope must include 90 deg within +-u) or a
    shelf upright (principal-axis tilt must include 0 within +-u)."""
    names = json.loads(ref["names"].read_text())
    checks = []
    for o, e, _ in pairs:
        theirs = names[e["entityId"]]["category"]
        for keys, key, want in ((VERTICAL_PLANES, "slope", 90.), (VERTICAL_AXES, "tilt", 0.)):
            if any(k in theirs for k in keys):
                f = o[key]
                checks.append({"id": o["id"], "delivered": theirs, "quantity": key, "shown": f is not None,
                               **({"value": round(float(f[0]), 2), "u": round(f[1], 2), "includes": abs(f[0] - want) <= f[1]} if f else {})})
    shown = [c for c in checks if c["shown"]]
    return {"n": len(checks), "shown": len(shown), "misses": [c for c in shown if not c["includes"]], "checks": checks,
            "pass": not any(not c["includes"] for c in shown) if shown else None}


# ---------- repeatability (warm call vs the shifted window) ----------

def interp_cameras(keys, c2w, want):
    """c2w at frames `want` from keyframes (sorted): slerp between neighbours; None outside the covered range."""
    import m3_exp_geometry as geo
    keys, out = np.asarray(keys), []
    for f in want:
        j = int(np.searchsorted(keys, f, side="right"))
        if j == 0 or j > len(keys) or (j == len(keys) and keys[-1] != f):
            out.append(None)
            continue
        a, b = (j - 1, j - 1) if keys[j - 1] == f else (j - 1, j)
        out.append(np.asarray(c2w[a], float) if a == b else geo.interp_c2w(np.asarray(c2w[a], float), np.asarray(c2w[b], float), (f - keys[a]) / (keys[b] - keys[a])))
    return out


def repeat_row(layers_a, layers_b, offset, min_keys=5):
    """Spec 8.3: two calls (B's frame f is A's f + offset), each A shot aligned to the B shot covering it by a Sim3 on A's
    keyframes (B's cameras interpolated there); objects matched 1:1 (Hungarian, <= 0.5 m); per family |delta| against
    sqrt(uA^2 + uB^2) with the scale parts removed (B's lengths x the Sim3 scale: geometry, not scale).
    -> (row, {family: [(|delta|, u)]}) for k_family."""
    import m3_exp_geometry as geo
    from scipy.optimize import linear_sum_assignment
    A, B = ours_objects(layers_a), ours_objects(layers_b)
    fam = {"height": [], "extent": [], "position": [], "angle": []}
    deltas = {q: [] for q in ("top", "base", "height", "long_side", "short_side", "position", "slope", "tilt")}
    shots, matched, angle_seen = [], 0, {"both": 0, "one": 0, "neither": 0}
    for sa in layers_a["cameras"]["shots"]:
        ka = sa["keyframes"]
        sb = max(layers_b["cameras"]["shots"], key=lambda s: sum(ka[0] <= k + offset <= ka[-1] for k in s["keyframes"]))
        kb = [k + offset for k in sb["keyframes"]]
        cams = interp_cameras(kb, sb["c2w_m"], ka)
        use = [i for i, c in enumerate(cams) if c is not None]
        if len(use) < min_keys:
            continue
        ca = np.asarray(sa["c2w_m"], float)[use][:, :3, 3]
        if np.linalg.norm(np.diff(ca, axis=0), axis=1).sum() < MIN_PATH_M:  # a still camera: the Sim3's scale and rotation are undetermined
            shots.append({"a": sa["index"], "b": sb["index"], "keyframes_aligned": len(use), "skipped": "camera path < 0.5 m: no Sim3"})
            continue
        s, R, t = geo.align_sim3(np.stack([cams[i] for i in use]), np.asarray(sa["c2w_m"], float)[use])
        floor = shot_floor(sa)
        up = floor[1] if floor else np.array([0, -1., 0])
        oa = [o for o in A if o["shot"] == sa["index"] and o["centroid"] is not None]
        ob = [o for o in B if o["shot"] == sb["index"] and o["centroid"] is not None]
        shots.append({"a": sa["index"], "b": sb["index"], "keyframes_aligned": len(use), "sim3_scale_b_to_a": round(s, 4), "objects": [len(oa), len(ob)]})
        if not oa or not ob:
            continue
        xa = np.array([o["centroid"] for o in oa], float)
        xb = (s * (R @ np.array([o["centroid"] for o in ob], float).T)).T + t
        D = np.linalg.norm(xa[:, None] - xb[None], axis=2)
        r, c = linear_sum_assignment(np.where(D <= MATCH_DELIVERED_M, D, 1e6))
        for i, j in zip(r, c):
            if D[i, j] > MATCH_DELIVERED_M:
                continue
            a, b = oa[i], ob[j]
            matched += 1
            for q, f in (("top", "height"), ("base", "height"), ("height", "extent")):
                if a[q] and b[q]:
                    deltas[q].append(abs(a[q][0] - s * b[q][0]))
                    if a[q][2] is not None:
                        fam[f].append((deltas[q][-1], float(np.hypot(a[q][2], s * b[q][2]))))
            if a["sides"] and b["sides"]:
                for q, x, y in zip(("long_side", "short_side"), a["sides"], b["sides"]):
                    deltas[q].append(abs(x[0] - s * y[0]))
                    if x[2] is not None:
                        fam["extent"].append((deltas[q][-1], float(np.hypot(x[2], s * y[2]))))
            if a["position"] and b["position"] and a["floor_frame"] and b["floor_frame"]:
                pa = floor_to_shot(a["floor_frame"], a["position"][0])
                pb = s * R @ floor_to_shot(b["floor_frame"], b["position"][0]) + t
                u = float(np.hypot(a["position"][2], s * b["position"][2]))
            else:  # today's boxes: the centroids, horizontal part
                pa, pb, u = xa[i], xb[j], None
            v = pa - pb
            deltas["position"].append(float(np.linalg.norm(v - (v @ up) * up)))
            if u is not None:
                fam["position"].append((deltas["position"][-1], u))
            for q in ("slope", "tilt"):
                both = a[q] is not None and b[q] is not None
                angle_seen["both" if both else "one" if (a[q] or b[q]) else "neither"] += 1
                if both:
                    deltas[q].append(abs(a[q][0] - b[q][0]))
                    fam["angle"].append((deltas[q][-1], float(np.hypot(a[q][1], b[q][1]))))
    cov = {f: {"n": len(x), "coverage": round(float(np.mean([d <= u for d, u in x])), 4) if x else None} for f, x in fam.items()}
    return {"shots": shots, "matched": matched, "objects": [len(A), len(B)], "delta": {q: dist(v) for q, v in deltas.items()},
            "coverage_k1": cov, "angles_shown": angle_seen,
            "not_measurable_share": round((angle_seen["one"] + angle_seen["neither"]) / max(sum(angle_seen.values()), 1), 4),
            "rule": "coverage: |delta| <= sqrt(u1^2 + u2^2), scale parts removed, B's lengths x the Sim3 scale"}, fam


# ---------- judgements (spec 8.4) ----------

def judgement_sample(by_video, n=60, seed=0):
    """Rows to label: every FAIL, then PASS and NEEDS_REVIEW rows dealt per check (seeded) until n."""
    rng = np.random.default_rng(seed)
    rows = [(v, r) for v, j in sorted(by_video.items()) for r in j.get("rows", [])]
    pick = [x for x in rows if x[1]["verdict"] == "FAIL"]
    pool = {}
    for x in rows:
        if x[1]["verdict"] in ("PASS", "NEEDS_REVIEW"):
            pool.setdefault((x[1]["check"], x[1]["verdict"]), []).append(x)
    queues = [list(np.array(v, dtype=object)[rng.permutation(len(v))]) for _, v in sorted(pool.items())]
    while len(pick) < n and any(queues):
        for q in queues:
            if q and len(pick) < n:
                pick.append(tuple(q.pop(0)))
    return pick


def judgement_row(by_video, labels):
    """labels: {'<video>:<row id>': 'present' | 'absent' | 'cannot tell'} (agent-made). Per check: verdict shares over all
    rows; FAIL precision on labelled FAILs; false PASS = a PASS labelled 'present' (target 0)."""
    per = {}
    for v, j in sorted(by_video.items()):
        for r in j.get("rows", []):
            p = per.setdefault(r["check"], {"rows": 0, "verdicts": {}, "labelled": 0, "fail_present": 0, "fail_absent": 0, "false_pass": []})
            p["rows"] += 1
            p["verdicts"][r["verdict"]] = p["verdicts"].get(r["verdict"], 0) + 1
            lab = labels.get(f"{v}:{r['id']}")
            if lab:
                p["labelled"] += 1
                p["fail_present"] += r["verdict"] == "FAIL" and lab == "present"
                p["fail_absent"] += r["verdict"] == "FAIL" and lab == "absent"
                if r["verdict"] == "PASS" and lab == "present":
                    p["false_pass"].append(f"{v}:{r['id']}")
    out = {}
    for check, p in sorted(per.items()):
        n = p["rows"]
        judged = p["fail_present"] + p["fail_absent"]
        out[check] = {"rows": n, "labelled": p["labelled"], "shares": {k: round(c / n, 4) for k, c in sorted(p["verdicts"].items())},
                      "fail_precision": round(p["fail_present"] / judged, 4) if judged else None, "fail_precision_n": judged,
                      "false_pass": len(p["false_pass"]), "false_pass_rows": p["false_pass"]}
    return {"checks": out, "false_pass_total": sum(c["false_pass"] for c in out.values()), "labelled": sum(c["labelled"] for c in out.values()),
            "labeller": "agent-made labels (contact sheets); not ground truth", "pass": (sum(c["false_pass"] for c in out.values()) == 0) if labels else None}


# ---------- latency (spec 7, 8.5) ----------

MVP_TARGETS = [  # (row, layer, version, target): an absolute second, or (base layer, its version, + seconds)
    ("pick v1", "pick", 1, ("outlines", 1, 1.)), ("cards v1", "object_cards", 1, ("objects", 1, 10.)),
    ("judgements v1", "judgements", 1, ("objects", 1, 12.)), ("cards v2", "object_cards", 2, ("objects", 1, 60.)),
    ("judgements v2", "judgements", 2, ("objects", 1, 60.)), ("pick v2", "pick", 2, ("objects", 1, 40.)),
    ("cards v3", "object_cards", 3, ("objects", 1, 40.))]
UNCHANGED = ("cameras", "people", "events")  # within +-1 s of fb/integrate's warm call on the same video


def first_written(run, layer, version=None, want=None, patches=None):
    rows = sorted((r for r in run.get("layers", []) if r["layer"] == layer and (version is None or r.get("version") == version)), key=lambda r: r["seq"])
    if want is not None:  # a predicate on the patch data (first SAM 3D model, splat preview)
        ok = {p["seq"] for p in patches or [] if p["layer"] == layer and want(p.get("data") or {})}
        rows = [r for r in rows if r["seq"] in ok]
    return rows[0].get("written_s") if rows else None


def sam3_done(run):
    ends = [s["end_s"] for s in run.get("stages", []) if s["stage"].startswith("sam3.vocab")]
    return max(ends) if ends else None


def stage_s(run, name):
    return next((s["s"] for s in run.get("stages", []) if s["stage"] == name), None)


def latency_row(run, patches, fb_run, click_latency=None):
    """One call's clock against spec 7: per layer written time and its target (+ ok), fb/integrate's warm call beside,
    GPU peaks and flags, the click latency from C's headless check when given ({'p50_ms', 'p95_ms', 'n'})."""
    rows = {}
    for layer in UNCHANGED:
        w, ref = first_written(run, layer), first_written(fb_run, layer)
        rows[layer] = {"written_s": w, "fb_written_s": ref, "target": "fb +- 1 s", "ok": None if w is None or ref is None else abs(w - ref) <= 1.}
    lift, lift_fb = stage_s(run, "lift"), stage_s(fb_run, "lift")
    gap = None if sam3_done(run) is None or first_written(run, "objects", 1) is None else first_written(run, "objects", 1) - sam3_done(run)
    gap_fb = first_written(fb_run, "objects", 1) - sam3_done(fb_run)
    rows["objects v1"] = {"written_s": first_written(run, "objects", 1), "lift_s": lift, "fb_lift_s": lift_fb,
                          "sam3_done_to_objects_s": None if gap is None else round(gap, 3), "fb_sam3_done_to_objects_s": round(gap_fb, 3),
                          "target": "lift <= fb + 0.5 s; SAM 3 done -> objects <= fb + 0.5 s",
                          "ok": None if lift is None or gap is None else lift <= lift_fb + .5 and gap <= gap_fb + .5}
    for name, layer, version, target in MVP_TARGETS:
        w = first_written(run, layer, version)
        base = first_written(run, target[0], target[1])
        limit = None if base is None else base + target[2]
        rows[name] = {"written_s": w, "target_s": None if limit is None else round(limit, 3), "target": f"{target[0]} v{target[1]} + {target[2]:g} s",
                      "ok": None if w is None or limit is None else w <= limit}
    for name, layer, want, limit in (("first SAM 3D model", "models", lambda d: bool(d.get("models")), 120.),
                                     ("splat preview", "splat", lambda d: d.get("kind") == "preview", 230.)):
        w = first_written(run, layer, want=want, patches=patches)
        rows[name] = {"written_s": w, "target_s": limit, "ok": None if w is None else w <= limit,
                      "fb_written_s": ((fb_run.get("milestones") or {}).get("first_model" if layer == "models" else "splat_preview") or {}).get("written_s")}
    return {"layers": rows, "gpu_peak_gib": [g["peak_gb"] for g in run.get("gpu_peak", [])], "flags": run.get("flags") or [],
            "over_90": [f"{s['stage']}: {s['peak_gb']}" for s in run.get("stages", []) if any(s.get("over_90") or [])],
            "click_latency": click_latency or "not measured here (C's headless check writes it)",
            "click_ok": None if not click_latency else click_p95(click_latency) < 100}


def click_p95(c):
    """p95 click -> card latency in ms from {'p95_ms'} or C's click-check.json ({'videos': [{'latencyMs': {'p95'}}]}; the worst video)."""
    return c["p95_ms"] if "p95_ms" in c else max(v["latencyMs"]["p95"] for v in c["videos"])


# ---------- one video, one call: every MVP row ----------

def mvp_rows(run_dir, report, site, refs=None, fb=True, clicks=None):
    """Every spec-8 row that one mirrored call supports (no GPU; refs = load_refs(...) for clicks). -> (rows, extras)."""
    layers = load_layers(run_dir, report)
    for name in ("object_cards", "judgements"):  # large cards / judgements go out of line as blobs
        vs = patch_versions(run_dir, report, name)
        if vs:
            layers[name] = patch_data(run_dir, vs[-1], "cards" if name == "object_cards" else None)
    ref = reference(site)
    fps = layers["video"]["fps"]
    rows, extras = {}, {"layers": layers}
    rows["cameras"], rows["scale"], align = camera_rows(layers, ref)
    ours = ours_objects(layers)
    extras["ours"] = ours
    rows["source"] = "object_cards" if layers.get("object_cards") else "today's objects boxes (baseline: no u, no cards)"
    rows["identity"], extras["identity_rows"] = identity_row(ours, ref, align, site)
    rows["physical"] = physical_row(ours, ref, align)
    if refs is not None:
        keys = {k for s in layers["cameras"]["shots"] for k in s["keyframes"]}
        clicks = clicks or make_clicks(refs)
        rows["clicks"], extras["clicks"] = {}, {}
        for name, pick in run_picks(run_dir, report, layers):
            rows["clicks"][name], extras["clicks"][name] = click_row(pick, refs, clicks, fps, drop_frames=keys)
            extras.setdefault("picks", {})[name] = pick
    run_json = Path(run_dir) / "reports" / report / "run.json"
    if run_json.exists() and fb:
        fb_dir, fb_report = FB_RUNS[site]
        fb_run = json.loads((PHASE2 / "runs" / fb_dir / "reports" / fb_report / "run.json").read_text())
        run = json.loads(run_json.read_text())
        patches = [json.loads(p.read_text()) for p in (Path(run_dir) / "reports" / report / "patches").glob("*.json")]
        rows["latency"] = latency_row(run, patches, fb_run)
        rows["latency"]["first_call_after_boot"] = (run.get("boot") or {}).get("first_call_after_boot")
    if layers.get("object_cards"):
        rows["card_check"] = card_check(layers["object_cards"])
    if layers.get("judgements"):
        rows["judgements_counts"] = layers["judgements"].get("counts")
        extras["judgements"] = layers["judgements"]
    return rows, extras


# ---------- the decider on X8 set d (spec 8.4, 5.5) ----------

X8_RUN = PHASE2 / "runs/fx-x8-jev-001"
B_ANSWERS = PHASE2 / "runs/mvp-b-judge-decider-001/answers.json"  # B2's set-d run with the deployed prompt


def x8_probs(decisions, model):
    """{item id: p_yes} for one decider in an X8-shaped decisions file ({'res': {model: {'d|yn|<id>': {'probs': [yes, no]}}}})."""
    res = json.loads(Path(decisions).read_text())["res"].get(model) or {}
    return {k.split("|")[-1]: float((v["probs"] if isinstance(v, dict) else v)[0]) for k, v in res.items() if k.startswith("d|yn|")}


def decider_rows(decisions=None):
    """Raw vs calibrated (Platt per question, cross-fitted by source clip) for every decider with set-d probabilities; the
    Jev-Omni swap test of spec 5.5 (Brier >= 0.02 lower, ECE <= 0.05, <= 0.2 s a question) against calibrated Qwen."""
    from fast_report import calibration as cb
    items = json.loads((X8_RUN / "sets/d.json").read_text())["items"]
    decisions = Path(decisions or X8_RUN / "decisions-d-pass2.json")
    got = {m: x8_probs(decisions, m) for m in ("qwen", "jev")}
    gem = X8_RUN / "gemini-d.json"
    if gem.exists():
        got["gemini-stated-p"] = {k: min(max(float(a["p_yes"]), 0.), 1.) for k, a in json.loads(gem.read_text())["answers"].items()}
    if B_ANSWERS.exists():  # B's deployed prompt (yes / no / cannot tell), the one the pipeline calibrates
        got["qwen deployed (B)"] = {x["id"]: float(x["p"]) for x in json.loads(B_ANSWERS.read_text())["items"] if x.get("p") is not None}
    out = {m: cb.decider(items, p, f"{decisions.name}:{m}") for m, p in got.items() if p}
    lat = (json.loads(decisions.read_text())["res"].get("latency") or {})
    swap = None
    if "qwen" in out and "jev" in out:
        q, j = out["qwen"]["all"]["calibrated_cv"], out["jev"]["all"]["calibrated_cv"]
        s_q = ((lat.get("jev_batched_d_yn") or {}).get("per_question_s"))
        swap = {"brier_qwen": q["brier"], "brier_jev": j["brier"], "ece_jev": j["ece"], "jev_s_per_question_batched": s_q,
                "swap": bool(j["brier"] <= q["brier"] - .02 and j["ece"] <= .05 and (s_q or 1) <= .2),
                "rule": "spec 5.5: Brier lower by >= 0.02, ECE <= 0.05, <= 0.2 s a question, all cross-fitted on set d"}
    return out, swap


# ---------- the whole MVP evaluation over mirrored calls ----------

def mvp(out, runs, repeats, gpu=True, labels_dir=None, decisions=None, write_calibration=False):
    """runs: {site: (run dir, report)}; repeats: {site: [(run dir, report, frame offset, name)]} compared with runs[site].
    -> out/summary.json, out/summary.md, clicks-*.json (per-click records), refs-*.npz (the click references)."""
    from fast_report import calibration as cb
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    need = {}
    for site, (run_dir, report) in runs.items():
        if not (out / f"refs-{site}.npz").exists():
            need[site] = (eval_frames(site), load_layers(run_dir, report)["objects"]["vocabulary"], CLIPS[site])
    if need and gpu:
        compute_refs(need, out)
    summary = {"schema": "panoptes-mvp-eval-v1", "runs": {k: list(map(str, v)) for k, v in runs.items()}, "sites": {}, "ident_rows": []}
    fam_all, judged = {}, {}
    for site, (run_dir, report) in runs.items():
        refs = load_refs(out / f"refs-{site}.npz") if (out / f"refs-{site}.npz").exists() else None
        clicks = None
        if refs is not None:  # the seed-0 clicks depend on the references only: made once, kept beside them
            cpath = out / f"clicks-{site}.json"
            if not cpath.exists():
                cpath.write_text(json.dumps(make_clicks(refs)))
            clicks = json.loads(cpath.read_text())
        rows, extras = mvp_rows(run_dir, report, site, refs, clicks=clicks)
        for name, recs in (extras.get("clicks") or {}).items():
            (out / f"clicks-{site}-{name.split(' ')[1]}.json").write_text(json.dumps(recs))
        summary["ident_rows"] += extras["identity_rows"]
        if extras.get("judgements"):
            judged[site] = extras["judgements"]
        rows["repeat"] = {}
        for run_b, report_b, offset, name in repeats.get(site, []):
            rows["repeat"][name], fam = repeat_row(extras["layers"], load_layers(run_b, report_b), offset)
            for f, xs in fam.items():  # k comes from the shifted window when there is one (spec 8.3), else from every repeat
                fam_all.setdefault("shifted" in name, {}).setdefault(f, []).extend(xs)
        if labels_dir and (Path(labels_dir) / f"labels-clicks-{site}.json").exists():
            rows["click_audit"] = click_audit(json.loads((Path(labels_dir) / f"labels-clicks-{site}.json").read_text()))
        if labels_dir and (Path(labels_dir) / f"labels-boxes-{site}.json").exists():
            rows["box_audit"] = box_audit(json.loads((Path(labels_dir) / f"labels-boxes-{site}.json").read_text()))
        summary["sites"][site] = rows
    if judged:
        lab = Path(labels_dir or out) / "labels-judgements.json"
        labels = {k: v["label"] for k, v in json.loads(lab.read_text()).items() if v.get("label")} if lab.exists() else {}
        summary["judgements"] = judgement_row(judged, labels)
    summary["decider"], summary["jev_swap"] = decider_rows(decisions)
    has_cards = any(r["source"] == "object_cards" for r in summary["sites"].values())
    fam_k = fam_all.get(True) or fam_all.get(False) or {}
    summary["k_family"] = cb.k_family(fam_k) if any(fam_k.values()) else None
    summary["k_source"] = "warm vs shifted window" if fam_all.get(True) else "repeats of the same window" if fam_k else None
    summary["identity_calibration"] = cb.identity([r for r in summary["ident_rows"] if r["p"] is not None]) if summary["ident_rows"] else None
    if write_calibration and has_cards:  # identity and k come from cards (today's boxes carry neither); B owns 'questions'
        cal = cb.write({k: v for k, v in (("identity", summary["identity_calibration"]), ("k", summary["k_family"])) if v})
        (out / "calibration.json").write_text(json.dumps(cal, indent=1))
    elif write_calibration:
        summary["calibration_note"] = "not written: no object_cards in these runs (identity and k are learnt from cards)"
    summary["acceptance"] = acceptance(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "summary.md").write_text(summary_md(summary))
    return summary


def decider_acceptance(deciders):
    """Spec 10: the deployed decider's cross-fitted ECE <= 0.10 on set d, per question that has >= 3 yes and >= 3 no
    (binned p(yes), B's gate); questions without positives measure false alarms only and are listed apart."""
    name = "qwen deployed (B)" if "qwen deployed (B)" in deciders else "qwen" if "qwen" in deciders else None
    if not name:
        return {"pass": None}
    qs = deciders[name]["questions"]
    judged = {q: v["calibrated_cv"]["ece_p_yes"] for q, v in qs.items() if v["positives"] >= 3 and v["n"] - v["positives"] >= 3}
    return {"pass": all(e <= .1 for e in judged.values()) if judged else None, "decider": name, "ece_p_yes_cv": judged,
            "false_alarm_only": {q: v["calibrated_cv"]["false_yes"] for q, v in qs.items() if q not in judged}}


CARD_META = {"box", "dropped_share", "merged_from", "size_check", "footprint_xy", "walkway", "nearest_walked_path", "primitive",
             "level", "box_min_m", "box_max_m", "fragmented_support", "reason"}  # A's card bookkeeping, not facts


def card_check(cards):
    """Spec 10 'Cards': every physical field has value +- u, a level and a scale label, or a not-observed / not-measurable
    reason; an implausible size never shows a number (L1). -> {cards, fields, violations: [id.field: why]}."""
    bad, n, loud = [], 0, set()
    for c in (cards or {}).get("cards", []):
        if c.get("kind", "object") != "object":
            continue
        ph = c.get("physical") or {}
        implausible = (ph.get("size_check") or {}).get("status") not in (None, "plausible")
        for k, v in ph.items():
            if k in CARD_META:
                continue
            n += 1
            if not isinstance(v, dict):
                bad.append(f"{c['id']}.{k}: not a fact record")
            elif "value" in v and not v.get("status"):
                miss = [x for x in ("u", "level", "scale") if v.get(x) is None and not (x == "scale" and k.endswith("_deg"))]
                if miss:
                    bad.append(f"{c['id']}.{k}: no {', '.join(miss)}")
                if implausible and k not in ("position_xy",) and "needs review" not in str(v.get("status", "")):
                    loud.add(c["id"])  # spec 4.2: the field should read 'implausible for a <class> ...: needs review'
            elif not v.get("reason"):
                bad.append(f"{c['id']}.{k}: no value and no reason")
    return {"cards": len((cards or {}).get("cards", [])), "fields": n, "violations": len(bad), "examples": bad[:20],
            "implausible_cards_with_numbers": len(loud), "implausible_examples": sorted(loud)[:10]}


def acceptance(summary):
    """Spec 10, fixed before the runs: one verdict per criterion per video (None = cannot be judged from these runs)."""
    out = {}
    for site, r in summary["sites"].items():
        lat = (r.get("latency") or {}).get("layers") or {}
        timed = {k: v["ok"] for k, v in lat.items() if v.get("ok") is not None}
        mvp_layers = [n for n, *_ in MVP_TARGETS]
        missing = [n for n in mvp_layers if n in lat and lat[n]["written_s"] is None]
        is_mvp = len(missing) < len(mvp_layers)  # a run with none of the MVP layers is a baseline: its times are not judged
        inf = r["physical"]["inflation"]
        clicks = r.get("clicks") or {}
        v1 = next((c for n, c in clicks.items() if n.startswith("pick v1")), None)
        v2 = clicks.get("pick v2")
        a = {"times": {"pass": (all(timed.values()) and not missing) if timed and is_mvp else None,
                       "missed": sorted(k for k, ok in timed.items() if not ok), "not written": missing if is_mvp else "baseline: no MVP layers"},
             "memory": {"pass": None if "latency" not in r else max(r["latency"]["gpu_peak_gib"] or [0]) <= 72, "peaks_gib": (r.get("latency") or {}).get("gpu_peak_gib")},
             "L1_shown_non_large_over_3m": {"pass": None if inf["shown_non_large_over_3m_share"] is None else inf["shown_non_large_over_3m_share"] <= .05,
                                           "share": inf["shown_non_large_over_3m_share"]},
             "clicks_v2_beats_v1": {"pass": None if not (v1 and v2) else v2["object"]["correct_rate"] > v1["object"]["correct_rate"],
                                    "v1": v1 and v1["object"]["correct_rate"], "v2": v2 and v2["object"]["correct_rate"]},
             "clicks_background_false_hits": {"pass": None if not clicks else all(c["background"]["false_hit_rate"] <= .1 for c in clicks.values()),
                                              "rates": {n: c["background"]["false_hit_rate"] for n, c in clicks.items()}},
             "known_verticals_within_u": {"pass": r["physical"]["known_verticals"]["pass"], "misses": len(r["physical"]["known_verticals"]["misses"])},
             "cards_complete": {"pass": None if "card_check" not in r else r["card_check"]["violations"] == 0,
                                **{k: v for k, v in (r.get("card_check") or {}).items() if not k.startswith("implausible")}},
             "L1_no_implausible_number": {"pass": None if "card_check" not in r else r["card_check"]["implausible_cards_with_numbers"] == 0,
                                          "cards": (r.get("card_check") or {}).get("implausible_cards_with_numbers"),
                                          "note": "numbers kept on an implausible card count unless the field says 'needs review' (spec 4.2); "
                                                  "a viewer that greys them (spec 6) is checked on C's screenshots"}}
        out[site] = a
    k = summary.get("k_family") or {}
    out["all"] = {"repeat_coverage_k_le_2": {"pass": None if not k else all(v["k"] <= 2 and (v["coverage_after"] or 0) >= .9 for v in k.values() if v["n"]),
                                             "k": {f: v["k"] for f, v in k.items()}, "source": summary.get("k_source")},
                  "decider_ece_le_0.10": decider_acceptance(summary.get("decider") or {}),
                  "judgements_zero_false_pass": {"pass": (summary.get("judgements") or {}).get("pass"),
                                                 "false_pass": (summary.get("judgements") or {}).get("false_pass_total")}}
    return out


def click_audit(labels):
    """labels: {tile: {'auto': bool, 'label': 'same object' | 'part of it' | 'different' | 'unclear'}} -> agreement of the
    automatic rule with the agent's labels ('same object' and 'part of it' count as right)."""
    rows = [v for v in labels.values() if v.get("label") in ("same object", "part of it", "different")]
    agree = [bool(v["auto"]) == (v["label"] != "different") for v in rows]
    count = {k: sum(v.get("label") == k for v in labels.values()) for k in ("same object", "part of it", "different", "unclear")}
    rules = {}  # the spec's rule against simpler ones, where the tiles carry iou / cover (mvp_sheets writes them)
    if rows and all("iou" in v for v in rows):
        for name, rule in (("spec: iou >= 0.3 or cover >= 0.5", lambda v: v["iou"] >= PICK_IOU or v["cover"] >= PICK_COVER),
                           ("iou >= 0.3", lambda v: v["iou"] >= PICK_IOU), ("iou >= 0.2", lambda v: v["iou"] >= .2)):
            rules[name] = {"vs same-or-part": round(float(np.mean([rule(v) == (v["label"] != "different") for v in rows])), 4),
                           "vs same only": round(float(np.mean([rule(v) == (v["label"] == "same object") for v in rows])), 4)}
    return {"tiles": len(labels), "labels": count, "agreement_with_auto_rule": round(float(np.mean(agree)), 4) if agree else None, "rules": rules,
            "auto_correct_but_labelled_different": sum(bool(v["auto"]) and v["label"] == "different" for v in rows),
            "auto_wrong_but_labelled_right": sum(not v["auto"] and v["label"] != "different" for v in rows), "labeller": "agent"}


def box_audit(labels):
    """L1 audit (spec 8.3): labels {tile: {'group': 'flagged' | 'largest-unflagged', 'label': 'inflated' | 'size right' | 'unclear'}}
    -> the flag's precision (inflated among decided flagged boxes) and what it misses (inflated among the largest unflagged)."""
    out = {}
    for g in ("flagged", "largest-unflagged"):
        c = {k: sum(v["group"] == g and v["label"] == k for v in labels.values()) for k in ("inflated", "size right", "unclear")}
        out[g] = {**c, "inflated_share_of_decided": round(c["inflated"] / max(c["inflated"] + c["size right"], 1), 4)}
    return {**out, "labeller": "agent", "reading": "flagged: share truly inflated (the flag's precision); largest-unflagged: share the flag missed"}


def summary_md(summary):
    """One table across the videos (spec 8.6), then the decider table."""
    sites = list(summary["sites"])
    lines = ["| row | " + " | ".join(sites) + " |", "|---|" + "---|" * len(sites)]
    f3 = lambda x: None if x is None else f"{x:.3f}"  # noqa: E731
    pct = lambda x: None if x is None else f"{100 * x:.1f}%"  # noqa: E731

    def row(name, fn):
        vals = []
        for s in sites:
            try:
                v = fn(summary["sites"][s])
            except (KeyError, TypeError, IndexError):
                v = None
            vals.append("—" if v is None else str(v))
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    row("source of the physical rows", lambda r: r["source"].split(" (")[0])
    for name in sorted({n for r in summary["sites"].values() for n in (r.get("clicks") or {})}):
        c = lambda r, n=name: r["clicks"][n]  # noqa: E731
        row(f"{name}: object clicks correct / unknown / wrong (n)", lambda r, c=c: "{} / {} / {} ({})".format(
            pct(c(r)["object"]["correct_rate"]), pct(c(r)["object"]["unknown_rate"]), pct(c(r)["object"]["wrong_entity_rate"]), c(r)["object"]["n"]))
        row(f"{name}: object clicks correct by IoU >= 0.3 alone", lambda r, c=c: pct(c(r)["object"].get("correct_iou_only_rate")))
        row(f"{name}: correct on segmented / projected pick frames", lambda r, c=c: " / ".join(
            f"{pct(v.get('correct_rate'))} ({v['n']})" for v in (c(r)["object_by_source"].get(k, {"n": 0}) for k in ("segmented", "projected"))))
        row(f"{name}: person clicks correct (n)", lambda r, c=c: f"{pct(c(r)['person'].get('correct_rate'))} ({c(r)['person']['n']})")
        row(f"{name}: background false hits (<= 10%)", lambda r, c=c: pct(c(r)["background"]["false_hit_rate"]))
    row("click audit: auto rule agrees with the agent", lambda r: f"{pct(r['click_audit']['agreement_with_auto_rule'])} {r['click_audit']['labels']}")
    row("click audit, agent 'same object' only: spec rule / IoU >= 0.3 / IoU >= 0.2 agree", lambda r: " / ".join(
        pct(v["vs same only"]) for v in r["click_audit"]["rules"].values()) or None)
    row("L1 audit: flagged boxes truly inflated (inflated / size right / unclear)", lambda r: "{} ({} / {} / {})".format(
        pct(r["box_audit"]["flagged"]["inflated_share_of_decided"]), *(r["box_audit"]["flagged"][k] for k in ("inflated", "size right", "unclear"))))
    row("L1 audit: 30 largest unflagged boxes inflated (inflated / size right / unclear)", lambda r: "{} ({} / {} / {})".format(
        pct(r["box_audit"]["largest-unflagged"]["inflated_share_of_decided"]), *(r["box_audit"]["largest-unflagged"][k] for k in ("inflated", "size right", "unclear"))))
    row("identity: matched / delivered clear; agreement", lambda r: f"{r['identity']['matched']}/{r['identity']['delivered_clear']}; {pct(r['identity']['agreement'])}")
    row("identity: SAM 3 word on the same pairs", lambda r: pct(r["identity"].get("sam3_word_agreement")))
    for q in ("top", "base", "long_side", "short_side"):
        a = lambda r, q=q: r["physical"]["agreement"][q]  # noqa: E731
        row(f"vs delivered: {q} median / p90 abs delta m (coverage; n)", lambda r, a=a: "{} / {} ({}; {})".format(
            f3(a(r).get("median")), f3(a(r).get("p90")), pct(a(r)["coverage"]) or "no u", a(r)["n"]))
    row("scale: ours / delivered", lambda r: f3(r["scale"]["ours_over_reference"]))
    i = lambda r: r["physical"]["inflation"]  # noqa: E731
    row("L1: longest side > 3 m, all boxes", lambda r: f"{pct(i(r)['over_3m_share'])} of {i(r)['objects']}; p90 {i(r)['p90']} m, max {i(r)['max']} m")
    row("L1: shown non-large boxes > 3 m (<= 5%)", lambda r: f"{pct(i(r)['shown_non_large_over_3m_share'])} of {i(r)['shown_non_large']}")
    row("L1: implausible sizes flagged", lambda r: i(r)["implausible"])
    row("known verticals: shown of n / misses", lambda r: f"{r['physical']['known_verticals']['shown']} of {r['physical']['known_verticals']['n']} / {len(r['physical']['known_verticals']['misses'])}")
    for name in sorted({n for r in summary["sites"].values() for n in (r.get("repeat") or {})}):
        p = lambda r, n=name: r["repeat"][n]  # noqa: E731
        row(f"repeat ({name}): matched; median abs delta top / long side / position m", lambda r, p=p: "{} of {}; {} / {} / {}".format(
            p(r)["matched"], p(r)["objects"], f3(p(r)["delta"]["top"].get("median")), f3(p(r)["delta"]["long_side"].get("median")), f3(p(r)["delta"]["position"].get("median"))))
        row(f"repeat ({name}): coverage at k=1 height / extent / position / angle", lambda r, p=p: " / ".join(
            pct(p(r)["coverage_k1"][f]["coverage"]) or "no u" for f in ("height", "extent", "position", "angle")))
    for layer in ("cameras", "people", "events", "objects v1", "pick v1", "cards v1", "judgements v1", "cards v2", "judgements v2", "pick v2",
                  "cards v3", "first SAM 3D model", "splat preview"):
        t = lambda r, lay=layer: r["latency"]["layers"][lay]  # noqa: E731
        row(f"time: {layer}, s (target)", lambda r, t=t: "{} ({}){}".format(t(r)["written_s"], t(r).get("target_s") or t(r).get("target"),
                                                                             "" if t(r)["ok"] is None else " ✓" if t(r)["ok"] else " ✗"))
    row("GPU peaks GiB (flags)", lambda r: f"{r['latency']['gpu_peak_gib']} ({len(r['latency']['flags'])})")
    md = ("# Click MVP validation\n\nAgreement with the delivered reports (themselves at estimated scale), not ground truth. "
          "Labels are agent-made.\n\n" + "\n".join(lines) + "\n")
    if summary.get("decider"):
        md += ("\n## Decider on X8 set d (Platt per question, cross-fitted by source clip; agent labels)\n\n"
               "| decider | n | Brier raw → cal | ECE raw → cal | AUROC raw → cal | said yes raw → cal |\n|---|---|---|---|---|---|\n")
        for m, d in summary["decider"].items():
            a, c = d["all"]["raw"], d["all"]["calibrated_cv"]
            md += f"| {m} | {a['n']} | {a['brier']} → {c['brier']} | {a['ece']} → {c['ece']} | {a['auroc']} → {c['auroc']} | {a['said_yes_share']} → {c['said_yes_share']} |\n"
        if summary.get("jev_swap"):
            md += f"\nJev-Omni swap test (spec 5.5): {json.dumps(summary['jev_swap'])}\n"
    if summary.get("k_family"):
        md += f"\nk_family ({summary.get('k_source')}): {json.dumps(summary['k_family'])}\n"
    if summary.get("judgements"):
        md += "\n## Judgements (agent labels)\n\n| check | rows | shares | FAIL precision (n) | false PASS |\n|---|---|---|---|---|\n"
        for c, v in summary["judgements"]["checks"].items():
            md += f"| {c} | {v['rows']} | {v['shares']} | {v['fail_precision']} ({v['fail_precision_n']}) | {v['false_pass']} |\n"
    acc = summary.get("acceptance") or {}
    if acc:
        mark = lambda v: "—" if v is None else "✓" if v else "✗"  # noqa: E731
        crit = [c for c in next(iter(acc.values()))] if sites else []
        md += "\n## Acceptance (spec 10; — = not judged by these runs)\n\n| criterion | " + " | ".join(sites) + " |\n|---|" + "---|" * len(sites) + "\n"
        for c in crit:
            md += f"| {c} | " + " | ".join(mark(acc[s][c]["pass"]) for s in sites) + " |\n"
        for c, v in acc.get("all", {}).items():
            md += f"| {c} (all videos) | {mark(v['pass'])} {json.dumps({k: x for k, x in v.items() if k != 'pass'})} |" + " |" * (len(sites) - 1) + "\n"
    return md


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
    mvp_self_check()
    print("fast_report_eval self-check passed: names, recalls, rule ticks, interpolation, rasterising, fixture graph, MVP rows")


def mvp_self_check():
    """Spec 8 rows on synthetic data: pick RLE round trip and lookup, outlines -> pick, clicks and their scoring, card facts,
    repeatability under a known Sim3 and frame offset, judgement shares, latency targets, the size table."""
    import cv2
    # pick: a 300x300 frame of zeros (runs > 65535 split) and a small frame with two entities
    big, small = np.zeros((300, 300), np.uint16), np.zeros((36, 64), np.uint16)
    small[5:20, 10:30], small[10:15, 20:25] = 1, 2
    data, blob = pick_encode([big, small], [{"t": 0., "frame": 0}, {"t": .2, "frame": 6}], [None, "obj-1-0", "person:1-3"])
    pk = Pick(data, gzip.decompress(blob))
    assert np.array_equal(pk.map(0), big) and np.array_equal(pk.map(1), small) and data["frames"][0]["pairs"] == 2
    assert pk.at(.1, 640, 360)[0] is None and pk.at(.25, 22.5 * 20, 12 * 20)[0] == "person:1-3" and pk.at(.25, 12 * 20, 6 * 20)[0] == "obj-1-0"
    assert pk.at(-.1, 0, 0)[1] == -1 and pk.at(9., 12 * 20, 6 * 20)[1] == 1, "before the first frame: none; after the last: the last"
    sq = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]  # noqa: E731
    d2, b2 = pick_from_outlines({"width": 1280, "frames": [{"timeSec": .5, "sourceFrame": 15, "source": "projected", "objects": [
        {"entityId": "shelf", "polygons": [sq(0, 0, 1000, 700)]}, {"entityId": "box", "polygons": [sq(100, 100, 300, 300)]}]}]})
    p2 = Pick(d2, gzip.decompress(b2))
    assert p2.at(.6, 200, 200)[0] == "box" and p2.at(.6, 800, 600)[0] == "shelf" and p2.at(.6, 1200, 710)[0] is None, "the smaller wins"
    # clicks: an object reference (the box), a person reference, background; the pick above resolves them
    H, W = 720, 1280
    obj, person = np.zeros((H, W), bool), np.zeros((H, W), bool)
    obj[100:300, 100:300], person[400:700, 1100:1250] = True, True
    refs = {"frames": [15], "words": ["box"], "H": H, "W": W, "packed": np.stack([np.packbits(obj.ravel()), np.packbits(person.ravel())]),
            "frame": np.array([0, 0]), "word": np.array([0, -1]), "score": np.array([.9, .9]), "area": np.array([obj.sum(), person.sum()], float),
            "on_person": np.array([0., 1.])}
    clicks = make_clicks(refs)
    assert [c["kind"] for c in clicks].count("object") == 2 and [c["kind"] for c in clicks].count("person") == 2 and [c["kind"] for c in clicks].count("background") == 5
    assert all(obj[c["y"], c["x"]] for c in clicks if c["kind"] == "object") and not any(obj[c["y"], c["x"]] or person[c["y"], c["x"]] for c in clicks if c["kind"] == "background")
    row, recs = click_row(p2, refs, clicks, fps=30.)
    assert row["object"]["correct_rate"] == 1. and row["person"]["correct_rate"] == 0., row  # no people in an outlines-built pick
    assert row["person"]["unknown_rate"] == 1. and row["background"]["n"] == 5, row  # the shelf covers most of the frame
    assert click_row(p2, refs, clicks, fps=30., drop_frames=[15])[0]["object"]["n"] == 0, "eval frames on keyframes are dropped"
    # card facts: the scale part comes out of u; 'not observed' and bounds give None
    f = fact({"value": 1.4, "u": .33, "parts": {"views": .09, "depth": .05, "floor": .02, "scale": .28}})
    assert abs(f[2] - np.sqrt(.33 ** 2 - .28 ** 2)) < 1e-9 and fact({"status": "not observed"}) is None and fact({"value": 2., "u": .1, "bound": "at least"}) is None
    assert fact({"value": 3., "u": 1.}, angle=True) == (3., 1., 1.)
    # repeatability: B = A moved by a Sim3 (s 1.25), its frames shifted by 3; heights scale by 1/s in B
    rng = np.random.default_rng(1)
    from scipy.spatial.transform import Rotation
    keys = list(range(0, 120, 6))
    c2w_a = np.stack([np.eye(4)] * len(keys))
    c2w_a[:, :3, 3] = np.stack([np.linspace(0, 4, len(keys)), 0.1 * np.sin(np.arange(len(keys))), np.linspace(0, 1, len(keys))], 1)
    c2w_a[:, :3, :3] = Rotation.from_euler("y", np.linspace(0, 40, len(keys))[:, None], degrees=True).as_matrix()
    s, R = 1.25, Rotation.from_euler("xyz", [5, 20, -3], degrees=True).as_matrix()
    t = np.array([.3, -.2, 1.])
    to_b = lambda x: (R.T @ (np.asarray(x) - t)) / s  # noqa: E731  B's frame: A = s R B + t
    c2w_b = np.stack([np.eye(4)] * len(keys))
    c2w_b[:, :3, :3] = R.T @ c2w_a[:, :3, :3]
    c2w_b[:, :3, 3] = [to_b(c) for c in c2w_a[:, :3, 3]]
    floor = {"normal": [0, 1., 0], "point_m": [0, -1.6, 0]}
    cents = rng.uniform(-2, 2, (6, 3)) + [2, 0, 3]
    mk = lambda o, v: {"id": f"o{o}", "shot": 0, "word": "box", "centroid_m": list(v), "box_min_m": list(np.asarray(v) - .2), "box_max_m": list(np.asarray(v) + .2)}  # noqa: E731
    la = {"cameras": {"shots": [{"index": 0, "keyframes": keys, "c2w_m": c2w_a.tolist(), "floor": floor}]}, "objects": {"objects": [mk(i, c) for i, c in enumerate(cents)]}}
    lb = {"cameras": {"shots": [{"index": 0, "keyframes": [k - 3 for k in keys], "c2w_m": c2w_b.tolist(), "floor": floor}]},
          "objects": {"objects": [mk(i, to_b(c)) for i, c in enumerate(cents[::-1])]}}
    rep, fam = repeat_row(la, lb, 3)
    assert rep["matched"] == 6 and rep["shots"][0]["sim3_scale_b_to_a"] == 1.25 and rep["delta"]["position"]["median"] < 1e-6, rep
    assert fam["position"] == [], "today's boxes carry no u"
    # judgements: FAIL precision, false PASS, shares; the sample takes every FAIL first
    jv = {"me340": {"rows": [{"id": "J4:a", "check": "J4", "verdict": "FAIL"}, {"id": "J4:b", "check": "J4", "verdict": "PASS"},
                             {"id": "J4:c", "check": "J4", "verdict": "NEEDS_REVIEW"}, {"id": "J1:d", "check": "J1", "verdict": "NO_DATA"}]}}
    jr = judgement_row(jv, {"me340:J4:a": "present", "me340:J4:b": "present", "me340:J4:c": "absent"})
    assert jr["checks"]["J4"]["fail_precision"] == 1. and jr["false_pass_total"] == 1 and jr["pass"] is False and jr["checks"]["J1"]["shares"] == {"NO_DATA": 1.}
    assert judgement_sample(jv, n=2)[0][1]["id"] == "J4:a" and len(judgement_sample(jv, n=9)) == 3
    # latency: relative targets and the fb comparison
    run = {"layers": [{"layer": "objects", "version": 1, "seq": 1, "written_s": 30.}, {"layer": "outlines", "version": 1, "seq": 2, "written_s": 33.},
                      {"layer": "pick", "version": 1, "seq": 3, "written_s": 33.5}, {"layer": "object_cards", "version": 1, "seq": 4, "written_s": 41.},
                      {"layer": "cameras", "version": 1, "seq": 0, "written_s": 20.}],
           "stages": [{"stage": "lift", "s": 1.1}, {"stage": "sam3.vocab.wave2@gpu0", "end_s": 27.}], "gpu_peak": [{"peak_gb": 60.}, {"peak_gb": 50.}]}
    fbr = {"layers": [{"layer": "objects", "version": 1, "seq": 1, "written_s": 35.}, {"layer": "cameras", "version": 1, "seq": 0, "written_s": 18.5}],
           "stages": [{"stage": "lift", "s": .5}, {"stage": "sam3.vocab.wave1@gpu0", "end_s": 32.}]}
    assert click_p95({"videos": [{"latencyMs": {"p95": 33.9}}, {"latencyMs": {"p95": 120.}}]}) == 120.
    lat = latency_row(run, [], fbr, {"p50_ms": 12, "p95_ms": 40})["layers"]
    assert lat["pick v1"]["ok"] and lat["cards v1"]["ok"] is False and lat["objects v1"]["ok"] is False and lat["cameras"]["ok"] is False
    assert lat["judgements v1"]["ok"] is None and lat["pick v1"]["target_s"] == 34.
    good = {"value": 1.4, "u": .3, "level": "coarse", "scale": "estimated"}
    cc = card_check({"cards": [{"id": "a", "physical": {"top_above_floor": good, "depth": {"status": "not observed", "reason": "one side"},
                                                        "planar_slope_deg": {"value": 88., "u": 3., "level": "coarse"}, "box": {}}},
                               {"id": "b", "physical": {"height": {"value": 9., "u": 1., "level": "coarse", "scale": "estimated"},
                                                        "width": {"value": 1.}, "depth": {"status": "not observed"},
                                                        "size_check": {"status": "implausible"}}}]})
    assert cc["fields"] == 6 and cc["violations"] == 2 and cc["implausible_cards_with_numbers"] == 1, cc  # b: width without u/level/scale, depth without reason; numbers on an implausible size
    assert size_class("tool box")[0] == "box" and size_class("exit sign")[0] == "exit sign" and size_class("shelf label")[0] == "sign"
    assert size_class("pallet of goods")[0] == "stacked boxes" and size_class("control panel")[0] == "other" and size_class("shelves")[3]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", type=Path)
    p.add_argument("--site", choices=sorted(CLIPS))
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--e9", type=Path, help="evaluate every call of an E9 run folder instead of a mirrored report")
    p.add_argument("--gaps", type=Path, help="measure the review's gaps into this new folder")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--mvp", type=Path, help="spec 8 rows into this folder; default runs: fb/integrate's warm calls (D1's baseline)")
    p.add_argument("--runs", default="", help="site=RUN_DIR:REPORT,... (default: FB_RUNS)")
    p.add_argument("--repeat", default="", help="site=RUN_DIR:REPORT:OFFSET:NAME,... compared with --runs (default: fb's first calls, offset 0)")
    p.add_argument("--labels", type=Path, help="folder with labels-clicks-<site>.json (the agent's audit)")
    p.add_argument("--decisions", type=Path, help="X8-shaped set-d decisions (default: X8 run 001, pass 2)")
    p.add_argument("--write-calibration", action="store_true", help="also write fast_report/calibration.json")
    p.add_argument("--no-gpu", dest="gpu_refs", action="store_false", help="never start Modal (clicks need refs-<site>.npz in the folder)")
    a = p.parse_args()
    if a.self_check:
        self_check()
    elif a.mvp:
        R = PHASE2 / "runs"
        runs = {k: (R / d, r) for k, (d, r) in FB_RUNS.items()}
        repeats = {"me340": [(R / "fb-integrate-me340-006", "fb-me340-e84efffd-1790649664", 0, "first call, same window")],
                   "samsclub-a2": [(R / "fb-integrate-samsclub-002", "fb-samsclub-a2-d5e0c855-1790650409", 0, "first call, same window")],
                   "walmart": [(R / "fb-integrate-walmart-001", "fb-walmart-c0761a2a-1790651034", 0, "first call, same window")]}
        if a.runs:
            runs = {s: (Path(v.rsplit(":", 1)[0]), v.rsplit(":", 1)[1]) for s, v in (x.split("=", 1) for x in a.runs.split(","))}
            repeats = {}
        if a.repeat:
            for s, v in (x.split("=", 1) for x in a.repeat.split(",")):
                d, r, off, name = v.rsplit(":", 3)
                repeats.setdefault(s, []).append((Path(d), r, int(off), name))
        mvp(a.mvp, runs, repeats, a.gpu_refs, a.labels, a.decisions, a.write_calibration)
        print((a.mvp / "summary.md").read_text())
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
