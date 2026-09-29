"""X13 hazards: hazard judgement without a run-time VLM. Rules decide first; a local decider (Jev-Omni, a service on its own GPU)
answers only the visual questions whose answer can still change a verdict; a look-alike label bank (SigLIP 2 kNN, no training)
reuses answers; what stays unsure is what would still need a VLM. Gemini's recorded answers and the agent labels are offline
label sources only: nothing here calls Gemini.

  python scripts/x13_hazards.py replay --work W      # the judge per cards version of runs 007 (both calls, 3 sites): the questions it
                                                     # sends, whether an answer can change a verdict, 2 x 2 evidence (768 px)
  python scripts/x13_hazards.py items --work W       # labelled items (object evidence, X8 set d frames, factory frames) + run items
  modal run modal_apps/x13_jev_service.py --work W   # Jev-Omni + SigLIP 2 as an HTTPS service on one A100 -> W/answers.json
  python scripts/x13_hazards.py score --work W --out RUNS/x13-hazards-NNN
  python scripts/x13_hazards.py --self-check
"""
import argparse
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
RUNS = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs")
CALLS = {"me340": ("mvp2-integrate-me340-007", ["mvp-me340-e84efffd-1790690525", "mvp-me340-e84efffd-1790690775"]),
         "samsclub": ("mvp2-integrate-samsclub-a2-007", ["mvp-samsclub-a2-d5e0c855-1790691480", "mvp-samsclub-a2-d5e0c855-1790691934"]),
         "walmart": ("mvp2-integrate-walmart-007", ["mvp-walmart-c0761a2a-1790692313", "mvp-walmart-c0761a2a-1790692552"])}
PASS, FAIL, REVIEW, NO_DATA = "PASS", "FAIL", "NEEDS_REVIEW", "NO_DATA"


# ---------- which answers can change a verdict (judge.combine, read backwards) ----------

def effect(row):
    """-> what a picture answer can do to this row under judge.combine:
      'veto'        geometry PASS: only 'hazard'/'likely' changes it (-> NEEDS_REVIEW); the safety net against a false PASS
      'keep-pass'   geometry PASS on one seen face (J2): only 'clear' keeps the PASS
      'unfail'      geometry FAIL: only 'clear' changes it (-> NEEDS_REVIEW)
      'make-fail'   geometry NEEDS_REVIEW with the measured value past the threshold (hazard_side, not J3a): 'hazard' -> FAIL
      'hint'        no geometry / NO_DATA: 'hazard'/'likely' -> NEEDS_REVIEW
      'none'        geometry NEEDS_REVIEW otherwise: no answer changes the verdict (the picture is shown, never decides)"""
    g = row.get("geometry") or {}
    r = g.get("result")
    if r == PASS:
        return "keep-pass" if g.get("needs_clear_picture") else "veto"
    if r == FAIL:
        return "unfail"
    if r == REVIEW:
        return "make-fail" if g.get("hazard_side") and row["check"] != "J3a" else "none"
    return "hint"


RANK = {"none": 0, "hint": 1, "veto": 2, "keep-pass": 3, "unfail": 3, "make-fail": 4}


# ---------- replay: the judge per cards version ----------

def patches(run, report):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(RUNS / run / "mirror/reports" / report / "patches/*.json")))]


def replay(site, work):
    """Every cards version of both calls: rows (judge.evaluate on that version's cards with that time's outlines and the final
    cameras, people and room), the (object, question) pairs the judge wants, which of them are new (the pipeline sends only those:
    judge.send skips carried answers), each question's strongest effect, and its evidence image (hazard.evidence, 768 px)."""
    import judge_offline as jo
    from fast_report import hazard, judge
    run, reports = CALLS[site]
    out = []
    (work / "images").mkdir(parents=True, exist_ok=True)
    for report in reports:
        d = jo.load(run=run, report=report)
        ps = patches(run, report)
        outl = [p for p in ps if p["layer"] == "outlines"]
        sent, ev_done = set(), {}
        for p in (p for p in ps if p["layer"] == "object_cards"):
            cards = json.loads(jo.blob(RUNS / run, p["blobs"]["cards"]["sha256"]).read_text()) if p["data"]["cards"] == "blob" else p["data"]["cards"]
            o = [x for x in outl if x["seq"] < p["seq"]][-1]
            frames_out = json.loads(jo.blob(RUNS / run, o["blobs"]["analysis"]["sha256"]).read_text())["frames"] if o["data"].get("analysis") == "blob" else o["data"]["frames"]
            ctx = {**d["ctx"], "outlines": frames_out, "outlines_by_frame": {f["sourceFrame"]: f["objects"] for f in frames_out},
                   "shots": [{k: v for k, v in s.items() if not k.startswith("_")} for s in d["ctx"]["shots"]]}  # judge caches per shot
            rows = judge.evaluate(cards, ctx)
            by = {c["id"]: c for c in (judge.follow_name(c) for c in cards)}
            want = judge.wanted(rows, by)
            for oid, qs in want.items():
                for q in qs:
                    rs = [r for r in rows if r["subject"] == oid and hazard.CHECK_Q.get(r["check"]) == q]
                    eff = max((effect(r) for r in rs), key=RANK.get)
                    top = max(rs, key=lambda r: RANK[effect(r)]).get("geometry") or {}
                    if oid not in ev_done:
                        jpg, keys = hazard.evidence(by[oid], ctx, judge.frame_at, judge.marks_on, judge.som)
                        name = f"{report}-{oid}.jpg".replace(":", "_")
                        if jpg is not None:
                            (work / "images" / name).write_bytes(jpg)
                        ev_done[oid] = (f"images/{name}" if jpg is not None else None, keys)
                    ident = by[oid].get("identity") or {}
                    pos = judge.fact(by[oid], "position_xy")
                    out.append({"site": site, "report": report, "call": "first" if report == reports[0] else "warm", "version": p["version"],
                                "id": oid, "q": q, "name": judge.name_of(by[oid]), "canonical": ident.get("canonical"),
                                "checks": [r["check"] for r in rs], "geometry": [(r.get("geometry") or {}).get("result") for r in rs],
                                "effect": eff, "effects": [effect(r) for r in rs], "new": (oid, q) not in sent,
                                "image": ev_done[oid][0], "keys": ev_done[oid][1], "shot": by[oid].get("shot"),
                                "xy": None if pos is None else [round(float(v), 3) for v in pos["value"]],
                                "g": {k: top.get(k) for k in ("quantity", "value", "u", "threshold", "direction")}})
                    sent.add((oid, q))
            print(site, report, "cards v", p["version"], "rows", len(rows), "questions", sum(map(len, want.values())),
                  "new", sum(1 for x in out if x["report"] == report and x["version"] == p["version"] and x["new"]), flush=True)
    return out


def recorded(site):
    """What the runs themselves recorded: the final judgements rows (verdict, geometry, the Gemini/Qwen answer) per report."""
    run, reports = CALLS[site]
    rec = {}
    for report in reports:
        js = [p for p in patches(run, report) if p["layer"] == "judgements"]
        rec[report] = js[-1]["data"]["rows"]
    return rec


# ---------- labelled items ----------

DECISIVE = []  # extra label files (work/labels-decisive.jsonl: the items whose Gemini answer changed a verdict in runs 007)


def load_labels():
    """(site, report or None, id, q) -> label, from the agent audits (hazard-001's round-1 objects; run 007's audit; the decisive items
    labelled here)."""
    lab = {}
    for f in DECISIVE:
        for line in open(f):
            r = json.loads(line)
            lab[(r["site"], r["report"], r["id"], r["q"])] = r["label"]
    for line in open(RUNS / "mvp2-judge-hazard-001/labels-agent.jsonl"):
        r = json.loads(line)
        if r["q"].startswith("q") and r["label"] in (0, 1):
            lab[(r["site"], None, r["id"], r["q"])] = r["label"]
    for f in glob.glob(str(RUNS / "mvp2-results/judge/*/labels-run.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["q"].startswith("q") and r["label"] in (0, 1):
                lab[(r["site"], r["report"], r["id"], r["q"])] = r["label"]
    return lab


def object_items(lab):
    """hazard-002's evidence (v2, 768 px; the round-1 objects) with hazard-001's agent labels and Gemini v2's stated p."""
    base = RUNS / "mvp2-judge-hazard-002"
    out = []
    for site in ("me340", "samsclub", "walmart"):
        ans = json.loads((base / f"answers-{site}.json").read_text())["answers"]
        for it in json.loads((base / f"items-{site}.json").read_text()):
            for q in it["questions"]:
                y = lab.get((site, None, it["id"], q))
                if y is None:
                    continue
                a = ans.get(f"{it['id']}|{q}") or {}
                out.append({"key": f"obj|{site}|{it['id']}|{q}", "form": "object", "video": site, "q": q, "truth": y, "name": it["name"],
                            "image": str(base / it["image"]), "gemini_p": a.get("p")})
    return out


def frame_items():
    """X8 set d: whole frames, 8 clips (the factory clip is 'lightning'), agent-labelled, Gemini 3.5 Flash's stated p."""
    base = RUNS / "fx-x8-jev-001"
    d = json.loads((base / "sets/d.json").read_text())
    gem = json.loads((base / "gemini-d.json").read_text())
    gp = gemini_d(gem)
    return [{"key": f"frame|{x['source']}|{x['id']}|{x['question_id']}", "form": "frame", "video": x["source"], "q": x["question_id"],
             "truth": int(x["truth"]), "clear": x.get("clear"), "question": x["question"], "image": str(base / "sets/crops" / f"{x['crop']}.jpg"),
             "gemini_p": gp.get(x["id"])} for x in d["items"]]


def gemini_d(gem):
    """gemini-d.json -> {item id: stated p(yes)} (the file's own shape, whichever of its two layouts)."""
    out = {}
    items = (gem.get("items") or gem.get("answers")) if isinstance(gem, dict) else gem
    if isinstance(items, dict):
        items = [{"id": k, **v} for k, v in items.items()]
    for x in items or []:
        p = x.get("p_yes", x.get("p"))
        if x.get("id") is not None and isinstance(p, (int, float)):
            out[x["id"]] = float(p)
    return out


FACTORY = RUNS / "lightning-outlines-bc836cad93/out/analysis.json"  # the factory clip's object map re-projected into every frame
FACTORY_PNG = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2/data/clips/lightning-3585/rgb")
FLOOR_WORDS = ("chair", "cart", "box", "crate", "bin", "trash can", "cabinet", "shelf", "column", "wooden board", "container",
               "checkout counter", "bottle", "tool tray")  # ponytail: no floor geometry for this clip; the label says 'on floor'


class Pngs:
    """ctx['frames'] for the factory clip: source frame k = the k-th PNG (640 x 480, the outlines' space)."""

    def __init__(self, folder):
        self.files = sorted(Path(folder).glob("*.png"))

    def __getitem__(self, k):
        import cv2
        return cv2.imread(str(self.files[int(k)]))


def factory_candidates():
    """The factory clip's objects and the question the judge would ask by name (judge.applicable -> hazard.CHECK_Q): q1 for a
    deformable class, q4 for a floor-standing class (J5), q2 for boxes and crates (their stacks). -> [(entity, name, [q])]."""
    from fast_report import cards as A, judge
    d = json.loads(FACTORY.read_text())
    names = {}
    for f in d["frames"]:
        for o in f["objects"]:
            names[o["entityId"]] = o["label"]
    out = []
    for e, name in sorted(names.items()):
        canon = A.canonical(name)
        card = {"kind": "object", "identity": {"name": name, "canonical": canon if isinstance(canon, str) and canon != A.NOT_OBJECT else None}}
        cls, qs = judge.class_of(card), []
        if judge.mobility(card) == "deformable":
            qs.append("q1")
        if "J5" in judge.applicable(card) and judge.is_a(cls, FLOOR_WORDS):
            qs.append("q4")
        if judge.is_a(cls, ("box", "crate")):
            qs.append("q2")
        if qs:
            out.append((e, name, qs))
    return out


def factory_render(work):
    """2 x 2 evidence (hazard.evidence, v2) for each factory candidate -> [{id, name, questions, image}]."""
    from fast_report import hazard, judge
    d = json.loads(FACTORY.read_text())
    ctx = {"outlines_by_frame": {f["sourceFrame"]: f["objects"] for f in d["frames"]}, "frames": Pngs(FACTORY_PNG), "source_wh": (640, 480)}
    (work / "images").mkdir(parents=True, exist_ok=True)
    out = []
    for e, name, qs in factory_candidates():
        jpg, keys = hazard.evidence({"id": e, "kind": "object", "views": {"best": []}}, ctx, judge.frame_at, judge.marks_on, judge.som)
        if jpg is None:
            continue
        (work / "images" / f"factory-{e}.jpg").write_bytes(jpg)
        out.append({"id": e, "name": name, "questions": qs, "image": f"images/factory-{e}.jpg", "keys": keys})
    return out


def sheet(paths, captions, out, cols=3, side=420):
    """A contact sheet for labelling by looking: each image fitted into side x side, its caption above it."""
    import cv2
    rows = math.ceil(len(paths) / cols)
    img = np.full((rows * (side + 28), cols * side, 3), 255, np.uint8)
    for i, (p, c) in enumerate(zip(paths, captions)):
        im = cv2.imread(str(p))
        s = side / max(im.shape[:2])
        im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        y, x = (i // cols) * (side + 28), (i % cols) * side
        img[y + 28:y + 28 + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(img, c[:52], (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


FACTORY_MP4 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2/data/clips/lightning-3585/source-rgb.mp4")
SETD_LIGHTNING = (30, 90, 150, 210, 270, 330, 360, 390, 420, 450, 509, 540, 569, 600, 629, 660, 689, 749)
NEW_LIGHTNING = (0, 60, 120, 180, 240, 300, 480, 720, 770)  # the factory clip's other 1-2 s steps, not in set d
SPILL = "Is there a spill (liquid, oil, powder or loose small items) on the floor where people walk?"
D_QUESTIONS = {"q1": "Is there a cable or hose lying across the floor where people walk?",  # X8 set d's wording (x8_sets.D_QUESTIONS)
               "q2": "Are boxes or goods stacked unstably, so that they could fall?",
               "q3": "Is a person standing on a shelf, rack or pallet?",
               "q4": "Is the aisle or exit route blocked or narrowed by an object standing in it?",
               "q5": "Is a person standing on a raised platform, step or box instead of the floor?", "spill": SPILL}


def factory_frames(work):
    """The factory clip's new frames, cut as X8 cut set d's (cv2 seek on source-rgb.mp4, JPEG q88) -> {frame: path}."""
    import cv2
    out = {}
    (work / "images").mkdir(parents=True, exist_ok=True)
    for f in NEW_LIGHTNING:
        p = work / "images" / f"frame-lightning-{f}.jpg"
        if not p.exists():
            cap = cv2.VideoCapture(str(FACTORY_MP4))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            assert ok, f
            cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out[f] = p
    return out


def frame_labels(work):
    """Agent labels made here by looking (work/labels-frames.jsonl: {video, frame, q, label, clear, note})."""
    p = work / "labels-frames.jsonl"
    return {(r["video"], str(r["frame"]), r["q"]): r for r in map(json.loads, p.read_text().splitlines()) if r.get("label") in (0, 1)} if p.exists() else {}


def build_items(work):
    """Every item: labelled object items (hazard-002), set d frames (+ spill on every frame, + the whole question matrix on the
    factory clip's 27 frames), run items (the replay's questions, labelled where run 007's audit labelled them)."""
    lab = load_labels()
    items = object_items(lab)
    fr = frame_items()
    have = {(x["video"], x["image"], x["q"]) for x in fr}
    frames = {}
    for x in fr:
        frames.setdefault((x["video"], x["image"]), x["key"].split("|")[2])
    fl = frame_labels(work)
    extra = []
    for f, path in factory_frames(work).items():
        frames[("lightning", str(path))] = f"new-{f}"
    for (video, image), fid in sorted(frames.items()):
        frame = Path(image).stem.rsplit("-", 1)[-1]
        qs = ["spill"] + (["q1", "q2", "q3", "q4", "q5"] if video == "lightning" else [])
        for q in qs:
            if (video, image, q) in have:
                continue
            r = fl.get((video, frame, q))
            extra.append({"key": f"frame|{video}|{fid}|{q}", "form": "frame", "video": video, "q": q, "truth": None if r is None else r["label"],
                          "clear": None if r is None else r.get("clear"), "question": D_QUESTIONS[q], "image": image, "gemini_p": None,
                          "frame": frame, "agent_labelled_here": True})
    items += fr + extra
    for site in CALLS:
        rep = json.loads((work / f"replay-{site}.json").read_text())
        seen = set()
        for r in rep:
            k = (r["report"], r["id"], r["q"])
            if k in seen or r["image"] is None:
                continue
            seen.add(k)
            items.append({"key": f"run|{site}|{r['report']}|{r['id']}|{r['q']}", "form": "run", "video": site, "q": r["q"],
                          "truth": lab.get((site, r["report"], r["id"], r["q"])), "name": r["name"], "canonical": r["canonical"],
                          "image": str(work / r["image"]), "call": r["call"]})
    return items


# ---------- metrics ----------

def auroc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if not len(p):
        return None
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() for b in range(bins) if (idx == b).any()) / len(p))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_temperature(z, y):
    """One temperature T for p = sigmoid(z / T) by minimising the log-loss (a 1-D golden-section search over log T)."""
    z, y = np.asarray(z, float), np.asarray(y, float)

    def nll(lt):
        p = 1 / (1 + np.exp(-np.clip(z / math.exp(lt), -50, 50)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    a, b = math.log(.05), math.log(50.)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(80):
        if nll(c) < nll(d):
            b, d = d, c
            c = b - g * (b - a)
        else:
            a, c = c, d
            d = a + g * (b - a)
    return math.exp((a + b) / 2)


def cuts(p, y, precision=.8, npv=.97, min_pos=3, min_hits=3, min_clear=10):
    """hazard.thresholds' rule on calibrated p: the lowest t with precision(p >= t) >= `precision` on >= min_hits calls, the highest
    t with NPV(p <= t) >= `npv` on >= min_clear calls (below the hazard cut); None with fewer than min_pos positives."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    if y.sum() < min_pos:
        return None, None
    cand = np.unique(p)
    hz = [t for t in cand if (p >= t).sum() >= min_hits and y[p >= t].mean() >= precision]
    th = min(hz) if hz else None
    cl = [t for t in cand if (p <= t).sum() >= min_clear and 1 - y[p <= t].mean() >= npv and (th is None or t < th)]
    return th, (max(cl) if cl else None)


def wilson(k, n, z=1.96):
    if not n:
        return None
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [round(max(0., c - h), 3), round(min(1., c + h), 3)]


def knn_prior(emb, train_idx, test_idx, y, k=10):
    """Similarity-weighted share of positives among the k nearest labelled look-alikes (cosine on unit vectors) -> (prior, top-1 cos)."""
    if not len(train_idx):
        return np.full(len(test_idx), np.nan), np.full(len(test_idx), np.nan)
    E = emb[test_idx] @ emb[train_idx].T
    kk = min(k, len(train_idx))
    nn = np.argsort(-E, 1)[:, :kk]
    s = np.take_along_axis(E, nn, 1)
    w = np.exp((s - s[:, :1]) / .02)  # ponytail: a fixed 0.02 softness; neighbours well below the top one count little
    yy = np.asarray(y)[np.asarray(train_idx)][nn]
    return (w * yy).sum(1) / w.sum(1), s[:, 0]


# ---------- scoring ----------

PRIMARY = {"object": "grid", "run": "grid", "frame": "frame"}  # declared before any answer was read: the grid is what Gemini sees
GROUP = {"object": "obj", "run": "obj", "frame": "frame"}
REUSE_T = (.85, .90, .95)
WIDE_M = 1.  # a J5 PASS whose free width minus u clears 0.711 m by >= 1 m needs no picture (policy '+ wide J5 margin')
SAME_M = .3  # a card centre within 0.3 m of an earlier version's card, same shot: the same object (densify re-ids)
CASCADE_T = .95  # a look-alike reuses an answer at cosine >= 0.95 (declared, not fitted; the sensitivity is in 'reuse')


def gemini_run(site):
    """(report, object, question) -> Gemini's stated p(yes) on the run's own items (every judgements patch of both calls)."""
    run, reports = CALLS[site]
    out = {}
    for report in reports:
        for p in patches(run, report):
            if p["layer"] == "judgements":
                for r in p["data"]["rows"]:
                    v = r.get("vlm") or {}
                    if str(v.get("decider", "")).startswith("gemini") and v.get("p_yes") is not None:
                        out[(report, r["subject"], v["question"])] = float(v["p_yes"])
    return out


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def fold(train, test):
    """Temperature scaling and the cuts fitted on `train` (other videos), applied to `test`. -> (T, hazard cut, clear cut, calibrated
    p of test)."""
    y = np.array([x["truth"] for x in train], int)
    z = logit([x["p"] for x in train])
    T = fit_temperature(z, y) if 0 < y.sum() < len(y) else 1.
    th, tc = cuts(sigmoid(z / T), y)
    return T, th, tc, sigmoid(logit([x["p"] for x in test]) / T)


def fit_validated(train, precision=.8, npv=.97):
    """fold()'s cuts, each kept only where it held on held-out videos inside `train` (leave one of its videos out, refit, apply):
    the hazard cut when the inner held-out precision >= `precision` on >= 3 calls, the clear cut when the inner held-out NPV >= `npv`
    on >= 10 calls. -> (T, hazard cut or None, clear cut or None, inner record)."""
    T, th, tc, _ = fold(train, [{"p": .5}])
    inner = {"hazard_calls": 0, "hazard_right": 0, "clear_calls": 0, "clear_right": 0}
    for w in sorted({x["video"] for x in train}):
        tr, te = [x for x in train if x["video"] != w], [x for x in train if x["video"] == w]
        _, ith, itc, pc = fold(tr, te)
        for x, p_ in zip(te, pc):
            if ith is not None and p_ >= ith:
                inner["hazard_calls"] += 1
                inner["hazard_right"] += x["truth"]
            elif itc is not None and p_ <= itc:
                inner["clear_calls"] += 1
                inner["clear_right"] += 1 - x["truth"]
    ok_h = inner["hazard_calls"] >= 3 and inner["hazard_right"] / inner["hazard_calls"] >= precision
    ok_c = inner["clear_calls"] >= 10 and inner["clear_right"] / inner["clear_calls"] >= npv
    return T, th if ok_h else None, tc if ok_c else None, {**inner, "hazard_kept": bool(ok_h and th is not None), "clear_kept": bool(ok_c and tc is not None)}


def call(p, th, tc, veto=.8):
    """A calibrated p against the cuts -> hazard | likely (the uncalibrated veto, only stops a PASS) | clear | unsure (hazard.verdict)."""
    return "hazard" if th is not None and p >= th else "likely" if p >= (th if th is not None else veto) else "clear" if tc is not None and p <= tc else "unsure"


def confusion(xs, key="call"):
    pos = [x for x in xs if x["truth"] == 1]
    hz = [x for x in xs if x[key] == "hazard"]
    cl = [x for x in xs if x[key] == "clear"]
    out = {"n": len(xs), "positives": len(pos), "hazard_calls": len(hz), "hazard_right": sum(x["truth"] for x in hz),
           "clear_calls": len(cl), "clear_right": sum(1 - x["truth"] for x in cl), "likely_calls": sum(x[key] == "likely" for x in xs),
           "unsure": sum(x[key] == "unsure" for x in xs), "positives_called_clear": sum(x[key] == "clear" for x in pos),
           "positives_called_hazard_or_likely": sum(x[key] in ("hazard", "likely") for x in pos),
           "negatives_called_hazard_or_likely": sum(x[key] in ("hazard", "likely") for x in xs if x["truth"] == 0)}
    out["precision"] = round(out["hazard_right"] / out["hazard_calls"], 3) if out["hazard_calls"] else None
    out["precision_ci95"] = wilson(out["hazard_right"], out["hazard_calls"])
    out["recall"] = round(out["hazard_right"] / out["positives"], 3) if out["positives"] else None
    out["recall_ci95"] = wilson(out["hazard_right"], out["positives"])
    out["npv"] = round(out["clear_right"] / out["clear_calls"], 3) if out["clear_calls"] else None
    out["npv_ci95"] = wilson(out["clear_right"], out["clear_calls"])
    out["decided_share"] = round((out["hazard_calls"] + out["clear_calls"]) / len(xs), 3) if xs else None
    out["false_clear_rate_on_positives"] = round(out["positives_called_clear"] / out["positives"], 3) if out["positives"] else None
    out["false_alarm_rate"] = round(out["negatives_called_hazard_or_likely"] / (len(xs) - len(pos)), 3) if len(xs) > len(pos) else None
    return out


def trust(n_pos, n_neg):
    """How far a number on these items can be trusted (stated with every question's result)."""
    if n_pos == 0:
        return "no positives: only the false-alarm rate on negatives is measured; precision, recall and AUROC are undefined"
    if n_pos < 10:
        return f"only {n_pos} positives: recall and precision move by >= {100 / n_pos:.0f} points per item; read them as anecdotes"
    return f"{n_pos} positives / {n_neg} negatives"


def jev_heldout(items, answers):
    """Jev-Omni zero-shot, leave one video out per question: temperature scaling and the cuts (hazard.thresholds' rule: precision >=
    0.8 on >= 3 calls, NPV >= 0.97 on >= 10) fitted on every other video's labelled items of that question (all forms pooled, as the
    production Gemini/Qwen cuts were), applied to the held-out video. -> (per question record, labelled items with p, p_cal, call)."""
    lab = []
    for x in items:
        a = answers.get(f"{x['key']}|{PRIMARY[x['form']]}")
        if x["truth"] in (0, 1) and a is not None:
            lab.append({**x, "p": float(a[0]), "p_tiles": (answers.get(f"{x['key']}|tiles") or [None])[0]})
    out = {}
    for q in sorted({x["q"] for x in lab}):
        xs = [x for x in lab if x["q"] == q]
        folds = {}
        for v in sorted({x["video"] for x in xs}):
            train, test = [x for x in xs if x["video"] != v], [x for x in xs if x["video"] == v]
            T, th, tc, pc = fold(train, test)
            _, vth, vtc, inner = fit_validated(train)
            folds[v] = {"T": round(T, 3), "hazard_cut": None if th is None else round(float(th), 4), "clear_cut": None if tc is None else round(float(tc), 4),
                        "train_positives": sum(x["truth"] for x in train), "validated_on_other_videos": inner}
            for x, p_ in zip(test, pc):
                x["p_cal"], x["call"], x["vcall"] = float(p_), call(float(p_), th, tc), call(float(p_), vth, vtc)
        y = [x["truth"] for x in xs]
        rec = {"n": len(xs), "positives": sum(y), "trust": trust(sum(y), len(y) - sum(y)),
               "auroc_raw": auroc([x["p"] for x in xs], y), "ece_raw": ece([x["p"] for x in xs], y), "ece_heldout_calibrated": ece([x["p_cal"] for x in xs], y),
               "heldout": confusion(xs), "heldout_validated_cuts": confusion(xs, "vcall"), "folds": folds, "by_video": {}, "by_form": {}}
        for k, key in (("by_video", "video"), ("by_form", "form")):
            for v in sorted({x[key] for x in xs}):
                t = [x for x in xs if x[key] == v]
                rec[k][v] = {**confusion(t), "validated": confusion(t, "vcall"), "auroc_raw": auroc([x["p"] for x in t], [x["truth"] for x in t]),
                             "gemini_auroc_same_items": auroc([x["gemini_p"] for x in t if x.get("gemini_p") is not None], [x["truth"] for x in t if x.get("gemini_p") is not None])}
        neg = [x for x in xs if x["truth"] == 0]
        gneg = [x for x in neg if x.get("gemini_p") is not None]
        rec["negatives"] = {"n": len(neg), "jev_p_yes_ge_0.5": sum(x["p"] >= .5 for x in neg), "jev_calibrated_ge_0.8": sum(x["p_cal"] >= .8 for x in neg),
                            "gemini_n": len(gneg), "gemini_p_yes_ge_0.5": sum(x["gemini_p"] >= .5 for x in gneg),
                            "jev_p_yes_median": pct([x["p"] for x in neg], 50), "jev_p_yes_p99": pct([x["p"] for x in neg], 99)}
        tl = [x for x in xs if x["p_tiles"] is not None]
        rec["auroc_tiles_variant"] = auroc([x["p_tiles"] for x in tl], [x["truth"] for x in tl]) if tl else None
        rec["auroc_grid_same_items"] = auroc([x["p"] for x in tl], [x["truth"] for x in tl]) if tl else None
        g = [x for x in xs if x.get("gemini_p") is not None]
        if g:
            gy = [x["truth"] for x in g]
            for x in g:
                x["gcall"] = "hazard" if x["gemini_p"] >= .5 else "clear" if x["gemini_p"] <= .1 else "unsure"  # production's stated cuts
            rec["gemini_same_items"] = {"n": len(g), "positives": sum(gy), "auroc": auroc([x["gemini_p"] for x in g], gy), "jev_auroc": auroc([x["p"] for x in g], gy),
                                        "stated_cuts_0.5_0.1": confusion(g, "gcall"), "jev_heldout_same_items": confusion(g)}
        out[q] = rec
    return out, lab


def vectors(emb):
    """answers.json's float16 embeddings -> {image|view: unit vector}."""
    import base64
    return {k: np.frombuffer(base64.b64decode(v), np.float16).astype(np.float32) for k, v in emb.items()}


def key_vec(x, V, view="both"):
    if x["form"] == "frame":
        return V.get(f"{x['image']}|frame")
    a, b = V.get(f"{x['image']}|plain"), V.get(f"{x['image']}|context")
    if a is None or b is None:
        return None
    v = a if view == "plain" else b if view == "context" else a + b
    return v / np.linalg.norm(v)


def retrieval(lab, V):
    """(b) the k nearest labelled look-alikes from other videos (same question, same form group) as a prior, alone and joined to
    Jev's held-out calibrated p (sum of logits minus the train base rate's); (c) the nearest look-alike's label reused at a cosine
    cut, cross-video. Per question and key view."""
    out = {}
    for q in sorted({x["q"] for x in lab}):
        for g in ("obj", "frame"):
            xs = [x for x in lab if x["q"] == q and GROUP[x["form"]] == g]
            if len({x["video"] for x in xs}) < 2:
                continue
            for view in (("both", "plain", "context") if g == "obj" else ("frame",)):
                vs = [key_vec(x, V, view) for x in xs]
                keep = [i for i, v in enumerate(vs) if v is not None]
                E = np.stack([vs[i] for i in keep])
                X = [xs[i] for i in keep]
                y = np.array([x["truth"] for x in X])
                prior, top, lab_nn = np.full(len(X), np.nan), np.full(len(X), np.nan), np.full(len(X), -1)
                base = np.zeros(len(X))
                for v in sorted({x["video"] for x in X}):
                    te = [i for i, x in enumerate(X) if x["video"] == v]
                    tr = [i for i, x in enumerate(X) if x["video"] != v]
                    pr, tp = knn_prior(E, tr, te, y)
                    prior[te], top[te] = pr, tp
                    nn = np.argmax(E[te] @ E[tr].T, 1)
                    lab_nn[te] = y[np.array(tr)[nn]]
                    base[te] = y[tr].mean()
                pc = np.array([x["p_cal"] for x in X])
                sm = np.clip(prior, .02, .98)
                comb = logit(pc) + logit(sm) - logit(np.clip(base, .02, .98))
                rec = {"n": len(X), "positives": int(y.sum()), "auroc_knn_prior": auroc(prior, y), "auroc_jev": auroc(pc, y),
                       "auroc_jev_plus_prior": auroc(comb, y), "top1_cos_median": round(float(np.median(top)), 3), "reuse": {}}
                for t in REUSE_T:
                    c = top >= t
                    rec["reuse"][str(t)] = {"covered": int(c.sum()), "covered_share": round(float(c.mean()), 3), "right": int((lab_nn[c] == y[c]).sum()),
                                            "positives_covered": int((c & (y == 1)).sum()), "positives_reused_as_hazard": int((c & (y == 1) & (lab_nn == 1)).sum())}
                out[f"{q}|{g}|{view}"] = rec
    return out


def within_video(work, items, V):
    """(c) within a call: each question the judge sent, against the nearest question of the same kind sent earlier in the same call
    (densify re-asks the same things under new ids; retail repeats look-alike stacks). At each cosine cut: how many would reuse
    an answer, and whether the reused answer is the one the item got itself (Gemini's recorded answer at production's cuts; the
    agent label where both are labelled)."""
    from fast_report import hazard, judge
    cal = judge.load_calibration()
    by_key = {x["key"]: x for x in items}
    lab = load_labels()
    out = {}
    for site in CALLS:
        rep = json.loads((work / f"replay-{site}.json").read_text())
        grec = gemini_run(site)
        for c in ("first", "warm"):
            seq = []
            for r in (r for r in rep if r["call"] == c and r["new"]):
                x = by_key.get(f"run|{site}|{r['report']}|{r['id']}|{r['q']}")
                v = key_vec(x, V) if x else None
                if v is None:
                    continue
                gp = grec.get((r["report"], r["id"], r["q"]))
                seq.append({"q": r["q"], "v": v, "g": hazard.verdict(gp, judge.hazard_cut(cal, "gemini", r["q"])) if gp is not None else None,
                            "y": lab.get((site, r["report"], r["id"], r["q"])), "shot": r.get("shot"), "xy": r.get("xy"), "version": r["version"]})
            rec = {"questions": len(seq)}
            n = same_g = both_g = 0  # the same physical object re-asked under a new id (densify): same shot, centre within SAME_M
            for i, a in enumerate(seq):
                prev = [b for b in seq[:i] if b["q"] == a["q"] and b["shot"] == a["shot"] and a["xy"] and b["xy"] and b["version"] < a["version"]
                        and math.dist(a["xy"], b["xy"]) <= SAME_M]
                if prev:
                    n += 1
                    b = min(prev, key=lambda b: math.dist(a["xy"], b["xy"]))
                    if a["g"] and b["g"]:
                        both_g += 1
                        same_g += a["g"] == b["g"]
            rec["same_3d_object_earlier_version"] = {"reused": n, "reused_share": round(n / len(seq), 3) if seq else None, "gemini_answer_same": same_g,
                                                     "gemini_both_answered": both_g}
            for t in REUSE_T:
                n = same_g = both_g = same_y = both_y = 0
                for i, a in enumerate(seq):
                    prev = [(float(a["v"] @ b["v"]), b) for b in seq[:i] if b["q"] == a["q"]]
                    if not prev:
                        continue
                    cos, b = max(prev, key=lambda z: z[0])
                    if cos < t:
                        continue
                    n += 1
                    if a["g"] and b["g"]:
                        both_g += 1
                        same_g += a["g"] == b["g"]
                    if a["y"] is not None and b["y"] is not None:
                        both_y += 1
                        same_y += a["y"] == b["y"]
                rec[str(t)] = {"reused": n, "reused_share": round(n / len(seq), 3) if seq else None, "gemini_answer_same": same_g, "gemini_both_answered": both_g,
                               "label_same": same_y, "both_labelled": both_y}
            out[f"{site}|{c}"] = rec
    return out


def rel(effect_, th, tc, veto=True):
    """Can an answer of this decider change the verdict? (judge.combine read backwards; a missing cut makes that answer impossible.)
    -> 'no' | 'uncalibrated veto only' | 'yes'."""
    if effect_ == "none":
        return "no"
    if effect_ in ("keep-pass", "unfail"):
        return "yes" if tc is not None else "no"
    if effect_ == "make-fail":
        return "yes" if th is not None else "no"
    return "yes" if th is not None else "uncalibrated veto only" if veto else "no"  # veto / hint


def cascade(work, items, answers, lab, V, policy="keep safety net", cuts_mode="validated", reuse="3d"):
    """(d) per call, over the questions the judge sent to Gemini (replay, in send order):
      1 rules: not asked when no answer can change the verdict, not even the VLM's at production's cuts (judge.combine read
        backwards: geometry NEEDS_REVIEW without a hazard side; a 'clear' needed but q1/q2 have no clear cut; a 'hazard' needed but
        q2 has no hazard cut).
        policy 'trust geometry' also leaves out the rows where a picture is only a safety net (geometry PASS: veto; NO_DATA: hint);
        'keep calibrated safety net' leaves out only the safety-net rows whose question has no calibrated cut for any decider (q2:
        production reads Gemini's q2 at an uncalibrated 0.8 veto; no q2 positive exists in any labelled video); '+ wide J5 margin' also
        leaves out a J5 PASS whose free width minus u clears the limit by >= WIDE_M (Gemini's three q4 vetoes in runs 007 were all at
        3.7-4.4 m, all agent-labelled not blocking);
      2 reuse: the answer given to the same object in an earlier cards version ('3d': same shot, centre within SAME_M; 'embed':
        SigLIP cos >= CASCADE_T to an earlier question; 'none');
      3 Jev-Omni on its own GPU, temperature and cuts fitted on the other videos' labelled items ('fitted'), or only the cuts that held
        on held-out videos inside those ('validated');
      4 VLM: a relevant question the local decider left unsure (for a veto or hint row with no hazard cut, the uncalibrated veto at
        p >= 0.8 still counts as an answer, as production reads Gemini's q2).
    Also lists every question where Gemini's recorded answer changed a verdict, with Jev's call and the agent label."""
    from fast_report import hazard, judge
    cal = judge.load_calibration()
    by_key = {x["key"]: x for x in items}
    labels = load_labels()
    out = {}
    for site in CALLS:
        rep = json.loads((work / f"replay-{site}.json").read_text())
        grec = gemini_run(site)
        cutsv = {}
        for q in sorted({r["q"] for r in rep}):
            train = [x for x in lab if x["q"] == q and x["video"] != site]
            T, th, tc, _ = (fit_validated(train) if cuts_mode == "validated" else fold(train, [{"p": .5}])) if train else (1., None, None, None)
            cutsv[q] = (T, th, tc)
        for callname in ("first", "warm"):
            done, rec, decisive = [], Counter(), []
            for r in (r for r in rep if r["call"] == callname and r["new"]):
                x = by_key.get(f"run|{site}|{r['report']}|{r['id']}|{r['q']}")
                gcut = judge.hazard_cut(cal, "gemini", r["q"])
                gp = grec.get((r["report"], r["id"], r["q"]))
                gv = hazard.verdict(gp, gcut) if gp is not None else "not answered"
                rec["0_gemini_sent"] += 1
                T, th, tc = cutsv[r["q"]]
                a = answers.get(f"{x['key']}|grid") if x else None
                c = call(float(sigmoid(logit([a[0]]) / T)[0]), th, tc) if a else None
                g_rel = rel(r["effect"], (gcut.get("hazard") or {}).get("t"), (gcut.get("clear") or {}).get("t"))
                g_changes = (r["effect"] in ("veto", "hint") and gv in ("hazard", "likely")) or (r["effect"] in ("unfail", "keep-pass") and gv == "clear") \
                    or (r["effect"] == "make-fail" and gv == "hazard")
                if g_changes:
                    decisive.append({"id": r["id"], "q": r["q"], "effect": r["effect"], "gemini": gv, "gemini_p": gp, "jev": c,
                                     "label": labels.get((site, r["report"], r["id"], r["q"]))})
                rec[f"gemini_answer_can_change_verdict: {g_rel}"] += 1
                g = r.get("g") or {}
                wide = r["effect"] == "veto" and g.get("value") is not None and g.get("direction") == "min" and \
                    g["value"] - g["u"] - g["threshold"] >= WIDE_M  # J5: the free width's lower bound clears the limit by >= WIDE_M
                if g_rel == "no" or (policy == "trust geometry" and r["effect"] in ("veto", "hint")) or \
                        (policy.startswith("keep calibrated safety net") and g_rel == "uncalibrated veto only") or \
                        (policy.endswith("+ wide J5 margin") and wide):  # not even the VLM's answer could change it, or the policy's rule decides
                    rec["1_rules_decide (not asked)"] += 1
                    continue
                v = key_vec(x, V) if x else None
                me = {"q": r["q"], "shot": r.get("shot"), "xy": r.get("xy"), "version": r["version"], "v": v}
                hit = False
                if reuse == "3d" and me["xy"]:
                    hit = any(d["q"] == me["q"] and d["shot"] == me["shot"] and d["xy"] and d["version"] < me["version"]
                              and math.dist(d["xy"], me["xy"]) <= SAME_M for d in done)
                elif reuse == "embed" and v is not None:
                    hit = any(d["q"] == me["q"] and d["v"] is not None and float(v @ d["v"]) >= CASCADE_T for d in done)
                done.append(me)
                if hit:
                    rec["2_reused (same object, earlier answer)"] += 1
                    continue
                if c is None:
                    rec["3_no_local_answer (no evidence image)"] += 1
                    continue
                rec["3_jev_asked"] += 1
                rec[f"3_jev_{c}"] += 1
                if rel(r["effect"], th, tc) == "no":  # the answer this row needs has no validated local cut: the VLM's question
                    rec["4_vlm_needed (no local cut for the answer this row needs)"] += 1
                    continue
                if c in ("hazard", "clear") or (c == "likely" and r["effect"] in ("veto", "hint")):
                    rec["3_jev_decided"] += 1
                    continue
                rec["4_vlm_needed"] += 1
            out[f"{site}|{callname}"] = {**dict(sorted(rec.items())), "gemini_decisive_answers": decisive}
        out[f"{site}|jev_cuts"] = {q: {"T": round(T, 3), "hazard": None if th is None else round(float(th), 4), "clear": None if tc is None else round(float(tc), 4)}
                                   for q, (T, th, tc) in cutsv.items()}
    return out


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 4) if len(xs) else None


def latency(ans):
    """The service's numbers: boot (apart), per-question round trip vs the service's own time (the hop is the difference), the
    batched pattern, embeddings, memory."""
    single = ans["single"]
    out = {"gpu": ans["stats"]["boot"].get("gpu"), "boot": ans["stats"]["boot"], "client_ready_wait_s (container start + model load, not analysis)": ans["client_ready_wait_s"],
           "memory": {k: ans["stats"][k] for k in ("nvml_peak_gib", "torch_peak_reserved_gib", "torch_peak_allocated_gib")}, "single_question_requests": {}}
    for v in ("grid", "tiles", "frame"):
        xs = [x for x in single if x["key"].endswith("|" + v)]
        if xs:
            hop = [x["rtt_s"] - x["server_s"] for x in xs]
            out["single_question_requests"][v] = {"n": len(xs), "rtt_median_s": pct([x["rtt_s"] for x in xs], 50), "rtt_p90_s": pct([x["rtt_s"] for x in xs], 90),
                                                  "server_median_s": pct([x["server_s"] for x in xs], 50), "gpu_median_s": pct([x["gpu_s"] for x in xs], 50),
                                                  "hop_median_s": pct(hop, 50), "hop_p90_s": pct(hop, 90), "request_kb_median": round(pct([x["bytes"] for x in xs], 50) / 1e3, 1)}
    loc = ans.get("local_single") or []
    if loc:
        out["single_from_this_mac"] = {"n": len(loc), "rtt_median_s": pct([x["rtt_s"] for x in loc], 50), "hop_median_s": pct([x["rtt_s"] - x["server_s"] for x in loc], 50)}
    b = ans["batched"]
    by = {}
    for x in b:
        v = x["group"].split("|")[-1]
        by.setdefault(v, []).append(x)
    out["batched_requests"] = {v: {"requests": len(xs), "questions": sum(x["n"] for x in xs), "gpu_s_per_question": round(sum(x["gpu_s"] for x in xs) / sum(x["n"] for x in xs), 4),
                                   "rtt_s_per_question": round(sum(x["rtt_s"] for x in xs) / sum(x["n"] for x in xs), 4),
                                   "hop_s_per_request_median": pct([x["rtt_s"] - x["server_s"] for x in xs], 50),
                                   "decode_s_per_question": round(sum(x["server_s"] - x["gpu_s"] for x in xs) / sum(x["n"] for x in xs), 4),
                                   "largest": max(xs, key=lambda x: x["n"])} for v, xs in by.items()}
    e = ans["embed_latency"]
    out["embeddings"] = {"images": sum(x["n"] for x in e), "server_s_per_image": round(sum(x["server_s"] for x in e) / sum(x["n"] for x in e), 4),
                         "rtt_s_per_64": pct([x["rtt_s"] for x in e if x["n"] == 64], 50)}
    return out


def recorded_changes(work, lab):
    """What Gemini's answers actually changed in runs 007's final judgements (verdict != geometry), with the audit label if any."""
    out = []
    for site in CALLS:
        for report, rows in json.loads((work / f"recorded-{site}.json").read_text()).items():
            for r in rows:
                v, g = r.get("vlm"), r.get("geometry") or {}
                if v and r["verdict"] != g.get("result"):
                    out.append({"site": site, "report": report, "id": r["subject"], "check": r["check"], "q": v["question"], "decider": v["decider"][:6],
                                "p_yes": v["p_yes"], "answer": v["answer"], "geometry": g.get("result"), "verdict": r["verdict"],
                                "label": lab.get((site, report, r["subject"], v["question"]))})
    return out



def score(work, out):
    DECISIVE[:] = [p for p in [work / "labels-decisive.jsonl"] if p.exists()]
    items = json.loads((work / "items.json").read_text())
    ans = json.loads((work / "answers.json").read_text())
    answers, V = ans["answers"], vectors(ans["embeddings"])
    jev, lab = jev_heldout(items, answers)
    ret = retrieval(lab, V)
    labels = load_labels()
    casc = {f"{pol}|cuts {cm}|reuse {ru}": cascade(work, items, answers, lab, V, pol, cm, ru)
            for pol in ("keep safety net", "keep calibrated safety net", "keep calibrated safety net + wide J5 margin", "trust geometry") for cm in ("validated", "fitted") for ru in ("3d", "embed", "none")}
    rules = {}
    for site in CALLS:
        rep = json.loads((work / f"replay-{site}.json").read_text())
        for c in ("first", "warm"):
            new = [r for r in rep if r["call"] == c and r["new"]]
            last = max(r["version"] for r in rep if r["call"] == c)
            final = {(r["id"], r["q"]) for r in rep if r["call"] == c and r["version"] == last}
            rules[f"{site}|{c}"] = {"sent_to_gemini": len(new), "by_question": dict(Counter(r["q"] for r in new)), "by_effect": dict(Counter(r["effect"] for r in new)),
                                    "object_gone_by_final_cards": sum((r["id"], r["q"]) not in final for r in new),
                                    "judge_run_question_counts (as the runs report them)": [sum(1 for r in rep if r["call"] == c and r["version"] == v) for v in sorted({r["version"] for r in rep if r["call"] == c})]}
    margin = {}  # could a wider rule margin absorb the q4 safety net? free width - u past the 0.711 m limit, on veto rows
    for site in CALLS:
        rep = json.loads((work / f"replay-{site}.json").read_text())
        for c in ("first", "warm"):
            m = [r["g"]["value"] - r["g"]["u"] - r["g"]["threshold"] for r in rep if r["call"] == c and r["new"] and r["q"] == "q4"
                 and r["effect"] == "veto" and r["g"].get("value") is not None and r["g"].get("direction") == "min"]
            margin[f"{site}|{c}"] = {"q4_veto_rows": len(m), **{f"margin_ge_{t}m": sum(x >= t for x in m) for t in (.25, .5, 1., 1.5)}}
    res = {"schema": "x13-hazards-v1",
           "what": "hazard judgement without a run-time VLM: rules first, Jev-Omni (service on its own A100) with temperature scaling and cuts fitted on other "
                   "videos, SigLIP 2 so400m look-alike retrieval (kNN, no training), the cascade's question counts per call; Gemini's recorded answers are an "
                   "offline baseline only",
           "labels": "agent-labelled (by looking at contact sheets): hazard-001 (round-1 objects), run 007's judgement audit, X8 set d, and here labels-frames.jsonl "
                     "(spill on every set d frame, the factory clip's whole question matrix)",
           "held_out": "every threshold and temperature is fitted on the other videos (leave one video out) and scored on the held-out one",
           "latency_and_memory": latency(ans), "jev_heldout": jev, "retrieval_and_reuse": ret, "reuse_within_call": within_video(work, items, V), "rules": rules, "q4_veto_margin": margin, "cascade": casc,
           "recorded_gemini_changes": recorded_changes(work, labels), "items": dict(Counter(f"{x['form']}|{x['q']}|{x['truth']}" for x in items))}
    rec = "keep calibrated safety net + wide J5 margin|cuts {}|reuse 3d"
    res["headline_per_call"] = {k: {"gemini_questions_sent_today": v["0_gemini_sent"], "rules_decide_no_question": v.get("1_rules_decide (not asked)", 0),
                                    "reused_same_object": v.get("2_reused (same object, earlier answer)", 0), "reach_jev": v.get("3_jev_asked", 0),
                                    "jev_decides": v.get("3_jev_decided", 0), "still_need_vlm_fitted_cuts": v.get("4_vlm_needed", 0) + v.get("4_vlm_needed (no local cut for the answer this row needs)", 0),
                                    "still_need_vlm_validated_cuts": (lambda w: w.get("4_vlm_needed", 0) + w.get("4_vlm_needed (no local cut for the answer this row needs)", 0))(casc[rec.format("validated")][k]),
                                    "bracket_keep_every_safety_net_validated": (lambda w: w.get("4_vlm_needed", 0) + w.get("4_vlm_needed (no local cut for the answer this row needs)", 0))(casc["keep safety net|cuts validated|reuse 3d"][k]),
                                    "bracket_trust_geometry_fitted": (lambda w: w.get("4_vlm_needed", 0) + w.get("4_vlm_needed (no local cut for the answer this row needs)", 0))(casc["trust geometry|cuts fitted|reuse 3d"][k])}
                                for k, v in casc[rec.format("fitted")].items() if not k.endswith("jev_cuts")}
    res["headline_policy"] = rec.format("fitted") + " (the brackets: every safety net kept with validated cuts; geometry trusted on every veto/hint row)"
    res["licences"] = {"Jev-Omni (akhilaaa3/Jev-Omni)": "Apache-2.0, commercial use yes (HF model card metadata; base google/gemma-4-12B-it, Apache-2.0)",
                       "SigLIP 2 so400m (google/siglip2-so400m-patch14-384)": "Apache-2.0, commercial use yes (HF model card metadata)"}
    sp = work / "spend.json"
    res["spend"] = json.loads(sp.read_text()) if sp.exists() else None
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    import shutil  # the agent labels and the contact sheets they were made from; Jev's raw answers (no embeddings)
    for f in ["labels-frames.jsonl", "labels-decisive.jsonl"] + sorted(p.name for p in (work / "sheets").glob("*.jpg") if p.name.startswith(("frames-", "decisive-", "factory-00"))):
        src = work / f if (work / f).exists() else work / "sheets" / f
        if src.exists():
            (out / ("sheets" if src.parent.name == "sheets" else "")).mkdir(exist_ok=True)
            shutil.copy(src, out / ("sheets" if src.parent.name == "sheets" else "") / f)
    (out / "jev-answers.json").write_text(json.dumps({k: [round(v, 5) for v in a] for k, a in answers.items()}))
    return res


def self_check():
    row = lambda r, **g: {"check": "J5", "geometry": {"result": r, **g}}  # noqa: E731
    assert effect(row(PASS)) == "veto" and effect(row(PASS, needs_clear_picture="x")) == "keep-pass" and effect(row(FAIL)) == "unfail"
    assert effect(row(REVIEW)) == "none" and effect(row(REVIEW, hazard_side=True)) == "make-fail" and effect(row(NO_DATA)) == "hint"
    assert effect({"check": "J3a", "geometry": {"result": REVIEW, "hazard_side": True}}) == "none" and effect({"check": "J4"}) == "hint"
    assert auroc([.9, .8, .1], [1, 0, 0]) == 1. and auroc([.5, .5], [1, 0]) == .5 and auroc([.1], [1]) is None
    assert abs(ece([.9] * 10, [1] * 9 + [0]) - 0.) < 1e-9 and abs(ece([.5] * 4, [1, 1, 1, 1]) - .5) < 1e-9
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-z / 3))).astype(int)  # true T = 3 on logits z
    assert abs(fit_temperature(z, y) - 3) < .5, fit_temperature(z, y)
    th, tc = cuts([.9] * 9 + [.9, .2, .2] + [.05] * 20, [1] * 9 + [0, 1, 0] + [0] * 20, precision=.9)
    assert th == .9 and tc == .05, (th, tc)
    assert cuts([.9, .1], [1, 0]) == (None, None)
    e = np.eye(3)[[0, 0, 1, 2]] + np.array([[0, .1, 0], [0, 0, .1], [.1, 0, 0], [0, .1, 0]])
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    pr, top = knn_prior(e, [1, 2, 3], [0], [0, 1, 0, 0], k=2)
    assert pr[0] > .99 and top[0] > .9, (pr, top)  # item 0's nearest labelled look-alike is item 1 (a positive)
    assert wilson(0, 10)[0] == 0 and wilson(10, 10)[1] == 1
    assert rel("none", .5, .1) == "no" and rel("keep-pass", .5, None) == "no" and rel("unfail", None, .1) == "yes"
    assert rel("veto", None, None) == "uncalibrated veto only" and rel("make-fail", None, .1) == "no" and rel("hint", .5, None) == "yes"
    assert call(.9, .8, .1) == "hazard" and call(.85, None, None) == "likely" and call(.05, .8, .1) == "clear" and call(.5, .8, .1) == "unsure"
    xs = [{"truth": 1, "call": "hazard"}, {"truth": 1, "call": "clear"}, {"truth": 0, "call": "clear"}, {"truth": 0, "call": "likely"}]
    c = confusion(xs)
    assert c["precision"] == 1. and c["recall"] == .5 and c["npv"] == .5 and c["positives_called_clear"] == 1 and c["false_alarm_rate"] == .5
    print("x13 hazards self-check ok: effect table, AUROC, ECE, temperature fit, cuts, kNN prior, Wilson, relevance, calls, confusion")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", choices=["replay", "factory", "items", "score"])
    ap.add_argument("--work", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--sites", default="me340,samsclub,walmart")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check or not a.cmd:
        self_check()
        sys.exit(0)
    if a.cmd == "replay":
        a.work.mkdir(parents=True, exist_ok=True)
        for site in a.sites.split(","):
            (a.work / f"replay-{site}.json").write_text(json.dumps(replay(site, a.work)))
            (a.work / f"recorded-{site}.json").write_text(json.dumps(recorded(site)))
    if a.cmd == "score":
        r = score(a.work, a.out)
        print(json.dumps({q: {k: v[k] for k in ("n", "positives", "auroc_raw", "ece_raw", "ece_heldout_calibrated")} | {"heldout": {k: v["heldout"][k] for k in ("hazard_calls", "hazard_right", "clear_calls", "clear_right", "unsure", "positives_called_clear")}} for q, v in r["jev_heldout"].items()}, indent=1))
    if a.cmd == "items":
        items = build_items(a.work)
        (a.work / "items.json").write_text(json.dumps(items, indent=0))
        print(Counter((x["form"], x["q"], x["truth"]) for x in items))
        (a.work / "sheets").mkdir(exist_ok=True)  # frames to label by looking (every frame's spill; the factory's whole matrix)
        todo = sorted({(x["video"], x["image"]) for x in items if x["form"] == "frame" and x.get("agent_labelled_here")})
        for i in range(0, len(todo), 12):
            chunk = todo[i:i + 12]
            sheet([p for _, p in chunk], [f"{v} {Path(p).stem}" for v, p in chunk], a.work / f"sheets/frames-{i // 12:02d}.jpg", cols=4, side=400)
        print(len(todo), "frames to label")
    if a.cmd == "factory":  # evidence + contact sheets for labelling the factory clip's objects by looking
        fac = factory_render(a.work)
        (a.work / "factory.json").write_text(json.dumps(fac, indent=1))
        (a.work / "sheets").mkdir(exist_ok=True)
        for i in range(0, len(fac), 9):
            chunk = fac[i:i + 9]
            sheet([a.work / x["image"] for x in chunk], [f"{i + j} {x['name']} {','.join(x['questions'])}" for j, x in enumerate(chunk)],
                  a.work / f"sheets/factory-{i // 9:02d}.jpg")
        print(len(fac), "factory objects")
