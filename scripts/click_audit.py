"""Independent click audit (R8): clicks at random pixels on random video frames (not on SAM 3 references), resolved with
the viewer's own pick rule (fast_report_eval.Pick, the same code path as live-report.ts pickAt), drawn on contact sheets
(the frame with the picked entity tinted, a zoom around the click, the card's name) for an agent to label by looking:

    correct             the highlighted entity is the thing under the click (a coarse group, e.g. 'stacked boxes' for one
                        box of the stack, counts as correct and is noted 'coarse')
    wrong               an entity was picked, but not the thing under the click (a stale or misplaced mask, a neighbour)
    miss                nothing was picked, but the click is on a discrete object or person that should be clickable
    background          nothing was picked and the click is on floor, wall, ceiling or unstructured clutter (a right miss)
    background-hit      an entity was picked on floor / wall / ceiling where no object is (a false hit)

Labels are agent-made (not ground truth). --persons samples one click inside each SAM 3 person reference (X1's eval frames,
>= 400 px) instead: the sampling uses SAM 3, the labels do not.

    python scripts/click_audit.py sample --mirror M --report R --site walmart --out DIR [--n 60] [--seed 1] [--persons]
    python scripts/click_audit.py render --clicks DIR/clicks.json --mirror M --report R --out DIR2   # same clicks, another run
    python scripts/click_audit.py score DIR [DIR ...]      # clicks.json + labels.json -> table
    python scripts/click_audit.py --self-check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import fast_report_eval as ev  # noqa: E402

LABELS = ("correct", "wrong", "miss", "background", "background-hit", "not-person", "unclear")  # the last two: person-reference clicks
PERSON_MIN_PX, TILE_W, TILE_H, ZOOM = 400, 480, 270, 110  # zoom: +-110 source px around the click


def load_run(mirror, report, version=None):
    """-> (pick (ev.Pick), its patch, cards by id (aliases resolved), fps, video path)."""
    picks = ev.patch_versions(mirror, report, "pick")
    p = picks[-1] if version is None else next(x for x in picks if x["version"] == version)
    cards_patch = ev.patch_versions(mirror, report, "object_cards")[-1]
    layer = ev.patch_data(mirror, cards_patch, "cards")
    cards = {c["id"]: c for c in layer["cards"]}
    for old, new in (layer.get("aliases") or {}).items():
        if new in cards and old not in cards:
            cards[old] = cards[new]
    video = ev.patch_versions(mirror, report, "video")[0]
    return pick_of(mirror, p), p, cards, video["data"]["fps"], Path(mirror) / "blobs/sha256" / video["blobs"]["video"]["sha256"]


def pick_of(mirror, patch):
    """ev.Pick over a pick patch, one blob or per-chunk blobs (panoptes-pick-v1 'chunks')."""
    return ev.Pick(patch["data"], ev.pick_bytes(mirror, patch))


def name_of(cards, eid):
    if eid is None:
        return "UNKNOWN REGION"
    c = cards.get(eid)
    if c is None:
        return f"(no card) {eid}"
    return (c.get("identity") or {}).get("name") or c["kind"]


def last_rule(pick, t, x, y):
    """Round 1's lookup (the last keyframe with t_key <= t, none before the first or past a cut), for the baseline."""
    i = int(np.searchsorted(pick.times, t + 1e-6, side="right")) - 1
    if i < 0 or (pick.frames[i].get("t_end") is not None and t >= pick.frames[i]["t_end"]):
        return None, -1, 0
    f = pick.frames[i]
    code = int(pick.map(i)[min(int(y * f["h"] / pick.sh), f["h"] - 1), min(int(x * f["w"] / pick.sw), f["w"] - 1)])
    return pick.data["entities"][code] if code else None, i, code


def resolve(pick, cards, clicks, fps, rule="nearest"):
    for c in clicks:
        eid, i, code = (last_rule if rule == "last" else type(pick).at)(pick, c["frame"] / fps, c["x"], c["y"])
        f = pick.frames[i] if i >= 0 else {}
        c.update(entity=eid, name=name_of(cards, eid), kind=(cards.get(eid) or {}).get("kind"), pick_frame=f.get("frame"),
                 pick_source=f.get("source"), pick_index=i, code=code)
    return clicks


def random_clicks(pick, fps, n, seed):
    """Uniform over video frames that have a pick map (inside [t, t_end) of some pick frame) and over 1280 x 720 pixels."""
    rng = np.random.default_rng(seed)
    covered = sorted({q for f in pick.frames for q in range(int(round(f["t"] * fps)), int(np.ceil((f.get("t_end") or f["t"] + .2) * fps - 1e-6)))})
    frames = rng.choice(covered, n, replace=len(covered) < n)
    W, H = pick.sw, pick.sh
    return [{"k": k, "frame": int(q), "x": int(rng.integers(0, W)), "y": int(rng.integers(0, H)), "sample": "random"} for k, q in enumerate(frames)]


def person_clicks(site, seed, min_px=PERSON_MIN_PX, refs_dir=None):
    """One click inside each SAM 3 person reference (X1's eval frames), at a random pixel of its 2 px-eroded interior."""
    import cv2
    refs = ev.load_refs(Path(refs_dir or ev.PHASE2 / "runs/mvp-results/eval") / f"refs-{site}.npz")
    rng = np.random.default_rng(seed)
    out = []
    for k in np.flatnonzero((np.asarray(refs["word"]) == -1) & (np.asarray(refs["area"]) >= min_px)):
        m = cv2.erode(ev.ref_mask(refs, int(k)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ys, xs = np.nonzero(m)
        if not len(xs):
            continue
        j = rng.integers(0, len(xs))
        out.append({"k": len(out), "frame": int(refs["frames"][refs["frame"][k]]), "x": int(xs[j]), "y": int(ys[j]), "sample": "person ref",
                    "ref": int(k), "ref_px": int(refs["area"][k])})
    return out


def frames_of(video, wanted):
    """{frame index: BGR} for the wanted frames, one sequential decode."""
    import cv2
    cap, out, i, want = cv2.VideoCapture(str(video)), {}, 0, set(wanted)
    while want - set(out):
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = img
        i += 1
    return out


def tile(img, pick, c):
    """Frame (tinted pick region, click ring) + zoom, caption below. -> (TILE_H + 34, TILE_W + TILE_H) BGR."""
    import cv2
    H, W = img.shape[:2]
    vis = img.copy()
    if c["code"]:
        m = cv2.resize((pick.map(c["pick_index"]) == c["code"]).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        vis[m > 0] = (.55 * vis[m > 0] + .45 * np.array([255, 255, 0])).astype(np.uint8)  # cyan tint
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (0, 255, 255), 2)
    x, y = c["x"], c["y"]
    full = cv2.resize(vis, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    s = TILE_W / W
    cv2.circle(full, (int(x * s), int(y * s)), 9, (0, 0, 0), 4)
    cv2.circle(full, (int(x * s), int(y * s)), 9, (255, 0, 255), 2)
    a, b = int(np.clip(x - ZOOM, 0, W - 2 * ZOOM)), int(np.clip(y - ZOOM, 0, H - 2 * ZOOM))
    zoom = cv2.resize(vis[b:b + 2 * ZOOM, a:a + 2 * ZOOM], (TILE_H, TILE_H), interpolation=cv2.INTER_CUBIC)
    zs = TILE_H / (2 * ZOOM)
    zx, zy = int((x - a) * zs), int((y - b) * zs)
    for r, col, th in ((14, (0, 0, 0), 4), (14, (255, 0, 255), 2)):
        cv2.line(zoom, (zx - r, zy), (zx - 5, zy), col, th)
        cv2.line(zoom, (zx + 5, zy), (zx + r, zy), col, th)
        cv2.line(zoom, (zx, zy - r), (zx, zy - 5), col, th)
        cv2.line(zoom, (zx, zy + 5), (zx, zy + r), col, th)
    out = np.full((TILE_H + 34, TILE_W + TILE_H, 3), 255, np.uint8)
    out[:TILE_H, :TILE_W], out[:TILE_H, TILE_W:] = full, zoom
    cv2.rectangle(out, (TILE_W, 0), (TILE_W + TILE_H - 1, TILE_H - 1), (0, 0, 0), 1)
    src = (c.get("pick_source") or "-")[:4]
    txt = f"#{c['k']:02d} f{c['frame']} (map f{c.get('pick_frame')} {src}) -> {c['name']}"
    cv2.putText(out, txt[:70], (6, TILE_H + 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def sheets(video, pick, clicks, out, stem, per=10, cols=2):
    import cv2
    imgs = frames_of(video, [c["frame"] for c in clicks])
    paths = []
    for s0 in range(0, len(clicks), per):
        tiles = [tile(imgs[c["frame"]], pick, c) for c in clicks[s0:s0 + per]]
        th, tw = tiles[0].shape[:2]
        rows = -(-len(tiles) // cols)
        sheet = np.full((rows * (th + 8), cols * (tw + 8), 3), 200, np.uint8)
        for i, t in enumerate(tiles):
            r, q = divmod(i, cols)
            sheet[r * (th + 8):r * (th + 8) + th, q * (tw + 8):q * (tw + 8) + tw] = t
        p = Path(out) / f"{stem}-{s0 // per}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(str(p))
    return paths


def save(out, clicks, meta, stem):
    out.mkdir(parents=True, exist_ok=True)
    keep = ("k", "frame", "x", "y", "sample", "ref", "ref_px", "entity", "name", "kind", "pick_frame", "pick_source")
    (out / f"{stem}.json").write_text(json.dumps({**meta, "clicks": [{k: c.get(k) for k in keep} for c in clicks]}, indent=1))


def score(dirs):
    """Per clicks file: label shares (agent-labelled) and the entity-level click accuracy."""
    rows = []
    for d in dirs:
        for f in sorted(Path(d).glob("clicks*.json")):
            lab_path = f.with_name(f.name.replace("clicks", "labels"))
            if not lab_path.exists():
                continue
            meta, labels = json.loads(f.read_text()), json.loads(lab_path.read_text())["labels"]
            clicks = meta["clicks"]
            got = [labels.get(str(c["k"])) for c in clicks]
            assert all(g and g["label"] in LABELS for g in got), f"{lab_path}: unlabelled or unknown labels"
            n = len(got)
            cnt = {k: sum(g["label"] == k for g in got) for k in LABELS}
            on_thing = cnt["correct"] + cnt["wrong"] + cnt["miss"]  # clicks on something that should be clickable
            hit = [g for g, c in zip(got, clicks) if c["entity"]]
            rows.append({"file": str(f), "run": meta.get("report"), "site": meta.get("site"), "sample": meta.get("sample"), "n": n, **cnt,
                         "click_correct_share": round((cnt["correct"] + cnt["background"]) / n, 3),
                         "on_object_correct": round(cnt["correct"] / on_thing, 3) if on_thing else None,
                         "picked_precision": round(cnt["correct"] / len(hit), 3) if hit else None,
                         "coarse": sum("coarse" in (g.get("note") or "") for g in got),
                         "person_clicks": sum(1 for c in clicks if c.get("kind") == "person")})
            if meta.get("sample") == "person refs":  # clicks inside SAM 3 'person' masks: real people vs pictures of people
                real = [g for g in got if g["label"] in ("correct", "wrong", "miss")]
                pics = [c for g, c in zip(got, clicks) if g["label"] == "not-person"]
                rows[-1].update(real_people=len(real), real_person_correct=round(cnt["correct"] / len(real), 3) if real else None,
                                pictures=len(pics), pictures_as_person=sum(c.get("kind") == "person" for c in pics), unclear=cnt["unclear"])
    return rows


def self_check():
    import tempfile
    # a 2-frame pick layer through fast_report_eval's encoder: entity 1 top-left on frame 0, entity 2 everywhere on frame 1
    m0, m1 = np.zeros((36, 64), np.uint16), np.full((36, 64), 2, np.uint16)
    m0[:18, :32] = 1
    data, blob = ev.pick_encode([m0, m1], [{"t": 0., "t_end": .2, "frame": 0}, {"t": .2, "t_end": .4, "frame": 5}], [None, "obj-a", "person:x"])
    import gzip
    pk = ev.Pick(data, gzip.decompress(blob))
    cards = {"obj-a": {"kind": "object", "identity": {"name": "box"}}, "person:x": {"kind": "person", "identity": {"name": "person"}}}
    cl = resolve(pk, cards, [{"k": 0, "frame": 1, "x": 100, "y": 100}, {"k": 1, "frame": 1, "x": 1000, "y": 600},
                             {"k": 2, "frame": 6, "x": 5, "y": 5}], 25.)
    assert [c["entity"] for c in cl] == ["obj-a", None, "person:x"] and cl[0]["name"] == "box" and cl[1]["name"] == "UNKNOWN REGION"
    assert [c["entity"] for c in resolve(pk, cards, [{"k": 0, "frame": 4, "x": 5, "y": 5}], 25.)] == ["person:x"]  # 0.16 s: nearer frame 1
    assert [c["entity"] for c in resolve(pk, cards, [{"k": 0, "frame": 4, "x": 5, "y": 5}], 25., "last")] == ["obj-a"]  # round 1: frame 0
    rc = random_clicks(pk, 25., 50, 0)
    assert all(0 <= c["frame"] < 10 and 0 <= c["x"] < 1280 and 0 <= c["y"] < 720 for c in rc), rc[:3]
    with tempfile.TemporaryDirectory() as d:
        save(Path(d), cl, {"report": "r", "site": "s", "sample": "random"}, "clicks-x")
        (Path(d) / "labels-x.json").write_text(json.dumps({"labels": {"0": {"label": "correct", "note": "coarse"}, "1": {"label": "background"},
                                                                     "2": {"label": "wrong"}}}))
        r = score([d])[0]
        assert (r["correct"], r["background"], r["wrong"], r["coarse"], r["picked_precision"]) == (1, 1, 1, 1, .5), r
    print("click_audit self-check ok: resolve through the viewer's pick rule, random clicks in covered frames, scoring")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", choices=("sample", "render", "score"))
    p.add_argument("dirs", nargs="*")
    p.add_argument("--mirror", type=Path)
    p.add_argument("--report")
    p.add_argument("--site")
    p.add_argument("--out", type=Path)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--version", type=int)
    p.add_argument("--persons", action="store_true")
    p.add_argument("--clicks", type=Path)
    p.add_argument("--rule", choices=("nearest", "last"), default="nearest", help="pick frame rule: mvp2's nearest keyframe, or round 1's last")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        return self_check()
    if a.mode == "score":
        print(json.dumps(score(a.dirs), indent=1))
        return
    pick, patch, cards, fps, video = load_run(a.mirror, a.report, a.version)
    src = {}
    if a.mode == "render":
        src = json.loads(a.clicks.read_text())
        clicks, stem = [{k: c[k] for k in ("k", "frame", "x", "y", "sample", "ref", "ref_px") if k in c} for c in src["clicks"]], a.clicks.stem
    else:
        clicks = person_clicks(a.site, a.seed) if a.persons else random_clicks(pick, fps, a.n, a.seed)
        stem = f"clicks-{a.site}-{'persons' if a.persons else 'random'}-s{a.seed}"
    clicks = resolve(pick, cards, clicks, fps, a.rule)
    meta = {"report": a.report, "mirror": str(a.mirror), "site": a.site, "pick_version": patch["version"], "fps": fps, "rule": a.rule,
            "sample": src.get("sample") if a.mode == "render" else "person refs" if a.persons else "random",
            "seed": src.get("seed") if a.mode == "render" else a.seed, "labels_by": "agent (contact sheets)"}
    save(a.out, clicks, meta, stem)
    print(json.dumps({"clicks": len(clicks), "picked": sum(1 for c in clicks if c["entity"]),
                      "sheets": sheets(video, pick, clicks, a.out, stem.replace("clicks", "sheet"))}, indent=1))


if __name__ == "__main__":
    main()
