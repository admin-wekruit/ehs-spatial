"""r4/instances: one real object = one 3D card, measured against the delivered reports by mask overlap in the delivered frames.

Delivered objects are the delivered object map's entities (SAM 2 'everything' masks on the 4:3 centre crop, 640x480, one PNG per
instance; an entity's mask on a frame is the union of its instances there). Ours are the pick maps (640x360 over the 1280x720
frame, one id per pixel), each id resolved to its card (the cards' aliases) and a part to its parent card. Frames: our keyframes
that are delivered mask frames (+-1 frame, 33 ms). Per (card c, delivered object D), summed over the shared frames where D has a
mask: I = overlap pixels, A_c = c's pixels on those frames, A_D = D's pixels.
  piece      c is a piece of D:   I >= PIECE_IN x A_c (mostly inside D) and I >= PIECE_MIN x A_D (not a speck)
  fragments  D's pieces (a card whose parts are pieces counts once)
  covered    one card has I >= COVER x A_D, or D's pieces together do
  miss       not covered
  wrong merge  a card holds I >= COVER x A_D of two delivered objects that do not overlap each other (IoMin < 0.5 on shared frames)
Reference sets: the delivered objects with a model (22 / 67 / 40, X6's list) and every delivered object seen on >= 3 frames; the
'clean' subset has one mask per frame (entities made of several SAM 2 masks on one frame are often look-alikes the delivered
associator joined: splitting them is not fragmentation).

    python scripts/r4_instances_eval.py --run RUN_DIR --report REPORT --site me340 [--aliases A.json] [--out OUT.json]
    python scripts/r4_instances_eval.py --self-check
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
REPO = Path(__file__).resolve().parents[1]
MERGED = {"me340": "me340-object-models-303-merged", "samsclub-a2": "samsclub-a2-object-models-303-merged", "walmart": "walmart-object-models-303-merged"}
PIECE_IN, PIECE_MIN, COVER, TOL = .5, .03, .3, 1
CROP_X0, CROP_W = 80, 480  # the 4:3 centre crop (160..1120 of 1280) on the 640x360 pick grid


def delivered(site, frames_wanted):
    """{entity id: {name, frames: {frame: bool (360,480)}, multi: several masks on one frame, models: has a merged model}}."""
    import cv2
    sys.path.insert(0, str(REPO / "scripts"))
    import fast_report_eval as fe
    ref = fe.reference(site)
    omap = json.loads(Path(ref["object_map"]).read_text())
    names = json.loads(Path(ref["names"]).read_text())
    models = set(json.loads((PHASE2 / "runs" / MERGED[site] / "merge.json").read_text())["choice"])
    root = Path(ref["mask_root"])
    out = {}
    for e in omap["entities"]:
        if not e["entityId"].startswith("object-") or len(e["sourceFrames"]) < 3:
            continue
        per = {}
        for ob in e["observations"]:
            _, f, i = ob.split(":")
            per.setdefault(int(f), []).append(int(i))
        fr = {}
        for f, idx in per.items():
            if f not in frames_wanted:
                continue
            m = None
            for i in idx:
                x = cv2.imread(str(root / f"frame-{f:05d}" / f"instance-{i}-mask.png"), cv2.IMREAD_GRAYSCALE)
                if x is None:
                    continue
                m = x > 127 if m is None else m | (x > 127)
            if m is not None:
                fr[f] = cv2.resize(m.astype(np.uint8), (CROP_W, 360), interpolation=cv2.INTER_NEAREST).astype(bool)
        out[e["entityId"]] = {"name": (names.get(e["entityId"]) or {}).get("category") or e["label"], "frames": fr,
                              "multi": any(len(v) > 1 for v in per.values()), "model": e["entityId"] in models}
    return out


def delivered_frames(site):
    sys.path.insert(0, str(REPO / "scripts"))
    import fast_report_eval as fe
    return {int(p.name.split("-")[1]) for p in Path(fe.reference(site)["mask_root"]).iterdir() if p.name.startswith("frame-")}


def run_maps(run_dir, report):
    """Our final pick version: [(source frame, (360,640) uint16 map)], entity list, and the final cards (aliases, parts)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import fast_report_eval as fe
    pv = fe.patch_versions(run_dir, report, "pick")[-1]
    pick = fe.Pick(pv["data"], fe.pick_bytes(run_dir, pv))
    maps = [(f["frame"], pick.map(i)) for i, f in enumerate(pick.frames)]
    cv = fe.patch_versions(run_dir, report, "object_cards")[-1]
    cards = fe.patch_data(run_dir, cv, "cards")
    return maps, pv["data"]["entities"], cards


def resolve(entities, aliases):
    """pick code -> card id (aliases chained), people and background -> None."""
    out = []
    for e in entities:
        if not e or e.startswith("person:"):
            out.append(None)
            continue
        seen = set()
        while e in aliases and e not in seen:
            seen.add(e)
            e = aliases[e]
        out.append(e)
    return out


def root_of(parent):
    """card -> its top parent card (a part of a part counts with the whole)."""
    def root(c):
        seen = set()
        while c in parent and c not in seen:
            seen.add(c)
            c = parent[c]
        return c
    return root


def tables(dl, maps, codes):
    """Sums over the shared frames: I[(card, D)], A_c[(card, D)], A_D[D], and delivered pair overlaps."""
    I, AC, AD, DD, frames_used = {}, {}, {}, {}, set()
    by_frame = {}
    for f, m in maps:
        by_frame.setdefault(f, m)
    for D, d in dl.items():
        for f, mD in d["frames"].items():
            g = next((f + s for s in (0, -1, 1, -2, 2)[:1 + 2 * TOL] if f + s in by_frame), None)
            if g is None:
                continue
            frames_used.add(f)
            lab = by_frame[g][:, CROP_X0:CROP_X0 + CROP_W]
            AD[D] = AD.get(D, 0) + int(mD.sum())
            ids, cnt = np.unique(lab, return_counts=True)
            inside = np.bincount(lab[mD].ravel(), minlength=int(lab.max()) + 1)
            for code, n in zip(ids, cnt):
                c = codes[code] if code < len(codes) else None
                if c is None:
                    continue
                AC[(c, D)] = AC.get((c, D), 0) + int(n)
                if inside[code]:
                    I[(c, D)] = I.get((c, D), 0) + int(inside[code])
    Ds = list(dl)
    for i, a in enumerate(Ds):  # delivered objects that overlap each other (part / whole in the delivered map)
        for b in Ds[i + 1:]:
            for f in dl[a]["frames"].keys() & dl[b]["frames"].keys():
                x, y = dl[a]["frames"][f], dl[b]["frames"][f]
                inter = int((x & y).sum())
                if inter and inter >= .5 * min(x.sum(), y.sum()):
                    DD[(a, b)] = DD[(b, a)] = True
                    break
    return I, AC, AD, DD, frames_used


def score(dl, I, AC, AD, DD, subset=None, root=lambda c: c):
    """Fragments count a card and its parts once (root); coverage and merges are per card."""
    Ds = [D for D in dl if AD.get(D) and (subset is None or subset(dl[D]))]
    pieces = {D: [] for D in Ds}
    cover1 = {D: 0. for D in Ds}
    holds = {}
    for (c, D), n in I.items():
        if D not in pieces:
            continue
        if n >= PIECE_IN * AC[(c, D)] and n >= PIECE_MIN * AD[D]:
            pieces[D].append(c)
        cover1[D] = max(cover1[D], n / AD[D])
        if n >= COVER * AD[D]:
            holds.setdefault(c, []).append(D)
    covered = [D for D in Ds if cover1[D] >= COVER or sum(I[(c, D)] for c in pieces[D]) >= COVER * AD[D]]
    frag = {D: len({root(c) for c in pieces[D]}) for D in covered}
    merges = []
    for c, held in holds.items():
        apart = [(a, b) for i, a in enumerate(held) for b in held[i + 1:] if not DD.get((a, b))]
        if apart:
            merges.append({"card": c, "delivered": held})
    n = len(Ds)
    fr = np.array([max(v, 1) for v in frag.values()]) if frag else np.zeros(0)
    return {"delivered": n, "covered": len(covered), "missed": n - len(covered), "miss_rate": round((n - len(covered)) / n, 3) if n else None,
            "in_pieces": int((fr >= 2).sum()), "in_pieces_share": round(float((fr >= 2).mean()), 3) if len(fr) else None,
            "pieces_per_covered_mean": round(float(fr.mean()), 3) if len(fr) else None,
            "pieces_per_covered_median": float(np.median(fr)) if len(fr) else None,
            "wrong_merge_cards": len(merges), "delivered_in_wrong_merges": len({D for m in merges for D in m["delivered"] if D in pieces}),
            "detail": {"pieces": {D: pieces[D] for D in covered if len(pieces[D]) >= 2}, "missed": [D for D in Ds if D not in covered],
                       "merges": merges}}


def evaluate(site, maps, entities, aliases, parent=None, dl=None):
    frames = {f for f, _ in maps}
    want = {f + s for f in frames for s in range(-TOL, TOL + 1)}
    dl = dl if dl is not None else delivered(site, want & delivered_frames(site))
    codes = resolve(entities, aliases)
    root = root_of(parent or {})
    I, AC, AD, DD, used = tables(dl, maps, codes)
    cards_seen = {c for (c, _D) in AC}
    return {"frames_shared": len(used), "cards_on_shared_frames": len(cards_seen),
            "models": score(dl, I, AC, AD, DD, lambda d: d["model"], root),
            "all": score(dl, I, AC, AD, DD, None, root),
            "clean": score(dl, I, AC, AD, DD, lambda d: not d["multi"], root)}, dl


def self_check():
    """Two delivered objects A (left block) and B (right block) on one frame; ours: A in two cards (a fragment), B missed, and
    a card over A's lower half and B would be a wrong merge (checked in a second map)."""
    A = np.zeros((360, 480), bool)
    A[100:200, 50:150] = True
    B = np.zeros((360, 480), bool)
    B[100:200, 300:400] = True
    dl = {"A": {"name": "a", "frames": {10: A}, "multi": False, "model": True}, "B": {"name": "b", "frames": {10: B}, "multi": False, "model": True}}
    m = np.zeros((360, 640), np.uint16)
    m[100:150, 80 + 50:80 + 150] = 1
    m[150:200, 80 + 50:80 + 150] = 2
    r, _ = evaluate("x", [(10, m)], [None, "o1", "o2"], {}, dl=dl)
    s = r["all"]
    assert (s["covered"], s["missed"], s["in_pieces"], s["wrong_merge_cards"]) == (1, 1, 1, 0), s
    r, _ = evaluate("x", [(10, m)], [None, "o1", "o2"], {"o2": "o1"}, dl=dl)  # merged: one card
    assert r["all"]["in_pieces"] == 0 and r["all"]["pieces_per_covered_median"] == 1, r["all"]
    r, _ = evaluate("x", [(10, m)], [None, "o1", "o2"], {}, parent={"o2": "o1"}, dl=dl)  # a part of its parent card: one object
    assert r["all"]["in_pieces"] == 0, r["all"]
    m2 = np.zeros((360, 640), np.uint16)
    m2[100:200, 80 + 50:80 + 400] = 3
    r, _ = evaluate("x", [(10, m2)], [None, None, None, "o3"], {}, dl=dl)
    assert r["all"]["wrong_merge_cards"] == 1 and r["all"]["covered"] == 2, r["all"]
    print("r4_instances_eval self-check ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path)
    p.add_argument("--report")
    p.add_argument("--site")
    p.add_argument("--aliases", type=Path, help="{aliases: {object: card}, parent: {card: parent card}} instead of the run's cards")
    p.add_argument("--out", type=Path)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        return self_check()
    maps, entities, cards = run_maps(a.run, a.report)
    parent = {c["id"]: c["part_of"]["id"] for c in cards["cards"] if (c.get("part_of") or {}).get("kind") == "part"}
    al = json.loads(a.aliases.read_text()) if a.aliases else {"aliases": cards.get("aliases") or {}, "parent": parent}
    res, _ = evaluate(a.site, maps, entities, al["aliases"], al.get("parent"))
    res["cards"] = sum(c["kind"] == "object" for c in cards["cards"])
    res["parts"] = sum(bool(c.get("part_of")) for c in cards["cards"])
    res["cards_stats"] = cards.get("stats")
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "detail"} if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()


# ---------- contact sheets (agent audit) ----------

CLIPS = {"me340": "me340-165", "samsclub-a2": "samsclub-337", "walmart": "walmart-190"}
COLORS = [(66, 135, 245), (245, 66, 66), (66, 245, 120), (245, 200, 66), (200, 66, 245), (66, 230, 245), (245, 140, 66), (160, 245, 66)]


def video_frames(site, frames):
    """{frame: (360,480,3) BGR}: the source clip's 4:3 centre crop, the delivered masks' raster."""
    import cv2
    cap = cv2.VideoCapture(str(PHASE2 / "data/clips" / CLIPS[site] / "source-full.mp4"))
    out, want, i = {}, sorted(set(frames)), 0
    for f in want:
        if f != i:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        i = f + 1
        if ok:
            out[f] = cv2.resize(img[:, 160:1120], (CROP_W, 360), interpolation=cv2.INTER_AREA)
    return out


def zoom(x, masks, out_wh=(400, 225), min_h=90):
    """Crop x around the masks' joint box (x 2.2, at least min_h high, the frame's aspect) and resize: small things readable."""
    import cv2
    h, w = x.shape[:2]
    m = np.zeros((h, w), bool)
    for k in masks:
        m |= k
    ys, xs = np.nonzero(m)
    if not len(xs):
        return cv2.resize(x, out_wh, interpolation=cv2.INTER_AREA)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    hh = max(min_h, 2.2 * (ys.max() - ys.min() + 1), 2.2 * (xs.max() - xs.min() + 1) * out_wh[1] / out_wh[0])
    hh = min(hh, h)
    ww = min(hh * out_wh[0] / out_wh[1], w)
    y0, x0 = int(np.clip(cy - hh / 2, 0, h - hh)), int(np.clip(cx - ww / 2, 0, w - ww))
    return cv2.resize(x[y0:y0 + int(hh), x0:x0 + int(ww)], out_wh, interpolation=cv2.INTER_CUBIC)


def tile(img, mD, cards_here, title, zoomed=True):
    """One panel: the frame, the delivered mask's outline (white), our cards filled (colours) with their ids; zoomed on them."""
    import cv2
    x = img.copy()
    for k, (cid, m) in enumerate(cards_here):
        col = np.array(COLORS[k % len(COLORS)], np.uint8)
        x[m] = (.55 * x[m] + .45 * col).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if len(xs):
            cv2.putText(x, cid.replace("obj-", ""), (int(np.median(xs)), int(np.median(ys))), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
    if mD is not None:
        cs, _ = cv2.findContours(mD.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(x, cs, -1, (255, 255, 255), 2)
    if zoomed:
        x = zoom(x, [m for _, m in cards_here] + ([mD] if mD is not None else []))
    cv2.rectangle(x, (0, 0), (x.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(x, title[:64], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
    return x


def full_frames(site, frames, size=(640, 360)):
    """{frame: (360,640,3) BGR} of the whole 1280x720 source frame (the pick grid)."""
    import cv2
    cap = cv2.VideoCapture(str(PHASE2 / "data/clips" / CLIPS[site] / "source-full.mp4"))
    out = {}
    for f in sorted(set(frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            out[f] = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return out


def audit_sheets(site, maps, entities, titles, out_prefix, n=30, seed=0, per_sheet=10, min_px=150):
    """Fresh contact sheets for the agent audit: n cards drawn at random (seeded) among the cards with >= min_px on some
    keyframe; per card 4 keyframes spread over the ones it is on, its pixels filled, the rest of the frame as is. -> [ids]."""
    import cv2
    area = {}
    for f, m in maps:
        ids, cnt = np.unique(m, return_counts=True)
        for i, c in zip(ids, cnt):
            if i and entities[i] and c >= min_px:
                area.setdefault(i, []).append((f, int(c)))
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(sorted(area), min(n, len(area)), replace=False).tolist())
    by = dict(maps)
    rows = []
    for i in pick:
        fs = [f for f, _ in sorted(area[i])]
        fs = [fs[j] for j in np.linspace(0, len(fs) - 1, min(4, len(fs))).astype(int)]
        rows.append((i, fs))
    vf = full_frames(site, [f for _, fs in rows for f in fs])
    paths = []
    for s in range(0, len(rows), per_sheet):
        img_rows = []
        for i, fs in rows[s:s + per_sheet]:
            tiles = []
            for f in fs:
                x = vf[f].copy()
                m = by[f] == i
                x[m] = (.5 * x[m] + .5 * np.array((40, 40, 255))).astype(np.uint8)
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(x, cs, -1, (0, 255, 255), 1)
                x = zoom(x, [m])
                cv2.rectangle(x, (0, 0), (x.shape[1], 16), (0, 0, 0), -1)
                cv2.putText(x, f"#{pick.index(i) + 1} {titles.get(entities[i], entities[i])[:60]} f{f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(x)
            while len(tiles) < 4:
                tiles.append(np.zeros_like(tiles[0]))
            img_rows.append(np.hstack(tiles))
        path = f"{out_prefix}-{s // per_sheet + 1}.jpg"
        cv2.imwrite(path, np.vstack(img_rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
        paths.append(path)
    return [entities[i] for i in pick], paths


def look(site, dl, maps, entities, what, path, titles=None):
    """A contact sheet: per delivered object (object-...) up to 4 shared frames with its outline (white) and our cards over it
    (colours, ids); per card id up to 4 frames with its pixels. maps: [(frame, (360,640) codes)], entities[code] = card."""
    import cv2
    titles = titles or {}
    by = dict(maps)
    code_of = {}
    for i, e in enumerate(entities):
        if e:
            code_of.setdefault(e, []).append(i)
    rows = []
    for w in what:
        if w in dl:
            d = dl[w]
            fs = sorted(f for f in d["frames"] if any(f + s in by for s in (0, -1, 1)))
            fs = [fs[i] for i in np.linspace(0, len(fs) - 1, min(4, len(fs))).astype(int)] if fs else []
            vf = video_frames(site, fs)
            tiles = []
            for f in fs:
                g = next(f + s for s in (0, -1, 1) if f + s in by)
                lab = by[g][:, CROP_X0:CROP_X0 + CROP_W]
                mD = d["frames"][f]
                here = {}
                for i in np.unique(lab[mD]):
                    if i and entities[i]:
                        here.setdefault(entities[i], np.zeros_like(mD))
                        here[entities[i]] |= lab == i
                here = sorted(here.items(), key=lambda kv: -(kv[1] & mD).sum())[:6]
                tiles.append(tile(vf[f], mD, here, f"{w} {d['name']} f{f}"))
        else:
            fs = [f for f, m in maps if np.isin(m, code_of.get(w, [-1])).any()]
            fs = [fs[i] for i in np.linspace(0, len(fs) - 1, min(4, len(fs))).astype(int)] if fs else []
            vf = video_frames(site, fs)
            tiles = [tile(vf[f], None, [(w, np.isin(by[f][:, CROP_X0:CROP_X0 + CROP_W], code_of[w]))], f"{w} {titles.get(w, '')} f{f}") for f in fs]
        if tiles:
            while len(tiles) < 4:
                tiles.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(tiles))
    if rows:
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return path
