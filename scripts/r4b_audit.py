"""r4b integrate: the fresh card audit by eye (the user's four acceptance items on one sheet per card) and the acceptance tallies.

    python scripts/r4b_audit.py cards MIRROR REPORT OUT_DIR [--n 30 --seed 29]   # contact sheets + labels.json (labels by eye)
    python scripts/r4b_audit.py score OUT_DIR [...]                              # labels.json -> tallies
    python scripts/r4b_audit.py share MIRROR REPORT                              # typed / segmented / physical / model shares
    python scripts/r4b_audit.py --self-check

cards: n object cards drawn at random (seeded) among the final object cards with a pick region (with or without a model), each on
the keyframe where its region is largest (its own best views first, r4_models.sheet's rule). Left: the crop with the region outlined;
right: the same crop dimmed with the card's display model drawn from that camera (SAM 3D's accepted mesh, else the primitive: seen
faces solid, guessed faint). Above: the shown name, the type (family or shape, and whose votes gave it), the physical values (+-u;
'>=' / '<=' bounds; n/o not observed; n/m not measurable), the model line. Labels (agent-labelled, by looking):
  type      right | close | wrong | unclear       (the family, or the specific name when one is shown)
  outline   right | partial | wrong               (the region on this keyframe is the thing, a part of it, or something else)
  physical  plausible | implausible | unclear     (against the picture at the stated u)
  model     plausible | implausible | absent      (a model is drawn and has the thing's place, size and rough shape)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
LABELS = {"type": ("right", "close", "wrong", "unclear"), "outline": ("right", "partial", "wrong"),
          "physical": ("plausible", "implausible", "unclear"), "model": ("plausible", "implausible", "absent", "unclear")}
PHYS = (("top_above_floor", "top"), ("base_above_floor", "base"), ("height", "H"), ("width", "W"), ("depth", "D"), ("visible_length", "VL"),
        ("principal_axis_tilt_deg", "tilt"))


def type_line(c):
    import r4_naming_results as nr
    i = c.get("identity") or {}
    t = i.get("type") or {}
    return f"type: {t.get('label')} [{t.get('family')}] by {str(t.get('source'))[:34]}; route {nr.route(c)}"[:92]


def cards(mirror, report, out_dir, n=30, seed=29, per=10):
    import cv2
    import fast_report_eval as ev
    import r4_models as rm
    from r4_physical import short
    L, pick, models_patch = rm.load(mirror, report)
    root, ent = rm.owners(L), pick.data["entities"]
    cams = {s["index"]: s for s in L["cameras"]["shots"]}
    objs = [c for c in L["object_cards"]["cards"] if c.get("kind") == "object"]
    view = {c["id"]: set((c.get("views") or {}).get("best") or []) for c in objs}
    best = {}
    for i, f in enumerate(pick.frames):
        cam = cams.get(f.get("shot"))
        if f.get("source") != "segmented" or cam is None or f["frame"] not in cam["keys"]:
            continue
        m = pick.map(i)
        idx, cnt = np.unique(m[m > 0], return_counts=True)
        for j, a in zip(idx, cnt):
            cid = root(ent[int(j)]) if ent[int(j)] else None
            rank = (f["frame"] in view.get(cid, ()), int(a))
            if cid in view and rank > best.get(cid, ((False, 0),))[0]:
                best[cid] = (rank, i)
    pool = [c for c in objs if c["id"] in best]
    chosen = [pool[i] for i in sorted(np.random.default_rng(seed).choice(len(pool), min(n, len(pool)), replace=False))]
    sam = rm.sam_by_card(L, models_patch)
    vp = ev.patch_versions(mirror, report, "video")[-1]
    cap = cv2.VideoCapture(str(Path(mirror) / "blobs" / "sha256" / vp["blobs"]["video"]["sha256"]))
    tiles, rows = [], []
    for k, c in enumerate(chosen):
        i = best[c["id"]][1]
        f = pick.frames[i]
        cam = cams[f["shot"]]
        key = cam["keys"].index(f["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame"])
        ok, img = cap.read()
        H, W = img.shape[:2]
        sx, sy = W / cam["wh"][0], H / cam["wh"][1]
        K = np.asarray(cam["K"][key], float) * [[sx, 1, sx], [1, sy, sy], [1, 1, 1]]
        c2w = np.asarray(cam["c2w"][key], float)
        mk = cv2.resize(np.isin(pick.map(i), [j for j, e in enumerate(ent) if e and root(e) == c["id"]]).astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST)
        s, glb, uv = sam.get(c["id"]), None, None
        if s and s.get("accepted") and models_patch and f"model-{s['object']}" in models_patch["blobs"]:
            try:
                glb = rm.glb_tris(ev.blob_bytes(mirror, models_patch["blobs"][f"model-{s['object']}"]["sha256"]), s["transform"]["position"])
            except (OSError, KeyError, ValueError):  # a mesh blob left on the Volume (--mirror-max-mb): the primitive is drawn
                glb = None
        has_model = bool((c.get("model") or {}).get("kind"))
        right, uv = rm.render(img, c, glb, K, c2w) if has_model or glb is not None else ((img * .35).astype(np.uint8), None)
        ys, xs = np.nonzero(mk)
        box = [xs.min(), ys.min(), xs.max(), ys.max()]
        if uv is not None:
            q = uv.reshape(-1, 2)
            q = q[(q[:, 0] > -W) & (q[:, 0] < 2 * W) & (q[:, 1] > -H) & (q[:, 1] < 2 * H)]
            if len(q):
                box = [min(box[0], q[:, 0].min()), min(box[1], q[:, 1].min()), max(box[2], q[:, 0].max()), max(box[3], q[:, 1].max())]
        pad = .25 * max(box[2] - box[0], box[3] - box[1], 40)
        x0, y0, x1, y1 = int(max(0, box[0] - pad)), int(max(0, box[1] - pad)), int(min(W, box[2] + pad)), int(min(H, box[3] + pad))
        left = img.copy()
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, cs, -1, (255, 255, 255), 3)
        cv2.drawContours(left, cs, -1, (0, 160, 255), 1)
        ch = int(round(320 * (y1 - y0) / max(x1 - x0, 1)))
        pair = np.hstack([cv2.resize(a[y0:y1, x0:x1], (320, max(ch, 1))) for a in (left, right)])
        pair = cv2.resize(pair, (640, 360)) if pair.shape[0] > 360 else np.vstack([pair, np.zeros((360 - pair.shape[0], 640, 3), np.uint8)])
        ph, ident = c.get("physical") or {}, c.get("identity") or {}
        pos = ph.get("position_xy") or {}
        lines = [f"#{k} {c['id']}  {str(ident.get('name'))[:44]}", type_line(c),
                 "  ".join(f"{lab} {short(ph.get(fld))}" for fld, lab in PHYS[:4]),
                 "  ".join(f"{lab} {short(ph.get(fld))}" for fld, lab in PHYS[4:]) + f"  pos {short(pos)}  @{(c.get('views') or {}).get('distance_m', ['?'])[0]} m",
                 rm.model_line(c, s)[:92]]
        band = np.zeros((6 + 20 * len(lines), 640, 3), np.uint8)
        for j, t in enumerate(lines):
            cv2.putText(band, t, (6, 18 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, .44, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([band, pair]))
        rows.append({"k": k, "card": c["id"], "frame": f["frame"], "name": ident.get("name"), "type": ident.get("type"),
                     "route": type_line(c).split("; route ")[-1], "model": (c.get("model") or {}).get("kind"), "sam3d": bool(s and s.get("accepted")),
                     **{x: None for x in LABELS}, "note": None})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s0 in range(0, len(tiles), per):
        t = tiles[s0:s0 + per]
        t += [np.zeros_like(t[0])] * (-len(t) % 2)
        cv2.imwrite(str(out_dir / f"sheet-{s0 // per}.jpg"), np.vstack([np.hstack(t[q:q + 2]) for q in range(0, len(t), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 78])
    (out_dir / "labels.json").write_text(json.dumps({"report": report, "seed": seed, "labelled_by": "agent (by looking at the sheets)",
                                                     "legend": LABELS, "rows": rows}, indent=1))
    return rows


def score(dirs):
    """labels.json per folder -> {folder: {label kind: Counter}} (unlabelled rows counted as None)."""
    out = {}
    for d in dirs:
        rows = json.loads((Path(d) / "labels.json").read_text())["rows"]
        out[Path(d).name] = {"n": len(rows), **{x: dict(Counter(r[x] for r in rows)) for x in LABELS},
                             "type_by_route": {rt: dict(Counter(r["type"] for r in rows if r["route"] == rt)) for rt in sorted({r["route"] for r in rows})}}
    return out


def share(L):
    """The final object cards of one report -> the acceptance shares (a family type, a specific name, a shape only; a pick region
    (clickable); every required physical field; a display model; SAM 3D)."""
    import r4_naming_results as nr
    from fast_report import cards as fc
    cs = [c for c in L["object_cards"]["cards"] if c.get("kind") == "object"]
    n = max(1, len(cs))
    ents = set((L.get("pick") or {}).get("entities") or [])
    alias = L["object_cards"].get("aliases") or {}
    clickable = {alias.get(e, e) for e in ents if e}
    req = fc.completeness(L["object_cards"]["cards"])
    fam = sum(nr.type_family(c["identity"]["name"]) is not None for c in cs)
    return {"object_cards": len(cs), "family_typed": round(fam / n, 3), "specific_name": round(sum(nr.is_named(c["identity"]["name"]) for c in cs) / n, 3),
            "shape_only": round(sum(nr.type_family(c["identity"]["name"]) is None for c in cs) / n, 3),
            "clickable": round(sum(c["id"] in clickable for c in cs) / n, 3),
            "model": round(sum(bool((c.get("model") or {}).get("kind")) for c in cs) / n, 3),
            "model_kinds": dict(Counter((c.get("model") or {}).get("kind") for c in cs)), "completeness": req}


def self_check():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rows = [{"k": 0, "route": "vlm", "type": "right", "outline": "right", "physical": "plausible", "model": "plausible"},
                {"k": 1, "route": "type only (family)", "type": "close", "outline": "partial", "physical": "unclear", "model": "absent"}]
        (Path(tmp) / "labels.json").write_text(json.dumps({"rows": rows}))
        s = score([tmp])[Path(tmp).name]
    assert s["n"] == 2 and s["type"] == {"right": 1, "close": 1} and s["model"] == {"plausible": 1, "absent": 1}
    assert s["type_by_route"] == {"type only (family)": {"close": 1}, "vlm": {"right": 1}}
    print("r4b_audit self-check ok: tallies per label kind and per naming route")


if __name__ == "__main__":
    import argparse
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        sys.exit()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("cards", "score", "share"))
    p.add_argument("args", nargs="+")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=29)
    a = p.parse_args()
    if a.cmd == "cards":
        cards(Path(a.args[0]), a.args[1], Path(a.args[2]), a.n, a.seed)
    elif a.cmd == "score":
        print(json.dumps(score(a.args), indent=1))
    else:
        import fast_report_eval as ev
        print(json.dumps(share(ev.load_layers(Path(a.args[0]), a.args[1])), indent=1, default=str))
