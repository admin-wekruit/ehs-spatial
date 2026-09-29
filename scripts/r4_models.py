"""r4 (models): does every object card carry a display model, and does it look right from the camera that saw it?

    python scripts/r4_models.py sheet MIRROR REPORT OUT_DIR [--n 30 --seed 4]   # contact sheets for the audit by eye
    python scripts/r4_models.py table RUN_DIR [...]                              # per report: models, residuals, SAM 3D, time, GPU
    python scripts/r4_models.py --self-check

sheet: n object cards drawn at random (seeded) among the final cards with a pick region, each on the keyframe (with a camera)
where its region is largest: left the crop with the region outlined, right the same crop dimmed with the card's model drawn
from that camera (the SAM 3D mesh when its gate accepted one, else the primitive: seen faces solid, guessed ones faint).
labels.json lists them for the agent's labels (plausible | implausible | unclear, by looking; 'agent-labelled').
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
FACE_AXIS = {"-x": (0, -1), "+x": (0, 1), "-y": (1, -1), "+y": (1, 1), "-z": (2, -1), "+z": (2, 1)}
SEEN_BGR, GUESS_BGR, MESH_ALPHA = (60, 200, 255), (255, 170, 60), .85


def box_tris(lo, hi, seen):
    """12 triangles of the box [lo, hi]; seen: {face name: bool} -> (vertices (36, 3), seen (12,))."""
    tris, flags = [], []
    for name, (k, s) in FACE_AXIS.items():
        a, b = [i for i in range(3) if i != k]
        q = []
        for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
            p = np.empty(3)
            p[k] = hi[k] if s > 0 else lo[k]
            p[a], p[b] = (lo[a], hi[a])[u], (lo[b], hi[b])[v]
            q.append(p)
        tris += [q[0], q[1], q[2], q[0], q[2], q[3]]
        flags += [seen[name]] * 2
    return np.array(tris), np.array(flags)


def model_tris(m):
    """A card's model record -> world triangles (n, 3, 3) and seen flags (n,) (the viewer draws the same)."""
    from scipy.spatial.transform import Rotation
    R, t = Rotation.from_quat(m["quaternion"]).as_matrix(), np.asarray(m["position"], float)
    if m["kind"] == "cylinder":
        r, h, n = m["radius_m"], m["length_m"], len(m["arc_seen"])
        ang = np.radians(np.arange(n + 1) * 360 / n)
        ring = lambda z: np.c_[r * np.cos(ang), r * np.sin(ang), np.full(n + 1, z)]  # noqa: E731
        lo, hi = ring(-h / 2), ring(h / 2)
        tris, flags = [], []
        for i in range(n):
            seen = m["arc_seen"][i] == "1"
            tris += [lo[i], lo[i + 1], hi[i + 1], lo[i], hi[i + 1], hi[i]]
            flags += [seen, seen]
            for c, ring_, cap in ((-h / 2, lo, "bottom"), (h / 2, hi, "top")):
                tris += [np.array([0, 0, c]), ring_[i], ring_[i + 1]]
                flags.append(m["caps"][cap] == "seen")
        V, F = np.array(tris), np.array(flags)
    elif m["kind"] == "open frame":
        V, F = [], []
        for cx, cy, cz, sx, sy, sz, seen in m["parts"]:
            c, s = np.array([cx, cy, cz]), np.array([sx, sy, sz])
            v, f = box_tris(c - s / 2, c + s / 2, {k: bool(seen) for k in FACE_AXIS})
            V.append(v)
            F.append(f)
        V, F = np.concatenate(V), np.concatenate(F)
    else:
        s = np.asarray(m["size_m"], float)
        V, F = box_tris(-s / 2, s / 2, {k: v == "seen" for k, v in m["faces"].items()})
    return (V @ R.T + t).reshape(-1, 3, 3), F


def glb_tris(raw, position):
    """A SAM 3D display GLB (fast_report.sam3d.judge: centred, placed by its transform) -> world triangles and RGBA per face."""
    import io
    import trimesh
    mesh = trimesh.load(io.BytesIO(raw), file_type="glb", force="mesh", process=False)
    V = np.asarray(mesh.vertices) + np.asarray(position, float)
    cols = np.asarray(mesh.visual.vertex_colors, float) if hasattr(mesh.visual, "vertex_colors") else np.full((len(V), 4), 200.)
    F = np.asarray(mesh.faces)
    return V[F], cols[F].mean(1)


def project(P, K, c2w):
    cam = (np.asarray(P, float) - c2w[:3, 3]) @ c2w[:3, :3]
    z = cam[..., 2]
    uv = np.stack([K[0, 0] * cam[..., 0] / np.where(z > 1e-6, z, 1e-6) + K[0, 2], K[1, 1] * cam[..., 1] / np.where(z > 1e-6, z, 1e-6) + K[1, 2]], -1)
    return uv, z


def draw(img, tris, K, c2w, colors, alphas, edges=True):
    """Painter's order (far first), each triangle blended at its own alpha; triangles behind the camera are left out."""
    import cv2
    uv, z = project(tris, K, c2w)
    ok = (z > .05).all(1)
    order = [i for i in np.argsort(-z.mean(1)) if ok[i]]
    out = img.astype(np.float32)
    for i in order:
        poly = np.round(uv[i] * 8).astype(np.int32)
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(m, [poly], 1, cv2.LINE_AA, shift=3)
        sel = m > 0
        out[sel] = out[sel] * (1 - alphas[i]) + np.asarray(colors[i], np.float32) * alphas[i]
        if edges:
            cv2.polylines(out, [poly], True, tuple(float(c) for c in colors[i]), 1, cv2.LINE_AA, shift=3)
    return out.clip(0, 255).astype(np.uint8), uv[ok] if ok.any() else None


def render(img, card, model_glb, K, c2w):
    """The card's model drawn on a dimmed copy of img: the accepted SAM 3D mesh, else the primitive."""
    dim = (img * .35).astype(np.uint8)
    if model_glb is not None:
        tris, rgba = model_glb
        return draw(dim, tris, K, c2w, rgba[:, 2::-1], np.full(len(tris), MESH_ALPHA), edges=False)
    tris, seen = model_tris(card["model"])
    return draw(dim, tris, K, c2w, [SEEN_BGR if s else GUESS_BGR for s in seen], np.where(seen, .55, .18))


def model_line(card, sam):
    m = card.get("model") or {}
    if not m.get("kind"):
        return "model: none (" + (m.get("reason") or "no record") + ")"
    return (f"{'SAM 3D mesh (gate accepted)' if sam and sam.get('accepted') else m['kind']}; {m['chosen_by'][:34]}; "
            f"res {m['residual_m'] * 100:.1f} cm; seen {m.get('seen_share', 0):.0%}")


def load(mirror, report):
    """The final cards, pick, cameras and models layer of one mirrored report."""
    import fast_report_eval as ev
    L = ev.load_layers(mirror, report)
    pick = ev.run_picks(mirror, report, L)[-1][1]
    models = ev.patch_versions(mirror, report, "models")
    return L, pick, (models[-1] if models else None)


def owners(L):
    alias = (L.get("object_cards") or {}).get("aliases") or {}
    root = lambda e: root(alias[e]) if e in alias and alias[e] != e else e  # noqa: E731
    return root


def sam_by_card(L, models_patch):
    """{card id: {'accepted': bool, ...}} from the models layer's tried list (merged-away ids follow the cards' aliases)."""
    if not models_patch:
        return {}
    root, out = owners(L), {}
    for t in models_patch["data"].get("tried") or []:
        out.setdefault(root(t["object"]), t)
    for m in models_patch["data"].get("models") or []:
        out[root(m["object"])] = {**out.get(root(m["object"]), {}), "accepted": True, "object": m["object"], "transform": m["transform"]}
    return out


def sheet(mirror, report, out_dir, n=30, seed=4, per=10, cards=None):
    import cv2
    import fast_report_eval as ev
    L, pick, models_patch = load(mirror, report)
    root = owners(L)
    ent = pick.data["entities"]
    cams = {s["index"]: s for s in L["cameras"]["shots"]}
    cards = cards if cards is not None else [c for c in L["object_cards"]["cards"] if c.get("kind") == "object" and (c.get("model") or {}).get("kind")]
    want = {c["id"] for c in cards}
    best = {}
    for i, f in enumerate(pick.frames):
        cam = cams.get(f.get("shot"))
        if f.get("source") != "segmented" or cam is None or f["frame"] not in cam["keys"]:
            continue
        m = pick.map(i)
        idx, cnt = np.unique(m[m > 0], return_counts=True)
        for j, a in zip(idx, cnt):
            cid = root(ent[int(j)]) if ent[int(j)] else None
            if cid in want and a > best.get(cid, (0,))[0]:
                best[cid] = (int(a), i)
    cards = [c for c in cards if c["id"] in best]
    chosen = [cards[i] for i in np.random.default_rng(seed).choice(len(cards), min(n, len(cards)), replace=False)]
    sam = sam_by_card(L, models_patch)
    video = next(Path(mirror).rglob(L["video"]["blob_sha256"])) if L.get("video", {}).get("blob_sha256") else None
    if video is None:
        vp = ev.patch_versions(mirror, report, "video")[-1]
        video = Path(mirror) / "blobs" / "sha256" / vp["blobs"]["video"]["sha256"]
    cap = cv2.VideoCapture(str(video))
    tiles, rows = [], []
    for k, c in enumerate(chosen):
        i = best[c["id"]][1]
        f = pick.frames[i]
        cam = cams[f["shot"]]
        key = cam["keys"].index(f["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame"])
        ok, img = cap.read()
        H, W = img.shape[:2]
        sx = W / cam["wh"][0]
        K = np.asarray(cam["K"][key], float) * [[sx, 1, sx], [1, sx, sx], [1, 1, 1]]
        c2w = np.asarray(cam["c2w"][key], float)
        mk = np.isin(pick.map(i), [j for j, e in enumerate(ent) if e and root(e) == c["id"]]).astype(np.uint8)
        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
        s = sam.get(c["id"])
        glb = None
        if s and s.get("accepted") and models_patch and f"model-{s['object']}" in models_patch["blobs"]:
            glb = glb_tris(ev.blob_bytes(mirror, models_patch["blobs"][f"model-{s['object']}"]["sha256"]), s["transform"]["position"])
        right, uv = render(img, c, glb, K, c2w)
        ys, xs = np.nonzero(mk)
        box = [xs.min(), ys.min(), xs.max(), ys.max()]
        if uv is not None:
            q = uv.reshape(-1, 2)
            q = q[(q[:, 0] > -W) & (q[:, 0] < 2 * W) & (q[:, 1] > -H) & (q[:, 1] < 2 * H)]
            if len(q):
                box = [min(box[0], q[:, 0].min()), min(box[1], q[:, 1].min()), max(box[2], q[:, 0].max()), max(box[3], q[:, 1].max())]
        pad = .25 * max(box[2] - box[0], box[3] - box[1], 40)
        x0, y0 = int(max(0, box[0] - pad)), int(max(0, box[1] - pad))
        x1, y1 = int(min(W, box[2] + pad)), int(min(H, box[3] + pad))
        left = img.copy()
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, cs, -1, (255, 255, 255), 3)
        cv2.drawContours(left, cs, -1, (0, 160, 255), 1)
        ch = int(round(320 * (y1 - y0) / max(x1 - x0, 1)))
        pair = np.hstack([cv2.resize(a[y0:y1, x0:x1], (320, max(ch, 1))) for a in (left, right)])
        pair = cv2.resize(pair, (640, min(360, pair.shape[0]))) if pair.shape[0] > 360 else np.vstack([pair, np.zeros((360 - pair.shape[0], 640, 3), np.uint8)])
        ident = c.get("identity") or {}
        lines = [f"#{k} {c['id']} {str(ident.get('name'))[:36]}", model_line(c, s),
                 "SAM 3D: " + ("accepted" if s and s.get("accepted") else f"rejected ({'; '.join(s.get('reasons') or [])[:60]})" if s and "reasons" in s
                               else (s.get("why") or "not tried")[:70] if s else "not tried")]
        band = np.zeros((6 + 20 * len(lines), 640, 3), np.uint8)
        for j, t in enumerate(lines):
            cv2.putText(band, t, (6, 18 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([band, pair]))
        m = c["model"]
        rows.append({"k": k, "card": c["id"], "name": ident.get("name"), "frame": f["frame"], "kind": m["kind"], "chosen_by": m["chosen_by"],
                     "residual_m": m["residual_m"], "seen_share": m.get("seen_share"), "sam3d": bool(s and s.get("accepted")),
                     "label": None, "note": None})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s0 in range(0, len(tiles), per):
        t = tiles[s0:s0 + per]
        t += [np.zeros_like(t[0])] * (-len(t) % 2)
        cv2.imwrite(str(out_dir / f"sheet-{s0 // per}.jpg"), np.vstack([np.hstack(t[q:q + 2]) for q in range(0, len(t), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 78])
    (out_dir / "labels.json").write_text(json.dumps({"report": report, "seed": seed, "labelled_by": "agent (by looking at the sheets)",
                                                     "legend": "right: the model from the same camera; orange solid = seen faces, blue faint = guessed; "
                                                               "a coloured mesh = SAM 3D's accepted mesh", "rows": rows}, indent=1))
    return rows


def table(run_dir):
    """Per mirrored report: object cards with a model, kinds, how chosen, residuals, SAM 3D tried / accepted (on cards), the
    cards' times, the first and final models, GPU peaks."""
    import fast_report_eval as ev
    out = {}
    for rep in sorted(p.name for p in (Path(run_dir) / "mirror" / "reports").iterdir()):
        L = ev.load_layers(Path(run_dir) / "mirror", rep)
        cs = [c for c in L["object_cards"]["cards"] if c.get("kind") == "object"]
        ms = [c.get("model") or {} for c in cs]
        have = [m for m in ms if m.get("kind")]
        res = np.array([m["residual_m"] for m in have]) if have else np.zeros(0)
        models = ev.patch_versions(Path(run_dir) / "mirror", rep, "models")
        sam = sam_by_card(L, models[-1] if models else None)
        run = json.loads((Path(run_dir) / "mirror" / "reports" / rep / "run.json").read_text()) if (Path(run_dir) / "mirror" / "reports" / rep / "run.json").exists() else {}
        marks = {k: (v or {}).get("t") if isinstance(v, dict) else v for k, v in (run.get("marks") or {}).items()}
        stages = run.get("stages") or []
        cards_s = {s["name"]: round(s["end"] - s["start"], 2) for s in stages if isinstance(s, dict) and str(s.get("name", "")).startswith("cards.")} \
            if stages and isinstance(stages, list) else {}
        fp = ((run.get("summary") or {}).get("sam3d") or {})
        out[rep] = {"object_cards": len(cs), "with_primitive": len(have), "share": round(len(have) / max(len(cs), 1), 4),
                    "kinds": {k: sum(m["kind"] == k for m in have) for k in ("box", "cylinder", "plane", "open frame")},
                    "chosen_by_type": sum(m["chosen_by"].startswith("type") for m in have),
                    "type_overruled": sum(m["chosen_by"].startswith("residual: a") for m in have),
                    "residual_cm": {"median": round(float(np.median(res)) * 100, 2) if len(res) else None,
                                    "p90": round(float(np.percentile(res, 90)) * 100, 2) if len(res) else None},
                    "seen_share_median": round(float(np.median([m.get("seen_share", 0) for m in have])), 2) if have else None,
                    "one_sided": sum("depth" in m for m in have),
                    "sam3d": {"tried_on_cards": sum(1 for c in cs if c["id"] in sam), "accepted_on_cards": sum(bool(sam.get(c["id"], {}).get("accepted")) for c in cs),
                              "attempted": fp.get("attempted"), "accepted": fp.get("accepted"), "ranked": fp.get("ranked"),
                              "prepare_rejected": fp.get("prepare_rejected"), "eligible": fp.get("eligible")},
                    "model_cpu_s": (L["object_cards"].get("stats") or {}).get("s", {}).get("models"),
                    "cards_stage_s": cards_s, "marks_s": {k: v for k, v in marks.items() if k and ("cards" in k or "model" in k or "display" in k)},
                    "gpu_peak_gib": [g.get("peak_gb") for g in run.get("gpu_peak") or []]}
    return out


def self_check():
    from scipy.spatial.transform import Rotation
    from fast_report import display_model as dm
    rng = np.random.default_rng(1)
    # a box model 3 m in front of a camera looking down +z: its triangles project inside the frame, around the box's centre
    m = {"kind": "box", "size_m": [.6, .4, .5], "faces": {k: "seen" if k in ("-z", "+y") else "guessed" for k in FACE_AXIS},
         "position": [0., 0., 3.], "quaternion": [0, 0, 0, 1]}
    tris, seen = model_tris(m)
    assert tris.shape == (12, 3, 3) and seen.sum() == 4
    K, c2w = np.array([[500., 0, 320], [0, 500., 180], [0, 0, 1]]), np.eye(4)
    uv, z = project(tris.reshape(-1, 3), K, c2w)
    assert np.allclose(uv.mean(0), [320, 180], atol=1) and (z > 2.7).all() and uv[:, 0].max() - uv[:, 0].min() < 500 * .6 / 2.7 + 1
    img, _ = draw(np.zeros((360, 640, 3), np.uint8), tris, K, c2w, [SEEN_BGR if s else GUESS_BGR for s in seen], np.where(seen, .55, .18))
    assert img[180, 320].sum() > 0 and img[5, 5].sum() == 0
    # every kind the fitter makes renders: a fitted drum and shelf from display_model's own record
    th = rng.uniform(-np.pi, 0, 800)
    D = np.c_[.3 * np.cos(th), .3 * np.sin(th) + 3, rng.uniform(0, .9, 800)]
    fr = {"R": np.eye(3), "origin": np.zeros(3)}
    fs = dm.fits(D, np.array([[0., 0., 1.6]]))
    for kind in ("cylinder", "open frame", "plane", "box"):
        t, s = model_tris(dm.record(fs, kind, "test", fr))
        assert len(t) == len(s) and np.isfinite(t).all()
        assert np.allclose(t.reshape(-1, 3).mean(0)[:2], [0, 3], atol=.35), (kind, t.reshape(-1, 3).mean(0))
    # the model line and a SAM 3D card link through the cards' aliases
    L = {"object_cards": {"aliases": {"a": "b", "b": "b"}}}
    got = sam_by_card(L, {"data": {"tried": [{"object": "a", "accepted": False, "reasons": ["x"]}], "models": []}})
    assert got == {"b": {"object": "a", "accepted": False, "reasons": ["x"]}}
    assert model_line({"model": dm.record(fs, "cylinder", "type (drum)", fr)}, None).startswith("cylinder; type (drum)")
    assert Rotation.from_quat(dm.record(fs, "box", "t", fr)["quaternion"]).as_matrix()[2, 2] > .999  # a gravity box stays upright
    print("r4_models self-check ok: box / cylinder / frame / plane triangles, projection, painter's draw, SAM 3D links, model line")


if __name__ == "__main__":
    import argparse
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        sys.exit()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("mirror", type=Path)
    s.add_argument("report")
    s.add_argument("out", type=Path)
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=4)
    t = sub.add_parser("table")
    t.add_argument("runs", nargs="+", type=Path)
    a = p.parse_args()
    if a.cmd == "sheet":
        sheet(a.mirror, a.report, a.out, a.n, a.seed)
    else:
        print(json.dumps({str(r): table(r) for r in a.runs}, indent=1, default=str))
