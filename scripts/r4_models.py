"""r4 (models): does every object card carry a display model, and does it look right from the camera that saw it?

    python scripts/r4_models.py sheet MIRROR REPORT OUT_DIR [--n 30 --seed 4] [--sam3d]   # contact sheets for the audit by eye
    python scripts/r4_models.py table RUN_DIR [...]                              # per report: models, residuals, SAM 3D, time, GPU
    python scripts/r4_models.py results RUN_DIR OUT [--baseline RUN_DIR] [--audit DIR ...]   # the tables as markdown
    python scripts/r4_models.py --self-check

sheet: n object cards drawn at random (seeded) among the final cards with a pick region, each on one of the card's best views
(else the keyframe with a camera where its region is largest): left the crop with the region outlined, right the same crop dimmed with the card's model drawn
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
    """A SAM 3D display GLB (fast_report.sam3d.judge: centred, placed by its transform; one primitive, POSITION + COLOR_0 RGBA
    uint8 + uint32 indices, as trimesh writes it: trimesh reads it back without its vertex colours) -> world triangles, RGBA per face."""
    import struct
    n = struct.unpack("<I", raw[12:16])[0]
    doc, binary = json.loads(raw[20:20 + n]), raw[20 + n + 8:]
    dt = {5121: np.uint8, 5125: np.uint32, 5126: np.float32, 5123: np.uint16}
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}

    def acc(i):
        a = doc["accessors"][i]
        v = doc["bufferViews"][a["bufferView"]]
        off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(binary, dt[a["componentType"]], a["count"] * width[a["type"]], off).reshape(a["count"], -1)
    prim = doc["meshes"][0]["primitives"][0]
    V = acc(prim["attributes"]["POSITION"]).astype(float) + np.asarray(position, float)
    cols = acc(prim["attributes"]["COLOR_0"]).astype(float) if "COLOR_0" in prim["attributes"] else np.full((len(V), 4), 200.)
    F = acc(prim["indices"]).reshape(-1, 3)
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
    if model_glb is not None:  # 40k triangles: painted opaque in depth order on one layer, blended once
        import cv2
        tris, rgba = model_glb
        uv, z = project(tris, K, c2w)
        layer, cover = dim.copy(), np.zeros(dim.shape[:2], np.uint8)
        for i in [i for i in np.argsort(-z.mean(1)) if (z[i] > .05).all()]:
            poly = np.round(uv[i]).astype(np.int32)
            cv2.fillConvexPoly(layer, poly, tuple(float(c) for c in rgba[i, 2::-1]))
            cv2.fillConvexPoly(cover, poly, 1)
        out = dim.copy()
        out[cover > 0] = (dim[cover > 0] * (1 - MESH_ALPHA) + layer[cover > 0] * MESH_ALPHA).astype(np.uint8)
        ok = (z > .05).all(1)
        return out, uv[ok] if ok.any() else None
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


def sheet(mirror, report, out_dir, n=30, seed=4, per=10, cards=None, only_sam3d=False):
    import cv2
    import fast_report_eval as ev
    L, pick, models_patch = load(mirror, report)
    root = owners(L)
    ent = pick.data["entities"]
    cams = {s["index"]: s for s in L["cameras"]["shots"]}
    cards = cards if cards is not None else [c for c in L["object_cards"]["cards"] if c.get("kind") == "object" and (c.get("model") or {}).get("kind")]
    if only_sam3d:  # every card whose SAM 3D mesh the gate accepted (the mesh replaces its primitive)
        acc = sam_by_card(L, models_patch)
        cards = [c for c in cards if acc.get(c["id"], {}).get("accepted")]
    want = {c["id"] for c in cards}
    best, chosen_view = {}, {c["id"]: set((c.get("views") or {}).get("best") or []) for c in cards}
    for i, f in enumerate(pick.frames):
        cam = cams.get(f.get("shot"))
        if f.get("source") != "segmented" or cam is None or f["frame"] not in cam["keys"]:
            continue
        m = pick.map(i)
        idx, cnt = np.unique(m[m > 0], return_counts=True)
        for j, a in zip(idx, cnt):
            cid = root(ent[int(j)]) if ent[int(j)] else None
            # the card's own best views first (coverage x sharpness, >= 15 deg apart), then where its region is largest
            rank = (f["frame"] in chosen_view.get(cid, ()), int(a))
            if cid in want and rank > best.get(cid, ((False, 0),))[0]:
                best[cid] = (rank, i)
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
        sx, sy = W / cam["wh"][0], H / cam["wh"][1]  # the DA3 grid to the source frame, per axis (ondemand.lift's mapping)
        K = np.asarray(cam["K"][key], float) * [[sx, 1, sx], [1, sy, sy], [1, 1, 1]]
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
        if "object_cards" not in L or not (Path(run_dir) / "mirror" / "reports" / rep / "run.json").exists():
            continue  # a call still running
        cs = [c for c in L["object_cards"]["cards"] if c.get("kind") == "object"]
        ms = [c.get("model") or {} for c in cs]
        have = [m for m in ms if m.get("kind")]
        res = np.array([m["residual_m"] for m in have]) if have else np.zeros(0)
        models = ev.patch_versions(Path(run_dir) / "mirror", rep, "models")
        sam = sam_by_card(L, models[-1] if models else None)
        run = json.loads((Path(run_dir) / "mirror" / "reports" / rep / "run.json").read_text()) if (Path(run_dir) / "mirror" / "reports" / rep / "run.json").exists() else {}
        marks = run.get("marks") or {}
        cards_s = {}
        for st in run.get("stages") or []:  # every cards build and its writes (a stage can run more than once: summed)
            if st["stage"].startswith("cards.") or st["stage"] == "write.object_cards":
                cards_s[st["stage"]] = round(cards_s.get(st["stage"], 0) + st["s"], 3)
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
                    "model_cpu_s": (L["object_cards"].get("stats") or {}).get("s", {}).get("models_cpu"),
                    "cards_stage_s": cards_s, "marks_s": {k: v for k, v in marks.items() if "cards" in k or "model" in k or "display" in k or "splat" in k},
                    "cards_bytes": sum(st["n"].get("bytes", 0) for st in run.get("stages") or [] if st["stage"] == "write.object_cards"),
                    "elapsed_s": run.get("elapsed_s"), "gpu_peak_gib": [g.get("peak_gb") for g in run.get("gpu_peak") or []],
                    "over_72": [g.get("peak_gb", 0) > 72 for g in run.get("gpu_peak") or []]}
    return out


def site_of(report):
    return report.split("-")[1] if not report.startswith("mvp-samsclub") else "samsclub-a2"


def results_md(t, base=None, audits=()):
    """Markdown from table() of the run (and of a baseline run: the cards' time without the models), plus audit labels."""
    rows = "| video | call | object cards | with a model | box / cylinder / plane / open frame | by type (overruled) | residual median / p90 cm " \
           "| seen share (median) | one side | SAM 3D eligible / tried / accepted | first model / final models s (display start) | analysis s | GPU peaks GiB |\n|" + "---|" * 13 + "\n"
    calls = {}
    for rep, r in sorted(t.items(), key=lambda kv: (site_of(kv[0]), kv[0].rsplit("-", 1)[1])):
        site = site_of(rep)
        calls[site] = calls.get(site, 0) + 1
        k, s3, mk = r["kinds"], r["sam3d"], r["marks_s"]
        rows += (f"| {site} | {'first' if calls[site] == 1 else 'warm'} | {r['object_cards']} | {r['with_primitive']} ({r['share']:.0%}) | "
                 f"{k['box']} / {k['cylinder']} / {k['plane']} / {k['open frame']} | {r['chosen_by_type']} ({r['type_overruled']}) | "
                 f"{r['residual_cm']['median']} / {r['residual_cm']['p90']} | {r['seen_share_median']} | {r['one_sided']} | "
                 f"{s3['eligible']} / {s3['attempted']} / {s3['accepted']} | {mk.get('first_model_put', '-')} / {mk.get('models_final_put', '-')} "
                 f"({mk.get('display_started', '-')}) | {r['elapsed_s']} | {r['gpu_peak_gib']}{' FLAG > 72' if any(r['over_72']) else ''} |\n")
    out = "## Models per call\n\n" + rows
    if base:
        out += "\n## Time added to the cards (this run vs the baseline run, same videos and calls; s)\n\n| video | call | cards.v1 | cards.v3 | " \
               "cards write | model fits CPU s (all processes) | cards v1 put | cards v3 put | cards bytes written (all versions) |\n|" + "---|" * 9 + "\n"
        by = {}
        for rep, r in base.items():
            by.setdefault(site_of(rep), []).append(r)
        seen = {}
        for rep, r in sorted(t.items(), key=lambda kv: (site_of(kv[0]), kv[0].rsplit("-", 1)[1])):
            site = site_of(rep)
            i = seen[site] = seen.get(site, -1) + 1
            b = (by.get(site) or [None] * 2)[min(i, len(by.get(site) or [0]) - 1)] or {}
            f = lambda d, k: d.get(k, "-")  # noqa: E731
            cs, bs = r["cards_stage_s"], b.get("cards_stage_s", {})
            out += (f"| {site} | {'first' if i == 0 else 'warm'} | {f(cs, 'cards.v1')} (base {f(bs, 'cards.v1')}) | {f(cs, 'cards.v3')} (base {f(bs, 'cards.v3')}) | "
                    f"{f(cs, 'write.object_cards')} (base {f(bs, 'write.object_cards')}) | {r['model_cpu_s']} | {f(r['marks_s'], 'cards_v1_put')} "
                    f"(base {f(b.get('marks_s', {}), 'cards_v1_put')}) | {f(r['marks_s'], 'cards_v3_put')} (base {f(b.get('marks_s', {}), 'cards_v3_put')}) | "
                    f"{r['cards_bytes'] / 1e6:.1f} MB (base {b.get('cards_bytes', 0) / 1e6:.1f}) |\n")
    if audits:
        out += "\n## Audit by eye: 30 random models per video (agent-labelled; crop | model from the same camera)\n\n| video | looked at | plausible | implausible | unclear |\n|---|---|---|---|---|\n"
        for a in audits:
            lab = json.loads(Path(a, "labels.json").read_text())
            ls = [r["label"] for r in lab["rows"]]
            out += f"| {site_of(lab['report'])} | {len(ls)} | {ls.count('plausible')} | {ls.count('implausible')} | {ls.count('unclear')} |\n"
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
    tab = {"mvp-me340-a-2": {"kinds": {"box": 3, "cylinder": 1, "plane": 0, "open frame": 1}, "sam3d": {"eligible": 5, "attempted": 4, "accepted": 1},
                             "marks_s": {"first_model_put": 99.}, "object_cards": 5, "with_primitive": 5, "share": 1., "chosen_by_type": 2, "type_overruled": 0,
                             "residual_cm": {"median": 1.2, "p90": 3.}, "seen_share_median": .5, "one_sided": 2, "elapsed_s": 200., "gpu_peak_gib": [60., 50.],
                             "over_72": [False, False], "cards_stage_s": {"cards.v1": 2.}, "model_cpu_s": 1.5, "cards_bytes": 2e6}}
    md = results_md(tab, {"mvp-me340-b-1": {**tab["mvp-me340-a-2"], "cards_stage_s": {"cards.v1": 1.5}}})
    assert "| me340 | first | 5 | 5 (100%) | 3 / 1 / 0 / 1 |" in md and "2.0 (base 1.5)" in md, md
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
    s.add_argument("--sam3d", action="store_true", help="only the cards whose SAM 3D mesh was accepted")
    t = sub.add_parser("table")
    t.add_argument("runs", nargs="+", type=Path)
    r = sub.add_parser("results")
    r.add_argument("run", type=Path)
    r.add_argument("out", type=Path, help="a new results folder: table.json, results-tables.md")
    r.add_argument("--baseline", type=Path)
    r.add_argument("--audit", nargs="*", default=[])
    a = p.parse_args()
    if a.cmd == "sheet":
        sheet(a.mirror, a.report, a.out, a.n, a.seed, only_sam3d=a.sam3d)
    elif a.cmd == "table":
        print(json.dumps({str(r): table(r) for r in a.runs}, indent=1, default=str))
    else:
        a.out.mkdir(parents=True, exist_ok=True)
        t, b = table(a.run), table(a.baseline) if a.baseline else None
        (a.out / "table.json").write_text(json.dumps({"run": t, "baseline": b}, indent=1, default=str))
        (a.out / "results-tables.md").write_text(results_md(t, b, a.audit))
        print((a.out / "results-tables.md").read_text())
