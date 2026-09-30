"""r5b integrate: does the 3D picture line up with the video? Every object card's SHOWN model (its generated model or a look-alike's
copy, else its primitive, else its observed surface: r5b_models.shown, the viewer's order) drawn from its best keyframe's camera
(the cameras layer: K on the DA3 grid mapped to the source frame, camera-to-world in the shot frame) against the object's own SAM 3
outline on that keyframe: silhouette IoU, and the offset between the two centroids in source px and in cm at the object's depth
(px x depth / fx). Per tier and overall: median, p10 (IoU) / p90 (offsets). The silhouette is the model's own, unoccluded; the outline
is what the video saw (an object behind another reads lower). Also the worst generated models as a contact sheet and overlay frames
(every shown model in its video colours over the frame, far first).

    python scripts/r5b_align.py ROOT REPORT OUT_DIR          # ROOT: a bench mirror or the layers Volume's root (the same layout)
    python scripts/r5b_align.py --self-check
Runs on Modal where the reports are (modal_apps/r5b_align_app.py); numpy + OpenCV + scipy.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import r5b_models as rm  # noqa: E402
import r5b_render as rr  # noqa: E402

SCALE = .5  # measured at half the source resolution (640 x 360)


def cover(T, c2w, K, wh):
    """World triangles -> the pixels they cover (no depth order needed for a silhouette)."""
    import cv2
    w, h = wh
    T = np.asarray(T, float).reshape(-1, 3, 3)
    X = (T - c2w[:3, 3]) @ c2w[:3, :3]
    ok = (X[..., 2] > .05).all(1)
    X = X[ok]
    uv = np.stack([K[0, 0] * X[..., 0] / X[..., 2] + K[0, 2], K[1, 1] * X[..., 1] / X[..., 2] + K[1, 2]], -1)
    uv = uv[np.isfinite(uv).all((1, 2)) & (np.abs(uv).max((1, 2)) < 1e5)]
    m = np.zeros((h, w), np.uint8)
    for t in np.round(uv * 8).astype(np.int32):
        cv2.fillConvexPoly(m, t, 1, cv2.LINE_8, 3)
    return m > 0


def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


def card_polys(card, outl, aliases):
    ids = [card["id"], *((card.get("physical") or {}).get("merged_from") or [])] + [a for a, b in aliases.items() if b == card["id"]]
    polys = {}
    for i in ids:
        for q, p in outl.get(i, {}).items():
            polys.setdefault(q, []).extend(p)
    return polys


def view_of(card, cams, polys):
    keys = cams[card["shot"]]["keys"]
    best = [int(k) for k in (card.get("views") or {}).get("best", []) if int(k) in keys and int(k) in polys]
    cand = [k for k in polys if k in keys]
    return best[0] if best else (max(cand, key=lambda k: len(polys[k])) if cand else None)


def camera(s, q, wh):
    v = s["keys"].index(q)
    return np.asarray(s["c2w"][v], float), rr.k_full(np.asarray(s["K"][v], float), s["wh"], wh)


def rows_of(root, report):
    """-> (rows, context) per object card with an outline on a keyframe with a camera."""
    import cv2
    P = rm.patches(root, report)
    cs, aliases = rm.final_cards(root, P)
    cams, outl = rm.cameras(P), rm.outlines(root, P)
    out, cache = [], {}
    for c in cs:
        if c.get("kind") != "object" or c.get("shot") not in cams:
            continue
        polys = card_polys(c, outl, aliases)
        q = view_of(c, cams, polys)
        if q is None:
            continue
        s = cams[c["shot"]]
        W, H = int(s["source_wh"][0] * SCALE), int(s["source_wh"][1] * SCALE)
        c2w, K = camera(s, q, (W, H))
        mask = np.zeros((H, W), np.uint8)
        for p in polys[q]:
            cv2.fillPoly(mask, [np.round(np.asarray(p, float).reshape(-1, 2) * SCALE).astype(np.int32)], 1)
        mask = mask > 0
        if not mask.any():
            continue
        tier, T, _ = rm.shown(c, root, P, cache)
        cov = cover(T, c2w, K, (W, H)) if len(T) else np.zeros_like(mask)
        inter, uni = (cov & mask).sum(), (cov | mask).sum()
        row = {"card": c["id"], "name": (c.get("identity") or {}).get("name"), "tier": tier, "view": q, "shot": c["shot"], "mask_px": int(mask.sum() / SCALE ** 2),
               "iou": round(float(inter / max(uni, 1)), 3), "coverage": round(float(inter / mask.sum()), 3),
               "overflow": round(float((cov & ~mask).sum() / max(cov.sum(), 1)), 3)}
        a, b = centroid(cov), centroid(mask)
        ph = c.get("physical") or {}
        if a is not None and ph.get("box_min_m"):
            ctr = (np.asarray(ph["box_min_m"], float) + np.asarray(ph["box_max_m"], float)) / 2
            z = float((ctr - c2w[:3, 3]) @ c2w[:3, 2])
            px = float(np.linalg.norm(a - b)) / SCALE  # source px
            row.update(offset_px=round(px, 1), depth_m=round(z, 2), offset_cm=round(100 * px * z / (K[0, 0] / SCALE), 1) if z > 0 else None)
        out.append(row)
    return out, {"P": P, "cards": cs, "aliases": aliases, "cams": cams, "outl": outl, "cache": cache}


def summary(rows):
    def stats(rs):
        iou = np.array([r["iou"] for r in rs], float)
        off = np.array([r["offset_px"] for r in rs if r.get("offset_px") is not None], float)
        cm = np.array([r["offset_cm"] for r in rs if r.get("offset_cm") is not None], float)
        q = lambda a, p: round(float(np.percentile(a, p)), 3) if len(a) else None  # noqa: E731
        return {"n": len(rs), "iou_median": q(iou, 50), "iou_p10": q(iou, 10), "iou_ge_0.5": round(float((iou >= .5).mean()), 3) if len(iou) else None,
                "offset_px_median": q(off, 50), "offset_px_p90": q(off, 90), "offset_cm_median": q(cm, 50), "offset_cm_p90": q(cm, 90)}
    out = {"all": stats(rows)}
    for t in sorted({r["tier"] for r in rows}):
        out[t] = stats([r for r in rows if r["tier"] == t])
    return out


def frames(root, P, qs):
    return rm.frames_at(rm.video_path(root, P), qs)


def worst_sheet(root, rows, ctx, n=12, tier="generated", side=240):
    """The n lowest-IoU objects of a tier: [crop with the outline | the shown model over it], 4 a row; -> jpg bytes (<= 1600 px)."""
    import cv2
    rs = sorted([r for r in rows if r["tier"] == tier], key=lambda r: r["iou"])[:n]
    if not rs:
        return None, []
    by = {c["id"]: c for c in ctx["cards"]}
    imgs = frames(root, ctx["P"], {r["view"] for r in rs})
    tiles = []
    for r in rs:
        c, s = by[r["card"]], ctx["cams"][r["shot"]]
        img = imgs.get(r["view"])
        if img is None:
            continue
        c2w, K = camera(s, r["view"], tuple(s["source_wh"]))
        mask = np.zeros(img.shape[:2], np.uint8)
        for p in card_polys(c, ctx["outl"], ctx["aliases"])[r["view"]]:
            cv2.fillPoly(mask, [np.round(np.asarray(p, float).reshape(-1, 2)).astype(np.int32)], 1)
        _, T, _ = rm.shown(c, root, ctx["P"], ctx["cache"])
        cov = cover(T, c2w, K, tuple(s["source_wh"]))
        box = rr.crop_box((mask > 0) | cov)
        a = rr.label(rr.cut(rr.outline(img, mask > 0), box, side), f"{str(r['name'])[:24]}", y=14)
        b = rr.label(rr.cut(rr.outline(rr.overlay(img, cov, tint=(80, 220, 120)), mask > 0), box, side), f"IoU {r['iou']:.2f} off {r.get('offset_cm')} cm")
        tiles.append(np.hstack([a, b]))
    rows_img = [np.hstack(tiles[i:i + 3] + [np.full_like(tiles[0], 30)] * (3 - len(tiles[i:i + 3]))) for i in range(0, len(tiles), 3)]
    sheet = np.vstack(rows_img)
    return cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), [r["card"] for r in rs]


def overlay_frame(root, rows, ctx, q, shot, alpha=.7):
    """Every object's shown model on keyframe q (its video colours; primitives orange), far first, over the frame -> jpg bytes."""
    import cv2
    s = ctx["cams"][shot]
    img = frames(root, ctx["P"], [q]).get(q)
    if img is None:
        return None
    c2w, K = camera(s, q, tuple(s["source_wh"]))
    by = {c["id"]: c for c in ctx["cards"]}
    tris, cols = [], []
    for c in by.values():
        if c.get("kind") != "object" or c.get("shot") != shot:
            continue
        tier, T, _ = rm.shown(c, root, ctx["P"], ctx["cache"])
        if not len(T):
            continue
        col = model_colors(c, root, ctx, tier, len(T))
        tris.append(T)
        cols.append(col)
    if not tris:
        return None
    cov, rgb = rr.raster(np.concatenate(tris), c2w, K, tuple(s["source_wh"]), colors=np.concatenate(cols))
    out = img.copy()
    out[cov] = (img[cov] * (1 - alpha) + rgb[cov][:, ::-1] * alpha).astype(np.uint8)
    cs_, _ = cv2.findContours(cov.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs_, -1, (255, 255, 255), 1, cv2.LINE_AA)
    scale = min(1., 1600 / out.shape[1])
    if scale < 1:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def model_colors(card, root, ctx, tier, n):
    """Per-triangle RGB of the shown model: the generated GLB's / the surface node's vertex colours, else a flat tint."""
    import r4_models
    P = ctx["P"]
    if tier == "generated":
        models = P["models"][-1]
        r = next(x for x in models["data"]["models"] if x["object"] == card["id"])
        rep = r.get("reuse_of") or card["id"]
        raw = rm.blob(root, models["blobs"][f"model-{rep}"])
        base = next(x for x in models["data"]["models"] if x["object"] == rep)["transform"]["position"]
        _, rgba = r4_models.glb_tris(raw, base)
        return rgba[:, :3]
    if tier == "observed surface":
        sf = next(x for x in P["surfaces"][-1]["data"]["surfaces"] if x["object"] == card["id"])
        V, F, C = ctx["cache"][(P["surfaces"][-1]["seq"], sf["blob"])][card["id"]]
        if C is not None:
            return C[F][..., :3].mean(1)
    return np.tile([255., 176, 70], (n, 1))


def pick_overlays(rows, k=2):
    """The k keyframes (different shots when there are) where the most objects with a generated model were measured."""
    from collections import Counter
    cnt = Counter((r["view"], r["shot"]) for r in rows if r["tier"] == "generated")
    out, shots = [], set()
    for (q, s), _ in cnt.most_common():
        if s not in shots or len(shots) >= len({x[1] for x in cnt}):
            out.append((q, s))
            shots.add(s)
        if len(out) == k:
            break
    return out


def run(root, report, out_dir=None):
    rows, ctx = rows_of(root, report)
    res = {"report": report, "rule": __doc__.split("\n\n")[0], "summary": summary(rows), "rows": rows}
    sheet, worst = worst_sheet(root, rows, ctx)
    res["worst_generated"] = worst
    overlays = {}
    for q, s in pick_overlays(rows):
        jpg = overlay_frame(root, rows, ctx, q, s)
        if jpg:
            overlays[f"overlay-shot{s}-frame{q}.jpg"] = jpg
    res["overlays"] = sorted(overlays)
    files = {**({"worst-generated.jpg": sheet} if sheet else {}), **overlays}
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for k, v in files.items():
            (Path(out_dir) / k).write_bytes(v)
        (Path(out_dir) / "alignment.json").write_text(json.dumps(res, indent=1))
    return res, files


def self_check():
    K = np.array([[500., 0, 320], [0, 500., 180], [0, 0, 1]])
    c2w = np.eye(4)
    sq = np.array([[[-.5, -.5, 5.], [.5, -.5, 5.], [.5, .5, 5.]], [[-.5, -.5, 5.], [.5, .5, 5.], [-.5, .5, 5.]]])
    m = cover(sq, c2w, K, (640, 360))
    assert abs(m.sum() - 100 * 100) < 300 and m[180, 320] and not m[10, 10]
    assert np.allclose(centroid(m), [320, 180], atol=1.)
    s = summary([{"tier": "generated", "iou": .8, "offset_px": 4., "offset_cm": 2.}, {"tier": "primitive", "iou": .4, "offset_px": 10., "offset_cm": 5.}])
    assert s["all"]["n"] == 2 and s["generated"]["iou_median"] == .8 and s["all"]["iou_ge_0.5"] == .5
    print("r5b_align self-check ok: silhouette cover, centroid, summary")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    else:
        r, _ = run(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
        print(json.dumps(r["summary"], indent=1))
