"""r5b integrate: the timeline audit by eye. n object cards (every card with a change claim first, then seeded random cards seen in
>= 2 content windows): per card one strip, a tile per content window (at most 6) on the window's keyframe where the card's pick
region is largest (outlined; a window where it has none shows the middle keyframe with 'no region'), labelled with the timeline's
state and values there. labels.json lists the cards for the agent's labels ('agent-labelled'): states right | wrong | unclear,
values plausible | implausible | unclear.

    python scripts/r5b_timeline_strips.py MIRROR REPORT OUT [--n 5 --seed 31]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]


def main(mirror, report, out, n=5, seed=31, side=220):
    import cv2
    import fast_report_eval as ev
    import r5b_models as rm
    import r5b_render as rr
    import r5b_timeline_eval as te
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    L = ev.load_layers(Path(mirror), report)
    cs = [c for c in L["object_cards"]["cards"] if c.get("kind") == "object" and ((c.get("time") or {}).get("timeline") or {}).get("windows")]
    aliases = L["object_cards"].get("aliases") or {}
    claimed = {c["card"] for c in te.claims_of(L["object_cards"]["cards"])} if te.claims_of(L["object_cards"]["cards"]) else set()
    rng = np.random.default_rng(seed)
    multi = [c for c in cs if c["id"] not in claimed and len(c["time"]["timeline"]["windows"]) >= 2]
    pick = [c for c in cs if c["id"] in claimed] + [multi[i] for i in rng.choice(len(multi), min(n, len(multi)), replace=False)]
    pick = pick[:max(n, len(claimed))]
    pk = ev.run_picks(Path(mirror), report, L)[-1][1]
    ent = pk.data["entities"]
    frames = [f for f in pk.frames if f.get("frame") is not None]
    P = rm.patches(mirror, report)
    need, plan = set(), []
    for c in pick:
        ids = {c["id"], *((c.get("physical") or {}).get("merged_from") or []), *[a for a, b in aliases.items() if b == c["id"]]}
        codes = [k for k, e in enumerate(ent) if e in ids]
        row = []
        for w in c["time"]["timeline"]["windows"][:6]:
            inside = [i for i, f in enumerate(pk.frames) if w["t"][0] <= f["t"] <= w["t"][1] and f.get("shot") == c["shot"]]
            best, area = None, 0
            for i in inside[::2]:
                a = int(np.isin(pk.map(i), codes).sum()) if codes else 0
                if a > area:
                    best, area = i, a
            i = best if best is not None else (inside[len(inside) // 2] if inside else None)
            if i is not None:
                need.add(int(pk.frames[i]["frame"]))
            row.append((w, i, area))
        plan.append((c, codes, row))
    imgs = rm.frames_at(rm.video_path(mirror, P), need)
    strips, labels = [], []
    for c, codes, row in plan:
        tiles = []
        for w, i, area in row:
            if i is None or int(pk.frames[i]["frame"]) not in imgs:
                tiles.append(np.full((side, side, 3), 40, np.uint8))
                continue
            img = imgs[int(pk.frames[i]["frame"])]
            m = cv2.resize(np.isin(pk.map(i), codes).astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            box = rr.crop_box(m) if m.any() else (0, 0, img.shape[1], img.shape[0])
            t = rr.cut(rr.outline(img, m) if m.any() else img, box, side)
            v = w.get("v") or {}
            txt = f"{w['t'][0]:.1f}-{w['t'][1]:.1f}s {w.get('state')}"
            tiles.append(rr.label(rr.label(t, txt[:34], y=14), (f"top {v['top_above_floor'][0]:.2f}" if "top_above_floor" in v else "no values")
                                  + ("" if m.any() else " | no region")))
        strip = np.hstack(tiles + [np.full((side, side, 3), 30, np.uint8)] * (6 - len(tiles)))
        head = np.full((22, strip.shape[1], 3), 20, np.uint8)
        rr.label(head, f"{c['id']} {str((c.get('identity') or {}).get('name'))[:40]} changes: {len(c['time']['timeline'].get('changes') or [])}", y=16, scale=.5)
        strips.append(np.vstack([head, strip]))
        labels.append({"card": c["id"], "name": (c.get("identity") or {}).get("name"), "windows": [{"t": w["t"], "state": w.get("state"), "reason": w.get("reason")}
                       for w, _, _ in row], "changes": c["time"]["timeline"].get("changes"), "states": None, "values": None, "note": ""})
    cv2.imwrite(str(out / "timelines.jpg"), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 82])
    lp = out / "labels.json"
    if not lp.exists():
        lp.write_text(json.dumps({"labeller": "agent-labelled", "report": report, "claims_in_report": len(claimed), "rows": labels}, indent=1))
    print(json.dumps({"cards": len(plan), "claims": len(claimed)}))


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {"n": int(a[a.index("--n") + 1]) if "--n" in a else 5, "seed": int(a[a.index("--seed") + 1]) if "--seed" in a else 31}
    main(a[0], a[1], a[2], **kw)
