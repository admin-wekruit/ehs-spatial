"""r5b time: the timeline evaluation where the reports are (the layers Volume): the Mac's disk is under the 8 GB it may pull
at, so the cards and pick blobs of a run are read on Modal and only the numbers and small masks come back.

Per report: the cards' change claims (scripts/r5b_timeline_eval.claims_of), the place-test judgements (places), the cards'
timeline stats; per claim the claimed card's pick region on its evidence keyframes (PNG, for the local claim sheets); for a
planted call, each planted event against the claims (the box outline drawn on the Mac with DROID's cameras, sent as polygons
per source frame: IoU >= IOU_MIN on the claim's evidence keyframe, the kind and time agree) and which cards cover the box.
CPU only, one ephemeral container.

    modal run modal_apps/r5b_eval_app.py --inp IN.json --out OUT.json
"""
import base64
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("panoptes-r5b-eval")
image = (modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "opencv-python-headless")
         .add_local_file(HERE.parent / "scripts" / "r5b_timeline_eval.py", "/repo/scripts/r5b_timeline_eval.py"))
LAYERS = modal.Volume.from_name("panoptes-fb-layers")


def latest(folder):
    out = {}
    for p in sorted(Path(folder).glob("*.json")):
        x = json.loads(p.read_text())
        if x["layer"] not in out or (x["version"], x["seq"]) > (out[x["layer"]]["version"], out[x["layer"]]["seq"]):
            out[x["layer"]] = x
    return out


class Pick:
    """fast_report_eval.Pick's decode (a copy: that module imports the Modal apps chain): one id map per pick frame."""

    def __init__(self, data, raw):
        import numpy as np
        self.data, self.frames = data, data["frames"]
        self.pairs = np.frombuffer(raw, "<u2").reshape(-1, 2)
        n = 0
        for f in self.frames:
            f["_start"], n = n, n + f["pairs"]

    def map(self, i):
        import numpy as np
        f = self.frames[i]
        p = self.pairs[f["_start"]:f["_start"] + f["pairs"]]
        return np.repeat(p[:, 0], p[:, 1].astype(np.int64)).reshape(f["h"], f["w"])


@app.function(image=image, volumes={"/v/layers": LAYERS}, cpu=4, memory=16384, timeout=1800, retries=0)
def evaluate(inp):
    import gzip
    import cv2
    import numpy as np
    sys.path.insert(0, "/repo/scripts")
    import r5b_timeline_eval as E
    root, out = Path("/v/layers"), {}
    blob = lambda sha: (root / "blobs" / "sha256" / sha).read_bytes()  # noqa: E731
    for rep, meta in inp["reports"].items():
        L = latest(root / "reports" / rep / "patches")
        oc = L["object_cards"]
        cards = oc["data"]["cards"] if isinstance(oc["data"]["cards"], list) else json.loads(blob(oc["blobs"]["cards"]["sha256"]))
        pk = L["pick"]
        raw = b"".join(gzip.decompress(blob(pk["blobs"][c["blob"]]["sha256"])) for c in pk["data"]["chunks"]) if pk["data"].get("chunks") \
            else gzip.decompress(blob(pk["blobs"]["pick"]["sha256"]))
        pick = Pick(pk["data"], raw)
        aliases = oc["data"].get("aliases") or {}
        ent = pick.data["entities"]
        at = {f["frame"]: i for i, f in enumerate(pick.frames)}

        def mask(key, ids):
            i = at.get(key)
            codes = [k for k, e in enumerate(ent) if e in ids]
            return None if i is None or not codes else np.isin(pick.map(i), codes)
        members = lambda cid: {cid} | {a for a, b in aliases.items() if b == cid}  # noqa: E731
        claims = E.claims_of(cards)
        for c in claims:
            c["masks"] = {}
            for k in (c["before_key"], c["after_key"], c.get("new_place_key")):
                m = mask(k, members(c["card"])) if k is not None else None
                if m is not None and m.any():
                    c["masks"][str(k)] = base64.b64encode(cv2.imencode(".png", m.astype(np.uint8) * 255)[1].tobytes()).decode()
        rec = {"meta": meta, "cards_version": oc["version"], "objects": sum(c.get("kind") == "object" for c in cards),
               "with_timeline": sum(bool((c.get("time") or {}).get("timeline")) for c in cards), "claims": claims,
               "places_judged": E.places(cards), "stats": oc["data"].get("stats"),
               "width": {}}
        for c in cards:  # the width field's outcome per card (r5b's rule), for the retail coverage table
            if c.get("kind") == "object":
                w = (c.get("physical") or {}).get("width") or {}
                k = w.get("status") or ("value" if "value" in w else "none")
                rec["width"][k] = rec["width"].get(k, 0) + 1
        polys = inp.get("polys", {}).get(rep)
        if polys:
            rows = []
            for e in polys["events"]:
                if not e.get("placed"):
                    rows.append({"kind": e["kind"], "placed": False})
                    continue

                def box(frame, place, shape):
                    pts = (e["frames"].get(str(frame)) or {}).get(str(place))
                    if not pts:
                        return None
                    m = np.zeros(shape, np.uint8)
                    cv2.fillConvexPoly(m, cv2.convexHull((np.asarray(pts, float) * np.array([shape[1] / 1280, shape[0] / 720])).astype(np.int32)), 1)
                    return m > 0
                best = None
                for i, c in enumerate(claims):
                    if c["kind"] not in E.MATCH[e["kind"]] or c["t_before"] is None or not (c["t_before"] - 1 <= e["change_s"] <= (c["t_after"] or c["t_before"]) + 1):
                        continue
                    appeared_side = c["kind"] == "appeared" or c["kind"] == "moved" and c.get("from")
                    key = c["after_key"] if appeared_side else c["before_key"]
                    m = mask(key, members(c["card"]))
                    b = box(key, 1 if (e["kind"] == "moved" and appeared_side) else 0, m.shape) if m is not None else None
                    v = E.iou(m, b)
                    if v >= E.IOU_MIN and (best is None or v > best[1]):
                        best = (i, round(v, 3))
                seen = []
                for f in range(e["change_frame"] - 60, e["change_frame"] + 60):
                    i = at.get(f)
                    if i is None:
                        continue
                    before = f < e["change_frame"]
                    if (e["kind"] == "disappeared" and not before) or (e["kind"] == "appeared" and before):
                        continue
                    m = pick.map(i)
                    b = box(f, 0 if (before or e["kind"] != "moved") else 1, m.shape)
                    if b is None or not b.any():
                        continue
                    codes, n = np.unique(m[b], return_counts=True)
                    seen += [ent[c_] for c_, k_ in zip(codes, n) if c_ and k_ / b.sum() >= .3]
                if best:
                    claims[best[0]]["planted_match"] = True
                rows.append({"kind": e["kind"], "placed": True, "change_s": e["change_s"], "found": best is not None,
                             "claim": {k: v for k, v in claims[best[0]].items() if k != "masks"} if best else None, "iou": best[1] if best else None,
                             "box_cards": sorted({x for x in seen if x and str(x).startswith("obj-")})})
            rec["planted"] = rows
        out[rep] = rec
    return out


@app.local_entrypoint()
def main(inp: str, out: str):
    res = evaluate.remote(json.loads(Path(inp).read_text()))
    Path(out).write_text(json.dumps(res, default=float))
    print({r: (x["objects"], len(x["claims"]), [(e["kind"], e.get("found")) for e in x.get("planted", [])]) for r, x in res.items()})
