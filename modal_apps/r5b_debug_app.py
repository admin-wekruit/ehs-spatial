"""r5b time (dev): replay one report's cards from its r4-instances dump on the layers Volume with timeline.TRACE on, the pick
maps repainted from the dump's kept masks (first pass and densify, larger first): per place judged free, what the camera saw
through it (the pick labels' words) and how far past the place (depth excess over the place's range). CPU, ephemeral.

    modal run modal_apps/r5b_debug_app.py --report REPORT --out OUT.json
"""
import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
app = modal.App("panoptes-r5b-debug")
image = (modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "scipy", "opencv-python-headless", "shapely")
         .add_local_dir(REPO / "fast_report", "/repo/fast_report", ignore=["**/__pycache__/**"])
         .add_local_dir(REPO / "ehs_spatial", "/repo/ehs_spatial", ignore=["**/__pycache__/**"]))
LAYERS = modal.Volume.from_name("panoptes-fb-layers")


@app.function(image=image, volumes={"/v/layers": LAYERS}, cpu=8, memory=32768, timeout=1800, retries=0)
def replay(report):
    import gzip
    import pickle
    import sys
    import numpy as np
    sys.path.insert(0, "/repo")
    from fast_report import cards, timeline
    d = pickle.load(gzip.open(f"/v/layers/reports/{report}/r4-instances-dump.pkl.gz"))
    shots = []
    for s in d["shots"]:
        s = dict(s)
        s["depth"] = np.asarray(s["depth"], np.float32)
        s["person"] = np.unpackbits(s["person"], axis=-1)[..., :s["depth"].shape[2]].astype(bool)
        shots.append(s)
    unpack = lambda p: np.unpackbits(p, axis=-1)[..., :504].astype(bool)  # noqa: E731
    paint = []  # (area, keyframe q, code, packed)
    for i, mem in enumerate(d["members"]):
        for m in mem:
            paint.append((int(unpack(d["packed"][m]).sum()), int(d["vf"][d["kept"][m]]), i + 1, d["packed"][m]))
    for g, e in enumerate(d["ent"]):
        if e > 0:
            paint.append((int(unpack(d["dens_packed"][g]).sum()), int(d["dens"]["q"][g]), int(e), d["dens_packed"][g]))
    labels = {}
    for si, pos in enumerate(d["shot_pos"]):
        at = {q: j for j, q in enumerate(pos)}
        arr = np.zeros((len(pos), 280, 504), np.int16)
        for area, q, code, pk in sorted(paint, key=lambda x: -x[0]):
            if q in at:
                arr[at[q]][unpack(pk)] = code
        labels[si] = arr
    timeline.TRACE = []
    out = cards.build({"shots": shots, "objects": d["objects"], "points": d["points"], "counts": d["counts"], "people": d["people"],
                       "calibration": d["calibration"], "labels": labels}, None)
    claims = [{"card": c["id"], "name": c["identity"]["name"], "distance_m": c["views"].get("distance_m"), "views": c["views"]["n"], **ch}
              for c in out["cards"] if c.get("kind") == "object" and (c.get("time") or {}).get("timeline") for ch in c["time"]["timeline"]["changes"]]
    return {"claims": claims, "trace": timeline.TRACE, "stats": out["stats"]}


@app.local_entrypoint()
def main(report: str, out: str):
    res = replay.remote(report)
    Path(out).write_text(json.dumps(res, default=float))
    for c in res["claims"]:
        print(c["card"], c["name"], c["kind"], c.get("t_before"), c.get("t_after"), c.get("distance_m"), c.get("views"))
    print(len(res["trace"]), "free judgements")
