"""r5 (models) internal profile: RecGen (TRI; code non-commercial, weights CC-BY-NC-4.0: internal use only) as the report's
complete display models, resident on GPU 0 in place of SAM 3D (fast_report.x7.recgen_worker x2 under MPS: no per-call reload).
The photo-workcell recipe on video, per well-observed card (display_model.well_observed, best first): X7's view selection
(<= 4 generation views >= 15 deg apart, else 8 deg; the last view taken is held out), RecGen generate_multiview on them with
depth + mask + K, the bounded placement on the generation views and the source-consistency gate on the held-out view
(r5_bench.gate_mesh, the gate venv's processes). Each object is dispatched as soon as its selection is back; accepted models
come out as sam3d.gate's do ({object, glb, transform, bounds, gate}). Generated display layers: never measurements.

    python -m fast_report.recgen_models --self-check
"""
import queue
import time

SEPARATIONS = (15., 8.)  # deg: X7's held-out rule, then its looser arm where a walk-past gives no 15 deg set
LICENCE = "RecGen (TRI): non-commercial licence, internal profile only"


def gate(objs, shots, frames_host, clock, recgen, cpu, eligible, records, deadline=None, first=None):
    """Accepted RecGen models, one dict each as soon as the gate passes it; every object's outcome to `records` (tried_rows'
    shape: stage 'assess' with accepted / reasons / iou / view, else rejected with its reason). The best `first` (SAM 3D's first
    pass: 30) of the ranked objects, one try each."""
    from fast_report import sam3d
    ranked = sam3d.rank(objs, (), eligible)[:first or sam3d.FIRST_PASS]
    src, keys = sam3d.stage(ranked, shots, frames_host)
    events, pending = queue.Queue(), [0]
    late = lambda: deadline is not None and time.time() > deadline  # noqa: E731

    def submit(pool, msg, prio, kind, r, sep=None):
        pending[0] += 1
        pool.submit(msg, prio).add_done_callback(lambda f: events.put((kind, r, sep, f)))

    def select(r, sep):
        o = ranked[r]
        submit(cpu, {"op": "select", "src": src, "shot": int(o["shot"]), "key": keys[o["id"]], "obj": {"centroid_m": o.get("centroid_m") or
               list((sum(x) / 2 for x in zip(o["box_min_m"], o["box_max_m"])))}, "min_sep": sep, "jobs": True}, (0, r), "selected", r, sep)
    for r in range(len(ranked)):
        select(r, SEPARATIONS[0])
    sel = {}
    while pending[0]:
        kind, r, sep, fut = events.get()
        pending[0] -= 1
        o = ranked[r]
        base = {"object": o["id"], "word": o["word"], "rank": r}
        try:
            res = fut.result()
        except Exception as error:  # noqa: BLE001  one object's failure is recorded, the rest go on
            records.append({**base, "stage": kind, "error": str(error)[-2000:]})
            continue
        if kind == "selected":
            sam3d._external(clock, "recgen.select", None, res["start_unix"], res["end_unix"], object=o["id"])
            if not res.get("eligible"):
                if sep == SEPARATIONS[0]:
                    select(r, SEPARATIONS[1])
                else:
                    records.append({**base, "stage": "prepare", "rejected": res.get("reason") or "no held-out view"})
                continue
            if not res.get("recgen_job") or late():
                records.append({**base, "stage": "prepare", "rejected": "no generation view keeps mask pixels with depth" if not late() else "deadline"})
                continue
            sel[r] = {"src": src, "shot": int(o["shot"]), "key": keys[o["id"]], "gen": [g["frame"] for g in res["gen"]], "held": res["held"]["frame"],
                      "crop": res["held_crop"], "anchor": res["recgen_views"][0], "sep": sep}
            submit(recgen, res["recgen_job"], (0, r), "generated", r)
        elif kind == "generated":
            sam3d._external(clock, "recgen.generate", None, res["start_unix"], res["end_unix"], object=o["id"])
            s = sel[r]
            submit(cpu, {"op": "gate", **s, "kind": "recgen", "source_frame": s["anchor"], "tile": False, "glb": True,
                         "mesh": {k: res[k] for k in ("vertices", "faces", "colors")}}, (0, r), "gated", r)
            sel[r]["generate_s"] = res["seconds"]
        else:
            sam3d._external(clock, "recgen.gate", None, res["start_unix"], res["end_unix"], object=o["id"])
            g = res.get("gate") or {}
            reasons = [] if g.get("accepted_source_consistency") else [f"held-out gate: IoU {g.get('silhouette_iou') or 0:.2f}, depth p50 "
                                                                       f"{g.get('relative_depth_median') or 0:.3f}"]
            records.append({**base, "stage": "assess", "accepted": bool(g.get("accepted_source_consistency")), "reasons": reasons,
                            "iou": g.get("silhouette_iou"), "view": sel[r]["held"], "attempt": 1, "generate_s": sel[r].get("generate_s")})
            if g.get("accepted_source_consistency") and res.get("glb"):
                import numpy as np
                t = np.eye(4)
                t[:3, 3] = res["centre"]
                yield {"object": o["id"], "glb": res["glb"], "transform": t, "bounds": res["bounds"],
                       "gate": {**{k: g.get(k) for k in ("silhouette_iou", "relative_depth_median", "relative_depth_p95", "supported_pixels")},
                                "accepted_source_consistency": True, "separation_deg": sel[r]["sep"], "views": len(sel[r]["gen"]),
                                "held_out_view": sel[r]["held"], "generator": "RecGen", "licence": LICENCE, "faces": res.get("faces"),
                                "observed_vertex_share": res.get("observed_vertex_share"), "status": "generated display model: never used for measurement"}}


def self_check():
    """The dispatch loop on fake pools: selection at 15 deg, the 8 deg retry, generation, the gate; records and yields."""
    import numpy as np
    from concurrent.futures import Future
    import fast_report.sam3d as s3

    class FakePool:
        def __init__(self, fn):
            self.fn = fn

        def submit(self, msg, prio=(0,)):
            f = Future()
            f.set_result({**self.fn(msg), "start_unix": 0., "end_unix": 1.})
            return f

    def cpu_fn(m):
        if m["op"] == "select":
            ok = m["key"] != "o1" or m["min_sep"] == 8.
            return {"eligible": ok, "reason": "views within 15 deg", "gen": [{"frame": 3}, {"frame": 9}], "held": {"frame": 20},
                    "held_crop": [0, 0, 10], "recgen_views": [3, 9], "recgen_job": {"views": ["a", "b"], "seed": 42}}
        acc = m["key"] != "o2"
        return {"gate": {"accepted_source_consistency": acc, "silhouette_iou": .8 if acc else .3, "relative_depth_median": .01}, "glb": b"glTF",
                "centre": [1., 2., 3.], "bounds": {"min": [0, 1, 2], "max": [2, 3, 4]}, "faces": 10}
    objs = [{"id": f"obj-{i}", "shot": 0, "word": "box", "box_min_m": [0, 0, 0], "box_max_m": [1, 1, 1], "masks_lr": {1: 0, 2: 0, 3: 0}} for i in range(3)]
    stage, rank = s3.stage, s3.rank
    s3.stage = lambda ranked, shots, frames: ({"dir": "x"}, {o["id"]: f"o{i}" for i, o in enumerate(ranked)})
    s3.rank = lambda objs, vocab, eligible: objs
    try:
        records = []
        got = list(gate(objs, [], "f.npy", None, FakePool(lambda m: {"vertices": np.zeros((3, 3)), "faces": [[0, 1, 2]], "colors": np.zeros((3, 3)),
                                                                     "seconds": 7.}), FakePool(cpu_fn), None, records))
    finally:
        s3.stage, s3.rank = stage, rank
    assert [x["object"] for x in got] == ["obj-0", "obj-1"] and got[1]["gate"]["separation_deg"] == 8. and got[0]["transform"][0, 3] == 1.
    assert sorted(r["object"] for r in records if r["stage"] == "assess") == ["obj-0", "obj-1", "obj-2"]
    assert next(r for r in records if r["object"] == "obj-2")["accepted"] is False
    print("recgen_models self-check ok: select (15, then 8 deg), generate, gate, yields and records")


if __name__ == "__main__":
    import sys
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
