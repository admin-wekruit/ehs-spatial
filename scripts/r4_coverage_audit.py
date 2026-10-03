"""r4/coverage: the fresh held-out click audit. Random clicks (a seed no earlier round used) on one video, resolved with the
viewer's own pick rule (click_audit.resolve) on two calls of the same bench and container: coverage on (a warm call) and
coverage off (the 'warm-off' call). Contact sheets for both; labels by eye (agent-labelled, click_audit's five labels, plus the
shown name of a correct pick: right / close / wrong / unidentified); a label is carried from 'on' to 'off' only where the
outcome cannot differ (nothing picked in both: the label is about the scene; or the same region picked, IoU >= 0.5).

    python scripts/r4_coverage_audit.py sample --on MIRROR REPORT --off MIRROR REPORT --site walmart --seed 97 --out DIR [--n 60]
    python scripts/r4_coverage_audit.py carry DIR       # labels-on.json -> labels-off.json where the outcome cannot differ
    python scripts/r4_coverage_audit.py score DIR [DIR ...]
    python scripts/r4_coverage_audit.py --self-check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import click_audit as ca  # noqa: E402
import fast_report_eval as ev  # noqa: E402

BOX_SOURCES = ("boxes", "densify+boxes")


def sources(mirror, report):
    """object id -> its objects-layer source ('densify', 'boxes', ...) on the newest objects version."""
    p = ev.patch_versions(mirror, report, "objects")[-1]
    return {o["id"]: o.get("source") or "first pass" for o in ev.patch_data(mirror, p).get("objects", [])}


def region(pick, i, code):
    return None if i < 0 or not code else pick.map(i) == code


def iou(a, b):
    if a is None or b is None:
        return 0.
    return float((a & b).sum() / max(1, (a | b).sum()))


def carry(on, off, pick_on, pick_off, labels_on):
    """Labels for 'off' where they cannot differ from 'on': nothing picked in either, or the same region picked (IoU >= 0.5).
    -> ({k: label}, [k left to look at])."""
    out, todo = {}, []
    for a, b in zip(on, off):
        lab = labels_on.get(str(a["k"]))
        if lab is None:
            todo.append(a["k"])
            continue
        if not a["entity"] and not b["entity"]:
            out[str(a["k"])] = {**lab, "carried": "nothing picked in both calls: the label is about the scene"}
            continue
        if a["entity"] and b["entity"]:
            v = iou(region(pick_on, a["pick_index"], a["code"]), region(pick_off, b["pick_index"], b["code"]))
            if v >= .5:
                out[str(a["k"])] = {**lab, "carried": f"picked in both, regions IoU {v:.2f}"}
                continue
        todo.append(a["k"])
    return out, todo


def score(dirs):
    rows = []
    for d in map(Path, dirs):
        for side in ("on", "off"):
            cf, lf = d / f"clicks-{side}.json", d / f"labels-{side}.json"
            if not (cf.exists() and lf.exists()):
                continue
            meta, labels = json.loads(cf.read_text()), json.loads(lf.read_text())["labels"]
            got = [(c, labels[str(c["k"])]) for c in meta["clicks"]]
            assert all(g["label"] in ca.LABELS for _, g in got), f"{lf}: unknown label"
            n = {k: sum(g["label"] == k for _, g in got) for k in ("correct", "wrong", "miss", "background", "background-hit")}
            real = n["correct"] + n["wrong"] + n["miss"]
            bg = n["background"] + n["background-hit"]
            box = [(c, g) for c, g in got if c.get("source") in BOX_SOURCES]
            names = [g.get("name") for c, g in got if g["label"] == "correct" and c.get("source") in BOX_SOURCES]
            rows.append({"site": meta["site"], "side": side, "report": meta["report"], "seed": meta["seed"], "clicks": len(got), **n,
                         "real_object_clicks": real, "real_opens_right_card": round(n["correct"] / real, 3) if real else None,
                         "real_opens_nothing": round(n["miss"] / real, 3) if real else None,
                         "real_opens_wrong_card": round(n["wrong"] / real, 3) if real else None,
                         "background_right": round(n["background"] / bg, 3) if bg else None,
                         "coarse": sum("coarse" in (g.get("note") or "") for _, g in got),
                         "picks_on_box_objects": len(box), "box_picks_correct": sum(g["label"] == "correct" for _, g in box),
                         "box_pick_names": {k: names.count(k) for k in ("right", "close", "wrong", "unidentified") if names.count(k)}})
    return rows


def self_check():
    import gzip
    import tempfile
    m0, m1 = np.zeros((36, 64), np.uint16), np.zeros((36, 64), np.uint16)
    m0[:18, :32], m1[:18, :30] = 1, 1
    data, blob = ev.pick_encode([m0], [{"t": 0., "t_end": .4, "frame": 0}], [None, "obj-a"])
    pa = ev.Pick(data, gzip.decompress(blob))
    data, blob = ev.pick_encode([m1], [{"t": 0., "t_end": .4, "frame": 0}], [None, "obj-b"])
    pb = ev.Pick(data, gzip.decompress(blob))
    cards = {}
    clicks = [{"k": 0, "frame": 1, "x": 100, "y": 100}, {"k": 1, "frame": 1, "x": 1000, "y": 600}, {"k": 2, "frame": 1, "x": 610, "y": 100}]
    on = ca.resolve(pa, cards, [dict(c) for c in clicks], 25.)
    off = ca.resolve(pb, cards, [dict(c) for c in clicks], 25.)
    assert [c["entity"] for c in on] == ["obj-a", None, "obj-a"] and [c["entity"] for c in off] == ["obj-b", None, None]
    got, todo = carry(on, off, pa, pb, {"0": {"label": "correct"}, "1": {"label": "background"}, "2": {"label": "correct"}})
    assert set(got) == {"0", "1"} and todo == [2], (got, todo)  # 2: picked on, nothing off: looked at again
    with tempfile.TemporaryDirectory() as d:
        for side, cl, lab in (("on", on, {"0": {"label": "correct", "name": "right"}, "1": {"label": "background"}, "2": {"label": "correct", "name": "unidentified"}}),
                              ("off", off, {"0": {"label": "correct"}, "1": {"label": "background"}, "2": {"label": "miss"}})):
            for c in cl:
                c["source"] = "boxes" if side == "on" and c["k"] == 2 else None
            (Path(d) / f"clicks-{side}.json").write_text(json.dumps({"site": "s", "report": side, "seed": 97, "clicks": cl}))
            (Path(d) / f"labels-{side}.json").write_text(json.dumps({"labels": lab}))
        r = {x["side"]: x for x in score([d])}
        assert r["on"]["real_opens_right_card"] == 1. and r["off"]["real_opens_nothing"] == .5 and r["on"]["box_pick_names"] == {"unidentified": 1}, r
    print("r4_coverage_audit self-check ok: carry only where the outcome cannot differ, tallies, box-object names")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", choices=("sample", "carry", "score"))
    p.add_argument("dirs", nargs="*")
    p.add_argument("--on", nargs=2)
    p.add_argument("--off", nargs=2)
    p.add_argument("--site")
    p.add_argument("--seed", type=int, default=97)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--out", type=Path)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        return self_check()
    if a.mode == "score":
        print(json.dumps(score(a.dirs), indent=1))
        return
    if a.mode == "carry":
        d = Path(a.dirs[0])
        on, off = (json.loads((d / f"clicks-{s}.json").read_text()) for s in ("on", "off"))
        pick_on = ca.load_run(on["mirror"], on["report"])[0]
        pick_off = ca.load_run(off["mirror"], off["report"])[0]
        got, todo = carry(on["clicks"], off["clicks"], pick_on, pick_off, json.loads((d / "labels-on.json").read_text())["labels"])
        (d / "labels-off-carried.json").write_text(json.dumps({"labels_by": "agent: carried from labels-on.json where the outcome cannot differ",
                                                               "labels": got, "to_look_at": todo}, indent=1))
        print(json.dumps({"carried": len(got), "to_look_at": todo}))
        return
    a.out.mkdir(parents=True, exist_ok=False)
    pick_on, _, cards_on, fps, video = ca.load_run(*a.on)
    clicks = ca.random_clicks(pick_on, fps, a.n, a.seed)
    for side, (mirror, report) in (("on", a.on), ("off", a.off)):
        pick, patch, cards, fps, video = ca.load_run(mirror, report)
        src = sources(mirror, report)
        cl = ca.resolve(pick, cards, [{k: c[k] for k in ("k", "frame", "x", "y", "sample")} for c in clicks], fps)
        for c in cl:
            c["source"] = src.get(c["entity"]) if c["entity"] and not str(c["entity"]).startswith("person") else None
            if c["source"] in BOX_SOURCES:
                c["name"] = f"[{c['source']}] {c['name']}"
        meta = {"report": report, "mirror": mirror, "site": a.site, "side": side, "pick_version": patch["version"], "fps": fps, "seed": a.seed,
                "sample": "random", "labels_by": "agent (contact sheets)"}
        keep = ("k", "frame", "x", "y", "sample", "entity", "name", "kind", "source", "pick_frame", "pick_source", "pick_index", "code")
        (a.out / f"clicks-{side}.json").write_text(json.dumps({**meta, "clicks": [{k: c.get(k) for k in keep} for c in cl]}, indent=1))
        paths = ca.sheets(video, pick, cl, a.out, f"sheet-{a.site}-{side}-s{a.seed}")
        print(side, json.dumps({"clicks": len(cl), "picked": sum(1 for c in cl if c["entity"]), "box_objects": sum(c["source"] in BOX_SOURCES for c in cl),
                                "sheets": paths}))


if __name__ == "__main__":
    main()
