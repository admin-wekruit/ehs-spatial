"""r4/instances: the instance layer replayed offline from a run's dump (core.write_dump, bench --dump): every kept SAM 3 mask
(first pass and densify) with its keyframe, word and object; the objects, their points and the shots; cards.build re-run on
them with or without the r4 rules (fast_report.instances), and label maps painted from the masks (the smaller mask wins, as
segment.paint) at the pick grid (640x360) for r4_instances_eval.

    python scripts/r4_instances_replay.py DUMP.pkl.gz --site me340 [--rules off|on] [--out OUT.json]
"""
import argparse
import gzip
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
H, W = 280, 504
PICK = (360, 640)


def load(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def masks_table(D):
    """One row per kept mask: source ('v1' first pass / 'd' densify), index in its packed array, keyframe q (global), shot,
    word, score, object index (-1 = none)."""
    kept = np.asarray(D["kept"])
    obj1 = np.full(len(kept), -1)
    for oi, mem in enumerate(D["members"]):
        obj1[np.asarray(mem, int)] = oi
    vf = np.asarray(D["vf"])
    q1 = vf[kept]
    qd = np.asarray(D["dens"]["q"], int)
    objd = np.asarray(D["ent"], int) - 1
    shot_of = {q: si for si, pos in enumerate(D["shot_pos"]) for q in pos}
    rows = {"src": np.r_[np.zeros(len(kept), int), np.ones(len(qd), int)], "i": np.r_[np.arange(len(kept)), np.arange(len(qd))],
            "q": np.r_[q1, qd].astype(int), "word": np.r_[np.asarray(D["word"], int), np.asarray(D["dens"]["word"], int)],
            "score": np.r_[np.asarray(D["score"], float), np.asarray(D["dens"]["score"], float)], "obj": np.r_[obj1, objd]}
    rows["shot"] = np.array([shot_of.get(int(q), -1) for q in rows["q"]])
    return rows


def unpack(D, src, i):
    p = D["packed"] if src == 0 else D["dens_packed"]
    return np.unpackbits(p[i], axis=-1).astype(bool)[..., :W]


def frame_masks(D, T, q):
    """Row indices of the kept masks on keyframe q and their (n,H,W) bool masks."""
    rows = np.flatnonzero(T["q"] == q)
    return rows, (np.stack([unpack(D, T["src"][r], T["i"][r]) for r in rows]) if len(rows) else np.zeros((0, H, W), bool))


def paint(ms):
    """(n,H,W) -> (H,W) int: -1 = nothing, i = ms[i]; the smaller mask wins where they overlap (segment.paint)."""
    out = np.full((H, W), -1, np.int64)
    for i in np.argsort(-ms.reshape(len(ms), -1).sum(1), kind="stable"):
        out[ms[i]] = i
    return out


def label_maps(D, T, keep_rows=None):
    """[(source frame, (360,640) int map of mask rows, -1 = nothing)] for every keyframe with masks; keep_rows: only these
    masks are painted (the rest are background)."""
    import cv2
    out = []
    for q in sorted(set(T["q"].tolist())):
        rows, ms = frame_masks(D, T, q)
        if keep_rows is not None:
            k = np.isin(rows, keep_rows)
            rows, ms = rows[k], ms[k]
        lab = paint(ms) if len(rows) else np.full((H, W), -1, np.int64)
        lab = np.where(lab >= 0, rows[np.clip(lab, 0, None)] if len(rows) else -1, -1)
        big = cv2.resize(lab.astype(np.float32), (PICK[1], PICK[0]), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        out.append((int(D["keys"][q]), big))
    return out


def to_eval(maps, row_card):
    """Mask-row maps -> (maps of codes, entity list) for r4_instances_eval: code 0 = nothing, k = entities[k]."""
    cards = sorted({c for c in row_card if c})
    code = {c: i + 1 for i, c in enumerate(cards)}
    lut = np.array([code.get(c, 0) if c else 0 for c in row_card] + [0], np.int64)  # index -1 -> the trailing 0
    return [(f, lut[m]) for f, m in maps], [None] + cards


def build_inputs(D):
    """cards.build's input as core's cards v3 gave it (depth float32 again, people unpacked)."""
    shots = []
    for s in D["shots"]:
        s = dict(s)
        if s.get("depth") is not None:
            s["depth"] = s["depth"].astype(np.float32)
        if s.get("person") is not None:
            s["person"] = np.unpackbits(s["person"], axis=-1).astype(bool)[..., :W]
        shots.append(s)
    return {"shots": shots, "objects": D["objects"], "points": D["points"], "counts": D["counts"], "people": D["people"],
            "calibration": D["calibration"]}


def run_cards(D, pool=None, **kw):
    from fast_report import cards
    t = time.perf_counter()
    out = cards.build(build_inputs(D), pool, 16, **kw)
    out["replay_s"] = round(time.perf_counter() - t, 2)
    return out


def load_points(path):
    """-> (objects, points, obj_first, obj_dens) in the venv."""
    Z = np.load(path)
    objects, meta = json.loads(str(Z["objects"])), json.loads(str(Z["meta"]))
    cut = {k: np.cumsum(Z[k + "_len"])[:-1] for k in ("world", "frame", "z", "top_world", "top_frame", "bottom_world", "bottom_frame")}
    split = {k: np.split(Z[k], cut[k]) for k in cut}
    points = []
    for i, m in enumerate(meta):
        p = {k: split[k][i] for k in split}
        p["frame"] = p["frame"].astype(np.int32)
        p["sample_ratio"] = m["sample_ratio"]
        p["views"] = {int(k): v for k, v in m["views"].items()}
        points.append(p)
    return objects, points, Z["obj_first"], Z["obj_dens"]


def evaluate_replay(site, D, rep, pool=None, dl=None, maps=None, **build_kw):
    """A lift replay (r4_lift_replay's npz) -> cards.build on its objects -> each mask's card -> r4_instances_eval on label
    maps painted from the masks. -> (result, cards output, (maps, entities), dl, maps)."""
    import r4_instances_eval as ev
    objects, points, of, od = load_points(rep)
    inp = build_inputs(D)
    inp.update(objects=objects, points=points, counts={})
    t = time.perf_counter()
    from fast_report import cards
    out = cards.build(inp, pool, 16, **build_kw)
    build_s = time.perf_counter() - t
    T = masks_table(D)
    obj = np.r_[of, od]
    ids = [o["id"] for o in objects]
    al = out["aliases"]
    row_card = [None if o < 0 else al.get(ids[o], ids[o]) for o in obj]
    parent = {c["id"]: c["part_of"]["id"] for c in out["cards"] if c.get("part_of") and c["part_of"].get("kind") == "part"}
    maps = maps if maps is not None else label_maps(D, T)
    em, ents = to_eval(maps, row_card)
    res, dl = ev.evaluate(site, em, ents, {}, parent, dl=dl)
    shown = [c for c in out["cards"] if c["kind"] == "object"]
    res.update(cards=len(shown), objects_in=len(objects), merged_away=len(al), parts=sum(bool(c.get("part_of")) for c in shown),
               cards_build_s=round(build_s, 2), masks_assigned=int((obj >= 0).sum()), masks=len(obj))
    return res, out, (em, ents), dl, maps


def brief(res):
    return {k: ({kk: v[kk] for kk in ("delivered", "covered", "missed", "in_pieces", "pieces_per_covered_mean", "wrong_merge_cards",
                                      "delivered_in_wrong_merges")} if isinstance(v, dict) and "covered" in v else v)
            for k, v in res.items() if k != "detail"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump", type=Path)
    p.add_argument("--site", required=True)
    p.add_argument("--rep", type=Path, required=True, help="r4_lift_replay's npz")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    D = load(a.dump)
    res, out, *_ = evaluate_replay(a.site, D, a.rep)
    print(json.dumps(brief(res)))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()


def to_npz(D, path):
    """The lift's inputs as a plain npz (no pickles: the torch replay runs in another Python with numpy 1.x)."""
    arr = {"keys": np.asarray(D["keys"], np.int64), "kept": np.asarray(D["kept"], np.int64), "vf": np.asarray(D["vf"], np.int64),
           "word": np.asarray(D["word"], np.int64), "score": np.asarray(D["score"], np.float32), "packed": D["packed"],
           "members_flat": np.concatenate([np.asarray(m, np.int64) for m in D["members"]]) if D["members"] else np.zeros(0, np.int64),
           "members_len": np.array([len(m) for m in D["members"]], np.int64),
           "votes": np.array([(a, w, s) for a, w, s in D["votes"]], np.float64).reshape(-1, 3),
           "dens_q": np.asarray(D["dens"]["q"], np.int64), "dens_word": np.asarray(D["dens"]["word"], np.int64),
           "dens_score": np.asarray(D["dens"]["score"], np.float32), "dens_packed": D["dens_packed"], "ent": np.asarray(D["ent"], np.int64),
           "dens_votes": np.array([(a, w, s) for a, w, s in D["dens"]["votes"]], np.float64).reshape(-1, 3), "n_shots": np.array(len(D["shots"]))}
    for si, (s, pos) in enumerate(zip(D["shots"], D["shot_pos"])):
        arr[f"s{si}_pos"] = np.asarray(pos, np.int64)
        for k in ("c2w", "K", "depth", "person", "normal", "point_m"):
            arr[f"s{si}_{k}"] = np.asarray(s[k])
    np.savez(path, **arr)


CARD_VARIANTS = {"C0": {"merge_v2": False, "parts": False}, "C1": {"merge_v2": True, "parts": True}}


def matrix(site, runs, out_json, dump_dir):
    """runs: [(lift variant, card variant)]: every run on one set of label maps (painted once). -> {name: result}."""
    from fast_report import cards
    D = load(Path(dump_dir) / f"{site}.pkl.gz")
    T = masks_table(D)
    maps = label_maps(D, T)
    dl, out = None, {}
    for lv, cv in runs:
        cards.R4.update(CARD_VARIANTS[cv])
        res, co, _, dl, _ = evaluate_replay(site, D, Path(dump_dir).parent / f"rep-{site}-{lv}.npz", dl=dl, maps=maps)
        res["cards_stats"] = co["stats"]
        out[f"{lv}{cv}"] = res
        print(site, lv + cv, json.dumps(brief(res)), flush=True)
    cards.R4.update(CARD_VARIANTS["C1"])
    Path(out_json).write_text(json.dumps(out, indent=1, default=str))
    return out
