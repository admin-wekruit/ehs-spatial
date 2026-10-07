"""r4/instances: the object lift replayed on the CPU from a run's dump (r4_instances_replay.to_npz), with the production code
(fast_report.segment: lift, mask_points, join; core's object_points / object_voxels / merge_points), in core.analyse's order:
first pass per shot (lift of the kept masks on the object keyframes), then densify per shot (join, the rest lifted among
themselves). Runs in a Python with torch (numpy 1.x is fine: the npz holds no pickles).

    python scripts/r4_lift_replay.py DUMP.npz OUT.npz [--rules 0|1]
-> OUT.npz: obj_first (per kept mask, -1 = none), obj_dens (per densify mask), objects.json-in-npz (ids, shot, word, votes),
   and the objects' points (flat arrays + offsets; views and edges as JSON).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
from fast_report import segment  # noqa: E402

H, W = 280, 504


def unpack(p):
    return torch.from_numpy(np.unpackbits(p, axis=-1).astype(bool)[..., :W])


def object_points(pts, frame_of_lifted, conf, cap=20000):
    """core.object_points (copied: core imports detect_shot_cuts at the top), unchanged."""
    lab = pts["label"]
    if not len(conf):
        return []
    want = torch.full((int(lab.max()) + 1,), -1, dtype=torch.long)
    want[torch.as_tensor(conf)] = torch.arange(len(conf))
    oi_pt = want[lab[pts["mid"]]]
    k = oi_pt >= 0
    oi_pt, world, fr_pt, z = (t.cpu().numpy() for t in (oi_pt[k], pts["world"][k], frame_of_lifted[pts["mid"][k]], pts["z"][k]))
    oi_m, fm = want[lab].cpu().numpy(), frame_of_lifted.cpu().numpy()
    px, bd = pts["pixels"].cpu().numpy(), pts["border"].cpu().numpy()
    order = np.argsort(oi_pt, kind="stable")
    rng = np.random.default_rng(0)
    out = []
    for ix in np.split(order, np.cumsum(np.bincount(oi_pt, minlength=len(conf)))[:-1]):
        ratio = min(1., cap / max(len(ix), 1))
        if ratio < 1:
            ix = np.sort(rng.choice(ix, cap, replace=False))
        out.append({"world": world[ix], "frame": fr_pt[ix].astype(np.int32), "z": z[ix], "sample_ratio": ratio, "views": {}})
    for j in np.flatnonzero(oi_m >= 0):
        v = out[oi_m[j]]["views"].setdefault(int(fm[j]), [0, False, False, False, False])
        v[0] += int(px[j])
        v[1:] = [a or bool(b) for a, b in zip(v[1:], bd[j])]
    for name, (m, w) in (pts.get("edge") or {}).items():
        oi = want[lab[m]]
        k = oi >= 0
        oi, w, f = oi[k].cpu().numpy(), w[k].cpu().numpy(), frame_of_lifted[m[k]].cpu().numpy().astype(np.int32)
        order = np.argsort(oi, kind="stable")
        for i, ix in enumerate(np.split(order, np.cumsum(np.bincount(oi, minlength=len(conf)))[:-1])):
            out[i][f"{name}_world"], out[i][f"{name}_frame"] = w[ix], f[ix]
    return out


def object_voxels(obj_voxel, conf, first):
    oc, codes = obj_voxel
    want = torch.full((int(oc.max()) + 1 if len(oc) else 1,), -1, dtype=torch.long)
    want[torch.as_tensor(conf)] = torch.arange(len(conf)) + first
    o = want[oc]
    keep = o >= 0
    codes, o = codes[keep], o[keep]
    u, inv = torch.unique(codes, return_inverse=True)
    owner = torch.full((len(u),), -1, dtype=torch.long).scatter_reduce(0, inv, o, "amax")
    return u, owner


def merge_points(parts, cap=20000):
    world = np.concatenate([p["world"] for p in parts])
    frame = np.concatenate([p["frame"] for p in parts])
    z = np.concatenate([p["z"] for p in parts])
    ratio = float(np.mean([p.get("sample_ratio", 1.) for p in parts]))
    if len(world) > cap:
        ix = np.sort(np.random.default_rng(1).choice(len(world), cap, replace=False))
        world, frame, z, ratio = world[ix], frame[ix], z[ix], ratio * cap / len(world)
    views = {}
    for p in parts:
        for v, m in (p.get("views") or {}).items():
            a = views.setdefault(v, [0, False, False, False, False])
            a[0] += m[0]
            a[1:] = [x or bool(y) for x, y in zip(a[1:], m[1:])]
    edge = {f"{e}_{k}": np.concatenate([p[f"{e}_{k}"] for p in parts if f"{e}_{k}" in p] or [np.zeros((0, 3) if k == "world" else 0)])
            for e in ("top", "bottom") for k in ("world", "frame")}
    return {"world": world, "frame": frame, "z": z, "sample_ratio": ratio, "views": views, **edge}


def shots(Z):
    out = []
    for si in range(int(Z["n_shots"])):
        person = torch.from_numpy(np.unpackbits(Z[f"s{si}_person"], axis=-1).astype(bool)[..., :W])
        out.append({"pos": Z[f"s{si}_pos"].tolist(), "depth_m": torch.from_numpy(Z[f"s{si}_depth"].astype(np.float32)),
                    "K": torch.from_numpy(Z[f"s{si}_K"].astype(np.float32)), "c2w_m": torch.from_numpy(Z[f"s{si}_c2w"].astype(np.float32)),
                    "dyn": F.max_pool2d(person[:, None].float(), 5, 1, 2)[:, 0] > 0})
    return out


def votes_of(rows):
    out = {}
    for a, w, s in rows:
        out.setdefault(int(a), []).append((int(w), float(s)))
    return out


def replay(Z, words):
    keys, kept, vf = Z["keys"], Z["kept"], Z["vf"]
    geo = shots(Z)
    n_keys = len(keys)
    vote_of = votes_of(Z["votes"])
    score_k = Z["score"]
    fr_kept = vf[kept]
    objects, members, obj_points, shot_voxels = [], [], [], {}
    obj_first = np.full(len(kept), -1)
    stats = []
    for si, gg in enumerate(geo):
        local = torch.full((n_keys,), -1, dtype=torch.long)
        local[torch.tensor(gg["pos"])] = torch.arange(len(gg["pos"]))
        sel = np.flatnonzero(local[torch.from_numpy(fr_kept)].numpy() >= 0)  # positions in kept
        if len(sel) < 2:
            continue
        masks = unpack(Z["packed"][sel])
        comp, arr, st, pts = segment.lift(masks, local[torch.from_numpy(fr_kept[sel])], gg["depth_m"], gg["K"], gg["c2w_m"], gg["dyn"])
        stats.append(st)
        if arr is None:
            continue
        conf = np.flatnonzero(arr["frames"] >= segment.CONFIRMED)
        obj_points += object_points(pts, pts["fr"], conf)
        shot_voxels[si] = object_voxels(pts["obj_voxel"], conf, len(objects))
        area = masks.sum((1, 2)).numpy()
        for c in conf:
            mem = np.flatnonzero(comp == c)
            vv = {}
            for gi in kept[sel[mem]]:
                for w_, s_ in vote_of.get(int(gi), []):
                    vv[words[w_]] = vv.get(words[w_], 0.) + s_
            obj_first[sel[mem]] = len(objects)
            objects.append({"id": f"obj-{si}-{len(objects)}", "shot": si, "word": segment.name(vv, words),
                            "votes": {w: round(v, 3) for w, v in sorted(vv.items(), key=lambda x: -x[1])[:5]}, "frames": int(arr["frames"][c]),
                            "masks": int(len(mem))})
            members.append(sel[mem])
        del masks
    # densify (core.densify's lift part)
    qs, packed = Z["dens_q"], Z["dens_packed"]
    n_d = len(qs)
    ent = np.zeros(n_d, np.int64)
    vote_d = votes_of(Z["dens_votes"])
    n_old = len(members)
    added, new_objs, new_points = {}, [], []
    for si, gg in enumerate(geo):
        pos = gg["pos"]
        idx = np.flatnonzero(np.isin(qs, pos))
        if not len(idx):
            continue
        local = torch.full((n_keys,), -1, dtype=torch.long)
        local[torch.tensor(pos)] = torch.arange(len(pos))
        codes, owner = shot_voxels.get(si, (None, None))
        unmatched = []
        for b0 in range(0, len(idx), 4096):
            b = idx[b0:b0 + 4096]
            p, _ = segment.mask_points(unpack(packed[b]), local[torch.from_numpy(qs[b])], gg["depth_m"], gg["K"], gg["c2w_m"], gg["dyn"])
            if p is None:
                continue
            lifted = p["lifted"].numpy()
            who = segment.join(p, codes, owner)[0] if codes is not None else torch.full((len(lifted),), -1, dtype=torch.long)
            w_np = who.numpy()
            ent[b[lifted[w_np >= 0]]] = w_np[w_np >= 0] + 1
            unmatched += b[lifted[w_np < 0]].tolist()
            got = np.unique(w_np[w_np >= 0])
            lab = torch.where(who >= 0, who, torch.full_like(who, int(max(got.max(initial=0), 0)) + 1))
            for oi, d in zip(got, object_points({**p, "label": lab}, p["fr"], got)):
                added.setdefault(int(oi), []).append(d)
        if len(unmatched) >= 2:
            um = np.asarray(unmatched)
            comp, arr, _, pts = segment.lift(unpack(packed[um]), local[torch.from_numpy(qs[um])], gg["depth_m"], gg["K"], gg["c2w_m"], gg["dyn"])
            if arr is not None:
                conf = np.flatnonzero(arr["frames"] >= segment.CONFIRMED)
                for c, pp in zip(conf, object_points(pts, pts["fr"], conf)):
                    mem = np.flatnonzero(comp == c)
                    vv = {}
                    for g_ in um[mem]:
                        for w_, s_ in vote_d.get(int(g_), []):
                            vv[words[w_]] = vv.get(words[w_], 0.) + s_
                    oi = n_old + len(new_objs)
                    new_objs.append({"id": f"obj-{si}-{oi}", "shot": si, "word": segment.name(vv, words),
                                     "votes": {w: round(v, 3) for w, v in sorted(vv.items(), key=lambda x: -x[1])[:5]},
                                     "frames": int(arr["frames"][c]), "masks": int(len(mem)), "source": "densify"})
                    new_points.append(pp)
                    ent[um[mem]] = oi + 1
    points = list(obj_points)
    for oi, ds in added.items():
        points[oi] = merge_points([obj_points[oi]] + ds)
    return objects + new_objs, points + new_points, obj_first, ent - 1, stats


def save(path, objects, points, obj_first, obj_dens):
    flat = {}
    for k in ("world", "frame", "z", "top_world", "top_frame", "bottom_world", "bottom_frame"):
        parts = [np.asarray(p.get(k, np.zeros((0, 3) if k.endswith("world") else 0))) for p in points]
        flat[k] = np.concatenate(parts) if parts else np.zeros(0)
        flat[k + "_len"] = np.array([len(x) for x in parts], np.int64)
    meta = [{"sample_ratio": p.get("sample_ratio", 1.), "views": {str(k): [int(v[0])] + [bool(x) for x in v[1:]] for k, v in p["views"].items()}}
            for p in points]
    np.savez_compressed(path, obj_first=obj_first, obj_dens=obj_dens, objects=np.array(json.dumps(objects)), meta=np.array(json.dumps(meta)), **flat)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--words", type=Path, required=True, help="the vocabulary (JSON list): the dump's words")
    p.add_argument("--off", default="", help="comma list of segment.R4 switches to turn off (the baseline: seam)")
    p.add_argument("--set", default="", help="comma list of MODULE.NAME=value for fast_report.instances / segment constants (sweeps)")
    a = p.parse_args()
    for k in filter(None, a.off.split(",")):
        segment.R4[k] = False
    from fast_report import instances
    for kv in filter(None, a.set.split(",")):
        k, v = kv.split("=")
        mod, name = k.split(".")
        setattr({"instances": instances, "segment": segment}[mod], name, json.loads(v))
    torch.set_num_threads(8)
    Z = np.load(a.dump)
    words = json.loads(a.words.read_text())
    objects, points, obj_first, obj_dens, stats = replay(Z, words)
    save(a.out, objects, points, obj_first, obj_dens)
    print(json.dumps({"objects": len(objects), "first_assigned": int((obj_first >= 0).sum()), "dens_assigned": int((obj_dens >= 0).sum()),
                      "switches": segment.R4, "lift": [{k: v for k, v in st.items() if k in ("components", "seams", "reproject_links", "reproject_s")} for st in stats]}))


if __name__ == "__main__":
    main()
