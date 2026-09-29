"""r5 (models): the model bench's tables, contact sheets and the planar-part angle checks (local, numpy).

    python scripts/r5_models.py sheets BENCH OUT [--n 30 --seed 5]     # crop | primitive | observed | RecGen | SAM 3D | TRELLIS | TripoSR | splat
    python scripts/r5_models.py table BENCH [--labels OUT/labels.json]  # per video and method: coverage, time, held-out gate, look
    python scripts/r5_models.py gt BENCH DUMP_RUN [--inputs RUNS/mvp2-accuracy-inputs]   # (d) against metric ground truth
    python scripts/r5_models.py angles BENCH OUT                       # (d) on the videos: 0 / 90 checks, tilted parts to look at, 30 deg decidability
    python scripts/r5_models.py measure BENCH                          # (e) the workcell measurement on accepted models against (d)
    python scripts/r5_models.py --self-check

BENCH holds modal_apps/r5_models_bench.py's <name>-<arm>.json and -tiles.npz. Labels by eye are the agent's ('agent-labelled').
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
NAMES = ("me340", "samsclub-a2", "walmart")
COLS = ("crop", "primitive", "observed", "recgen", "sam3d", "trellis", "triposr", "splat")
GEN = ("recgen", "sam3d", "trellis", "triposr")
THRESH_DEG = 30.


def load(bench, name, arm):
    p = Path(bench) / f"{name}-{arm}.json"
    return json.loads(p.read_text()) if p.exists() else None


def tiles(bench, name, arm):
    p = Path(bench) / f"{name}-{arm}-tiles.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    out = {}
    for k in z.files:
        cid, col = k.split("__")
        out.setdefault(cid, {})[col] = z[k].tobytes()
    return out


def med(v, q=50):
    v = [x for x in v if x is not None]
    return round(float(np.percentile(v, q)), 3) if v else None


# ---------------------------------------------------------------- contact sheets
def sheets(bench, out, n=30, seed=5, per=10):
    import cv2
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    labels = []
    t = 160
    blank = np.full((t, t, 3), 235, np.uint8)
    for name in NAMES:
        cpu = load(bench, name, "cpu")
        if not cpu:
            continue
        tl = tiles(bench, name, "cpu")
        gens = {arm: load(bench, name, arm) for arm in ("internal", "commercial")}
        gtl = {arm: tiles(bench, name, arm) for arm in gens}
        picks = [p["id"] for p in json.loads((Path(bench) / f"{name}-picks.json").read_text())] if (Path(bench) / f"{name}-picks.json").exists() else []
        pool = [i for i in picks if i in tl]
        chosen = sorted(np.random.default_rng(seed).choice(pool, min(n, len(pool)), replace=False).tolist(), key=pool.index) if pool else []
        rows = []
        by = {r["id"]: r for r in cpu["cards"]}
        gen_rec = {arm: {o["id"]: o for o in (g or {}).get("objects", [])} for arm, g in gens.items()}
        for cid in chosen:
            cells, cap = [], []
            for col in COLS:
                raw = tl.get(cid, {}).get(col) if col in ("crop", "primitive", "observed", "splat") else \
                    next((gtl[a].get(cid, {}).get(col) for a in gtl if gtl[a].get(cid, {}).get(col)), None)
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else blank
                cells.append(cv2.resize(img, (t, t)))
                g = None
                if col in ("primitive", "observed"):
                    g = ((by[cid].get("heldout") or {}).get(col) or {}).get("gate")
                elif col in GEN:
                    m = next((gen_rec[a].get(cid, {}).get("methods", {}).get(col) for a in gen_rec if gen_rec[a].get(cid, {}).get("methods", {}).get(col)), None)
                    g = (m or {}).get("gate")
                elif col == "splat":
                    g = (by[cid].get("heldout") or {}).get("splat")
                cap.append("" if not g else f"{g.get('silhouette_iou', 0):.2f}{'+' if g.get('accepted_source_consistency') else ''}")
            strip = np.hstack(cells)
            head = np.full((16, strip.shape[1], 3), 255, np.uint8)
            r = by[cid]
            cv2.putText(head, f"{cid} {str(r.get('name'))[:22]}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 1, cv2.LINE_AA)
            for j, c in enumerate(cap):
                if c:
                    cv2.putText(head, c, (j * t + t - 44, 12), cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 90, 0) if c.endswith("+") else (0, 0, 160), 1, cv2.LINE_AA)
            rows.append(np.vstack([head, strip]))
            labels.append({"video": name, "id": cid, "name": r.get("name"), "look": {c: None for c in COLS[1:]}, "note": ""})
        for s in range(0, len(rows), per):
            top = np.full((22, rows[0].shape[1], 3), 255, np.uint8)
            cv2.putText(top, f"{name}: held-out crop | " + " | ".join(COLS[1:]) + "  (from the held-out camera; IoU, + = gate passed)", (4, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(out / f"{name}-{s // per + 1}.jpg"), np.vstack([top, *rows[s:s + per]]), [cv2.IMWRITE_JPEG_QUALITY, 82])
    lp = out / "labels.json"
    if not lp.exists():
        lp.write_text(json.dumps({"labeller": "agent-labelled", "scale": "outline right | partly | wrong | none (no model)", "rows": labels}, indent=1))
    return len(labels)


# ---------------------------------------------------------------- tables
def table(bench, labels=None):
    lab = json.loads(Path(labels).read_text())["rows"] if labels and Path(labels).exists() else []
    rows = []
    for name in NAMES:
        cpu = load(bench, name, "cpu")
        if not cpu:
            continue
        cards = cpu["cards"]
        n = len(cards)
        surf = [c["surface"] for c in cards if "surface" in c]
        held = [c["heldout"] for c in cards if "heldout" in c]
        picks = json.loads((Path(bench) / f"{name}-picks.json").read_text()) if (Path(bench) / f"{name}-picks.json").exists() else []
        look = lambda m: [x["look"].get(m) for x in lab if x["video"] == name and x["look"].get(m)]  # noqa: E731

        def lk(m):
            v = look(m)
            return f"{sum(x == 'outline right' for x in v)}/{sum(x == 'partly' for x in v)}/{sum(x == 'wrong' for x in v)}" if v else "—"
        mesh_s = [s["mesh"]["s"] for s in surf]
        parts_s = [s["parts_s"] for s in surf]
        rows.append({"video": name, "method": "(a) observed-surface mesh", "modelled": f"{sum(s['mesh'].get('triangles', 0) > 0 for s in surf)}/{n}",
                     "s_per_object": f"{med(mesh_s)} (p90 {med(mesh_s, 90)})", "cpu_s_all": round(sum(mesh_s), 1),
                     "iou": med([(h.get("observed") or {}).get("gate", {}).get("silhouette_iou") for h in held]),
                     "depth_p50": med([(h.get("observed") or {}).get("gate", {}).get("relative_depth_median") for h in held]),
                     "gate": f"{sum(bool((h.get('observed') or {}).get('gate', {}).get('accepted_source_consistency')) for h in held)}/{len(held)}", "look": lk("observed")})
        rows.append({"video": name, "method": "(d) planar parts (angles)", "modelled": f"{sum('parts' in s['parts'] for s in surf)}/{n}",
                     "s_per_object": f"{med(parts_s)} (p90 {med(parts_s, 90)})", "cpu_s_all": round(sum(parts_s), 1), "iou": None, "depth_p50": None,
                     "gate": "—", "look": "—"})
        rows.append({"video": name, "method": "r4 primitive (refit)", "modelled": f"{sum(bool(c.get('model_r4')) for c in cards)}/{n}", "s_per_object": "—",
                     "cpu_s_all": None, "iou": med([(h.get("primitive") or {}).get("gate", {}).get("silhouette_iou") for h in held]),
                     "depth_p50": med([(h.get("primitive") or {}).get("gate", {}).get("relative_depth_median") for h in held]),
                     "gate": f"{sum(bool((h.get('primitive') or {}).get('gate', {}).get('accepted_source_consistency')) for h in held)}/{len(held)}", "look": lk("primitive")})
        sp = [h["splat"] for h in held if h.get("splat")]
        if sp:
            rows.append({"video": name, "method": "(c) splat crop", "modelled": f"{len(sp)}/{n} (splat shot)", "s_per_object": "~0 (a cut)", "cpu_s_all": None,
                         "iou": med([x["silhouette_iou"] for x in sp]), "depth_p50": None, "gate": f"bleed median {med([x['bleed'] for x in sp])}",
                         "look": lk("splat")})
        for arm in ("internal", "commercial"):
            g = load(bench, name, arm)
            if not g:
                continue
            for m, s in g["summary"].items():
                objs = [o["methods"][m] for o in g["objects"] if m in o["methods"]]
                gated = [x for x in objs if x.get("gate")]
                rate = len(picks) / s["last_decided_s"] if s.get("last_decided_s") else None
                rows.append({"video": name, "method": m + (" (internal: non-commercial)" if m == "recgen" else ""),
                             "modelled": f"{s['accepted']}/{n} (of {len(picks)} top picks)",
                             "s_per_object": f"gen {(s['generate_s'] or {}).get('median')} + gate {(s['gate_s'] or {}).get('median')}",
                             "cpu_s_all": None, "first_last_s": [s["first_accepted_s"], s["last_accepted_s"], s["last_decided_s"]],
                             "objects_per_min": round(60 * rate, 1) if rate else None,
                             "iou": med([x["gate"].get("silhouette_iou") for x in gated]), "depth_p50": med([x["gate"].get("relative_depth_median") for x in gated]),
                             "gate": f"{s['accepted']}/{s['gated']}", "look": lk(m), "gpu_peak_gib": [p.get("peak_gb") for p in g.get("gpu_peak") or []],
                             "max_reserved_gib_process": s.get("max_reserved_gib_process")})
    return rows


def markdown(rows):
    head = ["video", "method", "modelled", "s_per_object", "cpu_s_all", "first_last_s", "objects_per_min", "iou", "depth_p50", "gate", "look"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| " + " | ".join("—" if r.get(h) is None else str(r.get(h)) for h in head) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- (d) angles on the videos
def angles(bench, out=None):
    """Every card's planar parts: the share near 0 / 90 deg within u, the tilted ones (15-75 deg) listed for a look, and how many
    angles a 30 deg rule could decide (value +- u clear of 30)."""
    from fast_report.surface import decided
    res = {}
    for name in NAMES:
        cpu = load(bench, name, "cpu")
        if not cpu:
            continue
        parts, bends, tilted = [], [], []
        for c in cpu["cards"]:
            sp = (c.get("surface") or {}).get("parts") or {}
            for i, p in enumerate(sp.get("parts") or []):
                t = p["tilt_deg"]
                parts.append(t)
                if 15 <= t["value"] <= 75:
                    tilted.append({"id": c["id"], "name": c.get("name"), "part": i, "tilt": t["value"], "u": t["u"], "share": p["share"], "area_m2": p["area_m2"]})
            bends += [b["angle_deg"] for b in sp.get("bends") or []]
        v = np.array([p["value"] for p in parts]) if parts else np.zeros(0)
        u = np.array([p["u"] for p in parts]) if parts else np.zeros(0)
        res[name] = {"cards": len(cpu["cards"]), "cards_with_parts": sum(bool(((c.get("surface") or {}).get("parts") or {}).get("parts")) for c in cpu["cards"]),
                     "parts": len(parts), "u_median": med(list(u)), "u_p90": med(list(u), 90),
                     "near_0": int(((v <= 15)).sum()), "near_90": int((v >= 75).sum()), "tilted_15_75": len(tilted),
                     "within_u_of_0_or_90": int(((np.minimum(v, 90 - v) <= u)).sum()),
                     "decidable_vs_30": int(sum(decided(p, THRESH_DEG) for p in parts)), "bends": len(bends),
                     "bends_decidable_vs_90": int(sum(decided(b, 90.) for b in bends)), "tilted": sorted(tilted, key=lambda x: -x["share"])[:40]}
    if out:
        import cv2
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "angles.json").write_text(json.dumps(res, indent=1))
        labels = []
        for name in res:
            tl = tiles(bench, name, "cpu")
            tilted = {t["id"] for t in res[name]["tilted"]}
            ids = [cid for cid in tl if "parts" in tl[cid]]
            ids = sorted(ids, key=lambda c: (c not in tilted, c))
            by = {c["id"]: c for c in load(bench, name, "cpu")["cards"]}
            for s0 in range(0, len(ids), 20):
                cells = []
                for cid in ids[s0:s0 + 20]:
                    img = cv2.resize(cv2.imdecode(np.frombuffer(tl[cid]["parts"], np.uint8), cv2.IMREAD_COLOR), (240, 240))
                    cv2.putText(img, f"{cid} {str(by[cid].get('name'))[:16]}", (3, 234), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
                    cells.append(img)
                    labels.append({"video": name, "id": cid, "name": by[cid].get("name"), "tilted_listed": cid in tilted, "tilt_real": None, "note": ""})
                cells += [np.full((240, 240, 3), 255, np.uint8)] * (-len(cells) % 5)
                grid = np.vstack([np.hstack(cells[i:i + 5]) for i in range(0, len(cells), 5)])
                cv2.imwrite(str(Path(out) / f"parts-{name}-{s0 // 20 + 1}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
        lp = Path(out) / "parts-labels.json"
        if not lp.exists():
            lp.write_text(json.dumps({"labeller": "agent-labelled", "question": "do the listed parts' angles match what the picture shows "
                                      "(yes | roughly | no | cannot tell)?", "rows": labels}, indent=1))
    return res


# ---------------------------------------------------------------- (d) against ground truth
def gt(bench, dump_run, inputs):
    """Per GT report: each card's GT points (accuracy_gt.regions: its own pick regions lifted with GT depth and pose, in the GT floor
    frame) cut into planar parts the same way; our parts (the bench's cpu arm) matched to them by normal (after the shots' floor
    yaw fit) and centre; the tilt and bend errors, the coverage of +-u, and whether a 30 deg rule decides right."""
    import accuracy_gt as ag
    import fast_report_eval as ev
    from fast_report import cards as fc
    from fast_report.surface import decided, planar_parts, tilt_deg
    from fast_report.core import CAMERA_HEIGHT_M
    rows = []
    dump_run, inputs = Path(dump_run), Path(inputs)
    for rep in sorted((dump_run / "reports").iterdir()):
        run = json.loads((rep / "run.json").read_text())
        site = run["call"]["site"]
        if not site.startswith("gt-"):
            continue
        cpu = load(bench, site, "cpu")
        if not cpu:
            continue
        ours = {c["id"]: c for c in cpu["cards"]}
        name, wi = site.split("-")[1], int(site.rsplit("-w", 1)[1])
        seq = ag.Seq(inputs / name)
        w = seq.g["windows"][wi]
        a0, hold = w["source_frames"][0], w["hold"]
        src = lambda f: a0 + int(f) // hold  # noqa: E731
        L = ev.load_layers(dump_run, rep.name)
        pick = ev.run_picks(dump_run, rep.name, L)[-1][1]
        oc = L["object_cards"]
        alias = oc.get("aliases") or {}
        reg = ag.regions(pick, seq, src)
        merged = {}
        for e, r in reg.items():
            cid = alias.get(e, e)
            while cid in alias and alias[cid] != cid:
                cid = alias[cid]
            merged.setdefault(cid, []).extend(r["P"])
        shots = {}
        for s in L["cameras"]["shots"]:
            keys, C = s["keys"], np.asarray(s["c2w"], float)[:, :3, 3]
            G = [seq.c2w(src(f)) for f in keys]
            ok = np.array([g is not None for g in G])
            Gc = np.array([g[:3, 3] for g in G if g is not None])
            fr = seq.floor_frame(src(keys[0]))
            ff = next((x["floor_frame"] for x in oc["shots"] if x["index"] == s["index"]), None)
            h_true = float(np.median((Gc - np.asarray(seq.floor["point"])) @ np.asarray(seq.floor["normal"])))
            mine = ag.to_card_frame(C[ok], ff)
            R, t = ag.rigid2(h_true / CAMERA_HEIGHT_M * mine[:, :2], fc.to_floor(Gc, fr)[:, :2])
            shots[s["index"]] = {"fr": fr, "R": R, "t": t, "s": h_true / CAMERA_HEIGHT_M, "cams": fc.to_floor(Gc, fr)}
        for c in oc["cards"]:
            o = ours.get(c["id"])
            sp = ((o or {}).get("surface") or {}).get("parts") or {}
            if not sp.get("parts") or not merged.get(c["id"]) or c["shot"] not in shots:
                continue
            sh = shots[c["shot"]]
            Q = fc.to_floor(np.concatenate(merged[c["id"]]), sh["fr"])
            Q = Q[fc.main_cluster(Q, fc.EPS_MIN)]
            if len(Q) < 200:
                continue
            g = planar_parts(Q, np.zeros(len(Q), int), [], np.array([np.median(sh["cams"], 0)]))
            if not g.get("parts"):
                continue
            R3 = np.eye(3)
            R3[:2, :2] = sh["R"]
            size = max(max(p["sides_m"]) for p in g["parts"])
            used = set()
            for i, p in enumerate(sp["parts"]):
                n = R3 @ np.asarray(p["normal"])
                cen = np.r_[sh["R"] @ (sh["s"] * np.asarray(p["centre_m"][:2])) + sh["t"], sh["s"] * p["centre_m"][2]]
                best = None
                for j, q in enumerate(g["parts"]):
                    if j in used:
                        continue
                    ang = float(np.degrees(np.arccos(np.clip(abs(n @ np.asarray(q["normal"])), 0, 1))))
                    dist = float(np.linalg.norm(cen - np.asarray(q["centre_m"])))
                    if ang <= 30 and dist <= .5 * size + .1 and (best is None or ang + 100 * dist / size < best[0]):
                        best = (ang + 100 * dist / size, j, q)
                if best is None:
                    continue
                used.add(best[1])
                q = best[2]
                v, u, gv = p["tilt_deg"]["value"], p["tilt_deg"]["u"], q["tilt_deg"]["value"]
                dec = decided(p["tilt_deg"], THRESH_DEG)
                rows.append({"seq": name, "window": wi, "card": c["id"], "name": (c.get("identity") or {}).get("name"), "part": i, "tilt": v, "u": u, "gt": gv,
                             "err": abs(v - gv), "covered": abs(v - gv) <= u, "decided_30": dec, "right_30": (v > THRESH_DEG) == (gv > THRESH_DEG) if dec else None,
                             "gt_near_30": abs(gv - THRESH_DEG) <= 5, "share": p["share"], "gt_tilt_check": round(tilt_deg(q["normal"]), 2)})
    return rows


def gt_table(rows):
    out = {}
    for seq in sorted({r["seq"] for r in rows}):
        rs = [r for r in rows if r["seq"] == seq]
        dec = [r for r in rs if r["decided_30"]]
        out[seq] = {"parts_matched": len(rs), "err_median_deg": med([r["err"] for r in rs]), "err_p90_deg": med([r["err"] for r in rs], 90),
                    "u_median_deg": med([r["u"] for r in rs]), "coverage": round(float(np.mean([r["covered"] for r in rs])), 3),
                    "decided_vs_30": f"{len(dec)}/{len(rs)}", "decided_right": f"{sum(r['right_30'] for r in dec)}/{len(dec)}",
                    "gt_tilted_15_75": sum(15 <= r["gt"] <= 75 for r in rs)}
    return out


def measure(bench):
    """(e): each accepted generated model's surface next to each observed planar part, measured the workcell's way
    (scene_measurements.fitted_plane) against the observed part's tilt (value +- u): the difference, inside u or not, and whether the
    model's faces there were seen (a model angle is only worth anything where its faces agree with the observed points)."""
    out = {}
    for name in NAMES:
        for arm in ("internal", "commercial"):
            g = load(bench, name, arm)
            if not g:
                continue
            for o in g["objects"]:
                for m, r in o["methods"].items():
                    for q in ((r.get("measure") or {}).get("parts") or []):
                        row = out.setdefault(m, {"parts": 0, "measured": 0, "refused": 0, "diffs": [], "within_u": 0, "seen": []})
                        row["parts"] += 1
                        if q.get("model_tilt") is None:
                            row["refused"] += 1
                            continue
                        d = abs(q["model_tilt"] - q["observed_tilt"])
                        row["measured"] += 1
                        row["diffs"].append(d)
                        row["within_u"] += int(d <= max(q["observed_u"], 1e-9))
                        row["seen"].append(q.get("model_faces_seen_share"))
    return {m: {"parts": r["parts"], "measured": r["measured"], "refused": r["refused"], "diff_median_deg": med(r["diffs"]),
                "diff_p90_deg": med(r["diffs"], 90), "within_observed_u": f"{r['within_u']}/{r['measured']}", "model_faces_seen_median": med(r["seen"])}
            for m, r in out.items()}


def self_check():
    rows = [{"video": "me340", "method": "x", "modelled": "1/2"}]
    md = markdown(rows)
    assert md.count("\n") == 2 and "| me340 | x | 1/2 |" in md
    assert med([1, None, 3]) == 2.0 and med([], 90) is None
    gtr = [{"seq": "tum", "err": 1., "u": 2., "covered": True, "decided_30": True, "right_30": True, "gt": 90.},
           {"seq": "tum", "err": 5., "u": 2., "covered": False, "decided_30": False, "right_30": None, "gt": 30.}]
    t = gt_table(gtr)["tum"]
    assert t["coverage"] == .5 and t["decided_vs_30"] == "1/2" and t["decided_right"] == "1/1" and t["gt_tilted_15_75"] == 1
    print("r5_models self-check ok: markdown table, medians, the GT angle table")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheets", "table", "gt", "angles", "measure"])
    ap.add_argument("bench")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--labels")
    ap.add_argument("--inputs", default="/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/mvp2-accuracy-inputs")
    a = ap.parse_args()
    if a.cmd == "sheets":
        print(sheets(a.bench, a.out, a.n, a.seed), "objects on the sheets")
    elif a.cmd == "table":
        rows = table(a.bench, a.labels)
        print(markdown(rows))
        (Path(a.bench) / "table.json").write_text(json.dumps(rows, indent=1))
    elif a.cmd == "measure":
        print(json.dumps(measure(a.bench), indent=1))
    elif a.cmd == "angles":
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "tilted"} for k, v in angles(a.bench, a.out).items()}, indent=1))
    else:
        rows = gt(a.bench, a.out, a.inputs)
        (Path(a.bench) / "gt-angles.json").write_text(json.dumps({"rows": rows, "table": gt_table(rows)}, indent=1, default=str))
        print(json.dumps(gt_table(rows), indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    else:
        main()
