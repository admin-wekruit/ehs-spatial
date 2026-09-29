"""X2 driver: videos through one FastReport container (the fast core as it is, no label cache, nothing written to the
shared caches), then the discovery designs on each video's kept state (FastReport.discover), scored against the
delivered report with fb/d-harness's rules. Local output stays small: JSON and contact-sheet JPGs; nothing heavy mirrored.

    python scripts/x2_discover.py --out RUNS/fx-x2-discover-NNN [--sites me340,samsclub-a2,walmart] [--designs sam3-generic,amg,vlm-ground]
    python scripts/x2_discover.py --score RUNS/fx-x2-discover-NNN    # rescore saved records (no GPU)
    python scripts/x2_discover.py --harness-2d RUNS/fx-x2-discover-NNN   # the harness's objects-2D row, vocabulary vs + new words
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "modal_apps"), str(REPO)]
sys.path.append(str(REPO / "scripts"))
import fast_report_eval as ev  # noqa: E402  the harness: references, recall inputs, name rule, 3D recall


def eval_inputs(site):
    """Every delivered mask frame of the reference shot (not thinned): label maps + named observations (harness rule)."""
    ref = ev.reference(site)
    frames, _, labels, obs = ev.recall_inputs(ref, ev.mask_grid(ref))
    return {"frames": frames, "labels": labels, "obs": [(o[1], o[2]) for o in obs]}, obs


def recall(rows, obs):
    """Position recall (any mask IoU >= 0.5) and word-match recall (a mask that close whose name matches the delivered
    name, fast_report_eval.same_name), pool = delivered named objects seen on a paired frame; vocabulary vs + discovery."""
    pool, pos, word = set(), {"vocab": set(), "with_discovery": set()}, {"vocab": set(), "with_discovery": set()}
    only = {}
    for o, r in zip(obs, rows):
        if not r["paired"]:
            continue
        ent, cat = o[0], o[3]
        pool.add(ent)
        if r["sets"].get("vocab", 0) >= ev.IOU_FOUND:
            pos["vocab"].add(ent)
        if max(r["sets"].values(), default=0) >= ev.IOU_FOUND:
            pos["with_discovery"].add(ent)
        for s, name, _ in r["hits"]:
            if ev.same_name(name, cat):
                word["with_discovery"].add(ent)
                if s == "vocab":
                    word["vocab"].add(ent)
        new_hits = [(s, n) for s, n, _ in r["hits"] if s != "vocab"]
        if new_hits and r["sets"].get("vocab", 0) < ev.IOU_FOUND:
            only.setdefault(ent, {"delivered": cat, "ours": sorted({f"{s}:{n}" for s, n in new_hits})})
    only = {e: v for e, v in only.items() if e not in pos["vocab"]}  # entities the vocabulary never found on any paired frame
    n = max(len(pool), 1)
    out = {"pool": len(pool)}
    for k in ("vocab", "with_discovery"):
        out[k] = {"position_found": len(pos[k]), "position_recall": round(len(pos[k]) / n, 3), "word_match_found": len(word[k]),
                  "word_match_recall": round(len(word[k]) / n, 3)}
    for part in ("discovered", "expansion"):  # which half of the design found them: the clusters' own views, or the new words' masks
        got = {o[0] for o, r in zip(obs, rows) if r["paired"] and max(r["sets"].get("vocab", 0), r["sets"].get(part, 0)) >= ev.IOU_FOUND}
        out[f"position_found_vocab+{part}"] = len(got)
    out["position_gain"] = out["with_discovery"]["position_found"] - out["vocab"]["position_found"]
    out["word_match_gain"] = out["with_discovery"]["word_match_found"] - out["vocab"]["word_match_found"]
    out["found_only_by_discovery"] = [{"entity": e, **v} for e, v in sorted(only.items())]
    return out


def objects3d(core_layers, extra, site):
    """fast_report_eval.objects3d_row before/after adding the discovered objects (centroids, estimated metres)."""
    ref = ev.reference(site)
    shots = [{**s, "c2w_m": s["c2w"]} for s in core_layers["cameras"]["shots"]]  # fb/a-core names the metres pose 'c2w'
    layers = {"cameras": {"shots": shots}, "objects": {"objects": core_layers["objects"]["objects"]}}
    _, _, align = ev.camera_rows(layers, ref)
    before = ev.objects3d_row(layers, ref, align)
    layers["objects"] = {"objects": core_layers["objects"]["objects"] + [{"shot": o["shot"], "centroid_m": o["centroid_m"]} for o in extra]}
    after = ev.objects3d_row(layers, ref, align)
    return {"before": before, "after": after}


def score(out):
    """Every design of every site: recall rows -> recall numbers, 3D before/after; writes scores.json."""
    table = {}
    for site_dir in sorted(p for p in out.iterdir() if (p / "discover.json").exists()):
        site = json.loads((site_dir / "site.json").read_text())["site"]
        rec = json.loads((site_dir / "discover.json").read_text())
        core_layers = json.loads((site_dir / "core-layers.json").read_text())
        _, obs = eval_inputs(site)
        for d, r in rec["designs"].items():
            if "error" in r:
                table.setdefault(site, {})[d] = {"error": r["error"][-400:]}
                continue
            row = {"recall": recall(r["recall"]["rows"], obs), "object_keyframes_paired": r["recall"]["object_keyframes_paired"]}
            row["objects_3d"] = objects3d(core_layers, r["objects"], site)
            row["objects_3d_with_expansion"] = objects3d(core_layers, r["objects"] + r["expansion_objects"], site)["after"]
            table.setdefault(site, {})[d] = row
    (out / "scores.json").write_text(json.dumps(table, indent=1))
    return table


def harness_2d(out):
    """fast_report_eval's objects-2D row (its GPU job, its frames, its rules) for the run's vocabulary and for the
    vocabulary + each design's new words: what the expansion words alone add on the delivered mask frames."""
    import modal
    rows, calls = {}, {}
    with modal.enable_output(), ev.app.run():
        for site_dir in sorted(p for p in out.iterdir() if (p / "discover.json").exists()):
            site = json.loads((site_dir / "site.json").read_text())["site"]
            words = json.loads((site_dir / "core-layers.json").read_text())["objects"]["words"]
            rec = json.loads((site_dir / "discover.json").read_text())
            sets = {"run": words, **{f"run+{d}": list(dict.fromkeys(words + r["new_words"])) for d, r in rec["designs"].items()
                                     if "error" not in r and r["new_words"]}}
            frames, pngs, labels, obs = ev.recall_inputs(ev.reference(site))
            calls[site] = (ev.sam3_eval.spawn({"recall": {"frames": pngs, "labels": labels, "obs": [(o[1], o[2]) for o in obs], "sets": sets}}), obs, len(frames))
        for site, (call, obs, n) in calls.items():
            result = json.loads(call.get())
            rows[site] = {"frames": n, "gpu": result["gpu"], "function_wall_s": result["function_wall_s"],
                          **{k: ev.recall_row(v, obs) for k, v in result["recall"].items()}}
    (out / "harness-2d.json").write_text(json.dumps(rows, indent=1))
    return rows


def run(out, sites, designs):
    import modal
    import fast_report_app as fa
    out.mkdir(parents=True, exist_ok=False)  # never reuse a run folder
    meta = {"sites": sites, "designs": designs, "started_unix": time.time()}
    with modal.enable_output(), fa.app.run():
        meta["app_id"] = fa.app.app_id
        fr = fa.FastReport()
        submitted = time.time()
        boot = fr.boot_info.remote()
        meta["boot"] = {**boot, "submit_to_ready_s_two_clocks": round(time.time() - submitted, 1)}
        for site in sites:
            d = out / site
            d.mkdir()
            clip = ev.PHASE2 / "data/clips" / ev.CLIPS[site]
            mp4 = (clip / "source-full.mp4").read_bytes()
            sha = hashlib.sha256(mp4).hexdigest()
            report = f"x2-{site}-{sha[:8]}-{int(time.time())}"
            options = {"cache": False, "site_vocab": False, "discover": True, "window_s": None}
            layers, run_json, t = {}, None, time.time()
            for e in fr.run.remote_gen(mp4, site, report, options):
                if e["type"] == "patch":
                    layers[e["patch"]["layer"]] = e["patch"]["data"]  # data only: blobs (mesh, points, video) stay on the Volume
                elif e["type"] == "run":
                    run_json = e["run"]
            (d / "site.json").write_text(json.dumps({"site": site, "report": report, "mp4": str(clip / "source-full.mp4"), "sha256": sha,
                                                     "core_client_s": round(time.time() - t, 1)}, indent=1))
            (d / "core-run.json").write_text(json.dumps(run_json, indent=1, default=str))
            (d / "core-layers.json").write_text(json.dumps({k: layers[k] for k in ("cameras", "objects") if k in layers}, default=str))
            print(site, "core:", json.dumps({r["layer"] + ".v" + str(r["version"]): r["written_s"] for r in run_json["layers"]}), "error:", bool(run_json["error"]), flush=True)
            if run_json["error"]:
                print(run_json["error"][-3000:], flush=True)
                continue
            ev_data, _ = eval_inputs(site)
            t = time.time()
            rec, sheets = fr.discover.remote(report, designs, ev_data)
            rec["client_s"] = round(time.time() - t, 1)
            (d / "discover.json").write_text(json.dumps(rec, indent=1))
            for name, (pages, pages_exp) in sheets.items():
                for k, jpg in enumerate(pages or []):
                    if jpg:
                        (d / f"sheet-{name.replace('/', '-every')}-{k}.jpg").write_bytes(jpg)
                for k, jpg in enumerate(pages_exp or []):
                    if jpg:
                        (d / f"sheet-{name.replace('/', '-every')}-expansion-{k}.jpg").write_bytes(jpg)
            print(site, "discover:", json.dumps({k: {kk: v.get(kk) for kk in ("analysis_s", "found", "new_words", "rounds_run", "error")}
                                                 for k, v in rec["designs"].items()})[:3000], flush=True)
    meta["finished_unix"] = time.time()
    meta["usd_estimate_upper"] = round((meta["finished_unix"] - submitted) * (2 * fa.PRICE["A100-80GB"] + fa.CPU * fa.PRICE["cpu_core"] + fa.MEMORY_GIB * fa.PRICE["gib"]), 3)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(score(out), indent=1)[:6000])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path)
    p.add_argument("--sites", default="me340,samsclub-a2,walmart")
    p.add_argument("--designs", default="sam3-generic,amg,amg:part,amg16,vlm-ground")
    p.add_argument("--score", type=Path)
    p.add_argument("--harness-2d", type=Path)
    a = p.parse_args()
    if a.harness_2d:
        print(json.dumps(harness_2d(a.harness_2d), indent=1))
    elif a.score:
        print(json.dumps(score(a.score), indent=1))
    else:
        run(a.out, a.sites.split(","), a.designs.split(","))
