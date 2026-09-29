"""Assemble fx-x2-discover results.json from runs 001-005 (every number with units and how it was measured).

    python scripts/x2_results.py SUMMARY.json   # SUMMARY: the hand-written headline, spend and method notes merged on top
"""
import json
import sys
from pathlib import Path

WT = Path("/Users/adam/.codex/worktrees/panoptes-phase2-video-fx-x2-discover")
sys.path.insert(0, str(WT))
from fast_report import discover as dsc  # noqa: E402

RUNS = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs")
SITES = ("me340", "samsclub-a2", "walmart")
CORE = {dsc.norm(w) for w in ["fire extinguisher", "exit sign", "forklift", "ladder", "spill", "cable", "hose", "guard"]}

# my own look at the contact sheets (A whole object, B part of a larger object, C surface/background/hole/overlay/shadow,
# D several objects or none); n = tiles looked at (a seeded random sample when the design found more than one page)
OWN = {
    ("001", "me340", "amg"): {"n": 38, "of": 38, "A": 17, "B": 13, "C": 8, "D": 0, "note": "C: 4 burned-in subtitle bars, 2 shadows, 1 hole, 1 ceiling"},
    ("001", "me340", "amg:expansion"): {"n": 33, "of": 33, "A": 16, "B": 8, "C": 7, "D": 2, "note": "'wooden block' = the workbench's front edge 8x; 'hole' 6x"},
    ("001", "me340", "vlm-ground"): {"n": 8, "of": 8, "A": 5, "B": 0, "C": 0, "D": 3},
    ("001", "me340", "sam3-generic"): {"n": 2, "of": 2, "A": 2, "B": 0, "C": 0, "D": 0, "note": "a power cord hanging on the mill, a stack of yellow boxes"},
    ("002", "me340", "amg"): {"n": 32, "of": 32, "A": 11, "B": 13, "C": 8, "D": 0, "note": "C: 4 subtitle bars (the subtitle filter did not catch them), 3 shadows, 1 hole"},
    ("002", "me340", "amg:part"): {"n": 16, "of": 16, "A": 6, "B": 4, "C": 6, "D": 0},
    ("002", "me340", "amg16"): {"n": 20, "of": 20, "A": 8, "B": 11, "C": 1, "D": 0},
    ("002", "me340", "sam3-generic"): {"n": 2, "of": 2, "A": 2, "B": 0, "C": 0, "D": 0},
    ("002", "me340", "vlm-ground"): {"n": 6, "of": 6, "A": 1, "B": 0, "C": 2, "D": 3},
    ("002", "samsclub-a2", "amg"): {"n": 48, "of": 81, "A": 33, "B": 6, "C": 1, "D": 8, "note": "A: packs, price tags, a pole guard; D: bands across a pallet layer"},
    ("002", "samsclub-a2", "amg16"): {"n": 48, "of": 49, "A": 28, "B": 3, "C": 0, "D": 17},
    ("002", "samsclub-a2", "vlm-ground"): {"n": 3, "of": 3, "A": 0, "B": 0, "C": 0, "D": 3},
    ("002", "samsclub-a2", "amg:expansion"): {"n": 48, "of": 193, "A": 9, "B": 35, "C": 0, "D": 4, "note": "B: printed logos/labels on packs ('product label', 'label', 'package')"},
    ("002", "walmart", "amg"): {"n": 48, "of": 68, "A": 26, "B": 18, "C": 1, "D": 3, "note": "A mostly slippers named 'spill' by zero-shot; B: letters of the SHOES sign, shelf edges"},
    ("002", "walmart", "amg16"): {"n": 44, "of": 44, "A": 23, "B": 14, "C": 0, "D": 7},
    ("002", "walmart", "sam3-generic"): {"n": 5, "of": 5, "A": 4, "B": 0, "C": 0, "D": 1},
    ("002", "walmart", "amg:expansion"): {"n": 48, "of": 149, "A": 2, "B": 40, "C": 1, "D": 5, "note": "B: 'label' on shoe boxes and packs"},
    ("003", "walmart", "amg16:vlm"): {"n": 36, "of": 36, "A": 20, "B": 12, "C": 0, "D": 4, "note": "no 'spill' any more; 1 'fire extinguisher' on a white tag"},
    ("003", "walmart", "amg:vlm"): {"n": 43, "of": 43, "A": 26, "B": 16, "C": 0, "D": 1, "note": "6 'fire extinguisher' names, all wrong (slippers, a sign letter); 3 real floor items (a grey board, a white sheet, a plank) named ladder rung / laptop / power cord"},
}
OWN_005 = {}  # filled below from the file written after looking at run 005


def ehs(o, words):
    """EHS tags with the fixed keyword list ('paper' is not debris) and the detector veto (a name SAM 3 already ran)."""
    floor = "on the floor" in o.get("ehs", [])
    vetoed = dsc.norm(o["label"]) in words
    return ([] if vetoed else dsc.ehs_tags(o["label"], floor)) or (["on the floor"] if floor and vetoed else []), vetoed


def design_rows(run, site):
    d = json.loads((RUNS / f"fx-x2-discover-{run}/{site}/discover.json").read_text())
    scores = json.loads((RUNS / f"fx-x2-discover-{run}/scores.json").read_text()).get(site, {})
    words = {dsc.norm(w) for w in json.loads((RUNS / f"fx-x2-discover-{run}/{site}/core-layers.json").read_text())["objects"]["words"]}
    stages = {}
    for s in d["timing"]["stages"]:
        st = stages.setdefault(s["stage"], {"s": 0., "peak_gb_per_gpu": [0., 0.], "over_90": [False, False]})
        st["s"] = round(st["s"] + s["s"], 3)
        st["peak_gb_per_gpu"] = [max(a, b) for a, b in zip(st["peak_gb_per_gpu"], s["peak_gb"])]
        st["over_90"] = [a or b for a, b in zip(st["over_90"], s["over_90"])]
    out = {"_video": {"object_keyframes": d["object_keyframes"], "prep_s": d["prep_s"], "sam2_load_s_cold": d["sam2_load_s"],
                      "gpu_peak": d["timing"]["gpu_peak"], "flags_over_90pct": d["timing"]["flags"]}}
    for name, r in d["designs"].items():
        if "error" in r:
            out[name] = {"error": r["error"][-500:]}
            continue
        tags, vetoed_ehs, vetoed = {}, [], 0
        for o in r["objects"]:
            t, v = ehs(o, words)
            vetoed += v
            if v and dsc.norm(o["label"]) in CORE:
                vetoed_ehs.append(o["label"])
            for x in t:
                tags[x] = tags.get(x, 0) + 1
        sc = scores.get(name, {})
        rc = sc.get("recall", {})
        row = {
            "time_s": {"analysis_s": r["analysis_s"], "proposal_s": r.get("proposal_s"), "shared_prep_s": d["prep_s"], "write_s": r["write_s"],
                       "added_analysis_s_total": round(r["analysis_s"] + d["prep_s"] + r["write_s"], 2),
                       "stages": {k: v for k, v in stages.items() if k.startswith(f"discover.{name}.") or
                                  (k.startswith(f"discover.{name.split(':')[0]}.") and k.endswith(".propose"))}},
            "proposals": {"raw_masks": r["raw_masks"], "per_frame": r["per_frame"], "frames": len(r.get("frames_asked") or []) or d["object_keyframes"],
                          "dropped": r["dropped"], "kept": r["kept"]},
            "clusters_confirmed_2plus_frames": r["clusters_confirmed"], "lift_siglip_cos_min": r["lift_emb_min"],
            "new_clusters_round1": r["rounds"][0]["new_clusters"], "vlm_naming_crops_round1": (r["rounds"][0].get("vlm") or {}).get("crops"),
            "vlm_naming_s_round1": (r["rounds"][0].get("vlm") or {}).get("s"),
            "found_clusters": r.get("found_clusters", r["found"]), "found_objects_after_same_name_dedupe": r["found"], "rejected": r["rejected"],
            "name_sources": r["sources"], "new_words": r["new_words"], "rounds_run": r["rounds_run"],
            "rounds_that_added_words": r["rounds_that_added_words"],
            "round2_clusters_covered_by_new_words": r["rounds"][1]["covered_since_last"] if len(r["rounds"]) > 1 else None,
            "expansion": {"sam3_masks": r["rounds"][0].get("expansion_masks"), "masks_not_found_before": r.get("expansion_masks_not_found_before"),
                          "objects_lifted": r.get("expansion_clusters"), "objects_after_dedupe": r.get("expansion_found", len(r["expansion_objects"]))},
            "vlm_judge_qwen3vl8b": r.get("judge") or {"run_001_yes_no": r.get("judge_real")},
            "own_look": OWN.get((run, site, name)) or OWN_005.get((run, site, name)),
            "own_look_expansion": OWN.get((run, site, name + ":expansion")),
            "ehs_tags_found_objects": tags, "names_vetoed_by_detector": vetoed, "ehs_core_names_vetoed": vetoed_ehs,
        }
        if rc:
            row["recall_paired_keyframes"] = {
                "rule": "delivered named objects (clear/partial) on delivered mask frames within 1 frame of an object keyframe; found = a mask with IoU >= 0.5 (fast_report_eval.IOU_FOUND) at 280x378 (our DA3 grid, 4:3 crop); word match = fast_report_eval.same_name",
                "pool": rc["pool"], "object_keyframes_paired": sc["object_keyframes_paired"],
                "vocab": rc["vocab"], "with_discovery": rc["with_discovery"],
                "position_found_vocab+discovered_views_only": rc["position_found_vocab+discovered"],
                "position_found_vocab+expansion_only": rc["position_found_vocab+expansion"],
                "position_gain": rc["position_gain"], "word_match_gain": rc["word_match_gain"],
                "found_only_by_discovery": rc["found_only_by_discovery"]}
            b, a, e = sc["objects_3d"]["before"], sc["objects_3d"]["after"], sc["objects_3d_with_expansion"]
            keys = ("ours", "reference", "reference_recall_0.3m", "reference_recall_0.5m", "ours_near_reference_0.5m")
            row["objects_3d_harness"] = {"rule": "fast_report_eval.objects3d_row (centroids, Sim3 to DROID on the reference shot, metres estimated)",
                                         "before": {k: b.get(k) for k in keys}, "with_found": {k: a.get(k) for k in keys},
                                         "with_found_and_expansion": {k: e.get(k) for k in keys}}
        out[name] = row
    return out


def core_rows(run, site):
    r = json.loads((RUNS / f"fx-x2-discover-{run}/{site}/core-run.json").read_text())
    return {"layers_written_s": {f"{x['layer']}.v{x['version']}": x["written_s"] for x in r["layers"]}, "analysis_wall_s": r["analysis_wall_s"],
            "gpu_peak": [{k: g[k] for k in ("gpu", "peak_gb", "total_gb")} for g in r["gpu_peak"]], "flags": r["flags"],
            "vocabulary_words": (r.get("summary") or {}).get("words"), "objects": ((r.get("summary") or {}).get("cascade") or {}).get("objects")}


def main():
    global OWN_005
    own5 = RUNS / "fx-x2-discover-005/own-look.json"
    if own5.exists():
        OWN_005 = {tuple(k.split("|")): v for k, v in json.loads(own5.read_text()).items()}
    res = {"schema": "fx-x2-discover-results-v1", "branch": "fx/x2-discover", "runs": {}}
    for run in ("001", "002", "003", "004", "005"):
        base = RUNS / f"fx-x2-discover-{run}"
        if not base.exists():
            continue
        meta = json.loads((base / "meta.json").read_text()) if (base / "meta.json").exists() else {}
        rr = {"path": str(base), "designs": meta.get("designs"), "gpus": (meta.get("boot") or {}).get("gpus"),
              "boot_ready_s_cold_excluded": (meta.get("boot") or {}).get("ready_s"), "usd_upper_list_price": meta.get("usd_estimate_upper"), "videos": {}}
        for site in SITES:
            if (base / site / "discover.json").exists():
                rr["videos"][site] = {"core": core_rows(run, site), "designs": design_rows(run, site)}
        if (base / "harness-2d.json").exists():
            rr["harness_objects_2d"] = json.loads((base / "harness-2d.json").read_text())
        res["runs"][run] = rr
    return res


if __name__ == "__main__":
    out = main()
    path = RUNS / "fx-x2-discover-005/results.json"
    extra = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else {}
    out.update(extra)
    path.write_text(json.dumps(out, indent=1))
    print(path, len(json.dumps(out)))
