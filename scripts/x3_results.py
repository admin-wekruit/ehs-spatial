"""X3 results.json from the run folders (fx-x3-lingbot-001: lane alone; -002: core + lane; -003: evaluation). Every number
carries its unit and how it was measured. Local, numpy-free.

    python scripts/x3_results.py RUNS_DIR OUT_JSON [--spend JSON]
"""
import json
import sys
from pathlib import Path

CORE_LAYERS = ("cameras", "room", "people", "events", "objects", "outlines")


def load(p):
    return json.loads(Path(p).read_text())


def stage_rows(r):
    return [{"stage": s["stage"], "start_s": s["start_s"], "end_s": s["end_s"], "s": s["s"], "n": s.get("n"), "peak_gb_whole_device": s.get("peak_gb")}
            for s in r["stages"]]


def lane_row(r):
    shots = []
    for s in r["shots"]:
        infer = next((x for x in r["stages"] if x["stage"] == f"lingbot.infer.shot{s['shot']}"), {})
        pts = next((x for x in r["stages"] if x["stage"] == f"lingbot.points.shot{s['shot']}"), {})
        shots.append({"shot": s["shot"], "frames": s["frames"], "lingbot_frames": len(s["source_frames"]), "keyframe_interval": s["keyframe_interval"],
                      "infer_s": infer.get("s"), "fps": s["fps_infer"], "conf_threshold_D7": s["conf_threshold"], "withheld": s.get("withheld"),
                      "points_stage_s": pts.get("s"), "points": (s.get("points_stats") or {}).get("points"),
                      "filtered_before_one_layer": (s.get("points_stats") or {}).get("kept"),
                      "after_one_layer": (s.get("points_stats") or {}).get("after_one_layer"),
                      "cell_native": (s.get("points_stats") or {}).get("cell_native"),
                      "neighbour_share_within_4pct": s["diagnose"]["share_within_4pct"], "peak_alloc_infer_gb": s["peak_alloc_infer_gb"]})
    order = sorted(r["shots"], key=lambda s: s["frames"][0] - s["frames"][1])  # the worker's order: longest shot first
    first = next((w for s, w in zip(order, r["dense_written_s"]) if not s.get("withheld")), None)
    return {"video": r["video"], "stride": r["stride"], "report": r["report"],
            "dense_written_s": {"first_version": r["dense_written_s"][0], "first_version_with_points": first, "all_shots": r["dense_written_s"][-1],
                                "versions_in_shot_order": [{"shot": s["shot"], "frames": s["frames"], "withheld": s.get("withheld"), "written_s": w}
                                                           for s, w in zip(order, r["dense_written_s"])]},
            "unit": "s from the MP4 bytes in the container to the 'dense' layer patch written (Volume commit returned)",
            "gpu_peak_whole_device_gb": [g["peak_gb"] for g in r["gpu_peak"]], "gpu_total_gb": [g["total_gb"] for g in r["gpu_peak"]],
            "worker_torch_peak_alloc_gb": r["worker_peak_alloc_gb"], "worker_torch_peak_reserved_gb": r["worker_peak_reserved_gb"],
            "flags_over_90pct": r["flags"], "cuts": r["cuts"], "shots": shots, "stages": stage_rows(r),
            "container_usd_estimate_call_only": r["usd_estimate"]}


def par_row(r):
    lay = {}
    for x in r["layers"]:
        lay.setdefault(x["layer"], []).append(x["written_s"])
    lb = r.get("lingbot") or {}
    shots = lb.get("shots") or []
    seq_s = {x["seq"]: x["written_s"] for x in r["layers"] if x["layer"] == "dense"}
    by_shot = {s["shot"]: s for s in shots}
    versions = [{"shot": v["shot"], "written_s": seq_s.get(v["seq"]), "withheld_D7": by_shot.get(v["shot"], {}).get("withheld"),
                 "gate_use": (by_shot.get(v["shot"], {}).get("align") or {}).get("use")} for v in (lb.get("versions") or [])]
    first_da3 = next((v["written_s"] for v in versions if v["gate_use"] and not v["withheld_D7"]), None)
    return {"plan_index": r.get("plan_index"), "video": r["video"], "mode": r["mode"], "report": r["report"], "error": r["error"],
            "core_layers_written_s": {k: lay.get(k) for k in CORE_LAYERS if k in lay}, "dense_written_s": lay.get("dense"),
            "dense_versions": versions, "dense_first_passing_gate_in_da3_frame_s": first_da3,
            "note": "run with the first point rule and the centres-only Sim3; a shot failing D8 was still written in LingBot's own frame then (now withheld)",
            "gpu_peak_gb": [{"gpu": g["gpu"], "peak_gb": g["peak_gb"], "total_gb": g["total_gb"], "at_s": g["at_s"], "stages_active": g["stages_active"]}
                            for g in r["gpu_peak"]],
            "flags_over_90pct": r["flags"], "lingbot_error": lb.get("error"), "lingbot_picked_up_s": lb.get("picked_up_s"),
            "lingbot_shots": [{"shot": s["shot"], "frames": s["frames"], "fps": s["fps_infer"], "conf_threshold_D7": s["conf_threshold"],
                               "withheld": s.get("withheld"), "points": (s.get("points_stats") or {}).get("points"),
                               "align": {k: v for k, v in (s.get("align") or {}).items() if k not in ("transform",)}} for s in shots],
            "stages": stage_rows(r), "call_usd_estimate": r["usd_estimate"]}


EHS = ("cable", "cord", "wire", "hose", "pipe", "rail", "pole", "post", "ladder", "shelf", "shelving", "rack", "box", "boxes", "carton", "pallet", "bin", "cart")
MEASURABLE = {"min_points_in_box": 200, "max_plane_thickness_cm": 3.0}


def measurable(cloud):
    """Objects with an EHS word whose box holds >= 200 points and whose largest RANSAC plane (2 cm) is <= 3 cm thick (p90-p10):
    a shelf face / box side / cable run one could measure a tilt or an edge on. A proxy, labelled as such."""
    rows = [o for o in cloud.get("objects", []) if any(w in o["word"].lower().split() for w in EHS)]
    ok = [o for o in rows if o["points"] >= MEASURABLE["min_points_in_box"] and o.get("plane_thickness_cm") is not None
          and o["plane_thickness_cm"] <= MEASURABLE["max_plane_thickness_cm"]]
    pts = sorted(o["points"] for o in rows)
    return {"ehs_objects": len(rows), "measurable": len(ok), "median_points_in_box": pts[len(pts) // 2] if pts else None,
            "median_plane_thickness_cm": sorted(o["plane_thickness_cm"] for o in ok)[len(ok) // 2] if ok else None}


def main(runs, out, spend=None):
    runs = Path(runs)
    res = {"experiment": "X3 LingBot fine lane (branch fx/x3-lingbot)",
           "licence": "LingBot-Map weights (robbyant/lingbot-map @ 204754b7): demo only until written licence confirmation; DA3-GIANT-1.1 CC BY-NC 4.0 (the core)",
           "policy": "observed vs inferred kept apart; every metre is 'estimated' (DA3 floor plane + an assumed 1.6 m camera height); LingBot points are display geometry, not measurements",
           "clock": "analysis time = s from the MP4 bytes in the container to the layer patch written (stub Writer: patch + Volume commit returned); boot never counted",
           "memory_unit": "GB = 1e9 bytes; 'whole device' = torch.cuda.mem_get_info sampled every 50 ms (every process incl. MPS clients and vLLM); 'torch_peak' = one process",
           "inputs": "data/clips/{me340-165, samsclub-337, walmart-190}/source-full.mp4 (the bench's MP4, 1280x720, 30 s)",
           "folders": {"fx-x3-lingbot-001": "lane alone, first point rule (superseded) + torch.compile tests",
                       "fx-x3-lingbot-002": "3 x A100: core alone vs core + lane on GPU 0/1/2 (point rule of 001 for the dense layer; alignment centres-only Sim3)",
                       "fx-x3-lingbot-004": "lane alone, final point rule (one layer, see-through drop), 1 x A100",
                       "fx-x3-lingbot-003": "first evaluation (superseded)", "fx-x3-lingbot-005": "final evaluation (alignment, detail, noise, fusion, renders)"}}
    f4 = sorted((runs / "fx-x3-lingbot-004").glob("lane-*.json"))
    res["task1_lane_alone"] = [lane_row(load(p)) | {"compile": False, "point_rule": "final"} for p in f4]
    f1 = sorted((runs / "fx-x3-lingbot-001").glob("lane-*.json"))
    res["task1_lane_alone_first_rule_and_compile"] = [lane_row(load(p)) | {"compile": "-compile" in p.name, "point_rule": "first (no see-through drop)"} for p in f1]
    res["task1_boot"] = {p.parent.name + "/" + p.name: load(p) for p in sorted(runs.glob("fx-x3-lingbot-00[14]/boot*.json"))}
    par = sorted((runs / "fx-x3-lingbot-002").glob("par-*.json"))
    if par:
        rows = [par_row(load(p)) for p in par]
        res["task4_parallel"] = rows
        res["task4_boot"] = load(runs / "fx-x3-lingbot-002/boot-parallel.json")
        summary = {}
        for r in rows:
            base = [x for x in rows if x["video"] == r["video"] and x["mode"] == "core" and x["plan_index"] != 0]
            if r["mode"] == "core" or not base:
                continue
            mean = lambda k, i: round(sum(b["core_layers_written_s"][k][i] for b in base) / len(base), 2)  # noqa: E731
            summary.setdefault(r["video"], []).append({"mode": r["mode"], "baseline_runs": [b["plan_index"] for b in base],
                "core_delta_s": {f"{k}.v{i + 1}": round(r["core_layers_written_s"][k][i] - mean(k, i), 2) for k in ("cameras", "room", "objects") for i in range(len(r["core_layers_written_s"][k]))},
                "dense_written_s": r["dense_written_s"], "dense_first_passing_gate_in_da3_frame_s": r["dense_first_passing_gate_in_da3_frame_s"],
                "gpu_peak_gb": [g["peak_gb"] for g in r["gpu_peak_gb"]],
                "lingbot_fps_main_shot": r["lingbot_shots"][0]["fps"] if r["lingbot_shots"] else None})
        res["task4_summary"] = {"rule": "core layer written_s minus the mean of the same video's core-alone runs (run 0, the first call after boot, excluded)", **summary}
    ev = sorted((runs / "fx-x3-lingbot-005").glob("eval-*.json"))
    if ev:
        evs = {p.stem.removeprefix("eval-"): load(p) for p in ev}
        res["task2_3_5_evaluation"] = evs
        res["task3_measurable_proxy"] = {"rule": measurable.__doc__.strip(), **{v: {c: measurable(m) for c, m in e["clouds"].items() if "error" not in m} for v, e in evs.items()}}
        res["comparison_images"] = [str(p) for p in sorted((runs / "fx-x3-lingbot-005").glob("compare-*.jpg"))]
    if spend is not None:
        res["spend_usd_estimate"] = spend
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"wrote {out}: {len(res['task1_lane_alone'])} lane runs, {len(res.get('task4_parallel', []))} parallel runs, {len(ev)} evaluations")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], json.loads(a[a.index("--spend") + 1]) if "--spend" in a else None)
