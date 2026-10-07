"""Build local paired reports from completed experiment artifacts; no inference.

  .venv/bin/python scripts/build_candidate_report.py --candidate da3-large-1.1-01
  .venv/bin/python scripts/build_candidate_report.py --self-check
"""

import argparse
from collections import Counter
import hashlib
from html import escape
import json
import math
from pathlib import Path
from statistics import median
import xml.etree.ElementTree as ET

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/candidate-evaluation"
RUNS = ("real-clean-01", "real-clean-02", "real-clean-03", "user-bor1-02")
BASELINE = ROOT / "replay-01"
ARCHIVE = ROOT / "archive-01"
MARKDOWN = REPO / "docs/research/model-evaluation-results-2026-09-08.md"
DOCS = [REPO / "docs/research/candidate-feasibility-reconstruction-2026-09-08.md",
        REPO / "docs/research/candidate-feasibility-world-models-2026-09-08.md",
        REPO / "docs/research/model-evaluation-protocol-2026-09-08.md"]
CANDIDATES = [
    ("da3", "DA3 Large 1.1", "da3-large-1.1-01", "尚未选择已完成的本地实验。",
     "https://huggingface.co/depth-anything/DA3-LARGE-1.1"),
    ("mapanything", "MapAnything Apache v1.1", "mapanything-apache-01", "尚未选择已完成的本地实验。",
     "https://huggingface.co/facebook/map-anything-apache"),
    ("pi3x", "Pi3X", "pi3x-01", "尚未选择已完成的本地实验；不能把官方 CUDA demo 当成本机实测。",
     "https://huggingface.co/yyfz233/Pi3X"),
    ("vggt-omega", "VGGT-Omega 416 reproduction", None, "权重访问受限，未运行四个工位。",
     "https://huggingface.co/facebook/VGGT-Omega"),
    ("openspatial", "OpenSpatial-Qwen3-VL-8B", None, "本地 VLM adapter 与人工 climb/关系 GT 未完成；不输出几何槽 schema，未实测。",
     "https://huggingface.co/VINHYU/OpenSpatial-Qwen3-VL-8B"),
    ("marble", "Marble 1.1", None, "托管 API；当前没有工厂照片上传授权，也没有原输入逐像素几何合同，未实测。",
     "https://docs.worldlabs.ai/api/models"),
    ("atlas", "World Labs Atlas", None, "选择性 early access；未取得可核验 endpoint、权重或私有运行条件，未实测。",
     "https://www.worldlabs.ai/blog/atlas"),
    ("spaceformer", "SpaCeFormer", None, "CUDA 扩展与 3D 实例 schema 尚未接入；权重为非商业研究发布，未实测。",
     "https://huggingface.co/chrischoy/SpaCeFormer"),
    ("spatialvla", "SpatialVLA-4B-224-pt", None, "输出机器人动作，任务不匹配 grounded climb review；未实测。",
     "https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt"),
]


def read(path):
    return json.loads(path.read_text())


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def key(obj):
    obj = obj.get("key", obj)
    return (obj.get("frame_id", obj.get("frame")), obj.get("label"),
            obj.get("instance"), obj.get("refine_slug"))


def members(obj):
    # Same mask membership rule as verify_reprojection; no physical-ID inference.
    values = obj.get("merged_instances") or [obj.get("instance")]
    return sorted(values, key=lambda value: json.dumps(value))


def indexed(objects):
    result = {key(obj): obj for obj in objects}
    assert len(result) == len(objects), "Duplicate frame/label/instance/refine_slug keys"
    return result


def aggregate(values):
    values = [value for value in values if number(value)]
    return {"count": len(values), "worst": max(values) if values else None,
            "median": median(values) if values else None}


def pair_scores(before, after, objects_before, objects_after):
    left, right = indexed(before), indexed(after)
    bobj, aobj = indexed(objects_before), indexed(objects_after)
    paired, lost, excluded = [], [], []
    for ident, row in left.items():
        other = right.get(ident)
        same_members = ident in bobj and ident in aobj and members(bobj[ident]) == members(aobj[ident])
        if other is not None and not same_members:
            excluded.append({"key": list(ident), "reason": "source_members_changed_or_unknown"})
        if number(row.get("mean_dv_frac")):
            if other is not None and number(other.get("mean_dv_frac")) and same_members:
                assert key(row) == key(other)
                assert members(bobj[ident]) == members(aobj[ident])
                paired.append({"key": list(ident), "source_members": members(bobj[ident]),
                               "before": row["mean_dv_frac"], "after": other["mean_dv_frac"],
                               "footprint_method_before": bobj[ident].get("footprint_method"),
                               "footprint_method_after": aobj[ident].get("footprint_method")})
            else:
                lost.append({"key": list(ident), "before": row["mean_dv_frac"],
                             "after_status": "source_members_changed_or_unknown" if other is not None and not same_members else
                             other.get("status", "unscored") if other is not None else "not_eligible_or_absent"})
    shared_keys = {tuple(row["key"]) for row in paired}
    new = [{"key": list(ident), "after": row["mean_dv_frac"],
            "before_status": left.get(ident, {}).get("status", "not_eligible_or_absent")}
           for ident, row in right.items() if number(row.get("mean_dv_frac")) and ident not in shared_keys]
    return {"before": aggregate(row["before"] for row in paired),
            "after": aggregate(row["after"] for row in paired), "pairs": paired,
            "lost_baseline_scored": lost, "candidate_only_scored": new,
            "excluded_member_matches": excluded}


def gate(directory):
    path = directory / "geometry-tests.xml"
    if not path.exists():
        return None
    cases = list(ET.parse(path).getroot().iter("testcase"))
    counts = Counter("failed" if case.find("failure") is not None else
                     "error" if case.find("error") is not None else
                     "skipped" if case.find("skipped") is not None else "passed" for case in cases)
    failures = []
    for case in cases:
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        if problem is not None:
            name = case.get("name", "")
            run = next((r for r in RUNS if r in name or r.replace("-", "_") in name), None)
            failures.append({"test": name, "run_id": run,
                             "message": problem.get("message", (problem.text or "").splitlines()[0])})
    return {"tests": len(cases), **{k: counts[k] for k in ("passed", "failed", "error", "skipped")},
            "failures": failures, "xml": str(path), "log": str(directory / "geometry-tests.log")}


def run_data(directory, run_id, manifest):
    record = manifest.get("results", {}).get(run_id, {})
    if record.get("status") != "measured":
        return None
    metrics = read(directory / f"{run_id}-metrics.json")
    assert metrics["run_id"] == run_id == record["source_run_id"], "Cross-run comparison"
    run = directory / "runs" / run_id
    inv, scene, policy = (read(run / p) for p in ("inventory/inventory.json", "scene.json", "policies.json"))
    frames = []
    for frame in sorted((run / "geometry/frames").iterdir()):
        if not frame.is_dir():
            continue
        mask = np.load(frame / "valid_mask.npy", allow_pickle=False).astype(bool)
        points = np.load(frame / "pts3d.npy", allow_pickle=False)
        frames.append({"frame_id": frame.name, "pixels": mask.size,
                       "valid_pixels": int(mask.sum()),
                       "finite_points": int(np.isfinite(points).all(axis=-1).sum())})
    pixels = sum(frame["pixels"] for frame in frames)
    coverage = sum(frame["valid_pixels"] for frame in frames) / pixels if pixels else None
    return {"run_id": run_id, "record": record, "metrics": metrics, "objects": inv["objects"],
            "scene_entity_count": len(scene["entities"]), "policy": policy,
            "coverage": {"valid_fraction": coverage, "per_frame": frames}, "directory": str(run)}


def control_hashes(directory):
    # These are actual final caches, which can differ from the common immutable
    # starting snapshot because inventory writes cleaned masks back in place.
    prefixes = ("input/", "observations.json", "inventory/sam/", "inventory/phrases.json",
                "detection/", "refinements/", "geometry/moge")
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()
            and str(path.relative_to(directory)).startswith(prefixes)}


def delta(before, after):
    if number(before) and number(after):
        return after - before
    if isinstance(before, bool) and isinstance(after, bool):
        return "不变" if before == after else f"{'是' if before else '否'} → {'是' if after else '否'}"
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [b - a for a, b in zip(before, after)] if all(number(v) for v in before + after) else None
    return None


def same_footprints(before, after):
    bobj, aobj = indexed(before["objects"]), indexed(after["objects"])
    bf = {key(row): row for row in before["metrics"]["fence_footprints"] or []}
    af = {key(row): row for row in after["metrics"]["fence_footprints"] or []}
    return [{"key": list(ident), "source_members": members(bobj[ident]),
             "before": row, "after": af[ident]}
            for ident, row in bf.items() if ident in af and members(bobj[ident]) == members(aobj[ident])]


def table_rows(run_id, before, after, pairing, footprints):
    rows = []

    def add(ident, label, getter, diagnostic=False):
        b = getter(before) if before is not None else None
        a = getter(after) if after is not None else None
        rows.append({"run_id": run_id, "metric_id": ident, "metric": label,
                     "before": b, "after": a, "delta": delta(b, a),
                     "diagnostic_only": diagnostic and run_id == "user-bor1-02"})

    def metric(*path):
        def value(data):
            obj = data["metrics"]
            for part in path:
                obj = obj.get(part) if isinstance(obj, dict) else None
            return obj
        return value

    for name, label in [("worst_frac", "回投 worst / 图高"), ("median_frac", "回投 median / 图高"),
                        ("scored_count", "回投 scored 数"), ("total_count", "回投 eligible 数")]:
        add("reprojection." + name, label, metric("reprojection", name), True)
    for status in ("base-occluded", "unprojectable"):
        add("reprojection." + status, status + " 数", lambda d, s=status:
            (d["metrics"]["reprojection"].get("status_counts") or {}).get(s, 0), True)
    for name, label in [("worst", "共同 key+成员回投 worst"), ("median", "共同 key+成员回投 median"),
                        ("count", "共同可评分 key+成员数")]:
        b = pairing["before"][name] if pairing else None
        a = pairing["after"][name] if pairing else None
        rows.append({"run_id": run_id, "metric_id": "paired." + name, "metric": label,
                     "before": b, "after": a, "delta": delta(b, a),
                     "diagnostic_only": run_id == "user-bor1-02"})
    rows.append({"run_id": run_id, "metric_id": "paired.lost", "metric": "基线 scored 失去可比评分数（比较量）",
                 "before": None, "after": len(pairing["lost_baseline_scored"]) if pairing else None,
                 "delta": None, "diagnostic_only": run_id == "user-bor1-02"})
    for name, label in [("collinearity_max_residual_m", "共线最大残差 m"), ("parallel_max_spread_deg", "链内平行最大角差 °"),
                        ("manhattan_max_off_axis_deg", "Manhattan 最大离轴 °"), ("thin_max_short_side_m", "薄结构最大短边 m")]:
        add("geometry." + name, label, metric("geometry", name))
    add("cell.sides", "cell 有证据边数", metric("inventory", "cell_side_count"))
    add("cell.closed", "cell 闭合（非准确率）", lambda d: None if d["metrics"]["inventory"].get("cell_rect") is None else
        bool(d["metrics"]["inventory"]["cell_rect"].get("corners")))
    add("cell.size", "cell 尺寸 m（u × v）", lambda d:
        (d["metrics"]["inventory"].get("cell_rect") or {}).get("size_m"))
    for name, label in [("outside_cell_count", "outside_cell 数"), ("object_count", "inventory 实体数（非召回率）"),
                        ("unresolved_count", "unresolved 数"), ("phrase_count", "固定枚举短语数")]:
        add("inventory." + name, label, metric("inventory", name))
    add("scene.entities", "policy scene 实体数", lambda d: d["scene_entity_count"])
    add("inventory.dropped", "枚举静默丢弃数", lambda d: None if d["metrics"]["inventory"].get("silently_dropped_phrases") is None else
        len(d["metrics"]["inventory"]["silently_dropped_phrases"]))
    for name, label in [("factor", "MoGe/native 尺度锚倍率（非误差）"), ("confidence", "尺度锚一致性 confidence（非准确率）")]:
        add("scale." + name, label, metric("scale", name))
    add("coverage", "有效像素覆盖率 %（adapter mask）", lambda d:
        d["coverage"]["valid_fraction"] * 100 if d["coverage"]["valid_fraction"] is not None else None)
    for side in ("before", "after"):
        if footprints is not None:
            assert all(row[side]["key"] for row in footprints)
    for area, label in [("polygon_area_m2", "共同围栏家族 footprint 面积合计 m²"),
                        ("snapped_area_m2", "共同围栏家族 snapped 面积合计 m²")]:
        values = {}
        for side in ("before", "after"):
            sample = [row[side].get(area) for row in footprints or []]
            values[side] = sum(sample) if sample and all(number(v) for v in sample) else None
        rows.append({"run_id": run_id, "metric_id": "footprint." + area, "metric": label,
                     **values, "delta": delta(values["before"], values["after"]), "diagnostic_only": False})
    for field, label in [("geometry_adapter_seconds", "几何 adapter 秒（基线未重新推理）"),
                         ("postprocessing_seconds", "固定后处理秒")]:
        add("timing." + field, label, lambda d, f=field: d["record"].get(f))
    add("timing.model_load", "模型加载秒（共享进程值，勿逐 run 相加）", lambda d:
        d["record"].get("model", {}).get("model_load_seconds"))
    for ident, label in [("scale_gt", "真实尺度误差 / GT"), ("recall_gt", "人工实例召回率"),
                         ("identity_gt", "跨帧身份误差"), ("cost", "总成本 USD（硬件/电耗未计价）")]:
        add(ident, label, lambda d: None)
    return rows


def build(selected):
    baseline_manifest, archive_manifest = read(BASELINE / "experiment.json"), read(ARCHIVE / "experiment.json")
    base = {r: run_data(BASELINE, r, baseline_manifest) for r in RUNS}
    assert all(base.values()), "Baseline must contain all four measured runs"
    sources = read(BASELINE / "source_sha256.json")
    output = {"baseline": str(BASELINE), "archive": str(ARCHIVE), "baseline_gate": gate(BASELINE),
              "nm": "null/NM means not measured or unavailable, never zero or pass",
              "candidates": [], "archive_diagnostics": [], "sources": [str(p) for p in DOCS]}
    for run in RUNS:
        old = run_data(ARCHIVE, run, archive_manifest)
        raw = read(REPO / "runs" / run / "inventory/reprojection.json")
        output["archive_diagnostics"].append({
            "run_id": run, "source_stored_worst": aggregate(s.get("mean_dv_frac") for s in raw["scores"])["worst"],
            "archive_rescored_worst": old["metrics"]["reprojection"]["worst_frac"],
            "replay_worst": base[run]["metrics"]["reprojection"]["worst_frac"],
            "archive_scene_entities": old["scene_entity_count"], "replay_scene_entities": base[run]["scene_entity_count"],
            "archive_policy": [(p["policy_id"], p["status"]) for p in old["policy"]["results"]],
            "replay_policy": [(p["policy_id"], p["status"]) for p in base[run]["policy"]["results"]],
        })
    for ident, title, default_dir, reason, official in CANDIDATES:
        directory = selected.get(ident)
        manifest = read(directory / "experiment.json") if directory else None
        result = {"id": ident, "title": title, "official_url": official, "reason": reason,
                  "directory": str(directory) if directory else None, "gate": gate(directory) if directory else None,
                  "runs": [], "measured_runs": 0}
        if ident == "vggt-omega":
            access = ROOT / "vggt-omega-access.json"
            result["access_evidence"] = read(access) if access.exists() else None
            if result["access_evidence"]:
                evidence = result["access_evidence"]
                if evidence.get("authenticated_http_status") is not None:
                    result["reason"] = f"使用已配置账号请求权重仍返回 HTTP {evidence['authenticated_http_status']} / {evidence.get('authenticated_error_code')}，当前账号无权访问；未认证请求 HTTP {evidence.get('http_status')} 也保留在证据中。未执行候选推理。"
                else:
                    result["reason"] = f"本地保存的权重请求返回 HTTP {evidence.get('http_status')} / {evidence.get('error_code')}；未执行候选推理。"
        for run in RUNS:
            after = run_data(directory, run, manifest) if manifest else None
            pairing, footprints, controls = None, None, None
            if after:
                assert base[run]["run_id"] == after["run_id"] == run
                assert base[run]["policy"]["specs"] == after["policy"]["specs"], "Policy specs differ"
                assert manifest["criterion"] == baseline_manifest["criterion"]
                assert manifest["camera_height_m"] == baseline_manifest["camera_height_m"]
                candidate_sources = read(directory / "source_sha256.json")
                assert sources[run] == candidate_sources[run], "Different starting input/cache snapshot"
                core = [p for p in baseline_manifest["code_sha256"] if p.startswith("ehs_spatial/") or
                        p in ("scripts/scene_inventory.py", "scripts/detect_devices.py", "scripts/verify_reprojection.py",
                              "tests/test_geometry_invariants.py", "tests/test_cell_rect.py")]
                assert all(manifest["code_sha256"].get(p) == baseline_manifest["code_sha256"][p] for p in core), "Pipeline semantics/code differ"
                assert not manifest.get("network_attempts"), "Recorded network attempt"
                if "source_unchanged" in manifest:
                    assert manifest["source_unchanged"] is True
                bh, ah = control_hashes(Path(base[run]["directory"])), control_hashes(Path(after["directory"]))
                controls = {"starting_input_and_cache_hashes_equal": True, "policy_specs_equal": True,
                            "pipeline_code_hashes_equal": True,
                            "full_recorded_code_hashes_equal": manifest["code_sha256"] == baseline_manifest["code_sha256"],
                            "final_control_cache_hashes_equal": bh == ah,
                            "final_control_cache_differences": sorted(p for p in bh.keys() | ah.keys() if bh.get(p) != ah.get(p)),
                            "source_unchanged": manifest.get("source_unchanged"), "network_attempts": manifest.get("network_attempts")}
                pairing = pair_scores(base[run]["metrics"]["reprojection"]["scores"] or [],
                                      after["metrics"]["reprojection"]["scores"] or [], base[run]["objects"], after["objects"])
                footprints = same_footprints(base[run], after)
                result["measured_runs"] += 1
            raw_after = after["metrics"]["reprojection"]["scores"] if after else None
            result["runs"].append({"run_id": run, "status": "measured" if after else "NM",
                "error": manifest.get("results", {}).get(run, {}).get("error") if manifest else None,
                "controls": controls, "paired_reprojection": pairing, "paired_footprints": footprints,
                "raw_scores_before": base[run]["metrics"]["reprojection"]["scores"], "raw_scores_after": raw_after,
                "objects_before": base[run]["objects"], "objects_after": after["objects"] if after else None,
                "model": after["record"].get("model") if after else None,
                "coverage_before": base[run]["coverage"], "coverage_after": after["coverage"] if after else None,
                "nonfirst_frame_scored_after": sum(number(s.get("mean_dv_frac")) and key(s)[0] != "frame_0001" for s in raw_after or []) if after else None,
                "rows": table_rows(run, base[run], after, pairing, footprints)})
        result["status"] = f"实测 {result['measured_runs']}/4 run" if result["measured_runs"] else "NM · 未完成 A/B"
        output["candidates"].append(result)
    return output


def fmt(value):
    if value is None:
        return "NM"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return " × ".join(fmt(v) for v in value)
    return str(value)


def gate_text(value):
    return (f"{value['passed']} passed / {value['failed']} failed / {value['error']} errors / {value['skipped']} skipped"
            if value else "NM · 原测试尚无完成结果")


def analysis(candidate):
    lines = []
    if not candidate["measured_runs"]:
        return [candidate["reason"]]
    lines.append(gate_text(candidate["gate"]) + "。原测试通过也不等于所有数值持平或提升。")
    for failure in (candidate["gate"] or {}).get("failures", []):
        lines.append(f"失败 {failure['test']}：{failure['message'].splitlines()[0]}")
    for run in candidate["runs"]:
        paired = run["paired_reprojection"]
        if not paired:
            continue
        if run["run_id"] == "user-bor1-02":
            lines.append(f"BOR1：共同可评分 {paired['before']['count']}；候选 {run['nonfirst_frame_scored_after']} 条 scored 来自非首帧却被旧 scorer 投到首帧。该 run 回投只作诊断，不作为改善证据。")
        else:
            lines.append(f"{run['run_id']}：共同 key+成员 {paired['before']['count']} 个，worst {fmt(paired['before']['worst'])} → {fmt(paired['after']['worst'])}，median {fmt(paired['before']['median'])} → {fmt(paired['after']['median'])}；基线失去可比评分 {len(paired['lost_baseline_scored'])} 个。")
        if run["run_id"] == "real-clean-01":
            old_gate = [o for o in run["objects_before"] if o.get("footprint_method") == "guard-line" and "fence" in o["label"]]
            old_gate.sort(key=lambda o: (o["image_bbox"][0] + o["image_bbox"][2]) / 2)
            after_objects = indexed(run["objects_after"])
            if all(key(o) in after_objects for o in old_gate):
                old_depths = [math.hypot(*o["centroid_xy"]) for o in old_gate]
                new_depths = [math.hypot(*after_objects[key(o)]["centroid_xy"]) for o in old_gate]
                if not all(a > b for a, b in zip(new_depths, new_depths[1:])):
                    lines.append("固定原基线闸门实例，按原图顺序的距离：" +
                                 ", ".join(f"{v:.4f}" for v in old_depths) + " m → " +
                                 ", ".join(f"{v:.4f}" for v in new_depths) +
                                 " m；原有实例自身不再单调，因此不能只归因于新增 rails 混入评分。")
        groups = {}
        for obj in run["objects_after"]:
            if obj.get("footprint_method") == "guard-line" and obj.get("guard_chain") is not None and obj.get("rect_snapped"):
                rect = np.asarray(obj["rect_snapped"], float)
                edges = [rect[1] - rect[0], rect[2] - rect[1]]
                edge = max(edges, key=np.linalg.norm)
                groups.setdefault(obj["guard_chain"], []).append(math.degrees(math.atan2(edge[1], edge[0])) % 180)
        for angles in groups.values():
            if len(angles) > 1 and max(angles) - min(angles) > 90:
                undirected = max(min(abs(a - b), 180 - abs(a - b)) for a in angles for b in angles)
                lines.append(f"{run['run_id']} 的链角 max−min={max(angles) - min(angles):.6g}°；折回无向角差仍为 {undirected:.6g}°。这不是可直接忽略的 180° wrap 假故障，既有门槛为 <2°。")
    measured = [run for run in candidate["runs"] if run["controls"]]
    if measured and all(run["controls"]["starting_input_and_cache_hashes_equal"] and run["controls"]["pipeline_code_hashes_equal"] for run in measured):
        lines.append("已核对：每个实测 run 的输入/缓存起点 SHA-256、policy specs、生产管线源码相同；MoGe/native 尺度按候选几何重新求比值。")
    if measured and all(run["controls"]["final_control_cache_hashes_equal"] for run in measured):
        lines.append("处理后 input、observations、SAM、refinement、detection 与 MoGe 缓存也逐文件相同。共享实例仅表示同来源 key+成员，不是人工物理实例 GT。")
    return lines


def images(candidate):
    if not candidate["measured_runs"]:
        return []
    measured = [run["run_id"] for run in candidate["runs"] if run["status"] == "measured"]
    failures = [f["run_id"] for f in (candidate["gate"] or {}).get("failures", []) if f["run_id"] in measured]
    selected = list(dict.fromkeys(failures)) or measured[:1]
    return [(run, name, BASELINE / "runs" / run / "inventory" / name,
             Path(candidate["directory"]) / "runs" / run / "inventory" / name)
            for run in selected for name in ("reprojection.png", "floor_plan.png")]


def render(payload):
    tested = [c for c in payload["candidates"] if c["measured_runs"]]
    failed = [c for c in tested if c["gate"] and (c["gate"]["failed"] or c["gate"]["error"])]
    lead = (f"{len(tested)} 个候选 × 4 个 run 已实测，{len(failed)}/{len(tested)} 个候选未通过固定几何 gate。"
            if tested and all(c["measured_runs"] == 4 for c in tested) else
            f"已实测 {sum(c['measured_runs'] for c in tested)} 个候选/run 组合；未完成项目保留 NM。")
    summary = [[c["title"], f"{c['measured_runs']}/4",
                f"{c['gate']['passed']}/{c['gate']['tests']}" if c["gate"] else "NM",
                "；".join(f["message"].splitlines()[0] for f in (c["gate"] or {}).get("failures", [])) or
                "原测试未记录失败；仍需逐项比较指标"] for c in tested]
    intro = [
        "这是固定语义缓存的 geometry-slot A/B：基线固定 replay-01，档案只用于漂移诊断；没有重新调用 VLM、SAM、MoGe 或 climb review，不能称完整端到端 A/B。",
        "NM / null 表示未测或无可评分数据。差值为候选减基线；数值下降未必代表正确。共同评分只保留相同 frame/label/instance/refine_slug 与 merged mask 成员；失评分另列，不能用筛掉困难实例宣称改善。",
        "BOR1† 既有 scorer 将其他帧实体也放到首帧 floor/mask 上评分，且 normals/points 缓存复用首帧；其回投数值均为诊断，现有 canonical 测试不覆盖 BOR1。尺度 confidence 是一致性，面积、实体数和 cell 闭合都不是准确率。",
        "真实尺度 GT、人工召回、跨帧身份误差未测；模型加载/adapter/后处理分列，缓存基线没有重新推理，不能声称 E2E 加速。实验没有模型云 API 支出记录；硬件/电耗未计价，总成本为 NM。",
        "官方选型与许可核验见两份 feasibility 文档；其中官方性能表和本机 smoke 不代填下面四个工位实验。DA3 许可来源有冲突，Pi3X/SpaCeFormer 等权重条款见原文，本报告不作部署授权判断。",
        "实验副本里的旧 manifest/report.html/viewer.html 来自档案复制，不能证明候选身份或新报告结果；当前证据是 experiment.json、geometry/candidate_manifest.json、重建 scene/inventory/reprojection 与 geometry-tests.xml。MapAnythingAdapter 的请求元数据仍硬编码旧 model_id，真实候选以 candidate_manifest 的权重 SHA 为准。",
    ]
    md = ["# Panoptes 候选模型实测对比 · 2026-09-08", "", lead, "",
          "| 候选 | 完成 run | 原测试通过 | 关键失败 |", "|---|---:|---:|---|",
          *["| " + " | ".join(v.replace("|", "\\|") for v in row) + " |" for row in summary], "",
          *[p + "\n" for p in intro],
          "基线原测试：" + gate_text(payload["baseline_gate"]) + "。", "",
          f"[交互本地报告]({ROOT / 'index.html'}) · [全部逐实例 JSON]({ROOT / 'paired-comparison.json'})", "",
          " | ".join(f"[{p.stem}]({p})" for p in DOCS), "", "## 档案到重放的漂移", "",
          "源文件已保存分数、档案重新评分、重建重放是三种状态。候选只与第三列 replay 比较。", "",
          "| run | 源保存 worst | archive 重评分 | replay worst | scene 实体 archive → replay |",
          "|---|---:|---:|---:|---:|"]
    html = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>Panoptes · 模型 A/B 实测</title><style>body{font:15px/1.55 system-ui,sans-serif;color:#172b2b;background:#f6f7f4;margin:0}main{max-width:1360px;margin:auto;padding:32px}h1{font-size:30px}h2{margin-top:30px}p{max-width:1100px}a{color:#006c68}header,section{background:white;padding:24px;margin-bottom:18px;border:1px solid #dfe5df;border-radius:8px}.filters{display:flex;gap:24px;flex-wrap:wrap;position:sticky;top:0;background:#f6f7f4;padding:12px 0;z-index:1}select{font:inherit;padding:7px;max-width:100%}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #e3e7e2}td:nth-last-child(-n+3){font-variant-numeric:tabular-nums}th{background:#eef3ef}.scroll{overflow-x:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair img{width:100%;height:360px;object-fit:contain;background:#f0f2ef}figure{margin:0}figcaption{padding:8px 0;color:#43544c}.note{color:#655d31}.nm{color:#787f79}details{margin:14px 0}summary{cursor:pointer;font-weight:650}.tag{font-size:13px;color:#43544c}button{font:inherit} @media(max-width:700px){main{padding:12px}header,section{padding:14px}.pair{grid-template-columns:1fr}.pair img{height:auto}h1{font-size:24px}}</style><main><header><p class='tag'>PANOPTES / PRIVATE LOCAL EXPERIMENTS</p>",
            f"<h1>{escape(lead)}</h1><div class='scroll'><table><tr><th>候选</th><th>完成 run</th><th>原测试通过</th><th>关键失败</th></tr>" +
            "".join("<tr>" + "".join(f"<td>{escape(v)}</td>" for v in row) + "</tr>" for row in summary) + "</table></div>",
            *[f"<p>{escape(p)}</p>" for p in intro],
            f"<p>基线：{escape(gate_text(payload['baseline_gate']))} · <a href='paired-comparison.json'>完整逐实例 JSON</a></p></header>",
            "<nav class='filters' aria-label='报告筛选'><label>候选 <select id='candidate'><option value='all'>全部候选</option>" +
            "".join(f"<option value='{c['id']}'>{escape(c['title'])}</option>" for c in payload["candidates"]) +
            "</select></label><label>工位 <select id='run'><option value='all'>全部 run</option>" +
            "".join(f"<option>{r}</option>" for r in RUNS) + "</select></label></nav>",
            "<section><h2>档案 → 重放：单独记录漂移</h2><p>候选与 replay-01 比较，旧 archive 不冒充完全相同的重建基线。</p><div class='scroll'><table><tr><th>run</th><th>源保存 worst</th><th>archive 重评分</th><th>replay worst</th><th>scene 实体 archive → replay</th></tr>"]
    for row in payload["archive_diagnostics"]:
        values = [row["run_id"], fmt(row["source_stored_worst"]), fmt(row["archive_rescored_worst"]), fmt(row["replay_worst"]),
                  f"{row['archive_scene_entities']} → {row['replay_scene_entities']}"]
        md.append("| " + " | ".join(values) + " |")
        html.append(f"<tr data-run='{row['run_id']}'>" + "".join(f"<td>{escape(v)}</td>" for v in values) + "</tr>")
    html.append("</table></div>")
    for row in payload["archive_diagnostics"]:
        old = dict(row["archive_policy"])
        changes = [f"{name}: {old.get(name)} → {value}" for name, value in row["replay_policy"] if old.get(name) != value]
        if changes:
            text = row["run_id"] + " policy 重放漂移：" + "; ".join(changes)
            md.extend(["", text])
            html.append(f"<p class='note'>{escape(text)}</p>")
    html.append("</section>")
    for candidate in payload["candidates"]:
        md.extend(["", "## " + candidate["title"], "", candidate["status"] + f" · [官方来源]({candidate['official_url']})", ""])
        html.append(f"<section data-candidate='{candidate['id']}'><h2>{escape(candidate['title'])}</h2><p>{escape(candidate['status'])} · <a href='{candidate['official_url']}'>官方来源</a></p>")
        for line in analysis(candidate):
            md.append(line + "\n")
            html.append(f"<p>{escape(line)}</p>")
        if candidate["directory"]:
            directory = Path(candidate["directory"])
            md.append(f"[实验 manifest]({directory / 'experiment.json'}) · [原测试日志]({directory / 'geometry-tests.log'})\n")
            html.append(f"<p><a href='{directory.name}/experiment.json'>实验 manifest</a> · <a href='{directory.name}/geometry-tests.log'>原测试日志</a></p>")
        if candidate["id"] == "vggt-omega":
            md.append(f"[HTTP 访问原始证据]({ROOT / 'vggt-omega-access.json'})\n")
            html.append("<p><a href='vggt-omega-access.json'>HTTP 访问原始证据</a></p>")
        md.extend(["| run | 指标 | 基线 replay | 候选 | Δ 候选−基线 |", "|---|---|---:|---:|---:|"])
        html.append("<details open><summary>同 run、同指标 B/A/Δ 表</summary><div class='scroll'><table><thead><tr><th>run</th><th>指标</th><th>基线 replay</th><th>候选</th><th>Δ 候选−基线</th></tr></thead><tbody>")
        for run in candidate["runs"]:
            for row in run["rows"]:
                values = [row["run_id"] + ("†" if row["diagnostic_only"] else ""), row["metric"], fmt(row["before"]), fmt(row["after"]), fmt(row["delta"])]
                md.append("| " + " | ".join(v.replace("|", "\\|") for v in values) + " |")
                html.append(f"<tr data-run='{run['run_id']}'>" + "".join(f"<td class='{'nm' if v == 'NM' else ''}'>{escape(v)}</td>" for v in values) + "</tr>")
        html.append("</tbody></table></div></details>")
        for run, name, before, after in images(candidate):
            label = f"{run} · {name}（保留原始图，点击查看）"
            md.extend(["", label, "", f"[基线原图]({before}) · [候选原图]({after})", ""])
            html.append(f"<div data-run='{run}'><h3>{escape(label)}</h3><div class='pair'>")
            for title, path in [("基线 replay", before), ("候选", after)]:
                if path.is_file():
                    relative = path.relative_to(ROOT).as_posix()
                    html.append(f"<figure><a href='{relative}'><img loading='lazy' src='{relative}' alt='{escape(title + ' ' + run + ' ' + name)}'></a><figcaption>{title}</figcaption></figure>")
                else:
                    html.append(f"<p>{title}：图像产物缺失</p>")
            html.append("</div></div>")
        html.append("</section>")
    md.extend(["", "## 重建与可复核边界", "", "```bash", ".venv/bin/python scripts/build_candidate_report.py --self-check",
               ".venv/bin/python scripts/build_candidate_report.py " + " ".join(f"--candidate {Path(c['directory']).name}" for c in payload["candidates"] if c["directory"]), "```", "",
               "所有候选逐 run 原始 score、实例 key、merged 成员、footprint 明细、有效点覆盖与控制变量证据保存在 paired-comparison.json。未测方向的表格不是已完成 A/B；NM 不可计入胜率或均值。",
               "", "实验 CLI 的 measured 表示指标已提取；必须另外检查 geometry_tests_exit_code / JUnit XML，不能用进程退出 0 推断几何 gate 通过。", ""])
    html.append("<section><h2>来源与重建</h2><p>官方资料只支持版本与接口判断，实验数字来自本地 manifest、metrics 和 JUnit XML。feasibility 与冻结协议见 Markdown 报告。</p><p>读取 JSON 中的同 key+成员分数与失评分清单；NM 不进入均值，不把 scored 变少当改善。</p></section><script>function filter(){const c=document.getElementById('candidate').value,r=document.getElementById('run').value;document.querySelectorAll('[data-candidate]').forEach(e=>e.hidden=c!=='all'&&e.dataset.candidate!==c);document.querySelectorAll('[data-run]').forEach(e=>e.hidden=r!=='all'&&e.dataset.run!==r)}document.querySelectorAll('select').forEach(e=>e.addEventListener('change',filter));</script></main></html>")
    return "\n".join(md), "\n".join(html)


def self_check():
    obj = {"frame": "frame_0001", "label": "safety fence", "instance": 2, "refine_slug": None, "merged_instances": [2, 3]}
    score = {"key": {"frame_id": "frame_0001", "label": "safety fence", "instance": 2, "refine_slug": None}, "mean_dv_frac": .1}
    paired = pair_scores([score], [{**score, "mean_dv_frac": .2}], [obj], [obj])
    assert paired["before"]["worst"] == .1 and paired["after"]["worst"] == .2
    changed = pair_scores([score], [score], [obj], [{**obj, "merged_instances": [2, 4]}])
    assert changed["before"]["worst"] is None and len(changed["lost_baseline_scored"]) == 1
    empty = pair_scores([{**score, "mean_dv_frac": None}], [{**score, "mean_dv_frac": None}], [obj], [obj])
    assert empty["before"]["worst"] is None and empty["before"]["count"] == 0
    assert delta(None, 1) is None
    assert fmt(None) == "NM"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", help="Experiment directory; repeat for multiple candidates")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    self_check()
    if args.self_check:
        print("candidate report self-check passed")
        return 0
    selected = {}
    for value in args.candidate or ["da3-large-1.1-01"]:
        path = Path(value).resolve() if Path(value).is_dir() else ROOT / value
        manifest = read(path / "experiment.json")
        ident = "pi3x" if manifest["arm"] in ("pi3", "pi3x") else manifest["arm"]
        if ident not in {c[0] for c in CANDIDATES}:
            parser.error(f"Unknown candidate arm: {ident}")
        assert ident not in selected, "Only one selected experiment per candidate"
        selected[ident] = path.resolve()
    payload = build(selected)
    markdown, html = render(payload)
    (ROOT / "paired-comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    MARKDOWN.write_text(markdown)
    (ROOT / "index.html").write_text(html)
    print(json.dumps({"measured_candidates": [c["title"] for c in payload["candidates"] if c["measured_runs"]],
                      "html": str(ROOT / "index.html"), "markdown": str(MARKDOWN)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
