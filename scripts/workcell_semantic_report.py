"""Attach a frozen semantic experiment to the existing report, without rebuilding models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

TOPICS = {
    "fence": "候选：固定防护围栏的覆盖区域与开口",
    "light_curtain": "候选：光电保护装置的检测区与危险源关系",
    "emergency_button": "候选：急停装置的身份、位置与可达性",
    "signal_light": "候选：状态指示的含义与运行模式",
    "robot": "候选：机械臂运动范围与人员接近区域",
    "vguard": "候选：护板覆盖范围与被防护危险源",
    "bollard": "候选：防撞设施与通行区域",
    "floor_marking": "候选：标线含义与通道用途",
    "sign": "候选：标牌内容、适用设备与位置",
}


def attach(baseline, experiment, page, previous_run=None):
    report = json.loads((baseline / "scene-report.json").read_text())
    result = json.loads((experiment / "semantic-experiment.json").read_text())
    if result["sourceRevisionId"] != report["revision"]["id"]:
        raise ValueError("Experiment and scene revisions differ")
    for source in result["protocol"]["sources"]:
        if hashlib.sha256((baseline / source["file"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Frozen experiment source changed: " + source["file"])
    entities = {e["id"]: e for e in report["revision"]["document"]["entities"] if not e.get("sourceContext")}
    if {o["entityId"] for o in result["objects"]} != set(entities):
        raise ValueError("Semantic object set differs from rendered scene")
    observations = {o["id"]: o for o in report["revision"]["document"]["observations"]}
    endpoints = {e["objectId"]: e for e in report.get("endpointEstimation", {}).get("endpoints", [])}
    for obj in result["objects"]:
        entity = entities[obj["entityId"]]
        for ref in obj["sourceRefs"]:
            relative = Path(ref["cropPath"])
            if relative.is_absolute() or ".." in relative.parts or not (experiment / relative).is_file():
                raise ValueError("Missing or unsafe semantic crop")
            ref["cropPath"] = "semantic/" + relative.as_posix()
            ids = [oid for oid in entity["observationRefs"] if observations[oid]["imageId"] == f"photo-{ref['photo']}"]
            if len(ids) != 1:
                raise ValueError("Ambiguous semantic-to-scene observation link")
            ref["experimentObservationId"], ref["observationId"] = ref["observationId"], ids[0]
        candidates = {r["top3"][0]["id"] for r in obj["results"] if r["variant"] == "multiview" and r["top3"]}
        policy = obj["policyContext"]
        policy["candidateTopics"] = sorted({TOPICS[c] for c in candidates if c in TOPICS})
        policy["candidateBasis"] = "Top-1 multiview predictions from each encoder; candidate inspection topics only, no policy clause applied"
        policy["facts"] = [
            {"label": "照片支持", "value": f"{len(obj['sourceRefs'])} 个观察", "status": "source_linked", "source": report["revision"]["id"]},
            {"label": "测量地面", "value": "共用 workcell-floor 坐标系，Z=0", "status": "inferred_ground", "source": "scene-report.json"},
            {"label": "真实尺度验证", "value": "尚未通过；模型卡尺使用条件比例", "status": report["modelMeasurementScale"]["status"], "source": "modelMeasurementScale"},
        ]
        if obj["entityId"] in endpoints:
            endpoint = endpoints[obj["entityId"]]
            policy["facts"].append({"label": endpoint["label"] + "离地模型估计", "value": f"{endpoint['estimateCm']:.2f} cm",
                "status": "conditional_model_estimate; snapshot, not physically verified", "source": "scene-report.json#endpointEstimation"})
        policy["missingEvidence"] = list(dict.fromkeys(policy["missingEvidence"] + ["functional_identity", "equipment_operation_context"]))
        if "light_curtain" in candidates or "robot" in candidates:
            policy["missingEvidence"].append("stopping_performance_and_detection_zone")
        assert policy["applicability"] == "unknown" and policy["machineResult"] is None
    ledger = json.loads((experiment / "spend-ledger.json").read_text())
    if ledger["status"] != "completed":
        raise ValueError("Cannot publish a failed experiment as completed")
    result["timing"]["run"] = {"containerSeconds": ledger["functionSeconds"], "callSeconds": ledger["callSeconds"],
                              "estimateUsd": ledger["estimateUsd"], "callWindowEstimateUsd": ledger["callWindowEstimateUsd"],
                              "actualBilledUsd": ledger["actualBilledUsd"]}
    report["semanticExperiment"] = result
    destination = page / "semantic"
    destination.mkdir(parents=True, exist_ok=True)
    if previous_run:
        prior = json.loads((previous_run / "spend-ledger.json").read_text())
        result["timing"]["run"]["allAttemptsCallWindowEstimateUsd"] = prior["callWindowEstimateUsd"] + ledger["callWindowEstimateUsd"]
        result["timing"]["run"]["previousAttemptStatus"] = prior["status"]
        shutil.copyfile(previous_run / "spend-ledger.json", destination / "previous-attempt-spend-ledger.json")
    shutil.copytree(experiment / "crops", destination / "crops", dirs_exist_ok=True)
    for name in ("input-manifest.json", "spend-ledger.json", "manifest.json", "timing-pe-core-l.json", "timing-siglip2-so400m.json"):
        shutil.copyfile(experiment / name, destination / name)
    for path, value in ((destination / "semantic-experiment.json", result), (page / "scene-report.json", report)):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Attached {len(result['objects'])} semantic objects; models and measurements preserved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "experiment", "page"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous-run", type=Path)
    args = parser.parse_args()
    attach(args.baseline, args.experiment, args.page, args.previous_run)
