"""Bind a frozen semantic experiment to one report revision during the report build.

The experiment consumed only the canonical photos, catalog observation
polygons, frame point maps under their validity masks, and the rigid floor
transform. Model GLBs, scale, ground-clearance endpoints and representations
never entered it. Reuse is therefore checked at that content level for every
revision: identical photo bytes, pixel-identical observation masks rebuilt
from this revision's polygons, identical supported/sampled point counts, the
same rigid transform, and frame files that are identical or carry a recorded
bit-exact replay proof. ``sourceRevisionId`` keeps the revision the experiment
actually ran on. Spatial values are never copied into the semantic record; the
page derives them from the bound revision's own measurements.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

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
EXPERIMENT_DIR = "semantic-experiment"
PAGE_DIR = "semantic"
ANALYZE_EVIDENCE = ["applicability_confirmation", "verified_metric_scale", "hazard_relationship", "policy_source_and_version"]
BOUND_EVIDENCE = ("functional_identity", "equipment_operation_context", "stopping_performance_and_detection_zone")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _core(result):
    """Analyze output, also from a copy a page attached earlier (crop prefix, scene IDs)."""
    core = json.loads(json.dumps(result))
    core["timing"].pop("run", None)
    for obj in core["objects"]:
        for ref in obj["sourceRefs"]:
            if "experimentObservationId" in ref:
                ref["observationId"] = ref.pop("experimentObservationId")
            ref["cropPath"] = ref["cropPath"].removeprefix(PAGE_DIR + "/")
        policy = obj["policyContext"]
        if policy["missingEvidence"][:len(ANALYZE_EVIDENCE)] != ANALYZE_EVIDENCE or not set(
                policy["missingEvidence"][len(ANALYZE_EVIDENCE):]) <= set(BOUND_EVIDENCE):
            raise ValueError("Semantic policy evidence differs from the analyzed experiment")
        obj["policyContext"] = {"applicability": policy["applicability"], "machineResult": policy["machineResult"],
                                "candidateTopics": [], "missingEvidence": list(ANALYZE_EVIDENCE)}
        if policy["applicability"] != "unknown" or policy["machineResult"] is not None:
            raise ValueError("A semantic experiment cannot carry a policy result")
    return core


def _frames_match(root, sources):
    """Frame identity: same file, or a replay that proved the source frame's valid points."""
    replay_path = root / "replay-manifest.json"
    replay = json.loads(replay_path.read_text())["frames"] if replay_path.is_file() else {}
    proofs = {}
    for source in sources:
        name = source["file"]
        if not name.startswith("frame_"):
            continue
        current = _sha(root / name)
        if current == source["sha256"]:
            proofs[name] = "identical file"
            continue
        entry = replay.get(name, {})
        if entry.get("sha256") != current or entry.get("originalSha256") != source["sha256"]:
            raise ValueError("Semantic experiment frame changed: " + name)
        proofs[name] = "replayed; " + entry["check"]
    return proofs


def bind(root, report, experiment=None):
    """The semanticExperiment block for this revision, or ValueError when inputs differ."""
    import cv2
    from scripts.route_jev import tight_crop
    from scripts.workcell_semantic_match import observed_points, polygon_mask
    from scripts.workcell_photo_oneshot import _array, _frame

    root = Path(root)
    experiment = Path(experiment or root / EXPERIMENT_DIR)
    if not (experiment / "spend-ledger.json").is_file():
        raise ValueError("Semantic experiment has no spend ledger")
    ledger = json.loads((experiment / "spend-ledger.json").read_text())
    if ledger["status"] != "completed":
        raise ValueError("Cannot publish a failed experiment as completed")
    missing = [name for name in ("semantic-experiment.json", "manifest.json") if not (experiment / name).is_file()]
    if missing:
        raise ValueError("Semantic experiment is incomplete: missing " + ", ".join(missing))
    result = _core(json.loads((experiment / "semantic-experiment.json").read_text()))
    manifest = json.loads((experiment / "manifest.json").read_text())
    if manifest["sourceRevisionId"] != result["sourceRevisionId"]:
        raise ValueError("Semantic manifest and result name different revisions")
    sources = result["protocol"]["sources"]
    for source in sources:
        if source["file"].startswith("photo-") and _sha(root / source["file"]) != source["sha256"]:
            raise ValueError("Semantic experiment photo changed: " + source["file"])
    frames_proof = _frames_match(root, sources)
    transform = np.asarray(report["sceneTransformNative"], float)
    if not np.allclose(manifest["sceneTransformNative"], transform, atol=1e-9, rtol=0):
        raise ValueError("Semantic experiment used another floor transform")
    catalog = {item["id"]: item for item in json.loads((root / "objects.json").read_text())["objects"]}
    document = report["revision"]["document"]
    entities = {e["id"]: e for e in document["entities"] if not e.get("sourceContext")}
    observations = {o["id"]: o for o in document["observations"]}
    if {o["entityId"] for o in result["objects"]} != set(entities) or set(entities) != set(catalog):
        raise ValueError("Semantic object set differs from rendered scene")
    expected = [(item_id, index) for item_id, item in catalog.items() for index in range(len(item["observations"]))]
    if sorted((row["entityId"], row["sourceObservationIndex"]) for row in manifest["observations"]) != sorted(expected):
        raise ValueError("Semantic observations differ from this catalog")
    rasters, crop_difference, scene_ids = {}, 0, {}
    for row in manifest["observations"]:
        source = catalog[row["entityId"]]["observations"][row["sourceObservationIndex"]]
        photo = row["photo"]
        if source["photo"] != photo:
            raise ValueError("Semantic observation photo changed: " + row["observationId"])
        if photo not in rasters:
            frame = _frame(root, photo)
            rasters[photo] = (cv2.imread(str(root / f"photo-{photo}.png")), _array(frame["pts3d"]), _array(frame["non_ambiguous_mask"]))
        bgr, points, valid = rasters[photo]
        mask = polygon_mask(source["polygons"], bgr.shape[:2])
        support, count = observed_points(points, valid, mask, transform)
        crop, crop_mask = tight_crop(bgr, source["polygons"])
        saved_mask = cv2.imread(str(experiment / row["maskPath"]), cv2.IMREAD_UNCHANGED)
        saved_crop = cv2.imread(str(experiment / row["cropPath"]))
        if (int(mask.sum()) != row["maskPixels"] or count != row["supportedPixels"] or len(support) != row["sampledPoints"]
                or saved_mask is None or not np.array_equal(crop_mask, saved_mask) or saved_crop is None or saved_crop.shape != crop.shape):
            raise ValueError("Semantic observation input changed: " + row["observationId"])
        # Same photo bytes and polygon; only the resampler implementation can differ.
        crop_difference = max(crop_difference, int(np.abs(crop.astype(int) - saved_crop.astype(int)).max()))
        ids = [oid for oid in entities[row["entityId"]]["observationRefs"] if observations[oid]["imageId"] == f"photo-{photo}"
               and observations[oid]["originalPixelPolygons"] == source["polygons"]]
        if len(ids) != 1:
            raise ValueError("Ambiguous semantic-to-scene observation link: " + row["observationId"])
        scene_ids[row["observationId"]] = ids[0]
    for obj in result["objects"]:
        for ref in obj["sourceRefs"]:
            ref["experimentObservationId"], ref["observationId"] = ref["observationId"], scene_ids[ref["observationId"]]
            ref["sceneObservationId"] = ref["observationId"]
            ref["cropPath"] = PAGE_DIR + "/" + ref["cropPath"]
        candidates = {r["top3"][0]["id"] for r in obj["results"] if r["variant"] == "multiview" and r["top3"]}
        policy = obj["policyContext"]
        policy["candidateTopics"] = sorted({TOPICS[c] for c in candidates if c in TOPICS})
        policy["candidateBasis"] = "Top-1 multiview predictions from each encoder; candidate inspection topics only, no policy clause applied"
        policy["missingEvidence"] = list(dict.fromkeys(policy["missingEvidence"] + ["functional_identity", "equipment_operation_context"]
                                                       + (["stopping_performance_and_detection_zone"] if {"light_curtain", "robot"} & candidates else [])))
        policy["spatialFacts"] = "Derived on the page from this revision's measurements; no value copied here"
    run = {"containerSeconds": ledger["functionSeconds"], "callSeconds": ledger["callSeconds"], "estimateUsd": ledger["estimateUsd"],
           "callWindowEstimateUsd": ledger["callWindowEstimateUsd"], "actualBilledUsd": ledger["actualBilledUsd"]}
    previous = experiment / "previous-attempt-spend-ledger.json"
    if previous.is_file():
        prior = json.loads(previous.read_text())
        run.update(allAttemptsCallWindowEstimateUsd=prior["callWindowEstimateUsd"] + ledger["callWindowEstimateUsd"],
                   previousAttemptStatus=prior["status"])
    result["timing"]["run"] = run
    revision = report["revision"]
    result["binding"] = {
        "revisionId": revision["id"], "documentSha256": revision["documentSha256"],
        "experimentRevisionId": result["sourceRevisionId"],
        "reuse": "identical semantic inputs verified" if revision["id"] == result["sourceRevisionId"] else "reused on a revision with identical semantic inputs",
        "verified": ["photo bytes", "catalog observation set", "observation masks rebuilt pixel-identically from this revision's polygons",
                     "supported and sampled point counts", "rigid floor transform", "scene observation links"],
        "framesProof": frames_proof, "maxCropResampleDifference": crop_difference,
        "notInputs": "Model GLBs, representations, ground clearances and scale are not experiment inputs; spatial facts come from this revision"}
    return result


def package(root, page, semantic):
    """Copy exactly the bound experiment's evidence next to the page."""
    import shutil
    experiment = Path(root) / EXPERIMENT_DIR
    destination = Path(page) / PAGE_DIR
    destination.mkdir(parents=True, exist_ok=True)
    referenced = {ref["cropPath"].removeprefix(PAGE_DIR + "/") for obj in semantic["objects"] for ref in obj["sourceRefs"]}
    manifest = json.loads((experiment / "manifest.json").read_text())
    referenced |= {row[key] for row in manifest["observations"] for key in ("cropPath", "maskPath")}
    for relative in sorted(referenced):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not (experiment / path).is_file():
            raise ValueError("Missing or unsafe semantic crop: " + relative)
        (destination / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(experiment / path, destination / path)
    for name in ("input-manifest.json", "spend-ledger.json", "manifest.json", "previous-attempt-spend-ledger.json",
                 "timing-pe-core-l.json", "timing-siglip2-so400m.json"):
        if (experiment / name).is_file():
            shutil.copyfile(experiment / name, destination / name)
    (destination / "semantic-experiment.json").write_text(json.dumps(semantic, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
