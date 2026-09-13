"""Saved report provenance, pixel geometry and immutable publication checks."""
import hashlib
import json
from uuid import NAMESPACE_URL, uuid5, uuid4

import numpy as np
from PIL import Image
import pytest

from ehs_spatial.platform.contracts import PlatformError
from ehs_spatial.platform.storage import LocalBlobStore
from scripts.import_public_scene import import_document, run_import
from scripts.import_report_evidence import original_box, original_polygons, report_dependencies
from test_platform_import import make_public_scene
from test_platform_backend import repo


def put(data, media_type, metadata):
    sha = hashlib.sha256(data).hexdigest()
    return {"id": str(uuid5(NAMESPACE_URL, sha)), "sha256": sha, "sizeBytes": len(data), "mediaType": media_type, "metadata": metadata}


def report_fixture(root):
    scene_path = make_public_scene(root)
    scene = json.loads(scene_path.read_text())
    scene["objects"][1]["measurements"] = {"status": "available", "dimensions_native": {"height": 3, "width": 1, "depth": 2},
        "basis": {"kind": "floor_aligned_native", "axes_native": np.eye(3).tolist(), "corners_native": [[1, 2, 3], [4, 5, 6]]},
        "quality": {"supported_points": 4, "warnings": ["Visible support only"]}}
    scene_path.write_text(json.dumps(scene))
    Image.new("RGB", (10, 10)).save(root / "cad.png")
    evidence = {"source_run_id": "old-run", "assessment": {"status": "FAIL", "climb_review": {"rationale": "REVIEW only: saved hint"}},
        "policies": [{"id": "p1", "statement": "Saved rule", "status": "FAIL", "warnings": ["saved uncertainty"],
                      "violations": [{"subject_id": "old-entity", "measured": .1, "unit": "m"}], "facts": [{"fact_id": "f1", "value": .1, "unit": "m"}]}],
        "facts": [], "entities": [], "source_sha256": {}}
    (root / "evidence.json").write_text(json.dumps(evidence))
    (root / "cad-map.json").write_text(json.dumps({"inventory_sha256": "a" * 64,
        "image_sha256": hashlib.sha256((root / "cad.png").read_bytes()).hexdigest(),
        "objects": [{"inv": 1, "polygon": [[1, 1], [1, 2], [2, 2]], "centroid": [1.5, 1.5]}]}))
    report = {"source_run_id": "old-run", "reconstruction_run_id": "arbitrary-source", "scene_url": "scene.json",
        "frames": [{"id": "camera-a"}], "objects": [{"id": "generated", "label": "Proposal", "source": "parametric",
            "views": [{"frame_id": "camera-a", "bbox": [1, 1, 3, 4], "polygons": [[[1, 1], [1, 3], [2, 3]]]}],
            "plan": {"hull": [[1, 2], [3, 4]], "z_min": 0, "z_max": 2}, "metrics": {"before_iou": .2, "after_iou": .4, "views": [{"frame_id": "camera-a", "before_depth": .1}]},
            "metrics_source": "Original generated model, not current primitive"}], "plan": {"native_to_floor": np.eye(4).tolist()}, "scale": {"reconstruction": "uncalibrated"},
        "source_sha256": {"legacy_inventory": "a" * 64}, "mapping_notes": ["Frame IDs are scoped per run"],
        "legacy": {"source_run_id": "old-run", "evidence_url": "evidence.json", "summary": {"status": "FAIL"},
            "findings": [{"id": "p1", "title": "Saved rule", "status": "FAIL", "metrics": {"threshold": .6, "unit": "m"}}],
            "inventory": [{"inv": 1, "label": "Proposal", "mapped_object_ids": ["generated"], "mapping_status": "exact_mask"},
                          {"inv": 2, "label": "Proposal", "mapped_object_ids": []}],
            "cad": {"url": "cad.png", "map_url": "cad-map.json", "width": 10, "height": 10, "regions": []}}}
    (root / "report.json").write_text(json.dumps(report))
    return scene_path


def test_canonical_report_keeps_explicit_ids_historical_findings_and_all_assets(tmp_path):
    path = report_fixture(tmp_path)
    document, manifest = import_document(path, put)
    report = document["reportEvidence"]
    assert len(document["entities"]) == 30
    assert report["schemaVersion"] == 1 and report["historical"]["runId"] == "old-run"
    historical = report["historical"]
    assert historical["inventory"][0]["entityIds"] == [manifest["entityIds"]["generated"]]
    assert historical["inventory"][1]["entityIds"] == []  # same label is not identity
    assert historical["findings"][0]["warnings"] == ["saved uncertainty"]
    assert historical["findings"][0]["facts"][0]["factId"] == "f1"
    assert historical["assessment"]["climbReview"]["rationale"] == "REVIEW only: saved hint"
    assert historical["cad"]["regions"][0]["entityIds"] == [manifest["entityIds"]["generated"]]
    assert "not current-scene compliance" in historical["notice"]
    assert "evaluations" not in document
    obj = report["objects"][0]
    assert obj["views"][0]["originalPixelPolygons"] == [[[1, 1], [1, 3], [2, 3]]]
    assert obj["metrics"]["beforeIou"] == .2 and "not current primitive" in obj["metricsMeaning"]
    measurement = document["entities"][1]["measurements"]
    assert [measurement[key] for key in ("widthNative", "depthNative", "groundHeightNative")] == [1, 2, 3]
    assert measurement["basis"]["cornersNative"] == [[1, 2, 3], [4, 5, 6]]
    assert "dimensionsNative" not in measurement
    projected = measurement["projectedHull"]
    assert projected["points"] == [[1, 2], [3, 4]] and projected["nativeToPlane"] == np.eye(4).tolist()
    assert projected["representationSnapshot"] == document["entities"][1]["representations"]
    assert projected["modelTransformSnapshot"] == document["entities"][1]["currentModelTransform"]
    document["entities"][1]["representations"][0]["transform"]["position"][0] = 42
    assert projected["representationSnapshot"][0]["transform"]["position"][0] != 42
    assets = {asset["id"] for asset in document["assets"]}
    def check(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ("assetId", "imageId") and child is not None:
                    assert child in assets
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, str):
            assert str(tmp_path) not in value
    check(report)
    assert import_document(path, put)[1]["entityIds"] == manifest["entityIds"]


def test_report_pixel_centres_and_box_edges_use_distinct_conventions():
    matrix = [[2, 0, .5], [0, 3, 1], [0, 0, 1]]
    assert original_polygons([[[0, 0], [1, 1]]], matrix) == [[[.5, 1], [2.5, 4]]]
    assert original_box([0, 0, 2, 2], matrix, 20, 20) == [0, 0, 4, 6]


def test_report_dependencies_and_source_integrity_are_not_silently_ignored(tmp_path):
    path = report_fixture(tmp_path)
    source = json.loads(path.read_text())
    assert {p.name for p in report_dependencies(path, source)} == {"report.json", "evidence.json", "cad.png", "cad-map.json"}
    (tmp_path / "cad.png").write_bytes(b"changed")
    with pytest.raises(PlatformError, match="import_report_source_hash_mismatch"):
        import_document(path, put)


def test_detection_registry_pointer_and_verified_bridge_are_required_for_entity_links(tmp_path):
    path = report_fixture(tmp_path)
    original = tmp_path / "original-run"
    (original / "detection").mkdir(parents=True)
    photo = (tmp_path / "photo.png").read_bytes()
    (original / "photo.png").write_bytes(photo)
    photo_sha = hashlib.sha256(photo).hexdigest()
    detection = {"run_id": "sweep-run", "detections": [
        {"item_id": "a", "label": "Same label", "frame_id": "old-camera", "source_image_sha256": photo_sha,
         "note": "Saved visual interpretation", "rle": "1 2", "box": [1, 1, 3, 4]},
        {"item_id": "b", "label": "Same label", "frame_id": "old-camera", "source_image_sha256": None,
         "source_binding": "legacy_first_frame_unverified", "rle": "1 2", "box": [1, 1, 3, 4]}], "missing": [], "rejected": []}
    detection_path = original / "detection/detections.json"
    detection_path.write_text(json.dumps(detection))
    detection_sha = hashlib.sha256(detection_path.read_bytes()).hexdigest()
    registry = {"run_id": "sweep-run", "frames": [{"frame_id": "old-camera", "image_path": "photo.png", "width": 8, "height": 6, "source_sha256": photo_sha}],
        "candidates": [{"id": identity, "frame_id": "old-camera", "source_refs": [{"type": "detection", "path": "detection/detections.json", "pointer": ["detections", index, "rle"], "sha256": detection_sha}]}
                       for index, identity in enumerate(("generated", "unrelated"))]}
    registry_path = original / "object-evidence.json"
    registry_path.write_text(json.dumps(registry))
    bridge = {"source_run_id": "sweep-run", "target_run_id": "arbitrary-source",
        "records": [{"id": "generated", "source_frame_id": "old-camera", "target_frame_id": "camera-a"}],
        "mapped_source_frames": {"old-camera": {"target_frame_id": "camera-a", "source_image_sha256": photo_sha,
            "target_image_sha256": photo_sha, "original_to_original": np.eye(3).tolist(), "assumption": "Verified same capture"}}}
    (tmp_path / "bridge.json").write_text(json.dumps(bridge))
    scene = json.loads(path.read_text())
    scene["provenance"] = {"source_bridge": {"audit": "bridge.json", "source_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(), "registration": "Source 3D never used"}}
    path.write_text(json.dumps(scene))
    document, manifest = import_document(path, put, observation_root=original)
    sweep = document["reportEvidence"]["imageInterpretations"][0]
    assert sweep["runId"] == "sweep-run" and len(sweep["items"]) == 2
    assert sweep["items"][0]["entityIds"] == [manifest["entityIds"]["generated"]]
    assert sweep["items"][0]["imageId"] == document["cameras"][0]["imageId"]
    assert sweep["items"][0]["originalPixelBox"] == [1, 1, 3, 4]
    assert sweep["items"][0]["note"] == "Saved visual interpretation"
    assert sweep["items"][1]["imageId"] is None and sweep["items"][1]["entityIds"] == []
    assert sweep["items"][1]["mappingLimitation"] == "Source-image identity unverified"
    detection["detections"][0]["note"] = "Changed after the pinned registry"
    detection_path.write_text(json.dumps(detection))
    with pytest.raises(PlatformError, match="import_report_detection_binding_invalid"):
        import_document(path, put, observation_root=original)


def test_new_evidence_creates_fixed_publication_without_replacing_edits(repo, tmp_path):
    path = report_fixture(tmp_path)
    blobs = LocalBlobStore(tmp_path / "blobs")
    out = tmp_path / "imports"
    first = run_import(path, repo, blobs, out)
    first_publication = repo.get_publication(first["publicationId"])
    assert first_publication["evaluationIds"] == [] and first_publication["snapshot"]["evaluations"] == []
    assert "reportEvidence" in first_publication["snapshot"]["revision"]["document"]
    assert run_import(path, repo, blobs, out)["publicationId"] == first["publicationId"]
    management = json.loads(next(out.glob("*.management.json")).read_text())
    edit = repo.commit_edits(first["projectId"], management["capability"], {"requestId": str(uuid4()), "branchId": first["branchId"],
        "baseRevisionId": first["sceneRevisionId"], "operations": [{"type": "setLabel", "entityId": first["entityIds"]["generated"], "label": "Human edit"}]})
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text()); report["mapping_notes"].append("new verified provenance")
    report_path.write_text(json.dumps(report))
    second = run_import(path, repo, blobs, out)
    assert second["projectId"] == first["projectId"] and second["entityIds"] == first["entityIds"]
    assert second["publicationId"] != first["publicationId"] and second["branchId"] != first["branchId"]
    assert repo.get_project(first["projectId"])["revision"]["id"] == edit["revision"]["id"]
    assert repo.get_publication(first["publicationId"]) == first_publication
    assert len(repo.get_revision(second["sceneRevisionId"])["document"]["entities"]) == 30
    with repo._connect() as connection:
        assert connection.execute("SELECT count(*) AS n FROM model_calls").fetchone()["n"] == 0
