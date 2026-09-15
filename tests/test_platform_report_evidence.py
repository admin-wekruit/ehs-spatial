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


def test_source_cad_coverage_uses_exact_mask_owner_and_keeps_original_geometry():
    from copy import deepcopy
    from ehs_spatial.platform.source_cad import refresh_source_cad_links
    payloads, assets = {}, []
    def asset(aid, value):
        raw = json.dumps(value).encode()
        payloads[aid] = raw
        assets.append({"id": aid, "sha256": hashlib.sha256(raw).hexdigest(), "sizeBytes": len(raw), "mediaType": "application/json"})
        return assets[-1]["sha256"]
    image_sha = asset("image", {"image": "photo"})
    sam_sha = asset("sam", {"rle": [json.dumps({"size": [2, 3], "counts": [0, 6]})]})
    inventory_sha = asset("inventory", {"objects": [{"inv": i, "frame": "old" if i < 4 else "missing-photo", "label": "arbitrary fence", "instance": 0} for i in range(5)]})
    asset("map", {"inventory_sha256": inventory_sha})
    asset("manifest", {"schemaVersion": 1, "kind": "source_cad_segmentation_manifest", "sourceRunId": "source-run", "inventorySha256": inventory_sha,
        "files": [{"path": "inventory/sam/old__arbitrary_fence.json", "sha256": sam_sha, "sizeBytes": len(payloads["sam"])}]})
    mapping = {"target_frame_id": "new", "source_image_sha256": "a" * 64, "target_image_sha256": image_sha,
               "source_canonical_to_target_canonical": np.eye(3).tolist()}
    source = {"id": "arbitrary-record", "provenance": {"source_image_sha256": image_sha, "source_record": {
        "candidate_id": "arbitrary-record", "source_frame_id": "old", "target_frame_id": "new", "capture_mapping": mapping,
        "source_mask": {"ref": {"path": "inventory/sam/old__arbitrary_fence.json", "sha256": sam_sha,
                                    "pointer": ["rle", 0], "encoding": "rle", "shape_hw": [2, 3]}}}}}
    asset("source", {"observed_regions": [source]})
    document = {"assets": assets, "entities": [{"id": "current-owner", "observationRefs": ["obs"]}, {"id": "old-explicit", "observationRefs": []}],
        "observations": [{"id": "obs", "revision": 3, "imageId": "image", "sourceRefs": [{"assetId": "source", "sourceRecordId": "arbitrary-record"}]}],
        "reportEvidence": {"frameRelations": {"frames": [{"sourceFrameId": "old", "targetFrameId": "new", "sourceImageSha256": "a" * 64,
            "targetImageSha256": image_sha, "sourceCanonicalToTargetCanonical": np.eye(3).tolist()}]},
            "historical": {"runId": "source-run", "sourceCadManifestAssetId": "manifest", "inventory": [{"inventoryIndex": 0, "entityIds": ["old-explicit"]}, {"inventoryIndex": 1, "entityIds": []},
                                           {"inventoryIndex": 4, "entityIds": []}],
                "findings": [{"status": "FAIL", "measured": 7}], "cad": {"sourceRefs": [{"assetId": "map"}], "regions": [
                    {"inventoryIndex": i, "entityIds": [], "polygon": [[i, 0], [i, 1], [i + 1, 1]]} for i in (0, 1, 4)]}}}}
    # An existing explicit binding conflicting with a new proof must remain unresolved.
    before = deepcopy(document)
    masks = {"obs": np.ones((2, 3), bool)}
    result = refresh_source_cad_links(document, payloads.__getitem__, masks, inventory_asset_id="inventory")
    assert [r["reason"] for r in result["records"]] == ["ambiguous_entity_ownership", "exact_source_mask", "no_verified_same_photo"]
    assert result["linkedRecordCount"] == 1 and result["unresolvedRecordCount"] == 2
    assert result["geometryRegistration"] == "not_registered" and not result["geometryConstraintsApplied"]
    proof = result["records"][1]["evidence"][0]
    assert proof["entityId"] == "current-owner" and proof["observationRevision"] == 3
    assert proof["sourceRefs"][0] == {"assetId": "inventory", "jsonPointer": "/objects/1"}
    assert proof["canonicalMaskSha256"] == hashlib.sha256(masks["obs"].tobytes()).hexdigest()
    assert refresh_source_cad_links(document, payloads.__getitem__, masks)["records"][0]["reason"] == "ambiguous_entity_ownership"
    assert document["entities"] == before["entities"] and document["observations"] == before["observations"]
    assert document["reportEvidence"]["historical"]["findings"] == before["reportEvidence"]["historical"]["findings"]
    assert [r["polygon"] for r in document["reportEvidence"]["historical"]["cad"]["regions"]] == [r["polygon"] for r in before["reportEvidence"]["historical"]["cad"]["regions"]]
    # A current mask edit invalidates exact evidence; the old link is not promoted into authority.
    masks["obs"][0, 0] = False
    invalidated = refresh_source_cad_links(document, payloads.__getitem__, masks)
    assert invalidated["records"][1]["entityIds"] == [] and invalidated["records"][1]["reason"] == "canonical_masks_differ"
    # Same source evidence with multiple current owners cannot select either silently.
    ambiguous = deepcopy(before)
    ambiguous["entities"].append({"id": "another-owner", "observationRefs": ["obs-2"]})
    ambiguous["observations"].append({**deepcopy(ambiguous["observations"][0]), "id": "obs-2"})
    masks = {oid: np.ones((2, 3), bool) for oid in ("obs", "obs-2")}
    result = refresh_source_cad_links(ambiguous, payloads.__getitem__, masks, inventory_asset_id="inventory")
    assert result["records"][1]["reason"] == "ambiguous_entity_ownership"
    # A different coordinate mapping is not treated as exact canonical mask evidence.
    wrong_grid = deepcopy(before)
    wrong_grid["reportEvidence"]["frameRelations"]["frames"][0]["sourceCanonicalToTargetCanonical"][0][0] = 2
    result = refresh_source_cad_links(wrong_grid, payloads.__getitem__, masks, inventory_asset_id="inventory")
    assert result["records"][1]["entityIds"] == []
    assert result["records"][1]["reason"] == "canonical_grid_mapping_requires_review"
    corrupted = {**payloads, "inventory": b"corrupt"}
    with pytest.raises(PlatformError, match="source_cad_asset_hash_mismatch"):
        refresh_source_cad_links(deepcopy(before), corrupted.__getitem__, masks, inventory_asset_id="inventory")
    # A matching filename/instance is insufficient without the independent source-run hash.
    missing_manifest = deepcopy(before)
    del missing_manifest["reportEvidence"]["historical"]["sourceCadManifestAssetId"]
    result = refresh_source_cad_links(missing_manifest, payloads.__getitem__, masks, inventory_asset_id="inventory")
    assert result["records"][0]["entityIds"] == ["old-explicit"]
    assert result["records"][0]["proofStatus"] == "explicit_binding_only"
    assert result["records"][1]["entityIds"] == [] and result["records"][1]["reason"] == "source_segmentation_manifest_missing"
    wrong_source = deepcopy(before)
    manifest = json.loads(payloads["manifest"])
    manifest["files"][0]["sha256"] = "b" * 64
    raw = json.dumps(manifest).encode()
    wrong_payloads = {**payloads, "manifest": raw}
    next(a for a in wrong_source["assets"] if a["id"] == "manifest").update(sha256=hashlib.sha256(raw).hexdigest(), sizeBytes=len(raw))
    result = refresh_source_cad_links(wrong_source, wrong_payloads.__getitem__, masks, inventory_asset_id="inventory")
    assert result["records"][1]["entityIds"] == [] and result["records"][1]["reason"] == "source_segmentation_hash_mismatch"


def test_source_cad_manifest_freezes_source_files_independently(tmp_path):
    from scripts.import_report_evidence import build_source_cad_manifest
    root = tmp_path / "inventory"
    (root / "sam").mkdir(parents=True)
    raw = json.dumps({"objects": [{"inv": 7, "frame": "frame-a", "label": "any fence", "instance": 1}]}).encode()
    (root / "inventory.json").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    sam = root / "sam/frame-a__any_fence.json"
    sam.write_bytes(b'{"rle": []}')
    original = json.loads(build_source_cad_manifest(tmp_path, sha, source_run_id="original"))
    assert original["inventorySha256"] == sha and original["sourceRunId"] == "original"
    assert original["files"] == [{"path": "inventory/sam/frame-a__any_fence.json", "sha256": hashlib.sha256(sam.read_bytes()).hexdigest(), "sizeBytes": len(sam.read_bytes())}]
    sam.write_bytes(b'{"rle": ["different"]}')
    updated = json.loads(build_source_cad_manifest(tmp_path, sha, source_run_id="original"))
    assert updated["files"][0]["sha256"] != original["files"][0]["sha256"]
    with pytest.raises(PlatformError, match="import_report_source_hash_mismatch"):
        build_source_cad_manifest(tmp_path, "c" * 64, source_run_id="original")


def test_source_equivalence_proof_requires_exact_instance_and_mask(tmp_path):
    from copy import deepcopy
    import scripts.import_report_evidence as importer
    assert hasattr(importer, "import_source_equivalences"), "same-source provenance must produce an immutable identity proof"
    original = tmp_path / "arbitrary-segmentation.json"
    original.write_text(json.dumps({"rle": [json.dumps({"size": [2, 3], "counts": [0, 6]}), json.dumps({"size": [2, 3], "counts": [6]})]}))
    checksum = hashlib.sha256(original.read_bytes()).hexdigest()
    source = {"nested": {"anything": {"candidate_id": "raw-any-name", "target_frame_id": "photo-frame",
        "source_frame_id": "older-frame", "source_mask": {"ref": {"sha256": checksum, "pointer": ["rle", 0],
        "encoding": "rle", "shape_hw": [2, 3]}}}}}
    view = {"frame_id": "photo-frame", "provenance": {"source_path": str(original), "source_sha256": checksum,
        "source_instance": 0, "source_frame": "older-frame"}}
    raw = json.dumps({"objects": [{"object_id": "model-any-name", "views": [view]}]}).encode()
    records = [{"kind": "objects", "sourceRecordId": "model-any-name", "view": view,
        "sourceRaw": raw, "jsonPointer": "/objects/0/views/0"}]
    document = {"assets": [{"id": "photo", "sha256": "a"*64}, {"id": "scene-asset", "sha256": hashlib.sha256(json.dumps(source).encode()).hexdigest()}], "cameras": [{"imageId": "photo", "sourceRefs": [{"sourceCameraId": "photo-frame"}]}],
        "entities": [{"id": "raw-entity", "observationRefs": ["raw"], "lineage": [{"operation": "offline_import", "sourceRecordId": "raw-any-name"}]},
                     {"id": "model-entity", "observationRefs": ["model"], "lineage": [{"operation": "offline_import", "sourceRecordId": "model-any-name"}]}],
        "observations": [{"id": x, "revision": 1, "imageId": "photo", "sourceRefs": []} for x in ["raw", "model"]]}
    masks = {x: np.ones((2, 3), bool) for x in ["raw", "model"]}
    saved = {}
    def include(data, media, metadata, key=None):
        identity = hashlib.sha256(data).hexdigest()
        saved[identity] = data
        return identity
    before = deepcopy(document)
    result = importer.import_source_equivalences(document, source, "scene-asset", records, masks, include)
    proof = json.loads(saved[document["sourceIdentityEvidence"][0]["assetId"]])
    pair = proof["pairs"][0]
    assert result["pairCount"] == 1 and pair["kind"] == "canonical_sam_rle"
    assert pair["observationRefs"] == [{"observationId": "model", "revision": 1}, {"observationId": "raw", "revision": 1}]
    assert pair["sourceRef"]["sha256"] == checksum and pair["sourceRef"]["jsonPointer"] == "/rle/0"
    assert saved[pair["sourceRef"]["assetId"]] == original.read_bytes()
    assert {ref["role"] for ref in pair["evidenceRefs"]} == {"raw_source_reference", "native_mask_provenance"}
    assert pair["canonicalMaskSha256"] == hashlib.sha256(masks["raw"].tobytes()).hexdigest()
    assert document["observations"] == before["observations"] and document["entities"] == before["entities"]
    mismatched = deepcopy(masks); mismatched["raw"][0, 0] = False
    assert importer.import_source_equivalences(deepcopy(before), source, "scene-asset", records, mismatched, include)["pairCount"] == 0
    other_instance = deepcopy(source); other_instance["nested"]["anything"]["source_mask"]["ref"]["pointer"][-1] = 1
    assert importer.import_source_equivalences(deepcopy(before), other_instance, "scene-asset", records, masks, include)["pairCount"] == 0
    original.write_text("corrupt")
    with pytest.raises(PlatformError, match="source_hash_mismatch"):
        importer.import_source_equivalences(deepcopy(before), source, "scene-asset", records, masks, include)


def test_native_segmentation_assets_keep_original_bytes_and_canonical_grid(tmp_path):
    from copy import deepcopy
    from scripts.import_report_evidence import import_observation_masks, observation_mask_sources
    root = tmp_path / "geometry"
    root.mkdir()
    Image.new("L", (6, 4), 255).save(root / "mask.png")
    np.save(root / "canonical_mask.npy", np.ones((2, 3), bool))
    view = {"frame_id": "frame", "rgb_path": "original.jpg", "observed_only": True,
        "mask_path": "mask.png", "canonical_mask_path": "canonical_mask.npy",
        "sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ("mask.png", "canonical_mask.npy")}}
    evidence = {"objects": [{"object_id": "source-object", "views": [view]}]}
    (root / "objects.json").write_text(json.dumps(evidence))
    frozen = {"experiment": "source-run", "frames": [{"frame_id": "frame", "input": "original.jpg", "sha256": "a" * 64,
        "width": 6, "height": 4, "input_to_canonical_pixel_centres": [[.5, 0, -.25], [0, .5, -.25], [0, 0, 1]]}],
        "evidence": {"objects": "objects.json", "objects_sha256": hashlib.sha256((root/"objects.json").read_bytes()).hexdigest()}}
    (root / "manifest.json").write_text(json.dumps(frozen))
    source = {"source_run_id": "source-run", "cameras": [{"id": "frame", "width": 3, "height": 2}],
        "provenance": {"source_bridge": {"target_manifest_sha256": hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest()}}}
    document = {"entities": [{"id": "entity", "observationRefs": ["observation"]}],
        "observations": [{"id": "observation", "revision": 1, "imageId": "photo", "maskAssetId": None,
            "originalPixelBox": [1, 1, 2, 2], "originalPixelPolygons": [[[1, 1], [1, 2], [2, 2]]], "missingEvidence": ["source_mask_not_packaged", "other"]}],
        "cameras": [{"id": "camera", "imageId": "photo"}], "geometryEvidence": {"manifestAssetId": "geometry-manifest"}}
    manifest = {"entityIds": {"source-object": "entity"}, "cameraIds": {"frame": "camera"}}
    stored = {}
    def include(raw, media, metadata, key):
        identity = hashlib.sha256(raw).hexdigest()
        stored[identity] = raw
        return identity
    before = deepcopy(document)
    result = import_observation_masks(root, source, document, manifest, include)
    observation = document["observations"][0]
    assert result["attachedObservationIds"] == ["observation"]
    assert observation["id"] == "observation" and observation["revision"] == 1
    assert observation["originalPixelBox"] == before["observations"][0]["originalPixelBox"]
    assert observation["originalPixelPolygons"] == before["observations"][0]["originalPixelPolygons"]
    assert stored[observation["maskAssetId"]] == (root / "mask.png").read_bytes()
    assert stored[observation["maskEvidence"]["canonicalMaskAssetId"]] == (root / "canonical_mask.npy").read_bytes()
    assert observation["maskEvidence"]["originalShape"] == [4, 6] and observation["maskEvidence"]["canonicalShape"] == [2, 3]
    assert observation["missingEvidence"] == ["other"]
    assert set(observation_mask_sources(root, source)[2]) == {root / name for name in ("objects.json", "mask.png", "canonical_mask.npy")}
    assert import_observation_masks(root, source, document, manifest, include)["attachedObservationIds"] == []
    (root / "canonical_mask.npy").write_bytes(b"corrupt")
    saved_count = len(stored)
    with pytest.raises(PlatformError, match="source_hash_mismatch"):
        import_observation_masks(root, source, before, manifest, include)
    assert len(stored) == saved_count and before["observations"][0]["maskAssetId"] is None


def test_identity_batch_registers_real_capture_once_and_preserves_cas(repo, tmp_path):
    from copy import deepcopy
    from ehs_spatial.platform.contracts import canonical, digest
    from ehs_spatial.platform.reconstruction import run_reassociation
    from scripts.research.reprocess_object_identity import make_plan, register_target, assert_source_conserved
    from scripts.research.scan_object_identity import encoded, file_ref
    from test_platform_backend import project
    repo.blobs = LocalBlobStore(tmp_path / "store")
    cap, created = project(repo)
    project_id = created["project"]["id"]
    registry = {}
    def save(raw, media, metadata):
        blob = repo.blobs.put(raw, media)
        blob["metadata"] = metadata
        asset = repo.register_asset(project_id, blob)
        registry[asset["id"]] = {**asset, "file": file_ref(repo.blobs.root / asset["storageKey"])}
        return asset
    source, _ = import_document(make_public_scene(tmp_path), save)
    assert source["captureId"] is None and source["observations"]
    job = repo.create_job(project_id, cap, {"requestId": str(uuid4()), "branchId": created["branch"]["id"],
        "baseRevisionId": created["revision"]["id"], "kind": "import_scene", "inputs": {}, "config": {}})
    job = repo.claim_job(job["id"])
    finished = repo.finish_job(job["id"], job["attemptToken"], "succeeded", document=source)
    base_id = finished["resultRevisionId"]
    publication = repo.create_publication(project_id, cap, {"requestId": str(uuid4()), "sceneRevisionId": base_id,
        "evaluationIds": [], "reviewIds": [], "title": "Frozen source"})
    prepared = deepcopy(source)
    # Same bytes under a provisional preparation ID must resolve to the old asset.
    old_asset = source["assets"][0]
    provisional = str(uuid4())
    prepared["assets"].append({**old_asset, "id": provisional})
    registry[provisional] = {**registry[old_asset["id"]], "id": provisional}
    extra = repo.blobs.put(b"new source evidence", "application/octet-stream")
    extra.update(id=str(uuid4()), metadata={"kind": "identity_test_evidence"})
    prepared["assets"].append({**extra, "kind": "identity_test_evidence"})
    registry[extra["id"]] = {**extra, "file": file_ref(repo.blobs.root / extra["storageKey"])}
    source_path, prepared_path = tmp_path / "base.json", tmp_path / "prepared.json"
    source_path.write_bytes(canonical(source)); prepared_path.write_bytes(canonical(prepared))
    target = {"projectId": project_id, "revisionId": base_id, "document": file_ref(source_path), "publicationIds": [publication["id"]], "catalogPublicationIds": []}
    scan_path = tmp_path / "scan.json"
    scan_path.write_bytes(encoded({"evaluationRevisions": [target], "publications": [{"id": publication["id"]}]}))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(encoded({"sourceScan": file_ref(scan_path), "evaluationRevisions": [{**target,
        "baseDocument": file_ref(source_path), "document": file_ref(prepared_path)}], "assets": list(registry.values())}))
    plan = make_plan(manifest_path); target = plan["targets"][0]
    queued = register_target(repo, repo.blobs, plan, target, registry)
    assert queued["status"] == "queued"
    assert register_target(repo, repo.blobs, plan, target, registry) == queued
    captures = repo.list_project_records(project_id, "captures")["items"]
    assert len(captures) == 1 and captures[0]["id"] == target["capture"]["id"]
    assert captures[0]["task"]["sourceDocumentSha256"] == digest(source)
    job = repo.claim_job(queued["id"])
    document, result = run_reassociation(repo, repo.blobs, job, {})
    assert document["captureIds"] == [captures[0]["id"]]
    assert_source_conserved(source, document)
    assert len({a["id"] for a in document["assets"]}) == len(document["assets"])
    assert result["newModelCalls"] == 0 and result["status"] == "incomplete"
    concurrent = repo.create_job(project_id, cap, {"requestId": str(uuid4()), "branchId": target["branchId"],
        "baseRevisionId": base_id, "kind": "import_scene", "inputs": {}, "config": {}})
    concurrent = repo.claim_job(concurrent["id"])
    moved = repo.finish_job(concurrent["id"], concurrent["attemptToken"], "succeeded", document=source)
    saved = repo.finish_job(job["id"], job["attemptToken"], result["status"], document=document, result=result)
    assert saved["headAdvanced"] is False and saved["resultRevisionId"] != moved["resultRevisionId"]
    assert repo.get_publication(publication["id"])["snapshot"] == publication["snapshot"]
    assert repo.get_revision(base_id)["document"] == source
    assert repo.get_project(project_id)["branch"]["headRevisionId"] == base_id

    # Proof blobs contain asset references too: DB deduplication must resolve
    # them before the proof bytes and prepared document are frozen.
    proof = repo.blobs.put(canonical({"schemaVersion": 1, "kind": "same_source_observation_equivalences",
        "pairs": [{"sourceRef": {"assetId": provisional, "sha256": old_asset["sha256"]},
                   "evidenceRefs": [{"assetId": extra["id"], "sha256": extra["sha256"]}]}]}), "application/json")
    proof.update(id=str(uuid4()), metadata={"kind": "source_identity_evidence"})
    prepared["assets"].insert(0, proof)
    prepared["sourceIdentityEvidence"] = [{"assetId": proof["id"], "sha256": proof["sha256"]}]
    registry[proof["id"]] = {**proof, "file": file_ref(repo.blobs.root / proof["storageKey"])}
    prepared_path.write_bytes(canonical(prepared))
    manifest = json.loads(manifest_path.read_bytes())
    manifest["evaluationRevisions"][0]["document"] = file_ref(prepared_path)
    manifest["assets"] = list(registry.values())
    manifest_path.write_bytes(encoded(manifest))
    proof_plan = make_plan(manifest_path)
    proof_job = register_target(repo, repo.blobs, proof_plan, proof_plan["targets"][0], registry)
    prepared_asset = repo.get_asset(proof_job["inputs"]["preparedDocumentAssetId"])
    actual = json.loads(repo.blobs.get(prepared_asset["storageKey"], prepared_asset["sha256"], prepared_asset["sizeBytes"]))
    proof_ref = actual["sourceIdentityEvidence"][0]
    actual_proof_asset = repo.get_asset(proof_ref["assetId"])
    actual_proof = json.loads(repo.blobs.get(actual_proof_asset["storageKey"], actual_proof_asset["sha256"], actual_proof_asset["sizeBytes"]))
    assert actual_proof["pairs"][0]["sourceRef"]["assetId"] == old_asset["id"]
    assert actual_proof["pairs"][0]["evidenceRefs"][0]["assetId"] != extra["id"]
    assert proof_ref["sha256"] == actual_proof_asset["sha256"] == digest(actual_proof)
    assert next(a for a in actual["assets"] if a["id"] == proof_ref["assetId"])["sha256"] == proof_ref["sha256"]
    assert register_target(repo, repo.blobs, proof_plan, proof_plan["targets"][0], registry) == proof_job


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
