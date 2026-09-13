from copy import deepcopy
import io
import json
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from PIL import Image
import pytest

from ehs_spatial.platform.contracts import PlatformError
from scripts.import_geometry_evidence import (FRAME_FILES, SELECTION_RULE, geometry_dependencies,
    import_geometry_evidence, read_cloud, sha, write_cloud)


def test_frozen_cloud_import_preserves_points_colors_and_rejects_wrong_source(tmp_path):
    root, published = tmp_path / "run", tmp_path / "published"
    frame = root / "geometry/frames/frame_0001"
    frame.mkdir(parents=True); published.mkdir()
    points = np.array([[[0, 0, 1], [1, 0, 1], [2, 0, 1]], [[0, 1, 1], [1, 1, 1], [2, 1, 1]]], dtype=np.float32)
    rgb = np.array([[[10, 20, 30], [40, 50, 60], [70, 80, 90]], [[15, 25, 35], [45, 55, 65], [75, 85, 95]]], dtype=np.uint8)
    alpha = np.array([[False, True, True], [False, True, True]])
    confidence = np.ones((2, 3), np.float32); confidence[1, 2] = .05
    valid = np.ones((2, 3), bool); valid[0, 2] = False
    content = alpha & valid & (confidence >= .1)
    arrays = {"pts3d.npy": points, "conf.npy": confidence, "valid_mask.npy": valid,
              "content_valid_mask.npy": content, "intrinsics.npy": np.eye(3), "camera_to_world.npy": np.eye(4)}
    for name, value in arrays.items():
        np.save(frame / name, value)
    np.save(root / "alpha.npy", alpha)
    Image.fromarray(rgb).save(frame / "canonical.png")
    Image.fromarray(rgb).save(root / "input.jpg")
    (published / "photo.jpg").write_bytes((root / "input.jpg").read_bytes())
    (published / "canonical.png").write_bytes((frame / "canonical.png").read_bytes())
    rgba = np.column_stack((rgb[valid], np.full(valid.sum(), 255, np.uint8)))
    (root / "geometry/point_cloud.glb").write_bytes(write_cloud(points[valid], rgba))
    frozen = {"experiment": "test-same-frame", "frames": [{"frame_id": "frame_0001", "input": "input.jpg",
        "sha256": sha((root / "input.jpg").read_bytes()), "width": 3, "height": 2,
        "canonical": "geometry/frames/frame_0001/canonical.png", "canonical_sha256": sha((frame / "canonical.png").read_bytes()),
        "alpha": "alpha.npy", "alpha_sha256": sha((root / "alpha.npy").read_bytes()), "content_rect_xyxy": [1, 0, 3, 2],
        "input_to_canonical_pixel_centres": np.eye(3).tolist()}], "geometry": {"content_valid_rule": SELECTION_RULE,
        "frames": [{"frame_id": "frame_0001", "valid_content_points": int(content.sum()),
        "files": {name: sha((frame / name).read_bytes()) for name in FRAME_FILES}}]}}
    (root / "manifest.json").write_text(json.dumps(frozen))
    manifest_hash = sha((root / "manifest.json").read_bytes())
    camera = {"id": "frame_0001", "image": "canonical.png", "original_image": "photo.jpg", "width": 3, "height": 2,
        "original_width": 3, "original_height": 2, "K": np.eye(3).tolist(), "original_K": np.eye(3).tolist(),
        "camera_to_world": np.eye(4).tolist(), "input_to_canonical_pixel_centres": np.eye(3).tolist()}
    source = {"source_run_id": "test-same-frame", "cameras": [camera], "provenance": {
        "source_bridge": {"target_manifest_sha256": manifest_hash},
        "observed_ranges": {"source_run_id": "test-same-frame", "source_sha256": {"manifest.json": manifest_hash}}}}
    doc = {"entities": [{"id": "existing", "label": "Do not mutate", "representations": []}], "assets": [],
        "cameras": [{"id": "camera", "coordinateFrameId": "native", "cameraToWorld": np.eye(4).tolist(), "K": np.eye(3).tolist()}]}
    baseline = deepcopy(doc["entities"])
    manifest = {"cameraIds": {"frame_0001": "camera"}}
    assets = {}

    def include(raw, media_type, metadata, source_key=None):
        asset_id = sha(raw)
        assets[asset_id] = raw
        doc["assets"].append({"id": asset_id, "mediaType": media_type, **metadata})
        return asset_id

    ident = lambda kind, key: str(uuid5(NAMESPACE_URL, kind + ":" + key))
    deps = geometry_dependencies(root, source)
    assert set(deps) == {p.resolve() for p in root.rglob("*") if p.is_file()}
    evidence = import_geometry_evidence(root, published / "scene.json", source, doc, manifest, include, ident, "native")
    out_points, out_colors = read_cloud(assets[evidence["pointCloudAssetId"]])
    np.testing.assert_array_equal(out_points, points[content])
    np.testing.assert_array_equal(out_colors[:, :3], rgb[content])
    assert (out_colors[:, 3] == 255).all()
    assert evidence["contentPointCount"] == 2 and evidence["sourcePointCount"] == 5 and evidence["newModelCalls"] == 0
    assert doc["entities"][:-1] == baseline
    cloud = doc["entities"][-1]
    assert cloud["sourceContext"] and cloud["representations"][0]["kind"] == "point_cloud"
    assert cloud["representations"][0]["transform"]["position"] == [0, 0, 0]
    assert str(tmp_path) not in json.dumps(evidence)
    assert "pts3d.npy" in evidence["frames"][0]["assets"]
    before = deepcopy(doc)
    broken = deepcopy(source); broken["provenance"]["source_bridge"]["target_manifest_sha256"] = "0" * 64
    with pytest.raises(PlatformError, match="source_hash_mismatch"):
        import_geometry_evidence(root, published / "scene.json", broken, doc, manifest, include, ident, "native")
    assert doc == before
    # An existing cloud with an unrelated coordinate pose must never be accepted
    # because its point count happens to match.
    wrong = points[valid].copy(); wrong[:, 0] += 1
    (root / "geometry/point_cloud.glb").write_bytes(write_cloud(wrong, rgba))
    with pytest.raises(PlatformError, match="cloud_native_mismatch"):
        import_geometry_evidence(root, published / "scene.json", source, doc, manifest, include, ident, "native")
    assert doc == before
