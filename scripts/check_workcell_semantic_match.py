"""CPU self-check: python scripts/check_workcell_semantic_match.py."""
import base64
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from workcell_semantic_match import (ENCODERS, analyze, associate, fuse, nearest_matches, normalize, observed_points,
                                    point_overlaps, prepare, read_json, sha256, write_json)


def must_fail(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("Invalid input was accepted")


def check_pipeline():
    """Synthetic cached features exercise the full CPU path, never neural inference."""
    import cv2
    def encoded(array):
        return {"shape": list(array.shape), "dtype": str(array.dtype), "data": base64.b64encode(array.tobytes()).decode()}
    with TemporaryDirectory(prefix="semantic-self-check-") as directory:
        root, out = Path(directory) / "input", Path(directory) / "output"
        root.mkdir()
        rgb = np.full((12, 12, 3), 128, np.uint8)
        yy, xx = np.indices(rgb.shape[:2])
        points = np.stack((xx * .01, yy * .01, np.ones_like(xx)), axis=-1).astype(np.float32)
        for photo in (1, 2):
            cv2.imwrite(str(root / f"photo-{photo}.png"), rgb)
            frame = {"image": encoded(rgb), "pts3d": encoded(points), "non_ambiguous_mask": encoded(np.ones((12, 12), bool))}
            with gzip.open(root / f"frame_{photo:04d}.json.gz", "wt") as stream:
                import json
                json.dump(frame, stream)
        observation = {"polygons": [[[1, 1], [5, 1], [5, 5], [1, 5]]]}
        write_json(root / "objects.json", {"objects": [{"id": "one", "label": "Original proxy", "observations": [dict(observation, photo=p) for p in (1, 2)]}]})
        document = {"entities": [{"id": "one", "observationRefs": ["scene-1", "scene-2"]}],
                    "observations": [{"id": f"scene-{p}", "imageId": f"photo-{p}", "originalPixelPolygons": observation["polygons"]} for p in (1, 2)]}
        write_json(root / "scene-report.json", {"revision": {"id": "synthetic", "document": document}, "sceneTransformNative": np.eye(4).tolist()})
        config = {"classes": [{"id": "a", "label": "Class A", "phrase": "a shape"}, {"id": "b", "label": "Class B", "phrase": "another shape"}],
                  "queries": [{"id": "q", "label": "Query", "phrase": "a shape"}], "referenceClasses": {"one": "a"},
                  "overlapNative": .04, "overlapThreshold": .35, "semanticThreshold": .75}
        result = prepare(root, out, config)
        assert result["observations"] == 2 and result["unavailableGeometry"] == 0
        def embeddings():
            for key in ENCODERS:
                np.savez_compressed(out / f"embeddings-{key}.npz", plain=np.array([[1., 0.], [1., 0.]]),
                    masked=np.array([[1., 0.], [1., 0.]]), **{"global": np.array([[1., 0.], [1., 0.]])},
                    classText=np.eye(2), queryText=np.array([[1., 0.]]), observationIds=np.array(["obs-0000", "obs-0001"]),
                    photoIds=np.array([1, 2]), classPhrases=np.array(["a shape", "another shape"]), queryPhrases=np.array(["a shape"]),
                    manifestSha256=np.array(sha256(out / "manifest.json")))
                write_json(out / f"timing-{key}.json", {"syntheticSelfCheck": True})
        embeddings()
        analyze(root, out, config)
        original = read_json(out / "semantic-experiment.json")
        assert len(original["summary"]) == 6 and all(row["crossView"]["referenceCorrect"] == 2 for row in original["summary"])
        assert original["objects"][0]["results"][0]["top3"][0]["id"] == "a"
        assert [ref["sceneObservationId"] for ref in original["objects"][0]["sourceRefs"]] == ["scene-1", "scene-2"]
        config["referenceClasses"] = {"one": "b"}
        config["classes"][0]["label"] = "Permuted display label"
        analyze(root, out, config)
        permuted = read_json(out / "semantic-experiment.json")
        assert original["associations"] == permuted["associations"]
        assert all(row["referenceAgreement"]["correct"] == 0 for row in permuted["summary"])
        assert permuted["objects"][0]["results"][0]["top3"][0]["id"] == "a"
        config["classes"][0]["phrase"] = "a changed neural input"
        must_fail(lambda: analyze(root, out, config))


def main():
    # Shared inference imports without requiring the Modal SDK or creating cloud apps.
    import sys
    from fast_report.visual_encoder import Encoder
    assert callable(Encoder) and "modal" not in sys.modules
    # Identical semantic features describe two distinct objects ten units apart.
    points = [np.array([[x, 0., 0.], [x + .01, 0., 0.]]) for x in (0., 10., .001, 10.001)]
    photos = [1, 1, 2, 2]
    features = normalize(np.ones((4, 3)))
    overlap = point_overlaps(points, photos, .04)
    clusters, edges = associate(features, photos, overlap, .35, .75)
    assert clusters == [[0, 2], [1, 3]] and len(edges) == 2
    assert overlap[0, 3] == 0 and overlap[0, 2] == 1
    # Evaluation identity permutation cannot enter or change association.
    original = ["a", "b", "a", "b"]
    permuted = ["robot", "robot", "guard", "cart"]
    first, _ = nearest_matches(features, photos, original, overlap, .35)
    second, _ = nearest_matches(features, photos, permuted, overlap, .35)
    assert first["referenceCorrect"] == 4 and second["referenceCorrect"] == 0
    assert associate(features, photos, overlap, .35, .75) == (clusters, edges)
    # A transitive bridge must not introduce two observations from one photo.
    crowded = np.ones((4, 4))
    more, diagnostics = associate(features, [1, 2, 3, 1], crowded, .35, .75)
    assert any(e["decision"] == "rejected_duplicate_photo" for e in diagnostics)
    assert all(len(group) == len(set([1, 2, 3, 1][i] for i in group)) for group in more)
    # Per-photo softmax weights and every returned feature retain unit length.
    plain = np.array([[1., .2, 0], [0., 1., .2], [1., 0., .1], [.2, .9, 0]])
    masked = np.array([[1., 0., 0], [0., 1., 0], [.9, 0., .2], [0., .8, .2]])
    local, fused, weights = fuse(plain, masked, [[1., 1., .1], [.4, 1., .1]], [0, 0, 1, 1])
    assert np.allclose(np.linalg.norm(local, axis=1), 1)
    assert np.allclose(np.linalg.norm(fused, axis=1), 1)
    assert np.allclose([weights[:2].sum(), weights[2:].sum()], 1)
    assert not np.allclose(fused, normalize(plain))
    valid, mask = np.ones((2, 2), bool), np.ones((2, 2), bool)
    pointmap = np.arange(12, dtype=float).reshape(2, 2, 3)
    kept, count = observed_points(pointmap, valid, mask, np.eye(4), cap=2)
    assert count == 4 and np.array_equal(kept, pointmap.reshape(-1, 3)[[0, 3]])
    empty, count = observed_points(pointmap, np.zeros_like(valid), mask, np.eye(4))
    assert count == 0 and empty.shape == (0, 3)
    assert np.isnan(point_overlaps([empty, points[0]], [1, 2], .04)).all()
    pointmap[0, 0, 0] = np.nan
    must_fail(lambda: observed_points(pointmap, valid, mask, np.eye(4)))
    must_fail(lambda: observed_points(np.zeros((2, 3, 3)), valid, mask, np.eye(4)))
    must_fail(lambda: observed_points(np.zeros((2, 2, 3)), valid, mask, np.eye(4) * 2))
    must_fail(lambda: point_overlaps([np.array([[np.nan, 0, 0]])], [1], .04))
    must_fail(lambda: normalize([[0, 0, 0]]))
    check_pipeline()
    print("PASS: label-independent matching, spatial separation, one view per photo, normalized fusion, valid observed-point support, CPU pipeline and stale embedding rejection")


if __name__ == "__main__":
    main()
