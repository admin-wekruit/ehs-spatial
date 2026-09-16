"""Observed alignment contracts; synthetic checks do not validate complete shape."""
from copy import deepcopy
import importlib.util

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ehs_spatial.platform.contracts import PlatformError
from ehs_spatial.platform.spatial import MeshData, pixel_center_mapping, transform_points, unproject_pixels


def scene(*, rotated=False):
    camera = np.eye(4)
    if rotated:
        camera[:3, :3] = Rotation.from_euler("xyz", [.21, -.37, .12]).as_matrix()
        camera[:3, 3] = [1.2, -.3, .7]
    k = np.array([[28., 4., 17.5], [0., 24., 11.5], [0., 0., 1.]])
    mask = np.zeros((24, 36), bool)
    mask[4:17, 6:27] = True
    view = {"observationId": "observation-a", "observationRevision": 1,
            "imageId": "image-a", "coordinateFrameId": "native",
            "sourceHashes": {"image": "a" * 64, "mask": "b" * 64,
                             "geometry": "c" * 64, "camera": "d" * 64},
            "mask": mask, "depth": np.full(mask.shape, 4.),
            "valid": np.ones(mask.shape, bool), "K": k,
            "cameraToWorld": camera, "maskComplete": True}
    vertices = unproject_pixels(np.array([[5.5, 3.5], [26.5, 3.5], [26.5, 16.5], [5.5, 16.5]]),
                                np.full(4, 4.), view)
    return MeshData(vertices, np.array([[0, 1, 2], [0, 2, 3]])), view


def test_quality_module_exists():
    assert importlib.util.find_spec("ehs_spatial.platform.model_quality") is not None


def test_camera_centres_skew_non_square_rotated_and_source_binding():
    from ehs_spatial.platform.model_quality import assess_model
    mesh, view = scene(rotated=True)
    report = assess_model(mesh, np.eye(4), [view])
    assert report["status"] == "observed_consistent"
    assert report["semanticShapeStatus"] == "not_assessed"
    assert report["physicalCalibrationStatus"] == "not_assessed"
    score = report["perView"][0]
    assert score["coverage"] == score["precision"] == score["iou"] == 1
    assert score["relativeDepthP95"] < 1e-5
    assert score["targetPixels"] == score["depthComparisonPixels"] == 273
    assert score["observationRevision"] == 1
    assert score["sourceHashes"] == view["sourceHashes"]
    changed = deepcopy(view)
    changed["depth"][5, 7] = 4.01
    assert assess_model(mesh, np.eye(4), [changed])["evidenceSha256"] != report["evidenceSha256"]
    changed = deepcopy(view)
    changed["observationRevision"] += 1
    assert assess_model(mesh, np.eye(4), [changed])["evidenceSha256"] != report["evidenceSha256"]
    pose = np.eye(4)
    pose[0, 3] = .2
    assert assess_model(mesh, pose, [view])["poseSha256"] != report["poseSha256"]
    shifted = MeshData(mesh.vertices + .01, mesh.faces)
    assert assess_model(shifted, np.eye(4), [view])["meshSha256"] != report["meshSha256"]


def test_partial_mask_does_not_demand_complete_silhouette():
    from ehs_spatial.platform.model_quality import assess_model
    mesh, view = scene()
    view["mask"][:, 13:] = False
    view.pop("maskComplete")
    report = assess_model(mesh, np.eye(4), [view])
    score = report["perView"][0]
    assert score["coverage"] == 1
    assert score["precision"] < .5
    assert score["silhouetteRole"] == "diagnostic_partial_mask"
    assert report["status"] == "observed_consistent"
    view["maskComplete"] = True
    assert assess_model(mesh, np.eye(4), [view])["status"] == "observed_inconsistent"


def test_background_occlusion_never_excludes_wrong_target_depth():
    from ehs_spatial.platform.model_quality import assess_model
    mesh, view = scene()
    view["mask"][:, 16:] = False
    view["depth"][:, 16:] = 2
    report = assess_model(mesh, np.eye(4), [view])
    assert report["status"] == "observed_consistent"
    assert report["perView"][0]["occludedPredictedPixels"] == 13 * 11
    view["depth"][view["mask"]] = 2
    report = assess_model(mesh, np.eye(4), [view])
    assert report["status"] == "observed_inconsistent"
    assert report["perView"][0]["coverage"] == 1
    assert report["perView"][0]["relativeDepthP50"] == pytest.approx(1)


def test_unknown_depth_cannot_hide_predicted_background_and_domain_is_real():
    from ehs_spatial.platform.model_quality import assess_model
    mesh, view = scene()
    view["mask"][:, 16:] = False
    view["depth"][:, 16:] = np.nan
    view["valid"][:, 16:] = False
    report = assess_model(mesh, np.eye(4), [view])
    assert report["status"] == "observed_inconsistent"
    assert report["perView"][0]["occludedPredictedPixels"] == 0
    view["domain"] = np.ones(view["mask"].shape, bool)
    view["domain"][:, 16:] = False
    report = assess_model(mesh, np.eye(4), [view])
    assert report["status"] == "observed_consistent"
    assert report["perView"][0]["excludedDomainPixels"] == 24 * 20
    assert report["perView"][0]["visiblePredictedPixels"] == 13 * 10


def test_missing_depth_and_empty_views_are_insufficient():
    from ehs_spatial.platform.model_quality import assess_model, refine_model_pose
    mesh, view = scene()
    view["valid"][:] = False
    report = assess_model(mesh, np.eye(4), [view])
    assert report["status"] == "insufficient_evidence"
    assert report["perView"][0]["depthComparisonPixels"] == 0
    assert report["perView"][0]["relativeDepthP50"] is None
    assert assess_model(mesh, np.eye(4), [])["status"] == "insufficient_evidence"
    result = refine_model_pose(mesh, np.eye(4), [view], max_iterations=8)
    assert not result["accepted"]
    assert result["status"] == "insufficient_evidence"
    assert np.array_equal(result["objectToNative"], np.eye(4))


@pytest.mark.parametrize("invalid", ["K", "cameraToWorld", "mask", "sourceHashes", "missingSourceHash", "observationRevision"])
def test_invalid_views_fail_closed(invalid):
    from ehs_spatial.platform.model_quality import assess_model
    mesh, view = scene()
    if invalid == "K":
        view[invalid][1, 0] = 2
    elif invalid == "cameraToWorld":
        view[invalid][0, 0] = 2
    elif invalid == "mask":
        view[invalid] = np.ones((3, 4), bool)
    elif invalid == "sourceHashes":
        view[invalid] = {"image": "not-a-hash"}
    elif invalid == "missingSourceHash":
        view["sourceHashes"].pop("mask")
    else:
        view[invalid] = True
    report = assess_model(mesh, np.eye(4), [view])
    assert report["status"] == "insufficient_evidence"
    assert report["scoreableViewCount"] == 0
    assert report["perView"][0]["reason"]


def test_pose_validation_and_frame_mismatch():
    from ehs_spatial.platform.model_quality import assess_model
    mesh, view = scene()
    for pose in (np.zeros((4, 4)), np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan)):
        with pytest.raises(PlatformError):
            assess_model(mesh, pose, [view])
    another = deepcopy(view)
    another.update(observationId="observation-b", imageId="image-b", coordinateFrameId="other")
    assert assess_model(mesh, np.eye(4), [view, another])["status"] == "insufficient_evidence"


def test_each_view_matters_and_duplicate_views_do_not_manufacture_evidence():
    from ehs_spatial.platform.model_quality import assess_model
    mesh, view = scene()
    other = deepcopy(view)
    other.update(observationId="observation-b", imageId="image-b")
    other["depth"][other["mask"]] = 2
    report = assess_model(mesh, np.eye(4), [view, other])
    assert report["status"] == "observed_inconsistent"
    assert report["consistentViewCount"] == report["inconsistentViewCount"] == 1
    assert report["viewCount"] == report["independentImageCount"] == 2
    assert assess_model(mesh, np.eye(4), [view, view])["status"] == "insufficient_evidence"


def test_rigid_refinement_improves_depth_preserves_scale_and_input_mesh():
    from ehs_spatial.platform.model_quality import refine_model_pose
    world_mesh, view = scene()
    scale = np.diag([1.3, .7, 1.6, 1.])
    mesh = MeshData(transform_points(world_mesh.vertices, np.linalg.inv(scale)), world_mesh.faces)
    original = mesh.vertices.copy()
    initial = scale.copy()
    initial[2, 3] = .35
    result = refine_model_pose(mesh, initial, [view], max_iterations=100)
    assert result["accepted"]
    assert result["status"] == "accepted"
    matrix = np.asarray(result["objectToNative"])
    assert np.allclose(np.linalg.norm(matrix[:3, :3], axis=0), [1.3, .7, 1.6])
    assert np.array_equal(mesh.vertices, original)
    assert initial[2, 3] == .35
    before, after = result["before"]["perView"][0], result["after"]["perView"][0]
    assert after["coverage"] >= before["coverage"]
    assert after["relativeDepthP95"] < before["relativeDepthP95"] - .01
    assert result["after"]["semanticShapeStatus"] == "not_assessed"
    assert result["after"]["poseSha256"] != result["before"]["poseSha256"]


def test_refinement_cannot_trade_a_good_view_for_mean_improvement():
    from ehs_spatial.platform.model_quality import refine_model_pose
    mesh, view = scene()
    conflicting = deepcopy(view)
    conflicting.update(observationId="observation-b", imageId="image-b")
    conflicting["depth"][:] = 3.5
    result = refine_model_pose(mesh, np.eye(4), [view, conflicting], max_iterations=40)
    assert not result["accepted"]
    assert np.array_equal(result["objectToNative"], np.eye(4))
    assert result["before"] == result["after"]


def test_render_actual_mesh_colors_in_source_camera_with_white_background():
    from ehs_spatial.platform import model_quality
    assert hasattr(model_quality, "render_model_views")
    mesh, view = scene(rotated=True)
    colored = MeshData(mesh.vertices, mesh.faces, np.tile([.8, .2, .1], (4, 1)))
    view["rgb"] = np.zeros((24, 36, 3), np.uint8)  # Must not appear in the mesh image.
    image, = model_quality.render_model_views(colored, np.eye(4), [view])
    assert image.shape == (24, 36, 3) and image.dtype == np.uint8
    assert np.all(image[~view["mask"]] == 255)
    assert np.all(image[view["mask"], 0] > image[view["mask"], 1])
    assert np.all(image[view["mask"], 1] > image[view["mask"], 2])


def test_render_requires_valid_source_cameras():
    from ehs_spatial.platform import model_quality
    assert hasattr(model_quality, "render_model_views")
    mesh, view = scene()
    view["cameraToWorld"][0, 0] = 2
    with pytest.raises(PlatformError):
        model_quality.render_model_views(mesh, np.eye(4), [view])


def test_refinement_reassesses_native_resolution_after_sampled_search():
    from ehs_spatial.platform.model_quality import refine_model_pose
    mesh, view = scene()
    for key in ("mask", "valid", "depth"):
        view[key] = view[key].repeat(10, axis=0).repeat(10, axis=1)
    view["K"] = pixel_center_mapping((36, 24), (360, 240)) @ view["K"]
    initial = np.eye(4)
    initial[2, 3] = .35
    result = refine_model_pose(mesh, initial, [view], max_iterations=100)
    assert result["accepted"] and result["fullResolutionReassessed"]
    assert result["after"]["perView"][0]["targetPixels"] == 27300
    assert result["after"]["perView"][0]["depthComparisonPixels"] > 26000


def partitioned_family():
    mesh, view = scene(rotated=True)
    # Separate triangles are an exact assembly; neither residual alone covers it.
    offset = np.array([.2, -.3, .4])
    pose = np.eye(4)
    pose[:3, 3] = offset
    return mesh, view, [
        {'entityId':'parent', 'parentEntityId':None, 'coordinateFrameId':'native',
         'mesh':MeshData(mesh.vertices, mesh.faces[:1]), 'objectToNative':np.eye(4)},
        {'entityId':'child', 'parentEntityId':'parent', 'coordinateFrameId':'native',
         'mesh':MeshData(mesh.vertices-offset, mesh.faces[1:]), 'objectToNative':pose}]


def test_family_scores_declared_union_on_parent_evidence_without_promoting_members():
    from ehs_spatial.platform.model_quality import assess_model, assess_model_family
    _, view, members = partitioned_family()
    originals = [member['mesh'].vertices.copy() for member in members]
    separate = [assess_model(m['mesh'],m['objectToNative'],[view]) for m in members]
    assert all(r['status'] == 'observed_inconsistent' for r in separate)
    result = assess_model_family('parent', members, [view])
    assert result['assessmentScope'] == 'parent_family'
    assert result['parentEntityId'] == 'parent'
    assert result['status'] == 'observed_consistent'
    assert result['perView'][0]['coverage'] == 1
    assert result['semanticShapeStatus'] == result['physicalCalibrationStatus'] == 'not_assessed'
    assert {m['entityId'] for m in result['familyMembers']} == {'parent','child'}
    for member, original in zip(members, originals, strict=True):
        assert np.array_equal(member['mesh'].vertices, original)
    # Family evidence does not replace the member's separately owned mask.
    assert assess_model(members[0]['mesh'],members[0]['objectToNative'],[view]) == separate[0]
    changed = deepcopy(members)
    changed[1]['entityId'] = 'another-child'
    rebound = assess_model_family('parent',changed,[view])
    assert rebound['perView'] == result['perView']
    assert rebound['evidenceSha256'] != result['evidenceSha256']
    changed[1]['objectToNative'][0,3] += .5
    moved = assess_model_family('parent',changed,[view])
    assert moved['evidenceSha256'] != rebound['evidenceSha256']
    assert moved['perView'][0]['coverage'] < 1


@pytest.mark.parametrize('corruption', ['unrelated','missing_parent','cycle','duplicate','frame','view_frame'])
def test_family_rejects_unrelated_mesh_or_unregistered_frame(corruption):
    from ehs_spatial.platform.model_quality import assess_model_family
    _, view, members = partitioned_family()
    if corruption == 'unrelated': members[1]['parentEntityId'] = 'someone-else'
    elif corruption == 'missing_parent': members.pop(0)
    elif corruption == 'cycle': members[0]['parentEntityId'] = 'child'
    elif corruption == 'duplicate': members[1]['entityId'] = 'parent'
    elif corruption == 'frame': members[1]['coordinateFrameId'] = 'unregistered'
    else: view['coordinateFrameId'] = 'unregistered'
    with pytest.raises(PlatformError):
        assess_model_family('parent',members,[view])


def test_family_keeps_all_parent_views_and_does_not_hide_bad_depth():
    from ehs_spatial.platform.model_quality import assess_model_family
    _, view, members = partitioned_family()
    bad = deepcopy(view)
    bad['observationId'] = 'another-observation'
    bad['imageId'] = 'another-image'
    bad['depth'][bad['mask']] = 2.
    result = assess_model_family('parent',members,[view,bad])
    assert result['status'] == 'observed_inconsistent'
    assert result['perView'][0]['status'] == 'observed_consistent'
    assert result['perView'][1]['relativeDepthP50'] > .9
