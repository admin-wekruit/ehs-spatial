"""CPU-only capture quality integration; fake stages never invoke paid providers."""
from copy import deepcopy
import base64
import hashlib
import io
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError

from ehs_spatial.platform.contracts import PlatformError, canonical, empty_document
from ehs_spatial.platform.reconstruction import (
    _ModelReviewResponse, _assess_generation, _packed, _record_generated_representation,
)
from ehs_spatial.platform.spatial import FrameGeometry, transform_points, unproject_pixels


class Stages:
    def __init__(self, review):
        self.repo = self
        self.providers = {} if review is None else {"model_review": object()}
        self.review, self.calls, self.assets, self.values = review, [], {}, {}

    def put(self, value, metadata, media_type="application/json"):
        data = value if isinstance(value, bytes) else canonical(_packed(value))
        asset = {"id": str(uuid4()), "sha256": hashlib.sha256(data).hexdigest(),
                 "metadata": metadata, "mediaType": media_type, "sizeBytes": len(data)}
        self.assets[asset["id"]] = asset
        self.values[asset["id"]] = deepcopy(value)
        return asset

    def get_asset(self, asset_id):
        if asset_id not in self.assets:
            raise PlatformError("asset_not_found", 404)
        return deepcopy(self.assets[asset_id])

    def call(self, stage, images, payload, refs):
        assert stage == "model_review" and self.review is not None
        self.calls.append({"stage": stage, "images": deepcopy(images), "payload": deepcopy(payload), "refs": refs})
        value = {"review": deepcopy(self.review)}
        return value, self.put(value, {"kind": "stage_cache", "stage": stage})


def quality_case(*, review_status="pass"):
    review = {"status": review_status, "reason": "Visible structure reviewed in both source photographs.",
              "observationIds": ["observation-a", "observation-b"],
              "visibleShapeIssues": [] if review_status == "pass" else ["Opaque slab replaces visible openings."],
              "nextAction": "none" if review_status == "pass" else "alternate_view"}
    stages = Stages(review if review_status is not None else None)
    document = empty_document()
    document["captureId"] = str(uuid4())
    document["coordinateFrames"] = [{"id": "native", "ground": None}]
    document["geometryBindings"] = {}
    k = np.array([[28., 4., 17.5], [0., 24., 11.5], [0., 0., 1.]])
    frames, records, masks, observations = {}, {}, {}, {}
    for i, letter in enumerate(("a", "b")):
        image = stages.put({"photo": letter}, {"kind": "image"})
        geometry = stages.put({"geometry": letter}, {"kind": "geometry"})
        c2w = np.eye(4)
        c2w[0, 3] = i * 4 / 28  # A genuinely different camera; silhouette moves one pixel.
        y, x = np.mgrid[:24, :36]
        points = unproject_pixels(np.column_stack((x.ravel(), y.ravel())), np.full(24 * 36, 4.),
                                  {"K": k, "cameraToWorld": c2w}).reshape(24, 36, 3)
        frame = FrameGeometry(image["id"], "native", image["sha256"], points,
                              np.ones((24, 36), bool), k.copy(), c2w)
        mask = np.zeros((24, 36), bool)
        mask[4:17, 6 - i:27 - i] = True
        mask_asset = stages.put(mask, {"kind": "mask"})
        oid = "observation-" + letter
        observations[oid] = {"id": oid, "revision": i + 1, "imageId": image["id"],
                             "maskAssetId": mask_asset["id"], "maskComplete": True}
        frames[image["id"]], masks[oid] = frame, mask
        records[image["id"]] = {"rgb": np.full((24, 36, 3), 60 + i * 20, np.uint8),
                                 "points": points, "inputToCanonical": np.eye(3), "originalShape": (24, 36)}
        document["geometryBindings"][image["id"]] = {"geometrySolutionId": geometry["id"]}
    entity = {"id": "entity", "label": "fixture panel", "observationRefs": list(observations),
              "representations": [], "activeModelRepresentationId": None, "currentModelTransform": None}
    document["entities"], document["observations"] = [entity], list(observations.values())
    vertices = unproject_pixels(np.array([[5.5, 3.5], [26.5, 3.5], [26.5, 16.5], [5.5, 16.5]]),
                                np.full(4, 4.), {"K": k, "cameraToWorld": np.eye(4)})
    response = {"vertices": vertices, "faces": np.array([[0, 1, 2], [0, 2, 3]]),
                "proposedObjectToNative": np.eye(4)}
    evidence = stages.put(response, {"kind": "generation"})
    document["assets"] = list(stages.assets.values())
    return SimpleNamespace(document=document, entity=entity, observations=observations, response=response,
                           evidence=evidence, frames=frames, records=records, masks=masks,
                           stages=stages, coordinate_frame_id="native")


def assess(case):
    return _assess_generation(**vars(case))


def record(case, response, quality, accepted):
    return _record_generated_representation(case.document, case.stages, case.entity,
        list(case.observations.values()), case.coordinate_frame_id, response, case.evidence,
        activate=accepted, quality=quality)


def decode_image(data_uri):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))))


def test_all_view_geometry_pass_cannot_override_semantic_rejection():
    case = quality_case(review_status="fail")
    response, quality, accepted = assess(case)
    assert not accepted and quality["status"] == "rejected"
    assert quality["geometric"]["status"] == "observed_consistent"
    assert quality["geometric"]["consistentViewCount"] == quality["geometric"]["independentImageCount"] == 2
    assert quality["shapeReview"]["status"] == "fail"
    assert quality["geometric"]["semanticShapeStatus"] == "not_assessed"
    assert quality["physicalPlacementConfirmed"] is False
    assert quality["geometric"]["physicalCalibrationStatus"] == "not_assessed"
    call, = case.stages.calls
    assert call["payload"]["candidateAssetSha256"] == case.evidence["sha256"]
    assert [pair["observationRevision"] for pair in call["payload"]["views"]] == [1, 2]
    for pair in call["payload"]["views"]:
        source, candidate, mask = [decode_image(pair[key]) for key in ("source", "candidate", "mask")]
        assert source.shape == candidate.shape == (24, 36, 3) and mask.shape == (24, 36)
        assert not np.array_equal(source, candidate)
    representation = record(case, response, quality, accepted)
    assert representation["shapeStatus"] == "candidate"
    assert representation["placementState"] == "unconfirmed"
    assert case.entity["activeModelRepresentationId"] is None and case.entity["currentModelTransform"] is None


def test_candidate_stays_inactive_without_a_shape_review_provider():
    case = quality_case(review_status=None)
    response, quality, accepted = assess(case)
    assert quality["geometric"]["status"] == "observed_consistent"
    assert not accepted and quality["status"] == "needs_information"
    assert quality["shapeReview"]["reason"] == "shape_review_not_configured"
    assert not case.stages.calls
    record(case, response, quality, accepted)
    assert case.entity["activeModelRepresentationId"] is None


def test_review_decode_failure_preserves_candidate_and_needs_information():
    case = quality_case()
    def decode_failure(*_args):
        raise ValueError("malformed model response")
    case.stages.call = decode_failure
    response, quality, accepted = assess(case)
    assert not accepted and quality["status"] == "needs_information"
    assert quality["shapeReview"]["reason"] == "model_review_invalid"
    representation = record(case, response, quality, accepted)
    assert representation["shapeStatus"] == "candidate"
    assert representation in case.entity["representations"]
    assert case.entity["activeModelRepresentationId"] is None


def test_semantic_pass_cannot_replace_missing_depth():
    case = quality_case()
    next(iter(case.frames.values())).valid[:] = False
    _, quality, accepted = assess(case)
    assert not accepted and quality["status"] == "needs_information"
    assert quality["shapeReview"]["status"] == "pass"
    assert quality["geometric"]["status"] == "insufficient_evidence"
    assert quality["physicalPlacementConfirmed"] is False


@pytest.mark.parametrize("ids", [["observation-a", "foreign"], ["observation-a"],
                                  ["observation-a", "observation-a", "observation-b"]])
def test_review_must_name_exactly_the_current_owned_views(ids):
    case = quality_case()
    case.stages.review["observationIds"] = ids
    _, quality, accepted = assess(case)
    assert not accepted and quality["status"] == "needs_information"
    assert quality["shapeReview"]["reason"] in {"model_review_evidence_mismatch", "model_review_invalid"}


@pytest.mark.parametrize("missing", ["observation", "view", "geometry_binding", "mask_asset_id", "source_asset"])
def test_missing_owned_evidence_blocks_review_and_is_retained_in_audit(missing):
    case = quality_case()
    observation = case.observations["observation-b"]
    image_id = observation["imageId"]
    if missing == "observation":
        case.observations.pop(observation["id"])
    elif missing == "view":
        case.frames.pop(image_id)
    elif missing == "geometry_binding":
        case.document["geometryBindings"].pop(image_id)
    elif missing == "mask_asset_id":
        observation.pop("maskAssetId")
    else:
        case.stages.assets.pop(observation["maskAssetId"])
    _, quality, accepted = assess(case)
    assert not accepted and quality["status"] == "needs_information"
    assert any(item.get("observationId") == observation["id"] for item in quality["missingEvidence"])
    assert quality["geometric"]["consistentViewCount"] == 1
    assert not case.stages.calls


def test_quality_domain_excludes_canonical_padding_without_hiding_unknown_depth():
    case = quality_case()
    image_id = case.observations["observation-b"]["imageId"]
    case.records[image_id]["originalShape"] = (24, 16)
    _, quality, accepted = assess(case)
    assert accepted
    scores = {item["observationId"]: item for item in quality["geometric"]["perView"]}
    assert scores["observation-b"]["excludedDomainPixels"] == 24 * 20
    assert scores["observation-b"]["targetPixels"] == 13 * 11
    assert scores["observation-a"]["excludedDomainPixels"] == 0


@pytest.mark.parametrize("issues,action", [(["Missing visible openings"], "none"), ([], "correct_mask")])
def test_review_pass_requires_no_visible_issues_and_no_pending_action(issues, action):
    case = quality_case()
    case.stages.review.update(visibleShapeIssues=issues, nextAction=action)
    with pytest.raises(ValidationError):
        _ModelReviewResponse.model_validate(case.stages.review)
    _, quality, accepted = assess(case)
    assert not accepted and quality["shapeReview"]["status"] == "needs_information"


def test_pose_correction_plus_semantic_pass_never_confirms_physical_placement():
    case = quality_case()
    scale = np.diag([1.3, .7, 1.6, 1.])
    case.response["vertices"] = transform_points(case.response["vertices"], np.linalg.inv(scale))
    case.response["proposedObjectToNative"] = scale.copy()
    case.response["proposedObjectToNative"][2, 3] = .35
    original_vertices = case.response["vertices"].copy()
    response, quality, accepted = assess(case)
    assert accepted and quality["status"] == "accepted"
    assert quality["correction"]["accepted"] and quality["correction"]["fullResolutionReassessed"]
    assert quality["geometric"]["status"] == "observed_consistent"
    assert np.allclose(np.linalg.norm(response["proposedObjectToNative"][:3, :3], axis=0), [1.3, .7, 1.6])
    assert np.array_equal(case.response["vertices"], original_vertices)
    assert case.response["proposedObjectToNative"][2, 3] == .35
    assert quality["physicalPlacementConfirmed"] is False
    representation = record(case, response, quality, accepted)
    assert representation["shapeStatus"] == "observed_accepted"
    assert case.entity["activeModelRepresentationId"] == representation["id"]
    assert representation["placementState"] == "unconfirmed"
    assert representation["placementReason"] == "requires_alignment_confirmation"


def test_accepted_regeneration_preserves_preexisting_active_model_and_pose():
    case = quality_case()
    case.entity["activeModelRepresentationId"] = "existing-model"
    case.entity["currentModelTransform"] = {"source": "existing-confirmed-pose"}
    response, quality, accepted = assess(case)
    assert accepted
    record(case, response, quality, accepted)
    assert case.entity["activeModelRepresentationId"] == "existing-model"
    assert case.entity["currentModelTransform"] == {"source": "existing-confirmed-pose"}


def test_reassessment_of_cached_generation_activates_current_review_evidence():
    case = quality_case(review_status=None)
    response, first_quality, accepted = assess(case)
    original = record(case, response, first_quality, accepted)
    case.stages.review = {"status": "pass", "reason": "Both current views agree with visible structure.",
                          "observationIds": list(case.observations), "visibleShapeIssues": [], "nextAction": "none"}
    case.stages.providers["model_review"] = object()
    response, current_quality, accepted = assess(case)
    assert accepted
    current = record(case, response, current_quality, accepted)
    active = next(r for r in case.entity["representations"] if r["id"] == case.entity["activeModelRepresentationId"])
    assert active["shapeStatus"] == "observed_accepted"
    assert active["qualityEvidence"]["evidenceRef"] == current_quality["evidenceRef"]
    assert active["transform"] == case.entity["currentModelTransform"]
    assert current["id"] != original["id"]
