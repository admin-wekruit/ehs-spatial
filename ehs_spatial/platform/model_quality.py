"""Source-bound observed mesh alignment, never complete-shape or metric approval.

Views use integer pixel centres and camera z depth, in a shared native frame.
``sourceHashes`` binds original assets; array hashes bind the actual scoring inputs.
Invalid evidence fails closed. No threshold here is a measured accuracy claim.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

import numpy as np

from .contracts import PlatformError, digest
from .spatial import MeshData, affine, camera_intrinsics, matrix_to_transform, transform_points


THRESHOLDS_VERSION = "observed-model-quality-v1"
# Engineering checks in model units, not learned confidence or physical tolerance.
THRESHOLDS = {"minimumTargetPixels": 8, "minimumDepthPixels": 8,
              "minimumCoverage": .8, "maximumRelativeDepthP50": .05,
              "maximumRelativeDepthP95": .15, "minimumCompleteMaskIoU": .65,
              "minimumCompleteMaskPrecision": .7, "occlusionRelativeDepth": .04,
              "meaningfulCoverageImprovement": .005, "meaningfulDepthImprovement": .002,
              "nonRegressionTolerance": 1e-6}


def _array_hash(value):
    array = np.ascontiguousarray(value)
    result = hashlib.sha256()
    result.update(digest({"shape": list(array.shape), "dtype": array.dtype.str}).encode())
    result.update(array.tobytes())
    return result.hexdigest()


def _mesh_parts(mesh):
    if not isinstance(mesh, MeshData):
        raise PlatformError("invalid_mesh")
    # Validate again: frozen dataclasses still contain mutable NumPy arrays.
    mesh.__post_init__()
    for part in mesh.primitives or (mesh,):
        part.__post_init__()
        yield part


def _pose(value):
    matrix = affine(value)
    matrix_to_transform(matrix, "quality-native")  # Reject shear, reflection and zero scale.
    return matrix.copy()


def _prepare_view(view):
    binding = {}
    try:
        if not isinstance(view, Mapping):
            raise PlatformError("invalid_quality_view")
        for key in ("observationId", "imageId", "coordinateFrameId"):
            if not isinstance(view.get(key), str) or not view[key]:
                raise PlatformError("missing_quality_source_identity")
            binding[key] = view[key]
        revision = view.get("observationRevision")
        if type(revision) is not int or revision < 1:
            raise PlatformError("invalid_quality_observation_revision")
        binding["observationRevision"] = revision
        hashes = view.get("sourceHashes")
        if (not isinstance(hashes, Mapping) or not {"image", "mask", "geometry", "camera"} <= hashes.keys() or
                any(not isinstance(k, str) or not k or not isinstance(v, str) or
                    re.fullmatch(r"[a-f0-9]{64}", v) is None for k, v in hashes.items())):
            raise PlatformError("invalid_quality_source_hashes")
        binding["sourceHashes"] = dict(hashes)
        mask, depth, valid = (np.asarray(view[key]) for key in ("mask", "depth", "valid"))
        domain = np.asarray(view.get("domain", np.ones(mask.shape, bool)))
        if (mask.ndim != 2 or min(mask.shape) < 1 or mask.dtype != bool or
                depth.shape != mask.shape or depth.dtype.kind not in "fiu" or
                valid.shape != mask.shape or valid.dtype != bool or
                domain.shape != mask.shape or domain.dtype != bool):
            raise PlatformError("invalid_quality_view_grid")
        complete = view.get("maskComplete", False)
        if type(complete) is not bool:
            raise PlatformError("invalid_quality_mask_completeness")
        k = camera_intrinsics(view["K"])
        c2w = affine(view["cameraToWorld"], rigid=True)
        binding.update(maskComplete=complete, width=mask.shape[1], height=mask.shape[0],
                       inputHashes={key: _array_hash(value) for key, value in
                                    (("mask", mask), ("depth", depth), ("valid", valid),
                                     ("domain", domain), ("K", k), ("cameraToWorld", c2w))})
        binding["viewEvidenceSha256"] = digest(binding)
        y, x = np.mgrid[:mask.shape[0], :mask.shape[1]]
        local = np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(k).T
        directions = local @ c2w[:3, :3].T
        rays = np.concatenate((np.broadcast_to(c2w[:3, 3], directions.shape), directions), -1)
        return {"binding": binding, "mask": mask, "depth": depth, "valid": valid,
                "domain": domain, "rays": rays.astype(np.float32)}
    except (PlatformError, ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        return {"binding": binding, "error": error.code if isinstance(error, PlatformError) else "invalid_quality_view"}


def _prepare(mesh, object_to_native, views):
    parts = tuple(_mesh_parts(mesh))
    matrix = _pose(object_to_native)
    if not isinstance(views, (list, tuple)):
        raise PlatformError("invalid_quality_views")
    prepared = [_prepare_view(view) for view in views]
    ids = [v["binding"].get("observationId") for v in prepared]
    frames = {v["binding"].get("coordinateFrameId") for v in prepared} - {None}
    reason = ("quality_coordinate_frame_mismatch" if len(frames) > 1 else
              "duplicate_quality_observation" if len([i for i in ids if i]) != len({i for i in ids if i}) else None)
    if reason:
        for view in prepared:
            view["error"] = reason
    mesh_hash = digest([{ "vertices": _array_hash(np.asarray(p.vertices)),
                         "faces": _array_hash(np.asarray(p.faces))} for p in parts])
    return parts, matrix, prepared, mesh_hash


def _ray_scene(parts):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    for part in parts:
        scene.add_triangles(o3d.core.Tensor(np.asarray(part.vertices, np.float32)),
                            o3d.core.Tensor(np.asarray(part.faces, np.uint32)))
    return scene


def _cast(scene, matrix, rays):
    import open3d as o3d
    inverse = np.linalg.inv(matrix)
    local = np.concatenate((transform_points(rays[..., :3], inverse),
                            rays[..., 3:] @ inverse[:3, :3].T), -1).astype(np.float32)
    # Never normalise: t_hit must remain camera z, including nonuniform object scale.
    return scene.cast_rays(o3d.core.Tensor(local), nthreads=1)


def _score(scene, matrix, view):
    binding = view["binding"]
    if "error" in view:
        return {**binding, "status": "insufficient_evidence", "scoreable": False, "reason": view["error"]}
    predicted = _cast(scene, matrix, view["rays"])["t_hit"].numpy()
    domain, depth = view["domain"], view["depth"]
    target = view["mask"] & domain
    usable = domain & view["valid"] & np.isfinite(depth) & (depth > 0)
    hit = domain & np.isfinite(predicted) & (predicted > 0)
    occluded = hit & ~target & usable & (predicted > depth * (1 + THRESHOLDS["occlusionRelativeDepth"]))
    visible = hit & ~occluded
    overlap, compared = visible & target, visible & target & usable
    residual = np.abs(predicted[compared] - depth[compared]) / depth[compared]
    count, overlap_count, visible_count = int(target.sum()), int(overlap.sum()), int(visible.sum())
    depth_count = int((target & usable).sum())
    p50, p95 = np.quantile(residual, [.5, .95]).tolist() if len(residual) else (None, None)
    coverage = overlap_count / count if count else None
    precision = overlap_count / visible_count if visible_count else 0.
    union = int((visible | target).sum())
    iou = overlap_count / union if union else None
    scoreable = count >= THRESHOLDS["minimumTargetPixels"] and depth_count >= THRESHOLDS["minimumDepthPixels"]
    reasons = []
    if not scoreable:
        status = "insufficient_evidence"
        reasons.append("insufficient_target_pixels" if count < THRESHOLDS["minimumTargetPixels"] else "insufficient_target_depth")
    else:
        if coverage < THRESHOLDS["minimumCoverage"]:
            reasons.append("observed_coverage_failed")
        if binding["maskComplete"] and (iou < THRESHOLDS["minimumCompleteMaskIoU"] or precision < THRESHOLDS["minimumCompleteMaskPrecision"]):
            reasons.append("complete_mask_silhouette_failed")
        if p50 is not None and (p50 > THRESHOLDS["maximumRelativeDepthP50"] or p95 > THRESHOLDS["maximumRelativeDepthP95"]):
            reasons.append("observed_depth_failed")
        if reasons:
            status = "observed_inconsistent"
        elif len(residual) < THRESHOLDS["minimumDepthPixels"]:
            status = "insufficient_evidence"
            reasons.append("insufficient_depth_comparison")
        else:
            status = "observed_consistent"
    return {**binding, "status": status, "scoreable": scoreable, "reason": ";".join(reasons) or None,
            "coverage": coverage, "precision": precision, "iou": iou,
            "silhouetteRole": "complete_mask_check" if binding["maskComplete"] else "diagnostic_partial_mask",
            "relativeDepthP50": p50, "relativeDepthP95": p95,
            "targetPixels": count, "targetDepthPixels": depth_count,
            "unknownTargetDepthPixels": count - depth_count, "depthComparisonPixels": int(len(residual)),
            "overlapPixels": overlap_count, "predictedPixels": int(hit.sum()),
            "visiblePredictedPixels": visible_count, "occludedPredictedPixels": int(occluded.sum()),
            "excludedDomainPixels": int((~domain).sum()), "sourceMaskPixels": int(view["mask"].sum())}


def _report(scene, matrix, views, mesh_hash):
    scores = [_score(scene, matrix, view) for view in views]
    statuses = [score["status"] for score in scores]
    status = ("insufficient_evidence" if not scores or "insufficient_evidence" in statuses else
              "observed_inconsistent" if "observed_inconsistent" in statuses else "observed_consistent")
    pose_hash = digest(matrix.tolist())
    return {"version": THRESHOLDS_VERSION, "thresholds": dict(THRESHOLDS),
            "thresholdMeaning": "engineering_checks_not_accuracy_claims", "status": status,
            "semanticShapeStatus": "not_assessed", "physicalCalibrationStatus": "not_assessed",
            "meshSha256": mesh_hash, "poseSha256": pose_hash,
            "evidenceSha256": digest({"mesh": mesh_hash, "pose": pose_hash,
                                      "views": [v["binding"] for v in views],
                                      "version": THRESHOLDS_VERSION, "thresholds": THRESHOLDS}),
            "viewCount": len(scores), "independentImageCount": len({s["imageId"] for s in scores if s.get("imageId")}),
            "scoreableViewCount": sum(s["scoreable"] for s in scores),
            "consistentViewCount": statuses.count("observed_consistent"),
            "inconsistentViewCount": statuses.count("observed_inconsistent"),
            "insufficientViewCount": statuses.count("insufficient_evidence"), "perView": scores}


def assess_model(mesh, object_to_native, views):
    """Assess every supplied view at native resolution, without mutating geometry.

    A view requires observationId/revision, imageId, coordinateFrameId, sourceHashes,
    bool mask/valid, depth, K and cameraToWorld. sourceHashes requires image, mask,
    geometry and camera SHA256 values. Optional bool domain excludes padding;
    maskComplete defaults to False. One failed or unscoreable view prevents acceptance.
    """
    parts, matrix, prepared, mesh_hash = _prepare(mesh, object_to_native, views)
    return _report(_ray_scene(parts), matrix, prepared, mesh_hash)


def render_model_views(mesh, object_to_native, views):
    """Native-camera RGB geometry previews, white background, no source-photo pixels.

    Vertex colours interpolate over actual triangles; uncoloured meshes use grey.
    Textures and material lighting are not assessed by these geometry previews.
    """
    parts, matrix, prepared, _ = _prepare(mesh, object_to_native, views)
    if any("error" in view for view in prepared):
        raise PlatformError("invalid_quality_render_evidence")
    scene, images = _ray_scene(parts), []
    for view in prepared:
        cast = _cast(scene, matrix, view["rays"])
        depth = cast["t_hit"].numpy()
        hit = np.isfinite(depth) & (depth > 0) & view["domain"]
        image = np.full((*depth.shape, 3), 255, np.uint8)
        geometry_ids = cast["geometry_ids"].numpy()
        face_ids, uv = cast["primitive_ids"].numpy(), cast["primitive_uvs"].numpy()
        normal = cast["primitive_normals"].numpy()[hit] @ np.linalg.inv(matrix[:3, :3])
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
        ray = view["rays"][..., 3:][hit].astype(float)
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        shading = np.ones(depth.shape)
        shading[hit] = .4 + .6 * np.abs(np.sum(normal * ray, axis=-1))
        for index, part in enumerate(parts):
            selected = hit & (geometry_ids == index)
            if not selected.any():
                continue
            colors = np.full((int(selected.sum()), 3), .65)
            if part.colors is not None:
                barycentric = np.column_stack((1 - uv[selected].sum(axis=-1), uv[selected]))
                triangles = np.asarray(part.faces)[face_ids[selected]]
                colors = np.sum(np.asarray(part.colors)[triangles, :3] * barycentric[..., None], axis=1)
            image[selected] = np.round(np.clip(colors * shading[selected, None], 0, 1) * 255).astype(np.uint8)
        images.append(image)
    return images


def _nonworsening(before, after):
    epsilon = THRESHOLDS["nonRegressionTolerance"]
    improved = False
    for a, b in zip(before["perView"], after["perView"], strict=True):
        if not a["scoreable"] or not b["scoreable"] or b["coverage"] + epsilon < a["coverage"]:
            return False
        if b["depthComparisonPixels"] < a["depthComparisonPixels"]:
            return False
        improved |= b["coverage"] - a["coverage"] >= THRESHOLDS["meaningfulCoverageImprovement"]
        for key in ("relativeDepthP50", "relativeDepthP95"):
            if a[key] is not None:
                if b[key] is None or b[key] > a[key] + epsilon:
                    return False
                improved |= a[key] - b[key] >= THRESHOLDS["meaningfulDepthImprovement"]
        if a["maskComplete"]:
            if (b["iou"] + epsilon < a["iou"] or
                    b["precision"] + epsilon < min(a["precision"], THRESHOLDS["minimumCompleteMaskPrecision"])):
                return False
    return bool(improved)


def refine_model_pose(mesh, object_to_native, views, max_iterations=80):
    """Bounded rigid candidate only; no scale optimization or automatic placement approval."""
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    if type(max_iterations) is not int or not 1 <= max_iterations <= 500:
        raise PlatformError("invalid_quality_refinement_iterations")
    parts, initial, prepared, mesh_hash = _prepare(mesh, object_to_native, views)
    scene = _ray_scene(parts)
    before = _report(scene, initial, prepared, mesh_hash)
    result = {"method": "bounded_rigid_observed_alignment_v1", "accepted": False,
              "status": "insufficient_evidence", "objectToNative": initial.tolist(),
              "before": before, "after": before, "evaluations": 0, "fullResolutionReassessed": False}
    if not prepared or before["scoreableViewCount"] != len(prepared):
        return result
    points = np.concatenate([transform_points(part.vertices, initial) for part in parts])
    lo, hi = points.min(axis=0), points.max(axis=0)
    center, radius = (lo + hi) / 2, float(np.linalg.norm(hi - lo))
    if radius <= 1e-10:
        result["reason"] = "degenerate_model_extent"
        return result
    # ponytail: bounded local pose search, not global registration. Sample exact
    # source pixel centres up to ~160 pixels/axis; native grids gate the result.
    sampled = []
    for view in prepared:
        stride = max(1, int(np.ceil(max(view["mask"].shape) / 160)))
        while stride > 1:
            target = view["mask"][::stride, ::stride] & view["domain"][::stride, ::stride]
            supported = target & view["valid"][::stride, ::stride] & np.isfinite(view["depth"][::stride, ::stride]) & (view["depth"][::stride, ::stride] > 0)
            if supported.sum() >= THRESHOLDS["minimumDepthPixels"]:
                break
            stride = max(1, stride // 2)
        sampled.append({**view, **{key: view[key][::stride, ::stride] for key in ("mask", "depth", "valid", "domain", "rays")}})

    def candidate(x):
        rotation = Rotation.from_rotvec(x[3:]).as_matrix()
        matrix = initial.copy()
        matrix[:3, :3] = rotation @ initial[:3, :3]
        matrix[:3, 3] = center + rotation @ (initial[:3, 3] - center) + x[:3] * radius
        return matrix

    def loss(scores):
        losses = []
        for score in scores:
            value = 1 - score["coverage"]
            value += min(score["relativeDepthP50"] if score["relativeDepthP50"] is not None else 1., 1.)
            value += min(score["relativeDepthP95"] if score["relativeDepthP95"] is not None else 1., 1.)
            if score["maskComplete"]:
                value += 1 - score["iou"]
            losses.append(value)
        return max(losses) + .1 * sum(losses)

    def objective(x):
        if np.max(np.abs(x[:3])) > .3 or np.linalg.norm(x[3:]) > .65:
            return 100. + float(x @ x)
        result["evaluations"] += 1
        return loss([_score(scene, candidate(x), view) for view in sampled])

    simplex = np.vstack((np.zeros(6), np.diag([.025] * 3 + [.04] * 3)))
    fit = minimize(objective, np.zeros(6), method="Nelder-Mead", options={
        "initial_simplex": simplex, "maxiter": max_iterations, "xatol": .0005, "fatol": .0001})
    proposed = _pose(candidate(fit.x))
    after = _report(scene, proposed, prepared, mesh_hash)
    accepted = _nonworsening(before, after)
    result.update(accepted=accepted, status="accepted" if accepted else "rejected",
                  fullResolutionReassessed=True, reason=None if accepted else "per_view_improvement_not_proven")
    if accepted:
        result.update(objectToNative=proposed.tolist(), after=after)
    return result
