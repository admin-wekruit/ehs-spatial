"""Source-bound observed mesh alignment, never complete-shape or metric approval.

Views use integer pixel centres and camera z depth, in a shared native frame.
``sourceHashes`` binds original assets; array hashes bind the actual scoring inputs.
Invalid evidence fails closed. No threshold here is a measured accuracy claim.
"""
from __future__ import annotations

import base64
import hashlib
import io
import re
from collections.abc import Mapping

import numpy as np
from PIL import Image

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
COARSE_THRESHOLDS = {"silhouetteToleranceFraction": .05, "minimumTolerantCoverage": .9,
                     "minimumDepthInlierFraction": .8}


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


def _score(scene, matrix, view, *, coarse=False):
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
    coarse_metrics = {}
    if coarse and count:
        from scipy.ndimage import distance_transform_edt
        y, x = np.nonzero(target)
        tolerance = max(1., COARSE_THRESHOLDS['silhouetteToleranceFraction'] * np.hypot(np.ptp(x)+1, np.ptp(y)+1))
        tolerant = int((distance_transform_edt(~visible)[target] <= tolerance).sum()) if visible.any() else 0
        coarse_metrics = {'exactCoverage':coverage, 'silhouetteTolerancePixels':float(tolerance),
                          'tolerantCoverage':tolerant/count,
                          'depthInlierFraction':float(np.mean(residual <= THRESHOLDS['maximumRelativeDepthP95'])) if len(residual) else None}
    scoreable = count >= THRESHOLDS["minimumTargetPixels"] and depth_count >= THRESHOLDS["minimumDepthPixels"]
    reasons = []
    if not scoreable:
        status = "insufficient_evidence"
        reasons.append("insufficient_target_pixels" if count < THRESHOLDS["minimumTargetPixels"] else "insufficient_target_depth")
    else:
        if coarse and coarse_metrics['tolerantCoverage'] < COARSE_THRESHOLDS['minimumTolerantCoverage']:
            reasons.append('coarse_extent_or_position_failed')
        elif not coarse and coverage < THRESHOLDS["minimumCoverage"]:
            reasons.append("observed_coverage_failed")
        if binding["maskComplete"] and (iou < THRESHOLDS["minimumCompleteMaskIoU"] or precision < THRESHOLDS["minimumCompleteMaskPrecision"]):
            reasons.append("complete_mask_silhouette_failed")
        # Coarse layout checks retain the median position requirement. Local
        # omitted bars/holes may disagree; preserve P95 as a detailed diagnostic.
        depth_tail_failed = (coarse_metrics.get('depthInlierFraction',0) < COARSE_THRESHOLDS['minimumDepthInlierFraction']
                             if coarse and p50 is not None else p95 is not None and p95 > THRESHOLDS['maximumRelativeDepthP95'])
        if p50 is not None and (p50 > THRESHOLDS["maximumRelativeDepthP50"] or depth_tail_failed):
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
            "excludedDomainPixels": int((~domain).sum()), "sourceMaskPixels": int(view["mask"].sum()), **coarse_metrics}


def _report(scene, matrix, views, mesh_hash, *, coarse=False):
    scores = [_score(scene, matrix, view, coarse=coarse) for view in views]
    statuses = [score["status"] for score in scores]
    status = ("insufficient_evidence" if not scores or "insufficient_evidence" in statuses else
              "observed_inconsistent" if "observed_inconsistent" in statuses else "observed_consistent")
    pose_hash = digest(matrix.tolist())
    version = 'coarse-layout-position-v1' if coarse else THRESHOLDS_VERSION
    thresholds = {**THRESHOLDS, **(COARSE_THRESHOLDS if coarse else {})}
    return {"version": version, "thresholds": thresholds,
            "acceptanceScope":"coarse_layout" if coarse else "observed_surface_alignment",
            "thresholdMeaning": "engineering_checks_not_accuracy_claims", "status": status,
            "semanticShapeStatus": "not_assessed", "physicalCalibrationStatus": "not_assessed",
            "meshSha256": mesh_hash, "poseSha256": pose_hash,
            "evidenceSha256": digest({"mesh": mesh_hash, "pose": pose_hash,
                                      "views": [v["binding"] for v in views],
                                      "version": version, "thresholds": thresholds}),
            "viewCount": len(scores), "independentImageCount": len({s["imageId"] for s in scores if s.get("imageId")}),
            "scoreableViewCount": sum(s["scoreable"] for s in scores),
            "consistentViewCount": statuses.count("observed_consistent"),
            "inconsistentViewCount": statuses.count("observed_inconsistent"),
            "insufficientViewCount": statuses.count("insufficient_evidence"), "perView": scores}


def assess_model(mesh, object_to_native, views, *, coarse=False):
    """Assess every supplied view at native resolution, without mutating geometry.

    A view requires observationId/revision, imageId, coordinateFrameId, sourceHashes,
    bool mask/valid, depth, K and cameraToWorld. sourceHashes requires image, mask,
    geometry and camera SHA256 values. Optional bool domain excludes padding;
    maskComplete defaults to False. One failed or unscoreable view prevents acceptance.
    """
    parts, matrix, prepared, mesh_hash = _prepare(mesh, object_to_native, views)
    return _report(_ray_scene(parts), matrix, prepared, mesh_hash, coarse=coarse)


def assess_model_family(parent_entity_id, members, views, *, coarse=False):
    """Score an explicit mesh family against the parent's whole-object evidence.

    Each member supplies entityId, parentEntityId, coordinateFrameId, mesh and
    objectToNative. Callers supply the current declared family, including its
    residual parent. This report never substitutes for members' own-view checks.
    """
    if (not isinstance(parent_entity_id, str) or not parent_entity_id or
            not isinstance(members, (list, tuple)) or not members or
            any(not isinstance(m, Mapping) or not isinstance(m.get('entityId'), str) or
                not m['entityId'] or not isinstance(m.get('coordinateFrameId'), str) or
                not m['coordinateFrameId'] or
                m.get('parentEntityId') is not None and not isinstance(m['parentEntityId'], str)
                for m in members)):
        raise PlatformError('invalid_quality_family')
    lookup = {m['entityId']: m for m in members}
    if len(lookup) != len(members) or parent_entity_id not in lookup:
        raise PlatformError('invalid_quality_family')
    if lookup[parent_entity_id].get('parentEntityId') in lookup:
        raise PlatformError('invalid_quality_family')
    for identity in lookup:
        seen = set()
        while identity != parent_entity_id:
            if identity not in lookup or identity in seen:
                raise PlatformError('invalid_quality_family')
            seen.add(identity)
            identity = lookup[identity].get('parentEntityId')
    frame = lookup[parent_entity_id]['coordinateFrameId']
    if (any(m['coordinateFrameId'] != frame for m in members) or
            isinstance(views, (list, tuple)) and any(isinstance(v, Mapping) and
                v.get('coordinateFrameId') not in (None, frame) for v in views)):
        raise PlatformError('quality_coordinate_frame_mismatch')
    native_parts, bindings = [], []
    for identity in sorted(lookup):
        member = lookup[identity]
        parts, matrix, _, mesh_hash = _prepare(member.get('mesh'), member.get('objectToNative'), [])
        native_parts.extend(MeshData(transform_points(p.vertices, matrix), p.faces) for p in parts)
        bindings.append({'entityId':identity, 'parentEntityId':member.get('parentEntityId'),
                         'coordinateFrameId':frame, 'meshSha256':mesh_hash,
                         'poseSha256':digest(matrix.tolist())})
    combined = MeshData(native_parts[0].vertices, native_parts[0].faces, primitives=tuple(native_parts))
    report = assess_model(combined, np.eye(4), views, coarse=coarse)
    scope = {'assemblyVersion':'observed-model-family-v1', 'assessmentScope':'parent_family',
             'parentEntityId':parent_entity_id, 'familyMembers':bindings}
    return {**report, **scope,
            'evidenceSha256':digest({'assessment':report['evidenceSha256'], **scope})}


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


def build_model_review_payload(entity, mesh, object_to_native, geometric, candidate_ref,
                               views, records, *, family_binding=None):
    """Build the existing review wire input from already owned, derived evidence.

    Rendering is CPU-only; this function performs no asset or provider I/O and
    neither changes nor grants acceptance to the supplied geometry/evidence.
    """
    def png_data_uri(rgb):
        output = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(output, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()

    rendered = render_model_views(mesh, object_to_native, views)
    pairs = [{'observationId':view['observationId'], 'observationRevision':view['observationRevision'],
              'source':png_data_uri(records[view['imageId']]['rgb']),
              'candidate':png_data_uri(candidate),
              'mask':png_data_uri(view['mask'].astype(np.uint8) * 255)}
             for view, candidate in zip(views, rendered, strict=True)]
    payload = {'entityId':entity['id'], 'label':entity['label'],
               'candidateAssetSha256':candidate_ref['sha256'], 'geometryEvidence':geometric,
               'views':pairs, 'reviewVersion':'coarse-layout-shape-v2'}
    if family_binding is not None:
        payload['familyBinding'] = family_binding
    return payload


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


def refine_model_pose(mesh, object_to_native, views, max_iterations=80, *, coarse=False, fit_scale=False):
    """Bounded pose candidate; optional uniform scale for source-local generated shapes."""
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    if type(max_iterations) is not int or not 1 <= max_iterations <= 500 or type(coarse) is not bool or type(fit_scale) is not bool:
        raise PlatformError("invalid_quality_refinement_iterations")
    parts, initial, prepared, mesh_hash = _prepare(mesh, object_to_native, views)
    scene = _ray_scene(parts)
    before = _report(scene, initial, prepared, mesh_hash, coarse=coarse)
    result = {"method": "bounded_similarity_observed_alignment_v1" if fit_scale else "bounded_rigid_observed_alignment_v1", "accepted": False,
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
        rotation = Rotation.from_rotvec(x[3:6]).as_matrix()
        scale = float(np.exp(x[6])) if fit_scale else 1.
        matrix = initial.copy()
        matrix[:3, :3] = scale * rotation @ initial[:3, :3]
        matrix[:3, 3] = center + scale * rotation @ (initial[:3, 3] - center) + x[:3] * radius
        return matrix

    def loss(scores):
        losses = []
        for score in scores:
            value = 1 - (score["tolerantCoverage"] if coarse else
                         score["iou"] if fit_scale else score["coverage"])
            value += min(score["relativeDepthP50"] if score["relativeDepthP50"] is not None else 1., 1.)
            value += (1 - (score['depthInlierFraction'] or 0.) if coarse else
                      min(score["relativeDepthP95"] if score["relativeDepthP95"] is not None else 1., 1.))
            if coarse:
                value += max(0., 1 - score['depthComparisonPixels'] / THRESHOLDS['minimumDepthPixels'])
            if score["maskComplete"]:
                value += 1 - score["iou"]
            losses.append(value)
        return max(losses) + .1 * sum(losses)

    def objective(x):
        if (np.max(np.abs(x[:3])) > (.4 if fit_scale else .3) or
                np.linalg.norm(x[3:6]) > (1. if fit_scale else .65) or
                fit_scale and abs(x[6]) > .7):
            return 100. + float(x @ x)
        result["evaluations"] += 1
        return loss([_score(scene, candidate(x), view, coarse=coarse) for view in sampled])

    steps = [.025] * 3 + [.04] * 3 + ([.04] if fit_scale else [])
    start = np.zeros(len(steps))
    if fit_scale:
        # Complete silhouettes supply a scale seed that crosses pixel plateaus;
        # partial masks cannot estimate the full object area.
        ratios = [s['targetPixels'] / s['predictedPixels'] for s in before['perView']
                  if s['maskComplete'] and s['predictedPixels'] > 0]
        if ratios:
            start[6] = np.clip(.5*np.log(np.median(ratios)), -.5, .5)
    simplex = np.vstack((start, start + np.diag(steps)))
    fit = minimize(objective, start, method="Nelder-Mead", options={
        "initial_simplex": simplex, "maxiter": max_iterations, "xatol": .0005, "fatol": .0001})
    proposed = _pose(candidate(fit.x))
    after = _report(scene, proposed, prepared, mesh_hash, coarse=coarse)
    accepted = (_nonworsening(before, after) if not fit_scale and not coarse else
                after['status'] == 'observed_consistent' and
                (before['status'] != 'observed_consistent' or loss(after['perView']) < loss(before['perView'])))
    result.update(accepted=accepted, status="accepted" if accepted else "rejected",
                  fullResolutionReassessed=True, reason=None if accepted else "per_view_improvement_not_proven")
    if accepted:
        result.update(objectToNative=proposed.tolist(), after=after)
    return result
