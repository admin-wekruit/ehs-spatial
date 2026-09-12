"""Deterministic spatial contracts. Geometry evidence is never inferred from labels."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence
import math

import numpy as np

from .contracts import PlatformError, digest, validate_transform


def affine(value: Any, *, rigid: bool = False) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != (4, 4) or not np.isfinite(a).all() or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8):
        raise PlatformError("invalid_affine")
    if rigid and (not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(a[:3, :3]), 1, atol=1e-6)):
        raise PlatformError("invalid_rigid_camera")
    return a


def transform_matrix(value: dict[str, Any]) -> np.ndarray:
    validate_transform(value, {value.get("coordinateFrameId")})
    x, y, z, w = value["quaternion"]
    r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    result = np.eye(4)
    result[:3, :3] = r @ np.diag(value["scale"])
    result[:3, 3] = value["position"]
    return result


def matrix_to_transform(matrix: Any, coordinate_frame_id: str) -> dict[str,Any]:
    from scipy.spatial.transform import Rotation
    matrix = affine(matrix)
    scales = np.linalg.norm(matrix[:3,:3],axis=0)
    if np.any(scales <= 0):
        raise PlatformError("invalid_transform_scale")
    rotation = matrix[:3,:3]/scales
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or np.linalg.det(rotation) < .999999:
        raise PlatformError("transform_shear_or_reflection")
    result = {"coordinateFrameId":coordinate_frame_id,"position":matrix[:3,3].tolist(),"quaternion":Rotation.from_matrix(rotation).as_quat().tolist(),"scale":scales.tolist()}
    if not np.allclose(transform_matrix(result),matrix,atol=1e-6):
        raise PlatformError("transform_roundtrip_failed")
    return result


def transform_points(points: np.ndarray, matrix: Any) -> np.ndarray:
    matrix = affine(matrix)
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def camera_intrinsics(value: Any) -> np.ndarray:
    k = np.asarray(value, dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-8) or abs(k[1, 0]) > 1e-8:
        raise PlatformError("invalid_camera_intrinsics")
    return k


def pixel_center_mapping(source_size: Sequence[int], target_size: Sequence[int]) -> np.ndarray:
    """Sizes are (width,height); integer coordinates denote pixel centres."""
    if len(source_size) != 2 or len(target_size) != 2 or any(type(v) is not int or v <= 0 for v in (*source_size, *target_size)):
        raise PlatformError("invalid_image_dimensions")
    sx, sy = np.asarray(target_size) / np.asarray(source_size)
    return np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])


def project_native(points: np.ndarray, camera: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return source pixel centres and camera depth, retaining skew."""
    k = camera_intrinsics(camera["K"])
    local = transform_points(points, np.linalg.inv(affine(camera["cameraToWorld"], rigid=True)))
    homogeneous = local @ k.T
    uv = np.full(local.shape[:-1] + (2,), np.nan)
    np.divide(homogeneous[..., :2], local[..., 2, None], out=uv, where=local[..., 2, None] > 0)
    return uv, local[..., 2]


def unproject_pixels(pixels: np.ndarray, depths: np.ndarray, camera: Mapping[str, Any]) -> np.ndarray:
    pixels, depths = np.asarray(pixels, dtype=float), np.asarray(depths, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or depths.shape != pixels.shape[:1] or not np.isfinite(pixels).all() or not np.isfinite(depths).all() or np.any(depths <= 0):
        raise PlatformError("invalid_pixel_depth")
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(camera_intrinsics(camera["K"])).T
    return transform_points(rays * depths[:, None], affine(camera["cameraToWorld"], rigid=True))


def intersect_camera_plane(pixel: Sequence[float], camera: Mapping[str, Any], plane: Sequence[float]) -> np.ndarray:
    """Explicit support-plane anchoring; caller records manual placement provenance."""
    p = np.asarray(plane, dtype=float)
    if p.shape != (4,) or not np.isfinite(p).all() or np.linalg.norm(p[:3]) == 0:
        raise PlatformError("invalid_support_plane")
    c2w = affine(camera["cameraToWorld"], rigid=True)
    point = unproject_pixels(np.asarray([pixel]), np.ones(1), camera)[0]
    origin, ray = c2w[:3, 3], point-c2w[:3, 3]
    denominator = float(p[:3] @ ray)
    if abs(denominator) <= 1e-10:
        raise PlatformError("anchor_ray_parallel_to_surface")
    distance = -float(p[:3] @ origin+p[3])/denominator
    if distance <= 0:
        raise PlatformError("anchor_surface_behind_camera")
    return origin+distance*ray


@dataclass(frozen=True)
class FrameGeometry:
    image_id: str
    coordinate_frame_id: str
    image_sha256: str
    points: np.ndarray
    valid: np.ndarray
    K: np.ndarray
    camera_to_world: np.ndarray

    def __post_init__(self):
        if self.points.ndim != 3 or self.points.shape[2] != 3 or self.valid.shape != self.points.shape[:2]:
            raise PlatformError("geometry_grid_mismatch")
        camera_intrinsics(self.K)
        affine(self.camera_to_world, rigid=True)

    def camera(self) -> dict[str, Any]:
        return {"K": self.K, "cameraToWorld": self.camera_to_world}

    def support(self) -> np.ndarray:
        _, depth = project_native(self.points, self.camera())
        return self.valid.astype(bool) & np.isfinite(self.points).all(axis=-1) & (depth > 0)


@dataclass(frozen=True)
class MaskObservation:
    id: str
    image_id: str
    mask: np.ndarray


@dataclass(frozen=True)
class AssociationConfig:
    # Engineering acceptance settings; these are not calibrated confidence scores.
    min_support: int = 32
    min_containment: float = .65
    relative_depth_tolerance: float = .03
    best_margin: float = .15

    def __post_init__(self):
        if self.min_support < 1 or not 0 < self.min_containment <= 1 or not 0 < self.relative_depth_tolerance < 1 or not 0 <= self.best_margin <= 1:
            raise PlatformError("invalid_association_config")


def _direction(source: MaskObservation, target: MaskObservation, frames: Mapping[str, FrameGeometry], config: AssociationConfig, support_points: Mapping[str, np.ndarray], grids: Mapping[str, tuple[np.ndarray,np.ndarray]]) -> dict[str, Any]:
    a, b = frames[source.image_id], frames[target.image_id]
    points = support_points[source.id]
    uv, z = project_native(points, b.camera())
    h, w = b.valid.shape
    inside = np.isfinite(uv).all(axis=1) & (z > 0)
    inside &= (uv[:, 0] >= -.5) & (uv[:, 0] < w-.5) & (uv[:, 1] >= -.5) & (uv[:, 1] < h-.5)
    xy = np.floor(uv[inside] + .5).astype(int)
    z = z[inside]
    if not len(xy):
        return {"support": 0, "containment": None, "depthConsistent": 0}
    # Count independent target pixels, not a dense source splatting onto one pixel.
    _, indices = np.unique(xy[:, 1]*w + xy[:, 0], return_index=True)
    xy, z = xy[indices], z[indices]
    target_support, target_depth = grids[target.image_id]
    visible = target_support[xy[:, 1], xy[:, 0]]
    dz = target_depth[xy[:, 1], xy[:, 0]]
    # Closer target surfaces occlude source support. Missing/inconsistent depths
    # cannot become positive identity evidence.
    consistent = visible & (np.abs(z-dz) <= config.relative_depth_tolerance * np.maximum(dz, 1e-12))
    count = int(consistent.sum())
    contained = target.mask[xy[:, 1], xy[:, 0]][consistent].astype(bool)
    return {"support": count, "containment": float(contained.mean()) if count else None, "depthConsistent": count}


def associate_observations(observations: Sequence[MaskObservation], frames: Mapping[str, FrameGeometry], config: AssociationConfig = AssociationConfig()) -> dict[str, Any]:
    """Return observation groups; caller allocates durable entity UUIDs once.

    No same-frame or transitive bridge merges. Unresolved observations are singleton
    groups, including one-pixel detections and observations without geometry.
    """
    obs = sorted(observations, key=lambda x: x.id)
    if len({o.id for o in obs}) != len(obs):
        raise PlatformError("duplicate_observation_id")
    for o in obs:
        if o.image_id in frames and o.mask.shape != frames[o.image_id].valid.shape:
            raise PlatformError("mask_geometry_grid_mismatch", observationId=o.id)
    grids = {}
    for image_id, frame in frames.items():
        _, depth = project_native(frame.points, frame.camera())
        grids[image_id] = (frame.valid.astype(bool) & np.isfinite(frame.points).all(axis=-1) & (depth > 0), depth)
    support_points = {o.id: frames[o.image_id].points[o.mask.astype(bool) & grids[o.image_id][0]] for o in obs if o.image_id in frames}
    # ponytail: all candidate pairs are O(N²); cached grids avoid O(N²*image size).
    # If inventories grow beyond workcells, shortlist pairs using projected bounds.
    links, scores = [], {}
    for i, a in enumerate(obs):
        for b in obs[i+1:]:
            if a.image_id == b.image_id or a.image_id not in frames or b.image_id not in frames:
                continue
            if frames[a.image_id].coordinate_frame_id != frames[b.image_id].coordinate_frame_id:
                continue
            ab, ba = _direction(a, b, frames, config, support_points, grids), _direction(b, a, frames, config, support_points, grids)
            enough = min(ab["support"], ba["support"]) >= config.min_support
            score = min(ab["containment"] or 0, ba["containment"] or 0) if enough else 0.
            link = {"observationIds": [a.id, b.id], "directions": [ab, ba], "score": score, "eligible": enough and score >= config.min_containment, "accepted": False}
            links.append(link)
            scores[a.id, b.id] = scores[b.id, a.id] = score
    lookup = {o.id: o for o in obs}
    def clear_best(a: str, b: str) -> bool:
        competitors = [scores.get((a, o.id), 0.) for o in obs if o.id != b and o.image_id == lookup[b].image_id]
        return scores[a, b] - max(competitors, default=0.) >= config.best_margin
    accepted = set()
    for link in links:
        a, b = link["observationIds"]
        if link["eligible"] and clear_best(a, b) and clear_best(b, a):
            accepted.add(frozenset((a, b)))
    groups = [[o.id] for o in obs]
    for link in sorted(links, key=lambda x: (-x["score"], x["observationIds"])):
        a, b = link["observationIds"]
        if frozenset((a, b)) not in accepted:
            continue
        ga, gb = next(g for g in groups if a in g), next(g for g in groups if b in g)
        if ga is gb:
            link["accepted"] = True
            continue
        image_ids = [lookup[x].image_id for x in ga+gb]
        if len(set(image_ids)) != len(image_ids):
            continue
        if all(frozenset((x, y)) in accepted for x in ga for y in gb):
            groups.remove(gb)
            ga.extend(gb)
            ga.sort()
            link["accepted"] = True
    grouped = {x: g for g in groups for x in g}
    for link in links:
        a, b = link["observationIds"]
        link["accepted"] = grouped[a] is grouped[b]
    return {"groups": sorted(groups), "links": links, "ambiguous": [x for x in links if x["eligible"] and not x["accepted"]], "config": asdict(config)}


def stage_cache_key(stage: str, frames: Sequence[Mapping[str, Any]], pins: Mapping[str, str], parameters: Mapping[str, Any] | None = None, source_refs: Sequence[Any] = ()) -> str:
    """Per-frame callers pass exactly one frame; joint geometry passes all frames."""
    if not stage or not pins or any(not key or not value for key, value in pins.items()):
        raise PlatformError("unpinned_provider")
    normalized = []
    for frame in frames:
        sha = frame.get("sha256", frame.get("imageSha256"))
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise PlatformError("invalid_image_hash")
        normalized.append({"imageId": frame["imageId"], "sha256": sha, "pixelMapping": frame.get("pixelMapping", [])})
    if len({f["imageId"] for f in normalized}) != len(normalized):
        raise PlatformError("duplicate_image_id")
    return digest({"version": 1, "stage": stage, "frames": normalized, "pins": dict(pins), "parameters": dict(parameters or {}), "sourceRefs": list(source_refs)})


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray | None = None
    uv: np.ndarray | None = None
    texture_bytes: bytes | None = None
    texture_mime_type: str | None = None
    material: Mapping[str, Any] | None = None
    primitives: tuple["MeshData", ...] = ()

    def __post_init__(self):
        v, f = np.asarray(self.vertices), np.asarray(self.faces)
        if v.ndim != 2 or v.shape[1] != 3 or len(v) < 3 or not np.isfinite(v).all() or f.ndim != 2 or f.shape[1] != 3 or not len(f) or f.dtype.kind not in "iu" or f.min() < 0 or f.max() >= len(v):
            raise PlatformError("invalid_mesh")
        if self.colors is not None:
            c = np.asarray(self.colors)
            if c.shape not in ((len(v),3),(len(v),4)) or not np.isfinite(c).all() or np.any(c < 0) or np.any(c > 1):
                raise PlatformError("invalid_vertex_colors")
        if self.uv is not None and (np.asarray(self.uv).shape != (len(v),2) or not np.isfinite(self.uv).all()):
            raise PlatformError("invalid_mesh_uv")
        if self.texture_bytes is not None and (self.uv is None or not isinstance(self.texture_bytes,bytes) or self.texture_mime_type not in ("image/png","image/jpeg")):
            raise PlatformError("invalid_mesh_texture")
        if any(not isinstance(p,MeshData) or p.primitives for p in self.primitives):
            raise PlatformError("invalid_mesh_primitives")


def primitive_mesh(spec: Mapping[str, Any]) -> MeshData:
    """Local native units; centred box or Z-axis cylinder. No ground assumption."""
    kind = spec.get("type", spec.get("kind"))
    if kind == "box":
        size = np.asarray(spec.get("dimensions"), dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise PlatformError("invalid_primitive_dimensions")
        v = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=float) * size / 2
        f = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
    elif kind == "cylinder":
        radius, height, n = spec.get("radius"), spec.get("height"), spec.get("segments", 64)
        if any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0 for x in (radius, height)) or type(n) is not int or not 8 <= n <= 256:
            raise PlatformError("invalid_primitive_dimensions")
        angles = np.arange(n)*2*np.pi/n
        ring = np.column_stack((radius*np.cos(angles), radius*np.sin(angles)))
        v = np.vstack((np.column_stack((ring, np.full(n,-height/2))), np.column_stack((ring, np.full(n,height/2))), [0,0,-height/2], [0,0,height/2]))
        f = []
        for i in range(n):
            j = (i+1) % n
            f.extend([[i,j,n+j],[i,n+j,n+i],[2*n,j,i],[2*n+1,n+i,n+j]])
    else:
        raise PlatformError("unsupported_primitive")
    return MeshData(v.astype(np.float32), np.asarray(f, dtype=np.uint32))


def ground_measurements(vertices: np.ndarray, transform: Any, ground: Mapping[str, Any] | None, native_to_meters: float | None = None) -> dict[str, Any]:
    points = transform_points(vertices, transform)
    result = {"dimensionsNative": np.ptp(points, axis=0).tolist(), "groundHeightNative": None, "groundTiltDegrees": None, "groundHeightMeters": None}
    if ground is None:
        return result
    plane = np.asarray(ground.get("plane"), dtype=float)
    if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) == 0:
        raise PlatformError("invalid_ground")
    plane /= np.linalg.norm(plane[:3])
    signed = points @ plane[:3] + plane[3]
    axis = affine(transform)[:3, 2]
    result["groundHeightNative"] = float(np.max(signed))
    result["groundTiltDegrees"] = math.degrees(math.acos(float(np.clip(abs(np.dot(axis/np.linalg.norm(axis), plane[:3])), 0, 1))))
    if native_to_meters is not None:
        if not math.isfinite(native_to_meters) or native_to_meters <= 0:
            raise PlatformError("invalid_scale")
        result["groundHeightMeters"] = result["groundHeightNative"] * native_to_meters
    return result


def estimate_native_ground(frames: Mapping[str, FrameGeometry], observations: Sequence[MaskObservation]) -> tuple[dict[str,Any] | None,dict[str,Any]]:
    """Fit only explicit floor-mask evidence; preserve native scale and cameras.

    Thresholds are conservative engineering configuration, not calibrated
    confidence. Reuses the product's deterministic RANSAC, without its camera-up
    assumption, camera-height anchor, metric tolerance, or coordinate rewrite.
    """
    config = {"minPointsPerView":200,"requiredViews":min(2,len(frames)),"relativeDistanceThreshold":.005,
              "minInlierFraction":.75,"minSpreadRatio":.08,"maxCrossViewAngleDegrees":5.,"maxFitPoints":60000}
    report = {"method":"semantic_floor_mask_native_ransac_v1","configuration":config,"status":"insufficient_evidence"}
    def missing(reason):
        return None,{**report,"reason":reason}
    if not frames or not observations:
        return missing("no_explicit_floor_support")
    if len({f.coordinate_frame_id for f in frames.values()}) != 1:
        return missing("unregistered_coordinate_frames")
    masks = {}
    for observation in observations:
        frame = frames.get(observation.image_id)
        if frame is None or observation.mask.shape != frame.valid.shape:
            return missing("floor_mask_grid_mismatch")
        masks.setdefault(observation.image_id,np.zeros_like(frame.valid))
        masks[observation.image_id] |= observation.mask.astype(bool)
    selected = {fid:frames[fid].points[mask & frames[fid].support()] for fid,mask in masks.items()}
    selected = {fid:p for fid,p in selected.items() if len(p) >= config["minPointsPerView"]}
    if len(selected) < config["requiredViews"]:
        return missing("insufficient_floor_views_or_pixels")
    points = np.concatenate(list(selected.values()))
    if len(points) > config["maxFitPoints"]:
        points = points[::int(np.ceil(len(points)/config["maxFitPoints"]))]
    center = np.median(points,axis=0)
    extent = float(np.quantile(np.linalg.norm(points-center,axis=1),.95))
    if not np.isfinite(extent) or extent <= 1e-10:
        return missing("degenerate_floor_support")
    tolerance = extent*config["relativeDistanceThreshold"]
    cameras = np.asarray([f.camera_to_world[:3,3] for f in frames.values()])
    _,_,vectors = np.linalg.svd(points-center,full_matrices=False)
    normal = vectors[-1]
    if np.median((cameras-center)@normal) < 0:
        normal = -normal
    from ..geometry import _ransac_floor_plane
    inliers = _ransac_floor_plane(points,cameras,normal,tolerance)
    if inliers is None or len(inliers) < 200 or len(inliers)/len(points) < config["minInlierFraction"]:
        return missing("floor_consensus_too_low")
    support = points[inliers]
    center = support.mean(axis=0)
    _,singular,vectors = np.linalg.svd(support-center,full_matrices=False)
    if singular[1]/max(singular[0],1e-20) < config["minSpreadRatio"] or singular[1]/np.sqrt(len(support)) < 20*tolerance:
        return missing("floor_support_not_two_dimensional")
    normal = vectors[-1]
    if np.median((cameras-center)@normal) < 0:
        normal = -normal
    offset = -float(normal@center)
    camera_heights = cameras@normal+offset
    if not np.isfinite(camera_heights).all() or np.any(camera_heights <= 3*tolerance):
        return missing("cameras_not_consistently_above_floor")
    per_view = []
    for fid,view_points in selected.items():
        residuals = np.abs(view_points@normal+offset)
        accepted = view_points[residuals <= tolerance]
        if len(accepted) < 200 or len(accepted)/len(view_points) < config["minInlierFraction"]:
            return missing("floor_cross_view_support_inconsistent")
        local_center = accepted.mean(axis=0)
        _,sv,vt = np.linalg.svd(accepted-local_center,full_matrices=False)
        if sv[1]/max(sv[0],1e-20) < config["minSpreadRatio"]:
            return missing("floor_view_support_degenerate")
        angle = float(np.degrees(np.arccos(np.clip(abs(vt[-1]@normal),0,1))))
        if angle > config["maxCrossViewAngleDegrees"] or abs(normal@local_center+offset) > 2*tolerance:
            return missing("floor_cross_view_plane_inconsistent")
        per_view.append({"imageId":fid,"validPixels":len(view_points),"inlierPixels":len(accepted),"normalDifferenceDegrees":angle})
    residuals = np.abs(support@normal+offset)
    report.update(status="estimated",inlierFraction=len(inliers)/len(points),fitPointCount=len(points),inlierPointCount=len(inliers),
                  thresholdNative=tolerance,residualMedianNative=float(np.median(residuals)),residualP95Native=float(np.quantile(residuals,.95)),views=per_view)
    ground = {"plane":[*normal.tolist(),offset],"normal":normal.tolist(),"origin":(-offset*normal).tolist(),"status":"estimated",
              "source":"observed_floor_mask_and_estimated_depth","unit":"native","uncertainty":{"status":"not_quantified","causes":["semantic_floor_hypothesis","estimated_depth","partial_visible_support"]}}
    return ground,report


@dataclass(frozen=True)
class GeometryResult:
    frames: Mapping[str, FrameGeometry]
    source_refs: tuple[Mapping[str, Any], ...] = ()


class GeometryProvider(Protocol):
    pins: Mapping[str, str]
    def reconstruct(self, images: Sequence[Mapping[str, Any]]) -> GeometryResult: ...


@dataclass(frozen=True)
class DiscoveryResult:
    image_id: str
    items: tuple[Mapping[str,Any], ...]
    source_refs: tuple[Mapping[str,Any], ...] = ()


class DiscoveryProvider(Protocol):
    pins: Mapping[str,str]
    def discover(self, image: Mapping[str,Any]) -> DiscoveryResult: ...


@dataclass(frozen=True)
class DepthResult:
    image_id: str
    image_sha256: str
    points: np.ndarray
    valid: np.ndarray
    intrinsics: np.ndarray
    source_refs: tuple[Mapping[str,Any], ...] = ()


class DepthProvider(Protocol):
    pins: Mapping[str,str]
    def estimate_depth(self, image: Mapping[str,Any]) -> DepthResult: ...


class SegmentationProvider(Protocol):
    pins: Mapping[str, str]
    def segment(self, image: Mapping[str, Any], prompts: Sequence[Mapping[str, Any]]) -> Sequence[MaskObservation]: ...


@dataclass(frozen=True)
class GenerationRequest:
    entity_id: str
    image: np.ndarray
    mask: np.ndarray
    frame: FrameGeometry
    seed: int = 0
    source_refs: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class ShapeResult:
    entity_id: str
    mesh: MeshData
    # A valid provider pose is a proposal, not independent evidence of placement.
    proposed_object_to_native: np.ndarray | None
    placement_state: str = "unconfirmed"
    provenance: Mapping[str, Any] = field(default_factory=dict)


class ShapeProvider(Protocol):
    pins: Mapping[str, str]
    def generate(self, request: GenerationRequest) -> ShapeResult: ...


def verify_native_pose(raw_vertices: np.ndarray, official_posed_vertices: np.ndarray, object_to_provider: Any, provider_to_opencv: Any, camera_to_world: Any, tolerance: float = 2e-5) -> tuple[np.ndarray, float]:
    """The official posed vertices must already include the official mesh basis.

    Adapter output must compose that same basis into object_to_provider. This
    comparison prevents copying RecGen conventions or double-applying a pose.
    """
    raw, official = np.asarray(raw_vertices), np.asarray(official_posed_vertices)
    if raw.shape != official.shape or not raw.size or not np.isfinite(raw).all() or not np.isfinite(official).all():
        raise PlatformError("invalid_official_pose_fixture")
    pose = affine(object_to_provider)
    residual = float(np.max(np.abs(transform_points(raw, pose)-official)))
    if residual > tolerance:
        raise PlatformError("sam3d_pose_mismatch", residual=residual)
    conversion = affine(provider_to_opencv, rigid=True)
    return affine(camera_to_world, rigid=True) @ conversion @ pose, residual


class SAM3DMeshAdapter:
    """A gated mesh-only adapter around a pinned official runtime callable.

    The runtime returns mesh + official posed vertices + explicit object matrix.
    Live release evidence is REQUIRED; synthetic unit tests do not enable it.
    """
    def __init__(self, runtime: Callable[..., Mapping[str, Any]], pins: Mapping[str, str], release_evidence: Mapping[str, Any], provider_to_opencv: Any):
        self.runtime, self.pins = runtime, dict(pins)
        self.provider_to_opencv = affine(provider_to_opencv, rigid=True)
        checks = ("licenseAudit", "meshOnlyDependencyAudit", "officialPoseFixture", "externalPointmapNoDepth")
        if not pins or any(not v for v in pins.values()) or release_evidence.get("pins") != self.pins or any(not isinstance(release_evidence.get(c), dict) or release_evidence[c].get("status") != "passed" or not release_evidence[c].get("artifactSha256") for c in checks):
            raise PlatformError("sam3d_release_gate_unverified", 409)
        if release_evidence.get("providerToOpenCV") != self.provider_to_opencv.tolist():
            raise PlatformError("sam3d_basis_gate_unverified", 409)
        self.evidence_sha = digest(release_evidence)
        self.research_only = False

    @classmethod
    def for_research(cls, runtime, pins, provider_to_opencv, protocol):
        """Explicit experiment-only construction; never supplies release approval."""
        if not isinstance(protocol,dict) or not {"id","inputHashes","baselineRevision","metricDefinitions","policyThresholds","split"} <= set(protocol):
            raise PlatformError("frozen_research_protocol_required",409)
        instance = cls.__new__(cls)
        instance.runtime,instance.pins = runtime,dict(pins)
        instance.provider_to_opencv = affine(provider_to_opencv,rigid=True)
        instance.evidence_sha,instance.research_only = digest(protocol),True
        return instance

    def generate(self, request: GenerationRequest) -> ShapeResult:
        frame = request.frame
        if request.image.shape[:2] != frame.valid.shape or request.mask.shape != frame.valid.shape or not request.mask.astype(bool).any():
            raise PlatformError("generation_grid_mismatch")
        # A sparse target can still yield shape, but receives no measured pose.
        camera_points = transform_points(frame.points, np.linalg.inv(frame.camera_to_world))
        provider_points = transform_points(camera_points, np.linalg.inv(self.provider_to_opencv))
        if not np.isfinite(provider_points).all():
            raise PlatformError("sam3d_pointmap_nonfinite")
        result = self.runtime(image=request.image, mask=request.mask, pointmap=provider_points,
                              seed=request.seed, decode_formats=["mesh"], with_mesh_postprocess=False,
                              with_texture_baking=False, with_layout_postprocess=False, use_vertex_color=True)
        if result.get("internalDepthCalls") != 0 or result.get("decodeFormats") != ["mesh"]:
            raise PlatformError("sam3d_runtime_contract_violation")
        mesh = MeshData(np.asarray(result["vertices"]), np.asarray(result["faces"]), np.asarray(result["colors"]) if result.get("colors") is not None else None)
        pose, residual = verify_native_pose(mesh.vertices, result["officialPosedVertices"], result["objectToProvider"], self.provider_to_opencv, frame.camera_to_world)
        support = int((request.mask.astype(bool) & frame.support()).sum())
        return ShapeResult(request.entity_id, mesh, pose if support >= 8 else None, provenance={
            "pins": self.pins, "sourceRefs": list(request.source_refs), "seed": request.seed,
            "poseResidual": residual, "observedSupportPixels": support, "releaseEvidenceSha256": None if self.research_only else self.evidence_sha,
            "researchProtocolSha256": self.evidence_sha if self.research_only else None,
            "shapeStatus": "ready", "placementReason": "requires_alignment_confirmation" if support >= 8 else "insufficient_observed_depth"})
