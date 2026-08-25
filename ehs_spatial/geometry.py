from collections import Counter
from dataclasses import dataclass
import re

import numpy as np
import open3d as o3d
from PIL import Image
from shapely.geometry import MultiPoint, Point, Polygon

from .contracts import Entity3D, GeometryFrame, Observation2D


FLOOR_LABEL = "factory floor"


@dataclass(frozen=True)
class _FrameData:
    points: np.ndarray
    valid: np.ndarray
    shape: tuple[int, int]


@dataclass(frozen=True)
class _FloorTransform:
    plane: tuple[float, float, float, float]
    scale_factor: float
    rotation: np.ndarray
    origin: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale_factor * ((points - self.origin) @ self.rotation.T)


@dataclass(frozen=True)
class _GeometryResult:
    transform: _FloorTransform | None
    entities: list[Entity3D]
    warnings: list[str]


def _load_frame(frame: GeometryFrame) -> _FrameData:
    points = np.load(frame.pts3d_path, allow_pickle=False)
    confidence = np.load(frame.conf_path, allow_pickle=False)
    valid = np.load(frame.valid_mask_path, allow_pickle=False)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"frame {frame.frame_id} pts3d must have shape HxWx3")
    shape = points.shape[:2]
    if confidence.shape != shape:
        raise ValueError(f"frame {frame.frame_id} confidence shape does not match pts3d")
    if valid.shape != shape:
        raise ValueError(f"frame {frame.frame_id} valid mask shape does not match pts3d")
    with Image.open(frame.canonical_image_path) as image:
        if image.size != (shape[1], shape[0]):
            raise ValueError(
                f"frame {frame.frame_id} canonical image shape does not match pts3d"
            )
    return _FrameData(points=points, valid=valid.astype(bool), shape=shape)


def _select_points(observation: Observation2D, frame_data: _FrameData) -> np.ndarray:
    if observation.mask_path is None:
        raise ValueError(f"observation {observation.observation_id} has no local mask_path")
    with Image.open(observation.mask_path) as image:
        mask = np.asarray(image)
    if mask.shape != frame_data.shape:
        raise ValueError(
            f"observation {observation.observation_id} mask shape does not match pts3d"
        )
    selected = mask.astype(bool) & frame_data.valid
    selected &= np.isfinite(frame_data.points).all(axis=2)
    return frame_data.points[selected]


def _data_for(
    frame_id: str,
    frames: dict[str, GeometryFrame],
    loaded: dict[str, _FrameData],
) -> _FrameData:
    if frame_id not in loaded:
        loaded[frame_id] = _load_frame(frames[frame_id])
    return loaded[frame_id]


def _rotation_to_positive_z(normal: np.ndarray) -> np.ndarray:
    target = np.array([0.0, 0.0, 1.0])
    cross = np.cross(normal, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(normal, target))
    if sine < 1e-12:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


_FLOOR_NORMAL_MAX_ANGLE_COS = float(np.cos(np.radians(20.0)))
_FLOOR_RANSAC_ITERATIONS = 1000
_FLOOR_MAX_FIT_POINTS = 60_000


def _camera_up_prior(frames: dict[str, GeometryFrame]) -> np.ndarray:
    # OpenCV camera convention points +Y down in the image, so -Y of each
    # camera_to_world rotation approximates world "up" for roughly upright
    # captures (nerfstudio-style average-camera-up prior). Used to gate plane
    # orientation and to place the floor below the cameras.
    ups = [
        -np.asarray(frame.camera_to_world, dtype=float)[:3, 1]
        for frame in frames.values()
    ]
    mean_up = np.mean(ups, axis=0)
    norm = float(np.linalg.norm(mean_up))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return mean_up / norm


def _ransac_floor_plane(
    points: np.ndarray,
    camera_centers: np.ndarray,
    up_axis: np.ndarray,
    distance_threshold: float,
) -> np.ndarray | None:
    # Deterministic NumPy RANSAC replacing Open3D segment_plane, whose
    # OpenMP-parallel implementation returns different inlier sets run-to-run
    # even when seeded. Candidate normals must lie near the up axis (rejects
    # walls) and are oriented toward it. The floor is the LOWEST adequately
    # supported horizontal plane below the cameras, not the largest: in real
    # close-up captures a tabletop or seat cushion can dominate the view, so
    # max-consensus alone selects furniture, not floor.
    rng = np.random.default_rng(0)
    sample_indices = rng.integers(0, len(points), size=(_FLOOR_RANSAC_ITERATIONS, 3))
    a = points[sample_indices[:, 0]]
    b = points[sample_indices[:, 1]]
    c = points[sample_indices[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-12
    normals = normals[keep] / lengths[keep, None]
    anchors = a[keep]

    alignment = normals @ up_axis
    aligned = np.abs(alignment) >= _FLOOR_NORMAL_MAX_ANGLE_COS
    normals = normals[aligned] * np.sign(alignment[aligned])[:, None]
    anchors = anchors[aligned]
    if not len(normals):
        return None
    offsets = -np.einsum("ij,ij->i", normals, anchors)

    counts = np.zeros(len(normals), dtype=np.int64)
    for start in range(0, len(normals), 64):
        distances = np.abs(
            points @ normals[start : start + 64].T + offsets[start : start + 64]
        )
        counts[start : start + 64] = (distances < distance_threshold).sum(axis=0)
    if not counts.max():
        return None
    finite_cameras = camera_centers[np.isfinite(camera_centers).all(axis=1)]
    cameras_above = (
        ((finite_cameras @ normals.T + offsets) > 0).all(axis=0)
        if len(finite_cameras)
        else np.ones(len(normals), dtype=bool)
    )
    support_floor = max(150, int(0.1 * counts.max()))
    eligible = cameras_above & (counts >= support_floor)
    if eligible.any():
        # Rank candidates by the median up-coordinate of their actual inlier
        # points (extrapolation-proof: a tilted plane can dip below the floor
        # far from its own support, but its inliers cannot). Floor = lowest.
        candidates = np.flatnonzero(eligible)
        points_up = points @ up_axis
        median_heights = np.empty(len(candidates))
        for position, candidate in enumerate(candidates):
            member = (
                np.abs(points @ normals[candidate] + offsets[candidate])
                < distance_threshold
            )
            median_heights[position] = np.median(points_up[member])
        chosen = int(candidates[np.argmin(median_heights)])
    else:
        chosen = int(np.argmax(counts))
    distances = np.abs(points @ normals[chosen] + offsets[chosen])
    return np.flatnonzero(distances < distance_threshold)


def _fit_floor(
    frames: dict[str, GeometryFrame],
    frame_data: dict[str, _FrameData],
    camera_height_m: float | None,
    *,
    scale_factor_override: float | None = None,
) -> tuple[_FloorTransform | None, list[str]]:
    # The floor is a geometric primitive (dominant horizontal plane below the
    # cameras), not a semantic class: it is fitted from the full reconstructed
    # cloud so no segmentation model has to recognise it.
    selected_by_frame: list[tuple[str, np.ndarray]] = []
    for frame_id in sorted(frames):
        data = _data_for(frame_id, frames, frame_data)
        finite = data.valid & np.isfinite(data.points).all(axis=2)
        points = data.points[finite]
        if len(points):
            selected_by_frame.append((frame_id, points))

    evidence_frames = {frame_id for frame_id, _ in selected_by_frame}
    point_count = sum(len(points) for _, points in selected_by_frame)
    # Gates scale down for reduced captures (a monocular pointmap is one
    # frame's worth of evidence by construction).
    required_frames = min(2, len(frames))
    if len(evidence_frames) < required_frames or point_count < 200:
        return None, [
            f"floor evidence requires at least {required_frames} distinct "
            "frame(s) and 200 selected points"
        ]

    floor_points = np.vstack([points for _, points in selected_by_frame])
    if len(floor_points) > _FLOOR_MAX_FIT_POINTS:
        stride = int(np.ceil(len(floor_points) / _FLOOR_MAX_FIT_POINTS))
        floor_points = floor_points[::stride]
    camera_centers = np.asarray(
        [np.asarray(frame.camera_to_world)[:3, 3] for frame in frames.values()]
    )
    up_axis = _camera_up_prior(frames)
    scale_factor: float | None = None
    for fit_index in range(2):
        distance_threshold = 0.03 if scale_factor is None else 0.03 / scale_factor
        inlier_indices = _ransac_floor_plane(
            floor_points, camera_centers, up_axis, distance_threshold
        )
        if inlier_indices is None or len(inlier_indices) < 3:
            return None, ["floor plane RANSAC produced fewer than 3 inliers"]

        inliers = floor_points[inlier_indices]
        center = np.mean(inliers, axis=0)
        _, singular_values, right_vectors = np.linalg.svd(
            inliers - center, full_matrices=False
        )
        if not np.isfinite(singular_values).all() or singular_values[1] <= 1e-12:
            return None, ["floor plane inliers are degenerate"]
        normal = right_vectors[-1]
        normal /= np.linalg.norm(normal)
        offset = -float(np.dot(normal, center))

        if float(np.dot(normal, up_axis)) < 0:
            normal = -normal
            offset = -offset
        signed_heights = camera_centers @ normal + offset
        if not np.isfinite(signed_heights).all() or np.any(signed_heights <= 0):
            return None, [
                "camera-to-floor heights must all be finite and strictly positive"
            ]
        predicted_height = float(np.median(signed_heights))
        if scale_factor_override is not None:
            # An external metric anchor (MoGe) supplies the raw->metres
            # gauge; the fitted plane still provides orientation and origin.
            scale_factor = scale_factor_override
        elif camera_height_m is not None:
            scale_factor = camera_height_m / predicted_height
        else:
            return None, [
                "no scale source: neither an anchor override nor an "
                "operator camera height is available"
            ]
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            return None, ["resolved scale is nonfinite or nonpositive"]
        if fit_index == 1 and scale_factor_override is None:
            scaled_heights = signed_heights * scale_factor
            scaled_mad = float(
                np.median(np.abs(scaled_heights - np.median(scaled_heights)))
            )
            if scaled_mad > 0.25:
                return None, ["scaled camera-height MAD exceeds 0.25 m"]

    origin = -offset * normal
    return (
        _FloorTransform(
            plane=tuple(float(value) for value in (*normal, offset)),
            scale_factor=float(scale_factor),
            rotation=_rotation_to_positive_z(normal),
            origin=origin,
        ),
        [],
    )


def _voxel_downsample(points: np.ndarray) -> np.ndarray:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return np.asarray(cloud.voxel_down_sample(voxel_size=0.03).points)


# Depth reconstruction smears thin structures (fence rails) toward the
# background; statistical denoising trims that tail before clustering.
# Conservative on purpose: nb=20/std=2.0 leaves dense uniform structure
# untouched, while harsher settings (nb=50/std=0.8) measurably shrink
# single-view smear but delete most real fence points too — tradeoff
# numbers in docs/reviews/2026-07-20-sor-preclustering.md.
_OUTLIER_NB_NEIGHBORS = 20
_OUTLIER_STD_RATIO = 2.0


def _remove_outliers(points: np.ndarray) -> np.ndarray:
    if len(points) <= _OUTLIER_NB_NEIGHBORS:
        return points
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cleaned, _ = cloud.remove_statistical_outlier(
        nb_neighbors=_OUTLIER_NB_NEIGHBORS, std_ratio=_OUTLIER_STD_RATIO
    )
    return np.asarray(cleaned.points)


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def _spatial_state(
    points: np.ndarray, height: float
) -> tuple[float | None, float | None, float | None]:
    """Return (orientation_deg, tilt_deg, overhang_m); None per field on degenerate input.

    orientation_deg: yaw of the XY first principal axis, [0, 180) (undirected).
    tilt_deg: angle of the 3D first principal axis from +Z, [0, 90].
    overhang_m: max XY distance of upper-band points (z > 60% of height)
    outside the base-band (z < 40% of height) convex hull.
    """
    orientation = None
    if len(points) >= 3:
        xy_eigenvalues, xy_eigenvectors = np.linalg.eigh(
            np.cov(points[:, :2], rowvar=False)
        )
        # eigenvalue ratio < 1.2 means near-isotropic: no trustworthy axis
        if xy_eigenvalues[1] > 0 and (
            xy_eigenvalues[0] <= 0 or xy_eigenvalues[1] / xy_eigenvalues[0] >= 1.2
        ):
            major = xy_eigenvectors[:, 1]
            orientation = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
            if orientation >= 180.0:  # float rounding at the wrap point
                orientation = 0.0

    tilt = None
    if len(points) >= 20:
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points, rowvar=False))
        # elongation ratio < 1.5 means no dominant 3D axis to measure against +Z
        if eigenvalues[2] > 0 and (
            eigenvalues[1] <= 0 or eigenvalues[2] / eigenvalues[1] >= 1.5
        ):
            axis_z = min(1.0, abs(float(eigenvectors[2, 2])))
            tilt = float(np.degrees(np.arccos(axis_z)))

    overhang = None
    upper = points[points[:, 2] > 0.6 * height][:, :2]
    base = points[points[:, 2] < 0.4 * height][:, :2]
    if len(upper) >= 10 and len(base) >= 10:
        base_hull = MultiPoint(base).convex_hull
        overhang = max(
            0.0, max(float(Point(x, y).distance(base_hull)) for x, y in upper)
        )
    return orientation, tilt, overhang


def _reconcile_entities(
    frames: dict[str, GeometryFrame],
    observations: list[Observation2D],
    frame_data: dict[str, _FrameData],
    transform: _FloorTransform,
) -> tuple[list[Entity3D], list[str]]:
    warnings: list[str] = []
    grouped: dict[str, list[tuple[np.ndarray, Observation2D]]] = {}
    for observation in sorted(
        (item for item in observations if item.label != FLOOR_LABEL),
        key=lambda item: (item.label, item.frame_id, item.observation_id),
    ):
        data = _data_for(observation.frame_id, frames, frame_data)
        points = transform.apply(_select_points(observation, data))
        points = points[points[:, 2] > 0.05]
        points = _remove_outliers(points)
        if len(points) < 20:
            warnings.append(
                f"observation {observation.observation_id} has fewer than 20 object points above floor"
            )
            continue
        grouped.setdefault(observation.label, []).append(
            (_voxel_downsample(points), observation)
        )

    candidates: list[dict[str, object]] = []
    minimum_cluster_support = 5
    for label in sorted(grouped):
        point_groups = grouped[label]
        pooled = np.vstack([points for points, _ in point_groups])
        point_observations = np.concatenate(
            [
                np.full(len(points), observation.observation_id, dtype=object)
                for points, observation in point_groups
            ]
        )
        point_frames = np.concatenate(
            [
                np.full(len(points), observation.frame_id, dtype=object)
                for points, observation in point_groups
            ]
        )
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pooled))
        cluster_labels = np.asarray(
            cloud.cluster_dbscan(
                eps=0.15,
                min_points=minimum_cluster_support,
                print_progress=False,
            )
        )
        for cluster_id in sorted(set(cluster_labels) - {-1}):
            selected = cluster_labels == cluster_id
            cluster_points = pooled[selected]
            hull = MultiPoint(cluster_points[:, :2]).convex_hull
            if not isinstance(hull, Polygon) or hull.area <= 0:
                warnings.append(f"{label} cluster has no nonzero-area polygon footprint")
                continue
            centroid = np.median(cluster_points, axis=0)
            observation_counts = Counter(point_observations[selected])
            frame_counts = Counter(point_frames[selected])
            height = max(0.0, float(np.quantile(cluster_points[:, 2], 0.95)))
            orientation, tilt, overhang = _spatial_state(cluster_points, height)
            candidates.append(
                {
                    "label": label,
                    "centroid": centroid,
                    "observation_ids": sorted(
                        observation_id
                        for observation_id, count in observation_counts.items()
                        if count >= minimum_cluster_support
                    ),
                    "frame_ids": sorted(
                        frame_id
                        for frame_id, count in frame_counts.items()
                        if count >= minimum_cluster_support
                    ),
                    "footprint": [
                        (float(x), float(y)) for x, y in list(hull.exterior.coords)[:-1]
                    ],
                    "height": height,
                    "orientation": orientation,
                    "tilt": tilt,
                    "overhang": overhang,
                }
            )

    candidates.sort(
        key=lambda item: (
            item["label"],
            *(float(value) for value in item["centroid"]),
        )
    )
    label_counts: dict[str, int] = {}
    entities: list[Entity3D] = []
    for candidate in candidates:
        label = str(candidate["label"])
        label_counts[label] = label_counts.get(label, 0) + 1
        entities.append(
            Entity3D(
                entity_id=f"entity-{_slug(label)}-{label_counts[label]:02d}",
                label=label,
                observation_ids=candidate["observation_ids"],
                centroid_xyz=tuple(candidate["centroid"]),
                footprint_xy=candidate["footprint"],
                height_m=candidate["height"],
                evidence_frame_ids=candidate["frame_ids"],
                orientation_deg=candidate["orientation"],
                tilt_deg=candidate["tilt"],
                overhang_m=candidate["overhang"],
            )
        )
    return entities, warnings


def _build_geometry(
    frames: list[GeometryFrame],
    observations: list[Observation2D],
    camera_height_m: float | None,
    *,
    scale_factor_override: float | None = None,
) -> _GeometryResult:
    frames_by_id = {frame.frame_id: frame for frame in frames}
    if len(frames_by_id) != len(frames):
        raise ValueError("geometry frame ids must be unique")
    unknown_frames = sorted(
        {observation.frame_id for observation in observations} - frames_by_id.keys()
    )
    if unknown_frames:
        raise ValueError(f"observations reference unknown frames: {unknown_frames}")

    frame_data: dict[str, _FrameData] = {}
    transform, warnings = _fit_floor(
        frames_by_id,
        frame_data,
        camera_height_m,
        scale_factor_override=scale_factor_override,
    )
    if transform is None:
        return _GeometryResult(transform=None, entities=[], warnings=warnings)
    entities, entity_warnings = _reconcile_entities(
        frames_by_id, observations, frame_data, transform
    )
    return _GeometryResult(
        transform=transform,
        entities=entities,
        warnings=[*warnings, *entity_warnings],
    )
