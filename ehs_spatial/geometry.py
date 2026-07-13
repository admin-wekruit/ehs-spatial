from dataclasses import dataclass
import re

import numpy as np
import open3d as o3d
from PIL import Image
from shapely.geometry import MultiPoint, Polygon

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


def _fit_floor(
    frames: dict[str, GeometryFrame],
    floor_observations: list[Observation2D],
    frame_data: dict[str, _FrameData],
    camera_height_m: float,
) -> tuple[_FloorTransform | None, list[str]]:
    selected_by_frame: list[tuple[str, np.ndarray]] = []
    for observation in sorted(floor_observations, key=lambda item: item.observation_id):
        data = _data_for(observation.frame_id, frames, frame_data)
        points = _select_points(observation, data)
        if len(points):
            selected_by_frame.append((observation.frame_id, points))

    evidence_frames = {frame_id for frame_id, _ in selected_by_frame}
    point_count = sum(len(points) for _, points in selected_by_frame)
    if len(evidence_frames) < 2 or point_count < 200:
        return None, [
            "floor evidence requires at least 2 distinct frames and 200 selected points"
        ]

    floor_points = np.vstack([points for _, points in selected_by_frame])
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(floor_points))
    o3d.utility.random.seed(0)
    _, inlier_indices = cloud.segment_plane(
        distance_threshold=0.03,
        ransac_n=3,
        num_iterations=1000,
    )
    if len(inlier_indices) < 3:
        return None, ["floor plane RANSAC produced fewer than 3 inliers"]

    inliers = floor_points[np.asarray(inlier_indices)]
    center = np.mean(inliers, axis=0)
    _, singular_values, right_vectors = np.linalg.svd(inliers - center)
    if not np.isfinite(singular_values).all() or singular_values[1] <= 1e-12:
        return None, ["floor plane inliers are degenerate"]
    normal = right_vectors[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(np.dot(normal, center))

    camera_centers = np.asarray(
        [np.asarray(frame.camera_to_world)[:3, 3] for frame in frames.values()]
    )
    signed_heights = camera_centers @ normal + offset
    if float(np.median(signed_heights)) < 0:
        normal = -normal
        offset = -offset
        signed_heights = -signed_heights
    positive_heights = signed_heights[signed_heights > 0]
    if not len(positive_heights):
        return None, ["camera-to-floor height is nonpositive"]
    predicted_height = float(np.median(positive_heights))
    if not np.isfinite(predicted_height) or predicted_height <= 0:
        return None, ["camera-to-floor height is nonfinite or nonpositive"]
    scale_factor = camera_height_m / predicted_height
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        return None, ["camera-height scale is nonfinite or nonpositive"]
    scaled_heights = positive_heights * scale_factor
    scaled_mad = float(np.median(np.abs(scaled_heights - np.median(scaled_heights))))
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


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


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
        if len(points) < 20:
            warnings.append(
                f"observation {observation.observation_id} has fewer than 20 object points above floor"
            )
            continue
        grouped.setdefault(observation.label, []).append(
            (_voxel_downsample(points), observation)
        )

    candidates: list[dict[str, object]] = []
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
            cloud.cluster_dbscan(eps=0.15, min_points=5, print_progress=False)
        )
        for cluster_id in sorted(set(cluster_labels) - {-1}):
            selected = cluster_labels == cluster_id
            cluster_points = pooled[selected]
            hull = MultiPoint(cluster_points[:, :2]).convex_hull
            if not isinstance(hull, Polygon) or hull.area <= 0:
                warnings.append(f"{label} cluster has no nonzero-area polygon footprint")
                continue
            centroid = np.median(cluster_points, axis=0)
            candidates.append(
                {
                    "label": label,
                    "centroid": centroid,
                    "observation_ids": sorted(set(point_observations[selected])),
                    "frame_ids": sorted(set(point_frames[selected])),
                    "footprint": [
                        (float(x), float(y)) for x, y in list(hull.exterior.coords)[:-1]
                    ],
                    "height": float(
                        np.quantile(cluster_points[:, 2], 0.95)
                        - np.quantile(cluster_points[:, 2], 0.05)
                    ),
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
            )
        )
    return entities, warnings


def _build_geometry(
    frames: list[GeometryFrame],
    observations: list[Observation2D],
    camera_height_m: float,
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
        [item for item in observations if item.label == FLOOR_LABEL],
        frame_data,
        camera_height_m,
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
