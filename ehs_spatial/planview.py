"""Measured plan view and semantic point cloud run artifacts.

Both are free derived views over the run's cached evidence (reconstructed
geometry + segmentation masks): a CAD-style plan with a metre grid and
dimension callouts, and a label-tinted point cloud any viewer can rotate.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from .contracts import Assessment, Entity3D, GeometryFrame, Observation2D
from .geometry import _FloorTransform, _load_frame, _select_points
from .rules import FENCE_LABEL


_CANVAS = 760
_MARGIN = 56


def _render_plan_view(
    path: str | Path,
    entities: list[Entity3D],
    fence_polygon: list[tuple[float, float]],
    selected_entity_id: str | None,
    assessment: Assessment,
) -> None:
    """CAD-style measured plan: metre grid, camera at origin, labelled hulls,
    and a dimension callout for the ruled clearance."""
    image = Image.new("RGB", (_CANVAS, _CANVAS), "white")
    draw = ImageDraw.Draw(image)

    polygons: list[tuple[float, float]] = [(0.0, 0.0)]
    for entity in entities:
        polygons.extend(entity.footprint_xy)
    polygons.extend(fence_polygon)
    xs = [x for x, _ in polygons]
    ys = [y for _, y in polygons]
    min_x, max_x = min(xs) - 0.8, max(xs) + 0.8
    min_y, max_y = min(ys) - 0.8, max(ys) + 0.8
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    scale = (_CANVAS - 2 * _MARGIN) / span

    def pixel(point: tuple[float, float]) -> tuple[float, float]:
        return (
            _MARGIN + (point[0] - min_x) * scale,
            _CANVAS - _MARGIN - (point[1] - min_y) * scale,
        )

    for gx in np.arange(np.floor(min_x), np.ceil(max_x) + 1):
        draw.line([pixel((gx, min_y)), pixel((gx, max_y))], fill="#e4e4e4")
    for gy in np.arange(np.floor(min_y), np.ceil(max_y) + 1):
        draw.line([pixel((min_x, gy)), pixel((max_x, gy))], fill="#e4e4e4")
    draw.text(
        (_MARGIN, _CANVAS - _MARGIN + 8),
        "grid: 1 m  |  camera at origin",
        fill="#8a8a8a",
    )

    if fence_polygon:
        draw.polygon(
            [pixel(point) for point in fence_polygon],
            fill="#dcecff",
            outline="#1769aa",
            width=3,
        )

    selected_entity = None
    for entity in entities:
        if entity.label == FENCE_LABEL and fence_polygon:
            continue
        selected = entity.entity_id == selected_entity_id
        if selected:
            selected_entity = entity
        outline = "#c62828" if selected else "#7d7d7d"
        fill = "#ffcdb8" if selected else "#efefef"
        draw.polygon(
            [pixel(point) for point in entity.footprint_xy],
            fill=fill,
            outline=outline,
            width=3 if selected else 1,
        )
        cx = float(np.mean([x for x, _ in entity.footprint_xy]))
        cy = float(np.mean([y for _, y in entity.footprint_xy]))
        caption = f"{entity.label}  H={entity.height_m:.2f}m"
        if entity.tilt_deg is not None:
            caption += f"  tilt {entity.tilt_deg:.0f}\N{DEGREE SIGN}"
        draw.text(pixel((cx, cy)), caption, fill=outline, anchor="mm")

    camera_px = pixel((0.0, 0.0))
    draw.ellipse(
        [camera_px[0] - 6, camera_px[1] - 6, camera_px[0] + 6, camera_px[1] + 6],
        fill="#c62828",
    )
    draw.text(
        (camera_px[0] + 9, camera_px[1] - 6), "camera (0,0)", fill="#c62828"
    )

    if (
        selected_entity is not None
        and fence_polygon
        and assessment.approximate_distance_m is not None
    ):
        fence_poly = Polygon(fence_polygon)
        selected_poly = Polygon(selected_entity.footprint_xy)
        if not selected_poly.intersects(fence_poly):
            a, b = nearest_points(selected_poly, fence_poly)
            draw.line(
                [pixel((a.x, a.y)), pixel((b.x, b.y))], fill="#2e7d46", width=3
            )
            mid = pixel(((a.x + b.x) / 2, (a.y + b.y) / 2))
            draw.text(
                (mid[0] + 6, mid[1] - 14),
                f"{assessment.approximate_distance_m:.2f} m",
                fill="#2e7d46",
            )
        camera_distance = Point(0.0, 0.0).distance(selected_poly)
        draw.text(
            (camera_px[0] + 9, camera_px[1] + 8),
            f"to {selected_entity.label}: {camera_distance:.2f} m",
            fill="#c62828",
        )

    status = f"{assessment.status.value}"
    if assessment.approximate_distance_m is not None:
        status += f"  clearance {assessment.approximate_distance_m:.2f} m"
    draw.text((_MARGIN, 20), status, fill="black")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")


_LABEL_TINTS = (
    (242, 26, 166),
    (255, 140, 0),
    (0, 176, 255),
    (60, 220, 60),
    (240, 200, 0),
    (170, 90, 255),
)


def _write_semantic_ply(
    path: str | Path,
    frames: list[GeometryFrame],
    observations: list[Observation2D],
    transform: _FloorTransform,
) -> None:
    """Full-scene point cloud in the floor frame (z = height above floor),
    with points belonging to a segmented observation tinted per label."""
    labels = sorted({obs.label for obs in observations})
    tint_by_label = {
        label: _LABEL_TINTS[index % len(_LABEL_TINTS)]
        for index, label in enumerate(labels)
    }

    chunks: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for frame in frames:
        data = _load_frame(frame)
        finite = data.valid & np.isfinite(data.points).all(axis=2)
        with Image.open(frame.canonical_image_path) as canonical:
            rgb = np.asarray(canonical.convert("RGB"))
        if rgb.shape[:2] != data.shape:
            rgb = np.asarray(
                Image.fromarray(rgb).resize((data.shape[1], data.shape[0]))
            )
        frame_colors = rgb.astype(np.float32)
        for obs in observations:
            if obs.frame_id != frame.frame_id or obs.mask_path is None:
                continue
            try:
                points = _select_points(obs, data)
            except ValueError:
                continue
            if not len(points):
                continue
            with Image.open(obs.mask_path) as mask_image:
                mask = np.asarray(mask_image).astype(bool)
            frame_colors[mask & finite] = tint_by_label[obs.label]
        chunks.append(transform.apply(data.points[finite]))
        colors.append(frame_colors[finite])

    points = np.vstack(chunks).astype("<f4") if chunks else np.zeros((0, 3), "<f4")
    rgb = (
        np.vstack(colors).clip(0, 255).astype(np.uint8)
        if colors
        else np.zeros((0, 3), np.uint8)
    )
    record = np.zeros(len(points), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    record["xyz"] = points
    record["rgb"] = rgb
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        stream.write(header)
        stream.write(record.tobytes())


__all__ = ["_render_plan_view", "_write_semantic_ply"]
