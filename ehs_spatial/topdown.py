from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from .contracts import Assessment, Entity3D
from .rules import FENCE_LABEL, MOVABLE_LABELS


def _valid_movables(entities: list[Entity3D]) -> list[Entity3D]:
    return [
        entity
        for entity in entities
        if entity.label in MOVABLE_LABELS
        and len(set(entity.evidence_frame_ids)) >= 2
        and Polygon(entity.footprint_xy).area >= 0.0025
        and entity.height_m >= 0.05
    ]


def _render_topdown(
    path: str | Path,
    entities: list[Entity3D],
    fence_polygon: list[tuple[float, float]],
    selected_entity_id: str | None,
    assessment: Assessment,
) -> None:
    width = height = 640
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    status_text = f"Status: {assessment.status.value}"
    if assessment.approximate_distance_m is not None:
        status_text += f"  clearance: {assessment.approximate_distance_m:.2f} m"
    draw.text((24, 20), status_text, fill="black")

    fences = [fence_polygon] if fence_polygon else [
        entity.footprint_xy for entity in entities if entity.label == FENCE_LABEL
    ]
    movables = _valid_movables(entities)
    # The display filter must never hide the entity the rule ruled on: the
    # rule engine's gates scale with capture size, this cosmetic one does not.
    if selected_entity_id and all(
        entity.entity_id != selected_entity_id for entity in movables
    ):
        movables += [
            entity for entity in entities if entity.entity_id == selected_entity_id
        ]
    polygons = [*fences, *(entity.footprint_xy for entity in movables)]
    points = [point for polygon in polygons for point in polygon]
    if points:
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        padding = 48
        top = 72
        drawable_width = width - 2 * padding
        drawable_height = height - top - padding
        scale = min(drawable_width / span_x, drawable_height / span_y)
        x_offset = padding + (drawable_width - span_x * scale) / 2
        y_offset = top + (drawable_height - span_y * scale) / 2

        def pixel(point: tuple[float, float]) -> tuple[float, float]:
            return (
                x_offset + (point[0] - min_x) * scale,
                y_offset + (max_y - point[1]) * scale,
            )

        for polygon in fences:
            draw.polygon(
                [pixel(point) for point in polygon],
                fill="#dcecff",
                outline="#1769aa",
                width=4,
            )
        for entity in movables:
            selected = entity.entity_id == selected_entity_id
            draw.polygon(
                [pixel(point) for point in entity.footprint_xy],
                fill="#ff8a65" if selected else "#ffd180",
                outline="#c62828" if selected else "#ef6c00",
                width=4 if selected else 2,
            )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
