"""E3/E4 rule-semantics scenario matrix.

E4 scenarios exercise the REAL rule path (_assess_clearance) with hand-built
Entity3D fixtures so boundary distances are exact — the full synthetic-scene
pipeline adds voxel/cluster noise that would swamp a 2 cm boundary triplet.
E3 scenarios exercise _spatial_state on analytic clusters and assert the
Entity3D fields recover designed values under the settled conventions
(orientation_deg [0, 180) undirected yaw; tilt_deg [0, 90] from +Z;
overhang_m max XY protrusion of the upper band beyond the base-band hull).
"""

import numpy as np
import pytest

from ehs_spatial.contracts import AssessmentStatus, Criterion, Entity3D
from ehs_spatial.geometry import _spatial_state
from ehs_spatial.rules import _assess_clearance


def _fence() -> Entity3D:
    # 2x2 m square fence: edges on x=0, x=2, y=0, y=2.
    return Entity3D(
        entity_id="entity-safety-fence-01",
        label="safety fence",
        observation_ids=[],
        centroid_xyz=(1.0, 1.0, 0.5),
        footprint_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
        height_m=1.0,
        evidence_frame_ids=["frame-1", "frame-2", "frame-3"],
    )


def _movable(
    footprint: list[tuple[float, float]],
    entity_id: str = "entity-pallet-01",
    label: str = "pallet",
    **fields,
) -> Entity3D:
    xs = [x for x, _ in footprint]
    ys = [y for _, y in footprint]
    return Entity3D(
        entity_id=entity_id,
        label=label,
        observation_ids=[],
        centroid_xyz=(sum(xs) / len(xs), sum(ys) / len(ys), 0.25),
        footprint_xy=footprint,
        height_m=0.5,
        evidence_frame_ids=["frame-1", "frame-2"],
        **fields,
    )


def _square(gap: float) -> list[tuple[float, float]]:
    # 0.2 m square left of the fence's x=0 edge; nearest-edge gap is exact
    # in float (distance is gap - 0, no summation rounding at the boundary).
    return [(-0.2 - gap, 0.9), (-gap, 0.9), (-gap, 1.1), (-0.2 - gap, 1.1)]


def _rotated_rect(
    center: tuple[float, float],
    half_length: float,
    half_width: float,
    angle_deg: float,
) -> list[tuple[float, float]]:
    angle = np.deg2rad(angle_deg)
    u = np.array([np.cos(angle), np.sin(angle)])
    v = np.array([-np.sin(angle), np.cos(angle)])
    c = np.asarray(center)
    return [
        (float(x), float(y))
        for x, y in (
            c + half_length * u + half_width * v,
            c + half_length * u - half_width * v,
            c - half_length * u - half_width * v,
            c - half_length * u + half_width * v,
        )
    ]


# --- E4: clearance rule semantics -----------------------------------------


@pytest.mark.parametrize(
    ("gap", "frames", "expected"),
    [
        # Multi-view band is ±0.20 m around the 0.6 m threshold: anything
        # inside [0.40, 0.80] is within the measured noise floor and must
        # say NEEDS_REVIEW rather than flip a verdict on noise.
        (0.39, 4, AssessmentStatus.FAIL),
        (0.58, 4, AssessmentStatus.NEEDS_REVIEW),
        (0.60, 4, AssessmentStatus.NEEDS_REVIEW),
        (0.62, 4, AssessmentStatus.NEEDS_REVIEW),
        (0.81, 4, AssessmentStatus.PASS),
        # Mono band is ±0.35 m: wider, because single-view accuracy is worse.
        (0.24, 1, AssessmentStatus.FAIL),
        (0.81, 1, AssessmentStatus.NEEDS_REVIEW),
        (0.96, 1, AssessmentStatus.PASS),
    ],
)
def test_boundary_bands_around_minimum_clearance(gap, frames, expected):
    fence = _fence()
    movable = _movable(_square(gap))
    if frames == 1:
        fence = fence.model_copy(update={"evidence_frame_ids": ["frame-1"]})
        movable = movable.model_copy(update={"evidence_frame_ids": ["frame-1"]})
    result = _assess_clearance(
        [fence, movable], Criterion(), capture_frame_count=frames
    )
    assert result.assessment.status == expected
    assert result.assessment.approximate_distance_m == pytest.approx(gap)
    assert result.assessment.distance_error_budget_m == (
        0.20 if frames >= 2 else 0.35
    )


def test_straddling_movable_fails_with_zero_distance():
    straddling = _movable([(1.9, 0.9), (2.1, 0.9), (2.1, 1.1), (1.9, 1.1)])
    result = _assess_clearance([_fence(), straddling], Criterion())
    assert result.assessment.status == AssessmentStatus.FAIL
    assert result.assessment.approximate_distance_m == 0.0


def test_rotated_movable_judged_by_nearest_corner_not_centroid():
    # 1.6 x 0.2 m box rotated 45 deg: nearest corner 0.45 m from the x=2
    # fence edge, centroid 0.45 + 0.9*sqrt(2)/2 ~= 1.086 m away. Centroid
    # semantics would PASS; nearest-point semantics must FAIL.
    corner_gap = 0.30
    center_x = 2.0 + corner_gap + 0.9 * np.sqrt(2.0) / 2.0
    ladder = _movable(
        _rotated_rect((center_x, 1.0), 0.8, 0.1, 45.0),
        entity_id="entity-step-ladder-01",
        label="step ladder",
    )
    assert center_x - 2.0 > 0.7  # centroid gap is comfortably above minimum
    result = _assess_clearance([_fence(), ladder], Criterion())
    assert result.assessment.status == AssessmentStatus.FAIL
    assert result.assessment.approximate_distance_m == pytest.approx(
        corner_gap, abs=0.03
    )


def test_nearest_movable_governs_over_rotated_distractor_centroid():
    # Plain pallet: polygon gap 0.70 (PASS range), centroid gap 0.80.
    # Rotated ladder: corner gap 0.35 but centroid gap ~= 0.986 — a
    # centroid-based selector would pick the pallet and PASS; nearest-polygon
    # selection must pick the ladder and FAIL.
    pallet = _movable([(2.70, 0.9), (2.90, 0.9), (2.90, 1.1), (2.70, 1.1)])
    ladder_center_x = 2.0 + 0.35 + 0.9 * np.sqrt(2.0) / 2.0
    ladder = _movable(
        _rotated_rect((ladder_center_x, 1.5), 0.8, 0.1, 45.0),
        entity_id="entity-step-ladder-01",
        label="step ladder",
    )
    result = _assess_clearance([_fence(), pallet, ladder], Criterion())
    assert result.selected_entity_id == "entity-step-ladder-01"
    assert result.assessment.status == AssessmentStatus.FAIL
    assert result.assessment.approximate_distance_m == pytest.approx(0.35, abs=0.03)


def test_overhang_toward_fence_is_fact_only_and_does_not_fail_verdict():
    # Recorded semantic gap awaiting owner decision: the base polygon clears
    # the fence by 0.9 m, but a 0.5 m overhang toward the fence leaves only
    # ~0.4 m of true clearance. The owner has NOT approved overhang affecting
    # verdicts, so the current semantics is PASS on base-footprint distance.
    overhanging = _movable(
        [(2.9, 0.9), (3.1, 0.9), (3.1, 1.1), (2.9, 1.1)],
        overhang_m=0.5,
    )
    result = _assess_clearance([_fence(), overhanging], Criterion())
    assert result.assessment.status == AssessmentStatus.PASS
    assert result.assessment.approximate_distance_m == pytest.approx(0.9)


# --- E3: spatial-state accuracy on analytic clusters ----------------------


def _entity_with_state(points: np.ndarray, height: float) -> Entity3D:
    orientation, tilt, overhang = _spatial_state(points, height)
    return Entity3D(
        entity_id="entity-cluster-01",
        label="step ladder",
        observation_ids=[],
        centroid_xyz=tuple(np.median(points, axis=0)),
        footprint_xy=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        height_m=height,
        evidence_frame_ids=[],
        orientation_deg=orientation,
        tilt_deg=tilt,
        overhang_m=overhang,
    )


def test_leaning_ladder_recovers_orientation_tilt_and_overhang():
    # 3 m ladder leaning 20 deg from vertical, yawed 30 deg in XY: designed
    # orientation 30, tilt 20, overhang (3 - 1.2) * sin(20 deg) ~= 0.616 m
    # (upper band starts at 60% of height, base band ends at 40%).
    tilt = np.deg2rad(20.0)
    yaw = np.deg2rad(30.0)
    direction = np.array(
        [np.sin(tilt) * np.cos(yaw), np.sin(tilt) * np.sin(yaw), np.cos(tilt)]
    )
    points = np.outer(np.linspace(0.0, 3.0, 80), direction)
    entity = _entity_with_state(points, height=float(points[:, 2].max()))
    assert entity.orientation_deg == pytest.approx(30.0, abs=3.0)
    assert entity.tilt_deg == pytest.approx(20.0, abs=3.0)
    assert entity.overhang_m == pytest.approx(1.8 * np.sin(tilt), abs=0.03)


def test_rotated_box_recovers_orientation_with_horizontal_tilt_and_no_overhang():
    # 2.0 x 0.4 x 0.5 m box yawed 60 deg: orientation 60; dominant axis is
    # horizontal so tilt reads ~90 by convention; uniform vertical footprint
    # means zero overhang.
    x, y, z = np.meshgrid(
        np.linspace(0.0, 2.0, 41),
        np.linspace(0.0, 0.4, 9),
        np.linspace(0.1, 0.5, 5),
        indexing="ij",
    )
    box = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    angle = np.deg2rad(60.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    entity = _entity_with_state(box @ rotation.T, height=0.5)
    assert entity.orientation_deg == pytest.approx(60.0, abs=3.0)
    assert entity.tilt_deg == pytest.approx(90.0, abs=1.0)
    assert entity.overhang_m == 0.0
