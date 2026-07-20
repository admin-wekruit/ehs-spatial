"""Spatial-state primitive tests.

Conventions under test:
- orientation_deg: undirected yaw of the XY footprint's first principal axis,
  degrees in [0, 180).
- tilt_deg: angle between the cluster's 3D first principal axis and vertical
  +Z, degrees in [0, 90]. A ladder leaning 75 deg from horizontal is 15 deg
  from vertical, so tilt_deg ~= 15.
- overhang_m: max XY distance of upper-band points (z > 60% of height)
  outside the base-band (z < 40% of height) convex hull, metres.
"""

import numpy as np

from ehs_spatial.contracts import Entity3D
from ehs_spatial.geometry import _spatial_state


def _grid(x_extent: float, y_extent: float, angle_deg: float = 0.0) -> np.ndarray:
    x, y, z = np.meshgrid(
        np.linspace(0.0, x_extent, 41),
        np.linspace(0.0, y_extent, 9),
        np.linspace(0.1, 0.5, 5),
        indexing="ij",
    )
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    angle = np.deg2rad(angle_deg)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return points @ rotation.T


def test_orientation_recovers_45_degree_rotated_elongated_box():
    orientation, _, _ = _spatial_state(_grid(2.0, 0.4, angle_deg=45.0), height=0.5)
    assert orientation is not None
    assert abs(orientation - 45.0) <= 3.0


def test_tilt_of_ladder_leaning_75_from_horizontal_is_15_from_vertical():
    # tilt_deg is measured from vertical +Z: a ladder 75 deg above the
    # horizontal has its long axis 15 deg off vertical, so tilt_deg ~= 15.
    rng = np.random.default_rng(7)
    direction = np.array([np.sin(np.deg2rad(15.0)), 0.0, np.cos(np.deg2rad(15.0))])
    t = np.linspace(0.0, 3.0, 80)
    points = np.outer(t, direction) + rng.normal(scale=0.02, size=(80, 3))
    _, tilt, _ = _spatial_state(points, height=float(points[:, 2].max()))
    assert tilt is not None
    assert abs(tilt - 15.0) <= 3.0


def test_overhang_of_mushroom_cap_beyond_narrow_base():
    # Stem radius 0.2 m, cap radius 0.5 m -> designed protrusion 0.3 m.
    angles = np.linspace(0.0, 2.0 * np.pi, 33)[:-1]
    ring = np.column_stack([np.cos(angles), np.sin(angles)])
    stem = np.vstack(
        [
            np.column_stack([0.2 * ring, np.full(len(ring), z)])
            for z in np.linspace(0.05, 0.95, 6)
        ]
    )
    cap = np.vstack(
        [
            np.column_stack([radius * ring, np.full(len(ring), z)])
            for z in (1.0, 1.1, 1.2)
            for radius in (0.5, 0.3)
        ]
    )
    _, _, overhang = _spatial_state(np.vstack([stem, cap]), height=1.2)
    assert overhang is not None
    assert abs(overhang - 0.3) <= 0.03


def test_isotropic_blob_has_no_orientation():
    # An equal-sampled square grid has an exactly isotropic XY covariance
    # (eigenvalue ratio 1 < 1.2 gate).
    x, y, z = np.meshgrid(
        np.linspace(0.0, 1.0, 15),
        np.linspace(0.0, 1.0, 15),
        np.linspace(0.1, 0.5, 5),
        indexing="ij",
    )
    blob = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    orientation, _, _ = _spatial_state(blob, height=0.5)
    assert orientation is None


def test_too_few_points_yield_none_fields_without_crashing():
    for points in (
        np.empty((0, 3)),
        np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]),
    ):
        assert _spatial_state(points, height=1.0) == (None, None, None)


def test_entity3d_constructs_without_spatial_state_fields():
    entity = Entity3D(
        entity_id="entity-ladder-01",
        label="ladder",
        observation_ids=[],
        centroid_xyz=(0.0, 0.0, 0.5),
        footprint_xy=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        height_m=1.0,
        evidence_frame_ids=[],
    )
    assert entity.orientation_deg is None
    assert entity.tilt_deg is None
    assert entity.overhang_m is None
