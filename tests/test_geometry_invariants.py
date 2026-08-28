"""Geometric invariants over the real-photo runs' inventories.

Every perspective regression this project shipped (background-anchored
planes, flattened gate recession, scattered rail sections, half-metre
"thin" boards) violated one of the physical priors below while unit tests
stayed green. These tests read the actual run inventories and assert the
priors, so a geometry change that breaks perspective fails HERE instead of
in the owner's screenshots. Skipped when the runs are absent (CI without
the data directory).
"""

import json
from pathlib import Path

import numpy as np
import pytest

RUNS = [
    Path("runs") / run_id
    for run_id in ("real-clean-01", "real-clean-02", "real-clean-03")
]

runs_present = pytest.mark.skipif(
    not all((run / "inventory" / "inventory.json").exists() for run in RUNS),
    reason="real-clean runs not present",
)


def _inventory(run: Path) -> dict:
    return json.loads((run / "inventory" / "inventory.json").read_text())


def _rect_angle_deg(rect) -> float:
    r = np.asarray(rect, float)
    edge1, edge2 = r[1] - r[0], r[2] - r[1]
    if np.linalg.norm(edge1) < np.linalg.norm(edge2):
        edge1 = edge2
    return float(np.degrees(np.arctan2(edge1[1], edge1[0]))) % 180.0


def _rect_short_side(rect) -> float:
    r = np.asarray(rect, float)
    return min(
        float(np.linalg.norm(r[1] - r[0])), float(np.linalg.norm(r[2] - r[1]))
    )


def _guard_chains(inventory: dict) -> list[list[dict]]:
    """Members of each aligned guard line, keyed by chain id."""
    groups: dict[int, list[dict]] = {}
    for obj in inventory["objects"]:
        if obj.get("footprint_method") != "guard-line":
            continue
        if obj.get("guard_chain") is None:
            continue
        groups.setdefault(obj["guard_chain"], []).append(obj)
    return [members for members in groups.values() if len(members) >= 2]


@runs_present
@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_guard_line_members_are_collinear(run):
    for chain in _guard_chains(_inventory(run)):
        cents = np.array([m["centroid_xy"] for m in chain], float)
        angles = [_rect_angle_deg(m["rect_snapped"]) for m in chain]
        assert max(angles) - min(angles) < 2.0, (
            f"{run.name}: guard-line members not parallel: {angles}"
        )
        theta = np.radians(angles[0])
        normal = np.array([-np.sin(theta), np.cos(theta)])
        offsets = (cents - cents.mean(axis=0)) @ normal
        assert float(np.abs(offsets).max()) < 0.30, (
            f"{run.name}: guard-line members off the shared line: {offsets}"
        )


@runs_present
@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_guard_lines_stay_manhattan(run):
    inventory = _inventory(run)
    theta = inventory.get("manhattan_theta_deg")
    if theta is None:
        pytest.skip("no manhattan axis")
    for chain in _guard_chains(inventory):
        if not chain[0].get("guard_axis_snapped"):
            # the pipeline deliberately kept a free direction (contact-edge
            # evidence >20 deg off both axes); collinearity/order/thickness
            # still apply, axis adherence does not
            continue
        angle = _rect_angle_deg(chain[0]["rect_snapped"])
        off_axis = min(
            abs((angle - (theta % 180) + 90) % 180 - 90),
            abs((angle - ((theta + 90) % 180) + 90) % 180 - 90),
        )
        assert off_axis < 6.0, (
            f"{run.name}: aligned guard line {angle:.1f}° is {off_axis:.1f}° "
            f"off both Manhattan axes (theta {theta})"
        )


@runs_present
@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_guard_line_order_matches_image_order(run):
    """The photo's left-to-right section order must survive projection —
    the flattened-recession bug scrambled along-line positions."""
    for chain in _guard_chains(_inventory(run)):
        members = sorted(
            chain, key=lambda m: (m["image_bbox"][0] + m["image_bbox"][2]) / 2
        )
        cents = np.array([m["centroid_xy"] for m in members], float)
        theta = np.radians(_rect_angle_deg(members[0]["rect_snapped"]))
        direction = np.array([np.cos(theta), np.sin(theta)])
        along = cents @ direction
        increasing = all(along[i] < along[i + 1] for i in range(len(along) - 1))
        decreasing = all(along[i] > along[i + 1] for i in range(len(along) - 1))
        assert increasing or decreasing, (
            f"{run.name}: along-line order scrambled vs image order: {along}"
        )


@runs_present
@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_thin_structures_stay_thin(run):
    """A fence-family board past ~35 cm of plan thickness is depth smear,
    not the object (the 0.5 m 'thin' rail bug)."""
    for obj in _inventory(run)["objects"]:
        if obj.get("footprint_method") not in ("contact-edge", "guard-line"):
            continue
        if "fence" not in obj["label"]:
            continue
        thickness = _rect_short_side(obj["rect_snapped"])
        assert thickness <= 0.35, (
            f"{run.name}: {obj['label']} inst {obj.get('instance')} plan "
            f"thickness {thickness:.2f} m"
        )


# per-run ceilings for scored fence entries (fraction of image height).
# 01/03: verified-contact scenes, pinned tight. 02: occlusion-heavy
# edge-on scene — 0.25 is the honest current state and the open work item;
# tightening it requires better front-section mask bottoms or a second
# viewpoint, not another threshold.
REPROJECTION_CEILING = {
    "real-clean-01": 0.10,
    "real-clean-02": 0.25,
    "real-clean-03": 0.10,
}


@runs_present
@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_reprojection_scores_within_ceiling(run):
    path = run / "inventory" / "reprojection.json"
    assert path.exists(), f"{run.name}: reprojection not run"
    scores = json.loads(path.read_text())["scores"]
    scored = [s for s in scores if s.get("mean_dv_frac") is not None]
    assert scored, f"{run.name}: nothing scored"
    worst = max(s["mean_dv_frac"] for s in scored)
    assert worst <= REPROJECTION_CEILING[run.name], (
        f"{run.name}: worst reprojection {worst} exceeds ceiling"
    )


@runs_present
def test_real_clean_01_gate_recedes_and_l_is_orthogonal():
    """Pinned facts about real-clean-01 the owner verified in the photo:
    the gate line recedes toward camera-right (right section nearest), and
    the tall panel meets it at a right angle."""
    inventory = _inventory(RUNS[0])
    rails = [
        obj
        for obj in inventory["objects"]
        if obj.get("footprint_method") == "guard-line"
        and "fence" in obj["label"]
    ]
    assert len(rails) >= 3
    rails.sort(key=lambda m: (m["image_bbox"][0] + m["image_bbox"][2]) / 2)
    depths = [float(np.hypot(*m["centroid_xy"])) for m in rails]
    assert all(depths[i] > depths[i + 1] for i in range(len(depths) - 1)), (
        f"gate must recede toward camera-right, got depths {depths}"
    )
    tall = [
        obj
        for obj in inventory["objects"]
        if obj.get("footprint_method") == "contact-edge"
        and "fence" in obj["label"]
        and obj["height_m"] > 1.5
    ]
    assert tall, "tall panel missing"
    angle_gap = abs(
        _rect_angle_deg(rails[0]["rect_snapped"])
        - _rect_angle_deg(tall[0]["rect_snapped"])
    )
    assert min(angle_gap, 180 - angle_gap) > 84.0, (
        f"L-shaped guard arms must be orthogonal, gap {angle_gap:.1f}°"
    )


@runs_present
@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_enumeration_never_silently_drops(run):
    """The vocabulary guarantee: every phrase the VLM enumerated either
    has a measured inventory instance or an explicit unresolved record —
    a branded parts container once vanished between enumeration and
    segmentation with no trace."""
    phrases_path = run / "inventory" / "phrases.json"
    if not phrases_path.exists():
        pytest.skip("no enumeration for this run")
    phrases = json.loads(phrases_path.read_text())
    labels = {o["label"] for o in _inventory(run)["objects"]}
    unresolved_path = run / "inventory" / "unresolved.json"
    noted = (
        {u["phrase"] for u in json.loads(unresolved_path.read_text())}
        if unresolved_path.exists()
        else set()
    )
    silently_dropped = [
        ph for ph in phrases if ph not in labels and ph not in noted
    ]
    assert not silently_dropped, (
        f"{run.name}: enumerated but neither measured nor recorded "
        f"unresolved: {silently_dropped}"
    )
