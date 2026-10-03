"""Small runnable recovery, held-out evidence, shared-angle and GLB checks.

python scripts/check_workcell_guard_joint.py
Synthetic images are exact pinhole renders; this cannot establish photo accuracy.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from workcell_guard_joint import _features, _fit_board, _geometry, _leave_view, _mesh, _record, _solve, _shared, _stats, _triangulate
from workcell_photo_geometry import _intersect
from workcell_photo_objects import _project


def scene(theta=137.):
    truth = np.r_[np.zeros(6), np.deg2rad(theta)]
    _, axis, directions, normals = _geometry(truth)
    K = np.array([[165., 0, 128], [0, 165., 128], [0, 0, 1]])
    frames, masks = {}, {}
    for photo, center in enumerate(([3, 4, 1.5], [-2, 5, 1.], [1, 5, -1.3], [2, 6, 2.6]), 1):
        z = -np.asarray(center, float); z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1]); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        pose = np.eye(4); pose[:3, :3] = np.column_stack([x, y, z]); pose[:3, 3] = center
        rgb = np.zeros((256, 256, 3), np.uint8); rgb[:] = [230, 175, 20]
        frame = {'K': K.copy(), 'pose': pose, 'rgb': rgb, 'analysisRgb': rgb, 'C': np.eye(3)}
        mask = np.zeros((256, 256), np.uint8); points = np.full((256, 256, 3), np.nan)
        zbuffer = np.full((256, 256), np.inf)
        for panel in range(2):
            polygon = np.asarray([directions[panel] * u + axis * v for u, v in ((0, -.6), (1.1, -.6), (1.1, .6), (0, .6))])
            uv, _ = _project(polygon, frame)
            part = np.zeros_like(mask); cv2.fillConvexPoly(part, np.rint(uv).astype(np.int32), 1)
            yy, xx = np.nonzero(part)
            xyz = _intersect(np.c_[xx, yy], K, pose, normals[panel], 0)
            _, depth = _project(xyz, frame)
            visible = depth < zbuffer[yy, xx]
            yy, xx, xyz, depth = yy[visible], xx[visible], xyz[visible], depth[visible]
            points[yy, xx] = xyz; zbuffer[yy, xx] = depth; mask[yy, xx] = 1
        frame['points'] = points; frames[photo] = frame; masks[photo] = mask.astype(bool)
    rng = np.random.default_rng(72)
    tracks, assignments = [], []
    for panel in range(2):
        for index in range(16):
            xyz = rng.uniform(.1, 1.) * directions[panel] + rng.uniform(-.5, .5) * axis
            observations = []
            for photo, frame in frames.items():
                uv, _ = _project([xyz], frame)
                observations.append({'photo': photo, 'uv': (uv[0] + rng.normal(0, .025, 2)).tolist()})
            tri = _triangulate(observations, frames)
            tracks.append({'xyz': tri[0], 'observations': observations, 'parallaxDeg': tri[2]})
            assignments.append(panel)
    return truth, frames, {'side': 'left', 'masks': masks, 'associations': []}, tracks, np.asarray(assignments)


truth, frames, board, tracks, assignments = scene()
training = [t for index, t in enumerate(tracks) if index % 4]
held = [t for index, t in enumerate(tracks) if not index % 4]
a_train = np.asarray([a for index, a in enumerate(assignments) if index % 4])
a_held = np.asarray([a for index, a in enumerate(assignments) if not index % 4])
initial = truth + np.r_[.02, -.03, .01, .03, -.04, .02, np.deg2rad(-9.)]
fitted, diagnostic = _solve(initial, training, a_train, frames, 2.)
assert diagnostic['converged'] and abs(np.rad2deg(fitted[6] - truth[6])) < .3
validation = _stats(fitted, held, a_held, frames)
assert validation['medianPx'] < .15 and validation['p95Px'] < .3
corrupted = [{**t, 'observations': [{**o, 'uv': (np.asarray(o['uv']) + ([8, 0] if o['photo'] == 4 else [0, 0])).tolist()} for o in t['observations']]} for t in held]
assert _stats(fitted, corrupted, a_held, frames)['p95Px'] > 6, 'Held-out mismatch must remain visible'
automatic = _fit_board(board, tracks, frames)
assert abs(np.rad2deg(automatic['parameters'][6] - truth[6])) < .3
assert len([r for r in automatic['leaveViews'] if r['status'] == 'fit']) == 4
assert max(abs(r['angleDeg'] - 137.) for r in automatic['leaveViews']) < .4
subset_training = training[:6] + training[-6:]
isolated = _leave_view(board, subset_training, held, frames, 4)
changed_frames = {**frames, 4: {**frames[4], 'points': frames[4]['points'] + [800, -900, 200]}}
def change_excluded(rows):
    return [{**t, 'xyz': np.full(3, 999.), 'observations': [
        {**o, 'uv': (np.asarray(o['uv']) + ([12, 0] if o['photo'] == 4 else [0, 0])).tolist()}
        for o in t['observations']]} for t in rows]
isolated_changed = _leave_view(board, change_excluded(subset_training), change_excluded(held), changed_frames, 4)
assert isolated['status'] == isolated_changed['status'] == 'fit'
assert abs(isolated['angleDeg'] - isolated_changed['angleDeg']) < 1e-9, 'Excluded observations, depth and all-view triangulation must not affect subset fitting'
assert isolated_changed['excludedViewResidual']['medianPx'] > 10
with patch('workcell_guard_joint._solve', return_value=(truth, {'converged': False})):
    failed_subset = _leave_view(board, subset_training, held, frames, 4)
assert failed_subset['status'] == 'unsupported' and failed_subset['angleDeg'] is None
# Extra cameras/masks with zero feature observations cannot manufacture two
# leave-view fits. The only observed pair becomes untriangulatable if either
# contributing view is removed, even though all four camera records exist.
two_view_tracks = []
for track in tracks:
    observations = [o for o in track['observations'] if o['photo'] in (3, 4)]
    tri = _triangulate(observations, frames)
    two_view_tracks.append({**track, 'observations': observations, 'xyz': tri[0], 'parallaxDeg': tri[2]})
two_view_fit = _fit_board(board, two_view_tracks, frames)
assert not any(row['status'] == 'fit' for row in two_view_fit['leaveViews'])
assert all(row['excludedTrainingTracks'] == 0 for row in two_view_fit['leaveViews'] if row['photo'] in (1, 2))
assert all(row['trainingTracksAfterRetriangulation'] == 0 for row in two_view_fit['leaveViews'] if row['photo'] in (3, 4))
# The plane initializer owns its >=16 support gate. Nineteen real RGB training
# tracks must be tried even when an unrelated depth seed would fail to bend.
small_tracks = []
_, axis, directions, _ = _geometry(truth)
for panel in range(2):
    for width in (.5, .75, 1.):
        for height in (-.5, -.15, .15, .5):
            point = width * directions[panel] + height * axis
            observations = [{'photo': photo, 'uv': _project([point], frame)[0][0].tolist()} for photo, frame in frames.items()]
            tri = _triangulate(observations, frames)
            small_tracks.append({'xyz': tri[0], 'observations': observations, 'parallaxDeg': tri[2]})
bad_depth_frames = {}
for photo, frame in frames.items():
    points = frame['points'].copy(); points[..., 2] = 0.
    bad_depth_frames[photo] = {**frame, 'points': points}
small_fit = _fit_board(board, small_tracks, bad_depth_frames)
assert 16 <= len(small_fit['training']) < 24
assert abs(np.rad2deg(small_fit['initial'][6]) - 137.) < .5
assert small_fit['fit']['converged'] and abs(np.rad2deg(small_fit['parameters'][6]) - 137.) < .4
assert all(row['status'] == 'fit' for row in small_fit['leaveViews'])

# A world Sim(3) changes cameras and all scene quantities together, never angle.
R = Rotation.from_euler('xyz', [.3, -.4, .7]).as_matrix(); scale = 2.7; translation = np.array([3., -2., 4.])
transformed_frames = {}
for photo, frame in frames.items():
    pose = frame['pose'].copy(); pose[:3, :3] = R @ pose[:3, :3]; pose[:3, 3] = scale * R @ pose[:3, 3] + translation
    transformed_frames[photo] = {**frame, 'pose': pose}
transformed = fitted.copy(); transformed[:3] = scale * R @ fitted[:3] + translation
transformed[3:6] = Rotation.from_matrix(R @ Rotation.from_rotvec(fitted[3:6]).as_matrix()).as_rotvec()
assert abs(_stats(transformed, held, a_held, transformed_frames)['medianPx'] - validation['medianPx']) < 1e-8

record = {'parameters': fitted, 'initial': initial, 'training': training, 'trainAssignments': a_train, 'extent': 2.}
shared, shared_fit = _shared(record, record, frames)
assert shared_fit['converged'] and shared[0][6] == shared[1][6]
assert abs(np.rad2deg(shared[0][6] - truth[6])) < .3
with tempfile.TemporaryDirectory() as directory:
    file, panels, mesh_angle = _mesh(fitted, board, frames, Path(directory))
    exported = trimesh.load(Path(directory) / file, force='scene')
    assert len(exported.geometry) == 2 and len(panels) == 2
    normals = [np.linalg.svd(m.vertices - m.vertices.mean(0), full_matrices=False)[2][-1] for m in exported.geometry.values()]
    actual = 180 - np.rad2deg(np.arccos(np.clip(abs(normals[0] @ normals[1]), -1, 1)))
    assert abs(actual - mesh_angle) < .001 and abs(mesh_angle - np.rad2deg(fitted[6])) < 1e-6
    bad_subsets = {**automatic, 'leaveViews': [{**r, 'optimizer': {'converged': False}} for r in automatic['leaveViews']]}
    unsupported = _record(board, bad_subsets, automatic['parameters'], frames, Path(directory),
                          {'mPerNative': 1.2}, {'normal': [0, 0, 1], 'offset': 2.})
    assert unsupported['status'] == 'unsupported' and unsupported['conditionalAngleRangeDeg'] is None
    for name in ('groundClearanceNative', 'groundClearanceM', 'panelWidthsM'):
        assert unsupported[name] is None and unsupported['candidateGeometry'][name] is not None
    two_view_record = _record(board, two_view_fit, two_view_fit['parameters'], frames, Path(directory),
                              {'mPerNative': 1.2}, {'normal': [0, 0, 1], 'offset': 2.})
    assert two_view_record['status'] == 'unsupported' and two_view_record['conditionalAngleRangeDeg'] is None
    assert two_view_record['measurementAngleDeg'] is None

# The LK provider alone is checked on a translated textured plane. This tests
# correspondence/dedup plumbing without claiming a fold from one planar surface.
rng = np.random.default_rng(118)
texture = cv2.GaussianBlur(rng.integers(0, 256, (128, 128), dtype=np.uint8), (3, 3), .5)
plane_frames, plane_masks = {}, {}
K = np.array([[80., 0, 64], [0, 80., 64], [0, 0, 1]])
yy, xx = np.indices(texture.shape)
for photo, tx in ((1, 0.), (2, .2)):
    image = cv2.warpAffine(texture, np.array([[1., 0, -80 * tx / 3], [0, 1, 0.]]), (128, 128))
    pose = np.eye(4); pose[0, 3] = tx
    points = np.stack([(xx - 64) * 3 / 80 + tx, (yy - 64) * 3 / 80, np.full_like(xx, 3.)], axis=2)
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    plane_frames[photo] = {'K': K, 'pose': pose, 'C': np.eye(3), 'points': points, 'analysisRgb': rgb}
    plane_masks[photo] = (xx > 15) & (xx < 112) & (yy > 15) & (yy < 112)
lk_tracks, lk_diagnostics = _features(plane_frames, {'left': {'masks': plane_masks}}, method='lk')
assert len(lk_tracks['left']) >= 10
assert max(t['triangulationResidualPx'] for t in lk_tracks['left']) < .2
for index, track in enumerate(lk_tracks['left']):
    for other in lk_tracks['left'][index + 1:]:
        for a in track['observations']:
            for b in other['observations']:
                if a['photo'] == b['photo']:
                    assert np.linalg.norm(np.asarray(a['analysisUv']) - b['analysisUv']) >= 6
assert any('pairCounts' in row for row in lk_diagnostics)
print('PASS: angle recovery, isolated leave-view initialization, unused-view false-acceptance and 19-track RGB initializer regressions, convergence and measurement gates, held-out corruption, Sim(3), shared theta, GLB consistency and distinct LK pixel tracks')
