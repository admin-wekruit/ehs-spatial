"""Run: python scripts/workcell_photo_geometry_check.py [artifact-directory]."""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

from workcell_photo_geometry import _edge_overlap, _beam, _clearance, _intersect, _rays, build
from workcell_photo_oneshot import _array, _frame


def project(points, K, pose):
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    p = local @ K.T
    return p[:, :2] / p[:, 2:]


# A tilted front plane makes bbox * median-depth / focal-length wrong.
K = np.array([[700., 0, 320], [0, 700., 240], [0, 0, 1]])
pose = np.eye(4)
up = np.array([0., -1., 0.])
side = np.array([.6, 0., .8])
normal = np.cross(side, up)
center = np.array([.2, -.3, 3.])
points = np.array([center + x * side + y * up for x, y in [(-.4, -.25), (.4, -.25), (.4, .25), (-.4, .25)]])
offset = -float(center @ normal)
uv = project(points, K, pose)
recovered = _intersect(uv, K, pose, normal, offset)
assert np.allclose(points, recovered, atol=1e-10)
assert np.isclose(np.ptp(recovered @ side), .8)
assert np.isclose(np.ptp(recovered @ up), .5)
assert abs(np.ptp(uv[:, 0]) * np.median(points[:, 2]) / K[0, 0] - .8) > .1

# Opposite edge detection may be fragmented on a different OpenCV build.
# Matching is symmetric and uses only the overlapping, observed interval.
long = np.array([[0., 0., 0.], [1., 0., 0.]])
fragment = np.array([[.6, .02, 0.], [.9, .02, 0.]])
for a, b in [(long, fragment), (fragment, long), (long[::-1], fragment[::-1])]:
    overlap, supported = _edge_overlap(a, b, np.array([1., 0, 0]))
    assert supported and np.isclose(overlap, .3)
assert not _edge_overlap(long, fragment + [1., 0, 0], np.array([1., 0, 0]))[1]
assert not _edge_overlap(long, fragment + [.35, 0, 0], np.array([1., 0, 0]))[1]

# The same physical line in two camera images reconstructs independently of
# the pointmap. Its floor clearance and normal projection must agree.
line = np.array([[-1., -.3, 3.], [1., -.3, 3.]])
poses = [np.eye(4), np.eye(4)]
poses[1][:3, 3] = [.3, .2, -.5]
view_planes, view_offsets = [], []
for camera in poses:
    rays = _rays(project(line, K, camera), K, camera)
    n = np.cross(*rays); n /= np.linalg.norm(n)
    view_planes.append(n); view_offsets.append(n @ camera[:3, 3])
axis = np.array([1., 0, 0])
point = np.linalg.solve(np.vstack([view_planes, axis]), np.r_[view_offsets, 0.])
height, foot = _clearance(point, up, 0.)
assert np.allclose(point, line.mean(0), atol=1e-9)
assert np.isclose(height, .3) and np.isclose(foot @ up, 0)

# Display scale must multiply geometry, cameras, offsets, and clearances once.
for scale in [.1, 2., 11.]:
    scaled_pose = pose.copy(); scaled_pose[:3, 3] *= scale
    result = _intersect(uv, K, scaled_pose, normal, offset * scale)
    assert np.allclose(result, points * scale)
    scaled_h, scaled_foot = _clearance(point * scale, up, 0.)
    assert np.isclose(scaled_h, height * scale)
    assert np.allclose(scaled_foot, foot * scale)

mesh = _beam(*line, .04, .01, up, np.array([0., 0., 1.]))
assert np.isclose(min(mesh.vertices @ up), .28)
for bad in [0., -1., float('nan'), float('inf')]:
    try:
        build(Path('/nonexistent'), [], height_m=bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid reference height accepted')

if len(sys.argv) > 1:
    root = Path(sys.argv[1])
    data = json.loads((root / 'geometry.json').read_text())
    scene = trimesh.load(root / 'fence-fitted.glb', force='scene')
    n = np.array(data['floor']['normal']); d = data['floor']['offset']
    assert data['clearances'], 'Known four-photo capture lost its multiview lower-rail clearance'
    for clearance in data['clearances']:
        point = np.array(clearance['pointNative']); foot = np.array(clearance['footNative'])
        h = clearance['heightNative']
        assert np.isclose(point @ n + d, h)
        assert abs(foot @ n + d) < 1e-5
        assert np.allclose(point - foot, n * h)
        beam = scene.geometry[clearance['meshNode']]
        assert np.isclose(min(beam.vertices @ n + d), h, atol=1e-5)
        assert len(clearance['sourcePhotos']) >= 2
        record = next(b for b in data['fence']['beams'] if b['id'] == clearance['meshNode'])
        plane = data['fence']['planes'][record['plane']]
        frame = _frame(root, record['sourcePhoto'])
        A = np.asarray(frame['input_mask_transform']['input_to_canonical_pixel_centres'])
        axis = np.diff(record['endsNative'], axis=0)[0]; axis /= np.linalg.norm(axis)
        spans = []
        for edge in record['rawEdges']:
            pixels = (np.c_[edge, np.ones(2)] @ A.T)[:, :2]
            xyz = _intersect(pixels, _array(frame['intrinsics']), _array(frame['camera_poses']),
                             np.asarray(plane['normal']), plane['offset'])
            spans.append(sorted(xyz @ axis))
        # Check exported vertices, not just the intermediate JSON endpoints.
        assert min(beam.vertices @ axis) >= max(x[0] for x in spans) - 1e-5
        assert max(beam.vertices @ axis) <= min(x[1] for x in spans) + 1e-5
    assert len(data['anchor']['views']) >= 2
    assert all((root / name).is_file() for name in data['evidenceImages'])
print('PASS: fragmented-edge pairing, known-capture nonempty clearance, perspective, multiview line triangulation, model clearance, scale invariance, input validation')
