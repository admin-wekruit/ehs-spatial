"""Run with the serving venv: same-frame color must not leak through occluders."""
import sys
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from color_video_bodies import source_colors

# Two identical triangles on the same camera rays: only the nearer may get RGB.
front = np.array([[-.4, -.4, 2], [.4, -.4, 2], [0, .4, 2]])
mesh = trimesh.Trimesh(vertices=np.r_[front, front*1.5], faces=[[0, 1, 2], [3, 4, 5]], process=False)
yy, xx = np.mgrid[:64, :64]
rgb = np.stack((xx*3, yy*3, xx+yy), axis=-1).astype('uint8'); mask = np.ones((64, 64), bool)
k = np.array([[50., 0, 32], [0, 50, 32], [0, 0, 1]])
before = mesh.vertices.copy()
colors, visible = source_colors(mesh, rgb, mask, k, np.eye(4))
assert set(visible) == {0, 1, 2}
assert np.array_equal(colors[:3, :3], rgb[[22, 22, 42], [22, 42, 32]])
assert (colors[3:] == [178, 207, 222, 255]).all()
assert np.array_equal(mesh.vertices, before)
_, hidden = source_colors(mesh, rgb, ~mask, k, np.eye(4)); assert not len(hidden)
# Native world rotation/translation and scale must preserve source sampling.
c = np.eye(4); c[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]; c[:3, 3] = [4, -2, 8]
mesh.vertices = before*7 @ c[:3, :3].T+c[:3, 3]
shifted, same = source_colors(mesh, rgb, mask, k, c)
assert np.array_equal(shifted, colors) and np.array_equal(same, visible)
try: source_colors(mesh, rgb[:-1], mask, k, c)
except ValueError: pass
else: raise AssertionError('Mismatched pixel domain accepted')
print('PASS: RGB provenance domain, mask exclusion, self-occlusion, unchanged geometry and world/scale invariance')
