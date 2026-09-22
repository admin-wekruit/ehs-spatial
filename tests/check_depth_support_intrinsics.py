"""A static tilted plane remains consistent when each frame has different K."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_droid_replay import depth_support

poses = np.repeat(np.eye(4)[None], 3, axis=0)
ks = np.array([[20., 20., 20., 20.], [40., 40., 20., 20.], [80., 80., 20., 20.]])
y, x = np.indices((41, 41))
# z = 2 + .5*x_world + .3*y_world, rendered analytically in each camera.
inverse_depth = np.array([(1 - .5 * (x-cx)/fx - .3 * (y-cy)/fy) / 2 for fx, fy, cx, cy in ks])
neighbours = [[1, 2], [0, 2], [0, 1]]
votes, _, retained = depth_support(poses, inverse_depth, ks, neighbours=neighbours)
for i, pixel in enumerate([24, 28, 36]):
    assert votes[i, pixel, pixel] == 2 and retained[i, pixel, pixel], 'A refocused camera must use its own rays and each target camera K'
# A mismatched plane must still be rejected; changing K is not a relaxed depth check.
wrong = inverse_depth.copy(); wrong[0] /= 1.2
assert depth_support(poses, wrong, ks, neighbours=neighbours)[0][0, 24, 24] == 0
print('PASS: per-frame camera intrinsics preserve consistent geometry and reject wrong depth')
