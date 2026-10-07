"""Run: PYTHONPATH=.:scripts python scripts/check_workcell_camera_pixels.py."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as N

import numpy as np

from scripts.workcell_guard_controls import _export_reconstruction


def check():
    images, cameras, originals, expected, tracks = {}, {}, {}, {}, []
    xyz = np.array([.2, .1, 3.])
    for i in (1, 2):
        raw_K = np.array([[800.+20*i, 0, 1511.5], [0, 800.+20*i, 2015.5], [0, 0, 1.]])
        A = np.array([[.129+i*.001, 0, -.44], [0, .128, -2.42], [0, 0, 1.]])
        resize = np.array([[.385, 0, (.385-1)/2], [0, .385, (.385-1)/2], [0, 0, 1.]])
        S = np.array([[3., 0, 1], [0, 3., 1], [0, 0, 1.]]) if i == 1 else resize @ np.linalg.inv(A)
        K = A @ raw_K; control_K = S @ K
        if i == 2:
            assert np.isclose(control_K[0, 0], control_K[1, 1]), 'Square raw pixels must survive isotropic source resize'
        colmap_K = control_K.copy(); colmap_K[:2, 2] += .5
        pose = np.eye(4); pose[0, 3] = i-1
        canonical_uv = K @ (xyz-pose[:3, 3]); canonical_uv = canonical_uv[:2] / canonical_uv[2]
        control_uv = S @ np.r_[canonical_uv, 1.]
        name = f'photo-{i}.png'
        cameras[i] = N(calibration_matrix=lambda k=colmap_K: k.copy())
        images[i] = N(name=name, camera_id=i, points2D=[N(xy=control_uv[:2]+.5)],
                      cam_from_world=lambda p=pose: N(inverse=lambda: N(matrix=lambda: p[:3])))
        originals[name] = {'photo': i, 'pose': pose, 'pixelTransform': S}
        expected[i] = (K, canonical_uv)
        tracks.append(N(image_id=i, point2D_idx=0))
    recon = N(images=images, cameras=cameras, points3D={1:N(xyz=xyz, error=0., track=N(elements=tracks))})
    with tempfile.TemporaryDirectory(prefix='workcell-camera-pixels-') as directory:
        out = Path(directory); _export_reconstruction(recon, originals, out)
        for row in json.loads((out/'cameras.json').read_text())['frames']:
            assert np.allclose(row['K'], expected[row['photo']][0])
        for row in json.loads((out/'tracks.json').read_text())['tracks'][0]['observations']:
            assert np.allclose(row['uv'], expected[row['photo']][1]), 'Each camera must use its own exact pixel transform'
    print('PASS: square raw pixels, per-camera crop/resize, COLMAP half-pixel convention, track export')


if __name__ == '__main__':
    check()
