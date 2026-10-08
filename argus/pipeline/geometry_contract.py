"""Validate the frozen canonical point-map and rigid-camera contract."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
SIZE = 518

def check_arrays(geom: Path, n_frames: int, size: int = SIZE) -> dict:
    """Shapes, dtypes, finiteness, K and rigid c2w, frame count; returns the point-map/camera consistency per frame (px)."""
    geom = Path(geom); ids = sorted(p.name for p in (geom / 'frames').iterdir())
    assert ids == [f'frame_{i:04d}' for i in range(1, n_frames + 1)], ids
    out = {}
    for fid in ids:
        g = geom / 'frames' / fid
        img = np.asarray(Image.open(g / 'canonical.png').convert('RGB'))
        pts, conf, valid = np.load(g / 'pts3d.npy'), np.load(g / 'conf.npy'), np.load(g / 'valid_mask.npy')
        K, M = np.load(g / 'intrinsics.npy').astype(float), np.load(g / 'camera_to_world.npy').astype(float)
        assert img.shape == (size, size, 3) and pts.shape == (size, size, 3) and pts.dtype == np.float32, (fid, img.shape, pts.shape, pts.dtype)
        assert conf.shape == valid.shape == (size, size) and valid.dtype == bool, (fid, conf.shape, valid.dtype)
        assert valid.mean() > .05, (fid, 'valid fraction', valid.mean())  # MVS point maps are sparse; this catches empty outputs
        assert np.isfinite(pts[valid]).all() and np.isfinite(conf[valid]).all(), (fid, 'non-finite valid values')
        assert K.shape == (3, 3) and np.isfinite(K).all() and np.allclose(K[2], [0, 0, 1]) and min(K[0, 0], K[1, 1]) > 0, (fid, K)
        assert abs(K[0, 1]) < 1e-3 * K[0, 0] and 0 < K[0, 2] < size and 0 < K[1, 2] < size, (fid, K)
        R = M[:3, :3]
        assert M.shape == (4, 4) and np.isfinite(M).all() and np.allclose(M[3], [0, 0, 0, 1]), (fid, M)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-3) and abs(np.linalg.det(R) - 1) < 1e-3, (fid, 'c2w rotation not orthonormal')
        v, u = np.nonzero(valid); Xc = (pts[v, u].astype(float) - M[:3, 3]) @ R  # world -> camera
        front = Xc[:, 2] > 0
        assert front.mean() > .99, (fid, 'points behind the camera', 1 - front.mean())
        uv = (Xc[front] / Xc[front, 2:]) @ K.T
        err = np.hypot(uv[:, 0] - u[front], uv[:, 1] - v[front])
        out[fid] = dict(validFraction=float(valid.mean()), reprojMedianPx=float(np.median(err)), reprojP95Px=float(np.quantile(err, .95)))
        assert out[fid]['reprojMedianPx'] < 5., (fid, 'point map is not on the canonical grid of (K, c2w)', out[fid])
    return out

def check_geometry(geom: Path, run: Path) -> dict:
    """check_arrays + the frame ids and canonical images of the frozen run (pixel-identical input)."""
    man = json.loads((Path(run) / 'manifest.json').read_text())
    out = check_arrays(geom, len(man['frames']))
    for f in man['frames']:
        seen = np.asarray(Image.open(Path(geom) / 'frames' / f['frame_id'] / 'canonical.png').convert('RGB'))
        assert np.array_equal(seen, np.asarray(Image.open(Path(run) / f['canonical']).convert('RGB'))), (f['frame_id'], 'not the frozen frame')
    return out
