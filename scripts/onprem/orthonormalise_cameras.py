"""Project each camera rotation of a geometry directory onto the nearest rotation (generic: any backbone, any run).

  python scripts/onprem/orthonormalise_cameras.py RUN/geometry [--tolerance 1e-6] [--max-change 1e-4] [--keep-as pi3x-bf16]
  python scripts/onprem/orthonormalise_cameras.py --self-test

Why: Pi3X on CUDA runs under bf16 autocast, and its camera_to_world rotations come out a few 1e-6 from orthonormal; the platform
importer accepts at most 1e-6 (max |R^T R - I|). For every RUN/geometry/frames/*/camera_to_world.npy with a deviation above
--tolerance: R -> U V^T of its SVD (det +1; the translation and the bottom row are unchanged, the dtype is kept), the original
file is kept as camera_to_world.<keep-as>.npy (never overwritten), and RUN/geometry/camera-orthonormalisation.json records the
deviation before / after and the largest entry change. A change above --max-change (1e-4) is not rounding: nothing is written
and the tool exits non-zero (as it does for a reflection, det < 0). Frames already within the tolerance are left alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np


def deviation(R) -> float:
    R = np.asarray(R, np.float64)
    return float(np.abs(R.T @ R - np.eye(3)).max())


def nearest_rotation(R) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, np.float64))
    if np.linalg.det(U @ Vt) < 0:
        raise ValueError('reflection (det < 0): not a camera rotation')
    return U @ Vt


def plan(geometry: Path, tolerance: float, max_change: float) -> list:
    rows = []
    for path in sorted(geometry.glob('frames/*/camera_to_world.npy')):
        M = np.load(path)
        before = deviation(M[:3, :3])
        if before <= tolerance:
            continue
        new = M.copy()
        new[:3, :3] = nearest_rotation(M[:3, :3]).astype(M.dtype)
        change = float(np.abs(new.astype(np.float64) - M.astype(np.float64)).max())
        rows.append(dict(path=path, matrix=new, frame_id=path.parent.name, deviation_before=before, det_before=float(np.linalg.det(M[:3, :3].astype(np.float64))),
                         deviation_after=deviation(new[:3, :3]), max_entry_change=change))
    bad = [r for r in rows if r['max_entry_change'] > max_change]
    if bad:
        raise ValueError(f"rotation change {max(r['max_entry_change'] for r in bad):.3g} > {max_change:g} in "
                         f"{[r['frame_id'] for r in bad]}: not bf16 rounding, refusing to touch the geometry")
    return rows


def orthonormalise(geometry: Path, tolerance=1e-6, max_change=1e-4, keep_as='original') -> dict:
    rows = plan(geometry, tolerance, max_change)
    for r in rows:
        kept = r['path'].with_name(f'camera_to_world.{keep_as}.npy')
        if kept.exists():
            raise FileExistsError(f'{kept} exists: the original is already kept there, not overwriting it')
    record_path = geometry / 'camera-orthonormalisation.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else dict(
        method='nearest rotation (SVD of the 3x3, U V^T, det +1), translation and dtype unchanged',
        tool='scripts/onprem/orthonormalise_cameras.py', frames=[])
    record.update(tolerance=tolerance, maxChangeAllowed=max_change)
    for r in rows:
        kept = r['path'].with_name(f'camera_to_world.{keep_as}.npy')
        r['path'].rename(kept)
        np.save(r['path'], r['matrix'])
        record['frames'].append(dict(frame_id=r['frame_id'], original=kept.name, **{k: r[k] for k in
                                     ('deviation_before', 'det_before', 'deviation_after', 'max_entry_change')}))
    if rows:
        record_path.write_text(json.dumps(record, indent=1) + '\n')
    return dict(changed=[r['frame_id'] for r in rows], record=str(record_path) if rows else None,
                frames=[{k: r[k] for k in ('frame_id', 'deviation_before', 'deviation_after', 'max_entry_change')} for r in rows])


def _check():
    rng = np.random.default_rng(0)
    def rot():
        q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    with tempfile.TemporaryDirectory() as tmp:
        g = Path(tmp) / 'geometry'
        mats = {}
        for name, noise in (('frame_0001', 4e-6), ('frame_0002', 0.)):
            M = np.eye(4, dtype=np.float32); M[:3, :3] = rot() + noise * rng.normal(size=(3, 3)); M[:3, 3] = rng.normal(size=3)
            (g / 'frames' / name).mkdir(parents=True); np.save(g / 'frames' / name / 'camera_to_world.npy', M); mats[name] = M
        assert deviation(mats['frame_0001'][:3, :3]) > 1e-6 >= deviation(mats['frame_0002'][:3, :3])
        out = orthonormalise(g, keep_as='pi3x-bf16')
        assert out['changed'] == ['frame_0001'], out
        new, old = np.load(g / 'frames/frame_0001/camera_to_world.npy'), np.load(g / 'frames/frame_0001/camera_to_world.pi3x-bf16.npy')
        assert np.array_equal(old, mats['frame_0001']) and new.dtype == old.dtype and np.array_equal(new[:3, 3], old[:3, 3])
        assert deviation(new[:3, :3]) < 1e-6 and np.isclose(np.linalg.det(new[:3, :3].astype(float)), 1, atol=1e-6)
        assert 0 < out['frames'][0]['max_entry_change'] < 2e-5 and np.array_equal(new[3], old[3])
        assert np.array_equal(np.load(g / 'frames/frame_0002/camera_to_world.npy'), mats['frame_0002'])
        assert orthonormalise(g)['changed'] == []  # idempotent
        rec = json.loads((g / 'camera-orthonormalisation.json').read_text())
        assert [f['frame_id'] for f in rec['frames']] == ['frame_0001'] and rec['frames'][0]['original'] == 'camera_to_world.pi3x-bf16.npy'
        bad = mats['frame_0002'].copy(); bad[:3, :3] += 1e-3 * rng.normal(size=(3, 3)); np.save(g / 'frames/frame_0002/camera_to_world.npy', bad)
        for call in (lambda: orthonormalise(g), lambda: nearest_rotation(np.diag([1., 1, -1]))):
            try:
                call(); raise AssertionError('not refused')
            except ValueError:
                pass
        assert np.array_equal(np.load(g / 'frames/frame_0002/camera_to_world.npy'), bad)  # refused: nothing written
        assert not (g / 'frames/frame_0002/camera_to_world.original.npy').exists()
    print('orthonormalise_cameras self-test passed: bf16-size deviation fixed (original kept, translation/dtype unchanged),',
          'orthonormal frame untouched, idempotent, 1e-3 change and a reflection refused')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('geometry', type=Path, nargs='?')
    ap.add_argument('--tolerance', type=float, default=1e-6)
    ap.add_argument('--max-change', type=float, default=1e-4)
    ap.add_argument('--keep-as', default='original')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _check()
    elif a.geometry is None:
        ap.error('give RUN/geometry or --self-test')
    else:
        try:
            print(json.dumps(orthonormalise(a.geometry, a.tolerance, a.max_change, a.keep_as)))
        except (ValueError, FileExistsError) as error:
            sys.exit(f'orthonormalise_cameras: {error}')
