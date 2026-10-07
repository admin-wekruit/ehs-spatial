"""Licence-clean MVS route, completed (2026-10-06): the MVS of licence-clean-stack-2026-10-06/geometry (BA-f cameras from a clean
start + RoMa dense warp + two-view midpoint triangulation, no learned depth), with ONE generic change decided from a band
diagnostic before any new number was scored:

  CERT 0.5 -> 0.05 = RoMa's own sample_thresh (romatch RegressionMatcher: certainty above it is set to 1, i.e. 'certain').
  In the pinned housing bands certainty, not geometry, removed the pixels: e.g. 090 photo 3 <- photo 1 had forward-backward
  0.14 px and reprojection 0.14 px at the median but certainty 0.14 < 0.5. The geometric checks are unchanged
  (forward-backward < 1 px, ray angle >= 2 deg, reprojection < 1 px, both depths > 0).
  conf follows the same convention: 1 where certainty > sample_thresh (every kept pixel), so the harness's conf >= 0.1 keeps them.

  python mvs_route.py [base ...]     local CPU (numpy): writes SCR/checks/bbab-geom/CELL-mvs-BASE-padded/geometry
Code reused unchanged: geometry_clean_ab.mvs / read_geometry / write_geometry (worktree panoptes-workcell-photo-speed).
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

NOTE = Path(__file__).resolve().parent
sys.path[:0] = [str(NOTE), os.environ.get('PANOPTES_WORKCELL', '/Users/adam/.codex/worktrees/panoptes-workcell-photo-speed') + '/modal_apps']
import geometry_clean_ab as gc  # noqa: E402
import backbones as bb  # noqa: E402

ROMA_SAMPLE_THRESH = .05
THRESHOLDS = dict(gc.MVS, CERT=ROMA_SAMPLE_THRESH)
GEOM = gc.SCR / 'checks/bbab-geom'


def dense_for(cell):
    out = {}
    for f in sorted((gc.GPUD / cell).glob('dense-*.npz')):
        i, j = map(int, f.stem.split('-')[1:]); d = np.load(f); out[(i, j)] = (d['uvAB'], d['certA'], d['uvBA'], d['certB'])
    return out


def run(cell, base, cert=ROMA_SAMPLE_THRESH):
    """cert != ROMA_SAMPLE_THRESH: sensitivity rows only (same everything else)."""
    src = gc.GEOM / f'{cell}-{base}-ba-f' / 'geometry'; thr = dict(THRESHOLDS, CERT=cert)
    saved = dict(gc.MVS); gc.MVS.update(thr)  # ponytail: gc.mvs reads its module thresholds; patched for this call only
    try:
        out, st = gc.mvs(gc.read_geometry(src), dense_for(cell))
    finally:
        gc.MVS.clear(); gc.MVS.update(saved)
    for o in out:
        o['conf'] = o['valid'].astype(np.float32)
    dst = GEOM / f"{cell}-mvs-{base}{'' if cert == ROMA_SAMPLE_THRESH else f'-c{cert:g}'}-padded" / 'geometry'
    assert not dst.exists(), dst
    gc.write_geometry(dst, cell, out, dict(model_id=f'licence-clean MVS ({base} BA-f cameras, RoMa outdoor dense warp, midpoint triangulation)',
                                           cameras=str(src), thresholds=thr, change='CERT 0.5 -> RoMa sample_thresh 0.05; conf = 1 on kept pixels',
                                           report=st, created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    st['contract'] = bb.check_arrays(dst, len(out))
    print(cell, base, json.dumps(st))
    return st


if __name__ == '__main__':  # python mvs_route.py BASE [CERT ...]
    base, certs = (sys.argv[1:2] or ['da3-base'])[0], [float(x) for x in sys.argv[2:]] or [ROMA_SAMPLE_THRESH]
    for cert in certs:
        for c in ('090', '030'):
            run(c, base, cert)
