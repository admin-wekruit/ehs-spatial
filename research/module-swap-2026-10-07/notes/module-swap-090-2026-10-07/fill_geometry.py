"""Geometry module v2 for cell 090: the GPL-free MVS point maps with their holes INSIDE OBJECT MASKS filled from MoGe-3 (MIT)
monocular depth, aligned per object and per photo to the MVS points (z_mvs = s * z_moge + t, Huber IRLS on the object's own
MVS-valid pixels plus a ring around its mask; ring-only / photo-global fallbacks). Floor and everything outside object masks
stay pure MVS. Output = the same geometry contract (pts3d / conf / valid_mask / K / c2w per frame) in the backbone-A/B layout
GEOM/090-mvs-fill-padded/geometry, so the unchanged fair analyse + export_run produce the run directory.

Method and held-out errors: scratchpad/review-coverage-gaps/moge_fill_eval.py (in-mask fit median 1-5 cm on held-out bands).
Filled pixels get conf 0.3 (content rule conf >= 0.1), original MVS pixels keep conf 1.

    nice python fill_geometry.py            -> GEOM/090-mvs-fill-padded/{geometry, fill-record.json}
"""
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

import os
SP = Path(os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad'))
CELL = os.environ.get('FILL_CELL', '090')                 # FILL_CELL=030: the same module on the 030 cell's MVS export
SRC = SP / f'checks/bbab-export-{CELL}-mvs-scipyba'      # the GPL-free MVS run (geometry + masks)
MOGE = SP / f'checks/clean-gpu/{CELL}'                   # moge-frame_000N.npz on the same frozen 518 frames (content crop at X0)
GEOM = SP / f'checks/bbab-geom/{CELL}-mvs-fill-padded'
_MAN = json.loads((SRC / 'manifest.json').read_text())
X0 = int(_MAN['frames'][0]['content_rect_xyxy'][0])     # content crop offset of the frozen frames (63 for the 4:3 photos)
FILL_CONF, MIN_INMASK, MIN_RING, RING_PX = 0.3, 40, 30, (10, 20, 40, 80, 160)
LOCAL_SIGMA_PX, MIN_LOCAL_WEIGHT, RANGE_MARGIN = (48, 24, 12), 0.02, 0.03   # offset field scales (coarse -> fine), min Gaussian mass on MVS pixels, range gate (native, ~10 cm)
RIM_PX, EDGE_REL = 3, 0.05   # no fill within 3 px of the mask rim; no fill where MoGe depth jumps > 5 % within 5x5 px (discontinuity)
FRAMES = tuple(f['frame_id'] for f in _MAN['frames'])


def huber_fit(x, y, iters=20, k=1.345):
    """y ~ s x + t, Huber IRLS (scale from MAD)."""
    A = np.c_[x, np.ones_like(x)]
    w = np.ones_like(x)
    for _ in range(iters):
        p = np.linalg.lstsq(A * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
        r = y - A @ p
        sig = 1.4826 * np.median(np.abs(r)) + 1e-9
        u = np.abs(r) / (k * sig)
        w = np.where(u <= 1, 1, 1 / u)
    return p, float(1.4826 * np.median(np.abs(y - A @ p)))


def load_frame(fid):
    g = SRC / 'geometry/frames' / fid
    P = np.load(g / 'pts3d.npy').astype(np.float64); c2w = np.load(g / 'camera_to_world.npy').astype(np.float64)
    K = np.load(g / 'intrinsics.npy').astype(np.float64); conf = np.load(g / 'conf.npy').astype(np.float32)
    valid = np.load(g / 'valid_mask.npy').astype(bool) & np.load(g / 'content_valid_mask.npy').astype(bool)
    z = ((P - c2w[:3, 3]) @ c2w[:3, :3])[..., 2]
    valid &= np.isfinite(z) & (z > 0)
    d = np.load(MOGE / f'moge-{fid}.npz')
    zm = np.full(z.shape, np.nan)
    pm = d['points'][..., 2].astype(np.float64)
    zm[:, X0:X0 + pm.shape[1]] = np.where(d['mask'] & np.isfinite(pm) & (pm > 0), pm, np.nan)
    return dict(P=P, c2w=c2w, K=K, conf=conf, valid=valid, valid0=valid.copy(), z=z, zm=zm, g=g)


def ring(m, ok):
    for r in RING_PX:
        rr = cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool) & ~m
        if (rr & ok).sum() >= 200:
            return rr & ok, r
    return rr & ok, r


def backproject(fr, mask, z):
    v, u = np.nonzero(mask)
    K, c2w = fr['K'], fr['c2w']
    d = np.c_[(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(len(u))] * z[v, u][:, None]
    return v, u, d @ c2w[:3, :3].T + c2w[:3, 3]


def main():
    frames = {f: load_frame(f) for f in FRAMES}
    glob_fit = {}
    for f, fr in frames.items():
        ok = fr['valid'] & np.isfinite(fr['zm'])
        glob_fit[f] = huber_fit(fr['zm'][ok], fr['z'][ok]); print(f, 'global s=%.4f t=%+.4f mad=%.4f n=%d' % (*glob_fit[f][0], glob_fit[f][1], ok.sum()))
    objs = json.loads((SRC / 'evidence/objects.json').read_text())['objects']   # observed_floor is not in objects.json: floor untouched
    record, filled_px = {}, {f: np.zeros(frames[f]['z'].shape, bool) for f in FRAMES}
    # pixels inside more than one object mask (SAM 3 masks overlap: right fence / right light curtain) have no single owner: never
    # filled. (The fence's fills at the fence's depth landed inside the housing mask and tilted the evaluator's housing plane.)
    cover = {f: np.zeros(frames[f]['z'].shape, np.int32) for f in FRAMES}
    for o in objs:
        for vw in o['views']:
            cover[vw['frame_id']] += np.load(SRC / vw['canonical_mask_path']).astype(bool)
    overlap = {f: c > 1 for f, c in cover.items()}
    for o in sorted(objs, key=lambda o: -sum(v['mask_pixels'] for v in o['views'])):
        for vw in o['views']:
            f, fr = vw['frame_id'], frames[vw['frame_id']]
            m = np.load(SRC / vw['canonical_mask_path']).astype(bool)
            ok = fr['valid'] & np.isfinite(fr['zm'])
            inm = m & ok
            rr, rpx = ring(m, ok)
            # scale s: the photo's global Huber fit (thousands of pixels over the whole depth range; a free per-object or object+ring fit
            # is ill-conditioned on thin / fronto-parallel objects and went negative on two light-curtain views). Offset t: the object's
            # own MVS pixels (median residual; moge_fill_eval: the in-mask alignment had the lowest held-out error), else the ring.
            (s, t), mad = glob_fit[f]; mode = 'global-s'
            for sel, name in ((inm, 'inmask-t'), (rr, 'ring-t')):
                if sel.sum() >= MIN_INMASK:
                    res = fr['z'][sel] - (s * fr['zm'][sel] + t)
                    t += float(np.median(res)); mad = float(1.4826 * np.median(np.abs(res - np.median(res)))); mode += '+' + name
                    break
            # offset field: one object-wide t leaves MoGe's local relative-depth error (a few cm) in the fill, which tilted the fair
            # evaluator's housing plane (8.8 -> 57.6 deg from 7 off-surface points). Normalized convolution of the residual over the
            # object's own MVS pixels (coarse to fine, finer scale wins where it has enough support) keeps the fill continuous with them.
            t_field = np.full(m.shape, t)
            r, w = np.where(inm, fr['z'] - s * fr['zm'], 0.), inm.astype(np.float64)
            local = np.zeros(m.shape, bool)
            for sig in LOCAL_SIGMA_PX:
                num, den = cv2.GaussianBlur(r, (0, 0), sig), cv2.GaussianBlur(w, (0, 0), sig)
                good = den > MIN_LOCAL_WEIGHT
                t_field[good] = num[good] / den[good]; local |= good
            z_new = s * fr['zm'] + t_field
            # never the mask rim, never a MoGe depth discontinuity (max-min over 5x5 > 5 % of depth): monocular depth blends across
            # object edges, and those pixels are exactly where MVS has its holes (the housing's lower edge: 7 blended points tilted the
            # evaluator's free plane and moved the field value 24.6 -> 14.4 cm)
            core = cv2.erode(m.astype(np.uint8), np.ones((2 * RIM_PX + 1, 2 * RIM_PX + 1), np.uint8)).astype(bool)
            zmf = np.where(np.isfinite(fr['zm']), fr['zm'], 0.).astype(np.float32)
            k5 = np.ones((5, 5), np.uint8)
            edge = (cv2.dilate(zmf, k5) - cv2.erode(np.where(np.isfinite(fr['zm']), zmf, np.inf).astype(np.float32), k5)) > EDGE_REL * zmf
            hole = core & ~edge & ~overlap[f] & ~fr['valid'] & np.isfinite(fr['zm']) & ~filled_px[f] & (z_new > 0)
            if inm.sum() >= MIN_INMASK:   # outlier gate: the object's own MVS depth range, widened by its noise
                lo, hi = np.quantile(fr['z'][inm], [.02, .98]); mg = max(3 * mad, RANGE_MARGIN)
                in_range = (z_new >= lo - mg) & (z_new <= hi + mg)
            else:
                in_range = np.ones(m.shape, bool)
            fill = hole & in_range
            v, u, X = backproject(fr, fill, z_new)
            fr['P'][v, u] = X; fr['valid'][v, u] = True; fr['conf'][v, u] = FILL_CONF; filled_px[f][v, u] = True
            record[f"{o['object_id']}/{f}"] = dict(maskPx=int(m.sum()), coverageBefore=round(float((m & fr['valid0']).sum() / max(m.sum(), 1)), 3),
                                                   overlapPx=int((m & overlap[f]).sum()), rejectedOverlap=int((m & overlap[f] & ~fr['valid0'] & np.isfinite(fr['zm'])).sum()),
                                                   coverageAfter=round(float((m & fr['valid']).sum() / max(m.sum(), 1)), 3), filledPx=int(fill.sum()),
                                                   rejectedRange=int((hole & ~in_range).sum()), rejectedRimEdge=int((m & ~fr['valid'] & np.isfinite(fr['zm']) & ~(core & ~edge)).sum()),
                                                   localOffsetShare=round(float((fill & local).sum() / max(fill.sum(), 1)), 3),
                                                   fit=mode, s=round(float(s), 4), t=round(float(t), 4), madNative=round(mad, 5), ringPx=rpx,
                                                   inmaskN=int(inm.sum()), ringN=int(rr.sum()))
    out = GEOM / 'geometry'
    if GEOM.exists():
        shutil.rmtree(GEOM)
    for f, fr in frames.items():
        g = out / 'frames' / f; g.mkdir(parents=True)
        np.save(g / 'pts3d.npy', fr['P'].astype(np.float32)); np.save(g / 'conf.npy', fr['conf'].astype(np.float32))
        np.save(g / 'valid_mask.npy', fr['valid'])
        for name in ('intrinsics.npy', 'camera_to_world.npy', 'canonical.png'):   # canonical.png: the contract check's frozen-frame assert
            shutil.copy(fr['g'] / name, g / name)
    cm = json.loads((SRC / 'geometry/candidate_manifest.json').read_text())
    cm['model_id'] = str(cm.get('model_id', 'mvs-scipyba')) + ' + MoGe-3 in-mask fill (per object/photo Huber affine on MVS)'
    cm['fill'] = dict(method='MoGe-3 (MIT) z aligned to MVS per photo (global Huber scale s) and per object (offset field: normalized '
                      'convolution of z_mvs - s*z_moge over the object\'s own MVS pixels at sigma 48/24/12 px, object median fallback); '
                      'filled only where MVS invalid inside object masks and inside the object\'s MVS depth range; conf 0.3; floor untouched',
                      evaluation='review-coverage-gaps/moge_fill_eval.py', params=dict(sigmaPx=LOCAL_SIGMA_PX, minLocalWeight=MIN_LOCAL_WEIGHT, rangeMarginNative=RANGE_MARGIN, conf=FILL_CONF))
    (out / 'candidate_manifest.json').write_text(json.dumps(cm, indent=2) + '\n')
    (GEOM / 'fill-record.json').write_text(json.dumps(record, indent=1) + '\n')
    for k, r in record.items():
        print(f"{k:32s} {r['coverageBefore']:.2f} -> {r['coverageAfter']:.2f}  +{r['filledPx']:5d}px  rej={r['rejectedRange']:4d} local={r['localOffsetShare']:.2f}  "
              f"s={r['s']:.3f} t={r['t']:+.3f} mad={r['madNative']:.4f}")
    print('->', out)


if __name__ == '__main__':
    main()
