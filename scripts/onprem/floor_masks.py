"""SAM 3 floor masks for a new capture (on-prem, replaces Gemini + fal SAM 3.1): which instances are floor, by score and geometry.

  python scripts/onprem/floor_masks.py --run RUN --sam3 SAM_DIR/sam3.json --out DIR [--word floor] [--min-score 0.5] [--min-near 0.6]
  python SERVING/scripts/research/prepare_capture_evidence.py floor --output RUN --spec DIR/floor-spec.json
  python scripts/onprem/floor_masks.py --self-test

Input: sam3.json of scripts/workcell_sam_worker.py (text prompt 'floor'; view i = source-i.jpg = RUN manifest frame i, every
instance with score >= 0.4, original-resolution RLE) and RUN's Pi3X geometry (pts3d.npy, content_valid_mask.npy, camera_to_world).
Rule (generic, the same for every site):
  1. candidates = instances with score >= --min-score (0.5);
  2. each candidate is lifted through its frame's canonical point map (canonical pixel -> nearest input pixel of the mask);
     the floor of all candidates' points = the LARGEST-consensus RANSAC plane within 20 deg of up with every camera above it
     (threshold 0.5 % of the points' p5-p95 extent; two SVD refits on its inliers). Not ehs_spatial's _ransac_floor_plane
     ('lowest adequately supported plane'): on cell 090 that rule tilted 1.5-4.3 deg with the subsample offset and dropped
     every true floor instance (research-notes/workcell-onprem-final-2026-10-05). The candidates are already 'floor' masks,
     so the furniture case that rule guards against is a minority here;
  3. an instance is kept when >= --min-near (0.6) of its lifted points lie within --near-factor (2) x threshold of that plane.
Writes DIR/<frame_id>.png (union of the kept instances, input resolution), DIR/floor-spec.json (prepare_capture_evidence.py floor
--spec) and DIR/floor-selection.json (every instance: score, lifted points, near fraction, kept or why not; the plane).
Freeze DIR with the capture's evidence and review it: the floor plane decides every object's bottom height
(prepare_capture_evidence.py floor still refits with the lowest-plane rule: see ONPREM.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROVENANCE = "SAM 3 text prompt '{word}' (facebook/sam3@3c879f3, scripts/workcell_sam_worker.py), instances kept by scripts/onprem/floor_masks.py"


def rle_decode(text: str) -> np.ndarray:
    """workcell_sam_worker.py RLE: column-major run lengths starting with a background run."""
    r = json.loads(text); h, w = r['size']
    flat = np.repeat(np.arange(len(r['counts'])) % 2 == 1, r['counts'])
    return flat.reshape((w, h)).T


def lift(mask: np.ndarray, points: np.ndarray, valid: np.ndarray, input_to_canonical) -> np.ndarray:
    """World points of the canonical pixels whose nearest input pixel is in the (input-resolution) mask."""
    h, w = valid.shape
    yy, xx = np.indices((h, w))
    src = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(np.asarray(input_to_canonical, float)).T
    u, v = np.floor(src[..., 0] + .5).astype(int), np.floor(src[..., 1] + .5).astype(int)
    inside = (u >= 0) & (v >= 0) & (u < mask.shape[1]) & (v < mask.shape[0]) & valid
    hit = np.zeros((h, w), bool)
    hit[inside] = mask[v[inside], u[inside]]
    return points[hit]


def floor_plane(points: np.ndarray, cameras: list, iterations=1000, max_tilt_deg=20.) -> tuple[np.ndarray, float, float]:
    """Largest-consensus plane (unit normal towards the cameras, offset, threshold): deterministic 3-point RANSAC on <= 20000
    points, candidates within max_tilt_deg of the cameras' up and below every camera, then two SVD refits on all points."""
    points = np.asarray(points, float)
    up = np.mean([-c[:3, 1] for c in cameras], axis=0); up /= np.linalg.norm(up)
    centres = np.asarray([c[:3, 3] for c in cameras], float)
    threshold = .005 * np.linalg.norm(np.quantile(points, .95, axis=0) - np.quantile(points, .05, axis=0))
    fit = points[::max(1, len(points) // 20000)]
    i = np.random.default_rng(0).integers(0, len(fit), (iterations, 3))
    a, b, c = fit[i[:, 0]], fit[i[:, 1]], fit[i[:, 2]]
    N = np.cross(b - a, c - a); L = np.linalg.norm(N, axis=1); ok = L > 1e-12; N, a = N[ok] / L[ok, None], a[ok]
    al = N @ up; keep = np.abs(al) >= np.cos(np.radians(max_tilt_deg)); N, a = N[keep] * np.sign(al[keep])[:, None], a[keep]
    D = -np.einsum('ij,ij->i', N, a)
    counts = np.concatenate([(np.abs(fit @ N[k:k + 128].T + D[k:k + 128]) < threshold).sum(0) for k in range(0, len(N), 128)]) if len(N) else np.zeros(0, int)
    counts[~((centres @ N.T + D) > 0).all(0)] = -1
    if not len(counts) or counts.max() < 150:
        raise ValueError('floor_masks: no supported floor plane among the candidate instances')
    normal, offset = N[int(np.argmax(counts))], float(D[int(np.argmax(counts))])
    for _ in range(2):
        inl = points[np.abs(points @ normal + offset) < threshold]
        center = inl.mean(0); normal = np.linalg.svd(inl - center, full_matrices=False)[2][-1]
        normal *= np.sign(normal @ up); offset = -float(normal @ center)
    return normal, offset, float(threshold)


def select(frames: list, min_score=.5, min_near=.6, near_factor=2., min_points=50) -> dict:
    """frames: [{frame_id, instances: [(mask, score)], points, valid, input_to_canonical, camera_to_world}] -> selection record."""
    rows = []
    for f in frames:
        for k, (mask, score) in enumerate(f['instances']):
            rows.append(dict(frame_id=f['frame_id'], instance=k, score=float(score), points=lift(mask, f['points'], f['valid'], f['input_to_canonical'])))
    candidates = [r for r in rows if r['score'] >= min_score and len(r['points']) >= min_points]
    if not candidates:
        raise ValueError('floor_masks: no instance passes the score threshold')
    normal, offset, threshold = floor_plane(np.concatenate([r['points'] for r in candidates]), [f['camera_to_world'] for f in frames])
    for r in rows:
        pts = r.pop('points'); r['liftedPoints'] = len(pts)
        r['nearFraction'] = float(np.mean(np.abs(pts @ normal + offset) < near_factor * threshold)) if len(pts) else 0.
        r['kept'] = r['score'] >= min_score and len(pts) >= min_points and r['nearFraction'] >= min_near
        r['why'] = ('floor' if r['kept'] else f'score < {min_score}' if r['score'] < min_score else
                    f'< {min_points} lifted points' if len(pts) < min_points else f'near fraction < {min_near}')
    return dict(rule=dict(minScore=min_score, minNearFraction=min_near, nearFactor=near_factor, minLiftedPoints=min_points,
                          fit='largest-consensus RANSAC plane on all candidates (<= 20 deg from up, cameras above), threshold '
                              '0.5 % of p5-p95 extent, two SVD refits'),
                plane=dict(normal=normal.tolist(), offset=offset, thresholdNative=threshold), instances=rows)


def run(run_dir: Path, sam3: Path, out: Path, word='floor', **rule):
    manifest = json.loads((run_dir / 'manifest.json').read_text()); sam = json.loads(sam3.read_text())
    w = [p['text'] for p in sam['prompts']].index(word)
    if len(sam['results']) != len(manifest['frames']):
        raise ValueError('sam3.json views do not match the manifest frames (view i = source-i.jpg = frame i)')
    frames = []
    for fr, view in zip(manifest['frames'], sam['results']):
        g = run_dir / 'geometry/frames' / fr['frame_id']
        masks = [rle_decode(t) for t in view[w]['rle']]
        if any(m.shape != (fr['height'], fr['width']) for m in masks):
            raise ValueError(f"{fr['frame_id']}: SAM 3 mask size differs from the frozen photo")
        frames.append(dict(frame_id=fr['frame_id'], instances=list(zip(masks, view[w]['scores'])), points=np.load(g / 'pts3d.npy'),
                           valid=np.load(g / 'content_valid_mask.npy').astype(bool), input_to_canonical=fr['input_to_canonical_pixel_centres'],
                           camera_to_world=np.load(g / 'camera_to_world.npy')))
    record = select(frames, **rule)
    out.mkdir(parents=True, exist_ok=False)
    views = []
    for f in frames:
        kept = [m for k, (m, _) in enumerate(f['instances']) if any(r['kept'] and r['frame_id'] == f['frame_id'] and r['instance'] == k for r in record['instances'])]
        if kept:
            Image.fromarray(np.any(kept, 0).astype(np.uint8) * 255).save(out / f"{f['frame_id']}.png")
            views.append(dict(frame_id=f['frame_id'], mask=str((out / f"{f['frame_id']}.png").resolve()), provenance=PROVENANCE.format(word=word)))
    (out / 'floor-spec.json').write_text(json.dumps({'views': views}, indent=1) + '\n')
    (out / 'floor-selection.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({'kept': [(r['frame_id'], r['instance'], r['score']) for r in record['instances'] if r['kept']],
                      'dropped': [(r['frame_id'], r['instance'], r['score'], r['why']) for r in record['instances'] if not r['kept']]}))
    return record


def _check():
    """Synthetic two-camera scene: floor (z=0), a wall, a box top; six instances, two of them floor with score >= 0.5."""
    rng = np.random.default_rng(0)
    n = 64; A = [[.5, 0, 0], [0, .5, 0], [0, 0, 1]]  # input 128x128 -> canonical 64x64
    c2w = lambda x: np.array([[1, 0, 0, x], [0, 0, 1, 0], [0, -1, 0, 1.5], [0, 0, 0, 1.]])  # OpenCV y down = world -z
    region = lambda r0, r1, c0, c1: (slice(r0, r1), slice(c0, c1))
    floor_a, floor_b, wall, box = region(40, 64, 0, 40), region(30, 40, 0, 64), region(0, 20, 0, 64), region(42, 60, 44, 62)
    frames = []
    for fid, x in (('frame_0001', 0.), ('frame_0002', 1.)):
        pts = np.zeros((n, n, 3)); pts[..., 0] = rng.uniform(-2, 2, (n, n)); pts[..., 1] = rng.uniform(1, 3, (n, n))
        pts[..., 2] = rng.normal(0, .002, (n, n))  # floor everywhere, then the wall and the box top
        sub = pts[wall]; sub[..., 1] = 3.; sub[..., 2] = rng.uniform(.2, 2, (20, n)); pts[wall] = sub
        sub = pts[box]; sub[..., 2] = .4; pts[box] = sub
        def mask(*regions):
            m = np.zeros((2 * n, 2 * n), bool)
            for r in regions:
                m[2 * r[0].start:2 * r[0].stop, 2 * r[1].start:2 * r[1].stop] = True
            return m
        mixed = mask(region(5, 35, 40, 64))  # 15 wall rows + 15 floor rows: half floor
        instances = [(mask(floor_a), .93), (mask(floor_b), .57), (mask(wall), .71), (mask(box), .62), (mask(floor_a), .45), (mixed, .8)]
        frames.append(dict(frame_id=fid, instances=instances if fid == 'frame_0001' else instances[:2], points=pts,
                           valid=np.ones((n, n), bool), input_to_canonical=A, camera_to_world=c2w(x)))
    # a 'floor' instance on a tilted plane that dips below the floor (min height -0.12): the lowest supported plane would be
    # it, and every true floor instance would then be dropped; the largest-consensus plane keeps the floor
    sliver = np.zeros((128, 128), bool); sliver[2 * 40:2 * 64, 2 * 40:2 * 64] = True
    pts = frames[1]['points'].copy(); sub = pts[40:64, 40:64]; sub[..., 2] = -.12 + .1 * (sub[..., 0] + 2) / 4; pts[40:64, 40:64] = sub
    frames[1] = dict(frames[1], points=pts, instances=frames[1]['instances'] + [(sliver, .7)])
    rec = select(frames)
    kept = {(r['frame_id'], r['instance']) for r in rec['instances'] if r['kept']}
    assert kept == {('frame_0001', 0), ('frame_0001', 1), ('frame_0002', 0), ('frame_0002', 1)}, rec['instances']
    assert next(r for r in rec['instances'] if r['frame_id'] == 'frame_0002' and r['instance'] == 2)['why'] == 'near fraction < 0.6'
    why = {r['instance']: r['why'] for r in rec['instances'] if r['frame_id'] == 'frame_0001'}
    assert why[2] == why[3] == why[5] == 'near fraction < 0.6' and why[4] == 'score < 0.5', why
    assert abs(rec['plane']['normal'][2] - 1) < 1e-3 and abs(rec['plane']['offset']) < .01, rec['plane']
    m = frames[0]['instances'][0][0]; flat = m.flatten(order='F').astype(np.int8)
    b = np.concatenate(([0], np.flatnonzero(np.diff(flat)) + 1, [flat.size])); counts = np.diff(b).tolist()
    if flat[0]:
        counts.insert(0, 0)
    assert np.array_equal(rle_decode(json.dumps({'size': list(m.shape), 'counts': counts})), m)
    print('floor_masks self-test passed: score rule, wall / box top / half-wall / tilted below-floor instance dropped by the',
          'largest-consensus floor, plane z=0, RLE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path); ap.add_argument('--sam3', type=Path); ap.add_argument('--out', type=Path)
    ap.add_argument('--word', default='floor'); ap.add_argument('--min-score', type=float, default=.5)
    ap.add_argument('--min-near', type=float, default=.6); ap.add_argument('--near-factor', type=float, default=2.)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _check()
    else:
        run(a.run, a.sam3, a.out, a.word, min_score=a.min_score, min_near=a.min_near, near_factor=a.near_factor)
