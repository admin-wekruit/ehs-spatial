"""Shape check for every model in a four-view report.

A generated model can match every photo's outline and still sit at the wrong depth: the outline only fixes the directions of its
edges from each camera. This check asks the photos themselves. Points sampled on each model surface are projected into all
photos that see them (masks where the report has them, other models and the model itself as occluders); if the surface is
where the model says, the photos agree about what is there (normalised correlation of the sampled intensities).

The model is then scaled about its reference camera (the photo with the largest mask), which leaves that photo's outline
unchanged and moves the model along that camera's rays. The scale at which the photos agree best says how far the model
would have to move: the usual failure of single-view placement. A shifted-sampling control says whether the surface has
enough texture to tell.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_packed(data: bytes, layout: dict):
    vc, ic = layout['vertexCount'], layout['indexCount']
    v = np.frombuffer(data, np.float32, vc * 9, layout.get('byteOffset', 0)).reshape(vc, 9)
    f = np.frombuffer(data, np.uint32, ic, layout['indexByteOffset']).reshape(-1, 3)
    return v[:, :3].astype(np.float64), f.astype(np.int64)


def primitive_mesh(spec: dict):
    """Box and cylinder as the viewer builds them (centred, cylinder axis = local z)."""
    p, kind = spec.get('parameters') or spec, spec.get('kind') or spec.get('primitiveType') or spec.get('type')
    if kind == 'box':
        d = np.array(p['dimensions'], float) / 2
        V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * d
        F = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
        return V, F
    if kind == 'cylinder':
        r, h, n = p['radius'], p['height'], int(p.get('segments') or 64)
        a = np.arange(n) * 2 * math.pi / n
        ring = np.c_[r * np.cos(a), r * np.sin(a)]
        V = np.vstack([np.c_[ring, np.full(n, -h / 2)], np.c_[ring, np.full(n, h / 2)], [[0, 0, -h / 2], [0, 0, h / 2]]])
        F = [[i, (i + 1) % n, n + (i + 1) % n] for i in range(n)] + [[i, n + (i + 1) % n, n + i] for i in range(n)]
        F += [[2 * n, (i + 1) % n, i] for i in range(n)] + [[2 * n + 1, n + i, n + (i + 1) % n] for i in range(n)]
        return V, np.array(F)
    raise ValueError(f'unsupported primitive {kind}')


def placed(V, transform):
    R, s, p = quat_matrix(transform['quaternion']), np.array(transform['scale'], float), np.array(transform['position'], float)
    return (V * s) @ R.T + p


def sample_surface(V, F, n, rng):
    tri = V[F]; cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); area = np.linalg.norm(cross, axis=1)
    keep = area > 0
    tri, cross, area = tri[keep], cross[keep], area[keep]
    pick = rng.choice(len(tri), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n); flip = u + v > 1; u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[pick]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0]), cross[pick] / area[pick, None]


def camera(c):
    M = np.array(c['cameraToWorld'], float); R = M[:3, :3].T; C = M[:3, 3]
    return dict(K=np.array(c['K'], float), R=R, t=-R @ C, C=C, w=int(c['width']), h=int(c['height']), imageId=c['imageId'])


def project(cam, P):
    x = (cam['K'] @ (cam['R'] @ P.T + cam['t'][:, None])).T
    return x[:, :2] / x[:, 2:], x[:, 2]


def polygon_mask(polygons, shape):
    """Even-odd fill of pixel-centre polygons."""
    m = np.zeros(shape, np.uint8)
    for poly in polygons:
        one = np.zeros(shape, np.uint8)
        cv2.fillPoly(one, [np.round(np.array(poly)).astype(np.int32)], 1)
        m ^= one
    return m.astype(bool)


def bilinear(img, uv):
    """cv2.remap wants maps under SHRT_MAX per side: rows of 1024."""
    n = len(uv); m = -(-n // 1024) * 1024 or 1024
    mx, my = np.zeros(m, np.float32), np.zeros(m, np.float32); mx[:n], my[:n] = uv[:, 0], uv[:, 1]
    return cv2.remap(img, mx.reshape(-1, 1024), my.reshape(-1, 1024), cv2.INTER_LINEAR).ravel()[:n]


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean(); d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def raycast_scene(meshes):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    for V, F in meshes:
        if len(F):
            scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    return scene


def first_hit(scene, origins, targets):
    """Ray parameter of the first hit with target at t = 1."""
    import open3d as o3d
    rays = np.hstack([origins, targets - origins]).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()


def check_entity(entity_id, P, N, own, others, cams, images, masks, scales, native_eps, min_pairs=150, shift=8.0):
    """NCC between photos for surface points P (normals N) of one model, as the model is scaled about its reference camera."""
    area = {k: int(m.sum()) for k, m in masks.items()}
    ref = max(area, key=area.get) if area else int(np.argmax([((project(c, P)[1] > 0)).sum() for c in cams]))
    Cr = cams[ref]['C']; rows = []
    for s in scales:
        Ps = Cr + s * (P - Cr); vals = []
        for k, cam in enumerate(cams):
            uv, z = project(cam, Ps)
            ok = (z > 0) & (uv[:, 0] >= 1) & (uv[:, 1] >= 1) & (uv[:, 0] < cam['w'] - 2) & (uv[:, 1] < cam['h'] - 2)
            view = cam['C'][None] - Ps; dist = np.linalg.norm(view, axis=1)
            ok &= np.abs(np.einsum('ij,ij->i', N, view)) / dist > .1
            if k in masks:
                ui = np.clip(uv.astype(int), 0, [cam['w'] - 1, cam['h'] - 1]); ok &= masks[k][ui[:, 1], ui[:, 0]]
            idx = np.nonzero(ok)[0]
            if len(idx):
                eps = native_eps / dist[idx]
                # self-occlusion of the scaled model == the unscaled model seen from the camera mapped back by 1/s
                Ck = Cr + (cam['C'] - Cr) / s
                t_own = first_hit(own, np.repeat(Ck[None], len(idx), 0), P[idx])
                t_oth = first_hit(others, np.repeat(cam['C'][None], len(idx), 0), Ps[idx]) if others is not None else np.full(len(idx), np.inf)
                vis = (t_own >= 1 - eps) & (t_oth >= 1 - eps)
                ok[idx[~vis]] = False
            v = np.full(len(P), np.nan); v[ok] = bilinear(images[k], uv[ok]); vals.append(v)
        pairs = []
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                both = ~np.isnan(vals[i]) & ~np.isnan(vals[j])
                if both.sum() >= min_pairs:
                    pairs.append((i, j, int(both.sum()), ncc(vals[i][both], vals[j][both])))
        score = sum(n * c for *_, n, c in pairs) / sum(n for *_, n, _ in pairs) if pairs else float('nan')
        rows.append(dict(scale=float(s), ncc=score, pairs=pairs))
    at1 = min(rows, key=lambda r: abs(r['scale'] - 1)); valid = [r for r in rows if not math.isnan(r['ncc'])]
    best = max(valid, key=lambda r: r['ncc']) if valid else at1
    # control: the same points with the reference photo sampled a few pixels off
    control = float('nan')
    if at1['pairs']:
        cam = cams[ref]; uv, _ = project(cam, P); shifted = bilinear(images[ref], uv + shift)
        cs, ns = [], []
        for k, c2 in enumerate(cams):
            if k == ref:
                continue
            uv2, z2 = project(c2, P); inside = (z2 > 0) & (uv2[:, 0] >= 1) & (uv2[:, 1] >= 1) & (uv2[:, 0] < c2['w'] - 2) & (uv2[:, 1] < c2['h'] - 2)
            if k in masks:
                ui = np.clip(uv2.astype(int), 0, [c2['w'] - 1, c2['h'] - 1]); inside &= masks[k][ui[:, 1], ui[:, 0]]
            if ref in masks:
                ui = np.clip(uv.astype(int), 0, [cam['w'] - 1, cam['h'] - 1]); inside &= masks[ref][ui[:, 1], ui[:, 0]]
            if inside.sum() >= min_pairs:
                cs.append(ncc(shifted[inside], bilinear(images[k], uv2[inside]))); ns.append(int(inside.sum()))
        if cs:
            control = float(np.average(cs, weights=ns))
    return dict(entityId=entity_id, referencePhoto=ref, distanceToReference=float(np.median(np.linalg.norm(P - Cr, axis=1))),
                nccAtModel=at1['ncc'], pairsAtModel=at1['pairs'], bestScale=best['scale'], nccAtBest=best['ncc'], pairsAtBest=best['pairs'],
                shiftedControl=control, curve=[(r['scale'], r['ncc']) for r in rows])


def verdict(row, scale_tolerance=.03):
    """ok / depth_off / ambiguous (another depth fits about as well) / inconclusive (texture cannot localise depth).

    Judged on the agreement-versus-depth curve: a usable surface has one clear peak well above the curve's floor; wire grids
    and see-through panels give flat or many-peaked curves. Rises at the ends of the scanned range are not peaks."""
    curve = [(s, c) for s, c in row['curve'] if not math.isnan(c)]
    if len(curve) < 3 or math.isnan(row['nccAtBest']):
        return 'inconclusive'
    best, peak = row['bestScale'], row['nccAtBest']
    row['curveContrast'] = contrast = peak - min(c for _, c in curve)
    local = [c for (s0, c0), (s, c), (s1, c1) in zip(curve, curve[1:], curve[2:]) if c >= c0 and c >= c1 and abs(s - best) > .04]
    row['secondPeak'] = second = max(local, default=float('nan'))
    if peak < .3 or contrast < .25:
        return 'inconclusive'
    if local and second > peak - .1:
        return 'ambiguous'
    if abs(best - 1) > scale_tolerance and peak - row['nccAtModel'] > .1:
        return 'depth_off'
    return 'ok'


def _check():
    """Synthetic check: a textured plane seen by three cameras; a copy placed 10 % too far must come back at scale ~0.91."""
    rng = np.random.default_rng(0)
    K = np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1.]])
    def cam_at(x):
        C = np.array([x, 0, 0.]); M = np.eye(4); M[:3, 3] = C
        return camera(dict(cameraToWorld=M.tolist(), K=K.tolist(), width=800, height=600, imageId=str(x)))
    cams = [cam_at(x) for x in (-.3, 0, .3)]
    tex = cv2.GaussianBlur((rng.random((400, 400)) * 255).astype(np.float32), (0, 0), 2)
    plane = lambda Q: bilinear(tex, np.c_[(Q[:, 0] + 1) * 200, (Q[:, 1] + 1) * 200])
    images = []
    for c in cams:  # render the plane z = 3 by inverse mapping each pixel
        ys, xs = np.mgrid[0:600, 0:800]; rays = (np.linalg.inv(K) @ np.vstack([xs.ravel(), ys.ravel(), np.ones(xs.size)])).T
        Q = c['C'] + rays * (3 / rays[:, 2:]); images.append(plane(Q).reshape(600, 800).astype(np.float32))
    V = np.array([[-.8, -.6, 3], [.8, -.6, 3], [.8, .6, 3], [-.8, .6, 3]]); F = np.array([[0, 1, 2], [0, 2, 3]])
    Vfar = cams[1]['C'] + 1.1 * (V - cams[1]['C'])
    P, N = sample_surface(Vfar, F, 4000, rng)
    mask = np.ones((600, 800), bool)
    row = check_entity('plane', P, N, raycast_scene([(Vfar, F)]), None, cams, images, {1: mask}, np.linspace(.8, 1.2, 41), 1e-3)
    assert abs(row['bestScale'] - 1 / 1.1) < .015, row['bestScale']
    assert row['nccAtBest'] > .8 and row['nccAtBest'] - row['nccAtModel'] > .3, row
    assert verdict(row) == 'depth_off'
    print('shape check self-test passed: best scale', row['bestScale'], 'ncc', round(row['nccAtModel'], 2), '->', round(row['nccAtBest'], 2))


if __name__ == '__main__':
    _check()
