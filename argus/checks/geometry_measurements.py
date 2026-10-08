"""Geometry backbone A/B (Pi3X vs licence-clean candidates), every number by the code the published reports use.

Per cell and backbone, from a run-dir-like geometry folder (frames/frame_000k/{pts3d,conf,valid_mask,intrinsics,camera_to_world}.npy
on the frozen 518 canonical grid):
  floor   = prepare_capture_evidence.floor(): ehs_spatial.geometry._ransac_floor_plane + SVD on the frozen canonical floor masks
            (evidence/objects/observed_floor), content rule of prepare_capture_evidence.geometry().
  e-stop  = estop_cylinder.fit + reference_object_scale.joint_scale, red lip 4 cm + yellow body 8 cm (estop_all.py): 090 = three
            photos, P3 triangulated from the axis seeds (estop_sept2.py); 030 = photo 2, P3 = point-map median in the seed box.
  heights = camera centres above that floor, times that scale.
  clearance = clearance_b.run(ctx) with the report's image-space masks and photos, and the backbone's cameras, point maps, floor
            and scale. clearance_b's thresholds are in native units of the published reports (~1.1-1.3 m); the candidate's world
            is first rescaled by S_cand / S_pi3x so they mean the same metric distances (floor fit and e-stop are scale-equivariant).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

FIELD = dict(post=.24, panel=.20)  # user, both cells: light-curtain housing bottom, fence panel bottom (m)
SPEC = (('red lip 4 cm', 'red lip', .04), ('yellow body 8 cm', 'yellow body', .08))  # estop_all.py features


# ------------------------------------------------------------------ geometry folder
def content_mask(pts, valid, conf, alpha):
    """prepare_capture_evidence.RULE: native valid & declared alpha & finite & nonzero points & conf>=0.1."""
    return valid & alpha & np.isfinite(pts).all(-1) & (np.linalg.norm(pts, axis=-1) > 1e-6) & (conf >= .1)


def rescaled(frames, f):
    """The same world in units f times smaller (points and camera centres times f); rotations, K and masks unchanged."""
    out = []
    for fr in frames:
        M = fr['c2w'].copy(); M[:3, 3] *= f
        out.append({**fr, 'pts3d': fr['pts3d'] * np.float32(f), 'c2w': M})
    return out


def doc_cameras(frames, image_ids, sizes, A):
    """Report-style cameras at original pixels: K_input = inv(input_to_canonical) K_canonical (as the platform import)."""
    return [dict(imageId=i, width=w, height=h, K=(np.linalg.inv(a) @ fr['K']).tolist(), cameraToWorld=fr['c2w'].tolist())
            for fr, i, (w, h), a in zip(frames, image_ids, sizes, A)]


# ------------------------------------------------------------------ floor (prepare_capture_evidence.floor)
def floor_fit(frames, masks, ransac):
    points = np.concatenate([fr['pts3d'][fr['content'] & m].astype(np.float32) for fr, m in zip(frames, masks) if m is not None])  # frames without a frozen mask: cameras only
    cams = [fr['c2w'] for fr in frames]
    up = np.mean([-c[:3, 1] for c in cams], axis=0); up /= np.linalg.norm(up)
    threshold = .005 * np.linalg.norm(np.quantile(points, .95, axis=0) - np.quantile(points, .05, axis=0))
    fit = points[::max(1, len(points) // 20000)]
    idx = ransac(fit, np.asarray([c[:3, 3] for c in cams]), up, threshold)
    if idx is None or len(idx) < 150:
        raise ValueError('Insufficient observed floor plane support')
    centre = np.mean(fit[idx], axis=0); n = np.linalg.svd(fit[idx] - centre, full_matrices=False)[2][-1]
    n *= np.sign(n @ up); d = -float(n @ centre)
    res = np.abs(points @ n + d); inside = res < threshold
    return dict(normal=n.astype(float), offset=d, plane=[*map(float, n), d], thresholdNative=float(threshold), points=len(points),
                inliers=int(inside.sum()), residualP95Native=float(np.quantile(res, .95)),
                cameraHeightsNative=[float(c[:3, 3] @ n + d) for c in cams])


# ------------------------------------------------------------------ e-stop (estop_all.py / estop_sept2.py)
def scaled_K(K, sx, sy):  # pixel-centre convention: x' = (x + .5) s - .5
    K = np.array(K, float); return np.array([[K[0, 0] * sx, 0, (K[0, 2] + .5) * sx - .5], [0, K[1, 1] * sy, (K[1, 2] + .5) * sy - .5], [0, 0, 1]])


def triangulate_axis(views):
    """estop_sept2.py: least-squares point closest to the rays through each photo's axis seed."""
    Aq, bq = [], []
    for v in views:
        M = v['M']; d = M[:3, :3] @ (np.linalg.inv(v['K']) @ [v['seed'][0], v['seed'][1], 1]); d /= np.linalg.norm(d)
        Pm = np.eye(3) - np.outer(d, d); Aq.append(Pm); bq.append(Pm @ M[:3, 3])
    return np.linalg.solve(np.sum(Aq, 0), np.sum(bq, 0))


def pointmap_point(fr, A, box):
    """estop_all.py (030): median point-map point over the seed box (original pixels, 4 / 6 px steps) where content is valid."""
    pts = []
    for y in range(*box[1], 4):
        for x in range(*box[0], 6):
            u, v, _ = A @ [x, y, 1]; ui, vi = int(round(u)), int(round(v))
            if fr['content'][vi, ui]:
                pts.append(fr['pts3d'][vi, ui])
    return np.median(pts, axis=0)


def estop_scale(views, P3, up, fit, joint_scale):
    rows, per = [], {}
    for v in views:
        M = v['M']; res = fit(v['image'], v['K'], M, up, P3, v['seed'], v['top'], v['bot'])
        z = float((M[:3, :3].T @ (P3 - M[:3, 3]))[2]); fx = float(v['K'][0, 0]); w = {k: r['symmetricWidthPx'] for k, r in res['parts'].items()}
        per[v['name']] = dict(widthsPx=w, depthNative=z, fx=fx, yellowOverRed=w['yellow body'] / w['red lip'], axisTiltDeg=res['vanishingTiltDeg'],
                              blurPx={k: r['blurPx'] for k, r in res['parts'].items()}, edgeSources={k: r['edgeSources'] for k, r in res['parts'].items()})
        rows += [dict(photo=v['name'], feature=f, specM=s, measuredNative=w[part] * z / fx) for f, part, s in SPEC]
    return dict(P3=P3.tolist(), perPhoto=per, **joint_scale(rows))


# ------------------------------------------------------------------ clearance context (shape_check.load_report, local data)
def clearance_ctx(view, layer, cams_doc, photos, floor, S, frames, wsc):
    """load_report without the network: same cameras/photos/masks/objects/labels/floor orientation; models not loaded
    (clearance_b uses them only for its 'modelPlane' variant)."""
    doc = view['publication']['snapshot']['revision']['document']
    if layer and layer.get('revisionId') not in (None, view['publication']['sceneRevisionId']):
        layer = None
    cams = [wsc.camera(c) for c in cams_doc]
    gray, rgb = [], []
    for c in cams:
        buf = np.frombuffer(photos[c['imageId']], np.uint8)  # decoded exactly as load_report (gray) and the clearance app (rgb)
        g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        assert g.shape == (c['h'], c['w']), (g.shape, c['h'], c['w'])
        gray.append(g); rgb.append(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32))
    index = {c['imageId']: k for k, c in enumerate(cams)}
    observations = {o['id']: o for o in doc['observations']}
    objects = []
    for e in doc['entities']:
        if e.get('sourceContext') or e.get('visible') is False or not e.get('activeModelRepresentationId'):
            continue
        rep = next(r for r in e['representations'] if r['id'] == e['activeModelRepresentationId'])
        if rep['kind'] not in ('generated_mesh', 'primitive'):
            continue
        masks = {}
        for oid in e.get('observationRefs') or []:
            o = observations.get(oid)
            if o and o['imageId'] in index and o.get('originalPixelPolygons'):
                k = index[o['imageId']]; m = wsc.polygon_mask(o['originalPixelPolygons'], (cams[k]['h'], cams[k]['w']))
                masks[k] = masks[k] | m if k in masks else m
        objects.append(dict(id=e['id'], label=((layer or {}).get('labels') or {}).get(e['id'], e.get('label')), kind=rep['kind'],
                            mesh=None, original=None, masks=masks))
    n, d = np.asarray(floor[0], float), float(floor[1]); k = np.linalg.norm(n); n, d = n / k, d / k
    if np.median([n @ c['C'] + d for c in cams]) < 0:
        n, d = -n, -d
    ctx = dict(doc=doc, layer=layer, cams=cams, gray=gray, images=[cv2.GaussianBlur(g, (0, 0), 1.0) for g in gray], objects=objects,
               skipped=[], S=float(S), floor=(n, d), index=index, rgb=rgb)
    ctx['pi3x'] = {k: dict(pts3d=fr['pts3d'], conf=fr['conf'], valid=fr['valid'], content=fr['content']) for k, fr in enumerate(frames)}
    return ctx


# ------------------------------------------------------------------ poses
def rot_angle(R):
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def pose_summary(cams_doc, floor, S):
    """Per camera: height (m), pitch below the horizon and roll (deg) w.r.t. its own floor, fx at original pixels; per pair:
    baseline (m) and relative rotation."""
    n = np.asarray(floor[0], float); d = float(floor[1]); out = dict(cameras=[], pairs={})
    Ms = [np.array(c['cameraToWorld'], float) for c in cams_doc]
    for c, M in zip(cams_doc, Ms):
        fwd, right = M[:3, 2], M[:3, 0]
        out['cameras'].append(dict(heightM=float((n @ M[:3, 3] + d) * S), pitchDownDeg=float(-math.degrees(math.asin(np.clip(fwd @ n, -1, 1)))),
                                   rollDeg=float(math.degrees(math.asin(np.clip(right @ n, -1, 1)))), fxPx=float(c['K'][0][0])))
    for i in range(len(Ms)):
        for j in range(i + 1, len(Ms)):
            Rij = Ms[i][:3, :3].T @ Ms[j][:3, :3]; tij = Ms[i][:3, :3].T @ (Ms[j][:3, 3] - Ms[i][:3, 3])
            out['pairs'][f'{i + 1}-{j + 1}'] = dict(baselineM=float(np.linalg.norm(tij) * S), relRotationDeg=rot_angle(Rij),
                                                     R=Rij.tolist(), tDir=(tij / np.linalg.norm(tij)).tolist())
    return out


def pose_agreement(a, b):
    """Candidate b vs reference a: relative-rotation error, baseline-direction angle (in camera i), baseline ratio."""
    out = {}
    for key, pa in a['pairs'].items():
        pb = b['pairs'][key]
        out[key] = dict(relRotationErrDeg=rot_angle(np.array(pa['R']).T @ np.array(pb['R'])),
                        baselineDirErrDeg=math.degrees(math.acos(np.clip(np.dot(pa['tDir'], pb['tDir']), -1, 1))),
                        baselineRatio=pb['baselineM'] / pa['baselineM'])
    return out


# ------------------------------------------------------------------ self-test
def _check():
    rng = np.random.default_rng(0)
    # floor fit on a synthetic tilted floor seen from two cameras is exact and scale-equivariant
    n = np.array([.1, -.2, .97]); n /= np.linalg.norm(n); d = .3
    e1 = np.cross(n, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    P = (-d * n + rng.uniform(-1, 1, (4000, 1)) * e1 + rng.uniform(-1, 1, (4000, 1)) * e2).astype(np.float32)
    def cam(C):
        z = -n + .3 * e1; z /= np.linalg.norm(z); x = np.cross(z, n); x /= np.linalg.norm(x); y = np.cross(z, x)
        M = np.eye(4); M[:3, :3] = np.c_[x, y, z]; M[:3, 3] = C; return M
    def ransac(points, centres, up, thr):  # stand-in for _ransac_floor_plane (exact plane): all inliers
        return np.flatnonzero(np.abs(points @ n + d) < thr)
    frames = [dict(pts3d=P.reshape(40, 100, 3), content=np.ones((40, 100), bool), c2w=cam(-d * n + 1.5 * n + .2 * e1 * s), K=np.eye(3)) for s in (-1, 1)]
    masks = [np.ones((40, 100), bool)] * 2
    f0 = floor_fit(frames, masks, ransac)
    assert np.allclose(f0['normal'], n, atol=1e-5) and abs(f0['offset'] - d) < 1e-5 and np.allclose(f0['cameraHeightsNative'], 1.5, atol=1e-5)
    def ransac2(points, centres, up, thr):
        return np.flatnonzero(np.abs(points @ n + 2 * d) < thr)
    f2 = floor_fit(rescaled(frames, 2.), masks, ransac2)
    assert abs(f2['offset'] - 2 * d) < 1e-4 and np.allclose(f2['cameraHeightsNative'], 3., atol=1e-4)
    # pose summary: height x S, pitch of a camera looking 30 deg down, identical poses agree exactly
    docs = [dict(K=[[1000, 0, 0], [0, 1000, 0], [0, 0, 1]], cameraToWorld=fr['c2w'].tolist()) for fr in frames]
    ps = pose_summary(docs, (n, d), 2.)
    assert abs(ps['cameras'][0]['heightM'] - 3.) < 1e-6 and abs(ps['cameras'][0]['rollDeg']) < 1e-6, ps['cameras'][0]
    assert abs(ps['cameras'][0]['pitchDownDeg'] - math.degrees(math.atan(1 / .3))) < 1e-6
    ag = pose_agreement(ps, ps)['1-2']; assert ag['relRotationErrDeg'] < 1e-6 and abs(ag['baselineRatio'] - 1) < 1e-12
    # triangulation of rays through one point
    X = np.array([.2, .1, 2.]); views = []
    for C in ([0, 0, 0], [1, 0, 0], [0, 1, .5]):
        M = np.eye(4); M[:3, 3] = C; K = np.array([[800., 0, 320], [0, 800, 240], [0, 0, 1]]); x = K @ (X - C)
        views.append(dict(M=M, K=K, seed=x[:2] / x[2]))
    assert np.allclose(triangulate_axis(views), X, atol=1e-9)
    print('geometry_ab self-test passed')


if __name__ == '__main__':
    _check()
